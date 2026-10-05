#!/usr/bin/env python3
"""Estimate how long a protocol takes on the DGX, from timing runs (review-plan/plans/week-budget.md, B7).

    .venv/bin/python scripts/project-time.py --timing ml20m=TIMING_WORK [--timing amazon=WORK ...]
        [--protocol protocol.toml] [--factor otto=3] [--speedup 1.7] [--refit-share 0.6]

A timing run is `scripts/local-run.sh` with a short protocol: TRIALS trials of EPOCHS epochs each. Its trials
draw the same configurations as the real search's first trials, with the epochs changed. Each run records its
`fit_seconds` and `eval_seconds` in `done.json`.

The estimate, per dataset and trained model (the GPU models: those trained in `epochs`):
- the cost of one epoch of each configuration: measured where that configuration was timed on the reference
  dataset (the one with the most timed trials), otherwise a log-linear fit to the timed ones over the
  configuration's sizes (batch, widths, depth, history length, negatives);
- another dataset: the reference cost times that dataset's measured ratio on the configurations timed on both;
  `--factor DATASET=X` supplies the ratio for a dataset with no timing run; any other is left out, and said so;
- search: the protocol's planned trials, epochs x cost per epoch, plus each trial's validation scoring;
- finals: seeds x the selected configuration, priced at the mean of the planned configurations (and, for the
  worst case, the most expensive) times the measured final/trial ratio;
- ablations: per retraining level and sweep seed, one final's fit times `--refit-share`, the share of training
  data a level keeps, on average; each rescoring (inference sweeps, references) one scoring.
EASE and popularity run on the CPU beside the GPU and are left out of the GPU total.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from seqrec_eval.ablations import scope_of  # noqa: E402
from seqrec_eval.protocol import load_protocol  # noqa: E402
from seqrec_eval.runner import plan_trials  # noqa: E402

#: configuration sizes that set the cost of an epoch; rates, dropouts and the like do not
SIZES = ("batch_size", "latent_dim", "hidden_dim", "embedding_dim", "num_layers", "max_history_length", "d_model",
         "n_blocks", "n_heads", "n_negatives")
DEFAULT_FINAL_RATIO = 1.5


def _key(params: dict) -> tuple:
    return tuple((name, params[name]) for name in SIZES if name in params)


def read_timing(work: Path, dataset: str) -> dict[str, dict]:
    """model -> {"trials": [(params, seconds per epoch, eval seconds)], "finals": [(params, fit, eval)]}."""
    out: dict[str, dict] = {}
    for done in sorted(Path(work).glob(f"runs/{dataset}/*/*/*/done.json")):
        record = json.loads(done.read_text())
        if record.get("status") != "done" or "fit_seconds" not in record:
            continue
        spec = json.loads((done.parent / "spec.json").read_text())
        params, model = spec["params"], spec["model"]
        entry = out.setdefault(model, {"trials": [], "finals": []})
        if spec["kind"] == "trial":
            epochs = params.get("epochs") or 1
            entry["trials"].append((params, record["fit_seconds"] / epochs, record.get("eval_seconds", 0.0)))
        elif spec["kind"] == "final":
            entry["finals"].append((params, record["fit_seconds"], record.get("eval_seconds", 0.0)))
    return out


class EpochCost:
    """Seconds per epoch of a configuration of one model, from its timed trials on the reference dataset."""

    def __init__(self, trials: list[tuple[dict, float, float]]):
        if not trials:
            raise ValueError("no timed trials")
        self.timed = {_key(params): seconds for params, seconds, _ in trials}
        self.eval = float(np.mean([e for _, _, e in trials]))
        varying = [n for n in SIZES if len({p.get(n) for p, _, _ in trials}) > 1
                   and all(isinstance(p.get(n), (int, float)) and p.get(n) > 0 for p, _, _ in trials)]
        self.features = varying
        x = np.array([[1.0] + [math.log(p[n]) for n in varying] for p, _, _ in trials])
        y = np.log([s for _, s, _ in trials])
        self.coef, *_ = np.linalg.lstsq(x, y, rcond=1e-6)
        fitted = np.exp(x @ self.coef)
        self.error = float(np.max(np.abs(fitted / np.exp(y) - 1.0)))  # worst relative error on the timed ones

    def __call__(self, params: dict) -> tuple[float, bool]:
        """Seconds per epoch, and whether this configuration was timed itself."""
        seconds = self.timed.get(_key(params))
        if seconds is not None:
            return seconds, True
        x = np.array([1.0] + [math.log(params[n]) for n in self.features])
        return float(np.exp(x @ self.coef)), False


def final_ratio(cost: EpochCost, finals: list[tuple[dict, float, float]]) -> float:
    """A final's fit (train+validation) against a trial of the same configuration, measured, or a default."""
    ratios = [fit / (cost(params)[0] * (params.get("epochs") or 1)) for params, fit, _ in finals]
    return float(np.mean(ratios)) if ratios else DEFAULT_FINAL_RATIO


def dataset_factor(reference: dict, other: dict, model: str) -> float | None:
    """The other dataset's cost per epoch against the reference's, on the configurations timed on both."""
    ref = {_key(p): s for p, s, _ in reference.get(model, {}).get("trials", [])}
    common = [s / ref[_key(p)] for p, s, _ in other.get(model, {}).get("trials", []) if _key(p) in ref]
    return float(np.exp(np.mean(np.log(common)))) if common else None


def project(protocol, timing: dict[str, dict], *, factors: dict[str, float], refit_share: float) -> dict:
    """Seconds per dataset and GPU model: search, finals, refits, rescorings, each at the mean selected
    configuration, and the finals and refits at the worst."""
    reference = max(timing, key=lambda d: sum(len(m["trials"]) for m in timing[d].values()))
    out = {"reference": reference, "rows": [], "skipped": [], "fit_error": {}}
    first = next(iter(protocol.datasets))
    # trained in epochs, on the GPU: `epochs` searched or fixed (a trial's params hold both)
    gpu_models = [name for name in protocol.models if "epochs" in plan_trials(protocol, first, name)[0].params]
    costs = {}
    for model in gpu_models:
        trials = timing[reference].get(model, {}).get("trials", [])
        if trials:
            costs[model] = EpochCost(trials)
            out["fit_error"][model] = costs[model].error
    for dataset in protocol.datasets:
        for model in gpu_models:
            if model not in costs:
                out["skipped"].append(f"{dataset}/{model}: not timed on {reference}")
                continue
            if dataset == reference:
                factor, source = 1.0, "timed"
            elif dataset in timing and dataset_factor(timing[reference], timing[dataset], model) is not None:
                factor, source = dataset_factor(timing[reference], timing[dataset], model), "ratio timed"
            elif dataset in factors:
                factor, source = factors[dataset], "factor given"
            else:
                out["skipped"].append(f"{dataset}/{model}: no timing run and no --factor")
                continue
            cost = costs[model]
            ratio = final_ratio(cost, timing[reference][model]["finals"])
            fits = []
            for spec in plan_trials(protocol, dataset, model):
                seconds, _ = cost(spec.params)
                fits.append(spec.params["epochs"] * seconds * factor)
            evaluation = cost.eval * factor
            search = sum(fits) + evaluation * len(fits)
            mean_final, worst_final = float(np.mean(fits)) * ratio, max(fits) * ratio
            seeds = len(protocol.seeds)
            refits = rescorings = 0
            for name, ablation in protocol.ablations.items():
                if dataset not in ablation.datasets or model not in ablation.models:
                    continue
                n = len(ablation.levels) * len(ablation.seeds)
                rescorings += len(ablation.seeds)  # each seed's reference rescores the stage-1 model
                if scope_of(ablation) == "inference":
                    rescorings += n
                else:
                    refits += n
            out["rows"].append({
                "dataset": dataset, "model": model, "source": source, "factor": factor,
                "search": search,
                "finals": seeds * (mean_final + evaluation),
                "refits": refits * (mean_final * refit_share + evaluation),
                "rescorings": rescorings * evaluation,
                "worst_extra": (seeds + refits * refit_share) * (worst_final - mean_final),
                "n_refits": refits, "n_rescorings": rescorings,
            })
    return out


def _parse_pairs(values: list[str], what: str) -> dict[str, str]:
    pairs = {}
    for value in values:
        name, sep, rest = value.partition("=")
        if not sep or not name or not rest:
            raise SystemExit(f"{what} must be DATASET=VALUE, got {value!r}")
        pairs[name] = rest
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--protocol", default=str(HERE / "protocol.toml"))
    parser.add_argument("--timing", action="append", default=[], metavar="DATASET=WORK_DIR", required=True,
                        help="a timing run's work dir for a dataset; the one with the most trials is the reference")
    parser.add_argument("--factor", action="append", default=[], metavar="DATASET=X",
                        help="a dataset without a timing run costs X times the reference")
    parser.add_argument("--speedup", type=float, default=1.0,
                        help="throughput of the processes run at once against one (measure it: plans, A2)")
    parser.add_argument("--refit-share", type=float, default=0.6,
                        help="average share of training data a retraining level keeps (default 0.6)")
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    timing = {dataset: read_timing(Path(work), dataset)
              for dataset, work in _parse_pairs(args.timing, "--timing").items()}
    factors = {dataset: float(x) for dataset, x in _parse_pairs(args.factor, "--factor").items()}
    result = project(protocol, timing, factors=factors, refit_share=args.refit_share)

    print(f"protocol {args.protocol}; reference dataset {result['reference']}; worst error of the cost fit on the "
          "timed configurations: " + ", ".join(f"{m} {e:.0%}" for m, e in result["fit_error"].items()))
    print(f"\n{'dataset':10} {'model':8} {'cost':>6} {'source':12} {'search':>8} {'finals':>8} {'refits':>14} "
          f"{'rescorings':>14} {'total':>8}   (GPU hours)")
    total = worst = 0.0
    for row in result["rows"]:
        hours = (row["search"] + row["finals"] + row["refits"] + row["rescorings"]) / 3600
        total += hours
        worst += hours + row["worst_extra"] / 3600
        print(f"{row['dataset']:10} {row['model']:8} {row['factor']:5.2f}x {row['source']:12} "
              f"{row['search'] / 3600:8.1f} {row['finals'] / 3600:8.1f} "
              f"{row['refits'] / 3600:8.1f} ({row['n_refits']:3}) {row['rescorings'] / 3600:8.1f} "
              f"({row['n_rescorings']:3}) {hours:8.1f}")
    for line in result["skipped"]:
        print(f"left out: {line}")
    days = total / 24
    print(f"\nGPU total: {total:,.0f} h = {days:.1f} days on one process, {days / args.speedup:.1f} days at "
          f"{args.speedup:g}x; if every search selected its most expensive configuration, {worst / 24:.1f} days "
          f"({worst / 24 / args.speedup:.1f} at {args.speedup:g}x). EASE and popularity run on the CPU beside it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
