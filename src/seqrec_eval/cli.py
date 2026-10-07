"""``seqrec-eval``: the stage-1 workflow, one command per step.

    seqrec-eval plan                       what the protocol will run
    seqrec-eval prepare --data-dir DIR     build each dataset's temporal split once, with event times
    seqrec-eval analyse                    profile, baselines and floor, sequence signal (CPU; every condition)
    seqrec-eval analysis-report            work/reports/analysis.md
    seqrec-eval search  --device cuda:0    random-search trials, validation only
    seqrec-eval final   --device cuda:0    selected configuration x seeds, with test
    seqrec-eval diversity --device cuda:0  coverage and intra-list diversity of the saved final models
    seqrec-eval status | report | latency
    seqrec-eval ablate  --device cuda:0    each [ablations.*] sweep on the stage-1 configurations
    seqrec-eval ablation-report            work/reports/ablation-<sweep>.md and its CSVs
    seqrec-eval repeat-strata              stage-1 results by repeat share within history-length bins

``search``, ``final`` and ``ablate`` are resumable and lock each run, so on a two-GPU
machine run the same command twice, once per ``--device``, and the processes
share the work.

``search``, ``final`` and ``ablate`` exit 0 when everything they planned is done, 1 when a run failed or
was refused (a final made under another selection), and 3 when work is left: trials still running in the
other process, a model not ready for its finals, a run that failed before and was not retried. So
``search && final && ablate`` stops where a step is not finished.

More seeds can be added once the protocol's have run, without touching any run
already made: ``final --dataset D --add-seeds 3 4`` gives dataset D's final runs
seeds 3 and 4, and ``ablate --dataset D --sweep S --add-seeds 3 4`` gives the
sweeps the same (they need the stage-1 seeds first). Each is recorded in the
work directory, and every later command and report uses it. ``--work-dir`` and ``--data-dir`` default to the environment
variables ``SEQREC_EVAL_WORK`` and ``COMPRESSO_DATA_DIR``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch

from .ablation_report import build_ablation_report
from .ablations import (
    REFERENCE,
    ablation_root,
    build_condition,
    build_reference,
    add_sweep_seeds,
    check_ablation,
    check_added_seeds,
    check_stage1_seeds,
    condition_name,
    data_seeds,
    fixed_test_rows,
    per_condition_users,
    plan_ablation,
    record_characteristics,
    scope_of,
    sweep_seeds,
)
from .analysis import analyse_full, analyse_sweep
from .analysis_report import build_analysis_report
from .diversity import diversity_table, measure as measure_diversity
from .latency import benchmark, latency_table, parse_cores
from .models import PLUGIN_ERRORS, model_spec
from .protocol import load_protocol
from .report import build_report, status_table
from .results import read_json
from .runner import add_final_seeds, execute, final_seeds, plan_finals, plan_trials
from .splits import final_split, load_split, prepare_split, split_dir
from .strata import build_strata_report


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def _select(requested: list[str] | None, available: dict, kind: str) -> list[str]:
    if not requested or requested == ["all"]:
        return list(available)
    if kind == "sweep" and requested == ["none"]:
        return []  # e.g. `analyse --sweep none`: the full data only, no ablation condition
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise SystemExit(f"unknown {kind}(s) {unknown}; the protocol defines {list(available)}")
    return requested


#: the environment variable that names extra protocol files when ``--protocol-extra`` is not given
EXTRA_ENV = "SEQREC_EVAL_PROTOCOL_EXTRA"


def _extras(given: list[str] | None) -> list[str]:
    if given is not None:
        return given
    return [path for path in os.environ.get(EXTRA_ENV, "").split(os.pathsep) if path]


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seqrec-eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", default="protocol.toml", help="protocol file (default: %(default)s)")
    parser.add_argument("--protocol-extra", action="append", metavar="FILE",
                        help="an extra protocol file of [models.*] sections only, added to the protocol's: a model "
                             "kept out of the repository (repeatable; default: the files in $" + EXTRA_ENV + ", "
                             "separated by " + repr(os.pathsep) + ")")
    parser.add_argument("--work-dir", default=os.environ.get("SEQREC_EVAL_WORK", "work"),
                        help="splits, runs and reports (default: $SEQREC_EVAL_WORK or ./work)")
    commands = parser.add_subparsers(dest="command", required=True)

    def selection(sub, *, models: bool = True):
        sub.add_argument("--dataset", nargs="+", help="dataset names from the protocol (default: all)")
        if models:
            sub.add_argument("--model", nargs="+", help="model names from the protocol (default: all)")

    commands.add_parser("plan", help="list datasets, models, trial counts and fingerprints")

    prepare = commands.add_parser("prepare", help="build and extract each dataset's temporal split")
    selection(prepare, models=False)
    prepare.add_argument("--data-dir", default=os.environ.get("COMPRESSO_DATA_DIR"),
                         help="compresso-recsys data directory (default: $COMPRESSO_DATA_DIR)")
    prepare.add_argument("--force", action="store_true", help="rebuild a split prepared from a different section")
    prepare.add_argument("--quiet", action="store_true", help="no builder progress bars")
    prepare.add_argument("--no-timestamps", action="store_true",
                         help="skip recovering event times (the time-decayed baseline then cannot run)")

    for name, text in (("search", "run random-search trials (validation only)"),
                       ("final", "run the selected configuration under every seed (validation and test)")):
        sub = commands.add_parser(name, help=text)
        selection(sub)
        sub.add_argument("--device", default=_default_device(), help="torch device (default: %(default)s)")
        sub.add_argument("--threads", type=int, help="torch CPU threads for this process")
        sub.add_argument("--retry-failed", action="store_true", help="rerun runs that failed before")
        if name == "final":
            sub.add_argument("--allow-incomplete", action="store_true",
                             help="select from an unfinished search (breaks the equal budget)")
            sub.add_argument("--accept-failed", action="store_true",
                             help="select although trials failed, for failures that cannot be fixed (e.g. out of "
                                  "memory); they are recorded and listed in the report")
            sub.add_argument("--add-seeds", nargs="+", type=int, metavar="SEED",
                             help="give the selected datasets these further final seeds (all their models), beside "
                                  "the protocol's; recorded, so later commands and reports use them too")

    status = commands.add_parser("status", help="progress per dataset and model")
    selection(status)

    report = commands.add_parser("report", help="write work/reports/stage1.md and final_metrics.csv")
    selection(report)
    report.add_argument("--reference", default="elsa", help="model every other is compared against (default: %(default)s)")

    diversity = commands.add_parser("diversity", help="coverage and intra-list diversity of the saved finals")
    selection(diversity)
    diversity.add_argument("--device", default=_default_device(), help="torch device (default: %(default)s)")
    diversity.add_argument("--force", action="store_true", help="measure again finals that have a record")

    latency = commands.add_parser("latency", help="CPU inference latency of the saved final models")
    selection(latency)
    latency.add_argument("--threads", type=int, default=4, help="torch threads (default: %(default)s)")
    latency.add_argument("--cores", help="pin to these CPUs, e.g. 16-19, away from training jobs")

    analyse = commands.add_parser("analyse", help="profile, baselines and sequence signal, on CPU, per condition")
    selection(analyse, models=False)
    analyse.add_argument("--sweep", nargs="+",
                         help="sweep names from the protocol (default: all; none: the full data only)")
    analyse.add_argument("--threads", type=int, help="torch CPU threads for this process")

    analysis_report = commands.add_parser("analysis-report", help="write work/reports/analysis.md")
    selection(analysis_report, models=False)

    ablate = commands.add_parser("ablate", help="run the ablation sweeps on the stage-1 configurations")
    selection(ablate)
    ablate.add_argument("--sweep", nargs="+", help="sweep names from the protocol (default: all)")
    ablate.add_argument("--device", default=_default_device(), help="torch device (default: %(default)s)")
    ablate.add_argument("--threads", type=int, help="torch CPU threads for this process")
    ablate.add_argument("--retry-failed", action="store_true", help="rerun runs that failed before")
    ablate.add_argument("--add-seeds", nargs="+", type=int, metavar="SEED",
                        help="give the selected sweeps on the selected datasets these further seeds (all their "
                             "models); each must be a stage-1 seed of the dataset already (final --add-seeds)")

    ablation_report = commands.add_parser("ablation-report", help="write work/reports/ablation-<sweep>.md")
    selection(ablation_report)
    ablation_report.add_argument("--reference", default="elsa",
                                 help="non-sequential model the gap is tested against (default: %(default)s)")
    ablation_report.add_argument("--sweep", nargs="+", help="sweep names from the protocol (default: all)")

    strata = commands.add_parser("repeat-strata", help="write work/reports/repeat-strata.md from the stage-1 finals")
    selection(strata)
    return parser


def _operator_stop(signum, frame) -> None:
    raise KeyboardInterrupt(signal.Signals(signum).name)


@contextlib.contextmanager
def _operator_stops():
    """``kill`` (SIGTERM) and a closed terminal (SIGHUP) stop a command as Ctrl-C does, so the run in progress
    is not counted as a death (review N31): a ``nohup`` launch cannot be stopped otherwise. A SIGHUP that
    ``nohup`` ignores stays ignored. SIGKILL (``kill -9``) and the kernel's out-of-memory killer cannot be caught,
    and still count."""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGHUP):
            if signal.getsignal(signum) is not signal.SIG_IGN:
                previous[signum] = signal.signal(signum, _operator_stop)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)


def main(argv: list[str] | None = None) -> int:
    with _operator_stops():
        try:
            return _main(argv)
        except KeyboardInterrupt as stop:
            print(f"stopped{f' by {stop}' if str(stop) else ''}: the run in progress was not counted as a failed "
                  "attempt; rerun the command to go on", file=sys.stderr)
            return 130


def _main(argv: list[str] | None) -> int:
    args = _parser().parse_args(argv)
    protocol = load_protocol(args.protocol, extra=_extras(args.protocol_extra))
    work_dir = Path(args.work_dir)
    if "TMPDIR" not in os.environ:
        # the library stages every model save and load in a temporary directory: keep that on the work dir's
        # disk (on the DGX /raid, not the small shared /), unless TMPDIR says otherwise (review B6)
        (work_dir / "tmp").mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(work_dir / "tmp")
    datasets = _select(getattr(args, "dataset", None), protocol.datasets, "dataset")
    models = _select(getattr(args, "model", None), protocol.models, "model")
    sweeps = _select(getattr(args, "sweep", None), protocol.ablations, "sweep")
    for sweep in sweeps:
        check_ablation(protocol.ablation(sweep))  # a bad level fails now, not mid-sweep

    if args.command == "plan":
        total = 0
        print(f"protocol {protocol.path} v{protocol.version}: primary {protocol.primary_metric}, "
              f"seeds {list(protocol.seeds)}, cutoffs {list(protocol.cutoffs)}")
        for extra in protocol.extras:
            print(f"  with the models of {extra}")
        for plugin, error in PLUGIN_ERRORS.items():
            print(f"  ⚠ model plugin {plugin!r} failed to load: {error}")
        for dataset in datasets:
            prepared = (split_dir(work_dir, dataset) / "split_info.json").exists()
            seeds = final_seeds(protocol, work_dir, dataset)
            added = f", seeds {list(seeds)} with those added" if len(seeds) > len(protocol.seeds) else ""
            print(f"\n{dataset}  (split {'prepared' if prepared else 'not prepared'}, "
                  f"fingerprint {protocol.dataset_fingerprint(dataset)[:12]}{added})")
            for model in models:
                mp = protocol.model(model)
                registered = "registered" if _registered(model) else "NOT REGISTERED"
                runs = mp.trials + len(seeds)
                total += runs
                print(f"  {model:<16} {mp.family:<8} {mp.trials:>3} trials + {len(seeds)} finals"
                      f"  run fingerprint {protocol.run_fingerprint(dataset, model)[:12]}  [{registered}]")
        print(f"\n{total} runs in total")
        for name, baseline in protocol.baselines.items():
            search = "grid" if baseline.grid else "random search"
            print(f"baseline {name}: {baseline.kind}, {baseline.trials} validation trials ({search}), CPU")
        for sweep in sweeps:
            ablation = protocol.ablation(sweep)
            work = "rescorings" if scope_of(ablation) == "inference" else "fits"
            print(f"\nablation {sweep}: {ablation.transform} at {list(ablation.levels)}, scope "
                  f"{scope_of(ablation)}, seeds {list(ablation.seeds)}; {len(ablation.levels) * len(ablation.seeds)} "
                  f"{work} + {len(ablation.seeds)} reference rescorings per dataset and model, on "
                  f"{list(ablation.datasets)} x {list(ablation.models)}")
            for dataset in ablation.datasets:
                seeds = sweep_seeds(protocol, work_dir, sweep, dataset)
                if len(seeds) > len(ablation.seeds):
                    print(f"  on {dataset} under seeds {list(seeds)}, with those added: "
                          f"{len(ablation.levels) * len(seeds)} {work} + {len(seeds)} reference rescorings per model")
        return 0

    if args.command == "prepare":
        if not args.data_dir:
            raise SystemExit("--data-dir is required (or set COMPRESSO_DATA_DIR)")
        for dataset in datasets:
            _log(f"preparing {dataset}")
            path = prepare_split(protocol, dataset, data_dir=Path(args.data_dir), work_dir=work_dir,
                                 force=args.force, show_progress=not args.quiet,
                                 timestamps=not args.no_timestamps)
            stages = read_json(path / "split_info.json")["stages"]
            sampled = read_json(path / "split_info.json").get("val_rows_sampled")
            _log(f"{dataset} ready at {path}: {stages['train']['rows']:,} training users, "
                 f"{stages['val']['rows']:,} validation users ("
                 + (f"search scores a fixed {sampled:,}" if sampled else "search scores all of them")
                 + f"), {stages['test']['rows']:,} test users")
        return 0

    if args.command in ("search", "final"):
        if args.threads:
            torch.set_num_threads(args.threads)
        for model in models:
            model_spec(model)  # fail before any split is loaded, not halfway through
        counts: dict[str, int] = {}
        if args.command == "final" and args.add_seeds:
            for dataset in datasets:
                try:
                    new = add_final_seeds(protocol, work_dir, dataset, args.add_seeds)
                except ValueError as error:
                    raise SystemExit(str(error)) from None
                _log(f"[{dataset}] final seeds {list(final_seeds(protocol, work_dir, dataset))}"
                     + (f", {new} added" if new else ", nothing new to add"))
        for dataset in datasets:
            split = load_split(work_dir, dataset, protocol)
            if args.command == "final":
                split = final_split(protocol, split)  # refitted on train+validation when the protocol says so
            for model in models:
                if args.command == "search":
                    specs = plan_trials(protocol, dataset, model)
                else:
                    try:
                        specs = plan_finals(protocol, work_dir, dataset, model,
                                            allow_incomplete=args.allow_incomplete,
                                            accept_failed=args.accept_failed)
                    except RuntimeError as error:
                        _log(f"[{dataset}/{model}] not ready for final runs: {error}")
                        counts["not-ready"] = counts.get("not-ready", 0) + 1
                        continue
                for spec in specs:
                    status = execute(spec, split, protocol, work_dir, device=args.device,
                                     retry_failed=args.retry_failed, log=_log)
                    counts[status] = counts.get(status, 0) + 1
            del split
        _log("finished: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        return _exit_code(counts)

    if args.command == "status":
        print(status_table(protocol, work_dir, datasets, models))
        return 0

    if args.command == "report":
        markdown, table = build_report(protocol, work_dir, datasets, models, args.reference)
        latency = latency_table(protocol, work_dir, datasets, models)
        if latency.count("\n") > 1:
            markdown += "\n## CPU inference latency\n\n" + latency + "\n"
        diversity = diversity_table(protocol, work_dir, datasets, models)
        if diversity.count("\n") > 1:
            markdown += ("\n## Diversity (diagnostics)\n\nOn the same test users and lists as the metrics, mean ± sd "
                         "over seeds. Coverage: the share of the training catalogue in anyone's top k. Intra-list "
                         "diversity: the mean dissimilarity of the pairs in a list, two items being similar when the "
                         "same users interacted with them in training (the cosine on a rank-64 factorisation). "
                         "Nothing is selected on these (`seqrec-eval diversity`).\n\n" + diversity + "\n")
        out = work_dir / "reports"
        out.mkdir(parents=True, exist_ok=True)
        (out / "stage1.md").write_text(markdown)
        if table:
            (out / "final_metrics.csv").write_text(table)
        print(markdown)
        _log(f"wrote {out / 'stage1.md'}")
        return 0

    if args.command == "analyse":
        if args.threads:
            torch.set_num_threads(args.threads)
        for dataset in datasets:
            split = load_split(work_dir, dataset, protocol)
            _log(f"[{dataset}] analysing the full data")
            analyse_full(protocol, work_dir, split, log=_log)
            for sweep in sweeps:
                if dataset in protocol.ablation(sweep).datasets:
                    _log(f"[{dataset}] analysing {sweep}")
                    analyse_sweep(protocol, work_dir, split, sweep, log=_log)
            del split
        _log("analysis finished")
        return 0

    if args.command == "analysis-report":
        markdown, table = build_analysis_report(protocol, work_dir, datasets)
        out = work_dir / "reports"
        out.mkdir(parents=True, exist_ok=True)
        (out / "analysis.md").write_text(markdown)
        if table:
            (out / "analysis.csv").write_text(table)
        print(markdown)
        _log(f"wrote {out / 'analysis.md'}")
        return 0

    if args.command == "ablate":
        if args.threads:
            torch.set_num_threads(args.threads)
        for model in models:
            model_spec(model)
        counts = {}
        if args.add_seeds:
            # every sweep and dataset is checked before any is recorded, so a refusal leaves nothing half-added
            try:
                for dataset in datasets:
                    for sweep in (s for s in sweeps if dataset in protocol.ablation(s).datasets):
                        check_stage1_seeds(protocol, work_dir, sweep, dataset, args.add_seeds)
            except ValueError as error:
                raise SystemExit(str(error)) from None
        for dataset in datasets:
            active = [s for s in sweeps if dataset in protocol.ablation(s).datasets]
            if not active:
                continue
            split = final_split(protocol, load_split(work_dir, dataset, protocol))
            for sweep in active:
                if args.add_seeds:
                    try:
                        check_added_seeds(protocol, work_dir, split, sweep, args.add_seeds)
                        new = add_sweep_seeds(protocol, work_dir, sweep, dataset, args.add_seeds)
                    except (RuntimeError, ValueError) as error:
                        raise SystemExit(str(error)) from None
                    _log(f"[{dataset}] {sweep}: seeds {list(sweep_seeds(protocol, work_dir, sweep, dataset))}"
                         + (f", {new} added" if new else ", nothing new to add"))
                    if new and data_seeds(protocol, work_dir, sweep, dataset) != [None]:
                        _log(f"[{dataset}] {sweep} draws a subsample per seed: run `analyse --dataset {dataset} "
                             f"--sweep {sweep}` too, for the floor and controls on the new subsamples")
                _ablate(protocol, work_dir, split, sweep, [m for m in models if m in protocol.ablation(sweep).models],
                        args, counts)
            del split
        _log("finished: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        return _exit_code(counts)

    if args.command == "ablation-report":
        out = work_dir / "reports"
        out.mkdir(parents=True, exist_ok=True)
        for sweep in sweeps:
            built = build_ablation_report(protocol, work_dir, sweep, datasets, models, args.reference)
            (out / f"ablation-{sweep}.md").write_text(built["markdown"])
            if built["figure"] is not None:
                (out / f"ablation-{sweep}-gap.png").write_bytes(built["figure"])
            for suffix in ("metrics", "gap", "analysis"):
                if built[suffix]:
                    (out / f"ablation-{sweep}-{suffix}.csv").write_text(built[suffix])
            print(built["markdown"])
            _log(f"wrote {out / f'ablation-{sweep}.md'}")
        return 0

    if args.command == "repeat-strata":
        markdown, table = build_strata_report(protocol, work_dir, datasets, models)
        out = work_dir / "reports"
        out.mkdir(parents=True, exist_ok=True)
        (out / "repeat-strata.md").write_text(markdown)
        if table:
            (out / "repeat-strata.csv").write_text(table)
        print(markdown)
        _log(f"wrote {out / 'repeat-strata.md'}")
        return 0

    if args.command == "diversity":
        for dataset in datasets:
            split = final_split(protocol, load_split(work_dir, dataset, protocol))
            for model in models:
                measure_diversity(protocol, work_dir, split, model, device=args.device, force=args.force, log=_log)
            del split
        return 0

    if args.command == "latency":
        cores = parse_cores(args.cores)
        for dataset in datasets:
            split = final_split(protocol, load_split(work_dir, dataset, protocol))
            for model in models:
                try:
                    result = benchmark(protocol, work_dir, split, model, threads=args.threads, cores=cores)
                except FileNotFoundError as error:
                    _log(str(error))
                    continue
                o = result["overall"]
                _log(f"[{dataset}/{model}] P50 {o['p50_ms']:.2f} ms  P95 {o['p95_ms']:.2f} ms  "
                     f"P99 {o['p99_ms']:.2f} ms over {o['n']} requests")
            del split
        return 0
    return 2


def _ablate(protocol, work_dir: Path, split, sweep: str, models: list[str], args, counts: dict) -> None:
    """One sweep on one dataset: the reference first, then each condition in turn.

    A condition's split is built only if one of its runs, or its manipulation
    check, is still missing, so resuming a finished sweep costs no transform.
    """
    dataset = split.dataset
    plans = {}
    for model in models:
        try:
            plan = plan_ablation(protocol, work_dir, sweep, dataset, model)
        except RuntimeError as error:
            _log(f"[{dataset}/{model}] not ready for {sweep}: {error}")
            counts["not-ready"] = counts.get("not-ready", 0) + 1
            continue
        if plan:
            plans[model] = plan
    if not plans:
        return
    if per_condition_users(protocol, sweep):
        _log(f"[{dataset}] {sweep}: each condition is scored on its own users (targets change with the level)")
    else:
        rows = fixed_test_rows(protocol, work_dir, split, sweep)
        _log(f"[{dataset}] {sweep}: {rows.size:,} test users eligible at every level")

    def run(condition, name: str) -> None:
        record_characteristics(protocol, work_dir, condition, split)
        for plan in plans.values():
            for spec in plan[name]:
                status = execute(spec, condition, protocol, work_dir, device=args.device,
                                 retry_failed=args.retry_failed, log=_log)
                counts[status] = counts.get(status, 0) + 1

    def pending(name: str) -> bool:
        measured = (ablation_root(protocol, work_dir, sweep, dataset) / "conditions" / f"{name}.json").exists()
        return not measured or any(not (spec.directory(work_dir) / "done.json").exists()
                                   for plan in plans.values() for spec in plan[name])

    run(build_reference(protocol, work_dir, split, sweep), REFERENCE)
    for level in protocol.ablation(sweep).levels:
        for data_seed in data_seeds(protocol, work_dir, sweep, dataset):
            name = condition_name(level, data_seed)
            if not pending(name):
                continue
            _log(f"[{dataset}] {sweep}: building condition {name}")
            run(build_condition(protocol, work_dir, split, sweep, level, data_seed), name)


#: Run statuses that mean something is wrong (exit 1) and that work is left to do (exit 3).
_WRONG = ("failed", "stale-selection", "stage1-failed")
_UNFINISHED = ("not-ready", "running-elsewhere", "failed-before", "waiting-for-stage1")


def _exit_code(counts: dict[str, int]) -> int:
    """1 if a run failed or was refused, 3 if work is left (so ``search && final`` does not go on), else 0."""
    if any(counts.get(status) for status in _WRONG):
        return 1
    return 3 if any(counts.get(status) for status in _UNFINISHED) else 0


def _registered(model: str) -> bool:
    try:
        model_spec(model)
    except KeyError:
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
