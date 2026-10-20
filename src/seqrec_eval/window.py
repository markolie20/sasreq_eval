"""A model at the serving window that suits it: the last n events of each history, n chosen on validation.

DECISIONS §51. ELSA reads a history as a set of items, and on these data it scores higher on test when a request
carries only its most recent events: in the inference sweep it reaches 1.5 to 1.9 times its whole-history score at 5
to 20 events on MovieLens-20M, Music4All and Yambda. Production may serve it that way, so the sequential models are
also compared with that ELSA. The window must be chosen without the test set, or the comparison would be with ELSA's
luckiest window.

For a model whose section lists ``serving_windows``:

1. **Chosen on validation.** The stale run (:func:`~seqrec_eval.runner.plan_stale`) is the selected configuration
   fitted on the training window under the first seed, exactly as its search trial was. It is scored on the
   validation sample with every history cut to its last n events, for each n listed, and with the whole history. The
   best is the window; a tie keeps the longer window, the whole history longest of all.
2. **Scored on test.** Every current final is rescored on test with its histories cut to that window, on the same
   users as its own test score: seen items are still excluded against the whole history, so the users who can be
   scored do not change. On a dataset with the new-item diagnostic, it is scored on new items too.

The report sets every model, at its own whole history, against the windowed model, with the seed-aware test of the
main comparison.

    runs/<dataset>/<model>/<fp[:12]>/stale-seed<s>/val_last<n>.{json,npz}
    runs/<dataset>/<model>/<fp[:12]>/final-seed<s>/test_last<n>.{json,npz}, test_new_last<n>.{json,npz}
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from compresso_recsys.evaluation import EvaluationResult

from .ablations import KEPT, TRANSFORMS
from .evaluate import evaluate_phase
from .models import model_spec
from .protocol import Protocol
from .results import evaluation_exists, load_evaluation, save_evaluation
from .runner import (
    RunSpec,
    execute,
    final_seeds,
    load_final_evaluations,
    plan_stale,
    run_root,
    stale_final_seeds,
    summarize_trials,
)
from .splits import Split

#: the transform that cuts a history to its last n events (``history_length`` with scope inference)
_CUT = "history_length"


def windowed(protocol: Protocol, split: Split, n: int) -> Split:
    """``split`` with every validation and test history cut to its last ``n`` events; training as it is.

    What the user has seen is still the whole history, so excluding seen items -- and with it which users can be
    scored -- is as without the cut.
    """
    data = TRANSFORMS[_CUT].apply(split.data, int(n), rng=None, options={},
                                  value=protocol.dataset(split.dataset).set_all_values_to, phases=("val", "test"))
    data.pop(KEPT, None)
    for phase in ("val", "test"):
        key = f"{phase}_source_matrix"
        if data[key] is not split.data[key]:
            data[f"{phase}_seen_matrix"] = split.data.get(f"{phase}_seen_matrix", split.data[key])
    return replace(split, data=data)


def _stem(phase: str, n: int) -> str:
    return f"{phase}_last{int(n)}"


def stale_spec(protocol: Protocol, work_dir: Path, dataset: str, model: str) -> RunSpec | None:
    """The model's stale run, or ``None`` while it has no selection (or no finals here)."""
    try:
        specs = plan_stale(protocol, work_dir, dataset, model)
    except RuntimeError:
        return None
    return specs[0] if specs else None


def window_choice(protocol: Protocol, work_dir: Path, dataset: str, model: str) -> dict[str, Any] | None:
    """The window chosen on validation, with every candidate's validation value; ``None`` until all are scored.

    ``window`` is the chosen n, or ``None`` when the whole history scored best.
    """
    mp = protocol.model(model)
    spec = stale_spec(protocol, work_dir, dataset, model)
    if not mp.serving_windows or spec is None:
        return None
    directory = spec.directory(work_dir)
    if not (directory / "done.json").exists() or not evaluation_exists(directory / "val"):
        return None
    primary = protocol.primary_metric
    values: dict[int | None, float] = {None: load_evaluation(directory / "val").metrics[primary]}
    for n in mp.serving_windows:
        if not evaluation_exists(directory / _stem("val", n)):
            return None
        values[n] = load_evaluation(directory / _stem("val", n)).metrics[primary]
    # the best on validation; a tie keeps the longer window, the whole history longest of all
    window = max(values, key=lambda n: (values[n], math.inf if n is None else n))
    summary = summarize_trials(protocol, work_dir, dataset, model)
    return {"window": window, "values": values, "stale": spec,
            "trial_value": summary.best_value, "selection": None if summary.best is None else summary.best.selection}


def select_window(protocol: Protocol, work_dir: Path, split: Split, model: str, *, device: str,
                  retry_failed: bool = False, log=print) -> dict[str, Any] | None:
    """Fit the stale run if needed, score it on validation at every window, and return :func:`window_choice`.

    ``split`` is the search's (trained on the training window). Raises as :func:`plan_stale` does while the model
    has no selection; ``None`` if the stale run could not be made here (failed, or held by another process).
    """
    specs = plan_stale(protocol, work_dir, split.dataset, model)
    if not specs:
        return None
    spec = specs[0]
    execute(spec, split, protocol, work_dir, device=device, retry_failed=retry_failed, log=log)
    directory = spec.directory(work_dir)
    if not (directory / "done.json").exists():
        return None
    mp, dp = protocol.model(model), protocol.dataset(split.dataset)
    missing = [n for n in mp.serving_windows if not evaluation_exists(directory / _stem("val", n))]
    if missing:
        registered = model_spec(model)
        trainer = registered.cls.load(directory / "model.zip", device=device)
        for n in missing:
            result = evaluate_phase(trainer, windowed(protocol, split, n), "val", family=registered.family,
                                    protocol=protocol, exclude_seen=dp.exclude_seen, rows=split.val_rows)
            save_evaluation(result, directory / _stem("val", n))
            log(f"[{split.dataset}/{model}] validation {protocol.primary_metric} at the last {n} events: "
                f"{result.metrics[protocol.primary_metric]:.4f}")
    return window_choice(protocol, work_dir, split.dataset, model)


def score_window(protocol: Protocol, work_dir: Path, split: Split, model: str, window: int | None, *,
                 device: str, log=print) -> int:
    """Rescore every current final of ``model`` on test (``split``: the final split) with histories cut to
    ``window``, and on new items where the dataset has that diagnostic; returns how many it scored."""
    if window is None:
        return 0  # the whole history won: the finals' own scores are the windowed ones
    dataset = split.dataset
    dp = protocol.dataset(dataset)
    root = run_root(work_dir, dataset, model, protocol.run_fingerprint(dataset, model))
    stale = stale_final_seeds(protocol, work_dir, dataset, model)
    registered = model_spec(model)
    cut = None
    done = 0
    views = [("primary", "test", dp.exclude_seen)] + ([("new", "test_new", True)] if dp.new_item_diagnostic else [])
    for seed in final_seeds(protocol, work_dir, dataset):
        directory = root / f"final-seed{seed}"
        if seed in stale or not (directory / "done.json").exists() or not (directory / "model.zip").exists():
            continue
        todo = [(targets, name, exclude) for targets, name, exclude in views
                if not evaluation_exists(directory / _stem(name, window))]
        if not todo:
            continue
        cut = cut if cut is not None else windowed(protocol, split, window)
        trainer = registered.cls.load(directory / "model.zip", device=device)
        for targets, name, exclude in todo:
            result = evaluate_phase(trainer, cut, "test", family=registered.family, protocol=protocol,
                                    exclude_seen=exclude, rows=split.test_rows, targets=targets)
            save_evaluation(result, directory / _stem(name, window))
        log(f"[{dataset}/{model}] final-seed{seed} at the last {window} events: test {protocol.primary_metric} "
            f"{load_evaluation(directory / _stem('test', window)).metrics[protocol.primary_metric]:.4f}")
        done += 1
    return done


def window_finals(protocol: Protocol, work_dir: Path, dataset: str, model: str, window: int | None,
                  view: str = "test") -> list[EvaluationResult]:
    """The current finals of ``model`` scored at ``window`` (``view``: "test" or "test_new"), one per seed; at the
    whole history, the finals' own scores."""
    stem = view if window is None else _stem(view, window)
    return [evaluation for _, evaluation in load_final_evaluations(protocol, work_dir, dataset, model, stem=stem)]
