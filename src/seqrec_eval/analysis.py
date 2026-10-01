"""The analysis that comes before any model: the data, whether order matters, and the floor.

Three steps, for each dataset and again for every condition of every ablation
sweep, all on CPU and all on the models' own terms -- the same split, test
users, metrics and evaluator as the models, so every number here can be set
beside a model's.

1. **Profile.** What the data looks like: the five characteristics and the
   audit's descriptive fields (:func:`seqrec_eval.ablations.characteristics`).
   It is the same computation the manipulation check reads.
2. **Baselines and the floor.** Popularity, time-decayed popularity, replay and
   first-order Markov (:mod:`seqrec_eval.baselines`). Each is searched on
   validation with the full data, exactly as a model is, then scored on test
   -- refitted on train and validation first when the protocol refits, as the
   models are; the strongest on the primary metric is the **floor** every
   model has to beat. Across an ablation's levels each baseline keeps its full-data
   setting, as the models keep their stage-1 configuration, so a gap cannot
   move because only one side was retuned.
3. **Sequence signal.** Whether order carries information at all. Markov is
   fitted again on training histories shuffled within each user (order gone,
   contents kept) and on reversed histories (direction gone), and scored on
   the same test users. And Markov's own score is split by whether a test
   history's last two events tie in time: where they do, which item is "last"
   -- the item Markov predicts from -- was decided by the source file, not by
   the user.

The analysis only describes. It decides nothing about which datasets count;
but a profile that contradicts what a transform is meant to do is a bug, and
the ablation report says so.

Everything is stored under fingerprints, so a rerun computes only what is
missing, and a changed baseline section starts its own search:

    analysis/<dataset>/<dfp[:12]>/profile-<evaluation key[:12]>.json
    analysis/<dataset>/<dfp[:12]>/baselines/<name>/<bfp[:12]>/trial-NNN.json, selected.json, test.{json,npz}
    analysis/<dataset>/<dfp[:12]>/controls/<control>/test.{json,npz}
    ablations/<sweep>/<dataset>/<cfp[:12]>/analysis/<name>/<fp[:12]>/<condition>/test.{json,npz}
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from compresso_recsys import ItemSequences
from compresso_recsys.evaluation import EvaluationResult

from .ablations import (
    REFERENCE,
    _event_rows,
    ablation_root,
    build_condition,
    build_reference,
    characteristics,
    condition_name,
    data_seeds,
    last_pair_tied,
    record_characteristics,
    scope_of,
)
from .baselines import (
    MarkovChain,
    Popularity,
    PopularityConfig,
    Replay,
    ReplayConfig,
    TimeDecayedPopularity,
    TimeDecayedPopularityConfig,
)
from .evaluate import evaluate_phase, other_definition, target_key
from .protocol import BaselineProtocol, Protocol, _digest
from .results import (
    evaluation_exists,
    load_evaluation,
    read_json,
    save_evaluation,
    write_json,
)
from .search import _stream, grid_params, trial_params
from .splits import Split, final_split

#: Bump when a control's behaviour changes; it enters the controls' fingerprint. 3 (2026-09-30): ties ranked in
#: one fixed order (see :mod:`seqrec_eval.baselines`), and an inference sweep's conditions reuse the full data's
#: shuffle, since their training data is the full data's.
CONTROLS_VERSION = 3
CONTROLS = ("markov_shuffled", "markov_backwards")

#: kind -> (baseline class, its config, whether fit takes the training timestamps)
KINDS: dict[str, tuple[type, type | None, bool]] = {
    "popularity": (Popularity, PopularityConfig, False),
    "time_popularity": (TimeDecayedPopularity, TimeDecayedPopularityConfig, True),
    "replay": (Replay, ReplayConfig, False),
    "markov": (MarkovChain, None, False),
}


def _build(kind: str, params: dict[str, Any]):
    cls, config, _ = KINDS[kind]
    if config is None:
        if params:
            raise ValueError(f"{kind} takes no settings, got {sorted(params)}")
        return cls()
    return cls(config(**params))


def fit_baseline(kind: str, params: dict[str, Any], split: Split, sequences: ItemSequences | None = None):
    """``kind`` fitted on the split's training histories (or on ``sequences`` in their place)."""
    model = _build(kind, params)
    extra = {}
    if KINDS[kind][2]:
        times = split.data.get("x_train_timestamps")
        if times is None:
            raise ValueError(f"{kind} needs the training events' timestamps, but the {split.dataset} split has "
                             "none; run `seqrec-eval prepare` without --no-timestamps")
        extra["timestamps"] = times
    model.fit(sequences if sequences is not None else split.data["x_train_sequences"],
              item_ids=split.data["train_item_ids"], **extra)
    return model


def _score(protocol: Protocol, split: Split, model, phase: str, targets: str = "primary") -> EvaluationResult:
    rows = split.val_rows if phase == "val" else split.test_rows
    return evaluate_phase(model, split, phase, family="sequence", protocol=protocol,
                          exclude_seen=protocol.dataset(split.dataset).exclude_seen, rows=rows, targets=targets)


def analysis_root(protocol: Protocol, work_dir: Path, dataset: str) -> Path:
    return Path(work_dir) / "analysis" / dataset / protocol.dataset_fingerprint(dataset)[:12]


def _baseline_dir(protocol: Protocol, work_dir: Path, dataset: str, name: str) -> Path:
    return (analysis_root(protocol, work_dir, dataset) / "baselines" / name
            / protocol.baseline_fingerprint(dataset, name)[:12])


def baseline_params(protocol: Protocol, baseline: BaselineProtocol, trial: int) -> dict[str, Any]:
    if baseline.grid:
        return grid_params(baseline.space, baseline.fixed, trial)
    return trial_params(protocol, baseline, trial)


# ---------------------------------------------------------------------------
# baselines: search on validation, score on test
# ---------------------------------------------------------------------------

def select_baseline(protocol: Protocol, work_dir: Path, split: Split, name: str, log=print) -> dict[str, Any]:
    """Search ``name`` on validation (fixed sample) and record the best; cached once done."""
    baseline = protocol.baseline(name)
    directory = _baseline_dir(protocol, work_dir, split.dataset, name)
    selected_path = directory / "selected.json"
    if selected_path.exists():
        return read_json(selected_path)
    primary = protocol.primary_metric
    best = None
    for trial in range(baseline.trials):
        path = directory / f"trial-{trial:03d}.json"
        if path.exists():
            record = read_json(path)
        else:
            params = baseline_params(protocol, baseline, trial)
            started = time.perf_counter()
            val = _score(protocol, split, fit_baseline(baseline.kind, params, split), "val")
            record = {"trial": trial, "params": params, "val": dict(val.metrics),
                      "seconds": time.perf_counter() - started}
            write_json(path, record)
        if not np.isfinite(record["val"][primary]):
            raise ValueError(f"[{split.dataset}] baseline {name} trial {trial}: validation {primary} is not a finite "
                             f"number ({record['val'][primary]}); no baseline is selected until that is fixed")
        if best is None or record["val"][primary] > best["val"][primary]:
            best = record
    log(f"[{split.dataset}] baseline {name}: trial {best['trial']} {best['params']} "
        f"(val {primary} {best['val'][primary]:.4f})")
    selected = {"baseline": name, "kind": baseline.kind, **best}
    write_json(selected_path, selected)
    return selected


def _cached(stem: Path, compute: Callable[[], EvaluationResult], record: dict[str, Any]) -> EvaluationResult:
    """The evaluation at ``stem``, computed and saved (with ``record`` beside it) if missing."""
    if evaluation_exists(stem):
        return load_evaluation(stem)
    started = time.perf_counter()
    result = compute()
    save_evaluation(result, stem)
    write_json(stem.with_name(stem.name + ".done.json"), {**record, "seconds": time.perf_counter() - started,
                                                          "test": dict(result.metrics)})
    return result


def _save_tied(split: Split, stem: Path, result: EvaluationResult) -> None:
    """Beside Markov's result, which of its users' histories end in a tie (nothing without timestamps)."""
    path = stem.with_name("tied.npy")
    if path.exists():
        return
    split_by_tie = tied_split(split, result)
    if split_by_tie is not None:
        np.save(path, split_by_tie["tied"])


def load_tied(stem: Path) -> np.ndarray | None:
    path = stem.with_name("tied.npy")
    return np.load(path) if path.exists() else None


def shuffled_within_users(sequences: ItemSequences, rng: np.random.Generator) -> ItemSequences:
    """Each history's events in random order; which items a history holds is unchanged."""
    order = np.lexsort((rng.random(sequences.values.size), _event_rows(sequences)))
    return ItemSequences(values=sequences.values[order], indptr=sequences.indptr, n_items=sequences.n_items)


def reversed_histories(sequences: ItemSequences) -> ItemSequences:
    """Each history back to front, so a transition a -> b is counted as b -> a."""
    rows = _event_rows(sequences)
    order = np.lexsort((-np.arange(sequences.values.size), rows))
    return ItemSequences(values=sequences.values[order], indptr=sequences.indptr, n_items=sequences.n_items)


def fit_control(protocol: Protocol, split: Split, control: str, stream: tuple) -> MarkovChain:
    training = split.data["x_train_sequences"]
    if control == "markov_shuffled":
        training = shuffled_within_users(training, _stream(protocol.search_seed, "control", *stream))
    elif control == "markov_backwards":
        training = reversed_histories(training)
    else:
        raise ValueError(f"unknown control {control!r}")
    return fit_baseline("markov", {}, split, training)


def _controls_fingerprint(protocol: Protocol, dataset: str) -> str:
    # the whole dataset section: exclude_seen changes the controls' scores but not the split's fingerprint
    return _digest({"split": protocol.dataset_fingerprint(dataset), "search_seed": protocol.search_seed,
                    "version": CONTROLS_VERSION, "protocol": protocol._result_settings(),
                    "dataset": protocol.dataset(dataset).raw})


# ---------------------------------------------------------------------------
# one dataset, full data
# ---------------------------------------------------------------------------

def profile_path(protocol: Protocol, work_dir: Path, dataset: str) -> Path:
    """The full-data profile: of the data the tested models train on (train+validation under refit), and of the
    users they are scored on, so keyed by the evaluation key too."""
    return analysis_root(protocol, work_dir, dataset) / f"profile-{protocol.evaluation_key(dataset)[:12]}.json"


def analyse_full(protocol: Protocol, work_dir: Path, split: Split, log=print) -> None:
    """Profile, baselines (searched here) and controls on the full data of one dataset.

    Baselines are searched on ``split`` (fitted on the training window, scored
    on validation); everything scored on test is fitted on
    :func:`final_split`'s data, as the models' final runs are.
    """
    root = analysis_root(protocol, work_dir, split.dataset)
    tested = final_split(protocol, split)
    path = profile_path(protocol, work_dir, split.dataset)
    if not path.exists():
        write_json(path, {"dataset": split.dataset, "trained_on": tested.trained_on,
                          "characteristics": characteristics(tested, None, target_key("test", protocol.targets))})
    other = other_definition(protocol.targets)
    diagnose = tested.data.get(target_key("test", other)) is not None
    for name in protocol.baselines:
        selected = select_baseline(protocol, work_dir, split, name, log=log)
        stem = _baseline_dir(protocol, work_dir, split.dataset, name) / "test"
        fit = lambda s=selected: fit_baseline(s["kind"], s["params"], tested)
        record = {"baseline": name, "params": selected["params"], "trained_on": tested.trained_on}
        result = _cached(stem, lambda: _score(protocol, tested, fit(), "test"), record)  # noqa: B023
        if selected["kind"] == "markov":
            _save_tied(tested, stem, result)
        if diagnose:
            _cached(stem.with_name(f"test_{other}"), lambda: _score(protocol, tested, fit(), "test", other),  # noqa: B023
                    {**record, "targets": other})
    for control in CONTROLS:
        stem = root / "controls" / control / _controls_fingerprint(protocol, split.dataset)[:12] / "test"
        fit = lambda c=control: fit_control(protocol, tested, c, _control_stream(protocol, tested))
        record = {"control": control, "trained_on": tested.trained_on}
        _cached(stem, lambda: _score(protocol, tested, fit(), "test"), record)  # noqa: B023
        if diagnose:
            _cached(stem.with_name(f"test_{other}"), lambda: _score(protocol, tested, fit(), "test", other),  # noqa: B023
                    {**record, "targets": other})


# ---------------------------------------------------------------------------
# every ablation condition
# ---------------------------------------------------------------------------

def _condition_stem(protocol: Protocol, work_dir: Path, sweep: str, dataset: str, name: str, fingerprint: str,
                    condition: str) -> Path:
    return (ablation_root(protocol, work_dir, sweep, dataset) / "analysis" / name / fingerprint[:12]
            / condition / "test")


def _condition_scorers(protocol: Protocol, work_dir: Path, split: Split) -> dict[str, tuple[str, Callable]]:
    """name -> (fingerprint, fit on a condition's split): the selected baselines, then the controls."""
    scorers = {}
    for name in protocol.baselines:
        selected = select_baseline(protocol, work_dir, split, name, log=lambda _: None)
        fingerprint = _digest({"baseline": protocol.baseline_fingerprint(split.dataset, name),
                               "params": selected["params"]})
        scorers[name] = (fingerprint, lambda condition, s=selected: fit_baseline(s["kind"], s["params"], condition))
    controls = _controls_fingerprint(protocol, split.dataset)
    for control in CONTROLS:
        scorers[control] = (controls, lambda condition, c=control: fit_control(
            protocol, condition, c, _control_stream(protocol, condition)))
    return scorers


def _control_stream(protocol: Protocol, condition: Split) -> tuple:
    """The shuffle's stream: the full data's wherever the training data is the full data's -- the reference, and
    every condition of an inference sweep -- so those are all the same control; a stream of its own elsewhere.

    Drawing a new shuffle for each level of an inference sweep would move the control between levels by chance
    alone (on ML-20M it varied from 0.0155 to 0.0178 while Markov did not move).
    """
    if (condition.condition is None or condition.condition["label"] == REFERENCE
            or scope_of(protocol.ablation(condition.condition["sweep"])) == "inference"):
        return (condition.dataset, "full")
    return (condition.dataset, condition.condition["sweep"], condition.condition["label"],
            condition.condition["data_seed"])


def analyse_sweep(protocol: Protocol, work_dir: Path, split: Split, sweep: str, log=print) -> None:
    """Profile, baselines and controls at every condition of ``sweep``, on the users each is scored on.

    The baselines keep the setting selected on ``split``; the conditions are
    transforms of :func:`final_split`'s data, like the models' ablation runs.
    """
    scorers = _condition_scorers(protocol, work_dir, split)
    tested = final_split(protocol, split)

    def run(condition: Split, name: str) -> None:
        record_characteristics(protocol, work_dir, condition, tested)
        for scorer, (fingerprint, fit) in scorers.items():
            stem = _condition_stem(protocol, work_dir, sweep, split.dataset, scorer, fingerprint, name)
            result = _cached(stem, lambda: _score(protocol, condition, fit(condition), "test"),  # noqa: B023
                             {"scorer": scorer, "condition": name})
            if scorer in protocol.baselines and protocol.baseline(scorer).kind == "markov":
                _save_tied(condition, stem, result)

    def pending(name: str) -> bool:
        return any(not evaluation_exists(_condition_stem(protocol, work_dir, sweep, split.dataset, scorer,
                                                         fingerprint, name))
                   for scorer, (fingerprint, _) in scorers.items())

    if pending(REFERENCE):
        run(build_reference(protocol, work_dir, tested, sweep), REFERENCE)
    for level in protocol.ablation(sweep).levels:
        for data_seed in data_seeds(protocol, work_dir, sweep, split.dataset):
            name = condition_name(level, data_seed)
            if pending(name):
                log(f"[{split.dataset}] {sweep}: analysing condition {name}")
                run(build_condition(protocol, work_dir, tested, sweep, level, data_seed), name)


# ---------------------------------------------------------------------------
# reading results back
# ---------------------------------------------------------------------------

def full_results(protocol: Protocol, work_dir: Path, dataset: str) -> dict[str, Any]:
    """Whatever the full-data analysis of ``dataset`` has finished: profile, baselines, controls."""
    root = analysis_root(protocol, work_dir, dataset)
    other = other_definition(protocol.targets)
    out: dict[str, Any] = {"profile": None, "baselines": {}, "selected": {}, "controls": {},
                           "other_targets": other, "diagnostic": {}}
    if profile_path(protocol, work_dir, dataset).exists():
        out["profile"] = read_json(profile_path(protocol, work_dir, dataset))["characteristics"]
    for name in protocol.baselines:
        directory = _baseline_dir(protocol, work_dir, dataset, name)
        if (directory / "selected.json").exists():
            out["selected"][name] = read_json(directory / "selected.json")
        if evaluation_exists(directory / "test"):
            out["baselines"][name] = load_evaluation(directory / "test")
            tied = load_tied(directory / "test")
            if tied is not None:
                out.setdefault("tied", {})[name] = tied
        if evaluation_exists(directory / f"test_{other}"):
            out["diagnostic"][name] = load_evaluation(directory / f"test_{other}")
    for control in CONTROLS:
        stem = root / "controls" / control / _controls_fingerprint(protocol, dataset)[:12] / "test"
        if evaluation_exists(stem):
            out["controls"][control] = load_evaluation(stem)
        if evaluation_exists(stem.with_name(f"test_{other}")):
            out["diagnostic"][control] = load_evaluation(stem.with_name(f"test_{other}"))
    return out


def condition_results(protocol: Protocol, work_dir: Path, sweep: str, dataset: str
                      ) -> dict[str, dict[str, list[tuple[EvaluationResult, np.ndarray | None]]]]:
    """scorer -> condition column -> (evaluation, tied mask or ``None``) per subsample, for what has finished."""
    ablation_names = [REFERENCE] + [condition_name(level, seed) for level in protocol.ablation(sweep).levels
                                    for seed in data_seeds(protocol, work_dir, sweep, dataset)]
    out: dict[str, dict[str, list[EvaluationResult]]] = {}
    for name in list(protocol.baselines) + list(CONTROLS):
        if name in protocol.baselines:
            directory = _baseline_dir(protocol, work_dir, dataset, name)
            if not (directory / "selected.json").exists():
                continue
            selected = read_json(directory / "selected.json")
            fingerprint = _digest({"baseline": protocol.baseline_fingerprint(dataset, name),
                                   "params": selected["params"]})
        else:
            fingerprint = _controls_fingerprint(protocol, dataset)
        for condition in ablation_names:
            stem = _condition_stem(protocol, work_dir, sweep, dataset, name, fingerprint, condition)
            if evaluation_exists(stem):
                out.setdefault(name, {}).setdefault(condition.split("/")[0], []).append(
                    (load_evaluation(stem), load_tied(stem)))
    return out


def floor_of(results: dict[str, EvaluationResult], metric: str, *,
             expected: Iterable[str]) -> tuple[str, EvaluationResult] | None:
    """The strongest baseline on ``metric``: the floor. ``None`` until every baseline in ``expected`` has a
    result, since the strongest of those finished would be a lower floor, silently (review N34)."""
    expected = list(expected)
    if not expected or any(name not in results for name in expected):
        return None
    name = max(expected, key=lambda n: results[n].metrics[metric])
    return name, results[name]


def tied_split(split: Split, result: EvaluationResult) -> dict[str, Any] | None:
    """``result``'s users split by whether their history's last two events tie; ``None`` without timestamps."""
    tied = last_pair_tied(split)
    if tied is None:
        return None
    rows = pd.Index(split.eval_user_ids("test")).get_indexer(np.asarray(result.sample_ids).astype(str))
    if (rows < 0).any():
        raise ValueError(f"{split.dataset}: scored users missing from the split's test users")
    mask = tied[rows]
    return {"tied": mask, "n_tied": int(mask.sum()), "n_real_gap": int((~mask).sum())}


def mean_over_subsamples(results: list[EvaluationResult], *, strip_targets: bool = False) -> EvaluationResult:
    """Per-user values over a stochastic condition's subsamples, pooled where they scored different users."""
    from .report import pool_over_seeds

    if strip_targets:
        results = [replace(result, target_fingerprint=None) for result in results]
    return results[0] if len(results) == 1 else pool_over_seeds(results)
