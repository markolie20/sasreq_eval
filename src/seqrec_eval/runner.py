"""Planning and executing runs so that nothing is lost and nothing is done twice.

A run is one fit of one configuration: a search *trial*, or a *final* run of the
selected configuration under one seed. Each lives in its own directory,

    runs/<dataset>/<model>/<fingerprint[:12]>/trial-007/
    runs/<dataset>/<model>/<fingerprint[:12]>/final-seed1/
    runs/<dataset>/added_seeds.json          seeds added with ``final --add-seeds``

and is finished exactly when ``done.json`` exists. That gives three properties
an unattended weekend needs:

* **Resumable.** Rerunning a command skips every finished run, so a crash costs
  only the run it interrupted.
* **Parallel without coordination.** A run is claimed with an exclusive lock
  file before it starts, so two processes -- one per GPU -- can execute the same
  command and simply divide the work between them. A lock left by a process that
  died on this host is reclaimed.
* **Failures stop, not loop.** A run that raises records ``failed.json`` and is
  skipped afterwards until ``--retry-failed``, so a configuration that runs out
  of memory does not retry itself all weekend.

A selected configuration still improving at its last epoch is trained again at multiples of its epochs
(``[protocol] epoch_extension``, DECISIONS §42): ``trial-007-x2``, ``trial-007-x4`` beside ``trial-007``, scored on
validation like any trial, and the best of them on validation is the selection.

Seeds can be added to a dataset after its runs are made. Only the first seed
enters the fingerprint (it seeds every trial), so ``final --add-seeds 3 4``
adds ``final-seed3`` and ``final-seed4`` beside the finals already there and
records the addition in ``added_seeds.json``; from then on the dataset's seeds
(:func:`final_seeds`) are the protocol's followed by the added ones, for every
command and report.

Search trials score **validation only**. Test is scored in final runs, after the
configuration has been chosen, so the stage-1 choice never sees the test set.
With ``refit = true`` a final run fits the chosen configuration on train and
validation (:func:`~seqrec_eval.splits.final_split`) and scores test only: its
validation score would be leaked. Trials are always fitted on the training
window, and a run given the wrong one refuses before it starts.

The ablations (:mod:`seqrec_eval.ablations`) reuse the same machinery. Their
runs carry a ``condition`` naming where they live, and come in three kinds: a
``final`` refitted on a transformed split, a ``reference`` that reloads a
stage-1 final model and scores it on the original split, and a ``rescore``
that reloads one and scores it on a split whose inference inputs were
transformed. All of them score only the sweep's fixed test users, skip
validation, which nothing selects on any more, and keep no model.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import random
import socket
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluate import evaluate_phase, other_definition, target_key
from .models import model_provenance, model_spec
from .protocol import Protocol
from .results import (
    evaluation_exists,
    load_evaluation,
    read_json,
    save_evaluation,
    write_json,
)
from .search import trial_params
from .splits import Split, code_provenance


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    model: str
    kind: str                     # "trial", "final", or (ablations only) "reference" / "rescore"
    index: int                    # trial number, or the seed of a final run
    seed: int
    params: dict[str, Any]
    fingerprint: str
    source_trial: int | None = None
    #: Set on an ablation run: its sweep, level and directory under the work dir.
    condition: dict[str, Any] | None = None
    #: The multiple of the searched configuration's epochs it trains (DECISIONS §42): 1 for a search trial, and for
    #: a final run of a configuration selected without extension.
    epochs_factor: int = 1

    @property
    def name(self) -> str:
        if self.kind != "trial":
            return f"final-seed{self.seed}"
        return f"trial-{self.index:03d}" + (f"-x{self.epochs_factor}" if self.epochs_factor != 1 else "")

    @property
    def selection(self) -> str:
        """How reports name the configuration: ``#7``, or ``#7 ×2`` when trained at twice its epochs."""
        return f"#{self.source_trial if self.kind != 'trial' else self.index}" + (
            f" ×{self.epochs_factor}" if self.epochs_factor != 1 else "")

    @property
    def label(self) -> str:
        """How the logs name the run. An ablation run's directory is also ``final-seed<s>``, which would read
        like a stage-1 final, so it is named by its sweep, condition and what it does instead."""
        if self.condition is None:
            return self.name
        condition = self.condition["label"]
        if self.condition.get("data_seed") is not None:
            condition += f"/seed{self.condition['data_seed']}"
        what = {"final": "refit", "reference": "reference (stage-1 model)", "rescore": "rescore (stage-1 model)"}
        return f"{self.condition['sweep']} {condition}: {what.get(self.kind, self.kind)} seed {self.seed}"

    def directory(self, work_dir: Path) -> Path:
        if self.condition is not None:
            return Path(work_dir) / self.condition["path"] / self.name
        return run_root(work_dir, self.dataset, self.model, self.fingerprint) / self.name


def run_root(work_dir: Path, dataset: str, model: str, fingerprint: str) -> Path:
    return Path(work_dir) / "runs" / dataset / model / fingerprint[:12]


# ---------------------------------------------------------------------------
# seeds added after the protocol's
# ---------------------------------------------------------------------------

ADDED_SEEDS = "added_seeds.json"


def added_seeds_path(work_dir: Path, dataset: str) -> Path:
    """The record of seeds added to ``dataset``'s final runs, beside its models' run directories.

    It is keyed by the dataset, not by a fingerprint, so the decision to give a dataset more seeds holds for
    its runs under a changed protocol too.
    """
    return Path(work_dir) / "runs" / dataset / ADDED_SEEDS


def seeds_with_added(protocol_seeds: tuple[int, ...], path: Path) -> tuple[int, ...]:
    """The protocol's seeds, then those recorded at ``path`` that the protocol does not already list."""
    added = [int(seed) for seed in read_json(path)["seeds"]] if path.exists() else []
    return tuple(protocol_seeds) + tuple(seed for seed in added if seed not in protocol_seeds)


def final_seeds(protocol: Protocol, work_dir: Path, dataset: str) -> tuple[int, ...]:
    """The seeds of ``dataset``'s final runs: the protocol's, then any added with ``final --add-seeds``."""
    return seeds_with_added(protocol.seeds, added_seeds_path(work_dir, dataset))


def record_added_seeds(path: Path, requested: list[int], present: tuple[int, ...]) -> list[int]:
    """Add the ``requested`` seeds that are not already ``present`` to the record at ``path``; return them.

    Asking again for a seed that is already there changes nothing, so a command that adds seeds can be
    rerun, or started once per GPU, like any other.
    """
    requested = [int(seed) for seed in requested]
    if len(set(requested)) != len(requested):
        raise ValueError(f"--add-seeds lists a seed twice: {requested}")
    if any(seed < 0 for seed in requested):
        raise ValueError(f"seeds must be non-negative integers (numpy's global seed is), got {requested}")
    new = [seed for seed in requested if seed not in present]
    if new:
        record = read_json(path) if path.exists() else {"seeds": [], "added": []}
        record["seeds"] += new
        record["added"].append({"seeds": new, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "host": socket.gethostname()})
        write_json(path, record)
    return new


def add_final_seeds(protocol: Protocol, work_dir: Path, dataset: str, seeds: list[int]) -> list[int]:
    """Record ``seeds`` as further final seeds of ``dataset``; returns the ones that were new."""
    return record_added_seeds(added_seeds_path(work_dir, dataset), seeds, final_seeds(protocol, work_dir, dataset))


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def plan_trials(protocol: Protocol, dataset: str, model: str) -> list[RunSpec]:
    """The search's trials, ``trial-000`` onwards; without the epoch extensions (:func:`plan_search`). None on a
    dataset the model does not run on (its ``datasets``, DECISIONS §45)."""
    mp = protocol.model(model)
    if not mp.runs_on(dataset) or mp.latency_only:
        return []  # a latency-only model is not searched (DECISIONS §48)
    fingerprint = protocol.run_fingerprint(dataset, model)
    return [
        RunSpec(dataset, model, "trial", trial, protocol.seeds[0],
                trial_params(protocol, mp, trial), fingerprint)
        for trial in range(mp.trials)
    ]


#: The parameter an epoch extension multiplies; a model without it is never extended.
EPOCHS = "epochs"
#: a run whose training loss fell by more than this share over the last tenth of its epochs was stopped while still
#: improving: the epoch grid (SASRec's capped at 100, DECISIONS §32) may hold the model back
STILL_IMPROVING = 0.01


def still_improving(history: list[dict[str, Any]] | None,
                    threshold: float = STILL_IMPROVING) -> tuple[float, int, int] | None:
    """``(fall, window, epochs)`` if the training loss fell by more than ``threshold`` (:data:`STILL_IMPROVING`)
    over the last ``window`` epochs (a tenth of them, at least one), else ``None``; also ``None`` without a loss
    per epoch (EASE, popularity). A model trained in phases is judged on its last phase."""
    if not history:
        return None
    phase = history[-1].get("phase")
    losses = [float(h["loss"]) for h in history if h.get("phase") == phase
              and isinstance(h.get("loss"), (int, float)) and math.isfinite(h["loss"])]
    if len(losses) < 2:
        return None
    window = max(1, round(0.1 * len(losses)))
    before, last = losses[-window - 1], losses[-1]
    fall = (before - last) / abs(before) if before else 0.0
    return (fall, window, len(losses)) if fall > threshold else None


def plan_extensions(protocol: Protocol, work_dir: Path, dataset: str, model: str,
                    base: TrialSummary | None = None) -> list[RunSpec]:
    """The epoch extensions of the search's selection (DECISIONS §42): its configuration at each multiple of its
    epochs in ``[protocol] epoch_extension``, as trials (training window, validation only).

    Planned only once the search is complete with nothing failed and open, and only if the selected trial was
    still improving at its last epoch (:func:`still_improving`, by more than ``[protocol]
    epoch_extension_min_fall``, DECISIONS §43): a rule fixed in advance, the same for every model, and decided on the
    training loss of a validation-selected trial, never on test.
    """
    if not protocol.epoch_extension:
        return []
    base = base if base is not None else _base_summary(protocol, work_dir, dataset, model)
    best = base.best
    if (best is None or base.unaccepted or base.done + base.skipped + base.failed < base.planned
            or EPOCHS not in best.params):
        return []
    done = best.directory(work_dir) / "done.json"
    threshold = STILL_IMPROVING if protocol.epoch_extension_min_fall is None else protocol.epoch_extension_min_fall
    if not still_improving(read_json(done).get("history"), threshold):
        return []
    return [replace(best, params={**best.params, EPOCHS: best.params[EPOCHS] * factor}, epochs_factor=factor)
            for factor in protocol.epoch_extension]


def plan_search(protocol: Protocol, work_dir: Path, dataset: str, model: str) -> list[RunSpec]:
    """Every run of the search: its trials, then the epoch extensions of its selection once they apply."""
    return plan_trials(protocol, dataset, model) + plan_extensions(protocol, work_dir, dataset, model)


@dataclass
class TrialSummary:
    planned: int
    done: int = 0
    skipped: int = 0
    failed: int = 0
    best: RunSpec | None = None
    best_value: float | None = None
    skip_reason: str | None = None
    #: every failed trial, ``{"trial": index, "error": message}``: a run that raised, or a non-finite score
    failures: list[dict[str, Any]] = field(default_factory=list)
    #: the failures accepted with ``final --accept-failed``, as recorded, or ``None``
    accepted: dict[str, Any] | None = None
    #: the epoch extensions planned (DECISIONS §42), and the selection before them with its validation value
    extensions: list[RunSpec] = field(default_factory=list)
    base_best: RunSpec | None = None
    base_value: float | None = None
    #: the validation value of each finished extension, by its multiple of the epochs
    extension_values: dict[int, float] = field(default_factory=dict)

    @property
    def unaccepted(self) -> list[dict[str, Any]]:
        accepted = set((self.accepted or {}).get("trials", []))
        return [failure for failure in self.failures if failure["trial"] not in accepted]


def accepted_failures_path(protocol: Protocol, work_dir: Path, dataset: str, model: str) -> Path:
    return run_root(work_dir, dataset, model, protocol.run_fingerprint(dataset, model)) / "accepted_failures.json"


def incomplete_selection_path(protocol: Protocol, work_dir: Path, dataset: str, model: str) -> Path:
    """Where ``final --allow-incomplete`` records that a configuration was selected before the search ended."""
    return run_root(work_dir, dataset, model, protocol.run_fingerprint(dataset, model)) / "incomplete_selection.json"


def _summarize(protocol: Protocol, work_dir: Path, trials: list[RunSpec],
               summary: TrialSummary | None = None) -> TrialSummary:
    """``summary`` (a new one if ``None``) with ``trials`` counted in, and its best updated: a later trial must
    score strictly higher, so a tie keeps the earlier -- the lower trial, the fewer epochs."""
    summary = summary if summary is not None else TrialSummary(planned=0)
    summary.planned += len(trials)
    for spec in trials:
        directory = spec.directory(work_dir)
        name = spec.index if spec.epochs_factor == 1 else f"{spec.index}x{spec.epochs_factor}"
        if (directory / "failed.json").exists() and not (directory / "done.json").exists():
            summary.failed += 1
            summary.failures.append({"trial": name, "error": read_json(directory / "failed.json")["error"]})
            continue
        if not (directory / "done.json").exists():
            continue
        record = read_json(directory / "done.json")
        if record["status"] == "skipped":
            summary.skipped += 1
            summary.skip_reason = record.get("reason")
            continue
        value = record["val"][protocol.primary_metric]
        if not np.isfinite(value):
            # a NaN can never be beaten by ">", so it would stay "best" for good (H07)
            summary.failed += 1
            summary.failures.append({"trial": name, "rerun_repeats_it": True,
                                     "error": f"validation {protocol.primary_metric} is not a finite number ({value})"})
            continue
        summary.done += 1
        if spec.epochs_factor != 1:
            summary.extension_values[spec.epochs_factor] = value
        if summary.best_value is None or value > summary.best_value:
            summary.best, summary.best_value = spec, value
    return summary


def _base_summary(protocol: Protocol, work_dir: Path, dataset: str, model: str) -> TrialSummary:
    """The search's trials alone, with the failures accepted for them.

    On a dataset the model does not run on, every trial counts as skipped, without a run: the report, ``status``
    and the final runs then treat it as they treat a model ``max_items`` keeps off a catalogue (DECISIONS §45).
    Nothing is written, so widening the model's ``datasets`` later finds no record in the way.
    """
    mp = protocol.model(model)
    if not mp.runs_on(dataset):
        return TrialSummary(planned=mp.trials, skipped=mp.trials, skip_reason=mp.not_run_reason(dataset))
    summary = TrialSummary(planned=0)
    accepted = accepted_failures_path(protocol, work_dir, dataset, model)
    if accepted.exists():
        summary.accepted = read_json(accepted)
    return _summarize(protocol, work_dir, plan_trials(protocol, dataset, model), summary)


def summarize_trials(protocol: Protocol, work_dir: Path, dataset: str, model: str) -> TrialSummary:
    """Finished trials, and the best by validation primary metric (ties: lowest trial), counting the epoch
    extensions of the selection once they apply (:func:`plan_extensions`).

    A trial that raised, or whose validation score is not a finite number, is
    failed: it is listed, never selected, and blocks selection until it is
    rerun or accepted (:func:`plan_finals`).
    """
    summary = _base_summary(protocol, work_dir, dataset, model)
    summary.base_best, summary.base_value = summary.best, summary.best_value
    summary.extensions = plan_extensions(protocol, work_dir, dataset, model, base=summary)
    return _summarize(protocol, work_dir, summary.extensions, summary)


def plan_finals(protocol: Protocol, work_dir: Path, dataset: str, model: str, *,
                allow_incomplete: bool = False, accept_failed: bool = False) -> list[RunSpec]:
    """The final runs of the selected configuration, one per seed of the dataset (:func:`final_seeds`), once
    the search is complete and has no open failure.

    No model is selected while one of its trials has failed: a failure is a
    bug or a configuration that does not run, and selecting around it would
    give the model a smaller, silently different search. Fix the cause and
    rerun (``search --retry-failed``). Only a failure that cannot be fixed --
    a configuration too large for the hardware, say -- may be accepted with
    ``accept_failed``: it is then recorded in ``accepted_failures.json`` and
    listed in the report.
    """
    mp = protocol.model(model)
    if mp.latency_only:
        # one final of its fixed configuration, at the first seed: the one latency times (DECISIONS §48)
        if not mp.runs_on(dataset):
            return []
        seed = protocol.seeds[0]
        return [RunSpec(dataset, model, "final", seed, seed, dict(mp.fixed), protocol.run_fingerprint(dataset, model))]
    summary = summarize_trials(protocol, work_dir, dataset, model)
    if summary.skipped == summary.planned:
        return []
    open_failures = summary.unaccepted
    if open_failures and not accept_failed:
        shown = "; ".join(f"trial {f['trial']}: {f['error'][:120]}" for f in open_failures[:3])
        repeats = [f["trial"] for f in open_failures if f.get("rerun_repeats_it")]
        raise RuntimeError(
            f"{dataset}/{model}: {len(open_failures)} trial(s) failed ({shown}). No model is selected while a trial "
            "has failed: fix the cause and rerun them (`search --retry-failed`), or, if they cannot run at all "
            "(for example out of memory at the corner of the search space), accept them with "
            "`final --accept-failed`, which records them for the report."
            + (f" Trial(s) {repeats} finished with a non-finite score: a rerun uses the same seed and gives the "
               "same score, so `--retry-failed` does not help; accept them, or narrow the search space."
               if repeats else "")
        )
    if open_failures:
        path = accepted_failures_path(protocol, work_dir, dataset, model)
        write_json(path, {"trials": sorted((f["trial"] for f in summary.failures), key=str),
                          "failures": summary.failures,
                          "accepted_at": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname()})
    finished = summary.done + summary.skipped + summary.failed
    if finished < summary.planned and not allow_incomplete:
        raise RuntimeError(
            f"{dataset}/{model}: {finished}/{summary.planned} trials finished. Selecting before the "
            "search is complete would give this model a smaller budget than the others (§5.2); "
            "wait for it, or pass --allow-incomplete knowingly."
        )
    if summary.best is None:
        raise RuntimeError(f"{dataset}/{model}: no trial finished successfully, so there is nothing to select")
    best = summary.best
    if finished < summary.planned:
        # recorded, so the report says the budget was smaller (review B3); if the search later picks another
        # trial, the finals made now are refused as stale (see execute)
        write_json(incomplete_selection_path(protocol, work_dir, dataset, model),
                   {"finished": finished, "planned": summary.planned, "selected_trial": best.index,
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname()})
    return [
        RunSpec(dataset, model, "final", seed, seed, best.params, best.fingerprint, source_trial=best.index,
                epochs_factor=best.epochs_factor)
        for seed in final_seeds(protocol, work_dir, dataset)
    ]


# ---------------------------------------------------------------------------
# locking
# ---------------------------------------------------------------------------

#: How many times a run may be started by a process that then dies without a word before it counts as failed.
MAX_ATTEMPTS = 2

#: Open descriptors of the locks this process holds, by path: a lock lives as long as its descriptor, and
#: outlives the RunLock object that took it until ``release``.
_HELD: dict[Path, int] = {}


class RunLock:
    """A run's claim, as an exclusive ``flock`` on ``<run>/.lock``.

    The kernel holds the lock for the process and drops it the moment the process ends, however it ends (a
    crash, an out-of-memory kill, a reboot). So a claim is never stale, there are no process ids to trust --
    after a reboot a dead owner's id can belong to any live process, which once kept a run claimed for ever --
    and two processes cannot both reclaim one (final review B2). It needs a local file system, as ``/raid`` and
    the laptop's are; the file itself is left in place, with its last owner written in it for people.
    """

    def __init__(self, directory: Path) -> None:
        self.path = Path(directory) / ".lock"

    def acquire(self) -> bool:
        if self.path in _HELD:
            return False  # held by this process already
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        for attempt in range(3):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if attempt == 2:
                    os.close(descriptor)
                    return False
                time.sleep(0.05)  # a `status` probe holds it for an instant; an owner holds it for the whole run
        os.ftruncate(descriptor, 0)
        os.write(descriptor, json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                                         "started": time.time()}).encode())
        _HELD[self.path] = descriptor
        return True

    def release(self) -> None:
        descriptor = _HELD.pop(self.path, None)
        if descriptor is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def held_by_other(self) -> bool:
        """Whether another process holds the run now (this process's own claims count too)."""
        if self.path in _HELD:
            return True
        if not self.path.exists():
            return False
        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            os.close(descriptor)  # closing drops the probe's own shared lock
        return False


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _check_condition(spec: RunSpec, split: Split) -> None:
    """Refuse to run an ablation spec on any split but its own condition's."""
    if spec.condition is None:
        if split.condition is not None:
            raise ValueError(f"{spec.name} is a stage-1 run but the split is ablation condition {split.condition}")
        return
    wanted = {key: spec.condition[key] for key in ("sweep", "label", "data_seed")}
    have = None if split.condition is None else {key: split.condition[key] for key in wanted}
    if have != wanted or split.test_rows is None:
        raise ValueError(f"{spec.name} belongs to condition {wanted}, but the split is {have}")


def _check_training(spec: RunSpec, split: Split, protocol: Protocol) -> None:
    """Trials fit on the training window; everything scored on test on :func:`final_split`'s data."""
    wanted = "train+val" if protocol.refit and spec.kind != "trial" else "train"
    if split.trained_on != wanted:
        raise ValueError(f"{spec.name} of {spec.dataset}/{spec.model} must be given a split trained on {wanted}, "
                         f"not {split.trained_on}" + (" (pass final_split(protocol, split))" if wanted != "train"
                                                      else ""))


def _execute(spec: RunSpec, split: Split, protocol: Protocol, directory: Path, device: str,
             work_dir: Path) -> str:
    mp = protocol.model(spec.model)
    dp = protocol.dataset(spec.dataset)
    registered = model_spec(spec.model)
    if registered.family != mp.family:
        raise ValueError(f"{spec.model} is registered as {registered.family} but the protocol says {mp.family}")
    _check_condition(spec, split)
    ablation = spec.condition is not None

    train_item_ids = split.data["train_item_ids"]
    record: dict[str, Any] = {"spec": asdict(spec), "fingerprint": spec.fingerprint,
                              "host": socket.gethostname(), "device": device,
                              "trained_on": split.trained_on, "train_items": len(train_item_ids),
                              "code": {**code_provenance(), **model_provenance(spec.model)}}
    # Under refit the final runs fit the validation catalogue, which is larger: a trial decides on that one
    # too, or a model could be searched in full and then skipped at the final runs.
    catalogue = (len(split.data["val_item_ids"]) if protocol.refit and split.trained_on == "train"
                 else len(train_item_ids))
    if mp.max_items is not None and catalogue > mp.max_items:
        record.update(status="skipped",
                      reason=f"{catalogue} items in the catalogue the final runs fit exceed max_items={mp.max_items}")
        write_json(directory / "done.json", record)
        return "skipped"

    cuda = _reset_peak_memory(device)
    if spec.kind in ("reference", "rescore"):
        checkpoint = Path(work_dir) / spec.condition["checkpoint"]
        if not checkpoint.exists():
            raise FileNotFoundError(f"{spec.kind} reloads stage 1's final model, but {checkpoint} does not exist")
        trainer = registered.cls.load(checkpoint, device=device)
        record["loaded_from"] = spec.condition["checkpoint"]
    else:
        _seed_everything(spec.seed)
        trainer = registered.build(spec.params, n_items=len(train_item_ids), device=device, seed=spec.seed)
        data = split.data["x_train_sequences"] if mp.family == "sequence" else split.data["x_train"]
        if mp.latency_only:
            # the whole catalogue and the architecture, which is what a request's time depends on, fitted on a
            # few users only: its weights mean nothing, so it is never scored (DECISIONS §48)
            rows = np.arange(min(LATENCY_ONLY_ROWS, _n_rows(data)), dtype=np.int64)
            data = data.select_rows(rows) if hasattr(data, "select_rows") else data[rows]
            record["latency_only"] = {"fit_rows": int(rows.size)}
        started = time.perf_counter()
        trainer.fit(data, item_ids=train_item_ids)
        record["fit_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    if mp.latency_only:
        _save_model(trainer, directory, record)
    elif not ablation and split.trained_on == "train":
        val = evaluate_phase(trainer, split, "val", family=mp.family, protocol=protocol,
                             exclude_seen=dp.exclude_seen, rows=split.val_rows)
        save_evaluation(val, directory / "val")
        record["val"] = dict(val.metrics)

    if spec.kind in ("final", "reference", "rescore") and not mp.latency_only:
        test = evaluate_phase(trainer, split, "test", family=mp.family, protocol=protocol,
                              exclude_seen=dp.exclude_seen, rows=split.test_rows)
        save_evaluation(test, directory / "test")
        record["test"] = dict(test.metrics)
        if ablation and dp.new_item_diagnostic:
            # the next items the user never had, at every level: what a model adds beyond what the user already had
            # (DECISIONS §38). Seen items are judged on the whole history, so the targets are the same at every
            # level of a sweep that keeps the item space, and so are the users scored on them.
            new = evaluate_phase(trainer, split, "test", family=mp.family, protocol=protocol,
                                 exclude_seen=True, rows=split.test_rows, targets="new")
            save_evaluation(new, directory / "test_new")
            record["test_new"] = dict(new.metrics)
    if spec.kind == "final" and not ablation and not mp.latency_only:
        other = other_definition(protocol.targets)
        if split.data.get(target_key("test", other)) is not None:
            # the target definition the protocol did not choose, as a diagnostic beside it
            diagnostic = evaluate_phase(trainer, split, "test", family=mp.family, protocol=protocol,
                                        exclude_seen=dp.exclude_seen, targets=other)
            save_evaluation(diagnostic, directory / f"test_{other}")
            record[f"test_{other}"] = dict(diagnostic.metrics)
        if dp.new_item_diagnostic:
            new = evaluate_phase(trainer, split, "test", family=mp.family, protocol=protocol,
                                 exclude_seen=True, targets="new")
            save_evaluation(new, directory / "test_new")
            record["test_new"] = dict(new.metrics)
        _save_model(trainer, directory, record)
    record["eval_seconds"] = time.perf_counter() - started

    history = getattr(trainer, "history", None)
    if history:
        record["history"] = list(history)
    diagnostics = getattr(trainer, "run_diagnostics", None)
    if callable(diagnostics):
        # what a model counts about its own run, over every scoring in it (a plugin's, say: DECISIONS §39)
        record["model_diagnostics"] = diagnostics()
    if cuda:
        record["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated(device))
    record["status"] = "done"
    write_json(directory / "done.json", record)
    return "done"


#: how many training users a latency-only model is fitted on (DECISIONS §48)
LATENCY_ONLY_ROWS = 256


def _n_rows(data) -> int:
    return int(data.n_rows) if hasattr(data, "n_rows") else int(data.shape[0])


def _save_model(trainer, directory: Path, record: dict[str, Any]) -> None:
    try:
        trainer.save(directory / "model.zip")
        record["model_saved"] = True
    except Exception as error:  # latency needs it, but a result without it is still a result  # noqa: BLE001
        record["model_saved"] = False
        record["model_save_error"] = f"{type(error).__name__}: {error}"


def _reset_peak_memory(device: str) -> bool:
    """Start counting the run's peak GPU memory; ``False`` on the CPU.

    CUDA is started first: until it has started, torch's memory counters refuse an explicit device
    (``cuda:0``) though not plain ``cuda``, so on the first DGX run every run failed at once with "Invalid
    device argument" (2026-10-01). The tests ran on ``cuda`` and did not see it.
    """
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return False
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    return True


def made_with_another_selection(spec: RunSpec, directory: Path) -> str | None:
    """Why the finished run in ``directory`` does not belong to ``spec``'s selection, or ``None`` if it does.

    A final run (and every ablation run built on one) uses the configuration selected when it was planned. If
    the selection changes afterwards -- a failed trial accepted, then rerun and found best; a search finished
    after ``--allow-incomplete`` -- the directory is the same, so the old run would pass for the new
    selection's (final review A3). It is refused instead, and must be moved aside to be redone.
    """
    if spec.kind == "trial" or not (directory / "spec.json").exists():
        return None
    made = read_json(directory / "spec.json")
    if made.get("source_trial") == spec.source_trial and made.get("params") == spec.params:
        return None

    def named(trial, factor) -> str:
        return f"trial {trial}" + (f" ×{factor}" if factor and factor != 1 else "")

    return (f"Made with {named(made.get('source_trial'), made.get('epochs_factor'))}'s configuration, but "
            f"{named(spec.source_trial, spec.epochs_factor)} is selected now, so its results describe another "
            f"model. Move {directory} aside to redo it with the selection.")


def differs_in_epochs_only(spec: RunSpec, directory: Path) -> bool:
    """Whether the stage-1 final in ``directory`` was made with ``spec``'s configuration at another number of
    epochs: the selection moved by an epoch extension (DECISIONS §42), not to another configuration."""
    if spec.kind != "final" or spec.condition is not None or not (directory / "spec.json").exists():
        return False
    made = read_json(directory / "spec.json")
    params = made.get("params") or {}
    return (made.get("source_trial") == spec.source_trial and params.get(EPOCHS) != spec.params.get(EPOCHS)
            and {k: v for k, v in params.items() if k != EPOCHS}
            == {k: v for k, v in spec.params.items() if k != EPOCHS})


def _supersede(spec: RunSpec, directory: Path, log=print) -> bool:
    """Move a stage-1 final made at another number of epochs aside (``final-seed0.superseded-<time>``), keeping
    its records but not its model file, so it is redone with the selection; ``False`` if another process holds it.
    """
    lock = RunLock(directory)
    if not lock.acquire():
        return False
    try:
        if not ((directory / "done.json").exists() and differs_in_epochs_only(spec, directory)):
            return True  # moved aside already, by another process
        target = directory.with_name(f"{directory.name}.superseded-{time.strftime('%Y%m%d-%H%M%S')}")
        directory.rename(target)
        (target / "model.zip").unlink(missing_ok=True)  # disk: the records stay, the superseded model goes
    finally:
        lock.release()
    log(f"[{spec.dataset}/{spec.model}] {spec.label}: made at other epochs than the selection "
        f"({spec.selection}, DECISIONS §42); moved aside to {target.name}, without its model, and redone")
    return True


def execute(spec: RunSpec, split: Split, protocol: Protocol, work_dir: Path, *,
            device: str, retry_failed: bool = False, log=print) -> str:
    """Run ``spec`` unless it is finished, failed before, or claimed elsewhere."""
    _check_training(spec, split, protocol)
    mp = protocol.model(spec.model)
    if not mp.runs_on(spec.dataset):
        # never planned (plan_trials, plan_finals return none): refused before anything is written, so no record of
        # it can outlive the model's datasets (DECISIONS §45)
        raise ValueError(f"{spec.name} of {spec.dataset}/{spec.model}: {mp.not_run_reason(spec.dataset)}")
    directory = spec.directory(work_dir)
    if (directory / "done.json").exists() and differs_in_epochs_only(spec, directory):
        if not _supersede(spec, directory, log=log):
            return "running-elsewhere"
    if (directory / "done.json").exists():
        stale = made_with_another_selection(spec, directory)
        if stale:
            log(f"[{spec.dataset}/{spec.model}] {spec.label}: {stale}")
            return "stale-selection"
        return "cached"
    if (directory / "failed.json").exists() and not retry_failed:
        return "failed-before"
    if spec.kind in ("reference", "rescore"):
        stage1 = (Path(work_dir) / spec.condition["checkpoint"]).parent
        if (stage1 / "failed.json").exists() and not (stage1 / "done.json").exists():
            log(f"[{spec.dataset}/{spec.model}] {spec.label}: the stage-1 final it reloads failed; rerun it "
                "(`final --retry-failed`) first")
            return "stage1-failed"
        if not (stage1 / "done.json").exists():
            # the stage-1 final it reloads is not made yet -- a seed added to stage 1 and to a sweep at once, run
            # on two GPUs, say. That is an order to wait for, not a failure to record: a later run picks it up.
            return "waiting-for-stage1"
        stale = made_with_another_selection(spec, stage1)
        if stale:
            log(f"[{spec.dataset}/{spec.model}] {spec.label}: the stage-1 final it reloads {stale[0].lower()}"
                f"{stale[1:]}")
            return "stale-selection"
    lock = RunLock(directory)
    if not lock.acquire():
        return "running-elsewhere"
    if (directory / "done.json").exists():
        # finished by another process between the check above and the claim (H45)
        lock.release()
        return "cached"
    # A process killed while running it (the kernel's out-of-memory killer, SIGKILL) records nothing, so every
    # restart would run it first, for ever; after MAX_ATTEMPTS starts it is a failure like any other (review B3).
    attempts_path = directory / "attempts.json"
    # --retry-failed asks for another try: earlier deaths no longer count (review N31)
    attempts = int(read_json(attempts_path).get("started", 0)) if attempts_path.exists() and not retry_failed else 0
    if attempts >= MAX_ATTEMPTS:
        write_json(directory / "failed.json", {
            "spec": asdict(spec), "host": socket.gethostname(), "device": device,
            "error": (f"ProcessDied: the process running it ended {attempts} times without finishing or "
                      "failing it -- killed, most likely out of memory (the kernel's OOM killer) or by a signal"),
        })
        attempts_path.unlink(missing_ok=True)
        lock.release()
        log(f"[{spec.dataset}/{spec.model}] {spec.label} FAILED: its process died {attempts} times")
        return "failed"
    write_json(attempts_path, {"started": attempts + 1, "last": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "host": socket.gethostname()})
    try:
        write_json(directory / "spec.json", asdict(spec))
        trains = spec.kind in ("trial", "final")  # a reference or rescore reloads a stage-1 model
        log(f"[{spec.dataset}/{spec.model}] {spec.label} on {device}"
            + (f": {json.dumps(spec.params, sort_keys=True)}" if trains else ""))
        started = time.perf_counter()
        status = _execute(spec, split, protocol, directory, device, work_dir)
        (directory / "failed.json").unlink(missing_ok=True)
        attempts_path.unlink(missing_ok=True)
        log(f"[{spec.dataset}/{spec.model}] {spec.label} {status} in {time.perf_counter() - started:,.0f}s")
        return status
    except KeyboardInterrupt:
        attempts_path.unlink(missing_ok=True)  # stopped by the operator, not a death
        raise
    except Exception as error:  # noqa: BLE001
        write_json(directory / "failed.json", {
            "spec": asdict(spec), "host": socket.gethostname(), "device": device,
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
        })
        attempts_path.unlink(missing_ok=True)
        log(f"[{spec.dataset}/{spec.model}] {spec.label} FAILED: {type(error).__name__}: {error}")
        return "failed"
    finally:
        lock.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def stale_final_seeds(protocol: Protocol, work_dir: Path, dataset: str, model: str) -> list[int]:
    """Seeds whose finished final was made with another trial's configuration than the one selected now.

    Such a final describes another model (see :func:`made_with_another_selection`): every report leaves it out
    of its values and comparisons, and latency is not measured on it (review N33). Empty while no trial can be
    selected.
    """
    best = summarize_trials(protocol, work_dir, dataset, model).best
    if best is None:
        return []
    root = run_root(work_dir, dataset, model, protocol.run_fingerprint(dataset, model))
    stale = []
    for seed in final_seeds(protocol, work_dir, dataset):
        directory = root / f"final-seed{seed}"
        if (directory / "done.json").exists() and (directory / "spec.json").exists():
            made = read_json(directory / "spec.json")
            # the configuration, epochs included: an epoch extension keeps the trial but changes the epochs
            if made.get("source_trial") != best.index or made.get("params") != best.params:
                stale.append(seed)
    return stale


def load_final_evaluations(protocol: Protocol, work_dir: Path, dataset: str, model: str,
                           stem: str = "test") -> list[tuple[RunSpec, Any]]:
    """Finished final runs of the selected configuration, with one saved evaluation each. A final made with
    another trial's configuration than the one selected now is left out (:func:`stale_final_seeds`)."""
    fingerprint = protocol.run_fingerprint(dataset, model)
    root = run_root(work_dir, dataset, model, fingerprint)
    stale = stale_final_seeds(protocol, work_dir, dataset, model)
    out = []
    for seed in final_seeds(protocol, work_dir, dataset):
        directory = root / f"final-seed{seed}"
        if seed in stale or not (directory / "done.json").exists() or not evaluation_exists(directory / stem):
            continue
        spec = RunSpec(**read_json(directory / "spec.json"))
        out.append((spec, load_evaluation(directory / stem)))
    return out
