"""Data ablations: one data characteristic varied, everything else held fixed.

A sweep is a transform applied to a prepared split at each of its levels. The
stage-1 configuration of every model is used at every level under every seed
and scored on test; nothing is searched again, so a difference between levels
is the data's, not the tuning's. Four rules make levels comparable.

* **Only inputs change.** A transform edits the training data and the input
  histories of validation and test users. Test targets stay as they are, except
  where the characteristic itself is the catalogue. Under ``refit`` the
  training data is train+validation (:func:`~seqrec_eval.splits.final_split`),
  as in the stage-1 final runs the reference reloads.
* **One set of test users** -- except where the targets change. Every level
  of a sweep, and the full-data reference, is scored on the test users that are
  still eligible at every level, in practice those who survive the most
  extreme one. A sweep that removes targets (the catalogue sweep) cannot keep
  one set: a user whose next target is a known item survives every random
  catalogue only by chance, so the set would shrink to users whose targets are
  all unseen, who score 0 for every model (H01b). There, each condition is
  scored on **its own users** -- a known next target still in the catalogue,
  and a history item left -- and models are compared within a condition.
* **The reference is not refitted.** "full" is the stage-1 final model of each
  seed, reloaded and scored on those users, so the reference is exactly the
  model stage 1 reported.
* **The manipulation is measured.** The five characteristics are measured on
  every condition, so the report can show whether the transform moved only
  what it targets and what it is declared to move along with it.

A sweep has a **scope**. ``"all"`` edits training data and inference inputs
alike, and every condition refits the model. ``"inference"`` leaves training
data untouched and edits only the histories a model is given when it
recommends, so a condition just rescores the stage-1 models: nothing is
refitted, and the question becomes what the model needs at serving time rather
than what it needs to learn from.

Stochastic transforms draw one subsample per seed, from a stream of their own,
and each phase is subsampled independently: the split carries no event identity
across phases, so the same event can be kept in a test history and dropped from
training. Both still hold the same regime. Subsample s is always fitted with
model seed s, so a stochastic condition has one run per seed and its seed spread
holds both the draw and the training.

A sweep's seeds on a dataset (:func:`sweep_seeds`) are the protocol's, then any
added with ``ablate --add-seeds``, recorded per sweep and dataset because the
subsamples, the fixed test users, the manipulation check and the analysis's
floor are shared by every model of the sweep. An added seed must already be a
stage-1 seed of the dataset: its reference is that seed's final model. For a
stochastic sweep it adds a subsample at every level, and the fixed test users
must survive it (see :func:`fixed_test_rows`).

Everything is stored under a fingerprint of what determines it. The condition
fingerprint covers the dataset's split, the evaluation key (target definition,
refit, scoring code), and the sweep's transform, levels, options and scope; a run's fingerprint adds its stage-1 run
fingerprint, so a change to either starts fresh runs, as in stage 1.

    ablations/<sweep>/<dataset>/added_seeds.json                  seeds added with ``ablate --add-seeds``
    ablations/<sweep>/<dataset>/<cfp[:12]>/test_rows.npy
    ablations/<sweep>/<dataset>/<cfp[:12]>/test_rows.json        the subsamples those users were checked on
    ablations/<sweep>/<dataset>/<cfp[:12]>/conditions/<level>[/seedS].json   characteristics
    ablations/<sweep>/<dataset>/<cfp[:12]>/runs/<model>/<fp[:12]>/<level>/final-seedS/

To add a transform, register a function that returns the transformed data dict
and declare what it does (see :func:`register`). Transforms that only drop
events build masks and hand them to :func:`keep_events`, which rebuilds every
matrix exactly as the library's builder would have from the kept events.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from compresso_recsys import ItemSequences
from scipy.sparse import csr_matrix

from .evaluate import fills_list, reachable_targets, recommendable_next, target_key
from .protocol import AblationProtocol, Protocol, ProtocolError
from .results import durable_replace, read_json, write_json
from .runner import RunSpec, final_seeds, plan_finals, record_added_seeds, run_root, seeds_with_added
from .search import _stream
from .splits import Split
from .timestamps import TIMESTAMP_KEYS

#: Label of the full-data reference condition; no level may use it.
REFERENCE = "full"
CHARACTERISTICS = ("history_length", "catalogue", "density", "popularity_gini", "repeat_rate")
SCOPES = ("all", "inference")
PHASES = ("train", "val", "test")
#: Relative change beyond which an unexpected characteristic is flagged.
DEFAULT_TOLERANCE = 0.05
#: The knee's margin δ, as a fraction of the model's value on the full data (DECISIONS.md §18).
DEFAULT_KNEE_MARGIN = 0.10
#: A level scored on fewer users is reported as descriptive only.
DEFAULT_MIN_LEVEL_USERS = 1000
#: Keys of an ``[ablations.*]`` section that change results. ``datasets`` and
#: ``models`` only choose what runs; ``expected_to_move``, the tolerance, the
#: knee margin and ``min_level_users`` only change how the report reads the runs.
_RESULT_KEYS = ("transform", "levels", "options", "scope")
#: Private key of a transformed data dict: the event masks it was built from.
KEPT = "_kept"


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# the transform registry
# ---------------------------------------------------------------------------

#: ``apply(data, level, *, rng, options, value, phases) -> data``. ``rng`` is
#: ``None`` for a deterministic transform, ``value`` the dataset's per-event
#: value, and ``phases`` the phases the sweep's scope allows it to edit.
Apply = Callable[..., dict[str, Any]]
Options = dict[str, Any]


@dataclass(frozen=True)
class Transform:
    name: str
    #: Bump when the transform's behaviour changes; it enters every fingerprint.
    version: int
    #: Whether the transform draws a subsample, given the sweep's options.
    stochastic: Callable[[Options], bool]
    #: The characteristic the transform is meant to move; ``None`` for a
    #: transform that should move none of them (shuffling order).
    target: str | None
    #: What else the transform moves by construction, given its options.
    expected: Callable[[Options], tuple[str, ...]]
    validate: Callable[[tuple[Any, ...], Options], None]
    apply: Apply
    scopes: tuple[str, ...]
    #: Whether test targets change between levels; see the ablation report.
    changes_targets: bool = False
    #: Each level's place on the knee's axis, larger meaning closer to the full
    #: data (the reference sits at infinity); ``None`` where no knee applies.
    position: Callable[[Any], float] | None = None
    #: The way the target must move, -1 down or +1 up. A condition that moves it
    #: the other way is not a finding but a sign of a bug (see the report).
    direction: int = -1
    #: Whether the levels of one seed share one random draw, so that a smaller
    #: level is part of every larger one (the stream then ignores the level).
    nested: bool = False


TRANSFORMS: dict[str, Transform] = {}


def register(name: str, *, version: int, stochastic: bool | Callable[[Options], bool], target: str | None,
             expected: tuple[str, ...] | Callable[[Options], tuple[str, ...]] = (),
             validate: Callable, scopes: tuple[str, ...] = SCOPES, changes_targets: bool = False,
             position: Callable | None = None, direction: int = -1, nested: bool = False
             ) -> Callable[[Apply], Apply]:
    stochastic_of = stochastic if callable(stochastic) else (lambda options, value=bool(stochastic): value)
    expected_of = expected if callable(expected) else (lambda options, value=tuple(expected): value)
    if target is not None and target not in CHARACTERISTICS:
        raise ValueError(f"transform {name!r} targets {target!r}, not one of {CHARACTERISTICS}")

    def decorate(apply: Apply) -> Apply:
        if name in TRANSFORMS:
            raise ValueError(f"transform {name!r} is registered twice")
        TRANSFORMS[name] = Transform(name, version, stochastic_of, target, expected_of, validate, apply,
                                     tuple(scopes), changes_targets, position, direction, nested)
        return apply
    return decorate


def transform_of(ablation: AblationProtocol) -> Transform:
    if ablation.transform not in TRANSFORMS:
        raise ProtocolError(f"[ablations.{ablation.name}] uses unknown transform {ablation.transform!r}; "
                            f"registered: {sorted(TRANSFORMS)}")
    return TRANSFORMS[ablation.transform]


def scope_of(ablation: AblationProtocol) -> str:
    return str(ablation.raw.get("scope", "all"))


def _is_declaration(name: str) -> bool:
    """A characteristic, alone (both parts) or with the part it applies to: ``"catalogue"``, ``"test catalogue"``."""
    words = name.split()
    return (len(words) == 1 and words[0] in CHARACTERISTICS) or \
        (len(words) == 2 and words[0] in ("train", "test") and words[1] in CHARACTERISTICS)


def expected_to_move(ablation: AblationProtocol) -> tuple[str, ...]:
    """The characteristics the sweep declares it moves besides its target."""
    declared = ablation.raw.get("expected_to_move")
    return tuple(declared) if declared is not None else transform_of(ablation).expected(ablation.options)


def check_ablation(ablation: AblationProtocol) -> Transform:
    """The sweep's transform, after it has accepted the sweep's levels and settings."""
    where = f"[ablations.{ablation.name}]"
    transform = transform_of(ablation)
    if any(level_label(level) == REFERENCE for level in ablation.levels):
        raise ProtocolError(f"{where}: {REFERENCE!r} is reserved for the reference")
    transform.validate(ablation.levels, ablation.options)
    scope = scope_of(ablation)
    if scope not in transform.scopes:
        raise ProtocolError(f"{where}.scope must be one of {transform.scopes} for {transform.name}, got {scope!r}")
    declared = ablation.raw.get("expected_to_move")
    if declared is not None:
        if not isinstance(declared, list) or not all(isinstance(name, str) and _is_declaration(name)
                                                     for name in declared):
            raise ProtocolError(f"{where}.expected_to_move must list names from {CHARACTERISTICS}, each alone or "
                                "with the part it applies to (\"test catalogue\", \"train density\")")
        if any(name.split()[-1] == transform.target for name in declared):
            raise ProtocolError(f"{where}.expected_to_move names the target {transform.target!r} itself")
    tolerance = ablation.raw.get("manipulation_tolerance", DEFAULT_TOLERANCE)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise ProtocolError(f"{where}.manipulation_tolerance must be a non-negative number")
    margin = ablation.raw.get("knee_margin", DEFAULT_KNEE_MARGIN)
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 < margin < 1:
        raise ProtocolError(f"{where}.knee_margin must be a fraction in (0, 1)")
    minimum = ablation.raw.get("min_level_users", DEFAULT_MIN_LEVEL_USERS)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ProtocolError(f"{where}.min_level_users must be a positive integer")
    return transform


def level_label(level: Any) -> str:
    return str(level).replace("/", "_").replace(" ", "")


# ---------------------------------------------------------------------------
# event-level helpers
# ---------------------------------------------------------------------------

def _event_rows(sequences) -> np.ndarray:
    """The row of every event, in storage order."""
    return np.repeat(np.arange(sequences.n_rows, dtype=np.int64), sequences.row_lengths)


def _positions(sequences) -> np.ndarray:
    """Each event's position within its row, oldest first."""
    return np.arange(sequences.values.size, dtype=np.int64) - np.repeat(sequences.indptr[:-1], sequences.row_lengths)


def _indptr(lengths: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)


def _subset(sequences, keep: np.ndarray, *, values: np.ndarray | None = None, n_items: int | None = None):
    """The kept events, rows intact; ``values`` and ``n_items`` re-address them in a new item space."""
    lengths = np.bincount(_event_rows(sequences)[keep], minlength=sequences.n_rows)
    return ItemSequences(values=(sequences.values if values is None else values)[keep], indptr=_indptr(lengths),
                         n_items=sequences.n_items if n_items is None else n_items)


def _carry_times(out: dict[str, Any], data: dict[str, Any], view: str, *, keep: np.ndarray | None = None,
                 order: np.ndarray | None = None) -> None:
    """Keep the times of ``view`` with its events: the same mask, or the same reordering.

    A split prepared without timestamps has none to carry, and stays without.
    """
    times = data.get(TIMESTAMP_KEYS[view])
    if times is None:
        return
    if keep is not None:
        times = times[keep]
    if order is not None:
        times = times[order]
    out[TIMESTAMP_KEYS[view]] = times


def _random_keys(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.random(n)


def _rank_within_rows(sequences, keys: np.ndarray) -> np.ndarray:
    """Each event's rank within its row when the row is ordered by ``keys``."""
    order = np.lexsort((keys, _event_rows(sequences)))
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size) - np.repeat(sequences.indptr[:-1], sequences.row_lengths)
    return rank


def _first_occurrence(sequences) -> np.ndarray:
    """Whether each event is the first occurrence of its item in its row."""
    keys = _event_rows(sequences) * np.int64(max(sequences.n_items, 1)) + sequences.values
    first = np.zeros(keys.size, dtype=bool)
    first[np.unique(keys, return_index=True)[1]] = True
    return first


def count_matrix(sequences, value: float, keep: np.ndarray | None = None) -> csr_matrix:
    """The builder's matrix view of a phase: each event's value, summed per pair."""
    rows, cols = _event_rows(sequences), sequences.values
    if keep is not None:
        rows, cols = rows[keep], cols[keep]
    matrix = csr_matrix((np.full(rows.size, value, dtype=np.float32), (rows, cols)),
                        shape=(sequences.n_rows, sequences.n_items), dtype=np.float32)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def keep_events(data: dict[str, Any], keep: dict[str, np.ndarray], *, value: float) -> dict[str, Any]:
    """Drop events, and rebuild every view of each phase from those that remain.

    ``keep`` maps ``"train"``, ``"val"`` and ``"test"`` to a mask over the events
    of ``x_train_sequences`` and of each phase's source sequences; a phase not
    named is left alone. The masks are kept under :data:`KEPT` for the
    manipulation check.

    The training window is the train stage's source followed by its target, and
    the builder's ``x_train`` is ``source.maximum(target)`` over the two, not a
    count over the window. The boundary is where the train-stage source ends,
    so both matrices, and their maximum, are rebuilt from the events kept on
    each side of it.
    """
    out = dict(data)
    if "train" in keep:
        window, source = data["x_train_sequences"], data["train_source_sequences"]
        mask = keep["train"]
        if window.n_rows != source.n_rows:
            raise ValueError("x_train_sequences and train_source_sequences disagree on rows")
        is_source = _positions(window) < np.repeat(source.row_lengths, window.row_lengths)
        source_matrix = count_matrix(window, value, mask & is_source)
        target_matrix = count_matrix(window, value, mask & ~is_source)
        out.update(
            x_train_sequences=_subset(window, mask),
            train_source_sequences=_subset(window, mask & is_source),
            train_source_matrix=source_matrix,
            train_target_matrix=target_matrix,
            x_train=source_matrix.maximum(target_matrix).tocsr(),
        )
        _carry_times(out, data, "x_train_sequences", keep=mask)
        # the train-stage source is the start of the window, so its times are the window's
        if data.get(TIMESTAMP_KEYS["x_train_sequences"]) is not None:
            out[TIMESTAMP_KEYS["train_source_sequences"]] = data[TIMESTAMP_KEYS["x_train_sequences"]][mask & is_source]
    for phase in ("val", "test"):
        if phase not in keep:
            continue
        sequences = _subset(data[f"{phase}_source_sequences"], keep[phase])
        out[f"{phase}_source_sequences"] = sequences
        out[f"{phase}_source_matrix"] = count_matrix(sequences, value)
        _carry_times(out, data, f"{phase}_source_sequences", keep=keep[phase])
    out[KEPT] = dict(keep)
    return out


def _phase_sequences(data: dict[str, Any], phases: tuple[str, ...]) -> dict[str, Any]:
    keys = {"train": "x_train_sequences", "val": "val_source_sequences", "test": "test_source_sequences"}
    return {phase: data[keys[phase]] for phase in phases}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_fraction(value: Any, *, closed: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0 < value <= 1 if closed else 0 < value < 1


def _no_options(name: str, options: Options) -> None:
    if options:
        raise ProtocolError(f"{name} takes no options, got {sorted(options)}")


# ---------------------------------------------------------------------------
# history length
# ---------------------------------------------------------------------------

def _validate_history_length(levels, options) -> None:
    if not all(_is_int(n) and n >= 1 for n in levels):
        raise ProtocolError(f"history_length levels must be positive integers, got {list(levels)}")
    _no_options("history_length", options)


@register("history_length", version=3, stochastic=False, target="history_length",
          # Fewer events per history, a smaller share of the matrix, and fewer
          # chances for an item to recur: all three follow from the definition.
          # So do fewer distinct items and a changed popularity spread: a history of
          # n events holds at most n items, so at level 1 the test inputs of 2,942
          # users hold at most 2,942 of ML-20M's 19,400 (2026-09-30: flagged at every
          # level on ML-20M before this was declared).
          expected=("density", "repeat_rate", "catalogue", "popularity_gini"),
          validate=_validate_history_length, position=float)
def _history_length(data, level, *, rng, options, value, phases):
    """A context of ``level`` items: the last ``level + 1`` events of every training history, the last
    ``level`` of every input history.

    A training history of *n* events teaches contexts of at most *n* − 1 items,
    since its last event is only ever a target; so training keeps one event
    more than a model is then shown, and a model is trained and served with
    the same context. At level 1 it learns (one item → the next); keeping one
    event would leave it nothing to learn from. The matrix models agree: they
    learn to predict each item of a row from the others, so a row of
    ``level + 1`` items teaches the same *n* items → one more. The stage-1
    ``max_history_length`` of a sequential model still applies, so for that
    model every level at or above it differs from the full data only in what
    the matrix models see.
    """
    keep = {}
    for phase, sequences in _phase_sequences(data, phases).items():
        from_end = np.repeat(sequences.indptr[1:], sequences.row_lengths) - np.arange(sequences.values.size)
        keep[phase] = from_end <= int(level) + (1 if phase == "train" else 0)
    return keep_events(data, keep, value=value)


# ---------------------------------------------------------------------------
# density
# ---------------------------------------------------------------------------

def _validate_density(levels, options) -> None:
    if not all(_is_fraction(p) for p in levels):
        raise ProtocolError(f"density levels must be fractions in (0, 1), got {list(levels)}")
    unknown = sorted(set(options) - {"keep_catalogue"})
    if unknown or not isinstance(options.get("keep_catalogue", True), bool):
        raise ProtocolError(f"density takes only keep_catalogue = true|false, got {sorted(options)}")


@register("density", version=2, stochastic=True, target="density",
          # Thinning a history shortens it by the same fraction, and a thinner
          # history holds fewer repeats. The training catalogue is held fixed
          # (keep_catalogue), but thinned test inputs hold fewer distinct items:
          # -6% at p = 0.2 on long-tailed data (final review C1).
          expected=lambda options: ("history_length", "repeat_rate",
                                    "test catalogue" if options.get("keep_catalogue", True) else "catalogue"),
          validate=_validate_density, position=float)
def _density(data, level, *, rng, options, value, phases):
    """Keep a random fraction ``level`` of each history's events, spread across its whole length.

    Each history keeps ``max(1, round(p × length))`` events chosen uniformly
    among its positions, in their original order, so a thinned history reaches
    as far back as the original -- unlike truncation, which keeps only the most
    recent events. With ``keep_catalogue`` (the default) an item that would
    lose every training event keeps one, chosen at random, so the training
    catalogue does not shrink; that adds a few events beyond ``p``.
    """
    keep = {}
    for phase, sequences in _phase_sequences(data, phases).items():
        keys = _random_keys(rng, sequences.values.size)
        lengths = sequences.row_lengths
        quota = np.where(lengths > 0, np.maximum(1, np.rint(level * lengths)), 0).astype(np.int64)
        mask = _rank_within_rows(sequences, keys) < np.repeat(quota, lengths)
        if phase == "train" and options.get("keep_catalogue", True):
            values = sequences.values
            lost = (np.bincount(values, minlength=sequences.n_items) > 0) & \
                   (np.bincount(values[mask], minlength=sequences.n_items) == 0)
            candidates = np.flatnonzero(lost[values])
            if candidates.size:
                order = candidates[np.lexsort((keys[candidates], values[candidates]))]
                first = np.concatenate(([True], values[order][1:] != values[order][:-1]))
                mask[order[first]] = True
        keep[phase] = mask
    return keep_events(data, keep, value=value)


# ---------------------------------------------------------------------------
# repeat removal
# ---------------------------------------------------------------------------

def _validate_repeat_removal(levels, options) -> None:
    if not all(_is_fraction(q, closed=True) for q in levels):
        raise ProtocolError(f"repeat_removal levels must be fractions in (0, 1], got {list(levels)}")
    _no_options("repeat_removal", options)


@register("repeat_removal", version=1, stochastic=True, target="repeat_rate",
          # Removing events shortens histories. The first occurrence of every
          # item stays, so which items a history holds -- and so the catalogue
          # and the density -- does not change. The event counts do, and with
          # them the popularity spread: -13% at q = 1 on data with 40% repeats
          # (final review C1).
          expected=("history_length", "popularity_gini"),
          validate=_validate_repeat_removal, position=lambda q: 1.0 - float(q))
def _repeat_removal(data, level, *, rng, options, value, phases):
    """Remove each repeat event with probability ``level``.

    A repeat event is one whose item already occurred earlier in the same
    history. First occurrences are never removed, so every repeat that remains
    is still a repeat. On the knee's axis, less removal is closer to the full
    data, so the knee is the most removal that still costs less than δ.
    """
    keep = {}
    for phase, sequences in _phase_sequences(data, phases).items():
        repeat = ~_first_occurrence(sequences)
        keep[phase] = ~(repeat & (_random_keys(rng, sequences.values.size) < float(level)))
    return keep_events(data, keep, value=value)


# ---------------------------------------------------------------------------
# shuffled order
# ---------------------------------------------------------------------------

def _validate_shuffle(levels, options) -> None:
    if not all(level == "all" or (_is_int(level) and level >= 2) for level in levels):
        raise ProtocolError(f'shuffle levels must be "all" or block sizes >= 2, got {list(levels)}')
    _no_options("shuffle", options)


@register("shuffle", version=1, stochastic=True, target=None, validate=_validate_shuffle)
def _shuffle(data, level, *, rng, options, value, phases):
    """Shuffle the order of events within each history; ``level`` is the block size.

    ``"all"`` shuffles a whole history. A block size ``b`` shuffles only within
    consecutive blocks of ``b`` events, which destroys order at short range and
    keeps it at long range. Only the order changes -- which items a history
    holds, and how often, does not -- so none of the five characteristics
    should move, and the matrix views are left exactly as they were: the
    matrix models are a control that should score as with the full data. Each
    event keeps its own time, so a model that reads times rather than order
    (the time-decayed popularity baseline) is a control too.
    """
    out = dict(data)
    keys = {"train": "x_train_sequences", "val": "val_source_sequences", "test": "test_source_sequences"}
    for phase, sequences in _phase_sequences(data, phases).items():
        positions = _positions(sequences)
        block = np.zeros_like(positions) if level == "all" else positions // int(level)
        order = np.lexsort((_random_keys(rng, positions.size), block, _event_rows(sequences)))
        out[keys[phase]] = ItemSequences(values=sequences.values[order], indptr=sequences.indptr,
                                         n_items=sequences.n_items)
        # each time travels with its event, so what happened when is unchanged; only the order is not
        _carry_times(out, data, keys[phase], order=order)
    return out


# ---------------------------------------------------------------------------
# catalogue size
# ---------------------------------------------------------------------------

def _validate_catalogue(levels, options) -> None:
    ints, fractions = all(_is_int(k) and k >= 1 for k in levels), all(_is_fraction(f) for f in levels)
    if not (ints or fractions):
        raise ProtocolError("catalogue levels must all be item counts (integers >= 1) or all fractions "
                            f"of the training catalogue in (0, 1), got {list(levels)}")
    unknown = sorted(set(options) - {"strategy", "strata"})
    strategy = options.get("strategy", "top")
    strata = options.get("strata", 10)
    if unknown or strategy not in ("top", "stratified") or not (_is_int(strata) and strata >= 1):
        raise ProtocolError('catalogue takes strategy = "top"|"stratified" and strata = <int >= 1>, '
                            f"got {options}")


def _stratified_order(counts: np.ndarray, strata: int, rng: np.random.Generator) -> np.ndarray:
    """Every item in one random order that keeps each popularity stratum in proportion at every prefix.

    Items are ranked by popularity and cut into ``strata`` equal groups. Within
    a group of size s, a random permutation gives each item a slot r, and the
    item is placed at (r + U) / s with U uniform. Sorting all items by that
    place interleaves the groups evenly, so the first k items hold about
    k · s / n of each group (within one) -- and the first k are always among
    the first k' > k. Taking a prefix of this one order at every level is what
    makes the levels nested.
    """
    ranked = np.argsort(-counts, kind="stable")
    place = np.empty(ranked.size)
    for group in np.array_split(ranked, min(strata, ranked.size)):
        slots = rng.permutation(group.size)
        place[group] = (slots + rng.random(group.size)) / group.size
    return np.argsort(place, kind="stable")


def _stratified_sample(counts: np.ndarray, k: int, strata: int, rng: np.random.Generator) -> np.ndarray:
    """``k`` items drawn at random within popularity strata, each stratum in proportion to its size."""
    return _stratified_order(counts, strata, rng)[:k]


def _catalogue_expected(options: Options) -> tuple[str, ...]:
    # Dropping items drops their events. Keeping only the most popular also
    # flattens the popularity distribution; sampling within strata is meant not to.
    base = ("history_length", "density")
    return base + ("popularity_gini",) if options.get("strategy", "top") == "top" else base


@register("catalogue", version=2, stochastic=lambda options: options.get("strategy", "top") == "stratified",
          target="catalogue", expected=_catalogue_expected, validate=_validate_catalogue,
          scopes=("all",), changes_targets=True, nested=True)
def _catalogue(data, level, *, rng, options, value, phases):
    """Reduce the training catalogue to ``level`` items, with users fixed.

    ``level`` is an item count, or a fraction of the training catalogue.
    ``strategy = "top"`` keeps the most popular items by training events;
    ``"stratified"`` ranks items by popularity, cuts the ranking into
    ``strata`` equal groups, and draws from each at random in proportion. Both
    are nested: within a seed, a smaller catalogue is part of every larger one,
    so the levels differ in size and not in which random items were drawn.

    A removed item leaves every phase: its events leave the training data and
    the input histories, its targets leave validation and test, and it leaves
    the item space, so no model can recommend it. Items first seen in the
    validation or test window are kept, as new items no model can recommend,
    exactly as in the full data. Training users left without events are
    dropped from the training data, since they carry nothing to fit. Each
    condition is scored on its own test users (see :func:`own_test_rows`).
    """
    n_train = len(data["train_item_ids"])
    counts = np.bincount(data["x_train_sequences"].values, minlength=n_train)
    k = min(n_train, int(level) if _is_int(level) else max(1, round(level * n_train)))
    if options.get("strategy", "top") == "top":
        chosen = np.argsort(-counts, kind="stable")[:k]
    else:
        chosen = _stratified_sample(counts, k, int(options.get("strata", 10)), rng)
    kept_train = np.sort(chosen).astype(np.int64)

    out = dict(data)
    kept_masks = {}
    for phase in PHASES:
        ids = data[f"{phase}_item_ids"]
        columns = np.concatenate((kept_train, np.arange(n_train, len(ids), dtype=np.int64)))
        remap = np.full(len(ids), -1, dtype=np.int64)
        remap[columns] = np.arange(columns.size, dtype=np.int64)
        out[f"{phase}_item_ids"] = np.asarray(ids)[columns]
        keys = (("x_train_sequences", "train_source_sequences") if phase == "train"
                else (f"{phase}_source_sequences",))
        for key in keys:
            sequences = data[key]
            new_values = remap[sequences.values]
            mask = new_values >= 0
            out[key] = _subset(sequences, mask, values=new_values, n_items=columns.size)
            _carry_times(out, data, key, keep=mask)
            if key != "train_source_sequences":
                kept_masks[phase] = mask
        matrices = (("x_train", "train_source_matrix", "train_target_matrix") if phase == "train"
                    else (f"{phase}_source_matrix", f"{phase}_target_matrix", f"{phase}_next_target_matrix"))
        for key in [key for key in matrices if data.get(key) is not None]:
            out[key] = data[key][:, columns].tocsr()

    # training users left with nothing are dropped from every training view
    occupied = out["x_train_sequences"].row_lengths > 0
    if not occupied.all():
        for key in ("x_train_sequences", "train_source_sequences"):
            sequences = out[key]
            out[key] = ItemSequences(values=sequences.values, indptr=_indptr(sequences.row_lengths[occupied]),
                                     n_items=sequences.n_items)
        for key in ("x_train", "train_source_matrix", "train_target_matrix"):
            out[key] = out[key][occupied].tocsr()
        if out.get("train_user_ids") is not None:
            out["train_user_ids"] = np.asarray(out["train_user_ids"])[occupied]

    # the partitions of the test-space catalogue, and the stale index lists
    shift = k - n_train
    out["item_ids"] = out["test_item_ids"]
    out["warm_item_indices"] = np.arange(k, dtype=np.int64)
    for key in ("val_cold_item_indices", "test_cold_item_indices"):
        if data.get(key) is not None:
            out[key] = np.asarray(data[key], dtype=np.int64) + shift
    for key in ("val_source_indices", "val_target_indices", "test_source_indices", "test_target_indices"):
        out[key] = None
    out[KEPT] = kept_masks
    return out


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------

def condition_fingerprint(protocol: Protocol, sweep: str, dataset: str) -> str:
    """Identity of a sweep's conditions on one dataset: its split and its transform."""
    ablation = protocol.ablation(sweep)
    transform = transform_of(ablation)
    settings = {key: ablation.raw.get(key) for key in _RESULT_KEYS}
    settings["scope"] = scope_of(ablation)  # an explicit default and an omitted one are the same sweep
    return _digest({
        "split": protocol.dataset_fingerprint(dataset),
        # which users are scored and on what (targets, refit, scoring code): the fixed test users depend on it
        "evaluation": protocol.evaluation_key(dataset),
        "sweep": settings,
        "transform_version": transform.version,
    })


def ablation_fingerprint(protocol: Protocol, sweep: str, dataset: str, model: str) -> str:
    return _digest({"conditions": condition_fingerprint(protocol, sweep, dataset),
                    "stage1": protocol.run_fingerprint(dataset, model)})


def ablation_root(protocol: Protocol, work_dir: Path, sweep: str, dataset: str) -> Path:
    return Path(work_dir) / "ablations" / sweep / dataset / condition_fingerprint(protocol, sweep, dataset)[:12]


def sweep_seeds_path(work_dir: Path, sweep: str, dataset: str) -> Path:
    """The record of seeds added to ``sweep`` on ``dataset``; keyed like stage 1's, by name and not fingerprint."""
    return Path(work_dir) / "ablations" / sweep / dataset / "added_seeds.json"


def sweep_seeds(protocol: Protocol, work_dir: Path, sweep: str, dataset: str) -> tuple[int, ...]:
    """The seeds of ``sweep`` on ``dataset``: the protocol's, then any added with ``ablate --add-seeds``."""
    return seeds_with_added(protocol.seeds, sweep_seeds_path(work_dir, sweep, dataset))


def check_stage1_seeds(protocol: Protocol, work_dir: Path, sweep: str, dataset: str, seeds: list[int]) -> None:
    """Refuse seeds for ``sweep`` that are not stage-1 seeds of ``dataset``.

    The reference of a seed is that seed's stage-1 final model, and every condition reuses the configuration
    it was selected with, so a sweep can only have seeds stage 1 has.
    """
    stage1 = final_seeds(protocol, work_dir, dataset)
    absent = [seed for seed in seeds if seed not in stage1]
    if absent:
        raise ValueError(f"{sweep} on {dataset}: seed(s) {absent} are not stage-1 seeds of {dataset} "
                         f"({list(stage1)}), and a sweep's reference is that seed's stage-1 model. Add them "
                         f"there first: final --dataset {dataset} --add-seeds {' '.join(map(str, absent))}")


def add_sweep_seeds(protocol: Protocol, work_dir: Path, sweep: str, dataset: str, seeds: list[int]) -> list[int]:
    """Record ``seeds`` as further seeds of ``sweep`` on ``dataset``, for all its models; returns the new ones."""
    check_stage1_seeds(protocol, work_dir, sweep, dataset, seeds)
    return record_added_seeds(sweep_seeds_path(work_dir, sweep, dataset), seeds,
                              sweep_seeds(protocol, work_dir, sweep, dataset))


def data_seeds(protocol: Protocol, work_dir: Path, sweep: str, dataset: str) -> list[int | None]:
    """One subsample per seed of the sweep for a stochastic transform; one condition otherwise."""
    ablation = protocol.ablation(sweep)
    stochastic = transform_of(ablation).stochastic(ablation.options)
    return list(sweep_seeds(protocol, work_dir, sweep, dataset)) if stochastic else [None]


def condition_name(level: Any, data_seed: int | None) -> str:
    label = level_label(level)
    return label if data_seed is None else f"{label}/seed{data_seed}"


def _eligible_test_rows(data: dict[str, Any], targets: str = "test_target_matrix", *,
                        exclude_seen: bool = False, k: int | None = None) -> np.ndarray:
    """Test rows that can be scored: a non-empty history, and a target -- for next-item targets, one a model
    can recommend, exactly as :func:`~seqrec_eval.evaluate.scored_rows` scores them (H16, review C4). With
    ``exclude_seen``, also a full list of ``k`` unseen items, as :func:`~seqrec_eval.evaluate.evaluate_phase`
    requires (review N49)."""
    if targets.endswith("next_target_matrix"):
        has_targets = recommendable_next(reachable_targets(data, "test", data[targets], exclude_seen),
                                         len(data["train_item_ids"]))
    else:
        has_targets = np.diff(data[targets].indptr) > 0
    return has_targets & (data["test_source_sequences"].row_lengths > 0) & _fills(data, exclude_seen, k)


def _fills(data: dict[str, Any], exclude_seen: bool, k: int | None) -> np.ndarray | bool:
    if not exclude_seen:
        return True
    if k is None:
        raise ValueError("with exclude_seen, which test rows can be scored depends on k (max cutoff)")
    return fills_list(data, "test", k)


def apply_condition(protocol: Protocol, split: Split, sweep: str, level: Any,
                    data_seed: int | None = None, *, test_rows: np.ndarray | None = None) -> Split:
    """``split`` transformed to ``level``; the original split is not modified."""
    ablation = protocol.ablation(sweep)
    transform = transform_of(ablation)
    stochastic = transform.stochastic(ablation.options)
    if stochastic != (data_seed is not None):
        raise ValueError(f"{sweep}: a {'stochastic' if stochastic else 'deterministic'} "
                         f"transform {'needs' if stochastic else 'takes no'} data seed")
    # a nested transform draws once per seed, so every level of that seed shares the draw
    level_part = ("nested",) if transform.nested else (level_label(level),)
    rng = None if data_seed is None else _stream(protocol.search_seed, "ablation", sweep, split.dataset,
                                                 *level_part, data_seed)
    phases = PHASES if scope_of(ablation) == "all" else ("val", "test")
    data = transform.apply(split.data, level, rng=rng, options=ablation.options,
                           value=protocol.dataset(split.dataset).set_all_values_to, phases=phases)
    # A transform that drops events from the input histories, in the same item space, changes what
    # a model reads but not what the user has seen: exclusion keeps the original histories.
    for phase in ("val", "test"):
        key = f"{phase}_source_matrix"
        same_space = len(data[f"{phase}_item_ids"]) == len(split.data[f"{phase}_item_ids"])
        if data[key] is not split.data[key] and same_space:
            data[f"{phase}_seen_matrix"] = split.data[key]
    condition = {"sweep": sweep, "level": level, "label": level_label(level), "data_seed": data_seed}
    return replace(split, data=data, test_rows=test_rows, condition=condition)


def reference_condition(split: Split, sweep: str, test_rows: np.ndarray) -> Split:
    return replace(split, test_rows=test_rows,
                   condition={"sweep": sweep, "level": None, "label": REFERENCE, "data_seed": None})


def own_test_rows(data: dict[str, Any], targets: str, *, exclude_seen: bool = False,
                  k: int | None = None) -> np.ndarray:
    """Test rows a condition can score on its own: a known next target in its catalogue, and a history item.

    A known target is one inside the training catalogue (``index < len(train_item_ids)``); an item first seen
    after training can never be recommended, so a user whose targets are all such items scores 0 for every
    model and says nothing about any of them. With ``exclude_seen``, the catalogue must also hold ``k`` items
    the user has not seen (review N49: at a small catalogue, the heaviest users have seen nearly all of it).
    """
    known = recommendable_next(reachable_targets(data, "test", data[targets], exclude_seen),
                               len(data["train_item_ids"]))
    usable = known & (data["test_source_sequences"].row_lengths > 0) & _fills(data, exclude_seen, k)
    return np.flatnonzero(usable).astype(np.int64)


def per_condition_users(protocol: Protocol, sweep: str) -> bool:
    """Whether each condition of ``sweep`` is scored on its own users rather than one fixed set."""
    return transform_of(protocol.ablation(sweep)).changes_targets


def build_reference(protocol: Protocol, work_dir: Path, split: Split, sweep: str) -> Split:
    """The full-data condition of ``sweep``, scored on the fixed set or, where targets change, its own users."""
    if per_condition_users(protocol, sweep):
        rows = own_test_rows(split.data, target_key("test", protocol.targets),
                             exclude_seen=protocol.dataset(split.dataset).exclude_seen, k=max(protocol.cutoffs))
    else:
        rows = fixed_test_rows(protocol, work_dir, split, sweep)
    return reference_condition(split, sweep, rows)


def build_condition(protocol: Protocol, work_dir: Path, split: Split, sweep: str, level: Any,
                    data_seed: int | None) -> Split:
    """``split`` at one level (and subsample) of ``sweep``, with the users it is scored on."""
    if per_condition_users(protocol, sweep):
        condition = apply_condition(protocol, split, sweep, level, data_seed)
        return replace(condition, test_rows=own_test_rows(condition.data, target_key("test", protocol.targets),
                                                          exclude_seen=protocol.dataset(split.dataset).exclude_seen,
                                                          k=max(protocol.cutoffs)))
    rows = fixed_test_rows(protocol, work_dir, split, sweep)
    return apply_condition(protocol, split, sweep, level, data_seed, test_rows=rows)


def fixed_test_rows(protocol: Protocol, work_dir: Path, split: Split, sweep: str) -> np.ndarray:
    """Test rows eligible in the full data and at every level (and subsample) of the sweep.

    Cached with the subsamples they were checked on (``test_rows.json``). A subsample added later
    (``ablate --add-seeds`` on a stochastic sweep) is checked against the cached users, not intersected into
    them: every run already made was scored on those users, so they must all stay eligible, or the new runs
    could not be paired with the old. Every transform registered so far keeps them so by construction -- it
    drops or reorders events inside a fixed item space, so a next item stays recommendable, and a non-empty
    history keeps at least one event -- and a transform that did not is refused here rather than scored on
    fewer users.
    """
    root = ablation_root(protocol, work_dir, sweep, split.dataset)
    path, record = root / "test_rows.npy", root / "test_rows.json"
    targets = target_key("test", protocol.targets)
    seeds = data_seeds(protocol, work_dir, sweep, split.dataset)
    if path.exists():
        rows = np.load(path)
        checked = read_json(record)["data_seeds"] if record.exists() else []
        missing = [seed for seed in seeds if seed not in checked]
        if missing:
            _check_subsamples(protocol, split, sweep, rows, missing, remedy=(
                f"Remove the seed from {sweep_seeds_path(work_dir, sweep, split.dataset)} to keep the sweep as it "
                f"was, or start the sweep over with it (delete {root})."))
            write_json(record, {"data_seeds": checked + missing, "n_rows": int(rows.size)})
        return rows
    exclude_seen = protocol.dataset(split.dataset).exclude_seen
    k = max(protocol.cutoffs)
    eligible = _eligible_test_rows(split.data, targets, exclude_seen=exclude_seen, k=k)
    for level in protocol.ablation(sweep).levels:
        for data_seed in seeds:
            eligible &= _eligible_test_rows(apply_condition(protocol, split, sweep, level, data_seed).data, targets,
                                            exclude_seen=exclude_seen, k=k)
    rows = np.flatnonzero(eligible).astype(np.int64)
    if rows.size == 0:
        raise RuntimeError(f"{sweep} on {split.dataset}: no test user is eligible at every level")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
    np.save(temporary, rows)
    durable_replace(temporary, path)
    write_json(record, {"data_seeds": seeds, "n_rows": int(rows.size)})
    return rows


def _check_subsamples(protocol: Protocol, split: Split, sweep: str, rows: np.ndarray, seeds: list[int | None],
                      *, remedy: str) -> None:
    """Refuse subsamples under which any of the fixed test users ``rows`` would no longer be eligible."""
    targets = target_key("test", protocol.targets)
    for data_seed in seeds:
        for level in protocol.ablation(sweep).levels:
            eligible = _eligible_test_rows(apply_condition(protocol, split, sweep, level, data_seed).data, targets,
                                           exclude_seen=protocol.dataset(split.dataset).exclude_seen,
                                           k=max(protocol.cutoffs))
            lost = int((~eligible[rows]).sum())
            if lost:
                raise RuntimeError(
                    f"{sweep} on {split.dataset}: subsample {data_seed} at level {level_label(level)} leaves "
                    f"{lost:,} of the {rows.size:,} fixed test users ineligible. Every run of the sweep so far was "
                    f"scored on all of them, so runs of this seed could not be paired with those. {remedy}")


def check_added_seeds(protocol: Protocol, work_dir: Path, split: Split, sweep: str, seeds: list[int]) -> None:
    """Before seeds are added to ``sweep``: refuse any whose subsamples would shrink its cached fixed test users.

    Nothing to check for a deterministic sweep (its conditions do not depend on the seed), for one whose
    conditions score their own users, or before the fixed users are first computed (they are then computed
    over every seed).
    """
    root = ablation_root(protocol, work_dir, sweep, split.dataset)
    path, record = root / "test_rows.npy", root / "test_rows.json"
    stochastic = data_seeds(protocol, work_dir, sweep, split.dataset) != [None]
    if not stochastic or per_condition_users(protocol, sweep) or not path.exists() or not record.exists():
        return  # without its record, fixed_test_rows checks every subsample itself
    rows, checked = np.load(path), read_json(record)["data_seeds"]
    new = [seed for seed in seeds if seed not in checked]
    if new:
        _check_subsamples(protocol, split, sweep, rows, new,
                          remedy=f"The seed was not added. To use it, start the sweep over (delete {root}).")
        # checked, so fixed_test_rows need not check it again; a subsample checked but never used is harmless
        write_json(record, {"data_seeds": checked + new, "n_rows": int(rows.size)})


# ---------------------------------------------------------------------------
# the manipulation check
# ---------------------------------------------------------------------------

def gini(counts: np.ndarray) -> float:
    counts = np.sort(np.asarray(counts, dtype=np.float64))
    n, total = counts.size, counts.sum()
    if n == 0 or total == 0:
        return float("nan")
    return float(2.0 * np.sum(np.arange(1, n + 1) * counts) / (n * total) - (n + 1) / n)


def _row_mask(n_rows: int, rows: np.ndarray | None) -> np.ndarray:
    if rows is None:
        return np.ones(n_rows, dtype=bool)
    wanted = np.zeros(n_rows, dtype=bool)
    wanted[rows] = True
    return wanted


def _describe(sequences, rows: np.ndarray | None = None) -> dict[str, float]:
    wanted = _row_mask(sequences.n_rows, rows)
    event_rows = _event_rows(sequences)
    in_rows = wanted[event_rows]
    values, event_rows, lengths = sequences.values[in_rows], event_rows[in_rows], sequences.row_lengths[wanted]
    n_rows = int(wanted.sum())
    counts = np.bincount(values, minlength=sequences.n_items)
    pairs = np.unique(event_rows * np.int64(max(sequences.n_items, 1)) + values).size
    catalogue = int(np.count_nonzero(counts))
    return {
        "history_length": float(lengths.mean()) if lengths.size else float("nan"),
        "history_length_p50": float(np.median(lengths)) if lengths.size else float("nan"),
        "catalogue": catalogue,
        "density": pairs / (n_rows * catalogue) if n_rows and catalogue else float("nan"),
        "popularity_gini": gini(counts[counts > 0]),
        "repeat_rate": 1.0 - pairs / values.size if values.size else float("nan"),
        "rows": n_rows,
        "events": int(values.size),
    }


def _edits(before, after, before_ids, after_ids, kept: np.ndarray | None,
           rows: np.ndarray | None) -> dict[str, float | None]:
    """How much of the data a condition actually touched, relative to the original.

    ``rows_changed`` is the share of rows whose history differs in any way:
    length, items or order. ``span_kept`` is, over rows that keep an event, the
    mean share of the original history between the first and the last kept
    event -- 1 when the kept events still reach back as far as the original,
    ``n / length`` after truncation to the last ``n``. It needs the event
    masks, so it is ``None`` for a transform that reorders rather than drops.
    Both are ``None`` where the transform dropped rows.
    """
    if before.n_rows != after.n_rows:
        return {"rows_changed": None, "span_kept": None}
    wanted = _row_mask(before.n_rows, rows)
    lengths_before, lengths_after = before.row_lengths, after.row_lengths
    changed = lengths_before != lengths_after
    same = ~changed
    if same.any():
        # after's values in before's item space, so a re-addressed item is not a change
        lookup = pd.Index(np.asarray(before_ids).astype(str)).get_indexer(np.asarray(after_ids).astype(str))
        in_before, in_after = same[_event_rows(before)], same[_event_rows(after)]
        differs = before.values[in_before] != lookup[after.values[in_after]]
        changed[np.unique(_event_rows(before)[in_before][differs])] = True
    result: dict[str, float | None] = {"rows_changed": float(changed[wanted].mean()) if wanted.any() else None,
                                       "span_kept": None}
    if kept is not None:
        positions, event_rows = _positions(before)[kept], _event_rows(before)[kept]
        counts = np.bincount(event_rows, minlength=before.n_rows)
        starts = _indptr(counts)
        has = (counts > 0) & wanted
        if has.any():
            first, last = positions[starts[:-1][has]], positions[starts[1:][has] - 1]
            result["span_kept"] = float(np.mean((last - first + 1) / lengths_before[has]))
    return result


def _order_stats(sequences, rows: np.ndarray | None = None) -> dict[str, float]:
    """Users with any repeat, and the share of adjacent pairs that are self-transitions."""
    wanted = _row_mask(sequences.n_rows, rows)
    event_rows = _event_rows(sequences)
    repeats = np.bincount(event_rows, weights=~_first_occurrence(sequences), minlength=sequences.n_rows)
    adjacent = (event_rows[:-1] == event_rows[1:]) & wanted[event_rows[:-1]] if event_rows.size > 1 else \
        np.zeros(0, dtype=bool)
    same = sequences.values[:-1][adjacent] == sequences.values[1:][adjacent]
    return {"users_with_any_repeat": float((repeats[wanted] > 0).mean()) if wanted.any() else float("nan"),
            "self_transition_share": float(same.mean()) if same.size else float("nan")}


def _time_stats(sequences, times: np.ndarray | None, rows: np.ndarray | None = None) -> dict[str, float | None]:
    """The audit's timing questions, asked of these histories; ``None`` without timestamps.

    ``tie_rate`` is the share of adjacent pairs with the same timestamp -- pairs
    whose order was decided by the source file, not by time. ``last_tied`` is
    the share of histories whose last two events tie, i.e. where which item is
    "last", the item a Markov model predicts from, is arbitrary.
    """
    keys = ("tie_rate", "last_tied", "median_gap_seconds", "gap_under_30_min", "gap_over_1_day",
            "median_span_days")
    if times is None or times.shape != sequences.values.shape:
        return dict.fromkeys(keys)
    wanted = _row_mask(sequences.n_rows, rows)
    event_rows = _event_rows(sequences)
    adjacent = (event_rows[:-1] == event_rows[1:]) & wanted[event_rows[:-1]] if event_rows.size > 1 else \
        np.zeros(0, dtype=bool)
    gaps = (times[1:] - times[:-1])[adjacent]
    lengths = sequences.row_lengths
    has_pair = wanted & (lengths >= 2)
    ends = sequences.indptr[1:]
    last_tied = times[ends[has_pair] - 1] == times[ends[has_pair] - 2]
    has_event = wanted & (lengths >= 1)
    spans = (times[ends[has_event] - 1] - times[sequences.indptr[:-1][has_event]]) / 86_400.0
    return {
        "tie_rate": float((gaps == 0).mean()) if gaps.size else None,
        "last_tied": float(last_tied.mean()) if last_tied.size else None,
        "median_gap_seconds": float(np.median(gaps)) if gaps.size else None,
        "gap_under_30_min": float((gaps < 1800).mean()) if gaps.size else None,
        "gap_over_1_day": float((gaps > 86_400).mean()) if gaps.size else None,
        "median_span_days": float(np.median(spans)) if spans.size else None,
    }


def last_pair_tied(split: Split) -> np.ndarray | None:
    """Per test row, whether the last two history events share a timestamp; ``None`` without timestamps."""
    sequences, times = split.data["test_source_sequences"], split.data.get(TIMESTAMP_KEYS["test_source_sequences"])
    if times is None or times.shape != sequences.values.shape:
        return None
    ends = sequences.indptr[1:]
    tied = np.zeros(sequences.n_rows, dtype=bool)
    has_pair = sequences.row_lengths >= 2
    tied[has_pair] = times[ends[has_pair] - 1] == times[ends[has_pair] - 2]
    return tied


def characteristics(split: Split, original: Split | None = None,
                    targets: str = "test_target_matrix") -> dict[str, dict[str, Any]]:
    """The profile of the training data and of the scored test inputs: one computation, read twice.

    It is the data profile the analysis reports for every condition, and the
    manipulation check reads its first five fields to compare a condition
    with the full data -- so what is described and what is checked cannot
    drift apart.

    The five characteristics: history length is events per row; catalogue the
    items with at least one event; density the share of the row-by-catalogue
    matrix that is non-zero; popularity Gini over the event counts of those
    items; repeat rate the share of events that repeat an item already in the
    same history. Beside them, the audit's descriptive fields: length spread,
    users with any repeat, self-transitions, and, with timestamps, ties, gaps
    and spans (see :func:`_time_stats`); for the test part, the share of targets
    that are items first seen after training. Given the original split, each
    part also records how much of it the condition touched (see :func:`_edits`).
    """
    parts = {"train": ("x_train_sequences", "train_item_ids", None),
             "test": ("test_source_sequences", "test_item_ids", split.test_rows)}
    out = {}
    kept = split.data.get(KEPT) or {}
    for part, (key, ids, rows) in parts.items():
        sequences = split.data[key]
        out[part] = _describe(sequences, rows)
        lengths = sequences.row_lengths[_row_mask(sequences.n_rows, rows)]
        out[part]["history_length_p90"] = float(np.quantile(lengths, 0.9)) if lengths.size else float("nan")
        out[part].update(_order_stats(sequences, rows))
        out[part].update(_time_stats(sequences, split.data.get(TIMESTAMP_KEYS[key]), rows))
        if original is not None:
            out[part].update(_edits(original.data[key], split.data[key], original.data[ids], split.data[ids],
                                    kept.get(part), rows))
    targets = split.data[targets]
    if split.test_rows is not None:
        targets = targets[split.test_rows]
    n_train = len(split.data["train_item_ids"])
    out["test"]["new_item_target_share"] = float((targets.indices >= n_train).mean()) if targets.nnz else None
    return out


def record_characteristics(protocol: Protocol, work_dir: Path, split: Split,
                           original: Split | None = None) -> dict[str, Any]:
    condition = split.condition
    name = condition_name(condition["level"], condition["data_seed"]) if condition["label"] != REFERENCE else REFERENCE
    path = ablation_root(protocol, work_dir, condition["sweep"], split.dataset) / "conditions" / f"{name}.json"
    if path.exists():
        return read_json(path)
    record = {**condition,
              "characteristics": characteristics(split, original, target_key("test", protocol.targets))}
    write_json(path, record)
    return record


def manipulation_check(reference: dict[str, dict[str, float]], measured: dict[str, dict[str, float]],
                       target: str | None, tolerance: float, expected: tuple[str, ...] = (),
                       direction: int = -1) -> dict[str, Any]:
    """Relative change of every characteristic, those that moved without being declared to, and any
    part where the target itself moved the wrong way.

    An undeclared move may be a property of the data worth reporting. A target
    moving against the transform's own direction cannot be: truncating
    histories cannot lengthen them. That is flagged as ``wrong_way``, a bug or
    an incompatibility nobody foresaw, to be understood before any result of
    the sweep is read.
    """
    change: dict[str, dict[str, float]] = {}
    moved, wrong_way = [], []
    for part in ("train", "test"):
        change[part] = {}
        for name in CHARACTERISTICS:
            before, after = reference[part][name], measured[part][name]
            relative = (after - before) / abs(before) if before and math.isfinite(before) else float("nan")
            change[part][name] = relative
            declared = name == target or name in expected or f"{part} {name}" in expected
            if not declared and math.isfinite(relative) and abs(relative) > tolerance:
                moved.append(f"{part} {name}")
            if name == target and math.isfinite(relative) and relative * direction < -1e-9:
                wrong_way.append(f"{part} {name}")
    return {"change": change, "moved": moved, "wrong_way": wrong_way}


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def plan_ablation(protocol: Protocol, work_dir: Path, sweep: str, dataset: str, model: str,
                  *, allow_incomplete: bool = False) -> dict[str, list[RunSpec]]:
    """The runs of one model in one sweep, by condition name, the reference first.

    With scope ``"all"`` every condition refits the configuration stage 1
    selected, under each of the sweep's seeds (:func:`sweep_seeds`); with
    ``"inference"`` it rescores stage 1's final model of that seed. The
    reference always reloads stage 1's final models. Raises, as ``plan_finals``
    does, while stage 1's search is unfinished, and when the sweep has a seed
    stage 1 does not.
    """
    finals = plan_finals(protocol, work_dir, dataset, model, allow_incomplete=allow_incomplete)
    if not finals:
        return {}
    by_seed = {final.seed: final for final in finals}
    seeds = sweep_seeds(protocol, work_dir, sweep, dataset)
    absent = [seed for seed in seeds if seed not in by_seed]
    if absent:
        raise RuntimeError(f"{sweep} on {dataset} has seed(s) {absent}, which are not stage-1 seeds of {dataset}; "
                           f"add them there (final --dataset {dataset} --add-seeds ...) or remove them from "
                           f"{sweep_seeds_path(work_dir, sweep, dataset)}")
    finals = [by_seed[seed] for seed in seeds]  # stage 1 may have more seeds than the sweep
    fingerprint = ablation_fingerprint(protocol, sweep, dataset, model)
    base = (ablation_root(protocol, work_dir, sweep, dataset) / "runs" / model / fingerprint[:12]).relative_to(work_dir)
    stage1 = run_root(work_dir, dataset, model, protocol.run_fingerprint(dataset, model)).relative_to(work_dir)
    rescore = scope_of(protocol.ablation(sweep)) == "inference"

    def checkpoint(final: RunSpec) -> str:
        return str(stage1 / f"final-seed{final.seed}" / "model.zip")

    plans: dict[str, list[RunSpec]] = {REFERENCE: [
        replace(final, kind="reference", fingerprint=fingerprint, condition={
            "sweep": sweep, "level": None, "label": REFERENCE, "data_seed": None,
            "path": str(base / REFERENCE), "checkpoint": checkpoint(final),
        })
        for final in finals
    ]}
    for level in protocol.ablation(sweep).levels:
        for data_seed in data_seeds(protocol, work_dir, sweep, dataset):
            chosen = finals if data_seed is None else [f for f in finals if f.seed == data_seed]
            specs = []
            for final in chosen:
                condition = {"sweep": sweep, "level": level, "label": level_label(level), "data_seed": data_seed,
                             "path": str(base / level_label(level))}
                if rescore:
                    condition["checkpoint"] = checkpoint(final)
                specs.append(replace(final, kind="rescore" if rescore else "final", fingerprint=fingerprint,
                                     condition=condition))
            plans[condition_name(level, data_seed)] = specs
    return plans
