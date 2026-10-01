"""The refit: every model scored on test is first trained on everything before the test window.

Without it, a model is trained up to the validation window and tested on
histories that run to the test window. Validation and test are then different
tasks -- no gap between training and history in one, a whole window in the
other -- so the search selects settings under conditions the test does not
have. The window also costs models unequally: an item first seen in validation
has no embedding and cannot be recommended, which costs recency-based models
most, and a longer history than any training sequence puts BERT4Rec's
``[MASK]`` on a position its training never used. Search trials still fit on
the training window and score validation; the settings they select are then
fitted once more, per seed, on train and validation, and only that model is
tested.

The library's split has no such training set, so it is rebuilt here from the
builder's prepared events, by the builder's own rule for a train stage, one
window later:

- the catalogue is the validation stage's (``val_item_ids``), the start of the
  test catalogue, so a refitted model's rankings map into the test catalogue
  as a trained one's do; the few items seen before the test window but dropped
  by the validation stage's filters stay unknown, as they are now;
- the users are everyone the train-stage rule keeps (with
  ``temporal_train_users = "all"``: at least ``min_user_support`` distinct
  catalogue items before the test window), in the builder's row order;
- each history holds every event before the test window on a catalogue item,
  ordered by (user, time) with a stable sort;
- the matrix is built like ``x_train``: the events before the validation window
  and those inside it are each summed per (user, item), and ``x_refit`` is
  their maximum.

Rebuilding outside the library is safe only if it is the library's rule, so it
is proved every time: the same code, given the train stage's boundaries and
catalogue, must reproduce the split's ``x_train``, ``x_train_sequences``,
``train_source_sequences`` and ``train_user_ids`` exactly. If the library ever
builds training data another way, the refit refuses rather than drifting.

Stored beside the split::

    x_refit.npz                   x_refit_sequences.npz       x_refit_timestamps.npy
    refit_source_matrix.npz       refit_source_sequences.npz  refit_source_timestamps.npy
    refit_target_matrix.npz       refit_user_ids.npy
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from compresso_recsys import ItemSequences, builder
from compresso_recsys.sequences import load_item_sequences, save_item_sequences
from scipy.sparse import csr_matrix, load_npz, save_npz

from .timestamps import _boundaries, prepared_events

MATRICES = ("x_refit", "refit_source_matrix", "refit_target_matrix")
SEQUENCES = ("x_refit_sequences", "refit_source_sequences")
TIMES = ("x_refit_timestamps", "refit_source_timestamps")
USERS = "refit_user_ids"


class RefitAlignmentError(RuntimeError):
    """The rebuilt training set does not reproduce the split's own."""


def train_rule(params: dict[str, Any], data_dir: Path) -> tuple[int, int, int]:
    """The train stage's user rule: (min_user_support, min_source_items, min_target_items)."""
    resolved, _ = builder._resolve_args(builder._build_args(**params, data_dir=str(data_dir)))
    stage = builder._train_stage_args(resolved)
    return int(stage.min_user_support), int(stage.min_source_items), int(stage.min_target_items)


def _indptr(lengths: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)


def _matrix(rows: np.ndarray, columns: np.ndarray, values: np.ndarray, shape: tuple[int, int]) -> csr_matrix:
    # as the builder builds a stage's matrices: values summed per (user, item), zeros dropped
    matrix = csr_matrix((values, (rows, columns)), shape=shape, dtype=np.float32)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def training_set(events: pd.DataFrame, item_ids: np.ndarray, *, source_end: float, window_end: float,
                 rule: tuple[int, int, int]) -> dict[str, Any]:
    """A train stage over ``item_ids``: the events before ``window_end``, split at ``source_end``.

    The catalogue is fixed, so the builder's fixed point reduces to one pass of
    its user filter: distinct catalogue items in the window, before
    ``source_end``, and from it on.
    """
    min_user_support, min_source_items, min_target_items = rule
    n_items = len(item_ids)
    times = events["timestamp"].to_numpy(dtype=np.float64)
    columns = pd.Index(np.asarray(item_ids).astype(str)).get_indexer(events["item_id"])
    codes = events["user_code"].to_numpy(dtype=np.int64)
    n_codes = int(codes.max()) + 1 if codes.size else 0
    inside = (times < window_end) & (columns >= 0)
    codes, columns, times = codes[inside], columns[inside], times[inside]
    values = events["value"].to_numpy(dtype=np.float32)[inside]
    before = times < source_end

    def distinct(mask: np.ndarray) -> np.ndarray:
        pairs = np.unique(codes[mask] * n_items + columns[mask])
        return np.bincount(pairs // n_items, minlength=n_codes)

    keep = distinct(np.ones(codes.size, dtype=bool)) >= min_user_support
    keep &= distinct(before) >= min_source_items
    keep &= distinct(~before) >= min_target_items
    user_codes = np.flatnonzero(keep)
    lookup = np.full(n_codes, -1, dtype=np.int64)
    lookup[user_codes] = np.arange(user_codes.size, dtype=np.int64)
    ids = np.empty(n_codes, dtype=object)
    ids[events["user_code"].to_numpy(dtype=np.int64)] = events["user_id"].to_numpy()

    rows = lookup[codes]
    chosen = rows >= 0
    rows, columns, times, values, before = (a[chosen] for a in (rows, columns, times, values, before))
    order = np.lexsort((times, rows))  # stable: equal times keep the source order, as in the builder
    rows, columns, times, values, before = (a[order] for a in (rows, columns, times, values, before))

    n_rows = user_codes.size
    shape = (n_rows, n_items)
    source = _matrix(rows[before], columns[before], values[before], shape)
    target = _matrix(rows[~before], columns[~before], values[~before], shape)
    return {
        "user_ids": ids[user_codes].astype(str),
        "sequences": ItemSequences(values=columns, indptr=_indptr(np.bincount(rows, minlength=n_rows)),
                                   n_items=n_items),
        "source_sequences": ItemSequences(values=columns[before],
                                          indptr=_indptr(np.bincount(rows[before], minlength=n_rows)),
                                          n_items=n_items),
        "source_matrix": source,
        "target_matrix": target,
        "matrix": source.maximum(target).tocsr(),
        "timestamps": times,
        "source_timestamps": times[before],
    }


def _same_sequences(name: str, built: ItemSequences, saved: ItemSequences, users: np.ndarray,
                    item_ids: np.ndarray) -> None:
    if not np.array_equal(built.row_lengths, saved.row_lengths):
        row = int(np.flatnonzero(built.row_lengths != saved.row_lengths)[0])
        raise RefitAlignmentError(f"{name}: user {users[row]} has {int(saved.row_lengths[row])} events in the "
                                  f"split but {int(built.row_lengths[row])} rebuilt")
    differ = np.flatnonzero(built.values != saved.values)
    if differ.size:
        event = int(differ[0])
        row = int(np.searchsorted(saved.indptr, event, side="right") - 1)
        raise RefitAlignmentError(f"{name}: user {users[row]} differs at position {event - int(saved.indptr[row])}: "
                                  f"the split has {str(item_ids[saved.values[event]])!r}, the rebuild "
                                  f"{str(item_ids[built.values[event]])!r} ({differ.size:,} events differ in all)")


def prove(events: pd.DataFrame, data: dict[str, Any], manifest: dict[str, Any],
          rule: tuple[int, int, int]) -> None:
    """Rebuild the train stage and require it to equal the split's, or raise :class:`RefitAlignmentError`."""
    boundaries = _boundaries(manifest)
    item_ids = np.asarray(data["train_item_ids"]).astype(str)
    built = training_set(events, item_ids, source_end=boundaries["train_target_start"],
                         window_end=boundaries["validation_target_start"], rule=rule)
    saved_users = np.asarray(data["train_user_ids"]).astype(str)
    if not np.array_equal(built["user_ids"], saved_users):
        extra = np.setdiff1d(built["user_ids"], saved_users)
        missing = np.setdiff1d(saved_users, built["user_ids"])
        detail = (f"{extra.size:,} users rebuilt that the split lacks (e.g. {extra[0]!r}), {missing.size:,} the "
                  f"other way" if extra.size or missing.size else "the same users in another order")
        raise RefitAlignmentError(f"train users: {detail}")
    _same_sequences("x_train_sequences", built["sequences"], data["x_train_sequences"], saved_users, item_ids)
    _same_sequences("train_source_sequences", built["source_sequences"], data["train_source_sequences"],
                    saved_users, item_ids)
    saved = data["x_train"].tocsr()
    if built["matrix"].shape != saved.shape or (built["matrix"] != saved).nnz:
        cell = (built["matrix"] != saved).tocoo() if built["matrix"].shape == saved.shape else None
        where = "" if cell is None else f", first at user {saved_users[cell.row[0]]!r}, item {item_ids[cell.col[0]]!r}"
        raise RefitAlignmentError(f"x_train: the rebuilt matrix differs from the split's{where}")


def attach_refit(params: dict[str, Any], data_dir: Path, split_path: Path, data: dict[str, Any],
                 manifest: dict[str, Any], *, events: pd.DataFrame | None = None) -> dict[str, Any]:
    """Prove the rule, build the train+validation set, save it beside the split; the summary for split_info."""
    if events is None:
        events = prepared_events(params, data_dir)
    rule = train_rule(params, data_dir)
    prove(events, data, manifest, rule)
    boundaries = _boundaries(manifest)
    built = training_set(events, np.asarray(data["val_item_ids"]).astype(str),
                         source_end=boundaries["validation_target_start"],
                         window_end=boundaries["test_target_start"], rule=rule)
    split_path = Path(split_path)
    for key, part in zip(MATRICES, ("matrix", "source_matrix", "target_matrix")):
        save_npz(split_path / f"{key}.npz", built[part])
    for key, part in zip(SEQUENCES, ("sequences", "source_sequences")):
        save_item_sequences(split_path / f"{key}.npz", built[part])
    for key, part in zip(TIMES, ("timestamps", "source_timestamps")):
        np.save(split_path / f"{key}.npy", built[part])
    np.save(split_path / f"{USERS}.npy", built["user_ids"])

    summary = {
        "proved_against": ["x_train", "x_train_sequences", "train_source_sequences", "train_user_ids"],
        "rule": dict(zip(("min_user_support", "min_source_items", "min_target_items"), rule)),
        "users": int(built["sequences"].n_rows), "train_users": int(data["x_train_sequences"].n_rows),
        "events": int(built["sequences"].values.size), "train_events": int(data["x_train_sequences"].values.size),
        "items": len(data["val_item_ids"]), "train_items": len(data["train_item_ids"]),
    }
    targets = data.get("test_next_target_matrix")
    if targets is not None:
        # how many test users have a next target a model can recommend, before and after the refit
        targets = targets.tocsr()
        rows = np.repeat(np.arange(targets.shape[0]), np.diff(targets.indptr))
        for name, size in (("train", len(data["train_item_ids"])), ("refit", len(data["val_item_ids"]))):
            known = np.bincount(rows, weights=targets.indices < size, minlength=targets.shape[0]) > 0
            summary[f"test_users_with_a_known_next_target_{name}"] = int(known.sum())
    return summary


def load_refit(split_path: Path) -> dict[str, Any]:
    """The train+validation set of a split, if it was built."""
    split_path = Path(split_path)
    if not (split_path / f"{USERS}.npy").exists():
        return {}
    out: dict[str, Any] = {key: load_npz(split_path / f"{key}.npz").tocsr() for key in MATRICES}
    out.update({key: load_item_sequences(split_path / f"{key}.npz") for key in SEQUENCES})
    out.update({key: np.load(split_path / f"{key}.npy") for key in TIMES})
    out[USERS] = np.load(split_path / f"{USERS}.npy")
    return out


def swap_training(data: dict[str, Any]) -> dict[str, Any]:
    """``data`` with its training views replaced by the train+validation set, and the catalogue indices to match."""
    missing = [key for key in (*MATRICES, *SEQUENCES, USERS) if key not in data]
    if missing:
        raise ValueError(f"the split has no train+validation set ({missing[0]} is missing); run `seqrec-eval prepare` "
                         "with `refit = true` in [protocol]")
    n_items = len(data["val_item_ids"])
    out = dict(data)
    out.update(
        x_train=data["x_refit"], train_source_matrix=data["refit_source_matrix"],
        train_target_matrix=data["refit_target_matrix"], x_train_sequences=data["x_refit_sequences"],
        train_source_sequences=data["refit_source_sequences"], train_user_ids=data[USERS],
        train_item_ids=np.asarray(data["val_item_ids"]),
        # the refitted model knows every validation item: none is cold to it any more
        warm_item_indices=np.arange(n_items, dtype=np.int64),
        val_cold_item_indices=np.zeros(0, dtype=np.int64),
    )
    for times, refit in (("x_train_timestamps", "x_refit_timestamps"),
                         ("train_source_timestamps", "refit_source_timestamps")):
        out[times] = data.get(refit)
    return out
