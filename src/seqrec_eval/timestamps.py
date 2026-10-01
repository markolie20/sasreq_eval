"""The time of every event in a prepared split, recovered and proved.

The library's temporal builder sorts each history by time and cuts the three
windows by time, but the split it saves stores a history as items in order,
without their times. A time-decayed baseline needs them, so they are recovered
here and stored beside the split, without touching the library.

Recovery is a join, not a second split. Everything that defines a phase's
histories is saved with the split: the phase's users, its items, and the time
its window starts. The builder builds a history from the events before that
time whose user and item survived the phase's filtering, ordered by (user,
time) with a stable sort, so ties keep the source order. So:

1. the prepared events are rebuilt exactly as the builder rebuilt them -- the
   same dataset adapter, options and seed, the same rating threshold;
2. each history is assembled from them by that same rule;
3. the result must equal the split's own history, item for item. Only then are
   the times saved; any difference fails with the first user and position that
   disagree.

Step 3 is what makes this safe. If the library ever builds histories another
way, the join does not quietly attach the wrong times: it refuses.

The same join, run on the validation and test *target* windows, orders each
user's targets in time. It must reproduce the saved target matrix exactly, pair
for pair; then each user's **next-item target** is saved: what they did at the
first moment after their history ends -- every item at that moment, since
events can tie:

    val_next_target_matrix.npz, test_next_target_matrix.npz

That first moment is taken over **every** event of the user in the window,
including events on items the stage then deleted: the builder keeps an item
first seen in the window only with ``item_min_support`` users counted inside
the window, so a rare new item vanishes from the split (H16). Taking the first
moment among surviving events only would silently replace such a user's real
next item with a later one -- a next-but-one task that costs sequential models
most. Instead the row keeps only the surviving items of the real first moment,
and is empty when none survived; :func:`seqrec_eval.evaluate.scored_rows` then
leaves the user out of next-item scoring, as it does anyone whose next item no
model can recommend (``NEXT_TARGETS_VERSION`` 2).

Times are unix seconds, converted from the dataset's own unit exactly as the
builder converts them, and stored as float64, one array per history view,
aligned with its ``values``:

    x_train_timestamps.npy        with x_train_sequences      (the training window)
    train_source_timestamps.npy   with train_source_sequences
    val_source_timestamps.npy     with val_source_sequences
    test_source_timestamps.npy    with test_source_sequences
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from compresso_recsys import builder
from scipy.sparse import csr_matrix, load_npz, save_npz

#: Each stored history view, the times file beside it, the stage whose users
#: and items define its rows and columns, and the boundary its events precede.
VIEWS = {
    "x_train_sequences": ("x_train_timestamps", "train", "validation_target_start"),
    "train_source_sequences": ("train_source_timestamps", "train", "train_target_start"),
    "val_source_sequences": ("val_source_timestamps", "val", "validation_target_start"),
    "test_source_sequences": ("test_source_timestamps", "test", "test_target_start"),
}
TIMESTAMP_KEYS = {view: times for view, (times, _, _) in VIEWS.items()}
#: Each phase's target window: from its own boundary to the next (the test window runs to the end).
TARGET_WINDOWS = {"val": ("validation_target_start", "test_target_start"), "test": ("test_target_start", None)}
NEXT_TARGET_KEYS = {phase: f"{phase}_next_target_matrix" for phase in TARGET_WINDOWS}
#: 1: the first moment among the events the split kept. 2: the first moment among all the user's events in
#: the window; items the stage deleted leave the row, and an empty row means the real next item was deleted.
NEXT_TARGETS_VERSION = 2


class TimestampAlignmentError(RuntimeError):
    """The recovered histories do not reproduce the split's own."""


def prepared_events(params: dict[str, Any], data_dir: Path) -> pd.DataFrame:
    """The builder's prepared events for ``params``: users, items, unix-second times and values, in source order.

    This repeats the builder's steps up to the split: resolve the arguments,
    seed as it seeds, load the adapter's interactions, and apply the same
    preprocessing, which for a temporal split is the rating threshold alone
    (support filters run per stage, inside the split).

    ``user_code`` is the builder's own user number: it factorizes the ids,
    sorted, in their original type, and a stage's rows follow that order. The
    ids here are strings (an integer id would sort differently as text), so
    the order is carried beside them.
    """
    resolved, spec = builder._resolve_args(builder._build_args(**params, data_dir=str(data_dir)))
    if resolved.split_mode != "temporal":
        raise ValueError(f"timestamps can only be recovered for a temporal split, not {resolved.split_mode!r}")
    random.seed(resolved.seed)
    np.random.seed(resolved.seed)
    dataset = builder._make_dataset(resolved, spec)
    events = dataset.preprocess_interactions_for_recsys(
        dataset.get_interactions(),
        min_value_to_keep=resolved.min_value_to_keep,
        user_min_support=1,
        item_min_support=1,
        set_all_values_to=resolved.set_all_values_to,
    )
    seconds = builder._timestamps_in_seconds(events["timestamp"])
    finite = np.isfinite(seconds)
    user_codes, _ = pd.factorize(events["user_id"][finite], sort=True)
    return pd.DataFrame({
        "user_id": events["user_id"].astype(str).to_numpy()[finite],
        "item_id": events["item_id"].astype(str).to_numpy()[finite],
        "timestamp": seconds[finite],
        "user_code": user_codes.astype(np.int64, copy=False),
        "value": events["value"].to_numpy(dtype=np.float32)[finite],
    })


def _boundaries(manifest: dict[str, Any]) -> dict[str, float]:
    stage = manifest.get("stages", {}).get("data", {})
    keys = ("train_target_start", "validation_target_start", "test_target_start")
    missing = [key for key in keys if key not in stage]
    if missing:
        raise ValueError(f"the split's manifest records no {missing}; was it built with split_mode='temporal'?")
    return {key: float(stage[key]) for key in keys}


def align(events: pd.DataFrame, data: dict[str, Any], manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    """The time of every event of every history view in ``data``, or :class:`TimestampAlignmentError`."""
    boundaries = _boundaries(manifest)
    times = events["timestamp"].to_numpy(dtype=np.float64)
    out = {}
    for view, (key, stage, boundary) in VIEWS.items():
        sequences = data.get(view)
        if sequences is None:
            continue
        users = data[f"{stage}_user_ids"]
        if users is None:
            raise ValueError(f"the split stores no {stage}_user_ids, so {view} cannot be joined")
        rows = pd.Index(np.asarray(users).astype(str)).get_indexer(events["user_id"])
        columns = pd.Index(np.asarray(data[f"{stage}_item_ids"]).astype(str)).get_indexer(events["item_id"])
        keep = (times < boundaries[boundary]) & (rows >= 0) & (columns >= 0)
        rows, columns, kept = rows[keep], columns[keep], times[keep]
        order = np.lexsort((kept, rows))  # stable: equal times keep the source order, as in the builder
        rows, columns, kept = rows[order], columns[order], kept[order]

        lengths = np.bincount(rows, minlength=sequences.n_rows)
        if lengths.size != sequences.n_rows or not np.array_equal(lengths, sequences.row_lengths):
            differ = np.flatnonzero(lengths[:sequences.n_rows] != sequences.row_lengths)
            row = int(differ[0]) if differ.size else sequences.n_rows
            raise TimestampAlignmentError(
                f"{view}: user {np.asarray(users)[row] if row < len(users) else row} has "
                f"{int(sequences.row_lengths[row]) if row < sequences.n_rows else 0} events in the split but "
                f"{int(lengths[row]) if row < lengths.size else 0} before the boundary in the prepared data"
            )
        mismatch = np.flatnonzero(columns != sequences.values)
        if mismatch.size:
            event = int(mismatch[0])
            row = int(np.searchsorted(sequences.indptr, event, side="right") - 1)
            raise TimestampAlignmentError(
                f"{view}: user {np.asarray(users)[row]} differs at position {event - int(sequences.indptr[row])}: "
                f"the split has item {data[f'{stage}_item_ids'][sequences.values[event]]!r}, the prepared data "
                f"{data[f'{stage}_item_ids'][columns[event]]!r} ({mismatch.size:,} events differ in all)"
            )
        out[key] = kept
    return out


def next_targets(events: pd.DataFrame, data: dict[str, Any], manifest: dict[str, Any]) -> dict[str, csr_matrix]:
    """Each phase's next-item targets, after proving the target window reproduces the saved targets."""
    boundaries = _boundaries(manifest)
    times = events["timestamp"].to_numpy(dtype=np.float64)
    out = {}
    for phase, (start, end) in TARGET_WINDOWS.items():
        targets = data.get(f"{phase}_target_matrix")
        users = data.get(f"{phase}_user_ids")
        if targets is None or users is None:
            continue
        targets = targets.tocsr()
        n_rows, n_items = targets.shape
        rows = pd.Index(np.asarray(users).astype(str)).get_indexer(events["user_id"])
        columns = pd.Index(np.asarray(data[f"{phase}_item_ids"]).astype(str)).get_indexer(events["item_id"])
        in_window = (times >= boundaries[start]) & (rows >= 0)
        if end is not None:
            in_window &= times < boundaries[end]
        # every event of the phase's users in the window, on any item, the stage's deletions included
        all_rows, all_columns, all_times = rows[in_window], columns[in_window], times[in_window]
        surviving = all_columns >= 0
        rows, columns, kept = all_rows[surviving], all_columns[surviving], all_times[surviving]

        # the window's (user, item) pairs must be exactly the saved targets
        pairs = np.unique(rows.astype(np.int64) * n_items + columns)
        saved_rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(targets.indptr))
        saved = np.unique(saved_rows * n_items + targets.indices)
        if not np.array_equal(pairs, saved):
            extra, missing = np.setdiff1d(pairs, saved), np.setdiff1d(saved, pairs)
            first = int((extra if extra.size else missing)[0])
            raise TimestampAlignmentError(
                f"{phase} targets: user {np.asarray(users)[first // n_items]} and item "
                f"{data[f'{phase}_item_ids'][first % n_items]!r} are "
                f"{'in the prepared window but not the split' if extra.size else 'in the split but not the window'} "
                f"({extra.size:,} extra, {missing.size:,} missing pairs in all)"
            )

        # per user, the surviving items at the first moment of the window over *all* their events (H16):
        # an item the stage deleted is not replaced by a later event, it just leaves the row
        first_time = np.full(n_rows, np.inf)
        np.minimum.at(first_time, all_rows, all_times)
        at_first = kept == first_time[rows]
        matrix = csr_matrix((np.ones(int(at_first.sum()), dtype=np.float32), (rows[at_first], columns[at_first])),
                            shape=(n_rows, n_items), dtype=np.float32)
        matrix.sum_duplicates()
        matrix.data[:] = 1.0
        matrix.sort_indices()
        out[phase] = matrix
    return out


def attach_timestamps(params: dict[str, Any], data_dir: Path, split_path: Path, data: dict[str, Any],
                      manifest: dict[str, Any], *, events: pd.DataFrame | None = None) -> dict[str, Any]:
    """Recover, prove and save the event times and next-item targets of the split at ``split_path``.

    ``events`` are the prepared events, if the caller already has them. Returns
    the summary recorded in ``split_info.json``.
    """
    if events is None:
        events = prepared_events(params, data_dir)
    aligned = align(events, data, manifest)
    upcoming = next_targets(events, data, manifest)
    for key, values in aligned.items():
        np.save(Path(split_path) / f"{key}.npy", values)
    for phase, matrix in upcoming.items():
        save_npz(Path(split_path) / f"{NEXT_TARGET_KEYS[phase]}.npz", matrix)
    return {
        "unit": "unix_seconds", "aligned": sorted(aligned),
        "events": {key: int(values.size) for key, values in aligned.items()},
        "next_targets_version": NEXT_TARGETS_VERSION,
        "next_targets": {phase: {"users_with_a_target": int((np.diff(m.indptr) > 0).sum()),
                                 # the real next item was deleted by the stage's new-item filter (H16)
                                 "users_whose_next_item_was_deleted": int((np.diff(m.indptr) == 0).sum()),
                                 "mean_items": float(m.nnz / max(1, (np.diff(m.indptr) > 0).sum()))}
                         for phase, m in upcoming.items()},
    }


def load_timestamps(split_path: Path) -> dict[str, Any]:
    """The recovered event times and next-item targets of a split, whichever exist."""
    out: dict[str, Any] = {}
    for key in TIMESTAMP_KEYS.values():
        path = Path(split_path) / f"{key}.npy"
        if path.exists():
            out[key] = np.load(path)
    for key in NEXT_TARGET_KEYS.values():
        path = Path(split_path) / f"{key}.npz"
        if path.exists():
            out[key] = load_npz(path).tocsr()
    return out
