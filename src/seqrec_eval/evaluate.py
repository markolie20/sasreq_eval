"""Scoring a fitted model on one phase of a split.

Three things happen around the library's ``evaluate_recommender``.

The model is wrapped in :class:`~compresso_recsys.models.WarmCatalogAdapter`.
It was fitted on the training catalogue, but the validation and test catalogues
append the items first seen in their windows. The adapter hands the model its
own item space and maps its rankings back, so those new items stay valid
targets that no model can recommend -- equally for every model. A sequence
history is handed over as it is (the model's tokenizer reads a new item as
``unk``, in place); a matrix history is first projected onto the training
catalogue, because a matrix model has no column for a new item, and the
adapter refuses a wider matrix rather than guess.

``exclude_seen`` is fixed per dataset. ``evaluate_recommender`` calls
``predict_on_batch(source, k=...)`` and nothing more, so the policy has to be
bound to the model before it is handed over.

What is excluded is the user's *whole* history, even when the model is shown
less of it. An ablation that truncates or thins the input histories changes
what a model reads, not what the user has already seen: a film rated five
years ago is still not a recommendation. Such a condition stores the original
histories as ``{phase}_seen_matrix`` and the policy masks those instead of the
input -- the library's own contract, "truncation is not exclusion".

Per-user values are always collected, because the paired comparisons in the
report need them.

Two target definitions exist (``[protocol].targets``). "window" is everything
a user does in the phase's window -- a year on ML-20M. "next" is only the first
thing they do after their history ends, which is what next-item prediction
and serving are about; it is recovered and proved with the event times (see
:mod:`seqrec_eval.timestamps`). The protocol's choice drives selection, the
floor and the statistics, and the other is reported as a diagnostic.

Next-item targets are scored only for users whose next item a model can
recommend (:func:`scored_rows`, H16). A next item first seen after training,
or deleted by the builder's new-item filter, cannot be recommended by any
model: such a user scores 0 for every one, carries nothing for a comparison,
and only lowers every mean. They are counted in the result's metadata. Window
targets, the diagnostic, are scored on every user with a target.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from compresso_recsys.evaluation import EvaluationResult, evaluate_recommender
from compresso_recsys.metrics import (
    MAP,
    MRR,
    NDCG,
    CalibratedRecall,
    HitRate,
    Precision,
    Recall,
)
from compresso_recsys.models import WarmCatalogAdapter
from scipy.sparse import csr_matrix

from .protocol import METRIC_NAMES, TARGET_DEFINITIONS, Protocol
from .splits import Split

METRIC_CLASSES = {cls.result_prefix: cls for cls in (NDCG, Recall, CalibratedRecall, HitRate, Precision, MAP, MRR)}
if set(METRIC_CLASSES) != set(METRIC_NAMES):  # the protocol validates names against METRIC_NAMES
    raise ImportError(f"metric names drifted from the library: {sorted(METRIC_CLASSES)} vs {sorted(METRIC_NAMES)}")


class BatchAlignmentError(RuntimeError):
    """The evaluator's batches are not the rows the seen matrix describes."""


class ExcludeSeenPolicy:
    """Bind ``exclude_seen`` to a model, since the evaluator never passes it.

    With ``seen`` -- a CSR over the phase catalogue, one row per row the
    evaluator will pass, in order -- items are excluded against it rather than
    against the input. The model is asked for its top ``k + extra``, with its
    own filter off; the seen items are dropped and the first ``k`` remain, in
    the model's order. ``extra`` is the most items any user **of the whole
    phase** has seen, so every batch asks for the same length: a top-k breaks
    ties differently for a different length, so a width taken per batch (as
    until 2026-10-01) made a tie-heavy model's lists depend on the batch size
    and on which users shared a batch (review N30). Without ``extra``, the
    batch's heaviest user sets it. ``limit`` is how many items the model can
    return at all.

    A batch carries no user ids, so which ``seen`` rows belong to it is a
    running offset: correct only if the evaluator hands over every row once,
    contiguously and in order, as the library's does today. A reordered,
    skipped or repeated batch would put each mask on the wrong user and score
    silently wrong, so none of that is trusted (H33); each breach raises
    :class:`BatchAlignmentError`:

    - with ``source`` (what the evaluator is given), every batch must *equal*
      the source's next rows, content and order -- so the offset names them;
    - every item a batch row reads must be in its ``seen`` row: a transform
      only drops events, so the input is part of the history, and a ``seen``
      matrix whose rows do not line up with the source's fails here;
    - :meth:`finish` requires every row to have been used exactly once.

    The second alone would not do: users whose histories overlap (popular
    items, short truncated inputs) can pass it for one another.
    ``source_columns`` maps the source's columns into the ``seen`` space when
    they differ: a matrix source is projected onto the training catalogue.
    """

    def __init__(self, model: Any, exclude_seen: bool, *, seen: csr_matrix | None = None,
                 limit: int | None = None, source_columns: np.ndarray | None = None, source=None,
                 extra: int | None = None) -> None:
        self.model = model
        self.source = source
        self.exclude_seen = exclude_seen
        self.seen = seen
        self.limit = limit
        self.extra = extra
        self.source_columns = None if source_columns is None else np.asarray(source_columns, dtype=np.int64)
        self._offset = 0

    @property
    def masking(self) -> bool:
        return bool(self.exclude_seen) and self.seen is not None

    def _check_rows(self, batch, n_rows: int) -> None:
        """The batch is the source's next ``n_rows`` rows, exactly."""
        if self.source is None:
            return
        if isinstance(self.source, csr_matrix):
            expected = self.source[self._offset:self._offset + n_rows]
            same = (isinstance(batch, csr_matrix) and batch.shape == expected.shape
                    and (batch != expected).nnz == 0)
        else:
            expected = self.source.take_rows(self._offset, min(self._offset + n_rows, self.source.n_rows))
            same = (not isinstance(batch, csr_matrix) and expected.n_rows == n_rows
                    and np.array_equal(batch.row_lengths, expected.row_lengths)
                    and np.array_equal(batch.values, expected.values))
        if not same:
            raise BatchAlignmentError(
                f"the evaluator's batch of {n_rows:,} rows is not rows {self._offset:,}-{self._offset + n_rows - 1:,} "
                "of the source it was given: it reordered, skipped or repeated rows, so every mask would land on "
                "another user")

    def _check_batch(self, source, seen: csr_matrix) -> None:
        """Every item a batch row reads is in that row's seen history, or the batch is not the rows assumed."""
        if isinstance(source, csr_matrix):
            rows = np.repeat(np.arange(source.shape[0], dtype=np.int64), np.diff(source.indptr))
            items = source.indices.astype(np.int64)
        else:
            rows = np.repeat(np.arange(source.n_rows, dtype=np.int64), source.row_lengths)
            items = np.asarray(source.values, dtype=np.int64)
        if self.source_columns is not None:
            items = self.source_columns[items]
        width = np.int64(max(seen.shape[1], int(items.max()) + 1 if items.size else 1))
        seen_keys = np.repeat(np.arange(seen.shape[0], dtype=np.int64), np.diff(seen.indptr)) * width + seen.indices
        inside = np.isin(rows * width + items, seen_keys)
        if not inside.all():
            event = int(np.flatnonzero(~inside)[0])
            raise BatchAlignmentError(
                f"row {self._offset + int(rows[event])} of the evaluation reads item {int(items[event])}, which its "
                f"seen history lacks ({int((~inside).sum()):,} such events in this batch): the evaluator's batches are "
                "not the rows the seen matrix was built for, so every mask would land on another user")

    def finish(self) -> None:
        """After the evaluation: every seen row was used, exactly once."""
        if self.masking and self._offset != self.seen.shape[0]:
            raise BatchAlignmentError(f"the evaluator passed {self._offset:,} rows, but the seen matrix has "
                                      f"{self.seen.shape[0]:,}: some rows were skipped or passed twice")

    def predict_on_batch(self, source, *, k: int):
        if not self.masking:
            return self.model.predict_on_batch(source, k=k, exclude_seen=self.exclude_seen)
        import torch
        from compresso import SRPTensor

        n_rows = source.n_rows if hasattr(source, "n_rows") else source.shape[0]
        seen = self.seen[self._offset:self._offset + n_rows]
        if seen.shape[0] != n_rows:
            raise BatchAlignmentError("the seen matrix has fewer rows than the evaluator passed")
        self._check_rows(source, n_rows)
        self._check_batch(source, seen)
        self._offset += n_rows
        heaviest = int(np.diff(seen.indptr).max()) if n_rows else 0
        if self.extra is not None and self.extra < heaviest:
            raise ValueError(f"extra={self.extra} is less than a batch row's {heaviest} seen items")
        extra = heaviest if self.extra is None else self.extra
        wide = k + extra if self.limit is None else min(k + extra, self.limit)
        ranked = self.model.predict_on_batch(source, k=wide, exclude_seen=False)
        cols = ranked.cols.cpu().numpy()
        rows = np.repeat(np.arange(n_rows, dtype=np.int64), cols.shape[1])
        width = np.int64(max(seen.shape[1], int(cols.max()) + 1 if cols.size else 1))
        seen_keys = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(seen.indptr)) * width + seen.indices
        excluded = np.isin(rows * width + cols.ravel(), seen_keys).reshape(cols.shape)
        available = (~excluded).sum(axis=1)
        if n_rows and available.min() < k:
            row = int(np.argmin(available))
            raise ValueError(f"source row {self._offset - n_rows + row} has only {int(available[row])} unseen "
                             f"items among the model's top {wide}, fewer than k={k}")
        # a stable sort puts each row's unseen items first, still in the model's order
        order = np.argsort(excluded, axis=1, kind="stable")[:, :k]
        index = torch.from_numpy(order).to(ranked.cols.device)
        return SRPTensor(cols=torch.gather(ranked.cols, 1, index), vals=torch.gather(ranked.vals, 1, index),
                         shape=ranked.shape)


def recommendable_next(targets: csr_matrix, n_known: int) -> np.ndarray:
    """Per row: does its next-item target hold an item inside the model's catalogue (the first ``n_known``)?

    False for an empty row: the user's real next item was deleted by the builder (see
    :mod:`seqrec_eval.timestamps`).
    """
    targets = targets.tocsr()
    rows = np.repeat(np.arange(targets.shape[0], dtype=np.int64), np.diff(targets.indptr))
    return np.bincount(rows, weights=targets.indices < n_known, minlength=targets.shape[0]) > 0


def seen_history(data: dict[str, Any], phase: str) -> csr_matrix:
    """What each user of ``phase`` has seen: the whole history, also where a condition truncated the input."""
    seen = data.get(f"{phase}_seen_matrix")
    return data[f"{phase}_source_matrix"] if seen is None else seen


def fills_list(data: dict[str, Any], phase: str, k: int) -> np.ndarray:
    """Per row: does the training catalogue hold ``k`` items the user has not seen, so excluding seen items
    still leaves a full top-k list?

    False only where the catalogue is small beside a history: a catalogue sweep's smallest levels, for its
    heaviest users (review N49). Such a row would stop the whole evaluation (:class:`ExcludeSeenPolicy`), so
    it is not scored under ``exclude_seen``.
    """
    seen = seen_history(data, phase).tocsr()
    if not seen.has_canonical_format:
        seen = seen.copy()
        seen.sum_duplicates()
    limit = len(data["train_item_ids"])  # the training catalogue is the first columns of the phase's
    rows = np.repeat(np.arange(seen.shape[0], dtype=np.int64), np.diff(seen.indptr))
    inside = np.bincount(rows, weights=seen.indices < limit, minlength=seen.shape[0])
    return limit - inside >= k


def reachable_targets(data: dict[str, Any], phase: str, targets: csr_matrix, exclude_seen: bool) -> csr_matrix:
    """``targets`` less those that excluding seen items puts out of reach: an item already in the history."""
    return new_item_targets(targets, seen_history(data, phase)) if exclude_seen else targets


def scored_rows(split: Split, phase: str, definition: str, *, exclude_seen: bool = False,
                targets: csr_matrix | None = None) -> np.ndarray | None:
    """The rows scored on ``definition``'s targets: for next-item targets, the users whose next item a model
    fitted on ``split`` can recommend; ``None`` (every row) for window targets.

    The catalogue is the split's training catalogue, so under refit (:func:`~seqrec_eval.splits.final_split`)
    the items first seen in validation count as recommendable at test. With ``exclude_seen`` a next item already
    in the user's history cannot be recommended either (review C4: on Amazon, variants of one product share an
    item). ``targets`` are the targets actually scored, if not the definition's own (the "new" diagnostic).
    """
    if definition != "next":
        return None
    targets = phase_targets(split, phase, "next") if targets is None else targets
    known = recommendable_next(reachable_targets(split.data, phase, targets, exclude_seen),
                               len(split.data["train_item_ids"]))
    return np.flatnonzero(known).astype(np.int64)


def build_metrics(protocol: Protocol) -> list:
    return [METRIC_CLASSES[name](protocol.cutoffs) for name in protocol.metrics]


def new_item_targets(targets: csr_matrix, source: csr_matrix) -> csr_matrix:
    """Only the targets that are not already in the user's history."""
    seen = source.copy()
    seen.data[:] = 1.0
    remaining = (targets - targets.multiply(seen)).tocsr()
    remaining.eliminate_zeros()
    return remaining


def target_key(phase: str, definition: str) -> str:
    if definition not in TARGET_DEFINITIONS:
        raise ValueError(f"target definition must be one of {TARGET_DEFINITIONS}, got {definition!r}")
    return f"{phase}_target_matrix" if definition == "window" else f"{phase}_next_target_matrix"


def other_definition(definition: str) -> str:
    return "window" if definition == "next" else "next"


def phase_targets(split: Split, phase: str, definition: str) -> csr_matrix:
    matrix = split.data.get(target_key(phase, definition))
    if matrix is None:
        raise ValueError(f"{split.dataset} has no {definition!r} targets for {phase}: next-item targets are "
                         "recovered with the event times, so run `seqrec-eval prepare` without --no-timestamps")
    return matrix


def phase_model(model: Any, split: Split, phase: str) -> WarmCatalogAdapter:
    return WarmCatalogAdapter(
        model,
        train_item_ids=split.data["train_item_ids"],
        catalog_item_ids=split.data[f"{phase}_item_ids"],
    )


def phase_source(split: Split, phase: str, family: str):
    key = f"{phase}_source_sequences" if family == "sequence" else f"{phase}_source_matrix"
    source = split.data[key]
    if source is None:
        raise ValueError(f"{split.dataset} has no {key}; sequential models need a chronological split")
    return source


def phase_inputs(model: Any, split: Split, phase: str, family: str):
    """The adapted model and the source it scores: a matrix source projected onto the training catalogue."""
    adapter = phase_model(model, split, phase)
    source = phase_source(split, phase, family)
    if family == "matrix":
        source = adapter.align_source(source)
    return adapter, source


def _take(source, rows: np.ndarray):
    return source.select_rows(rows) if hasattr(source, "select_rows") else source[rows]


def evaluate_phase(model: Any, split: Split, phase: str, *, family: str, protocol: Protocol,
                   exclude_seen: bool, rows: np.ndarray | None = None,
                   targets: str = "primary") -> EvaluationResult:
    """Score ``model`` on ``phase``.

    ``targets`` is "primary" (the protocol's definition), "next" or "window"
    explicitly, or "new": the protocol's targets not already in the history.
    """
    if targets not in ("primary", "new", *TARGET_DEFINITIONS):
        raise ValueError(f"targets must be 'primary', 'new' or one of {TARGET_DEFINITIONS}, got {targets!r}")
    if phase == "val" and split.trained_on != "train":
        raise ValueError(f"{split.dataset}: a model trained on {split.trained_on} has seen the validation window, "
                         "so its validation score would be leaked; score it on test")
    definition = protocol.targets if targets in ("primary", "new") else targets
    target_matrix = phase_targets(split, phase, definition)
    asked = target_matrix.shape[0] if rows is None else len(rows)  # every row, or the given sample
    if targets == "new":
        # against the whole history: a condition's input may be truncated
        target_matrix = new_item_targets(target_matrix, seen_history(split.data, phase))
    scorable = scored_rows(split, phase, definition, exclude_seen=exclude_seen, targets=target_matrix)
    if scorable is not None:
        rows = scorable if rows is None else rows[np.isin(rows, scorable)]
    unrecommendable = 0 if scorable is None else asked - len(rows)
    too_few_unseen = 0
    if exclude_seen:
        # a user who has seen all but fewer than k items of the catalogue cannot be given k unseen ones: left
        # out and counted, rather than stopping the evaluation (review N49)
        fills = fills_list(split.data, phase, max(protocol.cutoffs))
        if not fills.all():
            before = fills.size if rows is None else len(rows)
            rows = np.flatnonzero(fills).astype(np.int64) if rows is None else rows[fills[rows]]
            too_few_unseen = before - len(rows)
    adapter, source = phase_inputs(model, split, phase, family)
    sample_ids = split.eval_user_ids(phase)
    # Seen items are excluded after the model ranks, against the whole history, in every evaluation: inside
    # the model and after it, a top-k picks among tied scores differently, so a condition (which must exclude
    # after) and the full data would differ by ties alone (review A6). Models without ties lose nothing.
    seen = seen_history(split.data, phase) if exclude_seen else None
    # one width for every batch of the phase, from the heaviest user of the whole phase -- not of the batch, not of
    # the rows scored -- so a user's list depends on its own scores and history only (review N30)
    extra = int(np.diff(seen.indptr).max()) if seen is not None and seen.shape[0] else None
    if rows is not None:
        source = _take(source, rows)
        target_matrix = target_matrix[rows]
        sample_ids = sample_ids[rows]
        seen = None if seen is None else seen[rows]
    policy = ExcludeSeenPolicy(adapter, exclude_seen, seen=seen, limit=len(split.data["train_item_ids"]),
                               # a matrix source was projected onto the training catalogue; seen is the phase's
                               source_columns=adapter.train_to_catalog if family == "matrix" else None,
                               source=source, extra=extra)
    result = evaluate_recommender(
        policy,
        source=source,
        targets=target_matrix,
        metrics=build_metrics(protocol),
        sample_ids=sample_ids,
        collect_per_user=True,
        batch_size=protocol.eval_batch_size,
        metadata={"phase": phase, "exclude_seen": exclude_seen, "targets": targets, "definition": definition,
                  "rows_sampled": None if rows is None else len(rows),
                  # of the rows asked for (all, or the given sample), those whose next item no model can recommend
                  "rows_unrecommendable_next": unrecommendable,
                  # of the rest, those left out because the catalogue holds fewer than k items they have not seen
                  "rows_too_few_unseen": too_few_unseen},
    )
    policy.finish()
    return result
