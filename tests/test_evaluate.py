"""The seen-item mask of ablation conditions, and the evaluator order it relies on (H33).

A batch carries no user ids, so :class:`ExcludeSeenPolicy` finds a batch's seen
rows with a running offset. That is right only while the library's evaluator
hands over every row once, contiguously and in order. The first test pins that
behaviour of the library, so a change to it fails here rather than in the
results; the rest show the policy refusing batches that are not the rows it
assumes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from compresso import SRPTensor
from compresso_recsys import ItemSequences
from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import HitRate
from scipy.sparse import csr_matrix

from seqrec_eval.evaluate import BatchAlignmentError, ExcludeSeenPolicy

N_ITEMS = 40


class Recorder:
    """Ranks every item by index, lowest first, and records the rows of each call."""

    def __init__(self, n_items: int = N_ITEMS) -> None:
        self.n_items = n_items
        self.calls: list[list[list[int]]] = []

    def predict_on_batch(self, source, *, k, exclude_seen=True, candidate_ids=None):
        if isinstance(source, csr_matrix):
            rows = [sorted(source[r].indices.tolist()) for r in range(source.shape[0])]
        else:
            rows = [source.row(r).tolist() for r in range(source.n_rows)]
        self.calls.append(rows)
        cols = torch.arange(k).repeat(len(rows), 1)
        return SRPTensor(cols=cols, vals=-cols.float(), shape=(len(rows), self.n_items))


def _users(n: int):
    """User r has seen items r and r + 1, and the input (a truncated history) keeps only r + 1."""
    seen = csr_matrix((np.ones(2 * n, dtype=np.float32),
                       (np.repeat(np.arange(n), 2), np.stack([np.arange(n), np.arange(n) + 1], 1).ravel())),
                      shape=(n, N_ITEMS))
    inputs = ItemSequences.from_rows([[r + 1] for r in range(n)], n_items=N_ITEMS)
    return seen, inputs


def test_the_library_evaluator_passes_every_row_once_contiguously_and_in_order():
    n, batch = 23, 5
    model = Recorder()
    source = ItemSequences.from_rows([[r] for r in range(n)], n_items=N_ITEMS)
    targets = csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), np.zeros(n))), shape=(n, N_ITEMS))
    evaluate_recommender(model, source=source, targets=targets, metrics=[HitRate([1])], batch_size=batch)
    assert [len(call) for call in model.calls] == [5, 5, 5, 5, 3]
    assert [row for call in model.calls for row in call] == [[r] for r in range(n)]


def test_the_mask_lands_on_the_right_users_across_batches():
    n = 11
    seen, inputs = _users(n)
    # ranked by index, user r's first item neither seen (r, r + 1) nor already taken: 0 unless r = 0, then 2
    expected = np.array([2] + [0] * (n - 1))
    targets = csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), expected)), shape=(n, N_ITEMS))
    policy = ExcludeSeenPolicy(Recorder(), True, seen=seen, limit=N_ITEMS)
    result = evaluate_recommender(policy, source=inputs, targets=targets, metrics=[HitRate([1])], batch_size=4)
    policy.finish()
    assert result.metrics["hit_rate@1"] == 1.0


def test_a_batch_of_other_rows_is_refused():
    seen, inputs = _users(8)
    policy = ExcludeSeenPolicy(Recorder(), True, seen=seen, limit=N_ITEMS)
    # an evaluator that started at row 4: the policy takes it for rows 0-3, whose histories lack its items
    with pytest.raises(BatchAlignmentError, match="row 0 of the evaluation reads item 5"):
        policy.predict_on_batch(inputs.take_rows(4, 8), k=1)


def test_a_reordered_batch_is_refused():
    seen, inputs = _users(8)
    policy = ExcludeSeenPolicy(Recorder(), True, seen=seen, limit=N_ITEMS)
    with pytest.raises(BatchAlignmentError):
        policy.predict_on_batch(inputs.select_rows([1, 0, 2, 3]), k=1)


def test_skipped_or_repeated_rows_are_refused_at_the_end():
    seen, inputs = _users(8)
    skipped = ExcludeSeenPolicy(Recorder(), True, seen=seen, limit=N_ITEMS)
    skipped.predict_on_batch(inputs.take_rows(0, 4), k=1)
    with pytest.raises(BatchAlignmentError, match="passed 4 rows, but the seen matrix has 8"):
        skipped.finish()
    repeated = ExcludeSeenPolicy(Recorder(), True, seen=seen, limit=N_ITEMS)
    repeated.predict_on_batch(inputs, k=1)
    repeated.finish()
    with pytest.raises(BatchAlignmentError):
        repeated.predict_on_batch(inputs.take_rows(0, 2), k=1)  # a second pass has no rows left


def test_a_matrix_source_is_checked_in_the_seen_space():
    # the source's columns are a training catalogue that maps onto the seen space in reverse
    seen, _ = _users(3)
    to_seen = np.arange(N_ITEMS)[::-1].copy()
    wanted = [[1], [2], [3]]  # seen-space items, each in its user's history
    columns = [int(np.flatnonzero(to_seen == item[0])[0]) for item in wanted]
    source = csr_matrix((np.ones(3, dtype=np.float32), (np.arange(3), columns)), shape=(3, N_ITEMS))
    mapped = ExcludeSeenPolicy(Recorder(), True, seen=seen, limit=N_ITEMS, source_columns=to_seen)
    mapped.predict_on_batch(source, k=1)
    mapped.finish()
    unmapped = ExcludeSeenPolicy(Recorder(), True, seen=seen, limit=N_ITEMS)
    with pytest.raises(BatchAlignmentError):
        unmapped.predict_on_batch(source, k=1)


def test_without_a_seen_matrix_nothing_is_checked():
    _, inputs = _users(4)
    policy = ExcludeSeenPolicy(Recorder(), True)
    policy.predict_on_batch(inputs.select_rows([3, 2, 1, 0]), k=1)
    policy.finish()


def test_a_batch_that_is_not_the_next_rows_of_the_source_is_refused():
    seen, inputs = _users(8)
    policy = ExcludeSeenPolicy(Recorder(), True, seen=seen, limit=N_ITEMS, source=inputs)
    with pytest.raises(BatchAlignmentError, match="is not rows 0-3 of the source"):
        policy.predict_on_batch(inputs.take_rows(4, 8), k=1)


def test_overlapping_histories_fool_the_subset_check_but_not_the_row_check():
    # every user has seen items 0-9, so any user's input is inside any other's history
    n = 6
    seen = csr_matrix(np.ones((n, 10), dtype=np.float32), shape=(n, 10))
    seen = csr_matrix((seen.data, seen.indices, seen.indptr), shape=(n, N_ITEMS))
    inputs = ItemSequences.from_rows([[r] for r in range(n)], n_items=N_ITEMS)
    reordered = inputs.select_rows([2, 3, 0, 1, 4, 5])
    subset_only = ExcludeSeenPolicy(Recorder(), True, seen=seen, limit=N_ITEMS)
    subset_only.predict_on_batch(reordered.take_rows(0, 3), k=1)  # passes: the weakness the row check covers
    with_rows = ExcludeSeenPolicy(Recorder(), True, seen=seen, limit=N_ITEMS, source=inputs)
    with pytest.raises(BatchAlignmentError):
        with_rows.predict_on_batch(reordered.take_rows(0, 3), k=1)


def _tie_heavy_popularity(rng: np.random.Generator, n_items: int):
    """The library's popularity model on counts that tie everywhere."""
    from compresso_recsys.models.baselines import PopularityBaseline

    events = rng.integers(0, n_items, 3_000)
    train = csr_matrix((np.ones(events.size, np.float32), (rng.integers(0, 500, events.size), events)),
                       shape=(500, n_items))
    train.sum_duplicates()
    train.data[:] = 1.0
    return PopularityBaseline().fit(train)


def test_one_width_per_phase_makes_a_users_list_independent_of_its_batch():
    # review N30: asked for k + the heaviest seen row *of its batch*, a model whose scores tie answered with lists
    # that depended on who shared the batch, so on eval_batch_size; one width for the whole phase cannot
    rng = np.random.default_rng(0)
    n_items, n_users = 300, 400
    model = _tie_heavy_popularity(rng, n_items)
    lengths = rng.integers(1, 30, n_users)
    lengths[7] = 200  # one heavy user, in one batch only
    rows = np.repeat(np.arange(n_users), lengths)
    seen = csr_matrix((np.ones(rows.size, np.float32), (rows, rng.integers(0, n_items, rows.size))),
                      shape=(n_users, n_items))
    seen.sum_duplicates()
    seen.data[:] = 1.0
    extra = int(np.diff(seen.indptr).max())

    def lists(batch, width):
        policy = ExcludeSeenPolicy(model, True, seen=seen, limit=n_items, source=seen, extra=width)
        return np.vstack([policy.predict_on_batch(seen[start:start + batch], k=10).cols.numpy()
                          for start in range(0, n_users, batch)])

    assert np.array_equal(lists(1, extra), lists(400, extra))
    assert np.array_equal(lists(16, extra), lists(400, extra))
    assert not np.array_equal(lists(16, None), lists(400, None))  # the data does tie where widths differ
    with pytest.raises(ValueError, match="less than a batch row"):
        lists(400, 3)

