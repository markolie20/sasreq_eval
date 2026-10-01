"""The §5.2 baselines, and the profile's timing fields, on histories small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from compresso_recsys import ItemSequences

from seqrec_eval.ablations import _time_stats, count_matrix, manipulation_check
from seqrec_eval.analysis import reversed_histories, shuffled_within_users
from seqrec_eval.baselines import (DAY, MarkovChain, Popularity, PopularityConfig, Replay, ReplayConfig,
                                   TimeDecayedPopularity, TimeDecayedPopularityConfig, _top_k, _top_k_full)
from seqrec_eval.evaluate import ExcludeSeenPolicy
from seqrec_eval.timestamps import TimestampAlignmentError, align


def ranked(model, rows, *, n_items, k=None, exclude_seen=False):
    source = ItemSequences.from_rows(rows, n_items=n_items)
    k = k or n_items - (max(len(set(r)) for r in rows) if exclude_seen else 0)
    return model.predict_on_batch(source, k=k, exclude_seen=exclude_seen).cols.numpy().tolist()


# ---------------------------------------------------------------------------
# popularity
# ---------------------------------------------------------------------------

def test_popularity_counts_events_or_users():
    # item 0: one user, five times; item 1: three users, once each
    train = ItemSequences.from_rows([[0, 0, 0, 0, 0], [1], [1], [1, 2]], n_items=3)
    by_events = Popularity(PopularityConfig("events")).fit(train)
    by_users = Popularity(PopularityConfig("users")).fit(train)
    assert ranked(by_events, [[2]], n_items=3, k=2) == [[0, 1]]
    assert ranked(by_users, [[2]], n_items=3, k=2) == [[1, 0]]


# ---------------------------------------------------------------------------
# time-decayed popularity
# ---------------------------------------------------------------------------

def _decay_data():
    # item 0: ten events a year ago; item 1: two events yesterday; item 2: never
    rows = [[0] * 10, [1, 1]]
    now = 400 * DAY
    times = np.array([now - 365 * DAY] * 10 + [now - DAY, now])
    return ItemSequences.from_rows(rows, n_items=3), times


def test_a_short_half_life_prefers_what_is_recent_and_a_long_one_what_is_frequent():
    train, times = _decay_data()
    short = TimeDecayedPopularity(TimeDecayedPopularityConfig(half_life_days=7)).fit(train, timestamps=times)
    long = TimeDecayedPopularity(TimeDecayedPopularityConfig(half_life_days=10_000)).fit(train, timestamps=times)
    assert ranked(short, [[2]], n_items=3) == [[1, 0, 2]]
    assert ranked(long, [[2]], n_items=3) == [[0, 1, 2]]


def test_an_event_decayed_to_almost_nothing_still_outranks_no_event():
    # item 0 has one event 1,000 half-lives ago (weight ~1e-301); item 1 none, so it is last
    train = ItemSequences.from_rows([[0], [1]], n_items=3)
    times = np.array([0.0, 1000 * DAY])
    model = TimeDecayedPopularity(TimeDecayedPopularityConfig(half_life_days=1)).fit(train, timestamps=times)
    assert model.weights_[0] > 0
    assert ranked(model, [[]], n_items=3) == [[1, 0, 2]]


def test_time_decayed_popularity_refuses_to_fit_without_timestamps():
    train, times = _decay_data()
    with pytest.raises(ValueError, match="timestamps"):
        TimeDecayedPopularity().fit(train)
    with pytest.raises(ValueError, match="timestamps for"):
        TimeDecayedPopularity().fit(train, timestamps=times[:-1])


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def test_replay_returns_the_history_most_recent_first_then_popularity():
    train = ItemSequences.from_rows([[4, 4, 4, 3, 3]], n_items=5)  # popularity: 4, then 3, then the rest
    model = Replay(ReplayConfig("recency")).fit(train)
    # history 1, 2, 0, 1: most recent is 1, then 0, then 2; then 4 and 3 by popularity
    assert ranked(model, [[1, 2, 0, 1]], n_items=5) == [[1, 0, 2, 4, 3]]


def test_replay_by_frequency_breaks_ties_by_recency():
    train = ItemSequences.from_rows([[4]], n_items=5)
    model = Replay(ReplayConfig("frequency")).fit(train)
    # 2 twice; 0 and 3 once each, 3 more recently
    assert ranked(model, [[2, 0, 2, 3]], n_items=5, k=3) == [[2, 3, 0]]


def test_replay_ignores_items_first_seen_after_training_and_reduces_to_popularity_when_excluding_seen():
    train = ItemSequences.from_rows([[2, 2, 1]], n_items=3)
    model = Replay().fit(train)
    # index 5 is beyond the fitted catalogue: a new item, which keeps its place but is not scored
    assert ranked(model, [[0, 5]], n_items=6, k=3) == [[0, 2, 1]]
    assert ranked(model, [[0]], n_items=3, k=2, exclude_seen=True) == [[2, 1]]


# ---------------------------------------------------------------------------
# Markov
# ---------------------------------------------------------------------------

def test_first_order_markov_predicts_the_most_frequent_successor():
    train = ItemSequences.from_rows([[0, 1, 2], [0, 1, 3], [0, 2]], n_items=4)
    model = MarkovChain().fit(train)
    # after 0: 1 twice, 2 once
    assert ranked(model, [[3, 0]], n_items=4, k=2) == [[1, 2]]


def test_a_rare_transition_outranks_a_popular_item_with_none():
    # item 3 is very popular but never follows 0; 0 -> 1 happens once in a hundred transitions from 0
    train = ItemSequences.from_rows([[3] * 500, *([[0, 2]] * 99), [0, 1]], n_items=4)
    model = MarkovChain().fit(train)
    top = ranked(model, [[0]], n_items=4, k=3)[0]
    assert top.index(1) < top.index(3)


def test_markov_falls_back_to_popularity_for_an_item_without_successors_or_a_new_one():
    train = ItemSequences.from_rows([[0, 1], [4, 4, 4, 4]], n_items=7)
    model = MarkovChain().fit(train)
    assert ranked(model, [[0]], n_items=7, k=1) == [[1]]
    assert ranked(model, [[2]], n_items=7, k=1) == [[4]]  # 2 has no successor
    assert ranked(model, [[9]], n_items=10, k=1) == [[4]]  # 9 is beyond the fitted catalogue


def test_the_controls_remove_order_or_direction_and_nothing_else():
    history = ItemSequences.from_rows([[0, 1, 2, 3], [5, 6]], n_items=7)
    backwards = reversed_histories(history)
    assert [backwards.row(i).tolist() for i in range(2)] == [[3, 2, 1, 0], [6, 5]]
    shuffled = shuffled_within_users(history, np.random.default_rng(0))
    assert [sorted(shuffled.row(i).tolist()) for i in range(2)] == [[0, 1, 2, 3], [5, 6]]
    chain = ItemSequences.from_rows([[0, 1, 2, 3]] * 5, n_items=4)
    assert ranked(MarkovChain().fit(chain), [[1]], n_items=4, k=1) == [[2]]
    assert ranked(MarkovChain().fit(reversed_histories(chain)), [[1]], n_items=4, k=1) == [[0]]


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: TimeDecayedPopularity(TimeDecayedPopularityConfig(half_life_days=3)),
    lambda: Replay(ReplayConfig("frequency")),
    lambda: MarkovChain(),
    lambda: Popularity(PopularityConfig("users")),
])
def test_a_saved_baseline_reloads_to_the_same_rankings(tmp_path, make):
    train = ItemSequences.from_rows([[0, 1, 2, 0, 1], [2, 1, 0]], n_items=4)
    times = np.arange(train.values.size, dtype=np.float64) * DAY
    model = make()
    model.fit(train, **({"timestamps": times} if isinstance(model, TimeDecayedPopularity) else {}),
              item_ids=np.array(["a", "b", "c", "d"]))
    model.save(tmp_path / "model.zip")
    reloaded = type(model).load(tmp_path / "model.zip")
    histories = [[0, 1], [2], [3, 3, 1]]
    assert ranked(reloaded, histories, n_items=4) == ranked(model, histories, n_items=4)
    assert reloaded.cfg == model.cfg


# ---------------------------------------------------------------------------
# the timestamp join
# ---------------------------------------------------------------------------

def _join_case(sequence_rows):
    events = pd.DataFrame({
        "user_id": ["a", "a", "a", "b", "a", "b"],
        "item_id": ["x", "y", "z", "x", "late", "gone"],   # "gone" did not survive the stage's filtering
        "timestamp": [5.0, 5.0, 7.0, 1.0, 99.0, 2.0],       # a's x and y tie; "late" is after the boundary
    })
    data = {"test_user_ids": np.array(["a", "b"]), "test_item_ids": np.array(["x", "y", "z", "late"]),
            "test_source_sequences": ItemSequences.from_rows(sequence_rows, n_items=4)}
    manifest = {"stages": {"data": {"train_target_start": 0.0, "validation_target_start": 0.0,
                                    "test_target_start": 50.0}}}
    return events, data, manifest


def test_the_join_keeps_ties_in_source_order_and_proves_the_histories():
    events, data, manifest = _join_case([[0, 1, 2], [0]])
    times = align(events, data, manifest)["test_source_timestamps"]
    assert times.tolist() == [5.0, 5.0, 7.0, 1.0]


def test_the_join_refuses_histories_it_cannot_reproduce():
    events, data, manifest = _join_case([[1, 0, 2], [0]])  # the tie in the other order
    with pytest.raises(TimestampAlignmentError, match="user a differs at position 0"):
        align(events, data, manifest)
    events, data, manifest = _join_case([[0, 1], [0]])     # one event too few
    with pytest.raises(TimestampAlignmentError, match="user a has 2 events in the split but 3"):
        align(events, data, manifest)


# ---------------------------------------------------------------------------
# the profile's timing fields, and the bug alarm
# ---------------------------------------------------------------------------

def test_timing_fields_count_ties_gaps_and_spans():
    history = ItemSequences.from_rows([[0, 1, 2], [3, 4]], n_items=5)
    times = np.array([0.0, 0.0, 2 * DAY, 100.0, 100.0])  # user a: a tie then two days; user b ends in a tie
    stats = _time_stats(history, times)
    assert stats["tie_rate"] == pytest.approx(2 / 3)
    assert stats["last_tied"] == pytest.approx(1 / 2)
    assert stats["gap_over_1_day"] == pytest.approx(1 / 3)
    assert stats["median_span_days"] == pytest.approx(1.0)  # 2 days and 0 days
    assert _time_stats(history, None)["tie_rate"] is None


def test_a_target_moving_the_wrong_way_is_flagged_as_a_bug():
    reference = {part: {"history_length": 10.0, "catalogue": 5, "density": 0.1, "popularity_gini": 0.3,
                        "repeat_rate": 0.1} for part in ("train", "test")}
    longer = {part: {**values, "history_length": 12.0} for part, values in reference.items()}
    shorter = {part: {**values, "history_length": 8.0} for part, values in reference.items()}
    assert manipulation_check(reference, longer, "history_length", 0.05)["wrong_way"] == \
        ["train history_length", "test history_length"]
    assert manipulation_check(reference, shorter, "history_length", 0.05)["wrong_way"] == []


# ---------------------------------------------------------------------------
# next-item targets
# ---------------------------------------------------------------------------

def _target_case(saved_pairs):
    from scipy.sparse import csr_matrix

    from seqrec_eval.timestamps import next_targets

    events = pd.DataFrame({
        "user_id": ["a", "a", "a", "b", "b", "a"],
        "item_id": ["x", "y", "z", "y", "x", "old"],
        "timestamp": [60.0, 60.0, 90.0, 70.0, 80.0, 10.0],  # a's x and y tie at the first target time
    })
    items = np.array(["x", "y", "z", "old"])
    rows, cols = zip(*saved_pairs)
    targets = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(2, 4))
    data = {"test_user_ids": np.array(["a", "b"]), "test_item_ids": items, "test_target_matrix": targets}
    manifest = {"stages": {"data": {"train_target_start": 0.0, "validation_target_start": 0.0,
                                    "test_target_start": 50.0}}}
    return lambda: next_targets(events, data, manifest)


def test_the_next_target_is_every_item_at_the_first_target_time():
    upcoming = _target_case([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])()["test"]
    assert [upcoming[r].indices.tolist() for r in range(2)] == [[0, 1], [1]]  # a: the tie x, y; b: y came first


def test_next_targets_refuse_a_window_that_does_not_match_the_saved_targets():
    from seqrec_eval.timestamps import TimestampAlignmentError

    with pytest.raises(TimestampAlignmentError, match="in the prepared window but not the split"):
        _target_case([(0, 0), (0, 1), (1, 0), (1, 1)])()  # a's z is missing from the saved targets


# ---------------------------------------------------------------------------
# ties: one fixed order, so every way of asking for a list gives the same list
# ---------------------------------------------------------------------------
# On ML-20M (2026-09-30) Markov scored 0.0230 on the full data and 0.0226 at every level of an inference sweep,
# although it reads only the last item, which truncation keeps. The ablation conditions exclude seen items against
# the full history by asking the model for a longer list, and a top-k that breaks ties as it happens to picks
# different items for a different length.

TIE_ITEMS = 400


def _random_histories(seed: int, n_rows: int, shortest: int, longest: int) -> ItemSequences:
    rng = np.random.default_rng(seed)
    return ItemSequences.from_rows([list(rng.integers(0, TIE_ITEMS, rng.integers(shortest, longest)))
                                    for _ in range(n_rows)], n_items=TIE_ITEMS)


def _tied_baselines():
    train = _random_histories(0, 300, 3, 15)  # few events per item: equal counts and equal transitions everywhere
    hours = np.arange(train.values.size, dtype=np.float64) * 3_600.0
    return {
        "markov": MarkovChain().fit(train),
        "popularity": Popularity(PopularityConfig(count="users")).fit(train),
        "replay": Replay(ReplayConfig(order="frequency")).fit(train),
        "time_popularity": TimeDecayedPopularity(TimeDecayedPopularityConfig(half_life_days=1.0)).fit(
            train, timestamps=hours),
    }


@pytest.mark.parametrize("name", ["markov", "popularity", "replay", "time_popularity"])
def test_excluding_seen_items_in_the_model_or_after_it_gives_the_same_list(name):
    model = _tied_baselines()[name]
    test = _random_histories(1, 120, 5, 60)
    own = model.predict_on_batch(test, k=10, exclude_seen=True)
    policy = ExcludeSeenPolicy(model, True, seen=count_matrix(test, 1.0), limit=TIE_ITEMS, source=test)
    after = policy.predict_on_batch(test, k=10)
    assert np.array_equal(own.cols.numpy(), after.cols.numpy())
    assert np.array_equal(own.vals.numpy(), after.vals.numpy())
    # and a shorter list is the start of a longer one
    longer = model.predict_on_batch(test, k=40, exclude_seen=False).cols.numpy()
    assert np.array_equal(model.predict_on_batch(test, k=10, exclude_seen=False).cols.numpy(), longer[:, :10])


def test_ties_go_to_the_more_popular_item_then_the_lower_index():
    model = _tied_baselines()["markov"]
    ranked_ = model.predict_on_batch(_random_histories(2, 50, 1, 4), k=30, exclude_seen=False)
    cols, vals = ranked_.cols.numpy(), ranked_.vals.numpy()
    popularity = model.popularity_
    for row_cols, row_vals in zip(cols, vals):
        keys = [(-v, -popularity[c], c) for c, v in zip(row_cols, row_vals)]
        assert keys == sorted(keys)
    assert any(len(set(row)) < len(row) for row in vals)  # the data does have ties to break


@pytest.mark.parametrize("n_columns", [25, 300])  # 300: ties longer than the fast path's room, so some rows overflow
def test_top_k_is_the_first_k_of_one_total_order(n_columns):
    rng = np.random.default_rng(3)
    scores = rng.integers(0, 4, size=(60, n_columns)).astype(np.float64)  # four values: ties everywhere
    scores[rng.random(scores.shape) < 0.2] = -np.inf                       # seen items, masked
    scores[:5, :n_columns - 3] = -np.inf                                   # rows with fewer finite items than k
    order = rng.permutation(n_columns)
    position = np.argsort(order)
    for k in (1, 5, 25):
        for rank in (_top_k, _top_k_full):
            columns, values = rank(scores, order, k)
            for row in range(scores.shape[0]):
                expected = sorted(range(n_columns), key=lambda c: (-scores[row, c], position[c]))[:k]
                assert columns[row].tolist() == expected, (rank.__name__, k, row)
                assert values[row].tolist() == scores[row, expected].tolist()


def test_the_full_ranking_is_only_for_rows_whose_tie_outruns_the_fast_path(monkeypatch):
    # it is 100-200 times slower, so it must stay the exception: rows with short ties never reach it
    import seqrec_eval.baselines as baselines

    calls = []
    full = baselines._top_k_full
    monkeypatch.setattr(baselines, "_top_k_full", lambda scores, order, k: calls.append(len(scores)) or full(scores, order, k))
    rng = np.random.default_rng(4)
    scores = rng.integers(0, 50, size=(40, 500)).astype(np.float64)  # ties of about 10 at any value: short
    scores[:3, :] = 1.0                                               # three rows that are one long tie
    baselines._top_k(scores, rng.permutation(500), 10)
    assert calls == [3]


@pytest.mark.parametrize("name", ["markov", "popularity", "replay", "time_popularity"])
@pytest.mark.parametrize("exclude_seen", [False, True])
def test_ranking_without_a_dense_array_gives_the_dense_lists(name, exclude_seen):
    # the sparse path (evidence, then one popularity order) replaces rows x items scores (review B1)
    from seqrec_eval.baselines import _seen_matrix

    model = _tied_baselines()[name]
    test = _random_histories(5, 150, 0, 80)  # empty histories included
    seen = _seen_matrix(model._prepare_source(test), TIE_ITEMS)
    candidates = np.arange(TIE_ITEMS)
    for k in (1, 10, 60):
        sparse = model._rank_sparse(test, seen if exclude_seen else None, k)
        dense = model._rank_dense(test, seen, exclude_seen, candidates, k)
        assert np.array_equal(sparse[0], dense[0]), (name, exclude_seen, k)
        assert np.array_equal(sparse[1], dense[1]), (name, exclude_seen, k)

