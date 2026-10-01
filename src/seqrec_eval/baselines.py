"""The non-learned baselines of §5.2: popularity, time-decayed popularity, replay, Markov.

They are not models under study. The analysis step (:mod:`seqrec_eval.analysis`)
tunes each on validation, scores it on test through the same adapter and
evaluator as the models, and the strongest sets the floor every model has to
beat. First-order Markov is also the instrument of the sequence-signal
analysis, fitted on shuffled and on reversed training histories as controls.

They are built on the library's own sequential base class, so they rank and
mask seen items exactly as the library's models do.

All of them read chronological histories. Histories
arrive through :class:`~compresso_recsys.models.WarmCatalogAdapter` whole, so
an index at or beyond the fitted catalogue is an item first seen after
training: it keeps its position but is otherwise ignored.

Every baseline ranks in two tiers: first the items it has evidence for,
ordered by its own score, then everything else by training popularity, so a
list is always full and never arbitrary. The tiers are strict -- a score of
any size outranks popularity -- rather than a small popularity term added to
the score, which a tiny decayed weight or a rare transition could fall below. A replay model under ``exclude_seen = true`` therefore
reduces to popularity, since everything it would replay is excluded.

Equal scores are common here -- the successors a Markov chain saw once each,
items with the same count -- so ties are broken in one fixed order: the more
popular item in training first, then the lower index (:func:`_top_k`). A
top-k that breaks ties as it happens to (torch's) can pick different items for
a different ``k``; the evaluation's seen-item filter asks for a longer list
when it excludes against the full history (ablation conditions), so the same
histories scored differently on the full data than at every level (found on
ML-20M, 2026-09-30: Markov 0.0230 against 0.0226).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from compresso import SRPTensor
from compresso_recsys import ItemSequences
from compresso_recsys.models._ranking import (
    mask_seen_numpy,
    validate_candidate_topk,
)
from compresso_recsys.models.base import BaseSequentialRecommender
from scipy.sparse import csr_matrix

DAY = 86_400.0


def _event_rows(sequences: ItemSequences) -> np.ndarray:
    return np.repeat(np.arange(sequences.n_rows, dtype=np.int64), sequences.row_lengths)


def _known(sequences: ItemSequences, n_items: int) -> np.ndarray:
    """Which events name an item the model was fitted on."""
    return sequences.values < n_items


def _seen_matrix(sequences: ItemSequences, n_items: int) -> csr_matrix:
    """The histories as a binary CSR over the fitted catalogue, for masking seen items."""
    known = _known(sequences, n_items)
    rows, cols = _event_rows(sequences)[known], sequences.values[known]
    matrix = csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(sequences.n_rows, n_items))
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _tiebreak(popularity: np.ndarray) -> np.ndarray:
    """Popularity squeezed into [0, 1): the lower tier, below every score of 1 or more."""
    top = float(popularity.max()) if popularity.size else 0.0
    return popularity / (top + 1.0)


#: How many candidates past ``k`` the fast path of :func:`_top_k` takes from torch, to hold every item tied with
#: the k-th; a row whose tie runs further is ranked in full.
_TIE_ROOM = 64


def _top_k(scores: np.ndarray, order: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Each row's ``k`` best columns of ``scores`` and their values, in one total order.

    A higher score comes first; among equal scores, the column that comes first in ``order`` (a permutation of
    the columns). So the first ``k`` of a longer list are always this list, whatever ``k`` is asked for.

    torch's top ``k + _TIE_ROOM`` holds every column scoring at least the k-th value unless the tie at that value
    runs past it, so ordering those candidates exactly gives the answer; the rare row whose tie does run past is
    ranked in full (:func:`_top_k_full`). The same lists either way, at torch's speed: the full ranking alone is
    100-200 times slower (2026-09-30, a 1,024 x 24,093 batch: 1.7 s against 8 ms).
    """
    n_rows, n = scores.shape
    k = min(int(k), n)
    if n_rows == 0 or k == 0:
        return np.zeros((n_rows, k), dtype=np.int64), np.zeros((n_rows, k), dtype=scores.dtype)
    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n)
    wide = min(n, k + _TIE_ROOM)
    values, columns = torch.from_numpy(scores).topk(wide, dim=1)  # sorted, best first
    values, columns = values.numpy(), columns.numpy()
    best = np.lexsort((position[columns], -values), axis=1)[:, :k]  # by value, then by place in ``order``
    out_columns = np.take_along_axis(columns, best, axis=1)
    out_values = np.take_along_axis(values, best, axis=1)
    # the tie at the k-th value may reach the last candidate, and so perhaps beyond it
    overflow = np.flatnonzero((wide < n) & (values[:, -1] >= values[:, k - 1]))
    if overflow.size:
        out_columns[overflow], out_values[overflow] = _top_k_full(scores[overflow], order, k)
    return out_columns, out_values


def _top_k_full(scores: np.ndarray, order: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_top_k` over every column of every row: exact however long the ties, and slow."""
    ranked = scores[:, order]  # ties now resolve by position
    n_rows, n = ranked.shape
    k = min(int(k), n)
    threshold = -np.partition(-ranked, k - 1, axis=1)[:, k - 1:k]  # each row's k-th best value
    above = ranked > threshold
    level = ranked == threshold
    room = k - above.sum(axis=1, keepdims=True)
    # all that beat the k-th value, and of those equal to it the first that fit, in ``order``
    take = above | (level & (np.cumsum(level, axis=1) <= room))
    positions = np.nonzero(take)[1].reshape(n_rows, k)  # row by row, ascending position
    values = np.take_along_axis(ranked, positions, axis=1)
    best_first = np.argsort(-values, axis=1, kind="stable")  # stable: equal values keep their position order
    positions = np.take_along_axis(positions, best_first, axis=1)
    return order[positions], np.take_along_axis(values, best_first, axis=1)


def _first_free(order: np.ndarray, values: np.ndarray, need: np.ndarray, blocked_rows: np.ndarray,
                blocked_items: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row ``r``, the first ``need[r]`` items of ``order`` that are not blocked for ``r``, with their values.

    The blocked items are (``blocked_rows``, ``blocked_items``) pairs. Those wanted lie within the first
    ``need[r]`` plus the row's number of blocked items, so only that much of ``order`` is ever looked at.
    Returns flat (rows, items, values), row by row, each row in ``order``.
    """
    n_rows, n = need.size, order.size
    if n_rows == 0 or not need.any():
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, values.dtype)
    width = int(min(n, (need + np.bincount(blocked_rows, minlength=n_rows)).max()))
    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n)
    at = position[blocked_items]
    inside = at < width
    blocked = np.zeros((n_rows, width), dtype=bool)
    blocked[blocked_rows[inside], at[inside]] = True
    free = ~blocked
    rows, places = np.nonzero(free & (np.cumsum(free, axis=1) <= need[:, None]))
    items = order[places]
    return rows, items, values[items]


def _two_tiers(scores: np.ndarray, popularity: np.ndarray) -> np.ndarray:
    """Positive scores rescaled into (1, 2] per row, above popularity in [0, 1) for the rest."""
    top = scores.max(axis=-1, keepdims=True)
    upper = 1.0 + scores / np.where(top > 0, top, 1.0)
    return np.where(scores > 0, upper, _tiebreak(popularity))


class _SequenceBaseline(BaseSequentialRecommender):
    """What the three baselines share: a catalogue, popularity, and the ranking step."""

    checkpoint_type = "seqrec_eval.baseline"

    def __init__(self, config) -> None:
        self.cfg = config
        self.n_items_: int | None = None
        self.popularity_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.n_items_ is not None

    @property
    def n_items(self) -> int | None:
        return self.n_items_

    def _start_fit(self, sequences: ItemSequences, item_ids) -> None:
        if not isinstance(sequences, ItemSequences):
            raise TypeError(f"{type(self).__name__} fits on ItemSequences, got {type(sequences).__name__}")
        n_items = int(sequences.n_items)
        if n_items < 1:
            raise ValueError("the training catalogue is empty")
        self._publish_item_vocabulary(self._prepare_item_vocabulary(item_ids, n_items=n_items))
        self.n_items_ = n_items
        self.popularity_ = np.bincount(sequences.values, minlength=n_items).astype(np.float64)

    def _scores(self, source: ItemSequences) -> np.ndarray:
        """Scores of every fitted item for every history, before masking and ranking (the dense path)."""
        raise NotImplementedError

    def _evidence(self, source: ItemSequences, limit: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The upper tier of each row as (rows, items, values), exactly the values :meth:`_scores` gives them;
        every other item is in the popularity tier. ``None`` where the scores are the same for every row
        (:meth:`_global_scores`).

        A baseline may return only each row's best ``limit[row]`` entries in the ranking's order (score, then
        popularity, then index): ``limit`` is k plus the row's seen items, which is all a list of k can use.
        """
        return None

    def _global_scores(self) -> np.ndarray:
        """Every item's score, for a baseline that scores every history alike."""
        raise NotImplementedError

    def predict_on_batch(self, source: ItemSequences, *, k: int, exclude_seen: bool = True,
                         candidate_ids: Sequence[Hashable] | np.ndarray | None = None) -> SRPTensor:
        source = self._prepare_source(source)
        candidate_rows = self._candidate_rows(candidate_ids)
        seen = _seen_matrix(source, self.n_items_)
        validate_candidate_topk(seen, candidate_rows, k=k, exclude_seen=exclude_seen)
        if candidate_ids is None:
            columns, values = self._rank_sparse(source, seen if exclude_seen else None, k)
        else:
            columns, values = self._rank_dense(source, seen, exclude_seen, candidate_rows, k)
        return SRPTensor(cols=torch.from_numpy(np.ascontiguousarray(columns)),
                         vals=torch.from_numpy(np.ascontiguousarray(values)), shape=(source.n_rows, self.n_items_))

    def _rank_dense(self, source: ItemSequences, seen: csr_matrix, exclude_seen: bool, candidate_rows: np.ndarray,
                    k: int) -> tuple[np.ndarray, np.ndarray]:
        """Every candidate's score for every row, then the top k: rows x candidates of memory."""
        scores = np.ascontiguousarray(self._scores(source)[:, candidate_rows])
        if exclude_seen:
            mask_seen_numpy(scores, seen, candidate_rows)
        # ties: the more popular training item first, then the lower index (see the module docstring)
        order = np.lexsort((candidate_rows, -self.popularity_[candidate_rows]))
        columns, values = _top_k(scores, order, k)
        return candidate_rows[columns], values

    def _rank_sparse(self, source: ItemSequences, seen: csr_matrix | None, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The same lists as :meth:`_rank_dense` over every item, without a rows x items array.

        Each row's list is its upper tier, best first, then the popularity tier, which is one order for everyone
        (more popular first, then the lower index), less the row's own upper-tier items and, when excluding, its
        seen ones. Both orders break ties as the dense path does, so the lists are the same (tested); dense, an
        Amazon-sized catalogue took 12 GiB and 7 s per batch of 1,024 (final review B1).
        """
        n_rows, n = source.n_rows, self.n_items_
        k = min(int(k), n)
        popularity = self.popularity_
        seen_rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(seen.indptr)) if seen is not None \
            else np.zeros(0, np.int64)
        seen_items = seen.indices.astype(np.int64) if seen is not None else np.zeros(0, np.int64)
        limit = k + (np.diff(seen.indptr) if seen is not None else np.zeros(n_rows, np.int64))
        evidence = self._evidence(source, limit)
        if evidence is None:  # one order for everyone, less each row's seen items
            scores = self._global_scores()
            order = np.lexsort((np.arange(n), -popularity, -scores))
            rows, items, values = _first_free(order, scores, np.full(n_rows, k), seen_rows, seen_items)
            return items.reshape(n_rows, k), values.reshape(n_rows, k)
        rows, items, values = evidence
        if seen is not None and rows.size:  # a seen item cannot be recommended, however strong its evidence
            width = np.int64(n)
            kept = ~np.isin(rows * width + items, seen_rows * width + seen_items)
            upper_rows, upper_items, upper_values = rows[kept], items[kept], values[kept]
        else:
            upper_rows, upper_items, upper_values = rows, items, values
        by_row = np.lexsort((upper_items, -popularity[upper_items], -upper_values, upper_rows))
        upper_rows, upper_items, upper_values = upper_rows[by_row], upper_items[by_row], upper_values[by_row]
        starts = np.searchsorted(upper_rows, np.arange(n_rows))
        rank = np.arange(upper_rows.size) - starts[upper_rows]
        top = rank < k
        upper_rows, upper_items, upper_values = upper_rows[top], upper_items[top], upper_values[top]
        need = k - np.bincount(upper_rows, minlength=n_rows)
        lower = _tiebreak(popularity)
        fill_rows, fill_items, fill_values = _first_free(
            np.lexsort((np.arange(n), -popularity)), lower, need,
            np.concatenate([rows, seen_rows]), np.concatenate([items, seen_items]))
        # Per row: its upper tier, then its fill in popularity order. The fill skips the row's evidence items:
        # a row given only its best `limit` entries needs no fill unless those were all it had.
        all_rows = np.concatenate([upper_rows, fill_rows])
        part = np.concatenate([np.zeros(upper_rows.size, np.int8), np.ones(fill_rows.size, np.int8)])
        within = np.concatenate([np.arange(upper_rows.size), np.arange(fill_rows.size)])
        order = np.lexsort((within, part, all_rows))
        columns = np.concatenate([upper_items, fill_items])[order].reshape(n_rows, k)
        values = np.concatenate([upper_values, fill_values])[order].reshape(n_rows, k)
        return columns, values

    @classmethod
    def _from_checkpoint_config(cls, config: dict, reader, *, device) -> _SequenceBaseline:
        del reader, device
        return cls(cls.Config(**config))

    def _save_checkpoint_state(self, writer) -> None:
        writer.write_json("state/baseline.json", {"n_items": self.n_items_})
        writer.write_numpy("state/popularity.npy", self.popularity_)

    def _load_checkpoint_state(self, reader) -> None:
        self.n_items_ = int(reader.read_json("state/baseline.json")["n_items"])
        self.popularity_ = reader.read_numpy("state/popularity.npy")


# ---------------------------------------------------------------------------
# popularity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PopularityConfig:
    #: "events": every training event counts. "users": each user counts once per item.
    count: str = "events"

    def __post_init__(self) -> None:
        if self.count not in ("events", "users"):
            raise ValueError('count must be "events" or "users"')


class Popularity(_SequenceBaseline):
    """The most popular training items, for everyone."""

    checkpoint_type = "seqrec_eval.popularity"
    Config = PopularityConfig

    def __init__(self, config: PopularityConfig | None = None) -> None:
        super().__init__(config or PopularityConfig())
        self.counts_: np.ndarray | None = None

    def fit(self, sequences: ItemSequences, *, item_ids=None) -> Popularity:
        self._start_fit(sequences, item_ids)
        if self.cfg.count == "events":
            self.counts_ = self.popularity_
        else:
            pairs = np.unique(_event_rows(sequences) * np.int64(self.n_items_) + sequences.values)
            self.counts_ = np.bincount(pairs % self.n_items_, minlength=self.n_items_).astype(np.float64)
        return self

    def _scores(self, source: ItemSequences) -> np.ndarray:
        return np.broadcast_to(self._global_scores(), (source.n_rows, self.n_items_))

    def _global_scores(self) -> np.ndarray:
        return _two_tiers(self.counts_, self.popularity_)

    def _save_checkpoint_state(self, writer) -> None:
        super()._save_checkpoint_state(writer)
        writer.write_numpy("state/counts.npy", self.counts_)

    def _load_checkpoint_state(self, reader) -> None:
        super()._load_checkpoint_state(reader)
        self.counts_ = reader.read_numpy("state/counts.npy")


# ---------------------------------------------------------------------------
# time-decayed popularity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeDecayedPopularityConfig:
    #: Days after which an event counts half as much as one at the end of training.
    half_life_days: float = 7.0

    def __post_init__(self) -> None:
        if not self.half_life_days > 0:
            raise ValueError("half_life_days must be positive")


class TimeDecayedPopularity(_SequenceBaseline):
    """Popularity in which each event counts ``0.5 ** (age / half_life)``.

    Age is measured from the last training event. Measuring it from any later
    moment, such as the test window's start, multiplies every weight by the
    same factor, so the ranking is the same; the model is non-personalised and
    needs the events' times, which the split does not store but the suite
    recovers (:mod:`seqrec_eval.timestamps`).
    """

    checkpoint_type = "seqrec_eval.time_decayed_popularity"
    Config = TimeDecayedPopularityConfig

    def __init__(self, config: TimeDecayedPopularityConfig | None = None) -> None:
        super().__init__(config or TimeDecayedPopularityConfig())
        self.weights_: np.ndarray | None = None

    def fit(self, sequences: ItemSequences, *, item_ids=None, timestamps: np.ndarray | None = None
            ) -> TimeDecayedPopularity:
        if timestamps is None:
            raise ValueError("time-decayed popularity needs the training events' timestamps; "
                             "prepare the split without --no-timestamps")
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if timestamps.shape != sequences.values.shape:
            raise ValueError(f"{timestamps.size} timestamps for {sequences.values.size} training events")
        self._start_fit(sequences, item_ids)
        age_days = (timestamps.max() - timestamps) / DAY if timestamps.size else timestamps
        weights = np.power(0.5, age_days / float(self.cfg.half_life_days))
        self.weights_ = np.bincount(sequences.values, weights=weights, minlength=self.n_items_)
        return self

    def _scores(self, source: ItemSequences) -> np.ndarray:
        return np.broadcast_to(self._global_scores(), (source.n_rows, self.n_items_))

    def _global_scores(self) -> np.ndarray:
        # an item whose every event decayed below float precision falls to the popularity tier
        return _two_tiers(self.weights_, self.popularity_)

    def _save_checkpoint_state(self, writer) -> None:
        super()._save_checkpoint_state(writer)
        writer.write_numpy("state/weights.npy", self.weights_)

    def _load_checkpoint_state(self, reader) -> None:
        super()._load_checkpoint_state(reader)
        self.weights_ = reader.read_numpy("state/weights.npy")


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayConfig:
    #: "recency": the most recently seen item first. "frequency": the most
    #: often seen first, recency breaking ties.
    order: str = "recency"

    def __post_init__(self) -> None:
        if self.order not in ("recency", "frequency"):
            raise ValueError('order must be "recency" or "frequency"')


class Replay(_SequenceBaseline):
    """Recommend the user's own history back, then popular items after it."""

    checkpoint_type = "seqrec_eval.replay"
    Config = ReplayConfig

    def __init__(self, config: ReplayConfig | None = None) -> None:
        super().__init__(config or ReplayConfig())

    def fit(self, sequences: ItemSequences, *, item_ids=None) -> Replay:
        self._start_fit(sequences, item_ids)
        return self

    def _scores(self, source: ItemSequences) -> np.ndarray:
        n = self.n_items_
        scores = np.broadcast_to(_tiebreak(self.popularity_), (source.n_rows, n)).copy()
        rows, items, values = self._evidence(source)
        scores[rows, items] = values
        return scores

    def _evidence(self, source: ItemSequences, limit: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n_items_
        known = _known(source, n)
        rows, items = _event_rows(source)[known], source.values[known]
        # position from the end, 1 for the last event: smaller is more recent
        from_end = (np.repeat(source.indptr[1:], source.row_lengths) - np.arange(source.values.size))[known]
        pairs = rows * np.int64(n) + items
        order = np.lexsort((from_end, pairs))  # per pair, its most recent occurrence first
        pairs, from_end = pairs[order], from_end[order]
        first = np.concatenate(([True], pairs[1:] != pairs[:-1])) if pairs.size else np.zeros(0, dtype=bool)
        unique, latest = pairs[first], from_end[first]
        recency = 1.0 / latest  # in (0, 1], larger for more recent
        if self.cfg.order == "recency":
            replayed = recency
        else:
            counts = np.diff(np.append(np.flatnonzero(first), pairs.size))
            replayed = counts + recency / 2.0  # recency stays below 1, so it only separates equal counts
        # the replay tier sits above popularity: 1 + a positive score
        return unique // n, unique % n, 1.0 + replayed


# ---------------------------------------------------------------------------
# first-order Markov
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarkovConfig:
    """First-order Markov has no settings; the class exists so every baseline is built alike."""


class MarkovChain(_SequenceBaseline):
    """Next item from the last one: ``P(next | last)``, counted over adjacent training events.

    Each row of the transition counts is normalised to a probability, and a
    history is scored by the row of its last item. Self-transitions (an item
    following itself) are counted like any other, so on repeat-heavy data the
    model learns repeats; the analysis reports how much of the data they are.
    A history whose last item has no recorded successor, or is new, is ranked
    by popularity.
    """

    checkpoint_type = "seqrec_eval.markov"
    Config = MarkovConfig

    def __init__(self, config: MarkovConfig | None = None) -> None:
        super().__init__(config or MarkovConfig())
        self.transitions_: csr_matrix | None = None

    def fit(self, sequences: ItemSequences, *, item_ids=None) -> MarkovChain:
        self._start_fit(sequences, item_ids)
        n = self.n_items_
        rows, values = _event_rows(sequences), sequences.values
        same_row = rows[:-1] == rows[1:] if values.size > 1 else np.zeros(0, dtype=bool)
        source, target = values[:-1][same_row], values[1:][same_row]
        counts = csr_matrix((np.ones(source.size), (source, target)), shape=(n, n))
        counts.sum_duplicates()
        totals = np.asarray(counts.sum(axis=1)).ravel()
        counts.data /= np.repeat(np.where(totals > 0, totals, 1.0), np.diff(counts.indptr))
        self.transitions_ = counts.tocsr()
        self._ranked = None
        return self

    def _last(self, source: ItemSequences) -> tuple[np.ndarray, np.ndarray]:
        """The rows whose last item the chain knows, and that item."""
        last = np.full(source.n_rows, -1, dtype=np.int64)
        has = source.row_lengths > 0
        last[has] = source.values[source.indptr[1:][has] - 1]
        usable = np.flatnonzero((last >= 0) & (last < self.n_items_))
        return usable, last[usable]

    def _scores(self, source: ItemSequences) -> np.ndarray:
        scores = np.zeros((source.n_rows, self.n_items_))
        usable, last = self._last(source)
        if usable.size:
            scores[usable] = self.transitions_[last].toarray()
        # items the chain has no transition to fall to the popularity tier
        return _two_tiers(scores, self.popularity_)

    def _ranked_successors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every item's successors in the ranking's order (score, then popularity, then index), with their
        upper-tier values, as CSR (indptr, items, values); sorted once, on first use, since a popular item can
        have hundreds of thousands of successors and every history ending in it would sort them again."""
        if getattr(self, "_ranked", None) is None:
            transitions = self.transitions_.tocsr()
            counts = np.diff(transitions.indptr)
            owner = np.repeat(np.arange(transitions.shape[0], dtype=np.int64), counts)
            probabilities = transitions.data.astype(np.float64)
            order = np.lexsort((transitions.indices, -self.popularity_[transitions.indices], -probabilities, owner))
            top = np.zeros(transitions.shape[0])
            nonempty = counts > 0
            if nonempty.any():
                top[nonempty] = np.maximum.reduceat(probabilities, transitions.indptr[:-1][nonempty])
            # as _two_tiers: 1 + the score over the row's best, so each item's best successor sits at 2
            values = 1.0 + probabilities[order] / np.repeat(top, counts)
            self._ranked = (transitions.indptr.astype(np.int64), transitions.indices[order].astype(np.int64), values)
        return self._ranked

    def _evidence(self, source: ItemSequences, limit: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        usable, last = self._last(source)
        indptr, items, values = self._ranked_successors()
        starts = indptr[last]
        lengths = np.minimum(indptr[last + 1] - starts, limit[usable])
        rows = np.repeat(usable, lengths)
        within = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        taken = np.repeat(starts, lengths) + within
        return rows, items[taken], values[taken]

    def _save_checkpoint_state(self, writer) -> None:
        super()._save_checkpoint_state(writer)
        for part in ("data", "indices", "indptr"):
            writer.write_numpy(f"state/transitions_{part}.npy", getattr(self.transitions_, part))

    def _load_checkpoint_state(self, reader) -> None:
        super()._load_checkpoint_state(reader)
        n = self.n_items_
        self.transitions_ = csr_matrix(
            tuple(reader.read_numpy(f"state/transitions_{part}.npy") for part in ("data", "indices", "indptr")),
            shape=(n, n))
        self._ranked = None
