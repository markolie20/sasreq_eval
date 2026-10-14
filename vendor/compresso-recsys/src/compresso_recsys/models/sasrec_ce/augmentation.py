# Modified for seqrec_eval: differs from upstream compresso-recsys; see vendor/compresso-recsys/VENDORED.md
"""Training-time augmentation for :class:`SASRecCETrainer`.

What an augmentation may touch decides where it runs, and there are two places,
one on each side of the next-item shift::

    observed history (catalog indices)
      -> keep the window, plus the one interaction the shift spends
      -> sequence edits      rewrite the history: inputs *and* targets move
      -> keep the window again
      -> tokenize, shift     input i predicts target i
      -> input corruption    rewrite what the model reads, never a target
      -> left-pad to the window

**Sequence edits** change the history itself, so an edit at position ``i`` moves
both the input read at step ``i`` and the target predicted at step ``i - 1``.
:func:`insert_cooccurring` is one. An edit takes and returns an
:class:`AugmentedHistory`, whose ``observed`` flags say which items the user
actually interacted with, so the trainer can decide whether an introduced item
is something to predict or only something to read. An edit may change the
length -- the window is applied again after it. The trainer's hook is
:meth:`SASRecCETrainer._augment_history`.

**Input corruption** changes only what the model reads, after the shift, so no
target is lost: :func:`replace_inputs` with one special token, ``mask`` or
``unk``, and :func:`replace_with_cooccurring` with a co-occurring item per
position. Replacement belongs here rather than among the edits because it does
not need to move a target: the input changes and the model is still asked for
the item that really came next.

Both co-occurrence augmentations draw from a :class:`CooccurrenceSampler`, which
the trainer builds once per ``fit`` from the training sequences. Nothing here
decides what the user's history *is* for the softmax's history exclusion: that
is read from the observed sequence, never from an augmented one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from compresso_recsys.sequences import ItemSequences

__all__ = [
    "AugmentedHistory",
    "CooccurrenceSampler",
    "insert_cooccurring",
    "replace_inputs",
    "replace_with_cooccurring",
]


@dataclass(frozen=True)
class AugmentedHistory:
    """A history of catalog indices, and which of them really happened.

    ``items`` is chronological, like a row of
    :class:`~compresso_recsys.ItemSequences`. ``observed`` is aligned with it
    and false for an item a sequence edit introduced.
    """

    items: np.ndarray
    observed: np.ndarray

    def __post_init__(self) -> None:
        if self.items.ndim != 1 or self.items.shape != self.observed.shape:
            raise ValueError(
                "items and observed must be aligned one-dimensional arrays, got "
                f"shapes {self.items.shape} and {self.observed.shape}"
            )
        if self.observed.dtype != np.bool_:
            raise TypeError(
                f"observed must be a bool array, got {self.observed.dtype}"
            )

    @classmethod
    def observed_history(cls, items: np.ndarray) -> AugmentedHistory:
        """A history as recorded: every item is one the user interacted with."""
        items = np.asarray(items, dtype=np.int64)
        return cls(items, np.ones(items.shape, dtype=bool))

    def tail(self, length: int) -> AugmentedHistory:
        """The most recent ``length`` items, the direction every window takes."""
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        return AugmentedHistory(self.items[-length:], self.observed[-length:])


@dataclass(frozen=True)
class CooccurrenceSampler:
    """For an item, draws another that appeared near it in training histories.

    Stored like a CSR matrix with one row per catalog item:
    ``neighbors[indptr[a]:indptr[a + 1]]`` are ``a``'s partners, heaviest
    first, and ``probabilities`` over the same slice sum to one. An item with
    an empty slice co-occurred with nothing and cannot be drawn for.

    Build one with :meth:`from_sequences`; the constructor takes the arrays as
    they are, for a table built some other way.
    """

    indptr: np.ndarray
    neighbors: np.ndarray
    probabilities: np.ndarray
    # Each row's cumulative distribution, shifted up by the row's index, so a
    # single searchsorted over the whole table draws for many rows at once.
    _keys: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.indptr.ndim != 1 or self.indptr.size < 1 or self.indptr[0] != 0:
            raise ValueError("indptr must be one-dimensional and start at zero")
        if np.any(np.diff(self.indptr) < 0):
            raise ValueError("indptr must be non-decreasing")
        if not (
            self.neighbors.shape == self.probabilities.shape
            == (int(self.indptr[-1]),)
        ):
            raise ValueError(
                "neighbors and probabilities must hold indptr[-1] entries each"
            )
        if self.probabilities.size and float(self.probabilities.min()) <= 0.0:
            raise ValueError("every stored partner needs a positive probability")

        counts = np.diff(self.indptr)
        rows = np.repeat(np.arange(counts.size, dtype=np.float64), counts)
        running = np.cumsum(self.probabilities, dtype=np.float64)
        before = np.concatenate(([0.0], running))[self.indptr[:-1]]
        within = running - np.repeat(before, counts)
        # Rounding can leave a row's total a hair off one. Pinning each row's
        # last entry to exactly one keeps every row's keys inside (r, r + 1],
        # so a draw of r + u, u in [0, 1), can only land in row r.
        if within.size:
            last = self.indptr[1:][counts > 0] - 1
            within = np.minimum(within / np.repeat(within[last], counts[counts > 0]), 1.0)
            within[last] = 1.0
        object.__setattr__(self, "_keys", rows + within)

    @classmethod
    def from_sequences(
        cls,
        sequences: ItemSequences,
        *,
        window: int,
        top_k: int,
        popularity_exponent: float,
    ) -> CooccurrenceSampler:
        """Count co-occurrences in ``sequences`` and keep each item's heaviest.

        Two items co-occur once for every time they sit within ``window``
        steps of each other in the same history, in either order; a repeated
        item is never its own partner. Each count is then weighted by the
        partner's popularity -- its number of interactions -- raised to
        ``popularity_exponent``, the ``top_k`` heaviest partners of each item
        are kept, and their weights normalised into probabilities. See
        :class:`~.config.SASRecCEConfig` for what the exponent's sign means.

        Counted one offset at a time, so the transient pairs never exceed one
        per interaction rather than ``window`` per interaction.
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        n_items = int(sequences.n_items)
        values = np.asarray(sequences.values, dtype=np.int64)
        users = np.repeat(np.arange(sequences.n_rows), sequences.row_lengths)

        counts = sparse.csr_matrix((n_items, n_items), dtype=np.float64)
        for offset in range(1, min(window, max(values.size - 1, 0)) + 1):
            first, second = values[:-offset], values[offset:]
            paired = (users[:-offset] == users[offset:]) & (first != second)
            pairs = sparse.csr_matrix(
                (
                    np.ones(int(paired.sum()), dtype=np.float64),
                    (first[paired], second[paired]),
                ),
                shape=(n_items, n_items),
            )
            counts = counts + pairs + pairs.T

        popularity = np.bincount(values, minlength=n_items).astype(np.float64)
        # Only items that occur can be partners; the substitution keeps a
        # negative exponent from raising 0 to it for the ones that do not.
        scale = np.where(popularity > 0, popularity, 1.0) ** popularity_exponent
        weights = (counts @ sparse.diags(scale)).tocsr()
        weights.sum_duplicates()
        weights.eliminate_zeros()

        rows = np.repeat(np.arange(n_items), np.diff(weights.indptr))
        # Row first, then heaviest first; lexsort is stable, so tied weights
        # keep ascending item order and the table is deterministic.
        order = np.lexsort((-weights.data, rows))
        rank = np.arange(order.size) - weights.indptr[rows[order]]
        kept = order[rank < top_k]
        kept_rows = rows[kept]
        kept_weights = weights.data[kept]
        totals = np.bincount(kept_rows, weights=kept_weights, minlength=n_items)
        return cls(
            indptr=np.concatenate(
                ([0], np.cumsum(np.bincount(kept_rows, minlength=n_items)))
            ).astype(np.int64),
            neighbors=weights.indices[kept].astype(np.int64),
            probabilities=kept_weights / totals[kept_rows],
        )

    @property
    def n_items(self) -> int:
        """Catalog size the table covers."""
        return int(self.indptr.size - 1)

    def has_neighbors(self, items: np.ndarray) -> np.ndarray:
        """Whether each item co-occurred with anything, shaped like ``items``."""
        return np.diff(self.indptr)[np.asarray(items, dtype=np.int64)] > 0

    def neighbors_of(self, item: int) -> tuple[np.ndarray, np.ndarray]:
        """One item's partners and their probabilities, heaviest first."""
        start, stop = int(self.indptr[item]), int(self.indptr[item + 1])
        return self.neighbors[start:stop], self.probabilities[start:stop]

    def sample(self, items: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One partner per item, drawn independently, shaped like ``items``.

        Every item must have a partner; filter with :meth:`has_neighbors`
        first.
        """
        items = np.asarray(items, dtype=np.int64)
        if items.size and not self.has_neighbors(items).all():
            raise ValueError(
                "cannot draw a partner for an item that co-occurred with nothing"
            )
        keys = items + rng.random(items.shape)
        return self.neighbors[np.searchsorted(self._keys, keys, side="right")]


def insert_cooccurring(
    history: AugmentedHistory,
    sampler: CooccurrenceSampler,
    probability: float,
    rng: np.random.Generator,
) -> AugmentedHistory:
    """After each item but the last, insert a partner of it at ``probability``.

    The inserted items are unobserved. None follows the last item: there the
    insertion would only take the last real transition's place as a target
    and fill a slot of the window with an item nothing reads. Items without a
    partner are skipped. At a rate of zero no random number is drawn.
    """
    if probability <= 0.0 or history.items.size < 2:
        return history
    anchors = history.items[:-1]
    chosen = (rng.random(anchors.size) < probability) & sampler.has_neighbors(
        anchors
    )
    if not chosen.any():
        return history
    after = np.flatnonzero(chosen) + 1
    inserted = sampler.sample(anchors[chosen], rng)
    return AugmentedHistory(
        np.insert(history.items, after, inserted),
        np.insert(history.observed, after, False),
    )


def replace_inputs(
    inputs: np.ndarray,
    eligible: np.ndarray,
    probability: float,
    token: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace each eligible input with ``token`` at rate ``probability``.

    ``inputs`` is modified in place and the positions replaced are returned, so
    a later corruption can be drawn over the positions this one left alone. At
    a rate of zero no random number is drawn, so a batch is exactly the one a
    config without the corruption would have produced from the same seed.
    """
    if probability <= 0.0:
        return np.zeros(inputs.shape, dtype=bool)
    chosen = eligible & (rng.random(inputs.shape) < probability)
    inputs[chosen] = token
    return chosen


def replace_with_cooccurring(
    inputs: np.ndarray,
    eligible: np.ndarray,
    probability: float,
    sampler: CooccurrenceSampler,
    offset: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace each eligible input with a partner of its item at ``probability``.

    ``inputs`` holds token ids, and ``offset`` is the tokenizer's
    ``n_reserved``: only catalog tokens are candidates, so a special token is
    never swapped for an item, and an item without a partner is left alone.
    Modified in place; returns the positions replaced, and draws nothing at a
    rate of zero, as :func:`replace_inputs` does.
    """
    if probability <= 0.0:
        return np.zeros(inputs.shape, dtype=bool)
    catalog = eligible & (inputs >= offset)
    items = np.where(catalog, inputs - offset, 0)
    chosen = (
        catalog
        & sampler.has_neighbors(items)
        & (rng.random(inputs.shape) < probability)
    )
    inputs[chosen] = sampler.sample(inputs[chosen] - offset, rng) + offset
    return chosen
