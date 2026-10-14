# Modified for seqrec_eval: differs from upstream compresso-recsys; see vendor/compresso-recsys/VENDORED.md
"""Full-softmax training and prediction for :class:`SASRecCE`.

:class:`~compresso_recsys.models.SASRecTrainer`'s shape -- one sample per user,
the most recent window of it, left padding, the final state ranks the catalog --
with the binary sampled objective replaced by cross entropy over the whole
catalog, and with training-time augmentation. The shape of a fit::

    ItemSequences, one sample per user
      -> the most recent max_history_length + 1 interactions
      -> co-occurrence insertion                              augmentation.py
      -> shift: input i predicts target i, left-padded to the window
      -> mask / co-occurrence replacement / unk on inputs     augmentation.py
      -> SASRecCE -> states at the labelled positions
      -> logits over the catalog, or over each user's n_negatives
         sampled items, the user's history kept out of either
         unless allow_history_negatives
      -> cross entropy against the true next items

Prediction is SASRec's, unchanged: encode the most recent ``max_history_length``
interactions, read the final state, rank the catalog.
"""

from __future__ import annotations

import time
from collections.abc import Hashable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from compresso import SRPTensor
from torch import nn

from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    _Reporter,
    _format_duration,
    _resolve_reporter,
)
from compresso_recsys.models.base import BaseSequentialRecommender
from compresso_recsys.models.identifiers import ItemVocabulary
from compresso_recsys.models.sequence_batching import SequenceBatcher
from compresso_recsys.models.tokenizer import ItemTokenizer
from compresso_recsys.persistence import (
    ModelCheckpointReader,
    ModelCheckpointWriter,
)
from compresso_recsys.sequences import ItemSequences

from .augmentation import (
    AugmentedHistory,
    CooccurrenceSampler,
    insert_cooccurring,
    replace_inputs,
    replace_with_cooccurring,
)
from .config import SASRecCEConfig
from .model import SASRecCE

__all__ = ["SASRecCETrainer"]

# Bert4Rec's layout. ``mask`` is what mask_probability substitutes, ``unk`` what
# unk_dropout substitutes and what an out-of-catalog item becomes at prediction.
SPECIAL_TOKENS = {"pad": 0, "mask": 1, "unk": 2}


class SASRecCETrainer(BaseSequentialRecommender):
    """Trains :class:`SASRecCE` with a full softmax, and serves it.

    Follows the package's shape, where ``fit`` returns the trainer and the
    trainer answers the prediction contract::

        model = SASRecCETrainer(
            SASRecCEConfig(cooccurrence_insert_probability=0.1)
        ).fit(split["x_train_sequences"])
        result = evaluate_recommender(
            model, source=split["test_source_sequences"],
            targets=split["test_target_matrix"], metrics=[NDCG(20)],
        )

    The batcher is a *parameter*, as it is for
    :class:`~compresso_recsys.models.SASRecTrainer`: passing one is how you
    change the vocabulary. The default vocabulary names ``pad``, ``mask`` and
    ``unk``. A vocabulary of your own needs ``mask`` while
    ``mask_probability`` is above zero, which it is by default, so either give
    it one or set the rate to zero. Leave the batcher's ``max_length``
    unset and it inherits the config's window; a batcher stating a different
    one is refused, because the window sizes the positional table.
    """

    checkpoint_type = "sasrec_ce_trainer"

    def __init__(
        self,
        config: SASRecCEConfig | None = None,
        batcher: SequenceBatcher | None = None,
        logger: Any | None = None,
    ) -> None:
        """Hold the config and encoder; build nothing until ``fit``.

        ``self._cooccurrence`` is the table the co-occurrence augmentations
        draw from. ``fit`` builds it from the training sequences, and it is
        deliberately not checkpointed: only training reads it, and a further
        ``fit`` builds it again from whatever it is given.
        """
        self.cfg = config or SASRecCEConfig()
        self.logger = logger
        self.device = torch.device(self.cfg.device)
        self.history: list[dict[str, float]] = []
        self.model: SASRecCE | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.batcher = batcher
        self._owns_batcher = batcher is None
        self._n_items: int | None = None
        self._cooccurrence: CooccurrenceSampler | None = None

    # -- contract -----------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been built and trained."""
        return self.model is not None

    @property
    def n_items(self) -> int | None:
        """Number of scoreable candidates, or ``None`` before fitting."""
        return self._n_items

    def _reporter(self, logger: Any, show_progress: Any) -> _Reporter:
        return _resolve_reporter(
            default_logger=self.logger,
            logger=logger,
            default_show_progress=self.cfg.show_progress,
            show_progress=show_progress,
            prefix=self.cfg.log_prefix,
            log_every_n_steps=self.cfg.log_every_n_steps,
        )

    # -- training -----------------------------------------------------------

    def fit(
        self,
        sequences: ItemSequences,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> SASRecCETrainer:
        """Train on chronological histories, one example per position.

        Validates the input, builds a default batcher when none was supplied,
        checks it against the training catalog and the config, records the
        item IDs, seeds Torch and NumPy from ``cfg.seed``, builds the model and
        optimizer, then runs ``cfg.epochs`` passes over shuffled rows and
        appends one entry per epoch to :attr:`history`.

        A history needs two retained interactions to yield one shifted
        example, as SASRec's does. Unlike SASRec's, a one-item catalog is not
        refused: a softmax needs no negative to be drawn, it just has nothing
        to learn.

        Rebuilds the model on every call: early stopping and incremental
        training are not part of this contract.
        """
        reporter = self._reporter(logger, show_progress)
        if not isinstance(sequences, ItemSequences):
            raise TypeError(
                "SASRecCETrainer trains on ItemSequences, got "
                f"{type(sequences).__name__}"
            )
        if sequences.n_rows == 0:
            raise ValueError("cannot train on zero sequences")
        if sequences.n_items < 1:
            raise ValueError("cannot train on an empty catalog")

        if self._owns_batcher:
            self.batcher = SequenceBatcher(
                ItemTokenizer(sequences.n_items, special_tokens=SPECIAL_TOKENS),
                max_length=self.cfg.max_history_length,
                padding="left",
            )
        if self.batcher is None:  # pragma: no cover - defensive against mutation
            raise RuntimeError("trainer batcher is unavailable")
        if self.batcher.tokenizer.n_items != sequences.n_items:
            raise ValueError(
                "batcher tokenizer has "
                f"{self.batcher.tokenizer.n_items} items, but training "
                f"sequences have {sequences.n_items}"
            )
        self._check_tokenizer(self.batcher.tokenizer)
        tokenizer_ids = getattr(self.batcher.tokenizer, "item_ids", None)
        if item_ids is not None and tokenizer_ids is not None:
            supplied = ItemVocabulary.from_ids(item_ids).item_ids
            if not np.array_equal(supplied, tokenizer_ids):
                raise ValueError(
                    "item_ids must match the batcher tokenizer item IDs"
                )
        self._set_item_ids(
            tokenizer_ids if item_ids is None else item_ids,
            n_items=sequences.n_items,
        )
        # A batcher stating no window inherits the config's, and the batcher is
        # frozen, so this is a new one rather than a mutation of what the
        # caller handed over.
        if self.batcher.max_length is None:
            self.batcher = replace(
                self.batcher, max_length=self.cfg.max_history_length
            )
        # Left padding is the architecture rather than a preference -- see
        # SASRecCE -- so it is set here rather than asked of the caller.
        if self.batcher.padding != "left":
            self.batcher = replace(self.batcher, padding="left")
        self._check_batcher(self.batcher)
        # Training keeps max_history_length + 1 interactions, and two of them
        # are enough for one example, so truncation never makes a trainable
        # history untrainable and the raw lengths answer the question.
        if not (sequences.row_lengths >= 2).any():
            raise ValueError(
                "no history retains two or more interactions, so there is no "
                "next-item example to learn from"
            )

        torch.manual_seed(int(self.cfg.seed))
        # One generator for the row order and the augmentation both, so a run
        # is reproducible from cfg.seed alone.
        rng = np.random.default_rng(int(self.cfg.seed))
        # From the full training histories rather than the windows training
        # reads, since more history is more evidence about which items go
        # together, and none of it is held out.
        self._cooccurrence = (
            CooccurrenceSampler.from_sequences(
                sequences,
                window=self.cfg.cooccurrence_window,
                top_k=self.cfg.cooccurrence_top_k,
                popularity_exponent=self.cfg.cooccurrence_popularity_exponent,
            )
            if self.cfg.uses_cooccurrence
            else None
        )

        self._n_items = self.batcher.tokenizer.n_items
        self.model = self._build_model()
        self._build_checkpoint_optimizer()
        assert self.optimizer is not None
        self.history = []

        n_rows = sequences.n_rows
        batch_size = self.cfg.batch_size
        starts = range(0, n_rows, batch_size)
        fit_started = time.monotonic()
        reporter.log(
            "fit started: "
            f"{n_rows} sequences | {self._n_items} items | {len(starts)} batches of "
            f"{batch_size} | {self.cfg.epochs} epochs | device {self.device}"
        )
        epoch_iter = reporter.wrap(
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc="SASRecCE fit",
        )
        batch_bar = reporter.bar(total=len(starts), desc="SASRecCE epoch 1")
        try:
            for epoch in epoch_iter:
                epoch_started = time.monotonic()
                self.model.train()
                order = rng.permutation(n_rows)
                if batch_bar is not None:
                    batch_bar.reset(total=len(starts))
                    batch_bar.set_description(f"SASRecCE epoch {epoch}")
                loss_sum, positions = 0.0, 0
                for step_index, start in enumerate(starts, start=1):
                    batch = sequences.select_rows(
                        order[start : start + batch_size]
                    )
                    step = self._train_step(batch, rng)
                    if step is not None:
                        batch_loss, batch_positions = step
                        loss_sum += batch_loss * batch_positions
                        positions += batch_positions
                    if batch_bar is not None:
                        batch_bar.update(1)
                    log_steps = reporter.log_every_n_steps
                    if log_steps and step_index % log_steps == 0:
                        reporter.step(
                            f"epoch {epoch}/{self.cfg.epochs} step "
                            f"{step_index}/{len(starts)}",
                            step_index,
                            len(starts),
                            epoch_started,
                            {
                                "loss": (
                                    loss_sum / positions
                                    if positions
                                    else float("nan")
                                )
                            },
                        )
                mean_loss = loss_sum / positions if positions else float("nan")
                record = {
                    "epoch": float(epoch),
                    "loss": mean_loss,
                    "positions": float(positions),
                }
                self.history.append(record)
                reporter.epoch(
                    f"epoch {epoch}/{self.cfg.epochs}",
                    record,
                    epoch_started,
                )
                if hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix({"loss": f"{mean_loss:.4f}"})
        finally:
            if batch_bar is not None:
                batch_bar.close()
            if hasattr(epoch_iter, "close"):
                epoch_iter.close()

        reporter.log(
            f"fit finished: {_format_duration(time.monotonic() - fit_started)} total | "
            f"{len(self.history)} epochs recorded"
        )
        return self

    def _check_tokenizer(self, tokenizer: Any) -> None:
        """Require a ``mask`` token whenever masking is switched on.

        Masking with an item id instead would teach the model that that item
        means "something was here", and masking with ``unk`` would merge two
        tokens that mean different things at prediction time. At a rate of
        zero nothing is substituted, so any vocabulary will do.
        """
        if self.cfg.mask_probability <= 0.0:
            return
        token_id = getattr(tokenizer, "token_id", None)
        if token_id is None:
            raise TypeError(
                "mask_probability needs a tokenizer with named special tokens, "
                f"got {type(tokenizer).__name__}"
            )
        try:
            token_id("mask")
        except KeyError:
            raise ValueError(
                "mask_probability > 0 needs a 'mask' special token; build the "
                "tokenizer with ItemTokenizer(n_items, special_tokens="
                f"{SPECIAL_TOKENS}), or set mask_probability=0"
            ) from None

    def _check_batcher(self, batcher: SequenceBatcher) -> None:
        """Reject a batcher whose window disagrees with the config's.

        The window has one owner, ``cfg.max_history_length``, because it sizes
        the positional embedding and a checkpoint cannot grow one after the
        fact.
        """
        if batcher.max_length is None:  # pragma: no cover - fit resolves it
            raise RuntimeError(
                "batcher window was not resolved before _check_batcher"
            )
        if batcher.max_length != self.cfg.max_history_length:
            raise ValueError(
                f"batcher max_length is {batcher.max_length} but "
                f"cfg.max_history_length is {self.cfg.max_history_length}. "
                "The window sizes the positional table, so it has a single "
                "owner: set it on the config, or leave the batcher's "
                "max_length as None to inherit it"
            )

    def _build_model(self) -> SASRecCE:
        """Construct :class:`SASRecCE` from the config and the batcher's tokenizer."""
        assert self.batcher is not None
        tokenizer = self.batcher.tokenizer
        return SASRecCE(
            n_items=tokenizer.n_items,
            n_reserved=tokenizer.n_reserved,
            # No +1: SASRecCE adds the extra row itself, because it numbers
            # positions from one and keeps index 0 for padding steps.
            max_history_length=int(self.batcher.max_length),
            pad_id=tokenizer.pad_id,
            d_model=self.cfg.d_model,
            n_blocks=self.cfg.n_blocks,
            n_heads=self.cfg.n_heads,
            dropout=self.cfg.dropout,
        ).to(self.device)

    # -- samples ------------------------------------------------------------

    def _training_batch(
        self, batch: ItemSequences, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment, shift, tokenize and left-pad one batch of histories.

        Returns ``(inputs, targets)``, both ``(rows, max_history_length)`` token
        ids, aligned so that the state at ``inputs[:, i]`` predicts
        ``targets[:, i]``. A target of ``pad`` marks a position with nothing to
        predict.

        Each row keeps its most recent ``max_history_length + 1`` interactions:
        the shift spends one, so a history that fills the window puts an input
        on every position the model owns, and prediction -- which does not
        shift -- reads every position that training trained. Sequence edits run
        on that tail through :meth:`_augment_history`, and the window is
        applied again afterwards in case an edit lengthened it.

        A target counts when it is a catalog item the user interacted with, or
        an inserted one under ``cfg.score_inserted_items``. Padding is excluded
        because it is not an item, and ``unk`` because "predict the item you
        cannot identify" has no answer. The step before the first real input
        has a real target on a padded input, and is excluded by construction:
        the targets are the tokens after the first, so nothing is laid over a
        pad input.

        Input corruption then runs over the real inputs in three draws, each
        over the positions the earlier ones left alone: ``mask``, a
        co-occurring item, ``unk``. Targets are never corrupted, so no example
        is lost to it, and padding never is, so no replacement comes to mean
        "nothing here".
        """
        assert self.batcher is not None and self.batcher.max_length is not None
        tokenizer = self.batcher.tokenizer
        pad_id = tokenizer.pad_id
        offset = tokenizer.n_reserved
        window = int(self.batcher.max_length)

        inputs = np.full((batch.n_rows, window), pad_id, dtype=np.int64)
        targets = np.full((batch.n_rows, window), pad_id, dtype=np.int64)
        for row in range(batch.n_rows):
            history = AugmentedHistory.observed_history(
                batch.row(row)[-(window + 1) :]
            )
            history = self._augment_history(history, rng).tail(window + 1)
            if history.items.size < 2:
                continue
            tokens = tokenizer.encode_indices(history.items)
            length = int(tokens.size) - 1
            scored = tokens[1:] >= offset
            if not self.cfg.score_inserted_items:
                scored &= history.observed[1:]
            inputs[row, window - length :] = tokens[:-1]
            targets[row, window - length :] = np.where(scored, tokens[1:], pad_id)

        real = inputs != pad_id
        taken = np.zeros_like(real)
        if self.cfg.mask_probability > 0.0:
            taken |= replace_inputs(
                inputs,
                real,
                self.cfg.mask_probability,
                tokenizer.token_id("mask"),
                rng,
            )
        if self._cooccurrence is not None:
            taken |= replace_with_cooccurring(
                inputs,
                real & ~taken,
                self.cfg.cooccurrence_replace_probability,
                self._cooccurrence,
                offset,
                rng,
            )
        unk_id = getattr(tokenizer, "unk_id", None)
        if unk_id is not None:
            replace_inputs(inputs, real & ~taken, self.cfg.unk_dropout, unk_id, rng)

        return (
            torch.from_numpy(inputs).to(self.device),
            torch.from_numpy(targets).to(self.device),
        )

    def _augment_history(
        self, history: AugmentedHistory, rng: np.random.Generator
    ) -> AugmentedHistory:
        """Sequence edits for one row: co-occurrence insertion.

        The hook for edits that rewrite the history before the shift.
        ``history`` is the row's retained tail, every item observed. An edit
        returns a new :class:`~.augmentation.AugmentedHistory`, marking what it
        introduced as unobserved so that :meth:`_training_batch` can tell it
        apart from what the user did; it may change the length, and may draw
        from ``rng``, the generator ``fit`` seeds from ``cfg.seed``. A further
        edit is composed here, after or before this one.
        """
        if self._cooccurrence is not None:
            history = insert_cooccurring(
                history,
                self._cooccurrence,
                self.cfg.cooccurrence_insert_probability,
                rng,
            )
        return history

    # -- objective ----------------------------------------------------------

    def _train_step(
        self, batch: ItemSequences, rng: np.random.Generator
    ) -> tuple[float, int] | None:
        """One optimizer step, or ``None`` when the batch carries no target.

        Builds the batch, runs the model once and gathers its states at the
        labelled positions. The loss of the objective ``cfg.n_negatives``
        selects -- :meth:`_full_softmax_loss` or :meth:`_sampled_softmax_loss`
        -- is then taken over those positions, in chunks when they would hold
        more than ``cfg.loss_chunk_elements`` at once (see
        :class:`~.config.SASRecCEConfig`). Each chunk backpropagates into the
        item embeddings and into a detached copy of the states, and the encoder
        gets the summed gradient once. Whatever the objective draws at random
        is drawn before the chunks, so chunking cannot change it.

        Returns the mean loss and the number of positions it covers, so
        ``fit`` can weight epochs by position count rather than by batch.
        """
        assert self.model is not None and self.optimizer is not None
        assert self.batcher is not None
        inputs, targets = self._training_batch(batch, rng)
        labelled = targets != self.batcher.tokenizer.pad_id
        n_positions = int(labelled.sum())
        if n_positions == 0:
            return None

        states = self.model(inputs)
        # Boolean indexing gathers in row-major order, and nonzero lists
        # positions in the same order, so ``rows`` is each position's user.
        selected = states[labelled]
        rows = labelled.nonzero(as_tuple=True)[0]
        target_tokens = targets[labelled]
        if self.cfg.n_negatives is None:
            loss_of = self._full_softmax_loss(batch)
            per_position = int(self.model.n_items)
        else:
            loss_of = self._sampled_softmax_loss(batch, rng)
            per_position = int(self.cfg.n_negatives) * int(selected.shape[-1])
        chunk = max(1, self.cfg.loss_chunk_elements // per_position)

        self.optimizer.zero_grad(set_to_none=True)
        if chunk >= n_positions:
            loss = loss_of(selected, rows, target_tokens) / n_positions
            loss.backward()
            total = float(loss.detach())
        else:
            detached = selected.detach().requires_grad_()
            total = 0.0
            for start in range(0, n_positions, chunk):
                part = slice(start, start + chunk)
                part_loss = (
                    loss_of(detached[part], rows[part], target_tokens[part])
                    / n_positions
                )
                part_loss.backward()
                total += float(part_loss.detach())
            selected.backward(detached.grad)
        self.optimizer.step()
        return total, n_positions

    def _full_softmax_loss(self, batch: ItemSequences):
        """The summed cross entropy over the whole catalog, for some positions.

        Returns ``loss_of(states, rows, target_tokens)``: ``states`` is
        ``(positions, d_model)``, ``rows`` each position's batch row and
        ``target_tokens`` its target's token id. Unless history negatives are
        allowed, each position's user's history is first removed from its
        softmax (:meth:`_without_history_negatives`); the table that needs is
        built once for the batch, and each call reads only its own positions'
        rows of it, so a chunk holds a mask of its own size and no more.
        """
        assert self.model is not None and self.batcher is not None
        offset = self.batcher.tokenizer.n_reserved
        seen = (
            None
            if self.cfg.allow_history_negatives
            else self._history_table(batch)
        )

        def loss_of(states, rows, target_tokens):
            logits = self.model.score(states)
            labels = target_tokens - offset
            if seen is not None:
                logits = self._without_history_negatives(logits, labels, seen[rows])
            return F.cross_entropy(logits, labels, reduction="sum")

        return loss_of

    def _sampled_softmax_loss(self, batch: ItemSequences, rng: np.random.Generator):
        """The summed cross entropy over each target and its user's negatives.

        Returns ``loss_of(states, rows, target_tokens)`` as
        :meth:`_full_softmax_loss` does. The negatives are drawn here, once per
        user for the whole batch (:meth:`_sample_negatives`), and each position
        is scored against its own user's: a ``positions x n_negatives x
        d_model`` gather, which the chunking bounds. Only labelled positions
        are scored, so padding costs nothing.

        The target's logit goes first, at index 0. A negative equal to the
        position's own target is ``-inf``, so the target never competes with
        itself, and so is every negative of a user who had none to draw; their
        positions see only the target and contribute zero loss, as they would
        under the full softmax.
        """
        assert self.model is not None
        negatives, drawable = self._sample_negatives(batch, rng)

        def loss_of(states, rows, target_tokens):
            drawn = negatives[rows]
            negative_logits = torch.bmm(
                self.model.item_embedding(drawn), states.unsqueeze(-1)
            ).squeeze(-1)
            dropped = (drawn == target_tokens.unsqueeze(1)) | ~drawable[
                rows
            ].unsqueeze(1)
            negative_logits = negative_logits.masked_fill(dropped, -torch.inf)
            positive_logits = self.model.score_items(states, target_tokens)
            logits = torch.cat(
                (positive_logits.unsqueeze(1), negative_logits), dim=1
            )
            return F.cross_entropy(
                logits, torch.zeros_like(target_tokens), reduction="sum"
            )

        return loss_of

    def _sample_negatives(
        self, batch: ItemSequences, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``cfg.n_negatives`` catalog tokens per user, and who had any to draw.

        Returns ``(rows, n_negatives)`` token ids and a ``(rows,)`` mask. The
        draw is uniform with replacement: over the whole catalog when history
        negatives are allowed, otherwise over the catalog minus the user's full
        history, ``I \\ S_u``. Reserved ids are never drawn.

        The second draw is SASRec's mapping rather than a rejection loop whose
        length would depend on the data. Each draw is uniform over the
        ``n_items - |S_u|`` allowed slots and then stepped onto the complement:
        with ``S_u`` sorted, ``seen[j] - j`` is how many allowed items fall
        below ``seen[j]``, so where a draw lands in that sequence is exactly
        how many exclusions it has to step over.

        A user whose history covers the catalog has nothing to draw. The row is
        filled with a valid token, so the embedding lookup stays in range, and
        reported as undrawable, so the loss drops it.
        """
        assert self._n_items is not None and self.batcher is not None
        assert self.cfg.n_negatives is not None
        n_items = self._n_items
        shape = (batch.n_rows, int(self.cfg.n_negatives))
        if self.cfg.allow_history_negatives:
            draws = rng.integers(0, n_items, size=shape)
            drawable = np.ones(batch.n_rows, dtype=bool)
        else:
            excluded, n_excluded = self._excluded_items(batch, n_items)
            available = n_items - n_excluded
            drawable = available > 0
            draws = rng.integers(0, np.maximum(available, 1)[:, None], size=shape)
            below = excluded - np.arange(excluded.shape[1])
            # Padding sits above every possible draw, so it is never stepped over.
            below[excluded >= n_items] = n_items + 1
            draws = draws + torch.searchsorted(
                torch.from_numpy(below), torch.from_numpy(draws), right=True
            ).numpy()
            draws[~drawable] = 0
        return (
            torch.as_tensor(
                draws + self.batcher.tokenizer.n_reserved,
                dtype=torch.long,
                device=self.device,
            ),
            torch.as_tensor(drawable, device=self.device),
        )

    @staticmethod
    def _excluded_items(
        batch: ItemSequences, n_items: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Each row's item set, sorted and padded, with the count of real entries.

        Unused slots hold ``n_items``, which is above every catalog position and
        so sorts to the end and compares out of range wherever it is tested.
        Duplicates collapse: an item interacted with twice is one exclusion, and
        counting it twice would shrink the range the draw is uniform over and
        push the mapping past items that were never excluded.
        """
        lengths = batch.row_lengths
        width = int(lengths.max()) if batch.n_rows else 0
        excluded = np.full((batch.n_rows, width), n_items, dtype=np.int64)
        if width:
            filled = np.arange(width)[None, :] < lengths[:, None]
            excluded[filled] = np.asarray(batch.values, dtype=np.int64)
            excluded.sort(axis=1)
            duplicate = np.zeros_like(excluded, dtype=bool)
            duplicate[:, 1:] = excluded[:, 1:] == excluded[:, :-1]
            # Real items only: the pad value repeats by construction.
            duplicate &= excluded < n_items
            excluded[duplicate] = n_items
            excluded.sort(axis=1)
        return excluded, (excluded < n_items).sum(axis=1)

    def _history_table(self, batch: ItemSequences) -> torch.Tensor:
        """Each batch row's item set as a ``(rows, n_items)`` boolean table.

        ``S_u`` is the row's *full* sequence in ``batch``: the part truncated
        from the window, and interactions after a position as well as before
        it. It is the observed history, so an item a sequence edit introduced
        is not in it.
        """
        assert self._n_items is not None
        n_rows = batch.n_rows
        # np.array copies, because the buffers are read-only and
        # torch.as_tensor would otherwise warn about sharing them.
        user = np.repeat(np.arange(n_rows), batch.row_lengths)
        item = np.array(batch.values, dtype=np.int64)
        seen = torch.zeros((n_rows, self._n_items), dtype=torch.bool, device=self.device)
        seen[
            torch.as_tensor(user, device=self.device),
            torch.as_tensor(item, device=self.device),
        ] = True
        return seen

    @staticmethod
    def _without_history_negatives(
        logits: torch.Tensor, labels: torch.Tensor, seen: torch.Tensor
    ) -> torch.Tensor:
        """Remove each user's other interactions from their softmax.

        ``logits`` is ``(positions, n_items)``, ``labels`` the catalog index of
        each position's target and ``seen`` each position's own row of
        :meth:`_history_table`, already gathered. Every item in that history
        becomes ``-inf`` except the target itself, so it leaves the normaliser
        and receives no gradient -- the softmax counterpart of drawing
        negatives from ``I \\ S_u``. The target is restored even when the user
        interacted with it elsewhere too, so no position loses its own label --
        and since the target's logit always stays finite, the normaliser is
        never empty and the loss never reaches ``nan``, even for a user whose
        history covers the whole catalog.
        """
        # ``seen[rows]`` is an advanced-indexing copy, so clearing the targets
        # cannot leak into another position of the same user; cloned anyway,
        # since a caller may pass a view.
        excluded = seen.clone()
        excluded[torch.arange(labels.shape[0], device=logits.device), labels] = False
        return logits.masked_fill(excluded, -torch.inf)

    # -- prediction ---------------------------------------------------------

    def predict_on_batch(
        self,
        source: ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Rank the catalog for each history from its final state.

        Resolves candidates, validates ``k``, and returns an empty
        :class:`~compresso.SRPTensor` for an empty batch. Otherwise encodes the
        source, takes each row's last real state through the batcher's
        ``gather_final``, scores the catalog, masks seen items when asked, and
        takes the top ``k`` over the candidate columns. No augmentation runs
        here: ``mask`` is a training-time token only.
        """
        if self.model is None or self.batcher is None or self._n_items is None:
            raise RuntimeError("SASRecCETrainer must be fitted before predicting")
        if not isinstance(source, ItemSequences):
            raise TypeError(
                "SASRecCETrainer predicts from ItemSequences, got "
                f"{type(source).__name__}"
            )
        n_items = self._n_items
        candidate_rows = self._candidate_rows(candidate_ids)
        candidate_count = int(candidate_rows.size)
        if not 1 <= k <= candidate_count:
            raise ValueError(f"k must be in [1, {candidate_count}], got {k}")
        if exclude_seen:
            self._check_unseen_capacity(
                source,
                n_items=n_items,
                k=k,
                candidate_rows=candidate_rows,
            )

        rows = source.n_rows
        if rows == 0:
            return SRPTensor(
                cols=torch.empty((0, k), dtype=torch.long, device=self.device),
                vals=torch.empty(
                    (0, k), dtype=torch.float32, device=self.device
                ),
                shape=(0, n_items),
            )

        self.model.eval()
        step = self._prediction_rows()
        with torch.no_grad():
            item_history, mask = self.batcher.encode(
                source, device=self.device
            )
            candidates = torch.from_numpy(candidate_rows).long().to(self.device)
            parts = []
            for start in range(0, rows, step):
                stop = min(start + step, rows)
                final = self.batcher.gather_final(
                    self.model(item_history[start:stop]), mask[start:stop]
                )
                scores = self.model.score(final)
                if exclude_seen:
                    self._mask_seen(scores, source.take_rows(start, stop))
                parts.append(torch.topk(scores[:, candidates], k, dim=1))
            vals = torch.cat([part.values for part in parts])
            cols = candidates[torch.cat([part.indices for part in parts])]

        return SRPTensor(cols=cols, vals=vals, shape=(rows, n_items))

    def _prediction_rows(self) -> int:
        """How many histories :meth:`predict_on_batch` scores at once.

        Each history holds an attention map of ``n_heads x window x window`` per
        block and a row of ``n_items`` scores, and a batch holds that many of
        each, so a caller's batch is scored in parts of at most
        ``cfg.prediction_chunk_elements`` of the larger. At window 400 with two
        heads a map is 320,000 elements, and 1,024 histories' maps are 1.2 GB
        per block, which is what a 6 GB card runs out on. The rows are
        independent and every one is padded to the full window, so the parts
        rank exactly as one batch would.
        """
        assert self.model is not None and self.batcher is not None
        window = int(self.batcher.max_length)
        per_row = max(self.model.n_heads * window * window, self.model.n_items)
        return max(1, self.cfg.prediction_chunk_elements // per_row)

    def _mask_seen(self, scores: torch.Tensor, source: ItemSequences) -> None:
        """Forbid every item in the *full* history, truncated part included.

        Items beyond the fitted catalog are dropped from the mask rather than
        clipped: they were never scoreable, so there is nothing to forbid.
        """
        if source.values.size == 0:
            return
        n_items = int(scores.shape[1])
        rows = np.repeat(np.arange(source.n_rows), source.row_lengths)
        cols = np.array(source.values, dtype=np.int64)
        scoreable = cols < n_items
        if not scoreable.all():
            rows, cols = rows[scoreable], cols[scoreable]
        if cols.size == 0:
            return
        scores[
            torch.as_tensor(rows, dtype=torch.long, device=scores.device),
            torch.as_tensor(cols, dtype=torch.long, device=scores.device),
        ] = -torch.inf

    # -- persistence --------------------------------------------------------

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> SASRecCETrainer:
        """Rebuild the trainer's shape before learned state is installed."""
        config = dict(config)
        # The stored device is where the model was trained, which says nothing
        # about where it is being loaded. The caller's choice wins.
        config["device"] = str(device)
        trainer_state = reader.read_json("state/trainer.json")
        tokenizer_state = reader.read_json("state/tokenizer.json")
        if reader.exists("state/tokenizer_item_ids.json"):
            tokenizer_state["item_ids"] = reader.read_item_ids(
                "state/tokenizer_item_ids.json"
            )
        tokenizer = ItemTokenizer.from_dict(tokenizer_state)
        batcher = SequenceBatcher(
            tokenizer,
            max_length=int(trainer_state["max_length"]),
            padding="left",
        )
        trainer = cls(SASRecCEConfig(**config), batcher)
        trainer._n_items = tokenizer.n_items
        # Built by _build_model rather than inline, so a reloaded model is
        # constructed by exactly the path that trained it.
        trainer.model = trainer._build_model()
        return trainer

    def _checkpoint_module(self) -> nn.Module | None:
        return self.model

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        """Write the non-module state: ``max_length``, history, and tokenizer."""
        if self.batcher is None or not isinstance(
            self.batcher.tokenizer, ItemTokenizer
        ):
            raise TypeError(
                "SASRecCETrainer checkpoints support ItemTokenizer only"
            )
        assert self.batcher.max_length is not None
        writer.write_json(
            "state/trainer.json",
            {
                "max_length": int(self.batcher.max_length),
                "history": self.history,
            },
        )
        writer.write_json(
            "state/tokenizer.json",
            self.batcher.tokenizer.to_dict(include_item_ids=False),
        )
        item_ids = self.batcher.tokenizer.item_ids
        if item_ids is not None:
            writer.write_item_ids("state/tokenizer_item_ids.json", item_ids)

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        """Restore :attr:`history` from the archive."""
        state = reader.read_json("state/trainer.json")
        history = state.get("history")
        if not isinstance(history, list):
            raise TypeError("SASRecCE training history must be a list")
        self.history = list(history)

    def _build_checkpoint_optimizer(self) -> None:
        """Construct the optimizer, for ``fit`` and before state is loaded into it."""
        if self.model is None:
            raise RuntimeError("SASRecCE model must be built before its optimizer")
        self.optimizer = getattr(torch.optim, self.cfg.optimizer)(
            self.model.parameters(),
            lr=self.cfg.lr,
            **self.cfg.optimizer_kwargs(),
        )
