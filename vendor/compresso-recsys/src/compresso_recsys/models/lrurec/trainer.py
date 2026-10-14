# Modified for seqrec_eval: differs from upstream compresso-recsys; see vendor/compresso-recsys/VENDORED.md
"""Next-item training and prediction for LRURec.

Yue et al., *Linear Recurrent Units for Sequential Recommendation*, WSDM 2024.
The model in :mod:`.model` is the paper's architecture; this module is its
training objective -- Eq. 1's negative log-likelihood of the next item -- and the
sample construction the paper leaves to its implementation.

The shape of a fit::

    ItemSequences
      -> end-to-end windows over histories that outgrow the window  (reference)
      -> each sample shifted one step: input i predicts label i     (Eq. 1)
      -> tokenize, left-pad to the batch's longest row
      -> LRURec -> vocabulary logits at every labelled position
      -> cross entropy against the true next items                  (§5.1)
         or, with n_negatives set, against sampled items only        (§5.3)

Prediction encodes the most recent ``max_history_length`` interactions and ranks
the catalog from the final position's state, as the reference's evaluation does.
:meth:`LRURecTrainer.score_candidates` scores chosen items from that same state,
which is what XLong's sampled evaluation needs.

Where the paper does not specify something, this follows the authors' released
PyTorch implementation (https://github.com/yueqirex/LRURec) and says so at the
point of the decision. Where the two disagree, the implementation wins.
"""

from __future__ import annotations

import time
from collections.abc import Hashable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from compresso import SRPTensor
from torch import nn
from torch.nn import functional as F

from compresso_recsys._reporting import (
    _INHERIT,
    _format_duration,
    _Inherit,
    _Reporter,
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

from .config import LRURecConfig
from .model import LRULayer, LRURec

__all__ = ["LRURecTrainer"]


def _sliding_windows(
    history: np.ndarray, width: int, step: int
) -> list[np.ndarray]:
    """One history as the reference's ``SASTrainDataset`` cuts it.

    A history shorter than ``width + step`` is a single sample, kept whole: the
    shift in :meth:`LRURecTrainer._next_item_batch` reads only its last
    ``width`` labels, so the part of a history between ``width`` and
    ``width + step`` long is never trained on. A longer history becomes windows
    of exactly ``width``, the first ending at the most recent interaction and
    each further one ``step`` earlier, for as long as one fits.

    Two things here differ from BERT4Rec's windows, deliberately. The oldest
    window is not pinned to the start, so up to ``step - 1`` of the earliest
    interactions are dropped; and at the default step of one whole window the
    windows do not overlap, so the transition across each boundary is never a
    training example. Both are the reference's, which is what was published.
    """
    length = int(history.size)
    if length < width + step:
        return [history]
    return [
        history[start : start + width]
        for start in range(length - width, -1, -step)
    ]


class LRURecTrainer(BaseSequentialRecommender):
    """Train :class:`LRURec` on next-item prediction and rank from its last state.

    Follows the package's shape, where ``fit`` returns the trainer and the
    trainer answers the prediction contract::

        model = LRURecTrainer(LRURecConfig()).fit(split["x_train_sequences"])
        result = evaluate_recommender(
            model, source=split["test_source_sequences"],
            targets=split["test_target_matrix"], metrics=[NDCG(20)],
        )

    The batcher is a *parameter*, not something ``fit`` invents. Passing one is
    how you change the vocabulary -- including the ``unk`` slot the default
    vocabulary already has, which lets a later split stage's unseen items
    become a reserved id rather than an error. Leave its ``max_length`` unset
    and it inherits the config's window, so the number stays in one place.

    The trainer owns tokenization because the checkpointed histories
    deliberately do not: :class:`~compresso_recsys.ItemSequences` holds catalog
    indices with no padding and no special tokens.
    """

    checkpoint_type = "lrurec_trainer"

    def __init__(
        self,
        config: LRURecConfig | None = None,
        batcher: SequenceBatcher | None = None,
        logger: Any | None = None,
    ) -> None:
        self.cfg = config or LRURecConfig()
        self.logger = logger
        self.device = torch.device(self.cfg.device)
        self.history: list[dict[str, float]] = []
        self.model: LRURec | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.batcher = batcher
        self._owns_batcher = batcher is None
        self._n_items: int | None = None

    # -- contract -----------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    @property
    def n_items(self) -> int | None:
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
    ) -> LRURecTrainer:
        """Train on chronological histories, one target per interaction.

        Validates the input, builds a default batcher when none was supplied,
        reconciles its window with the config, records the item IDs, seeds Torch
        and NumPy from ``cfg.seed``, builds the model and optimizer, then runs
        ``cfg.epochs`` shuffled passes over the samples.

        Rebuilds the model on every call: early stopping and incremental
        training are not part of this contract.
        """
        reporter = self._reporter(logger, show_progress)
        if not isinstance(sequences, ItemSequences):
            raise TypeError(
                "LRURecTrainer trains on ItemSequences, got "
                f"{type(sequences).__name__}"
            )
        if sequences.n_rows == 0:
            raise ValueError("cannot train on zero sequences")
        if sequences.n_items < 1:
            raise ValueError("cannot train on an empty catalog")

        if self._owns_batcher:
            self.batcher = SequenceBatcher(
                ItemTokenizer(sequences.n_items),
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
        # A batcher handed over for its vocabulary usually states no window; it
        # inherits the config's. Nothing in the model is sized by the window --
        # the recurrence has no positional table -- but it is still one number:
        # it decides how training histories are cut into samples and how much
        # history prediction reads, and those two disagreeing would score
        # histories of a length the model was never trained on. So stating both
        # and disagreeing is an error rather than a silent win for either.
        if self.batcher.max_length is None:
            self.batcher = replace(
                self.batcher, max_length=self.cfg.max_history_length
            )
        elif int(self.batcher.max_length) != self.cfg.max_history_length:
            raise ValueError(
                f"batcher max_length ({self.batcher.max_length}) and config "
                f"max_history_length ({self.cfg.max_history_length}) disagree; "
                "the window decides both how training histories are cut and "
                "how much history prediction reads, so it cannot be two numbers"
            )
        # Left padding is the architecture rather than a preference: the scan's
        # padding mask is exact only when padding is a prefix (see
        # LRULayer._parallel_scan), and it is the reference's. It also puts
        # every row's newest interaction in the last column, which is where
        # prediction reads.
        if self.batcher.padding != "left":
            self.batcher = replace(self.batcher, padding="left")

        # Checked before the model is built, so a refused fit does not leave
        # an untrained model behind that reports itself as fitted.
        samples = self._training_samples(sequences)
        if not samples:
            raise ValueError(
                "every history is empty, so there is no next item to learn from"
            )

        torch.manual_seed(int(self.cfg.seed))
        # One generator for row order and unk corruption alike, so a run is
        # reproducible from cfg.seed alone.
        rng = np.random.default_rng(int(self.cfg.seed))
        self._n_items = sequences.n_items
        self.model = self._build_model()
        self._build_checkpoint_optimizer()
        assert self.optimizer is not None
        self.history = []

        n_samples = len(samples)
        batch_size = self.cfg.batch_size
        starts = range(0, n_samples, batch_size)

        fit_started = time.monotonic()
        reporter.log(
            "fit started: "
            f"{sequences.n_rows} sequences -> {n_samples} samples | "
            f"{self._n_items} items | {len(starts)} batches of {batch_size} | "
            f"{self.cfg.epochs} epochs | device {self.device}"
        )
        epoch_iter = reporter.wrap(
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc="LRURec fit",
        )
        batch_bar = reporter.bar(total=len(starts), desc="LRURec epoch 1")
        try:
            for epoch in epoch_iter:
                epoch_started = time.monotonic()
                self.model.train()
                order = rng.permutation(n_samples)
                if batch_bar is not None:
                    batch_bar.reset(total=len(starts))
                    batch_bar.set_description(f"LRURec epoch {epoch}")
                loss_sum, positions = 0.0, 0
                for step_index, start in enumerate(starts, start=1):
                    selected = order[start : start + batch_size]
                    step = self._train_step(
                        [samples[int(index)] for index in selected], rng
                    )
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
                    f"epoch {epoch}/{self.cfg.epochs}", record, epoch_started
                )
                if hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix({"loss": f"{mean_loss:.4f}"})
        finally:
            if batch_bar is not None:
                batch_bar.close()
            if hasattr(epoch_iter, "close"):
                epoch_iter.close()

        reporter.log(
            f"fit finished: {_format_duration(time.monotonic() - fit_started)} "
            f"total | {len(self.history)} epochs recorded"
        )
        return self

    # -- sample construction ------------------------------------------------

    def _training_samples(self, sequences: ItemSequences) -> list[np.ndarray]:
        """Every sample to train on, as the reference's ``SASTrainDataset``.

        See :func:`_sliding_windows` for how one history is cut. The step is
        ``int(sliding_window_step * max_history_length)``, truncated as the
        reference truncates it.

        Empty histories are dropped. They carry no label, so the reference's
        cross entropy would ignore every position of one; dropping them here
        changes no gradient and keeps a batch made only of them from averaging
        over nothing.
        """
        width = self.cfg.max_history_length
        step = int(self.cfg.sliding_window_step * width)
        samples: list[np.ndarray] = []
        for row in range(sequences.n_rows):
            history = np.asarray(sequences.row(row), dtype=np.int64)
            if history.size == 0:
                continue
            samples.extend(_sliding_windows(history, width, step))
        return samples

    def _next_item_batch(
        self,
        samples: list[np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shift, tokenize and left-pad one batch of samples.

        Returns ``(inputs, labels)``, both ``(rows, width)`` token ids. A label
        of ``pad`` marks a position with nothing to predict.

        The shift is the reference's ``SASTrainDataset.__getitem__``::

            labels = seq[-max_len:]
            tokens = seq[:-1][-max_len:]

        left-padded separately. For a sample that fits the window, ``tokens`` is
        one shorter than ``labels``, so the sample's first interaction is
        labelled at a position whose input is padding: the model is trained to
        predict it from no history at all. A sample is the whole history only
        when the history is short enough. A longer one is cut into windows by
        :func:`_sliding_windows`, and each window's first interaction is
        labelled from padding in the same way, although its real predecessor
        ends the window before. So the pad-only state learns a prior over the
        items that *start a sample*: every history's first item and, on long
        histories, every window's first item, which is a mid-history
        interaction. That prior is what an empty history is answered with at
        prediction time, since the state there is the same pad-only state. Here
        the same alignment is written as a ``pad`` prepended to the inputs.

        Rows are padded to the batch's longest, where the reference pads every
        row to ``max_len``. The recurrence never carries a state out of
        padding, so a real position's output does not depend on how much
        padding precedes it, and packing only removes work.

        ``unk_dropout`` then replaces a fraction of the real *inputs* with
        ``unk``. Labels are never touched, so no target is lost, and padding is
        exempt so ``unk`` and ``pad`` stay distinct. At a rate of zero no random
        number is drawn, so the batch is exactly the one a config without the
        field would have produced from the same seed.
        """
        assert self.batcher is not None
        tokenizer = self.batcher.tokenizer
        pad_id = tokenizer.pad_id
        window = self.cfg.max_history_length
        unk_id = getattr(tokenizer, "unk_id", None)
        unk_dropout = self.cfg.unk_dropout if unk_id is not None else 0.0

        width = max(min(int(sample.size), window) for sample in samples)
        rows = len(samples)
        inputs = np.full((rows, width), pad_id, dtype=np.int64)
        labels = np.full((rows, width), pad_id, dtype=np.int64)

        for row, sample in enumerate(samples):
            encoded = tokenizer.encode_indices(sample)
            shifted = np.concatenate(([pad_id], encoded[:-1]))[-window:]
            target = encoded[-window:]
            length = int(target.size)
            inputs[row, width - length :] = shifted
            labels[row, width - length :] = target

        if unk_dropout > 0.0:
            real = inputs != pad_id
            inputs[real & (rng.random(inputs.shape) < unk_dropout)] = unk_id

        return (
            torch.from_numpy(inputs).to(self.device),
            torch.from_numpy(labels).to(self.device),
        )

    def _train_step(
        self,
        samples: list[np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[float, int] | None:
        """Optimize one batch against Eq. 1's next-item likelihood.

        The reference's ``LRUTrainer.calculate_loss`` is
        ``CrossEntropyLoss(ignore_index=0)`` over the logits at every position,
        which averages over the positions with a real label. This takes the
        logits at exactly those positions and averages over them, which is the
        same loss without forming a ``rows x width x vocabulary`` tensor for
        the padding. With ``n_negatives`` set, the logits are the sampled ones
        of :meth:`_sampled_softmax_loss` instead of the whole vocabulary's.

        Returns the mean loss and the number of labelled positions, so ``fit``
        can weight epochs by position count rather than by batch.
        """
        assert self.model is not None
        assert self.optimizer is not None
        if not samples:
            return None

        inputs, labels = self._next_item_batch(samples, rng)
        assert self.batcher is not None
        labelled = labels != self.batcher.tokenizer.pad_id
        n_positions = int(labelled.sum())
        if n_positions == 0:
            return None

        states = self.model(inputs)
        if self.cfg.n_negatives is None:
            logits = self.model.score(states[labelled])
            loss = F.cross_entropy(logits, labels[labelled])
        else:
            loss = self._sampled_softmax_loss(states[labelled], labels[labelled])

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.max_grad_norm
        )
        self.optimizer.step()
        return float(loss.detach()), n_positions

    def _sampled_softmax_loss(
        self, states: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """§5.3's sampled softmax, as the reference's ``LRUModel`` computes it.

        ``states`` and ``targets`` are the labelled positions only. Each is
        scored against ``n_negatives`` catalog tokens drawn uniformly with
        replacement, then its own target, which goes last, at index
        ``n_negatives``, as in the reference. Nothing is excluded from the
        draw -- see :class:`~.config.LRURecConfig` -- so a target drawn as its
        own negative appears twice among the logits with the same score, and
        the loss at that position cannot fall below ``log 2``, exactly as there.

        Drawn from torch's generator, which ``fit`` seeds from ``cfg.seed``,
        rather than from the NumPy generator that shuffles: the reference
        draws from torch too, and at XLong's 32 rows x 1000 positions x 100
        negatives a draw on the host would cross to the device every step.
        """
        assert self.model is not None
        n_negatives = int(self.cfg.n_negatives)
        first = self.model.n_reserved
        negatives = torch.randint(
            first,
            first + self.model.n_items,
            (targets.shape[0], n_negatives),
            device=targets.device,
        )
        candidates = torch.cat((negatives, targets.unsqueeze(1)), dim=1)
        logits = self.model.score_items(states, candidates)
        return F.cross_entropy(logits, torch.full_like(targets, n_negatives))

    # -- prediction ---------------------------------------------------------

    def predict_on_batch(
        self,
        source: ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Rank the catalog from the state at each history's final position.

        The reference's ``LRUTrainer.calculate_metrics`` reads ``[:, -1]`` of a
        left-padded window of the most recent ``max_len`` interactions; under
        left padding that is every row's newest interaction, which the
        batcher's ``gather_final`` returns.

        The ranking is over the whole catalog, as §5.1's is. By default every
        item in the *full* history is excluded, truncated part included, where
        the reference excludes only the items inside the window it encoded. The
        package's ``exclude_seen`` contract is the stricter of the two, and the
        same for every model in it.
        """
        if self.model is None or self.batcher is None or self._n_items is None:
            raise RuntimeError("LRURecTrainer must be fitted before predicting")
        if not isinstance(source, ItemSequences):
            raise TypeError(
                "LRURecTrainer predicts from ItemSequences, got "
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
                vals=torch.empty((0, k), dtype=torch.float32, device=self.device),
                shape=(0, n_items),
            )

        self.model.eval()
        with torch.no_grad():
            item_history, mask = self.batcher.encode(source, device=self.device)
            final = self.batcher.gather_final(self.model(item_history), mask)
            # The model scores the whole vocabulary; the catalog is the part
            # above the reserved ids, and nothing downstream of a model sees a
            # token id. Logits rather than probabilities rank identically.
            scores = self.model.score(final)[:, self._item_offset :]
            if exclude_seen:
                self._mask_seen(scores, source)
            candidates = torch.from_numpy(candidate_rows).long().to(self.device)
            vals, local_cols = torch.topk(scores[:, candidates], k, dim=1)
            cols = candidates[local_cols]

        return SRPTensor(cols=cols, vals=vals, shape=(rows, n_items))

    def score_candidates(
        self,
        source: ItemSequences,
        candidates: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Score each history against its own row of catalog items.

        ``candidates`` is ``(rows, n)`` catalog indices, one row per history in
        ``source``. The result is ``(rows, n)`` logits from the same final
        state :meth:`predict_on_batch` ranks from. This is what
        :func:`~compresso_recsys.evaluation.evaluate_sampled_recommender` asks
        of a model, and the reference's XLong evaluation computes the same
        scores: the final state against ``embedding_weight[all_items]``.

        Nothing is excluded: an item the user has seen is scored like any
        other, since the sampled protocol does not strike out history. An
        index outside the fitted catalog cannot be scored and gets ``-inf``,
        so it ranks below every catalog item, just as a full-catalog ranking
        could never contain it.
        """
        if self.model is None or self.batcher is None or self._n_items is None:
            raise RuntimeError("LRURecTrainer must be fitted before scoring")
        if not isinstance(source, ItemSequences):
            raise TypeError(
                "LRURecTrainer scores from ItemSequences, got "
                f"{type(source).__name__}"
            )
        columns = torch.as_tensor(candidates, device=self.device)
        if columns.ndim != 2 or columns.shape[0] != source.n_rows:
            raise ValueError(
                f"candidates must be ({source.n_rows}, n), one row per history, "
                f"got shape {tuple(columns.shape)}"
            )
        if columns.dtype.is_floating_point or columns.dtype == torch.bool:
            raise TypeError(
                f"candidates must be integer catalog indices, got {columns.dtype}"
            )
        columns = columns.long()
        if columns.numel() and int(columns.min()) < 0:
            raise ValueError("candidates must be non-negative catalog indices")
        if source.n_rows == 0:
            return torch.empty(columns.shape, dtype=torch.float32, device=self.device)

        scoreable = columns < self._n_items
        self.model.eval()
        with torch.no_grad():
            item_history, mask = self.batcher.encode(source, device=self.device)
            final = self.batcher.gather_final(self.model(item_history), mask)
            tokens = torch.where(scoreable, columns, 0) + self._item_offset
            scores = self.model.score_items(final, tokens)
        return scores.masked_fill(~scoreable, -torch.inf)

    @property
    def _item_offset(self) -> int:
        assert self.batcher is not None
        return int(self.batcher.tokenizer.n_reserved)

    def _mask_seen(self, scores: torch.Tensor, source: ItemSequences) -> None:
        """Forbid every item in the *full* history, truncated part included.

        Scores are indexed by catalog position, and a history may span a wider
        catalog than this model was fitted on -- a later split stage does
        exactly that. Items beyond the fitted catalog are dropped from the mask
        rather than clipped: they were never scoreable, so there is nothing to
        forbid.
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

    # -- construction -------------------------------------------------------

    def _build_model(self) -> LRURec:
        if self.batcher is None:
            raise RuntimeError("trainer batcher is unavailable")
        tokenizer = self.batcher.tokenizer
        model = LRURec(
            n_items=tokenizer.n_items,
            d_model=self.cfg.d_model,
            n_blocks=self.cfg.n_blocks,
            dropout=self.cfg.dropout,
            lru_dropout=self.cfg.lru_dropout,
            r_min=self.cfg.r_min,
            r_max=self.cfg.r_max,
            pad=tokenizer.pad_id,
            n_reserved=tokenizer.n_reserved,
        )
        self._init_weights(model)
        return model.to(self.device)

    def _init_weights(self, model: LRURec) -> None:
        """The reference's ``truncated_normal_init``, parameter for parameter.

        A normal of standard deviation ``initializer_range`` truncated at two
        deviations, applied to every parameter *except* the LayerNorms and the
        ring parameters ``params_log``, which keep their own initialization.
        "Every" is literal: the biases, the output bias and the pad row of the
        embedding are drawn from it too, where BERT4Rec zeroes all three. A
        complex parameter has its real and imaginary parts drawn independently,
        as the reference draws them.

        The reference skips by *name*, matching ``'layer_norm'`` and
        ``'params_log'``. This skips by module type instead, because the
        embedding's norm is not named ``layer_norm`` here; the set of
        parameters skipped is the same.
        """
        std = float(self.cfg.initializer_range)
        bound = 2.0 * std
        skipped = self._layer_norm_parameters(model)
        skipped |= {
            id(module.params_log)
            for module in model.modules()
            if isinstance(module, LRULayer)
        }
        with torch.no_grad():
            for parameter in model.parameters():
                if id(parameter) in skipped:
                    continue
                target = (
                    torch.view_as_real(parameter)
                    if parameter.is_complex()
                    else parameter
                )
                nn.init.trunc_normal_(target, std=std, a=-bound, b=bound)

    @staticmethod
    def _layer_norm_parameters(model: nn.Module) -> set[int]:
        return {
            id(parameter)
            for module in model.modules()
            if isinstance(module, nn.LayerNorm)
            for parameter in module.parameters()
        }

    def _build_checkpoint_optimizer(self) -> None:
        """AdamW with weight decay spared on biases and LayerNorms.

        The reference's ``_create_optimizer`` spares any parameter whose name
        contains ``'bias'`` or ``'layer_norm'`` and decays everything else.
        That includes ``params_log``: ``ν_log``, ``θ_log`` and ``γ_log`` are all
        decayed toward zero, which pulls every ``|λ|`` toward ``e^-1``. It is
        not obviously deliberate, but it is what produced the published
        numbers, so it is kept.

        As in :meth:`_init_weights`, LayerNorms are found by type rather than by
        name; biases by name, since every bias here ends in ``bias``.
        """
        if self.model is None:
            raise RuntimeError("LRURec model must be built before its optimizer")
        norms = self._layer_norm_parameters(self.model)
        decayed, spared = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in norms or name.endswith("bias"):
                spared.append(parameter)
            else:
                decayed.append(parameter)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decayed, "weight_decay": self.cfg.weight_decay},
                {"params": spared, "weight_decay": 0.0},
            ],
            lr=self.cfg.lr,
            betas=tuple(self.cfg.betas),
            eps=self.cfg.epsilon,
        )

    # -- persistence --------------------------------------------------------

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> LRURecTrainer:
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
        trainer = cls(LRURecConfig(**config), batcher)
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
            raise TypeError("LRURecTrainer checkpoints support ItemTokenizer only")
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
            raise TypeError("LRURec training history must be a list")
        self.history = list(history)
