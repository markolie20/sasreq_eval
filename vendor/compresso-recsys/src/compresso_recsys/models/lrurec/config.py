# Modified for seqrec_eval: differs from upstream compresso-recsys; see vendor/compresso-recsys/VENDORED.md
"""Configuration for :class:`~compresso_recsys.models.LRURec`.

Every default here is the published setting. The paper states few of them --
the architecture's shape in §3, the optimizer and learning rate in §5.1 -- and
defers the rest to "the original implementation", so most numbers come from the
authors' PyTorch code at https://github.com/yueqirex/LRURec. Each field says
which of the two it came from.

"The reference" means what ``train.py`` actually runs: the ``argparse`` defaults
in ``config.py`` *after* ``set_template`` has overwritten them. The two disagree
on several fields -- ``set_template`` switches the learning-rate schedule and its
warmup off, and fixes the batch size at 128 against the flag default of 64 --
and a citation to the flag default would be a citation to a setting no
published run used. Where the reference differs from the paper the reference
wins, because it is what produced the paper's tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from compresso_recsys._reporting import _validate_log_every_n_steps

__all__ = ["LRURecConfig"]

OptimizerName = Literal["AdamW"]


@dataclass(frozen=True)
class LRURecConfig:
    """Architecture, sample construction, and optimization settings for LRURec.

    **Architecture.** ``d_model`` is §5's ``H = 64`` and the reference's
    ``bert_hidden_units``; ``n_blocks`` is the two LRU blocks §3.4 stacks
    "following [16, 31]". The recurrence's own width is ``2 * d_model`` and is
    not a field: §3.2 says "we double the size of H in Λ", and the reference's
    ``LRULayer`` hard-codes ``hidden_size = 2 * d_model``. The feed-forward
    inner width is ``4 * d_model`` for the same kind of reason -- Eq. 8 fixes
    it.

    ``dropout`` and ``lru_dropout`` are the reference's ``bert_dropout`` and
    ``bert_attn_dropout``, both 0.2 after ``set_template``. The paper names one
    dropout rate and a grid of [0.2, 0.4, 0.6, 0.8] for it (§5.1), but the
    reference exposes two: ``dropout`` covers the embedding and both
    feed-forward sites, ``lru_dropout`` the LRU layer's output alone. The
    BERT-derived names are the reference's inheritance from the codebase it
    grew out of; nothing here attends.

    ``max_history_length`` is the reference's ``bert_max_len``: 200 for ML-1M,
    1000 for XLong and 50 for every other dataset (§5.1). 200 is the default
    because the package's reproductions start from ML-1M. Nothing in the model
    is sized by it -- the recurrence has no positional table -- so it is purely
    how much history a sample and a prediction read.

    ``r_min`` and ``r_max`` bound the ring the eigenvalues are drawn from at
    initialization, ``|λ| ∈ [0.8, 0.99]`` (§3.2, the reference's ``LRULayer``
    defaults). The phases are drawn uniformly from ``[0, 2π)``.

    ``initializer_range`` is the standard deviation of the truncated normal
    that initializes every weight except the LayerNorms and the ring
    parameters. The paper cites BERT4Rec's scheme; the reference's
    ``truncated_normal_init`` uses ``std=0.02`` truncated at ``±0.04``, two
    standard deviations, which is how ``initializer_range`` is read here.

    **Samples.** ``sliding_window_step`` is the reference's
    ``sliding_window_size``, a fraction of ``max_history_length``. It is never
    overridden by ``set_template``, so the flag default of 1.0 is the published
    setting: windows of ``max_history_length`` laid end to end, without
    overlap. See :meth:`~.trainer.LRURecTrainer._training_samples` for exactly
    when a history is split.

    ``unk_dropout`` replaces that fraction of the real *input* positions with
    the tokenizer's ``unk`` token, teaching the model to read a history
    containing an item it cannot identify. It defaults to zero for paper
    parity, as SASRec's and BERT4Rec's do. Set it above zero when otherwise
    ``unk`` would never be trained: the training vocabulary *is* the training
    window, so an out-of-catalog item cannot occur until evaluation. The right
    rate tracks the out-of-catalog share the split will actually produce --
    near zero under ``leave_last_out``, far higher on a late ``temporal``
    stage. Labels are never replaced, so no target is lost. It is ignored when
    the tokenizer has no ``unk`` to substitute.

    **Objective.** ``n_negatives`` chooses the softmax. ``None``, the default,
    is the full softmax over the vocabulary that §5.1 trains every model with,
    and that the reference uses on every dataset but XLong. An integer is the
    sampled softmax §5.3 introduces for XLong, "randomly sampling 100 negative
    items to compute loss", which the reference's ``negative_sample_size`` sets
    to 100. Each labelled position then scores its true next item against
    ``n_negatives`` items drawn uniformly from the catalog with replacement,
    and the cross entropy runs over those ``n_negatives + 1`` logits alone,
    output bias included. Nothing is excluded from the draw: not the user's
    history, and not the position's own target, which turns up among its own
    negatives with probability about ``n_negatives / n_items``. That is the
    reference's ``torch.randint(1, num_items + 1)`` exactly. SASRec's sampler
    in this package excludes the history, so the two are not interchangeable.
    Prediction is unaffected either way: it always scores the whole catalog.

    **XLong.** §5.3's long-sequence experiment changes four settings from the
    main tables, and the reference applies all four when ``dataset_code`` is
    ``'xlong'``. Training uses ``max_history_length=1000``, ``batch_size=32``
    and ``n_negatives=100``. Evaluation ranks each held-out item against 10,000
    uniformly sampled items rather than the catalog, which is
    :func:`~compresso_recsys.evaluation.evaluate_sampled_recommender` with
    ``n_negatives=10_000``. The data is
    :class:`~compresso_recsys.datasets.XLong`, split by the builder's ``xlong``
    spec. §5.3 also narrows the search to weight decay in [0, 1e-2] and dropout
    in [0.2, 0.4], so the defaults of 0.01 and 0.2 are both on its grid.

    **Optimization.** §5.1: AdamW at "the static learning rate of 1e-3", batch
    size 128. ``set_template`` agrees on all three and is the source of the
    rest: ``weight_decay`` 0.01 (from the paper's grid of [0, 1e-6, 1e-4,
    1e-2]), ``epsilon`` 1e-9 against PyTorch's 1e-8, gradients clipped at an l2
    norm of 5, and PyTorch's default ``betas``. "Static" is literal: the
    reference sets ``enable_lr_schedule = False``, so there is no warmup and no
    decay, and therefore no schedule fields here.

    ``epochs`` is the reference's ``num_epochs``, 500, which the paper reports
    as the maximum. The reference stops well before it, on validation
    Recall@10 with a patience of ten validation rounds -- §5.5 reports ML-1M
    stopping at about 23k steps. Early stopping is not part of this package's
    contract, so here 500 is a fixed budget; select ``epochs`` on validation
    data, the way the reference's stopping rule effectively does.
    """

    # -- architecture -------------------------------------------------------
    d_model: int = 64
    n_blocks: int = 2
    dropout: float = 0.2
    lru_dropout: float = 0.2
    max_history_length: int = 200
    r_min: float = 0.8
    r_max: float = 0.99
    initializer_range: float = 0.02

    # -- samples ------------------------------------------------------------
    sliding_window_step: float = 1.0
    unk_dropout: float = 0.0

    # -- objective ----------------------------------------------------------
    n_negatives: int | None = None

    # -- optimization -------------------------------------------------------
    batch_size: int = 128
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-9
    max_grad_norm: float = 5.0
    optimizer: OptimizerName = "AdamW"

    # -- runtime ------------------------------------------------------------
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    log_prefix: str = "LRURec"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.lru_dropout < 1.0:
            raise ValueError(
                f"lru_dropout must be in [0, 1), got {self.lru_dropout}"
            )
        if self.max_history_length < 1:
            raise ValueError(
                "max_history_length must be >= 1, got "
                f"{self.max_history_length}"
            )
        # The ring is sampled as |λ|² ~ U[r_min², r_max²], and every |λ| must
        # stay strictly inside the unit circle or the recurrence diverges and
        # the normalization factor sqrt(1 - |λ|²) reaches zero.
        if not 0.0 < self.r_min <= self.r_max < 1.0:
            raise ValueError(
                "the eigenvalue ring needs 0 < r_min <= r_max < 1, got "
                f"r_min={self.r_min}, r_max={self.r_max}"
            )
        if self.initializer_range <= 0.0:
            raise ValueError(
                f"initializer_range must be > 0, got {self.initializer_range}"
            )
        if not 0.0 < self.sliding_window_step <= 1.0:
            raise ValueError(
                "sliding_window_step is a fraction of max_history_length and "
                f"must be in (0, 1], got {self.sliding_window_step}"
            )
        if int(self.sliding_window_step * self.max_history_length) < 1:
            # The reference asserts the same thing: a step that rounds to zero
            # would never move the window.
            raise ValueError(
                "sliding_window_step * max_history_length must be at least one "
                f"interaction, got {self.sliding_window_step} * "
                f"{self.max_history_length}"
            )
        if not 0.0 <= self.unk_dropout < 1.0:
            raise ValueError(
                f"unk_dropout must be in [0, 1), got {self.unk_dropout}"
            )
        if self.n_negatives is not None and (
            isinstance(self.n_negatives, bool)
            or not isinstance(self.n_negatives, int)
            or self.n_negatives < 1
        ):
            raise ValueError(
                "n_negatives must be None for the full softmax or an integer "
                f">= 1 for the sampled one, got {self.n_negatives!r}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        # asdict writes a JSON array and reading it back gives a list, so a
        # reloaded config would otherwise carry a different type than a fresh
        # one and compare unequal to it. Frozen, hence object.__setattr__.
        object.__setattr__(self, "betas", tuple(self.betas))
        if len(self.betas) != 2:
            raise ValueError(f"betas must be two values, got {self.betas!r}")
        if not all(0.0 <= beta < 1.0 for beta in self.betas):
            raise ValueError(f"betas must each be in [0, 1), got {self.betas!r}")
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.max_grad_norm <= 0.0:
            raise ValueError(
                f"max_grad_norm must be > 0, got {self.max_grad_norm}"
            )
        if self.optimizer != "AdamW":
            raise ValueError(
                "LRURec's published optimizer is AdamW (§5.1); got "
                f"{self.optimizer!r}"
            )
