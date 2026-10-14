# Modified for seqrec_eval: differs from upstream compresso-recsys; see vendor/compresso-recsys/VENDORED.md
"""Configuration for :class:`~compresso_recsys.models.SASRecCE`.

The architecture and optimization fields are :class:`~compresso_recsys.models.
SASRecConfig`'s, with the same defaults, so that a comparison between the two
trainers moves the objective and the augmentation and nothing else. What is new
is grouped under *objective* and *augmentation* below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from compresso_recsys._reporting import _validate_log_every_n_steps

__all__ = ["SASRecCEConfig"]

OptimizerName = Literal["Adam"]


@dataclass(frozen=True)
class SASRecCEConfig:
    """Configuration for :class:`SASRecCE`.

    **Architecture.** ``d_model``, ``n_blocks``, ``n_heads``, ``dropout`` and
    ``max_history_length`` are :class:`~compresso_recsys.models.SASRecConfig`'s,
    read the same way: one width for the whole residual stream, one dropout
    rate, and a context window that this field owns and that sizes the
    positional table. A batcher passed with ``max_length=None`` inherits it,
    and one stating a different window is refused.

    **Objective.** Cross entropy, over the whole catalog or over a sample of
    it, chosen by ``n_negatives``.

    ``None``, the default, is the full softmax: every labelled position is
    scored against the whole catalog, so every other item is a negative at
    every position -- the change Klenitskiy and Vasilev show closes most of the
    gap between SASRec and BERT4Rec. It costs a ``positions x n_items`` logit
    tensor per step; at the default batch and window that is about 25,600
    rows of logits.

    An integer is a sampled softmax: each position is scored against its
    target and ``n_negatives`` items drawn uniformly, with replacement, and
    the cross entropy runs over those ``n_negatives + 1`` logits. The draw is
    made **once per user**, and every position of that user shares it, so a
    batch touches up to ``rows x n_negatives`` distinct items for one draw of
    ``rows x n_negatives`` numbers. A negative that equals the position's own
    target is dropped, so the target never competes with itself. Uniform draws need no
    log-probability correction -- it would be the same constant for every
    candidate -- but they are mostly easy, tail items, so the objective
    approaches the full softmax only as ``n_negatives`` grows. Klenitskiy and
    Vasilev report that sampled training needs far more than one negative;
    think hundreds to a few thousand, against a catalog where the full
    softmax is too expensive. Prediction always scores the whole catalog.

    ``allow_history_negatives`` decides whether the user's other interactions
    are negatives at all. ``False``, the default, keeps every item in the
    row's *full* history -- the truncated part included, and later
    interactions as much as earlier ones -- out of each position's softmax,
    except that position's own target: under the full softmax by setting
    their logits to ``-inf``, under the sampled one by drawing from the
    catalog minus the history, SASRec's ``I \\ S_u``. An item the user did
    engage with is not evidence of what they would not, so it is neither
    rewarded nor penalised. ``True`` treats such an item like any other. The
    history is the user's observed sequence, never an augmented one.

    **Augmentation.** Redrawn for every batch, so each epoch sees a different
    corruption of every history and a run of ``epochs`` reads that many
    variants of each. Two stages, decided by what an augmentation may touch;
    :mod:`compresso_recsys.models.sasrec_ce.augmentation` lays them out.

    *Input corruption* rewrites what the model reads, after the next-item
    shift, so no target is lost and padding is never touched. Three rates
    share it, drawn in this order over the positions the earlier ones left
    alone, so they never compete for a position:

    ``mask_probability`` replaces that fraction of the real inputs with the
    tokenizer's ``mask`` token, teaching the model to predict the next item
    from a history with gaps. It needs a ``mask`` special token, which the
    default vocabulary has; a passed batcher whose tokenizer has none is
    refused while the rate is above zero. The default of 0.2 is a starting
    point rather than a published setting.

    ``cooccurrence_replace_probability`` replaces that fraction of the inputs
    with an item that co-occurs with them, while the target stays the true
    next item: "a similar item was here, predict what followed anyway". Inputs
    whose item co-occurs with nothing are left alone.

    ``unk_dropout`` is SASRec's: that fraction becomes ``unk``, which is what
    trains the unknown-item embedding for evaluation histories containing
    items training never saw. It is ignored when the tokenizer names no
    ``unk``. Masking and ``unk`` look alike but teach different things:
    ``unk`` is a token the model will read at prediction time, ``mask`` is
    one it never will.

    *Sequence edits* rewrite the history before the shift.
    ``cooccurrence_insert_probability`` inserts, after each interaction but
    the last, an item that co-occurs with it, at that rate. An inserted item
    is read by the step after it. Whether it is also a target -- predicted by
    the step it follows -- is ``score_inserted_items``: ``False``, the
    default, leaves that step out of the loss, so insertion adds context but
    no invented lesson; ``True`` trains on it like an observed interaction,
    which is what lets ``cooccurrence_popularity_exponent`` move predictions
    and not only the inputs they are made from.

    **Co-occurrence.** Two items co-occur when they sit within
    ``cooccurrence_window`` steps of each other in a training history; the
    counts are symmetric, a repeated item is not its own partner, and nothing
    is counted across users. Each item keeps its ``cooccurrence_top_k``
    heaviest partners, and a partner ``j`` of ``a`` is drawn with probability
    proportional to ``count(a, j) * popularity(j) ** exponent``, popularity
    being ``j``'s number of training interactions.

    ``cooccurrence_popularity_exponent`` is that exponent, and its sign is the
    direction. Zero draws partners by co-occurrence alone, which already
    favours popular items, since they co-occur with everything. Above zero
    boosts popular partners further. Below zero boosts the tail; at -1 the
    weight is ``count(a, j) / popularity(j)``, proportional to ``P(a | j)``,
    which cancels the partner's popularity entirely, and further below that
    favours rare items outright -- including ones that co-occurred with ``a``
    once, so expect noise there. It also decides which partners survive
    ``cooccurrence_top_k``, so the tail it boosts is the tail kept.

    The table is built once per ``fit`` from the training sequences, and only
    when one of the two co-occurrence rates is above zero. Both default to
    zero.

    **Optimization.** SASRec's: Adam at a constant ``lr``, with ``betas``
    applied through :meth:`optimizer_kwargs` because they are Adam's own. The
    second moment stays at SASRec's 0.98 so that only the objective differs
    between the two models; the reason SASRec shortens it -- a single sampled
    negative makes the gradient noisy -- is weaker under a full softmax, so it
    is a reasonable thing to tune.

    **Memory.** ``loss_chunk_elements`` bounds what the loss holds at once.
    The full softmax scores every labelled position against every item, so a
    batch holds ``positions x n_items`` logits -- 102,400 positions at batch
    256 and window 400, times 150,000 items, is 61 GB before the softmax and
    its gradient. The sampled one gathers ``positions x n_negatives x
    d_model`` embeddings. The labelled positions are therefore scored in
    chunks of at most ``loss_chunk_elements // n_items`` positions (full) or
    ``loss_chunk_elements // (n_negatives * d_model)`` (sampled), and their
    gradients added up before the encoder's backward pass, which runs once.
    The loss is a mean over positions, so this is the same loss and the same
    gradient up to the order of floating-point additions, and the negatives
    and every augmentation are drawn exactly as they would be unchunked.

    ``prediction_chunk_elements`` does the same for prediction, where each
    history holds an ``n_heads x window x window`` attention map per block
    and ``n_items`` scores: a batch is scored in parts of at most that many
    elements of the larger, and the parts rank exactly as the whole batch
    would.

    Both change memory only, so neither is a hyperparameter. Their defaults
    keep the corner of the suite's search space -- 150,000 items, window 400,
    batch 256, d 128, 3 blocks, 2 heads, the history excluded -- inside a
    6 GB card: a training step peaks at 3.8 GB on an RTX 4050 at
    ``2**26`` where ``2**27`` reached 4.6 GB.
    """

    # -- architecture -------------------------------------------------------
    d_model: int = 50
    n_blocks: int = 2
    n_heads: int = 1
    dropout: float = 0.2
    max_history_length: int = 200

    # -- objective ----------------------------------------------------------
    n_negatives: int | None = None
    allow_history_negatives: bool = False

    # -- augmentation -------------------------------------------------------
    mask_probability: float = 0.2
    cooccurrence_replace_probability: float = 0.0
    unk_dropout: float = 0.0
    cooccurrence_insert_probability: float = 0.0
    score_inserted_items: bool = False

    # -- co-occurrence ------------------------------------------------------
    cooccurrence_window: int = 3
    cooccurrence_top_k: int = 20
    cooccurrence_popularity_exponent: float = 0.0

    # -- optimization -------------------------------------------------------
    batch_size: int = 128
    epochs: int = 201
    lr: float = 0.001
    optimizer: OptimizerName = "Adam"
    betas: tuple[float, float] = (0.9, 0.98)

    # -- runtime ------------------------------------------------------------
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    log_prefix: str = "SASRecCE"
    log_every_n_steps: int = 1000
    loss_chunk_elements: int = 2**26
    prediction_chunk_elements: int = 2**26

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        for name in ("loss_chunk_elements", "prediction_chunk_elements"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in (
            "d_model",
            "n_blocks",
            "n_heads",
            "max_history_length",
            "batch_size",
            "cooccurrence_window",
            "cooccurrence_top_k",
        ):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model must be divisible by n_heads, got d_model={self.d_model} "
                f"and n_heads={self.n_heads}"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.n_negatives is not None and (
            isinstance(self.n_negatives, bool)
            or not isinstance(self.n_negatives, int)
            or self.n_negatives < 1
        ):
            raise ValueError(
                "n_negatives must be None for the full softmax or an integer "
                f">= 1 for the sampled one, got {self.n_negatives!r}"
            )
        for name in ("allow_history_negatives", "score_inserted_items"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool, got {value!r}")
        for name in (
            "mask_probability",
            "cooccurrence_replace_probability",
            "unk_dropout",
            "cooccurrence_insert_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")
        if not math.isfinite(self.cooccurrence_popularity_exponent):
            raise ValueError(
                "cooccurrence_popularity_exponent must be finite, got "
                f"{self.cooccurrence_popularity_exponent}"
            )
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.optimizer != "Adam":
            raise ValueError(
                f"optimizer must be 'Adam', got {self.optimizer!r}"
            )
        # asdict writes a JSON array and reading it back gives a list, so a
        # reloaded config would otherwise carry a different type than a fresh
        # one and compare unequal to it. Frozen, hence object.__setattr__.
        object.__setattr__(self, "betas", tuple(self.betas))
        if len(self.betas) != 2:
            raise ValueError(f"betas must be two values, got {self.betas!r}")
        if not all(0.0 <= beta < 1.0 for beta in self.betas):
            raise ValueError(
                f"betas must each be in [0, 1), got {self.betas!r}"
            )

    @property
    def uses_cooccurrence(self) -> bool:
        """Whether any augmentation draws from the co-occurrence table."""
        return (
            self.cooccurrence_insert_probability > 0.0
            or self.cooccurrence_replace_probability > 0.0
        )

    def optimizer_kwargs(self) -> dict[str, object]:
        """Optimizer arguments beyond the parameters and ``lr``.

        ``betas`` is Adam's own hyperparameter rather than a universal one, so
        it is selected by :attr:`optimizer` here instead of being handed to
        whatever ``torch.optim`` class the name resolves to.
        """
        if self.optimizer == "Adam":
            return {"betas": self.betas}
        return {}
