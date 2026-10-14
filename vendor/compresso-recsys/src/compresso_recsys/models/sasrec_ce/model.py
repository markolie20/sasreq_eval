# Modified for seqrec_eval: differs from upstream compresso-recsys; see vendor/compresso-recsys/VENDORED.md
"""The SASRecCE network: SASRec's, copied rather than imported.

A copy so that :class:`~compresso_recsys.models.SASRec` stays exactly what it is
while this one is free to change. Today the two are the same architecture
module for module, and a ``state_dict`` from one loads into the other.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["LAYER_NORM_EPS", "PointWiseFeedForward", "SASRecCE"]

# The published architecture's epsilon, against PyTorch's 1e-5 default.
# Nothing in the paper or its follow-ups tunes it.
LAYER_NORM_EPS = 1e-8


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout2(
            self.linear2(self.relu(self.dropout1(self.linear1(inputs))))
        )


class SASRecCE(nn.Module):
    """Item and position embeddings, causal blocks, and a tied dot-product score.

    The same modernized pre-norm SASRec as
    :class:`~compresso_recsys.models.SASRec`: the normalized residual stream
    supplies queries, keys and values, :class:`~torch.nn.MultiheadAttention`
    applies its output projection, and the result is added to the unnormalized
    stream. What differs is how :class:`SASRecCETrainer` trains it, not what it
    computes.

    There is no output head. A candidate is scored by the dot product of a state
    with that candidate's *input* embedding, so the tie is structural rather
    than an option.

    :meth:`forward` returns states; :meth:`score` and :meth:`score_items` turn
    states into scores. Training and prediction both use :meth:`score` -- the
    whole catalog -- but on different rows: training gathers the labelled
    positions first, prediction the final one, so neither materialises
    ``rows x length x n_items``. :meth:`score_items` scores chosen items only.

    The embedding table holds ``n_reserved + n_items`` rows, reserved ids first,
    so catalog item ``i`` lives at row ``i + n_reserved`` --
    :class:`~compresso_recsys.models.ItemTokenizer`'s layout. A ``mask`` token,
    when the tokenizer names one, is a reserved row like ``unk``: it is embedded
    and trained, and it is never scored.

    **Padding is on the left**, so the newest interaction always lands in the
    final column and position *n* means "n from the end" for every history
    length alike. The pad steps then precede the real ones and sit inside every
    causal window, so :meth:`forward` masks them out of attention explicitly.
    """

    def __init__(
        self,
        *,
        n_items: int,
        n_reserved: int,
        max_history_length: int,
        pad_id: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_items < 1:
            raise ValueError(f"n_items must be >= 1, got {n_items}")
        if n_reserved < 1:
            raise ValueError(
                f"n_reserved must be >= 1, got {n_reserved}: padding alone needs "
                "an id below the catalog"
            )
        if max_history_length < 1:
            raise ValueError(
                f"max_history_length must be >= 1, got {max_history_length}"
            )
        if not 0 <= pad_id < n_reserved:
            raise ValueError(
                f"pad_id must be one of the {n_reserved} reserved ids, got {pad_id}"
            )
        self.n_items = int(n_items)
        self.n_reserved = int(n_reserved)
        self.max_history_length = int(max_history_length)
        self.pad_id = int(pad_id)
        self.n_heads = int(n_heads)

        self.item_embedding = nn.Embedding(
            self.n_reserved + self.n_items, d_model, padding_idx=pad_id
        )
        # Positions are numbered from one so index 0 stays reserved for padding
        # steps, hence the extra row.
        self.position_embedding = nn.Embedding(
            max_history_length + 1, d_model, padding_idx=0
        )
        self.embedding_dropout = nn.Dropout(p=dropout)

        self.attention_norms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_norms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(n_blocks):
            self.attention_norms.append(nn.LayerNorm(d_model, eps=LAYER_NORM_EPS))
            self.attention_layers.append(
                nn.MultiheadAttention(
                    d_model,
                    n_heads,
                    dropout=dropout,
                    batch_first=True,  # keeps (batch, steps, d_model) throughout
                )
            )
            self.forward_norms.append(nn.LayerNorm(d_model, eps=LAYER_NORM_EPS))
            self.forward_layers.append(PointWiseFeedForward(d_model, dropout))

        self.last_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier normal on every matrix, as the reference implementation does."""
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_normal_(parameter)

        # nn.Embedding zeroes padding_idx at construction and the loop above
        # overwrote it. Re-zero explicitly: padding_idx keeps the gradient zero,
        # so whatever sits there at the start stays there for good.
        with torch.no_grad():
            self.item_embedding.weight[self.pad_id].fill_(0)
            self.position_embedding.weight[0].fill_(0)

    def forward(self, item_history: torch.Tensor) -> torch.Tensor:
        """States for every step, shape ``(rows, length, d_model)``.

        ``item_history`` is ``(rows, length)`` of embedding-row ids, left
        padded. ``states[:, i]`` has read ``item_history[:, :i + 1]``, so it is
        the state from which ``item_history[:, i + 1]`` should be predicted.
        """
        if item_history.ndim != 2:
            raise ValueError(
                "item_history must be (rows, length), got "
                f"{tuple(item_history.shape)}"
            )
        n_steps = item_history.shape[1]
        if n_steps > self.max_history_length:
            raise ValueError(
                f"a history of {n_steps} items needs {n_steps} positions, but "
                f"this model was built for {self.max_history_length}"
            )

        real_mask = item_history != self.pad_id
        hidden = self.item_embedding(item_history)
        # Xavier gives the embedding a fan-based scale rather than the unit-ish
        # one the norms downstream expect, and the reference rescales here to
        # compensate. Paired with _init_weights.
        hidden = hidden * (self.item_embedding.embedding_dim**0.5)

        # Padding steps take position 0, whose row is pinned to zero. Causal
        # attention already keeps them out of every real state, so this only
        # stops a pad row's own state from drifting into something readable.
        positions = torch.arange(1, n_steps + 1, device=item_history.device) * real_mask
        hidden = self.embedding_dropout(hidden + self.position_embedding(positions))

        # True marks a pair that may not attend: step i reads 0..i, nothing
        # later, and never a padding step. The padding half is what left padding
        # makes necessary -- pad steps precede the real ones, so causal masking
        # alone would let every real step read them.
        causal = torch.triu(
            torch.ones(
                (n_steps, n_steps),
                dtype=torch.bool,
                device=item_history.device,
            ),
            diagonal=1,
        )
        blocked = causal.unsqueeze(0) | ~real_mask.unsqueeze(1)
        # A pad step's own causal window is all padding, and a row masked
        # everywhere softmaxes over nothing and returns NaN, which the residual
        # would then spread to the whole row. Letting every step read itself
        # costs nothing: a pad step's output is discarded either way.
        blocked = blocked & ~torch.eye(
            n_steps, dtype=torch.bool, device=item_history.device
        )
        causal_mask = blocked.repeat_interleave(self.n_heads, dim=0)

        for attention_norm, attention, forward_norm, feed_forward in zip(
            self.attention_norms,
            self.attention_layers,
            self.forward_norms,
            self.forward_layers,
        ):
            normed = attention_norm(hidden)
            # need_weights=False keeps the fused attention kernel. The averaged
            # weights it would otherwise build are discarded here.
            attended, _ = attention(
                normed, normed, normed, attn_mask=causal_mask, need_weights=False
            )
            hidden = hidden + attended
            hidden = hidden + feed_forward(forward_norm(hidden))

        return self.last_norm(hidden)

    def score(self, states: torch.Tensor) -> torch.Tensor:
        """Catalog scores for the given states, one per item.

        The weight is a *slice* of the embedding rather than its own parameter.
        The reserved rows -- padding, ``mask`` and ``unk`` when the tokenizer
        names them -- sit below ``n_reserved`` and so stay out of the scores:
        none of them is ever a recommendation, and none of them is a class the
        cross entropy normalises over.
        """
        return F.linear(states, self.item_embedding.weight[self.n_reserved :])

    def score_items(
        self, states: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        """Score each state against specific items.

        ``items`` holds embedding-row ids: ``(rows, length)`` to score one item
        per step, or ``(rows, length, n)`` for ``n`` of them, and the result
        carries the shape of ``items``. Training does not use it -- the full
        softmax is :meth:`score` -- but a sampled evaluation scores chosen items
        this way rather than paying for the whole catalog.
        """
        embeddings = self.item_embedding(items)
        if embeddings.ndim == states.ndim + 1:
            states = states.unsqueeze(-2)
        return (embeddings * states).sum(dim=-1)
