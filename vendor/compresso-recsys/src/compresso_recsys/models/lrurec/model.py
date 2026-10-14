# Modified for seqrec_eval: differs from upstream compresso-recsys; see vendor/compresso-recsys/VENDORED.md
"""The LRURec architecture: §3 of Yue et al., WSDM 2024.

An item embedding, a stack of LRU blocks -- a diagonal complex linear recurrence
followed by a position-wise feed-forward network -- and a prediction layer tied
to the embedding. The objective that trains it and the batching that feeds it
live in :mod:`.trainer`; this module is the network and nothing else.

Where the paper's equations and the authors' code disagree, this follows the
code (https://github.com/yueqirex/LRURec, ``model/lru.py``), which is what
produced the published tables. Each disagreement is noted where it is decided.
There are four:

* the embedding is dropped out *before* its LayerNorm (Eq. 7 has no dropout);
* ``B`` and ``C`` carry biases (Eq. 5 has none);
* the LRU output is dropped out before the residual (Eq. 5 adds it directly);
* the feed-forward network has no outer GELU (Eq. 8 wraps one around ``W2``).

The output is *logits* over the vocabulary, not probabilities, for the reason
BERT4Rec's is: ``cross_entropy`` evaluates the softmax and its log together,
stably.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["LRUBlock", "LRULayer", "LRURec", "PositionWiseFeedForward"]

# The reference builds every LayerNorm with PyTorch's default epsilon, so this
# is that default written down rather than a choice. BERT4Rec and SASRec carry
# their own references' values; the number belongs to the implementation it
# was published with.
LAYER_NORM_EPS = 1e-5


def _next_power_of_two(length: int) -> int:
    """``2 ** ceil(log2(length))``, the reference's padded width, exactly.

    In integers rather than through ``np.log2``, which the reference uses, so a
    width that is already a power of two cannot round up to the next one.
    """
    return 1 << max(0, length - 1).bit_length()


class LRULayer(nn.Module):
    """Eq. 5 followed by its LayerNorm: §3.4's ``LRUNorm``.

    ``h_k = Λ h_{k-1} + γ ⊙ B x_k`` and ``y_k = Re(C h_k) + x_k``, with ``Λ``
    diagonal and complex, so the recurrence is one element-wise multiply per
    state dimension rather than a matrix product. The state is ``2 * d_model``
    wide -- §3.2's "we double the size of H in Λ".

    ``params_log`` stacks ``ν_log``, ``θ_log`` and ``γ_log`` as rows of one
    ``(3, 2 * d_model)`` parameter, as the reference does. That layout is
    load-bearing: the reference excludes parameters from weight decay *by name*,
    and ``params_log`` matches neither of its exclusions, so all three rows are
    decayed toward zero -- which pulls ``|λ| = exp(-exp(ν_log))`` toward
    ``e^-1``. Split into three vectors, a rule that spares one-dimensional
    parameters would silently stop decaying them.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.2,
        r_min: float = 0.8,
        r_max: float = 0.99,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.state_size = 2 * d_model

        # The ring initialization of Orvieto et al., which §3.2 adopts: |λ|²
        # uniform on [r_min², r_max²], the phase uniform on [0, 2π), and γ the
        # factor that keeps a unit-variance input at unit variance through the
        # recurrence. Drawn from torch's global generator, as the reference
        # draws it, so torch.manual_seed reproduces it.
        u1 = torch.rand(self.state_size)
        u2 = torch.rand(self.state_size)
        nu_log = torch.log(
            -0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2)
        )
        theta_log = torch.log(u2 * math.pi * 2)
        diag_lambda = torch.exp(
            torch.complex(-torch.exp(nu_log), torch.exp(theta_log))
        )
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # B and C. Eq. 5 writes them without biases; the reference's LRULayer
        # passes use_bias=True to both. Built complex at construction rather
        # than by the reference's ``nn.Linear(...).to(torch.cfloat)``, which
        # warns that complex modules are experimental; the values are replaced
        # by the trainer's initialization either way.
        self.in_proj = nn.Linear(d_model, self.state_size, dtype=torch.cfloat)
        self.out_proj = nn.Linear(self.state_size, d_model, dtype=torch.cfloat)

        # Eq. 5 adds the residual to Re(C h) directly; the reference drops the
        # projection out first, at bert_attn_dropout.
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)

    def eigenvalues(self) -> torch.Tensor:
        """``Λ``'s diagonal, ``exp(-exp(ν_log) + j exp(θ_log))``."""
        nu, theta, _ = torch.exp(self.params_log)
        return torch.exp(torch.complex(-nu, theta))

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the recurrence over a left-padded, power-of-two-wide batch.

        ``mask`` is true at real positions. The scan relies on padding being a
        *prefix* of each row, which is what the reference's left padding
        guarantees; see :meth:`_parallel_scan`.
        """
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        # γ scales the whole of B x + b. Eq. 5 has no bias to scale; the
        # reference multiplies after in_proj, so its bias is normalized too.
        driven = self.in_proj(states.to(torch.cfloat)) * gamma
        hidden = self._parallel_scan(driven, lamb, mask)
        # D is the identity (§3.2), which makes the input a residual.
        output = self.dropout(self.out_proj(hidden).real) + states
        return self.layer_norm(output)

    @staticmethod
    def _parallel_scan(
        driven: torch.Tensor, lamb: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """§3.3's recursive parallelization, as Algorithm 1 and the reference.

        At level ``i`` the sequence is cut into blocks of ``2^i``, and each
        block's second half receives its first half's final state, multiplied
        by ``[Λ, Λ², …, Λ^(2^(i-1))]``. After ``log2(L)`` levels every position
        holds ``Σ_s Λ^(t-s) γ B x_s``. The width must already be a power of two.

        Algorithm 1 has no mask; the reference adds one, and it is what makes
        left padding harmless. A carry is dropped when the position it comes
        from is padding. Padding is a prefix, so a block whose first half ends
        on padding is all padding up to that point, and dropping its carry is
        exactly starting the recurrence at the first real interaction from
        ``h_0 = 0``. That makes the first state ``γ ⊙ (B x_1 + b)``, the first
        driven input itself. The paper writes ``h_1 = B x_1``, carried over
        from its bias-free Eq. 3, without the ``γ`` its own Eq. 5 applies at
        every other step. A carry from a real position is kept, and by
        induction over the levels it already excludes every pad before it. So
        a real position's state does not depend on how much padding precedes
        it, which is what lets the trainer pack a batch to its longest row.

        Padded positions get a state too, built from their own input and
        nothing before it. Nothing real ever reads one.
        """
        rows, width, size = driven.shape
        hidden = driven
        for level in range(1, int(math.log2(width)) + 1):
            span = 2**level
            hidden = hidden.reshape(rows * width // span, span, size)
            block_mask = mask.reshape(rows * width // span, span)
            first, second = hidden[:, : span // 2], hidden[:, span // 2 :]
            if level > 1:
                # [Λ, …, Λ^k] -> [Λ, …, Λ^k, Λ^(k+1), …, Λ^(2k)].
                lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
            carried = first[:, -1:] * block_mask[
                :, span // 2 - 1 : span // 2
            ].unsqueeze(-1)
            hidden = torch.cat([first, second + lamb * carried], dim=1)
        return hidden.reshape(rows, width, size)


class PositionWiseFeedForward(nn.Module):
    """Eq. 8's two-layer network and Eq. 9's ``SubLayer`` around it.

    The reference computes ``LayerNorm(x + Dropout(W2 Dropout(GELU(W1 x))))``.
    Eq. 8 has a second GELU around ``W2``; the reference's
    ``PositionwiseFeedForward`` does not, and the code wins. The inner width is
    ``4 * d_model``, which Eq. 8 fixes, so it is not a parameter.
    """

    def __init__(self, d_model: int, dropout: float = 0.2) -> None:
        super().__init__()

        self.w_1 = nn.Linear(d_model, 4 * d_model)
        self.w_2 = nn.Linear(4 * d_model, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        inner = self.dropout(self.activation(self.w_1(states)))
        return self.layer_norm(self.dropout(self.w_2(inner)) + states)


class LRUBlock(nn.Module):
    """One LRU block of §3.4: ``LRUNorm`` and then the feed-forward sublayer."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.2,
        lru_dropout: float = 0.2,
        r_min: float = 0.8,
        r_max: float = 0.99,
    ) -> None:
        super().__init__()

        self.lru_layer = LRULayer(
            d_model, dropout=lru_dropout, r_min=r_min, r_max=r_max
        )
        self.feed_forward = PositionWiseFeedForward(d_model, dropout=dropout)

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.lru_layer(states, mask))


class LRURec(nn.Module):
    """Linear recurrent units over an item history, scored by a tied head.

    Causal by construction, like SASRec and unlike BERT4Rec: the state at a
    position is a sum over that position and the ones before it, so the output
    there is a next-item prediction and every position is a training target.
    Order comes from the recurrence itself, so there is no positional table and
    nothing here is sized by the context window.

    Constructed by :meth:`~.trainer.LRURecTrainer._build_model`, which takes
    the architecture from :class:`~.config.LRURecConfig` and the token ids from
    the tokenizer, so the defaults here are the config's rather than a second
    opinion about them.
    """

    def __init__(
        self,
        n_items: int,
        d_model: int = 64,
        n_blocks: int = 2,
        dropout: float = 0.2,
        lru_dropout: float = 0.2,
        r_min: float = 0.8,
        r_max: float = 0.99,
        pad: int = 0,
        n_reserved: int = 2,
    ) -> None:
        super().__init__()

        self.n_items = n_items
        self.d_model = d_model
        self.pad = pad
        self.n_reserved = n_reserved
        self.vocab_size = n_reserved + n_items

        # The catalog sits above the reserved ids, as in every sequential model
        # here. The reference's table is num_items + 1 rows with padding at 0,
        # so this is the same layout with the package's reserve in place of its
        # single pad row. No padding_idx: the reference's nn.Embedding has
        # none, so the pad row is initialized and trained like any item's. Its
        # embedding is what a position with no history reads -- see
        # LRURecTrainer._next_item_batch.
        self.item_embedding = nn.Embedding(self.vocab_size, d_model)
        # Eq. 7 is LayerNorm(E x). The reference's LRUEmbedding computes
        # layer_norm(embed_dropout(token(x))) -- dropout first, then the norm.
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)

        self.blocks = nn.ModuleList(
            LRUBlock(
                d_model,
                dropout=dropout,
                lru_dropout=lru_dropout,
                r_min=r_min,
                r_max=r_max,
            )
            for _ in range(n_blocks)
        )

        # Eq. 10's b_o, one per vocabulary row like the reference's
        # LRUModel.bias, so the reserved ids are scored too.
        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))

    def forward(self, item_history: torch.Tensor) -> torch.Tensor:
        """Hidden state at every position, ``(rows, length, d_model)``.

        ``item_history`` must be *left*-padded with ``pad``: the recurrence
        masks carries out of padding, which is exact only when padding is a
        prefix. Real positions are found as the reference's ``get_mask`` finds
        them, by comparing against the pad id, so an ``unk`` substituted into a
        history still counts as an interaction.

        The batch is padded further on the left to a power of two for the
        scan, with zero vectors as in the reference, and the extra columns are
        cut off again before returning.
        """
        if item_history.ndim != 2 or item_history.shape[1] < 1:
            # Zero columns would make the closing [:, -length:] a [:, -0:],
            # which keeps the whole padded width instead of none of it. An
            # empty history is still one pad column, which is a valid input.
            raise ValueError(
                "item_history must be (rows, length) with length >= 1, got "
                f"shape {tuple(item_history.shape)}"
            )
        mask = item_history != self.pad
        states = self.embedding_norm(
            self.embedding_dropout(self.item_embedding(item_history))
        )

        length = states.shape[1]
        extra = _next_power_of_two(length) - length
        states = F.pad(states, (0, 0, extra, 0))
        mask = F.pad(mask, (extra, 0))

        for block in self.blocks:
            states = block(states, mask)
        return states[:, -length:]

    def score(self, states: torch.Tensor) -> torch.Tensor:
        """Eq. 10: logits over the whole vocabulary, reserved ids included.

        ``E h + b_o`` with ``E`` the embedding matrix itself. The reference
        scores its pad row in the same softmax, so the reserved ids here are
        scored too: cross entropy then pushes them down along with every other
        wrong answer. Prediction slices them away before ranking.
        """
        return states @ self.item_embedding.weight.T + self.output_bias

    def score_items(self, states: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Eq. 10 at chosen vocabulary rows: ``(..., n)`` logits for ``items``.

        ``states`` is ``(..., d_model)`` and ``items`` is ``(..., n)`` token ids
        with the same leading shape, so each state is scored against its own
        row of items. The logits are :meth:`score`'s at those ids, output bias
        included, without forming the rest of the vocabulary, though not
        always to the last bit: the two sum in different orders. That
        is the reference's sampled ``einsum`` over ``embedding_weight[all_items]``
        for XLong, in training and in evaluation alike.
        """
        embeddings = self.item_embedding(items)
        return (
            torch.einsum("...d,...nd->...n", states, embeddings)
            + self.output_bias[items]
        )
