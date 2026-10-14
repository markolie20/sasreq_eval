# Modified for seqrec_eval: differs from upstream compresso-recsys; see vendor/compresso-recsys/VENDORED.md
"""LRURec: the linear recurrence, its parallel scan, and next-item training.

Organized like ``tests/test_bert4rec.py``. LRURec is causal like SASRec rather
than bidirectional like BERT4Rec, so where a test here mirrors one there the
assertion is often inverted -- a BERT4Rec position must read the items after
it, an LRURec position must not.

The scan sections have no counterpart elsewhere. They pin the recursive
parallelization of §3.3 against the plain sequential recurrence it computes,
and the padding mask that makes a left-padded batch exact.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from compresso_recsys.models.lrurec import (
    LRURec,
    LRURecConfig,
    LRURecTrainer,
)
from compresso_recsys.models.lrurec.model import LRULayer, _next_power_of_two
from compresso_recsys.models.lrurec.trainer import _sliding_windows
from compresso_recsys.models.sequence_batching import SequenceBatcher
from compresso_recsys.models.tokenizer import ItemTokenizer
from compresso_recsys.sequences import ItemSequences

N_ITEMS = 8
PAD_ID = 0
UNK_ID = 1
N_RESERVED = 2


def _seqs(rows, n_items=N_ITEMS):
    return ItemSequences.from_rows(rows, n_items=n_items)


def _cycle_rows(n_rows=48, length=4, n_items=N_ITEMS):
    """Histories drawn from a single cycle, so ``next(i) == (i + 1) % n_items``.

    Learnable from the last item alone, which is the point: it isolates whether
    the wiring works from whether the architecture is any good.
    """
    return [
        [(start + step) % n_items for step in range(length)]
        for start in range(n_rows)
    ]


def _tokenizer(n_items=N_ITEMS, **kwargs):
    return ItemTokenizer(n_items, **kwargs)


def _batcher(max_length=None, n_items=N_ITEMS, **kwargs):
    return SequenceBatcher(_tokenizer(n_items), max_length=max_length, **kwargs)


def _config(**overrides):
    defaults = dict(
        d_model=16,
        n_blocks=2,
        dropout=0.0,
        lru_dropout=0.0,
        max_history_length=6,
        epochs=1,
        batch_size=16,
        show_progress=False,
        seed=0,
    )
    return LRURecConfig(**{**defaults, **overrides})


def _trainer(config=None, batcher=None, **overrides):
    """A trainer whose batcher is already built, so the sample helpers can run.

    ``fit`` builds one from the training catalog; the sample-construction tests
    call ``_training_samples`` and ``_next_item_batch`` directly and need the
    vocabulary without a fit.
    """
    config = config or _config(**overrides)
    if batcher is None:
        batcher = _batcher(
            max_length=config.max_history_length, padding="left"
        )
    return LRURecTrainer(config, batcher)


def _fitted_on_cycle(epochs=60, **overrides):
    config = _config(epochs=epochs, lr=0.01, batch_size=48, **overrides)
    return LRURecTrainer(config).fit(_seqs(_cycle_rows()))


def _model(**overrides):
    defaults = dict(
        n_items=N_ITEMS,
        d_model=16,
        n_blocks=2,
        dropout=0.0,
        lru_dropout=0.0,
        pad=PAD_ID,
        n_reserved=N_RESERVED,
    )
    model = LRURec(**{**defaults, **overrides})
    model.eval()
    return model


def _tokens(*rows, width=None):
    """Left-pad token rows to a common width, as the batcher does."""
    width = width or max(len(row) for row in rows)
    out = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    for index, row in enumerate(rows):
        if row:
            out[index, width - len(row) :] = torch.tensor(row)
    return out


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(d_model=0), "d_model must be >= 1"),
        (dict(n_blocks=0), "n_blocks must be >= 1"),
        (dict(dropout=1.0), r"dropout must be in \[0, 1\)"),
        (dict(lru_dropout=-0.1), r"lru_dropout must be in \[0, 1\)"),
        (dict(max_history_length=0), "max_history_length must be >= 1"),
        (dict(r_min=0.0), "0 < r_min <= r_max < 1"),
        (dict(r_min=0.9, r_max=0.8), "0 < r_min <= r_max < 1"),
        (dict(r_max=1.0), "0 < r_min <= r_max < 1"),
        (dict(initializer_range=0.0), "initializer_range must be > 0"),
        (dict(sliding_window_step=0.0), r"sliding_window_step .* \(0, 1\]"),
        (dict(sliding_window_step=1.5), r"sliding_window_step .* \(0, 1\]"),
        (
            dict(sliding_window_step=0.1, max_history_length=5),
            "at least one interaction",
        ),
        (dict(unk_dropout=1.0), r"unk_dropout must be in \[0, 1\)"),
        (dict(unk_dropout=-0.1), r"unk_dropout must be in \[0, 1\)"),
        (dict(n_negatives=0), "n_negatives must be None"),
        (dict(n_negatives=True), "n_negatives must be None"),
        (dict(n_negatives=2.0), "n_negatives must be None"),
        (dict(batch_size=0), "batch_size must be >= 1"),
        (dict(epochs=0), "epochs must be >= 1"),
        (dict(lr=0.0), "lr must be > 0"),
        (dict(weight_decay=-0.1), "weight_decay must be >= 0"),
        (dict(epsilon=0.0), "epsilon must be > 0"),
        (dict(max_grad_norm=0.0), "max_grad_norm must be > 0"),
        (dict(optimizer="Adam"), "published optimizer is AdamW"),
        (dict(betas=(0.9,)), "betas must be two values"),
        (dict(betas=(0.9, 1.0)), r"betas must each be in \[0, 1\)"),
    ],
)
def test_invalid_configuration_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _config(**kwargs)


def test_the_defaults_are_the_published_settings():
    """Every default is the paper's, or the reference's where it is silent.

    The config docstring cites a provenance per field; this pins the numbers so
    the citation cannot quietly stop describing the code.

    "The reference" means what train.py runs: config.py's flag defaults after
    set_template has overwritten them. The two differ on batch size (128
    against 64) and on the learning-rate schedule (off against on), so citing
    the wrong tier is worth a test rather than a comment.
    """
    cfg = LRURecConfig()

    # Architecture: H = 64 (§5), two stacked blocks (§3.4), N = 200 for ML-1M
    # (§5.1), the ring [0.8, 0.99] (§3.2), and the reference's truncated normal
    # at std 0.02. Both dropouts are set_template's 0.2.
    assert (cfg.d_model, cfg.n_blocks) == (64, 2)
    assert cfg.max_history_length == 200
    assert (cfg.dropout, cfg.lru_dropout) == (0.2, 0.2)
    assert (cfg.r_min, cfg.r_max) == (0.8, 0.99)
    assert cfg.initializer_range == 0.02

    # Samples: sliding_window_size is never overridden by set_template, so the
    # flag default of 1.0 -- windows end to end -- is what was published.
    assert cfg.sliding_window_step == 1.0
    # No unk corruption: the paper has none, and SASRec and BERT4Rec default
    # it off for the same parity reason.
    assert cfg.unk_dropout == 0.0
    # The full softmax of §5.1; §5.3 samples 100 negatives for XLong alone.
    assert cfg.n_negatives is None

    # Optimization: AdamW at a static 1e-3 and a batch of 128 (§5.1);
    # weight decay 0.01, eps 1e-9 and clipping at 5 from the reference;
    # PyTorch's default betas, which the reference never changes.
    assert cfg.optimizer == "AdamW"
    assert cfg.lr == 1e-3
    assert cfg.batch_size == 128
    assert cfg.weight_decay == 0.01
    assert cfg.epsilon == 1e-9
    assert cfg.max_grad_norm == 5.0
    assert cfg.betas == (0.9, 0.999)
    # The reference's num_epochs, a cap it stops well short of; a fixed budget
    # here, since early stopping is not part of the package's contract.
    assert cfg.epochs == 500


def test_there_is_no_learning_rate_schedule_to_configure():
    """§5.1's "static learning rate": set_template turns the schedule off."""
    fields = {field.name for field in dataclasses.fields(LRURecConfig)}
    assert not fields & {"warmup_steps", "warmup_fraction", "schedule"}


def test_the_inner_widths_are_not_configurable():
    """§3.2 doubles the state and Eq. 8 quadruples the feed-forward width."""
    fields = {field.name for field in dataclasses.fields(LRURecConfig)}
    assert not fields & {"d_ff", "state_size", "hidden_size"}
    model = _model(d_model=16)
    assert model.blocks[0].lru_layer.state_size == 32
    assert model.blocks[0].feed_forward.w_1.out_features == 64


def test_betas_survive_a_json_round_trip_as_a_tuple():
    config = _config()
    restored = LRURecConfig(
        **json.loads(json.dumps(dataclasses.asdict(config), default=str))
    )
    assert isinstance(restored.betas, tuple)
    assert restored == config


# --------------------------------------------------------------------------
# the linear recurrence
# --------------------------------------------------------------------------


def _sequential(driven, lamb, mask):
    """Eq. 5's recurrence one step at a time, restarting after padding."""
    rows, width, size = driven.shape
    out = torch.zeros_like(driven)
    for row in range(rows):
        state = torch.zeros(size, dtype=driven.dtype)
        for step in range(width):
            if not mask[row, step]:
                state = torch.zeros(size, dtype=driven.dtype)
                continue
            state = lamb[0] * state + driven[row, step]
            out[row, step] = state
    return out


@pytest.mark.parametrize("width", [1, 2, 4, 8, 16])
def test_the_parallel_scan_is_the_sequential_recurrence(width):
    """§3.3 is a faster way to compute Eq. 4, not a different function."""
    torch.manual_seed(0)
    size = 6
    driven = torch.randn(5, width, size, dtype=torch.cfloat)
    lamb = LRULayer(3).eigenvalues().unsqueeze(0)
    lengths = [width, width // 2, 1, 0, max(0, width - 3)]
    mask = torch.zeros(5, width, dtype=torch.bool)
    for row, length in enumerate(lengths):
        if length:
            mask[row, width - length :] = True
    scanned = LRULayer._parallel_scan(driven, lamb, mask)
    expected = _sequential(driven, lamb, mask)
    assert torch.allclose(scanned[mask], expected[mask], atol=1e-5)


def test_padding_is_never_carried_into_a_real_position():
    """The reference's mask: a block whose first half ends on padding passes
    nothing on, which is the paper's h_1 = B x_1."""
    driven = torch.zeros(1, 4, 2, dtype=torch.cfloat)
    driven[0, :2] = 100.0  # padding with a loud state of its own
    driven[0, 2:] = 1.0
    lamb = torch.full((1, 2), 0.5, dtype=torch.cfloat)
    mask = torch.tensor([[False, False, True, True]])
    scanned = LRULayer._parallel_scan(driven, lamb, mask)
    assert torch.allclose(scanned[0, 2], torch.ones(2, dtype=torch.cfloat))
    assert torch.allclose(scanned[0, 3], torch.full((2,), 1.5, dtype=torch.cfloat))


@pytest.mark.parametrize(
    "length,expected", [(1, 1), (2, 2), (3, 4), (8, 8), (9, 16), (200, 256)]
)
def test_the_scan_width_is_the_next_power_of_two(length, expected):
    assert _next_power_of_two(length) == expected


def test_the_state_is_twice_the_model_width():
    """§3.2: "we double the size of H in Λ"."""
    layer = LRULayer(8)
    assert layer.params_log.shape == (3, 16)
    assert layer.in_proj.weight.shape == (16, 8)
    assert layer.out_proj.weight.shape == (8, 16)


def test_the_projections_are_complex():
    layer = LRULayer(8)
    assert layer.in_proj.weight.dtype == torch.cfloat
    assert layer.out_proj.weight.dtype == torch.cfloat
    assert layer.params_log.dtype == torch.float32


def test_the_eigenvalues_start_on_the_ring():
    """|λ| in [r_min, r_max], phase in [0, 2π), and γ = sqrt(1 - |λ|²)."""
    torch.manual_seed(0)
    layer = LRULayer(64, r_min=0.8, r_max=0.99)
    lamb = layer.eigenvalues().detach()
    radius = lamb.abs()
    assert float(radius.min()) >= 0.8 - 1e-6
    assert float(radius.max()) <= 0.99 + 1e-6
    phase = torch.exp(layer.params_log[1]).detach()
    assert float(phase.min()) >= 0.0
    assert float(phase.max()) < 2 * math.pi
    gamma = torch.exp(layer.params_log[2]).detach()
    assert torch.allclose(gamma, torch.sqrt(1 - radius**2), atol=1e-6)


def test_the_lru_output_is_the_real_projection_plus_the_input_normalized():
    """Eq. 5 and Eq. 9: LayerNorm(Re(C h + c) + x), with γ scaling B x + b.

    The biases are the reference's; Eq. 5 has none.
    """
    torch.manual_seed(0)
    layer = LRULayer(4, dropout=0.0)
    layer.eval()
    states = torch.randn(2, 4, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    _, _, gamma = torch.exp(layer.params_log).split((1, 1, 1))
    driven = layer.in_proj(states.to(torch.cfloat)) * gamma
    hidden = _sequential(driven, layer.eigenvalues().unsqueeze(0), mask)
    expected = layer.layer_norm(layer.out_proj(hidden).real + states)
    with torch.no_grad():
        assert torch.allclose(layer(states, mask), expected, atol=1e-5)


def test_the_feed_forward_has_no_outer_gelu():
    """Eq. 8 wraps a GELU around W2; the reference's network does not."""
    torch.manual_seed(0)
    model = _model()
    feed_forward = model.blocks[0].feed_forward
    states = torch.randn(2, 3, model.d_model)
    expected = feed_forward.layer_norm(
        feed_forward.w_2(F.gelu(feed_forward.w_1(states))) + states
    )
    with torch.no_grad():
        assert torch.allclose(feed_forward(states), expected, atol=1e-6)


def test_the_embedding_is_dropped_out_before_it_is_normalized():
    """Eq. 7 is LayerNorm(E x); the reference is layer_norm(dropout(E x))."""
    model = _model(dropout=0.5)
    model.train()
    read = []
    model.embedding_norm.register_forward_pre_hook(
        lambda module, args: read.append(args[0])
    )
    torch.manual_seed(0)
    model(_tokens([2, 3, 4]))
    # Dropout zeroes entries of what the norm reads. In the other order the
    # norm would read the raw embedding, which has no exact zeros.
    assert (read[0] == 0).any()


def test_the_eigenvalues_stay_inside_the_unit_circle_whatever_the_parameters():
    """The exp(-exp(·)) parameterization keeps |λ| < 1 by construction."""
    layer = LRULayer(4)
    with torch.no_grad():
        layer.params_log.normal_(0.0, 5.0)
    assert float(layer.eigenvalues().detach().abs().max()) < 1.0


# --------------------------------------------------------------------------
# architecture
# --------------------------------------------------------------------------


def test_the_vocabulary_is_the_catalog_above_the_reserved_ids():
    model = _model()
    assert model.vocab_size == N_RESERVED + N_ITEMS
    assert model.item_embedding.num_embeddings == N_RESERVED + N_ITEMS
    assert model.output_bias.shape == (N_RESERVED + N_ITEMS,)


def test_there_is_no_positional_table():
    """Order comes from the recurrence, so nothing is sized by the window."""
    model = _model()
    assert not any("position" in name for name, _ in model.named_parameters())
    # And so any length runs, including one longer than any training window.
    states = model(_tokens(list(range(2, 2 + N_ITEMS)) * 5))
    assert states.shape == (1, 5 * N_ITEMS, model.d_model)


def test_the_embedding_has_no_padding_index():
    """The reference's nn.Embedding has none, so the pad row is a real row."""
    assert _model().item_embedding.padding_idx is None


def test_a_history_with_no_columns_is_refused():
    """Zero columns would turn the closing [:, -length:] into [:, -0:], the
    whole padded width. An empty history is one pad column instead."""
    model = _model()
    with pytest.raises(ValueError, match="length >= 1"):
        model(torch.empty((2, 0), dtype=torch.long))
    assert model(torch.full((2, 1), PAD_ID)).shape == (2, 1, 16)


def test_the_states_are_one_per_position():
    model = _model()
    states = model(_tokens([2, 3, 4], [5]))
    assert states.shape == (2, 3, model.d_model)


def test_the_score_is_logits_over_the_whole_vocabulary():
    model = _model()
    with torch.no_grad():
        scores = model.score(model(_tokens([2, 3, 4]))[:, -1])
    assert scores.shape == (1, N_RESERVED + N_ITEMS)
    assert not torch.allclose(scores.exp().sum(dim=-1), torch.ones(1))


def test_the_head_is_tied_to_the_embedding():
    """Eq. 10: Pred(h) = E h + b_o with E the embedding matrix itself."""
    model = _model()
    states = torch.randn(3, model.d_model)
    expected = states @ model.item_embedding.weight.T + model.output_bias
    assert torch.allclose(model.score(states), expected)


def test_a_position_does_not_read_the_items_after_it():
    """Causal, like SASRec: the opposite of BERT4Rec's bidirectional test."""
    model = _model()
    with torch.no_grad():
        first = model(_tokens([2, 3, 4, 5]))
        second = model(_tokens([2, 3, 4, 9]))
    assert torch.allclose(first[:, :3], second[:, :3], atol=1e-6)
    assert not torch.allclose(first[:, 3], second[:, 3])


def test_a_position_reads_the_items_before_it():
    model = _model()
    with torch.no_grad():
        first = model(_tokens([2, 3, 4, 5]))
        second = model(_tokens([9, 3, 4, 5]))
    assert not torch.allclose(first[:, -1], second[:, -1])


@pytest.mark.parametrize("width", [3, 4, 5, 9, 17])
def test_the_amount_of_left_padding_does_not_change_a_real_state(width):
    """What lets the trainer pack each batch to its own longest row."""
    model = _model()
    row = [2, 3, 4]
    with torch.no_grad():
        tight = model(_tokens(row))
        padded = model(_tokens(row, width=width))
    assert torch.allclose(tight[0], padded[0, -3:], atol=1e-5)


def test_the_padding_embedding_does_not_reach_a_real_state():
    model = _model()
    tokens = _tokens([2, 3, 4], width=6)
    with torch.no_grad():
        before = model(tokens)[:, -3:]
        model.item_embedding.weight[PAD_ID].normal_(0.0, 10.0)
        after = model(tokens)[:, -3:]
    assert torch.allclose(before, after, atol=1e-5)


def test_unk_is_read_as_an_interaction_not_as_padding():
    """Real positions are found by comparing against pad, as the reference's
    get_mask does, so a history with an unknown item keeps its length."""
    model = _model()
    with torch.no_grad():
        with_unk = model(_tokens([2, UNK_ID, 4]))
        shorter = model(_tokens([2, 4]))
    assert not torch.allclose(with_unk[0, -1], shorter[0, -1])


# --------------------------------------------------------------------------
# training samples
# --------------------------------------------------------------------------


def test_a_history_shorter_than_window_plus_step_is_one_sample():
    history = np.arange(5)
    assert [list(s) for s in _sliding_windows(history, 4, 4)] == [list(range(5))]


def test_a_history_at_window_plus_step_is_cut_end_to_end():
    windows = _sliding_windows(np.arange(8), 4, 4)
    assert [list(window) for window in windows] == [[4, 5, 6, 7], [0, 1, 2, 3]]


def test_the_earliest_interactions_are_dropped_when_the_windows_do_not_fit():
    """The oldest window is not pinned to the start, unlike BERT4Rec's."""
    windows = _sliding_windows(np.arange(10), 4, 4)
    assert [list(window) for window in windows] == [[6, 7, 8, 9], [2, 3, 4, 5]]


def test_a_smaller_step_overlaps_the_windows():
    windows = _sliding_windows(np.arange(8), 4, 2)
    assert [list(window) for window in windows] == [
        [4, 5, 6, 7],
        [2, 3, 4, 5],
        [0, 1, 2, 3],
    ]


def test_the_step_is_a_fraction_of_the_window():
    trainer = _trainer(max_history_length=4, sliding_window_step=0.5)
    samples = trainer._training_samples(_seqs([list(range(8))]))
    assert [list(sample) for sample in samples] == [
        [4, 5, 6, 7],
        [2, 3, 4, 5],
        [0, 1, 2, 3],
    ]


def test_an_empty_history_contributes_nothing():
    trainer = _trainer()
    assert len(trainer._training_samples(_seqs([[], [0, 1]]))) == 1


def test_a_single_item_history_is_a_sample():
    """Its one item is labelled at a padding input -- see the batch tests."""
    trainer = _trainer()
    assert len(trainer._training_samples(_seqs([[3]]))) == 1


def test_fitting_only_empty_histories_is_refused():
    trainer = LRURecTrainer(_config())
    with pytest.raises(ValueError, match="every history is empty"):
        trainer.fit(_seqs([[], []]))
    # Refused before the model is built, so nothing claims to be fitted.
    assert not trainer.is_fitted


def _batch(trainer, rows, seed=0):
    samples = trainer._training_samples(_seqs(rows))
    return trainer._next_item_batch(samples, np.random.default_rng(seed))


def test_each_input_predicts_the_next_item():
    trainer = _trainer(max_history_length=6)
    inputs, labels = _batch(trainer, [[0, 1, 2, 3]])
    offset = N_RESERVED
    assert inputs[0].tolist() == [PAD_ID, 0 + offset, 1 + offset, 2 + offset]
    assert labels[0].tolist() == [0 + offset, 1 + offset, 2 + offset, 3 + offset]


def test_the_first_item_is_labelled_at_a_padding_input():
    """The reference's ``tokens = seq[:-1]`` is one shorter than its labels, so
    every sample's first interaction is predicted from no history at all."""
    trainer = _trainer(max_history_length=6)
    inputs, labels = _batch(trainer, [[5]])
    assert inputs.tolist() == [[PAD_ID]]
    assert labels.tolist() == [[5 + N_RESERVED]]


def test_each_window_start_is_labelled_at_a_padding_input():
    """A window's first item has a real predecessor, at the end of the window
    before, but the shift is per sample, so it is predicted from padding too."""
    trainer = _trainer(max_history_length=3)
    # Length 6 = window 3 + step 3, so two windows, newest first.
    inputs, labels = _batch(trainer, [[0, 1, 2, 3, 4, 5]])
    offset = N_RESERVED
    assert inputs.tolist() == [
        [PAD_ID, 3 + offset, 4 + offset],
        [PAD_ID, 0 + offset, 1 + offset],
    ]
    assert labels[:, 0].tolist() == [3 + offset, 0 + offset]


def test_a_sample_longer_than_the_window_keeps_its_last_labels():
    """``labels = seq[-max_len:]`` and ``tokens = seq[:-1][-max_len:]``."""
    trainer = _trainer(max_history_length=3)
    # Length 5 < window 3 + step 3, so one sample, longer than the window.
    inputs, labels = _batch(trainer, [[0, 1, 2, 3, 4]])
    offset = N_RESERVED
    assert labels[0].tolist() == [2 + offset, 3 + offset, 4 + offset]
    assert inputs[0].tolist() == [1 + offset, 2 + offset, 3 + offset]


def test_rows_are_left_padded_to_the_longest_in_the_batch():
    trainer = _trainer(max_history_length=6)
    inputs, labels = _batch(trainer, [[0, 1, 2], [3]])
    assert inputs.shape == (2, 3)
    assert inputs[1].tolist() == [PAD_ID, PAD_ID, PAD_ID]
    assert labels[1].tolist() == [PAD_ID, PAD_ID, 3 + N_RESERVED]


def test_unk_is_never_inserted_at_the_default_rate():
    trainer = _trainer()
    inputs, _ = _batch(trainer, [list(range(6))] * 8)
    assert not (inputs == UNK_ID).any()


def test_unk_dropout_replaces_inputs_and_never_labels():
    trainer = _trainer(_config(unk_dropout=0.5))
    rows = [list(range(6))] * 16
    inputs, labels = _batch(trainer, rows, seed=3)
    clean_inputs, clean_labels = _batch(_trainer(), rows, seed=3)
    replaced = inputs == UNK_ID
    assert replaced.any()
    assert torch.equal(labels, clean_labels)
    # Only real inputs are eligible: padding stays padding.
    assert not (replaced & (clean_inputs == PAD_ID)).any()
    assert torch.equal(inputs[~replaced], clean_inputs[~replaced])


def test_unk_dropout_is_ignored_without_an_unk_token():
    batcher = SequenceBatcher(
        ItemTokenizer(N_ITEMS, special_tokens={"pad": 0}),
        max_length=6,
        padding="left",
    )
    trainer = _trainer(_config(unk_dropout=0.5), batcher)
    inputs, _ = _batch(trainer, [list(range(6))] * 8)
    assert int(inputs.min()) == 0  # pad only; nothing became an unk id


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------


def test_fit_builds_a_left_padded_batcher_from_the_config():
    trainer = LRURecTrainer(_config(max_history_length=5)).fit(
        _seqs(_cycle_rows())
    )
    assert trainer.batcher.max_length == 5
    assert trainer.batcher.padding == "left"


def test_the_built_vocabulary_is_the_package_default():
    """Only pad is needed; unk is there for a later split stage's new items."""
    trainer = LRURecTrainer(_config()).fit(_seqs(_cycle_rows()))
    assert dict(trainer.batcher.tokenizer.special_tokens) == {"pad": 0, "unk": 1}


def test_a_batcher_naming_no_window_inherits_the_configs():
    trainer = LRURecTrainer(_config(max_history_length=5), _batcher())
    trainer.fit(_seqs(_cycle_rows()))
    assert trainer.batcher.max_length == 5


def test_a_batcher_naming_a_different_window_is_refused():
    trainer = LRURecTrainer(_config(max_history_length=5), _batcher(max_length=4))
    with pytest.raises(ValueError, match="disagree"):
        trainer.fit(_seqs(_cycle_rows()))


def test_a_right_padded_batcher_is_put_on_the_left():
    trainer = LRURecTrainer(_config(), _batcher(padding="right"))
    trainer.fit(_seqs(_cycle_rows()))
    assert trainer.batcher.padding == "left"


def test_fit_returns_the_trainer_and_reports_the_catalog():
    trainer = LRURecTrainer(_config())
    assert trainer.fit(_seqs(_cycle_rows())) is trainer
    assert trainer.is_fitted
    assert trainer.n_items == N_ITEMS


def test_before_fitting_nothing_is_claimed():
    trainer = LRURecTrainer(_config())
    assert not trainer.is_fitted
    assert trainer.n_items is None


def test_fit_refuses_a_matrix_source():
    with pytest.raises(TypeError, match="trains on ItemSequences"):
        LRURecTrainer(_config()).fit(np.zeros((2, N_ITEMS)))


def test_fit_refuses_an_empty_training_set():
    with pytest.raises(ValueError, match="zero sequences"):
        LRURecTrainer(_config()).fit(_seqs([]))


def test_fit_rejects_a_supplied_batcher_for_a_different_catalog():
    trainer = LRURecTrainer(_config(), _batcher(n_items=N_ITEMS + 1))
    with pytest.raises(ValueError, match="batcher tokenizer has"):
        trainer.fit(_seqs(_cycle_rows()))


def test_training_records_a_loss_per_epoch():
    trainer = _fitted_on_cycle(epochs=3)
    assert [record["epoch"] for record in trainer.history] == [1.0, 2.0, 3.0]
    # Four labels per history of four: the first is labelled at padding.
    assert all(record["positions"] == 48 * 4 for record in trainer.history)


def test_training_reduces_the_loss_on_a_learnable_cycle():
    trainer = _fitted_on_cycle(epochs=40)
    assert trainer.history[-1]["loss"] < trainer.history[0]["loss"]


def test_refitting_starts_the_history_over():
    trainer = _fitted_on_cycle(epochs=3)
    trainer.fit(_seqs(_cycle_rows()))
    assert len(trainer.history) == 3


def test_the_same_seed_gives_the_same_model():
    first = _fitted_on_cycle(epochs=2, seed=7)
    second = _fitted_on_cycle(epochs=2, seed=7)
    for left, right in zip(
        first.model.state_dict().values(), second.model.state_dict().values()
    ):
        assert torch.equal(left, right)


def test_a_different_seed_gives_a_different_model():
    first = _fitted_on_cycle(epochs=2, seed=1)
    second = _fitted_on_cycle(epochs=2, seed=2)
    assert not torch.equal(
        first.model.item_embedding.weight, second.model.item_embedding.weight
    )


# --------------------------------------------------------------------------
# the objective and the optimizer
# --------------------------------------------------------------------------


def test_the_loss_is_cross_entropy_over_every_labelled_position():
    """The reference's CrossEntropyLoss(ignore_index=0) over the vocabulary,
    reserved ids included in the softmax."""
    trainer = LRURecTrainer(_config()).fit(_seqs(_cycle_rows()))
    rows = [[0, 1, 2, 3], [4, 5]]
    inputs, labels = _batch(trainer, rows)
    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model.score(trainer.model(inputs))
        expected = float(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=PAD_ID,
            )
        )
    samples = trainer._training_samples(_seqs(rows))
    loss, count = trainer._train_step(samples, np.random.default_rng(0))
    assert count == 6
    assert loss == pytest.approx(expected, abs=1e-5)


def test_sampled_logits_are_the_full_logits_at_those_items():
    model = _model()
    states = torch.randn(3, 5, 16)
    items = torch.randint(0, model.vocab_size, (3, 5, 7))
    expected = model.score(states).gather(-1, items)
    assert torch.allclose(model.score_items(states, items), expected, atol=1e-6)


def test_the_sampled_loss_is_cross_entropy_over_the_target_and_its_negatives():
    """The reference's XLong branch: the target appended after the negatives,
    at index n_negatives, and cross entropy over those logits alone."""
    trainer = LRURecTrainer(_config(n_negatives=5)).fit(_seqs(_cycle_rows()))
    model = trainer.model.eval()
    states = torch.randn(6, 16)
    targets = torch.arange(6) + N_RESERVED
    with torch.no_grad():
        torch.manual_seed(3)
        loss = trainer._sampled_softmax_loss(states, targets)
        torch.manual_seed(3)
        negatives = torch.randint(N_RESERVED, N_RESERVED + N_ITEMS, (6, 5))
        full = model.score(states)
        logits = torch.cat(
            (full.gather(1, negatives), full.gather(1, targets[:, None])), 1
        )
        expected = F.cross_entropy(logits, torch.full((6,), 5))
    assert float(loss) == pytest.approx(float(expected), abs=1e-5)


def test_sampled_negatives_are_catalog_items_and_may_be_the_target(monkeypatch):
    trainer = LRURecTrainer(_config(n_negatives=200)).fit(_seqs(_cycle_rows()))
    seen = []
    original = trainer.model.score_items
    monkeypatch.setattr(
        trainer.model,
        "score_items",
        lambda states, items: seen.append(items) or original(states, items),
    )
    samples = trainer._training_samples(_seqs(_cycle_rows()))
    trainer._train_step(samples, np.random.default_rng(0))
    (candidates,) = seen
    negatives, targets = candidates[:, :-1], candidates[:, -1]
    # Never padding or unk: the draw is over the catalog's token ids.
    assert int(negatives.min()) >= N_RESERVED
    assert int(negatives.max()) < N_RESERVED + N_ITEMS
    # Nothing is excluded, so 200 draws from 8 items include every target.
    assert (negatives == targets[:, None]).any(dim=1).all()


def test_the_full_softmax_draws_no_negatives(monkeypatch):
    calls = []
    monkeypatch.setattr(
        LRURecTrainer, "_sampled_softmax_loss", lambda *args: calls.append(args)
    )
    LRURecTrainer(_config()).fit(_seqs(_cycle_rows()))
    assert not calls


def test_sampled_training_learns_the_cycle():
    trainer = _fitted_on_cycle(epochs=200, n_negatives=3)
    predictions = trainer.predict_on_batch(
        _seqs([[0, 1, 2], [2, 3, 4]]), k=1, exclude_seen=False
    )
    assert predictions.cols[:, 0].tolist() == [3, 5]


def test_gradients_are_clipped_at_the_configured_norm(monkeypatch):
    calls = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        calls.append(max_norm)
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    LRURecTrainer(_config(max_grad_norm=3.0)).fit(_seqs(_cycle_rows()))
    assert calls and set(calls) == {3.0}


def test_the_learning_rate_is_static():
    """§5.1: "the static learning rate of 1e-3" -- no warmup, no decay."""
    trainer = _fitted_on_cycle(epochs=3)
    assert all(
        group["lr"] == trainer.cfg.lr for group in trainer.optimizer.param_groups
    )


def test_the_optimizer_is_adamw_with_the_reference_epsilon():
    trainer = _fitted_on_cycle(epochs=1)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.optimizer.defaults["eps"] == 1e-9


def test_weight_decay_spares_exactly_biases_and_layer_norms():
    """The reference's no_decay = ['bias', 'layer_norm'], matched by name.

    That rule decays params_log, so this asserts it rather than the
    one-dimensional rule BERT4Rec uses, which would match it here only by the
    accident of params_log being stacked into two dimensions.
    """
    trainer = _fitted_on_cycle(epochs=1)
    decayed, spared = trainer.optimizer.param_groups
    assert decayed["weight_decay"] == trainer.cfg.weight_decay
    assert spared["weight_decay"] == 0.0
    names = {id(p): n for n, p in trainer.model.named_parameters()}
    decayed_names = {names[id(p)] for p in decayed["params"]}
    spared_names = {names[id(p)] for p in spared["params"]}
    assert "blocks.0.lru_layer.params_log" in decayed_names
    assert "item_embedding.weight" in decayed_names
    assert "output_bias" in spared_names
    assert "blocks.0.lru_layer.in_proj.bias" in spared_names
    assert "embedding_norm.weight" in spared_names
    assert "blocks.1.feed_forward.layer_norm.weight" in spared_names
    assert all(
        name.endswith("bias") or "norm" in name for name in spared_names
    )
    assert not any(name.endswith("bias") for name in decayed_names)


def test_the_truncated_normal_covers_biases_and_complex_weights():
    """The reference initializes everything but LayerNorms and params_log."""
    trainer = _trainer()
    model = trainer._build_model()
    bound = 2 * trainer.cfg.initializer_range
    layer = model.blocks[0].lru_layer
    for parameter in (
        layer.in_proj.weight,
        layer.in_proj.bias,
        layer.out_proj.weight,
        model.blocks[0].feed_forward.w_1.bias,
        model.output_bias,
        model.item_embedding.weight,
    ):
        values = (
            torch.view_as_real(parameter) if parameter.is_complex() else parameter
        ).detach()
        assert float(values.abs().max()) <= bound
        assert float(values.abs().max()) > 0.0
    # Real and imaginary parts are drawn independently, so neither is zero.
    assert float(layer.in_proj.weight.detach().imag.abs().max()) > 0.0


def test_the_layer_norms_and_the_ring_keep_their_own_initialization():
    trainer = _trainer()
    model = trainer._build_model()
    assert torch.equal(
        model.embedding_norm.weight, torch.ones_like(model.embedding_norm.weight)
    )
    assert torch.equal(
        model.blocks[0].lru_layer.layer_norm.bias,
        torch.zeros_like(model.blocks[0].lru_layer.layer_norm.bias),
    )
    radius = model.blocks[0].lru_layer.eigenvalues().detach().abs()
    assert float(radius.min()) >= trainer.cfg.r_min - 1e-6


def test_the_padding_row_is_initialized_like_any_other():
    """No padding_idx and no zeroing: the reference trains its pad row."""
    model = _trainer()._build_model()
    assert float(model.item_embedding.weight[PAD_ID].detach().abs().max()) > 0.0


def test_a_reserved_id_is_never_recommended():
    trainer = _fitted_on_cycle(epochs=2)
    predictions = trainer.predict_on_batch(
        _seqs(_cycle_rows(n_rows=4)), k=3, exclude_seen=False
    )
    assert int(predictions.cols.min()) >= 0
    assert int(predictions.cols.max()) < N_ITEMS


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------


def test_predict_refuses_before_fitting():
    with pytest.raises(RuntimeError, match="must be fitted before predicting"):
        LRURecTrainer(_config()).predict_on_batch(_seqs([[0, 1]]), k=2)


def test_predict_refuses_a_matrix_source():
    trainer = _fitted_on_cycle(epochs=1)
    with pytest.raises(TypeError, match="predicts from ItemSequences"):
        trainer.predict_on_batch(np.zeros((2, N_ITEMS)), k=2)


@pytest.mark.parametrize("k", [0, N_ITEMS + 1])
def test_predict_refuses_an_impossible_k(k):
    trainer = _fitted_on_cycle(epochs=1)
    with pytest.raises(ValueError, match="k must be in"):
        trainer.predict_on_batch(_seqs([[0, 1]]), k=k, exclude_seen=False)


def test_predictions_are_shaped_and_ordered():
    trainer = _fitted_on_cycle(epochs=2)
    predictions = trainer.predict_on_batch(
        _seqs([[0, 1, 2], [3, 4, 5]]), k=3, exclude_seen=False
    )
    assert predictions.cols.shape == (2, 3)
    assert predictions.shape == (2, N_ITEMS)
    for row in predictions.vals:
        assert torch.equal(row, torch.sort(row, descending=True).values)


def test_no_rows_at_all():
    trainer = _fitted_on_cycle(epochs=1)
    predictions = trainer.predict_on_batch(_seqs([]), k=2, exclude_seen=False)
    assert predictions.cols.shape == (0, 2)


def test_an_empty_history_is_answered_with_the_sample_start_prior():
    """The state of an all-padding row is the one every sample's first item --
    a history's first, or a window's -- was trained from, so an empty history
    gets the ranking of what starts a sample."""
    trainer = _fitted_on_cycle(epochs=200)
    predictions = trainer.predict_on_batch(
        _seqs([[], [0, 1]]), k=2, exclude_seen=False
    )
    assert predictions.cols.shape == (2, 2)


def test_it_predicts_the_successor_of_the_last_item():
    trainer = _fitted_on_cycle(epochs=200)
    predictions = trainer.predict_on_batch(
        _seqs([[0, 1, 2], [2, 3, 4]]), k=1, exclude_seen=False
    )
    assert predictions.cols[:, 0].tolist() == [3, 5]


def test_prediction_reads_the_final_position():
    trainer = _fitted_on_cycle(epochs=2)
    source = _seqs([[0, 1, 2]])
    tokens, _ = trainer.batcher.encode(source)
    trainer.model.eval()
    with torch.no_grad():
        expected = trainer.model.score(trainer.model(tokens)[:, -1])[
            :, N_RESERVED:
        ]
    predictions = trainer.predict_on_batch(source, k=N_ITEMS, exclude_seen=False)
    assert torch.allclose(
        predictions.vals[0], expected[0, predictions.cols[0]], atol=1e-6
    )


def test_prediction_is_batching_invariant():
    trainer = _fitted_on_cycle(epochs=2)
    rows = [[0, 1, 2], [3, 4, 5, 6, 7], [2]]
    together = trainer.predict_on_batch(_seqs(rows), k=2, exclude_seen=False)
    apart = [
        trainer.predict_on_batch(_seqs([row]), k=2, exclude_seen=False)
        for row in rows
    ]
    for index, single in enumerate(apart):
        assert torch.equal(together.cols[index], single.cols[0])
        assert torch.allclose(together.vals[index], single.vals[0], atol=1e-5)


def test_a_history_longer_than_the_window_is_truncated_not_refused():
    trainer = _fitted_on_cycle(epochs=1)
    long_row = [(index % N_ITEMS) for index in range(30)]
    predictions = trainer.predict_on_batch(
        _seqs([long_row]), k=2, exclude_seen=False
    )
    assert predictions.cols.shape == (1, 2)


def test_candidate_ids_restrict_the_ranking():
    trainer = LRURecTrainer(_config(epochs=2)).fit(
        _seqs(_cycle_rows()), item_ids=[f"item-{index}" for index in range(N_ITEMS)]
    )
    predictions = trainer.predict_on_batch(
        _seqs([[0, 1, 2]]),
        k=2,
        exclude_seen=False,
        candidate_ids=["item-4", "item-5", "item-6"],
    )
    assert set(predictions.cols[0].tolist()) <= {4, 5, 6}


# --------------------------------------------------------------------------
# exclude_seen
# --------------------------------------------------------------------------


def test_exclude_seen_masks_the_whole_history_including_repeats():
    trainer = _fitted_on_cycle(epochs=2)
    predictions = trainer.predict_on_batch(
        _seqs([[0, 1, 1, 2]]), k=5, exclude_seen=True
    )
    assert not ({0, 1, 2} & set(predictions.cols[0].tolist()))


def test_exclude_seen_masks_what_truncation_dropped():
    """Stricter than the reference, which excludes only the encoded window."""
    trainer = _fitted_on_cycle(epochs=2, max_history_length=4)
    row = [0, 1, 2, 3, 4, 5]
    predictions = trainer.predict_on_batch(_seqs([row]), k=2, exclude_seen=True)
    assert not (set(row) & set(predictions.cols[0].tolist()))


def test_exclude_seen_refuses_when_fewer_than_k_unseen_items_remain():
    trainer = _fitted_on_cycle(epochs=1)
    with pytest.raises(ValueError):
        trainer.predict_on_batch(
            _seqs([list(range(N_ITEMS - 1))]), k=3, exclude_seen=True
        )


def test_exclude_seen_false_may_return_the_history():
    trainer = _fitted_on_cycle(epochs=2)
    predictions = trainer.predict_on_batch(
        _seqs([[0, 1, 2]]), k=N_ITEMS, exclude_seen=False
    )
    assert {0, 1, 2} <= set(predictions.cols[0].tolist())


def test_masking_does_not_mutate_the_source():
    trainer = _fitted_on_cycle(epochs=1)
    source = _seqs([[0, 1, 2]])
    before = np.array(source.values, copy=True)
    trainer.predict_on_batch(source, k=2, exclude_seen=True)
    assert np.array_equal(np.array(source.values), before)


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_a_reloaded_model_predicts_identically(tmp_path):
    """The complex projections have to survive weights_only loading."""
    trainer = _fitted_on_cycle(epochs=2)
    source = _seqs([[0, 1, 2], [3, 4, 5]])
    before = trainer.predict_on_batch(source, k=3, exclude_seen=False)
    path = tmp_path / "lrurec.ckpt"
    trainer.save(path)
    restored = LRURecTrainer.load(path)
    after = restored.predict_on_batch(source, k=3, exclude_seen=False)
    assert restored.model.blocks[0].lru_layer.in_proj.weight.dtype == torch.cfloat
    assert torch.equal(before.cols, after.cols)
    assert torch.allclose(before.vals, after.vals)


def test_the_optimizer_state_round_trips(tmp_path):
    """AdamW's moments for a complex parameter are complex too."""
    trainer = _fitted_on_cycle(epochs=1)
    path = tmp_path / "lrurec.ckpt"
    trainer.save(path, include_optimizer=True)
    restored = LRURecTrainer.load(path, load_optimizer=True)
    saved = trainer.optimizer.state_dict()["state"]
    loaded = restored.optimizer.state_dict()["state"]
    assert saved.keys() == loaded.keys()
    for key in saved:
        assert torch.equal(saved[key]["exp_avg"], loaded[key]["exp_avg"])


def test_a_reloaded_model_keeps_the_window_and_the_padding_side(tmp_path):
    trainer = _fitted_on_cycle(epochs=1, max_history_length=5)
    path = tmp_path / "lrurec.ckpt"
    trainer.save(path)
    restored = LRURecTrainer.load(path)
    assert restored.batcher.max_length == 5
    assert restored.batcher.padding == "left"
    assert restored.cfg.max_history_length == 5


def test_a_reloaded_config_equals_the_saved_one(tmp_path):
    trainer = _fitted_on_cycle(epochs=1)
    path = tmp_path / "lrurec.ckpt"
    trainer.save(path)
    assert LRURecTrainer.load(path).cfg == trainer.cfg


def test_a_reloaded_model_keeps_the_history(tmp_path):
    trainer = _fitted_on_cycle(epochs=2)
    path = tmp_path / "lrurec.ckpt"
    trainer.save(path)
    assert LRURecTrainer.load(path).history == trainer.history


def test_a_reloaded_model_can_be_fitted_again(tmp_path):
    trainer = _fitted_on_cycle(epochs=1)
    path = tmp_path / "lrurec.ckpt"
    trainer.save(path)
    restored = LRURecTrainer.load(path)
    restored.fit(_seqs(_cycle_rows()))
    assert len(restored.history) == 1
    assert restored.is_fitted


def test_the_vocabulary_survives_the_round_trip(tmp_path):
    item_ids = [f"item-{index}" for index in range(N_ITEMS)]
    batcher = SequenceBatcher(_tokenizer(item_ids=item_ids), max_length=6)
    trainer = LRURecTrainer(_config(max_history_length=6), batcher).fit(
        _seqs(_cycle_rows()), item_ids=item_ids
    )
    path = tmp_path / "lrurec.ckpt"
    trainer.save(path)
    restored = LRURecTrainer.load(path)
    assert list(restored.batcher.tokenizer.item_ids) == item_ids


# --------------------------------------------------------------------------
# the package's entry points
# --------------------------------------------------------------------------


def test_it_evaluates_through_the_standard_entry_point():
    """A sequential model and a matrix model meet the same evaluation API."""
    from scipy.sparse import csr_matrix

    from compresso_recsys.evaluation import evaluate_recommender
    from compresso_recsys.metrics import NDCG

    trainer = _fitted_on_cycle(epochs=200)
    sources = _seqs([[0, 1, 2], [3, 4, 5], [6, 7, 0]])
    wanted = [3, 6, 1]
    targets = csr_matrix(
        (
            np.ones(len(wanted), dtype=np.float32),
            (np.arange(len(wanted)), np.array(wanted)),
        ),
        shape=(len(wanted), N_ITEMS),
    )
    result = evaluate_recommender(
        trainer, source=sources, targets=targets, metrics=[NDCG(1)]
    )
    assert result.n_scored_rows == 3
    assert result["ndcg@1"] == pytest.approx(1.0)


def test_it_evaluates_against_sampled_negatives():
    """§5.3's protocol: each target ranked against uniformly drawn items. A
    target drawn among its own negatives ties itself and loses the tie."""
    from scipy.sparse import csr_matrix

    from compresso_recsys.evaluation import (
        _sampled_negatives,
        evaluate_sampled_recommender,
    )
    from compresso_recsys.metrics import Recall

    trainer = _fitted_on_cycle(epochs=200)
    sources = _seqs([[0, 1, 2], [3, 4, 5], [6, 7, 0]])
    wanted = np.array([3, 6, 1])
    targets = csr_matrix(
        (np.ones(3, dtype=np.float32), (np.arange(3), wanted)), shape=(3, N_ITEMS)
    )
    result = evaluate_sampled_recommender(
        trainer, source=sources, targets=targets, metrics=[Recall(1)], n_negatives=4
    )
    drawn = _sampled_negatives(0, range(3), N_ITEMS, 4)
    expected = ~(drawn == wanted[:, None]).any(axis=1)
    np.testing.assert_array_equal(result.per_user["recall@1"], expected.astype(np.float32))


def test_score_candidates_are_the_logits_prediction_ranks_by():
    trainer = _fitted_on_cycle(epochs=5)
    # An empty history and one longer than the window, which is truncated.
    source = _seqs([[0, 1, 2], [], [5, 6, 7, 0, 1, 2, 3, 4, 5, 6]])
    scores = trainer.score_candidates(source, np.tile(np.arange(N_ITEMS), (3, 1)))
    predictions = trainer.predict_on_batch(source, k=N_ITEMS, exclude_seen=False)
    assert torch.allclose(scores.gather(1, predictions.cols), predictions.vals)


def test_score_candidates_strikes_nothing_out():
    """The sampled protocol does not exclude history, so neither does this."""
    trainer = _fitted_on_cycle(epochs=5)
    scores = trainer.score_candidates(_seqs([[0, 1, 2]]), np.array([[0, 1, 2]]))
    assert torch.isfinite(scores).all()


def test_score_candidates_gives_items_beyond_the_catalog_minus_infinity():
    trainer = _fitted_on_cycle(epochs=1)
    source = ItemSequences.from_rows([[0, 1]], n_items=N_ITEMS + 2)
    scores = trainer.score_candidates(
        source, np.array([[0, N_ITEMS, N_ITEMS + 1, 3]])
    )[0]
    assert torch.isfinite(scores[[0, 3]]).all()
    assert scores[1] == -torch.inf and scores[2] == -torch.inf


@pytest.mark.parametrize(
    "candidates,error,message",
    [
        (np.zeros((2, 3), dtype=np.int64), ValueError, "one row per history"),
        (np.zeros(3, dtype=np.int64), ValueError, "one row per history"),
        (np.array([[-1, 0]]), ValueError, "non-negative"),
        (np.array([[0.0, 1.0]]), TypeError, "integer"),
    ],
)
def test_score_candidates_refuses_malformed_candidates(candidates, error, message):
    trainer = _fitted_on_cycle(epochs=1)
    with pytest.raises(error, match=message):
        trainer.score_candidates(_seqs([[0, 1]]), candidates)


def test_score_candidates_refuses_before_fitting():
    with pytest.raises(RuntimeError, match="fitted"):
        LRURecTrainer(_config()).score_candidates(
            _seqs([[0]]), np.zeros((1, 1), dtype=np.int64)
        )


def test_the_trainer_is_a_candidate_scorer():
    from compresso_recsys.models import CandidateScorer

    assert isinstance(LRURecTrainer(), CandidateScorer)


def test_recommend_round_trips_real_item_ids():
    item_ids = [f"item-{index}" for index in range(N_ITEMS)]
    batcher = SequenceBatcher(_tokenizer(item_ids=item_ids), max_length=6)
    trainer = LRURecTrainer(
        _config(epochs=200, lr=0.01, batch_size=48, max_history_length=6), batcher
    ).fit(_seqs(_cycle_rows()), item_ids=item_ids)

    recommendations = trainer.recommend(
        [["item-0", "item-1", "item-2"]], k=3, exclude_seen=True
    )

    assert list(recommendations.item_ids[0])[0] == "item-3"
    assert not {"item-0", "item-1", "item-2"} & set(recommendations.item_ids[0])


def test_item_ids_disagreeing_with_the_tokenizer_are_refused():
    item_ids = [f"item-{index}" for index in range(N_ITEMS)]
    batcher = SequenceBatcher(_tokenizer(item_ids=item_ids), max_length=6)
    trainer = LRURecTrainer(_config(max_history_length=6), batcher)
    with pytest.raises(ValueError, match="must match the batcher tokenizer"):
        trainer.fit(_seqs(_cycle_rows()), item_ids=[f"x{i}" for i in range(N_ITEMS)])
