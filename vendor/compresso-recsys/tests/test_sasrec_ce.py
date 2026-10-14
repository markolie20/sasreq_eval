# Modified for seqrec_eval: differs from upstream compresso-recsys; see vendor/compresso-recsys/VENDORED.md
"""Tests for SASRecCE: SASRec's network under a full softmax, with augmentation.

Organized like ``tests/test_sasrec.py``, but only where the two differ. The
network is a copy of SASRec's, so one test pins it to the original; the rest
cover what is new -- the objective, the history exclusion, how a batch is laid
out, and the two augmentation stages.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from compresso_recsys.models import (
    ItemTokenizer,
    SASRec,
    SASRecCE,
    SASRecCEConfig,
    SASRecCETrainer,
    SASRecConfig,
    SequenceBatcher,
)
from compresso_recsys.models.sasrec_ce.augmentation import (
    AugmentedHistory,
    CooccurrenceSampler,
    insert_cooccurring,
    replace_inputs,
    replace_with_cooccurring,
)
from compresso_recsys.sequences import ItemSequences

N_ITEMS = 8
# The default vocabulary: {"pad": 0, "mask": 1, "unk": 2}.
PAD_ID, MASK_ID, UNK_ID = 0, 1, 2
N_RESERVED = 3


def _seqs(rows, n_items=N_ITEMS):
    return ItemSequences.from_rows(rows, n_items=n_items)


def _cycle_rows(n_rows=48, length=4, n_items=N_ITEMS):
    """Histories drawn from a single cycle, so ``next(i) == (i + 1) % n_items``."""
    return [
        [(start + step) % n_items for step in range(length)]
        for start in range(n_rows)
    ]


def _token(item):
    return item + N_RESERVED


def _rng(seed=0):
    return np.random.default_rng(seed)


def _config(**overrides):
    """A small model with masking off, so each test names the augmentation it
    exercises rather than inheriting one."""
    defaults = dict(
        d_model=16,
        n_blocks=2,
        n_heads=2,
        max_history_length=6,
        mask_probability=0.0,
        epochs=1,
        batch_size=16,
        show_progress=False,
        seed=0,
    )
    return SASRecCEConfig(**{**defaults, **overrides})


def _ready(trainer_class=SASRecCETrainer, batcher=None, **overrides):
    """A trainer with its batcher and model built, for calling the internals."""
    return trainer_class(_config(**overrides), batcher).fit(_seqs(_cycle_rows(8)))


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(d_model=0), "d_model must be >= 1"),
        (dict(max_history_length=0), "max_history_length must be >= 1"),
        (dict(d_model=10, n_heads=4), "divisible by n_heads"),
        (dict(dropout=1.0), r"dropout must be in \[0, 1\)"),
        (dict(mask_probability=1.0), r"mask_probability must be in \[0, 1\)"),
        (dict(mask_probability=-0.1), r"mask_probability must be in \[0, 1\)"),
        (dict(unk_dropout=1.0), r"unk_dropout must be in \[0, 1\)"),
        (
            dict(cooccurrence_insert_probability=1.0),
            r"cooccurrence_insert_probability must be in \[0, 1\)",
        ),
        (
            dict(cooccurrence_replace_probability=-0.1),
            r"cooccurrence_replace_probability must be in \[0, 1\)",
        ),
        (dict(cooccurrence_window=0), "cooccurrence_window must be >= 1"),
        (dict(cooccurrence_top_k=0), "cooccurrence_top_k must be >= 1"),
        (
            dict(cooccurrence_popularity_exponent=float("inf")),
            "cooccurrence_popularity_exponent must be finite",
        ),
        (dict(n_negatives=0), "n_negatives must be None"),
        (dict(n_negatives=True), "n_negatives must be None"),
        (dict(n_negatives=2.5), "n_negatives must be None"),
        (dict(allow_history_negatives=1), "allow_history_negatives must be a bool"),
        (dict(score_inserted_items=0), "score_inserted_items must be a bool"),
        (dict(lr=0.0), "lr must be > 0"),
        (dict(optimizer="NAdam"), "optimizer must be 'Adam'"),
        (dict(betas=(0.9, 1.0)), r"betas must each be in \[0, 1\)"),
    ],
)
def test_invalid_configuration_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _config(**kwargs)


def test_the_shared_defaults_are_sasrecs():
    """So that a comparison of the two trainers moves the objective and the
    augmentation and nothing else."""
    ours = {field.name for field in dataclasses.fields(SASRecCEConfig)}
    theirs = {field.name for field in dataclasses.fields(SASRecConfig)}
    # Memory bounds are not hyperparameters, and a SASRec that has one sizes it
    # for its own loss: they change no result, so they are not compared.
    memory = {"loss_chunk_elements", "prediction_chunk_elements"}
    for name in (ours & theirs) - {"log_prefix", "n_negatives"} - memory:
        assert getattr(SASRecCEConfig(), name) == getattr(SASRecConfig(), name), name
    # The one shared field that differs on purpose: SASRec samples one
    # negative, this model defaults to the full softmax.
    assert SASRecCEConfig().n_negatives is None


def test_the_defaults_exclude_history_and_mask_but_leave_cooccurrence_off():
    cfg = SASRecCEConfig()
    assert cfg.allow_history_negatives is False
    assert cfg.mask_probability == 0.2
    assert cfg.unk_dropout == 0.0
    assert cfg.cooccurrence_insert_probability == 0.0
    assert cfg.cooccurrence_replace_probability == 0.0
    assert cfg.score_inserted_items is False
    assert cfg.cooccurrence_popularity_exponent == 0.0
    assert not cfg.uses_cooccurrence


def test_config_survives_a_json_round_trip():
    cfg = SASRecCEConfig(
        n_negatives=500,
        allow_history_negatives=True,
        mask_probability=0.3,
        cooccurrence_insert_probability=0.1,
        cooccurrence_popularity_exponent=-0.5,
    )
    restored = SASRecCEConfig(**json.loads(json.dumps(dataclasses.asdict(cfg))))
    assert restored == cfg


# --------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------


def test_the_network_is_sasrecs():
    """A copy rather than an import, so it is pinned here: SASRec's weights load
    into it by name and compute the same states and scores."""
    arguments = dict(
        n_items=N_ITEMS,
        n_reserved=N_RESERVED,
        max_history_length=6,
        pad_id=PAD_ID,
        d_model=16,
        n_blocks=2,
        n_heads=2,
        dropout=0.0,
    )
    original = SASRec(**arguments).eval()
    copy = SASRecCE(**arguments).eval()
    copy.load_state_dict(original.state_dict())

    history = torch.tensor([[0, 0, 3, 4, 5, 6], [7, 8, 9, 10, 3, 4]])
    torch.testing.assert_close(copy(history), original(history))
    torch.testing.assert_close(
        copy.score(copy(history)), original.score(original(history))
    )


def test_reserved_tokens_are_embedded_but_never_scored():
    model = _ready().model
    assert model.item_embedding.num_embeddings == N_RESERVED + N_ITEMS
    assert model.score(torch.zeros(2, 16)).shape == (2, N_ITEMS)


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------


def test_the_default_vocabulary_names_pad_mask_and_unk():
    trainer = _ready()
    assert dict(trainer.batcher.tokenizer.special_tokens) == {
        "pad": PAD_ID,
        "mask": MASK_ID,
        "unk": UNK_ID,
    }


def test_masking_without_a_mask_token_is_refused():
    """Masking is on by default, so the default config refuses this too."""
    batcher = SequenceBatcher(ItemTokenizer(N_ITEMS))
    with pytest.raises(ValueError, match="'mask' special token"):
        SASRecCETrainer(SASRecCEConfig(show_progress=False), batcher).fit(
            _seqs(_cycle_rows(8))
        )


def test_a_vocabulary_without_mask_serves_when_masking_is_off():
    batcher = SequenceBatcher(ItemTokenizer(N_ITEMS))
    trainer = SASRecCETrainer(
        _config(mask_probability=0.0, unk_dropout=0.2), batcher
    ).fit(_seqs(_cycle_rows(8)))
    assert trainer.is_fitted


# --------------------------------------------------------------------------
# batch layout
# --------------------------------------------------------------------------


def test_a_batch_is_shifted_and_left_padded_to_the_window():
    trainer = _ready(max_history_length=4)
    inputs, targets = trainer._training_batch(
        _seqs([[0, 1, 2, 3, 4, 5], [6, 7], [3]]), _rng()
    )
    # A history that outgrows the window keeps window + 1 interactions, so
    # every position the model owns holds an input.
    assert inputs[0].tolist() == [_token(i) for i in (1, 2, 3, 4)]
    assert targets[0].tolist() == [_token(i) for i in (2, 3, 4, 5)]
    # A short one is padded on the left, and no target sits on a pad input.
    assert inputs[1].tolist() == [PAD_ID, PAD_ID, PAD_ID, _token(6)]
    assert targets[1].tolist() == [PAD_ID, PAD_ID, PAD_ID, _token(7)]
    # One interaction is no example at all.
    assert inputs[2].tolist() == [PAD_ID] * 4
    assert targets[2].tolist() == [PAD_ID] * 4


def test_masking_replaces_real_inputs_and_never_a_target():
    rows = _seqs(_cycle_rows(64, length=5))
    clean_inputs, clean_targets = _ready()._training_batch(rows, _rng())
    inputs, targets = _ready(mask_probability=0.5)._training_batch(rows, _rng())

    assert torch.equal(targets, clean_targets)
    changed = inputs != clean_inputs
    assert (inputs[changed] == MASK_ID).all()
    assert (clean_inputs[changed] != PAD_ID).all()
    real = clean_inputs != PAD_ID
    assert float(changed.sum() / real.sum()) == pytest.approx(0.5, abs=0.08)


def test_unk_is_drawn_over_what_masking_left():
    rows = _seqs(_cycle_rows(64, length=5))
    clean_inputs, clean_targets = _ready()._training_batch(rows, _rng())
    inputs, targets = _ready(mask_probability=0.5, unk_dropout=0.5)._training_batch(
        rows, _rng()
    )
    assert torch.equal(targets, clean_targets)
    real = clean_inputs != PAD_ID
    masked = inputs == MASK_ID
    unknown = inputs == UNK_ID
    assert not (masked & unknown).any()
    assert not ((masked | unknown) & ~real).any()
    # Half of the half masking left alone.
    assert float(unknown.sum() / real.sum()) == pytest.approx(0.25, abs=0.08)


def test_zero_rates_draw_no_random_numbers():
    """So a batch is exactly the one a config without augmentation would have
    produced from the same seed."""
    trainer = _ready()
    rng = _rng()
    before = rng.bit_generator.state
    trainer._training_batch(_seqs(_cycle_rows(8)), rng)
    assert rng.bit_generator.state == before


class _InsertAfterEach(SASRecCETrainer):
    """A sequence edit for the hook: item 7 after every interaction, unobserved."""

    INSERTED = 7

    def _augment_history(self, history, rng):
        items = np.repeat(history.items, 2)
        items[1::2] = self.INSERTED
        observed = np.tile([True, False], history.items.size)
        return AugmentedHistory(items, observed)


def test_sequence_edits_are_read_but_never_scored():
    trainer = _ready(_InsertAfterEach, max_history_length=4)
    inputs, targets = trainer._training_batch(_seqs([[0, 1, 2]]), _rng())
    # 0 X 1 X 2 X, cut back to window + 1 after the edit lengthened it.
    x = _token(_InsertAfterEach.INSERTED)
    assert inputs[0].tolist() == [x, _token(1), x, _token(2)]
    assert targets[0].tolist() == [_token(1), PAD_ID, _token(2), PAD_ID]


def test_augmented_history_keeps_items_and_flags_aligned():
    with pytest.raises(ValueError, match="aligned"):
        AugmentedHistory(np.arange(3), np.ones(2, dtype=bool))
    with pytest.raises(TypeError, match="bool"):
        AugmentedHistory(np.arange(3), np.ones(3, dtype=np.int64))
    history = AugmentedHistory.observed_history(np.arange(5))
    assert history.observed.all()
    assert history.tail(2).items.tolist() == [3, 4]


def test_replace_inputs_touches_only_eligible_positions():
    inputs = np.arange(12).reshape(3, 4)
    eligible = np.zeros((3, 4), dtype=bool)
    eligible[1] = True
    chosen = replace_inputs(inputs, eligible, 0.99, -1, _rng())
    assert not chosen[~eligible].any()
    assert (inputs[chosen] == -1).all()
    assert inputs[0].tolist() == [0, 1, 2, 3]


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------


def _history_excluded(batch, rows, labels):
    excluded = torch.zeros((len(rows), N_ITEMS), dtype=torch.bool)
    for position, (row, label) in enumerate(zip(rows, labels)):
        for item in set(batch.row(row).tolist()) - {label}:
            excluded[position, item] = True
    return excluded


@pytest.mark.parametrize("allow", [True, False])
def test_the_loss_is_cross_entropy_over_the_catalog(allow):
    trainer = _ready(dropout=0.0, allow_history_negatives=allow)
    batch = _seqs([[0, 1, 2, 3], [4, 5], [6]])
    inputs, targets = trainer._training_batch(batch, _rng())
    labelled = targets != PAD_ID
    with torch.no_grad():
        logits = trainer.model.score(trainer.model(inputs)[labelled])
    labels = targets[labelled] - N_RESERVED
    assert logits.shape == (4, N_ITEMS)
    if not allow:
        rows = labelled.nonzero(as_tuple=True)[0].tolist()
        excluded = _history_excluded(batch, rows, labels.tolist())
        logits = logits.masked_fill(excluded, -torch.inf)
    expected = F.cross_entropy(logits, labels)

    loss, positions = trainer._train_step(batch, _rng())
    assert positions == 4
    assert loss == pytest.approx(float(expected), rel=1e-5)


def test_excluding_history_covers_the_whole_sequence_but_not_the_target():
    trainer = _ready(max_history_length=2, allow_history_negatives=False)
    # Row 0 outgrows the window: 0 and 1 are truncated away and still count,
    # and 5 comes after position 0's target and still counts.
    batch = _seqs([[0, 1, 2, 3, 4, 5], [6, 3, 6]])
    rows = torch.tensor([0, 0, 1])
    labels = torch.tensor([4, 5, 6])
    out = trainer._without_history_negatives(
        torch.zeros((3, N_ITEMS)), labels, trainer._history_table(batch)[rows]
    )
    excluded = torch.isinf(out)
    assert excluded[0].nonzero().flatten().tolist() == [0, 1, 2, 3, 5]
    assert excluded[1].nonzero().flatten().tolist() == [0, 1, 2, 3, 4]
    # The target repeats in its own history and keeps its logit regardless.
    assert excluded[2].nonzero().flatten().tolist() == [3]


def test_excluded_items_receive_no_gradient():
    trainer = _ready(allow_history_negatives=False)
    batch = _seqs([[0, 1, 2]])
    labels = torch.tensor([1, 2])
    logits = torch.randn(2, N_ITEMS, requires_grad=True)
    out = trainer._without_history_negatives(
        logits, labels, trainer._history_table(batch)[torch.tensor([0, 0])]
    )
    F.cross_entropy(out, labels).backward()
    excluded = torch.isinf(out)
    assert (logits.grad[excluded] == 0).all()
    assert (logits.grad[~excluded] != 0).all()


def test_a_history_covering_the_catalog_trains_without_nan():
    trainer = _ready(allow_history_negatives=False)
    loss, positions = trainer._train_step(_seqs([list(range(N_ITEMS))]), _rng())
    assert positions == 6
    # Each softmax holds only its own target.
    assert loss == pytest.approx(0.0, abs=1e-6)


def test_allowing_history_negatives_never_builds_the_exclusion(monkeypatch):
    trainer = _ready(allow_history_negatives=True)

    def explode(*args, **kwargs):
        raise AssertionError("the plain softmax excludes nothing")

    monkeypatch.setattr(trainer, "_history_table", explode)
    monkeypatch.setattr(trainer, "_without_history_negatives", explode)
    assert trainer._train_step(_seqs(_cycle_rows(8)), _rng()) is not None


# --------------------------------------------------------------------------
# loss_chunk_elements: memory only
# --------------------------------------------------------------------------


def _one_step(trainer, batch, chunk):
    """One step's loss, positions and every gradient; the parameters stay put."""
    trainer.cfg = dataclasses.replace(trainer.cfg, loss_chunk_elements=chunk)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
    torch.manual_seed(1)  # the same dropout draw for every chunk size
    loss, positions = trainer._train_step(batch, _rng(3))
    return loss, positions, [p.grad.clone() for p in trainer.model.parameters()]


@pytest.mark.parametrize("allow", [True, False])
@pytest.mark.parametrize("n_negatives", [None, 4], ids=["full", "sampled"])
def test_a_chunked_loss_is_the_same_loss_and_gradient(n_negatives, allow):
    # Dropout and masking on, so their draws have to be the same in every
    # chunking for the gradients to match; negatives are drawn once per batch.
    trainer = _ready(
        dropout=0.3,
        mask_probability=0.2,
        n_negatives=n_negatives,
        allow_history_negatives=allow,
    )
    batch = _seqs(_cycle_rows(n_rows=12, length=6))
    per_position = N_ITEMS if n_negatives is None else n_negatives * 16
    whole_loss, positions, whole = _one_step(trainer, batch, 2**27)
    assert positions > 7
    # per_position elements is one position per chunk: the most chunks there are
    for chunk in (per_position, 7 * per_position):
        loss, _, grads = _one_step(trainer, batch, chunk)
        assert loss == pytest.approx(whole_loss, rel=1e-6)
        for a, b in zip(whole, grads):
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("n_negatives", [None, 4], ids=["full", "sampled"])
def test_a_chunk_holds_no_more_than_the_bound(monkeypatch, n_negatives):
    trainer = _ready(n_negatives=n_negatives)
    per_position = N_ITEMS if n_negatives is None else n_negatives * 16
    trainer.cfg = dataclasses.replace(
        trainer.cfg, loss_chunk_elements=3 * per_position
    )
    scored = []
    original = trainer.model.score_items if n_negatives else trainer.model.score

    def counting(states, *args):
        scored.append(states.shape[0])
        return original(states, *args)

    monkeypatch.setattr(
        trainer.model, "score_items" if n_negatives else "score", counting
    )
    trainer._train_step(_seqs(_cycle_rows(n_rows=12, length=6)), _rng())
    assert len(scored) > 1
    assert max(scored) <= 3


@pytest.mark.parametrize("name", ["loss_chunk_elements", "prediction_chunk_elements"])
def test_chunk_bounds_must_be_positive(name):
    with pytest.raises(ValueError, match=f"{name} must be >= 1"):
        _config(**{name: 0})


@pytest.mark.parametrize("exclude_seen", [True, False])
def test_prediction_in_parts_ranks_as_one_batch(exclude_seen):
    item_ids = [f"item-{index}" for index in range(N_ITEMS)]
    trainer = SASRecCETrainer(_config(epochs=3)).fit(
        _seqs(_cycle_rows()), item_ids=item_ids
    )
    # Repeats, a one-item history, and an empty one, so each part masks its
    # own rows' history and nobody else's.
    source = _seqs(_cycle_rows(n_rows=7, length=3) + [[1], [0, 0, 5, 2], []])
    asked = dict(k=3, exclude_seen=exclude_seen, candidate_ids=item_ids[1:])
    whole = trainer.predict_on_batch(source, **asked)
    trainer.cfg = dataclasses.replace(trainer.cfg, prediction_chunk_elements=1)
    parts = trainer.predict_on_batch(source, **asked)
    assert torch.equal(parts.cols, whole.cols)
    torch.testing.assert_close(parts.vals, whole.vals)


def test_prediction_holds_no_more_histories_than_the_bound(monkeypatch):
    trainer = _ready()
    # Two heads over a window of 6 is a 72-element map per history, which is
    # more than the 8 scores, so the bound is three histories at a time.
    trainer.cfg = dataclasses.replace(trainer.cfg, prediction_chunk_elements=3 * 72)
    sizes = []
    original = trainer.model.forward

    def counting(history):
        sizes.append(history.shape[0])
        return original(history)

    monkeypatch.setattr(trainer.model, "forward", counting)
    trainer.predict_on_batch(_seqs(_cycle_rows(n_rows=10)), k=2)
    assert sizes == [3, 3, 3, 1]


# --------------------------------------------------------------------------
# sampled softmax
# --------------------------------------------------------------------------


@pytest.mark.parametrize("allow", [True, False])
def test_the_sampled_loss_is_cross_entropy_over_the_target_and_its_users_negatives(
    allow,
):
    # Sixteen draws over eight items: under allow, targets are drawn as their
    # own negatives, which the loss has to drop.
    trainer = _ready(dropout=0.0, n_negatives=16, allow_history_negatives=allow)
    batch = _seqs([[0, 1, 2, 3], [4, 5], [6]])
    # Every augmentation rate is zero, so the step's only draw is the
    # negatives, and the same seed reproduces them.
    inputs, targets = trainer._training_batch(batch, _rng())
    negatives, _ = trainer._sample_negatives(batch, _rng())

    expected, collisions = [], 0
    with torch.no_grad():
        states = trainer.model(inputs)
        weight = trainer.model.item_embedding.weight
        for row, step in (targets != PAD_ID).nonzero().tolist():
            target = int(targets[row, step])
            drawn = negatives[row].tolist()
            collisions += drawn.count(target)
            candidates = [target] + [token for token in drawn if token != target]
            logits = weight[candidates] @ states[row, step]
            expected.append(-torch.log_softmax(logits, dim=0)[0])
    assert (collisions > 0) == allow

    loss, positions = trainer._train_step(batch, _rng())
    assert positions == len(expected) == 4
    assert loss == pytest.approx(float(torch.stack(expected).mean()), rel=1e-5)


def test_negatives_are_drawn_once_per_user():
    trainer = _ready(n_negatives=5)
    negatives, drawable = trainer._sample_negatives(_seqs(_cycle_rows(3)), _rng())
    assert negatives.shape == (3, 5)
    assert drawable.tolist() == [True, True, True]


def test_negatives_are_uniform_outside_the_history_unless_allowed():
    # The short last row leaves several padding slots in the exclusion table,
    # which the mapping must never step over.
    batch = _seqs([[0, 1, 2], [5, 5, 6, 7], [3]])
    drawn = _ready(n_negatives=4000)._sample_negatives(batch, _rng())[0]
    drawn = (drawn - N_RESERVED).numpy()
    assert set(drawn[0].tolist()) == {3, 4, 5, 6, 7}
    # A repeated item is one exclusion, not two.
    assert set(drawn[1].tolist()) == {0, 1, 2, 3, 4}
    assert set(drawn[2].tolist()) == {0, 1, 2, 4, 5, 6, 7}
    for row in drawn:
        counts = np.unique(row, return_counts=True)[1]
        np.testing.assert_allclose(counts / row.size, 1 / counts.size, atol=0.03)

    allowed = _ready(n_negatives=4000, allow_history_negatives=True)
    drawn = allowed._sample_negatives(batch, _rng())[0] - N_RESERVED
    assert set(drawn[0].tolist()) == set(range(N_ITEMS))


def test_a_user_with_nothing_left_to_draw_trains_without_nan():
    trainer = _ready(n_negatives=4)
    negatives, drawable = trainer._sample_negatives(
        _seqs([list(range(N_ITEMS)), [0, 1]]), _rng()
    )
    assert drawable.tolist() == [False, True]
    # Still valid catalog tokens, so the embedding lookup stays in range.
    assert ((negatives >= N_RESERVED) & (negatives < N_RESERVED + N_ITEMS)).all()
    loss, _ = trainer._train_step(_seqs([list(range(N_ITEMS))]), _rng())
    assert loss == pytest.approx(0.0, abs=1e-6)


def test_the_sampled_softmax_never_scores_the_catalog(monkeypatch):
    trainer = _ready(n_negatives=4)

    def explode(*args, **kwargs):
        raise AssertionError("the sampled softmax scores its negatives only")

    monkeypatch.setattr(trainer.model, "score", explode)
    assert trainer._train_step(_seqs(_cycle_rows(8)), _rng()) is not None


# --------------------------------------------------------------------------
# co-occurrence
# --------------------------------------------------------------------------


def _sampler(rows, n_items=N_ITEMS, *, window=1, top_k=20, exponent=0.0):
    return CooccurrenceSampler.from_sequences(
        _seqs(rows, n_items),
        window=window,
        top_k=top_k,
        popularity_exponent=exponent,
    )


def _partners(sampler, item):
    return set(sampler.neighbors_of(item)[0].tolist())


def test_cooccurrence_is_counted_within_the_window_and_within_a_user():
    sampler = _sampler([[0, 1, 2], [3, 4]], window=1)
    assert [_partners(sampler, item) for item in range(N_ITEMS)] == [
        {1}, {0, 2}, {1}, {4}, {3}, set(), set(), set()
    ]
    # A wider window reaches further, and still never across users.
    wider = _sampler([[0, 1, 2], [3, 4]], window=2)
    assert _partners(wider, 0) == {1, 2}
    assert _partners(wider, 2) == {0, 1}


def test_a_repeated_item_is_not_its_own_partner():
    assert _partners(_sampler([[0, 0, 1]]), 0) == {1}


def test_partners_are_weighted_by_how_often_they_co_occur():
    # 0 sits next to 1 twice and next to 2 once.
    neighbors, probabilities = _sampler([[1, 0, 1], [2, 0]]).neighbors_of(0)
    assert neighbors.tolist() == [1, 2]
    np.testing.assert_allclose(probabilities, [2 / 3, 1 / 3])


# 0 co-occurs once each with 1 and 2, and 1 is four times as popular as 2.
SKEWED = [[1, 0, 2], [1], [1], [1]]


@pytest.mark.parametrize(
    "exponent,expected",
    [
        (1.0, {1: 0.8, 2: 0.2}),
        (0.0, {1: 0.5, 2: 0.5}),
        # count / popularity: the partner's popularity cancelled out.
        (-1.0, {1: 0.2, 2: 0.8}),
    ],
)
def test_the_exponent_boosts_popular_or_tail_partners(exponent, expected):
    sampler = _sampler(SKEWED, n_items=3, exponent=exponent)
    neighbors, probabilities = sampler.neighbors_of(0)
    assert dict(zip(neighbors.tolist(), probabilities.tolist())) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("exponent,kept", [(1.0, 1), (-1.0, 2)])
def test_top_k_keeps_the_partners_the_exponent_favours(exponent, kept):
    sampler = _sampler(SKEWED, n_items=3, top_k=1, exponent=exponent)
    assert sampler.neighbors_of(0)[0].tolist() == [kept]
    assert sampler.neighbors_of(0)[1].tolist() == [1.0]


def test_draws_follow_the_probabilities_and_stay_in_their_row():
    sampler = _sampler(SKEWED + [[3, 4, 5]], n_items=6, exponent=1.0)
    rng = _rng()
    draws = sampler.sample(np.zeros(20_000, dtype=np.int64), rng)
    assert float(np.mean(draws == 1)) == pytest.approx(0.8, abs=0.02)
    mixed = np.array([0, 4, 3, 5, 0, 2] * 200)
    for item, partner in zip(mixed.tolist(), sampler.sample(mixed, rng).tolist()):
        assert partner in _partners(sampler, item)


def test_an_item_without_partners_cannot_be_drawn_for():
    sampler = _sampler([[0, 1]])
    assert sampler.has_neighbors(np.array([0, 1, 2])).tolist() == [True, True, False]
    with pytest.raises(ValueError, match="co-occurred with nothing"):
        sampler.sample(np.array([2]), _rng())


def test_insertion_follows_every_anchor_but_the_last_and_is_unobserved():
    # 7 is the only partner of 0, 1 and 2.
    sampler = _sampler([[0, 7], [1, 7], [2, 7]])
    history = AugmentedHistory.observed_history(np.array([0, 1, 2]))
    edited = insert_cooccurring(history, sampler, 0.999, _rng())
    assert edited.items.tolist() == [0, 7, 1, 7, 2]
    assert edited.observed.tolist() == [True, False, True, False, True]


def test_insertion_skips_items_without_partners_and_draws_nothing_at_zero():
    sampler = _sampler([[0, 7]])
    history = AugmentedHistory.observed_history(np.array([0, 3, 0]))
    assert insert_cooccurring(history, sampler, 0.999, _rng()).items.tolist() == [
        0, 7, 3, 0
    ]
    rng = _rng()
    before = rng.bit_generator.state
    assert insert_cooccurring(history, sampler, 0.0, rng) is history
    assert rng.bit_generator.state == before


def test_replacement_swaps_catalog_inputs_for_partners_only():
    sampler = _sampler([[0, 7], [1, 7]])
    # Padding, a special token, two items with a partner, one without.
    inputs = np.array([[PAD_ID, MASK_ID, _token(0), _token(1), _token(3)]])
    chosen = replace_with_cooccurring(
        inputs, inputs != PAD_ID, 0.999, sampler, N_RESERVED, _rng()
    )
    assert chosen.tolist() == [[False, False, True, True, False]]
    assert inputs.tolist() == [[PAD_ID, MASK_ID, _token(7), _token(7), _token(3)]]


def test_the_table_is_built_only_when_a_cooccurrence_rate_is_on():
    assert _ready()._cooccurrence is None
    trainer = _ready(cooccurrence_replace_probability=0.1)
    assert isinstance(trainer._cooccurrence, CooccurrenceSampler)
    assert trainer._cooccurrence.n_items == N_ITEMS


@pytest.mark.parametrize("score", [False, True])
def test_inserted_items_are_read_and_scored_only_when_asked(score):
    trainer = _ready(cooccurrence_insert_probability=0.99, score_inserted_items=score)
    inputs, targets = trainer._training_batch(_seqs([[0, 1, 2]]), _rng())
    # 0 x 1 y 2: the inputs end 0 x 1 y and the targets x 1 y 2.
    x, y = int(inputs[0, -3]), int(inputs[0, -1])
    assert inputs[0, -4:].tolist() == [_token(0), x, _token(1), y]
    assert x - N_RESERVED in _partners(trainer._cooccurrence, 0)
    assert y - N_RESERVED in _partners(trainer._cooccurrence, 1)
    expected = [x, _token(1), y, _token(2)] if score else [
        PAD_ID, _token(1), PAD_ID, _token(2)
    ]
    assert targets[0, -4:].tolist() == expected


def test_replacement_is_drawn_over_what_masking_left_and_never_a_target():
    rows = _seqs(_cycle_rows(64, length=5))
    clean_inputs, clean_targets = _ready()._training_batch(rows, _rng())
    trainer = _ready(mask_probability=0.3, cooccurrence_replace_probability=0.5)
    inputs, targets = trainer._training_batch(rows, _rng())

    assert torch.equal(targets, clean_targets)
    real = clean_inputs != PAD_ID
    masked = inputs == MASK_ID
    # No item is its own partner, so every replacement shows as a change.
    replaced = (inputs != clean_inputs) & ~masked
    assert not ((masked | replaced) & ~real).any()
    for before, after in zip(
        clean_inputs[replaced].tolist(), inputs[replaced].tolist()
    ):
        assert after - N_RESERVED in _partners(
            trainer._cooccurrence, before - N_RESERVED
        )
    # Half of what masking left; every cycle item has a partner.
    assert float(replaced.sum() / real.sum()) == pytest.approx(0.35, abs=0.08)


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        dict(allow_history_negatives=True, mask_probability=0.0),
        # The real defaults for the two switches.
        dict(allow_history_negatives=False, mask_probability=0.2),
        dict(
            cooccurrence_insert_probability=0.2,
            cooccurrence_replace_probability=0.2,
            cooccurrence_popularity_exponent=-1.0,
            score_inserted_items=True,
        ),
        dict(n_negatives=3, allow_history_negatives=True),
        dict(n_negatives=3, mask_probability=0.2),
    ],
    ids=[
        "plain-softmax",
        "defaults",
        "cooccurrence",
        "sampled",
        "sampled-without-history",
    ],
)
def test_learns_a_cycle(overrides):
    trainer = SASRecCETrainer(
        _config(epochs=60, lr=0.01, batch_size=48, **overrides)
    ).fit(_seqs(_cycle_rows()))
    assert trainer.history[-1]["loss"] < trainer.history[0]["loss"]
    predicted = trainer.predict_on_batch(_seqs([[0, 1, 2], [3, 4], [6]]), k=1)
    assert predicted.cols.squeeze(1).tolist() == [3, 5, 7]


def test_fit_is_reproducible_from_the_seed():
    rows = _seqs(_cycle_rows())
    augmented = dict(
        epochs=2,
        mask_probability=0.3,
        cooccurrence_insert_probability=0.2,
        cooccurrence_replace_probability=0.2,
    )
    first = SASRecCETrainer(_config(**augmented)).fit(rows)
    second = SASRecCETrainer(_config(**augmented)).fit(rows)
    other = SASRecCETrainer(_config(**augmented, seed=1)).fit(rows)
    assert first.history == second.history
    assert first.history != other.history


def test_a_one_item_catalog_trains():
    """No negative has to be drawn, so nothing refuses it; there is just
    nothing to learn."""
    trainer = SASRecCETrainer(_config()).fit(_seqs([[0, 0, 0]], n_items=1))
    assert trainer.history[0]["loss"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "sequences,error,message",
    [
        (_seqs([]), ValueError, "zero sequences"),
        (_seqs([[1], [2]]), ValueError, "two or more interactions"),
    ],
)
def test_untrainable_input_is_refused(sequences, error, message):
    with pytest.raises(error, match=message):
        SASRecCETrainer(_config()).fit(sequences)


def test_fit_refuses_a_matrix():
    with pytest.raises(TypeError, match="ItemSequences"):
        SASRecCETrainer(_config()).fit(np.zeros((2, N_ITEMS)))


def test_a_batcher_inherits_the_window_and_is_left_padded():
    batcher = SequenceBatcher(ItemTokenizer(N_ITEMS, special_tokens={"pad": 0, "mask": 1}))
    trainer = SASRecCETrainer(_config(max_history_length=5), batcher).fit(
        _seqs(_cycle_rows(8))
    )
    assert trainer.batcher.max_length == 5
    assert trainer.batcher.padding == "left"


def test_a_batcher_naming_another_window_is_refused():
    batcher = SequenceBatcher(ItemTokenizer(N_ITEMS), max_length=3)
    with pytest.raises(ValueError, match="single owner"):
        SASRecCETrainer(_config(max_history_length=6), batcher).fit(
            _seqs(_cycle_rows(8))
        )


# --------------------------------------------------------------------------
# prediction and persistence
# --------------------------------------------------------------------------


def test_prediction_refuses_before_fitting():
    with pytest.raises(RuntimeError, match="fitted before predicting"):
        SASRecCETrainer(_config()).predict_on_batch(_seqs([[1, 2]]), k=2)


def test_candidate_ids_restrict_the_ranking():
    trainer = SASRecCETrainer(_config(epochs=2)).fit(
        _seqs(_cycle_rows()), item_ids=[f"item-{index}" for index in range(N_ITEMS)]
    )
    predictions = trainer.predict_on_batch(
        _seqs([[0, 1, 2]]),
        k=2,
        exclude_seen=False,
        candidate_ids=["item-4", "item-5", "item-6"],
    )
    assert set(predictions.cols[0].tolist()) <= {4, 5, 6}


def test_items_after_the_fitted_catalog_are_read_as_unk_and_never_recommended():
    trainer = _ready()
    # A later split stage: two items the model was never fitted on.
    later = ItemSequences.from_rows(
        [[0, N_ITEMS, 1, N_ITEMS + 1]], n_items=N_ITEMS + 2
    )
    tokens, _ = trainer.batcher.encode(later)
    assert tokens[0, -4:].tolist() == [_token(0), UNK_ID, _token(1), UNK_ID]
    predictions = trainer.predict_on_batch(later, k=3)
    assert predictions.shape == (1, N_ITEMS)
    assert not set(predictions.cols[0].tolist()) & {0, 1}


def test_recommend_round_trips_real_item_ids(tmp_path):
    item_ids = [f"item-{index}" for index in range(N_ITEMS)]
    trainer = SASRecCETrainer(
        _config(epochs=60, lr=0.01, batch_size=48)
    ).fit(_seqs(_cycle_rows()), item_ids=item_ids)
    history = [["item-0", "item-1", "item-2"]]
    recommendations = trainer.recommend(history, k=3, exclude_seen=True)
    assert list(recommendations.item_ids[0])[0] == "item-3"
    assert not {"item-0", "item-1", "item-2"} & set(recommendations.item_ids[0])

    trainer.save(tmp_path / "sasrec_ce.zip")
    restored = SASRecCETrainer.load(tmp_path / "sasrec_ce.zip")
    again = restored.recommend(history, k=3, exclude_seen=True)
    assert list(again.item_ids[0]) == list(recommendations.item_ids[0])


def test_prediction_excludes_the_whole_history():
    trainer = _ready()
    source = _seqs([[0, 1, 2, 3, 4]])
    cols = trainer.predict_on_batch(source, k=3).cols[0].tolist()
    assert not set(cols) & {0, 1, 2, 3, 4}


def test_a_round_trip_keeps_predictions_config_and_vocabulary(tmp_path):
    trainer = _ready(
        epochs=2,
        mask_probability=0.2,
        cooccurrence_insert_probability=0.2,
        cooccurrence_popularity_exponent=-0.5,
    )
    path = tmp_path / "sasrec_ce.zip"
    trainer.save(path)
    restored = SASRecCETrainer.load(path)

    assert restored.cfg == trainer.cfg
    # Training-only state: a further fit builds it again.
    assert restored._cooccurrence is None
    assert restored.history == trainer.history
    assert dict(restored.batcher.tokenizer.special_tokens) == dict(
        trainer.batcher.tokenizer.special_tokens
    )
    source = _seqs([[0, 1, 2], [5, 6]])
    expected = trainer.predict_on_batch(source, k=3)
    actual = restored.predict_on_batch(source, k=3)
    assert torch.equal(actual.cols, expected.cols)
    torch.testing.assert_close(actual.vals, expected.vals)


def test_a_sasrec_checkpoint_is_not_a_sasrec_ce_checkpoint(tmp_path):
    from compresso_recsys.models import SASRecTrainer

    path = tmp_path / "sasrec.zip"
    SASRecTrainer(
        SASRecConfig(d_model=8, n_heads=1, max_history_length=4, epochs=1, show_progress=False)
    ).fit(_seqs(_cycle_rows(8))).save(path)
    with pytest.raises(Exception, match="sasrec_trainer"):
        SASRecCETrainer.load(path)
