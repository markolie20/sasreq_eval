"""Which models the suite can run, and how to build each from a configuration.

A model is one decorated function: it receives the trial's parameters and
returns an *unfitted* trainer that follows the library's contract,
``trainer.fit(data, item_ids=...)`` then ``predict_on_batch(source, k=...,
exclude_seen=..., candidate_ids=...)``. ``family`` decides what it is fitted on
and scored from: ``"matrix"`` reads the CSR view of a split, ``"sequence"`` the
chronological :class:`~compresso_recsys.ItemSequences` view of the same events.

To add a model (Mamba4Rec, ComiRec, SANSA, a Markov baseline, ...): write its
trainer against that contract, register a builder here, and give it a
``[models.<name>]`` section in the protocol. Nothing else in the suite changes.
``max_history_length`` is the one parameter name shared by every sequential
model, because the history-length ablation varies it across all of them.

Only learned models are registered here. The non-learned baselines and the
sequence-signal controls are not models under study: they are scored by the
analysis step (:mod:`seqrec_eval.analysis`), and the strongest baseline sets
the floor every model has to beat.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from compresso_recsys.models import (
    EASE,
    Bert4RecConfig,
    Bert4RecTrainer,
    EASEConfig,
    ELSAConfig,
    ELSATrainer,
    ItemTokenizer,
    PopularityBaseline,
    PopularityBaselineConfig,
    SASRecConfig,
    SASRecTrainer,
    SequenceBatcher,
    SimpleRNNConfig,
    SimpleRNNTrainer,
)

Builder = Callable[..., Any]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    #: The trainer class, used to reload a saved final model for latency runs.
    cls: type
    build: Builder


REGISTRY: dict[str, ModelSpec] = {}


def register(name: str, *, family: str, cls: type) -> Callable[[Builder], Builder]:
    def decorate(build: Builder) -> Builder:
        if name in REGISTRY:
            raise ValueError(f"model {name!r} is registered twice")
        REGISTRY[name] = ModelSpec(name=name, family=family, cls=cls, build=build)
        return build
    return decorate


def model_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"model {name!r} is in the protocol but not registered in seqrec_eval.models; "
                       f"registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


# Every builder takes (params, *, n_items, device, seed). Progress bars are off
# because runs are unattended and their output goes to log files.

@register("popularity", family="matrix", cls=PopularityBaseline)
def _popularity(params, *, n_items, device, seed):
    return PopularityBaseline(PopularityBaselineConfig(**params))


@register("ease", family="matrix", cls=EASE)
def _ease(params, *, n_items, device, seed):
    return EASE(EASEConfig(**params))


@register("elsa", family="matrix", cls=ELSATrainer)
def _elsa(params, *, n_items, device, seed):
    return ELSATrainer(ELSAConfig(**params, device=device, seed=seed, show_progress=False))


@register("gru", family="sequence", cls=SimpleRNNTrainer)
def _gru(params, *, n_items, device, seed):
    # SimpleRNN takes its context window from the batcher rather than its config.
    params = dict(params)
    max_length = params.pop("max_history_length", SimpleRNNTrainer.DEFAULT_MAX_LENGTH)
    return SimpleRNNTrainer(
        SimpleRNNConfig(**params, device=device, seed=seed, show_progress=False),
        SequenceBatcher(ItemTokenizer(n_items), max_length=int(max_length)),
    )


@register("sasrec", family="sequence", cls=SASRecTrainer)
def _sasrec(params, *, n_items, device, seed):
    return SASRecTrainer(SASRecConfig(**params, device=device, seed=seed, show_progress=False))


@register("bert4rec", family="sequence", cls=Bert4RecTrainer)
def _bert4rec(params, *, n_items, device, seed):
    # The trainer builds its own batcher, whose tokenizer has the [MASK] token BERT4Rec needs; a batcher built
    # here on a plain ItemTokenizer would not (review H26).
    return Bert4RecTrainer(Bert4RecConfig(**params, device=device, seed=seed, show_progress=False))
