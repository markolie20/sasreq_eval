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

A model can also come from another installed package, which this file then
never names (a model whose code cannot be published, say): the package
declares an entry point in the group :data:`PLUGIN_GROUP`, the module it names
calls :func:`register` when imported, and its ``[models.<name>]`` section can
live in an extra protocol file (``--protocol-extra``). See :func:`_load_plugins`.
``max_history_length`` is the one parameter name shared by every sequential
model, because the history-length ablation varies it across all of them.

Only learned models are registered here. The non-learned baselines and the
sequence-signal controls are not models under study: they are scored by the
analysis step (:mod:`seqrec_eval.analysis`), and the strongest baseline sets
the floor every model has to beat.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
        failed = "".join(f"; plugin {plugin!r} failed to load: {error}" for plugin, error in PLUGIN_ERRORS.items())
        raise KeyError(f"model {name!r} is in the protocol but not registered in seqrec_eval.models; "
                       f"registered: {sorted(REGISTRY)}{failed}")
    return REGISTRY[name]


def _installed_version(package: str) -> str | None:
    """The installed version of the distribution that provides the import package ``package``, if any."""
    for name in importlib.metadata.packages_distributions().get(package) or [package]:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


#: The packages whose code every run already records (:func:`seqrec_eval.splits.code_provenance`).
_RECORDED = {"seqrec_eval", "compresso_recsys"}


def model_provenance(name: str) -> dict[str, Any]:
    """The code of a model from a plugin package, for the run record: ``{}`` for the suite's own models.

    The plugin's top-level package, and any packages its model class lists in ``provenance_packages`` (the
    library it wraps, say), each with its installed version and a hash of its imported Python source.
    """
    from .splits import _source_hash

    cls = model_spec(name).cls
    packages = [cls.__module__.split(".")[0], *getattr(cls, "provenance_packages", ())]
    record = {}
    for package in dict.fromkeys(packages):
        if package in _RECORDED:
            continue
        module = sys.modules.get(package) or importlib.import_module(package)
        record[package] = {"version": _installed_version(package), "code_sha256": _source_hash(module),
                           "path": str(Path(module.__file__).resolve().parent)}
    return {"plugins": record} if record else {}


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


#: The entry-point group through which other installed packages add models (DECISIONS §39).
PLUGIN_GROUP = "seqrec_eval.models"
#: plugin name -> why it failed to load; its models then stay unregistered, which ``plan`` shows and runs refuse
PLUGIN_ERRORS: dict[str, str] = {}


def _load_plugins() -> None:
    """Import every module declared in :data:`PLUGIN_GROUP`; each registers its models with :func:`register`.

    Run once, as this module finishes loading, so every caller of :func:`model_spec` sees the same registry. A
    plugin that fails to import is reported on stderr and recorded in :data:`PLUGIN_ERRORS`, rather than stopping
    every command: the suite's own models still run, and a run of the plugin's model fails with the reason.
    """
    for entry in importlib.metadata.entry_points(group=PLUGIN_GROUP):
        try:
            entry.load()
        except Exception as error:  # noqa: BLE001 - any import failure leaves the suite's own models usable
            PLUGIN_ERRORS[entry.name] = f"{type(error).__name__}: {error}"
            print(f"seqrec_eval: model plugin {entry.name!r} ({entry.value}) failed to load: "
                  f"{PLUGIN_ERRORS[entry.name]}", file=sys.stderr)


_load_plugins()
