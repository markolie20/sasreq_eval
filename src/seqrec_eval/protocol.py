"""The frozen evaluation protocol, read from ``protocol.toml``.

Research design §5.3 fixes the protocol before the first experiment so that it
cannot be chosen to suit the results. This module turns that promise into a
check. A run's fingerprint covers exactly the sections that determine its
result -- ``[protocol]``, its dataset and its model -- and runs are stored under
that fingerprint, so an edit cannot silently mix with results produced before
it.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FAMILIES = ("matrix", "sequence")
#: What a user is scored against. "next": the first event after their history
#: (every item at that timestamp). "window": everything they do in the window.
TARGET_DEFINITIONS = ("next", "window")
#: The non-learned baselines of §5.2, scored by the analysis step (seqrec_eval.analysis).
BASELINE_KINDS = ("popularity", "time_popularity", "replay", "markov")
DISTRIBUTIONS = ("choice", "uniform", "loguniform", "int")
#: The library's own ``result_prefix`` of each metric, so a protocol name is
#: exactly the key the results carry (``hit_rate@10``, not ``hitrate@10``).
METRIC_NAMES = ("ndcg", "recall", "calibrated_recall", "hit_rate", "precision", "map", "mrr")

#: Keys of ``[protocol]`` that change results. ``eval_batch_size`` only changes
#: speed, so leaving it out lets it be tuned without invalidating any run. Of the
#: seeds only the first enters (as ``trial_seed``, see ``Protocol._result_settings``):
#: every final or ablation run of a seed lives in a directory of its own, so seeds
#: added later join the runs already made instead of starting them over (2026-09-30).
_RESULT_KEYS = (
    "version", "cutoffs", "metrics", "primary_metric",
    "trials_per_model", "search_seed", "max_val_users", "targets", "refit",
)

#: Bumped when the code changes what a scored target means, so results computed under the old meaning are not
#: reused. 1: next-item targets from the events the split kept, every user scored. 2 (2026-09-29, H16): next-item
#: targets are the user's real first moment in the window, and only users whose next item a model can recommend
#: are scored on them. 3 (2026-10-01, final review C4): under ``exclude_seen`` a next item already in the user's
#: history cannot be recommended either; and every evaluation excludes seen items the same way (review A6).
#: 4 (2026-10-01, review N30): that exclusion asks every batch of a phase for the same length, so a model's
#: lists no longer depend on the batch size or on which users share a batch.
SCORING_VERSION = 4

#: Keys of a ``[datasets.*]`` section that change how a split is scored but not the split itself.
_SCORING_ONLY_KEYS = ("exclude_seen", "new_item_diagnostic")

#: Bumped when the baselines' code changes what they recommend; it enters every baseline fingerprint, so results
#: of the older code are not reused. 2 (2026-09-30): equal scores are ranked in one fixed order (more popular
#: first, then the lower index), not as the top-k happened to break them.
BASELINE_VERSION = 2


class ProtocolError(ValueError):
    """The protocol file is malformed or internally inconsistent."""


#: The keys each section may hold. Anything else is refused by name: a misspelt key would otherwise fall back
#: to its default without a word -- ``scop = "inference"`` would refit every model at every level, and
#: ``max_item`` would let EASE run unbounded (final review, 2026-09-30).
_ALLOWED_KEYS = {
    "file": {"protocol", "latency", "datasets", "models", "ablations", "baselines", "repeat_strata"},
    "protocol": {"version", "cutoffs", "metrics", "primary_metric", "seeds", "trials_per_model", "search_seed",
                 "max_val_users", "targets", "refit", "eval_batch_size"},
    "latency": {"history_bins", "requests_per_bin", "warmup_requests"},
    "datasets": {"builder", "temporal_period_hours", "train_users", "min_user_support", "item_min_support",
                 "min_value_to_keep", "set_all_values_to", "exclude_seen", "new_item_diagnostic",
                 "amazon_category", "options"},
    "models": {"family", "trials", "fixed", "space", "max_items"},
    "ablations": {"transform", "levels", "options", "scope", "datasets", "models", "expected_to_move",
                  "manipulation_tolerance", "knee_margin", "min_level_users"},
    "baselines": {"kind", "trials", "fixed", "space"},
    "repeat_strata": {"history_bins", "repeat_bins", "min_users", "n_resamples"},
}


def _check_keys(table: dict, section: str, where: str) -> None:
    allowed = _ALLOWED_KEYS[section]
    unknown = sorted(set(table) - allowed)
    if unknown:
        hints = [f"{key!r} (did you mean {close[0]!r}?)" if (close := difflib.get_close_matches(key, allowed, 1))
                 else repr(key) for key in unknown]
        raise ProtocolError(f"{where} has unknown key(s) {', '.join(hints)}; it may hold {sorted(allowed)}")


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _require(table: dict, key: str, where: str) -> Any:
    if key not in table:
        raise ProtocolError(f"{where} is missing {key!r}")
    return table[key]


@dataclass(frozen=True)
class DatasetProtocol:
    name: str
    builder: str
    temporal_period_hours: float
    min_user_support: int
    item_min_support: int
    #: ``None`` keeps every interaction; see :meth:`build_parameters`.
    min_value_to_keep: float | None
    set_all_values_to: float
    exclude_seen: bool
    new_item_diagnostic: bool
    #: "window" or "all": who the temporal split trains on (see the library's
    #: ``temporal_train_users``); "all" is everything before the validation window.
    train_users: str
    amazon_category: str | None
    options: dict[str, Any]
    raw: dict[str, Any]

    def build_parameters(self) -> dict[str, Any]:
        """Keyword arguments for ``compresso_recsys.build_recsys_checkpoint``.

        Every preprocessing value is passed explicitly rather than left to the
        builder's registry. "Keep everything" is sent as ``-inf`` because the
        builder reads ``None`` as "use the registry default", which for ML-20M
        would quietly apply a 4-star threshold.
        """
        params: dict[str, Any] = {
            "dataset": self.builder,
            "split_mode": "temporal",
            "temporal_period_hours": self.temporal_period_hours,
            "min_user_support": self.min_user_support,
            "item_min_support": self.item_min_support,
            "min_value_to_keep": (
                -math.inf if self.min_value_to_keep is None else self.min_value_to_keep
            ),
            "set_all_values_to": self.set_all_values_to,
            "min_entity_text_words": 0,
            "dataset_options": dict(self.options) or None,
        }
        if self.amazon_category is not None:
            params["amazon_category"] = self.amazon_category
        # Passed only when it differs from the library's default, so a release
        # without the option still builds a split that does not use it.
        if self.train_users != "window":
            params["temporal_train_users"] = self.train_users
        return params


@dataclass(frozen=True)
class ModelProtocol:
    name: str
    family: str
    trials: int
    fixed: dict[str, Any]
    space: dict[str, dict[str, Any]]
    max_items: int | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class BaselineProtocol:
    """A non-learned baseline: its kind, and the settings searched on validation.

    A space of choices small enough to enumerate is searched in full, as a
    grid, rather than by random draws that would repeat configurations.
    """

    name: str
    kind: str
    trials: int
    grid: bool
    fixed: dict[str, Any]
    space: dict[str, dict[str, Any]]
    raw: dict[str, Any]


@dataclass(frozen=True)
class AblationProtocol:
    """One sweep: a transform applied to the split at each of ``levels``.

    ``levels`` and ``options`` are checked by the transform itself, in
    :mod:`seqrec_eval.ablations`, since only it knows what a level means.
    """

    name: str
    transform: str
    levels: tuple[Any, ...]
    datasets: tuple[str, ...]
    models: tuple[str, ...]
    options: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class Protocol:
    path: Path
    version: int
    cutoffs: tuple[int, ...]
    metrics: tuple[str, ...]
    primary_metric: str
    seeds: tuple[int, ...]
    trials_per_model: int
    search_seed: int
    max_val_users: int | None
    #: The target definition that selection, the floor and the statistics use; the other is a diagnostic.
    targets: str
    #: Whether every model scored on test is first refitted on train+validation (see :mod:`seqrec_eval.refit`).
    refit: bool
    eval_batch_size: int
    latency: dict[str, Any]
    datasets: dict[str, DatasetProtocol]
    models: dict[str, ModelProtocol]
    ablations: dict[str, AblationProtocol]
    baselines: dict[str, BaselineProtocol]
    raw: dict[str, Any]

    def dataset(self, name: str) -> DatasetProtocol:
        if name not in self.datasets:
            raise ProtocolError(f"unknown dataset {name!r}; the protocol defines {sorted(self.datasets)}")
        return self.datasets[name]

    def model(self, name: str) -> ModelProtocol:
        if name not in self.models:
            raise ProtocolError(f"unknown model {name!r}; the protocol defines {sorted(self.models)}")
        return self.models[name]

    def baseline(self, name: str) -> BaselineProtocol:
        if name not in self.baselines:
            raise ProtocolError(f"unknown baseline {name!r}; the protocol defines {sorted(self.baselines)}")
        return self.baselines[name]

    def baseline_fingerprint(self, dataset: str, baseline: str) -> str:
        """Identity of a baseline's search and scores on ``dataset`` under this file."""
        return _digest({
            "protocol": self._result_settings(),
            "dataset": self.dataset(dataset).raw,
            "baseline": self.baseline(baseline).raw,
            "code": BASELINE_VERSION,
        })

    def ablation(self, name: str) -> AblationProtocol:
        if name not in self.ablations:
            raise ProtocolError(f"unknown ablation {name!r}; the protocol defines {sorted(self.ablations)}")
        return self.ablations[name]

    def _result_settings(self) -> dict[str, Any]:
        # the first seed is every search trial's seed, so it decides which configuration is selected; the
        # others each add one final run of it, which the fingerprint does not need to know about
        return {**{key: self.raw["protocol"].get(key) for key in _RESULT_KEYS}, "trial_seed": self.seeds[0],
                "scoring_version": SCORING_VERSION}

    def evaluation_key(self, dataset: str) -> str:
        """What decides which users of ``dataset`` are scored, and on what: the target definition, the refit, the
        scoring code, and whether seen items are excluded (a next item already seen is then out of reach).

        Keys the caches that hold users or profiles rather than runs -- an ablation's fixed test users and
        per-condition characteristics, the analysis profile -- which the run fingerprints do not cover.
        """
        return _digest({"targets": self.targets, "refit": self.refit, "scoring_version": SCORING_VERSION,
                        "exclude_seen": self.dataset(dataset).exclude_seen})

    def dataset_fingerprint(self, dataset: str) -> str:
        """Identity of a prepared split: what builds it -- the dataset section's build settings -- and the
        validation sample (its size and seed).

        The settings that only change how a split is scored (:data:`_SCORING_ONLY_KEYS`) are left out, so
        flipping one does not rebuild the split or redo the analysis; every fingerprint of a scored result
        carries them instead (run, baseline and controls fingerprints hold the whole dataset section).
        """
        build = {key: value for key, value in self.dataset(dataset).raw.items() if key not in _SCORING_ONLY_KEYS}
        return _digest({"dataset": build, "max_val_users": self.max_val_users, "search_seed": self.search_seed})

    def run_fingerprint(self, dataset: str, model: str) -> str:
        """Identity of every run of ``model`` on ``dataset`` under this file."""
        return _digest({
            "protocol": self._result_settings(),
            "dataset": self.dataset(dataset).raw,
            "model": self.model(model).raw,
        })


def _parse_value_threshold(value: Any, where: str) -> float | None:
    if isinstance(value, str):
        if value.lower() != "none":
            raise ProtocolError(f"{where}.min_value_to_keep must be a number or \"none\", got {value!r}")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{where}.min_value_to_keep must be a number or \"none\", got {value!r}")
    return float(value)


def _parse_dataset(name: str, table: dict) -> DatasetProtocol:
    where = f"[datasets.{name}]"
    _check_keys(table, "datasets", where)
    options = table.get("options", {})
    if not isinstance(options, dict):
        raise ProtocolError(f"{where}.options must be a table")
    period = float(_require(table, "temporal_period_hours", where))
    if period <= 0:
        raise ProtocolError(f"{where}.temporal_period_hours must be positive")
    # Required, with no default: a missing key once silently ran another protocol (review H11).
    if "train_users" not in table:
        raise ProtocolError(f'{where}.train_users is required: "all" trains on every user with events before the '
                            'validation window, "window" only on users also active in the train window')
    train_users = table["train_users"]
    if train_users not in ("window", "all"):
        raise ProtocolError(f'{where}.train_users must be "window" or "all", got {train_users!r}')
    return DatasetProtocol(
        name=name,
        builder=str(_require(table, "builder", where)),
        temporal_period_hours=period,
        min_user_support=int(_require(table, "min_user_support", where)),
        item_min_support=int(_require(table, "item_min_support", where)),
        min_value_to_keep=_parse_value_threshold(_require(table, "min_value_to_keep", where), where),
        set_all_values_to=float(_require(table, "set_all_values_to", where)),
        exclude_seen=bool(_require(table, "exclude_seen", where)),
        new_item_diagnostic=bool(table.get("new_item_diagnostic", False)),
        train_users=train_users,
        amazon_category=table.get("amazon_category"),
        options=dict(options),
        raw=table,
    )


def _check_distribution(model: str, parameter: str, spec: Any) -> None:
    where = f"[models.{model}.space].{parameter}"
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ProtocolError(f"{where} must be one of {{choice|uniform|loguniform|int = ...}}")
    kind, value = next(iter(spec.items()))
    if kind not in DISTRIBUTIONS:
        raise ProtocolError(f"{where} uses unknown distribution {kind!r}; choose from {DISTRIBUTIONS}")
    if kind == "choice":
        if not isinstance(value, list) or not value:
            raise ProtocolError(f"{where}.choice must be a non-empty list")
        return
    if not isinstance(value, list) or len(value) != 2:
        raise ProtocolError(f"{where}.{kind} must be [low, high]")
    low, high = value
    if kind == "int":
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in value) or low > high:
            raise ProtocolError(f"{where}.int must be two integers with low <= high")
    elif not low < high:
        raise ProtocolError(f"{where}.{kind} needs low < high")
    if kind == "loguniform" and low <= 0:
        raise ProtocolError(f"{where}.loguniform needs a positive low bound")


def _parse_model(name: str, table: dict, default_trials: int) -> ModelProtocol:
    where = f"[models.{name}]"
    _check_keys(table, "models", where)
    family = _require(table, "family", where)
    if family not in FAMILIES:
        raise ProtocolError(f"{where}.family must be one of {FAMILIES}, got {family!r}")
    fixed = dict(table.get("fixed", {}))
    space = dict(table.get("space", {}))
    for parameter, spec in space.items():
        _check_distribution(name, parameter, spec)
    overlap = sorted(set(fixed) & set(space))
    if overlap:
        raise ProtocolError(f"{where} sets {overlap} both as fixed and as searched")
    trials = int(table.get("trials", default_trials if space else 1))
    if trials < 1:
        raise ProtocolError(f"{where}.trials must be >= 1")
    max_items = table.get("max_items")
    return ModelProtocol(
        name=name, family=family, trials=trials, fixed=fixed, space=space,
        max_items=None if max_items is None else int(max_items), raw=table,
    )


def _parse_baseline(name: str, table: dict, default_trials: int) -> BaselineProtocol:
    where = f"[baselines.{name}]"
    _check_keys(table, "baselines", where)
    kind = _require(table, "kind", where)
    if kind not in BASELINE_KINDS:
        raise ProtocolError(f"{where}.kind must be one of {BASELINE_KINDS}, got {kind!r}")
    fixed = dict(table.get("fixed", {}))
    space = dict(table.get("space", {}))
    for parameter, spec in space.items():
        _check_distribution(name, parameter, spec)
    overlap = sorted(set(fixed) & set(space))
    if overlap:
        raise ProtocolError(f"{where} sets {overlap} both as fixed and as searched")
    grid_size = 1
    for spec in space.values():
        kind_of, value = next(iter(spec.items()))
        grid_size = grid_size * len(value) if kind_of == "choice" else math.inf
    grid = grid_size <= default_trials
    trials = int(table.get("trials", grid_size if grid else default_trials))
    if trials < 1:
        raise ProtocolError(f"{where}.trials must be >= 1")
    if grid and trials != grid_size:
        raise ProtocolError(f"{where} is a grid of {grid_size} configurations; trials must be {grid_size}")
    return BaselineProtocol(name=name, kind=kind, trials=trials, grid=grid, fixed=fixed, space=space, raw=table)


def _parse_ablation(name: str, table: dict, datasets: dict, models: dict) -> AblationProtocol:
    where = f"[ablations.{name}]"
    _check_keys(table, "ablations", where)
    levels = _require(table, "levels", where)
    if not isinstance(levels, list) or not levels:
        raise ProtocolError(f"{where}.levels must be a non-empty list")
    if len(set(map(str, levels))) != len(levels):
        raise ProtocolError(f"{where}.levels must be distinct")
    options = table.get("options", {})
    if not isinstance(options, dict):
        raise ProtocolError(f"{where}.options must be a table")
    chosen = {}
    for key, available in (("datasets", datasets), ("models", models)):
        names = table.get(key, list(available))
        unknown = [n for n in names if n not in available]
        if unknown:
            raise ProtocolError(f"{where}.{key} names {unknown}, which the protocol does not define")
        chosen[key] = tuple(names)
    return AblationProtocol(
        name=name, transform=str(_require(table, "transform", where)), levels=tuple(levels),
        datasets=chosen["datasets"], models=chosen["models"], options=dict(options), raw=table,
    )


def load_protocol(path: str | Path) -> Protocol:
    path = Path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    _check_keys(raw, "file", "the protocol file")
    top = _require(raw, "protocol", "protocol file")
    _check_keys(top, "protocol", "[protocol]")
    latency = dict(raw.get("latency", {}))
    _check_keys(latency, "latency", "[latency]")
    bins = latency.get("history_bins")
    if bins is not None and (not bins or not all(_is_count(b) for b in bins)
                             or any(b >= c for b, c in zip(bins, bins[1:]))):
        raise ProtocolError(f"[latency].history_bins must be positive integers in increasing order, got {bins}")
    _check_keys(raw.get("repeat_strata", {}), "repeat_strata", "[repeat_strata]")

    cutoffs = tuple(sorted({int(k) for k in _require(top, "cutoffs", "[protocol]")}))
    if not cutoffs or cutoffs[0] < 1:
        raise ProtocolError("[protocol].cutoffs must be positive integers")
    metrics = tuple(_require(top, "metrics", "[protocol]"))
    unknown = [m for m in metrics if m not in METRIC_NAMES]
    if unknown:
        raise ProtocolError(f"unknown metrics {unknown}; choose from {METRIC_NAMES}")

    primary = str(_require(top, "primary_metric", "[protocol]"))
    match = re.fullmatch(r"([a-z_]+)@(\d+)", primary)
    if not match or match[1] not in metrics or int(match[2]) not in cutoffs:
        raise ProtocolError(
            f"primary_metric {primary!r} must be <metric>@<k> with the metric in "
            f"{list(metrics)} and k in {list(cutoffs)}"
        )

    seeds = tuple(int(s) for s in _require(top, "seeds", "[protocol]"))
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        # numpy's global seed must be non-negative: a negative one would fail at the first run, not here
        raise ProtocolError("[protocol].seeds must be a non-empty list of distinct non-negative integers")
    trials_per_model = int(_require(top, "trials_per_model", "[protocol]"))
    max_val_users = top.get("max_val_users")
    # Required, with no default: a missing key once silently ran another protocol (review H11).
    for key, meaning in (("targets", '"next" (the next item) or "window" (everything in the window)'),
                         ("refit", "true (refit on train+validation before test) or false")):
        if key not in top:
            raise ProtocolError(f"[protocol].{key} is required: {meaning}")
    targets = top["targets"]
    if targets not in TARGET_DEFINITIONS:
        raise ProtocolError(f"[protocol].targets must be one of {TARGET_DEFINITIONS}, got {targets!r}")
    refit = top["refit"]
    if not isinstance(refit, bool):
        raise ProtocolError(f"[protocol].refit must be true or false, got {refit!r}")

    datasets = {name: _parse_dataset(name, table) for name, table in raw.get("datasets", {}).items()}
    models = {name: _parse_model(name, table, trials_per_model) for name, table in raw.get("models", {}).items()}
    if not datasets:
        raise ProtocolError("the protocol defines no [datasets.*]")
    if not models:
        raise ProtocolError("the protocol defines no [models.*]")
    ablations = {name: _parse_ablation(name, table, datasets, models)
                 for name, table in raw.get("ablations", {}).items()}
    baselines = {name: _parse_baseline(name, table, trials_per_model)
                 for name, table in raw.get("baselines", {}).items()}

    return Protocol(
        path=path,
        version=int(_require(top, "version", "[protocol]")),
        cutoffs=cutoffs,
        metrics=metrics,
        primary_metric=primary,
        seeds=seeds,
        trials_per_model=trials_per_model,
        search_seed=int(_require(top, "search_seed", "[protocol]")),
        max_val_users=None if max_val_users is None else int(max_val_users),
        targets=targets,
        refit=refit,
        eval_batch_size=int(top.get("eval_batch_size", 1024)),
        latency=latency,
        datasets=datasets,
        models=models,
        ablations=ablations,
        baselines=baselines,
        raw=raw,
    )
