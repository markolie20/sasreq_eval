"""Random-search configurations, reproducible trial by trial.

The stage-1 search exists to give every architecture the same budget (§5.2),
so the configurations it draws have to be a fixed function of the protocol:
rerunning a trial, or running it on another machine, must produce the same
configuration. Two choices follow.

Trial ``i`` of a model draws the same configuration on every dataset. That
keeps "trial 7" meaning one thing across the whole study, and makes it possible
to see whether a configuration that wins on one dataset wins elsewhere.

Each parameter has its own random stream. With one shared stream, adding a
parameter to a space would shift every draw after it, changing configurations
that were never meant to change.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from typing import Any

import numpy as np

from .protocol import ModelProtocol, Protocol


def _stream(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _draw(spec: dict[str, Any], rng: np.random.Generator) -> Any:
    kind, value = next(iter(spec.items()))
    if kind == "choice":
        # Index rather than rng.choice, which would turn Python values into numpy
        # scalars that neither the configs nor json accept as they stand.
        return value[int(rng.integers(len(value)))]
    low, high = value
    if kind == "int":
        return int(rng.integers(low, high + 1))
    if kind == "uniform":
        return float(rng.uniform(low, high))
    if kind == "loguniform":
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    raise ValueError(f"unknown distribution {kind!r}")  # protocol validation rules this out


def grid_params(space: dict[str, dict[str, Any]], fixed: dict[str, Any], trial: int) -> dict[str, Any]:
    """Configuration ``trial`` of a space of choices, enumerated in a fixed order (parameters sorted by name)."""
    names = sorted(space)
    combinations = list(itertools.product(*(space[name]["choice"] for name in names)))
    if not 0 <= trial < len(combinations):
        raise ValueError(f"the grid has configurations 0..{len(combinations) - 1}, not {trial}")
    return {**fixed, **dict(zip(names, combinations[trial]))}


def trial_params(protocol: Protocol, model: ModelProtocol, trial: int) -> dict[str, Any]:
    """The configuration of ``trial`` for ``model``: fixed values plus one draw per parameter."""
    if not 0 <= trial < model.trials:
        raise ValueError(f"{model.name} has trials 0..{model.trials - 1}, not {trial}")
    params = dict(model.fixed)
    for name in sorted(model.space):
        params[name] = _draw(model.space[name], _stream(protocol.search_seed, model.name, trial, name))
    return params
