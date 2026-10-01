"""Writing results so that a crash never leaves a half-written file behind.

Runs are resumable because a run counts as finished only when its ``done.json``
exists, and every file is written to a temporary name and renamed into place.
A process killed mid-write leaves a stray temporary file, never a truncated
result that a later process would mistake for a real one.

:class:`~compresso_recsys.evaluation.EvaluationResult` has no persistence of its
own, and the paired statistics need its per-user arrays after the process that
produced them has exited. It is stored as a pair: ``<stem>.npz`` for the arrays
and ``<stem>.json`` for everything else.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from compresso_recsys.evaluation import EvaluationResult

_PER_USER = "per_user__"


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path, set, frozenset)):
        return str(value) if isinstance(value, Path) else sorted(value)
    return str(value)


def durable_replace(temporary: Path, path: Path) -> None:
    """Rename ``temporary`` onto ``path`` so that a power loss leaves the old file or the new one, never an
    empty one: the data is flushed to disk before the rename, and the rename itself after it (review B5)."""
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")
    durable_replace(temporary, path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON ({error}); it may have been cut short by a crash -- move it "
                         "aside and rerun the step that writes it") from None


def save_evaluation(result: EvaluationResult, stem: Path) -> None:
    if result.per_user is None or result.sample_ids is None:
        raise ValueError("only results collected with per-user values can be saved for comparison")
    stem.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{_PER_USER}{name}": np.asarray(values, dtype=np.float64)
              for name, values in result.per_user.items()}
    arrays["sample_ids"] = np.asarray(result.sample_ids).astype(str)
    npz = stem.with_suffix(".npz")
    temporary = _temporary(npz)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    durable_replace(temporary, npz)
    write_json(stem.with_suffix(".json"), {
        "metrics": dict(result.metrics),
        "n_rows": result.n_rows,
        "n_scored_rows": result.n_scored_rows,
        "required_k": result.required_k,
        "metadata": dict(result.metadata),
        "target_fingerprint": result.target_fingerprint,
    })


def load_evaluation(stem: Path) -> EvaluationResult:
    record = read_json(stem.with_suffix(".json"))
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as arrays:
        per_user = {key[len(_PER_USER):]: arrays[key] for key in arrays.files if key.startswith(_PER_USER)}
        sample_ids = arrays["sample_ids"]
    return EvaluationResult(
        metrics={k: float(v) for k, v in record["metrics"].items()},
        per_user=per_user,
        sample_ids=sample_ids,
        n_rows=record["n_rows"],
        n_scored_rows=record["n_scored_rows"],
        required_k=record["required_k"],
        metadata=record["metadata"],
        target_fingerprint=record["target_fingerprint"],
    )


def evaluation_exists(stem: Path) -> bool:
    return stem.with_suffix(".json").exists() and stem.with_suffix(".npz").exists()
