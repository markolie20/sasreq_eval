"""Building each dataset's temporal split once, and loading it for every run.

The split comes from the library's own temporal builder: three consecutive
windows at the end of the log, each ``temporal_period_hours`` wide. Search trials
train on everything before the validation window; validation and test histories
run up to their own windows, and the catalogue grows by appending the items first
seen in each window. The builder writes a zip; it is extracted once here, so a
run loads a directory instead of unpacking hundreds of megabytes per trial.

``split_info.json`` records what the split actually is -- the parameters the
builder resolved, stage sizes and the window boundaries -- because the research
design (§6.2) asks for every preprocessing decision to be reproducible from a
record rather than recalled.

The saved split keeps each history's order but not its times, so the time of
every event is recovered afterwards from the library's prepared events and
proved against the split (see :mod:`seqrec_eval.timestamps`). A split prepared
without them, or with next-item targets of an older definition, gets them on the
next ``prepare``, without being rebuilt.

With ``refit = true`` in ``[protocol]``, ``prepare`` also builds the training set
of everything before the test window, proved the same way (see
:mod:`seqrec_eval.refit`), and :func:`final_split` is the split every model
scored on test trains on. Search trials keep the original.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import inspect
import json
import shutil
import time
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from compresso_recsys import (
    build_recsys_checkpoint,
    builder,
    load_manifest,
    load_recsys_split,
)

from .protocol import Protocol
from .refit import attach_refit, load_refit, swap_training
from .results import read_json, write_json
from .search import _stream
from .timestamps import NEXT_TARGETS_VERSION, attach_timestamps, load_timestamps, prepared_events

PHASES = ("train", "val", "test")


@dataclass
class Split:
    dataset: str
    path: Path
    data: dict[str, Any]
    info: dict[str, Any]
    #: Fixed validation rows scored in every search trial, or ``None`` for all.
    val_rows: np.ndarray | None
    #: Fixed test rows, or ``None`` for all. Set on an ablation condition, where
    #: every level of a sweep is scored on the same users.
    test_rows: np.ndarray | None = None
    #: The ablation condition this split was transformed by, ``None`` for the original.
    condition: dict[str, Any] | None = None
    #: What its training views hold: ``"train"``, the training window, or ``"train+val"``, everything
    #: before the test window (:func:`final_split`). A model fitted on the latter has seen the
    #: validation window, so the split refuses to score validation.
    trained_on: str = "train"

    def eval_user_ids(self, phase: str) -> np.ndarray:
        ids = self.data.get(f"{phase}_eval_user_ids")
        if ids is None:
            ids = self.data[f"{phase}_user_ids"]
        ids = np.asarray(ids).astype(str)
        rows = self.data[f"{phase}_target_matrix"].shape[0]
        if ids.shape[0] != rows:
            raise ValueError(f"{self.dataset}: {phase} has {rows} rows but {ids.shape[0]} user ids")
        return ids


def split_dir(work_dir: Path, dataset: str) -> Path:
    return Path(work_dir) / "splits" / dataset


def _resolved_parameters(params: dict[str, Any], data_dir: Path) -> dict[str, Any] | str:
    """What the builder will use once its registry defaults are merged in."""
    try:
        resolved, _ = builder._resolve_args(builder._build_args(**params, data_dir=str(data_dir)))
        return {key: value for key, value in vars(resolved).items() if not key.startswith("_")}
    except Exception as error:  # a private helper: record its absence, never fail on it  # noqa: BLE001
        return f"unavailable: {type(error).__name__}: {error}"


def _stage_stats(data: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for phase in PHASES:
        source = data[f"{phase}_source_matrix"]
        targets = data[f"{phase}_target_matrix"]
        sequences = data.get(f"{phase}_source_sequences")
        seen = source.copy()
        seen.data[:] = 1.0
        repeats = int(targets.multiply(seen).nnz)
        entry = {
            "rows": int(targets.shape[0]),
            "catalog_items": len(data[f"{phase}_item_ids"]),
            "target_pairs": int(targets.nnz),
            "repeat_target_pairs": repeats,
            "repeat_target_fraction": repeats / targets.nnz if targets.nnz else None,
        }
        if sequences is not None:
            lengths = sequences.row_lengths
            entry["history_length"] = {
                "mean": float(lengths.mean()) if lengths.size else None,
                "p50": float(np.median(lengths)) if lengths.size else None,
                "p90": float(np.quantile(lengths, 0.9)) if lengths.size else None,
                "max": int(lengths.max()) if lengths.size else None,
            }
        stats[phase] = entry
    stats["train_events"] = int(data["x_train_sequences"].values.size) if data.get("x_train_sequences") is not None else None
    return stats


def _source_hash(package: Any) -> str:
    """A hash of the package's Python source as imported: what actually ran, whatever the metadata says."""
    root = Path(package.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def code_provenance() -> dict[str, Any]:
    """The code a run executed: the library and the suite, each as installed metadata and as a hash of the
    imported source (review H05, H10). Recorded in every run; the report lists the builds a result rests on."""
    import compresso_recsys

    import seqrec_eval

    try:
        suite_version = importlib.metadata.version("seqrec-eval")
    except importlib.metadata.PackageNotFoundError:
        suite_version = None
    return {"library": library_provenance(),
            "suite": {"version": suite_version, "code_sha256": _source_hash(seqrec_eval),
                      "path": str(Path(seqrec_eval.__file__).resolve().parent)}}


def _checkout_version(package_dir: Path) -> str | None:
    """The version a source checkout declares, when the library was imported from one (``<checkout>/src/<pkg>``)."""
    try:
        project = tomllib.loads((package_dir.parent.parent / "pyproject.toml").read_text()).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return project.get("version") if project.get("name") == "compresso-recsys" else None


def library_provenance() -> dict[str, Any]:
    """Which compresso-recsys built a split: its version and, if installed from git, the commit, and a hash of the
    imported source, since the installed metadata need not be what was imported (H10).

    Imported from a source checkout (``PYTHONPATH=<checkout>/src``), the version is the checkout's own: the
    metadata Python finds may be another build's, or a stale ``*.egg-info`` left in ``src/`` (2026-09-30: a
    2026-09-24 egg-info reported 0.3.6 for the 0.3.7+trainusers branch). What the metadata said is kept beside it.
    """
    import compresso_recsys

    path = Path(compresso_recsys.__file__).resolve().parent
    installed = importlib.metadata.version("compresso-recsys")
    declared = _checkout_version(path)
    record: dict[str, Any] = {"version": declared or installed, "code_sha256": _source_hash(compresso_recsys),
                              "path": str(path)}
    if declared is not None and declared != installed:
        record["metadata_version"] = installed
    try:
        origin = json.loads(importlib.metadata.distribution("compresso-recsys").read_text("direct_url.json") or "{}")
    except (FileNotFoundError, ValueError):
        origin = {}
    if origin:
        record["source"] = origin.get("url")
        record["commit"] = origin.get("vcs_info", {}).get("commit_id")
        record["editable"] = bool(origin.get("dir_info", {}).get("editable"))
    return record


def _check_library(params: dict[str, Any]) -> None:
    """Refuse, before the slow build starts, a setting the installed library does not have."""
    accepted = inspect.signature(build_recsys_checkpoint).parameters
    missing = [key for key in params if key not in accepted]
    if missing:
        raise RuntimeError(
            f"the installed compresso-recsys {library_provenance()['version']} does not accept {missing}. "
            "temporal_train_users needs the build with that option (see the README's install section)."
        )


def prepare_split(protocol: Protocol, dataset: str, *, data_dir: Path, work_dir: Path,
                  force: bool = False, show_progress: bool = True, timestamps: bool = True) -> Path:
    """Build, extract and describe ``dataset``'s split, with its timestamps; a no-op if both exist.

    One process at a time per dataset: two would share the staging directory and overwrite each other's
    files (review B11).
    """
    from .runner import RunLock  # the runner imports this module

    lock = RunLock(Path(work_dir) / "splits" / f".{dataset}.prepare")
    if not lock.acquire():
        raise RuntimeError(f"another process is preparing {dataset} in {work_dir} right now; wait for it to finish")
    try:
        return _prepare_split(protocol, dataset, data_dir=data_dir, work_dir=work_dir, force=force,
                              show_progress=show_progress, timestamps=timestamps)
    finally:
        lock.release()


def _prepare_split(protocol: Protocol, dataset: str, *, data_dir: Path, work_dir: Path,
                   force: bool, show_progress: bool, timestamps: bool) -> Path:
    out = split_dir(work_dir, dataset)
    fingerprint = protocol.dataset_fingerprint(dataset)
    info_path = out / "split_info.json"
    if info_path.exists() and not force:
        info = read_json(info_path)
        existing = info.get("dataset_fingerprint")
        if existing == fingerprint:
            # next targets of an older definition are recomputed too (see NEXT_TARGETS_VERSION)
            missing_times = timestamps and (info.get("timestamps", {}).get("next_targets_version")
                                            != NEXT_TARGETS_VERSION)
            missing_refit = protocol.refit and "refit" not in info
            if missing_times or missing_refit:
                data = load_recsys_split(out)
                data.update(load_timestamps(out))
                events = prepared_events(info["build_parameters"], data_dir)
                if missing_times:
                    info["timestamps"] = attach_timestamps(info["build_parameters"], data_dir, out, data,
                                                           info["manifest"], events=events)
                    data.update(load_timestamps(out))
                if missing_refit:
                    info["refit"] = attach_refit(info["build_parameters"], data_dir, out, data, info["manifest"],
                                                 events=events)
                write_json(info_path, info)
            return out
        raise RuntimeError(
            f"{out} was prepared from a different [datasets.{dataset}] section. Rebuilding it "
            f"changes the split every run on {dataset} was scored against; pass --force if that is intended."
        )

    params = protocol.dataset(dataset).build_parameters()
    _check_library(params)
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = out.parent / f".{dataset}.building.zip"
    staging = out.parent / f".{dataset}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    archive.unlink(missing_ok=True)

    started = time.perf_counter()
    build_recsys_checkpoint(**params, data_dir=str(data_dir), checkpoint_path=str(archive),
                            show_progress=show_progress)
    build_seconds = time.perf_counter() - started
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(staging)
    archive.unlink()

    data = load_recsys_split(staging)
    manifest = load_manifest(staging)
    val_rows = None
    n_val = data["val_target_matrix"].shape[0]
    if protocol.max_val_users is not None and n_val > protocol.max_val_users:
        rng = _stream(protocol.search_seed, dataset, "val_rows")
        val_rows = np.sort(rng.choice(n_val, size=protocol.max_val_users, replace=False)).astype(np.int64)
        np.save(staging / "val_rows.npy", val_rows)

    info = {
        "dataset": dataset,
        "dataset_fingerprint": fingerprint,
        "build_parameters": params,
        "resolved_build_parameters": _resolved_parameters(params, data_dir),
        "library": library_provenance(),
        "build_seconds": build_seconds,
        "val_rows_sampled": None if val_rows is None else int(val_rows.size),
        "stages": _stage_stats(data),
        "manifest": manifest,
    }
    events = prepared_events(params, data_dir) if timestamps or protocol.refit else None
    if timestamps:
        info["timestamps"] = attach_timestamps(params, data_dir, staging, data, manifest, events=events)
        data.update(load_timestamps(staging))
    if protocol.refit:
        info["refit"] = attach_refit(params, data_dir, staging, data, manifest, events=events)
    del events
    write_json(staging / "split_info.json", info)
    del data
    if out.exists():
        shutil.rmtree(out)
    staging.rename(out)
    return out


def load_split(work_dir: Path, dataset: str, protocol: Protocol | None = None) -> Split:
    """The prepared split; given ``protocol``, refused unless it was prepared from that protocol's settings.

    Every command that runs or scores passes the protocol: only ``prepare`` used to check, so a changed
    ``[datasets.*]`` section or validation sample would be scored on the old split while the runs were filed
    under the new fingerprint (final review A4).
    """
    path = split_dir(work_dir, dataset)
    info_path = path / "split_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{dataset} has no prepared split at {path}; run `seqrec-eval prepare` first")
    if protocol is not None:
        prepared = read_json(info_path).get("dataset_fingerprint")
        if prepared != protocol.dataset_fingerprint(dataset):
            raise RuntimeError(
                f"{path} was prepared from other build settings than [datasets.{dataset}] (or another "
                "max_val_users or search_seed) now gives. Every result would be scored on the old split: prepare "
                f"it again (`seqrec-eval prepare --dataset {dataset} --force`), or restore the settings.")
    val_rows_path = path / "val_rows.npy"
    data = load_recsys_split(path)
    data.update(load_timestamps(path))
    data.update(load_refit(path))
    return Split(
        dataset=dataset,
        path=path,
        data=data,
        info=read_json(info_path),
        val_rows=np.load(val_rows_path) if val_rows_path.exists() else None,
    )


def final_split(protocol: Protocol, split: Split) -> Split:
    """The split a model scored on test is fitted on: trained on train+validation under ``refit``, else itself.

    Final runs, latency, the test side of the analysis and every ablation
    condition use it; search trials never do, since they are selected on
    validation.
    """
    if not protocol.refit or split.trained_on == "train+val":
        return split
    if split.condition is not None:
        raise ValueError("refit the full split, then transform it: a condition's training data cannot be swapped")
    return Split(dataset=split.dataset, path=split.path, data=swap_training(split.data), info=split.info,
                 val_rows=split.val_rows, test_rows=split.test_rows, trained_on="train+val")
