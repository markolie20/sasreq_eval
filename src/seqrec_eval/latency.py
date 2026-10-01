"""Model inference latency on CPU, per request, by history length.

The constraint in the requirements is model inference latency on CPU (A-03),
100 ms at P95, characterised against history length rather than averaged
(NFR-03). So each request here is one user, scored against the full test
catalogue on CPU, and the report is a distribution per history-length bin.

Only the model call is timed. Slicing the request out of the split happens
before the clock starts, and the first requests are discarded as warm-up.

Run it when the machine is otherwise quiet, or on cores nothing else uses
(``--cores``): a P95 measured next to training jobs measures the contention.
"""

from __future__ import annotations

import os
import platform
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluate import ExcludeSeenPolicy, phase_inputs
from .models import model_spec
from .protocol import Protocol
from .results import read_json, write_json
from .runner import run_root
from .search import _stream
from .splits import Split


def parse_cores(text: str | None) -> set[int] | None:
    """``"16-19"`` or ``"0,2,4"`` into a CPU set."""
    if not text:
        return None
    cores: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            cores.update(range(int(low), int(high) + 1))
        else:
            cores.add(int(part))
    return cores


def _percentiles(values_ms: np.ndarray) -> dict[str, float]:
    return {"n": int(values_ms.size), "mean_ms": float(values_ms.mean()),
            "p50_ms": float(np.percentile(values_ms, 50)), "p95_ms": float(np.percentile(values_ms, 95)),
            "p99_ms": float(np.percentile(values_ms, 99)), "max_ms": float(values_ms.max())}


#: The fewest requests a history-length bin needs for its P95 to count as the worst bin's.
MIN_BIN_REQUESTS = 50


def benchmark(protocol: Protocol, work_dir: Path, split: Split, model: str, *,
              threads: int, cores: set[int] | None = None) -> dict[str, Any]:
    dataset = split.dataset
    settings = protocol.latency
    edges = [int(e) for e in settings.get("history_bins", [1, 10, 100, 1000])]
    per_bin = int(settings.get("requests_per_bin", 200))
    warmup = int(settings.get("warmup_requests", 20))

    directory = run_root(work_dir, dataset, model, protocol.run_fingerprint(dataset, model)) / f"final-seed{protocol.seeds[0]}"
    checkpoint = directory / "model.zip"
    if not checkpoint.exists():
        raise FileNotFoundError(f"{dataset}/{model}: no saved final model at {checkpoint}; run `final` first")

    if cores:
        os.sched_setaffinity(0, cores)
    torch.set_num_threads(threads)
    registered = model_spec(model)
    trainer = registered.cls.load(checkpoint, device="cpu")
    # A matrix source is projected onto the training catalogue once, untimed: per request it is a column lookup.
    adapter, source = phase_inputs(trainer, split, "test", registered.family)
    predictor = ExcludeSeenPolicy(adapter, protocol.dataset(dataset).exclude_seen)
    sequences = split.data.get("test_source_sequences")
    lengths = sequences.row_lengths if sequences is not None else np.diff(split.data["test_source_matrix"].indptr)
    k = max(protocol.cutoffs)

    rng = _stream(protocol.search_seed, dataset, model, "latency")
    bounds = list(zip(edges, edges[1:] + [None]))
    chosen: list[tuple[str, int]] = []
    for low, high in bounds:
        label = f"{low}+" if high is None else f"{low}-{high - 1}"
        rows = np.flatnonzero((lengths >= low) & ((lengths < high) if high is not None else True))
        if rows.size:
            chosen += [(label, int(r)) for r in rng.choice(rows, size=min(per_bin, rows.size), replace=False)]
    order = rng.permutation(len(chosen))  # interleave bins so drift over time hits all of them alike
    chosen = [chosen[i] for i in order]

    def request(row: int):
        return source.select_rows([row]) if hasattr(source, "select_rows") else source[row:row + 1]

    timings: dict[str, list[float]] = {}
    load_before = os.getloadavg()
    with torch.inference_mode():
        for _, row in chosen[:warmup]:
            predictor.predict_on_batch(request(row), k=k)
        for label, row in chosen:
            single = request(row)
            started = time.perf_counter_ns()
            predictor.predict_on_batch(single, k=k)
            timings.setdefault(label, []).append((time.perf_counter_ns() - started) / 1e6)

    everything = np.concatenate([np.asarray(v) for v in timings.values()])
    result = {
        "dataset": dataset, "model": model, "checkpoint": str(checkpoint),
        "catalog_items": len(split.data["test_item_ids"]), "k": k,
        "threads": threads, "cores": sorted(cores) if cores else None,
        "host": socket.gethostname(), "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__, "settings": settings,
        # how busy the machine was (1-minute load before and after), and which configuration was timed
        "load_average": [round(load_before[0], 2), round(os.getloadavg()[0], 2)],
        "source_trial": read_json(directory / "spec.json").get("source_trial")
        if (directory / "spec.json").exists() else None,
        "overall": _percentiles(everything),
        "by_history_length": {label: _percentiles(np.asarray(v)) for label, v in timings.items()},
    }
    write_json(directory / "latency.json", result)
    return result


def latency_table(protocol: Protocol, work_dir: Path, datasets: list[str], models: list[str]) -> str:
    lines = ["| dataset | model | catalogue | threads | P50 ms | P95 ms | P99 ms | worst-bin P95 |", "|---|---|---|---|---|---|---|---|"]
    for dataset in datasets:
        for model in models:
            path = (run_root(work_dir, dataset, model, protocol.run_fingerprint(dataset, model))
                    / f"final-seed{protocol.seeds[0]}" / "latency.json")
            if not path.exists():
                continue
            r = read_json(path)
            # a P95 of a handful of requests is noise: the worst bin is taken over bins with enough of them
            bins = {label: b for label, b in r["by_history_length"].items() if b["n"] >= MIN_BIN_REQUESTS}
            if bins:
                label, b = max(bins.items(), key=lambda item: item[1]["p95_ms"])
                worst = f"{b['p95_ms']:.2f} ({label}, n {b['n']})"
            else:
                worst = f"— (no bin with {MIN_BIN_REQUESTS} requests)"
            o = r["overall"]
            lines.append(f"| {dataset} | {model} | {r['catalog_items']:,} | {r['threads']} | {o['p50_ms']:.2f} | "
                         f"{o['p95_ms']:.2f} | {o['p99_ms']:.2f} | {worst} |")
    return "\n".join(lines)
