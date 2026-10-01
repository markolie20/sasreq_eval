"""Stage-1 results by repeat rate, within history-length bins: a slice, not a sweep.

Nothing is refitted or rescored. Each test user's history is described by its
length and its repeat share -- the share of its events that repeat an item
already earlier in it -- and the stage-1 final results are cut into cells by
both. Cutting by length too matters: long histories hold more repeats simply
because they are long, so a repeat effect read across all lengths would partly
be a length effect.

Per cell the report gives each model's mean primary metric and each
sequential model's gap to the best non-sequential model, with a paired
bootstrap percentile interval. The best non-sequential model is chosen once
per dataset, on all test users, not per cell, so a small cell cannot pick its
own comparator by chance.

The cells are observational: users with many repeats differ from users with
few in more than their repeats. So the report makes no test and claims no
cause; the manipulated counterpart is the ``repeat_removal`` sweep.

Settings come from an optional ``[repeat_strata]`` section:

    history_bins = [...]   lower edges of the length bins; the last is open
    repeat_bins  = [...]   edges of the repeat-share bins, from 0 to 1
    min_users    = 100     smaller cells are not reported
    n_resamples  = 1999    bootstrap resamples per interval
"""

from __future__ import annotations

import csv
import io
import itertools
from typing import Any

import numpy as np
import pandas as pd

from .ablations import _event_rows, _first_occurrence
from .protocol import Protocol, ProtocolError
from .report import _table, mean_over_seeds
from .runner import final_seeds, load_final_evaluations
from .search import _stream
from .splits import load_split

DEFAULTS = {"history_bins": [1, 5, 10, 20, 50, 100, 200, 500], "repeat_bins": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
            "min_users": 100, "n_resamples": 1999}


def settings(protocol: Protocol) -> dict[str, Any]:
    table = {**DEFAULTS, **protocol.raw.get("repeat_strata", {})}
    unknown = sorted(set(table) - set(DEFAULTS))
    history, repeat = table["history_bins"], table["repeat_bins"]
    if unknown:
        raise ProtocolError(f"[repeat_strata] has unknown keys {unknown}")
    if not history or any(not isinstance(e, int) or e < 1 for e in history) or history != sorted(set(history)):
        raise ProtocolError("[repeat_strata].history_bins must be increasing positive integers")
    if len(repeat) < 2 or repeat[0] != 0 or repeat[-1] != 1 or repeat != sorted(set(repeat)):
        raise ProtocolError("[repeat_strata].repeat_bins must increase from 0 to 1")
    if int(table["min_users"]) < 1 or int(table["n_resamples"]) < 1:
        raise ProtocolError("[repeat_strata].min_users and n_resamples must be positive")
    return table


def repeat_share(sequences) -> np.ndarray:
    """Per row, the share of events that repeat an item earlier in the same history."""
    repeats = np.bincount(_event_rows(sequences), weights=~_first_occurrence(sequences), minlength=sequences.n_rows)
    lengths = sequences.row_lengths
    return np.divide(repeats, lengths, out=np.zeros(lengths.size), where=lengths > 0)


def _bins(values: np.ndarray, edges: list[float], *, closed_top: bool) -> np.ndarray:
    index = np.searchsorted(np.asarray(edges), values, side="right") - 1
    if closed_top:
        index[values == edges[-1]] = len(edges) - 2  # the top edge belongs to the last bin
        index[(values < edges[0]) | (values > edges[-1])] = -1
    else:
        index[values < edges[0]] = -1
    return index


def _labels(edges: list[float], *, closed_top: bool, percent: bool) -> list[str]:
    show = (lambda v: f"{v:.0%}") if percent else str
    if closed_top:
        return [f"{show(a)}–{show(b)}" for a, b in itertools.pairwise(edges)]
    return [f"{a}–{b - 1}" for a, b in itertools.pairwise(edges)] + [f"{edges[-1]}+"]


def _bootstrap(d: np.ndarray, n_resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    means = np.empty(n_resamples)
    step = max(1, min(n_resamples, 2_000_000 // max(d.size, 1)))
    for start in range(0, n_resamples, step):
        size = min(step, n_resamples - start)
        means[start:start + size] = d[rng.integers(0, d.size, size=(size, d.size))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def dataset_strata(protocol: Protocol, work_dir, dataset: str, models: list[str],
                   config: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    primary = protocol.primary_metric
    lines = [f"## {dataset}", ""]
    averaged, missing = {}, []
    for model in models:
        finals = load_final_evaluations(protocol, work_dir, dataset, model)
        if len(finals) == len(final_seeds(protocol, work_dir, dataset)):
            averaged[model] = mean_over_seeds([e for _, e in finals])
        else:
            missing.append(model)
    if missing:
        lines += [f"_Not included, final runs unfinished or skipped: {', '.join(missing)}._", ""]
    families = {m: protocol.model(m).family for m in averaged}
    baselines = [m for m in averaged if families[m] == "matrix"]
    candidates = [m for m in averaged if families[m] == "sequence"]
    if not averaged:
        return "\n".join(lines + ["_No finished final runs._"]), []

    ids = np.asarray(next(iter(averaged.values())).sample_ids)
    for model, result in averaged.items():
        if not np.array_equal(np.asarray(result.sample_ids), ids):
            raise ValueError(f"{dataset}: {model} scored different test users from the others")
    split = load_split(work_dir, dataset, protocol)
    sequences = split.data["test_source_sequences"]
    rows = pd.Index(split.eval_user_ids("test")).get_indexer(ids.astype(str))
    if (rows < 0).any():
        raise ValueError(f"{dataset}: scored users missing from the split's test users")
    lengths, shares = sequences.row_lengths[rows], repeat_share(sequences)[rows]
    del split

    h_edges, r_edges = config["history_bins"], config["repeat_bins"]
    h_bin = _bins(lengths, h_edges, closed_top=False)
    r_bin = _bins(shares, r_edges, closed_top=True)
    h_labels = _labels(h_edges, closed_top=False, percent=False)
    r_labels = _labels(r_edges, closed_top=True, percent=True)
    per_user = {m: np.asarray(r.per_user[primary], dtype=np.float64) for m, r in averaged.items()}
    best = max(baselines, key=lambda m: per_user[m].mean()) if baselines else None

    counts = [[int(np.count_nonzero((h_bin == h) & (r_bin == r))) for r in range(len(r_labels))]
              for h in range(len(h_labels))]
    lines += ["Test users per cell (history length × repeat share):", "",
              _table(["history"] + r_labels, [[h_labels[h]] + [f"{c:,}" for c in counts[h]]
                                              for h in range(len(h_labels))])]

    records = []
    for h in range(len(h_labels)):
        for r in range(len(r_labels)):
            cell = (h_bin == h) & (r_bin == r)
            n = int(cell.sum())
            record = {"dataset": dataset, "history_bin": h_labels[h], "repeat_bin": r_labels[r], "users": n,
                      "best_non_sequential": best}
            if n >= config["min_users"]:
                record.update({f"{m}": float(v[cell].mean()) for m, v in per_user.items()})
                for model in candidates if best else []:
                    d = per_user[model][cell] - per_user[best][cell]
                    low, high = _bootstrap(d, int(config["n_resamples"]),
                                           _stream(protocol.search_seed, "strata", dataset, model, h, r))
                    record.update({f"gap_{model}": float(d.mean()), f"gap_{model}_low": low,
                                   f"gap_{model}_high": high})
            records.append(record)

    if best is None:
        lines += ["", "_No non-sequential model has finished, so there is no gap to show._"]
    for model in candidates if best else []:
        grid = []
        for h in range(len(h_labels)):
            cells = []
            for r in range(len(r_labels)):
                record = records[h * len(r_labels) + r]
                key = f"gap_{model}"
                cells.append("—" if key not in record else
                             f"{record[key]:+.4f} [{record[key + '_low']:+.4f}, {record[key + '_high']:+.4f}]")
            grid.append([h_labels[h]] + cells)
        lines += ["", (f"**{model}** − **{best}** on test {primary}, with a paired bootstrap percentile 95% "
                      f"interval ({int(config['n_resamples']):,} resamples); cells under {config['min_users']} "
                      "users are not shown:"), "",
                  _table(["history"] + r_labels, grid)]
    return "\n".join(lines), records


def build_strata_report(protocol: Protocol, work_dir, datasets: list[str], models: list[str]) -> tuple[str, str]:
    config = settings(protocol)
    header = (f"# Stage-1 results by repeat share\n\nEach test user's history is binned by its length and by "
              f"the share of its events that repeat an earlier item. Primary metric {protocol.primary_metric}, "
              "per-user values averaged over the final seeds. Each sequential model is compared with the best "
              "non-sequential model, chosen per dataset on all test users. The cells are observational, so they carry no test: see the "
              "`repeat_removal` sweep for the manipulated counterpart.\n")
    sections, records = [], []
    for dataset in datasets:
        text, rows = dataset_strata(protocol, work_dir, dataset, models, config)
        sections.append(text)
        records += rows
    buffer = io.StringIO()
    if records:
        leading = ["dataset", "history_bin", "repeat_bin", "users", "best_non_sequential"]
        fields = leading + sorted({key for row in records for key in row} - set(leading))
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return header + "\n" + "\n\n".join(sections) + "\n", buffer.getvalue()
