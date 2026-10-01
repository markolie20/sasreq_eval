"""The analysis report: the data, whether order matters, and the floor, per dataset.

Read it before any model result. The profile says what the data is; the
sequence-signal table says whether order carries information a model could
use; the baselines say how high the floor is. The same analysis is repeated
at every ablation condition, and those numbers appear in each sweep's report.

Differences in the sequence-signal table are paired over the same test users,
with a paired bootstrap percentile interval (descriptive, like the gap): they
are there to be read, not tested.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from compresso_recsys.evaluation import EvaluationResult
from compresso_recsys.stats import compare_models

from .analysis import CONTROLS, floor_of, full_results
from .protocol import Protocol
from .report import _table

#: (field, label, format) of the profile, in the order the audit asks its questions.
PROFILE_ROWS = (
    ("rows", "users", "count"),
    ("events", "events", "count"),
    ("catalogue", "catalogue (items with an event)", "count"),
    ("history_length", "history length, mean", "number"),
    ("history_length_p50", "history length, median", "number"),
    ("history_length_p90", "history length, p90", "number"),
    ("density", "density", "number"),
    ("popularity_gini", "popularity Gini", "number"),
    ("repeat_rate", "repeat rate (events)", "percent"),
    ("users_with_any_repeat", "users with any repeat", "percent"),
    ("self_transition_share", "self-transitions (a → a)", "percent"),
    ("tie_rate", "adjacent pairs tied in time", "percent"),
    ("last_tied", "histories ending in a tie", "percent"),
    ("median_gap_seconds", "median gap between events", "duration"),
    ("gap_under_30_min", "gaps under 30 minutes", "percent"),
    ("gap_over_1_day", "gaps over a day", "percent"),
    ("median_span_days", "median history span", "days"),
    ("new_item_target_share", "targets that are new items", "percent"),
)


def _format(value: Any, kind: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if kind == "count":
        return f"{int(value):,}"
    if kind == "percent":
        return f"{value:.1%}"
    if kind == "duration":
        for unit, seconds in (("d", 86_400), ("h", 3_600), ("min", 60)):
            if value >= seconds:
                return f"{value / seconds:.1f} {unit}"
        return f"{value:.0f} s"
    if kind == "days":
        return f"{value:.1f} days"
    return f"{value:.4g}"


def profile_table(profile: dict[str, dict[str, Any]]) -> str:
    rows = [[label, _format(profile["train"].get(field), kind), _format(profile["test"].get(field), kind)]
            for field, label, kind in PROFILE_ROWS]
    return _table(["", "training data", "test inputs"], rows)


def paired(baseline: EvaluationResult, candidate: EvaluationResult, metric: str, names=("a", "b")):
    return compare_models({names[0]: baseline, names[1]: candidate}, metrics=[metric], reference=names[0],
                          correction=None).comparisons[0]


#: A slice of fewer test users is not reported: its mean would be noise.
MIN_SLICE_USERS = 100


def _slice_mean(result: EvaluationResult, mask: np.ndarray, metric: str) -> float | None:
    values = np.asarray(result.per_user[metric])
    return float(values[mask].mean()) if mask.sum() >= MIN_SLICE_USERS else None


def dataset_analysis(protocol: Protocol, work_dir: Path, dataset: str) -> tuple[str, list[dict[str, Any]]]:
    primary = protocol.primary_metric
    k = primary.split("@")[1]
    shown = [primary] + [f"{m}@{k}" for m in ("recall", "hit_rate") if m in protocol.metrics]
    found = full_results(protocol, work_dir, dataset)
    lines, rows = [f"## {dataset}", ""], []
    if found["profile"] is None:
        return "\n".join(lines + ["_Not analysed yet: run `seqrec-eval analyse`._"]), rows

    lines += ["### Data profile", "", profile_table(found["profile"]), ""]

    baselines = found["baselines"]
    floor = floor_of(baselines, primary, expected=protocol.baselines)
    table = []
    for name in protocol.baselines:
        selected = found["selected"].get(name)
        result = baselines.get(name)
        settings = "—" if selected is None else (", ".join(f"{p}={_setting(v)}" for p, v in selected["params"].items())
                                                   or "none")
        val = "—" if selected is None else f"{selected['val'][primary]:.4f}"
        cells = ["—"] * len(shown) if result is None else [f"{result.metrics[m]:.4f}" for m in shown]
        mark = " **(floor)**" if floor and floor[0] == name else ""
        table.append([f"{name}{mark}", protocol.baseline(name).kind, settings, protocol.baseline(name).trials, val]
                     + cells)
        if result is not None:
            rows.append({"dataset": dataset, "scorer": name, "role": "baseline",
                         "params": selected["params"] if selected else None, **result.metrics})
    lines += ["### Baselines and the floor", "",
              ("Each baseline is searched on validation, exactly as a model is (in full where its settings form "
              "a small grid, by random search otherwise), and scored on all test users with its selected "
              f"setting. The strongest on test {primary} is the floor every model has to beat."), "",
              _table(["baseline", "kind", "selected", "trials", f"val {primary}"] + [f"test {m}" for m in shown],
                     table)]
    if floor is None:
        lines += ["", "_No floor yet: it is the strongest of every baseline, and those marked — have no test "
                  "result._"]

    markov = next((baselines[n] for n in protocol.baselines if protocol.baseline(n).kind == "markov"
                   and n in baselines), None)
    controls = found["controls"]
    if markov is not None:
        signal = [["markov", f"{markov.metrics[primary]:.4f}", "—", "—"]]
        for control in CONTROLS:
            if control in controls:
                c = paired(markov, controls[control], primary, ("markov", control))
                signal.append([control, f"{controls[control].metrics[primary]:.4f}", f"{c.difference:+.4f}",
                               f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}]"])
                rows.append({"dataset": dataset, "scorer": control, "role": "control", **controls[control].metrics})
        lines += ["", "### Sequence signal", "",
                  (f"First-order Markov against itself with order removed (training histories shuffled within "
                  f"each user) and with direction removed (histories reversed), on the same test users. A large "
                  f"drop to *shuffled* is order the data carries; a small drop to *backwards* means much of it is "
                  f"co-occurrence rather than direction. Test {primary}; the difference is control − markov, "
                  "with a paired bootstrap 95% interval."), "",
                  _table(["scorer", f"test {primary}", "difference", "95% bootstrap interval"], signal)]

        markov_name = next(n for n in protocol.baselines if protocol.baseline(n).kind == "markov")
        tied = found.get("tied", {}).get(markov_name)
        if tied is None:
            lines += ["", "_No timestamps, so Markov cannot be split by ties._"]
        else:
            slices = [["histories ending in a tie", f"{int(tied.sum()):,}", _fmt(_slice_mean(markov, tied, primary))]
                      + [_fmt(_slice_mean(controls[c], tied, primary)) if c in controls else "—" for c in CONTROLS],
                      ["histories ending in a real gap", f"{int((~tied).sum()):,}",
                       _fmt(_slice_mean(markov, ~tied, primary))]
                      + [_fmt(_slice_mean(controls[c], ~tied, primary)) if c in controls else "—" for c in CONTROLS]]
            lines += ["", (f"Markov predicts from the last item of a history. Where the last two events share a "
                          "timestamp, which of them is last was decided by the source file, so a score there "
                          f"is not evidence of learned order. Test {primary} by slice; a slice of fewer than "
                          f"{MIN_SLICE_USERS} users is not shown:"), "",
                      _table(["test users whose", "users", "markov"] + list(CONTROLS), slices)]

    diagnostic = found["diagnostic"]
    if diagnostic:
        other = found["other_targets"]
        names = [n for n in list(protocol.baselines) + list(CONTROLS) if n in diagnostic]
        table = [[n, "—" if n not in {**baselines, **controls} else
                  f"{({**baselines, **controls})[n].metrics[primary]:.4f}", f"{diagnostic[n].metrics[primary]:.4f}"]
                 for n in names]
        lines += ["", "### The other target definition", "",
                  (f"The protocol scores against **{protocol.targets}** targets; the same scorers against "
                  f"**{other}** targets, as a diagnostic (test {primary}). *next* is the first thing a user does "
                  "after their history ends; *window* is everything they do in the test window."), "",
                  _table(["scorer", f"{protocol.targets} (primary)", f"{other} (diagnostic)"], table)]
        for name in names:
            rows.append({"dataset": dataset, "scorer": name, "role": f"diagnostic ({other} targets)",
                         **diagnostic[name].metrics})
    return "\n".join(lines), rows


def _setting(value: Any) -> str:
    return f"{value:.4g}" if isinstance(value, float) else repr(value)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def build_analysis_report(protocol: Protocol, work_dir: Path, datasets: list[str]) -> tuple[str, str]:
    header = (f"# Analysis\n\nThe data, whether order matters, and the floor, before any model. Primary metric "
              f"{protocol.primary_metric}, on the same split, test users and evaluator as the models. Every "
              "ablation condition has the same analysis, in that sweep's report.\n")
    sections, records = [], []
    for dataset in datasets:
        text, rows = dataset_analysis(protocol, work_dir, dataset)
        sections.append(text)
        records += rows
    buffer = io.StringIO()
    if records:
        leading = ["dataset", "scorer", "role", "params"]
        fields = leading + sorted({key for row in records for key in row} - set(leading))
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return header + "\n" + "\n\n".join(sections) + "\n", buffer.getvalue()
