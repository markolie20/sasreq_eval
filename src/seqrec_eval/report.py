"""Progress and results, read back from the run directories.

The comparison follows the plan agreed for stage 1. The family of hypotheses
is one dataset: every model's final runs are compared against one reference
model on the primary metric, and Holm corrects within that family only, so each
dataset's conclusions do not pay for the number of datasets in the study.

Every model is also compared with the floor: the strongest non-learned
baseline from the analysis step, scored on the same test users. That is a
second family per dataset, Holm-corrected on its own, because it asks a
second question -- does the model beat the floor -- rather than how it ranks
against the reference.

One test family throughout (:mod:`seqrec_eval.seedstats`, review H23 and the
final review): the estimate is the difference of the seed-averaged per-user
values; the p-value and the interval come from a t-test whose standard error
adds the spread of each model's k seed means to the users' part, with
Welch–Satterthwaite degrees of freedom, so a difference that rests on one lucky
seed is not claimed. Beside it, the users-only p is the same test without the
seed term -- the paired t-test over users -- so the effect of counting the seeds
is visible.

Across datasets nothing is pooled (H51): each dataset answers on its own, and a
last table counts, per model, the datasets where it is significantly better,
not significantly different, or significantly worse than the reference, with the
sign of every difference.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from compresso_recsys.evaluation import EvaluationResult

from .analysis import floor_of, full_results
from .evaluate import other_definition
from .protocol import Protocol
from .results import read_json
from .runner import (
    RunLock,
    final_seeds,
    incomplete_selection_path,
    load_final_evaluations,
    stale_final_seeds,
    plan_trials,
    run_root,
    summarize_trials,
)
from .seedstats import compare as compare_seeds, holm
from .splits import PHASES, split_dir


#: Significance level and interval coverage of every comparison in the reports.
ALPHA = 0.05
CONFIDENCE = 1.0 - ALPHA


def mean_over_seeds(evaluations: list[EvaluationResult]) -> EvaluationResult:
    first = evaluations[0]
    for other in evaluations[1:]:
        if not np.array_equal(np.asarray(other.sample_ids), np.asarray(first.sample_ids)):
            raise ValueError("final runs of one model scored different users; they cannot be averaged")
        if other.target_fingerprint != first.target_fingerprint:
            raise ValueError("final runs of one model scored different targets; they cannot be averaged")
    per_user = {name: np.mean([e.per_user[name] for e in evaluations], axis=0) for name in first.per_user}
    return EvaluationResult(
        metrics={name: float(np.mean(values)) for name, values in per_user.items()},
        per_user=per_user,
        sample_ids=first.sample_ids,
        n_rows=first.n_rows,
        n_scored_rows=first.n_scored_rows,
        required_k=first.required_k,
        metadata={**first.metadata, "averaged_over_seeds": len(evaluations)},
        target_fingerprint=first.target_fingerprint,
    )


def pool_over_seeds(evaluations: list[EvaluationResult]) -> EvaluationResult:
    """Seeds combined per user: :func:`mean_over_seeds` when they scored the same users, else pooled.

    A condition drawn at random per seed (the stratified catalogue) scores a
    different set of users in each seed. Pooling takes every user scored in at
    least one seed and averages their values over the seeds that scored them,
    so a user counts once however many seeds kept their target. Models run on
    the same conditions pool to the same users, so they stay paired. Target
    fingerprints differ between such seeds by construction, so the pooled
    result carries none.
    """
    ids = [np.asarray(e.sample_ids).astype(str) for e in evaluations]
    if all(np.array_equal(ids[0], other) for other in ids[1:]):
        return mean_over_seeds(evaluations)
    for x in ids:
        if np.unique(x).size != x.size:
            raise ValueError("a seed scored the same user twice; its users cannot be pooled")
    union = np.unique(np.concatenate(ids))
    counts = np.zeros(union.size)
    for x in ids:
        counts[np.searchsorted(union, x)] += 1
    per_user = {}
    for name in evaluations[0].per_user:
        total = np.zeros(union.size)
        for e, x in zip(evaluations, ids):
            total[np.searchsorted(union, x)] += np.asarray(e.per_user[name], dtype=np.float64)
        per_user[name] = total / counts
    first = evaluations[0]
    return EvaluationResult(
        metrics={name: float(np.mean(values)) for name, values in per_user.items()},
        per_user=per_user,
        sample_ids=union,
        n_rows=int(union.size),
        n_scored_rows=int(union.size),
        required_k=first.required_k,
        metadata={**first.metadata, "pooled_over_seeds": len(evaluations),
                  "user_seed_units": int(sum(x.size for x in ids))},
        target_fingerprint=None,
    )


def _df(df: float) -> str:
    return "∞" if df == float("inf") or df > 1e6 else f"{df:,.1f}"


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def status_table(protocol: Protocol, work_dir: Path, datasets: list[str], models: list[str]) -> str:
    rows = []
    for dataset in datasets:
        prepared = (split_dir(work_dir, dataset) / "split_info.json").exists()
        for model in models:
            trials = plan_trials(protocol, dataset, model)
            summary = summarize_trials(protocol, work_dir, dataset, model)
            running = sum(RunLock(spec.directory(work_dir)).held_by_other() for spec in trials)
            root = run_root(work_dir, dataset, model, protocol.run_fingerprint(dataset, model))
            seeds = final_seeds(protocol, work_dir, dataset)
            finals = sum((root / f"final-seed{seed}" / "done.json").exists() for seed in seeds)
            skipped = summary.planned and summary.skipped == summary.planned
            best = "—" if summary.best is None else f"#{summary.best.index} ({summary.best_value:.4f})"
            rows.append([
                dataset, model, "yes" if prepared else "no",
                f"{summary.done + summary.skipped}/{summary.planned}", summary.failed, running,
                "skipped" if skipped else f"{finals}/{len(seeds)}", best,
            ])
    return _table(["dataset", "model", "split", "trials", "failed", "running", "finals",
                   f"best (val {protocol.primary_metric})"], rows)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _split_section(protocol: Protocol, work_dir: Path, dataset: str) -> str:
    info_path = split_dir(work_dir, dataset) / "split_info.json"
    if not info_path.exists():
        return "_Split not prepared._"
    info = read_json(info_path)
    resolved = info.get("resolved_build_parameters")
    keys = ["temporal_period_hours", "min_user_support", "item_min_support", "min_value_to_keep",
            "set_all_values_to", "dataset_options"]
    if protocol.dataset(dataset).builder == "amazon2023":
        keys.append("amazon_category")
    if isinstance(resolved, dict):
        settings = ", ".join(f"{k}={resolved.get(k)!r}" for k in keys if k in resolved)
    else:
        settings = str(resolved)
    rows = []
    for phase in PHASES:
        stats = info["stages"][phase]
        history = stats.get("history_length") or {}
        repeat = stats.get("repeat_target_fraction")
        rows.append([phase, f"{stats['rows']:,}", f"{stats['catalog_items']:,}", f"{stats['target_pairs']:,}",
                     "—" if repeat is None else f"{repeat:.1%}",
                     "—" if history.get("p50") is None else f"{history['p50']:.0f} / {history['p90']:.0f}"])
    sampled = info.get("val_rows_sampled")
    return (f"Resolved build settings: {settings}\n\n"
            + _table(["stage", "rows", "catalogue", "target pairs", "repeat targets", "history p50 / p90"], rows)
            + ("" if sampled is None else f"\n\nSearch trials score a fixed sample of {sampled:,} validation rows."))


def code_builds(protocol: Protocol, work_dir: Path, dataset: str, models: list[str]) -> dict[str, int]:
    """The code builds (library and suite) that a dataset's trials and final runs recorded, with run counts."""
    builds: dict[str, int] = {}
    for model in models:
        root = run_root(work_dir, dataset, model, protocol.run_fingerprint(dataset, model))
        records = [spec.directory(work_dir) / "done.json" for spec in plan_trials(protocol, dataset, model)]
        records += [root / f"final-seed{seed}" / "done.json" for seed in final_seeds(protocol, work_dir, dataset)]
        for path in records:
            if not path.exists():
                continue
            code = json.loads(path.read_text()).get("code")
            key = ("not recorded" if code is None else
                   f"compresso-recsys {code['library']['version']} (source {code['library']['code_sha256']}), "
                   f"suite source {code['suite']['code_sha256']}")
            builds[key] = builds.get(key, 0) + 1
    return builds


def _builds_note(builds: dict[str, int]) -> list[str]:
    if not builds:
        return []
    if len(builds) == 1:
        (build, count), = builds.items()
        return ["", f"Code: {build}, for all {count} runs."]
    listed = "; ".join(f"{build}: {count} runs" for build, count in builds.items())
    return ["", f"⚠ These results come from {len(builds)} different code builds ({listed}). Runs of one build are "
                "not comparable with another's unless the difference cannot touch them: rerun the older ones, or "
                "check (review H05)."]


def dataset_report(protocol: Protocol, work_dir: Path, dataset: str, models: list[str],
                   reference: str | None) -> tuple[str, list[dict[str, Any]]]:
    k = protocol.primary_metric.split("@")[1]
    shown = [protocol.primary_metric] + [f"{m}@{k}" for m in ("recall", "calibrated_recall", "hit_rate")
                                         if m in protocol.metrics and f"{m}@{k}" != protocol.primary_metric]
    lines = [f"## {dataset}", "", _split_section(protocol, work_dir, dataset), ""]
    seeds = final_seeds(protocol, work_dir, dataset)
    if len(seeds) > len(protocol.seeds):
        lines += [f"Seeds {list(seeds)}: the protocol's, and {list(seeds[len(protocol.seeds):])} added later "
                  "(`final --add-seeds`).", ""]
    rows, csv_rows, averaged, per_seed, notes, new_rows, scoring = [], [], {}, {}, [], [], None
    for model in models:
        summary = summarize_trials(protocol, work_dir, dataset, model)
        if summary.skipped == summary.planned and summary.planned:
            notes.append(f"- **{model}** skipped: {summary.skip_reason}")
            continue
        finals = load_final_evaluations(protocol, work_dir, dataset, model)  # without stale finals (review N33)
        stale = stale_final_seeds(protocol, work_dir, dataset, model)
        per_metric = {name: [e.metrics[name] for _, e in finals] for name in shown}
        best = "—" if summary.best is None else f"#{summary.best.index}"
        val = "—" if summary.best_value is None else f"{summary.best_value:.4f}"
        rows.append([model, f"{summary.done}/{summary.planned}", best, val,
                     len(finals) if len(finals) == len(seeds) else f"{len(finals)}/{len(seeds)}"]
                    + [_mean_sd(per_metric[name]) for name in shown])
        if summary.unaccepted:
            notes.append(f"- ⛔ **{model}**: {len(summary.unaccepted)} trial(s) failed and are not resolved, so no "
                         "configuration can be selected: rerun them (`search --retry-failed`) or accept them as "
                         "unrunnable (`final --accept-failed`).")
        if summary.accepted:
            reasons = "; ".join(f"trial {f['trial']}: {f['error'][:100]}" for f in summary.accepted["failures"][:3])
            notes.append(f"- **{model}**: {len(summary.accepted['trials'])} trial(s) failed and were accepted as "
                         f"unrunnable on {summary.accepted['accepted_at']} ({reasons}); the search selected from "
                         f"{summary.done} of {summary.planned} trials.")
        root = run_root(work_dir, dataset, model, protocol.run_fingerprint(dataset, model))
        early = incomplete_selection_path(protocol, work_dir, dataset, model)
        if early.exists():
            record = read_json(early)
            notes.append(f"- **{model}**: selected on {record['at']} from an unfinished search ({record['finished']} "
                         f"of {record['planned']} trials, `final --allow-incomplete`), so with a smaller budget than "
                         "the others" + ("; the search has finished since." if summary.done + summary.skipped
                                         + summary.failed >= summary.planned else "."))
        if stale:
            notes.append(f"- ⛔ **{model}**: final seed(s) {stale} were made with another trial's configuration than "
                         f"the one selected now (#{summary.best.index}), so they describe another model. They are "
                         "left out of its values and every comparison; `final` refuses them until they are moved "
                         "aside and redone.")
        failed = [seed for seed in seeds if (root / f"final-seed{seed}" / "failed.json").exists()]
        if failed:
            notes.append(f"- ⛔ **{model}**: final seed(s) {failed} failed, so its test values rest on fewer seeds "
                         "than the protocol asks; rerun them (`final --retry-failed`).")
        waiting = [seed for seed in seeds if seed not in {spec.seed for spec, _ in finals} and seed not in failed
                   and seed not in stale]
        if finals and waiting:
            notes.append(f"- **{model}**: final seed(s) {waiting} have not run yet, so its values and comparisons "
                         f"rest on {len(finals)} of {len(seeds)} seeds for now.")
        for spec, evaluation in finals:
            csv_rows.append({"dataset": dataset, "model": model, "seed": spec.seed,
                             "source_trial": spec.source_trial, **evaluation.metrics})
        if finals:
            averaged[model] = mean_over_seeds([e for _, e in finals])
            per_seed[model] = [e for _, e in finals]
            scoring = scoring or finals[0][1]
        new = load_final_evaluations(protocol, work_dir, dataset, model, stem="test_new")
        if new:
            scored = new[0][1].n_scored_rows
            value = (_mean_sd([e.metrics[protocol.primary_metric] for _, e in new]) if scored
                     else "— (no row has a new target)")
            new_rows.append([model, value, f"{scored:,}"])

    lines.append(_table(["model", "trials", "selected", f"val {protocol.primary_metric}", "seeds"]
                        + [f"test {name}" for name in shown], rows))
    excluded = None if scoring is None else scoring.metadata.get("rows_unrecommendable_next")
    if excluded is not None and scoring.metadata.get("definition") == "next":
        scored = scoring.metadata.get("rows_sampled") or scoring.n_scored_rows
        lines += ["", f"Next-item metrics over {scored:,} test users. {excluded:,} more are left out: their next "
                      "item is one no model can recommend (first seen after training, or deleted by the builder's "
                      "new-item filter), so they would score 0 for every model."]
    if notes:
        lines += ["", *notes]
    lines += _builds_note(code_builds(protocol, work_dir, dataset, models))

    if new_rows:
        lines += ["", "Targets not already in the history (diagnostic, `exclude_seen=true`):", "",
                  _table(["model", f"test {protocol.primary_metric}", "rows with a new target"], new_rows)]

    outcome = {}
    if reference and reference in averaged and len(averaged) > 1:
        primary = protocol.primary_metric
        seeded = [compare_seeds(per_seed[model], per_seed[reference], primary, names=(model, reference),
                                confidence=CONFIDENCE)
                  for model in averaged if model != reference]
        adjusted = holm([c.p_value for c in seeded])
        users_only = holm([c.p_users for c in seeded])
        reference_mean = averaged[reference].metrics[primary]
        comparison_rows = [[c.candidate, f"{c.difference:+.4f}",
                            "—" if not reference_mean else f"{c.difference / reference_mean:+.1%}",
                            f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}]", f"{c.se_users:.4f}", f"{c.se_seeds:.4f}",
                            _df(c.df), f"{p:.4g}", f"{u:.4g}", "yes" if p <= ALPHA else "no"]
                           for c, p, u in zip(seeded, adjusted, users_only)]
        outcome = {c.candidate: (c.difference, p) for c, p in zip(seeded, adjusted)}
        one_seed = sorted({c.candidate for c in seeded if c.seeds[0] == 1} | ({reference} if seeded
                                                                              and seeded[0].seeds[1] == 1 else set()))
        lines += ["", (f"Comparison against **{reference}** on test {primary}: the difference of the per-user "
                      "values averaged over seeds. p: two-sided t-test whose standard error adds the spread of "
                      "each model's seed means to the users' part (SE users, SE seeds), Welch–Satterthwaite df, "
                      "Holm within this dataset; significance is read from it. Interval: that test's "
                      f"{CONFIDENCE:.0%} t interval. Users-only p: the same test without the seed term (the "
                      "paired t-test over users), Holm-adjusted, for comparison.")
                  + (f" With one seed ({', '.join(one_seed)}) there is no seed spread to count: for those the "
                     "test is over users only." if one_seed else ""), "",
                  _table(["model", "difference", "relative", f"{CONFIDENCE:.0%} interval", "SE users",
                          "SE seeds", "df", "adjusted p", "users-only p", "significant"],
                         comparison_rows)]
    elif reference and reference not in averaged:
        lines += ["", f"_No comparison yet: the reference model {reference!r} has no finished final runs._"]

    other = other_definition(protocol.targets)
    diagnostic_rows = []
    for model in models:
        finals = load_final_evaluations(protocol, work_dir, dataset, model, stem=f"test_{other}")
        if finals:
            primary_finals = load_final_evaluations(protocol, work_dir, dataset, model)
            diagnostic_rows.append([model, _mean_sd([e.metrics[protocol.primary_metric] for _, e in primary_finals]),
                                    _mean_sd([e.metrics[protocol.primary_metric] for _, e in finals])])
    if diagnostic_rows:
        lines += ["", (f"The same final runs against **{other}** targets, as a diagnostic (the protocol scores "
                      f"**{protocol.targets}** targets), test {protocol.primary_metric}:"), "",
                  _table(["model", f"{protocol.targets} (primary)", f"{other} (diagnostic)"], diagnostic_rows)]

    baselines = full_results(protocol, work_dir, dataset)["baselines"]
    floor = floor_of(baselines, protocol.primary_metric, expected=protocol.baselines)
    if floor is None:
        missing = ", ".join(name for name in protocol.baselines if name not in baselines)
        lines += ["", (f"_No floor yet: {missing or 'no baseline'} not analysed. The floor is the strongest of "
                       "every baseline, so it waits for all of them (`seqrec-eval analyse`)._")]
    elif averaged:
        name, result = floor
        key = f"floor: {name}"
        primary = protocol.primary_metric
        # the floor is a deterministic baseline, scored once: it contributes no seed term
        seeded = [compare_seeds(per_seed[model], [result], primary, names=(model, key), confidence=CONFIDENCE)
                  for model in averaged]
        adjusted = holm([c.p_value for c in seeded])
        users_only = holm([c.p_users for c in seeded])
        lines += ["", (f"Against the floor, **{name}** at test {primary} {result.metrics[primary]:.4f} (the "
                      "strongest non-learned baseline, see the analysis report). p and interval: the seed-aware "
                      f"t-test as above, Holm across these {len(seeded)} comparisons."), "",
                  _table(["model", "difference", f"{CONFIDENCE:.0%} interval", "adjusted p",
                          "users-only p", "beats the floor"],
                         [[c.candidate, f"{c.difference:+.4f}", f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}]",
                           f"{p:.4g}", f"{u:.4g}", "yes" if p <= ALPHA and c.difference > 0 else "no"]
                          for c, p, u in zip(seeded, adjusted, users_only)])]
    return "\n".join(lines), csv_rows, outcome


def _across_datasets(outcomes: dict[str, dict[str, tuple[float, float]]], datasets: list[str],
                     reference: str) -> str:
    """Per model, in how many datasets it is significantly better than, not different from, or worse than
    the reference, and the sign of each difference. Nothing is pooled: the rule fixed in advance (H51) is that
    each dataset answers on its own, and an overall claim needs the same answer on most of them."""
    models = sorted({model for outcome in outcomes.values() for model in outcome})
    if not models:
        return ""
    rows = []
    for model in models:
        cells, better, worse, level = [], 0, 0, 0
        for dataset in datasets:
            if model not in outcomes.get(dataset, {}):
                cells.append("—")
                continue
            difference, p = outcomes[dataset][model]
            significant = p <= ALPHA
            better += significant and difference > 0
            worse += significant and difference < 0
            level += not significant
            cells.append(("+" if difference > 0 else "−") + ("*" if significant else ""))
        rows.append([model, *cells, better, level, worse])
    return "\n".join(["## Across datasets", "", (
        f"Against **{reference}**, per dataset: the sign of the difference, * where significant (Holm within the "
        "dataset). Nothing is pooled across datasets (H51): each answers on its own, and a claim about all of them "
        "needs the same answer on most of them, which the counts show."), "",
        _table(["model", *datasets, "better", "not different", "worse"], rows)])


def build_report(protocol: Protocol, work_dir: Path, datasets: list[str], models: list[str],
                 reference: str | None) -> tuple[str, str]:
    sections, all_rows, outcomes = [], [], {}
    for dataset in datasets:
        section, rows, outcomes[dataset] = dataset_report(protocol, work_dir, dataset, models, reference)
        sections.append(section)
        all_rows += rows
    if reference and len(datasets) > 1:
        sections.append(_across_datasets(outcomes, datasets, reference))
    # the full path and the file's hash: a report of a quick local run (work-local/<dataset>/protocol.toml) must
    # not read like one of the real protocol.toml
    digest = hashlib.sha256(protocol.path.read_bytes()).hexdigest()[:12]
    header = (f"# Stage-1 results\n\nProtocol `{protocol.path.resolve()}` (file sha256 {digest}), version "
              f"{protocol.version}. "
              f"Models are ranked on {protocol.primary_metric}; everything else is a diagnostic (§5.3). "
              "Test values are the mean ± sd over final seeds of the configuration selected on validation"
              + (", refitted on train and validation before test; validation values are from the search."
                 if protocol.refit else
                 ", fitted on the training window only: every tested model is one window stale.") + "\n")
    markdown = header + "\n" + "\n\n".join(sections) + "\n"
    buffer = io.StringIO()
    if all_rows:
        fields = sorted({key for row in all_rows for key in row},
                        key=lambda key: (key not in ("dataset", "model", "seed", "source_trial"), key))
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    return markdown, buffer.getvalue()
