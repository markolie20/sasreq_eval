"""The report of one ablation sweep, read back from its run directories.

Per dataset it has four parts, in the order they should be read.

* **Manipulation check.** The five characteristics at every level, as the
  relative change from the full data. A characteristic that is neither the
  sweep's target nor declared in ``expected_to_move`` is flagged when it moves
  beyond ``manipulation_tolerance``. Beside them, how much of the data the
  condition actually touched: the share of histories changed, and how far back
  the kept events still reach.
* **Each level against the full data.** One family per sweep and dataset: every
  model's every level is compared with that model's full-data reference on the
  primary metric, and Holm corrects across the whole family.
* **The knee**, fixed in advance: walking from the full data to ever more
  reduced levels, the most reduced level whose interval for the loss stays
  above −δ, stopping at the first that does not (see :func:`find_knee`).
  Beside it, the power of that test -- whether the data could have shown it at
  all -- and, as a sensitivity, the knee at other margins. Only for transforms
  with an ordered axis.
* **The gap**, which is what the sweep is about: each sequential model minus
  the comparator fixed in advance (ELSA), at every level, tested and
  Holm-corrected across the sweep. Beside it, descriptive only, the gap to the
  best non-sequential model at each level, chosen on test. The floor -- the
  strongest non-learned baseline at that level, from the analysis step -- is
  shown with them.

One test family throughout (:mod:`seqrec_eval.seedstats`): per-user values are
averaged over seeds, and every p-value and interval comes from a t-test whose
standard error adds the seeds' spread to the users', with Welch–Satterthwaite
degrees of freedom. In an ablation the seeds are **paired**: seed s of a level
is the full data's seed s, retrained on the level or rescored, and two models
at one level of a random sweep share its subsample s, so the seed term is the
spread of the per-seed differences (final review A1). The users-only p beside
each is the same test without the seed term, the paired t-test over users.
Runs that failed or have not finished are listed per model, never dropped
silently.

A catalogue sweep drops the targets of removed items, so its levels are not
scored against the same targets as the full data, or -- with a stratified
sample per seed -- as each other. One set of users cannot survive every
random catalogue (H01b), so each condition is scored on its own users (a
known next target still in the catalogue), seeds are pooled per user, and
levels are compared through the gap, which pairs models within a level. The
level-against-full test is not run for it. Target fingerprints are set aside,
since the targets differ by construction.

Every level also reports how many users it was scored on; a level below
``min_level_users`` (default 1,000) is marked descriptive, and no claim rests
on it.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
from compresso_recsys.evaluation import EvaluationResult

from .ablations import (
    CHARACTERISTICS,
    DEFAULT_KNEE_MARGIN,
    DEFAULT_MIN_LEVEL_USERS,
    DEFAULT_TOLERANCE,
    REFERENCE,
    ablation_root,
    check_ablation,
    condition_name,
    data_seeds,
    expected_to_move,
    level_label,
    manipulation_check,
    per_condition_users,
    plan_ablation,
    scope_of,
    sweep_seeds,
)
from .analysis import CONTROLS, condition_results, floor_of, mean_over_subsamples
from .analysis_report import MIN_SLICE_USERS
from .plots import gap_figure
from .protocol import Protocol
from .report import ALPHA, CONFIDENCE, _mean_sd, _table, pool_over_seeds
from .results import evaluation_exists, load_evaluation, read_json
from .runner import made_with_another_selection
from .seedstats import compare as compare_seeds
from .seedstats import holm, noninferiority, variance

N_RESAMPLES = 9_999

#: Margins the knee is also shown at, as a sensitivity beside the one fixed in advance (``knee_margin``).
KNEE_SENSITIVITY = (0.05, 0.10, 0.20)

_SHORT = {"history_length": "history", "catalogue": "catalogue", "density": "density",
          "popularity_gini": "Gini", "repeat_rate": "repeats"}


def noninferiority_p(differences: np.ndarray, margin: float, *, rng: np.random.Generator,
                     n_resamples: int = N_RESAMPLES, batch_size: int = 64) -> float:
    """One-sided paired sign-flip randomisation test that the loss is smaller than ``margin``.

    ``differences`` are per-user values of candidate minus baseline. The null
    is that the mean difference is ``-margin`` or lower, i.e. the candidate is
    at least ``margin`` worse. Shifting every difference by ``+margin`` puts
    the boundary of that null at zero, where each shifted difference is taken
    to be equally likely positive or negative, so random sign flips give its
    null distribution. A small p-value is evidence that the loss is below the
    margin. The p-value counts the observed statistic among the resamples, so
    it is never zero.
    """
    shifted = np.asarray(differences, dtype=np.float64) + float(margin)
    observed = shifted.mean()
    exceed, done = 0, 0
    while done < n_resamples:
        size = min(batch_size, n_resamples - done)
        signs = rng.integers(0, 2, size=(size, shifted.size), dtype=np.int8) * 2 - 1
        exceed += int(np.count_nonzero((signs * shifted).mean(axis=1) >= observed))
        done += size
    return (1 + exceed) / (n_resamples + 1)


def noninferiority_power(differences: np.ndarray, margin: float, alpha: float) -> float:
    """The chance that :func:`noninferiority_p` rejects at ``alpha`` for a level that loses nothing at all.

    With n paired per-user differences of standard deviation s, their mean has
    standard error s / √n. The test shows "the loss is below the margin δ"
    when the observed mean difference lies above −δ + z(1−α) · s / √n. For a
    level whose true mean difference is zero, that happens with probability

        power = Φ( δ · √n / s  −  z(1−α) )

    (normal approximation; Φ is the standard normal distribution function and
    z(1−α) its 1−α quantile, 1.645 at α = 0.05). Read the other way, the
    smallest margin that reaches a given power is

        δ = ( z(1−α) + z(power) ) · s / √n ,

    2.93 · s / √n for 90% power at α = 0.05. Where the power is low, a knee
    that stays at the full data means the test could not tell, not that the
    reduced levels are worse.
    """
    differences = np.asarray(differences, dtype=np.float64)
    spread = float(differences.std(ddof=1)) if differences.size > 1 else 0.0
    if spread == 0.0:
        return 1.0 if margin > 0 else 0.0
    normal = NormalDist()
    return float(normal.cdf(margin * math.sqrt(differences.size) / spread - normal.inv_cdf(1.0 - alpha)))


@dataclass(frozen=True)
class Knee:
    #: The level every other is compared with: the full data (the largest position).
    reference: str
    margin: float                 # δ in metric units
    knee: str
    #: Every level tested, largest first: (label, mean difference from the reference, p).
    tested: tuple[tuple[str, float, float], ...]
    #: The first level that could not show a loss below δ, or ``None``.
    stopped_at: str | None
    #: Per level, the power of its test: see :func:`noninferiority_power`.
    power: dict[str, float]


def find_knee(per_user: dict[str, np.ndarray], position: dict[str, float], *, margin_fraction: float,
              alpha: float, rng: Callable[[str], np.random.Generator] | None,
              seed_means: dict[str, list[float]] | None = None, paired: bool = False) -> Knee:
    """The most reduced level from which the curve is demonstrably within δ of the full data.

    ``per_user`` holds each level's per-user values, aligned on the same users;
    ``position`` places each level on the axis, larger meaning closer to the
    full data (the reference at infinity). δ is ``margin_fraction`` of the
    full data's mean.

    Every level is compared with the **full data**, not with the level that
    scored best. The best-looking of several noisy levels is lucky by
    construction, so comparing with it biases every test against the others
    (the winner's curse); the full data is fixed in advance. It is also the
    question asked: how far the data can be reduced without losing against
    what the model has now.

    Levels are tested from the largest position down, each with
    :func:`noninferiority_p` against the full data, and testing stops at the
    first that fails. The knee **starts at the full data** and moves down only
    through levels that passed, so it is the last level that passed, and the
    full data itself when the first test fails: nothing was then shown. That
    rule is two things at once. It asks the curve to stay flat from the knee
    towards the full data, so a reduced level cannot become the knee by one
    lucky result beyond a level that failed. And it is the fixed-sequence
    procedure: hypotheses tested in an order fixed in advance, stopping at the
    first non-rejection, keep the family-wise error at ``alpha`` without any
    correction.

    Unlike "the interval contains zero", this does not reward noise. With few
    users, few levels can show their loss is under δ, so the knee stays high;
    with many, it converges on where the loss really crosses δ. ``power``
    says which of the two a result is.

    With ``seed_means`` (each level's mean score per seed), each test is the
    seed-aware one-sided t-test of :func:`seqrec_eval.seedstats.noninferiority`
    rather than the sign-flip test over users: a level must then be within δ
    by more than its seeds' spread too, and the power counts it as well.
    ``paired``: seed s of every level belongs with seed s of the full data
    (the same trained model or the same seed), as in every ablation; the seed
    term is then the spread of the per-seed differences (final review A1).
    Passing a one-sided test at α is the same as the lower end of the
    two-sided 1 − 2α interval for the loss lying above −δ.
    """
    reference = max(per_user, key=lambda label: position[label])
    margin = margin_fraction * float(np.mean(per_user[reference]))
    ordered = sorted((label for label in per_user if label != reference), key=lambda label: -position[label])
    power, seeded = {}, {}
    for label in ordered:
        differences = per_user[label] - per_user[reference]
        if seed_means is None:
            power[label] = noninferiority_power(differences, margin, alpha)
        else:
            spread = variance(differences, seed_means[label], seed_means[reference], paired=paired)
            seeded[label], power[label] = noninferiority(float(differences.mean()), spread, margin, alpha)
    knee, tested, stopped_at = reference, [], None
    for label in ordered:
        differences = per_user[label] - per_user[reference]
        p = seeded[label] if seed_means is not None else noninferiority_p(differences, margin, rng=rng(label))
        tested.append((label, float(differences.mean()), p))
        if p > alpha:
            stopped_at = label
            break
        knee = label
    return Knee(reference=reference, margin=margin, knee=knee, tested=tuple(tested), stopped_at=stopped_at,
                power=power)

def _load(work_dir: Path, specs) -> tuple[list[tuple[Any, EvaluationResult]], list[Any]]:
    """The finished runs of ``specs`` with their evaluations, and the specs whose finished run was made under
    another selection than the current one: it describes another model, so it is not loaded (review N33)."""
    out, stale = [], []
    for spec in specs:
        directory = spec.directory(work_dir)
        if (directory / "done.json").exists() and evaluation_exists(directory / "test"):
            if made_with_another_selection(spec, directory):
                stale.append(spec)
            else:
                out.append((spec, load_evaluation(directory / "test")))
    return out, stale


def _share(value: float | None) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.0%}"


def _characteristics_section(protocol: Protocol, work_dir: Path, sweep: str, dataset: str,
                             labels: list[str], target: str | None, expected: tuple[str, ...],
                             tolerance: float, direction: int = -1) -> str:
    folder = ablation_root(protocol, work_dir, sweep, dataset) / "conditions"
    reference_path = folder / f"{REFERENCE}.json"
    if not reference_path.exists():
        return "_No condition measured yet._"
    reference = read_json(reference_path)["characteristics"]
    edits = [("train", "rows_changed", "train rows changed"), ("test", "rows_changed", "test rows changed"),
             ("test", "span_kept", "test span kept")]
    header = (["condition"] + [f"{part} {_SHORT[name]}" for part in ("train", "test") for name in CHARACTERISTICS]
              + [title for _, _, title in edits])
    rows = [[REFERENCE] + [f"{reference[part][name]:.4g}" for part in ("train", "test") for name in CHARACTERISTICS]
            + ["0%", "0%", "100%"]]
    flagged = []
    for name in labels:
        path = folder / f"{name}.json"
        if not path.exists():
            rows.append([name] + ["—"] * (len(header) - 1))
            continue
        measured = read_json(path)["characteristics"]
        check = manipulation_check(reference, measured, target, tolerance, expected, direction)
        cells = []
        for part in ("train", "test"):
            for characteristic in CHARACTERISTICS:
                relative = check["change"][part][characteristic]
                mark = " ⚠" if f"{part} {characteristic}" in check["moved"] else ""
                cells.append("—" if not math.isfinite(relative) else f"{relative:+.1%}{mark}")
        cells += [_share(measured[part].get(key)) for part, key, _ in edits]
        rows.append([name] + cells)
        if check["moved"]:
            flagged.append(f"- **{name}** also moved: {', '.join(check['moved'])}")
        if check["wrong_way"]:
            flagged.append(f"- ⛔ **{name}**: the target moved the wrong way ({', '.join(check['wrong_way'])}). "
                           "That is not a finding: it points to a bug or an unforeseen interaction, and this "
                           "sweep's results should not be read until it is understood.")
    aim = (f"The sweep targets **{target}**" if target else "The sweep should move none of them")
    declared = f", and declares that it also moves {', '.join(expected)}" if expected else ""
    text = (f"Full data in absolute terms, every level as the relative change from it. {aim}{declared}; "
            f"⚠ marks any other characteristic that moved by more than {tolerance:.0%}. "
            "*Rows changed* is the share of histories that differ from the original in any way; *span kept* "
            "is how much of the original history lies between the first and last kept event, which "
            "separates dropping the oldest events (low) from thinning throughout (high).\n\n"
            + _table(header, rows))
    return text + ("\n\n" + "\n".join(flagged) if flagged else "")


def dataset_ablation(protocol: Protocol, work_dir: Path, sweep: str, dataset: str, models: list[str],
                     comparator: str = "elsa") -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    ablation = protocol.ablation(sweep)
    transform = check_ablation(ablation)
    tolerance = float(ablation.raw.get("manipulation_tolerance", DEFAULT_TOLERANCE))
    margin_fraction = float(ablation.raw.get("knee_margin", DEFAULT_KNEE_MARGIN))
    primary = protocol.primary_metric
    labels = [condition_name(level, seed) for level in ablation.levels
              for seed in data_seeds(protocol, work_dir, sweep, dataset)]
    seeds = sweep_seeds(protocol, work_dir, sweep, dataset)
    level_of = {level_label(level): level for level in ablation.levels}
    columns = [REFERENCE] + [level_label(level) for level in ablation.levels]

    minimum = int(ablation.raw.get("min_level_users", DEFAULT_MIN_LEVEL_USERS))
    own_users = per_condition_users(protocol, sweep)
    lines = [f"## {dataset}", ""]
    if len(seeds) > len(ablation.seeds):
        added = list(seeds[len(ablation.seeds):])
        lines += [(f"Seeds {list(seeds)}: the sweep's, and {added} added later (`ablate --add-seeds`). Every "
                   "condition enters the comparisons only once all of them have finished."), ""]
    rows_path = ablation_root(protocol, work_dir, sweep, dataset) / "test_rows.npy"
    if own_users:
        lines += [("Each condition is scored on **its own users**: those with a known next target still in its "
                  "catalogue and a history item left. The targets change with the level, so one set of users "
                  "cannot be kept across levels (H01b); models are compared within a level. Seeds of a random "
                  "catalogue score different users and are pooled per user."), ""]
    elif rows_path.exists():
        lines += [f"Every condition is scored on the same {np.load(rows_path).size:,} test users.", ""]
    lines += ["### Manipulation check", "",
              _characteristics_section(protocol, work_dir, sweep, dataset, labels, transform.target,
                                       expected_to_move(ablation), tolerance, transform.direction), ""]

    averaged: dict[str, dict[str, EvaluationResult]] = {}
    seeds_of: dict[str, dict[str, list[EvaluationResult]]] = {}
    seeds_done: dict[str, dict[str, list[float]]] = {}
    incomplete: dict[str, list[str]] = {}
    csv_rows, notes = [], []
    for model in models:
        try:
            plans = plan_ablation(protocol, work_dir, sweep, dataset, model)
        except RuntimeError as error:
            notes.append(f"- **{model}**: {error}")
            continue
        if not plans:
            notes.append(f"- **{model}** was skipped in stage 1")
            continue
        by_column: dict[str, list] = {}
        failed, pending, stale = [], [], []
        for name, specs in plans.items():
            column = name.split("/")[0]
            finished, old = _load(work_dir, specs)
            stale += [(name, spec.seed) for spec in old]
            if transform.changes_targets:
                finished = [(spec, replace(e, target_fingerprint=None)) for spec, e in finished]
            by_column.setdefault(column, []).extend(finished)
            done = {spec.seed for spec, _ in finished} | {spec.seed for spec in old}
            for spec in specs:
                if spec.seed in done:
                    continue
                path = spec.directory(work_dir) / "failed.json"
                if path.exists():
                    failed.append((name, spec.seed, str(read_json(path).get("error", ""))))
                else:
                    pending.append((name, spec.seed))
        averaged[model], seeds_of[model], seeds_done[model] = {}, {}, {}
        for column, finished in by_column.items():
            seeds_done[model][column] = [e.metrics[primary] for _, e in finished]
            for spec, evaluation in finished:
                csv_rows.append({"dataset": dataset, "model": model, "condition": column, "seed": spec.seed,
                                 "source_trial": spec.source_trial, **evaluation.metrics})
            if len(finished) == len(seeds):
                # seed by seed in the sweep's order, so paired comparisons align seed s with seed s
                ordered = sorted(finished, key=lambda pair: seeds.index(pair[0].seed))
                averaged[model][column] = pool_over_seeds([e for _, e in ordered])
                seeds_of[model][column] = [e for _, e in ordered]
        incomplete[model] = [c for c in columns if c not in averaged[model]]
        # nothing leaves a comparison silently (final review A5)
        if failed:
            shown = "; ".join(f"{name} seed {seed}: {error[:90]}" for name, seed, error in failed[:3])
            notes.append(f"- ⛔ **{model}**: {len(failed)} run(s) failed ({shown}). Their levels are left out of the "
                         "comparisons, the gap and the knee until rerun (`ablate --retry-failed`).")
        if stale:
            shown = ", ".join(f"{name} seed {seed}" for name, seed in stale[:4]) + (", …" if len(stale) > 4 else "")
            notes.append(f"- ⛔ **{model}**: {len(stale)} run(s) were made with another configuration than the one "
                         f"selected now ({shown}), so they describe another model. Their levels are left out of the "
                         "comparisons, the gap and the knee; `ablate` refuses them until they are moved aside and "
                         "redone.")
        if pending:
            names = sorted({name.split("/")[0] for name, _ in pending}, key=columns.index)
            notes.append(f"- **{model}**: {len(pending)} run(s) not finished yet, at {', '.join(names)}; those "
                         "levels enter the comparisons once every seed has.")

    lines += [f"### Test {primary}", "", ("Mean ± sd over seeds; a condition enters the comparisons "
              "below only once every seed has finished.")
              + (" With one seed there is no seed spread: every test below is over users only."
                 if len(seeds) == 1 else ""), ""]
    users = {c: next((r[c].n_rows for r in averaged.values() if c in r), None) for c in columns}
    per_seed_users = {c: next(([e.n_rows for e in s[c]] for s in seeds_of.values() if c in s), None)
                      for c in columns}

    def users_cell(column: str) -> str:
        if users[column] is None:
            return "—"
        counts = per_seed_users[column]
        if own_users and counts and len(counts) > 1 and min(counts) != max(counts):
            # seeds of a random catalogue score different users: the pooled count is their union
            return f"{users[column]:,} ({min(counts):,}–{max(counts):,} per seed)"
        return f"{users[column]:,}"

    lines.append(_table(["model"] + columns,
                        [[model] + [_mean_sd(seeds_done[model].get(c, [])) for c in columns] for model in seeds_done]
                        + [["*users scored*"] + [users_cell(c) for c in columns]]))
    low = {c for c in columns if users[c] is not None and users[c] < minimum}
    if low:
        lines += ["", (f"Scored on fewer than {minimum:,} users, so **descriptive only**: "
                      f"{', '.join(c for c in columns if c in low)}. No claim rests on these levels: their tests "
                      "below say so in place of a verdict.")]
    if notes:
        lines += ["", *notes]

    def verdict(p: float, *involved: str) -> str:
        if any(c in low for c in involved):
            return "— (descriptive)"
        return "yes" if p <= ALPHA else "no"

    # each level against the full data: one Holm family for the sweep on this dataset. Seed s of a level and
    # seed s of the full data are the same model, or the same seed, so the seeds are paired (review A1). Where
    # each condition has its own users there is nothing to pair across levels, so the test is not run.
    raw = []
    if own_users:
        lines += ["", ("_No level-against-full test: each level is scored on its own users, and its targets "
                      "differ from the full data's, so levels are compared through the gap below._")]
    for model, results in ({} if own_users else averaged).items():
        if REFERENCE not in results:
            continue
        for column in columns[1:]:
            if column in results:
                raw.append((model, compare_seeds(seeds_of[model][column], seeds_of[model][REFERENCE], primary,
                                                 names=(column, REFERENCE), paired=True, confidence=CONFIDENCE,
                                                 pooled=(results[column], results[REFERENCE]))))
    if raw:
        adjusted = holm([c.p_value for _, c in raw])
        users_only = holm([c.p_users for _, c in raw])
        rows = [[model, c.candidate, f"{c.difference:+.4f}", f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}]",
                 f"{c.se_seeds:.4f}", f"{p:.4g}", f"{u:.4g}", verdict(p, c.candidate)]
                for (model, c), p, u in zip(raw, adjusted, users_only)]
        lines += ["", "### Each level against the full data", "",
                  (f"{primary} difference (level − full) of the seed-averaged values. p: two-sided seed-aware "
                  "t-test, users and seeds, the seeds paired (seed s of a level is the full data's seed s, "
                  f"retrained or rescored); Holm across all {len(raw)} comparisons of this sweep on {dataset}; "
                  f"significance is read from it. Interval: that test's {CONFIDENCE:.0%} t interval. Users-only "
                  "p: the paired t-test over users alone, Holm-adjusted the same way."), "",
                  _table(["model", "level", "difference", f"{CONFIDENCE:.0%} interval", "SE seeds", "adjusted p",
                          "users-only p", "significant"], rows)]

    # the knee
    if transform.position is not None:
        rows, sensitivity, skipped = [], [], []
        margins = sorted({margin_fraction, *KNEE_SENSITIVITY})
        for model, results in averaged.items():
            if incomplete[model]:
                skipped.append(f"**{model}** (incomplete: {', '.join(incomplete[model])})")
                continue  # the knee needs the whole curve
            per_user = {c: np.asarray(r.per_user[primary], dtype=np.float64) for c, r in results.items()}
            ids = np.asarray(results[REFERENCE].sample_ids)
            if any(not np.array_equal(np.asarray(r.sample_ids), ids) for r in results.values()):
                raise ValueError(f"{sweep}/{dataset}/{model}: conditions scored different users")
            position = {c: (math.inf if c == REFERENCE else transform.position(level_of[c])) for c in results}
            seed_means = {c: [float(np.mean(e.per_user[primary])) for e in seeds_of[model][c]] for c in results}
            knees = {m: find_knee(per_user, position, margin_fraction=m, alpha=ALPHA, rng=None,
                                  seed_means=seed_means, paired=True) for m in margins}
            knee = knees[margin_fraction]
            stopped = ("—" if knee.stopped_at is None else
                       next(f"{label} (Δ {d:+.4f}, p {p:.3g})" for label, d, p in knee.tested
                            if label == knee.stopped_at))
            powers = list(knee.power.values())
            label = f"**{knee.knee}**" + (" (descriptive)" if low else "")
            rows.append([model, f"{knee.margin:.4f}", label, stopped, f"{min(powers):.2f} to {max(powers):.2f}"])
            sensitivity.append([model] + [knees[m].knee for m in margins])
        if rows or skipped:
            lines += ["", "### Knee", "",
                      (f"δ is {margin_fraction:.0%} of each model's {primary} on the full data, fixed in advance. "
                      "Walking from the full data to ever more reduced levels, the knee is the most reduced level "
                      f"whose {1 - 2 * ALPHA:.0%} interval for the loss (level − full) stays above −δ, stopping at "
                      "the first level whose interval does not; the full data when the first already fails. That "
                      f"is a one-sided seed-aware t-test of \"the loss is smaller than δ\" at α = {ALPHA:g} for each "
                      "level, users and seeds counted, the seeds paired, in an order fixed in advance, which keeps "
                      "the family-wise error at α without correction. Power is the chance that this test shows "
                      "\"within δ\" for a level that loses nothing, from that level's spread over users and seeds, "
                      "lowest to highest over the levels: where it is low, a knee at the full data means the test "
                      "could not tell.")
                      + (" No knee for " + "; ".join(skipped) + "." if skipped else ""), ""]
            if rows:
                lines += [_table(["model", "δ", "knee", "first level that failed", "power of the test"], rows), "",
                          ("Sensitivity, not the result fixed in advance: the knee at other margins, as a share of "
                           "the full data's value."), "",
                          _table(["model"] + [f"δ = {m:.0%}" for m in margins], sensitivity)]

    # the gap: the question is sequential against ELSA, fixed in advance and tested; the best non-sequential
    # model per level is chosen on test, so that gap is descriptive. Two models at one level of a random sweep
    # share its subsample s, so there the seeds are paired.
    paired_gap = data_seeds(protocol, work_dir, sweep, dataset) != [None]
    families = {model: protocol.model(model).family for model in averaged}
    tested, gap_rows, descriptive = [], [], []
    for column in columns:
        matrix = {m: r[column] for m, r in averaged.items() if families[m] == "matrix" and column in r}
        for model, results in averaged.items():
            if families[model] != "sequence" or column not in results:
                continue
            if comparator in matrix:
                tested.append((column, model, compare_seeds(
                    seeds_of[model][column], seeds_of[comparator][column], primary, names=(model, comparator),
                    paired=paired_gap, confidence=CONFIDENCE, pooled=(results[column], matrix[comparator]))))
            if matrix:
                best = max(matrix, key=lambda m: matrix[m].metrics[primary])
                c = compare_seeds(seeds_of[model][column], seeds_of[best][column], primary, names=(model, best),
                                  paired=paired_gap, confidence=CONFIDENCE, pooled=(results[column], matrix[best]))
                descriptive.append({"dataset": dataset, "condition": column, "model": model, "against": best,
                                    "descriptive": True, "gap": c.difference, "ci_low": c.ci_low,
                                    "ci_high": c.ci_high, "users": int(results[column].n_rows),
                                    "below_min_users": column in low})
    if tested:
        adjusted = holm([c.p_value for _, _, c in tested])
        for (column, model, c), p in zip(tested, adjusted):
            gap_rows.append({"dataset": dataset, "condition": column, "model": model, "against": comparator,
                             "descriptive": False, "gap": c.difference, "ci_low": c.ci_low, "ci_high": c.ci_high,
                             "p_adjusted": p, "users": int(c.n_users), "below_min_users": column in low})
        lines += ["", f"### Gap: sequential − {comparator}", "",
                  (f"The study's question: test {primary} of each sequential model minus {comparator}'s, at every "
                  "level, with the seed-aware t-test (users and seeds"
                  + (", the seeds paired by subsample" if paired_gap else "") + f"); Holm across all {len(tested)} "
                  f"gaps of this sweep on {dataset}. Interval: that test's {CONFIDENCE:.0%} t interval."), "",
                  _table(["condition", "model", "users", "gap", f"{CONFIDENCE:.0%} interval", "adjusted p",
                          "significant"],
                         [[g["condition"], g["model"], f"{g['users']:,}", f"{g['gap']:+.4f}",
                           f"[{g['ci_low']:+.4f}, {g['ci_high']:+.4f}]", f"{g['p_adjusted']:.4g}",
                           verdict(g["p_adjusted"], g["condition"])] for g in gap_rows])]
    elif averaged and comparator not in averaged:
        lines += ["", f"_No gap against {comparator}: it has no finished runs in this sweep on {dataset}._"]
    if descriptive:
        lines += ["", "### Gap: sequential − best non-sequential (descriptive)", "",
                  (f"Test {primary}, with the seed-aware {CONFIDENCE:.0%} t interval. The non-sequential model is the "
                  "best at each level, chosen on test, so this gap carries no test: it shows whether any "
                  f"non-sequential model, not only {comparator}, closes the gap."), "",
                  _table(["condition", "model", "best non-sequential", "users", "gap", f"{CONFIDENCE:.0%} interval"],
                         [[g["condition"], g["model"], g["against"],
                           f"{g['users']:,}" + (" *descriptive*" if g["below_min_users"] else ""),
                           f"{g['gap']:+.4f}", f"[{g['ci_low']:+.4f}, {g['ci_high']:+.4f}]"] for g in descriptive])]
    analysis_lines, analysis_rows, floor_rows = _analysis_section(
        protocol, work_dir, sweep, dataset, columns, averaged, transform.changes_targets, comparator)
    lines += analysis_lines
    gap_rows += descriptive
    return "\n".join(lines), csv_rows, gap_rows, analysis_rows, floor_rows


def _analysis_section(protocol: Protocol, work_dir: Path, sweep: str, dataset: str, columns: list[str],
                      averaged: dict[str, dict[str, EvaluationResult]], strip_targets: bool, comparator: str):
    """The baselines, floor and sequence-signal controls at every condition, from the analysis step.

    A random sweep's condition is analysed once per subsample; it is shown only once every subsample has been,
    or marked partial, so a floor never averages over fewer subsamples than the models do (final review C2).
    """
    primary = protocol.primary_metric
    found = condition_results(protocol, work_dir, sweep, dataset)
    subsamples = len(data_seeds(protocol, work_dir, sweep, dataset))
    if not found:
        return ["", "### Baselines and sequence signal", "", "_Not analysed yet: run `seqrec-eval analyse`._"], [], []
    markov = next((n for n in protocol.baselines if protocol.baseline(n).kind == "markov"), None)
    scorers = [n for n in protocol.baselines] + list(CONTROLS)
    header = ["condition"] + list(protocol.baselines) + ["floor"] + list(CONTROLS)
    if markov:
        header += ["markov, tied end", "markov, real gap"]
    table, records, floor_rows, partial = [], [], [], []
    for column in columns:
        expected = 1 if column == REFERENCE else subsamples
        means, short = {}, {}
        for name in scorers:
            runs = found.get(name, {}).get(column)
            if runs and len(runs) == expected:
                means[name] = mean_over_subsamples([r for r, _ in runs], strip_targets=strip_targets)
            elif runs:
                short[name] = len(runs)
        if short:
            partial.append(f"{column} ({min(short.values())} of {expected} subsamples)")
        baselines = {n: means[n] for n in protocol.baselines if n in means}
        # the floor needs every baseline: the strongest of some is not the floor
        floor = floor_of(baselines, primary, expected=protocol.baselines)

        def cell(name: str) -> str:
            if name in means:
                return f"{means[name].metrics[primary]:.4f}"
            return f"partial {short[name]}/{expected}" if name in short else "—"

        row = [column] + [cell(n) for n in protocol.baselines]
        row.append("—" if floor is None else f"{floor[1].metrics[primary]:.4f} ({floor[0]})")
        row += [cell(c) for c in CONTROLS]
        if markov:
            slices = []
            for tied_end in (True, False):
                values = [float(np.asarray(r.per_user[primary])[t == tied_end].mean())
                          for r, t in found.get(markov, {}).get(column, [])
                          if t is not None and (t == tied_end).sum() >= MIN_SLICE_USERS]
                slices.append(f"{np.mean(values):.4f}" if values else "—")
            row += slices
        table.append(row)
        for name, result in means.items():
            records.append({"dataset": dataset, "condition": column, "scorer": name, **result.metrics})
        if floor is not None:
            # on the gap plot's scale: the floor minus the comparator the gap is taken against
            matrix = {m: r[column] for m, r in averaged.items()
                      if protocol.model(m).family == "matrix" and column in r}
            against = matrix.get(comparator) or (max(matrix.values(), key=lambda r: r.metrics[primary])
                                                 if matrix else None)
            if against is not None:
                floor_rows.append({"dataset": dataset, "condition": column, "floor": floor[0],
                                   "gap": floor[1].metrics[primary] - against.metrics[primary]})
    lines = ["", "### Baselines and sequence signal", "",
             (f"Test {primary} at every condition, on the same users as the models. Each baseline keeps the setting "
             "it was selected with on the full data; the floor is the strongest of them. *markov_shuffled* and "
             "*markov_backwards* are Markov fitted on training histories with order, or direction, removed. The "
             "last two columns split Markov's users by whether their history's last two events tie in time "
             f"(not shown for a slice of fewer than {MIN_SLICE_USERS} users).")
             + (f" Not every subsample analysed yet, so left out: {', '.join(partial)}; run `seqrec-eval analyse "
                f"--dataset {dataset} --sweep {sweep}`." if partial else ""), "",
             _table(header, table)]
    return lines, records, floor_rows


def _csv(rows: list[dict[str, Any]], leading: tuple[str, ...]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in leading,
                    leading.index(key) if key in leading else 0, key))
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def level_order(protocol: Protocol, sweep: str) -> list[str]:
    """The sweep's conditions in plotting order: along the knee's axis if it has one, the full data last."""
    ablation = protocol.ablation(sweep)
    transform = check_ablation(ablation)
    labels = [level_label(level) for level in ablation.levels]
    if transform.position is not None:
        labels = [level_label(level) for level in sorted(ablation.levels, key=transform.position)]
    return labels + [REFERENCE]


def build_ablation_report(protocol: Protocol, work_dir: Path, sweep: str, datasets: list[str],
                          models: list[str], comparator: str = "elsa") -> dict[str, Any]:
    """Markdown; CSVs of per-seed metrics, of the gap (and floor), and of the analysis; the gap plot or ``None``.

    ``comparator`` is the non-sequential model the study's gap is taken against, fixed in advance.
    """
    ablation = protocol.ablation(sweep)
    transform = check_ablation(ablation)
    datasets = [d for d in datasets if d in ablation.datasets]
    models = [m for m in models if m in ablation.models]
    if scope_of(ablation) == "inference":
        how = ("Scope **inference**: training data is untouched and only the histories given at recommendation "
               "time are transformed, so every condition rescores the stage-1 final models")
    else:
        how = ("Scope **all**: training data and inference histories are both transformed, and every model "
               "refits its stage-1 configuration at each level")
    options = f", options {ablation.options}" if ablation.options else ""
    seeds = ("one subsample per seed" if transform.stochastic(ablation.options) else "the same data for every seed")
    header = (f"# Ablation: {sweep}\n\nTransform `{transform.name}` (version {transform.version}), levels "
              f"{list(ablation.levels)}{options}. {how}, under seeds {list(ablation.seeds)}"
              f"{'' if ablation.seeds == protocol.seeds else f' of stage 1’s {list(protocol.seeds)}'} (and any a "
              f"dataset lists as added), with {seeds}. "
              f"**{REFERENCE}** is the stage-1 final model itself, rescored on the same users. "
              f"Primary metric {protocol.primary_metric}.\n")
    if transform.changes_targets:
        header += ("\nThe catalogue shrinks with the level and the targets of removed items go with it, so levels "
                   "score different targets from the full data; the comparisons pair the same users.\n")
    notes = _transform_notes(protocol, transform.name, models)
    sections, metric_rows, gap_rows, analysis_rows, floor_rows = [], [], [], [], []
    for dataset in datasets:
        text, rows, gaps, analysed, floors = dataset_ablation(protocol, work_dir, sweep, dataset, models,
                                                              comparator)
        sections.append(text)
        metric_rows += rows
        gap_rows += gaps
        analysis_rows += analysed
        floor_rows += floors
    candidates = [m for m in models if protocol.model(m).family == "sequence"]
    # the plot shows the tested gap against the comparator; without one, the descriptive gap
    primary_gaps = [g for g in gap_rows if not g["descriptive"]]
    figure = gap_figure(primary_gaps or gap_rows, sweep=sweep, order=level_order(protocol, sweep),
                        candidates=candidates, metric=protocol.primary_metric, floor_rows=floor_rows,
                        against=comparator if primary_gaps else "best non-sequential")
    if figure is not None:
        header += f"\n![Gap per level](ablation-{sweep}-gap.png)\n"
    markdown = header + ("\n" + notes + "\n" if notes else "") + "\n" + "\n\n".join(sections) + "\n"
    return {
        "markdown": markdown,
        "metrics": _csv(metric_rows, ("dataset", "model", "condition", "seed", "source_trial")),
        "gap": _csv(gap_rows + [{**f, "model": "floor"} for f in floor_rows], ("dataset", "condition", "model")),
        "analysis": _csv(analysis_rows, ("dataset", "condition", "scorer")),
        "figure": figure,
    }


def _transform_notes(protocol: Protocol, transform: str, models: list[str]) -> str:
    if transform != "history_length":
        return ""
    capped = []
    for model in models:
        mp = protocol.model(model)
        if mp.family == "sequence" and ("max_history_length" in mp.space or "max_history_length" in mp.fixed):
            capped.append(model)
    if not capped:
        return ""
    return ("Histories are truncated in the data, for every model alike. The sequential models "
            f"({', '.join(capped)}) also keep their stage-1 `max_history_length`, so at levels at or above it "
            "their input at inference is the same as with full data.")
