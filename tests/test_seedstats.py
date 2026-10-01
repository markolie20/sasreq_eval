"""Comparisons that count training seeds beside the users (review H23)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from compresso_recsys.evaluation import EvaluationResult
from scipy import stats

from seqrec_eval.ablation_report import find_knee
from seqrec_eval.seedstats import compare, noninferiority, variance

METRIC = "ndcg@10"


def _seeds(per_seed: list[np.ndarray]) -> list[EvaluationResult]:
    """One evaluation per seed, all on the same users."""
    ids = np.array([f"u{i}" for i in range(per_seed[0].size)])
    return [EvaluationResult(metrics={METRIC: float(values.mean())}, per_user={METRIC: values}, sample_ids=ids,
                             n_rows=values.size, n_scored_rows=values.size, required_k=10) for values in per_seed]


def _model(rng, n, seed_means, noise=0.15, users=None):
    """Per-seed per-user values: a shared user effect, per-seed noise, each seed shifted to an exact mean."""
    users = rng.normal(0.0, noise, n) if users is None else users
    out = []
    for mean in seed_means:
        values = users + rng.normal(0.0, 0.05, n)
        out.append(values - values.mean() + mean)
    return out


def test_with_identical_seeds_it_is_the_users_only_test():
    rng = np.random.default_rng(0)
    users = rng.normal(0.0, 0.15, 5_000)
    a = _seeds([users + 0.004] * 3)  # a deterministic model: three identical seeds
    b = _seeds([users + rng.normal(0.0, 0.05, 5_000)] * 3)
    result = compare(a, b, METRIC, names=("a", "b"))
    d = a[0].per_user[METRIC] - b[0].per_user[METRIC]
    assert result.se_seeds == 0.0 and result.df > 1e3
    assert result.se == pytest.approx(d.std(ddof=1) / math.sqrt(d.size))
    # no seed term: the paired t-test over users, n - 1 degrees of freedom
    assert result.df == pytest.approx(d.size - 1)
    assert result.p_value == pytest.approx(2 * stats.t.sf(abs(d.mean()) / result.se, d.size - 1), rel=1e-6)


def test_a_difference_resting_on_one_lucky_seed_is_not_claimed():
    # the design's example: A 0.100 / 0.112 / 0.094, B 0.099 / 0.100 / 0.101
    rng = np.random.default_rng(1)
    users = rng.normal(0.1, 0.15, 20_000)
    a = _seeds(_model(rng, 20_000, [0.100, 0.112, 0.094], users=users))
    b = _seeds(_model(rng, 20_000, [0.099, 0.100, 0.101], users=users))
    result = compare(a, b, METRIC, names=("a", "b"))
    assert result.difference == pytest.approx(0.002, abs=1e-9)
    assert result.se_seeds == pytest.approx(math.sqrt((8.4e-5 + 1e-6) / 3), rel=1e-6)
    assert 1.5 < result.df < 3.5
    assert result.p_value > 0.5
    users_only_se = result.se_users
    assert 2 * stats.norm.sf(0.002 / users_only_se) < 0.2  # over users alone it looks like a result


def test_with_no_true_difference_the_seed_aware_test_keeps_its_level_and_users_only_does_not():
    # both models equally good; each training run shifts a model's score by N(0, 0.004)
    rng = np.random.default_rng(2)
    n, reps, seed_sd = 2_000, 400, 0.004
    seeded, users_only = 0, 0
    for _ in range(reps):
        users = rng.normal(0.1, 0.15, n)
        a = _seeds(_model(rng, n, 0.1 + rng.normal(0.0, seed_sd, 3), users=users))
        b = _seeds(_model(rng, n, 0.1 + rng.normal(0.0, seed_sd, 3), users=users))
        result = compare(a, b, METRIC, names=("a", "b"))
        seeded += result.p_value <= 0.05
        users_only += 2 * stats.norm.sf(abs(result.difference) / result.se_users) <= 0.05
    assert seeded / reps <= 0.08          # near its nominal 5% (Welch–Satterthwaite with k = 3 is approximate)
    assert users_only / reps >= 0.25      # counting users only, training noise reads as a difference


def test_a_single_seed_or_a_deterministic_baseline_adds_no_seed_term():
    rng = np.random.default_rng(3)
    users = rng.normal(0.1, 0.15, 3_000)
    model = _seeds(_model(rng, 3_000, [0.10, 0.11, 0.09], users=users))
    floor = _seeds([users + rng.normal(0.0, 0.05, 3_000)])  # one evaluation, as the floor is
    result = compare(model, floor, METRIC, names=("model", "floor"))
    assert result.seeds == (3, 1)
    assert result.se_seeds == pytest.approx(math.sqrt(np.var([0.10, 0.11, 0.09], ddof=1) / 3))


def test_the_users_only_p_is_the_paired_t_test_over_users():
    rng = np.random.default_rng(4)
    users = rng.normal(0.1, 0.15, 3_000)
    a = _seeds(_model(rng, 3_000, [0.104, 0.101, 0.108], users=users))
    b = _seeds(_model(rng, 3_000, [0.100, 0.100, 0.100], users=users))
    result = compare(a, b, METRIC, names=("a", "b"))
    pooled_a = np.mean([e.per_user[METRIC] for e in a], axis=0)
    pooled_b = np.mean([e.per_user[METRIC] for e in b], axis=0)
    assert result.p_users == pytest.approx(stats.ttest_rel(pooled_a, pooled_b).pvalue, rel=1e-9)
    assert result.p_value > result.p_users  # counting the seeds can only add uncertainty here


def test_the_t_interval_covers_with_three_disagreeing_seeds():
    # review A2: the two-level bootstrap covered 0 only 82.5% of the time here; the t interval keeps ~95%
    rng = np.random.default_rng(8)
    n, reps, seed_sd, covered = 2_000, 400, 0.004, 0
    for _ in range(reps):
        users = rng.normal(0.1, 0.15, n)
        a = _seeds(_model(rng, n, 0.1 + rng.normal(0.0, seed_sd, 3), users=users))
        b = _seeds(_model(rng, n, 0.1 + rng.normal(0.0, seed_sd, 3), users=users))
        result = compare(a, b, METRIC, names=("a", "b"))
        covered += result.ci_low <= 0.0 <= result.ci_high
    assert covered / reps >= 0.92


def test_paired_seeds_cancel_what_the_two_sides_share():
    # review A1: a level identical to the full data (an inference level above max_history_length) loses
    # nothing, seed for seed; counted as unpaired, the seeds' own spread made it fail "within delta"
    users = np.random.default_rng(9).normal(0.04, 0.1, 3_000)
    means = [0.037, 0.040, 0.043]
    d = np.zeros(users.size)
    unpaired = variance(d, means, means)
    paired = variance(d, means, means, paired=True)
    assert unpaired.seeds > 0 and noninferiority(0.0, unpaired, margin=0.004, alpha=0.05)[0] > 0.05
    assert paired.seeds == 0.0 and noninferiority(0.0, paired, margin=0.004, alpha=0.05)[0] < 0.05
    with pytest.raises(ValueError, match="as many on each side"):
        variance(d, means, means[:2], paired=True)


def test_paired_seeds_keep_their_level_when_the_pairs_are_correlated():
    # each seed shifts both sides alike (the same model rescored), plus a little of its own: no true difference
    rng = np.random.default_rng(10)
    n, reps, rejected = 2_000, 400, 0
    for _ in range(reps):
        users = rng.normal(0.1, 0.15, n)
        shared = rng.normal(0.0, 0.01, 3)
        a = _seeds(_model(rng, n, 0.1 + shared + rng.normal(0.0, 0.001, 3), users=users))
        b = _seeds(_model(rng, n, 0.1 + shared + rng.normal(0.0, 0.001, 3), users=users))
        rejected += compare(a, b, METRIC, names=("a", "b"), paired=True).p_value <= 0.05
    assert rejected / reps <= 0.08


def test_results_scored_against_different_targets_are_refused():
    rng = np.random.default_rng(11)
    a, b = _seeds([rng.normal(0.1, 0.1, 100)]), _seeds([rng.normal(0.1, 0.1, 100)])
    from dataclasses import replace
    a, b = [replace(a[0], target_fingerprint="x")], [replace(b[0], target_fingerprint="y")]
    with pytest.raises(ValueError, match="different targets"):
        compare(a, b, METRIC, names=("a", "b"))


def test_an_unchanged_level_passes_the_knee_when_its_seeds_are_paired():
    users = np.random.default_rng(12).normal(0.04, 0.1, 3_000)
    per_user = {"full": users, "50": users.copy(), "10": users - 0.02}
    position = {"full": math.inf, "50": 50.0, "10": 10.0}
    seed_means = {"full": [0.037, 0.040, 0.043], "50": [0.037, 0.040, 0.043], "10": [0.017, 0.020, 0.023]}
    unpaired = find_knee(per_user, position, margin_fraction=0.10, alpha=0.05, rng=None, seed_means=seed_means)
    paired = find_knee(per_user, position, margin_fraction=0.10, alpha=0.05, rng=None, seed_means=seed_means,
                       paired=True)
    assert (unpaired.knee, unpaired.stopped_at) == ("full", "50")  # the bug: an identical level failed
    assert (paired.knee, paired.stopped_at) == ("50", "10")


def test_the_seed_aware_non_inferiority_test_and_its_power():
    rng = np.random.default_rng(6)
    d = rng.normal(0.0, 0.1, 20_000)
    steady = variance(d, [0.1, 0.1, 0.1], [0.1, 0.1, 0.1])
    shaky = variance(d, [0.08, 0.10, 0.12], [0.1, 0.1, 0.1])
    p_steady, power_steady = noninferiority(0.0, steady, margin=0.01, alpha=0.05)
    p_shaky, power_shaky = noninferiority(0.0, shaky, margin=0.01, alpha=0.05)
    assert p_steady < 0.001 and power_steady > 0.99   # no loss, users only: shown
    assert p_shaky > 0.05 and power_shaky < 0.5       # the seeds' spread makes it undecidable


def test_a_knee_level_whose_seeds_disagree_does_not_pass():
    rng = np.random.default_rng(7)
    users = rng.normal(0.1, 0.05, 20_000)
    per_user = {"full": users + rng.normal(0, 0.02, 20_000), "50": users + rng.normal(0, 0.02, 20_000),
                "10": users + rng.normal(0, 0.02, 20_000)}
    position = {"full": math.inf, "50": 50.0, "10": 10.0}
    steady = {"full": [0.1] * 3, "50": [0.1] * 3, "10": [0.1] * 3}
    shaky = {**steady, "50": [0.07, 0.10, 0.13]}
    rng_for = lambda label: np.random.default_rng(0)  # noqa: E731
    assert find_knee(per_user, position, margin_fraction=0.10, alpha=0.05, rng=rng_for,
                     seed_means=steady).knee == "10"
    knee = find_knee(per_user, position, margin_fraction=0.10, alpha=0.05, rng=rng_for, seed_means=shaky)
    assert (knee.knee, knee.stopped_at) == ("full", "50")
