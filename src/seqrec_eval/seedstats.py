"""Comparisons that count the training seeds as a source of uncertainty, beside the users (review H23).

Every model is trained under k seeds. The estimate is the difference of the per-user values averaged over each
side's seeds; its uncertainty has two sources: which users happened to be in the test set, and which training
runs happened to be drawn.

With ``d`` the per-user differences of the seed-averaged values (n users), and ``M[m, s]`` side m's mean score in
seed s (each seed scored on the same users, so their spread is training noise):

    D        = mean(d)
    V_users  = var(d) / n
    V_seeds  = var_s(M[candidate, s]) / k_candidate  +  var_s(M[reference, s]) / k_reference      (unpaired)
             = var_s(M[candidate, s] − M[reference, s]) / k                                         (paired)
    SE       = sqrt(V_users + V_seeds)
    df       = (V_users + V_seeds)² / ( V_users² / (n − 1) + Σ (each seed term)² / (its k − 1) )

``t = D / SE`` is read against Student's t with ``df``: Welch's construction (Welch 1947; Satterthwaite 1946)
with a third variance term, which is what a mixed model with a random seed effect approximates. When the seeds
agree it is the paired t-test over users; when they do not, df falls towards k − 1 and a difference must be
clearly larger than the seed spread to count.

**Paired or not.** In stage 1, seed s of one model has nothing to do with seed s of another: the seed terms add.
In an ablation, seed s of a level and seed s of the full data are the same trained model (inference scope) or
the same seed (scope all), and two models at a level of a random sweep share subsample s: the seeds are paired,
and the seed term is the spread of the per-seed differences, with k − 1 degrees of freedom. Treating paired
seeds as independent adds both sides' spread where most of it cancels, so a level identical to the full data
could fail "within δ" (final review A1). The paired term is valid whether or not the pairs are correlated.

**Users only.** Beside it, ``p_users`` is the same test without the seed term: the paired t-test over users
(standard in IR, e.g. Smucker et al., CIKM 2007), so the effect of counting the seeds is visible.

**Interval.** ``D ± t(1 − α/2, df) · SE``, the test's own. A two-level bootstrap (users and seeds resampled)
was used for stage 1 until 2026-10-01; with k = 3 it is far too narrow -- resampling 3 seeds shrinks their
spread by (k − 1)/k, and percentiles act like z where t has 2 df -- 2.6 times narrower than the t interval in
the design example, while p was 0.74 (final review A2).

The one-sided form for the knee, "the loss is smaller than δ", is ``t = (D + δ) / SE`` against the upper tail,
and its power for a level that loses nothing is ``P(T_df > t(1 − α, df) − δ / SE)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from compresso_recsys.evaluation import EvaluationResult
from scipy import stats

#: Past this many degrees of freedom Student's t is the normal distribution to the precision that matters here.
_NORMAL_DF = 1e6


def holm(p_values: list[float]) -> list[float]:
    """Holm's step-down adjustment of a family of p-values."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        return []
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate((p.size - np.arange(p.size)) * p[order]))
    out = np.empty_like(p)
    out[order] = adjusted
    return out.tolist()


def _tail(value: float, df: float) -> float:
    """P(T_df > value)."""
    return float(stats.norm.sf(value) if df > _NORMAL_DF else stats.t.sf(value, df))


def _quantile(probability: float, df: float) -> float:
    """The value T_df exceeds with ``probability``."""
    return float(stats.norm.isf(probability) if df > _NORMAL_DF else stats.t.isf(probability, df))


@dataclass(frozen=True)
class Variance:
    """The two components of a difference's uncertainty, and the degrees of freedom of their sum."""

    users: float
    seeds: float
    df: float

    @property
    def se(self) -> float:
        return math.sqrt(self.users + self.seeds)


def _seed_term(means) -> tuple[float, int]:
    means = np.asarray(means, dtype=np.float64)
    k = means.size
    return (float(np.var(means, ddof=1)) / k if k > 1 else 0.0), k


def variance(differences: np.ndarray, candidate_means, reference_means, *, paired: bool = False) -> Variance:
    """Users' and seeds' share of the variance of mean(differences), with Welch–Satterthwaite df.

    ``paired``: the two sides' seed means are aligned seed by seed (see the module docstring).
    """
    differences = np.asarray(differences, dtype=np.float64)
    n = differences.size
    users = float(np.var(differences, ddof=1)) / n if n > 1 else 0.0
    if paired:
        candidate_means, reference_means = np.asarray(candidate_means), np.asarray(reference_means)
        if candidate_means.shape != reference_means.shape:
            raise ValueError(f"paired seeds need as many on each side, got {candidate_means.size} and "
                             f"{reference_means.size}")
        parts = [_seed_term(candidate_means - reference_means)]
    else:
        parts = [_seed_term(candidate_means), _seed_term(reference_means)]
    seeds = sum(part for part, _ in parts)
    denominator = (users**2 / (n - 1) if n > 1 else 0.0) + sum(part**2 / (k - 1) for part, k in parts if k > 1)
    df = (users + seeds) ** 2 / denominator if denominator > 0 else math.inf
    return Variance(users=users, seeds=seeds, df=df)


def users_only(differences: np.ndarray) -> Variance:
    """The paired t-test's variance over users alone, n − 1 df."""
    n = np.asarray(differences).size
    return Variance(users=float(np.var(differences, ddof=1)) / n if n > 1 else 0.0, seeds=0.0,
                    df=float(n - 1) if n > 1 else math.inf)


def two_sided_p(difference: float, spread: Variance) -> float:
    if spread.se == 0.0:
        return 1.0 if difference == 0.0 else 0.0
    return min(1.0, 2.0 * _tail(abs(difference) / spread.se, spread.df))


def noninferiority(difference: float, spread: Variance, margin: float, alpha: float) -> tuple[float, float]:
    """(p, power) of the one-sided test that the loss is smaller than ``margin``, counting the seeds."""
    if spread.se == 0.0:
        return (0.0 if difference > -margin else 1.0), (1.0 if margin > 0 else 0.0)
    p = _tail((difference + margin) / spread.se, spread.df)
    power = _tail(_quantile(alpha, spread.df) - margin / spread.se, spread.df)
    return p, power


@dataclass(frozen=True)
class SeedComparison:
    candidate: str
    reference: str
    difference: float
    ci_low: float
    ci_high: float
    #: the seed-aware test's two-sided p, which significance is read from
    p_value: float
    #: the paired t-test over users alone, for comparison
    p_users: float
    se: float
    se_users: float
    se_seeds: float
    df: float
    seeds: tuple[int, int]
    n_users: int
    paired: bool


def _values(evaluations: list[EvaluationResult], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """(sample ids, seeds x users matrix) of one model, every seed on the same users in the same order."""
    ids = np.asarray(evaluations[0].sample_ids).astype(str)
    rows = []
    for evaluation in evaluations:
        if not np.array_equal(np.asarray(evaluation.sample_ids).astype(str), ids):
            raise ValueError("the seeds of a model scored different users; give their pooled values")
        rows.append(np.asarray(evaluation.per_user[metric], dtype=np.float64))
    return ids, np.vstack(rows)


def _same_targets(a: EvaluationResult, b: EvaluationResult, names: tuple[str, str]) -> None:
    """Refuse two results that scored the same users on different relevant items (as the library's
    ``compare_models`` did); a result whose fingerprint was set aside on purpose (``None``) passes."""
    if a.target_fingerprint is not None and b.target_fingerprint is not None \
            and a.target_fingerprint != b.target_fingerprint:
        raise ValueError(f"{names[0]} and {names[1]} were scored against different targets")


def compare(candidate: list[EvaluationResult], reference: list[EvaluationResult], metric: str, *,
            names: tuple[str, str], paired: bool = False, confidence: float = 0.95,
            pooled: tuple[EvaluationResult, EvaluationResult] | None = None) -> SeedComparison:
    """``candidate`` minus ``reference`` on ``metric``, each given as its per-seed evaluations.

    ``paired``: the i-th evaluation of each side belongs to the same seed (or subsample), see the module
    docstring. ``pooled`` (the per-user values averaged over each side's seeds, already aligned) is for seeds
    that scored different users, as a random catalogue's do.
    """
    if pooled is None:
        candidate_ids, candidate_values = _values(candidate, metric)
        reference_ids, reference_values = _values(reference, metric)
        if not np.array_equal(candidate_ids, reference_ids):
            raise ValueError(f"{names[0]} and {names[1]} were scored on different users")
        _same_targets(candidate[0], reference[0], names)
        differences = candidate_values.mean(axis=0) - reference_values.mean(axis=0)
    else:
        a, b = pooled
        if not np.array_equal(np.asarray(a.sample_ids).astype(str), np.asarray(b.sample_ids).astype(str)):
            raise ValueError(f"{names[0]} and {names[1]} were scored on different users")
        _same_targets(a, b, names)
        differences = np.asarray(a.per_user[metric], dtype=np.float64) - np.asarray(b.per_user[metric], np.float64)
    candidate_means = [float(np.mean(e.per_user[metric])) for e in candidate]
    reference_means = [float(np.mean(e.per_user[metric])) for e in reference]
    spread = variance(differences, candidate_means, reference_means, paired=paired)
    estimate = float(differences.mean())
    half = _quantile((1.0 - confidence) / 2.0, spread.df) * spread.se
    return SeedComparison(candidate=names[0], reference=names[1], difference=estimate, ci_low=estimate - half,
                          ci_high=estimate + half, p_value=two_sided_p(estimate, spread),
                          p_users=two_sided_p(estimate, users_only(differences)), se=spread.se,
                          se_users=math.sqrt(spread.users), se_seeds=math.sqrt(spread.seeds), df=spread.df,
                          seeds=(len(candidate), len(reference)), n_users=int(differences.size), paired=paired)
