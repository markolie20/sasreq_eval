"""What a model loses by being one window older, and how fast the catalogue goes stale (DECISIONS §53).

The finals are refitted on everything before the test window. Served for real, a model is retrained every so often
and is older than that when it answers. Two measurements say what that costs, from what the suite already has:

* **The catalogue.** Of the test users' next items, the share first seen in the validation window: items a model
  fitted one window earlier has never seen and cannot recommend. Beside it, the share first seen in the test window,
  which no model can recommend (the users the metrics leave out). Exact, from the split; counted by ``analyse``.
* **The model.** The stale run (:func:`~seqrec_eval.runner.plan_stale`) is the selection fitted on the training
  window alone, under the first seed, and scored on test. It is set against the first seed's final, on the final's
  users: a user whose next item the stale model has never seen counts 0 for it, which is part of the cost. One seed
  on each side, so the test is the paired t-test over users, Holm-corrected across a dataset's models.

The window is the dataset's ``temporal_period_hours``: 339 days on MovieLens-20M and Amazon, 30 days on the music
data, two days on OTTO. The event times that would split the test window by days since training are not kept in the
split (only ``prepare`` reads them, from the raw data), so the window is the unit.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from compresso_recsys.evaluation import EvaluationResult

from .evaluate import target_key
from .protocol import Protocol
from .results import evaluation_exists, load_evaluation, read_json
from .runner import load_final_evaluations, plan_stale
from .seedstats import compare, holm
from .splits import Split

#: the record's format; a change counts again
AGE_VERSION = 1


def next_item_age(split: Split, tested: Split) -> dict[str, Any]:
    """Of the test users with a next item, how many a model fitted on the training window alone could recommend
    (``split``'s catalogue), how many only the refitted one can (``tested``'s), and how many neither."""
    targets = tested.data[target_key("test", "next")].tocsr()
    n_window, n_refit = len(split.data["train_item_ids"]), len(tested.data["train_item_ids"])
    rows = np.repeat(np.arange(targets.shape[0], dtype=np.int64), np.diff(targets.indptr))
    has = np.diff(targets.indptr) > 0
    known_window = np.bincount(rows, weights=targets.indices < n_window, minlength=targets.shape[0]) > 0
    known_refit = np.bincount(rows, weights=targets.indices < n_refit, minlength=targets.shape[0]) > 0
    return {"version": AGE_VERSION, "users_with_next_item": int(has.sum()),
            "known_one_window_older": int((has & known_window).sum()),
            "known_only_refitted": int((has & known_refit & ~known_window).sum()),
            "known_to_neither": int((has & ~known_refit).sum()),
            "catalogue_one_window_older": n_window, "catalogue_refitted": n_refit,
            "window_hours": float(split.info.get("build_parameters", {}).get("temporal_period_hours") or 0.0)}


def _expand(stale: EvaluationResult, final: EvaluationResult) -> EvaluationResult:
    """``stale``'s per-user values on ``final``'s users, 0 for those it could not score: a user whose next item the
    stale model has never seen. Target fingerprints are set aside, since the two scored different user sets."""
    ids = np.asarray(final.sample_ids).astype(str)
    own = {user: i for i, user in enumerate(np.asarray(stale.sample_ids).astype(str))}
    outside = sorted(set(own) - set(ids))
    if outside:
        raise ValueError(f"the stale run scored {len(outside)} users the final did not (e.g. {outside[0]}): its "
                         "catalogue is the smaller one, so it cannot have more users")
    where = np.array([own.get(user, -1) for user in ids], dtype=np.int64)
    scored = where >= 0
    per_user = {}
    for name, values in stale.per_user.items():
        expanded = np.zeros(ids.size)
        expanded[scored] = np.asarray(values, dtype=np.float64)[where[scored]]
        per_user[name] = expanded
    return replace(stale, per_user=per_user, sample_ids=ids, n_rows=int(ids.size), n_scored_rows=int(ids.size),
                   metrics={name: float(v.mean()) for name, v in per_user.items()}, target_fingerprint=None)


def stale_comparison(protocol: Protocol, work_dir: Path, dataset: str, model: str, view: str = "test"):
    """(stale on the final's users, the first seed's final, comparison) for ``view`` ("test" or "test_new"), or
    ``None`` until both exist."""
    try:
        specs = plan_stale(protocol, work_dir, dataset, model)
    except RuntimeError:
        return None
    if not specs or not evaluation_exists(specs[0].directory(work_dir) / view):
        return None
    finals = [e for spec, e in load_final_evaluations(protocol, work_dir, dataset, model, stem=view)
              if spec.seed == specs[0].seed]
    if not finals:
        return None
    final = replace(finals[0], target_fingerprint=None)
    stale = _expand(load_evaluation(specs[0].directory(work_dir) / view), final)
    return stale, final, compare([stale], [final], protocol.primary_metric, names=("stale", "final"))


def freshness_section(protocol: Protocol, work_dir: Path, datasets: list[str], models: list[str],
                      age: dict[str, dict[str, Any] | None]) -> str:
    """The stage-1 report's section: the catalogue's age per dataset, then every model one window older.
    ``age``: :func:`next_item_age` per dataset, as ``analyse`` recorded it (``None`` where it has not)."""
    from .report import ALPHA, _table

    primary = protocol.primary_metric
    rows = []
    for dataset in datasets:
        found = age.get(dataset)
        if not found:
            continue
        users = found["users_with_next_item"]
        share = (lambda n: f"{n / users:.1%}") if users else (lambda n: "—")
        rows.append([dataset, f"{found['window_hours'] / 24:g} days", f"{users:,}",
                     share(found["known_only_refitted"]), share(found["known_to_neither"])])
    model_rows, any_model = [], False
    for dataset in datasets:
        for view, label in (("test", "next item"), ("test_new", "new items")):
            found = [(model, stale_comparison(protocol, work_dir, dataset, model, view)) for model in models
                     if protocol.model(model).scored_on(dataset)]
            found = [(model, result) for model, result in found if result is not None]
            if not found:
                continue
            any_model = True
            adjusted = holm([c.p_value for _, (_, _, c) in found])
            for (model, (stale, final, c)), p in zip(found, adjusted):
                before = final.metrics[primary]
                model_rows.append([dataset, label, model, f"{before:.4f}", f"{stale.metrics[primary]:.4f}",
                                   f"{c.difference / before:+.1%}" if before else "—",
                                   f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}]", f"{p:.4g}",
                                   "yes" if p <= ALPHA else "no", f"{c.n_users:,}"])
    if not rows and not any_model:
        return ""
    lines = ["## Freshness: one window older (diagnostics)", "",
             ("A model is retrained every so often, so it answers older than the finals, which are refitted on "
              "everything before the test window (DECISIONS §53). The window is the dataset's own: 339 days on "
              "MovieLens-20M and Amazon, 30 on the music data, two on OTTO.")]
    if rows:
        lines += ["", ("Of the test users with a next item, the share whose next item a model fitted one window "
                       "earlier has never seen (first seen in the validation window), and the share no model has "
                       "seen (first seen in the test window, left out of every metric):"), "",
                  _table(["dataset", "window", "users with a next item", "unseen one window older",
                          "unseen by any model"], rows)]
    if model_rows:
        lines += ["", (f"Test {primary} of the first seed's final and of the same selection fitted one window "
                       "earlier (the stale run), on the final's users: a user whose next item the older model has "
                       "never seen counts 0 for it. One seed each, so the test is the paired t-test over users, Holm "
                       "across a dataset's models per view."), "",
                  _table(["dataset", "view", "model", "final", "one window older", "change",
                          "95% interval", "adjusted p", "significant", "users"], model_rows)]
    return "\n".join(lines)


def age_path(protocol: Protocol, work_dir: Path, dataset: str) -> Path:
    from .analysis import analysis_root

    return analysis_root(protocol, work_dir, dataset) / f"next-item-age-{protocol.evaluation_key(dataset)[:12]}.json"


def read_age(protocol: Protocol, work_dir: Path, dataset: str) -> dict[str, Any] | None:
    path = age_path(protocol, work_dir, dataset)
    if not path.exists():
        return None
    record = read_json(path)
    return record if record.get("version") == AGE_VERSION else None
