"""scripts/project-time.py: the run-time estimate from timing runs (review-plan/plans/week-budget.md, B7)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from seqrec_eval.protocol import load_protocol
from seqrec_eval.runner import plan_trials
from test_smoke import PROTOCOL as STAGE1

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("project_time", ROOT / "scripts" / "project-time.py")
project_time = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(project_time)

_DATASET = STAGE1.split("[datasets.synth]")[1].split("[models.popularity]")[0]
PROTOCOL = STAGE1.replace("[models.popularity]", f"[datasets.other]{_DATASET}[models.popularity]") + """
[ablations.history]
transform = "history_length"
levels = [2, 5]
seeds = [0]

[ablations.history_inference]
transform = "history_length"
scope = "inference"
levels = [2, 5, 10]
"""
#: a cost law the log-linear fit can recover exactly: seconds per epoch by model
LAW = {"elsa": lambda p: 0.1 * p["latent_dim"],
       "gru": lambda p: 0.05 * p["hidden_dim"] * p["max_history_length"],
       "sasrec": lambda p: 0.2 * p["n_negatives"] ** 0.5 * p["max_history_length"]}
TIMED = {"elsa": [{"latent_dim": 4}, {"latent_dim": 8}],
         "gru": [{"hidden_dim": h, "max_history_length": n, "embedding_dim": 8} for h, n in ((8, 10), (16, 10), (8, 20))],
         "sasrec": [{"n_negatives": k, "max_history_length": n} for k, n in ((1, 10), (4, 10), (1, 20))]}
EVAL, RATIO = 1.0, 2.0


def _timing(work: Path, dataset: str, protocol, scale: float = 1.0) -> None:
    """Timed trials of 3 epochs under LAW (times `scale`), and one final per model at RATIO x a trial's fit."""
    for model, configs in TIMED.items():
        base = plan_trials(protocol, "synth", model)[0].params
        for index, sizes in enumerate(configs):
            params = {**base, **sizes, "epochs": 3}
            _write(work / "runs" / dataset / model / "fp" / f"trial-{index:03d}", model, "trial", params,
                   3 * LAW[model](params) * scale)
        params = {**base, **configs[0], "epochs": 3}
        _write(work / "runs" / dataset / model / "fp" / "final-seed0", model, "final", params,
               RATIO * 3 * LAW[model](params) * scale)


def _write(directory: Path, model: str, kind: str, params: dict, fit: float) -> None:
    directory.mkdir(parents=True)
    (directory / "spec.json").write_text(json.dumps({"model": model, "kind": kind, "params": params}))
    (directory / "done.json").write_text(json.dumps({"status": "done", "fit_seconds": fit, "eval_seconds": EVAL}))


def test_the_estimate_follows_the_planned_trials_the_seeds_and_the_sweeps(tmp_path):
    (tmp_path / "protocol.toml").write_text(PROTOCOL)
    protocol = load_protocol(tmp_path / "protocol.toml")
    _timing(tmp_path / "synth-work", "synth", protocol)
    _timing(tmp_path / "other-work", "other", protocol, scale=3.0)  # a dataset three times as costly
    timing = {"synth": project_time.read_timing(tmp_path / "synth-work", "synth"),
              "other": project_time.read_timing(tmp_path / "other-work", "other")}
    result = project_time.project(protocol, timing, factors={}, refit_share=0.5)
    assert result["skipped"] == [] and max(result["fit_error"].values()) < 1e-9
    rows = {(row["dataset"], row["model"]): row for row in result["rows"]}
    assert set(rows) == {(d, m) for d in ("synth", "other") for m in LAW}  # EASE, popularity: not on the GPU
    for (dataset, model), row in rows.items():
        scale = 3.0 if dataset == "other" else 1.0
        assert row["factor"] == pytest.approx(scale)
        fits = [spec.params["epochs"] * LAW[model](spec.params) * scale for spec in plan_trials(protocol, dataset, model)]
        mean_final = np.mean(fits) * RATIO
        assert row["search"] == pytest.approx(sum(fits) + len(fits) * EVAL * scale)
        assert row["finals"] == pytest.approx(2 * (mean_final + EVAL * scale))  # two seeds
        assert row["n_refits"] == 2  # history: 2 levels x seed 0
        assert row["refits"] == pytest.approx(2 * (mean_final * 0.5 + EVAL * scale))
        assert row["n_rescorings"] == 3 * 2 + 1 + 2  # the inference sweep, and each sweep's reference per seed
        assert row["worst_extra"] == pytest.approx((2 + 2 * 0.5) * (max(fits) * RATIO - mean_final))


def test_a_dataset_without_timing_needs_a_factor(tmp_path, capsys):
    (tmp_path / "protocol.toml").write_text(PROTOCOL)
    protocol = load_protocol(tmp_path / "protocol.toml")
    _timing(tmp_path / "work", "synth", protocol)
    timing = {"synth": project_time.read_timing(tmp_path / "work", "synth")}
    left_out = project_time.project(protocol, timing, factors={}, refit_share=0.6)
    assert {row["dataset"] for row in left_out["rows"]} == {"synth"}
    assert left_out["skipped"] == [f"other/{model}: no timing run and no --factor" for model in LAW]
    given = project_time.project(protocol, timing, factors={"other": 2.0}, refit_share=0.6)
    other = {row["model"]: row for row in given["rows"] if row["dataset"] == "other"}
    synth = {row["model"]: row for row in given["rows"] if row["dataset"] == "synth"}
    for model in LAW:
        assert other[model]["source"] == "factor given"
        assert other[model]["search"] == pytest.approx(2.0 * synth[model]["search"])
    assert project_time.main(["--protocol", str(tmp_path / "protocol.toml"), "--timing", f"synth={tmp_path / 'work'}",
                              "--factor", "other=2", "--speedup", "2"]) == 0
    out = capsys.readouterr().out
    assert "reference dataset synth" in out and "GPU total:" in out and "at 2x" in out
    with pytest.raises(SystemExit, match="DATASET=VALUE"):
        project_time.main(["--protocol", str(tmp_path / "protocol.toml"), "--timing", "synth"])


def test_a_timed_configuration_is_priced_at_its_measured_time_not_the_fit():
    # the fit is only for configurations not timed: a measured one keeps its own time, off the law or not
    trials = [({"batch_size": b, "max_history_length": n}, 0.01 * n * 256 / b, 1.0)
              for b, n in ((64, 50), (128, 50), (256, 100), (64, 200))]
    trials.append(({"batch_size": 128, "max_history_length": 400}, 999.0, 1.0))  # far off the law
    cost = project_time.EpochCost(trials)
    assert cost({"batch_size": 128, "max_history_length": 400}) == (999.0, True)
    seconds, timed = cost({"batch_size": 256, "max_history_length": 400})
    assert not timed and seconds != 999.0
