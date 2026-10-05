"""``scripts/local-run.sh``: the quick protocol it writes and the commands it runs, checked without running them."""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path

import pytest

from seqrec_eval import cli
from seqrec_eval.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "local-run.sh"


def _dry(work: Path, *args: str, **env: str) -> list[str]:
    """The seqrec-eval commands the script would run, one per line."""
    done = subprocess.run([str(SCRIPT), *args], env={**os.environ, "DRY": "1", "WORK": str(work), **env},
                          capture_output=True, text=True, check=True)
    return [line.removeprefix("== seqrec-eval ") for line in done.stderr.splitlines() if line.startswith("== ")]


def test_the_quick_protocol_shortens_only_trials_seeds_and_epochs(tmp_path):
    _dry(tmp_path, "ml20m", "gru")
    quick, local = load_protocol(tmp_path / "protocol.toml"), load_protocol(ROOT / "protocol.local.toml")
    assert quick.seeds == (0,) and quick.trials_per_model == 2
    expected = copy.deepcopy(local.raw)
    expected["protocol"].update(seeds=[0], trials_per_model=2)
    trained = [name for name, model in local.models.items() if "epochs" in model.space]
    assert sorted(trained) == ["elsa", "gru", "sasrec"]
    for name in trained:
        expected["models"][name]["space"]["epochs"] = {"choice": [1]}
    own = [name for name, sweep in local.raw["ablations"].items() if "seeds" in sweep]
    assert own and all(local.ablation(name).seeds == (0,) for name in own)  # the retraining sweeps, at seed 0
    assert quick.raw == expected  # nothing else differs
    for name, model in quick.models.items():
        assert model.trials == (2 if model.space else 1)
    # the split is the one a real run would prepare
    assert quick.dataset_fingerprint("ml20m") == local.dataset_fingerprint("ml20m")
    # other seeds: stage 1 runs under all of them, a sweep with seeds of its own under the first
    _dry(tmp_path, "ml20m", "gru", SEEDS="1, 2")
    other = load_protocol(tmp_path / "protocol.toml")
    assert other.seeds == (1, 2)
    for name in local.ablations:
        assert other.ablation(name).seeds == ((1,) if name in own else (1, 2))


def test_a_full_run_uses_the_local_protocol_unchanged(tmp_path):
    _dry(tmp_path, "ml20m", "gru", FULL="1")
    assert (tmp_path / "protocol.toml").read_text() == (ROOT / "protocol.local.toml").read_text()


def test_the_run_covers_every_step_for_the_chosen_dataset_and_models(tmp_path):
    steps = _dry(tmp_path, "ml20m", "elsa", "gru")
    assert [step.split()[0] for step in steps] == ["plan", "prepare", "analyse", "search", "final", "latency",
                                                   "report", "repeat-strata", "status"]
    assert "analyse --dataset ml20m --sweep none" in steps  # the full data only, not every sweep's conditions
    for step in steps[3:]:
        assert "--dataset ml20m --model elsa gru" in step
    with_sweep = _dry(tmp_path, "ml20m", "gru", SWEEP="density", DEVICE="cuda")
    assert [step.split()[0] for step in with_sweep][-4:] == ["ablate", "analyse", "ablation-report", "status"]
    assert "ablate --dataset ml20m --model gru --sweep density --device cuda" in with_sweep


def test_it_refuses_without_a_dataset_and_a_model(tmp_path):
    done = subprocess.run([str(SCRIPT), "ml20m"], env={**os.environ, "DRY": "1", "WORK": str(tmp_path)},
                          capture_output=True, text=True)
    assert done.returncode == 2 and "scripts/local-run.sh DATASET MODEL" in done.stdout


def test_sweep_none_selects_no_sweep_and_only_for_sweeps():
    assert cli._select(["none"], {"history": 1, "density": 2}, "sweep") == []
    assert cli._select(None, {"history": 1, "density": 2}, "sweep") == ["history", "density"]
    with pytest.raises(SystemExit, match="unknown dataset"):
        cli._select(["none"], {"ml20m": 1}, "dataset")
