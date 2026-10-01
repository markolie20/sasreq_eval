"""Seeds added after the protocol's have run: only the first seed is in any fingerprint, so an added seed joins
the runs already made (``final --add-seeds``, ``ablate --add-seeds``) instead of starting them over.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import numpy as np
import pytest
from compresso_recsys import builder

from seqrec_eval import ablations, cli
from seqrec_eval.ablations import (REFERENCE, ablation_fingerprint, ablation_root, check_added_seeds,
                                   condition_fingerprint, fixed_test_rows, plan_ablation, sweep_seeds,
                                   sweep_seeds_path)
from seqrec_eval.analysis import condition_results
from seqrec_eval.protocol import load_protocol
from seqrec_eval.runner import (added_seeds_path, execute, final_seeds, load_final_evaluations, plan_finals,
                                plan_trials)
from seqrec_eval.splits import final_split, load_split
from test_smoke import DEVICE, PERIOD_DAYS, PROTOCOL as STAGE1, Synthetic

SWEEPS = """
[ablations.history]
transform = "history_length"
levels = [2]

[ablations.density]
transform = "density"
levels = [0.5]

[ablations.shuffle]
transform = "shuffle"
levels = [3]
"""
PROTOCOL = STAGE1 + SWEEPS
MODELS = ("popularity", "elsa", "gru", "sasrec")  # EASE is skipped in stage 1


def _stamps(folder):
    return {path: path.stat().st_mtime_ns for path in folder.rglob("done.json")}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("seeds")
    (root / "protocol.toml").write_text(PROTOCOL)
    patch = pytest.MonkeyPatch()
    patch.setitem(builder.DATASETS, "synthetic", builder.DatasetSpec(
        Synthetic, str(root / "unused.zip"), seed=0, val_users=5, test_users=5,
        min_value_to_keep=4.0, min_entity_text_words=0, temporal_period_hours=PERIOD_DAYS * 24,
    ))
    work = root / "work"
    common = ["--protocol", str(root / "protocol.toml"), "--work-dir", str(work)]
    for step in (["prepare", "--data-dir", str(root / "data"), "--quiet"], ["search", "--device", DEVICE],
                 ["final", "--device", DEVICE], ["ablate", "--device", DEVICE], ["analyse"]):
        assert cli.main(common + step) == 0
    state = {"before": _stamps(work), "rows": {s: np.load(ablation_root(load_protocol(root / "protocol.toml"), work,
                                                                          s, "synth") / "test_rows.npy")
                                               for s in ("history", "density", "shuffle")}}
    # seed 2 for one model first: the report must say the others have not run it yet
    assert cli.main(common + ["final", "--device", DEVICE, "--model", "gru", "--add-seeds", "2"]) == 0
    assert cli.main(common + ["report"]) == 0
    state["partial_report"] = (work / "reports" / "stage1.md").read_text()
    assert cli.main(common + ["final", "--device", DEVICE]) == 0
    state["record"] = (added_seeds_path(work, "synth")).read_text()
    # seed 2 for two of the three sweeps; shuffle keeps the protocol's seeds while stage 1 has three
    assert cli.main(common + ["ablate", "--device", DEVICE, "--sweep", "history", "density", "--add-seeds", "2"]) == 0
    assert cli.main(common + ["analyse", "--sweep", "density"]) == 0
    assert cli.main(common + ["report"]) == 0
    assert cli.main(common + ["ablation-report"]) == 0
    yield root, common, load_protocol(root / "protocol.toml"), state
    patch.undo()


def _edited(tmp_path, old: str, new: str):
    path = tmp_path / "protocol.toml"
    text = PROTOCOL.replace(old, new)
    assert text != PROTOCOL
    path.write_text(text)
    return load_protocol(path)


def test_only_the_first_seed_enters_any_fingerprint(tmp_path):
    (tmp_path / "base.toml").write_text(PROTOCOL)
    protocol = load_protocol(tmp_path / "base.toml")
    more = _edited(tmp_path, "seeds = [0, 1]", "seeds = [0, 1, 5, 6]")
    for model in MODELS:
        assert more.run_fingerprint("synth", model) == protocol.run_fingerprint("synth", model)
    assert more.baseline_fingerprint("synth", "markov") == protocol.baseline_fingerprint("synth", "markov")
    assert condition_fingerprint(more, "density", "synth") == condition_fingerprint(protocol, "density", "synth")
    assert ablation_fingerprint(more, "density", "synth", "gru") == ablation_fingerprint(protocol, "density", "synth",
                                                                                       "gru")
    # the first seed seeds every trial, so it decides the selection: changing it starts over
    first = _edited(tmp_path, "seeds = [0, 1]", "seeds = [1, 0]")
    assert first.run_fingerprint("synth", "gru") != protocol.run_fingerprint("synth", "gru")
    assert {spec.seed for spec in plan_trials(first, "synth", "gru")} == {1}


def test_added_final_seeds_join_the_runs_already_made(workspace):
    root, _, protocol, state = workspace
    work = root / "work"
    now = _stamps(work)
    assert {path: now[path] for path in state["before"]} == state["before"]  # nothing made before was redone
    assert final_seeds(protocol, work, "synth") == (0, 1, 2)
    record = json.loads(state["record"])
    assert record["seeds"] == [2] and record["added"][0]["seeds"] == [2]
    for model in MODELS:
        assert [spec.seed for spec in plan_finals(protocol, work, "synth", model)] == [0, 1, 2]
        assert [spec.seed for spec, _ in load_final_evaluations(protocol, work, "synth", model)] == [0, 1, 2]
    report = (work / "reports" / "stage1.md").read_text()
    assert "Seeds [0, 1, 2]: the protocol's, and [2] added later" in report
    assert "have not run yet" not in report


def _seeds_cell(report: str, model: str) -> str:
    """The "seeds" cell of ``model``'s row in the stage-1 results table (its first row in the report)."""
    row = next(line for line in report.splitlines() if line.startswith(f"| {model} |"))
    return row.split(" | ")[4]


def test_the_report_says_which_models_have_not_run_an_added_seed(workspace):
    _, _, _, state = workspace
    report = state["partial_report"]
    for model in ("popularity", "elsa", "sasrec"):
        assert f"- **{model}**: final seed(s) [2] have not run yet" in report
        assert _seeds_cell(report, model) == "2/3"
    assert "- **gru**: final seed(s) [2]" not in report
    assert _seeds_cell(report, "gru") == "3"


def test_adding_a_seed_again_changes_nothing(workspace):
    root, common, _, state = workspace
    path = added_seeds_path(root / "work", "synth")
    stamp = path.stat().st_mtime_ns
    assert cli.main(common + ["final", "--device", DEVICE, "--add-seeds", "2"]) == 0
    assert path.stat().st_mtime_ns == stamp and path.read_text() == state["record"]


@pytest.mark.parametrize("command, seeds, message", [
    ("final", ["3", "3"], "lists a seed twice"),
    ("final", ["-1"], "non-negative"),
    ("ablate", ["9"], "not stage-1 seeds of synth"),
])
def test_bad_added_seeds_are_refused_before_anything_is_recorded(workspace, command, seeds, message):
    root, common, protocol, _ = workspace
    work = root / "work"
    before = {path: path.read_text() for path in work.rglob("added_seeds.json")}
    with pytest.raises(SystemExit, match=message):
        cli.main(common + [command, "--device", DEVICE, "--add-seeds", *seeds])
    assert {path: path.read_text() for path in work.rglob("added_seeds.json")} == before
    assert final_seeds(protocol, work, "synth") == (0, 1, 2)


def test_a_sweep_takes_added_seeds_for_every_model(workspace):
    root, _, protocol, _ = workspace
    work = root / "work"
    assert sweep_seeds(protocol, work, "history", "synth") == (0, 1, 2)
    assert sweep_seeds(protocol, work, "shuffle", "synth") == (0, 1)
    for model in MODELS:
        history = plan_ablation(protocol, work, "history", "synth", model)
        assert {name: [spec.seed for spec in specs] for name, specs in history.items()} == {
            REFERENCE: [0, 1, 2], "2": [0, 1, 2]}
        # a random sweep adds a subsample per seed, fitted with that seed
        density = plan_ablation(protocol, work, "density", "synth", model)
        assert {name: [spec.seed for spec in specs] for name, specs in density.items()} == {
            REFERENCE: [0, 1, 2], "0.5/seed0": [0], "0.5/seed1": [1], "0.5/seed2": [2]}
        # stage 1 has three seeds, the shuffle sweep still two: its reference is those two
        shuffle = plan_ablation(protocol, work, "shuffle", "synth", model)
        assert {name: [spec.seed for spec in specs] for name, specs in shuffle.items()} == {
            REFERENCE: [0, 1], "3/seed0": [0], "3/seed1": [1]}
        for specs in (*history.values(), *density.values()):
            for spec in specs:
                assert json.loads((spec.directory(work) / "done.json").read_text())["status"] == "done"
    # the added subsample kept every fixed test user, and says it was checked
    for sweep in ("history", "density", "shuffle"):
        root_dir = ablation_root(protocol, work, sweep, "synth")
        assert np.array_equal(np.load(root_dir / "test_rows.npy"), workspace[3]["rows"][sweep])
    assert json.loads((ablation_root(protocol, work, "density", "synth") / "test_rows.json").read_text())[
        "data_seeds"] == [0, 1, 2]
    assert json.loads((ablation_root(protocol, work, "history", "synth") / "test_rows.json").read_text())[
        "data_seeds"] == [None]


def test_the_ablation_report_uses_every_seed_of_the_sweep(workspace):
    root, _, protocol, _ = workspace
    reports = root / "work" / "reports"
    density = (reports / "ablation-density.md").read_text()
    assert "Seeds [0, 1, 2]: the protocol's, and [2] added later (`ablate --add-seeds`)" in density
    assert "### Each level against the full data" in density  # every seed finished, so the condition entered
    metrics = (reports / "ablation-density-metrics.csv").read_text().splitlines()
    assert len(metrics) - 1 == len(MODELS) * 2 * 3  # reference and one level, three seeds each
    shuffle = (reports / "ablation-shuffle.md").read_text()
    assert "added later" not in shuffle
    assert len((reports / "ablation-shuffle-metrics.csv").read_text().splitlines()) - 1 == len(MODELS) * 2 * 2
    # the floor and controls were analysed on the new subsample too
    results = condition_results(protocol, root / "work", "density", "synth")
    assert results and all(len(columns["0.5"]) == 3 for columns in results.values())


def test_a_subsample_that_would_shrink_the_fixed_users_is_refused(workspace, monkeypatch, tmp_path):
    root, _, protocol, _ = workspace
    work = root / "work"
    split = final_split(protocol, load_split(work, "synth"))
    rows = np.load(ablation_root(protocol, work, "density", "synth") / "test_rows.npy")
    real = ablations._eligible_test_rows

    def losing_one(data, targets="test_target_matrix", **options):
        eligible = real(data, targets, **options).copy()
        eligible[rows[0]] = False  # as a transform that could empty a history or remove a target item would
        return eligible

    monkeypatch.setattr(ablations, "_eligible_test_rows", losing_one)
    with pytest.raises(RuntimeError, match=f"leaves 1 of the {rows.size:,} fixed test users ineligible"):
        check_added_seeds(protocol, work, split, "density", [3])
    check_added_seeds(protocol, work, split, "density", [2])  # already checked: nothing to refuse
    check_added_seeds(protocol, work, split, "history", [3])  # deterministic: its conditions have no seed

    # a seed written into the record by hand is caught where the fixed users are read
    copy = tmp_path / "work"
    source = ablation_root(protocol, work, "density", "synth")
    target = ablation_root(protocol, copy, "density", "synth")
    target.mkdir(parents=True)
    for name in ("test_rows.npy", "test_rows.json"):
        shutil.copy(source / name, target / name)
    sweep_seeds_path(copy, "density", "synth").write_text(json.dumps({"seeds": [2, 3], "added": []}))
    with pytest.raises(RuntimeError, match="Remove the seed from"):
        fixed_test_rows(protocol, copy, split, "density")
    assert json.loads((target / "test_rows.json").read_text())["data_seeds"] == [0, 1, 2]
    monkeypatch.setattr(ablations, "_eligible_test_rows", real)
    assert np.array_equal(fixed_test_rows(protocol, copy, split, "density"), rows)
    assert json.loads((target / "test_rows.json").read_text())["data_seeds"] == [0, 1, 2, 3]


def test_a_reference_waits_for_a_stage1_final_that_is_not_made_yet(workspace, tmp_path):
    root, _, protocol, _ = workspace
    work = root / "work"
    split = final_split(protocol, load_split(work, "synth"))
    spec = plan_ablation(protocol, work, "history", "synth", "gru")[REFERENCE][0]
    spec = replace(spec, condition={**spec.condition, "checkpoint": "runs/synth/gru/nowhere/final-seed9/model.zip"})
    condition = ablations.build_reference(protocol, work, split, "history")
    assert execute(spec, condition, protocol, tmp_path, device="cpu", log=lambda _: None) == "waiting-for-stage1"
    assert not (spec.directory(tmp_path) / "failed.json").exists()
