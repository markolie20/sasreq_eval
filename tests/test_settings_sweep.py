"""A settings sweep (DECISIONS §52): the data as it is, each level the stage-1 selection with some settings replaced,
refitted and compared with the stage-1 final. And the ablation runs built on a final follow it when an epoch extension
moves the selection (DECISIONS §55), as the finals do."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from seqrec_eval import cli
from seqrec_eval.ablations import REFERENCE, ablation_root, plan_ablation
from seqrec_eval.protocol import ProtocolError, load_protocol
from seqrec_eval.runner import plan_finals, summarize_trials
from test_placement import place
from test_smoke import DEVICE, PROTOCOL, workspace  # noqa: F401  (the shared synthetic workspace)

SETTINGS = """
[ablations.parts]
transform = "settings"
models = ["gru"]
levels = ["wide", "long"]
seeds = [0]
[ablations.parts.options]
wide = { hidden_dim = 16 }
long = { max_history_length = 20 }
"""


def _protocol(tmp_path, text: str = PROTOCOL + SETTINGS):
    (tmp_path / "protocol.toml").write_text(text)
    return load_protocol(tmp_path / "protocol.toml")


@pytest.mark.parametrize("change, message", [
    ('models = ["gru"]\n', "must name its models"),
    ("long = { max_history_length = 20 }\n", r"missing \['long'\]"),
    ("wide = { hidden_dim = 16 }\n", r"missing \['wide'\]"),
    ('levels = ["wide", "long"]', 'levels = ["wide", "long", "full"]'),
])
def test_a_settings_sweep_is_checked(tmp_path, change, message):
    from seqrec_eval.ablations import check_ablation

    text = PROTOCOL + (SETTINGS.replace(change, "") if "levels" not in change else
                       SETTINGS.replace(change, message))
    if "levels" in change:
        with pytest.raises(ProtocolError, match="reserved for the reference"):
            check_ablation(_protocol(tmp_path, text).ablation("parts"))
        return
    with pytest.raises(ProtocolError, match=message):
        check_ablation(_protocol(tmp_path, text).ablation("parts"))


def test_a_settings_sweep_may_name_a_model_kept_out_of_the_data_sweeps(tmp_path):
    protocol = _protocol(tmp_path, place(PROTOCOL + SETTINGS, "gru", "in_ablations", "false"))
    assert protocol.ablation("parts").models == ("gru",)
    data_sweep = '\n[ablations.history]\ntransform = "history_length"\nlevels = [2]\nmodels = ["gru"]\n'
    with pytest.raises(ProtocolError, match="in_ablations is false"):
        _protocol(tmp_path, place(PROTOCOL + SETTINGS + data_sweep, "gru", "in_ablations", "false"))


def test_each_level_refits_the_selection_with_its_settings_and_is_compared_with_the_final(workspace, tmp_path):
    from seqrec_eval.ablation_report import build_ablation_report

    root, _ = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    protocol = _protocol(tmp_path)
    common = ["--protocol", str(tmp_path / "protocol.toml"), "--work-dir", str(work)]
    assert cli.main(common + ["ablate", "--sweep", "parts", "--device", DEVICE]) == 0
    assert cli.main(common + ["analyse", "--sweep", "parts"]) == 0  # nothing to analyse: the data is the full data's
    assert not (ablation_root(protocol, work, "parts", "synth") / "analysis").exists()
    selected = plan_finals(protocol, work, "synth", "gru")[0].params
    plans = plan_ablation(protocol, work, "parts", "synth", "gru")
    assert [spec.kind for spec in plans[REFERENCE]] == ["reference"]
    for level, changes in (("wide", {"hidden_dim": 16}), ("long", {"max_history_length": 20})):
        assert any(selected[key] != value for key, value in changes.items())  # a level that changes something
        (spec,) = plans[level]
        assert spec.kind == "final" and spec.seed == 0 and spec.params == {**selected, **changes}
        record = json.loads((spec.directory(work) / "done.json").read_text())
        assert record["spec"]["params"] == {**selected, **changes} and "test" in record
    report = build_ablation_report(protocol, work, "parts", ["synth"], list(protocol.models))["markdown"]
    assert "each level refits the stage-1 selection with the level's settings" in report
    assert "### Each level against the full data" in report
    against = report.split("### Each level against the full data")[1]
    assert "| gru | wide |" in against and "| gru | long |" in against
    assert "No gap against" not in report and "A settings sweep leaves the data as it is" in report
    # the manipulation check: nothing about the data moved
    check = report.split("### Manipulation check")[1].split("### Test")[0]
    rows = [line for line in check.splitlines() if line.startswith(("| wide |", "| long |"))]
    assert len(rows) == 2 and not any("⚠" in row for row in rows) and all("+0.0%" in row for row in rows)
    # a level that sets a setting the model does not have is refused before anything runs
    bad = _protocol(tmp_path, PROTOCOL + SETTINGS.replace("max_history_length = 20", "max_histroy_length = 20"))
    with pytest.raises(ProtocolError, match=r"sets \['max_histroy_length'\], which \[models.gru\] neither fixes"):
        plan_ablation(bad, work, "parts", "synth", "gru")


FALLING = [{"epoch": 1, "loss": 1.0}, {"epoch": 2, "loss": 0.5}]
INFERENCE = """
[ablations.inference]
transform = "history_length"
scope = "inference"
levels = [2]
models = ["gru"]
"""


def test_ablation_runs_follow_a_selection_an_epoch_extension_moved(workspace, tmp_path):
    root, _ = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    text = PROTOCOL.replace("refit = true\n", "refit = true\nepoch_extension = [2, 3]\n", 1) + INFERENCE
    protocol = _protocol(tmp_path, text)
    common = ["--protocol", str(tmp_path / "protocol.toml"), "--work-dir", str(work)]
    assert cli.main(common + ["ablate", "--sweep", "inference", "--device", DEVICE]) == 0
    before = {name: [spec.directory(work) for spec in specs]
              for name, specs in plan_ablation(protocol, work, "inference", "synth", "gru").items()}

    # the selection was still improving: extended, and the 3x run made to win on validation
    base = summarize_trials(protocol, work, "synth", "gru").base_best
    path = base.directory(work) / "done.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "history": FALLING}))
    assert cli.main(common + ["search", "--model", "gru", "--device", DEVICE]) == 0
    longest = summarize_trials(protocol, work, "synth", "gru").extensions[-1].directory(work) / "done.json"
    longest.write_text(json.dumps({**json.loads(longest.read_text()), "val": {"ndcg@5": 1.0}}))
    assert summarize_trials(protocol, work, "synth", "gru").best.epochs_factor == 3
    assert cli.main(common + ["final", "--model", "gru", "--device", DEVICE]) == 0

    # the sweep's runs, made on the finals at the old epochs, are moved aside and redone, not refused
    messages = []
    import seqrec_eval.cli as module

    original = module._log
    module._log = messages.append
    try:
        assert cli.main(common + ["ablate", "--sweep", "inference", "--device", DEVICE]) == 0
    finally:
        module._log = original
    assert any("made at other epochs than the selection" in m for m in messages)
    epochs = 3 * base.params["epochs"]
    for name, directories in before.items():
        for directory in directories:
            assert list(directory.parent.glob(f"{directory.name}.superseded-*")), (name, directory)
            made = json.loads((directory / "spec.json").read_text())
            assert made["params"]["epochs"] == epochs and (directory / "done.json").exists()
    assert np.all([json.loads((d / "done.json").read_text())["status"] == "done"
                   for directories in before.values() for d in directories])
