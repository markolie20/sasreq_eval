"""Latency-only models (DECISIONS §48): no search, one final at the first seed fitted on a few users and never scored,
timed by `latency` and shown in the latency table alone."""

from __future__ import annotations

import csv
import io
import json
import shutil

import pytest

from seqrec_eval import cli, runner
from seqrec_eval.protocol import ProtocolError, load_protocol
from seqrec_eval.runner import plan_finals, plan_trials, run_root, summarize_trials
from test_smoke import DEVICE, PROTOCOL, workspace  # noqa: F401  (the shared synthetic workspace)

LRUREC = """
[models.lrurec]
family = "sequence"
latency_only = true
[models.lrurec.fixed]
d_model = 8
n_blocks = 1
max_history_length = 20
n_negatives = 5
batch_size = 16
epochs = 1
"""


def _protocol(tmp_path, section: str = LRUREC):
    (tmp_path / "protocol.toml").write_text(PROTOCOL + section)
    return load_protocol(tmp_path / "protocol.toml")


@pytest.mark.parametrize("extra, message", [
    ("[models.lrurec.space]\nlr = { loguniform = [0.001, 0.01] }\n", r"latency_only, so it takes one fixed configuration and no \['space'\]"),
    ("trials = 3\n", r"no \['trials'\]"),
    ("in_ablations = true\n", r"no \['in_ablations'\]"),
])
def test_a_latency_only_model_takes_one_fixed_configuration(tmp_path, extra, message):
    section = LRUREC.replace("latency_only = true\n", "latency_only = true\n" + extra) if "[" not in extra \
        else LRUREC + extra
    with pytest.raises(ProtocolError, match=message):
        _protocol(tmp_path, section)
    with pytest.raises(ProtocolError, match="latency_only must be true or false"):
        _protocol(tmp_path, LRUREC.replace("latency_only = true", 'latency_only = "yes"'))


def test_latency_only_is_part_of_the_fingerprint(tmp_path):
    # a final fitted on a few users for latency is not a model whose accuracy means anything: switching the key off
    # must start fresh runs, not reuse that one as a real final
    latency = _protocol(tmp_path)
    (tmp_path / "scored.toml").write_text(PROTOCOL + LRUREC.replace("latency_only = true\n", ""))
    scored = load_protocol(tmp_path / "scored.toml")
    assert latency.model("lrurec").latency_only and not scored.model("lrurec").latency_only
    assert latency.run_fingerprint("synth", "lrurec") != scored.run_fingerprint("synth", "lrurec")
    assert not latency.model("lrurec").in_ablations and latency.model("lrurec").scored_on("synth") is False
    assert plan_trials(latency, "synth", "lrurec") == []
    (final,) = plan_finals(latency, tmp_path / "work", "synth", "lrurec")
    assert (final.seed, final.source_trial, final.params["epochs"]) == (latency.seeds[0], None, 1)


def test_a_latency_only_model_is_timed_and_shown_in_the_latency_table_alone(workspace, tmp_path, monkeypatch, capsys):
    root, _ = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    protocol = _protocol(tmp_path)
    common = ["--protocol", str(tmp_path / "protocol.toml"), "--work-dir", str(work)]
    monkeypatch.setattr(runner, "LATENCY_ONLY_ROWS", 10)

    assert cli.main(common + ["search", "--model", "lrurec", "--device", DEVICE]) == 0
    runs = run_root(work, "synth", "lrurec", protocol.run_fingerprint("synth", "lrurec"))
    assert not list(runs.glob("trial-*"))  # nothing searched
    assert cli.main(common + ["final", "--model", "lrurec", "--device", DEVICE]) == 0
    assert sorted(p.name for p in runs.iterdir() if p.is_dir()) == [f"final-seed{protocol.seeds[0]}"]
    final = runs / f"final-seed{protocol.seeds[0]}"
    record = json.loads((final / "done.json").read_text())
    assert record["latency_only"] == {"fit_rows": 10} and record["model_saved"]
    assert "test" not in record and "val" not in record and not list(final.glob("test*"))  # never scored
    assert summarize_trials(protocol, work, "synth", "lrurec").best is None

    assert cli.main(common + ["diversity", "--model", "lrurec", "--device", DEVICE]) == 0
    assert not (final / "diversity.json").exists()
    assert cli.main(common + ["latency", "--model", "lrurec", "--threads", "1"]) == 0
    assert json.loads((final / "latency.json").read_text())["overall"]["n"] > 0

    capsys.readouterr()
    assert cli.main(common + ["status"]) == 0
    assert "| synth | lrurec | yes | — | 0 | 0 | 1/1 | latency only |" in capsys.readouterr().out
    assert cli.main(common + ["plan"]) == 0
    assert "latency only: 1 unscored final" in capsys.readouterr().out

    assert cli.main(common + ["report", "--reference", "elsa"]) == 0
    report = (work / "reports" / "stage1.md").read_text()
    lines = [line for line in report.splitlines() if "| lrurec |" in line]
    assert len(lines) == 1 and lines[0].startswith("| synth | lrurec |")  # the latency table's row, nothing else
    assert "**lrurec**: timed for latency only, untrained and never scored" in report
    rows = list(csv.DictReader(io.StringIO((work / "reports" / "final_metrics.csv").read_text())))
    assert rows and not [row for row in rows if row["model"] == "lrurec"]
    # the models with accuracy are reported as before
    assert any(line.startswith("| sasrec |") for line in report.splitlines())
