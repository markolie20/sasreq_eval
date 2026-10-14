"""Where a model runs (DECISIONS §45): ``datasets`` keeps a model to the datasets it names, and ``in_ablations =
false`` keeps it out of the sweeps that name no models. Both only place a model, so neither enters a fingerprint:
changing either value moves nothing, and a model kept off a dataset is reported as skipped there, with no record."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from seqrec_eval import cli
from seqrec_eval.ablations import ablation_fingerprint, condition_fingerprint
from seqrec_eval.diversity import diversity_table
from seqrec_eval.latency import latency_table
from seqrec_eval.protocol import ProtocolError, load_protocol
from seqrec_eval.report import build_report, code_builds, status_table
from seqrec_eval.runner import execute, plan_finals, plan_trials, summarize_trials
from seqrec_eval.splits import load_split
from seqrec_eval.strata import build_strata_report
from test_smoke import DEVICE, PROTOCOL, workspace  # noqa: F401  (the shared synthetic workspace)

REPO = Path(__file__).resolve().parents[1]
#: a second dataset, defined but never prepared, so a model can be kept off synth
OTHER = PROTOCOL[PROTOCOL.index("[datasets.synth]"):PROTOCOL.index("[models.")].replace("[datasets.synth]",
                                                                                     "[datasets.other]")


def fingerprints(protocol) -> dict[tuple, str]:
    """Every fingerprint the protocol gives, keyed by what it identifies."""
    out = {}
    for dataset in protocol.datasets:
        out[("dataset", dataset)] = protocol.dataset_fingerprint(dataset)
        for baseline in protocol.baselines:
            out[("baseline", dataset, baseline)] = protocol.baseline_fingerprint(dataset, baseline)
        for model in protocol.models:
            out[("run", dataset, model)] = protocol.run_fingerprint(dataset, model)
    for sweep, ablation in protocol.ablations.items():
        for dataset in ablation.datasets:
            out[("condition", sweep, dataset)] = condition_fingerprint(protocol, sweep, dataset)
            for model in ablation.models:
                out[("ablation", sweep, dataset, model)] = ablation_fingerprint(protocol, sweep, dataset, model)
    return out


def place(text: str, model: str, key: str, value: str) -> str:
    """``text`` with ``key = value`` in ``[models.<model>]``: the line replaced if there is one, else added."""
    header = f"[models.{model}]\n"
    start = text.index(header) + len(header)
    end = text.find("\n[", start)
    end = len(text) if end < 0 else end + 1
    table = text[start:end]
    line = re.compile(rf"^{key} = .*$", re.M)
    table = line.sub(f"{key} = {value}", table) if line.search(table) else f"{key} = {value}\n{table}"
    return text[:start] + table + text[end:]


def _write(tmp_path, text: str, name: str = "protocol.toml"):
    path = tmp_path / name
    path.write_text(text)
    return load_protocol(path)


# ---------------------------------------------------------------------------
# no fingerprint moves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key, value", [
    ("datasets", '["ml20m"]'),
    ("datasets", '["music4all", "yambda", "otto"]'),
    ("datasets", '["ml20m", "amazon", "music4all", "yambda", "otto"]'),
    ("in_ablations", "false"),
    ("in_ablations", "true"),
])
def test_placing_any_model_of_the_real_protocol_moves_no_fingerprint(tmp_path, key, value):
    text = (REPO / "protocol.toml").read_text()
    before = fingerprints(load_protocol(REPO / "protocol.toml"))
    for model in load_protocol(REPO / "protocol.toml").models:
        after = fingerprints(_write(tmp_path, place(text, model, key, value)))
        # a model taken out of the sweeps has no ablation fingerprints; everything that remains is unmoved
        assert {k: v for k, v in after.items() if k in before} == {k: v for k, v in before.items() if k in after}
        assert {k for k in before if k[0] != "ablation"} == {k for k in after if k[0] != "ablation"}
        if key == "in_ablations" and value == "false":
            assert not [k for k in after if k[0] == "ablation" and k[-1] == model]


def test_the_placement_keys_are_the_only_ones_left_out(tmp_path):
    # anything else in a model's section still moves its fingerprint, as before (a control on the test above)
    text = (REPO / "protocol.toml").read_text()
    before = load_protocol(REPO / "protocol.toml").run_fingerprint("ml20m", "gru")
    after = _write(tmp_path, place(text, "gru", "max_items", "10000000"))
    assert after.run_fingerprint("ml20m", "gru") != before


# ---------------------------------------------------------------------------
# the values are checked
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key, value, message", [
    ("datasets", '"synth"', "datasets must be a non-empty list"),
    ("datasets", "[]", "datasets must be a non-empty list"),
    ("datasets", '["synth", "synth"]', "distinct"),
    ("datasets", '["nowhere"]', r"datasets names \['nowhere'\], which the protocol does not define"),
    ("in_ablations", "0", "in_ablations must be true or false"),
])
def test_placement_values_are_checked(tmp_path, key, value, message):
    with pytest.raises(ProtocolError, match=message):
        _write(tmp_path, place(PROTOCOL, "gru", key, value))


SWEEP = """
[ablations.history]
transform = "history_length"
levels = [2]
"""


def test_a_model_out_of_the_sweeps_is_in_none_that_names_no_models(tmp_path):
    protocol = _write(tmp_path, place(PROTOCOL + SWEEP, "gru", "in_ablations", "false"))
    assert "gru" not in protocol.ablation("history").models
    assert set(protocol.ablation("history").models) == set(protocol.models) - {"gru"}


def test_a_sweep_may_not_name_a_model_out_of_the_sweeps(tmp_path):
    text = place(PROTOCOL + SWEEP + 'models = ["gru", "sasrec"]\n', "gru", "in_ablations", "false")
    with pytest.raises(ProtocolError, match=r"names \['gru'\], whose \[models\.\*\] in_ablations is false"):
        _write(tmp_path, text)


# ---------------------------------------------------------------------------
# a model kept off a dataset
# ---------------------------------------------------------------------------

def test_a_model_kept_off_a_dataset_is_skipped_there_and_leaves_no_record(workspace, tmp_path):
    root, _ = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    before = load_protocol(root / "protocol.toml")
    assert cli.main(["--protocol", str(root / "protocol.toml"), "--work-dir", str(work), "diversity", "--model", "gru",
                     "--device", DEVICE]) == 0
    # gru ran on synth in the workspace; kept to "other" now, those runs are no longer gru's on synth
    text = place(PROTOCOL.replace("[models.", OTHER + "[models.", 1), "gru", "datasets", '["other"]')
    protocol = _write(tmp_path, text)
    assert protocol.run_fingerprint("synth", "gru") == before.run_fingerprint("synth", "gru")

    assert plan_trials(protocol, "synth", "gru") == []
    assert plan_finals(protocol, work, "synth", "gru") == []
    summary = summarize_trials(protocol, work, "synth", "gru")
    assert summary.skipped == summary.planned == 2
    assert summary.skip_reason == "not run on synth: [models.gru] datasets = ['other']"
    row = next(line for line in status_table(protocol, work, ["synth"], ["gru"]).splitlines() if "| gru |" in line)
    assert "skipped" in row
    report, csv = build_report(protocol, work, ["synth"], list(protocol.models), "elsa")
    assert "- **gru** skipped: not run on synth" in report
    assert "| gru |" not in report and ",gru," not in csv
    # measured before, and no longer shown
    for table in (latency_table, diversity_table):
        assert "| synth | gru |" in table(before, work, ["synth"], ["gru"])
        assert "| synth | gru |" not in table(protocol, work, ["synth"], ["gru"])
    assert "gru" in build_strata_report(before, work, ["synth"], list(before.models))[0]
    assert "gru" not in build_strata_report(protocol, work, ["synth"], list(protocol.models))[0]
    # nor are its runs counted among the dataset's code builds
    others = [m for m in before.models if m != "gru"]
    assert sum(code_builds(protocol, work, "synth", list(protocol.models)).values()) == \
        sum(code_builds(before, work, "synth", others).values())

    def files():
        return {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in work.rglob("*") if p.is_file()}

    unchanged = files()
    common = ["--protocol", str(tmp_path / "protocol.toml"), "--work-dir", str(work)]
    for step in (["search", "--dataset", "synth", "--model", "gru", "--device", DEVICE],
                 ["final", "--dataset", "synth", "--model", "gru", "--device", DEVICE],
                 ["diversity", "--dataset", "synth", "--model", "gru", "--device", DEVICE, "--force"],
                 ["latency", "--dataset", "synth", "--model", "gru", "--threads", "1"]):
        assert cli.main(common + step) == 0, step
    split = load_split(work, "synth", protocol)
    with pytest.raises(ValueError, match="not run on synth"):
        execute(plan_trials(before, "synth", "gru")[0], split, protocol, work, device="cpu")
    assert files() == unchanged  # nothing was written, nothing rewritten

    # widened again: the same fingerprint, so the runs made before are found where they are
    widened = _write(tmp_path, place(text, "gru", "datasets", '["synth", "other"]'))
    assert plan_trials(widened, "synth", "gru") == plan_trials(before, "synth", "gru")
    assert summarize_trials(widened, work, "synth", "gru").done == 2
    assert len(plan_finals(widened, work, "synth", "gru")) == 2
