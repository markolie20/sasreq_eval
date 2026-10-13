"""Epoch extensions (DECISIONS §42): a selected configuration still improving at its last epoch is trained again at
multiples of its epochs, scored on validation like any trial, and the best on validation is the selection."""

from __future__ import annotations

import json
import shutil

import pytest

from seqrec_eval import cli
from seqrec_eval.protocol import ProtocolError, load_protocol
from seqrec_eval.runner import (
    execute,
    plan_extensions,
    plan_finals,
    plan_trials,
    stale_final_seeds,
    summarize_trials,
)
from seqrec_eval.splits import final_split, load_split
from test_smoke import DEVICE, PROTOCOL, workspace  # noqa: F401  (the shared synthetic workspace)

FALLING = [{"epoch": 1, "loss": 1.0}, {"epoch": 2, "loss": 0.5}]
FLAT = [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.5}]


def _protocol(tmp_path, extension: str = "[2, 3]"):
    text = PROTOCOL.replace("refit = true\n", f"refit = true\nepoch_extension = {extension}\n", 1)
    assert text != PROTOCOL
    (tmp_path / "protocol.toml").write_text(text)
    return load_protocol(tmp_path / "protocol.toml")


def _record(directory, **fields) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "done.json").write_text(json.dumps({"status": "done", **fields}))


def _history(spec, work, history) -> None:
    path = spec.directory(work) / "done.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "history": history}))


@pytest.mark.parametrize("value", ["[1]", "[2, 2]", "[2.5]", '["2"]', "[true]", "2"])
def test_epoch_extension_is_checked_and_moves_no_fingerprint(tmp_path, value):
    protocol = _protocol(tmp_path, "[4, 2]")
    assert protocol.epoch_extension == (2, 4)
    (tmp_path / "plain.toml").write_text(PROTOCOL)
    plain = load_protocol(tmp_path / "plain.toml")
    assert plain.epoch_extension == ()
    for model in plain.models:
        assert protocol.run_fingerprint("synth", model) == plain.run_fingerprint("synth", model)
    with pytest.raises(ProtocolError, match="epoch_extension must be a list of distinct integers of at least 2"):
        _protocol(tmp_path, value)


def test_extensions_wait_for_a_complete_search_and_need_a_selection_still_improving(tmp_path):
    protocol = _protocol(tmp_path)
    work = tmp_path / "work"
    first, second = plan_trials(protocol, "synth", "gru")
    _record(first.directory(work), val={"ndcg@5": 0.3}, history=FALLING)
    assert plan_extensions(protocol, work, "synth", "gru") == []  # the search is not finished
    (second.directory(work)).mkdir(parents=True)
    (second.directory(work) / "failed.json").write_text(json.dumps({"error": "RuntimeError: out of memory"}))
    assert plan_extensions(protocol, work, "synth", "gru") == []  # a failure is open
    (second.directory(work) / "failed.json").unlink()
    _record(second.directory(work), val={"ndcg@5": 0.2}, history=FALLING)
    extensions = plan_extensions(protocol, work, "synth", "gru")
    assert [(s.name, s.params["epochs"], s.epochs_factor) for s in extensions] == [
        ("trial-000-x2", 2 * first.params["epochs"], 2), ("trial-000-x3", 3 * first.params["epochs"], 3)]
    assert all({k: v for k, v in s.params.items() if k != "epochs"}
               == {k: v for k, v in first.params.items() if k != "epochs"} for s in extensions)
    # the finals wait for them, and a tie keeps the shorter run
    with pytest.raises(RuntimeError, match="2/4 trials finished"):
        plan_finals(protocol, work, "synth", "gru")
    _record(extensions[0].directory(work), val={"ndcg@5": 0.3})
    _record(extensions[1].directory(work), val={"ndcg@5": 0.25})
    summary = summarize_trials(protocol, work, "synth", "gru")
    assert (summary.done, summary.planned, summary.best.name) == (4, 4, "trial-000")
    assert summary.extension_values == {2: 0.3, 3: 0.25}
    finals = plan_finals(protocol, work, "synth", "gru")
    assert {(f.source_trial, f.epochs_factor, f.params["epochs"]) for f in finals} == {(0, 1, first.params["epochs"])}
    # a selection whose loss had flattened, or a model without epochs, is not extended
    _history(first, work, FLAT)
    assert plan_extensions(protocol, work, "synth", "gru") == []
    for spec in plan_trials(protocol, "synth", "popularity"):
        _record(spec.directory(work), val={"ndcg@5": 0.1}, history=FALLING)
    assert plan_extensions(protocol, work, "synth", "popularity") == []
    # without the protocol key nothing changes
    (tmp_path / "plain.toml").write_text(PROTOCOL)
    _history(first, work, FALLING)
    assert plan_extensions(load_protocol(tmp_path / "plain.toml"), work, "synth", "gru") == []


def test_a_failed_extension_blocks_selection_until_accepted(tmp_path):
    from seqrec_eval.runner import accepted_failures_path

    protocol = _protocol(tmp_path, "[2]")
    work = tmp_path / "work"
    first, second = plan_trials(protocol, "synth", "gru")
    _record(first.directory(work), val={"ndcg@5": 0.3}, history=FALLING)
    second.directory(work).mkdir(parents=True)
    (second.directory(work) / "failed.json").write_text(json.dumps({"error": "RuntimeError: out of memory"}))
    assert plan_extensions(protocol, work, "synth", "gru") == []  # the failure is open
    plan_finals(protocol, work, "synth", "gru", accept_failed=True)  # accepted: the search is complete
    (extension,) = plan_extensions(protocol, work, "synth", "gru")
    extension.directory(work).mkdir(parents=True)
    (extension.directory(work) / "failed.json").write_text(json.dumps({"error": "RuntimeError: out of memory"}))
    with pytest.raises(RuntimeError, match="trial 0x2: RuntimeError"):
        plan_finals(protocol, work, "synth", "gru")
    finals = plan_finals(protocol, work, "synth", "gru", accept_failed=True)
    assert {f.epochs_factor for f in finals} == {1}
    # both failures recorded: a trial's number beside an extension's name
    assert json.loads(accepted_failures_path(protocol, work, "synth", "gru").read_text())["trials"] == ["0x2", 1]


def test_a_selection_still_improving_is_extended_and_its_finals_redone(workspace, tmp_path):
    root, _ = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    protocol = _protocol(tmp_path)
    common = ["--protocol", str(tmp_path / "protocol.toml"), "--work-dir", str(work)]
    base = summarize_trials(protocol, work, "synth", "gru").base_best
    _history(base, work, FALLING)
    _history(summarize_trials(protocol, work, "synth", "sasrec").best, work, FLAT)
    old_finals = {f.seed: json.loads((f.directory(work) / "spec.json").read_text())
                  for f in plan_finals(protocol, work, "synth", "sasrec")}

    assert cli.main(common + ["final", "--model", "gru", "--device", DEVICE]) == 3  # waits for the extensions
    assert cli.main(common + ["search", "--model", "gru", "sasrec", "popularity", "ease", "--device", DEVICE]) == 0
    extensions = summarize_trials(protocol, work, "synth", "gru").extensions
    assert [s.name for s in extensions] == [f"trial-{base.index:03d}-x2", f"trial-{base.index:03d}-x3"]
    for spec in extensions:
        record = json.loads((spec.directory(work) / "done.json").read_text())
        made = json.loads((spec.directory(work) / "spec.json").read_text())
        assert made["params"] == {**base.params, "epochs": base.params["epochs"] * spec.epochs_factor}
        assert record["trained_on"] == "train" and "val" in record and "test" not in record  # never test
    for model in ("sasrec", "popularity", "ease"):  # flat, or no epochs: nothing more
        assert summarize_trials(protocol, work, "synth", model).extensions == []
        assert not list((work / "runs" / "synth" / model).glob("*/trial-*-x*"))

    # the best on validation is the selection; make the 3x run win, as if it had
    longest = extensions[1].directory(work) / "done.json"
    longest.write_text(json.dumps({**json.loads(longest.read_text()), "val": {"ndcg@5": 1.0}}))
    summary = summarize_trials(protocol, work, "synth", "gru")
    assert (summary.best.name, summary.best.selection) == (extensions[1].name, f"#{base.index} ×3")
    seeds = sorted(protocol.seeds)
    assert stale_final_seeds(protocol, work, "synth", "gru") == seeds  # made at the base epochs: left out

    assert cli.main(common + ["final", "--model", "gru", "sasrec", "--device", DEVICE]) == 0
    runs = next((work / "runs" / "synth" / "gru").glob("*/"))
    for seed in seeds:
        (moved,) = runs.glob(f"final-seed{seed}.superseded-*")
        assert (moved / "done.json").exists() and not (moved / "model.zip").exists()  # records kept, model not
        made = json.loads((runs / f"final-seed{seed}" / "spec.json").read_text())
        assert (made["source_trial"], made["epochs_factor"], made["params"]["epochs"]) == (
            base.index, 3, 3 * base.params["epochs"])
        assert (runs / f"final-seed{seed}" / "model.zip").exists()
    assert stale_final_seeds(protocol, work, "synth", "gru") == []
    assert {f.seed: json.loads((f.directory(work) / "spec.json").read_text())
            for f in plan_finals(protocol, work, "synth", "sasrec")} == old_finals  # untouched

    assert cli.main(common + ["report"]) == 0
    report = (work / "reports" / "stage1.md").read_text()
    assert f"| gru | 4/4 | #{base.index} ×3 | 1.0000 |" in report
    assert (f"the selected trial #{base.index} was still improving at its last epoch, so it was also trained at 2× "
            f"and 3× its epochs (DECISIONS §42)") in report and f"#{base.index} ×3 is selected" in report

    # a final made with another trial is still refused rather than moved aside, even one whose configuration differs
    # from the selection only in its epochs
    spec = plan_finals(protocol, work, "synth", "gru")[0]
    path = spec.directory(work) / "spec.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "source_trial": 1 - base.index, "epochs_factor": 1,
                                "params": base.params}))
    messages = []
    split = final_split(protocol, load_split(work, "synth"))
    assert execute(spec, split, protocol, work, device="cpu", log=messages.append) == "stale-selection"
    assert f"trial {base.index} ×3 is selected now" in messages[0]


def test_only_a_selection_falling_more_than_the_threshold_is_extended(tmp_path):
    # DECISIONS §43: the budget extends only the clearly unconverged selections, not every one the report flags
    protocol = _protocol(tmp_path)
    work = tmp_path / "work"
    first, second = plan_trials(protocol, "synth", "gru")
    _record(first.directory(work), val={"ndcg@5": 0.3}, history=[{"epoch": 1, "loss": 1.0}, {"epoch": 2, "loss": 0.97}])
    _record(second.directory(work), val={"ndcg@5": 0.2})
    assert len(plan_extensions(protocol, work, "synth", "gru")) == 2  # 3% > the report's 1% flag: extended
    strict = _protocol(tmp_path, "[2, 3]\nepoch_extension_min_fall = 0.05")
    assert strict.epoch_extension_min_fall == 0.05
    assert plan_extensions(strict, work, "synth", "gru") == []  # 3% < 5%: not
    _history(first, work, FALLING)  # 50%
    assert len(plan_extensions(strict, work, "synth", "gru")) == 2
    for model in strict.models:
        assert strict.run_fingerprint("synth", model) == protocol.run_fingerprint("synth", model)
    for bad in ("0", "1", "-0.1", "true", '"5%"'):
        with pytest.raises(ProtocolError, match="epoch_extension_min_fall must be a share between 0 and 1"):
            _protocol(tmp_path, f"[2]\nepoch_extension_min_fall = {bad}")
