"""The stale run, a model at its serving window, and freshness (DECISIONS §51, §53).

The stale run is the selection fitted on the training window alone under the first seed -- the search trial
again, kept -- and scored on validation and test. Its validation scores at the last n events choose the window a model
with ``serving_windows`` is served at; the finals are then rescored there, on their own users. Scored on test against
the first seed's final, it is the final one window older."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from compresso_recsys.evaluation import EvaluationResult

from seqrec_eval import cli
from seqrec_eval.evaluate import evaluate_phase
from seqrec_eval.freshness import _expand, next_item_age, read_age, stale_comparison
from seqrec_eval.protocol import ProtocolError, load_protocol
from seqrec_eval.results import load_evaluation, save_evaluation
from seqrec_eval.runner import execute, plan_finals, plan_stale, run_root, summarize_trials
from seqrec_eval.splits import final_split, load_split
from seqrec_eval.window import window_choice, windowed
from test_placement import place
from test_smoke import DEVICE, PROTOCOL, workspace  # noqa: F401  (the shared synthetic workspace)

WINDOWS = "[1, 2, 5]"


def _protocol(tmp_path, windows: str = WINDOWS):
    (tmp_path / "protocol.toml").write_text(place(PROTOCOL, "elsa", "serving_windows", windows))
    return load_protocol(tmp_path / "protocol.toml")


@pytest.mark.parametrize("value, message", [
    ("[0]", "distinct positive integers"), ("[2, 2]", "distinct positive integers"), ('["5"]', "positive integers"),
    ("5", "must be a list"),
])
def test_serving_windows_are_checked_and_move_no_fingerprint(tmp_path, value, message):
    protocol = _protocol(tmp_path, "[5, 1]")
    assert protocol.model("elsa").serving_windows == (1, 5)
    plain = load_protocol(_write_plain(tmp_path))
    assert protocol.run_fingerprint("synth", "elsa") == plain.run_fingerprint("synth", "elsa")
    with pytest.raises(ProtocolError, match=message):
        _protocol(tmp_path, value)


def _write_plain(tmp_path):
    (tmp_path / "plain.toml").write_text(PROTOCOL)
    return tmp_path / "plain.toml"


def test_a_window_cuts_the_inputs_but_not_what_was_seen_nor_who_is_scored(workspace):
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    tested = final_split(protocol, load_split(root / "work", "synth", protocol))
    cut = windowed(protocol, tested, 2)
    lengths = cut.data["test_source_sequences"].row_lengths
    original = tested.data["test_source_sequences"]
    assert lengths.max() == 2 and np.array_equal(lengths, np.minimum(original.row_lengths, 2))
    # the last two events, in order
    ends = original.indptr[1:]
    row = int(np.flatnonzero(original.row_lengths >= 2)[0])
    assert cut.data["test_source_sequences"].values[cut.data["test_source_sequences"].indptr[row]:][:2].tolist() == \
        original.values[ends[row] - 2:ends[row]].tolist()
    assert (cut.data["test_seen_matrix"] != tested.data["test_source_matrix"]).nnz == 0
    assert cut.data["x_train_sequences"] is tested.data["x_train_sequences"]  # training untouched
    from seqrec_eval.analysis import fit_baseline

    model = fit_baseline("markov", {}, tested)
    for exclude in (False, True):
        full = evaluate_phase(model, tested, "test", family="sequence", protocol=protocol, exclude_seen=exclude)
        short = evaluate_phase(model, cut, "test", family="sequence", protocol=protocol, exclude_seen=exclude)
        assert np.array_equal(np.asarray(full.sample_ids), np.asarray(short.sample_ids))


def test_the_window_is_chosen_on_validation_and_the_finals_are_scored_there(workspace, tmp_path):
    root, _ = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    protocol = _protocol(tmp_path)
    common = ["--protocol", str(tmp_path / "protocol.toml"), "--work-dir", str(work)]
    assert cli.main(common + ["window", "--device", DEVICE]) == 0
    (stale,) = plan_stale(protocol, work, "synth", "elsa")
    directory = stale.directory(work)
    assert directory.name == "stale-seed0" and (directory / "model.zip").exists()
    record = json.loads((directory / "done.json").read_text())
    assert record["trained_on"] == "train" and "val" in record and "test" in record
    # the stale run is the selected search trial again: the same configuration and seed, the same score
    summary = summarize_trials(protocol, work, "synth", "elsa")
    assert record["spec"]["params"] == summary.best.params and record["spec"]["seed"] == protocol.seeds[0]
    assert record["val"]["ndcg@5"] == pytest.approx(summary.best_value, rel=1e-6)
    choice = window_choice(protocol, work, "synth", "elsa")
    assert set(choice["values"]) == {None, 1, 2, 5}
    best = max(choice["values"].values())
    assert choice["values"][choice["window"]] == best
    root_dir = run_root(work, "synth", "elsa", protocol.run_fingerprint("synth", "elsa"))
    if choice["window"] is not None:
        for seed in protocol.seeds:
            final = root_dir / f"final-seed{seed}"
            assert (final / f"test_last{choice['window']}.json").exists()
            assert (final / f"test_new_last{choice['window']}.json").exists()  # synth has the new-item diagnostic
            ours = load_evaluation(final / f"test_last{choice['window']}")
            assert np.array_equal(np.asarray(ours.sample_ids), np.asarray(load_evaluation(final / "test").sample_ids))
    assert cli.main(common + ["report", "--reference", "elsa"]) == 0
    report = (work / "reports" / "stage1.md").read_text()
    assert "**elsa at its serving window** (DECISIONS §51)" in report
    assert "whole history" in report and "(the search trial itself:" in report
    if choice["window"] is not None:
        assert f"Against elsa at its last {choice['window']} events: every model against **elsa, last" in report
        assert "| elsa, whole history |" in report
    # rerunning scores nothing again
    before = sorted(p.name for p in directory.iterdir())
    assert cli.main(common + ["window", "--device", DEVICE]) == 0
    assert sorted(p.name for p in directory.iterdir()) == before


def _fake(value: float, ids: np.ndarray) -> EvaluationResult:
    values = np.full(ids.size, value)
    return EvaluationResult(metrics={"ndcg@5": value}, per_user={"ndcg@5": values}, sample_ids=ids,
                            n_rows=ids.size, n_scored_rows=ids.size, required_k=5, metadata={},
                            target_fingerprint=None)


def test_a_tie_keeps_the_longer_window_and_the_whole_history_longest(workspace, tmp_path):
    root, _ = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    protocol = _protocol(tmp_path)
    split = load_split(work, "synth", protocol)
    (stale,) = plan_stale(protocol, work, "synth", "elsa")
    assert execute(stale, split, protocol, work, device="cpu", log=lambda _: None) == "done"
    directory = stale.directory(work)
    ids = np.array(["a", "b"])
    whole = load_evaluation(directory / "val").metrics["ndcg@5"]
    for n, value in ((1, whole), (2, whole), (5, whole - 0.01)):
        save_evaluation(_fake(value, ids), directory / f"val_last{n}")
    assert window_choice(protocol, work, "synth", "elsa")["window"] is None  # the whole history wins a tie
    save_evaluation(_fake(whole + 0.01, ids), directory / "val_last1")
    save_evaluation(_fake(whole + 0.01, ids), directory / "val_last2")
    assert window_choice(protocol, work, "synth", "elsa")["window"] == 2  # of the tied windows, the longer
    (directory / "val_last5.json").unlink()
    assert window_choice(protocol, work, "synth", "elsa") is None  # not every window scored yet


def test_the_stale_run_is_the_final_one_window_older(workspace, tmp_path):
    root, _ = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    common = ["--protocol", str(root / "protocol.toml"), "--work-dir", str(work)]
    protocol = load_protocol(root / "protocol.toml")
    assert cli.main(common + ["stale", "--model", "gru", "elsa", "ease", "--device", DEVICE]) == 0
    assert plan_stale(protocol, work, "synth", "ease") == []  # skipped at its finals: nothing to make older
    (spec,) = plan_stale(protocol, work, "synth", "gru")
    assert spec.trains_on_window and spec.source_trial == plan_finals(protocol, work, "synth", "gru")[0].source_trial
    # a stale run refuses the refitted split, as a trial does
    with pytest.raises(ValueError, match="must be given a split trained on train"):
        execute(spec, final_split(protocol, load_split(work, "synth", protocol)), protocol, work, device="cpu")
    stale, final, comparison = stale_comparison(protocol, work, "synth", "gru")
    assert np.array_equal(np.asarray(stale.sample_ids), np.asarray(final.sample_ids))
    assert comparison.seeds == (1, 1) and comparison.se_seeds == 0.0  # one seed each: the test is over users
    assert stale_comparison(protocol, work, "synth", "gru", "test_new") is not None

    # the age of the catalogue, counted by analyse
    split = load_split(work, "synth", protocol)
    tested = final_split(protocol, split)
    age = read_age(protocol, work, "synth")
    assert age == next_item_age(split, tested)
    targets = tested.data["test_next_target_matrix"].tocsr()
    users = int((np.diff(targets.indptr) > 0).sum())
    assert age["users_with_next_item"] == users
    assert age["known_one_window_older"] + age["known_only_refitted"] + age["known_to_neither"] == users
    assert age["catalogue_one_window_older"] < age["catalogue_refitted"]
    assert age["known_only_refitted"] > 0  # the synthetic users who switch to new items do so in validation too
    assert cli.main(common + ["report", "--reference", "elsa"]) == 0
    report = (work / "reports" / "stage1.md").read_text()
    section = report.split("## Freshness: one window older (diagnostics)")[1]
    assert "| synth | next item | gru |" in section and "| synth | new items | elsa |" in section
    assert "| synth | 5 days |" in section


def test_a_user_the_older_model_cannot_score_counts_zero():
    final = _fake(0.5, np.array(["a", "b", "c"]))
    stale = EvaluationResult(metrics={"ndcg@5": 0.4}, per_user={"ndcg@5": np.array([0.2, 0.6])},
                             sample_ids=np.array(["c", "a"]), n_rows=2, n_scored_rows=2, required_k=5, metadata={},
                             target_fingerprint="x")
    expanded = _expand(stale, final)
    assert expanded.sample_ids.tolist() == ["a", "b", "c"]
    assert expanded.per_user["ndcg@5"] == pytest.approx([0.6, 0.0, 0.2])
    assert expanded.metrics["ndcg@5"] == pytest.approx(0.8 / 3) and expanded.target_fingerprint is None
    with pytest.raises(ValueError, match="cannot have more users"):
        _expand(_fake(0.1, np.array(["a", "z"])), final)
