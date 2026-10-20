"""New items (DECISIONS §50, §54): the baselines are scored on them too, so new items have a floor; every model is
compared with the reference and with that floor on them; and a second view counts every new item a user went on to
have in the test window. Plus novelty and the long-tail share of the lists (§54)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from seqrec_eval import cli
from seqrec_eval.analysis import _baseline_dir, discovery_views, fit_baseline, full_results, score_view
from seqrec_eval.diversity import item_users, list_diversity, tail_items
from seqrec_eval.evaluate import evaluate_phase, new_item_targets, seen_history
from seqrec_eval.protocol import load_protocol
from seqrec_eval.results import load_evaluation
from seqrec_eval.runner import load_final_evaluations
from seqrec_eval.splits import final_split, load_split
from test_smoke import DEVICE, workspace  # noqa: F401  (the shared synthetic workspace)


def test_every_baseline_is_scored_on_both_views_of_new_items(workspace):
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    assert discovery_views(protocol, "synth") == [("new", "test_new"), ("window_new", "test_window_new")]
    found = full_results(protocol, root / "work", "synth")["views"]
    for view in ("test_new", "test_window_new"):
        assert set(found[view]) == set(protocol.baselines)
        record = load_evaluation(_baseline_dir(protocol, root / "work", "synth", "markov") / view)
        assert record.metadata["exclude_seen"] is True and record.n_scored_rows > 0
    # by construction: under exclusion replay has nothing left to replay, so it is popularity by training events
    tested = final_split(protocol, load_split(root / "work", "synth", protocol))
    replay = found["test_new"]["replay"]
    popularity = score_view(protocol, tested, fit_baseline("popularity", {"count": "events"}, tested), "new")
    assert np.array_equal(np.asarray(replay.sample_ids), np.asarray(popularity.sample_ids))
    assert np.allclose(replay.per_user["ndcg@5"], popularity.per_user["ndcg@5"])


def test_later_new_items_are_the_window_less_the_history(workspace):
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    tested = final_split(protocol, load_split(root / "work", "synth", protocol))
    model = fit_baseline("markov", {}, tested)
    result = evaluate_phase(model, tested, "test", family="sequence", protocol=protocol, exclude_seen=True,
                            targets="window_new")
    window = tested.data["test_target_matrix"]
    expected = new_item_targets(window, seen_history(tested.data, "test"))
    has = np.flatnonzero(np.diff(expected.tocsr().indptr) > 0)
    # every user with a new item later in the window, and no one else
    assert np.array_equal(np.sort(np.asarray(result.sample_ids).astype(str)),
                          np.sort(tested.eval_user_ids("test")[has]))
    assert result.metadata["definition"] == "window" and result.metadata["targets"] == "window_new"
    # a user keeps every new item of the window, several where there are several (the synthetic data has none)
    from scipy.sparse import csr_matrix

    window = csr_matrix(np.array([[1, 1, 1, 0], [0, 1, 0, 1]], dtype=np.float32))
    history = csr_matrix(np.array([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    assert new_item_targets(window, history).toarray().tolist() == [[1, 0, 1, 0], [0, 0, 0, 1]]
    with pytest.raises(ValueError, match="window_new"):
        evaluate_phase(model, tested, "test", family="sequence", protocol=protocol, exclude_seen=True,
                       targets="later")


def test_the_report_compares_every_model_on_new_items_with_the_reference_and_the_floor(workspace, tmp_path):
    root, _ = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    common = ["--protocol", str(root / "protocol.toml"), "--work-dir", str(work)]
    # the finals are scored on later new items in the diversity pass, which reloads every final
    assert cli.main(common + ["diversity", "--device", DEVICE]) == 0
    protocol = load_protocol(root / "protocol.toml")
    for model in ("elsa", "gru", "sasrec"):
        assert load_final_evaluations(protocol, work, "synth", model, stem="test_window_new")
    assert cli.main(common + ["report", "--reference", "elsa"]) == 0
    report = (work / "reports" / "stage1.md").read_text()
    for what in ("New items only", "Later new items"):
        assert f"{what}: every model against **elsa**" in report
        assert f"{what}: against the floor, **" in report
        block = report.split(f"{what}: every model against **elsa**")[1].split("\n\n")[1]
        assert {line.split(" | ")[0][2:] for line in block.splitlines()[2:]} == {"popularity", "gru", "sasrec"}
    assert "Every new item the user went on to have in the test window" in report
    # the floor of each view is the strongest baseline scored that way
    found = full_results(protocol, work, "synth")["views"]["test_new"]
    best = max(found, key=lambda name: found[name].metrics["ndcg@5"])
    assert f"New items only: against the floor, **{best}** at test ndcg@5 {found[best].metrics['ndcg@5']:.4f}" in report


def test_a_view_waits_for_every_baseline_before_it_has_a_floor(workspace, tmp_path):
    root, _ = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    protocol = load_protocol(root / "protocol.toml")
    for path in _baseline_dir(protocol, work, "synth", "replay").glob("test_new.*"):
        path.unlink()
    from seqrec_eval.report import build_report

    report, _ = build_report(protocol, work, ["synth"], ["popularity", "elsa", "gru", "sasrec"], "elsa")
    assert "_New items only: no floor yet, every baseline must be scored on it first" in report
    assert "New items only: against the floor" not in report
    assert "New items only: every model against **elsa**" in report  # the reference comparison does not wait


def test_novelty_and_the_long_tail_share_of_a_list():
    users = np.array([50.0, 30.0, 10.0, 6.0, 3.0, 1.0, 0.0])  # 100 users had something
    # the head holds 80% of the interactions: items 0 and 1 (80 of 100)
    assert tail_items(users).tolist() == [False, False, True, True, True, True, True]
    vectors = np.eye(7)
    lists = np.array([[0, 1], [2, 6]])
    found = list_diversity(lists, vectors, [1, 2], users=users, n_users=100)
    information = -np.log2(np.array([50, 30, 10, 1]) / 100)  # an item no user had counts as had by one
    assert found["2"]["novelty"] == pytest.approx(information.mean())
    assert found["2"]["tail_share"] == pytest.approx(0.5)
    assert found["1"]["novelty"] == pytest.approx((information[0] + information[2]) / 2)
    assert found["1"]["tail_share"] == pytest.approx(0.5)
    # out of the catalogue: as novel as an item can be, and in the tail
    out = list_diversity(np.array([[7, 0]]), vectors, [2], users=users, n_users=100)
    assert out["2"]["tail_share"] == pytest.approx(0.5)
    assert out["2"]["novelty"] == pytest.approx((np.log2(100) - np.log2(0.5)) / 2)
    from scipy.sparse import csr_matrix

    matrix = csr_matrix(np.array([[1, 1, 0], [2, 0, 0], [1, 0, 1]], dtype=np.float32))  # a repeat counts once
    assert item_users(matrix).tolist() == [3.0, 1.0, 1.0]


def test_the_diversity_table_shows_novelty_and_the_long_tail(workspace, tmp_path):
    from seqrec_eval.diversity import diversity_table

    root, _ = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    common = ["--protocol", str(root / "protocol.toml"), "--work-dir", str(work)]
    assert cli.main(common + ["diversity", "--model", "gru", "--device", DEVICE]) == 0
    table = diversity_table(load_protocol(root / "protocol.toml"), work, ["synth"], ["gru"])
    assert "| novelty@5 | long-tail share@5 |" in table
    row = next(line for line in table.splitlines() if line.startswith("| synth | gru |"))
    novelty, tail = (float(cell.split(" ± ")[0]) for cell in row.strip(" |").split(" | ")[-2:])
    assert novelty > 0 and 0 <= tail <= 1
