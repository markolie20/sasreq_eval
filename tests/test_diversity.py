"""Diversity diagnostics of the final models (research design §3.5, §5.3; decided 2026-10-06): catalogue
coverage and intra-list diversity, on the very lists the metrics scored."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from seqrec_eval import cli
from seqrec_eval.diversity import DIVERSITY_VERSION, diversity_path, item_vectors, list_diversity
from seqrec_eval.evaluate import evaluate_phase, phase_targets
from seqrec_eval.models import model_spec
from seqrec_eval.protocol import load_protocol
from seqrec_eval.runner import plan_finals
from seqrec_eval.splits import final_split, load_split
from test_smoke import DEVICE, workspace  # noqa: F401  (the shared synthetic workspace)


def test_coverage_and_intra_list_diversity_on_known_vectors():
    # items 0 and 1 point the same way, 2 is orthogonal to both, 3 has no training interaction (zero)
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    lists = np.array([[0, 1, 2], [0, 2, 3]])
    out = list_diversity(lists, vectors, [1, 2, 3])
    assert out["1"]["coverage"] == 0.25 and np.isnan(out["1"]["intra_list_diversity"])  # one item has no pairs
    # top 2: user 0 holds two identical items (dissimilarity 0), user 1 two orthogonal ones (1)
    assert out["2"] == {"coverage": 0.75, "intra_list_diversity": pytest.approx(0.5)}  # items 0, 1 and 2
    # top 3: user 0's pairs 0-1, 0-2, 1-2 give 0, 1, 1; user 1's 0-2, 0-3, 2-3 give 1, 1, 1
    assert out["3"]["coverage"] == 1.0
    assert out["3"]["intra_list_diversity"] == pytest.approx((2 / 3 + 1) / 2)
    # an index outside the catalogue is uncovered and dissimilar to everything
    assert list_diversity(np.array([[0, 9]]), vectors, [2])["2"] == {"coverage": 0.25, "intra_list_diversity": 1.0}


def test_item_similarity_is_co_occurrence():
    # 6 users × 5 items: items 0 and 1 had the same users {0, 1, 4}, item 2 others {2, 3, 5}, item 3 nobody,
    # item 4 one user of each {0, 2}. The matrix has rank 3, so its rank-4 approximation is exact and the
    # cosines are the exact co-occurrence cosines: |shared users| / sqrt(|users of i| · |users of j|).
    users = {0: [0, 1, 4], 1: [0, 1, 4], 2: [2, 3, 5], 4: [0, 2]}
    dense = np.zeros((6, 5))
    for item, who in users.items():
        dense[who, item] = 1.0
    counts = dense * np.arange(1, 31).reshape(6, 5)  # uneven counts: only who interacted may count
    vectors = item_vectors(csr_matrix(counts))
    assert vectors[0] @ vectors[1] == pytest.approx(1.0)
    assert vectors[0] @ vectors[2] == pytest.approx(0.0, abs=1e-9)
    assert vectors[0] @ vectors[4] == pytest.approx(1 / np.sqrt(3 * 2))
    assert vectors[2] @ vectors[4] == pytest.approx(1 / np.sqrt(3 * 2))
    assert np.allclose(vectors[3], 0.0)
    assert np.allclose(np.linalg.norm(vectors[[0, 1, 2, 4]], axis=1), 1.0)


def test_the_diversity_of_each_final_is_measured_on_the_lists_its_metrics_scored(workspace, tmp_path):
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    common = ["--protocol", str(root / "protocol.toml"), "--work-dir", str(work)]
    # a final made under another selection, and one not finished: neither describes the selected model
    stale, unfinished = (spec.directory(work) for spec in plan_finals(protocol, work, "synth", "sasrec"))
    made = json.loads((stale / "spec.json").read_text())
    (stale / "spec.json").write_text(json.dumps({**made, "source_trial": made["source_trial"] + 1}))
    (unfinished := plan_finals(protocol, work, "synth", "elsa")[1].directory(work)).joinpath("done.json").unlink()
    assert cli.main(common + ["diversity", "--device", DEVICE]) == 0
    assert not diversity_path(stale).exists() and not diversity_path(unfinished).exists()
    measured = 0
    for model in protocol.models:
        for spec in plan_finals(protocol, work, "synth", model):
            directory = spec.directory(work)
            if not (directory / "model.zip").exists() or directory in (stale, unfinished):
                continue
            record = json.loads(diversity_path(directory).read_text())
            test = json.loads((directory / "test.json").read_text())
            assert record["version"] == DIVERSITY_VERSION
            assert record["n_users"] == record["n_scored_rows"] == test["n_scored_rows"] > 0
            for k, values in record["cutoffs"].items():
                assert 0.0 < values["coverage"] <= 1.0
                assert k == "1" or 0.0 <= values["intra_list_diversity"] <= 2.0
            measured += 1
    assert measured >= 6  # popularity, ELSA, GRU and SASRec, two seeds each, less the two left out
    stamps = {p: p.stat().st_mtime_ns for p in work.rglob("diversity.json")}
    assert cli.main(common + ["diversity", "--device", DEVICE]) == 0
    assert {p: p.stat().st_mtime_ns for p in work.rglob("diversity.json")} == stamps  # nothing measured twice
    assert cli.main(common + ["report", "--reference", "elsa"]) == 0
    report = (work / "reports" / "stage1.md").read_text()
    assert "## Diversity (diagnostics)" in report and "| synth | gru | 2 |" in report

    # the lists are the scored ones, row for row: a user is a hit at 5 exactly when their top 5 holds a next item
    split = final_split(protocol, load_split(work, "synth", protocol))
    users = split.eval_user_ids("test")
    targets = phase_targets(split, "test", "next").tocsr()
    checked = 0
    for model in ("popularity", "gru", "sasrec", "elsa"):
        spec = plan_finals(protocol, work, "synth", model)[0]
        registered = model_spec(model)
        trainer = registered.cls.load(spec.directory(work) / "model.zip", device="cpu")
        lists = []
        result = evaluate_phase(trainer, split, "test", family=registered.family, protocol=protocol,
                                exclude_seen=protocol.dataset("synth").exclude_seen, lists=lists)
        ranked = np.vstack(lists)
        rows = np.flatnonzero(np.isin(users, np.asarray(result.sample_ids).astype(str)))
        assert ranked.shape[0] == len(rows) == len(result.sample_ids)
        hit = np.array([targets[row, ranked[i, :5]].sum() > 0 for i, row in enumerate(rows)], dtype=float)
        assert np.array_equal(hit, result.per_user["hit_rate@5"]), model
        checked += 0 < hit.sum() < hit.size  # a check that could fail: some users hit, some not
    assert checked
