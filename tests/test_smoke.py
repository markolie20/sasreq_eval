"""The whole workflow on a tiny synthetic dataset, through the real library.

A synthetic adapter is registered with the library's builder, so ``prepare``
exercises the actual temporal split rather than a stand-in, and every command
after it runs on what that split produced. The data is a cycle -- user ``u``
reads 30 consecutive items from ``u`` onwards, one per day, then starts that loop
again, so later windows contain repeats -- with ratings alternating 1 and 5, so a
4-star threshold, if one were wrongly applied, would visibly halve every history.
Every fifth user switches to three new items for their last 8 days, so the
validation and test catalogues hold items the training catalogue lacks, as on
every real temporal split; a model that cannot take those in fails here. And
one user (``RARE_USER``) starts the test window on a track nobody else plays:
with ``item_min_support = 2`` the builder deletes it, as Yambda's filter deletes
rare new tracks, and that user's real next item is gone (H16).
Each history covers only half the catalogue, which SASRec needs: it samples
negatives from outside the history.
"""

from __future__ import annotations

import json
import math
import os
import socket
from pathlib import Path

import pandas as pd
import pytest
from compresso_recsys import builder
from compresso_recsys.datasets._public import PublicDataset

import numpy as np
from compresso_recsys import ItemSequences

from seqrec_eval import cli, refit
from seqrec_eval.analysis import fit_baseline
from seqrec_eval.evaluate import evaluate_phase, scored_rows
from seqrec_eval.models import model_spec
from seqrec_eval.protocol import load_protocol
from seqrec_eval.runner import RunLock, execute, plan_finals, plan_trials
from seqrec_eval.search import trial_params
from seqrec_eval.splits import final_split, load_split
from seqrec_eval.timestamps import _boundaries, prepared_events

#: The device the workflow trains on: ``SEQREC_EVAL_TEST_DEVICE=cuda`` runs every search, final and ablation
#: run of these tests on the GPU, as the DGX will. Checks that call a run directly stay on the CPU.
DEVICE = os.environ.get("SEQREC_EVAL_TEST_DEVICE", "cpu")

N_USERS, N_ITEMS, LOOP, DAYS, PERIOD_DAYS, NEW_ITEMS, NEW_DAYS = 80, 60, 30, 40, 5, 3, 8
RARE_USER = 1
#: the first step of RARE_USER inside the test window (the log ends on day DAYS + 1, users start 0-2 days late)
RARE_STEP = DAYS + 1 - PERIOD_DAYS - RARE_USER % 3
DAY = 86_400
START = 1_700_000_000


def _item(user: int, step: int) -> str:
    if user == RARE_USER and step == RARE_STEP:
        return "rare"  # one listener: deleted by the test stage's new-item filter
    if user % 5 == 0 and step >= DAYS - NEW_DAYS:
        return f"n{step % NEW_ITEMS}"  # first seen after the training window
    return f"i{(user + step % LOOP) % N_ITEMS:02d}"


def synthetic_events() -> pd.DataFrame:
    return pd.DataFrame([
        {"user_id": f"u{user:03d}", "item_id": _item(user, step),
         "value": 5.0 if step % 2 else 1.0, "timestamp": START + (step + user % 3) * DAY}
        for user in range(N_USERS) for step in range(DAYS)
    ])


class Synthetic(PublicDataset):
    name = "synthetic"

    def download(self) -> None:
        pass

    def prepare(self) -> None:
        self.finish(synthetic_events(), pd.DataFrame({"item_id": pd.Series(dtype=str)}))


PROTOCOL = f"""
[protocol]
version = 1
cutoffs = [1, 5]
metrics = ["ndcg", "recall", "calibrated_recall", "hit_rate"]
primary_metric = "ndcg@5"
seeds = [0, 1]
trials_per_model = 2
search_seed = 7
max_val_users = 30
targets = "next"
refit = true

[latency]
history_bins = [1, 10, 30]
requests_per_bin = 5
warmup_requests = 2

[datasets.synth]
builder = "synthetic"
temporal_period_hours = {PERIOD_DAYS * 24}
train_users = "all"
min_user_support = 2
item_min_support = 2
min_value_to_keep = "none"
set_all_values_to = 1.0
exclude_seen = false
new_item_diagnostic = true

[models.popularity]
family = "matrix"

[models.ease]
family = "matrix"
max_items = 10
[models.ease.space]
l2 = {{ loguniform = [1.0, 100.0] }}

[models.elsa]
family = "matrix"
[models.elsa.space]
latent_dim = {{ choice = [4, 8] }}
epochs = {{ choice = [2] }}

[models.gru]
family = "sequence"
[models.gru.fixed]
epochs = 2
batch_size = 16
[models.gru.space]
embedding_dim = {{ choice = [8] }}
hidden_dim = {{ choice = [8, 16] }}
max_history_length = {{ choice = [10, 20] }}

[models.sasrec]
family = "sequence"
[models.sasrec.fixed]
epochs = 2
batch_size = 16
d_model = 8
n_blocks = 1
[models.sasrec.space]
n_negatives = {{ choice = [1, 4] }}
max_history_length = {{ choice = [10, 20] }}

[baselines.popularity]
kind = "popularity"
[baselines.popularity.space]
count = {{ choice = ["events", "users"] }}

[baselines.time_popularity]
kind = "time_popularity"
[baselines.time_popularity.space]
half_life_days = {{ loguniform = [1.0, 30.0] }}

[baselines.replay]
kind = "replay"
[baselines.replay.space]
order = {{ choice = ["recency", "frequency"] }}

[baselines.markov]
kind = "markov"
"""


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("suite")
    (root / "protocol.toml").write_text(PROTOCOL)
    patch = pytest.MonkeyPatch()
    patch.setitem(builder.DATASETS, "synthetic", builder.DatasetSpec(
        Synthetic, str(root / "unused.zip"), seed=0, val_users=5, test_users=5,
        # The registry default the protocol must override: "none" has to reach the
        # builder as -inf, because None would fall back to this 4-star threshold.
        min_value_to_keep=4.0, min_entity_text_words=0, temporal_period_hours=PERIOD_DAYS * 24,
    ))
    common = ["--protocol", str(root / "protocol.toml"), "--work-dir", str(root / "work")]
    assert cli.main(common + ["prepare", "--data-dir", str(root / "data"), "--quiet"]) == 0
    assert cli.main(common + ["search", "--device", DEVICE]) == 0
    assert cli.main(common + ["final", "--device", DEVICE]) == 0
    assert cli.main(common + ["latency", "--threads", "1"]) == 0
    assert cli.main(common + ["analyse"]) == 0
    assert cli.main(common + ["report", "--reference", "elsa"]) == 0
    yield root, common
    patch.undo()


def _runs(root: Path, name: str) -> list[Path]:
    return sorted((root / "work" / "runs").glob(f"synth/*/*/{name}"))


def test_keep_everything_reaches_the_builder_as_minus_infinity(workspace):
    root, _ = workspace
    split = load_split(root / "work", "synth")
    assert split.info["build_parameters"]["min_value_to_keep"] == -math.inf
    assert split.info["resolved_build_parameters"]["min_value_to_keep"] == -math.inf

    # Every event before the test window is in each test history, ratings of 1 included.
    events = synthetic_events()
    test_start = events.timestamp.max() - PERIOD_DAYS * DAY
    expected = events[events.timestamp < test_start].groupby("user_id").size()
    lengths = dict(zip(split.eval_user_ids("test"), split.data["test_source_sequences"].row_lengths))
    assert lengths == {user: int(expected[user]) for user in lengths}


def test_the_test_catalogue_holds_items_the_training_catalogue_lacks(workspace):
    root, _ = workspace
    split = load_split(root / "work", "synth")
    n_train = len(split.data["train_item_ids"])
    assert set(split.data["test_item_ids"][n_train:]) == {f"n{i}" for i in range(NEW_ITEMS)}
    assert split.data["test_source_matrix"][:, n_train:].nnz > 0  # in the histories the models are scored from


# ---------------------------------------------------------------------------
# refit: everything scored on test is trained on train+validation
# ---------------------------------------------------------------------------

def test_the_refit_set_is_everything_before_the_test_window(workspace):
    root, _ = workspace
    split = load_split(root / "work", "synth")
    assert split.info["refit"]["proved_against"] == ["x_train", "x_train_sequences", "train_source_sequences",
                                                     "train_user_ids"]
    # computed here from the generated events, not by the code under test
    test_start = _boundaries(split.info["manifest"])["test_target_start"]
    events = synthetic_events()
    catalogue = list(split.data["val_item_ids"])
    before = events[(events.timestamp < test_start) & events.item_id.isin(catalogue)]
    expected = {user: list(rows.sort_values("timestamp", kind="stable").item_id)
                for user, rows in before.groupby("user_id")}
    sequences, users = split.data["x_refit_sequences"], split.data["refit_user_ids"]
    assert {user: [catalogue[i] for i in sequences.row(r)] for r, user in enumerate(users)} == expected
    # the items first seen in the validation window are in it
    assert {"n0", "n1", "n2"} <= {catalogue[i] for i in sequences.values}
    assert set(split.data["train_user_ids"]) <= set(users)


def test_the_proof_refuses_a_training_set_the_library_did_not_build(workspace):
    root, _ = workspace
    split = load_split(root / "work", "synth")
    params, manifest = split.info["build_parameters"], split.info["manifest"]
    events = prepared_events(params, root / "data")
    rule = refit.train_rule(params, root / "data")
    refit.prove(events, split.data, manifest, rule)  # the real split passes
    changed = split.data["x_train_sequences"]
    values = changed.values.copy()
    values[3] = (values[3] + 1) % changed.n_items
    wrong = {**split.data, "x_train_sequences": ItemSequences(values=values, indptr=changed.indptr,
                                                              n_items=changed.n_items)}
    with pytest.raises(refit.RefitAlignmentError, match="differs at position"):
        refit.prove(events, wrong, manifest, rule)
    stages = manifest["stages"]["data"]
    shifted = {"stages": {"data": {**stages, "validation_target_start": stages["validation_target_start"] + DAY}}}
    with pytest.raises(refit.RefitAlignmentError):
        refit.prove(events, split.data, shifted, rule)


def test_final_models_are_refitted_on_train_and_validation(workspace):
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    split = load_split(root / "work", "synth")
    n_train, n_refit = len(split.data["train_item_ids"]), len(split.data["val_item_ids"])
    assert n_refit == n_train + NEW_ITEMS
    for model in ("popularity", "elsa", "gru", "sasrec"):
        for trial in plan_trials(protocol, "synth", model):
            record = json.loads((trial.directory(root / "work") / "done.json").read_text())
            assert (record["trained_on"], record["train_items"]) == ("train", n_train)
            assert "val" in record
        for final in plan_finals(protocol, root / "work", "synth", model):
            directory = final.directory(root / "work")
            record = json.loads((directory / "done.json").read_text())
            assert (record["trained_on"], record["train_items"]) == ("train+val", n_refit)
            # a refitted model has seen the validation window: it is scored on test only
            assert "val" not in record and not (directory / "val.json").exists()
            assert "test" in record


def test_a_refitted_split_refuses_validation_and_a_run_refuses_the_wrong_split(workspace):
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    split = load_split(root / "work", "synth")
    refitted = final_split(protocol, split)
    model = model_spec("popularity").build({}, n_items=len(refitted.data["train_item_ids"]), device="cpu", seed=0)
    model.fit(refitted.data["x_train"], item_ids=refitted.data["train_item_ids"])
    evaluate_phase(model, refitted, "test", family="matrix", protocol=protocol, exclude_seen=False)
    with pytest.raises(ValueError, match="seen the validation window"):
        evaluate_phase(model, refitted, "val", family="matrix", protocol=protocol, exclude_seen=False)
    elsewhere = root / "work" / "elsewhere"
    with pytest.raises(ValueError, match="must be given a split trained on train"):
        execute(plan_trials(protocol, "synth", "popularity")[0], refitted, protocol, elsewhere, device="cpu")
    with pytest.raises(ValueError, match=r"trained on train\+val"):
        execute(plan_finals(protocol, root / "work", "synth", "popularity")[0], split, protocol, elsewhere,
                device="cpu")


def test_baselines_are_refitted_before_test(workspace):
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    split = load_split(root / "work", "synth")
    refitted = final_split(protocol, split)
    popularity = fit_baseline("popularity", {"count": "events"}, refitted)
    assert popularity.popularity_.sum() == refitted.data["x_train_sequences"].values.size
    assert refitted.data["x_train_sequences"].values.size > split.data["x_train_sequences"].values.size
    records = sorted((root / "work" / "analysis" / "synth").glob("*/baselines/*/*/test.done.json"))
    records += sorted((root / "work" / "analysis" / "synth").glob("*/controls/*/*/test.done.json"))
    assert records and all(json.loads(r.read_text())["trained_on"] == "train+val" for r in records)
    selected = sorted((root / "work" / "analysis" / "synth").glob("*/baselines/*/*/selected.json"))
    assert selected and all("val" in json.loads(r.read_text()) for r in selected)  # still searched on validation


# ---------------------------------------------------------------------------
# next-item targets: the real next moment, scored only where a model could get it (H16)
# ---------------------------------------------------------------------------

def test_the_next_target_is_the_real_first_moment_in_the_window(workspace):
    root, _ = workspace
    split = load_split(root / "work", "synth")
    catalogue = list(split.data["test_item_ids"])
    assert "rare" not in catalogue  # one listener: the test stage deleted it
    test_start = _boundaries(split.info["manifest"])["test_target_start"]
    events = synthetic_events()
    window = events[events.timestamp >= test_start]
    first = window[window.timestamp == window.groupby("user_id").timestamp.transform("min")]
    expected = {user: {item for item in rows.item_id if item in catalogue} for user, rows in first.groupby("user_id")}
    targets = split.data["test_next_target_matrix"]
    users = split.eval_user_ids("test")
    got = {user: {catalogue[i] for i in targets[r].indices} for r, user in enumerate(users)}
    assert got == {user: expected[user] for user in users}
    # the rare user's real next item was deleted: an empty row, not their following listen
    assert got[f"u{RARE_USER:03d}"] == set()
    assert split.info["timestamps"]["next_targets"]["test"]["users_whose_next_item_was_deleted"] == 1


def test_only_users_whose_next_item_a_model_can_recommend_are_scored(workspace):
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    split = load_split(root / "work", "synth")
    refitted = final_split(protocol, split)
    users = split.eval_user_ids("test")
    targets = split.data["test_next_target_matrix"]
    new_next = {users[r] for r in range(len(users))
                if any(split.data["test_item_ids"][i].startswith("n") for i in targets[r].indices)}
    assert new_next  # the users on new items start the test window on one
    no_refit = set(users[scored_rows(split, "test", "next")])
    refit = set(users[scored_rows(refitted, "test", "next")])
    # the rare user is never scored; the new items are unrecommendable to a model trained on the training
    # window, and recommendable once it is refitted on validation, where they first appeared
    assert f"u{RARE_USER:03d}" not in refit | no_refit
    assert refit - no_refit == new_next
    assert set(users) - refit == {f"u{RARE_USER:03d}"}
    assert scored_rows(split, "test", "window") is None  # the window diagnostic scores every user

    for final in _runs(root, "final-seed*"):
        test = json.loads((final / "test.json").read_text())
        assert test["metadata"]["rows_sampled"] == len(refit)
        assert test["metadata"]["rows_unrecommendable_next"] == 1
    report = (root / "work" / "reports" / "stage1.md").read_text()
    assert f"Next-item metrics over {len(refit):,} test users. 1 more are left out" in report
    # the search: the validation sample, less the users whose next item the training catalogue lacks
    scorable_val = set(scored_rows(split, "val", "next"))
    expected_val = len([r for r in split.val_rows if r in scorable_val])
    ran = [trial for trial in _runs(root, "trial-*") if (trial / "val.json").exists()]  # EASE was skipped
    assert ran
    for trial in ran:
        val = json.loads((trial / "val.json").read_text())
        assert val["metadata"]["rows_sampled"] == expected_val
        assert val["metadata"]["rows_unrecommendable_next"] == len(split.val_rows) - expected_val


def test_a_model_too_large_for_the_refit_catalogue_is_skipped_at_the_search_already(workspace, tmp_path):
    # training catalogue 60 <= max_items 61 < refit catalogue 63: searched in full, then unable to be finalised
    root, _ = workspace
    path = tmp_path / "protocol.toml"
    path.write_text(PROTOCOL.replace("max_items = 10", "max_items = 61"))
    protocol = load_protocol(path)
    split = load_split(root / "work", "synth")
    assert len(split.data["train_item_ids"]) <= 61 < len(split.data["val_item_ids"])
    trial = plan_trials(protocol, "synth", "ease")[0]
    assert execute(trial, split, protocol, tmp_path / "work", device="cpu", log=lambda _: None) == "skipped"
    reason = json.loads((trial.directory(tmp_path / "work") / "done.json").read_text())["reason"]
    assert "catalogue the final runs fit" in reason


def test_search_scores_a_fixed_validation_sample_and_never_the_test_set(workspace):
    root, _ = workspace
    split = load_split(root / "work", "synth")
    assert split.val_rows is not None and len(split.val_rows) == 30
    for trial in _runs(root, "trial-*"):
        record = json.loads((trial / "done.json").read_text())
        if record["status"] == "skipped":
            continue
        assert json.loads((trial / "val.json").read_text())["metadata"]["rows_sampled"] == 30
        assert not (trial / "test.json").exists()
        assert "test" not in record


def test_every_planned_run_finished_and_the_dense_model_was_skipped(workspace):
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    for model in protocol.models:
        for spec in plan_trials(protocol, "synth", model):
            record = json.loads((spec.directory(root / "work") / "done.json").read_text())
            assert record["status"] == ("skipped" if model == "ease" else "done")
    finals = _runs(root, "final-seed*")
    assert {path.parent.parent.name for path in finals} == {"popularity", "elsa", "gru", "sasrec"}
    for final in finals:
        assert (final / "test.json").exists() and (final / "test_new.json").exists()
        assert json.loads((final / "done.json").read_text())["model_saved"]


def test_rerunning_is_a_no_op(workspace):
    root, common = workspace
    stamps = {path: path.stat().st_mtime_ns for path in (root / "work" / "runs").rglob("done.json")}
    assert cli.main(common + ["search", "--device", DEVICE]) == 0
    assert cli.main(common + ["final", "--device", DEVICE]) == 0
    assert {path: path.stat().st_mtime_ns for path in stamps} == stamps


def test_a_run_claimed_by_a_live_process_is_left_alone(workspace, tmp_path):
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    spec = plan_trials(protocol, "synth", "gru")[0]
    directory = spec.directory(tmp_path)
    directory.mkdir(parents=True)
    assert RunLock(directory).acquire()  # held by this, very much alive, process
    split = load_split(root / "work", "synth")
    assert execute(spec, split, protocol, tmp_path, device="cpu", log=lambda _: None) == "running-elsewhere"
    RunLock(directory).release()


def test_a_dead_process_lock_is_reclaimed(workspace, tmp_path):
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    spec = plan_trials(protocol, "synth", "popularity")[0]
    directory = spec.directory(tmp_path)
    directory.mkdir(parents=True)
    (directory / ".lock").write_text(json.dumps({"host": socket.gethostname(), "pid": 2**22 + 12345}))
    split = load_split(root / "work", "synth")
    assert execute(spec, split, protocol, tmp_path, device="cpu", log=lambda _: None) == "done"


def test_report_compares_against_the_reference_and_includes_latency(workspace):
    root, _ = workspace
    report = (root / "work" / "reports" / "stage1.md").read_text()
    assert "Comparison against **elsa**" in report
    assert "| SE users | SE seeds | df | adjusted p | users-only p |" in report  # seeds counted (H23)
    assert "Against the floor, **markov**" in report  # the synthetic loop is a first-order chain
    assert "**ease** skipped" in report
    assert "CPU inference latency" in report
    assert "refitted on train and validation before test" in report
    for model in ("popularity", "gru", "sasrec"):
        assert f"| {model} |" in report
    # a bin of 5 requests is too few for its P95 to stand as the worst (review C6)
    assert "— (no bin with 50 requests)" in report
    for final in _runs(root, "final-seed0"):
        latency = json.loads((final / "latency.json").read_text())
        assert latency["threads"] == 1
        assert len(latency["load_average"]) == 2 and latency["source_trial"] is not None
        assert latency["catalog_items"] == len(load_split(root / "work", "synth").data["test_item_ids"])
        assert latency["overall"]["n"] == sum(b["n"] for b in latency["by_history_length"].values()) > 0


def test_trials_are_reproducible_and_fingerprints_are_isolated(workspace, tmp_path):
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    gru = protocol.model("gru")
    assert trial_params(protocol, gru, 1) == trial_params(protocol, gru, 1)

    edited = tmp_path / "protocol.toml"
    edited.write_text(PROTOCOL.replace("hidden_dim = { choice = [8, 16] }", "hidden_dim = { choice = [8, 32] }"))
    changed = load_protocol(edited)
    assert changed.run_fingerprint("synth", "gru") != protocol.run_fingerprint("synth", "gru")
    assert changed.run_fingerprint("synth", "sasrec") == protocol.run_fingerprint("synth", "sasrec")
    assert changed.dataset_fingerprint("synth") == protocol.dataset_fingerprint("synth")


def test_the_local_protocol_differs_from_the_main_one_only_in_batch_size():
    # a stale local copy silently ran another protocol (2026-09-29: no refit, window targets, Yambda at 20%)
    repo = Path(__file__).resolve().parents[1]
    main, local = load_protocol(repo / "protocol.toml"), load_protocol(repo / "protocol.local.toml")
    for section in ("protocol", "latency", "datasets", "models", "ablations", "baselines", "repeat_strata"):
        a, b = dict(main.raw.get(section, {})), dict(local.raw.get(section, {}))
        if section == "protocol":
            a.pop("eval_batch_size", None), b.pop("eval_batch_size", None)
        assert a == b, f"protocol.local.toml differs from protocol.toml in [{section}]"
    assert all(main.run_fingerprint(d, m) == local.run_fingerprint(d, m) for d in main.datasets for m in main.models)


# ---------------------------------------------------------------------------
# required protocol keys (H11), the failure policy (H06, H07), recorded code (H05)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key, line", [("targets", 'targets = "next"\n'), ("refit", "refit = true\n"),
                                       ("train_users", 'train_users = "all"\n')])
def test_a_protocol_must_state_targets_refit_and_training_users(tmp_path, key, line):
    from seqrec_eval.protocol import ProtocolError

    path = tmp_path / "protocol.toml"
    path.write_text(PROTOCOL.replace(line, "", 1))
    with pytest.raises(ProtocolError, match=f"{key} is required"):
        load_protocol(path)


def _trials(tmp_path, values):
    """A fresh work dir where GRU's trials ended as given: a number (done) or an error string (failed)."""
    protocol_path = tmp_path / "protocol.toml"
    protocol_path.write_text(PROTOCOL)
    protocol = load_protocol(protocol_path)
    work = tmp_path / "work"
    for spec, value in zip(plan_trials(protocol, "synth", "gru"), values):
        directory = spec.directory(work)
        directory.mkdir(parents=True)
        if isinstance(value, str):
            (directory / "failed.json").write_text(json.dumps({"error": value}))
        else:
            (directory / "done.json").write_text(json.dumps({"status": "done", "val": {"ndcg@5": value}}))
    return protocol, work


def test_no_model_is_selected_while_a_trial_has_failed(tmp_path):
    from seqrec_eval.runner import accepted_failures_path

    protocol, work = _trials(tmp_path, [0.3, "RuntimeError: CUDA out of memory"])
    with pytest.raises(RuntimeError, match="No model is selected while a trial has failed"):
        plan_finals(protocol, work, "synth", "gru")
    specs = plan_finals(protocol, work, "synth", "gru", accept_failed=True)
    assert {spec.source_trial for spec in specs} == {0}
    record = json.loads(accepted_failures_path(protocol, work, "synth", "gru").read_text())
    assert record["trials"] == [1] and "out of memory" in record["failures"][0]["error"]
    assert plan_finals(protocol, work, "synth", "gru")  # accepted once, it no longer blocks


def test_a_non_finite_validation_score_is_a_failed_trial(tmp_path):
    from seqrec_eval.runner import summarize_trials

    protocol, work = _trials(tmp_path, [float("nan"), 0.2])
    summary = summarize_trials(protocol, work, "synth", "gru")
    assert (summary.failed, summary.done, summary.best.index) == (1, 1, 1)  # NaN is never "best"
    assert "not a finite number" in summary.failures[0]["error"]
    with pytest.raises(RuntimeError, match="trial 0") as refused:
        plan_finals(protocol, work, "synth", "gru")
    # a rerun repeats the same seed and the same score: the message says so (review B4)
    assert "non-finite score" in str(refused.value) and "does not help" in str(refused.value)


def test_a_non_finite_baseline_score_stops_its_search(workspace, tmp_path):
    from seqrec_eval.analysis import _baseline_dir, select_baseline

    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    directory = _baseline_dir(protocol, tmp_path, "synth", "popularity")
    directory.mkdir(parents=True)
    for trial, value in ((0, float("nan")), (1, 0.2)):
        (directory / f"trial-{trial:03d}.json").write_text(json.dumps(
            {"trial": trial, "params": {}, "val": {"ndcg@5": value}}))
    with pytest.raises(ValueError, match="not a finite"):
        select_baseline(protocol, tmp_path, load_split(root / "work", "synth"), "popularity", log=lambda _: None)


def test_the_report_lists_failures_and_the_code_the_runs_used(workspace, tmp_path):
    import shutil

    from seqrec_eval.report import build_report
    from seqrec_eval.runner import accepted_failures_path

    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    models = ["popularity", "elsa", "gru", "sasrec"]
    report, _ = build_report(protocol, root / "work", ["synth"], models, "elsa")
    assert "Code: compresso-recsys" in report and "for all" in report  # one build, recorded in every run
    # the protocol by full path and content, so a quick local run's report cannot pass for a real one
    import hashlib
    digest = hashlib.sha256((root / "protocol.toml").read_bytes()).hexdigest()[:12]
    assert f"Protocol `{(root / 'protocol.toml').resolve()}` (file sha256 {digest})" in report

    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    # gru: trial 1 failed and was accepted; sasrec: final seed 1 failed; one elsa trial from another build
    gru = plan_trials(protocol, "synth", "gru")[1].directory(work)
    (gru / "done.json").unlink()
    (gru / "failed.json").write_text(json.dumps({"error": "RuntimeError: CUDA out of memory"}))
    accepted_failures_path(protocol, work, "synth", "gru").write_text(json.dumps(
        {"trials": [1], "failures": [{"trial": 1, "error": "RuntimeError: CUDA out of memory"}],
         "accepted_at": "2026-09-29 12:00:00"}))
    sasrec = plan_finals(protocol, work, "synth", "sasrec")[1].directory(work)
    (sasrec / "done.json").unlink()
    (sasrec / "failed.json").write_text(json.dumps({"error": "RuntimeError: boom"}))
    elsa = plan_trials(protocol, "synth", "elsa")[0].directory(work) / "done.json"
    record = json.loads(elsa.read_text())
    record["code"]["library"]["code_sha256"] = "0000000000000000"
    elsa.write_text(json.dumps(record))

    report, _ = build_report(protocol, work, ["synth"], models, "elsa")
    assert "**gru**: 1 trial(s) failed and were accepted as unrunnable on 2026-09-29 12:00:00" in report
    assert "⛔ **sasrec**: final seed(s) [1] failed" in report
    assert "⚠ These results come from 2 different code builds" in report


def test_the_floor_waits_for_every_baseline(workspace, tmp_path):
    # review N34: after an interrupted `analyse`, the floor was the strongest of the baselines that had finished
    import shutil

    from seqrec_eval.analysis import _baseline_dir
    from seqrec_eval.analysis_report import dataset_analysis
    from seqrec_eval.report import build_report

    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    for path in _baseline_dir(protocol, work, "synth", "markov").glob("test.*"):
        path.unlink()  # the floor, here: Markov wins on the synthetic chain
    report, _ = build_report(protocol, work, ["synth"], ["popularity", "elsa", "gru", "sasrec"], "elsa")
    assert "Against the floor" not in report
    assert "_No floor yet: markov not analysed." in report
    analysis, _ = dataset_analysis(protocol, work, "synth")
    assert "(floor)" not in analysis and "_No floor yet:" in analysis


def test_the_library_version_is_the_checkouts_when_imported_from_one(tmp_path, monkeypatch):
    """Imported from a checkout (PYTHONPATH=<checkout>/src), the metadata Python finds can be another build's or
    a stale egg-info left in src/ (2026-09-30: 0.3.6 reported for the 0.3.7+trainusers branch)."""
    import importlib.metadata
    import sys
    import types

    from seqrec_eval import splits

    package = tmp_path / "src" / "compresso_recsys"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    fake = types.ModuleType("compresso_recsys")
    fake.__file__ = str(package / "__init__.py")
    monkeypatch.setitem(sys.modules, "compresso_recsys", fake)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.3.6")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "compresso-recsys"\nversion = "9.9.9+branch"\n')
    record = splits.library_provenance()
    assert (record["version"], record["metadata_version"]) == ("9.9.9+branch", "0.3.6")
    # another project's pyproject, or none: the metadata is all there is
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "something-else"\nversion = "1.0"\n')
    assert splits.library_provenance()["version"] == "0.3.6"
    (tmp_path / "pyproject.toml").unlink()
    record = splits.library_provenance()
    assert record["version"] == "0.3.6" and "metadata_version" not in record


def test_a_final_made_under_another_selection_is_refused(workspace, tmp_path):
    # review A3: after --accept-failed and a successful retry, the old finals passed for the new selection's
    from dataclasses import asdict

    root, _ = workspace
    protocol, work = _trials(tmp_path, [0.1, 0.2])  # trial 1 is selected
    spec = plan_finals(protocol, work, "synth", "gru")[0]
    directory = spec.directory(work)
    directory.mkdir(parents=True)
    (directory / "spec.json").write_text(json.dumps({**asdict(spec), "source_trial": 0, "params": {"x": 1}}))
    (directory / "done.json").write_text(json.dumps({"status": "done"}))
    split = final_split(protocol, load_split(root / "work", "synth"))
    messages = []
    assert execute(spec, split, protocol, work, device="cpu", log=messages.append) == "stale-selection"
    assert "Made with trial 0's configuration, but trial 1 is selected now" in messages[0]


def test_selecting_from_an_unfinished_search_is_recorded(tmp_path):
    from seqrec_eval.runner import incomplete_selection_path

    protocol, work = _trials(tmp_path, [0.1])  # 1 of GRU's 2 trials
    plan_finals(protocol, work, "synth", "gru", allow_incomplete=True)
    record = json.loads(incomplete_selection_path(protocol, work, "synth", "gru").read_text())
    assert (record["finished"], record["planned"], record["selected_trial"]) == (1, 2, 0)


def test_a_run_whose_process_keeps_dying_becomes_a_failure(workspace, tmp_path):
    # review B3: a kernel OOM kill records nothing, so a restart would run it first, for ever
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    spec = plan_trials(protocol, "synth", "popularity")[0]
    split = load_split(root / "work", "synth")
    directory = spec.directory(tmp_path / "dying")
    directory.mkdir(parents=True)
    (directory / "attempts.json").write_text(json.dumps({"started": 2}))
    assert execute(spec, split, protocol, tmp_path / "dying", device="cpu", log=lambda _: None) == "failed"
    assert json.loads((directory / "failed.json").read_text())["error"].startswith("ProcessDied")
    assert execute(spec, split, protocol, tmp_path / "fine", device="cpu", log=lambda _: None) == "done"
    assert not (spec.directory(tmp_path / "fine") / "attempts.json").exists()


def test_retry_failed_starts_the_count_of_deaths_over(workspace, tmp_path):
    # review N31: under --retry-failed, a run at the limit was marked failed without running
    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    spec = plan_trials(protocol, "synth", "popularity")[0]
    directory = spec.directory(tmp_path)
    directory.mkdir(parents=True)
    (directory / "attempts.json").write_text(json.dumps({"started": 2}))
    assert execute(spec, load_split(root / "work", "synth"), protocol, tmp_path, device="cpu", retry_failed=True,
                   log=lambda _: None) == "done"
    assert not (directory / "attempts.json").exists() and not (directory / "failed.json").exists()


def test_kill_stops_a_run_as_ctrl_c_does_not_as_a_death(workspace, tmp_path, monkeypatch):
    # review N31: `kill` is how a nohup launch is stopped; counted as a death, two stops failed the run for good
    import signal
    import time

    from seqrec_eval import cli, runner

    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    spec = plan_trials(protocol, "synth", "popularity")[0]
    directory = spec.directory(tmp_path)

    def killed(*args, **kwargs):
        assert (directory / "attempts.json").exists()  # counted while it runs
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)  # the handler interrupts this
        raise AssertionError("SIGTERM did not stop the run")

    def own(signum, frame):  # whatever handled SIGTERM before the command
        raise AssertionError("the command's handler was not in place")

    monkeypatch.setattr(runner, "_execute", killed)
    previous = signal.signal(signal.SIGTERM, own)
    try:
        with cli._operator_stops(), pytest.raises(KeyboardInterrupt):
            execute(spec, load_split(root / "work", "synth"), protocol, tmp_path, device="cpu", log=lambda _: None)
        assert signal.getsignal(signal.SIGTERM) is own  # put back after the command
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert not (directory / "attempts.json").exists() and not (directory / "failed.json").exists()
    lock = runner.RunLock(directory)
    assert lock.acquire()  # and the claim is released
    lock.release()

    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)  # as under nohup
    try:
        with cli._operator_stops():
            assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN  # nohup's choice is kept
            assert signal.getsignal(signal.SIGTERM) is cli._operator_stop
    finally:
        signal.signal(signal.SIGHUP, previous)

    def stopped(argv):
        raise KeyboardInterrupt("SIGTERM")

    monkeypatch.setattr(cli, "_main", stopped)
    assert cli.main(["status"]) == 130  # neither 1 (failed) nor 3 (work left): `search && final` stops


def test_a_split_from_other_settings_is_refused_and_scoring_flags_are_not_build_settings(workspace, tmp_path):
    # review A4 and B10
    from seqrec_eval.analysis import _controls_fingerprint

    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    load_split(root / "work", "synth", protocol)
    edited = tmp_path / "protocol.toml"
    edited.write_text(PROTOCOL.replace("min_user_support = 2", "min_user_support = 3"))
    with pytest.raises(RuntimeError, match="prepared from other build settings"):
        load_split(root / "work", "synth", load_protocol(edited))
    edited.write_text(PROTOCOL.replace("exclude_seen = false", "exclude_seen = true"))
    flipped = load_protocol(edited)
    assert flipped.dataset_fingerprint("synth") == protocol.dataset_fingerprint("synth")
    load_split(root / "work", "synth", flipped)  # the same split: no rebuild
    assert flipped.run_fingerprint("synth", "gru") != protocol.run_fingerprint("synth", "gru")
    assert _controls_fingerprint(flipped, "synth") != _controls_fingerprint(protocol, "synth")
    assert flipped.evaluation_key("synth") != protocol.evaluation_key("synth")


def test_every_evaluation_excludes_seen_items_after_the_model_ranks(workspace, monkeypatch):
    # review A6: the full data excluded inside the model, a condition after it, and ties fell differently
    from seqrec_eval import evaluate

    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    split = final_split(protocol, load_split(root / "work", "synth"))
    made = []
    real = evaluate.ExcludeSeenPolicy
    monkeypatch.setattr(evaluate, "ExcludeSeenPolicy", lambda *a, **k: made.append(real(*a, **k)) or made[-1])
    model = fit_baseline("popularity", {"count": "events"}, split)
    evaluate.evaluate_phase(model, split, "test", family="sequence", protocol=protocol, exclude_seen=True,
                            targets="window")
    assert made[0].masking


def test_the_report_flags_finals_made_under_another_selection(workspace, tmp_path):
    import shutil

    from seqrec_eval.report import build_report
    from seqrec_eval.runner import summarize_trials

    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    best = summarize_trials(protocol, work, "synth", "gru").best.index
    final = plan_finals(protocol, work, "synth", "gru")[1].directory(work)
    made = json.loads((final / "spec.json").read_text())
    (final / "spec.json").write_text(json.dumps({**made, "source_trial": 1 - best}))
    report, table = build_report(protocol, work, ["synth"], ["popularity", "elsa", "gru", "sasrec"], "elsa")
    assert f"⛔ **gru**: final seed(s) [1] were made with another trial's configuration than the one selected now " \
           f"(#{best})" in report
    # and left out of everything, not only flagged (review N33): its values rest on seed 0 alone
    import csv
    import io

    assert [row["seed"] for row in csv.DictReader(io.StringIO(table)) if row["model"] == "gru"] == ["0"]
    assert next(line for line in report.splitlines() if line.startswith("| gru |")).split(" | ")[4] == "1/2"
    assert "**gru**: final seed(s) [1] have not run yet" not in report


BERT4REC = """
[models.bert4rec]
family = "sequence"
[models.bert4rec.fixed]
epochs = 2
batch_size = 16
d_model = 8
n_blocks = 1
n_heads = 1
duplication_factor = 2
lr = 0.001
[models.bert4rec.space]
max_history_length = { choice = [10, 20] }
"""


def test_bert4rec_runs_through_every_step(workspace, tmp_path):
    # H26, registered 2026-10-06: the trainer builds its own batcher, with the [MASK] token BERT4Rec reads
    import shutil

    from seqrec_eval.report import build_report

    root, _ = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    (tmp_path / "protocol.toml").write_text(PROTOCOL + BERT4REC)
    protocol = load_protocol(tmp_path / "protocol.toml")
    # a model added to the protocol leaves every other model's runs where they are
    before = load_protocol(root / "protocol.toml")
    assert all(protocol.run_fingerprint("synth", m) == before.run_fingerprint("synth", m) for m in before.models)
    common = ["--protocol", str(tmp_path / "protocol.toml"), "--work-dir", str(work)]
    for step in (["search", "--model", "bert4rec", "--device", DEVICE],
                 ["final", "--model", "bert4rec", "--device", DEVICE],
                 ["latency", "--model", "bert4rec", "--threads", "1"]):
        assert cli.main(common + step) == 0, step
    for spec in plan_trials(protocol, "synth", "bert4rec") + plan_finals(protocol, work, "synth", "bert4rec"):
        record = json.loads((spec.directory(work) / "done.json").read_text())
        assert record["status"] == "done" and len(record["history"]) == 2
    report, _ = build_report(protocol, work, ["synth"], ["popularity", "elsa", "gru", "sasrec", "bert4rec"], "elsa")
    row = next(line for line in report.splitlines() if line.startswith("| bert4rec |"))
    assert row.split(" | ")[1:5] == ["2/2", row.split(" | ")[2], row.split(" | ")[3], "2"]  # 2 trials, 2 seeds
    from seqrec_eval.latency import latency_table

    assert "| synth | bert4rec |" in latency_table(protocol, work, ["synth"], ["bert4rec"])


def test_a_run_still_improving_at_its_last_epoch_is_recognised():
    # the epoch grid is capped (SASRec at 100, DECISIONS §32): a run stopped while its loss still fell is flagged
    from seqrec_eval.report import still_improving

    def history(losses, **extra):
        return [{"epoch": float(i + 1), "loss": loss, **extra} for i, loss in enumerate(losses)]

    assert still_improving(None) is None and still_improving(history([0.5])) is None  # nothing to compare
    assert still_improving(history([0.5, 0.5])) is None
    assert still_improving(history([0.5, 0.48])) == pytest.approx((0.04, 1, 2))
    flat_end = [1.0 - 0.01 * i for i in range(80)] + [0.2] * 20  # fell, then levelled off
    assert still_improving(history(flat_end)) is None
    steady = [1.0 - 0.005 * i for i in range(100)]  # still falling by 0.5% of 1.0 an epoch at the end
    fall, window, epochs = still_improving(history(steady))
    assert (window, epochs) == (10, 100) and fall == pytest.approx(0.05 / 0.555)
    assert still_improving(history([0.5, 0.499])) is None  # 0.2%: noise
    assert still_improving(history([0.5, float("nan"), 0.4])) == pytest.approx((0.2, 1, 2))
    # in phases, the last phase alone counts: losses of different phases are not comparable
    assert still_improving(history([1.0, 0.5], phase="warmup") + history([0.3], phase="main")) is None
    assert still_improving(history([0.2, 0.1], phase="warmup") + history([0.5, 0.4], phase="main")) == \
        pytest.approx((0.2, 1, 2))


def test_the_report_flags_finals_still_improving_at_their_last_epoch(workspace, tmp_path):
    import shutil

    from seqrec_eval.report import build_report

    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    models = ["popularity", "elsa", "gru", "sasrec"]
    for done in (work / "runs").rglob("done.json"):  # every trained run converged
        record = json.loads(done.read_text())
        if "history" in record:
            record["history"] = [{"epoch": 1.0, "loss": 0.5}, {"epoch": 2.0, "loss": 0.5}]
            done.write_text(json.dumps(record))
    report, _ = build_report(protocol, work, ["synth"], models, "elsa")
    assert "still improving when training stopped" not in report
    final = plan_finals(protocol, work, "synth", "gru")[1].directory(work) / "done.json"
    record = json.loads(final.read_text())
    record["history"] = [{"epoch": 1.0, "loss": 0.5}, {"epoch": 2.0, "loss": 0.45}]
    final.write_text(json.dumps(record))
    report, _ = build_report(protocol, work, ["synth"], models, "elsa")
    assert ("⚠ **gru**: still improving when training stopped: final seed 1 (training loss −10.0% over the last 1 "
            "of 2 epochs)") in report
    assert "**sasrec**: still improving" not in report


def test_latency_is_neither_measured_nor_shown_for_a_final_of_another_selection(workspace, tmp_path):
    # review N33: latency times the first seed's saved model, which would be another configuration's
    import shutil

    from seqrec_eval.latency import benchmark, latency_table
    from seqrec_eval.runner import summarize_trials

    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    assert "| synth | gru |" in latency_table(protocol, work, ["synth"], ["gru"])
    best = summarize_trials(protocol, work, "synth", "gru").best.index
    final = plan_finals(protocol, work, "synth", "gru")[0].directory(work)
    made = json.loads((final / "spec.json").read_text())
    (final / "spec.json").write_text(json.dumps({**made, "source_trial": 1 - best}))
    table = latency_table(protocol, work, ["synth"], ["gru"])
    assert "| synth | gru |" not in table and "⛔ Not shown: synth/gru" in table
    with pytest.raises(FileNotFoundError, match="another trial's configuration"):
        benchmark(protocol, work, final_split(protocol, load_split(work, "synth")), "gru", threads=1)


def test_every_batch_of_a_phase_asks_the_model_for_the_same_length(workspace):
    # review N30: the width came from each batch's heaviest user, so it changed with eval_batch_size and with
    # the rows scored (stage 1 scores all, an ablation's reference a subset)
    from dataclasses import replace as replaced

    from seqrec_eval import evaluate

    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    split = final_split(protocol, load_split(root / "work", "synth"))
    model = model_spec("popularity").build({}, n_items=len(split.data["train_item_ids"]), device="cpu", seed=0)
    model.fit(split.data["x_train"], item_ids=split.data["train_item_ids"])
    asked, real = [], model.predict_on_batch
    model.predict_on_batch = lambda source, *, k, **options: asked.append(k) or real(source, k=k, **options)
    results, widths = {}, set()
    every = np.arange(split.data["test_target_matrix"].shape[0])
    seen_counts = np.diff(split.data["test_source_matrix"].indptr)
    lighter = every[seen_counts < seen_counts.max()][::2]  # a subset without the phase's heaviest users
    for batch, rows in ((1, None), (7, None), (128, None), (7, lighter)):
        asked.clear()
        results[batch, rows is None] = evaluate.evaluate_phase(
            model, split, "test", family="matrix", protocol=replaced(protocol, eval_batch_size=batch),
            exclude_seen=True, rows=rows, targets="window")
        widths |= set(asked)
    assert len(widths) == 1  # one length, whatever the batch size or the rows scored
    heaviest = int(seen_counts.max())
    assert widths == {min(max(protocol.cutoffs) + heaviest, len(split.data["train_item_ids"]))}
    metric = protocol.primary_metric
    full = results[128, True]
    for (batch, all_rows), result in results.items():
        if all_rows:
            assert np.array_equal(result.per_user[metric], full.per_user[metric])
        else:  # a subset scores its users as the full evaluation did
            index = {u: i for i, u in enumerate(np.asarray(full.sample_ids).astype(str))}
            at = [index[u] for u in np.asarray(result.sample_ids).astype(str)]
            assert np.array_equal(result.per_user[metric], full.per_user[metric][at])



def test_all_cold_histories_are_counted_over_the_users_scored(workspace):
    # DECISIONS §40: users whose history holds no training-catalogue item reach a matrix model empty
    from dataclasses import replace

    from scipy.sparse import csr_matrix

    from seqrec_eval.analysis import count_cold_histories
    from seqrec_eval.evaluate import cold_histories, phase_inputs, scored_users

    root, _ = workspace
    protocol = load_protocol(root / "protocol.toml")
    split = load_split(root / "work", "synth")
    n_train = len(split.data["train_item_ids"])
    source = split.data["val_source_matrix"].tolil()
    assert source.shape[1] > n_train  # the validation catalogue holds items training lacks
    scored = scored_users(split, "val", protocol, exclude_seen=False, rows=split.val_rows)
    cold_row, empty_row = int(scored[0]), int(scored[1])
    source.rows[cold_row], source.data[cold_row] = [n_train, n_train + 1], [1.0, 1.0]  # only items training lacks
    source.rows[empty_row], source.data[empty_row] = [], []
    changed = replace(split, data={**split.data, "val_source_matrix": csr_matrix(source)})

    record = count_cold_histories(protocol, changed, final_split(protocol, changed))
    model = model_spec("ease").build({"l2": 10.0}, n_items=n_train, device="cpu", seed=0)
    model.fit(changed.data["x_train"], item_ids=changed.data["train_item_ids"])
    for exclude_seen in (False, True):
        users = scored_users(changed, "val", protocol, exclude_seen=exclude_seen, rows=changed.val_rows)
        result = evaluate_phase(model, changed, "val", family="matrix", protocol=protocol,
                                exclude_seen=exclude_seen, rows=changed.val_rows)
        assert users.size == result.n_scored_rows  # the users the metrics are computed over
    # what the matrix model reads: its input projected onto the training catalogue, empty for exactly those users
    _, read = phase_inputs(model, changed, "val", "matrix")
    reads_nothing = np.diff(read.indptr) == 0
    assert np.array_equal(cold_histories(changed, "val"), reads_nothing)
    users = scored_users(changed, "val", protocol, exclude_seen=False, rows=changed.val_rows)
    assert record["phases"]["val"] == {"trained_on": "train", "catalogue": n_train, "scored_users": int(users.size),
                                       "all_cold": int(reads_nothing[users].sum()), "empty": 1}
    assert reads_nothing[[cold_row, empty_row]].all() and reads_nothing[users].sum() >= 2
    # test: as the refitted finals are scored, every user (the synthetic data has no all-cold test history)
    tested = final_split(protocol, split)
    assert record["phases"]["test"]["trained_on"] == "train+val"
    assert record["phases"]["test"]["scored_users"] == scored_users(tested, "test", protocol, exclude_seen=False).size


def test_the_reports_give_the_cold_history_count(workspace, tmp_path):
    import shutil

    from seqrec_eval.analysis import cold_path

    root, common = workspace
    protocol = load_protocol(root / "protocol.toml")
    record = json.loads(cold_path(protocol, root / "work", "synth").read_text())  # written by `analyse`
    test = record["phases"]["test"]
    line = f"**All-cold histories** (DECISIONS §40): {test['all_cold']:,} of the {test['scored_users']:,} users"
    assert line in (root / "work" / "reports" / "stage1.md").read_text()
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    cold_path(protocol, work, "synth").unlink()
    local = ["--protocol", str(root / "protocol.toml"), "--work-dir", str(work)]
    assert cli.main(local + ["report"]) == 0
    assert "All-cold histories not counted yet" in (work / "reports" / "stage1.md").read_text()
    assert cli.main(local + ["analyse", "--sweep", "none"]) == 0  # counts what is missing, redoes nothing else
    assert cli.main(local + ["analysis-report"]) == 0
    assert line in (work / "reports" / "analysis.md").read_text()
    assert json.loads(cold_path(protocol, work, "synth").read_text()) == record
