"""The ablation workflow on the smoke test's synthetic dataset.

Stage 1 runs first, because every sweep refits the configuration it selected
and its reference is stage 1's own final models. Every transform runs once,
end to end, on a level or two; the properties that define each transform are
checked on the transformed splits directly.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from compresso_recsys import builder

from seqrec_eval import cli
from seqrec_eval.ablation_report import find_knee, noninferiority_p, noninferiority_power
from compresso_recsys import ItemSequences

from seqrec_eval.ablations import (KEPT, REFERENCE, TRANSFORMS, _first_occurrence, _stratified_sample,
                                   ablation_fingerprint, ablation_root, apply_condition, condition_fingerprint,
                                   count_matrix, data_seeds, keep_events, plan_ablation)
from seqrec_eval.protocol import ProtocolError, load_protocol
from seqrec_eval.runner import execute, plan_trials
from seqrec_eval.splits import final_split, load_split, prepare_split, split_dir
from seqrec_eval.timestamps import TIMESTAMP_KEYS
from test_smoke import DEVICE, PERIOD_DAYS, PROTOCOL as STAGE1, Synthetic

LEVELS = [1, 2, 5]
SWEEP = """
[ablations.history]
transform = "history_length"
levels = [1, 2, 5]

[ablations.history_inference]
transform = "history_length"
scope = "inference"
levels = [5]

[ablations.density]
transform = "density"
levels = [0.5]

[ablations.repeats]
transform = "repeat_removal"
levels = [1.0]

[ablations.shuffle]
transform = "shuffle"
levels = [3, "all"]

[ablations.catalogue]
transform = "catalogue"
levels = [45]

[ablations.catalogue_strata]
transform = "catalogue"
levels = [0.75]
options = { strategy = "stratified", strata = 3 }

[repeat_strata]
history_bins = [1, 30]
repeat_bins = [0.0, 0.1, 1.0]
min_users = 5
n_resamples = 99
"""
PROTOCOL = STAGE1 + SWEEP


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("ablations")
    (root / "protocol.toml").write_text(PROTOCOL)
    patch = pytest.MonkeyPatch()
    patch.setitem(builder.DATASETS, "synthetic", builder.DatasetSpec(
        Synthetic, str(root / "unused.zip"), seed=0, val_users=5, test_users=5,
        min_value_to_keep=4.0, min_entity_text_words=0, temporal_period_hours=PERIOD_DAYS * 24,
    ))
    common = ["--protocol", str(root / "protocol.toml"), "--work-dir", str(root / "work")]
    assert cli.main(common + ["prepare", "--data-dir", str(root / "data"), "--quiet"]) == 0
    assert cli.main(common + ["search", "--device", DEVICE]) == 0
    assert cli.main(common + ["final", "--device", DEVICE]) == 0
    assert cli.main(common + ["ablate", "--device", DEVICE]) == 0
    assert cli.main(common + ["analyse"]) == 0
    assert cli.main(common + ["ablation-report"]) == 0
    assert cli.main(common + ["analysis-report"]) == 0
    assert cli.main(common + ["repeat-strata"]) == 0
    yield root, common, load_protocol(root / "protocol.toml")
    patch.undo()


def _same(a, b) -> bool:
    return a.shape == b.shape and (a != b).nnz == 0


def test_keeping_every_event_rebuilds_the_split_exactly(workspace):
    root, _, protocol = workspace
    data = load_split(root / "work", "synth").data
    everything = {"train": np.ones(data["x_train_sequences"].values.size, bool),
                  "val": np.ones(data["val_source_sequences"].values.size, bool),
                  "test": np.ones(data["test_source_sequences"].values.size, bool)}
    rebuilt = keep_events(data, everything, value=protocol.dataset("synth").set_all_values_to)
    # Items repeat across the train source/target boundary, so a plain count
    # over the window differs from x_train: it only matches if the boundary and
    # the maximum are rebuilt as the builder does.
    assert not _same(count_matrix(data["x_train_sequences"], 1.0), data["x_train"])
    assert data["val_source_matrix"].max() > 1  # and phase sources are counts
    for key in ("x_train", "train_source_matrix", "train_target_matrix", "val_source_matrix", "test_source_matrix"):
        assert _same(rebuilt[key], data[key]), key
    for key in ("x_train_sequences", "train_source_sequences", "val_source_sequences", "test_source_sequences"):
        assert np.array_equal(rebuilt[key].values, data[key].values)
        assert np.array_equal(rebuilt[key].indptr, data[key].indptr)


def test_history_length_trains_on_one_event_more_than_it_serves(workspace):
    # a training history of n + 1 events teaches contexts of n items, and the model is then shown n
    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    before = {key: split.data[key] for key in ("test_target_matrix", "test_source_sequences")}
    condition = apply_condition(protocol, split, "history", 5)
    for key, kept in (("x_train_sequences", 6), ("val_source_sequences", 5), ("test_source_sequences", 5)):
        original, truncated = split.data[key], condition.data[key]
        assert truncated.n_rows == original.n_rows
        for row in range(original.n_rows):
            assert np.array_equal(truncated.row(row), original.row(row)[-kept:])
    assert condition.data["test_target_matrix"] is split.data["test_target_matrix"]
    assert split.data["test_source_sequences"] is before["test_source_sequences"]  # the input is untouched
    assert condition.data["x_train"].sum() < split.data["x_train"].sum()


def test_every_condition_scores_the_same_users_and_skips_validation(workspace):
    root, _, protocol = workspace
    work = root / "work"
    rows = np.load(ablation_root(protocol, work, "history", "synth") / "test_rows.npy")
    ids = None
    for model in ("popularity", "elsa", "gru", "sasrec"):
        plans = plan_ablation(protocol, work, "history", "synth", model)
        assert list(plans) == [REFERENCE, "1", "2", "5"]  # level 1 trains: pairs of (one item -> next)
        for name, specs in plans.items():
            assert len(specs) == len(protocol.seeds)
            for spec in specs:
                directory = spec.directory(work)
                record = json.loads((directory / "done.json").read_text())
                assert record["status"] == "done"
                assert record["trained_on"] == "train+val"  # conditions transform the refitted training data
                assert "val" not in record and not (directory / "val.json").exists()
                assert not (directory / "model.zip").exists()
                scored = np.load(directory / "test.npz")["sample_ids"]
                ids = scored if ids is None else ids
                assert np.array_equal(scored, ids)
                assert json.loads((directory / "test.json").read_text())["metadata"]["rows_sampled"] == rows.size
                assert ("loaded_from" in record) == (name == REFERENCE)
    # EASE was skipped in stage 1, so it has nothing to ablate
    assert plan_ablation(protocol, work, "history", "synth", "ease") == {}


def test_the_logs_name_an_ablation_run_by_its_condition(workspace):
    # its directory is final-seed<s> too, which in a log would read like a stage-1 final
    root, _, protocol = workspace
    history = plan_ablation(protocol, root / "work", "history", "synth", "gru")
    assert history[REFERENCE][0].label == "history full: reference (stage-1 model) seed 0"
    assert history["2"][1].label == "history 2: refit seed 1"
    assert plan_ablation(protocol, root / "work", "history_inference", "synth", "gru")["5"][0].label == \
        "history_inference 5: rescore (stage-1 model) seed 0"
    assert plan_ablation(protocol, root / "work", "density", "synth", "gru")["0.5/seed1"][0].label == \
        "density 0.5/seed1: refit seed 1"
    assert plan_trials(protocol, "synth", "gru")[0].label == "trial-000"


def test_the_reference_is_the_stage1_model(workspace):
    root, _, protocol = workspace
    work = root / "work"
    rows = np.load(ablation_root(protocol, work, "history", "synth") / "test_rows.npy")
    split = load_split(work, "synth")
    from seqrec_eval.evaluate import scored_rows
    from seqrec_eval.splits import final_split
    # truncation never empties a history here, so the fixed users are those scored on the full data:
    # everyone whose next item the refitted model can recommend
    assert np.array_equal(rows, scored_rows(final_split(protocol, split), "test", "next"))
    for model in ("popularity", "elsa"):
        for spec in plan_ablation(protocol, work, "history", "synth", model)[REFERENCE]:
            reference = json.loads((spec.directory(work) / "done.json").read_text())["test"]
            stage1 = work / spec.condition["checkpoint"]
            original = json.loads((stage1.parent / "done.json").read_text())["test"]
            assert reference == pytest.approx(original)


def test_the_manipulation_is_measured_for_every_condition(workspace):
    root, _, protocol = workspace
    folder = ablation_root(protocol, root / "work", "history", "synth") / "conditions"
    full = json.loads((folder / f"{REFERENCE}.json").read_text())["characteristics"]
    for level in LEVELS:
        measured = json.loads((folder / f"{level}.json").read_text())["characteristics"]
        assert measured["test"]["history_length"] <= level < full["test"]["history_length"]
        assert measured["train"]["history_length"] <= level + 1
        assert measured["test"]["rows"] == full["test"]["rows"]


def test_rerunning_ablate_is_a_no_op(workspace):
    root, common, _ = workspace
    folder = root / "work" / "ablations"
    stamps = {path: path.stat().st_mtime_ns for path in folder.rglob("*.json")}
    assert cli.main(common + ["ablate", "--device", DEVICE]) == 0
    assert {path: path.stat().st_mtime_ns for path in stamps} == stamps


def test_report_has_every_part(workspace):
    root, _, _ = workspace
    reports = root / "work" / "reports"
    report = (reports / "ablation-history.md").read_text()
    for heading in ("### Manipulation check", "### Test ndcg@5", "### Each level against the full data",
                    "### Knee", "### Gap: sequential − elsa", "### Gap: sequential − best non-sequential (descriptive)",
                    "### Baselines and sequence signal"):
        assert heading in report
    assert "max_history_length" in report
    assert "Sensitivity, not the result fixed in advance" in report  # the knee at other margins
    gap = (reports / "ablation-history-gap.csv").read_text().splitlines()
    assert gap[0].startswith("dataset,condition,model")
    # per condition, reference included: each sequential model (gru, sasrec) against elsa (tested) and against
    # the best non-sequential model (descriptive), and the floor's line
    assert len(gap) - 1 == 5 * (len(LEVELS) + 1)
    assert sum(row.split(",")[2] == "floor" for row in gap[1:]) == len(LEVELS) + 1
    metrics = (reports / "ablation-history-metrics.csv").read_text().splitlines()
    ran = len(load_protocol(root / "protocol.toml").models) - 1  # every model but EASE, which stage 1 skipped
    assert len(metrics) - 1 == ran * (len(LEVELS) + 1) * 2


def test_a_stage1_run_refuses_a_transformed_split(workspace):
    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    condition = apply_condition(protocol, split, "history", 2, test_rows=np.arange(3))
    spec = plan_trials(protocol, "synth", "popularity")[0]
    assert execute(spec, condition, protocol, root / "work" / "elsewhere", device="cpu",
                   log=lambda _: None) == "failed"


def test_fingerprints_follow_what_determines_the_result(workspace, tmp_path):
    _, _, protocol = workspace

    def edited(old: str, new: str):
        path = tmp_path / "protocol.toml"
        text = PROTOCOL.replace(old, new)
        assert text != PROTOCOL
        path.write_text(text)
        return load_protocol(path)

    more_levels = edited("levels = [1, 2, 5]", "levels = [1, 2, 5, 10]")
    assert condition_fingerprint(more_levels, "history", "synth") != condition_fingerprint(protocol, "history", "synth")
    fewer_models = edited('transform = "history_length"', 'transform = "history_length"\nmodels = ["gru"]')
    assert condition_fingerprint(fewer_models, "history", "synth") == condition_fingerprint(protocol, "history", "synth")
    tolerance = edited('transform = "history_length"', 'transform = "history_length"\nmanipulation_tolerance = 0.2')
    assert condition_fingerprint(tolerance, "history", "synth") == condition_fingerprint(protocol, "history", "synth")
    # which users are scored, and on what, keys the cached fixed users and profiles (H04)
    from seqrec_eval.analysis import profile_path
    for key, value in (('targets = "next"', 'targets = "window"'), ("refit = true", "refit = false")):
        other = edited(key, value)
        assert condition_fingerprint(other, "history", "synth") != condition_fingerprint(protocol, "history", "synth")
        assert profile_path(other, tmp_path, "synth") != profile_path(protocol, tmp_path, "synth")
    gru = edited("hidden_dim = { choice = [8, 16] }", "hidden_dim = { choice = [8, 32] }")
    assert ablation_fingerprint(gru, "history", "synth", "gru") != ablation_fingerprint(protocol, "history", "synth", "gru")
    assert ablation_fingerprint(gru, "history", "synth", "sasrec") == ablation_fingerprint(protocol, "history", "synth", "sasrec")


def test_a_change_to_the_baselines_code_starts_their_analysis_over(workspace, monkeypatch):
    # a baseline's results depend on its code, not only on its settings: the fixed tie order (2026-09-30)
    # changed what they recommend with nothing in the protocol changed
    import seqrec_eval.protocol as protocol_module
    from seqrec_eval.analysis import _baseline_dir

    root, _, protocol = workspace
    before = {name: _baseline_dir(protocol, root / "work", "synth", name) for name in protocol.baselines}
    monkeypatch.setattr(protocol_module, "BASELINE_VERSION", protocol_module.BASELINE_VERSION + 1)
    for name, directory in before.items():
        assert _baseline_dir(protocol, root / "work", "synth", name) != directory


@pytest.mark.parametrize("levels", ["[0, 5]", "[2.5]", '["full"]', "[]", "[5, 5]"])
def test_bad_levels_are_refused(tmp_path, levels):
    path = tmp_path / "protocol.toml"
    path.write_text(PROTOCOL.replace("levels = [1, 2, 5]", f"levels = {levels}"))
    with pytest.raises(ProtocolError):
        cli.main(["--protocol", str(path), "--work-dir", str(tmp_path / "work"), "plan"])


@pytest.mark.parametrize("margin", ["0", "1.5", "-0.1", "true"])
def test_bad_knee_margins_are_refused(tmp_path, margin):
    path = tmp_path / "protocol.toml"
    path.write_text(PROTOCOL.replace("levels = [1, 2, 5]", f"levels = [1, 2, 5]\nknee_margin = {margin}"))
    with pytest.raises(ProtocolError):
        cli.main(["--protocol", str(path), "--work-dir", str(tmp_path / "work"), "plan"])


def _rng(seed: int = 0):
    return lambda label: np.random.default_rng([seed, hash(label) % 2**32])


def test_noninferiority_rejects_only_a_loss_smaller_than_the_margin():
    noise = np.random.default_rng(1).normal(0.0, 0.05, 20_000)
    assert noninferiority_p(noise - 0.001, 0.01, rng=np.random.default_rng(2), n_resamples=999) < 0.01
    assert noninferiority_p(noise - 0.02, 0.01, rng=np.random.default_rng(2), n_resamples=999) > 0.5


def _curve(n_users: int, means: dict[str, float], seed: int = 0) -> dict[str, np.ndarray]:
    """Per-user values around ``means``: a shared user effect plus independent noise."""
    rng = np.random.default_rng(seed)
    user = rng.uniform(0.0, 0.04, n_users)  # best mean ≈ 0.12, so δ at 2% ≈ 0.0024
    return {label: user + mean + rng.normal(0.0, 0.05, n_users) for label, mean in means.items()}


POSITION = {"full": math.inf, "50": 50, "20": 20, "10": 10, "5": 5, "2": 2}


def test_the_knee_is_where_the_flat_curve_drops():
    # flat down to 10, then a 10% drop at 5 and more at 2
    per_user = _curve(20_000, {"full": 0.1, "50": 0.1, "20": 0.1, "10": 0.1, "5": 0.09, "2": 0.07})
    knee = find_knee(per_user, POSITION, margin_fraction=0.02, alpha=0.05, rng=_rng())
    assert knee.knee == "10" and knee.stopped_at == "5"


def test_a_lucky_short_level_below_a_failure_is_not_the_knee():
    # 2 happens to match the best, but 5 is clearly worse: the curve is not flat from 2 upward
    per_user = _curve(20_000, {"full": 0.1, "50": 0.1, "20": 0.1, "10": 0.1, "5": 0.08, "2": 0.1})
    knee = find_knee(per_user, POSITION, margin_fraction=0.02, alpha=0.05, rng=_rng())
    assert knee.knee == "10"


def test_a_level_that_only_looks_best_is_not_the_knee_when_nothing_was_shown():
    # N10: level 2 scores highest, but the first level tested (50) is clearly worse than the full data.
    # Nothing was shown, so the knee is the full data; it used to start at the best-looking level and stay there.
    per_user = _curve(20_000, {"full": 0.1, "50": 0.08, "20": 0.1, "10": 0.1, "5": 0.1, "2": 0.13})
    knee = find_knee(per_user, POSITION, margin_fraction=0.02, alpha=0.05, rng=_rng())
    assert (knee.knee, knee.stopped_at) == ("full", "50")
    assert [label for label, _, _ in knee.tested] == ["50"]  # and it stopped there


def test_levels_are_compared_with_the_full_data_not_with_the_best_looking_level():
    # 20 is far better than everything else; against "the best" every other level would fail, against the full
    # data they all pass
    per_user = _curve(20_000, {"full": 0.1, "50": 0.1, "20": 0.2, "10": 0.1, "5": 0.1, "2": 0.1})
    knee = find_knee(per_user, POSITION, margin_fraction=0.02, alpha=0.05, rng=_rng())
    assert knee.reference == "full" and knee.knee == "2" and knee.stopped_at is None
    assert knee.margin == pytest.approx(0.02 * per_user["full"].mean())
    for label, difference, _ in knee.tested:
        assert difference == pytest.approx((per_user[label] - per_user["full"]).mean())


def test_the_power_formula_matches_how_often_the_test_shows_no_loss():
    # a level that loses nothing: n = 2,000 users, spread 0.05, margin 0.0035
    rng = np.random.default_rng(3)
    margin, spread, n = 0.0035, 0.05, 2_000
    predicted = noninferiority_power(rng.normal(0.0, spread, n), margin, alpha=0.05)
    shown = np.mean([noninferiority_p(rng.normal(0.0, spread, n), margin, rng=rng, n_resamples=499) <= 0.05
                     for _ in range(300)])
    assert predicted == pytest.approx(0.93, abs=0.03)       # Φ(0.0035·√2000/0.05 − 1.645)
    assert shown == pytest.approx(predicted, abs=0.06)
    # more users, a wider margin or less spread all raise it; at ML-20M's size a 2% margin had almost none
    small = noninferiority_power(rng.normal(0.0, 0.09, 2_942), 0.001, alpha=0.05)
    wide = noninferiority_power(rng.normal(0.0, 0.09, 2_942), 0.005, alpha=0.05)
    assert small < 0.2 < 0.8 < wide
    knee = find_knee(_curve(20_000, {label: 0.1 for label in POSITION}), POSITION, margin_fraction=0.02,
                     alpha=0.05, rng=_rng())
    assert set(knee.power) == set(POSITION) - {"full"} and all(0 < value <= 1 for value in knee.power.values())


def test_fewer_users_cannot_pull_the_knee_lower():
    means = {"full": 0.1, "50": 0.1, "20": 0.1, "10": 0.1, "5": 0.099, "2": 0.095}
    many = find_knee(_curve(50_000, means), POSITION, margin_fraction=0.02, alpha=0.05, rng=_rng())
    few = find_knee(_curve(200, means), POSITION, margin_fraction=0.02, alpha=0.05, rng=_rng())
    # "the interval contains zero" would accept every level with 200 users; this rule cannot
    assert POSITION[few.knee] >= POSITION[many.knee]
    assert many.knee == "5"


# ---------------------------------------------------------------------------
# the transforms, one property at a time
# ---------------------------------------------------------------------------

def _rows(sequences):
    return [sequences.row(i) for i in range(sequences.n_rows)]


def _is_subsequence(short, long) -> bool:
    it = iter(long.tolist())
    return all(any(x == y for y in it) for x in short.tolist())


SEQUENCES = ("x_train_sequences", "val_source_sequences", "test_source_sequences")


def test_inference_scope_leaves_training_alone_and_rescores(workspace):
    root, _, protocol = workspace
    work = root / "work"
    split = load_split(work, "synth")
    condition = apply_condition(protocol, split, "history_inference", 5)
    for key in ("x_train", "x_train_sequences", "train_source_matrix"):
        assert condition.data[key] is split.data[key]
    assert max(condition.data["test_source_sequences"].row_lengths) <= 5
    for model in ("popularity", "gru"):
        for spec in plan_ablation(protocol, work, "history_inference", "synth", model)["5"]:
            assert spec.kind == "rescore"
            record = json.loads((spec.directory(work) / "done.json").read_text())
            assert "loaded_from" in record and "fit_seconds" not in record


def test_density_thins_every_history_in_order_and_keeps_the_catalogue(workspace):
    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    first, second = (apply_condition(protocol, split, "density", 0.5, seed) for seed in protocol.seeds)
    for key in SEQUENCES:
        for original, thinned in zip(_rows(split.data[key]), _rows(first.data[key])):
            assert _is_subsequence(thinned, original)
            if key != "x_train_sequences":  # training may keep one extra event per rescued item
                assert thinned.size == max(1, round(0.5 * original.size))
    before = np.bincount(split.data["x_train_sequences"].values) > 0
    after = np.bincount(first.data["x_train_sequences"].values, minlength=before.size) > 0
    assert np.array_equal(before, after)
    assert not np.array_equal(first.data["test_source_sequences"].values, second.data["test_source_sequences"].values)
    assert data_seeds(protocol, root / "work", "density", "synth") == list(protocol.seeds)


def test_removing_every_repeat_keeps_which_items_each_history_holds(workspace):
    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    condition = apply_condition(protocol, split, "repeats", 1.0, protocol.seeds[0])
    for key in SEQUENCES:
        assert _first_occurrence(condition.data[key]).all()
        for original, kept in zip(_rows(split.data[key]), _rows(condition.data[key])):
            assert set(kept.tolist()) == set(original.tolist())
    source = split.data["test_source_matrix"]
    assert _same(condition.data["test_source_matrix"] > 0, source > 0)  # the pairs survive; only counts drop
    assert condition.data["test_source_matrix"].sum() < source.sum()


def test_shuffling_changes_only_order_within_blocks(workspace):
    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    condition = apply_condition(protocol, split, "shuffle", 3, protocol.seeds[0])
    changed = 0
    for key in SEQUENCES:
        for original, shuffled in zip(_rows(split.data[key]), _rows(condition.data[key])):
            for start in range(0, original.size, 3):
                assert sorted(shuffled[start:start + 3].tolist()) == sorted(original[start:start + 3].tolist())
            changed += not np.array_equal(original, shuffled)
    assert changed > 0
    for key in ("x_train", "val_source_matrix", "test_source_matrix"):
        assert condition.data[key] is split.data[key]


def test_shuffle_leaves_the_matrix_models_as_they_were(workspace):
    model = "popularity"
    root, _, protocol = workspace
    work = root / "work"
    plans = plan_ablation(protocol, work, "shuffle", "synth", model)
    reference = {spec.seed: json.loads((spec.directory(work) / "done.json").read_text())["test"]
                 for spec in plans[REFERENCE]}
    for name in ("all/seed0", "all/seed1"):
        for spec in plans[name]:
            refit = json.loads((spec.directory(work) / "done.json").read_text())["test"]
            assert refit == pytest.approx(reference[spec.seed])


def test_the_top_catalogue_keeps_the_most_popular_items_everywhere(workspace):
    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    condition = apply_condition(protocol, split, "catalogue", 45)
    data, reduced = split.data, condition.data
    counts = np.bincount(data["x_train_sequences"].values, minlength=len(data["train_item_ids"]))
    kept = np.sort(np.argsort(-counts, kind="stable")[:45])
    assert np.array_equal(reduced["train_item_ids"], data["train_item_ids"][kept])
    for phase in ("val", "test"):
        ids = reduced[f"{phase}_item_ids"]
        assert np.array_equal(ids[:45], reduced["train_item_ids"])  # the adapter needs the training prefix
        assert reduced[f"{phase}_target_matrix"].shape[1] == len(ids)
        assert reduced[f"{phase}_source_sequences"].n_items == len(ids)
    # every surviving target and history event names a kept item, and a removed item's targets are gone
    removed = np.setdiff1d(np.arange(len(data["train_item_ids"])), kept)
    assert data["test_target_matrix"][:, removed].nnz > 0
    kept_ids = set(reduced["train_item_ids"].tolist())
    test_ids = reduced["test_item_ids"]
    targets = reduced["test_target_matrix"]
    assert set(test_ids[targets.indices].tolist()) <= kept_ids | set(test_ids[45:].tolist())
    assert targets.nnz == data["test_target_matrix"].nnz - data["test_target_matrix"][:, removed].nnz


def test_the_full_catalogue_rebuilds_the_split_exactly(workspace):
    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    n = len(split.data["train_item_ids"])
    data = TRANSFORMS["catalogue"].apply(split.data, n, rng=None, options={}, value=1.0,
                                         phases=("train", "val", "test"))
    for key in ("x_train", "train_source_matrix", "train_target_matrix", "val_source_matrix",
                "test_source_matrix", "val_target_matrix", "test_target_matrix"):
        assert _same(data[key], split.data[key]), key
    for key in SEQUENCES:
        assert np.array_equal(data[key].values, split.data[key].values)


def test_the_stratified_catalogue_samples_each_stratum_in_proportion():
    counts = np.arange(100)[::-1]  # item i has popularity 99 - i, so the strata are 0-24, 25-49, ...
    chosen = _stratified_sample(counts, 40, 4, np.random.default_rng(0))
    assert chosen.size == np.unique(chosen).size == 40
    assert np.bincount(chosen // 25).tolist() == [10, 10, 10, 10]


def test_block_shuffle_and_edits_on_a_hand_made_history():
    from seqrec_eval.ablations import _edits

    before = ItemSequences.from_rows([[0, 1, 2, 3, 4, 5], [2, 2]], n_items=6)
    truncated = ItemSequences.from_rows([[4, 5], [2, 2]], n_items=6)
    kept = np.array([False, False, False, False, True, True, True, True])
    edits = _edits(before, truncated, np.arange(6), np.arange(6), kept, None)
    assert edits["rows_changed"] == 0.5
    assert edits["span_kept"] == pytest.approx((2 / 6 + 1.0) / 2)


def test_the_report_pairs_a_levels_seeds_with_the_full_datas(workspace):
    # review A1: seed s of an inference level is the full data's model s, rescored; the report's seed term must be
    # the spread of the per-seed differences, not the sum of both sides' spreads
    root, _, protocol = workspace
    work = root / "work"
    plans = plan_ablation(protocol, work, "history_inference", "synth", "gru")
    mean = lambda spec: json.loads((spec.directory(work) / "test.json").read_text())["metrics"]["ndcg@5"]  # noqa: E731
    level, full = [mean(s) for s in plans["5"]], [mean(s) for s in plans[REFERENCE]]
    paired = math.sqrt(np.var(np.subtract(level, full), ddof=1) / len(level))
    unpaired = math.sqrt((np.var(level, ddof=1) + np.var(full, ddof=1)) / len(level))
    assert abs(paired - unpaired) >= 0.0001  # the two readings differ here, so the table tells them apart
    report = (work / "reports" / "ablation-history_inference.md").read_text()
    row = next(line for line in report.splitlines() if line.startswith("| gru | 5 |"))
    assert row.split(" | ")[4] == f"{paired:.4f}"


def test_the_transforms_declare_what_they_move_by_construction():
    # review C1
    assert "test catalogue" in TRANSFORMS["density"].expected({})
    assert "catalogue" in TRANSFORMS["density"].expected({"keep_catalogue": False})
    assert "popularity_gini" in TRANSFORMS["repeat_removal"].expected({})
    assert {"catalogue", "popularity_gini"} <= set(TRANSFORMS["history_length"].expected({}))


def test_a_characteristic_can_be_declared_for_one_part_only():
    # density thins test inputs (fewer distinct items there) while keep_catalogue holds the training catalogue:
    # "test catalogue" declares the first and still flags the second (review C1)
    from seqrec_eval.ablations import manipulation_check

    base = {"history_length": 10.0, "catalogue": 100.0, "density": 0.1, "popularity_gini": 0.5, "repeat_rate": 0.1}
    reference = {"train": dict(base), "test": dict(base)}
    measured = {"train": {**base, "catalogue": 80.0}, "test": {**base, "catalogue": 80.0}}
    check = manipulation_check(reference, measured, "density", 0.05, ("test catalogue",))
    assert check["moved"] == ["train catalogue"]
    assert manipulation_check(reference, measured, "density", 0.05, ("catalogue",))["moved"] == []


def _copied(workspace, tmp_path):
    import shutil

    root, _, protocol = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    return protocol, work


def test_failed_and_unfinished_ablation_runs_are_listed_not_dropped(workspace, tmp_path):
    # review A5: a failed level used to leave the comparisons and the knee without a word
    from seqrec_eval.ablation_report import build_ablation_report

    protocol, work = _copied(workspace, tmp_path)
    plans = plan_ablation(protocol, work, "history", "synth", "gru")
    failed, pending = plans["2"][0].directory(work), plans["5"][1].directory(work)
    (failed / "done.json").unlink()
    (failed / "failed.json").write_text(json.dumps({"error": "RuntimeError: CUDA out of memory"}))
    (pending / "done.json").unlink()
    report = build_ablation_report(protocol, work, "history", ["synth"], list(protocol.models))["markdown"]
    assert "⛔ **gru**: 1 run(s) failed (2 seed 0: RuntimeError: CUDA out of memory)" in report
    assert "**gru**: 1 run(s) not finished yet, at 5" in report
    assert "No knee for **gru** (incomplete: 2, 5)" in report
    knees = report.split("### Knee")[1].split("Sensitivity")[0]
    assert "| sasrec | " in knees and "| gru | " not in knees  # the other models keep their knee


def test_an_ablation_run_made_under_an_old_selection_is_left_out_of_the_report(workspace, tmp_path):
    # review N33: the ablation report loaded such runs silently, into the comparisons, the gap and the knee
    from seqrec_eval.ablation_report import build_ablation_report

    protocol, work = _copied(workspace, tmp_path)
    run = plan_ablation(protocol, work, "history", "synth", "gru")["2"][0].directory(work)
    made = json.loads((run / "spec.json").read_text())
    (run / "spec.json").write_text(json.dumps({**made, "source_trial": 99}))
    built = build_ablation_report(protocol, work, "history", ["synth"], list(protocol.models))
    report = built["markdown"]
    assert "⛔ **gru**: 1 run(s) were made with another configuration than the one selected now (2 seed 0)" in report
    assert "No knee for **gru** (incomplete: 2)" in report
    assert "**gru**: 1 run(s) not finished yet" not in report  # not passed off as pending either
    knees = report.split("### Knee")[1].split("Sensitivity")[0]
    assert "| sasrec | " in knees and "| gru | " not in knees


def test_an_ablation_run_built_on_an_old_selection_is_refused(workspace, tmp_path):
    # review A3: a reference reloads stage 1's model, so it must belong to the selection its conditions refit
    protocol, work = _copied(workspace, tmp_path)
    split = final_split(protocol, load_split(work, "synth"))
    reference = plan_ablation(protocol, work, "history", "synth", "gru")[REFERENCE][0]
    stage1 = (work / reference.condition["checkpoint"]).parent
    made = json.loads((stage1 / "spec.json").read_text())
    (stage1 / "spec.json").write_text(json.dumps({**made, "source_trial": 99}))
    (reference.directory(work) / "done.json").unlink()
    condition = __import__("seqrec_eval.ablations", fromlist=["build_reference"]).build_reference(
        protocol, work, split, "history")
    messages = []
    assert execute(reference, condition, protocol, work, device="cpu", log=messages.append) == "stale-selection"
    assert "the stage-1 final it reloads made with trial 99's configuration" in messages[0]


def test_a_random_sweeps_floor_waits_for_every_subsample(workspace, tmp_path):
    # review C2: after --add-seeds without analyse, the floor would average over fewer subsamples than the models
    from seqrec_eval.ablation_report import build_ablation_report

    protocol, work = _copied(workspace, tmp_path)
    analysis = ablation_root(protocol, work, "density", "synth") / "analysis"
    removed = [path for path in analysis.glob("markov/*/0.5/seed1/test.*")]
    assert removed
    for path in removed:
        path.unlink()
    report = build_ablation_report(protocol, work, "density", ["synth"], list(protocol.models))["markdown"]
    assert "partial 1/2" in report
    assert "Not every subsample analysed yet, so left out: 0.5 (1 of 2 subsamples)" in report


def test_levels_too_small_to_claim_anything_say_so_in_place_of_a_verdict(workspace):
    # review C3: the synthetic data has 79 test users, below min_level_users, so no level gets a verdict
    root, _, _ = workspace
    report = (root / "work" / "reports" / "ablation-history.md").read_text()
    assert "— (descriptive)" in report and "| yes |" not in report and "| no |" not in report
    assert "(descriptive)" in report.split("### Knee")[1].split("###")[0]


def test_a_random_catalogue_shows_its_users_per_seed(workspace):
    root, _, protocol = workspace
    work = root / "work"
    plans = plan_ablation(protocol, work, "catalogue_strata", "synth", "gru")
    counts = [json.loads((spec.directory(work) / "test.json").read_text())["n_rows"]
              for name, specs in plans.items() if name != REFERENCE for spec in specs]
    report = (work / "reports" / "ablation-catalogue_strata.md").read_text()
    if min(counts) != max(counts):  # seeds of a random catalogue score different users
        assert f"({min(counts):,}–{max(counts):,} per seed)" in report


def test_the_manipulation_check_flags_only_undeclared_moves(workspace):
    root, _, protocol = workspace
    report = (root / "work" / "reports" / "ablation-history.md").read_text()
    flagged = [line for line in report.splitlines() if line.startswith("- **") and "also moved" in line]
    for line in flagged:
        for declared in ("history_length", "density", "repeat_rate", "catalogue", "popularity_gini"):
            assert declared not in line
    assert "declares that it also moves density, repeat_rate, catalogue, popularity_gini" in report
    shuffle = (root / "work" / "reports" / "ablation-shuffle.md").read_text()
    assert "should move none of them" in shuffle and "also moved" not in shuffle


def test_every_sweep_has_a_report(workspace):
    root, _, protocol = workspace
    reports = root / "work" / "reports"
    for sweep in protocol.ablations:
        text = (reports / f"ablation-{sweep}.md").read_text()
        assert "### Manipulation check" in text
        has_knee = TRANSFORMS[protocol.ablation(sweep).transform].position is not None
        assert ("### Knee" in text) == has_knee, sweep
    assert "Scope **inference**" in (reports / "ablation-history_inference.md").read_text()
    assert "targets of removed items" in (reports / "ablation-catalogue.md").read_text()
    strata = (reports / "repeat-strata.md").read_text()
    assert "Test users per cell" in strata and "bootstrap" in strata
    rows = (reports / "repeat-strata.csv").read_text().splitlines()
    assert len(rows) - 1 == 2 * 2  # two history bins × two repeat bins


def test_an_omitted_scope_is_the_default_scope(workspace, tmp_path):
    _, _, protocol = workspace
    path = tmp_path / "protocol.toml"
    path.write_text(PROTOCOL.replace("levels = [1, 2, 5]", 'levels = [1, 2, 5]\nscope = "all"'))
    assert condition_fingerprint(load_protocol(path), "history", "synth") == \
        condition_fingerprint(protocol, "history", "synth")


@pytest.mark.parametrize("section", [
    'transform = "catalogue"\nscope = "inference"\nlevels = [10]',
    'transform = "catalogue"\nlevels = [10, 0.5]',
    'transform = "density"\nlevels = [1.0]',
    'transform = "shuffle"\nlevels = [1]',
    'transform = "repeat_removal"\nlevels = [0]',
    'transform = "history_length"\nlevels = [5]\nexpected_to_move = ["history_length"]',
    'transform = "history_length"\nlevels = [5]\nexpected_to_move = ["colour"]',
])
def test_bad_sweeps_are_refused(tmp_path, section):
    path = tmp_path / "protocol.toml"
    path.write_text(STAGE1 + "\n[ablations.bad]\n" + section + "\n")
    with pytest.raises(ProtocolError):
        cli.main(["--protocol", str(path), "--work-dir", str(tmp_path / "work"), "plan"])


# ---------------------------------------------------------------------------
# timestamps
# ---------------------------------------------------------------------------

def _pairs(sequences, times):
    """Per row, the sorted (item, time) pairs."""
    return [sorted(zip(sequences.row(i).tolist(), times[sequences.indptr[i]:sequences.indptr[i + 1]].tolist()))
            for i in range(sequences.n_rows)]


def test_every_history_has_its_times_in_order(workspace):
    root, _, _ = workspace
    split = load_split(root / "work", "synth")
    assert split.info["timestamps"]["aligned"] == sorted(TIMESTAMP_KEYS.values())
    for view, key in TIMESTAMP_KEYS.items():
        sequences, times = split.data[view], split.data[key]
        assert times.shape == sequences.values.shape
        within_rows = np.diff(times)[np.diff(np.repeat(np.arange(sequences.n_rows), sequences.row_lengths)) == 0]
        assert (within_rows >= 0).all()
    # the training window's source part is where the train-stage source's times come from
    window, source = split.data["x_train_sequences"], split.data["train_source_sequences"]
    first = window.indptr[0]
    assert split.data["train_source_timestamps"][: source.row_lengths[0]].tolist() == \
        split.data["x_train_timestamps"][first: first + source.row_lengths[0]].tolist()


@pytest.mark.parametrize("sweep, level, stochastic", [
    ("history", 5, False), ("density", 0.5, True), ("repeats", 1.0, True), ("shuffle", "all", True),
    ("catalogue", 45, False),
])
def test_every_transform_keeps_each_time_with_its_event(workspace, sweep, level, stochastic):
    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    condition = apply_condition(protocol, split, sweep, level, protocol.seeds[0] if stochastic else None)
    for view, key in TIMESTAMP_KEYS.items():
        sequences, times = condition.data[view], condition.data[key]
        assert times.shape == sequences.values.shape, (sweep, view)
        before = _pairs(split.data[view], split.data[key])
        after = _pairs(sequences, times)
        if sweep == "catalogue":
            # re-addressed items: compare item ids rather than indices
            stage = {"x_train_sequences": "train", "train_source_sequences": "train"}.get(view, view.split("_")[0])
            old_ids, new_ids = split.data[f"{stage}_item_ids"], condition.data[f"{stage}_item_ids"]
            before = [[(old_ids[i], t) for i, t in row] for row in before]
            after = [[(new_ids[i], t) for i, t in row] for row in after]
            kept = set(new_ids.tolist())
            before = [[pair for pair in row if pair[0] in kept] for row in before]
            if len(after) < len(before):  # training users left without events were dropped
                before = [row for row in before if row]
            assert after == before, (sweep, view)
        elif sweep == "shuffle":
            assert after == before, (sweep, view)  # the same events at the same times, in another order
        else:
            for kept, original in zip(after, before):
                assert set(kept) <= set(original), (sweep, view)


def test_the_time_decayed_baseline_refuses_a_split_without_times(workspace):
    from seqrec_eval.analysis import fit_baseline

    root, _, _ = workspace
    split = load_split(root / "work", "synth")
    for key in TIMESTAMP_KEYS.values():
        split.data.pop(key)
    with pytest.raises(ValueError, match="needs the training events' timestamps"):
        fit_baseline("time_popularity", {"half_life_days": 7.0}, split)


def test_prepare_adds_times_to_a_split_prepared_without_them(workspace, tmp_path):
    import shutil

    root, _, protocol = workspace
    work = tmp_path / "work"
    shutil.copytree(split_dir(root / "work", "synth"), split_dir(work, "synth"))
    info_path = split_dir(work, "synth") / "split_info.json"
    info = json.loads(info_path.read_text())
    del info["timestamps"]
    info_path.write_text(json.dumps(info))
    for key in TIMESTAMP_KEYS.values():
        (split_dir(work, "synth") / f"{key}.npy").unlink()

    prepare_split(protocol, "synth", data_dir=root / "data", work_dir=work, show_progress=False)
    again, original = load_split(work, "synth"), load_split(root / "work", "synth")
    for key in TIMESTAMP_KEYS.values():
        assert np.array_equal(again.data[key], original.data[key])
    assert "timestamps" in json.loads(info_path.read_text())


def test_prepare_recomputes_next_targets_of_an_older_definition(workspace, tmp_path):
    # a split prepared before H16's fix holds next targets of the first surviving event; prepare replaces them
    import shutil

    from scipy.sparse import save_npz

    from seqrec_eval.timestamps import NEXT_TARGETS_VERSION

    root, _, protocol = workspace
    work = tmp_path / "work"
    shutil.copytree(split_dir(root / "work", "synth"), split_dir(work, "synth"))
    folder = split_dir(work, "synth")
    info = json.loads((folder / "split_info.json").read_text())
    del info["timestamps"]["next_targets_version"]  # as version 1 wrote it
    (folder / "split_info.json").write_text(json.dumps(info))
    window = load_split(work, "synth").data["test_target_matrix"]
    save_npz(folder / "test_next_target_matrix.npz", window.tocsr())  # stale content, to be replaced

    prepare_split(protocol, "synth", data_dir=root / "data", work_dir=work, show_progress=False)
    again, original = load_split(work, "synth"), load_split(root / "work", "synth")
    assert json.loads((folder / "split_info.json").read_text())["timestamps"]["next_targets_version"] == \
        NEXT_TARGETS_VERSION
    assert (again.data["test_next_target_matrix"] != original.data["test_next_target_matrix"]).nnz == 0


def test_a_next_target_outside_the_models_catalogue_is_not_eligible():
    from scipy.sparse import csr_matrix

    from seqrec_eval.ablations import _eligible_test_rows

    # three items the model knows (0-2), one it does not (3); four users with a history
    targets = csr_matrix(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]], dtype=np.float32))
    data = {"test_next_target_matrix": targets, "test_target_matrix": targets,
            "train_item_ids": np.array(["a", "b", "c"]),
            "test_source_sequences": ItemSequences.from_rows([[0], [1], [2], [0]], n_items=4)}
    # a known next item; an unrecommendable one; a deleted one (empty); a mix, which has a known one
    assert _eligible_test_rows(data, "test_next_target_matrix").tolist() == [True, False, False, True]
    # window targets (the diagnostic) only need a target
    assert _eligible_test_rows(data, "test_target_matrix").tolist() == [True, True, False, True]


# ---------------------------------------------------------------------------
# the analysis: profile, baselines and the floor, sequence signal
# ---------------------------------------------------------------------------

def test_baselines_are_searched_on_validation_as_a_grid_where_they_can_be(workspace):
    from seqrec_eval.analysis import full_results

    root, _, protocol = workspace
    assert {name: b.trials for name, b in protocol.baselines.items()} == \
        {"popularity": 2, "time_popularity": 2, "replay": 2, "markov": 1}
    assert protocol.baseline("replay").grid and not protocol.baseline("time_popularity").grid
    found = full_results(protocol, root / "work", "synth")
    for name, selected in found["selected"].items():
        assert "test" not in selected  # selection saw validation only
        trials = sorted((root / "work").glob(f"analysis/synth/*/baselines/{name}/*/trial-*.json"))
        best = max((json.loads(t.read_text()) for t in trials), key=lambda r: r["val"]["ndcg@5"])
        assert selected["params"] == best["params"]


def test_the_analysis_finds_the_order_the_synthetic_data_was_built_with(workspace):
    from seqrec_eval.analysis import floor_of, full_results

    root, _, protocol = workspace
    found = full_results(protocol, root / "work", "synth")
    markov = found["baselines"]["markov"].metrics["ndcg@5"]
    # each user walks a first-order chain
    assert floor_of(found["baselines"], "ndcg@5", expected=protocol.baselines)[0] == "markov"
    assert found["controls"]["markov_shuffled"].metrics["ndcg@5"] < markov
    assert found["controls"]["markov_backwards"].metrics["ndcg@5"] < markov
    assert found["profile"]["train"]["tie_rate"] == 0.0  # one event a day per user


def test_every_condition_is_analysed_with_the_full_data_settings(workspace):
    from seqrec_eval.analysis import full_results

    root, _, protocol = workspace
    selected = full_results(protocol, root / "work", "synth")["selected"]
    for sweep in protocol.ablations:
        folder = ablation_root(protocol, root / "work", sweep, "synth") / "analysis"
        for name in protocol.baselines:
            records = [json.loads(path.read_text()) for path in folder.glob(f"{name}/*/**/test.done.json")]
            assert records, (sweep, name)
            assert len(list(folder.glob(f"{name}/*"))) == 1  # one fingerprint: one setting for every level


def test_the_reference_condition_is_the_full_data_analysis(workspace):
    from seqrec_eval.analysis import condition_results, full_results

    root, _, protocol = workspace
    full = full_results(protocol, root / "work", "synth")
    at_reference = condition_results(protocol, root / "work", "shuffle", "synth")  # every test user is eligible
    for name, result in {**full["baselines"], **full["controls"]}.items():
        assert at_reference[name][REFERENCE][0][0].metrics == pytest.approx(result.metrics), name


def test_order_blind_baselines_are_unmoved_by_shuffling(workspace):
    from seqrec_eval.analysis import condition_results

    root, _, protocol = workspace
    found = condition_results(protocol, root / "work", "shuffle", "synth")
    for name in ("popularity", "time_popularity"):  # counts, and each event's time, travel with the event
        reference = found[name][REFERENCE][0][0].metrics
        for column in ("3", "all"):
            for result, _ in found[name][column]:
                assert result.metrics == pytest.approx(reference), (name, column)


def test_histories_are_split_by_whether_they_end_in_a_tie():
    from seqrec_eval.ablations import last_pair_tied
    from seqrec_eval.splits import Split

    sequences = ItemSequences.from_rows([[0, 1], [2, 3, 4], [5]], n_items=6)
    data = {"test_source_sequences": sequences, "test_source_timestamps": np.array([1.0, 1.0, 1.0, 2.0, 3.0, 9.0])}
    split = Split("toy", None, data, {}, None)
    assert last_pair_tied(split).tolist() == [True, False, False]
    del data["test_source_timestamps"]
    assert last_pair_tied(split) is None


def test_rerunning_analyse_is_a_no_op(workspace):
    root, common, _ = workspace
    stamps = {path: path.stat().st_mtime_ns for path in (root / "work").rglob("*.json")
              if "analysis" in path.parts}
    assert stamps
    assert cli.main(common + ["analyse"]) == 0
    assert {path: path.stat().st_mtime_ns for path in stamps} == stamps


def test_the_analysis_report_reads_like_the_audit(workspace):
    root, _, _ = workspace
    report = (root / "work" / "reports" / "analysis.md").read_text()
    for heading in ("### Data profile", "### Baselines and the floor", "### Sequence signal"):
        assert heading in report
    assert "markov **(floor)**" in report and "histories ending in a tie" in report


# ---------------------------------------------------------------------------
# exclusion keeps the whole history
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["sequence", "matrix"])
@pytest.mark.parametrize("sweep, level, seed", [("history", 2, None), ("density", 0.5, 0)])
def test_truncating_the_input_does_not_shrink_what_is_excluded(workspace, sweep, level, seed, family):
    """Popularity ignores the history, so with seen items excluded it must score the same at every level.

    The matrix case reads its input projected onto the training catalogue while the seen matrix covers the
    test catalogue, cold items included: the policy's batch check has to map one onto the other (H33).
    """
    from seqrec_eval.analysis import fit_baseline
    from seqrec_eval.ablations import reference_condition
    from seqrec_eval.evaluate import evaluate_phase
    from seqrec_eval.models import model_spec

    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    rows = np.arange(split.data["test_target_matrix"].shape[0])
    full = reference_condition(split, sweep, rows)
    condition = apply_condition(protocol, split, sweep, level, seed, test_rows=rows)
    assert "test_seen_matrix" in condition.data and "test_seen_matrix" not in full.data
    assert len(split.data["test_item_ids"]) > len(split.data["train_item_ids"])  # there are cold items
    # the model sees the same training data in both, so only the input and the exclusion could differ
    if family == "sequence":
        model = fit_baseline("popularity", {"count": "events"}, full)
    else:
        model = model_spec("popularity").build({}, n_items=len(full.data["train_item_ids"]), device="cpu", seed=0)
        model.fit(full.data["x_train"], item_ids=full.data["train_item_ids"])
    scored = [evaluate_phase(model, s, "test", family=family, protocol=protocol, exclude_seen=True, rows=rows,
                             targets="window") for s in (full, condition)]
    assert scored[0].n_rows > 0  # under exclude_seen no next item is left to reach here: window targets
    assert scored[1].metrics == pytest.approx(scored[0].metrics)


@pytest.mark.parametrize("level", [1, 2])
def test_markov_recommends_alike_at_every_truncation_with_seen_items_excluded(workspace, level):
    """Markov reads only the last item, which truncation keeps, so with seen items excluded against the whole
    history it must recommend the same lists at every level, ties included (ML-20M, 2026-09-30: 0.0230 on the
    full data, 0.0226 at every level, before ties had one order and every evaluation one exclusion path).
    The lists are compared, not a metric: under exclude_seen no synthetic user has a reachable target."""
    from seqrec_eval.analysis import fit_baseline
    from seqrec_eval.ablations import reference_condition
    from seqrec_eval.evaluate import ExcludeSeenPolicy, phase_inputs, seen_history

    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    rows = np.arange(split.data["test_target_matrix"].shape[0])
    full = reference_condition(split, "history", rows)
    condition = apply_condition(protocol, split, "history", level, test_rows=rows)
    model = fit_baseline("markov", {}, full)
    lists = []
    for data in (full, condition):
        adapter, source = phase_inputs(model, data, "test", "sequence")
        policy = ExcludeSeenPolicy(adapter, True, seen=seen_history(data.data, "test"),
                                   limit=len(data.data["train_item_ids"]), source=source)
        lists.append(policy.predict_on_batch(source, k=10).cols.numpy())
    assert not np.array_equal(full.data["test_source_sequences"].values,
                              condition.data["test_source_sequences"].values)  # the inputs do differ
    assert np.array_equal(lists[0], lists[1])


def test_a_next_item_already_seen_is_out_of_reach_when_seen_items_are_excluded(workspace):
    # review C4: no model can hit it, so the user is not scored (on Amazon, variants of one product share an item)
    from seqrec_eval.evaluate import scored_rows

    root, _, protocol = workspace
    split = final_split(protocol, load_split(root / "work", "synth"))
    allowed = set(scored_rows(split, "test", "next").tolist())
    kept = set(scored_rows(split, "test", "next", exclude_seen=True).tolist())
    targets, history = split.data["test_next_target_matrix"], split.data["test_source_matrix"]
    for row in allowed - kept:
        assert set(targets[row].indices) <= set(history[row].indices)  # every next item was seen already
    for row in kept:
        assert set(targets[row].indices) - set(history[row].indices)
    assert allowed - kept  # the synthetic users loop, so their next items are repeats


def test_a_catalogue_smaller_than_a_history_leaves_out_the_user_not_the_run(workspace):
    # review N49: excluding seen items, a user who has seen all but fewer than k items of a small catalogue cannot
    # be given k unseen ones; the evaluation stopped with "fewer than k unseen", now that user is left out, counted
    from seqrec_eval.analysis import fit_baseline
    from seqrec_eval.evaluate import evaluate_phase, fills_list

    root, _, protocol = workspace
    condition = apply_condition(protocol, final_split(protocol, load_split(root / "work", "synth")), "catalogue", 12)
    k = max(protocol.cutoffs)
    fills = fills_list(condition.data, "test", k)
    assert 0 < fills.sum() < fills.size  # some users have seen nearly all of the 12 items, others not
    model = fit_baseline("popularity", {}, condition)
    result = evaluate_phase(model, condition, "test", family="sequence", protocol=protocol, exclude_seen=True,
                            targets="window")  # every row asked: under exclude_seen, the synthetic next items repeat
    with_targets = np.diff(condition.data["test_target_matrix"].indptr) > 0  # the library scores only those
    assert set(result.sample_ids) == set(condition.eval_user_ids("test")[fills & with_targets])
    assert result.metadata["rows_too_few_unseen"] == int((~fills).sum())


def test_a_condition_lists_as_its_users_only_those_it_can_give_k_unseen_items():
    # review N49, where a catalogue sweep picks its users: the same rule as the evaluation's
    from scipy.sparse import csr_matrix

    from seqrec_eval.ablations import own_test_rows

    # items 0-6 are the training catalogue, 7 is first seen after training; k = 5 leaves room for 2 seen items.
    # Row 1 has seen 3 of the catalogue; row 2 has seen 2 of it and the new item, which is not in the catalogue.
    history = [[0, 1], [0, 1, 2], [0, 1, 7]]
    rows = np.repeat(np.arange(3), [len(items) for items in history])
    data = {"train_item_ids": np.array(list("abcdefg")),
            "test_next_target_matrix": csr_matrix((np.ones(3, np.float32), ([0, 1, 2], [3, 4, 5])), shape=(3, 8)),
            "test_source_matrix": csr_matrix((np.ones(rows.size, np.float32), (rows, np.concatenate(history))),
                                             shape=(3, 8)),
            "test_source_sequences": ItemSequences.from_rows(history, n_items=8)}
    assert own_test_rows(data, "test_next_target_matrix", exclude_seen=True, k=5).tolist() == [0, 2]
    assert own_test_rows(data, "test_next_target_matrix", exclude_seen=True, k=4).tolist() == [0, 1, 2]
    assert own_test_rows(data, "test_next_target_matrix").tolist() == [0, 1, 2]  # nothing excluded, all fill
    with pytest.raises(ValueError, match="depends on k"):
        own_test_rows(data, "test_next_target_matrix", exclude_seen=True)


def test_an_inference_sweep_keeps_the_full_datas_shuffled_control(workspace):
    # its training data is the full data's, so a new shuffle per level would move the control by chance alone
    from seqrec_eval.analysis import condition_results

    root, _, protocol = workspace
    found = condition_results(protocol, root / "work", "history_inference", "synth")
    reference = found["markov_shuffled"][REFERENCE][0][0]
    for result, _ in found["markov_shuffled"]["5"]:
        assert np.array_equal(result.sample_ids, reference.sample_ids)
        # Markov reads the last item, which truncation keeps: the same chain gives the same values
        assert np.array_equal(result.per_user["ndcg@5"], reference.per_user["ndcg@5"])


@pytest.mark.parametrize("evaluator", ["reversed", "skips the last batch"])
def test_an_evaluator_that_changes_its_batches_is_caught(workspace, monkeypatch, evaluator):
    """If the library's evaluator ever reordered or skipped batches, a masked condition would fail loudly (H33)."""
    from seqrec_eval import evaluate
    from seqrec_eval.ablations import reference_condition
    from seqrec_eval.analysis import fit_baseline
    from seqrec_eval.evaluate import BatchAlignmentError

    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    rows = np.arange(split.data["test_target_matrix"].shape[0])
    condition = apply_condition(protocol, split, "history", 2, None, test_rows=rows)
    model = fit_baseline("popularity", {"count": "events"}, reference_condition(split, "history", rows))

    def changed(policy, *, source, targets, **_):
        starts = list(range(0, source.n_rows, 8))
        for start in (starts[::-1] if evaluator == "reversed" else starts[:-1]):
            policy.predict_on_batch(source.take_rows(start, min(start + 8, source.n_rows)), k=5)

    monkeypatch.setattr(evaluate, "evaluate_recommender", changed)
    with pytest.raises(BatchAlignmentError):
        # window targets: under exclude_seen no synthetic user has a next item left to reach (all are repeats)
        evaluate.evaluate_phase(model, condition, "test", family="sequence", protocol=protocol, exclude_seen=True,
                                rows=rows, targets="window")


def test_the_policy_checks_batches_against_the_source_the_evaluator_is_given(workspace, monkeypatch):
    # the row check only guards anything if the policy holds the very source the evaluator batches
    from seqrec_eval import evaluate
    from seqrec_eval.ablations import reference_condition
    from seqrec_eval.analysis import fit_baseline

    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    rows = np.arange(split.data["test_target_matrix"].shape[0])
    condition = apply_condition(protocol, split, "history", 2, None, test_rows=rows)
    model = fit_baseline("popularity", {"count": "events"}, reference_condition(split, "history", rows))
    real, seen = evaluate.evaluate_recommender, {}

    def recording(policy, *, source, **kwargs):
        seen.update(policy=policy, source=source)
        return real(policy, source=source, **kwargs)

    monkeypatch.setattr(evaluate, "evaluate_recommender", recording)
    evaluate.evaluate_phase(model, condition, "test", family="sequence", protocol=protocol, exclude_seen=True,
                            rows=rows)
    assert seen["policy"].masking and seen["policy"].source is seen["source"]


def test_the_policy_excludes_the_full_history_rather_than_the_input():
    from scipy.sparse import csr_matrix

    from seqrec_eval.baselines import Popularity
    from seqrec_eval.evaluate import ExcludeSeenPolicy

    train = ItemSequences.from_rows([[0] * 5 + [1] * 4 + [2] * 3 + [3] * 2 + [4]], n_items=6)
    model = Popularity().fit(train)
    truncated = ItemSequences.from_rows([[3], [4]], n_items=6)           # the input: only the last event
    seen = csr_matrix(np.array([[1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]], dtype=np.float32))  # the whole history
    policy = ExcludeSeenPolicy(model, True, seen=seen, limit=6)
    assert policy.predict_on_batch(truncated, k=2).cols.tolist() == [[2, 4], [0, 1]]
    assert ExcludeSeenPolicy(model, True).predict_on_batch(truncated, k=2).cols.tolist() == [[0, 1], [0, 1]]


# ---------------------------------------------------------------------------
# next-item targets, training on every user, and provenance
# ---------------------------------------------------------------------------

def test_next_targets_are_each_users_first_target_event(workspace):
    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    window, upcoming = split.data["test_target_matrix"], split.data["test_next_target_matrix"]
    assert upcoming.shape == window.shape
    assert ((upcoming > 0) > (window > 0)).nnz == 0  # every next target is also a window target
    has_target = np.diff(window.indptr) > 0
    has_next = np.diff(upcoming.indptr) > 0
    # every user has a next target except the one whose real next item the builder deleted (H16)
    assert np.array_equal(has_next, has_target & (split.eval_user_ids("test") != "u001"))
    # the synthetic users do one thing a day, so the next target is a single item
    assert set(np.diff(upcoming.indptr)[has_next].tolist()) == {1}
    assert split.info["timestamps"]["next_targets"]["test"]["mean_items"] == 1.0


def test_finals_score_the_protocol_targets_and_the_other_definition_beside_them(workspace):
    root, _, protocol = workspace
    assert protocol.targets == "next"
    for final in sorted((root / "work" / "runs").glob("synth/*/*/final-seed0")):
        record = json.loads((final / "done.json").read_text())
        assert json.loads((final / "test.json").read_text())["metadata"]["definition"] == "next"
        assert json.loads((final / "test_window.json").read_text())["metadata"]["definition"] == "window"
        assert "test_window" in record
    report = (root / "work" / "reports" / "analysis.md").read_text()
    assert "### The other target definition" in report


def test_training_covers_every_user_and_the_split_records_its_library(workspace):
    from seqrec_eval.splits import library_provenance

    root, _, _ = workspace
    split = load_split(root / "work", "synth")
    assert split.info["build_parameters"]["temporal_train_users"] == "all"
    assert split.info["manifest"]["stages"]["data"]["temporal_train_users"] == "all"
    assert split.info["library"]["version"] == library_provenance()["version"]


def test_a_setting_the_installed_library_lacks_is_refused_before_building():
    from seqrec_eval.splits import _check_library

    with pytest.raises(RuntimeError, match="does not accept \\['no_such_option'\\]"):
        _check_library({"dataset": "ml20m", "no_such_option": 1})


def test_a_slice_too_small_to_mean_anything_is_not_reported():
    from compresso_recsys.evaluation import EvaluationResult

    from seqrec_eval.analysis_report import MIN_SLICE_USERS, _slice_mean

    values = np.arange(MIN_SLICE_USERS + 5, dtype=np.float64)
    result = EvaluationResult(metrics={"m": 0.0}, per_user={"m": values}, sample_ids=np.arange(values.size),
                              n_rows=values.size, n_scored_rows=values.size, required_k=1)
    assert _slice_mean(result, values < 3, "m") is None
    assert _slice_mean(result, values >= 0, "m") == pytest.approx(values.mean())


# ---------------------------------------------------------------------------
# H01b: catalogue sweeps score each condition on its own users
# ---------------------------------------------------------------------------

def test_a_condition_scores_only_users_with_a_known_next_target_and_a_history():
    from scipy.sparse import csr_matrix

    from seqrec_eval.ablations import own_test_rows

    # items 0-2 are known (in training); 3 is first seen after training. Row 2's only target is unseen,
    # row 3 has a known target but no history left.
    targets = csr_matrix(np.array([[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=np.float32))
    data = {"test_next_target_matrix": targets, "train_item_ids": np.array(["a", "b", "c"]),
            "test_source_sequences": ItemSequences.from_rows([[1], [0, 3], [2], []], n_items=4)}
    assert own_test_rows(data, "test_next_target_matrix").tolist() == [0, 1]


def test_the_stratified_order_is_proportional_at_every_prefix():
    from seqrec_eval.ablations import _stratified_order

    counts = np.arange(1000)[::-1]  # item i has popularity 999 - i: the strata are 0-99, 100-199, ...
    order = _stratified_order(counts, 10, np.random.default_rng(0))
    assert np.array_equal(np.sort(order), np.arange(1000))
    for k in (50, 100, 250, 500, 750):
        per_stratum = np.bincount(order[:k] // 100, minlength=10)
        assert per_stratum.sum() == k and np.all(np.abs(per_stratum - k / 10) <= 1), (k, per_stratum)


def test_stratified_catalogues_are_nested_within_a_seed_and_differ_between_seeds(workspace):
    from seqrec_eval.ablations import build_condition

    root, _, protocol = workspace
    split = load_split(root / "work", "synth")
    kept = {(level, seed): set(build_condition(protocol, root / "work", split, "catalogue_strata", level, seed)
                               .data["train_item_ids"].tolist())
            for level in (0.25, 0.5, 0.75) for seed in (0, 1)}
    for seed in (0, 1):
        assert kept[(0.25, seed)] < kept[(0.5, seed)] < kept[(0.75, seed)]
    assert kept[(0.5, 0)] != kept[(0.5, 1)]


def test_catalogue_conditions_keep_no_fixed_set_and_score_their_own_users(workspace):
    from seqrec_eval.ablations import build_condition, build_reference, own_test_rows

    root, _, protocol = workspace
    work = root / "work"
    for sweep in ("catalogue", "catalogue_strata"):
        assert not (ablation_root(protocol, work, sweep, "synth") / "test_rows.npy").exists()
    split = load_split(work, "synth")
    condition = build_condition(protocol, work, split, "catalogue_strata", 0.75, 0)
    rows = condition.test_rows
    assert np.array_equal(rows, own_test_rows(condition.data, "test_next_target_matrix"))
    known = len(condition.data["train_item_ids"])
    targets = condition.data["test_next_target_matrix"][rows]
    assert all((targets[r].indices < known).any() for r in range(targets.shape[0]))
    reference = build_reference(protocol, work, split, "catalogue_strata")
    assert np.array_equal(reference.test_rows, own_test_rows(split.data, "test_next_target_matrix"))


def test_seeds_that_scored_different_users_are_pooled_per_user():
    from compresso_recsys.evaluation import EvaluationResult

    from seqrec_eval.report import pool_over_seeds

    def result(ids, values):
        v = np.asarray(values, dtype=np.float64)
        return EvaluationResult(metrics={"m": float(v.mean())}, per_user={"m": v}, sample_ids=np.array(ids),
                                n_rows=v.size, n_scored_rows=v.size, required_k=1)

    pooled = pool_over_seeds([result(["a", "b"], [1.0, 0.0]), result(["b", "c"], [1.0, 0.5])])
    assert pooled.sample_ids.tolist() == ["a", "b", "c"]
    assert pooled.per_user["m"].tolist() == [1.0, 0.5, 0.5]  # b is averaged over its two seeds
    assert pooled.metadata["user_seed_units"] == 4 and pooled.target_fingerprint is None
    same = pool_over_seeds([result(["a", "b"], [1.0, 0.0]), result(["a", "b"], [0.0, 0.0])])
    assert same.per_user["m"].tolist() == [0.5, 0.0]  # identical users: the plain seed average


def test_the_catalogue_report_compares_within_levels_and_marks_small_levels(workspace):
    root, _, _ = workspace
    reports = root / "work" / "reports"
    text = (reports / "ablation-catalogue_strata.md").read_text()
    assert "its own users" in text
    assert "### Each level against the full data" not in text and "No level-against-full test" in text
    assert "*users scored*" in text and "descriptive" in text  # 80 synthetic users < 1,000
    gap = (reports / "ablation-catalogue_strata-gap.csv").read_text().splitlines()
    header = gap[0].split(",")
    assert "users" in header and "below_min_users" in header
    history = (reports / "ablation-history.md").read_text()
    assert "Every condition is scored on the same" in history  # fixed-set sweeps are unchanged


@pytest.mark.parametrize("value", ["0", "1.5", "true"])
def test_a_bad_minimum_of_users_is_refused(tmp_path, value):
    path = tmp_path / "protocol.toml"
    path.write_text(PROTOCOL.replace("levels = [1, 2, 5]", f"levels = [1, 2, 5]\nmin_level_users = {value}"))
    with pytest.raises(ProtocolError):
        cli.main(["--protocol", str(path), "--work-dir", str(tmp_path / "work"), "plan"])
