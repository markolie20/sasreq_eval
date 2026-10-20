# Decisions and changes

What was decided or changed after the protocol was first frozen (2026-09-24), why, and what it does to
results. Each entry names its issue in the review (`review-plan/ISSUES.md`; evidence in
`review-plan/findings/`), where it lives in the code, and the tests that hold it. The per-setting choices
made before the freeze are the "Decided" lines in `protocol.toml`; they are summarised at the end.

No run on the DGX exists yet, so none of the changes below discards a result. Every one of them changes
fingerprints, and earlier local runs (the ML-20M analysis of 2026-09-24) are not reused.

---

## 1. Catalogue sweeps score each level on its own users (2026-09-29, H01b)

**Decision.** In the catalogue sweeps each condition is scored on its own test users: those with a next
target still in the reduced catalogue and a history item left. Stratified catalogues are nested within a
seed, seeds are pooled per user, and a level scored on fewer than `min_level_users` (1,000) is descriptive.

**Why.** One fixed set of users across levels and seeds shrank to users whose next targets are all items no
model can recommend. Stratified sweep: 83–100% of the fixed set on every dataset, so every model scored 0 and
the sweep measured nothing.

**Effect on results.**
- Models are compared within a level, through the gap.
- There is no level-against-full test for catalogue sweeps, because different levels have different users.
- Before refit and H16, ML-20M stratified 0.1 had about 530 pooled users, so it is descriptive.
- Amazon All_Beauty could not support the sweep; it was replaced by Toys_and_Games (entry 9), which has
  3,096 to 37,972 users per stratified level and 24,579 to 46,637 per top-k level (sized before entries 4
  and 7).

**Side effects.**
- The catalogue transform is at version 2.
- No `test_rows.npy` is written for catalogue sweeps.
- Result metadata carries the pooled per-user seed counts.
- The report states the users per level and draws descriptive levels hollow.

**Where.** `ablations.py` (`_stratified_order`, `own_test_rows`, `build_condition`), `report.py`
(`pool_over_seeds`), `ablation_report.py`; `tests/test_ablations.py`. Findings: `H01b.md`.

## 2. Yambda at 50% of the 50m release's users (2026-09-29, H14)

**Decision.** `user_sample = 0.5` (was 0.2).

**Why.** 20% left 1,745 test users and too few for the catalogue levels.

**Effect.** 4,373 test users; every catalogue level has at least 1,000 pooled users. After the H16 rule
(entry 7): 3,840.

**Side effects.** No cache exists for 50% yet: the first `prepare` (on the DGX) builds one from the raw
parquet, about 23M listens. Samples are nested by user hash, so the 20% users are all inside the 50%.
`organic_only` and the download revision: decided 2026-10-05, below.

**H14 (Mark, 2026-10-05).**
- `organic_only` stays `false`, the library default. Every Yambda listen has an `is_organic` flag: 1 when the
  user started it, 0 when Yandex's recommender served it. The adapter would filter on it once, at load, from
  histories, training data and targets alike. Recommender-served listens are kept, for three reasons:
  - a listen is recorded only once at least half the track has played (Yambda documentation), so it records
    the user's engagement, not mere exposure;
  - Recombee's own clients' logs also hold interactions their recommendations caused;
  - none of the other datasets can separate the two kinds.
- **The download is pinned by checksum.** The release was fetched from its latest version on Hugging Face on
  2026-09-09: `listens-50m.parquet`, sha256 `eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1`, 369,110,332 bytes. The DGX copy is that file, and the
  library does not download again while it is present. Check it there with `sha256sum`.
- **Limitation:** part of Yambda's targets, and of its repeat rate, comes from Yandex's recommender (the radio).
- **Effect:** none: the settings are those the protocol already had. A comment in both protocols records it.

## 3. Matrix models read their input in their own catalogue (2026-09-29, N0, bug fix)

**Change.** Popularity, EASE and ELSA receive their input projected onto their training catalogue
(`phase_inputs`). The library's adapter refuses a wider matrix.

**Why.** Every real temporal split appends items first seen after training. Without this, every matrix-model
trial on the DGX would have failed.

**Side effects.**
- A test history made only of items first seen after training reaches a matrix model empty, and is ranked by
  tie order, not by popularity. Sequence models read those items as `unk` instead.
- How many users this affects has not been counted yet. It is a DGX check. (Counted since entry 40.)
- The smoke test's synthetic data now contains cold items, so this can no longer pass unnoticed.

**Where.** `evaluate.py` (`phase_inputs`), `latency.py`; `tests/test_smoke.py`. Findings: `H53.md` §A.

## 4. Refit on train+validation before test (2026-09-29, H55, N1; `refit = true`)

**Decision.** Search trials fit on the training window and score validation. Everything scored on test is
fitted once more, per seed and with the selected settings, on everything before the test window. That covers
final runs, latency, the analysis's baselines and controls, and every ablation condition.

**Why.** Without it, validation and test are different tasks: a test history runs a whole window past the end
of training. Three consequences:
- items first seen in validation cannot be recommended;
- BERT4Rec's `[MASK]` lands on untrained positions;
- the search cannot see either.

It is also the standard temporal protocol, and what the design's "all interactions before the cutoff" reads
as.

**How.** The library has no such training set. The suite rebuilds it from the library's prepared events by
the library's train-stage rule, one window later:
- users: everyone with at least `min_user_support` distinct items;
- catalogue: the validation catalogue;
- matrix: the maximum of the two windows' counts, as `x_train`.

Every `prepare` proves the rule: the same code must reproduce the split's own `x_train`,
`x_train_sequences`, `train_source_sequences` and `train_user_ids` exactly. It did on planted data, on Amazon
and on Yambda 20% (whose user ids are numeric, where text order differs).

**Effect on results.**
- More test users have a recommendable next item: Amazon Toys_and_Games 51,639 → 77,976 of 109,774
  (All_Beauty, since replaced: 13 → 73 of 259), Yambda 20% 1,427 → 1,628 of 1,745.
- BERT4Rec on planted data: 0.31 → 1.00 of the oracle, except for the longest users (below).
- Recency baselines are no longer stale.

**Side effects.**
- Final runs record no validation score; the report takes validation from the search.
- Scoring validation on refitted data raises an error, and a run given the wrong split refuses to start.
- The selected settings are applied unchanged to a larger training set, the accepted trade-off.
- Time-decayed popularity's "now" moves to the start of the test window.
- BERT4Rec keeps one off-by-one: users at exactly the longest history length still meet an untrained position
  (N1). The complete fix is a library change.
- New files beside each split (`x_refit*`, `refit_*`). An existing split gets them on the next `prepare`,
  without a rebuild.
- `refit` is part of every run fingerprint and of the evaluation key (entry 8).
- It needs the library branch with `_train_stage_args`.
- A model over `max_items` is skipped already at the search, judged on the refit catalogue the final runs will
  fit, not on the smaller training catalogue (entry 11).

**Where.** `refit.py`, `splits.py` (`final_split`), `runner.py`, `cli.py`, `analysis.py`, `latency.py`,
`report.py`; `tests/test_smoke.py` (proof, refit set, finals, baselines, refusals). Plan:
`review-plan/plans/refit-and-history-n1.md`; evidence: `findings/refit/`, `findings/H53.md` §K.

## 5. History length: level *n* is a context of *n* items (2026-09-29, N6 = H30)

**Decision.** At level *n*, training histories keep their last *n* + 1 events, and input histories their last
*n*.

**Why.** A history of *n* events trains contexts of at most *n* − 1 items, because its last event is only a
target. Keeping *n* in training served every model one item more than it had ever trained with, trained
nothing at all at level 1, and put BERT4Rec's `[MASK]` on an untrained position at every level.

**Effect.**
- A model is trained and served with the same context.
- Level 1 trains: pairs of one item → the next.
- EASE and ELSA agree, since a row of *n* + 1 items teaches "*n* items → one more".

**Side effects.**
- `history_length` is at version 3, so earlier condition results do not match.
- The manipulation check's training bound is *n* + 1.
- The inference-only sweep is unchanged.

**Where.** `ablations.py` (`_history_length`); `tests/test_ablations.py` (level 1 is in the test grid).

## 6. The seen-item mask checks its batches (2026-09-29, H33)

**Change.** Where an ablation keeps the full history for exclusion while the model reads a shortened one,
each evaluation batch must satisfy three checks:
- it must be exactly the next rows of the input;
- every item it reads must be in that user's history;
- every row must be used once.

Otherwise the run fails with `BatchAlignmentError`.

**Why.** Batches carry no user ids, so the mask found its rows by counting. That is right only while the
library's evaluator passes every row once, in order. It does so today, so no result was affected, but nothing
checked it.

**Side effects.**
- None on results.
- A test pins the library evaluator's batching, so a library change fails the tests rather than the results.

**Where.** `evaluate.py` (`ExcludeSeenPolicy`); `tests/test_evaluate.py`, `tests/test_ablations.py`.

## 7. "Next item" is the real next moment, scored where a model could get it (2026-09-29, H16)

**Decision.**
- **B1.** The next-item target is what the user did at their *real* first moment in the window. That moment
  is taken over all their events, including those on items the library deleted because they were too rare
  in the window. A deleted item leaves the row empty; it is not replaced by a later event.
- **B2.** Next-item metrics are computed only over users whose next item is in the training catalogue (for
  test, the refit catalogue).

**Why.**
- The library keeps a *new* item only with at least `item_min_support` users counted inside the target
  window. Which new items survive is therefore decided by the window itself.
- A user whose first event was a rare new track was silently scored on the event after it. That is a
  next-but-one task, which costs sequential models most.
- A user whose next item was a popular new track kept an unwinnable target. Which of the two happened
  depended on the track's popularity during the test window.
- Users who score 0 for every model carry nothing for a comparison.

**Effect on results.** Test users scored, with refit, before → after:

| Dataset | Before | After | Removed |
|---|---|---|---|
| ML-20M | 3,321 | 2,942 | next item new |
| Amazon Toys_and_Games (entry 9) | 109,774 | 77,976 | next item new |
| Music4All | 30,369 | 30,180 | both kinds |
| Yambda 50% | 4,373 | 3,840 | mostly deleted rare items (324) |
| OTTO 2% | 18,458 | 14,605 | mostly deleted rare items (3,542) |

Validation (the search): ML-20M 3,012, Amazon Toys_and_Games 99,898, Music4All 30,676, Yambda 3,278, OTTO
13,209. The Toys_and_Games counts come from the validated emulation (see entry 9); the first `prepare` gives
the real ones.

- Paired differences between models do not change for the users removed by B2, who scored 0 everywhere.
- The next-but-one bias against sequential models is gone.
- Absolute scores rise, since fewer zeros are averaged.

**Side effects.**
- Next targets are at version 2, and an existing split gets them recomputed on the next `prepare`.
- A search trial may score fewer than `max_val_users` users.
- The window diagnostic still scores every user with a target, so its population is larger than the primary
  metric's.
- Every result records how many users it left out (`rows_unrecommendable_next`), and the stage-1 report
  states it per dataset.
- The ablations' fixed test users follow the same rule. The catalogue sweeps' own-users rule was already
  this rule.
- These rules left Amazon All_Beauty with 73 test users, which decided its replacement (entry 9).

**Where.** `timestamps.py` (`next_targets`), `evaluate.py` (`scored_rows`, `recommendable_next`),
`ablations.py` (`_eligible_test_rows`), `report.py`; tests in `tests/test_smoke.py` and
`tests/test_ablations.py`. Findings: `H16.md` (with the checks against the real splits).

## 8. What decides the scored users keys every cache that holds them (2026-09-29, H04)

**Change.**
- `SCORING_VERSION` (`protocol.py`) is part of every run fingerprint. It is bumped whenever code changes what
  a scored target means, and entry 7 made it 2.
- The caches that hold users rather than runs are keyed by the evaluation key: the target definition, `refit`
  and the scoring version. Those caches are an ablation's fixed test users, its per-condition profiles, and
  the analysis profile.

**Why.** These caches ignored the target definition and the refit, so switching either would have silently
reused users and profiles computed under the other.

**Side effects.**
- Every run, condition and profile fingerprint changed once.
- Other code changes are still not in any fingerprint (H05, open).

**Where.** `protocol.py`, `ablations.py` (`condition_fingerprint`), `analysis.py` (`profile_path`);
`tests/test_ablations.py`.

## 9. Amazon: Toys_and_Games replaces All_Beauty (2026-09-29, H56)

**Decision.** `amazon_category = "Toys_and_Games"`. The dataset keeps its name (`amazon`) and every other
setting:
- 339-day windows;
- `min_user_support` 5;
- `item_min_support` 1;
- no rating threshold;
- `exclude_seen = true`.

The report prints the category.

**Why.** All_Beauty has about one review per user (93% have a single one). It left 259 test users, 95% of
them with next items no model can recommend, and 73 once entry 7 scores only users a model could score. No
statistic can rest on that.

**Effect.**
- 109,774 test users, 77,976 of them scored under entries 4 and 7; 99,898 of 142,171 validation users.
- The catalogue sweeps pass at every level.
- These counts come from the stage emulation, which reproduced the real splits exactly on All_Beauty, Yambda
  and Music4All. But the emulation reads the ratings file directly, while the library's adapter first keeps
  only reviews of products in the metadata. The real counts may be slightly lower, and the first `prepare`
  records them in `split_info.json`.

**Side effects.**
- **EASE is skipped on Amazon now.** The training catalogue is 447,069 items (516,867 under refit), far over
  `max_items` = 40,000, where All_Beauty had about 5,500. EASE is then compared only on ML-20M and Music4All.
- **A far larger dataset:** 16.1M rated events, ML-20M's scale. Its `prepare` and runs belong on the DGX.
  ELSA and GRU score the full 447k-item catalogue.
- The ratings file and the five metadata parts are already downloaded, so there is no download at `prepare`.
- H39 (1-star reviews count as positive) is still open and applies here as it did to All_Beauty.
- The All_Beauty figures in `review-plan/findings/` stay as the record of why.

**Where.** `protocol.toml` (`[datasets.amazon]`). Evidence: `findings/H01b.md` (Toys sizing),
`findings/H16/h16_sizes_toys.json`.

## 10. The local protocol is the main protocol with a smaller batch (2026-09-29)

**Change.** `protocol.local.toml` was a copy from 2026-09-24 that had missed every later decision. It had no
`targets` or `train_users`, so both fell back to "window". It had no `refit`, Yambda at 20%, and the history
grid stopped at 500. It is now `protocol.toml` with `eval_batch_size = 128`. A test fails if the two differ
in anything else.

**Effect.** Batch size enters no fingerprint, so every run fingerprint of the local file equals the main
one's. Local and DGX runs share their directories and results.

**Side effect.** A local `prepare` of Yambda now builds the 50% sample (about 23M listens), which is heavy
for the laptop. It is meant for the DGX, as is Amazon Toys_and_Games.

**Where.** `protocol.local.toml`; `tests/test_smoke.py`
(`test_the_local_protocol_differs_from_the_main_one_only_in_batch_size`).

## 11. `max_items` is judged on the catalogue the final runs fit (2026-09-29, fix to entry 4)

**Change.** Under refit, a search trial decides whether a model exceeds `max_items` on the refit catalogue,
not on the smaller training catalogue.

**Why.** Otherwise a dataset whose training catalogue is under the limit but whose refit catalogue is over
it would have its model searched in full and then skipped at the final runs.

**Effect.** None on the current datasets: ML-20M (19,698 and 21,557 items) and Music4All (36,508 and
36,797) stay under 40,000 either way, and Amazon, Yambda and OTTO are far over it either way.

**Where.** `runner.py` (`_execute`); `tests/test_smoke.py`
(`test_a_model_too_large_for_the_refit_catalogue_is_skipped_at_the_search_already`).

## 12. The lock pins compresso-recsys 0.3.7; the DGX runs the branch build (2026-09-29, H09)

**Change.** `uv.lock` pins compresso-recsys 0.3.7 from PyPI (was 0.3.6, below `pyproject.toml`'s own
`>= 0.3.7`). Only that package changed in the lock.

**Why.** The suite needs 0.3.7's temporal split, with `dataset_options` and item sequences. A fresh
`uv sync` from the old lock installed a version the suite cannot use.

**Side effects.**
- 0.3.7 does not contain `temporal_train_users` or `_train_stage_args`. They exist only as uncommitted
  changes on the library's `local-temporal-train-all-users` branch. Against 0.3.7, `prepare` refuses the
  protocol's `train_users = "all"`, and the refit cannot be built. Both failures are loud.
- The DGX therefore gets the branch build installed by hand, which records itself as `0.3.7+trainusers`.
- That build has to be committed first. On 2026-09-29 the branch's five changed files were still uncommitted,
  and its commit (6547522) has none of it. A `git+file` install, a clone or a push takes only commits.
- `uv sync`, and `uv run` without `--no-sync`, would replace that build with 0.3.7 from the lock. Run the suite
  with the environment's own `seqrec-eval` or `uv run --no-sync` there.
- Once the branch is released, the lock should pin that release.

The torch half of the environment is entry 13.

## 13. torch from PyTorch's CUDA 12.6 index (2026-09-29, H08)

**Change.** `pyproject.toml` takes torch from `https://download.pytorch.org/whl/cu126` (`[tool.uv.sources]`,
with `explicit = true`, so only torch comes from there). The lock now pins `torch 2.14.0+cu126` and the matching
`nvidia-*-cu12` libraries, which replace the CUDA 13 ones. No other package changed.

**Why.** PyPI's Linux wheel of torch 2.14.0 is the CUDA 13 build, which fails on the DGX twice:
- it is compiled for sm_75 and up, so the DGX's two V100-DGXS-32GB (sm_70) have no kernels;
- CUDA 13 needs driver 580+, and the DGX has 550.163.01 (CUDA 12.4).

The cu126 wheel exists for Python 3.10 to 3.15. Its compiled architecture list, read from the wheel's
`libtorch_python.so`, is `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`. Driver 550 runs a CUDA 12.6 build through
minor-version compatibility, and nothing needs compiling on the device.

**Effect.** The same lock serves both machines. The laptop's RTX 4050 (sm_89) runs the sm_86 kernels.

**Side effects.**
- The laptop's `.venv` still has the CUDA 13 torch until its next `uv sync`. That sync would also reinstall
  compresso-recsys from the lock (entry 12).
- The final check is on the DGX: a matrix product on a V100 with this build. It waits on the machine's
  conventions for installing tools.
- The DGX's GPUs are shared: both were at 97–98% use by another user's jobs when checked.

**Where.** `pyproject.toml`, `uv.lock`, README "Install". Evidence: `review-plan/ISSUES.md` #7.

## 14. cr's ELSA is the ELSA under study (2026-09-29, H25)

**Decision.** The ELSA the suite runs, `compresso_recsys`' `ELSATrainer`, is the production ELSA the research
design compares against.

**Implications.** Two behaviours found in the model review are production behaviour, not deviations, so they
stay. The report should say what they do (review N3, N4):
- **A ReLU on the scores**, in training and prediction. On planted data where item groups matter and order does
  not, ELSA reached 0.81 of the best possible NDCG with seen items masked, against 0.96 without the ReLU and 0.98
  for EASE. Searching `use_relu` would test a variant that is not production.
- **No self-term subtraction at prediction.** A seen item scores its own count on top. It is irrelevant where
  seen items are excluded (ML-20M, Amazon), but on Music4All, Yambda and OTTO ELSA partly recommends by
  repeat count. That favours repeats, as production does.

Whether to also run the published variants as a sensitivity check is open.

**Where.** `review-plan/findings/H53.md` §C–D; ISSUES #6, N3, N4.

## 15. `targets`, `refit` and `train_users` must be stated (2026-09-29, H11)

**Change.** A protocol without `[protocol].targets`, `[protocol].refit` or a dataset's `train_users` no longer
loads. It used to fall back to "window", false and "window".

**Why.** The stale local protocol lacked the first and last key, and silently ran window targets, no refit and
training on the train window's users only (entry 10).

**Effect.** No result changes; a missing key now fails with the choices spelled out.

**Where.** `protocol.py`; `tests/test_smoke.py` (`test_a_protocol_must_state_targets_refit_and_training_users`).

## 16. No model is chosen around a failed trial (2026-09-29, H06, H07)

**Decision (Mark).** No run should fail. While any trial of a model has failed, `final` refuses to select its
configuration: fix the cause and rerun. Only a failure that cannot be fixed, for example a configuration too
large for the hardware, may be excluded, explicitly, with `final --accept-failed`.

**Why.** Selecting from the trials that happened to succeed gives a model a smaller, silently different search
than the others. The equal budget is part of the comparison. A trial whose validation score is not a finite
number is treated as failed too: a NaN can never be beaten by ">", so it would have stayed "best" (H07). cr
never produces one today (it returns 0.0 when no user can be scored and raises on NaN scores), so this is a
guard.

**Effect.** No result changes while nothing fails.

**Side effects.**
- Accepted failures are written to `accepted_failures.json` in the model's run folder, and the stage-1 report
  lists them with their reasons and how many trials the search selected from.
- An unresolved failure is marked ⛔ in the report, and so is a failed final seed.
- A baseline with a non-finite validation score stops `analyse`.
- SASRec's out-of-memory search corner (N8) is the expected first user of `--accept-failed`, unless the space is
  trimmed.

**Where.** `runner.py` (`summarize_trials`, `plan_finals`), `cli.py`, `analysis.py`, `report.py`; tests in
`tests/test_smoke.py`.

## 17. Every run records the code it ran (2026-09-29, H05, H10)

**Decision (Mark).** Report it; do not put it in the fingerprints.

**Change.**
- Each run's `done.json` holds the library's and the suite's version and a hash of their imported source.
- `split_info.json` holds the same for the library.
- The stage-1 report states, per dataset, the build its runs used, and warns when there is more than one.

**Why.** Fingerprints cover the protocol and `SCORING_VERSION` but not the code. A fix to a model in the library
would silently reuse runs made before it. The installed metadata need not be what was imported: with the tests'
`PYTHONPATH` it reported 0.3.6 while the branch code ran (H10). The source hash records what actually ran.

**Effect.** No result changes. Laptop and DGX runs still share directories, and the report shows whether they
came from the same build.

**Where.** `splits.py` (`code_provenance`, `library_provenance`), `runner.py`, `report.py`
(`code_builds`); `tests/test_smoke.py` (`test_the_report_lists_failures_and_the_code_the_runs_used`).

## 18. The knee: anchored on the full data, a 10% margin, and its power reported (2026-09-30, N10, H21, H20)

**Bug fixed (N10).** The knee started at the best-looking level. When the first test (the full data against
that level) failed, a reduced level that merely looked best was reported as the knee with nothing shown. With
every level truly equal, the reported knee was spread evenly over all levels (200 simulated sweeps). It now
starts at the full data, and is the full data when no level passes.

**Decisions (Mark).**
- **Anchor.** Every level is compared with the full data, not with the level that scored best. The best of
  several noisy levels is lucky by construction: with all levels equal, it was a reduced level in 167 of 200
  sweeps. That inflated the yardstick and biased every test against the other levels. It is also the question
  asked: how far can the data be reduced without losing against what the model has now?
- **Margin.** `knee_margin` = 10% (was 2%), now as a share of the full-data score.
- **Power is reported.** For each model, the chance the test shows "within δ" for a level that loses nothing.
  No cut-off is applied yet. A per-dataset margin tied to a target power (90% was mentioned) is open.

**The calculation.** With n test users and s the standard deviation of the per-user differences (level minus
full), the mean difference has standard error s/√n. The test shows "the loss is below δ" when the observed mean
lies above −δ + 1.645·s/√n. For a level with no loss:

    power = Φ(δ·√n / s − 1.645)        and, for a target power,        δ = (1.645 + z(power)) · s / √n

(α = 0.05; z(0.90) = 1.282, so δ = 2.93·s/√n for 90%). This was checked against the exact sign-flip test by
simulation: 0.93 predicted against 0.93 observed in the test suite, and 0.15 against 0.15 at ML-20M's size.

**Effect.** Chance of showing "within δ" for a level with no loss, across plausible NDCG levels and
correlations between levels:

| Test users | At 2% | At 10% |
|---|---|---|
| ML-20M (2,942) | 9–25% | 47–100% |
| Yambda (3,840) | 10–30% | 56–100% |
| OTTO (14,605) | 17–70% | 97–100% |
| Music4All (30,180) | 26–93% | 100% |
| Toys (77,976) | 49–100% | 100% |

**Side effects.**
- A knee at 10% is a weaker statement than one at 2%: "loses less than a tenth of the full-data score".
- The knee table's "best level" column is gone; it shows δ, the knee, the first level that failed and the power.
- The test's validity was checked and holds: it rejects 3–7% of the time at nominal 5% when the true loss is
  exactly δ.
- `knee_margin` is report-only, so no run has to be repeated.

**Where.** `ablation_report.py` (`find_knee`, `noninferiority_power`), `ablations.py` (default),
`protocol.toml`; tests in `tests/test_ablations.py`; simulations in `review-plan/findings/S1-rest/`.

## 19. Training memory bounded in the library, same maths (2026-09-30, H35, N8)

**Decision (Mark).** Fix the out-of-memory corners in the library on the local branch
(`local-temporal-train-all-users`), rather than trim the search spaces or accept failed trials.

**Change**, in compresso-recsys, uncommitted. Each trainer gets `loss_chunk_elements` (default 2²⁷), which bounds
what one training step holds at once:
- **GRU** (`simple_rnn.py`): the full-catalogue loss is scored in chunks of positions. The dropout on the states
  is still drawn once over all positions, and one chunk takes the exact old code path.
- **SASRec** (`sasrec.py`): the negatives are scored only at valid positions (padding was scored and then
  thrown away before), in chunks. The negatives are drawn exactly as before.
- **ELSA** (`elsa.py`): each batch is scored in groups of users, and the gradients are added up before the one
  optimiser step.

Every loss here is a mean over positions or users, so the chunked loss and gradient are the same function.
Tests in the library take one step whole and chunked and require the same loss and every gradient to 1e-5,
for GRU with and without dropout, SASRec with 1 and 4 negatives, and ELSA with and without candidates. Four
mutants (a wrong scaling in each, and a lost gradient) are caught. The library's full suite passes: 2,464
passed, 1 skipped, run twice.

**Effect.**

Peak GPU memory of one training step, measured on the laptop's RTX 4050:

| Model (test size) | Whole | Chunked |
|---|---|---|
| GRU: 64 users × 50 positions × 60k items | 2.12 GB | 0.19 GB |
| SASRec: 32 users × 200 positions × 256 negatives × d 128 | 2.42 GB | 0.26 GB |
| ELSA: 512 users × 200k items, 256 factors | 2.68 GB | 1.09 GB (the rest is parameters and optimiser state) |

On the DGX's 32 GB V100s, at the default chunk size:

| Setting | Before | After |
|---|---|---|
| GRU on Yambda, batch 512, window 200 | ~152 GB | a few GB |
| SASRec's corner | ~40 GB | ~2 GB |
| ELSA on Toys, 2,048 factors | ~34 GB | ~25–28 GB, fits on a free GPU |

So the search spaces stay as they are, and no trial should fail for memory (H06).

**Side effects.**
- When one chunk holds everything, which is the usual case, the code and the results are exactly as before.
- When chunking starts, only the order of floating-point additions changes. Adaptive optimisers (Adam, NAdam)
  divide by a gradient's own recent size, so rounding on near-zero gradient coordinates can grow over many
  steps: a chunked fit is not bit-identical to an unchunked one, as two GPUs are not. That is why the tests
  compare one step's gradients rather than whole fits.
- Runs record the library's source hash (entry 17), so results from before and after this change show up as
  two builds in the report.
- The branch now carries eight uncommitted files. It has to be committed before it is moved to the DGX
  (entry 12).

**Where.** compresso-recsys: `models/simple_rnn.py`, `models/sasrec.py`, `models/elsa.py` and their tests
(`tests/test_simple_rnn.py`, `tests/test_sasrec.py`, `tests/test_elsa.py`). The memory probe is in
`review-plan/findings/S1-rest/chunk_memory.py`.

## 20. Comparisons count the training seeds (2026-09-30, H23, option C)

**Decision (Mark).** Option C: the seeds become a source of uncertainty in every test and interval. The design
defaults were used for the points left open (`review-plan/plans/seed-aware-test.md`).

**Change** (`seedstats.py`).
- The estimate is unchanged: the difference of the seed-averaged per-user values.
- Its standard error adds each side's seed-to-seed spread, var(seed means)/k, to the users' var(d)/n. The p-value
  comes from Student's t with Welch–Satterthwaite degrees of freedom.

Where it is used:

| Place | Test | Interval |
|---|---|---|
| Stage 1, against the reference and against the floor | seed-aware t-test, Holm within the dataset | two-level bootstrap (users and seeds resampled, 9,999 draws) |
| Ablation, each level against the full data | seed-aware t-test, Holm across the sweep | t interval |
| Ablation, the gap | — | t interval |
| The knee | one-sided seed-aware t-test | its power counts the seeds too |

The users-only p stays beside each result as a second column.

**Why.** Averaging over seeds and resampling users only answers whether *these* trained models would differ on
other users, not whether another training run would give the same winner. In the design's example, A leads by
0.002 only through one lucky seed: users-only p ≈ 0.06, seed-aware p ≈ 0.74.

**Effect.**
- With seeds that agree, results are as before.
- With seeds that disagree, fewer differences are significant. At k = 3 the degrees of freedom can fall to about
  2, so a difference must clearly exceed the seed spread.
- A simulation in the tests checks the calibration. With no true difference but seed noise, the seed-aware test
  rejects ≤ 8% at nominal 5%, while the users-only test rejects ≥ 25%.

**Side effects.**
- Works for any number of seeds per model, including 1, and for deterministic models, which add no seed term.
  The floor baseline has one evaluation.
- The two-level bootstrap is slightly conservative: each seed's own per-user noise is drawn again with the seeds.
- Its CPU cost is seconds per stage-1 comparison. The ablations use the t interval to avoid the bootstrap's cost
  over hundreds of comparisons.
- `holm` now lives in `seedstats.py`.
- No run has to be repeated: the per-seed per-user values were already saved.

**Where.** `seedstats.py`, `report.py`, `ablation_report.py` (`find_knee(..., seed_means=)`); tests in
`tests/test_seedstats.py` (7 tests; 4 mutants caught) and the smoke report.

**Open, since settled.** Adding seeds later (3 → 5) was not free: `[protocol].seeds` was part of every run's
fingerprint. Entry 21 takes it out.

---

## 21. Seeds can be added to runs already made (2026-09-30, follows entry 20)

**Decision (Mark).** Start with 3 seeds and run everything, then add seeds if there is time. Only the first
seed enters the fingerprint, and a CLI parameter adds seed runs to an existing run.

**Change.**
- **Fingerprints** (`protocol.py`). `seeds` left `_RESULT_KEYS`; the settings carry `trial_seed` = the first seed
  instead.
  - The first seed seeds every search trial, so it decides which configuration is selected. Changing it still
    starts everything over.
  - The other seeds each add one final run of that configuration, in a directory of its own
    (`final-seed<s>`), so the fingerprint does not need them.
  - Run, baseline, condition and ablation fingerprints are now the same for `seeds = [0, 1, 2]` and
    `[0, 1, 2, 3, 4]`.
- **`final --add-seeds S …`** (`runner.py`, `cli.py`).
  - Records the seeds in `runs/<dataset>/added_seeds.json` (which seeds, when, on which host), for each
    selected dataset, then runs the finals as usual.
  - From then on the dataset's seeds (`final_seeds`) are the protocol's followed by the added ones, for every
    command and report: `plan`, `status`, `final`, `report`, `repeat-strata`, the code-build count, and the
    ablations' reference.
  - The record covers every model of the dataset, even when `--model` narrows what this command runs. The
    others run the seed at their next `final`.
- **`ablate --add-seeds S …`** (`ablations.py`, `cli.py`).
  - Records the seeds per sweep and dataset in `ablations/<sweep>/<dataset>/added_seeds.json`, for every model of
    the sweep.
  - Each seed must already be a stage-1 seed of the dataset, because a seed's reference is that seed's stage-1
    final model. Every sweep and dataset of the command is checked before any is recorded.
  - A sweep's seeds (`sweep_seeds`) can be fewer than stage 1's. Its reference then uses only its own seeds, so
    both sides of every comparison have the same k.
- **Random sweeps** (density, repeat removal, shuffle, stratified catalogue).
  - An added seed adds a subsample at every level, fitted with that model seed, as the first three are.
  - The fixed test users now record the subsamples they were checked on (`test_rows.json`). A new subsample is
    checked against them, not intersected into them: runs already made were scored on those users, so all of
    them must stay eligible.
  - For every transform registered today they stay eligible by construction. The transforms drop or reorder
    events inside a fixed item space, so a next item stays recommendable, and a non-empty history keeps at least
    one event.
  - A future transform that broke this would be refused with the count of users lost, before the seed is
    recorded, not scored on fewer users.
  - The stratified catalogue scores each condition on its own users, so nothing is shared to check.
  - The floor and controls are also per subsample: `ablate` says to run `analyse --sweep <sweep>` for the new
    ones.
- **Order** (`runner.execute`). A reference or rescore whose stage-1 final is not made yet now returns
  `waiting-for-stage1` instead of writing `failed.json`. Before, running `ablate` ahead of `final` recorded a
  failure that needed `--retry-failed`; now a later `ablate` simply picks it up.
- **Reports.** A dataset or sweep with added seeds says so in one line. The stage-1 table shows `2/3` in the
  seeds column, and a note, for a model that has not run an added seed yet. A random sweep's condition enters the
  comparisons only once every seed of the sweep has finished, as before.

**Why.** The number of seeds is a budget decision that may change once the first round of results is in.
Before, adding a seed changed every fingerprint and restarted the whole suite, the search included, although
nothing a finished run computed depends on the seeds that come after it.

**Effect.**
- Adding seeds costs only the new runs: one final per model and seed, and per sweep one condition run per level
  and seed plus one reference rescoring.
- Results of the seeds already run are unchanged, and so are their files.
- With more seeds, each model's seed term in the seed-aware test (entry 20) shrinks and its degrees of freedom
  grow.

**Side effects.**
- This change itself changes every fingerprint, once: `seeds` left them and `trial_seed` entered. No DGX run
  exists yet.
- Editing `seeds` in `protocol.toml` now also adds seeds to every dataset, without new fingerprints, as long as
  the first seed stays first. Both ways can be combined; a seed listed in both counts once.
- The records are keyed by dataset (and sweep) name, not by fingerprint, so they carry over to fresh runs after a
  protocol change. Deleting a record takes its seeds out of every command and report again; their runs stay on
  disk, unused.
- Different datasets can have different numbers of seeds. Within a dataset, all models share them.
- Two processes adding the same seeds at once write the same record. Adding different seeds from two processes at
  the same moment could lose one of the two additions, so add seeds from one command and start the second GPU's
  command after it.

**Where.** `protocol.py` (`_RESULT_KEYS`, `_result_settings`), `runner.py` (`added_seeds_path`, `final_seeds`,
`record_added_seeds`, `add_final_seeds`, `execute`), `ablations.py` (`sweep_seeds`, `add_sweep_seeds`,
`check_stage1_seeds`, `check_added_seeds`, `data_seeds`, `fixed_test_rows`, `plan_ablation`), `cli.py`,
`report.py`, `ablation_report.py`, `analysis.py`, `strata.py`. Tests: `tests/test_seeds.py`, 11 tests on
a workspace that runs 2 seeds, adds a third to stage 1 (first for one model only) and to two of three sweeps.
12 mutants tried, 11 caught. The survivor drops the pre-check's own record, so `fixed_test_rows` checks the new
subsample a second time: slower, not wrong.

---

## 22. The tests can train on the GPU (2026-09-30)

**Change.** `SEQREC_EVAL_TEST_DEVICE=cuda pytest` runs every search, final and ablation run of the workflow
tests on the GPU; the default stays the CPU. `DEVICE` in `tests/test_smoke.py`, used by the workspaces of
`test_smoke.py`, `test_ablations.py` and `test_seeds.py`. Checks that call a single run directly stay on the CPU.

**Why.** The DGX trains on CUDA, and until now the whole workflow had only run on the CPU in tests. A device bug
(a tensor left on the wrong device, say) would first have shown on the DGX.

**Effect.** On the laptop's RTX 4050 all 160 tests pass on either device; the run records confirm the GPU was
used (every final, reference and rescore, and every trial but the one a test runs directly on the CPU).

**Side effects.** None for the default run. It does not test the V100 itself (sm_70 kernels in the cu126
wheel): that is the matrix-product check on the DGX.

---

## 23. A quick real run on the laptop, and `analyse --sweep none` (2026-09-30)

**Change.**
- `scripts/local-run.sh DATASET MODEL …` runs every step on one real dataset and the models named, each step
  under a memory cap (6 GB by default), against the library's branch checkout.
- By default it writes its own protocol: `protocol.local.toml` with 2 trials per model, seed 0, and 1 epoch for
  every trained model. `FULL=1` uses `protocol.local.toml` unchanged; `DRY=1` prints the commands.
- `analyse --sweep none` analyses the full data only. Plain `analyse` analyses every sweep's conditions too.

**Why.** Before the DGX, Mark wants to see the whole chain run on real data. The quick protocol keeps that to
minutes rather than the hours of 20 trials, and the full-data-only analysis avoids scoring the baselines at
every condition of every sweep.

**Effect.** None on any real run. The quick protocol has fingerprints of its own and lives in its own work
directory (`work-local/<dataset>`), so its numbers cannot be mistaken for results or mixed with them. Its split
has the real protocol's dataset fingerprint: it is the split a real run would prepare.

**Side effects.** `none` is accepted by `--sweep` only; `--dataset none` and `--model none` are still refused.
The script needs `systemd-run` for the cap (`MEM=none` runs without it).

**Where.** `scripts/local-run.sh`, `cli.py` (`_select`); `tests/test_local_run.py` (5 tests: the quick protocol
differs from the local one only in those settings, the steps and their selection, the refusal without a model,
`--sweep none`). 3 mutants of the script caught.

---

## 24. Fixes from the first real run on the laptop (2026-09-30, ML-20M)

Mark ran `scripts/local-run.sh ml20m elsa gru`, with the `history_length_inference` sweep. Every step finished,
and the numbers agree across the tables. These are the problems it showed, all fixed (Mark: "fix the problems, big
and small").

**1. Baselines ranked ties arbitrarily, so a condition and the full data scored differently.**
- *Seen.* Markov scored 0.0230 on the full data and exactly 0.0226 at every level from 1 to 1000. It reads only
  the last item, which truncation keeps, so all 11 values should be equal; `markov_backwards` showed the same.
- *Cause.* The two ways the evaluation excludes seen items ask for lists of different lengths. On the full data
  the model masks them itself and returns the top 10. In a condition, seen items come from the full history (H33),
  so the model is asked for a longer list and they are removed after. torch's top-k orders equal scores
  arbitrarily, and differently for a different length. On synthetic data the two paths agreed on the score in every
  slot but on the items in 0% of rows. Markov ties constantly: successors seen once each have equal probability.
- *Change* (`baselines.py`). Every baseline ranks in one total order: score, then the more popular training item,
  then the lower index (`_top_k`, exact, no float nudging). A shorter list is now always the start of a longer one,
  so both paths return the same list.
- *Effect.* Baseline numbers change a little wherever ties reached the top k: the floor, the controls, the
  sequence signal. Only on `exclude_seen = true` datasets (ML-20M, Amazon) did the two paths differ; the learned
  models almost never tie (GRU at levels ≥ 50 already matched the full data for all 2,942 users).
- *Side effects.* `BASELINE_VERSION` (= 2) now enters every baseline fingerprint, and `CONTROLS_VERSION` is 3, so
  results of the old ranking are not reused: `analyse` runs the baselines again. The models' runs are untouched.
- *Speed, fixed the same day.* The first version ranked every row in full with numpy. Mark's rerun showed the
  baseline steps about twice as slow (a condition's analysis 15 s → 28 s); measured alone, that ranking was
  100–200 times slower than torch's top-k (1.7 s against 8 ms for 1,024 × 24,093 scores), which the DGX's larger
  catalogues would have turned into hours. `_top_k` now takes torch's top k + 64 and orders those candidates
  exactly; only a row whose tie at the k-th value runs past them is ranked in full (`_top_k_full`). The lists are
  the same (tested against a brute-force order, with rows that overflow), at torch's speed (11 ms for that batch).
  The rerun's numbers were made with the full ranking and stand as they are.
- *Verified on the rerun* (Mark, ML-20M): Markov 0.0237, `markov_backwards` 0.0211 and `markov_shuffled` 0.0166 at
  the full data and at every level; the models' numbers unchanged (ELSA 0.0235, GRU 0.0403); no ⚠; the report
  names `0.3.7+trainusers` and the protocol's path and hash; the logs name each ablation run.

**2. The shuffled-Markov control moved between the levels of an inference sweep by chance.**
- *Seen.* `markov_shuffled` ranged from 0.0155 to 0.0178 across levels while Markov stayed put.
- *Cause.* Each condition drew its own shuffle of the training histories, although an inference sweep's training
  data is the full data's at every level.
- *Change* (`analysis._control_stream`). An inference sweep's conditions use the full data's shuffle, as the
  reference already did. A sweep that changes the training data keeps a shuffle per condition.
- *Effect.* The control of an inference sweep is now the same model at every level, so the sequence signal
  (Markov minus its shuffled control) no longer carries shuffle noise between levels.

**3. The manipulation check flagged what history truncation does by construction.**
- *Seen.* ⚠ on the test catalogue and popularity Gini at almost every level.
- *Change* (`ablations.py`). `history_length` declares `catalogue` and `popularity_gini` as expected to move,
  beside density and repeats: a history of *n* events holds at most *n* items. ⚠ is left for real surprises.
- *Effect.* Report only; no fingerprint changes. The values are still shown.

**4. The report named the wrong library version.**
- *Seen.* "compresso-recsys 0.3.6" for the `0.3.7+trainusers` branch. The source hash was right.
- *Cause.* A stale `src/compresso_recsys.egg-info` of 2026-09-24 in the cr checkout. With `PYTHONPATH` on
  `src`, Python's metadata lookup finds it first.
- *Change.* Deleted it: it was ignored build output (`*.egg-info/` in cr's `.gitignore`), and cr's own `.venv`
  keeps its editable install's metadata in `site-packages`. And `library_provenance` (`splits.py`) now takes the
  version from the checkout's `pyproject.toml` when the library is imported from one, keeping the metadata's
  version beside it as `metadata_version`.
- *Effect.* Records made from now on say `0.3.7+trainusers` (metadata: the `.venv`'s `0.3.7`). Records already
  written keep what they said.

**5. Two labels that could mislead.**
- The stage-1 report named the protocol by file name only, which for a quick local run is also
  `protocol.toml`. It now gives the full path and the file's sha256.
- The logs called every ablation run `final-seed<s>`, the name of its directory, which reads like a stage-1 final.
  They now name the sweep, the condition and what the run does (`history_length_inference 5: rescore (stage-1
  model) seed 0`), and show settings only for runs that train.

**Not changed.** Latency differed between the two runs (ELSA P95 15.6 and 20.7 ms): it is measured afresh each
time, and the laptop was busy. On the DGX, measure on a quiet machine with `--cores`.

**Where.** `baselines.py`, `protocol.py` (`BASELINE_VERSION`), `analysis.py`, `ablations.py`, `splits.py`,
`report.py`, `runner.py` (`RunSpec.label`). Tests: `tests/test_baselines.py` (6: both exclusion paths give one
list for all four baselines; the tie order; `_top_k` against a brute-force order), `tests/test_ablations.py` (Markov
equal at every truncation with seen items excluded; the inference control; the log labels; the baseline code
version in the fingerprint), `tests/test_smoke.py` (the checkout version; the protocol path and hash). 179 tests;
13 mutants, all caught (9 for the fixes, 4 for the fast ranking, one of them a speed-only change caught by a test
that the full ranking stays the exception).

**Mishap during the fix.** Writing the new tie tests replaced the existing `tests/test_baselines.py` (21 tests).
It was rebuilt the same hour from the session records: its creation on 2026-09-24 and every later edit, replayed
in order. It is exact as far as can be checked: all 18 test functions start on the lines the records quote, as do 7
lines quoted in later tracebacks, and the suite came back to its earlier 165 tests before the new ones were added.

---

## 25. Three model decisions after the final review (2026-10-01, N5, N2, N7)

**N5, matrix inputs: kept as the library builds them (Mark: "if it came with the models, keep it").**
- *What it is.* cr's builder (`builder.py`, `x_train = train_source.maximum(train_target)`) trains ELSA and EASE
  on the larger of the two training windows' counts per user and item. At test they read the source matrix, the
  sum over the whole history. The suite's refit mirrors the builder.
- *Where it came from.* The library, not the suite. It was written by the library's author (one of ELSA's
  authors) on 2026-08-14 for the cold-start work. ELSA and EASE use the values as given: ELSA reconstructs the
  counted row, EASE takes the Gram of counts.
- *Effect.* None on ML-20M and Amazon (one event per item). On Music4All, Yambda and OTTO, a model reads larger
  counts at test than in training; 13% of input cells differ on the smoke data. Documented in the README as
  library behaviour, to name in the thesis.

**N2, SASRec's negatives: kept as published, documented.** Negatives are sampled from outside the whole history,
so a seen item is never pushed down, which holds back re-consumption on repeat-heavy data (0.64 of the oracle on
a planted chain, findings/H53). Mark may later add an option to reuse the history. A "Decided" line is in both
protocol files.

**N7, `unk_dropout`: searched.** `{choice = [0.0, 0.02, 0.05, 0.1]}` for SASRec (default 0: the embedding of
items first seen after training was never trained) and, so neither model has a choice the other lacks, for GRU
(library default 0.05).
- *Effect.* New trial configurations for both. Their fingerprints change, but no run of the protocol existed yet.
- *Where.* `protocol.toml`, `protocol.local.toml` (still identical apart from the batch size).

---

## 26. Fixes before the DGX run: what gets computed and saved (2026-10-01, final review)

Mark: "fix the rest of the issues". Evidence for each finding is in `review-plan/findings/final-review.md`.

| Issue | Change | Where |
|---|---|---|
| N19 (A4) | Every command that runs or scores refuses a split prepared from other build settings: `load_split(..., protocol)`. | `splits.py`, `cli.py`, `strata.py` |
| B10 | The dataset fingerprint covers the build settings only. `exclude_seen` and `new_item_diagnostic` moved out (`_SCORING_ONLY_KEYS`), so flipping one neither rebuilds the split nor redoes the analysis. Every scored result still carries them: run and baseline fingerprints hold the whole section, and the controls fingerprint now does too. | `protocol.py`, `analysis.py` |
| N26 (B9) | Allowed keys per section. A misspelt key is refused by name, with a "did you mean". Seeds ≥ 0; latency bins positive and increasing. | `protocol.py` |
| C4 | Under `exclude_seen`, a next item already in the history is out of reach, so it does not count as recommendable. The `new` diagnostic picks its users on its own targets, against the whole history. The evaluation key holds the dataset's `exclude_seen`, so the fixed users and profiles follow it. `SCORING_VERSION` → 3. | `evaluate.py`, `ablations.py`, `protocol.py` |
| A6 | Every evaluation excludes seen items after the model ranks, against the whole history: the full data and stage 1 as the ablation conditions already did. Tied scores now fall the same way in both. Nothing changes for a model without ties. | `evaluate.py` |
| N18 (A3) | A finished final (or ablation run) whose saved trial or settings differ from the current selection is refused as `stale-selection`. A reference or rescore is refused before it reloads a stage-1 final made under another selection. The report flags such finals. `--allow-incomplete` is recorded (`incomplete_selection.json`) and reported. | `runner.py`, `report.py` |
| N23 (B2) | Locks are `fcntl.flock`, released by the kernel when their process ends, however it ends. No process ids, no stale claims, no reclaim race. `done.json` is re-checked after locking (H45). | `runner.py` |
| N24 (B3) | `attempts.json` counts starts. A run whose process died twice without a word becomes `failed.json` ("ProcessDied"). Ctrl-C does not count. | `runner.py` |
| N25 (B8) | Exit codes: 0 all done, 1 a run failed or was refused, 3 work left. A reference whose stage-1 final failed reports `stage1-failed`, not waiting for ever. | `cli.py`, `runner.py` |
| N27 | • The NaN-trial message says a rerun repeats it (B4).<br>• fsync before every rename, and a broken JSON file is named on read (B5).<br>• The library's temp files go under the work dir unless `TMPDIR` is set (B6).<br>• Thread variables for numpy in the README's launch lines and in `local-run.sh` (B7).<br>• `prepare` takes a lock per dataset (B11). | several |
| N22 (B1) | The baselines rank without a rows × items array: each row's evidence items (Markov's successors, sorted once per item; Replay's history), then one global popularity order, less the row's own items and seen ones. **The same lists and values as the dense path** (tested across all four baselines). Amazon-sized batches (1,024 × 517k) went from ~7 s and 12 GiB to 23–60 ms, under 1 GiB. | `baselines.py` |
| C6 | `latency.json` records the machine's load and the source trial; the worst-bin P95 needs 50 requests in the bin. | `latency.py` |

**Side effects.**
- Every dataset fingerprint changes once (B10), and with it every condition and analysis fingerprint. The
  `SCORING_VERSION` and evaluation-key changes add to that.
- Local work dirs from before need `prepare --force` or a fresh start. No DGX run exists yet.
- Under `exclude_seen`, users whose next item they had already seen are no longer scored. The synthetic test
  data loops, so none of its users stays scored then, and the exclusion tests moved to window targets or to
  comparing lists.

**Tests.** `tests/test_robustness.py` (new, 17), and additions to `test_smoke.py`, `test_ablations.py` and
`test_baselines.py`. 224 tests on CPU and on GPU.

---

## 27. Report-only fixes: one test family, paired seeds, the gap against ELSA (2026-10-01, final review)

None of these changes a run; every report can be rebuilt from the saved results.

| Issue | Change |
|---|---|
| N16 (A1) | In ablations the seeds are paired: level-vs-full, the knee, and the gap in random sweeps (shared subsample). The seed term is var(per-seed differences)/k with k − 1 df. Before, an identical level could fail "within δ" (p 0.089 at seed means 0.037/0.040/0.043). |
| N17 (A2) | Stage-1 intervals are the seed-aware test's t interval. The two-level bootstrap is gone: with 3 seeds it was 2.6 times too narrow, and it disagreed with the p-values. |
| N29 statistics | **One test family.** The users-only p is the same t-test without the seed term (the paired t-test over users), not the library's randomisation test. `compare_models` is no longer called, nor are its 9,999 resamples and warnings. Its target check moved into `seedstats.compare`: results scored against different targets are refused. |
| N20 (A5) | Failed and unfinished ablation runs are listed per model, and a model without a knee says why. |
| N29 gap | **The gap is sequential − ELSA**, fixed in advance, tested and Holm-corrected across the sweep; it is the plot. The gap to the best non-sequential model, chosen on test, stays as a descriptive table. `ablation-report --reference` picks the comparator. The floor line on the plot is floor − ELSA. |
| N29 knee | Described as the equivalent interval rule: the most reduced level whose 90% interval for the loss stays above −δ, walking down from the full data. A sensitivity table gives the knee at δ = 5/10/20%, marked as not the result fixed in advance. |
| H51 | The stage-1 report ends with a table across datasets, per model: the sign of each difference, * where significant, and counts of better / not different / worse. Nothing is pooled. |
| C1 | Declarations can name one part (`"test catalogue"`). Density declares the test catalogue (the training one is held by `keep_catalogue`; both if it is off). Repeat removal declares popularity Gini. |
| C2 | A random sweep's baselines and floor at a condition are shown only once every subsample is analysed, otherwise marked partial. The floor needs every baseline. |
| C3 | A level below `min_level_users` gets "— (descriptive)" in place of a verdict, in every test and in the knee. A random catalogue shows its users per seed beside the union. A one-seed test is labelled users-only. |
| C5 | Stale text rewritten: the gap plot (title and docstring), the ablation report's docstring, the README's knee and power, and the margin comment. |

**Effect.**
- Ablation knees and level tests become sharper where seeds agree level by level, as they do in inference sweeps.
- Stage-1 intervals become wider and consistent with their p-values.
- The headline gap answers the research question directly.

**Where.** `seedstats.py` (rewritten), `report.py`, `ablation_report.py`, `ablations.py`, `plots.py`, `cli.py`.
Tests: `test_seedstats.py` (paired seeds, coverage of the t interval, the users-only p equals scipy's paired t,
the target check, the knee with paired seeds), `test_ablations.py` (the report's seed term against the paired
formula from the run files, listed failures, partial floors, descriptive labels, declarations),
`test_robustness.py` (the cross-dataset table). 30 mutants over this batch and entry 26, all caught.

**Still open, Mark's call.** SANSA in or out of the research question (it can be added later: fingerprints are
per model). A retuning spot-check after the main runs.

---

## 28. One exclusion width per phase (2026-10-01, change review N30)

**Found by** the independent change review of entries 25–27
(`review-plan/findings/2026-10-01-change-fixes-25-27.md`), class R. Mark approved the fix.

**Problem.** Entry 26's A6 sent every `exclude_seen` evaluation through `ExcludeSeenPolicy`. The policy asks the
model for k plus the most items **its batch's** heaviest user has seen, then removes the seen items. torch's top-k
breaks ties differently for a different length, so a model whose scores tie got lists that depended on who
shared its batch. That made them depend on:
- `eval_batch_size`, which is in no fingerprint (128 locally, 1,024 on the DGX);
- which rows were scored: all of them in stage 1, a fixed subset for an ablation's reference.

cr's popularity model, on the same users: nDCG@10 0.02026 at batch 1,024, 0.01954 at 128. Entry 26's "tied
scores now fall the same way" held for the suite's baselines (one fixed tie order, entry 24) but not for the
library's models. That breaks M3: every number from the current protocol and code, not from a speed setting.

**Change** (`evaluate.py`). `evaluate_phase` computes the width once per phase: the largest row of the whole
phase's seen history, not the batch's and not the scored subset's. It passes that as `extra` to the policy, so
every batch asks for `k + extra`. The policy refuses an `extra` smaller than a batch row's seen count. Without
`extra` (latency, which excludes inside the model), nothing changes.

**Effect.**
- The exclusion no longer depends on the batch. For a model whose per-row scores do not depend on the batch, a
  user's list depends on its own scores and history only: identical at every batch size, in every subset, and
  between stage 1, an ablation's reference and its conditions, since they share the phase's seen history.
- ELSA is not such a model. It builds its input from the batch's union of source columns, so its scores move by
  ~1e-6 with the batch, and on CUDA 1 row in 512 changed its top 20. That is a separate, noise-level effect,
  parked as N48 (X, R?) by the re-review.
- Models without ties are unchanged.

**Side effects.**
- `SCORING_VERSION` → 4, so results made under entry 26's code are not reused: every run and condition
  fingerprint changes. No DGX run exists; local work dirs need a fresh start.
- Each batch asks for a longer list: k plus the phase's heaviest history, e.g. ~5,000 on ML-20M. On ML-20M and
  Amazon every evaluation goes through the policy; Music4All, Yambda and OTTO only through the `new` diagnostic.
  The policy's own peak per batch of 1,024 is about 0.4 GB at width 5,000 and 0.8 GB at 10,000 (~75 B per
  entry; measured by the re-review), and ~0.35 s. That peak is no higher than before whenever the heaviest user
  is scored, since that user's batch already asked for that width; only the number of batches paying it rises.
- The fix relies on torch's top-k ordering each row the same way for the same k whatever the batch's shape. The
  tests confirm it on CPU, at batch sizes 1, 16 and 400 (lists) and 1, 7 and 128 (an evaluation). The re-review
  confirmed it on CUDA too, on the laptop's RTX 4050 (batch sizes 1 to 1,024, n up to 200,000, k up to 30,000,
  where CUDA switches algorithm). Not yet on the V100: a CUDA test is proposed (N47).

**Where.** `evaluate.py` (`ExcludeSeenPolicy(extra=)`, `evaluate_phase`), `protocol.py` (`SCORING_VERSION`).

**Tests.**
- `tests/test_evaluate.py`: a tie-heavy library model's lists are identical across batch partitions with one
  width, and differ with per-batch widths, so the data does tie; a too-small width is refused.
- `tests/test_smoke.py`: every batch asks the model for the same length, at batch sizes 1, 7 and 128 and on a
  subset without the heaviest users; the subset's values equal the full evaluation's.

226 tests on CPU and on GPU. 4 mutants, all caught. One of them first survived, because the subset happened to
hold the heaviest user; the test now excludes that user.

**Independent re-review** (2026-10-01, fresh-context reviewer, rule 5 of the review framework): no R, P or O.
- It instrumented a whole synthetic pipeline: every call path, model, baseline and sweep. All 156 call groups
  asked the model for exactly one width per phase.
- 5 mutants, all caught.
- Its three low findings and three parked items are N46–N49 in ISSUES.md. This entry's text was corrected from
  its finding H2: the memory figure, and the ELSA caveat.

---

## 29. The library is vendored; the suite is its own repository (2026-10-01)

**Decision (Mark).** He did not want the local library changes pushed to compresso-recsys. Instead the suite
carries its own copy of the library and becomes a repository of its own.

**Change.**
- `vendor/compresso-recsys` is a copy of the checkout's package as the suite runs on it:
  - release 0.3.7 plus the local branch `local-temporal-train-all-users` (committed and uncommitted), version
    `0.3.7+trainusers`;
  - `src`, `tests`, `pyproject.toml`, `README.md` and `LICENSE`;
  - the library's data, artifacts, docs and examples are left out.
- `scripts/vendor-cr.sh` makes the copy and writes `VENDORED.md` (source, branch, commit, upstream compared
  against, every changed file) and `CHANGES.patch` (the full difference from upstream). It puts a one-line notice
  at the top of every changed Python file. The library is Apache-2.0, which allows a modified copy if the license
  comes with it and changed files say so (§4).
- The suite's `pyproject.toml` requires `compresso-recsys==0.3.7+trainusers` from that folder
  (`[tool.uv.sources]`), and `uv.lock` was updated to match. Only that entry changed.
- The suite folder is a git repository (`.gitignore`: results, environments, local settings, scratch notes and
  the `build/` folder `uv sync` leaves in the vendored copy are left out).

**Why.**
- Installing became one step on every machine: `uv sync`.
- Before, the DGX needed the branch committed, a `git+file` install over the locked release, and a rule never
  to run `uv sync` again, since it put the release back. All three are gone.
- What is installed is exactly what was reviewed, and it travels with the suite.

**Effect.**
- `uv sync`, `uv run` and plain `pytest` are safe everywhere. Tests no longer need a `PYTHONPATH` to the
  checkout: 226 pass on CPU and on GPU against the installed copy, and 226 in a fresh clone after `uv sync`.
- The library's own tests pass on the copy: 1,726. The rest read the library repository's docs and examples,
  which are not copied.

**Side effects.**
- The library's source hash in run records changes once, because of the notices. Any run made before records
  another build, and the report would warn about mixed builds. No DGX run exists; local work dirs need a fresh
  start anyway (`SCORING_VERSION` 4).
- The checkout at `~/Documents/recombee/compresso-recsys` is no longer what the suite runs. A change there
  reaches the suite only through `scripts/vendor-cr.sh` and `uv sync`, followed by a change review, since
  `tools/review-scope.sh` now snapshots `vendor/` and `uv.lock` too.
- `scripts/local-run.sh` no longer puts the checkout on the path by default (`CR_SRC` is empty).

**Where.** `vendor/compresso-recsys/`, `scripts/vendor-cr.sh`, `pyproject.toml`, `uv.lock`, `.gitignore`, the
README's Install section, `scripts/local-run.sh`, the review skill and `review-plan/tools/review-scope.sh`.

---

## 30. Four fixes before the DGX: N49, N31, N34, N33 (2026-10-01)

**Decision (Mark).** Of the change review's open findings, fix these four before deploying. The others stay
open (ISSUES.md). There is no change review of these fixes, at Mark's request.

**N49: a catalogue too small for a history stopped the run.**
- *Problem.* Under `exclude_seen`, a user who has seen all but fewer than k items of the training catalogue
  cannot be given k unseen ones, and `ExcludeSeenPolicy` stops the whole evaluation ("fewer than k unseen").
  At a catalogue sweep's level 0.1 on ML-20M or Amazon, the heaviest users do exactly that.
- *Change.* `evaluate.fills_list`: per row, does the training catalogue hold k = max(cutoffs) items the user
  has not seen? Under `exclude_seen`, `evaluate_phase` leaves out the rows that do not, and counts them in the
  run's metadata (`rows_too_few_unseen`, beside `rows_unrecommendable_next`). `own_test_rows` and
  `_eligible_test_rows` apply the same rule, so the users a condition lists are the users it scores; they
  refuse `exclude_seen` without k.
- *Effect.* The run goes on, without those users.
- *Side effects.* No fingerprint changes. Every result that exists is unchanged: the users left out are
  exactly those that stopped the run before. The fixed test rows of the other sweeps are unchanged too, since
  those sweeps keep the item space. A selection effect at the smallest catalogues is a limitation (below).

**N31: stopping a run counted as a death.**
- *Problem.* A `nohup` launch can only be stopped by `kill` (SIGTERM). That counted as a death like an
  out-of-memory kill, so two stops marked the run failed for good. And under `--retry-failed`, a run already
  at two deaths was marked failed without running.
- *Change.*
  - `cli.main` turns SIGTERM, and SIGHUP unless `nohup` ignores it, into the same stop as Ctrl-C. The run in
    progress is not counted, and its claim is released. The command prints why it stopped and exits 130.
  - `--retry-failed` starts the count of deaths over.
- *Effect.* Stop and restart as often as needed. Only SIGKILL and the out-of-memory killer still count.
- *Side effects.* Ctrl-C now ends with that message and exit code 130 instead of a traceback. No fingerprint
  changes.

**N34: the stage-1 floor came from whichever baselines had finished.**
- *Problem.* After an interrupted `analyse`, the floor was the strongest of the baselines with a result, so it
  could be lower than the real floor, silently.
- *Change.* `analysis.floor_of` requires `expected`, the protocol's baselines, and returns no floor until all
  of them have a result. The stage-1 report then says which baselines are missing; the analysis report marks
  no floor. The ablation report already required every baseline, and now uses the same rule.
- *Effect.* No floor rather than a wrong one.
- *Side effects.* None on runs.

**N33: runs made under an older selection were still counted.**
- *Problem.* A final made with another trial's configuration than the one selected now describes another model.
  The stage-1 report flagged it ⛔ but still averaged it into the table, the comparisons, the cross-dataset
  table, the CSV and the diagnostics. The ablation report loaded such runs without saying so.
- *Change.*
  - `runner.stale_final_seeds` lists those finals, and `load_final_evaluations` leaves them out. Every reader
    of the finals goes through it: the stage-1 report and its diagnostics, `repeat-strata`.
  - The stage-1 report lists them ⛔ and no longer calls them "not run yet".
  - The ablation report checks every run with `made_with_another_selection`, the runner's own refusal (made
    public). A stale run is left out and listed ⛔. Its level then lacks a seed, so it leaves the comparisons,
    the gap and the knee, as a failed run's level does.
  - Latency is not measured on a stale final (`latency` names it) and not shown in the table (a ⛔ note).
- *Effect.* Nothing from another selection reaches a value, a test or a table. `final` and `ablate` already
  refused to redo such runs in place, and still do.
- *Side effects.* No fingerprint changes. Every place that read the finals now also reads the trial summary.

**Where.**
- `evaluate.py` (`fills_list`, `evaluate_phase`); `ablations.py` (`own_test_rows`, `_eligible_test_rows`,
  their callers).
- `cli.py` (`main`, `_operator_stops`); `runner.py` (`execute`, `stale_final_seeds`,
  `load_final_evaluations`, `made_with_another_selection`).
- `analysis.py` (`floor_of`); `report.py`; `analysis_report.py`; `ablation_report.py` (`_load`,
  `dataset_ablation`); `latency.py`.
- README: the analysis steps, the per-condition users, the ablation report, "Unattended runs are safe", and
  the launch notes.

**Tests.** Seven new; 233 pass on CPU and on GPU. A deliberately broken version of each piece of each fix was
caught, 16 of 16. One (the SIGTERM handler not put back) survived at first, because the test fixture had
already left the handler installed; the test now installs its own first:
- `test_a_catalogue_smaller_than_a_history_leaves_out_the_user_not_the_run` and
  `test_a_condition_lists_as_its_users_only_those_it_can_give_k_unseen_items` (N49);
- `test_retry_failed_starts_the_count_of_deaths_over` and `test_kill_stops_a_run_as_ctrl_c_does_not_as_a_death`
  (N31);
- `test_the_floor_waits_for_every_baseline` (N34);
- `test_the_report_flags_finals_made_under_another_selection` (extended),
  `test_latency_is_neither_measured_nor_shown_for_a_final_of_another_selection` and
  `test_an_ablation_run_made_under_an_old_selection_is_left_out_of_the_report` (N33).

---

## 31. GPU runs on an explicit device failed at once (2026-10-01)

**Problem.** On the first DGX run, every run on `--device cuda:0` (the README's form) failed before any work
with "RuntimeError: Invalid device argument", popularity included. The runner resets torch's peak-memory counter
at the start of each run. Until CUDA has started in the process, that call refuses an explicit device, though
not plain `cuda`. The tests ran on `cuda`, in a process where CUDA had already started, so they never saw it.

**Change.** `runner._reset_peak_memory` starts CUDA (`torch.cuda.init()`, a no-op once started) before the
reset.

**Effect.** Runs on `cuda:0` and `cuda:1` work. No fingerprint changes; nothing had run.

**Where.** `runner.py`.

**Tests.** `test_a_run_on_an_explicit_gpu_starts_its_memory_counters_in_a_fresh_process` (skipped without a
GPU; it fails without the fix). 234 pass on CPU, and on the GPU with `SEQREC_EVAL_TEST_DEVICE=cuda:0`, which is
how the GPU tests should be run from now on.

---

## 32. The suite fits one to two weeks on one GPU (2026-10-05)

**Decision (Mark).** The whole suite must run in a week or two, not 2–4 months. Plan:
`review-plan/plans/week-budget.md`, B1–B4.

**Basis.** The DGX timing run on ML-20M (10 configurations per model × 2 epochs on a V100, 2026-10-01). Priced
from it, the protocol as written cost about 15 GPU-days per ML-20M-sized dataset, 90% of it SASRec: its epoch
grid reached 400, and its history length 400 costs 5× per epoch.

**Changes, in `protocol.toml` and `protocol.local.toml`:**
- **SASRec epochs** `[20, 50, 100, 200, 400]` → `[10, 20, 50, 100]`. An epoch is (training users ÷ batch)
  steps: about 1,000 on ML-20M against about 50 on ML-1M, where the published 200 epochs were set. So 100 epochs
  is still ~10× the published training in steps.
- **Trials** 20 → 10 per model. The first 10 draws are unchanged: they are exactly the configurations the timing
  run measured, apart from SASRec's epochs, which come from the new grid. Checked with `plan_trials`.
- **Ablations:**
  - every retraining sweep runs under seed 0 only (`seeds = [0]`, below);
  - fewer levels, 13 in all where there were 28: `history_length` `[5, 20, 100, 500]`, `density`
    `[0.25, 0.5]`, `repeat_removal` `[0.5, 1.0]`, `catalogue_top` `[0.25, 0.5]`; `shuffle` unchanged;
  - `catalogue_stratified` dropped from the protocol (its code stays);
  - the inference sweep keeps every level and seed: it only rescores.
- **Their datasets** are chosen after the per-dataset timing (plan A1). Until then they list every dataset.
- **Kept:** all 5 datasets and 3 seeds in stage 1; every model; SASRec's history length up to 400 (capping it
  saved only ~15%, and long histories are part of the question); the GRU and ELSA grids.

**A sweep's own seeds (code).**
- `[ablations.*]` takes `seeds`: a non-empty list of distinct stage-1 seeds. The default is the protocol's.
  Booleans, strings, repeats and seeds stage 1 lacks are refused at load.
- `ablations.sweep_seeds` starts from them.
- Not in any fingerprint (the condition fingerprint holds transform, levels, options and scope), so
  `ablate --add-seeds` grows a sweep later beside the runs already made.
- A random transform then draws one subsample per sweep seed.
- With one seed, the sweep's tests are over users only. The paired seed term needs k > 1, and the report already
  said so.
- `plan`, the ablation report's header and its added-seeds line name the sweep's seeds.
- `scripts/local-run.sh` sets a sweep's own seeds to the first quick seed. It rewrote every `seeds` line before,
  and refused the protocol once the sweeps had their own.

**Effect.**
- About 24 GPU-hours of stage 1 and 17.5 h of retraining ablations per ML-20M-sized dataset.
- About 8.6 days for the whole suite on one process if all five datasets cost what ML-20M does: 6.4 days with
  the retraining sweeps on two datasets.
- Two processes on one GPU (memory allows: at most 2.7 GB per run) may divide that by ~1.7, not yet measured.

**Side effects.**
- Every run fingerprint changes (trials, epochs), as do the condition fingerprints of the sweeps whose levels
  changed. No real run existed.
- The timing runs keep their own fingerprints and stay valid as timing.
- The review checklist's "full = stage 1" must compare seed for seed when a sweep has fewer seeds
  (REVIEW_FRAMEWORK §6).

**Where.** Both protocols; `protocol.py` (`_ALLOWED_KEYS`, `AblationProtocol.seeds`, `_parse_ablation`);
`ablations.sweep_seeds`; `ablation_report.py`; `cli.py` (`plan`); `scripts/local-run.sh`; README (Adding seeds,
Ablations, the guarantees, How long it takes).

**Tests.**
- `test_a_sweeps_own_seeds_must_be_stage1_seeds` and
  `test_a_sweep_runs_under_its_own_seeds_and_grows_with_add_seeds`: fingerprints unchanged; seed-0 plan; nothing
  rerun; report wording; `--add-seeds` reuses the runs.
- `test_the_quick_protocol_shortens_only_trials_seeds_and_epochs` (extended).
- 12 deliberate breaks, all caught.

---

## 33. EASE no longer copies its weights on every call (2026-10-05)

**Problem (the DGX timing run).** The library's EASE scored with `coefficients_[:, candidate_rows]`. With every
item a candidate (every suite call), that copies the whole items × items matrix on each call: 1.6 GB on ML-20M.
The result:
- 3.8 s per latency request;
- ~80 s per validation scoring;
- `latency` alone took 1.5 h;
- the latency table claimed EASE misses 100 ms by 40×. That was the copy, not EASE.

**Change (Mark approved).**
- In the cr checkout's working tree (`local-temporal-train-all-users`, not committed): with every item a
  candidate, `ease.py` scores with the matrix itself; a subset still selects its columns. Same numbers: a
  sorted, unique selection of all n items is `arange(n)`.
- Re-vendored with `scripts/vendor-cr.sh`.
- The script now also appends uv `cache-keys` to the copy's `pyproject.toml`
  (`[{ file = "pyproject.toml" }, { file = "src/**/*.py" }]`). By default uv reinstalls a path dependency only
  when its `pyproject.toml` changes, which a `git pull` of a library change leaves alone, so the DGX would have
  kept the old copy without a word.

**Effect.**
- A request on a 20k-item EASE: 2 ms, where the copy alone took 1.3 s.
- EASE's validation scoring and `latency` take seconds instead of minutes.
- EASE runs on the CPU, so the GPU time is unchanged.

**Side effects.**
- The library's source hash in run records changes: the vendored copy differs.
- The other models only select columns of each batch's scores, not of a weight matrix (checked; their latency
  of 2–6 ms agrees).

**Where.** cr `src/compresso_recsys/models/ease.py`, `tests/test_ease.py`; `vendor/compresso-recsys`
(re-vendored); `scripts/vendor-cr.sh`; README Install.

**Tests.**
- cr: `test_scoring_every_item_does_not_copy_the_weights`. The weights refuse indexing; scores are exactly
  equal; a subset still selects. It fails with the copy back.
- cr's EASE tests (22) and every cr test that touches EASE (127) pass.
- uv: a source-only change to the vendored copy is reinstalled by `uv sync --locked` (checked by hand).

---

## 34. The report flags runs still improving at their last epoch (2026-10-05)

**Why.** §32 caps SASRec at 100 epochs. If the cap binds, the model is under-trained, and that should show
rather than pass for its best.

**Change.**
- `report.still_improving`: a run whose training loss fell by more than 1% over its last tenth of epochs (at
  least the last one) was stopped while still improving. A model trained in phases is judged on its last phase.
  NaN losses are skipped. EASE and popularity have no loss per epoch.
- The stage-1 report adds a ⚠ note per model, naming the selected trial and the finals concerned and their fall.
- A diagnostic only: it changes no selection.

**Where.** `report.py`; README (the guarantees).

**Tests.** `test_a_run_still_improving_at_its_last_epoch_is_recognised` and
`test_the_report_flags_finals_still_improving_at_their_last_epoch`. 5 deliberate breaks, all caught after the
phase case was strengthened.

---

## 35. Two scripts: the time estimate and the DGX run (2026-10-05)

**`scripts/project-time.py`** prices a protocol from timing runs, per dataset and GPU model: search, finals,
retraining levels and rescorings, in GPU hours, with a worst case.
- **Cost per epoch:** measured where a configuration was timed, otherwise a log-linear fit over the
  configuration's sizes. Within 15–23% on the timed ML-20M configurations.
- **Other datasets:** priced by their measured ratio on configurations timed on both, or by a `--factor`.
- **Finals and refits:** at the mean planned configuration × the measured final/trial ratio. Refits also ×
  `--refit-share` (0.6).
- **Check:** on the ML-20M timing and §32's protocol it gives 206 h (8.6 days) with every dataset at 1×, the hand
  estimate.

**`scripts/dgx-run.sh`** runs the whole suite in order of importance:
1. `prepare`;
2. `search` and `final`, with the full data's `analyse` beside them;
3. the stage-1 report;
4. each sweep, with its `analyse` beside the next;
5. `latency` last and alone;
6. every report.

How it runs:
- One process per entry in `GPUS` takes the GPU models; one CPU process takes EASE and popularity.
- A step left with work (exit 3) is repeated, up to `ROUNDS`; a failure stops it with the log to read.
- A SIGTERM to the script reaches every process it started (they stop cleanly, §30).
- A log per process in `$WORK/logs`.
- `SWEEPS=none` runs stage 1 alone, so it can start before the ablation datasets are chosen.

**Where.** `scripts/project-time.py`, `scripts/dgx-run.sh`; README (Workflow, How long it takes).

**Tests.**
- `tests/test_project_time.py`: an exactly log-linear cost law is recovered; search, finals, refits, rescorings
  and the worst case are all checked; dataset ratios and factors; a timed configuration keeps its measured time.
  7 breaks, caught.
- `tests/test_dgx_run.py`, against a fake `seqrec-eval` that records calls and exits as told:
  - the order, devices and models;
  - latency starting only after every background analysis has finished;
  - a step repeated on exit 3;
  - a stop after `ROUNDS`;
  - a failure stopping before the next step;
  - SIGTERM leaving no process behind;
  - settings checked first;
  - `SWEEPS=none` running stage 1 alone.

  10 breaks, caught.
- 252 tests pass on CPU and on the GPU (`cuda:0`).

---

## 36. Decisions of 2026-10-06: Amazon sampled, the ablation datasets, BERT4Rec in, ELSA as it is

**Decisions (Mark), on the questions left open by entries 32–35.**

**Amazon at 25% of its users.**
- *Why.* At full size (428,896 training users, 447,069 items) ELSA cost 55 times its ML-20M cost per epoch on the
  DGX, and Amazon alone projected to about 8 GPU-days.
- *Change.* The library's Amazon loader gains `user_sample`, the hashed user sample Yambda's loader has (cr
  working tree, `datasets/amazon2023.py`, with a test; re-vendored). `[datasets.amazon]` sets
  `options = { user_sample = 0.25 }`.
- *Effect.* About 27,000 test users remain. Amazon's dataset fingerprint changes, so its split is prepared again.
  Its 6 finished ELSA trials are not reused, and nothing else of Amazon had run.

**The retraining ablations run on Music4All and OTTO.**
- *Why.* Both have enough scored test users for a 10% knee, and they contrast on almost everything: music
  against shopping, histories of hundreds against sessions of a dozen, heavy repetition against little.
- *Change.* Every retraining sweep lists `datasets = ["music4all", "otto"]`; `repeat_removal` lists Music4All
  alone. The inference sweep keeps all five. Not in any fingerprint (entry 21).

**BERT4Rec is compared too (H26).**
- *Change.* Registered (`models.py`; the trainer builds its own batcher, which has the `[MASK]` token). A
  `[models.bert4rec]` section is centred on the published settings: d 64, 2 blocks, 2 heads, mask proportion
  0.2–0.6, N 50–200, batch 256, lr 1e-4 (Sun et al. 2019, and the authors' run scripts as the library cites
  them).
  - The learning rate is searched around 1e-3, as SASRec's is, since the published 1e-4 under-trains with the
    fixed warm-up and decay (H53).
  - One epoch is 11 passes: the published 10 re-maskings plus the last-item samples. So `epochs` 1–10 matches
    SASRec's 10–100 passes.
  - `unk_dropout` is searched as for SASRec and GRU (N7).
- *Kept as published:* positions count from the oldest item (N1). Since the refit, that leaves a mask on an
  untrained position only for users whose history is exactly the longest training sequence, and only where no
  user fills the window. On ML-20M, Music4All and Yambda some user always fills it; on OTTO and Amazon it is a
  handful of users. A limitation, below.
- *Effect.* A sixth model, and the third sequential one. `scripts/dgx-run.sh` trains it on the GPU by default.
  Its cost is unmeasured: it is priced from its first trials on the DGX. The full softmax at up to 20 masked
  positions per sequence fits a 32 GB V100 at batch 256 even on Yambda's 122,162 items (about 8 GB), so no
  memory change was needed.

**ELSA runs only as Recombee runs it (N3, N4 closed).** The published variants, without the ReLU or with the
self-term subtracted, are not run.

**N43 and N48 are accepted as limitations.**
- *N43:* a seen item tied with the next item stays a target where seen items are excluded. It is out of reach
  for every model alike.
- *N48:* ELSA's scores move by about 1e-6 with the batch's composition.

Both are below any difference the tests can show; changing either would redo every run on ML-20M and
Amazon (N43) or change the production ELSA (N48).

**A retuning spot-check of the ablations runs after the main runs, if time is left.**

**Side effect for runs already made.** The library copy changed (EASE's scoring path in entry 33, the Amazon loader here), so its source hash differs between runs made before and after the update, and the stage-1 report warns that a dataset's results come from two builds. Neither change alters anything those runs computed: EASE's scores are bit-identical (entry 33), and only Amazon reads the loader.

**Still open:** SANSA.

**Where.**
- cr: `datasets/amazon2023.py`, `tests/test_amazon2023_dataset.py`.
- Suite: `vendor/compresso-recsys` (re-vendored); `models.py`; both protocols; `scripts/dgx-run.sh`; README.

**Tests.**
- `test_amazon2023_user_sample_keeps_whole_users_by_hash` (cr): whole users, the hash rule, the same sample
  every time, nested samples, bad values refused. It fails with the filter removed; cr's 436 Amazon tests pass.
- `test_bert4rec_runs_through_every_step`: search, final and latency on synthetic data; adding a model moves
  no other model's runs. It fails with the registration removed.

---

## 37. Diversity: catalogue coverage and intra-list diversity of the finals (2026-10-06)

**Decision (Mark).** The design's diversity diagnostics are added. Two items are similar when the same users
interacted with them: co-occurrence, since not every dataset has item attributes (OTTO has none), and those
that exist differ in kind between datasets.

**Change.**
- `evaluate_phase(..., lists=)` hands back each batch's ranked lists, the very lists the metrics are computed
  from: the same users and the same exclusion of seen items.
- `seqrec-eval diversity` (`diversity.py`) scores each finished, current stage-1 final once more on test, keeps
  its lists, and writes `diversity.json` beside the run:
  - **Coverage@K:** the share of the training catalogue in anyone's top K. Exact.
  - **Intra-list diversity@K:** the mean of 1 − cos(i, j) over the pairs in a list, averaged over users.
    The cosine is taken on the best rank-64 approximation of the binarised training matrix, i.e. the cosine of
    that approximation's columns. The exact co-occurrence cosine does not scale to catalogues of 100,000 items
    and more. An item no user had is dissimilar to every other.
- A final with a record is not measured again (`--force` re-measures). A stale or unfinished final is skipped.
- The stage-1 report gains a "Diversity (diagnostics)" table: coverage and intra-list diversity at the primary
  cutoff, mean ± sd over seeds.
- `scripts/dgx-run.sh` runs `diversity` after the finals, on the first GPU device.

**Effect.** Diagnostics only: nothing is selected on them, and no fingerprint changes. It costs one more test
scoring per final, and one factorisation per dataset.

**Where.** `diversity.py`, `evaluate.py`, `cli.py`, `scripts/dgx-run.sh`, README.

**Tests.** `tests/test_diversity.py`:
- the two measures on vectors with known answers, items outside the catalogue included;
- the factorisation reproducing exact co-occurrence cosines on a matrix of low rank, with uneven counts
  (binarised first);
- end to end:
  - every current final is measured on exactly its scored users;
  - a stale final and an unfinished one are skipped;
  - a rerun measures nothing;
  - the report shows the table;
  - row for row, each recorded top 5 holds a next item exactly when the user is a hit at 5, on a model whose
    users both hit and miss.

10 deliberate breaks, all caught after two tests were strengthened. 256 tests pass on CPU and on the GPU.

---

## 38. The ablations also score new items only; repeat removal on OTTO too (2026-10-07)

**Decision (Mark).** Asked whether the ablations test non-repeat behaviour at all. They did not: removing
repeats keeps each item's first occurrence and leaves the targets as they are, so even at 100% removal a next
item can be one the user had before. And the stage-1 report's new-items diagnostic was only made for the
finals.

**Change.**
- Every ablation run on a dataset with `new_item_diagnostic` (Music4All and OTTO among the ablation datasets;
  Yambda in the serving-time sweep) is also scored on the next items its users never had.
  - Seen items are excluded, on the condition's own test users, and written as `test_new` beside `test`.
  - Seen items are judged on the whole history, so in every sweep that keeps the item space the new targets,
    and the users scored on them, are the same at every level.
- The ablation report adds "New items only (descriptive)": mean ± sd over seeds per model and level, the users
  scored, and each sequential model minus ELSA, as a difference of the seed means.
  - No test is made on these and nothing is selected on them, so the tested family of each sweep is unchanged.
- `repeat_removal` also runs on OTTO (35% of its events are repeats), so both ablation datasets have it.

**Why.** On the repeat-heavy data, replay is the floor, and much of what the sequential models gain may be repeats
and recency. The new-items view, where replay scores 0 by construction, shows at every level what a model adds
beyond what its users already had.

**Effect.**
- One more scoring per ablation run, seconds to minutes each.
- Two more refits per model on OTTO for `repeat_removal`, about 2 GPU-hours with BERT4Rec.
- No fingerprint changes. No ablation had run yet, so nothing is redone.

**Where.** `runner.py` (`_execute`); `ablation_report.py` (`_new_items_section`, `dataset_ablation`); both
protocols.

**Tests.**
- `test_every_ablation_run_is_also_scored_on_new_items`: retrained, removal and serving-time sweeps; the same
  users at every level; the report section.
- `test_an_ablation_runs_new_items_are_scored_on_its_own_users`: a condition restricted to half its users is
  scored on those only.

5 deliberate breaks, all caught after the second test was added. 258 tests on CPU and on the GPU.

## 39. Models from other packages, and an extra protocol file for them (2026-10-07)

**Decision (Mark).** A model whose code may not be published, from a private repository, is evaluated like
every other, and neither its code nor its settings enter this repository.

**Change.**
- **Plugins.** `models.py` imports every module that an installed package declares in the entry-point group
  `seqrec_eval.models`. The module registers its models with `register`, as the suite's own models are
  registered.
  - A plugin that fails to import is reported on stderr and listed by `plan`. The suite's own models still run,
    and a run of the plugin's model fails with the reason (`model_spec`).
- **Extra protocol files.** `--protocol-extra FILE` adds the `[models.*]` sections of FILE to the protocol. It is
  repeatable, and defaults to the files in `$SEQREC_EVAL_PROTOCOL_EXTRA`, `:`-separated.
  - Models only: any other section is refused, and so is a model that the protocol or another extra file
    already defines. Everything that decides how the models are evaluated stays in the protocol.
  - A model's fingerprint is its own section, so adding one moves no other model's runs. A sweep that names no
    models runs the added model too.
  - `plan` lists the extra files, and the stage-1 report names each one with its sha256.
  - `scripts/dgx-run.sh` passes the variable on to every step and names it in its first line.
  - The tests unset the variable (`tests/conftest.py`), so they run only the protocols they write.
- **Provenance.** Beside the library and the suite, a plugin model's runs record its top-level package, and
  each package its class lists in `provenance_packages`: the installed version, a source hash and the path
  (`code.plugins`). The stage-1 report lists plugin builds on their own lines, so a plugin's runs do not
  count as another build of the suite.
- **Model diagnostics.** If a model has a `run_diagnostics()` method, its result goes into each run's
  `done.json` (`model_diagnostics`): what the model counts about its own scoring.

**Why.**
- A git-ignored folder inside this repository is one `git add -f` away from the public history.
- It would also leave the model's settings in the protocol file, which is public.
- An installed package plus a separate protocol file keep both outside. The runs still record exactly which
  code ran.

**Effect.**
- No fingerprint changes and nothing is redone.
- Runs of the suite's own models record what they did before.
- `uv sync` without `--inexact` removes an installed plugin, since the lock does not list it.

**Where.**
- `models.py`: `PLUGIN_GROUP`, `_load_plugins`, `PLUGIN_ERRORS`, `model_provenance`.
- `protocol.py`: `load_protocol(extra=)`, `_add_extra_models`, `Protocol.extras`.
- `cli.py`: `--protocol-extra`, `EXTRA_ENV`, `plan`.
- `runner.py`: the run record.
- `report.py`: `plugin_builds`, `_builds_note`, the header.
- `scripts/dgx-run.sh`, `tests/conftest.py`.

**Tests.** `tests/test_plugins.py`, with a fake plugin installed with real `*.dist-info` metadata. It checks
that:
- the plugin registers through its entry point, in this process and in a new one;
- a failing plugin is named, and the suite still runs;
- extra files add models without moving any other fingerprint, may add only models, and may not define a model
  twice;
- extra files come from the option, or else from the variable;
- end to end, the plugin's model runs through search, final, diversity and the report, with its code and
  diagnostics recorded, and the report names the extra file and the plugin build on its own line.

In `test_dgx_run.py`, a test checks that the variable reaches every step and is named.

15 deliberate breaks, all caught; the `dgx-run.sh` one only after its test was added. One more break changed
nothing (`if given:` for `if given is not None:`; the option never gives an empty list). 268 tests pass.

## 40. All-cold histories are counted (2026-10-07, entry 3)

**Change.** `analyse` counts, per dataset, the users whose history holds no item of the training catalogue
(`all_cold`), and among them those with no history at all (`empty`).
- **Which users:** exactly those every model's metrics are computed over.
  - On validation, as the search trials score them: the search split, and its sample of rows.
  - On test, as the finals score them: the refitted split.
  - The row selection is shared with `evaluate_phase` (`_select_rows`), so the two cannot differ.
- **Where it is written:** `analysis/<dataset>/<dfp>/cold-histories-<evaluation key>.json`, once per dataset.
- **Where it is shown:** a line in the stage-1 report and in the analysis report. Where it is missing, both
  reports say "not counted yet".

**Why.** Entry 3 left this open. A matrix model reads its input projected onto its training catalogue, so it
reads such a history as empty and ranks the user by tie order, while a sequence model reads the items as unknown.
This slightly favours the sequential side. The research design (§6.4) says the share is counted on the stage-1
data before the results are read.

**Effect.**
- No fingerprint changes and nothing is redone.
- An existing work dir gets the count from `analyse --sweep none`, which computes only what is missing. The next
  `report` shows it.
- Scoring is unchanged: `evaluate_phase` only moved its row selection into a function.

**Where.** `evaluate.py` (`_select_rows`, `scored_users`, `cold_histories`); `analysis.py`
(`count_cold_histories`, `cold_note`, `analyse_full`); `report.py`; `analysis_report.py`.

**Tests.**
- `test_all_cold_histories_are_counted_over_the_users_scored`, on validation rows rewritten to a cold history and
  an empty one. It checks that:
  - the counted users are those `evaluate_phase` scores, with and without excluding seen items;
  - a history counts as cold exactly where the matrix model's input is empty;
  - test is counted on the refitted split.
- `test_the_reports_give_the_cold_history_count`: both reports, the "not counted yet" note, and `analyse`
  filling in only the count.

11 deliberate breaks, all caught. 270 tests pass.

## 41. A slow model runs beside the main run, in processes of its own (2026-10-08)

**What happened.** In run4, SANSA trained on the CPU process beside EASE and popularity. Each step of
`scripts/dgx-run.sh` waits for all its processes before the next starts, and SANSA's search was far slower than the
GPU models':
- one trial (the densest configuration, with the most inverse epochs) ran for more than 11 hours on Amazon, on
  one core;
- the same configurations come back on every dataset, since a trial's configuration does not depend on the
  dataset.

The GPU finished the search at 05:00 on 2026-10-08 and then stood idle, while the finals waited for SANSA.

**Decision (Mark).** SANSA runs separately, in parallel processes, with its search space unchanged.

**Change.** `MODELS_ONLY=1` runs only the steps that train and score the models named, then their status:
- search, final, their diversity, and each sweep's `ablate`;
- no prepare, analysis, latency or report, which the main run does.

With it:
- **The processes:** `GPUS` names one torch device per process, `cpu` included, so `GPUS="cpu cpu cpu cpu"` gives 4
  CPU processes. The run locks keep them apart, here and from the main run.
- **Logs:** they default to `$WORK/logs-models`, since the log names would otherwise collide with the main run's.
- **The pid:** the first line of every run now gives the script's pid, so one of two runs can be stopped on its own.
  In a container it is the container's pid: `docker exec <container> kill <pid>`.

**Why.** A model's finals depend on its own search only. Waiting for every process at each step was a convenience
of the script, not part of the method.

**Effect.**
- No result changes.
- **The main run:** it leaves SANSA out (no extra protocol), so it moves straight on to the finals.
- **SANSA's run:** it searches, finalises and ablates SANSA in parallel, as long as SANSA needs; its slowest
  trial still bounds how long that takes.
- **Latency:** the main run's latency step will run while SANSA's processes load the CPU. Latency is measured again
  on a quiet machine once every model is in.
- **Reports:** they include SANSA only when generated with the extra protocol.

**Where.** `scripts/dgx-run.sh`; README (*The whole suite on the DGX*).

**Tests.**
- `test_models_only_runs_the_named_models_steps_and_nothing_else`: the steps, their order, the processes, the
  model selection and the separate logs.
- The header test checks the pid line.

7 deliberate breaks, all caught.

## 42. Epoch extensions for selections still improving (2026-10-12)

**What the first finals showed.** 13 selected configurations were still improving at their last epoch (the §34
flag):
- ELSA on ML-20M; GRU, SASRec and BERT4Rec on Amazon; SASRec and BERT4Rec on Music4All;
- ELSA, GRU, SASRec and BERT4Rec on Yambda; GRU and BERT4Rec on OTTO.

Most fell 1–3% over their last epoch. SASRec on Yambda fell 18% over its last 5 of 50 epochs, and GRU 5% over
its last. An epoch is one pass over the training users, and Yambda has only 4,056 of them, so 50 epochs is about
1,600 optimizer steps.

**Decision (Mark).** Every flagged selection gets more epochs: its own configuration retrained at 2× and 4× its
epochs, the best of 1×, 2× and 4× on validation kept. The other option, rerunning each flagged search with a longer
epoch list, cost several times as much.

**Change.**
- **The setting:** `[protocol] epoch_extension = [2, 4]`.
- **When an extension runs:** once a model's search is complete, with nothing failed and open, and its selected
  trial's training loss was still falling (§34's rule). `search` then trains that configuration at each multiple of
  its epochs, as trials: `trial-007-x2`, `trial-007-x4`, on the training window, validation only.
- **Selection:** the best of the selected trial and its extensions on validation. A tie keeps the shorter run.
  `final` waits for the extensions, as it waits for the search.
- **Which models:** only those with an `epochs` parameter, so EASE, popularity and SANSA are never extended.
- **Finals already made:** a final whose configuration differs from the selection only in its epochs is moved
  aside (`final-seed0.superseded-<time>`) and redone. Its records stay; its model file is deleted to save disk. A
  final made with another trial is still refused, as before (A3).
- **Records:**
  - every run records `epochs_factor`;
  - the report names the selection `#7 ×2`, with a note giving each extension's validation score;
  - `status` counts the extensions among the trials;
  - stale finals are judged on the whole configuration, epochs included, not only the trial.
- **The flag rule** (`still_improving`) moved from `report.py` to `runner.py`, unchanged; `report.py` still
  exports it.

**Why.**
- The flag showed under-training, so it is better acted on than only reported.
- Validation decides, not the training loss, because a falling training loss can also mean overfitting.
- The rule is fixed in advance and the same for every model. A flagged model gets up to two more validation trials,
  a slightly larger selection than the others' 10; it is kept this small for that reason.

**Effect.**
- The setting is in no fingerprint: every fingerprint was checked identical before and after. The 10 trials of
  every search stay, and only the extensions and the finals of a changed selection run.
- **When:** before the sweeps, which reuse the selected configuration. The GPU models' sweeps had not started yet,
  so nothing there is redone. SANSA has no `epochs`, so its finished sweeps are untouched.
- **Cost:** about 40–80 GPU-hours with the redone finals; most pairs are cheap, BERT4Rec on Music4All and Yambda
  is the expensive part. If a longer configuration wins on Music4All or OTTO, its ablation refits cost
  proportionally more.

**Where.**
- `runner.py`: `plan_extensions`, `plan_search`, `_summarize`, `differs_in_epochs_only`, `_supersede`,
  `RunSpec.epochs_factor`, `RunSpec.selection`, `stale_final_seeds`.
- `protocol.py`; `cli.py` (`search`); `report.py` (report, status).
- Both protocols; README (the guarantees).

**Tests.** `tests/test_extension.py`:
- the setting is checked and moves no fingerprint;
- extensions wait for a complete search with nothing open, and need a selection still improving and an `epochs`
  parameter;
- a tie keeps the shorter run, and the finals wait for the extensions;
- a failed extension blocks selection until accepted, beside a failed trial;
- end to end: the search trains the extensions on the training window only, finals made at the old epochs are
  moved aside without their model and redone, another model's finals are untouched, and the report names the
  selection;
- a final made with another trial is still refused, even one that differs from the selection only in epochs.

16 deliberate breaks. Two survived at first, and both tests were strengthened; all are caught now. 280 tests pass.

## 43. Only clearly unconverged selections are extended; the suite moves to an RTX 2060 (2026-10-12)

**What happened.** The DGX is no longer available. The rest of the suite runs on Mark's own PC: one RTX 2060
(6 GB, roughly 0.4 of a V100 for this work, an estimate) and 64 GB of RAM. What was left came to 260–410
V100-hours; a week on the 2060 is about 65.

**Decision (Mark).** The suite is cut to fit one week (the plan of 2026-10-12):
- **Stage 1 is finished:** diversity, the history-length inference sweep, the remaining finals, latency on the
  PC's CPU.
- **Epoch extensions:** only for a selection whose training loss still fell more than 5% over its last tenth of
  epochs.
- **Retraining ablations:** on Music4All and OTTO, three sweeps (`history_length`, `shuffle`, `repeat_removal`),
  for ELSA, GRU and SASRec, and for BERT4Rec on OTTO only. Popularity and EASE run them as before.
- **SASRec v2:** a full stage 1 on Music4All, Yambda and OTTO.
- **LRURec:** latency only.

**Change (this entry).** `[protocol] epoch_extension_min_fall = 0.05`: the training-loss fall over the last tenth
of epochs above which a selection is extended. Without it, the report's 1% flag decides, as in §42.

**Why 5%.** The 13 selections the report flags fall into two groups:
- **Clearly unconverged:** SASRec on Yambda fell 17.6% over its last tenth of epochs, GRU on Yambda 5.8%.
- **Mild:** the other 11 fell 1–3% over their last epoch, mostly in runs whose few epochs validation preferred.

At 5% exactly the first group is extended: checked on the DGX records. The threshold was chosen after seeing the
flags, for the budget; it is recorded as that, not as a rule fixed in advance.

**Effect.**
- 4 extension trials (2× and 4×) instead of 26, about 3 V100-hours.
- No fingerprint changes.
- The other 11 flags stay in the report as they are.

**Where.** `protocol.py`; `runner.py` (`still_improving` takes a threshold; `plan_extensions`); both protocols.

**Tests.** `test_only_a_selection_falling_more_than_the_threshold_is_extended`: a 3% fall is extended at the
default and not at 5%, a 50% fall at both; no fingerprint moves; bad values are refused. 4 deliberate breaks, all
caught. 281 tests pass.

**Limitations to record:**
- **Cut entirely:** the GPU models' `density` and `catalogue_top` sweeps (SANSA's exist, and are reported as
  SANSA's only), and BERT4Rec's Music4All ablations.
- **Kept smaller:** SASRec v2 on three datasets and without ablations; LRURec untrained beyond latency.
- **Mixed hardware:** runs on two GPUs (V100 and RTX 2060), which every run's host records; latency is measured
  on the PC's CPU.

## 44. SASRec v2 joins stage 1, vendored from another branch of the library (2026-10-13)

**Decision (Mark).** §43's SASRec v2 is added: a full stage 1 on Music4All, Yambda and OTTO, no ablations. Its
search space is SASRec's plus its own settings, as below (D1), with `allow_history_negatives` fixed by a planted
check (§46). It is vendored without refreshing the rest of the copy, by a script.

**The model.** compresso-recsys's `SASRecCE`, on branch `32-bert4rec-model-implementation` (uncommitted):
- SASRec's network, copied so that SASRec itself stays as published, trained with cross entropy over the whole
  catalogue in place of SASRec's sampled binary objective;
- `allow_history_negatives`: whether the user's own history is among the softmax's negatives;
- input masking (`mask_probability`, with a `[mask]` token);
- co-occurrence augmentation: an input replaced by an item that co-occurs with it (the target stays the true next
  item), or one inserted after it; partners are drawn in proportion to count × popularity^exponent;
- a sampled softmax (`n_negatives`) exists in the library; the suite does not use it.

**Changes in the library** (the checkout's working tree, `sasrec_ce/` and its tests only):
- **`loss_chunk_elements`** (default 2²⁶): the loss over the labelled positions in chunks, their gradients summed
  before the encoder's one backward pass. This is GRU's and SASRec's pattern (§19). BERT4Rec has no such setting:
  it bounds its memory through `max_predictions`.
- **`prediction_chunk_elements`** (default 2²⁶): prediction in parts (§47).
- **Tests:**
  - a chunked step gives the same loss and every gradient to 1e-5 / 1e-7, for the full and the sampled softmax,
    with the history in and out, dropout and masking on, down to one position per chunk;
  - no chunk holds more than the bound;
  - the contract: `candidate_ids`, items after the fitted catalogue read as `unk` and never recommended, and
    `recommend` with item IDs across a save and load.
- 94 tests in `test_sasrec_ce.py`. 8 deliberate breaks, all caught; one was caught only by the whole file, not by
  the tests selected for it.

**Vendoring:** `scripts/vendor-models.sh`.
- **What it copies:** the two packages and their tests (11 files) from the checkout, each Python file with the
  modified-file notice.
- **The check, before anything is written:** every library module they import, directly or through another, must
  be byte-identical in the checkout and the copy. 7 were checked; only `models/base.py` differs (the checkout adds
  the `CandidateScorer` protocol, which the overlay's code does not import).
- **What else it writes:** the exports in `models/__init__.py` and `tests/test_public_api.py`, and a section in
  `CHANGES.patch` and in `VENDORED.md`, each between markers and replaced on every run. A second run changes no byte.
- **Refreshes:** `vendor-cr.sh` now warns, before touching anything, while VENDORED.md records an overlay. That
  section says a full refresh must be followed by `vendor-models.sh`.
- **Two of the overlay's tests are deselected in the copy, both recorded in VENDORED.md:**
  - LRURec's `test_the_trainer_is_a_candidate_scorer` (D4);
  - LRURec's `test_it_evaluates_against_sampled_negatives`, found when the tests first ran against the copy.

  Both need branch 32's sampled evaluation, which the suite does not use.
- **Run against the copy in the suite's venv:**
  - the overlay's tests: 242 passed, 2 deselected;
  - the copy's 38 model, evaluation and API test files: 1,681 passed, 3 skipped. Its dataset and download tests
    were not run (no real data on the laptop).

**The suite.**
- **Registered** in `models.py`. As for BERT4Rec, the trainer builds its own batcher, whose vocabulary has the
  `[mask]` token (H26).
- **`[models.sasrec_ce]` in both protocols:**
  - `datasets = ["music4all", "yambda", "otto"]` and `in_ablations = false` (§45);
  - fixed: `allow_history_negatives = true` (§46), and `cooccurrence_popularity_exponent = -0.5` (partners by
    count / √popularity, cosine's weighting);
  - space: SASRec's ranges without `n_negatives`; `mask_probability` {0, 0.1, 0.2, 0.3};
    `cooccurrence_replace_probability` and `cooccurrence_insert_probability` {0, 0.1} each.
- **The 10 trials as drawn:** insertion is on in 8 of them, replacement in 3, masking in 7. Only trial 9 has no
  co-occurrence augmentation at all.

**Why.**
- Branch 32 lacks what the suite needs (`temporal_train_users`, the Music4All/Yambda/OTTO loaders, loss
  chunking), so `vendor-cr.sh` could not take the models from it.
- The new models' dependencies are the same files in both trees, so an overlay checked file by file is exact.
- The full softmax is what distinguishes v2 from SASRec, which already covers 1–256 sampled negatives.
- The epoch grid stays SASRec's. On Yambda, the extension rule (§42–43) is what took SASRec past 100 epochs, and it
  applies to this model alike. A larger grid would cost mostly on Music4All and OTTO, which have 12 and 18 times
  Yambda's users per epoch.

**Effect.**
- **Fingerprints:** every fingerprint of both protocols was computed before and after (runs, sweep conditions,
  ablation runs, baselines, datasets). None changed or disappeared; 5 were added, SASRec v2's runs.
- **Cost, projected:**
  - **Method:** on the laptop's RTX 4050, with synthetic data shaped like each dataset (catalogue, users, median
    history), and the 10 configurations as drawn (`review-plan/findings/S1-rest/sasrec_ce_cost.py`).
  - **Search:** 4.2 h on Music4All, 2.0 h on Yambda, 3.0 h on OTTO.
  - **Finals:** 2.8 h if the selection costs the mean, 8.7 h if it is the most expensive.
  - **In all:** 12–18 h, before a Yambda epoch extension, which could add up to about 14 h in the worst case.
  - **On the RTX 2060:** a factor of this that has not been measured. After the first trials,
    `scripts/project-time.py` reprojects from the PC's own records.
- **Two code builds in the report.** Every run records a hash of the whole library source and of the whole suite
  source (§17). Runs made after this change record new ones, so the reports will list two builds per dataset.
  - The difference cannot touch an earlier run: the library gained two packages and their exports, and nothing else
    in it changed.
  - The suite's changes place models (§45) and do not change what a run of an existing model computes. Their run
    fingerprints are unchanged, and the suite's tests pass as before.
  - This entry is the check that §17's warning asks for.

**Side effects.**
- **The laptop's `.venv`.** `uv sync` would have replaced the whole environment, because its Python changed from
  3.14.7 to 3.14.8. That would have removed the SANSA plugin, even with `--inexact`. The library alone was
  reinstalled with `uv pip install --no-deps --reinstall ./vendor/compresso-recsys`. The README says when to use
  which.
- `tests/test_local_run.py` lists the models with an `epochs` search; SASRec v2 joins the list.

**Where.**
- Library: `models/sasrec_ce/` (`config.py`, `trainer.py`), `tests/test_sasrec_ce.py`.
- Suite: `scripts/vendor-models.sh` (new), `scripts/vendor-cr.sh`, `vendor/compresso-recsys/`, `models.py`, both
  protocols, the README.
- Measurements: `review-plan/findings/S1-rest/sasrec_ce_memory.py`, `sasrec_ce_cost.py` (with its `.out`).

**Tests.**
- **`tests/test_vendor_models.py`** (fake checkout and copy):
  - the overlay, exports and records, and a second run that changes no byte;
  - a dependency that differs only through another module's import is refused before anything is written;
  - `vendor-cr.sh` warns once an overlay is recorded.
- **`tests/test_smoke.py::test_sasrec_ce_runs_through_every_step`:** search, final, diversity, latency and the
  report, with every augmentation on. No other model's fingerprint moves, and SASRec v2 is compared with ELSA like
  every model.
- **Deliberate breaks:** 5 in the scripts. One survived at first, because it was redundant code rather than a test
  gap; the code was simplified and the real mechanism broken instead. All are caught.

## 45. Where a model runs: `datasets` and `in_ablations` (2026-10-13)

**Problem.**
- A sweep that names no models runs every model of the protocol. SASRec v2 would therefore have joined the
  Music4All and OTTO sweeps, and the inference sweep on every dataset.
- Listing the models in every sweep instead would silently drop the SANSA plugin from them: it comes from an extra
  file, and the public protocol cannot name it.
- The report had no "not on this dataset": SASRec v2 would have shown as an unfinished `0/10` on ML-20M and Amazon,
  and a full `final` would exit 3 for it.

**Change.** Two optional keys in a `[models.*]` section:
- **`datasets = [...]`:** the datasets the model runs on (default: all). Elsewhere:
  - no trial is planned;
  - its summary counts every trial as skipped, without reading or writing anything;
  - so the report notes it as skipped, as for EASE's `max_items`, and `status` shows "skipped";
  - `final`, `diversity` and `latency` pass it by;
  - the latency, diversity and repeat-strata tables, the window diagnostic and the code-builds note leave out any
    runs of it that exist there;
  - `plan` shows "not run here";
  - a run of it is refused before anything is written, so no record can outlive the scope.
- **`in_ablations = false`:** the model is left out of every sweep that names no models, and a sweep that names it
  is refused.
- **(Mark) Neither enters a fingerprint,** like `eval_batch_size`. Widening `datasets` later starts the new
  datasets' runs and leaves the finished ones where they are. Narrowing hides the old runs everywhere, and widening
  again finds them.

**Effect.**
- No fingerprint moves: SASRec v2 is the only model with either key.
- A later `datasets` change on any model moves nothing either; this is tested on the real protocol.
- `plan` lists SASRec v2 in no sweep, and as "not run here" on ML-20M and Amazon.

**Where.** `protocol.py` (`_PLACEMENT_KEYS`, `ModelProtocol.runs_on`, the sweeps' default), `runner.py`
(`plan_trials`, `_base_summary`, `execute`), `cli.py`, `report.py`, `latency.py`, `diversity.py`, `strata.py`;
the README.

**Tests.**
- **`tests/test_placement.py`:**
  - every model of the real `protocol.toml`, given each of five placements, moves no fingerprint at all (runs,
    conditions, ablation runs, baselines, datasets); as a control, another key still moves a run fingerprint;
  - bad values are refused;
  - a sweep's default leaves an `in_ablations = false` model out, and naming one in a sweep is refused;
  - end to end on the synthetic workspace, a model with runs on a dataset is then kept off it:
    - it is planned nowhere and reported as skipped;
    - it is gone from status, report, latency, diversity, strata and the code-builds note;
    - search, final, diversity (forced) and latency write and rewrite nothing;
    - a stray run is refused;
    - widened again, its old runs are found.
- **`tests/test_ablations.py::test_a_model_kept_out_of_the_sweeps_runs_in_none_and_is_reported_in_none`:**
  SASRec v2 with stage-1 runs and no ablation runs. Every sweep's models, conditions and ablation fingerprints are
  unchanged; `ablate` over every model writes nothing; no ablation report mentions it.
- **Deliberate breaks:** 14, each caught. The first run of these tests also found a real gap: the window
  diagnostic and the code-builds note read finals directly. Both now follow the scope.

## 46. SASRec v2 keeps the user's history among its negatives: the planted chain (2026-10-13)

**The check (D1, Mark's rule).** Set `allow_history_negatives` from H53's planted chain: true if it confirms that
keeping the history out of the softmax holds back the ranking where seen items are ranked.
- **Data:** next = σ(last), 1,500 test users, 300 items, through the real temporal split and the suite's evaluator.
- **SASRec v2's settings:** H53's settings for SASRec (window 50, d 64, 2 blocks, 1 head, dropout 0.2, lr 1e-3,
  batch 128, 50 epochs), with the full softmax in place of 16 negatives.
- **Control:** SASRec, run again beside it.

| model | its history among the negatives | NDCG@10 / oracle, seen items ranked | the same, seen items masked |
|---|---|---|---|
| SASRec (control; its sampler keeps the history out) | no | 0.640 | 1.000 |
| SASRec v2, masking 0.2 | no | 0.613 | 1.000 |
| SASRec v2, no masking | no | 0.696 | 1.000 |
| SASRec v2, masking 0.2 | **yes** | **1.000** | 1.000 |
| SASRec v2, no masking | **yes** | **1.000** | 1.000 |

Hit@10 is 1.0 everywhere: the planted item is always in the top 10. Kept out of the softmax, history items are
never pushed down and outrank it whenever seen items are ranked, as on all three of SASRec v2's datasets. The
control reproduces H53's 0.64.

**Decision.** `allow_history_negatives = true`, fixed in both protocols.

**Where.** `review-plan/findings/H53/h53_sasrec_ce.py` and `chain_sasrec_ce.out`. The H53 harness predates two
changes, and the script adapts it without editing the original:
- its protocol must now state `refit` (H11), so it states `refit = false`, which is what the harness did (fit on
  the training window, score test);
- the baselines' full-catalogue ranking has become a sparse path its oracle does not implement, so the oracle is
  ranked by the dense path it was written for.

## 47. Prediction at window 400 on a 6 GB card: SASRec v2 scores in parts; SASRec's own risk (2026-10-13)

**What the memory probe found.** On the laptop's RTX 4050 (6 GB, like the PC's RTX 2060):
- **Training:** SASRec v2's largest setting fits. At 150,000 items, window 400, batch 256, d 128, 3 blocks and
  2 heads, with the history excluded and every augmentation on, a step peaks at 3.8 GB allocated (4.1 reserved) with
  chunks of 2²⁶, against 4.6 GB at 2²⁷. Hence the default.
- **Prediction:** scoring 1,024 histories at once (the main protocol's `eval_batch_size`) at window 400 with
  2 heads ran out of memory at every catalogue tried, from Music4All's 36,797 items up. Each block builds a
  1,024 × 2 × 400 × 400 attention mask: 1.2 GiB.
- **SASRec itself:** its prediction is the same code, so the vendored SASRec was measured too, at Music4All's
  catalogue, d 128, 3 blocks:

| window | 1 head | 2 heads |
|---|---|---|
| 100 | 0.6 GB | 0.7 GB |
| 200 | 1.2 GB | 1.8 GB |
| 400 | 2.8 GB | **out of memory** |

**Change (SASRec v2 only).** `predict_on_batch` scores a caller's batch in parts, so that each part holds at most
`prediction_chunk_elements` elements of the larger of the attention map (heads × window², per history) and the
catalogue scores. The rows are independent and padded to the full window, so the parts rank exactly as one batch
would. A batch of 1,024 at the corner then peaks at 1.6 GB.

**Not changed:** SASRec, whose code the week-long run uses. Any SASRec run on the RTX 2060 whose configuration has
window 400 and 2 heads would fail where it scores 1,024 users at once. That covers validation of the Yambda epoch
extensions, finals, diversity, the inference sweep's rescoring, and the ablation refits. BERT4Rec, with window up
to 400 and up to 4 heads, is likely exposed too; it was not measured. Lowering `eval_batch_size` in
`protocol.toml` (no fingerprint) avoids it for every model. An open decision, below.

**Where.** Library: `sasrec_ce/trainer.py` (`_prediction_rows`), `config.py`. Probes:
`review-plan/findings/S1-rest/sasrec_ce_memory.py`, `sasrec_predict_memory.py`.

**Tests.** `tests/test_sasrec_ce.py`:
- in parts, the rankings and scores equal one batch's, with seen items excluded and kept and with candidates;
- no part holds more histories than the bound.

3 deliberate breaks, all caught (§44).

## 48. LRURec is timed for latency only: `latency_only` (2026-10-14)

**Decision (Mark, §43; settled with the SASRec v2 session's plan, D2).** LRURec enters stage 1 for its inference
latency only, not trained for accuracy:
- the paper's architecture: d 64, 2 blocks, dropouts 0.2;
- a window of 400, SASRec's longest;
- on all five datasets;
- nothing of it on the GPU.

**Change.** A model key `latency_only = true`:
- **Planning:** no search. One final at the first seed, of the model's fixed configuration. A latency-only model
  takes no `space` or `trials`, and is in no sweep (`in_ablations` false; naming it in a sweep is refused).
- **The fit:** that final is fitted on the first 256 training users (`LATENCY_ONLY_ROWS`), with the whole catalogue.
  For LRURec, 1 epoch with 100 sampled negatives, the paper's large-catalogue setting; prediction ranks every item.
  It is saved and never scored: no validation, no test, no diagnostics. Its record says
  `"latency_only": {"fit_rows": …}`.
- **Where it appears:** `latency` times it like any final. The stage-1 report leaves it out of every table,
  comparison and CSV, with one note per dataset. Diversity and the repeat strata skip it. `status` shows
  "latency only", and `plan` shows "1 unscored final".
- **LRURec** is registered in `models.py`, with its section in both protocols.

**Why.**
- A request's time depends on the architecture, the window and the catalogue size, not on what the weights hold.
  So a model fitted on a few users times exactly as a trained one would, at no GPU cost.
- LRURrefuses `epochs = 0`, and fitting it on every user with a full softmax does not fit 6 GB: Yambda's
  logits alone are about 13.7 GB.

**In the fingerprint, unlike `datasets` and `in_ablations` (§45).** A latency-only final is not a model whose
accuracy means anything. If the key were left out of the fingerprint, switching it off later would reuse that final
as a real one. So it is part of the model's section and its fingerprint (tested). (The plan sent to the SASRec v2
session listed it among the fingerprint-neutral keys; that was wrong for this key.)

**Effect.**
- No fingerprint of a finished run moves: every run, baseline and sweep fingerprint was compared against the
  committed protocol.
- LRURec costs one short CPU fit per dataset, plus its latency timing.
- `scripts/project-time.py` skips models without trials. `scripts/local-run.sh` shortens only searched epochs;
  a fixed count, as LRURec's, no longer stops it (its check counted every `epochs` line).

**Where.**
- `protocol.py` (`ModelProtocol.latency_only`, `scored_on`); `runner.py` (`plan_trials`, `plan_finals`,
  `_execute`, `LATENCY_ONLY_ROWS`, `_save_model`).
- `report.py` (stage-1 report, status); `diversity.py`; `strata.py`; `cli.py` (`plan`); `models.py`.
- Both protocols; `scripts/project-time.py`; `scripts/local-run.sh`; README; OVERVIEW.

**Tests.** `tests/test_latency_only.py`:
- the key takes one fixed configuration (no space, trials or `in_ablations = true`) and must be a boolean;
- it is part of the fingerprint, nothing is searched, and the one final is at the first seed;
- end to end on the synthetic workspace: no trials; one final fitted on the capped number of users, saved, never
  scored; no diversity; latency measured; `status` and `plan` show it; in the report it appears only in the latency
  table, and not in the CSV.

`tests/test_placement.py` (§45) no longer tries `in_ablations = true` on a latency-only model, which refuses
it. 13 deliberate breaks, all caught (one in `local-run.sh`). 305 tests pass on CPU and on the GPU.

## 49. SASRec v2 on ML-20M and Amazon too (2026-10-19)

**Decision (Mark).** The week's run finished on 2026-10-16 and the client's data has not arrived, so the PC has
time. SASRec v2 now runs a full stage 1 on all five datasets: `datasets` in `[models.sasrec_ce]` adds `ml20m` and
`amazon` to Music4All, Yambda and OTTO. Still no ablations.

**Why.** §43 kept SASRec v2 to three datasets for the week's budget only. Without the other two:
- there is no SASRec v2 against SASRec on ML-20M and Amazon;
- the gap is in the two datasets whose main test already excludes seen items, which is what a client's
  "recommend only new items" feed looks like.

**Settings unchanged.** `allow_history_negatives = true` (§46) was chosen for datasets where seen items are ranked.
On ML-20M and Amazon they are excluded (`exclude_seen = true`), and there the planted chain scored the same either way
(1.000 with seen items masked). The search space and its 10 configurations are the same (trial *i* draws the same
configuration on every dataset).

**Cost.** Measured with the probe that priced the first three datasets, on synthetic data shaped like each dataset
(training users, the catalogue the finals fit, median history; no real data read), on the laptop's RTX 4050:

| dataset | search | finals (3 seeds, mean / most expensive selection) |
|---|---|---|
| ML-20M (21,557 items) | 4.1 h | 1.2 / 3.4 h |
| Amazon (275,514 items) | 7.2 h | 2.2 / 4.2 h |

That is 11 to 19 hours in all. On the first three datasets the probe's search estimate was within 2–25% of what the
RTX 2060 took, always on the high side: Music4All 4.2 h estimated, 3.95 h actual; Yambda 1.97 h, 1.48 h; OTTO 2.99 h,
2.94 h. Every configuration trained and predicted within the 6 GB card at Amazon's catalogue, twice the 150,000 items
§47 measured.

**Effect.**
- No fingerprint moves (§45); the runs on Music4All, Yambda and OTTO keep their fingerprints, checked against the
  copied work folder.
- The stage-1 report, status and plan show SASRec v2 on ML-20M and Amazon as unfinished until their runs exist.
- §43's limitation "SASRec v2 on three datasets" no longer holds once they do.

**Where.** Both protocols; README; OVERVIEW (models table, status). Probe:
`review-plan/findings/S1-rest/sasrec_ce_cost_ml20m_amazon.py` and its `.out`.

**Tests.** No code changed. `tests/test_placement.py` already checks this placement on the real protocol (every
dataset, no fingerprint moves). 305 tests pass.

## 50. New items get a floor and the tests (2026-10-20)

**Problem.** The client's feed (SpicyChat's "For You") never shows an item the user already had, so the comparison
that matches it is on new items. The suite scored only the models there (§38's diagnostic):
- **No floor:** recent popularity beat every model on MovieLens and Amazon, where every next item is new, but it was
  never scored on new items elsewhere;
- **No test:** the new-items table was means only.

**Decision (Mark).** The baselines are scored on new items too, and the report compares every model on them, with
the reference and with that floor, by the main comparison's test.

**Change.**
- **`analyse`:** each baseline, at the setting selected on all next items, is also scored on the test users' next
  items they never had, seen items excluded (`test_new` beside its `test`), on every dataset with
  `new_item_diagnostic`. The baselines keep their selection, as the models keep theirs: neither is tuned for new items.
- **The stage-1 report**, per dataset with the diagnostic, below the new-items table:
  - the new-items floor: the strongest baseline scored that way, waiting for every baseline as the main floor does;
  - every model against the reference, and every model against the floor: the seed-aware t-test, each a Holm family;
  - and across datasets, a table like the main one, on new items.
- **By construction:** replay under exclusion has nothing left to replay, so it ranks by training popularity.

**Effect.**
- No fingerprint moves; `analyse` adds the new scores beside the old.
- **The order of reading changes, not the protocol:** the main comparison fixed in advance stays all next items.
  The new-items comparison is added after the preliminary results, because the client's feed excludes history, and
  is reported as that.

**Where.** `analysis.py` (`discovery_views`, `score_view`, `analyse_full`, `full_results`), `report.py`
(`_view_comparisons`, `dataset_report`, `_across_datasets`).

**Tests.** `tests/test_discovery.py`:
- every baseline is scored on both views of new items, with exclusion;
- replay equals popularity by events there;
- the report compares on new items, against the reference and against the floor, and the floor waits for every
  baseline.

## 51. A model at its serving window (2026-10-20)

**Problem.** In the inference sweep ELSA scored 1.5 to 1.9 times its whole-history score when a request carried
only its last 5 to 20 events (MovieLens-20M, Music4All, Yambda). Production may serve it so. Against that ELSA the best sequential model's lead fell from +21–136% to +11–42%, but the window was picked on
test, so that was an upper bound for ELSA, with no test.

**Decision (Mark).** Choose the window on validation, score it on test, and test it like every other comparison.
Production's own window, once Recombee gives it, can be fixed by listing it alone.

**Change.**
- **`[models.*] serving_windows = [...]`:** the windows a model is also served at; in no fingerprint, since it
  rescores the finals and changes none (`_MEASUREMENT_KEYS`). ELSA gets the inference sweep's levels, 1 to 1,000;
  SANSA the same, in the private protocol (P7 there). A latency-only model refuses it.
- **The stale run** (`stale-seed0`, with §53): the selection fitted on the training window under the first seed,
  exactly as its search trial was, kept, and scored on validation and test.
- **`seqrec-eval window`:**
  1. fits the stale run if it is missing;
  2. scores it on the validation sample with every history cut to its last n events, for each n, and whole;
  3. keeps the best, a tie keeping the longer window and the whole history longest of all;
  4. rescores every current final on test at that window (`test_last<n>`, and `test_new_last<n>` on new items).

  Seen items are still excluded against the whole history, so the users scored are the finals' own, and every
  comparison pairs them.
- **The stage-1 report:**
  - per such model, the validation score at every window, and the search trial's own as a check;
  - the test score at the chosen window;
  - every model, at its whole history, against the windowed one (the model itself included, as "whole history"):
    the seed-aware test, Holm; and the same on new items.

**Effect.**
- No fingerprint moves.
- **Cost:** one extra fit per model and dataset (ELSA on the GPU, SANSA on the CPU), then rescorings.
- **A check comes with it:** the stale run reproduces the selected trial's validation score, since it is the same
  configuration and seed. A difference beyond the GPU's nondeterminism would point to a problem.

**Where.** New `window.py`; `runner.py` (`plan_stale`, the run kind `stale`); `protocol.py`; `cli.py` (`window`);
`report.py` (`_window_section`).

**Tests.** `tests/test_window.py`:
- the key is checked and moves no fingerprint;
- a window cuts the inputs but not what was seen, nor who is scored;
- on synth, the window is chosen on validation, the stale run reproduces the selected trial, the finals are
  rescored on their own users, and the report compares;
- a tie keeps the longer window, and the whole history longest;
- nothing is chosen until every window is scored.

## 52. A settings sweep: SASRec v2's parts (2026-10-20)

**Problem.** SASRec v2's selection, trial #0 on Music4All, Yambda and OTTO, has width 64, 3 blocks, co-occurrence
replacement and insertion, and no masking. The random search cannot say which of these the result rests on: every
setting changes between trials at once.

**Decision (Mark).** Refit the selection with one part changed at a time, and compare each with the stage-1 final.

**Change.**
- **The transform `settings`:** the data as it is. Each level is the selection with the settings in `options[level]`
  put in place of its own (`settings_of`), refitted under the sweep's seeds. It reuses the ablation machinery:
  - the reference reloads the stage-1 final;
  - the fixed test users, the manipulation check (nothing moves), the new-items scores;
  - and "each level against the full data", which is the result.

  A setting the model's section neither fixes nor searches is refused before anything runs.
- **The protocol:**
  - a settings sweep names its models;
  - it may name a model kept out of the data sweeps (`in_ablations = false`, §45), since it is about the model itself.
- **Not done for it:**
  - `analyse` skips it: its baselines are the full data's;
  - the report has no knee and no gap for it.
- **`[ablations.sasrec_ce_parts]`:** SASRec v2 on Music4All, Yambda and OTTO, seed 0, at six levels:
  - `no_augmentation`, `replace_only`, `insert_only`: masking off, the co-occurrence parts one at a time;
  - `masking`: masking 0.2 added;
  - `width_32`: width 32;
  - `one_block`: one block.

**Effect.**
- No other sweep's fingerprint moves, nor its models.
- **Cost:** about 8 to 10 hours on the RTX 2060, from the finals' own times (Music4All 0.8 h, Yambda 0.3 h, OTTO
  0.5 h per run, less for the smaller variants).

**Where.** `ablations.py` (the transform, `settings_of`, `plan_ablation`), `protocol.py` (`SETTINGS_TRANSFORM`,
`_parse_ablation`), `analysis.py`, `ablation_report.py`, both protocols.

**Tests.** `tests/test_settings_sweep.py`:
- the sweep is checked: it names its models, every level has its settings, and `full` stays reserved;
- it may name a model out of the data sweeps, which a data sweep still may not;
- on synth, each level refits the selection with its settings, which differ from the selection's, and is compared
  with the final;
- nothing about the data moves, and nothing is analysed;
- a misspelt setting is refused.

`tests/test_placement.py` now allows a model out of the sweeps to have fingerprints in a settings sweep, and in no
other sweep.

## 53. Freshness: the finals one window older (2026-10-20)

**Problem.** How quickly a model goes stale decides how often it must be retrained. The plan was to split test by
days since training, but the split keeps no time for the test window's events: only `prepare` reads them, from the
raw data, and the PC runs without it.

**Decision (Mark: measure freshness; the form follows from what the split has).** Measure one window of staleness,
two ways:
- **The catalogue** (`analyse`, exact): of the test users with a next item, the share whose next item was first seen
  in the validation window, which a model one window older has never seen, beside the share no model has seen
  (`next-item-age-*.json` in the analysis folder).
- **The model:** the stale run (§51) scored on test, against the first seed's final, on the final's users. A user
  whose next item the older model has never seen counts 0 for it, which is part of the cost. One seed each, so the
  test is the paired t-test over users, Holm across a dataset's models; on next items and on new items.

The window is each dataset's own: 339 days on MovieLens-20M and Amazon, 30 on the music data, two on OTTO.

**Change.**
- **`seqrec-eval stale`:** the stale run of every selection asked for.
- **The run kind `stale`:**
  - it fits the training window, as a trial does, and refuses the refitted split;
  - it keeps its model, and is scored on validation and test (and new items).
- **The stage-1 report:** "Freshness: one window older", after the across-datasets tables.

**Effect.**
- No fingerprint moves.
- **Cost:** one extra fit per model and dataset, as one search trial.
- **Limit:** the measure is one window, not a curve over days.

**Where.** New `freshness.py`; `runner.py`; `analysis.py`; `cli.py` (`stale`); `report.py`.

**Tests.** `tests/test_window.py`:
- the stale run is the selection one window older and refuses the refitted split;
- the comparison pairs the final's users, one seed each;
- a user the older model cannot score counts 0;
- the catalogue's age adds up and is reported.

## 54. Discovery: novelty, the long tail, and later new items (2026-10-20)

**Decision (Mark).** Two measures that bear on the gap between offline and online:

- **What the lists hold** (`seqrec-eval diversity`, beside coverage and intra-list diversity):
  - **novelty@K**, the mean self-information −log2(p) of the recommended items, with p the share of training users
    who had the item (Zhou et al., 2010);
  - **the long-tail share@K**, the share of recommended items outside the head, the most popular items that together
    hold 80% of the training interactions.

  An item no training user had counts as had by one, and as a tail item.
- **Later new items** (targets `window_new`): every new item a user went on to have in the test window, seen items
  excluded. A list that holds several of them counts each, where the next-item score counts only the first: one
  target per user undercounts a list that is good in several places.
  - **Models:** the finals are scored on it in the diversity pass, which reloads every final (`test_window_new`).
  - **Baselines:** scored on it by `analyse`, which gives it a floor.
  - **Report:** it is compared as new items are (§50), on every dataset.

**Effect.**
- No fingerprint moves.
- `DIVERSITY_VERSION` 2: every final is measured again, once, on its next `diversity`.

**Where.** `diversity.py` (`item_users`, `tail_items`, `list_diversity`, `measure_final`, the table), `evaluate.py`
(`window_new`), `analysis.py`, `report.py`.

**Tests.** `tests/test_discovery.py`:
- later new items are exactly the window less the history, and every user with one is scored;
- novelty and the tail share are checked on a hand-made list, the head included;
- the report and the diversity table show them.

## 55. Loose ends: SASRec at 8× on Yambda, and EASE in less memory (2026-10-20)

**SASRec on Yambda (Mark).**
- **The problem:** it was still improving at 4× its epochs: validation 0.051, 0.070, 0.078 at 1×, 2×, 4×, and its
  loss still falling 15%.
- **The change:** `epoch_extension = [2, 4, 8]`, the same rule for every selection still falling more than 5%
  (§43). On today's runs it adds SASRec and GRU on Yambda at 8×: about 0.25 and 0.8 hours, and SASRec's finals
  (about 0.26 hours each at 8×) if 8× wins.
- **Ablation runs follow the finals:**
  - a run built on a final at other epochs than the selection's is moved aside (`*.superseded-<time>`) and redone,
    as the finals are (§42), rather than refused;
  - `differs_in_epochs_only` covers the stale run and every ablation run;
  - `ablate` counts such a run as work left, so a condition whose runs all finished is still built for it.

  So the inference sweep on Yambda redoes those models' rescorings after the new finals.

**EASE (Mark).**
- **The problem:** EASE's Music4All ablation runs were killed by the out-of-memory killer on 2026-10-14. The
  library forms XᵀX as a sparse product, densifies it, and inverts it with `np.linalg.inv`: about nine items × items
  matrices at once, 45 GB on 36,797 items.
- **The change:** `LowMemoryEASE`, in the suite's `models.py`, is now the registered `ease`. It is the library's
  class with its own fit, the same closed form:
  - the Gram matrix is built dense, a block of items at a time;
  - it is inverted in place by Cholesky (`potrf`, `potri`; positive definite for λ > 0);
  - its triangle is mirrored, a block at a time;
  - the peak is about 1.4 such matrices, about 7.6 GB on Music4All.
- **Why not in the library:** the compresso-recsys checkout is on another branch than the vendored copy came from,
  and branches are not switched, so the vendored library is untouched.

**Effect.**
- **Weights:** equal to the library's to rounding (float64 2e-16, float32 1e-7 relative). Cholesky and LU round
  differently, so float32 weights can differ in their last digits from those of EASE's earlier finals.
- **Checkpoints:** saved and read as the library's, both ways.
- **The missing runs** (EASE's Music4All ablations: `history_length`, `shuffle`, `repeat_removal`) can run on the
  PC's CPU. *Superseded by §56: in float32 three of them failed, the fit is now float64, and EASE's ablations are
  dropped.*
- No fingerprint moves.

**Where.** `protocol.toml` and `protocol.local.toml` (`epoch_extension`), `runner.py` (`differs_in_epochs_only`),
`cli.py` (`_ablate`), `models.py` (`LowMemoryEASE`).

**Tests.**
- **`tests/test_ease_memory.py`:**
  - the weights equal the library's in float64 and float32, over uneven blocks;
  - it is the registered EASE, and reads and writes the library's checkpoints;
  - it holds under 1.6 items × items matrices where the library held over 3 (`tracemalloc`);
  - an indefinite matrix is refused.
- **`tests/test_settings_sweep.py`:** on synth, an epoch extension moves the selection; `ablate` then moves the
  inference sweep's runs aside and redoes them at the new epochs.

**Deliberate breaks for §50–§55:** 16, each caught: exclusion, targets, floors, the window's tie and exclusion, the
stale split and model, the zero for unscored users, the settings, the fingerprint, the supersede (two), the EASE
mirror, the head, novelty, the placement rule. The first round missed one, because the test's settings equalled the
synthetic selection; the test now changes them. 332 tests pass on the CPU and on the GPU.

## 56. EASE in float64, and its ablations dropped (2026-10-20)

**What happened.** On the PC, three of EASE's Music4All ablation runs failed in §55's float32 fit: "not positive
definite", at shuffle 10, shuffle all and repeat removal 0.5. A fourth (shuffle 2) was the death count left by the
out-of-memory kills of 2026-10-14. Seven others succeeded.

**Why.** EASE on Music4All was selected at λ ≈ 1.06 (trial #6), on a matrix of play counts.
- **The conditioning:** a track played thousands of times makes XᵀX's largest entries huge, while near-duplicate
  items leave eigenvalues near λ. The condition number then exceeds what float32 can hold (about 10⁷).
- **What each fit does with it:** float32 Cholesky meets a non-positive pivot and stops. The library's float32 LU
  does not stop, but its inverse is wrong.
- **A synthetic matrix of that kind** (counts, 40 near-duplicates, condition 1.8·10¹⁰) shows how wrong. The library's
  weights were off by 2.4 times the largest weight, and its top-10 lists shared 85% of their items with the exact
  solution's. The float64 fit's were off by 3·10⁻⁷ relative, with identical top-10s.

**Change.** `LowMemoryEASE` accumulates and inverts the Gram matrix in float64, whatever the model's dtype, and stores
only the finished weights in that dtype (float32), a block of rows at a time.
- **Peak:** about three float32 items × items matrices, about 16 GB on Music4All (the library's: nine, 45 GB).
- **Where the problem is well conditioned,** the weights equal the library's to 10⁻⁶ relative in float32 and 10⁻¹²
  in float64.

**Decision (Mark).** EASE's ablations are dropped, and its stage-1 results stay as they are.
- **`[models.ease] in_ablations = false`:** EASE is in no sweep. Placement only, so no fingerprint moves, and its
  runs stay on disk.
- **The side effect:** EASE's serving-time history sweep runs (ML-20M, Music4All), which rescored its stage-1
  finals and were sound, also leave the reports.
- EASE is not one of the study's comparators: those are ELSA and SANSA.
- No EASE model is fitted again. The float64 fit stays the registered EASE, for any later fit; it cannot run on the
  PC in the library's form.

**A limitation to record.** EASE's stage-1 finals on Music4All were fitted with the library's float32 inverse, on a
problem too ill-conditioned for it. Their weights are likely inaccurate. Read EASE's Music4All result (test
NDCG@10 0.0810, +28.5% against ELSA) with that in mind; whether an exact fit would score higher or lower is not
known. ML-20M (λ ≈ 999, one rating per film, a well-conditioned problem) is not affected in this way.

**Where.** `models.py` (`LowMemoryEASE`), both protocols, README, OVERVIEW.

**Tests.** `tests/test_ease_memory.py`:
- on a well-conditioned problem the weights equal the library's;
- on an ill-conditioned one, float32 Cholesky fails, the library's float32 weights are far off, and ours equal the
  exact solution, with the same top-10 lists;
- it reads and writes the library's checkpoints;
- it holds under 3.3 float32 items × items matrices where the library held over 6.

3 deliberate breaks, all caught: the mirrored triangle, a float32 inverse, float64 weights kept. 332 tests pass.

---

## Known limitations recorded by the review

- **EASE's stage-1 finals on Music4All** were fitted in float32 on an ill-conditioned problem (λ ≈ 1, play counts),
  so their weights are likely inaccurate (§56).

- **ML-20M includes only users with at least 20 ratings over all time** (GroupLens README; checked: minimum
  20). Users are selected on their future. Together with H48, they are all heavy raters.
- **Amazon keeps only reviews of products in the 2023 metadata snapshot.** The source is the 0-core file, so
  there is no k-core.
- **Source selection not checked:** how the Music4All-Onion, Yambda-50M and OTTO releases were drawn.
- **Yambda keeps the listens Yandex's recommender served** (H14, entry 2): part of its targets and of its
  repeat rate come from the platform's radio.
- **OTTO is loaded as clicks only** (H13, decided 2026-10-05: kept). Carts and orders, about a tenth of the
  events, are left out rather than recorded as the same kind of event as a click. The research design
  (`producten/tussenproducten/research_design.odt`, §5.4) says so since 2026-10-05; it listed three
  interaction types before.
- **Matrix models rank an all-cold history by tie order** (entry 3): the users concerned are counted per
  dataset in both reports (entry 40).
- **Refitted models use settings chosen on less data** (entry 4).
- **BERT4Rec keeps its published position numbering** (entries 4 and 36, N1): a user whose history is exactly the
  longest training sequence meets an untrained mask position, where no user fills the window (a handful on
  OTTO and Amazon).
- **A seen item tied with the next item stays a target where seen items are excluded** (N43, entry 36): out of
  reach for every model alike.
- **ELSA's scores depend on the batch's composition by about 1e-6** (N48, entry 36).
- **Amazon is a 25% sample of Toys and Games' users** (entry 36), hashed, every history whole.
- **Matrix models train on the max of two windows' counts and read the sum at test** (entry 25, N5): the
  library's design, which matters on the repeat-heavy datasets only.
- **SASRec's negatives come from outside the whole history, as published** (entry 25, N2): seen items are never
  pushed down, which holds back re-consumption on repeat-heavy data.
- **The ablations reuse the stage-1 configuration at every level** (no retuning): the curves answer what the
  chosen model needs, not what is achievable with that much data. A retuning spot-check is planned after the
  main runs (final review E).
- **Users whose next item they already had are not scored where seen items are excluded** (entry 26, C4).
- **The budget (entry 32):** 10 search trials per model; SASRec trained for at most 100 epochs (the report
  flags a selected run still improving, entry 34, and such a selection is trained at 2× and 4× its epochs, entry 42); retraining ablations at one seed, with tests over users
  only, at 13 levels where there were 28, and on the datasets chosen after the timing.
- **At the smallest catalogues, the heaviest users are not scored where seen items are excluded**
  (entry 30, N49): they have seen all but fewer than k of the catalogue. Their number is in each run's
  metadata (`rows_too_few_unseen`).

## Open decisions

- SANSA: in (Mark, 2026-10-07). It comes from outside this repository as a model plugin, with its section in
  an extra protocol file (entry 39); its code and settings are private.
- The retuning spot-check of the ablations after the main runs, if time is left (entries 27 and 36).
- N8: SASRec's out-of-memory search corner (fixed with H35 in memory; kept here as a check on the DGX).
- SASRec's and BERT4Rec's prediction at window 400 with two or more heads does not fit a 6 GB card at
  `eval_batch_size = 1024` (entry 47): lower `eval_batch_size` on the RTX 2060 (no fingerprint), or give
  them SASRec v2's prediction in parts. Not changed while the week-long run uses their code.

## Decisions before the freeze (2026-09-23 to 09-25)

Recorded as "Decided" lines in `protocol.toml`, and described in the README:
- next-item targets, with the window as a diagnostic;
- training on every user (`train_users = "all"`, the library's `temporal_train_users`);
- ML-20M ratings of 3 and up;
- `exclude_seen` per dataset: ML-20M and Amazon true, the others false;
- search ranges centred on each model's published values;
- baselines as evaluation baselines, searched on validation and fixed across ablation levels, the strongest
  being the floor;
- the knee as a fixed-sequence one-sided sign-flip non-inferiority test, with margin `knee_margin`;
- randomisation p-values beside bootstrap intervals;
- event times recovered by a proved join rather than by re-splitting;
- `max_val_users` = 20,000;
- the ablation grids.
