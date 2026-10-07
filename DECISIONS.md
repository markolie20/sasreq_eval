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
- How many users this affects has not been counted yet. It is a DGX check.
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

---

## Known limitations recorded by the review

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
- **Matrix models rank an all-cold history by tie order** (entry 3); not yet counted.
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
  flags a selected run still improving, entry 34); retraining ablations at one seed, with tests over users
  only, at 13 levels where there were 28, and on the datasets chosen after the timing.
- **At the smallest catalogues, the heaviest users are not scored where seen items are excluded**
  (entry 30, N49): they have seen all but fewer than k of the catalogue. Their number is in each run's
  metadata (`rows_too_few_unseen`).

## Open decisions

- SANSA: in (Mark, 2026-10-07). It comes from outside this repository as a model plugin, with its section in
  an extra protocol file (entry 39); its code and settings are private.
- The retuning spot-check of the ablations after the main runs, if time is left (entries 27 and 36).
- N8: SASRec's out-of-memory search corner (fixed with H35 in memory; kept here as a check on the DGX).

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
