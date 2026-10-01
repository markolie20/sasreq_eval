# seqrec-eval

The evaluation suite for the stage-1 model comparison (research design §5), and
the base the data ablations will build on. It runs on top of `compresso_recsys`
and lives outside it. The library supplies the datasets, the temporal split,
the models, the metrics and the paired statistics. This suite supplies the
frozen protocol, the random search, a resumable runner, the report, the CPU
latency benchmark, the §5.2 baselines the library lacks, the event timestamps
the saved split leaves out, and the data ablations.

Every decision made after the first protocol freeze, with its reasons, effects
and side effects, is recorded in [DECISIONS.md](DECISIONS.md). Reviews of the
suite follow `~/Documents/stage/review-plan/REVIEW_FRAMEWORK.md`: each has a
declared scope (by default the diff since the last snapshot), every finding is
classified by whether it can change a conclusion, and a review stops by rule.
In Claude Code, `/suite-review` runs one.

## Install

```bash
git clone <this repository> seqrec_eval && cd seqrec_eval
uv sync                                      # torch 2.14.0+cu126, and compresso-recsys from vendor/
pytest                                       # ~30 s, synthetic data, no downloads
SEQREC_EVAL_TEST_DEVICE=cuda pytest          # the same, every search, final and ablation run on the GPU
```

That is the whole installation, on the laptop and on the DGX alike. Plain `uv sync` and `uv run` are safe: the
lock holds exactly the library the suite was reviewed against.

**The library is vendored** (DECISIONS.md §29). `vendor/compresso-recsys` is a copy of
[compresso-recsys](https://github.com/zombak79/compresso-recsys) (Apache-2.0): release 0.3.7 plus the local
branch `local-temporal-train-all-users`, at version `0.3.7+trainusers`. It adds:
- **training on every user** (`temporal_train_users`), which the protocol's `train_users = "all"` and the refit
  before test need;
- **bounded training memory** (`loss_chunk_elements` in the GRU, SASRec and ELSA trainers: the same loss,
  computed in chunks), so the search spaces' largest settings fit a 32 GB V100 (§19);
- the branch's BERT4Rec work.

`uv sync` installs it as a fixed build from that folder, never from PyPI.
- `vendor/compresso-recsys/VENDORED.md` says where it was copied from (branch, commit, upstream) and lists every
  file that differs from upstream; `CHANGES.patch` holds the full difference.
- Changed library files start with a one-line notice, as the license asks.
- To take in later changes to the library, run `scripts/vendor-cr.sh [CHECKOUT]`, then `uv sync`; never edit the
  copy by hand.
- Nothing is pushed to the library itself.

The tests run every command end to end on a small synthetic dataset, through the installed library's real
builder. `prepare` refuses a library without `temporal_train_users` before it starts building, and every
`split_info.json` and run records the library version and a hash of its source.

torch comes from PyTorch's CUDA 12.6 index (`[tool.uv.sources]` in `pyproject.toml`), not from PyPI. PyPI's
Linux wheel is the CUDA 13 build, which has no kernels for the DGX's V100s (sm_70) and needs a newer driver than
the DGX's 550. The cu126 build covers sm_50 to sm_90, so it runs on the V100s and on the laptop alike.
Installing with `uv pip` instead of `uv sync`? Take torch from that index yourself:
`uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126`.

### A real run on the laptop

`scripts/local-run.sh DATASET MODEL [MODEL ...]` runs every step on real data for one dataset and the models
named: `prepare`, `analyse` (full data only), `search`, `final`, `latency`, `report`, `repeat-strata`, `status`,
and with `SWEEP=<name>` one ablation sweep with its analysis and report.

```bash
scripts/local-run.sh ml20m elsa gru                                  # quick: 2 trials, 1 seed, 1 epoch
SWEEP=history_length_inference scripts/local-run.sh ml20m elsa gru    # and one sweep
DRY=1 scripts/local-run.sh ml20m elsa gru                            # print the commands only
```

- **Quick by default.** It writes its own protocol next to the results (`work-local/<dataset>/protocol.toml`):
  `protocol.local.toml` with 2 trials per model, seed 0, and 1 epoch for ELSA, GRU and SASRec. The run checks
  that the chain works on real data; its numbers are not results. It has fingerprints of its own, so it
  cannot mix with a real run. `FULL=1` uses `protocol.local.toml` unchanged, in `work-local-full/`.
- **The library.** It uses the `.venv`'s, which `uv sync` installed from `vendor/compresso-recsys`. Set `CR_SRC`
  to a library source folder to try another one ahead of it.
- **Memory.** Every step runs under a 6 GB cap (`MEM`): a step that needs more is killed, not the laptop.
  ML-20M is the dataset to start with. Yambda (50M listens), Music4All and OTTO (whose 2% sample is rebuilt
  from the 11 GB `train.jsonl`) may not fit. EASE builds a dense item-by-item matrix, several GB for ML-20M's
  catalogue, so leave it out locally.
- Everything else: `DATA_DIR`, `DEVICE`, `REFERENCE` (default `elsa`, the model the report compares with, so
  include it), `TRIALS`, `SEEDS`, `EPOCHS`, `WORK`; see the script's header.

matplotlib is a dependency for the gap plots. Without it, the reports and CSVs
are still written; only the PNGs are missing.

## Workflow

Everything is driven by `protocol.toml`. **Review the lines marked `DECISION`,
then freeze the file before the first run.**

```bash
export COMPRESSO_DATA_DIR=/path/to/compresso-recsys/data   # raw downloads, manual ones included, stay where they are
export SEQREC_EVAL_WORK=/path/to/work        # splits, runs and reports; needs disk

seqrec-eval plan                             # what will run, and each run's fingerprint
seqrec-eval prepare                          # each temporal split once, then its event times (slow: OTTO re-parses)
seqrec-eval analyse                          # CPU: profile, baselines and floor, sequence signal, every condition
seqrec-eval analysis-report                  # work/reports/analysis.md + .csv

# Two GPUs: start the same command twice. Runs are locked, so the two
# processes divide the work between them and never run the same trial.
# On the DGX: work dir on /raid, numpy's threads capped (it ignores --threads), in tmux.
export SEQREC_EVAL_WORK=/raid/$USER/seqrec OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
nohup seqrec-eval search --device cuda:0 > search0.log 2>&1 &
nohup seqrec-eval search --device cuda:1 > search1.log 2>&1 &
seqrec-eval status                           # progress, failures, current best
# search, final and ablate exit 0 when all they planned is done, 1 when a run failed or was
# refused, 3 when work is left (the other GPU still running, a model not ready): so
# `search && final && ablate` stops where a step is unfinished.

seqrec-eval final --device cuda:0            # the selected configuration × every seed, with test
# final refuses while a model has a failed trial: rerun it (search --retry-failed),
# or, if it cannot run at all (out of memory, say), accept it: final --accept-failed
seqrec-eval latency --threads 4 --cores 16-19   # when the machine is quiet (see below)
seqrec-eval report --reference elsa          # work/reports/stage1.md + final_metrics.csv

seqrec-eval ablate --device cuda:0           # every [ablations.*] sweep; two GPUs share it as search does
seqrec-eval ablation-report                  # work/reports/ablation-<sweep>.md + -metrics.csv + -gap.csv/.png
seqrec-eval repeat-strata                    # work/reports/repeat-strata.md + .csv, from the stage-1 finals

# More seeds later, beside the runs already made (nothing is rerun):
seqrec-eval final --dataset amazon --add-seeds 3 4 --device cuda:0
seqrec-eval ablate --dataset amazon --sweep density --add-seeds 3 4 --device cuda:0
seqrec-eval analyse --dataset amazon --sweep density   # a random sweep's floor, on the new subsamples
```

Every command takes `--dataset` and `--model` to narrow what it does, e.g.
`seqrec-eval search --dataset amazon ml20m --model sasrec gru`.

### Adding seeds

Only the first seed is in any fingerprint: it seeds every search trial, so it
decides the selection. Every other seed adds one final run of the selected
configuration in a directory of its own. So seeds can be added once the first
round has run:

- `final --add-seeds 3 4` gives the selected datasets two more final seeds, for
  all their models (`--model` only narrows what this command runs). It is
  recorded in `work/runs/<dataset>/added_seeds.json`, and from then on every
  command and report uses the protocol's seeds followed by the added ones.
- `ablate --add-seeds 3 4` does the same for the selected sweeps on the selected
  datasets (`work/ablations/<sweep>/<dataset>/added_seeds.json`). Each seed must
  already be a stage-1 seed of the dataset, since its reference is that seed's
  stage-1 model. A sweep may keep fewer seeds than stage 1; its reference then
  uses only its own seeds.
- In a random sweep, a seed is also a subsample. An added seed adds one at every
  level, and the fixed test users must stay eligible under it, because every run
  already made was scored on them. Every current transform keeps them eligible by
  construction (see [Ablations](#ablations)), and one that did not would be refused
  before the seed is recorded. Run `analyse` for the sweep afterwards: the floor
  and controls are computed per subsample too.
- Asking for a seed that is already there changes nothing, so the command can be
  started once per GPU. Add *different* seeds from one command at a time.
- Listing more seeds in `protocol.toml` (keeping the first one first) adds them to
  every dataset instead, again without new fingerprints.

## The order of work

1. **`prepare`**: the temporal split, then the time of every event, proved against it.
2. **`analyse`**: CPU only, minutes per dataset, before any model. For the full
   data and again for every ablation condition:
   - **profile**: what the data looks like (the audit's questions: lengths,
     popularity, repeats, self-transitions, ties, gaps, spans, new items);
   - **baselines and the floor**: the non-learned baselines, each searched on
     validation like a model and scored on test on the models' terms. The
     strongest is the **floor**;
   - **sequence signal**: first-order Markov against itself fitted on
     shuffled training histories (order removed) and reversed ones (direction
     removed), and split by whether a test history's last two events tie in
     time. Where they do, which item is "last", the one Markov predicts from,
     was decided by the source file.
3. **`search` / `final`**: the learned models.
4. **`ablate`**: the learned models at every ablation level.

The analysis describes; it decides nothing about which datasets count. A
profile that contradicts what a transform is meant to do is a bug, and the
ablation report flags it (see *Manipulation check*).

## Models and baselines

`[models.*]` holds the learned models only: the ones under study and the
tuned incumbents. Each has a `family`, what it reads (`matrix` or `sequence`);
the ablation gap is each sequential model minus ELSA, fixed in advance, with the
best non-sequential model beside it as a descriptive line. `gru` is the
library's `SimpleRNN` with a GRU cell and a full softmax: GRU4Rec's architecture
and objective family, not GRU4Rec's sampled BPR-max/TOP1-max losses, so it is
reported as "GRU (full softmax)", not as GRU4Rec.

Two behaviours come with the library and are kept, documented (DECISIONS.md §25):
- **Matrix inputs (N5).** cr's builder trains ELSA and EASE on the larger of the
  two training windows' counts per user and item, and gives them the sum over the
  whole history at test. On the repeat-heavy datasets (Music4All, Yambda, OTTO) a
  model therefore reads larger counts at test than it was trained on. It is the
  library's design, not a choice of this suite.
- **SASRec's negatives (N2).** Sampled from outside the whole history, as
  published. On repeat-heavy data that never pushes a seen item down, which
  holds back re-consumption (0.64 of the oracle on a planted chain).
- Both sequence models search `unk_dropout` in {0, 0.02, 0.05, 0.1} (N7), so the
  embedding of items first seen after training is learned.

`[baselines.*]` holds the non-learned baselines of §5.2, from `baselines.py`.
They are evaluation baselines, not models: `analyse` scores them, and every
model is compared with the strongest of them, the floor, in the stage-1 report
(its own Holm family per dataset) and on every ablation's gap plot.

| baseline | `kind` | what it does | searched |
|---|---|---|---|
| `popularity` | `popularity` | the most popular training items, counting `events` or `users` | grid |
| `time_popularity` | `time_popularity` | popularity with each event weighted `0.5 ** (age / half_life_days)` | random search |
| `replay` | `replay` | the user's own history, by `recency` or `frequency` | grid |
| `markov` | `markov` | first-order transitions from the last item | nothing to search |

A space made only of choices small enough to enumerate is searched in full, as a
grid. Otherwise it gets `trials_per_model` random draws. Each baseline keeps its
full-data setting at every ablation level, as the models keep their stage-1
configuration. Each ranks in two strict tiers: items it has evidence for, then
popularity for the rest. They subclass the library's sequential base class, so
they mask seen items as its models do. Equal scores are common (the successors
a Markov chain saw once each, items with the same count), so they rank ties in
one fixed order: the more popular training item first, then the lower index.
Whichever way the evaluation asks for a list -- excluding seen items inside the
model, or asking for a longer one and excluding them after, as an ablation
condition does -- it gets the same list.

In an inference sweep every condition's training data is the full data's, so
the shuffled-Markov control is the full data's too, not a new shuffle per level.

## Targets

`[protocol].targets` decides what a user is scored against:

- **`next`** (the protocol's choice): the first thing the user does after their
  history ends. That is every item at the earliest timestamp in the window,
  since events can tie. This is next-item prediction, the research question.
- **`window`**: everything the user does in the phase's window, a year on
  ML-20M.

The chosen definition drives model selection, the floor and every statistic.
The other is scored beside it as a diagnostic: `test_<other>` in each final
run, and a table in the stage-1 and analysis reports. Next-item targets need
the event times (below).

Two rules make "next" mean the next thing the user actually did (H16):

- **The real first moment.** The library keeps an item first seen in a
  validation or test window only if at least `item_min_support` users have it,
  **counted inside that window**. A rare new item is deleted from the split.
  "Next" is therefore taken over all of the user's events in the window,
  including those on deleted items, and a deleted item leaves the row empty. It
  is not replaced by a later event: that would ask for the next-but-one item, a
  task that costs sequential models most. `split_info.json` records how many
  users this affects.
- **Only users a model could score.** Next-item metrics are computed over the
  users whose next item is in the training catalogue (for test, the refit
  catalogue). A user whose next item is new, or was deleted, scores 0 for every
  model: they change no comparison and only lower every mean. Each result's
  metadata counts them (`rows_unrecommendable_next`), and the stage-1 report
  states how many users each dataset was scored on and how many were left out.
  Search trials apply the same rule to the fixed validation sample, so a trial
  may score fewer than `max_val_users` users. The window diagnostic is scored
  on every user with a target, so its population is larger.

Test users scored, with refit (and before these rules):

| dataset | before | after |
|---|---|---|
| ML-20M | 3,321 | 2,942 |
| Amazon Toys_and_Games | 109,774 | 77,976 |
| Music4All | 30,369 | 30,180 |
| Yambda 50% | 4,373 | 3,840 |
| OTTO 2% | 18,458 | 14,605 |

The users removed on Yambda and OTTO are mostly those whose first test-window
event was a rare new item (7% and 19%); on ML-20M and Amazon, those whose next
item is new. The Amazon counts come from the stage emulation; the first
`prepare` records the real ones. Absolute scores rise, since fewer zeros are averaged, and paired
differences between models do not move for the users removed.

## Training on every user

The library's temporal split keeps, in each stage, only users with a history
before that stage's window and an event inside it. For validation and test
that is what makes a user scorable. For training it drops every user who was
inactive in the train window: on ML-20M that leaves 3,940 of 130,281 users.
`train_users = "all"` trains on every user with enough events before the
validation window, as §5.3 defines the training set. Validation and test are
unchanged.

## Refit before test

With `refit = true` in `[protocol]`, search trials still fit on the training
window and score validation. The configuration they select, and each baseline
and Markov control, is then fitted once more, per seed, on **everything before
the test window**, and only that model is tested. Without it, validation and
test are different tasks. A validation history ends where training ends, but a
test history runs a whole window further. So:

- items first seen in validation have no embedding and cannot be recommended;
- a test history is longer than any training sequence, which puts BERT4Rec's
  `[MASK]` on a position its training never used;
- the search, which sees neither, cannot select against them.

The refit applies everywhere a model is scored on test: `final`, `latency`,
the test side of `analyse`, and every ablation condition, which transforms the
refitted training data. A refitted final run records no validation score (its
model has seen that window), and scoring validation on refitted data raises an
error.

The library's split has no such training set, so `prepare` rebuilds it from the
builder's prepared events by the builder's own train-stage rule, one window
later:

- **users:** everyone with at least `min_user_support` distinct items before
  the test window;
- **catalogue:** the validation catalogue;
- **matrix:** the max of the two parts, as `x_train` is.

It **proves the rule every time**: the same code, given the train stage's
boundaries, must reproduce the split's `x_train`, `x_train_sequences`,
`train_source_sequences` and `train_user_ids` exactly, or `prepare` fails. The
set is stored as `x_refit*`, `refit_*` beside the split. A split prepared
before this existed gets it on the next `prepare`, without a rebuild. The
selected settings (epochs, λ, …) are applied unchanged to the larger training
set.

Side effects to keep in mind when reading results:

- **Validation scores come from the search.** A refitted final run records no
  validation score, and every final-run fingerprint includes `refit`.
- **More data per final fit.** The selected epochs, λ and so on are applied
  to a training set one window larger. The settings were chosen for the smaller
  set, which is the accepted trade-off of refitting.
- **Recency baselines move with it.** Time-decayed popularity counts age from
  the latest training event, so under refit that is the start of the test
  window.
- **More users have a recommendable next item** (H55): on Amazon Toys_and_Games
  51,639 → 77,976 of 109,774, on Yambda 20% 1,427 → 1,628 of 1,745.
- **BERT4Rec keeps one off-by-one.** Its `[MASK]` sits one position past the
  longest training sequence, so the users with exactly that history length
  still meet an untrained position (N1). On real data those are few.
- **Latency is measured on the refitted model.** Its catalogue is the
  validation catalogue.
- **It needs the library build with `_train_stage_args`**, the
  `temporal_train_users` branch.

## Timestamps

The library's saved split keeps each history's order but not its times, and
the time-decayed baseline and the analysis need them. `prepare` recovers them without changing the
library. It rebuilds the builder's prepared events (same adapter, options, seed
and rating threshold). It assembles every history from them by the builder's own
rule: events before the phase's boundary, whose user and item survived the
phase, sorted by (user, time) with a stable sort. It then **checks that the
result equals the split's history, item for item**. Only then are the times
saved, as `*_timestamps.npy` beside the split. Any difference fails `prepare`
with the first user and position that disagree, so times are never attached to
the wrong events. A split prepared before this existed gets its times on the
next `prepare`, without a rebuild. `--no-timestamps` skips the step, and then
`time_popularity` refuses to run. Every ablation transform carries each time
with its event. The same join, run on the validation and test target windows,
must reproduce the saved target matrices pair for pair. It then saves each
user's next-item targets as `val_next_target_matrix.npz` and
`test_next_target_matrix.npz`.

Times are in seconds, converted exactly as the library converts them (by
magnitude: milliseconds for Amazon, seconds for the others). The unit was
checked per dataset against each dataset's documented period and the typical
gap between a user's events (review H47). Yambda's clock starts at 0, since its
timestamps are seconds since the start of its log, so its dates read as 1970.
Windows and time decay use only differences, so that is harmless.

## What the protocol guarantees

- **Temporal split, from the library.** There are three consecutive windows at
  the end of the log. Search trials train on everything before the validation
  window, and tested models on everything before the test window (see "Refit
  before test"). Validation and test histories run up to their own window.
  Items first seen in a window are valid targets but cannot be recommended by
  any model (via `WarmCatalogAdapter`). With `refit = false`, every tested
  model is one window stale instead, and that should be stated in the report.
- **Test is never seen during selection.** Search trials score validation only,
  on one fixed sample of users per dataset. Test is scored only in the final
  runs of the configuration already selected.
- **Equal budget.** Every model gets `trials_per_model` random-search trials.
  Trial *i* is the same configuration on every dataset. `final` refuses to
  select from an unfinished search unless you pass `--allow-incomplete`.
- **Frozen means checked.** Each run is stored under a fingerprint of the
  protocol sections that determine its result. Editing a section starts fresh
  runs for exactly the (dataset, model) pairs it affects, and the report only
  reads runs that match the file as it stands. A code change to what a scored
  target means bumps `SCORING_VERSION` (`protocol.py`), which is part of every
  fingerprint, so results under the old meaning are never reused. Caches that
  hold *users* rather than runs, an ablation's fixed test users and
  per-condition profiles and the analysis profile, are keyed by the evaluation
  key: target definition, refit, scoring version and the dataset's
  `exclude_seen` (H04). Code changes that do not change a target's meaning are
  not in any fingerprint (H05); the baselines' code has its own version,
  `BASELINE_VERSION`. A split is identified by its build settings only, so
  flipping `exclude_seen` or `new_item_diagnostic` reuses it, and **every command
  refuses a split prepared from other build settings** (prepare it again with
  `--force`). A misspelt key in the protocol is refused by name, not left to fall
  back to its default.
- **Scoring is checked, not trusted.** A model fitted on train+validation
  refuses to be scored on validation, and a run refuses a split trained on the
  wrong data. Matrix models read their input projected onto their training
  catalogue: an item first seen later has no column in them, so a history of
  only such items reaches them empty and is ranked by tie order, not by
  popularity. Where an ablation keeps the full history for exclusion while the
  model reads a shortened one, each evaluation batch must be exactly the next
  rows of the input, every item read must be in that user's history, and every
  row must be used once. Otherwise the run fails instead of masking the wrong
  users (H33). Seen items are excluded after the model ranks, in every
  evaluation, from one list length for the whole phase: k plus the most items
  any user of the phase has seen. A model's tied scores then fall the same way
  whatever the batch size and whichever users are scored (N30).
- **Unattended runs are safe.** A run is finished only when `done.json` exists,
  and every file is written atomically and flushed to disk before it is renamed
  into place, so a power loss leaves the old file or the new one. Rerunning a
  command skips finished runs. A run is claimed with a kernel lock (`flock`),
  which dies with its process however that ends, so no claim outlives a crash or
  a reboot. A failure writes `failed.json` (with traceback) and is not retried
  unless you pass `--retry-failed`, so an out-of-memory configuration does not
  loop all weekend; a process killed outright (the kernel's out-of-memory
  killer) leaves no failure behind, so a run whose process dies twice is
  recorded as failed. A final made under another selection than the current one
  (after `--accept-failed` and a successful retry, say) is refused, not passed
  off as the new selection's: move it aside to redo it.
- **No model is chosen around a failure.** A failed trial, or one whose
  validation score is not a finite number, blocks `final` for that model: a
  search that silently lost trials would not be the equal budget the comparison
  rests on. Fix the cause and rerun. A failure that cannot be fixed (a
  configuration too large for the GPU) can be accepted with
  `final --accept-failed`. It is recorded in `accepted_failures.json` beside the
  model's runs, and the report lists it with its reason. A failed *final* seed
  is flagged ⛔ in the report. A baseline with a non-finite validation score
  stops `analyse`.
- **Every run records its code.** Each run's `done.json` holds the library's
  and the suite's version and a hash of their imported source, so it records
  what actually ran, whatever the installed metadata says. Imported from a
  source checkout (`PYTHONPATH=<checkout>/src`), the library's version is the
  checkout's own `pyproject.toml` version, with the metadata's beside it. The
  report states the build behind each dataset's results, and the protocol file
  by full path and hash, and warns when results come from more than one build
  (H05, H10).
- **Statistics: one test family.** The difference between two models is taken
  from the per-user values averaged over each model's seeds. Its uncertainty
  counts **both the users and the training seeds** (H23): the standard error adds
  the seeds' spread to the users' part, and p is read from Student's t with
  Welch–Satterthwaite degrees of freedom. That is Welch's own construction with a
  third variance term, what a mixed model with a random seed effect
  approximates. Intervals are that test's t interval, everywhere. Beside every p,
  the users-only p is the same test without the seed term: the paired t-test over
  users (standard in IR), so the effect of the seeds is visible. In stage 1 the
  models' seeds are independent. In an ablation they are **paired**: seed s of a
  level is the full data's seed s, retrained or rescored, and two models at one
  level of a random sweep share subsample s, so the seed term is the spread of the
  per-seed differences. Holm corrects **within each dataset**; across datasets
  nothing is pooled, and the stage-1 report counts per model the datasets where it
  is better, not different or worse (H51). Any number of seeds works: a model with
  one seed, or with identical seeds (popularity, EASE), adds no seed term, and the
  report says when a test is therefore over users only. (A two-level bootstrap was
  used for stage-1 intervals until 2026-10-01: with 3 seeds it was far too narrow,
  DECISIONS.md §27.)

## Ablations

Each `[ablations.<sweep>]` section applies one transform to the prepared split
at each of its levels. Every model uses its **stage-1 configuration** at each
level under every seed; nothing is searched again, so `ablate` waits for the
stage-1 search (and uses its final models) exactly as `final` does.

| transform | level | what it does | knee |
|---|---|---|---|
| `history_length` | *n* ≥ 1 | a context of *n* items: keep the last *n* + 1 events of every training history (its last event is only a target) and the last *n* of every input history | yes |
| `density` | *p* in (0, 1) | keep a random *p* of each history's events, spread over its whole length; every training item keeps one event (`keep_catalogue`) | yes |
| `repeat_removal` | *q* in (0, 1] | remove each repeat event (item already earlier in the history) with probability *q*; first occurrences stay | yes, most removal within δ |
| `shuffle` | block size ≥ 2, or `"all"` | shuffle order within consecutive blocks; matrices untouched, so the matrix models are a control | no |
| `catalogue` | item count, or fraction | keep the top items by training popularity (`strategy = "top"`), or a random draw within popularity strata (`"stratified"`, `strata`), nested across levels; removed items leave the item space, histories and targets; each level is scored on its own users | no |

Random transforms (`density`, `repeat_removal`, `shuffle`, stratified
`catalogue`) draw one subsample per seed from a stream of their own. Each
phase is subsampled independently, because the split carries no event identity
across phases. Subsample *s* is fitted with model seed *s*, so the seed spread
of a random condition holds both the draw and the training. None of the random
transforms can change which test users are eligible. They drop or reorder events
inside a fixed item space, so a next item stays recommendable, and every non-empty
history keeps at least one event. The fixed test users are therefore the same
whichever subsamples are drawn, and seeds can be added to these sweeps later
(`test_rows.json` records the subsamples they were checked on).

- **Scope.** `scope = "all"` (the default) transforms training data and
  inference histories alike, and refits. `scope = "inference"` leaves training
  untouched and transforms only the histories given at recommendation time. Each
  condition then rescores the stage-1 models, with nothing refitted. It answers
  what a trained model needs at serving time, which is the serving-cost
  question. `catalogue` supports only `"all"`.

- **Only inputs change.** Transforms edit the training data and the validation
  and test input histories. Test targets are left alone, except in a catalogue
  sweep. Matrices are rebuilt from the kept events exactly as the library's
  builder would have (`x_train` is the maximum of the train source and target
  windows, not a count).
- **One set of test users, except in the catalogue sweeps.** Every level and the
  reference are scored on the test users eligible at every level, stored in
  `test_rows.npy`: a history left, and a next item a model can recommend, as in
  stage 1. The catalogue sweeps remove targets, and a user whose next
  target is a known item survives every random catalogue only by chance, so a
  fixed set shrinks to users whose targets are all unseen, who score 0 for every
  model (H01b). There, each condition is scored on **its own users**: a known
  next target still in its catalogue, and a history item left. Seeds of the
  stratified catalogue score different users and are pooled per user. Models are
  compared within a level, through the gap, and the level-against-full test is
  not run.
- **Nested catalogues.** Within a seed, a smaller catalogue is part of every
  larger one: the stratified sweep draws one random order per seed that keeps
  each popularity stratum in proportion at every prefix, and each level takes a
  prefix. The levels then differ in size, not in which random items were drawn.
- **Too few users.** A level scored on fewer than `min_level_users` users
  (default 1,000) is marked *descriptive*, drawn hollow in the gap plot, and no
  claim rests on it.
- **The reference is stage 1.** `full` reloads each seed's stage-1 `model.zip`
  and rescores it on those users. It is not refitted.
- **Manipulation check.** History length, catalogue, density, popularity Gini
  and repeat rate are measured for the training data and the scored test inputs
  of every condition. Each transform declares what it moves by construction
  besides its target (for example, truncation also lowers density and repeats,
  and leaves fewer distinct items with a different popularity spread),
  and `expected_to_move` in the sweep overrides that declaration. The report
  flags only undeclared characteristics that move by more than
  `manipulation_tolerance`. Beside them it shows how much the condition actually
  touched: *rows changed* (the share of histories that differ at all) and
  *span kept* (how much of the original history lies between the first and last
  kept event). Span kept is low after truncation and high after thinning, which
  the five characteristics cannot tell apart. The five are the first fields of
  the analysis's data profile: one computation, so the description and the
  check cannot disagree. Each transform also states which way its target must
  move. A condition where it moved the other way is marked ⛔ as a probable bug,
  and that sweep's results should not be read until it is explained.
- **Report.** One Holm family per sweep and dataset: every level of every model
  against its reference, with the seed-aware test, seeds paired. **The gap** is
  each sequential model minus ELSA (`ablation-report --reference`), at every
  level, tested and Holm-corrected across the sweep: that is the study's
  question. Beside it, descriptive only, the gap to the best non-sequential model
  at that level, chosen on test. Every condition also has its analysis: each
  baseline and the floor, the shuffled and backwards Markov controls, and Markov
  split by tied history ends; a random sweep's condition is shown only once every
  subsample is analysed. The gap plot draws the floor on the same scale, so a
  model under it has not beaten the floor. Runs that failed or have not finished
  are listed per model, never dropped silently, and a level scored on fewer than
  `min_level_users` users gets "descriptive" in place of a verdict.
  `ablation-<sweep>-gap.csv` is the plot input.
- **The knee.** δ is `knee_margin` (10%) of the model's score on the **full
  data**, fixed in advance. Walking from the full data to ever more reduced
  levels, the knee is the most reduced level whose 90% interval for the loss
  (level − full) stays above −δ, stopping at the first level whose interval does
  not: a one-sided seed-aware t-test at α = 0.05 per level, seeds paired, in a
  fixed order. The knee starts at the full data and is the last level that
  passed, so when nothing could be shown it is the full data. The report also
  gives the knee at δ = 5%, 10% and 20%, as a sensitivity, not the result. Levels are compared with the full data and not with the
  best-looking level, which is lucky by construction and would bias every test
  against the others. A short level cannot become the knee through one lucky
  result below a level that failed. The fixed testing order keeps the
  family-wise error at α without correction.
- **The knee's power.** Beside each knee, the report gives the chance that the
  test would show "within δ" for a level that loses nothing:
  P(T_df > t(0.95, df) − δ / SE), with SE and df the seed-aware test's, seeds
  paired. Where it is low, a knee at the full data means the test could not
  tell, not that the history is needed. With the seeds agreeing and many users
  this is Φ(δ·√n / s − 1.645), n test users and s the spread of the per-user
  differences, so the smallest margin that reaches 90% power is about
  δ = 2.93 · s / √n.
  For `repeat_removal` the axis runs the other way: less removal is closer to
  the full data.
- **Catalogue targets.** A catalogue sweep drops the targets of removed items,
  so its levels are scored against different targets from the full data. The
  report sets the library's target fingerprints aside for that sweep only;
  users are still paired.
- **Fingerprints.** A sweep's transform, levels, options and scope, plus the
  transform's version, fingerprint its conditions. Each run adds its stage-1 run
  fingerprint. `datasets`, `models`, `expected_to_move`, the tolerance and the
  knee margin only choose what runs or how the report reads it.

To add a transform, register it in `ablations.py`. Declare its level validator,
its target, what it is expected to move, its scopes, whether it changes targets,
and its knee axis. A transform that only drops events builds masks for
`keep_events`.

### Repeat strata

`repeat-strata` refits and rescores nothing. It cuts the stage-1 final
results into cells by each test user's history length and repeat share (the
share of events that repeat an earlier item). For each cell it gives each
model's mean and each sequential model's gap to the best non-sequential
model, with a paired bootstrap interval over users (descriptive cells, so
users only, of the seed-averaged values). That model is chosen once per dataset
on all test users. The cells are observational, so they carry no test:
`repeat_removal` is the manipulated counterpart. Settings are in
`[repeat_strata]`.

## Latency

`latency` loads each final model on CPU and times one request at a time
against the full test catalogue, reporting P50/P95/P99 per history-length bin
(NFR-03). Only the model call is timed. **Measure when no training is running,
or pin the benchmark to cores nothing else uses (`--cores`).** A P95 measured
next to training jobs is a measurement of the contention. `latency.json` records
the machine's load before and after, and which trial's configuration was timed.
The report's worst-bin P95 counts only bins with at least 50 requests. Latency is
measured again every time the command runs, so the last run's numbers stand:
state the hardware with them (the DGX's CPU is a 2016 Xeon E5-2698 v4).

## Layout

```
work/splits/<dataset>/                      extracted split + split_info.json (resolved settings, stage sizes)
                                            + *_timestamps.npy, one per history view, proved against the split
                                            + x_refit*, refit_* (refit = true): train+validation, proved likewise
work/runs/<dataset>/<model>/<fp>/trial-007/ spec.json, done.json | failed.json, val.{json,npz}, .lock (flock),
                                            attempts.json while a run is under way (2 deaths: failed)
work/runs/<dataset>/<model>/<fp>/incomplete_selection.json   final --allow-incomplete was used
work/tmp/                                   the library's temporary files (unless TMPDIR is set)
work/runs/<dataset>/<model>/<fp>/final-seed1/  test, test_new (diagnostic), model.zip, latency.json; no val under refit
work/runs/<dataset>/added_seeds.json        seeds added with final --add-seeds (which, when, where)
work/reports/stage1.md, final_metrics.csv
work/ablations/<sweep>/<dataset>/added_seeds.json   seeds added with ablate --add-seeds
work/ablations/<sweep>/<dataset>/<cfp>/test_rows.npy, test_rows.json (subsamples checked), conditions/<level>[/seedS].json (characteristics)
work/ablations/<sweep>/<dataset>/<cfp>/runs/<model>/<fp>/<level|full>/final-seed1/   done.json, test.{json,npz}
work/reports/ablation-<sweep>.md, ablation-<sweep>-metrics.csv, ablation-<sweep>-gap.csv, ablation-<sweep>-gap.png
work/reports/repeat-strata.md, repeat-strata.csv
work/analysis/<dataset>/<dfp>/profile-<evaluation key>.json, baselines/<name>/<bfp>/{trial-*,selected}.json + test, controls/
work/ablations/<sweep>/<dataset>/<cfp>/analysis/<scorer>/<fp>/<condition>/test.{json,npz}
work/reports/analysis.md, analysis.csv
```

## Not in the suite yet

| Missing | How it enters |
|---|---|
| Mamba4Rec, ComiRec | implement against the trainer contract, register in `models.py`, add a `[models.*]` section |
| SANSA (Recombee's implementation) | a matrix-family builder in `models.py` |
| GRU4Rec proper | `gru` is the library's `SimpleRNN`: GRU4Rec's objective, but full softmax and no BPR-max/TOP1-max loss |
| Early stopping | not part of the library's trainer contract; `epochs` is a searched hyperparameter instead |
| Time-based manipulation checks | possible now that timestamps are recovered: span and rate in days, beside *span kept* |
