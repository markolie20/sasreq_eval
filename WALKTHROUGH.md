# seqrec_eval: a step-by-step walkthrough of the code

This document follows every command of the suite through the code, in the order the code runs, and says
which file and line each thing happens in. It describes the code as it stands on 2026-10-05. Line numbers
go stale when the code changes; the function names do not, so search for the name if a link lands a few
lines off.

**How to read the references**

- `cli.py:225-258` style links are relative to the repository root and open the file at those lines in
  VS Code (Ctrl+click in the editor, or click in the Markdown preview).
- **cr** is compresso-recsys, the library the suite is built on. The suite runs its own copy of it,
  vendored in [vendor/compresso-recsys/](vendor/compresso-recsys/) (version `0.3.7+trainusers`: release
  0.3.7 plus the local branch's `temporal_train_users`, bounded training memory, BERT4Rec, and EASE
  scoring without copying its weights;
  [VENDORED.md](vendor/compresso-recsys/VENDORED.md) says where it came from). `uv sync` installs that copy
  ([pyproject.toml:39-41](pyproject.toml#L39-L41)), so every library link below points into it. Each Python
  file the suite's copy changed starts with a one-line notice, so its line numbers are one more than in the
  library's own repository.
- A **phase** is `train`, `val` or `test`. A **row** is one user in one phase. A phase's **catalogue** is
  its list of item ids (`train_item_ids`, `val_item_ids`, `test_item_ids`); each is a prefix of the next,
  because each window only appends the items first seen in it.
- A **fingerprint** is a SHA-256 of the settings that determine a result
  ([protocol.py:94-96](src/seqrec_eval/protocol.py#L94-L96)). Results are stored in a folder named after its
  first 12 characters, so a changed setting writes to a new folder instead of mixing with old results.
  Appendix B lists every fingerprint.

## The pipeline at a glance

```
command          what it does                                              writes under work/
---------------  --------------------------------------------------------  ---------------------------------------------
plan             print what the protocol will run                          nothing
prepare          build each dataset's temporal split once, with times      splits/<dataset>/
analyse          profile, baselines and floor, sequence-signal controls    analysis/<dataset>/..., ablations/*/*/*/analysis/
analysis-report  the analysis report                                       reports/analysis.md, analysis.csv
search           random-search trials, scored on validation                runs/<dataset>/<model>/<fp>/trial-NNN/
final            selected configuration x seeds, refitted, scored on test  runs/<dataset>/<model>/<fp>/final-seedS/
status           progress table                                            nothing (prints)
report           stage-1 report: comparisons, floor, latency, datasets     reports/stage1.md, final_metrics.csv
latency          CPU latency of the saved final models                     runs/.../final-seed<first seed>/latency.json
ablate           every sweep, on the stage-1 configurations                ablations/<sweep>/<dataset>/<cfp>/
ablation-report  one report per sweep, the gap plot, CSVs                  reports/ablation-<sweep>*.md|csv|png
repeat-strata    stage-1 results cut by repeat share and history length    reports/repeat-strata.md, .csv
```

What depends on what: every command except `plan` needs `prepare` first. `analyse` needs only `prepare`
and runs on CPU. `final` needs a finished `search`. `report` reads the finals and the floor from
`analyse`. `latency` needs the finals' saved `model.zip`. `ablate` needs the finals too, because its
full-data reference reloads their `model.zip`. `ablation-report` reads what `ablate` and `analyse` wrote.
`repeat-strata` reads only the finals.

`search`, `final` and `ablate` exit with 0 when everything they planned is done, 1 when a run failed or was
refused, and 3 when work is left over, so `search && final && ablate` stops at the first step that has not
finished (Step 0.1). Seeds can be added after the protocol's have run: `final --add-seeds` and
`ablate --add-seeds` (Steps 6.1 and 10.1). Four scripts drive the commands: a quick real run on the laptop,
the whole suite on the DGX, a run-time estimate, and the refresh of the vendored library (Part 13).

---

# Part 0: what every command does first

## Step 0.1: the entry point

**code:**
`seqrec-eval` is the console script `seqrec_eval.cli:main` ([pyproject.toml:27-28](pyproject.toml#L27-L28));
`python -m seqrec_eval` does the same through [__main__.py:1-5](src/seqrec_eval/__main__.py#L1-L5).
`main` is at [cli.py:200-207](src/seqrec_eval/cli.py#L200-L207) and runs `_main`
([cli.py:210-436](src/seqrec_eval/cli.py#L210-L436)) inside `_operator_stops`
([cli.py:182-197](src/seqrec_eval/cli.py#L182-L197)).

**explanation:**
There is one entry function for every command. `_main` parses the command line
([cli.py:211](src/seqrec_eval/cli.py#L211)), loads the protocol ([cli.py:212](src/seqrec_eval/cli.py#L212)),
points the temporary directory at the work folder (below), resolves which datasets, models and sweeps the
command is about ([cli.py:219-223](src/seqrec_eval/cli.py#L219-L223), Step 0.4), and then branches on the
command name ([cli.py:225-435](src/seqrec_eval/cli.py#L225-L435)).

**Stopping a command.** `_operator_stops` makes `kill` (SIGTERM) and a closed terminal (SIGHUP) raise
`KeyboardInterrupt`, as Ctrl-C does ([cli.py:178-179](src/seqrec_eval/cli.py#L178-L179)). A SIGHUP that
`nohup` already ignores stays ignored ([cli.py:191](src/seqrec_eval/cli.py#L191)). `main` catches the
interrupt, says the run in progress was not counted as a failed attempt, and returns 130
([cli.py:204-207](src/seqrec_eval/cli.py#L204-L207)). This matters because the runner counts a run whose
process died without a word as an attempt, and fails it after two (Step 5.3). An operator's stop must not
count as one (review N31). SIGKILL (`kill -9`) and the kernel's out-of-memory killer cannot be caught, and
still count.

**The temporary directory** ([cli.py:214-218](src/seqrec_eval/cli.py#L214-L218)). The library stages every
model save and load in a temporary directory. Unless `TMPDIR` is set, that is `work/tmp/`, so it lands on the
work folder's disk (on the DGX the large data disk, not the small shared `/`; review B6).

**Exit codes.** `search`, `final` and `ablate` count the status every run returned and end with `_exit_code`
([cli.py:487-496](src/seqrec_eval/cli.py#L487-L496)):

| code | when | statuses |
|---|---|---|
| 0 | everything planned is done (or was skipped as too large) | `done`, `cached`, `skipped` |
| 1 | a run failed or was refused | `failed`, `stale-selection`, `stage1-failed` (`_WRONG`) |
| 3 | work is left over | `not-ready`, `running-elsewhere`, `failed-before`, `waiting-for-stage1` (`_UNFINISHED`) |
| 130 | stopped by the operator | |

1 wins over 3. Every other command returns 0, and an unknown command 2
([cli.py:436](src/seqrec_eval/cli.py#L436); in practice `argparse` refuses it first, also with 2).

## Step 0.2: parsing the command line

**code:**
`_parser()` at [cli.py:95-175](src/seqrec_eval/cli.py#L95-L175).

**explanation:**
Two options apply to every command:

- `--protocol` (default `protocol.toml`, [cli.py:98](src/seqrec_eval/cli.py#L98)): which protocol file to
  load. `protocol.local.toml` is the laptop's copy. It must be identical to `protocol.toml` apart from
  `eval_batch_size` (128 instead of 1024), which enters no fingerprint, so local and DGX runs share their
  folders. Tests enforce this ([tests/test_smoke.py:481-486](tests/test_smoke.py#L481-L486)).
- `--work-dir` (default `$SEQREC_EVAL_WORK`, else `./work`, [cli.py:99-100](src/seqrec_eval/cli.py#L99-L100)):
  where splits, runs and reports live.

The helper `selection()` ([cli.py:103-106](src/seqrec_eval/cli.py#L103-L106)) adds `--dataset` and, for
commands that concern models, `--model`. Per command:

| command | options | lines |
|---|---|---|
| `plan` | none | [cli.py:108](src/seqrec_eval/cli.py#L108) |
| `prepare` | `--dataset`, `--data-dir` (default `$COMPRESSO_DATA_DIR`), `--force`, `--quiet`, `--no-timestamps` | [cli.py:110-117](src/seqrec_eval/cli.py#L110-L117) |
| `search`, `final` | `--dataset`, `--model`, `--device`, `--threads`, `--retry-failed`; `final` also `--allow-incomplete`, `--accept-failed`, `--add-seeds SEED...` | [cli.py:119-134](src/seqrec_eval/cli.py#L119-L134) |
| `status` | `--dataset`, `--model` | [cli.py:136-137](src/seqrec_eval/cli.py#L136-L137) |
| `report` | `--dataset`, `--model`, `--reference` (default `elsa`) | [cli.py:139-141](src/seqrec_eval/cli.py#L139-L141) |
| `latency` | `--dataset`, `--model`, `--threads` (default 4), `--cores` | [cli.py:143-146](src/seqrec_eval/cli.py#L143-L146) |
| `analyse` | `--dataset`, `--sweep` (`none`: the full data only), `--threads` | [cli.py:148-152](src/seqrec_eval/cli.py#L148-L152) |
| `analysis-report` | `--dataset` | [cli.py:154-155](src/seqrec_eval/cli.py#L154-L155) |
| `ablate` | `--dataset`, `--model`, `--sweep`, `--device`, `--threads`, `--retry-failed`, `--add-seeds SEED...` | [cli.py:157-165](src/seqrec_eval/cli.py#L157-L165) |
| `ablation-report` | `--dataset`, `--model`, `--reference` (default `elsa`), `--sweep` | [cli.py:167-171](src/seqrec_eval/cli.py#L167-L171) |
| `repeat-strata` | `--dataset`, `--model` | [cli.py:173-174](src/seqrec_eval/cli.py#L173-L174) |

`--device` defaults to `cuda` when torch sees a GPU, else `cpu` (`_default_device`,
[cli.py:91-92](src/seqrec_eval/cli.py#L91-L92)). `--threads` sets torch's CPU threads for the process.

## Step 0.3: loading the protocol

**code:**
`load_protocol(args.protocol)` at [cli.py:212](src/seqrec_eval/cli.py#L212) runs
[protocol.py:447-526](src/seqrec_eval/protocol.py#L447-L526).

**explanation:**
The protocol file is the frozen description of the whole study: what is measured, on which data, with
which models and baselines, and which ablations run. `load_protocol` reads the TOML
([protocol.py:448-450](src/seqrec_eval/protocol.py#L448-L450)), checks every section, and returns one
frozen `Protocol` object ([protocol.py:208-293](src/seqrec_eval/protocol.py#L208-L293)) that every other
module reads. A mistake in the file fails here, before any data is touched.

**Unknown keys are refused.** Every section is checked against the keys it may hold, `_ALLOWED_KEYS`
([protocol.py:69-82](src/seqrec_eval/protocol.py#L69-L82)), by `_check_keys`
([protocol.py:85-91](src/seqrec_eval/protocol.py#L85-L91)). A misspelt key would otherwise fall back to its
default without a word: `scop = "inference"` would refit every model at every level. The error names the
unknown key and suggests the closest allowed one ("did you mean 'scope'?"). The top level
([protocol.py:451](src/seqrec_eval/protocol.py#L451)), `[protocol]` ([protocol.py:453](src/seqrec_eval/protocol.py#L453)),
`[latency]` ([protocol.py:455](src/seqrec_eval/protocol.py#L455)) and `[repeat_strata]`
([protocol.py:460](src/seqrec_eval/protocol.py#L460)) are checked here, every other section in its own parser.
`[latency].history_bins` must be positive integers in increasing order
([protocol.py:456-459](src/seqrec_eval/protocol.py#L456-L459)). The steps below go through the file section
by section.

### Step 0.3.1: the `[protocol]` section

**code:**
[protocol.py:462-494](src/seqrec_eval/protocol.py#L462-L494); the values in
[protocol.toml:14-54](protocol.toml#L14-L54).

**explanation:**
Each key is read and checked. `_require` ([protocol.py:103-106](src/seqrec_eval/protocol.py#L103-L106))
raises `ProtocolError` for a missing key.

| key | value now | check | what it means |
|---|---|---|---|
| `version` | 1 | required integer ([protocol.py:509](src/seqrec_eval/protocol.py#L509)) | the protocol's own version |
| `cutoffs` | [5, 10, 20] | positive integers, sorted and de-duplicated ([protocol.py:462-464](src/seqrec_eval/protocol.py#L462-L464)) | the k of every metric@k; the model is always asked for the top max(cutoffs) = 20 |
| `metrics` | ndcg, recall, calibrated_recall, hit_rate, precision, map, mrr | each must be in `METRIC_NAMES` ([protocol.py:32](src/seqrec_eval/protocol.py#L32), [protocol.py:465-468](src/seqrec_eval/protocol.py#L465-L468)) | the metric families computed; `evaluate.py` checks at import that these are exactly the library's names ([evaluate.py:65-67](src/seqrec_eval/evaluate.py#L65-L67)) |
| `primary_metric` | `ndcg@10` | must match `<metric>@<k>` with the metric listed and k a cutoff ([protocol.py:470-476](src/seqrec_eval/protocol.py#L470-L476)) | the one metric that selects configurations, sets the floor and drives every test |
| `seeds` | [0, 1, 2] | non-empty, distinct, non-negative ([protocol.py:478-481](src/seqrec_eval/protocol.py#L478-L481)) | final runs per selected configuration; the first also seeds every search trial; the data seeds of stochastic ablations |
| `trials_per_model` | 10 | integer ([protocol.py:482](src/seqrec_eval/protocol.py#L482)) | default random-search budget per model and baseline (10, was 20: the one-week budget) |
| `search_seed` | 20260923 | required ([protocol.py:515](src/seqrec_eval/protocol.py#L515)) | the root of every random stream in the suite (search draws, validation sample, shuffles, ablation subsamples, latency samples, strata resamples) |
| `max_val_users` | 20000 | optional ([protocol.py:483](src/seqrec_eval/protocol.py#L483)) | size of the fixed validation sample every search trial is scored on |
| `targets` | `"next"` | required, `"next"` or `"window"` ([protocol.py:485-491](src/seqrec_eval/protocol.py#L485-L491)) | what a user is scored against: the first thing they do after their history (`next`), or everything in the window (`window`); the other one is computed as a diagnostic |
| `refit` | `true` | required boolean ([protocol.py:485-494](src/seqrec_eval/protocol.py#L485-L494)) | whether every model scored on test is first refitted on train + validation |
| `eval_batch_size` | 1024 | default 1024 ([protocol.py:519](src/seqrec_eval/protocol.py#L519)) | rows per evaluator batch; speed only, so it is in no fingerprint |

`targets` and `refit` have no default on purpose: a missing key once silently ran a different protocol
(review H11). `[latency]` ([protocol.toml:56-61](protocol.toml#L56-L61)) is stored as a plain dict
([protocol.py:454](src/seqrec_eval/protocol.py#L454)) and read only by the latency benchmark.

### Step 0.3.2: the datasets

**code:**
[protocol.py:496](src/seqrec_eval/protocol.py#L496) calls `_parse_dataset`
([protocol.py:306-336](src/seqrec_eval/protocol.py#L306-L336)) for every `[datasets.<name>]`, which returns a
`DatasetProtocol` ([protocol.py:109-155](src/seqrec_eval/protocol.py#L109-L155)). The five datasets are at
[protocol.toml:85-151](protocol.toml#L85-L151).

**explanation:**
Each field of `DatasetProtocol`:

| field | example (ml20m) | meaning |
|---|---|---|
| `name` | `ml20m` | the key used everywhere in the suite and in folder names |
| `builder` | `ml20m` | the library's dataset name (`amazon2023`, `music4all-onion`, `yambda`, `otto` for the others) |
| `temporal_period_hours` | 8136 (339 days) | width of each of the three windows at the end of the log; must be positive ([protocol.py:312-314](src/seqrec_eval/protocol.py#L312-L314)) |
| `min_user_support` | 5 | a user needs this many distinct items in a stage to be kept |
| `item_min_support` | 1 | an item first seen in a stage needs this many users in it to be kept |
| `min_value_to_keep` | 3.0 | rating threshold; `"none"` keeps everything ([protocol.py:296-303](src/seqrec_eval/protocol.py#L296-L303)) |
| `set_all_values_to` | 1.0 | every kept event gets this value |
| `exclude_seen` | true | whether items already in the history may be recommended (true for ML-20M and Amazon, false for Music4All, Yambda, OTTO) |
| `new_item_diagnostic` | false | whether finals are also scored on targets not already in the history (true where `exclude_seen` is false) |
| `train_users` | `"all"` | who the training stage keeps: `"all"` = everyone with enough events before the validation window, `"window"` = only users also active in the train target window. Required, no default ([protocol.py:316-321](src/seqrec_eval/protocol.py#L316-L321)) |
| `amazon_category` | (amazon only) `Toys_and_Games` | which Amazon category |
| `options` | (music4all, yambda, otto) | extra dataset options for the library: Music4All's 2014 window, Yambda `user_sample = 0.5`, OTTO `session_sample = 0.02` |
| `raw` | the whole TOML table | what the fingerprints hash |

`build_parameters()` ([protocol.py:128-155](src/seqrec_eval/protocol.py#L128-L155)) turns this into the
keyword arguments for the library's builder. Two details matter. "Keep everything" is sent as `-inf`, not
`None`, because the builder reads `None` as "use the registry default", which for ML-20M is a 4-star
threshold ([protocol.py:142-144](src/seqrec_eval/protocol.py#L142-L144)). And `temporal_train_users` is only
passed when it is not the library default `"window"` ([protocol.py:153-154](src/seqrec_eval/protocol.py#L153-L154)),
so a library without that option still works for a `"window"` protocol.

### Step 0.3.3: the models

**code:**
[protocol.py:497](src/seqrec_eval/protocol.py#L497) calls `_parse_model`
([protocol.py:362-382](src/seqrec_eval/protocol.py#L362-L382)) for every `[models.<name>]`, returning a
`ModelProtocol` ([protocol.py:158-166](src/seqrec_eval/protocol.py#L158-L166)). The models are at
[protocol.toml:176-246](protocol.toml#L176-L246).

**explanation:**
- `family` must be `"matrix"` or `"sequence"` ([protocol.py:365-367](src/seqrec_eval/protocol.py#L365-L367)).
  A matrix model trains on the user × item CSR matrix, a sequence model on the ordered histories.
- `fixed` is passed unchanged to every trial; `space` is searched. Each `space` entry must be one
  distribution, checked by `_check_distribution` ([protocol.py:339-359](src/seqrec_eval/protocol.py#L339-L359)):
  `choice = [...]` (non-empty list), `uniform = [low, high]`, `loguniform = [low, high]` (low > 0), or
  `int = [low, high]` (inclusive).
- A parameter cannot be both fixed and searched ([protocol.py:372-374](src/seqrec_eval/protocol.py#L372-L374)).
- `trials` defaults to `trials_per_model` when there is a space, else 1
  ([protocol.py:375](src/seqrec_eval/protocol.py#L375)). So `popularity` (no space) has 1 trial; `ease`,
  `elsa`, `gru` and `sasrec` have 10 each.
- `max_items` skips the model on catalogues larger than this (EASE: 40,000, because it builds a dense
  item × item matrix; it is skipped on Amazon, Yambda and OTTO).

### Step 0.3.4: the ablations and the baselines

**code:**
Ablations: [protocol.py:502-503](src/seqrec_eval/protocol.py#L502-L503) → `_parse_ablation`
([protocol.py:411-444](src/seqrec_eval/protocol.py#L411-L444)) → `AblationProtocol`
([protocol.py:186-205](src/seqrec_eval/protocol.py#L186-L205)).
Baselines: [protocol.py:504-505](src/seqrec_eval/protocol.py#L504-L505) → `_parse_baseline`
([protocol.py:385-408](src/seqrec_eval/protocol.py#L385-L408)) → `BaselineProtocol`
([protocol.py:169-183](src/seqrec_eval/protocol.py#L169-L183)).

**explanation:**
An ablation is a `transform`, a non-empty list of distinct `levels`, optional `options`, which `datasets` and
`models` it covers (default: all of them; unknown names fail,
[protocol.py:423-429](src/seqrec_eval/protocol.py#L423-L429)), and which `seeds` it runs under
([protocol.py:430-439](src/seqrec_eval/protocol.py#L430-L439)). The seeds default to the protocol's; a sweep
may name a subset (every retraining sweep now runs under seed 0 only), but never a seed stage 1 does not
have, because a condition of seed s reuses stage 1's model or configuration of seed s. The seeds are in no
fingerprint, so `ablate --add-seeds` can grow them later (Step 10.1). Whether the levels make sense is not
checked here but by the transform itself (Step 0.4), because only it knows what a level means. The other keys
of the section (`scope`, `knee_margin`, `expected_to_move`, `manipulation_tolerance`, `min_level_users`) stay
in `raw` and are read by `ablations.py` and `ablation_report.py`.

A baseline has a `kind`, one of `popularity`, `time_popularity`, `replay`, `markov`
([protocol.py:28](src/seqrec_eval/protocol.py#L28)), and a `space` checked like a model's. The parser
decides how it is searched ([protocol.py:398-407](src/seqrec_eval/protocol.py#L398-L407)): if every
parameter is a `choice`, the grid size is the product of the choice lengths; a continuous parameter makes
it infinite. A grid no larger than `trials_per_model` is searched **in full** (`grid = True`, and `trials`
must then equal the grid size); otherwise it is random search with `trials_per_model` trials. With the
current protocol ([protocol.toml:258-274](protocol.toml#L258-L274)):

| baseline | space | search | trials |
|---|---|---|---|
| popularity | `count` ∈ {events, users} | grid | 2 |
| time_popularity | `half_life_days` loguniform [0.25, 365] | random | 10 |
| replay | `order` ∈ {recency, frequency} | grid | 2 |
| markov | none | grid of one | 1 |

At least one dataset and one model must be defined ([protocol.py:498-501](src/seqrec_eval/protocol.py#L498-L501)).

### Step 0.3.5: the `Protocol` object and its fingerprints

**code:**
`Protocol` at [protocol.py:208-293](src/seqrec_eval/protocol.py#L208-L293), built at
[protocol.py:507-526](src/seqrec_eval/protocol.py#L507-L526).

**explanation:**
Besides the parsed values, `Protocol` has lookups that fail with a clear message for an unknown name
(`dataset()`, `model()`, `baseline()`, `ablation()`, [protocol.py:231-258](src/seqrec_eval/protocol.py#L231-L258))
and the fingerprints every result folder is named after:

- `_result_settings()` ([protocol.py:260-264](src/seqrec_eval/protocol.py#L260-L264)): the `[protocol]`
  keys that change results (`_RESULT_KEYS`, [protocol.py:39-42](src/seqrec_eval/protocol.py#L39-L42)), plus
  `trial_seed` = the **first** seed only, plus `SCORING_VERSION` ([protocol.py:51](src/seqrec_eval/protocol.py#L51)).
  `eval_batch_size` is left out (speed only), and so are the other seeds: the first seed seeds every search
  trial and so decides the selection, while each other seed only adds one final run in a folder of its own.
  That is what lets seeds be added later without starting anything over (Step 6.1).
  `SCORING_VERSION` is 4. It is bumped when the code changes what a scored target means, so older results are
  not reused: 2 = the real first moment, only recommendable next items (H16); 3 = under `exclude_seen` a next
  item already in the history cannot be recommended either, and every evaluation excludes seen items the same
  way (review C4, A6); 4 = one exclusion width per phase (review N30, Step 3.9.5).
- `dataset_fingerprint(d)` ([protocol.py:276-285](src/seqrec_eval/protocol.py#L276-L285)): the dataset
  section's **build** settings (without `exclude_seen` and `new_item_diagnostic`, `_SCORING_ONLY_KEYS`,
  [protocol.py:54](src/seqrec_eval/protocol.py#L54)), `max_val_users` and `search_seed`. It identifies a
  prepared split. Flipping a scoring-only key therefore neither rebuilds the split nor redoes its analysis
  folder; every fingerprint of a scored result carries those keys instead.
- `run_fingerprint(d, m)` ([protocol.py:287-293](src/seqrec_eval/protocol.py#L287-L293)): result settings +
  the whole dataset section + the model section. It identifies every trial and final of a model on a dataset.
- `baseline_fingerprint(d, b)` ([protocol.py:246-253](src/seqrec_eval/protocol.py#L246-L253)): the same with
  the baseline section in place of the model's, plus `BASELINE_VERSION` = 2
  ([protocol.py:59](src/seqrec_eval/protocol.py#L59); 2 = ties ranked in one fixed order, Step 3.8.1).
- `evaluation_key(d)` ([protocol.py:266-274](src/seqrec_eval/protocol.py#L266-L274)): `targets`, `refit`,
  the scoring version and the dataset's `exclude_seen`, i.e. what decides which users are scored and on what.
  It keys caches that hold users or profiles rather than runs (the analysis profile, an ablation's fixed test
  users and conditions).

## Step 0.4: choosing datasets, models and sweeps

**code:**
[cli.py:219-223](src/seqrec_eval/cli.py#L219-L223), `_select` at [cli.py:80-88](src/seqrec_eval/cli.py#L80-L88),
`check_ablation` at [ablations.py:201-228](src/seqrec_eval/ablations.py#L201-L228).

**explanation:**
`_select` turns `--dataset`, `--model` and `--sweep` into lists: nothing given, or `all`, means every name
in the protocol; `--sweep none` means no sweep at all ([cli.py:83-84](src/seqrec_eval/cli.py#L83-L84), used
as `analyse --sweep none` for the full data only); an unknown name stops the program with the list of valid
names. A command that has no such option (for example `prepare` has no `--model`) gets all of them.

Every selected sweep is then checked once ([cli.py:222-223](src/seqrec_eval/cli.py#L222-L223)), so a bad
level fails now rather than halfway through a sweep. `check_ablation` looks the transform up in the
registry (`transform_of`, [ablations.py:177-181](src/seqrec_eval/ablations.py#L177-L181)), refuses a level
labelled `full` (reserved for the reference), runs the transform's own `validate` on the levels and options,
and checks that the `scope` is one the transform supports. An `expected_to_move` given in the protocol must
list characteristic names, each alone or with the part it applies to (`"test catalogue"`, `_is_declaration`,
[ablations.py:188-192](src/seqrec_eval/ablations.py#L188-L192)), and must not name the target itself
([ablations.py:211-218](src/seqrec_eval/ablations.py#L211-L218)). Last, `manipulation_tolerance` (≥ 0),
`knee_margin` (in (0, 1)) and `min_level_users` (positive integer) are checked.

---

# Part 1: `seqrec-eval plan`

## Step 1.1

**code:**
[cli.py:225-258](src/seqrec_eval/cli.py#L225-L258).

**explanation:**
`plan` only reads the protocol and the work folder; it writes nothing. It prints:

1. the protocol path, version, primary metric, seeds and cutoffs ([cli.py:227-228](src/seqrec_eval/cli.py#L227-L228));
2. per dataset, whether `splits/<dataset>/split_info.json` exists, the dataset fingerprint, and the dataset's
   final seeds if seeds were added (`final_seeds`, Step 6.1) ([cli.py:229-234](src/seqrec_eval/cli.py#L229-L234));
3. per model on that dataset: family, number of trials, number of finals (= the dataset's seeds), run
   fingerprint, and whether the model is registered in `models.py` (`_registered`,
   [cli.py:499-504](src/seqrec_eval/cli.py#L499-L504)) ([cli.py:235-241](src/seqrec_eval/cli.py#L235-L241));
4. the total number of runs (trials + finals over all datasets and models, [cli.py:242](src/seqrec_eval/cli.py#L242));
5. per baseline: kind, number of validation trials, grid or random search ([cli.py:243-245](src/seqrec_eval/cli.py#L243-L245));
6. per sweep: transform, levels, scope and seeds, how many fits (scope `all`) or rescorings (scope
   `inference`) plus reference rescorings it needs per dataset and model, and on which datasets and models
   ([cli.py:246-252](src/seqrec_eval/cli.py#L246-L252)); and, for a dataset where seeds were added to the
   sweep, the counts under those seeds ([cli.py:253-257](src/seqrec_eval/cli.py#L253-L257)).

---

# Part 2: `seqrec-eval prepare --data-dir DIR`

`prepare` builds each dataset's temporal split once, extracts it to `work/splits/<dataset>/`, records what
it is, recovers the time of every event, derives the next-item targets, and (with `refit = true`) builds
the train + validation training set. Every later command loads this folder instead of building anything.

## Step 2.1: the CLI branch

**code:**
[cli.py:260-274](src/seqrec_eval/cli.py#L260-L274).

**explanation:**
`--data-dir` (or `$COMPRESSO_DATA_DIR`) is required: it is the library's data folder with the raw
downloads and the library's own caches ([cli.py:261-262](src/seqrec_eval/cli.py#L261-L262)). For each
selected dataset, `prepare_split` is called ([cli.py:265-267](src/seqrec_eval/cli.py#L265-L267)) with
`force` (rebuild a split prepared from different build settings), `show_progress` (off with `--quiet`)
and `timestamps` (off with `--no-timestamps`). Afterwards `split_info.json` is read back and the sizes are
logged: training users, validation users and how many of them the search scores, test users
([cli.py:268-273](src/seqrec_eval/cli.py#L268-L273)).

## Step 2.2: one process per dataset, and is there already a split?

**code:**
`prepare_split` at [splits.py:205-221](src/seqrec_eval/splits.py#L205-L221) and `_prepare_split` at
[splits.py:224-303](src/seqrec_eval/splits.py#L224-L303); the reuse check at
[splits.py:226-253](src/seqrec_eval/splits.py#L226-L253).

**explanation:**
`prepare_split` first claims `work/splits/.<dataset>.prepare/.lock` with the runner's `RunLock` (Step 5.3,
[splits.py:214-216](src/seqrec_eval/splits.py#L214-L216)). Two processes preparing the same dataset would
share the staging folder and overwrite each other's files, so the second one stops with an error (review
B11). The lock is released whatever happens ([splits.py:220-221](src/seqrec_eval/splits.py#L220-L221)).

The output folder is `work/splits/<dataset>` (`split_dir`, [splits.py:88-89](src/seqrec_eval/splits.py#L88-L89)).
If its `split_info.json` exists and `--force` is not given, the recorded `dataset_fingerprint` is compared
with the protocol's:

- **Same fingerprint:** the split is reused. Two things can still be missing and are added without
  rebuilding the split ([splits.py:232-249](src/seqrec_eval/splits.py#L232-L249)):
  - the timestamps, or next-item targets of an older definition (`next_targets_version` in the info is not
    `NEXT_TARGETS_VERSION` = 2, [timestamps.py:78](src/seqrec_eval/timestamps.py#L78));
  - the refit set, when `refit = true` and the info has no `refit` entry.

  If either is missing, the split is loaded, the prepared events are rebuilt once
  ([splits.py:238-240](src/seqrec_eval/splits.py#L238-L240)), the missing part is attached (Steps 2.8
  and 2.9), and `split_info.json` is rewritten. Then the function returns.
- **Different fingerprint:** it refuses ([splits.py:250-253](src/seqrec_eval/splits.py#L250-L253)).
  Rebuilding would change the split that every existing run on this dataset was scored against, so it
  takes `--force`. Since the fingerprint covers only the build settings (Step 0.3.5), changing
  `exclude_seen` or `new_item_diagnostic` does not trigger this.

## Step 2.3: the build parameters and the library check

**code:**
[splits.py:255-261](src/seqrec_eval/splits.py#L255-L261); `_check_library` at
[splits.py:194-202](src/seqrec_eval/splits.py#L194-L202).

**explanation:**
`build_parameters()` (Step 0.3.2) gives the builder's keyword arguments. Before the slow build,
`_check_library` compares them with the signature of the installed `build_recsys_checkpoint`. If a key is
not accepted (in practice `temporal_train_users`, which only the vendored build has), it stops with the
installed version in the message. Then any leftover `.<dataset>.building.zip` and `.<dataset>.staging/`
from an interrupted run are removed ([splits.py:258-261](src/seqrec_eval/splits.py#L258-L261)).

## Step 2.4: the library builds the temporal split

**code:**
[splits.py:263-266](src/seqrec_eval/splits.py#L263-L266) calls cr's `build_recsys_checkpoint`
([cr builder.py:1860](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1860)), which runs
`_build_recsys_checkpoint_from_args`
([cr builder.py:1686](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1686)) and, for
`split_mode = "temporal"`, `_build_temporal_split`
([cr builder.py:1495-1657](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1495-L1657)).

**explanation:**
This is library code; the suite only calls it and times it (`build_seconds`). What it does, in order:

1. **Resolve and seed.** Registry defaults are merged into the arguments and `random`/`numpy` are seeded
   with the resolved seed ([cr builder.py:1687-1689](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1687-L1689)).
2. **Load and preprocess.** The dataset adapter loads the raw interactions
   ([cr builder.py:1693-1694](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1693-L1694)).
   For a temporal split, preprocessing applies only the rating threshold and `set_all_values_to`; user
   and item support are set to 1 here, because the support filters run later, per stage
   ([cr builder.py:1703-1711](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1703-L1711)).
3. **Times and boundaries.** Timestamps are converted to unix seconds by magnitude (≥ 1e17 nanoseconds,
   ≥ 1e14 microseconds, ≥ 1e11 milliseconds; `_timestamps_in_seconds`,
   [cr builder.py:1339-1353](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1339-L1353)).
   With P = `temporal_period_hours` and T = the last timestamp, the three windows are
   `train_target_start = T − 3P`, `validation_target_start = T − 2P`, `test_target_start = T − P`
   ([cr builder.py:1518-1520](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1518-L1520)).
   Users and items are numbered by sorting their ids (`pd.factorize(sort=True)`,
   [cr builder.py:1504-1509](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1504-L1509)); this order
   is the row order of every stage.
4. **Three stages**, each built by `_build_temporal_stage`
   ([cr builder.py:1356-1474](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1356-L1474)):

   | stage | source (history) | target | items |
   |---|---|---|---|
   | train | t < train_target_start | train_target_start ≤ t < validation_target_start | all seen |
   | validation | t < validation_target_start | validation_target_start ≤ t < test_target_start | train items + new ones |
   | test | t < test_target_start | t ≥ test_target_start | validation items + new ones |

   Each stage's catalogue is the previous stage's items (inherited, kept) followed by the items first
   seen in this stage. The support filter `_filter_temporal_pair`
   ([cr builder.py:1170-1241](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1170-L1241))
   repeats until nothing changes: a user needs ≥ `min_source_items` distinct items in the source,
   ≥ `min_target_items` in the target (both default 1) and ≥ `min_user_support` in source ∪ target; a
   *new* item needs ≥ `item_min_support` users in the stage (inherited items are never dropped).
   With `temporal_train_users = "all"`, the train stage sets `min_source_items` and `min_target_items`
   to 0 (`_train_stage_args`,
   [cr builder.py:1477-1492](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1477-L1492), passed at
   [cr builder.py:1541](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1541)), so it keeps every
   user with `min_user_support` items before the validation window, not only users active in the train
   target window.
5. **Matrices and sequences.** Each stage's source and target are CSR matrices (users × stage catalogue)
   whose entries are the summed event values per (user, item), i.e. event counts, since every value is
   1.0. `x_train = train_source.maximum(train_target)`
   ([cr builder.py:1585](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1585)). Beside the
   matrices, the source histories are also saved in order as `ItemSequences` (Step 2.5).
6. **Write.** Everything is written into the zip by `save_recsys_split`
   ([cr builder.py:1733-1735](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1733-L1735)),
   and the window boundaries go into the manifest
   ([cr builder.py:1647-1649](vendor/compresso-recsys/src/compresso_recsys/builder.py#L1647-L1649)),
   which is how the suite finds them later (`_boundaries`,
   [timestamps.py:123-129](src/seqrec_eval/timestamps.py#L123-L129)).

## Step 2.5: extracting and loading the split

**code:**
[splits.py:267-272](src/seqrec_eval/splits.py#L267-L272); cr's `load_recsys_split`
([cr checkpoint.py:533-679](vendor/compresso-recsys/src/compresso_recsys/checkpoint.py#L533-L679))
and `load_manifest`
([cr checkpoint.py:135-138](vendor/compresso-recsys/src/compresso_recsys/checkpoint.py#L135-L138)).

**explanation:**
The zip is extracted once into the staging folder and deleted, so no run ever unpacks hundreds of
megabytes again. `load_recsys_split` reads the files in `data/` and returns a plain dict. This dict is
`split.data` everywhere in the suite. Its fields for a temporal split:

| key | type, shape | meaning |
|---|---|---|
| `train_item_ids`, `val_item_ids`, `test_item_ids` | string arrays | each phase's catalogue; train ⊂ val ⊂ test as prefixes |
| `item_ids` | string array | the largest catalogue (= `test_item_ids` for temporal) |
| `x_train` | CSR, train users × train items | the training window (source ∪ target of the train stage), `max(source counts, target counts)` per cell; what matrix models fit on |
| `train_source_matrix`, `train_target_matrix` | CSR, same shape | the train stage split at `train_target_start` |
| `val_source_matrix`, `val_target_matrix` | CSR, val users × val items | validation histories and the validation window's targets |
| `test_source_matrix`, `test_target_matrix` | CSR, test users × test items | test histories and the test window's targets ("window" targets) |
| `x_train_sequences` | `ItemSequences`, train users | the training window in time order; what sequence models and the baselines fit on |
| `train_source_sequences` | `ItemSequences` | the train stage's source part only |
| `val_source_sequences`, `test_source_sequences` | `ItemSequences` | the ordered histories of validation and test users |
| `train_user_ids`, `val_user_ids`, `test_user_ids` | string arrays | the user of each row |
| `val_eval_user_ids`, `test_eval_user_ids` | string arrays or `None` | the ids evaluation reports per row (equal to the phase's user ids here) |
| `warm_item_indices` | int array | `0 … len(train_item_ids)−1`: items a trained model knows |
| `val_cold_item_indices`, `test_cold_item_indices` | int arrays | the indices first seen in validation / test |
| `val_source_indices` … `test_target_indices` | per-row index lists | an older, list-shaped copy of the same histories and targets |
| `entity_tag_matrix`, `tag_names`, `entity_metadata` | or `None` | item annotations and metadata; not used by the suite |

An `ItemSequences` ([cr sequences.py:53](vendor/compresso-recsys/src/compresso_recsys/sequences.py#L53)) is
CSR without data: `values` holds item indices of every row concatenated, oldest first; `indptr[i]` to
`indptr[i+1]` is row i; `n_items` is the size of the item space. Duplicates are kept, so a repeat is a
second event. Its arrays are made read-only, so no code can reorder a history after the fact.
`row_lengths` is the length of each history; `take_rows` and `select_rows` cut rows out of it.

`load_manifest` returns the manifest dict, including `stages.data` with the three window boundaries.

## Step 2.6: the fixed validation sample

**code:**
[splits.py:273-278](src/seqrec_eval/splits.py#L273-L278); `_stream` at
[search.py:29-31](src/seqrec_eval/search.py#L29-L31).

**explanation:**
If there are more validation users than `max_val_users` (20,000), a sorted random sample of that many rows
is drawn once and saved as `val_rows.npy`. Every search trial of every model, and every baseline trial, is
scored on exactly these rows, so trials stay paired and cheap. The random generator comes from
`_stream(search_seed, dataset, "val_rows")`: the parts are joined, SHA-256 hashed, and the first 8 bytes
seed a numpy generator. This is the pattern for every random stream in the suite: a stream is a pure
function of its name, so it never depends on what ran before it.

## Step 2.7: `split_info.json`

**code:**
[splits.py:280-290](src/seqrec_eval/splits.py#L280-L290), with `_resolved_parameters`
([splits.py:92-98](src/seqrec_eval/splits.py#L92-L98)), `library_provenance`
([splits.py:166-191](src/seqrec_eval/splits.py#L166-L191)) and `_stage_stats`
([splits.py:101-127](src/seqrec_eval/splits.py#L101-L127)).

**explanation:**
The record of what the split actually is (research design §6.2):

| field | content |
|---|---|
| `dataset`, `dataset_fingerprint` | which section built it; the reuse check of Step 2.2 and every `load_split` (Step 3.2) read the fingerprint |
| `build_parameters` | exactly what was passed to the builder |
| `resolved_build_parameters` | what the builder used after merging its registry defaults, obtained by calling its private `_resolve_args`; if that fails, the error is recorded rather than failing the build |
| `library` | the library's version, a hash of its imported source (`_source_hash`, [splits.py:130-137](src/seqrec_eval/splits.py#L130-L137)), its path, and for an install with `direct_url.json` the source URL, commit and whether it is editable. Imported from a source checkout, the version is the one the checkout's `pyproject.toml` declares (`_checkout_version`, [splits.py:157-163](src/seqrec_eval/splits.py#L157-L163)), with the installed metadata's version kept beside it as `metadata_version` when they differ: a stale `egg-info` once reported 0.3.6 for the 0.3.7 branch |
| `build_seconds` | build time |
| `val_rows_sampled` | size of the validation sample, or `null` |
| `stages` | per phase: rows, catalogue size, target pairs, how many targets repeat an item in the history and that fraction, and history length mean/p50/p90/max; plus the number of training events |
| `manifest` | the library's manifest, with the window boundaries |
| `timestamps` | added in Step 2.8 |
| `refit` | added in Step 2.9 |

## Step 2.8: recovering event times and the next-item targets

**code:**
[splits.py:291-294](src/seqrec_eval/splits.py#L291-L294) → `prepared_events`
([timestamps.py:85-120](src/seqrec_eval/timestamps.py#L85-L120)), `attach_timestamps`
([timestamps.py:223-247](src/seqrec_eval/timestamps.py#L223-L247)), which calls `align`
([timestamps.py:132-170](src/seqrec_eval/timestamps.py#L132-L170)) and `next_targets`
([timestamps.py:173-220](src/seqrec_eval/timestamps.py#L173-L220)).

**explanation:**
The library's saved split keeps each history's order but not its times. The time-decayed popularity
baseline, the tie analysis and the next-item targets need them, so they are recovered here, without
editing the library. It is a join, not a second split, and it is proved before anything is saved.

#### Step 2.8.1: the prepared events

`prepared_events` repeats the builder's steps up to the split: resolve the arguments, seed as the builder
seeds, load the adapter's interactions, apply the same preprocessing (rating threshold, value; support 1),
and convert the times to seconds with the builder's own function. It returns a DataFrame with one row per
event, in source order: `user_id`, `item_id` (both strings), `timestamp` (unix seconds), `user_code` (the
builder's row order: ids factorized in their original type, sorted) and `value`. It is computed once per
`prepare`, when timestamps or the refit need it, and shared by both ([splits.py:291](src/seqrec_eval/splits.py#L291));
it is dropped as soon as they are done ([splits.py:297](src/seqrec_eval/splits.py#L297)).

#### Step 2.8.2: `align`, the times of every history

For each of the four history views in `VIEWS` ([timestamps.py:66-71](src/seqrec_eval/timestamps.py#L66-L71)):

| view | stage (rows and columns) | events before |
|---|---|---|
| `x_train_sequences` | train | `validation_target_start` |
| `train_source_sequences` | train | `train_target_start` |
| `val_source_sequences` | val | `validation_target_start` |
| `test_source_sequences` | test | `test_target_start` |

every event whose user is a row of that stage and whose item is in that stage's catalogue, and whose time
is before the boundary, is kept and sorted by (row, time) with a stable sort, so ties keep the source
order as in the builder ([timestamps.py:144-149](src/seqrec_eval/timestamps.py#L144-L149)). The result must
equal the split's own history: the same number of events per row
([timestamps.py:151-159](src/seqrec_eval/timestamps.py#L151-L159)) and the same item at every position
([timestamps.py:160-168](src/seqrec_eval/timestamps.py#L160-L168)). Any difference raises
`TimestampAlignmentError` naming the first user and position that disagree. Only then are the times kept.
If the library ever builds histories another way, this refuses rather than attaching wrong times.

#### Step 2.8.3: `next_targets`, what the user did first

For validation and test (`TARGET_WINDOWS`, [timestamps.py:74](src/seqrec_eval/timestamps.py#L74)):

1. All events of the phase's users inside the phase's target window are taken, **on any item**, including
   items the stage's `item_min_support` filter then deleted ([timestamps.py:185-193](src/seqrec_eval/timestamps.py#L185-L193)).
2. The surviving (user, item) pairs must be exactly the saved target matrix's pairs, or it raises
   ([timestamps.py:195-207](src/seqrec_eval/timestamps.py#L195-L207)).
3. Per user, the first moment is the minimum time over **all** their window events
   ([timestamps.py:211-212](src/seqrec_eval/timestamps.py#L211-L212)). The next-item target row holds the
   surviving items at exactly that moment (several items when events tie), as a binary matrix
   ([timestamps.py:213-219](src/seqrec_eval/timestamps.py#L213-L219)).

If the real first item was deleted, the row keeps only the surviving items of that moment and is empty
when none survived. It is **not** replaced by a later event, because that would silently make the task
"next-but-one" (review H16). Empty rows are left out of scoring later (Step 3.9.2).

#### Step 2.8.4: saving

`attach_timestamps` saves `x_train_timestamps.npy`, `train_source_timestamps.npy`,
`val_source_timestamps.npy`, `test_source_timestamps.npy` (float64 seconds, aligned with each view's
`values`) and `val_next_target_matrix.npz`, `test_next_target_matrix.npz`
([timestamps.py:234-237](src/seqrec_eval/timestamps.py#L234-L237)). It returns the `timestamps` entry of
`split_info.json`: unit, which views were aligned and their event counts, `next_targets_version`, and per
phase the users with a next target, the users whose next item was deleted, and the mean number of items per
next target ([timestamps.py:238-247](src/seqrec_eval/timestamps.py#L238-L247)). `load_timestamps`
([timestamps.py:250-261](src/seqrec_eval/timestamps.py#L250-L261)) reads whichever of these files exist and
is merged into `data` right away ([splits.py:294](src/seqrec_eval/splits.py#L294)), because the refit step
reads `test_next_target_matrix`.

## Step 2.9: the refit set (train + validation)

**code:**
[splits.py:295-296](src/seqrec_eval/splits.py#L295-L296) → `attach_refit`
([refit.py:184-219](src/seqrec_eval/refit.py#L184-L219)), which uses `train_rule`
([refit.py:68-72](src/seqrec_eval/refit.py#L68-L72)), `prove` ([refit.py:160-181](src/seqrec_eval/refit.py#L160-L181))
and `training_set` ([refit.py:88-142](src/seqrec_eval/refit.py#L88-L142)).

**explanation:**
With `refit = true`, every model scored on test is first trained on everything before the test window.
Without it, a test history runs a whole window past the end of training, so validation and test are
different tasks, and items first seen in validation cannot be recommended at all
([refit.py:1-15](src/seqrec_eval/refit.py#L1-L15)). The library's split has no such training set, so
it is rebuilt here by the library's own train-stage rule, one window later, and that rule is proved first.

1. **The rule.** `train_rule` asks the builder for the train stage's user filter:
   `(min_user_support, min_source_items, min_target_items)`, which with `"all"` is `(5, 0, 0)`.
2. **`training_set`** builds a train stage over a fixed catalogue from the prepared events: events before
   `window_end` on catalogue items; users kept if they have enough distinct items overall, before
   `source_end`, and from it on ([refit.py:102-116](src/seqrec_eval/refit.py#L102-L116)); rows in the
   builder's user order; histories sorted by (row, time), stable ([refit.py:120-124](src/seqrec_eval/refit.py#L120-L124));
   source and target matrices summed per (user, item) like the builder's (`_matrix`,
   [refit.py:79-85](src/seqrec_eval/refit.py#L79-L85)), and `matrix = source.maximum(target)` like `x_train`.
   Because the catalogue is fixed, the builder's repeat-until-stable filter reduces to one pass.
3. **`prove`.** The same function, given the *train* stage's catalogue and boundaries, must reproduce the
   split's `train_user_ids`, `x_train_sequences`, `train_source_sequences` and `x_train` exactly, or it
   raises `RefitAlignmentError` with the first difference ([refit.py:160-181](src/seqrec_eval/refit.py#L160-L181)).
4. **The refit set.** Then it is built with the *validation* catalogue (`val_item_ids`), source before
   `validation_target_start` and window up to `test_target_start`
   ([refit.py:191-194](src/seqrec_eval/refit.py#L191-L194)). Using the validation catalogue means a refitted
   model's rankings map into the test catalogue exactly as a trained one's do.
5. **Saved** ([refit.py:195-202](src/seqrec_eval/refit.py#L195-L202)): `x_refit.npz`,
   `refit_source_matrix.npz`, `refit_target_matrix.npz`, `x_refit_sequences.npz`,
   `refit_source_sequences.npz`, `x_refit_timestamps.npy`, `refit_source_timestamps.npy`,
   `refit_user_ids.npy`.
6. **The summary** in `split_info.json` ([refit.py:204-219](src/seqrec_eval/refit.py#L204-L219)): what it
   was proved against, the rule, users/events/items of the refit set beside the train set's, and how many
   test users have a next target a model can recommend before the refit (train catalogue) and after it
   (validation catalogue).

## Step 2.10: making it final

**code:**
[splits.py:297-303](src/seqrec_eval/splits.py#L297-L303); `write_json` at
[results.py:54-58](src/seqrec_eval/results.py#L54-L58) with `durable_replace`
([results.py:41-51](src/seqrec_eval/results.py#L41-L51)).

**explanation:**
`split_info.json` is written into the staging folder, the old split folder (if any) is removed, and the
staging folder is renamed to `work/splits/<dataset>`. Until that rename, nothing a run would load has
changed. `write_json` writes to a temporary name (`.<name>.<pid>.tmp`, [results.py:27-28](src/seqrec_eval/results.py#L27-L28))
and `durable_replace` moves it into place: the data is flushed to disk before the rename, and the folder
after it, so a power loss leaves the old file or the new one, never an empty one (review B5). Every JSON
file and every saved evaluation in the suite is written this way. `read_json`
([results.py:61-66](src/seqrec_eval/results.py#L61-L66)) turns a file that is not valid JSON into an error
that names the file and says to move it aside and rerun the step that writes it.

**Output:** `work/splits/<dataset>/` holds `manifest.json` and `data/` (the library's split), `split_info.json`,
`val_rows.npy` (if sampled), the four `*_timestamps.npy`, the two `*_next_target_matrix.npz` and the refit
files.

---

# Part 3: `seqrec-eval analyse`

The analysis comes before any model and runs on CPU. For each dataset it computes: (1) the **profile**
of the data, (2) the four non-learned **baselines**, each searched on validation and scored on test, the
strongest being the **floor** every model has to beat, and (3) the **sequence signal**: Markov against
itself with order removed. Then it repeats (1)–(3) at every condition of every ablation sweep on that
dataset. It uses the same split, users, metrics and evaluator as the models, so its numbers can be set
beside a model's ([analysis.py:1-38](src/seqrec_eval/analysis.py#L1-L38)).

## Step 3.1: the CLI branch

**code:**
[cli.py:332-345](src/seqrec_eval/cli.py#L332-L345).

**explanation:**
`--threads` sets torch's CPU threads for the process ([cli.py:333-334](src/seqrec_eval/cli.py#L333-L334)).
For each selected dataset: load the prepared split, checked against the protocol
([cli.py:336](src/seqrec_eval/cli.py#L336)), run the full-data analysis
([cli.py:338](src/seqrec_eval/cli.py#L338)), then for each selected sweep that covers this dataset run the
per-condition analysis ([cli.py:339-342](src/seqrec_eval/cli.py#L339-L342)), and drop the split before the
next dataset ([cli.py:343](src/seqrec_eval/cli.py#L343)) to free memory. `analyse --sweep none` selects no
sweep, so only the full data is analysed (Step 0.4).

## Step 3.2: `load_split`

**code:**
`load_split(work_dir, dataset, protocol)` at [splits.py:306-334](src/seqrec_eval/splits.py#L306-L334),
returning a `Split` ([splits.py:59-85](src/seqrec_eval/splits.py#L59-L85)).

**explanation:**
`load_split` reads what `prepare` left in `work/splits/<dataset>/` and wraps it in a `Split`, the object
every step from here on receives. Given the protocol, as every command that runs or scores gives it, it
first refuses a split prepared from other build settings than the protocol now has
([splits.py:317-323](src/seqrec_eval/splits.py#L317-L323)): the dataset fingerprint in `split_info.json`
must equal the protocol's. Without that check a changed `[datasets.*]` section would be scored on the old
split while the runs were filed under the new fingerprint (final review A4). The fix is
`prepare --dataset D --force`, or restoring the settings. The `Split`'s fields:

| field | content |
|---|---|
| `dataset` | the dataset name |
| `path` | `work/splits/<dataset>` |
| `data` | the dict of Step 2.5, plus the suite's own arrays (Step 3.3) |
| `info` | the parsed `split_info.json` (Step 2.7) |
| `val_rows` | the fixed validation sample (`val_rows.npy`), or `None` to score every validation row |
| `test_rows` | which test rows to score; `None` (all) on the full data, set on an ablation condition |
| `condition` | `None` for the original split; `{sweep, level, label, data_seed}` on an ablation condition |
| `trained_on` | `"train"` (the training views hold the training window) or `"train+val"` (after `final_split`, Step 3.4) |

`eval_user_ids(phase)` ([splits.py:77-85](src/seqrec_eval/splits.py#L77-L85)) returns the user id of each
row of a phase (the `*_eval_user_ids` if the split has them, else `*_user_ids`) and checks that there is one
per target row. These are the `sample_ids` every evaluation is keyed on, which is what makes two results
pairable user by user.

## Step 3.3: what is read from disk

**code:**
[splits.py:313-334](src/seqrec_eval/splits.py#L313-L334); `load_timestamps` at
[timestamps.py:250-261](src/seqrec_eval/timestamps.py#L250-L261); `load_refit` at
[refit.py:222-231](src/seqrec_eval/refit.py#L222-L231).

**explanation:**
If `split_info.json` is missing, it stops and says to run `prepare` ([splits.py:315-316](src/seqrec_eval/splits.py#L315-L316)).
Otherwise:

1. `load_recsys_split(path)` ([splits.py:325](src/seqrec_eval/splits.py#L325)) reads the library's split
   from `data/` (the dict of Step 2.5).
2. `data.update(load_timestamps(path))` ([splits.py:326](src/seqrec_eval/splits.py#L326)) adds, if they
   exist: `x_train_timestamps`, `train_source_timestamps`, `val_source_timestamps`,
   `test_source_timestamps` (one time per event, aligned with each view's `values`), and
   `val_next_target_matrix`, `test_next_target_matrix` (the next-item targets). Without these, the
   time-decayed baseline cannot fit and `targets = "next"` cannot score.
3. `data.update(load_refit(path))` ([splits.py:327](src/seqrec_eval/splits.py#L327)) adds, if
   `refit_user_ids.npy` exists: `x_refit`, `refit_source_matrix`, `refit_target_matrix`,
   `x_refit_sequences`, `refit_source_sequences`, `x_refit_timestamps`, `refit_source_timestamps`,
   `refit_user_ids`. They sit beside the training views and are used only once `final_split` swaps them in.
4. `val_rows.npy` is loaded if it exists ([splits.py:333](src/seqrec_eval/splits.py#L333)).

## Step 3.4: `analyse_full`: paths, the tested split, the profile

**code:**
`analyse_full(protocol, work_dir, split)` at [analysis.py:251-284](src/seqrec_eval/analysis.py#L251-L284);
the setup at [analysis.py:258-265](src/seqrec_eval/analysis.py#L258-L265). `final_split` at
[splits.py:337-349](src/seqrec_eval/splits.py#L337-L349), `swap_training` at
[refit.py:234-254](src/seqrec_eval/refit.py#L234-L254).

**explanation:**
Two splits are used side by side, because searching and testing need different training data:

- `split` (the one loaded) is trained on the training window and is used to **search** each baseline on
  validation.
- `tested = final_split(protocol, split)` ([analysis.py:259](src/seqrec_eval/analysis.py#L259)) is what
  everything scored on **test** is fitted on, exactly as for the models' final runs.

**`final_split`** returns the split unchanged if `refit` is off or it is already `train+val`
([splits.py:344-345](src/seqrec_eval/splits.py#L344-L345)). It refuses an ablation condition, since a
condition must be made from the refitted split rather than the other way round
([splits.py:346-347](src/seqrec_eval/splits.py#L346-L347)). Otherwise it returns a new `Split` whose data
went through `swap_training` and whose `trained_on` is `"train+val"`. **`swap_training`** replaces every
training view with its refit counterpart: `x_train ← x_refit`, `train_source_matrix ← refit_source_matrix`,
`train_target_matrix ← refit_target_matrix`, `x_train_sequences ← x_refit_sequences`,
`train_source_sequences ← refit_source_sequences`, `train_user_ids ← refit_user_ids`, and both timestamp
arrays. `train_item_ids` becomes `val_item_ids`, so the model's catalogue is the validation catalogue; every
item is now warm (`warm_item_indices` = all, `val_cold_item_indices` empty). It fails if the split was
prepared without the refit set ([refit.py:236-239](src/seqrec_eval/refit.py#L236-L239)). Nothing is copied:
the new dict points at the arrays already loaded.

**Paths** (Appendix A has the full tree):

- `root = analysis_root(...)` ([analysis.py:258](src/seqrec_eval/analysis.py#L258), defined at
  [analysis.py:132-133](src/seqrec_eval/analysis.py#L132-L133)) is `work/analysis/<dataset>/<dataset
  fingerprint[:12]>/`. Everything of this dataset's full-data analysis lives under it, so a re-prepared
  split (new fingerprint) starts a fresh analysis.
- `profile_path(...)` ([analysis.py:260](src/seqrec_eval/analysis.py#L260), defined at
  [analysis.py:245-248](src/seqrec_eval/analysis.py#L245-L248)) is `root/profile-<evaluation key[:12]>.json`.
  The profile depends on which users are scored and on what (targets, refit, scoring version,
  `exclude_seen`), so it is keyed by the evaluation key too.

**The profile** ([analysis.py:261-263](src/seqrec_eval/analysis.py#L261-L263)) is written only if missing:
`{"dataset", "trained_on", "characteristics"}`, where
`characteristics(tested, None, target_key("test", protocol.targets))` describes the data the tested models
train on and the test inputs they are scored from. Step 3.5 explains every field. It is the same function
the ablation manipulation check uses, so what is described and what is checked cannot drift apart.

**The diagnostic flag** ([analysis.py:264-265](src/seqrec_eval/analysis.py#L264-L265)): `other` is the
target definition the protocol did *not* choose (`other_definition`,
[evaluate.py:283-284](src/seqrec_eval/evaluate.py#L283-L284)), so `"window"` under `targets = "next"`.
`diagnose` is true when the tested split has that definition's test targets
(`target_key("test", "window")` = `test_target_matrix`, [evaluate.py:277-280](src/seqrec_eval/evaluate.py#L277-L280)),
which is always the case for `"window"`. For `targets = "window"` it would need the next-item targets, which
a split prepared with `--no-timestamps` lacks. When `diagnose` is true, every baseline and control is also
scored against the other definition's targets, as a diagnostic that never enters selection or the floor.

## Step 3.5: the profile, field by field

**code:**
`characteristics` at [ablations.py:1043-1081](src/seqrec_eval/ablations.py#L1043-L1081), with `_describe`
([ablations.py:928-946](src/seqrec_eval/ablations.py#L928-L946)), `gini`
([ablations.py:912-917](src/seqrec_eval/ablations.py#L912-L917)), `_order_stats`
([ablations.py:986-995](src/seqrec_eval/ablations.py#L986-L995)), `_time_stats`
([ablations.py:998-1028](src/seqrec_eval/ablations.py#L998-L1028)) and `_edits`
([ablations.py:949-983](src/seqrec_eval/ablations.py#L949-L983)).

**explanation:**
Two parts are described ([ablations.py:1062-1063](src/seqrec_eval/ablations.py#L1062-L1063)):
`train` = `x_train_sequences` (all training rows; under refit the train+validation histories) and `test` =
`test_source_sequences` restricted to `split.test_rows` (all test rows on the full data). For each part:

| field | computed as | from |
|---|---|---|
| `rows`, `events` | rows described; events in them | `_describe` |
| `history_length`, `history_length_p50`, `history_length_p90` | mean, median and 90th percentile of events per row | `_describe`, [ablations.py:1069-1070](src/seqrec_eval/ablations.py#L1069-L1070) |
| `catalogue` | items with at least one event | `_describe` |
| `density` | distinct (row, item) pairs / (rows × catalogue) | `_describe` |
| `popularity_gini` | Gini of the event counts of those items: 0 = every item equally popular, → 1 = a few items take everything | `gini` |
| `repeat_rate` | 1 − distinct pairs / events: the share of events that repeat an item already in the same history | `_describe` |
| `users_with_any_repeat` | share of rows with at least one repeat | `_order_stats` |
| `self_transition_share` | share of adjacent pairs where the same item follows itself (a → a) | `_order_stats` |
| `tie_rate` | share of adjacent pairs with the same timestamp, whose order the source file decided | `_time_stats` |
| `last_tied` | share of histories whose last two events tie, so which item is "last" is arbitrary | `_time_stats` |
| `median_gap_seconds`, `gap_under_30_min`, `gap_over_1_day` | the gaps between adjacent events | `_time_stats` |
| `median_span_days` | median time from a history's first to last event | `_time_stats` |
| `rows_changed`, `span_kept` | only when an `original` split is given (ablation conditions, Step 10.7) | `_edits` |

The time fields are `None` without timestamps. The test part also gets `new_item_target_share`
([ablations.py:1076-1080](src/seqrec_eval/ablations.py#L1076-L1080)): the share of target entries (of the
protocol's definition) whose item is outside the training catalogue, i.e. that no model can recommend. The
first five (`history_length`, `catalogue`, `density`, `popularity_gini`, `repeat_rate`) are the five
`CHARACTERISTICS` ([ablations.py:94](src/seqrec_eval/ablations.py#L94)) the manipulation check compares.

## Step 3.6: the baseline loop

**code:**
[analysis.py:266-276](src/seqrec_eval/analysis.py#L266-L276).

**explanation:**
For every `[baselines.<name>]` in the protocol, in file order (popularity, time_popularity, replay, markov):

1. **Select** ([analysis.py:267](src/seqrec_eval/analysis.py#L267)): `select_baseline` searches the
   baseline on validation with `split` and returns the selected record (Step 3.7).
2. **Where the test result goes** ([analysis.py:268](src/seqrec_eval/analysis.py#L268)): the stem
   `<baseline dir>/test`, where the baseline dir (`_baseline_dir`, [analysis.py:136-138](src/seqrec_eval/analysis.py#L136-L138))
   is `root/baselines/<name>/<baseline fingerprint[:12]>/`. Editing the baseline's section, or a code change
   that bumps `BASELINE_VERSION`, changes the fingerprint and starts a new folder with a new search.
3. **How to fit it** ([analysis.py:269](src/seqrec_eval/analysis.py#L269)): `fit` is a function, not a
   fitted model: `fit_baseline(kind, selected params, tested)`, the selected setting fitted on the tested
   (refit) split. It is only called if a result is missing, so a rerun with every result cached fits nothing.
4. **What is recorded beside it** ([analysis.py:270](src/seqrec_eval/analysis.py#L270)): the baseline
   name, the params and `trained_on`.
5. **Score and cache on test** ([analysis.py:271](src/seqrec_eval/analysis.py#L271)): `_cached(stem,
   compute, record)` (Step 3.10) returns the saved result if `test.json` and `test.npz` exist; otherwise it
   fits, scores on test with the protocol's targets (`_score`, Step 3.9), saves the result and writes
   `test.done.json`.
6. **Markov's tie mask** ([analysis.py:272-273](src/seqrec_eval/analysis.py#L272-L273)): for the Markov
   baseline, `_save_tied` saves which scored users' histories end in a tie (Step 3.10).
7. **The diagnostic** ([analysis.py:274-276](src/seqrec_eval/analysis.py#L274-L276)): with `diagnose`,
   the same fitted setting is scored against the other definition's targets and cached as `test_window.*`,
   its done record marked `"targets": "window"`.

**Output per baseline:** `trial-NNN.json` (one per validation trial), `selected.json`, `test.json` +
`test.npz` + `test.done.json`, `test_window.json` + `.npz` + `.done.json`, and for Markov `tied.npy`.

## Step 3.7: `select_baseline`, the validation search

**code:**
`select_baseline` at [analysis.py:151-180](src/seqrec_eval/analysis.py#L151-L180); `baseline_params` at
[analysis.py:141-144](src/seqrec_eval/analysis.py#L141-L144); `grid_params` and `trial_params` at
[search.py:50-66](src/seqrec_eval/search.py#L50-L66).

**explanation:**
A baseline is searched exactly as a model is, so the floor is tuned as carefully as the models it bounds.

1. If `selected.json` exists, it is returned at once ([analysis.py:155-157](src/seqrec_eval/analysis.py#L155-L157)).
   The search is done once per baseline fingerprint.
2. Otherwise, for each trial `0 … trials−1` ([analysis.py:160-170](src/seqrec_eval/analysis.py#L160-L170)):
   - a finished trial's `trial-NNN.json` is read back, so an interrupted search resumes;
   - else its parameters come from `baseline_params`: for a grid, `grid_params` enumerates the choices with
     parameters sorted by name (`itertools.product`, [search.py:50-56](src/seqrec_eval/search.py#L50-L56)),
     so trial *i* is always the same combination; for random search, `trial_params` draws each parameter
     from its own stream (Step 5.2);
   - the baseline is fitted on `split` (the training window) and scored on **validation**, on the fixed
     `val_rows` sample (`_score(..., "val")`);
   - `{trial, params, val metrics, seconds}` is written to `trial-NNN.json`.
3. A trial whose validation primary metric is not a finite number stops the search with an error
   ([analysis.py:171-173](src/seqrec_eval/analysis.py#L171-L173)). A NaN can never win a `>` comparison,
   so it would otherwise hide a bug.
4. The best is the highest validation `ndcg@10`; a strict `>` means ties go to the lowest trial
   ([analysis.py:174-175](src/seqrec_eval/analysis.py#L174-L175)).
5. `selected.json` = `{baseline, kind, trial, params, val, seconds}` is written and logged
   ([analysis.py:176-180](src/seqrec_eval/analysis.py#L176-L180)).

### Step 3.7.1: fitting a baseline: `KINDS`, `_build`, `fit_baseline`

**code:**
`KINDS` at [analysis.py:94-99](src/seqrec_eval/analysis.py#L94-L99), `_build` at
[analysis.py:102-108](src/seqrec_eval/analysis.py#L102-L108), `fit_baseline` at
[analysis.py:111-123](src/seqrec_eval/analysis.py#L111-L123).

**explanation:**
`KINDS` maps each baseline kind to three things: its class, its config dataclass, and whether its `fit`
needs the training timestamps.

| kind | class | config | needs times |
|---|---|---|---|
| `popularity` | `Popularity` | `PopularityConfig` | no |
| `time_popularity` | `TimeDecayedPopularity` | `TimeDecayedPopularityConfig` | yes |
| `replay` | `Replay` | `ReplayConfig` | no |
| `markov` | `MarkovChain` | none | no |

`_build(kind, params)` constructs the unfitted baseline: `cls(config(**params))`, or `cls()` for a kind
without a config, which refuses any params ([analysis.py:104-107](src/seqrec_eval/analysis.py#L104-L107)).
An unknown parameter name fails in the config's constructor.

`fit_baseline(kind, params, split, sequences=None)` builds it, adds `timestamps = x_train_timestamps` for a
kind that needs them (with a clear error if the split has none, [analysis.py:115-120](src/seqrec_eval/analysis.py#L115-L120)),
and fits on `x_train_sequences`, or on `sequences` when given (the controls pass shuffled or reversed
histories), with `item_ids = train_item_ids` ([analysis.py:121-122](src/seqrec_eval/analysis.py#L121-L122)).
Every baseline therefore reads the ordered histories, never the matrix. Given `split`, it fits the training
window; given `tested`, the train+validation histories and the validation catalogue.

## Step 3.8: the four baselines

All four are in [baselines.py](src/seqrec_eval/baselines.py). Steps 3.8.1 to 3.8.5 cover what they share
and then each one: its code, what it predicts, and what it measures.

### Step 3.8.1: what they share

**code:**
`_SequenceBaseline` at [baselines.py:162-293](src/seqrec_eval/baselines.py#L162-L293); helpers at
[baselines.py:53-159](src/seqrec_eval/baselines.py#L53-L159).

**explanation:**
They subclass the library's `BaseSequentialRecommender`, so they take the same inputs and return the same
`SRPTensor` (ranked columns and values) as the library's sequential models.

- **Fitting** (`_start_fit`, [baselines.py:180-188](src/seqrec_eval/baselines.py#L180-L188)): check the input
  is `ItemSequences`, register the item vocabulary, set `n_items_` (the fitted catalogue size), and count
  `popularity_` = training events per item. Every baseline has this popularity for its lower tier and its
  tie order.
- **New items in a history.** Histories arrive whole through the `WarmCatalogAdapter` (Step 3.9.3), so a
  history can contain an index ≥ `n_items_`: an item first seen after training. `_known`
  ([baselines.py:57-59](src/seqrec_eval/baselines.py#L57-L59)) marks those, and they are ignored except that
  they keep their position.
- **Two tiers** (`_two_tiers`, [baselines.py:155-159](src/seqrec_eval/baselines.py#L155-L159)). Every item with
  a positive score is rescaled into `(1, 2]` per row; every other item gets its training popularity squeezed
  into `[0, 1)` (`_tiebreak`, [baselines.py:72-75](src/seqrec_eval/baselines.py#L72-L75)). So items the
  baseline has evidence for always come first in its own order, and the rest of the list is filled by
  popularity rather than left arbitrary. The tiers are strict: a tiny decayed weight or a rare transition
  still ranks above the most popular item without evidence.
- **One order for ties** ([baselines.py:24-31](src/seqrec_eval/baselines.py#L24-L31)). Equal scores are
  common here (successors a Markov chain saw once each, items with the same count), so they are broken in one
  fixed order: the more popular training item first, then the lower index. A top-k that broke ties as it
  happened to could pick different items for a different `k`, and the evaluation asks for longer lists when
  it excludes seen items (Step 3.9.5), so the same histories once scored differently on the full data than at
  an ablation level (ML-20M, 2026-09-30: Markov 0.0230 against 0.0226). This is `BASELINE_VERSION` 2.
- **Predicting** (`predict_on_batch`, [baselines.py:209-220](src/seqrec_eval/baselines.py#L209-L220)): turn
  the histories into a binary seen matrix over the fitted catalogue (`_seen_matrix`,
  [baselines.py:62-69](src/seqrec_eval/baselines.py#L62-L69)), check `k` against it with the library's
  `validate_candidate_topk`, and rank by one of two paths that give the same lists (tested):
  - **`_rank_sparse`** ([baselines.py:233-280](src/seqrec_eval/baselines.py#L233-L280)), used whenever no
    `candidate_ids` are given, which is always the case in the suite. It never builds a rows × items array:
    dense, an Amazon-sized catalogue took 12 GiB and 7 s per batch of 1,024 (final review B1). Each row's list
    is its **upper tier** (the items it has evidence for), best first, then the **popularity tier**, which is
    one order for everyone. A subclass gives its upper tier as `(rows, items, values)` through `_evidence`
    ([baselines.py:194-203](src/seqrec_eval/baselines.py#L194-L203)); one that scores every history alike
    returns `None` there and gives `_global_scores` ([baselines.py:205-207](src/seqrec_eval/baselines.py#L205-L207))
    instead, and then every row takes the first `k` items of that one order that it has not seen
    (`_first_free`, [baselines.py:131-152](src/seqrec_eval/baselines.py#L131-L152); [baselines.py:249-253](src/seqrec_eval/baselines.py#L249-L253)).
    Otherwise: seen items leave the upper tier when excluding ([baselines.py:255-260](src/seqrec_eval/baselines.py#L255-L260));
    each row's upper tier is sorted by (value, popularity, index) and cut at `k`
    ([baselines.py:261-266](src/seqrec_eval/baselines.py#L261-L266)); the rest of the row is filled from the
    popularity order, skipping the row's own evidence and seen items ([baselines.py:267-271](src/seqrec_eval/baselines.py#L267-L271));
    the two parts are joined row by row ([baselines.py:272-280](src/seqrec_eval/baselines.py#L272-L280)).
  - **`_rank_dense`** ([baselines.py:222-231](src/seqrec_eval/baselines.py#L222-L231)), for a call with
    `candidate_ids`: the subclass's `_scores(source)` gives a dense score per (history, item), seen items are
    masked with the library's `mask_seen_numpy`, and `_top_k` ([baselines.py:83-110](src/seqrec_eval/baselines.py#L83-L110))
    takes each row's best `k` in the tie order above. `_top_k` asks torch for `k + 64` candidates
    (`_TIE_ROOM`, [baselines.py:80](src/seqrec_eval/baselines.py#L80)) and orders them exactly; only a row
    whose tie runs past those is ranked in full (`_top_k_full`, [baselines.py:113-128](src/seqrec_eval/baselines.py#L113-L128)),
    which alone is 100 to 200 times slower.
- **Saving** ([baselines.py:282-293](src/seqrec_eval/baselines.py#L282-L293)) follows the library's
  checkpoint contract; the analysis never saves them.

### Step 3.8.2: `popularity`

**code:**
`PopularityConfig` and `Popularity` at [baselines.py:300-341](src/seqrec_eval/baselines.py#L300-L341);
searched over `count` ∈ {events, users} ([protocol.toml:258-261](protocol.toml#L258-L261)).

**explanation:**
The same list for everyone: the most popular training items. With `count = "events"` an item's score is
its number of training events (`counts_ = popularity_`); with `"users"` it is the number of distinct users
who had it ([baselines.py:322-326](src/seqrec_eval/baselines.py#L322-L326)), so one heavy re-consumer
counts once. `_global_scores` puts those counts into the two tiers
([baselines.py:332-333](src/seqrec_eval/baselines.py#L332-L333)), and every user gets that one order, less
what they have seen. **It measures** how far you get knowing nothing about the user. A model that does not
beat it has learned nothing personal.

### Step 3.8.3: `time_popularity`

**code:**
`TimeDecayedPopularityConfig` and `TimeDecayedPopularity` at [baselines.py:348-402](src/seqrec_eval/baselines.py#L348-L402);
searched over `half_life_days` loguniform [0.25, 365] with 10 random trials
([protocol.toml:263-266](protocol.toml#L263-L266)).

**explanation:**
Popularity in which each event counts `0.5 ** (age / half_life)`, age in days measured from the last
training event ([baselines.py:384-386](src/seqrec_eval/baselines.py#L384-L386)). A short half-life means
"what is popular right now"; a long one approaches plain popularity. Measuring age from any later moment
would multiply every weight by the same factor and leave the ranking unchanged. It needs the recovered
timestamps and checks there is one per training event ([baselines.py:377-382](src/seqrec_eval/baselines.py#L377-L382)).
An item whose events all decayed below floating-point precision falls to the popularity tier
([baselines.py:392-394](src/seqrec_eval/baselines.py#L392-L394)). **It measures** how much of the next item
is a current trend rather than all-time popularity.

### Step 3.8.4: `replay`

**code:**
`ReplayConfig` and `Replay` at [baselines.py:409-459](src/seqrec_eval/baselines.py#L409-L459);
searched over `order` ∈ {recency, frequency} ([protocol.toml:268-271](protocol.toml#L268-L271)).

**explanation:**
Recommends the user's own history back, then popular items after it. Its upper tier, `_evidence`
([baselines.py:440-459](src/seqrec_eval/baselines.py#L440-L459)), finds for each (user, item) in the
history its most recent position from the end (1 = the last event,
[baselines.py:445-451](src/seqrec_eval/baselines.py#L445-L451)) and gives it a score above the popularity tier:

- `"recency"`: score `1 + 1/position`, so the most recent item first;
- `"frequency"`: score `1 + count + recency/2`, so the most repeated item first, with recency (always < 1
  after halving) only separating equal counts ([baselines.py:453-459](src/seqrec_eval/baselines.py#L453-L459)).

`_scores` ([baselines.py:433-438](src/seqrec_eval/baselines.py#L433-L438)) writes the same values into a
dense array for the dense path. Under `exclude_seen = true` (ML-20M, Amazon) everything it would replay is
masked, so it reduces to popularity. **It measures** how much of the next item is re-consumption. It only
differs from popularity where `exclude_seen = false` (Music4All, Yambda, OTTO).

### Step 3.8.5: `markov`

**code:**
`MarkovConfig` and `MarkovChain` at [baselines.py:466-560](src/seqrec_eval/baselines.py#L466-L560);
no settings, so one trial ([protocol.toml:273-274](protocol.toml#L273-L274)).

**explanation:**
First-order Markov: the next item from the last one only, P(next | last).

- **fit** ([baselines.py:489-501](src/seqrec_eval/baselines.py#L489-L501)): count every transition between
  adjacent events of the same history (`same_row` excludes the jump from one user's last event to the next
  user's first), including self-transitions a → a, then normalise each row of the item × item count matrix
  to probabilities.
- **score**: take each history's last event (`_last`, [baselines.py:503-509](src/seqrec_eval/baselines.py#L503-L509));
  if it is a known item, its row of transition probabilities is the upper tier; if the history is empty or
  ends in a new item, the row has no evidence and the whole list is popularity. On the sparse path,
  `_ranked_successors` ([baselines.py:519-536](src/seqrec_eval/baselines.py#L519-L536)) sorts every item's
  successors once, on first use, in the ranking's order (probability, popularity, index), and `_evidence`
  ([baselines.py:538-547](src/seqrec_eval/baselines.py#L538-L547)) hands each row the first `k` + (its seen
  items) of its last item's successors, which is all a list of `k` can use. A popular item can have hundreds
  of thousands of successors, so sorting them per history would be slow. `_scores`
  ([baselines.py:511-517](src/seqrec_eval/baselines.py#L511-L517)) is the dense equivalent.

**It measures** the simplest use of order. It is also the instrument of the sequence-signal analysis
(Step 3.11): compared with itself trained on shuffled or reversed histories, it shows whether order carries
information at all.

## Step 3.9: scoring: `_score` → `evaluate_phase`

**code:**
`_score` at [analysis.py:126-129](src/seqrec_eval/analysis.py#L126-L129); `evaluate_phase` at
[evaluate.py:324-390](src/seqrec_eval/evaluate.py#L324-L390). The models use the same function
(Step 5.4).

**explanation:**
`_score` picks the rows (`split.val_rows` for validation, `split.test_rows` for test, which is `None` =
all on the full data), and calls `evaluate_phase` with `family = "sequence"` (baselines read histories) and
the dataset's `exclude_seen`. `evaluate_phase` is where every number in the suite comes from:

### Step 3.9.1: guards and the target definition

[evaluate.py:332-338](src/seqrec_eval/evaluate.py#L332-L338). `targets` is `"primary"` (the protocol's
definition), `"next"` or `"window"` explicitly, or `"new"` (the protocol's targets minus items already in the
history). A split trained on `train+val` refuses to score validation: that model has seen the validation
window, so its score would be leaked ([evaluate.py:334-336](src/seqrec_eval/evaluate.py#L334-L336)). The
target matrix is `phase_targets` ([evaluate.py:287-292](src/seqrec_eval/evaluate.py#L287-L292)):
`{phase}_next_target_matrix` for `"next"`, `{phase}_target_matrix` for `"window"`, with a clear error if the
next-item targets were never prepared. For `"new"`, `new_item_targets`
([evaluate.py:268-274](src/seqrec_eval/evaluate.py#L268-L274)) removes the targets already in the user's whole
history ([evaluate.py:340-342](src/seqrec_eval/evaluate.py#L340-L342)).

### Step 3.9.2: which users are scored

[evaluate.py:339-355](src/seqrec_eval/evaluate.py#L339-L355); `scored_rows` at
[evaluate.py:246-261](src/seqrec_eval/evaluate.py#L246-L261), `recommendable_next` at
[evaluate.py:206-214](src/seqrec_eval/evaluate.py#L206-L214), `reachable_targets` at
[evaluate.py:241-243](src/seqrec_eval/evaluate.py#L241-L243), `fills_list` at
[evaluate.py:223-238](src/seqrec_eval/evaluate.py#L223-L238).

Two filters, each counted in the result's metadata:

1. **A next item a model can recommend** ([evaluate.py:343-346](src/seqrec_eval/evaluate.py#L343-L346)).
   For next-item targets, only users whose next target contains at least one item a model could put in its
   list are scored. That excludes a next item first seen after training or deleted by the builder (an empty
   row, Step 2.8.3), and, under `exclude_seen`, a next item already in the user's history, which the
   exclusion takes out of every list (`reachable_targets`; review C4: on Amazon, variants of one product
   share an item). Such a user would score 0 for every model, carry nothing for a comparison, and only pull
   every mean down. Under refit the catalogue is the validation catalogue, so items first seen in validation
   count as recommendable at test. Those rows are intersected with any given sample (`val_rows`,
   `test_rows`), and how many of the rows asked for were dropped is `rows_unrecommendable_next`. For window
   targets every row is kept (`scored_rows` returns `None`).
2. **A full list of unseen items** ([evaluate.py:347-355](src/seqrec_eval/evaluate.py#L347-L355)). Under
   `exclude_seen`, a user who has seen all but fewer than `k` = 20 items of the training catalogue cannot be
   given 20 unseen ones. That only happens where the catalogue is small beside a history (a catalogue
   sweep's smallest levels, for its heaviest users), and such a row would stop the whole evaluation
   (Step 3.9.5), so it is left out and counted as `rows_too_few_unseen` (review N49).

### Step 3.9.3: the adapter and the source

[evaluate.py:356](src/seqrec_eval/evaluate.py#L356); `phase_inputs` at
[evaluate.py:311-317](src/seqrec_eval/evaluate.py#L311-L317), `phase_model` at
[evaluate.py:295-300](src/seqrec_eval/evaluate.py#L295-L300), `phase_source` at
[evaluate.py:303-308](src/seqrec_eval/evaluate.py#L303-L308); cr's `WarmCatalogAdapter`
([cr models/cold_start.py:209](vendor/compresso-recsys/src/compresso_recsys/models/cold_start.py#L209)).

The model was fitted on the training catalogue; the phase's catalogue is larger (it appends the items
first seen in later windows). `WarmCatalogAdapter(model, train_item_ids, phase_item_ids)` hands the model its
own item space and maps its ranked columns back into the phase's catalogue, so new items remain valid
targets that no model can ever recommend, equally for every model. The source is
`{phase}_source_sequences` for a sequence model (passed whole: a new item keeps its position and the model
reads it as unknown) or `{phase}_source_matrix` for a matrix model, projected onto the training columns by
`adapter.align_source` ([cr models/cold_start.py:416-441](vendor/compresso-recsys/src/compresso_recsys/models/cold_start.py#L416-L441);
a matrix model has no column for a new item).

### Step 3.9.4: ids, the seen history, and cutting to the rows

[evaluate.py:357-369](src/seqrec_eval/evaluate.py#L357-L369). The sample ids are `split.eval_user_ids(phase)`.
The seen history is `seen_history` ([evaluate.py:217-220](src/seqrec_eval/evaluate.py#L217-L220)):
`{phase}_seen_matrix` where an ablation condition stored one (Step 10.5), else the phase's source matrix,
so always the user's whole original history. If rows were chosen, source, targets, ids and seen matrix are
all cut to them together (`_take`, [evaluate.py:320-321](src/seqrec_eval/evaluate.py#L320-L321)).

### Step 3.9.5: excluding seen items

[evaluate.py:358-373](src/seqrec_eval/evaluate.py#L358-L373); `ExcludeSeenPolicy` at
[evaluate.py:74-203](src/seqrec_eval/evaluate.py#L74-L203).

The library's evaluator calls `predict_on_batch(source, k=...)` and nothing else, so `exclude_seen` has to
be bound to the model beforehand: `ExcludeSeenPolicy` wraps the adapter and is what the evaluator calls.

- **Without `exclude_seen`** (Music4All, Yambda, OTTO) the policy just forwards the call with
  `exclude_seen=False` ([evaluate.py:171-172](src/seqrec_eval/evaluate.py#L171-L172)).
- **With `exclude_seen`** (ML-20M, Amazon), seen items are excluded *after* the model ranks, against the
  whole history, in every evaluation ([evaluate.py:358-361](src/seqrec_eval/evaluate.py#L358-L361)). The
  policy asks the model for its top `k + extra` with its own filter off, drops the seen items, and keeps the
  first `k` in the model's order ([evaluate.py:176-203](src/seqrec_eval/evaluate.py#L176-L203)). Excluding
  inside the model on the full data but after it on an ablation condition (where it must be after, because
  the seen history is longer than the input) would let the two differ by tie-breaking alone (review A6), so
  both go the same way.
- **One width per phase** ([evaluate.py:362-364](src/seqrec_eval/evaluate.py#L362-L364)). `extra` is the
  largest number of items any user **of the whole phase** has seen, so every batch asks the model for the
  same list length. A top-k breaks ties differently for a different length, so a width taken per batch made
  a tie-heavy model's lists depend on the batch size and on which users shared a batch (review N30). The
  list is capped at the training catalogue size (`limit`). If a row still has fewer than `k` unseen items in
  it, the policy raises ([evaluate.py:194-198](src/seqrec_eval/evaluate.py#L194-L198)); `fills_list` keeps
  such rows out beforehand (Step 3.9.2).
- **Batches checked** (review H33). A batch carries no user ids, so the seen rows are found by a running
  offset, and three checks make sure that offset is right: each batch must equal the source's next rows
  (`_check_rows`, [evaluate.py:125-142](src/seqrec_eval/evaluate.py#L125-L142)), every item a row reads must
  be in its seen row (`_check_batch`, [evaluate.py:144-162](src/seqrec_eval/evaluate.py#L144-L162); a matrix
  source's columns are first mapped into the phase's catalogue with `source_columns`), and after the
  evaluation every row must have been used exactly once (`finish`, [evaluate.py:164-168](src/seqrec_eval/evaluate.py#L164-L168),
  called at [evaluate.py:389](src/seqrec_eval/evaluate.py#L389)). A breach raises `BatchAlignmentError`.

### Step 3.9.6: the library's evaluator and what comes back

[evaluate.py:374-390](src/seqrec_eval/evaluate.py#L374-L390); cr's `evaluate_recommender`
([cr evaluation.py:800](vendor/compresso-recsys/src/compresso_recsys/evaluation.py#L800)) and
`EvaluationResult` ([cr evaluation.py:131](vendor/compresso-recsys/src/compresso_recsys/evaluation.py#L131)).

`evaluate_recommender` asks the policy for the top `max(cutoffs)` = 20 per row, in batches of
`eval_batch_size`, and computes every metric of `build_metrics` ([evaluate.py:264-265](src/seqrec_eval/evaluate.py#L264-L265):
the seven metric classes at the three cutoffs, 21 values such as `ndcg@10`). Rows without any target are
skipped (`valid = target_counts > 0`, [cr evaluation.py:623](vendor/compresso-recsys/src/compresso_recsys/evaluation.py#L623)).
It returns an `EvaluationResult`:

| field | content |
|---|---|
| `metrics` | mean of each metric over the scored rows, e.g. `{"ndcg@10": 0.0831, ...}` |
| `per_user` | per metric, one value per scored row; what every paired comparison uses |
| `sample_ids` | the user id of each scored row, in order; two results are paired only if these match |
| `n_rows`, `n_scored_rows` | rows given; rows with at least one target |
| `required_k` | 20 |
| `metadata` | the suite's: `phase`, `exclude_seen`, `targets`, `definition`, `rows_sampled`, `rows_unrecommendable_next` and `rows_too_few_unseen` (the two filters of Step 3.9.2) |
| `target_fingerprint` | a hash of the targets scored against; the statistics refuse to pair two results scored on different targets |

## Step 3.10: caching: `_cached`, `save_evaluation`, the tie mask

**code:**
`_cached` at [analysis.py:183-192](src/seqrec_eval/analysis.py#L183-L192); `save_evaluation`,
`load_evaluation`, `evaluation_exists` at [results.py:69-109](src/seqrec_eval/results.py#L69-L109);
`_save_tied` at [analysis.py:195-202](src/seqrec_eval/analysis.py#L195-L202) with `tied_split`
([analysis.py:426-435](src/seqrec_eval/analysis.py#L426-L435)) and `last_pair_tied`
([ablations.py:1031-1040](src/seqrec_eval/ablations.py#L1031-L1040)).

**explanation:**
`_cached(stem, compute, record)` is the whole caching rule of the analysis: if `<stem>.json` and
`<stem>.npz` both exist, load and return them; otherwise call `compute()` (fit + score), save, and write
`<stem>.done.json` = the record plus the seconds taken and the test metrics. A rerun computes only what is
missing.

`EvaluationResult` has no file format of its own, so `save_evaluation` stores it as a pair: `<stem>.npz`
with every per-user array (`per_user__ndcg@10`, …) and the `sample_ids`, and `<stem>.json` with metrics,
row counts, `required_k`, metadata and the target fingerprint, both written durably (Step 2.10).
`load_evaluation` rebuilds the object, so a report can pair users long after the process that scored them
has exited.

For Markov, `_save_tied` writes `tied.npy` beside the result: for each scored user, whether the last two
events of their test history share a timestamp. `last_pair_tied` computes it per test row from
`test_source_timestamps`, and `tied_split` maps it onto the result's `sample_ids`. Without timestamps nothing
is written. The mask is used by the report to split Markov's score into "history ends in a tie" and "ends in
a real gap" (Step 4.1).

## Step 3.11: the sequence-signal controls

**code:**
[analysis.py:277-284](src/seqrec_eval/analysis.py#L277-L284); `CONTROLS` at
[analysis.py:91](src/seqrec_eval/analysis.py#L91); `fit_control` at
[analysis.py:223-231](src/seqrec_eval/analysis.py#L223-L231); `shuffled_within_users` and
`reversed_histories` at [analysis.py:210-220](src/seqrec_eval/analysis.py#L210-L220);
`_controls_fingerprint` at [analysis.py:234-238](src/seqrec_eval/analysis.py#L234-L238);
`_control_stream` at [analysis.py:312-323](src/seqrec_eval/analysis.py#L312-L323).

**explanation:**
Two controls ask whether order carries information:

- `markov_shuffled`: Markov fitted on training histories whose events are shuffled within each user
  (`np.lexsort` by row, then a random key, [analysis.py:212](src/seqrec_eval/analysis.py#L212)). Which items
  a history holds is unchanged; the order is gone. A large drop from Markov is order the data carries.
- `markov_backwards`: Markov fitted on each history reversed ([analysis.py:218-219](src/seqrec_eval/analysis.py#L218-L219)),
  so a → b is counted as b → a. A small drop means much of what Markov learned is co-occurrence, not
  direction.

Both are fitted on `tested` and scored on the same test users as Markov, with the same caching as the
baselines (and the same `test_window` diagnostic). They live at `root/controls/<control>/<controls
fingerprint[:12]>/`. The fingerprint covers the split, `search_seed`, `CONTROLS_VERSION` (3,
[analysis.py:90](src/seqrec_eval/analysis.py#L90)), the result settings and the whole dataset section
(`exclude_seen` changes the controls' scores but not the split's fingerprint). The shuffle's random stream
is `_stream(search_seed, "control", dataset, "full")` wherever the training data is the full data's: the
full data itself, every sweep's reference, and every condition of an inference sweep, so those are all the
same control. Drawing a new shuffle per level of an inference sweep made the control move between levels by
chance alone (on ML-20M from 0.0155 to 0.0178 while Markov did not move). A condition that changes the
training data gets a stream of its own (sweep, level, data seed).

## Step 3.12: `analyse_sweep`, the analysis at every ablation condition

**code:**
[cli.py:339-342](src/seqrec_eval/cli.py#L339-L342) → `analyse_sweep` at
[analysis.py:326-356](src/seqrec_eval/analysis.py#L326-L356), with `_condition_scorers`
([analysis.py:297-309](src/seqrec_eval/analysis.py#L297-L309)) and `_condition_stem`
([analysis.py:291-294](src/seqrec_eval/analysis.py#L291-L294)).

**explanation:**
Each sweep that lists this dataset gets the same analysis at every condition.

1. **Scorers** ([analysis.py:332](src/seqrec_eval/analysis.py#L332)): for each baseline, its setting
   selected on the full data (`select_baseline` again; it returns the cached `selected.json`) and a
   fingerprint of the baseline's fingerprint plus those params; for each control, the controls fingerprint.
   Baselines are **not retuned per level**, as the models keep their stage-1 configuration, so a gap cannot
   move because only one side was retuned.
2. **The base** ([analysis.py:333](src/seqrec_eval/analysis.py#L333)): `tested = final_split(split)`; every
   condition is a transform of the refitted data, as the models' ablation runs are.
3. **`run(condition, name)`** ([analysis.py:335-342](src/seqrec_eval/analysis.py#L335-L342)): record the
   condition's characteristics against `tested` (`record_characteristics`, Step 10.7); then for each
   scorer, fit it on the condition's training data and score it on the condition's test users, cached at
   `ablations/<sweep>/<dataset>/<condition fingerprint[:12]>/analysis/<scorer>/<fp[:12]>/<condition>/test`,
   with Markov's `tied.npy`.
4. **`pending(name)`** ([analysis.py:344-347](src/seqrec_eval/analysis.py#L344-L347)): true if any scorer's
   result is missing. A condition is only built (a transform can be expensive) when something is missing.
5. **The order** ([analysis.py:349-356](src/seqrec_eval/analysis.py#L349-L356)): the full-data reference
   first (`build_reference`, Step 10.4), then every level × data seed (`build_condition`, Step 10.4). The
   data seeds are the sweep's seeds, including any added later (Step 10.1). The conditions and their test
   users are exactly those the models' ablation runs use (Part 10).

---

# Part 4: `seqrec-eval analysis-report`

## Step 4.1: building the report

**code:**
[cli.py:347-356](src/seqrec_eval/cli.py#L347-L356) → `build_analysis_report` at
[analysis_report.py:187-203](src/seqrec_eval/analysis_report.py#L187-L203) → `dataset_analysis` at
[analysis_report.py:89-176](src/seqrec_eval/analysis_report.py#L89-L176); results read by `full_results`
([analysis.py:363-388](src/seqrec_eval/analysis.py#L363-L388)).

**explanation:**
Nothing is computed from data here; the report reads what `analyse` cached. `full_results` collects, for
one dataset: the profile, every baseline's `selected.json` and test result (and Markov's tie mask), both
controls' test results, and the diagnostic (`test_window`) results. Per dataset the report has four sections:

1. **Data profile** ([analysis_report.py:98](src/seqrec_eval/analysis_report.py#L98)): `profile_table`
   ([analysis_report.py:69-72](src/seqrec_eval/analysis_report.py#L69-L72)) prints the fields of Step 3.5 in
   the order of `PROFILE_ROWS` ([analysis_report.py:30-49](src/seqrec_eval/analysis_report.py#L30-L49)), for
   the training data and the test inputs, formatted by `_format` ([analysis_report.py:52-66](src/seqrec_eval/analysis_report.py#L52-L66)).
2. **Baselines and the floor** ([analysis_report.py:100-124](src/seqrec_eval/analysis_report.py#L100-L124)):
   per baseline its kind, selected setting, number of trials, validation `ndcg@10`, and test `ndcg@10`,
   `recall@10`, `hit_rate@10`. The **floor** is `floor_of` ([analysis.py:415-423](src/seqrec_eval/analysis.py#L415-L423)):
   the baseline with the highest **test** primary metric, marked in the table. Choosing it on test makes the
   floor the strongest bound a model could be held to. It waits for **every** baseline: until each has a
   test result `floor_of` returns `None` and the report says so, because the strongest of those finished
   would be a lower floor, silently (review N34).
3. **Sequence signal** ([analysis_report.py:126-159](src/seqrec_eval/analysis_report.py#L126-L159)): Markov's
   test score and each control's, with the paired difference (control − Markov) and its 95% bootstrap
   interval from the library's `compare_models` without correction (`paired`,
   [analysis_report.py:75-77](src/seqrec_eval/analysis_report.py#L75-L77)). These differences are descriptive:
   read, not tested. Below it, Markov and both controls split by the tie mask: users whose history ends in a
   tie and users whose history ends in a real gap; a slice under `MIN_SLICE_USERS` = 100 users is not shown
   (`_slice_mean`, [analysis_report.py:84-86](src/seqrec_eval/analysis_report.py#L84-L86)).
4. **The other target definition** ([analysis_report.py:161-175](src/seqrec_eval/analysis_report.py#L161-L175)):
   every scorer's test `ndcg@10` on the primary targets beside the diagnostic ones.

The CSV (`analysis.csv`) has one row per baseline, control and diagnostic with all metrics and its role
([analysis_report.py:196-202](src/seqrec_eval/analysis_report.py#L196-L202)).

---

# Part 5: `seqrec-eval search --device cuda:0`

## Step 5.1: the CLI branch

**code:**
[cli.py:276-312](src/seqrec_eval/cli.py#L276-L312) (the `search` path).

**explanation:**
`--threads` sets torch's CPU threads ([cli.py:277-278](src/seqrec_eval/cli.py#L277-L278)). Every selected
model is looked up in the registry first ([cli.py:279-280](src/seqrec_eval/cli.py#L279-L280)), so a model in
the protocol but not in `models.py` fails before any split is loaded. Then per dataset the split is loaded as
prepared, checked against the protocol, trained on the training window, with no `final_split`
([cli.py:291](src/seqrec_eval/cli.py#L291)). Per model, `plan_trials` gives the trial specs
([cli.py:295-296](src/seqrec_eval/cli.py#L295-L296)) and each is passed to `execute`
([cli.py:306-309](src/seqrec_eval/cli.py#L306-L309)). The statuses are counted and logged at the end
([cli.py:311](src/seqrec_eval/cli.py#L311)), and they decide the exit code (Step 0.1,
[cli.py:312](src/seqrec_eval/cli.py#L312)).

## Step 5.2: planning the trials and drawing the configurations

**code:**
`plan_trials` at [runner.py:172-179](src/seqrec_eval/runner.py#L172-L179); `RunSpec` at
[runner.py:76-108](src/seqrec_eval/runner.py#L76-L108); `trial_params` at
[search.py:59-66](src/seqrec_eval/search.py#L59-L66), `_draw` at [search.py:34-47](src/seqrec_eval/search.py#L34-L47).

**explanation:**
A `RunSpec` describes one run: `dataset`, `model`, `kind` (`"trial"`, `"final"`, or for ablations
`"reference"`/`"rescore"`), `index` (trial number, or the seed of a final), `seed`, `params`, `fingerprint`
(the run fingerprint), `source_trial` (for a final: the trial it came from) and `condition` (only for
ablation runs). Its folder is `work/runs/<dataset>/<model>/<run fingerprint[:12]>/trial-007`
(`name`, `directory`, `run_root`: [runner.py:89-112](src/seqrec_eval/runner.py#L89-L112)). `label`
([runner.py:93-103](src/seqrec_eval/runner.py#L93-L103)) is how the logs name the run; an ablation run's
folder is also `final-seed<s>`, so it is logged by sweep, condition and what it does instead
("history_length 20: refit seed 0").

`plan_trials` makes one spec per trial, all with seed `seeds[0]`. The configuration of trial *i* is
`trial_params`: the `fixed` values, plus one draw per searched parameter in name order, each from its own
stream `_stream(search_seed, model, trial, parameter)`. Two consequences:

- trial *i* of a model draws the same configuration on **every dataset** (the dataset is not in the
  stream), so "trial 7" means one configuration across the study;
- adding a parameter to a space does not move the draws of the others.

`_draw` samples a `choice` by index (so values stay plain Python), an `int` inclusive, a `uniform`, and a
`loguniform` as `exp(uniform(log low, log high))`.

## Step 5.3: `execute`: resumable, locked, failures recorded

**code:**
`execute` at [runner.py:516-593](src/seqrec_eval/runner.py#L516-L593); `_check_training` at
[runner.py:393-399](src/seqrec_eval/runner.py#L393-L399); `RunLock` at
[runner.py:316-368](src/seqrec_eval/runner.py#L316-L368); `made_with_another_selection` at
[runner.py:499-513](src/seqrec_eval/runner.py#L499-L513).

**explanation:**
Every run of the suite (trials, finals, ablation runs) goes through `execute`. It returns a status string,
in this order:

1. `_check_training` refuses the wrong split: a trial must get a split trained on `"train"`; under refit,
   everything else must get `"train+val"` ([runner.py:519](src/seqrec_eval/runner.py#L519)).
2. `done.json` exists → `"cached"`; a run is finished exactly when this file exists. Unless the finished run
   was made under another selection: a final (and every ablation run built on one) records the trial and
   params it was planned with in `spec.json`, and if the search now selects another trial (a failed trial
   accepted, then rerun and found best; a search finished after `--allow-incomplete`), the folder is the same
   but the run describes another model. It is refused as `"stale-selection"` (exit 1), and must be moved
   aside to be redone ([runner.py:521-526](src/seqrec_eval/runner.py#L521-L526); final review A3).
3. `failed.json` exists and no `--retry-failed` → `"failed-before"`, so a configuration that runs out of
   memory does not retry itself all weekend ([runner.py:527-528](src/seqrec_eval/runner.py#L527-L528)).
4. A `reference` or `rescore` (ablations, Part 10) reloads a stage-1 final, so that final must be finished
   and of the current selection ([runner.py:529-543](src/seqrec_eval/runner.py#L529-L543)): a failed one →
   `"stage1-failed"` (exit 1); one not made yet → `"waiting-for-stage1"` (exit 3: an order to wait for, as when
   a seed is added to stage 1 and to a sweep at once on two GPUs); a stale one → `"stale-selection"`.
5. **The lock** ([runner.py:544-546](src/seqrec_eval/runner.py#L544-L546)). `RunLock.acquire`
   ([runner.py:329-347](src/seqrec_eval/runner.py#L329-L347)) takes an exclusive `flock` on `<run>/.lock`,
   without waiting (three tries 50 ms apart, since a `status` probe holds it for an instant). If another
   process holds it → `"running-elsewhere"`. That is how two processes (one per GPU) running the same command
   divide the work. The kernel holds the lock for the process and drops it the moment the process ends,
   however it ends (crash, out-of-memory kill, reboot), so a claim is never stale and no process id has to be
   trusted (final review B2). It needs a local file system. The file stays in place with its last owner
   written in it, for people. `done.json` is checked once more after the claim, since another process may
   have finished the run in between ([runner.py:547-550](src/seqrec_eval/runner.py#L547-L550); H45).
6. **Silent deaths** ([runner.py:551-567](src/seqrec_eval/runner.py#L551-L567)). A process killed while
   running (the kernel's out-of-memory killer, SIGKILL) records nothing, so every restart would run the same
   run first, for ever. `attempts.json` counts the starts; once a run has been started `MAX_ATTEMPTS` = 2
   times ([runner.py:309](src/seqrec_eval/runner.py#L309)) without finishing or failing, it is written as
   failed with a `ProcessDied` error (review B3). `--retry-failed` resets the count.
7. `spec.json` is written, the run is logged (with its params if it trains), `_execute` runs it, and on
   success any old `failed.json` and `attempts.json` are removed ([runner.py:568-578](src/seqrec_eval/runner.py#L568-L578)).
8. A `KeyboardInterrupt` (Ctrl-C, or `kill`/a closed terminal through Step 0.1) removes `attempts.json` and
   is re-raised: stopped by the operator, not a death ([runner.py:579-581](src/seqrec_eval/runner.py#L579-L581)).
9. On any other exception, `failed.json` gets the spec, host, device, error and traceback; status `"failed"`
   ([runner.py:582-589](src/seqrec_eval/runner.py#L582-L589)).
10. Always: release the lock and empty the CUDA cache ([runner.py:590-593](src/seqrec_eval/runner.py#L590-L593)).

## Step 5.4: `_execute` for a trial

**code:**
`_execute` at [runner.py:402-482](src/seqrec_eval/runner.py#L402-L482).

**explanation:**
1. **Checks** ([runner.py:404-410](src/seqrec_eval/runner.py#L404-L410)): the registered family must match
   the protocol's; `_check_condition` ([runner.py:381-390](src/seqrec_eval/runner.py#L381-L390)) refuses a
   stage-1 spec on an ablation split and vice versa.
2. **The record** ([runner.py:412-416](src/seqrec_eval/runner.py#L412-L416)): spec, fingerprint, host,
   device, `trained_on`, training catalogue size, and `code_provenance()`
   ([splits.py:140-154](src/seqrec_eval/splits.py#L140-L154)): the library's and the suite's version and
   source hash, so the report can tell when results come from different code builds (review H05, H10).
3. **`max_items`** ([runner.py:417-425](src/seqrec_eval/runner.py#L417-L425)): under refit a trial checks
   the *validation* catalogue, the one its finals will fit, so a model is never searched in full and then
   skipped at the finals. Too large → `done.json` with `status: "skipped"` and the reason.
4. **Peak memory** ([runner.py:427](src/seqrec_eval/runner.py#L427)): `_reset_peak_memory`
   ([runner.py:485-496](src/seqrec_eval/runner.py#L485-L496)) starts CUDA before resetting torch's memory
   counter, which otherwise refuses an explicit device such as `cuda:0` (on the first DGX run every run failed
   at once with "Invalid device argument").
5. **Fit** ([runner.py:434-440](src/seqrec_eval/runner.py#L434-L440)): seed `random`, numpy and torch
   (`_seed_everything`, [runner.py:375-378](src/seqrec_eval/runner.py#L375-L378)); build the trainer from the
   registry (Step 5.5) with `n_items = len(train_item_ids)`; fit on `x_train_sequences` (sequence family) or
   `x_train` (matrix family); record `fit_seconds`.
6. **Validation** ([runner.py:442-447](src/seqrec_eval/runner.py#L442-L447)): `evaluate_phase(trainer,
   split, "val", rows=split.val_rows)` (Step 3.9, with the model's own family), saved as `val.json` +
   `val.npz`, metrics into the record. A trial never touches test.
7. **Finish** ([runner.py:473-482](src/seqrec_eval/runner.py#L473-L482)): `eval_seconds`, the trainer's
   training `history` (loss per epoch) if it has one, peak GPU memory, `status: "done"`, `done.json`.

**Output:** `runs/<dataset>/<model>/<fp>/trial-NNN/` with `spec.json`, `val.json`, `val.npz`, `done.json`
(or `failed.json`), `.lock`, and `attempts.json` while it runs.

## Step 5.5: the models

**code:**
[models.py:46-105](src/seqrec_eval/models.py#L46-L105); the search spaces at
[protocol.toml:176-246](protocol.toml#L176-L246).

**explanation:**
`register(name, family, cls)` ([models.py:58-64](src/seqrec_eval/models.py#L58-L64)) puts a builder in
`REGISTRY` as a `ModelSpec` (name, family, trainer class for reloading, builder). `model_spec(name)`
([models.py:67-71](src/seqrec_eval/models.py#L67-L71)) looks it up. Every builder takes `(params, n_items,
device, seed)` and returns an unfitted trainer following the library's contract
(`fit(data, item_ids=...)`, then `predict_on_batch(source, k=..., exclude_seen=...)`):

| name | family | library trainer | built at | searched |
|---|---|---|---|---|
| `popularity` | matrix | `PopularityBaseline` | [models.py:77-79](src/seqrec_eval/models.py#L77-L79) | nothing, 1 trial. The library's popularity model; the suite's own popularity *baseline* (Step 3.8.2) is separate |
| `ease` | matrix | `EASE` | [models.py:82-84](src/seqrec_eval/models.py#L82-L84) | `l2`; `max_items = 40000` (dense item × item) |
| `elsa` | matrix | `ELSATrainer` | [models.py:87-89](src/seqrec_eval/models.py#L87-L89) | `latent_dim`, `batch_size`, `lr`, `epochs`; gets device and seed |
| `gru` | sequence | `SimpleRNNTrainer` | [models.py:92-100](src/seqrec_eval/models.py#L92-L100) | `embedding_dim`, `hidden_dim`, `num_layers`, `dropout`, `lr`, `batch_size`, `epochs`, `max_history_length`, `unk_dropout` (`rnn_type = "gru"` fixed). `max_history_length` is taken out of the params and given to the `SequenceBatcher`, which is where SimpleRNN takes its context from |
| `sasrec` | sequence | `SASRecTrainer` | [models.py:103-105](src/seqrec_eval/models.py#L103-L105) | `d_model`, `n_blocks`, `n_heads`, `dropout`, `lr`, `batch_size`, `epochs` (10 to 100), `n_negatives`, `unk_dropout`, `max_history_length` (a config field) |

Progress bars are off because runs are unattended. `max_history_length` is the one parameter name every
sequential model shares. `unk_dropout` is how often a training input is replaced by the unknown token, the
only way that token's embedding (used for items first seen after training) learns anything (N7).

EASE scores a batch as `source @ weights`. The suite always asks for every item, and the vendored copy then
uses the item × item weight matrix itself ([cr models/ease.py:190-199](vendor/compresso-recsys/src/compresso_recsys/models/ease.py#L190-L199)).
The library selected its columns on every call, which copied the whole matrix (1.6 GB on ML-20M): 3.8 s per
latency request and about 80 s per validation scoring, for the same scores (DECISIONS §33).

---

# Part 6: `seqrec-eval final --device cuda:0`

## Step 6.1: the CLI branch, and added seeds

**code:**
[cli.py:276-312](src/seqrec_eval/cli.py#L276-L312) (the `final` path); added seeds at
[cli.py:282-289](src/seqrec_eval/cli.py#L282-L289) → `add_final_seeds` ([runner.py:163-165](src/seqrec_eval/runner.py#L163-L165)),
with `final_seeds` ([runner.py:137-139](src/seqrec_eval/runner.py#L137-L139)) and `record_added_seeds`
([runner.py:142-160](src/seqrec_eval/runner.py#L142-L160)).

**explanation:**
The same loop as `search`, with three differences.

- **Added seeds** first, with `--add-seeds 3 4`: for each selected dataset the seeds not already there are
  recorded in `runs/<dataset>/added_seeds.json` (`added_seeds_path`, [runner.py:122-128](src/seqrec_eval/runner.py#L122-L128)),
  with the time and host. From then on the dataset's seeds (`final_seeds`) are the protocol's followed by the
  added ones, for every command and report: `plan`, `final`, `status`, `report`, `repeat-strata`. Only the
  first seed is in any fingerprint (Step 0.3.5), so an added seed's finals sit beside those already made
  instead of starting them over. Asking again for a seed that is there changes nothing, so the command can be
  rerun or started once per GPU. The record is keyed by the dataset, not a fingerprint, so the decision holds
  under a changed protocol too. A duplicated or negative seed is refused.
- The split is `final_split(load_split(...))` ([cli.py:291-293](src/seqrec_eval/cli.py#L291-L293)):
  trained on train + validation under refit (Step 3.4).
- The specs come from `plan_finals` ([cli.py:298-305](src/seqrec_eval/cli.py#L298-L305)); if a model is not
  ready (search unfinished, open failures), the reason is logged, counted as `not-ready` (exit 3), and the
  next model runs.

## Step 6.2: `summarize_trials`: what the search produced

**code:**
`summarize_trials` at [runner.py:211-246](src/seqrec_eval/runner.py#L211-L246); `TrialSummary` at
[runner.py:182-199](src/seqrec_eval/runner.py#L182-L199).

**explanation:**
Per planned trial, from its folder: `failed.json` without `done.json` → failed (with the error); no
`done.json` → still open; `status: "skipped"` → skipped (with the reason); otherwise its validation primary
metric is read, and a non-finite value counts as a failure, marked as one a rerun would repeat
([runner.py:236-242](src/seqrec_eval/runner.py#L236-L242), review H07). The best is the highest value; strict
`>` in trial order, so ties go to the lowest trial. Any previously accepted failures are read from
`accepted_failures.json` ([runner.py:220-222](src/seqrec_eval/runner.py#L220-L222)); `unaccepted` is the
failures not in it.

## Step 6.3: `plan_finals`: selecting the configuration

**code:**
`plan_finals` at [runner.py:249-301](src/seqrec_eval/runner.py#L249-L301).

**explanation:**
1. Every trial skipped (catalogue too large) → no finals, an empty list ([runner.py:263-264](src/seqrec_eval/runner.py#L263-L264)).
2. Open failures → refuse ([runner.py:265-277](src/seqrec_eval/runner.py#L265-L277)): selecting around a
   failed trial would give the model a smaller, silently different search. Fix and rerun
   (`search --retry-failed`); for a non-finite score the message says a rerun would repeat it. For a failure
   that cannot be fixed (out of memory at a corner of the space), `final --accept-failed` writes
   `accepted_failures.json` (the trials, errors, time and host; [runner.py:278-281](src/seqrec_eval/runner.py#L278-L281))
   so the report lists them.
3. Fewer trials finished than planned → refuse unless `--allow-incomplete`, because selecting early breaks
   the equal budget across models ([runner.py:282-288](src/seqrec_eval/runner.py#L282-L288)). If it is
   allowed, the early selection is recorded in `incomplete_selection.json` (finished/planned, the selected
   trial, time, host; [runner.py:292-297](src/seqrec_eval/runner.py#L292-L297)) so the report says the budget
   was smaller (review B3); if the finished search later selects another trial, the finals made now are
   refused as stale (Step 5.3).
4. No successful trial → refuse ([runner.py:289-290](src/seqrec_eval/runner.py#L289-L290)).
5. Otherwise one `RunSpec` per seed of the dataset (`final_seeds`, so added seeds included): `kind = "final"`,
   `index = seed`, `seed`, the best trial's `params` and fingerprint, `source_trial` = the best trial's number
   ([runner.py:298-301](src/seqrec_eval/runner.py#L298-L301)). Folder: `runs/<dataset>/<model>/<fp>/final-seed<seed>`.

## Step 6.4: `_execute` for a final run

**code:**
[runner.py:402-482](src/seqrec_eval/runner.py#L402-L482), the parts a final takes.

**explanation:**
The same steps as a trial (Step 5.4), except:

- `max_items` is judged on `len(train_item_ids)`, which under refit is the validation catalogue.
- The trainer is seeded with the final's own seed and fitted on the swapped training views, i.e. `x_refit`
  or `x_refit_sequences` with the validation catalogue.
- **No validation score** under refit: `split.trained_on` is `"train+val"`, so the validation block is
  skipped ([runner.py:443](src/seqrec_eval/runner.py#L443)). The report takes the validation value from the
  search. Without refit a final scores validation too.
- **Test** ([runner.py:449-453](src/seqrec_eval/runner.py#L449-L453)): `evaluate_phase(..., "test",
  rows=split.test_rows)`, which on the full data is `None`: every test user that passes the two filters of
  Step 3.9.2. Saved as `test.json` + `test.npz`.
- **Diagnostics** of a stage-1 final only ([runner.py:454-466](src/seqrec_eval/runner.py#L454-L466)): the
  other target definition as `test_window.*`, and, for datasets with `new_item_diagnostic = true`
  (Music4All, Yambda, OTTO), the targets not already in the history with `exclude_seen = True` as
  `test_new.*`.
- **The model** is saved as `model.zip` ([runner.py:467-472](src/seqrec_eval/runner.py#L467-L472)). Latency
  and every ablation reference reload it. A failed save is recorded (`model_saved: false` and the error) but
  does not fail the run.

**Output:** `final-seedS/` with `spec.json`, `test.*`, `test_window.*`, `test_new.*` (some datasets),
`model.zip`, `done.json`.

---

# Part 7: `seqrec-eval status`

## Step 7.1

**code:**
[cli.py:314-316](src/seqrec_eval/cli.py#L314-L316) → `status_table` at
[report.py:167-186](src/seqrec_eval/report.py#L167-L186).

**explanation:**
One table row per dataset and model: whether the split is prepared, trials finished (done + skipped) of
planned, failed trials, runs whose lock another process holds right now (`held_by_other`,
[runner.py:355-368](src/seqrec_eval/runner.py#L355-L368), which probes the lock with a shared `flock`),
finals done of the dataset's seeds (added ones included) or "skipped", and the best trial with its validation
`ndcg@10`. It reads the run folders only.

---

# Part 8: `seqrec-eval report`

## Step 8.1: the CLI branch and the header

**code:**
[cli.py:318-330](src/seqrec_eval/cli.py#L318-L330) → `build_report` at
[report.py:453-480](src/seqrec_eval/report.py#L453-L480).

**explanation:**
`build_report` writes one section per dataset from `dataset_report`, a last section across datasets when
there is more than one ([report.py:460-461](src/seqrec_eval/report.py#L460-L461)), and a header
([report.py:462-471](src/seqrec_eval/report.py#L462-L471)): the protocol's full path and the SHA-256 of the
file, so a report of a quick local run (`work-local/<dataset>/protocol.toml`) cannot be mistaken for one of
the real protocol; its version; the primary metric; and whether finals were refitted. It also collects a CSV
of every final run's metrics. The CLI appends the CPU latency table if any `latency.json` exists
(`latency_table`, Step 9.1; [cli.py:320-322](src/seqrec_eval/cli.py#L320-L322)), and writes
`reports/stage1.md` and `reports/final_metrics.csv`.

## Step 8.2: one dataset's section

**code:**
`dataset_report` at [report.py:265-421](src/seqrec_eval/report.py#L265-L421).

**explanation:**
1. **The split** (`_split_section`, [report.py:193-218](src/seqrec_eval/report.py#L193-L218)): the resolved
   build settings and, per phase, rows, catalogue, target pairs, repeat-target share and history p50/p90,
   all from `split_info.json`. A note follows if the dataset has added seeds ([report.py:271-274](src/seqrec_eval/report.py#L271-L274)).
2. **Per model** ([report.py:276-336](src/seqrec_eval/report.py#L276-L336)): skipped models are noted; for
   the others the finished finals are loaded (`load_final_evaluations`, [runner.py:616-630](src/seqrec_eval/runner.py#L616-L630)),
   leaving out any final made under another selection (`stale_final_seeds`,
   [runner.py:596-613](src/seqrec_eval/runner.py#L596-L613); review N33). The table shows trials done, the
   selected trial, its validation value, the number of seeds (or done/total), and each test metric as
   mean ± sd over seeds. Notes flag, per model: open trial failures (⛔), accepted failures with their
   reasons, a selection made from an unfinished search, a selected trial or final whose training loss was
   still falling at its last epoch (⚠, below), stale finals (⛔), failed final seeds (⛔), and seeds that have
   not run yet. The seeds' per-user values are averaged into one result per model (`mean_over_seeds`,
   [report.py:84-101](src/seqrec_eval/report.py#L84-L101)), which requires every seed to have scored the same
   users on the same targets.
3. **Still improving** (`still_improving`, [report.py:67-81](src/seqrec_eval/report.py#L67-L81), and
   `_still_improving_runs`, [report.py:251-262](src/seqrec_eval/report.py#L251-L262)): from the `history`
   each `done.json` records, a run whose training loss fell by more than `STILL_IMPROVING` = 1%
   ([report.py:64](src/seqrec_eval/report.py#L64)) over the last tenth of its epochs (at least one) was stopped
   while still learning. Every epoch is paid in full (no early stopping) and SASRec's epochs are capped at
   100 for the one-week budget, so the report says where the cap may hold a model back. Models without a
   loss per epoch (EASE, popularity) are never flagged.
4. **Who was scored** ([report.py:340-345](src/seqrec_eval/report.py#L340-L345)): how many test users the
   next-item metrics cover, and how many were left out because no model can recommend their next item.
5. **Code builds** ([report.py:348](src/seqrec_eval/report.py#L348); `code_builds` and `_builds_note` at
   [report.py:221-248](src/seqrec_eval/report.py#L221-L248)): every `done.json` records its code; one build
   is stated, several are flagged ⚠.
6. **New-item diagnostic** ([report.py:350-352](src/seqrec_eval/report.py#L350-L352)), where it was run.
7. **Against the reference model** ([report.py:354-383](src/seqrec_eval/report.py#L354-L383)): every model
   is compared with `--reference` (default `elsa`) on test `ndcg@10` by the seed-aware t-test (Step 8.3),
   given each model's per-seed results. The table shows the difference, the relative difference, the 95% t
   interval, the users' and the seeds' standard errors, the degrees of freedom, the Holm-adjusted p, the
   users-only p (Holm-adjusted too), and "significant" read from the adjusted p. The Holm family is one
   dataset, so a dataset's conclusions do not pay for the number of datasets in the study. A model with one
   seed has no seed spread to count, and the text says so.
8. **The other target definition** ([report.py:385-396](src/seqrec_eval/report.py#L385-L396)): each model's
   primary value beside its `test_window` value.
9. **Against the floor** ([report.py:398-420](src/seqrec_eval/report.py#L398-L420)): the floor (Part 4) is
   read from the analysis; it waits for every baseline, and the report names the missing ones meanwhile.
   Every model is compared with it by the same test, the floor contributing no seed term (a deterministic
   baseline scored once), in a second Holm family. "Beats the floor" means significant *and* positive.

## Step 8.3: the seed-aware t-test

**code:**
[seedstats.py](src/seqrec_eval/seedstats.py): `compare` at [seedstats.py:178-209](src/seqrec_eval/seedstats.py#L178-L209),
`variance` at [seedstats.py:95-114](src/seqrec_eval/seedstats.py#L95-L114), `holm` at
[seedstats.py:54-63](src/seqrec_eval/seedstats.py#L54-L63); the formulas in the module docstring
([seedstats.py:1-39](src/seqrec_eval/seedstats.py#L1-L39)).

**explanation:**
Every p-value and interval in the stage-1 and ablation reports comes from this one test (review H23, final
review A1, A2). Each model is trained under k seeds, so a difference has two sources of uncertainty: which
users are in the test set, and which training runs were drawn. `compare(candidate, reference, metric)` takes
each side's per-seed evaluations:

1. **The estimate** D is the mean of the per-user differences d of the seed-averaged values (every seed of a
   model must have scored the same users, `_values`, [seedstats.py:159-167](src/seqrec_eval/seedstats.py#L159-L167);
   both sides the same users and the same targets, `_same_targets`, [seedstats.py:170-175](src/seqrec_eval/seedstats.py#L170-L175)).
   Seeds that scored different users (a random catalogue) are passed in already pooled (`pooled=`).
2. **The variance** has a users' part, var(d)/n, and a seeds' part. Unpaired (stage 1, where seed s of one
   model has nothing to do with seed s of another), the seeds' part is each side's variance of its k seed
   means over k, added. **Paired** (every ablation, where seed s of a level is the full data's seed s
   retrained or rescored, and two models at a level of a random sweep share subsample s), it is the variance
   of the per-seed differences over k, so the shared part cancels.
3. **Degrees of freedom** by Welch–Satterthwaite over the two (or three) parts. When the seeds agree it is the
   paired t-test over users; when they do not, df falls towards k − 1 and a difference has to be clearly
   larger than the seed spread to count.
4. **p** is two-sided from t = D/SE against Student's t with that df (`two_sided_p`,
   [seedstats.py:124-127](src/seqrec_eval/seedstats.py#L124-L127)); the **interval** is D ± t(0.975, df)·SE,
   the test's own. Beside it, `p_users` is the same test without the seed term (the paired t-test over
   users), so the effect of counting the seeds is visible.
5. **Holm** adjusts a family of p-values step-down.

The knee's one-sided form, "the loss is smaller than δ", is `noninferiority`
([seedstats.py:130-136](src/seqrec_eval/seedstats.py#L130-L136)): t = (D + δ)/SE against the upper tail, and
its power for a level that loses nothing is P(T > t(1−α, df) − δ/SE). A two-level bootstrap was used until
2026-10-01; with k = 3 it was far too narrow (2.6 times narrower than the t interval in the design example),
so it was dropped.

## Step 8.4: across datasets

**code:**
`_across_datasets` at [report.py:424-450](src/seqrec_eval/report.py#L424-L450).

**explanation:**
Nothing is pooled across datasets (H51): each dataset answers on its own. The last table shows, per model and
dataset, the sign of the difference against the reference with * where it is significant within that
dataset, and counts per model the datasets where it is significantly better, not significantly different,
and significantly worse. A claim about all datasets needs the same answer on most of them.

---

# Part 9: `seqrec-eval latency`

## Step 9.1

**code:**
[cli.py:421-435](src/seqrec_eval/cli.py#L421-L435) → `benchmark` at
[latency.py:60-129](src/seqrec_eval/latency.py#L60-L129); `parse_cores` at
[latency.py:36-47](src/seqrec_eval/latency.py#L36-L47); `latency_table` at
[latency.py:132-158](src/seqrec_eval/latency.py#L132-L158).

**explanation:**
The requirement is model inference latency on CPU, 100 ms at P95, characterised against history length.
Per dataset the tested split is used (`final_split`, [cli.py:424](src/seqrec_eval/cli.py#L424)), and per
model:

1. Settings from `[latency]`: history bins, requests per bin (200), warm-up requests (20)
   ([latency.py:63-66](src/seqrec_eval/latency.py#L63-L66)).
2. The first protocol seed's `final-seed<seed>/model.zip`; a missing model, or a final made under another
   selection than the current one (review N33), raises `FileNotFoundError`, which the CLI logs before going on
   to the next model ([latency.py:68-75](src/seqrec_eval/latency.py#L68-L75), [cli.py:428-430](src/seqrec_eval/cli.py#L428-L430)).
3. Optionally pin the process to `--cores`, set torch threads, and reload the model on CPU
   ([latency.py:77-81](src/seqrec_eval/latency.py#L77-L81)).
4. Build the adapter and the test source once, untimed (`phase_inputs`), wrapped in `ExcludeSeenPolicy`
   without a seen matrix, so the model applies the dataset's `exclude_seen` itself, as it would when serving
   ([latency.py:83-84](src/seqrec_eval/latency.py#L83-L84)).
5. Draw up to `requests_per_bin` test users per history-length bin and interleave the bins in random order,
   so drift over the run hits every bin alike ([latency.py:89-98](src/seqrec_eval/latency.py#L89-L98)).
6. Run the warm-up requests, then time only the model call for each single-user request against the full
   test catalogue, top 20 ([latency.py:103-112](src/seqrec_eval/latency.py#L103-L112)).
7. Write `latency.json` beside the model: overall and per-bin n, mean, P50, P95, P99, max, plus catalogue
   size, threads, cores, host, CPU, torch version, the one-minute load average before and after (how busy the
   machine was), and which trial's configuration was timed ([latency.py:114-128](src/seqrec_eval/latency.py#L114-L128)).

`latency_table`, which `report` appends, shows per model P50, P95, P99 and the **worst bin's P95**, taken only
over bins with at least `MIN_BIN_REQUESTS` = 50 requests ([latency.py:57](src/seqrec_eval/latency.py#L57),
[latency.py:145-151](src/seqrec_eval/latency.py#L145-L151)), since a P95 of a handful of requests is noise.
A measurement made on a stale final is not shown, and the table says so.

---

# Part 10: `seqrec-eval ablate --device cuda:0`

An ablation sweep varies one data characteristic at a series of levels and holds everything else fixed.
Every model keeps its stage-1 configuration at every level; nothing is searched again, so a difference
between levels comes from the data, not from tuning ([ablations.py:1-66](src/seqrec_eval/ablations.py#L1-L66)).
The sweeps in the protocol ([protocol.toml:310-373](protocol.toml#L310-L373)), cut on 2026-10-05 to fit
the one-week budget (13 retraining levels, was 28):

| sweep | transform | levels | scope | seeds | knee |
|---|---|---|---|---|---|
| `history_length` | history_length | 5, 20, 100, 500 | all (refit per level) | 0 | yes |
| `history_length_inference` | history_length | 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 | inference (rescore only) | all | yes |
| `density` | density | 0.25, 0.5 | all | 0 | yes |
| `shuffle` | shuffle | block 2, 10, all | all | 0 | no |
| `catalogue_top` | catalogue (top) | 0.25, 0.5 | all | 0 | no |
| `repeat_removal` | repeat_removal | 0.5, 1.0 | all | 0 | yes |

The stratified catalogue sweep is no longer in the protocol; its code stays, and
`options = { strategy = "stratified", strata = 10 }` brings it back.

## Step 10.1: the CLI branch, and added seeds

**code:**
[cli.py:358-393](src/seqrec_eval/cli.py#L358-L393).

**explanation:**
Models are checked against the registry first ([cli.py:361-362](src/seqrec_eval/cli.py#L361-L362)).

**Added seeds** (`--add-seeds`). Every selected sweep on every selected dataset is checked before any is
recorded, so a refusal leaves nothing half-added ([cli.py:364-371](src/seqrec_eval/cli.py#L364-L371)): each
seed must already be a stage-1 seed of the dataset (`check_stage1_seeds`,
[ablations.py:698-709](src/seqrec_eval/ablations.py#L698-L709)), because a seed's reference is that seed's
stage-1 final model. Then, per dataset and sweep, `check_added_seeds`
([ablations.py:887-905](src/seqrec_eval/ablations.py#L887-L905)) refuses a seed whose subsamples would leave
any of the sweep's fixed test users ineligible (Step 10.4), and `add_sweep_seeds`
([ablations.py:712-716](src/seqrec_eval/ablations.py#L712-L716)) records the new ones in
`ablations/<sweep>/<dataset>/added_seeds.json` (`sweep_seeds_path`, [ablations.py:687-689](src/seqrec_eval/ablations.py#L687-L689);
[cli.py:378-385](src/seqrec_eval/cli.py#L378-L385)). The sweep's seeds on that dataset (`sweep_seeds`,
[ablations.py:692-695](src/seqrec_eval/ablations.py#L692-L695)) are then the sweep's own followed by the
added ones. For a stochastic sweep each added seed is a new subsample, so the CLI reminds you to run
`analyse --sweep` too, for the floor and controls on it ([cli.py:386-388](src/seqrec_eval/cli.py#L386-L388)).

**The sweeps.** Per dataset, the sweeps that cover it are collected
([cli.py:373-375](src/seqrec_eval/cli.py#L373-L375)); a dataset with none is skipped without loading. The
split is loaded, checked and refitted once, `final_split(load_split(...))` ([cli.py:376](src/seqrec_eval/cli.py#L376)),
because every condition is a transform of the data the stage-1 finals were fitted on. Then `_ablate` runs
each sweep with the models it covers ([cli.py:389-390](src/seqrec_eval/cli.py#L389-L390)). The exit code is
counted as for `search` and `final` ([cli.py:393](src/seqrec_eval/cli.py#L393)).

## Step 10.2: `_ablate`: one sweep on one dataset

**code:**
`_ablate` at [cli.py:439-484](src/seqrec_eval/cli.py#L439-L484).

**explanation:**
1. **Plans** ([cli.py:446-457](src/seqrec_eval/cli.py#L446-L457)): `plan_ablation` per model (Step 10.3).
   A model whose stage-1 search or finals are not ready is logged as `not-ready`; a model skipped in stage 1
   has an empty plan and drops out.
2. **Users** ([cli.py:458-462](src/seqrec_eval/cli.py#L458-L462)): for the catalogue sweeps each condition
   has its own users (logged); for every other sweep the fixed test users are computed now
   (`fixed_test_rows`, Step 10.4) and their number logged.
3. **`run(condition, name)`** ([cli.py:464-470](src/seqrec_eval/cli.py#L464-L470)): record the condition's
   characteristics (Step 10.7), then `execute` every model's specs for that condition name (Step 10.8).
4. **`pending(name)`** ([cli.py:472-475](src/seqrec_eval/cli.py#L472-L475)): the condition's
   characteristics file is missing, or some run has no `done.json`. A finished level is not even built,
   so resuming a finished sweep costs no transform.
5. **Order** ([cli.py:477-484](src/seqrec_eval/cli.py#L477-L484)): the reference (`full`) first, always (it
   is the unmodified split, so it costs nothing to build; its finished runs return `cached`), then every level
   that is pending, and within a level every data seed (one per seed of the sweep for a stochastic transform,
   a single `None` otherwise, `data_seeds`, [ablations.py:719-723](src/seqrec_eval/ablations.py#L719-L723)).
   The condition name is the level label, plus `/seed<S>` for a stochastic one (`condition_name`,
   [ablations.py:726-728](src/seqrec_eval/ablations.py#L726-L728)).

## Step 10.3: `plan_ablation`: which runs each condition has

**code:**
`plan_ablation` at [ablations.py:1129-1178](src/seqrec_eval/ablations.py#L1129-L1178); fingerprints at
[ablations.py:663-684](src/seqrec_eval/ablations.py#L663-L684).

**explanation:**
It starts from the stage-1 finals, `plan_finals(...)` ([ablations.py:1140](src/seqrec_eval/ablations.py#L1140)),
so an ablation refuses exactly when a final would (Step 6.3). Of those it keeps the sweep's seeds
([ablations.py:1143-1150](src/seqrec_eval/ablations.py#L1143-L1150)): stage 1 may have more seeds than the
sweep (a retraining sweep runs under seed 0 only), and a sweep seed that stage 1 does not have is refused.
Each spec is a copy of a stage-1 final spec with another `kind`, fingerprint and condition:

- **Fingerprints.** `condition_fingerprint` ([ablations.py:663-675](src/seqrec_eval/ablations.py#L663-L675))
  covers the dataset fingerprint, the evaluation key, the sweep's `transform`, `levels`, `options` and
  `scope` (`_RESULT_KEYS`, [ablations.py:106](src/seqrec_eval/ablations.py#L106); an omitted scope counts as
  `"all"`; `datasets`, `models`, `seeds` and the report-only keys are left out), and the transform's version.
  It names the sweep's folder on a dataset, `ablations/<sweep>/<dataset>/<cfp[:12]>` (`ablation_root`).
  `ablation_fingerprint` ([ablations.py:678-680](src/seqrec_eval/ablations.py#L678-L680)) adds the model's
  stage-1 run fingerprint and names its runs, `…/runs/<model>/<afp[:12]>/`.
- **Reference** ([ablations.py:1159-1165](src/seqrec_eval/ablations.py#L1159-L1165)): one spec per seed,
  `kind = "reference"`, folder `…/full/final-seed<S>`, and a `checkpoint` pointing at the stage-1
  `runs/<dataset>/<model>/<fp>/final-seed<S>/model.zip`. It reloads the stage-1 model, so the reference is
  exactly the model stage 1 reported.
- **Levels** ([ablations.py:1166-1177](src/seqrec_eval/ablations.py#L1166-L1177)): for a deterministic
  transform every seed's final runs at every level; for a stochastic one, condition `<level>/seed<S>` runs
  only seed S's final, so each seed has its own subsample and the seed spread holds both the draw and the
  training. With scope `"all"` the kind is `"final"` (the stage-1 params refitted with the stage-1 seed on the
  condition's training data); with scope `"inference"` it is `"rescore"` and carries the checkpoint (the
  stage-1 model reloaded and scored on transformed inputs). Folder: `…/<level label>/final-seed<S>`.

## Step 10.4: which test users each condition is scored on

**code:**
`fixed_test_rows` at [ablations.py:827-867](src/seqrec_eval/ablations.py#L827-L867) with
`_eligible_test_rows` ([ablations.py:731-742](src/seqrec_eval/ablations.py#L731-L742)) and
`_check_subsamples` ([ablations.py:870-884](src/seqrec_eval/ablations.py#L870-L884)); `own_test_rows` at
[ablations.py:785-797](src/seqrec_eval/ablations.py#L785-L797); `per_condition_users` at
[ablations.py:800-802](src/seqrec_eval/ablations.py#L800-L802); `build_reference` at
[ablations.py:805-812](src/seqrec_eval/ablations.py#L805-L812); `build_condition` at
[ablations.py:815-824](src/seqrec_eval/ablations.py#L815-L824).

**explanation:**
Levels are only comparable if they are scored on the same users. Two rules:

- **One fixed set** (every sweep whose transform keeps the targets). A test row is eligible if its history is
  not empty and it passes the two filters of `evaluate_phase` (Step 3.9.2): a next target a model can reach,
  and under `exclude_seen` a full list of `k` unseen items. `fixed_test_rows` takes the rows eligible on the
  full data **and** at every level and data seed, which means building every condition once
  ([ablations.py:852-859](src/seqrec_eval/ablations.py#L852-L859)). It saves them as `test_rows.npy` in the
  sweep's folder (durably, [ablations.py:862-866](src/seqrec_eval/ablations.py#L862-L866)) and records in
  `test_rows.json` which subsamples they were checked on; every later call reads them. An empty set is an
  error. A subsample added later (`ablate --add-seeds` on a stochastic sweep) is **checked** against the
  cached users rather than intersected into them ([ablations.py:842-851](src/seqrec_eval/ablations.py#L842-L851)):
  every run already made was scored on those users, so they must all stay eligible, or the new runs could not
  be paired with the old. Every transform so far keeps them so by construction (it drops or reorders events
  inside a fixed item space, and a non-empty history keeps at least one event); one that did not would be
  refused here rather than scored on fewer users.
- **Its own users** (the catalogue sweep, whose transform removes targets: `changes_targets = True`). One
  set cannot survive every random catalogue: a user whose next target is a known item survives all levels
  only by chance, so the set would shrink to users whose targets are all unseen, who score 0 for every model
  (review H01b). So each condition is scored on its own rows: a reachable next target still in *its*
  catalogue, a history item left, and under `exclude_seen` a full list (`own_test_rows`). Models are then
  compared within a condition, not across levels.

`build_reference` wraps the unmodified split as condition `full` with those rows
(`reference_condition`, [ablations.py:780-782](src/seqrec_eval/ablations.py#L780-L782)).
`build_condition` applies the transform (Step 10.5) and attaches the rows: the fixed set, or the
condition's own.

## Step 10.5: `apply_condition`: transforming the split

**code:**
`apply_condition` at [ablations.py:753-777](src/seqrec_eval/ablations.py#L753-L777).

**explanation:**
1. The transform is looked up; a stochastic transform needs a data seed and a deterministic one refuses
   it ([ablations.py:758-761](src/seqrec_eval/ablations.py#L758-L761)).
2. **The random stream** ([ablations.py:762-765](src/seqrec_eval/ablations.py#L762-L765)):
   `_stream(search_seed, "ablation", sweep, dataset, <level or "nested">, data_seed)`. A nested transform
   (the catalogue) leaves the level out, so every level of one seed shares one random draw and a smaller
   catalogue is part of every larger one.
3. **The phases it may edit** ([ablations.py:766](src/seqrec_eval/ablations.py#L766)): scope `"all"` →
   train, val and test; scope `"inference"` → val and test only, so the training data stays untouched.
4. **Apply** ([ablations.py:767-768](src/seqrec_eval/ablations.py#L767-L768)): the transform gets the data
   dict, the level, the stream, the sweep's options, the dataset's event value (`set_all_values_to`) and the
   phases, and returns a new dict. The original split is never modified.
5. **What the user has seen** ([ablations.py:769-775](src/seqrec_eval/ablations.py#L769-L775)): if a val or
   test source matrix was rebuilt and the item space is unchanged, the original source matrix is stored as
   `{phase}_seen_matrix`. Scoring then excludes the user's whole original history, not only the shortened
   input (Step 3.9.5): a film rated five years ago is still not a recommendation.
6. The result is the same `Split` with the new data, the test rows and `condition = {sweep, level, label,
   data_seed}` ([ablations.py:776-777](src/seqrec_eval/ablations.py#L776-L777)). It keeps `trained_on =
   "train+val"`.

## Step 10.6: the transform registry and `keep_events`

**code:**
`Transform` at [ablations.py:127-152](src/seqrec_eval/ablations.py#L127-L152); `register` at
[ablations.py:158-174](src/seqrec_eval/ablations.py#L158-L174); `keep_events` at
[ablations.py:309-351](src/seqrec_eval/ablations.py#L309-L351), with `count_matrix`
([ablations.py:296-306](src/seqrec_eval/ablations.py#L296-L306)), `_subset`
([ablations.py:253-257](src/seqrec_eval/ablations.py#L253-L257)) and `_carry_times`
([ablations.py:260-273](src/seqrec_eval/ablations.py#L260-L273)).

**explanation:**
A transform is a function registered with a declaration of what it does:

| field | meaning |
|---|---|
| `version` | bumped when its behaviour changes; enters the condition fingerprint, so old conditions are not reused |
| `stochastic(options)` | whether it draws a subsample (then one per data seed) |
| `target` | the characteristic it is meant to move, or `None` (shuffle: none should move) |
| `expected(options)` | what else it moves by construction, as a characteristic or as `"<part> <characteristic>"`; the manipulation check does not flag these |
| `validate` | checks levels and options (called by `check_ablation`, Step 0.4) |
| `scopes` | which scopes it supports |
| `changes_targets` | whether test targets change between levels (→ per-condition users, no level-vs-full test) |
| `position(level)` | the level's place on the knee's axis, larger = closer to the full data; `None` = no knee |
| `direction` | which way the target must move (−1 = down); the other way is flagged as a bug |
| `nested` | whether the levels of one seed share one draw |

Transforms that only drop events build a boolean mask per phase and hand it to `keep_events`, which rebuilds
every view from the kept events exactly as the builder would have:

- **train:** the training window is the train-stage source followed by its target, so an event is "source"
  if its position is before the source's length ([ablations.py:329](src/seqrec_eval/ablations.py#L329)).
  Both matrices are recounted from the kept events on each side (`count_matrix`: each event's value summed
  per pair), and `x_train` is again their maximum. The kept events' times go with them.
- **val / test:** the source sequences are subset and the source matrix recounted.
- The masks are stored under `_kept` for the manipulation check's `span_kept` (Step 10.7).

### Step 10.6.1: `history_length` (version 3)

**code:**
[ablations.py:378-412](src/seqrec_eval/ablations.py#L378-L412).

**explanation:**
Level n is a context of n items. Every training history keeps its last **n + 1** events, every input
history (val, test) its last **n** ([ablations.py:408-412](src/seqrec_eval/ablations.py#L408-L412)). A
training history of n + 1 events teaches contexts of n items (its last event is only ever a target), so a
model is trained and served with the same context, and level 1 still has something to learn (one item → the
next). Deterministic. Target `history_length`; declared to move `density`, `repeat_rate`, `catalogue` and
`popularity_gini` too ([ablations.py:391](src/seqrec_eval/ablations.py#L391)): a history of n events holds
at most n items, so at level 1 the test inputs of 2,942 users hold at most 2,942 of ML-20M's items. Position
= n. The sequential models also keep their stage-1 `max_history_length`, so at levels at or above it their
input at inference is the same as with full data (the report says so, Step 11.2). With scope `"inference"`
(the `history_length_inference` sweep) only the val and test inputs are cut, and the stage-1 models are
rescored: how much history a trained model needs at serving time.

### Step 10.6.2: `density` (version 2)

**code:**
[ablations.py:419-461](src/seqrec_eval/ablations.py#L419-L461).

**explanation:**
Each history keeps a random fraction p of its events: `max(1, round(p × length))` of them, chosen uniformly
among its positions by ranking random keys within the row, in their original order
([ablations.py:447-450](src/seqrec_eval/ablations.py#L447-L450)). A thinned history still reaches as far
back as the original, unlike truncation. With `keep_catalogue` (default on), a training item that would lose
all its events gets one back, chosen at random, so the training catalogue does not shrink
([ablations.py:451-459](src/seqrec_eval/ablations.py#L451-L459)). Stochastic. Target `density`; declared to
move `history_length`, `repeat_rate`, and the **test** catalogue (thinned test inputs hold fewer distinct
items), or the whole catalogue without `keep_catalogue` ([ablations.py:432-433](src/seqrec_eval/ablations.py#L432-L433));
position = p.

### Step 10.6.3: `repeat_removal` (version 1)

**code:**
[ablations.py:468-494](src/seqrec_eval/ablations.py#L468-L494).

**explanation:**
A repeat event is one whose item already occurred earlier in the same history (`_first_occurrence`,
[ablations.py:288-293](src/seqrec_eval/ablations.py#L288-L293)). Each repeat is removed with probability q;
first occurrences are never removed, so which items a history holds (and so catalogue and density) does not
change, and every remaining repeat is still a repeat. Stochastic. Target `repeat_rate`; declared to move
`history_length` and `popularity_gini` (the event counts change, −13% Gini at q = 1 on data with 40%
repeats, [ablations.py:480](src/seqrec_eval/ablations.py#L480)); position = 1 − q, so less removal is closer
to the full data and the knee is the most removal that still costs less than δ. It removes nothing where
users do not repeat (ML-20M, Amazon).

### Step 10.6.4: `shuffle` (version 1)

**code:**
[ablations.py:501-530](src/seqrec_eval/ablations.py#L501-L530).

**explanation:**
Events are reordered within each history: with `"all"` the whole history, with block size b only within
consecutive blocks of b events, which destroys order at short range and keeps it at long range
([ablations.py:524-525](src/seqrec_eval/ablations.py#L524-L525)). Each event keeps its own time. It does not
use `keep_events`: the matrices stay exactly as they were, because a matrix row is a set. So the matrix
models, and the time-decayed baseline, are controls that should score as with the full data. Stochastic.
Target `None`: none of the five characteristics should move. No position, so no knee.

### Step 10.6.5: `catalogue` (version 2): top and stratified

**code:**
[ablations.py:537-656](src/seqrec_eval/ablations.py#L537-L656); `_stratified_order` at
[ablations.py:550-566](src/seqrec_eval/ablations.py#L550-L566).

**explanation:**
Reduces the training catalogue to k items: a fraction of it, or an item count
([ablations.py:604](src/seqrec_eval/ablations.py#L604)).

- `strategy = "top"` (the protocol's `catalogue_top`): the k items with the most training events.
  Deterministic.
- `strategy = "stratified"` (not in the protocol now): items ranked by popularity and cut into `strata` equal
  groups; within a group of size s each item gets a random slot r and a place (r + U)/s; sorting every item by
  place interleaves the groups, so any prefix holds each stratum in proportion. Taking a prefix of this one
  order at every level makes the levels nested. Stochastic, one order per seed.

A removed item leaves **every** phase ([ablations.py:611-632](src/seqrec_eval/ablations.py#L611-L632)): its
events leave the training data and the input histories (the remaining items are re-addressed in the smaller
item space), its targets leave the val and test target matrices and the next-item targets, and it leaves the
item space, so no model can recommend it. Items first seen after training stay, as new items no model can
recommend, exactly as in the full data. Training users left with no events are dropped from the training
views ([ablations.py:634-644](src/seqrec_eval/ablations.py#L634-L644)); the catalogue partitions are updated
and the stale list-shaped index fields cleared ([ablations.py:646-655](src/seqrec_eval/ablations.py#L646-L655)).
Target `catalogue`; declared to move `history_length` and `density`, and for `"top"` also `popularity_gini`
(keeping only the popular items flattens the distribution; stratified is meant not to,
[ablations.py:574-578](src/seqrec_eval/ablations.py#L574-L578)). Scope `"all"` only; `changes_targets`, so
per-condition users; no knee. The item space shrinks, so no seen matrix is stored (Step 10.5): the source
matrix of the condition is the user's history in the smaller catalogue, and that is what is excluded.

## Step 10.7: the condition's characteristics

**code:**
`record_characteristics` at [ablations.py:1084-1094](src/seqrec_eval/ablations.py#L1084-L1094) →
`characteristics(split, original, …)` (Step 3.5) with `_edits` at
[ablations.py:949-983](src/seqrec_eval/ablations.py#L949-L983).

**explanation:**
Before a condition's runs, its profile is written once to `…/conditions/<name>.json` (`full.json`,
`20.json`, `0.5/seed0.json`, …) as `{sweep, level, label, data_seed, characteristics}`. The original is the
refitted full split, so each part also gets `_edits`:

- `rows_changed`: the share of rows whose history differs from the original in any way (length, items
  compared by id, or order);
- `span_kept`: over rows that keep an event, the mean share of the original history between the first and
  the last kept event. 1 when the kept events still reach back as far as the original (density), n/length
  after truncation to the last n. It needs the `_kept` masks, so it is `None` for shuffle.

Both are `None` for a part whose rows were dropped (the catalogue sweep's training part). The analysis step
writes the same file for the same condition (Step 3.12); whichever runs first writes it.

## Step 10.8: running a condition: `execute` on an ablation spec

**code:**
[cli.py:464-470](src/seqrec_eval/cli.py#L464-L470) → `execute` (Step 5.3) → `_execute` at
[runner.py:402-482](src/seqrec_eval/runner.py#L402-L482).

**explanation:**
The same machinery as stage 1: resumable, locked, deaths counted, failures recorded. What differs for an
ablation spec:

- `_check_condition` ([runner.py:381-390](src/seqrec_eval/runner.py#L381-L390)) requires the spec's
  sweep, label and data seed to match the split's condition, and `test_rows` to be set. A run cannot be
  scored on another condition's data.
- `reference` and `rescore` first need their stage-1 final finished and of the current selection
  (`stage1-failed`, `waiting-for-stage1`, `stale-selection`; Step 5.3), then reload its `model.zip` on the
  run's device (`registered.cls.load`, [runner.py:428-433](src/seqrec_eval/runner.py#L428-L433)); `final`
  refits the stage-1 params with the stage-1 seed on the condition's training data.
- No validation: nothing selects on it any more ([runner.py:443](src/seqrec_eval/runner.py#L443)).
- Test on `split.test_rows`, the condition's users ([runner.py:449-453](src/seqrec_eval/runner.py#L449-L453)),
  saved as `test.*`.
- No diagnostics and no saved model ([runner.py:454](src/seqrec_eval/runner.py#L454)).

**Output:** `ablations/<sweep>/<dataset>/added_seeds.json` (if seeds were added), and under
`ablations/<sweep>/<dataset>/<cfp>/`: `test_rows.npy` and `test_rows.json` (fixed-set sweeps),
`conditions/*.json`, `runs/<model>/<afp>/<condition>/final-seedS/{spec.json, test.json, test.npz, done.json}`,
and from `analyse` the `analysis/` subfolder.

---

# Part 11: `seqrec-eval ablation-report`

## Step 11.1: the CLI branch

**code:**
[cli.py:395-408](src/seqrec_eval/cli.py#L395-L408) → `build_ablation_report` at
[ablation_report.py:651-703](src/seqrec_eval/ablation_report.py#L651-L703).

**explanation:**
Per selected sweep it writes `reports/ablation-<sweep>.md`, the gap plot `ablation-<sweep>-gap.png` (if
matplotlib is installed and there is a gap), and up to three CSVs: `-metrics` (every run's metrics, per
seed), `-gap` (the tested and the descriptive gap rows, plus the floor rows as model `floor`), `-analysis`
(the baselines and controls at every condition). `--reference` (default `elsa`) is the **comparator**: the
non-sequential model the study's gap is taken against, fixed in advance.

## Step 11.2: the header

**code:**
[ablation_report.py:657-677](src/seqrec_eval/ablation_report.py#L657-L677); `_transform_notes` at
[ablation_report.py:706-718](src/seqrec_eval/ablation_report.py#L706-L718).

**explanation:**
Datasets and models are restricted to those the sweep covers ([ablation_report.py:659-660](src/seqrec_eval/ablation_report.py#L659-L660)).
The header states the transform and its version, the levels and options, the scope (refit per level, or
rescoring), the sweep's seeds (and stage 1's, when the sweep runs under fewer), whether each seed has its own
subsample, and that `full` is the stage-1 model itself. For a catalogue sweep it says the targets change with
the level. For `history_length` it notes which sequential models also keep a `max_history_length`.

## Step 11.3: one dataset: setup and the manipulation check

**code:**
`dataset_ablation` at [ablation_report.py:301-553](src/seqrec_eval/ablation_report.py#L301-L553); the
manipulation section `_characteristics_section` at [ablation_report.py:254-298](src/seqrec_eval/ablation_report.py#L254-L298)
with `manipulation_check` at [ablations.py:1097-1122](src/seqrec_eval/ablations.py#L1097-L1122).

**explanation:**
The report-only settings are read with their defaults ([ablation_report.py:303-315](src/seqrec_eval/ablation_report.py#L303-L315)):
`manipulation_tolerance` 0.05, `knee_margin` 0.10, `min_level_users` 1,000. The columns are `full` and the
level labels (seeds of one level share a column). A note says if seeds were added, and on how many users the
conditions are scored ([ablation_report.py:317-328](src/seqrec_eval/ablation_report.py#L317-L328)).

The **manipulation check** is read before any result. The full data's five characteristics are shown in
absolute terms, every condition as the relative change `(after − before) / |before|`, for the training data
and the test inputs. `manipulation_check` marks ⚠ any characteristic that moved by more than the tolerance
without being the target or declared in `expected_to_move` (a declaration may name one part only, `"test
catalogue"`), and ⛔ a target that moved against the transform's direction (truncation cannot lengthen
histories): that is a bug or an unforeseen interaction, and the sweep should not be read until it is
understood. Beside them: train rows changed, test rows changed, test span kept.

## Step 11.4: the results table

**code:**
[ablation_report.py:333-419](src/seqrec_eval/ablation_report.py#L333-L419); `_load` at
[ablation_report.py:236-247](src/seqrec_eval/ablation_report.py#L236-L247); `pool_over_seeds` at
[report.py:104-142](src/seqrec_eval/report.py#L104-L142).

**explanation:**
Per model, `plan_ablation` gives the specs and `_load` the finished test results, leaving out any run made
under another selection than the current one. Each finished run becomes a CSV row. For a catalogue sweep the
target fingerprints are set aside, since the targets differ between conditions by construction
([ablation_report.py:353-354](src/seqrec_eval/ablation_report.py#L353-L354)). A column is **averaged only once
every seed has finished** ([ablation_report.py:371-375](src/seqrec_eval/ablation_report.py#L371-L375)), seed
by seed in the sweep's order so paired comparisons align seed s with seed s, by `pool_over_seeds`: if the
seeds scored the same users it is `mean_over_seeds`; if not (a random catalogue, a different subsample per
seed), every user scored in at least one seed is kept and averaged over the seeds that scored them, so a user
counts once. Nothing leaves a comparison silently (final review A5): failed runs (⛔), stale runs (⛔) and
unfinished runs are listed per model with the levels they hold back ([ablation_report.py:377-391](src/seqrec_eval/ablation_report.py#L377-L391)).
The table shows mean ± sd over seeds per level and a row with the number of users scored (for a random
catalogue the union, with the per-seed range). A level under `min_level_users` is **descriptive only**: its
tests below show "— (descriptive)" in place of a verdict.

## Step 11.5: each level against the full data

**code:**
[ablation_report.py:426-454](src/seqrec_eval/ablation_report.py#L426-L454); the test of Step 8.3.

**explanation:**
Not run where each condition has its own users (the catalogue sweep): there is nothing to pair across
levels, and the report says so. Otherwise each model's levels are compared with its `full` by the seed-aware
t-test with **paired** seeds (seed s of a level is the full data's seed s, retrained or rescored), and Holm is
applied to all p-values of the sweep on this dataset at once: one family per sweep and dataset. With one seed
there is no seed spread and the test is over users only. Table: difference (level − full), 95% t interval,
SE of the seeds, adjusted p, users-only p, significant.

## Step 11.6: the knee

**code:**
[ablation_report.py:456-497](src/seqrec_eval/ablation_report.py#L456-L497); `find_knee` at
[ablation_report.py:169-234](src/seqrec_eval/ablation_report.py#L169-L234); `noninferiority` at
[seedstats.py:130-136](src/seqrec_eval/seedstats.py#L130-L136).

**explanation:**
Only for transforms with a `position` (history_length, density, repeat_removal), and only for a model whose
whole curve has finished. The per-user values of every column must be on the same users
([ablation_report.py:465-467](src/seqrec_eval/ablation_report.py#L465-L467)).

1. δ = `knee_margin` × the model's mean on the **full data** ([ablation_report.py:213-214](src/seqrec_eval/ablation_report.py#L213-L214)).
2. Levels are tested from the one closest to the full data downwards ([ablation_report.py:215](src/seqrec_eval/ablation_report.py#L215)),
   each against the full data, not against the best-looking level, which is lucky by construction.
3. The test is the seed-aware one-sided t-test of "the loss is smaller than δ" (Step 8.3), seeds paired:
   t = (mean difference + δ)/SE, upper tail ([ablation_report.py:216-223](src/seqrec_eval/ablation_report.py#L216-L223)).
   A small p is evidence the loss is below δ. Passing it at α is the same as the lower end of the two-sided
   90% interval for the loss lying above −δ, which is how the report words it.
4. Testing stops at the first level with p > α = 0.05 ([ablation_report.py:224-232](src/seqrec_eval/ablation_report.py#L224-L232)).
   The **knee** starts at `full` and moves down only through levels that passed, so it is the last level
   that passed, or `full` when the first test fails. This is the fixed-sequence procedure: hypotheses
   tested in an order fixed in advance, stopping at the first non-rejection, keep the family-wise error at α
   without correction.
5. **Power** per level: the chance the test shows "within δ" for a level that truly loses nothing, from that
   level's spread over users and seeds. The report shows the lowest and highest over the levels. Where it is
   low, a knee at `full` means the test could not tell, not that the reduced levels are worse.
6. **Sensitivity** ([ablation_report.py:459](src/seqrec_eval/ablation_report.py#L459), `KNEE_SENSITIVITY`
   at [ablation_report.py:97](src/seqrec_eval/ablation_report.py#L97)): the knee is also found at δ = 5%, 10%
   and 20% and shown in a second table, as a sensitivity beside the result fixed in advance.

Table: model, δ, knee (marked descriptive where a level is under `min_level_users`), the first level that
failed with its difference and p, and the power range. Without seed means, `find_knee` falls back to a
one-sided paired sign-flip randomisation test over users (`noninferiority_p`,
[ablation_report.py:103-124](src/seqrec_eval/ablation_report.py#L103-L124)) and its normal-approximation power
(`noninferiority_power`, [ablation_report.py:127-152](src/seqrec_eval/ablation_report.py#L127-L152)); the
report always passes seed means, so that path is used only by the tests.

## Step 11.7: the gap

**code:**
[ablation_report.py:499-548](src/seqrec_eval/ablation_report.py#L499-L548).

**explanation:**
The quantity the sweeps are about, in two tables:

- **Sequential − comparator (tested).** At every level, each sequential model's test `ndcg@10` minus the
  comparator's (ELSA by default, fixed in advance), by the seed-aware t-test; the seeds are paired where the
  sweep draws a subsample per seed, since two models at one level share subsample s
  ([ablation_report.py:502](src/seqrec_eval/ablation_report.py#L502)). Holm across all gaps of the sweep on
  this dataset. Table: condition, model, users, gap, 95% t interval, adjusted p, significant. If the
  comparator has no finished runs, the report says so.
- **Sequential − best non-sequential (descriptive).** Per level the best matrix model is the one with the
  highest test `ndcg@10`, chosen on test, so this gap carries no test; it shows whether any non-sequential
  model, not only the comparator, closes the gap.

Each row records the users scored and whether that is below `min_level_users`.

## Step 11.8: baselines and sequence signal at every condition

**code:**
`_analysis_section` at [ablation_report.py:556-626](src/seqrec_eval/ablation_report.py#L556-L626);
`condition_results` at [analysis.py:391-412](src/seqrec_eval/analysis.py#L391-L412);
`mean_over_subsamples` at [analysis.py:438-444](src/seqrec_eval/analysis.py#L438-L444).

**explanation:**
What `analyse` cached per condition (Step 3.12) is read back and pooled over subsamples, per column:
every baseline's test `ndcg@10`, the floor (the strongest baseline at that level, with its name, and only
when every baseline has a result), both controls, and Markov split into tied-end and real-gap users (slices
under 100 users left out). A column of a random sweep is shown only once every subsample has been analysed;
until then its cells say "partial", so a floor never averages over fewer subsamples than the models do
(final review C2; [ablation_report.py:574-584](src/seqrec_eval/ablation_report.py#L574-L584)). For the plot
the floor is also put on the gap's scale: floor minus the comparator at that level (or the best matrix model
where the comparator has no result; [ablation_report.py:608-616](src/seqrec_eval/ablation_report.py#L608-L616)).

## Step 11.9: the gap plot

**code:**
`gap_figure` at [plots.py:50-124](src/seqrec_eval/plots.py#L50-L124); `level_order` at
[ablation_report.py:641-648](src/seqrec_eval/ablation_report.py#L641-L648); called at
[ablation_report.py:688-695](src/seqrec_eval/ablation_report.py#L688-L695).

**explanation:**
The plot shows the tested gap against the comparator; only when there is none, the descriptive gap against
the best non-sequential model. One panel per dataset (at most three per row), each on its own y-axis. The
x-axis lists the levels evenly spaced, along the knee's axis where there is one, with `full` last. Each
sequential model is a line with its 95% t interval as a band, hollow markers where a level is descriptive
only, a dashed zero line ("no better than the comparator"), and the floor as a grey dashed line. Lines are
labelled at their ends (`_label_ends`, [plots.py:33-47](src/seqrec_eval/plots.py#L33-L47)), so identity never
rests on colour alone. Returns PNG bytes, or `None` without matplotlib (`available`, [plots.py:25-30](src/seqrec_eval/plots.py#L25-L30)).

---

# Part 12: `seqrec-eval repeat-strata`

## Step 12.1

**code:**
[cli.py:410-419](src/seqrec_eval/cli.py#L410-L419) → `build_strata_report` at
[strata.py:179-198](src/seqrec_eval/strata.py#L179-L198) → `dataset_strata` at
[strata.py:98-176](src/seqrec_eval/strata.py#L98-L176); settings at [strata.py:45-61](src/seqrec_eval/strata.py#L45-L61)
from `[repeat_strata]` ([protocol.toml:376-380](protocol.toml#L376-L380)).

**explanation:**
A slice of the stage-1 results, not a sweep: nothing is refitted or rescored.

1. **Settings** (`settings`): `history_bins` (lower edges; the last bin is open), `repeat_bins` (0 to 1),
   `min_users` (100) and `n_resamples` (1,999), each checked.
2. **Results** ([strata.py:102-113](src/seqrec_eval/strata.py#L102-L113)): per model, the finals averaged
   over seeds, only for models with every seed of the dataset finished (added seeds included). Matrix models
   are the non-sequential side, sequence models the candidates. All must have scored the same users.
3. **Per user** ([strata.py:121-127](src/seqrec_eval/strata.py#L121-L127)): each scored test user's history
   length and repeat share (`repeat_share`, [strata.py:64-68](src/seqrec_eval/strata.py#L64-L68): the share
   of its events that repeat an item earlier in the same history). The test histories are the same with or
   without refit, so the split is loaded as prepared (checked against the protocol).
4. **Cells** ([strata.py:129-141](src/seqrec_eval/strata.py#L129-L141)): users binned by both
   (`_bins`, [strata.py:71-78](src/seqrec_eval/strata.py#L71-L78)). Binning by length too matters: long
   histories hold more repeats simply because they are long. The first table counts users per cell.
5. **Gap per cell** ([strata.py:143-158](src/seqrec_eval/strata.py#L143-L158)): the best non-sequential model
   is chosen **once per dataset** on all test users ([strata.py:135](src/seqrec_eval/strata.py#L135)), so a
   small cell cannot pick its own comparator. In each cell with at least `min_users` users: every model's
   mean, and each sequential model's mean gap with a bootstrap percentile 95% interval (`_bootstrap`,
   [strata.py:88-95](src/seqrec_eval/strata.py#L88-L95); stream `search_seed, "strata", dataset, model, cell`).
6. **Output:** a gap grid per sequential model, `reports/repeat-strata.md` and `.csv`. The cells are
   observational (users with many repeats differ in more than their repeats), so no test is made; the
   manipulated counterpart is the `repeat_removal` sweep.

---

# Part 13: the scripts around the commands

Four scripts in [scripts/](scripts/) drive the commands; none of them computes a result itself.

## Step 13.1: `scripts/local-run.sh`: a quick real run on the laptop

**code:**
[scripts/local-run.sh](scripts/local-run.sh); usage and settings at
[local-run.sh:1-28](scripts/local-run.sh#L1-L28).

**explanation:**
`scripts/local-run.sh DATASET MODEL...` runs every step, from `prepare` to the reports, on one dataset and the
named models, to check the whole chain works on real data; it does not produce results.

- **A protocol of its own** ([local-run.sh:64-85](scripts/local-run.sh#L64-L85)): `protocol.local.toml` with
  `seeds`, `trials_per_model` and every `epochs` choice replaced by `SEEDS` (default `0`), `TRIALS` (2) and
  `EPOCHS` (1), written next to the results as `$WORK/protocol.toml`. A sweep with seeds of its own runs under
  the first. The script stops if any of those lines could not be replaced, so it is never a full run by
  accident. Its fingerprints are its own, so nothing here can mix with a real run. `FULL=1` copies
  `protocol.local.toml` unchanged instead.
- **Every command** goes through `run` ([local-run.sh:87-98](scripts/local-run.sh#L87-L98)): inside
  `systemd-run --user --scope` with a memory cap (`MEM`, default 6G, `none` for no cap), so a run that grows
  too large is killed instead of the laptop; `DRY=1` only prints the commands; `CR_SRC` puts a library source
  folder ahead of the installed copy. numpy's BLAS threads are capped at 4 ([local-run.sh:57-60](scripts/local-run.sh#L57-L60)).
- **The order** ([local-run.sh:100-114](scripts/local-run.sh#L100-L114)): `plan`, `prepare`,
  `analyse --sweep none`, `search`, `final`, `latency`, `report`, `repeat-strata`, and with `SWEEP=<name>`
  also `ablate`, `analyse --sweep` and `ablation-report` for that sweep; `status` last. Results go to
  `work-local/<dataset>/` (or `work-local-full/<dataset>/`).

Tested in [tests/test_local_run.py](tests/test_local_run.py) without running anything.

## Step 13.2: `scripts/dgx-run.sh`: the whole suite on the DGX

**code:**
[scripts/dgx-run.sh](scripts/dgx-run.sh); settings at [dgx-run.sh:20-33](scripts/dgx-run.sh#L20-L33), the
order at [dgx-run.sh:162-190](scripts/dgx-run.sh#L162-L190).

**explanation:**
Runs every step in order of importance, with several processes sharing each step through their run locks
(Step 5.3). Settings come from the environment: `WORK`, `DATA_DIR`, `PROTOCOL`, `GPUS` (one torch device per
GPU process, default `cuda:0`), `GPU_MODELS` (default `elsa gru sasrec`), `CPU_MODELS` (default
`popularity ease`), `DATASETS`, `SWEEPS` (in the order they run), `LATENCY_THREADS`/`LATENCY_CORES`, `LOGS`,
`ROUNDS`; `DRY=1` prints the steps only.

- **Shared steps** (`shared`, [dgx-run.sh:104-132](scripts/dgx-run.sh#L104-L132)): `search`, `final` and each
  `ablate --sweep` start one process per GPU device for the GPU models and one CPU process for EASE and
  popularity, each with its own log (`$WORK/logs/<step>-r<round>-gpu<i>.log`, `-cpu.log`). The exit codes of
  Step 0.1 decide what happens next (`collect`, [dgx-run.sh:88-100](scripts/dgx-run.sh#L88-L100)): 0 goes on;
  3 (work left over, e.g. a run another process held) repeats the step, up to `ROUNDS` = 3 times; anything else
  stops the script with a pointer to the logs and `seqrec-eval status`.
- **The order** ([dgx-run.sh:162-190](scripts/dgx-run.sh#L162-L190)): `plan`; `prepare`; stage 1 (`search`,
  then `final`) with the full data's analysis (`analyse --sweep none`, CPU) running beside it; the stage-1
  report; then the sweeps one by one (by default `history_length_inference`, `shuffle`, `history_length`,
  `density`, `repeat_removal`, `catalogue_top`), each sweep's analysis running beside the next sweep; then
  `latency` alone, since it times the CPU every other step loads; then every report and `status`.
- **Stopping** ([dgx-run.sh:62-70](scripts/dgx-run.sh#L62-L70)): `kill` on the script sends SIGTERM to every
  process it started, which stop cleanly without counting a failed attempt (Step 0.1), and exits 130. A rerun
  skips every finished run.

Tested in [tests/test_dgx_run.py](tests/test_dgx_run.py) against a fake `seqrec-eval` that records each call.

## Step 13.3: `scripts/project-time.py`: how long a protocol takes

**code:**
[scripts/project-time.py](scripts/project-time.py); the method in its docstring
([project-time.py:1-23](scripts/project-time.py#L1-L23)).

**explanation:**
Estimates the GPU time of a protocol from timing runs (short `local-run.sh` runs on the DGX). It reads each
timed run's `fit_seconds` and `eval_seconds` from `done.json` (`read_timing`,
[project-time.py:52-67](scripts/project-time.py#L52-L67)). Per GPU model (those trained in epochs), the cost of
one epoch of a configuration is the measured one where that configuration was timed, else a log-linear fit
over its sizes (batch, widths, depth, history length, negatives; `EpochCost`,
[project-time.py:70-93](scripts/project-time.py#L70-L93)). Another dataset costs the reference dataset's time
the ratio measured on configurations timed on both, or a `--factor` given by hand
([project-time.py:102-106](scripts/project-time.py#L102-L106)). `project`
([project-time.py:109-166](scripts/project-time.py#L109-L166)) prices the protocol's planned trials, the
finals (seeds × the mean, and for the worst case the most expensive, planned configuration, times the measured
final/trial ratio), every retraining level (a final's fit times `--refit-share`) and every rescoring. `main`
prints a table of GPU hours per dataset and model, the total in days, and the days at a `--speedup` for
processes run at once. EASE and popularity run on the CPU and are left out. Tested in
[tests/test_project_time.py](tests/test_project_time.py).

## Step 13.4: `scripts/vendor-cr.sh`: refreshing the vendored library

**code:**
[scripts/vendor-cr.sh](scripts/vendor-cr.sh).

**explanation:**
Copies the library checkout the suite runs on (default `~/Documents/recombee/compresso-recsys`) into
`vendor/compresso-recsys`, committed and uncommitted changes alike, so the repository is self-contained and
nothing has to be pushed to the library. It only reads the checkout. It copies what the build needs (`src`,
`tests`, `pyproject.toml`, `README.md`, `LICENSE`; [vendor-cr.sh:29-32](scripts/vendor-cr.sh#L29-L32)),
writes `CHANGES.patch`, the full difference from upstream (`origin/main` unless `UPSTREAM` says otherwise;
[vendor-cr.sh:34-42](scripts/vendor-cr.sh#L34-L42)), puts a one-line notice at the top of every changed
Python file, as the Apache-2.0 licence asks ([vendor-cr.sh:44-48](scripts/vendor-cr.sh#L44-L48)), adds a
`[tool.uv]` `cache-keys` to the copy's `pyproject.toml` so `uv sync` reinstalls it whenever its source changes
([vendor-cr.sh:50-59](scripts/vendor-cr.sh#L50-L59)), and writes `VENDORED.md`: where it came from, which
commit, how many uncommitted files, and the changed files ([vendor-cr.sh:61-95](scripts/vendor-cr.sh#L61-L95)).
After it, `uv sync` installs the new copy.

---

# Appendix A: everything the suite writes

```
work/
├── tmp/                                                the library's temporary files (unless TMPDIR is set)
├── logs/                                               scripts/dgx-run.sh: one log per process and round
├── splits/<dataset>/                                   prepare (Part 2)
│   ├── manifest.json, data/…                           the library's split
│   ├── split_info.json                                 the record (Step 2.7)
│   ├── val_rows.npy                                    fixed validation sample (Step 2.6)
│   ├── x_train_timestamps.npy, train_source_…, val_source_…, test_source_timestamps.npy   (Step 2.8)
│   ├── val_next_target_matrix.npz, test_next_target_matrix.npz                            (Step 2.8)
│   └── x_refit*.npz|npy, refit_source_*, refit_target_matrix.npz, refit_user_ids.npy       (Step 2.9)
├── splits/.<dataset>.prepare/.lock                     one prepare per dataset at a time (Step 2.2)
├── analysis/<dataset>/<dataset fp[:12]>/               analyse (Part 3)
│   ├── profile-<evaluation key[:12]>.json
│   ├── baselines/<name>/<baseline fp[:12]>/
│   │   ├── trial-NNN.json, selected.json               validation search
│   │   ├── test.json, test.npz, test.done.json         selected setting, refitted, on test
│   │   ├── test_window.json, .npz, .done.json          the other target definition
│   │   └── tied.npy                                    Markov only
│   └── controls/<markov_shuffled|markov_backwards>/<controls fp[:12]>/test*, test_window*
├── runs/<dataset>/                                     search, final (Parts 5, 6)
│   ├── added_seeds.json                                final --add-seeds
│   └── <model>/<run fp[:12]>/
│       ├── trial-NNN/  spec.json, val.json, val.npz, done.json | failed.json, .lock, attempts.json while running
│       ├── final-seedS/  spec.json, test.*, test_window.*, test_new.*, model.zip, done.json, latency.json
│       ├── accepted_failures.json                      final --accept-failed
│       └── incomplete_selection.json                   final --allow-incomplete before the search ended
├── ablations/<sweep>/<dataset>/                        ablate, analyse (Parts 10, 3)
│   ├── added_seeds.json                                ablate --add-seeds
│   └── <condition fp[:12]>/
│       ├── test_rows.npy, test_rows.json               fixed-set sweeps only
│       ├── conditions/full.json, <level>.json, <level>/seedS.json
│       ├── runs/<model>/<ablation fp[:12]>/<full|level>/final-seedS/  spec.json, test.*, done.json
│       └── analysis/<scorer>/<fp[:12]>/<condition>/test.*, test.done.json, tied.npy
└── reports/
    ├── analysis.md, analysis.csv                       analysis-report
    ├── stage1.md, final_metrics.csv                    report
    ├── ablation-<sweep>.md, -gap.png, -metrics.csv, -gap.csv, -analysis.csv
    └── repeat-strata.md, repeat-strata.csv
```

Two writing rules hold everywhere: JSON and saved evaluations are written to a temporary file, flushed to
disk and renamed into place ([results.py:41-58](src/seqrec_eval/results.py#L41-L58)), and a run is finished
exactly when its `done.json` exists.

# Appendix B: fingerprints and keys

| name | covers | names | defined at |
|---|---|---|---|
| dataset fingerprint | the dataset section's build settings (without `exclude_seen`, `new_item_diagnostic`), `max_val_users`, `search_seed` | `split_info.json` (checked by `prepare` and every `load_split`), `analysis/<dataset>/<fp>` | [protocol.py:276-285](src/seqrec_eval/protocol.py#L276-L285) |
| run fingerprint | result settings, the whole dataset section, model section | `runs/<dataset>/<model>/<fp>` | [protocol.py:287-293](src/seqrec_eval/protocol.py#L287-L293) |
| baseline fingerprint | result settings, the whole dataset section, baseline section, `BASELINE_VERSION` | `analysis/…/baselines/<name>/<fp>` | [protocol.py:246-253](src/seqrec_eval/protocol.py#L246-L253) |
| evaluation key | `targets`, `refit`, `SCORING_VERSION`, the dataset's `exclude_seen` | the profile file; part of the condition fingerprint | [protocol.py:266-274](src/seqrec_eval/protocol.py#L266-L274) |
| controls fingerprint | dataset fingerprint, `search_seed`, `CONTROLS_VERSION`, result settings, the whole dataset section | `analysis/…/controls/<control>/<fp>`, and the controls' folders at every condition | [analysis.py:234-238](src/seqrec_eval/analysis.py#L234-L238) |
| condition fingerprint | dataset fingerprint, evaluation key, sweep transform/levels/options/scope, transform version | `ablations/<sweep>/<dataset>/<fp>` | [ablations.py:663-675](src/seqrec_eval/ablations.py#L663-L675) |
| ablation fingerprint | condition fingerprint, stage-1 run fingerprint | `…/runs/<model>/<fp>` | [ablations.py:678-680](src/seqrec_eval/ablations.py#L678-L680) |
| condition scorer fingerprint | baseline fingerprint + selected params | `…/analysis/<baseline>/<fp>` | [analysis.py:302-303](src/seqrec_eval/analysis.py#L302-L303) |

"Result settings" are the `[protocol]` keys of `_RESULT_KEYS` (all but `seeds` and `eval_batch_size`), the
first seed as `trial_seed`, and `SCORING_VERSION` ([protocol.py:39-51](src/seqrec_eval/protocol.py#L39-L51),
[protocol.py:260-264](src/seqrec_eval/protocol.py#L260-L264)).

Versions that invalidate caches when code changes:

| version | now | where | what bumping it redoes |
|---|---|---|---|
| `SCORING_VERSION` | 4 | [protocol.py:51](src/seqrec_eval/protocol.py#L51) | every run, baseline, control and condition |
| `BASELINE_VERSION` | 2 | [protocol.py:59](src/seqrec_eval/protocol.py#L59) | every baseline search and score |
| `CONTROLS_VERSION` | 3 | [analysis.py:90](src/seqrec_eval/analysis.py#L90) | the sequence-signal controls |
| `NEXT_TARGETS_VERSION` | 2 | [timestamps.py:78](src/seqrec_eval/timestamps.py#L78) | a timestamp refresh at the next `prepare` |
| transform `version` | history_length 3, density 2, repeat_removal 1, shuffle 1, catalogue 2 | each `@register` in [ablations.py](src/seqrec_eval/ablations.py) | that transform's conditions |

Not fingerprinted on purpose: `eval_batch_size`; every seed but the first (so seeds can be added); `[latency]`;
`[repeat_strata]`; an ablation's `datasets`, `models`, `seeds`, `expected_to_move`, `manipulation_tolerance`,
`knee_margin`, `min_level_users` (they choose what runs or how the report reads it, not what a result is).
`exclude_seen` and `new_item_diagnostic` are left out of the dataset fingerprint only: they do not change the
split, but every fingerprint of a scored result holds them.

# Appendix C: the keys of `split.data`

| key | from | changed by |
|---|---|---|
| `item_ids`, `train_item_ids`, `val_item_ids`, `test_item_ids` | library (Step 2.5) | `swap_training` (train ← val catalogue); catalogue transform |
| `x_train`, `train_source_matrix`, `train_target_matrix` | library | `swap_training`; `keep_events`; catalogue |
| `x_train_sequences`, `train_source_sequences` | library | `swap_training`; `keep_events`; shuffle; catalogue |
| `val_source_matrix`, `val_target_matrix`, `test_source_matrix`, `test_target_matrix` | library | `keep_events` (sources); catalogue (sources and targets) |
| `val_source_sequences`, `test_source_sequences` | library | `keep_events`; shuffle; catalogue |
| `train_user_ids`, `val_user_ids`, `test_user_ids`, `val_eval_user_ids`, `test_eval_user_ids` | library | `swap_training` (train); catalogue (train rows dropped) |
| `warm_item_indices`, `val_cold_item_indices`, `test_cold_item_indices` | library | `swap_training`; catalogue |
| `val_source_indices` … `test_target_indices`, `entity_*`, `tag_names` | library | catalogue clears the index lists |
| `x_train_timestamps`, `train_source_timestamps`, `val_source_timestamps`, `test_source_timestamps` | `load_timestamps` (Step 2.8) | `swap_training`; carried along by every transform |
| `val_next_target_matrix`, `test_next_target_matrix` | `load_timestamps` | catalogue (removed items' targets leave) |
| `x_refit`, `refit_source_matrix`, `refit_target_matrix`, `x_refit_sequences`, `refit_source_sequences`, `x_refit_timestamps`, `refit_source_timestamps`, `refit_user_ids` | `load_refit` (Step 2.9) | read only by `swap_training` |
| `val_seen_matrix`, `test_seen_matrix` | `apply_condition` (Step 10.5) | only on conditions whose inputs were thinned in the same item space; `seen_history` falls back to the source matrix (Step 3.9.4) |
| `_kept` | `keep_events`, catalogue (Step 10.6) | only on conditions; read by `_edits` |

# Appendix D: the tests

Run with `.venv/bin/python -m pytest` from the repository root; `SEQREC_EVAL_TEST_DEVICE=cuda` trains the
workflow tests on the GPU ([tests/test_smoke.py:45-47](tests/test_smoke.py#L45-L47)). They use tiny synthetic
data only.

| file | tests | covers |
|---|---|---|
| [tests/test_smoke.py](tests/test_smoke.py) | 37 | the whole workflow on a synthetic dataset registered with the real library builder: `prepare` through `report`; the local protocol matching the main one |
| [tests/test_ablations.py](tests/test_ablations.py) | 77 | every transform end to end and its defining properties, the fixed and per-condition users, the knee, the reports |
| [tests/test_baselines.py](tests/test_baselines.py) | 23 | the four baselines, their tie order and sparse ranking, and the profile's timing fields, on histories small enough to check by hand |
| [tests/test_evaluate.py](tests/test_evaluate.py) | 10 | the seen-item mask and the evaluator batch order it relies on (H33) |
| [tests/test_seeds.py](tests/test_seeds.py) | 11 | seeds added after the protocol's: `final --add-seeds`, `ablate --add-seeds` |
| [tests/test_seedstats.py](tests/test_seedstats.py) | 12 | the seed-aware t-test, paired and unpaired, and Holm |
| [tests/test_robustness.py](tests/test_robustness.py) | 12 | unknown protocol keys, locks, killed processes, exit codes, broken files, the cross-dataset table |
| [tests/test_plots.py](tests/test_plots.py) | 3 | the gap plot and its level order |
| [tests/test_local_run.py](tests/test_local_run.py) | 5 | `scripts/local-run.sh`: the quick protocol and the commands, without running them |
| [tests/test_dgx_run.py](tests/test_dgx_run.py) | 6 | `scripts/dgx-run.sh`, against a fake `seqrec-eval` |
| [tests/test_project_time.py](tests/test_project_time.py) | 3 | `scripts/project-time.py` |

(Counts are `def test_` functions; parametrised tests run more cases.)
