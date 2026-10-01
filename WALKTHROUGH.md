# seqrec_eval: a step-by-step walkthrough of the code

This document follows every command of the suite through the code, in the order the code runs, and says
which file and line each thing happens in. It describes the code as it stands on 2026-09-30. Line numbers
go stale when the code changes; the function names do not, so search for the name if a link lands a few
lines off.

**How to read the references**

- `cli.py:249-262` style links are relative to the repository root and open the file at those lines in
  VS Code (Ctrl+click in the editor, or click in the Markdown preview).
- **cr** is compresso-recsys, the library the suite is built on. Library links point at the checkout
  `~/Documents/recombee/compresso-recsys`, branch `local-temporal-train-all-users`, whose `builder.py`
  has the uncommitted `temporal_train_users` change that the protocol's `train_users = "all"` needs. An
  installed release can have different line numbers.
- A **phase** is `train`, `val` or `test`. A **row** is one user in one phase. A phase's **catalogue** is
  its list of item ids (`train_item_ids`, `val_item_ids`, `test_item_ids`); each is a prefix of the next,
  because each window only appends the items first seen in it.
- A **fingerprint** is a SHA-256 of the settings that determine a result
  ([protocol.py:51-53](src/seqrec_eval/protocol.py#L51-L53)). Results are stored in a folder named after its
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
report           stage-1 report: metrics, comparisons, floor, latency      reports/stage1.md, final_metrics.csv
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

---

# Part 0: what every command does first

## Step 0.1: the entry point

**code:**
`seqrec-eval` is the console script `seqrec_eval.cli:main` ([pyproject.toml:27-28](pyproject.toml#L27-L28));
`python -m seqrec_eval` does the same through [__main__.py:1-5](src/seqrec_eval/__main__.py#L1-L5).
`main` starts at [cli.py:148](src/seqrec_eval/cli.py#L148).

**explanation:**
There is one entry function for every command. It parses the command line
([cli.py:149](src/seqrec_eval/cli.py#L149)), loads the protocol
([cli.py:150](src/seqrec_eval/cli.py#L150)), resolves which datasets, models and sweeps the command is
about ([cli.py:151-156](src/seqrec_eval/cli.py#L151-L156)), and then branches on the command name
([cli.py:158-334](src/seqrec_eval/cli.py#L158-L334)). Each command returns an exit code: 0 for success,
1 for `search`, `final` and `ablate` when a run failed ([cli.py:229](src/seqrec_eval/cli.py#L229),
[cli.py:291](src/seqrec_eval/cli.py#L291)), 2 for an unknown command
([cli.py:334](src/seqrec_eval/cli.py#L334)).

## Step 0.2: parsing the command line

**code:**
`_parser()` at [cli.py:74-145](src/seqrec_eval/cli.py#L74-L145).

**explanation:**
Two options apply to every command:

- `--protocol` (default `protocol.toml`, [cli.py:77](src/seqrec_eval/cli.py#L77)): which protocol file
  to load. `protocol.local.toml` is a local copy that differs only in `eval_batch_size`.
- `--work-dir` (default `$SEQREC_EVAL_WORK`, else `./work`, [cli.py:78-79](src/seqrec_eval/cli.py#L78-L79)):
  where splits, runs and reports live.

The helper `selection()` ([cli.py:82-85](src/seqrec_eval/cli.py#L82-L85)) adds `--dataset` and, for
commands that concern models, `--model`. Per command:

| command | options | lines |
|---|---|---|
| `plan` | none | [cli.py:87](src/seqrec_eval/cli.py#L87) |
| `prepare` | `--dataset`, `--data-dir` (default `$COMPRESSO_DATA_DIR`), `--force`, `--quiet`, `--no-timestamps` | [cli.py:89-96](src/seqrec_eval/cli.py#L89-L96) |
| `search`, `final` | `--dataset`, `--model`, `--device`, `--threads`, `--retry-failed`; `final` also `--allow-incomplete`, `--accept-failed` | [cli.py:98-110](src/seqrec_eval/cli.py#L98-L110) |
| `status` | `--dataset`, `--model` | [cli.py:112-113](src/seqrec_eval/cli.py#L112-L113) |
| `report` | `--dataset`, `--model`, `--reference` (default `elsa`) | [cli.py:115-117](src/seqrec_eval/cli.py#L115-L117) |
| `latency` | `--dataset`, `--model`, `--threads` (default 4), `--cores` | [cli.py:119-122](src/seqrec_eval/cli.py#L119-L122) |
| `analyse` | `--dataset`, `--sweep`, `--threads` | [cli.py:124-127](src/seqrec_eval/cli.py#L124-L127) |
| `analysis-report` | `--dataset` | [cli.py:129-130](src/seqrec_eval/cli.py#L129-L130) |
| `ablate` | `--dataset`, `--model`, `--sweep`, `--device`, `--threads`, `--retry-failed` | [cli.py:132-137](src/seqrec_eval/cli.py#L132-L137) |
| `ablation-report` | `--dataset`, `--model`, `--sweep` | [cli.py:139-141](src/seqrec_eval/cli.py#L139-L141) |
| `repeat-strata` | `--dataset`, `--model` | [cli.py:143-144](src/seqrec_eval/cli.py#L143-L144) |

`--device` defaults to `cuda` when torch sees a GPU, else `cpu` ([cli.py:70-71](src/seqrec_eval/cli.py#L70-L71)).

## Step 0.3: loading the protocol

**code:**
`load_protocol(args.protocol)` at [cli.py:150](src/seqrec_eval/cli.py#L150) runs
[protocol.py:370-439](src/seqrec_eval/protocol.py#L370-L439).

**explanation:**
The protocol file is the frozen description of the whole study: what is measured, on which data, with
which models and baselines, and which ablations run. `load_protocol` reads the TOML
([protocol.py:371-373](src/seqrec_eval/protocol.py#L371-L373)), checks every section, and returns one
frozen `Protocol` object ([protocol.py:156-232](src/seqrec_eval/protocol.py#L156-L232)) that every other
module reads. A mistake in the file fails here, before any data is touched. The steps below go through it
section by section.

### Step 0.3.1: the `[protocol]` section

**code:**
[protocol.py:374-407](src/seqrec_eval/protocol.py#L374-L407); the values in
[protocol.toml:14-54](protocol.toml#L14-L54).

**explanation:**
Each key is read and checked. `_require` ([protocol.py:56-59](src/seqrec_eval/protocol.py#L56-L59))
raises `ProtocolError` for a missing key.

| key | value now | check | what it means |
|---|---|---|---|
| `version` | 1 | required integer ([protocol.py:422](src/seqrec_eval/protocol.py#L422)) | the protocol's own version |
| `cutoffs` | [5, 10, 20] | positive integers, sorted and de-duplicated ([protocol.py:376-378](src/seqrec_eval/protocol.py#L376-L378)) | the k of every metric@k; the model is always asked for the top max(cutoffs) = 20 |
| `metrics` | ndcg, recall, calibrated_recall, hit_rate, precision, map, mrr | each must be in `METRIC_NAMES` ([protocol.py:31](src/seqrec_eval/protocol.py#L31), [protocol.py:379-382](src/seqrec_eval/protocol.py#L379-L382)) | the metric families computed; `evaluate.py` checks at import that these are exactly the library's names ([evaluate.py:65-67](src/seqrec_eval/evaluate.py#L65-L67)) |
| `primary_metric` | `ndcg@10` | must match `<metric>@<k>` with the metric listed and k a cutoff ([protocol.py:384-390](src/seqrec_eval/protocol.py#L384-L390)) | the one metric that selects configurations, sets the floor and drives every test |
| `seeds` | [0, 1, 2] | non-empty, distinct ([protocol.py:392-394](src/seqrec_eval/protocol.py#L392-L394)) | final runs per selected configuration; also the data seeds of stochastic ablations |
| `trials_per_model` | 20 | integer ([protocol.py:395](src/seqrec_eval/protocol.py#L395)) | default random-search budget per model and baseline |
| `search_seed` | 20260923 | required ([protocol.py:428](src/seqrec_eval/protocol.py#L428)) | the root of every random stream in the suite (search draws, validation sample, shuffles, ablation subsamples, bootstrap and knee resamples) |
| `max_val_users` | 20000 | optional ([protocol.py:396](src/seqrec_eval/protocol.py#L396)) | size of the fixed validation sample every search trial is scored on |
| `targets` | `"next"` | required, `"next"` or `"window"` ([protocol.py:398-404](src/seqrec_eval/protocol.py#L398-L404)) | what a user is scored against: the first thing they do after their history (`next`), or everything in the window (`window`); the other one is computed as a diagnostic |
| `refit` | `true` | required boolean ([protocol.py:398-407](src/seqrec_eval/protocol.py#L398-L407)) | whether every model scored on test is first refitted on train + validation |
| `eval_batch_size` | 1024 | default 1024 ([protocol.py:432](src/seqrec_eval/protocol.py#L432)) | rows per evaluator batch; speed only, so it is in no fingerprint |

`targets` and `refit` have no default on purpose: a missing key once silently ran a different protocol
(review H11). `[latency]` ([protocol.toml:56-61](protocol.toml#L56-L61)) is stored as a plain dict
([protocol.py:433](src/seqrec_eval/protocol.py#L433)) and read only by the latency benchmark.

### Step 0.3.2: the datasets

**code:**
[protocol.py:409](src/seqrec_eval/protocol.py#L409) calls `_parse_dataset`
([protocol.py:245-274](src/seqrec_eval/protocol.py#L245-L274)) for every `[datasets.<name>]`, which returns
a `DatasetProtocol` ([protocol.py:62-108](src/seqrec_eval/protocol.py#L62-L108)). The five datasets are at
[protocol.toml:85-151](protocol.toml#L85-L151).

**explanation:**
Each field of `DatasetProtocol`:

| field | example (ml20m) | meaning |
|---|---|---|
| `name` | `ml20m` | the key used everywhere in the suite and in folder names |
| `builder` | `ml20m` | the library's dataset name (`amazon2023`, `music4all-onion`, `yambda`, `otto` for the others) |
| `temporal_period_hours` | 8136 (339 days) | width of each of the three windows at the end of the log; must be positive ([protocol.py:250-252](src/seqrec_eval/protocol.py#L250-L252)) |
| `min_user_support` | 5 | a user needs this many distinct items in a stage to be kept |
| `item_min_support` | 1 | an item first seen in a stage needs this many users in it to be kept |
| `min_value_to_keep` | 3.0 | rating threshold; `"none"` keeps everything ([protocol.py:235-242](src/seqrec_eval/protocol.py#L235-L242)) |
| `set_all_values_to` | 1.0 | every kept event gets this value |
| `exclude_seen` | true | whether items already in the history may be recommended |
| `new_item_diagnostic` | false | whether finals are also scored on targets not already in the history |
| `train_users` | `"all"` | who the training stage keeps: `"all"` = everyone with enough events before the validation window, `"window"` = only users also active in the train target window. Required, no default ([protocol.py:254-259](src/seqrec_eval/protocol.py#L254-L259)) |
| `amazon_category` | (amazon only) `Toys_and_Games` | which Amazon category |
| `options` | (music4all, yambda, otto) | extra dataset options for the library, e.g. `{ user_sample = 0.5 }` |
| `raw` | the whole TOML table | what the fingerprints hash |

`build_parameters()` ([protocol.py:81-108](src/seqrec_eval/protocol.py#L81-L108)) turns this into the
keyword arguments for the library's builder. Two details matter. "Keep everything" is sent as `-inf`, not
`None`, because the builder reads `None` as "use the registry default", which for ML-20M is a 4-star
threshold ([protocol.py:95-97](src/seqrec_eval/protocol.py#L95-L97)). And `temporal_train_users` is only
passed when it is not the library default `"window"` ([protocol.py:104-107](src/seqrec_eval/protocol.py#L104-L107)),
so a library without that option still works for a `"window"` protocol.

### Step 0.3.3: the models

**code:**
[protocol.py:410](src/seqrec_eval/protocol.py#L410) calls `_parse_model`
([protocol.py:300-319](src/seqrec_eval/protocol.py#L300-L319)) for every `[models.<name>]`, returning a
`ModelProtocol` ([protocol.py:111-119](src/seqrec_eval/protocol.py#L111-L119)). The models are at
[protocol.toml:176-235](protocol.toml#L176-L235).

**explanation:**
- `family` must be `"matrix"` or `"sequence"` ([protocol.py:302-304](src/seqrec_eval/protocol.py#L302-L304)).
  A matrix model trains on the user × item CSR matrix, a sequence model on the ordered histories.
- `fixed` is passed unchanged to every trial; `space` is searched. Each `space` entry must be one
  distribution, checked by `_check_distribution` ([protocol.py:277-297](src/seqrec_eval/protocol.py#L277-L297)):
  `choice = [...]` (non-empty list), `uniform = [low, high]`, `loguniform = [low, high]` (low > 0), or
  `int = [low, high]` (inclusive).
- A parameter cannot be both fixed and searched ([protocol.py:309-311](src/seqrec_eval/protocol.py#L309-L311)).
- `trials` defaults to `trials_per_model` when there is a space, else 1
  ([protocol.py:312](src/seqrec_eval/protocol.py#L312)). So `popularity` (no space) has 1 trial; `ease`,
  `elsa`, `gru` and `sasrec` have 20 each.
- `max_items` skips the model on catalogues larger than this (EASE: 40,000, because it builds a dense
  item × item matrix).

### Step 0.3.4: the ablations and the baselines

**code:**
Ablations: [protocol.py:415-416](src/seqrec_eval/protocol.py#L415-L416) → `_parse_ablation`
([protocol.py:347-367](src/seqrec_eval/protocol.py#L347-L367)) → `AblationProtocol`
([protocol.py:139-153](src/seqrec_eval/protocol.py#L139-L153)).
Baselines: [protocol.py:417-418](src/seqrec_eval/protocol.py#L417-L418) → `_parse_baseline`
([protocol.py:322-344](src/seqrec_eval/protocol.py#L322-L344)) → `BaselineProtocol`
([protocol.py:122-136](src/seqrec_eval/protocol.py#L122-L136)).

**explanation:**
An ablation is a `transform`, a non-empty list of distinct `levels`, optional `options`, and which
`datasets` and `models` it covers (default: all of them; unknown names fail,
[protocol.py:358-363](src/seqrec_eval/protocol.py#L358-L363)). Whether the levels make sense is not
checked here but by the transform itself (Step 0.4), because only it knows what a level means. The other
keys of the section (`scope`, `knee_margin`, `expected_to_move`, `manipulation_tolerance`,
`min_level_users`) stay in `raw` and are read by `ablations.py` and `ablation_report.py`.

A baseline has a `kind`, one of `popularity`, `time_popularity`, `replay`, `markov`
([protocol.py:27](src/seqrec_eval/protocol.py#L27)), and a `space` checked like a model's. The parser
decides how it is searched ([protocol.py:334-343](src/seqrec_eval/protocol.py#L334-L343)): if every
parameter is a `choice`, the grid size is the product of the choice lengths; a continuous parameter makes
it infinite. A grid no larger than `trials_per_model` is searched **in full** (`grid = True`, and `trials`
must then equal the grid size); otherwise it is random search with `trials_per_model` trials. With the
current protocol ([protocol.toml:247-263](protocol.toml#L247-L263)):

| baseline | space | search | trials |
|---|---|---|---|
| popularity | `count` ∈ {events, users} | grid | 2 |
| time_popularity | `half_life_days` loguniform [0.25, 365] | random | 20 |
| replay | `order` ∈ {recency, frequency} | grid | 2 |
| markov | none | grid of one | 1 |

At least one dataset and one model must be defined ([protocol.py:411-414](src/seqrec_eval/protocol.py#L411-L414)).

### Step 0.3.5: the `Protocol` object and its fingerprints

**code:**
`Protocol` at [protocol.py:156-232](src/seqrec_eval/protocol.py#L156-L232), built at
[protocol.py:420-439](src/seqrec_eval/protocol.py#L420-L439).

**explanation:**
Besides the parsed values, `Protocol` has lookups that fail with a clear message for an unknown name
(`dataset()`, `model()`, `baseline()`, `ablation()`,
[protocol.py:179-205](src/seqrec_eval/protocol.py#L179-L205)) and the fingerprints every result folder is
named after:

- `_result_settings()` ([protocol.py:207-208](src/seqrec_eval/protocol.py#L207-L208)): the `[protocol]`
  keys that change results (`_RESULT_KEYS`, [protocol.py:35-38](src/seqrec_eval/protocol.py#L35-L38): all
  except `eval_batch_size`) plus `SCORING_VERSION` ([protocol.py:44](src/seqrec_eval/protocol.py#L44)),
  currently 2. The scoring version is bumped when the code changes what a scored target means (2 = the
  H16 fix: real first moment, only recommendable next items scored), so older results are not reused.
- `dataset_fingerprint(d)` ([protocol.py:218-224](src/seqrec_eval/protocol.py#L218-L224)): the dataset
  section, `max_val_users` and `search_seed`. It identifies a prepared split.
- `run_fingerprint(d, m)` ([protocol.py:226-232](src/seqrec_eval/protocol.py#L226-L232)): result settings +
  dataset section + model section. It identifies every trial and final of a model on a dataset.
- `baseline_fingerprint(d, b)` ([protocol.py:194-200](src/seqrec_eval/protocol.py#L194-L200)): the same with
  the baseline section in place of the model's.
- `evaluation_key()` ([protocol.py:210-216](src/seqrec_eval/protocol.py#L210-L216)): only `targets`, `refit`
  and the scoring version. It keys caches that hold users or profiles rather than runs (the analysis
  profile, an ablation's fixed test users and conditions).

## Step 0.4: choosing datasets, models and sweeps

**code:**
[cli.py:151-156](src/seqrec_eval/cli.py#L151-L156), `_select` at [cli.py:61-67](src/seqrec_eval/cli.py#L61-L67),
`check_ablation` at [ablations.py:182-207](src/seqrec_eval/ablations.py#L182-L207).

**explanation:**
`_select` turns `--dataset`, `--model` and `--sweep` into lists: nothing given, or `all`, means every name
in the protocol; an unknown name stops the program with the list of valid names. A command that has no
such option (for example `prepare` has no `--model`) gets all of them.

Every selected sweep is then checked once ([cli.py:155-156](src/seqrec_eval/cli.py#L155-L156)), so a bad
level fails now rather than halfway through a sweep. `check_ablation` looks the transform up in the
registry (`transform_of`, [ablations.py:165-169](src/seqrec_eval/ablations.py#L165-L169)), refuses a level
labelled `full` (reserved for the reference), runs the transform's own `validate` on the levels and
options, checks the `scope` is one the transform supports, and checks `expected_to_move`,
`manipulation_tolerance` (≥ 0), `knee_margin` (in (0, 1)) and `min_level_users` (positive integer).

---

# Part 1: `seqrec-eval plan`

## Step 1.1

**code:**
[cli.py:158-183](src/seqrec_eval/cli.py#L158-L183).

**explanation:**
`plan` only reads the protocol and the work folder; it writes nothing. It prints:

1. the protocol path, version, primary metric, seeds and cutoffs ([cli.py:160-161](src/seqrec_eval/cli.py#L160-L161));
2. per dataset, whether `splits/<dataset>/split_info.json` exists and the dataset fingerprint
   ([cli.py:162-165](src/seqrec_eval/cli.py#L162-L165));
3. per model on that dataset: family, number of trials, number of finals (= seeds), run fingerprint, and
   whether the model is registered in `models.py` (`_registered`, [cli.py:385-390](src/seqrec_eval/cli.py#L385-L390))
   ([cli.py:166-172](src/seqrec_eval/cli.py#L166-L172));
4. the total number of runs (trials + finals over all datasets and models, [cli.py:173](src/seqrec_eval/cli.py#L173));
5. per baseline: kind, number of validation trials, grid or random search ([cli.py:174-176](src/seqrec_eval/cli.py#L174-L176));
6. per sweep: transform, levels, scope and how many fits or rescorings it needs per dataset and model
   ([cli.py:177-182](src/seqrec_eval/cli.py#L177-L182)).

---

# Part 2: `seqrec-eval prepare --data-dir DIR`

`prepare` builds each dataset's temporal split once, extracts it to `work/splits/<dataset>/`, records what
it is, recovers the time of every event, derives the next-item targets, and (with `refit = true`) builds
the train + validation training set. Every later command loads this folder instead of building anything.

## Step 2.1: the CLI branch

**code:**
[cli.py:185-199](src/seqrec_eval/cli.py#L185-L199).

**explanation:**
`--data-dir` (or `$COMPRESSO_DATA_DIR`) is required: it is the library's data folder with the raw
downloads and the library's own caches ([cli.py:186-187](src/seqrec_eval/cli.py#L186-L187)). For each
selected dataset, `prepare_split` is called ([cli.py:190-192](src/seqrec_eval/cli.py#L190-L192)) with
`force` (rebuild a split prepared from a different dataset section), `show_progress` (off with `--quiet`)
and `timestamps` (off with `--no-timestamps`). Afterwards `split_info.json` is read back and the sizes are
logged: training users, validation users and how many of them the search scores, test users
([cli.py:193-198](src/seqrec_eval/cli.py#L193-L198)).

## Step 2.2: is there already a split?

**code:**
`prepare_split` at [splits.py:186-266](src/seqrec_eval/splits.py#L186-L266); the reuse check at
[splits.py:189-216](src/seqrec_eval/splits.py#L189-L216).

**explanation:**
The output folder is `work/splits/<dataset>` (`split_dir`, [splits.py:87-88](src/seqrec_eval/splits.py#L87-L88)).
If its `split_info.json` exists and `--force` is not given, the recorded `dataset_fingerprint` is compared
with the protocol's:

- **Same fingerprint:** the split is reused. Two things can still be missing and are added without
  rebuilding the split ([splits.py:197-211](src/seqrec_eval/splits.py#L197-L211)):
  - the timestamps, or next-item targets of an older definition (`next_targets_version` in the info is not
    `NEXT_TARGETS_VERSION` = 2, [timestamps.py:78](src/seqrec_eval/timestamps.py#L78));
  - the refit set, when `refit = true` and the info has no `refit` entry.

  If either is missing, the split is loaded, the prepared events are rebuilt once
  ([splits.py:201-203](src/seqrec_eval/splits.py#L201-L203)), the missing part is attached (Steps 2.8
  and 2.9), and `split_info.json` is rewritten. Then the function returns.
- **Different fingerprint:** it refuses ([splits.py:213-216](src/seqrec_eval/splits.py#L213-L216)).
  Rebuilding would change the split that every existing run on this dataset was scored against, so it
  takes `--force`.

## Step 2.3: the build parameters and the library check

**code:**
[splits.py:218-224](src/seqrec_eval/splits.py#L218-L224); `_check_library` at
[splits.py:175-183](src/seqrec_eval/splits.py#L175-L183).

**explanation:**
`build_parameters()` (Step 0.3.2) gives the builder's keyword arguments. Before the slow build,
`_check_library` compares them with the signature of the installed `build_recsys_checkpoint`. If a key is
not accepted (in practice `temporal_train_users`, which only the local library build has), it stops with
the installed version in the message. Then any leftover `.<dataset>.building.zip` and
`.<dataset>.staging/` from an interrupted run are removed ([splits.py:221-224](src/seqrec_eval/splits.py#L221-L224)).

## Step 2.4: the library builds the temporal split

**code:**
[splits.py:226-229](src/seqrec_eval/splits.py#L226-L229) calls cr's `build_recsys_checkpoint`
([cr builder.py:1859](../../recombee/compresso-recsys/src/compresso_recsys/builder.py#L1859)), which runs
`_build_recsys_checkpoint_from_args`
([cr builder.py:1685-1760](../../recombee/compresso-recsys/src/compresso_recsys/builder.py#L1685-L1760)) and,
for `split_mode = "temporal"`, `_build_temporal_split`
([cr builder.py:1494-1657](../../recombee/compresso-recsys/src/compresso_recsys/builder.py#L1494-L1657)).

**explanation:**
This is library code; the suite only calls it and times it (`build_seconds`). What it does, in order:

1. **Resolve and seed.** Registry defaults are merged into the arguments and `random`/`numpy` are seeded
   with the resolved seed ([cr builder.py:1686-1688](../../recombee/compresso-recsys/src/compresso_recsys/builder.py#L1686-L1688)).
2. **Load and preprocess.** The dataset adapter loads the raw interactions
   ([cr builder.py:1692-1693](../../recombee/compresso-recsys/src/compresso_recsys/builder.py#L1692-L1693)).
   For a temporal split, preprocessing applies only the rating threshold and `set_all_values_to`; user
   and item support are set to 1 here, because the support filters run later, per stage
   ([cr builder.py:1701-1709](../../recombee/compresso-recsys/src/compresso_recsys/builder.py#L1701-L1709)).
3. **Times and boundaries.** Timestamps are converted to unix seconds by magnitude (≥ 1e17 nanoseconds,
   ≥ 1e14 microseconds, ≥ 1e11 milliseconds; `_timestamps_in_seconds`,
   [cr builder.py:1338-1352](../../recombee/compresso-recsys/src/compresso_recsys/builder.py#L1338-L1352)).
   With P = `temporal_period_hours` and T = the last timestamp, the three windows are
   `train_target_start = T − 3P`, `validation_target_start = T − 2P`, `test_target_start = T − P`
   ([cr builder.py:1514-1525](../../recombee/compresso-recsys/src/compresso_recsys/builder.py#L1514-L1525)).
   Users and items are numbered by sorting their ids (`pd.factorize(sort=True)`); this order is the row
   order of every stage.
4. **Three stages**, each built by `_build_temporal_stage`
   ([cr builder.py:1355-1474](../../recombee/compresso-recsys/src/compresso_recsys/builder.py#L1355-L1474)):

   | stage | source (history) | target | items |
   |---|---|---|---|
   | train | t < train_target_start | train_target_start ≤ t < validation_target_start | all seen |
   | validation | t < validation_target_start | validation_target_start ≤ t < test_target_start | train items + new ones |
   | test | t < test_target_start | t ≥ test_target_start | validation items + new ones |

   Each stage's catalogue is the previous stage's items (inherited, kept) followed by the items first
   seen in this stage. The support filter `_filter_temporal_pair`
   ([cr builder.py:1169-1240](../../recombee/compresso-recsys/src/compresso_recsys/builder.py#L1169-L1240))
   repeats until nothing changes: a user needs ≥ `min_source_items` distinct items in the source,
   ≥ `min_target_items` in the target (both default 1) and ≥ `min_user_support` in source ∪ target; a
   *new* item needs ≥ `item_min_support` users in the stage (inherited items are never dropped).
   With `temporal_train_users = "all"`, the train stage sets `min_source_items` and `min_target_items`
   to 0 (`_train_stage_args`,
   [cr builder.py:1476-1491](../../recombee/compresso-recsys/src/compresso_recsys/builder.py#L1476-L1491)),
   so it keeps every user with `min_user_support` items before the validation window, not only users active
   in the train target window.
5. **Matrices and sequences.** Each stage's source and target are CSR matrices (users × stage catalogue)
   whose entries are the summed event values per (user, item), i.e. event counts, since every value is
   1.0. `x_train = train_source.maximum(train_target)`. Beside the matrices, the source histories are also
   saved in order as `ItemSequences` (Step 2.5).
6. **Write.** Everything is written into the zip by `save_recsys_split`
   ([cr builder.py:1733-1760](../../recombee/compresso-recsys/src/compresso_recsys/builder.py#L1733-L1760)),
   and the window boundaries go into the manifest
   ([cr builder.py:1646-1648](../../recombee/compresso-recsys/src/compresso_recsys/builder.py#L1646-L1648)),
   which is how the suite finds them later (`_boundaries`,
   [timestamps.py:123-129](src/seqrec_eval/timestamps.py#L123-L129)).

## Step 2.5: extracting and loading the split

**code:**
[splits.py:230-235](src/seqrec_eval/splits.py#L230-L235); cr's `load_recsys_split`
([cr checkpoint.py:533-679](../../recombee/compresso-recsys/src/compresso_recsys/checkpoint.py#L533-L679))
and `load_manifest`
([cr checkpoint.py:135-138](../../recombee/compresso-recsys/src/compresso_recsys/checkpoint.py#L135-L138)).

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

An `ItemSequences` ([cr sequences.py:53-110](../../recombee/compresso-recsys/src/compresso_recsys/sequences.py#L53-L110))
is CSR without data: `values` holds item indices of every row concatenated, oldest first; `indptr[i]` to
`indptr[i+1]` is row i; `n_items` is the size of the item space. Duplicates are kept, so a repeat is a
second event. Its arrays are made read-only, so no code can reorder a history after the fact.
`row_lengths` is the length of each history.

`load_manifest` returns the manifest dict, including `stages.data` with the three window boundaries.

## Step 2.6: the fixed validation sample

**code:**
[splits.py:236-241](src/seqrec_eval/splits.py#L236-L241); `_stream` at
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
[splits.py:243-253](src/seqrec_eval/splits.py#L243-L253), with `_resolved_parameters`
([splits.py:91-97](src/seqrec_eval/splits.py#L91-L97)), `library_provenance`
([splits.py:156-172](src/seqrec_eval/splits.py#L156-L172)) and `_stage_stats`
([splits.py:100-126](src/seqrec_eval/splits.py#L100-L126)).

**explanation:**
The record of what the split actually is (research design §6.2):

| field | content |
|---|---|
| `dataset`, `dataset_fingerprint` | which section built it; the reuse check of Step 2.2 reads the fingerprint |
| `build_parameters` | exactly what was passed to the builder |
| `resolved_build_parameters` | what the builder used after merging its registry defaults, obtained by calling its private `_resolve_args`; if that fails, the error is recorded rather than failing the build |
| `library` | the library's version, a hash of its imported source (`_source_hash`, [splits.py:129-136](src/seqrec_eval/splits.py#L129-L136)), its path, and for a git install the URL, commit and whether it is editable |
| `build_seconds` | build time |
| `val_rows_sampled` | size of the validation sample, or `null` |
| `stages` | per phase: rows, catalogue size, target pairs, how many targets repeat an item in the history and that fraction, and history length mean/p50/p90/max; plus the number of training events |
| `manifest` | the library's manifest, with the window boundaries |
| `timestamps` | added in Step 2.8 |
| `refit` | added in Step 2.9 |

## Step 2.8: recovering event times and the next-item targets

**code:**
[splits.py:254-257](src/seqrec_eval/splits.py#L254-L257) → `prepared_events`
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
`prepare` and shared with the refit step ([splits.py:254](src/seqrec_eval/splits.py#L254)).

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
is merged into `data` right away ([splits.py:257](src/seqrec_eval/splits.py#L257)), because the refit step
reads `test_next_target_matrix`.

## Step 2.9: the refit set (train + validation)

**code:**
[splits.py:258-259](src/seqrec_eval/splits.py#L258-L259) → `attach_refit`
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
   `source_end`, and from it on ([refit.py:103-114](src/seqrec_eval/refit.py#L103-L114)); rows in the
   builder's user order; histories sorted by (row, time), stable ([refit.py:120-124](src/seqrec_eval/refit.py#L120-L124));
   source and target matrices summed per (user, item) like the builder's (`_matrix`,
   [refit.py:79-85](src/seqrec_eval/refit.py#L79-L85)), and `matrix = source.maximum(target)` like `x_train`.
   Because the catalogue is fixed, the builder's repeat-until-stable filter reduces to one pass.
3. **`prove`.** The same function, given the *train* stage's catalogue and boundaries, must reproduce the
   split's `train_user_ids`, `x_train_sequences`, `train_source_sequences` and `x_train` exactly, or it
   raises `RefitAlignmentError` with the first difference ([refit.py:160-181](src/seqrec_eval/refit.py#L160-L181)).
4. **The refit set.** Then it is built with the *validation* catalogue (`val_item_ids`), source before
   `validation_target_start` and window up to `test_target_start`
   ([refit.py:192-194](src/seqrec_eval/refit.py#L192-L194)). Using the validation catalogue means a refitted
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
[splits.py:260-266](src/seqrec_eval/splits.py#L260-L266); `write_json` at
[results.py:41-45](src/seqrec_eval/results.py#L41-L45).

**explanation:**
`split_info.json` is written into the staging folder, the old split folder (if any) is removed, and the
staging folder is renamed to `work/splits/<dataset>`. Until that rename, nothing a run would load has
changed. `write_json` itself writes to a temporary name (`.<name>.<pid>.tmp`) and renames it into place, so
a process killed mid-write never leaves a truncated file. Every JSON file in the suite is written this way.

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
[cli.py:249-262](src/seqrec_eval/cli.py#L249-L262).

**explanation:**
`--threads` sets torch's CPU threads for the process ([cli.py:250-251](src/seqrec_eval/cli.py#L250-L251)).
For each selected dataset: load the prepared split ([cli.py:253](src/seqrec_eval/cli.py#L253)), run the
full-data analysis ([cli.py:255](src/seqrec_eval/cli.py#L255)), then for each selected sweep that covers
this dataset run the per-condition analysis ([cli.py:256-259](src/seqrec_eval/cli.py#L256-L259)), and drop
the split before the next dataset ([cli.py:260](src/seqrec_eval/cli.py#L260)) to free memory.

## Step 3.2: `load_split`

**code:**
`load_split(work_dir, dataset)` at [splits.py:269-284](src/seqrec_eval/splits.py#L269-L284), returning a
`Split` ([splits.py:58-84](src/seqrec_eval/splits.py#L58-L84)).

**explanation:**
`load_split` reads what `prepare` left in `work/splits/<dataset>/` and wraps it in a `Split`, the object
every step from here on receives. Its fields:

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

`eval_user_ids(phase)` ([splits.py:76-84](src/seqrec_eval/splits.py#L76-L84)) returns the user id of each
row of a phase (the `*_eval_user_ids` if the split has them, else `*_user_ids`) and checks that there is one
per target row. These are the `sample_ids` every evaluation is keyed on, which is what makes two results
pairable user by user.

## Step 3.3: what is read from disk

**code:**
[splits.py:270-277](src/seqrec_eval/splits.py#L270-L277); `load_timestamps` at
[timestamps.py:250-261](src/seqrec_eval/timestamps.py#L250-L261); `load_refit` at
[refit.py:222-231](src/seqrec_eval/refit.py#L222-L231).

**explanation:**
If `split_info.json` is missing, it stops and says to run `prepare` ([splits.py:272-273](src/seqrec_eval/splits.py#L272-L273)).
Otherwise:

1. `load_recsys_split(path)` ([splits.py:275](src/seqrec_eval/splits.py#L275)) reads the library's split
   from `data/` (the dict of Step 2.5).
2. `data.update(load_timestamps(path))` ([splits.py:276](src/seqrec_eval/splits.py#L276)) adds, if they
   exist: `x_train_timestamps`, `train_source_timestamps`, `val_source_timestamps`,
   `test_source_timestamps` (one time per event, aligned with each view's `values`), and
   `val_next_target_matrix`, `test_next_target_matrix` (the next-item targets). Without these, the
   time-decayed baseline cannot fit and `targets = "next"` cannot score.
3. `data.update(load_refit(path))` ([splits.py:277](src/seqrec_eval/splits.py#L277)) adds, if
   `refit_user_ids.npy` exists: `x_refit`, `refit_source_matrix`, `refit_target_matrix`,
   `x_refit_sequences`, `refit_source_sequences`, `x_refit_timestamps`, `refit_source_timestamps`,
   `refit_user_ids`. They sit beside the training views and are used only once `final_split` swaps them in.
4. `val_rows.npy` is loaded if it exists ([splits.py:283](src/seqrec_eval/splits.py#L283)).

## Step 3.4: `analyse_full`: paths, the tested split, the profile

**code:**
`analyse_full(protocol, work_dir, split)` at [analysis.py:246-279](src/seqrec_eval/analysis.py#L246-L279);
the setup at [analysis.py:253-260](src/seqrec_eval/analysis.py#L253-L260). `final_split` at
[splits.py:287-299](src/seqrec_eval/splits.py#L287-L299), `swap_training` at
[refit.py:234-254](src/seqrec_eval/refit.py#L234-L254).

**explanation:**
Two splits are used side by side, because searching and testing need different training data:

- `split` (the one loaded) is trained on the training window and is used to **search** each baseline on
  validation.
- `tested = final_split(protocol, split)` ([analysis.py:254](src/seqrec_eval/analysis.py#L254)) is what
  everything scored on **test** is fitted on, exactly as for the models' final runs.

**`final_split`** returns the split unchanged if `refit` is off or it is already `train+val`
([splits.py:294-295](src/seqrec_eval/splits.py#L294-L295)). It refuses an ablation condition, since a
condition must be made from the refitted split rather than the other way round
([splits.py:296-297](src/seqrec_eval/splits.py#L296-L297)). Otherwise it returns a new `Split` whose data
went through `swap_training` and whose `trained_on` is `"train+val"`. **`swap_training`** replaces every
training view with its refit counterpart: `x_train ← x_refit`, `train_source_matrix ← refit_source_matrix`,
`train_target_matrix ← refit_target_matrix`, `x_train_sequences ← x_refit_sequences`,
`train_source_sequences ← refit_source_sequences`, `train_user_ids ← refit_user_ids`, and both timestamp
arrays. `train_item_ids` becomes `val_item_ids`, so the model's catalogue is the validation catalogue; every
item is now warm (`warm_item_indices` = all, `val_cold_item_indices` empty). It fails if the split was
prepared without the refit set ([refit.py:236-240](src/seqrec_eval/refit.py#L236-L240)). Nothing is copied:
the new dict points at the arrays already loaded.

**Paths** (Appendix A has the full tree):

- `root = analysis_root(...)` ([analysis.py:253](src/seqrec_eval/analysis.py#L253), defined at
  [analysis.py:129-130](src/seqrec_eval/analysis.py#L129-L130)) is `work/analysis/<dataset>/<dataset
  fingerprint[:12]>/`. Everything of this dataset's full-data analysis lives under it, so a re-prepared
  split (new fingerprint) starts a fresh analysis.
- `profile_path(...)` ([analysis.py:255](src/seqrec_eval/analysis.py#L255), defined at
  [analysis.py:240-243](src/seqrec_eval/analysis.py#L240-L243)) is `root/profile-<evaluation key[:12]>.json`.
  The profile depends on which users are scored and on what (targets, refit, scoring version), so it is keyed
  by the evaluation key too.

**The profile** ([analysis.py:256-258](src/seqrec_eval/analysis.py#L256-L258)) is written only if missing:
`{"dataset", "trained_on", "characteristics"}`, where `characteristics(tested, None, "test_next_target_matrix")`
describes the data the tested models train on and the test inputs they are scored from. Step 3.5 explains
every field. It is the same function the ablation manipulation check uses, so what is described and what is
checked cannot drift apart.

**The diagnostic flag** ([analysis.py:259-260](src/seqrec_eval/analysis.py#L259-L260)): `other` is the
target definition the protocol did *not* choose (`other_definition`,
[evaluate.py:239-240](src/seqrec_eval/evaluate.py#L239-L240)), so `"window"` under `targets = "next"`.
`diagnose` is true when the tested split has that definition's test targets
(`target_key("test", "window")` = `test_target_matrix`, [evaluate.py:233-236](src/seqrec_eval/evaluate.py#L233-L236)),
which is always the case for `"window"`. For `targets = "window"` it would need the next-item targets, which
a split prepared with `--no-timestamps` lacks. When `diagnose` is true, every baseline and control is also
scored against the other definition's targets, as a diagnostic that never enters selection or the floor.

## Step 3.5: the profile, field by field

**code:**
`characteristics` at [ablations.py:900-938](src/seqrec_eval/ablations.py#L900-L938), with `_describe`
([ablations.py:785-803](src/seqrec_eval/ablations.py#L785-L803)), `gini`
([ablations.py:769-774](src/seqrec_eval/ablations.py#L769-L774)), `_order_stats`
([ablations.py:843-852](src/seqrec_eval/ablations.py#L843-L852)), `_time_stats`
([ablations.py:855-885](src/seqrec_eval/ablations.py#L855-L885)) and `_edits`
([ablations.py:806-840](src/seqrec_eval/ablations.py#L806-L840)).

**explanation:**
Two parts are described ([ablations.py:919-920](src/seqrec_eval/ablations.py#L919-L920)):
`train` = `x_train_sequences` (all training rows; under refit the train+validation histories) and `test` =
`test_source_sequences` restricted to `split.test_rows` (all test rows on the full data). For each part:

| field | computed as | from |
|---|---|---|
| `rows`, `events` | rows described; events in them | `_describe` |
| `history_length`, `history_length_p50`, `history_length_p90` | mean, median and 90th percentile of events per row | `_describe`, [ablations.py:926-927](src/seqrec_eval/ablations.py#L926-L927) |
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
([ablations.py:933-937](src/seqrec_eval/ablations.py#L933-L937)): the share of target entries (of the
protocol's definition) whose item is outside the training catalogue, i.e. that no model can recommend. The
first five (`history_length`, `catalogue`, `density`, `popularity_gini`, `repeat_rate`) are the five
`CHARACTERISTICS` ([ablations.py:82](src/seqrec_eval/ablations.py#L82)) the manipulation check compares.

## Step 3.6: the baseline loop

**code:**
[analysis.py:261-270](src/seqrec_eval/analysis.py#L261-L270).

**explanation:**
For every `[baselines.<name>]` in the protocol, in file order (popularity, time_popularity, replay, markov):

1. **Select** ([analysis.py:262](src/seqrec_eval/analysis.py#L262)): `select_baseline` searches the
   baseline on validation with `split` and returns the selected record (Step 3.7).
2. **Where the test result goes** ([analysis.py:263](src/seqrec_eval/analysis.py#L263)): the stem
   `<baseline dir>/test`, where the baseline dir (`_baseline_dir`, [analysis.py:133-135](src/seqrec_eval/analysis.py#L133-L135))
   is `root/baselines/<name>/<baseline fingerprint[:12]>/`. Editing the baseline's section changes the
   fingerprint and starts a new folder with a new search.
3. **How to fit it** ([analysis.py:264](src/seqrec_eval/analysis.py#L264)): `fit` is a function, not a
   fitted model: `fit_baseline(kind, selected params, tested)`, the selected setting fitted on the tested
   (refit) split. It is only called if a result is missing, so a rerun with every result cached fits nothing.
4. **What is recorded beside it** ([analysis.py:265](src/seqrec_eval/analysis.py#L265)): the baseline
   name, the params and `trained_on`.
5. **Score and cache on test** ([analysis.py:266](src/seqrec_eval/analysis.py#L266)): `_cached(stem,
   compute, record)` (Step 3.10) returns the saved result if `test.json` and `test.npz` exist; otherwise it
   fits, scores on test with the protocol's targets (`_score`, Step 3.9), saves the result and writes
   `test.done.json`.
6. **Markov's tie mask** ([analysis.py:267-268](src/seqrec_eval/analysis.py#L267-L268)): for the Markov
   baseline, `_save_tied` saves which scored users' histories end in a tie (Step 3.10).
7. **The diagnostic** ([analysis.py:269-270](src/seqrec_eval/analysis.py#L269-L270)): with `diagnose`,
   the same fitted setting is scored against the other definition's targets and cached as `test_window.*`,
   its done record marked `"targets": "window"`.

**Output per baseline:** `trial-NNN.json` (one per validation trial), `selected.json`, `test.json` +
`test.npz` + `test.done.json`, `test_window.json` + `.npz` + `.done.json`, and for Markov `tied.npy`.

## Step 3.7: `select_baseline`, the validation search

**code:**
`select_baseline` at [analysis.py:148-177](src/seqrec_eval/analysis.py#L148-L177); `baseline_params` at
[analysis.py:138-141](src/seqrec_eval/analysis.py#L138-L141); `grid_params` and `trial_params` at
[search.py:50-66](src/seqrec_eval/search.py#L50-L66).

**explanation:**
A baseline is searched exactly as a model is, so the floor is tuned as carefully as the models it bounds.

1. If `selected.json` exists, it is returned at once ([analysis.py:152-154](src/seqrec_eval/analysis.py#L152-L154)).
   The search is done once per baseline fingerprint.
2. Otherwise, for each trial `0 … trials−1` ([analysis.py:157-167](src/seqrec_eval/analysis.py#L157-L167)):
   - a finished trial's `trial-NNN.json` is read back, so an interrupted search resumes;
   - else its parameters come from `baseline_params`: for a grid, `grid_params` enumerates the choices with
     parameters sorted by name (`itertools.product`, [search.py:50-56](src/seqrec_eval/search.py#L50-L56)),
     so trial *i* is always the same combination; for random search, `trial_params` draws each parameter
     from its own stream (Step 5.2);
   - the baseline is fitted on `split` (the training window) and scored on **validation**, on the fixed
     `val_rows` sample (`_score(..., "val")`);
   - `{trial, params, val metrics, seconds}` is written to `trial-NNN.json`.
3. A trial whose validation primary metric is not a finite number stops the search with an error
   ([analysis.py:168-170](src/seqrec_eval/analysis.py#L168-L170)). A NaN can never lose a `>` comparison,
   so it would otherwise stay "best" or hide a bug.
4. The best is the highest validation `ndcg@10`; a strict `>` means ties go to the lowest trial
   ([analysis.py:171-172](src/seqrec_eval/analysis.py#L171-L172)).
5. `selected.json` = `{baseline, kind, trial, params, val, seconds}` is written and logged
   ([analysis.py:173-177](src/seqrec_eval/analysis.py#L173-L177)).

### Step 3.7.1: fitting a baseline: `KINDS`, `_build`, `fit_baseline`

**code:**
`KINDS` at [analysis.py:91-96](src/seqrec_eval/analysis.py#L91-L96), `_build` at
[analysis.py:99-105](src/seqrec_eval/analysis.py#L99-L105), `fit_baseline` at
[analysis.py:108-120](src/seqrec_eval/analysis.py#L108-L120).

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
without a config, which refuses any params ([analysis.py:101-104](src/seqrec_eval/analysis.py#L101-L104)).
An unknown parameter name fails in the config's constructor.

`fit_baseline(kind, params, split, sequences=None)` builds it, adds `timestamps = x_train_timestamps` for a
kind that needs them (with a clear error if the split has none, [analysis.py:112-117](src/seqrec_eval/analysis.py#L112-L117)),
and fits on `x_train_sequences`, or on `sequences` when given (the controls pass shuffled or reversed
histories), with `item_ids = train_item_ids` ([analysis.py:118-119](src/seqrec_eval/analysis.py#L118-L119)).
Every baseline therefore reads the ordered histories, never the matrix. Given `split`, it fits the training
window; given `tested`, the train+validation histories and the validation catalogue.

## Step 3.8: the four baselines

All four are in [baselines.py](src/seqrec_eval/baselines.py). Steps 3.8.1 to 3.8.5 cover what they share
and then each one: its code, what it predicts, and what it measures.

### Step 3.8.1: what they share

**code:**
`_SequenceBaseline` at [baselines.py:76-130](src/seqrec_eval/baselines.py#L76-L130); helpers at
[baselines.py:44-73](src/seqrec_eval/baselines.py#L44-L73).

**explanation:**
They subclass the library's `BaseSequentialRecommender`, so they rank, mask and exclude exactly as the
library's sequential models do.

- **Fitting** (`_start_fit`, [baselines.py:94-102](src/seqrec_eval/baselines.py#L94-L102)): check the input
  is `ItemSequences`, register the item vocabulary, set `n_items_` (the fitted catalogue size), and count
  `popularity_` = training events per item. Every baseline has this popularity for its lower tier.
- **Predicting** (`predict_on_batch`, [baselines.py:108-117](src/seqrec_eval/baselines.py#L108-L117)): the
  subclass's `_scores(source)` gives a dense score per (history, fitted item); `_seen_matrix`
  ([baselines.py:53-60](src/seqrec_eval/baselines.py#L53-L60)) turns the histories into a binary CSR over the
  fitted catalogue; with `exclude_seen`, seen items are masked; the library's `rank_numpy_scores` returns
  the top k as an `SRPTensor` (ranked columns and values).
- **New items in a history.** Histories arrive whole through the `WarmCatalogAdapter` (Step 3.9.3), so a
  history can contain an index ≥ `n_items_`: an item first seen after training. `_known`
  ([baselines.py:48-50](src/seqrec_eval/baselines.py#L48-L50)) marks those, and they are ignored except that
  they keep their position.
- **Two tiers** (`_two_tiers`, [baselines.py:69-73](src/seqrec_eval/baselines.py#L69-L73)). Every item with
  a positive score is rescaled into `(1, 2]` per row; every other item gets its training popularity squeezed
  into `[0, 1)` (`_tiebreak`, [baselines.py:63-66](src/seqrec_eval/baselines.py#L63-L66)). So items the
  baseline has evidence for always come first in its own order, and the rest of the list is filled by
  popularity rather than left arbitrary. The tiers are strict: a tiny decayed weight or a rare transition
  still ranks above the most popular item without evidence.
- **Saving** ([baselines.py:119-130](src/seqrec_eval/baselines.py#L119-L130)) follows the library's
  checkpoint contract; the analysis never saves them.

### Step 3.8.2: `popularity`

**code:**
`PopularityConfig` and `Popularity` at [baselines.py:137-175](src/seqrec_eval/baselines.py#L137-L175);
searched over `count` ∈ {events, users} ([protocol.toml:247-250](protocol.toml#L247-L250)).

**explanation:**
The same list for everyone: the most popular training items. With `count = "events"` an item's score is
its number of training events (`counts_ = popularity_`); with `"users"` it is the number of distinct users
who had it ([baselines.py:161-163](src/seqrec_eval/baselines.py#L161-L163)), so one heavy re-consumer
counts once. `_scores` broadcasts that single row to every user
([baselines.py:166-167](src/seqrec_eval/baselines.py#L166-L167)). **It measures** how far you get knowing
nothing about the user. A model that does not beat it has learned nothing personal.

### Step 3.8.3: `time_popularity`

**code:**
`TimeDecayedPopularityConfig` and `TimeDecayedPopularity` at [baselines.py:182-234](src/seqrec_eval/baselines.py#L182-L234);
searched over `half_life_days` loguniform [0.25, 365] with 20 random trials
([protocol.toml:252-255](protocol.toml#L252-L255)).

**explanation:**
Popularity in which each event counts `0.5 ** (age / half_life)`, age in days measured from the last
training event ([baselines.py:218-220](src/seqrec_eval/baselines.py#L218-L220)). A short half-life means
"what is popular right now"; a long one approaches plain popularity. Measuring age from any later moment
would multiply every weight by the same factor and leave the ranking unchanged. It needs the recovered
timestamps and checks there is one per training event ([baselines.py:211-216](src/seqrec_eval/baselines.py#L211-L216)).
An item whose events all decayed below floating-point precision falls to the popularity tier. **It
measures** how much of the next item is a current trend rather than all-time popularity.

### Step 3.8.4: `replay`

**code:**
`ReplayConfig` and `Replay` at [baselines.py:241-285](src/seqrec_eval/baselines.py#L241-L285);
searched over `order` ∈ {recency, frequency} ([protocol.toml:257-260](protocol.toml#L257-L260)).

**explanation:**
Recommends the user's own history back, then popular items after it. `_scores`
([baselines.py:265-285](src/seqrec_eval/baselines.py#L265-L285)) starts every row at the popularity
tier, finds for each (user, item) in the history its most recent position from the end (1 = the last event,
[baselines.py:271-276](src/seqrec_eval/baselines.py#L271-L276)), and lifts those items above the tier:

- `"recency"`: score `1 + 1/position`, so the most recent item first;
- `"frequency"`: score `1 + count + recency/2`, so the most repeated item first, with recency (always < 1
  after halving) only separating equal counts ([baselines.py:280-282](src/seqrec_eval/baselines.py#L280-L282)).

Under `exclude_seen = true` (ML-20M, Amazon) everything it would replay is masked, so it reduces to
popularity. **It measures** how much of the next item is re-consumption. It only differs from popularity
where `exclude_seen = false` (Music4All, Yambda, OTTO).

### Step 3.8.5: `markov`

**code:**
`MarkovConfig` and `MarkovChain` at [baselines.py:292-350](src/seqrec_eval/baselines.py#L292-L350);
no settings, so one trial ([protocol.toml:262-263](protocol.toml#L262-L263)).

**explanation:**
First-order Markov: the next item from the last one only, P(next | last).

- **fit** ([baselines.py:315-326](src/seqrec_eval/baselines.py#L315-L326)): count every transition between
  adjacent events of the same history (`same_row` excludes the jump from one user's last event to the next
  user's first), including self-transitions a → a, then normalise each row of the item × item count matrix
  to probabilities.
- **score** ([baselines.py:328-338](src/seqrec_eval/baselines.py#L328-L338)): take each history's last
  event; if it is a known item, its row of transition probabilities is the score; if the history is empty or
  ends in a new item, all scores are zero and the whole list falls to popularity.

**It measures** the simplest use of order. It is also the instrument of the sequence-signal analysis
(Step 3.11): compared with itself trained on shuffled or reversed histories, it shows whether order carries
information at all.

## Step 3.9: scoring: `_score` → `evaluate_phase`

**code:**
`_score` at [analysis.py:123-126](src/seqrec_eval/analysis.py#L123-L126); `evaluate_phase` at
[evaluate.py:280-329](src/seqrec_eval/evaluate.py#L280-L329). The models use the same function
(Step 5.5).

**explanation:**
`_score` picks the rows (`split.val_rows` for validation, `split.test_rows` for test, which is `None` =
all on the full data), and calls `evaluate_phase` with `family = "sequence"` (baselines read histories) and
the dataset's `exclude_seen`. `evaluate_phase` is where every number in the suite comes from:

### Step 3.9.1: guards and the target definition

[evaluate.py:288-293](src/seqrec_eval/evaluate.py#L288-L293). `targets` is `"primary"` (the protocol's
definition), `"next"` or `"window"` explicitly, or `"new"` (the protocol's targets minus items already in the
history). A split trained on `train+val` refuses to score validation: that model has seen the validation
window, so its score would be leaked ([evaluate.py:290-292](src/seqrec_eval/evaluate.py#L290-L292)).

### Step 3.9.2: which users are scored

[evaluate.py:294-297](src/seqrec_eval/evaluate.py#L294-L297); `scored_rows` at
[evaluate.py:207-217](src/seqrec_eval/evaluate.py#L207-L217) and `recommendable_next` at
[evaluate.py:196-204](src/seqrec_eval/evaluate.py#L196-L204).

For next-item targets, only users whose next target contains at least one item inside the model's training
catalogue are scored. A next item first seen after training, or deleted by the builder (an empty row, Step
2.8.3), cannot be recommended by any model: the user would score 0 for all of them, carry nothing for a
comparison, and only pull every mean down. Under refit the catalogue is the validation catalogue, so items
first seen in validation count as recommendable at test. Those rows are intersected with any given sample
(`val_rows`, `test_rows`). For window targets every row is kept (`None`).

### Step 3.9.3: the adapter and the source

[evaluate.py:298](src/seqrec_eval/evaluate.py#L298); `phase_inputs` at
[evaluate.py:267-273](src/seqrec_eval/evaluate.py#L267-L273), `phase_model` at
[evaluate.py:251-256](src/seqrec_eval/evaluate.py#L251-L256), `phase_source` at
[evaluate.py:259-264](src/seqrec_eval/evaluate.py#L259-L264); cr's `WarmCatalogAdapter`
([cr models/cold_start.py:209](../../recombee/compresso-recsys/src/compresso_recsys/models/cold_start.py#L209)).

The model was fitted on the training catalogue; the phase's catalogue is larger (it appends the items
first seen in later windows). `WarmCatalogAdapter(model, train_item_ids, phase_item_ids)` hands the model its
own item space and maps its ranked columns back into the phase's catalogue, so new items remain valid
targets that no model can ever recommend, equally for every model. The source is
`{phase}_source_sequences` for a sequence model (passed whole: a new item keeps its position and the model
reads it as unknown) or `{phase}_source_matrix` for a matrix model, projected onto the training columns by
`adapter.align_source` (a matrix model has no column for a new item).

### Step 3.9.4: targets, ids, rows

[evaluate.py:299-308](src/seqrec_eval/evaluate.py#L299-L308). The target matrix is `phase_targets`
([evaluate.py:243-248](src/seqrec_eval/evaluate.py#L243-L248)): `{phase}_next_target_matrix` for `"next"`,
`{phase}_target_matrix` for `"window"`, with a clear error if the next-item targets were never prepared. For
`"new"`, `new_item_targets` ([evaluate.py:224-230](src/seqrec_eval/evaluate.py#L224-L230)) removes targets
already in the history. The sample ids are `split.eval_user_ids(phase)`. `{phase}_seen_matrix` exists only on
ablation conditions (Step 10.5). If rows were chosen, source, targets, ids and seen matrix are all cut to
them together (`_take`, [evaluate.py:276-277](src/seqrec_eval/evaluate.py#L276-L277)).

### Step 3.9.5: excluding seen items

[evaluate.py:309-312](src/seqrec_eval/evaluate.py#L309-L312); `ExcludeSeenPolicy` at
[evaluate.py:74-193](src/seqrec_eval/evaluate.py#L74-L193).

The library's evaluator calls `predict_on_batch(source, k=...)` and nothing else, so `exclude_seen` has to
be bound to the model beforehand. On the full data (no seen matrix) the policy just forwards the call with
the dataset's `exclude_seen` ([evaluate.py:164-165](src/seqrec_eval/evaluate.py#L164-L165)), and the model
masks the items in the history it was given. On an ablation condition whose inputs were thinned, the seen
matrix holds the user's *original* history, because truncating what a model reads does not change what the
user has seen. Then the policy asks the model for `k +` (the batch's longest history) items with its own
filter off, drops the seen ones, and keeps the first `k` in the model's order
([evaluate.py:169-193](src/seqrec_eval/evaluate.py#L169-L193)). Because a batch carries no user ids, the seen
rows are found by a running offset, and three checks make sure that offset is right (review H33): each batch
must equal the source's next rows (`_check_rows`, [evaluate.py:118-135](src/seqrec_eval/evaluate.py#L118-L135)),
every item a row reads must be in its seen row (`_check_batch`, [evaluate.py:137-155](src/seqrec_eval/evaluate.py#L137-L155)),
and after the evaluation every row must have been used exactly once (`finish`,
[evaluate.py:157-161](src/seqrec_eval/evaluate.py#L157-L161), called at [evaluate.py:328](src/seqrec_eval/evaluate.py#L328)).
A breach raises `BatchAlignmentError`.

### Step 3.9.6: the library's evaluator and what comes back

[evaluate.py:313-327](src/seqrec_eval/evaluate.py#L313-L327); cr's `evaluate_recommender`
([cr evaluation.py:800](../../recombee/compresso-recsys/src/compresso_recsys/evaluation.py#L800)) and
`EvaluationResult` ([cr evaluation.py:131-160](../../recombee/compresso-recsys/src/compresso_recsys/evaluation.py#L131-L160)).

`evaluate_recommender` asks the model for the top `max(cutoffs)` = 20 per row, in batches of
`eval_batch_size`, and computes every metric of `build_metrics` ([evaluate.py:220-221](src/seqrec_eval/evaluate.py#L220-L221):
the seven metric classes at the three cutoffs, 21 values such as `ndcg@10`). Rows without any target are
skipped (`valid = target_counts > 0`, [cr evaluation.py:623](../../recombee/compresso-recsys/src/compresso_recsys/evaluation.py#L623)).
It returns an `EvaluationResult`:

| field | content |
|---|---|
| `metrics` | mean of each metric over the scored rows, e.g. `{"ndcg@10": 0.0831, ...}` |
| `per_user` | per metric, one value per scored row; what every paired comparison and bootstrap uses |
| `sample_ids` | the user id of each scored row, in order; two results are paired only if these match |
| `n_rows`, `n_scored_rows` | rows given; rows with at least one target |
| `required_k` | 20 |
| `metadata` | the suite's: `phase`, `exclude_seen`, `targets`, `definition`, `rows_sampled`, and `rows_unrecommendable_next` (how many asked-for rows were left out in Step 3.9.2) |
| `target_fingerprint` | a hash of the targets scored against; the statistics refuse to pair two results scored on different targets |

## Step 3.10: caching: `_cached`, `save_evaluation`, the tie mask

**code:**
`_cached` at [analysis.py:180-189](src/seqrec_eval/analysis.py#L180-L189); `save_evaluation`,
`load_evaluation`, `evaluation_exists` at [results.py:52-92](src/seqrec_eval/results.py#L52-L92);
`_save_tied` at [analysis.py:192-199](src/seqrec_eval/analysis.py#L192-L199) with `tied_split`
([analysis.py:412-421](src/seqrec_eval/analysis.py#L412-L421)) and `last_pair_tied`
([ablations.py:888-897](src/seqrec_eval/ablations.py#L888-L897)).

**explanation:**
`_cached(stem, compute, record)` is the whole caching rule of the analysis: if `<stem>.json` and
`<stem>.npz` both exist, load and return them; otherwise call `compute()` (fit + score), save, and write
`<stem>.done.json` = the record plus the seconds taken and the test metrics. A rerun computes only what is
missing.

`EvaluationResult` has no file format of its own, so `save_evaluation` stores it as a pair: `<stem>.npz`
with every per-user array (`per_user__ndcg@10`, …) and the `sample_ids`, and `<stem>.json` with metrics,
row counts, `required_k`, metadata and the target fingerprint. `load_evaluation` rebuilds the object, so a
report can pair users long after the process that scored them has exited.

For Markov, `_save_tied` writes `tied.npy` beside the result: for each scored user, whether the last two
events of their test history share a timestamp. `last_pair_tied` computes it per test row from
`test_source_timestamps`, and `tied_split` maps it onto the result's `sample_ids`. Without timestamps nothing
is written. The mask is used by the report to split Markov's score into "history ends in a tie" and "ends in
a real gap" (Step 4.1).

## Step 3.11: the sequence-signal controls

**code:**
[analysis.py:272-279](src/seqrec_eval/analysis.py#L272-L279); `CONTROLS` at
[analysis.py:87-88](src/seqrec_eval/analysis.py#L87-L88); `fit_control` at
[analysis.py:220-228](src/seqrec_eval/analysis.py#L220-L228); `shuffled_within_users` and
`reversed_histories` at [analysis.py:207-217](src/seqrec_eval/analysis.py#L207-L217);
`_controls_fingerprint` at [analysis.py:231-233](src/seqrec_eval/analysis.py#L231-L233);
`_control_stream` at [analysis.py:307-312](src/seqrec_eval/analysis.py#L307-L312).

**explanation:**
Two controls ask whether order carries information:

- `markov_shuffled`: Markov fitted on training histories whose events are shuffled within each user
  (`np.lexsort` by row, then a random key, [analysis.py:209](src/seqrec_eval/analysis.py#L209)). Which items
  a history holds is unchanged; the order is gone. A large drop from Markov is order the data carries.
- `markov_backwards`: Markov fitted on each history reversed ([analysis.py:215-216](src/seqrec_eval/analysis.py#L215-L216)),
  so a → b is counted as b → a. A small drop means much of what Markov learned is co-occurrence, not
  direction.

Both are fitted on `tested` and scored on the same test users as Markov, with the same caching as the
baselines (and the same `test_window` diagnostic). They live at `root/controls/<control>/<controls
fingerprint[:12]>/`. The fingerprint covers the split, `search_seed`, `CONTROLS_VERSION` (2) and the result
settings. The shuffle's random stream is `_stream(search_seed, "control", dataset, "full")`
(`_control_stream`); an ablation reference uses the same stream, so it is the same control.

## Step 3.12: `analyse_sweep`, the analysis at every ablation condition

**code:**
[cli.py:256-259](src/seqrec_eval/cli.py#L256-L259) → `analyse_sweep` at
[analysis.py:315-345](src/seqrec_eval/analysis.py#L315-L345), with `_condition_scorers`
([analysis.py:292-304](src/seqrec_eval/analysis.py#L292-L304)) and `_condition_stem`
([analysis.py:286-289](src/seqrec_eval/analysis.py#L286-L289)).

**explanation:**
Each sweep that lists this dataset gets the same analysis at every condition.

1. **Scorers** ([analysis.py:321](src/seqrec_eval/analysis.py#L321)): for each baseline, its setting
   selected on the full data (`select_baseline` again; it returns the cached `selected.json`) and a
   fingerprint of the baseline's fingerprint plus those params; for each control, the controls fingerprint.
   Baselines are **not retuned per level**, as the models keep their stage-1 configuration, so a gap cannot
   move because only one side was retuned.
2. **The base** ([analysis.py:322](src/seqrec_eval/analysis.py#L322)): `tested = final_split(split)`; every
   condition is a transform of the refitted data, as the models' ablation runs are.
3. **`run(condition, name)`** ([analysis.py:324-331](src/seqrec_eval/analysis.py#L324-L331)): record the
   condition's characteristics against `tested` (`record_characteristics`, Step 10.7); then for each
   scorer, fit it on the condition's training data and score it on the condition's test users, cached at
   `ablations/<sweep>/<dataset>/<condition fingerprint[:12]>/analysis/<scorer>/<fp[:12]>/<condition>/test`,
   with Markov's `tied.npy`.
4. **`pending(name)`** ([analysis.py:333-336](src/seqrec_eval/analysis.py#L333-L336)): true if any scorer's
   result is missing. A condition is only built (a transform can be expensive) when something is missing.
5. **The order** ([analysis.py:338-345](src/seqrec_eval/analysis.py#L338-L345)): the full-data reference
   first (`build_reference`, Step 10.4), then every level × data seed (`build_condition`, Step 10.4). The
   conditions and their test users are exactly those the models' ablation runs use (Part 10).

---

# Part 4: `seqrec-eval analysis-report`

## Step 4.1: building the report

**code:**
[cli.py:264-273](src/seqrec_eval/cli.py#L264-L273) → `build_analysis_report` at
[analysis_report.py:184-200](src/seqrec_eval/analysis_report.py#L184-L200) → `dataset_analysis` at
[analysis_report.py:89-173](src/seqrec_eval/analysis_report.py#L89-L173); results read by `full_results`
([analysis.py:352-377](src/seqrec_eval/analysis.py#L352-L377)).

**explanation:**
Nothing is computed from data here; the report reads what `analyse` cached. `full_results` collects, for
one dataset: the profile, every baseline's `selected.json` and test result (and Markov's tie mask), both
controls' test results, and the diagnostic (`test_window`) results. Per dataset the report has four sections:

1. **Data profile** ([analysis_report.py:98](src/seqrec_eval/analysis_report.py#L98)): `profile_table`
   ([analysis_report.py:69-72](src/seqrec_eval/analysis_report.py#L69-L72)) prints the fields of Step 3.5 in
   the order of `PROFILE_ROWS` ([analysis_report.py:30-49](src/seqrec_eval/analysis_report.py#L30-L49)), for
   the training data and the test inputs, formatted by `_format` ([analysis_report.py:52-66](src/seqrec_eval/analysis_report.py#L52-L66)).
2. **Baselines and the floor** ([analysis_report.py:100-121](src/seqrec_eval/analysis_report.py#L100-L121)):
   per baseline its kind, selected setting, number of trials, validation `ndcg@10`, and test `ndcg@10`,
   `recall@10`, `hit_rate@10`. The **floor** is `floor_of` ([analysis.py:404-409](src/seqrec_eval/analysis.py#L404-L409)):
   the baseline with the highest **test** primary metric, marked in the table. Choosing it on test makes the
   floor the strongest bound a model could be held to.
3. **Sequence signal** ([analysis_report.py:123-156](src/seqrec_eval/analysis_report.py#L123-L156)): Markov's
   test score and each control's, with the paired difference (control − Markov) and its 95% bootstrap
   interval from the library's `compare_models` without correction (`paired`,
   [analysis_report.py:75-77](src/seqrec_eval/analysis_report.py#L75-L77)). These differences are descriptive:
   read, not tested. Below it, Markov and both controls split by the tie mask: users whose history ends in a
   tie and users whose history ends in a real gap; a slice under `MIN_SLICE_USERS` = 100 users is not shown
   (`_slice_mean`, [analysis_report.py:84-86](src/seqrec_eval/analysis_report.py#L84-L86)).
4. **The other target definition** ([analysis_report.py:158-172](src/seqrec_eval/analysis_report.py#L158-L172)):
   every scorer's test `ndcg@10` on the primary targets beside the diagnostic ones.

The CSV (`analysis.csv`) has one row per baseline, control and diagnostic with all metrics
([analysis_report.py:193-199](src/seqrec_eval/analysis_report.py#L193-L199)).

---

# Part 5: `seqrec-eval search --device cuda:0`

## Step 5.1: the CLI branch

**code:**
[cli.py:201-229](src/seqrec_eval/cli.py#L201-L229) (the `search` path).

**explanation:**
Every selected model is looked up in the registry first ([cli.py:204-205](src/seqrec_eval/cli.py#L204-L205)),
so a model in the protocol but not in `models.py` fails before any split is loaded. Then per dataset the
split is loaded as prepared, trained on the training window, with no `final_split`
([cli.py:208](src/seqrec_eval/cli.py#L208)). Per model, `plan_trials` gives the trial specs
([cli.py:213](src/seqrec_eval/cli.py#L213)) and each is passed to `execute`
([cli.py:223-226](src/seqrec_eval/cli.py#L223-L226)). The statuses are counted and logged at the end, and
the exit code is 1 if any run failed.

## Step 5.2: planning the trials and drawing the configurations

**code:**
`plan_trials` at [runner.py:98-105](src/seqrec_eval/runner.py#L98-L105); `RunSpec` at
[runner.py:67-87](src/seqrec_eval/runner.py#L67-L87); `trial_params` at
[search.py:59-66](src/seqrec_eval/search.py#L59-L66), `_draw` at [search.py:34-47](src/seqrec_eval/search.py#L34-L47).

**explanation:**
A `RunSpec` describes one run: `dataset`, `model`, `kind` (`"trial"`, `"final"`, or for ablations
`"reference"`/`"rescore"`), `index` (trial number, or the seed of a final), `seed`, `params`, `fingerprint`
(the run fingerprint), `source_trial` (for a final: the trial it came from) and `condition` (only for
ablation runs). Its folder is `work/runs/<dataset>/<model>/<run fingerprint[:12]>/trial-007`
(`name`, `directory`, `run_root`: [runner.py:80-91](src/seqrec_eval/runner.py#L80-L91)).

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
`execute` at [runner.py:382-412](src/seqrec_eval/runner.py#L382-L412); `_check_training` at
[runner.py:288-294](src/seqrec_eval/runner.py#L288-L294); `RunLock` at
[runner.py:228-263](src/seqrec_eval/runner.py#L228-L263).

**explanation:**
Every run of the suite (trials, finals, ablation runs) goes through `execute`. It returns a status string:

1. `_check_training` refuses the wrong split: a trial must get a split trained on `"train"`; under refit,
   everything else must get `"train+val"`.
2. `done.json` exists → `"cached"`. A run is finished exactly when this file exists.
3. `failed.json` exists and no `--retry-failed` → `"failed-before"`, so a configuration that runs out of
   memory does not retry itself all weekend.
4. The lock: `RunLock.acquire` creates `.lock` with `O_CREAT | O_EXCL` (atomic: only one process can
   create it) and writes host, pid and start time into it. If another live process holds it →
   `"running-elsewhere"`. That is how two processes (one per GPU) running the same command divide the work.
   A lock is stale, and is reclaimed, if its process no longer exists on this host, or if it is unreadable
   and older than 60 seconds (`_stale`, [runner.py:232-241](src/seqrec_eval/runner.py#L232-L241)). A lock
   written on another host is never reclaimed automatically.
5. `spec.json` is written, the run is logged, `_execute` runs it, and on success any old `failed.json` is
   removed.
6. On any exception, `failed.json` gets the spec, host, device, error and traceback; status `"failed"`.
7. Always: release the lock and empty the CUDA cache.

## Step 5.4: `_execute` for a trial

**code:**
`_execute` at [runner.py:297-379](src/seqrec_eval/runner.py#L297-L379).

**explanation:**
1. **Checks** ([runner.py:299-305](src/seqrec_eval/runner.py#L299-L305)): the registered family must match
   the protocol's; `_check_condition` ([runner.py:276-285](src/seqrec_eval/runner.py#L276-L285)) refuses a
   stage-1 spec on an ablation split and vice versa.
2. **The record** ([runner.py:307-311](src/seqrec_eval/runner.py#L307-L311)): spec, fingerprint, host,
   device, `trained_on`, training catalogue size, and `code_provenance()`
   ([splits.py:139-153](src/seqrec_eval/splits.py#L139-L153)): the library's and the suite's version and
   source hash, so the report can tell when results come from different code builds (review H05, H10).
3. **`max_items`** ([runner.py:314-320](src/seqrec_eval/runner.py#L314-L320)): under refit a trial checks
   the *validation* catalogue, the one its finals will fit, so a model is never searched in full and then
   skipped at the finals. Too large → `done.json` with `status: "skipped"` and the reason.
4. **Fit** ([runner.py:331-337](src/seqrec_eval/runner.py#L331-L337)): seed `random`, numpy and torch
   (`_seed_everything`, [runner.py:270-273](src/seqrec_eval/runner.py#L270-L273)); build the trainer from the
   registry (Step 5.5) with `n_items = len(train_item_ids)`; fit on `x_train_sequences` (sequence family) or
   `x_train` (matrix family); record `fit_seconds`.
5. **Validation** ([runner.py:340-344](src/seqrec_eval/runner.py#L340-L344)): `evaluate_phase(trainer,
   split, "val", rows=split.val_rows)` (Step 3.9, with the model's own family), saved as `val.json` +
   `val.npz`, metrics into the record. A trial never touches test.
6. **Finish** ([runner.py:370-379](src/seqrec_eval/runner.py#L370-L379)): `eval_seconds`, the trainer's
   training `history` if it has one, peak GPU memory, `status: "done"`, `done.json`.

**Output:** `runs/<dataset>/<model>/<fp>/trial-NNN/` with `spec.json`, `val.json`, `val.npz`, `done.json`
(or `failed.json`).

## Step 5.5: the models

**code:**
[models.py:46-105](src/seqrec_eval/models.py#L46-L105); the search spaces at
[protocol.toml:176-235](protocol.toml#L176-L235).

**explanation:**
`register(name, family, cls)` ([models.py:58-64](src/seqrec_eval/models.py#L58-L64)) puts a builder in
`REGISTRY` as a `ModelSpec` (name, family, trainer class for reloading, builder). `model_spec(name)`
([models.py:67-71](src/seqrec_eval/models.py#L67-L71)) looks it up. Every builder takes `(params, n_items,
device, seed)` and returns an unfitted trainer following the library's contract
(`fit(data, item_ids=...)`, then `predict_on_batch(source, k=..., exclude_seen=...)`):

| name | family | library trainer | built at | notes |
|---|---|---|---|---|
| `popularity` | matrix | `PopularityBaseline` | [models.py:77-79](src/seqrec_eval/models.py#L77-L79) | the library's popularity model; no space, 1 trial. The suite's own popularity *baseline* (Step 3.8.2) is separate |
| `ease` | matrix | `EASE` | [models.py:82-84](src/seqrec_eval/models.py#L82-L84) | `l2` searched; `max_items = 40000` (dense item × item) |
| `elsa` | matrix | `ELSATrainer` | [models.py:87-89](src/seqrec_eval/models.py#L87-L89) | factors, batch, lr, epochs searched; gets device and seed |
| `gru` | sequence | `SimpleRNNTrainer` | [models.py:92-100](src/seqrec_eval/models.py#L92-L100) | `max_history_length` is taken out of the params and given to the `SequenceBatcher`, which is where SimpleRNN takes its context from |
| `sasrec` | sequence | `SASRecTrainer` | [models.py:103-105](src/seqrec_eval/models.py#L103-L105) | `max_history_length` is a config field |

Progress bars are off because runs are unattended. `max_history_length` is the one parameter name every
sequential model shares.

---

# Part 6: `seqrec-eval final --device cuda:0`

## Step 6.1: the CLI branch

**code:**
[cli.py:201-229](src/seqrec_eval/cli.py#L201-L229) (the `final` path: [cli.py:209-222](src/seqrec_eval/cli.py#L209-L222)).

**explanation:**
The same loop as `search`, with two differences. The split is `final_split(load_split(...))`
([cli.py:210](src/seqrec_eval/cli.py#L210)): trained on train + validation under refit (Step 3.4). And the
specs come from `plan_finals` ([cli.py:216-218](src/seqrec_eval/cli.py#L216-L218)); if a model is not ready
(search unfinished, open failures), the reason is logged, counted as `not-ready`, and the next model runs.

## Step 6.2: `summarize_trials`: what the search produced

**code:**
`summarize_trials` at [runner.py:132-167](src/seqrec_eval/runner.py#L132-L167); `TrialSummary` at
[runner.py:108-125](src/seqrec_eval/runner.py#L108-L125).

**explanation:**
Per planned trial, from its folder: `failed.json` without `done.json` → failed (with the error); no
`done.json` → still open; `status: "skipped"` → skipped (with the reason); otherwise its validation primary
metric is read, and a non-finite value counts as a failure ([runner.py:158-163](src/seqrec_eval/runner.py#L158-L163),
review H07). The best is the highest value; strict `>` in trial order, so ties go to the lowest trial. Any
previously accepted failures are read from `accepted_failures.json` ([runner.py:141-143](src/seqrec_eval/runner.py#L141-L143));
`unaccepted` is the failures not in it.

## Step 6.3: `plan_finals`: selecting the configuration

**code:**
`plan_finals` at [runner.py:170-211](src/seqrec_eval/runner.py#L170-L211).

**explanation:**
1. Every trial skipped (catalogue too large) → no finals, an empty list ([runner.py:183-184](src/seqrec_eval/runner.py#L183-L184)).
2. Open failures → refuse ([runner.py:185-193](src/seqrec_eval/runner.py#L185-L193)): selecting around a
   failed trial would give the model a smaller, silently different search. Fix and rerun
   (`search --retry-failed`), or, for a failure that cannot be fixed (out of memory at a corner of the
   space), `final --accept-failed`, which writes `accepted_failures.json` (the trials, errors, time and host;
   [runner.py:194-197](src/seqrec_eval/runner.py#L194-L197)) so the report lists them.
3. Fewer trials finished than planned → refuse unless `--allow-incomplete`, because selecting early breaks
   the equal budget across models ([runner.py:198-204](src/seqrec_eval/runner.py#L198-L204)).
4. No successful trial → refuse.
5. Otherwise one `RunSpec` per protocol seed: `kind = "final"`, `index = seed`, `seed`, the best trial's
   `params` and fingerprint, `source_trial` = the best trial's number ([runner.py:207-211](src/seqrec_eval/runner.py#L207-L211)).
   Folder: `runs/<dataset>/<model>/<fp>/final-seed<seed>`.

## Step 6.4: `_execute` for a final run

**code:**
[runner.py:297-379](src/seqrec_eval/runner.py#L297-L379), the parts a final takes.

**explanation:**
The same steps as a trial (Step 5.4), except:

- `max_items` is judged on `len(train_item_ids)`, which under refit is the validation catalogue.
- The trainer is seeded with the final's own seed and fitted on the swapped training views, i.e. `x_refit`
  or `x_refit_sequences` with the validation catalogue.
- **No validation score** under refit: `split.trained_on` is `"train+val"`, so the validation block is
  skipped ([runner.py:340](src/seqrec_eval/runner.py#L340)). The report takes the validation value from the
  search. Without refit a final scores validation too.
- **Test** ([runner.py:346-350](src/seqrec_eval/runner.py#L346-L350)): `evaluate_phase(..., "test",
  rows=None)`, so every test user whose next item is recommendable (Step 3.9.2). Saved as `test.json` +
  `test.npz`.
- **Diagnostics** of a stage-1 final only ([runner.py:351-363](src/seqrec_eval/runner.py#L351-L363)): the
  other target definition as `test_window.*`, and, for datasets with `new_item_diagnostic = true`
  (Music4All, Yambda, OTTO), the targets not already in the history with `exclude_seen = True` as
  `test_new.*`.
- **The model** is saved as `model.zip` ([runner.py:364-369](src/seqrec_eval/runner.py#L364-L369)). Latency
  and every ablation reference reload it. A failed save is recorded (`model_saved: false` and the error) but
  does not fail the run.

**Output:** `final-seedS/` with `spec.json`, `test.*`, `test_window.*`, `test_new.*` (some datasets),
`model.zip`, `done.json`.

---

# Part 7: `seqrec-eval status`

## Step 7.1

**code:**
[cli.py:231-233](src/seqrec_eval/cli.py#L231-L233) → `status_table` at
[report.py:129-147](src/seqrec_eval/report.py#L129-L147).

**explanation:**
One table row per dataset and model: whether the split is prepared, trials finished (done + skipped) of
planned, failed trials, runs whose lock is held by a live process right now (`held_by_other`,
[runner.py:262-263](src/seqrec_eval/runner.py#L262-L263)), finals done of seeds (or "skipped"), and the best
trial with its validation `ndcg@10`. It reads the run folders only.

---

# Part 8: `seqrec-eval report`

## Step 8.1: the CLI branch

**code:**
[cli.py:235-247](src/seqrec_eval/cli.py#L235-L247) → `build_report` at
[report.py:325-346](src/seqrec_eval/report.py#L325-L346).

**explanation:**
`build_report` writes a header (protocol, version, primary metric, and whether finals were refitted) and
one section per dataset from `dataset_report`, and collects a CSV of every final run's metrics. The CLI
appends the CPU latency table if any `latency.json` exists (`latency_table`,
[latency.py:119-132](src/seqrec_eval/latency.py#L119-L132)), and writes `reports/stage1.md` and
`reports/final_metrics.csv`.

## Step 8.2: one dataset's section

**code:**
`dataset_report` at [report.py:212-322](src/seqrec_eval/report.py#L212-L322).

**explanation:**
1. **The split** (`_split_section`, [report.py:154-179](src/seqrec_eval/report.py#L154-L179)): the resolved
   build settings and, per phase, rows, catalogue, target pairs, repeat-target share and history p50/p90,
   all from `split_info.json`.
2. **Per model** ([report.py:219-256](src/seqrec_eval/report.py#L219-L256)): skipped models are noted; for
   the others the finished finals are loaded (`load_final_evaluations`, [runner.py:415-427](src/seqrec_eval/runner.py#L415-L427)),
   and the table shows trials done, the selected trial, its validation value, the number of seeds, and each
   test metric as mean ± sd over seeds. Notes flag open trial failures (⛔), accepted failures with their
   reasons, and failed final seeds. The seeds' per-user values are averaged into one result per model
   (`mean_over_seeds`, [report.py:50-67](src/seqrec_eval/report.py#L50-L67)), which requires every seed to
   have scored the same users on the same targets. Averaging per user keeps the comparison paired on users
   and treats training noise as part of the model's measurement.
3. **Who was scored** ([report.py:260-265](src/seqrec_eval/report.py#L260-L265)): how many test users the
   next-item metrics cover, and how many were left out because no model can recommend their next item.
4. **Code builds** ([report.py:268](src/seqrec_eval/report.py#L268); `code_builds` and `_builds_note` at
   [report.py:182-209](src/seqrec_eval/report.py#L182-L209)): every `done.json` records its code; one build
   is stated, several are flagged ⚠.
5. **New-item diagnostic** ([report.py:270-272](src/seqrec_eval/report.py#L270-L272)), where it was run.
6. **Against the reference model** ([report.py:274-290](src/seqrec_eval/report.py#L274-L290)): the
   library's `compare_models` ([cr stats.py:851](../../recombee/compresso-recsys/src/compresso_recsys/stats.py#L851))
   compares every model with `--reference` (default `elsa`) on test `ndcg@10`: the difference, the relative
   difference, a paired bootstrap percentile 95% interval, and a p-value from a two-sided paired sign-flip
   randomisation test, Holm-corrected within this dataset. Significance is read from p; the interval
   describes the size. The family is one dataset, so a dataset's conclusions do not pay for the number of
   datasets in the study.
7. **The other target definition** ([report.py:292-303](src/seqrec_eval/report.py#L292-L303)): each model's
   primary value beside its `test_window` value.
8. **Against the floor** ([report.py:305-321](src/seqrec_eval/report.py#L305-L321)): the floor (Part 4) is
   read from the analysis, and every model is compared with it in a second Holm family. "Beats the floor"
   means significant *and* positive. Without an analysis the report says to run `analyse`.

---

# Part 9: `seqrec-eval latency`

## Step 9.1

**code:**
[cli.py:319-333](src/seqrec_eval/cli.py#L319-L333) → `benchmark` at
[latency.py:56-116](src/seqrec_eval/latency.py#L56-L116); `parse_cores` at
[latency.py:36-47](src/seqrec_eval/latency.py#L36-L47).

**explanation:**
The requirement is model inference latency on CPU, 100 ms at P95, characterised against history length.
Per dataset the tested split is used (`final_split`, [cli.py:322](src/seqrec_eval/cli.py#L322)), and per
model:

1. Settings from `[latency]`: history bins, requests per bin (200), warm-up requests (20)
   ([latency.py:59-62](src/seqrec_eval/latency.py#L59-L62)).
2. Reload the first seed's `final-seed<seed>/model.zip` on CPU; a missing model is logged and skipped
   ([latency.py:64-67](src/seqrec_eval/latency.py#L64-L67), [cli.py:326-328](src/seqrec_eval/cli.py#L326-L328)).
3. Optionally pin the process to `--cores` and set torch threads ([latency.py:69-71](src/seqrec_eval/latency.py#L69-L71)).
4. Build the adapter and the test source once, untimed (`phase_inputs`), wrapped in `ExcludeSeenPolicy`
   ([latency.py:75-76](src/seqrec_eval/latency.py#L75-L76)).
5. Draw up to `requests_per_bin` test users per history-length bin and interleave the bins in random order,
   so drift over the run hits every bin alike ([latency.py:81-90](src/seqrec_eval/latency.py#L81-L90)).
6. Run the warm-up requests, then time only the model call for each single-user request against the full
   test catalogue, top 20 ([latency.py:96-103](src/seqrec_eval/latency.py#L96-L103)).
7. Write `latency.json` beside the model: overall and per-bin n, mean, P50, P95, P99, max, plus catalogue
   size, threads, cores, host, CPU, torch version ([latency.py:105-115](src/seqrec_eval/latency.py#L105-L115)).

---

# Part 10: `seqrec-eval ablate --device cuda:0`

An ablation sweep varies one data characteristic at a series of levels and holds everything else fixed.
Every model keeps its stage-1 configuration at every level; nothing is searched again, so a difference
between levels comes from the data, not from tuning ([ablations.py:1-54](src/seqrec_eval/ablations.py#L1-L54)).
The sweeps in the protocol ([protocol.toml:290-348](protocol.toml#L290-L348)):

| sweep | transform | levels | scope | knee |
|---|---|---|---|---|
| `history_length` | history_length | 1, 2, 5, … 1000 | all (refit per level) | yes |
| `history_length_inference` | history_length | the same | inference (rescore only) | yes |
| `density` | density | 0.1, 0.25, 0.5, 0.75 | all | yes |
| `shuffle` | shuffle | block 2, 10, all | all | no |
| `catalogue_top` | catalogue (top) | 0.1, 0.25, 0.5, 0.75 | all | no |
| `catalogue_stratified` | catalogue (stratified, 10 strata) | the same | all | no |
| `repeat_removal` | repeat_removal | 0.25, 0.5, 1.0 | all | yes |

## Step 10.1: the CLI branch

**code:**
[cli.py:275-291](src/seqrec_eval/cli.py#L275-L291).

**explanation:**
Models are checked against the registry first. Per dataset, the sweeps that cover it are collected
([cli.py:282-284](src/seqrec_eval/cli.py#L282-L284)); a dataset with none is skipped without loading. The
split is loaded and refitted once, `final_split(load_split(...))` ([cli.py:285](src/seqrec_eval/cli.py#L285)),
because every condition is a transform of the data the stage-1 finals were fitted on. Then `_ablate` runs
each sweep with the models it covers ([cli.py:286-288](src/seqrec_eval/cli.py#L286-L288)).

## Step 10.2: `_ablate`: one sweep on one dataset

**code:**
`_ablate` at [cli.py:337-382](src/seqrec_eval/cli.py#L337-L382).

**explanation:**
1. **Plans** ([cli.py:344-355](src/seqrec_eval/cli.py#L344-L355)): `plan_ablation` per model (Step 10.3).
   A model whose stage-1 search or finals are not ready is logged as `not-ready`; a model skipped in stage 1
   has an empty plan and drops out.
2. **Users** ([cli.py:356-360](src/seqrec_eval/cli.py#L356-L360)): for the catalogue sweeps each condition
   has its own users (logged); for every other sweep the fixed test users are computed now
   (`fixed_test_rows`, Step 10.4) and their number logged.
3. **`run(condition, name)`** ([cli.py:362-368](src/seqrec_eval/cli.py#L362-L368)): record the condition's
   characteristics (Step 10.7), then `execute` every model's specs for that condition name (Step 10.8).
4. **`pending(name)`** ([cli.py:370-373](src/seqrec_eval/cli.py#L370-L373)): the condition's
   characteristics file is missing, or some run has no `done.json`. A finished condition is not even built,
   so resuming a finished sweep costs no transform.
5. **Order** ([cli.py:375-382](src/seqrec_eval/cli.py#L375-L382)): the reference (`full`) first, then every
   level, and within a level every data seed (one per protocol seed for a stochastic transform, a single
   `None` otherwise, `data_seeds`, [ablations.py:657-660](src/seqrec_eval/ablations.py#L657-L660)). The
   condition name is the level label, plus `/seed<S>` for a stochastic one (`condition_name`,
   [ablations.py:663-665](src/seqrec_eval/ablations.py#L663-L665)).

## Step 10.3: `plan_ablation`: which runs each condition has

**code:**
`plan_ablation` at [ablations.py:986-1026](src/seqrec_eval/ablations.py#L986-L1026); fingerprints at
[ablations.py:633-654](src/seqrec_eval/ablations.py#L633-L654).

**explanation:**
It starts from the stage-1 finals, `plan_finals(...)` ([ablations.py:996](src/seqrec_eval/ablations.py#L996)),
so an ablation refuses exactly when a final would (Step 6.3). Each spec is a copy of a stage-1 final spec
with another `kind`, fingerprint and condition:

- **Fingerprints.** `condition_fingerprint` ([ablations.py:633-645](src/seqrec_eval/ablations.py#L633-L645))
  covers the dataset fingerprint, the evaluation key, the sweep's `transform`, `levels`, `options` and
  `scope` (`_RESULT_KEYS`, [ablations.py:94](src/seqrec_eval/ablations.py#L94); `datasets`, `models` and the
  report-only keys are left out), and the transform's version. It names the sweep's folder on a dataset,
  `ablations/<sweep>/<dataset>/<cfp[:12]>` (`ablation_root`). `ablation_fingerprint`
  ([ablations.py:648-650](src/seqrec_eval/ablations.py#L648-L650)) adds the model's stage-1 run fingerprint
  and names its runs, `…/runs/<model>/<afp[:12]>/`.
- **Reference** ([ablations.py:1007-1013](src/seqrec_eval/ablations.py#L1007-L1013)): one spec per seed,
  `kind = "reference"`, folder `…/full/final-seed<S>`, and a `checkpoint` pointing at the stage-1
  `runs/<dataset>/<model>/<fp>/final-seed<S>/model.zip`. It reloads the stage-1 model, so the reference is
  exactly the model stage 1 reported.
- **Levels** ([ablations.py:1014-1025](src/seqrec_eval/ablations.py#L1014-L1025)): for a deterministic
  transform every seed's final runs at every level; for a stochastic one, condition `<level>/seed<S>` runs
  only seed S's final, so each seed has its own subsample. With scope `"all"` the kind is `"final"` (the
  stage-1 params refitted with the stage-1 seed on the condition's training data); with scope `"inference"`
  it is `"rescore"` and carries the checkpoint (the stage-1 model reloaded and scored on transformed inputs).
  Folder: `…/<level label>/final-seed<S>`.

## Step 10.4: which test users each condition is scored on

**code:**
`fixed_test_rows` at [ablations.py:745-762](src/seqrec_eval/ablations.py#L745-L762) with
`_eligible_test_rows` ([ablations.py:668-675](src/seqrec_eval/ablations.py#L668-L675)); `own_test_rows` at
[ablations.py:710-718](src/seqrec_eval/ablations.py#L710-L718); `per_condition_users` at
[ablations.py:721-723](src/seqrec_eval/ablations.py#L721-L723); `build_reference` at
[ablations.py:726-732](src/seqrec_eval/ablations.py#L726-L732); `build_condition` at
[ablations.py:735-742](src/seqrec_eval/ablations.py#L735-L742).

**explanation:**
Levels are only comparable if they are scored on the same users. Two rules:

- **One fixed set** (every sweep whose transform keeps the targets). A test row is eligible if its history is
  not empty and its next target holds a recommendable item, the same rule as `scored_rows` (Step 3.9.2).
  `fixed_test_rows` takes the rows eligible on the full data **and** at every level and data seed, which
  means building every condition once ([ablations.py:751-754](src/seqrec_eval/ablations.py#L751-L754)). It
  saves them as `test_rows.npy` in the sweep's folder (atomically), and every later call reads that file.
  An empty set is an error.
- **Its own users** (the catalogue sweeps, whose transform removes targets: `changes_targets = True`). One
  set cannot survive every random catalogue: a user whose next target is a known item survives all levels
  only by chance, so the set would shrink to users whose targets are all unseen, who score 0 for every model
  (review H01b). So each condition is scored on its own rows: a known next target still in *its* catalogue,
  and a history item left (`own_test_rows`). Models are then compared within a condition, not across levels.

`build_reference` wraps the unmodified split as condition `full` with those rows
(`reference_condition`, [ablations.py:705-707](src/seqrec_eval/ablations.py#L705-L707)).
`build_condition` applies the transform (Step 10.5) and attaches the rows: the fixed set, or the
condition's own.

## Step 10.5: `apply_condition`: transforming the split

**code:**
`apply_condition` at [ablations.py:678-702](src/seqrec_eval/ablations.py#L678-L702).

**explanation:**
1. The transform is looked up; a stochastic transform needs a data seed and a deterministic one refuses
   it ([ablations.py:683-686](src/seqrec_eval/ablations.py#L683-L686)).
2. **The random stream** ([ablations.py:688-690](src/seqrec_eval/ablations.py#L688-L690)):
   `_stream(search_seed, "ablation", sweep, dataset, <level or "nested">, data_seed)`. A nested transform
   (the catalogue) leaves the level out, so every level of one seed shares one random draw and a smaller
   catalogue is part of every larger one.
3. **The phases it may edit** ([ablations.py:691](src/seqrec_eval/ablations.py#L691)): scope `"all"` →
   train, val and test; scope `"inference"` → val and test only, so the training data stays untouched.
4. **Apply** ([ablations.py:692-693](src/seqrec_eval/ablations.py#L692-L693)): the transform gets the data
   dict, the level, the stream, the sweep's options, the dataset's event value (`set_all_values_to`) and the
   phases, and returns a new dict. The original split is never modified.
5. **What the user has seen** ([ablations.py:696-700](src/seqrec_eval/ablations.py#L696-L700)): if a val or
   test source matrix was rebuilt and the item space is unchanged, the original source matrix is stored as
   `{phase}_seen_matrix`. Scoring then excludes the user's whole original history, not only the shortened
   input (Step 3.9.5): a film rated five years ago is still not a recommendation.
6. The result is the same `Split` with the new data, the test rows and `condition = {sweep, level, label,
   data_seed}` ([ablations.py:701-702](src/seqrec_eval/ablations.py#L701-L702)). It keeps `trained_on =
   "train+val"`.

## Step 10.6: the transform registry and `keep_events`

**code:**
`Transform` at [ablations.py:115-140](src/seqrec_eval/ablations.py#L115-L140); `register` at
[ablations.py:146-162](src/seqrec_eval/ablations.py#L146-L162); `keep_events` at
[ablations.py:288-330](src/seqrec_eval/ablations.py#L288-L330), with `count_matrix`
([ablations.py:275-285](src/seqrec_eval/ablations.py#L275-L285)), `_subset`
([ablations.py:232-236](src/seqrec_eval/ablations.py#L232-L236)) and `_carry_times`
([ablations.py:239-252](src/seqrec_eval/ablations.py#L239-L252)).

**explanation:**
A transform is a function registered with a declaration of what it does:

| field | meaning |
|---|---|
| `version` | bumped when its behaviour changes; enters the condition fingerprint, so old conditions are not reused |
| `stochastic(options)` | whether it draws a subsample (then one per data seed) |
| `target` | the characteristic it is meant to move, or `None` (shuffle: none should move) |
| `expected(options)` | what else it moves by construction; the manipulation check does not flag these |
| `validate` | checks levels and options (called by `check_ablation`, Step 0.4) |
| `scopes` | which scopes it supports |
| `changes_targets` | whether test targets change between levels (→ per-condition users, no level-vs-full test) |
| `position(level)` | the level's place on the knee's axis, larger = closer to the full data; `None` = no knee |
| `direction` | which way the target must move (−1 = down); the other way is flagged as a bug |
| `nested` | whether the levels of one seed share one draw |

Transforms that only drop events build a boolean mask per phase and hand it to `keep_events`, which rebuilds
every view from the kept events exactly as the builder would have:

- **train:** the training window is the train-stage source followed by its target, so an event is "source"
  if its position is before the source's length ([ablations.py:308](src/seqrec_eval/ablations.py#L308)).
  Both matrices are recounted from the kept events on each side (`count_matrix`: each event's value summed
  per pair), and `x_train` is again their maximum. The kept events' times go with them.
- **val / test:** the source sequences are subset and the source matrix recounted.
- The masks are stored under `_kept` for the manipulation check's `span_kept` (Step 10.7).

### Step 10.6.1: `history_length` (version 3)

**code:**
[ablations.py:357-387](src/seqrec_eval/ablations.py#L357-L387).

**explanation:**
Level n is a context of n items. Every training history keeps its last **n + 1** events, every input
history (val, test) its last **n** ([ablations.py:384-386](src/seqrec_eval/ablations.py#L384-L386)). A
training history of n + 1 events teaches contexts of n items (its last event is only ever a target), so a
model is trained and served with the same context, and level 1 still has something to learn (one item → the
next). Deterministic. Target `history_length`; declared to move `density` and `repeat_rate` too; position =
n. The sequential models also keep their stage-1 `max_history_length`, so at levels at or above it their
input at inference is the same as with full data (the report says so, Step 11.2). With scope `"inference"`
(the `history_length_inference` sweep) only the val and test inputs are cut, and the stage-1 models are
rescored: how much history a trained model needs at serving time.

### Step 10.6.2: `density` (version 2)

**code:**
[ablations.py:394-433](src/seqrec_eval/ablations.py#L394-L433).

**explanation:**
Each history keeps a random fraction p of its events: `max(1, round(p × length))` of them, chosen uniformly
among its positions by ranking random keys within the row, in their original order
([ablations.py:419-422](src/seqrec_eval/ablations.py#L419-L422)). A thinned history still reaches as far
back as the original, unlike truncation. With `keep_catalogue` (default on), a training item that would lose
all its events gets one back, chosen at random, so the training catalogue does not shrink
([ablations.py:423-431](src/seqrec_eval/ablations.py#L423-L431)). Stochastic. Target `density`; declared to
move `history_length` and `repeat_rate`; position = p.

### Step 10.6.3: `repeat_removal` (version 1)

**code:**
[ablations.py:440-464](src/seqrec_eval/ablations.py#L440-L464).

**explanation:**
A repeat event is one whose item already occurred earlier in the same history (`_first_occurrence`,
[ablations.py:267-272](src/seqrec_eval/ablations.py#L267-L272)). Each repeat is removed with probability q;
first occurrences are never removed, so which items a history holds (and so catalogue and density) does not
change, and every remaining repeat is still a repeat. Stochastic. Target `repeat_rate`; declared to move
`history_length`; position = 1 − q, so less removal is closer to the full data and the knee is the most
removal that still costs less than δ.

### Step 10.6.4: `shuffle` (version 1)

**code:**
[ablations.py:471-500](src/seqrec_eval/ablations.py#L471-L500).

**explanation:**
Events are reordered within each history: with `"all"` the whole history, with block size b only within
consecutive blocks of b events, which destroys order at short range and keeps it at long range
([ablations.py:493-497](src/seqrec_eval/ablations.py#L493-L497)). Each event keeps its own time. It does not
use `keep_events`: the matrices stay exactly as they were, because a matrix row is a set. So the matrix
models, and the time-decayed baseline, are controls that should score as with the full data. Stochastic.
Target `None`: none of the five characteristics should move. No position, so no knee.

### Step 10.6.5: `catalogue` (version 2): top and stratified

**code:**
[ablations.py:507-626](src/seqrec_eval/ablations.py#L507-L626); `_stratified_order` at
[ablations.py:520-536](src/seqrec_eval/ablations.py#L520-L536).

**explanation:**
Reduces the training catalogue to k items: a fraction of it, or an item count
([ablations.py:574](src/seqrec_eval/ablations.py#L574)).

- `strategy = "top"`: the k items with the most training events. Deterministic.
- `strategy = "stratified"`: items ranked by popularity and cut into `strata` equal groups; within a group of
  size s each item gets a random slot r and a place (r + U)/s; sorting every item by place interleaves the
  groups, so any prefix holds each stratum in proportion. Taking a prefix of this one order at every level
  makes the levels nested. Stochastic, one order per seed.

A removed item leaves **every** phase ([ablations.py:583-602](src/seqrec_eval/ablations.py#L583-L602)): its
events leave the training data and the input histories (the remaining items are re-addressed in the smaller
item space), its targets leave the val and test target matrices and the next-item targets, and it leaves the
item space, so no model can recommend it. Items first seen after training stay, as new items no model can
recommend, exactly as in the full data. Training users left with no events are dropped from the training
views ([ablations.py:604-614](src/seqrec_eval/ablations.py#L604-L614)); the catalogue partitions are updated
and the stale list-shaped index fields cleared ([ablations.py:616-624](src/seqrec_eval/ablations.py#L616-L624)).
Target `catalogue`; declared to move `history_length` and `density`, and for `"top"` also `popularity_gini`
(keeping only the popular items flattens the distribution; stratified is meant not to). Scope `"all"` only;
`changes_targets`, so per-condition users; no knee.

## Step 10.7: the condition's characteristics

**code:**
`record_characteristics` at [ablations.py:941-951](src/seqrec_eval/ablations.py#L941-L951) →
`characteristics(split, original, …)` (Step 3.5) with `_edits` at
[ablations.py:806-840](src/seqrec_eval/ablations.py#L806-L840).

**explanation:**
Before a condition's runs, its profile is written once to `…/conditions/<name>.json` (`full.json`,
`10.json`, `0.5/seed0.json`, …) as `{sweep, level, label, data_seed, characteristics}`. The original is the
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
[cli.py:362-368](src/seqrec_eval/cli.py#L362-L368) → `execute` (Step 5.3) → `_execute` at
[runner.py:297-379](src/seqrec_eval/runner.py#L297-L379).

**explanation:**
The same machinery as stage 1: resumable, locked, failures recorded. What differs for an ablation spec:

- `_check_condition` ([runner.py:276-285](src/seqrec_eval/runner.py#L276-L285)) requires the spec's
  sweep, label and data seed to match the split's condition, and `test_rows` to be set. A run cannot be
  scored on another condition's data.
- `reference` and `rescore` reload the stage-1 `model.zip` (`registered.cls.load`,
  [runner.py:325-330](src/seqrec_eval/runner.py#L325-L330)); `final` refits the stage-1 params with the
  stage-1 seed on the condition's training data.
- No validation: nothing selects on it any more ([runner.py:340](src/seqrec_eval/runner.py#L340)).
- Test on `split.test_rows`, the condition's users ([runner.py:346-350](src/seqrec_eval/runner.py#L346-L350)),
  saved as `test.*`.
- No diagnostics and no saved model ([runner.py:351](src/seqrec_eval/runner.py#L351)).

**Output:** `ablations/<sweep>/<dataset>/<cfp>/` with `test_rows.npy` (fixed-set sweeps),
`conditions/*.json`, `runs/<model>/<afp>/<condition>/final-seedS/{spec.json, test.json, test.npz, done.json}`,
and from `analyse` the `analysis/` subfolder.

---

# Part 11: `seqrec-eval ablation-report`

## Step 11.1: the CLI branch

**code:**
[cli.py:293-306](src/seqrec_eval/cli.py#L293-L306) → `build_ablation_report` at
[ablation_report.py:524-567](src/seqrec_eval/ablation_report.py#L524-L567).

**explanation:**
Per selected sweep it writes `reports/ablation-<sweep>.md`, the gap plot `ablation-<sweep>-gap.png` (if
matplotlib is installed and there is a gap), and up to three CSVs: `-metrics` (every run's metrics),
`-gap` (the gap rows plus the floor rows as model `floor`), `-analysis` (the baselines and controls at every
condition).

## Step 11.2: the header

**code:**
[ablation_report.py:527-546](src/seqrec_eval/ablation_report.py#L527-L546); `_transform_notes` at
[ablation_report.py:570-582](src/seqrec_eval/ablation_report.py#L570-L582).

**explanation:**
Datasets and models are restricted to those the sweep covers. The header states the transform and its
version, the levels and options, the scope (refit per level, or rescoring), the seeds and whether each has
its own subsample, and that `full` is the stage-1 model itself. For a catalogue sweep it says the targets
change with the level. For `history_length` it notes which sequential models also keep a
`max_history_length`.

## Step 11.3: one dataset: setup and the manipulation check

**code:**
`dataset_ablation` at [ablation_report.py:292-447](src/seqrec_eval/ablation_report.py#L292-L447); the
manipulation section `_characteristics_section` at [ablation_report.py:245-289](src/seqrec_eval/ablation_report.py#L245-L289)
with `manipulation_check` at [ablations.py:954-979](src/seqrec_eval/ablations.py#L954-L979).

**explanation:**
The report-only settings are read with their defaults ([ablation_report.py:296-303](src/seqrec_eval/ablation_report.py#L296-L303)):
`manipulation_tolerance` 0.05, `knee_margin` 0.10, `min_level_users` 1,000. The columns are `full` and the
level labels (seeds of one level share a column).

The **manipulation check** is read before any result. The full data's five characteristics are shown in
absolute terms, every condition as the relative change `(after − before) / |before|`, for the training data
and the test inputs. `manipulation_check` marks ⚠ any characteristic that is neither the target nor declared
in `expected_to_move` and moved by more than the tolerance, and ⛔ a target that moved against the
transform's direction (truncation cannot lengthen histories): that is a bug or an unforeseen interaction,
and the sweep should not be read until it is understood. Beside them: train rows changed, test rows
changed, test span kept.

## Step 11.4: the results table

**code:**
[ablation_report.py:318-358](src/seqrec_eval/ablation_report.py#L318-L358); `pool_over_seeds` at
[report.py:70-108](src/seqrec_eval/report.py#L70-L108).

**explanation:**
Per model, `plan_ablation` gives the specs and `_load` ([ablation_report.py:232-238](src/seqrec_eval/ablation_report.py#L232-L238))
the finished test results; each becomes a CSV row. For a catalogue sweep the target fingerprints are set
aside, since the targets differ between conditions by construction ([ablation_report.py:334-335](src/seqrec_eval/ablation_report.py#L334-L335)).
A column is **averaged only once every seed has finished** ([ablation_report.py:343-345](src/seqrec_eval/ablation_report.py#L343-L345)),
by `pool_over_seeds`: if the seeds scored the same users it is `mean_over_seeds`; if not (stratified
catalogue, a different subsample per seed), every user scored in at least one seed is kept and averaged over
the seeds that scored them, so a user counts once. The table shows mean ± sd over seeds per level and a row
with the number of users scored. A level under `min_level_users` is **descriptive only**.

## Step 11.5: each level against the full data

**code:**
[ablation_report.py:360-383](src/seqrec_eval/ablation_report.py#L360-L383); `holm` at
[ablation_report.py:97-105](src/seqrec_eval/ablation_report.py#L97-L105).

**explanation:**
Not run for a catalogue sweep: each level has its own users, so there is nothing to pair across levels.
Otherwise each model's levels are compared with its `full` by `compare_models` without correction, and Holm
is applied to all p-values of the sweep on this dataset at once: one family per sweep and dataset. Table:
difference (level − full), 95% bootstrap interval, adjusted p, significant.

## Step 11.6: the knee

**code:**
[ablation_report.py:385-415](src/seqrec_eval/ablation_report.py#L385-L415); `find_knee` at
[ablation_report.py:183-230](src/seqrec_eval/ablation_report.py#L183-L230); `noninferiority_p` at
[ablation_report.py:108-129](src/seqrec_eval/ablation_report.py#L108-L129); `noninferiority_power` at
[ablation_report.py:132-157](src/seqrec_eval/ablation_report.py#L132-L157).

**explanation:**
Only for transforms with a `position` (history_length, density, repeat_removal), and only for a model whose
whole curve has finished. The per-user values of every column must be on the same users.

1. δ = `knee_margin` × the model's mean on the **full data** ([ablation_report.py:217](src/seqrec_eval/ablation_report.py#L217)).
2. Levels are tested from the one closest to the full data downwards ([ablation_report.py:218](src/seqrec_eval/ablation_report.py#L218)),
   each against the full data (not against the best-looking level, which is lucky by construction).
3. The test (`noninferiority_p`) is a one-sided paired sign-flip randomisation test of "the loss is smaller
   than δ": each per-user difference (level − full) is shifted by +δ, and 9,999 random sign flips give the
   null distribution of the mean at the boundary. p = (1 + number of flips ≥ observed) / 10,000, so it is
   never 0. A small p is evidence the loss is below δ.
4. Testing stops at the first level with p > α = 0.05 ([ablation_report.py:225-227](src/seqrec_eval/ablation_report.py#L225-L227)).
   The **knee** starts at `full` and moves down only through levels that passed, so it is the last level
   that passed, or `full` when the first test fails. This is the fixed-sequence procedure: hypotheses
   tested in an order fixed in advance, stopping at the first non-rejection, keep the family-wise error at α
   without correction.
5. **Power** per level (`noninferiority_power`): Φ(δ·√n / s − 1.645), the chance the test shows "within δ"
   for a level that truly loses nothing, from that level's per-user spread s over n users. The report shows
   the lowest and highest over the levels. Where it is low, a knee at `full` means the test could not tell,
   not that the reduced levels are worse.

The random stream of each test is `_stream(search_seed, "knee", sweep, dataset, model, level)`.

## Step 11.7: the gap

**code:**
[ablation_report.py:417-443](src/seqrec_eval/ablation_report.py#L417-L443).

**explanation:**
The quantity the sweeps are about. Per column, the best non-sequential (matrix) model is the one with the
highest test `ndcg@10` at that level, chosen on test, so the gap is descriptive and carries no test. Each
sequential model's gap is its paired difference from that model, with a paired bootstrap 95% interval from
`compare_models`. Each row records the users scored and whether that is below `min_level_users`.

## Step 11.8: baselines and sequence signal at every condition

**code:**
`_analysis_section` at [ablation_report.py:450-499](src/seqrec_eval/ablation_report.py#L450-L499);
`condition_results` at [analysis.py:380-401](src/seqrec_eval/analysis.py#L380-L401);
`mean_over_subsamples` at [analysis.py:424-430](src/seqrec_eval/analysis.py#L424-L430).

**explanation:**
What `analyse` cached per condition (Step 3.12) is read back and pooled over subsamples, per column:
every baseline's test `ndcg@10`, the floor (the strongest baseline at that level, with its name), both
controls, and Markov split into tied-end and real-gap users (slices under 100 users left out). For the plot
the floor is also expressed on the gap's scale: floor minus the best matrix model at that level
([ablation_report.py:485-491](src/seqrec_eval/ablation_report.py#L485-L491)).

## Step 11.9: the gap plot

**code:**
`gap_figure` at [plots.py:60-133](src/seqrec_eval/plots.py#L60-L133); `level_order` at
[ablation_report.py:514-521](src/seqrec_eval/ablation_report.py#L514-L521).

**explanation:**
One panel per dataset (at most three per row), each on its own y-axis. The x-axis lists the levels evenly
spaced, along the knee's axis where there is one, with `full` last. Each sequential model is a line with
its bootstrap interval as a band, hollow markers where a level is descriptive only, a dashed zero line
("no better than the best non-sequential model"), and the floor as a grey dashed line. Lines are labelled
at their ends (`_label_ends`, [plots.py:43-57](src/seqrec_eval/plots.py#L43-L57)), so identity never rests
on colour alone. Returns PNG bytes, or `None` without matplotlib (`available`, [plots.py:35-40](src/seqrec_eval/plots.py#L35-L40)).

---

# Part 12: `seqrec-eval repeat-strata`

## Step 12.1

**code:**
[cli.py:308-317](src/seqrec_eval/cli.py#L308-L317) → `build_strata_report` at
[strata.py:179-198](src/seqrec_eval/strata.py#L179-L198) → `dataset_strata` at
[strata.py:98-176](src/seqrec_eval/strata.py#L98-L176); settings at [strata.py:45-61](src/seqrec_eval/strata.py#L45-L61)
from `[repeat_strata]` ([protocol.toml:350-355](protocol.toml#L350-L355)).

**explanation:**
A slice of the stage-1 results, not a sweep: nothing is refitted or rescored.

1. **Settings** (`settings`): `history_bins` (lower edges; the last bin is open), `repeat_bins` (0 to 1),
   `min_users` (100) and `n_resamples` (1,999), each checked.
2. **Results** ([strata.py:103-115](src/seqrec_eval/strata.py#L103-L115)): per model, the finals averaged
   over seeds, only for models with every seed finished. Matrix models are the non-sequential side,
   sequence models the candidates. All must have scored the same users.
3. **Per user** ([strata.py:121-127](src/seqrec_eval/strata.py#L121-L127)): each scored test user's history
   length and repeat share (`repeat_share`, [strata.py:64-68](src/seqrec_eval/strata.py#L64-L68): the share
   of its events that repeat an item earlier in the same history). The test histories are the same with or
   without refit, so the split is loaded as prepared.
4. **Cells** ([strata.py:129-138](src/seqrec_eval/strata.py#L129-L138)): users binned by both
   (`_bins`, [strata.py:71-78](src/seqrec_eval/strata.py#L71-L78)). Binning by length too matters: long
   histories hold more repeats simply because they are long. The first table counts users per cell.
5. **Gap per cell** ([strata.py:143-158](src/seqrec_eval/strata.py#L143-L158)): the best non-sequential model
   is chosen **once per dataset** on all test users, so a small cell cannot pick its own comparator. In each
   cell with at least `min_users` users: every model's mean, and each sequential model's mean gap with a
   bootstrap percentile 95% interval (`_bootstrap`, [strata.py:88-95](src/seqrec_eval/strata.py#L88-L95); stream
   `search_seed, "strata", dataset, model, cell`).
6. **Output:** a gap grid per sequential model, `reports/repeat-strata.md` and `.csv`. The cells are
   observational (users with many repeats differ in more than their repeats), so no test is made; the
   manipulated counterpart is the `repeat_removal` sweep.

---

# Appendix A: everything the suite writes

```
work/
├── splits/<dataset>/                                   prepare (Part 2)
│   ├── manifest.json, data/…                           the library's split
│   ├── split_info.json                                 the record (Step 2.7)
│   ├── val_rows.npy                                    fixed validation sample (Step 2.6)
│   ├── x_train_timestamps.npy, train_source_…, val_source_…, test_source_timestamps.npy   (Step 2.8)
│   ├── val_next_target_matrix.npz, test_next_target_matrix.npz                            (Step 2.8)
│   └── x_refit*.npz|npy, refit_source_*, refit_target_matrix.npz, refit_user_ids.npy       (Step 2.9)
├── analysis/<dataset>/<dataset fp[:12]>/               analyse (Part 3)
│   ├── profile-<evaluation key[:12]>.json
│   ├── baselines/<name>/<baseline fp[:12]>/
│   │   ├── trial-NNN.json, selected.json               validation search
│   │   ├── test.json, test.npz, test.done.json         selected setting, refitted, on test
│   │   ├── test_window.json, .npz, .done.json          the other target definition
│   │   └── tied.npy                                    Markov only
│   └── controls/<markov_shuffled|markov_backwards>/<controls fp[:12]>/test*, test_window*
├── runs/<dataset>/<model>/<run fp[:12]>/               search, final (Parts 5, 6)
│   ├── trial-NNN/  spec.json, val.json, val.npz, done.json | failed.json, .lock while running
│   ├── final-seedS/  spec.json, test.*, test_window.*, test_new.*, model.zip, done.json, latency.json
│   └── accepted_failures.json                          final --accept-failed
├── ablations/<sweep>/<dataset>/<condition fp[:12]>/    ablate, analyse (Parts 10, 3)
│   ├── test_rows.npy                                   fixed-set sweeps only
│   ├── conditions/full.json, <level>.json, <level>/seedS.json
│   ├── runs/<model>/<ablation fp[:12]>/<full|level>/final-seedS/  spec.json, test.*, done.json
│   └── analysis/<scorer>/<fp[:12]>/<condition>/test.*, test.done.json, tied.npy
└── reports/
    ├── analysis.md, analysis.csv                       analysis-report
    ├── stage1.md, final_metrics.csv                    report
    ├── ablation-<sweep>.md, -gap.png, -metrics.csv, -gap.csv, -analysis.csv
    └── repeat-strata.md, repeat-strata.csv
```

Two writing rules hold everywhere: JSON is written to a temporary file and renamed into place
([results.py:41-45](src/seqrec_eval/results.py#L41-L45)), and a run is finished exactly when its
`done.json` exists.

# Appendix B: fingerprints and keys

| name | covers | names | defined at |
|---|---|---|---|
| dataset fingerprint | dataset section, `max_val_users`, `search_seed` | `split_info.json`, `analysis/<dataset>/<fp>` | [protocol.py:218-224](src/seqrec_eval/protocol.py#L218-L224) |
| run fingerprint | result settings, dataset section, model section | `runs/<dataset>/<model>/<fp>` | [protocol.py:226-232](src/seqrec_eval/protocol.py#L226-L232) |
| baseline fingerprint | result settings, dataset section, baseline section | `analysis/…/baselines/<name>/<fp>` | [protocol.py:194-200](src/seqrec_eval/protocol.py#L194-L200) |
| evaluation key | `targets`, `refit`, scoring version | the profile file; part of the condition fingerprint | [protocol.py:210-216](src/seqrec_eval/protocol.py#L210-L216) |
| controls fingerprint | dataset fingerprint, `search_seed`, `CONTROLS_VERSION`, result settings | `analysis/…/controls/<control>/<fp>` | [analysis.py:231-233](src/seqrec_eval/analysis.py#L231-L233) |
| condition fingerprint | dataset fingerprint, evaluation key, sweep transform/levels/options/scope, transform version | `ablations/<sweep>/<dataset>/<fp>` | [ablations.py:633-645](src/seqrec_eval/ablations.py#L633-L645) |
| ablation fingerprint | condition fingerprint, stage-1 run fingerprint | `…/runs/<model>/<fp>` | [ablations.py:648-650](src/seqrec_eval/ablations.py#L648-L650) |
| condition scorer fingerprint | baseline fingerprint + selected params | `…/analysis/<baseline>/<fp>` | [analysis.py:297-298](src/seqrec_eval/analysis.py#L297-L298) |

"Result settings" are the `[protocol]` keys except `eval_batch_size`, plus `SCORING_VERSION`
([protocol.py:35-44](src/seqrec_eval/protocol.py#L35-L44), [protocol.py:207-208](src/seqrec_eval/protocol.py#L207-L208)).
Versions that invalidate caches when code changes: `SCORING_VERSION` (2), `NEXT_TARGETS_VERSION` (2,
triggers a timestamp refresh at the next `prepare`), `CONTROLS_VERSION` (2), each transform's `version`.
Not fingerprinted on purpose: `eval_batch_size`, `[latency]`, an ablation's `datasets`, `models`,
`expected_to_move`, `manipulation_tolerance`, `knee_margin`, `min_level_users` (they choose what runs or how
the report reads it, not what a result is).

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
| `val_seen_matrix`, `test_seen_matrix` | `apply_condition` (Step 10.5) | only on conditions whose inputs were thinned |
| `_kept` | `keep_events`, catalogue (Step 10.6) | only on conditions; read by `_edits` |

# Appendix D: the tests

Run with `.venv/bin/python -m pytest` from the repository root. They use tiny synthetic data only.

| file | tests | covers |
|---|---|---|
| [tests/test_smoke.py](tests/test_smoke.py) | 23 | the whole workflow on a synthetic dataset registered with the real library builder: `prepare` through `report` |
| [tests/test_ablations.py](tests/test_ablations.py) | 61 | every transform end to end and its defining properties, the fixed and per-condition users, the knee, the reports |
| [tests/test_baselines.py](tests/test_baselines.py) | 18 | the four baselines and the profile's timing fields, on histories small enough to check by hand |
| [tests/test_evaluate.py](tests/test_evaluate.py) | 9 | the seen-item mask of ablation conditions and the evaluator batch order it relies on (H33) |
| [tests/test_plots.py](tests/test_plots.py) | 3 | the gap plot and its level order |

(Counts are `def test_` functions; parametrised tests run more cases.)
