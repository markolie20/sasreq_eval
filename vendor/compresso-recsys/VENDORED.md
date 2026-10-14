# Vendored compresso-recsys

A modified copy of [compresso-recsys](https://github.com/zombak79/compresso-recsys) (Apache-2.0, see
`LICENSE`), kept here so the evaluation suite runs on exactly the library it was built and reviewed
against, without pushing the local changes to the library. `uv sync` installs it (the suite's
`pyproject.toml`: `[tool.uv.sources]`). Refresh it with `scripts/vendor-cr.sh`; never edit it by hand.
The script adds one thing of its own: a `[tool.uv]` `cache-keys` at the end of `pyproject.toml`, so
`uv sync` reinstalls the copy whenever its source changes (by default uv watches `pyproject.toml` alone).

| | |
|---|---|
| Copied | 2026-10-06 13:44:23 from `/home/mark/Documents/recombee/compresso-recsys` |
| Version | `0.3.7+trainusers` |
| Branch | `local-temporal-train-all-users` at `6547522` (`65475227e226f0eabae5f0471d3813ad86f22f9e`) |
| Uncommitted changes copied | 13 file(s) |
| Upstream compared against | `origin/main` at `1016bfe` |

## Changes from upstream

Every file below differs from upstream; each changed Python file starts with a notice saying so. The
full difference is `CHANGES.patch`. What the changes do is recorded in the suite's DECISIONS.md:
training on every user (`temporal_train_users`, entry 4 and before), bounded training memory
(`loss_chunk_elements`, entry 19), and the BERT4Rec work of the local branch.

- `README.md`
- `pyproject.toml`
- `src/compresso_recsys/builder.py`
- `src/compresso_recsys/datasets/amazon2023.py`
- `src/compresso_recsys/models/__init__.py`
- `src/compresso_recsys/models/bert4rec/__init__.py`
- `src/compresso_recsys/models/bert4rec/config.py`
- `src/compresso_recsys/models/bert4rec/model.py`
- `src/compresso_recsys/models/bert4rec/trainer.py`
- `src/compresso_recsys/models/ease.py`
- `src/compresso_recsys/models/elsa.py`
- `src/compresso_recsys/models/sasrec.py`
- `src/compresso_recsys/models/simple_rnn.py`
- `tests/test_amazon2023_dataset.py`
- `tests/test_bert4rec.py`
- `tests/test_ease.py`
- `tests/test_elsa.py`
- `tests/test_model_persistence.py`
- `tests/test_progress_bars.py`
- `tests/test_public_api.py`
- `tests/test_recommendations.py`
- `tests/test_sasrec.py`
- `tests/test_simple_rnn.py`
- `tests/test_temporal_split.py`

## The library's own tests

`cd vendor/compresso-recsys && ../../.venv/bin/python -m pytest -q -o addopts=""` runs them against the
installed copy. Tests that read the library repository's `docs/` or `examples/` (dataset audits,
notebooks, documentation snippets) cannot find them here, since only the package is copied; they fail or
error by design. All the others pass: 1,726 on 2026-10-01.

<!-- vendor-models.sh overlay: begin -->
## Overlay: models from another branch (`scripts/vendor-models.sh`)

Added 2026-10-13 17:00:17 from `/home/mark/Documents/recombee/compresso-recsys`, branch `32-bert4rec-model-implementation` at `c920c1c` (`c920c1c047a9746c8e75e3c8ea8be5aff694bc0e`), with 11
of the overlay's files uncommitted there: SASRec v2 (`sasrec_ce`) and LRURec, which that branch has and the copy above
does not (the suite's DECISIONS §44). That branch lacks much of what the suite needs, so it is not vendored whole.

**A full refresh with `scripts/vendor-cr.sh` replaces the whole copy and drops this overlay: run
`scripts/vendor-models.sh` after it**, unless the refreshed checkout has these packages itself. vendor-cr.sh warns
while this section exists.

Files, each Python file with the notice above:

- `src/compresso_recsys/models/lrurec/__init__.py` (new)
- `src/compresso_recsys/models/lrurec/config.py` (new)
- `src/compresso_recsys/models/lrurec/model.py` (new)
- `src/compresso_recsys/models/lrurec/trainer.py` (new)
- `src/compresso_recsys/models/sasrec_ce/__init__.py` (new)
- `src/compresso_recsys/models/sasrec_ce/augmentation.py` (new)
- `src/compresso_recsys/models/sasrec_ce/config.py` (new)
- `src/compresso_recsys/models/sasrec_ce/model.py` (new)
- `src/compresso_recsys/models/sasrec_ce/trainer.py` (new)
- `tests/test_lrurec.py` (new)
- `tests/test_sasrec_ce.py` (new)

Exports added to `src/compresso_recsys/models/__init__.py` and `tests/test_public_api.py`: `LRURec`, `LRURecConfig`, `LRURecTrainer`, `SASRecCE`, `SASRecCEConfig`, `SASRecCETrainer`.

Dependency check: every library module the overlay's code imports, directly or through another, is byte-identical in
the checkout and here (`_reporting.py`, `models/identifiers.py`, `models/sequence_batching.py`, `models/tokenizer.py`, `persistence.py`, `sequences.py`), except:
- `src/compresso_recsys/models/base.py` differs by 22 lines: the checkout adds the CandidateScorer protocol (branch 32's sampled evaluation), which the overlay's code does not import.

The overlay's tests against this copy: `cd vendor/compresso-recsys && ../../.venv/bin/python -m pytest -q -o
addopts="" tests/test_sasrec_ce.py tests/test_lrurec.py tests/test_public_api.py --deselect tests/test_lrurec.py::test_the_trainer_is_a_candidate_scorer --deselect tests/test_lrurec.py::test_it_evaluates_against_sampled_negatives`. Deselected, since they cannot pass here:
- `tests/test_lrurec.py::test_the_trainer_is_a_candidate_scorer`: imports compresso_recsys.models.CandidateScorer, which only the checkout's models/base.py has.
- `tests/test_lrurec.py::test_it_evaluates_against_sampled_negatives`: imports the sampled evaluation (_sampled_negatives, evaluate_sampled_recommender), which only the checkout's evaluation.py has.
<!-- vendor-models.sh overlay: end -->
