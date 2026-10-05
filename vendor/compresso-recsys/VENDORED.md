# Vendored compresso-recsys

A modified copy of [compresso-recsys](https://github.com/zombak79/compresso-recsys) (Apache-2.0, see
`LICENSE`), kept here so the evaluation suite runs on exactly the library it was built and reviewed
against, without pushing the local changes to the library. `uv sync` installs it (the suite's
`pyproject.toml`: `[tool.uv.sources]`). Refresh it with `scripts/vendor-cr.sh`; never edit it by hand.
The script adds one thing of its own: a `[tool.uv]` `cache-keys` at the end of `pyproject.toml`, so
`uv sync` reinstalls the copy whenever its source changes (by default uv watches `pyproject.toml` alone).

| | |
|---|---|
| Copied | 2026-10-05 11:03:51 from `/home/mark/Documents/recombee/compresso-recsys` |
| Version | `0.3.7+trainusers` |
| Branch | `local-temporal-train-all-users` at `6547522` (`65475227e226f0eabae5f0471d3813ad86f22f9e`) |
| Uncommitted changes copied | 11 file(s) |
| Upstream compared against | `origin/main` at `1016bfe` |

## Changes from upstream

Every file below differs from upstream; each changed Python file starts with a notice saying so. The
full difference is `CHANGES.patch`. What the changes do is recorded in the suite's DECISIONS.md:
training on every user (`temporal_train_users`, entry 4 and before), bounded training memory
(`loss_chunk_elements`, entry 19), and the BERT4Rec work of the local branch.

- `README.md`
- `pyproject.toml`
- `src/compresso_recsys/builder.py`
- `src/compresso_recsys/models/__init__.py`
- `src/compresso_recsys/models/bert4rec/__init__.py`
- `src/compresso_recsys/models/bert4rec/config.py`
- `src/compresso_recsys/models/bert4rec/model.py`
- `src/compresso_recsys/models/bert4rec/trainer.py`
- `src/compresso_recsys/models/ease.py`
- `src/compresso_recsys/models/elsa.py`
- `src/compresso_recsys/models/sasrec.py`
- `src/compresso_recsys/models/simple_rnn.py`
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
