---
name: suite-review
description: Review the seqrec_eval suite, its compresso-recsys branch, its results or a method decision within a declared scope, classifying every finding by whether it can change a thesis conclusion, so reviews converge. Use when Mark asks to review the suite, the project, a change, a report or results, or a methodology choice. Modes - change (default, the diff since the last snapshot), result, decision, full (milestones only).
---

# Scoped review of seqrec_eval

The rules live in `~/Documents/stage/review-plan/REVIEW_FRAMEWORK.md`. **Read it first, every time**: §1 is the
conclusions register (K1–K6, M1–M3) that every finding is weighed against, §3 the classes, §6 the result
checklist. This file says how to run a review under it.

## 1. Pick the mode and state the scope, before reading anything

The first word of the arguments is the mode; the rest narrows the scope.
- **`change`, or no argument:** the scope is the diff since the latest snapshot **and its reach**.
  - Run `~/Documents/stage/review-plan/tools/review-scope.sh diff`. It writes the patch and a `.reach.txt` listing
    every changed function, class or constant and each line that uses it.
  - Review the hunks, then every listed use, for what the change does to it, and the tests that cover them.
    Add uses the list misses: changed settings in the protocol, changed file formats, changed meanings of a
    return value.
  - A bug in the reach that the change did not cause is X.
  - Narrow the scope by any files or topic given. If the diff is empty, say so and stop.
  - If REVIEW_LOG.md "Next" names a pending change review of fixes, that is the scope instead.
- **`result <dataset> [sweep]`:** the reports under the work dir (ask which work dir if unclear) and their run
  records.
  - Go through REVIEW_FRAMEWORK.md §6 item by item, ticking or failing each.
  - Then **the open pass**: read every report once more for anything implausible the list does not cover.
  - Each surprise gets classified. Once understood, propose it as a new checklist item, or as an assertion in
    the report code.
- **`decision <topic>`:** one method choice. Give the options, what each does to K1–K6 and M1–M3, and a
  recommendation. No code.
- **`full`:** only if Mark asked for a full review in so many words. Split by area (operations, scoring,
  statistics), one reviewer per area. Reviewers may read and reproduce on synthetic data but **never edit
  files**. Then verify their serious findings yourself.

Tell Mark the mode and the scope in one or two lines before starting.

## 2. Review the scope only

- Anything noticed outside the scope is **X (parked)**: note it in one line and do not investigate further.
  If it could plausibly be R (it could change saved results or a conclusion), mark it **"X (R?)"**. Those must
  get a real class before the freeze; say so in the report.
- Every finding needs:
  - a file and line;
  - a concrete scenario: these inputs or this state give this wrong outcome;
  - the conclusions it touches (K…/M…);
  - **its size**, measured on synthetic data or bounded from the mechanism, against the threshold of the
    materiality rule (REVIEW_FRAMEWORK.md §3): a tenth of the comparison's standard error, a tenth of δ, or
    across 100 ms for latency;
  - its class, R, P, O or L, from the three questions of §3. **Below the threshold it is L however real** (it
    changing a saved number is not enough); without a size it cannot be R or P. Wording that misleads no
    verdict is L too, left for one polish pass before the write-up.
- R and P findings are reproduced on tiny synthetic data before they are reported, or explicitly marked "read
  only".
- Working rules:
  - never load real datasets;
  - run Python under `systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0`, with the suite's
    `.venv` (`uv sync` installed the library from `vendor/compresso-recsys`; no `PYTHONPATH` needed);
  - no git commands that change state;
  - nothing in cr's `artifacts/`;
  - never write a file without checking it does not exist yet.

## 3. Report, then stop for approval

- Write `~/Documents/stage/review-plan/findings/YYYY-MM-DD-<mode>-<scope>.md` from the template in
  REVIEW_FRAMEWORK.md §8. Add a row per finding to `ISSUES.md`, with the next free N-number and status "open"
  (X: "parked YYYY-MM-DD").
- Give Mark the list, grouped by class, with the proposed fix for each, and **stop**. Fix nothing until Mark
  approves which findings to fix.

## 4. Fix only what was approved

- **One fix at a time.** Each gets a test that fails without it: break the fix on purpose, one change at a time,
  never while another test run is going, and check that a test catches it.
- **The full suite on CPU and GPU after the fixes:**
  `.venv/bin/python -m pytest -q`, and again with
  `SEQREC_EVAL_TEST_DEVICE=cuda`.
- **A DECISIONS.md entry for every change:** what, why, effect, side effects (fingerprints that change and the
  runs they rerun), where, tests. Update the README where behaviour changed, and ISSUES statuses.
- **Something new found while fixing is X,** not part of this round. Say so; do not fold it in.

## 5. Close the review

- **One change review of the fixes' own diff and its reach,** then stop. Only its R findings are fixed; nothing
  else opens a new round.
- **That re-review is independent** when any fix is more than a handful of lines, or any fix is R.
  - Launch a fresh-context reviewer (the Agent tool, `general-purpose`) that did not write the fixes.
  - Give it only: the scope (the patch and reach files, or the list of changed functions), REVIEW_FRAMEWORK.md,
    the working rules above, and the instruction to read and reproduce but never edit.
  - Verify its R and P findings yourself before passing them on.
- **Record:**
  - a row in `REVIEW_LOG.md` (counts per class, fixed, parked);
  - `tools/review-scope.sh snapshot <label>`, so the next change review has a base;
  - update "Next" in REVIEW_LOG.md.
- **Then the freeze, if the project is ready.** If this was a change review with no R findings and FREEZE.md
  says "not frozen", tell Mark the freeze can happen (FREEZE.md lists its steps).
