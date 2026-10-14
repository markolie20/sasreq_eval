#!/usr/bin/env bash
# Add model packages from a compresso-recsys checkout to the vendored copy, without refreshing the rest of it
# (DECISIONS §44).
#
#   scripts/vendor-models.sh [CHECKOUT]     default: ~/Documents/recombee/compresso-recsys
#
# scripts/vendor-cr.sh copies a whole checkout. This copies only the overlay below -- SASRec v2 (sasrec_ce) and
# LRURec, with their tests -- from a checkout on another branch, which lacks much of what the suite needs
# (temporal_train_users, the Music4All/Yambda/OTTO/RetailRocket loaders, loss chunking) and so cannot be vendored
# whole. It:
#   - refuses, before writing anything, unless every library module the overlay's code imports, directly or through
#     another module, is byte-identical in the checkout and in the vendored copy. models/base.py may differ (it has
#     branch 32's CandidateScorer protocol, which the overlay's code does not import); the overlay's own tests, run
#     against the vendored copy, are what show that the difference does not matter;
#   - copies the overlay, each Python file with the modified-file notice vendor-cr.sh gives (Apache-2.0 §4b);
#   - adds the overlay's exports to models/__init__.py and tests/test_public_api.py;
#   - records the overlay in CHANGES.patch and VENDORED.md, between markers.
# Running it again replaces what an earlier run added: nothing is ever added twice. The checkout is only read.
# Afterwards, reinstall the library (the venv holds a copy of it) with `uv sync --locked --inexact` -- a plain
# `uv sync` removes installed model plugins -- or, where uv would replace the whole environment (its interpreter has
# changed), with `uv pip install --no-deps --reinstall ./vendor/compresso-recsys`; then run the test command
# VENDORED.md gives.
#
# VENDOR_DIR overrides where the vendored copy is (default: vendor/compresso-recsys beside this folder) and
# VENDOR_DATE the date recorded (default: now); both are for the tests.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
CR=${1:-$HOME/Documents/recombee/compresso-recsys}
DEST=${VENDOR_DIR:-$HERE/vendor/compresso-recsys}
PY=$HERE/.venv/bin/python
[[ -x $PY ]] || PY=python3

[[ -f $CR/pyproject.toml && -d $CR/src/compresso_recsys ]] || { echo "no compresso-recsys checkout at $CR" >&2; exit 2; }
[[ -f $DEST/VENDORED.md && -d $DEST/src/compresso_recsys ]] || { echo "no vendored copy at $DEST" >&2; exit 2; }

OVERLAY=(src/compresso_recsys/models/sasrec_ce src/compresso_recsys/models/lrurec tests/test_sasrec_ce.py
         tests/test_lrurec.py)
if git -C "$CR" rev-parse --git-dir > /dev/null 2>&1; then
    branch=$(git -C "$CR" rev-parse --abbrev-ref HEAD)
    head=$(git -C "$CR" rev-parse HEAD)
    uncommitted=$(git -C "$CR" status --short --untracked-files=all -- "${OVERLAY[@]}" | wc -l)
else
    branch="(not a git checkout)" head="" uncommitted="?"
fi

"$PY" - "$CR" "$DEST" "$branch" "$head" "$uncommitted" "${VENDOR_DATE:-$(date '+%F %T')}" "${OVERLAY[@]}" <<'PY'
import ast
import difflib
import shutil
import sys
from pathlib import Path

checkout, dest = Path(sys.argv[1]), Path(sys.argv[2])
branch, head, uncommitted, date = sys.argv[3:7]
overlay = sys.argv[7:]

NOTICE = ("# Modified for seqrec_eval: differs from upstream compresso-recsys; "
          "see vendor/compresso-recsys/VENDORED.md")
#: library modules the overlay may meet in another form here, and why that is safe
ALLOWED_DIFFERENT = {
    "src/compresso_recsys/models/base.py":
        "the checkout adds the CandidateScorer protocol (branch 32's sampled evaluation), which the overlay's code "
        "does not import",
}
#: overlay tests that cannot pass against the vendored copy, and why
DESELECT = {
    "tests/test_lrurec.py::test_the_trainer_is_a_candidate_scorer":
        "imports compresso_recsys.models.CandidateScorer, which only the checkout's models/base.py has",
    "tests/test_lrurec.py::test_it_evaluates_against_sampled_negatives":
        "imports the sampled evaluation (_sampled_negatives, evaluate_sampled_recommender), which only the "
        "checkout's evaluation.py has",
}
#: the names each overlay package exports through compresso_recsys.models
EXPORTS = {"lrurec": ["LRURec", "LRURecConfig", "LRURecTrainer"],
           "sasrec_ce": ["SASRecCE", "SASRecCEConfig", "SASRecCETrainer"]}
INIT, API = "src/compresso_recsys/models/__init__.py", "tests/test_public_api.py"
IMPORT_ANCHOR = "from .sasrec import SASRec, SASRecConfig, SASRecTrainer"
INIT_ANCHOR, API_ANCHOR = '    "SASRecTrainer",', '            "SASRecTrainer",'
PATCH_BEGIN = "# >>> scripts/vendor-models.sh overlay: generated, replaced on every run"
PATCH_END = "# <<< scripts/vendor-models.sh overlay"
MD_BEGIN, MD_END = "<!-- vendor-models.sh overlay: begin -->", "<!-- vendor-models.sh overlay: end -->"


def fail(message: str) -> None:
    print(f"vendor-models.sh: {message}", file=sys.stderr)
    sys.exit(1)


def python_files(root: Path, part: str) -> list[str]:
    path = root / part
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*.py") if "__pycache__" not in p.parts)
    return [str(p.relative_to(root)) for p in files]


def module_file(name: str) -> str | None:
    """``compresso_recsys.a.b`` as a file of the checkout: ``a/b.py`` or ``a/b/__init__.py``."""
    base = Path("src", *name.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if (checkout / candidate).is_file():
            return str(candidate)
    return None


def library_imports(relative: str) -> set[str]:
    """The checkout's library files that ``relative`` imports a name from: the module defining it, never the package
    __init__ files Python also runs on the way (those differ by design, and only re-export)."""
    path = checkout / relative
    # the package a relative import starts from: the module's own, which for an __init__.py is its directory
    package = list(Path(relative).with_suffix("").parts[1:-1])  # without "src" and the file itself
    found = set()
    for node in ast.walk(ast.parse(path.read_text(), relative)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            root = node.module or ""
            if node.level:
                anchor = package[: len(package) - node.level + 1]
                root = ".".join(anchor + ([root] if root else []))
            # "from pkg import name" may name a submodule or an attribute of the package
            names = [f"{root}.{alias.name}" for alias in node.names if module_file(f"{root}.{alias.name}")] or [root]
        else:
            continue
        for name in names:
            if name.split(".")[0] != "compresso_recsys":
                continue
            file = module_file(name)
            if file is None:
                fail(f"{relative} imports {name}, which is not in the checkout")
            found.add(file)
    return found


# -- 1. the dependency check, before anything is written -------------------------------------------------------
sources = [f for part in overlay if part.startswith("src/") for f in python_files(checkout, part)]
inside = set(sources)
checked, differing, queue = set(), {}, list(sources)
while queue:
    for dependency in library_imports(queue.pop()):
        if dependency in inside or dependency in checked:
            continue
        checked.add(dependency)
        here = dest / dependency
        if not here.is_file():
            fail(f"the overlay needs {dependency}, which the vendored copy does not have")
        if here.read_bytes() != (checkout / dependency).read_bytes():
            if dependency.endswith("__init__.py"):
                fail(f"the overlay imports names from {dependency}, which differs here; import them from the module "
                     "that defines them")
            differing[dependency] = sum(line[:1] in "+-" and line[:3] not in ("+++", "---") for line in
                                        difflib.unified_diff((checkout / dependency).read_text().splitlines(),
                                                             here.read_text().splitlines(), lineterm=""))
            if dependency not in ALLOWED_DIFFERENT:
                fail(f"{dependency} differs between the checkout and the vendored copy ({differing[dependency]} "
                     "lines), and the overlay depends on it: refresh the copy with scripts/vendor-cr.sh from a "
                     "checkout that has both, or check the difference and allow it in ALLOWED_DIFFERENT")
            continue  # an allowed difference: its own imports are the vendored copy's business
        queue.append(dependency)

for anchor, file in ((IMPORT_ANCHOR, INIT), (INIT_ANCHOR, INIT), (API_ANCHOR, API)):
    if (dest / file).read_text().splitlines().count(anchor) != 1:
        fail(f"{file} in the vendored copy does not hold the line {anchor.strip()!r} exactly once")

# -- 2. the copy ---------------------------------------------------------------------------------------------------
copied = []
for part in overlay:
    source, target = checkout / part, dest / part
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    if source.is_dir():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for file in python_files(dest, part):
        path = dest / file
        path.write_text(f"{NOTICE}\n{path.read_text()}")
        copied.append(file)
copied += [str(p.relative_to(dest)) for part in overlay if (dest / part).is_dir()
           for p in sorted((dest / part).rglob("*")) if p.is_file() and p.suffix != ".py"]
copied.sort()


# -- 3. the exports, added to the file as it was before any run (`before`), so nothing is added twice -----------
def with_exports(text: str, anchor: str, lines: list[str]) -> str:
    kept = text.splitlines()
    at = kept.index(anchor) + 1
    return "\n".join(kept[:at] + lines + kept[at:]) + ("\n" if text.endswith("\n") else "")


names = sorted(name for group in EXPORTS.values() for name in group)
edits = {
    INIT: [(IMPORT_ANCHOR, [f"from .{package} import {', '.join(group)}" for package, group in sorted(EXPORTS.items())]),
           (INIT_ANCHOR, [f'    "{name}",' for name in names])],
    API: [(API_ANCHOR, [f'            "{name}",' for name in names])],
}
export_diffs = []
for file, changes in edits.items():
    path = dest / file
    original = path.read_text()
    before = "\n".join(line for line in original.splitlines()
                       if line not in {added for _, lines in changes for added in lines}) + "\n"
    after = before
    for anchor, lines in changes:
        after = with_exports(after, anchor, lines)
    path.write_text(after)
    export_diffs.append("".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                     f"a/{file}", f"b/{file}")))


# -- 4. CHANGES.patch: the overlay's difference from upstream, in one block between markers ---------------------
def new_file_diff(file: str) -> str:
    lines = (dest / file).read_text().splitlines(True)
    body = "".join("+" + line for line in lines)
    if lines and not lines[-1].endswith("\n"):
        body += "\n\\ No newline at end of file\n"
    return (f"diff --git a/{file} b/{file}\nnew file mode 100644\n--- /dev/null\n+++ b/{file}\n"
            f"@@ -0,0 +1,{len(lines)} @@\n{body}")


def replace_block(text: str, begin: str, end: str, block: str) -> str:
    if begin in text:
        head, rest = text.split(begin, 1)
        text = head.rstrip("\n") + "\n" + rest.split(end, 1)[1].lstrip("\n")
    return text.rstrip("\n") + "\n" + block


patch = dest / "CHANGES.patch"
block = (f"{PATCH_BEGIN}\n"
         + "".join(f"diff --git a/{file} b/{file}\n{diff}" for file, diff in zip(edits, export_diffs) if diff)
         + "".join(new_file_diff(file) for file in copied)
         + f"{PATCH_END}\n")
patch.write_text(replace_block(patch.read_text(), PATCH_BEGIN, PATCH_END, block))

# -- 5. VENDORED.md --------------------------------------------------------------------------------------------
listed = "\n".join(f"- `{file}` (new)" for file in copied)
identical = ", ".join(f"`{Path(file).relative_to('src/compresso_recsys')}`" for file in sorted(checked)
                      if file not in differing)
allowed = "".join(f"\n- `{file}` differs by {count} lines: {ALLOWED_DIFFERENT[file]}." for file, count in
                  sorted(differing.items()))
deselect = " ".join(f"--deselect {test}" for test in DESELECT)
reasons = "".join(f"\n- `{test}`: {reason}." for test, reason in DESELECT.items())
tests = " ".join(f for f in overlay if f.startswith("tests/")) + f" {API}"
section = f"""{MD_BEGIN}
## Overlay: models from another branch (`scripts/vendor-models.sh`)

Added {date} from `{checkout}`, branch `{branch}`{f" at `{head[:7]}` (`{head}`)" if head else ""}, with {uncommitted}
of the overlay's files uncommitted there: SASRec v2 (`sasrec_ce`) and LRURec, which that branch has and the copy above
does not (the suite's DECISIONS §44). That branch lacks much of what the suite needs, so it is not vendored whole.

**A full refresh with `scripts/vendor-cr.sh` replaces the whole copy and drops this overlay: run
`scripts/vendor-models.sh` after it**, unless the refreshed checkout has these packages itself. vendor-cr.sh warns
while this section exists.

Files, each Python file with the notice above:

{listed}

Exports added to `{INIT}` and `{API}`: {", ".join(f"`{name}`" for name in names)}.

Dependency check: every library module the overlay's code imports, directly or through another, is byte-identical in
the checkout and here ({identical}), except:{allowed}

The overlay's tests against this copy: `cd vendor/compresso-recsys && ../../.venv/bin/python -m pytest -q -o
addopts="" {tests} {deselect}`. Deselected, since they cannot pass here:{reasons}
{MD_END}
"""
vendored = dest / "VENDORED.md"
vendored.write_text(replace_block(vendored.read_text(), MD_BEGIN, MD_END, "\n" + section))

print(f"vendor-models.sh: {len(copied)} overlay file(s) from {branch} into {dest}; {len(checked)} dependencies "
      f"checked, {len(differing)} allowed to differ ({', '.join(differing) or 'none'})")
print("next: uv sync --locked --inexact (or uv pip install --no-deps --reinstall ./vendor/compresso-recsys where uv would replace the environment), then the test command in VENDORED.md")
PY
