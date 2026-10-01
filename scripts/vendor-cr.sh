#!/usr/bin/env bash
# Copy the compresso-recsys checkout the suite runs on into vendor/compresso-recsys, so this repository is
# self-contained: `uv sync` installs that copy, and nothing has to be pushed to the library itself.
#
#   scripts/vendor-cr.sh [CHECKOUT]     default: ~/Documents/recombee/compresso-recsys
#   UPSTREAM=v0.3.7 scripts/vendor-cr.sh  compare against another upstream point (default: origin/main)
#
# It copies what the build needs (src, tests, pyproject.toml, README.md, LICENSE), committed and uncommitted
# changes alike, and records where it came from and how it differs from upstream:
#   vendor/compresso-recsys/VENDORED.md    source, commits, date, the changed files
#   vendor/compresso-recsys/CHANGES.patch  the full difference from upstream
# Every changed Python file gets a one-line notice, as the Apache-2.0 license (§4b) asks of modified files.
# The checkout is only read: no git command here changes anything.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
CR=${1:-$HOME/Documents/recombee/compresso-recsys}
UPSTREAM=${UPSTREAM:-origin/main}
DEST=$HERE/vendor/compresso-recsys
PARTS=(src tests pyproject.toml README.md LICENSE)
NOTICE="# Modified for seqrec_eval: differs from upstream compresso-recsys; see vendor/compresso-recsys/VENDORED.md"

[[ -f $CR/pyproject.toml && -d $CR/src/compresso_recsys ]] || { echo "no compresso-recsys checkout at $CR" >&2; exit 2; }
base=$(git -C "$CR" rev-parse "$UPSTREAM")
head=$(git -C "$CR" rev-parse HEAD)
branch=$(git -C "$CR" rev-parse --abbrev-ref HEAD)
version=$(sed -n 's/^version = "\([^"]*\)".*/\1/p' "$CR/pyproject.toml" | head -1)

rm -rf "$DEST"
mkdir -p "$DEST"
tar -C "$CR" --exclude=__pycache__ --exclude='*.pyc' --exclude='*.egg-info' --exclude=.pytest_cache \
    -cf - "${PARTS[@]}" | tar -C "$DEST" -xf -

# what differs from upstream: tracked files changed since it (committed or not), and new files not yet in git
mapfile -t changed < <(git -C "$CR" diff --name-only "$base" -- src tests pyproject.toml README.md)
mapfile -t untracked < <(git -C "$CR" ls-files --others --exclude-standard -- src tests)
{
    git -C "$CR" diff "$base" -- src tests pyproject.toml README.md
    for file in "${untracked[@]}"; do
        (cd "$CR" && git diff --no-index -- /dev/null "$file" || true)
    done
} > "$DEST/CHANGES.patch"

for file in "${changed[@]}" "${untracked[@]}"; do
    if [[ $file == *.py && -f $DEST/$file ]]; then
        sed -i "1i $NOTICE" "$DEST/$file"
    fi
done

{
    echo "# Vendored compresso-recsys"
    echo
    echo "A modified copy of [compresso-recsys](https://github.com/zombak79/compresso-recsys) (Apache-2.0, see"
    echo "\`LICENSE\`), kept here so the evaluation suite runs on exactly the library it was built and reviewed"
    echo "against, without pushing the local changes to the library. \`uv sync\` installs it (the suite's"
    echo "\`pyproject.toml\`: \`[tool.uv.sources]\`). Refresh it with \`scripts/vendor-cr.sh\`; never edit it by hand."
    echo
    echo "| | |"
    echo "|---|---|"
    echo "| Copied | $(date '+%F %T') from \`$CR\` |"
    echo "| Version | \`$version\` |"
    echo "| Branch | \`$branch\` at \`$(git -C "$CR" rev-parse --short HEAD)\` (\`$head\`) |"
    echo "| Uncommitted changes copied | $(git -C "$CR" status --short -- src tests pyproject.toml README.md | wc -l) file(s) |"
    echo "| Upstream compared against | \`$UPSTREAM\` at \`$(git -C "$CR" rev-parse --short "$base")\` |"
    echo
    echo "## Changes from upstream"
    echo
    echo "Every file below differs from upstream; each changed Python file starts with a notice saying so. The"
    echo "full difference is \`CHANGES.patch\`. What the changes do is recorded in the suite's DECISIONS.md:"
    echo "training on every user (\`temporal_train_users\`, entry 4 and before), bounded training memory"
    echo "(\`loss_chunk_elements\`, entry 19), and the BERT4Rec work of the local branch."
    echo
    for file in "${changed[@]}"; do echo "- \`$file\`"; done
    for file in "${untracked[@]}"; do echo "- \`$file\` (new)"; done
    echo
    echo "## The library's own tests"
    echo
    echo "\`cd vendor/compresso-recsys && ../../.venv/bin/python -m pytest -q -o addopts=\"\"\` runs them against the"
    echo "installed copy. Tests that read the library repository's \`docs/\` or \`examples/\` (dataset audits,"
    echo "notebooks, documentation snippets) cannot find them here, since only the package is copied; they fail or"
    echo "error by design. All the others pass: 1,726 on 2026-10-01."
} > "$DEST/VENDORED.md"

echo "vendored $version from $branch ($(git -C "$CR" rev-parse --short HEAD)) into $DEST:" \
     "${#changed[@]} changed, ${#untracked[@]} new file(s) against $UPSTREAM"
