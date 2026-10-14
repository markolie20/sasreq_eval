"""scripts/vendor-models.sh (DECISIONS §44): model packages from another branch of the library, added to the
vendored copy without refreshing the rest of it. Run on a small fake checkout and vendored copy, not the real ones."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTICE = ("# Modified for seqrec_eval: differs from upstream compresso-recsys; "
          "see vendor/compresso-recsys/VENDORED.md")
#: the library both trees share: what the overlay imports, directly or through another module
SHARED = {
    "pyproject.toml": "[project]\nname = 'compresso-recsys'\n",
    "src/compresso_recsys/__init__.py": "",
    "src/compresso_recsys/sequences.py": "X = 1\n",
    "src/compresso_recsys/persistence.py": "from compresso_recsys.sequences import X\n",
    "src/compresso_recsys/models/identifiers.py": "I = 1\n",
    "src/compresso_recsys/models/tokenizer.py": "K = 1\n",
    "src/compresso_recsys/models/sequence_batching.py": "from .tokenizer import K\n",
}
OVERLAY = {
    "src/compresso_recsys/models/sasrec_ce/__init__.py": "from .trainer import T\n",
    "src/compresso_recsys/models/sasrec_ce/trainer.py": (
        "from compresso_recsys.models.base import B\n"
        "from compresso_recsys.models.sequence_batching import K\n"
        "from compresso_recsys.persistence import X\nT = 1\n"),
    "src/compresso_recsys/models/lrurec/__init__.py": "from .trainer import T\n",
    "src/compresso_recsys/models/lrurec/trainer.py": "from ..identifiers import I\nT = 2\n",
    "tests/test_sasrec_ce.py": "def test_a():\n    pass\n",
    "tests/test_lrurec.py": "def test_b():\n    pass\n",
}
API_PATH = "tests/test_public_api.py"
INIT = (f'{NOTICE}\n"""Models."""\n\nfrom .sasrec import SASRec, SASRecConfig, SASRecTrainer\n\n'
        '__all__ = [\n    "SASRec",\n    "SASRecTrainer",\n]\n')
API = (f'{NOTICE}\ndef test_api():\n    expected = {{\n            "SASRec",\n            "SASRecTrainer",\n'
       '    }\n')


def _tree(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def trees(tmp_path):
    checkout = _tree(tmp_path / "checkout", {**SHARED, **OVERLAY,
                                             "src/compresso_recsys/models/base.py": "B = 1\n\n\nclass CandidateScorer:\n    pass\n",
                                             "src/compresso_recsys/models/__init__.py": "anything\n"})
    vendor = _tree(tmp_path / "vendor", {**SHARED, "src/compresso_recsys/models/base.py": "B = 1\n",
                                         "src/compresso_recsys/models/__init__.py": INIT,
                                         "tests/test_public_api.py": API,
                                         "VENDORED.md": "# Vendored compresso-recsys\n\n| | |\n|---|---|\n",
                                         "CHANGES.patch": "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
                                                          "@@ -1 +1 @@\n-a\n+b\n"})
    return checkout, vendor


def _run(script: str, *args, vendor: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "VENDOR_DIR": str(vendor), "VENDOR_DATE": "2026-10-13 12:00:00"}
    return subprocess.run([str(REPO / "scripts" / script), *map(str, args)], env=env, capture_output=True,
                          text=True)


def test_it_copies_the_overlay_records_it_and_adds_nothing_twice(trees):
    checkout, vendor = trees
    first = _run("vendor-models.sh", checkout, vendor=vendor)
    assert first.returncode == 0, first.stderr
    for relative, text in OVERLAY.items():
        assert (vendor / relative).read_text() == f"{NOTICE}\n{text}"
    init, api = (vendor / "src/compresso_recsys/models/__init__.py").read_text(), (vendor / API_PATH).read_text()
    assert "from .lrurec import LRURec, LRURecConfig, LRURecTrainer\n" in init
    assert "from .sasrec_ce import SASRecCE, SASRecCEConfig, SASRecCETrainer\n" in init
    for name in ("LRURec", "LRURecConfig", "LRURecTrainer", "SASRecCE", "SASRecCEConfig", "SASRecCETrainer"):
        assert init.count(f'    "{name}",\n') == 1 and api.count(f'            "{name}",\n') == 1
    vendored, patch = (vendor / "VENDORED.md").read_text(), (vendor / "CHANGES.patch").read_text()
    assert vendored.startswith("# Vendored compresso-recsys\n") and patch.startswith("diff --git a/x.py")
    assert vendored.count("<!-- vendor-models.sh overlay: begin -->") == 1
    assert "run\n`scripts/vendor-models.sh` after it" in vendored
    assert "`src/compresso_recsys/models/base.py` differs by" in vendored
    for test in ("test_the_trainer_is_a_candidate_scorer", "test_it_evaluates_against_sampled_negatives"):
        assert f"--deselect tests/test_lrurec.py::{test}" in vendored
    assert patch.count("vendor-models.sh overlay: generated") == 1
    assert "+++ b/src/compresso_recsys/models/sasrec_ce/trainer.py" in patch
    assert subprocess.run(["git", "apply", "--stat", str(vendor / "CHANGES.patch")], capture_output=True).returncode == 0

    once = _snapshot(vendor)
    again = _run("vendor-models.sh", checkout, vendor=vendor)
    assert again.returncode == 0, again.stderr
    assert _snapshot(vendor) == once  # the second run changed no byte



def test_a_dependency_that_differs_is_refused_before_anything_is_written(trees):
    checkout, vendor = trees
    # sequence_batching is identical, but what it imports is not: the check follows the imports
    (vendor / "src/compresso_recsys/models/tokenizer.py").write_text("K = 2\n")
    before = _snapshot(vendor)
    result = _run("vendor-models.sh", checkout, vendor=vendor)
    assert result.returncode == 1
    assert "src/compresso_recsys/models/tokenizer.py differs" in result.stderr
    assert _snapshot(vendor) == before


def test_vendor_cr_warns_that_a_refresh_drops_the_overlay(trees):
    checkout, vendor = trees
    quiet = _run("vendor-cr.sh", "/nowhere", vendor=vendor)
    assert quiet.returncode == 2 and "warning" not in quiet.stderr  # no overlay recorded yet: nothing to say
    assert _run("vendor-models.sh", checkout, vendor=vendor).returncode == 0
    before = _snapshot(vendor)
    warned = _run("vendor-cr.sh", "/nowhere", vendor=vendor)
    assert warned.returncode == 2  # stopped by the missing checkout, after the warning
    assert "records an overlay from scripts/vendor-models.sh (lrurec sasrec_ce)" in warned.stderr
    assert "run scripts/vendor-models.sh afterwards" in warned.stderr
    assert _snapshot(vendor) == before
