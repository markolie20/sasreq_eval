"""Settings every test shares."""

from __future__ import annotations

import pytest

from seqrec_eval.cli import EXTRA_ENV


@pytest.fixture(autouse=True, scope="session")
def _no_extra_protocol():
    """The tests run the protocols they write, without the extra protocol files the shell may name (on the DGX
    the container sets ``SEQREC_EVAL_PROTOCOL_EXTRA`` for the real runs; DECISIONS §39). Session-wide, so the
    module-wide workspaces built before any test function see it too."""
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(EXTRA_ENV, raising=False)
        yield
