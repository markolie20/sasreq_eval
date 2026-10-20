"""EASE fitted in a fraction of the memory (DECISIONS §55): the library's closed form, the same weights to rounding,
read and written as the library's checkpoints, at about one items × items matrix instead of nine."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from compresso_recsys.models import EASE, EASEConfig
from scipy.sparse import random as sparse_random

from seqrec_eval.models import LowMemoryEASE, model_spec


def _interactions(users: int, items: int, density: float, seed: int):
    matrix = sparse_random(users, items, density=density, format="csr", random_state=seed)
    matrix.data[:] = 1.0
    return matrix


@pytest.mark.parametrize("dtype, tolerance", [("float64", 1e-12), ("float32", 1e-5)])
def test_the_weights_are_the_librarys(monkeypatch, dtype, tolerance):
    monkeypatch.setattr(LowMemoryEASE, "BLOCK", 50)  # several blocks, the last one short
    x = _interactions(400, 137, 0.08, seed=1)
    ours = LowMemoryEASE(EASEConfig(l2=3.0, dtype=dtype)).fit(x).coefficients_
    theirs = EASE(EASEConfig(l2=3.0, dtype=dtype)).fit(x).coefficients_
    assert ours.dtype == theirs.dtype and ours.flags["C_CONTIGUOUS"]
    assert np.abs(ours - theirs).max() <= tolerance * np.abs(theirs).max()
    assert (np.diag(ours) == 0).all()


def test_it_is_the_registered_ease_and_reads_the_librarys_checkpoints(tmp_path):
    assert model_spec("ease").cls is LowMemoryEASE
    x = _interactions(300, 60, 0.1, seed=2)
    EASE(EASEConfig(l2=5.0)).fit(x).save(tmp_path / "library.zip")  # a final made before this class existed
    loaded = LowMemoryEASE.load(tmp_path / "library.zip", device="cpu")
    fitted = LowMemoryEASE(EASEConfig(l2=5.0)).fit(x)
    source = x[:20]
    ours, theirs = (m.predict_on_batch(source, k=10, exclude_seen=True) for m in (fitted, loaded))
    assert np.array_equal(ours.cols.numpy(), theirs.cols.numpy())
    fitted.save(tmp_path / "ours.zip")
    assert np.array_equal(EASE.load(tmp_path / "ours.zip", device="cpu").coefficients_, fitted.coefficients_)


def _peak(cls, x) -> float:
    tracemalloc.start()
    try:
        cls(EASEConfig(l2=10.0)).fit(x)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_it_holds_about_one_items_by_items_matrix(monkeypatch):
    # the library held about nine; on Music4All's 36,797 items that was 45 GB, and the run was killed (2026-10-14)
    items = 2000
    monkeypatch.setattr(LowMemoryEASE, "BLOCK", 100)  # a block as small beside the catalogue as on Music4All
    x = _interactions(4000, items, 0.02, seed=3)
    one = items * items * 4  # one float32 items × items matrix
    assert _peak(LowMemoryEASE, x) < 1.6 * one
    assert _peak(EASE, x) > 3 * one


def test_a_gram_matrix_that_is_not_positive_definite_is_refused(monkeypatch):
    x = _interactions(50, 20, 0.2, seed=4)
    model = LowMemoryEASE(EASEConfig(l2=1.0))
    monkeypatch.setattr(type(model.cfg), "__post_init__", lambda self: None)
    object.__setattr__(model.cfg, "l2", -1e6)  # a negative ridge makes the matrix indefinite
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        model.fit(x)
