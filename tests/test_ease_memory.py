"""EASE fitted in a fraction of the memory and in float64 (DECISIONS §55, §56): the library's closed form, read and
written as the library's checkpoints, at about three float32 items × items matrices instead of nine, and exact where
the library's float32 inverse is not."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from compresso_recsys.models import EASE, EASEConfig
from scipy.sparse import csr_matrix, hstack
from scipy.sparse import random as sparse_random

from seqrec_eval.models import LowMemoryEASE, model_spec


def _interactions(users: int, items: int, density: float, seed: int):
    matrix = sparse_random(users, items, density=density, format="csr", random_state=seed)
    matrix.data[:] = 1.0
    return matrix


def _ill_conditioned(seed: int = 0):
    """Play counts with a heavy tail and 40 near-duplicate items: at λ = 1 the Gram matrix's condition is ~1e10,
    as on Music4All, where EASE was selected at λ ≈ 1."""
    rng = np.random.default_rng(seed)
    users, items = 2000, 400
    popularity = 1 / np.arange(1, items + 1) ** 0.8
    popularity /= popularity.sum()
    rows, columns, counts = [], [], []
    for user in range(users):
        rows += [user] * 40
        columns += list(rng.choice(items, size=40, replace=False, p=popularity))
        counts += list(np.minimum(rng.zipf(1.3, size=40), 20000))
    x = csr_matrix((np.array(counts, dtype=np.float64), (rows, columns)), shape=(users, items))
    duplicates = x[:, :40].copy()
    duplicates.data = duplicates.data + (rng.random(duplicates.data.size) < 0.05)
    return hstack([x, duplicates]).tocsr()


def _exact(x, l2: float) -> np.ndarray:
    precision = np.linalg.inv((x.T @ x).toarray().astype(np.float64) + l2 * np.eye(x.shape[1]))
    weights = precision / -np.diag(precision)
    np.fill_diagonal(weights, 0)
    return weights


def test_the_weights_are_the_librarys_where_the_problem_is_well_conditioned(monkeypatch):
    monkeypatch.setattr(LowMemoryEASE, "BLOCK", 50)  # several blocks, the last one short
    x = _interactions(400, 137, 0.08, seed=1)
    for dtype, tolerance in (("float64", 1e-12), ("float32", 1e-6)):
        ours = LowMemoryEASE(EASEConfig(l2=3.0, dtype=dtype)).fit(x).coefficients_
        theirs = EASE(EASEConfig(l2=3.0, dtype=dtype)).fit(x).coefficients_
        assert ours.dtype == theirs.dtype and ours.flags["C_CONTIGUOUS"]
        assert np.abs(ours - theirs).max() <= tolerance * np.abs(theirs).max()
        assert (np.diag(ours) == 0).all()


def test_it_stays_exact_where_float32_does_not(monkeypatch):
    # DECISIONS §56: float32 Cholesky stops on such a matrix, and the library's float32 LU inverts it badly
    from scipy.linalg import lapack

    monkeypatch.setattr(LowMemoryEASE, "BLOCK", 64)
    x = _ill_conditioned()
    gram = (x.T @ x).toarray() + np.eye(x.shape[1])
    assert lapack.spotrf(gram.astype(np.float32), lower=False)[1] != 0  # float32 Cholesky fails
    exact = _exact(x, 1.0)
    ours = LowMemoryEASE(EASEConfig(l2=1.0)).fit(x).coefficients_
    assert ours.dtype == np.float32
    assert np.abs(ours - exact).max() <= 1e-5 * np.abs(exact).max()
    library = EASE(EASEConfig(l2=1.0)).fit(x).coefficients_
    assert np.abs(library - exact).max() > 0.1 * np.abs(exact).max()  # the library's float32 inverse is far off

    def top(weights):
        return np.argsort(-np.asarray(x[:500] @ weights.astype(np.float64)), axis=1)[:, :10]

    overlap = [len(set(a) & set(b)) / 10 for a, b in zip(top(ours), top(exact))]
    assert np.mean(overlap) == 1.0


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


def test_it_holds_about_three_float32_items_by_items_matrices(monkeypatch):
    # the library held about nine; on Music4All's 36,797 items that was 45 GB, and the run was killed (2026-10-14).
    # Ours: the float64 Gram matrix (two) and the float32 weights (one)
    items = 2000
    monkeypatch.setattr(LowMemoryEASE, "BLOCK", 100)  # a block as small beside the catalogue as on Music4All
    x = _interactions(4000, items, 0.02, seed=3)
    one = items * items * 4  # one float32 items × items matrix
    assert _peak(LowMemoryEASE, x) < 3.3 * one
    assert _peak(EASE, x) > 6 * one


def test_a_gram_matrix_that_is_not_positive_definite_is_refused(monkeypatch):
    x = _interactions(50, 20, 0.2, seed=4)
    model = LowMemoryEASE(EASEConfig(l2=1.0))
    monkeypatch.setattr(type(model.cfg), "__post_init__", lambda self: None)
    object.__setattr__(model.cfg, "l2", -1e6)  # a negative ridge makes the matrix indefinite
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        model.fit(x)
