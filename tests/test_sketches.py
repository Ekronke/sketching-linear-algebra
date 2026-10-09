import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

from sketchla import CountSketch, HAS_CPP_KERNELS, make_sketch
from sketchla._backend import countsketch_numpy, fwht, fwht_numpy

KINDS = ["gaussian", "srht", "countsketch"]


def embedding_distortion(S, A):
    """max |sigma_i(S U) - 1| for U an orthonormal basis of range(A)."""
    U, _ = np.linalg.qr(A)
    s = np.linalg.svd(S.apply(U), compute_uv=False)
    return np.max(np.abs(s - 1))


def test_fwht_matches_hadamard_matrix():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 3))
    H = sla.hadamard(16)
    np.testing.assert_allclose(fwht_numpy(X), H @ X, atol=1e-12)
    np.testing.assert_allclose(fwht(X), H @ X, atol=1e-12)


@pytest.mark.skipif(not HAS_CPP_KERNELS, reason="C++ kernels not built")
def test_cpp_countsketch_matches_numpy():
    rng = np.random.default_rng(1)
    S = CountSketch(20, 100, rng=rng)
    A = rng.standard_normal((100, 5))
    np.testing.assert_allclose(
        S.apply(A), countsketch_numpy(A, S.buckets, S.signs, 20), atol=1e-12
    )


@pytest.mark.parametrize("kind", KINDS)
def test_unbiased_norm(kind):
    """E||Sx||^2 = ||x||^2: check the empirical mean over many draws."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(200)
    vals = [np.sum(make_sketch(kind, 40, 200, rng=s).apply(x) ** 2) for s in range(400)]
    assert abs(np.mean(vals) / np.sum(x**2) - 1) < 0.05


@pytest.mark.parametrize("kind,m", [("gaussian", 400), ("srht", 400), ("countsketch", 2000)])
def test_subspace_embedding(kind, m):
    rng = np.random.default_rng(3)
    A = rng.standard_normal((4000, 10))
    S = make_sketch(kind, m, 4000, rng=rng)
    assert embedding_distortion(S, A) < 0.5


def test_countsketch_sparse_input():
    rng = np.random.default_rng(4)
    A = sp.random(500, 8, density=0.05, random_state=4, format="csr")
    S = CountSketch(50, 500, rng=rng)
    np.testing.assert_allclose(S.apply(A).toarray(), S.apply(A.toarray()), atol=1e-12)


def test_unknown_sketch():
    with pytest.raises(ValueError):
        make_sketch("nope", 3, 10)
