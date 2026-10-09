import numpy as np
import pytest

from sketchla import randomized_svd, sketch_and_solve

KINDS = ["gaussian", "srht", "countsketch"]


@pytest.mark.parametrize("kind", KINDS)
def test_sketch_and_solve_near_optimal(kind):
    rng = np.random.default_rng(0)
    n, d = 5000, 20
    A = rng.standard_normal((n, d))
    b = A @ rng.standard_normal(d) + rng.standard_normal(n)
    x_opt, *_ = np.linalg.lstsq(A, b, rcond=None)
    m = 3000 if kind == "countsketch" else 600
    x = sketch_and_solve(A, b, m, kind=kind, rng=rng)
    ratio = np.linalg.norm(A @ x - b) / np.linalg.norm(A @ x_opt - b)
    assert 1 <= ratio < 1.1


@pytest.mark.parametrize("kind", KINDS)
def test_randomized_svd_near_optimal(kind):
    rng = np.random.default_rng(1)
    n, d, k = 600, 300, 10
    U, _ = np.linalg.qr(rng.standard_normal((n, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = 0.7 ** np.arange(d)                       # fast spectral decay
    A = (U * s) @ V.T
    Uk, sk, Vtk = randomized_svd(A, k, oversample=10, power_iters=1, kind=kind, rng=rng)
    err = np.linalg.norm(A - (Uk * sk) @ Vtk)
    best = np.sqrt(np.sum(s[k:] ** 2))            # Eckart-Young
    assert err < 1.5 * best
    np.testing.assert_allclose(Uk.T @ Uk, np.eye(k), atol=1e-10)
