"""Algorithms that consume a sketch: sketch-and-solve and the randomized SVD."""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .sketches import make_sketch


def sketch_and_solve(A, b, m: int, kind: str = "gaussian", rng=None):
    """Approximately solve min_x ||Ax - b|| by solving min_x ||SAx - Sb||.

    If S is a subspace embedding for span([A b]) with distortion eps, the
    result satisfies ||A x - b|| <= (1+eps)/(1-eps) * min_x ||A x - b||.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    S = make_sketch(kind, m, A.shape[0], rng)
    SAb = S.apply(np.column_stack([A, b]))
    x, *_ = sla.lstsq(SAb[:, :-1], SAb[:, -1])
    return x


def randomized_svd(A, k: int, oversample: int = 10, power_iters: int = 0,
                   kind: str = "gaussian", rng=None):
    """Halko-Martinsson-Tropp randomized SVD with a pluggable test matrix.

    The test matrix Omega (d x l, l = k + oversample) is S^T for a sketch S of
    shape (l, d), so Y = A Omega = (S A^T)^T. Power iterations use QR
    re-orthonormalization for numerical stability.

    Returns U (n x k), s (k,), Vt (k x d) with A ~= U diag(s) Vt.
    """
    A = np.asarray(A, dtype=np.float64)
    n, d = A.shape
    ell = min(k + oversample, min(n, d))
    S = make_sketch(kind, ell, d, rng)
    Y = S.apply(A.T).T                      # n x ell
    Q, _ = np.linalg.qr(Y)
    for _ in range(power_iters):
        Z, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Z)
    B = Q.T @ A                             # ell x d
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]
