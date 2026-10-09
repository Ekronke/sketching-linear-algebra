"""Kernel dispatch: use the compiled C++ kernels when available, else NumPy.

The NumPy versions are reference implementations. They are correct but
slower, so runtime experiments should check ``HAS_CPP_KERNELS``.
"""

import numpy as np

try:
    from . import _kernels  # type: ignore[attr-defined]

    HAS_CPP_KERNELS = True
except ImportError:  # pragma: no cover - depends on the build
    _kernels = None
    HAS_CPP_KERNELS = False


def fwht_numpy(X: np.ndarray) -> np.ndarray:
    """Unnormalized fast Walsh-Hadamard transform along axis 0 (returns a copy)."""
    X = np.array(X, dtype=np.float64, copy=True)
    n = X.shape[0]
    if n & (n - 1):
        raise ValueError("number of rows must be a power of two")
    h = 1
    while h < n:
        Y = X.reshape(n // (2 * h), 2, h, *X.shape[1:])
        a = Y[:, 0].copy()
        Y[:, 0] += Y[:, 1]
        Y[:, 1] = a - Y[:, 1]
        h *= 2
    return X


def countsketch_numpy(A: np.ndarray, buckets: np.ndarray, signs: np.ndarray, m: int) -> np.ndarray:
    """Compute S A where row i of A is added, times signs[i], into row buckets[i]."""
    out = np.zeros((m, A.shape[1]))
    np.add.at(out, buckets, signs[:, None] * A)
    return out


def fwht(X: np.ndarray, use_cpp: bool = True) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim == 1:
        return fwht(X[:, None], use_cpp)[:, 0]
    if use_cpp and HAS_CPP_KERNELS:
        return _kernels.fwht(X)
    return fwht_numpy(X)


def countsketch_apply(A, buckets, signs, m, use_cpp: bool = True) -> np.ndarray:
    A = np.ascontiguousarray(A, dtype=np.float64)
    buckets = np.ascontiguousarray(buckets, dtype=np.int64)
    signs = np.ascontiguousarray(signs, dtype=np.float64)
    if use_cpp and HAS_CPP_KERNELS:
        return _kernels.countsketch(A, buckets, signs, int(m))
    return countsketch_numpy(A, buckets, signs, m)
