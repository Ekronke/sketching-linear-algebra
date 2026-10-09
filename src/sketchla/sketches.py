"""Sketching distributions.

Each sketch represents a random matrix S of shape (m, n). Calling
``sketch.apply(A)`` returns S @ A for an (n, d) matrix (or length-n vector) A.
The randomness is drawn once at construction, so the same S can be applied to
several matrices, as sketch-and-solve needs for [A b].

All three are scaled so that E[S^T S] = I.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._backend import countsketch_apply, fwht


class Sketch:
    """Base class: a random (m x n) matrix S."""

    def __init__(self, m: int, n: int, rng=None):
        if m <= 0 or n <= 0:
            raise ValueError("m and n must be positive")
        self.m, self.n = int(m), int(n)
        self.rng = np.random.default_rng(rng)

    def apply(self, A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=np.float64)
        vec = A.ndim == 1
        A2 = A[:, None] if vec else A
        if A2.shape[0] != self.n:
            raise ValueError(f"expected {self.n} rows, got {A2.shape[0]}")
        out = self._apply(A2)
        return out[:, 0] if vec else out

    __matmul__ = apply

    def _apply(self, A: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def to_dense(self) -> np.ndarray:
        """Materialize S. For tests and small examples only."""
        return self.apply(np.eye(self.n))


class GaussianSketch(Sketch):
    """i.i.d. N(0, 1/m) entries. Cost of S @ A: O(mnd)."""

    def __init__(self, m, n, rng=None):
        super().__init__(m, n, rng)
        self.S = self.rng.standard_normal((self.m, self.n)) / np.sqrt(self.m)

    def _apply(self, A):
        return self.S @ A


class SRHT(Sketch):
    """Subsampled randomized Hadamard transform: S = sqrt(N/m) P H D.

    D: random signs, H: normalized Walsh-Hadamard matrix of size N (n padded
    to a power of two), P: m rows sampled uniformly without replacement.
    Cost of S @ A: O(N d log N).
    """

    def __init__(self, m, n, rng=None, use_cpp: bool = True):
        super().__init__(m, n, rng)
        self.N = 1 << (self.n - 1).bit_length()
        if self.m > self.N:
            raise ValueError("m cannot exceed the padded dimension")
        self.signs = self.rng.choice([-1.0, 1.0], size=self.n)
        self.rows = self.rng.choice(self.N, size=self.m, replace=False)
        self.use_cpp = use_cpp

    def _apply(self, A):
        X = np.zeros((self.N, A.shape[1]))
        X[: self.n] = self.signs[:, None] * A
        HX = fwht(X, self.use_cpp) / np.sqrt(self.N)
        return np.sqrt(self.N / self.m) * HX[self.rows]


class CountSketch(Sketch):
    """One random +/-1 per column of S, in a uniformly random row.

    Cost of S @ A: O(nnz(A)). Accepts dense arrays or SciPy sparse matrices.
    """

    def __init__(self, m, n, rng=None, use_cpp: bool = True):
        super().__init__(m, n, rng)
        self.buckets = self.rng.integers(0, self.m, size=self.n)
        self.signs = self.rng.choice([-1.0, 1.0], size=self.n)
        self.use_cpp = use_cpp

    def as_sparse(self) -> sp.csr_matrix:
        return sp.csr_matrix(
            (self.signs, (self.buckets, np.arange(self.n))), shape=(self.m, self.n)
        )

    def apply(self, A):
        if sp.issparse(A):
            return self.as_sparse() @ A
        return super().apply(A)

    def _apply(self, A):
        return countsketch_apply(A, self.buckets, self.signs, self.m, self.use_cpp)


_SKETCHES = {"gaussian": GaussianSketch, "srht": SRHT, "countsketch": CountSketch}


def make_sketch(kind: str, m: int, n: int, rng=None) -> Sketch:
    """Construct a sketch by name: 'gaussian', 'srht' or 'countsketch'."""
    try:
        return _SKETCHES[kind.lower()](m, n, rng)
    except KeyError:
        raise ValueError(f"unknown sketch {kind!r}; choose from {sorted(_SKETCHES)}") from None
