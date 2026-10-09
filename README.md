# Sketching for Linear Algebra

Randomized sketching for least squares and low-rank approximation: theory, a
small Python library with C++ kernels, and experiments.

Individual project in theoretical computer science (DD2467) at KTH, autumn 2026.

## What's here

A sketch is a random matrix $S \in \mathbb{R}^{m \times n}$ with $m \ll n$ that
approximately preserves the geometry of a subspace. This repository implements
three sketch families and two algorithms that use them:

| Sketch | Cost of computing $SA$ | Module |
|---|---|---|
| Gaussian | $O(mnd)$ | `sketchla.GaussianSketch` |
| SRHT (subsampled randomized Hadamard transform) | $O(nd \log n)$ | `sketchla.SRHT` |
| CountSketch | $O(\mathrm{nnz}(A))$ | `sketchla.CountSketch` |

- `sketch_and_solve(A, b, m, kind)`: approximate least squares via $\min_x \|SAx - Sb\|$.
- `randomized_svd(A, k, kind=...)`: the Halko–Martinsson–Tropp randomized SVD with any of the sketches as its test matrix.

The fast Walsh–Hadamard transform and CountSketch application are written in
C++ (`cpp/kernels.cpp`) so that runtime comparisons between sketches measure
the algorithms rather than Python overhead. Pure NumPy fallbacks are used if
the extension isn't built; check `sketchla.HAS_CPP_KERNELS`.

## Layout

```
cpp/            C++ kernels (pybind11)
src/sketchla/   Python package: sketches, algorithms, kernel dispatch
tests/          pytest suite
experiments/    experiment scripts; write to results/
report/         LaTeX report
```

## Getting started

Requires Python ≥ 3.10 and a C++17 compiler.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python experiments/least_squares_sketch_size.py
```

After editing `cpp/kernels.cpp`, rerun `pip install -e .` to rebuild.

```python
import numpy as np
from sketchla import sketch_and_solve, randomized_svd

A = np.random.randn(20_000, 50)
b = np.random.randn(20_000)
x = sketch_and_solve(A, b, m=1000, kind="srht")
U, s, Vt = randomized_svd(A, k=10, kind="countsketch", power_iters=1)
```

## References

- N. Halko, P.-G. Martinsson, J. A. Tropp. *Finding structure with randomness.* SIAM Review, 2011.
- D. P. Woodruff. *Sketching as a tool for numerical linear algebra.* FnT-TCS, 2014.
- P.-G. Martinsson, J. A. Tropp. *Randomized numerical linear algebra: foundations and algorithms.* Acta Numerica, 2020.

Full bibliography in `report/refs.bib`.

## License

MIT
