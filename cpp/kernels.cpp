// Performance-critical sketching kernels, exposed to Python via pybind11.
//
// Both kernels take row-major (C-contiguous) float64 matrices, matching
// NumPy's default layout, and return new arrays.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

// Unnormalized fast Walsh-Hadamard transform along axis 0 of an (n x d)
// matrix, n a power of two. O(n d log n). The butterfly updates whole rows
// at a time, so the inner loop is contiguous and vectorizes well.
Array fwht(Array X) {
    if (X.ndim() != 2) throw std::invalid_argument("expected a 2-D array");
    const py::ssize_t n = X.shape(0), d = X.shape(1);
    if (n == 0 || (n & (n - 1)) != 0)
        throw std::invalid_argument("number of rows must be a power of two");

    Array out({n, d});
    std::copy(X.data(), X.data() + n * d, out.mutable_data());
    double* x = out.mutable_data();

    {
        py::gil_scoped_release release;
        for (py::ssize_t h = 1; h < n; h *= 2) {
            for (py::ssize_t i = 0; i < n; i += 2 * h) {
                for (py::ssize_t j = i; j < i + h; ++j) {
                    double* a = x + j * d;
                    double* b = x + (j + h) * d;
                    for (py::ssize_t c = 0; c < d; ++c) {
                        const double u = a[c], v = b[c];
                        a[c] = u + v;
                        b[c] = u - v;
                    }
                }
            }
        }
    }
    return out;
}

// CountSketch: out[buckets[i], :] += signs[i] * A[i, :]. O(n d) for dense A.
Array countsketch(Array A, IndexArray buckets, Array signs, py::ssize_t m) {
    if (A.ndim() != 2) throw std::invalid_argument("expected a 2-D array");
    const py::ssize_t n = A.shape(0), d = A.shape(1);
    if (buckets.size() != n || signs.size() != n)
        throw std::invalid_argument("buckets and signs must have length n");
    if (m <= 0) throw std::invalid_argument("m must be positive");

    Array out({m, d});
    double* o = out.mutable_data();
    std::fill(o, o + m * d, 0.0);
    const double* a = A.data();
    const std::int64_t* h = buckets.data();
    const double* s = signs.data();

    for (py::ssize_t i = 0; i < n; ++i)
        if (h[i] < 0 || h[i] >= m) throw std::out_of_range("bucket index out of range");

    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i) {
            double* row = o + h[i] * d;
            const double* src = a + i * d;
            const double sg = s[i];
            for (py::ssize_t c = 0; c < d; ++c) row[c] += sg * src[c];
        }
    }
    return out;
}

PYBIND11_MODULE(_kernels, m) {
    m.doc() = "C++ kernels for sketchla";
    m.def("fwht", &fwht, py::arg("X"),
          "Unnormalized fast Walsh-Hadamard transform along axis 0.");
    m.def("countsketch", &countsketch, py::arg("A"), py::arg("buckets"),
          py::arg("signs"), py::arg("m"), "Apply a CountSketch to the rows of A.");
}
