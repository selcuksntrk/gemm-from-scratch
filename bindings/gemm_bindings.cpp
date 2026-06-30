//
// Created by Selcuk Senturk on 30.06.2026.
//

#include "gemm.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <stdexcept>

namespace py = pybind11;

int add(int lhs, int rhs) {
    return lhs + rhs;
}

using FloatMatrix = py::array_t<float, py::array::c_style | py::array::forcecast>;

void validate_matrices(const FloatMatrix& A, const FloatMatrix& B) {
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("A and B must be 2D float32 arrays");
    }

    if (A.shape(1) != B.shape(0)) {
        throw std::invalid_argument("A.shape[1] must equal B.shape[0]");
    }
}

py::array_t<float> gemm_naive_py(const FloatMatrix& A, const FloatMatrix& B) {
    validate_matrices(A, B);

    const auto M = static_cast<std::size_t>(A.shape(0));
    const auto K = static_cast<std::size_t>(A.shape(1));
    const auto N = static_cast<std::size_t>(B.shape(1));

    py::array_t<float> C({M, N});

    gemm_naive(
        A.data(),
        B.data(),
        C.mutable_data(),
        M,
        N,
        K
    );

    return C;
}

py::array_t<float> gemm_tiled_py(
    const FloatMatrix& A,
    const FloatMatrix& B,
    std::size_t block_size
) {
    validate_matrices(A, B);

    const auto M = static_cast<std::size_t>(A.shape(0));
    const auto K = static_cast<std::size_t>(A.shape(1));
    const auto N = static_cast<std::size_t>(B.shape(1));

    py::array_t<float> C({M, N});
    std::fill_n(C.mutable_data(), M * N, 0.0f);

    gemm_tiled(
        A.data(),
        B.data(),
        C.mutable_data(),
        M,
        N,
        K,
        block_size
    );

    return C;
}

PYBIND11_MODULE(gemm_cpp, m) {
    m.doc() = "GEMM from scratch C++ bindings";

    m.def("add", &add, "Add two integers");

    m.def(
        "gemm_naive",
        &gemm_naive_py,
        "Naive C++ GEMM for two float32 NumPy arrays"
    );

    m.def(
        "gemm_tiled",
        &gemm_tiled_py,
        "Cache-tiled C++ GEMM for two float32 NumPy arrays",
        py::arg("A"),
        py::arg("B"),
        py::arg("block_size") = kDefaultBlockSize
    );
}