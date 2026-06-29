//
// Created by Selcuk Senturk on 29.06.2026.
//

#include "gemm.hpp"

#include <algorithm>

void gemm_omp(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t N,
    std::size_t K,
    std::size_t block_size
) {
#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t ii = 0; ii < M; ii += block_size) {
        for (std::size_t jj = 0; jj < N; jj += block_size) {
            for (std::size_t kk = 0; kk < K; kk += block_size) {
                const std::size_t i_end = std::min(ii + block_size, M);
                const std::size_t j_end = std::min(jj + block_size, N);
                const std::size_t k_end = std::min(kk + block_size, K);

                for (std::size_t i = ii; i < i_end; ++i) {
                    for (std::size_t k = kk; k < k_end; ++k) {
                        const float a = A[i * K + k];

                        for (std::size_t j = jj; j < j_end; ++j) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}