//
// Created by Selcuk Senturk on 28.06.2026.
//

#include "gemm.hpp"

#include <algorithm>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

void gemm_neon(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t N,
    std::size_t K,
    std::size_t block_size
) {
    for (std::size_t ii = 0; ii < M; ii += block_size) {
        for (std::size_t jj = 0; jj < N; jj += block_size) {
            for (std::size_t kk = 0; kk < K; kk += block_size) {
                const std::size_t i_end = std::min(ii + block_size, M);
                const std::size_t j_end = std::min(jj + block_size, N);
                const std::size_t k_end = std::min(kk + block_size, K);

                for (std::size_t i = ii; i < i_end; ++i) {
                    for (std::size_t k = kk; k < k_end; ++k) {
                        const float a = A[i * K + k];

#if defined(__ARM_NEON)
                        const float32x4_t va = vdupq_n_f32(a);

                        std::size_t j = jj;
                        for (; j + 4 <= j_end; j += 4) {
                            const float32x4_t vb = vld1q_f32(&B[k * N + j]);
                            float32x4_t vc = vld1q_f32(&C[i * N + j]);
                            vc = vfmaq_f32(vc, va, vb);
                            vst1q_f32(&C[i * N + j], vc);
                        }

                        for (; j < j_end; ++j) {
                            C[i * N + j] += a * B[k * N + j];
                        }
#else
                        for (std::size_t j = jj; j < j_end; ++j) {
                            C[i * N + j] += a * B[k * N + j];
                        }
#endif
                    }
                }
            }
        }
    }
}