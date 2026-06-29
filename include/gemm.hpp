//
// Created by Selcuk Senturk on 26.06.2026.
// GEMM header Declaration
//

#pragma once

#include <cstddef>

inline constexpr std::size_t kDefaultBlockSize = 64;


void gemm_naive(        // Pure general matrix multiplication without any optimization
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t N,
    std::size_t K
);

void gemm_tiled(        // General matrix multiplication split by block sizes
    const float* A,     // compatible with the L1 cache limit (kDefaultBlockSize = 64KB) to optimize processing speed
    const float* B,
    float* C,
    std::size_t M,
    std::size_t N,
    std::size_t K,
    std::size_t block_size = kDefaultBlockSize
);

void gemm_neon(         // General matrix multiplication compatible with L1 cache limit and ARM Neon SIMD
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t N,
    std::size_t K,
    std::size_t block_size = kDefaultBlockSize
);

void gemm_omp(          // General matrix multiplication using OpenMP for parallelization
    const float* A,     // compatible with L1 cache limit
    const float* B,
    float* C,
    std::size_t M,
    std::size_t N,
    std::size_t K,
    std::size_t block_size = kDefaultBlockSize
);

