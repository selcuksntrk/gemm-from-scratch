//
// Created by Selcuk Senturk on 26.06.2026.
// GEMM header Declaration
//

#pragma once

#include <cstddef>

inline constexpr std::size_t kDefaultBlockSize = 64;


void gemm_naive(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t N,
    std::size_t K
);

void gemm_tiled(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t N,
    std::size_t K,
    std::size_t block_size = kDefaultBlockSize
);

void gemm_neon(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t N,
    std::size_t K,
    std::size_t block_size = kDefaultBlockSize
);

