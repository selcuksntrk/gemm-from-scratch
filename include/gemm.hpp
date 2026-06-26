//
// Created by Selcuk Senturk on 26.06.2026.
// GEMM header Declaration
//

#pragma once

#include <cstddef>

void gemm_naive(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t N,
    std::size_t K
);

