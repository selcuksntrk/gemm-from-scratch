//
// Created by Selcuk Senturk on 26.06.2026.
//

#include "gemm.hpp"

#include <iostream>
#include <vector>


int main()
{
    constexpr std::size_t M = 2;
    constexpr std::size_t N = 3;
    constexpr std::size_t K = 4;

    const std::vector<float> A{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };

    const std::vector<float> B{
        1.0f,  2.0f,  3.0f,
        4.0f,  5.0f,  6.0f,
        7.0f,  8.0f,  9.0f,
        10.0f, 11.0f, 12.0f
    };

    std::vector<float> C(M * N, 0.0f);

    gemm_naive(A.data(), B.data(), C.data(), M, N, K);

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << ' ';
        }
        std::cout << '\n';
    }

    return 0;
}