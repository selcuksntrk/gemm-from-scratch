//
// Created by Selcuk Senturk on 26.06.2026.
//

#include "gemm.hpp"

#include <iostream>
#include <vector>
#include <chrono>

int main()
{
    constexpr std::size_t M = 512;
    constexpr std::size_t N = 512;
    constexpr std::size_t K = 512;

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);



    const auto start = std::chrono::steady_clock::now();

    gemm_naive(A.data(), B.data(), C.data(), M, N, K);

    const auto end = std::chrono::steady_clock::now();

    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count();

    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

    const double gflops = flops / seconds / 1.0e9; // Gflops measurement to see processing capacity of our program in seconds

    std::cout << "C[0,0]: " << C[0] << '\n';
    std::cout << "Elapsed seconds: " << seconds << '\n';

    std::cout << "FLOPs: " << flops << '\n';
    std::cout << "GFLOPS: " << gflops << '\n';

    return 0;
}