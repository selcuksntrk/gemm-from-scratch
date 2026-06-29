//
// Created by Selcuk Senturk on 27.06.2026.
// Google Benchmarks for our programs processing measurements
//

#include "gemm.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <vector>

static void BM_GemmNaive(benchmark::State& state) {
    const auto size = static_cast<std::size_t>(state.range(0));

    const std::size_t M = size;
    const std::size_t N = size;
    const std::size_t K = size;

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);

    for (auto _ : state) {
        std::fill(C.begin(), C.end(), 0.0f);

        gemm_naive(A.data(), B.data(), C.data(), M, N, K);

        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * static_cast<double>(M) *
                         static_cast<double>(N) *
                         static_cast<double>(K);

    state.counters["GFLOPS"] = benchmark::Counter(
        flops,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000
    );
}

static void BM_GemmTiled(benchmark::State& state) {
    const auto matrix_size = static_cast<std::size_t>(state.range(0));
    const auto block_size = static_cast<std::size_t>(state.range(1));

    const std::size_t M = matrix_size;
    const std::size_t N = matrix_size;
    const std::size_t K = matrix_size;

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);

    for (auto _ : state) {
        std::fill(C.begin(), C.end(), 0.0f);

        gemm_tiled(A.data(), B.data(), C.data(), M, N, K, block_size);

        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * static_cast<double>(M) *
                         static_cast<double>(N) *
                         static_cast<double>(K);

    state.counters["GFLOPS"] = benchmark::Counter(
        flops,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000
    );
}

static void BM_GemmNeon(benchmark::State& state) {
    const auto matrix_size = static_cast<std::size_t>(state.range(0));
    const auto block_size = static_cast<std::size_t>(state.range(1));

    const std::size_t M = matrix_size;
    const std::size_t N = matrix_size;
    const std::size_t K = matrix_size;

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);

    for (auto _ : state) {
        std::fill(C.begin(), C.end(), 0.0f);

        gemm_neon(A.data(), B.data(), C.data(), M, N, K, block_size);

        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * static_cast<double>(M) *
                         static_cast<double>(N) *
                         static_cast<double>(K);

    state.counters["GFLOPS"] = benchmark::Counter(
        flops,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000
    );
}

static void BM_GemmOmp(benchmark::State& state) {
    const auto matrix_size = static_cast<std::size_t>(state.range(0));
    const auto block_size = static_cast<std::size_t>(state.range(1));

    const std::size_t M = matrix_size;
    const std::size_t N = matrix_size;
    const std::size_t K = matrix_size;

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);

    for (auto _ : state) {
        std::fill(C.begin(), C.end(), 0.0f);

        gemm_omp(A.data(), B.data(), C.data(), M, N, K, block_size);

        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * static_cast<double>(M) *
                         static_cast<double>(N) *
                         static_cast<double>(K);

    state.counters["GFLOPS"] = benchmark::Counter(
        flops,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000
    );
}

BENCHMARK(BM_GemmNaive)->Arg(128)->Arg(256)->Arg(512);

BENCHMARK(BM_GemmTiled)
    ->Args({512, 16})
    ->Args({512, 32})
    ->Args({512, 64})
    ->Args({512, 128})
    ->Args({512, 512});

BENCHMARK(BM_GemmNeon)
    ->Args({512, 16})
    ->Args({512, 32})
    ->Args({512, 64})
    ->Args({512, 128})
    ->Args({512, 256})
    ->Args({512, 512});

BENCHMARK(BM_GemmOmp)
    ->Args({512, 64})
    ->Args({512, 128})
    ->Args({512, 256})
    ->Args({512, 512});

BENCHMARK_MAIN();
