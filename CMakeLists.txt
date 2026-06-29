cmake_minimum_required(VERSION 4.2)
project(GEMM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(gemm_core
        src/gemm_naive.cpp
        src/gemm_tiled.cpp
        src/gemm_neon.cpp
        src/gemm_omp.cpp
)

find_package(OpenMP REQUIRED)

target_include_directories(gemm_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gemm_core PUBLIC
        OpenMP::OpenMP_CXX
)

add_executable(GEMM
        src/main.cpp
)

target_link_libraries(GEMM PRIVATE
        gemm_core
)

include(FetchContent)

FetchContent_Declare(
        benchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.zip
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(benchmark)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(benchmark PRIVATE -Wno-c2y-extensions)
        target_compile_options(benchmark_main PRIVATE -Wno-c2y-extensions)
endif()

add_executable(gemm_benchmark
        benchmarks/gemm_benchmark.cpp
)

target_link_libraries(gemm_benchmark PRIVATE
        gemm_core
        benchmark::benchmark
)