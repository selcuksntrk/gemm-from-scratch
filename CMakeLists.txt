cmake_minimum_required(VERSION 4.2)
project(GEMM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

find_package(OpenMP REQUIRED)

add_library(gemm_core
        src/gemm_naive.cpp
        src/gemm_tiled.cpp
        src/gemm_neon.cpp
        src/gemm_omp.cpp
)

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

FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(gemm_cpp
        bindings/gemm_bindings.cpp
)

target_link_libraries(gemm_cpp PRIVATE
        gemm_core
)