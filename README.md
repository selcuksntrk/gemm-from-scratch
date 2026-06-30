# GEMM from Scratch

C++20 matrix multiplication project implementing GEMM in progressive optimization stages:

1. naive triple loop
2. cache-tiled loop order
3. ARM NEON SIMD
4. OpenMP thread-level parallelism
5. pybind11 Python bindings

The goal is not to beat BLAS. The goal is to explain every optimization from hardware first principles.

## Hardware and toolchain

Measured on:

- Apple M5 Pro, ARM64
- macOS
- Homebrew LLVM/Clang
- CMake + Ninja
- Python 3.13
- pybind11
- Google Benchmark

Google Benchmark reported:

```text
L1 Data 64 KiB
L1 Instruction 128 KiB
L2 Unified 8192 KiB
```

## Build

```bash
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

Build only the benchmark:

```bash
cmake --build cmake-build-release --target gemm_benchmark
```

Build only the Python extension:

```bash
cmake --build cmake-build-release --target gemm_cpp
```

## Run C++ benchmarks

```bash
./cmake-build-release/gemm_benchmark
```

Run OpenMP scaling:

```bash
OMP_NUM_THREADS=1  ./cmake-build-release/gemm_benchmark
OMP_NUM_THREADS=2  ./cmake-build-release/gemm_benchmark
OMP_NUM_THREADS=4  ./cmake-build-release/gemm_benchmark
OMP_NUM_THREADS=8  ./cmake-build-release/gemm_benchmark
OMP_NUM_THREADS=10 ./cmake-build-release/gemm_benchmark
```

## Run Python benchmark

```bash
PYTHONPATH=cmake-build-release python3 scripts/benchmark_python.py
```

## Benchmark results

### C++ kernels, 512x512 float32

| Kernel | Best configuration | GFLOPS | Interpretation |
|---|---:|---:|---|
| Naive | 512 | ~3.1 | baseline scalar triple loop |
| Tiled | BLOCK=512 | ~37.4 | cache locality and loop-order improvement |
| NEON | BLOCK=256 | ~17.6 | educational SIMD kernel; slower than tiled |
| OpenMP | BLOCK=64, 10 threads | ~203.3 | thread-level scaling across output tiles |

### OpenMP scaling, BM_GemmOmp/512/64

| Threads | GFLOPS | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 24.62 | 1.00x | 100% |
| 2 | 46.57 | 1.89x | 94.6% |
| 4 | 92.30 | 3.75x | 93.7% |
| 8 | 180.66 | 7.34x | 91.8% |
| 10 | 203.33 | 8.26x | 82.6% |

### Python benchmark, 512x512 float32

| Kernel | GFLOPS | Notes |
|---|---:|---|
| NumPy `matmul` | 1607.40 | optimized native backend |
| C++ naive binding | 2.91 | Python calls compiled C++ triple loop |
| C++ tiled binding | 23.41 | Python calls cache-tiled C++ kernel |

## Architecture

```text
gemm_core
  src/gemm_naive.cpp
  src/gemm_tiled.cpp
  src/gemm_neon.cpp
  src/gemm_omp.cpp
        |
        +-- GEMM executable
        +-- gemm_benchmark
        +-- gemm_cpp Python extension
```

`gemm_core` is the single implementation owner. The executable, benchmark suite, and Python module all link against the same kernels.

## Key lessons

- Naive GEMM is correct but memory-inefficient.
- Cache tiling improves reuse before eviction.
- Loop order matters because row-major `B[k*N+j]` is contiguous when `j` changes fastest.
- Hand-written SIMD is not automatically faster than compiler auto-vectorized scalar code.
- OpenMP scales only when enough independent output-tile tasks exist.
- Python bindings are useful only when the heavy loop runs in compiled C++.
