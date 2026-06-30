# Created by Selcuk Senturk

import time
from collections.abc import Callable

import numpy as np
import gemm_cpp


def gflops(m: int, n: int, k: int, seconds: float) -> float:
    return (2.0 * m * n * k) / seconds / 1.0e9


def benchmark(
        name: str,
        fn: Callable[[], np.ndarray],
        reference: np.ndarray,
        m: int,
        n: int,
        k: int,
        repeats: int = 5,
) -> None:
    result = fn()

    if not np.allclose(result, reference, rtol=1.0e-4, atol=1.0e-4):
        raise RuntimeError(f"{name} produced incorrect output")

    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        end = time.perf_counter()
        times.append(end - start)

    best = min(times)
    print(f"{name:12s} {best * 1e3:10.3f} ms  {gflops(m, n, k, best):10.2f} GFLOPS")


def main() -> None:
    m = 512
    n = 512
    k = 512

    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)

    reference = a @ b

    print(f"Matrix size: {m}x{k} @ {k}x{n}")
    benchmark("np.matmul", lambda: a @ b, reference, m, n, k)
    benchmark("cpp_naive", lambda: gemm_cpp.gemm_naive(a, b), reference, m, n, k)
    benchmark("cpp_tiled", lambda: gemm_cpp.gemm_tiled(a, b, 64), reference, m, n, k)


if __name__ == "__main__":
    main()