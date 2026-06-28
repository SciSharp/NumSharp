#!/usr/bin/env python3
"""
NumPy reference companion to np.concatenate.cs.

Mirrors the C# benchmark harness one-for-one so the two outputs can
be diff-compared. Run with:

    python np.concatenate.bench.py [section]

Where section is one of: dtype, layout, size, count, promotion, kwargs.
Omit for the full sweep.

Reports median wall time (ms) per scenario.
Aligns with NumPy 2.x; tested against 2.4.2.
"""
import sys
import time
import numpy as np


WARMUP = 20
DEFAULT_REPS = 100


def bench(label: str, fn, warmup: int = WARMUP, reps: int = DEFAULT_REPS) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    median_ms = ts[len(ts) // 2] * 1000.0
    print(f"  {label:<44}  {median_ms:>9.3f} ms")
    return median_ms


def header(section: str) -> None:
    print()
    print(f"== {section} ==")


# ------------------- 1. Dtype sweep -------------------
def run_dtype_sweep() -> None:
    header("DTYPE SWEEP (1M+1M, same dtype, contig, axis=0)")
    cases = [
        ("bool",       np.bool_),
        ("int8",       np.int8),
        ("uint8",      np.uint8),
        ("int16",      np.int16),
        ("uint16",     np.uint16),
        ("int32",      np.int32),
        ("uint32",     np.uint32),
        ("int64",      np.int64),
        ("uint64",     np.uint64),
        # NumPy has no native char dtype; closest is uint16 (already covered).
        ("float16",    np.float16),
        ("float32",    np.float32),
        ("float64",    np.float64),
        # NumPy has no decimal dtype.
        ("complex128", np.complex128),
    ]
    for name, dt in cases:
        a = np.ones(1_000_000, dtype=dt)
        b = np.ones(1_000_000, dtype=dt)
        bench(f"dtype_{name}_1M", lambda: np.concatenate([a, b], axis=0))


# ------------------- 2. Layout sweep -------------------
def run_layout_sweep() -> None:
    header("LAYOUT SWEEP (axis varies, int32, 1M elements/src)")
    src = np.arange(1_000_000, dtype=np.int32)

    # C-contig 1D
    a = src.copy(); b = src.copy()
    bench("layout_c_contig_1d", lambda: np.concatenate([a, b], axis=0))

    # C-contig 2D (1000x1000) axes 0 and 1
    a2 = src.reshape(1000, 1000).copy(); b2 = src.reshape(1000, 1000).copy()
    bench("layout_c_contig_2d_axis0", lambda: np.concatenate([a2, b2], axis=0))
    bench("layout_c_contig_2d_axis1", lambda: np.concatenate([a2, b2], axis=1))

    # C-contig 3D (100x100x100) axes 0 and 2
    a3 = src.reshape(100, 100, 100).copy(); b3 = src.reshape(100, 100, 100).copy()
    bench("layout_c_contig_3d_axis0", lambda: np.concatenate([a3, b3], axis=0))
    bench("layout_c_contig_3d_axis2", lambda: np.concatenate([a3, b3], axis=2))

    # F-contig 2D
    af = np.asfortranarray(a2); bf = np.asfortranarray(b2)
    bench("layout_f_contig_2d_axis0", lambda: np.concatenate([af, bf], axis=0))
    bench("layout_f_contig_2d_axis1", lambda: np.concatenate([af, bf], axis=1))

    # Strided (every other row of 2x bigger source)
    big = np.arange(2_000_000, dtype=np.int32).reshape(2000, 1000)
    sa = big[::2]; sb = src.reshape(1000, 1000).copy()
    bench("layout_strided_2d_axis0", lambda: np.concatenate([sa, sb], axis=0))

    # Transposed
    ta = src.reshape(1000, 1000).T; tb = src.reshape(1000, 1000).T
    bench("layout_transposed_2d_axis0", lambda: np.concatenate([ta, tb], axis=0))

    # Simple slice
    sla = big[500:1500, :]; slb = src.reshape(1000, 1000).copy()
    bench("layout_sliced_2d_axis0", lambda: np.concatenate([sla, slb], axis=0))

    # Broadcast (1, 1000) -> (1000, 1000)
    ba = np.broadcast_to(np.arange(1000, dtype=np.int32).reshape(1, 1000), (1000, 1000))
    bb = src.reshape(1000, 1000).copy()
    bench("layout_broadcast_2d_axis0", lambda: np.concatenate([ba, bb], axis=0))


# ------------------- 3. Size sweep -------------------
def run_size_sweep() -> None:
    header("SIZE SWEEP (1D int32, 2 arrays of N elements)")
    for n in [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
        a = np.arange(n, dtype=np.int32)
        b = np.arange(n, dtype=np.int32)
        reps = 500 if n < 100_000 else (200 if n < 1_000_000 else 50)
        bench(f"size_{n}", lambda: np.concatenate([a, b], axis=0), reps=reps)


# ------------------- 4. Array-count sweep -------------------
def run_count_sweep() -> None:
    header("ARRAY COUNT SWEEP (each 100k int32, axis=0)")
    for n in [2, 4, 8, 16, 64, 256, 1024]:
        arrs = [np.arange(100_000, dtype=np.int32) for _ in range(n)]
        reps = 100 if n <= 64 else 30
        bench(f"count_{n}", lambda: np.concatenate(arrs, axis=0), reps=reps)


# ------------------- 5. Promotion sweep -------------------
def run_promotion_sweep() -> None:
    header("PROMOTION SWEEP (1M+1M, mixed dtypes)")

    def make(dt): return np.ones(1_000_000, dtype=dt)

    def pair(name, A, B):
        a = make(A); b = make(B)
        bench(f"prom_{name}", lambda: np.concatenate([a, b], axis=0))

    pair("int8_int16",      np.int8,    np.int16)
    pair("int8_uint8",      np.int8,    np.uint8)
    pair("int32_int64",     np.int32,   np.int64)
    pair("int32_uint32",    np.int32,   np.uint32)
    pair("int32_float32",   np.int32,   np.float32)
    pair("int32_float64",   np.int32,   np.float64)
    pair("int64_float64",   np.int64,   np.float64)
    pair("half_single",     np.float16, np.float32)
    pair("float32_float64", np.float32, np.float64)
    pair("float64_complex", np.float64, np.complex128)
    pair("int32_complex",   np.int32,   np.complex128)


# ------------------- 6. Kwargs sweep -------------------
def run_kwargs_sweep() -> None:
    header("KWARG SURFACE (out=, dtype=, axis=None, casting=)")

    a = np.arange(1_000_000, dtype=np.int32)
    b = np.arange(1_000_000, dtype=np.int32)
    out_i32 = np.empty(2_000_000, dtype=np.int32)
    bench("out_int32_1M", lambda: np.concatenate([a, b], axis=0, out=out_i32))

    af = np.arange(1_000_000, dtype=np.float32)
    bi = np.arange(1_000_000, dtype=np.int32)
    out_f64 = np.empty(2_000_000, dtype=np.float64)
    bench("out_mixed_to_float64", lambda: np.concatenate([af, bi], axis=0, out=out_f64, casting="unsafe"))

    bench("dtype_override_int32_to_float64",
          lambda: np.concatenate([a, b], axis=0, dtype=np.float64))

    a2 = a.reshape(1000, 1000); b2 = b.reshape(1000, 1000)
    bench("axis_none_2x_1M_2D",
          lambda: np.concatenate([a2, b2], axis=None))

    ai64 = np.arange(1_000_000, dtype=np.int64)
    bi64 = np.arange(1_000_000, dtype=np.int64)
    bench("casting_unsafe_int64_to_int32",
          lambda: np.concatenate([ai64, bi64], axis=0, dtype=np.int32, casting="unsafe"))


def main() -> None:
    section = sys.argv[1] if len(sys.argv) > 1 else None
    print("=== NumPy np.concatenate variation sweep ===")
    print(f"Runtime: NumPy {np.__version__} on Python {sys.version.split()[0]}")
    print(f"Warmup={WARMUP} iters, median-of-{DEFAULT_REPS}.")

    run_all = section is None
    if run_all or section == "dtype":     run_dtype_sweep()
    if run_all or section == "layout":    run_layout_sweep()
    if run_all or section == "size":      run_size_sweep()
    if run_all or section == "count":     run_count_sweep()
    if run_all or section == "promotion": run_promotion_sweep()
    if run_all or section == "kwargs":    run_kwargs_sweep()


if __name__ == "__main__":
    main()
