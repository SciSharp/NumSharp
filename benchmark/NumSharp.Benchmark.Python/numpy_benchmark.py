#!/usr/bin/env python3
"""
NumPy Performance Benchmarks
============================

Comprehensive benchmarks matching the C# NumSharp.Benchmark.CSharp suite.
These provide the baseline for comparing NumSharp performance against NumPy.

Usage:
    python numpy_benchmark.py                    # Run all benchmarks
    python numpy_benchmark.py --suite dispatch   # Run specific suite
    python numpy_benchmark.py --quick            # Quick run (fewer reps)
    python numpy_benchmark.py --json             # Output JSON for parsing
    python numpy_benchmark.py --type int32       # Run specific type
    python numpy_benchmark.py --size 10000000   # Specific array size

Requirements:
    pip install numpy tabulate
"""

import os

# Single-threaded BLAS baseline (must be set BEFORE `import numpy`). NumSharp is measured
# single-threaded everywhere — its managed kernels have no thread pool and its optional OpenBLAS
# backend is enabled with threads=1 (byte-parity with NumPy requires a fixed thread count) — so an
# apples-to-apples matrix has to pin NumPy's BLAS/OpenMP pools to 1 as well. Without this, the
# BLAS-backed cells (matmul / dot / linalg / tensordot / the product gufuncs) compared NumSharp's
# single-thread OpenBLAS against a multi-threaded NumPy, understating those ratios by the core count
# (e.g. matmul 316x316 / 384x384 read ~0.3-0.5x purely from the thread gap). Elementwise NumPy is
# single-threaded regardless, so only the matrix-product cells are affected. Matches the backend
# profile twin (benchmark/backends/backend_profile_bench.py), which already pins these.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import time
import argparse
import json
import sys
import contextlib
import io
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional, Dict, Any
import statistics
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from benchmark_modes import ALL_DTYPES, DEPTHS, parse_dtypes, selected  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================

ARRAY_SIZES = {
    'scalar': 1,           # Pure overhead measurement (dispatch, allocation, no loop)
    'tiny': 100,           # Common small collections (configs, batches, embeddings)
    'small': 1_000,        # L1 cache, per-element overhead
    'medium': 100_000,     # L2/L3 cache, typical use case
    'large': 10_000_000    # Memory-bound, throughput measurement
}

# Physical cap for allocation-heavy families that still publish under the universal 10M tier.
# The reported/join key remains n=10_000_000; this helper changes only the arrays used by the body.
MEMORY_HEAVY_LARGE_WORKLOAD = 1_000_000

def memory_heavy_work_n(tier_n: int) -> int:
    return MEMORY_HEAVY_LARGE_WORKLOAD if tier_n == ARRAY_SIZES['large'] else tier_n

# NumPy dtypes matching NumSharp's 12 supported types
DTYPES = {
    'bool': np.bool_,
    'uint8': np.uint8,
    'int8': np.int8,            # NumSharp SByte
    'int16': np.int16,
    'uint16': np.uint16,
    'int32': np.int32,
    'uint32': np.uint32,
    'int64': np.int64,
    'uint64': np.uint64,
    'float16': np.float16,      # NumSharp Half
    'float32': np.float32,
    'float64': np.float64,
    'complex128': np.complex128, # NumSharp Complex
}

# Set by main(); direct imports/tests retain the full measured defaults.
ACTIVE_BENCHMARK_DEPTH = "measure"
ACTIVE_DTYPES = set(DTYPES)

# Common types for quick benchmarks
COMMON_DTYPES = ['int32', 'int64', 'float32', 'float64']

# Arithmetic types (excludes bool). complex128 is fine: run_arithmetic restricts divide/modulo
# to COMMON_DTYPES, so complex only sees +, -, * (and sum/mean/min/max in run_reduction).
ARITHMETIC_DTYPES = ['uint8', 'int8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
                     'float16', 'float32', 'float64', 'complex128']

# Transcendental inputs currently covered by the official NumPy twin. NumPy accepts int32
# operands for these ufuncs and promotes the result where required, matching the C# scenarios.
TRANSCENDENTAL_DTYPES = ['int32', 'float16', 'float32', 'float64']

# =============================================================================
# Benchmark Infrastructure
# =============================================================================

@dataclass
class BenchmarkResult:
    name: str
    category: str
    suite: str
    dtype: str
    n: int
    mean_ms: float
    stddev_ms: float
    min_ms: float
    max_ms: float
    iterations: int
    ops_per_sec: float
    allocated_mb: float = 0.0

def benchmark(func: Callable, n: int, warmup: int = 10, iterations: int = 50,
              min_measure_ms: float = 1.0, pilot_ms: float = 0.3) -> BenchmarkResult:
    """Run a benchmark with warmup, per-test adaptive batching, and a MIN-TIME budget.

    Min-time methodology (time-bound, NOT a fixed iteration count):
      * A slow op (>20 ms/call) runs EXACTLY ``slow_samples`` real calls (min over them).
      * Every other op batches ``inner`` calls per sample — a count computed PER TEST — so each
        sample spans >= ``min_measure_ms`` (~1 ms), and takes as many samples as needed for the
        TOTAL measurement window to span ``total_ms``. A sub-µs op therefore runs hundreds of
        thousands of times (batched); a ~5 ms/call op does ``inner`` == 1 for ~40 real calls.
    ``total_ms`` and ``slow_samples`` scale off ``iterations`` (50 -> 200 ms / 100 samples ;
    10 -> 40 ms / 20), so the official run hits the 200 ms / 100-sample floor while --quick stays
    a fast dev sweep. Per-call cost is first estimated with a GROWING PILOT batch — a single
    calibration call is too noisy for a sub-µs op (its lone perf_counter pair is mostly timer
    overhead). Reported ``iterations`` is the TOTAL calls executed (samples x inner).
    """
    depth = DEPTHS[ACTIVE_BENCHMARK_DEPTH]
    if depth.name == "pass":
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000.0
        return BenchmarkResult(
            name=func.__name__ if hasattr(func, '__name__') else str(func),
            category="", suite="", dtype="", n=n,
            mean_ms=elapsed, stddev_ms=0.0, min_ms=elapsed, max_ms=elapsed,
            iterations=1, ops_per_sec=1000.0 / elapsed if elapsed > 0 else 0)

    warmup = depth.numpy_warmups
    for _ in range(warmup):
        func()

    # Pilot: grow a batch until it takes >= pilot_ms, then estimate per-call cost from it.
    pilot_n, dt = 1, 0.0
    while True:
        t0 = time.perf_counter()
        for _ in range(pilot_n):
            func()
        dt = (time.perf_counter() - t0) * 1000.0
        if dt >= pilot_ms or pilot_n >= (1 << 24):
            break
        pilot_n *= 4
    one_ms = dt / pilot_n if (pilot_n and dt > 0) else 1e-9

    # Min-time policy (repo perf convention): a call >20 ms/call runs EXACTLY `slow_samples` times
    # (each sample is one real call); everything else batches ~min_measure_ms (>=1 ms) windows and
    # accumulates enough of them for the TOTAL window to span `total_ms` — time-bound, not a fixed
    # sample count. Both budgets scale off `iterations`; depth then applies the shared divisor so
    # measure lands on the 200 ms / 100-sample floor and light uses one sixth of both budgets.
    divisor = depth.numpy_budget_divisor
    total_ms = max(1.0, iterations * 4.0 / divisor)     # measure 200 ms; light ~33 ms
    slow_samples = max(1, math.ceil(iterations * 2 / divisor))  # measure 100; light 17
    if one_ms > 20.0:
        inner, samples = 1, slow_samples
    else:
        inner = max(1, math.ceil(min_measure_ms / one_ms))          # ~min_measure_ms (>=1 ms) window
        samples = max(2, math.ceil(total_ms / (inner * one_ms)))    # enough windows to span total_ms

    # Timed runs
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(inner):
            func()
        times.append((time.perf_counter() - start) * 1000.0 / inner)  # per-call ms

    mean = statistics.mean(times)
    stddev = statistics.stdev(times) if len(times) > 1 else 0

    return BenchmarkResult(
        name=func.__name__ if hasattr(func, '__name__') else str(func),
        category="",
        suite="",
        dtype="",
        n=n,
        mean_ms=mean,
        stddev_ms=stddev,
        min_ms=min(times),
        max_ms=max(times),
        iterations=samples * inner,   # total op calls executed (50 for a slow op, many for fast)
        ops_per_sec=1000.0 / mean if mean > 0 else 0
    )

def create_random_array(n: int, dtype_name: str, seed: int = 42) -> np.ndarray:
    """Create a random array of the specified dtype."""
    np.random.seed(seed)
    dtype = DTYPES[dtype_name]

    if dtype == np.bool_:
        return np.random.randint(0, 2, n, dtype=dtype)
    elif np.issubdtype(dtype, np.integer):
        if np.issubdtype(dtype, np.unsignedinteger):
            return np.random.randint(0, 100, n, dtype=dtype)
        else:
            return np.random.randint(-50, 50, n, dtype=dtype)
    else:
        return (np.random.random(n) * 100 - 50).astype(dtype)

def create_positive_array(n: int, dtype_name: str, seed: int = 42) -> np.ndarray:
    """Create a positive random array (for sqrt, log, etc.)."""
    np.random.seed(seed)
    dtype = DTYPES[dtype_name]
    return (np.random.random(n) * 100 + 1).astype(dtype)

# =============================================================================
# Arithmetic Benchmarks
# =============================================================================

def run_arithmetic_benchmarks(n: int, dtype_name: str, iterations: int) -> List[BenchmarkResult]:
    """Benchmark arithmetic operations for a specific dtype.

    Matches C# benchmark classes:
    - AddBenchmarks: Add_Elementwise, NpAdd, Add_Scalar, Add_ScalarLiteral
    - SubtractBenchmarks: Subtract_Elementwise, Subtract_Scalar, Subtract_ScalarLeft
    - MultiplyBenchmarks: Multiply_Elementwise, Multiply_Square, Multiply_Scalar, Multiply_ScalarLiteral
    - DivideBenchmarks: Divide_Elementwise, Divide_Scalar, Divide_ScalarLeft (CommonTypes only)
    - ModuloBenchmarks: Modulo_Elementwise, Modulo_Scalar (CommonTypes only)
    """
    results = []
    dtype = DTYPES[dtype_name]

    np.random.seed(42)
    a = create_random_array(n, dtype_name, seed=42)
    b = create_random_array(n, dtype_name, seed=43)
    b_positive = create_positive_array(n, dtype_name, seed=43)  # Avoid div by zero
    scalar = dtype(5)
    scalar2 = dtype(2)

    # ========================================================================
    # Add (matches AddBenchmarks.cs - ArithmeticTypes = 10 types)
    # ========================================================================

    # Add_Elementwise: a + b (element-wise)
    def add_elementwise(): return a + b
    r = benchmark(add_elementwise, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a + b (element-wise) ({dtype_name})", "Add", "Arithmetic", dtype_name
    results.append(r)

    # NpAdd: np.add(a, b)
    def np_add(): return np.add(a, b)
    r = benchmark(np_add, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.add(a, b) ({dtype_name})", "Add", "Arithmetic", dtype_name
    results.append(r)

    # Add_Scalar: a + scalar
    def add_scalar(): return a + scalar
    r = benchmark(add_scalar, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a + scalar ({dtype_name})", "Add", "Arithmetic", dtype_name
    results.append(r)

    # Add_ScalarLiteral: a + 5 (literal)
    def add_scalar_literal(): return a + 5
    r = benchmark(add_scalar_literal, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a + 5 (literal) ({dtype_name})", "Add", "Arithmetic", dtype_name
    results.append(r)

    # ========================================================================
    # Subtract (matches SubtractBenchmarks.cs - ArithmeticTypes = 10 types)
    # ========================================================================

    # Subtract_Elementwise: a - b (element-wise)
    def sub_elementwise(): return a - b
    r = benchmark(sub_elementwise, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a - b (element-wise) ({dtype_name})", "Subtract", "Arithmetic", dtype_name
    results.append(r)

    def np_subtract(): return np.subtract(a, b)
    r = benchmark(np_subtract, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.subtract(a, b) ({dtype_name})", "Subtract", "Arithmetic", dtype_name
    results.append(r)

    # Subtract_Scalar: a - scalar
    def sub_scalar(): return a - scalar
    r = benchmark(sub_scalar, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a - scalar ({dtype_name})", "Subtract", "Arithmetic", dtype_name
    results.append(r)

    # Subtract_ScalarLeft: scalar - a
    def sub_scalar_left(): return scalar - a
    r = benchmark(sub_scalar_left, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"scalar - a ({dtype_name})", "Subtract", "Arithmetic", dtype_name
    results.append(r)

    # ========================================================================
    # Multiply (matches MultiplyBenchmarks.cs - ArithmeticTypes = 10 types)
    # ========================================================================

    # Multiply_Elementwise: a * b (element-wise)
    def mul_elementwise(): return a * b
    r = benchmark(mul_elementwise, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a * b (element-wise) ({dtype_name})", "Multiply", "Arithmetic", dtype_name
    results.append(r)

    def np_multiply(): return np.multiply(a, b)
    r = benchmark(np_multiply, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.multiply(a, b) ({dtype_name})", "Multiply", "Arithmetic", dtype_name
    results.append(r)

    # Multiply_Square: a * a (square)
    def mul_square(): return a * a
    r = benchmark(mul_square, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a * a (square) ({dtype_name})", "Multiply", "Arithmetic", dtype_name
    results.append(r)

    # Multiply_Scalar: a * scalar
    def mul_scalar(): return a * scalar2
    r = benchmark(mul_scalar, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a * scalar ({dtype_name})", "Multiply", "Arithmetic", dtype_name
    results.append(r)

    # Multiply_ScalarLiteral: a * 2 (literal)
    def mul_scalar_literal(): return a * 2
    r = benchmark(mul_scalar_literal, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a * 2 (literal) ({dtype_name})", "Multiply", "Arithmetic", dtype_name
    results.append(r)

    # ========================================================================
    # Divide (matches DivideBenchmarks.cs - CommonTypes = 4 types)
    # ========================================================================
    if dtype_name in COMMON_DTYPES:
        # Divide_Elementwise: a / b (element-wise)
        def div_elementwise(): return a / b_positive
        r = benchmark(div_elementwise, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"a / b (element-wise) ({dtype_name})", "Divide", "Arithmetic", dtype_name
        results.append(r)

        def np_divide(): return np.divide(a, b_positive)
        r = benchmark(np_divide, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.divide(a, b) ({dtype_name})", "Divide", "Arithmetic", dtype_name
        results.append(r)

        def np_true_divide(): return np.true_divide(a, b_positive)
        r = benchmark(np_true_divide, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.true_divide(a, b) ({dtype_name})", "Divide", "Arithmetic", dtype_name
        results.append(r)

        def np_floor_divide(): return np.floor_divide(a, b_positive)
        r = benchmark(np_floor_divide, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.floor_divide(a, b) ({dtype_name})", "Divide", "Arithmetic", dtype_name
        results.append(r)

        # Divide_Scalar: a / scalar
        def div_scalar(): return a / scalar2
        r = benchmark(div_scalar, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"a / scalar ({dtype_name})", "Divide", "Arithmetic", dtype_name
        results.append(r)

        # Divide_ScalarLeft: scalar / a
        def div_scalar_left(): return scalar2 / b_positive
        r = benchmark(div_scalar_left, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"scalar / a ({dtype_name})", "Divide", "Arithmetic", dtype_name
        results.append(r)

    # ========================================================================
    # Modulo (matches ModuloBenchmarks.cs - CommonTypes = 4 types)
    # ========================================================================
    if dtype_name in COMMON_DTYPES:
        # Modulo_Elementwise: a % b (element-wise)
        def mod_elementwise(): return a % b_positive
        r = benchmark(mod_elementwise, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"a % b (element-wise) ({dtype_name})", "Modulo", "Arithmetic", dtype_name
        results.append(r)

        def np_mod(): return np.mod(a, b_positive)
        r = benchmark(np_mod, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.mod(a, b) ({dtype_name})", "Modulo", "Arithmetic", dtype_name
        results.append(r)

        # Modulo_Scalar: a % 7 (literal)
        def mod_scalar(): return a % 7
        r = benchmark(mod_scalar, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"a % 7 (literal) ({dtype_name})", "Modulo", "Arithmetic", dtype_name
        results.append(r)

    return results

# =============================================================================
# Unary Benchmarks
# =============================================================================

def run_unary_benchmarks(n: int, dtype_name: str, iterations: int) -> List[BenchmarkResult]:
    """Benchmark unary operations for a specific dtype."""
    results = []
    supports_float_ufuncs = dtype_name == "int32" or np.issubdtype(DTYPES[dtype_name], np.floating)

    a = create_random_array(n, dtype_name, seed=42)
    a_positive = create_positive_array(n, dtype_name, seed=42)
    a_small = (np.random.random(n) * 10).astype(DTYPES[dtype_name])  # For exp

    # Math functions
    def np_sqrt(): return np.sqrt(a_positive)
    r = benchmark(np_sqrt, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.sqrt ({dtype_name})", "Math", "Unary", dtype_name
    results.append(r)

    def np_abs(): return np.abs(a)
    r = benchmark(np_abs, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.abs ({dtype_name})", "Math", "Unary", dtype_name
    results.append(r)

    def np_sign(): return np.sign(a)
    r = benchmark(np_sign, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.sign ({dtype_name})", "Math", "Unary", dtype_name
    results.append(r)

    # NumPy promotes int32 as needed for these floating-domain ufuncs.
    if supports_float_ufuncs:
        def np_floor(): return np.floor(a)
        r = benchmark(np_floor, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.floor ({dtype_name})", "Rounding", "Unary", dtype_name
        results.append(r)

        def np_ceil(): return np.ceil(a)
        r = benchmark(np_ceil, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.ceil ({dtype_name})", "Rounding", "Unary", dtype_name
        results.append(r)

        def np_around(): return np.around(a)
        r = benchmark(np_around, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.around ({dtype_name})", "Rounding", "Unary", dtype_name
        results.append(r)

    # Exp/Log
    if supports_float_ufuncs:
        def np_exp(): return np.exp(a_small)
        r = benchmark(np_exp, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.exp ({dtype_name})", "ExpLog", "Unary", dtype_name
        results.append(r)

        def np_log(): return np.log(a_positive)
        r = benchmark(np_log, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.log ({dtype_name})", "ExpLog", "Unary", dtype_name
        results.append(r)

        def np_log10(): return np.log10(a_positive)
        r = benchmark(np_log10, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.log10 ({dtype_name})", "ExpLog", "Unary", dtype_name
        results.append(r)

    # Trig
    if supports_float_ufuncs:
        angles = (((np.random.random(n) * 4 - 2) * np.pi)
                  .astype(DTYPES[dtype_name]))

        def np_sin(): return np.sin(angles)
        r = benchmark(np_sin, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.sin ({dtype_name})", "Trig", "Unary", dtype_name
        results.append(r)

        def np_cos(): return np.cos(angles)
        r = benchmark(np_cos, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.cos ({dtype_name})", "Trig", "Unary", dtype_name
        results.append(r)

        def np_tan(): return np.tan(angles)
        r = benchmark(np_tan, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.tan ({dtype_name})", "Trig", "Unary", dtype_name
        results.append(r)

    # Extra exp/log (mirror C# ExpLogBenchmarks: exp2, expm1, log2, log1p).
    if supports_float_ufuncs:
        def np_exp2(): return np.exp2(a_small)
        r = benchmark(np_exp2, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.exp2 ({dtype_name})", "ExpLog", "Unary", dtype_name
        results.append(r)

        def np_expm1(): return np.expm1(a_small)
        r = benchmark(np_expm1, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.expm1 ({dtype_name})", "ExpLog", "Unary", dtype_name
        results.append(r)

        def np_log2(): return np.log2(a_positive)
        r = benchmark(np_log2, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.log2 ({dtype_name})", "ExpLog", "Unary", dtype_name
        results.append(r)

        def np_log1p(): return np.log1p(a_positive)
        r = benchmark(np_log1p, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.log1p ({dtype_name})", "ExpLog", "Unary", dtype_name
        results.append(r)

    # Clip (mirror C# MathBenchmarks) + scalar Power (mirror C# PowerBenchmarks).
    if supports_float_ufuncs:
        def np_clip(): return np.clip(a, -10.0, 10.0)
        r = benchmark(np_clip, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.clip(a, -10, 10) ({dtype_name})", "Math", "Unary", dtype_name
        results.append(r)

        def np_power2(): return np.power(a, 2)
        r = benchmark(np_power2, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.power(a, 2) ({dtype_name})", "Power", "Unary", dtype_name
        results.append(r)

        def np_power3(): return np.power(a, 3)
        r = benchmark(np_power3, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.power(a, 3) ({dtype_name})", "Power", "Unary", dtype_name
        results.append(r)

        def square_via_multiply(): return a * a
        r = benchmark(square_via_multiply, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"a * a (square via multiply) ({dtype_name})", "Power", "Unary", dtype_name
        results.append(r)

        def cube_via_multiply(): return a * a * a
        r = benchmark(cube_via_multiply, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"a * a * a (cube via multiply) ({dtype_name})", "Power", "Unary", dtype_name
        results.append(r)

        def np_power_half(): return np.power(a_positive, 0.5)
        r = benchmark(np_power_half, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.power(a, 0.5) ({dtype_name})", "Power", "Unary", dtype_name
        results.append(r)

        # Remaining public unary families. Keep every domain identical to the C# setup so
        # timings compare kernels rather than different warning/exception paths.
        family_a = (np.random.random(n) * 8 - 4).astype(DTYPES[dtype_name])
        unit = (np.random.random(n) * 2 - 1).astype(DTYPES[dtype_name])
        acosh_input = (np.random.random(n) * 8 + 1).astype(DTYPES[dtype_name])
        atanh_input = (np.random.random(n) * 1.8 - 0.9).astype(DTYPES[dtype_name])
        other = (np.random.random(n) * 8 - 4).astype(DTYPES[dtype_name])
        family_cases = [
            ("np.absolute(a)", lambda: np.absolute(family_a)),
            ("np.arcsin(a)", lambda: np.arcsin(unit)),
            ("np.arccos(a)", lambda: np.arccos(unit)),
            ("np.arctan(a)", lambda: np.arctan(family_a)),
            ("np.arctan2(a, b)", lambda: np.arctan2(family_a, other)),
            ("np.sinh(a)", lambda: np.sinh(unit)),
            ("np.cosh(a)", lambda: np.cosh(unit)),
            ("np.tanh(a)", lambda: np.tanh(family_a)),
            ("np.arcsinh(a)", lambda: np.arcsinh(family_a)),
            ("np.asinh(a)", lambda: np.asinh(family_a)),
            ("np.arccosh(a)", lambda: np.arccosh(acosh_input)),
            ("np.acosh(a)", lambda: np.acosh(acosh_input)),
            ("np.arctanh(a)", lambda: np.arctanh(atanh_input)),
            ("np.atanh(a)", lambda: np.atanh(atanh_input)),
            ("np.deg2rad(a)", lambda: np.deg2rad(family_a)),
            ("np.radians(a)", lambda: np.radians(family_a)),
            ("np.rad2deg(a)", lambda: np.rad2deg(family_a)),
            ("np.degrees(a)", lambda: np.degrees(family_a)),
            ("np.angle(a)", lambda: np.angle(family_a)),
            ("np.conjugate(a)", lambda: np.conjugate(family_a)),
            ("np.conj(a)", lambda: np.conj(family_a)),
            ("np.real(a)", lambda: np.real(family_a)),
            ("np.imag(a)", lambda: np.imag(family_a)),
            ("np.rint(a)", lambda: np.rint(family_a)),
            ("np.round(a)", lambda: np.round(family_a)),
        ]
        if dtype_name != "float16":
            family_cases.append(("np.modf(a)", lambda: np.modf(family_a)))
        for name, func in family_cases:
            r = benchmark(func, n, iterations=iterations)
            r.name, r.category, r.suite, r.dtype = f"{name} ({dtype_name})", "Families", "Unary", dtype_name
            results.append(r)

    return results

# =============================================================================
# Reduction Benchmarks
# =============================================================================

def run_reduction_benchmarks(n: int, dtype_name: str, iterations: int) -> List[BenchmarkResult]:
    """Benchmark reduction operations."""
    results = []

    a = create_random_array(n, dtype_name, seed=42)
    rows = int(np.sqrt(n))
    cols = n // rows
    a_2d = create_random_array(rows * cols, dtype_name, seed=42).reshape(rows, cols)

    # Sum
    def np_sum(): return np.sum(a)
    r = benchmark(np_sum, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.sum ({dtype_name})", "Sum", "Reduction", dtype_name
    results.append(r)

    def method_sum(): return a.sum()
    r = benchmark(method_sum, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a.sum() [method] ({dtype_name})", "Sum", "Reduction", dtype_name
    results.append(r)

    def np_sum_axis0(): return np.sum(a_2d, axis=0)
    r = benchmark(np_sum_axis0, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.sum axis=0 ({dtype_name})", "Sum", "Reduction", dtype_name
    results.append(r)

    def np_sum_axis1(): return np.sum(a_2d, axis=1)
    r = benchmark(np_sum_axis1, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.sum axis=1 ({dtype_name})", "Sum", "Reduction", dtype_name
    results.append(r)

    # Mean
    def np_mean(): return np.mean(a)
    r = benchmark(np_mean, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.mean ({dtype_name})", "Mean", "Reduction", dtype_name
    results.append(r)

    def method_mean(): return a.mean()
    r = benchmark(method_mean, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a.mean() [method] ({dtype_name})", "Mean", "Reduction", dtype_name
    results.append(r)

    # Var/Std. NumPy promotes int32 accumulators to float64.
    if dtype_name == "int32" or np.issubdtype(DTYPES[dtype_name], np.floating):
        # float16 var/std sum SQUARED deviations — all positive, so (unlike the mean, whose terms
        # cancel) the float16 accumulator (max 65504) saturates to +inf for these [-50,50) magnitudes
        # at every N (sum-of-squares ~= 833*N). NumSharp widens Half reductions to f32 and stays finite
        # (~833); the kernel streams all N either way, so the TIMING is identical — we only silence
        # NumPy's benign overflow RuntimeWarning. (float32/float64 never overflow; nothing else does.)
        with np.errstate(over='ignore'):
            def np_var(): return np.var(a)
            r = benchmark(np_var, n, iterations=iterations)
            r.name, r.category, r.suite, r.dtype = f"np.var ({dtype_name})", "VarStd", "Reduction", dtype_name
            results.append(r)

            def method_var(): return a.var()
            r = benchmark(method_var, n, iterations=iterations)
            r.name, r.category, r.suite, r.dtype = f"a.var() [method] ({dtype_name})", "VarStd", "Reduction", dtype_name
            results.append(r)

            def np_std(): return np.std(a)
            r = benchmark(np_std, n, iterations=iterations)
            r.name, r.category, r.suite, r.dtype = f"np.std ({dtype_name})", "VarStd", "Reduction", dtype_name
            results.append(r)

            def method_std(): return a.std()
            r = benchmark(method_std, n, iterations=iterations)
            r.name, r.category, r.suite, r.dtype = f"a.std() [method] ({dtype_name})", "VarStd", "Reduction", dtype_name
            results.append(r)

    # Min/Max
    def np_amin(): return np.amin(a)
    r = benchmark(np_amin, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.amin ({dtype_name})", "MinMax", "Reduction", dtype_name
    results.append(r)

    def method_amin(): return a.min()
    r = benchmark(method_amin, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a.amin() [method] ({dtype_name})", "MinMax", "Reduction", dtype_name
    results.append(r)

    def np_min(): return np.min(a)
    r = benchmark(np_min, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.min ({dtype_name})", "MinMax", "Reduction", dtype_name
    results.append(r)

    def np_amax(): return np.amax(a)
    r = benchmark(np_amax, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.amax ({dtype_name})", "MinMax", "Reduction", dtype_name
    results.append(r)

    def method_amax(): return a.max()
    r = benchmark(method_amax, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a.amax() [method] ({dtype_name})", "MinMax", "Reduction", dtype_name
    results.append(r)

    def np_max(): return np.max(a)
    r = benchmark(np_max, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.max ({dtype_name})", "MinMax", "Reduction", dtype_name
    results.append(r)

    # ArgMin/ArgMax
    def np_argmin(): return np.argmin(a)
    r = benchmark(np_argmin, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.argmin ({dtype_name})", "ArgMinMax", "Reduction", dtype_name
    results.append(r)

    def np_argmax(): return np.argmax(a)
    r = benchmark(np_argmax, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.argmax ({dtype_name})", "ArgMinMax", "Reduction", dtype_name
    results.append(r)

    # Cumulative sum (mirror C# SumBenchmarks.CumSum) — all arithmetic dtypes.
    def np_cumsum(): return np.cumsum(a)
    r = benchmark(np_cumsum, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.cumsum ({dtype_name})", "Sum", "Reduction", dtype_name
    results.append(r)

    # Axis min/max + mean (mirror C# MinMaxBenchmarks / MeanBenchmarks axis variants).
    def np_amin_axis0(): return np.amin(a_2d, axis=0)
    r = benchmark(np_amin_axis0, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.amin axis=0 ({dtype_name})", "MinMax", "Reduction", dtype_name
    results.append(r)

    def np_amax_axis0(): return np.amax(a_2d, axis=0)
    r = benchmark(np_amax_axis0, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.amax axis=0 ({dtype_name})", "MinMax", "Reduction", dtype_name
    results.append(r)

    def np_mean_axis0(): return np.mean(a_2d, axis=0)
    r = benchmark(np_mean_axis0, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.mean axis=0 ({dtype_name})", "Mean", "Reduction", dtype_name
    results.append(r)

    def np_mean_axis1(): return np.mean(a_2d, axis=1)
    r = benchmark(np_mean_axis1, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.mean axis=1 ({dtype_name})", "Mean", "Reduction", dtype_name
    results.append(r)

    # Axis var/std (mirror C# VarStdBenchmarks axis variants).
    if dtype_name == "int32" or np.issubdtype(DTYPES[dtype_name], np.floating):
        # Same float16 sum-of-squares saturation as the flat var/std above — silence the benign
        # overflow; the per-column reductions still time identically (see note above).
        with np.errstate(over='ignore'):
            def np_var_axis0(): return np.var(a_2d, axis=0)
            r = benchmark(np_var_axis0, n, iterations=iterations)
            r.name, r.category, r.suite, r.dtype = f"np.var axis=0 ({dtype_name})", "VarStd", "Reduction", dtype_name
            results.append(r)

            def np_std_axis0(): return np.std(a_2d, axis=0)
            r = benchmark(np_std_axis0, n, iterations=iterations)
            r.name, r.category, r.suite, r.dtype = f"np.std axis=0 ({dtype_name})", "VarStd", "Reduction", dtype_name
            results.append(r)

    return results

# =============================================================================
# Broadcasting Benchmarks
# =============================================================================

def run_broadcast_benchmarks(n: int, dtype_name: str, iterations: int) -> List[BenchmarkResult]:
    """Benchmark broadcasting operations."""
    results = []

    np.random.seed(42)
    matrix_size = int(np.sqrt(n))
    matrix = create_random_array(matrix_size * matrix_size, dtype_name).reshape(matrix_size, matrix_size)
    row_vector = create_random_array(matrix_size, dtype_name)
    col_vector = create_random_array(matrix_size, dtype_name, seed=43).reshape(matrix_size, 1)
    scalar = np.array(42, dtype=DTYPES[dtype_name])

    # Scalar broadcast
    def broadcast_scalar(): return matrix + scalar
    r = benchmark(broadcast_scalar, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "matrix + scalar", "Scalar", "Broadcasting", dtype_name
    results.append(r)

    def broadcast_scalar_literal(): return matrix * 2.0
    r = benchmark(broadcast_scalar_literal, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "matrix * 2.0", "Scalar", "Broadcasting", dtype_name
    results.append(r)

    # Row broadcast
    def broadcast_row(): return matrix + row_vector
    r = benchmark(broadcast_row, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "matrix + row_vector (N,M)+(M,)", "Row", "Broadcasting", dtype_name
    results.append(r)

    def broadcast_row_mul(): return matrix * row_vector
    r = benchmark(broadcast_row_mul, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "matrix * row_vector (N,M)*(M,)", "Row", "Broadcasting", dtype_name
    results.append(r)

    # Column broadcast
    def broadcast_col(): return matrix + col_vector
    r = benchmark(broadcast_col, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "matrix + col_vector (N,M)+(N,1)", "Column", "Broadcasting", dtype_name
    results.append(r)

    def broadcast_col_mul(): return matrix * col_vector
    r = benchmark(broadcast_col_mul, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "matrix * col_vector (N,M)*(N,1)", "Column", "Broadcasting", dtype_name
    results.append(r)

    d = int(n ** (1 / 3))
    tensor3d = create_random_array(d * d * d, dtype_name).reshape(d, d, d)
    broadcast2d = create_random_array(d * d, dtype_name, seed=43).reshape(d, d)
    def broadcast_3d(): return tensor3d + broadcast2d
    r = benchmark(broadcast_3d, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "tensor3D + matrix2D (D,D,D)+(D,D)", "3D", "Broadcasting", dtype_name
    results.append(r)

    # broadcast_to
    def broadcast_to_row(): return np.broadcast_to(row_vector, (matrix_size, matrix_size))
    r = benchmark(broadcast_to_row, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.broadcast_to(row, (N,M))", "BroadcastTo", "Broadcasting", dtype_name
    results.append(r)

    def broadcast_to_col(): return np.broadcast_to(col_vector, (matrix_size, matrix_size))
    r = benchmark(broadcast_to_col, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.broadcast_to(col, (N,M))", "BroadcastTo", "Broadcasting", dtype_name
    results.append(r)

    return results

# =============================================================================
# Creation Benchmarks
# =============================================================================

def run_creation_benchmarks(n: int, dtype_name: str, iterations: int) -> List[BenchmarkResult]:
    """Benchmark array creation functions."""
    results = []
    dtype = DTYPES[dtype_name]
    source = create_random_array(n, dtype_name)

    # Initialized arrays
    def np_zeros(): return np.zeros(n, dtype=dtype)
    r = benchmark(np_zeros, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.zeros ({dtype_name})", "Initialized", "Creation", dtype_name
    results.append(r)

    def np_ones(): return np.ones(n, dtype=dtype)
    r = benchmark(np_ones, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.ones ({dtype_name})", "Initialized", "Creation", dtype_name
    results.append(r)

    def np_full(): return np.full(n, 42, dtype=dtype)
    r = benchmark(np_full, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.full ({dtype_name})", "Initialized", "Creation", dtype_name
    results.append(r)

    def np_empty(): return np.empty(n, dtype=dtype)
    r = benchmark(np_empty, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.empty ({dtype_name})", "Uninitialized", "Creation", dtype_name
    results.append(r)

    # Copy
    def np_copy(): return np.copy(source)
    r = benchmark(np_copy, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.copy ({dtype_name})", "Copy", "Creation", dtype_name
    results.append(r)

    def method_copy(): return source.copy()
    r = benchmark(method_copy, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a.copy() ({dtype_name})", "Copy", "Creation", dtype_name
    results.append(r)

    def np_arange(): return np.arange(n, dtype=dtype)
    r = benchmark(np_arange, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.arange(N) ({dtype_name})", "Range", "Creation", dtype_name
    results.append(r)

    def np_linspace(): return np.linspace(0, n, n, dtype=dtype)
    r = benchmark(np_linspace, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.linspace(0, N, N) ({dtype_name})", "Range", "Creation", dtype_name
    results.append(r)

    # Like-based
    def np_zeros_like(): return np.zeros_like(source)
    r = benchmark(np_zeros_like, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.zeros_like ({dtype_name})", "Like", "Creation", dtype_name
    results.append(r)

    def np_ones_like(): return np.ones_like(source)
    r = benchmark(np_ones_like, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.ones_like(a) ({dtype_name})", "Like", "Creation", dtype_name
    results.append(r)

    def np_empty_like(): return np.empty_like(source)
    r = benchmark(np_empty_like, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.empty_like(a) ({dtype_name})", "Like", "Creation", dtype_name
    results.append(r)

    def np_full_like(): return np.full_like(source, 42)
    r = benchmark(np_full_like, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.full_like(a, 42) ({dtype_name})", "Like", "Creation", dtype_name
    results.append(r)

    if dtype_name in {"int32", "float64"} and n <= ARRAY_SIZES["large"]:
        work_n = memory_heavy_work_n(n)
        conversion_source = create_random_array(work_n, dtype_name)
        side = int(work_n ** 0.5)
        matrix = create_random_array(side * side, dtype_name).reshape(side, side)
        strided = conversion_source[::2]
        axis = np.arange(side, dtype=DTYPES[dtype_name])
        buffer = bytearray(work_n * np.dtype(DTYPES[dtype_name]).itemsize)
        text = " ".join(str(i % 100) for i in range(work_n))
        conversion_cases = [
            ("np.array(a)", lambda: np.array(conversion_source)),
            ("np.asarray(a)", lambda: np.asarray(conversion_source)),
            ("np.asanyarray(a)", lambda: np.asanyarray(conversion_source)),
            ("np.asarray_chkfinite(a)", lambda: np.asarray_chkfinite(conversion_source)),
            ("np.ascontiguousarray(a)", lambda: np.ascontiguousarray(strided)),
            ("np.asfortranarray(a)", lambda: np.asfortranarray(matrix)),
            ("np.asmatrix(a)", lambda: np.asmatrix(conversion_source)),
            ("np.require(a, requirements=C)", lambda: np.require(strided, requirements=["C"])),
            ("np.frombuffer(buffer)", lambda: np.frombuffer(buffer, dtype=DTYPES[dtype_name])),
            ("np.fromstring(text)", lambda: np.fromstring(text, dtype=DTYPES[dtype_name], sep=" ")),
            ("np.eye(n)", lambda: np.eye(side, dtype=DTYPES[dtype_name])),
            ("np.identity(n)", lambda: np.identity(side, dtype=DTYPES[dtype_name])),
            ("np.meshgrid(x, y)", lambda: np.meshgrid(axis, axis)),
            ("np.vander(x)", lambda: np.vander(axis)),
        ]
        for name, func in conversion_cases:
            results.append(_b(func, n, iterations, name, "Creation", dtype_name, "Conversion"))

    return results

# =============================================================================
# Manipulation Benchmarks
# =============================================================================

def run_manipulation_benchmarks(n: int, dtype_name: str, iterations: int) -> List[BenchmarkResult]:
    """Benchmark shape manipulation operations."""
    results = []

    np.random.seed(42)
    rows = int(np.sqrt(n))
    cols = n // rows
    actual_n = rows * cols  # May be slightly less than n due to integer division
    arr_1d = create_random_array(actual_n, dtype_name)  # Use actual_n to ensure reshape works
    arr_2d = create_random_array(rows * cols, dtype_name).reshape(rows, cols)
    d = int(n ** (1/3))
    arr_3d = create_random_array(d * d * d, dtype_name).reshape(d, d, d)
    arr_1d_for_3d = create_random_array(d * d * max(1, n // (d * d)), dtype_name)

    # Reshape
    def reshape_1d_2d(): return arr_1d.reshape(rows, cols)
    r = benchmark(reshape_1d_2d, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "reshape 1D->2D", "Reshape", "Manipulation", dtype_name
    results.append(r)

    def reshape_2d_1d(): return arr_2d.reshape(-1)
    r = benchmark(reshape_2d_1d, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "reshape 2D->1D", "Reshape", "Manipulation", dtype_name
    results.append(r)

    def reshape_1d_3d(): return arr_1d_for_3d.reshape(d, d, -1)
    r = benchmark(reshape_1d_3d, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "reshape 1D->3D", "Reshape", "Manipulation", dtype_name
    results.append(r)

    def np_reshape(): return np.reshape(arr_1d, (rows, cols))
    r = benchmark(np_reshape, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.reshape(a, shape)", "Reshape", "Manipulation", dtype_name
    results.append(r)

    # Transpose
    def transpose_2d(): return arr_2d.T
    r = benchmark(transpose_2d, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a.T (transpose 2D)", "Transpose", "Manipulation", dtype_name
    results.append(r)

    def np_transpose(): return np.transpose(arr_2d)
    r = benchmark(np_transpose, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.transpose(a)", "Transpose", "Manipulation", dtype_name
    results.append(r)

    def np_transpose_axes(): return np.transpose(arr_3d, (2, 0, 1))
    r = benchmark(np_transpose_axes, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.transpose(a, axes)", "Transpose", "Manipulation", dtype_name
    results.append(r)

    # Ravel/Flatten
    def np_ravel(): return np.ravel(arr_2d)
    r = benchmark(np_ravel, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.ravel", "Flatten", "Manipulation", dtype_name
    results.append(r)

    def np_flatten(): return arr_2d.flatten()
    r = benchmark(np_flatten, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a.flatten", "Flatten", "Manipulation", dtype_name
    results.append(r)

    def method_ravel(): return arr_2d.ravel()
    r = benchmark(method_ravel, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a.ravel() (view)", "Flatten", "Manipulation", dtype_name
    results.append(r)

    dims_source = create_random_array(rows * cols, dtype_name).reshape(rows, 1, cols)
    dim_cases = [
        ("np.expand_dims(a, axis=0)", lambda: np.expand_dims(arr_1d, axis=0)),
        ("np.expand_dims(a, axis=-1)", lambda: np.expand_dims(arr_1d, axis=-1)),
        ("np.expand_dims(a2D, axis=1)", lambda: np.expand_dims(arr_2d, axis=1)),
        ("np.squeeze(a)", lambda: np.squeeze(dims_source)),
        ("np.squeeze(a, axis=1)", lambda: np.squeeze(dims_source, axis=1)),
        ("np.swapaxes(a, 0, 1)", lambda: np.swapaxes(arr_2d, 0, 1)),
        ("np.swapaxes(a3D, 0, 2)", lambda: np.swapaxes(arr_3d, 0, 2)),
        ("np.moveaxis(a, 0, -1)", lambda: np.moveaxis(arr_3d, 0, -1)),
        ("np.rollaxis(a, 2)", lambda: np.rollaxis(arr_3d, 2)),
    ]
    for name, func in dim_cases:
        results.append(_b(func, n, iterations, name, "Manipulation", dtype_name, "Dims"))

    # Stack — the universal 10M tier uses 1M physical elements for this allocation-heavy family.
    if n <= ARRAY_SIZES["large"]:
        stack_n = memory_heavy_work_n(n)
        stack_rows = int(np.sqrt(stack_n))
        stack_cols = stack_n // stack_rows
        stack_actual_n = stack_rows * stack_cols
        stack_1d_a = create_random_array(stack_actual_n, dtype_name)
        stack_1d_b = create_random_array(stack_actual_n, dtype_name, seed=43)
        stack_2d = create_random_array(stack_actual_n, dtype_name).reshape(stack_rows, stack_cols)
        stack_cases = [
            ("np.concatenate([a, b])", lambda: np.concatenate([stack_1d_a, stack_1d_b])),
            ("np.concatenate([a, b, c])", lambda: np.concatenate([stack_1d_a, stack_1d_b, stack_1d_a])),
            ("np.concatenate([a, b], axis=0)", lambda: np.concatenate([stack_2d, stack_2d], axis=0)),
            ("np.concatenate([a, b], axis=1)", lambda: np.concatenate([stack_2d, stack_2d], axis=1)),
            ("np.stack([a, b])", lambda: np.stack([stack_1d_a, stack_1d_b])),
            ("np.stack([a, b], axis=1)", lambda: np.stack([stack_1d_a, stack_1d_b], axis=1)),
            ("np.hstack([a, b])", lambda: np.hstack([stack_1d_a, stack_1d_b])),
            ("np.vstack([a, b])", lambda: np.vstack([stack_1d_a, stack_1d_b])),
            ("np.dstack([a, b])", lambda: np.dstack([stack_2d, stack_2d])),
        ]
        for name, func in stack_cases:
            results.append(_b(func, n, iterations, name, "Manipulation", dtype_name, "Stack"))

    # Reversal / rotation / transpose-alias views + the trim_zeros crop (journey2 additions —
    # twins of Benchmarks/Manipulation/FlipRotBenchmarks.cs). The first six are O(1)/O(ndim) pure
    # views; trim_zeros does an O(N) nonzero bounding-box scan. Names normalize 1:1 onto the C#
    # [Benchmark(Description)] labels ("np.flip(a)" -> "np.flip", ...) for the (op, dtype, N) join.
    def np_flip(): return np.flip(arr_2d)
    r = benchmark(np_flip, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.flip", "Flip", "Manipulation", dtype_name
    results.append(r)

    def np_fliplr(): return np.fliplr(arr_2d)
    r = benchmark(np_fliplr, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.fliplr", "Flip", "Manipulation", dtype_name
    results.append(r)

    def np_flipud(): return np.flipud(arr_2d)
    r = benchmark(np_flipud, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.flipud", "Flip", "Manipulation", dtype_name
    results.append(r)

    def np_rot90(): return np.rot90(arr_2d)
    r = benchmark(np_rot90, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.rot90", "Rot90", "Manipulation", dtype_name
    results.append(r)

    def np_permute_dims(): return np.permute_dims(arr_2d)
    r = benchmark(np_permute_dims, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.permute_dims", "Transpose", "Manipulation", dtype_name
    results.append(r)

    def np_matrix_transpose(): return np.matrix_transpose(arr_2d)
    r = benchmark(np_matrix_transpose, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.matrix_transpose", "Transpose", "Manipulation", dtype_name
    results.append(r)

    def np_trim_zeros(): return np.trim_zeros(arr_2d, 'fb')
    r = benchmark(np_trim_zeros, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.trim_zeros", "TrimZeros", "Manipulation", dtype_name
    results.append(r)

    # byteswap: reverse each element's bytes (endian toggle) into a fresh copy — the one
    # O(N) byte-permutation op here (a fused read->VPSHUFB->write pass, non-temporal above the
    # L2-scale threshold). Twin of Benchmarks/Manipulation/ByteswapBenchmarks.cs; "np.byteswap"
    # normalizes onto the C# "np.byteswap(a)" label for the (op, dtype, N) join.
    def np_byteswap(): return arr_2d.byteswap()
    r = benchmark(np_byteswap, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.byteswap", "Byteswap", "Manipulation", dtype_name
    results.append(r)

    # The two-dimensional base family: diag / diagflat / tri / tril / triu /
    # fill_diagonal / tril_indices / triu_indices. Three distinct cost classes, kept apart
    # deliberately: diag(2-D) is an O(1) read-only diagonal VIEW (measures dispatch overhead);
    # tri/tril/triu are the O(N) workers where NumPy pays for an N*M bool mask plus a full
    # `where` select; fill_diagonal and diag(1-D)/diagflat are sparse/alloc-bound. The 1-D
    # input is sized to sqrt(n) so the CONSTRUCTED n x n matrix stays ~n elements.
    arr_1d_sq = create_random_array(rows, dtype_name)
    fill_target = create_random_array(rows * cols, dtype_name).reshape(rows, cols)

    def np_diag_2d(): return np.diag(arr_2d)
    r = benchmark(np_diag_2d, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.diag", "Diag", "Manipulation", dtype_name
    results.append(r)

    def np_diag_1d(): return np.diag(arr_1d_sq)
    r = benchmark(np_diag_1d, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.diag(a1d)", "Diag", "Manipulation", dtype_name
    results.append(r)

    def np_diagflat(): return np.diagflat(arr_1d_sq)
    r = benchmark(np_diagflat, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.diagflat", "Diag", "Manipulation", dtype_name
    results.append(r)

    def np_tri(): return np.tri(rows, cols, dtype=DTYPES[dtype_name])
    r = benchmark(np_tri, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.tri", "Tri", "Manipulation", dtype_name
    results.append(r)

    def np_tril(): return np.tril(arr_2d)
    r = benchmark(np_tril, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.tril", "Tri", "Manipulation", dtype_name
    results.append(r)

    def np_triu(): return np.triu(arr_2d)
    r = benchmark(np_triu, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.triu", "Tri", "Manipulation", dtype_name
    results.append(r)

    def np_fill_diagonal(): np.fill_diagonal(fill_target, DTYPES[dtype_name](1)); return fill_target
    r = benchmark(np_fill_diagonal, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.fill_diagonal", "FillDiagonal", "Manipulation", dtype_name
    results.append(r)

    def np_tril_indices(): return np.tril_indices(rows, 0, cols)[0]
    r = benchmark(np_tril_indices, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.tril_indices", "TriIndices", "Manipulation", dtype_name
    results.append(r)

    def np_triu_indices(): return np.triu_indices(rows, 0, cols)[0]
    r = benchmark(np_triu_indices, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.triu_indices", "TriIndices", "Manipulation", dtype_name
    results.append(r)

    # The index-expression DSL: r_ / c_ / ix_. Three cost classes, kept apart —
    # O(N) concatenation (r_/c_ over arrays, with and without a weak-scalar tail, which prices
    # the per-entry NEP50 promotion), O(N) generation (the arange and imaginary-step linspace
    # branches, no input array at all), and the O(1) open-mesh VIEW ix_ builds by inserting
    # length-1 axes, whose timing is construction overhead rather than throughput.
    arr_1d_b = create_random_array(actual_n, dtype_name, seed=43)
    ix_rows = np.arange(rows)
    ix_cols = np.arange(rows)

    def np_r_two(): return np.r_[arr_1d, arr_1d_b]
    r = benchmark(np_r_two, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.r_", "IndexTricks", "Manipulation", dtype_name
    results.append(r)

    def np_r_scalars(): return np.r_[arr_1d, DTYPES[dtype_name](0), DTYPES[dtype_name](0), arr_1d_b]
    r = benchmark(np_r_scalars, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.r_(a, 0, 0, b)", "IndexTricks", "Manipulation", dtype_name
    results.append(r)

    def np_c_two(): return np.c_[arr_1d, arr_1d_b]
    r = benchmark(np_c_two, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.c_", "IndexTricks", "Manipulation", dtype_name
    results.append(r)

    def np_r_arange(): return np.r_[0:n]
    r = benchmark(np_r_arange, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.r_(0:N)", "IndexTricks", "Manipulation", dtype_name
    results.append(r)

    def np_r_linspace(): return np.r_[0:1:complex(0, n)]
    r = benchmark(np_r_linspace, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.r_(0:1:Nj)", "IndexTricks", "Manipulation", dtype_name
    results.append(r)

    def np_ix(): return np.ix_(ix_rows, ix_cols)
    r = benchmark(np_ix, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.ix_", "IndexTricks", "Manipulation", dtype_name
    results.append(r)

    if n <= ARRAY_SIZES["large"]:
        work_n = memory_heavy_work_n(n)
        np.random.seed(42)
        # Set operations are meaningful for both the historical int32 corpus and the universal
        # float64 benchmark profile. Keep identical integer-valued samples after casting so the
        # two dtype rows differ only in representation, not cardinality/distribution.
        set_seed_a = np.random.randint(0, max(2, work_n // 2), work_n, dtype=np.int32)
        set_seed_b = np.random.randint(0, max(2, work_n // 2), work_n, dtype=np.int32)
        set_a = set_seed_a.astype(DTYPES[dtype_name], copy=False)
        set_b = set_seed_b.astype(DTYPES[dtype_name], copy=False)
        set_cases = [
            ("np.intersect1d(a, b)", lambda: np.intersect1d(set_a, set_b)),
            ("np.isin(a, b)", lambda: np.isin(set_a, set_b)),
            ("np.setdiff1d(a, b)", lambda: np.setdiff1d(set_a, set_b)),
            ("np.setxor1d(a, b)", lambda: np.setxor1d(set_a, set_b)),
            ("np.union1d(a, b)", lambda: np.union1d(set_a, set_b)),
            ("np.unique(a)", lambda: np.unique(set_a)),
            ("np.unique_all(a)", lambda: np.unique_all(set_a)),
            ("np.unique_counts(a)", lambda: np.unique_counts(set_a)),
            ("np.unique_inverse(a)", lambda: np.unique_inverse(set_a)),
            ("np.unique_values(a)", lambda: np.unique_values(set_a)),
        ]
        for name, func in set_cases:
            results.append(_b(func, n, iterations, name, "Manipulation", dtype_name, "SetOperations"))

        shape_a = create_random_array(work_n, dtype_name)
        shape_b = create_random_array(work_n, dtype_name, seed=43)
        shape_matrix = create_random_array(work_n, dtype_name).reshape(10, work_n // 10)
        shape_cube = create_random_array(work_n, dtype_name).reshape(10, 10, work_n // 100)
        shape_row = np.arange(work_n // 10, dtype=DTYPES[dtype_name])
        shape_cases = [
            ("np.append(a, b)", lambda: np.append(shape_a, shape_b)),
            ("np.array_split(a, sections)", lambda: np.array_split(shape_a, 7)),
            ("np.atleast_1d(a)", lambda: np.atleast_1d(shape_a)),
            ("np.atleast_2d(a)", lambda: np.atleast_2d(shape_a)),
            ("np.atleast_3d(a)", lambda: np.atleast_3d(shape_a)),
            ("np.block([a, b])", lambda: np.block([shape_a, shape_b])),
            ("np.broadcast_arrays(a, b)", lambda: np.broadcast_arrays(shape_matrix, shape_row)),
            ("np.column_stack((a, b))", lambda: np.column_stack((shape_a, shape_b))),
            ("np.concat((a, b))", lambda: np.concat((shape_a, shape_b))),
            ("np.delete(a, index)", lambda: np.delete(shape_a, work_n // 2)),
            ("np.insert(a, index, value)", lambda: np.insert(shape_a, work_n // 2, DTYPES[dtype_name](0))),
            ("np.dsplit(a, sections)", lambda: np.dsplit(shape_cube, 10)),
            ("np.hsplit(a, sections)", lambda: np.hsplit(shape_matrix, 10)),
            ("np.vsplit(a, sections)", lambda: np.vsplit(shape_matrix, 10)),
            ("np.split(a, sections)", lambda: np.split(shape_a, 10)),
            ("np.pad(a, width)", lambda: np.pad(shape_a, 16)),
            ("np.repeat(a, repeats)", lambda: np.repeat(shape_a, 2)),
            ("np.resize(a, shape)", lambda: np.resize(shape_a, work_n * 2)),
            ("np.roll(a, shift)", lambda: np.roll(shape_a, work_n // 2)),
            ("np.size(a)", lambda: np.size(shape_a)),
            ("np.tile(a, reps)", lambda: np.tile(shape_a, 2)),
            ("np.unstack(a)", lambda: np.unstack(shape_matrix)),
        ]
        for name, func in shape_cases:
            results.append(_b(func, n, iterations, name, "Manipulation", dtype_name, "ExtendedShape"))

    return results

# =============================================================================
# Slicing Benchmarks
# =============================================================================

def run_slicing_benchmarks(n: int, dtype_name: str, iterations: int) -> List[BenchmarkResult]:
    """Benchmark slicing operations."""
    results = []

    np.random.seed(42)
    arr_1d = create_random_array(n, dtype_name)
    rows = int(np.sqrt(n))
    cols = n // rows
    arr_2d = create_random_array(rows * cols, dtype_name).reshape(rows, cols)

    # Contiguous slice
    contiguous_slice = arr_1d[100:1000]
    strided_slice = arr_1d[::2]

    # Slice creation
    def slice_contiguous(): return arr_1d[100:1000]
    r = benchmark(slice_contiguous, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a[100:1000] (contiguous slice)", "Create", "Slicing", dtype_name
    results.append(r)

    def slice_strided(): return arr_1d[::2]
    r = benchmark(slice_strided, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a[::2] (strided slice)", "Create", "Slicing", dtype_name
    results.append(r)

    def slice_row_2d(): return arr_2d[10:100, :]
    r = benchmark(slice_row_2d, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a[10:100, :] (row slice 2D)", "Create", "Slicing", dtype_name
    results.append(r)

    def slice_col_2d(): return arr_2d[:, 10:100]
    r = benchmark(slice_col_2d, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a[:, 10:100] (col slice 2D)", "Create", "Slicing", dtype_name
    results.append(r)

    def slice_reversed(): return arr_1d[::-1]
    r = benchmark(slice_reversed, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a[::-1] (reversed)", "Create", "Slicing", dtype_name
    results.append(r)

    # Operations on slices
    def sum_contiguous(): return np.sum(contiguous_slice)
    r = benchmark(sum_contiguous, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.sum(contiguous_slice)", "SumSlice", "Slicing", dtype_name
    results.append(r)

    def sum_strided(): return np.sum(strided_slice)
    r = benchmark(sum_strided, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.sum(strided_slice)", "SumSlice", "Slicing", dtype_name
    results.append(r)

    row_slice = arr_2d[10:100, :]
    col_slice = arr_2d[:, 10:100]

    def sum_row(): return np.sum(row_slice)
    r = benchmark(sum_row, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.sum(row_slice)", "SumSlice", "Slicing", dtype_name
    results.append(r)

    def sum_col(): return np.sum(col_slice)
    r = benchmark(sum_col, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.sum(col_slice)", "SumSlice", "Slicing", dtype_name
    results.append(r)

    # Middle 80% ([n/10 : 9n/10]) so the copy/multiply work scales with N — the fixed
    # "100:1000" did the same ~900 elements at every size (non-comparable across N).
    lo, hi = n // 10, n // 10 * 9

    def slice_op_contiguous(): return arr_1d[lo:hi] * 2
    r = benchmark(slice_op_contiguous, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a[N/10:9N/10] * 2", "SliceOp", "Slicing", dtype_name
    results.append(r)

    def slice_op_strided(): return arr_1d[::2] * 2
    r = benchmark(slice_op_strided, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a[::2] * 2", "SliceOp", "Slicing", dtype_name
    results.append(r)

    def slice_copy(): return arr_1d[lo:hi].copy()
    r = benchmark(slice_copy, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a[N/10:9N/10].copy()", "Copy", "Slicing", dtype_name
    results.append(r)

    def np_slice_copy(): return np.copy(arr_1d[lo:hi])
    r = benchmark(np_slice_copy, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.copy(a[N/10:9N/10])", "Copy", "Slicing", dtype_name
    results.append(r)

    return results

# =============================================================================
# Dispatch Benchmarks (matching DispatchBenchmarks.cs)
# =============================================================================

def run_dispatch_benchmarks(n: int, iterations: int) -> List[BenchmarkResult]:
    """Compare different ways to perform c = a + b."""
    print(f"\n{'='*60}")
    print(f"  Dispatch Benchmarks (int32, N={n:,})")
    print(f"{'='*60}\n")

    # Setup
    np.random.seed(42)
    a = np.random.randint(0, 100, n, dtype=np.int32)
    b = np.random.randint(0, 100, n, dtype=np.int32)
    c = np.empty(n, dtype=np.int32)

    results = []

    # np.add with pre-allocated output
    def numpy_add_out():
        np.add(a, b, out=c)
    r = benchmark(numpy_add_out, n, iterations=iterations)
    r.name = "np.add(a, b, out=c)"
    r.category = "Dispatch"
    r.suite = "Dispatch"
    r.dtype = "int32"
    results.append(r)
    print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms (±{r.stddev_ms:.3f})")

    # Operator syntax (allocates new array)
    def numpy_operator():
        return a + b
    r = benchmark(numpy_operator, n, iterations=iterations)
    r.name = "c = a + b (allocates)"
    r.category = "Dispatch"
    r.suite = "Dispatch"
    r.dtype = "int32"
    results.append(r)
    print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms (±{r.stddev_ms:.3f})")

    # Scalar broadcast
    scalar = b[0]
    def numpy_broadcast():
        np.add(a, scalar, out=c)
    r = benchmark(numpy_broadcast, n, iterations=iterations)
    r.name = "np.add(a, scalar, out=c)"
    r.category = "Dispatch"
    r.suite = "Dispatch"
    r.dtype = "int32"
    results.append(r)
    print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms (±{r.stddev_ms:.3f})")

    return results

# =============================================================================
# Fusion Benchmarks (matching FusionBenchmarks.cs)
# =============================================================================

def run_fusion_benchmarks(n: int, iterations: int) -> List[BenchmarkResult]:
    """Compare compound expressions - NumPy cannot fuse these."""
    print(f"\n{'='*60}")
    print(f"  Fusion Benchmarks (float64, N={n:,})")
    print(f"{'='*60}\n")

    # Setup
    np.random.seed(42)
    a = np.random.random(n) * 10
    b = np.random.random(n) * 10
    mean_val = np.mean(a)

    results = []

    # Pattern 1: c = a * a
    print("  --- Pattern 1: c = a * a ---")
    def p1_numpy():
        return a * a
    r = benchmark(p1_numpy, n, iterations=iterations)
    r.name = "NumPy: a * a"
    r.category = "Pattern1_Square"
    r.suite = "Fusion"
    r.dtype = "float64"
    results.append(r)
    print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms")

    # Pattern 2: c = a*a + 2*b
    print("\n  --- Pattern 2: c = a*a + 2*b ---")
    def p2_numpy():
        return a*a + 2*b
    r = benchmark(p2_numpy, n, iterations=iterations)
    r.name = "NumPy: a*a + 2*b"
    r.category = "Pattern2_AaBb"
    r.suite = "Fusion"
    r.dtype = "float64"
    results.append(r)
    print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms")

    # Pattern 3: variance
    print("\n  --- Pattern 3: variance ---")
    def p3_manual():
        return np.sum((a - mean_val) ** 2) / n
    r = benchmark(p3_manual, n, iterations=iterations)
    r.name = "NumPy: sum((a-mean)**2)/N"
    r.category = "Pattern3_Variance"
    r.suite = "Fusion"
    r.dtype = "float64"
    results.append(r)
    print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms")

    def p3_optimized():
        return np.var(a)
    r = benchmark(p3_optimized, n, iterations=iterations)
    r.name = "NumPy: np.var(a) [optimized]"
    r.category = "Pattern3_Variance"
    r.suite = "Fusion"
    r.dtype = "float64"
    results.append(r)
    print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms")

    # Pattern 4: polynomial a³ + a² + a
    print("\n  --- Pattern 4: c = a³ + a² + a ---")
    def p4_power():
        return a**3 + a**2 + a
    r = benchmark(p4_power, n, iterations=iterations)
    r.name = "NumPy: a**3 + a**2 + a"
    r.category = "Pattern4_Polynomial"
    r.suite = "Fusion"
    r.dtype = "float64"
    results.append(r)
    print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms")

    def p4_multiply():
        return a*a*a + a*a + a
    r = benchmark(p4_multiply, n, iterations=iterations)
    r.name = "NumPy: a*a*a + a*a + a"
    r.category = "Pattern4_Polynomial"
    r.suite = "Fusion"
    r.dtype = "float64"
    results.append(r)
    print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms")

    # Pattern 5: euclidean sqrt(a² + b²)
    print("\n  --- Pattern 5: c = sqrt(a² + b²) ---")
    def p5_manual():
        return np.sqrt(a**2 + b**2)
    r = benchmark(p5_manual, n, iterations=iterations)
    r.name = "NumPy: sqrt(a**2 + b**2)"
    r.category = "Pattern5_Euclidean"
    r.suite = "Fusion"
    r.dtype = "float64"
    results.append(r)
    print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms")

    def p5_hypot():
        return np.hypot(a, b)
    r = benchmark(p5_hypot, n, iterations=iterations)
    r.name = "NumPy: np.hypot(a, b) [optimized]"
    r.category = "Pattern5_Euclidean"
    r.suite = "Fusion"
    r.dtype = "float64"
    results.append(r)
    print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms")

    return results

# =============================================================================
# Main
# =============================================================================

def print_summary(results: List[BenchmarkResult]):
    """Print a summary table of all results."""
    try:
        from tabulate import tabulate
        headers = ["Name", "Suite", "DType", "N", "Mean (ms)", "StdDev"]
        rows = [
            [r.name, r.suite, r.dtype, f"{r.n:,}", f"{r.mean_ms:.3f}", f"{r.stddev_ms:.3f}"]
            for r in results
        ]
        print(f"\n{'='*80}")
        print("  SUMMARY")
        print(f"{'='*80}")
        print(tabulate(rows, headers=headers, tablefmt="github"))
    except ImportError:
        print("\n(Install 'tabulate' for formatted table output: pip install tabulate)")

# =============================================================================
# Extended coverage suites (mirror the new C# benchmark classes 1:1 by op name)
# =============================================================================

# dtype sets that mirror the C# TypeParameterSource collections.
BITWISE_DTYPES = ['bool', 'uint8', 'int8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
FLOAT_DTYPES = ['float16', 'float32', 'float64']


def _b(func, n, iterations, name, suite, dtype, category=""):
    """Run one benchmark and tag it. name MUST equal the C# [Benchmark(Description=...)]."""
    if dtype not in ACTIVE_DTYPES:
        return None
    r = benchmark(func, n, iterations=iterations)
    r.name = f"{name} ({dtype})"
    r.suite = suite
    r.dtype = dtype
    r.category = category or suite
    return r


def _rejection_evidence(func, n, name, suite, dtype, category="UnsupportedDtype"):
    """Execute and prove an expected NumPy dtype rejection without timing exception overhead.

    These rows make the function × dtype inventory complete, but zero duration forces the merge
    classifier's measured-negligible path. An exception-construction ratio is not array throughput
    and must never enter a geomean, ranking, explorer, or scoreboard evidence list.
    """
    if dtype not in ACTIVE_DTYPES:
        return None
    try:
        func()
    except TypeError:
        pass
    else:
        raise AssertionError(f"{name} unexpectedly accepted dtype {dtype}")
    return BenchmarkResult(
        name=f"{name} ({dtype})", category=category, suite=suite, dtype=dtype, n=n,
        mean_ms=0.0, stddev_ms=0.0, min_ms=0.0, max_ms=0.0,
        iterations=1, ops_per_sec=0.0)


def run_comparison_benchmarks(n, dtype_name, iterations):
    a = create_random_array(n, dtype_name, seed=42)
    b = create_random_array(n, dtype_name, seed=43)
    return [
        _b(lambda: a == b, n, iterations, "a == b", "Comparison", dtype_name),
        _b(lambda: a != b, n, iterations, "a != b", "Comparison", dtype_name),
        _b(lambda: a < b,  n, iterations, "a < b",  "Comparison", dtype_name),
        _b(lambda: a > b,  n, iterations, "a > b",  "Comparison", dtype_name),
        _b(lambda: a <= b, n, iterations, "a <= b", "Comparison", dtype_name),
        _b(lambda: a >= b, n, iterations, "a >= b", "Comparison", dtype_name),
        _b(lambda: np.equal(a, b), n, iterations, "np.equal(a, b)", "Comparison", dtype_name),
        _b(lambda: np.not_equal(a, b), n, iterations, "np.not_equal(a, b)", "Comparison", dtype_name),
        _b(lambda: np.less(a, b), n, iterations, "np.less(a, b)", "Comparison", dtype_name),
        _b(lambda: np.greater(a, b), n, iterations, "np.greater(a, b)", "Comparison", dtype_name),
        _b(lambda: np.less_equal(a, b), n, iterations, "np.less_equal(a, b)", "Comparison", dtype_name),
        _b(lambda: np.greater_equal(a, b), n, iterations, "np.greater_equal(a, b)", "Comparison", dtype_name),
    ]


def run_bitwise_benchmarks(n, dtype_name, iterations):
    a = create_random_array(n, dtype_name, seed=42)
    b = create_random_array(n, dtype_name, seed=43)
    cases = [
        ("a & b", lambda: a & b),
        ("a | b", lambda: a | b),
        ("a ^ b", lambda: a ^ b),
        ("np.invert(a)", lambda: np.invert(a)),
        ("np.left_shift(a, 2)", lambda: np.left_shift(a, 2)),
        ("np.right_shift(a, 2)", lambda: np.right_shift(a, 2)),
        ("np.bitwise_and(a, b)", lambda: np.bitwise_and(a, b)),
        ("np.bitwise_or(a, b)", lambda: np.bitwise_or(a, b)),
        ("np.bitwise_xor(a, b)", lambda: np.bitwise_xor(a, b)),
        ("np.bitwise_not(a)", lambda: np.bitwise_not(a)),
    ]
    if dtype_name == "float64":
        return [_rejection_evidence(func, n, name, "Bitwise", dtype_name) for name, func in cases]
    return [_b(func, n, iterations, name, "Bitwise", dtype_name) for name, func in cases]


def run_unary_extra_benchmarks(n, dtype_name, iterations):
    a = create_positive_array(n, dtype_name, seed=42)
    return [
        _b(lambda: np.cbrt(a), n, iterations, "np.cbrt(a)", "Unary", dtype_name),
        _b(lambda: np.reciprocal(a), n, iterations, "np.reciprocal(a)", "Unary", dtype_name),
        _b(lambda: np.square(a), n, iterations, "np.square(a)", "Unary", dtype_name),
        _b(lambda: np.negative(a), n, iterations, "np.negative(a)", "Unary", dtype_name),
        _b(lambda: np.positive(a), n, iterations, "np.positive(a)", "Unary", dtype_name),
        _b(lambda: np.trunc(a), n, iterations, "np.trunc(a)", "Unary", dtype_name),
    ]


def run_logic_benchmarks(n, dtype_name, iterations):
    a = create_random_array(n, dtype_name, seed=42)
    b = create_random_array(n, dtype_name, seed=43)
    rows = [
        _b(lambda: np.isnan(a), n, iterations, "np.isnan(a)", "Logic", dtype_name),
        _b(lambda: np.isinf(a), n, iterations, "np.isinf(a)", "Logic", dtype_name),
        _b(lambda: np.isfinite(a), n, iterations, "np.isfinite(a)", "Logic", dtype_name),
        _b(lambda: np.maximum(a, b), n, iterations, "np.maximum(a, b)", "Logic", dtype_name),
        _b(lambda: np.minimum(a, b), n, iterations, "np.minimum(a, b)", "Logic", dtype_name),
        _b(lambda: np.fmax(a, b), n, iterations, "np.fmax(a, b)", "Logic", dtype_name),
        _b(lambda: np.fmin(a, b), n, iterations, "np.fmin(a, b)", "Logic", dtype_name),
        _b(lambda: np.iscomplex(a), n, iterations, "np.iscomplex(a)", "Logic", dtype_name),
        _b(lambda: np.isreal(a), n, iterations, "np.isreal(a)", "Logic", dtype_name),
        _b(lambda: np.iscomplexobj(a), n, iterations, "np.iscomplexobj(a)", "Logic", dtype_name),
        _b(lambda: np.isrealobj(a), n, iterations, "np.isrealobj(a)", "Logic", dtype_name),
        _b(lambda: np.isscalar(a), n, iterations, "np.isscalar(a)", "Logic", dtype_name),
        _b(lambda: np.isfortran(a), n, iterations, "np.isfortran(a)", "Logic", dtype_name),
        _b(lambda: np.array_equal(a, b), n, iterations, "np.array_equal(a, b)", "Logic", dtype_name),
    ]
    if n <= ARRAY_SIZES["large"]:
        close_n = memory_heavy_work_n(n)
        close_a = a[:close_n]
        close_b = b[:close_n]
        rows.extend([
            _b(lambda: np.isclose(close_a, close_b), n, iterations, "np.isclose(a, b)", "Logic", dtype_name, "Close"),
            _b(lambda: np.allclose(close_a, close_b), n, iterations, "np.allclose(a, b)", "Logic", dtype_name, "Close"),
        ])
    if dtype_name in {"int32", "float64"}:
        # logical_* and all/any accept arbitrary numeric inputs.
        rows.extend([
            _b(lambda: bool(np.all(a)), n, iterations, "np.all(a)", "Logic", dtype_name),
            _b(lambda: bool(np.any(a)), n, iterations, "np.any(a)", "Logic", dtype_name),
            _b(lambda: np.logical_and(a, b), n, iterations, "np.logical_and(a, b)", "Logic", dtype_name),
            _b(lambda: np.logical_or(a, b), n, iterations, "np.logical_or(a, b)", "Logic", dtype_name),
            _b(lambda: np.logical_xor(a, b), n, iterations, "np.logical_xor(a, b)", "Logic", dtype_name),
            _b(lambda: np.logical_not(a), n, iterations, "np.logical_not(a)", "Logic", dtype_name),
        ])
    return rows


def run_bool_logic_benchmarks(n, iterations):
    np.random.seed(42)
    mask = np.random.random(n) > 0.5
    other = np.random.random(n) > 0.5
    return [
        _b(lambda: bool(np.all(mask)), n, iterations, "np.all(a)", "Logic", "bool"),
        _b(lambda: bool(np.any(mask)), n, iterations, "np.any(a)", "Logic", "bool"),
        _b(lambda: np.logical_and(mask, other), n, iterations, "np.logical_and(a, b)", "Logic", "bool"),
        _b(lambda: np.logical_or(mask, other), n, iterations, "np.logical_or(a, b)", "Logic", "bool"),
        _b(lambda: np.logical_xor(mask, other), n, iterations, "np.logical_xor(a, b)", "Logic", "bool"),
        _b(lambda: np.logical_not(mask), n, iterations, "np.logical_not(a)", "Logic", "bool"),
    ]


def run_nan_reduction_benchmarks(n, dtype_name, iterations):
    a = create_random_array(n, dtype_name, seed=42)
    # nanprod multiplies the whole array; full-range data blows the product up to inf (and the C#
    # Decimal cell, which has no inf, threw OverflowException). Give nanprod its own [0.5, 1.0] input
    # so the product stays finite, mirroring the C# NanReductionBenchmarks._aProd (and ProdBenchmarks).
    np.random.seed(42)
    a_prod = (np.random.rand(n) * 0.5 + 0.5).astype(DTYPES[dtype_name])
    # float16 nanvar/nanstd saturate the f16 sum-of-squares accumulator to +inf (positive terms, no
    # cancellation like the mean) at every N; NumSharp widens Half and stays finite. The kernels stream
    # all N either way, so silencing NumPy's benign overflow RuntimeWarning leaves the timing unchanged.
    # (nanprod uses the bounded a_prod above; nansum/nanmean cancel — neither overflows. See run_reduction.)
    with np.errstate(over='ignore'):
        return [
            _b(lambda: np.nansum(a), n, iterations, "np.nansum(a)", "Reduction", dtype_name),
            _b(lambda: np.nanmean(a), n, iterations, "np.nanmean(a)", "Reduction", dtype_name),
            _b(lambda: np.nanmax(a), n, iterations, "np.nanmax(a)", "Reduction", dtype_name),
            _b(lambda: np.nanmin(a), n, iterations, "np.nanmin(a)", "Reduction", dtype_name),
            _b(lambda: np.nanstd(a), n, iterations, "np.nanstd(a)", "Reduction", dtype_name),
            _b(lambda: np.nanvar(a), n, iterations, "np.nanvar(a)", "Reduction", dtype_name),
            _b(lambda: np.nanprod(a_prod), n, iterations, "np.nanprod(a)", "Reduction", dtype_name),
            _b(lambda: np.nanmedian(a), n, iterations, "np.nanmedian(a)", "Reduction", dtype_name),
            _b(lambda: np.nanpercentile(a, 50), n, iterations, "np.nanpercentile(a, 50)", "Reduction", dtype_name),
            _b(lambda: np.nanquantile(a, 0.5), n, iterations, "np.nanquantile(a, 0.5)", "Reduction", dtype_name),
            _b(lambda: np.nanargmax(a), n, iterations, "np.nanargmax(a)", "Reduction", dtype_name),
            _b(lambda: np.nanargmin(a), n, iterations, "np.nanargmin(a)", "Reduction", dtype_name),
        ]


def run_statistics_benchmarks(n, dtype_name, iterations):
    a = create_random_array(n, dtype_name, seed=42)
    rows = [
        _b(lambda: np.median(a), n, iterations, "np.median(a)", "Statistics", dtype_name),
        _b(lambda: np.percentile(a, 50), n, iterations, "np.percentile(a, 50)", "Statistics", dtype_name),
        _b(lambda: np.quantile(a, 0.5), n, iterations, "np.quantile(a, 0.5)", "Statistics", dtype_name),
        _b(lambda: np.average(a), n, iterations, "np.average(a)", "Statistics", dtype_name),
        _b(lambda: np.ptp(a), n, iterations, "np.ptp(a)", "Statistics", dtype_name),
        _b(lambda: np.count_nonzero(a), n, iterations, "np.count_nonzero(a)", "Statistics", dtype_name),
    ]
    if dtype_name in {"int32", "float64"} and n <= ARRAY_SIZES["large"]:
        work_n = memory_heavy_work_n(n)
        signal_a = a[:work_n]
        b = create_random_array(work_n, dtype_name, seed=43)
        kernel = create_random_array(31, dtype_name)
        vector_rows = work_n // 3
        vectors = create_random_array(vector_rows * 3, dtype_name).reshape(vector_rows, 3)
        other_vectors = create_random_array(vector_rows * 3, dtype_name, seed=43).reshape(vector_rows, 3)
        side = int(work_n ** 0.5)
        kron_a = create_random_array(side, dtype_name)
        kron_b = create_random_array(side, dtype_name, seed=43)
        matrix = create_random_array(side * side, dtype_name).reshape(side, side)
        bins = np.linspace(-50, 50, 257).astype(DTYPES[dtype_name])
        values = np.random.randint(0, max(2, work_n // 10), work_n, dtype=np.int32)
        rows.extend([
            _b(lambda: np.convolve(signal_a, kernel, mode="same"), n, iterations, "np.convolve(a, kernel)", "Statistics", dtype_name, "Signal"),
            _b(lambda: np.correlate(signal_a, kernel, mode="same"), n, iterations, "np.correlate(a, kernel)", "Statistics", dtype_name, "Signal"),
            _b(lambda: np.diff(signal_a), n, iterations, "np.diff(a)", "Statistics", dtype_name, "Signal"),
            _b(lambda: np.ediff1d(signal_a), n, iterations, "np.ediff1d(a)", "Statistics", dtype_name, "Signal"),
            _b(lambda: np.cross(vectors, other_vectors), n, iterations, "np.cross(a, b)", "Statistics", dtype_name, "Signal"),
            _b(lambda: np.inner(signal_a, b), n, iterations, "np.inner(a, b)", "Statistics", dtype_name, "Signal"),
            _b(lambda: np.kron(kron_a, kron_b), n, iterations, "np.kron(a, b)", "Statistics", dtype_name, "Signal"),
            _b(lambda: np.trace(matrix), n, iterations, "np.trace(a)", "Statistics", dtype_name, "Signal"),
            _b(lambda: np.cov(signal_a, b), n, iterations, "np.cov(a, b)", "Statistics", dtype_name, "Signal"),
            _b(lambda: np.corrcoef(signal_a, b), n, iterations, "np.corrcoef(a, b)", "Statistics", dtype_name, "Signal"),
            _b(lambda: np.digitize(signal_a, bins), n, iterations, "np.digitize(a, bins)", "Statistics", dtype_name, "Signal"),
        ])
        if dtype_name == "int32":
            rows.append(_b(lambda: np.bincount(values), n, iterations, "np.bincount(a)",
                           "Statistics", dtype_name, "Histogram"))
        else:
            float_values = values.astype(np.float64)
            rows.append(_rejection_evidence(
                lambda: np.bincount(float_values), n, "np.bincount(a)", "Statistics", dtype_name))
    return rows


def run_sorting_benchmarks(n, dtype_name, iterations):
    a = create_random_array(n, dtype_name, seed=42)
    srt = np.arange(n, dtype=DTYPES[dtype_name])
    rows = [
        _b(lambda: np.argsort(a), n, iterations, "np.argsort(a)", "Sorting", dtype_name),
        _b(lambda: np.nonzero(a), n, iterations, "np.nonzero(a)", "Sorting", dtype_name),
        # Query N points (a) into the sorted target → N binary searches (real work that
        # scales with N), matching the C# benchmark. A single scalar lookup is pure call
        # overhead, not a throughput comparison.
        _b(lambda: np.searchsorted(srt, a), n, iterations, "np.searchsorted(a, v)", "Sorting", dtype_name),
    ]
    if n <= ARRAY_SIZES["large"]:
        work_n = memory_heavy_work_n(n)
        extended_a = a[:work_n]
        b = create_random_array(work_n, dtype_name, seed=43)
        rows.extend([
            _b(lambda: np.sort(extended_a), n, iterations, "np.sort(a)", "Sorting", dtype_name),
            _b(lambda: np.sort_complex(extended_a), n, iterations, "np.sort_complex(a)", "Sorting", dtype_name),
            _b(lambda: np.partition(extended_a, work_n // 2), n, iterations, "np.partition(a, kth)", "Sorting", dtype_name),
            _b(lambda: np.argpartition(extended_a, work_n // 2), n, iterations, "np.argpartition(a, kth)", "Sorting", dtype_name),
            _b(lambda: np.lexsort((b, extended_a)), n, iterations, "np.lexsort((b, a))", "Sorting", dtype_name),
            _b(lambda: np.argwhere(extended_a), n, iterations, "np.argwhere(a)", "Sorting", dtype_name),
            _b(lambda: np.flatnonzero(extended_a), n, iterations, "np.flatnonzero(a)", "Sorting", dtype_name),
        ])
    return rows


def run_linalg_benchmarks(n, dtype_name, iterations):
    np.random.seed(42)
    m = int(n ** 0.5)
    mc = min(m, 384)
    v = create_random_array(n, dtype_name)
    vM = create_random_array(m, dtype_name)
    matA = create_random_array(mc * mc, dtype_name).reshape(mc, mc)
    matB = create_random_array(mc * mc, dtype_name, seed=43).reshape(mc, mc)
    v_mc = create_random_array(mc, dtype_name)
    rows = [
        _b(lambda: np.dot(v, v), n, iterations, "np.dot(a, b)", "LinearAlgebra", dtype_name),
        _b(lambda: np.outer(vM, vM), n, iterations, "np.outer(a, b)", "LinearAlgebra", dtype_name),
        _b(lambda: np.matmul(matA, matB), n, iterations, "np.matmul(A, B)", "LinearAlgebra", dtype_name),
        _b(lambda: np.diagonal(matA), n, iterations, "np.diagonal(a)", "LinearAlgebra", dtype_name),
        _b(lambda: np.einsum("i,i->", v, v), n, iterations, "np.einsum(subscripts, operands)", "LinearAlgebra", dtype_name),
        _b(lambda: np.einsum_path("ij,jk->ik", matA, matB), n, iterations, "np.einsum_path(subscripts, operands)", "LinearAlgebra", dtype_name),
        _b(lambda: np.matvec(matA, v_mc), n, iterations, "np.matvec(a, b)", "LinearAlgebra", dtype_name),
        _b(lambda: np.tensordot(v, v, axes=1), n, iterations, "np.tensordot(a, b)", "LinearAlgebra", dtype_name),
        _b(lambda: np.vdot(v, v), n, iterations, "np.vdot(a, b)", "LinearAlgebra", dtype_name),
        _b(lambda: np.vecdot(v, v), n, iterations, "np.vecdot(a, b)", "LinearAlgebra", dtype_name),
        _b(lambda: np.vecmat(v_mc, matA), n, iterations, "np.vecmat(a, b)", "LinearAlgebra", dtype_name),
    ]
    vector_rows = n // 3
    short_vector = create_random_array(int(n ** 0.5), dtype_name)
    vectors_a = create_random_array(vector_rows * 3, dtype_name).reshape(vector_rows, 3)
    vectors_b = create_random_array(vector_rows * 3, dtype_name, seed=43).reshape(vector_rows, 3)
    side = min(int(n ** 0.5), 128)
    matrix_a = create_random_array(side * side, dtype_name).reshape(side, side)
    matrix_b = create_random_array(side * side, dtype_name, seed=43).reshape(side, side)
    rows.extend([
        _b(lambda: np.linalg.cross(vectors_a, vectors_b), n, iterations, "np.linalg.cross(a, b)", "LinearAlgebra", dtype_name, "LinalgApi"),
        _b(lambda: np.linalg.diagonal(matrix_a), n, iterations, "np.linalg.diagonal(a)", "LinearAlgebra", dtype_name, "LinalgApi"),
        # linalg.matmul IS np.matmul (a pure Array-API alias), so measure it on the SAME operands
        # as np.matmul above (matA/matB, side = min(sqrt(n), 384)). Using the 128-capped matrix_a
        # here made it a different shape at the same N, which then collided on the merge key with
        # the backend-profile bench's 316x316 linalg.matmul and produced an impossible cross-paired
        # ratio (single-thread OpenBLAS time / a 128x128 NumPy time).
        _b(lambda: np.linalg.matmul(matA, matB), n, iterations, "np.linalg.matmul(a, b)", "LinearAlgebra", dtype_name, "LinalgApi"),
        _b(lambda: np.linalg.matrix_norm(matrix_a), n, iterations, "np.linalg.matrix_norm(a)", "LinearAlgebra", dtype_name, "LinalgApi"),
        _b(lambda: np.linalg.matrix_power(matrix_a, 3), n, iterations, "np.linalg.matrix_power(a, 3)", "LinearAlgebra", dtype_name, "LinalgApi"),
        _b(lambda: np.linalg.matrix_transpose(matrix_a), n, iterations, "np.linalg.matrix_transpose(a)", "LinearAlgebra", dtype_name, "LinalgApi"),
        _b(lambda: np.linalg.multi_dot((matrix_a, matrix_b, matrix_a)), n, iterations, "np.linalg.multi_dot(arrays)", "LinearAlgebra", dtype_name, "LinalgApi"),
        _b(lambda: np.linalg.norm(v), n, iterations, "np.linalg.norm(a)", "LinearAlgebra", dtype_name, "LinalgApi"),
        _b(lambda: np.linalg.outer(short_vector, short_vector), n, iterations, "np.linalg.outer(a, b)", "LinearAlgebra", dtype_name, "LinalgApi"),
        _b(lambda: np.linalg.tensordot(matrix_a, matrix_b, axes=1), n, iterations, "np.linalg.tensordot(a, b)", "LinearAlgebra", dtype_name, "LinalgApi"),
        _b(lambda: np.linalg.trace(matrix_a), n, iterations, "np.linalg.trace(a)", "LinearAlgebra", dtype_name, "LinalgApi"),
        _b(lambda: np.linalg.vecdot(v, v), n, iterations, "np.linalg.vecdot(a, b)", "LinearAlgebra", dtype_name, "LinalgApi"),
        _b(lambda: np.linalg.vector_norm(v), n, iterations, "np.linalg.vector_norm(a)", "LinearAlgebra", dtype_name, "LinalgApi"),
    ])
    return rows


def run_fft_benchmarks(n, dtype_name, iterations):
    if n > ARRAY_SIZES["large"]:
        return []
    work_n = memory_heavy_work_n(n)
    np.random.seed(42)
    vector = create_random_array(work_n, dtype_name)
    spectrum = np.fft.fft(vector)
    side = int(work_n ** 0.5)
    matrix = create_random_array(side * side, dtype_name).reshape(side, side)
    matrix_spectrum = np.fft.fft2(matrix)
    rows = [
        _b(lambda: np.fft.fft(vector), n, iterations, "np.fft.fft(a)", "Fourier", dtype_name, "Complex"),
        _b(lambda: np.fft.ifft(spectrum), n, iterations, "np.fft.ifft(a)", "Fourier", dtype_name, "Complex"),
        _b(lambda: np.fft.fft2(matrix), n, iterations, "np.fft.fft2(a)", "Fourier", dtype_name, "Complex"),
        _b(lambda: np.fft.ifft2(matrix_spectrum), n, iterations, "np.fft.ifft2(a)", "Fourier", dtype_name, "Complex"),
        _b(lambda: np.fft.fftn(matrix), n, iterations, "np.fft.fftn(a)", "Fourier", dtype_name, "Complex"),
        _b(lambda: np.fft.ifftn(matrix_spectrum), n, iterations, "np.fft.ifftn(a)", "Fourier", dtype_name, "Complex"),
        _b(lambda: np.fft.fftshift(vector), n, iterations, "np.fft.fftshift(a)", "Fourier", dtype_name, "Helpers"),
        _b(lambda: np.fft.ifftshift(vector), n, iterations, "np.fft.ifftshift(a)", "Fourier", dtype_name, "Helpers"),
    ]
    if dtype_name != "complex128":
        real_spectrum = np.fft.rfft(vector)
        real_matrix_spectrum = np.fft.rfft2(matrix)
        rows.extend([
            _b(lambda: np.fft.fftfreq(work_n), n, iterations, "np.fft.fftfreq(n)", "Fourier", dtype_name, "Helpers"),
            _b(lambda: np.fft.rfftfreq(work_n), n, iterations, "np.fft.rfftfreq(n)", "Fourier", dtype_name, "Helpers"),
            _b(lambda: np.fft.rfft(vector), n, iterations, "np.fft.rfft(a)", "Fourier", dtype_name, "Real"),
            _b(lambda: np.fft.irfft(real_spectrum, n=work_n), n, iterations, "np.fft.irfft(a)", "Fourier", dtype_name, "Real"),
            _b(lambda: np.fft.hfft(real_spectrum, n=work_n), n, iterations, "np.fft.hfft(a)", "Fourier", dtype_name, "Real"),
            _b(lambda: np.fft.ihfft(vector), n, iterations, "np.fft.ihfft(a)", "Fourier", dtype_name, "Real"),
            _b(lambda: np.fft.rfft2(matrix), n, iterations, "np.fft.rfft2(a)", "Fourier", dtype_name, "Real"),
            _b(lambda: np.fft.irfft2(real_matrix_spectrum, s=(side, side)), n, iterations, "np.fft.irfft2(a)", "Fourier", dtype_name, "Real"),
            _b(lambda: np.fft.rfftn(matrix), n, iterations, "np.fft.rfftn(a)", "Fourier", dtype_name, "Real"),
            _b(lambda: np.fft.irfftn(real_matrix_spectrum, s=(side, side), axes=(0, 1)), n, iterations, "np.fft.irfftn(a)", "Fourier", dtype_name, "Real"),
        ])
    return rows


def run_random_benchmarks(n, dtype_name, iterations):
    if n > ARRAY_SIZES["large"]:
        return []
    work_n = memory_heavy_work_n(n)
    np.random.seed(42)
    size = work_n
    source = np.arange(work_n, dtype=DTYPES[dtype_name])
    shuffle_target = source.copy()
    alpha = np.array([1.0, 2.0, 3.0])
    probabilities = np.array([0.2, 0.3, 0.5])
    mean = np.array([0.0, 1.0, -1.0])
    covariance = np.array([[1.0, 0.2, 0.1], [0.2, 1.5, 0.3], [0.1, 0.3, 2.0]])
    continuous = [
        ("np.random.beta(a, b)", lambda: np.random.beta(2.0, 5.0, size)),
        ("np.random.chisquare(df)", lambda: np.random.chisquare(4.0, size)),
        ("np.random.exponential(scale)", lambda: np.random.exponential(2.0, size)),
        ("np.random.f(dfnum, dfden)", lambda: np.random.f(5.0, 7.0, size)),
        ("np.random.gamma(shape, scale)", lambda: np.random.gamma(2.0, 3.0, size)),
        ("np.random.gumbel(loc, scale)", lambda: np.random.gumbel(0.0, 1.0, size)),
        ("np.random.laplace(loc, scale)", lambda: np.random.laplace(0.0, 1.0, size)),
        ("np.random.logistic(loc, scale)", lambda: np.random.logistic(0.0, 1.0, size)),
        ("np.random.lognormal(mean, sigma)", lambda: np.random.lognormal(0.0, 1.0, size)),
        ("np.random.noncentral_chisquare(df, nonc)", lambda: np.random.noncentral_chisquare(4.0, 1.5, size)),
        ("np.random.noncentral_f(dfnum, dfden, nonc)", lambda: np.random.noncentral_f(5.0, 7.0, 1.5, size)),
        ("np.random.normal(loc, scale)", lambda: np.random.normal(0.0, 1.0, size)),
        ("np.random.pareto(a)", lambda: np.random.pareto(3.0, size)),
        ("np.random.power(a)", lambda: np.random.power(3.0, size)),
        ("np.random.rand(n)", lambda: np.random.rand(work_n)),
        ("np.random.randn(n)", lambda: np.random.randn(work_n)),
        ("np.random.random(n)", lambda: np.random.random(work_n)),
        ("np.random.random_sample(n)", lambda: np.random.random_sample(work_n)),
        ("np.random.rayleigh(scale)", lambda: np.random.rayleigh(2.0, size)),
        ("np.random.standard_cauchy(n)", lambda: np.random.standard_cauchy(size)),
        ("np.random.standard_exponential(n)", lambda: np.random.standard_exponential(size)),
        ("np.random.standard_gamma(shape)", lambda: np.random.standard_gamma(2.0, size)),
        ("np.random.standard_normal(n)", lambda: np.random.standard_normal(size)),
        ("np.random.standard_t(df)", lambda: np.random.standard_t(5.0, size)),
        ("np.random.triangular(left, mode, right)", lambda: np.random.triangular(-1.0, 0.0, 2.0, size)),
        ("np.random.uniform(low, high)", lambda: np.random.uniform(-1.0, 2.0, size)),
        ("np.random.vonmises(mu, kappa)", lambda: np.random.vonmises(0.0, 2.0, size)),
        ("np.random.wald(mean, scale)", lambda: np.random.wald(2.0, 1.0, size)),
        ("np.random.weibull(a)", lambda: np.random.weibull(2.0, size)),
    ]
    discrete = [
        ("np.random.binomial(n, p)", lambda: np.random.binomial(10, 0.4, size)),
        ("np.random.geometric(p)", lambda: np.random.geometric(0.4, size)),
        ("np.random.hypergeometric(ngood, nbad, nsample)", lambda: np.random.hypergeometric(20, 30, 10, size)),
        ("np.random.logseries(p)", lambda: np.random.logseries(0.6, size)),
        ("np.random.negative_binomial(n, p)", lambda: np.random.negative_binomial(5.0, 0.4, size)),
        ("np.random.poisson(lam)", lambda: np.random.poisson(3.0, size)),
        ("np.random.randint(low, high)", lambda: np.random.randint(0, 100, size, dtype=np.int64)),
        ("np.random.zipf(a)", lambda: np.random.zipf(2.0, size)),
        ("np.random.multinomial(n, pvals, size)", lambda: np.random.multinomial(10, probabilities, size)),
    ]
    structured = [
        ("np.random.choice(a, size)", lambda: np.random.choice(source, size)),
        ("np.random.dirichlet(alpha, size)", lambda: np.random.dirichlet(alpha, size)),
        ("np.random.multivariate_normal(mean, cov, size)", lambda: np.random.multivariate_normal(mean, covariance, size)),
        ("np.random.permutation(a)", lambda: np.random.permutation(source)),
        ("np.random.shuffle(a)", lambda: np.random.shuffle(shuffle_target)),
    ]
    # Continuous/discrete generators have fixed output dtypes. Their scenario dtype identifies
    # the requested official matrix cell, matching the C# parameter convention; structured cases
    # additionally consume a source array of that dtype.
    if dtype_name == "int64":
        return [_b(func, n, iterations, name, "Random", dtype_name, "Discrete")
                for name, func in discrete]
    rows = [_b(func, n, iterations, name, "Random", dtype_name, "Continuous")
            for name, func in continuous]
    rows.extend(_b(func, n, iterations, name, "Random", dtype_name, "DiscreteFixedOutput")
                for name, func in discrete)
    rows.extend(_b(func, n, iterations, name, "Random", dtype_name, "Structured")
                for name, func in structured)
    return rows


def run_ndarray_benchmarks(n, dtype_name, iterations):
    if n > ARRAY_SIZES["large"]:
        return []
    import os
    import tempfile

    work_n = memory_heavy_work_n(n)
    np.random.seed(42)
    a = create_random_array(work_n, dtype_name)
    b = create_random_array(work_n, dtype_name, seed=43)
    product_input = (np.random.random(work_n) * 0.5 + 0.5).astype(DTYPES[dtype_name])
    matrix = a.reshape(10, work_n // 10)
    with_singletons = a.reshape(1, work_n, 1)
    mask = a > 0.0
    indices = np.arange(0, work_n, 2)
    selector = np.random.randint(0, 2, work_n)
    sorted_a = np.arange(work_n, dtype=DTYPES[dtype_name])
    put_target = a.copy()
    fill_target = a.copy()
    field_target = a.copy()
    file_handle, file_path = tempfile.mkstemp(prefix="numsharp-ndarray-bench-", suffix=".bin")
    os.close(file_handle)

    def resize_copy():
        copy = a.copy()
        copy.resize(work_n + 1, refcheck=False)
        return copy

    def partition_copy():
        copy = a.copy()
        copy.partition(work_n // 2)
        return copy

    def sort_copy():
        copy = a.copy()
        copy.sort()
        return copy

    cases = [
        ("a.all() [ndarray]", lambda: a.all()),
        ("a.any() [ndarray]", lambda: a.any()),
        ("a.argmax() [ndarray]", lambda: a.argmax()),
        ("a.argmin() [ndarray]", lambda: a.argmin()),
        ("a.cumprod() [ndarray]", lambda: product_input.cumprod()),
        ("a.cumsum() [ndarray]", lambda: a.cumsum()),
        ("a.max() [ndarray]", lambda: a.max()),
        ("a.min() [ndarray]", lambda: a.min()),
        ("a.prod() [ndarray]", lambda: product_input.prod()),
        ("a.choose(choices) [ndarray]", lambda: selector.choose((a, b))),
        ("a.clip(min, max) [ndarray]", lambda: a.clip(DTYPES[dtype_name](-10), DTYPES[dtype_name](10))),
        ("a.compress(mask) [ndarray]", lambda: a.compress(mask)),
        ("a.conj() [ndarray]", lambda: a.conj()),
        ("a.conjugate() [ndarray]", lambda: a.conjugate()),
        ("a.diagonal() [ndarray]", lambda: matrix.diagonal()),
        ("a.dot(b) [ndarray]", lambda: a.dot(b)),
        ("a.fill(value) [ndarray]", lambda: fill_target.fill(DTYPES[dtype_name](3))),
        ("a.getfield(dtype) [ndarray]", lambda: a.getfield(DTYPES[dtype_name])),
        ("a.item(index) [ndarray]", lambda: a.item(work_n // 2)),
        ("a.put(indices, values) [ndarray]", lambda: put_target.put(indices, b[::2])),
        ("a.repeat(repeats) [ndarray]", lambda: a.repeat(2)),
        ("a.reshape(shape) [ndarray]", lambda: a.reshape(10, work_n // 10)),
        ("a.resize(shape) [ndarray]", resize_copy),
        ("a.round() [ndarray]", lambda: a.round()),
        ("a.setfield(value, dtype) [ndarray]", lambda: field_target.setfield(DTYPES[dtype_name](1), DTYPES[dtype_name])),
        ("a.setflags(write=True) [ndarray]", lambda: a.setflags(write=True)),
        ("a.squeeze() [ndarray]", lambda: with_singletons.squeeze()),
        ("a.swapaxes(0, 1) [ndarray]", lambda: matrix.swapaxes(0, 1)),
        ("a.take(indices) [ndarray]", lambda: a.take(indices)),
        ("a.to_device(cpu) [ndarray]", lambda: a.to_device("cpu")),
        ("a.tobytes() [ndarray]", lambda: a.tobytes()),
        ("a.tofile(path) [ndarray]", lambda: a.tofile(file_path)),
        ("a.tolist() [ndarray]", lambda: a.tolist()),
        ("a.trace() [ndarray]", lambda: matrix.trace()),
        ("a.transpose() [ndarray]", lambda: matrix.transpose()),
        ("a.view() [ndarray]", lambda: a.view()),
        ("a.argpartition(kth) [ndarray]", lambda: a.argpartition(work_n // 2)),
        ("a.argsort() [ndarray]", lambda: a.argsort()),
        ("a.nonzero() [ndarray]", lambda: a.nonzero()),
        ("a.partition(kth) [ndarray]", partition_copy),
        ("a.searchsorted(v) [ndarray]", lambda: sorted_a.searchsorted(a)),
        ("a.sort() [ndarray]", sort_copy),
    ]
    try:
        return [_b(func, n, iterations, name, "NDArray", dtype_name, "Methods") for name, func in cases]
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def run_api_surface_benchmarks(n, dtype_name, iterations):
    # N=1 (Scalar) is the pure dispatch/call-overhead tier for the scalar/dtype/text ops —
    # "scalar x scalar"; the real-work poly/IO ops are emitted only at the standard size.
    if n not in (1, ARRAY_SIZES["small"]):
        return []
    import os
    import tempfile

    dtype = DTYPES[dtype_name]
    a = np.arange(n, dtype=dtype)
    b_float = np.arange(n, dtype=np.float64)
    mrows = 10 if n >= 10 else 1          # keep reshape valid at the N=1 tier
    matrix = a.reshape(mrows, n // mrows)
    dtype_cases = [
        ("np.can_cast(from, to)", lambda: np.can_cast(dtype, np.float64)),
        ("np.common_type(a, b)", lambda: np.common_type(a, b_float)),
        ("np.isdtype(dtype, kind)", lambda: np.isdtype(dtype, "real floating")),
        ("np.issubdtype(dtype, kind)", lambda: np.issubdtype(dtype, np.floating)),
        ("np.min_scalar_type(value)", lambda: np.min_scalar_type(dtype(42))),
        ("np.mintypecode(typechars)", lambda: np.mintypecode("fd")),
        ("np.promote_types(a, b)", lambda: np.promote_types(dtype, np.float64)),
        ("np.result_type(a, b)", lambda: np.result_type(a, b_float)),
        ("np.iterable(a)", lambda: np.iterable(a)),
        ("np.copyto(dst, src)", lambda: np.copyto(b_float, a, casting="unsafe")),
        ("np.nested_iters(a, axes)", lambda: np.nested_iters(matrix, [[0], [1]])),
    ]

    text_a = np.linspace(-100, 100, n).astype(dtype)
    def printoptions_scope():
        with np.printoptions(precision=6):
            pass
    text_cases = [
        ("np.array_repr(a)", lambda: np.array_repr(text_a)),
        ("np.array_str(a)", lambda: np.array_str(text_a)),
        ("np.array2string(a)", lambda: np.array2string(text_a)),
        ("np.format_float_positional(x)", lambda: np.format_float_positional(np.pi, precision=12)),
        ("np.format_float_scientific(x)", lambda: np.format_float_scientific(np.pi, precision=12)),
        ("np.get_printoptions()", lambda: np.get_printoptions()),
        ("np.printoptions()", printoptions_scope),
        ("np.set_printoptions()", lambda: np.set_printoptions(precision=6)),
    ]

    np.random.seed(42)
    p = create_random_array(17, dtype_name)
    q = create_random_array(9, dtype_name, seed=43)
    roots_input = np.linspace(-2, 2, 16).astype(dtype)
    x = np.linspace(-1, 1, n).astype(dtype)
    polynomial_cases = [
        ("np.poly(roots)", lambda: np.poly(roots_input)),
        ("np.polyadd(p, q)", lambda: np.polyadd(p, q)),
        ("np.polyder(p)", lambda: np.polyder(p)),
        ("np.polydiv(p, q)", lambda: np.polydiv(p, q)),
        ("np.polyint(p)", lambda: np.polyint(p)),
        ("np.polymul(p, q)", lambda: np.polymul(p, q)),
        ("np.polysub(p, q)", lambda: np.polysub(p, q)),
        ("np.polyval(p, x)", lambda: np.polyval(p, x)),
    ]

    rows = [_b(func, n, iterations, name, "ApiSurface", dtype_name, "DType") for name, func in dtype_cases]
    rows.extend(_b(func, n, iterations, name, "ApiSurface", dtype_name, "Text") for name, func in text_cases)

    # The real-work families (Polynomial, IO) only make sense at the standard size — at the
    # N=1 dispatch tier they would produce no-op cells with no matching C# benchmark.
    if n != ARRAY_SIZES["small"]:
        return rows

    rows.extend(_b(func, n, iterations, name, "ApiSurface", dtype_name, "Polynomial") for name, func in polynomial_cases)

    with tempfile.TemporaryDirectory(prefix="numsharp-api-benchmark-") as directory:
        io_a = np.linspace(-100, 100, n).astype(dtype)
        npy = os.path.join(directory, "input.npy")
        raw = os.path.join(directory, "input.bin")
        text = os.path.join(directory, "input.txt")
        save = os.path.join(directory, "save.npy")
        save_text = os.path.join(directory, "save.txt")
        save_zip = os.path.join(directory, "save.npz")
        save_zip_compressed = os.path.join(directory, "save-compressed.npz")
        np.save(npy, io_a)
        io_a.tofile(raw)
        np.savetxt(text, io_a, fmt="%d" if dtype_name == "int32" else "%.18e")
        io_cases = [
            ("np.fromfile(path)", lambda: np.fromfile(raw, dtype=dtype)),
            ("np.loadtxt(path)", lambda: np.loadtxt(text, dtype=dtype)),
            ("np.load(path)", lambda: np.load(npy)),
            ("np.save(path, a)", lambda: np.save(save, io_a)),
            ("np.savetxt(path, a)", lambda: np.savetxt(save_text, io_a)),
            ("np.savez(path, a)", lambda: np.savez(save_zip, io_a)),
            ("np.savez_compressed(path, a)", lambda: np.savez_compressed(save_zip_compressed, io_a)),
        ]
        rows.extend(_b(func, n, iterations, name, "ApiSurface", dtype_name, "IO") for name, func in io_cases)
    return rows


def run_where_benchmarks(n, dtype_name, iterations):
    np.random.seed(42)
    a = create_random_array(n, dtype_name)
    b = create_random_array(n, dtype_name, seed=43)
    cond = a > 0
    negative = a < 0
    rows = [
        _b(lambda: np.where(cond, a, b), n, iterations, "np.where(cond, a, b)", "Selection", dtype_name),
        _b(lambda: np.where(cond), n, iterations, "np.where(cond)", "Selection", dtype_name),
    ]
    if n <= ARRAY_SIZES["large"]:
        work_n = memory_heavy_work_n(n)
        select_a = a[:work_n]
        select_b = b[:work_n]
        select_cond = cond[:work_n]
        select_negative = negative[:work_n]
        indices = np.arange(0, work_n, 2)
        selector = np.random.randint(0, 2, work_n)
        side = int(work_n ** 0.5)
        matrix = np.arange(side * side).reshape(side, side)
        put_target = select_a.copy()
        place_target = select_a.copy()
        place_values = np.array([1, 2, 3], dtype=DTYPES[dtype_name])
        rows.extend([
            _b(lambda: np.choose(selector, (select_a, select_b)), n, iterations, "np.choose(selector, choices)", "Selection", dtype_name),
            _b(lambda: np.compress(select_cond, select_a), n, iterations, "np.compress(mask, a)", "Selection", dtype_name),
            _b(lambda: np.extract(select_cond, select_a), n, iterations, "np.extract(mask, a)", "Selection", dtype_name),
            _b(lambda: np.take(select_a, indices), n, iterations, "np.take(a, indices)", "Selection", dtype_name),
            _b(lambda: np.take_along_axis(select_a, indices, 0), n, iterations, "np.take_along_axis(a, indices)", "Selection", dtype_name),
            _b(lambda: np.select((select_negative, select_cond), (select_a, select_b), DTYPES[dtype_name](0)), n, iterations, "np.select(condlist, choicelist)", "Selection", dtype_name),
            _b(lambda: np.put(put_target, indices, select_b[::2]), n, iterations, "np.put(a, indices, values)", "Selection", dtype_name),
            _b(lambda: np.place(place_target, select_cond, place_values), n, iterations, "np.place(a, mask, values)", "Selection", dtype_name),
            _b(lambda: np.diag_indices(side), n, iterations, "np.diag_indices(n)", "Selection", dtype_name),
            _b(lambda: np.diag_indices_from(matrix), n, iterations, "np.diag_indices_from(a)", "Selection", dtype_name),
            _b(lambda: np.tril_indices_from(matrix), n, iterations, "np.tril_indices_from(a)", "Selection", dtype_name),
            _b(lambda: np.triu_indices_from(matrix), n, iterations, "np.triu_indices_from(a)", "Selection", dtype_name),
            _b(lambda: np.mask_indices(side, np.triu), n, iterations, "np.mask_indices(n, triu)", "Selection", dtype_name),
            _b(lambda: np.indices((side, side)), n, iterations, "np.indices(shape)", "Selection", dtype_name),
            _b(lambda: np.ravel_multi_index((indices,), (work_n,)), n, iterations, "np.ravel_multi_index(coords, dims)", "Selection", dtype_name),
            _b(lambda: np.unravel_index(indices, (work_n,)), n, iterations, "np.unravel_index(indices, shape)", "Selection", dtype_name),
        ])
    return rows


def run_cumulative_benchmarks(n, dtype_name, iterations):
    """Cumulative product (mirror C# CumulativeBenchmarks). Values in [0.5, 1.0] keep the running
    product finite at every size (the C# class uses the same range), so it stays a real workload
    instead of collapsing to inf almost immediately — full-range data blows up (and the C# Decimal
    cell, which has no inf, threw OverflowException)."""
    np.random.seed(42)
    a = (np.random.rand(n) * 0.5 + 0.5).astype(DTYPES[dtype_name])
    return [
        _b(lambda: np.cumprod(a), n, iterations, "np.cumprod(a)", "Reduction", dtype_name),
    ]


def run_prod_benchmarks(n, dtype_name, iterations):
    """Product reduction (mirror C# ProdBenchmarks): full + axis. Values in [0.5, 1.0] keep the
    product finite at every size (the C# class uses the same range), so this is overflow-safe
    even at 10M, unlike a full-range random array."""
    np.random.seed(42)
    a = (np.random.rand(n) * 0.5 + 0.5).astype(DTYPES[dtype_name])
    rows = int(np.sqrt(n)); cols = n // rows
    np.random.seed(42)
    a_2d = (np.random.rand(rows * cols) * 0.5 + 0.5).astype(DTYPES[dtype_name]).reshape(rows, cols)
    return [
        _b(lambda: np.prod(a), n, iterations, "np.prod", "Reduction", dtype_name),
        _b(lambda: np.prod(a_2d, axis=0), n, iterations, "np.prod axis=0", "Reduction", dtype_name),
        _b(lambda: np.prod(a_2d, axis=1), n, iterations, "np.prod axis=1", "Reduction", dtype_name),
    ]


def run_suites(n: int, suite: str, dtypes_to_run: List[str], iterations: int) -> List[BenchmarkResult]:
    """Run all selected suites at a single array size N and return the results.

    Extracted from main() so the official run can sweep multiple sizes in one invocation
    (each result carries its own n, which the merge keys on)."""
    results_all: List[BenchmarkResult] = []

    if suite in ["dispatch", "all"]:
        results_all.extend(run_dispatch_benchmarks(n, iterations))

    if suite in ["fusion", "all"]:
        results_all.extend(run_fusion_benchmarks(n, iterations))

    if suite in ["arithmetic", "all"]:
        print(f"\n{'='*60}\n  Arithmetic Benchmarks (N={n:,})\n{'='*60}")
        for dtype in dtypes_to_run:
            if dtype in ARITHMETIC_DTYPES:
                print(f"\n  --- {dtype} ---")
                results = run_arithmetic_benchmarks(n, dtype, iterations)
                results_all.extend(results)
                for r in results:
                    print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms")

    if suite in ["unary", "all"]:
        print(f"\n{'='*60}\n  Unary Benchmarks (N={n:,})\n{'='*60}")
        for dtype in dtypes_to_run:
            if dtype in TRANSCENDENTAL_DTYPES:
                print(f"\n  --- {dtype} ---")
                results = run_unary_benchmarks(n, dtype, iterations)
                results_all.extend(results)
                for r in results:
                    print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms")
        # Extra unary math (cbrt/reciprocal/square/negative/positive/trunc) — mirrors
        # the C# UnaryExtraBenchmarks class (also under the Unary namespace).
        for dtype in selected(['int32'] + FLOAT_DTYPES, dtypes_to_run):
            results_all.extend(run_unary_extra_benchmarks(n, dtype, iterations))

    if suite in ["reduction", "all"]:
        print(f"\n{'='*60}\n  Reduction Benchmarks (N={n:,})\n{'='*60}")
        for dtype in selected(ARITHMETIC_DTYPES, dtypes_to_run):
            print(f"\n  --- {dtype} ---")
            results = run_reduction_benchmarks(n, dtype, iterations)
            results_all.extend(results)
            for r in results:
                print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms")
        # NaN-aware reductions + cumprod — mirror C# NanReductionBenchmarks / CumulativeBenchmarks.
        for dtype in selected(['int32'] + FLOAT_DTYPES, dtypes_to_run):
            results_all.extend(run_nan_reduction_benchmarks(n, dtype, iterations))
            results_all.extend(run_cumulative_benchmarks(n, dtype, iterations))
        # Product reduction — bounded inputs keep the full int32/int64/float64 workloads finite.
        for dtype in selected(['int32', 'int64', 'float64'], dtypes_to_run):
            results_all.extend(run_prod_benchmarks(n, dtype, iterations))

    if suite in ["broadcast", "all"]:
        for dtype in selected(['int32', 'float64'], dtypes_to_run):
            results_all.extend(run_broadcast_benchmarks(n, dtype, iterations))

    if suite in ["creation", "all"]:
        print(f"\n{'='*60}\n  Creation Benchmarks (N={n:,})\n{'='*60}")
        for dtype in selected(COMMON_DTYPES, dtypes_to_run):
            print(f"\n  --- {dtype} ---")
            results = run_creation_benchmarks(n, dtype, iterations)
            results_all.extend(results)
            for r in results:
                print(f"  {r.name:<40} avg {r.mean_ms:>8.3f} ms  min {r.min_ms:>8.3f} ms")

    if suite in ["manipulation", "all"]:
        for dtype in selected(['int32', 'float64'], dtypes_to_run):
            results_all.extend(run_manipulation_benchmarks(n, dtype, iterations))

    if suite in ["slicing", "all"]:
        for dtype in selected(['int32', 'float64'], dtypes_to_run):
            results_all.extend(run_slicing_benchmarks(n, dtype, iterations))

    if suite in ["comparison", "all"]:
        for dtype in selected(COMMON_DTYPES, dtypes_to_run):
            results_all.extend(run_comparison_benchmarks(n, dtype, iterations))

    if suite in ["bitwise", "all"]:
        for dtype in selected(BITWISE_DTYPES + ['float64'], dtypes_to_run):
            results_all.extend(run_bitwise_benchmarks(n, dtype, iterations))

    if suite in ["logic", "all"]:
        for dtype in selected(['int32'] + FLOAT_DTYPES, dtypes_to_run):
            results_all.extend(run_logic_benchmarks(n, dtype, iterations))
        if "bool" in dtypes_to_run:
            results_all.extend(run_bool_logic_benchmarks(n, iterations))

    if suite in ["statistics", "all"]:
        for dtype in selected(['int32'] + FLOAT_DTYPES, dtypes_to_run):
            results_all.extend(run_statistics_benchmarks(n, dtype, iterations))

    if suite in ["sorting", "all"]:
        for dtype in selected(COMMON_DTYPES, dtypes_to_run):
            results_all.extend(run_sorting_benchmarks(n, dtype, iterations))

    if suite in ["linalg", "all"]:
        for dtype in selected(['int32', 'float64'], dtypes_to_run):
            results_all.extend(run_linalg_benchmarks(n, dtype, iterations))

    if suite in ["fft", "all"]:
        for dtype in selected(['int32', 'float64', 'complex128'], dtypes_to_run):
            results_all.extend(run_fft_benchmarks(n, dtype, iterations))

    if suite in ["random", "all"]:
        for dtype in selected(['int32', 'int64', 'float64'], dtypes_to_run):
            results_all.extend(run_random_benchmarks(n, dtype, iterations))

    if suite in ["ndarray", "all"]:
        for dtype in selected(['int32', 'float64'], dtypes_to_run):
            results_all.extend(run_ndarray_benchmarks(n, dtype, iterations))

    if suite in ["api", "all"]:
        for dtype in selected(['int32', 'float64'], dtypes_to_run):
            results_all.extend(run_api_surface_benchmarks(n, dtype, iterations))

    if suite in ["selection", "all"]:
        for dtype in selected(['int32', 'float64'], dtypes_to_run):
            results_all.extend(run_where_benchmarks(n, dtype, iterations))

    return [row for row in results_all if row is not None and row.dtype in ACTIVE_DTYPES]


OFFICIAL_SCENARIO_SUITES = (
    "arithmetic", "unary", "reduction", "broadcast", "creation", "manipulation", "slicing",
    "comparison", "bitwise", "logic", "statistics", "sorting", "linalg", "selection", "fft",
    "random", "ndarray", "api",
)


def discover_scenarios() -> List[Dict[str, Any]]:
    """Discover the Python twin titles/dtypes without executing timed benchmark bodies.

    The real suite builders remain the single source of truth: only the timing primitive is
    replaced, so dtype loops, conditional cases, titles, suites, and categories are exactly what
    an official run would schedule at the standard tier.
    """
    original_benchmark = globals()["benchmark"]

    def discovery_benchmark(_func, n, **_kwargs):
        return BenchmarkResult("", "", "", "", n, 0.0, 0.0, 0.0, 0.0, 1, 0.0)

    rows: List[BenchmarkResult] = []
    globals()["benchmark"] = discovery_benchmark
    try:
        # Suite builders print progress and may trigger benign NumPy warnings while constructing
        # inputs. Discovery stdout must remain pure JSON for the generator.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for suite in OFFICIAL_SCENARIO_SUITES:
                rows.extend(run_suites(
                    ARRAY_SIZES["small"], suite,
                    [dtype for dtype in ALL_DTYPES if dtype in DTYPES], 1))
    finally:
        globals()["benchmark"] = original_benchmark

    grouped: Dict[tuple, set] = {}
    metadata: Dict[tuple, tuple] = {}
    for row in rows:
        suffix = f" ({row.dtype})"
        # Newer helpers append the dtype to the result name; several original suite builders keep
        # the exact C# Description without that reporting suffix. Both forms are official output.
        title = row.name[:-len(suffix)] if row.name.endswith(suffix) else row.name
        key = (title, row.suite, row.category)
        grouped.setdefault(key, set()).add(row.dtype)
        metadata[key] = (title, row.suite, row.category)

    return [
        {"title": title, "suite": suite, "category": category, "dtypes": sorted(grouped[key])}
        for key, (title, suite, category) in sorted(metadata.items())
    ]


def main():
    parser = argparse.ArgumentParser(description="NumPy Performance Benchmarks")
    parser.add_argument("--suite", choices=["dispatch", "fusion", "arithmetic", "unary", "reduction",
                                            "broadcast", "creation", "manipulation", "slicing",
                                            "comparison", "bitwise", "logic", "statistics", "sorting",
                                            "linalg", "selection", "fft", "random", "ndarray", "api", "all"],
                        default="all", help="Benchmark suite to run")
    parser.add_argument("--n", type=int, default=10_000_000, help="Array size")
    parser.add_argument("--size", choices=["small", "medium", "large", "all"], default=None,
                        help="Array size preset ('all' sweeps small+medium+large in one run)")
    parser.add_argument("--cache-sizes", action="store_true",
                        help="Sweep all three cache-tier sizes (small, medium, large) in one invocation")
    parser.add_argument("--with-scalar", action="store_true",
                        help="Also sweep the N=1 (Scalar) dispatch-overhead tier — the 'scalar x scalar' point")
    parser.add_argument("--type", type=str, default=None, help="Specific dtype (e.g., int32, float64)")
    parser.add_argument("--iterations", type=int, default=50, help="Measure-mode iteration baseline")
    parser.add_argument("--quick", action="store_true", help="Deprecated alias for --depth light")
    parser.add_argument("--depth", choices=tuple(DEPTHS), default="measure",
                        help="pass=1 call/0 warmup; light=1/6 budget/max 3 warmups; measure=full")
    parser.add_argument("--dtypes", help="Comma-separated dtype filter")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--output", type=str, default=None, help="Output JSON to file")
    parser.add_argument("--list-scenarios-json", action="store_true",
                        help="Discover official scenario titles/dtypes without running timings")
    args = parser.parse_args()

    if args.list_scenarios_json:
        print(json.dumps({
            "schema_version": 1,
            "source": "NumPy suite-builder dry discovery",
            "numpy_version": np.__version__,
            "rows": discover_scenarios(),
        }, indent=2))
        return

    global ACTIVE_BENCHMARK_DEPTH, ACTIVE_DTYPES
    if args.quick:
        if args.depth != "measure":
            parser.error("--quick cannot be combined with --depth; use --depth light")
        args.depth = "light"
    ACTIVE_BENCHMARK_DEPTH = args.depth

    if args.size and args.size != "all":
        args.n = ARRAY_SIZES[args.size]

    # Determine which dtypes to run
    dtype_text = args.dtypes or args.type
    try:
        requested_dtypes = parse_dtypes(dtype_text)
    except ValueError as error:
        parser.error(str(error))
    dtypes_to_run = [dtype for dtype in requested_dtypes if dtype in DTYPES]
    ACTIVE_DTYPES = set(dtypes_to_run)

    print(f"\nNumPy {np.__version__}")
    print(f"Python {sys.version.split()[0]}")
    print(f"Array size: N = {args.n:,}")
    if args.depth == "pass":
        print("Measurement budget: exactly 1 call; 0 warmups")
    elif args.depth == "light":
        print(f"Measurement budget: 1/6 of the {args.iterations}-iteration baseline; max 3 warmups")
    else:
        print(f"Measurement baseline: {args.iterations} iterations; 10 warmups")
    print(f"Depth: {args.depth}")
    print(f"Types: {dtypes_to_run}")

    # Sizes to sweep: --size all (or --cache-sizes) runs the three cache-tier sizes in one
    # invocation so a single JSON carries all three; otherwise the single resolved args.n.
    if args.cache_sizes or args.size == "all":
        sizes_to_run = [ARRAY_SIZES["small"], ARRAY_SIZES["medium"], ARRAY_SIZES["large"]]
    else:
        sizes_to_run = [args.n]

    # The N=1 (Scalar) tier measures pure dispatch/call overhead — prepended so it runs first.
    if args.with_scalar and 1 not in sizes_to_run:
        sizes_to_run = [1] + sizes_to_run

    print(f"Sizes to run: {[f'{n:,}' for n in sizes_to_run]}")

    all_results = []
    for n in sizes_to_run:
        print(f"\n{'#'*64}\n#  ARRAY SIZE  N = {n:,}\n{'#'*64}")
        all_results.extend(run_suites(n, args.suite, dtypes_to_run, args.iterations))

    # Tier-key invariant: MEMORY_HEAVY_LARGE_WORKLOAD (1M) is a PHYSICAL cap only, never a
    # published tier — a memory-heavy row must keep n=10M so every calc, geomean, and UI size
    # grouping treats it at the 10M level. A suite that accidentally emitted n=work_n would
    # otherwise mint a silent fourth "1M" level (the universal-tier contract in merge-results.py
    # catches the common 1K+100K-without-10M shape of this mistake; this guard catches the rest).
    leaked = sorted({r.name for r in all_results if r.n == MEMORY_HEAVY_LARGE_WORKLOAD})
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} result(s) published n={MEMORY_HEAVY_LARGE_WORKLOAD:,} — the memory-heavy "
            f"workload is a physical cap, not a tier; emit them under n=10,000,000: {leaked[:10]}")

    # Output
    if args.json or args.output:
        json_output = json.dumps([asdict(r) for r in all_results], indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_output)
            print(f"\nJSON results written to: {args.output}")
        if args.json:
            print("\n" + json_output)
    else:
        print_summary(all_results)

    print(f"\n{'='*60}")
    print(f"  Benchmark complete ({len(all_results)} results)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
