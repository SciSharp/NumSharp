#!/usr/bin/env python3
"""
NumPy Performance Benchmarks
============================

Comprehensive benchmarks matching the C# NumSharp.Benchmark.Core suite.
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

import numpy as np
import time
import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional, Dict, Any
import statistics

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

# NumPy dtypes matching NumSharp's 12 supported types
DTYPES = {
    'bool': np.bool_,
    'uint8': np.uint8,
    'int16': np.int16,
    'uint16': np.uint16,
    'int32': np.int32,
    'uint32': np.uint32,
    'int64': np.int64,
    'uint64': np.uint64,
    'float32': np.float32,
    'float64': np.float64,
}

# Common types for quick benchmarks
COMMON_DTYPES = ['int32', 'int64', 'float32', 'float64']

# Arithmetic types (excludes bool)
ARITHMETIC_DTYPES = ['uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float32', 'float64']

# Transcendental types (for sqrt, exp, log, trig)
TRANSCENDENTAL_DTYPES = ['float32', 'float64']

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

def benchmark(func: Callable, n: int, warmup: int = 10, iterations: int = 50) -> BenchmarkResult:
    """Run a benchmark with proper warmup and statistical analysis."""
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

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
        iterations=iterations,
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
    """Benchmark arithmetic operations for a specific dtype."""
    results = []
    dtype = DTYPES[dtype_name]

    np.random.seed(42)
    a = create_random_array(n, dtype_name, seed=42)
    b = create_random_array(n, dtype_name, seed=43)
    b_positive = np.abs(b) + 1  # For division
    scalar = dtype(5)

    # Add
    def add_elementwise(): return a + b
    r = benchmark(add_elementwise, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a + b ({dtype_name})", "Add", "Arithmetic", dtype_name
    results.append(r)

    def add_scalar(): return a + scalar
    r = benchmark(add_scalar, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a + scalar ({dtype_name})", "Add", "Arithmetic", dtype_name
    results.append(r)

    # Subtract
    def sub_elementwise(): return a - b
    r = benchmark(sub_elementwise, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a - b ({dtype_name})", "Subtract", "Arithmetic", dtype_name
    results.append(r)

    # Multiply
    def mul_elementwise(): return a * b
    r = benchmark(mul_elementwise, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a * b ({dtype_name})", "Multiply", "Arithmetic", dtype_name
    results.append(r)

    def mul_square(): return a * a
    r = benchmark(mul_square, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"a * a ({dtype_name})", "Multiply", "Arithmetic", dtype_name
    results.append(r)

    # Divide (float types only to avoid integer division issues)
    if np.issubdtype(dtype, np.floating):
        def div_elementwise(): return a / b_positive
        r = benchmark(div_elementwise, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"a / b ({dtype_name})", "Divide", "Arithmetic", dtype_name
        results.append(r)

    return results

# =============================================================================
# Unary Benchmarks
# =============================================================================

def run_unary_benchmarks(n: int, dtype_name: str, iterations: int) -> List[BenchmarkResult]:
    """Benchmark unary operations for a specific dtype."""
    results = []

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

    # Rounding (float only)
    if np.issubdtype(DTYPES[dtype_name], np.floating):
        def np_floor(): return np.floor(a)
        r = benchmark(np_floor, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.floor ({dtype_name})", "Rounding", "Unary", dtype_name
        results.append(r)

        def np_ceil(): return np.ceil(a)
        r = benchmark(np_ceil, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.ceil ({dtype_name})", "Rounding", "Unary", dtype_name
        results.append(r)

        def np_round(): return np.round(a)
        r = benchmark(np_round, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.round ({dtype_name})", "Rounding", "Unary", dtype_name
        results.append(r)

    # Exp/Log (float only)
    if np.issubdtype(DTYPES[dtype_name], np.floating):
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

    # Trig (float only)
    if np.issubdtype(DTYPES[dtype_name], np.floating):
        angles = (np.random.random(n) * 4 - 2) * np.pi

        def np_sin(): return np.sin(angles)
        r = benchmark(np_sin, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.sin ({dtype_name})", "Trig", "Unary", dtype_name
        results.append(r)

        def np_cos(): return np.cos(angles)
        r = benchmark(np_cos, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.cos ({dtype_name})", "Trig", "Unary", dtype_name
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

    # Var/Std (float only for accuracy)
    if np.issubdtype(DTYPES[dtype_name], np.floating):
        def np_var(): return np.var(a)
        r = benchmark(np_var, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.var ({dtype_name})", "VarStd", "Reduction", dtype_name
        results.append(r)

        def np_std(): return np.std(a)
        r = benchmark(np_std, n, iterations=iterations)
        r.name, r.category, r.suite, r.dtype = f"np.std ({dtype_name})", "VarStd", "Reduction", dtype_name
        results.append(r)

    # Min/Max
    def np_amin(): return np.amin(a)
    r = benchmark(np_amin, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.amin ({dtype_name})", "MinMax", "Reduction", dtype_name
    results.append(r)

    def np_amax(): return np.amax(a)
    r = benchmark(np_amax, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.amax ({dtype_name})", "MinMax", "Reduction", dtype_name
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

    return results

# =============================================================================
# Broadcasting Benchmarks
# =============================================================================

def run_broadcast_benchmarks(n: int, iterations: int) -> List[BenchmarkResult]:
    """Benchmark broadcasting operations."""
    results = []
    dtype_name = 'float64'

    np.random.seed(42)
    matrix_size = int(np.sqrt(n))
    matrix = np.random.random((matrix_size, matrix_size)) * 100
    row_vector = np.random.random(matrix_size) * 100
    col_vector = np.random.random((matrix_size, 1)) * 100
    scalar = np.array(42.0)

    # Scalar broadcast
    def broadcast_scalar(): return matrix + scalar
    r = benchmark(broadcast_scalar, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "matrix + scalar", "Scalar", "Broadcasting", dtype_name
    results.append(r)

    # Row broadcast
    def broadcast_row(): return matrix + row_vector
    r = benchmark(broadcast_row, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "matrix + row_vector (N,M)+(M,)", "Row", "Broadcasting", dtype_name
    results.append(r)

    # Column broadcast
    def broadcast_col(): return matrix + col_vector
    r = benchmark(broadcast_col, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "matrix + col_vector (N,M)+(N,1)", "Column", "Broadcasting", dtype_name
    results.append(r)

    # broadcast_to
    def broadcast_to_row(): return np.broadcast_to(row_vector, (matrix_size, matrix_size))
    r = benchmark(broadcast_to_row, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.broadcast_to(row, (N,M))", "BroadcastTo", "Broadcasting", dtype_name
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

    # Like-based
    def np_zeros_like(): return np.zeros_like(source)
    r = benchmark(np_zeros_like, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = f"np.zeros_like ({dtype_name})", "Like", "Creation", dtype_name
    results.append(r)

    return results

# =============================================================================
# Manipulation Benchmarks
# =============================================================================

def run_manipulation_benchmarks(n: int, iterations: int) -> List[BenchmarkResult]:
    """Benchmark shape manipulation operations."""
    results = []
    dtype_name = 'float64'

    np.random.seed(42)
    rows = int(np.sqrt(n))
    cols = n // rows
    actual_n = rows * cols  # May be slightly less than n due to integer division
    arr_1d = np.random.random(actual_n) * 100  # Use actual_n to ensure reshape works
    arr_2d = np.random.random((rows, cols)) * 100
    d = int(n ** (1/3))
    arr_3d = np.random.random((d, d, d)) * 100

    # Reshape
    def reshape_1d_2d(): return arr_1d.reshape(rows, cols)
    r = benchmark(reshape_1d_2d, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "reshape 1D->2D", "Reshape", "Manipulation", dtype_name
    results.append(r)

    def reshape_2d_1d(): return arr_2d.reshape(-1)
    r = benchmark(reshape_2d_1d, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "reshape 2D->1D", "Reshape", "Manipulation", dtype_name
    results.append(r)

    # Transpose
    def transpose_2d(): return arr_2d.T
    r = benchmark(transpose_2d, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a.T (2D)", "Transpose", "Manipulation", dtype_name
    results.append(r)

    def np_transpose(): return np.transpose(arr_2d)
    r = benchmark(np_transpose, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.transpose (2D)", "Transpose", "Manipulation", dtype_name
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

    # Stack
    arr_1d_b = np.random.random(actual_n) * 100  # Same size as arr_1d

    def np_concatenate(): return np.concatenate([arr_1d, arr_1d_b])
    r = benchmark(np_concatenate, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.concatenate", "Stack", "Manipulation", dtype_name
    results.append(r)

    def np_stack(): return np.stack([arr_1d, arr_1d_b])
    r = benchmark(np_stack, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.stack", "Stack", "Manipulation", dtype_name
    results.append(r)

    return results

# =============================================================================
# Slicing Benchmarks
# =============================================================================

def run_slicing_benchmarks(n: int, iterations: int) -> List[BenchmarkResult]:
    """Benchmark slicing operations."""
    results = []
    dtype_name = 'float64'

    np.random.seed(42)
    arr_1d = np.random.random(n) * 100
    rows = int(np.sqrt(n))
    cols = n // rows
    arr_2d = np.random.random((rows, cols)) * 100

    # Contiguous slice
    contiguous_slice = arr_1d[100:1000]
    strided_slice = arr_1d[::2]

    # Slice creation
    def slice_contiguous(): return arr_1d[100:1000]
    r = benchmark(slice_contiguous, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a[100:1000] (contiguous)", "Create", "Slicing", dtype_name
    results.append(r)

    def slice_strided(): return arr_1d[::2]
    r = benchmark(slice_strided, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a[::2] (strided)", "Create", "Slicing", dtype_name
    results.append(r)

    def slice_reversed(): return arr_1d[::-1]
    r = benchmark(slice_reversed, n, iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "a[::-1] (reversed)", "Create", "Slicing", dtype_name
    results.append(r)

    # Operations on slices
    def sum_contiguous(): return np.sum(contiguous_slice)
    r = benchmark(sum_contiguous, len(contiguous_slice), iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.sum(contiguous_slice)", "SumSlice", "Slicing", dtype_name
    results.append(r)

    def sum_strided(): return np.sum(strided_slice)
    r = benchmark(sum_strided, len(strided_slice), iterations=iterations)
    r.name, r.category, r.suite, r.dtype = "np.sum(strided_slice)", "SumSlice", "Slicing", dtype_name
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
    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms (±{r.stddev_ms:.3f})")

    # Operator syntax (allocates new array)
    def numpy_operator():
        return a + b
    r = benchmark(numpy_operator, n, iterations=iterations)
    r.name = "c = a + b (allocates)"
    r.category = "Dispatch"
    r.suite = "Dispatch"
    r.dtype = "int32"
    results.append(r)
    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms (±{r.stddev_ms:.3f})")

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
    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms (±{r.stddev_ms:.3f})")

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
    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

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
    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

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
    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

    def p3_optimized():
        return np.var(a)
    r = benchmark(p3_optimized, n, iterations=iterations)
    r.name = "NumPy: np.var(a) [optimized]"
    r.category = "Pattern3_Variance"
    r.suite = "Fusion"
    r.dtype = "float64"
    results.append(r)
    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

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
    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

    def p4_multiply():
        return a*a*a + a*a + a
    r = benchmark(p4_multiply, n, iterations=iterations)
    r.name = "NumPy: a*a*a + a*a + a"
    r.category = "Pattern4_Polynomial"
    r.suite = "Fusion"
    r.dtype = "float64"
    results.append(r)
    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

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
    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

    def p5_hypot():
        return np.hypot(a, b)
    r = benchmark(p5_hypot, n, iterations=iterations)
    r.name = "NumPy: np.hypot(a, b) [optimized]"
    r.category = "Pattern5_Euclidean"
    r.suite = "Fusion"
    r.dtype = "float64"
    results.append(r)
    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

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

def main():
    parser = argparse.ArgumentParser(description="NumPy Performance Benchmarks")
    parser.add_argument("--suite", choices=["dispatch", "fusion", "arithmetic", "unary", "reduction",
                                            "broadcast", "creation", "manipulation", "slicing", "all"],
                        default="all", help="Benchmark suite to run")
    parser.add_argument("--n", type=int, default=10_000_000, help="Array size")
    parser.add_argument("--size", choices=["small", "medium", "large"], default=None, help="Array size preset")
    parser.add_argument("--type", type=str, default=None, help="Specific dtype (e.g., int32, float64)")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--quick", action="store_true", help="Quick run (10 iterations, common types only)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--output", type=str, default=None, help="Output JSON to file")
    args = parser.parse_args()

    if args.quick:
        args.iterations = 10

    if args.size:
        args.n = ARRAY_SIZES[args.size]

    # Determine which dtypes to run
    dtypes_to_run = COMMON_DTYPES if args.quick else ARITHMETIC_DTYPES
    if args.type:
        dtypes_to_run = [args.type]

    print(f"\nNumPy {np.__version__}")
    print(f"Python {sys.version.split()[0]}")
    print(f"Array size: N = {args.n:,}")
    print(f"Iterations: {args.iterations}")
    print(f"Types: {dtypes_to_run}")

    all_results = []

    # Dispatch and Fusion benchmarks (original)
    if args.suite in ["dispatch", "all"]:
        all_results.extend(run_dispatch_benchmarks(args.n, args.iterations))

    if args.suite in ["fusion", "all"]:
        all_results.extend(run_fusion_benchmarks(args.n, args.iterations))

    # Comprehensive benchmarks with type iteration
    if args.suite in ["arithmetic", "all"]:
        print(f"\n{'='*60}")
        print(f"  Arithmetic Benchmarks (N={args.n:,})")
        print(f"{'='*60}")
        for dtype in dtypes_to_run:
            if dtype in ARITHMETIC_DTYPES:
                print(f"\n  --- {dtype} ---")
                results = run_arithmetic_benchmarks(args.n, dtype, args.iterations)
                all_results.extend(results)
                for r in results:
                    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

    if args.suite in ["unary", "all"]:
        print(f"\n{'='*60}")
        print(f"  Unary Benchmarks (N={args.n:,})")
        print(f"{'='*60}")
        for dtype in dtypes_to_run:
            if dtype in TRANSCENDENTAL_DTYPES:
                print(f"\n  --- {dtype} ---")
                results = run_unary_benchmarks(args.n, dtype, args.iterations)
                all_results.extend(results)
                for r in results:
                    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

    if args.suite in ["reduction", "all"]:
        print(f"\n{'='*60}")
        print(f"  Reduction Benchmarks (N={args.n:,})")
        print(f"{'='*60}")
        for dtype in dtypes_to_run:
            print(f"\n  --- {dtype} ---")
            results = run_reduction_benchmarks(args.n, dtype, args.iterations)
            all_results.extend(results)
            for r in results:
                print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

    if args.suite in ["broadcast", "all"]:
        all_results.extend(run_broadcast_benchmarks(args.n, args.iterations))

    if args.suite in ["creation", "all"]:
        print(f"\n{'='*60}")
        print(f"  Creation Benchmarks (N={args.n:,})")
        print(f"{'='*60}")
        for dtype in COMMON_DTYPES:
            print(f"\n  --- {dtype} ---")
            results = run_creation_benchmarks(args.n, dtype, args.iterations)
            all_results.extend(results)
            for r in results:
                print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

    if args.suite in ["manipulation", "all"]:
        all_results.extend(run_manipulation_benchmarks(args.n, args.iterations))

    if args.suite in ["slicing", "all"]:
        all_results.extend(run_slicing_benchmarks(args.n, args.iterations))

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
