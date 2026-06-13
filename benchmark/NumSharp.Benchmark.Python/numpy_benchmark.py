#!/usr/bin/env python3
"""
NumPy Performance Benchmarks
============================

Comprehensive benchmarks matching the C# NumSharp.Benchmark.GraphEngine suite.
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

# Common types for quick benchmarks
COMMON_DTYPES = ['int32', 'int64', 'float32', 'float64']

# Arithmetic types (excludes bool). complex128 is fine: run_arithmetic restricts divide/modulo
# to COMMON_DTYPES, so complex only sees +, -, * (and sum/mean/min/max in run_reduction).
ARITHMETIC_DTYPES = ['uint8', 'int8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
                     'float16', 'float32', 'float64', 'complex128']

# Transcendental types (for sqrt, exp, log, trig)
TRANSCENDENTAL_DTYPES = ['float16', 'float32', 'float64']

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

# =============================================================================
# Extended coverage suites (mirror the new C# benchmark classes 1:1 by op name)
# =============================================================================

# dtype sets that mirror the C# TypeParameterSource collections.
BITWISE_DTYPES = ['bool', 'uint8', 'int8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
FLOAT_DTYPES = ['float16', 'float32', 'float64']


def _b(func, n, iterations, name, suite, dtype, category=""):
    """Run one benchmark and tag it. name MUST equal the C# [Benchmark(Description=...)]."""
    r = benchmark(func, n, iterations=iterations)
    r.name = f"{name} ({dtype})"
    r.suite = suite
    r.dtype = dtype
    r.category = category or suite
    return r


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
    ]


def run_bitwise_benchmarks(n, dtype_name, iterations):
    a = create_random_array(n, dtype_name, seed=42)
    b = create_random_array(n, dtype_name, seed=43)
    return [
        _b(lambda: a & b, n, iterations, "a & b", "Bitwise", dtype_name),
        _b(lambda: a | b, n, iterations, "a | b", "Bitwise", dtype_name),
        _b(lambda: a ^ b, n, iterations, "a ^ b", "Bitwise", dtype_name),
        _b(lambda: np.invert(a), n, iterations, "np.invert(a)", "Bitwise", dtype_name),
        _b(lambda: np.left_shift(a, 2), n, iterations, "np.left_shift(a, 2)", "Bitwise", dtype_name),
        _b(lambda: np.right_shift(a, 2), n, iterations, "np.right_shift(a, 2)", "Bitwise", dtype_name),
    ]


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
    return [
        _b(lambda: np.isnan(a), n, iterations, "np.isnan(a)", "Logic", dtype_name),
        _b(lambda: np.isinf(a), n, iterations, "np.isinf(a)", "Logic", dtype_name),
        _b(lambda: np.isfinite(a), n, iterations, "np.isfinite(a)", "Logic", dtype_name),
        _b(lambda: np.maximum(a, b), n, iterations, "np.maximum(a, b)", "Logic", dtype_name),
        _b(lambda: np.minimum(a, b), n, iterations, "np.minimum(a, b)", "Logic", dtype_name),
        _b(lambda: np.isclose(a, b), n, iterations, "np.isclose(a, b)", "Logic", dtype_name),
        _b(lambda: np.allclose(a, b), n, iterations, "np.allclose(a, b)", "Logic", dtype_name),
        _b(lambda: np.array_equal(a, b), n, iterations, "np.array_equal(a, b)", "Logic", dtype_name),
    ]


def run_bool_logic_benchmarks(n, iterations):
    np.random.seed(42)
    mask = np.random.random(n) > 0.5
    return [
        _b(lambda: bool(np.all(mask)), n, iterations, "np.all(a)", "Logic", "bool"),
        _b(lambda: bool(np.any(mask)), n, iterations, "np.any(a)", "Logic", "bool"),
    ]


def run_nan_reduction_benchmarks(n, dtype_name, iterations):
    a = create_random_array(n, dtype_name, seed=42)
    return [
        _b(lambda: np.nansum(a), n, iterations, "np.nansum(a)", "Reduction", dtype_name),
        _b(lambda: np.nanmean(a), n, iterations, "np.nanmean(a)", "Reduction", dtype_name),
        _b(lambda: np.nanmax(a), n, iterations, "np.nanmax(a)", "Reduction", dtype_name),
        _b(lambda: np.nanmin(a), n, iterations, "np.nanmin(a)", "Reduction", dtype_name),
        _b(lambda: np.nanstd(a), n, iterations, "np.nanstd(a)", "Reduction", dtype_name),
        _b(lambda: np.nanvar(a), n, iterations, "np.nanvar(a)", "Reduction", dtype_name),
        _b(lambda: np.nanprod(a), n, iterations, "np.nanprod(a)", "Reduction", dtype_name),
        _b(lambda: np.nanmedian(a), n, iterations, "np.nanmedian(a)", "Reduction", dtype_name),
        _b(lambda: np.nanpercentile(a, 50), n, iterations, "np.nanpercentile(a, 50)", "Reduction", dtype_name),
        _b(lambda: np.nanquantile(a, 0.5), n, iterations, "np.nanquantile(a, 0.5)", "Reduction", dtype_name),
    ]


def run_statistics_benchmarks(n, dtype_name, iterations):
    a = create_random_array(n, dtype_name, seed=42)
    return [
        _b(lambda: np.median(a), n, iterations, "np.median(a)", "Statistics", dtype_name),
        _b(lambda: np.percentile(a, 50), n, iterations, "np.percentile(a, 50)", "Statistics", dtype_name),
        _b(lambda: np.quantile(a, 0.5), n, iterations, "np.quantile(a, 0.5)", "Statistics", dtype_name),
        _b(lambda: np.average(a), n, iterations, "np.average(a)", "Statistics", dtype_name),
        _b(lambda: np.ptp(a), n, iterations, "np.ptp(a)", "Statistics", dtype_name),
        _b(lambda: np.count_nonzero(a), n, iterations, "np.count_nonzero(a)", "Statistics", dtype_name),
    ]


def run_sorting_benchmarks(n, dtype_name, iterations):
    a = create_random_array(n, dtype_name, seed=42)
    srt = np.arange(n, dtype=DTYPES[dtype_name])
    return [
        _b(lambda: np.argsort(a), n, iterations, "np.argsort(a)", "Sorting", dtype_name),
        _b(lambda: np.nonzero(a), n, iterations, "np.nonzero(a)", "Sorting", dtype_name),
        # Query N points (a) into the sorted target → N binary searches (real work that
        # scales with N), matching the C# benchmark. A single scalar lookup is pure call
        # overhead, not a throughput comparison.
        _b(lambda: np.searchsorted(srt, a), n, iterations, "np.searchsorted(a, v)", "Sorting", dtype_name),
    ]


def run_linalg_benchmarks(n, iterations):
    np.random.seed(42)
    m = int(n ** 0.5)
    mc = min(m, 384)
    v = np.random.random(n)
    vM = np.random.random(m)
    matA = np.random.random((mc, mc))
    matB = np.random.random((mc, mc))
    return [
        _b(lambda: np.dot(v, v), n, iterations, "np.dot(a, b)", "LinearAlgebra", "float64"),
        _b(lambda: np.outer(vM, vM), n, iterations, "np.outer(a, b)", "LinearAlgebra", "float64"),
        _b(lambda: np.matmul(matA, matB), n, iterations, "np.matmul(A, B)", "LinearAlgebra", "float64"),
    ]


def run_where_benchmarks(n, iterations):
    np.random.seed(42)
    a = np.random.random(n) * 100 - 50
    b = np.random.random(n) * 100 - 50
    cond = a > 0
    return [
        _b(lambda: np.where(cond, a, b), n, iterations, "np.where(cond, a, b)", "Selection", "float64"),
        _b(lambda: np.where(cond), n, iterations, "np.where(cond)", "Selection", "float64"),
    ]


def run_cumulative_benchmarks(n, dtype_name, iterations):
    a = create_random_array(n, dtype_name, seed=42)
    return [
        _b(lambda: np.cumprod(a), n, iterations, "np.cumprod(a)", "Reduction", dtype_name),
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
                    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

    if suite in ["unary", "all"]:
        print(f"\n{'='*60}\n  Unary Benchmarks (N={n:,})\n{'='*60}")
        for dtype in dtypes_to_run:
            if dtype in TRANSCENDENTAL_DTYPES:
                print(f"\n  --- {dtype} ---")
                results = run_unary_benchmarks(n, dtype, iterations)
                results_all.extend(results)
                for r in results:
                    print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")
        # Extra unary math (cbrt/reciprocal/square/negative/positive/trunc) — mirrors
        # the C# UnaryExtraBenchmarks class (also under the Unary namespace).
        for dtype in FLOAT_DTYPES:
            results_all.extend(run_unary_extra_benchmarks(n, dtype, iterations))

    if suite in ["reduction", "all"]:
        print(f"\n{'='*60}\n  Reduction Benchmarks (N={n:,})\n{'='*60}")
        for dtype in dtypes_to_run:
            print(f"\n  --- {dtype} ---")
            results = run_reduction_benchmarks(n, dtype, iterations)
            results_all.extend(results)
            for r in results:
                print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")
        # NaN-aware reductions + cumprod — mirror C# NanReductionBenchmarks / CumulativeBenchmarks.
        for dtype in FLOAT_DTYPES:
            results_all.extend(run_nan_reduction_benchmarks(n, dtype, iterations))
            results_all.extend(run_cumulative_benchmarks(n, dtype, iterations))

    if suite in ["broadcast", "all"]:
        results_all.extend(run_broadcast_benchmarks(n, iterations))

    if suite in ["creation", "all"]:
        print(f"\n{'='*60}\n  Creation Benchmarks (N={n:,})\n{'='*60}")
        for dtype in COMMON_DTYPES:
            print(f"\n  --- {dtype} ---")
            results = run_creation_benchmarks(n, dtype, iterations)
            results_all.extend(results)
            for r in results:
                print(f"  {r.name:<40} {r.mean_ms:>8.3f} ms")

    if suite in ["manipulation", "all"]:
        results_all.extend(run_manipulation_benchmarks(n, iterations))

    if suite in ["slicing", "all"]:
        results_all.extend(run_slicing_benchmarks(n, iterations))

    if suite in ["comparison", "all"]:
        for dtype in COMMON_DTYPES:
            results_all.extend(run_comparison_benchmarks(n, dtype, iterations))

    if suite in ["bitwise", "all"]:
        for dtype in BITWISE_DTYPES:
            results_all.extend(run_bitwise_benchmarks(n, dtype, iterations))

    if suite in ["logic", "all"]:
        for dtype in FLOAT_DTYPES:
            results_all.extend(run_logic_benchmarks(n, dtype, iterations))
        results_all.extend(run_bool_logic_benchmarks(n, iterations))

    if suite in ["statistics", "all"]:
        for dtype in FLOAT_DTYPES:
            results_all.extend(run_statistics_benchmarks(n, dtype, iterations))

    if suite in ["sorting", "all"]:
        for dtype in COMMON_DTYPES:
            results_all.extend(run_sorting_benchmarks(n, dtype, iterations))

    if suite in ["linalg", "all"]:
        results_all.extend(run_linalg_benchmarks(n, iterations))

    if suite in ["selection", "all"]:
        results_all.extend(run_where_benchmarks(n, iterations))

    return results_all


def main():
    parser = argparse.ArgumentParser(description="NumPy Performance Benchmarks")
    parser.add_argument("--suite", choices=["dispatch", "fusion", "arithmetic", "unary", "reduction",
                                            "broadcast", "creation", "manipulation", "slicing",
                                            "comparison", "bitwise", "logic", "statistics", "sorting",
                                            "linalg", "selection", "all"],
                        default="all", help="Benchmark suite to run")
    parser.add_argument("--n", type=int, default=10_000_000, help="Array size")
    parser.add_argument("--size", choices=["small", "medium", "large", "all"], default=None,
                        help="Array size preset ('all' sweeps small+medium+large in one run)")
    parser.add_argument("--cache-sizes", action="store_true",
                        help="Sweep all three cache-tier sizes (small, medium, large) in one invocation")
    parser.add_argument("--type", type=str, default=None, help="Specific dtype (e.g., int32, float64)")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--quick", action="store_true", help="Quick run (10 iterations, common types only)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--output", type=str, default=None, help="Output JSON to file")
    args = parser.parse_args()

    if args.quick:
        args.iterations = 10

    if args.size and args.size != "all":
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

    # Sizes to sweep: --size all (or --cache-sizes) runs the three cache-tier sizes in one
    # invocation so a single JSON carries all three; otherwise the single resolved args.n.
    if args.cache_sizes or args.size == "all":
        sizes_to_run = [ARRAY_SIZES["small"], ARRAY_SIZES["medium"], ARRAY_SIZES["large"]]
    else:
        sizes_to_run = [args.n]

    print(f"Sizes to run: {[f'{n:,}' for n in sizes_to_run]}")

    all_results = []
    for n in sizes_to_run:
        print(f"\n{'#'*64}\n#  ARRAY SIZE  N = {n:,}\n{'#'*64}")
        all_results.extend(run_suites(n, args.suite, dtypes_to_run, args.iterations))

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
