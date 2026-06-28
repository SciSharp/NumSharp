#!/usr/bin/env python3
"""
NumPy baseline benchmarks for SIMD & Broadcasting investigation.

This script runs the same benchmarks as the C# exploration suite to establish
NumPy performance baselines for comparison.

Usage:
    python numpy_baseline.py                    # All benchmarks
    python numpy_baseline.py --suite broadcast  # Specific suite
    python numpy_baseline.py --quick            # Quick mode
    python numpy_baseline.py --output results.json  # Export JSON
"""

import argparse
import json
import time
import sys
from dataclasses import dataclass, asdict
from typing import List, Callable, Any
import numpy as np

# Benchmark configuration
SEED = 42
WARMUP = 3
MEASURE = 10
QUICK_MEASURE = 5

# Standard sizes matching C# ArraySizes
SIZES = {
    'tiny': 32,
    'small': 1_000,
    'medium': 100_000,
    'large': 1_000_000,
    'huge': 10_000_000,
    'massive': 100_000_000,
}

STANDARD_SIZES = [SIZES['small'], SIZES['medium'], SIZES['large'], SIZES['huge']]
QUICK_SIZES = [SIZES['medium'], SIZES['huge']]

# Dtypes matching C# Dtypes
DTYPES = {
    'byte': np.uint8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'float32': np.float32,
    'float64': np.float64,
}

COMMON_DTYPES = ['int32', 'float64']
ALL_DTYPES = list(DTYPES.keys())


@dataclass
class BenchResult:
    scenario: str
    strategy: str
    dtype: str
    size: int
    mean_us: float
    stddev_us: float
    min_us: float
    max_us: float
    gbps: float
    reps: int
    timestamp: str
    suite: str = ""
    notes: str = ""
    speedup_vs_baseline: float = None


def benchmark(func: Callable, size: int, element_bytes: int,
              warmup: int = WARMUP, measure: int = MEASURE) -> tuple:
    """Run a benchmark with warmup and measurement phases."""
    # Warmup
    for _ in range(warmup):
        func()

    # Measure
    times = []
    for _ in range(measure):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1_000_000)  # Convert to microseconds

    mean_us = np.mean(times)
    stddev_us = np.std(times)
    min_us = np.min(times)
    max_us = np.max(times)

    # Calculate GB/s (read 2 inputs + write 1 output = 3 arrays)
    total_bytes = size * element_bytes * 3
    seconds = mean_us / 1_000_000
    gbps = (total_bytes / 1e9) / seconds if seconds > 0 else 0

    return mean_us, stddev_us, min_us, max_us, gbps


def print_header(title: str):
    """Print formatted header."""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()
    print(f"{'Scenario':<12} {'Strategy':<10} {'Dtype':<8} {'Size':>12} | {'Mean (us)':>12} ± {'StdDev':>8} | {'GB/s':>8}")
    print("-" * 80)


def print_result(r: BenchResult):
    """Print a single result."""
    speedup = f" ({r.speedup_vs_baseline:.2f}x)" if r.speedup_vs_baseline else ""
    print(f"{r.scenario:<12} {r.strategy:<10} {r.dtype:<8} {r.size:>12,} | {r.mean_us:>12.2f} ± {r.stddev_us:>8.2f} | {r.gbps:>8.2f}{speedup}")


def run_broadcast_scenarios(sizes: List[int], dtypes: List[str], measure: int) -> List[BenchResult]:
    """Run all 7 broadcasting scenarios."""
    results = []
    suite = "BroadcastScenarios"

    print_header(f"{suite}: NumPy Baseline")

    for dtype_name in dtypes:
        print(f"\n--- dtype={dtype_name} ---")
        dtype = DTYPES[dtype_name]
        element_bytes = np.dtype(dtype).itemsize

        for size in sizes:
            np.random.seed(SEED)

            # S1: Contiguous same shape
            a = np.random.rand(size).astype(dtype)
            b = np.random.rand(size).astype(dtype)

            mean, std, min_t, max_t, gbps = benchmark(
                lambda: a + b, size, element_bytes, measure=measure)

            r = BenchResult(
                scenario="S1_contiguous",
                strategy="NumPy",
                dtype=dtype_name,
                size=size,
                mean_us=mean,
                stddev_us=std,
                min_us=min_t,
                max_us=max_t,
                gbps=gbps,
                reps=measure,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                suite=suite
            )
            results.append(r)
            print_result(r)

            # S2: Scalar broadcast
            scalar = dtype(42.5) if np.issubdtype(dtype, np.floating) else dtype(42)

            mean, std, min_t, max_t, gbps = benchmark(
                lambda: a + scalar, size, element_bytes, measure=measure)

            r = BenchResult(
                scenario="S2_scalar",
                strategy="NumPy",
                dtype=dtype_name,
                size=size,
                mean_us=mean,
                stddev_us=std,
                min_us=min_t,
                max_us=max_t,
                gbps=gbps,
                reps=measure,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                suite=suite
            )
            results.append(r)
            print_result(r)

            # S4: Row broadcast
            rows = int(np.sqrt(size))
            cols = size // rows
            actual_size = rows * cols

            matrix = np.random.rand(rows, cols).astype(dtype)
            row = np.random.rand(cols).astype(dtype)

            mean, std, min_t, max_t, gbps = benchmark(
                lambda: matrix + row, actual_size, element_bytes, measure=measure)

            r = BenchResult(
                scenario="S4_rowBC",
                strategy="NumPy",
                dtype=dtype_name,
                size=actual_size,
                mean_us=mean,
                stddev_us=std,
                min_us=min_t,
                max_us=max_t,
                gbps=gbps,
                reps=measure,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                suite=suite
            )
            results.append(r)
            print_result(r)

            # S5: Column broadcast
            col = np.random.rand(rows, 1).astype(dtype)

            mean, std, min_t, max_t, gbps = benchmark(
                lambda: matrix + col, actual_size, element_bytes, measure=measure)

            r = BenchResult(
                scenario="S5_colBC",
                strategy="NumPy",
                dtype=dtype_name,
                size=actual_size,
                mean_us=mean,
                stddev_us=std,
                min_us=min_t,
                max_us=max_t,
                gbps=gbps,
                reps=measure,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                suite=suite
            )
            results.append(r)
            print_result(r)

    return results


def run_size_thresholds(dtypes: List[str], measure: int) -> List[BenchResult]:
    """Find performance crossover points at different sizes."""
    results = []
    suite = "SizeThresholds"

    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
             16384, 32768, 65536, 131072]

    print_header(f"{suite}: NumPy Performance by Size")

    for dtype_name in dtypes:
        print(f"\n--- dtype={dtype_name} ---")
        dtype = DTYPES[dtype_name]
        element_bytes = np.dtype(dtype).itemsize

        for size in sizes:
            np.random.seed(SEED)
            a = np.random.rand(size).astype(dtype)
            b = np.random.rand(size).astype(dtype)

            mean, std, min_t, max_t, gbps = benchmark(
                lambda: a + b, size, element_bytes, measure=measure)

            r = BenchResult(
                scenario="Threshold",
                strategy="NumPy",
                dtype=dtype_name,
                size=size,
                mean_us=mean,
                stddev_us=std,
                min_us=min_t,
                max_us=max_t,
                gbps=gbps,
                reps=measure,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                suite=suite
            )
            results.append(r)
            print(f"  {size:>8,} | {mean:>10.2f} us | {gbps:>8.2f} GB/s")

    return results


def run_memory_patterns(sizes: List[int], measure: int) -> List[BenchResult]:
    """Test strided access patterns."""
    results = []
    suite = "MemoryPatterns"

    print_header(f"{suite}: Strided Access Performance")

    for size in sizes:
        print(f"\n--- Size: {size:,} ---")

        strides = [1, 2, 4, 8, 16, 32, 64]

        for stride in strides:
            actual_size = size * stride
            if actual_size > 100_000_000:
                continue

            np.random.seed(SEED)
            src = np.random.rand(actual_size)
            dst = np.empty(size)

            def strided_copy():
                dst[:] = src[::stride]

            mean, std, min_t, max_t, gbps = benchmark(
                strided_copy, size, 8, measure=measure)

            r = BenchResult(
                scenario="Strided",
                strategy=f"Stride{stride}",
                dtype="float64",
                size=size,
                mean_us=mean,
                stddev_us=std,
                min_us=min_t,
                max_us=max_t,
                gbps=gbps,
                reps=measure,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                suite=suite
            )
            results.append(r)
            print(f"  Stride {stride:>2}: {mean:>10.2f} us | {gbps:>8.2f} GB/s")

    return results


def run_chained_operations(sizes: List[int], measure: int) -> List[BenchResult]:
    """Test chained operations (a + b + c + d)."""
    results = []
    suite = "ChainedOps"

    print_header(f"{suite}: Chained Operations (a+b+c+d)")

    for size in sizes:
        print(f"\n--- Size: {size:,} ---")

        np.random.seed(SEED)
        a = np.random.rand(size)
        b = np.random.rand(size)
        c = np.random.rand(size)
        d = np.random.rand(size)

        # Standard chained add
        mean, std, min_t, max_t, gbps = benchmark(
            lambda: a + b + c + d, size, 8, measure=measure)

        r = BenchResult(
            scenario="Chained",
            strategy="NumPy",
            dtype="float64",
            size=size,
            mean_us=mean,
            stddev_us=std,
            min_us=min_t,
            max_us=max_t,
            gbps=gbps,
            reps=measure,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            suite=suite
        )
        results.append(r)
        print_result(r)

        # Using np.add.reduce
        arr = np.stack([a, b, c, d])
        mean, std, min_t, max_t, gbps = benchmark(
            lambda: np.sum(arr, axis=0), size, 8, measure=measure)

        r = BenchResult(
            scenario="Chained",
            strategy="np.sum",
            dtype="float64",
            size=size,
            mean_us=mean,
            stddev_us=std,
            min_us=min_t,
            max_us=max_t,
            gbps=gbps,
            reps=measure,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            suite=suite
        )
        results.append(r)
        print_result(r)

    return results


def print_summary(results: List[BenchResult]):
    """Print summary statistics."""
    print()
    print("=== Summary ===")
    print(f"Total benchmarks: {len(results)}")

    by_scenario = {}
    for r in results:
        key = r.scenario
        if key not in by_scenario:
            by_scenario[key] = []
        by_scenario[key].append(r)

    print("\nBy scenario:")
    for scenario, rs in sorted(by_scenario.items()):
        avg_gbps = np.mean([r.gbps for r in rs])
        print(f"  {scenario}: {len(rs)} tests, avg {avg_gbps:.2f} GB/s")


def export_json(results: List[BenchResult], filepath: str):
    """Export results to JSON."""
    data = [asdict(r) for r in results]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nJSON exported to: {filepath}")


def print_environment():
    """Print environment information."""
    print("Environment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy: {np.__version__}")
    print(f"  BLAS: {np.show_config(mode='dicts').get('Build Dependencies', {}).get('blas', 'unknown')}")
    print()


def main():
    parser = argparse.ArgumentParser(description="NumPy baseline benchmarks")
    parser.add_argument("--suite", choices=["broadcast", "thresholds", "memory", "chained", "all"],
                        default="all", help="Suite to run")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer iterations)")
    parser.add_argument("--output", type=str, help="JSON output file")
    parser.add_argument("--dtypes", type=str, default="common",
                        choices=["common", "all"], help="Dtypes to test")

    args = parser.parse_args()

    print_environment()

    sizes = QUICK_SIZES if args.quick else STANDARD_SIZES
    dtypes = COMMON_DTYPES if args.dtypes == "common" else ALL_DTYPES
    measure = QUICK_MEASURE if args.quick else MEASURE

    all_results = []

    if args.suite in ["broadcast", "all"]:
        all_results.extend(run_broadcast_scenarios(sizes, dtypes, measure))

    if args.suite in ["thresholds", "all"]:
        all_results.extend(run_size_thresholds(dtypes, measure))

    if args.suite in ["memory", "all"]:
        all_results.extend(run_memory_patterns(sizes, measure))

    if args.suite in ["chained", "all"]:
        all_results.extend(run_chained_operations(sizes, measure))

    print_summary(all_results)

    if args.output:
        export_json(all_results, args.output)


if __name__ == "__main__":
    main()
