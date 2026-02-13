#!/usr/bin/env python3
"""
Merge NumPy and NumSharp benchmark results into a unified comparison table.

Outputs:
  - benchmark-report.json: Machine-readable merged data with ratios
  - benchmark-report.md: Markdown table for documentation
  - benchmark-report.csv: CSV for spreadsheet analysis

Usage:
  python merge-results.py
  python merge-results.py --numpy ../benchmark-report.json --csharp ../NumSharp.Benchmark.GraphEngine/BenchmarkDotNet.Artifacts/results/
  python merge-results.py --format csv

Note: This script is typically invoked from run-benchmarks.ps1 with explicit paths.
"""

import json
import os
import sys
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class UnifiedResult:
    """A single benchmark comparison result."""
    operation: str
    suite: str
    category: str
    dtype: str
    n: int
    numpy_ms: float
    numsharp_ms: Optional[float]
    ratio: Optional[float]  # NumSharp / NumPy
    status: str  # "faster", "close", "slower", "much_slower", "no_data"

    def to_dict(self) -> dict:
        return asdict(self)


def load_numpy_results(path: str) -> List[dict]:
    """Load NumPy benchmark results from JSON."""
    if not os.path.exists(path):
        print(f"Warning: NumPy results not found at {path}")
        return []
    with open(path, 'r') as f:
        return json.load(f)


def load_csharp_results(artifacts_dir: str) -> List[dict]:
    """Load BenchmarkDotNet results from artifacts directory."""
    results = []
    if not os.path.exists(artifacts_dir):
        print(f"Warning: C# artifacts not found at {artifacts_dir}")
        return []

    # Find all *-report*.json files (including -full-compressed.json)
    for pattern in ["*-report.json", "*-report-full-compressed.json"]:
        full_pattern = os.path.join(artifacts_dir, pattern)
        for json_file in glob.glob(full_pattern):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'Benchmarks' in data:
                        for bench in data['Benchmarks']:
                            result = parse_bdn_benchmark(bench)
                            if result:
                                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to parse {json_file}: {e}")

    return results


def parse_bdn_benchmark(bench: dict) -> Optional[dict]:
    """Parse a single BenchmarkDotNet benchmark result."""
    try:
        method = bench.get('Method', '')
        method_title = bench.get('MethodTitle', method)
        params = bench.get('Parameters', '')
        stats = bench.get('Statistics', {})

        # Extract N and DType from parameters (format: "N=1000&DType=UInt16" or "N=1000, DType=UInt16")
        n = 10_000_000
        dtype = 'float64'

        if params:
            # Handle both "&" and ", " separators
            parts = params.replace('&', ', ').split(', ')
            for part in parts:
                part = part.strip()
                if part.startswith('N='):
                    n = int(part[2:])
                elif part.startswith('DType='):
                    dtype = part[6:].lower()

        # Only use Large array size (10M) for comparison
        if n != 10_000_000:
            return None

        # Convert nanoseconds to milliseconds
        mean_ns = stats.get('Mean', 0)
        mean_ms = mean_ns / 1_000_000

        stddev_ns = stats.get('StandardDeviation', stats.get('StdDev', 0))
        stddev_ms = stddev_ns / 1_000_000

        # Map dtype to numpy names
        dtype_map = {
            'int32': 'int32', 'int64': 'int64', 'single': 'float32', 'double': 'float64',
            'byte': 'uint8', 'uint16': 'uint16', 'uint32': 'uint32', 'uint64': 'uint64',
            'int16': 'int16', 'boolean': 'bool', 'decimal': 'decimal'
        }
        dtype = dtype_map.get(dtype.lower(), dtype.lower())

        # Clean up method title
        operation = method_title.strip("'")

        return {
            'name': operation,
            'method': method,
            'dtype': dtype,
            'n': n,
            'mean_ms': mean_ms,
            'stddev_ms': stddev_ms
        }
    except Exception as e:
        print(f"Warning: Failed to parse benchmark: {e}")
        return None


def method_to_operation(method: str) -> str:
    """Convert C# method name to operation name matching NumPy results."""
    # Map common method names to NumPy-style names
    mappings = {
        'Add_Elementwise': 'a + b',
        'Add_Scalar': 'a + scalar',
        'Subtract_Elementwise': 'a - b',
        'Multiply_Elementwise': 'a * b',
        'Multiply_Same': 'a * a',
        'Divide_Elementwise': 'a / b',
        'Sum_Full': 'np.sum',
        'Sum_Axis0': 'np.sum axis=0',
        'Sum_Axis1': 'np.sum axis=1',
        'Mean_Full': 'np.mean',
        'Var_Full': 'np.var',
        'Std_Full': 'np.std',
        'Min_Full': 'np.amin',
        'Max_Full': 'np.amax',
        'ArgMin_Full': 'np.argmin',
        'ArgMax_Full': 'np.argmax',
        'Sqrt': 'np.sqrt',
        'Abs': 'np.abs',
        'Sign': 'np.sign',
        'Floor': 'np.floor',
        'Ceil': 'np.ceil',
        'Round': 'np.round',
        'Exp': 'np.exp',
        'Log': 'np.log',
        'Log10': 'np.log10',
        'Sin': 'np.sin',
        'Cos': 'np.cos',
        'Zeros': 'np.zeros',
        'Ones': 'np.ones',
        'Full': 'np.full',
        'Empty': 'np.empty',
        'Copy': 'np.copy',
        'Zeros_Like': 'np.zeros_like',
    }

    return mappings.get(method, method)


def get_status(ratio: Optional[float]) -> str:
    """Get status string from ratio."""
    if ratio is None:
        return "no_data"
    if ratio <= 1.0:
        return "faster"
    if ratio <= 2.0:
        return "close"
    if ratio <= 5.0:
        return "slower"
    return "much_slower"


def get_status_icon(status: str) -> str:
    """Get status icon for markdown."""
    icons = {
        "faster": "✅",
        "close": "🟡",
        "slower": "🟠",
        "much_slower": "🔴",
        "no_data": "⚪"
    }
    return icons.get(status, "⚪")


def normalize_op_name(name: str) -> str:
    """Normalize operation name for matching."""
    # Remove dtype suffix like " (int32)"
    import re
    name = re.sub(r'\s*\([^)]*\)\s*$', '', name)
    # Remove quotes
    name = name.strip("'\"")
    # Normalize common patterns
    name = name.lower()
    name = re.sub(r'\s+', ' ', name)
    # Map C# names to NumPy names
    mappings = {
        'np.sum(a) [full]': 'np.sum',
        'np.sum(a, axis=0)': 'np.sum axis=0',
        'np.sum(a, axis=1)': 'np.sum axis=1',
        'a + b (element-wise)': 'a + b',
        'np.add(a, b)': 'a + b',
        'a + scalar': 'a + scalar',
    }
    return mappings.get(name, name)


def merge_results(numpy_results: List[dict], csharp_results: List[dict]) -> List[UnifiedResult]:
    """Merge NumPy and C# results into unified comparison."""
    unified = []

    # Index C# results by (normalized_operation, dtype)
    csharp_index: Dict[tuple, dict] = {}
    for r in csharp_results:
        norm_name = normalize_op_name(r['name'])
        key = (norm_name, r['dtype'].lower())
        csharp_index[key] = r
        # Debug
        # print(f"C# key: {key}")

    # Process each NumPy result
    for np_result in numpy_results:
        name = np_result.get('name', '')
        dtype = np_result.get('dtype', 'float64')
        n = np_result.get('n', 10_000_000)
        suite = np_result.get('suite', 'General')
        category = np_result.get('category', '')
        numpy_ms = np_result.get('mean_ms', 0)

        # Look for matching C# result
        norm_name = normalize_op_name(name)
        key = (norm_name, dtype.lower())
        cs_result = csharp_index.get(key)

        numsharp_ms = cs_result['mean_ms'] if cs_result else None
        ratio = numsharp_ms / numpy_ms if (numsharp_ms and numpy_ms > 0) else None
        status = get_status(ratio)

        unified.append(UnifiedResult(
            operation=name,
            suite=suite,
            category=category,
            dtype=dtype,
            n=n,
            numpy_ms=round(numpy_ms, 3),
            numsharp_ms=round(numsharp_ms, 3) if numsharp_ms else None,
            ratio=round(ratio, 2) if ratio else None,
            status=status
        ))

    return unified


def generate_json(results: List[UnifiedResult], output_path: str):
    """Generate JSON output."""
    data = [r.to_dict() for r in results]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"JSON written to: {output_path}")


def generate_csv(results: List[UnifiedResult], output_path: str):
    """Generate CSV output."""
    import csv
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Operation', 'Suite', 'Category', 'DType', 'N',
                        'NumPy (ms)', 'NumSharp (ms)', 'Ratio', 'Status'])
        for r in results:
            writer.writerow([
                r.operation, r.suite, r.category, r.dtype, r.n,
                r.numpy_ms, r.numsharp_ms or '', r.ratio or '', r.status
            ])
    print(f"CSV written to: {output_path}")


def generate_markdown(results: List[UnifiedResult], output_path: str):
    """Generate concise Markdown comparison matrix."""

    # Count stats
    faster = sum(1 for r in results if r.status == 'faster')
    close = sum(1 for r in results if r.status == 'close')
    slower = sum(1 for r in results if r.status == 'slower')
    much_slower = sum(1 for r in results if r.status == 'much_slower')
    no_data = sum(1 for r in results if r.status == 'no_data')
    total = len(results)

    lines = [
        "# NumSharp vs NumPy Performance",
        "",
        "**Baseline:** NumPy (N=10M elements)",
        "",
        "**Ratio** = NumSharp ÷ NumPy → Lower is better for NumSharp",
        "",
        "| | Status | Ratio | Meaning |",
        "|:-:|--------|:-----:|---------|",
        "|✅| Faster | <1.0 | NumSharp beats NumPy |",
        "|🟡| Close | 1-2x | Acceptable parity |",
        "|🟠| Slower | 2-5x | Optimization target |",
        "|🔴| Slow | >5x | Priority fix |",
        "|⚪| Pending | - | C# benchmark not run |",
        "",
        "---",
        "",
        f"**Summary:** {total} ops | ✅ {faster} | 🟡 {close} | 🟠 {slower} | 🔴 {much_slower} | ⚪ {no_data}",
        "",
    ]

    # Get results with valid data (both sides, NumPy >= 0.001ms to avoid division issues)
    with_data = [r for r in results if r.ratio is not None and r.numpy_ms >= 0.001]

    if with_data:
        # Sort by ratio - best (lowest) first
        sorted_by_ratio = sorted(with_data, key=lambda r: r.ratio)

        # Top 15 best (NumSharp faster or closest)
        best_15 = sorted_by_ratio[:15]
        lines.append("### 🏆 Top 15 Best (NumSharp closest to NumPy)")
        lines.append("")
        lines.append("| | Operation | Type | NumPy | NumSharp | Ratio |")
        lines.append("|:-:|-----------|:----:|------:|---------:|------:|")
        for r in best_15:
            icon = get_status_icon(r.status)
            lines.append(f"|{icon}| {r.operation} | {r.dtype} | {r.numpy_ms:.1f} | {r.numsharp_ms:.1f} | {r.ratio:.1f}x |")
        lines.append("")

        # Top 15 worst (NumPy much faster)
        worst_15 = sorted_by_ratio[-15:][::-1]  # Reverse to show worst first
        lines.append("### 🔻 Top 15 Worst (Optimization priorities)")
        lines.append("")
        lines.append("| | Operation | Type | NumPy | NumSharp | Ratio |")
        lines.append("|:-:|-----------|:----:|------:|---------:|------:|")
        for r in worst_15:
            icon = get_status_icon(r.status)
            lines.append(f"|{icon}| {r.operation} | {r.dtype} | {r.numpy_ms:.1f} | {r.numsharp_ms:.1f} | {r.ratio:.1f}x |")
        lines.append("")

        lines.append("---")
        lines.append("")

    # Group by suite
    suites: Dict[str, List[UnifiedResult]] = {}
    for r in results:
        suite = r.suite or "General"
        if suite not in suites:
            suites[suite] = []
        suites[suite].append(r)

    # Generate compact table for each suite
    for suite_name, suite_results in suites.items():
        lines.append(f"### {suite_name}")
        lines.append("")
        lines.append("| | Operation | Type | NumPy | NumSharp | Ratio |")
        lines.append("|:-:|-----------|:----:|------:|---------:|------:|")

        for r in suite_results:
            icon = get_status_icon(r.status)
            numsharp_str = f"{r.numsharp_ms:.1f}" if r.numsharp_ms else "-"
            ratio_str = f"{r.ratio:.1f}x" if r.ratio else "-"
            lines.append(f"|{icon}| {r.operation} | {r.dtype} | {r.numpy_ms:.1f} | {numsharp_str} | {ratio_str} |")

        lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Markdown written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Merge NumPy and NumSharp benchmark results')
    parser.add_argument('--numpy', default='benchmark-report.json', help='Path to NumPy results JSON')
    parser.add_argument('--csharp', default='NumSharp.Benchmark.GraphEngine/BenchmarkDotNet.Artifacts/results',
                       help='Path to BenchmarkDotNet artifacts directory')
    parser.add_argument('--output', default='benchmark-report', help='Output file base name (without extension)')
    parser.add_argument('--format', choices=['all', 'json', 'csv', 'md'], default='all',
                       help='Output format(s)')
    args = parser.parse_args()

    # Load results
    print("Loading NumPy results...")
    numpy_results = load_numpy_results(args.numpy)
    print(f"  Found {len(numpy_results)} NumPy results")

    print("Loading C# results...")
    csharp_results = load_csharp_results(args.csharp)
    print(f"  Found {len(csharp_results)} C# results")

    # Merge
    print("Merging results...")
    unified = merge_results(numpy_results, csharp_results)
    print(f"  Generated {len(unified)} unified results")

    # Generate outputs
    if args.format in ('all', 'json'):
        generate_json(unified, f"{args.output}.json")
    if args.format in ('all', 'csv'):
        generate_csv(unified, f"{args.output}.csv")
    if args.format in ('all', 'md'):
        generate_markdown(unified, f"{args.output}.md")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  ✅ Faster: {sum(1 for r in unified if r.status == 'faster')}")
    print(f"  🟡 Close:  {sum(1 for r in unified if r.status == 'close')}")
    print(f"  🟠 Slower: {sum(1 for r in unified if r.status == 'slower')}")
    print(f"  🔴 Much slower: {sum(1 for r in unified if r.status == 'much_slower')}")
    print(f"  ⚪ No data: {sum(1 for r in unified if r.status == 'no_data')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
