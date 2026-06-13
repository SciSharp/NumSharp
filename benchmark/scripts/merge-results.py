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
    ratio: Optional[float]  # NumPy / NumSharp  (>1.0× = NumSharp faster)
    pct_numpy: Optional[float]  # NumSharp/NumPy × 100 = share of NumPy's time NumSharp uses
    status: str  # "faster", "close", "slower", "much_slower", "negligible", "no_data"

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

        # Keep ALL sizes (Small/Medium/Large) — the comparison is per-(op, dtype, N).
        # (Historically this dropped everything but N=10M, collapsing the report to a
        # single size; the 3-size matrix requires every parameterized N to flow through.)

        # Convert nanoseconds to milliseconds
        mean_ns = stats.get('Mean', 0)
        mean_ms = mean_ns / 1_000_000

        stddev_ns = stats.get('StandardDeviation', stats.get('StdDev', 0))
        stddev_ms = stddev_ns / 1_000_000

        # Map dtype to numpy names
        dtype_map = {
            'int32': 'int32', 'int64': 'int64', 'single': 'float32', 'double': 'float64',
            'byte': 'uint8', 'uint16': 'uint16', 'uint32': 'uint32', 'uint64': 'uint64',
            'int16': 'int16', 'boolean': 'bool', 'decimal': 'decimal',
            'sbyte': 'int8', 'half': 'float16', 'complex': 'complex128'
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
    """Status band from ratio = NumPy ÷ NumSharp (>1.0× = NumSharp faster)."""
    if ratio is None:
        return "no_data"
    if ratio >= 1.0:
        return "faster"          # NumSharp ≥ NumPy speed
    if ratio >= 0.5:
        return "close"           # within 2× slower
    if ratio >= 0.2:
        return "slower"          # 2–5× slower
    return "much_slower"         # >5× slower


# A row is only a CREDIBLE throughput comparison when BOTH sides did measurable work.
# Sub-microsecond timings are dominated by call overhead — e.g. the historical
# np.searchsorted benchmark issued a SINGLE scalar binary search (~18ns at every N), so
# comparing it to NumPy's ~1µs Python overhead manufactured a meaningless 50–1000x "win".
# An implausible >20x speedup likewise almost always means the NumSharp side did ~no work
# (a view return, a lazy allocation, or a dead-code-eliminated kernel — or a one-off bad
# reading). Such rows are marked "negligible": kept OUT of the Best/Worst rankings and the
# geomean, but still listed in the per-suite tables — never showcased as a win.
WORK_FLOOR_MS = 0.001          # 1 µs — below this an op isn't doing comparable array work
MAX_CREDIBLE_SPEEDUP = 20.0    # ratio > 20 ⇒ "NumSharp >20x faster" ⇒ artifact, not a win
CREDIBLE = ("faster", "close", "slower", "much_slower")


def classify(numpy_ms: float, numsharp_ms: Optional[float], ratio: Optional[float]) -> str:
    """Status that also gates credibility (see WORK_FLOOR_MS / MAX_CREDIBLE_SPEEDUP)."""
    if numsharp_ms is None or ratio is None:
        return "no_data"
    if (numpy_ms < WORK_FLOOR_MS or numsharp_ms < WORK_FLOOR_MS
            or ratio > MAX_CREDIBLE_SPEEDUP):
        return "negligible"
    return get_status(ratio)


def pct_fmt(pct: Optional[float]) -> str:
    """Share of NumPy's time NumSharp uses; compact for the rare huge slowdowns."""
    if pct is None:
        return "-"
    return f"{pct:.0f}%" if pct < 1000 else f"{pct / 100:.0f}×"


def ratio_fmt(r: Optional[float]) -> str:
    if r is None:
        return "-"
    return f"{r:.2f}×" if r >= 0.1 else f"{r:.3f}×"


def get_status_icon(status: str) -> str:
    """Get status icon for markdown."""
    icons = {
        "faster": "✅",
        "close": "🟡",
        "slower": "🟠",
        "much_slower": "🔴",
        "negligible": "▫",
        "no_data": "⚪"
    }
    return icons.get(status, "⚪")


def normalize_op_name(name: str) -> str:
    """Canonicalize an op name so the C# [Benchmark(Description)] and the Python suite name
    collapse to the same string. Applied identically to both sides.

    C# descriptions are verbose ("np.sum(a) [full]", "np.sum(a, axis=0) [columns]",
    "np.sqrt(a)") while the original Python suites use short names ("np.sum", "np.sum axis=0",
    "np.sqrt"). Rather than maintain a per-op mapping table, normalize structurally:
      * strip the trailing dtype tag and any "[...]" annotation,
      * fold "(a, axis=k)" / "(axis=k)" into " axis=k",
      * strip identifier-only argument lists ("(a)", "(a, b)", "(cond, a, b)") but KEEP
        numeric args ("(a, 50)", "(a, 2)") that distinguish percentile / shift / etc.
    The two np.where forms are disambiguated up front so arg-stripping doesn't collide them.
    """
    import re
    name = re.sub(r'\s*\((int32|int64|float32|float64|uint8|int16|uint16|uint32|uint64|bool|decimal)\)\s*$', '', name)
    name = name.strip("'\"")
    name = re.sub(r'\s+', ' ', name).lower()

    # Disambiguate the two where ops before arg-stripping would collapse both to "np.where".
    pre = {
        'np.where(cond, a, b)': 'np.where ternary',
        'np.where(cond)': 'np.where nonzero',
    }
    name = pre.get(name, name)

    # Strip a space-separated " [annotation]" ([full]/[method]/[columns]/[asarray equivalent]/…)
    # but NOT array-indexing brackets attached to an identifier ("a[100:1000]", "a[::2]"): those
    # are part of the op identity. Stripping them collapsed the Slicing-suite "np.copy(a[100:1000])"
    # (a 900-element slice copy, ~3.6µs at every N) onto the Creation "np.copy(a)" key, where it
    # overwrote the real full-array measurement (the bogus "copy float64 = 0.0036ms").
    name = re.sub(r'\s+\[[^\]]*\]', '', name)
    name = re.sub(r'\(\s*(?:[a-z_][a-z0-9_]*\s*,\s*)?axis\s*=\s*(\d+)\s*\)', r' axis=\1', name)  # (a, axis=0) -> axis=0
    name = re.sub(r'\(\s*[a-z_][a-z0-9_]*(?:\s*,\s*[a-z_][a-z0-9_]*)*\s*\)', '', name)           # strip ident-only arg lists
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def merge_results(numpy_results: List[dict], csharp_results: List[dict]) -> List[UnifiedResult]:
    """Merge NumPy and C# results into unified comparison."""
    unified = []

    # Index C# results by (normalized_operation, dtype)
    csharp_index: Dict[tuple, dict] = {}
    for r in csharp_results:
        norm_name = normalize_op_name(r['name'])
        key = (norm_name, r['dtype'].lower(), r['n'])
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

        # Look for matching C# result at the SAME size (op, dtype, N)
        norm_name = normalize_op_name(name)
        key = (norm_name, dtype.lower(), n)
        cs_result = csharp_index.get(key)

        numsharp_ms = cs_result['mean_ms'] if cs_result else None
        ratio = numpy_ms / numsharp_ms if (numsharp_ms and numsharp_ms > 0) else None         # NP/NS, >1 = faster
        pct = numsharp_ms / numpy_ms * 100 if (numsharp_ms is not None and numpy_ms > 0) else None  # share of NumPy time
        status = classify(numpy_ms, numsharp_ms, ratio)

        unified.append(UnifiedResult(
            operation=name,
            suite=suite,
            category=category,
            dtype=dtype,
            n=n,
            numpy_ms=round(numpy_ms, 4),
            numsharp_ms=round(numsharp_ms, 4) if numsharp_ms is not None else None,
            ratio=round(ratio, 3) if ratio is not None else None,
            pct_numpy=round(pct, 1) if pct is not None else None,
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
                        'NumPy (ms)', 'NumSharp (ms)', 'Ratio (NumPy/NumSharp)', '%NumPy', 'Status'])
        for r in results:
            writer.writerow([
                r.operation, r.suite, r.category, r.dtype, r.n,
                r.numpy_ms,
                '' if r.numsharp_ms is None else r.numsharp_ms,
                '' if r.ratio is None else r.ratio,
                '' if r.pct_numpy is None else r.pct_numpy,
                r.status
            ])
    print(f"CSV written to: {output_path}")


def generate_markdown(results: List[UnifiedResult], output_path: str):
    """Generate concise Markdown comparison matrix."""

    # Count stats
    faster = sum(1 for r in results if r.status == 'faster')
    close = sum(1 for r in results if r.status == 'close')
    slower = sum(1 for r in results if r.status == 'slower')
    much_slower = sum(1 for r in results if r.status == 'much_slower')
    negligible = sum(1 for r in results if r.status == 'negligible')
    no_data = sum(1 for r in results if r.status == 'no_data')
    total = len(results)

    lines = [
        "# NumSharp vs NumPy Performance",
        "",
        "**Baseline:** NumPy · measured across all array sizes (per-(op, dtype, N))",
        "",
        "**Ratio** = NumPy ÷ NumSharp → Higher is better (>1.0× = NumSharp faster)",
        "",
        "**🕐 %NumPy** = NumSharp ÷ NumPy × 100 = the share of NumPy's time NumSharp uses "
        "(30% = NumSharp takes only 30% of the time NumPy would; <100% = faster).",
        "",
        "| | Status | Ratio | 🕐 %NumPy | Meaning |",
        "|:-:|--------|:-----:|:------:|---------|",
        "|✅| Faster | ≥1.0× | ≤100% | NumSharp ≥ NumPy speed |",
        "|🟡| Close | 0.5–1.0× | 100–200% | within 2× slower |",
        "|🟠| Slower | 0.2–0.5× | 200–500% | optimization target |",
        "|🔴| Slow | <0.2× | >500% | priority fix |",
        "|▫| Negligible | <1µs / >20× | — | too fast to compare — excluded from rankings |",
        "|⚪| Pending | - | — | C# benchmark not run |",
        "",
        "---",
        "",
        f"**Summary:** {total} ops | ✅ {faster} | 🟡 {close} | 🟠 {slower} | 🔴 {much_slower} | ▫ {negligible} | ⚪ {no_data}",
        "",
    ]

    # Per-size headline: geomean ratio (NumSharp/NumPy) across all matched ops at each N,
    # plus the status histogram. This is the "all ops at 3 sizes" summary.
    import math
    sizes = sorted({r.n for r in results})

    def _geo(vals):
        vals = [v for v in vals if v and v > 0]
        return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None

    lines.append("## Summary by size")
    lines.append("")
    lines.append("| N | ops | ✅ faster | 🟡 close | 🟠 slower | 🔴 much | ▫ negl | ⚪ n/a | geomean | 🕐 %NP |")
    lines.append("|---:|----:|--------:|--------:|---------:|------:|-----:|-----:|--------:|------:|")
    for n in sizes:
        rs = [r for r in results if r.n == n]
        gz = _geo([r.ratio for r in rs if r.status in CREDIBLE])   # credible rows only, NP/NS
        gz_s = f"{gz:.2f}x" if gz else "-"
        pz_s = pct_fmt(100.0 / gz) if gz else "-"
        lines.append(
            f"| {n:,} | {len(rs)} "
            f"| {sum(1 for r in rs if r.status == 'faster')} "
            f"| {sum(1 for r in rs if r.status == 'close')} "
            f"| {sum(1 for r in rs if r.status == 'slower')} "
            f"| {sum(1 for r in rs if r.status == 'much_slower')} "
            f"| {sum(1 for r in rs if r.status == 'negligible')} "
            f"| {sum(1 for r in rs if r.status == 'no_data')} | {gz_s} | {pz_s} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Best/Worst showcase ranks ONLY credible comparisons (see classify): both sides did
    # >=1µs of work and the speedup is within a believable 20x. This keeps sub-microsecond
    # call-overhead rows and dead-code-eliminated / lazy-alloc / one-off artifacts out of
    # the headline (they used to flood "Top Best" as meaningless 0.0 / 0.0x non-results).
    with_data = [r for r in results if r.status in CREDIBLE]
    negligible_n = sum(1 for r in results if r.status == 'negligible')

    if with_data:
        # Sort by ratio (NumPy ÷ NumSharp) — best (highest = most ahead of NumPy) first
        sorted_by_ratio = sorted(with_data, key=lambda r: r.ratio, reverse=True)
        note = (f"_Ranked over {len(with_data)} credible comparisons "
                f"(both sides ≥{WORK_FLOOR_MS * 1000:.0f}µs, within {MAX_CREDIBLE_SPEEDUP:.0f}×); "
                f"{negligible_n} negligible rows excluded as non-comparable (▫). "
                f"Ratio = NumPy ÷ NumSharp — above 1.0× = NumSharp faster · "
                f"🕐 %NumPy = share of NumPy's time NumSharp uses._")
        cols = "| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio | 🕐 %NumPy |"
        sep = "|:-:|-----------|:----:|----:|----------:|-------------:|------:|--------:|"

        def trow(r):
            return (f"|{get_status_icon(r.status)}| {r.operation} | {r.dtype} | {r.n:,} "
                    f"| {r.numpy_ms:.3f} | {r.numsharp_ms:.3f} | {ratio_fmt(r.ratio)} | {pct_fmt(r.pct_numpy)} |")

        lines.append("### 🏆 Top 15 Best (NumSharp fastest vs NumPy)")
        lines += ["", note, "", cols, sep]
        lines += [trow(r) for r in sorted_by_ratio[:15]]
        lines.append("")

        lines.append("### 🔻 Top 15 Worst (Optimization priorities)")
        lines += ["", cols, sep]
        lines += [trow(r) for r in sorted_by_ratio[-15:][::-1]]   # lowest ratio = slowest, worst first
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

    # Generate per-suite table: one row per (operation, dtype, N). The N column makes the
    # 3-size comparison explicit. Sorted by op, then dtype, then size so the three sizes of
    # each op sit together.
    for suite_name, suite_results in suites.items():
        lines.append(f"### {suite_name}")
        lines.append("")
        lines.append("| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio | 🕐 %NumPy |")
        lines.append("|:-:|-----------|:----:|----:|----------:|-------------:|------:|--------:|")

        for r in sorted(suite_results, key=lambda x: (x.operation, x.dtype, x.n)):
            icon = get_status_icon(r.status)
            numsharp_str = f"{r.numsharp_ms:.4f}" if r.numsharp_ms is not None else "-"
            lines.append(f"|{icon}| {r.operation} | {r.dtype} | {r.n:,} | {r.numpy_ms:.4f} | {numsharp_str} | {ratio_fmt(r.ratio)} | {pct_fmt(r.pct_numpy)} |")

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

    # Coverage check (P3): C# benchmarks that found NO NumPy counterpart at the same
    # (op, dtype, N). Expected for NumSharp-only dtypes (char/decimal) and experimental
    # suites; anything else is a join mismatch worth fixing.
    np_keys = {(normalize_op_name(r.get('name', '')), r.get('dtype', '').lower(), r.get('n'))
               for r in numpy_results}
    cs_only = [r for r in csharp_results
               if (normalize_op_name(r['name']), r['dtype'].lower(), r['n']) not in np_keys]
    if cs_only:
        distinct = sorted({f"{normalize_op_name(r['name'])} ({r['dtype']})" for r in cs_only})
        print(f"  C#-only (no NumPy match): {len(cs_only)} cases, {len(distinct)} distinct op×dtype:")
        for nm in distinct[:50]:
            print(f"    - {nm}")

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
    print(f"  ▫  Negligible (excluded from rankings): {sum(1 for r in unified if r.status == 'negligible')}")
    print(f"  ⚪ No data: {sum(1 for r in unified if r.status == 'no_data')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
