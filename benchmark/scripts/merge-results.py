#!/usr/bin/env python3
"""
Merge NumPy and NumSharp benchmark results into a unified comparison table.

Timing basis: each side's BEST WINDOW — the min of its >=50 recorded samples per case (>=20 when a
round exceeds 10 ms) — not the mean. Interference (GC pauses, page-fault storms, ambient machine
load) only ever adds time, and it lands almost entirely in the NumSharp side's right tail, so
comparing means turned machine state into fake ratio regressions. Per-case means are retained as
`numpy_mean_ms`/`numsharp_mean_ms` for tail diagnostics.

Outputs:
  - benchmark-report.json: Machine-readable merged data with ratios
  - benchmark-report.md: Markdown table for documentation
  - benchmark-report.csv: CSV for spreadsheet analysis

Usage:
  python merge-results.py
  python merge-results.py --numpy ../benchmark-report.json --csharp ../NumSharp.Benchmark.CSharp/BenchmarkDotNet.Artifacts/results/
  python merge-results.py --format csv

Note: This script is typically invoked from run-benchmarks.ps1 with explicit paths.
"""

import json
import os
import sys
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

UNIVERSAL_TIERS = {1_000, 100_000, 10_000_000}

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
    status: str  # "faster", "close", "slower", "much_slower", "negligible", "no_data", "failed"
    # Timing basis: numpy_ms/numsharp_ms above are each side's BEST WINDOW (min of its >=50
    # samples; >=20 when a round exceeds 10 ms). The per-case arithmetic means are retained
    # below as provenance — a large mean/min gap on one side flags interference (GC pauses,
    # page-fault storms, machine load) during that case's measurement window.
    numpy_mean_ms: Optional[float] = None
    numsharp_mean_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_numpy_results(path: str) -> List[dict]:
    """Load NumPy benchmark results from JSON."""
    if not os.path.exists(path):
        print(f"Warning: NumPy results not found at {path}")
        return []
    with open(path, 'r') as f:
        return json.load(f)


def load_csharp_results(artifacts_dir: str) -> Tuple[List[dict], List[dict]]:
    """Load BenchmarkDotNet results from artifacts directory.

    Returns ``(results, failures)``. ``failures`` are benchmarks that RAN but produced
    no measurement — BenchmarkDotNet writes ``"Statistics": null`` when a benchmark
    threw (OutOfMemory / exception) during its workload. Those are collected separately
    so the caller can surface them (a crash is NOT the same as a benchmark that was
    never run, and must never be silently dropped into a ⚪ "not run" cell)."""
    results = []
    failures: List[dict] = []
    if not os.path.exists(artifacts_dir):
        print(f"Warning: C# artifacts not found at {artifacts_dir}")
        return results, failures

    # Find all *-report*.json files (including -full-compressed.json)
    for pattern in ["*-report.json", "*-report-full-compressed.json"]:
        full_pattern = os.path.join(artifacts_dir, pattern)
        for json_file in glob.glob(full_pattern):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'Benchmarks' in data:
                        for bench in data['Benchmarks']:
                            result = parse_bdn_benchmark(bench, failures)
                            if result:
                                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to parse {json_file}: {e}")

    return results, failures


def validate_universal_tier_coverage(rows: List[dict], source: str) -> None:
    """Reject the asymmetric 1K+100K-without-10M pattern in an official run.

    A benchmark may deliberately publish only a scalar/1K tier, but once it publishes both Small
    and Medium it is part of the universal cache-tier matrix and must also schedule Large. C#
    failures are passed in as identity rows too, so a scheduled 10M crash remains a loud failure
    rather than being misdiagnosed as a missing configuration.
    """
    groups: Dict[Tuple[str, str], set[int]] = {}
    display: Dict[Tuple[str, str], str] = {}
    for row in rows:
        raw_name = str(row.get('name') or row.get('operation') or '')
        dtype = str(row.get('dtype') or '').lower()
        n = row.get('n')
        if not raw_name or not dtype or n is None:
            continue
        identity = str(row.get('coverage_id') or
                       f"{row.get('suite', '')}|{row.get('category', '')}|{raw_name}")
        key = (identity, dtype)
        groups.setdefault(key, set()).add(int(n))
        display.setdefault(key, raw_name)

    missing = sorted(
        (display[key], key[1])
        for key, tiers in groups.items()
        if {1_000, 100_000}.issubset(tiers) and 10_000_000 not in tiers
    )
    if missing:
        preview = ', '.join(f"{name} ({dtype})" for name, dtype in missing[:25])
        suffix = '' if len(missing) <= 25 else f" (+{len(missing) - 25} more)"
        raise RuntimeError(
            f"{source} violates universal tier coverage: {len(missing)} operation/dtype group(s) "
            f"have 1K and 100K but no 10M row: {preview}{suffix}")


def parse_bdn_benchmark(bench: dict, failures: Optional[list] = None) -> Optional[dict]:
    """Parse a single BenchmarkDotNet benchmark result.

    A benchmark that CRASHED during its workload (OutOfMemory / exception) is emitted by
    BenchmarkDotNet with ``"Statistics": null`` — it RAN but produced no timing. Detect that
    explicitly and, when a ``failures`` list is supplied, record the benchmark's identity
    (ident / op / dtype / N) so the caller can surface it. This replaces the old behaviour
    where the null value tripped the generic ``except`` with a cryptic
    "'NoneType' object has no attribute 'get'" warning and the row was silently dropped —
    which then masqueraded as a ⚪ "C# benchmark not run" cell, indistinguishable from a
    benchmark that was genuinely never run."""
    try:
        method = bench.get('Method', '')
        method_title = bench.get('MethodTitle', method)
        params = bench.get('Parameters', '')

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

        # A crashed / OOM'd benchmark: BenchmarkDotNet writes "Statistics": null (the workload
        # threw, so there is no measurement). Record it as a FAILURE — a benchmark that RAN and
        # failed is a real error, not a missing benchmark — and drop the row. The identity is
        # captured here (op/dtype/N + BDN FullName) because the null Statistics carries no timing.
        stats = bench.get('Statistics')
        if not stats:
            if failures is not None:
                failures.append({
                    'ident': bench.get('FullName') or f"{operation} [{params}]",
                    'coverage_id': f"{bench.get('Type', '')}|{operation}",
                    'name': operation,
                    'dtype': dtype,
                    'n': n,
                })
            return None

        # Keep ALL sizes (Small/Medium/Large) — the comparison is per-(op, dtype, N).
        # (Historically this dropped everything but N=10M, collapsing the report to a
        # single size; the 3-size matrix requires every parameterized N to flow through.)

        # Convert nanoseconds to milliseconds
        mean_ns = stats.get('Mean', 0)
        mean_ms = mean_ns / 1_000_000

        stddev_ns = stats.get('StandardDeviation', stats.get('StdDev', 0))
        stddev_ms = stddev_ns / 1_000_000

        # Best window (min of the >=50 BenchmarkDotNet iterations) — the statistic the report
        # publishes (see merge_results): interference only ever ADDS time, so the fastest window
        # is the closest estimate of the op's intrinsic cost. The mean rides along as provenance.
        min_ns = stats.get('Min', 0)
        min_ms = min_ns / 1_000_000

        return {
            'coverage_id': f"{bench.get('Type', '')}|{operation}",
            'name': operation,
            'method': method,
            'dtype': dtype,
            'n': n,
            'mean_ms': mean_ms,
            'min_ms': min_ms,
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
    if ratio >= 1.05:
        return "faster"
    if ratio >= 0.8:
        return "close"
    if ratio >= 0.33:
        return "slower"
    return "much_slower"


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
    """Share of NumPy's time NumSharp uses — always a percentage (e.g. 88000% = 880× as long)."""
    if pct is None:
        return "-"
    return f"{pct:.0f}%"


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
        "no_data": "⚪",
        "failed": "❌"      # C# benchmark RAN but crashed/OOM'd (no measurement) — not "not run"
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

    # Alias passes so a measured C# op JOINS its NumPy counterpart instead of being discarded as
    # "C#-only" — each recovers ⚪ "C# benchmark not run" cells the merge was silently dropping:
    #   * empty "()" left by a no-arg method call must go: C# "a.flatten()" -> "a.flatten" meets NumPy's "a.flatten".
    #   * spacing around "->": C# "reshape 2d -> 1d" meets NumPy's "reshape 2d->1d".
    # np.around and np.round are intentionally kept distinct: both are public API entry points and
    # the coverage-complete matrix benchmarks each call surface explicitly.
    name = re.sub(r'\(\s*\)', '', name)
    name = re.sub(r'\s*->\s*', '->', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def merge_results(numpy_results: List[dict], csharp_results: List[dict],
                  failures: Optional[List[dict]] = None) -> List[UnifiedResult]:
    """Merge NumPy and C# results into unified comparison.

    ``failures`` are C# benchmarks that RAN but crashed (no Statistics — see
    load_csharp_results). A NumPy row whose matching C# cell crashed is marked
    ``failed`` (❌) rather than ``no_data`` (⚪), so the report distinguishes a
    benchmark that errored from one that was never run."""
    unified = []

    # Index the crashed C# benchmarks by the SAME (op, dtype, N) key the join uses, so a
    # NumPy row that lines up with a crash is labelled "failed" instead of "not run".
    failed_index = set()
    for f in (failures or []):
        failed_index.add((normalize_op_name(f['name']), f['dtype'].lower(), f['n']))

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
        # Timing basis = best window (min): both harnesses record >=50 samples per case (>=20 when
        # a round exceeds 10 ms), and interference only ever ADDS time, so each side's fastest
        # window is the closest estimate of intrinsic cost. The mean is fat-tailed on the NumSharp
        # side under GC/page-fault/machine interference (measured: mean/min p75 = 1.68 vs NumPy's
        # 1.08 in a full run — enough to move the credible-row geomean 0.86 -> 0.99), so comparing
        # means turned machine state into fake ratio regressions. The `or` fallback keeps legacy
        # result JSONs without min fields mergeable.
        numpy_mean_ms = np_result.get('mean_ms', 0)
        numpy_ms = np_result.get('min_ms') or numpy_mean_ms

        # Look for matching C# result at the SAME size (op, dtype, N)
        norm_name = normalize_op_name(name)
        key = (norm_name, dtype.lower(), n)
        cs_result = csharp_index.get(key)

        numsharp_mean_ms = cs_result['mean_ms'] if cs_result else None
        numsharp_ms = (cs_result.get('min_ms') or numsharp_mean_ms) if cs_result else None
        ratio = numpy_ms / numsharp_ms if (numsharp_ms and numsharp_ms > 0) else None         # NP/NS, >1 = faster
        pct = numsharp_ms / numpy_ms * 100 if (numsharp_ms is not None and numpy_ms > 0) else None  # share of NumPy time
        status = classify(numpy_ms, numsharp_ms, ratio)
        # A crashed C# cell has no result (status == no_data), but it is a FAILURE, not a
        # never-run cell — relabel so ❌ is never rendered as ⚪ "not run".
        if status == "no_data" and key in failed_index:
            status = "failed"

        unified.append(UnifiedResult(
            operation=name,
            suite=suite,
            category=category,
            dtype=dtype,
            n=n,
            # 6 decimals (ns resolution): the adaptive 250x-batched averages of O(1)/scalar/
            # view ops land in the sub-µs range and must survive rounding — at 4 decimals a
            # sub-100ns op collapsed to 0.0000 and its ratio became undefined ("—").
            numpy_ms=round(numpy_ms, 6),
            numsharp_ms=round(numsharp_ms, 6) if numsharp_ms is not None else None,
            ratio=round(ratio, 3) if ratio is not None else None,
            pct_numpy=round(pct, 1) if pct is not None else None,
            status=status,
            numpy_mean_ms=round(numpy_mean_ms, 6) if numpy_mean_ms is not None else None,
            numsharp_mean_ms=round(numsharp_mean_ms, 6) if numsharp_mean_ms is not None else None
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
                        'NumPy (ms)', 'NumSharp (ms)', 'Ratio (NumPy/NumSharp)', '%NumPy', 'Status',
                        'NumPy mean (ms)', 'NumSharp mean (ms)'])
        for r in results:
            writer.writerow([
                r.operation, r.suite, r.category, r.dtype, r.n,
                r.numpy_ms,
                '' if r.numsharp_ms is None else r.numsharp_ms,
                '' if r.ratio is None else r.ratio,
                '' if r.pct_numpy is None else r.pct_numpy,
                r.status,
                '' if r.numpy_mean_ms is None else r.numpy_mean_ms,
                '' if r.numsharp_mean_ms is None else r.numsharp_mean_ms
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
    failed = sum(1 for r in results if r.status == 'failed')
    total = len(results)

    lines = [
        "# NumSharp vs NumPy Performance",
        "",
        "**Baseline:** NumPy · measured across all array sizes (per-(op, dtype, N))",
        "",
        "**Timing basis:** best window (min) of each side's per-case sweep. Min-based harnesses (NumPy "
        "op-matrix, subsystems, backends) run each case to a ~200 ms time budget — a >20 ms/call op runs "
        "exactly 100 times — while the C# op-matrix keeps BenchmarkDotNet's 50 iterations; interference "
        "only adds time, so the fastest window estimates intrinsic cost; per-case means are retained in "
        "the JSON (`numpy_mean_ms`/`numsharp_mean_ms`) as tail diagnostics.",
        "",
        "**Ratio** = NumPy ÷ NumSharp → Higher is better (>1.0× = NumSharp faster)",
        "",
        "**%NumPy🕐** = NumSharp ÷ NumPy × 100 = the share of NumPy's time NumSharp uses "
        "(30% = NumSharp takes only 30% of the time NumPy would; <100% = faster).",
        "",
        "| | Status | Ratio | %NumPy🕐 | Meaning |",
        "|:-:|--------|:-----:|:------:|---------|",
        "|✅| Faster | ≥1.0× | ≤100% | NumSharp ≥ NumPy speed |",
        "|🟡| Close | 0.5–1.0× | 100–200% | within 2× slower |",
        "|🟠| Slower | 0.2–0.5× | 200–500% | optimization target |",
        "|🔴| Slow | <0.2× | >500% | priority fix |",
        "|▫| Negligible | <1µs / >20× | — | too fast to compare — excluded from rankings |",
        "|⚪| Pending | - | — | C# benchmark not run |",
        "|❌| Failed | - | — | C# benchmark ran but crashed/OOM'd (no measurement) |",
        "",
        "---",
        "",
        f"**Summary:** {total} ops | ✅ {faster} | 🟡 {close} | 🟠 {slower} | 🔴 {much_slower} | ▫ {negligible} | ⚪ {no_data}"
        + (f" | ❌ {failed}" if failed else ""),
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
    # The ❌ fail column only appears when a benchmark crashed, so an ordinary clean run keeps
    # the historical table shape (no permanent column of zeros).
    fail_hdr = " ❌ fail |" if failed else ""
    fail_sep = "-----:|" if failed else ""
    lines.append("| N | ops | ✅ faster | 🟡 close | 🟠 slower | 🔴 much | ▫ negl | ⚪ n/a |" + fail_hdr + " geomean | %NP🕐 |")
    lines.append("|---:|----:|--------:|--------:|---------:|------:|-----:|-----:|" + fail_sep + "--------:|------:|")
    for n in sizes:
        rs = [r for r in results if r.n == n]
        gz = _geo([r.ratio for r in rs if r.status in CREDIBLE])   # credible rows only, NP/NS
        gz_s = f"{gz:.2f}x" if gz else "-"
        pz_s = pct_fmt(100.0 / gz) if gz else "-"
        fail_cell = f" {sum(1 for r in rs if r.status == 'failed')} |" if failed else ""
        lines.append(
            f"| {n:,} | {len(rs)} "
            f"| {sum(1 for r in rs if r.status == 'faster')} "
            f"| {sum(1 for r in rs if r.status == 'close')} "
            f"| {sum(1 for r in rs if r.status == 'slower')} "
            f"| {sum(1 for r in rs if r.status == 'much_slower')} "
            f"| {sum(1 for r in rs if r.status == 'negligible')} "
            f"| {sum(1 for r in rs if r.status == 'no_data')} |" + fail_cell + f" {gz_s} | {pz_s} |")
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
                f"%NumPy🕐 = share of NumPy's time NumSharp uses._")
        cols = "| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio | %NumPy🕐 |"
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
        lines.append("| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio | %NumPy🕐 |")
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
    parser.add_argument('--csharp', default='NumSharp.Benchmark.CSharp/BenchmarkDotNet.Artifacts/results',
                       help='Path to BenchmarkDotNet artifacts directory')
    parser.add_argument('--output', default='benchmark-report', help='Output file base name (without extension)')
    parser.add_argument('--format', choices=['all', 'json', 'csv', 'md'], default='all',
                       help='Output format(s)')
    parser.add_argument('--require-universal-tiers', action='store_true',
                        help='Fail when an operation/dtype has 1K and 100K rows but no 10M row')
    args = parser.parse_args()

    # Load results
    print("Loading NumPy results...")
    numpy_results = load_numpy_results(args.numpy)
    print(f"  Found {len(numpy_results)} NumPy results")

    print("Loading C# results...")
    csharp_results, csharp_failures = load_csharp_results(args.csharp)
    print(f"  Found {len(csharp_results)} C# results")

    if args.require_universal_tiers:
        validate_universal_tier_coverage(numpy_results, "NumPy profile")
        validate_universal_tier_coverage(
            csharp_results + [
                {'coverage_id': row.get('coverage_id'), 'name': row['name'],
                 'dtype': row['dtype'], 'n': row['n']}
                for row in csharp_failures
            ],
            "C# profile")

    # Surface crashed benchmarks LOUDLY. BenchmarkDotNet emits "Statistics": null for a
    # benchmark whose workload threw (OutOfMemory / exception); those must never be silently
    # dropped into a ⚪ "not run" cell — they are real errors and mask genuine coverage holes.
    if csharp_failures:
        print(f"\n  ❌ {len(csharp_failures)} C# benchmark(s) RAN BUT FAILED to measure "
              f"(no Statistics — OOM/crash; NOT the same as 'not run'):")
        for f in csharp_failures:
            print(f"     ✗ {f['ident']}")
        print()

    # Merge
    print("Merging results...")
    unified = merge_results(numpy_results, csharp_results, csharp_failures)
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
    print(f"  ⚪ No data (C# not run): {sum(1 for r in unified if r.status == 'no_data')}")
    failed_n = sum(1 for r in unified if r.status == 'failed')
    if failed_n:
        print(f"  ❌ Failed (C# ran but crashed/OOM'd): {failed_n}")
    print("=" * 60)


if __name__ == '__main__':
    main()
