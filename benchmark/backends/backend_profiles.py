#!/usr/bin/env python3
"""Run the targeted Managed C# and OpenBLAS benchmark profiles with one JSON schema."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUTPUT_DIR = REPO / "benchmark" / "openblas"
sys.path.insert(0, str(REPO / "benchmark" / "scripts"))
import bench_common as bc  # noqa: E402

CS_BENCH = HERE / "backend_profile_bench.cs"
PY_BENCH = HERE / "backend_profile_bench.py"
MANAGED_JSON = OUTPUT_DIR / "openblas_results.managed.json"
OPENBLAS_JSON = OUTPUT_DIR / "openblas_results.openblas.json"
REPORT_MD = OUTPUT_DIR / "openblas_results.md"
EXPORT_TSV = OUTPUT_DIR / "openblas_results.tsv"

WORKLOAD_TIERS = {1_000, 100_000, 10_000_000}
PRODUCT_TIERS = WORKLOAD_TIERS
SLIDING_TIERS = WORKLOAD_TIERS
LAPACK_TIERS = WORKLOAD_TIERS

# This is a coverage CONTRACT, not a rendering hint. If either twin silently drops one of these
# rows, the benchmark run must fail instead of publishing another asymmetric backend comparison.
EXPECTED_COVERAGE = {
    ("np.dot(a, b)", ""): PRODUCT_TIERS,
    ("a.dot(b) [ndarray]", ""): PRODUCT_TIERS,
    ("np.inner(a, b)", ""): PRODUCT_TIERS,
    ("np.vdot(a, b)", ""): PRODUCT_TIERS,
    ("np.vecdot(a, b)", ""): PRODUCT_TIERS,
    ("np.linalg.vecdot(a, b)", ""): PRODUCT_TIERS,
    ("np.matmul(a, b)", ""): PRODUCT_TIERS,
    ("np.linalg.matmul(a, b)", ""): PRODUCT_TIERS,
    ("np.matvec(a, b)", ""): PRODUCT_TIERS,
    ("np.vecmat(a, b)", ""): PRODUCT_TIERS,
    ("np.einsum(subscripts, operands)", ""): PRODUCT_TIERS,
    ("np.einsum('ij,jk->ik', a, b)", "ij,jk->ik"): PRODUCT_TIERS,
    ("np.tensordot(a, b)", ""): PRODUCT_TIERS,
    ("np.tensordot(a, b, axes=1)", "axes=1,matrix"): PRODUCT_TIERS,
    ("np.linalg.tensordot(a, b)", ""): PRODUCT_TIERS,
    ("np.linalg.multi_dot(arrays)", ""): PRODUCT_TIERS,
    ("np.linalg.matrix_power(a, 3)", ""): PRODUCT_TIERS,
    ("np.correlate(a, kernel)", ""): SLIDING_TIERS,
    ("np.convolve(a, kernel)", ""): SLIDING_TIERS,
}

for operation in (
    "np.linalg.cholesky(a)", "np.linalg.cond(a)", "np.linalg.det(a)",
    "np.linalg.eig(a)", "np.linalg.eigh(a)", "np.linalg.eigvals(a)",
    "np.linalg.eigvalsh(a)", "np.linalg.inv(a)", "np.linalg.lstsq(a, b)",
    "np.linalg.matrix_rank(a)", "np.linalg.pinv(a)", "np.linalg.qr(a)",
    "np.linalg.slogdet(a)", "np.linalg.solve(a, b)", "np.linalg.svd(a)",
    "np.linalg.svdvals(a)", "np.linalg.tensorinv(a)", "np.linalg.tensorsolve(a, b)",
):
    EXPECTED_COVERAGE[(operation, "")] = LAPACK_TIERS

EXPECTED_COVERAGE[("np.linalg.matrix_power(a, -1)", "n=-1")] = LAPACK_TIERS
EXPECTED_COVERAGE[("np.linalg.matrix_norm(a, ord=2)", "ord=2")] = LAPACK_TIERS
EXPECTED_COVERAGE[("np.linalg.norm(a, ord=2)", "ord=2")] = LAPACK_TIERS
EXPECTED_COVERAGE[("np.polyfit(x, y, 3)", "")] = WORKLOAD_TIERS
EXPECTED_COVERAGE[("np.roots(p)", "")] = WORKLOAD_TIERS


def json_lines(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def profile_result(profile: str, cs: dict, numpy_ms: float | None) -> dict:
    availability = cs.get("availability", "not_measured")
    numsharp_ms = cs.get("numsharp_ms")
    ratio = None
    pct_numpy = None
    status = "no_data"
    if availability == "available" and numpy_ms is not None and numsharp_ms is not None:
        if numpy_ms <= 0 or numsharp_ms <= 0:
            # The harness completed, but overhead subtraction rounded the duration to zero.
            # This is below measurement resolution, not an unavailable backend.
            status = "negligible"
        else:
            ratio = numpy_ms / numsharp_ms
            pct_numpy = numsharp_ms / numpy_ms * 100
    if ratio is not None:
        if numpy_ms < 0.001 or numsharp_ms < 0.001 or ratio > 20:
            status = "negligible"
        elif ratio >= 1.05:
            status = "faster"
        elif ratio >= 0.8:
            status = "close"
        elif ratio >= 0.33:
            status = "slower"
        else:
            status = "much_slower"
    return {
        "profile": profile,
        "actual_backend": cs.get("actual_backend"),
        "availability": availability,
        "numsharp_ms": round(numsharp_ms, 6) if numsharp_ms is not None else None,
        "ratio": round(ratio, 6) if ratio is not None else None,
        "pct_numpy": round(pct_numpy, 3) if pct_numpy is not None else None,
        "status": status,
        "exception_type": cs.get("exception_type"),
        "exception_message": cs.get("exception_message"),
    }


def build_envelope(profile: str, cs_rows: list[dict], numpy_by_key: dict[str, dict]) -> dict:
    rows = []
    for cs in cs_rows:
        numpy = numpy_by_key.get(cs["key"], {})
        numpy_ms = numpy.get("numpy_ms")
        rows.append({
            "operation": cs["operation"],
            "suite": cs["suite"],
            "category": cs["category"],
            "dtype": cs["dtype"],
            "n": cs["n"],
            "scenario": cs.get("scenario", ""),
            "numpy_ms": round(numpy_ms, 6) if numpy_ms is not None else None,
            "backend_route": cs["backend_route"],
            "result": profile_result(profile, cs, numpy_ms),
        })
    counts: dict[str, int] = {}
    for row in rows:
        availability = row["result"]["availability"]
        counts[availability] = counts.get(availability, 0) + 1
    return {
        "schema_version": 2,
        "profile": profile,
        "backend_info": next((row.get("backend_info") for row in cs_rows if row.get("backend_info")), None),
        "availability_counts": counts,
        "rows": sorted(rows, key=lambda row: (row["suite"], row["operation"], row["n"], row["scenario"])),
    }


def validate_coverage(managed: dict, openblas: dict) -> None:
    """Reject asymmetric or incomplete backend-profile output before it reaches the dashboard."""
    def identities(envelope: dict) -> set[tuple[str, str, int, str]]:
        return {(row["operation"], row["dtype"], row["n"], row["scenario"])
                for row in envelope["rows"]}

    managed_ids = identities(managed)
    openblas_ids = identities(openblas)
    if managed_ids != openblas_ids:
        only_managed = sorted(managed_ids - openblas_ids)[:10]
        only_openblas = sorted(openblas_ids - managed_ids)[:10]
        raise RuntimeError(
            "Backend profile twins emitted different cells: "
            f"managed-only={only_managed}; openblas-only={only_openblas}")

    sizes_by_case: dict[tuple[str, str], set[int]] = {}
    for operation, _dtype, n, scenario in managed_ids:
        sizes_by_case.setdefault((operation, scenario), set()).add(n)

    missing = []
    for case, expected_sizes in EXPECTED_COVERAGE.items():
        absent = expected_sizes - sizes_by_case.get(case, set())
        if absent:
            missing.append((case[0], case[1], sorted(absent)))
    if missing:
        raise RuntimeError(f"Backend profile coverage contract failed: {missing}")


def render(managed: dict, openblas: dict, backend_info: str) -> None:
    managed_by_key = {(r["operation"], r["dtype"], r["n"], r["scenario"]): r for r in managed["rows"]}
    openblas_by_key = {(r["operation"], r["dtype"], r["n"], r["scenario"]): r for r in openblas["rows"]}
    keys = sorted(set(managed_by_key) | set(openblas_by_key))
    lines = [
        "# Backend profiles — Managed C# and OpenBLAS", "",
        "Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.",
        "`missing_backend` and `not_supported` are availability outcomes, never timed failures.", "",
        f"OpenBLAS backend: `{backend_info or 'unavailable'}`", "",
        "| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    with EXPORT_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["profile", "operation", "dtype", "n", "scenario", "availability", "ns_ms", "np_ms", "ratio", "exception_type"])
        for profile, envelope in (("managed", managed), ("openblas", openblas)):
            for row in envelope["rows"]:
                result = row["result"]
                writer.writerow([profile, row["operation"], row["dtype"], row["n"], row["scenario"],
                                 result["availability"], result["numsharp_ms"], row["numpy_ms"],
                                 result["ratio"], result["exception_type"]])

    def result_text(row: dict | None) -> str:
        if not row:
            return "not_measured"
        result = row["result"]
        return f"{result['numsharp_ms']:.4f} ms" if result["availability"] == "available" else result["availability"]

    for key in keys:
        m = managed_by_key.get(key)
        o = openblas_by_key.get(key)
        source = o or m
        valid = [row for row in (m, o) if row and row["result"]["availability"] == "available"]
        best_ratio = max((row["result"]["ratio"] for row in valid if row["result"]["ratio"] is not None), default=None)
        lines.append(f"| `{key[0]}` | {key[2]:,} | {result_text(m)} | {result_text(o)} | "
                     f"{source['numpy_ms']:.4f} | {'—' if best_ratio is None else f'{best_ratio:.3f}x'} |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        bc.build_openblas(str(REPO))

    numpy_text = bc.run_py(str(REPO), str(PY_BENCH), timeout=1200,
                           env_extra={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    numpy_rows = json_lines(numpy_text)
    numpy_by_key = {row["key"]: row for row in numpy_rows}

    managed_text = bc.run_cs(str(REPO), str(CS_BENCH), timeout=1800, env_extra={
        "NUMSHARP_BENCHMARK_PROFILE": "managed",
        "NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL": "0",
    })
    openblas_text = bc.run_cs(str(REPO), str(CS_BENCH), timeout=1800, env_extra={
        "NUMSHARP_BENCHMARK_PROFILE": "openblas",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    })
    managed = build_envelope("managed", json_lines(managed_text), numpy_by_key)
    openblas = build_envelope("openblas", json_lines(openblas_text), numpy_by_key)
    validate_coverage(managed, openblas)
    MANAGED_JSON.write_text(json.dumps(managed, indent=2), encoding="utf-8")
    OPENBLAS_JSON.write_text(json.dumps(openblas, indent=2), encoding="utf-8")
    backend_info = openblas.get("backend_info") or ""
    render(managed, openblas, backend_info)
    bc.log(f"[backends] managed={len(managed['rows'])} openblas={len(openblas['rows'])} -> {OUTPUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
