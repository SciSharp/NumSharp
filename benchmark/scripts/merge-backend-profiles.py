#!/usr/bin/env python3
"""Merge Managed C# and OpenBLAS benchmark profiles into one canonical dataset.

Each profile remains available as its own JSON envelope, while the combined JSON stores both
profile results per exact benchmark cell and publishes one effective (fastest valid) result.
Legacy flat op-matrix JSON and the former OpenBLAS TSV are accepted so an existing snapshot can
be upgraded without inventing timings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from credibility import o1_exclusion_reason  # noqa: E402

SCHEMA_VERSION = 2
AVAILABLE = "available"
MISSING_BACKEND = "missing_backend"
NOT_SUPPORTED = "not_supported"
NOT_MEASURED = "not_measured"
FAILED = "failed"
WORK_FLOOR_MS = 0.001
MAX_CREDIBLE_SPEEDUP = 20.0

# Operations in the official LinearAlgebra suite that do not consult TensorEngine.Blas. They still
# run in the OpenBLAS PROFILE so the complete suite is symmetric, but provenance must say the actual
# implementation remained Managed Core rather than crediting an uninvolved native library.
MANAGED_CONTROL_LINEAR_ALGEBRA = {
    "np.outer", "np.diagonal", "np.einsum_path", "np.linalg.cross",
    "np.linalg.diagonal", "np.linalg.matrix_norm", "np.linalg.matrix_transpose",
    "np.linalg.norm", "np.linalg.outer", "np.linalg.trace", "np.linalg.vector_norm",
}


def report_metadata(output: Path) -> dict[str, str]:
    """Return provenance for the canonical report without requiring extra runner formats."""
    match = re.match(r"(\d{4})(\d{2})(\d{2})", output.parent.name)
    snapshot_date = f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else ""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    try:
        import numpy as np
        numpy_version = np.__version__
    except ImportError:
        numpy_version = ""
    return {
        "snapshot_date": snapshot_date,
        "commit": commit,
        "numpy_version": numpy_version,
        "memory_heavy_large_workload_n": "1000000",
        "memory_heavy_large_tier_label": "10000000",
    }


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def performance_status(numpy_ms: float | None, numsharp_ms: float | None,
                       operation: str = "") -> tuple[float | None, float | None, str]:
    if numpy_ms is None or numsharp_ms is None:
        return None, None, "no_data"
    # A completed overhead-subtracted microbenchmark can legitimately land at exactly 0 ns.
    # Preserve it as measured-but-negligible rather than pretending that the profile was absent.
    if numpy_ms <= 0 or numsharp_ms <= 0:
        return None, None, "negligible"
    ratio = numpy_ms / numsharp_ms
    pct_numpy = numsharp_ms / numpy_ms * 100 if numpy_ms > 0 else None
    if operation and o1_exclusion_reason(normalize_operation(operation)):
        return ratio, pct_numpy, "negligible"
    if numpy_ms < WORK_FLOOR_MS or numsharp_ms < WORK_FLOOR_MS or ratio > MAX_CREDIBLE_SPEEDUP:
        return ratio, pct_numpy, "negligible"
    if ratio >= 1.05:
        return ratio, pct_numpy, "faster"
    if ratio >= 0.8:
        return ratio, pct_numpy, "close"
    if ratio >= 0.33:
        return ratio, pct_numpy, "slower"
    return ratio, pct_numpy, "much_slower"


def api_name(operation: str) -> str:
    special = {
        "matrix_norm_2": "np.linalg.matrix_norm",
        "matrix_power_-1": "np.linalg.matrix_power",
        "norm_2": "np.linalg.norm",
    }
    linalg = {
        "cholesky", "cond", "det", "eig", "eigh", "eigvals", "eigvalsh", "inv",
        "lstsq", "matrix_rank", "pinv", "qr", "slogdet", "solve", "svd", "svdvals",
        "tensorinv", "tensorsolve", "multi_dot", "tensordot",
    }
    if operation in special:
        return special[operation]
    return f"np.linalg.{operation}" if operation in linalg else f"np.{operation}"


def display_operation(operation: str) -> str:
    scenarios = {
        "matrix_norm_2": "np.linalg.matrix_norm(a, ord=2)",
        "matrix_power_-1": "np.linalg.matrix_power(a, -1)",
        "norm_2": "np.linalg.norm(a, ord=2)",
    }
    return scenarios.get(operation, f"{api_name(operation)}(a)")


def normalize_operation(value: str) -> str:
    value = re.sub(r"\s*\((?:bool|u?int\d+|float\d+|complex\d+|decimal)\)\s*$", "", value, flags=re.I)
    value = value.strip("'\"")
    value = re.sub(r"\s+", " ", value).strip().lower()
    value = {
        "np.where(cond, a, b)": "np.where ternary",
        "np.where(cond)": "np.where nonzero",
        "np.sum(contiguous_slice)": "np.sum contiguous_slice",
        "np.sum(strided_slice)": "np.sum strided_slice",
        "np.sum(row_slice)": "np.sum row_slice",
        "np.sum(col_slice)": "np.sum col_slice",
        "np.diag": "np.diag view",
        "np.diag(a2d)": "np.diag view",
        "np.diag(a1d)": "np.diag construct",
        "np.transpose(a, axes)": "np.transpose axes",
    }.get(value, value)
    value = re.sub(r"\s+\[[^\]]*\]", "", value)
    value = re.sub(r"\(\s*(?:[a-z_][a-z0-9_]*\s*,\s*)?axis\s*=\s*(-?\d+)\s*\)", r" axis=\1", value)
    value = re.sub(r"\(\s*[a-z_][a-z0-9_]*(?:\s*,\s*[a-z_][a-z0-9_]*)*\s*\)", "", value)
    value = re.sub(r"\(\s*\)", "", value)
    value = re.sub(r"\s*->\s*", "->", value)
    return re.sub(r"\s+", " ", value).strip()


def cell_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        normalize_operation(str(row.get("operation", ""))),
        str(row.get("dtype", "")).lower(),
        int(row.get("n") or 0),
        str(row.get("scenario", "")),
    )


def profile_result(
    profile: str,
    numpy_ms: float | None,
    numsharp_ms: float | None,
    availability: str = AVAILABLE,
    *,
    operation: str = "",
    actual_backend: str | None = None,
    exception_type: str | None = None,
    exception_message: str | None = None,
) -> dict[str, Any]:
    ratio, pct_numpy, status = performance_status(numpy_ms, numsharp_ms, operation)
    if availability != AVAILABLE:
        ratio = pct_numpy = None
        status = "failed" if availability == FAILED else "no_data"
    return {
        "profile": profile,
        "actual_backend": actual_backend or ("managed" if profile == "managed" else "openblas"),
        "availability": availability,
        "numsharp_ms": round(numsharp_ms, 6) if numsharp_ms is not None else None,
        "ratio": round(ratio, 6) if ratio is not None else None,
        "pct_numpy": round(pct_numpy, 3) if pct_numpy is not None else None,
        "status": status,
        "exception_type": exception_type,
        "exception_message": exception_message,
    }


def base_row(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": str(raw.get("operation", "")),
        "suite": str(raw.get("suite", "General")),
        "category": str(raw.get("category", "")),
        "dtype": str(raw.get("dtype", "float64")),
        "n": int(raw.get("n") or 0),
        "scenario": str(raw.get("scenario", "")),
        "numpy_ms": number(raw.get("numpy_ms")),
        # Best-window (min) provenance from merge-results.py: the per-case MEANS ride along so a
        # large mean/min gap on one side keeps flagging a contaminated measurement window in the
        # final report JSON. Backend-profile harness rows have no mean fields -> None.
        "numpy_mean_ms": number(raw.get("numpy_mean_ms")),
        "numsharp_mean_ms": number(raw.get("numsharp_mean_ms")),
        "backend_route": str(raw.get("backend_route", "managed")),
    }


def load_json_profile(path: Path, profile: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("rows", []) if isinstance(payload, dict) else payload
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = base_row(raw)
        existing = raw.get("result") if isinstance(raw.get("result"), dict) else None
        availability = str(raw.get("availability") or (existing or {}).get("availability") or AVAILABLE)
        numsharp_ms = number(raw.get("numsharp_ms") if existing is None else existing.get("numsharp_ms"))
        source_status = str(raw.get("status") or (existing or {}).get("status") or "")
        if source_status == "failed" and numsharp_ms is None:
            availability = FAILED
        if numsharp_ms is None and availability == AVAILABLE:
            availability = NOT_MEASURED
        explicit_backend = raw.get("actual_backend") or (existing or {}).get("actual_backend")
        normalized = normalize_operation(row["operation"])
        is_managed_control = row["suite"] == "LinearAlgebra" and normalized in MANAGED_CONTROL_LINEAR_ALGEBRA
        actual_backend = str(explicit_backend or (
            "managed" if profile == "managed" or is_managed_control else "openblas"))
        if "backend_route" not in raw and existing is None:
            row["backend_route"] = "managed_control" if is_managed_control else profile
        row["result"] = profile_result(
            profile,
            row["numpy_ms"],
            numsharp_ms,
            availability,
            operation=row["operation"],
            actual_backend=actual_backend,
            exception_type=raw.get("exception_type") or (existing or {}).get("exception_type"),
            exception_message=raw.get("exception_message") or (existing or {}).get("exception_message"),
        )
        rows.append(row)
    return rows


def load_legacy_openblas_tsv(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    openblas_rows: list[dict[str, Any]] = []
    managed_availability: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            route, operation, size, dtype = raw["key"].split("|")
            numpy_ms = number(raw.get("np_ms"))
            numsharp_ms = number(raw.get("ns_ms"))
            row = {
                "operation": display_operation(operation),
                "suite": "Arithmetic" if route == "sliding" else "LinearAlgebra",
                "category": route,
                "dtype": {"f32": "float32", "f64": "float64", "c128": "complex128"}.get(dtype, dtype),
                "n": int(size),
                "scenario": ({
                    "matrix_norm_2": "ord=2",
                    "matrix_power_-1": "n=-1",
                    "norm_2": "ord=2",
                }.get(operation) or ("mode=same,kernel=31" if route == "sliding" else "")),
                "numpy_ms": numpy_ms,
                "backend_route": route,
                "result": profile_result(
                    "openblas", numpy_ms, numsharp_ms, AVAILABLE, operation=display_operation(operation)),
            }
            openblas_rows.append(row)
            if route in {"lapack", "composed", "hybrid"}:
                missing = dict(row)
                missing["result"] = profile_result(
                    "managed", numpy_ms, None, MISSING_BACKEND,
                    operation=display_operation(operation),
                    exception_type="OpenBlasMissingBackendException",
                    exception_message="Managed C# has no backend for this operation and parameter mode.",
                )
                managed_availability.append(missing)
    return openblas_rows, managed_availability


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quality = {AVAILABLE: 3, FAILED: 2, MISSING_BACKEND: 2, NOT_SUPPORTED: 2, NOT_MEASURED: 1}
    selected: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = cell_key(row)
        current = selected.get(key)
        if current is None or quality.get(row["result"]["availability"], 0) > quality.get(current["result"]["availability"], 0):
            selected[key] = row
    return list(selected.values())


def profile_envelope(profile: str, rows: list[dict[str, Any]], metadata: dict[str, str] | None = None) -> dict[str, Any]:
    rows = sorted(dedupe(rows), key=lambda row: (row["suite"], row["operation"], row["dtype"], row["n"], row["scenario"]))
    counts: dict[str, int] = {}
    for row in rows:
        availability = row["result"]["availability"]
        counts[availability] = counts.get(availability, 0) + 1
    return {"schema_version": SCHEMA_VERSION, "profile": profile, "metadata": metadata or {},
            "availability_counts": counts, "rows": rows}


def validate_linear_algebra_profile_coverage(managed: dict[str, Any], openblas: dict[str, Any]) -> None:
    """Require every measured Managed LinearAlgebra cell in the complete OpenBLAS profile."""
    if not openblas.get("rows"):
        return

    openblas_index = {cell_key(row): row for row in openblas["rows"]}
    missing = []
    for row in managed["rows"]:
        if row["suite"] != "LinearAlgebra" or row["result"]["availability"] != AVAILABLE:
            continue
        peer = openblas_index.get(cell_key(row))
        if peer is None or peer["result"]["availability"] != AVAILABLE:
            missing.append((row["operation"], row["dtype"], row["n"], row["scenario"]))

    if missing:
        preview = missing[:20]
        suffix = "" if len(missing) <= len(preview) else f" (+{len(missing) - len(preview)} more)"
        raise RuntimeError(
            "OpenBLAS profile does not cover the complete measured LinearAlgebra suite: "
            f"{preview}{suffix}")


def combine(managed: dict[str, Any], openblas: dict[str, Any]) -> dict[str, Any]:
    validate_linear_algebra_profile_coverage(managed, openblas)
    all_rows: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for profile, envelope in (("managed", managed), ("openblas", openblas)):
        for source in envelope["rows"]:
            key = cell_key(source)
            target = all_rows.setdefault(key, {
                "operation": source["operation"], "suite": source["suite"],
                "category": source["category"], "dtype": source["dtype"], "n": source["n"],
                "scenario": source["scenario"], "numpy_ms": source["numpy_ms"],
                "numpy_mean_ms": source.get("numpy_mean_ms"),
                "numsharp_mean_ms": source.get("numsharp_mean_ms"),
                "backend_route": source["backend_route"], "profiles": {},
            })
            # Keep the combined report independent of the per-profile envelopes: exact cells can
            # arrive from two benchmark runs with slightly different NumPy timing samples, and the
            # canonical row must rebase both backend ratios onto ONE NumPy baseline below.
            target["profiles"][profile] = dict(source["result"])
            if target["numpy_ms"] is None:
                target["numpy_ms"] = source["numpy_ms"]
            # Mean provenance comes from the managed op-matrix rows only (the backend harness is
            # best-of already); backfill so whichever source carries it wins, whatever the order.
            if target.get("numpy_mean_ms") is None:
                target["numpy_mean_ms"] = source.get("numpy_mean_ms")
            if target.get("numsharp_mean_ms") is None:
                target["numsharp_mean_ms"] = source.get("numsharp_mean_ms")

    combined_rows = []
    for row in all_rows.values():
        # A joined cell has one published NumPy timing (the Managed op-matrix baseline when that
        # row exists). Recompute every backend's ratio against it so Managed/OpenBLAS are directly
        # comparable and the displayed effective ratio agrees with numpy_ms / numsharp_ms.
        for result in row["profiles"].values():
            if result["availability"] != AVAILABLE or result["numsharp_ms"] is None:
                continue
            ratio, pct_numpy, status = performance_status(
                row["numpy_ms"], result["numsharp_ms"], row["operation"])
            result["ratio"] = round(ratio, 6) if ratio is not None else None
            result["pct_numpy"] = round(pct_numpy, 3) if pct_numpy is not None else None
            result["status"] = status

        valid = [result for result in row["profiles"].values()
                 if result["availability"] == AVAILABLE and result["numsharp_ms"] is not None]
        effective = min(valid, key=lambda result: result["numsharp_ms"]) if valid else None
        row["effective_profile"] = effective["profile"] if effective else None
        # Profile = process/configuration that produced the timing. Backend = implementation that
        # actually executed. They differ for controls measured in the OpenBLAS process that never
        # consult TensorEngine.Blas; winner labels must not credit OpenBLAS for managed work.
        row["effective_backend"] = effective["actual_backend"] if effective else None
        row["numsharp_ms"] = effective["numsharp_ms"] if effective else None
        row["ratio"] = effective["ratio"] if effective else None
        row["pct_numpy"] = effective["pct_numpy"] if effective else None
        row["status"] = effective["status"] if effective else (
            "failed" if any(result.get("status") == "failed" for result in row["profiles"].values())
            else "no_data")
        combined_rows.append(row)

    combined_rows.sort(key=lambda row: (row["suite"], row["operation"], row["dtype"], row["n"], row["scenario"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": managed.get("metadata") or openblas.get("metadata") or {},
        "profiles": ["managed", "openblas"],
        "selection": "fastest valid NumSharp timing per exact benchmark cell",
        "rows": combined_rows,
    }


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Operation", "Suite", "DType", "N", "Scenario", "NumPy ms",
                         "Managed ms", "Managed availability", "OpenBLAS ms", "OpenBLAS availability",
                         "Effective backend", "Effective NumSharp ms", "NPY/NS", "Status"])
        for row in payload["rows"]:
            managed = row["profiles"].get("managed", {})
            openblas = row["profiles"].get("openblas", {})
            writer.writerow([row["operation"], row["suite"], row["dtype"], row["n"], row["scenario"], row["numpy_ms"],
                             managed.get("numsharp_ms"), managed.get("availability", NOT_MEASURED),
                             openblas.get("numsharp_ms"), openblas.get("availability", NOT_MEASURED),
                             row["effective_backend"], row["numsharp_ms"], row["ratio"], row["status"]])


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    counts = {"managed": 0, "openblas": 0, "missing_backend": 0, "not_supported": 0, "failed": 0}
    for row in payload["rows"]:
        if row["effective_backend"]:
            counts[row["effective_backend"]] += 1
        for result in row["profiles"].values():
            if result["availability"] in counts:
                counts[result["availability"]] += 1
    lines = [
        "# NumSharp vs NumPy Performance — backend profiles", "",
        "One canonical cell model with separate Managed C# and OpenBLAS measurements.",
        "**Effective** selects the faster valid NumSharp timing for the exact same operation/dtype/N/scenario cell; "
        "the winner names the implementation that actually executed, not merely the process profile.", "",
        "**Timing basis:** best window (min) of each side's per-case sweep — min-based harnesses run each "
        "case to a ~200 ms time budget (a >20 ms/call op runs exactly 100 times), while the C# op-matrix "
        "keeps BenchmarkDotNet's 50 iterations; interference only adds time, so the fastest window "
        "estimates intrinsic cost; per-case means are retained in the JSON "
        "(`numpy_mean_ms`/`numsharp_mean_ms`) as tail diagnostics.", "",
        f"- Managed effective cells: **{counts['managed']:,}**",
        f"- OpenBLAS effective cells: **{counts['openblas']:,}**",
        f"- Missing backend records: **{counts['missing_backend']:,}**",
        f"- Not supported records: **{counts['not_supported']:,}**",
        f"- Failed benchmark records: **{counts['failed']:,}**", "",
        "| Operation | DType | N | Managed C# | OpenBLAS | Effective | NPY/NS |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in payload["rows"]:
        managed = row["profiles"].get("managed", {})
        openblas = row["profiles"].get("openblas", {})
        def cell(result: dict[str, Any]) -> str:
            if result.get("availability") != AVAILABLE:
                return result.get("availability", NOT_MEASURED)
            v = result["numsharp_ms"]
            if v is None:
                return "—"
            # Adaptive precision: a sub-µs stable mean (the 250x-batched average of an
            # O(1)/scalar/view op) collapses to "0.0000 ms" at 4 decimals. Show ns
            # resolution (data is already retained at 6 decimals) so it reads as a real
            # measurement rather than a zero.
            if v >= 0.1:
                return f"{v:.4f} ms"
            if v >= 0.001:
                return f"{v:.5f} ms"
            return f"{v:.6f} ms"
        ratio = "failed" if row["status"] == "failed" else (
            "—" if row["ratio"] is None else f"{row['ratio']:.3f}x")
        lines.append(f"| `{row['operation']}` | {row['dtype']} | {row['n']:,} | {cell(managed)} | {cell(openblas)} | {row['effective_backend'] or '—'} | {ratio} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--managed", required=True, type=Path, help="Managed profile JSON (legacy list or profile envelope)")
    parser.add_argument("--managed-extra", type=Path, help="Additional Managed availability/timing profile envelope")
    parser.add_argument("--openblas", type=Path, help="OpenBLAS profile JSON or legacy openblas_results.tsv")
    parser.add_argument("--openblas-extra", type=Path, help="Additional OpenBLAS profile envelope")
    parser.add_argument("--output", required=True, type=Path, help="Output base path, without extension")
    parser.add_argument("--snapshot-date", help="Override the report provenance date (YYYY-MM-DD)")
    parser.add_argument("--commit", help="Override the benchmarked commit hash")
    parser.add_argument("--numpy-version", help="Override the measured NumPy version")
    args = parser.parse_args()

    managed_rows = load_json_profile(args.managed, "managed")
    if args.managed_extra and args.managed_extra.exists():
        managed_rows.extend(load_json_profile(args.managed_extra, "managed"))
    openblas_rows: list[dict[str, Any]] = []
    if args.openblas and args.openblas.exists():
        if args.openblas.suffix.lower() == ".tsv":
            openblas_rows, managed_supplement = load_legacy_openblas_tsv(args.openblas)
            managed_rows.extend(managed_supplement)
        else:
            openblas_rows = load_json_profile(args.openblas, "openblas")
    if args.openblas_extra and args.openblas_extra.exists():
        openblas_rows.extend(load_json_profile(args.openblas_extra, "openblas"))

    metadata = report_metadata(args.output)
    if args.snapshot_date:
        metadata["snapshot_date"] = args.snapshot_date
    if args.commit:
        metadata["commit"] = args.commit
    if args.numpy_version:
        metadata["numpy_version"] = args.numpy_version
    managed = profile_envelope("managed", managed_rows, metadata)
    openblas = profile_envelope("openblas", openblas_rows, metadata)
    combined = combine(managed, openblas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".managed.json").write_text(json.dumps(managed, indent=2), encoding="utf-8")
    args.output.with_suffix(".openblas.json").write_text(json.dumps(openblas, indent=2), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    write_csv(args.output.with_suffix(".csv"), combined)
    write_markdown(args.output.with_suffix(".md"), combined)
    print(f"Backend profiles: managed={len(managed['rows'])} openblas={len(openblas['rows'])} combined={len(combined['rows'])}")


if __name__ == "__main__":
    main()
