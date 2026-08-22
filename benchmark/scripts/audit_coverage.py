#!/usr/bin/env python3
"""Audit benchmark coverage against NumSharp's generated NumPy API inventory.

This is a source-level coverage audit, not a timing run. It answers three questions:

* Which available NumPy-compatible APIs have an official NumSharp-vs-NumPy benchmark?
* Which APIs are covered only through an alias/subsystem or are deliberately excluded?
* Which APIs can use OpenBLAS, and which fail without its LAPACK backend?

The authoritative API surface is ``coverage/generated/coverage.json``. Benchmark evidence is
discovered from ``[Benchmark(Description = ...)]`` attributes in the 18 official op-matrix
namespaces, plus the small reviewed override ledger in ``benchmark/coverage/overrides.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
API_INVENTORY = ROOT / "coverage" / "generated" / "coverage.json"
OVERRIDES = ROOT / "benchmark" / "coverage" / "overrides.json"
OUTPUT_DIR = ROOT / "benchmark" / "coverage" / "generated"
BENCHMARK_ROOT = ROOT / "benchmark" / "NumSharp.Benchmark.CSharp" / "Benchmarks"

OFFICIAL_DIRS = {
    "Arithmetic",
    "Unary",
    "Reduction",
    "Broadcasting",
    "Creation",
    "Manipulation",
    "Slicing",
    "Comparison",
    "Bitwise",
    "Logic",
    "Statistics",
    "Sorting",
    "LinearAlgebra",
    "Selection",
    "Fourier",
    "Random",
    "NDArrayApi",
    "ApiSurface",
}

DESCRIPTION_RE = re.compile(r'\[Benchmark\(Description\s*=\s*"([^"]+)"\)\]')
COMMENT_RE = re.compile(r"//[^\r\n]*|/\*.*?\*/", re.MULTILINE | re.DOTALL)
NP_API_RE = re.compile(r"\bnp\.(?:(fft|linalg|random)\.)?([A-Za-z_][A-Za-z0-9_]*)")
NDARRAY_API_RE = re.compile(r"\ba\.([A-Za-z_][A-Za-z0-9_]*)")

# Direct OpenBLAS backend hooks with managed fallbacks.
OPENBLAS_OPTIONAL = {
    "numpy.dot", "numpy.inner", "numpy.matmul", "numpy.matvec", "numpy.vdot",
    "numpy.vecdot", "numpy.vecmat", "numpy.ndarray.dot",
    "numpy.linalg.matmul", "numpy.linalg.vecdot",
}

# These compose operations above and therefore use OpenBLAS for eligible contractions while still
# remaining fully functional with NumSharp.Core's managed kernels.
OPENBLAS_OPTIONAL_COMPOSED = {
    "numpy.einsum", "numpy.tensordot", "numpy.linalg.multi_dot", "numpy.linalg.tensordot",
}

# Only the long float32/float64/complex128 sliding-dot regions use the backend.
OPENBLAS_OPTIONAL_SLIDING = {"numpy.correlate", "numpy.convolve"}

# Direct LAPACK-backed entry points: no managed factorisation exists in NumSharp.Core.
OPENBLAS_REQUIRED = {
    "numpy.linalg.cholesky", "numpy.linalg.det", "numpy.linalg.slogdet",
    "numpy.linalg.eig", "numpy.linalg.eigvals", "numpy.linalg.eigh", "numpy.linalg.eigvalsh",
    "numpy.linalg.inv", "numpy.linalg.lstsq", "numpy.linalg.qr", "numpy.linalg.solve",
    "numpy.linalg.svd", "numpy.linalg.svdvals",
}

OPENBLAS_REQUIRED_COMPOSED = {
    "numpy.linalg.cond", "numpy.linalg.matrix_rank", "numpy.linalg.pinv",
    "numpy.linalg.tensorinv", "numpy.linalg.tensorsolve", "numpy.polyfit", "numpy.roots",
}

# These require LAPACK only for particular modes or because they compose a required primitive.
OPENBLAS_HYBRID = {
    "numpy.linalg.matrix_norm", "numpy.linalg.matrix_power", "numpy.linalg.norm",
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def api_id_from_np_match(module: str | None, name: str) -> str:
    return f"numpy.{module + '.' if module else ''}{name}"


def discover_benchmark_evidence() -> tuple[dict[str, list[dict]], list[dict]]:
    evidence: dict[str, list[dict]] = defaultdict(list)
    unmatched: list[dict] = []

    for directory in sorted(OFFICIAL_DIRS):
        suite_dir = BENCHMARK_ROOT / directory
        if not suite_dir.exists():
            continue
        for path in sorted(suite_dir.rglob("*.cs")):
            text = COMMENT_RE.sub("", path.read_text(encoding="utf-8-sig"))
            for match in DESCRIPTION_RE.finditer(text):
                description = match.group(1)
                found: set[str] = set()
                for api_match in NP_API_RE.finditer(description):
                    found.add(api_id_from_np_match(api_match.group(1), api_match.group(2)))
                for api_match in NDARRAY_API_RE.finditer(description):
                    name = api_match.group(1)
                    # Array indexing descriptions (a.shape-like text is rare) are accepted only
                    # when the inventory later confirms the member exists.
                    found.add(f"numpy.ndarray.{name}")

                row = {
                    "description": description,
                    "suite_directory": directory,
                    "path": relative(path),
                }
                if found:
                    for api_id in sorted(found):
                        evidence[api_id].append(row)
                else:
                    unmatched.append(row)
    return dict(evidence), unmatched


def backend_route(api_id: str) -> tuple[str, str]:
    if api_id in OPENBLAS_REQUIRED:
        return "openblas_required", "LAPACK backend required; NumSharp.Core has no managed factorisation fallback."
    if api_id in OPENBLAS_REQUIRED_COMPOSED:
        return "openblas_required_composed", "Composes a LAPACK factorisation and has no managed fallback."
    if api_id in OPENBLAS_HYBRID:
        return "openblas_hybrid", "Some modes compose LAPACK-backed operations; other modes remain managed."
    if api_id in OPENBLAS_OPTIONAL_SLIDING:
        return "openblas_optional_sliding_dot", "Eligible long float/complex kernels use OpenBLAS; all other cases stay managed."
    if api_id in OPENBLAS_OPTIONAL_COMPOSED:
        return "openblas_optional_composed", "Eligible contractions compose an OpenBLAS product; a managed fallback always exists."
    if api_id in OPENBLAS_OPTIONAL:
        return "openblas_optional", "Eligible float32/float64/complex128 products use OpenBLAS; a managed fallback always exists."
    return "managed", "Runs in NumSharp.Core managed code."


def make_outputs() -> tuple[list[dict], dict, list[dict]]:
    inventory = json.loads(API_INVENTORY.read_text(encoding="utf-8"))
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    discovered, unmatched = discover_benchmark_evidence()
    inventory_rows = {
        row["id"]: row
        for row in inventory["rows"]
        if row.get("in_default_scope") and row.get("status") in {"available", "partial"}
    }

    # Ignore tokens that look API-shaped in descriptions but are not part of the current inventory
    # (for example the historical, non-NumPy spelling "np.byteswap"). Keep them in the diagnostic
    # list so stale or misleading benchmark labels stay visible.
    for api_id in list(discovered):
        if api_id not in inventory_rows:
            for item in discovered.pop(api_id):
                unmatched.append({**item, "unmatched_api_id": api_id})

    manual = overrides.get("manual_coverage", {})
    blocked = overrides.get("blocked", {})
    excluded = overrides.get("excluded", {})
    unknown_override_ids = (set(manual) | set(blocked) | set(excluded)) - set(inventory_rows)
    if unknown_override_ids:
        raise SystemExit("Unknown benchmark coverage override IDs: " + ", ".join(sorted(unknown_override_ids)))

    direct_ids = set(discovered) | set(manual)
    covered_targets = {
        inventory_rows[api_id].get("numsharp_target")
        for api_id in direct_ids
        if api_id in inventory_rows and inventory_rows[api_id].get("numsharp_target")
    }

    rows: list[dict] = []
    for api_id, api in sorted(inventory_rows.items()):
        evidence: list[dict] = []
        notes = ""
        if api_id in discovered:
            coverage_status = "op_matrix"
            evidence = discovered[api_id]
        elif api_id in manual:
            coverage_status = manual[api_id]["kind"]
            evidence = [{"path": manual[api_id]["evidence"]}]
            notes = manual[api_id].get("notes", "")
        elif api.get("numsharp_target") and api.get("numsharp_target") in covered_targets:
            coverage_status = "covered_alias"
            source = next(
                other_id for other_id in sorted(direct_ids)
                if other_id in inventory_rows
                and inventory_rows[other_id].get("numsharp_target") == api.get("numsharp_target")
            )
            evidence = [{"covered_by": source}]
            notes = "Shares the exact compiled NumSharp target with a directly covered API."
        elif api_id in blocked:
            coverage_status = "blocked"
            evidence = [{"path": blocked[api_id]["evidence"]}]
            notes = blocked[api_id]["reason"]
        elif api_id in excluded:
            coverage_status = "excluded"
            notes = excluded[api_id]
        elif api.get("kind") == "property":
            coverage_status = "excluded"
            notes = "Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise."
        else:
            coverage_status = "missing"

        route, route_notes = backend_route(api_id)
        rows.append({
            "id": api_id,
            "surface": api["surface"],
            "name": api["name"],
            "kind": api["kind"],
            "category": api["category"],
            "api_status": api["status"],
            "coverage_status": coverage_status,
            "backend_route": route,
            "backend_notes": route_notes,
            "evidence": evidence,
            "notes": notes,
        })

    status_counts = Counter(row["coverage_status"] for row in rows)
    backend_counts = Counter(row["backend_route"] for row in rows)
    by_surface: dict[str, dict] = {}
    for surface in sorted({row["surface"] for row in rows}):
        surface_rows = [row for row in rows if row["surface"] == surface]
        counts = Counter(row["coverage_status"] for row in surface_rows)
        measured = sum(counts[key] for key in ("op_matrix", "subsystem", "covered_alias"))
        required = len(surface_rows) - counts["excluded"]
        by_surface[surface] = {
            "total": len(surface_rows),
            "measured_or_aliased": measured,
            "excluded": counts["excluded"],
            "missing": counts["missing"],
            "blocked": counts["blocked"],
            "coverage_percent": round(100 * measured / required, 1) if required else 100.0,
        }

    measured = sum(status_counts[key] for key in ("op_matrix", "subsystem", "covered_alias"))
    required = len(rows) - status_counts["excluded"]
    summary = {
        "schema_version": 1,
        "api_inventory": relative(API_INVENTORY),
        "total_available_or_partial": len(rows),
        "measured_or_aliased": measured,
        "excluded": status_counts["excluded"],
        "missing": status_counts["missing"],
        "blocked": status_counts["blocked"],
        "coverage_percent": round(100 * measured / required, 1) if required else 100.0,
        "by_status": dict(sorted(status_counts.items())),
        "by_backend_route": dict(sorted(backend_counts.items())),
        "by_surface": by_surface,
        "unmatched_benchmark_descriptions": len(unmatched),
    }
    return rows, summary, sorted(unmatched, key=lambda item: (item["path"], item["description"]))


def render_markdown(rows: list[dict], summary: dict, unmatched: list[dict]) -> str:
    lines = [
        "# NumSharp benchmark API coverage",
        "",
        "Generated from the compiled NumPy 2.4.2 API inventory and the official benchmark sources.",
        "A covered alias shares the exact compiled NumSharp target with a directly measured API.",
        "",
        f"Headline benchmark coverage: **{summary['coverage_percent']:.1f}%** "
        f"({summary['measured_or_aliased']} covered of "
        f"{summary['total_available_or_partial'] - summary['excluded']} benchmarkable APIs; "
        f"{summary['excluded']} reviewed exclusions).",
        "",
        "| Surface | Covered | Blocked | Excluded | Missing | Benchmarkable | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for surface, data in summary["by_surface"].items():
        lines.append(
            f"| {surface} | {data['measured_or_aliased']} | {data['blocked']} | {data['excluded']} | "
            f"{data['missing']} | {data['total'] - data['excluded']} | {data['coverage_percent']:.1f}% |"
        )

    lines.extend([
        "",
        "## Execution routes",
        "",
        "| Route | APIs | Meaning |",
        "|---|---:|---|",
        f"| managed | {summary['by_backend_route'].get('managed', 0)} | NumSharp.Core managed code only. |",
        f"| openblas_optional | {summary['by_backend_route'].get('openblas_optional', 0)} | Direct product hook with managed fallback. |",
        f"| openblas_optional_composed | {summary['by_backend_route'].get('openblas_optional_composed', 0)} | Composition can reach an accelerated product. |",
        f"| openblas_optional_sliding_dot | {summary['by_backend_route'].get('openblas_optional_sliding_dot', 0)} | Long eligible correlate/convolve regions only. |",
        f"| openblas_required | {summary['by_backend_route'].get('openblas_required', 0)} | LAPACK operation; no managed fallback. |",
        f"| openblas_required_composed | {summary['by_backend_route'].get('openblas_required_composed', 0)} | Top-level API composed from a required LAPACK factorisation. |",
        f"| openblas_hybrid | {summary['by_backend_route'].get('openblas_hybrid', 0)} | Backend requirement depends on parameters/composed path. |",
    ])

    lines.extend([
        "",
        "## Native/OpenBLAS API map",
        "",
        "Every API not listed below runs entirely in managed NumSharp.Core. Optional routes always",
        "retain a managed fallback; required routes throw when no suitable LAPACK backend is installed.",
        "",
        "| Route | APIs |",
        "|---|---|",
    ])
    for route in (
        "openblas_optional", "openblas_optional_composed", "openblas_optional_sliding_dot",
        "openblas_required", "openblas_required_composed", "openblas_hybrid",
    ):
        api_names = ", ".join(f"`{row['id']}`" for row in rows if row["backend_route"] == route)
        lines.append(f"| {route} | {api_names} |")

    lines.extend([
        "",
        "## Missing benchmark coverage",
        "",
        "| API | Surface | Kind | Category | Backend route |",
        "|---|---|---|---|---|",
    ])
    missing = [row for row in rows if row["coverage_status"] == "missing"]
    for row in missing:
        lines.append(
            f"| `{row['id']}` | {row['surface']} | {row['kind']} | {row['category']} | {row['backend_route']} |"
        )

    lines.extend([
        "",
        "## Benchmark-blocking open bugs",
        "",
        "These APIs remain in the benchmark denominator. They are not timed until the referenced",
        "repeated-execution hazard is fixed; the reason is explicit rather than a silent omission.",
        "",
        "| API | Reason | Evidence |",
        "|---|---|---|",
    ])
    for row in (row for row in rows if row["coverage_status"] == "blocked"):
        evidence = "; ".join(item.get("path", "") for item in row["evidence"])
        lines.append(f"| `{row['id']}` | {row['notes']} | `{evidence}` |")

    lines.extend([
        "",
        "## Reviewed exclusions",
        "",
        "| API | Reason |",
        "|---|---|",
    ])
    for row in (row for row in rows if row["coverage_status"] == "excluded"):
        lines.append(f"| `{row['id']}` | {row['notes']} |")

    lines.extend([
        "",
        "## Unmatched benchmark descriptions",
        "",
        "These official benchmark labels do not identify a current API-inventory row. Operator and",
        "scenario-only benchmarks are expected here; API-looking stale labels should be corrected.",
        "",
        "| Description | Source |",
        "|---|---|",
    ])
    for item in unmatched:
        suffix = f" (`{item['unmatched_api_id']}`)" if item.get("unmatched_api_id") else ""
        lines.append(f"| `{item['description']}`{suffix} | `{item['path']}` |")
    lines.append("")
    return "\n".join(lines)


def serialized_outputs() -> dict[Path, str]:
    rows, summary, unmatched = make_outputs()
    payload = {"schema_version": 1, "summary": summary, "rows": rows, "unmatched": unmatched}
    json_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    import io
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow([
        "id", "surface", "name", "kind", "category", "api_status", "coverage_status",
        "backend_route", "evidence", "notes",
    ])
    for row in rows:
        writer.writerow([
            row["id"], row["surface"], row["name"], row["kind"], row["category"],
            row["api_status"], row["coverage_status"], row["backend_route"],
            "; ".join(item.get("path", item.get("covered_by", "")) for item in row["evidence"]),
            row["notes"],
        ])
    return {
        OUTPUT_DIR / "coverage.json": json_text,
        OUTPUT_DIR / "coverage.csv": csv_buffer.getvalue(),
        OUTPUT_DIR / "summary.md": render_markdown(rows, summary, unmatched),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if committed coverage artifacts are stale.")
    args = parser.parse_args()

    outputs = serialized_outputs()
    if args.check:
        stale = [path for path, content in outputs.items() if not path.exists() or path.read_text(encoding="utf-8") != content]
        if stale:
            print("Benchmark coverage artifact is stale or missing:", file=sys.stderr)
            for path in stale:
                print(f"  - {relative(path)}", file=sys.stderr)
            print("Run: python benchmark/scripts/audit_coverage.py", file=sys.stderr)
            return 1
        print("Benchmark coverage artifact is current.")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8", newline="")
    summary = json.loads(outputs[OUTPUT_DIR / "coverage.json"])["summary"]
    print(
        f"Wrote {relative(OUTPUT_DIR)}: {summary['measured_or_aliased']}/"
        f"{summary['total_available_or_partial'] - summary['excluded']} benchmarkable APIs "
        f"({summary['coverage_percent']:.1f}%)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
