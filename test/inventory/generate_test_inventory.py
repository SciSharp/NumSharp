#!/usr/bin/env python3
"""Generate the deterministic Tests & Oracle Dashboard artifact.

The .NET helper reflects the actual MSTest assemblies; this layer adds source ownership and the
committed oracle corpus dimensions. No tests are executed and no pass rate is invented.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "generated"
TOOL = HERE / "NumSharp.Tools.TestInventory" / "NumSharp.Tools.TestInventory.csproj"
TOOL_DLL = HERE / "NumSharp.Tools.TestInventory" / "bin" / "Release" / "net8.0" / "NumSharp.Tools.TestInventory.dll"
CORPUS = ROOT / "test" / "NumSharp.Tests.Oracle" / "Fuzz" / "corpus"
GENERATOR_VERSION = "1.0.0"
OUTPUT_FILES = ("tests-oracle-report.json", "tests-oracle-report.csv", "summary.md", "tests-oracle-manifest.json")

PROJECT_ROOTS = {
    "NumSharp.Tests": ROOT / "test" / "NumSharp.Tests",
    "NumSharp.Tests.Oracle": ROOT / "test" / "NumSharp.Tests.Oracle",
    "NumSharp.Tests.Interop": ROOT / "test" / "NumSharp.Tests.Interop",
}

AREA_LABELS = {
    "APIs": "API & iteration",
    "AuditV2": "API audit",
    "Backends": "Backends & kernels",
    "Casting": "Casting",
    "Creation": "Creation",
    "Fuzz": "Differential oracle",
    "IO": "I/O & formats",
    "Indexing": "Indexing",
    "LinearAlgebra": "Linear algebra",
    "Logic": "Logic",
    "Manipulation": "Manipulation",
    "Math": "Math",
    "NumPyPortedTests": "NumPy port regressions",
    "Polynomial": "Polynomials",
    "Random": "Random",
    "RandomSampling": "Random",
    "Selection": "Selection",
    "Sorting": "Sorting & searching",
    "Statistics": "Statistics",
    "View": "Shape & views",
}

METHOD_RE = re.compile(
    r"^\s*(?:\[[^\]\r\n]+\]\s*)*public\s+"
    r"(?:(?:static|async|unsafe|virtual|override|sealed|new)\s+)*"
    r"(?:[A-Za-z_][A-Za-z0-9_\.<>\[\],?]*\s+)+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)


def run_reflection_inventory() -> dict[str, Any]:
    build = [
        "dotnet", "build", str(TOOL), "-c", "Release", "-f", "net8.0", "--nologo",
        "-v", "q", "-clp:ErrorsOnly", "-p:WarningLevel=0",
    ]
    subprocess.run(build, cwd=ROOT, check=True)
    completed = subprocess.run(
        ["dotnet", str(TOOL_DLL)], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(f"TestInventory did not emit valid JSON:\n{completed.stdout}\n{completed.stderr}") from error


def source_index(raw: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    wanted: dict[str, set[str]] = {}
    for project in raw["projects"]:
        wanted[project["name"]] = {test["method"] for test in project["tests"]}

    index: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for project, root in PROJECT_ROOTS.items():
        names = wanted.get(project, set())
        for path in sorted(root.rglob("*.cs")):
            if any(part in {"bin", "obj"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8-sig")
            classes = set(re.findall(r"\b(?:class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)", text))
            for match in METHOD_RE.finditer(text):
                name = match.group("name")
                if name not in names:
                    continue
                line = text.count("\n", 0, match.start()) + 1
                tail = text[match.start():match.start() + 1600]
                corpus_match = re.search(
                    r"(?:RunCorpus|RunIndexCorpus|RunDtypeCorpus|RunSetterDtypeCorpus)\(\s*\"([^\"]+)\"",
                    tail,
                )
                index[(project, name)].append({
                    "path": path.relative_to(ROOT).as_posix(),
                    "line": line,
                    "classes": classes,
                    "corpus_file": corpus_match.group(1) if corpus_match else None,
                })
    return index


def area_for(project: str, source_path: str | None) -> str:
    if project == "NumSharp.Tests.Interop":
        return "Python interoperability"
    if not source_path:
        return "Unlocated"
    root = PROJECT_ROOTS[project].relative_to(ROOT).as_posix() + "/"
    rel = source_path.removeprefix(root)
    first = rel.split("/", 1)[0]
    if "/" not in rel:
        if first.startswith("OpenBugs"):
            return "Open bugs"
        return "General"
    return AREA_LABELS.get(first, first.replace("_", " "))


def status_for(test: dict[str, Any]) -> str:
    categories = set(test["categories"])
    if test["ignored"]:
        return "ignored"
    if "OpenBugs" in categories:
        return "open_bug"
    if "HighMemory" in categories or "Explicit" in categories:
        return "manual"
    if "WindowsOnly" in categories:
        return "platform"
    return "active"


def oracle_kind(test: dict[str, Any], project: str, source_path: str | None) -> str | None:
    categories = set(test["categories"])
    if "FuzzMatrix" in categories:
        return "FuzzMatrix"
    if "NpyOracle" in categories:
        return "NPY/NPZ format oracle"
    if project == "NumSharp.Tests.Oracle":
        return "Oracle contract"
    if source_path and ("FlagsOracle" in source_path or "LayoutParityOracle" in source_path):
        return "Layout/flags oracle"
    if project == "NumSharp.Tests.Interop":
        return "Live NumPy interop"
    return None


def enrich_tests(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    index = source_index(raw)
    versions: dict[str, str] = {}
    tests: list[dict[str, Any]] = []
    for project in raw["projects"]:
        project_name = project["name"]
        versions[project_name] = project["assemblyVersion"]
        for test in project["tests"]:
            simple_type = test["type"].replace("+", ".").split(".")[-1]
            candidates = index.get((project_name, test["method"]), [])
            owned = [candidate for candidate in candidates if simple_type in candidate["classes"]]
            chosen = (owned or candidates or [None])[0]
            source_path = chosen["path"] if chosen else None
            categories = sorted(set(test["categories"]))
            status = status_for(test)
            declared_invocations = test["dataRows"] if test["dataRows"] > 0 else 1
            row = {
                **test,
                "project": project_name,
                "class": simple_type,
                "categories": categories,
                "status": status,
                "declaredInvocations": declared_invocations,
                "sourcePath": source_path,
                "sourceLine": chosen["line"] if chosen else None,
                "sourceUrl": (f"https://github.com/SciSharp/NumSharp/blob/master/{source_path}#L{chosen['line']}"
                              if chosen else None),
                "area": area_for(project_name, source_path),
                "oracleKind": oracle_kind(test, project_name, source_path),
                "corpusFiles": [chosen["corpus_file"]] if chosen and chosen["corpus_file"] else [],
            }
            tests.append(row)
    return sorted(tests, key=lambda row: row["id"]), versions


def classify_corpus_file(name: str) -> str:
    if name.endswith(".host.jsonl"):
        return "host_pin"
    if name.startswith("index_"):
        return "advanced_indexing"
    if name.startswith("decimal_"):
        return "decimal"
    if name == "random_parity_host.jsonl" or name == "matmul_parity.jsonl":
        return "host_sensitive_values"
    return "numpy"


def scan_oracle() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    files: list[dict[str, Any]] = []
    op_stats: dict[str, dict[str, Any]] = {}
    total_rows = char_rows = error_rows = 0
    dtype_rows: collections.Counter[str] = collections.Counter()
    all_layouts: set[str] = set()
    all_kinds: collections.Counter[str] = collections.Counter()

    for path in sorted(CORPUS.glob("*.jsonl")):
        kind = classify_corpus_file(path.name)
        rows = errors = file_char = 0
        ops: collections.Counter[str] = collections.Counter()
        dtypes: set[str] = set()
        layouts: set[str] = set()
        result_kinds: collections.Counter[str] = collections.Counter()
        value_classes: set[str] = set()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                case = json.loads(line)
                rows += 1
                total_rows += 1
                op = case.get("op")
                if op:
                    ops[op] += 1
                layout = case.get("layout")
                if layout:
                    layouts.add(layout)
                    all_layouts.add(layout)
                valueclass = case.get("valueclass")
                if valueclass:
                    value_classes.add(valueclass)
                has_error = bool(case.get("error") or case.get("expects_throw"))
                if has_error:
                    errors += 1
                    error_rows += 1
                expected = case.get("expected") or {}
                result_kind = expected.get("kind", "array")
                result_kinds[result_kind] += 1
                all_kinds[result_kind] += 1
                row_dtypes = {operand.get("dtype") for operand in case.get("operands", []) if operand.get("dtype")}
                if expected.get("dtype"):
                    row_dtypes.add(expected["dtype"])
                if case.get("dtype"):
                    row_dtypes.add(case["dtype"])
                dtypes.update(row_dtypes)
                for dtype in row_dtypes:
                    dtype_rows[dtype] += 1
                if "char" in row_dtypes:
                    file_char += 1
                    char_rows += 1

                if op:
                    stat = op_stats.setdefault(op, {
                        "op": op, "cases": 0, "files": set(), "layouts": set(), "dtypes": set(),
                        "valueClasses": set(), "parameterSignatures": set(), "resultKinds": set(),
                        "errorCases": 0, "families": set(),
                    })
                    stat["cases"] += 1
                    stat["files"].add(path.name)
                    stat["layouts"].update([layout] if layout else [])
                    stat["dtypes"].update(row_dtypes)
                    stat["valueClasses"].update([valueclass] if valueclass else [])
                    stat["parameterSignatures"].add(json.dumps(case.get("params", {}), sort_keys=True, separators=(",", ":")))
                    stat["resultKinds"].add(result_kind)
                    stat["errorCases"] += int(has_error)
                    stat["families"].add(kind)

        files.append({
            "name": path.name,
            "family": kind,
            "rows": rows,
            "opKeys": sorted(ops),
            "opCount": len(ops),
            "dtypes": sorted(dtypes),
            "layouts": sorted(layouts),
            "valueClasses": sorted(value_classes),
            "resultKinds": dict(sorted(result_kinds.items())),
            "errorRows": errors,
            "charRows": file_char,
        })

    ops_rendered: list[dict[str, Any]] = []
    for stat in op_stats.values():
        rendered = {
            **stat,
            "files": sorted(stat["files"]),
            "layouts": sorted(stat["layouts"]),
            "dtypes": sorted(stat["dtypes"]),
            "valueClasses": sorted(stat["valueClasses"]),
            "parameterSignatures": len(stat["parameterSignatures"]),
            "resultKinds": sorted(stat["resultKinds"]),
            "families": sorted(stat["families"]),
        }
        rendered["strength"] = {
            "underTen": rendered["cases"] < 10,
            "oneLayout": len(rendered["layouts"]) <= 1,
            "oneDtype": len(rendered["dtypes"]) <= 1,
            "noErrors": rendered["errorCases"] == 0,
        }
        ops_rendered.append(rendered)
    ops_rendered.sort(key=lambda row: (row["cases"], row["op"]))

    family_rows = collections.Counter()
    for file in files:
        family_rows[file["family"]] += file["rows"]

    flags_path = ROOT / "test" / "NumSharp.Tests" / "Backends" / "corpus" / "flags_oracle.jsonl"
    layout_path = ROOT / "test" / "NumSharp.Tests" / "Backends" / "corpus" / "layout_parity_oracle.jsonl"
    flags_rows = sum(1 for line in flags_path.open(encoding="utf-8") if line.strip())
    layout_rows = sum(1 for line in layout_path.open(encoding="utf-8") if line.strip())
    npy_path = ROOT / "test" / "NumSharp.Tests" / "IO" / "corpus" / "npy_oracle.zip"
    with zipfile.ZipFile(npy_path) as archive:
        npy_manifest = json.loads(archive.read("manifest.json"))
    npy_cases = len(npy_manifest.get("cases", []))

    summary = {
        "corpusFiles": len(files),
        "corpusRows": total_rows,
        "numpyRows": family_rows["numpy"],
        "hostSensitiveRows": family_rows["host_sensitive_values"],
        "advancedIndexRows": family_rows["advanced_indexing"],
        "decimalRows": family_rows["decimal"],
        "hostPinRows": family_rows["host_pin"],
        "charRows": char_rows,
        "opKeys": len(op_stats),
        "layouts": len(all_layouts),
        "errorRows": error_rows,
        "resultKinds": dict(sorted(all_kinds.items())),
        "dtypeRows": dict(sorted(dtype_rows.items())),
        "flagsOracleRows": flags_rows,
        "layoutOracleRows": layout_rows,
        "npyOracleCases": npy_cases,
        "specializedOracleCases": flags_rows + layout_rows + npy_cases,
        "underTenOps": sum(row["strength"]["underTen"] for row in ops_rendered),
        "oneLayoutOps": sum(row["strength"]["oneLayout"] for row in ops_rendered),
        "oneDtypeOps": sum(row["strength"]["oneDtype"] for row in ops_rendered),
        "noErrorOps": sum(row["strength"]["noErrors"] for row in ops_rendered),
    }
    return files, ops_rendered, summary


def rollup_tests(tests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_suite: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for test in tests:
        by_suite[(test["project"], test["area"])].append(test)

    suites = []
    for (project, area), rows in sorted(by_suite.items()):
        statuses = collections.Counter(row["status"] for row in rows)
        categories = collections.Counter(category for row in rows for category in row["categories"])
        suites.append({
            "id": f"{project}::{area}",
            "project": project,
            "area": area,
            "methods": len(rows),
            "declaredInvocations": sum(row["declaredInvocations"] for row in rows),
            "dataRows": sum(row["dataRows"] for row in rows),
            "dynamicMethods": sum(row["dynamicData"] for row in rows),
            "oracleMethods": sum(row["oracleKind"] is not None for row in rows),
            "statuses": dict(sorted(statuses.items())),
            "categories": dict(sorted(categories.items())),
            "sourceFiles": sorted({row["sourcePath"] for row in rows if row["sourcePath"]}),
        })

    statuses = collections.Counter(test["status"] for test in tests)
    categories = collections.Counter(category for test in tests for category in test["categories"])
    projects = collections.Counter(test["project"] for test in tests)
    summary = {
        "methods": len(tests),
        "declaredInvocations": sum(test["declaredInvocations"] for test in tests),
        "dataRows": sum(test["dataRows"] for test in tests),
        "dynamicMethods": sum(test["dynamicData"] for test in tests),
        "sourceLocated": sum(test["sourcePath"] is not None for test in tests),
        "oracleMethods": sum(test["oracleKind"] is not None for test in tests),
        "fuzzMatrixMethods": sum("FuzzMatrix" in test["categories"] for test in tests),
        "npyOracleMethods": sum("NpyOracle" in test["categories"] for test in tests),
        "statuses": dict(sorted(statuses.items())),
        "categories": dict(sorted(categories.items())),
        "projects": dict(sorted(projects.items())),
        "suites": len(suites),
    }
    return suites, summary


def rollup_classes(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for test in tests:
        groups[(test["project"], test["type"])].append(test)
    rendered = []
    for (project, type_name), rows in sorted(groups.items()):
        statuses = collections.Counter(row["status"] for row in rows)
        categories = collections.Counter(category for row in rows for category in row["categories"])
        oracle_kinds = collections.Counter(row["oracleKind"] for row in rows if row["oracleKind"])
        source_paths = sorted({row["sourcePath"] for row in rows if row["sourcePath"]})
        source_lines = [row["sourceLine"] for row in rows if row["sourceLine"] is not None]
        rendered.append({
            "id": f"{project}::{type_name}",
            "project": project,
            "type": type_name,
            "class": rows[0]["class"],
            "area": rows[0]["area"],
            "methods": len(rows),
            "declaredInvocations": sum(row["declaredInvocations"] for row in rows),
            "dataRows": sum(row["dataRows"] for row in rows),
            "dynamicMethods": sum(row["dynamicData"] for row in rows),
            "statuses": dict(sorted(statuses.items())),
            "categories": dict(sorted(categories.items())),
            "oracleKinds": dict(sorted(oracle_kinds.items())),
            "corpusFiles": sorted({name for row in rows for name in row["corpusFiles"]}),
            "sourcePaths": source_paths,
            "sourceUrl": (f"https://github.com/SciSharp/NumSharp/blob/master/{source_paths[0]}"
                          f"#L{min(source_lines)}" if source_paths and source_lines else None),
            "methodNames": [row["method"] for row in sorted(rows, key=lambda row: row["method"])],
        })
    return rendered


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def csv_text(tests: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    fields = [
        "id", "project", "area", "class", "method", "status", "declaredInvocations",
        "dataRows", "dynamicData", "categories", "oracleKind", "corpusFiles", "sourcePath", "sourceLine",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for test in tests:
        writer.writerow({
            key: ";".join(test[key]) if key in {"categories", "corpusFiles"} else (test.get(key) or "")
            for key in fields
        })
    return output.getvalue()


def summary_text(report: dict[str, Any]) -> str:
    tests = report["summary"]["tests"]
    oracle = report["summary"]["oracle"]
    lines = [
        "# Tests & Oracle inventory", "",
        "Deterministic inventory of MSTest declarations and committed oracle evidence. It is not a runtime pass rate.", "",
        "## Headline", "",
        "| Measure | Count |", "|---|---:|",
        f"| Test methods | {tests['methods']:,} |",
        f"| Declared invocations | {tests['declaredInvocations']:,} |",
        f"| Active methods | {tests['statuses'].get('active', 0):,} |",
        f"| Open-bug methods | {tests['statuses'].get('open_bug', 0):,} |",
        f"| Oracle-owned methods | {tests['oracleMethods']:,} |",
        f"| FuzzMatrix methods | {tests['fuzzMatrixMethods']:,} |",
        f"| Committed op-corpus rows | {oracle['corpusRows']:,} |",
        f"| Corpus op keys | {oracle['opKeys']:,} |",
        f"| Specialized flags/layout/NPY cases | {oracle['specializedOracleCases']:,} |", "",
        "## Test projects", "", "| Project | Methods |", "|---|---:|",
    ]
    for project, count in tests["projects"].items():
        lines.append(f"| `{project}` | {count:,} |")
    lines += ["", "## Oracle strength queue", "",
              f"- Ops below 10 cases: **{oracle['underTenOps']}**",
              f"- Ops with one/no serialized layout: **{oracle['oneLayoutOps']}**",
              f"- Ops with one/no dtype: **{oracle['oneDtypeOps']}**",
              f"- Ops without a recorded error row: **{oracle['noErrorOps']}**", ""]
    return "\n".join(lines)


def render() -> dict[str, str]:
    raw = run_reflection_inventory()
    tests, versions = enrich_tests(raw)
    suites, test_summary = rollup_tests(tests)
    test_classes = rollup_classes(tests)
    oracle_files, oracle_ops, oracle_summary = scan_oracle()
    expected_projects = set(PROJECT_ROOTS)
    if set(versions) != expected_projects:
        raise SystemExit(f"Test assembly scan mismatch: expected {sorted(expected_projects)}, got {sorted(versions)}")
    duplicate_ids = [test_id for test_id, count in collections.Counter(test["id"] for test in tests).items() if count > 1]
    if duplicate_ids:
        raise SystemExit("Duplicate reflected test ids: " + ", ".join(duplicate_ids[:20]))
    unlocated = [test["id"] for test in tests if test["sourcePath"] is None]
    if unlocated:
        raise SystemExit(f"{len(unlocated)} reflected tests have no source location; first: " + ", ".join(unlocated[:20]))
    if not oracle_files or not oracle_ops or oracle_summary["corpusRows"] == 0:
        raise SystemExit("Oracle scan is empty; refusing to generate a vacuous dashboard artifact.")
    report = {
        "schemaVersion": 1,
        "generator": "test/inventory/generate_test_inventory.py",
        "generatorVersion": GENERATOR_VERSION,
        "numpyVersion": "2.4.2",
        "assemblyVersions": versions,
        "summary": {"tests": test_summary, "oracle": oracle_summary},
        "suites": suites,
        "testClasses": test_classes,
        "oracleFiles": oracle_files,
        "oracleOps": oracle_ops,
    }
    manifest = {
        "schema_version": 1,
        "generator": report["generator"],
        "generator_version": GENERATOR_VERSION,
        "numpy_version": report["numpyVersion"],
        "assembly_versions": versions,
        "artifact_files": list(OUTPUT_FILES),
        "summary": report["summary"],
    }
    return {
        "tests-oracle-report.json": json_text(report),
        "tests-oracle-report.csv": csv_text(tests),
        "summary.md": summary_text(report),
        "tests-oracle-manifest.json": json_text(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render()
    if args.check:
        stale = []
        for name, content in rendered.items():
            path = args.output / name
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(name)
        if stale:
            raise SystemExit("Tests & Oracle inventory is stale: " + ", ".join(stale) +
                             "\nRun: python test/inventory/generate_test_inventory.py")
        print(f"Tests & Oracle inventory is current ({args.output}).")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        (args.output / name).write_text(content, encoding="utf-8", newline="\n")
    report = json.loads(rendered["tests-oracle-report.json"])
    print(f"Wrote {args.output}: {report['summary']['tests']['methods']} test methods, "
          f"{report['summary']['oracle']['corpusRows']} corpus rows, "
          f"{report['summary']['oracle']['opKeys']} op keys.")


if __name__ == "__main__":
    main()
