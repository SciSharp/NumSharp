#!/usr/bin/env python3
"""Check C# benchmark descriptions against NumPy smoke-result join names.

Run each NumPy suite with ``--quick --size small --output benchmark/results/smoke/<suite>.json``
first. This check ignores dtype/size cells and verifies the structural operation-name join in both
directions, using the exact normalizer used by the official merge.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CS_ROOT = ROOT / "benchmark" / "NumSharp.Benchmark.CSharp" / "Benchmarks"
DEFAULT_SMOKE = ROOT / "benchmark" / "results" / "smoke"
DESCRIPTION_RE = re.compile(r'\[Benchmark\(Description\s*=\s*"([^"]+)"\)\]')
COMMENT_RE = re.compile(r"//[^\r\n]*|/\*.*?\*/", re.MULTILINE | re.DOTALL)

SUITE_DIRS = {
    "arithmetic": "Arithmetic", "unary": "Unary", "reduction": "Reduction",
    "broadcast": "Broadcasting", "creation": "Creation", "manipulation": "Manipulation",
    "slicing": "Slicing", "comparison": "Comparison", "bitwise": "Bitwise", "logic": "Logic",
    "statistics": "Statistics", "sorting": "Sorting", "linalg": "LinearAlgebra",
    "selection": "Selection", "fft": "Fourier", "random": "Random",
    "ndarray": "NDArrayApi", "api": "ApiSurface",
}


def load_normalizer():
    path = ROOT / "benchmark" / "scripts" / "merge-results.py"
    spec = importlib.util.spec_from_file_location("benchmark_merge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.normalize_op_name


def csharp_names(directory: str, normalize) -> set[str]:
    names: set[str] = set()
    for path in (CS_ROOT / directory).rglob("*.cs"):
        text = COMMENT_RE.sub("", path.read_text(encoding="utf-8-sig"))
        names.update(normalize(match.group(1)) for match in DESCRIPTION_RE.finditer(text))
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--suites", nargs="*", choices=sorted(SUITE_DIRS), default=sorted(SUITE_DIRS))
    args = parser.parse_args()
    normalize = load_normalizer()
    failed = False

    for suite in args.suites:
        path = args.smoke_dir / f"{suite}.json"
        if not path.exists():
            print(f"{suite}: missing {path.relative_to(ROOT)}")
            failed = True
            continue
        python_names = {normalize(row["name"]) for row in json.loads(path.read_text(encoding="utf-8"))}
        cs_names = csharp_names(SUITE_DIRS[suite], normalize)
        missing_python = sorted(cs_names - python_names)
        missing_csharp = sorted(python_names - cs_names)
        if missing_python or missing_csharp:
            failed = True
            print(f"{suite}: join mismatch")
            if missing_python:
                print("  C# only: " + ", ".join(missing_python))
            if missing_csharp:
                print("  NumPy only: " + ", ".join(missing_csharp))
        else:
            print(f"{suite}: OK ({len(cs_names)} normalized operations)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
