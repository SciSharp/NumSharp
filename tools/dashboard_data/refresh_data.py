#!/usr/bin/env python3
"""Produce dashboard data into the refs/data submodule for ALL kinds and commit — the "same for all
kinds" companion to ``benchmark/run_benchmark.py`` (which publishes the ``benchmark`` type itself).

For each requested type it (re)generates the data, then publishes it into the refs/data submodule via
``publish.py --pull --commit`` — i.e. pull the ``data`` branch to its tip, override that type's
``latest/`` with the fresh output, and commit on branch ``data`` (locally). ``--push`` also pushes.

    python tools/dashboard_data/refresh_data.py                     # all kinds
    python tools/dashboard_data/refresh_data.py --type inventory    # one kind
    python tools/dashboard_data/refresh_data.py --push              # commit AND push to origin/data

``benchmark`` is NOT re-run here (it is heavy — use ``run_benchmark.py``); with ``--type benchmark``
or ``all`` the EXISTING ``benchmark/history/latest`` is published if present. Regeneration needs
``numpy==2.4.2`` (inventory) and the .NET SDK (tests-oracle builds a reflection tool).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # tools/dashboard_data
ROOT = HERE.parent.parent                        # repo root
PUBLISH = HERE / "publish.py"

INVENTORY_DIR = ROOT / "coverage" / "generated"
TESTS_ORACLE_DIR = ROOT / "test" / "inventory" / "generated"
BENCHMARK_COVERAGE_DIR = ROOT / "benchmark" / "coverage" / "generated"
BENCHMARK_LATEST = ROOT / "benchmark" / "history" / "latest"

# inventory before benchmark-coverage: audit_coverage.py reads coverage/generated/coverage.json
QUICK_ORDER = ("inventory", "benchmark-coverage", "tests-oracle")


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


def publish(type_name: str, src: Path, data_worktree: Path):
    run([sys.executable, PUBLISH, "--type", type_name, "--from", src,
         "--branch-worktree", data_worktree, "--pull", "--commit"])


def generate_inventory():
    run([sys.executable, ROOT / "coverage" / "generate_coverage.py", "--output", INVENTORY_DIR])


def generate_tests_oracle():
    run([sys.executable, ROOT / "test" / "inventory" / "generate_test_inventory.py", "--output", TESTS_ORACLE_DIR])


def generate_benchmark_coverage():
    # audit_coverage.py reads coverage/generated/coverage.json and writes benchmark/coverage/generated/.
    if not (INVENTORY_DIR / "coverage.json").exists():
        generate_inventory()
    run([sys.executable, ROOT / "benchmark" / "scripts" / "audit_coverage.py"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", default="all",
                    choices=("all", "inventory", "tests-oracle", "benchmark-coverage", "benchmark"))
    ap.add_argument("--data-worktree", default=str(ROOT / "refs" / "data"),
                    help="The data-branch worktree/submodule to publish into (default: refs/data)")
    ap.add_argument("--push", action="store_true", help="Also push the data branch to origin after committing")
    args = ap.parse_args(argv)

    data = Path(args.data_worktree).resolve()
    if not (data / ".git").exists():
        print(f"error: data worktree not initialized: {data}\n"
              f"  git clone --depth 1 -b data https://github.com/SciSharp/NumSharp.git refs/data", file=sys.stderr)
        return 2

    want = {"inventory", "tests-oracle", "benchmark-coverage", "benchmark"} if args.type == "all" else {args.type}

    for t in QUICK_ORDER:
        if t not in want:
            continue
        if t == "inventory":
            generate_inventory(); publish("inventory", INVENTORY_DIR, data)
        elif t == "tests-oracle":
            generate_tests_oracle(); publish("tests-oracle", TESTS_ORACLE_DIR, data)
        elif t == "benchmark-coverage":
            generate_benchmark_coverage(); publish("benchmark-coverage", BENCHMARK_COVERAGE_DIR, data)

    if "benchmark" in want:
        if BENCHMARK_LATEST.exists():
            publish("benchmark", BENCHMARK_LATEST, data)
        else:
            print("\nbenchmark: no benchmark/history/latest — run `python benchmark/run_benchmark.py` "
                  "to produce and publish it.", file=sys.stderr)

    if args.push:
        run(["git", "push", "origin", "data"], cwd=data)
        print("\npushed the data branch to origin")
    else:
        print("\nCommitted locally into refs/data (branch 'data'). Push when ready: "
              "git -C refs/data push origin data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
