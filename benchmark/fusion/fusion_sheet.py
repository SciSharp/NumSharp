#!/usr/bin/env python3
# =============================================================================
# fusion_sheet.py — THE Fusion subsystem orchestrator + renderer.
#
# np.evaluate compiles an NDExpr tree to ONE NDIter pass (numexpr-style), so
# chained expressions allocate no intermediates. The op-matrix can't express
# this. This subsystem runs the fusion gate: evaluate_bench.cs reports fused
# np.evaluate vs unfused np.* chains (NumSharp-internal speedups), and
# evaluate_bench.py reports the NumPy absolutes on the same box for context.
#
# Result model is a fixed expression report (the chain gate plus an operand-layout
# sweep of the flagship a*b+c — C/F/T/strided/bcast — that checks the fused
# single-pass advantage survives non-contiguous operands), not a dtype/layout
# ratio matrix, so it is rendered as a fenced block -> fusion_results.md. Driven
# by run_benchmark.py; also standalone:
#   python benchmark/fusion/fusion_sheet.py [--skip-build]
# =============================================================================
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmark", "scripts"))
import bench_common as bc  # noqa: E402

MD = os.path.join(HERE, "fusion_results.md")


def main():
    ap = argparse.ArgumentParser(description="Fusion subsystem (np.evaluate fused vs unfused vs NumPy)")
    ap.add_argument("--skip-build", action="store_true", help="reuse the existing Release build")
    args = ap.parse_args()

    if not args.skip_build:
        bc.build_core(REPO)

    bc.log("[fusion] evaluate_bench (NumSharp fused vs unfused) …")
    cs_out = bc.run_cs(REPO, os.path.join(HERE, "evaluate_bench.cs")).strip()
    bc.log("[fusion] evaluate_bench (NumPy absolutes) …")
    py_out = bc.run_py(REPO, os.path.join(HERE, "evaluate_bench.py")).strip()

    cs_out = cs_out or "(NumSharp side produced no output — build/run failed.)"
    py_out = py_out or "(NumPy side produced no output.)"

    body = ("NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; "
            "(Nx) = unfused ÷ fused, >1 = fusion faster):\n\n"
            + cs_out
            + "\n\nNumPy — absolutes on the same box (context for the unfused column):\n\n"
            + py_out)
    md = ("# Fusion — np.evaluate vs unfused chains (and NumPy context)\n\n"
          "`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). "
          "Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` "
          "(C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous "
          "operands?), not a dtype/layout matrix — so reported as-is.\n\n"
          "```\n" + body + "\n```\n")
    with open(MD, "w", encoding="utf-8") as f:
        f.write(md)
    bc.log(f"[fusion] -> {os.path.relpath(MD, REPO)}")
    print(md)


if __name__ == "__main__":
    main()
