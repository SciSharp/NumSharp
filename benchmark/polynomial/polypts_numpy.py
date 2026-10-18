"""NumPy 2.4.2 twin of polypts_bench.cs — the numpy.polynomial Chebyshev points and Gauss-Chebyshev quadrature (U8:
chebpts1 / chebpts2 / chebgauss) benchmark.

Writes data/polypts/manifest.json (gitignored) — the cell list the C# side replays — then prints NumPy's best-of-R
per-call time for every cell (TSV: cell<TAB>us). Each cell records the SHA-256 of NumPy's result bytes (+ dtype and
shape; chebgauss: both slots), which the C# runner checks BEFORE it times the cell.

Usage (from this directory; pin both sides with the same mask, never run them concurrently):
    set NS_PROBE_AFFINITY=0x14
    python polypts_numpy.py > numpy_pts_times.tsv
    set DOTNET_TC_CallCountingDelayMs=0
    dotnet run -c Release --no-cache polypts_bench.cs -- data/polypts numpy_pts_times.tsv > ns_pts.tsv
    python polypts_report.py ns_pts.tsv > polypts_results.md
POLYPTS_ONLY=P1/,G/ (comma-separated label prefixes) restricts both sides to matching cells.

Sections (cell-label prefixes):
    P1  chebpts1 at 1 / 2 / 5 / 16 / 100 / 1K / 10K / 100K / 1M points
    P2  chebpts2 at 2 / 5 / 16 / 100 / 1K / 10K / 100K / 1M points
    G   chebgauss at 1 / 5 / 16 / 100 / 1K / 10K / 100K / 1M nodes
    A   argument forms at 1K: a Python float, a 0-d int64 array, an np.uint16 scalar (NumSharp's char), a Python bool
"""
import json
import os
import sys
import timeit
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from numpy.polynomial import chebyshev as C  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "polypts"
sys.path.insert(0, str(HERE))
from polyeval_numpy import digest, pin_affinity  # noqa: E402  (the shared digest + host pinning)

FUNCS = {"chebpts1": C.chebpts1, "chebpts2": C.chebpts2, "chebgauss": C.chebgauss}
CELLS = []
ONLY = [p for p in os.environ.get("POLYPTS_ONLY", "").split(",") if p]


def loops_for(n):
    """Picks the timing loop of a cell: enough calls per round to dwarf the timer, a few calls for the big cells.

    Args:
        n: The number of points the call computes.

    Returns:
        (calls per round, rounds) — both sides time every cell with the same pair.
    """
    if n >= 500_000:
        return 3, 7
    if n >= 50_000:
        return 20, 7
    if n >= 5_000:
        return 100, 9
    if n >= 500:
        return 400, 9
    return 4000, 9


def cell(label, fn, n, form=None):
    """Registers one cell, fn(n) with the count passed in the given form, and NumPy's result digest(s).

    Args:
        label: The cell's label ("<section>/<case>"); POLYPTS_ONLY restricts the run to matching labels.
        fn: chebpts1, chebpts2 or chebgauss (the FUNCS key).
        n: The count.
        form: How the count is passed — None (a Python int), "float" (a Python float), "zerod" (a 0-d int64 array),
            "u16" (an np.uint16 scalar: NumSharp's char) or "bool" (a Python bool; n must be 1).

    Returns:
        None.
    """
    if ONLY and not any(label.startswith(p) for p in ONLY):
        return
    f = FUNCS[fn]
    # Built lazily: an np.uint16 of a count past 65535 would raise for the forms that never use it.
    make = {None: lambda: n, "float": lambda: float(n), "zerod": lambda: np.array(n, dtype=np.int64),
            "u16": lambda: np.uint16(n), "bool": lambda: bool(n)}[form]
    arg = make()
    call = lambda: f(arg)
    r = call()
    expect = [digest(a) for a in (r if isinstance(r, tuple) else (r,))]
    calls, rounds = loops_for(n)
    spec = {"label": label, "fn": fn, "n": n, "expect": expect, "calls": calls, "rounds": rounds}
    if form:
        spec["form"] = form
    CELLS.append({**spec, "_call": call})


def build():
    """Registers every cell of the four sections (see the module docstring).

    Returns:
        None.
    """
    for n in (1, 2, 5, 16, 100, 1_000, 10_000, 100_000, 1_000_000):
        cell(f"P1/n{n}", "chebpts1", n)
    for n in (2, 5, 16, 100, 1_000, 10_000, 100_000, 1_000_000):
        cell(f"P2/n{n}", "chebpts2", n)
    for n in (1, 5, 16, 100, 1_000, 10_000, 100_000, 1_000_000):
        cell(f"G/n{n}", "chebgauss", n)
    cell("A/chebpts1/pyfloat", "chebpts1", 1_000, "float")
    cell("A/chebpts2/pyfloat", "chebpts2", 1_000, "float")
    for fn in FUNCS:
        cell(f"A/{fn}/zerod", fn, 1_000, "zerod")
        cell(f"A/{fn}/npu16", fn, 1_000, "u16")
    cell("A/chebpts1/pybool", "chebpts1", 1, "bool")
    cell("A/chebgauss/pybool", "chebgauss", 1, "bool")


def main():
    """Pins the process, writes the manifest the C# runner replays, then prints NumPy's best per-call time per cell.

    Returns:
        None.
    """
    pin_affinity()
    DATA.mkdir(parents=True, exist_ok=True)
    build()
    (DATA / "manifest.json").write_text(json.dumps([{k: v for k, v in c.items() if k != "_call"} for c in CELLS],
                                                   indent=1), encoding="utf-8")
    print("cell\tnumpy_us")
    for c in CELLS:
        call, calls, rounds = c["_call"], c["calls"], c["rounds"]
        call()   # warm
        best = min(timeit.repeat(call, number=calls, repeat=rounds)) / calls
        print(f"{c['label']}\t{best * 1e6:.3f}", flush=True)


if __name__ == "__main__":
    main()
