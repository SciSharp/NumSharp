"""NumPy 2.4.2 twin of polyinterp_bench.cs — the numpy.polynomial chebinterpolate benchmark (U9).

Writes data/polyinterp/manifest.json (gitignored) — the cell list the C# side replays — then prints NumPy's best-of-R
per-call time for every cell (TSV: cell<TAB>us). Each cell records the SHA-256 of NumPy's result (+ dtype and shape),
which the C# runner checks BEFORE it times the cell. Both sides run OpenBLAS at ONE thread: np.dot's bits (gemv / gemm)
depend on the thread count, and so does its speed. The callable is named (FUNCS here, the same names in the C# runner),
and both sides time the whole call, the function's own work included.

Usage (from this directory; pin both sides with the same mask, never run them concurrently):
    set NS_PROBE_AFFINITY=0x14
    set OPENBLAS_NUM_THREADS=1
    python polyinterp_numpy.py > numpy_interp_times.tsv
    set DOTNET_TC_CallCountingDelayMs=0
    dotnet run -c Release --no-cache polyinterp_bench.cs -- data/polyinterp numpy_interp_times.tsv > ns_interp.tsv
    python polyinterp_report.py ns_interp.tsv > polyinterp_results.md
POLYINTERP_ONLY=T/,V/ (comma-separated label prefixes) restricts both sides to matching cells.

Sections (cell-label prefixes):
    T  tanh(x) + 0.5 (a vector result: gemv) at degree 1 / 2 / 5 / 10 / 20 / 50 / 100 / 200 / 500 / 1000
    V  [sin(x), cos(x)] stacked into a (deg + 1, 2) result (gemm) at degree 5 / 20 / 100 / 500
    C  exp(1j*x) (a complex result: zgemv) at degree 5 / 20 / 100 / 500
    R  result kinds at degree 20: a Python list, a Python float (the (21, 21) product), float32, int64
    A  argument forms at degree 20: args as a tuple / a list / an ndarray; deg as an np.uint16 (NumSharp's char) / a 0-d
       int64 array
"""
import json
import os
import sys
import timeit
from pathlib import Path

assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "set OPENBLAS_NUM_THREADS=1 before Python starts"

import numpy as np  # noqa: E402
from numpy.polynomial import chebyshev as C  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "polyinterp"
sys.path.insert(0, str(HERE))
from polyeval_numpy import digest, pin_affinity  # noqa: E402  (the shared digest + host pinning)

# The named callables (the C# runner holds the same names).
FUNCS = {
    "tanh_half": lambda x: np.tanh(x) + 0.5,
    "stack2": lambda x: np.stack([np.sin(x), np.cos(x)], axis=1),
    "expi": lambda x: np.exp(1j * x),
    "list": lambda x: [float(v) for v in np.sin(x)],
    "pyfloat": lambda x: 2.0,
    "f32": lambda x: np.sin(x).astype(np.float32),
    "i64": lambda x: (x * 10).astype(np.int64),
    "sin_ab": lambda x, a, b: np.sin(a * x) + b,
}
CELLS = []
ONLY = [p for p in os.environ.get("POLYINTERP_ONLY", "").split(",") if p]


def loops_for(deg):
    """Picks the timing loop of a cell: enough calls per round to dwarf the timer, a few calls for the big degrees.

    Args:
        deg: The degree (chebvander's matrix holds (deg + 1)**2 elements).

    Returns:
        (calls per round, rounds) — both sides time every cell with the same pair.
    """
    if deg >= 1000:
        return 5, 7
    if deg >= 500:
        return 10, 7
    if deg >= 200:
        return 50, 7
    if deg >= 50:
        return 300, 9
    return 2000, 9


def cell(label, fn, deg, args=None, deg_form=None):
    """Registers one cell, chebinterpolate(FUNCS[fn], deg[, args]), and NumPy's result digest.

    Args:
        label: The cell's label ("<section>/<case>"); POLYINTERP_ONLY restricts the run to matching labels.
        fn: The FUNCS key of the callable.
        deg: The degree.
        args: The args argument as ("tuple" | "list" | "ndarray", values), or None to omit it.
        deg_form: How the degree is passed — None (a Python int), "u16" (an np.uint16 scalar: NumSharp's char) or "nd0"
            (a 0-d int64 array).

    Returns:
        None.
    """
    if ONLY and not any(label.startswith(p) for p in ONLY):
        return
    f = FUNCS[fn]
    d = {None: lambda: deg, "u16": lambda: np.uint16(deg), "nd0": lambda: np.array(deg)}[deg_form]()
    if args is None:
        call = lambda: C.chebinterpolate(f, d)
    else:
        kind, values = args
        a = {"tuple": tuple, "list": list, "ndarray": np.array}[kind](values)
        call = lambda: C.chebinterpolate(f, d, a)
    r = call()
    calls, rounds = loops_for(deg)
    spec = {"label": label, "fn": fn, "deg": deg, "expect": [digest(r)], "calls": calls, "rounds": rounds}
    if args is not None:
        spec["args"] = {"kind": args[0], "values": list(args[1])}
    if deg_form:
        spec["deg_form"] = deg_form
    CELLS.append({**spec, "_call": call})


def build():
    """Registers every cell of the five sections (see the module docstring).

    Returns:
        None.
    """
    for deg in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        cell(f"T/deg{deg}", "tanh_half", deg)
    for deg in (5, 20, 100, 500):
        cell(f"V/deg{deg}", "stack2", deg)
    for deg in (5, 20, 100, 500):
        cell(f"C/deg{deg}", "expi", deg)
    for fn in ("list", "pyfloat", "f32", "i64"):
        cell(f"R/{fn}", fn, 20)
    for kind in ("tuple", "list", "ndarray"):
        cell(f"A/args_{kind}", "sin_ab", 20, args=(kind, (2.0, 1.0)))
    cell("A/deg_npu16", "tanh_half", 20, deg_form="u16")
    cell("A/deg_nd0", "tanh_half", 20, deg_form="nd0")


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
