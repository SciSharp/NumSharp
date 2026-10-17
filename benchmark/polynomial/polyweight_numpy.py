"""NumPy 2.4.2 twin of polyweight_bench.cs — the numpy.polynomial weight functions (U8: chebweight / legweight /
lagweight / hermweight / hermeweight) benchmark.

Writes data/polyweight/manifest.json (gitignored) — the cell list the C# side replays — then prints NumPy's
best-of-R per-call time for every cell (TSV: cell<TAB>us). Inputs come from the exact shared formula of
polyeval_numpy.py (G, reused here), and each cell records the SHA-256 of NumPy's result bytes (+ dtype and shape),
which the C# runner checks BEFORE it times the cell.

Usage (from this directory; pin both sides with the same mask, never run them concurrently):
    set NS_PROBE_AFFINITY=0x14
    python polyweight_numpy.py > numpy_weight_times.tsv
    set DOTNET_TC_CallCountingDelayMs=0
    dotnet run -c Release --no-cache polyweight_bench.cs -- data/polyweight numpy_weight_times.tsv > ns_weight.tsv
    python polyweight_report.py ns_weight.tsv > polyweight_results.md
POLYWEIGHT_ONLY=D/,L/ (comma-separated label prefixes) restricts both sides to matching cells.

Sections (cell-label prefixes; every section covers the five weights):
    D  1-D float64 points: 1 / 16 / 1K / 100K / 1M
    T  point dtypes: float32 / float16 / complex128 / int32 / int16 / uint8 / bool at 1K / 100K / 1M
       (no complex chebweight: np.sqrt's complex result is within a few ULP of NumPy's, not bit-exact; no bool lagweight:
       NumPy refuses the boolean negative)
    L  layouts @ 100K points: strided, reversed, F-order and transposed 2-D, a transposed 3-D (its axes reversed),
       a row broadcast, a stride-0 broadcast
    S  small calls (the fixed per-call cost): a Python float, a 0-d float64 array, 3 points
    A  argument forms: a Python int, a typed double[] (an ndarray to NumPy), a float16 NumPy scalar
"""
import json
import os
import sys
import timeit
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from numpy.polynomial import chebyshev as C  # noqa: E402
from numpy.polynomial import hermite as H  # noqa: E402
from numpy.polynomial import hermite_e as HE  # noqa: E402
from numpy.polynomial import laguerre as LG  # noqa: E402
from numpy.polynomial import legendre as L  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "polyweight"
sys.path.insert(0, str(HERE))
from polyeval_numpy import G, digest, pin_affinity  # noqa: E402  (the shared input formula + view grammar)

WEIGHTS = {"cheb": C.chebweight, "leg": L.legweight, "lag": LG.lagweight, "herm": H.hermweight, "herme": HE.hermeweight}
MODNAME = {"cheb": "chebyshev", "leg": "legendre", "lag": "laguerre", "herm": "hermite", "herme": "hermite_e"}
CELLS = []
ONLY = [p for p in os.environ.get("POLYWEIGHT_ONLY", "").split(",") if p]


def loops_for(work):
    """Picks the timing loop of a cell: enough calls per round to dwarf the timer, a few calls for the big cells.

    Args:
        work: The number of points the call computes.

    Returns:
        (calls per round, rounds) — both sides time every cell with the same pair.
    """
    if work >= 500_000:
        return 3, 7
    if work >= 50_000:
        return 20, 7
    if work >= 500:
        return 200, 9
    return 2000, 9


def cell(label, w, point, pform=None):
    """Registers one cell, {w}weight(x), with NumPy's result digest the C# runner checks before it times the cell.

    Args:
        label: The cell's label ("<section>/<weight>/<case>"); POLYWEIGHT_ONLY restricts the run to matching labels.
        w: The weight's prefix (cheb, leg, lag, herm, herme).
        point: The G array the argument is built from.
        pform: How the argument is passed — None (the array), "scalar" (a 0-d G as a Python float), "int" (a Python int:
            the 0-d G scaled to an int in [-5, 5]), "typed" (the C# side passes a typed double[], an ndarray to NumPy) or
            "f16" (an np.float16 scalar).
    """
    if ONLY and not any(label.startswith(p) for p in ONLY):
        return
    fn = WEIGHTS[w]
    a = point.array
    if pform == "scalar":
        arg = float(a)
    elif pform == "int":
        arg = int(round(float(a) * 5))
    elif pform == "f16":
        arg = np.float16(a)
    else:
        arg = a
    call = lambda: fn(arg)
    ref = np.asarray(call())
    calls, rounds = loops_for(max(int(np.prod(a.shape)), 1))
    spec = {"label": label, "module": MODNAME[w], "base": w, "point": point.spec, "expect": digest(ref), "calls": calls,
            "rounds": rounds}
    if pform:
        spec["pform"] = pform
    CELLS.append({**spec, "_call": call})


def build():
    """Registers every cell of the five sections (see the module docstring) for the five weights."""
    for w in WEIGHTS:
        # D: 1-D float64 points
        for n in (1, 16, 1_000, 100_000, 1_000_000):
            cell(f"D/{w}/n{n}", w, G((n,), seed=7))
        # T: point dtypes
        for dt in ("float32", "float16", "complex128", "int32", "int16", "uint8", "bool"):
            if (w == "cheb" and dt == "complex128") or (w == "lag" and dt == "bool"):
                continue
            for n in (1_000, 100_000, 1_000_000):
                cell(f"T/{w}/{dt}/n{n}", w, G((n,), dt, 7))
        # L: layouts @ 100K points
        for view, gshape, tag in (("slice:::2", (200_000,), "strided2"), ("slice:::-1", (100_000,), "reversed"),
                                  ("F", (316, 316), "forder"), ("T", (316, 316), "transposed"),
                                  ("T", (100, 100, 10), "transposed3d"), ("bcast:250,400", (400,), "rowbcast"),
                                  ("bcast:100000", (1,), "bcast")):
            cell(f"L/{w}/{tag}", w, G(gshape, seed=7, view=view))
        # S: small calls
        cell(f"S/{w}/pyfloat", w, G((), seed=7), pform="scalar")
        cell(f"S/{w}/zerod", w, G((), seed=7))
        cell(f"S/{w}/n3", w, G((3,), seed=7))
        # A: argument forms
        cell(f"A/{w}/pyint", w, G((), seed=7), pform="int")
        cell(f"A/{w}/typed/n1000", w, G((1_000,), seed=7), pform="typed")
        cell(f"A/{w}/npf16", w, G((), seed=7), pform="f16")


def main():
    """Pins the process, writes the manifest the C# runner replays, then prints NumPy's best per-call time per cell."""
    pin_affinity()
    DATA.mkdir(parents=True, exist_ok=True)
    build()
    (DATA / "manifest.json").write_text(json.dumps([{k: v for k, v in c.items() if k != "_call"} for c in CELLS],
                                                   indent=1), encoding="utf-8")
    print("cell\tnumpy_us")
    for c in CELLS:
        call, calls, rounds = c["_call"], c["calls"], c["rounds"]
        call()   # warm
        with np.errstate(all="ignore"):
            best = min(timeit.repeat(call, number=calls, repeat=rounds)) / calls
        print(f"{c['label']}\t{best * 1e6:.3f}", flush=True)


if __name__ == "__main__":
    with np.errstate(all="ignore"):
        main()
