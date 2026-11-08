"""NumPy 2.4.2 twin of polygauss_bench.cs — the numpy.polynomial Gauss-Legendre / -Laguerre / -Hermite quadrature
(U8: leggauss / laggauss / hermgauss / hermegauss) benchmark.

Writes data/polygauss/manifest.json (gitignored) — the cell list the C# side replays — then prints NumPy's best-of-R
per-call time for every cell (TSV: cell<TAB>us<TAB>eigvalsh us). Each cell records the SHA-256 of NumPy's two result
slots (+ dtype and shape), which the C# runner checks BEFORE it times the cell. Both sides run OpenBLAS at ONE thread:
eigvalsh's bits depend on the thread count, and so does its speed.

The third column is NumPy's time for the cell's np.linalg.eigvalsh call ALONE, on the companion matrix the function
builds ({p}companion of [0]*deg + [1]; the C# side builds the same matrix and times its own eigvalsh). Both libraries
call the same LAPACK syevd, so a cell's NPY/NS cannot exceed NumPy's whole call over NumSharp's eigvalsh time: the
report shows that ceiling next to every ratio.

Usage (from this directory; pin both sides with the same mask, never run them concurrently; three alternating pairs,
the report keeping each cell's best window of each side):
    set NS_PROBE_AFFINITY=0x14
    set OPENBLAS_NUM_THREADS=1
    set DOTNET_TC_CallCountingDelayMs=0
    python polygauss_numpy.py > numpy_gauss_times_1.tsv
    dotnet run -c Release --no-cache polygauss_bench.cs -- data/polygauss numpy_gauss_times_1.tsv > ns_gauss_1.tsv
    (the same for _2 and _3)
    python polygauss_report.py ns_gauss_1.tsv ns_gauss_2.tsv ns_gauss_3.tsv > polygauss_results.md
POLYGAUSS_ONLY=L/,G/ (comma-separated label prefixes) restricts both sides to matching cells.

Sections (cell-label prefixes):
    L  leggauss at degree 1 / 2 / 5 / 10 / 20 / 50 / 100 / 200 / 500
    G  laggauss at degree 1 / 2 / 5 / 10 / 20 / 50 / 100 / 200 / 500
    H  hermgauss at degree 1 / 2 / 5 / 10 / 20 / 50 / 100 / 200 / 500 / 1000 (1000: NumPy's own nodes and weights
       hold NaN — the normalised recurrence overflows)
    E  hermegauss at the same degrees
    A  argument forms at degree 20: an np.uint16 scalar (NumSharp's char); a Python bool (degree 1)
"""
import json
import os
import sys
import timeit
from pathlib import Path

assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "set OPENBLAS_NUM_THREADS=1 before Python starts"

import numpy as np  # noqa: E402
from numpy.polynomial import hermite as H  # noqa: E402
from numpy.polynomial import hermite_e as HE  # noqa: E402
from numpy.polynomial import laguerre as LG  # noqa: E402
from numpy.polynomial import legendre as L  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "polygauss"
sys.path.insert(0, str(HERE))
from polyeval_numpy import digest, pin_affinity  # noqa: E402  (the shared digest + host pinning)

FUNCS = {"leggauss": L.leggauss, "laggauss": LG.laggauss, "hermgauss": H.hermgauss, "hermegauss": HE.hermegauss}
# The companion each function hands np.linalg.eigvalsh (the shared syevd floor of both libraries).
COMPANIONS = {"leggauss": L.legcompanion, "laggauss": LG.lagcompanion, "hermgauss": H.hermcompanion,
              "hermegauss": HE.hermecompanion}
CELLS = []
ONLY = [p for p in os.environ.get("POLYGAUSS_ONLY", "").split(",") if p]


def loops_for(deg):
    """Picks the timing loop of a cell: enough calls per round to dwarf the timer, a few calls for the big degrees.

    Args:
        deg: The degree (the work grows as deg**3 in eigvalsh).

    Returns:
        (calls per round, rounds) — both sides time every cell with the same pair.
    """
    if deg >= 500:
        return 3, 7
    if deg >= 200:
        return 10, 7
    if deg >= 50:
        return 100, 9
    return 1000, 9


def cell(label, fn, deg, form=None):
    """Registers one cell, fn(deg) with the degree passed in the given form, NumPy's two result digests, and the cell's
    np.linalg.eigvalsh call alone (on the companion matrix fn builds), timed after it.

    Args:
        label: The cell's label ("<section>/<case>"); POLYGAUSS_ONLY restricts the run to matching labels.
        fn: leggauss, laggauss, hermgauss or hermegauss (the FUNCS key).
        deg: The degree.
        form: How the degree is passed — None (a Python int), "u16" (an np.uint16 scalar: NumSharp's char) or "bool"
            (a Python bool; deg must be 1).

    Returns:
        None.
    """
    if ONLY and not any(label.startswith(p) for p in ONLY):
        return
    f = FUNCS[fn]
    make = {None: lambda: deg, "u16": lambda: np.uint16(deg), "bool": lambda: bool(deg)}[form]
    arg = make()
    call = lambda: f(arg)
    x, w = call()
    # The eigvalsh call alone, on the companion matrix this function builds (the shared syevd floor).
    m = COMPANIONS[fn](np.array([0] * deg + [1], dtype=np.float64))
    ecall = lambda: np.linalg.eigvalsh(m)
    calls, rounds = loops_for(deg)
    spec = {"label": label, "fn": fn, "deg": deg, "expect": [digest(x), digest(w)], "calls": calls, "rounds": rounds}
    if form:
        spec["form"] = form
    CELLS.append({**spec, "_call": call, "_eig": ecall})


def build():
    """Registers every cell of the five sections (see the module docstring).

    Returns:
        None.
    """
    for deg in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        cell(f"L/deg{deg}", "leggauss", deg)
    for deg in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        cell(f"G/deg{deg}", "laggauss", deg)
    for deg in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        cell(f"H/deg{deg}", "hermgauss", deg)
    for deg in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        cell(f"E/deg{deg}", "hermegauss", deg)
    for fn in FUNCS:
        cell(f"A/{fn}/npu16", fn, 20, "u16")
        cell(f"A/{fn}/pybool", fn, 1, "bool")


def main():
    """Pins the process, writes the manifest the C# runner replays, then prints NumPy's best per-call time per cell and
    the best time of the cell's eigvalsh call alone.

    Returns:
        None.
    """
    pin_affinity()
    DATA.mkdir(parents=True, exist_ok=True)
    build()
    (DATA / "manifest.json").write_text(json.dumps([{k: v for k, v in c.items() if not k.startswith("_")}
                                                    for c in CELLS], indent=1), encoding="utf-8")
    print("cell\tnumpy_us\tnumpy_eigvalsh_us")
    for c in CELLS:
        call, ecall, calls, rounds = c["_call"], c["_eig"], c["calls"], c["rounds"]
        call()   # warm
        best = min(timeit.repeat(call, number=calls, repeat=rounds)) / calls
        ecall()
        best_e = min(timeit.repeat(ecall, number=calls, repeat=rounds)) / calls
        print(f"{c['label']}\t{best * 1e6:.3f}\t{best_e * 1e6:.3f}", flush=True)


if __name__ == "__main__":
    main()
