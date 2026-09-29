"""NumPy 2.4.2 twin of polycalc_bench.cs — the numpy.polynomial calculus family (U4: {p}der / {p}int) benchmark.

Writes data/polycalc/manifest.json (gitignored) — the cell list the C# side replays — then prints NumPy's
best-of-R per-call time for every cell (TSV: cell<TAB>us). Inputs come from the exact shared formula of
polyeval_numpy.py (gen(), reused here), and each cell records the SHA-256 of NumPy's result bytes (+ dtype and
shape), which the C# runner checks BEFORE it times the cell.

Usage (from this directory; pin both sides with the same mask, never run them concurrently):
    set NS_PROBE_AFFINITY=0xF0
    python polycalc_numpy.py > numpy_calc_times.tsv
    set DOTNET_TC_CallCountingDelayMs=0
    dotnet run -c Release --no-cache polycalc_bench.cs -- data/polycalc numpy_calc_times.tsv > ns_calc.tsv
    python polyeval_report.py ns_calc.tsv > polycalc_results.md
POLYCALC_ONLY=N/ (comma-separated label prefixes) restricts both sides to matching cells.

Sections (cell-label prefixes; every section covers all six bases, derivatives and integrals):
    D  1-D float64 series: length 4 / 11 / 100 / 1000 x order 1 / 3 — NumPy's Python loop over the coefficients
    P  parameters: scl (weak / 0-d array), k (list of weak scalars / typed array), lbnd (weak / complex / 0-d)
       on a 1-D and an N-D series
    N  N-D float64 series: (8, 1K), (8, 100K), (32, 10K), (4, 1M) along axis 0; (1000, 8) and (100K, 8) along
       axis 1; 3-D (8, 100, 100) along each axis
    T  dtypes: float32 / float16 / complex128 / int32 / int64 / uint8 / bool series, N-D (11, 100K) and 1-D (11,)
    L  layouts: F order, column-strided, reversed rows, a transposed 3-D view, a broadcast series
    M  orders: m = 1 / 2 / 5 / 10 on (16, 100K) — the fused multi-order derivative and the per-order integral
    S  small N-D series: (3, 3), (11, 10), (11, 100) — the fixed per-call cost
"""
import hashlib
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
from numpy.polynomial import polynomial as P  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "polycalc"
sys.path.insert(0, str(HERE))
from polyeval_numpy import G, digest, pin_affinity  # noqa: E402  (the shared input formula + view grammar)

MODULES = {"poly": P, "cheb": C, "leg": L, "lag": LG, "herm": H, "herme": HE}
MODNAME = {"poly": "polynomial", "cheb": "chebyshev", "leg": "legendre", "lag": "laguerre",
           "herm": "hermite", "herme": "hermite_e"}
CELLS = []
ONLY = [p for p in os.environ.get("POLYCALC_ONLY", "").split(",") if p]


def loops_for(work):
    """(calls per round, rounds): enough calls per round to dwarf the timer, a few calls for the big cells."""
    if work >= 5_000_000:
        return 1, 5
    if work >= 200_000:
        return 3, 7
    if work >= 20_000:
        return 20, 7
    return 200, 9


def arg(v):
    """(manifest spec, Python value) of one argument: a G array, a weak Python scalar, or a list of those."""
    if isinstance(v, G):
        return {"gen": v.spec}, v.array
    if isinstance(v, list):
        items = [arg(x) for x in v]
        return {"list": [s for s, _ in items]}, [x for _, x in items]
    if isinstance(v, bool):
        return {"weak": "bool", "value": v}, v
    if isinstance(v, int):
        return {"weak": "int", "value": v}, v
    if isinstance(v, complex):
        return {"weak": "complex", "re": v.real, "im": v.imag}, v
    if isinstance(v, float):
        return {"weak": "float", "value": v}, v
    raise TypeError(type(v))


def cell(label, base, kind, c, m=1, axis=0, scl=None, k=None, lbnd=None):
    """Register one cell: {base}{kind}(c, m, [k, lbnd,] scl, axis); None = NumPy's default (the argument is not
    passed). POLYCALC_ONLY=prefix[,prefix...] restricts the run (and the manifest) to matching labels."""
    if ONLY and not any(label.startswith(p) for p in ONLY):
        return
    fn = getattr(MODULES[base], base + kind)
    carr = c.array
    kw, spec = {"axis": axis}, {}
    for name, v in (("scl", scl), ("k", k), ("lbnd", lbnd)):
        if v is not None:
            s, py = arg(v)
            spec[name] = s
            kw[name] = py
    call = lambda: fn(carr, m, **kw)
    ref = np.asarray(call())
    work = max(carr.size, 1) * m
    calls, rounds = loops_for(work)
    CELLS.append({"label": label, "module": MODNAME[base], "base": base, "kind": kind, "c": c.spec, "m": m,
                  "axis": axis, **spec, "expect": digest(ref), "calls": calls, "rounds": rounds, "_call": call})


def build():
    bases = list(MODULES)
    for b in bases:
        # D: 1-D float64 series (NumPy runs a Python loop over the coefficients)
        for n in (4, 11, 100, 1000):
            for m in (1, 3):
                cell(f"D/{b}/der/n{n}/m{m}", b, "der", G((n,), seed=7), m)
                cell(f"D/{b}/int/n{n}/m{m}", b, "int", G((n,), seed=7), m)
        # P: parameters on a 1-D (11,) and an N-D (11, 10K) series
        for shape, tag in (((11,), "1d"), ((11, 10_000), "nd")):
            c = G(shape, seed=7)
            cell(f"P/{b}/der/scl/{tag}", b, "der", c, 2, scl=0.75)
            cell(f"P/{b}/der/scl0d/{tag}", b, "der", c, 2, scl=G((), seed=3))
            cell(f"P/{b}/int/k/{tag}", b, "int", c, 2, k=[0.5, -1.25])
            cell(f"P/{b}/int/lbnd/{tag}", b, "int", c, 2, lbnd=0.3)
            cell(f"P/{b}/int/all/{tag}", b, "int", c, 2, k=[0.5, -1.25], lbnd=0.3, scl=0.75)
            cell(f"P/{b}/int/lbnd0d/{tag}", b, "int", c, 1, lbnd=G((), seed=3))
            cell(f"P/{b}/int/karr/{tag}", b, "int", c, 2, k=G((2,), seed=4))
        cell(f"P/{b}/int/lbndc/1d", b, "int", G((11,), seed=7), 2, lbnd=0.3 - 0.2j)
        # N: N-D float64 series
        for shape, axis in (((8, 1_000), 0), ((8, 100_000), 0), ((32, 10_000), 0), ((4, 1_000_000), 0),
                            ((1_000, 8), 1), ((100_000, 8), 1), ((8, 100, 100), 0), ((100, 8, 100), 1),
                            ((100, 100, 8), 2)):
            tag = "x".join(str(d) for d in shape) + f"/ax{axis}"
            cell(f"N/{b}/der/{tag}", b, "der", G(shape, seed=7), 1, axis)
            cell(f"N/{b}/int/{tag}", b, "int", G(shape, seed=7), 1, axis)
        # T: dtypes
        for dt in ("float32", "float16", "complex128", "int32", "int64", "uint8", "bool"):
            cell(f"T/{b}/der/{dt}/nd", b, "der", G((11, 100_000), dt, 7), 2)
            cell(f"T/{b}/int/{dt}/nd", b, "int", G((11, 100_000), dt, 7), 2, k=[0.5], lbnd=0.3)
            cell(f"T/{b}/der/{dt}/1d", b, "der", G((11,), dt, 7), 2)
            cell(f"T/{b}/int/{dt}/1d", b, "int", G((11,), dt, 7), 2, k=[0.5], lbnd=0.3)
        # L: layouts @ (11, 100K)
        for view, gshape, tag in (("F", (11, 100_000), "forder"), ("slice::,::2", (11, 200_000), "colstride2"),
                                  ("slice:::-1", (11, 100_000), "revrows"), ("T", (100_000, 11), "transposed"),
                                  ("bcast:11,100000", (1, 100_000), "bcast")):
            cell(f"L/{b}/der/{tag}", b, "der", G(gshape, seed=7, view=view), 2)
            cell(f"L/{b}/int/{tag}", b, "int", G(gshape, seed=7, view=view), 2)
        # M: orders
        for m in (1, 2, 5, 10):
            cell(f"M/{b}/der/m{m}", b, "der", G((16, 100_000), seed=7), m)
            cell(f"M/{b}/int/m{m}", b, "int", G((16, 100_000), seed=7), m)
        # S: small N-D series (fixed per-call cost)
        for shape in ((3, 3), (11, 10), (11, 100)):
            tag = "x".join(str(d) for d in shape)
            cell(f"S/{b}/der/{tag}", b, "der", G(shape, seed=7), 1)
            cell(f"S/{b}/int/{tag}", b, "int", G(shape, seed=7), 1)


def main():
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
