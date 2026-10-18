"""NumPy 2.4.2 twin of polyvalfromroots_bench.cs — numpy.polynomial.polynomial.polyvalfromroots (unit 6 of
docs/plans/numpy-polynomial-remaining-compute.md): NumPy's np.prod(x - r, axis=0) against NumSharp's one-pass root product.

Writes data/polyvalfromroots/manifest.json (gitignored) — the cell list the C# side replays, inputs regenerated from the
shared exact formula (polyeval_numpy.gen) — then prints NumPy's best-of-R per-call time for every cell (TSV:
cell<TAB>us). Each cell records the SHA-256 of NumPy's result (+ dtype and shape), which the C# runner checks BEFORE it
times the cell.

Usage (from this directory; pin both sides with the same mask, never run them concurrently):
    set NS_PROBE_AFFINITY=0x14
    python polyvalfromroots_numpy.py > numpy_vfr_times.tsv
    set DOTNET_TC_CallCountingDelayMs=0
    dotnet run -c Release --no-cache polyvalfromroots_bench.cs -- data/polyvalfromroots numpy_vfr_times.tsv > ns_vfr.tsv
    python polyvalfromroots_report.py ns_vfr.tsv > polyvalfromroots_results.md
POLYVFR_ONLY=F/,T/ (comma-separated label prefixes) restricts both sides to matching cells.

Sections (cell-label prefixes):
    F  float64, a 1-D x and 1-D roots: 1K / 100K / 1M points x 3 / 10 / 30 roots
    T  dtypes at 100K points, 10 roots (x dtype - r dtype), and the same-dtype pairs at 1K / 1M
    L  x layouts at 100K points, 10 roots: strided, reversed, 2-D C / transposed / F, broadcast, 3-D, strided 2-D
    N  N-D roots: tensor (r (10, 8) at 100K points, r (10, 64) at 1K), one series per point (tensor=False, r C and F)
    S  a scalar x (Python float / complex, 0-d array): a 1-D r (a 0-d result) and r's columns (C and F)
    H  float16 product forms: the slab (a narrow per root) and the reduce form (a float32 accumulation)
    C  complex128 product forms: the slab (simd_cmul) and the reduce form (the contracted fallback product)
    E  edges: no roots, one root, ten points
    R  argument kinds: roots as a Python list of floats / ints, x as a Python list
"""
import json
import os
import sys
import timeit
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial as P

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "polyvalfromroots"
sys.path.insert(0, str(HERE))
from polyeval_numpy import G, digest, loops_for, pin_affinity  # noqa: E402  (shared inputs, digest, loop sizing, pinning)

CELLS = []
ONLY = [p for p in os.environ.get("POLYVFR_ONLY", "").split(",") if p]


def cell(label, x, r, tensor=True, work=None):
    """Registers one cell, P.polyvalfromroots(x, r, tensor), and NumPy's result digest.

    Args:
        label: The cell's label ("<section>/<case>"); POLYVFR_ONLY restricts the run to matching labels.
        x: The points: a G (a generated array), ("weak", python_scalar) or ("list", G) — the G's values as a Python list.
        r: The roots: a G, or ("list", G) / ("intlist", n) — Python lists (the C# side builds object[]).
        tensor: NumPy's tensor flag.
        work: The cell's work estimate for loops_for (default: result size x root count).

    Returns:
        None.
    """
    if ONLY and not any(label.startswith(p) for p in ONLY):
        return

    def build(a):
        if isinstance(a, G):
            return {"gen": a.spec}, a.array
        kind, v = a
        if kind == "weak":
            if isinstance(v, complex):
                return {"weak": "complex", "re": v.real, "im": v.imag}, v
            return {"weak": "float", "value": v}, v
        if kind == "list":
            return {"list": v.spec}, [float(e) for e in v.array.ravel()]
        if kind == "intlist":
            return {"intlist": v}, list(range(1, v + 1))
        raise ValueError(kind)

    xs, xv = build(x)
    rs, rv = build(r)
    call = lambda: P.polyvalfromroots(xv, rv, tensor=tensor)
    ref = np.asarray(call())
    nr = np.asarray(rv).shape[0] if np.ndim(rv) else 1
    if work is None:
        work = max(ref.size, 1) * max(nr, 1)
    k, rounds = loops_for(work)
    CELLS.append({"label": label, "x": xs, "r": rs, "tensor": tensor, "expect": digest(ref), "k": k, "r_": rounds,
                  "_call": call})


def build():
    """Registers every cell of the sections listed in the module docstring.

    Returns:
        None.
    """
    for nr in (3, 10, 30):
        for n in (1_000, 100_000, 1_000_000):
            cell(f"F/r{nr}/{n}", G((n,)), G((nr,), seed=7))
    pairs = [("float32", "float32"), ("float16", "float16"), ("complex128", "complex128"), ("float32", "float64"),
             ("float64", "float32"), ("float16", "float32"), ("int64", "float64"), ("int32", "float64"), ("int8", "float64"),
             ("uint16", "float64"), ("bool", "float64"), ("float64", "complex128"), ("complex128", "float64"),
             ("float64", "int64"), ("int32", "int32"), ("float64", "bool")]
    for tx, tr in pairs:
        cell(f"T/{tx}-{tr}/100000", G((100_000,), tx, 1), G((10,), tr, 7))
    for tx in ("float32", "float16", "complex128"):
        for n in (1_000, 1_000_000):
            cell(f"T/{tx}-{tx}/{n}", G((n,), tx, 1), G((10,), tx, 7))
    lay = [("strided2", G((200_000,), view="slice:::2")), ("reversed", G((100_000,), view="slice:::-1")),
           ("c2d", G((317, 317))), ("transposed", G((317, 317), view="T")), ("forder", G((317, 317), view="F")),
           ("bcast", G((1_000,), view="bcast:100,1000")), ("c3d", G((40, 50, 50))),
           ("strided2d", G((317, 634), view="slice::,::2"))]
    for name, xg in lay:
        cell(f"L/{name}", xg, G((10,), seed=7))
    cell("N/tensor8x100k", G((100_000,)), G((10, 8), seed=7))
    cell("N/tensor64x1k", G((1_000,)), G((10, 64), seed=7))
    cell("N/perpointC100k", G((100_000,)), G((10, 100_000), seed=7), tensor=False)
    cell("N/perpointF100k", G((100_000,)), G((100_000, 10), seed=7, view="T"), tensor=False)
    cell("N/c128perpointC10k", G((10_000,), "complex128"), G((10, 10_000), "complex128", 7), tensor=False)
    cell("N/c128perpointF10k", G((10_000,), "complex128"), G((10_000, 10), "complex128", 7, view="T"), tensor=False)
    cell("S/weakf/r10", ("weak", 0.37), G((10,), seed=7), work=10)
    cell("S/weakc/r10", ("weak", 0.25 - 0.5j), G((10,), "complex128", 7), work=10)
    cell("S/weakf/f16r10", ("weak", 0.37), G((10,), "float16", 7), work=10)
    cell("S/arr0d/r10", G(()), G((10,), seed=7), work=10)
    cell("S/weakf/r100", ("weak", 0.37), G((100,), seed=7), work=100)
    cell("S/weakf/colsC100k", ("weak", 0.37), G((10, 100_000), seed=7))
    cell("S/weakf/colsF100k", ("weak", 0.37), G((100_000, 10), seed=7, view="T"))
    cell("S/weakc/colsF10k", ("weak", 0.25 - 0.5j), G((10_000, 10), "complex128", 7, view="T"))
    cell("H/slab100k", G((100_000,), "float16", 1), G((10,), "float16", 7))
    cell("H/reduceF100k", G((100_000,), "float16", 1), G((100_000, 10), "float16", 7, view="T"), tensor=False)
    cell("H/onepoint", G((1,), "float16", 1), G((10,), "float16", 7), work=10)
    cell("C/slab100k", G((100_000,), "complex128", 1), G((10,), "complex128", 7))
    cell("C/reduceF100k", G((100_000,), "complex128", 1), G((100_000, 10), "complex128", 7, view="T"), tensor=False)
    cell("C/scalar_r100", ("weak", 0.25 - 0.5j), G((100,), "complex128", 7), work=100)
    cell("E/r0/100k", G((100_000,)), G((0,), seed=7), work=100_000)
    cell("E/r1/100k", G((100_000,)), G((1,), seed=7))
    cell("E/n10/r3", G((10,)), G((3,), seed=7))
    cell("R/rlist10", G((100_000,)), ("list", G((10,), seed=7)))
    cell("R/rint10", G((100_000,)), ("intlist", 10))
    cell("R/xlist1k", ("list", G((1_000,))), G((10,), seed=7))


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
        call, k, rounds = c["_call"], c["k"], c["r_"]
        call()   # warm
        best = min(timeit.repeat(call, number=k, repeat=rounds)) / k
        print(f"{c['label']}\t{best * 1e6:.3f}", flush=True)


if __name__ == "__main__":
    main()
