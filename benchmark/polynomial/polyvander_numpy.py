"""NumPy 2.4.2 twin of polyvander_bench.cs — the numpy.polynomial Vandermonde family (U5: {p}vander / {p}vander2d /
{p}vander3d) benchmark.

Writes data/polyvander/manifest.json (gitignored) — the cell list the C# side replays — then prints NumPy's
best-of-R per-call time for every cell (TSV: cell<TAB>us). Inputs come from the exact shared formula of
polyeval_numpy.py (gen(), reused here), and each cell records the SHA-256 of NumPy's result bytes (+ dtype and
shape), which the C# runner checks BEFORE it times the cell.

Usage (from this directory; pin both sides with the same mask, never run them concurrently):
    set NS_PROBE_AFFINITY=0xF0
    python polyvander_numpy.py > numpy_vander_times.tsv
    set DOTNET_TC_CallCountingDelayMs=0
    dotnet run -c Release --no-cache polyvander_bench.cs -- data/polyvander numpy_vander_times.tsv > ns_vander.tsv
    python polyvander_report.py ns_vander.tsv > polyvander_results.md
POLYVANDER_ONLY=D/,V/ (comma-separated label prefixes) restricts both sides to matching cells.

Sections (cell-label prefixes; every section covers all six bases):
    D  1-D float64 points: 1 / 16 / 1K / 100K / 1M points x degree 3 / 10 / 30 (30 up to 100K)
    T  point dtypes: float32 / float16 / complex128 / int32 / int64 / uint8 / bool at 1K and 100K points, degree 5
    L  layouts @ 100K points, degree 5: strided, reversed, F-order and transposed 2-D, broadcast
    N  N-D points: (100, 1000) degree 5, (10, 10, 1000) degree 3
    V  2-D outer products: 1 / 100 / 10K / 100K points x degrees (1,1) / (3,3) / (5,5) / (10,10)
    W  3-D outer products: 1 / 100 / 10K / 100K points x degrees (1,1,1) / (2,2,2) / (3,3,3); (5,5,5) up to 10K
    M  2-D mixed / other dtypes @ 10K points, degrees (3, 3): float32+int32 (float64), float16+int8 (float16),
       float32, complex128, and a strided / reversed pair
    S  small calls (the fixed per-call cost): a scalar x, 3 points; 2-D / 3-D scalar points
    A  argument forms: a Python-list x (16 / 1000 points), Python-list 2-D points, a tuple / an ndarray of degrees; the
       C#-only kinds (a typed double[] x — an ndarray to NumPy — a List<double> x / 2-D points — a list — a 0-d array
       degree), 1000-point Python-list 2-D points, and the object stack of scalars (a Python int past uint64 among 2-D / 3-D
       scalar points, one dtype and mixed dtypes)
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
DATA = HERE / "data" / "polyvander"
sys.path.insert(0, str(HERE))
from polyeval_numpy import G, digest, pin_affinity  # noqa: E402  (the shared input formula + view grammar)

MODULES = {"poly": P, "cheb": C, "leg": L, "lag": LG, "herm": H, "herme": HE}
MODNAME = {"poly": "polynomial", "cheb": "chebyshev", "leg": "legendre", "lag": "laguerre",
           "herm": "hermite", "herme": "hermite_e"}
CELLS = []
ONLY = [p for p in os.environ.get("POLYVANDER_ONLY", "").split(",") if p]


def loops_for(work):
    """(calls per round, rounds): enough calls per round to dwarf the timer, a few calls for the big cells. The biggest
    cells get 9 single-call rounds: their results are fresh pages whose demand-zero faults dominate both sides (~0.44 us
    a page on the dev host) and swing with the OS's zeroed-page list, so a best-of-5 there measured the host, not the
    kernel."""
    if work >= 5_000_000:
        return 1, 9
    if work >= 200_000:
        return 3, 7
    if work >= 20_000:
        return 20, 7
    return 200, 9


def cell(label, base, form, points, deg, pform=None, degform=None):
    """Register one cell: {base}vander{form}(*points, deg). points are G arrays; pform passes them as Python values —
    "list" (a.tolist(): nested Python floats, the C# side an object[] of boxed doubles) or "scalar" (a 0-d G as a
    Python float) — the conversion being part of the timed call on both sides, as it is part of NumPy's np.array(x).
    A 2-D / 3-D degree is a Python list of ints unless degform says "tuple" (a tuple; C# a ValueTuple) or "array"
    (an int64 ndarray; C# an int[] — the C# boundary map's typed array, which NumPy iterates as NumPy int scalars).
    POLYVANDER_ONLY=prefix[,prefix...] restricts the run (and the manifest) to matching labels."""
    if ONLY and not any(label.startswith(p) for p in ONLY):
        return
    suffix = "" if form == "1d" else form
    fn = getattr(MODULES[base], base + "vander" + suffix)
    arrays = [p.array for p in points]
    if pform in ("list", "glist"):
        # glist: the C# side passes a List<double> — a Python list of floats to NumPy, like "list"'s object[]
        args = [a.tolist() for a in arrays]
    elif pform == "scalar":
        args = [float(a) for a in arrays]
    else:
        # "typed": the C# side passes a typed double[] — an ndarray to NumPy (the house boundary map)
        args = arrays
    degarg = (tuple(deg) if degform == "tuple" else np.array(deg) if degform in ("array", "nd0") else deg)
    call = lambda: fn(*args, degarg)
    ref = np.asarray(call())
    npts = max(int(np.prod(arrays[0].shape)), 1)
    rows = (deg + 1) if isinstance(deg, int) else int(np.prod([d + 1 for d in deg]))
    calls, rounds = loops_for(npts * rows)
    spec = {"label": label, "module": MODNAME[base], "base": base, "form": form, "points": [p.spec for p in points],
            "deg": deg if isinstance(deg, int) else list(deg), "expect": digest(ref), "calls": calls, "rounds": rounds}
    if degform:
        spec["degform"] = degform
    if pform:
        spec["pform"] = pform
    CELLS.append({**spec, "_call": call})


def special(label, base, kind):
    """Register a cell whose arguments are not G arrays: the object stack of scalars. objstack2d is
    {p}vander2d(2**70, 1.5, [3, 3]) — both points become Python floats, one dtype (the kernel path on the C# side);
    objstack3d_mixed is {p}vander3d(2**70, np.float16(0.5), np.array(1.5, np.float32), [2, 2, 2]) — float64, float16 and
    float32 matrices multiplied with promotion (the C# side's np.multiply composition)."""
    if ONLY and not any(label.startswith(p) for p in ONLY):
        return
    m = MODULES[base]
    if kind == "objstack2d":
        fn = getattr(m, base + "vander2d")
        call = lambda: fn(2 ** 70, 1.5, [3, 3])
    elif kind == "objstack3d_mixed":
        fn = getattr(m, base + "vander3d")
        f32 = np.array(1.5, np.float32)
        call = lambda: fn(2 ** 70, np.float16(0.5), f32, [2, 2, 2])
    else:
        raise ValueError(kind)
    ref = np.asarray(call())
    calls, rounds = loops_for(1)
    CELLS.append({"label": label, "module": MODNAME[base], "base": base, "form": "special", "special": kind, "points": [],
                  "deg": 0, "expect": digest(ref), "calls": calls, "rounds": rounds, "_call": call})


def build():
    for b in MODULES:
        # D: 1-D float64 points
        for n in (1, 16, 1_000, 100_000, 1_000_000):
            for d in (3, 10, 30):
                if n == 1_000_000 and d == 30:
                    continue
                cell(f"D/{b}/n{n}/d{d}", b, "1d", [G((n,), seed=7)], d)
        # T: point dtypes
        for dt in ("float32", "float16", "complex128", "int32", "int64", "uint8", "bool"):
            for n in (1_000, 100_000):
                cell(f"T/{b}/{dt}/n{n}", b, "1d", [G((n,), dt, 7)], 5)
        # L: layouts @ 100K points
        for view, gshape, tag in (("slice:::2", (200_000,), "strided2"), ("slice:::-1", (100_000,), "reversed"),
                                  ("F", (316, 316), "forder"), ("T", (316, 316), "transposed"),
                                  ("bcast:100000", (1,), "bcast")):
            cell(f"L/{b}/{tag}", b, "1d", [G(gshape, seed=7, view=view)], 5)
        # N: N-D points
        cell(f"N/{b}/100x1000/d5", b, "1d", [G((100, 1_000), seed=7)], 5)
        cell(f"N/{b}/10x10x1000/d3", b, "1d", [G((10, 10, 1_000), seed=7)], 3)
        # V: 2-D outer products
        for n in (1, 100, 10_000, 100_000):
            for dg in ((1, 1), (3, 3), (5, 5), (10, 10)):
                cell(f"V/{b}/n{n}/d{dg[0]}x{dg[1]}", b, "2d", [G((n,), seed=7), G((n,), seed=8)], list(dg))
        # W: 3-D outer products
        for n in (1, 100, 10_000, 100_000):
            for dg in ((1, 1, 1), (2, 2, 2), (3, 3, 3), (5, 5, 5)):
                if n == 100_000 and dg[0] == 5:
                    continue
                cell(f"W/{b}/n{n}/d{dg[0]}x{dg[1]}x{dg[2]}", b, "3d", [G((n,), seed=7), G((n,), seed=8), G((n,), seed=9)],
                     list(dg))
        # M: mixed / other dtypes and layouts, 2-D @ 10K points
        for tag, gx, gy in (("f32_i32", G((10_000,), "float32", 7), G((10_000,), "int32", 8)),
                            ("f16_i8", G((10_000,), "float16", 7), G((10_000,), "int8", 8)),
                            ("f32", G((10_000,), "float32", 7), G((10_000,), "float32", 8)),
                            ("c128", G((10_000,), "complex128", 7), G((10_000,), "complex128", 8)),
                            ("strided_rev", G((20_000,), seed=7, view="slice:::2"), G((10_000,), seed=8, view="slice:::-1"))):
            cell(f"M/{b}/{tag}", b, "2d", [gx, gy], [3, 3])
        # S: small calls
        cell(f"S/{b}/scalar/d3", b, "1d", [G((), seed=7)], 3, pform="scalar")
        cell(f"S/{b}/n3/d3", b, "1d", [G((3,), seed=7)], 3)
        cell(f"S/{b}/2d_scalars/d2x2", b, "2d", [G((), seed=7), G((), seed=8)], [2, 2], pform="scalar")
        cell(f"S/{b}/3d_scalars/d1x1x1", b, "3d", [G((), seed=7), G((), seed=8), G((), seed=9)], [1, 1, 1], pform="scalar")
        # A: argument forms
        cell(f"A/{b}/list/n16", b, "1d", [G((16,), seed=7)], 5, pform="list")
        cell(f"A/{b}/list/n1000", b, "1d", [G((1_000,), seed=7)], 5, pform="list")
        cell(f"A/{b}/list2d/n100", b, "2d", [G((100,), seed=7), G((100,), seed=8)], [3, 3], pform="list")
        cell(f"A/{b}/list2d/n1000", b, "2d", [G((1_000,), seed=7), G((1_000,), seed=8)], [3, 3], pform="list")
        cell(f"A/{b}/degtuple/n1000", b, "2d", [G((1_000,), seed=7), G((1_000,), seed=8)], [3, 2], degform="tuple")
        cell(f"A/{b}/degarray/n1000", b, "2d", [G((1_000,), seed=7), G((1_000,), seed=8)], [3, 2], degform="array")
        cell(f"A/{b}/typed/n1000", b, "1d", [G((1_000,), seed=7)], 5, pform="typed")
        cell(f"A/{b}/glist/n1000", b, "1d", [G((1_000,), seed=7)], 5, pform="glist")
        cell(f"A/{b}/glist2d/n1000", b, "2d", [G((1_000,), seed=7), G((1_000,), seed=8)], [3, 2], pform="glist")
        cell(f"A/{b}/deg_nd0/n1000", b, "1d", [G((1_000,), seed=7)], 5, degform="nd0")
        special(f"A/{b}/objstack2d", b, "objstack2d")
        special(f"A/{b}/objstack3d_mixed", b, "objstack3d_mixed")


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
