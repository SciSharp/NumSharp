"""NumPy 2.4.2 twin of polyroots_bench.cs — the numpy.polynomial companion / roots (U7) benchmark: {p}companion and
{p}roots for the six bases.

Writes data/polyroots/manifest.json (gitignored) — the cell list the C# side replays — then prints NumPy's best-of-R
per-call time for every cell (TSV: cell<TAB>us). Inputs come from the exact shared formula of polyeval_numpy.py
(gen() and its view grammar, reused here). Each cell records the SHA-256 of NumPy's result bytes (+ dtype and shape),
which the C# runner checks BEFORE it times the cell. A float32 series whose roots are complex makes NumPy return
complex64; NumSharp has one complex width (#569) and carries exactly those values as complex128 (np.linalg.eigvals
rounds a float32 operand's complex result per component), so such a cell's digest is taken of the result widened to
complex128 and flagged "c64". A roots cell that reaches LAPACK geev (degree 2 or more) is flagged "lapack" and also
carries NumPy's values: byte parity needs the scipy-openblas NumPy itself loads, at one thread — the C# runner enables
NumSharp.Interop.OpenBLAS that way and, on a host whose OpenBLAS build differs, accepts values within a relative 1e-12
("close") instead.

Usage (from this directory; pin both sides with the same mask, never run them concurrently):
    set NS_PROBE_AFFINITY=0xF0
    python polyroots_numpy.py > numpy_roots_times.tsv
    set DOTNET_TC_CallCountingDelayMs=0
    dotnet run -c Release --no-cache polyroots_bench.cs -- data/polyroots numpy_roots_times.tsv > ns_roots.tsv
    python polyroots_report.py ns_roots.tsv > polyroots_results.md
POLYROOTS_ONLY=X/ (comma-separated label prefixes) restricts both sides to matching cells.

Sections (cell-label prefixes):
    C  {p}companion, float64: series of 2 (the linear root's 1x1 matrix) / 3 / 6 / 11 / 51 / 201 / 501 / 1001
       coefficients, and — power series only — 2898 (a 67 MB matrix: past the 64 MiB write-once limit, OS-zeroed pages)
    T  {p}companion in float32 / float16 / complex128 / int64: 11 / 51 / 201 coefficients
    R  {p}roots, float64: a constant (1), a linear (2) and 3 / 6 / 11 / 51 / 201-coefficient series
    S  {p}roots in float32 / complex128 (11, 51), int64 (11) and float16 (linear: scalarmath)
    L  layouts: companion of a stride-2 / reversed 51-coefficient series, roots of a stride-2 11-coefficient one
    A  argument forms: a Python list of 11 floats to companion and roots
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
DATA = HERE / "data" / "polyroots"
sys.path.insert(0, str(HERE))
from polyeval_numpy import G, pin_affinity  # noqa: E402  (the shared input formula + view grammar)

MODULES = {"poly": P, "cheb": C, "leg": L, "lag": LG, "herm": H, "herme": HE}
MODNAME = {"poly": "polynomial", "cheb": "chebyshev", "leg": "legendre", "lag": "laguerre",
           "herm": "hermite", "herme": "hermite_e"}
CELLS = []
ONLY = [p for p in os.environ.get("POLYROOTS_ONLY", "").split(",") if p]


def digest(r):
    """dtype, shape and the SHA-256 of the bytes (logical C order); a complex64 result widened to complex128 first."""
    r = np.asarray(r)
    c64 = r.dtype == np.complex64
    if c64:
        r = r.astype(np.complex128)
    h = hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest()
    d = {"dtype": r.dtype.name, "shape": [int(x) for x in r.shape], "sha256": h}
    if c64:
        d["c64"] = True
    return d


def loops_for(call):
    """(calls per round, rounds) from one timed call: ~50 ms rounds (at least one call), 7 rounds — 3 for a call of
    100 ms or more."""
    t = min(timeit.repeat(call, number=1, repeat=2))
    calls = max(1, int(0.05 / max(t, 1e-9)))
    return min(calls, 2000), (3 if t >= 0.1 else 7)


def cell(label, base, fn, c, form=None):
    """Register one cell: {module}.{fn}(c). c is a G array — passed as the array, or (form "list") as its tolist(), a
    Python list of Python floats whose conversion is part of the timed call on both sides."""
    if ONLY and not any(label.startswith(p) for p in ONLY):
        return
    f = getattr(MODULES[base], fn)
    arg = c.array.tolist() if form == "list" else c.array
    call = lambda: f(arg)
    ref = call()
    expect = digest(ref)
    lapack = fn.endswith("roots") and c.array.size >= 3
    if lapack:
        r = np.asarray(ref)
        expect["hex"] = np.ascontiguousarray(r.astype(np.complex128) if r.dtype == np.complex64 else r).tobytes().hex()
    calls, rounds = loops_for(call)
    spec = {"gen": c.spec}
    if form:
        spec["form"] = form
    CELLS.append({"label": label, "module": MODNAME[base], "base": base, "fn": fn, "arg": spec, "expect": expect,
                  "lapack": lapack, "calls": calls, "rounds": rounds, "_call": call})


def build():
    for b in MODULES:
        comp, roots = b + "companion", b + "roots"
        # C: companion float64
        for n in (2, 3, 6, 11, 51, 201, 501, 1001) + ((2898,) if b == "poly" else ()):
            cell(f"C/{b}/float64/n{n}", b, comp, G((n,), seed=7))
        # T: companion dtypes
        for dt in ("float32", "float16", "complex128", "int64"):
            for n in (11, 51, 201):
                cell(f"T/{b}/{dt}/n{n}", b, comp, G((n,), dt, 7))
        # R: roots float64
        for n in (1, 2, 3, 6, 11, 51, 201):
            cell(f"R/{b}/float64/n{n}", b, roots, G((n,), seed=7))
        # S: roots dtypes
        for dt in ("float32", "complex128"):
            for n in (11, 51):
                cell(f"S/{b}/{dt}/n{n}", b, roots, G((n,), dt, 7))
        cell(f"S/{b}/int64/n11", b, roots, G((11,), "int64", 7))
        cell(f"S/{b}/float16/n2", b, roots, G((2,), "float16", 7))
        # L: layouts
        cell(f"L/{b}/companion/strided51", b, comp, G((102,), seed=7, view="slice:::2"))
        cell(f"L/{b}/companion/reversed51", b, comp, G((51,), seed=7, view="slice:::-1"))
        cell(f"L/{b}/roots/strided11", b, roots, G((22,), seed=7, view="slice:::2"))
        # A: Python lists
        cell(f"A/{b}/companion/list11", b, comp, G((11,), seed=7), form="list")
        cell(f"A/{b}/roots/list11", b, roots, G((11,), seed=7), form="list")


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
