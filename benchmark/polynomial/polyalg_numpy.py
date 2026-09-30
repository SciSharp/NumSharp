"""NumPy 2.4.2 twin of polyalg_bench.cs — the numpy.polynomial series-algebra (U2) benchmark: {p}mulx, {p}mul,
{p}div, {p}pow, {p}fromroots for the six bases and X2poly / poly2X for the five non-power ones.

Writes data/polyalg/manifest.json (gitignored) — the cell list the C# side replays — then prints NumPy's best-of-R
per-call time for every cell (TSV: cell<TAB>us). Inputs come from the exact shared formula of polyeval_numpy.py
(gen(), reused here). Each cell records the SHA-256 of NumPy's result bytes (+ dtype and shape; both slots for
{p}div; NaNs canonicalized first, as the oracle tokenizes them), which the C# runner checks BEFORE it times the cell — and, for a cell whose np.convolve reaches OpenBLAS's
vector dot kernels (float64 dots of 16+ terms, float32 32+, complex128 8+: the products no managed kernel reproduces
without NumSharp.Interop.OpenBLAS, see gen_oracle.py's gen_polyalgebra), a "blas" flag: the C# side then checks the
values (stored for those cells) to a relative 1e-12 instead of the bytes, and the report says so.

Usage (from this directory; pin both sides with the same mask, never run them concurrently):
    set NS_PROBE_AFFINITY=0xF0
    python polyalg_numpy.py > numpy_alg_times.tsv
    set DOTNET_TC_CallCountingDelayMs=0
    dotnet run -c Release --no-cache polyalg_bench.cs -- data/polyalg numpy_alg_times.tsv > ns_alg.tsv
    python polyalg_report.py ns_alg.tsv > polyalg_results.md
POLYALG_ONLY=X/ (comma-separated label prefixes) restricts both sides to matching cells.

Sections (cell-label prefixes):
    X  {p}mulx: float64 / float32 / float16 / complex128 series of 3 / 11 / 100 / 1000 / 10000 coefficients
    M  {p}mul, float64: short (3x3, 10x10, 11x4), medium (50x50, 100x100), and — for the convolving bases — long
       (1000x1000, 2000x300) and a long signal against a short kernel (10000x11, 100000x5)
    T  {p}mul in float32 / float16 / complex128 / int64: 10x10, 100x100 and (convolving bases) 1000x1000
    D  {p}div: float64 5/3, 20/7, 50/16, 200/50; float32 / complex128 at 50/16
    P  {p}pow: (3, 2), (5, 5), (10, 8); the convolving bases also (50, 4) and complex128 / float32 (10, 4)
    R  {p}fromroots: 3 / 10 / 30 / 100 float64 roots; complex128 10 / 30
    C  X2poly / poly2X: 3 / 10 / 50 / 200 float64 coefficients; float32 / complex128 at 50
    A  argument forms: Python lists (mulx 100, mul 10x10, div 20/7, pow (10, 3), fromroots 10)
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
DATA = HERE / "data" / "polyalg"
sys.path.insert(0, str(HERE))
from polyeval_numpy import G, pin_affinity  # noqa: E402  (the shared input formula + view grammar)

MODULES = {"poly": P, "cheb": C, "leg": L, "lag": LG, "herm": H, "herme": HE}
MODNAME = {"poly": "polynomial", "cheb": "chebyshev", "leg": "legendre", "lag": "laguerre",
           "herm": "hermite", "herme": "hermite_e"}
CONVOLVING = ("poly", "cheb")
CELLS = []
ONLY = [p for p in os.environ.get("POLYALG_ONLY", "").split(",") if p]


class ConvRecorder:
    """gen_oracle.py's _PAConvRecorder: every np.convolve a polynomial function makes, with its operand lengths, dtype
    and finiteness (the polynomial modules reach np.convolve through the numpy module at call time)."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        self._orig = np.convolve
        orig, calls = self._orig, self.calls

        def conv(a, v, mode="full"):
            aa, vv = np.asarray(a), np.asarray(v)
            with np.errstate(all="ignore"):
                finite = bool(np.all(np.isfinite(aa))) and bool(np.all(np.isfinite(vv)))
            calls.append((aa.size, vv.size, np.result_type(aa, vv), finite))
            return orig(a, v, mode)

        np.convolve = conv
        return self

    def __exit__(self, *exc):
        np.convolve = self._orig
        return False

    def blas_bound(self):
        for n1, n2, dt, finite in self.calls:
            m = min(n1, n2)
            if (dt == np.float64 and m >= 16) or (dt == np.float32 and m >= 32) or \
                    (dt == np.complex128 and (m >= 8 or not finite)):
                return True
        return False


def canonical_nan(r):
    """r with every NaN rewritten to the canonical quiet NaN of its width (per component for complex) — the oracle's
    NaN tokenization: a NaN's sign and payload are not part of the contract (the recurrence bases' long products
    overflow, and which NaN two NaNs produce is the x86 operand order, not NumPy's)."""
    r = np.array(np.asarray(r), copy=True)
    if r.dtype.kind == "c":
        parts = r.view(r.real.dtype)
        parts[np.isnan(parts)] = np.nan
    elif r.dtype.kind == "f":
        r[np.isnan(r)] = np.nan
    return r


def digest(r):
    """dtype, shape and the SHA-256 of the NaN-canonicalized bytes (logical C order)."""
    r = np.asarray(r)
    h = hashlib.sha256(np.ascontiguousarray(canonical_nan(r)).tobytes()).hexdigest()
    return {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape], "sha256": h}


def loops_for(call):
    """(calls per round, rounds) from one timed call: ~50 ms rounds (at least one call) and 7 rounds, 3 for a call of
    100 ms or more — the recurrence bases' long products and divisions run Python loops of a second or more per call
    in NumPy, where a fixed per-size schedule would spend minutes on one cell."""
    t = min(timeit.repeat(call, number=1, repeat=2))
    calls = max(1, int(0.05 / max(t, 1e-9)))
    return min(calls, 2000), (3 if t >= 0.1 else 7)


def arg(v, form):
    """(manifest spec, Python value) of one argument: a G array (as the array, or — form "list" — as its tolist())."""
    return ({"gen": v.spec, "form": form} if form else {"gen": v.spec}), (v.array.tolist() if form == "list" else v.array)


def cell(label, base, fn, args, extra=None, form=None):
    """Register one cell: {module}.{fn}(*args[, **extra]). args are G arrays (form "list" passes each as a Python list
    of Python floats — the conversion is part of the timed call on both sides, as np.array(c, ndmin=1) is part of
    NumPy's); extra holds pow / maxpower. POLYALG_ONLY=prefix[,prefix...] restricts the run to matching labels."""
    if ONLY and not any(label.startswith(p) for p in ONLY):
        return
    f = getattr(MODULES[base], fn)
    specs, pyargs = zip(*(arg(a, form) for a in args))
    kw = dict(extra or {})
    call = lambda: f(*pyargs, **kw)
    rec = ConvRecorder()
    with rec:
        ref = call()
    blas = rec.blas_bound()

    def expect_of(x):
        # A BLAS-bound cell also carries NumPy's values (logical C order), which the C# side compares to a relative
        # 1e-12 — its managed product sums in another order, so the digest alone could only say "different".
        d = digest(x)
        if blas:
            d["hex"] = np.ascontiguousarray(np.asarray(x)).tobytes().hex()
        return d

    expect = [expect_of(x) for x in ref] if isinstance(ref, tuple) else expect_of(ref)
    calls, rounds = loops_for(call)
    CELLS.append({"label": label, "module": MODNAME[base], "base": base, "fn": fn, "args": list(specs),
                  **({"extra": extra} if extra else {}), "expect": expect, "blas": blas,
                  "calls": calls, "rounds": rounds, "_call": call})


def build():
    for b in MODULES:
        conv = b in CONVOLVING
        # X: mulx
        for dt in ("float64", "float32", "float16", "complex128"):
            for n in (3, 11, 100, 1000, 10_000):
                cell(f"X/{b}/{dt}/n{n}", b, b + "mulx", [G((n,), dt, 7)])
        # M: mul float64
        pairs = [(3, 3), (10, 10), (11, 4), (50, 50), (100, 100)]
        if conv:
            pairs += [(1000, 1000), (2000, 300), (10_000, 11), (100_000, 5)]
        else:
            pairs += [(200, 200)]
        for n1, n2 in pairs:
            cell(f"M/{b}/{n1}x{n2}", b, b + "mul", [G((n1,), seed=7), G((n2,), seed=11)])
        # T: mul dtypes
        for dt in ("float32", "float16", "complex128", "int64"):
            for n1, n2 in ((10, 10), (100, 100)) + (((1000, 1000),) if conv else ()):
                cell(f"T/{b}/{dt}/{n1}x{n2}", b, b + "mul", [G((n1,), dt, 7), G((n2,), dt, 11)])
        # D: div
        for n1, n2 in ((5, 3), (20, 7), (50, 16), (200, 50)):
            cell(f"D/{b}/float64/{n1}by{n2}", b, b + "div", [G((n1,), seed=7), G((n2,), seed=11)])
        for dt in ("float32", "complex128"):
            cell(f"D/{b}/{dt}/50by16", b, b + "div", [G((50,), dt, 7), G((16,), dt, 11)])
        # P: pow
        pows = [(3, 2), (5, 5), (10, 8)] + ([(50, 4)] if conv else [])
        for n, k in pows:
            cell(f"P/{b}/float64/n{n}/k{k}", b, b + "pow", [G((n,), seed=7)], extra={"pow": k})
        if conv:
            for dt in ("complex128", "float32"):
                cell(f"P/{b}/{dt}/n10/k4", b, b + "pow", [G((10,), dt, 7)], extra={"pow": 4})
        # R: fromroots
        for k in (3, 10, 30, 100):
            cell(f"R/{b}/float64/k{k}", b, b + "fromroots", [G((k,), seed=7)])
        for k in (10, 30):
            cell(f"R/{b}/complex128/k{k}", b, b + "fromroots", [G((k,), "complex128", 7)])
        # C: conversions
        if b != "poly":
            for fn in (b + "2poly", "poly2" + b):
                for n in (3, 10, 50, 200):
                    cell(f"C/{b}/{fn}/float64/n{n}", b, fn, [G((n,), seed=7)])
                for dt in ("float32", "complex128"):
                    cell(f"C/{b}/{fn}/{dt}/n50", b, fn, [G((50,), dt, 7)])
        # A: Python-list arguments
        cell(f"A/{b}/mulx/list100", b, b + "mulx", [G((100,), seed=7)], form="list")
        cell(f"A/{b}/mul/list10x10", b, b + "mul", [G((10,), seed=7), G((10,), seed=11)], form="list")
        cell(f"A/{b}/div/list20by7", b, b + "div", [G((20,), seed=7), G((7,), seed=11)], form="list")
        cell(f"A/{b}/pow/list10k3", b, b + "pow", [G((10,), seed=7)], extra={"pow": 3}, form="list")
        cell(f"A/{b}/fromroots/list10", b, b + "fromroots", [G((10,), seed=7)], form="list")


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
