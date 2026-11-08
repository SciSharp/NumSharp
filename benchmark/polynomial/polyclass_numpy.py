"""NumPy 2.4.2 twin of polyclass_bench.cs — numpy.polynomial's six series classes (U10: Polynomial, Chebyshev, Legendre,
Laguerre, Hermite, HermiteE over ABCPolyBase) and their printers (U11's class half).

Writes data/polyclass/manifest.json (gitignored) — the cell list the C# side replays, each with the SHA-256 of NumPy's
result — then prints NumPy's best-of-R per-call time for every cell (TSV: cell<TAB>us). The C# runner rebuilds the same
inputs from the shared formula (polyeval_numpy.gen) and checks its result against the digest BEFORE it times the cell.

Usage (from this directory; pin both sides with the same mask, never run them concurrently; three alternating pairs):
    set NS_PROBE_AFFINITY=0x14
    set OPENBLAS_NUM_THREADS=1
    python polyclass_numpy.py > numpy_class_times_1.tsv
    set DOTNET_TC_CallCountingDelayMs=0
    dotnet run -c Release --no-cache polyclass_bench.cs -- data/polyclass numpy_class_times_1.tsv > ns_class_1.tsv
    (the same for _2 and _3)
    python polyclass_report.py ns_class_1.tsv ns_class_2.tsv ns_class_3.tsv > polyclass_results.md
POLYCLASS_ONLY=A/,P/ (comma-separated label prefixes) restricts both sides to matching cells.

Sections (cell-label prefixes; every cell runs for each of the six classes, label <section>/<op>/<Class>):
    K  construction and the class methods: the constructor (8 and 1,000 coefficients), identity, basis(5), fromroots
       (5 roots), cast (16 coefficients, from another class)
    A  arithmetic: + and * of two series (8 and 64 coefficients; + also at 1,000), + / * / true division by a Python
       float, // and divmod (16 by 4), ** 3 (4 coefficients), unary -, ==
    E  evaluation: p(x) at 1,000 and 100,000 points and at a Python float (8 coefficients), p(q) (composition, 4 and 4)
    C  calculus and conversion: deriv, integ (64 coefficients; integ also with k and lbnd), convert(kind=...) (16),
       linspace(100)
    M  the members: copy (1,000), trim, cutdeg(8), truncate(8) (64), mapparms, degree
    P  the printers: str (ascii), repr, _repr_latex_, format 'unicode' (8 coefficients)
    S  pickling: __getstate__, __setstate__ (8 coefficients)
    L  LAPACK: roots (8 coefficients: eigvals of the companion matrix), fit (1,000 points, degree 8: lstsq) — run at
       OPENBLAS_NUM_THREADS=1, the configuration the C# side pins
"""
import hashlib
import json
import os
import struct
import sys
import timeit
from pathlib import Path

import numpy as np  # noqa: E402
import numpy.polynomial._polybase as pb  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "polyclass"
sys.path.insert(0, str(HERE))
from polyeval_numpy import gen, pin_affinity  # noqa: E402  (the shared input formula and host pinning)

CLASSES = ["Polynomial", "Chebyshev", "Legendre", "Laguerre", "Hermite", "HermiteE"]
CELLS = []
ONLY = [p for p in os.environ.get("POLYCLASS_ONLY", "").split(",") if p]
# Every series of a cell has this domain (the window is the class default): a mapped series, so evaluation and
# conversion run the mapping; both operands of a binary operator share it (NumPy refuses differing domains).
DOMAIN = [-2.0, 3.0]


def vec(n, seed):
    """The shared float64 input formula as a 1-D array.

    Args:
        n: The length.
        seed: The seed.

    Returns:
        The array.
    """
    return gen([n], "float64", seed)


def series(cls, n, seed, domain=True):
    """A series of the class over the shared formula's coefficients.

    Args:
        cls: The class name.
        n: The coefficient count.
        seed: The seed.
        domain: Whether to give it DOMAIN (else the class default).

    Returns:
        The series.
    """
    c = getattr(np.polynomial, cls)
    return c(vec(n, seed), domain=DOMAIN) if domain else c(vec(n, seed))


def digest(r):
    """NumPy's result as the hash the C# runner recomputes: a series by its coef, domain and window bytes, an array (or
    a NumPy scalar) by its C-order bytes, a str by its UTF-8, a bool / int / float by a fixed-width encoding, a tuple /
    dict by the digests of its parts in order.

    Args:
        r: The result.

    Returns:
        The hex SHA-256.
    """
    h = hashlib.sha256()

    def feed(x):
        if isinstance(x, pb.ABCPolyBase):
            for a in (x.coef, x.domain, x.window):
                h.update(np.ascontiguousarray(a).tobytes())
        elif isinstance(x, (tuple, list)):
            for part in x:
                feed(part)
        elif isinstance(x, dict):
            for k, v in x.items():
                h.update(k.encode("utf-8"))
                feed(v)
        elif isinstance(x, str):
            h.update(x.encode("utf-8"))
        elif isinstance(x, (bool, np.bool_)):
            h.update(b"\x01" if x else b"\x00")
        elif isinstance(x, int):
            h.update(struct.pack("<q", x))
        elif isinstance(x, float):
            h.update(struct.pack("<d", x))
        else:
            h.update(np.ascontiguousarray(np.asarray(x)).tobytes())

    feed(r)
    return h.hexdigest()


def cell(label, cls, op, call, args, calls=None, rounds=9):
    """Registers one cell: its C# recipe, NumPy's digest and its timed call.

    Args:
        label: "<section>/<op label>" (the class is appended).
        cls: The class name.
        op: The C# runner's operation key.
        call: The NumPy call (no arguments).
        args: The recipe's arguments the C# side reads (sizes, seeds, numbers).
        calls: Calls per timing round, or None to size it to ~20 ms a round.
        rounds: Rounds (the best one is reported).

    Returns:
        None.
    """
    label = f"{label}/{cls}"
    if ONLY and not any(label.startswith(p) for p in ONLY):
        return
    with np.errstate(all="ignore"):
        r = call()
    if calls is None:
        t = min(timeit.repeat(call, number=1, repeat=5))
        calls = max(1, min(20000, int(0.02 / max(t, 1e-7))))
    CELLS.append({"label": label, "cls": cls, "op": op, "args": args, "calls": calls, "rounds": rounds,
                  "sha256": digest(r), "_call": call})


def build():
    """Registers every cell of the sections (see the module docstring).

    Returns:
        None.
    """
    P = np.polynomial
    for cls in CLASSES:
        C = getattr(P, cls)
        other = "Chebyshev" if cls == "Polynomial" else "Polynomial"
        # (K) construction and the class methods
        for n in (8, 1000):
            c = vec(n, 1)
            cell(f"K/ctor{n}", cls, "ctor", lambda c=c: C(c, domain=DOMAIN), {"n": n})
        cell("K/identity", cls, "identity", lambda: C.identity(DOMAIN), {})
        cell("K/basis5", cls, "basis", lambda: C.basis(5, DOMAIN), {"deg": 5})
        r5 = vec(5, 3)
        cell("K/fromroots5", cls, "fromroots", lambda: C.fromroots(r5, DOMAIN), {"n": 5})
        src = series(other, 16, 1)
        cell("K/cast16", cls, "cast", lambda: C.cast(src, DOMAIN), {"n": 16, "from": other})
        # (A) arithmetic
        for n in (8, 64, 1000):
            p, q = series(cls, n, 1), series(cls, n, 2)
            cell(f"A/add{n}", cls, "add", lambda p=p, q=q: p + q, {"n": n})
            if n < 1000:
                cell(f"A/mul{n}", cls, "mul", lambda p=p, q=q: p * q, {"n": n})
        p8 = series(cls, 8, 1)
        cell("A/add_float", cls, "add_float", lambda: p8 + 2.5, {"n": 8, "v": 2.5})
        cell("A/mul_float", cls, "mul_float", lambda: p8 * 2.5, {"n": 8, "v": 2.5})
        cell("A/truediv_float", cls, "truediv_float", lambda: p8 / 2.5, {"n": 8, "v": 2.5})
        p16, q4 = series(cls, 16, 1), series(cls, 4, 2)
        cell("A/floordiv16x4", cls, "floordiv", lambda: p16 // q4, {"n": 16, "m": 4})
        cell("A/divmod16x4", cls, "divmod", lambda: divmod(p16, q4), {"n": 16, "m": 4})
        p4 = series(cls, 4, 1)
        cell("A/pow3", cls, "pow", lambda: p4 ** 3, {"n": 4, "e": 3})
        cell("A/neg", cls, "neg", lambda: -p8, {"n": 8})
        q8 = series(cls, 8, 1)
        cell("A/eq", cls, "eq", lambda: p8 == q8, {"n": 8})
        # (E) evaluation
        for m in (1000, 100000):
            x = vec(m, 5)
            cell(f"E/call{m}", cls, "call", lambda x=x: p8(x), {"n": 8, "m": m})
        cell("E/call_float", cls, "call_float", lambda: p8(0.5), {"n": 8, "v": 0.5})
        s4 = series(cls, 4, 2)
        cell("E/compose4", cls, "compose", lambda: p4(s4), {"n": 4, "m": 4})
        # (C) calculus and conversion
        p64 = series(cls, 64, 1)
        cell("C/deriv64", cls, "deriv", lambda: p64.deriv(), {"n": 64})
        cell("C/integ64", cls, "integ", lambda: p64.integ(), {"n": 64})
        cell("C/integ64_k_lbnd", cls, "integ_k_lbnd", lambda: p64.integ(2, [1.0, -1.0], 0.5), {"n": 64})
        K = getattr(P, other)
        cell("C/convert16", cls, "convert", lambda: p16.convert(kind=K), {"n": 16, "kind": other})
        cell("C/linspace100", cls, "linspace", lambda: p8.linspace(100), {"n": 8, "m": 100})
        # (M) members
        p1000 = series(cls, 1000, 1)
        cell("M/copy1000", cls, "copy", lambda: p1000.copy(), {"n": 1000})
        cell("M/trim64", cls, "trim", lambda: p64.trim(), {"n": 64})
        cell("M/cutdeg64", cls, "cutdeg", lambda: p64.cutdeg(8), {"n": 64, "deg": 8})
        cell("M/truncate64", cls, "truncate", lambda: p64.truncate(8), {"n": 64, "size": 8})
        cell("M/mapparms", cls, "mapparms", lambda: p8.mapparms(), {"n": 8})
        cell("M/degree", cls, "degree", lambda: p8.degree(), {"n": 8})
        # (P) printers, at NumPy's default print options; str under the ascii style (the class flag)
        cell("P/str", cls, "str", lambda: _ascii_str(p8), {"n": 8})
        cell("P/repr", cls, "repr", lambda: repr(p8), {"n": 8})
        cell("P/latex", cls, "latex", lambda: p8._repr_latex_(), {"n": 8})
        cell("P/format_unicode", cls, "format_unicode", lambda: format(p8, "unicode"), {"n": 8})
        # (S) pickling
        cell("S/getstate", cls, "getstate", lambda: p8.__getstate__(), {"n": 8})
        state = p8.__getstate__()
        target = series(cls, 3, 9)
        cell("S/setstate", cls, "setstate", lambda: _setstate(target, state), {"n": 8})
        # (L) LAPACK
        cell("L/roots8", cls, "roots", lambda: p8.roots(), {"n": 8})
        xs, ys = vec(1000, 6), vec(1000, 7)
        cell("L/fit1000x8", cls, "fit", lambda: C.fit(xs, ys, 8), {"m": 1000, "deg": 8})


def _ascii_str(p):
    """str(p) under the ascii style (the class flag, restored to NumPy's default after the call).

    Args:
        p: The series.

    Returns:
        The text.
    """
    before = pb.ABCPolyBase._use_unicode
    pb.ABCPolyBase._use_unicode = False
    try:
        return str(p)
    finally:
        pb.ABCPolyBase._use_unicode = before


def _setstate(target, state):
    """target.__setstate__(state), then the target (the digest reads the restored series).

    Args:
        target: The series to restore into.
        state: The state.

    Returns:
        The target.
    """
    target.__setstate__(state)
    return target


def main():
    """Pins the process, writes the manifest the C# runner replays, then prints NumPy's best per-call time per cell.

    Returns:
        None.
    """
    pin_affinity()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        sys.exit("polyclass_numpy.py needs OPENBLAS_NUM_THREADS=1 set before Python starts (the roots / fit bits and "
                 "times depend on the thread count, and the C# side runs one thread)")
    DATA.mkdir(parents=True, exist_ok=True)
    build()
    (DATA / "manifest.json").write_text(json.dumps([{k: v for k, v in c.items() if k != "_call"} for c in CELLS],
                                                   indent=1), encoding="utf-8")
    print("cell\tnumpy_us")
    for c in CELLS:
        call, calls, rounds = c["_call"], c["calls"], c["rounds"]
        with np.errstate(all="ignore"):
            call()   # warm
            best = min(timeit.repeat(call, number=calls, repeat=rounds)) / calls
        print(f"{c['label']}\t{best * 1e6:.3f}", flush=True)


if __name__ == "__main__":
    main()
