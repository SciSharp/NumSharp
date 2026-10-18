"""NumPy 2.4.2 twin of polyfit_bench.cs — the numpy.polynomial least-squares fit benchmark (U6: {p}fit).

Writes data/polyfit/manifest.json + data/polyfit/inputs.npz (gitignored) — the cell list and the arrays the C# side
replays — then prints NumPy's best-of-R per-call time for every cell (TSV: cell<TAB>us<TAB>lstsq us). Each cell records
the SHA-256 of NumPy's result (+ dtype and shape; for full=True every slot), which the C# runner checks BEFORE it times
the cell. Both sides run OpenBLAS at ONE thread: lstsq's gelsd bits depend on the thread count, and so does its speed.

The third column is NumPy's time for the cell's np.linalg.lstsq call ALONE, on the operands _fit hands it (saved as
<key>_A / <key>_b with the rcond in the manifest, so the C# side times its own lstsq on the same matrix). Both libraries
call the same gelsd, so a fit's NPY/NS cannot exceed NumPy's whole call over the lstsq time: the report shows that
ceiling next to each cell.

Usage (from this directory; pin both sides with the same mask, never run them concurrently):
    set NS_PROBE_AFFINITY=0x14
    set OPENBLAS_NUM_THREADS=1
    python polyfit_numpy.py > numpy_fit_times.tsv
    set DOTNET_TC_CallCountingDelayMs=0
    dotnet run -c Release --no-cache polyfit_bench.cs -- data/polyfit numpy_fit_times.tsv > ns_fit.tsv
    python polyfit_report.py ns_fit.tsv > polyfit_results.md
POLYFIT_ONLY=C/,P/ (comma-separated label prefixes) restricts both sides to matching cells.

Sections (cell-label prefixes):
    C  chebfit, n points x degree: n = 10 / 100 / 1,000 / 100,000, degree 3 / 10 / 30
    S  the six bases at n = 20, degree 5 (lstsq is a few microseconds: the composition's own cost)
    B  the other five bases at n = 1,000, degree 10
    K  kinds at n = 1,000, degree 10: weights, a 2-D y (3 data sets), full=True, a degree list, float32, complex, an
       integer x, Python lists
"""
import json
import os
import sys
import timeit
from pathlib import Path

assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "set OPENBLAS_NUM_THREADS=1 before Python starts"

import numpy as np  # noqa: E402
import numpy.polynomial as npp  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "polyfit"
sys.path.insert(0, str(HERE))
from polyeval_numpy import digest, pin_affinity  # noqa: E402  (the shared digest + host pinning)

FITS = {"poly": npp.polynomial.polyfit, "cheb": npp.chebyshev.chebfit, "leg": npp.legendre.legfit,
        "lag": npp.laguerre.lagfit, "herm": npp.hermite.hermfit, "herme": npp.hermite_e.hermefit}
VANDERS = {"poly": npp.polynomial.polyvander, "cheb": npp.chebyshev.chebvander, "leg": npp.legendre.legvander,
           "lag": npp.laguerre.lagvander, "herm": npp.hermite.hermvander, "herme": npp.hermite_e.hermevander}
CELLS = []
INPUTS = {}
ONLY = [p for p in os.environ.get("POLYFIT_ONLY", "").split(",") if p]


def loops_for(n, deg):
    """Picks the timing loop of a cell: enough calls per round to dwarf the timer, a few calls for the big cells.

    Args:
        n: The number of points.
        deg: The degree (gelsd's work grows as n * deg**2).

    Returns:
        (calls per round, rounds) — both sides time every cell with the same pair.
    """
    work = n * (deg + 1) ** 2
    if work >= 5e7:
        return 3, 7
    if work >= 5e6:
        return 10, 7
    if work >= 5e5:
        return 50, 9
    return 500, 9


def data(n, seed, kind="f64"):
    """A deterministic sample set: n points on [-1, 1] and noisy values of cos(2.5 x).

    Args:
        n: The number of points.
        seed: The RNG seed.
        kind: "f64", "f32", "complex", "int" (x as an integer ramp) or "y2" (three data sets).

    Returns:
        (x, y, w) arrays.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    y = np.cos(2.5 * x) + 0.05 * rng.standard_normal(n)
    w = 0.5 + rng.random(n)
    if kind == "f32":
        return x.astype(np.float32), y.astype(np.float32), w.astype(np.float32)
    if kind == "complex":
        return x, y + 1j * np.sin(x), w
    if kind == "int":
        return np.arange(n), y, w
    if kind == "y2":
        return x, np.stack([y, -y, 2 * y], axis=1), w
    return x, y, w


def lstsq_operands(fn, x, y, deg, w):
    """NumPy's polyutils._fit up to its lstsq call, statement for statement: the operands np.linalg.lstsq receives.

    Args:
        fn: The basis prefix (a VANDERS key).
        x: The sample points.
        y: The values.
        deg: The degree, or a list of degrees.
        w: The weights, or None.

    Returns:
        (lhs.T / scl, rhs.T, rcond) — the default rcond, len(x) * eps of x's dtype.
    """
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0
    deg = np.asarray(deg)
    if deg.ndim == 0:
        van = VANDERS[fn](x, int(deg))
    else:
        deg = np.sort(deg)
        van = VANDERS[fn](x, deg[-1])[:, deg]
    lhs = van.T
    rhs = y.T
    if w is not None:
        w = np.asarray(w) + 0.0
        lhs = lhs * w
        rhs = rhs * w
    rcond = len(x) * np.finfo(x.dtype).eps
    if issubclass(lhs.dtype.type, np.complexfloating):
        scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
    else:
        scl = np.sqrt(np.square(lhs).sum(1))
    scl[scl == 0] = 1
    return lhs.T / scl, rhs.T, rcond


def cell(label, fn, n, deg, kind="f64", weights=False, full=False, deg_list=None, lists=False):
    """Registers one cell, {fn}(x, y, deg, ...), its inputs and NumPy's result digests.

    Args:
        label: The cell's label ("<section>/<case>"); POLYFIT_ONLY restricts the run to matching labels.
        fn: The basis prefix (a FITS key).
        n: The number of points.
        deg: The degree (ignored when deg_list is given).
        kind: The data kind (see data()).
        weights: Pass w.
        full: Pass full=True (every slot is digested).
        deg_list: A list of degrees to fit instead of deg.
        lists: Pass x and y as Python lists.

    Returns:
        None.
    """
    if ONLY and not any(label.startswith(p) for p in ONLY):
        return
    x, y, w = data(n, 7 + n + deg, kind)
    key = label.replace("/", "_")
    INPUTS[key + "_x"], INPUTS[key + "_y"], INPUTS[key + "_w"] = x, y, w
    f = FITS[fn]
    d = deg_list if deg_list is not None else deg
    xa, ya = (x.tolist(), y.tolist()) if lists else (x, y)
    kwargs = {}
    if weights:
        kwargs["w"] = w
    if full:
        kwargs["full"] = True
    call = lambda: f(xa, ya, d, **kwargs)
    r = call()
    slots = [r[0], r[1][0], np.asarray(r[1][1]), r[1][2], np.asarray(r[1][3])] if full else [r]
    # The lstsq call alone, on the operands this fit hands it (the shared gelsd floor of both libraries).
    la, lb, lrcond = lstsq_operands(fn, x, y, d, w if weights else None)
    INPUTS[key + "_A"], INPUTS[key + "_b"] = la, lb
    lcall = lambda: np.linalg.lstsq(la, lb, lrcond)
    calls, rounds = loops_for(n, deg)
    spec = {"label": label, "fn": fn, "key": key, "deg": d, "weights": weights, "full": full, "lists": lists,
            "rcond": float(lrcond), "expect": [digest(s) for s in slots], "calls": calls, "rounds": rounds}
    CELLS.append({**spec, "_call": call, "_lstsq": lcall})


def build():
    """Registers every cell of the three sections (see the module docstring).

    Returns:
        None.
    """
    for n in (10, 100, 1000, 100_000):
        for deg in (3, 10, 30):
            if deg < n:
                cell(f"C/n{n}_d{deg}", "cheb", n, deg)
    for fn in FITS:
        cell(f"S/{fn}", fn, 20, 5)
    for fn in ("poly", "leg", "lag", "herm", "herme"):
        cell(f"B/{fn}", fn, 1000, 10)
    cell("K/weights", "cheb", 1000, 10, weights=True)
    cell("K/y2d", "cheb", 1000, 10, kind="y2")
    cell("K/full", "cheb", 1000, 10, full=True)
    cell("K/deglist", "cheb", 1000, 10, deg_list=[0, 2, 4, 6, 8, 10])
    cell("K/float32", "cheb", 1000, 10, kind="f32")
    cell("K/complex", "cheb", 1000, 10, kind="complex")
    cell("K/intx", "poly", 1000, 3, kind="int")
    cell("K/lists", "cheb", 1000, 10, lists=True)


def main():
    """Pins the process, writes the manifest and inputs the C# runner replays, then prints NumPy's best per-call time.

    Returns:
        None.
    """
    pin_affinity()
    DATA.mkdir(parents=True, exist_ok=True)
    build()
    np.savez(DATA / "inputs.npz", **INPUTS)
    (DATA / "manifest.json").write_text(json.dumps([{k: v for k, v in c.items() if not k.startswith("_")}
                                                    for c in CELLS], indent=1), encoding="utf-8")
    print("cell\tnumpy_us\tnumpy_lstsq_us")
    for c in CELLS:
        call, lcall, calls, rounds = c["_call"], c["_lstsq"], c["calls"], c["rounds"]
        call()   # warm
        best = min(timeit.repeat(call, number=calls, repeat=rounds)) / calls
        lcall()
        best_l = min(timeit.repeat(lcall, number=calls, repeat=rounds)) / calls
        print(f"{c['label']}\t{best * 1e6:.3f}\t{best_l * 1e6:.3f}", flush=True)


if __name__ == "__main__":
    main()
