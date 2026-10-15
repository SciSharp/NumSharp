"""NumPy 2.4.2 twin of polynomial_engine_probe.cs - the evidence behind docs/plans/numpy-polynomial.md §9.

Writes seeded inputs and NumPy's reference outputs as .npy files into ./data (next to this script) so
the C# probe can byte-compare every NumSharp implementation strategy, then prints best-of-7 timings
(microseconds, TSV) for exactly the cells the C# probe times, so NPY/NS ratios come from one host
state.

Usage (from this directory, both sides pinned to the same logical CPUs on a hybrid host):
    set NS_PROBE_AFFINITY=0xFF
    python polynomial_engine_numpy.py            > numpy_times.tsv
    dotnet run -c Release polynomial_engine_probe.cs -- data
Pin with the same mask on both sides and never run them at the same time. OPENBLAS_NUM_THREADS=1 is
set here because the fit/roots cells of later probes are BLAS-bound and a threaded BLAS changes bits.
"""
import os
import sys
import timeit
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402  (after the BLAS thread pin on purpose)
from numpy.polynomial import chebyshev as C  # noqa: E402
from numpy.polynomial import legendre as L  # noqa: E402
from numpy.polynomial import polynomial as P  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def pin_affinity():
    """Pin like the C# probe does (NS_PROBE_AFFINITY hex mask) through the ONE shared implementation
    in benchmark/scripts/benchmark_host.py, which reads the mask back and logs whether it took."""
    sys.path.insert(0, str(HERE.parents[1] / "scripts"))
    from benchmark_host import pin_current_process_from_env
    pin_current_process_from_env()


def write_inputs_and_references():
    """Generate the seeded inputs and NumPy's outputs; returns the namespace the timings read."""
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    coef = {d: rng.standard_normal(d + 1) for d in (3, 5, 10, 30, 50)}
    xs = {n: rng.uniform(-1, 1, n) for n in (1_000, 100_000, 10_000_000)}
    y100k = rng.uniform(-1, 1, 100_000)
    c55 = rng.standard_normal((6, 6))  # 2-D series of degree (5, 5)

    def save(name, arr):
        np.save(DATA / f"{name}.npy", np.asarray(arr))

    for d, c in coef.items():
        save(f"c{d}", c)
    for n, x in xs.items():
        save(f"x{n}", x)
    save("y100000", y100k)
    save("c55", c55)

    # small-series algebra (U1/U2/U4 shapes)
    save("ref_polyadd_5_10", P.polyadd(coef[5], coef[10]))
    save("ref_chebmul_10", C.chebmul(coef[10], coef[10]))
    save("ref_legmul_10", L.legmul(coef[10], coef[10]))
    save("ref_legmul_50", L.legmul(coef[50], coef[50]))
    q, r = L.legdiv(coef[50], coef[10])
    save("ref_legdiv_q", q)
    save("ref_legdiv_r", r)
    save("ref_chebder_50", C.chebder(coef[50]))
    save("ref_legint_50", L.legint(coef[50]))
    # evaluation / vander (U3/U5 shapes); 10M is timing-only
    for n in (1_000, 100_000):
        for d in (3, 10, 30):
            save(f"ref_polyval_c{d}_x{n}", P.polyval(xs[n], coef[d]))
            save(f"ref_chebval_c{d}_x{n}", C.chebval(xs[n], coef[d]))
            save(f"ref_legval_c{d}_x{n}", L.legval(xs[n], coef[d]))
    save("ref_chebval2d_x100000", C.chebval2d(xs[100_000], y100k, c55))
    save("ref_chebvander_x100000_d10", np.ascontiguousarray(C.chebvander(xs[100_000], 10)))
    save("ref_chebval_c10_strided50k", C.chebval(xs[100_000][::2], coef[10]))
    return dict(P=P, C=C, L=L, c=coef, x=xs, y=y100k, c55=c55,
                s50k=xs[100_000][::2], s5m=xs[10_000_000][::2])


def time_cells(g):
    """Print best-of-7 microseconds per call for every cell the C# probe measures."""
    def t(label, stmt, number=None):
        timer = timeit.Timer(stmt, globals=g)
        if number is None:
            number, _ = timer.autorange()
        best = min(timer.repeat(repeat=7, number=number)) / number
        print(f"{label}\t{best * 1e6:.3f}", flush=True)

    t("S polyadd 5+10", "P.polyadd(c[5], c[10])")
    t("S chebmul 10x10", "C.chebmul(c[10], c[10])")
    t("S legmul 10x10", "L.legmul(c[10], c[10])")
    t("S legmul 50x50", "L.legmul(c[50], c[50])")
    t("S legdiv 50/10", "L.legdiv(c[50], c[10])")
    t("S chebder 50", "C.chebder(c[50])")
    t("S legint 50", "L.legint(c[50])")
    for n in (1_000, 100_000, 10_000_000):
        number = 1 if n == 10_000_000 else None
        for d in (3, 10, 30):
            if n == 10_000_000 and d != 10:
                continue
            t(f"E polyval d{d} n{n}", f"P.polyval(x[{n}], c[{d}])", number)
            t(f"E chebval d{d} n{n}", f"C.chebval(x[{n}], c[{d}])", number)
            t(f"E legval d{d} n{n}", f"L.legval(x[{n}], c[{d}])", number)
    t("E chebval2d d5x5 n100000", "C.chebval2d(x[100000], y, c55)")
    t("V chebvander d10 n100000", "C.chebvander(x[100000], 10)")
    t("L chebval d10 strided 50K", "C.chebval(s50k, c[10])")
    t("L chebval d10 strided 5M", "C.chebval(s5m, c[10])", 1)


if __name__ == "__main__":
    pin_affinity()
    print(f"# numpy {np.__version__}", file=sys.stderr)
    time_cells(write_inputs_and_references())
