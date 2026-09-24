"""NumPy 2.4.2 twin of polynomial_il_eval_probe.cs - the evidence behind docs/plans/numpy-polynomial.md §10.

Writes seeded inputs and NumPy's reference outputs as .npy files into ./data/il (next to this script,
gitignored) so the C# probe can byte-compare every IL kernel, then prints best-of-7 timings
(microseconds, TSV: cell<TAB>us) for exactly the cells the C# probe times. The cell labels are
identical on both sides, so the two TSVs join on the first column.

Usage (from this directory, both sides pinned to the same logical CPUs on a hybrid host):
    set NS_PROBE_AFFINITY=0xFF
    python polynomial_il_numpy.py [SECTIONS]             > numpy_il_times.tsv
    dotnet run -c Release polynomial_il_eval_probe.cs -- data/il ACDFG numpy_il_times.tsv
    dotnet run -c Release polynomial_il_series_probe.cs -- data/il numpy_il_times.tsv
SECTIONS (optional) restricts references and timings to cells whose label starts with one of the letters.
Pin with the same mask on both sides and never run them at the same time.

Sections (the same letters as the C# probe):
    A  evaluation, all six bases, 1-D coefficients, float64, degree x size
    C  dtypes: float32 / float16 / complex128, mixed x/c dtypes (incl. the Clenshaw peeling cases), int x,
       IEEE edge values
    D  non-contiguous x layouts
    F  N-D coefficients: multi-series tensor=True, tensor=False, val2d (_valnd), grid2d (_gridnd)
    G  Vandermonde: all six bases, dtypes, int x, strided x, vander2d (_vander_nd)
    S  small-series algebra (polynomial_il_series_probe.cs): polyadd, legmul, legdiv, chebder across dtypes
"""
import os
import sys
import timeit
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402  (after the BLAS thread pin on purpose)
from numpy.polynomial import chebyshev as C  # noqa: E402
from numpy.polynomial import hermite as H  # noqa: E402
from numpy.polynomial import hermite_e as HE  # noqa: E402
from numpy.polynomial import laguerre as LG  # noqa: E402
from numpy.polynomial import legendre as L  # noqa: E402
from numpy.polynomial import polynomial as P  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "il"

# Basis short name -> (val, vander, val2d, grid2d, vander2d). The short names are the C# probe's too.
BASES = {
    "poly": (P.polyval, P.polyvander, P.polyval2d, P.polygrid2d, P.polyvander2d),
    "cheb": (C.chebval, C.chebvander, C.chebval2d, C.chebgrid2d, C.chebvander2d),
    "leg": (L.legval, L.legvander, L.legval2d, L.leggrid2d, L.legvander2d),
    "lag": (LG.lagval, LG.lagvander, LG.lagval2d, LG.laggrid2d, LG.lagvander2d),
    "herm": (H.hermval, H.hermvander, H.hermval2d, H.hermgrid2d, H.hermvander2d),
    "herme": (HE.hermeval, HE.hermevander, HE.hermeval2d, HE.hermegrid2d, HE.hermevander2d),
}


def pin_affinity():
    """Pin like the C# probe does (NS_PROBE_AFFINITY hex mask) through the ONE shared implementation
    in benchmark/scripts/benchmark_host.py, which reads the mask back and logs whether it took."""
    sys.path.insert(0, str(HERE.parents[1] / "scripts"))
    from benchmark_host import pin_current_process_from_env
    pin_current_process_from_env()


def build_inputs():
    """Seeded inputs shared by the references and the timings. Returns the eval namespace."""
    rng = np.random.default_rng(11)
    ns = {"np": np}
    ns["x"] = {n: rng.uniform(-1, 1, n) for n in (1_000, 100_000, 10_000_000)}
    # Vandermonde sizes: views of the evaluation inputs, 1M being the first million of the 10M array.
    ns["xm"] = {1_000: ns["x"][1_000], 100_000: ns["x"][100_000], 1_000_000: ns["x"][10_000_000][:1_000_000]}
    ns["c"] = {d: rng.standard_normal(d + 1) for d in (0, 1, 2, 3, 4, 10, 30)}
    ns["y"] = rng.uniform(-1, 1, 100_000)
    ns["cc10"] = ns["c"][10] + 1j * rng.standard_normal(11)
    ns["xc"] = ns["x"][100_000] + 1j * rng.uniform(-1, 1, 100_000)
    ns["x32"] = ns["x"][100_000].astype(np.float32)
    ns["x16"] = ns["x"][100_000].astype(np.float16)
    ns["xi"] = rng.integers(-1000, 1000, 100_000)
    # float32 coefficients against float64 x: NumPy runs the first two Clenshaw steps in float32.
    ns["cf32"] = {nc: rng.standard_normal(nc).astype(np.float32) for nc in (3, 4, 5, 11)}
    ns["cf16"] = rng.standard_normal(11).astype(np.float16)
    ns["edge32"] = np.array([3e38, -3e38, 1.5, np.inf, -np.inf, np.nan, -0.0, 0.0], dtype=np.float32)
    ns["edge64"] = np.array([np.inf, -np.inf, np.nan, -0.0, 0.0, 1e308, -1e308, 5e-324], dtype=np.float64)
    ns["cm"] = rng.standard_normal((11, 8))            # 8 series of degree 10 (multi-series, tensor=True)
    ns["cp"] = rng.standard_normal((11, 100_000))      # one degree-10 series PER POINT (tensor=False)
    ns["c55"] = rng.standard_normal((6, 6))            # a 2-D series of degree (5, 5)
    ns["gx"] = rng.uniform(-1, 1, 300)                 # grid2d axes (300 x 300 = 90K outputs)
    ns["gy"] = rng.uniform(-1, 1, 300)
    for short, fns in BASES.items():
        ns[short] = fns
    # Series inputs from their own generator, so adding them never shifts the evaluation inputs above.
    srng = np.random.default_rng(12)
    ns["s5"], ns["s10"], ns["s50"] = srng.standard_normal(6), srng.standard_normal(11), srng.standard_normal(51)
    ns["s10c"] = ns["s10"] + 1j * srng.standard_normal(11)
    ns["s50c"] = ns["s50"] + 1j * srng.standard_normal(51)
    ns["P"], ns["C"], ns["L"] = P, C, L
    return ns


def cells(ns):
    """(label, statement, reference-name-or-None, timed) for every cell of every section.

    The label is the join key with the C# probe; the reference name (when not None) is where the
    statement's output is saved for the byte comparison."""
    out = []
    # A — evaluation, all six bases, 1-D float64 coefficients.
    for b in BASES:
        for d in (0, 1, 2, 3, 4, 10, 30):
            for n in (1_000, 100_000, 10_000_000):
                if n == 100_000 and d not in (3, 10, 30):
                    continue
                if n == 10_000_000 and d != 10:
                    continue
                timed = d in (3, 10, 30)
                ref = None if n == 10_000_000 else f"ref_A_{b}_d{d}_n{n}"
                out.append((f"A {b}val d{d} n{n}", f"{b}[0](x[{n}], c[{d}])", ref, timed))
    # C — dtypes, mixed dtypes, integer x, IEEE edge values.
    for b in BASES:
        out.append((f"C {b}val f32 d10 n100000", f"{b}[0](x32, c[10].astype(np.float32))", f"ref_C_{b}_f32", True))
        out.append((f"C {b}val c128 d10 n100000", f"{b}[0](xc, cc10)", f"ref_C_{b}_c128", True))
    for b in ("poly", "cheb", "leg", "herm"):
        out.append((f"C {b}val f16 d10 n100000", f"{b}[0](x16, c[10].astype(np.float16))", f"ref_C_{b}_f16", True))
    for b in ("cheb", "leg"):
        out.append((f"C {b}val x32/c64 d10 n100000", f"{b}[0](x32, c[10])", f"ref_C_{b}_x32c64", True))
    for b in ("poly", "cheb", "lag"):
        out.append((f"C {b}val int64x d10 n100000", f"{b}[0](xi, c[10])", f"ref_C_{b}_xi", True))
    for b in ("cheb", "leg", "lag", "herm", "herme"):
        for nc in (3, 4, 5, 11):
            out.append((f"C {b}val c32/x64 nc{nc} n100000", f"{b}[0](x[100000], cf32[{nc}])",
                        f"ref_C_{b}_c32x64_nc{nc}", nc == 11))
    out.append(("C legval c16/x32 nc11 n100000", "leg[0](x32, cf16)", "ref_C_leg_c16x32", True))
    for b in BASES:
        out.append((f"C {b}val edge64 d10", f"{b}[0](edge64, c[10])", f"ref_C_{b}_edge64", False))
    out.append(("C chebval edge32/c64 d10", "cheb[0](edge32, c[10])", "ref_C_cheb_edge32", False))
    # D — non-contiguous x.
    for b in ("cheb", "leg"):
        out.append((f"D {b}val strided d10 n50000", f"{b}[0](x[100000][::2], c[10])", f"ref_D_{b}_strided", True))
        out.append((f"D {b}val reversed d10 n100000", f"{b}[0](x[100000][::-1], c[10])", f"ref_D_{b}_reversed", True))
        out.append((f"D {b}val 2dT d10 n100000", f"{b}[0](x[100000].reshape(250, 400).T, c[10])", f"ref_D_{b}_2dT", True))
    # F — N-D coefficients.
    for b in ("poly", "cheb", "leg"):
        out.append((f"F {b}val multi8 d10 n100000", f"{b}[0](x[100000], cm)", f"ref_F_{b}_multi", True))
        out.append((f"F {b}val tensorF d10 n100000", f"{b}[0](x[100000], cp, tensor=False)", f"ref_F_{b}_tensorF", True))
        out.append((f"F {b}val2d d5x5 n100000", f"{b}[2](x[100000], y, c55)", f"ref_F_{b}_val2d", True))
        out.append((f"F {b}grid2d d5x5 300x300", f"{b}[3](gx, gy, c55)", f"ref_F_{b}_grid2d", True))
    out.append(("F chebval tensorF strided d10 n50000", "cheb[0](x[100000][::2], cp[:, ::2], tensor=False)",
                "ref_F_cheb_tensorF_strided", True))
    # G — Vandermonde. The output is (deg+1) times the input, so the large size is 1M points (88 MB of
    # float64 output), not 10M (880 MB): "1m on large dtypes" in the house benchmark convention.
    for b in BASES:
        for n in (1_000, 100_000, 1_000_000):
            ref = None if n == 1_000_000 else f"ref_G_{b}_d10_n{n}"
            out.append((f"G {b}vander d10 n{n}", f"{b}[1](xm[{n}], 10)", ref, True))
        out.append((f"G {b}vander d0 n1000", f"{b}[1](x[1000], 0)", f"ref_G_{b}_d0", False))
        out.append((f"G {b}vander d1 n1000", f"{b}[1](x[1000], 1)", f"ref_G_{b}_d1", False))
        out.append((f"G {b}vander edge64 d10", f"{b}[1](edge64, 10)", f"ref_G_{b}_edge64", False))
    out.append(("G chebvander f32 d10 n100000", "cheb[1](x32, 10)", "ref_G_cheb_f32", True))
    out.append(("G chebvander f16 d10 n100000", "cheb[1](x16, 10)", "ref_G_cheb_f16", True))
    out.append(("G chebvander c128 d10 n100000", "cheb[1](xc, 10)", "ref_G_cheb_c128", True))
    out.append(("G chebvander int64x d10 n100000", "cheb[1](xi, 10)", "ref_G_cheb_xi", True))
    out.append(("G chebvander strided d10 n50000", "cheb[1](x[100000][::2], 10)", "ref_G_cheb_strided", True))
    out.append(("G chebvander 2dT d10 n100000", "cheb[1](x[100000].reshape(250, 400).T, 10)", "ref_G_cheb_2dT", True))
    for b in ("poly", "cheb"):
        out.append((f"G {b}vander2d d3x3 n100000", f"{b}[4](x[100000], y, (3, 3))", f"ref_G_{b}_vander2d", True))
    # S — small-series algebra. A tuple result (legdiv) is saved as ref_<name>_0 / _1.
    out.append(("S polyadd 5+10 f64", "P.polyadd(s5, s10)", "ref_S_polyadd_f64", True))
    for tag, conv in (("f64", ""), ("f32", ".astype(np.float32)"), ("f16", ".astype(np.float16)"), ("c128", "")):
        a10 = "s10c" if tag == "c128" else f"s10{conv}"
        a50 = "s50c" if tag == "c128" else f"s50{conv}"
        out.append((f"S legmul 10x10 {tag}", f"L.legmul({a10}, {a10})", f"ref_S_legmul10_{tag}", True))
        out.append((f"S chebder 50 {tag}", f"C.chebder({a50})", f"ref_S_chebder50_{tag}", True))
    out.append(("S legmul 50x50 f64", "L.legmul(s50, s50)", "ref_S_legmul50_f64", True))
    out.append(("S legdiv 50/10 f64", "L.legdiv(s50, s10)", "ref_S_legdiv_f64", True))
    out.append(("S legdiv 50/10 c128", "L.legdiv(s50c, s10c)", "ref_S_legdiv_c128", True))
    return out


def main():
    pin_affinity()
    sections = sys.argv[1] if len(sys.argv) > 1 else None
    DATA.mkdir(parents=True, exist_ok=True)
    ns = build_inputs()

    def save(name, arr):
        np.save(DATA / f"{name}.npy", np.asarray(arr))

    for n, v in ns["x"].items():
        save(f"x{n}", v)
    for d, v in ns["c"].items():
        save(f"c{d}", v)
    for nc, v in ns["cf32"].items():
        save(f"cf32_{nc}", v)
    for name in ("y", "cc10", "xc", "x32", "x16", "xi", "cf16", "edge32", "edge64", "cm", "cp", "c55", "gx", "gy",
                 "s5", "s10", "s50", "s10c", "s50c"):
        save(name, ns[name])

    todo = [c for c in cells(ns) if sections is None or c[0][0] in sections]
    with np.errstate(all="ignore"):
        for label, stmt, ref, _ in todo:
            if ref is not None:
                # The C# side compares logical C-order bytes; the vander layout (a moveaxis view) is
                # compared separately by shape/strides, so the payload is saved C-contiguous.
                result = eval(stmt, ns)
                if isinstance(result, tuple):
                    for k, part in enumerate(result):
                        save(f"{ref}_{k}", np.ascontiguousarray(part))
                else:
                    save(ref, np.ascontiguousarray(result))
        for label, stmt, _, timed in todo:
            if not timed:
                continue
            timer = timeit.Timer(stmt, globals=ns)
            number, _ = timer.autorange()
            best = min(timer.repeat(repeat=7, number=number)) / number
            print(f"{label}\t{best * 1e6:.3f}", flush=True)
    print(f"# numpy {np.__version__}", file=sys.stderr)


if __name__ == "__main__":
    main()
