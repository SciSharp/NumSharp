"""Head-to-head NumPy side for the numpy.polynomial names NumSharp lacks (twin: polynomial_rest_h2h.cs).

Representative cells of each missing name, plus the floors they sit on (float64 exp/sin/cos, lstsq, eigvalsh, dot) and
the fit's column-scaling spelling. Each cell prints ``label  microseconds``: the minimum over 25 batches of at least
5 ms, so a quiet slice of a loaded host shows. Pinned to CPUs 2 and 4 like the C# twin. Run the pair ALTERNATELY at
least twice and keep each cell's minimum; ratios are NPY/NS (NumPy time / NumSharp time, higher = NumSharp faster).
Run with ``OPENBLAS_NUM_THREADS=1``.
"""
import ctypes
import sys
import time


def pin(mask=0x14):
    """Pin this process to the logical CPUs in ``mask`` and prove it by reading the mask back.

    ``SetProcessAffinityMask`` through ``ctypes`` needs declared signatures: without them the pseudo-handle -1 is
    truncated to a 32-bit int, the call fails, and nothing reports it (benchmark/CLAUDE.md, host stability).

    Args:
        mask: Affinity bit mask (0x14 = CPUs 2 and 4, two P-cores on the reference host).

    Raises:
        OSError: The mask could not be applied or does not read back.
    """
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    k32.SetProcessAffinityMask.restype = ctypes.c_int
    k32.GetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.POINTER(ctypes.c_size_t))
    k32.GetProcessAffinityMask.restype = ctypes.c_int
    handle = k32.GetCurrentProcess()
    if not k32.SetProcessAffinityMask(handle, mask):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
    proc, system = ctypes.c_size_t(), ctypes.c_size_t()
    k32.GetProcessAffinityMask(handle, ctypes.byref(proc), ctypes.byref(system))
    if proc.value != mask:
        raise OSError(0, f"affinity reads back {proc.value:#x}, not {mask:#x}")


pin()  # before NumPy loads, so OpenBLAS's threads (one, by the environment) start on the pinned CPUs

import numpy as np  # noqa: E402
import numpy.polynomial as NP  # noqa: E402
from numpy.polynomial import chebyshev as C, hermite as H, hermite_e as HE  # noqa: E402
from numpy.polynomial import laguerre as L, legendre as LG, polynomial as PP, polyutils as pu  # noqa: E402


def data(n, amp):
    """The shared input ``sin(k * 1.2345678901) * amp`` (bit-identical to the C# twin's).

    Args:
        n: Element count.
        amp: Amplitude.

    Returns:
        numpy.ndarray: float64 values.
    """
    return np.sin(np.arange(n, dtype=np.int64) * 1.2345678901) * amp


def bench(label, fn):
    """Print the minimum microseconds per call of ``fn`` over 25 batches of at least 5 ms.

    Args:
        label: The cell label (identical in the C# twin).
        fn: Zero-argument callable.
    """
    fn()
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if time.perf_counter() - t0 >= 0.005 or n >= 1 << 20:
            break
        n *= 2
    best = float("inf")
    for _ in range(25):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t0) / n)
    print(f"{label:40s} {best * 1e6:12.3f}", flush=True)


def main():
    """Every head-to-head cell, in the C# twin's order."""
    for n in (1000, 1_000_000):
        x = data(n, 0.95)
        bench(f"chebweight f64 {n}", lambda x=x: C.chebweight(x))
        bench(f"hermweight f64 {n}", lambda x=x: H.hermweight(x))
    x = data(1_000_000, 0.95)
    bench("legweight f64 1000000", lambda: LG.legweight(x))
    bench("lagweight f64 1000000", lambda: L.lagweight(x))
    x32 = x.astype(np.float32)
    bench("chebweight f32 1000000", lambda: C.chebweight(x32))
    bench("hermweight f32 1000000", lambda: H.hermweight(x32))
    x16 = data(100_000, 0.95).astype(np.float16)
    bench("chebweight f16 100000", lambda: C.chebweight(x16))
    for n in (100, 1_000_000):
        bench(f"chebpts1 {n}", lambda n=n: C.chebpts1(n))
        bench(f"chebpts2 {n}", lambda n=n: C.chebpts2(n))
    bench("chebgauss 50", lambda: C.chebgauss(50))
    bench("chebgauss 100000", lambda: C.chebgauss(100_000))
    for d in (5, 20, 100):
        bench(f"leggauss {d}", lambda d=d: LG.leggauss(d))
        bench(f"hermgauss {d}", lambda d=d: H.hermgauss(d))
    bench("laggauss 20", lambda: L.laggauss(20))
    bench("hermegauss 20", lambda: HE.hermegauss(20))
    for d in (5, 20, 100):
        bench(f"chebinterpolate {d}", lambda d=d: C.chebinterpolate(np.sin, d))
    for n, d in ((100, 3), (1000, 10), (100_000, 10)):
        xf = data(n, 1.0)
        yf = np.sin(3.0 * xf)
        bench(f"chebfit n{n} d{d}", lambda xf=xf, yf=yf, d=d: C.chebfit(xf, yf, d))
    xf = data(1000, 1.0)
    yf = np.sin(3.0 * xf)
    bench("polyfit n1000 d10", lambda: PP.polyfit(xf, yf, 10))
    for n in (1000, 100_000, 1_000_000):
        xv = data(n, 1.0)
        r = data(10, 0.9)
        bench(f"polyvalfromroots f64 n{n} r10", lambda xv=xv, r=r: PP.polyvalfromroots(xv, r))
    xv = data(100_000, 1.0)
    xc = xv + 1j * xv[::-1]
    rc = data(10, 0.9) + 0.5j
    bench("polyvalfromroots c128 n100000 r10", lambda: PP.polyvalfromroots(xc, rc))
    r10 = data(10, 0.9)
    bench("polyvalfromroots scalar r10", lambda: PP.polyvalfromroots(0.3, r10))
    c10, c5 = data(11, 1.0), data(6, 1.0)
    p, q, ch = NP.Polynomial(c10), NP.Polynomial(c5), NP.Chebyshev(c10)
    bench("class p+q", lambda: p + q)
    bench("class p*q", lambda: p * q)
    x1k, x100k = data(1000, 1.0), data(100_000, 1.0)
    bench("class ch(x) 1000", lambda: ch(x1k))
    bench("class ch(x) 100000", lambda: ch(x100k))
    bench("class ch.convert(kind=Polynomial)", lambda: ch.convert(kind=NP.Polynomial))
    bench("class Chebyshev.fit n1000 d10", lambda: NP.Chebyshev.fit(xf, yf, 10))
    bench("format_float(0.1)", lambda: pu.format_float(0.1))
    # floors: the work both sides do identically
    for n in (1000, 1_000_000):
        xs = data(n, 1.0)
        bench(f"floor np.exp f64 {n}", lambda xs=xs: np.exp(xs))
        bench(f"floor np.sin f64 {n}", lambda xs=xs: np.sin(xs))
        bench(f"floor np.cos f64 {n}", lambda xs=xs: np.cos(xs))
    for n, d in ((100, 3), (1000, 10), (100_000, 10)):
        xs = data(n, 1.0)
        van = C.chebvander(xs, d)
        ys = np.sin(3.0 * xs)
        bench(f"floor lstsq n{n} d{d}", lambda van=van, ys=ys: np.linalg.lstsq(van, ys, None))
    for deg in (5, 20, 100):
        m = LG.legcompanion(np.array([0] * deg + [1]))
        bench(f"floor eigvalsh {deg}", lambda m=m: np.linalg.eigvalsh(m))
    for deg in (20, 100):
        xs = data(deg + 1, 0.9)
        mv = C.chebvander(xs, deg)
        ys = np.sin(xs)
        bench(f"floor dot(m.T, y) {deg}", lambda mv=mv, ys=ys: np.dot(mv.T, ys))
    for n in (1000, 100_000):
        lhs = data(11 * n, 1.0).reshape(11, n)
        scl = np.abs(np.sin(np.arange(11))) + 1.0
        bench(f"scale lhs.T / scl n{n} d10", lambda lhs=lhs, scl=scl: lhs.T / scl)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
