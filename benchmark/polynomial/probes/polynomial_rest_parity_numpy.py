"""NumPy 2.4.2 side of the route parity check for the numpy.polynomial names NumSharp lacks.

Prints ``label  dtype[shape] sha256-prefix`` for each function's result on inputs NumSharp builds bit-identically
(``sin(arange(n) * 1.2345678901) * amp``: an exact multiply and the same CRT ``sin``). The C# twin
``polynomial_rest_parity.cs`` prints the same labels once per candidate NumSharp route; a route is NumPy-exact when
its line equals this one. Run with ``OPENBLAS_NUM_THREADS=1`` (the fits, Gauss functions and chebinterpolate end in
LAPACK/BLAS, whose bits depend on the thread count).
"""
import hashlib

import numpy as np
from numpy.polynomial import chebyshev as C, hermite as H, hermite_e as HE
from numpy.polynomial import laguerre as L, legendre as LG, polynomial as PP


def data(n, amp):
    """The shared input: ``sin(k * 1.2345678901) * amp`` for k = 0..n-1.

    Args:
        n: Element count.
        amp: Amplitude.

    Returns:
        numpy.ndarray: float64 values in [-amp, amp].
    """
    return np.sin(np.arange(n, dtype=np.int64) * 1.2345678901) * amp


def digest(a):
    """dtype, shape and the first 16 hex digits of the SHA-256 of the C-order bytes.

    Args:
        a: Array or scalar result.

    Returns:
        str: ``dtype[shape] hash``.
    """
    a = np.asarray(a)
    return f"{a.dtype.name}{list(a.shape)} {hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]}"


def main():
    """Print the hash of every function result the parity check compares."""
    out = []
    for dname, dt in (("f64", np.float64), ("f32", np.float32), ("f16", np.float16)):
        x = data(100_000, 0.95).astype(dt)
        out += [(f"chebweight {dname}", C.chebweight(x)), (f"legweight {dname}", LG.legweight(x)),
                (f"lagweight {dname}", L.lagweight(x)), (f"hermweight {dname}", H.hermweight(x)),
                (f"hermeweight {dname}", HE.hermeweight(x))]
    for n in (10, 1000):
        gx, gw = C.chebgauss(n)
        out += [(f"chebpts1 {n}", C.chebpts1(n)), (f"chebpts2 {n}", C.chebpts2(n)),
                (f"chebgauss.x {n}", gx), (f"chebgauss.w {n}", gw)]
    x = data(1000, 1.0)
    r = data(10, 0.9)
    out.append(("polyvalfromroots f64", PP.polyvalfromroots(x, r)))
    # complex inputs built from the shared data so the C# side can rebuild them exactly: x + v*1j is exactly
    # complex(x, v) here (the real part gains only a signed zero, the imaginary part is v itself).
    xc = x + 1j * data(1000, 0.7)[::-1]
    rc = r + 1j * data(10, 0.5)
    out += [("polyvalfromroots c128", PP.polyvalfromroots(xc, rc)),
            ("polyvalfromroots c128 scalar", PP.polyvalfromroots(complex(xc[3]), rc))]
    y = np.sin(3.0 * x)
    lhs = C.chebvander(x, 10).T
    out += [("chebfit norms", np.sqrt(np.square(lhs).sum(1))), ("chebfit", C.chebfit(x, y, 10))]
    for d in (5, 20):
        lx, lw = LG.leggauss(d)
        hx, hw = H.hermgauss(d)
        out += [(f"leggauss.x {d}", lx), (f"leggauss.w {d}", lw), (f"hermgauss.x {d}", hx), (f"hermgauss.w {d}", hw)]
    for d in (5, 20):
        out.append((f"chebinterpolate {d}", C.chebinterpolate(np.sin, d)))
    for label, v in out:
        print(f"{label:30s} {digest(v)}")


if __name__ == "__main__":
    main()
