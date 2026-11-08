"""NumPy 2.4.2 facts behind docs/plans/numpy-polynomial-remaining-compute.md, printed for the record.

Two parts:

1. **The complex product of ``polyvalfromroots``.** NumPy computes ``np.prod(x - r, axis=0)``; its complex multiply
   loop has no reduce branch and runs ``simd_cmul`` unless the output stride is 0, where it falls back to the
   MSVC-contracted ``loop_scalar``. Each candidate form is emulated in exact rational arithmetic (a correctly rounded
   fma), starting from the multiply identity ``1+0j`` that ``PyUFunc_ReduceWrapper`` assigns, multiplying the roots in
   ascending order. The tally shows which form matches NumPy's bytes per call shape.
2. **Boundary facts** of the missing names: result type, dtype and shape per argument kind, and NumPy's verbatim error
   texts (the contract an implementation has to meet before any compute).

Run with NumPy 2.4.2: ``python benchmark/polynomial/probes/polynomial_rest_numpy_facts.py``.
"""
import warnings
from fractions import Fraction

import numpy as np
import numpy.polynomial as NP
from numpy.polynomial import chebyshev as C, hermite as H, hermite_e as HE
from numpy.polynomial import laguerre as L, legendre as LG, polynomial as PP, polyutils as pu

# The probes feed sqrt/log of negatives and rank-deficient fits on purpose; the warnings are not the facts recorded.
warnings.simplefilter("ignore")


def fma(a, b, c):
    """Correctly rounded fused multiply-add of finite doubles.

    Args:
        a: First factor.
        b: Second factor.
        c: Addend.

    Returns:
        float: ``a*b + c`` computed exactly in rationals, then rounded once (IEEE fma semantics for finite inputs).

    Raises:
        OverflowError: The exact result is outside the double range (never for the probe's magnitudes).
    """
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def cmul_simd(a, b):
    """NumPy's ``simd_cmul``: ``re = fma(ar, br, -(ai*bi))``, ``im = fma(ar, bi, ai*br)``.

    Args:
        a: Left complex operand (NumPy's first input).
        b: Right complex operand.

    Returns:
        complex: The product as the vector body computes it.
    """
    return complex(fma(a.real, b.real, -(a.imag * b.imag)), fma(a.real, b.imag, a.imag * b.real))


def cmul_loop_scalar(a, b):
    """``CDOUBLE_multiply``'s ``loop_scalar`` as MSVC contracted it: the imaginary part fuses the OTHER product.

    Args:
        a: Left complex operand.
        b: Right complex operand.

    Returns:
        complex: ``re = fma(ar, br, -(ai*bi))``, ``im = fma(ai, br, ar*bi)``.
    """
    return complex(fma(a.real, b.real, -(a.imag * b.imag)), fma(a.imag, b.real, a.real * b.imag))


def cmul_naive(a, b):
    """Scalarmath's product: four rounded products, no fma.

    Args:
        a: Left complex operand.
        b: Right complex operand.

    Returns:
        complex: ``(ar*br - ai*bi) + (ar*bi + ai*br)j`` with every product rounded.
    """
    return complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real)


FORMS = {"simd": cmul_simd, "loop_scalar": cmul_loop_scalar, "naive": cmul_naive}


def root_product(xv, roots, form):
    """``1+0j`` times ``(x - r0)`` times ``(x - r1)`` ... in root order, with one multiply form.

    Args:
        xv: The point (a Python complex).
        roots: The roots, ascending index order is NumPy's reduction order.
        form: A key of ``FORMS``.

    Returns:
        complex: The emulated product.
    """
    acc = complex(1.0, 0.0)  # PyUFunc_ReduceWrapper seeds multiply.reduce with the identity, not with r[0]
    for r in roots:
        acc = FORMS[form](acc, complex(xv - r))
    return acc


def raw_bits(z):
    """Both parts' float64 bit patterns, so equality is exact (no NaN or signed-zero aliasing).

    Args:
        z: A complex number.

    Returns:
        list: ``[bits(real), bits(imag)]`` as Python ints.
    """
    return np.array([z.real, z.imag]).view(np.uint64).tolist()


def complex_form_tally(trials=400):
    """Count, per call shape, how many of NumPy's ``polyvalfromroots`` results each multiply form reproduces.

    Args:
        trials: Random root sets per call shape (2-8 roots, magnitudes over seven decades so the forms differ).

    Returns:
        dict: ``{shape: {form: matches, "n": cases}}``.
    """
    rng = np.random.default_rng(7)
    tally = {}
    for _ in range(trials):
        nr = int(rng.integers(2, 9))
        roots = (rng.standard_normal(nr) * 10.0 ** rng.integers(-3, 4, nr)
                 + 1j * rng.standard_normal(nr) * 10.0 ** rng.integers(-3, 4, nr))
        xs = rng.standard_normal(5) * 3 + 1j * rng.standard_normal(5) * 3
        r2 = np.asfortranarray(np.stack([roots, roots[::-1]], axis=1))  # (nr, 2): the roots axis innermost in memory
        cases = {
            "python_scalar_x": (roots, [(complex(xs[0]), PP.polyvalfromroots(complex(xs[0]), roots))]),
            "0d_array_x": (roots, [(complex(xs[0]), PP.polyvalfromroots(np.array(xs[0]), roots)[()])]),
            "array_x_5": (roots, list(zip(map(complex, xs), PP.polyvalfromroots(xs, roots)))),
            "array_x_1": (roots, [(complex(xs[0]), PP.polyvalfromroots(xs[:1], roots)[0])]),
            "tensorF_r": (r2[:, 0], [(complex(xs[1]), PP.polyvalfromroots(complex(xs[1]), r2, tensor=False)[0])]),
            "tensorC_r": (r2[:, 0], [(complex(xs[1]),
                                      PP.polyvalfromroots(complex(xs[1]), np.ascontiguousarray(r2), tensor=False)[0])]),
        }
        for name, (col, pairs) in cases.items():
            entry = tally.setdefault(name, {"n": 0})
            for xv, got in pairs:
                entry["n"] += 1
                for form in FORMS:
                    entry[form] = entry.get(form, 0) + int(raw_bits(root_product(xv, col, form)) == raw_bits(complex(got)))
    return tally


def show(label, fn):
    """Print one boundary fact: the result's type/dtype/shape and first values, or the exception NumPy raises.

    Args:
        label: The call, as text.
        fn: A zero-argument callable making the call.
    """
    try:
        r = fn()
        if isinstance(r, tuple):
            desc = "tuple(" + ", ".join(f"{type(v).__name__}:{getattr(v, 'dtype', '')}" for v in r) + ")"
        else:
            desc = f"{type(r).__name__}:{getattr(r, 'dtype', '')}{list(np.shape(r))} {np.asarray(r).ravel()[:3]}"
        print(f"{label:52s} -> {desc}")
    except Exception as e:  # noqa: BLE001 - recording NumPy's exception IS the fact
        print(f"{label:52s} !! {type(e).__name__}: {e}")


def boundary_facts():
    """Print the argument-kind, dtype and error matrix of the missing names (§3 of the plan)."""
    weights = (("chebweight", C.chebweight), ("legweight", LG.legweight), ("lagweight", L.lagweight),
               ("hermweight", H.hermweight), ("hermeweight", HE.hermeweight))
    kinds = (("py float 0.5", 0.5), ("py int 2", 2), ("py complex 0.5j", 0.5j), ("np.float32(0.5)", np.float32(0.5)),
             ("bool arr", np.array([True, False])), ("int8 arr", np.array([1, 2], np.int8)),
             ("uint8 arr", np.array([1, 200], np.uint8)), ("int64 arr", np.array([1, 2])),
             ("float16 arr", np.array([0.5], np.float16)), ("complex arr", np.array([0.5 + 1j])),
             ("list [0.5]", [0.5]), ("tuple (0.5,)", (0.5,)), ("py int 2**70", 2 ** 70),
             ("0-d f32", np.array(0.5, np.float32)))
    print("--- weights")
    for name, w in weights:
        for arg_label, arg in kinds:
            show(f"{name}({arg_label})", lambda w=w, arg=arg: w(arg))
    # The operator x**2 is np.square for every array dtype (number.c fast_scalar_power); np.power is not.
    print("bool_array**2:", (np.array([True]) ** 2).dtype, " np.power(bool, 2):", np.power(np.array([True]), 2).dtype)
    print("--- chebpts / gauss arguments")
    for arg in (3, 3.0, 3.5, True, np.int64(3), "3", 0, -1, 1, None):
        show(f"chebpts1({arg!r})", lambda arg=arg: C.chebpts1(arg))
        show(f"chebpts2({arg!r})", lambda arg=arg: C.chebpts2(arg))
    for arg in (3, 3.0, True, np.int64(3), np.array(3), 0, "3", None):
        show(f"chebgauss({arg!r})", lambda arg=arg: C.chebgauss(arg))
        show(f"leggauss({arg!r})", lambda arg=arg: LG.leggauss(arg))
    print("--- chebinterpolate")
    for arg in (3, 3.0, np.int64(3), np.array(3), np.array([3]), True, -1, 0):
        show(f"chebinterpolate(sin, {arg!r})", lambda arg=arg: C.chebinterpolate(np.sin, arg))
    show("chebinterpolate(f->(n,2), 3)", lambda: C.chebinterpolate(lambda x: np.stack([x, x * x], 1), 3))
    show("chebinterpolate(f->float32, 3)", lambda: C.chebinterpolate(lambda x: x.astype(np.float32), 3))
    show("chebinterpolate(f, 3, args=(2.0,))", lambda: C.chebinterpolate(lambda x, a: a * x, 3, args=(2.0,)))
    print("--- polyvalfromroots")
    r = np.array([0.5, -0.25, 2.0])
    for xl, x in (("py 0.3", 0.3), ("py int 2", 2), ("np.float32(0.3)", np.float32(0.3)), ("0-d f64", np.array(0.3)),
                  ("list", [0.3, 1.0]), ("tuple", (0.3, 1.0)), ("2-D (2,2)", np.ones((2, 2))), ("py complex", 0.3j)):
        show(f"polyvalfromroots({xl}, r f64)", lambda x=x: PP.polyvalfromroots(x, r))
    for rl, rr in (("int r", [1, 2]), ("bool r", [True]), ("f32 r", np.array([0.5], np.float32)),
                   ("f16 r", np.array([0.5], np.float16)), ("empty", []), ("2-D r (2,3)", np.ones((2, 3)))):
        show(f"polyvalfromroots(0.3, {rl})", lambda rr=rr: PP.polyvalfromroots(0.3, rr))
        show(f"polyvalfromroots(arr(2,), {rl})", lambda rr=rr: PP.polyvalfromroots(np.array([0.3, 2.0]), rr))
    show("polyvalfromroots(x(2,3), r(2,3), tensor=False)",
         lambda: PP.polyvalfromroots(np.ones((2, 3)), np.ones((2, 3)), tensor=False))
    show("polyvalfromroots(None, r)", lambda: PP.polyvalfromroots(None, r))
    for label, x in (("x F (3,4)", np.asfortranarray(np.arange(12.).reshape(3, 4))),
                     ("x T (4,3)", np.arange(12.).reshape(3, 4).T), ("x empty", np.zeros(0))):
        out = PP.polyvalfromroots(x, r)
        print(f"layout {label:10s} C={out.flags.c_contiguous} F={out.flags.f_contiguous} strides={out.strides}")
    print("--- fits")
    x = np.linspace(-1, 1, 8)
    y = np.sin(x)
    for dl, d in (("2.0", 2.0), ("[0,2]", [0, 2]), ("[]", []), ("-1", -1), ("True", True)):
        show(f"chebfit(x, y, {dl})", lambda d=d: C.chebfit(x, y, d))
    show("chebfit(f16 x)", lambda: C.chebfit(x.astype(np.float16), y, 2))
    show("chebfit(f32 x, f32 y)", lambda: C.chebfit(x.astype(np.float32), y.astype(np.float32), 2))
    show("chebfit(f32 x, f64 y)", lambda: C.chebfit(x.astype(np.float32), y, 2))
    show("chebfit(x 2-D)", lambda: C.chebfit(np.ones((2, 4)), y, 2))
    show("chebfit(empty x)", lambda: C.chebfit([], [], 2))
    show("chebfit(len mismatch)", lambda: C.chebfit(x, y[:3], 2))
    print("--- printing")
    show("format_float(3)", lambda: pu.format_float(3))
    show("format_float(1e-5, parens=True)", lambda: pu.format_float(1e-5, parens=True))
    show("set_default_printstyle('latex')", lambda: NP.set_default_printstyle("latex"))


if __name__ == "__main__":
    print("numpy", np.__version__)
    print("=== polyvalfromroots: which complex product matches NumPy's bytes")
    for shape, t in complex_form_tally().items():
        n = t.pop("n")
        print(f"{shape:16s} n={n:5d}  " + "  ".join(f"{form}={count}" for form, count in t.items()))
    print("=== boundary facts")
    boundary_facts()
