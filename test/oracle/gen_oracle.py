"""
gen_oracle.py — emit a committed, bytes-exact NumPy 2.4.2 oracle corpus.

The corpus is JSONL (one case per line). C# replays the operand bytes EXACTLY and
compares its op result to `expected` bit-for-bit (NaN/inf tokenized). No Python at test time.

Case schema:
  {
    "id":      "<op>/<layout>/<src>-><dst>/<n>",
    "op":      "astype",                       # OpRegistry key
    "params":  {"dtype": "int32"},             # op-specific params
    "operands":[ <operand-descriptor>, ... ],  # see layout_catalog.describe()
    "expected":{"dtype":"int32","shape":[...],"buffer":"<hex C-contiguous result>"},
    "layout":  "strided_step2_1d",
    "valueclass":"mixed"
  }

operand-descriptor = {dtype, shape, strides(elements), offset(elements), bufferSize(elements), buffer(hex of base)}
"""
import io
import json
import os
import re
import sys
import tempfile
import warnings

import numpy as np

# Overflow / invalid-value-in-cast warnings ARE the edge cases we want to capture, not errors.
np.seterr(all="ignore")
warnings.simplefilter("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout_catalog import LAYOUTS, PAIR_LAYOUTS, WHERE_LAYOUTS, describe, _fill, _cbase  # noqa: E402

# 13 NumPy-representable dtypes (Char + Decimal have no NumPy analog -> covered by
# NumSharp's Converts-oracle tests, not by this differential corpus).
ALL_DTYPES = [
    "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "float16", "float32", "float64", "complex128",
]


def _expected(view, dst):
    exp = np.ascontiguousarray(view.astype(dst))
    return {"dtype": np.dtype(dst).name, "shape": [int(d) for d in view.shape], "buffer": exp.tobytes().hex()}


def gen_astype(srcs, dsts, layout_names):
    cases = []
    n = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in srcs:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            for d in dsts:
                cases.append({
                    "id": f"astype/{ln}/{s}->{d}/{n}",
                    "op": "astype",
                    "params": {"dtype": np.dtype(d).name},
                    "operands": [operand],
                    "expected": _expected(view, d),
                    "layout": ln,
                    "valueclass": "mixed",
                })
                n += 1
    return cases


# Binary ops: NumPy computes the result (value AND NEP50 result dtype) — it is the oracle.
# Bit-exact today (committed green matrix).
BINARY_OPS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,          # true_divide
}

# Known-divergent today (cataloged as [OpenBugs]): integer ÷0/mod0 throws-or-garbage vs NumPy 0,
# float //0 -> NaN vs NumPy ±inf, mixed-precision mod, complex power ~ULP/edge.
DIVMOD_POWER_OPS = {
    "floor_divide": lambda a, b: a // b,
    "mod": lambda a, b: a % b,             # NumPy: floored remainder (sign of divisor)
    "fmod": lambda a, b: np.fmod(a, b),    # C-style remainder (sign of dividend); truncated
    "power": lambda a, b: a ** b,
    # float_power: power at a MINIMUM precision of float64 (dd->d / DD->D loops only), so every real
    # input promotes to float64 and a complex operand to complex128 — and, unlike power, a negative
    # integer exponent is legal (float_power(2,-1)=0.5). Computed by NumSharp AS power on the forced
    # float loop, so it is bit-identical to power on those loops (same Math.Pow / npy_cpow).
    "float_power": lambda a, b: np.float_power(a, b),
}

# Comparison ops -> bool result. (NumPy raises TypeError for ordering complex; gen_binary skips those.)
COMPARISON_OPS = {
    "equal": lambda a, b: a == b,
    "not_equal": lambda a, b: a != b,
    "less": lambda a, b: a < b,
    "greater": lambda a, b: a > b,
    "less_equal": lambda a, b: a <= b,
    "greater_equal": lambda a, b: a >= b,
}

# Curated dtype pairs covering NEP50 promotion: same-type, int-width mixing, signed/unsigned,
# int->float, float widths, bool promotion, complex absorption.
DT_PAIRS = [
    ("int32", "int32"), ("int32", "int64"), ("int64", "int32"),
    ("int32", "float64"), ("float64", "int32"), ("int32", "float32"),
    ("float32", "float64"), ("float32", "float32"), ("float64", "float64"),
    ("uint8", "int8"), ("int8", "uint8"), ("uint8", "uint8"),
    ("int16", "int32"), ("uint32", "int32"), ("int32", "uint32"),
    ("bool", "int32"), ("bool", "float64"),
    ("complex128", "float64"), ("float64", "complex128"), ("complex128", "int32"),
    # W1: float16 as an operand (same-width, mixed-width-up, int->float16) and the narrow
    # integers (signed/unsigned width-mixing, the uint64+int64 -> float64 NEP50 special case).
    ("float16", "float16"), ("float16", "float32"), ("float16", "float64"),
    ("int8", "float16"), ("uint8", "float16"), ("float16", "int32"),
    ("int8", "int8"), ("int16", "int16"), ("uint16", "uint16"),
    ("uint32", "uint32"), ("uint64", "uint64"), ("int64", "uint64"),
    ("uint64", "int64"), ("int8", "int16"), ("uint8", "uint16"),
    ("int16", "uint16"), ("uint16", "int32"), ("complex128", "complex128"),
]


# Unary ops. NumPy is the oracle for result dtype (e.g. sqrt(int)->float64, abs(complex)->float64).
UNARY_OPS = {
    "negative": np.negative, "abs": np.abs, "fabs": np.fabs, "sign": np.sign,
    "sqrt": np.sqrt, "cbrt": np.cbrt, "square": np.square, "reciprocal": np.reciprocal,
    "floor": np.floor, "ceil": np.ceil, "trunc": np.trunc,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
    # Complex-number accessors (real inputs are valid too). These are central post-FFT paths and
    # must see the same full dtype x layout matrix as every other unary operation.
    "conjugate": np.conjugate, "conj": np.conj, "real": np.real, "imag": np.imag,
    "angle": np.angle, "angle_deg": lambda a: np.angle(a, deg=True),
}
# All 13 NumPy-representable dtypes (W1: was a 7-dtype subset — now exercises float16 as an
# INPUT and the narrow integers int8/int16/uint16/uint32/uint64 through every unary kernel).
UNARY_DTYPES = list(ALL_DTYPES)


# W3 — unary "stragglers": the transcendental / hyperbolic / inverse-trig / angle-conversion
# ufuncs that were absent from the unary tier. NumPy is the oracle for value AND width-based
# float result dtype (bool/int8/uint8 -> float16, int16/uint16 -> float32, int32+ -> float64).
UNARY_EXTRA_OPS = {
    "exp2": np.exp2, "expm1": np.expm1,
    "log2": np.log2, "log10": np.log10, "log1p": np.log1p,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "arcsinh": np.arcsinh, "asinh": np.asinh,
    "arccosh": np.arccosh, "acosh": np.acosh,
    "arctanh": np.arctanh, "atanh": np.atanh,
    "deg2rad": np.deg2rad, "rad2deg": np.rad2deg,
    "positive": np.positive,
    "rint": np.rint,   # round-half-to-even; float-tier dtype like the others in this group
    # spacing: distance to the adjacent representable value away from zero (one ULP). Float-tier
    # like the rest of this group, but a pure bit-fiddle (npy_spacing / npy_half_spacing) — held
    # BIT-EXACT, carved out of the "unary ~ULP" excuse (MisalignedRegistry.ByteExactArithmeticUnaryOps).
    "spacing": np.spacing,
}


# np.sinc — sin(pi*x)/(pi*x), its own tier (not folded into unary_extra) for ONE reason:
# sinc composes sin ∘ divide, and complex128 sinc amplifies NumSharp's complex-sin ≤3-ULP
# envelope through the division (and the exponential growth of sinh/cosh) to tens of ULP on
# interior points — beyond the tight 3-ULP complex-unary gate — so complex128 is EXCLUDED from
# the byte corpus (computed and allclose, pinned by a unit test instead). Every REAL dtype is
# bit-exact vs NumPy 2.4.2. Dtype follows `pi*x` (weak-float NEP 50): bool/all-ints/Char ->
# float64, float16/float32/float64 preserved.
SINC_OP = {"sinc": np.sinc}
SINC_DTYPES = [d for d in ALL_DTYPES if d != "complex128"]


# np.i0 — modified Bessel I_0, its own tier (like sinc, not folded into unary_extra): it is a
# cephes Chebyshev routine composing exp/sqrt, so its bytes are host-libm sensitive at float64
# (Math.Exp == win-amd64 ucrtbase) and float16 (BCL Half.Exp) — HOST-PINNED, Inconclusive
# off-Windows. float32 rides NumPy's OWN exp kernel (portable) but shares the tier. complex128 is
# EXCLUDED: NumPy raises TypeError("i0 not supported for complex values") for it (the rejection is
# unit-test-pinned). Dtype follows the input float precision (NEP 50): bool/all-ints/Char ->
# float64, float16/float32/float64 preserved. Every included dtype is BIT-EXACT vs NumPy 2.4.2.
I0_OP = {"i0": np.i0}
I0_DTYPES = [d for d in ALL_DTYPES if d != "complex128"]


def gen_unary(ops, dtypes, layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            for opname, f in ops.items():
                try:
                    r = f(view)
                except Exception:
                    skipped += 1  # NumPy raises (e.g. floor(complex)); error-parity tested separately
                    continue
                # Read the shape BEFORE ascontiguousarray (which forces ndim>=1, corrupting 0-D results).
                exp_shape = [int(d) for d in r.shape]
                exp_buf = np.ascontiguousarray(r).tobytes().hex()
                cases.append({
                    "id": f"{opname}/{ln}/{s}/{n}",
                    "op": opname,
                    "params": {},
                    "operands": [operand],
                    "expected": {"dtype": r.dtype.name, "shape": exp_shape, "buffer": exp_buf},
                    "layout": ln,
                    "valueclass": "mixed",
                })
                n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# Reductions. NumPy is the oracle for value, NEP50 accumulator dtype, and keepdims shape.
REDUCE_OPS = {
    "sum": lambda a, ax, kd: np.sum(a, axis=ax, keepdims=kd),
    "prod": lambda a, ax, kd: np.prod(a, axis=ax, keepdims=kd),
    "min": lambda a, ax, kd: np.min(a, axis=ax, keepdims=kd),
    "max": lambda a, ax, kd: np.max(a, axis=ax, keepdims=kd),
    "mean": lambda a, ax, kd: np.mean(a, axis=ax, keepdims=kd),
    "std": lambda a, ax, kd: np.std(a, axis=ax, keepdims=kd),
    "var": lambda a, ax, kd: np.var(a, axis=ax, keepdims=kd),
    "argmax": lambda a, ax, kd: np.argmax(a, axis=ax, keepdims=kd),
    "argmin": lambda a, ax, kd: np.argmin(a, axis=ax, keepdims=kd),
    "all": lambda a, ax, kd: np.all(a, axis=ax, keepdims=kd),
    "any": lambda a, ax, kd: np.any(a, axis=ax, keepdims=kd),
}
# All 13 dtypes (W1): exercises float16 + narrow-int accumulator promotion in every reduction.
REDUCE_DTYPES = list(ALL_DTYPES)
REDUCE_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "c_contiguous_3d", "f_contiguous_2d",
                  "transposed_3d", "strided_2d_cols", "broadcast_1d_to_2d", "scalar_0d",
                  "empty_2d", "one_element_1d",
                  # negative-stride views — exercise the reduce path's backward traversal
                  # (and, for f64/f32 min/max, the stride-ordered NDIter routing gated by
                  # DefaultEngine.MinMaxLayoutFavorsNDIter; the Direct kernel walks these
                  # cache-hostile, so they were a measured 6–10× cliff before the routing).
                  "negstride_1d", "negstride_2d_offset",
                  # G12 (F19): positive-offset slices + composed/0-d/reshape views — offset
                  # handling in reductions was previously reached only via negstride_2d_offset
                  # (and the W9-B repeat bug was precisely an offset bug).
                  "simple_slice_offset_1d", "sliced_composed", "zerod_from_index",
                  "reshape_view_2d"]


def _axes(ndim):
    if ndim == 0:
        return [None]
    if ndim == 1:
        return [None, 0]
    return [None, 0, ndim - 1]


def gen_reduce(ops, dtypes, layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            for opname, f in ops.items():
                for axis in _axes(view.ndim):
                    for keepdims in (False, True):
                        if opname in ("argmax", "argmin") and axis is None and keepdims:
                            continue  # NumSharp's flat argmax/argmin form (long) has no keepdims;
                                      # the keepdims=False flat cells replay via
                                      # NDArray.Scalar(np.argmax(a)) — the exact path whose Decimal
                                      # IL compare was silently wrong (G13)
                        try:
                            r = np.asarray(f(view, axis, keepdims))
                        except Exception:
                            skipped += 1
                            continue
                        cases.append({
                            "id": f"{opname}/{ln}/{s}/axis={axis}/kd={int(keepdims)}/{n}",
                            "op": opname,
                            "params": {"axis": axis, "keepdims": keepdims},
                            "operands": [operand],
                            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
                            "layout": ln,
                            "valueclass": "mixed",
                        })
                        n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# T10 — NaN-aware reductions. The float pools front-load NaN/±inf, so every slice contains NaNs:
# these ops must IGNORE them (NumPy contract). NumPy is the oracle for value, accumulator dtype,
# and the all-NaN-slice -> NaN behaviour.
NAN_REDUCE_OPS = {
    "nansum": lambda a, ax, kd: np.nansum(a, axis=ax, keepdims=kd),
    "nanprod": lambda a, ax, kd: np.nanprod(a, axis=ax, keepdims=kd),
    "nanmax": lambda a, ax, kd: np.nanmax(a, axis=ax, keepdims=kd),
    "nanmin": lambda a, ax, kd: np.nanmin(a, axis=ax, keepdims=kd),
    "nanmean": lambda a, ax, kd: np.nanmean(a, axis=ax, keepdims=kd),
    "nanstd": lambda a, ax, kd: np.nanstd(a, axis=ax, keepdims=kd),
    "nanvar": lambda a, ax, kd: np.nanvar(a, axis=ax, keepdims=kd),
    "nanmedian": lambda a, ax, kd: np.nanmedian(a, axis=ax, keepdims=kd),
    # issue #623: NaN-ignoring arg-reductions. All-NaN slices raise ValueError -> the gen's
    # try/except skips those cells (the error text is unit-test-pinned). Result is int64 indices,
    # deterministic (first-occurrence argmax/argmin contract) -> full bit-compare. Unlike
    # argmax/argmin these DO take axis=None here: np.nanarg* has the int?-axis NDArray overload.
    "nanargmax": lambda a, ax, kd: np.nanargmax(a, axis=ax, keepdims=kd),
    "nanargmin": lambda a, ax, kd: np.nanargmin(a, axis=ax, keepdims=kd),
}
NAN_REDUCE_DTYPES = list(ALL_DTYPES)   # widened: every dtype (NaN-erroring combos skipped by the gen)


def gen_binary(ops, dt_pairs, pair_layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in pair_layout_names:
        fn = PAIR_LAYOUTS[ln]
        for (sa, sb) in dt_pairs:
            ba, va, bb, vb = fn(np.dtype(sa), np.dtype(sb))
            op_a = describe(ba, va)
            op_b = describe(bb, vb)
            for opname, f in ops.items():
                try:
                    r = f(va, vb)
                except Exception:
                    skipped += 1  # NumPy raises (e.g. int**neg); error-parity is tested separately
                    continue
                # Read the shape BEFORE ascontiguousarray (which forces ndim>=1, corrupting 0-D results).
                exp_shape = [int(d) for d in r.shape]
                exp_buf = np.ascontiguousarray(r).tobytes().hex()
                cases.append({
                    "id": f"{opname}/{ln}/{sa},{sb}/{n}",
                    "op": opname,
                    "params": {},
                    "operands": [op_a, op_b],
                    "expected": {"dtype": r.dtype.name, "shape": exp_shape, "buffer": exp_buf},
                    "layout": ln,
                    "valueclass": "mixed",
                })
                n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# np.interp(x, xp, fp, left, right, period) — 1-D linear interpolation. Bit-exact with NumPy
# (managed binary_search_with_guess + slope port). Operands [x, xp, fp]; params carry left/right/period.
def gen_interp():
    cases = []
    n = [0]

    def emit(x, xp, fp, params, cid, layout="c_contiguous_1d"):
        kwargs = {k: params[k] for k in ("left", "right", "period") if k in params}
        try:
            r = np.asarray(np.interp(x, xp, fp, **kwargs))
        except Exception:
            return
        # asarray (not ascontiguousarray) so a 0-D x stays 0-D — ascontiguousarray min-1-D-promotes it.
        def _cc(a):
            a = np.asarray(a)
            return a if a.flags["C_CONTIGUOUS"] else np.ascontiguousarray(a)
        xa = _cc(x); xpa = _cc(xp); fpa = _cc(fp)
        cases.append({
            "id": f"interp/{cid}/{n[0]}",
            "op": "interp",
            "params": params,
            "operands": [describe(xa, xa), describe(xpa, xpa), describe(fpa, fpa)],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": layout,
            "valueclass": "mixed",
        })
        n[0] += 1

    xp = np.array([1., 2., 3.], np.float64)
    fp = np.array([3., 2., 0.], np.float64)
    xq = np.array([0., 1., 1.5, 2.72, 3.14], np.float64)
    emit(xq, xp, fp, {}, "basic")
    emit(np.array([-1., 5.]), xp, fp, {"left": -100.0, "right": 99.0}, "leftright")
    emit(np.array([-1., 5.]), xp, fp, {"left": -100.0}, "leftonly")
    emit(np.array([np.nan, 2.5, np.inf, -np.inf]), xp, fp, {}, "nan_inf_x")
    emit(np.array([0., 2., 5.]), np.array([2.]), np.array([9.]), {}, "single_point")
    emit(np.array([-180., -170., -185., 185., -10., -5., 0., 365.]),
         np.array([190., -190., 350., -350.]), np.array([5., 10., 3., 4.]), {"period": 360.0}, "period")
    emit(np.linspace(0., 10., 25), np.arange(11.), np.sin(np.arange(11.)), {}, "sine")
    emit(np.asarray(2.5), xp, fp, {}, "scalar0d")
    emit(np.array([[0.5, 2.5], [1.5, 3.5]]), xp, fp, {}, "highdim_2d", "c_contiguous_2d")
    emit(np.array([2], np.int64), np.array([1, 2, 3], np.int64), np.array([30, 20, 0], np.int64), {}, "int_inputs")
    emit(np.array([1.5, 4.0]), np.array([2., 3., 5.]), np.array([1j, 0, 2 + 3j]), {}, "complex_fp")
    emit(np.array([1.5, 2.5]), xp, np.array([np.inf, 0., np.inf]), {}, "inf_fp")
    emit(np.array([1.5]), np.array([1., 1., 2.]), np.array([10., 20., 30.]), {}, "equal_dx")
    return cases


# T12 — statistics. NumPy is the oracle for value, dtype (median/average/percentile/quantile ->
# float64; ptp preserves; count_nonzero -> int64), and keepdims shape.
STAT_REDUCE_OPS = {
    "median": lambda a, ax, kd: np.median(a, axis=ax, keepdims=kd),
    "average": lambda a, ax, kd: np.average(a, axis=ax, keepdims=kd),
    "ptp": lambda a, ax, kd: np.ptp(a, axis=ax, keepdims=kd),
}
STAT_DTYPES = list(ALL_DTYPES)         # widened: median/ptp/average across every dtype (skips on NumPy error)
STAT_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "c_contiguous_3d", "f_contiguous_2d",
                "transposed_3d", "strided_2d_cols", "one_element_1d"]
CNZ_DTYPES = list(ALL_DTYPES)          # widened: count_nonzero is dtype-agnostic
# clip dtypes. complex128 IS supported by NumPy's clip (probed 2.4.2: lexicographic
# real-then-imag ordering, NaN-poisoning comparisons — np.clip([1+2j,5+1j,-3+0j],0,2) ->
# [1+2j, 2+0j, 0+0j]); included below. bool is included too (fixed): NumSharp's general
# (strided/transposed/F-contig) clip kernel now handles Boolean by riding the Byte Min/Max
# path — bool bounds are (False, True) so the clip is an IDENTITY, gating that the strided
# path reads/writes each bool element in the right C-order position (the exact bug it had).
CLIP_DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
               "float16", "float32", "float64", "complex128"]
QUANTILE_SPECS = [
    ("percentile", lambda a, q, ax: np.percentile(a, q, axis=ax), [0.0, 25.0, 50.0, 75.0, 100.0]),
    ("quantile", lambda a, q, ax: np.quantile(a, q, axis=ax), [0.0, 0.25, 0.5, 0.75, 1.0]),
]


def gen_count_nonzero(dtypes, layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            if view.ndim == 0:
                continue
            operand = describe(base, view)
            axes = [0] if view.ndim == 1 else [0, view.ndim - 1]
            for axis in axes:
                for kd in (False, True):
                    try:
                        r = np.asarray(np.count_nonzero(view, axis=axis, keepdims=kd))
                    except Exception:
                        skipped += 1
                        continue
                    cases.append({
                        "id": f"count_nonzero/{ln}/{s}/axis={axis}/kd={int(kd)}/{n}",
                        "op": "count_nonzero",
                        "params": {"axis": axis, "keepdims": kd},
                        "operands": [operand],
                        "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                     "buffer": np.ascontiguousarray(r).tobytes().hex()},
                        "layout": ln,
                        "valueclass": "mixed",
                    })
                    n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


def gen_quantile(specs, dtypes, layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            for (opname, f, qs) in specs:
                for q in qs:
                    for axis in _axes(view.ndim):
                        try:
                            r = np.asarray(f(view, q, axis))
                        except Exception:
                            skipped += 1
                            continue
                        cases.append({
                            "id": f"{opname}/{ln}/{s}/q={q}/axis={axis}/{n}",
                            "op": opname,
                            "params": {"q": q, "axis": axis},
                            "operands": [operand],
                            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
                            "layout": ln,
                            "valueclass": "mixed",
                        })
                        n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


def gen_clip(dtypes, layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            dt = np.dtype(s)
            base, view = fn(dt)
            # bool has only two orderable values, so any scalar bound pair either passes the
            # array through (False, True) or saturates it to a constant — (False, True) makes clip
            # the identity, which is the informative case: it verifies the strided/transposed read
            # maps to the right C-order output slot (the bug that carved bool out originally).
            lo_v, hi_v = (0, 1) if dt.kind == "b" else (1, 100) if dt.kind == "u" else (-10, 10)
            lo = np.array(lo_v, dtype=dt).reshape(())
            hi = np.array(hi_v, dtype=dt).reshape(())
            try:
                r = np.asarray(np.clip(view, lo, hi))
            except Exception:
                skipped += 1
                continue
            cases.append({
                "id": f"clip/{ln}/{s}/{n}",
                "op": "clip",
                "params": {},
                "operands": [describe(base, view), describe(lo, lo), describe(hi, hi)],
                "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                             "buffer": np.ascontiguousarray(r).tobytes().hex()},
                "layout": ln,
                "valueclass": "mixed",
            })
            n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# np.where(cond, x, y) -> select. Result dtype = result_type(x, y); NumPy is the oracle.
WHERE_DT_PAIRS = [
    ("int32", "int32"), ("int32", "float64"), ("float32", "float64"), ("int32", "int64"),
    ("bool", "int32"), ("float64", "float64"), ("complex128", "float64"), ("uint8", "int8"),
    # W1: float16 + narrow-int select results.
    ("float16", "float16"), ("float16", "float32"), ("int8", "int16"), ("uint16", "uint16"),
    ("uint32", "int32"), ("int64", "uint64"),
]


# G4 (F4) — NON-bool where cond: NumPy selects by TRUTHINESS of any dtype cond (probed 2.4.2:
# NaN is truthy, -0.0 is falsy, complex is truthy iff re!=0 or im!=0). NumSharp matches
# (probed on all four cond dtypes). The float/complex pools front-load NaN/inf/-0.0, so the
# truthiness edges are exercised in every case.
WHERE_COND_DTYPES = ["int32", "float64", "uint8", "complex128"]
WHERE_COND_XY_PAIRS = [("int32", "int32"), ("float64", "int32"), ("float32", "float64")]


def gen_where_cond(cond_dtypes, xy_pairs):
    cases = []
    n = 0
    for cdt in cond_dtypes:
        for (sx, sy) in xy_pairs:
            # contiguous: cond/x/y all (4,5) C-contiguous
            cb = _cbase((4, 5), np.dtype(cdt))
            xb = _cbase((4, 5), np.dtype(sx))
            yb = _cbase((4, 5), np.dtype(sy))
            # strided: all three are [:, ::2] views of (4,10) bases
            cb2 = _cbase((4, 10), np.dtype(cdt))
            xb2 = _cbase((4, 10), np.dtype(sx))
            yb2 = _cbase((4, 10), np.dtype(sy))
            for (tag, c_pair, x_pair, y_pair) in [
                ("wh_cond_contig", (cb, cb), (xb, xb), (yb, yb)),
                ("wh_cond_strided", (cb2, cb2[:, ::2]), (xb2, xb2[:, ::2]), (yb2, yb2[:, ::2])),
            ]:
                r = np.where(c_pair[1], x_pair[1], y_pair[1])
                cases.append({
                    "id": f"where/{tag}/{cdt}-cond/{sx},{sy}/{n}",
                    "op": "where",
                    "params": {},
                    "operands": [describe(*c_pair), describe(*x_pair), describe(*y_pair)],
                    "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                 "buffer": np.ascontiguousarray(r).tobytes().hex()},
                    "layout": tag,
                    "valueclass": "mixed",
                })
                n += 1
    return cases


# T11 — cumulative scans (cumsum/cumprod) and finite differences (diff). NumPy is the oracle for
# value, NEP50 accumulator dtype (cumsum(int32)->int64), and the diff output shape (shrinks by n).
SCAN_OPS = {
    "cumsum": lambda a, ax: np.cumsum(a, axis=ax),
    "cumprod": lambda a, ax: np.cumprod(a, axis=ax),
}
SCAN_DTYPES = list(ALL_DTYPES)         # widened: cumsum/cumprod/diff are dtype-general (bool->int upcast)
SCAN_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "c_contiguous_3d", "f_contiguous_2d",
                "transposed_3d", "strided_2d_cols", "one_element_1d", "negstride_1d"]

# T11b — NaN-aware cumulative scans (nancumsum/nancumprod). NumPy's nancumsum/nancumprod are
# _replace_nan(a, 0|1) then cumsum/cumprod, so the oracle rides gen_scan over the SAME layouts and
# dtype set: the float pool front-loads nan/+-inf and the complex pool carries a NaN in BOTH the real
# and imaginary lanes, so every float/complex slice exercises the NaN -> identity replacement (nan->0
# for sum, nan->1 for prod; a complex element with EITHER lane NaN collapses to 0+0j / 1+0j). NumPy is
# the oracle for value, the NEP50 accumulator dtype (nancumsum(int32)->int64), and the
# leading-NaN/all-NaN-slice -> identity contract. Char has no NaN, so nancum(char) == cum(char) (already
# gated by the scan tier's char cumsum) and is not re-woven here.
NAN_SCAN_OPS = {
    "nancumsum": lambda a, ax: np.nancumsum(a, axis=ax),
    "nancumprod": lambda a, ax: np.nancumprod(a, axis=ax),
}


def gen_scan(ops, dtypes, layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            for opname, f in ops.items():
                for axis in _axes(view.ndim):
                    try:
                        r = np.asarray(f(view, axis))
                    except Exception:
                        skipped += 1
                        continue
                    cases.append({
                        "id": f"{opname}/{ln}/{s}/axis={axis}/{n}",
                        "op": opname,
                        "params": {"axis": axis},
                        "operands": [operand],
                        "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                     "buffer": np.ascontiguousarray(r).tobytes().hex()},
                        "layout": ln,
                        "valueclass": "mixed",
                    })
                    n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


def gen_diff(dtypes, layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            if view.ndim == 0:
                continue
            operand = describe(base, view)
            axes = [0] if view.ndim == 1 else [0, view.ndim - 1]
            for order in (1, 2):
                for axis in axes:
                    try:
                        r = np.asarray(np.diff(view, n=order, axis=axis))
                    except Exception:
                        skipped += 1
                        continue
                    cases.append({
                        "id": f"diff/{ln}/{s}/n={order}/axis={axis}/{n}",
                        "op": "diff",
                        "params": {"n": order, "axis": axis},
                        "operands": [operand],
                        "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                     "buffer": np.ascontiguousarray(r).tobytes().hex()},
                        "layout": ln,
                        "valueclass": "mixed",
                    })
                    n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


def gen_unwrap(dtypes, layout_names):
    # np.unwrap differential coverage: SCAN layouts x dtypes x a small parameter sweep.
    # Data comes from the layout catalog (nan/inf/extremes with large jumps that trigger the
    # correction / discont / boundary paths on the finite pairs, NaN tokenized on the rest).
    # A FLOAT-typed period keeps the float path (float input -> same width, integer input ->
    # float64); an INTEGER-typed period (period_is_int) selects NumPy's integer path for
    # integer/bool inputs. Complex (TypeError) and unsigned integer-period (OverflowError)
    # calls raise inside np.unwrap and are skipped here — their error parity is gated by the
    # unit tests — exactly as gen_scan/gen_diff skip cases where NumPy raises.
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            dt = np.dtype(s)
            base, view = fn(dt)
            if view.ndim == 0:
                continue
            operand = describe(base, view)
            axes = [0] if view.ndim == 1 else [0, view.ndim - 1]
            is_int_like = np.issubdtype(dt, np.integer) or dt == np.bool_
            # (call-kwargs, recorded-params). A Python int period flags the integer path.
            variants = [
                ({}, {"period_is_int": False}),
                ({"period": 4.0}, {"period": 4.0, "period_is_int": False}),
                ({"discont": 5.0}, {"discont": 5.0, "period_is_int": False}),
            ]
            if is_int_like:
                variants += [
                    ({"period": 4}, {"period": 4, "period_is_int": True}),   # even -> boundary ambiguous
                    ({"period": 5}, {"period": 5, "period_is_int": True}),   # odd  -> boundary NOT ambiguous
                    ({"period": 6}, {"period": 6, "period_is_int": True}),   # even
                ]
            for (kw, prec) in variants:
                for axis in axes:
                    callkw = dict(kw)
                    callkw["axis"] = axis
                    try:
                        r = np.asarray(np.unwrap(view, **callkw))
                    except Exception:
                        skipped += 1
                        continue
                    params = dict(prec)
                    params["axis"] = axis
                    ptag = f"p={prec.get('period', 'def')}{'i' if prec['period_is_int'] else ''}"
                    cases.append(_case("unwrap", params, [operand], _arr_expected(r), ln,
                                       "mixed", cid=f"unwrap/{ln}/{s}/{ptag}/axis={axis}/{n}"))
                    n += 1
    if skipped:
        print(f"  (skipped {skipped} unwrap cases where NumPy raised)")
    return cases


def gen_where(dt_pairs, layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = WHERE_LAYOUTS[ln]
        for (sx, sy) in dt_pairs:
            cb, cv, xb, xv, yb, yv = fn(np.dtype(sx), np.dtype(sy))
            try:
                r = np.where(cv, xv, yv)
            except Exception:
                skipped += 1
                continue
            cases.append({
                "id": f"where/{ln}/{sx},{sy}/{n}",
                "op": "where",
                "params": {},
                "operands": [describe(cb, cv), describe(xb, xv), describe(yb, yv)],
                "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                             "buffer": np.ascontiguousarray(r).tobytes().hex()},
                "layout": ln,
                "valueclass": "mixed",
            })
            n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# T13 — logic & element-wise extrema. isnan/isinf/isfinite (unary -> bool); maximum/minimum
# (NaN-propagating), fmax/fmin (NaN-ignoring), isclose (binary -> bool). NumPy is the oracle.
# signbit joins isnan/isinf/isfinite (unary -> bool). Unlike them it has NO complex loop, so the
# complex128 cases raise and gen_unary's per-case try/except skips them (the complex-error contract is
# gated by the dedicated np.signbit.Test.cs suite instead). Every other dtype is covered here.
LOGIC_UNARY_OPS = {"isnan": np.isnan, "isinf": np.isinf, "isfinite": np.isfinite, "signbit": np.signbit}
LOGIC_UNARY_DTYPES = list(ALL_DTYPES)  # widened: isnan/isinf/isfinite/signbit defined on every dtype
LOGIC_BIN_OPS = {
    "maximum": np.maximum, "minimum": np.minimum,
    "fmax": np.fmax, "fmin": np.fmin, "isclose": np.isclose,
}
LOGIC_BIN_PAIRS = [
    ("float32", "float32"), ("float64", "float64"), ("float16", "float16"),
    ("int32", "int32"), ("int32", "float64"), ("uint8", "int8"), ("int32", "int64"),
    ("complex128", "complex128"),
]

# G5 (F5) — iscomplex/isreal: FULL coverage — ALL dtypes (complex128 included) × EVERY layout.
# Ported to NumPy's own structure (_type_check_impl): isreal(x) = imag(x) == 0; iscomplex(x) =
# imag(x) != 0 for a complex dtype, else all-False. The complex128 pool carries nonzero imaginary
# parts (real + rolled-imag, with NaN/Inf/-0 among them), so this tier now exercises the imaginary
# inspection that was previously CARVED. The former carve had two causes, both fixed:
#   * complex128 input — NumSharp now inspects the imaginary lane (np.imag strided view) instead of
#     returning all-False/all-True regardless of value;
#   * strided/F-contiguous/transposed REAL input — the non-contiguous path emitted garbage bytes
#     because np.ones/np.zeros were handed the view's Shape (strides+offset); it now builds a fresh
#     C-contiguous bool array from the DIMENSIONS only.
ISCOMPLEX_OPS = {"iscomplex": np.iscomplex, "isreal": np.isreal}
ISCOMPLEX_DTYPES = list(ALL_DTYPES)

# Group A Batch 1: logical_and/or/xor (binary -> bool, truthiness of each element),
# logical_not (unary -> bool), arctan2 (binary -> float; NumPy promotes int -> float64,
# raises on complex so gen_binary skips those).
LOGICAL_BIN_OPS = {"logical_and": np.logical_and, "logical_or": np.logical_or, "logical_xor": np.logical_xor}
LOGICAL_NOT_OP = {"logical_not": np.logical_not}
ARCTAN2_OP = {"arctan2": np.arctan2}
LOGICAL_PAIRS = [
    ("bool", "bool"), ("int32", "int32"), ("float64", "float64"), ("bool", "int32"),
    ("int32", "float64"), ("uint8", "uint8"), ("float32", "float32"), ("complex128", "complex128"),
]
ARCTAN2_PAIRS = [
    ("float32", "float32"), ("float64", "float64"), ("float16", "float16"),
    ("int32", "int32"), ("int32", "float64"), ("uint8", "int8"), ("float32", "float64"),
]

# logaddexp / logaddexp2 / nextafter: binary float-tier ufuncs with arctan2's promotion
# (ee->e, ff->f, dd->d, gg->g). nextafter is bit-exact; logaddexp/logaddexp2 diverge <=2 ULP
# (managed fdlibm log1p vs NumPy's closed ucrtbase log1p) — excused in MisalignedRegistry.
LOGADDEXP_OPS = {"logaddexp": np.logaddexp, "logaddexp2": np.logaddexp2}
NEXTAFTER_OP = {"nextafter": np.nextafter}
# copysign: magnitude of x1 with the sign of x2. Same float-tier promotion; BIT-EXACT
# (Math.CopySign is the IEEE bit op), so no MisalignedRegistry excuse (NaN payloads are tokenized).
COPYSIGN_OP = {"copysign": np.copysign}
# hypot: sqrt(x1**2 + x2**2) without spurious overflow/underflow. Same float-tier promotion. float32/
# float16 are BIT-EXACT; float64 is <=1 ULP (NumSharp's correctly-rounded Borges FMA vs NumPy's only
# faithfully-rounded UCRT hypot) — excused Double-only in MisalignedRegistry.
HYPOT_OP = {"hypot": np.hypot}
# heaviside(x1, x2): 0 if x1<0, x2 if x1==0, 1 if x1>0, +NaN if x1 is NaN. Same float-tier promotion
# (ee/ff/dd/gg). BIT-EXACT at every dtype (the step is a port of npy_heaviside; the x1==0 fill passes
# x2's exact bits, a NaN x1 gives the canonical +NaN) — so NO MisalignedRegistry excuse. The x1==0 / NaN
# branches are gated by the `specials` tier (SPECIAL_BINARY_OPS), where the aligned pairs include
# (±0, ±inf), (±0, ±0) and (nan, *).
HEAVISIDE_OP = {"heaviside": np.heaviside}


# np.place(arr, mask, vals) mutates arr in-place where mask is True, cycling through vals.
# The operand is the ORIGINAL arr; the expected is arr AFTER place.
PLACE_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "c_contiguous_3d"]
PLACE_DTYPES = ["bool", "int32", "uint8", "float64", "complex128"]


def gen_place(dtypes, layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        for s in dtypes:
            arr_b, arr_v = LAYOUTS[ln](np.dtype(s))
            mask = (np.arange(arr_v.size).reshape(arr_v.shape) % 2 == 0)
            vals = np.arange(1, 4).astype(np.dtype(s))
            arr_after = np.array(arr_v, copy=True)
            try:
                np.place(arr_after, mask, vals)
            except Exception:
                skipped += 1
                continue
            cases.append({
                "id": f"place/{ln}/{s}/{n}",
                "op": "place",
                "params": {},
                "operands": [describe(arr_b, arr_v), describe(mask, mask), describe(vals, vals)],
                "expected": {"dtype": arr_after.dtype.name, "shape": [int(d) for d in arr_after.shape],
                             "buffer": np.ascontiguousarray(arr_after).tobytes().hex()},
                "layout": ln,
                "valueclass": "mixed",
            })
            n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# np.putmask(a, mask, values) mutates a in-place where mask is True, walking both in C-order.
# It is the sibling of np.place but differs in ONE probed way (NumPy 2.4.2): the values cursor
# advances by POSITION (every element), so a.flat[i] = values.flat[i % nv] — NumPy's
# npy_fastputmask, where j increments in lockstep with i (place advances only on True). The three
# value modes exercise BOTH kernel branches: "scalar" (nv==1, the cursor-free fast path), "cycle"
# (nv==3, the position cursor), and "long" (nv==size, cursor never resets — values indexed by
# position). "scalar"/"long" use a FLAT mask (a different shape, same size — NumPy checks size, not
# shape) while "cycle" uses arr's own shape; both align by C-order flat position. Non-contiguous
# layouts drive the ascontiguousarray+copyto writeback path. Same-dtype values throughout (NumPy
# casts values to a's dtype; the ndarray-vs-python casting-error split is a documented sibling
# simplification shared with place/put — see np.putmask.Test.cs).
PUTMASK_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "c_contiguous_3d",
                   "f_contiguous_2d", "transposed_3d", "negstride_1d",
                   "strided_step2_1d", "strided_2d_cols"]
PUTMASK_DTYPES = ["bool", "int8", "uint8", "int16", "int32", "int64", "uint64",
                  "float16", "float32", "float64", "complex128"]


def gen_putmask(dtypes, layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        for s in dtypes:
            arr_b, arr_v = LAYOUTS[ln](np.dtype(s))
            if arr_v.size == 0:
                continue
            dt = np.dtype(s)
            for vmode in ("scalar", "cycle", "long"):
                if vmode == "scalar":
                    mask = (np.arange(arr_v.size) % 2 == 0)                       # flat, different shape
                    vals = np.arange(3, 4).astype(dt)                            # nv == 1 (bool -> [True])
                elif vmode == "cycle":
                    mask = (np.arange(arr_v.size).reshape(arr_v.shape) % 3 != 0)  # arr's own shape
                    vals = np.arange(1, 4).astype(dt)                            # nv == 3
                else:  # long: nv == size, cursor never wraps; values indexed by position
                    mask = (np.arange(arr_v.size) % 4 == 1)                       # flat
                    vals = np.arange(1, arr_v.size + 1).astype(dt)               # nv == size
                arr_after = np.array(arr_v, copy=True)
                try:
                    np.putmask(arr_after, mask, vals)
                except Exception:
                    skipped += 1
                    continue
                cases.append({
                    "id": f"putmask/{ln}/{s}/{vmode}/{n}",
                    "op": "putmask",
                    "params": {},
                    "operands": [describe(arr_b, arr_v), describe(mask, mask), describe(vals, vals)],
                    "expected": {"dtype": arr_after.dtype.name, "shape": [int(d) for d in arr_after.shape],
                                 "buffer": np.ascontiguousarray(arr_after).tobytes().hex()},
                    "layout": ln,
                    "valueclass": "mixed",
                })
                n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# T8 — linear algebra: matmul / dot / outer. NumPy is the oracle for value, NEP50 result dtype,
# and the gufunc/broadcast output shape. Operands carry deterministic non-trivial values; the C/F
# layout variants exercise the stride-aware GEMM packers (an F-contiguous operand is a transposed
# view into a C-contiguous base, mirroring layout_catalog's f_contiguous pattern).
# W1: added float16 + the narrow integers (int8/int16/uint16/uint32/uint64) — exercises the
# stride-aware GEMM accumulator at every width (NumPy matmul preserves the input dtype, so e.g.
# int8@int8 -> int8 with modular overflow; float16@float16 -> float16).
# G3: added bool — NumPy matmul/dot/outer on bool run the AND/OR semiring with a bool result
# (probed 2.4.2: matmul(bool,bool) -> dtype bool, OR-of-ANDs).
MATMUL_DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                 "float16", "float32", "float64", "complex128", "bool"]

# (op, shapeA, shapeB) — spans the matmul gufunc shape space + dot/outer specifics.
MATMUL_SHAPE_CASES = [
    ("matmul", (2, 3), (3, 2)),               # 2-D x 2-D
    ("matmul", (4,), (4,)),                    # 1-D x 1-D -> 0-D (inner product)
    ("matmul", (2, 3), (3,)),                  # 2-D x 1-D -> 1-D
    ("matmul", (3,), (3, 2)),                  # 1-D x 2-D -> 1-D
    ("matmul", (2, 2, 3), (2, 3, 2)),          # batched 3-D
    ("matmul", (1, 2, 3), (4, 3, 2)),          # stack-broadcast batch
    ("matmul", (2, 3), (4, 3, 2)),             # 2-D x 3-D (lhs stack-broadcast)
    ("matmul", (2, 2, 3), (3,)),               # 3-D x 1-D
    ("matmul", (3,), (2, 3, 2)),               # 1-D x 3-D
    ("matmul", (2, 1, 3, 4), (1, 2, 4, 3)),    # 4-D batched broadcast
    ("dot", (2, 3), (3, 2)),                   # 2-D dot == matmul
    ("dot", (4,), (4,)),                       # 1-D dot -> scalar
    ("dot", (2, 3), (3,)),                     # matrix . vector
    ("dot", (3,), (3, 2)),                     # vector . matrix
    ("outer", (3,), (4,)),                     # outer product
    ("outer", (2, 3), (4,)),                   # outer flattens inputs
    ("outer", (5,), (2, 2)),
]
MATMUL_LAYOUTS = ["C", "F"]
_MATMUL_FNS = {"matmul": np.matmul, "dot": np.dot, "outer": np.outer}


def _mm_fill(shape, dt):
    """Deterministic, non-trivial operand values; kept small for ints so overflow stays legible."""
    n = int(np.prod(shape)) if shape else 1
    dtype = np.dtype(dt)
    if dtype.kind == "c":
        a = (((np.arange(n) % 7) - 3) + 1j * ((np.arange(n) % 5) - 2)).astype(dtype)
    elif dtype.kind == "b":
        a = (np.arange(n) % 3 != 1)                          # 2/3 True — AND/OR semiring gets a mix
    elif dtype.kind in "iu":
        a = ((np.arange(n) % 7) + 1).astype(dtype)          # 1..7 (uint-safe, positive)
    else:
        a = (((np.arange(n) % 11) - 5) * 0.5).astype(dtype)  # -2.5 .. 2.5
    return a.reshape(shape)


def _mm_layout(arr, layout):
    """(base, view) for the requested memory layout — base is ALWAYS C-contiguous (so base.tobytes()
    is its raw memory); an F-contiguous view is the C-contig transpose viewed back through .T."""
    if layout == "F" and arr.ndim >= 2:
        base = np.ascontiguousarray(arr.T)   # transposed data, C-contiguous
        view = base.T                        # logical `arr`, F-strided into base
        assert np.array_equal(view, arr)
        return base, view
    base = np.ascontiguousarray(arr)
    return base, base


def gen_matmul(shape_cases, dtypes, layouts):
    cases = []
    n = 0
    skipped = 0
    for (op, shA, shB) in shape_cases:
        f = _MATMUL_FNS[op]
        for dt in dtypes:
            A = _mm_fill(shA, dt)
            B = _mm_fill(shB, dt)
            for la in layouts:
                for lb in layouts:
                    baseA, viewA = _mm_layout(A, la)
                    baseB, viewB = _mm_layout(B, lb)
                    try:
                        r = np.asarray(f(viewA, viewB))
                    except Exception:
                        skipped += 1
                        continue
                    sa = "x".join(map(str, shA))
                    sb = "x".join(map(str, shB))
                    cases.append({
                        "id": f"{op}/{la}{lb}/{dt}/{sa}@{sb}/{n}",
                        "op": op,
                        "params": {},
                        "operands": [describe(baseA, viewA), describe(baseB, viewB)],
                        "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                     "buffer": np.ascontiguousarray(r).tobytes().hex()},
                        "layout": f"{la}{lb}",
                        "valueclass": "mixed",
                    })
                    n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# G14 — matmul edge layouts the C/F matrix misses: a NEGATIVE-STRIDE operand (B row-reversed
# via [::-1], nonzero offset) and the k=0 empty inner dimension ((2,0)@(0,3) -> (2,3) zeros;
# both probed against NumPy 2.4.2 and matching in NumSharp).
MATMUL_EDGE_DTYPES = ["int32", "float64", "complex128", "bool"]


def gen_matmul_edges(dtypes):
    cases = []
    n = 0
    for dt in dtypes:
        A = _mm_fill((2, 3), dt)
        Bbase = _mm_fill((3, 2), dt)
        Bneg = Bbase[::-1]                                    # negative row stride, offset != 0
        r = np.asarray(np.matmul(A, Bneg))
        cases.append({
            "id": f"matmul/negstride/{dt}/{n}", "op": "matmul", "params": {},
            "operands": [describe(A, A), describe(Bbase, Bneg)],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": "negstride", "valueclass": "mixed",
        })
        n += 1
        A0 = np.zeros((2, 0), dtype=np.dtype(dt))             # k=0: empty inner dim
        B0 = np.zeros((0, 3), dtype=np.dtype(dt))
        r0 = np.asarray(np.matmul(A0, B0))
        cases.append({
            "id": f"matmul/k0/{dt}/{n}", "op": "matmul", "params": {},
            "operands": [describe(A0, A0), describe(B0, B0)],
            "expected": {"dtype": r0.dtype.name, "shape": [int(d) for d in r0.shape],
                         "buffer": np.ascontiguousarray(r0).tobytes().hex()},
            "layout": "k0", "valueclass": "mixed",
        })
        n += 1
    return cases


# G15 — a zero-sized extent on a STACKED (>=3-D) operand. The gufunc signature is
# (n?,k),(k,m?)->(n?,m?), so a zero lands in one of two places and they behave differently:
# a zero in a stack dim (or in n / m) makes the RESULT empty, while a zero in k alone leaves it
# NON-empty and every entry is an EMPTY SUM, i.e. exactly zero (matmul_inner_noblas stores 0
# into the cell before its zero-trip accumulation loop). A stack `0` also broadcasts against `1`
# to 0, never to 1 — np.broadcast_shapes((0,), (1,)) is (0,).
# The whole family used to throw out of NumSharp's BatchedMatmul; the 2-D k=0 case above and the
# N-D `dot` routes below were already correct and are pinned here so they stay that way.
MATMUL_ZERODIM_CASES = [
    # (op, shapeA, shapeB) — every position of a zero, 3-D
    ("matmul", (0, 3, 4), (0, 4, 5)),      # zero stack dim   -> (0,3,5) empty
    ("matmul", (2, 3, 0), (2, 0, 5)),      # zero k           -> (2,3,5) ALL ZEROS
    ("matmul", (2, 0, 4), (2, 4, 5)),      # zero n           -> (2,0,5) empty
    ("matmul", (2, 3, 4), (2, 4, 0)),      # zero m           -> (2,3,0) empty
    ("matmul", (0, 0, 0), (0, 0, 0)),
    ("matmul", (0, 3, 0), (0, 0, 5)),      # stack + k
    ("matmul", (2, 0, 0), (2, 0, 5)),      # n + k
    ("matmul", (2, 0, 4), (2, 4, 0)),      # n + m
    ("matmul", (2, 3, 0), (2, 0, 0)),      # k + m
    # a zero stack dim against a broadcast 2-D operand, both orders
    ("matmul", (0, 3, 4), (4, 5)),
    ("matmul", (3, 4), (0, 4, 5)),
    ("matmul", (0, 3, 0), (0, 5)),
    ("matmul", (3, 0), (0, 0, 5)),
    # 0 against 1 in a stack dim stretches to 0, not 1
    ("matmul", (0, 3, 4), (1, 4, 5)),
    ("matmul", (1, 3, 4), (0, 4, 5)),
    ("matmul", (1, 1, 3, 4), (0, 1, 4, 5)),
    ("matmul", (1, 0, 3, 4), (2, 1, 4, 5)),   # a 0 and a >1 stretch in the same call
    # 4-D
    ("matmul", (0, 2, 3, 4), (0, 2, 4, 5)),
    ("matmul", (2, 0, 3, 4), (2, 0, 4, 5)),
    ("matmul", (2, 3, 4, 0), (2, 3, 0, 5)),   # zero k -> (2,3,4,5) ALL ZEROS
    ("matmul", (2, 3, 0, 4), (2, 3, 4, 5)),
    ("matmul", (2, 3, 4, 5), (2, 3, 5, 0)),
    ("matmul", (0, 2, 3, 4), (4, 5)),
    ("matmul", (2, 0, 3, 4), (4, 5)),
    # 1-D promotion around a zero extent (the inserted axis is squeezed back out)
    ("matmul", (0, 3, 4), (4,)),
    ("matmul", (3,), (0, 3, 4)),
    ("matmul", (2, 3, 0), (0,)),              # zero k -> (2,3) ALL ZEROS
    ("matmul", (0,), (2, 0, 5)),              # zero k -> (2,5) ALL ZEROS
    ("matmul", (2, 0, 4), (4,)),
    ("matmul", (4,), (2, 4, 0)),
    # np.dot's N-D route (dotfunc, NOT the gufunc) over the same degenerate shapes
    ("dot", (0, 3, 4), (4, 5)),
    ("dot", (2, 3, 0), (0, 5)),               # -> (2,3,5) ALL ZEROS
    ("dot", (0, 3, 4), (0, 4, 5)),
    ("dot", (2, 3, 0), (2, 0, 5)),            # -> (2,3,2,5) ALL ZEROS
    ("dot", (2, 0, 4), (4, 5)),
]

# The zero-sized operand carries no bytes, so a layout sweep over it is meaningless; the F pass
# exists for the operands that DO have data (the k=0 pair's outer dims, the broadcast 2-D side).
MATMUL_ZERODIM_LAYOUTS = ["C", "F"]


def gen_matmul_zerodim(dtypes):
    cases = []
    n = 0
    for (op, shA, shB) in MATMUL_ZERODIM_CASES:
        f = _MATMUL_FNS[op]
        for dt in dtypes:
            A = _mm_fill(shA, dt)
            B = _mm_fill(shB, dt)
            for lay in MATMUL_ZERODIM_LAYOUTS:
                baseA, viewA = _mm_layout(A, lay)
                baseB, viewB = _mm_layout(B, lay)
                r = np.asarray(f(viewA, viewB))
                sa = "x".join(map(str, shA))
                sb = "x".join(map(str, shB))
                cases.append({
                    "id": f"{op}/zerodim_{lay}/{dt}/{sa}@{sb}/{n}",
                    "op": op,
                    "params": {},
                    "operands": [describe(baseA, viewA), describe(baseB, viewB)],
                    "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                 "buffer": np.ascontiguousarray(r).tobytes().hex()},
                    "layout": f"zerodim_{lay}",
                    "valueclass": "degenerate",
                })
                n += 1
    return cases


# G16 — float16 contraction DEPTH. The matmul tier carries float16 already, but every shape in
# it has k <= 4, where a half accumulator and a float32 one agree exactly — so it could never see
# that NumPy accumulates half products in FLOAT32 and narrows once (`float sum = 0; ... *(op) =
# npy_float_to_half(sum)` in matmul_inner_noblas). Half is the ONLY float dtype NumPy does not
# send to cblas (USEBLAS=0 in matmul.c.src), so unlike float32/float64 this loop is reproducible
# exactly and belongs in the portable corpus rather than the host-pinned matmul_parity tier.
#
# Accumulating in half does not merely lose a ulp, it SATURATES: half's spacing above 2048 is 2,
# so ones(4096).ones(4096) stalls at 2048 where NumPy returns 4096.
MATMUL_HALF_KS = [8, 64, 256, 1024, 3000, 4096]


def _half_vals(n, vc):
    if vc == "ones":
        return np.ones(n)
    if vc == "tiny":          # 1.0 then values under half's resolution at 1.0 (2**-11)
        return np.concatenate([[1.0], np.full(max(n - 1, 0), 2.0 ** -11)])[:n]
    if vc == "ramp":
        return ((np.arange(n) % 11) - 5) * 0.5
    if vc == "alt":           # cancelling signs — the sum stays small while partials do not
        return np.where(np.arange(n) % 2 == 0, 3.5, -3.25)
    raise ValueError(vc)


def gen_matmul_half_depth():
    cases = []
    n = 0
    h = np.dtype("float16")
    for K in MATMUL_HALF_KS:
        for vc in ("ones", "tiny", "ramp", "alt"):
            A = _half_vals(3 * K, vc).astype(h).reshape(3, K)
            B = _half_vals(K * 2, vc).astype(h).reshape(K, 2)
            for la, lb in (("C", "C"), ("F", "C"), ("C", "F")):
                baseA, viewA = _mm_layout(A, la)
                baseB, viewB = _mm_layout(B, lb)
                for op in ("matmul", "dot"):
                    r = np.asarray(_MATMUL_FNS[op](viewA, viewB))
                    cases.append({
                        "id": f"{op}/halfdepth_{la}{lb}_{vc}/float16/k{K}/{n}",
                        "op": op, "params": {},
                        "operands": [describe(baseA, viewA), describe(baseB, viewB)],
                        "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                     "buffer": np.ascontiguousarray(r).tobytes().hex()},
                        "layout": f"halfdepth_{la}{lb}", "valueclass": vc,
                    })
                    n += 1

            # 1-D inner product (its own kernel) and the batched stack (the gufunc outer loop)
            v1 = _half_vals(K, vc).astype(h)
            v2 = _half_vals(K, vc).astype(h)[::-1].copy()
            for op in ("matmul", "dot"):
                r = np.asarray(_MATMUL_FNS[op](v1, v2))
                cases.append({
                    "id": f"{op}/halfdepth_vec_{vc}/float16/k{K}/{n}",
                    "op": op, "params": {},
                    "operands": [describe(v1, v1), describe(v2, v2)],
                    "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                 "buffer": np.ascontiguousarray(r).tobytes().hex()},
                    "layout": "halfdepth_vec", "valueclass": vc,
                })
                n += 1

            bA = _half_vals(2 * 3 * K, vc).astype(h).reshape(2, 3, K)
            bB = _half_vals(2 * K * 2, vc).astype(h).reshape(2, K, 2)
            r = np.asarray(np.matmul(bA, bB))
            cases.append({
                "id": f"matmul/halfdepth_batch_{vc}/float16/k{K}/{n}",
                "op": "matmul", "params": {},
                "operands": [describe(bA, bA), describe(bB, bB)],
                "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                             "buffer": np.ascontiguousarray(r).tobytes().hex()},
                "layout": "halfdepth_batch", "valueclass": vc,
            })
            n += 1

            # MIXED operands that still promote to float16 (bool / int8 / uint8) — a separate
            # kernel in NumSharp, and it needs the same float32 accumulator.
            for other in ("int8", "uint8", "bool"):
                o = np.dtype(other)
                Bm = (np.ones(K * 2) if o.kind == "b" else ((np.arange(K * 2) % 5) + 1)).astype(o).reshape(K, 2)
                r = np.asarray(np.matmul(A, Bm))
                cases.append({
                    "id": f"matmul/halfdepth_mixed_{other}_{vc}/float16/k{K}/{n}",
                    "op": "matmul", "params": {},
                    "operands": [describe(A, A), describe(Bm, Bm)],
                    "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                 "buffer": np.ascontiguousarray(r).tobytes().hex()},
                    "layout": "halfdepth_mixed", "valueclass": vc,
                })
                n += 1
    return cases


# T9 — bitwise & shift. NumPy defines bitwise_and/or/xor & invert for integer + bool; the shifts
# for integers. Float/complex raise TypeError (gen_binary/gen_unary skip those automatically).
BITWISE_BIN_OPS = {
    "bitwise_and": np.bitwise_and,
    "bitwise_or": np.bitwise_or,
    "bitwise_xor": np.bitwise_xor,
}
INVERT_OP = {"invert": np.invert}
# np.bitwise_count (NumPy 2.0): popcount of |x|, integer/bool input -> uint8 output. Unlike invert
# it is NOT carved for Char (its 2-byte SIMD path works), so char_tier weaves it in.
BITWISE_COUNT_OP = {"bitwise_count": np.bitwise_count}
INT_BOOL_DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]
BITWISE_DT_PAIRS = [
    ("int32", "int32"), ("uint8", "uint8"), ("int8", "int8"), ("int16", "int16"),
    ("uint16", "uint16"), ("uint32", "uint32"), ("int64", "int64"), ("uint64", "uint64"),
    ("bool", "bool"), ("int32", "int64"), ("uint8", "int8"), ("int32", "uint32"),
    ("bool", "int32"), ("int8", "int16"), ("uint16", "uint32"), ("int64", "uint64"),
]

SHIFT_OPS = {"left_shift": np.left_shift, "right_shift": np.right_shift}
SHIFT_DTYPES = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]

# gcd / lcm — number-theoretic binary ufuncs, INTEGER-ONLY (no bool/float/complex loop; a bool+bool
# or any float/complex operand, and the uint64+signed -> float64 NEP50 pair, raise the no-loop error
# — gen_binary skips those, error-parity is gated in errors_full). Both operands + output share one
# promoted integer dtype (NumPy's uniform resolver). A bool paired with an integer promotes to that
# integer loop and IS valid, so ("bool","int32") stays in the pair list.
GCDLCM_OPS = {"gcd": np.gcd, "lcm": np.lcm}
GCDLCM_DT_PAIRS = [
    ("int32", "int32"), ("uint8", "uint8"), ("int8", "int8"), ("int16", "int16"),
    ("uint16", "uint16"), ("uint32", "uint32"), ("int64", "int64"), ("uint64", "uint64"),
    ("int32", "int64"), ("uint8", "int8"), ("int32", "uint32"), ("int8", "int16"),
    ("uint16", "uint32"), ("bool", "int32"),
]


def gen_shift(ops, dtypes):
    """Shift kernels with shift-count edges that straddle the bit width — tests NumPy's
    overflow-shift semantics (shift >= width -> 0, or -1 for signed-negative right shift).
    Contiguous 1-D operands; counts are in the operand dtype so result dtype == operand dtype."""
    cases = []
    n = 0
    for s in dtypes:
        w = np.dtype(s).itemsize * 8
        counts = [0, 1, 2, 3, 5, 7, w - 1, w, w + 1, 2 * w]
        left = _fill(len(counts), np.dtype(s))
        cnt = np.array([c % (2 ** w) if np.dtype(s).kind == "u" else c for c in counts], dtype=np.dtype(s))
        for opname, f in ops.items():
            r = np.asarray(f(left, cnt))
            cases.append({
                "id": f"{opname}/shift_edges/{s}/{n}",
                "op": opname,
                "params": {},
                "operands": [describe(left, left), describe(cnt, cnt)],
                "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                             "buffer": np.ascontiguousarray(r).tobytes().hex()},
                "layout": "shift_edges",
                "valueclass": "shift",
            })
            n += 1
    return cases


# T7 — shape manipulation. These ops only move bytes, so dtype coverage is light but stride/shape
# coverage is heavy. NumPy is the oracle for the output shape, dtype, and C-contiguous bytes.
MANIP_DTYPES = list(ALL_DTYPES)        # widened: reshape/transpose/concat/stack/pad are dtype-agnostic


def gen_manip(dtypes, layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            sz = int(view.size)
            nd = view.ndim
            jobs = [
                ("ravel", {}, lambda v: np.ravel(v)),
                ("copy", {}, lambda v: np.copy(v, order="K")),
                ("resize", {"shape": [2, 3]}, lambda v: np.resize(v, (2, 3))),
                ("transpose", {}, lambda v: np.transpose(v)),
                ("expand_dims", {"axis": 0}, lambda v: np.expand_dims(v, 0)),
                ("squeeze", {}, lambda v: np.squeeze(v)),
                ("roll", {"shift": 1}, lambda v: np.roll(v, 1)),
                ("repeat", {"repeats": 2}, lambda v: np.repeat(v, 2)),
                ("tile", {"reps": 2}, lambda v: np.tile(v, 2)),
                ("atleast_1d", {}, lambda v: np.atleast_1d(v)),
                ("atleast_2d", {}, lambda v: np.atleast_2d(v)),
                ("atleast_3d", {}, lambda v: np.atleast_3d(v)),
                ("flip", {}, lambda v: np.flip(v)),            # reverse ALL axes (0-d -> scalar)
                # byteswap: reverse each element's bytes (dtype preserved, values reinterpreted).
                # ndarray METHOD (no np.byteswap); not-inplace never raises, so no nd/sz guard.
                # Complex swaps its two halves; 1-byte dtypes are a value no-op (still a copy).
                ("byteswap", {}, lambda v: v.byteswap()),
                # packbits: bool/integer -> uint8 bit-pack (float/complex/decimal RAISE -> skipped by
                # the try/except below; Char rides char_tier via the uint16 proxy). axis=None flattens
                # in C order. unpackbits: uint8 ONLY (every other dtype raises -> skipped), axis=None.
                ("packbits", {}, lambda v: np.packbits(v)),
                ("packbits", {"bitorder": "little"}, lambda v: np.packbits(v, bitorder="little")),
                ("unpackbits", {}, lambda v: np.unpackbits(v)),
                ("unpackbits", {"bitorder": "little"}, lambda v: np.unpackbits(v, bitorder="little")),
            ]
            if sz > 0:
                jobs.append(("reshape", {"shape": [sz]}, lambda v, sz=sz: v.reshape(sz)))
                # unpackbits count (truncate / trim-from-end). GUARDED to sz > 0: NumPy leaks
                # UNINITIALISED memory for a count-forced non-empty output over an EMPTY input (its
                # IterAllButAxis never runs the zeroing loop), a divergence NumSharp intentionally does
                # NOT reproduce (it returns the documented zeros) — so those cases are excluded from the
                # byte-exact corpus and pinned by np.packbits.Test.cs instead. count=-3 needs >=3 bits,
                # which any sz>0 uint8 (>=8 bits) has.
                jobs.append(("unpackbits", {"count": 5}, lambda v: np.unpackbits(v, count=5)))
                jobs.append(("unpackbits", {"count": -3}, lambda v: np.unpackbits(v, count=-3)))
            if nd >= 1:
                # flipud (>= 1-d) + single-axis flip (int overload). trim_zeros is value-dependent:
                # the int/uint pools are front-loaded with 0 and the float pool carries 0.0/-0.0 amid
                # nan/inf, so f / b / fb each exercise real leading/trailing edge cropping (not no-ops).
                jobs.append(("flipud", {}, lambda v: np.flipud(v)))
                jobs.append(("flip", {"axis": 0}, lambda v: np.flip(v, 0)))
                jobs.append(("trim_zeros", {"trim": "fb"}, lambda v: np.trim_zeros(v, "fb")))
                jobs.append(("trim_zeros", {"trim": "f"}, lambda v: np.trim_zeros(v, "f")))
                jobs.append(("trim_zeros", {"trim": "b"}, lambda v: np.trim_zeros(v, "b")))
                jobs.append(("trim_zeros", {"trim": "fb", "axis": 0},
                             lambda v: np.trim_zeros(v, "fb", axis=0)))
                # packbits/unpackbits along an explicit axis (first + innermost, both bit orders).
                # nd>=1 so the axis is valid; unpackbits stays uint8-only via the try/except skip.
                jobs.append(("packbits", {"axis": 0}, lambda v: np.packbits(v, axis=0)))
                jobs.append(("packbits", {"axis": nd - 1, "bitorder": "little"},
                             lambda v, nd=nd: np.packbits(v, axis=nd - 1, bitorder="little")))
                jobs.append(("unpackbits", {"axis": 0}, lambda v: np.unpackbits(v, axis=0)))
                jobs.append(("unpackbits", {"axis": nd - 1, "bitorder": "little"},
                             lambda v, nd=nd: np.unpackbits(v, axis=nd - 1, bitorder="little")))
                # tril/triu apply to the LAST TWO axes; a 1-D input squares up to (n, n)
                # (NumPy's `tri(*m.shape[-2:])` quirk) and a 0-d input raises, so nd >= 1.
                # Same-shape for nd >= 2, so no corpus blow-up; k spans keep/drop/saturate.
                jobs.append(("tril", {}, lambda v: np.tril(v)))
                jobs.append(("triu", {}, lambda v: np.triu(v)))
                jobs.append(("tril", {"k": 1}, lambda v: np.tril(v, 1)))
                jobs.append(("triu", {"k": 1}, lambda v: np.triu(v, 1)))
                jobs.append(("tril", {"k": -1}, lambda v: np.tril(v, -1)))
                jobs.append(("triu", {"k": -1}, lambda v: np.triu(v, -1)))
            if nd in (1, 2):
                # diag's two branches differ in kind: 1-D CONSTRUCTS an (n+|k|)^2 matrix,
                # 2-D EXTRACTS a read-only diagonal view. Both stay small here (n <= 8).
                jobs.append(("diag", {}, lambda v: np.diag(v)))
                jobs.append(("diag", {"k": 1}, lambda v: np.diag(v, 1)))
                jobs.append(("diag", {"k": -1}, lambda v: np.diag(v, -1)))
            if sz <= 8:
                # diagflat squares the FULL size, so it is capped here to keep the corpus
                # small; gen_diag_tri covers the bigger/strided shapes at controlled sizes.
                jobs.append(("diagflat", {}, lambda v: np.diagflat(v)))
                jobs.append(("diagflat", {"k": 2}, lambda v: np.diagflat(v, 2)))
            if nd >= 2:
                jobs.append(("swapaxes", {"a1": 0, "a2": nd - 1}, lambda v, nd=nd: np.swapaxes(v, 0, nd - 1)))
                jobs.append(("moveaxis", {"src": 0, "dst": nd - 1}, lambda v, nd=nd: np.moveaxis(v, 0, nd - 1)))
                jobs.append(("delete", {"obj": 0, "axis": 0}, lambda v: np.delete(v, 0, axis=0)))
                # rot90's three non-trivial k values exercise its three distinct paths:
                # k=1 flip+transpose, k=2 double-flip, k=3 transpose+flip. Default plane (0, 1)
                # plus the reversed plane (1, 0) — the inverse direction, axes[0] > axes[1].
                jobs.append(("rot90", {"k": 1, "axes": [0, 1]}, lambda v: np.rot90(v, 1, (0, 1))))
                jobs.append(("rot90", {"k": 2, "axes": [0, 1]}, lambda v: np.rot90(v, 2, (0, 1))))
                jobs.append(("rot90", {"k": 3, "axes": [0, 1]}, lambda v: np.rot90(v, 3, (0, 1))))
                jobs.append(("rot90", {"k": 1, "axes": [1, 0]}, lambda v: np.rot90(v, 1, (1, 0))))
                # fliplr + the transpose aliases (permute_dims == transpose; matrix_transpose swaps the
                # last two axes) + the int[]-axes forms of flip / trim_zeros — all pure O(1)/O(ndim) views.
                jobs.append(("fliplr", {}, lambda v: np.fliplr(v)))
                jobs.append(("flip", {"axes": [0, nd - 1]}, lambda v, nd=nd: np.flip(v, (0, nd - 1))))
                jobs.append(("permute_dims", {}, lambda v: np.permute_dims(v)))
                jobs.append(("matrix_transpose", {}, lambda v: np.matrix_transpose(v)))
                jobs.append(("trim_zeros", {"trim": "fb", "axes": [nd - 1]},
                             lambda v, nd=nd: np.trim_zeros(v, "fb", axis=(nd - 1,))))
                if nd >= 3:
                    # non-default planes: a non-adjacent pair, and a negative-axis pair.
                    jobs.append(("rot90", {"k": 1, "axes": [0, nd - 1]},
                                 lambda v, nd=nd: np.rot90(v, 1, (0, nd - 1))))
                    jobs.append(("rot90", {"k": 3, "axes": [-1, -2]},
                                 lambda v: np.rot90(v, 3, (-1, -2))))
                    # explicit-axes permutation (axis roll) — the permute_dims axes path.
                    jobs.append(("permute_dims", {"axes": list(range(1, nd)) + [0]},
                                 lambda v, nd=nd: np.permute_dims(v, tuple(range(1, nd)) + (0,))))
            for (opname, params, f) in jobs:
                try:
                    r = np.asarray(f(view))
                except Exception:
                    skipped += 1
                    continue
                cases.append({
                    "id": f"{opname}/{ln}/{s}/{n}",
                    "op": opname,
                    "params": params,
                    "operands": [operand],
                    "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                 "buffer": np.ascontiguousarray(r).tobytes().hex()},
                    "layout": ln,
                    "valueclass": "mixed",
                })
                n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


def gen_concat_stack(dtypes):
    """Two-operand join ops (concatenate/stack/hstack/vstack/dstack). The second operand is a
    rolled copy so the two halves are distinguishable; one strided case exercises non-contig joins."""
    cases = []
    n = 0
    skipped = 0
    pairs = []  # (label, a_base, a_view, b_base, b_view, shape ndim)
    for sh in [(3,), (2, 3), (2, 3, 4)]:
        for s in dtypes:
            a = _cbase(sh, np.dtype(s))
            b = np.ascontiguousarray(np.roll(a, 1))
            pairs.append((f"contig{len(sh)}d", s, a, a, b, b))
    # one strided pair: (4,6)[:, ::2] -> (4,3)
    for s in dtypes:
        a = _cbase((4, 6), np.dtype(s))
        b = _cbase((4, 6), np.dtype(s))
        pairs.append(("strided2d", s, a, a[:, ::2], b, b[:, ::2]))

    for (label, s, ab, av, bb, bv) in pairs:
        opnd = [describe(ab, av), describe(bb, bv)]
        nd = av.ndim
        jobs = [("hstack", {}, lambda x, y: np.hstack([x, y])),
                ("vstack", {}, lambda x, y: np.vstack([x, y])),
                ("dstack", {}, lambda x, y: np.dstack([x, y]))]
        for axis in range(nd):
            jobs.append((f"concatenate", {"axis": axis}, lambda x, y, axis=axis: np.concatenate([x, y], axis=axis)))
        for axis in range(nd + 1):
            jobs.append((f"stack", {"axis": axis}, lambda x, y, axis=axis: np.stack([x, y], axis=axis)))
        for (opname, params, f) in jobs:
            try:
                r = np.asarray(f(av, bv))
            except Exception:
                skipped += 1
                continue
            cases.append({
                "id": f"{opname}/{label}/{s}/axis={params.get('axis')}/{n}",
                "op": opname,
                "params": params,
                "operands": opnd,
                "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                             "buffer": np.ascontiguousarray(r).tobytes().hex()},
                "layout": label,
                "valueclass": "mixed",
            })
            n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# ---------------------------------------------------------------------------
# np.r_ / np.c_ / np.ix_ — the index-expression DSL (numpy/lib/_index_tricks_impl.py).
#
# These ops take an INDEX EXPRESSION, not a plain operand list, so the corpus carries the
# non-array parts in `params` and the array parts as ordinary operands:
#
#   params.kind       "r" | "c"                     which concatenator
#   params.directive  str | null                    NumPy's leading special directive
#   params.exprs      [str, ...]                    slice expressions, in NumSharp spelling
#   params.scalars    [[kind, value], ...]          weak python-scalar tail; kind i|f|b
#   operands          the array entries, in order
#
# The C# side rebuilds  [directive?] + exprs + operands + scalars  and indexes np.r_/np.c_.
# C# has no slice literal, so every expression is paired with the Python slice it must mean —
# the pair is what keeps NumSharp's string grammar honest against NumPy's syntax.
# ---------------------------------------------------------------------------

# (NumSharp spelling, Python slice) — arange branch, then the imaginary-step linspace branch.
R_SLICE_EXPRS = [
    ("0:5", slice(0, 5)),
    ("0:5:2", slice(0, 5, 2)),
    ("5:0:-1", slice(5, 0, -1)),
    ("5:0", slice(5, 0)),
    ("-3:0", slice(-3, 0)),
    ("3:-3:-1", slice(3, -3, -1)),
    (":5", slice(None, 5)),
    (":5:2", slice(None, 5, 2)),
    ("2:", slice(2, None)),
    ("5::2", slice(5, None, 2)),
    ("::2", slice(None, None, 2)),
    (":", slice(None, None)),
    ("0.0:1.0:0.25", slice(0.0, 1.0, 0.25)),
    ("0:1:0.3", slice(0, 1, 0.3)),
    ("2.5:", slice(2.5, None)),
    ("-1:1:6j", slice(-1, 1, 6j)),
    ("0:1:5j", slice(0, 1, 5j)),
    ("0:5:0j", slice(0, 5, 0j)),
    ("0:5:1j", slice(0, 5, 1j)),
    ("0:3:2j", slice(0, 3, 2j)),
    ("1:2:-3j", slice(1, 2, -3j)),
]

# Weak python-scalar tails. The kind letter picks the C# boxed type (long / double / bool)
# so the NEP50 weak-vs-strong mapping is under the gate, not just the values.
_SCALAR_KIND = {"i": int, "f": float, "b": bool, "u": int}


def _scalar_py(entry):
    kind, value = entry
    return _SCALAR_KIND[kind](value)


def gen_index_tricks(dtypes):
    """np.r_ / np.c_ / np.ix_ — the index-expression DSL.

    Four groups:
      1. r_/c_ over ARRAY entries at 1-D and 2-D layouts x dtype, bare and with a leading
         directive, plus weak-scalar tails (the NEP50 promotion matrix).
      2. r_ over pure SLICE expressions — no operands at all, since the dtype comes from the
         literals (int64 for integer literals, float64 the moment one is written as a float
         or the step is imaginary). Also directive x slice, which exercises the slice branch's
         swapaxes(-1, trans1d) rather than the array branch's defaxes permutation.
      3. ix_ over 1..3 one-dimensional operands, incl. bool masks (the nonzero branch);
         `which` selects the recorded tuple element, as gen_nonzero does.
      4. Weak-integer OVERFLOW: NumPy raises OverflowError rather than wrapping, so these
         carry expects_throw.
    """
    cases = []
    n = 0

    def emit(opname, params, operands, r, layout):
        nonlocal n
        r = np.asarray(r)
        cases.append({
            "id": f"{opname}/{layout}/{n}",
            "op": opname,
            "params": params,
            "operands": operands,
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": layout,
            "valueclass": "mixed",
        })
        n += 1

    def emit_throw(opname, params, operands, layout):
        nonlocal n
        cases.append({
            "id": f"{opname}/{layout}/{n}",
            "op": opname,
            "params": params,
            "operands": operands,
            "expects_throw": True,
            "layout": layout,
            "valueclass": "mixed",
        })
        n += 1

    def build(kind, directive, exprs, arrays, scalars):
        """The Python index expression a NumSharp `np.r_[...]` / `np.c_[...]` must equal."""
        key = []
        if directive is not None:
            key.append(directive)
        key.extend(sl for _, sl in exprs)
        key.extend(arrays)
        key.extend(_scalar_py(s) for s in scalars)
        obj = np.r_ if kind == "r" else np.c_
        return obj[tuple(key)]

    def params_of(kind, directive, exprs, scalars):
        return {"kind": kind, "directive": directive,
                "exprs": [s for s, _ in exprs], "scalars": scalars}

    # -- 1. r_ / c_ over array entries -------------------------------------------------
    for dt in dtypes:
        b1 = _cbase((8,), np.dtype(dt))
        b2 = _cbase((3, 4), np.dtype(dt))
        b2t = _cbase((4, 3), np.dtype(dt))
        b2w = _cbase((3, 8), np.dtype(dt))
        b2o = _cbase((5, 4), np.dtype(dt))

        views1 = [
            ("c_1d", b1, b1),
            ("step_1d", b1, b1[::2]),
            ("negstride_1d", b1, b1[::-1]),
            ("offset_1d", b1, b1[2:7]),
        ]
        views2 = [
            ("c_2d", b2, b2),
            ("f_2d", b2t, b2t.T),
            ("strided_2d", b2w, b2w[:, ::2]),
            ("negstride_2d", b2, b2[::-1]),
            ("offset_2d", b2o, b2o[1:4]),
        ]

        for (tag, base, view) in views1:
            desc = describe(base, view)
            for kind in ("r", "c"):
                for directive in (None, "0,2", "0,2,0", "1,2,0", "0,3,1", "r", "c"):
                    try:
                        r = build(kind, directive, [], [view, view], [])
                    except Exception:
                        continue
                    emit(f"{kind}_", params_of(kind, directive, [], []),
                         [desc, desc], r, f"{tag}/{directive}/{dt}")

            # Weak-scalar tails: the NEP50 promotion matrix (weak int / float / bool
            # adopting an array dtype), which no other tier reaches.
            for scalars in ([["i", 0]], [["f", 1.5]], [["i", 5], ["i", 6]], [["b", True]]):
                for kind in ("r", "c"):
                    try:
                        r = build(kind, None, [], [view], scalars)
                    except Exception:
                        continue
                    emit(f"{kind}_", params_of(kind, None, [], scalars),
                         [desc], r, f"{tag}/scalars/{dt}")

        for (tag, base, view) in views2:
            desc = describe(base, view)
            for kind in ("r", "c"):
                for directive in (None, "-1", "0", "0,3,0", "0,3,1", "r"):
                    try:
                        r = build(kind, directive, [], [view, view], [])
                    except Exception:
                        continue
                    emit(f"{kind}_", params_of(kind, directive, [], []),
                         [desc, desc], r, f"{tag}/{directive}/{dt}")

        # Mixed slice-expression + array entries: the two branches meet in one concatenate,
        # so the slice's strong int64/float64 must promote against the operand dtype. The
        # C# side builds exprs before operands, so the expression leads here too.
        desc1 = describe(b1, b1)
        for expr in [("0:3", slice(0, 3)), ("0:1:5j", slice(0, 1, 5j))]:
            for kind in ("r", "c"):
                try:
                    r = (np.r_ if kind == "r" else np.c_)[(expr[1], b1)]
                except Exception:
                    continue
                emit(f"{kind}_", params_of(kind, None, [expr], []),
                     [desc1], r, f"mixed/{expr[0]}/{dt}")

    # -- 2. r_ / c_ over pure slice expressions (no operands, dtype from the literals) ---
    for (s, sl) in R_SLICE_EXPRS:
        for kind in ("r", "c"):
            try:
                r = (np.r_ if kind == "r" else np.c_)[(sl,)]
            except Exception:
                continue
            emit(f"{kind}_", params_of(kind, None, [(s, sl)], []), [], r, f"expr/{s}")

    # Two expressions concatenated, and directive x expression (the slice branch's
    # ndmin + swapaxes(-1, trans1d) path, which differs from the array branch's transpose).
    for (s1, sl1) in R_SLICE_EXPRS[:8]:
        for (s2, sl2) in [("0:3", slice(0, 3)), ("1:2:3j", slice(1, 2, 3j))]:
            try:
                r = np.r_[(sl1, sl2)]
            except Exception:
                continue
            emit("r_", params_of("r", None, [(s1, sl1), (s2, sl2)], []), [], r,
                 f"expr2/{s1}+{s2}")

    for directive in ("0,2", "0,2,0", "1,2,0", "0,3,0", "0,3,1", "0,3,2", "0,4,1", "r", "c"):
        for (s, sl) in [("0:3", slice(0, 3)), ("-1:1:4j", slice(-1, 1, 4j))]:
            for kind in ("r", "c"):
                try:
                    r = (np.r_ if kind == "r" else np.c_)[(directive, sl)]
                except Exception:
                    continue
                emit(f"{kind}_", params_of(kind, directive, [(s, sl)], []), [], r,
                     f"expr_dir/{directive}/{s}")

    # Slice expressions with a weak-scalar tail — arange/linspace strong dtype vs weak literal.
    for (s, sl) in [("0:3", slice(0, 3)), ("0:1:3j", slice(0, 1, 3j))]:
        for scalars in ([["i", 7]], [["f", 1.5]], [["b", True]]):
            try:
                r = np.r_[tuple([sl] + [_scalar_py(x) for x in scalars])]
            except Exception:
                continue
            emit("r_", params_of("r", None, [(s, sl)], scalars), [], r, f"expr_scalar/{s}")

    # All-weak keys: no array anywhere, so the NEP50 defaults decide (int64/float64/bool).
    for scalars in ([["i", 1], ["i", 2]], [["b", True], ["b", False]],
                    [["i", 1], ["f", 2.0]], [["b", True], ["i", 2]], [["f", 3.5]]):
        for kind in ("r", "c"):
            r = (np.r_ if kind == "r" else np.c_)[tuple(_scalar_py(x) for x in scalars)]
            emit(f"{kind}_", params_of(kind, None, [], scalars), [], r,
                 "weak_only/" + "".join(k for k, _ in scalars))

    # -- 3. ix_ ------------------------------------------------------------------------
    for dt in dtypes:
        b = _cbase((8,), np.dtype(dt))
        seqs = [
            ("c_1d", b, b[:4]),
            ("step_1d", b, b[::2]),
            ("negstride_1d", b, b[::-1]),
            ("offset_1d", b, b[3:7]),
            ("empty_1d", b, b[4:4]),
        ]
        for (tag, base, view) in seqs:
            # 1-seq, 2-seq and 3-seq forms: the output rank equals the number of sequences,
            # and `which` walks every slot so each reshape target is compared.
            other = _cbase((3,), np.dtype("int64"))
            groups = [
                ("n1", [describe(base, view)], [view]),
                ("n2", [describe(base, view), describe(other, other)], [view, other]),
                ("n3", [describe(base, view), describe(other, other), describe(base, view)],
                 [view, other, view]),
            ]
            for (gtag, descs, arrays) in groups:
                try:
                    out = np.ix_(*arrays)
                except Exception:
                    continue
                for which in range(len(out)):
                    emit("ix_", {"which": which}, descs, out[which], f"{tag}/{gtag}/{dt}")

    # bool operands take ix_'s nonzero branch (mask -> intp indices).
    for mask in [[True, False, True, True], [False, False, False], [True], [True, True]]:
        m = np.array(mask, dtype=bool)
        other = np.array([1, 2], dtype=np.int64)
        out = np.ix_(m, other)
        for which in range(len(out)):
            emit("ix_", {"which": which},
                 [describe(m, m), describe(other, other)], out[which],
                 f"boolmask/{len(mask)}")
        out1 = np.ix_(m)
        emit("ix_", {"which": 0}, [describe(m, m)], out1[0], f"boolmask1/{len(mask)}")

    # -- 4. weak-integer overflow: NumPy raises OverflowError, it does NOT wrap ----------
    for (dt, value) in [("int8", 1000), ("int8", -1000), ("uint8", -1), ("uint8", 300),
                        ("int16", -40000), ("uint16", -1), ("int32", 2 ** 40),
                        ("uint64", -1), ("bool", 2)]:
        b = _cbase((4,), np.dtype(dt))
        try:
            _ = np.r_[(b, value)]
        except OverflowError:
            emit_throw("r_", params_of("r", None, [], [["i", value]]),
                       [describe(b, b)], f"overflow/{dt}/{value}")
        except Exception:
            continue

    # -- 5. edge sweep: ndmin at NumPy's ceiling, and the uint64 weak-integer default ----
    # `ndmin` reaches array(..., ndmin=n) from a user-typed directive, so it is swept over
    # every entry kind. Only ndmin=64 is emitted, and deliberately so: past it NumPy raises
    # `ndmin must be <= ndmax (64)` (NPY_MAXDIMS) while NumSharp has no 64-dimension ceiling
    # anywhere and happily builds the array, so there is no NumPy answer to bit-compare and
    # `expects_throw` would assert a limitation NumSharp does not have. That divergence is
    # pinned instead by np.r_.Test.cs -> R_HighNdmin_IsSupportedAndCheap, which asserts the
    # rank AND guards the O(ndim) expansion (the per-axis loop it replaced was quadratic:
    # 27.6 s at ndmin=100_000, unbounded at 2**31-1). Do not re-add the >64 rows here.
    b1 = _cbase((2,), np.dtype("int64"))
    for ndmin in (64,):
        for (tag, exprs, operands, scalars) in [
            ("array", [], [describe(b1, b1)], []),
            ("slice", [("0:3", slice(0, 3))], [], []),
            ("scalar", [], [], [["i", 5]]),
        ]:
            directive = f"0,{ndmin}"
            try:
                r = build("r", directive, exprs, [b1] if operands else [], scalars)
            except ValueError:
                emit_throw("r_", params_of("r", directive, exprs, scalars), operands,
                           f"ndmin_cap/{ndmin}/{tag}")
                continue
            except Exception:
                continue
            emit("r_", params_of("r", directive, exprs, scalars), operands, r,
                 f"ndmin_cap/{ndmin}/{tag}")

    # An all-literal key whose integer does not fit int64 lifts the default to uint64
    # (result_type(2**63) and result_type(2**64-1) are both uint64). The "u" scalar kind
    # boxes it as a C# ulong, which is the only C# type that can carry the value.
    for value in (2 ** 63, 2 ** 64 - 1, 2 ** 63 - 1):
        kind = "u" if value >= 2 ** 63 else "i"
        for concat in ("r", "c"):
            try:
                r = build(concat, None, [], [], [[kind, value]])
            except Exception:
                continue
            emit(f"{concat}_", params_of(concat, None, [], [[kind, value]]), [], r,
                 f"weak_uint64/{value}")

    return cases


def gen_pad(dtypes):
    cases = []
    n = 0
    skipped = 0
    modes = ["constant", "edge", "reflect", "wrap"]
    for sh in [(5,), (3, 4)]:
        for s in dtypes:
            base = _cbase(sh, np.dtype(s))
            for mode in modes:
                try:
                    r = np.asarray(np.pad(base, 1, mode=mode))
                except Exception:
                    skipped += 1
                    continue
                cases.append({
                    "id": f"pad/{mode}/{'x'.join(map(str, sh))}/{s}/{n}",
                    "op": "pad",
                    "params": {"pad_width": 1, "mode": mode},
                    "operands": [describe(base, base)],
                    "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                 "buffer": np.ascontiguousarray(r).tobytes().hex()},
                    "layout": "pad",
                    "valueclass": "mixed",
                })
                n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# T15 — multi-output. np.modf(x) -> (fractional, integral). Split into two corpus ops so the
# harness bit-compares EACH output buffer. NumPy is the oracle for value, dtype, and the C-standard
# sign rules (modf(-0.0)=(-0.0,-0.0), modf(inf)=(0.0,inf), modf(nan)=(nan,nan)).
# Every non-complex NumPy lane: the 59f99320 per-width promotion tier (bool/int8/uint8->f16,
# int16/uint16->f32, int32+->f64) made the integer/bool cells computable - complex raises (its
# no-loop TypeError is unit-gated) and Char rides char_tier("modf").
MODF_DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64",
               "uint64", "float16", "float32", "float64"]
MODF_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "c_contiguous_3d", "f_contiguous_2d",
                "transposed_3d", "strided_2d_cols", "negstride_1d", "one_element_1d"]


def gen_modf(dtypes, layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            try:
                frac, integ = np.modf(view)
            except Exception:
                skipped += 1
                continue
            for part_name, part in (("modf_frac", frac), ("modf_int", integ)):
                r = np.asarray(part)
                cases.append({
                    "id": f"{part_name}/{ln}/{s}/{n}",
                    "op": part_name,
                    "params": {},
                    "operands": [operand],
                    "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                 "buffer": np.ascontiguousarray(r).tobytes().hex()},
                    "layout": ln,
                    "valueclass": "mixed",
                })
                n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# T14 — sorting / searching. Distinct values avoid tie-break ambiguity (quicksort is unstable),
# so argsort is deterministic both sides. NumPy is the oracle for the int64 index results.
SORT_DTYPES = list(ALL_DTYPES)         # widened: argsort/searchsorted/nonzero (complex sorts lexicographically)


def _distinct(n, dt):
    """A deterministic permutation of 0..n-1 (distinct -> no ties), cast to dt. gcd(7,n)==1 for our n."""
    return np.array([(i * 7 + 3) % n for i in range(n)], dtype=np.dtype(dt))


def gen_argsort(dtypes):
    cases = []
    n = 0
    for dt in dtypes:
        a1 = _distinct(8, dt)
        a2 = _distinct(12, dt).reshape(3, 4)
        jobs = [(a1, -1)]
        for axis in (0, 1, -1):
            jobs.append((a2, axis))
        for (a, axis) in jobs:
            r = np.asarray(np.argsort(a, axis=axis))
            cases.append({
                "id": f"argsort/{a.ndim}d/{dt}/axis={axis}/{n}",
                "op": "argsort",
                "params": {"axis": axis},
                "operands": [describe(a, a)],
                "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                             "buffer": np.ascontiguousarray(r).tobytes().hex()},
                "layout": f"{a.ndim}d",
                "valueclass": "distinct",
            })
            n += 1
    return cases


def gen_searchsorted(dtypes):
    """np.searchsorted differential corpus (NumPy 2.4.2 is the oracle). Families:

      base    — sorted DISTINCT a + distinct v, same dtype, left/right (kernel baseline; UNCHANGED).
      dup     — a WITH duplicates + keys below/at/in-gap/above: first-vs-last occurrence (left/right
                MUST differ) AND the out-of-range clamp to 0 / len(a).
      mixed   — a.dtype != v.dtype: NumPy searches in result_type(a, v) and NEVER casts the key DOWN
                to a's dtype (the cc676ea8 promotion path — fractional/out-of-range/negative keys that
                would wrap or truncate under a naive down-cast).
      sorter  — UNSORTED a + argsort(a) as `sorter` (the separate argbinsearch IL-kernel path).
      nan     — NaN keys, and a NaN in a's tail: NaN sorts LAST and must not corrupt the carried bound.
      cplx    — complex with equal real / differing imag / NaN component: NumPy's CDOUBLE_LT is
                NaN-aware LEXICOGRAPHIC (real then imag), the same order np.sort uses.
      strided — non-contiguous (::2), offset slice, and negative-stride a (the contiguousA=false /
                arrStride!=elemSize kernel path; NumPy is the oracle even for a reversed view).
      empty   — empty a (all zeros), scalar v (0-d result), empty v (empty result).

    The cross-cutting families (mixed/sorter/nan/cplx/strided/empty) are gated on len(dtypes) > 1
    so char_tier's single-dtype uint16->char weave only picks up the same-dtype base/dup families.
    """
    cases = []
    n = 0

    # Family: base (UNCHANGED — keep the original 26 cases byte-identical).
    for dt in dtypes:
        a = np.sort(_distinct(8, dt))
        v = _distinct(6, dt)
        for side in ("left", "right"):
            r = np.asarray(np.searchsorted(a, v, side=side))
            cases.append({
                "id": f"searchsorted/{side}/{dt}/{n}",
                "op": "searchsorted",
                "params": {"side": side},
                "operands": [describe(a, a), describe(v, v)],
                "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                             "buffer": np.ascontiguousarray(r).tobytes().hex()},
                "layout": "searchsorted",
                "valueclass": "distinct",
            })
            n += 1

    def emit(a_pair, v_pair, side, tag, vclass, sorter_pair=None):
        """Serialize one (a, v[, sorter]) searchsorted case; NumPy computes the oracle result."""
        nonlocal n
        a_base, a_view = a_pair
        v_base, v_view = v_pair
        ops = [describe(a_base, a_view), describe(v_base, v_view)]
        s_view = None
        if sorter_pair is not None:
            s_base, s_view = sorter_pair
            ops.append(describe(s_base, s_view))
        r = np.asarray(np.searchsorted(a_view, v_view, side=side, sorter=s_view))
        cases.append({
            "id": f"searchsorted/{tag}/{side}/{n}",
            "op": "searchsorted",
            "params": {"side": side},
            "operands": ops,
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": f"searchsorted_{tag}",
            "valueclass": vclass,
        })
        n += 1

    # Family: duplicates (first-vs-last) + out-of-range clamp. Runs per dtype (incl. the char weave).
    # min(a)=1 so key 0 probes below-min for unsigned too; key 9 probes above-max.
    for dt in dtypes:
        a = np.sort(np.array([1, 2, 2, 4, 4, 4, 6, 8], dtype=dt))
        v = np.array([0, 1, 2, 3, 4, 5, 8, 9], dtype=dt)
        for side in ("left", "right"):
            emit((a, a), (v, v), side, f"dup/{dt}", "dup")

    # Cross-cutting families — skipped under the single-dtype char-weave call.
    if len(dtypes) > 1:
        # mixed dtype: (a_vals, a_dtype, v_vals, v_dtype, tag). The keys are chosen to expose a
        # naive down-cast to a's dtype (fractional, out-of-a's-range, negative-into-unsigned).
        mixed = [
            ([1, 2, 3, 4, 5, 6, 7, 8], "int32",      [0.5, 2.5, 4.0, 8.5, 9.0], "float64", "i32_f64"),
            ([0, 1, 2, 3, 4, 5, 6, 7], "int8",       [-200, 3, 100, 127, 300],  "int64",   "i8_i64_oob"),
            ([10, 20, 30, 40, 50],     "int8",       [5.0, 15.5, 45.0, 100.0],  "float64", "i8_f64"),
            ([1, 2, 3, 4, 5],          "uint8",      [-5, 0, 3, 300],           "int16",   "u8_i16_neg"),
            ([1, 2, 3, 4, 5, 6, 7, 8], "float32",    [0.5, 2.25, 8.75, 9.0],    "float64", "f32_f64"),
            ([1, 2, 3, 4, 5],          "int64",      [0, 3, 5, 6],              "uint64",  "i64_u64"),
            ([0, 0, 1, 1],             "bool",       [-1, 0, 1, 2],             "int32",   "bool_i32"),
            ([1, 2, 3, 4, 5],          "complex128", [0.5, 2.0, 5.5],           "float64", "c128_f64"),
            ([1, 2, 3, 4, 5, 6, 7, 8], "float64",    [-1, 3, 8, 9],             "int32",   "f64_i32"),
            ([1, 2, 3, 4, 5],          "uint16",     [-1, 3, 70000],            "int32",   "u16_i32_oob"),
        ]
        for (av, adt, vv, vdt, tag) in mixed:
            a = np.sort(np.array(av, dtype=adt))
            v = np.array(vv, dtype=vdt)
            for side in ("left", "right"):
                emit((a, a), (v, v), side, f"mixed/{tag}", "mixed")

        # sorter: UNSORTED a (with duplicates) + argsort(a). Exercises the argbinsearch kernel.
        for dt in ("int32", "int64", "uint8", "int16", "float32", "float64", "float16", "complex128"):
            a = np.array([5, 1, 3, 1, 5, 2, 3, 4], dtype=dt)
            sorter = np.argsort(a, kind="stable").astype(np.int64)   # NumPy sorter dtype is intp (int64)
            v = np.array([1, 3, 5, 0, 6], dtype=dt)
            for side in ("left", "right"):
                emit((a, a), (v, v), side, f"sorter/{dt}", "sorter", sorter_pair=(sorter, sorter))

        # NaN keys (carry reset), and a NaN sitting in a's sorted tail.
        for dt in ("float16", "float32", "float64"):
            a = np.sort(np.array([-2, 0, 1.5, 3, 10], dtype=dt))
            v = np.array([np.nan, 1.0, np.nan, 100.0, -5.0, 3.0], dtype=dt)
            for side in ("left", "right"):
                emit((a, a), (v, v), side, f"nan/{dt}", "nan")
        for dt in ("float32", "float64"):
            a = np.array([1.0, 2.0, 3.0, np.nan], dtype=dt)          # sorted: NaN last
            v = np.array([2.5, np.nan, 0.0, 5.0], dtype=dt)
            for side in ("left", "right"):
                emit((a, a), (v, v), side, f"nan_in_a/{dt}", "nan")

        # complex lexicographic (CDOUBLE_LT): equal real / differing imag / NaN component.
        ca1 = np.sort(np.array([1 + 0j, 2 + 1j, 2 + 3j, 2 + 5j, 3 + 0j, 3 + 2j], dtype=np.complex128))
        cv1 = np.array([2 + 2j, 2 + 4j, 2 + 3j, 3 + 1j, 0.5 + 0j,
                        complex(2, np.nan), complex(np.nan, 0)], dtype=np.complex128)
        ca2 = np.sort(np.array([0 + 0j, 0 + 1j, 0 + 2j, 1 + 0j, 1 + 1j], dtype=np.complex128))
        cv2 = np.array([0 + 1.5j, 0 - 1j, 1 + 0.5j, complex(0, np.nan)], dtype=np.complex128)
        for (a, v, tag) in ((ca1, cv1, "c1"), (ca2, cv2, "c2")):
            for side in ("left", "right"):
                emit((a, a), (v, v), side, f"cplx/{tag}", "cplx")

        # strided a: non-contiguous (::2), offset slice, negative-stride (reversed => descending;
        # NumPy is still the oracle — it searches the same strided memory).
        for dt in ("int32", "float64", "uint8", "complex128"):
            base = np.arange(16).astype(dt)     # already sorted
            view = base[::2]                    # ascending, stride 2 (non-contiguous)
            v = np.array([1, 4, 7, 10, 15, 0], dtype=dt)
            for side in ("left", "right"):
                emit((base, view), (v, v), side, f"strided/{dt}", "strided")
        for dt in ("int32", "float64"):
            base = np.arange(20).astype(dt)
            view = base[3:15]                   # offset 3, contiguous
            v = np.array([0, 5, 14, 19, 10], dtype=dt)
            for side in ("left", "right"):
                emit((base, view), (v, v), side, f"offset/{dt}", "strided")
        base = np.arange(8, dtype=np.int32)
        view = base[::-1]                       # [7..0], stride -1 (descending; oracle = NumPy)
        v = np.array([3, 0, 7, 5], dtype=np.int32)
        for side in ("left", "right"):
            emit((base, view), (v, v), side, "negstride/int32", "strided")

        # empty a (all zeros), scalar v (0-d result), empty v (empty result).
        for dt in ("int32", "float64"):
            a = np.array([], dtype=dt)
            v = np.array([1, 2, 3], dtype=dt)
            for side in ("left", "right"):
                emit((a, a), (v, v), side, f"empty_a/{dt}", "empty")
        for dt in ("int32", "float64"):
            a = np.sort(_distinct(8, dt))
            v = np.array(3, dtype=dt)           # 0-d scalar key -> 0-d result
            for side in ("left", "right"):
                emit((a, a), (v, v), side, f"scalar_v/{dt}", "scalar")
        for dt in ("int32", "float64"):
            a = np.sort(_distinct(8, dt))
            v = np.array([], dtype=dt)          # empty v -> empty result
            emit((a, a), (v, v), "left", f"empty_v/{dt}", "empty")

    return cases


# digitize applies to numeric non-complex dtypes (complex x raises TypeError).
DIGITIZE_DTYPES = ["float64", "float32", "int64", "int32", "int16", "uint8"]


def gen_digitize(dtypes):
    """np.digitize: searchsorted + monotonicity. Same-dtype inc/dec bins x right, a mixed-dtype
    promotion lock (float x into int bins), and NaN-in-x cases (which pin searchsorted's
    NaN-as-largest total order — the carry/bisect fix)."""
    cases = []
    n = 0

    def emit(x, bins, right, tag, vclass):
        nonlocal n
        r = np.asarray(np.digitize(x, bins, right=right))
        cases.append({
            "id": f"digitize/{tag}/right={right}/{n}",
            "op": "digitize",
            "params": {"right": bool(right)},
            "operands": [describe(x, x), describe(bins, bins)],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": "digitize",
            "valueclass": vclass,
        })
        n += 1

    for dt in dtypes:
        inc = np.sort(_distinct(6, dt))               # increasing, distinct edges
        dec = inc[::-1].copy()                        # decreasing edges
        x = _distinct(8, dt)                          # spans below/in/above the bin range
        for right in (False, True):
            emit(x, inc, right, f"inc/{dt}", "distinct")
            emit(x, dec, right, f"dec/{dt}", "distinct")

    # Mixed dtype: float64 x into int32 bins — locks the result_type(bins, x) promotion
    # (a naive cast of x into the bins dtype would truncate).
    xf = np.array([1.2, 10.0, 12.4, 15.5, 20.0, -1.0], dtype=np.float64)
    bi = np.array([0, 5, 10, 15, 20], dtype=np.int32)
    for right in (False, True):
        emit(xf, bi, right, "mixed", "mixed")

    # NaN in x — a NaN sorts to the end (len(bins)) and must not corrupt a following key.
    for dt in ("float64", "float32"):
        binsf = np.array([0, 1, 2.5, 4, 10], dtype=np.dtype(dt))
        xn = np.array([np.nan, 1.0, np.nan, 6.0, -1.0, 100.0], dtype=np.dtype(dt))
        for right in (False, True):
            emit(xn, binsf, right, f"nan/{dt}", "nan")

    return cases


def gen_nonzero(dtypes):
    cases = []
    n = 0
    for dt in dtypes:
        a = np.array([0, 1, 0, 2, 3, 0, 4, 0, 5, 0], dtype=np.dtype(dt))
        r = np.nonzero(a)[0].astype(np.int64)
        cases.append({
            "id": f"nonzero/1d/{dt}/{n}",
            "op": "nonzero",
            "params": {},
            "operands": [describe(a, a)],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": "nonzero",
            "valueclass": "mixed",
        })
        n += 1
    return cases


# bincount input must be NON-NEGATIVE integers (cast to int64); complex/float raise, so exclude them.
# Char has no NumPy dtype (it rides the uint16 proxy at runtime; uint16 here covers its value semantics).
BINCOUNT_DTYPES = ["bool", "uint8", "int8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]


def gen_bincount(dtypes):
    """np.bincount: count occurrences of each non-negative int (int64 result) + optional float64
    weighted sum + minlength. Covers integer/bool input dtypes, contiguous / reversed / strided
    input views (NumSharp copies to contiguous int64 — order-independent counting), the minlength
    pad, and the weighted path (a sequential accumulation that must be BIT-EXACT with NumPy).
    Small arrays are enough for value parity: the privatized count path is bit-identical to the
    plain scatter (integer add is associative), proven separately in the unit tests."""
    cases = []
    n = 0

    def emit(params, operands, r, tag):
        nonlocal n
        cases.append({
            "id": f"bincount/{tag}/{n}",
            "op": "bincount",
            "params": params,
            "operands": operands,
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": "bincount",
            "valueclass": "nonneg",
        })
        n += 1

    for dt in dtypes:
        # A pool WITH repeats (so counts exceed 1). bool only holds 0/1.
        if dt == "bool":
            a = np.array([True, False, True, True, False, True, True, False], dtype=bool)
        else:
            a = np.array([0, 1, 1, 2, 2, 2, 3, 0, 1, 2, 3, 3, 3, 0], dtype=np.dtype(dt))
        emit({"minlength": 0}, [describe(a, a)], np.asarray(np.bincount(a)), f"count/{dt}")
        emit({"minlength": 10}, [describe(a, a)], np.asarray(np.bincount(a, minlength=10)), f"minlen/{dt}")
        # reversed + strided input views -> exercise the copy-to-contiguous path.
        rev = a[::-1]
        emit({"minlength": 0}, [describe(a, rev)], np.asarray(np.bincount(rev)), f"rev/{dt}")
        strd = a[::2]
        emit({"minlength": 0}, [describe(a, strd)], np.asarray(np.bincount(strd)), f"strided/{dt}")

    # Weighted (float64 output). x fixed int64; weights span float64/float32/int32 (all cast to f64).
    xw = np.array([0, 1, 1, 2, 0, 2, 3], dtype=np.int64)
    for wdt in ("float64", "float32", "int32"):
        if wdt == "int32":
            w = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
        else:
            w = np.array([0.5, 1.5, -2.0, 3.25, 4.0, 0.0, 7.5], dtype=np.dtype(wdt))
        emit({"minlength": 0}, [describe(xw, xw), describe(w, w)],
             np.asarray(np.bincount(xw, weights=w)), f"weighted/{wdt}")
        emit({"minlength": 9}, [describe(xw, xw), describe(w, w)],
             np.asarray(np.bincount(xw, weights=w, minlength=9)), f"weighted_ml/{wdt}")

    return cases


# W13 — SIMD-tail boundary sizes. 1-D arrays straddling the V128/V256/V512 lane counts so the
# unrolled-SIMD body, 1-vector remainder, and scalar tail are all exercised at their seams.
TAIL_SIZES = [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
# widened: SIMD-seam sizes across every dtype EXCEPT bool (gen_tail subtracts; NumPy bans bool `-`).
TAIL_DTYPES = [d for d in ALL_DTYPES if d != "bool"]


def gen_tail(dtypes):
    cases = []
    n = 0
    skipped = 0
    BIN = [("add", np.add), ("subtract", np.subtract), ("multiply", np.multiply)]
    UN = [("negative", np.negative), ("abs", np.abs), ("sqrt", np.sqrt)]
    RED = [("sum", np.sum), ("prod", np.prod), ("max", np.max), ("min", np.min)]
    for sz in TAIL_SIZES:
        for s in dtypes:
            dt = np.dtype(s)
            a = _fill(sz, dt)
            b = np.ascontiguousarray(np.roll(a, 1))
            for opname, f in BIN:
                r = np.asarray(f(a, b))
                cases.append({"id": f"{opname}/tail{sz}/{s}/{n}", "op": opname, "params": {},
                              "operands": [describe(a, a), describe(b, b)],
                              "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                           "buffer": np.ascontiguousarray(r).tobytes().hex()},
                              "layout": f"tail{sz}", "valueclass": "tail"})
                n += 1
            for opname, f in UN:
                try:
                    r = np.asarray(f(a))
                except Exception:
                    skipped += 1
                    continue
                cases.append({"id": f"{opname}/tail{sz}/{s}/{n}", "op": opname, "params": {},
                              "operands": [describe(a, a)],
                              "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                           "buffer": np.ascontiguousarray(r).tobytes().hex()},
                              "layout": f"tail{sz}", "valueclass": "tail"})
                n += 1
            for opname, f in RED:
                r = np.asarray(f(a))
                cases.append({"id": f"{opname}/tail{sz}/{s}/{n}", "op": opname,
                              "params": {"axis": None, "keepdims": False},
                              "operands": [describe(a, a)],
                              "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                           "buffer": np.ascontiguousarray(r).tobytes().hex()},
                              "layout": f"tail{sz}", "valueclass": "tail"})
                n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# W12 — parameter sweep. The reduce tier only covered axis in {None, 0, last}; here we exercise
# the MIDDLE axis and NEGATIVE axes (-1/-2/-3), ddof=1 (sample std/var), and order='F' ravel.
PARAM_DTYPES = list(ALL_DTYPES)        # widened: axis/ddof/keepdims params across every dtype


def gen_params(dtypes):
    cases = []
    n = 0
    skipped = 0
    reduce_names = ["sum", "prod", "max", "min", "mean", "std", "var", "argmax", "argmin", "all", "any"]
    for s in dtypes:
        base, view = LAYOUTS["c_contiguous_3d"](np.dtype(s))      # (2,3,4)
        operand = describe(base, view)
        for opname in reduce_names:
            for axis in [1, -1, -2, -3]:                          # middle + every negative axis
                for kd in (False, True):
                    try:
                        r = np.asarray(REDUCE_OPS[opname](view, axis, kd))
                    except Exception:
                        skipped += 1
                        continue
                    cases.append({"id": f"{opname}/negaxis/{s}/axis={axis}/kd={int(kd)}/{n}",
                                  "op": opname, "params": {"axis": axis, "keepdims": kd},
                                  "operands": [operand],
                                  "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                               "buffer": np.ascontiguousarray(r).tobytes().hex()},
                                  "layout": "negaxis", "valueclass": "param"})
                    n += 1
    # ddof=1 (sample) std/var on a 2-D array, axis None/0/1.
    for s in ["float32", "float64"]:
        base, view = LAYOUTS["c_contiguous_2d"](np.dtype(s))
        operand = describe(base, view)
        for opname, npf in (("std_ddof", np.std), ("var_ddof", np.var)):
            for axis in [None, 0, 1]:
                r = np.asarray(npf(view, axis=axis, ddof=1))
                cases.append({"id": f"{opname}/ddof1/{s}/axis={axis}/{n}",
                              "op": opname, "params": {"axis": axis, "ddof": 1},
                              "operands": [operand],
                              "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                           "buffer": np.ascontiguousarray(r).tobytes().hex()},
                              "layout": "ddof1", "valueclass": "param"})
                n += 1
    # Multi-axis (tuple-axis) reductions — plan §C1. Exactly the reductions NumSharp exposes an
    # int[]-axis overload for (median / average / nanmedian; sum/prod/min/max/mean have NO tuple-axis
    # overload yet — a tracked feature gap, deliberately NOT generated so the gate stays honest).
    # The params key is "axes" (int[]), the registry's array-form convention, so OpRegistry binds the
    # int[] overload rather than mis-reading a scalar "axis".
    multiaxis_ops = {
        "median": lambda a, ax, kd: np.median(a, axis=ax, keepdims=kd),
        "average": lambda a, ax, kd: np.average(a, axis=ax, keepdims=kd),
        "nanmedian": lambda a, ax, kd: np.nanmedian(a, axis=ax, keepdims=kd),
    }
    for s in ["int32", "uint8", "float32", "float64"]:
        for ln in ["c_contiguous_3d", "f_contiguous_3d", "transposed_3d"]:
            base, view = LAYOUTS[ln](np.dtype(s))
            operand = describe(base, view)
            for opname, f in multiaxis_ops.items():
                for axes in ([0, 1], [-2, -1], [0, 2], [0, 1, 2]):
                    for kd in (False, True):
                        try:
                            r = np.asarray(f(view, tuple(axes), kd))
                        except Exception:
                            skipped += 1
                            continue
                        cases.append({"id": f"{opname}/multiaxis/{ln}/{s}/axes={axes}/kd={int(kd)}/{n}",
                                      "op": opname, "params": {"axes": axes, "keepdims": kd},
                                      "operands": [operand],
                                      "expected": _arr_expected(r),
                                      "layout": ln, "valueclass": "param"})
                        n += 1
    # order='F' ravel across C-contig, transposed, and F-contig sources.
    for s in dtypes:
        for ln in ["c_contiguous_2d", "transposed_2d", "f_contiguous_2d", "c_contiguous_3d"]:
            base, view = LAYOUTS[ln](np.dtype(s))
            r = np.asarray(np.ravel(view, order="F"))
            cases.append({"id": f"ravel_f/{ln}/{s}/{n}", "op": "ravel_f", "params": {},
                          "operands": [describe(base, view)],
                          "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                       "buffer": np.ascontiguousarray(r).tobytes().hex()},
                          "layout": ln, "valueclass": "param"})
            n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# W11 — operand-relationship flags (section C): input aliasing (a op a, SAME buffer both sides)
# and in-place out= (the output buffer IS an input). Exercises read-before-write within the kernel.
# widened: out=/overlap aliasing across every dtype EXCEPT bool (gen_aliasing subtracts; NumPy bans
# bool `-`). complex128 is now INCLUDED: the old exclusion feared the a*a self-multiply's
# catastrophic cancellation (a^2-b^2) diverging from NumPy's ARRAY ufunc, but NDComplexMath.Multiply
# now ports NumPy's fused simd_cmul (vfmaddsub) byte-for-byte — matching the ARRAY multiply exactly,
# including that cancellation regime (MisalignedRegistry branch (321)) — so multiply(a,a) is bit-exact
# and add/subtract/maximum/minimum/clip are pure IEEE / lexicographic. The whole tier stays strict.
ALIAS_DTYPES = [d for d in ALL_DTYPES if d != "bool"]


def gen_aliasing(dtypes):
    cases = []
    n = 0
    skipped = 0
    bin_ops = [("add", np.add), ("subtract", np.subtract), ("multiply", np.multiply),
               ("maximum", np.maximum), ("minimum", np.minimum)]
    for s in dtypes:
        dt = np.dtype(s)
        a = _cbase((4, 5), dt)
        # (1) input aliasing: a op a — one stored operand, harness passes it as both args.
        for opname, f in bin_ops:
            r = np.asarray(f(a, a))
            cases.append({"id": f"{opname}/alias/{s}/{n}", "op": opname, "params": {}, "alias": True,
                          "operands": [describe(a, a)],
                          "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                       "buffer": np.ascontiguousarray(r).tobytes().hex()},
                          "layout": "alias", "valueclass": "alias"})
            n += 1
        # (2) in-place out=: maximum(a,b,out=a), minimum(a,b,out=a), clip(a,lo,hi,out=a).
        b = np.ascontiguousarray(np.roll(a, 1))
        for opname, f in (("maximum_out", np.maximum), ("minimum_out", np.minimum)):
            acc = a.copy()
            f(acc, b, out=acc)
            cases.append({"id": f"{opname}/{s}/{n}", "op": opname, "params": {},
                          "operands": [describe(a, a), describe(b, b)],
                          "expected": {"dtype": acc.dtype.name, "shape": [int(d) for d in acc.shape],
                                       "buffer": np.ascontiguousarray(acc).tobytes().hex()},
                          "layout": "out", "valueclass": "alias"})
            n += 1
        lo_v, hi_v = (1, 100) if dt.kind == "u" else (-10, 10)
        lo = np.array(lo_v, dtype=dt).reshape(())
        hi = np.array(hi_v, dtype=dt).reshape(())
        acc = a.copy()
        np.clip(acc, lo, hi, out=acc)
        cases.append({"id": f"clip_out/{s}/{n}", "op": "clip_out", "params": {},
                      "operands": [describe(a, a), describe(lo, lo), describe(hi, hi)],
                      "expected": {"dtype": acc.dtype.name, "shape": [int(d) for d in acc.shape],
                                   "buffer": np.ascontiguousarray(acc).tobytes().hex()},
                      "layout": "out", "valueclass": "alias"})
        n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# W14 — error parity. The other generators SKIP every case where NumPy raises, so "NumPy raises =>
# NumSharp raises the same" was never asserted. These cases carry expects_throw=True (no expected
# buffer); the harness asserts the op throws SOMETHING rather than silently producing a result.
def _numpy_raises(opname, arrs, params):
    try:
        if opname == "power":
            _ = arrs[0] ** arrs[1]
        elif opname == "add":
            _ = arrs[0] + arrs[1]
        elif opname == "matmul":
            _ = np.matmul(arrs[0], arrs[1])
        elif opname == "bitwise_and":
            _ = arrs[0] & arrs[1]
        elif opname == "left_shift":
            _ = np.left_shift(arrs[0], arrs[1])
        elif opname == "concatenate":
            _ = np.concatenate(list(arrs), axis=params["axis"])
        elif opname == "reshape":
            _ = arrs[0].reshape(params["shape"])
        elif opname == "sum":
            _ = np.sum(arrs[0], axis=params["axis"])
        elif opname == "invert":
            _ = np.invert(arrs[0])
        elif opname == "stack":
            _ = np.stack(list(arrs), axis=params["axis"])
        elif opname == "subtract":
            _ = arrs[0] - arrs[1]
        elif opname == "min":
            _ = np.min(arrs[0], axis=params.get("axis"))
        elif opname == "max":
            _ = np.max(arrs[0], axis=params.get("axis"))
        elif opname == "argmax":
            _ = np.argmax(arrs[0], axis=params.get("axis"))
        elif opname == "floor":
            _ = np.floor(arrs[0])
        elif opname == "searchsorted":
            _ = np.searchsorted(arrs[0], arrs[1], side=params.get("side", "left"))
        return False
    except Exception:
        return True


def gen_errors():
    cases = []
    n = 0
    i32 = np.dtype("int32")
    f64 = np.dtype("float64")
    b = np.dtype("bool")
    specs = [
        ("power", [np.array([2, 3, 4], dtype=i32), np.array([-1, -2, -1], dtype=i32)], {}),       # int ** neg
        ("add", [_cbase((3,), i32), _cbase((4,), i32)], {}),                                        # broadcast mismatch
        ("subtract", [_cbase((2,), b), _cbase((2,), b)], {}),                                       # bool subtract
        ("matmul", [_cbase((2, 3), f64), _cbase((2, 2), f64)], {}),                                 # core-dim mismatch
        ("bitwise_and", [_cbase((4,), f64), _cbase((4,), f64)], {}),                                # bitwise on float
        ("left_shift", [_cbase((4,), f64), _cbase((4,), f64)], {}),                                 # shift on float
        ("concatenate", [_cbase((2, 3), i32), _cbase((2, 4), i32)], {"axis": 0}),                   # dim mismatch
        ("reshape", [_cbase((6,), i32)], {"shape": [4]}),                                           # incompatible size
        ("sum", [_cbase((3,), i32)], {"axis": 5, "keepdims": False}),                               # axis out of range
        ("stack", [_cbase((2, 3), i32), _cbase((2, 4), i32)], {"axis": 0}),                          # mismatched shapes
        # G13 (F17) additions — each probed to raise in NumPy 2.4.2 AND throw cleanly in NumSharp.
        # less(complex) was in the plan but NumPy 2.4.2 does NOT raise (comparisons on complex
        # return bool, lexicographic) — dropped.
        ("min", [np.array([], dtype=f64)], {"axis": None, "keepdims": False}),                      # zero-size reduce
        ("max", [np.array([], dtype=f64)], {"axis": None, "keepdims": False}),                      # zero-size reduce
        ("argmax", [np.array([], dtype=f64)], {"axis": 0, "keepdims": False}),                      # argmax of empty
        ("floor", [np.array([1.5 + 2.0j, -0.5 + 1.0j])], {}),                                       # floor(complex)
        ("searchsorted", [_cbase((2, 3), i32), np.array([1, 2], dtype=i32)], {"side": "left"}),     # 2-D a
        # invert(float): NumPy raises TypeError. Historically this was an ILLEGAL-INSTRUCTION
        # host crash in NumSharp; the B8 loop-resolution guard (Default.Invert.cs) now throws
        # NumPy's verbatim TypeError, so the spec is safe to gate (see COMPLETENESS_PLAN L1).
        ("invert", [_cbase((4,), f64)], {}),
    ]
    for (opname, arrs, params) in specs:
        if not _numpy_raises(opname, arrs, params):
            print(f"  WARN: NumPy did NOT raise for {opname}; skipping")
            continue
        cases.append({
            "id": f"{opname}/error/{n}",
            "op": opname,
            "params": params,
            "operands": [describe(x, x) for x in arrs],
            "expected": {"dtype": "bool", "shape": [], "buffer": ""},
            "expects_throw": True,
            "layout": "error",
            "valueclass": "error",
        })
        n += 1
    return cases


# W15 — copyto: (1) same-dtype OVERLAPPING copies (dst & src are different views of the SAME
# buffer) which NumPy makes safe via COPY_IF_OVERLAP, and (2) cross-dtype copyto INTO a strided
# destination view + scalar-broadcast source (the cast-into-non-contiguous-dst path astype never
# exercises, plus the scalar-broadcast cross-dtype fast fill).
COPYTO_OVERLAP_DTYPES = list(ALL_DTYPES)   # widened: same-dtype copyto overlap across every dtype
COPYTO_CROSS = [
    ("float64", "int32"), ("float32", "uint8"), ("float64", "int16"), ("int32", "float64"),
    ("int64", "int16"), ("float64", "float16"), ("int32", "uint8"), ("float64", "int64"),
    ("uint8", "float32"), ("int16", "int64"), ("float64", "uint32"), ("complex128", "float64"),
]


def _viewspec(base, view):
    """(shape, element-strides, element-offset) of a view into base — buffer stripped (it lives once)."""
    d = describe(base, view)
    return {"shape": d["shape"], "strides": d["strides"], "offset": d["offset"]}


def _copyto_cast_case(id_, dbase, dview, sbase, sview, exp, casting):
    return {
        "id": id_, "op": "copyto", "params": {"casting": casting},
        "operands": [describe(dbase, dview), describe(sbase, sview)],
        "expected": {"dtype": exp.dtype.name, "shape": [int(d) for d in exp.shape],
                     "buffer": np.ascontiguousarray(exp).tobytes().hex()},
        "layout": "copyto_cast", "valueclass": "cast",
    }


def gen_copyto(overlap_dtypes, cross_pairs):
    cases = []
    n = 0

    # (1) Same-dtype OVERLAPPING copyto — ONE buffer, two views. The harness rebuilds the base
    # buffer once (operand 0) and re-derives dst/src views from params, so they genuinely alias.
    specs_1d = [
        ("shift_fwd", 8, lambda a: (a[1:], a[:-1])),     # same-direction run -> memmove-safe
        ("shift_bwd", 8, lambda a: (a[:-1], a[1:])),
        ("reverse",   8, lambda a: (a[:], a[::-1])),     # opposite-direction -> needs temp
        ("step_wbr",  8, lambda a: (a[2:8:2], a[0:6:2])),# strided write-before-read overlap
    ]
    specs_2d = [
        ("rev2d",     (4, 4), lambda a: (a[:], a[::-1, ::-1])),
        ("transpose", (4, 4), lambda a: (a[:], a.T)),    # square -> in-place transpose overlap
    ]
    for s in overlap_dtypes:
        dt = np.dtype(s)
        for (tag, shp, fn) in [(t, (ln,), f) for (t, ln, f) in specs_1d] + list(specs_2d):
            base = _cbase(shp, dt)
            work = base.copy()
            dv, sv = fn(work)
            np.copyto(dv, sv)
            exp = np.ascontiguousarray(dv)
            bdv, bsv = fn(base)  # identical slicing on the pristine base -> same strides/offset
            cases.append({
                "id": f"copyto_overlap/{tag}/{s}/{n}", "op": "copyto_overlap",
                "params": {"dst": _viewspec(base, bdv), "src": _viewspec(base, bsv)},
                "operands": [describe(base, base)],
                "expected": {"dtype": exp.dtype.name, "shape": [int(d) for d in exp.shape],
                             "buffer": exp.tobytes().hex()},
                "layout": f"overlap_{tag}", "valueclass": "overlap"})
            n += 1

    # (2) Cross-dtype copyto (casting='unsafe') into contiguous / strided dst, + scalar-broadcast src.
    for (ss, ds) in cross_pairs:
        sdt, ddt = np.dtype(ss), np.dtype(ds)
        # 2a contiguous src -> contiguous dst
        src = _cbase((8,), sdt); dst = _cbase((8,), ddt)
        w = dst.copy(); np.copyto(w, src, casting="unsafe")
        cases.append(_copyto_cast_case(f"copyto/cast_contig/{ss}->{ds}/{n}", dst, dst, src, src, w, "unsafe")); n += 1
        # 2b contiguous src -> STRIDED dst (every other element of a 16-buffer)
        dbase = _cbase((16,), ddt); dview = dbase[::2]; src2 = _cbase((8,), sdt)
        wb = dbase.copy(); wv = wb[::2]; np.copyto(wv, src2, casting="unsafe")
        cases.append(_copyto_cast_case(f"copyto/cast_strided_dst/{ss}->{ds}/{n}", dbase, dview, src2, src2, wv, "unsafe")); n += 1
        # 2c SCALAR-BROADCAST src -> whole-buffer dst (the cross-dtype fast-fill path)
        sval = _fill(1, sdt); sview = np.broadcast_to(sval, (8,)); dst3 = _cbase((8,), ddt)
        w3 = dst3.copy(); np.copyto(w3, sview, casting="unsafe")
        cases.append(_copyto_cast_case(f"copyto/cast_bcast_src/{ss}->{ds}/{n}", dst3, dst3, sval, sview, w3, "unsafe")); n += 1

    return cases


def _relabel_dtype(cases, frm, to):
    """Re-label a NumPy proxy dtype to a NumSharp-only dtype across a case's operand /
    expected / params descriptors (+ id). The RAW BYTES are untouched — this only rewrites
    the dtype STRING. Used for Char: NumSharp's Char is bit-identical to uint16 (2-byte
    unsigned), but NumPy has no char dtype, so every Char op is generated with uint16 as the
    proxy and then relabelled uint16->char. NumPy's uint16 result is therefore a bytes-exact
    oracle for Char, and the gate asserts NumSharp's Char ≡ uint16 across every op."""
    out = []
    for c in cases:
        c = json.loads(json.dumps(c))   # deep copy (cases are JSON-serializable)
        for o in c.get("operands", []):
            if o.get("dtype") == frm:
                o["dtype"] = to
        exp = c.get("expected")
        if isinstance(exp, dict):
            if exp.get("dtype") == frm:
                exp["dtype"] = to
            # kind=tuple carries per-slot descriptors — without this the instance tier's in-place
            # mutators (sort/fill/put: slots [post-call view, base buffer], both proxy-dtyped)
            # kept "uint16" slots after the relabel and every char cell failed on a dtype
            # mismatch (found the moment char_tier("instance") landed).
            for slot in exp.get("slots") or []:
                if isinstance(slot, dict) and slot.get("dtype") == frm:
                    slot["dtype"] = to
        for k, v in list(c.get("params", {}).items()):
            if v == frm:
                c["params"][k] = to
        c["id"] = c["id"].replace(frm, to)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Group A Batch 2 generators: sort / round_ / trace / diagonal / ediff1d / nan-quantile.
# ---------------------------------------------------------------------------
# np.round is dtype-PRESERVING except bool: np.round(bool, 0) -> float16 (the rint float-tier),
# now matched by NumSharp (Default.Round remaps bool -> Half at decimals==0). bool with decimals!=0
# RAISES in NumPy (the multiply/divide -> bool same_kind cast fails), so gen_round's try/except skips
# those cells. complex128 and float16 with decimals!=0 are NO LONGER carved: NumSharp ports
# PyArray_Round's op2(rint(op1(x, 10^|d|)), 10^|d|) at the input precision, so both are BIT-EXACT
# (formerly the complex dec!=0 no-op and the float16 fractional divergence were [OpenBugs]).
ROUND_DTYPES = ["bool", "int8", "uint8", "int16", "int32", "int64", "uint16", "uint32", "uint64",
                "float16", "float32", "float64", "complex128"]
# uint8 CARVED: trace of an unsigned dtype upcasts to Int64 in NumSharp but uint64 in NumPy -> [OpenBugs].
TRACE_DTYPES = ["int16", "int32", "int64", "float16", "float32", "float64", "complex128"]
EDIFF_DTYPES = ["int16", "int32", "int64", "uint8", "float32", "float64", "complex128"]  # no bool (NumPy bans bool `-`)
NANQ_DTYPES = ["float16", "float32", "float64",  # NaN-laced float pools (the tier's point)
               "bool", "uint8", "int32", "int64"]  # + integer/bool lanes: no NaN can occur, so
                                                   # NumPy degenerates to percentile - gated too

# Group A Batch 3: searching (flatnonzero/argwhere -> int64 coords) + whole-array bool reductions
# (allclose/array_equal, wrapped to a 0-D bool via np.asarray). All GREEN.
# iscomplex/isreal are now GREEN at FULL coverage (ported to NumPy's imag(x)==0 / imag!=0 structure —
# see the ISCOMPLEX_* block above; the two OpenBugs pins were removed). unique stays CARVED (-> [OpenBugs]):
# it mishandles offset/strided views + NaN-complex ordering. flatnonzero/argwhere stay.
NZ_OPS = {"flatnonzero": np.flatnonzero, "argwhere": np.argwhere}
NZ_DTYPES = ["bool", "int32", "uint8", "float64", "complex128"]
ALLCLOSE_OPS = {"allclose": lambda a, b: np.asarray(np.allclose(a, b)),
                "array_equal": lambda a, b: np.asarray(np.array_equal(a, b))}
ALLCLOSE_PAIRS = [("float64", "float64"), ("float32", "float32"), ("int32", "int32"),
                  ("complex128", "complex128"), ("float64", "float32"), ("int32", "int64")]


def gen_sort(dtypes):
    """Value sort (np.sort) over distinct 1-D + 2-D arrays, axis in {-1,0,1}. Same dtype out."""
    cases = []
    n = 0
    for dt in dtypes:
        a1 = _distinct(8, dt)
        a2 = _distinct(12, dt).reshape(3, 4)
        jobs = [(a1, -1)] + [(a2, ax) for ax in (0, 1, -1)]
        for (a, axis) in jobs:
            try:
                r = np.asarray(np.sort(a, axis=axis))
            except Exception:
                continue
            cases.append({
                "id": f"sort/{a.ndim}d/{dt}/axis={axis}/{n}",
                "op": "sort",
                "params": {"axis": axis},
                "operands": [describe(a, a)],
                "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                             "buffer": np.ascontiguousarray(r).tobytes().hex()},
                "layout": f"{a.ndim}d",
                "valueclass": "distinct",
            })
            n += 1
    return cases


# G12 (issue #623) — partition/argpartition via the DERIVED kth-values compare: the arrangement
# BETWEEN kth anchors is introselect-implementation-specific on both sides (NumPy's median-of-3 vs
# NumSharp's QuickSelect pick different pivots), so whole-output bytes are NOT contractual. What IS
# contractual — and deterministic for any input multiset, ties included — is the VALUE at every kth
# position (== sorted[kth]): the corpus pins take(partition(a, ks), ks) and the take_along_axis
# gather of argpartition at ks. The two-sided <=/>= invariant is unit-test-pinned
# (Sorting/np.partition.Test.cs).
def gen_partition_family(dtypes):
    cases = []
    n = 0

    def emit(op, params, a, r, tag, dt):
        nonlocal n
        cases.append({
            "id": f"{op}/{tag}/{dt}/{n}",
            "op": op,
            "params": params,
            "operands": [describe(a, a)],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": tag,
            "valueclass": "distinct",
        })
        n += 1

    def part_take(a, ks, axis):
        p = np.partition(a, ks, axis=axis)
        return np.take(p, ks, axis=axis) if axis is not None else np.take(p, ks)

    def argpart_take(a, ks, axis):
        g = np.argpartition(a, ks, axis=axis)
        if axis is None:
            return np.take(np.take(np.asarray(a).ravel(), g), ks)
        return np.take(np.take_along_axis(np.asarray(a), g, axis=axis), ks, axis=axis)

    for dt in dtypes:
        a1 = _distinct(8, dt)
        a2 = _distinct(12, dt).reshape(3, 4)
        for ks in ([3], [-2], [1, 5]):
            try:
                emit("partition", {"kth": ks, "axis": -1}, a1, part_take(a1, ks, -1), "1d", dt)
                emit("argpartition", {"kth": ks, "axis": -1}, a1, argpart_take(a1, ks, -1), "1d", dt)
            except Exception:
                continue
        for ax in (0, 1, -1):
            try:
                emit("partition", {"kth": [1], "axis": ax}, a2, part_take(a2, [1], ax), "2d", dt)
                emit("argpartition", {"kth": [1], "axis": ax}, a2, argpart_take(a2, [1], ax), "2d", dt)
            except Exception:
                continue
        try:
            emit("partition", {"kth": [5], "axis": None}, a2, part_take(a2, [5], None), "flat", dt)
            emit("argpartition", {"kth": [5], "axis": None}, a2, argpart_take(a2, [5], None), "flat", dt)
        except Exception:
            continue
    return cases


def gen_partition_nan():
    """Float/complex NaN partition (main tier only — not re-run by char_tier): kth in the non-NaN
    region AND in the NaN tail (NaN bytes are tokenized by BitDiff, so payload policy differences
    — NumSharp preserves original bits, unit-test-pinned — do not enter the compare)."""
    cases = []
    n = 0

    def emit(op, params, a, r, tag, dt):
        nonlocal n
        cases.append({
            "id": f"{op}/{tag}/{dt}/{n}",
            "op": op,
            "params": params,
            "operands": [describe(a, a)],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": tag,
            "valueclass": "nan",
        })
        n += 1

    nan = float("nan")
    for dt in ("float16", "float32", "float64"):
        a = np.array([nan, 4.0, -1.0, nan, 2.5, 0.5, -3.0, 6.0], dtype=np.dtype(dt))
        for ks in ([2], [7], [1, 6]):
            try:
                p = np.partition(a, ks)
                emit("partition", {"kth": ks, "axis": -1}, a, np.take(p, ks), "nan_1d", dt)
                g = np.argpartition(a, ks)
                emit("argpartition", {"kth": ks, "axis": -1}, a, np.take(np.take(a, g), ks), "nan_1d", dt)
            except Exception:
                continue
    c = np.array([complex(nan, 1), 2 + 3j, complex(1, nan), 0 + 1j, -1 + 0j])
    for ks in ([1], [3]):
        p = np.partition(c, ks)
        emit("partition", {"kth": ks, "axis": -1}, c, np.take(p, ks), "nan_1d", "complex128")
        g = np.argpartition(c, ks)
        emit("argpartition", {"kth": ks, "axis": -1}, c, np.take(np.take(c, g), ks), "nan_1d", "complex128")
    return cases


# G12 (issue #623) — lexsort IS stable on both sides (NumPy: successive mergesort passes;
# NumSharp: successive stable-radix argsorts), so the raw int64 index output is fully
# deterministic EVEN WITH TIES — which is exactly what the primary keys here carry, making the
# secondary key + the stability guarantee the thing each case pins.
def gen_lexsort(dtypes):
    cases = []
    n = 0

    def emit(keys, axis, tag, dt):
        nonlocal n
        try:
            r = np.lexsort(tuple(keys), axis=axis)
        except Exception:
            return
        cases.append({
            "id": f"lexsort/{tag}/{dt}/axis={axis}/{n}",
            "op": "lexsort",
            "params": {"axis": axis},
            "operands": [describe(k, k) for k in keys],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": tag,
            "valueclass": "ties",
        })
        n += 1

    sec = _distinct(8, "int64")
    for dt in dtypes:
        prim = np.array([1, 0, 1, 0, 1, 1, 0, 0]).astype(np.dtype(dt))   # deliberate ties
        emit([sec, prim], -1, "1d_2key", dt)                              # LAST key is primary
        emit([prim], -1, "1d_1key", dt)                                   # single key == stable argsort
    k0 = _distinct(12, "int64").reshape(3, 4)
    k1 = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1]).astype("float64").reshape(3, 4)
    for ax in (0, 1, -1):
        emit([k0, k1], ax, "2d_2key", "int64+float64")
    return cases


# G12 (issue #623) — sort_complex: copy + sort along the LAST axis in the input's own dtype, then
# up-cast to complex. int8/uint8/int16/uint16 are EXCLUDED from the tier: NumPy up-casts those to
# complex64, a WIDTH NumSharp's single Complex (complex128) does not have — the values are
# identical (probed 2.4.2), so those four cells are unit-test-pinned instead
# (Sorting/np.sort_complex.Test.cs). Excluding uint16 also keeps char_tier from weaving a
# width-mismatched char case.
SORT_COMPLEX_DTYPES = [d for d in ALL_DTYPES if d not in ("int8", "uint8", "int16", "uint16")]


def gen_sort_complex(dtypes):
    cases = []
    n = 0

    def emit(a, tag, dt):
        nonlocal n
        try:
            r = np.sort_complex(a)
        except Exception:
            return
        cases.append({
            "id": f"sort_complex/{tag}/{dt}/{n}",
            "op": "sort_complex",
            "params": {},
            "operands": [describe(a, a)],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": tag,
            "valueclass": "nan" if "nan" in tag else "distinct",
        })
        n += 1

    for dt in dtypes:
        emit(_distinct(8, dt), "1d", dt)
        emit(_distinct(12, dt).reshape(3, 4), "2d", dt)
    nan = float("nan")
    for dt in ("float16", "float32", "float64"):
        emit(np.array([nan, 2.0, -1.0, np.inf, -np.inf, 0.5], dtype=np.dtype(dt)), "nan_1d", dt)
    emit(np.array([complex(nan, 1), 2 + 3j, complex(1, nan), 0 + 1j]), "nan_1d", "complex128")
    return cases


# G11 (F20) — NaN sorting IS contractual in NumPy (NaN to the end; complex extended order
# [R+Rj, R+nanj, nan+Rj, nan+nanj] — probed 2.4.2 and matching in NumSharp) + strided/negstride
# operands (the NumPy-oracle sort tier was contiguous-only; only the decimal tier covered strided).
# Determinism guards: argsort operands keep ALL values distinct with at most ONE NaN per
# axis-slice (default quicksort is UNSTABLE — duplicate keys would make the index permutation
# implementation-defined on both sides); sort-only operands may carry duplicate NaNs (identical
# bit pattern -> identical result bytes). bool is sort-only in the strided family for the same
# tie reason.
def gen_sort_special():
    cases = []
    n = 0

    def emit(op, a_base, a_view, axis, tag):
        nonlocal n
        try:
            r = np.asarray(np.sort(a_view, axis=axis) if op == "sort" else np.argsort(a_view, axis=axis))
        except Exception:
            return
        cases.append({
            "id": f"{op}/{tag}/{a_view.dtype.name}/axis={axis}/{n}",
            "op": op,
            "params": {"axis": axis},
            "operands": [describe(a_base, a_view)],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": tag,
            "valueclass": "nan" if "nan" in tag else "distinct",
        })
        n += 1

    nan = float("nan")
    for dt in ["float16", "float32", "float64"]:
        d = np.dtype(dt)
        a1 = np.array([3.5, nan, -2.0, np.inf, 0.25, -np.inf, 7.0, 1.5], dtype=d)   # distinct + 1 NaN
        for op in ("sort", "argsort"):
            emit(op, a1, a1, -1, "nan_1d")
        a2 = np.array([nan, 4.0, -1.0, nan, 2.5, 0.5, -3.0, 6.0], dtype=d)          # 2 NaNs -> sort only
        emit("sort", a2, a2, -1, "nan_1d_multi")
        m = np.array([5.0, -3.5, 12.25, 0.5, nan, 8.0, -7.25, 3.0, 1.5, -0.25, 9.75, -12.5],
                     dtype=d).reshape(3, 4)                                          # distinct + 1 NaN
        for op in ("sort", "argsort"):
            for ax in (0, 1, -1):
                emit(op, m, m, ax, "nan_2d")
        sb = np.array([nan, 9.0, 3.5, 1.0, -2.0, 8.0, 0.5, 2.0, 7.0, 4.0, -np.inf, 5.0,
                       12.0, 6.0, np.inf, 0.0], dtype=d)                             # [::2] -> distinct + 1 NaN
        for op in ("sort", "argsort"):
            emit(op, sb, sb[::2], -1, "nan_strided")

    # complex: exactly ONE entry per NaN group (R+nanj, nan+Rj, nan+nanj) -> no within-group ties.
    cx = np.array([3 + 1j, complex(nan, 1), 1 + 2j, complex(1, nan), 2 + 0j,
                   complex(nan, nan), -1j, 5 + 3j], dtype=np.complex128)
    for op in ("sort", "argsort"):
        emit(op, cx, cx, -1, "nan_1d")

    # strided + negstride views over the distinct permutation pool, every sortable dtype.
    for dt in SORT_DTYPES:
        b = _distinct(16, dt)
        ops = ("sort",) if dt == "bool" else ("sort", "argsort")   # bool: 15 dup keys -> unstable ties
        for op in ops:
            emit(op, b, b[::2], -1, "strided")
            emit(op, b, b[::-1], -1, "negstride")
    return cases


def gen_unique(dtypes):
    """np.unique -> sorted distinct values, over CONTIGUOUS finite data with duplicates. Contiguous +
    finite on purpose: unique is correct via the public API (verified), but the corpus's raw-offset
    reconstructions hit the documented '#11 unreachable-via-API' representation gap, and inf/NaN
    ordering in a COMPLEX sort is implementation-defined — both out of scope for a dedup differential."""
    pools = {
        "bool": [True, False, True, True, False, False],
        "int32": [3, -1, 3, 7, -1, 0, 7, -128, 3, 127],
        "uint8": [5, 2, 5, 9, 2, 0, 9, 255, 5, 17],
        "int64": [3, -1, 3, 7, -1, 0, 7, -9999, 3, 12345],
        "float64": [1.5, -2.0, 1.5, 3.25, -2.0, 0.0, 3.25, -7.5],
        "float32": [1.5, -2.0, 1.5, 3.25, -2.0, 0.0, 3.25, -7.5],
        "complex128": [3 + 1j, 1 + 2j, 3 + 1j, 2 + 0j, 1 + 2j, 0 + 0j],   # finite only
    }
    cases = []
    n = 0
    for dt in dtypes:
        a = np.array(pools[dt], dtype=np.dtype(dt))
        r = np.asarray(np.unique(a))
        cases.append({
            "id": f"unique/1d/{dt}/{n}",
            "op": "unique",
            "params": {},
            "operands": [describe(a, a)],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": "1d",
            "valueclass": "dup",
        })
        n += 1
    return cases


def gen_round(dtypes, layout_names):
    """np.round_/around with decimals; every layout. NumPy is the oracle (banker's rounding).
    Since the Default.Round rewrite (port of PyArray_Round: op2(rint(op1(x, 10^|decimals|)), 10^|decimals|)
    at the input precision), NEGATIVE decimals, float16 fractional rounding and complex128 dec!=0 are all
    BIT-EXACT — the former carve-outs are gone. The remaining try/except skips only the cells NumPy itself
    RAISES on: bool with decimals!=0 (the multiply/divide -> bool same_kind cast fails). Negative decimals on
    integers compute in float64 and cast back (wrapping), also bit-exact."""
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            for dec in (-2, -1, 0, 1, 2):                 # negative decimals now bit-exact (was carved)
                try:
                    r = np.asarray(np.round(view, dec))
                except Exception:
                    skipped += 1                          # e.g. bool with decimals!=0 (NumPy raises)
                    continue
                cases.append({
                    "id": f"round_/{ln}/{s}/dec={dec}/{n}",
                    "op": "round_",
                    "params": {"decimals": dec},
                    "operands": [operand],
                    "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                 "buffer": np.ascontiguousarray(r).tobytes().hex()},
                    "layout": ln,
                    "valueclass": "mixed",
                })
                n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


def gen_trace_diag(dtypes):
    """np.trace (2-D -> 0-D sum of diagonal) and np.diagonal (2-D -> 1-D).
    G14: contiguous bases PLUS strided/offset views (a[1:5].T, a[:, ::2]) — the diagonal
    walk must honor a nonzero offset and non-contiguous strides."""
    cases = []
    n = 0
    for dt in dtypes:
        for shape in [(4, 4), (3, 5), (5, 3)]:
            a = _cbase(shape, dt)
            for opname, f in (("trace", np.trace), ("diagonal", np.diagonal)):
                try:
                    r = np.asarray(f(a))
                except Exception:
                    continue
                cases.append({
                    "id": f"{opname}/{shape[0]}x{shape[1]}/{dt}/{n}",
                    "op": opname,
                    "params": {},
                    "operands": [describe(a, a)],
                    "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                 "buffer": np.ascontiguousarray(r).tobytes().hex()},
                    "layout": "2d",
                    "valueclass": "mixed",
                })
                n += 1
    # G14 — strided/offset views (appended so the contiguous case ids above stay stable).
    for dt in dtypes:
        b1 = _cbase((6, 4), dt)
        b2 = _cbase((4, 6), dt)
        for (tag, base, view) in [("sliced_T", b1, b1[1:5].T), ("strided_cols", b2, b2[:, ::2])]:
            for opname, f in (("trace", np.trace), ("diagonal", np.diagonal)):
                try:
                    r = np.asarray(f(view))
                except Exception:
                    continue
                cases.append({
                    "id": f"{opname}/{tag}/{dt}/{n}",
                    "op": opname,
                    "params": {},
                    "operands": [describe(base, view)],
                    "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                 "buffer": np.ascontiguousarray(r).tobytes().hex()},
                    "layout": tag,
                    "valueclass": "mixed",
                })
                n += 1
    return cases


def gen_diag_tri(dtypes):
    """The diag/tri family that gen_manip's layout x dtype loop cannot express.

    Three groups, all appended after gen_trace_diag so existing ids stay stable:
      1. `tri` — a pure GENERATOR (no array input). The operand is a 1-element carrier
         whose dtype selects tri's dtype; N/M/k come from params.
      2. `diag`/`diagflat`/`tril`/`triu` on hand-built strided / F / negative-stride /
         offset 2-D views at controlled sizes (diagflat squares its input, so gen_manip
         caps it at size 8 — the bigger and non-contiguous shapes live here).
      3. `fill_diagonal` (mutating; result IS the mutated operand, like place/copyto) and
         the index-tuple generators (`*_indices`, `*_indices_from`, `mask_indices`), which
         return a tuple — `which` selects the element recorded, as gen_nonzero does.
    """
    cases = []
    n = 0

    def emit(opname, params, operands, r, layout):
        nonlocal n
        r = np.asarray(r)
        cases.append({
            "id": f"{opname}/{layout}/{n}",
            "op": opname,
            "params": params,
            "operands": operands,
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": layout,
            "valueclass": "mixed",
        })
        n += 1

    # -- 1. tri: dtype rides the carrier operand; N/M/k sweep the clamp/saturate corners.
    for dt in dtypes:
        carrier = _cbase((1,), np.dtype(dt))
        for (N, M, k) in [(4, None, 0), (3, 5, 0), (3, 5, 1), (3, 5, -1), (5, 3, 2),
                          (5, 3, -2), (4, 4, 10), (4, 4, -10), (0, None, 0), (3, 0, 0),
                          (0, 3, 0), (-2, None, 0), (3, -2, 0), (1, 1, 0)]:
            try:
                r = np.tri(N, M, k, dtype=np.dtype(dt))
            except Exception:
                continue
            emit("tri", {"N": N, "M": M, "k": k}, [describe(carrier, carrier)], r,
                 f"{N}x{M}k{k}/{dt}")

    # -- 2. diag / diagflat / tril / triu over non-contiguous 2-D views.
    for dt in dtypes:
        b_tall = _cbase((6, 4), np.dtype(dt))
        b_wide = _cbase((4, 6), np.dtype(dt))
        b_sq = _cbase((5, 5), np.dtype(dt))
        b_1d = _cbase((9,), np.dtype(dt))
        views = [
            ("sliced_T", b_tall, b_tall[1:5].T),          # offset + transposed
            ("strided_cols", b_wide, b_wide[:, ::2]),     # last-axis stride != 1
            ("negstride_cols", b_wide, b_wide[:, ::-1]),  # negative last-axis stride
            ("negstride_rows", b_sq, b_sq[::-1]),         # negative row stride
            ("f_order", b_sq, b_sq.T),                    # F-contiguous
            ("offset_sub", b_sq, b_sq[1:4, 1:4]),         # offset sub-block
            ("strided_1d", b_1d, b_1d[::3]),              # 1-D step view
            ("negstride_1d", b_1d, b_1d[::-2]),           # 1-D negative step
        ]
        for (tag, base, view) in views:
            for k in (0, 1, -1, 3):
                for opname, f in (("diag", np.diag), ("diagflat", np.diagflat),
                                  ("tril", np.tril), ("triu", np.triu)):
                    try:
                        r = np.asarray(f(view, k))
                    except Exception:
                        continue
                    emit(opname, {"k": k}, [describe(base, view)], r, f"{tag}/{dt}")

    # -- 3a. fill_diagonal: mutating. NumPy's flat-slice addressing is layout-independent,
    # so (as gen_place does) the oracle mutates a C-contiguous COPY of the view while the
    # harness mutates the real view — both must land on the same logical contents.
    for dt in dtypes:
        for (tag, shape) in [("square", (4, 4)), ("tall", (6, 3)), ("wide", (3, 6)),
                             ("cube", (3, 3, 3)), ("tall_narrow", (7, 2))]:
            base = _cbase(shape, np.dtype(dt))
            for wrap in (False, True):
                for val in ([7], [1, 2, 3], [1, 2, 3, 4, 5]):
                    after = np.array(base, copy=True)
                    try:
                        np.fill_diagonal(after, np.array(val, dtype=np.dtype(dt)), wrap)
                    except Exception:
                        continue
                    emit("fill_diagonal", {"val": val, "wrap": wrap},
                         [describe(base, base)], after, f"{tag}/{dt}/w{int(wrap)}")

        # non-contiguous destinations — the alias-block writer must honour real strides.
        nb = _cbase((5, 8), np.dtype(dt))
        for (tag, mk) in [("dst_strided", lambda b: b[:, ::2]),
                          ("dst_negstride", lambda b: b[:, ::-1]),
                          ("dst_T", lambda b: b.T),
                          ("dst_offset", lambda b: b[1:4, 1:5])]:
            base = np.array(nb, copy=True)
            view = mk(base)
            after_base = np.array(nb, copy=True)
            try:
                np.fill_diagonal(mk(after_base), np.array([9], dtype=np.dtype(dt)), False)
            except Exception:
                continue
            # Record the mutated VIEW's contents (the harness returns the view too).
            emit("fill_diagonal", {"val": [9], "wrap": False},
                 [describe(base, view)], mk(after_base), f"{tag}/{dt}")

    # -- 3b. index-tuple generators. Results are int64 coordinates, so one dtype suffices
    # for the array-taking forms; `which` picks the tuple element being recorded.
    idx_dt = np.dtype("int32")
    carrier = _cbase((1,), idx_dt)
    for (nn, ndim) in [(4, 2), (3, 3), (1, 2), (0, 2), (5, 4), (3, 1), (-1, 2)]:
        for which in range(max(ndim, 0)):
            try:
                r = np.diag_indices(nn, ndim)[which]
            except Exception:
                continue
            emit("diag_indices", {"n": nn, "ndim": ndim, "which": which},
                 [describe(carrier, carrier)], r, f"{nn}nd{ndim}w{which}")

    for (nn, k, m) in [(4, 0, None), (4, 1, None), (4, -1, None), (4, 0, 6), (4, 0, 2),
                       (5, 2, 3), (3, 10, None), (3, -10, None), (0, 0, None),
                       (3, 0, 0), (1, 0, None), (-2, 0, None), (3, 0, -2)]:
        for opname, f in (("tril_indices", np.tril_indices), ("triu_indices", np.triu_indices)):
            for which in (0, 1):
                try:
                    r = f(nn, k, m)[which]
                except Exception:
                    continue
                emit(opname, {"n": nn, "k": k, "m": m, "which": which},
                     [describe(carrier, carrier)], r, f"{nn}k{k}m{m}w{which}")

    for shape in [(4, 4), (3, 5), (5, 3), (0, 0), (1, 1)]:
        arr = _cbase(shape, idx_dt)
        for k in (0, 1, -1):
            for opname, f in (("tril_indices_from", np.tril_indices_from),
                              ("triu_indices_from", np.triu_indices_from)):
                for which in (0, 1):
                    try:
                        r = f(arr, k)[which]
                    except Exception:
                        continue
                    emit(opname, {"k": k, "which": which}, [describe(arr, arr)], r,
                         f"{shape[0]}x{shape[1]}k{k}w{which}")
        if shape[0] == shape[1]:
            for which in (0, 1):
                try:
                    r = np.diag_indices_from(arr)[which]
                except Exception:
                    continue
                emit("diag_indices_from", {"which": which}, [describe(arr, arr)], r,
                     f"{shape[0]}x{shape[1]}w{which}")

    # mask_indices takes a FUNCTION — the name is serialised and re-bound C#-side.
    for (fname, fobj) in [("triu", np.triu), ("tril", np.tril), ("diag", np.diag)]:
        for nn in (4, 3, 1, 0):
            for k in (0, 1, -1):
                try:
                    res = np.mask_indices(nn, fobj, k)
                except Exception:
                    continue
                for which in range(len(res)):
                    emit("mask_indices", {"n": nn, "func": fname, "k": k, "which": which},
                         [describe(carrier, carrier)], res[which], f"{fname}{nn}k{k}w{which}")

    return cases


def gen_ediff1d(dtypes, layout_names):
    """np.ediff1d — consecutive differences of the FLATTENED array (n-1 elements)."""
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            try:
                r = np.asarray(np.ediff1d(view))
            except Exception:
                skipped += 1
                continue
            cases.append({
                "id": f"ediff1d/{ln}/{s}/{n}",
                "op": "ediff1d",
                "params": {},
                "operands": [describe(base, view)],
                "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                             "buffer": np.ascontiguousarray(r).tobytes().hex()},
                "layout": ln,
                "valueclass": "mixed",
            })
            n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


def gen_nanquantile(dtypes):
    """np.nanpercentile / np.nanquantile — NaN-skipping order statistics. Uses FINITE values with a
    few NaNs injected (NO inf: percentile INTERPOLATION across inf is ill-defined — inf-inf=NaN — so
    NumPy/NumSharp legitimately diverge there; that edge is out of scope for the nan-skip differential)."""
    cases = []
    n = 0
    skipped = 0
    specs = [("nanpercentile", np.nanpercentile, [0.0, 25.0, 50.0, 75.0, 100.0]),
             ("nanquantile", np.nanquantile, [0.0, 0.25, 0.5, 0.75, 1.0])]
    for s in dtypes:
        dt = np.dtype(s)
        if dt.kind == "f":
            base1 = np.array([3.5, -2.0, np.nan, 7.25, 0.0, -9.5, 4.0, np.nan, 1.5, 6.0, -3.0, 2.5], dtype=dt)
        else:
            # Integer/bool lanes: no NaN can exist, so nanpercentile degenerates to percentile —
            # the point of gating them. NaN/negative literals would RAISE at construction for
            # unsigned/bool (NumPy 2.x bounds-checks python ints), and a float→uint astype is
            # C-undefined — so the pool is built as int64 and astype'd (well-defined modular).
            base1 = np.array([3, 250, 7, 0, 9, 4, 200, 1, 6, 255, 2, 5], dtype=np.int64).astype(dt)
        base2 = base1.reshape(3, 4)
        jobs = [(base1, None), (base1, 0)] + [(base2, ax) for ax in (None, 0, 1)]
        for (a, axis) in jobs:
            operand = describe(a, a)
            for (opname, f, qs) in specs:
                for q in qs:
                    try:
                        r = np.asarray(f(a, q, axis))
                    except Exception:
                        skipped += 1
                        continue
                    cases.append({
                        "id": f"{opname}/{a.ndim}d/{s}/q={q}/axis={axis}/{n}",
                        "op": opname,
                        "params": {"q": q, "axis": axis},
                        "operands": [operand],
                        "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                     "buffer": np.ascontiguousarray(r).tobytes().hex()},
                        "layout": f"{a.ndim}d",
                        "valueclass": "nan",
                    })
                    n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# ---------------------------------------------------------------------------
# Group A Batches 4-6: shape (flatten/rollaxis/append/insert), selection (take/compress/extract),
# math (convolve), multi-output split (one case per output piece). NumPy is the oracle.
# ---------------------------------------------------------------------------
def gen_groupa():
    cases = []
    n = 0

    def emit(opname, params, operands, r):
        nonlocal n
        r = np.asarray(r)
        cases.append({
            "id": f"{opname}/{n}", "op": opname, "params": params, "operands": operands,
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": "groupa", "valueclass": "mixed",
        })
        n += 1

    for dt in ["int32", "float64", "uint8", "complex128"]:
        d = np.dtype(dt)
        a2 = _cbase((3, 4), d)
        a3 = _cbase((2, 3, 4), d)

        # flatten — C-order copy (contiguous + a transposed, non-contiguous source).
        emit("flatten", {}, [describe(a2, a2)], a2.flatten())
        t3 = a3.transpose(2, 0, 1)
        emit("flatten", {}, [describe(a3, t3)], t3.flatten())

        # rollaxis — move `axis` to `start`.
        for (axis, start) in [(2, 0), (1, 0), (0, 2)]:
            emit("rollaxis", {"axis": axis, "start": start}, [describe(a3, a3)], np.rollaxis(a3, axis, start))

        # take — int64 indices, along an axis.
        base1 = _cbase((6,), d)
        idx1 = np.array([0, 3, 1, 3, 2], dtype=np.int64)
        emit("take", {"axis": 0}, [describe(base1, base1), describe(idx1, idx1)], np.take(base1, idx1, 0))
        idx2 = np.array([0, 2, 1], dtype=np.int64)
        emit("take", {"axis": 1}, [describe(a2, a2), describe(idx2, idx2)], np.take(a2, idx2, 1))

        # compress — bool condition selects along an axis.
        cond = np.array([True, False, True], dtype=bool)
        emit("compress", {"axis": 0}, [describe(cond, cond), describe(a2, a2)], np.compress(cond, a2, 0))

        # extract — bool mask (same shape) -> 1-D.
        mask = (_cbase((3, 4), np.dtype("int32")) % 2 == 0)
        emit("extract", {}, [describe(mask, mask), describe(a2, a2)], np.extract(mask, a2))

        # convolve — 1-D, all three modes.
        av = _cbase((7,), d)
        vv = _cbase((3,), d)
        for mode in ["full", "same", "valid"]:
            try:
                r = np.convolve(av, vv, mode)
            except Exception:
                continue
            emit("convolve", {"mode": mode}, [describe(av, av), describe(vv, vv)], r)

        # correlate — 1-D, all three modes, BOTH size relations. correlate is non-commutative:
        # (vv, av) has len(a) < len(v), so the engine correlates(v, a) and reverses the output.
        # complex128 additionally exercises the conjugation of the second argument.
        for mode in ["valid", "same", "full"]:
            for (ea, ev) in [(av, vv), (vv, av)]:
                try:
                    r = np.correlate(ea, ev, mode)
                except Exception:
                    continue
                emit("correlate", {"mode": mode}, [describe(ea, ea), describe(ev, ev)], r)

        # convolve / correlate of complex values holding infinities and NaNs. cblas' zdotu builds its result with C99
        # complex arithmetic (re + im*_Complex_I): an infinite / NaN imaginary part turns the real part into NaN — unless
        # NumPy's dot cannot hand an operand to cblas: a ONE-element operand keeps its own stride (np.convolve's kernel is
        # v[::-1], and a one-element array is C-contiguous whatever its stride), and a non-positive stride takes
        # CDOUBLE_dot's plain loop. Kernels shorter than zdotu's vector block (8) only: longer ones reorder the sums.
        if dt == "complex128":
            crng = np.random.default_rng(31)
            specials = [complex(np.inf, 0), complex(-np.inf, -0.0), complex(0, np.inf), complex(np.inf, np.inf),
                        complex(np.nan, 0), complex(0, np.nan), complex(1e308, 1e308), complex(-1e308, 1e308)]
            for n1c, n2c in ((1, 1), (3, 1), (1, 3), (5, 2), (2, 5), (6, 3), (9, 4), (12, 7), (7, 7)):
                for trial in range(3):
                    ac = (crng.standard_normal(n1c) + 1j * crng.standard_normal(n1c)).round(3)
                    vc = (crng.standard_normal(n2c) + 1j * crng.standard_normal(n2c)).round(3)
                    for _ in range(1 + trial % 2):
                        ac[crng.integers(n1c)] = specials[crng.integers(len(specials))]
                    if trial == 2:
                        vc[crng.integers(n2c)] = specials[crng.integers(len(specials))]
                    for mode in ["full", "same", "valid"]:
                        emit("convolve", {"mode": mode}, [describe(ac, ac), describe(vc, vc)], np.convolve(ac, vc, mode))
                        emit("correlate", {"mode": mode}, [describe(ac, ac), describe(vc, vc)], np.correlate(ac, vc, mode))
            # One-element operands with their own strides: fresh (+16: cblas), reversed (-16), stepped (+32 / -32),
            # stride 0 — as np.convolve's kernel (reversed first), as its data, and as np.correlate's first argument.
            cdata = np.array([1.5 - 0.5j, complex(0, np.inf), -2 + 1j, complex(np.inf, 1), 0.5j])
            src = np.array([complex(np.inf, 1.0), 2 + 3j, complex(1.0, np.inf)])
            ones = [("fresh", src[:1].copy(), None), ("reversed", src[:1].copy(), None),
                    ("step", src, src[::2][:1]), ("negstep", src, src[::-2][:1]),
                    ("stride0", src[:1].copy(), None)]
            for oname, obase, oview in ones:
                if oname == "fresh":
                    oview = obase
                elif oname == "reversed":
                    oview = obase[::-1]
                elif oname == "stride0":
                    oview = np.lib.stride_tricks.as_strided(obase, (1,), (0,))
                for mode in ["full", "same", "valid"]:
                    emit("convolve", {"mode": mode}, [describe(cdata, cdata), describe(obase, oview)],
                         np.convolve(cdata, oview, mode))
                    emit("convolve", {"mode": mode}, [describe(obase, oview), describe(cdata, cdata)],
                         np.convolve(oview, cdata, mode))
                    emit("correlate", {"mode": mode}, [describe(obase, oview), describe(cdata, cdata)],
                         np.correlate(oview, cdata, mode))
                    emit("correlate", {"mode": mode}, [describe(cdata, cdata), describe(obase, oview)],
                         np.correlate(cdata, oview, mode))
                    emit("convolve", {"mode": mode}, [describe(obase, oview), describe(obase, oview)],
                         np.convolve(oview, oview, mode))

        # append — flatten form (axis=None) + along axis 0.
        vals1 = _cbase((4,), d)
        emit("append", {}, [describe(a2, a2), describe(vals1, vals1)], np.append(a2, vals1))
        row = _cbase((1, 4), d)
        emit("append", {"axis": 0}, [describe(a2, a2), describe(row, row)], np.append(a2, row, 0))

        # insert — insert a row at obj=1 along axis 0.
        insvals = _cbase((4,), d)
        emit("insert", {"obj": 1, "axis": 0}, [describe(a2, a2), describe(insvals, insvals)], np.insert(a2, 1, insvals, 0))

        # Public shape/join surface that previously had no direct value oracle.
        cs0 = _cbase((4,), d)
        cs1 = np.roll(cs0, 1)
        emit("column_stack", {}, [describe(cs0, cs0), describe(cs1, cs1)],
             np.column_stack((cs0, cs1)))

        b0 = _cbase((2, 2), d)
        b1 = b0 + np.array(10, dtype=d)
        b2 = b0 + np.array(20, dtype=d)
        b3 = b0 + np.array(30, dtype=d)
        emit("block", {}, [describe(x, x) for x in (b0, b1, b2, b3)],
             np.block([[b0, b1], [b2, b3]]))

        bv = _cbase((3,), d)
        emit("broadcast_to", {"shape": [4, 3]}, [describe(bv, bv)],
             np.broadcast_to(bv, (4, 3)))
        bs = _cbase((), d)
        emit("broadcast_to", {"shape": [2, 3]}, [describe(bs, bs)],
             np.broadcast_to(bs, (2, 3)))

        # split / hsplit / vsplit / dsplit — one case per output piece.
        s = _cbase((6,), d)
        for pi, part in enumerate(np.split(s, 3)):
            emit("split", {"sections": 3, "axis": 0, "piece": pi}, [describe(s, s)], part)
        h = _cbase((3, 4), d)
        for pi, part in enumerate(np.hsplit(h, 2)):
            emit("hsplit", {"sections": 2, "piece": pi}, [describe(h, h)], part)
        v = _cbase((4, 4), d)
        for pi, part in enumerate(np.vsplit(v, 2)):
            emit("vsplit", {"sections": 2, "piece": pi}, [describe(v, v)], part)
        dd = _cbase((2, 2, 4), d)
        for pi, part in enumerate(np.dsplit(dd, 2)):
            emit("dsplit", {"sections": 2, "piece": pi}, [describe(dd, dd)], part)

        # put — mutate a copy at flat indices with values (returns the mutated array).
        pa = _cbase((6,), d)
        pidx = np.array([0, 2, 4], dtype=np.int64)
        pvals = _cbase((3,), d)
        pc = pa.copy()
        np.put(pc, pidx, pvals)
        emit("put", {}, [describe(pa, pa), describe(pidx, pidx), describe(pvals, pvals)], pc)

        # take — NEGATIVE indices under RAISE (NumPy's check_and_adjust_index normalizes
        # idx += n before the bounds test). Regression pin: the IL kernel used to reject every
        # negative index. axis 0 (== flat on a 1-D operand) and axis 1.
        nidx1 = np.array([-1, -6, 0, -3], dtype=np.int64)
        emit("take", {"axis": 0}, [describe(base1, base1), describe(nidx1, nidx1)], np.take(base1, nidx1, 0))
        nidx2 = np.array([-1, -4, 1], dtype=np.int64)
        emit("take", {"axis": 1}, [describe(a2, a2), describe(nidx2, nidx2)], np.take(a2, nidx2, 1))

        # take — wrap / clip modes over a positive-OOB + negative index mix.
        midx = np.array([7, -8, 2], dtype=np.int64)
        emit("take", {"axis": 0, "mode": "wrap"}, [describe(base1, base1), describe(midx, midx)],
             np.take(base1, midx, 0, mode="wrap"))
        emit("take", {"axis": 0, "mode": "clip"}, [describe(base1, base1), describe(midx, midx)],
             np.take(base1, midx, 0, mode="clip"))

        # take_along_axis — per-slice gather (indices match arr ndim; non-axis dims broadcast).
        # argsort-produced indices reproduce a sort; plus axis=None-flatten, negative-wrap,
        # J != M along the axis, broadcast indices on a non-axis dim, and a transposed (non-contig)
        # source. Result dtype == arr dtype; a negative index normalizes once (idx += n) as in
        # advanced indexing.
        tla1 = np.argsort(base1)                                        # (6,)
        emit("take_along_axis", {"axis": 0}, [describe(base1, base1), describe(tla1, tla1)],
             np.take_along_axis(base1, tla1, axis=0))
        flatidx = np.array([5, 0, 3, 3, 1, 2, 0], dtype=np.int64)       # axis=None -> flatten a2
        emit("take_along_axis", {"axis": None}, [describe(a2, a2), describe(flatidx, flatidx)],
             np.take_along_axis(a2, flatidx, axis=None))
        ai1 = np.argsort(a2, axis=1)                                    # (3,4)
        emit("take_along_axis", {"axis": 1}, [describe(a2, a2), describe(ai1, ai1)],
             np.take_along_axis(a2, ai1, axis=1))
        ai0 = np.argsort(a2, axis=0)                                    # (3,4)
        emit("take_along_axis", {"axis": 0}, [describe(a2, a2), describe(ai0, ai0)],
             np.take_along_axis(a2, ai0, axis=0))
        jidx = np.array([[0, 3, 1, 2, 0], [3, 3, 2, 1, 0], [1, 0, 2, 3, 3]], dtype=np.int64)  # (3,5), M=4
        emit("take_along_axis", {"axis": -1}, [describe(a2, a2), describe(jidx, jidx)],
             np.take_along_axis(a2, jidx, axis=-1))
        nidxt = np.array([[-1, -2, -3, -4], [-4, -3, -2, -1], [0, -1, 0, -1]], dtype=np.int64)  # neg-wrap
        emit("take_along_axis", {"axis": 1}, [describe(a2, a2), describe(nidxt, nidxt)],
             np.take_along_axis(a2, nidxt, axis=1))
        bidx0 = np.array([[0, 1, 2, 0]], dtype=np.int64)               # (1,4) broadcast over axis 0
        emit("take_along_axis", {"axis": 0}, [describe(a2, a2), describe(bidx0, bidx0)],
             np.take_along_axis(a2, bidx0, axis=0))
        bidx1 = np.array([[2], [0], [1]], dtype=np.int64)              # (3,1) keepdims-argmax style
        emit("take_along_axis", {"axis": 1}, [describe(a2, a2), describe(bidx1, bidx1)],
             np.take_along_axis(a2, bidx1, axis=1))
        ci2 = np.argsort(a3, axis=2)
        emit("take_along_axis", {"axis": 2}, [describe(a3, a3), describe(ci2, ci2)],
             np.take_along_axis(a3, ci2, axis=2))
        ci0 = np.argsort(a3, axis=0)
        emit("take_along_axis", {"axis": 0}, [describe(a3, a3), describe(ci0, ci0)],
             np.take_along_axis(a3, ci0, axis=0))
        t3b = a3.transpose(1, 0, 2)                                    # (3,2,4) non-contig source
        ti = np.argsort(t3b, axis=2)
        emit("take_along_axis", {"axis": 2}, [describe(a3, t3b), describe(ti, ti)],
             np.take_along_axis(t3b, ti, axis=2))

        # put_along_axis — SETTER twin of take_along_axis (mutate a COPY; result IS the mutated
        # array, exactly like `put` above). Mirrors take_along_axis's index families, now scattering
        # `vals` into `arr`. Operands are CONTIGUOUS (non-contiguous WRITE-THROUGH is unit-tested);
        # `vals` is BROADCAST — not cycled — to the indexing result shape, so full-shape, (M,)-row,
        # 0-d scalar and arr-broadcast (several positions collapsing onto one element, LAST write in
        # C-order winning) forms all appear. `axis=None` scatters into the C-order flat view. NumPy
        # is the oracle for every collision.
        def emit_pla(arr, idx, vals, axis):
            ac = arr.copy()
            np.put_along_axis(ac, idx, vals, axis=axis)
            emit("put_along_axis", {"axis": (None if axis is None else int(axis))},
                 [describe(arr, arr), describe(idx, idx), describe(vals, vals)], ac)

        pB1 = _cbase((6,), d)
        emit_pla(pB1, np.argsort(pB1).astype(np.int64), _cbase((6,), d) + 7, 0)                # 1-D argsort scatter
        pA2 = _cbase((3, 4), d)
        emit_pla(pA2, np.argsort(pA2, axis=1).astype(np.int64), _cbase((3, 4), d) + 7, 1)      # axis 1
        emit_pla(_cbase((3, 4), d), np.argsort(pA2, axis=0).astype(np.int64), _cbase((3, 4), d) + 7, 0)  # axis 0
        pflat = np.array([5, 0, 3, 3, 1, 2, 0, 8, 11, 4, 9, 6], dtype=np.int64)                # axis=None (12 idx)
        emit_pla(_cbase((3, 4), d), pflat, _cbase((12,), d) + 7, None)
        pj = np.array([[0, 3, 1, 2, 0], [3, 3, 2, 1, 0], [1, 0, 2, 3, 3]], dtype=np.int64)     # (3,5) J=5 != M=4
        emit_pla(_cbase((3, 4), d), pj, _cbase((3, 5), d) + 7, -1)
        pneg = np.array([[-1, -2, -3, -4], [-4, -3, -2, -1], [0, -1, 0, -1]], dtype=np.int64)  # neg-wrap
        emit_pla(_cbase((3, 4), d), pneg, _cbase((3, 4), d) + 7, 1)
        pb0 = np.array([[0, 1, 2, 0]], dtype=np.int64)                                         # (1,4) idx bcast over NON-axis dim 0
        emit_pla(_cbase((3, 4), d), pb0, _cbase((3, 4), d) + 7, 1)                              # -> result (3,4), idx row broadcast
        pb1 = np.array([[2], [0], [1]], dtype=np.int64)                                        # (3,1) keepdims-argmax style
        emit_pla(_cbase((3, 4), d), pb1, _cbase((3, 1), d) + 7, 1)
        pA3 = _cbase((2, 3, 4), d)                                                             # 3-D axis 2
        emit_pla(pA3, np.argsort(pA3, axis=2).astype(np.int64), _cbase((2, 3, 4), d) + 7, 2)
        # scalar (0-d) value broadcast, and a (4,)-row value broadcast over the axis-1 slices.
        emit_pla(_cbase((3, 4), d), np.argsort(pA2, axis=1).astype(np.int64), _cbase((), d) + 9, 1)
        emit_pla(_cbase((3, 4), d), np.argsort(pA2, axis=1).astype(np.int64), _cbase((4,), d) + 9, 1)
        # arr broadcast dim: arr (1,4), idx (3,4) -> 3 iteration rows collapse onto arr row 0 (last wins).
        emit_pla(_cbase((1, 4), d), np.array([[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 1, 1]], dtype=np.int64),
                 _cbase((3, 4), d) + 7, 1)

        # put — NEGATIVE indices under RAISE (same normalization as take).
        npa = _cbase((6,), d)
        npidx = np.array([-1, -6], dtype=np.int64)
        npvals = _cbase((2,), d)
        npc = npa.copy()
        np.put(npc, npidx, npvals)
        emit("put", {}, [describe(npa, npa), describe(npidx, npidx), describe(npvals, npvals)], npc)

        # select — bool conds + choices (dt) + 0-d default (dt). NumPy is the oracle for
        # first-match precedence, result dtype and broadcast. Conds are standalone bool arrays
        # (dtype-independent) so complex choices work too.
        selA = np.array([True, True, False, False, True, False])
        selB = np.array([False, True, True, True, False, False])
        chA = _cbase((6,), d)
        chB = _cbase((6,), d) + 10   # NEP50 weak-int add keeps dtype d; distinct from chA
        seldef = _cbase((), d)       # 0-d strong default
        emit("select", {"nc": 1}, [describe(selA, selA), describe(chA, chA), describe(seldef, seldef)],
             np.select([selA], [chA], seldef))
        emit("select", {"nc": 2},
             [describe(selA, selA), describe(selB, selB), describe(chA, chA), describe(chB, chB), describe(seldef, seldef)],
             np.select([selA, selB], [chA, chB], seldef))

        # choose — an int64 index selects, element-wise, among `nc` choice arrays (dtype d).
        # NumPy is the oracle for the per-position gather, the broadcast, and the mode arithmetic.
        # Operands are [index, choice0..choice_{nc-1}]; "nc" is the choice count. The result is a
        # pure byte-gather, so it is bit-exact once the operands are replayed from recorded bytes.
        ch0 = _cbase((6,), d)
        ch1 = _cbase((6,), d) + 10
        ch2 = _cbase((6,), d) + 20
        cidx3 = np.array([2, 0, 1, 2, 1, 0], dtype=np.int64)
        emit("choose", {"nc": 3, "mode": "raise"},
             [describe(cidx3, cidx3), describe(ch0, ch0), describe(ch1, ch1), describe(ch2, ch2)],
             np.choose(cidx3, [ch0, ch1, ch2]))
        cidx2 = np.array([0, 1, 1, 0, 1, 0], dtype=np.int64)
        emit("choose", {"nc": 2, "mode": "raise"},
             [describe(cidx2, cidx2), describe(ch0, ch0), describe(ch1, ch1)],
             np.choose(cidx2, [ch0, ch1]))
        # out-of-range indices exercise wrap (modulo, sign-corrected) and clip (saturate).
        coob = np.array([3, -1, 5, 0, -4, 2], dtype=np.int64)
        emit("choose", {"nc": 3, "mode": "wrap"},
             [describe(coob, coob), describe(ch0, ch0), describe(ch1, ch1), describe(ch2, ch2)],
             np.choose(coob, [ch0, ch1, ch2], mode="wrap"))
        emit("choose", {"nc": 3, "mode": "clip"},
             [describe(coob, coob), describe(ch0, ch0), describe(ch1, ch1), describe(ch2, ch2)],
             np.choose(coob, [ch0, ch1, ch2], mode="clip"))
        # 2-D index + 2-D choices — the strided odometer over the full result shape.
        ci2 = (np.arange(12).reshape(3, 4) % 3).astype(np.int64)
        cd0 = _cbase((3, 4), d)
        cd1 = _cbase((3, 4), d) + 5
        cd2 = _cbase((3, 4), d) + 9
        emit("choose", {"nc": 3, "mode": "raise"},
             [describe(ci2, ci2), describe(cd0, cd0), describe(cd1, cd1), describe(cd2, cd2)],
             np.choose(ci2, [cd0, cd1, cd2]))

        # --- neighbouring selection ops (strengthen the family `choose` sits in) ---
        # compress along axis 1 (the corpus above only had axis 0).
        ccond = np.array([True, False, True, True], dtype=bool)
        emit("compress", {"axis": 1}, [describe(ccond, ccond), describe(a2, a2)], np.compress(ccond, a2, 1))
        # take along axis 2 of a 3-D source (the corpus above only had axis 0 / 1).
        tidx3 = np.array([0, 2, 1, 3], dtype=np.int64)
        emit("take", {"axis": 2}, [describe(a3, a3), describe(tidx3, tidx3)], np.take(a3, tidx3, 2))

    # select — layout coverage: a TRANSPOSED cond+choice, a BROADCAST cond over a 2-D choice,
    # and an all-false fall-through to the default. int32 payload; the kernel is dtype-agnostic
    # (per-dtype value coverage is in the loop above).
    sd = np.dtype("int32")
    sm = _cbase((3, 4), sd)
    smask = (np.arange(12).reshape(3, 4) % 3 == 0)          # standalone bool cond
    sdef = _cbase((), sd)
    emit("select", {"nc": 1}, [describe(smask, smask.T), describe(sm, sm.T), describe(sdef, sdef)],
         np.select([smask.T], [sm.T], sdef))
    svec = np.array([True, False, True, False])              # (4,) cond broadcast over (3,4) choice
    emit("select", {"nc": 1}, [describe(svec, svec), describe(sm, sm), describe(sdef, sdef)],
         np.select([svec], [sm], sdef))
    sfalse = np.zeros((6,), dtype=bool)                     # all-false -> default everywhere
    sch = _cbase((6,), sd)
    emit("select", {"nc": 1}, [describe(sfalse, sfalse), describe(sch, sch), describe(sdef, sdef)],
         np.select([sfalse], [sch], sdef))

    # choose — layout + broadcast coverage (int32 payload; the gather is dtype-agnostic, so the
    # per-dtype value coverage is in the loop above). Reversed (negative-stride) index, a
    # column-reversed choice VIEW, a (3,1)x(1,4) broadcast, a scalar (0-d) choice broadcast, and
    # a 0-d index against 0-d choices (0-d result).
    cd = np.dtype("int32")
    lidx = np.array([0, 1, 0, 1, 1, 0], dtype=np.int64)
    lch0 = _cbase((6,), cd)
    lch1 = _cbase((6,), cd) + 10
    emit("choose", {"nc": 2, "mode": "raise"},
         [describe(lidx, lidx[::-1]), describe(lch0, lch0), describe(lch1, lch1)],
         np.choose(lidx[::-1], [lch0, lch1]))
    tci = (np.arange(12).reshape(3, 4) % 2).astype(np.int64)
    tc0 = _cbase((3, 4), cd)
    tc1 = _cbase((3, 4), cd) + 100
    emit("choose", {"nc": 2, "mode": "raise"},
         [describe(tci, tci), describe(tc0, tc0[:, ::-1]), describe(tc1, tc1)],
         np.choose(tci, [tc0[:, ::-1], tc1]))
    bidx = np.array([[0], [1], [0]], dtype=np.int64)        # (3,1) index
    bc0 = _cbase((1, 4), cd)                                # (1,4) choices -> broadcast to (3,4)
    bc1 = _cbase((1, 4), cd) + 50
    emit("choose", {"nc": 2, "mode": "raise"},
         [describe(bidx, bidx), describe(bc0, bc0), describe(bc1, bc1)],
         np.choose(bidx, [bc0, bc1]))
    sidx = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    sc0 = np.array(7, dtype=cd)                             # 0-d scalar choice
    sc1 = _cbase((5,), cd) + 20
    emit("choose", {"nc": 2, "mode": "raise"},
         [describe(sidx, sidx), describe(sc0, sc0), describe(sc1, sc1)],
         np.choose(sidx, [sc0, sc1]))
    zidx = np.array(1, dtype=np.int64)                      # 0-d index + 0-d choices -> 0-d result
    zc0 = np.array(3, dtype=cd)
    zc1 = np.array(9, dtype=cd)
    emit("choose", {"nc": 2, "mode": "raise"},
         [describe(zidx, zidx), describe(zc0, zc0), describe(zc1, zc1)],
         np.choose(zidx, [zc0, zc1]))

    # ravel_multi_index / unravel_index — index<->coord transforms (int64, dtype-independent).
    index_specs = [
        ([3, 4], np.array([0, 1, 2, 0], dtype=np.int64),
         np.array([1, 3, 0, 2], dtype=np.int64), "raise", "C"),
        ([3, 4], np.array([0, 1, 2, 0], dtype=np.int64),
         np.array([1, 3, 0, 2], dtype=np.int64), "raise", "F"),
        ([3, 4], np.array([-1, 3, 4, 1], dtype=np.int64),
         np.array([4, -1, 8, 2], dtype=np.int64), "wrap", "C"),
        ([3, 4], np.array([-1, 3, 4, 1], dtype=np.int64),
         np.array([4, -1, 8, 2], dtype=np.int64), "clip", "C"),
    ]
    for dims, row, col, mode, order in index_specs:
        emit("ravel_multi_index", {"dims": dims, "mode": mode, "order": order},
             [describe(row, row), describe(col, col)],
             np.ravel_multi_index((row, col), tuple(dims), mode=mode, order=order))

    unravel_specs = [
        (np.array([0, 5, 11, 7], dtype=np.int64), [3, 4], "C"),
        (np.array([0, 5, 11, 7], dtype=np.int64), [3, 4], "F"),
        (np.array([0, 7, 23, 13], dtype=np.int64), [2, 3, 4], "C"),
    ]
    for flat, shape, order in unravel_specs:
        for pi, part in enumerate(np.unravel_index(flat, tuple(shape), order=order)):
            emit("unravel_index", {"shape": shape, "order": order, "piece": pi},
                 [describe(flat, flat)], part)

    # ---- isin + set operations (arraysetops; NumPy _arraysetops_impl.py). --------------------
    # Value-dependent 2-operand ops: element/test (and ar1/ar2) must SHARE values for membership
    # to bite, so the fixtures are explicit overlapping arrays (not the arange-like pools). NaN in
    # the float set exercises "NaN is never a member". intersect1d(return_indices) returns a tuple
    # and is unit-tested; here only the single-array forms ride the corpus.
    def _setop_pair(dt):
        d = np.dtype(dt)
        if d.kind in "iu":
            a = np.array([0, 1, 2, 3, 4, 5, 2, 4], dtype=d)          # has duplicates
            b = np.array([1, 3, 5, 7], dtype=d)
        elif d.kind == "c":
            a = np.array([1 + 1j, 2 + 0j, 3 + 1j, 0 + 0j, 2 + 0j, 5 + 5j], dtype=d)
            b = np.array([2 + 0j, 3 + 1j, 9 + 9j], dtype=d)
        else:                                                        # float: include NaN
            a = np.array([0.0, 1.5, 2.0, np.nan, 4.0, 5.0, 2.0], dtype=d)
            b = np.array([1.5, np.nan, 5.0, 7.0], dtype=d)
        return a, b

    for dt in ["int32", "float64", "uint8", "complex128"]:
        a, b = _setop_pair(dt)
        empty = np.array([], dtype=np.dtype(dt))
        au_a, au_b = np.unique(a), np.unique(b)                      # unique => valid assume_unique contract
        emit("isin", {}, [describe(a, a), describe(b, b)], np.isin(a, b))
        emit("isin", {"invert": True}, [describe(a, a), describe(b, b)], np.isin(a, b, invert=True))
        emit("isin", {"assume_unique": True}, [describe(au_a, au_a), describe(au_b, au_b)],
             np.isin(au_a, au_b, assume_unique=True))
        emit("isin", {}, [describe(a, a), describe(empty, empty)], np.isin(a, empty))          # empty test
        emit("union1d", {}, [describe(a, a), describe(b, b)], np.union1d(a, b))
        emit("intersect1d", {}, [describe(a, a), describe(b, b)], np.intersect1d(a, b))
        emit("setxor1d", {}, [describe(a, a), describe(b, b)], np.setxor1d(a, b))
        emit("setdiff1d", {}, [describe(a, a), describe(b, b)], np.setdiff1d(a, b))

    # isin — kind selection (int-only for 'table') and non-contiguous element layouts (transposed,
    # negative-stride) so the reshape-to-C-order path is gated, not just contiguous elements.
    di = np.dtype("int32")
    e2 = np.array([[0, 2, 4], [6, 1, 3]], dtype=di)                  # (2,3)
    t2 = np.array([1, 2, 3, 4], dtype=di)
    emit("isin", {"kind": "sort"}, [describe(e2, e2), describe(t2, t2)], np.isin(e2, t2, kind="sort"))
    emit("isin", {"kind": "table"}, [describe(e2, e2), describe(t2, t2)], np.isin(e2, t2, kind="table"))
    emit("isin", {}, [describe(e2, e2.T), describe(t2, t2)], np.isin(e2.T, t2))            # transposed element
    emit("isin", {"invert": True}, [describe(e2, e2.T), describe(t2, t2)], np.isin(e2.T, t2, invert=True))
    e1 = np.arange(6, dtype=di)
    emit("isin", {}, [describe(e1, e1[::-1]), describe(t2, t2)], np.isin(e1[::-1], t2))     # negative-stride
    e1s = np.arange(12, dtype=di)
    emit("isin", {}, [describe(e1s, e1s[::2]), describe(t2, t2)], np.isin(e1s[::2], t2))    # strided

    # ---- piecewise (numpy/lib/_function_base_impl.py). --------------------------------------
    # Output dtype = x's dtype (zeros_like), the LAST true condition wins (forward overwrite), and
    # one extra function is the default (evaluated where every condition is false). Only SCALAR
    # (constant) funcs ride the corpus — they are deterministic and bit-comparable; callables and
    # the weak-scalar overflow / complex-into-real edges are unit-tested (as select's corpus does).
    # Operands are [x, cond0..cond_{nc-1}]; "nc" is the condition count and "funcs" the scalar
    # funclist (length nc or nc+1). Func values stay in [0,255] so they are in-range for every dtype
    # (uint8 included). Conditions are standalone bool arrays, so complex x works too.
    for dt in ["int32", "float64", "uint8", "complex128"]:
        d = np.dtype(dt)
        px = _cbase((8,), d)
        pc0 = np.array([True, True, False, False, True, False, True, False])
        pc1 = np.array([False, True, True, True, False, False, False, True])
        # nc == n2 — one func per condition.
        emit("piecewise", {"nc": 1, "funcs": [9]},
             [describe(px, px), describe(pc0, pc0)],
             np.piecewise(px, [pc0], [9]))
        emit("piecewise", {"nc": 2, "funcs": [1, 7]},
             [describe(px, px), describe(pc0, pc0), describe(pc1, pc1)],
             np.piecewise(px, [pc0, pc1], [1, 7]))
        # nc + 1 funcs — the extra is the default (the "otherwise" ~any(condlist) branch).
        emit("piecewise", {"nc": 2, "funcs": [1, 7, 4]},
             [describe(px, px), describe(pc0, pc0), describe(pc1, pc1)],
             np.piecewise(px, [pc0, pc1], [1, 7, 4]))
        # Overlapping conditions — LAST true wins (forward overwrite, the opposite of select).
        pov0 = np.array([True, True, True, True, True, True, True, True])
        pov1 = np.array([False, False, True, True, True, True, True, True])
        emit("piecewise", {"nc": 2, "funcs": [3, 8]},
             [describe(px, px), describe(pov0, pov0), describe(pov1, pov1)],
             np.piecewise(px, [pov0, pov1], [3, 8]))

    # piecewise — layout coverage (int32 payload; the composition is dtype-agnostic, so per-dtype
    # value coverage is the loop above). A 2-D x with a default, a TRANSPOSED x + cond (the
    # non-contiguous C-order gather/scatter, result compared C-contiguous via ResultBytes), and an
    # all-false condition falling through to the default everywhere.
    pdt = np.dtype("int32")
    p2 = _cbase((3, 4), pdt)
    pm0 = (np.arange(12).reshape(3, 4) % 3 == 0)
    pm1 = (np.arange(12).reshape(3, 4) % 3 == 1)
    emit("piecewise", {"nc": 2, "funcs": [5, 6, 2]},
         [describe(p2, p2), describe(pm0, pm0), describe(pm1, pm1)],
         np.piecewise(p2, [pm0, pm1], [5, 6, 2]))
    emit("piecewise", {"nc": 1, "funcs": [9, 1]},
         [describe(p2, p2.T), describe(pm0, pm0.T)],
         np.piecewise(p2.T, [pm0.T], [9, 1]))
    pfalse = np.zeros((6,), dtype=bool)
    p1 = _cbase((6,), pdt)
    emit("piecewise", {"nc": 1, "funcs": [0, 42]},
         [describe(p1, p1), describe(pfalse, pfalse)],
         np.piecewise(p1, [pfalse], [0, 42]))

    return cases


# ---------------------------------------------------------------------------
# Char masquerade — WOVEN into every tier (not a separate corpus file).
# ---------------------------------------------------------------------------
# NumSharp's Char is a 2-byte UNSIGNED value, bit-identical to uint16. NumPy has no
# char dtype, so each Char op is generated through uint16 as the NumPy proxy and the
# uint16 STRING is relabelled to "char" (raw bytes untouched, see _relabel_dtype).
# The Char cases are appended into the SAME tier file as their NumPy-native kin
# (binary_arith / unary / reduce / ...), so the existing per-tier FuzzMatrix test
# replays Char alongside int32/float64/etc. — Char is a first-class grid axis member.
#
# CARVE-OUTS (kept OUT of the green corpus; each reproduced under [OpenBugs] in
# OpenBugs.Char.cs / class OpenBugsCharTests): the combos that hit verified NumSharp Char bugs —
#   * any Char × {uint8,bool} pair   -> promote(Char,Byte)->Byte truncation (arith,
#     comparison, bitwise) + (Boolean,Char) missing kernel  [BUG: char-promote]
#   * reciprocal(char)               -> result dtype Double, should be uint16/char
#   * power with a char operand       -> Convert(char) crash / Double result
#   * invert(char)                    -> NotSupportedException on the N>=16 SIMD path
# Everything else (Char × {char,int32,int64,uint64,float64}, all other unary/reduce/
# scan/stat/manip/sort/tail/astype) is bit-identical to uint16 and ships GREEN.
_C = "uint16"   # the NumPy proxy for Char

# Char-bearing operand pairs. The uint16 slot IS the Char. uint8/bool deliberately
# absent (promotion/kernel bugs). Both operand orders covered; partners are all wider
# than Char so the kernel casts Char UP (no narrowing trap).
CHAR_ARITH_PAIRS = [(_C, _C), (_C, "int32"), ("int32", _C), (_C, "int64"),
                    (_C, "uint64"), (_C, "float64"), ("float64", _C)]
CHAR_CMP_PAIRS   = [(_C, _C), (_C, "int32"), ("int32", _C), (_C, "float64"), ("float64", _C)]
CHAR_BIT_PAIRS   = [(_C, _C), (_C, "int32"), (_C, "uint64")]

# Power crashes on any char operand; reciprocal mis-types char -> excluded per-op. float_power is
# carved for a different reason: it promotes char (uint16) to float64 and a large char exponent
# (e.g. 42**42) yields a finite NON-exact float64 whose last bit is host-libm dependent — the main
# divmod_power tier already covers float_power across every NumPy dtype, so char adds only that risk.
_CHAR_DIVMOD_OPS = {k: v for k, v in DIVMOD_POWER_OPS.items() if k not in ("power", "float_power")}
_CHAR_UNARY_OPS  = {k: v for k, v in UNARY_OPS.items() if k != "reciprocal"}

# G9 (F8) — pairs/op-sets for the additionally woven modes. The uint16 slot IS the Char;
# uint8/bool partners stay carved (char-promote bug), power/reciprocal/invert stay carved.
CHAR_WHERE_PAIRS = [(_C, _C), (_C, "int32"), ("float64", _C)]   # cond stays bool
CHAR_EXTREMA_OPS = {"maximum": np.maximum, "minimum": np.minimum, "fmax": np.fmax, "fmin": np.fmin}
CHAR_LOGIC_UNARY = {"isnan": np.isnan, "isinf": np.isinf, "isfinite": np.isfinite,
                    "logical_not": np.logical_not, "signbit": np.signbit}
CHAR_COPYTO_CROSS = [(_C, "int32"), ("int32", _C), (_C, "float64"), ("float64", _C)]


# ---------------------------------------------------------------------------
# The NumPy-ported float32 kernels - the bit-exact tier.
#
# exp/log/sin/cos at a float32 result are no longer "close enough": NDFloatMath ports the kernels
# NumPy 2.4.2 actually runs (simd_exp_FLOAT, simd_log_FLOAT, simd_sincos_f32), and rad2deg now forms
# its constant at float precision the way NumPy's RAD2DEG macro does - so the MisalignedRegistry's
# blanket "unary ~ULP" excuse is carved out for all of them and every case here must match BIT-for-
# BIT. The generic unary tier cannot carry that claim: its shared float pool is dominated by huge
# magnitudes (1e20, 3.5e38, ...) that saturate or reduce to nothing, leaving barely a dozen values
# that reach a polynomial at all. This tier feeds each kernel the inputs that discriminate.
#
# Layouts are built by hand (rather than through LAYOUTS) because the VALUES, not the shapes, are
# the point here; shape/stride coverage still spans contiguous, 2-D, F-view (transpose of a C base),
# strided, reversed, offset, broadcast, 0-d, empty and the narrow-integer inputs that share the
# same NumPy loop.
# ---------------------------------------------------------------------------

# exp: every special, both saturation boundaries +-1 ULP, the subnormal-output band, NumPy's own
# worst-error input (0xc2781e37, 2.52 ULP) and the FMA-contraction tie (0xc26d0e6c, where
# x*log2(e) is exactly -85.5 so fused and unfused rounding of the quadrant disagree).
_EXP_F32_SPECIAL_BITS = [
    0x7fc00000, 0x7fc00001, 0xffc00000, 0x7f800001,   # NaN: canonical, payload, negative, signalling
    0x7f800000, 0xff800000,                           # +-inf
    0x00000000, 0x80000000,                           # +-0
    0x00000001, 0x80000001, 0x007fffff, 0x00800000,   # subnormal inputs / smallest normal
    0x42b17216, 0x42b17217, 0x42b17218, 0x42b17219,   # xmax = 0x42b17218, +-1 ULP
    0xc2cff1b3, 0xc2cff1b4, 0xc2cff1b5, 0xc2cff1b6,   # xmin = 0xc2cff1b5, +-1 ULP
    0xc2aea8f6, 0xc2b00000, 0xc2c00000, 0xc2ce0000,   # subnormal-output band: -87.33, -88, -96, -103
    0xc2781e37, 0xc26d0e6c,
    0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
]

# log: the mantissa/exponent seams. NumPy splits the mantissa at 1/sqrt(2), rescales subnormals by
# 2^100, and returns a NEGATIVE NaN for a negative argument (but a POSITIVE one for a NaN argument).
_LOG_F32_SPECIAL_BITS = [
    0x7fc00000, 0xffc00000, 0x7f800001,                # NaN spellings
    0x7f800000, 0xff800000,                            # +-inf
    0x00000000, 0x80000000,                            # +-0 -> -inf
    0xbf800000, 0xc2c80000,                            # negatives -> -NaN
    0x00000001, 0x00000002, 0x007fffff, 0x00800000,    # subnormals and the smallest normal
    0x3f800000, 0x3f3504f3, 0x3f3504f4, 0x3f3504f2,    # 1.0 and the 1/sqrt(2) split, +-1 ULP
    0x3f000000, 0x40000000, 0x402df854, 0x7f7fffff,    # 0.5, 2, e, max finite
    0x3f486945,                                        # NumPy's documented worst case (3.83 ULP)
]

# sin/cos: the quadrant seams and the Cody-Waite cutoffs past which NumPy hands over to libc - a
# DIFFERENT cutoff per function (117435.992 for sine, 71476.0625 for cosine).
_TRIG_F32_SPECIAL_BITS = [
    0x7fc00000, 0xffc00000, 0x7f800000, 0xff800000,    # NaN, +-inf
    0x00000000, 0x80000000,                            # +-0
    0x3fc90fdb, 0xbfc90fdb, 0x40490fdb, 0xc0490fdb,    # +-pi/2, +-pi
    0x40c90fdb, 0x41490fdb, 0x3f490fdb,                # 2pi, 4pi, pi/4
    0x47e55dfe, 0x47e55dff, 0x47e55e00,                # sine's Cody-Waite limit, +-1 ULP
    0x478b9a07, 0x478b9a08, 0x478b9a09,                # cosine's limit, +-1 ULP
    0x4b000000, 0x50000000, 0x7f7fffff,                # far past both limits (libc fallback)
    0x00000001, 0x3f800000, 0xbf800000,
]


# tanh: NumPy's kernel picks its polynomial from a 32-entry table indexed by the exponent of |x|,
# so the interesting inputs are the SUBINTERVAL SEAMS - a wrong index shows up only there. Index is
# clamp(bits & 0x7fe00000 - 0x3d400000, 0, 0x3e00000) >> 21, so seam k sits at bits
# 0x3d400000 + k*0x200000; the saturation cut (past which the answer is exactly +-1) is 0x7f000000.
_TANH_F32_SPECIAL_BITS = [
    0x7fc00000, 0x7fc00001, 0xffc00000, 0x7f800001,   # NaN: canonical, payload, negative, signalling
    0x7f800000, 0xff800000,                           # +-inf -> +-1
    0x00000000, 0x80000000,                           # +-0   -> +-0 (the sign must survive)
    0x00000001, 0x80000001, 0x007fffff, 0x00800000,   # subnormals / smallest normal
    0x7f7fffff, 0xff7fffff,                           # +-FLT_MAX (saturated)
    0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,   # +-1, +-2
]


def _f32(bits):
    return np.array(bits, dtype=np.uint32).view(np.float32)


def _f64(bits):
    return np.array(bits, dtype=np.uint64).view(np.float64)


def _tanh_f32_values():
    rng = np.random.RandomState(20260728)
    seams = []
    for k in range(33):                                    # every subinterval seam, +-1 ULP
        b = 0x3d400000 + k * 0x200000
        if b < 0x7f800000:
            seams += [b - 1, b, b + 1]
    seams += [0x7effffff, 0x7f000000, 0x7f000001]          # the saturation cut, +-1 ULP
    seams += [b | 0x80000000 for b in list(seams)]         # tanh is odd - mirror every seam
    return np.concatenate([
        _f32(_TANH_F32_SPECIAL_BITS),
        _f32(seams),
        np.linspace(-10.0, 10.0, 121).astype(np.float32),
        rng.uniform(-20.0, 20.0, 96).astype(np.float32),
    ])


def _tanh_f64_values():
    """The float64 half of the same kernel: 16 subintervals, seams every 0x0008.. in the exponent."""
    rng = np.random.RandomState(20260729)
    seams = []
    for k in range(17):
        b = 0x3fc0000000000000 + k * 0x0008000000000000
        if b < 0x7ff0000000000000:
            seams += [b - 1, b, b + 1]
    seams += [0x7fdfffffffffffff, 0x7fe0000000000000, 0x7fe0000000000001]
    seams += [b | 0x8000000000000000 for b in list(seams)]
    specials = [
        0x7ff8000000000000, 0xfff8000000000000, 0x7ff0000000000001,   # NaN variants
        0x7ff0000000000000, 0xfff0000000000000,                       # +-inf
        0x0000000000000000, 0x8000000000000000,                       # +-0
        0x0000000000000001, 0x000fffffffffffff, 0x0010000000000000,   # subnormals / smallest normal
        0x7fefffffffffffff, 0xffefffffffffffff,                       # +-DBL_MAX
        0x3ff0000000000000, 0xbff0000000000000,                       # +-1
    ]
    return np.concatenate([
        _f64(specials),
        _f64(seams),
        np.linspace(-10.0, 10.0, 121),
        rng.uniform(-20.0, 20.0, 96),
    ])


def _exp_f32_values():
    rng = np.random.RandomState(20260725)
    return np.concatenate([
        _f32(_EXP_F32_SPECIAL_BITS),
        np.linspace(-104.0, 88.7, 121).astype(np.float32),
        np.linspace(-3.0, 3.0, 61).astype(np.float32),
        rng.uniform(-104.0, 88.7, 96).astype(np.float32),
    ])


def _log_f32_values():
    rng = np.random.RandomState(20260726)
    return np.concatenate([
        _f32(_LOG_F32_SPECIAL_BITS),
        np.logspace(-38, 38, 121).astype(np.float32),          # the whole exponent range
        np.linspace(0.5, 2.0, 61).astype(np.float32),          # around the polynomial's centre
        np.abs(rng.uniform(0, 1, 96) * 10.0 ** rng.uniform(-30, 30, 96)).astype(np.float32),
    ])


def _trig_f32_values():
    rng = np.random.RandomState(20260727)
    quads = np.concatenate([np.float32(np.pi / 2) * k + np.linspace(-1e-3, 1e-3, 5).astype(np.float32)
                            for k in range(-6, 7)]).astype(np.float32)
    return np.concatenate([
        _f32(_TRIG_F32_SPECIAL_BITS),
        quads,                                                  # every quadrant boundary
        np.linspace(-20.0, 20.0, 121).astype(np.float32),
        rng.uniform(-1e5, 1e5, 96).astype(np.float32),          # straddles both libc cutoffs
        rng.uniform(-1e7, 1e7, 32).astype(np.float32),          # well past them
    ])


def gen_numpy_f32_kernels():
    cases = []
    n = 0

    def emit(op, f, layout, base, view):
        nonlocal n
        r = f(view)
        cases.append({
            "id": f"{op}/{layout}/{view.dtype.name}/{n}",
            "op": op, "params": {},
            "operands": [describe(base, view)],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": layout, "valueclass": "kernel_edges",
        })
        n += 1

    jobs = [
        ("exp", np.exp, _exp_f32_values()),
        ("log", np.log, _log_f32_values()),
        ("sin", np.sin, _trig_f32_values()),
        ("cos", np.cos, _trig_f32_values()),
        ("rad2deg", np.rad2deg, _trig_f32_values()),
        ("deg2rad", np.deg2rad, _trig_f32_values()),
        ("tanh", np.tanh, _tanh_f32_values()),
    ]

    for op, f, v in jobs:
        emit(op, f, "contig1d", v, v)
        emit(op, f, "strided2", v, v[::2])
        emit(op, f, "reversed", v, v[::-1])
        emit(op, f, "offset", v, v[7:])
        emit(op, f, "offset_strided3", v, v[5::3])

        rows = 4
        cols = (v.size // rows) * rows
        m = np.ascontiguousarray(v[:cols].reshape(rows, cols // rows))
        emit(op, f, "contig2d", m, m)
        # NB: an F-CONTIGUOUS operand is spelled as the transpose of a C base, never as an
        # asfortranarray base - describe() serializes base.tobytes() in C order, so an F-ordered
        # base would record bytes that disagree with its own strides.
        emit(op, f, "transposed", m, m.T)
        emit(op, f, "row_reversed", m, m[:, ::-1])
        emit(op, f, "col_strided", m, m[:, ::2])

        one = np.ascontiguousarray(v[:16].reshape(1, 16))
        emit(op, f, "broadcast", one, np.broadcast_to(one, (3, 16)))

        for bits in (0x7fc00000, 0x7f800000, 0xff800000, 0x80000000, 0x3f800000):
            z = np.array([bits], dtype=np.uint32).view(np.float32).reshape(())
            emit(op, f, "zerod", z, z)

        e = np.zeros(0, dtype=np.float32)
        emit(op, f, "empty", e, e)

        # The narrow integer dtypes whose NumPy loop is this SAME 'f->f' kernel (int32 and wider
        # promote to the float64 loop instead).
        for dt in ("int16", "uint16"):
            iv = np.array([0, 1, 2, 3, 5, 11, 87, 88, 89, 90, -1, -5, -87, -88, -103, -104],
                          dtype=np.int64).astype(dt)
            emit(op, f, "int_contig", iv, iv)
            emit(op, f, "int_reversed", iv, iv[::-1])
    return cases


def gen_numpy_f64_kernels():
    """
    tanh is the only one of the ported kernels that also replaces a FLOAT64 loop: NumPy ships its
    own table-driven tanh at both widths (loops_hyperbolic), where exp/log/sin/cos delegate to the
    platform's scalar npy_* at f8 and already agree. Hence a tier of its own rather than more rows
    in numpy_f32_kernels.jsonl - the layouts mirror it exactly, only the dtype axis differs.
    """
    cases = []
    n = 0

    def emit(op, f, layout, base, view):
        nonlocal n
        r = f(view)
        cases.append({
            "id": f"{op}/{layout}/{view.dtype.name}/{n}",
            "op": op, "params": {},
            "operands": [describe(base, view)],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": layout, "valueclass": "kernel_edges",
        })
        n += 1

    for op, f, v in [("tanh", np.tanh, _tanh_f64_values())]:
        emit(op, f, "contig1d", v, v)
        emit(op, f, "strided2", v, v[::2])
        emit(op, f, "reversed", v, v[::-1])
        emit(op, f, "offset", v, v[7:])
        emit(op, f, "offset_strided3", v, v[5::3])

        rows = 4
        cols = (v.size // rows) * rows
        m = np.ascontiguousarray(v[:cols].reshape(rows, cols // rows))
        emit(op, f, "contig2d", m, m)
        # NB: an F-CONTIGUOUS operand is spelled as the transpose of a C base, never as an
        # asfortranarray base - describe() serializes base.tobytes() in C order, so an F-ordered
        # base would record bytes that disagree with its own strides.
        emit(op, f, "transposed", m, m.T)
        emit(op, f, "row_reversed", m, m[:, ::-1])
        emit(op, f, "col_strided", m, m[:, ::2])

        one = np.ascontiguousarray(v[:16].reshape(1, 16))
        emit(op, f, "broadcast", one, np.broadcast_to(one, (3, 16)))

        for bits in (0x7ff8000000000000, 0x7ff0000000000000, 0xfff0000000000000,
                     0x8000000000000000, 0x3ff0000000000000):
            z = np.array([bits], dtype=np.uint64).view(np.float64).reshape(())
            emit(op, f, "zerod", z, z)

        e = np.zeros(0, dtype=np.float64)
        emit(op, f, "empty", e, e)

        # int32 and wider promote to THIS float64 loop (int16/uint16 take the 'f->f' one instead,
        # which is why they live in the float32 tier).
        for dt in ("int32", "uint32", "int64", "uint64"):
            iv = np.array([0, 1, 2, 3, 5, 9, 10, 11, 19, 20, 21, 100, 1000], dtype=np.int64).astype(dt)
            emit(op, f, "int_contig", iv, iv)
            emit(op, f, "int_reversed", iv, iv[::-1])
    return cases


def char_tier(mode):
    """Relabelled Char cases to append into tier-file `mode` (woven coverage)."""
    L = list(LAYOUTS.keys())
    PL = list(PAIR_LAYOUTS.keys())
    raw = []
    if mode == "binary":
        raw = gen_binary(BINARY_OPS, CHAR_ARITH_PAIRS, PL)
    elif mode == "divmod_power":
        raw = gen_binary(_CHAR_DIVMOD_OPS, CHAR_ARITH_PAIRS, PL)   # floor_divide, mod (power carved)
    elif mode == "comparison":
        raw = gen_binary(COMPARISON_OPS, CHAR_CMP_PAIRS, PL)
    elif mode == "unary":
        raw = gen_unary(_CHAR_UNARY_OPS, [_C], L)                  # reciprocal carved
    elif mode == "unary_extra":
        raw = gen_unary(UNARY_EXTRA_OPS, [_C], L)
    elif mode == "sinc":
        raw = gen_unary(SINC_OP, [_C], L)                         # Char (uint16 proxy) -> float64, bit-exact
    elif mode == "i0":
        raw = gen_unary(I0_OP, [_C], L)                           # Char (uint16 proxy) -> float64, bit-exact
    elif mode == "bitwise":
        raw = gen_binary(BITWISE_BIN_OPS, CHAR_BIT_PAIRS, PL)
        raw += gen_unary(BITWISE_COUNT_OP, [_C], L)               # bitwise_count(char): 2-byte SIMD path works
        raw += gen_shift(SHIFT_OPS, [_C])                          # invert(char) carved (SIMD gap)
    elif mode == "gcd":
        # Char (uint16 proxy) rides gcd/lcm — every CHAR_BIT_PAIR promotes to a valid integer loop
        # (char+char->char, char+int32->int32, char+uint64->uint64), all NumSharp-supported.
        raw = gen_binary(GCDLCM_OPS, CHAR_BIT_PAIRS, PL)
    elif mode == "reduce":
        raw = gen_reduce(REDUCE_OPS, [_C], REDUCE_LAYOUTS)
    elif mode == "scan":
        raw = gen_scan(SCAN_OPS, [_C], SCAN_LAYOUTS) + gen_diff([_C], SCAN_LAYOUTS)
    elif mode == "unwrap":
        raw = gen_unwrap([_C], SCAN_LAYOUTS)                          # Char (uint16 proxy): float-period only
    elif mode == "stat":
        raw = gen_reduce(STAT_REDUCE_OPS, [_C], STAT_LAYOUTS)
        raw += gen_count_nonzero([_C], STAT_LAYOUTS)
        raw += gen_quantile(QUANTILE_SPECS, [_C], STAT_LAYOUTS)
        raw += gen_clip([_C], STAT_LAYOUTS)
    elif mode == "manip":
        raw = gen_manip([_C], L) + gen_concat_stack([_C]) + gen_pad([_C])
    elif mode == "sort":
        raw = gen_argsort([_C]) + gen_searchsorted([_C]) + gen_nonzero([_C])
        raw += gen_partition_family([_C]) + gen_lexsort([_C])          # G12 (sort_complex carved: u16->c64 width)
    elif mode == "tail":
        raw = gen_tail([_C])
    elif mode == "astype_full":
        raw = gen_astype([_C], ALL_DTYPES, L) + gen_astype(ALL_DTYPES, [_C], L)
    elif mode == "where":                                          # G9: char select values
        raw = gen_where(CHAR_WHERE_PAIRS, list(WHERE_LAYOUTS.keys()))
    elif mode == "logic":                                          # G9: extrema + predicates
        raw = gen_binary(CHAR_EXTREMA_OPS, CHAR_CMP_PAIRS, PL)
        raw += gen_unary(CHAR_LOGIC_UNARY, [_C], L)
    elif mode == "matmul":                                         # G9: uint16@uint16 modular GEMM
        # dot 1-D.1-D CARVED: NumSharp's vector-dot reduces through sum_elementwise_il with an
        # explicit Char result typecode, and that switch has no Char arm -> NotSupportedException
        # ("Sum not supported for type Char"). matmul 1-D.1-D and every 2-D+ char case work.
        # Pinned at OpenBugsFuzzGapsTests.Dot_Char_1D_Throws.
        shape_cases = [c for c in MATMUL_SHAPE_CASES if not (c[0] == "dot" and c[1] == (4,))]
        raw = gen_matmul(shape_cases, [_C], MATMUL_LAYOUTS)
    elif mode == "rounding":                                       # G9: char identity, dec 0/1/2
        raw = gen_round([_C], L)
    elif mode == "copyto":                                         # G9: overlap + int32/float64 cross
        raw = gen_copyto([_C], CHAR_COPYTO_CROSS)
    elif mode == "instance":                                       # ndarray.* instance surface on the proxy
        raw = gen_instance([_C])
    elif mode == "modf":                                           # modf(char) -> (float32, float32) per the
        raw = gen_modf([_C], MODF_LAYOUTS)                         # 59f99320 per-width promotion tier
    return _relabel_dtype(raw, _C, "char")


# ---------------------------------------------------------------------------
# T-parity — np.dot / np.matmul BYTE parity for the opt-in BLAS backend
# (np.parity_matmul). Unlike every other tier this one is HOST-PINNED: NumPy
# computes float matrix products with cblas, and scipy-openblas' sgemm/dgemm
# accumulate in an arch-specific multi-accumulator scheme whose bits depend on
# the BLAS binary, the CPU kernel it dispatches to, AND the thread count. The
# expected bytes below are therefore only reproducible on a host that loads the
# SAME library and dispatches the same way, which is why the tier ships a
# `matmul_parity.host.jsonl` pin and the C# gate goes Inconclusive (never red)
# when the host does not match. Same precedent as the MSVC-pinned cast kernels.
#
# The ordinary `matmul` tier cannot cover this: its operands are tiny integers
# and its largest contraction is k=4, where every summation order agrees. Real
# divergence starts at k=10 (45% of elements on the MLP shapes) and reaches 94%
# at k=784, so this tier sweeps k across the blocking boundaries with random
# float values, in every layout the two dispatchers route differently.
MATMUL_PARITY_DTYPES = ["float32", "float64", "complex128"]

# k values: 1..4 (agreeing region), the powers of two and their +-1 neighbours
# (OpenBLAS panel edges), NumSharp's own KC=256 boundary, and the MLP's 784.
MATMUL_PARITY_KS = [1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                    127, 128, 129, 255, 256, 257, 511, 512, 784]


def _mp_values(shape, dt, rng, valueclass="normal"):
    """Operand values. Random by default — regular ramps hide reassociation error.

    complex128 draws an INDEPENDENT real and imaginary part (a real-only .astype would zero the
    imag and never exercise zgemm/zdotc's cross terms); every value class applies to both parts.
    """
    n = int(np.prod(shape)) if shape else 1
    ndt = np.dtype(dt)

    def draw():
        if valueclass == "wide":
            # Magnitudes spanning ~40 decades: summation order dominates the result.
            return rng.standard_normal(n) * (10.0 ** rng.randint(-18, 18, n))
        return rng.standard_normal(n)

    a = draw() + 1j * draw() if ndt.kind == "c" else draw()

    if valueclass == "specials" and n >= 4:
        if ndt.kind == "c":
            a[0] = complex(np.inf, 1.0)
            a[1] = complex(-np.inf, np.nan)
            a[2] = complex(np.nan, np.inf)
            a[3] = complex(0.0, -0.0)
        else:
            a[0] = np.inf
            a[1] = -np.inf
            a[2] = np.nan
            a[3] = 0.0

    return np.ascontiguousarray(a.astype(ndt).reshape(shape))


def _mp_layout(arr, kind, rng):
    """(base, view) holding EXACTLY arr's values in the requested memory layout.

    Every kind produces a genuine view into a C-contiguous base (what the corpus
    descriptor can express), so the C# side rebuilds the same strides NumPy had —
    which is what selects the route in both dispatchers.
    """
    if kind == "C" or arr.ndim == 0:
        base = np.ascontiguousarray(arr)
        return base, base
    if kind == "F":
        base = np.ascontiguousarray(arr.T)          # transposed data, C-contiguous
        return base, base.T
    if kind == "neg":                               # 1-D reversed
        base = np.ascontiguousarray(arr[::-1])
        return base, base[::-1]
    if kind == "negrow":
        base = np.ascontiguousarray(arr[::-1])
        return base, base[::-1]
    if kind == "negcol":
        base = np.ascontiguousarray(arr[:, ::-1])
        return base, base[:, ::-1]
    if kind == "stride2":                           # last axis step 2 — never blasable
        shape = arr.shape[:-1] + (arr.shape[-1] * 2,)
        base = _mp_values(shape, arr.dtype, rng)
        base[..., ::2] = arr
        return base, base[..., ::2]
    if kind == "slice":                             # row stride > ncols, offset != 0
        m, n = arr.shape
        base = _mp_values((m + 3, n + 7), arr.dtype, rng)
        base[2:2 + m, 5:5 + n] = arr
        return base, base[2:2 + m, 5:5 + n]
    raise ValueError(kind)


def _mp_case(cases, op, name, A, ar, B, br, rng, valueclass="normal"):
    """Emit one parity case: apply the layout recipes, ask NumPy, record."""
    baseA, viewA = _mp_layout(A, ar, rng)
    baseB, viewB = _mp_layout(B, br, rng)
    f = np.dot if op == "dot" else np.matmul
    r = np.asarray(f(viewA, viewB))
    cases.append({
        "id": f"{op}/{name}/{ar}{br}/{A.dtype.name}x{B.dtype.name}/{len(cases)}",
        "op": op,
        "params": {},
        "operands": [describe(baseA, viewA), describe(baseB, viewB)],
        "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                     "buffer": np.ascontiguousarray(r).tobytes().hex()},
        "layout": f"{ar}{br}",
        "valueclass": valueclass,
    })


_MP_PRODUCT_FNS = {"inner": np.inner, "vdot": np.vdot, "vecdot": np.vecdot,
                   "matvec": np.matvec, "vecmat": np.vecmat}


def _mp_prod_case(cases, op, name, A, ar, B, br, rng, valueclass="normal"):
    """Emit one parity case for the CBLAS product family beyond dot/matmul.

    Same host-pinned, backend-enabled gate as _mp_case, but for inner/vdot/vecdot/matvec/vecmat —
    whose managed fallbacks (Multiply+ReduceAdd for vecdot, conj+dotu for complex vdot, conj+gemv
    for complex vecmat) reassociate the sum differently than NumPy's cblas dot/dotc/gemm and so are
    NOT byte-identical here. Deep contraction lengths and every routing layout make that show.
    OpRegistry dispatches vecdot with axis=-1 by default, matching np.vecdot's last-axis core.
    """
    baseA, viewA = _mp_layout(A, ar, rng)
    baseB, viewB = _mp_layout(B, br, rng)
    r = np.asarray(_MP_PRODUCT_FNS[op](viewA, viewB))
    cases.append({
        "id": f"{op}/{name}/{ar}{br}/{A.dtype.name}x{B.dtype.name}/{len(cases)}",
        "op": op,
        "params": {},
        "operands": [describe(baseA, viewA), describe(baseB, viewB)],
        "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                     "buffer": np.ascontiguousarray(r).tobytes().hex()},
        "layout": f"{ar}{br}",
        "valueclass": valueclass,
    })


def gen_matmul_parity():
    cases = []
    rng = np.random.RandomState(20260725)

    def V(shape, dt, vc="normal"):
        return _mp_values(shape, dt, rng, vc)

    for dt in MATMUL_PARITY_DTYPES:
        # --- k sweep: the blocking boundaries the `matmul` tier (k<=4) never crosses.
        for k in MATMUL_PARITY_KS:
            _mp_case(cases, "dot", f"ksweep_k{k}", V((6, k), dt), "C", V((k, 5), dt), "C", rng)

        # --- the MLP sites, shrunk in M/N but at the real contraction depths.
        _mp_case(cases, "dot", "mlp_k784", V((8, 784), dt), "C", V((784, 8), dt), "C", rng)
        _mp_case(cases, "dot", "mlp_k128", V((16, 128), dt), "C", V((128, 10), dt), "C", rng)
        _mp_case(cases, "dot", "mlp_k10", V((16, 10), dt), "C", V((10, 16), dt), "C", rng)
        _mp_case(cases, "dot", "mlp_xT", V((784, 12), dt), "F", V((12, 12), dt), "C", rng)
        _mp_case(cases, "dot", "mlp_hT", V((128, 12), dt), "F", V((12, 10), dt), "C", rng)
        _mp_case(cases, "matmul", "mlp_k784", V((8, 784), dt), "C", V((784, 8), dt), "C", rng)
        _mp_case(cases, "matmul", "mlp_k10", V((16, 10), dt), "C", V((10, 16), dt), "C", rng)

        # --- full layout matrix. The copy-if-not-blasable rule, the F-order transpose
        # equivalence and np.dot's own _bad_strides copy all key off these strides.
        A = V((12, 40), dt)
        B = V((40, 9), dt)
        for la in ("C", "F", "negrow", "negcol", "stride2", "slice"):
            for lb in ("C", "F", "negrow", "negcol", "stride2", "slice"):
                _mp_case(cases, "dot", "layout", A, la, B, lb, rng)
                _mp_case(cases, "matmul", "layout", A, la, B, lb, rng)

        # --- the four special-shape routes (dm==1 / dn==1 / dp==1). np.dot and
        # np.matmul genuinely disagree here when the matrix is not blasable, so both
        # are recorded.
        for op in ("dot", "matmul"):
            _mp_case(cases, op, "vecvec", V((500,), dt), "C", V((500,), dt), "C", rng)
            _mp_case(cases, op, "vecvec_neg", V((37,), dt), "neg", V((37,), dt), "C", rng)
            _mp_case(cases, op, "vecvec_str", V((37,), dt), "stride2", V((37,), dt), "C", rng)
            _mp_case(cases, op, "rowcol", V((1, 500), dt), "C", V((500, 1), dt), "C", rng)
            for lm in ("C", "F", "negrow", "stride2", "slice"):
                _mp_case(cases, op, "matvec", V((30, 44), dt), lm, V((44,), dt), "C", rng)
                _mp_case(cases, op, "vecmat", V((44,), dt), "C", V((44, 30), dt), lm, rng)
            _mp_case(cases, op, "matvec_strided_v", V((30, 44), dt), "C", V((44,), dt), "stride2", rng)
            _mp_case(cases, op, "colrow", V((11, 1), dt), "C", V((1, 9), dt), "C", rng)
            _mp_case(cases, op, "onerow", V((1, 1), dt), "C", V((1, 9), dt), "C", rng)
            _mp_case(cases, op, "colone", V((11, 1), dt), "C", V((1, 1), dt), "C", rng)
            _mp_case(cases, op, "matcol", V((13, 29), dt), "C", V((29, 1), dt), "C", rng)
            _mp_case(cases, op, "rowmat", V((1, 29), dt), "C", V((29, 13), dt), "C", rng)

        # --- syrk: `a @ a.T` shares a DATA POINTER, which both dispatchers shortcut to
        # cblas_?syrk (upper triangle + mirror) instead of gemm. The corpus descriptor
        # gives every operand its own buffer, so the self-product cannot be expressed as
        # two operands — the op name carries the transpose instead and OpRegistry forms
        # `a @ a.T` from the single stored operand, preserving the shared pointer.
        for suffix, fn in (("aat", lambda v: (v, v.T)), ("ata", lambda v: (v.T, v))):
            for lay in ("C", "F"):
                S = V((16, 24), dt)
                baseS, viewS = _mp_layout(S, lay, rng)
                lhs, rhs = fn(viewS)
                for op in ("dot", "matmul"):
                    r = np.asarray((np.dot if op == "dot" else np.matmul)(lhs, rhs))
                    cases.append({
                        "id": f"{op}_{suffix}/syrk/{lay}/{dt}/{len(cases)}",
                        "op": f"{op}_{suffix}",
                        "params": {},
                        "operands": [describe(baseS, viewS)],
                        "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                                     "buffer": np.ascontiguousarray(r).tobytes().hex()},
                        "layout": f"syrk_{lay}",
                        "valueclass": "normal",
                    })

        # --- stacked matmul (the gufunc's outer loop) + N-D dot (the dotfunc route,
        # which NumPy does NOT send to gemm).
        _mp_case(cases, "matmul", "batch3", V((3, 8, 20), dt), "C", V((3, 20, 6), dt), "C", rng)
        _mp_case(cases, "matmul", "batch4", V((2, 3, 5, 12), dt), "C", V((2, 3, 12, 4), dt), "C", rng)
        _mp_case(cases, "matmul", "batch_bcast", V((3, 8, 20), dt), "C", V((20, 6), dt), "C", rng)
        _mp_case(cases, "matmul", "batch_vec", V((3, 8, 20), dt), "C", V((20,), dt), "C", rng)
        _mp_case(cases, "dot", "nd_3d_1d", V((3, 8, 20), dt), "C", V((20,), dt), "C", rng)
        _mp_case(cases, "dot", "nd_3d_2d", V((3, 8, 20), dt), "C", V((20, 7), dt), "C", rng)
        _mp_case(cases, "dot", "nd_2d_3d", V((9, 20), dt), "C", V((4, 20, 5), dt), "C", rng)
        _mp_case(cases, "dot", "nd_3d_3d", V((2, 5, 20), dt), "C", V((3, 20, 4), dt), "C", rng)

        # --- degenerate extents.
        _mp_case(cases, "dot", "k0", V((5, 0), dt), "C", V((0, 3), dt), "C", rng)
        _mp_case(cases, "matmul", "k0", V((5, 0), dt), "C", V((0, 3), dt), "C", rng)
        _mp_case(cases, "dot", "m0", V((0, 3), dt), "C", V((3, 4), dt), "C", rng)
        _mp_case(cases, "dot", "n0", V((5, 3), dt), "C", V((3, 0), dt), "C", rng)

        # --- value classes that punish reassociation, plus inf/NaN propagation.
        _mp_case(cases, "dot", "wide_k300", V((6, 300), dt, "wide"), "C",
                 V((300, 5), dt, "wide"), "C", rng, "wide")
        _mp_case(cases, "dot", "specials", V((6, 40), dt, "specials"), "C",
                 V((40, 5), dt, "specials"), "C", rng, "specials")
        _mp_case(cases, "dot", "vecvec_wide", V((400,), dt, "wide"), "C",
                 V((400,), dt, "wide"), "C", rng, "wide")

        # --- the CBLAS product family beyond dot/matmul. Backend byte-parity for inner (dot on the
        # swapaxes'd operand), vdot (the CONJUGATING zdotc for complex), vecdot (per-inner dotc, was
        # Multiply+ReduceAdd), matvec (gemv / per-row dotu) and vecmat (complex gemm-ConjTrans /
        # real gemv, else per-col dotc). Deep K (44/300) so reassociation shows; the F/stride2
        # layouts and the dn==1 shapes drive the blasable-vs-portable route split; the 3-D operands
        # exercise the gufunc's broadcast-of-leading-axes outer loop.
        for kk in (44, 300):
            _mp_prod_case(cases, "inner", f"vec_k{kk}", V((kk,), dt), "C", V((kk,), dt), "C", rng)
            _mp_prod_case(cases, "vdot", f"vec_k{kk}", V((kk,), dt), "C", V((kk,), dt), "C", rng)
            _mp_prod_case(cases, "vecdot", f"k{kk}", V((kk,), dt), "C", V((kk,), dt), "C", rng)
        _mp_prod_case(cases, "inner", "mat", V((6, 120), dt), "C", V((5, 120), dt), "C", rng)
        _mp_prod_case(cases, "inner", "matF", V((6, 120), dt), "F", V((5, 120), dt), "C", rng)
        _mp_prod_case(cases, "vdot", "frob", V((8, 30), dt), "C", V((8, 30), dt), "C", rng)
        _mp_prod_case(cases, "vdot", "frobF", V((8, 30), dt), "F", V((8, 30), dt), "C", rng)
        for lm in ("C", "F", "stride2"):
            _mp_prod_case(cases, "matvec", f"mv_{lm}", V((30, 60), dt), lm, V((60,), dt), "C", rng)
            _mp_prod_case(cases, "vecmat", f"vm_{lm}", V((60,), dt), "C", V((60, 30), dt), lm, rng)
        # gufunc outer loop over broadcast leading axes (stacked, and vector-vs-batch broadcast).
        _mp_prod_case(cases, "vecdot", "batch", V((7, 90), dt), "C", V((7, 90), dt), "C", rng)
        _mp_prod_case(cases, "vecdot", "bcast", V((90,), dt), "C", V((7, 90), dt), "C", rng)
        _mp_prod_case(cases, "matvec", "batch", V((5, 30, 60), dt), "C", V((5, 60), dt), "C", rng)
        _mp_prod_case(cases, "matvec", "bcast", V((5, 30, 60), dt), "C", V((60,), dt), "C", rng)
        _mp_prod_case(cases, "vecmat", "batch", V((5, 60), dt), "C", V((5, 60, 30), dt), "C", rng)
        _mp_prod_case(cases, "vecmat", "bcast", V((60,), dt), "C", V((5, 60, 30), dt), "C", rng)
        # non-blasable core (dn==1 / dm==1) -> per-row/col dot instead of gemv/gemm.
        _mp_prod_case(cases, "matvec", "dn1", V((30, 1), dt), "C", V((1,), dt), "C", rng)
        _mp_prod_case(cases, "vecmat", "dn1", V((1,), dt), "C", V((1, 30), dt), "C", rng)
        # complex vecmat/vecdot with WIDE magnitudes — the conjugating path under reassociation stress.
        _mp_prod_case(cases, "vecdot", "wide", V((256,), dt, "wide"), "C",
                      V((256,), dt, "wide"), "C", rng, "wide")
        _mp_prod_case(cases, "vdot", "wide", V((256,), dt, "wide"), "C",
                      V((256,), dt, "wide"), "C", rng, "wide")

    # --- blocked / multi-threaded kernel sizes (f32 only for corpus weight; f64 smaller).
    _mp_case(cases, "dot", "big", _mp_values((64, 256), "float32", rng), "C",
             _mp_values((256, 64), "float32", rng), "C", rng)
    _mp_case(cases, "dot", "big", _mp_values((48, 192), "float64", rng), "C",
             _mp_values((192, 48), "float64", rng), "C", rng)

    # --- mixed dtype: NumPy casts to the common type first (a C-contiguous copy).
    _mp_case(cases, "dot", "mixed", _mp_values((12, 40), "float32", rng), "C",
             _mp_values((40, 9), "float64", rng), "C", rng)
    _mp_case(cases, "dot", "mixed", _mp_values((12, 40), "float64", rng), "C",
             _mp_values((40, 9), "float32", rng), "C", rng)
    return cases


def blas_identity():
    """Identify the BLAS NumPy will call, so the replay can refuse a mismatched host.

    The bits this tier records depend on the library build, the DYNAMIC_ARCH kernel it
    picks for this CPU, and the worker-thread count — all three are read straight out of
    the loaded binary through the same OpenBLAS entry points NumSharp's parity backend uses.
    """
    import ctypes
    import glob
    import hashlib
    import platform

    info = {"numpy": np.__version__, "platform": platform.platform(),
            "machine": platform.machine()}
    roots = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(np.__file__))), "numpy.libs")]
    patterns = ["*scipy_openblas*.dll", "*scipy_openblas*.so*", "*scipy_openblas*.dylib",
                "*openblas*.dll", "*openblas*.so*", "*openblas*.dylib"]
    lib = None
    for root in roots:
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(root, pat)))
            if hits:
                lib = hits[-1]
                break
        if lib:
            break
    if lib is None:
        info["blas_library"] = ""
        return info

    info["blas_library"] = os.path.basename(lib)
    # The library's CONTENT hash, which is what the claim is actually about. The file NAME is a
    # poor proxy for it in both directions: pip's delvewheel/auditwheel mangle the name per build
    # (numpy ships libscipy_openblas64_-<hash>.dll), while NumSharp's bundled copy of the very same
    # bytes is plainly libscipy_openblas64_.dll. Comparing names alone therefore excuses a host that
    # is genuinely bit-identical, and would accept a differently-built library that happened to be
    # named the same. The C# gate prefers this field and keeps the name as a fallback for corpora
    # generated before it existed.
    with open(lib, "rb") as fh:
        info["blas_library_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    try:
        dll = ctypes.CDLL(lib)
        for prefix, suffix in (("scipy_", "64_"), ("", "64_"), ("", "")):
            try:
                cfg = getattr(dll, f"{prefix}openblas_get_config{suffix}")
                core = getattr(dll, f"{prefix}openblas_get_corename{suffix}")
                thr = getattr(dll, f"{prefix}openblas_get_num_threads{suffix}")
            except AttributeError:
                continue
            cfg.restype = ctypes.c_char_p
            core.restype = ctypes.c_char_p
            thr.restype = ctypes.c_int
            info["blas_config"] = cfg().decode("ascii", "replace")
            info["blas_corename"] = core().decode("ascii", "replace")
            info["blas_threads"] = int(thr())
            break
    except OSError as e:
        info["blas_error"] = str(e)
    return info


# ---------------------------------------------------------------------------
# T-linalg — the LAPACK FACTORISATION family byte-parity for the opt-in BLAS
# backend (linalg_parity). SAME host-pinned model as matmul_parity: NumPy's
# np.linalg.{cholesky,eig,eigvals,eigh,eigvalsh,svd,svdvals,pinv,matrix_rank,
# cond,lstsq,qr} and np.linalg.norm{2,-2,'nuc'} all delegate to scipy-openblas
# LAPACK, whose result bits depend on the library build, the DYNAMIC_ARCH kernel
# and the worker-thread count. NumSharp.Core ships NO managed LU/QR/SVD/eigen
# solver, so these are computable ONLY through NumSharp.Interop.OpenBLAS — and
# byte-identical to NumPy exactly when the three levers agree.
#
# It is deliberately a SEPARATE tier from matmul_parity because it is pinned at
# threads=1 (the config the interop live-parity suite proves deterministic — see
# test/NumSharp.Tests.Interop/*LiveParityTests.cs) rather than the ambient max,
# and its results are TUPLES (svd/eig/eigh/qr/lstsq) as well as arrays.
#
# Only the BYTE-REPRODUCIBLE surface is recorded (empirically probed on this host,
# 25/26 byte-exact). Three factorisation outputs are NOT byte-reproducible and are
# deliberately EXCLUDED (covered instead by the interop suite's reconstruction /
# tolerance checks, and documented in Fuzz/README.md):
#   * complex-HERMITIAN eigh EIGENVECTORS — heevd does not canonicalize the phase
#     and it is not reproducible across processes; complex-Hermitian eigenVALUES
#     (eigvalsh, and eigh's [0] slot for REAL-symmetric input) are recorded.
#   * float32 eig with COMPLEX eigenvalues — NumPy yields complex64, NumSharp
#     complex128 (no complex64 dtype); float32 eig is recorded only for matrices
#     with all-REAL eigenvalues.
#   * cond/norm ORDERS that are not SVD-based (fro/1/-1/inf) — they compose an
#     elementwise reduction whose summation order rounds 1 ULP off NumPy; only the
#     SVD-based orders (cond None/2/-2, norm 2/-2/'nuc') are recorded.
LINALG_PARITY_DTYPES = ["float64", "complex128", "float32"]


def _set_openblas_threads(n):
    """Force the loaded scipy-openblas to <n> threads so the recorded bytes are a
    deterministic single-threaded function of the input, independent of the ambient
    OPENBLAS_NUM_THREADS. Mirrors blas_identity()'s ctypes probe; best-effort."""
    import ctypes
    import glob
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(np.__file__))), "numpy.libs")
    for pat in ("*scipy_openblas*.dll", "*scipy_openblas*.so*", "*scipy_openblas*.dylib",
                "*openblas*.dll", "*openblas*.so*", "*openblas*.dylib"):
        hits = sorted(glob.glob(os.path.join(root, pat)))
        if not hits:
            continue
        try:
            dll = ctypes.CDLL(hits[-1])
        except OSError:
            continue
        for name in ("scipy_openblas_set_num_threads64_", "openblas_set_num_threads64_",
                     "openblas_set_num_threads"):
            fn = getattr(dll, name, None)
            if fn is not None:
                fn(ctypes.c_int(n))
                return True
    return False


def _lp_operand(a, layout, rng):
    """(base, view) holding EXACTLY a's values in the requested memory layout, reusing
    the matmul_parity layout recipes. C/F work for any rank; the strided/reversed recipes
    are 2-D only (guarded by the caller)."""
    if layout == "C" or a.ndim == 0:
        base = np.ascontiguousarray(a)
        return base, base
    return _mp_layout(a, layout, rng)


def _lp_arr(cases, op, name, a, params, fn, dt, layout, rng):
    """Record ONE array-result factorisation case (cholesky/eigvals/eigvalsh/svdvals/
    pinv/matrix_rank/cond/norm, and svd(compute_uv=False)/qr(mode='r'))."""
    base, view = _lp_operand(a, layout, rng)
    r = np.asarray(fn(view))
    cases.append({
        "id": f"{op}/{name}/{dt}/{layout}/{len(cases)}",
        "op": op, "params": params,
        "operands": [describe(base, view)],
        "expected": _arr_expected(r),
        "layout": layout, "valueclass": "linalg",
    })


def _lp_tuple(cases, op, name, a, params, fn, dt, layout, rng):
    """Record ONE tuple-result factorisation case (svd/eig/eigh/qr) — every slot, so
    ARITY is asserted too."""
    base, view = _lp_operand(a, layout, rng)
    r = [np.asarray(x) for x in fn(view)]
    cases.append({
        "id": f"{op}/{name}/{dt}/{layout}/{len(cases)}",
        "op": op, "params": params,
        "operands": [describe(base, view)],
        "expected": _tuple_expected(r),
        "layout": layout, "valueclass": "linalg",
    })


def _lp_lstsq(cases, name, a, b, dt):
    """lstsq takes TWO operands and returns a 4-tuple (solution, residuals, rank, s)."""
    ba, va = np.ascontiguousarray(a), np.ascontiguousarray(a)
    bb, vb = np.ascontiguousarray(b), np.ascontiguousarray(b)
    r = [np.asarray(x) for x in np.linalg.lstsq(va, vb, rcond=None)]
    cases.append({
        "id": f"lstsq/{name}/{dt}/C/{len(cases)}",
        "op": "lstsq", "params": {"rcond": None},
        "operands": [describe(ba, va), describe(bb, vb)],
        "expected": _tuple_expected(r),
        "layout": "C", "valueclass": "linalg",
    })


def _lp_arr2(cases, op, name, a, b, params, fn, dt, layout, rng):
    """A TWO-operand array-result factorisation (solve/tensorsolve): `a` carried in the
    requested layout (reusing the matmul_parity recipes), `b` always C-contiguous. Mirrors
    _lp_lstsq's operand shape but for a single array result rather than the 4-tuple."""
    base_a, view_a = _lp_operand(a, layout, rng)
    base_b, view_b = np.ascontiguousarray(b), np.ascontiguousarray(b)
    r = np.asarray(fn(view_a, view_b))
    cases.append({
        "id": f"{op}/{name}/{dt}/{layout}/{len(cases)}",
        "op": op, "params": params,
        "operands": [describe(base_a, view_a), describe(base_b, view_b)],
        "expected": _arr_expected(r),
        "layout": layout, "valueclass": "linalg",
    })


def _lp_poly_fit(cases, x, y, deg, dt):
    """polyfit reaches lstsq (backend); default return is the coefficient array."""
    bx, vx = np.ascontiguousarray(x), np.ascontiguousarray(x)
    by, vy = np.ascontiguousarray(y), np.ascontiguousarray(y)
    r = np.asarray(np.polyfit(vx, vy, deg))
    cases.append({
        "id": f"polyfit/deg{deg}/{dt}/C/{len(cases)}",
        "op": "polyfit", "params": {"deg": deg},
        "operands": [describe(bx, vx), describe(by, vy)],
        "expected": _arr_expected(r),
        "layout": "C", "valueclass": "linalg",
    })


def gen_linalg_parity():
    # Deterministic single-thread bytes regardless of ambient OPENBLAS_NUM_THREADS.
    _set_openblas_threads(1)
    cases = []
    rng = np.random.RandomState(20260821)

    # ---- reference operands (integer-valued so every dtype cast is exact) -----------------
    SPD3 = np.array([[4., 2, 1], [2, 5, 3], [1, 3, 6]])                       # symmetric PD
    SPD4 = np.array([[4., 1, 0, 1], [1, 5, 2, 0], [0, 2, 6, 1], [1, 0, 1, 7]])
    HPD2 = np.array([[2 + 0j, 1 - 1j], [1 + 1j, 3 + 0j]])                     # Hermitian PD
    HPD3 = np.array([[3 + 0j, 1 - 1j, 0], [1 + 1j, 4 + 0j, 0 - 1j], [0, 0 + 1j, 5 + 0j]])
    HERM_NPD = np.array([[1 + 0j, 0 - 2j], [0 + 2j, 5 + 0j]])                 # Hermitian, not used for chol
    SYM_NPD = np.array([[0., 1, 2], [1, 3, 1], [2, 1, 0]])                    # symmetric, indefinite
    REAL_EIG = np.array([[2., 0, 0], [1, 3, 0], [4, 5, 6]])                   # non-sym, all-real eigs
    CPLX_EIG = np.array([[1., -1], [1, 1]])                                   # non-sym, complex eigs
    CPLX_EIG3 = np.array([[1., 2, 0], [0, 3, 1], [2, 0, 4]])                  # non-sym, complex eigs
    CPLX_IN = np.array([[1 + 0j, 2 - 1j], [0 + 1j, 3 + 0j]])                  # complex input (zgeev)
    TALL = np.array([[1., 2, 3], [4, 5, 6], [7, 8, 10], [1, 0, 2]])          # 4x3
    WIDE = np.array([[1., 2, 3, 4], [5, 6, 7, 8], [9, 10, 12, 11]])          # 3x4
    SQR = np.array([[4., 1, 2], [0, 3, 1], [1, 0, 5]])                        # 3x3 non-symmetric
    CTALL = np.array([[1 + 2j, 3 - 1j], [4 + 0j, 1 + 1j], [-2 + 1j, 0 + 3j], [1 + 0j, 2 - 2j]])  # 4x2
    RANKDEF = np.array([[1., 2, 3], [2, 4, 6], [1, 1, 1]])                    # rank 2

    def cast(a, dt):
        return np.ascontiguousarray(a).astype(dt)

    # ---- layout sweeps ---------------------------------------------------------------------
    LAYOUTS_2D = ["C", "F", "negrow", "negcol", "stride2", "slice"]

    # ==========================  cholesky  ==============================================
    for dt in LINALG_PARITY_DTYPES:
        srcs = [("spd3", SPD3), ("spd4", SPD4)] + \
               ([("hpd2", HPD2), ("hpd3", HPD3)] if dt == "complex128" else [])
        for nm, a in srcs:
            for upper in (False, True):
                _lp_arr(cases, "cholesky", f"{nm}_u{int(upper)}", cast(a, dt),
                        {"upper": upper}, lambda v, u=upper: np.linalg.cholesky(v, upper=u), dt, "C", rng)
    # integer / bool widen to float64
    for dt in ("int32", "int64", "uint8", "bool"):
        _lp_arr(cases, "cholesky", "spd3_widen", cast(np.array([[4., 0, 0], [0, 9, 0], [0, 0, 16]]), dt),
                {"upper": False}, lambda v: np.linalg.cholesky(v), dt, "C", rng)
    # layouts (float64) + batched + degenerate
    for lay in LAYOUTS_2D:
        _lp_arr(cases, "cholesky", "spd3_lay", SPD3, {"upper": False},
                lambda v: np.linalg.cholesky(v), "float64", lay, rng)
    _lp_arr(cases, "cholesky", "batch", np.stack([SPD3, SPD3 * 2.0, SPD3 + np.eye(3)]),
            {"upper": False}, lambda v: np.linalg.cholesky(v), "float64", "C", rng)
    _lp_arr(cases, "cholesky", "1x1", np.array([[4.0]]), {"upper": False},
            lambda v: np.linalg.cholesky(v), "float64", "C", rng)
    _lp_arr(cases, "cholesky", "empty00", np.zeros((0, 0)), {"upper": False},
            lambda v: np.linalg.cholesky(v), "float64", "C", rng)

    # ==========================  eigvalsh / eigh  =======================================
    for dt in LINALG_PARITY_DTYPES:
        for nm, a in [("spd3", SPD3), ("sym_npd", SYM_NPD)]:
            for uplo in ("L", "U"):
                _lp_arr(cases, "eigvalsh", f"{nm}_{uplo}", cast(a, dt), {"UPLO": uplo},
                        lambda v, u=uplo: np.linalg.eigvalsh(v, UPLO=u), dt, "C", rng)
                # eigh: REAL-symmetric eigenVECTORS are reproducible -> tuple (both slots).
                _lp_tuple(cases, "eigh", f"{nm}_{uplo}", cast(a, dt), {"UPLO": uplo},
                          lambda v, u=uplo: np.linalg.eigh(v, UPLO=u), dt, "C", rng)
    # complex-Hermitian: eigenVALUES only (eigenvectors' heevd phase is not reproducible).
    for nm, a in [("hpd2", HPD2), ("hpd3", HPD3), ("herm_npd", HERM_NPD)]:
        for uplo in ("L", "U"):
            _lp_arr(cases, "eigvalsh", f"{nm}_{uplo}", np.ascontiguousarray(a), {"UPLO": uplo},
                    lambda v, u=uplo: np.linalg.eigvalsh(v, UPLO=u), "complex128", "C", rng)
    # integer / bool widen; batched; layouts (real symmetric)
    for dt in ("int32", "int64", "bool"):
        _lp_arr(cases, "eigvalsh", "diag_widen", cast(np.diag([4., 9, 16]), dt), {"UPLO": "L"},
                lambda v: np.linalg.eigvalsh(v), dt, "C", rng)
    _lp_arr(cases, "eigvalsh", "batch", np.stack([SPD3, SPD3 * 2.0, SPD3 + np.eye(3)]), {"UPLO": "L"},
            lambda v: np.linalg.eigvalsh(v), "float64", "C", rng)
    _lp_tuple(cases, "eigh", "batch", np.stack([SPD3, SPD3 * 2.0]), {"UPLO": "L"},
              lambda v: np.linalg.eigh(v), "float64", "C", rng)
    for lay in ("F", "negrow", "negcol"):
        _lp_arr(cases, "eigvalsh", "spd3_lay", SPD3, {"UPLO": "L"},
                lambda v: np.linalg.eigvalsh(v), "float64", lay, rng)

    # ==========================  eig / eigvals  =========================================
    # float64 + complex128 over real-eig / complex-eig / complex-input matrices.
    for dt in ("float64", "complex128"):
        srcs = [("realeig", REAL_EIG), ("cplxeig", CPLX_EIG), ("cplxeig3", CPLX_EIG3)]
        if dt == "complex128":
            srcs += [("cplxin", CPLX_IN)]
        for nm, a in srcs:
            _lp_arr(cases, "eigvals", f"{nm}", cast(a, dt), {}, lambda v: np.linalg.eigvals(v), dt, "C", rng)
            _lp_tuple(cases, "eig", f"{nm}", cast(a, dt), {}, lambda v: np.linalg.eig(v), dt, "C", rng)
    # float32: only REAL-eig matrices (complex-eig float32 -> complex64, a dtype divergence).
    _lp_arr(cases, "eigvals", "realeig", cast(REAL_EIG, "float32"), {}, lambda v: np.linalg.eigvals(v), "float32", "C", rng)
    _lp_tuple(cases, "eig", "realeig", cast(REAL_EIG, "float32"), {}, lambda v: np.linalg.eig(v), "float32", "C", rng)
    # batched: one complex-eig + one real-eig -> whole result complex on both sides.
    _lp_tuple(cases, "eig", "batch", np.stack([CPLX_EIG, np.array([[2., 0], [0, 3]])]), {},
              lambda v: np.linalg.eig(v), "float64", "C", rng)
    _lp_arr(cases, "eigvals", "batch", np.stack([REAL_EIG, REAL_EIG + np.eye(3)]), {},
            lambda v: np.linalg.eigvals(v), "float64", "C", rng)

    # ==========================  svd / svdvals  =========================================
    for dt in LINALG_PARITY_DTYPES:
        rects = [("tall", TALL), ("wide", WIDE), ("sqr", SQR)] + \
                ([("ctall", CTALL)] if dt == "complex128" else [])
        for nm, a in rects:
            for full in (False, True):
                _lp_tuple(cases, "svd", f"{nm}_f{int(full)}", cast(a, dt), {"full_matrices": full},
                          lambda v, f=full: np.linalg.svd(v, full_matrices=f), dt, "C", rng)
            # compute_uv=False -> just S (array kind)
            _lp_arr(cases, "svd", f"{nm}_novec", cast(a, dt), {"full_matrices": False, "compute_uv": False},
                    lambda v: np.linalg.svd(v, compute_uv=False), dt, "C", rng)
            _lp_arr(cases, "svdvals", f"{nm}", cast(a, dt), {}, lambda v: np.linalg.svdvals(v), dt, "C", rng)
    # layouts (float64 tall) + batched + degenerate
    for lay in LAYOUTS_2D:
        _lp_tuple(cases, "svd", "tall_lay", TALL, {"full_matrices": False},
                  lambda v: np.linalg.svd(v, full_matrices=False), "float64", lay, rng)
        _lp_arr(cases, "svdvals", "tall_lay", TALL, {}, lambda v: np.linalg.svdvals(v), "float64", lay, rng)
    _lp_tuple(cases, "svd", "batch", np.stack([TALL, TALL + 1.0, TALL * 2.0]), {"full_matrices": False},
              lambda v: np.linalg.svd(v, full_matrices=False), "float64", "C", rng)
    _lp_arr(cases, "svdvals", "batch", np.stack([TALL, TALL + 1.0]), {},
            lambda v: np.linalg.svdvals(v), "float64", "C", rng)
    _lp_tuple(cases, "svd", "1x1", np.array([[7.0]]), {"full_matrices": True},
              lambda v: np.linalg.svd(v, full_matrices=True), "float64", "C", rng)
    _lp_tuple(cases, "svd", "empty03", np.zeros((0, 3)), {"full_matrices": True},
              lambda v: np.linalg.svd(v, full_matrices=True), "float64", "C", rng)
    _lp_tuple(cases, "svd", "empty30", np.zeros((3, 0)), {"full_matrices": True},
              lambda v: np.linalg.svd(v, full_matrices=True), "float64", "C", rng)

    # ==========================  pinv  ==================================================
    for dt in LINALG_PARITY_DTYPES:
        rects = [("tall", TALL), ("wide", WIDE), ("sqr", SQR)] + \
                ([("ctall", CTALL)] if dt == "complex128" else [])
        for nm, a in rects:
            _lp_arr(cases, "pinv", f"{nm}", cast(a, dt), {}, lambda v: np.linalg.pinv(v), dt, "C", rng)
    _lp_arr(cases, "pinv", "tall_rcond", TALL, {"rcond": 1e-10},
            lambda v: np.linalg.pinv(v, rcond=1e-10), "float64", "C", rng)
    for lay in LAYOUTS_2D:
        _lp_arr(cases, "pinv", "tall_lay", TALL, {}, lambda v: np.linalg.pinv(v), "float64", lay, rng)
    _lp_arr(cases, "pinv", "batch", np.stack([TALL, TALL + 1.0]), {},
            lambda v: np.linalg.pinv(v), "float64", "C", rng)

    # ==========================  matrix_rank  ===========================================
    RANK_FULL = np.array([[1., 2, 3], [4, 5, 6], [7, 8, 10]])
    for dt in ("float64", "float32", "complex128", "int64"):
        _lp_arr(cases, "matrix_rank", "full", cast(RANK_FULL, dt), {},
                lambda v: np.linalg.matrix_rank(v), dt, "C", rng)
    _lp_arr(cases, "matrix_rank", "def", RANKDEF, {}, lambda v: np.linalg.matrix_rank(v), "float64", "C", rng)
    _lp_arr(cases, "matrix_rank", "def_tol", RANKDEF, {"tol": 0.5},
            lambda v: np.linalg.matrix_rank(v, tol=0.5), "float64", "C", rng)
    _lp_arr(cases, "matrix_rank", "def_rtol", RANKDEF, {"rtol": 0.1},
            lambda v: np.linalg.matrix_rank(v, rtol=0.1), "float64", "C", rng)
    _lp_arr(cases, "matrix_rank", "batch",
            np.stack([RANK_FULL, np.eye(3), RANKDEF]), {},
            lambda v: np.linalg.matrix_rank(v), "float64", "C", rng)

    # ==========================  cond (SVD-based orders None/2/-2)  ======================
    for dt in ("float64", "complex128"):
        a = SPD3 if dt == "float64" else HPD3
        for pk, pv, pf in [("none", None, None), ("2", 2, 2), ("neg2", -2, -2)]:
            params = {} if pv is None else {"p": pv}
            _lp_arr(cases, "cond", f"{dt[:4]}_{pk}", cast(a, dt), params,
                    lambda v, pp=pf: np.linalg.cond(v) if pp is None else np.linalg.cond(v, pp), dt, "C", rng)
    _lp_arr(cases, "cond", "batch", np.stack([SPD3, SPD3 + np.eye(3)]), {},
            lambda v: np.linalg.cond(v), "float64", "C", rng)

    # ==========================  norm (matrix orders 2/-2/'nuc')  =======================
    for dt in ("float64", "complex128"):
        rects = [("tall", TALL), ("sqr", SQR)] + ([("ctall", CTALL)] if dt == "complex128" else [])
        for nm, a in rects:
            for ok, ov in [("2", 2), ("neg2", -2), ("nuc", "nuc")]:
                _lp_arr(cases, "norm", f"{nm}_{ok}", cast(a, dt), {"ord": ov},
                        lambda v, o=ov: np.linalg.norm(v, o), dt, "C", rng)
    for lay in LAYOUTS_2D:
        _lp_arr(cases, "norm", "tall_2_lay", TALL, {"ord": 2}, lambda v: np.linalg.norm(v, 2), "float64", lay, rng)
        _lp_arr(cases, "norm", "tall_nuc_lay", TALL, {"ord": "nuc"}, lambda v: np.linalg.norm(v, "nuc"), "float64", lay, rng)
    # stacked over an axis tuple + keepdims
    STACK234 = np.arange(24.0).reshape(2, 3, 4)
    _lp_arr(cases, "norm", "stack_2_ax", STACK234, {"ord": 2, "axis": [1, 2]},
            lambda v: np.linalg.norm(v, 2, axis=(1, 2)), "float64", "C", rng)
    _lp_arr(cases, "norm", "stack_nuc_ax", STACK234, {"ord": "nuc", "axis": [1, 2]},
            lambda v: np.linalg.norm(v, "nuc", axis=(1, 2)), "float64", "C", rng)
    _lp_arr(cases, "norm", "stack_nuc_kd", STACK234, {"ord": "nuc", "axis": [1, 2], "keepdims": True},
            lambda v: np.linalg.norm(v, "nuc", axis=(1, 2), keepdims=True), "float64", "C", rng)

    # ==========================  qr (reduced/complete/r/raw)  ===========================
    for dt in LINALG_PARITY_DTYPES:
        rects = [("tall", TALL), ("wide", WIDE), ("sqr", SQR)] + \
                ([("ctall", CTALL)] if dt == "complex128" else [])
        for nm, a in rects:
            for mode in ("reduced", "complete", "raw"):
                _lp_tuple(cases, "qr", f"{nm}_{mode}", cast(a, dt), {"mode": mode},
                          lambda v, m=mode: np.linalg.qr(v, mode=m), dt, "C", rng)
            # r-mode returns just R (array kind)
            _lp_arr(cases, "qr", f"{nm}_r", cast(a, dt), {"mode": "r"},
                    lambda v: np.linalg.qr(v, mode="r"), dt, "C", rng)
    for dt in ("int64", "bool"):
        _lp_tuple(cases, "qr", "tall_widen", cast(TALL, dt), {"mode": "reduced"},
                  lambda v: np.linalg.qr(v, mode="reduced"), dt, "C", rng)
    for lay in LAYOUTS_2D:
        _lp_tuple(cases, "qr", "tall_lay", TALL, {"mode": "reduced"},
                  lambda v: np.linalg.qr(v, mode="reduced"), "float64", lay, rng)
    _lp_tuple(cases, "qr", "batch", np.arange(3 * 5 * 3.0).reshape(3, 5, 3) + np.eye(5, 3), {"mode": "reduced"},
              lambda v: np.linalg.qr(v, mode="reduced"), "float64", "C", rng)
    _lp_tuple(cases, "qr", "30_reduced", np.zeros((3, 0)), {"mode": "reduced"},
              lambda v: np.linalg.qr(v, mode="reduced"), "float64", "C", rng)
    _lp_tuple(cases, "qr", "1x1_complete", np.array([[5.0]]), {"mode": "complete"},
              lambda v: np.linalg.qr(v, mode="complete"), "float64", "C", rng)

    # ==========================  lstsq  =================================================
    OVER = np.array([[0., 1], [1, 1], [2, 1], [3, 1]])          # 4x2 overdetermined
    Y1 = np.array([-1., 0.2, 0.9, 2.1])                          # 1-D b
    Y2 = np.array([[-1., -2], [0.2, 0.4], [0.9, 1.8], [2.1, 4.2]])  # 2-D b
    UNDER = np.array([[1., 2, 3], [4, 5, 6]])                    # 2x3 underdetermined
    SQ2 = np.array([[1., 2], [3, 5]])                            # 2x2 square
    RD = np.array([[1., 2], [2, 4], [3, 6]])                     # 3x2 rank-deficient
    CA = np.array([[1 + 0j, 1 + 0j], [1 + 0j, 0 + 1j], [1 + 0j, 2 + 0j], [0 + 0j, 1 + 0j]])
    CB = np.array([1 + 1j, 2 + 0j, 3 + 0j, 0 + 1j])
    _lp_lstsq(cases, "over_y1", OVER, Y1, "float64")
    _lp_lstsq(cases, "over_y2", OVER, Y2, "float64")
    _lp_lstsq(cases, "under", UNDER, np.array([1., 2]), "float64")
    _lp_lstsq(cases, "square", SQ2, np.array([1., 2]), "float64")
    _lp_lstsq(cases, "rankdef", RD, np.array([1., 2, 3]), "float64")
    _lp_lstsq(cases, "complex", CA, CB, "complex128")
    for dt in ("float32", "int32"):
        _lp_lstsq(cases, f"over_{dt}", OVER.astype(dt) if dt != "int32" else OVER.astype("int32"),
                  Y1.astype(dt) if dt != "int32" else Y1.astype("int32"), dt)

    # ==========================  polynomial backend ops  ================================
    # roots (eigenvalues of the companion matrix), polyfit (lstsq) and poly of a 2-D matrix
    # (char poly via eigvals) all reach the LAPACK seam and THROW without the backend, so they
    # ride this host-pinned tier rather than the portable poly.jsonl. Small operands, threads=1.
    for dt in ("float64", "float32", "complex128"):
        rp = np.array([1, -6, 11, -6]).astype(dt)          # roots 1,2,3
        _lp_arr(cases, "roots", "cubic", rp, {}, lambda v: np.roots(v), dt, "C", rng)
    _lp_arr(cases, "roots", "quad_complex", np.array([1., 0, 1]), {},   # roots ±i
            lambda v: np.roots(v), "float64", "C", rng)
    _lp_arr(cases, "roots", "leading_zeros", np.array([0., 0, 1, -3, 2]), {},
            lambda v: np.roots(v), "float64", "C", rng)
    # poly of a 2-D matrix -> characteristic polynomial (eigvals path)
    for dt in ("float64", "complex128"):
        Mp = np.array([[1, 2], [3, 4]]).astype(dt)
        _lp_arr(cases, "poly", "mat2", Mp, {}, lambda v: np.poly(v), dt, "C", rng)
    _lp_arr(cases, "poly", "mat3", np.array([[2., 0, 0], [1, 3, 0], [4, 5, 6]]), {},
            lambda v: np.poly(v), "float64", "C", rng)
    # polyfit -> least-squares coefficients (lstsq path); default return is the coeff array.
    xf = np.array([0., 1, 2, 3, 4])
    yf = np.array([1., 3, 2, 5, 4])
    for deg in (1, 2, 3):
        _lp_poly_fit(cases, xf, yf, deg, "float64")
    _lp_poly_fit(cases, xf.astype(np.float32), yf.astype(np.float32), 2, "float32")

    # ==========================  LU-factorisation family  ===============================
    # solve/inv/det/slogdet/tensorinv/tensorsolve all reach getrf/gesv. Unlike the eigen/SVD
    # factorisations there is NO sign- or phase-ambiguity: LU with partial pivoting is a
    # deterministic function of the input, so EVERY output is byte-reproducible (probed 2/2
    # per case, cross-process). float32 upcasts to double and rounds back once (_commonType),
    # so it is byte-identical too; int/bool widen to float64. These are the OpenBLAS-dependent
    # linalg members with no managed fallback and, until now, no committed differential-fuzz
    # gate at all.
    INV2 = np.array([[2., 1], [1, 3]])                         # 2x2 invertible
    DINV = np.diag([2., 3, 4])                                 # diagonal invertible (int/bool widen -> chol/eye)
    B1 = np.array([1., 2, 3])                                  # solve RHS: a vector (b.ndim == 1)
    B2 = np.array([[1., 4], [2, 5], [3, 6]])                  # solve RHS: a 3x2 stack of columns
    CB = np.array([1 + 1j, 2 + 0j])                            # complex vector RHS
    CB2 = np.array([[1 + 1j, 0], [2 + 0j, 1 - 1j]])           # complex matrix RHS
    BASE6 = np.array([[10, 1, 0, 2, 0, 1], [0, 11, 1, 0, 1, 0], [1, 0, 12, 0, 0, 1],
                      [0, 2, 0, 13, 1, 0], [1, 0, 0, 1, 14, 0], [0, 1, 0, 0, 1, 15]], float)  # invertible 6x6

    # ---- inv ----
    for dt in LINALG_PARITY_DTYPES:
        srcs = [("sqr", SQR), ("inv2", INV2)] + ([("cplxin", CPLX_IN)] if dt == "complex128" else [])
        for nm, a in srcs:
            _lp_arr(cases, "inv", nm, cast(a, dt), {}, lambda v: np.linalg.inv(v), dt, "C", rng)
    for dt in ("int32", "int64", "uint8", "bool"):
        _lp_arr(cases, "inv", "diag_widen", cast(DINV, dt), {}, lambda v: np.linalg.inv(v), dt, "C", rng)
    for lay in LAYOUTS_2D:
        _lp_arr(cases, "inv", "sqr_lay", SQR, {}, lambda v: np.linalg.inv(v), "float64", lay, rng)
    _lp_arr(cases, "inv", "batch", np.stack([SQR, SQR + np.eye(3)]), {},
            lambda v: np.linalg.inv(v), "float64", "C", rng)
    _lp_arr(cases, "inv", "1x1", np.array([[4.0]]), {}, lambda v: np.linalg.inv(v), "float64", "C", rng)

    # ---- det (single -> 0-D scalar; a stack -> 1-D; a singular operand -> exactly the LU product) ----
    for dt in LINALG_PARITY_DTYPES:
        srcs = [("sqr", SQR), ("inv2", INV2)] + ([("cplxin", CPLX_IN)] if dt == "complex128" else [])
        for nm, a in srcs:
            _lp_arr(cases, "det", nm, cast(a, dt), {}, lambda v: np.linalg.det(v), dt, "C", rng)
    for dt in ("int32", "int64", "bool"):
        _lp_arr(cases, "det", "diag_widen", cast(DINV, dt), {}, lambda v: np.linalg.det(v), dt, "C", rng)
    _lp_arr(cases, "det", "singular", RANKDEF, {}, lambda v: np.linalg.det(v), "float64", "C", rng)
    for lay in LAYOUTS_2D:
        _lp_arr(cases, "det", "sqr_lay", SQR, {}, lambda v: np.linalg.det(v), "float64", lay, rng)
    _lp_arr(cases, "det", "batch", np.stack([SQR, SQR + np.eye(3), RANKDEF]), {},
            lambda v: np.linalg.det(v), "float64", "C", rng)
    _lp_arr(cases, "det", "1x1", np.array([[7.0]]), {}, lambda v: np.linalg.det(v), "float64", "C", rng)

    # ---- slogdet (tuple: sign, logabsdet; a complex sign is a unit-modulus complex, not +-1;
    #               a singular operand gives (0, -inf)) ----
    for dt in LINALG_PARITY_DTYPES:
        srcs = [("sqr", SQR)] + ([("cplxin", CPLX_IN)] if dt == "complex128" else [])
        for nm, a in srcs:
            _lp_tuple(cases, "slogdet", nm, cast(a, dt), {}, lambda v: np.linalg.slogdet(v), dt, "C", rng)
    for dt in ("int32", "int64", "bool"):
        _lp_tuple(cases, "slogdet", "diag_widen", cast(DINV, dt), {}, lambda v: np.linalg.slogdet(v), dt, "C", rng)
    _lp_tuple(cases, "slogdet", "singular", RANKDEF, {}, lambda v: np.linalg.slogdet(v), "float64", "C", rng)
    _lp_tuple(cases, "slogdet", "batch", np.stack([SQR, RANKDEF]), {},
              lambda v: np.linalg.slogdet(v), "float64", "C", rng)
    _lp_tuple(cases, "slogdet", "1x1", np.array([[7.0]]), {}, lambda v: np.linalg.slogdet(v), "float64", "C", rng)

    # ---- solve (b is a stack of VECTORS iff EXACTLY 1-D, else a stack of MATRICES — NumPy 2.0) ----
    for dt in ("float64", "float32"):
        _lp_arr2(cases, "solve", "sqr_vec", cast(SQR, dt), cast(B1, dt), {},
                 lambda va, vb: np.linalg.solve(va, vb), dt, "C", rng)
        _lp_arr2(cases, "solve", "sqr_mat", cast(SQR, dt), cast(B2, dt), {},
                 lambda va, vb: np.linalg.solve(va, vb), dt, "C", rng)
    _lp_arr2(cases, "solve", "cplx_vec", CPLX_IN, CB, {},
             lambda va, vb: np.linalg.solve(va, vb), "complex128", "C", rng)
    _lp_arr2(cases, "solve", "cplx_mat", CPLX_IN, CB2, {},
             lambda va, vb: np.linalg.solve(va, vb), "complex128", "C", rng)
    _lp_arr2(cases, "solve", "int_widen", cast(SQR, "int64"), cast(B1, "int64"), {},
             lambda va, vb: np.linalg.solve(va, vb), "int64", "C", rng)
    for lay in LAYOUTS_2D:
        _lp_arr2(cases, "solve", "sqr_lay", SQR, B1, {},
                 lambda va, vb: np.linalg.solve(va, vb), "float64", lay, rng)
    _lp_arr2(cases, "solve", "batch_mat", np.stack([SQR, SQR + np.eye(3)]), np.stack([B2, B2 + 1.0]), {},
             lambda va, vb: np.linalg.solve(va, vb), "float64", "C", rng)
    _lp_arr2(cases, "solve", "bcast_vec", np.stack([SQR, SQR + np.eye(3)]), B1, {},
             lambda va, vb: np.linalg.solve(va, vb), "float64", "C", rng)
    _lp_arr2(cases, "solve", "1x1", np.array([[4.0]]), np.array([2.0]), {},
             lambda va, vb: np.linalg.solve(va, vb), "float64", "C", rng)

    # ---- tensorinv (reshape -> inv -> reshape; `ind` LEADING axes form the row side) ----
    for dt in LINALG_PARITY_DTYPES:
        _lp_arr(cases, "tensorinv", "ind2", cast(BASE6.reshape(2, 3, 3, 2), dt), {"ind": 2},
                lambda v: np.linalg.tensorinv(v, ind=2), dt, "C", rng)
        _lp_arr(cases, "tensorinv", "ind1", cast(BASE6.reshape(6, 2, 3), dt), {"ind": 1},
                lambda v: np.linalg.tensorinv(v, ind=1), dt, "C", rng)

    # ---- tensorsolve (reshape -> solve -> reshape; `axes` move to the rightmost, in order) ----
    TB = np.arange(1.0, 7.0).reshape(2, 3)
    for dt in LINALG_PARITY_DTYPES:
        _lp_arr2(cases, "tensorsolve", "std", cast(BASE6.reshape(2, 3, 6), dt), cast(TB, dt), {},
                 lambda va, vb: np.linalg.tensorsolve(va, vb), dt, "C", rng)
    _lp_arr2(cases, "tensorsolve", "axes", BASE6.reshape(6, 2, 3), TB, {"axes": [0]},
             lambda va, vb: np.linalg.tensorsolve(va, vb, axes=[0]), "float64", "C", rng)

    # ---- matrix_power with NEGATIVE n (= inv(a) ** |n| -> LAPACK gesv). Positive/zero n is a
    #      portable managed product and lives in products.jsonl; negative n THROWS without the
    #      backend, so its byte gate belongs here next to inv. int/bool widen to float64. ----
    for dt in ("float64", "float32", "complex128"):
        src = CPLX_IN if dt == "complex128" else SQR
        for n in (-1, -2, -3):
            _lp_arr(cases, "matrix_power", f"n{n}", cast(src, dt), {"n": n},
                    lambda v, k=n: np.linalg.matrix_power(v, k), dt, "C", rng)
    _lp_arr(cases, "matrix_power", "int_widen", cast(SQR, "int64"), {"n": -2},
            lambda v: np.linalg.matrix_power(v, -2), "int64", "C", rng)
    for lay in LAYOUTS_2D:
        _lp_arr(cases, "matrix_power", "sqr_lay", SQR, {"n": -1},
                lambda v: np.linalg.matrix_power(v, -1), "float64", lay, rng)

    return cases
# =====================================================================================
# Result KINDS and ERROR parity.
#
# The corpus could originally express exactly ONE comparable thing: a single array, checked
# as (dtype, shape, C-contiguous bytes). Three classes of op fell outside that shape and so
# outside the gate entirely — tuple-returning, dtype/scalar-returning, and text-returning —
# and every raising case was reduced to "NumSharp threw something".
#
#   expected.kind  : array (default) | scalar | dtype | text | tuple
#   error          : {"type": <python class>, "text": str(e)}   — NumPy's exception, verbatim
#
# The generators below emit those kinds. They write their own tier files rather than
# interleaving rows into the existing ones: the value tiers are large and shared, and
# rewriting 87K committed lines to add error rows would bury the change in churn.
# =====================================================================================


def _exc(e):
    """NumPy's exception recorded verbatim — the Python class name and str(e)."""
    return {"type": type(e).__name__, "text": str(e)}


def _arr_expected(r, kind=None):
    """(dtype, shape, bytes) for one array result — the historical `expected` shape."""
    r = np.asarray(r)
    # Shape BEFORE ascontiguousarray, which forces ndim>=1 and would corrupt a 0-D result.
    exp = {"dtype": r.dtype.name,
           "shape": [int(d) for d in r.shape],
           "buffer": np.ascontiguousarray(r).tobytes().hex()}
    if kind:
        exp["kind"] = kind
    return exp


def _tuple_expected(arrays):
    """kind=tuple — every slot recorded, so ARITY is asserted as well as the values."""
    return {"kind": "tuple", "slots": [_arr_expected(a) for a in arrays]}


def _case(op, params, operands, expected, layout, valueclass="mixed", cid=None):
    c = {"id": cid, "op": op, "params": params, "operands": operands,
         "expected": expected, "layout": layout, "valueclass": valueclass}
    return c


# ---- multi-output public surface ----------------------------------------------------

def gen_multioutput():
    """Public APIs whose observable contract is a tuple/list of arrays.

    Older tiers often selected one result with a ``piece`` parameter. That proves the chosen
    values but cannot detect a wrong ARITY or a broken sibling result. This tier records every
    slot and lets the tuple comparator assert arity, dtype, shape and bytes for all of them.
    """
    cases = []
    n = 0

    def emit_tuple(opname, params, arrays, result, layout="multioutput"):
        nonlocal n
        operands = [describe(a, a) for a in arrays]
        cases.append(_case(opname, params, operands, _tuple_expected(list(result)), layout, "tuple",
                           cid=f"{opname}/{layout}/{n}"))
        n += 1

    def emit_array(opname, params, array, result, layout="multioutput"):
        nonlocal n
        cases.append(_case(opname, params, [describe(array, array)], _arr_expected(result),
                           layout, "array", cid=f"{opname}/{layout}/{n}"))
        n += 1

    for dt in ["int32", "float64", "uint8", "complex128"]:
        d = np.dtype(dt)

        a6 = _cbase((6,), d)
        emit_tuple("split", {"sections": 3, "axis": 0}, [a6], np.split(a6, 3), f"split/{dt}")

        a7 = _cbase((7,), d)
        emit_tuple("array_split", {"sections": 3, "axis": 0}, [a7], np.array_split(a7, 3),
                   f"array_split/{dt}")

        h = _cbase((3, 4), d)
        emit_tuple("hsplit", {"sections": 2}, [h], np.hsplit(h, 2), f"hsplit/{dt}")

        v = _cbase((4, 3), d)
        emit_tuple("vsplit", {"sections": 2}, [v], np.vsplit(v, 2), f"vsplit/{dt}")

        dd = _cbase((2, 3, 4), d)
        emit_tuple("dsplit", {"sections": 2}, [dd], np.dsplit(dd, 2), f"dsplit/{dt}")
        emit_tuple("unstack", {"axis": 0}, [dd], np.unstack(dd, axis=0), f"unstack0/{dt}")
        emit_tuple("unstack", {"axis": -1}, [dd], np.unstack(dd, axis=-1), f"unstack-1/{dt}")

        # Array-API unique_* family. inverse_indices must retain the ORIGINAL 2-D input shape;
        # all index/count outputs are intp. unique_values(integer/complex, sorted=False) has a
        # deliberately platform-specific hash order in NumPy, so only its portable float path is
        # byte-gated here; the other three functions are sorted and portable for every dtype.
        if d.kind == "c":
            u = np.array([[2 + 1j, 1 + 0j, 2 + 1j], [0 + 0j, 1 + 0j, 3 - 1j]], dtype=d)
        else:
            u = np.array([[2, 1, 2], [0, 1, 3]], dtype=d)
        emit_tuple("unique_counts", {}, [u], np.unique_counts(u), f"unique_counts/{dt}")
        emit_tuple("unique_inverse", {}, [u], np.unique_inverse(u), f"unique_inverse/{dt}")
        emit_tuple("unique_all", {}, [u], np.unique_all(u), f"unique_all/{dt}")
        if d.kind == "f":
            emit_array("unique_values", {}, u, np.unique_values(u), f"unique_values/{dt}")

    # unique_values(sorted=False) is portable for floating dtypes. Add both narrow widths and an
    # adversarial signed-zero/NaN payload class so this public op is not represented by one row.
    for dt in ["float16", "float32"]:
        u = np.array([[2, 1, 2], [0, 1, 3]], dtype=dt)
        emit_array("unique_values", {}, u, np.unique_values(u), f"unique_values/{dt}")
    u_special = np.array([np.nan, -0.0, 0.0, 2.0, np.nan, -1.0], dtype=np.float64)
    emit_array("unique_values", {}, u_special, np.unique_values(u_special), "unique_values/special")

    # Broadcasting returns one view per operand and preserves each operand's dtype.
    ba = np.arange(3, dtype=np.int32).reshape(3, 1)
    bb = np.arange(4, dtype=np.float64).reshape(1, 4)
    emit_tuple("broadcast_arrays", {}, [ba, bb], np.broadcast_arrays(ba, bb), "broadcast2")
    bc = np.array(2 + 3j, dtype=np.complex128)
    emit_tuple("broadcast_arrays", {}, [ba, bb, bc], np.broadcast_arrays(ba, bb, bc), "broadcast3")
    bd = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
    be = np.arange(4, dtype=np.int16).reshape(1, 4, 1)
    emit_tuple("broadcast_arrays", {}, [bd, be], np.broadcast_arrays(bd, be), "broadcast_rank3")
    bf = np.array(True, dtype=bool)
    bg = np.arange(5, dtype=np.uint64)
    emit_tuple("broadcast_arrays", {}, [bf, bg], np.broadcast_arrays(bf, bg), "broadcast_scalar")

    # meshgrid's indexing/sparse/copy parameters change both shape and result arity/slots.
    mx = np.array([1, 2, 4], dtype=np.int32)
    my = np.array([0.5, 1.5], dtype=np.float64)
    for indexing in ["xy", "ij"]:
        for sparse in [False, True]:
            emit_tuple("meshgrid", {"indexing": indexing, "sparse": sparse, "copy": True},
                       [mx, my], np.meshgrid(mx, my, indexing=indexing, sparse=sparse, copy=True),
                       f"meshgrid/{indexing}/sparse={int(sparse)}")
    # complex128 meshgrid: the grids are a pure broadcast COPY of each operand (no arithmetic) and
    # PRESERVE each input's dtype, so a complex operand round-trips bit-exact — both the dense and
    # the sparse (open-mesh) slot layouts. Two complex operands so every output slot is complex128.
    mc = np.array([1 + 1j, 2 - 1j, 0 + 3j], dtype=np.complex128)
    md = np.array([0.5 - 2j, 1.5 + 0j], dtype=np.complex128)
    for sparse in [False, True]:
        emit_tuple("meshgrid", {"indexing": "xy", "sparse": sparse, "copy": True},
                   [mc, md], np.meshgrid(mc, md, indexing="xy", sparse=sparse, copy=True),
                   f"meshgrid/complex/sparse={int(sparse)}")

    # unravel_index returns one coordinate array per dimension.
    flat = np.array([0, 5, 11, 7], dtype=np.int64)
    emit_tuple("unravel_index_all", {"shape": [3, 4]}, [flat], np.unravel_index(flat, (3, 4)),
               "unravel2")
    emit_tuple("unravel_index_all", {"shape": [3, 4], "order": "F"}, [flat],
               np.unravel_index(flat, (3, 4), order="F"), "unravel2F")
    flat3 = np.array([0, 7, 23, 13], dtype=np.int64)
    emit_tuple("unravel_index_all", {"shape": [2, 3, 4]}, [flat3],
               np.unravel_index(flat3, (2, 3, 4)), "unravel3")
    scalar_flat = np.array(5, dtype=np.int64)
    emit_tuple("unravel_index_all", {"shape": [2, 3]}, [scalar_flat],
               np.unravel_index(scalar_flat, (2, 3)), "unravel_scalar")

    # modf's two outputs have independent signed-zero/inf behavior; use both contiguous and
    # negative-stride operands at the two dtypes NumSharp currently implements.
    for dt in ["float32", "float64"]:
        mbase = np.array([-np.inf, -2.5, -0.0, 0.0, 1.25, np.inf], dtype=dt)
        emit_tuple("modf", {}, [mbase], np.modf(mbase), f"modf/c/{dt}")
        mrev = mbase[::-1]
        cases.append(_case("modf", {}, [describe(mbase, mrev)], _tuple_expected(np.modf(mrev)),
                           f"modf/neg/{dt}", "tuple", cid=f"modf/neg/{dt}/{n}"))
        n += 1

    # np.divmod's two outputs are (floor_divide, remainder). Cover the sign/edge grid at every
    # dtype NumPy has a divmod loop for (bb..QQ, ee/ff/dd), including integer ÷0 -> (0,0),
    # signed MIN/-1 -> (MIN,0), and float ÷0 -> (±inf, nan) plus ±inf/nan operands. Both the fused
    # kernel (contiguous) and the negative-stride materialize path are exercised.
    _DM_INT_DT = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
    for dt in _DM_INT_DT:
        d = np.dtype(dt)
        signed = np.issubdtype(d, np.signedinteger)
        if signed:
            a = np.array([7, -7, 6, -6, 0, 7, -20, 21, np.iinfo(d).min, 3, -9, 17], dtype=d)
            b = np.array([3, 3, 3, 3, 3, 0, 7, -7, -1, -1, 3, -3], dtype=d)  # ÷0 and ÷-1 (incl MIN/-1)
        else:
            a = np.array([7, 6, 0, 7, 20, 21, 100, 3, 9, 17, 5, 11], dtype=d)
            b = np.array([3, 3, 3, 0, 7, 7, 7, 3, 3, 3, 3, 0], dtype=d)      # ÷0
        emit_tuple("divmod", {}, [a, b], np.divmod(a, b), f"divmod/c/{dt}")
        cases.append(_case("divmod", {}, [describe(a, a[::-1]), describe(b, b[::-1])],
                           _tuple_expected(np.divmod(a[::-1], b[::-1])),
                           f"divmod/neg/{dt}", "tuple", cid=f"divmod/neg/{dt}/{n}"))
        n += 1

    for dt in ["float16", "float32", "float64"]:
        d = np.dtype(dt)
        a = np.array([7, -7, 5.3, -5.3, 6, -6, 0, 7, -0.0, np.inf, -np.inf, np.nan], dtype=d)
        b = np.array([3, -3, 2, 2, 3, 3, 0, 0, 3, 3, 3, 3], dtype=d)          # ÷0, ±inf, nan
        emit_tuple("divmod", {}, [a, b], np.divmod(a, b), f"divmod/c/{dt}")
        cases.append(_case("divmod", {}, [describe(a, a[::-1]), describe(b, b[::-1])],
                           _tuple_expected(np.divmod(a[::-1], b[::-1])),
                           f"divmod/neg/{dt}", "tuple", cid=f"divmod/neg/{dt}/{n}"))
        n += 1

    # Broadcast (2-D dividend, scalar-column divisor) — exercises the materialize-broadcast path.
    da = (np.arange(12, dtype=np.float64) + 1).reshape(3, 4)
    dcol = np.array([[2.0], [-3.0], [5.0]], dtype=np.float64)
    emit_tuple("divmod", {}, [da, dcol], np.divmod(da, dcol), "divmod/bcast/float64")
    # Mixed dtype (NEP50 promotion to float64).
    dia = np.array([7, -7, 8, -8, 9, -9], dtype=np.int32)
    dfb = np.array([3.0, 3.0, 2.0, 2.0, 4.0, 4.0], dtype=np.float64)
    emit_tuple("divmod", {}, [dia, dfb], np.divmod(dia, dfb), "divmod/mixed/i32_f64")

    # np.average(..., returned=True): the second slot (sum of weights) has its own shape/dtype
    # contract and was wholly invisible while only the first average result was gated.
    for dt in ["int32", "float64"]:
        a = _cbase((3, 4), np.dtype(dt))
        emit_tuple("average_returned", {"axis": None, "keepdims": False, "weighted": False},
                   [a], np.average(a, returned=True), f"average/all/{dt}")
        w = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
        emit_tuple("average_returned", {"axis": 1, "keepdims": True, "weighted": True},
                   [a, w], np.average(a, axis=1, weights=w, returned=True, keepdims=True),
                   f"average/weighted/{dt}")

    # np.trapezoid (composite trapezoidal integration -> array/scalar) and np.gradient (numerical
    # gradient -> a bare array for one axis, a tuple otherwise). Swept across layout × dtype so the
    # weak/strong spacing precision (float32/float16 stay their dtype), the edge_order stencils and
    # the multi-axis arity are all gated. Integer/bool tier to float64; gradient(bool) raises
    # (skipped). trapezoid over a 1-D view yields a 0-d scalar array.
    def _grad_tuple(r):
        return list(r) if isinstance(r, tuple) else [r]

    GRADTRAP_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "c_contiguous_3d", "f_contiguous_2d",
                        "transposed_3d", "strided_2d_cols", "negstride_1d", "strided_step2_1d"]
    GRADTRAP_DTYPES = ["float64", "float32", "float16", "int32", "int64", "uint8", "complex128"]
    for ln in GRADTRAP_LAYOUTS:
        fn = LAYOUTS[ln]
        for dt in GRADTRAP_DTYPES:
            base, view = fn(np.dtype(dt))
            nd = view.ndim
            op = describe(base, view)

            # trapezoid: dx / axis variations (all single-operand, x=None).
            trap_params = [{}, {"dx": 2.0}]
            if nd >= 1:
                trap_params.append({"axis": -1})
            if nd >= 2:
                trap_params.append({"axis": 0})
            for params in trap_params:
                try:
                    r = np.asarray(np.trapezoid(view, dx=params.get("dx", 1.0),
                                                axis=params.get("axis", -1)))
                except Exception:
                    continue
                cases.append(_case("trapezoid", params, [op], _arr_expected(r), ln, "mixed",
                                   cid=f"trapezoid/{ln}/{dt}/{n}"))
                n += 1

            # gradient: unit spacing, edge_order=2, single-axis. Recorded as a tuple so the arity
            # (one component per axis) is gated too.
            grad_params = [{}]
            if nd >= 1:
                grad_params.append({"edge_order": 2})
                grad_params.append({"axis": 0})
            for params in grad_params:
                try:
                    if "axis" in params:
                        r = np.gradient(view, axis=params["axis"],
                                        edge_order=params.get("edge_order", 1))
                    else:
                        r = np.gradient(view, edge_order=params.get("edge_order", 1))
                except Exception:
                    continue
                cases.append(_case("gradient", params, [op], _tuple_expected(_grad_tuple(r)), ln,
                                   "tuple", cid=f"gradient/{ln}/{dt}/{n}"))
                n += 1

    # np.frexp(x) -> (mantissa in [0.5,1), int32 exponent), the two-output inverse of ldexp. The
    # mantissa carries the input's float tier (int/bool promote: bool/int8/uint8 -> f16, int16/
    # uint16 -> f32, int32+ -> f64); the exponent is ALWAYS int32. Special values follow the scalar
    # C-runtime npy_frexp: frexp(±0)=(±0,0), frexp(±inf)=(±inf,-1), frexp(NaN)=(NaN,-1) with a
    # signalling NaN quieted. Both slots are bit-compared by the tuple comparator.
    for dt in ["float16", "float32", "float64"]:
        d = np.dtype(dt)
        fb = np.array([0.0, -0.0, 1.0, -8.5, 0.75, 1024.0, np.inf, -np.inf, np.nan], dtype=d)
        emit_tuple("frexp", {}, [fb], np.frexp(fb), f"frexp/c/{dt}")
        frev = fb[::-1]
        cases.append(_case("frexp", {}, [describe(fb, frev)], _tuple_expected(np.frexp(frev)),
                           f"frexp/neg/{dt}", "tuple", cid=f"frexp/neg/{dt}/{n}"))
        n += 1
    # Integer/bool inputs promote through the unary float tier (mantissa dtype varies by width).
    for dt in ["bool", "int8", "uint8", "int16", "uint16", "int32", "int64", "uint64"]:
        d = np.dtype(dt)
        ib = np.array([1, 0, 1, 1] if dt == "bool" else [1, 2, 3, 8], dtype=d)
        emit_tuple("frexp", {}, [ib], np.frexp(ib), f"frexp/int/{dt}")

    # np.ldexp(x1, x2) == x1 * 2^x2 (the inverse of frexp). x1 is the float mantissa (int/bool
    # promote through the same tier); x2 is an INTEGER exponent that does NOT widen x1's dtype, so
    # the result is purely x1's float tier. The exponent is clamped to the C-int range (a huge
    # magnitude overflows to ±inf / underflows to ±0). Single-array (kind="array") 2-operand cases.
    for dt in ["float16", "float32", "float64"]:
        d = np.dtype(dt)
        xf = np.array([1.0, -1.0, 0.0, -0.0, 1.5, 3.0, np.inf, -np.inf, np.nan], dtype=d)
        ef = np.array([0, 1, 5, -3, 2, -1, 1, 1, 1], dtype=np.int32)
        cases.append(_case("ldexp", {}, [describe(xf, xf), describe(ef, ef)],
                           _arr_expected(np.ldexp(xf, ef)), f"ldexp/c/{dt}", "array",
                           cid=f"ldexp/c/{dt}/{n}")); n += 1
        cases.append(_case("ldexp", {}, [describe(xf, xf[::-1]), describe(ef, ef[::-1])],
                           _arr_expected(np.ldexp(xf[::-1], ef[::-1])), f"ldexp/neg/{dt}", "array",
                           cid=f"ldexp/neg/{dt}/{n}")); n += 1
    # Integer x tier + varied exponent dtypes (int8 / uint32 / int64 exponent loops).
    for xdt, edt in [("int32", "int8"), ("int64", "uint32"), ("float64", "int64"), ("bool", "int16")]:
        xa = np.array([1, 0, 1, 1] if xdt == "bool" else [1, 2, 4, 8], dtype=np.dtype(xdt))
        ea = np.array([2, 3, 0, 1], dtype=np.dtype(edt))
        cases.append(_case("ldexp", {}, [describe(xa, xa), describe(ea, ea)],
                           _arr_expected(np.ldexp(xa, ea)), f"ldexp/xe/{xdt}-{edt}", "array",
                           cid=f"ldexp/xe/{xdt}-{edt}/{n}")); n += 1
    # int64 exponent clamp (huge magnitude -> overflow/underflow) and broadcasting.
    xc = np.array([1.0, 1.0, 2.0, -3.0], dtype=np.float64)
    nc = np.array([2**40, -2**40, 3, 4], dtype=np.int64)
    cases.append(_case("ldexp", {}, [describe(xc, xc), describe(nc, nc)],
                       _arr_expected(np.ldexp(xc, nc)), "ldexp/clamp64", "array",
                       cid=f"ldexp/clamp64/{n}")); n += 1
    xb = np.array([[1.0], [2.0], [4.0]], dtype=np.float64)
    eb = np.array([0, 1, 2, 3], dtype=np.int32)
    cases.append(_case("ldexp", {}, [describe(xb, xb), describe(eb, eb)],
                       _arr_expected(np.ldexp(xb, eb)), "ldexp/bcast", "array",
                       cid=f"ldexp/bcast/{n}")); n += 1

    return cases


# ---- deterministic creation ---------------------------------------------------------

def gen_creation(dtypes):
    """Deterministic creation APIs, including zero-operand calls.

    ``empty``/``empty_like`` have undefined initial bytes, so each result is immediately filled
    with zero on BOTH sides before serialization. That turns their deterministic shape/dtype/
    writeability contract into an ordinary byte comparison; the sibling flags oracle continues
    to own order/stride/ownership parity.
    """
    cases = []
    n = 0

    def emit(opname, params, operands, result, layout="creation"):
        nonlocal n
        cases.append(_case(opname, params, [describe(a, a) for a in operands],
                           _arr_expected(result), layout, "creation",
                           cid=f"{opname}/{layout}/{n}"))
        n += 1

    for dt in dtypes:
        d = np.dtype(dt)
        # Bool arange is defined only for result length <= 2 in NumPy 2.x.
        stop = 2 if d.kind == "b" else 7
        emit("arange", {"start": 0.0, "stop": float(stop), "step": 1.0, "dtype": dt}, [],
             np.arange(0, stop, 1, dtype=d), f"arange/{dt}")
        emit("linspace", {"start": -2.0, "stop": 3.0, "num": 7, "endpoint": True, "dtype": dt}, [],
             np.linspace(-2.0, 3.0, 7, endpoint=True, dtype=d), f"linspace/{dt}")
        emit("linspace", {"start": 0.0, "stop": 1.0, "num": 5, "endpoint": False, "dtype": dt}, [],
             np.linspace(0.0, 1.0, 5, endpoint=False, dtype=d), f"linspace_noend/{dt}")

        # logspace == power(base, linspace(...)).astype(dtype): float64 compute then cast (integer dtype
        # TRUNCATES). Small positive values (base=2 over [0,2] -> [1..4]) so no int dtype hits an
        # out-of-range float->int cast (that cell is platform-divergent — Fuzz/README "Host-dependent
        # values"). Bit-exact for every dtype incl. complex128 (real + 0j).
        emit("logspace", {"start": 0.0, "stop": 2.0, "num": 5, "endpoint": True, "base": 2.0, "dtype": dt}, [],
             np.logspace(0.0, 2.0, 5, endpoint=True, base=2.0, dtype=d), f"logspace/{dt}")
        emit("logspace", {"start": 0.0, "stop": 2.0, "num": 4, "endpoint": False, "base": 2.0, "dtype": dt}, [],
             np.logspace(0.0, 2.0, 4, endpoint=False, base=2.0, dtype=d), f"logspace_noend/{dt}")
        # geomspace REAL path is bit-exact (out_sign=±1 exact, endpoints=original). The complex128 dtype
        # computes in the complex128 domain (allclose within the ≤3-ULP complex-unary envelope, NOT
        # byte-reproducible) and is EXCLUDED here — unit-test-pinned, like np.sinc's complex path.
        if d.kind != "c":
            emit("geomspace", {"start": 1.0, "stop": 16.0, "num": 5, "endpoint": True, "dtype": dt}, [],
                 np.geomspace(1.0, 16.0, 5, endpoint=True, dtype=d), f"geomspace/{dt}")

        shape = [2, 3]
        emit("zeros", {"shape": shape, "dtype": dt}, [], np.zeros(shape, dtype=d), f"zeros/{dt}")
        emit("ones", {"shape": shape, "dtype": dt}, [], np.ones(shape, dtype=d), f"ones/{dt}")
        emit("full", {"shape": shape, "fill": 3, "dtype": dt}, [], np.full(shape, 3, dtype=d),
             f"full/{dt}")
        empty = np.empty(shape, dtype=d)
        empty.fill(0)
        emit("empty", {"shape": shape, "dtype": dt}, [], empty, f"empty/{dt}")
        emit("eye", {"n": 3, "m": 4, "k": -1, "dtype": dt, "order": "C"}, [],
             np.eye(3, 4, k=-1, dtype=d, order="C"), f"eye/{dt}")
        emit("identity", {"n": 4, "dtype": dt}, [], np.identity(4, dtype=d), f"identity/{dt}")

        # Like-creators must preserve the source shape/dtype while resolving K-order. Values are
        # checked here; the sibling layout oracle checks the resulting strides/flags.
        for lname in ["c_contiguous_2d", "f_contiguous_2d", "strided_2d_cols"]:
            base, view = LAYOUTS[lname](d)
            operand = describe(base, view)
            for opname, params, result in [
                ("empty_like", {}, np.empty_like(view)),
                ("zeros_like", {}, np.zeros_like(view)),
                ("ones_like", {}, np.ones_like(view)),
                ("full_like", {"fill": 2}, np.full_like(view, 2)),
            ]:
                if opname == "empty_like":
                    result.fill(0)
                cases.append(_case(opname, params, [operand], _arr_expected(result),
                                   f"{opname}/{lname}/{dt}", "creation",
                                   cid=f"{opname}/{lname}/{dt}/{n}"))
                n += 1

    # Coordinate-grid creators. Sparse mode is a tuple; dense mode is one array.
    for dt in ["int32", "float64"]:
        dims = [2, 3, 4]
        emit("indices", {"dimensions": dims, "dtype": dt}, [],
             np.indices(dims, dtype=np.dtype(dt)), f"indices/{dt}")
        sparse = np.indices(dims, dtype=np.dtype(dt), sparse=True)
        cases.append(_case("indices_sparse", {"dimensions": dims, "dtype": dt}, [],
                           _tuple_expected(sparse), f"indices_sparse/{dt}", "creation",
                           cid=f"indices_sparse/{dt}/{n}"))
        n += 1

    return cases


# ---- array conversion ---------------------------------------------------------------

CONVERSION_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "f_contiguous_2d",
                      "strided_step2_1d", "negstride_1d", "scalar_0d", "empty_2d",
                      "broadcast_1d_to_2d"]


def gen_conversion(dtypes):
    """Value/error half of the as*/array/require/frombuffer/fromstring surface.

    The sibling flags + layout oracles own strides/ownership/writeability; this tier owns the
    dtype/shape/bytes and the finite-check error. Together they cover the whole observable result.
    """
    cases = []
    n = 0

    def add_result(opname, params, operands, result, layout, dt):
        nonlocal n
        cases.append(_case(opname, params, operands, _arr_expected(result), layout, "conversion",
                           cid=f"{opname}/{layout}/{dt}/{n}"))
        n += 1

    for lname in CONVERSION_LAYOUTS:
        fn = LAYOUTS[lname]
        for dt in dtypes:
            base, view = fn(np.dtype(dt))
            operand = describe(base, view)
            one = [operand]

            add_result("array", {}, one, np.array(view, copy=True, order="K"), lname, dt)
            add_result("asarray", {}, one, np.asarray(view), lname, dt)
            add_result("asanyarray", {}, one, np.asanyarray(view), lname, dt)
            add_result("ascontiguousarray", {}, one, np.ascontiguousarray(view), lname, dt)
            add_result("asfortranarray", {}, one, np.asfortranarray(view), lname, dt)
            add_result("require", {"requirements": ["C"]}, one,
                       np.require(view, requirements=["C"]), lname + "/reqC", dt)
            add_result("require", {"requirements": ["F"]}, one,
                       np.require(view, requirements=["F"]), lname + "/reqF", dt)

            if view.ndim <= 2:
                add_result("asmatrix", {}, one, np.asmatrix(view), lname, dt)

            try:
                finite = np.asarray_chkfinite(view)
            except Exception as e:
                cases.append(_error_case("asarray_chkfinite", {}, one, e, lname,
                                         cid=f"asarray_chkfinite/{lname}/{dt}/err/{n}"))
                n += 1
            else:
                add_result("asarray_chkfinite", {}, one, finite, lname, dt)

    # frombuffer: the operand supplies deterministic raw bytes; count+byte-offset exercise the
    # byte-oriented API rather than reducing this to another astype case.
    for dt in dtypes:
        d = np.dtype(dt)
        raw_arr = _fill(6, d)
        result = np.frombuffer(raw_arr.tobytes(), dtype=d, count=4, offset=d.itemsize)
        add_result("frombuffer", {"dtype": dt, "count": 4, "offset": int(d.itemsize)},
                   [describe(raw_arr, raw_arr)], result, "frombuffer", dt)

        # fromfile has the same binary contract but crosses a real file descriptor on the NumPy
        # side and the public Stream overload on replay. The operand is the file's exact bytes.
        # Windows does not permit NumPy to reopen an already-open NamedTemporaryFile, so close
        # the descriptor before np.fromfile and remove the exact path in finally.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(raw_arr.tobytes())
            temp_name = f.name
        try:
            result = np.fromfile(temp_name, dtype=d, count=4, offset=d.itemsize)
        finally:
            os.unlink(temp_name)
        add_result("fromfile", {"dtype": dt, "count": 4, "offset": int(d.itemsize), "sep": ""},
                   [describe(raw_arr, raw_arr)], result, "fromfile/binary", dt)

        text = "0,1,3,5"
        result = np.fromstring(text, dtype=d, count=-1, sep=",")
        add_result("fromstring", {"dtype": dt, "count": -1, "sep": ",", "text": text},
                   [], result, "fromstring", dt)

        # Text I/O is an artifact oracle: loadtxt materializes the parsed array, while savetxt
        # records the emitted text verbatim. Safe small values make the cases portable for every
        # NumPy dtype, including complex128.
        text2d = "0,1,2\n3,4,5\n"
        result = np.loadtxt(io.StringIO(text2d), dtype=d, delimiter=",")
        add_result("loadtxt", {"dtype": dt, "delimiter": ",", "text": text2d},
                   [], result, "loadtxt", dt)

        saved = np.arange(6).astype(d).reshape(2, 3)
        fmt = "%d" if d.kind in "bui" else "%.6g"
        out = io.StringIO()
        np.savetxt(out, saved, fmt=fmt, delimiter=",", newline="\n",
                   header="head", footer="foot", comments="# ")
        cases.append(_case("savetxt",
                           {"fmt": fmt, "delimiter": ",", "newline": "\n",
                            "header": "head", "footer": "foot", "comments": "# "},
                           [describe(saved, saved)], {"kind": "text", "value": out.getvalue()},
                           "savetxt", "conversion", cid=f"savetxt/{dt}/{n}"))
        n += 1

    return cases


# ---- iterator traces ----------------------------------------------------------------
#
# np.ndindex / np.ndenumerate / np.nditer / np.broadcast return no array, which is why they
# were left out of this corpus. But what they actually promise is an ORDER, and the
# materialized trace of that order IS an array — so it bit-compares like anything else.
# Nothing else in the corpus can see a traversal-order drift: every other tier consumes
# NDIter's output already reduced to a value.

NDINDEX_SHAPES = [(), (1,), (3,), (0,), (2, 3), (3, 1), (1, 3), (0, 3), (3, 0),
                  (2, 2, 2), (2, 1, 3), (4, 1, 1), (5, 2), (1, 1, 1, 1), (2, 3, 4)]

# Layouts whose iteration order is NOT the memory order — where an order bug actually shows.
ITER_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "c_contiguous_3d", "f_contiguous_2d",
                "f_contiguous_3d", "transposed_2d", "transposed_3d", "strided_step2_1d",
                "negstride_1d", "negstride_2d_offset", "strided_2d_cols", "strided_outer_2d",
                "simple_slice_offset_1d", "sliced_composed", "broadcast_1d_to_2d",
                "broadcast_row_partial", "scalar_0d", "one_element_1d", "highrank_5d",
                "singleton_dim_3d", "newaxis_inserted", "reshape_view_2d"]

ITER_DTYPES = ["bool", "int8", "uint16", "int32", "int64", "float16", "float32",
               "float64", "complex128"]

ITER_ORDERS = ["C", "F", "A", "K"]


def gen_ndindex():
    cases = []
    for i, shp in enumerate(NDINDEX_SHAPES):
        idxs = list(np.ndindex(*shp))
        ndim = len(shp)
        arr = np.array(idxs, dtype=np.intp).reshape(len(idxs), ndim)
        cases.append(_case("ndindex", {"shape": [int(d) for d in shp]}, [],
                           _arr_expected(arr), "generator", "index",
                           cid=f"ndindex/{'x'.join(str(d) for d in shp) or '0d'}/{i}"))
    return cases


def gen_ndenumerate(dtypes, layout_names):
    """(index, value) for every element — always LOGICAL C-order, whatever the layout."""
    cases = []
    n = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            pairs = list(np.ndenumerate(view))
            idx = np.array([p[0] for p in pairs], dtype=np.intp).reshape(len(pairs), view.ndim)
            vals = np.array([p[1] for p in pairs], dtype=view.dtype) if pairs \
                else np.empty(0, view.dtype)
            cases.append(_case("ndenumerate", {}, [describe(base, view)],
                               _tuple_expected([idx, vals]), ln, "index",
                               cid=f"ndenumerate/{ln}/{s}/{n}"))
            n += 1
    return cases


def gen_nditer(dtypes, layout_names, orders):
    """
    The four observable streams of an nditer pass: values in iteration order, the
    multi_index stream, the tracked flat index (c_index / f_index), and — under
    external_loop — the CHUNK LENGTHS, i.e. how the iterator coalesced the dimensions.
    """
    cases = []
    n = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            for order in orders:
                # values
                try:
                    with np.nditer(view, order=order) as it:
                        vals = np.array([x.copy() for x in it], dtype=view.dtype)
                    cases.append(_case("nditer", {"order": order}, [operand],
                                       _arr_expected(vals), ln, "iter",
                                       cid=f"nditer/{ln}/{s}/{order}/{n}"))
                    n += 1
                except Exception as e:
                    cases.append(_error_case("nditer", {"order": order}, [operand], e, ln,
                                             cid=f"nditer/{ln}/{s}/{order}/{n}"))
                    n += 1
                    continue

                # multi_index + the values it labels
                try:
                    rows, mvals = [], []
                    with np.nditer(view, flags=["multi_index"], order=order) as it:
                        while not it.finished:
                            rows.append(it.multi_index)
                            mvals.append(it[0].copy())
                            it.iternext()
                    midx = np.array(rows, dtype=np.intp).reshape(len(rows), view.ndim)
                    mv = np.array(mvals, dtype=view.dtype) if mvals else np.empty(0, view.dtype)
                    cases.append(_case("nditer_multi_index", {"order": order}, [operand],
                                       _tuple_expected([midx, mv]), ln, "iter",
                                       cid=f"nditer_multi_index/{ln}/{s}/{order}/{n}"))
                    n += 1
                except Exception:
                    pass

                # tracked flat index, both spellings
                for flag in ("c_index", "f_index"):
                    try:
                        seen = []
                        with np.nditer(view, flags=[flag], order=order) as it:
                            while not it.finished:
                                seen.append(it.index)
                                it.iternext()
                        cases.append(_case("nditer_index", {"order": order, "index": flag}, [operand],
                                           _arr_expected(np.array(seen, dtype=np.intp)), ln, "iter",
                                           cid=f"nditer_index/{flag}/{ln}/{s}/{order}/{n}"))
                        n += 1
                    except Exception:
                        pass

                # external_loop: concatenated values + chunk lengths
                try:
                    chunks, lens = [], []
                    with np.nditer(view, flags=["external_loop"], order=order) as it:
                        while not it.finished:
                            chunk = it[0]
                            lens.append(len(chunk))
                            chunks.append(chunk.copy())
                            it.iternext()
                    flatv = np.concatenate(chunks) if chunks else np.empty(0, view.dtype)
                    cases.append(_case("nditer_extloop", {"order": order}, [operand],
                                       _tuple_expected([flatv, np.array(lens, dtype=np.intp)]),
                                       ln, "iter",
                                       cid=f"nditer_extloop/{ln}/{s}/{order}/{n}"))
                    n += 1
                except Exception:
                    pass
    return cases


def gen_nditer_pair(dt_pairs, pair_layout_names, orders):
    """Two operands walked in lockstep — broadcasting resolved inside the iterator."""
    cases = []
    n = 0
    for ln in pair_layout_names:
        fn = PAIR_LAYOUTS[ln]
        for (sa, sb) in dt_pairs:
            ba, va, bb, vb = fn(np.dtype(sa), np.dtype(sb))
            operands = [describe(ba, va), describe(bb, vb)]
            for order in orders:
                try:
                    sa_vals, sb_vals = [], []
                    with np.nditer([va, vb], order=order) as it:
                        while not it.finished:
                            sa_vals.append(it[0].copy())
                            sb_vals.append(it[1].copy())
                            it.iternext()
                    arr_a = np.array(sa_vals, dtype=va.dtype) if sa_vals else np.empty(0, va.dtype)
                    arr_b = np.array(sb_vals, dtype=vb.dtype) if sb_vals else np.empty(0, vb.dtype)
                    cases.append(_case("nditer_pair", {"order": order}, operands,
                                       _tuple_expected([arr_a, arr_b]), ln, "iter",
                                       cid=f"nditer_pair/{ln}/{sa},{sb}/{order}/{n}"))
                    n += 1
                except Exception:
                    pass
    return cases


def gen_broadcast(dt_pairs, pair_layout_names):
    """np.broadcast: the resolved shape and the per-operand value streams."""
    cases = []
    n = 0
    for ln in pair_layout_names:
        fn = PAIR_LAYOUTS[ln]
        for (sa, sb) in dt_pairs:
            ba, va, bb, vb = fn(np.dtype(sa), np.dtype(sb))
            operands = [describe(ba, va), describe(bb, vb)]
            try:
                b = np.broadcast(va, vb)
                shp = np.array(b.shape, dtype=np.intp)
                tuples = list(b)
                arr_a = np.array([t[0] for t in tuples], dtype=va.dtype) if tuples \
                    else np.empty(0, va.dtype)
                arr_b = np.array([t[1] for t in tuples], dtype=vb.dtype) if tuples \
                    else np.empty(0, vb.dtype)
            except Exception:
                continue
            cases.append(_case("broadcast_shape", {}, operands, _arr_expected(shp), ln, "iter",
                               cid=f"broadcast_shape/{ln}/{sa},{sb}/{n}"))
            cases.append(_case("broadcast_values", {}, operands,
                               _tuple_expected([arr_a, arr_b]), ln, "iter",
                               cid=f"broadcast_values/{ln}/{sa},{sb}/{n}"))
            n += 1
    return cases


def gen_iter():
    cases = gen_ndindex()
    cases += gen_ndenumerate(ITER_DTYPES, ITER_LAYOUTS)
    cases += gen_nditer(ITER_DTYPES, ITER_LAYOUTS, ITER_ORDERS)
    # complex128 pair-iteration: DT_PAIRS[:12] is all-real, so pairwise nditer / np.broadcast over a
    # complex operand was untested — yet iteration ORDER has no other gate and a 16-byte complex
    # element exercises the iterator's stride/coalesce handling differently from an 8-byte scalar. The
    # per-operand value stream is a pure COPY in traversal order (no arithmetic), so it is bit-exact by
    # construction (the single-operand nditer already gates complex via ITER_DTYPES; mixed-dtype pairs
    # like int32/float64 are already gated too — a complex/float pair is the same mechanism). Add a
    # complex-complex and a complex/float mixed pair.
    iter_pairs = DT_PAIRS[:12] + [("complex128", "complex128"), ("complex128", "float64")]
    cases += gen_nditer_pair(iter_pairs, list(PAIR_LAYOUTS.keys()), ["C", "K"])
    cases += gen_broadcast(iter_pairs, list(PAIR_LAYOUTS.keys()))
    cases += gen_nested_iters()
    return cases


def gen_nested_iters():
    """Materialize nested_iters' observable value stream.

    Object-returning APIs become oracle-friendly when their protocol is reduced to the trace they
    promise. Two axis partitions cover both two-level and three-level recursion; values retain the
    operand dtype so the ordinary byte comparator still owns dtype and order.
    """
    cases = []
    n = 0
    for dt in ["int32", "float64", "complex128"]:
        a = np.arange(12).astype(np.dtype(dt)).reshape(2, 3, 2)
        for axes in [[[1], [0, 2]], [[0], [1], [2]]]:
            iters = np.nested_iters(a, axes, flags=["multi_index"], order="C")
            values = []

            def walk(level):
                for _ in iters[level]:
                    if level == len(iters) - 1:
                        values.append(iters[level][0].copy())
                    else:
                        walk(level + 1)

            try:
                walk(0)
            finally:
                for it in iters:
                    it.close()

            result = np.array(values, dtype=a.dtype)
            cases.append(_case("nested_iters", {"axes": axes, "order": "C"},
                               [describe(a, a)], _arr_expected(result), "nested", "iter",
                               cid=f"nested_iters/{dt}/{len(axes)}level/{n}"))
            n += 1
    return cases


# ---- dtype / scalar / text / tuple results ------------------------------------------

DTYPE_TEXT_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "f_contiguous_2d", "transposed_2d",
                      "strided_step2_1d", "negstride_1d", "scalar_0d", "one_element_1d",
                      "empty_2d", "broadcast_1d_to_2d", "highrank_5d", "c_contiguous_3d"]

MIN_SCALAR_VALUES = [0, 1, -1, 127, 128, 255, 256, -128, -129, 32767, 65535, 2 ** 31 - 1,
                     2 ** 31, 2 ** 63 - 1, 0.5, -0.5, 1e10, 1e-10, True, False]

CASTING_RULES = ["no", "equiv", "safe", "same_kind", "unsafe"]


def gen_dtype_text():
    """
    The three non-array result kinds, plus the tuple kind on real multi-output ops.

    The promotion helpers (result_type / promote_types / min_scalar_type) are the NEP50
    table itself; until now it was only ever gated INDIRECTLY, through the dtype of some
    binary op's result.
    """
    cases = []
    n = 0

    # --- dtype-returning: the promotion table, gated directly ---
    for a in ALL_DTYPES:
        for b in ALL_DTYPES:
            r = np.promote_types(a, b)
            cases.append(_case("promote_types", {"a": a, "b": b}, [],
                               {"kind": "dtype", "value": r.name}, "generator", "dtype",
                               cid=f"promote_types/{a},{b}/{n}"))
            n += 1
            r2 = np.result_type(np.dtype(a), np.dtype(b))
            cases.append(_case("result_type_dtypes", {"a": a, "b": b}, [],
                               {"kind": "dtype", "value": r2.name}, "generator", "dtype",
                               cid=f"result_type_dtypes/{a},{b}/{n}"))
            n += 1

    for v in MIN_SCALAR_VALUES:
        r = np.min_scalar_type(v)
        if r.name not in ALL_DTYPES:      # e.g. float16 for tiny floats is fine; longdouble is not
            continue
        cases.append(_case("min_scalar_type", {"value": v}, [],
                           {"kind": "dtype", "value": r.name}, "generator", "dtype",
                           cid=f"min_scalar_type/{v}/{n}"))
        n += 1

    # dtype construction itself (canonical name) and common_type over real operand arrays.
    for dt in ALL_DTYPES:
        r = np.dtype(dt)
        cases.append(_case("dtype", {"dtype": dt}, [],
                           {"kind": "dtype", "value": r.name}, "generator", "dtype",
                           cid=f"dtype/{dt}/{n}"))
        n += 1

    # result_type over real arrays (operand dtypes, not just dtype tokens)
    for ln in ["pp_contig_contig", "pp_contig_fortran", "pp_scalar_right", "pp_broadcast_row"]:
        fn = PAIR_LAYOUTS[ln]
        for (sa, sb) in DT_PAIRS:
            ba, va, bb, vb = fn(np.dtype(sa), np.dtype(sb))
            r = np.result_type(va, vb)
            cases.append(_case("result_type_arrays", {}, [describe(ba, va), describe(bb, vb)],
                               {"kind": "dtype", "value": r.name}, ln, "dtype",
                               cid=f"result_type_arrays/{ln}/{sa},{sb}/{n}"))
            n += 1

    for (sa, sb) in DT_PAIRS:
        a = _cbase((3,), np.dtype(sa))
        b = _cbase((3,), np.dtype(sb))
        try:
            r = np.common_type(a, b)
        except Exception as e:
            cases.append(_error_case("common_type", {}, [describe(a, a), describe(b, b)], e,
                                     "common_type", kind="dtype",
                                     cid=f"common_type/{sa},{sb}/err/{n}"))
        else:
            cases.append(_case("common_type", {}, [describe(a, a), describe(b, b)],
                               {"kind": "dtype", "value": np.dtype(r).name},
                               "common_type", "dtype", cid=f"common_type/{sa},{sb}/{n}"))
        n += 1

    # --- scalar-returning predicates (wrapped 0-d, the np.allclose pattern) ---
    for frm in ALL_DTYPES:
        for to in ALL_DTYPES:
            for rule in CASTING_RULES:
                r = np.can_cast(np.dtype(frm), np.dtype(to), casting=rule)
                cases.append(_case("can_cast", {"from": frm, "to": to, "casting": rule}, [],
                                   _arr_expected(np.bool_(r), "scalar"), "generator", "predicate",
                                   cid=f"can_cast/{frm}->{to}/{rule}/{n}"))
                n += 1

    # NumPy 2.x dtype predicates. The concrete issubdtype matrix gates the exact scalar-type
    # relation; abstract categories are covered by isdtype's named kinds.
    for dt in ALL_DTYPES:
        for kind in ["bool", "integral", "real floating", "complex floating", "numeric"]:
            r = np.isdtype(np.dtype(dt), kind)
            cases.append(_case("isdtype", {"dtype": dt, "kind_name": kind}, [],
                               _arr_expected(np.bool_(r), "scalar"), "generator", "predicate",
                               cid=f"isdtype/{dt}/{kind}/{n}"))
            n += 1
        for other in ALL_DTYPES:
            r = np.issubdtype(np.dtype(dt), np.dtype(other))
            cases.append(_case("issubdtype", {"a": dt, "b": other}, [],
                               _arr_expected(np.bool_(r), "scalar"), "generator", "predicate",
                               cid=f"issubdtype/{dt},{other}/{n}"))
            n += 1

    for chars in ["fd", "if", "q", "Qf", "eF", "?"]:
        cases.append(_case("mintypecode", {"typechars": chars}, [],
                           {"kind": "text", "value": np.mintypecode(chars)},
                           "generator", "text", cid=f"mintypecode/{chars}/{n}"))
        n += 1

    for ln in DTYPE_TEXT_LAYOUTS:
        fn = LAYOUTS[ln]
        for s in ITER_DTYPES:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            for opname, val in (("isscalar", np.isscalar(view)),
                                ("iscomplexobj", np.iscomplexobj(view)),
                                ("isrealobj", np.isrealobj(view)),
                                ("isfortran", np.isfortran(view)),
                                ("iterable", np.iterable(view))):
                cases.append(_case(opname, {}, [operand], _arr_expected(np.bool_(val), "scalar"),
                                   ln, "predicate", cid=f"{opname}/{ln}/{s}/{n}"))
                n += 1
            cases.append(_case("size", {"axis": None}, [operand],
                               _arr_expected(np.int64(np.size(view)), "scalar"), ln, "predicate",
                               cid=f"size/{ln}/{s}/{n}"))
            n += 1

            # --- text-returning: printing, held verbatim ---
            for opname, f in (("array_str", np.array_str), ("array_repr", np.array_repr)):
                cases.append(_case(opname, {}, [operand],
                                   {"kind": "text", "value": f(view)}, ln, "text",
                                   cid=f"{opname}/{ln}/{s}/{n}"))
                n += 1

            # --- tuple-returning: nonzero over ANY rank (all slots + arity) ---
            # 0-d raises in NumPy 2.x ("Calling nonzero on 0d arrays is not allowed"), which
            # is worth pinning as an error case rather than skipping.
            try:
                nz = np.nonzero(view)
            except Exception as e:
                cases.append(_error_case("nonzero_all", {}, [operand], e, ln, kind="tuple",
                                         cid=f"nonzero_all/{ln}/{s}/err/{n}"))
                n += 1
            else:
                cases.append(_case("nonzero_all", {}, [operand], _tuple_expected(nz), ln, "tuple",
                                   cid=f"nonzero_all/{ln}/{s}/{n}"))
                n += 1

    return cases


# ---- ufunc out= / where= ------------------------------------------------------------
#
# The elementwise core accepts out=/where= on ~40 ufuncs, but the corpus reached them only
# through maximum_out / minimum_out / clip_out (11 cases each), all with a CONTIGUOUS out and
# no mask at all. Everything the parameters actually promise was ungated:
#
#   * `where` masking is defined by what does NOT change. Recording the out array's PRIOR
#     contents as an operand and re-checking them afterwards is the whole assertion.
#   * a STRIDED / OFFSET / NEGSTRIDE / F-order / TRANSPOSED out is where a kernel that walks
#     the buffer instead of the view corrupts elements outside the window — invisible to a
#     view-shaped comparison, which is why every case also records the full base buffer.
#   * `out` joins the broadcast but is never STRETCHED, and a read-only (broadcast) out must
#     be refused: those land as error cases with NumPy's message.

OUT_VIEW_KINDS = ["c", "f", "strided", "negstride", "offset", "transposed", "broadcast"]

# (name, builder) — masks over the result shape, plus the broadcast and scalar spellings.
WHERE_KINDS = [None, "all_true", "all_false", "alternating", "checker", "row_broadcast",
               "scalar_true", "scalar_false", "strided_mask"]

OUT_SHAPES = [(6,), (4, 5), (2, 3, 4)]

# ufunc -> the input dtypes to drive it with (its natural domain).
#
# complex128 is added ONLY to the ufuncs whose complex loop is BIT-EXACT-and-PORTABLE (pure IEEE
# arithmetic / lexicographic comparison), never to the transcendentals (sqrt/exp/log/sin) or the
# magnitude ops (abs/sign): those compose the host CRT libm, so they are held bit-exact ONLY on
# win-amd64 and only within the ≤3-ULP complex-unary envelope (MisalignedRegistry branch 6b) — an
# envelope scoped to Operands.Length == 1, which the out=/where= operands (out buffer + optional
# mask) push past, so a complex transcendental here would fail the STRICT (all-platform) OutWhere
# gate. The arithmetic loops below run the SAME kernel with or without out=/where= (only the store
# target and the mask change), so their complex parity is inherited from the strict binary_arith /
# unary_extra / specials tiers that already gate it bit-exact. mod/floor_divide/arctan2/bitwise_*
# raise TypeError for complex in NumPy (the emit() probe skips them), so they stay real-only.
OUT_BINARY_UFUNCS = {
    "add": ["int32", "float64", "float32", "uint8", "complex128"],
    "subtract": ["int32", "float64", "complex128"],
    "multiply": ["int64", "float32", "complex128"],
    "divide": ["float64", "int32", "complex128"],
    "power": ["float64", "int32"],
    "float_power": ["float64", "int32"],
    "mod": ["int32", "float64"],
    "floor_divide": ["int32", "float64"],
    "fmod": ["int32", "float64"],                          # C truncated remainder (a % b) — bit-exact/portable
    "arctan2": ["float64", "float32"],
    "bitwise_and": ["int32", "uint8", "bool"],
    "bitwise_or": ["int64"],
    "bitwise_xor": ["uint16"],
    # min/max family — pure compare-and-select (no libm), NaN handling matches NumPy, so
    # bit-exact AND portable across platforms (safe for the strict RunCorpus tier). maximum/
    # minimum PROPAGATE NaN, fmax/fmin IGNORE it. The engine already routed out=/where=; only
    # the np.* overloads gained the params (2026-09-18), so these had zero fuzz coverage before.
    "maximum": ["float64", "int32"],
    "minimum": ["float64", "int32"],
    "fmax": ["float64", "int32"],
    "fmin": ["float64", "int32"],
    "gcd": ["int32"],                                      # Euclidean — integer, bit-exact/portable
    "lcm": ["int32"],                                      # |a|/gcd*|b| (wraps on overflow, matches NumPy)
    "copysign": ["float64"],                              # |a| with sign(b) — bit op, portable
    "nextafter": ["float64"],                            # IEEE nextafter — bit op, portable
    "heaviside": ["float64"],                            # compare+select — portable (x1==NaN -> NaN)
    "less": ["int32", "float64", "complex128"],           # lexicographic on complex -> bool out
    "greater_equal": ["float32", "complex128"],
    "equal": ["int32", "complex128"],
}

OUT_UNARY_UFUNCS = {
    "sqrt": ["float64", "float32"],                       # IEEE hardware sqrt — bit-exact/portable
    "negative": ["int32", "float64", "complex128"],        # pure component negate — bit-exact
    "abs": ["int32", "float64"],
    "fabs": ["float64", "int32"],                         # float |x| (sign-bit clear) — portable
    "square": ["float64", "int32", "complex128"],          # fused simd_cmul (z*z) — bit-exact
    "positive": ["int32", "float64"],                    # identity copy — portable
    # exp/log/sin: float32 ONLY. NumSharp's float32 exp/log/sin are BIT-EXACT ports of NumPy's own
    # SIMD kernels (NDFloatMath), so they reproduce byte-for-byte on EVERY platform (pure managed,
    # no libm). float64 exp/log/sin are Math.Exp/Log/Sin == the host CRT libm, bit-exact vs NumPy
    # ONLY on win-amd64 (ucrtbase) — they belong in the host-pinned unary.jsonl tier, NOT this
    # strict all-platform RunCorpus tier (they would go red on Linux/macOS CI). Dropped 2026-09-18.
    "exp": ["float32"],
    "log": ["float32"],
    "sin": ["float32"],
    "floor": ["float64"],                                # round toward -inf — exact/portable
    "ceil": ["float32"],
    "trunc": ["float64"],                                # round toward 0 — exact/portable
    "rint": ["float64", "complex128"],                     # rounds each component — bit-exact
    "sign": ["int32", "float64"],
    "reciprocal": ["float64", "int32", "complex128"],      # CDOUBLE_reciprocal (-1/d) — bit-exact
    "conjugate": ["complex128"],                          # negate imag — bit op, portable
    "invert": ["int32", "uint8"],
    # Float-classification predicates (bool out). Bit tests (exponent / sign bit), no libm, so
    # bit-exact AND portable. They expose NumPy 2.4.2's OWN strided-bool-out buffering BUG at
    # rank >= 2 (leaks prior `out` contents into a [..., ::2] view — see MisalignedRegistry K12);
    # NumSharp overwrites correctly, so those specific cells are an excused NumSharp-is-correct
    # divergence, bit-exact everywhere else.
    "isnan": ["float64", "complex128"],                    # True iff either lane is NaN -> bool out
    "isinf": ["float64"],                                 # |x| == +inf -> bool out
    "isfinite": ["float64"],                              # |x| < +inf -> bool out
    "signbit": ["float64", "int32"],                      # IEEE / two's-complement sign bit -> bool out
}


def _out_view(shape, dt, kind):
    """
    A (base, view) pair whose VIEW has exactly `shape` in the requested layout. The base is
    always larger than or equal to the view so the elements outside the window are real and
    can be checked for corruption.
    """
    dt = np.dtype(dt)
    n = int(np.prod(shape)) if shape else 1
    if kind == "c":
        base = _fill(n, dt).reshape(shape)
        return base, base
    if kind == "f":
        # The BASE must stay C-contiguous: describe() serializes it with base.tobytes(), which
        # is a C-order walk, while the recorded strides/offset are PHYSICAL. An F-ordered base
        # (np.asfortranarray) makes those two disagree and every F case reads as a divergence
        # that is really a corpus bug. F-contiguity is expressed the way layout_catalog does it
        # — a transposed view over a C base (see its f_contiguous_2d).
        base = _fill(n, dt).reshape(tuple(reversed(shape)))
        return base, base.T
    if kind == "strided":                      # every other column of a doubly-wide base
        wide_shape = tuple(shape[:-1]) + (shape[-1] * 2,)
        base = _fill(int(np.prod(wide_shape)), dt).reshape(wide_shape)
        return base, base[..., ::2]
    if kind == "negstride":
        base = _fill(n, dt).reshape(shape)
        return base, base[..., ::-1]
    if kind == "offset":                       # window starts 3 elements into the buffer
        base = _fill(n + 3, dt)
        return base, base[3:].reshape(shape)
    if kind == "transposed":
        # A NON-reversing permutation, so this stays distinct from "f" (whose .T reverses every
        # axis). Only meaningful at rank >= 3; lower ranks are covered by "f".
        if len(shape) < 3:
            return None
        src = (shape[1], shape[0]) + tuple(shape[2:])
        base = _fill(n, dt).reshape(src)
        return base, base.transpose(1, 0, *range(2, len(shape)))
    if kind == "broadcast":                    # read-only: NumPy must REFUSE this as out
        base = _fill(int(shape[-1]), dt)
        return base, np.broadcast_to(base, shape)
    raise ValueError(kind)


def _where_mask(shape, kind):
    """The mask operand, or None for NumPy's default where=True."""
    if kind is None:
        return None
    n = int(np.prod(shape)) if shape else 1
    if kind == "all_true":
        return np.ones(shape, dtype=bool)
    if kind == "all_false":
        return np.zeros(shape, dtype=bool)
    if kind == "alternating":
        return (np.arange(n) % 2 == 0).reshape(shape)
    if kind == "checker":
        return (np.arange(n) % 3 != 0).reshape(shape)
    if kind == "row_broadcast":                # (1, …, k) stretched over the leading axes
        if len(shape) < 2:
            return None
        m = np.zeros((1,) * (len(shape) - 1) + (shape[-1],), dtype=bool)
        m[..., ::2] = True
        return m
    if kind == "scalar_true":
        return np.array(True)
    if kind == "scalar_false":
        return np.array(False)
    if kind == "strided_mask":
        wide = np.zeros(tuple(shape[:-1]) + (shape[-1] * 2,), dtype=bool)
        wide[..., ::4] = True
        return wide[..., ::2]
    raise ValueError(kind)


def _mask_view(shape, kind):
    """
    A (base, view) BOOL-mask pair whose VIEW broadcasts to `shape` — the np.evaluate where= analogue
    of _out_view (plan P4.5). The base is always C-contiguous (describe() serializes it with
    base.tobytes(), a C-order walk, against the recorded PHYSICAL strides/offset), so a strided /
    reversed view reads back exactly. "row"/"col" keep a SMALLER contiguous mask that np.evaluate
    broadcasts up (the recorded view is the small array itself). Returns None for a kind that does not
    apply to the rank (e.g. "col" on a 1-D shape).
    """
    n = int(np.prod(shape)) if shape else 1
    if kind == "all_true":
        b = np.ones(shape, dtype=bool);  return b, b
    if kind == "all_false":
        b = np.zeros(shape, dtype=bool); return b, b
    if kind == "checker":                          # ~2/3 True, a non-trivial dense pattern
        b = np.ascontiguousarray((np.arange(n) % 3 != 0).reshape(shape)); return b, b
    if kind == "alt":                              # strict alternation (run-length-1 mask)
        b = np.ascontiguousarray((np.arange(n) % 2 == 0).reshape(shape)); return b, b
    if kind == "strided":                          # every-other-col of a doubly-wide contiguous base
        wide = np.zeros(tuple(shape[:-1]) + (shape[-1] * 2,), dtype=bool)
        wide[..., ::4] = True
        return wide, wide[..., ::2]
    if kind == "negstride":                        # reversed last axis (negative stride + offset)
        b = np.ascontiguousarray((np.arange(n) % 2 == 0).reshape(shape))
        return b, b[..., ::-1]
    if kind == "row":                              # (1,…,C) broadcasting down the leading axes
        if len(shape) < 2:
            return None
        m = np.zeros((1,) * (len(shape) - 1) + (shape[-1],), dtype=bool)
        m[..., ::2] = True
        b = np.ascontiguousarray(m); return b, b
    if kind == "col":                              # (R,1) broadcasting across the last axis (2-D)
        if len(shape) != 2:
            return None
        m = np.zeros((shape[0], 1), dtype=bool)
        m[::2, :] = True
        b = np.ascontiguousarray(m); return b, b
    raise ValueError(kind)


def gen_out_where():
    cases = []
    n = 0

    def emit(opname, ufunc, f, inputs, shape, out_kind, where_kind, dts):
        """Build out/where, capture the PRIOR state, run, and record both slots."""
        nonlocal n
        # The natural result dtype, so `out` needs no cast (the cast rules are their own axis).
        try:
            probe = f(*inputs)
        except Exception:
            return
        built = _out_view(shape, probe.dtype, out_kind)
        if built is None:
            return
        out_base, out_view = built
        if out_view.shape != tuple(shape):
            return
        mask = _where_mask(shape, where_kind)
        if where_kind is not None and mask is None:
            return

        operands = [describe(_cbase(i.shape, i.dtype) if i.base is None else i.base, i)
                    for i in inputs]
        # PRIOR contents recorded here, BEFORE the ufunc writes — this is what "masked-off
        # slots keep their prior contents" is checked against.
        operands.append(describe(out_base, out_view))
        if mask is not None:
            mask_base = mask if mask.base is None else mask.base
            operands.append(describe(mask_base, mask))

        params = {"ufunc": ufunc, "where": mask is not None}
        cid = f"{opname}/{ufunc}/{'x'.join(map(str, shape))}/{dts}/out={out_kind}/where={where_kind}/{n}"
        try:
            returned = f(*inputs, out=out_view, **({"where": mask} if mask is not None else {}))
        except Exception as e:
            cases.append(_error_case(opname, params, operands, e, f"out_{out_kind}",
                                     kind="tuple", cid=cid))
            n += 1
            return

        cases.append(_case(opname, params, operands,
                           _tuple_expected([np.asarray(returned), out_base.ravel()]),
                           f"out_{out_kind}", "outwhere", cid=cid))
        n += 1

    for shape in OUT_SHAPES:
        cnt = int(np.prod(shape))
        for ufunc, dts in OUT_BINARY_UFUNCS.items():
            f = getattr(np, "remainder" if ufunc == "mod" else ufunc)
            # The full out x where cross product for `add`; a representative slice for the rest,
            # so the tier stays a few thousand cases rather than tens of thousands.
            wheres = WHERE_KINDS if ufunc == "add" else [None, "alternating", "all_false", "row_broadcast"]
            for s in dts:
                a = _fill(cnt, np.dtype(s)).reshape(shape)
                b = np.roll(_fill(cnt, np.dtype(s)), 1).reshape(shape)
                for out_kind in OUT_VIEW_KINDS:
                    for wk in wheres:
                        emit("out_binary", ufunc, f, (a, b), shape, out_kind, wk, s)

        for ufunc, dts in OUT_UNARY_UFUNCS.items():
            f = getattr(np, "absolute" if ufunc == "abs" else ufunc)
            wheres = WHERE_KINDS if ufunc == "sqrt" else [None, "alternating", "all_false"]
            for s in dts:
                x = _fill(cnt, np.dtype(s)).reshape(shape)
                for out_kind in OUT_VIEW_KINDS:
                    for wk in wheres:
                        emit("out_unary", ufunc, f, (x,), shape, out_kind, wk, s)

    # ---- out= beyond the elementwise ufuncs (coverage plan §B2) -------------------------
    # NumSharp exposes out= on cumsum/cumprod, round_, clip and nanargmax/nanargmin (the np.sum
    # reduction family has NO out= overload yet — a tracked feature gap, deliberately absent so
    # the tier gates only implemented surface). Same two-slot contract as out_binary/out_unary:
    # slot 0 the returned view, slot 1 the ENTIRE base buffer behind `out`, with the out
    # operand's PRIOR contents recorded — so an out-kernel writing outside a strided/offset
    # window is caught, exactly as for the ufuncs. `where=` is not offered by any of these.
    def emit_extra(opname, params, inputs, probe, out_kind, tag, run):
        """Record one non-ufunc out= case. `probe` fixes the out view's dtype+shape (the natural
        result, so the out-cast axis stays out of scope); `run(out_view)` performs the call."""
        nonlocal n
        built = _out_view(list(probe.shape), probe.dtype, out_kind)
        if built is None:
            return
        out_base, out_view = built
        if out_view.shape != tuple(probe.shape):
            return
        operands = [describe(b, v) for (b, v) in inputs]
        operands.append(describe(out_base, out_view))
        cid = f"{opname}/{tag}/out={out_kind}/{n}"
        try:
            with np.errstate(all="ignore"):
                returned = run(out_view)
        except Exception as e:
            cases.append(_error_case(opname, params, operands, e, f"out_{out_kind}",
                                     kind="tuple", cid=cid))
            n += 1
            return
        cases.append(_case(opname, params, operands,
                           _tuple_expected([np.asarray(returned), out_base.ravel()]),
                           f"out_{out_kind}", "outwhere", cid=cid))
        n += 1

    extra_kinds = ["c", "strided", "negstride", "offset", "transposed"]
    for shape in [(6,), (4, 5)]:
        cnt = int(np.prod(shape))
        for s in ["int32", "uint8", "float16", "float64", "complex128"]:
            # .copy() so `a` OWNS its buffer (base is None): describe(base, view) demands the
            # view alias the base's buffer, and a reshape of _fill would smuggle a hidden base.
            a = _fill(cnt, np.dtype(s)).reshape(shape).copy()
            apair = (a, a)
            axes = [None] + list(range(len(shape)))
            # Scans: NEP50 accumulator dtype (int32 -> int64) with axis=None flattening.
            # complex128 cumPROD is CARVED exactly as the nanscan tier carves it: the win-amd64
            # NumPy complex product chain is MSVC-FMA-contracted (and copies element 0 where a
            # 1*z seed poisons both lanes through NaN), so its bytes are not portably
            # reproducible — complex cumSUM (exact addition) stays.
            for ax in axes:
                scan_jobs = [("cumsum", np.cumsum)] + ([] if s == "complex128"
                                                       else [("cumprod", np.cumprod)])
                for ufname, f in scan_jobs:
                    probe = f(a, axis=ax)
                    for ok in extra_kinds:
                        emit_extra("out_scan", {"ufunc": ufname, "axis": ax}, [apair], probe, ok,
                                   f"{ufname}/{'x'.join(map(str, shape))}/{s}/axis={ax}",
                                   lambda o, f=f, ax=ax: f(a, axis=ax, out=o))
            # round_(decimals, out=) — dtype-preserving banker's rounding into a view.
            for dec in (0, 1):
                probe = np.round(a, dec)
                for ok in extra_kinds:
                    emit_extra("out_round", {"decimals": dec}, [apair], probe, ok,
                               f"{'x'.join(map(str, shape))}/{s}/dec={dec}",
                               lambda o, dec=dec: np.round(a, dec, out=o))
            # clip(min, max, out=) — scalar bounds as 0-D operands (the registry's clip shape).
            if s != "complex128":                       # NumPy clip on complex with real bounds raises
                lo = np.array(0, dtype=a.dtype)
                hi = np.array(2, dtype=a.dtype)
                probe = np.clip(a, lo, hi)
                for ok in OUT_VIEW_KINDS:
                    emit_extra("out_clip", {}, [apair, (lo, lo), (hi, hi)], probe, ok,
                               f"{'x'.join(map(str, shape))}/{s}",
                               lambda o: np.clip(a, lo, hi, out=o))
            # nanargmax/nanargmin(axis, out=) — int64 indices scattered into an out view.
            if len(shape) == 2:
                for ufname, f in (("nanargmax", np.nanargmax), ("nanargmin", np.nanargmin)):
                    for ax in (0, 1):
                        try:
                            with np.errstate(all="ignore"):
                                probe = np.asarray(f(a, axis=ax))
                        except Exception:
                            continue                     # all-NaN slice: unit-test-pinned, not here
                        for ok in extra_kinds:
                            emit_extra("out_nanarg", {"ufunc": ufname, "axis": ax}, [apair], probe, ok,
                                       f"{ufname}/{'x'.join(map(str, shape))}/{s}/axis={ax}",
                                       lambda o, f=f, ax=ax: np.asarray(f(a, axis=ax, out=o)))

    return cases


# ---- error parity -------------------------------------------------------------------
#
# Every value generator SKIPS the cells where NumPy raises ("error-parity is tested
# separately" — it was not, beyond 24 hand-picked cases that asserted only that SOMETHING
# was thrown). This re-runs the same deterministic matrices and keeps exactly the skipped
# cells, recording NumPy's exception type and message verbatim.

# Flood guard, set generously: the deterministic matrices raise in ~700 cells spread over ~22
# distinct messages, so nothing is dropped today and every (layout, dtype) instance is kept —
# the message is only half the claim, the other half is that NumSharp raises on the same CELLS.
# If a future matrix makes one message explode, the cap trims it and gen_errors_full reports it.
ERROR_INSTANCES_PER_MESSAGE = 1000


def _error_case(op, params, operands, exc, layout, cid=None, kind=None):
    expected = {"kind": kind} if kind else {}
    return {"id": cid, "op": op, "params": params, "operands": operands,
            "expected": expected, "expects_throw": True, "error": _exc(exc),
            "layout": layout, "valueclass": "error"}


def gen_errors_full():
    cases = []
    seen = {}
    n = 0

    def keep(op, exc):
        """Cap identical (op, type, message) triples so one broken cell can't flood the tier."""
        k = (op, type(exc).__name__, str(exc))
        seen[k] = seen.get(k, 0) + 1
        return seen[k] <= ERROR_INSTANCES_PER_MESSAGE

    def unary_matrix(ops_map, dtypes, layout_names):
        nonlocal n
        for ln in layout_names:
            fn = LAYOUTS[ln]
            for s in dtypes:
                base, view = fn(np.dtype(s))
                operand = None
                for opname, f in ops_map.items():
                    try:
                        f(view)
                    except Exception as e:
                        if not keep(opname, e):
                            continue
                        operand = operand or describe(base, view)
                        cases.append(_error_case(opname, {}, [operand], e, ln,
                                                 cid=f"{opname}/{ln}/{s}/err/{n}"))
                        n += 1

    def binary_matrix(ops_map, dt_pairs, pair_layout_names):
        nonlocal n
        for ln in pair_layout_names:
            fn = PAIR_LAYOUTS[ln]
            for (sa, sb) in dt_pairs:
                ba, va, bb, vb = fn(np.dtype(sa), np.dtype(sb))
                operands = None
                for opname, f in ops_map.items():
                    try:
                        f(va, vb)
                    except Exception as e:
                        if not keep(opname, e):
                            continue
                        operands = operands or [describe(ba, va), describe(bb, vb)]
                        cases.append(_error_case(opname, {}, operands, e, ln,
                                                 cid=f"{opname}/{ln}/{sa},{sb}/err/{n}"))
                        n += 1

    def reduce_matrix(ops_map, dtypes, layout_names):
        nonlocal n
        for ln in layout_names:
            fn = LAYOUTS[ln]
            for s in dtypes:
                base, view = fn(np.dtype(s))
                operand = None
                for opname, f in ops_map.items():
                    for axis in _axes(view.ndim):
                        if opname in ("argmax", "argmin") and axis is None:
                            continue
                        for keepdims in (False, True):
                            try:
                                np.asarray(f(view, axis, keepdims))
                            except Exception as e:
                                if not keep(opname, e):
                                    continue
                                operand = operand or describe(base, view)
                                cases.append(_error_case(
                                    opname, {"axis": axis, "keepdims": keepdims}, [operand], e, ln,
                                    cid=f"{opname}/{ln}/{s}/axis={axis}/kd={int(keepdims)}/err/{n}"))
                                n += 1

    unary_matrix(UNARY_OPS, UNARY_DTYPES, list(LAYOUTS.keys()))
    unary_matrix(UNARY_EXTRA_OPS, ALL_DTYPES, list(LAYOUTS.keys()))
    unary_matrix(INVERT_OP, ALL_DTYPES, list(LAYOUTS.keys()))
    binary_matrix(BINARY_OPS, DT_PAIRS, list(PAIR_LAYOUTS.keys()))
    binary_matrix(DIVMOD_POWER_OPS, DT_PAIRS, list(PAIR_LAYOUTS.keys()))
    binary_matrix(COMPARISON_OPS, DT_PAIRS, list(PAIR_LAYOUTS.keys()))
    binary_matrix(BITWISE_BIN_OPS, BITWISE_DT_PAIRS, list(PAIR_LAYOUTS.keys()))
    reduce_matrix(REDUCE_OPS, REDUCE_DTYPES, REDUCE_LAYOUTS)

    # Iterator construction errors — the zero-sized-operand guard and ndindex's negative dims.
    for shp in [(-1,), (2, -3), (-1, -1)]:
        try:
            list(np.ndindex(*shp))
        except Exception as e:
            cases.append(_error_case("ndindex", {"shape": [int(d) for d in shp]}, [], e,
                                     "generator", cid=f"ndindex/neg/{shp}/err/{n}"))
            n += 1

    for ln in ["empty_2d", "empty_composed"]:
        if ln not in LAYOUTS:
            continue
        base, view = LAYOUTS[ln](np.dtype("int32"))
        for order in ["C", "K"]:
            try:
                with np.nditer(view, order=order) as it:
                    _ = [x.copy() for x in it]
            except Exception as e:
                cases.append(_error_case("nditer_values", {"order": order},
                                         [describe(base, view)], e, ln,
                                         cid=f"nditer_values/{ln}/empty/{order}/err/{n}"))
                n += 1

    # ---- curated raising cells beyond the elementwise/reduce matrices (coverage plan §B1) ----
    # Each recipe re-uses a REGISTERED op name with params OpRegistry already parses, so the only
    # new claim per case is the raising cell itself (exception type + verbatim NumPy message).
    # Recipes are restricted to ops whose error texts NumSharp ports verbatim (reshape/expand_dims/
    # flip/take/put/partition/linalg validation/fft n-guard) plus a few probe cells whose parity is
    # adjudicated by the gate itself (percentile q-range, matrix_transpose ndim).
    def curated(op, params, operand_pairs, f, tag):
        """Run f(); if NumPy raises, record the cell with the given params/operands."""
        nonlocal n
        try:
            with np.errstate(all="ignore"):
                f()
        except Exception as e:
            if not keep(op, e):
                return
            cases.append(_error_case(op, params, [describe(b, v) for (b, v) in operand_pairs],
                                     e, "curated", cid=f"{op}/curated/{tag}/err/{n}"))
            n += 1

    a2 = LAYOUTS["c_contiguous_2d"](np.dtype("int32"))          # (4, 5)
    a1f = LAYOUTS["c_contiguous_1d"](np.dtype("float64"))       # (8,)
    a1i = LAYOUTS["c_contiguous_1d"](np.dtype("int32"))         # (8,)

    # G7 manipulation — reshape rejection family (verbatim `cannot reshape…` + one-unknown rule).
    for shape in ([7], [3, 3], [-1, -2], [0, -1]):
        curated("reshape", {"shape": shape}, [a2],
                lambda shape=shape: np.reshape(a2[1], tuple(shape)), f"shape={shape}")
    # expand_dims axis out of bounds, both signs (validated against OUTPUT ndim, reported as given).
    for ax in (5, -5):
        curated("expand_dims", {"axis": ax}, [a2],
                lambda ax=ax: np.expand_dims(a2[1], ax), f"axis={ax}")
    # flip: axis out of bounds (verbatim AxisError) + repeated axis (checked after the full pass).
    curated("flip", {"axis": 5}, [a2], lambda: np.flip(a2[1], 5), "axis=5")
    curated("flip", {"axes": [0, 0]}, [a2], lambda: np.flip(a2[1], (0, 0)), "axes=0,0")
    # matrix_transpose demands ndim >= 2 (probe cell: NumPy wording vs NumSharp's port).
    curated("matrix_transpose", {}, [a1i], lambda: np.matrix_transpose(a1i[1]), "1d")

    # G11 selection — take/put out-of-bounds index (mode='raise', post-wrap check) and the
    # float-index dtype rejection (same_kind for take, safe for put — each names its rule).
    oob = np.array([0, 99], dtype=np.int64)
    fidx = np.array([0.0, 1.0], dtype=np.float64)
    vals = np.array([1, 2], dtype=np.int32)
    curated("take", {"axis": 0, "mode": "raise"}, [a1i, (oob, oob)],
            lambda: np.take(a1i[1], oob, axis=0, mode="raise"), "oob")
    curated("take", {"axis": 0, "mode": "raise"}, [a1i, (fidx, fidx)],
            lambda: np.take(a1i[1], fidx, axis=0, mode="raise"), "floatidx")
    curated("put", {"mode": "raise"}, [a1i, (oob, oob), (vals, vals)],
            lambda: np.put(a1i[1], oob, vals, mode="raise"), "oob")
    curated("put", {"mode": "raise"}, [a1i, (fidx, fidx), (vals, vals)],
            lambda: np.put(a1i[1], fidx, vals, mode="raise"), "floatidx")

    # G12 sorting — partition kth out of bounds (verbatim `kth(=N) out of bounds (M)`).
    curated("partition", {"kth": [99], "axis": -1}, [a1i],
            lambda: np.partition(a1i[1], 99), "kth-oob")

    # G5 statistics — percentile/quantile out-of-range q (probe cells).
    curated("percentile", {"q": 101.0, "axis": None}, [a1f],
            lambda: np.percentile(a1f[1], 101.0), "q101")
    curated("quantile", {"q": 1.5, "axis": None}, [a1f],
            lambda: np.quantile(a1f[1], 1.5), "q1.5")

    # G10 linalg — the validation family raises BEFORE any factorisation, so these cells are
    # backend-free: 1-D operand, non-square trailing dims, and the float16-unsupported TypeError.
    m1 = np.arange(3, dtype=np.float64)
    m23 = np.arange(6, dtype=np.float64).reshape(2, 3)
    mh = np.eye(2, dtype=np.float16)
    for op in ("det", "inv"):
        curated(op, {}, [(m1, m1)], lambda op=op: getattr(np.linalg, op)(m1), "1d")
        curated(op, {}, [(m23, m23)], lambda op=op: getattr(np.linalg, op)(m23), "nonsquare")
        curated(op, {}, [(mh, mh)], lambda op=op: getattr(np.linalg, op)(mh), "float16")
    curated("solve", {}, [(m23, m23), (m1, m1)], lambda: np.linalg.solve(m23, m1), "nonsquare")

    # G16 FFT — the n-guard (verbatim `Invalid number of FFT data points (0) specified.`).
    cfft = np.arange(8, dtype=np.float64)
    curated("fft", {"n": 0, "axis": -1, "norm": None}, [(cfft, cfft)],
            lambda: np.fft.fft(cfft, n=0), "n0")
    curated("ifft", {"n": -3, "axis": -1, "norm": None}, [(cfft, cfft)],
            lambda: np.fft.ifft(cfft, n=-3), "nneg")

    distinct = len({(c["op"], c["error"]["type"], c["error"]["text"]) for c in cases})
    dropped = sum(max(0, v - ERROR_INSTANCES_PER_MESSAGE) for v in seen.values())
    print(f"  ({len(cases)} raising cells over {distinct} distinct NumPy messages"
          + (f"; {dropped} instances dropped by the per-message cap" if dropped else "") + ")")
    return cases


# =====================================================================================
# IEEE special-value parity ("specials" tier). The float/complex value pools already
# front-load nan/±inf/-0.0 (layout_catalog._FLOAT_POOL), so the elementwise/reduce tiers
# INCIDENTALLY exercise them — but three gaps motivated a DEDICATED, auditable tier:
#   (a) matmul/dot/outer draw from _mm_fill (clean integer/half ramps, NO specials), so
#       NaN/±inf PROPAGATION through the managed GEMM — the one product path a plain test
#       run takes, since NumSharp.Core ships no BLAS — was never gated against NumPy.
#   (b) the binary PAIR layouts align A[i] with B[i] drawn from the SAME pool in the SAME
#       order, so the cross-operand INTERACTIONS that are the whole point of IEEE arithmetic
#       (inf+(-inf), 0*inf, 0/0, inf/inf, 1**inf, max*max->inf) never occur.
#   (c) per-dtype max/tiny(min-normal) and the smallest SUBNORMAL were absent entirely.
# This tier arranges operands to FORCE those interactions, across float16/float32/float64/
# complex128 and the contiguous/2-D/F-contiguous/strided/negative-stride paths where a
# NaN-dropping SIMD kernel (cf. the clip MAXPS/MINPD bug, W11-A) would hide. It reuses the
# existing OpRegistry op names, so it needs no new harness wiring.
#
# What this proves and what it does NOT: BitDiff tokenizes NaN (any payload -> "NaN") and
# bit-compares ±0.0 / ±inf, so a green tier asserts exactly the CONTRACTUAL part of IEEE
# parity — is-NaN, the sign of a zero, the sign of an infinity — not the (non-contractual)
# NaN payload bits. The matmul operands are deliberately built so every output cell is
# order-independent (a NaN anywhere -> NaN; an inf against all-positive-finite -> +inf; no
# inf-inf cancellation inside a dot), so a divergence there is a real propagation bug, not a
# summation-reassociation artefact of managed-vs-BLAS.
SPECIAL_DTYPES = ["float16", "float32", "float64", "complex128"]


def _float_special_pool(dt):
    """Per-dtype special-value vector: nan, ±inf, ±0, and — sized to the dtype — ±max,
    ±tiny (smallest normal) and ±smallest subnormal, plus a spread of ordinary magnitudes
    that drive the transcendental edges (sqrt(-1), log(0), log(-1), reciprocal(0),
    arcsin(2), (-1)**0.5, tan(pi/2)). Built as float64 then narrowed: nan/±inf/-0 survive
    astype, and the extremes come from np.finfo(dt) so nothing overflows to inf by accident
    (verified: exactly two infinities for every dtype)."""
    dt = np.dtype(dt)
    fi = np.finfo(dt)
    base = [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -8.0, 27.0,
            0.25, -0.25, float(np.pi), float(-np.pi), 100.0, -100.0]
    ext = [float(fi.max), -float(fi.max), float(fi.tiny), -float(fi.tiny),
           float(fi.smallest_subnormal), -float(fi.smallest_subnormal)]
    return np.array(base + ext, dtype=np.float64).astype(dt)


def _complex_special_pool():
    """Complex specials: the float pool paired component-wise with a rolled copy (so a NaN
    real meets an inf imag, etc.), plus the pure combos a component-paired build cannot reach
    (nan+0j, 0+nanj, inf-infj, -inf+nanj, nan+infj) — the cases where npy_c* and
    System.Numerics.Complex most often disagree."""
    f = _float_special_pool("float64")
    z = (f + 1j * np.roll(f, 3)).astype("complex128")
    combos = np.array([complex(np.nan, 0.0), complex(0.0, np.nan), complex(np.nan, np.nan),
                       complex(np.inf, np.inf), complex(np.inf, -np.inf), complex(-np.inf, np.nan),
                       complex(np.nan, np.inf), complex(0.0, -0.0), complex(1.0, np.inf)], dtype="complex128")
    return np.concatenate([z, combos])


def _special_pool(dt):
    return _complex_special_pool() if np.dtype(dt).kind == "c" else _float_special_pool(dt)


def _binary_special_pairs(dt):
    """Aligned (A, B) float vectors whose element-wise pairing FORCES the IEEE interactions
    the ordinary pair layouts never produce: inf+(-inf)=nan, 0*inf=nan, 0/0=nan, inf/inf=nan,
    1**inf, max*max->inf, and the signed-zero / subnormal / tiny boundaries."""
    dt = np.dtype(dt)
    fi = np.finfo(dt)
    mx = float(fi.max); ti = float(fi.tiny); sub = float(fi.smallest_subnormal)
    inf = np.inf; nan = np.nan
    pairs = [(inf, inf), (inf, -inf), (-inf, inf), (-inf, -inf), (0.0, inf), (inf, 0.0), (-0.0, inf), (0.0, -inf),
             (0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0), (nan, 1.0), (1.0, nan), (nan, nan), (nan, inf), (inf, nan),
             (1.0, inf), (1.0, -inf), (inf, 1.0), (-1.0, 0.5), (0.5, -0.5), (2.0, 3.0), (-2.0, 3.0), (2.0, -3.0), (-8.0, 3.0),
             (mx, mx), (mx, 2.0), (2.0, mx), (mx, -mx), (ti, ti), (sub, sub), (ti, 2.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
    A = np.array([a for a, _ in pairs], dtype=np.float64).astype(dt)
    B = np.array([b for _, b in pairs], dtype=np.float64).astype(dt)
    return A, B


def _binary_special_operands(dt):
    if np.dtype(dt).kind == "c":
        z = _complex_special_pool()
        return z, np.roll(z, 5)   # misalign so a nan/inf real meets a different component
    return _binary_special_pairs(dt)


def _special_1d_layouts(v):
    """(tag, base, view) memory-layout variants of the 1-D special vector v, each a genuine
    VIEW sharing base's bytes (describe() validates that). Covers contiguous 1-D/2-D, an
    F-contiguous 2-D, a step-2 strided view and a negative-stride view — the SIMD-full /
    scalar-tail / strided / reversed kernel paths a NaN-dropping SIMD min/max would split on."""
    out = [("contig_1d", v, v)]
    n = len(v); m = n - (n % 2)
    if m >= 4:
        c2 = np.ascontiguousarray(v[:m].reshape(2, m // 2)); out.append(("contig_2d", c2, c2))
        fb = np.ascontiguousarray(v[:m].reshape(m // 2, 2)); out.append(("f_contig_2d", fb, fb.T))
        inter = np.empty(2 * m, dtype=v.dtype); inter[0::2] = v[:m]; inter[1::2] = v[:m][::-1]
        out.append(("strided_1d", inter, inter[0::2]))
    neg = v.copy(); out.append(("negstride_1d", neg, neg[::-1]))
    return out


def _special_pair_layouts(A, B):
    """(tag, baseA, viewA, baseB, viewB) for the binary specials — contiguous, both-strided
    and both-negative-stride. base/view MUST share a buffer, so the negative-stride variant
    reverses ONE copy (an early bug reversed a second copy -> a non-view operand)."""
    out = [("pp_contig", A, A, B, B)]
    nA = len(A) - (len(A) % 2)
    iA = np.empty(2 * nA, dtype=A.dtype); iA[0::2] = A[:nA]; iA[1::2] = A[:nA][::-1]
    iB = np.empty(2 * nA, dtype=B.dtype); iB[0::2] = B[:nA]; iB[1::2] = B[:nA][::-1]
    out.append(("pp_strided", iA, iA[0::2], iB, iB[0::2]))
    na = A.copy(); nb = B.copy()
    out.append(("pp_negstride", na, na[::-1], nb, nb[::-1]))
    return out


def _mm_special_layout(arr, layout):
    """(base, view) for a matmul operand — like _mm_layout, but WITHOUT its
    `assert np.array_equal(view, arr)` self-check, which is vacuously False whenever arr
    holds a NaN. Same construction: an F-contiguous view is the C-contiguous transpose
    viewed back through .T, so base.tobytes() is its raw memory."""
    if layout == "F" and arr.ndim >= 2:
        base = np.ascontiguousarray(arr.T)
        return base, base.T
    base = np.ascontiguousarray(arr)
    return base, base


# The op sets: exactly the elementwise-math / reduction / scan / product ops whose dtype
# family (float + complex) carries IEEE special values. Integer/bool-only ops (bitwise,
# shift) have no NaN and are deliberately absent; NumPy-raising cells (complex floor/ceil/
# trunc/cbrt/floor_divide/mod, complex ORDER comparisons, complex arctan2) are skipped by the
# try/except, exactly as the other gen_* tiers do.
SPECIAL_UNARY_OPS = {
    "negative": np.negative, "positive": np.positive, "abs": np.abs, "sign": np.sign, "sqrt": np.sqrt,
    "cbrt": np.cbrt, "square": np.square, "reciprocal": np.reciprocal, "floor": np.floor, "ceil": np.ceil,
    "trunc": np.trunc, "rint": np.rint, "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
    "exp2": np.exp2, "expm1": np.expm1, "log2": np.log2, "log10": np.log10, "log1p": np.log1p, "sinh": np.sinh,
    "cosh": np.cosh, "tanh": np.tanh, "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "arcsinh": np.arcsinh, "arccosh": np.arccosh, "arctanh": np.arctanh,
    "deg2rad": np.deg2rad, "rad2deg": np.rad2deg, "isnan": np.isnan, "isinf": np.isinf, "isfinite": np.isfinite,
}
SPECIAL_BINARY_OPS = {
    "add": lambda a, b: a + b, "subtract": lambda a, b: a - b, "multiply": lambda a, b: a * b, "divide": lambda a, b: a / b,
    "floor_divide": lambda a, b: a // b, "mod": lambda a, b: a % b, "power": lambda a, b: a ** b,
    "float_power": lambda a, b: np.float_power(a, b), "arctan2": np.arctan2, "heaviside": np.heaviside,
    "maximum": np.maximum, "minimum": np.minimum, "fmax": np.fmax, "fmin": np.fmin,
    "equal": lambda a, b: a == b, "not_equal": lambda a, b: a != b, "less": lambda a, b: a < b, "greater": lambda a, b: a > b,
    "less_equal": lambda a, b: a <= b, "greater_equal": lambda a, b: a >= b, "isclose": np.isclose,
    "logical_and": np.logical_and, "logical_or": np.logical_or, "logical_xor": np.logical_xor,
}
SPECIAL_REDUCE_OPS = {
    "sum": lambda a, ax, kd: np.sum(a, axis=ax, keepdims=kd), "prod": lambda a, ax, kd: np.prod(a, axis=ax, keepdims=kd),
    "min": lambda a, ax, kd: np.min(a, axis=ax, keepdims=kd), "max": lambda a, ax, kd: np.max(a, axis=ax, keepdims=kd),
    "mean": lambda a, ax, kd: np.mean(a, axis=ax, keepdims=kd), "all": lambda a, ax, kd: np.all(a, axis=ax, keepdims=kd),
    "any": lambda a, ax, kd: np.any(a, axis=ax, keepdims=kd), "argmax": lambda a, ax, kd: np.argmax(a, axis=ax, keepdims=kd),
    "argmin": lambda a, ax, kd: np.argmin(a, axis=ax, keepdims=kd), "nansum": lambda a, ax, kd: np.nansum(a, axis=ax, keepdims=kd),
    "nanprod": lambda a, ax, kd: np.nanprod(a, axis=ax, keepdims=kd), "nanmax": lambda a, ax, kd: np.nanmax(a, axis=ax, keepdims=kd),
    "nanmin": lambda a, ax, kd: np.nanmin(a, axis=ax, keepdims=kd), "nanmean": lambda a, ax, kd: np.nanmean(a, axis=ax, keepdims=kd),
}
SPECIAL_SCAN_OPS = {
    "cumsum": lambda a, ax: np.cumsum(a, axis=ax), "cumprod": lambda a, ax: np.cumprod(a, axis=ax),
}


def gen_specials():
    cases = []
    n = 0
    skipped = 0

    def emit(op, params, operands, r, tag, dt):
        nonlocal n
        # Read the shape BEFORE ascontiguousarray (which forces ndim>=1, corrupting 0-D results).
        exp_shape = [int(d) for d in r.shape]
        exp_buf = np.ascontiguousarray(r).tobytes().hex()
        cases.append({
            "id": f"{op}/{tag}/{dt}/{n}",
            "op": op,
            "params": params,
            "operands": operands,
            "expected": {"dtype": r.dtype.name, "shape": exp_shape, "buffer": exp_buf},
            "layout": tag,
            "valueclass": "specials",
        })
        n += 1

    # --- unary elementwise math over the special pool, every layout -----------------------
    for dt in SPECIAL_DTYPES:
        v = _special_pool(dt)
        for tag, base, view in _special_1d_layouts(v):
            od = describe(base, view)
            for opname, f in SPECIAL_UNARY_OPS.items():
                try:
                    r = np.asarray(f(view))
                except Exception:
                    skipped += 1
                    continue
                emit(opname, {}, [od], r, f"un_{tag}", dt)

    # --- binary elementwise math over the FORCED interaction pairs ------------------------
    for dt in SPECIAL_DTYPES:
        A, B = _binary_special_operands(dt)
        for tag, ba, va, bb, vb in _special_pair_layouts(A, B):
            oa = describe(ba, va); ob = describe(bb, vb)
            for opname, f in SPECIAL_BINARY_OPS.items():
                try:
                    r = np.asarray(f(va, vb))
                except Exception:
                    skipped += 1
                    continue
                emit(opname, {}, [oa, ob], r, f"bin_{tag}", dt)

    # --- reductions (incl. the nan* family) over special slices, axis + keepdims ----------
    for dt in SPECIAL_DTYPES:
        v = _special_pool(dt)
        for tag, base, view in _special_1d_layouts(v):
            od = describe(base, view)
            for opname, f in SPECIAL_REDUCE_OPS.items():
                for axis in _axes(view.ndim):
                    for keepdims in (False, True):
                        if opname in ("argmax", "argmin") and axis is None and keepdims:
                            continue  # flat argmax/argmin (long) has no keepdims — mirror gen_reduce
                        try:
                            r = np.asarray(f(view, axis, keepdims))
                        except Exception:
                            skipped += 1
                            continue
                        emit(opname, {"axis": axis, "keepdims": keepdims}, [od], r, f"red_{tag}", dt)

    # --- cumulative scans over special slices ---------------------------------------------
    for dt in SPECIAL_DTYPES:
        v = _special_pool(dt)
        for tag, base, view in _special_1d_layouts(v):
            od = describe(base, view)
            for opname, f in SPECIAL_SCAN_OPS.items():
                for axis in _axes(view.ndim):
                    try:
                        r = np.asarray(f(view, axis))
                    except Exception:
                        skipped += 1
                        continue
                    emit(opname, {"axis": axis}, [od], r, f"scan_{tag}", dt)

    # --- matmul / dot / outer: NaN/inf PROPAGATION through the managed GEMM (the gap) ------
    # A carries a finite row, a NaN row and an inf row; B is all-positive-finite with no
    # zeros, so every output cell is order-independent: finite, NaN (from the NaN row) or
    # +inf (from the inf row). This isolates propagation from summation reassociation.
    for dt in SPECIAL_DTYPES:
        A = np.array([[1, 2, 3], [np.nan, 1, 1], [np.inf, 2, 1]], dtype=np.dtype(dt))
        B = np.array([[1, 2], [2, 1], [1, 1]], dtype=np.dtype(dt))
        for la in ("C", "F"):
            for lb in ("C", "F"):
                bA, vA = _mm_special_layout(A, la); bB, vB = _mm_special_layout(B, lb)
                for op, f in (("matmul", np.matmul), ("dot", np.dot)):
                    r = np.asarray(f(vA, vB))
                    emit(op, {}, [describe(bA, vA), describe(bB, vB)], r, f"mm_{la}{lb}", dt)
        # 1-D inner product (its own kernel) — a NaN, an inf and a finite vector.
        for name, a1, b1 in (("nan", [np.nan, 1, 1], [1, 2, 3]),
                             ("inf", [np.inf, 1, 1], [1, 2, 3]),
                             ("fin", [1, 2, 3], [1, 2, 3])):
            a = np.array(a1, dtype=np.dtype(dt)); b = np.array(b1, dtype=np.dtype(dt))
            r = np.asarray(np.dot(a, b))
            emit("dot", {}, [describe(a, a), describe(b, b)], r, f"mm_vec_{name}", dt)
        # outer product — every cell is one product, so specials map straight through.
        a = np.array([1, np.inf, np.nan, 2], dtype=np.dtype(dt)); b = np.array([1, 2, -1], dtype=np.dtype(dt))
        r = np.asarray(np.outer(a, b))
        emit("outer", {}, [describe(a, a), describe(b, b)], r, "mm_outer", dt)

    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# =====================================================================================
# Truthful-vs-precise ("precision" tier). The vision is BYTE-IDENTICAL parity to NumPy,
# so the gate hierarchy is: bit-exact to NumPy ("precise") PASSES, always, without ever
# consulting mathematical truth — matching NumPy's documented 2.52-ulp-wrong f32 exp IS
# the contract. But when a case DIVERGES from NumPy, the parity bytes alone cannot say
# which side lost precision. Each case here therefore carries a THIRD buffer,
# expected.truth: the correctly-rounded mathematical reference (exact Fraction
# arithmetic for sum/mean/var/cumsum/prod; mpmath at 200-bit precision for std's sqrt
# and for expm1/log1p) — generator-side only, so the no-Python-in-CI rule holds. The
# harness adjudicates a divergence by ULP distance to truth (MisalignedRegistry P1/P2):
# NOT-LESS-truthful than NumPy -> excused as "prefer-precise" PARITY DEBT (the fix is
# to port NumPy's algorithm, exactly as exp/log/sin/cos/tanh were ported — being MORE
# accurate than NumPy is still a divergence, never a win); LESS truthful than NumPy
# (beyond slack) -> genuine precision LOSS, which falls through to the scoped known-bug
# branches or FAILS.
#
# The inputs are precision-ADVERSARIAL, because the ordinary pools cannot stress
# accumulation: at the corpus' 8-36 element sizes a f32 sum can sit at most ~1 ulp from
# exact, while the same values at N=2049 put a naive f32 loop 512 ulp off truth and
# NumPy's pairwise ~2 (measured). Closed-form deterministic arrays, no RNG: wide-
# magnitude sums whose unit elements straddle ulp/2 of the big element, cancellation
# triples, mixed-magnitude pseudo-noise, large-mean variance (the naive E[x^2]-E[x]^2
# killer — NumSharp's two-pass var measured EXACT on it), near-1 products, and the
# expm1/log1p small-|x| band (the S1 defect's home, quantified at ~6.7e7 ulp vs truth).
def _prec_hash01(i):
    """Deterministic pseudo-noise in [0,1) — Knuth multiplicative hash, no RNG state."""
    return ((i * 2654435761) % (2 ** 32)) / (2 ** 32)


def _prec_arrays():
    """(name, 1-D ndarray) adversarial operands. Values are what the dtype actually
    STORES (astype applied), so the exact-rational truth taken from them is the truth of
    the array the ops see, construction rounding excluded."""
    out = []
    out.append(("wide", np.concatenate([np.array([1e8], np.float32),
                                        np.ones(2048, np.float32)])))          # ones < ulp(1e8)/2
    out.append(("cancel", np.tile(np.array([2.0 ** 24, 1.0, -2.0 ** 24], np.float32), 342)))
    out.append(("wide", np.concatenate([np.array([2.0 ** 53]),
                                        np.ones(2047)]).astype(np.float64)))   # ones == ulp/2 ties
    out.append(("mixed", np.array([(_prec_hash01(i) - 0.5) * 2.0 ** ((i * 7) % 30)
                                   for i in range(1024)], np.float64)))
    out.append(("largemean", (np.float64(1e8) + (np.arange(1024) % 2)).astype(np.float64)))
    out.append(("largemean", np.array([1000.0 + _prec_hash01(i) for i in range(512)], np.float32)))
    return out


def _prec_prod_array(dt):
    """Near-1 factors: every multiply rounds, so the accumulation order is observable
    while the product stays in range. Exact rational product is the truth."""
    return np.array([1.0 + (((i * 37) % 61) - 30) * 2.0 ** -20 for i in range(64)], np.dtype(dt))


def _prec_band_array(dt):
    """The expm1/log1p small-|x| band plus anchors — finite only (non-finite semantics
    live in the specials tier; this tier is about ACCURACY on defined inputs)."""
    fi = np.finfo(np.dtype(dt))
    vals = [1e-30, -1e-30, 1e-25, 1e-20, -1e-20, 1e-15, 1e-12, -1e-12, 1e-10, 1e-8,
            -1e-8, 3e-8, 1e-6, -1e-6, 1e-4, 1e-3, -1e-3, 0.03125, -0.03125, 0.5, -0.5,
            1.0, 2.0, -0.9999, 0.0, -0.0, float(fi.tiny), -float(fi.tiny),
            float(fi.smallest_subnormal), -float(fi.smallest_subnormal)]
    return np.array(vals, np.float64).astype(dt)


def _prec_layouts(v):
    """contig + negstride views — the two traversal orders the reduce kernels take; the
    exact-rational truth is order-independent, NumPy's expected is computed per view."""
    neg = v.copy()
    return [("contig_1d", v, v), ("negstride_1d", neg, neg[::-1])]


def gen_precision():
    try:
        import mpmath as mp
    except ImportError:
        print("gen_precision needs mpmath (pip install mpmath) — generator-time only, never CI")
        sys.exit(2)
    mp.mp.prec = 200
    from fractions import Fraction

    def fr(view):
        return [Fraction(float(x)) for x in view.tolist()]   # exact rational of each stored value

    def round1(x, dt):
        """ONE float64 rounding of an exact value, then the dtype cast. For float32 this
        double-rounds (<=1 ulp off correctly-rounded) — inside the harness' +8 slack."""
        return np.array([float(x)], dtype=np.float64).astype(dt)

    def t_scalar(x, dt):
        return round1(x, dt).reshape(())

    def t_cumsum(view, dt):
        run, out = Fraction(0), []
        for f in fr(view):
            run += f
            out.append(float(run))
        return np.array(out, np.float64).astype(dt)

    def t_std(view, dt):
        fs = fr(view)
        m = sum(fs) / len(fs)
        var = sum((f - m) ** 2 for f in fs) / len(fs)
        s = mp.sqrt(mp.mpf(var.numerator) / mp.mpf(var.denominator))
        return t_scalar(float(s), dt)

    def t_unary(view, dt, f):
        return np.array([float(f(mp.mpf(float(x)))) for x in view.tolist()],
                        np.float64).astype(dt)

    cases = []
    n = 0

    def emit(op, params, operands, r, truth, tag, dt, arrname):
        nonlocal n
        truth = np.asarray(truth).astype(r.dtype)
        assert truth.shape == r.shape, f"truth shape {truth.shape} != result {r.shape} ({op}/{arrname})"
        cases.append({
            "id": f"{op}/{tag}/{dt}/{arrname}/{n}",
            "op": op,
            "params": params,
            "operands": operands,
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex(),
                         "truth": np.ascontiguousarray(truth).tobytes().hex()},
            "layout": tag,
            "valueclass": "precision",
        })
        n += 1

    # --- accumulation: sum / mean / var / std / cumsum over the adversarial arrays -----
    for arrname, arr in _prec_arrays():
        dt = arr.dtype
        for tag, base, view in _prec_layouts(arr):
            od = [describe(base, view)]
            fs = fr(view)
            m = sum(fs) / len(fs)
            rk = {"axis": None, "keepdims": False}
            emit("sum", rk, od, np.asarray(np.sum(view)), t_scalar(sum(fs), dt), tag, dt.name, arrname)
            emit("mean", rk, od, np.asarray(np.mean(view)), t_scalar(m, dt), tag, dt.name, arrname)
            emit("var", rk, od, np.asarray(np.var(view)),
                 t_scalar(sum((f - m) ** 2 for f in fs) / len(fs), dt), tag, dt.name, arrname)
            emit("std", rk, od, np.asarray(np.std(view)), t_std(view, dt), tag, dt.name, arrname)
            emit("cumsum", {"axis": None}, od, np.asarray(np.cumsum(view)),
                 t_cumsum(view, dt), tag, dt.name, arrname)

    # --- near-1 products ---------------------------------------------------------------
    for dt in ("float32", "float64"):
        arr = _prec_prod_array(dt)
        for tag, base, view in _prec_layouts(arr):
            od = [describe(base, view)]
            p = Fraction(1)
            for f in fr(view):
                p *= f
            emit("prod", {"axis": None, "keepdims": False}, od,
                 np.asarray(np.prod(view)), t_scalar(p, np.dtype(dt)), tag, dt, "near1")

    # --- expm1 / log1p over the small-|x| band -----------------------------------------
    for dt in ("float32", "float64"):
        arr = _prec_band_array(dt)
        for tag, base, view in _prec_layouts(arr):
            od = [describe(base, view)]
            emit("expm1", {}, od, np.expm1(view), t_unary(view, np.dtype(dt), mp.expm1),
                 tag, dt, "band")
            emit("log1p", {}, od, np.log1p(view), t_unary(view, np.dtype(dt), mp.log1p),
                 tag, dt, "band")

    # --- AXIS reductions over 2-D adversarial arrays -----------------------------------
    # The flat cases above never touch the axis kernels (Reduction.Axis.*), which are
    # different code AND different NumPy behavior: NumPy's axis-0 (outer-axis) reduction
    # is a NAIVE sequential accumulation per column — on the wide-magnitude input it
    # loses ALL the unit elements (1024 ULP from truth) — while axis-1 runs pairwise
    # (~8 ULP). Probed: NumSharp bit-matches BOTH, including reproducing the naive
    # axis-0 order. These cases pin that parity and protect the prefer-precise policy:
    # a future "improved" axis accumulation would diverge from NumPy and must surface
    # as P1 parity debt, not pass silently. The transposed view routes the SAME logical
    # reduction through the strided source path.
    def t_axis_slices(view, axis, dt, stat):
        outs = []
        m = view.shape[1] if axis == 0 else view.shape[0]
        for j in range(m):
            sl = view[:, j] if axis == 0 else view[j, :]
            fs = fr(sl)
            mean = sum(fs) / len(fs)
            if stat == "sum":
                outs.append(float(sum(fs)))
            elif stat == "mean":
                outs.append(float(mean))
            elif stat == "var":
                outs.append(float(sum((f - mean) ** 2 for f in fs) / len(fs)))
            else:  # std
                var = sum((f - mean) ** 2 for f in fs) / len(fs)
                outs.append(float(mp.sqrt(mp.mpf(var.numerator) / mp.mpf(var.denominator))))
        return np.array(outs, np.float64).astype(dt)

    def axis_cases(arr, axis, tag, arrname):
        dt = arr.dtype
        base = np.ascontiguousarray(arr)
        od = [describe(base, base)]
        for stat, f in (("sum", np.sum), ("mean", np.mean), ("var", np.var), ("std", np.std)):
            for kd in ((False, True) if stat == "sum" else (False,)):
                r = np.asarray(f(base, axis=axis, keepdims=kd))
                truth = t_axis_slices(base, axis, dt, stat).reshape(r.shape)
                emit(stat, {"axis": axis, "keepdims": kd}, od, r, truth, tag, dt.name, arrname)

    for dtn, big in (("float64", 2.0 ** 53), ("float32", 1e8)):
        ndt = np.dtype(dtn)
        colwise = np.ones((2049, 4), ndt); colwise[0, :] = big     # each COLUMN adversarial
        rowwise = np.ones((4, 2049), ndt); rowwise[:, 0] = big     # each ROW adversarial
        axis_cases(colwise, 0, "axis0_2d", "wide")
        axis_cases(rowwise, 1, "axis1_2d", "wide")
        # transposed view: logical == rowwise, memory == colwise (strided reduce source)
        baseT = np.ascontiguousarray(colwise)
        odT = [describe(baseT, baseT.T)]
        for stat, f in (("sum", np.sum), ("var", np.var)):
            r = np.asarray(f(baseT.T, axis=1))
            truth = t_axis_slices(baseT.T, 1, ndt, stat).reshape(r.shape)
            emit(stat, {"axis": 1, "keepdims": False}, odT, r, truth, "transposed_2d", dtn, "wide")
        # cumsum along each axis (smaller N so the full-array expected/truth stay lean)
        cw = np.ones((257, 4), ndt); cw[0, :] = big
        rw = np.ones((4, 257), ndt); rw[:, 0] = big
        for arr2, ax, tag in ((cw, 0, "axis0_2d"), (rw, 1, "axis1_2d")):
            b2 = np.ascontiguousarray(arr2)
            r = np.asarray(np.cumsum(b2, axis=ax))
            m = b2.shape[1] if ax == 0 else b2.shape[0]
            cols = []
            for j in range(m):
                sl = b2[:, j] if ax == 0 else b2[j, :]
                run, outs = Fraction(0), []
                for fv in fr(sl):
                    run += fv
                    outs.append(float(run))
                cols.append(outs)
            t = np.empty_like(r, dtype=np.float64)
            for j in range(m):
                if ax == 0:
                    t[:, j] = cols[j]
                else:
                    t[j, :] = cols[j]
            emit("cumsum", {"axis": ax}, [describe(b2, b2)], r, t.astype(ndt), tag, dtn, "wide")

    return cases


# =====================================================================================
# np.random byte-parity ("random_parity" tiers). The documented claim — MT19937 with
# 1-to-1 seed/state parity, "byte-identical sequences to NumPy 2.4.2" — was guarded only
# by STATISTICAL tests (mean within 0.01), which would pass with a completely different
# generator. These tiers pin the actual seeded streams: seed -> draw -> record bytes.
#
# TWO FILES, split by libm dependence, because CI replays on three OSes:
#   * random_parity.jsonl — the PORTABLE subset: distributions whose math is pure
#     MT19937 bit manipulation + exactly-rounded IEEE arithmetic (uniform, rand,
#     random_sample, randint, permutation, shuffle, choice). Bit-exact on every host;
#     hard-gated everywhere.
#   * random_parity_host.jsonl — every distribution whose transform consumes libm
#     (log/exp/pow/cos in the gauss polar method, exponential inversion, gamma/zipf/
#     poisson/binomial REJECTION loops — where a 1-ulp libm difference can flip an
#     accept/reject decision and shift the whole stream). NumPy calls the CRT and
#     NumSharp calls Math.* (the same CRT on Windows, glibc elsewhere), so these bytes
#     are win-amd64-authored: the C# gate runs them hard on Windows and reports
#     Inconclusive elsewhere (the matmul_parity pattern; same class as the MSVC cast
#     pins in "Host-dependent values").
#
# int-output distributions (randint/permutation/poisson/binomial/...) are ALSO
# host-dtype-pinned by NumPy itself: legacy RandomState uses C long — int32 on this
# win-amd64 authoring host, int64 on Linux/macOS. NumSharp models the LP64 long (int64)
# for every one of them, so the corpus records them widened to int64 (_RND_INT64_CAST
# below; the values fit, since every bound here is far below 2**31). Sequential-draw
# cases ("draws": 2) pin stream ADVANCEMENT, not just the first block.
_RND_SEEDS = [42, 987654321]
_RND_SIZES = [[7], [2, 3]]

# (dist, args) — generic dists callable as np.random.<dist>(*args, size=...).
_RND_PORTABLE_GENERIC = [
    ("uniform", [0.0, 1.0]), ("uniform", [-3.0, 7.0]),
]
_RND_HOST_GENERIC = [
    ("normal", [0.0, 1.0]), ("normal", [5.0, 2.5]),
    ("lognormal", [0.0, 1.0]),
    ("exponential", [1.0]), ("exponential", [2.5]),
    ("standard_t", [3.5]),
    ("standard_gamma", [2.0]), ("standard_gamma", [0.5]),   # shape<1: different sampler branch
    ("gamma", [2.0, 3.0]), ("gamma", [2.0, 1.0]),   # shape>=1, scale path included
    ("gamma", [0.5, 2.0]),                          # shape<1: the Johnk/Ahrens-Dieter branch (uncarved 2026-09-25)
    ("beta", [2.0, 3.0]), ("beta", [0.5, 0.5]),
    ("chisquare", [3.0]),
    ("gumbel", [0.5, 2.0]),
    ("laplace", [0.0, 1.0]),
    ("logistic", [0.0, 1.0]),
    ("power", [2.5]),
    ("rayleigh", [1.5]),
    ("triangular", [0.0, 3.0, 10.0]),
    ("vonmises", [0.5, 2.0]),
    ("wald", [3.0, 2.0]),
    ("weibull", [1.79]),
    ("poisson", [5.0]), ("poisson", [0.3]),
    ("poisson", [15.0]),                            # lam>=10: PTRS transformed rejection
    ("geometric", [0.35]),
    ("geometric", [0.1]),                           # p<1/3: the legacy inversion branch
    ("zipf", [3.0]),
    ("logseries", [0.6]),
    ("noncentral_chisquare", [3.0, 1.5]),
    ("noncentral_f", [5.0, 7.0, 1.5]),
    # Uncarved 2026-09-25 — the legacy samplers are now line-by-line ports of legacy-distributions.c:
    ("f", [5.0, 7.0]),
    ("pareto", [3.0]),
    ("binomial", [10, 0.35]),                       # n*p <= 30: legacy inversion
    ("binomial", [100, 0.4]),                       # n*p > 30: BTPE
    ("negative_binomial", [5.0, 0.4]),
    ("vonmises", [0.5, 1e7]),                       # large kappa: the legacy code has no wrapped-normal fallback
]
# Stream-advancement pins: draw the same spec twice, record the SECOND block.
_RND_DRAWS2 = {"uniform", "randint", "normal", "standard_gamma", "poisson",
               "binomial", "standard_cauchy"}   # binomial: the setup cache carried across calls; cauchy: the gauss cache

# CARVED on arrival (2026-08-14), UNCARVED 2026-09-25: eight samplers whose STREAM diverged
# from NumPy's (a different algorithm / draw order / accept-reject boundary, not mere
# rounding). Every legacy sampler is now a line-by-line port of NumPy's
# legacy-distributions.c (NumPyRandom.LegacyDistributions.cs / Distributions.cs), so seven
# are back in the tier; only multivariate_normal stays carved — it is byte-identical only
# with a LAPACK backend supplying NumPy's gesdd (the managed Jacobi fallback cannot
# reproduce LAPACK's singular-vector signs), pinned in OpenBugs.Random.cs. The findings:
#   * gamma(shape<1, scale)    — gross divergence while standard_gamma(shape<1) AND
#     gamma(shape>=1, any scale) match byte-for-byte: the two-arg gamma routes shape<1
#     differently than NumSharp's own (correct) standard_gamma.
#   * f(dfnum, dfden)          — not NumPy's (chisq/df)/(chisq/df) composition.
#   * pareto(a)                — not NumPy's expm1(standard_exponential/a).
#   * standard_cauchy()        — not NumPy's gauss/gauss ratio.
#   * binomial(n, p)           — counts drift on BOTH internal algorithms (inversion at
#     small n*p and BTPE at large): accept/reject boundaries land differently.
#   * negative_binomial(n, p)  — poisson(gamma(n, (1-p)/p)) chain: occasional count
#     flips downstream of ~ULP lambda differences (incl. one gross 21-vs-29).
#   * multinomial(n, pvals)    — the internal per-category binomial loop consumes the
#                                stream differently (values gross-diverge; dtype int32 matches).
#   * multivariate_normal      — different factorization/transform (sign flips + values).
# The small-ULP samplers (chisquare/wald/noncentral_f/dirichlet — arithmetic-ordering
# noise on an IDENTICAL stream, measured ≤5/≤24/≤3/≤3 ULP) are EXACT since the port (the
# Marsaglia constant, the legacy wald spelling and dirichlet's reciprocal multiply), so the
# R1 registry envelope that excused them is gone: a single-ULP regression fails the tier.
#
# int-output samplers: legacy RandomState returns C long — int32 on win-amd64, int64 on
# Linux — and NumSharp models ONE width for all of them, the LP64 int64 (since the
# 2026-09-27 type-parity audit: randint's default dtype, random_integers, permutation(int),
# choice's indices and multinomial's counts moved from the win-amd64 int32 to int64 to join
# the discrete samplers, which were already int64). The corpus records every one widened
# to int64 so the VALUE stream stays hard-gated and the dtype policy is documented here
# rather than silently failing per-host. (2026-09-25 to 2026-09-27 randint, permutation,
# choice and multinomial were recorded UNWIDENED, matching NumSharp's int32 of that time.)
_RND_INT64_CAST = {"poisson", "zipf", "logseries", "hypergeometric", "geometric", "binomial", "negative_binomial",
                   "randint", "permutation", "choice", "multinomial"}


def gen_random_parity():
    portable = []
    host = []
    n = 0

    def emit(into, dist, params, r):
        nonlocal n
        r = np.asarray(r)
        if dist in _RND_INT64_CAST:
            r = r.astype(np.int64)   # widen C-long (int32 here) to NumSharp's fixed int64 — see above
        into.append({
            "id": f"rnd/{dist}/seed{params['seed']}/{n}",
            "op": "rnd",
            "params": params,
            "operands": [],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": "rnd",
            "valueclass": "stream",
        })
        n += 1

    def run(dist, args, seed, size, draws=1, extra=None):
        np.random.seed(seed)
        r = None
        for _ in range(draws):
            if dist == "rand":
                r = np.random.rand(*size)
            elif dist == "randn":
                r = np.random.randn(*size)
            elif dist == "random_sample":
                r = np.random.random_sample(tuple(size))
            elif dist == "randint":
                r = np.random.randint(int(args[0]), int(args[1]), tuple(size))
            elif dist == "permutation":
                r = np.random.permutation(int(args[0]))
            elif dist == "shuffle":
                # arange default dtype (int64 in NumPy 2.x, matching NumSharp's np.arange);
                # the STREAM consumption is dtype-independent.
                r = np.arange(int(args[0]))
                np.random.shuffle(r)
            elif dist == "choice":
                r = np.random.choice(int(args[0]), tuple(size),
                                     p=extra.get("p") if extra else None)
            elif dist == "multinomial":
                r = np.random.multinomial(int(args[0]), extra["pvals"], tuple(size))
            elif dist == "dirichlet":
                r = np.random.dirichlet(extra["alpha"], tuple(size))
            elif dist == "multivariate_normal":
                d = len(extra["mean"])
                cov = np.array(extra["cov"]).reshape(d, d)
                r = np.random.multivariate_normal(extra["mean"], cov, tuple(size))
            else:
                r = getattr(np.random, dist)(*args, tuple(size))
        return r

    def cases_for(into, dist, args, sized=True, extra=None):
        specs = []
        if sized:
            specs.append((_RND_SEEDS[0], _RND_SIZES[0]))
            specs.append((_RND_SEEDS[0], _RND_SIZES[1]))
            specs.append((_RND_SEEDS[1], _RND_SIZES[0]))
        else:
            specs.append((_RND_SEEDS[0], None))
            specs.append((_RND_SEEDS[1], None))
        for seed, size in specs:
            params = {"dist": dist, "seed": seed, "size": size, "args": args}
            if extra:
                params.update(extra)
            emit(into, dist, params, run(dist, args, seed, size or [], extra=extra))
        if dist in _RND_DRAWS2:
            seed, size = _RND_SEEDS[0], _RND_SIZES[0]
            params = {"dist": dist, "seed": seed, "size": size, "args": args, "draws": 2}
            if extra:
                params.update(extra)
            emit(into, dist, params, run(dist, args, seed, size, draws=2, extra=extra))

    # --- portable: pure MT19937 bits + exactly-rounded arithmetic ---------------------
    for dist, args in _RND_PORTABLE_GENERIC:
        cases_for(portable, dist, args)
    cases_for(portable, "rand", [])
    cases_for(portable, "random_sample", [])
    for lo, hi in ((0, 100), (-50, 50), (0, 2)):
        cases_for(portable, "randint", [lo, hi])
    for nperm in (10, 1):
        cases_for(portable, "permutation", [nperm], sized=False)
    cases_for(portable, "shuffle", [10], sized=False)
    cases_for(portable, "choice", [10])
    cases_for(portable, "choice", [4], extra={"p": [0.1, 0.2, 0.3, 0.4]})

    # Public state-management surface. These are deliberately their OWN op keys rather than
    # being inferred from rnd: seed is checked through a portable draw, get_state through a
    # canonical full-state trace, and set_state by restoring a captured 624-word state before a
    # draw. This catches a state shape/position/cache drift that identical first draws can hide.
    for seed in [0, 42, 2 ** 31, 2 ** 32 - 1]:
        np.random.seed(seed)
        r = np.random.random_sample((8,))
        portable.append(_case("seed", {"seed": seed, "size": [8]}, [], _arr_expected(r),
                              "rnd/state", "stream", cid=f"seed/{seed}/{n}"))
        n += 1

    def state_text(state):
        alg, key, pos, has_gauss, cached = state
        bits = int(np.asarray(cached, dtype=np.float64).view(np.uint64))
        return f"{alg}|{int(pos)}|{int(has_gauss)}|{bits:016x}|" + \
               ",".join(str(int(x)) for x in key)

    for seed, draws in [(0, 0), (1, 1), (42, 5), (2 ** 32 - 1, 17)]:
        np.random.seed(seed)
        if draws:
            np.random.random_sample((draws,))
        portable.append(_case("get_state", {"seed": seed, "draws": draws}, [],
                              {"kind": "text", "value": state_text(np.random.get_state())},
                              "rnd/state", "stream", cid=f"get_state/{seed}/{draws}/{n}"))
        n += 1

    for source_seed, advance in [(0, 0), (42, 1), (24680, 7), (2 ** 32 - 1, 17)]:
        np.random.seed(source_seed)
        if advance:
            np.random.random_sample((advance,))
        state = np.random.get_state()
        np.random.set_state(state)
        restored = np.random.random_sample((8,))
        portable.append(_case("set_state",
                              {"pos": int(state[2]), "has_gauss": int(state[3]),
                               "cached_gaussian": float(state[4]), "size": [8]},
                              [describe(state[1], state[1])], _arr_expected(restored),
                              "rnd/state", "stream",
                              cid=f"set_state/{source_seed}/{advance}/{n}"))
        n += 1

    # --- host-libm: transform / rejection samplers ------------------------------------
    # multivariate_normal stays CARVED (see the _RND_INT64_CAST comment block) — exact only
    # with a LAPACK backend, pinned in OpenBugs.Random.cs.
    for dist, args in _RND_HOST_GENERIC:
        cases_for(host, dist, args)
    cases_for(host, "randn", [])
    cases_for(host, "standard_normal", [])
    cases_for(host, "standard_exponential", [])
    cases_for(host, "hypergeometric", [10, 7, 8])
    cases_for(host, "dirichlet", [], extra={"alpha": [2.0, 3.0, 5.0]})
    # Uncarved 2026-09-25:
    cases_for(host, "hypergeometric", [100, 200, 50])   # nsample > 10: the HRUA branch
    cases_for(host, "standard_cauchy", [])
    cases_for(host, "multinomial", [20], extra={"pvals": [0.2, 0.3, 0.5]})
    # Array-valued (broadcast) parameters of every legacy sampler — see _gen_random_broadcast.
    bp, bh, n = _gen_random_broadcast("legacy", n)
    portable += bp
    host += bh
    return portable, host


# =====================================================================================
# CBLAS product family ("products" tier). inner / vdot / vecdot / matvec / vecmat /
# tensordot / linalg.multi_dot / linalg.matrix_power had NO value gate at all — only
# their ERROR contracts were unit-tested — yet they carry exactly the cells that regress
# silently: vdot/vecdot conjugate the FIRST operand (complex), vecdot reduces in the
# LOOP dtype (int32 stays int32, not NEP50's int64), tensordot's axes forms, and
# matrix_power's binary-exponentiation dtype rules.
#
# Two value classes:
#   * SMALL-EXACT (the bulk): _mm_fill operands (small ints / halves / tiny complex) at
#     contraction depth <= 4, where every float sum is exact and therefore
#     order-independent — bit-equal is expected even against NumPy's BLAS-backed
#     dot/inner/vdot. Integer dtypes are modular and exact by construction.
#   * DEEP-TRUTH (f32/f64, K=2049): mixed-magnitude pseudo-noise where summation order
#     shows. NumPy routes inner/vdot through BLAS while NumSharp runs managed kernels,
#     so bit-parity is not the contract there — each case carries expected.truth (exact
#     Fraction dot products) and the P1/P2 prefer-precise branches adjudicate exactly as
#     in the precision tier.
PRODUCT_DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
                  "int64", "uint64", "float16", "float32", "float64", "complex128"]


def _prod_noise(k):
    """Deterministic mixed-magnitude values (2^0..2^9 spread) — enough for summation
    order to show at K=2049 without overflowing a float32 dot."""
    return np.array([(_prec_hash01(i) - 0.5) * 2.0 ** ((i * 5) % 10) for i in range(k)],
                    np.float64)


def gen_products():
    from fractions import Fraction
    cases = []
    n = 0
    skipped = 0

    def emit(op, params, pairs, r, tag, dt, truth=None):
        nonlocal n
        exp = {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
               "buffer": np.ascontiguousarray(r).tobytes().hex()}
        if truth is not None:
            truth = np.asarray(truth).astype(r.dtype)
            assert truth.shape == r.shape
            exp["truth"] = np.ascontiguousarray(truth).tobytes().hex()
        cases.append({
            "id": f"{op}/{tag}/{dt}/{n}",
            "op": op,
            "params": params,
            "operands": [describe(b, v) for (b, v) in pairs],
            "expected": exp,
            "layout": tag,
            "valueclass": "products",
        })
        n += 1

    def try_emit(op, params, pairs, f, tag, dt, truth=None):
        nonlocal skipped
        try:
            r = np.asarray(f())
        except Exception:
            skipped += 1   # NumPy raises (e.g. matrix_power on bool) — error parity is elsewhere
            return
        emit(op, params, pairs, r, tag, dt, truth)

    # --- small-exact family across every dtype ----------------------------------------
    for dt in PRODUCT_DTYPES:
        A34 = _mm_fill((3, 4), dt)
        B24 = _mm_fill((2, 4), dt)
        M43 = _mm_fill((4, 3), dt)
        v4 = _mm_fill((4,), dt)
        w4 = v4[::-1].copy()
        C = lambda a: (np.ascontiguousarray(a), np.ascontiguousarray(a))

        try_emit("inner", {}, [C(A34), C(B24)], lambda: np.inner(A34, B24), "2d", dt)
        try_emit("inner", {}, [C(v4), C(w4)], lambda: np.inner(v4, w4), "1d", dt)
        try_emit("vdot", {}, [C(v4), C(w4)], lambda: np.vdot(v4, w4), "1d", dt)
        try_emit("vdot", {}, [C(A34), C(np.ascontiguousarray(_mm_fill((3, 4), dt)[::-1]))],
                 lambda: np.vdot(A34, _mm_fill((3, 4), dt)[::-1]), "2d_flat", dt)
        try_emit("vecdot", {"axis": -1}, [C(B24), C(np.ascontiguousarray(B24[::-1]))],
                 lambda: np.vecdot(B24, B24[::-1], axis=-1), "ax-1", dt)
        try_emit("vecdot", {"axis": 0}, [C(M43), C(np.ascontiguousarray(M43[::-1]))],
                 lambda: np.vecdot(M43, M43[::-1], axis=0), "ax0", dt)
        try_emit("vecdot", {"keepdims": True}, [C(B24), C(np.ascontiguousarray(B24[::-1]))],
                 lambda: np.vecdot(B24, B24[::-1], keepdims=True), "kd", dt)
        try_emit("matvec", {}, [C(A34), C(v4)], lambda: np.matvec(A34, v4), "2d", dt)
        bA = _mm_fill((2, 3, 4), dt); bv = _mm_fill((2, 4), dt)
        try_emit("matvec", {}, [C(bA), C(bv)], lambda: np.matvec(bA, bv), "batched", dt)
        try_emit("vecmat", {}, [C(v4), C(M43)], lambda: np.vecmat(v4, M43), "2d", dt)
        bM = _mm_fill((2, 4, 3), dt)
        try_emit("vecmat", {}, [C(bv), C(bM)], lambda: np.vecmat(bv, bM), "batched", dt)
        T23 = _mm_fill((2, 3), dt); T34 = _mm_fill((3, 4), dt); T23b = _mm_fill((2, 3), dt)[::-1].copy()
        try_emit("tensordot", {"axes": 1}, [C(T23), C(T34)], lambda: np.tensordot(T23, T34, 1), "int1", dt)
        try_emit("tensordot", {"axes": 2}, [C(T23), C(T23b)], lambda: np.tensordot(T23, T23b, 2), "int2", dt)
        try_emit("tensordot", {"axes": 0}, [C(v4), C(_mm_fill((3,), dt))],
                 lambda: np.tensordot(v4, _mm_fill((3,), dt), 0), "int0", dt)
        try_emit("tensordot", {"axesA": [1], "axesB": [0]}, [C(T23), C(T34)],
                 lambda: np.tensordot(T23, T34, ([1], [0])), "pair", dt)
        try_emit("tensordot", {"axesA": [0, 1], "axesB": [0, 1]}, [C(T23), C(T23b)],
                 lambda: np.tensordot(T23, T23b, ([0, 1], [0, 1])), "pair2", dt)
        D1 = _mm_fill((2, 3), dt); D2 = _mm_fill((3, 4), dt); D3 = _mm_fill((4, 2), dt)
        try_emit("multi_dot", {}, [C(D1), C(D2), C(D3)],
                 lambda: np.linalg.multi_dot([D1, D2, D3]), "chain3", dt)
        vfirst = _mm_fill((3,), dt)
        try_emit("multi_dot", {}, [C(vfirst), C(D2), C(D3)],
                 lambda: np.linalg.multi_dot([vfirst, D2, D3]), "vec_first", dt)
        S33 = _mm_fill((3, 3), dt)
        for pw in (0, 2, 3):
            try_emit("matrix_power", {"n": pw}, [C(S33)],
                     lambda: np.linalg.matrix_power(S33, pw), f"n{pw}", dt)

        # kron — Kronecker product. Values are plain a[i]*b[j], so the clean _mm_fill ramps map
        # straight through (exact for every product dtype; integer/half products don't overflow at
        # these sizes). Covers 1d/2d, both mixed-rank orders (the ndmin promotion), a 3-D operand,
        # and a 0-d scalar operand (kron's multiply shortcut). Small sizes take the direct dispatch
        # path; the tile fast-path is bit-identical (unit-test-pinned at size).
        K22 = _mm_fill((2, 2), dt); K222 = _mm_fill((2, 2, 2), dt)
        ksc = np.array(_mm_fill((2,), dt)[1])                       # 0-d scalar of this dtype
        try_emit("kron", {}, [C(v4), C(w4)], lambda: np.kron(v4, w4), "1d", dt)
        try_emit("kron", {}, [C(T23), C(T34)], lambda: np.kron(T23, T34), "2d", dt)
        try_emit("kron", {}, [C(v4), C(T23)], lambda: np.kron(v4, T23), "1d_2d", dt)
        try_emit("kron", {}, [C(T23), C(v4)], lambda: np.kron(T23, v4), "2d_1d", dt)
        try_emit("kron", {}, [C(K222), C(K22)], lambda: np.kron(K222, K22), "3d_2d", dt)
        try_emit("kron", {}, [C(T23), C(ksc)], lambda: np.kron(T23, ksc), "scalar_b", dt)

    # Array-API norm spellings. Use Pythagorean fixtures whose sqrt is exact, isolating the
    # axis/keepdims/dtype contracts from platform-libm noise and from the SVD-only matrix orders.
    for dt in ("int32", "float32", "float64", "complex128"):
        d = np.dtype(dt)
        nv = np.array([3, 4], dtype=d)
        nm = np.array([[3, 0], [0, 4]], dtype=d)
        C = lambda a: (np.ascontiguousarray(a), np.ascontiguousarray(a))
        try_emit("vector_norm", {"axis": None, "keepdims": False}, [C(nv)],
                 lambda: np.linalg.vector_norm(nv), "default", dt)
        try_emit("vector_norm", {"axis": 0, "keepdims": True}, [C(nv)],
                 lambda: np.linalg.vector_norm(nv, axis=0, keepdims=True), "axis0_kd", dt)
        try_emit("matrix_norm", {"keepdims": False, "ord": "fro"}, [C(nm)],
                 lambda: np.linalg.matrix_norm(nm, ord="fro"), "fro", dt)
        try_emit("matrix_norm", {"keepdims": True, "ord": 1}, [C(nm)],
                 lambda: np.linalg.matrix_norm(nm, ord=1, keepdims=True), "ord1_kd", dt)

    # --- F-layout spot checks (the stride-aware read path) ----------------------------
    for dt in ("int32", "float64"):
        A = _mm_fill((3, 4), dt); B = _mm_fill((2, 4), dt)
        baseA, viewA = _mm_layout(A, "F")
        try_emit("inner", {}, [(baseA, viewA), (np.ascontiguousarray(B), B)],
                 lambda: np.inner(viewA, B), "F", dt)
        # kron reading a non-contiguous (F) operand — exercises the reshape-materialisation path.
        try_emit("kron", {}, [(baseA, viewA), (np.ascontiguousarray(B), B)],
                 lambda: np.kron(viewA, B), "F", dt)

    # --- deep-truth (f32/f64, K=2049): Fraction-exact references ----------------------
    K = 2049
    for dt in ("float32", "float64"):
        ndt = np.dtype(dt)
        a2 = _prod_noise(2 * K).reshape(2, K).astype(ndt)
        b3 = _prod_noise(3 * K)[::-1].reshape(3, K).astype(ndt)
        v = _prod_noise(K).astype(ndt)
        w = _prod_noise(K)[::-1].copy().astype(ndt)
        mk = np.ascontiguousarray(b3.T)      # (K,3)

        def frdot(x, y):
            return sum(Fraction(float(a)) * Fraction(float(b)) for a, b in zip(x.tolist(), y.tolist()))

        t = np.array([[float(frdot(a2[i], b3[j])) for j in range(3)] for i in range(2)])
        try_emit("inner", {}, [C(a2), C(b3)], lambda: np.inner(a2, b3), "deep", dt, truth=t)
        t = np.array(float(frdot(v, w)))
        try_emit("vdot", {}, [C(v), C(w)], lambda: np.vdot(v, w), "deep", dt, truth=t)
        t = np.array([float(frdot(a2[i], a2[1 - i])) for i in range(2)])
        try_emit("vecdot", {"axis": -1}, [C(a2), C(np.ascontiguousarray(a2[::-1]))],
                 lambda: np.vecdot(a2, a2[::-1], axis=-1), "deep", dt, truth=t)
        t = np.array([float(frdot(b3[i], v)) for i in range(3)])
        try_emit("matvec", {}, [C(b3), C(v)], lambda: np.matvec(b3, v), "deep", dt, truth=t)
        t = np.array([float(frdot(v, mk[:, j])) for j in range(3)])
        try_emit("vecmat", {}, [C(v), C(mk)], lambda: np.vecmat(v, mk), "deep", dt, truth=t)
        t = np.array([[float(frdot(a2[i], mk[:, j])) for j in range(3)] for i in range(2)])
        try_emit("tensordot", {"axes": 1}, [C(a2), C(mk)],
                 lambda: np.tensordot(a2, mk, 1), "deep", dt, truth=t)

    # --- cross: the lone product-family gap. The cross product is multiply-subtract
    #     (a1*b2 - a2*b1, ...), NO long reduction, so it is bit-exact for float/complex/
    #     int64 at every value and every layout (it reads through strides). int32 and
    #     narrower widen to int64 in NumPy 2.x's cross — a dtype divergence deliberately
    #     left out; the product logic is dtype-agnostic and the four dtypes here cover it.
    CROSS_DTYPES = ["float64", "float32", "complex128", "int64"]
    for dt in CROSS_DTYPES:
        ndt = np.dtype(dt)

        def cv(vals, _ndt=ndt):
            a = np.array(vals, dtype=np.float64)
            if _ndt.kind == "c":
                a = a + 1j * a[::-1]
            return np.ascontiguousarray(a.astype(_ndt))

        a3, b3 = cv([1, 2, 3]), cv([4, 5, 6])
        emit("cross", {}, [C(a3), C(b3)], np.cross(a3, b3), "3v", dt)
        A23 = np.ascontiguousarray(np.stack([cv([1, 2, 3]), cv([-2, 0, 4])]))
        B23 = np.ascontiguousarray(np.stack([cv([4, 5, 6]), cv([1, -1, 2])]))
        emit("cross", {}, [C(A23), C(B23)], np.cross(A23, B23), "batch", dt)
    # axis params: vectors laid out along axis 0 (columns).
    Ac = np.ascontiguousarray(np.array([[1., 4], [2, 5], [3, 6]]))
    Bc = np.ascontiguousarray(np.array([[7., 1], [8, 0], [9, 2]]))
    emit("cross", {"axisa": 0, "axisb": 0, "axisc": 0}, [C(Ac), C(Bc)],
         np.cross(Ac, Bc, axisa=0, axisb=0, axisc=0), "axis0", "float64")
    # strided/reversed reads of the cross operand (F / negrow / negcol).
    A23f = np.ascontiguousarray(np.array([[1., 2, 3], [4, 5, 6]]))
    B23f = np.ascontiguousarray(np.array([[7., 8, 9], [1, 0, 2]]))
    for lay in ("F", "negrow", "negcol"):
        baseA, viewA = _mm_layout(A23f, lay)
        emit("cross", {}, [(baseA, viewA), (np.ascontiguousarray(B23f), B23f)],
             np.cross(viewA, B23f), f"lay_{lay}", "float64")

    # --- cov / corrcoef: covariance is a normalized dot product, so byte-exact for SMALL
    #     observation counts (the dot is an exact short float sum). The UNWEIGHTED param
    #     surface (rowvar/bias/ddof/y/complex/int-widen) is recorded here; WEIGHTED cov
    #     (fweights/aweights) rounds 1 ULP off in the `fact` normalization and is left to
    #     cov's tolerance battle-tests. A second operand IS the `y` variable (OpRegistry
    #     keys off the operand count).
    Mcov = np.array([[0., 2, 1, 4], [3, 1, 5, 2]])
    x1, y1 = np.array([1., 2, 3, 4]), np.array([2., 1, 4, 3])
    Mc = np.array([[1 + 1j, 2 - 1j, 3 + 0j], [0 + 2j, 1 + 0j, 2 - 2j]])
    emit("cov", {}, [C(Mcov)], np.cov(Mcov), "2x4", "float64")
    emit("cov", {"bias": True}, [C(Mcov)], np.cov(Mcov, bias=True), "bias", "float64")
    emit("cov", {"ddof": 0}, [C(Mcov)], np.cov(Mcov, ddof=0), "ddof0", "float64")
    emit("cov", {"ddof": 2}, [C(Mcov)], np.cov(Mcov, ddof=2), "ddof2", "float64")
    emit("cov", {"rowvar": False}, [C(Mcov)], np.cov(Mcov, rowvar=False), "colvar", "float64")
    emit("cov", {}, [C(x1)], np.cov(x1), "1d", "float64")
    emit("cov", {}, [C(x1), C(y1)], np.cov(x1, y1), "xy", "float64")
    emit("cov", {}, [C(Mcov.astype(np.int64))], np.cov(Mcov.astype(np.int64)), "int", "int64")
    emit("cov", {}, [C(Mc)], np.cov(Mc), "complex", "complex128")
    emit("corrcoef", {}, [C(Mcov)], np.corrcoef(Mcov), "2x4", "float64")
    emit("corrcoef", {"rowvar": False}, [C(Mcov)], np.corrcoef(Mcov, rowvar=False), "colvar", "float64")
    emit("corrcoef", {}, [C(x1), C(y1)], np.corrcoef(x1, y1), "xy", "float64")
    emit("corrcoef", {}, [C(Mc)], np.corrcoef(Mc), "complex", "complex128")

    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


# ---------------------------------------------------------------------------
# T-poly — the PORTABLE polynomial family (poly.jsonl). These are pure array
# arithmetic / convolution / Horner with NO backend and NO long reduction, so
# they are bit-exact everywhere (probed: Horner order, leading-zero normalisation
# and polynomial division all match NumPy byte-for-byte). The three BACKEND
# polynomial ops — roots (eigvals of the companion matrix), polyfit (lstsq) and
# poly of a 2-D matrix (eigvals) — ride the host-pinned linalg_parity tier instead.
def gen_poly():
    cases = []
    n = 0

    def add(op, params, operands, r, tag, dt):
        nonlocal n
        exp = _tuple_expected([np.asarray(x) for x in r]) if isinstance(r, tuple) \
            else _arr_expected(np.asarray(r))
        cases.append({"id": f"{op}/{tag}/{dt}/{n}", "op": op, "params": params,
                      "operands": operands, "expected": exp, "layout": tag, "valueclass": "poly"})
        n += 1

    def C(a):
        a = np.ascontiguousarray(a)
        return describe(a, a)

    for dt in ["float64", "float32", "complex128", "int64"]:
        ndt = np.dtype(dt)

        def cast(vals, _ndt=ndt):
            a = np.array(vals, dtype=np.float64)
            if _ndt.kind == "c":
                a = a + 1j * (0.5 * a[::-1])
            return np.ascontiguousarray(a.astype(_ndt))

        # poly (1-D roots -> coefficients, the convolution branch)
        add("poly", {}, [C(cast([1, 2, 3]))], np.poly(cast([1, 2, 3])), "roots3", dt)
        add("poly", {}, [C(cast([1, -1, 2, -2]))], np.poly(cast([1, -1, 2, -2])), "roots4", dt)
        # polyval (Horner)
        p, x = cast([1, -2, 3]), cast([0, 1, 2, 3])
        add("polyval", {}, [C(p), C(x)], np.polyval(p, x), "vec", dt)
        add("polyval", {}, [C(cast([2, 0, -1, 5])), C(cast([-1, 0.5, 2]))],
            np.polyval(cast([2, 0, -1, 5]), cast([-1, 0.5, 2])), "vec2", dt)
        # vander
        xv = cast([1, 2, 3, 4])
        add("vander", {"N": 3}, [C(xv)], np.vander(xv, 3), "N3", dt)
        add("vander", {}, [C(xv)], np.vander(xv), "default", dt)
        add("vander", {"increasing": True}, [C(xv)], np.vander(xv, increasing=True), "inc", dt)
        # polyder
        pd = cast([1, 2, 3, 4, 5])
        add("polyder", {}, [C(pd)], np.polyder(pd), "m1", dt)
        add("polyder", {"m": 2}, [C(pd)], np.polyder(pd, 2), "m2", dt)
        # polyint (always float-returning; k = integration constant)
        add("polyint", {}, [C(pd)], np.polyint(pd), "m1", dt)
        add("polyint", {"m": 2}, [C(pd)], np.polyint(pd, 2), "m2", dt)
        add("polyint", {"k": 3.0}, [C(cast([1, 2, 3]))], np.polyint(cast([1, 2, 3]), k=3.0), "k", dt)
        # polyadd / polysub / polymul (different-length operands)
        a1, a2 = cast([1, 2, 3]), cast([4, 5])
        add("polyadd", {}, [C(a1), C(a2)], np.polyadd(a1, a2), "difflen", dt)
        add("polysub", {}, [C(a1), C(a2)], np.polysub(a1, a2), "difflen", dt)
        add("polymul", {}, [C(a1), C(a2)], np.polymul(a1, a2), "conv", dt)
        # polydiv -> (quotient, remainder) tuple
        add("polydiv", {}, [C(cast([1, 2, 3, 4])), C(cast([1, 1]))],
            np.polydiv(cast([1, 2, 3, 4]), cast([1, 1])), "div", dt)
        # poly1d: leading-zero normalisation, and construction from roots
        add("poly1d_coeffs", {}, [C(cast([0, 0, 1, 2, 3]))],
            np.poly1d(cast([0, 0, 1, 2, 3])).coeffs, "norm", dt)
        add("poly1d_fromroots", {}, [C(cast([1, 2, 3]))],
            np.poly1d(cast([1, 2, 3]), r=True).coeffs, "fromroots", dt)

    # layouts (float64): polyval and vander read their operands through strides.
    xL = np.array([1., 2, 3, 4])
    pL = np.array([1., -2, 3])
    for lay in ("F", "negrow"):
        # a 1-D operand: reverse it (F == C for 1-D, so use a reversed view for a real stride).
        base = np.ascontiguousarray(xL[::-1]); view = base[::-1]
        add("vander", {"N": 3}, [describe(base, view)], np.vander(view, 3), f"vander_{lay}", "float64")
        base2 = np.ascontiguousarray(pL[::-1]); view2 = base2[::-1]
        add("polyval", {}, [describe(base2, view2), C(xL)], np.polyval(view2, xL), f"polyval_{lay}", "float64")

    return cases


# ---------------------------------------------------------------------------
# T-einsum — np.einsum + np.einsum_path (einsum.jsonl). einsum is byte-exact vs
# NumPy for INTEGER/complex-integer contractions (order-independent) and for
# SMALL-EXACT float contractions (short exact sums), plus the whole VIEW path
# (transpose/diagonal/no-sum) — probed. Larger float contractions differ (NumSharp
# routes them through matmul; NumPy's default einsum uses its own C iterator), so
# operands are kept small-exact. einsum_path returns the info STRING (text kind);
# it is shape-derived (no values) and byte-identical to NumPy for non-ellipsis
# subscripts (the ellipsis placeholder letters are NumPy's one hash-randomised
# divergence and are avoided here).
def gen_einsum():
    cases = []
    n = 0

    def add(op, params, operands, expected, tag, dt):
        nonlocal n
        cases.append({"id": f"{op}/{tag}/{dt}/{n}", "op": op, "params": params,
                      "operands": operands, "expected": expected, "layout": tag, "valueclass": "einsum"})
        n += 1

    def C(a):
        a = np.ascontiguousarray(a)
        return describe(a, a)

    for dt in ["float64", "complex128", "int64"]:
        ndt = np.dtype(dt)

        # NONZERO value pool: a signed zero would diverge in the outer/hadamard einsums, where
        # NumPy's sop accumulator (seeded +0.0) absorbs the sign of a `-x * 0 = -0.0` term into
        # +0.0 while NumSharp's element-wise multiply keeps the raw -0.0. Avoiding zero operands
        # keeps the contraction gate strictly byte-exact; zero/negative accumulation is already
        # gated by the products tier.
        _re = np.array([1, 2, 3, -1, -2, 4, 5], dtype=np.float64)
        _im = np.array([1, -1, 2, -2, 3], dtype=np.float64)

        def sm(shape, off=0, _ndt=ndt):
            k = int(np.prod(shape))
            a = _re[(np.arange(k) + off) % len(_re)]
            if _ndt.kind == "c":
                a = a + 1j * _im[(np.arange(k) + off) % len(_im)]
            return np.ascontiguousarray(a.astype(_ndt).reshape(shape))

        E = lambda s, *ops: _arr_expected(np.asarray(np.einsum(s, *ops)))
        M23, M34, M33 = sm((2, 3)), sm((3, 4), 2), sm((3, 3))
        v3, w3 = sm((3,)), sm((3,), 4)
        B1, B2 = sm((2, 2, 3)), sm((2, 3, 2), 1)
        cases_specs = [
            ("ij,jk->ik", (M23, M34), "matmul"),
            ("ij->ji", (M23,), "transpose"),
            ("ii->i", (M33,), "diag"),
            ("ii->", (M33,), "trace"),
            ("ij->i", (M23,), "rowsum"),
            ("ij->j", (M23,), "colsum"),
            ("ij->", (M23,), "fullsum"),
            ("i,i->", (v3, w3), "dot"),
            ("i,j->ij", (v3, sm((2,), 3)), "outer"),
            ("ij,ij->ij", (M23, sm((2, 3), 5)), "hadamard"),
            ("ij,ij->", (M23, sm((2, 3), 5)), "frobenius"),
            ("bij,bjk->bik", (B1, B2), "batched"),
            ("ij->ij", (M23,), "copy"),
        ]
        for subs, ops, tag in cases_specs:
            add("einsum", {"subscripts": subs}, [C(o) for o in ops], E(subs, *ops), tag, dt)

    # einsum_path — the contraction planner's info string (text kind). Shape-only.
    def path(subs, shapes, optimize, tag):
        nonlocal n
        ops = [np.zeros(s) for s in shapes]
        _, rep = np.einsum_path(subs, *ops, optimize=optimize)
        cases.append({"id": f"einsum_path/{tag}/{n}", "op": "einsum_path",
                      "params": {"subscripts": subs, "optimize": optimize},
                      "operands": [C(o) for o in ops],
                      "expected": {"kind": "text", "value": rep}, "layout": tag, "valueclass": "einsum"})
        n += 1

    path("ij,jk,kl->il", [(4, 5), (5, 6), (6, 7)], "greedy", "chain3_greedy")
    path("ij,jk,kl->il", [(4, 5), (5, 6), (6, 7)], "optimal", "chain3_optimal")
    path("ij,jk->ik", [(3, 4), (4, 5)], "greedy", "matmul")
    path("ea,fb,abcd,gc,hd->efgh", [(5, 4), (5, 4), (4, 4, 4, 4), (5, 4), (5, 4)], "greedy", "tensor5_greedy")
    path("ea,fb,abcd,gc,hd->efgh", [(5, 4), (5, 4), (4, 4, 4, 4), (5, 4), (5, 4)], "optimal", "tensor5_optimal")

    return cases


# ============================================================================
# FFT tier (np.fft.*) — the differential gate for the managed pocketfft engine
# (src/NumSharp.Core/Fourier/). NumPy 2.4.2 is the oracle.
#
# DTYPE POLICY (probed 2.4.2): the forward/complex transforms return complex128
# for float64/complex128/int/bool input and complex64 for float32/float16;
# irfft/hfft return float64 (float32/float16 -> float32/float16). NumSharp has
# ONE complex type (complex128) and no complex64, so it promotes float32/float16
# to double and returns complex128 / float64 — a bit-verified equality with
# NumPy's OWN double computation (np.fft.fft(x32) == np.fft.fft(x32.astype(f8))
# was confirmed byte-identical to NumSharp). That makes float64/complex128/int/
# bool the CONTRACTUAL (bit-exact) cells, and float32/float16 the ONE documented
# divergence — recorded here as-produced (complex64/float32/float16) and excused
# in MisalignedRegistry as a dtype-ONLY difference (values = the correctly-
# rounded double result). complex64 has no NumSharp NPTypeCode, so its cases
# reach the harness through CompareArray's unmappable-dtype -> Dtype route.
# ============================================================================

# Clean, discriminating FFT values — NO NaN/inf. Every output bin sums the WHOLE
# signal, so a single NaN blanks the entire spectrum and stops discriminating the
# butterflies / twiddles / Bluestein chirp. Deterministic; small magnitudes keep
# float16 (11-bit mantissa) representable so its cells are honest.
_FFT_POOL = [1.0, -2.0, 3.5, 0.0, -0.5, 2.25, -1.75, 4.0, 6.5, -3.0, 0.25, 5.0,
             -4.5, 7.0, -1.0, 2.0, 0.75, -6.0, 3.0, -0.25, 8.0, -2.5, 1.25, -3.75]

FFT_REAL_DTYPES = ["float64", "float32", "float16", "int32", "bool"]      # rfft/ihfft (real input)
FFT_ANY_DTYPES = FFT_REAL_DTYPES + ["complex128"]                          # fft/ifft/irfft/hfft


def _fft_fill(n, dt):
    dt = np.dtype(dt)
    if dt.kind == "c":
        r = np.array(_FFT_POOL, dtype=np.float64)
        vals = (r + 1j * np.roll(r, 3)).astype(dt)
    elif dt.kind == "b":
        vals = (np.arange(len(_FFT_POOL)) % 2 == 0)
    elif dt.kind in "iu":
        vals = np.array([int(round(v)) for v in _FFT_POOL], dtype=dt)
    else:  # float16 / float32 / float64
        vals = np.array(_FFT_POOL, dtype=np.float64).astype(dt)
    if len(vals) < n:
        vals = np.tile(vals, (n + len(vals) - 1) // len(vals))
    return np.ascontiguousarray(vals[:n].copy())


def _fft_base(shape, dt):
    n = int(np.prod(shape)) if len(shape) else 1
    return np.ascontiguousarray(_fft_fill(n, dt).reshape(shape))


def _fft_layouts_1d(dt):
    """(name, base, view) for 1-D transforms — copy_input over C / strided / reversed / offset."""
    L = []
    b = _fft_base((8,), dt);   L.append(("c1d", b, b))
    b = _fft_base((16,), dt);  L.append(("strided2_1d", b, b[::2]))
    b = _fft_base((8,), dt);   L.append(("neg_1d", b, b[::-1]))
    b = _fft_base((10,), dt);  L.append(("offset_1d", b, b[2:9]))          # len 7 (radix-7, offset)
    return L


def _fft_layouts_nd(dt):
    """(name, base, view) for multi-axis transforms — C / F / col-strided / reversed / bcast / 3-D."""
    L = []
    b = _fft_base((4, 5), dt);   L.append(("c2d", b, b))
    b = _fft_base((5, 4), dt);   L.append(("f2d", b, b.T))                 # (4,5) F-contiguous
    b = _fft_base((4, 10), dt);  L.append(("strided_cols", b, b[:, ::2]))  # (4,5) column-strided
    b = _fft_base((4, 5), dt);   L.append(("neg2d", b, b[::-1, ::-1]))
    b = _fft_base((5,), dt);     L.append(("bcast_row", b, np.broadcast_to(b, (4, 5))))
    b = _fft_base((2, 3, 4), dt); L.append(("c3d", b, b))
    b = _fft_base((2, 3, 4), dt); L.append(("transp3d", b, b.transpose(2, 0, 1)))  # (4,2,3)
    return L


def _fft_expected(r):
    r = np.asarray(r)
    shape = [int(d) for d in r.shape]                                       # BEFORE ascontiguousarray
    return {"dtype": r.dtype.name, "shape": shape,
            "buffer": np.ascontiguousarray(r).tobytes().hex()}


# 1-D transforms: (valid input dtypes, callable(view, n, axis, norm)).
_FFT_1D_OPS = {
    "fft":   (FFT_ANY_DTYPES,  lambda v, n, ax, nm: np.fft.fft(v, n, ax, nm)),
    "ifft":  (FFT_ANY_DTYPES,  lambda v, n, ax, nm: np.fft.ifft(v, n, ax, nm)),
    "rfft":  (FFT_REAL_DTYPES, lambda v, n, ax, nm: np.fft.rfft(v, n, ax, nm)),
    "irfft": (FFT_ANY_DTYPES,  lambda v, n, ax, nm: np.fft.irfft(v, n, ax, nm)),
    "hfft":  (FFT_ANY_DTYPES,  lambda v, n, ax, nm: np.fft.hfft(v, n, ax, nm)),
    "ihfft": (FFT_REAL_DTYPES, lambda v, n, ax, nm: np.fft.ihfft(v, n, ax, nm)),
}

# N-D transforms: op -> valid input dtypes. The call goes through _fft_nd_call, which OMITS a
# None argument rather than passing it — because NumPy's 2-D forms DEFAULT axes to (-2,-1) but
# treat an EXPLICIT axes=None as "all axes" (fftn behaviour). NumSharp coalesces a null axes to
# (-2,-1) (its default), so the generator must exercise NumPy's DEFAULT (omit axes), not the
# axes=None path NumSharp cannot express. Same reasoning for s / norm.
_FFT_ND_OPS = {
    "fft2":   FFT_ANY_DTYPES,  "ifft2":  FFT_ANY_DTYPES,
    "fftn":   FFT_ANY_DTYPES,  "ifftn":  FFT_ANY_DTYPES,
    "rfft2":  FFT_REAL_DTYPES, "irfft2": FFT_ANY_DTYPES,
    "rfftn":  FFT_REAL_DTYPES, "irfftn": FFT_ANY_DTYPES,
}


def _fft_nd_call(op, view, s, axes, norm):
    kw = {}
    if s is not None:
        kw["s"] = list(s)
    if axes is not None:
        kw["axes"] = list(axes)
    if norm is not None:
        kw["norm"] = norm
    return getattr(np.fft, op)(view, **kw)


def gen_fft():
    cases = []
    n = [0]

    def emit(op, params, operands, r, layout, dt):
        cases.append({
            "id": f"{op}/{layout}/{dt}/{n[0]}",
            "op": op, "params": params, "operands": operands,
            "expected": _fft_expected(r), "layout": layout, "valueclass": "fft",
        })
        n[0] += 1

    def try_emit(op, params, operands, call, layout, dt):
        try:
            r = call()
        except Exception:
            return False                       # NumPy raised (e.g. rfft(complex)) — error parity is separate
        emit(op, params, operands, r, layout, dt)
        return True

    # ---- 1-D core -----------------------------------------------------------
    for op, (dtypes, f) in _FFT_1D_OPS.items():
        for dt in dtypes:
            # (a) default n/axis/norm across every layout — copy_input over each memory descriptor.
            for lname, base, view in (_fft_layouts_1d(dt) + _fft_layouts_nd(dt)):
                try_emit(op, {"n": None, "axis": -1, "norm": None}, [describe(base, view)],
                         lambda f=f, view=view: f(view, None, -1, None), lname, dt)
            # (b) n sweep on a contiguous 1-D signal. Beyond {4 truncate, 12 zero-pad, 13 prime
            # (Bluestein)} the perfect-square sizes force EACH mixed-radix codelet's distinct ido>1
            # branch — 9=3²(pass3), 25=5²(pass5), 49=7²(pass7), 64=8²(pass8), 121=11²(pass11),
            # 169=13²(passg ido>1); pass4 ido>1 is already hit by 12=4·3. Load-bearing: pass7/pass11's
            # ido>1 branch had a transcription bug (special_mul on ca/cb instead of ca±cb, the PM) that
            # {4,12,13} never reached — undetected until the float32 port exercised n=98/259
            # (FFT_PARITY.md §7). The squares gate the whole codelet family against that bug class.
            b = _fft_base((8,), dt)
            for nn in (4, 9, 12, 13, 25, 49, 64, 121, 169):
                try_emit(op, {"n": nn, "axis": -1, "norm": None}, [describe(b, b)],
                         lambda f=f, b=b, nn=nn: f(b, nn, -1, None), "c1d_n", dt)
            # (c) norm sweep (ortho, forward, explicit backward) on the same signal.
            for nm in ("ortho", "forward", "backward"):
                try_emit(op, {"n": None, "axis": -1, "norm": nm}, [describe(b, b)],
                         lambda f=f, b=b, nm=nm: f(b, None, -1, nm), "c1d_norm", dt)
            # (d) axis sweep on 2-D and 3-D contiguous inputs (middle/negative axes).
            b2 = _fft_base((4, 5), dt)
            for ax in (0, 1, -2):
                try_emit(op, {"n": None, "axis": ax, "norm": None}, [describe(b2, b2)],
                         lambda f=f, b2=b2, ax=ax: f(b2, None, ax, None), f"c2d_ax{ax}", dt)
            b3 = _fft_base((2, 3, 4), dt)
            for ax in (0, 1, -1):
                try_emit(op, {"n": None, "axis": ax, "norm": None}, [describe(b3, b3)],
                         lambda f=f, b3=b3, ax=ax: f(b3, None, ax, None), f"c3d_ax{ax}", dt)

    # ---- N-D ----------------------------------------------------------------
    for op, dtypes in _FFT_ND_OPS.items():
        for dt in dtypes:
            # (a) default s/axes over the multi-axis layouts (axes OMITTED -> the op's real default).
            for lname, base, view in _fft_layouts_nd(dt):
                try_emit(op, {"s": None, "axes": None, "norm": None}, [describe(base, view)],
                         lambda op=op, view=view: _fft_nd_call(op, view, None, None, None), lname, dt)
            b2 = _fft_base((4, 5), dt)
            # (b) s sweep: [2,3] truncate, [6,6] pad, [-1,3] the -1 "full length" sentinel.
            for s in ([2, 3], [6, 6], [-1, 3]):
                try_emit(op, {"s": s, "axes": None, "norm": None}, [describe(b2, b2)],
                         lambda op=op, b2=b2, s=s: _fft_nd_call(op, b2, s, None, None), "c2d_s", dt)
            # (c) norm sweep.
            for nm in ("ortho", "forward"):
                try_emit(op, {"s": None, "axes": None, "norm": nm}, [describe(b2, b2)],
                         lambda op=op, b2=b2, nm=nm: _fft_nd_call(op, b2, None, None, nm), "c2d_norm", dt)
            # (d) explicit axes (order + negative) on a 3-D input.
            b3 = _fft_base((2, 3, 4), dt)
            for ax in ([0, 1], [2, 0], [-1, -2]):
                try_emit(op, {"s": None, "axes": ax, "norm": None}, [describe(b3, b3)],
                         lambda op=op, b3=b3, ax=ax: _fft_nd_call(op, b3, None, ax, None), "c3d_axes", dt)

    # ---- helpers ------------------------------------------------------------
    # fftfreq / rfftfreq: pure generators (float64), no operand. n even/odd/prime/tiny, d varied.
    for nn in (8, 7, 13, 1, 2, 16):
        for d in (1.0, 0.5, 2.0):
            emit("fftfreq", {"n": nn, "d": d}, [], np.fft.fftfreq(nn, d), "freq", "float64")
            emit("rfftfreq", {"n": nn, "d": d}, [], np.fft.rfftfreq(nn, d), "freq", "float64")

    # fftshift / ifftshift: dtype-preserving cyclic roll. layouts x dtypes x axes {None,int,tuple}.
    for dt in ("int32", "float64", "complex128", "float32"):
        for lname, base, view in (_fft_layouts_1d(dt) + _fft_layouts_nd(dt)):
            nd = view.ndim
            specs = [("axes", None)]                      # all axes
            if nd >= 1:
                specs.append(("axis", 0))                 # single-int overload
            if nd >= 2:
                specs.append(("axis", -1))
                specs.append(("axes", [0, nd - 1]))       # tuple
            for kind, ax in specs:
                params = {"axis": ax} if kind == "axis" else {"axes": ax}
                np_ax = tuple(ax) if isinstance(ax, list) else ax
                try_emit("fftshift", params, [describe(base, view)],
                         lambda view=view, np_ax=np_ax: np.fft.fftshift(view, axes=np_ax), lname, dt)
                try_emit("ifftshift", params, [describe(base, view)],
                         lambda view=view, np_ax=np_ax: np.fft.ifftshift(view, axes=np_ax), lname, dt)

    return cases


# =====================================================================================
# np.random PCG64 Generator byte-parity ("generator_parity" tiers). The legacy
# random_parity tiers above pin the MT19937 RandomState streams; these pin the MODERN
# default_rng(seed) -> Generator(PCG64) streams (a DIFFERENT bit generator + different
# algorithms: Lemire bounded integers, ziggurat normal/exponential), plus the two new
# RandomState helpers np.random.random_integers and np.random.bytes.
#
# TWO FILES, same split as random_parity:
#   * generator_parity.jsonl      — PORTABLE: pure PCG64 bits + exactly-rounded IEEE
#     (random, integers, uniform, permutation, shuffle, choice, bytes; triangular, the urn-walk
#     hypergeometric, the geometric search, multivariate_hypergeometric count) + the RandomState
#     helpers random_integers/bytes (pure MT19937 bits). Hard-gated on every host.
#   * generator_parity_host.jsonl — HOST-libm: the ziggurat / rejection samplers whose
#     transform consumes log1p/exp/pow (standard_normal, standard_exponential, normal,
#     exponential, standard_gamma, gamma) and every distribution built on them (one parameter
#     set per internal branch). Byte-exact on win-amd64 (Kahan log1p + Math.* == ucrtbase)
#     except pareto/power, which call the CRT's closed in-band expm1 (MisalignedRegistry
#     bounds them per element); reported Inconclusive off-Windows (the random_parity_host
#     pattern). multivariate_normal is excluded: it is byte-exact only with a LAPACK backend.
#
# Every case seeds a FRESH default_rng / RandomState, so replaying never mutates global
# np.random state (matches the fresh-instance isolation the rnd tier uses). Op key "grnd";
# pairs 1:1 with OpRegistry.GeneratorDraw. int-output methods are int64 on BOTH sides
# (Generator.integers defaults int64; NumSharp matches). random_integers is the legacy C long
# — int32 on this win-amd64 authoring host — and is recorded widened to int64, NumSharp's
# LP64 model (the rnd tier's _RND_INT64_CAST policy).
_GEN_DTYPE = {
    "float64": np.float64, "float32": np.float32,
    "int8": np.int8, "int16": np.int16, "int32": np.int32, "int64": np.int64,
    "uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32, "uint64": np.uint64,
    "bool": np.bool_,
}
_GRND_SEEDS = [42, 987654321]
_GRND_SIZES = [[7], [2, 3]]
# The Generator distributions whose parameters are all scalars: `rng.<method>(*args, size)` on both sides (ints stay
# JSON ints, so binomial/hypergeometric receive Python ints exactly as a user call would).
_GRND_POSITIONAL = {
    "beta", "chisquare", "f", "noncentral_chisquare", "noncentral_f", "standard_cauchy", "standard_t", "vonmises",
    "pareto", "weibull", "power", "laplace", "gumbel", "logistic", "lognormal", "rayleigh", "wald", "triangular",
    "binomial", "negative_binomial", "poisson", "zipf", "geometric", "hypergeometric", "logseries",
}


def gen_generator_parity():
    portable = []
    host = []
    n = 0

    def emit(into, method, params, r):
        nonlocal n
        r = np.asarray(r)
        into.append({
            "id": f"grnd/{method}/{params.get('dtype', '-')}/seed{params['seed']}/{n}",
            "op": "grnd",
            "params": params,
            "operands": [],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": "grnd",
            "valueclass": "stream",
        })
        n += 1

    def run(method, params):
        seed = params["seed"]
        args = params.get("args", [])
        size = tuple(params["size"]) if "size" in params else ()
        dtype = params.get("dtype")
        npdt = _GEN_DTYPE[dtype] if dtype else None
        draws = params.get("draws", 1)

        # ---- RandomState helpers (fresh instance, MT19937) ----
        if method == "random_integers":
            rs = np.random.RandomState(seed)
            hi = None if len(args) < 2 else int(args[1])
            r = None
            for _ in range(draws):
                r = rs.random_integers(int(args[0]), hi, size if size else None)
            return np.asarray(r).astype(np.int64)   # C long (int32 here) -> NumSharp's LP64 int64; the values fit
        if method == "rs_bytes":
            rs = np.random.RandomState(seed)
            r = None
            for _ in range(draws):
                r = np.frombuffer(rs.bytes(int(args[0])), dtype=np.uint8)
            return r

        # ---- Generator (PCG64) ----
        rng = np.random.default_rng(seed)
        r = None
        for _ in range(draws):
            if method == "random":
                r = rng.random(size, dtype=npdt or np.float64)
            elif method == "integers":
                r = rng.integers(int(args[0]), int(args[1]), size,
                                 dtype=npdt or np.int64, endpoint=params.get("endpoint", False))
            elif method == "uniform":
                r = rng.uniform(args[0], args[1], size)
            elif method == "permutation":
                r = rng.permutation(int(args[0]))
            elif method == "shuffle":
                r = np.arange(int(args[0]))
                rng.shuffle(r)
            elif method == "choice":
                r = rng.choice(int(args[0]), size, replace=params.get("replace", True),
                               p=params.get("p"), shuffle=params.get("cshuffle", True))
            elif method == "bytes":
                r = np.frombuffer(rng.bytes(int(args[0])), dtype=np.uint8)
            elif method == "standard_normal":
                r = rng.standard_normal(size, dtype=npdt or np.float64)
            elif method == "standard_exponential":
                r = rng.standard_exponential(size, dtype=npdt or np.float64,
                                             method=params.get("emethod", "zig"))
            elif method == "normal":
                r = rng.normal(args[0], args[1], size)
            elif method == "exponential":
                r = rng.exponential(args[0], size)
            elif method == "standard_gamma":
                r = rng.standard_gamma(args[0], size, dtype=npdt or np.float64)
            elif method == "gamma":
                r = rng.gamma(args[0], args[1], size)
            elif method in _GRND_POSITIONAL:
                # The scalar-parameter distributions: NumPy's positional parameters, then size.
                r = getattr(rng, method)(*args, size)
            elif method == "multinomial":
                n_arg = np.array(params["narr"]) if "narr" in params else int(args[0])
                r = rng.multinomial(n_arg, params["pvals"], size if "size" in params else None)
            elif method == "dirichlet":
                r = rng.dirichlet(params["alpha"], size if "size" in params else None)
            elif method == "multivariate_hypergeometric":
                r = rng.multivariate_hypergeometric(params["colors"], int(args[0]),
                                                    size if "size" in params else None,
                                                    method=params.get("mvmethod", "marginals"))
            else:
                raise ValueError(f"unknown generator method '{method}'")
        return np.asarray(r)

    def cases(into, method, base_params, sized=True, draws2=False):
        specs = ([(_GRND_SEEDS[0], _GRND_SIZES[0]), (_GRND_SEEDS[0], _GRND_SIZES[1]),
                  (_GRND_SEEDS[1], _GRND_SIZES[0])]
                 if sized else [(_GRND_SEEDS[0], None), (_GRND_SEEDS[1], None)])
        for seed, size in specs:
            p = dict(base_params, method=method, seed=seed)
            if size is not None:
                p["size"] = size
            emit(into, method, p, run(method, p))
        if draws2:
            p = dict(base_params, method=method, seed=_GRND_SEEDS[0],
                     size=_GRND_SIZES[0], draws=2)
            emit(into, method, p, run(method, p))

    # ---- PORTABLE: pure PCG64 bits + exactly-rounded IEEE ----
    cases(portable, "random", {"args": [], "dtype": "float64"}, draws2=True)
    cases(portable, "random", {"args": [], "dtype": "float32"})
    cases(portable, "uniform", {"args": [-3.0, 7.0]})
    for dt in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
        cases(portable, "integers", {"args": [0, 100], "dtype": dt})
    cases(portable, "integers", {"args": [-50, 50], "dtype": "int32"}, draws2=True)
    cases(portable, "integers", {"args": [0, 100], "dtype": "int32", "endpoint": True})
    cases(portable, "integers", {"args": [0, 2], "dtype": "bool"})
    cases(portable, "permutation", {"args": [10]}, sized=False)
    cases(portable, "shuffle", {"args": [12]}, sized=False)
    cases(portable, "choice", {"args": [10]})
    cases(portable, "choice", {"args": [10], "replace": False})
    cases(portable, "choice", {"args": [4], "p": [0.1, 0.2, 0.3, 0.4]})
    for length in (1, 8, 13):
        cases(portable, "bytes", {"args": [length]}, sized=False)
    # RandomState helpers (portable MT19937 bits)
    cases(portable, "random_integers", {"args": [1, 6]})
    cases(portable, "random_integers", {"args": [5]}, sized=False)  # high=None -> [1, low]
    for length in (1, 10):
        cases(portable, "rs_bytes", {"args": [length]}, sized=False)

    # ---- HOST-libm: ziggurat / rejection samplers ----
    cases(host, "standard_normal", {"args": [], "dtype": "float64"}, draws2=True)
    cases(host, "standard_normal", {"args": [], "dtype": "float32"})
    cases(host, "standard_exponential", {"args": [], "dtype": "float64", "emethod": "zig"}, draws2=True)
    cases(host, "standard_exponential", {"args": [], "dtype": "float64", "emethod": "inv"})
    cases(host, "standard_exponential", {"args": [], "dtype": "float32", "emethod": "zig"})
    cases(host, "normal", {"args": [5.0, 2.5]})
    cases(host, "exponential", {"args": [2.5]})
    cases(host, "standard_gamma", {"args": [2.0], "dtype": "float64"})
    cases(host, "standard_gamma", {"args": [0.5], "dtype": "float64"})  # shape<1 branch
    cases(host, "standard_gamma", {"args": [2.0], "dtype": "float32"})
    cases(host, "gamma", {"args": [2.0, 3.0]})

    # ---- the distribution surface (Phase 6), one parameter set per internal branch ----
    # PORTABLE: pure PCG64 bits + exactly-rounded IEEE (next_double, sqrt, +-*/, random_interval).
    for args in ([0.0, 3.0, 10.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]):
        cases(portable, "triangular", {"args": args})
    for args in ([10, 7, 8], [15, 15, 20]):                       # the urn walk (sample < 10 or > total - 10)
        cases(portable, "hypergeometric", {"args": args})
    cases(portable, "geometric", {"args": [0.35]})                # the search (p >= 1/3): sums of products
    cases(portable, "multivariate_hypergeometric", {"args": [40], "colors": [30, 20, 50], "mvmethod": "count"})
    # HOST-libm: every sampler whose transform or setup consumes log/exp/pow/log1p/expm1.
    for args in ([2.0, 3.0], [0.5, 0.5], [1e-3, 1e-3], [1e-105, 1e-105], [0.5, 2.0]):
        cases(host, "beta", {"args": args})
    for args in ([3.0], [0.5]):
        cases(host, "chisquare", {"args": args})
    for args in ([5.0, 7.0], [0.5, 0.5]):
        cases(host, "f", {"args": args})
    for args in ([3.0, 1.5], [0.5, 1.5], [3.0, 0.0], [0.5, 30.0]):
        cases(host, "noncentral_chisquare", {"args": args})
    for args in ([5.0, 7.0, 1.5], [0.5, 7.0, 1.5]):
        cases(host, "noncentral_f", {"args": args})
    cases(host, "standard_cauchy", {"args": []})
    for args in ([3.5], [0.5]):
        cases(host, "standard_t", {"args": args})
    for args in ([0.5, 2.0], [0.0, 1e-9], [1.0, 1e-6], [-2.0, 1e7], [3.0, 50.0]):
        cases(host, "vonmises", {"args": args})
    # pareto/power: the CRT's in-band expm1 is closed; MisalignedRegistry bounds the difference per element.
    for args in ([3.0], [0.5]):
        cases(host, "pareto", {"args": args})
    for args in ([2.5], [0.3]):
        cases(host, "power", {"args": args})
    for args in ([1.79], [0.3], [0.0]):
        cases(host, "weibull", {"args": args})
    cases(host, "laplace", {"args": [0.0, 1.0]})
    cases(host, "gumbel", {"args": [0.5, 2.0]})
    cases(host, "logistic", {"args": [0.0, 1.0]})
    cases(host, "lognormal", {"args": [1.0, 0.5]})
    cases(host, "rayleigh", {"args": [1.5]})
    for args in ([3.0, 2.0], [0.5, 10.0]):
        cases(host, "wald", {"args": args})
    for args in ([10, 0.35], [100, 0.4], [1000, 0.7], [20, 0.9]):
        cases(host, "binomial", {"args": args})
    for args in ([5.0, 0.4], [0.5, 0.5]):
        cases(host, "negative_binomial", {"args": args})
    for args in ([3.5], [100.0], [1e6]):
        cases(host, "poisson", {"args": args})
    for args in ([3.0], [1.5], [1.05]):
        cases(host, "zipf", {"args": args})
    for args in ([0.1], [1e-5]):                                  # the inversion (p < 1/3): ceil(-E / log1p(-p))
        cases(host, "geometric", {"args": args})
    for args in ([100, 200, 50], [300, 100, 250], [1000, 1000, 1995]):   # HRUA
        cases(host, "hypergeometric", {"args": args})
    for args in ([0.6], [0.99], [0.1]):
        cases(host, "logseries", {"args": args})
    cases(host, "multinomial", {"args": [20], "pvals": [0.2, 0.3, 0.5]})
    cases(host, "multinomial", {"args": [], "narr": [[5], [10]], "pvals": [[0.5, 0.5], [0.1, 0.9]]}, sized=False)
    cases(host, "dirichlet", {"args": [], "alpha": [2.0, 3.0, 5.0]})
    cases(host, "dirichlet", {"args": [], "alpha": [0.05, 0.05, 0.05]})   # stick-breaking over random_beta
    cases(host, "multivariate_hypergeometric", {"args": [40], "colors": [30, 20, 50]})
    # Array-valued (broadcast) parameters of every Generator sampler — see _gen_random_broadcast.
    bp, bh, n = _gen_random_broadcast("gen", n)
    portable += bp
    host += bh

    return portable, host


# =====================================================================================
# Array-valued (broadcast) distribution parameters — BOTH random APIs. Appended to the
# random_parity tiers (legacy RandomState, op "rnd") and the generator_parity tiers (PCG64
# Generator, op "grnd"): every sampler whose parameters NumPy accepts as arrays
# (_common.pyx's cont / disc broadcast paths, mtrand.pyx / _generator.pyx), gated on the
# full observable contract —
#   * values: one draw per output position in C order with that position's parameters;
#   * the output shape: `size`, else the parameters' broadcast shape, then NumPy's
#     validate_output_shape ("Output size ... is not compatible with broadcast dimensions
#     of inputs ...") and MultiIterNew's "shape mismatch" text (arg 0 = the output);
#   * the conversion: PyArray_FROM_OTF's 'safe' gate (complex -> TypeError; ints/bool/
#     float32/float16 accepted; floats refused for integer parameters), 0-d arrays taking
#     the scalar path (PyFloat_AsDouble — None fails there, becomes NaN on the array path);
#   * the constraints over whole arrays (check_array_constraint's texts and ORDER — e.g. a
#     NaN Poisson mean reports "lam value too large" because the bound runs first), the
#     sampler-specific whole-array ufunc steps (uniform's np.subtract, triangular's
#     np.greater/np.equal, hypergeometric's np.add/np.less, the Generator negative_binomial's
#     Poisson-mean bound) with their ufunc broadcast errors;
#   * the stream: "draws": 2 records the SECOND call, so a call that over- or under-draws
#     (a read-ahead running past a position that draws nothing — weibull a == 0, poisson
#     lam == 0, a degrees of freedom whose df / 2 underflows to 0 — or RandomState's cached
#     Gaussian left in the wrong state) shows up in the next block.
# Parameters ride as real OPERANDS (layout_catalog.describe), so strided / reversed /
# F-order / transposed / 0-d views are rebuilt exactly on the C# side; Python scalars and
# None ride in params["bargs"] ({"op": k} | {"f": x} | {"i": n} | {"none": true}).
#
# Legacy int results are C long — int32 on this win-amd64 authoring host — and are widened
# to NumSharp's LP64 int64 like the scalar rnd cases (_RND_INT64_CAST). Inputs whose outcome
# DEPENDS on the long width (n / lam / ngood past 2^31, p == 0's integer cast of infinity,
# and the popsize overflow that makes Windows NumPy's legacy HRUA loop forever) cannot be
# authored here; NumSharp's unit tests pin them against Linux (LP64) NumPy.
_RB_DISTS = {
    "beta": ([[0.5, 2.0, 5.0], 2.0], "dd"),
    "exponential": ([[1.0, 2.0, 0.5]], "d"),
    "uniform": ([[0.0, 1.0, -1.0], 5.0], "dd"),
    "normal": ([[0.0, 1.0, -2.0], 2.0], "dd"),
    "standard_gamma": ([[0.5, 1.0, 3.0]], "d"),
    "gamma": ([[0.5, 1.0, 3.0], 2.0], "dd"),
    "f": ([[1.0, 5.0, 10.0], 7.0], "dd"),
    "noncentral_f": ([[1.0, 5.0, 10.0], 7.0, 1.5], "ddd"),
    "chisquare": ([[0.5, 3.0, 10.0]], "d"),
    "noncentral_chisquare": ([[0.5, 3.0, 10.0], 1.5], "dd"),
    "standard_t": ([[0.5, 3.5, 10.0]], "d"),
    "vonmises": ([[0.0, 1.0, -2.0], 2.0], "dd"),
    "pareto": ([[0.5, 3.0, 10.0]], "d"),
    "weibull": ([[0.0, 1.79, 5.0]], "d"),
    "power": ([[0.3, 2.5, 5.0]], "d"),
    "laplace": ([[0.0, 1.0, 2.0], 1.5], "dd"),
    "gumbel": ([[0.0, 1.0, 2.0], 1.5], "dd"),
    "logistic": ([[0.0, 1.0, 2.0], 1.5], "dd"),
    "lognormal": ([[0.0, 1.0, 2.0], 0.5], "dd"),
    "rayleigh": ([[1.0, 0.0, 2.0]], "d"),
    "wald": ([[3.0, 0.5, 1.0], 2.0], "dd"),
    "triangular": ([[0.0, 1.0, 2.0], 3.0, 10.0], "ddd"),
    "binomial": ([[10, 100, 1000], 0.4], "id"),
    "negative_binomial": ([[5.0, 0.5, 10.0], 0.4], "dd"),
    "poisson": ([[3.5, 100.0, 0.0]], "d"),
    "zipf": ([[3.0, 1.5, 1.05]], "d"),
    "geometric": ([[0.35, 0.1, 1.0]], "d"),
    "hypergeometric": ([[10, 100, 300], 200, 8], "iii"),
    "logseries": ([[0.6, 0.99, 0.0]], "d"),
}
_RB_NAN = float("nan")
_RB_INF = float("inf")
_RB_VIOLATIONS = [
    ("beta", 0, [1.0, 0.0], "a<=0"), ("beta", 1, [1.0, _RB_NAN], "b-nan"),
    ("exponential", 0, [1.0, -1.0], "scale<0"), ("exponential", 0, [1.0, -0.0], "scale-negzero"),
    ("normal", 1, [1.0, -1.0], "scale<0"), ("standard_gamma", 0, [1.0, -1.0], "shape<0"),
    ("gamma", 0, [1.0, -1.0], "shape<0"), ("gamma", 1, [1.0, -1.0], "scale<0"),
    ("f", 0, [1.0, 0.0], "dfnum<=0"), ("f", 1, [1.0, 0.0], "dfden<=0"),
    ("noncentral_f", 2, [1.0, -1.0], "nonc<0"), ("chisquare", 0, [1.0, 0.0], "df<=0"),
    ("noncentral_chisquare", 1, [1.0, -1.0], "nonc<0"), ("standard_t", 0, [1.0, 0.0], "df<=0"),
    ("vonmises", 1, [1.0, -1.0], "kappa<0"), ("pareto", 0, [1.0, 0.0], "a<=0"),
    ("weibull", 0, [1.0, -1.0], "a<0"), ("power", 0, [1.0, 0.0], "a<=0"),
    ("laplace", 1, [1.0, -1.0], "scale<0"), ("gumbel", 1, [1.0, -1.0], "scale<0"),
    ("logistic", 1, [1.0, -1.0], "scale<0"), ("lognormal", 1, [1.0, -1.0], "sigma<0"),
    ("rayleigh", 0, [1.0, -1.0], "scale<0"), ("wald", 0, [1.0, 0.0], "mean<=0"),
    ("wald", 1, [1.0, 0.0], "scale<=0"),
    ("binomial", 0, [10, -1], "n<0"), ("binomial", 1, [0.5, 1.5], "p>1"), ("binomial", 1, [0.5, _RB_NAN], "p-nan"),
    ("negative_binomial", 0, [5.0, _RB_NAN], "n-nan"), ("negative_binomial", 0, [5.0, 0.0], "n<=0"),
    ("negative_binomial", 1, [0.5, 0.0], "p<=0"), ("negative_binomial", 1, [0.5, 1e-18], "maxlam"),
    ("negative_binomial", 0, [5.0, 1e19], "maxlam2"),
    ("poisson", 0, [1.0, -1.0], "lam<0"), ("poisson", 0, [1.0, 1e20], "lam-big"), ("poisson", 0, [1.0, _RB_NAN], "lam-nan"),
    ("zipf", 0, [2.0, 1.0], "a<=1"), ("zipf", 0, [2.0, _RB_NAN], "a-nan"),
    ("geometric", 0, [0.5, 0.0], "p<=0"), ("geometric", 0, [0.5, 1.5], "p>1"),
    ("logseries", 0, [0.5, 1.0], "p>=1"), ("logseries", 0, [0.5, -0.1], "p<0"),
    ("hypergeometric", 0, [10, -1], "ngood<0"), ("hypergeometric", 0, [10, 10 ** 9], "ngood>=1e9"),
    ("hypergeometric", 2, [8, 300], "sum<nsample"), ("hypergeometric", 1, [10, -1], "nbad<0"),
    ("triangular", 0, [0.0, 5.0], "left>mode"), ("triangular", 2, [10.0, 2.0], "mode>right"),
    ("triangular", 0, [0.0, 10.0], "left==right"),
    ("uniform", 1, [5.0, _RB_INF], "range-inf"), ("uniform", 1, [5.0, -1.0], "range<0"),
]
# Long mixed streams: each parameter on its own axis, so the draws cycle through every combination (every internal
# branch of every sampler, the rejection retries, and the read-ahead refill boundaries of a (40, ...) output).
_RB_STREAMS = {
    "beta": [[0.05, 0.5, 1.0, 2.0, 5.0, 1e-200, 0.9, 30.0], [0.7, 3.0]],
    "exponential": [[0.5, 1.0, 2.0, 10.0]],
    "uniform": [[0.0, -5.0, 0.5, -3e5], [1.0, 2.5]],
    "normal": [[0.0, -2.0, 5.0, 1e3], [1.0, 0.5, 3.0]],
    "standard_gamma": [[0.05, 0.5, 1.0, 1.5, 3.0, 50.0, 0.999]],
    "gamma": [[0.3, 1.0, 2.5, 20.0], [1.0, 0.5]],
    "f": [[0.5, 1.0, 5.0, 40.0], [0.8, 7.0]],
    "noncentral_f": [[0.5, 1.0, 5.0, 40.0], [0.8, 7.0], [0.0, 1.5, 20.0]],
    "chisquare": [[0.3, 1.0, 2.0, 9.0, 100.0]],
    "noncentral_chisquare": [[0.5, 1.0, 1.5, 3.0, 10.0], [0.0, 0.5, 2.0, 30.0]],
    "standard_t": [[0.5, 1.0, 3.5, 30.0, 1e4]],
    "vonmises": [[0.0, 1.0, -2.0], [1e-9, 1e-6, 0.5, 4.0, 1e3, 2e6]],
    "pareto": [[0.5, 1.0, 3.0, 20.0]],
    "weibull": [[0.5, 1.0, 1.79, 5.0]],
    "power": [[0.3, 1.0, 2.5, 5.0]],
    "laplace": [[0.0, 1.0, -3.0], [1.5, 0.2]],
    "gumbel": [[0.0, 1.0, -3.0], [1.5, 0.2]],
    "logistic": [[0.0, 1.0, -3.0], [1.5, 0.2]],
    "lognormal": [[0.0, 1.0, -1.0], [0.5, 1.5]],
    "rayleigh": [[0.5, 1.0, 2.0, 9.0]],
    "wald": [[0.5, 1.0, 3.0, 20.0], [0.3, 2.0]],
    "triangular": [[0.0, 1.0, -2.0], [3.0, 4.0], [10.0, 12.0]],
    "binomial": [[1, 10, 50, 100, 1000, 100000], [0.05, 0.3, 0.5, 0.8, 0.97]],
    "negative_binomial": [[0.5, 1.0, 5.0, 40.0, 500.0], [0.1, 0.4, 0.9, 1.0]],
    "poisson": [[0.5, 3.5, 9.99, 10.0, 25.0, 1000.0, 1e6]],
    "zipf": [[1.05, 1.5, 2.0, 3.0, 10.0]],
    "geometric": [[0.01, 0.1, 0.3, 0.34, 0.5, 0.9, 1.0]],
    "hypergeometric": [[15, 100, 300, 1000], [200, 30], [12, 40, 8]],
    "logseries": [[0.0, 0.3, 0.6, 0.9, 0.99, 0.9999]],
}
# Legacy inputs whose OUTCOME depends on the width of C long, replaced/skipped in the legacy half (this win-amd64
# host's long is 32-bit; NumSharp models LP64). legacy_random_zipf rejects every candidate above LONG_MAX, so a small
# exponent (a < 2: 1.5, 1.05 draw candidates past 2^31 routinely) takes a different stream on the two widths — the
# legacy half uses exponents >= 2 (P(X > 2^31) <= 5e-10 per attempt). legacy negative_binomial's p == 0 / tiny p /
# huge n reach the Poisson's integer cast of an infinite or > 2^31 mean, which truncates differently.
_RB_LEGACY_DISTS = {"zipf": ([[3.0, 2.5, 2.0]], "d")}
_RB_LEGACY_STREAMS = {"zipf": [[2.0, 2.5, 3.0, 5.0, 10.0]]}
_RB_LEGACY_SKIP = {("negative_binomial", "viol:p<=0"), ("negative_binomial", "viol:maxlam"),
                   ("negative_binomial", "viol:maxlam2"), ("negative_binomial", "huge_n")}
# Portable (pure MT19937/PCG64 bits + exactly-rounded IEEE): the uniform transform everywhere, and the Generator's
# triangular (the scalar tiers classify them the same way). Every other sampler's VALUES consume the host libm.
_RB_PORTABLE = {"legacy": {"uniform"}, "gen": {"uniform", "triangular"}}


class _RBView:
    """An array parameter passed as a VIEW of a C-contiguous base (a slice, a reversal, an F-order reshape)."""

    def __init__(self, base, view):
        self.base, self.view = base, view


def _gen_random_broadcast(api, n):
    """The array-parameter cases of one API ('legacy' -> op rnd, 'gen' -> op grnd); returns (portable, host, n)."""
    op = "rnd" if api == "legacy" else "grnd"
    key = "dist" if api == "legacy" else "method"
    portable, host = [], []
    A = np.array
    nan, inf = _RB_NAN, _RB_INF

    def run(dist, case, args, size="__none__", draws=1, dtype=None, out=None):
        nonlocal n
        operands, bargs, real = [], [], []
        for a in args:
            if a is None:
                bargs.append({"none": True})
                real.append(None)
            elif isinstance(a, _RBView):
                bargs.append({"op": len(operands)})
                operands.append(describe(a.base, a.view))
                real.append(a.view)
            elif isinstance(a, np.ndarray):
                base = np.array(a, copy=True, order="C")   # np.array keeps a 0-d array 0-d
                bargs.append({"op": len(operands)})
                operands.append(describe(base, base))
                real.append(base)
            elif isinstance(a, bool):
                raise TypeError("bool scalars are not used as Python-literal parameters")
            elif isinstance(a, int):
                bargs.append({"i": a})
                real.append(a)
            else:
                bargs.append({"f": float(a)})
                real.append(float(a))
        params = {key: dist, "seed": 42, "args": [], "bargs": bargs}
        call_kw = {}
        if size != "__none__":
            params["size"] = [size] if isinstance(size, int) else list(size)
            call_kw["size"] = size
        if dtype is not None:
            params["dtype"] = dtype
            call_kw["dtype"] = np.dtype(dtype)
        if out is not None:
            params["out"] = out
        if draws != 1:
            params["draws"] = draws
        if api == "legacy" and (dist, case) in _RB_LEGACY_SKIP:
            return
        layout = f"{op}_bcast"
        cid = f"{op}/{dist}/bcast:{case}/{n}"
        obj = np.random.RandomState(42) if api == "legacy" else np.random.default_rng(42)
        try:
            r = None
            for _ in range(draws):
                if out is not None:
                    call_kw["out"] = np.empty(tuple(out["shape"]), dtype=out["dtype"], order=out["order"])
                r = getattr(obj, dist)(*real, **call_kw)
            r = np.asarray(r)
            if api == "legacy" and r.dtype == np.int32:
                r = r.astype(np.int64)   # C long -> NumSharp's LP64 int64 (values fit: see the block comment)
            into = portable if dist in _RB_PORTABLE[api] else host
            into.append(_case(op, params, operands, _arr_expected(r), layout, "stream", cid=cid))
        except Exception as e:  # noqa: BLE001 - NumPy's exception IS the recorded contract
            err = _error_case(op, params, operands, e, layout, cid=cid)   # validation: host-independent
            # The C# harness trims its own message, so NumPy's trailing blank (the ufunc broadcast error ends in a space)
            # is trimmed too — the _poly_exc rule; the text is otherwise verbatim.
            err["error"]["text"] = err["error"]["text"].strip()
            portable.append(err)
        n += 1

    dists = dict(_RB_DISTS, **(_RB_LEGACY_DISTS if api == "legacy" else {}))
    streams = dict(_RB_STREAMS, **(_RB_LEGACY_STREAMS if api == "legacy" else {}))
    for dist, (valid, kinds) in dists.items():
        base = [A(v) if isinstance(v, list) else v for v in valid]
        run(dist, "b1", base)
        run(dist, "b1", base, draws=2)
        run(dist, "b1_size43", base, (4, 3))
        run(dist, "b1_size1", base, (1,))
        run(dist, "b1_size42", base, (4, 2))
        run(dist, "b1_size_empty", base, ())
        run(dist, "b1_size_int3", base, 3)
        run(dist, "b1_size0", base, (0, 3))
        if len(valid) > 1:
            col = A(valid[0][:2]).reshape(2, 1)
            second = valid[1]
            row = A([second, second * 1.5 if kinds[1] == "d" else second + 1, second * 2 if kinds[1] == "d" else second + 2])
            if dist == "hypergeometric":
                row = A([200, 150, 400])
            if dist == "triangular":
                row = A([3.0, 4.0, 5.0])
            rest = [A(v) if isinstance(v, list) else v for v in valid[2:]]
            run(dist, "b2", [col, row] + rest)
            run(dist, "b2", [col, row] + rest, draws=2)
            run(dist, "b2_size", [col, row] + rest, (3, 2, 3))
            run(dist, "mismatch", [A(valid[0][:2]), row] + rest)
            run(dist, "mismatch_size", [A(valid[0][:2]), row[:2]] + rest, (3,))
            run(dist, "second_arr", [A(valid[0][0]), row] + rest)
        zd = [A(v[0] if isinstance(v, list) else v) for v in valid]
        run(dist, "zero_d", zd)
        run(dist, "zero_d_size3", zd, 3)
        run(dist, "zero_d_size_empty", zd, ())
        run(dist, "empty", [A(valid[0][:0], dtype=np.int64 if kinds[0] == "i" else np.float64)] + base[1:])
        if kinds[0] == "d":
            ints = A([1, 2, 3], dtype=np.int64)
            if dist == "zipf":
                ints = A([2, 3, 4], dtype=np.int64)
            if dist == "geometric":
                ints = A([1, 1, 1], dtype=np.int64)
            if dist == "logseries":
                ints = A([0, 0, 0], dtype=np.int64)
            run(dist, "int_param", [ints] + base[1:])
            run(dist, "f32_param", [A(valid[0], dtype=np.float32)] + base[1:])
            doubled = A(valid[0] + valid[0])
            run(dist, "strided_param", [_RBView(doubled, doubled[::2])] + base[1:])
            single = A(valid[0])
            run(dist, "reversed_param", [_RBView(single, single[::-1])] + base[1:])
        else:
            run(dist, "float_int_param", [A([10.5, 20.0])] + base[1:])
            run(dist, "uint64_param", [A([10, 20], dtype=np.uint64)] + base[1:])
            run(dist, "int32_param", [A([10, 20], dtype=np.int32)] + base[1:])
        run(dist, "complex_param", [A([1 + 0j, 2 + 0j])] + base[1:])
        run(dist, "complex_0d_param", [A(1 + 0j)] + base[1:])
    for dist, idx, bad, label in _RB_VIOLATIONS:
        valid, _ = dists[dist]
        args = [A(v) if isinstance(v, list) else v for v in valid]
        args[idx] = A(bad)
        args = [a if (not isinstance(a, np.ndarray) or a.ndim == 0 or a.shape == (2,)) else a[:2] for a in args]
        run(dist, f"viol:{label}", args)

    # ---- layouts, dtypes, draw-free positions, None ----
    f_base = A([0.0, 2.0, 1.0, 3.0])
    run("normal", "F_param", [_RBView(f_base, f_base.reshape((2, 2), order="F")), 1.0])
    t_base = A([[0.0, 1.0, 5.0], [2.0, 3.0, 7.0]])
    run("normal", "negstride_2d_param", [_RBView(t_base, t_base[:, ::-1]), 1.0])
    run("normal", "transposed_param", [_RBView(t_base, t_base.T), A([1.0, 2.0])])
    run("normal", "bool_param", [A([True, False]), 1.0])
    run("normal", "uint64_param", [A([1, 2], dtype=np.uint64), 1.0])
    run("normal", "f16_param", [A([0.1, 0.2], dtype=np.float16), 1.0])
    run("normal", "int8_param", [A([-3, 4], dtype=np.int8), 1.0])
    run("normal", "nan_scale", [A([0.0, 1.0]), A([nan, 1.0])])
    run("exponential", "nan", [A([nan, 1.0])])
    run("weibull", "zero_mid", [A([1.0, 0.0, 2.0, 0.0, 3.0])])
    run("weibull", "zero_mid", [A([1.0, 0.0, 2.0, 0.0, 3.0])], draws=2)
    run("weibull", "all_zero", [A([0.0, 0.0])], (3, 2))
    run("gamma", "zero_shape", [A([0.0, 1.0, 0.0]), 2.0], draws=2)
    run("gamma", "zero_shape_end", [A([2.0, 0.0, 0.0]), 1.0], draws=2)
    run("standard_gamma", "zero_shape", [A([0.0, 2.0, 0.0])], draws=2)
    run("poisson", "zero_mid", [A([0.0, 3.0, 0.0, 30.0])], draws=2)
    run("binomial", "zero_mid", [A([0, 5, 100, 0]), A([0.5, 0.0, 0.5, 0.3])], draws=2)
    run("noncentral_chisquare", "nan_nonc", [A([3.0, 3.0, 0.5]), A([nan, 1.0, nan])], draws=2)
    # A degrees of freedom that HALVES to 0 (5e-324 / 2 rounds to even = 0): its chi-square draws nothing, so a
    # position may draw nothing at all — the read-ahead must not run past it (checked by the second block).
    run("chisquare", "half_zero", [A([2.0, 5e-324, 5e-324])], draws=2)
    run("f", "half_zero", [A([5e-324, 5e-324]), A([2.0, 5e-324])], draws=2)
    run("noncentral_f", "half_zero", [A([5e-324, 5e-324]), A([2.0, 5e-324]), A([0.0, 0.0])], draws=2)
    # The same edge with a STREAM-SENSITIVE first value: the two cases above return 0 / NaN whatever the draws were,
    # so a surplus word taken by a wrong read-ahead would only show in the stream position, which a value corpus sees
    # through the second call's values alone.
    run("f", "half_zero_seq", [A([2.0, 5e-324, 5e-324]), A([2.0, 5e-324, 5e-324])], draws=2)
    run("noncentral_f", "half_zero_seq", [A([2.0, 5e-324, 5e-324]), A([2.0, 5e-324, 5e-324]), A([0.0, 0.0, 0.0])], draws=2)
    run("noncentral_f", "nan_nonc_tiny_den", [A([3.0, 3.0]), A([2.0, 5e-324]), A([1.0, nan])], draws=2)
    run("noncentral_chisquare", "half_zero", [A([3.0, 5e-324, 5e-324]), A([1.0, 0.0, 5e-324])], draws=2)
    run("standard_t", "half_zero", [A([3.0, 5e-324, 5e-324, 2.0])], draws=2)
    # A drawing position with a SMALL, known draw count followed by positions that draw nothing: the only shape in
    # which a read-ahead that assumed "every position draws" takes words NumPy never takes (a later drawing position
    # would absorb the surplus). One per draw-free condition the samplers have.
    # Where one drawing position's value is too coarse to reveal a shifted stream (a count that is usually 0 or 1),
    # a RUN of drawing positions precedes the draw-free run, so the second call's run of values shows any shift.
    run("vonmises", "nan_tail", [0.0, A([1e-9, nan, nan])], draws=2)            # kappa < 1e-8: one uniform; NaN: none
    run("poisson", "zero_tail", [A([0.1] * 50 + [0.0] * 50)], draws=2)          # ~1.1 uniforms each; lam == 0: none
    run("standard_gamma", "zero_tail", [A([1.0, 0.0, 0.0])], draws=2)           # one exponential; shape 0: none
    run("binomial", "zero_tail", [A([5] * 40 + [0] * 40), 0.5], draws=2)        # Generator n == 0: none (legacy draws)
    run("noncentral_chisquare", "nan_tail", [A([3.0, 3.0, 3.0]), A([1.0, nan, nan])], draws=2)  # Generator NaN: none
    run("standard_t", "half_zero_tail", [A([2.0, 5e-324])], draws=2)           # legacy: cached normal + gamma(0)
    run("hypergeometric", "zero_colour_tail",                                  # sample 1: one word each; absent colour: none
        [A([10] * 30 + [0] * 15 + [5] * 15), A([10] * 30 + [5] * 15 + [0] * 15), A([1] * 30 + [3] * 15 + [4] * 15)],
        draws=2)
    if api == "gen":
        run("zipf", "big_a_tail", [A([2.0] * 20 + [2000.0] * 40)], draws=2)     # a >= 1025: 1 without a draw
        run("vonmises", "kappa_regimes", [0.0, A([nan, 1.0, 1e-9, 2e6, inf, 1e-6])], draws=2)
        run("zipf", "big_a", [A([2000.0, 2.0, inf])], draws=2)
    else:
        # NumPy's legacy samplers never return for kappa = inf (a NaN envelope) or zipf a >= 1025.
        run("vonmises", "kappa_regimes", [0.0, A([nan, 1.0, 1e-9, 2e6, 1e-6])], draws=2)
        # RandomState's cached Gaussian carried into the next call (odd counts leave a cached half).
        run("normal", "gauss_cache_odd", [A([0.0, 1.0, 2.0]), 1.0], draws=2)
        run("lognormal", "gauss_cache_odd", [A([0.0, 1.0, 2.0]), 0.5], draws=2)
        run("gamma", "gauss_cache", [A([3.0, 0.5, 7.0]), 1.0], draws=2)
        run("standard_t", "gauss_cache", [A([3.0, 5.0, 7.0])], draws=2)
        run("wald", "gauss_cache", [A([1.0, 2.0, 3.0]), 1.0], draws=2)
        run("noncentral_chisquare", "gauss_cache", [A([3.0, 5.0, 0.5]), A([1.0, 2.0, 1.0])], draws=2)
        run("beta", "gauss_cache", [A([3.0, 0.5]), A([2.0, 0.7])], draws=2)
        run("negative_binomial", "n_nan", [A([nan, 3.0]), 0.5], draws=2)
    run("geometric", "mixed", [A([0.5, 0.1, 0.9, 0.2, 1.0])], draws=2)
    run("hypergeometric", "mixed", [A([5, 100, 0, 500]), A([5, 200, 10, 500]), A([3, 50, 5, 0])], draws=2)
    run("hypergeometric", "neg_and_sum", [A([-100]), 50, 10])
    run("hypergeometric", "neg_nbad", [A([10]), A([-1]), 5])
    run("hypergeometric", "neg_nsample", [A([10]), 5, A([-1])])
    run("hypergeometric", "zero_nsample", [A([10]), 5, A([0])])
    run("uniform", "neg_range", [A([1.0]), A([0.0])])
    run("uniform", "nan", [A([nan]), 1.0])
    run("uniform", "zero_range", [A([1.0]), A([1.0])])
    run("triangular", "eq", [A([1.0]), 1.0, 1.0])
    run("triangular", "nan", [A([nan]), 1.0, 2.0])
    run("negative_binomial", "mismatch_size", [A([5.0, 6.0]), 0.5], (3,))
    run("negative_binomial", "nan_p", [A([5.0]), A([nan])])
    run("negative_binomial", "huge_n", [A([5.0, 1e18]), 0.5])
    run("binomial", "p_then_n", [A([-1]), A([2.0])])
    run("binomial", "bool_n", [A([True, False]), 0.5])
    run("f", "nan_params", [A([nan, 1.0]), 1.0])
    run("wald", "nan", [A([nan, 1.0]), 1.0])
    run("logseries", "zero", [A([0.0, 0.5])])
    run("normal", "size_tuple3", [A([0.0, 1.0]), 1.0], (3, 2))
    run("normal", "size_zero_bad", [A([0.0, 1.0]), 1.0], (0,))
    run("normal", "param_empty_2d", [np.empty((0, 3)), A([1.0, 2.0, 3.0])])
    run("normal", "param_1elem", [A([5.0]), 1.0])
    run("normal", "param_1elem_size", [A([5.0]), 1.0], (2, 3))
    run("normal", "param_3d", [A([[[0.0], [10.0]]]), A([1.0, 2.0])])
    run("beta", "None_a", [None, 1.0])
    run("beta", "None_b_arr", [A([1.0, 2.0]), None])
    run("binomial", "None_n", [None, 0.5])
    run("binomial", "None_p", [5, None])
    run("binomial", "None_p_arr", [A([5, 6]), None])
    run("hypergeometric", "None", [None, 1, 1])
    run("negative_binomial", "None_n", [None, 0.5])
    run("negative_binomial", "None_p_arr", [A([5.0]), None])
    run("triangular", "None_mode", [0.0, None, 1.0])
    run("gamma", "None_shape_scalar_bad_scale", [None, -1.0])
    run("gamma", "bad_shape_None_scale", [-1.0, None])

    if api == "gen":
        # standard_gamma's float32 loop (FORCECAST: complex keeps its real part; the constraint runs on the float32
        # values) and its out= contract (C-contiguous, the loop dtype, a shape the parameter broadcasts into).
        run("standard_gamma", "f32_arr", [A([0.5, 1.0, 3.0])], dtype="float32")
        run("standard_gamma", "f32_arr_size", [A([0.5, 1.0, 3.0])], (2, 3), dtype="float32")
        run("standard_gamma", "f32_complex_arr", [A([1 + 2j, 2 + 0j])], dtype="float32")
        run("standard_gamma", "f32_complex_0d", [A(2 + 1j)], dtype="float32")
        run("standard_gamma", "f64_complex_0d", [A(2 + 1j)])
        run("standard_gamma", "f32_neg_arr", [A([1.0, -1.0])], dtype="float32")
        run("standard_gamma", "f32_tiny", [A([1e-50, 2.0])], dtype="float32")
        run("standard_gamma", "f32_zero_d", [A(2.5)], dtype="float32")
        run("standard_gamma", "f32_zero_d_size", [A(2.5)], (3,), dtype="float32")
        run("standard_gamma", "f32_None", [None], dtype="float32")
        run("standard_gamma", "out_ok", [A([0.5, 1.0, 3.0])], out={"dtype": "float64", "shape": [2, 3], "order": "C"})
        run("standard_gamma", "out_mismatch", [A([0.5, 1.0])], out={"dtype": "float64", "shape": [3], "order": "C"})
        run("standard_gamma", "out_size_mismatch", [A([0.5, 1.0, 3.0])], (4, 3),
            out={"dtype": "float64", "shape": [2, 3], "order": "C"})
        run("standard_gamma", "out_small", [A([0.5, 1.0, 3.0])], out={"dtype": "float64", "shape": [1], "order": "C"})
        run("standard_gamma", "out_size_ok", [A([0.5, 1.0, 3.0])], (2, 3),
            out={"dtype": "float64", "shape": [2, 3], "order": "C"})
        run("standard_gamma", "f32_out_ok", [A([0.5, 1.0, 3.0])], dtype="float32",
            out={"dtype": "float32", "shape": [2, 3], "order": "C"})
        run("standard_gamma", "out_F", [A([0.5, 1.0])], out={"dtype": "float64", "shape": [2, 2], "order": "F"})
        run("standard_gamma", "out_wrongdtype", [A([0.5, 1.0])], out={"dtype": "float32", "shape": [2], "order": "C"})
        run("standard_gamma", "out_zero_d_param", [A(2.0)], out={"dtype": "float64", "shape": [3], "order": "C"})

    for dist, ps in streams.items():
        arrs = []
        for k, p in enumerate(ps):
            shape = [1] * len(ps)
            shape[k] = len(p)
            dt = np.int64 if dist == "hypergeometric" or (dist == "binomial" and k == 0) else np.float64
            arrs.append(A(p, dtype=dt).reshape(shape))
        bshape = np.broadcast_shapes(*[a.shape for a in arrs])
        run(dist, "stream", arrs, (40,) + tuple(bshape))
        run(dist, "stream", arrs, (40,) + tuple(bshape), draws=2)
        run(dist, "stream_nosize", arrs)
    return portable, host, n


def gen_windows():
    """Window functions bartlett/blackman/hamming/hanning/kaiser — pure GENERATORS.

    Output is ALWAYS float64 (NumPy forces it via np.array([0.0, M])), so there is no
    dtype axis: the single carrier operand is a float64 placeholder the C# side ignores
    (M / beta come from params, exactly like the tri generator). M sweeps the
    empty / single / even / odd / multi-SIMD-chunk corners; kaiser additionally sweeps
    beta (0 = rectangular ... 20 = very narrow), crossing i0's Chebyshev split at x == 8.
    """
    cases = []
    n = 0
    carrier = _cbase((1,), np.dtype("float64"))

    def emit(opname, params, r):
        nonlocal n
        r = np.asarray(r)
        cases.append({
            "id": f"{opname}/{n}",
            "op": opname,
            "params": params,
            "operands": [describe(carrier, carrier)],
            "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                         "buffer": np.ascontiguousarray(r).tobytes().hex()},
            "layout": "gen",
            "valueclass": "mixed",
        })
        n += 1

    Ms = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 33, 64, 65, 128, 257]
    # NumPy's stub types M as _FloatLike_co: a NON-integer M yields a fractional-length
    # window (len == len(arange(1-M, M, 2))), and kaiser's 0 < M < 1 is a ONE-element array
    # (no M<1 guard) while a cosine window's 0 < M < 1 is empty. These gate the double-M path.
    FloatMs = [-0.5, 0.5, 0.999, 1.5, 2.5, 4.99, 5.5, 5.7, 12.3, 33.7]
    for opname, fn in [("bartlett", np.bartlett), ("blackman", np.blackman),
                       ("hamming", np.hamming), ("hanning", np.hanning)]:
        for M in Ms:
            emit(opname, {"M": M}, fn(M))
        for M in FloatMs:
            emit(opname, {"M": M}, fn(M))
    for M in Ms:
        for beta in [0.0, 0.5, 2.5, 5.0, 6.0, 8.0, 8.6, 10.0, 14.0, 20.0]:
            emit("kaiser", {"M": M, "beta": beta}, np.kaiser(M, beta))
    for M in FloatMs:
        for beta in [0.0, 5.0, 14.0]:
            emit("kaiser", {"M": M, "beta": beta}, np.kaiser(M, beta))
    return cases


# ---- np.evaluate / NDExpr: fused-expression differential tier -----------------------------
#
# NumPy has no expression fusion, so the oracle for a fused tree is the UNFUSED NumPy chain
# evaluated node by node — exactly the contract np.evaluate claims ("bit-compatible with the
# unfused NumPy sequence, per-node result_type incl. NEP50 weak literals"). Trees are encoded
# in params["expr"] as a prefix grammar over the NDExpr node catalog; the C# side
# (OpRegistry.Evaluate.cs) parses the same string into an NDExpr and runs np.evaluate.
#
#   in<k>                      operand k (the k-th entry of "operands")
#   li:<int>  lu:<uint64>      weak Python int (lu: above long.MaxValue — only uint64 carries it)
#   lf:<float>                 weak Python float ("nan"/"inf" spelled as Python prints them)
#   lb:0|1                     weak Python bool
#   lc:<re>;<im>               weak Python complex
#   lh:<float>                 STRONG np.float16 scalar (Half in C#)
#   <fn>(<arg>,...)            a node — see _EV_BINARY / _EV_UNARY / where
#
# Reductions are root-only in NDExpr, so they ride params["reduce"] = {kind, axis, keepdims}
# over the tree in "expr" (kind: sum | prod | min | max | mean).
#
# "out" cases (params["out"] = true) carry the out view as the LAST operand and record the
# out_where tuple shape [returned, out_base] so a kernel writing outside a strided / offset
# window is caught.

_EV_BINARY = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.true_divide,
    "mod": np.remainder, "pow": np.power, "floordiv": np.floor_divide, "atan2": np.arctan2,
    "and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor,
    "min": np.minimum, "max": np.maximum,
    "eq": np.equal, "ne": np.not_equal, "lt": np.less, "le": np.less_equal,
    "gt": np.greater, "ge": np.greater_equal,
    # Phase 4 binary node coverage. min/max already cover the NaN-propagating maximum/minimum;
    # these are the NaN-ignoring fmax/fmin, the float-tier family (copysign/nextafter/logaddexp/
    # logaddexp2/hypot/heaviside), the integer-only gcd/lcm + shifts, and C-style fmod. Cells where
    # NumPy has no loop (float shift, complex copysign, bool-bool gcd, ...) either RAISE a verbatim
    # error (recorded — the "not supported for the input types" family) or a non-verbatim error
    # (auto-skipped — gcd/lcm's "did not contain a loop"), so ok_for needs no per-op integer filter.
    "fmax": np.fmax, "fmin": np.fmin, "fmod": np.fmod,
    "copysign": np.copysign, "nextafter": np.nextafter,
    "logaddexp": np.logaddexp, "logaddexp2": np.logaddexp2,
    "hypot": np.hypot, "heaviside": np.heaviside,
    "gcd": np.gcd, "lcm": np.lcm,
    "lshift": np.left_shift, "rshift": np.right_shift,
    # Phase 4.3 logical nodes (LogicalNode): a NONZERO-TEST of each operand at its own dtype, result
    # ALWAYS bool — `(a != 0) op (b != 0)` — so they accept EVERY dtype (complex included) with no
    # no-loop cell, unlike the bitwise and/or/xor. Bit-exact vs NumPy (an integer/mask combination, no
    # host-libm), no ULP excuse.
    "land": np.logical_and, "lor": np.logical_or, "lxor": np.logical_xor,
}
_EV_UNARY = {
    "neg": np.negative, "abs": np.absolute, "sqrt": np.sqrt, "square": np.square,
    "recip": np.reciprocal, "sign": np.sign, "cbrt": np.cbrt,
    "exp": np.exp, "exp2": np.exp2, "expm1": np.expm1,
    "log": np.log, "log2": np.log2, "log10": np.log10, "log1p": np.log1p,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
    "deg2rad": np.deg2rad, "rad2deg": np.rad2deg,
    # round = np.round(x) (decimals=0): dtype-PRESERVING like NDExpr.Round / np.round_, an identity on
    # integers. rint = np.rint(x): the TRUE ufunc form of round-half-to-even — same VALUE (banker's
    # rounding, and NDExpr.Rint aliases Round's kernel), but float-TIER (bool/i1/u1->f16, i2/u2->f32,
    # i4+->f64; float/complex/decimal preserved), so rint(int32) is a float64 where round(int32) is the
    # int32 identity. Both cells are BIT-EXACT (Math.Round is portable), no ULP excuse; complex rounds
    # both lanes. rint has a loop for every dtype (no rejection cell).
    "floor": np.floor, "ceil": np.ceil, "round": lambda x: np.round(x), "rint": np.rint, "trunc": np.trunc,
    "not": np.invert, "lnot": np.logical_not,
    "isnan": np.isnan, "isfinite": np.isfinite, "isinf": np.isinf,
    # Phase 4 unary node coverage — the engine unary ufuncs that gained an NDExpr node. Every VALUE
    # cell is BIT-EXACT (identity / sign-bit clear / popcount / pure-bit-increment spacing — no
    # host-libm), so NO ULP excuse. The no-loop cells split by message: positive(bool) and
    # isposinf/isneginf(complex) raise a NON-verbatim message (auto-skipped by emit(); unit-test-
    # pinned), while signbit/spacing/fabs(complex) and bitwise_count(float/half/complex) raise the
    # RECORDED "not supported for the input types" family (the C# typing reproduces it verbatim).
    "positive": np.positive, "conj": np.conjugate, "fabs": np.fabs, "spacing": np.spacing,
    "signbit": np.signbit, "isposinf": np.isposinf, "isneginf": np.isneginf,
    "bitwise_count": np.bitwise_count,
    # Phase 4.1b — the complex→real component extractors (NOT ufuncs). complex128 → float64 (real lane
    # / imag lane / atan2(im,re)); a REAL input is real=identity (dtype PRESERVED), imag=zeros (dtype
    # PRESERVED), angle=arctan2(0,x) at NumPy's per-dtype float tier. real/imag are BIT-EXACT (pure lane
    # extract / identity / zeros — no host-libm); angle rides atan2 (host-libm), bit-exact only within the
    # host-pinned evaluate tier (win-amd64 shares MSVC ucrtbase). No dtype has a no-loop / rejection cell.
    "real": np.real, "imag": np.imag, "angle": np.angle,
}
_EV_REDUCE = {
    "sum": lambda a, ax, kd: np.sum(a, axis=ax, keepdims=kd),
    "prod": lambda a, ax, kd: np.prod(a, axis=ax, keepdims=kd),
    "min": lambda a, ax, kd: np.min(a, axis=ax, keepdims=kd),
    "max": lambda a, ax, kd: np.max(a, axis=ax, keepdims=kd),
    "mean": lambda a, ax, kd: np.mean(a, axis=ax, keepdims=kd),
    # P2 M4 — presence / count / NaN-aware reductions. any/all/count_nonzero are order-independent
    # (bool / int64 result), so they are EXACT at every dtype. nansum/nanprod compose Sum/Prod over
    # Where(isnan, 0/1, x): bit-exact where the plain Sum/Prod diverts (float32/float64 + complex SUM).
    "any": lambda a, ax, kd: np.any(a, axis=ax, keepdims=kd),
    "all": lambda a, ax, kd: np.all(a, axis=ax, keepdims=kd),
    "count_nonzero": lambda a, ax, kd: np.count_nonzero(a, axis=ax, keepdims=kd),
    "nansum": lambda a, ax, kd: np.nansum(a, axis=ax, keepdims=kd),
    "nanprod": lambda a, ax, kd: np.nanprod(a, axis=ax, keepdims=kd),
    # P2 M4-tail — ORDER-INDEPENDENT range / NaN-aware min-max. Bit-exact at EVERY dtype (min / max /
    # their difference do not depend on summation order), so they carry NO E1 excuse — a regression is
    # red. ptp preserves dtype (integer/unsigned overflow wraps) and PROPAGATES a NaN; nanmin/nanmax
    # SKIP NaN (an all-NaN slice → NaN).
    "ptp": lambda a, ax, kd: np.ptp(a, axis=ax, keepdims=kd),
    "nanmin": lambda a, ax, kd: np.nanmin(a, axis=ax, keepdims=kd),
    "nanmax": lambda a, ax, kd: np.nanmax(a, axis=ax, keepdims=kd),
    # P2 M4c — the int64 INDEX kinds (an index, not a value → always intp). Their result depends on the
    # C-order tie/NaN rule (first maximum / first NaN wins), but NumSharp reduces the SAME fresh
    # C-contiguous materialized child, so it is bit-exact — NO E1 excuse.
    "argmax": lambda a, ax, kd: np.argmax(a, axis=ax, keepdims=kd),
    "argmin": lambda a, ax, kd: np.argmin(a, axis=ax, keepdims=kd),
    # P2 M4c summation kinds — NanMean / Var / Std. Host-computed over the materialized child with the
    # SAME NumPy-exact pairwise sum the M1/M2 diverts use (NOT delegated to the engine's drifting
    # np.nanmean/np.var), so they are BIT-EXACT for the 13 supported dtypes with NO E1 excuse. var/std
    # carry a ddof; a complex128 var/std is a REAL float64. float16 + decimal are unsupported (a
    # bit-exact reduction there needs a pairwise kernel that dtype lacks) and are held out of this tier.
    "nanmean": lambda a, ax, kd: np.nanmean(a, axis=ax, keepdims=kd),
    "var": lambda a, ax, kd, dd: np.var(a, axis=ax, keepdims=kd, ddof=dd),
    "std": lambda a, ax, kd, dd: np.std(a, axis=ax, keepdims=kd, ddof=dd),
}


# C6 combinators (NDExpr.Combinators.cs) — the macro / decision vocabulary. Each combinator is a PURE
# COMPOSITION of the primitive nodes, so its NumPy reference here is the SAME composition spelled with
# NumPy ufuncs, honoring the WEAK Python scalars the C# factory bakes in: Const(0) is a weak int and
# Const(0.0) a weak float, so NEP50 promotion matches the fused per-node typing exactly (e.g. relu uses
# Const(0), hence np.maximum(x, 0) — not 0.0 — so an int input stays int). The operators map: `&`/`|`/`^`
# are bitwise (BitwiseAnd/Or/Xor) and `!` is LogicalNot (verified on BOTH master and the exprs branch),
# so the boolean-logic family is logical_not(bitwise_...). Heaviside is intentionally ABSENT — on the
# exprs branch it is the Phase-4.2 first-class binary NODE (token "heaviside" in _EV_BINARY), not a
# combinator, and it propagates NaN where the master composition would return h0.
def _c_switch(*a):
    # switch(default, c0, v0, c1, v1, ...): the first true case wins, so case 0 is the OUTERMOST where.
    default, rest = a[0], a[1:]
    pairs = [(rest[i], rest[i + 1]) for i in range(0, len(rest), 2)]
    acc = default
    for c, v in reversed(pairs):
        acc = np.where(c, v, acc)
    return acc


def _c_mux(*a):
    # mux(index, v0, v1, ...): values[k] where index == k (weak-int k), else 0; index 0 outermost.
    index, vals = a[0], a[1:]
    acc = 0
    for k in range(len(vals) - 1, -1, -1):
        acc = np.where(np.equal(index, k), vals[k], acc)
    return acc


def _c_bucketize(*a):
    # bucketize(x, e0, e1, ...): count of edges met/exceeded = Σ INTEGER Where(x>=e, 1, 0) (a bool sum
    # would be logical OR under NEP50, not a count — so the acc seeds with the weak int 0).
    x, edges = a[0], a[1:]
    acc = 0
    for e in edges:
        acc = acc + np.where(np.greater_equal(x, e), 1, 0)
    return acc


_EV_COMBINATOR = {
    # selection & masking
    "if": lambda c, t, f: np.where(c, t, f),
    "ifnot": lambda c, t, f: np.where(c, f, t),
    "when": lambda c, t: np.where(c, t, 0),
    "unless": lambda c, t: np.where(c, 0, t),
    "switch": _c_switch,
    "mux": _c_mux,
    # clamp & saturation
    "clampmin": lambda x, lo: np.maximum(x, lo),
    "clampmax": lambda x, hi: np.minimum(x, hi),
    "saturate": lambda x: np.minimum(np.maximum(x, 0.0), 1.0),
    "nanto": lambda x, fb: np.where(np.isnan(x), fb, x),
    "coalesce": lambda a, b: np.where(np.isfinite(a), a, b),
    # activations
    "relu": lambda x: np.maximum(x, 0),
    "leakyrelu": lambda x, s: np.where(x > 0.0, x, x * s),
    "elu": lambda x, al: np.where(x > 0.0, x, al * (np.exp(x) - 1.0)),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "swish": lambda x: x * (1.0 / (1.0 + np.exp(-x))),
    "softplus": lambda x: np.maximum(x, 0.0) + np.log(1.0 + np.exp(-np.abs(x))),
    "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))),
    "hardsigmoid": lambda x: np.minimum(np.maximum(x / 6.0 + 0.5, 0.0), 1.0),
    "step": lambda x: np.where(x > 0.0, 1, 0),
    # boolean logic (int / bool only — & | ^ are bitwise, ! is LogicalNot → a bool result)
    "nand": lambda a, b: np.logical_not(np.bitwise_and(a, b)),
    "nor": lambda a, b: np.logical_not(np.bitwise_or(a, b)),
    "xnor": lambda a, b: np.logical_not(np.bitwise_xor(a, b)),
    "implies": lambda a, b: np.bitwise_or(np.logical_not(a), b),
    "majority3": lambda a, b, c: np.bitwise_or(np.bitwise_or(np.bitwise_and(a, b), np.bitwise_and(a, c)),
                                               np.bitwise_and(b, c)),
    # predicates
    "ispositive": lambda x: np.greater(x, 0.0),
    "isnegative": lambda x: np.less(x, 0.0),
    "isinteger": lambda x: np.equal(np.floor(x), x),
    "isclose": lambda a, b, rt, at: np.where(np.bitwise_and(np.isfinite(a), np.isfinite(b)),
                                             np.less_equal(np.abs(a - b), at + rt * np.abs(b)),
                                             np.equal(a, b)),
    "samesign": lambda a, b: np.equal(np.less(a, 0.0), np.less(b, 0.0)),
    "between": lambda x, lo, hi: np.bitwise_and(np.greater_equal(x, lo), np.less_equal(x, hi)),
    # directional / sign
    "cmp": lambda a, b: np.where(np.greater(a, b), 1, np.where(np.less(a, b), -1, 0)),
    # np.negative(d) — NOT the folded literal -d — mirrors the `-delta` NEGATE NODE in the C# tree,
    # which (like np.negative) promotes a WEAK scalar delta to strong float64, so steptoward over
    # float32 inputs with a weak-float delta resolves to float64 on BOTH sides (a Negate node is a
    # ufunc, not a source-level literal fold). For an array delta both spellings agree (no promotion).
    "steptoward": lambda x, tg, d: x + np.minimum(np.maximum(tg - x, np.negative(d)), d),
    "maxmagnitude": lambda a, b: np.where(np.greater_equal(np.abs(a), np.abs(b)), a, b),
    # multi-way decision & interpolation
    "bucketize": _c_bucketize,
    "median3": lambda a, b, c: np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c)),
    "threshold": lambda x, t, v: np.where(np.greater(x, t), x, v),
    "lerp": lambda a, b, t: a + (b - a) * t,
}

_EV_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*(?::[^,()]+)?|[(),])")


def _ev_tokens(expr):
    pos, out = 0, []
    while pos < len(expr):
        m = _EV_TOKEN.match(expr, pos)
        if not m:
            raise ValueError(f"bad expr token at {pos}: {expr!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


def _ev_literal(tok):
    kind, _, val = tok.partition(":")
    if kind == "li" or kind == "lu":
        return int(val)
    if kind == "lf":
        return float(val)
    if kind == "lb":
        return bool(int(val))
    if kind == "lc":
        re_, im_ = val.split(";")
        return complex(float(re_), float(im_))
    if kind == "lh":
        return np.float16(float(val))
    raise ValueError(f"unknown literal {tok!r}")


def _ev_eval(expr, operands):
    """Evaluate a prefix-grammar tree with NumPy, node by node (the unfused chain)."""
    toks = _ev_tokens(expr)
    pos = [0]

    def parse():
        tok = toks[pos[0]]
        pos[0] += 1
        if tok.startswith("in") and tok[2:].isdigit():
            return operands[int(tok[2:])]
        if ":" in tok:
            return _ev_literal(tok)
        assert toks[pos[0]] == "(", f"expected '(' after {tok} in {expr}"
        pos[0] += 1
        args = []
        while True:
            args.append(parse())
            sep = toks[pos[0]]
            pos[0] += 1
            if sep == ")":
                break
            assert sep == ",", f"expected ',' in {expr}"
        if tok == "where":
            return np.where(*args)
        if tok.startswith("cast_"):
            # cast_<dtype>(child) == child.astype(<dtype>) with NumPy's default casting='unsafe'.
            return np.asarray(args[0]).astype(np.dtype(tok[len("cast_"):]))
        if tok.startswith("round_"):
            # round_<d>(child) == np.round(child, d); d is spelled m<n> for the negative -n.
            s = tok[len("round_"):]
            d = -int(s[1:]) if s.startswith("m") else int(s)
            return np.round(np.asarray(args[0]), d)
        if tok in _EV_BINARY:
            return _EV_BINARY[tok](*args)
        if tok in _EV_UNARY:
            return _EV_UNARY[tok](*args)
        if tok in _EV_COMBINATOR:
            return _EV_COMBINATOR[tok](*args)
        raise ValueError(f"unknown node {tok!r} in {expr}")

    r = parse()
    assert pos[0] == len(toks), f"trailing tokens in {expr}"
    return r


def _ev_ops_in_expr(expr):
    """Node names used by a tree (for the generator's per-op dtype filters)."""
    return {t for t in _ev_tokens(expr) if t not in "()," and ":" not in t and not (t.startswith("in") and t[2:].isdigit())}


# NumPy raises on these cells with a message NumSharp reproduces verbatim — recorded as error
# cases. Every other raise is skipped (error-text parity for the rest is the errors_full tier's
# job, not this one's).
_EV_VERBATIM_ERRORS = (
    "numpy boolean subtract",
    "numpy boolean negative",
    "Integers to negative integer powers are not allowed.",
    "not supported for the input types, and the inputs could not be safely coerced",
)

_EV_FLOAT_DTYPES = {"float16", "float32", "float64"}
_EV_INT_DTYPES = {"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}
_EV_INTBOOL = _EV_INT_DTYPES | {"bool"}


def gen_evaluate():
    cases = []
    n = 0
    skipped = 0

    def emit(expr, operands_bv, layout, params=None, out=None, where=None, cid_tag=""):
        """operands_bv: list of (base, view). out: (base, view) or None. where: (base, view) bool
        mask or None (plan P4.5, out= only — masked writes leave masked-off out slots at prior)."""
        nonlocal n, skipped
        views = [v for (_, v) in operands_bv]
        params = dict(params or {})
        params["expr"] = expr
        ops_desc = [describe(b, v) for (b, v) in operands_bv]
        cid = f"evaluate/{layout}/{cid_tag}/{n}"
        try:
            r = _ev_eval(expr, views)
            red = params.get("reduce")
            if red:
                rk = red["kind"]
                if rk == "average":
                    # Weighted average reduces over TWO trees: the values tree in "expr" and the
                    # weights tree in red["weights"]. np.average(v, weights=w) IS the fused contract
                    # (Σ(v·w)/Σ(w)); same-shape operands, so no 1-D-along-axis convenience is needed.
                    w = _ev_eval(red["weights"], views)
                    r = np.average(r, weights=w, axis=red.get("axis"),
                                   keepdims=bool(red.get("keepdims", False)))
                elif rk in ("var", "std"):   # the ddof-carrying kinds
                    r = _EV_REDUCE[rk](r, red.get("axis"), bool(red.get("keepdims", False)), int(red.get("ddof", 0)))
                else:
                    r = _EV_REDUCE[rk](r, red.get("axis"), bool(red.get("keepdims", False)))
            r = np.asarray(r)
            # Phase 4.5 dtype= — the implicit root cast: np.evaluate(expr, dtype=X) computes the tree at its
            # natural NEP50 result type, then casts the RESULT to X (== np.evaluate(expr).astype(X)). Only the
            # elementwise blocks set it (a reduction rejects dtype= in NumSharp), so this never runs after a
            # reduce. The out= path is mutually exclusive with dtype=, so it likewise never combines.
            if params.get("dtype") is not None:
                r = np.asarray(r).astype(np.dtype(params["dtype"]))
            if r.dtype.name == "complex64":
                skipped += 1          # NumSharp has one complex width; skip the width-only cells
                return
            if out is not None:
                ob, ov = out
                if ov.shape != r.shape:
                    skipped += 1
                    return
                ops_desc.append(describe(ob, ov))
                params["out"] = True
                if where is not None:
                    # Plan P4.5 where=: the mask rides as the LAST operand (after out), and the write is
                    # masked so masked-off slots keep the PRIOR out contents (recorded by describe above).
                    # Skip a mask that does not broadcast to the result — NumPy would raise, and this tier
                    # gates values, not the broadcast error (that is unit-tested).
                    wb, wv = where
                    try:
                        np.broadcast_shapes(wv.shape, r.shape)
                    except ValueError:
                        skipped += 1
                        return
                    ops_desc.append(describe(wb, wv))
                    params["where"] = True
                    np.copyto(ov, r, where=wv, casting="same_kind")
                else:
                    # PRIOR out contents are what ops_desc recorded above; run the write now.
                    np.copyto(ov, r, casting="same_kind")
                cases.append(_case("evaluate", params, ops_desc,
                                   _tuple_expected([np.asarray(ov), ob.ravel()]), layout, "mixed", cid=cid))
            else:
                cases.append(_case("evaluate", params, ops_desc, _arr_expected(r), layout, "mixed", cid=cid))
            n += 1
        except Exception as e:
            if any(s in str(e) for s in _EV_VERBATIM_ERRORS):
                cases.append(_error_case("evaluate", params, ops_desc, e, layout, cid=cid))
                n += 1
            else:
                skipped += 1

    def ok_for(expr, *dts):
        """Per-op dtype filter: keep NumPy's no-loop / width-only cells out of the value tier."""
        ops = _ev_ops_in_expr(expr)
        toks = _ev_tokens(expr)
        anyfloat = any(d in _EV_FLOAT_DTYPES for d in dts)
        anycomplex = any(d == "complex128" for d in dts)
        allintbool = all(d in _EV_INTBOOL for d in dts)
        if ops & {"and", "or", "xor", "not"} and not allintbool:
            return "err"                              # NumPy no-loop TypeError (verbatim) — keep as error cell
        if anycomplex and ops & {"floor", "ceil", "round", "trunc", "mod", "floordiv", "atan2",
                                  "lt", "le", "gt", "ge", "min", "max", "fmax", "fmin",
                                  "cbrt", "deg2rad", "rad2deg"}:
            return False                              # complex: no such NumPy loop / ordering
            # (fmax/fmin: NumPy computes complex lexicographically, but the NaN-identity divergence is
            #  the same E5 class as min/max — excluded here and unit-test-pinned instead.)
        if any(t.startswith("lc:") for t in toks) and not (allintbool or "float64" in dts or anycomplex):
            return False                              # f16/f32 + complex literal -> complex64 (width only)
        if any(t.startswith("lu:") for t in toks) and not all(d in ("uint64", "float64") for d in dts):
            return False                              # out-of-range for every other adopter (OverflowError)
        if anycomplex and "abs" in ops and len(ops) > 1:
            return False                              # |z| is 1-2 ULP off CRT hypot; a composition can cancel that into any size
        if "pow" in ops and set(dts) == {"int64", "uint64"}:
            return False                              # W1-C: NumSharp keeps the integer power path for the u8/i8 pair [known bug]
        return True

    # ---- A. two-operand trees over the pairwise layouts --------------------------------------
    pair_dts = [
        ("float64", "float64"), ("float32", "float32"), ("float16", "float16"),
        ("int32", "int32"), ("int64", "int64"), ("uint8", "uint8"), ("int8", "int8"), ("uint64", "uint64"),
        ("bool", "bool"), ("complex128", "complex128"),
        ("int32", "float64"), ("float32", "float64"), ("int8", "uint8"),
        ("bool", "int32"), ("int64", "uint64"), ("complex128", "float64"),
    ]
    templates_a = [
        "add(in0,in1)", "sub(in0,in1)", "mul(in0,in1)", "div(in0,in1)",
        "mod(in0,in1)", "floordiv(in0,in1)", "pow(in0,in1)", "atan2(in0,in1)",
        "and(in0,in1)", "or(in0,in1)", "xor(in0,in1)",
        "min(in0,in1)", "max(in0,in1)",
        "eq(in0,in1)", "ne(in0,in1)", "lt(in0,in1)", "le(in0,in1)", "gt(in0,in1)", "ge(in0,in1)",
        # fused compositions — what np.evaluate exists for
        "add(mul(in0,in1),in0)",
        "div(sub(in0,in1),add(in0,in1))",
        "sqrt(add(mul(in0,in0),mul(in1,in1)))",
        "where(gt(in0,in1),in0,in1)",
        "mul(in0,gt(in1,li:0))",
        "and(gt(in0,li:0),lt(in1,li:3))",
        "lnot(gt(in0,in1))",
        "where(gt(in0,li:0),in0,mul(in0,lf:0.01))",
        "max(min(in0,in1),lf:0.5)",
        "where(in0,in1,li:0)",
        "add(abs(in0),neg(in1))",
        "eq(in0,li:1)",
        "add(in0,lb:1)",
        "mul(in0,lf:2.5)",
        "sub(in0,li:2)",
        "add(in0,lc:0;1)",
        "add(in0,lh:2.0)",
        "add(in0,lu:18446744073709551615)",
        "gt(in0,li:-1)",
        "lt(in0,li:300)",
    ]
    for ln, fn in PAIR_LAYOUTS.items():
        for (sa, sb) in pair_dts:
            ba, va, bb, vb = fn(np.dtype(sa), np.dtype(sb))
            for expr in templates_a:
                verdict = ok_for(expr, sa, sb)
                if verdict is False:
                    continue
                emit(expr, [(ba, va), (bb, vb)], ln, cid_tag=f"{sa},{sb}/{expr}")

    # ---- A2. Phase 4 binary node coverage (the engine binary ufuncs that gained an NDExpr node) ----
    # Same pair layouts × dtypes as A; ok_for + the emit try/except drop the no-loop / width-only /
    # E5-complex cells. copysign/nextafter/heaviside/fmod/fmax/fmin/gcd/lcm/shifts are BIT-EXACT;
    # logaddexp/logaddexp2 (≤2 ULP) and hypot@f64 (≤1 ULP) ride the EvaluateLibmOps ~ULP excuse.
    templates_a2 = [
        "fmax(in0,in1)", "fmin(in0,in1)", "fmod(in0,in1)",
        "copysign(in0,in1)", "nextafter(in0,in1)",
        "logaddexp(in0,in1)", "logaddexp2(in0,in1)",
        "hypot(in0,in1)", "heaviside(in0,in1)",
        "gcd(in0,in1)", "lcm(in0,in1)",
        "lshift(in0,in1)", "rshift(in0,in1)",
        # the new nodes as SUB-trees (prove they compose in the fused kernel, not only as a root)
        "add(fmax(in0,in1),in0)",
        "mul(hypot(in0,in1),lf:2.0)",
    ]
    for ln, fn in PAIR_LAYOUTS.items():
        for (sa, sb) in pair_dts:
            ba, va, bb, vb = fn(np.dtype(sa), np.dtype(sb))
            for expr in templates_a2:
                if ok_for(expr, sa, sb) is False:
                    continue
                emit(expr, [(ba, va), (bb, vb)], ln, cid_tag=f"{sa},{sb}/{expr}")

    # ---- A3. Phase 4.3 logical nodes (LogicalNode) — bool result via a per-operand nonzero test ----
    # Accept every dtype (complex included), no no-loop cell; the input pools include ±0 / NaN / inf
    # rows (via PAIR_LAYOUTS' fills) so the truthiness edges are exercised. The sub-trees prove they
    # compose (feeding a where mask and OR-of-comparisons) inside the fused kernel.
    templates_a3 = [
        "land(in0,in1)", "lor(in0,in1)", "lxor(in0,in1)",
        "where(land(gt(in0,li:0),lt(in1,li:5)),in0,in1)",
        "lor(land(in0,in1),lxor(in0,in1))",
    ]
    for ln, fn in PAIR_LAYOUTS.items():
        for (sa, sb) in pair_dts:
            ba, va, bb, vb = fn(np.dtype(sa), np.dtype(sb))
            for expr in templates_a3:
                if ok_for(expr, sa, sb) is False:
                    continue
                emit(expr, [(ba, va), (bb, vb)], ln, cid_tag=f"{sa},{sb}/{expr}")

    # ---- A4. Phase 5.2 mixed-width SIMD ("lane groups") — the ratio-2 dtype pairs -------------
    # The pairs the mixed-width vector plan admits beyond what A already sweeps: every half-lane
    # widen direction (int → wider-int sign/zero-extend, int → float, float → double, and the
    # uint32→float64 sign-bias edge, which has no AVX2 instruction) plus the both-widen NEP50
    # promotions (i4+u4→i8, i2+u2→i4, i1+u1→i2). The composites pin the per-node-dtype contract:
    # mul(in0,li:2) WRAPS at in0's own dtype BEFORE the edge widens into the sum — NumPy's unfused
    # sequence does the same, and computing the product at the lane dtype instead would diverge.
    pair_dts_a4 = [
        ("uint32", "float64"), ("int32", "float32"), ("uint32", "float32"),
        ("int16", "float32"), ("uint16", "float32"),
        ("int32", "int64"), ("uint32", "int64"), ("uint32", "uint64"), ("int32", "uint32"),
        ("int16", "int32"), ("uint16", "uint32"), ("int16", "uint16"),
        ("int8", "int16"), ("uint8", "uint16"),
    ]
    templates_a4 = [
        "add(in0,in1)", "sub(in0,in1)", "mul(in0,in1)", "div(in0,in1)",
        "add(mul(in0,li:2),in1)",           # the acceptance tree: narrow wrap → widen → add
        "add(mul(in0,in1),in0)",            # one leaf consumed at two dtypes (own + widened)
        "mul(add(in0,in1),sub(in1,in0))",
        "or(and(in0,in1),in0)",             # bitwise through widen edges (ok_for drops float cells)
    ]
    for ln, fn in PAIR_LAYOUTS.items():
        for (sa, sb) in pair_dts_a4:
            ba, va, bb, vb = fn(np.dtype(sa), np.dtype(sb))
            for expr in templates_a4:
                if ok_for(expr, sa, sb) is False:
                    continue
                emit(expr, [(ba, va), (bb, vb)], ln, cid_tag=f"{sa},{sb}/{expr}")

    # ---- B. single-operand trees over the single layouts -------------------------------------
    unary_layouts = ["c_contiguous_1d", "c_contiguous_2d", "f_contiguous_2d", "strided_step2_1d",
                     "negstride_1d", "simple_slice_offset_1d"]
    unary_exprs = [f"{op}(in0)" for op in _EV_UNARY]
    for ln in unary_layouts:
        for dt in ALL_DTYPES:
            b, v = LAYOUTS[ln](np.dtype(dt))
            for expr in unary_exprs:
                verdict = ok_for(expr, dt)
                if verdict is False:
                    continue
                emit(expr, [(b, v)], ln, cid_tag=f"{dt}/{expr}")

    composite_exprs = [
        "mul(sqrt(abs(in0)),lf:2.0)",
        "exp(neg(square(in0)))",
        "where(isnan(in0),lf:0.0,in0)",
        "add(mul(in0,in0),in0)",
        "div(sub(in0,lf:1.5),add(in0,lf:1.5))",
        "gt(in0,lf:0.5)",
        "and(gt(in0,lf:0.2),lt(in0,lf:0.8))",
        "where(gt(in0,li:2),in0,neg(in0))",
        "max(in0,li:0)",
        "mul(in0,lb:1)",
        "not(in0)",
        "where(in0,li:1,li:0)",
        # Phase 4 unary nodes as SUB-trees (prove they compose in the fused kernel, not only as a
        # root): the float-tier fabs/spacing promote int→float then feed arithmetic; a complex child
        # raises the recorded "not supported" error cell (fabs/spacing have no complex loop).
        "mul(fabs(in0),lf:2.0)",
        "add(spacing(in0),in0)",
        # rint as a SUB-tree: the float-tier promote feeds arithmetic. rint(in0*1.5) rounds a
        # non-integer product half-to-even, then +1 — a genuine fused rint (not the identity an
        # integer input would give the root sweep).
        "add(rint(mul(in0,lf:1.5)),lf:1.0)",
        # Phase 4.1b component extractors as SUB-trees. add(real,imag) recombines the two lanes (over a
        # complex child both are float64; over a real child it is identity + zeros = the value, dtype
        # preserved). mul(angle,2.0) is the radians→(scaled) composition — angle feeds arithmetic, the
        # deg-style scale a caller writes since the fused primitive is radians-only.
        "add(real(in0),imag(in0))",
        "mul(angle(in0),lf:2.0)",
    ]
    # Every layout the catalog has, at every third position (the unary set above already walks
    # the contiguity / stride / offset axes; this pass is about the tree shapes).
    composite_layouts = list(LAYOUTS.keys())[::2]
    for ln in composite_layouts:
        for dt in ALL_DTYPES:
            b, v = LAYOUTS[ln](np.dtype(dt))
            for expr in composite_exprs:
                verdict = ok_for(expr, dt)
                if verdict is False:
                    continue
                emit(expr, [(b, v)], ln, cid_tag=f"{dt}/{expr}")

    # ---- B3. Cast(child, target) — the astype node (Phase 4.1b), src×target×layouts -----------
    # A controlled NON-NEGATIVE pool (small integers + positive fractionals) that every target dtype
    # represents WITHOUT overflow / negative-to-unsigned / NaN — so every cell is portable, bit-exact
    # NumPy parity with NO host-dependent conversion (the C-undefined float→int edges — NaN/±inf/
    # out-of-range/negative→unsigned — are covered by NDEvaluateTests.P41bCast_* against the engine's
    # own host-pinned Converts.* table instead). float→int TRUNCATION is still exercised (2.5→2, 4.5→4).
    cast_pool = np.array([0.0, 1.0, 2.0, 3.0, 2.5, 4.5, 6.0, 7.0])
    cast_src = ["bool", "int8", "uint8", "int16", "uint32", "int64", "uint64",
                "float16", "float32", "float64", "complex128"]
    cast_dst = ALL_DTYPES                       # 13 targets (bool..complex128; char/decimal via unit tests)
    cast_layouts = ["c_contiguous_1d", "c_contiguous_2d", "f_contiguous_2d",
                    "negstride_1d", "strided_step2_1d"]
    for ln in cast_layouts:
        for s in cast_src:
            b, v = LAYOUTS[ln](np.dtype(s))
            # Overwrite the layout's random data with the safe pool (tiled to b's size), in the source
            # dtype; v is a VIEW of b, so it reads the same values through its own strides.
            flat = b.reshape(-1)
            src_vals = np.resize(cast_pool, flat.size)
            flat[:] = (src_vals + 0j).astype(b.dtype) if np.dtype(s).kind == "c" else src_vals.astype(b.dtype)
            for d in cast_dst:
                emit(f"cast_{d}(in0)", [(b, v)], ln, cid_tag=f"{s}->{d}")
    # Cast as a SUB-tree: cast(a+b, int32) truncates a fused sum; add(cast(a, f4), 1.0) casts then
    # continues arithmetic — proving Cast composes (not only as a root), still over the safe pool.
    for ln in ["pp_contig_contig", "pp_contig_fortran"]:
        ba, va, bb, vb = PAIR_LAYOUTS[ln](np.dtype("float64"), np.dtype("float64"))
        ba.reshape(-1)[:] = np.resize(cast_pool, ba.size)
        bb.reshape(-1)[:] = np.resize(np.array([0.4, 0.6, 0.5, 0.1]), bb.size)
        emit("cast_int32(add(in0,in1))", [(ba, va), (bb, vb)], ln, cid_tag="f64,f64/cast_i4")
        emit("add(cast_float32(in0),lf:1.0)", [(ba, va), (bb, vb)], ln, cid_tag="f64,f64/cast_f4")

    # ---- B4. Round(child, decimals != 0) — np.round(x, d), the RoundNode (Phase 4.1b) ----------
    # A controlled MODERATE pool (fractional floats for the multiply→rint→divide, small integers whose
    # negative-decimals float64 round-trip stays well within range) so every cell is portable bit-exact
    # NumPy parity: NO C-undefined float→int cast (the integer round-trip never leaves the int-representable
    # range) and NO overflow to ±inf. bool is EXCLUDED (decimals != 0 raises the multiply/divide cast error,
    # NDEvaluateTests.P41bRound_* pins that); float/int/complex/char all preserve dtype. NB: the ENGINE's
    # np.around is BROKEN for negative decimals (Math.Round rejects them) and does not round complex — the
    # FUSED RoundNode is the CORRECT port, so this tier is validated against NumPy directly (the oracle),
    # never the engine.
    round_pool = np.array([1.2345, 2.5, -2.675, 3.15, 15.0, -25.0, 12.0, -100.0, 0.0, 60.0])
    round_src = ["int8", "uint8", "int16", "uint16", "int32", "int64", "uint64",
                 "float16", "float32", "float64", "complex128"]
    round_layouts = ["c_contiguous_1d", "c_contiguous_2d", "f_contiguous_2d",
                     "negstride_1d", "strided_step2_1d"]
    round_decimals = ["2", "1", "m1", "m2", "3", "m3"]     # m<n> == -n
    for ln in round_layouts:
        for s in round_src:
            b, v = LAYOUTS[ln](np.dtype(s))
            flat = b.reshape(-1)
            vals = np.resize(round_pool, flat.size)
            if np.dtype(s).kind == "u":               # unsigned: keep the pool non-negative
                vals = np.abs(vals)
            flat[:] = (vals + 1j * np.resize(np.array([0.5, -1.25, 2.55, -3.15]), flat.size)).astype(b.dtype) \
                if np.dtype(s).kind == "c" else vals.astype(b.dtype)
            for d in round_decimals:
                emit(f"round_{d}(in0)", [(b, v)], ln, cid_tag=f"{s}/d={d}")
    # Round as a SUB-tree: round(a+b, 2) rounds a fused sum; add(round(a, 1), 1.0) rounds then continues.
    for ln in ["pp_contig_contig", "pp_contig_fortran"]:
        ba, va, bb, vb = PAIR_LAYOUTS[ln](np.dtype("float64"), np.dtype("float64"))
        ba.reshape(-1)[:] = np.resize(round_pool, ba.size)
        bb.reshape(-1)[:] = np.resize(np.array([0.111, 0.222, 0.333, 0.444]), bb.size)
        emit("round_2(add(in0,in1))", [(ba, va), (bb, vb)], ln, cid_tag="f64,f64/round2")
        emit("add(round_1(in0),lf:1.0)", [(ba, va), (bb, vb)], ln, cid_tag="f64,f64/round1")

    # ---- B5. dtype= keyword — the implicit root cast (Phase 4.5) --------------------------------
    # np.evaluate(expr, dtype=X) computes the FUSED tree at its natural NEP50 result type and casts the
    # RESULT to X in ONE pass — the out= buffered-cast machinery, DISTINCT from the Cast NODE (B3) whose
    # per-element EmitConvertTo runs a different loop. Gated independently over the same safe NON-NEGATIVE
    # pool so every cell is portable bit-exact NumPy parity (the C-undefined float→int edges — NaN/±inf/
    # out-of-range — are unit-tested against the engine's host-pinned astype, and a complex→real DROP is
    # left to the Cast node, so B5 keeps a REAL result tree and sweeps real + complex TARGETS only). The
    # tree is a genuine fused elementwise (mul by a weak-float literal), so int sources promote to float
    # before the cast, exercising the fused-then-cast path, not a bare astype.
    dt_pool = np.array([0.0, 1.0, 2.0, 3.0, 2.5, 4.5, 6.0, 7.0])
    dt_src = ["float64", "float32", "int32", "int64", "uint8"]
    dt_dst = ["bool", "int8", "uint8", "int16", "int32", "int64", "uint64",
              "float16", "float32", "float64", "complex128"]
    dt_layouts = ["c_contiguous_1d", "c_contiguous_2d", "f_contiguous_2d",
                  "negstride_1d", "strided_step2_1d"]
    for ln in dt_layouts:
        for s in dt_src:
            b, v = LAYOUTS[ln](np.dtype(s))
            b.reshape(-1)[:] = np.resize(dt_pool, b.size).astype(b.dtype)
            for d in dt_dst:
                emit("mul(in0,lf:2.0)", [(b, v)], ln, params={"dtype": d}, cid_tag=f"dtype:{s}->{d}")
    # dtype= over a genuine two-operand fused sum, and over a where-tree — proving the keyword casts the
    # WHOLE resolved tree, not just a single node.
    for ln in ["pp_contig_contig", "pp_contig_fortran"]:
        ba, va, bb, vb = PAIR_LAYOUTS[ln](np.dtype("float64"), np.dtype("float64"))
        ba.reshape(-1)[:] = np.resize(dt_pool, ba.size)
        bb.reshape(-1)[:] = np.resize(np.array([0.4, 0.6, 0.5, 0.1]), bb.size)
        for d in ("int32", "float32", "complex128"):
            emit("add(in0,in1)", [(ba, va), (bb, vb)], ln, params={"dtype": d}, cid_tag=f"f64,f64/dtype:{d}")

    # ---- E. C6 combinators (NDExpr.Combinators.cs) — the macro / decision vocabulary ----------
    # Each combinator is a PURE COMPOSITION of primitive nodes, so this tier proves the FACTORY builds
    # the right tree by bit-comparing np.evaluate(NDExpr.X(...)) against NumPy's SAME composition
    # (_EV_COMBINATOR). Grouped by dtype policy: the transcendental activations stay float32/float64
    # (the exp/log/tanh ports are bit-exact there on the host-pinned tier; f16 has a 17-value exp edge
    # and is held out); the bitwise boolean-logic family is int/bool ONLY (a float operand is a NumPy
    # no-loop); everything else takes a real int+float mix. COMPLEX is excluded across the board — the
    # Max/Min/comparison/floor nodes these compose over have no complex loop.
    comb_real = ["float64", "float32", "int32", "int8"]
    comb_pred = ["float64", "float32", "int32"]
    comb_trans = ["float64", "float32"]
    comb_bool = ["bool", "int8", "uint8", "int32"]

    # E1. single-array roots (one operand in0, literals baked in) over the single layouts.
    comb_unary = {
        "saturate(in0)": comb_real, "relu(in0)": comb_real, "step(in0)": comb_real,
        "hardsigmoid(in0)": comb_real, "leakyrelu(in0,lf:0.01)": comb_real,
        "elu(in0,lf:1.0)": comb_trans, "sigmoid(in0)": comb_trans, "swish(in0)": comb_trans,
        "softplus(in0)": comb_trans, "gelu(in0)": comb_trans,
        "ispositive(in0)": comb_pred, "isnegative(in0)": comb_pred, "isinteger(in0)": comb_pred,
        "between(in0,lf:0.2,lf:0.8)": comb_pred,
        "threshold(in0,lf:0.0,lf:-1.0)": comb_real,
        "bucketize(in0,lf:0.0,lf:1.0,lf:5.0)": comb_real,
        "mux(in0,lf:10.0,lf:20.0,lf:30.0)": comb_real,
    }
    comb_unary_layouts = ["c_contiguous_1d", "c_contiguous_2d", "f_contiguous_2d",
                          "negstride_1d", "strided_step2_1d", "scalar_0d"]
    for ln in comb_unary_layouts:
        for expr, dts in comb_unary.items():
            for dt in dts:
                b, v = LAYOUTS[ln](np.dtype(dt))
                emit(expr, [(b, v)], ln, cid_tag=f"{dt}/{expr}")

    # E2. two-operand roots (in0, in1) — the ternary combinators reuse the two operands via subtrees /
    # literals so the pair layouts suffice. The boolean-logic family runs the int/bool pool.
    comb_binary = {
        "clampmin(in0,in1)": comb_real, "clampmax(in0,in1)": comb_real,
        "nanto(in0,in1)": comb_real, "coalesce(in0,in1)": comb_real,
        "maxmagnitude(in0,in1)": comb_real, "cmp(in0,in1)": comb_real,
        "samesign(in0,in1)": comb_pred,
        "when(in0,in1)": comb_real, "unless(in0,in1)": comb_real,
        "steptoward(in0,in1,lf:1.0)": comb_real,
        "if(gt(in0,in1),in0,in1)": comb_real, "ifnot(gt(in0,in1),in0,in1)": comb_real,
        "median3(in0,in1,mul(in0,in1))": comb_real,
        "lerp(in0,in1,lf:0.5)": comb_real,
        "isclose(in0,in1,lf:0.001,lf:0.0)": comb_pred,
        "nand(in0,in1)": comb_bool, "nor(in0,in1)": comb_bool, "xnor(in0,in1)": comb_bool,
        "implies(in0,in1)": comb_bool,
        "majority3(gt(in0,li:0),lt(in1,li:0),gt(in0,in1))": comb_real,   # 3 bool subtrees from 2 operands
        "switch(li:0,gt(in0,in1),in0,lt(in0,in1),in1)": comb_real,       # default + 2 (cond,value) cases
    }
    comb_pair_layouts = ["pp_contig_contig", "pp_contig_fortran", "pp_strided_strided",
                         "pp_negstride_both", "pp_scalar_right", "pp_broadcast_row"]
    for ln in comb_pair_layouts:
        for expr, dts in comb_binary.items():
            for dt in dts:
                ba, va, bb, vb = PAIR_LAYOUTS[ln](np.dtype(dt), np.dtype(dt))
                emit(expr, [(ba, va), (bb, vb)], ln, cid_tag=f"{dt},{dt}/{expr}")

    # E3. COMBINATIONS — combinators nested inside primitives AND inside each other, proving they fold
    # into the surrounding fused pass (the "doing combinations as well" coverage). Run over float32/64
    # (a safe superset for the transcendental members); the predicate→boolean-logic tree stays bool by
    # construction regardless of the float inputs.
    comb_combos = [
        "add(relu(in0),in1)",
        "mul(sigmoid(in0),lf:2.0)",
        "max(relu(in0),clampmin(in1,lf:0.0))",
        "step(sub(saturate(in0),lf:0.5))",                # the C# Combinators_ComposeWithEachOther shape
        "where(ispositive(in0),in0,in1)",
        "lerp(relu(in0),sigmoid(in1),lf:0.5)",            # combinator args feeding a combinator
        "add(cmp(in0,in1),clampmin(in0,in1))",
        "nand(ispositive(in0),isnegative(in1))",          # predicate → boolean-logic (bool operands)
        "sub(gelu(in0),softplus(in1))",
        "mul(hardsigmoid(in0),relu(in1))",
    ]
    for ln in ["pp_contig_contig", "pp_contig_fortran", "pp_strided_strided"]:
        for dt in ("float64", "float32"):
            ba, va, bb, vb = PAIR_LAYOUTS[ln](np.dtype(dt), np.dtype(dt))
            for expr in comb_combos:
                emit(expr, [(ba, va), (bb, vb)], ln, cid_tag=f"{dt},{dt}/{expr}")

    # ---- C. root reductions over fused trees (flat + axis + keepdims) -------------------------
    reduce_layouts = ["c_contiguous_1d", "c_contiguous_2d", "c_contiguous_3d", "f_contiguous_2d",
                      "transposed_3d", "strided_step2_1d", "negstride_1d", "simple_slice_offset_1d"]
    reduce_exprs = ["in0", "mul(in0,in0)", "gt(in0,li:2)", "where(gt(in0,li:2),in0,li:0)"]
    reduce_dts = ["bool", "int8", "int32", "uint64", "float16", "float32", "float64", "complex128"]
    for ln in reduce_layouts:
        for dt in reduce_dts:
            b, v = LAYOUTS[ln](np.dtype(dt))
            # min/max tie-breaking on a ±0 pair follows the fold order (NumPy's SIMD reduction vs the
            # fused 4-accumulator fold) and is not contractual: fold the signed zeros into +0 so the
            # VALUE is the only thing compared.
            if np.dtype(dt).kind in "fc":
                b[b == 0] = 0
            # A BENIGN twin for the float / complex Sum / Prod / Mean cells: until Phase 2 replaces the
            # fused 4-accumulator fold with NumPy's pairwise schedule at the result dtype, the order of
            # summation differs, and over the edge pool (2^31 - 2^31 + 0.5, inf - inf, ...) an order
            # difference is a VALUE difference of any size. Values in [0.75, 1.25] keep every partial
            # sum / product finite and normal in float16 and the drift within the ≤16-ULP excuse
            # (MisalignedRegistry E1). Phase 2 deletes both: the excuse and this twin.
            benign_b = None
            if np.dtype(dt).kind in "fc":
                benign_b = b.copy()
                flat = benign_b.reshape(-1)
                idx = np.arange(flat.size)
                mag = 0.75 + 0.5 * ((idx * 7) % 11) / 10.0
                if np.dtype(dt).kind == "c":
                    flat[:] = (mag + 1j * (0.75 + 0.5 * ((idx * 3) % 11) / 10.0)).astype(flat.dtype)
                else:
                    flat[:] = mag.astype(flat.dtype)
            for expr in reduce_exprs:
                if ok_for(expr, dt) is False:
                    continue
                if dt == "complex128" and _ev_ops_in_expr(expr) & {"gt"}:
                    continue
                # The five base kinds + the order-independent presence trio (any/all/count_nonzero —
                # bool / int64 result, EXACT at every dtype so no benign twin). nansum/nanprod are NOT
                # swept here: they need a NaN-carrying pool (block C4) and are folded/E1-excused for
                # float16 + complex prod, which this generic sweep would trip.
                for kind in ("sum", "prod", "min", "max", "mean", "any", "all", "count_nonzero"):
                    # Rebuild (base, view) with the SAME layout recipe over the pool this kind uses:
                    # the layout builders always hand back a fresh C-contiguous base the view aliases.
                    nb, nv = LAYOUTS[ln](np.dtype(dt))
                    nb[...] = benign_b if (benign_b is not None and kind in ("sum", "prod", "mean")) else b
                    # flat, every axis, keepdims on the first axis, AND flat+keepdims (plan P2 M5 —
                    # np.<reduce>(a, axis=None, keepdims=True) → shape (1,)*ndim, one element kept
                    # broadcast-friendly).
                    combos = [(None, False)] + [(ax, False) for ax in range(nv.ndim)]
                    if nv.ndim > 0:
                        combos.append((0, True))
                        combos.append((None, True))
                    for ax, kd in combos:
                        emit(expr, [(nb, nv)], ln, params={"reduce": {"kind": kind, "axis": ax, "keepdims": kd}},
                             cid_tag=f"{dt}/{kind}[{ax},{int(kd)}]/{expr}")

    # ---- C2. large-N FLAT reductions at the pairwise-schedule boundaries (plan P2.1 / M1) ------
    # Block C's pools are <=24 elements of LOW dynamic range, where NumPy's pairwise add.reduce and a
    # naive / 4-accumulator fold round IDENTICALLY — so they cannot tell a NumPy-exact reduction from a
    # drifting one (this is why the benign twin above kept the old fold "within 16 ULP" yet nothing red).
    # These cells cross PW_BLOCKSIZE (128) and the 8-accumulator leaf with WIDE-magnitude finite pools,
    # so the summation ORDER changes the rounded bits: the flat float32/float64/complex Sum · Mean and the
    # flat float Prod must reproduce NumPy bit-for-bit (M1 materializes the child and reduces with the same
    # pairwise / sequential schedule np.sum · np.prod use). Half, complex Prod, and every AXIS reduction
    # stay on the folded path (still E1-excused) and are deliberately absent here.
    def _wide_sum_pool(N):
        # Wide but finite magnitudes with sign changes → order-sensitive rounding (never inf / nan), so
        # pairwise (fix) and the 4-accumulator fold (old) round to DIFFERENT bits at every N below.
        base = np.array([1e7, 1.0, -1e7, 3.5, 1e-3, -2.0, 1e5, 0.25, -1e5, 7.0, 1e-2, -4.5, 9e6, 0.75, -9e6],
                        dtype=np.float64)
        return np.tile(base, (N + len(base) - 1) // len(base))[:N]

    def _near_one_prod_pool(N):
        # Products near 1 so the running product stays finite and normal at N = 1000, while the low bits
        # still depend on multiply ORDER (sequential vs a lane-interleaved fold).
        base = np.array([1.1, 0.9, 1.05, 0.95, 2.0, 0.5, 1.25, 0.8, 1.0, 0.75, 4.0 / 3.0, 3.0 / 4.0],
                        dtype=np.float64)
        return np.tile(base, (N + len(base) - 1) // len(base))[:N]

    for N in (7, 8, 127, 128, 129, 257, 1000):
        for dt in ("float32", "float64", "complex128"):
            npdt = np.dtype(dt)
            if npdt.kind == "c":
                sp = (_wide_sum_pool(N) + 1j * np.roll(_wide_sum_pool(N), 3)).astype(npdt)
            else:
                sp = _wide_sum_pool(N).astype(npdt)
            for kind in ("sum", "mean"):
                sb = np.ascontiguousarray(sp)
                emit("in0", [(sb, sb)], "c_contiguous_1d",
                     params={"reduce": {"kind": kind, "axis": None, "keepdims": False}},
                     cid_tag=f"bigN/{dt}/{kind}/N={N}")
            if npdt.kind == "f":  # complex Prod stays folded (npy_cmul FMA gap #12) — not diverted
                pb = np.ascontiguousarray(_near_one_prod_pool(N).astype(npdt))
                emit("in0", [(pb, pb)], "c_contiguous_1d",
                     params={"reduce": {"kind": "prod", "axis": None, "keepdims": False}},
                     cid_tag=f"bigN/{dt}/prod/N={N}")

    # ---- C3. large-N AXIS reductions at the pairwise-schedule boundaries (plan P2 M2) ---------
    # Block C's AXIS cells use <=24-element benign pools where pairwise and the 4-accumulator fold
    # round IDENTICALLY, so they cannot tell a NumPy-exact axis reduction from a drifting one. These
    # 2-D / 3-D cells cross PW_BLOCKSIZE (128) ALONG the reduced axis with WIDE-magnitude pools, so
    # the summation ORDER changes the rounded bits and the axis float32/float64/complex Sum · Mean and
    # the axis float Prod must reproduce NumPy bit-for-bit (M2 materializes the child and reduces the
    # C-contiguous result with np.add.reduce's pairwise / np.multiply.reduce's sequential schedule).
    # A reduced axis that is INNERMOST (contiguous) exercises the PINNED pairwise route; an OUTER one
    # the SLAB sequential route; the 3-D shape drives the inner-slab (inner > 1) sequential prod.
    # Half, complex Prod, and the all-F-contiguous corner stay folded (still E1-excused) and are
    # deliberately absent. All inputs are C-contiguous (the M2 divert's gate).
    axis_shapes = [
        (3, 200),        # innermost axis 1 (len 200 > 128) → PINNED pairwise; outer axis 0 (len 3)
        (200, 3),        # outer axis 0 (len 200 > 128) → SLAB sequential; innermost axis 1 (len 3)
        (2, 3, 60),      # 3-D: axis 2 innermost (len 60), axes 0/1 outer/mid drive inner>1 slab
    ]
    for shp in axis_shapes:
        N = int(np.prod(shp))
        lay = f"c_contiguous_{len(shp)}d"
        for dt in ("float32", "float64", "complex128"):
            npdt = np.dtype(dt)
            if npdt.kind == "c":
                sflat = (_wide_sum_pool(N) + 1j * np.roll(_wide_sum_pool(N), 3)).astype(npdt)
            else:
                sflat = _wide_sum_pool(N).astype(npdt)
            s2d = np.ascontiguousarray(sflat.reshape(shp))
            for ax in range(len(shp)):
                for kind in ("sum", "mean"):
                    sb = np.ascontiguousarray(s2d)
                    emit("in0", [(sb, sb)], lay,
                         params={"reduce": {"kind": kind, "axis": ax, "keepdims": False}},
                         cid_tag=f"bigNaxis/{dt}/{kind}[{ax}]/{'x'.join(map(str, shp))}")
            if npdt.kind == "f":  # complex Prod stays folded (npy_cmul FMA gap #12) — not diverted
                p2d = np.ascontiguousarray(_near_one_prod_pool(N).astype(npdt).reshape(shp))
                for ax in range(len(shp)):
                    pb = np.ascontiguousarray(p2d)
                    emit("in0", [(pb, pb)], lay,
                         params={"reduce": {"kind": "prod", "axis": ax, "keepdims": False}},
                         cid_tag=f"bigNaxis/{dt}/prod[{ax}]/{'x'.join(map(str, shp))}")

    # ---- C4. M4 reductions: NaN-aware sum/prod + the presence trio on a mixed pool (plan P2 M4) --
    # any/all/count_nonzero are order-independent (bool / int64 result) and already swept broadly in
    # block C; this block pins them — AND the NaN-aware sum/prod — on a pool that actually carries the
    # discriminating values: zeros (so any/all/count split), and NaN (truthy → any is True; skipped by
    # nansum/nanprod). nansum/nanprod are bit-exact only where the underlying Sum/Prod diverts
    # (float32/float64 + complex SUM via M1/M2); float16 (no f16 pairwise) and complex PROD (npy_cmul
    # gap #12) stay folded/E1-excused and are deliberately ABSENT.
    def _mixed_nan_pool(N, npdt):
        base = np.array([0, 1, -2, 0, 3, 0, -4, 5], dtype=np.float64)
        a = np.tile(base, (N + 7) // 8)[:N].astype(npdt)
        if npdt.kind in "fc":
            a[1::5] = np.nan          # scatter NaN through the pool (truthy; nan-skipped by nansum/nanprod)
        return a

    m4_layouts = ["c_contiguous_1d", "c_contiguous_2d", "f_contiguous_2d", "negstride_1d"]
    m4_dt_kinds = [
        ("bool",       ("any", "all", "count_nonzero")),
        ("int32",      ("any", "all", "count_nonzero")),
        ("uint64",     ("any", "all", "count_nonzero")),
        ("float32",    ("any", "all", "count_nonzero", "nansum", "nanprod")),
        ("float64",    ("any", "all", "count_nonzero", "nansum", "nanprod")),
        ("complex128", ("any", "all", "count_nonzero", "nansum")),  # complex prod folded (#12)
    ]
    for ln in m4_layouts:
        for dt, kinds in m4_dt_kinds:
            npdt = np.dtype(dt)
            for kind in kinds:
                nb, nv = LAYOUTS[ln](npdt)
                pool = _mixed_nan_pool(nb.size, npdt)
                if npdt.kind == "c":
                    pool = pool + 1j * np.roll(pool, 2)   # NaN in either component ⇒ isnan True
                nb.reshape(-1)[:] = pool
                combos = [(None, False)] + [(ax, False) for ax in range(nv.ndim)]
                if nv.ndim > 0:
                    combos.append((0, True))
                    combos.append((None, True))              # flat + keepdims (plan P2 M5)
                for ax, kd in combos:
                    emit("in0", [(nb, nv)], ln,
                         params={"reduce": {"kind": kind, "axis": ax, "keepdims": kd}},
                         cid_tag=f"m4/{dt}/{kind}[{ax},{int(kd)}]")

    # ---- C4b. WIDE-magnitude NaN pools: prove NanSum/NanProd take the M1/M2 divert -------------
    # The mixed pool above is LOW dynamic range, where pairwise and the 4-accumulator fold round
    # identically — so it pins the nan-SKIP but NOT that NanSum/NanProd actually reduce with NumPy's
    # pairwise/sequential schedule (a regression that silently took the fold path would still pass).
    # These cross PW_BLOCKSIZE with the same wide / near-one pools blocks C2/C3 use, then scatter NaN:
    # they match NumPy ONLY if the nan-replaced child is reduced pairwise (Sum) / sequentially (Prod).
    def _scatter_nan(a):
        a = a.copy()
        a[3::17] = np.nan
        return a

    for N in (129, 1000):
        for dt in ("float32", "float64", "complex128"):
            npdt = np.dtype(dt)
            wp = _scatter_nan(_wide_sum_pool(N))
            arr = (wp + 1j * np.roll(wp, 5)).astype(npdt) if npdt.kind == "c" else wp.astype(npdt)
            sb = np.ascontiguousarray(arr)
            emit("in0", [(sb, sb)], "c_contiguous_1d",
                 params={"reduce": {"kind": "nansum", "axis": None, "keepdims": False}},
                 cid_tag=f"m4wide/{dt}/nansum/N={N}")
            if npdt.kind == "f":  # complex nanprod stays folded (npy_cmul gap #12) — not diverted
                pp = np.ascontiguousarray(_scatter_nan(_near_one_prod_pool(N)).astype(npdt))
                emit("in0", [(pp, pp)], "c_contiguous_1d",
                     params={"reduce": {"kind": "nanprod", "axis": None, "keepdims": False}},
                     cid_tag=f"m4wide/{dt}/nanprod/N={N}")

    # And one AXIS case per divert-exact dtype (the M2 route: C-contiguous child, PINNED inner axis > 128).
    for dt in ("float32", "float64", "complex128"):
        npdt = np.dtype(dt)
        shp = (3, 200)  # axis 1 (len 200 > 128) → PINNED pairwise add.reduce
        wp = _scatter_nan(_wide_sum_pool(int(np.prod(shp))))
        arr = (wp + 1j * np.roll(wp, 5)).astype(npdt) if npdt.kind == "c" else wp.astype(npdt)
        sb = np.ascontiguousarray(arr.reshape(shp))
        emit("in0", [(sb, sb)], "c_contiguous_2d",
             params={"reduce": {"kind": "nansum", "axis": 1, "keepdims": False}},
             cid_tag=f"m4wideaxis/{dt}/nansum[1]/3x200")

    # ---- C5. M4-tail: the ORDER-INDEPENDENT range / NaN-aware min-max family (plan P2 M4-tail) --
    # ptp / nanmin / nanmax are host-DELEGATED (the engine materializes the child, then reduces it with
    # np.ptp / np.nanmin / np.nanmax). Because a minimum, a maximum and their difference are the same
    # value in ANY visiting order, they are BIT-EXACT at every dtype with NO E1 excuse — so this block
    # is their teeth, not a "within N ULP" cushion. It sweeps every layout {C-1d, C-2d, F-2d, negstride}
    # so a wrong strided/reversed read of the materialized child turns it red; every axis + keepdims; and
    # both integer (dtype preserved, unsigned/signed overflow WRAPS) and float (NaN scattered → nanmin/
    # nanmax SKIP it; ptp PROPAGATES it → NaN) pools. Integer pools are built int→int (never float→uint,
    # a host-dependent conversion) so uint64's wrap is well-defined parity.
    def _c5_pool(N, npdt, with_nan):
        if npdt.kind in "iu":
            base = np.array([3, -7, 1, 5, 0, 4, -2, 6], dtype=np.int64)
            return np.tile(base, (N + 7) // 8)[:N].astype(npdt)   # modular int→int, no float→uint UB
        base = np.array([3.0, -7.0, 1.5, 5.0, 0.0, 4.0, -2.5, 6.0], dtype=np.float64)
        a = np.tile(base, (N + 7) // 8)[:N].astype(npdt)
        if with_nan:
            a[1::5] = np.nan                                       # scattered NaN (never a whole all-NaN slice here)
        return a

    c5_layouts = ["c_contiguous_1d", "c_contiguous_2d", "f_contiguous_2d", "negstride_1d"]
    for ln in c5_layouts:
        for dt in ("int32", "int64", "uint64", "float32", "float64"):
            npdt = np.dtype(dt)
            isfloat = npdt.kind == "f"
            # ptp over a NaN-FREE pool (a real finite range); nanmin/nanmax over a NaN pool on floats
            # (plain min/max on ints, which carry no NaN); plus ptp over a NaN pool on floats to pin the
            # NaN-propagation → NaN result.
            specs = [("ptp", False), ("nanmin", isfloat), ("nanmax", isfloat)]
            if isfloat:
                specs.append(("ptp", True))
            for kind, with_nan in specs:
                nb, nv = LAYOUTS[ln](npdt)
                nb.reshape(-1)[:] = _c5_pool(nb.size, npdt, with_nan)
                combos = [(None, False)] + [(ax, False) for ax in range(nv.ndim)]
                if nv.ndim > 0:
                    combos.append((0, True))
                    combos.append((None, True))              # flat + keepdims (plan P2 M5)
                for ax, kd in combos:
                    emit("in0", [(nb, nv)], ln,
                         params={"reduce": {"kind": kind, "axis": ax, "keepdims": kd}},
                         cid_tag=f"m4tail/{dt}/{kind}{'_nan' if with_nan else ''}[{ax},{int(kd)}]")

    # And one WIDE-magnitude flat ptp per float width (crosses PW_BLOCKSIZE — though ptp is a max/min
    # selection, not a sum, so it is order-exact regardless; this pins the large-N materialize path).
    for N in (129, 1000):
        for dt in ("float32", "float64", "int64"):
            npdt = np.dtype(dt)
            arr = _wide_sum_pool(N).astype(npdt)
            sb = np.ascontiguousarray(arr)
            emit("in0", [(sb, sb)], "c_contiguous_1d",
                 params={"reduce": {"kind": "ptp", "axis": None, "keepdims": False}},
                 cid_tag=f"m4tailwide/{dt}/ptp/N={N}")

    # ---- C6. M4c: the int64 INDEX kinds argmax / argmin (plan P2 M4c) --------------------------
    # These are the M4c kinds that land NOW (the summation-bound NanMean/Std/Var/Average wait on M3).
    # Like the M4-tail min/max they are host-DELEGATED (materialize the child, then np.argmax /
    # np.argmin over it), but the property is subtler than order-independence: an index DOES depend on
    # the C-order tie/NaN rule (first maximum / first NaN wins). It is nonetheless BIT-EXACT with NO E1
    # excuse because the materialized child is the fresh C-contiguous buffer NumPy's own argmax builds
    # and reduces — so this block is their teeth. It sweeps every layout {C-1d, C-2d, F-2d, negstride}
    # (a wrong strided/reversed read of the materialized child, OR a memory-order rather than
    # logical-C-order tie walk, turns it red — the `_c5_pool` tiles a max/min that REPEATS, so a
    # first-tie regression is caught); every axis + keepdims; int (result still int64, proving the index
    # kinds are dtype-independent) and float; and — for floats — a NaN pool so the first-NaN-wins index
    # is pinned. The result dtype the oracle records is int64 on both sides.
    for ln in c5_layouts:
        for dt in ("int32", "int64", "uint64", "float32", "float64"):
            npdt = np.dtype(dt)
            isfloat = npdt.kind == "f"
            specs = [("argmax", False), ("argmin", False)]
            if isfloat:
                specs += [("argmax", True), ("argmin", True)]     # first-NaN-wins index
            for kind, with_nan in specs:
                nb, nv = LAYOUTS[ln](npdt)
                nb.reshape(-1)[:] = _c5_pool(nb.size, npdt, with_nan)
                combos = [(None, False)] + [(ax, False) for ax in range(nv.ndim)]
                if nv.ndim > 0:
                    combos.append((0, True))
                    combos.append((None, True))              # flat + keepdims (plan P2 M5)
                for ax, kd in combos:
                    emit("in0", [(nb, nv)], ln,
                         params={"reduce": {"kind": kind, "axis": ax, "keepdims": kd}},
                         cid_tag=f"m4c/{dt}/{kind}{'_nan' if with_nan else ''}[{ax},{int(kd)}]")

    # A FUSED child (not identity) proves the materialize-then-argmax path end to end, and a wide-N flat
    # case pins the large-N materialize (argmax is a selection, so order-exact regardless of N).
    for expr in ("mul(in0,in0)", "sub(in0,li:1)"):
        for dt in ("int64", "float64"):
            npdt = np.dtype(dt)
            nb, nv = LAYOUTS["c_contiguous_2d"](npdt)
            nb.reshape(-1)[:] = _c5_pool(nb.size, npdt, False)
            for kind in ("argmax", "argmin"):
                for ax, kd in [(None, False), (0, False), (1, True)]:
                    emit(expr, [(nb, nv)], "c_contiguous_2d",
                         params={"reduce": {"kind": kind, "axis": ax, "keepdims": kd}},
                         cid_tag=f"m4c/{dt}/{kind}[{ax},{int(kd)}]/{expr}")
    for N in (129, 1000):
        for dt in ("float64", "int64"):
            npdt = np.dtype(dt)
            sb = np.ascontiguousarray(_wide_sum_pool(N).astype(npdt))
            for kind in ("argmax", "argmin"):
                emit("in0", [(sb, sb)], "c_contiguous_1d",
                     params={"reduce": {"kind": kind, "axis": None, "keepdims": False}},
                     cid_tag=f"m4cwide/{dt}/{kind}/N={N}")

    # ---- C7. M4c summation kinds: NanMean / Var / Std (plan P2 M4c summation) ------------------
    # These are the summation-bound M4c kinds. Unlike the M4-tail/index kinds they SUM, so they can't
    # delegate to the engine's drifting np.nanmean/np.var — the host materializes the child once and
    # reproduces NumPy's nanmean / _var op for op over the SAME pairwise sum the M1/M2 diverts use
    # (mean = pairwise_sum/N; var = pairwise_sum((x-mean)²)/max(N-ddof,0); std = sqrt(var); nanmean =
    # pairwise_sum(NaN→0)/count). BIT-EXACT with NO E1 excuse, so this block is their teeth. The AXIS
    # path rides the engine's C-contiguous axis add.reduce, so — like M2 — the strict-F multi-D AXIS
    # corner is excluded here (only FLAT is emitted for f_contiguous_2d; a flat reduce of a contiguous
    # child is memory-order bit-exact whatever the layout). float16 + decimal are unsupported (they need
    # a pairwise kernel that dtype lacks — the "Half not diverted" gap M1/M2 share) and held out.
    def _c7_pool(N, npdt, with_nan):
        if npdt.kind in "iu":
            base = np.array([3, -7, 1, 5, 0, 4, -2, 6], dtype=np.int64)
            return np.tile(base, (N + 7) // 8)[:N].astype(npdt)     # int→int, no float→uint UB
        if npdt.kind == "c":
            base = np.array([1 + 2j, 3 - 1j, -2 + 0.5j, 4 + 4j, 0 + 0j, -3 - 2j, 2.5 + 1.5j, 5 - 3j])
            a = np.tile(base, (N + 7) // 8)[:N].astype(npdt)
        else:
            base = np.array([3.0, -7.0, 1.5, 5.0, 0.0, 4.0, -2.5, 6.0], dtype=np.float64)
            a = np.tile(base, (N + 7) // 8)[:N].astype(npdt)
        if with_nan:
            a[1::5] = np.nan                                          # scattered NaN (not a whole slice)
        return a

    c7_dtypes = ("bool", "uint8", "int32", "int64", "uint64", "float32", "float64", "complex128")
    for ln in c5_layouts:
        for dt in c7_dtypes:
            npdt = np.dtype(dt)
            isfloat = npdt.kind in "fc"
            isFlayout = (ln == "f_contiguous_2d")   # strict-F multi-D AXIS is the excluded corner
            # nanmean over a clean pool + (for float/complex) a scattered-NaN pool; var/std over a clean
            # pool at ddof 0 and 1 (bool var has zero variance but is a legal float64 result).
            specs = [("nanmean", False, 0), ("var", False, 0), ("var", False, 1), ("std", False, 0)]
            if isfloat:
                specs += [("nanmean", True, 0), ("var", True, 0)]    # NaN-skip mean; NaN-propagating var
            for kind, with_nan, dd in specs:
                nb, nv = LAYOUTS[ln](npdt)
                nb.reshape(-1)[:] = _c7_pool(nb.size, npdt, with_nan)
                combos = [(None, False)]
                if nv.ndim > 0:
                    combos.append((None, True))                       # flat + keepdims (plan P2 M5 — valid for every layout, incl. strict-F: a flat reduce of a contiguous child is memory-order exact)
                if not isFlayout:
                    combos += [(ax, False) for ax in range(nv.ndim)]
                    if nv.ndim > 1:
                        combos.append((1, True))                      # keepdims on a non-flat axis
                for ax, kd in combos:
                    red = {"kind": kind, "axis": ax, "keepdims": kd}
                    if kind in ("var", "std"):
                        red["ddof"] = dd
                    emit("in0", [(nb, nv)], ln, params={"reduce": red},
                         cid_tag=f"m4csum/{dt}/{kind}{'_nan' if with_nan else ''}{'_dd'+str(dd) if dd else ''}[{ax},{int(kd)}]")

    # WIDE-magnitude flat cases — the teeth for the pairwise summation (a naive/multi-accumulator sum
    # would diverge here where it matches on the benign pool above; crosses PW_BLOCKSIZE 128 + the leaf).
    for N in (7, 8, 127, 128, 129, 257, 1000):
        for dt in ("float32", "float64", "complex128"):
            npdt = np.dtype(dt)
            arr = _wide_sum_pool(N).astype(npdt)
            if npdt.kind == "c":
                arr = arr + 1j * _wide_sum_pool(N)[::-1].astype(np.float64)  # non-trivial imaginary part
            sb = np.ascontiguousarray(arr)
            for kind in ("var", "std", "nanmean"):
                red = {"kind": kind, "axis": None, "keepdims": False}
                if kind in ("var", "std"):
                    red["ddof"] = 0
                emit("in0", [(sb, sb)], "c_contiguous_1d", params={"reduce": red},
                     cid_tag=f"m4csumwide/{dt}/{kind}/N={N}")

    # A FUSED child (not identity) proves the materialize-then-stat path end to end, flat + axis.
    for expr in ("mul(in0,in0)", "sub(in0,lf:1.5)"):
        for dt in ("int64", "float64", "complex128"):
            npdt = np.dtype(dt)
            nb, nv = LAYOUTS["c_contiguous_2d"](npdt)
            nb.reshape(-1)[:] = _c7_pool(nb.size, npdt, False)
            for kind in ("var", "std", "nanmean"):
                for ax, kd in [(None, False), (0, False), (1, True)]:
                    red = {"kind": kind, "axis": ax, "keepdims": kd}
                    if kind in ("var", "std"):
                        red["ddof"] = 0
                    emit(expr, [(nb, nv)], "c_contiguous_2d", params={"reduce": red},
                         cid_tag=f"m4csum/{dt}/{kind}[{ax},{int(kd)}]/{expr}")

    # ---- C8. Weighted average: Σ(v·w)/Σ(w) over TWO trees (plan P2 M4c-average) ----------------
    # The FIRST reduction over two operand trees. Host-computed with the SAME NumPy-exact pairwise sum
    # the M1/M2 diverts use (ExactSumArray) — NOT the drifting multi-accumulator engine sum the library
    # np.average itself uses, so the FUSED Average is BIT-EXACT with NumPy where np.average is only
    # allclose at large N (this block's wide-N cases are the teeth). in0 = values, in1 = weights.
    # Weights are POSITIVE (Σw ≠ 0 — the zero-weight DivideByZeroException is unit-tested, not corpus'd).
    # Result dtype is float/complex; (float16,float16)→float16 and decimal are rejected (no pairwise
    # kernel) and held out — every pair below resolves to Single/Double/Complex. Like C7, the strict-F
    # multi-D AXIS corner is FLAT-only (M2's excused corner).
    def _c8_values(N, npdt):
        if npdt == np.bool_:
            base = np.array([True, False, True, True, False, True, False, True])
            return np.tile(base, (N + 7) // 8)[:N]
        if npdt.kind in "iu":
            base = np.array([3, -7, 1, 5, 2, 4, -2, 6], dtype=np.int64)
            return np.tile(base, (N + 7) // 8)[:N].astype(npdt)       # int→int, no float→uint UB
        if npdt.kind == "c":
            base = np.array([1 + 2j, 3 - 1j, -2 + 0.5j, 4 + 4j, 1 + 0j, -3 - 2j, 2.5 + 1.5j, 5 - 3j])
            return np.tile(base, (N + 7) // 8)[:N].astype(npdt)
        base = np.array([3.0, -7.0, 1.5, 5.0, 2.0, 4.0, -2.5, 6.0], dtype=np.float64)
        return np.tile(base, (N + 7) // 8)[:N].astype(npdt)

    def _c8_weights(N, npdt):        # POSITIVE weights so the total weight is never zero
        if npdt == np.bool_:
            return np.ones(N, dtype=np.bool_)                          # bool weights: all True (Σ = N)
        if npdt.kind in "iu":
            base = np.array([2, 3, 4, 5, 1, 2, 3, 4], dtype=np.int64)
            return np.tile(base, (N + 7) // 8)[:N].astype(npdt)
        if npdt.kind == "c":
            base = np.array([2 + 0j, 3 + 1j, 4 - 0.5j, 5 + 2j, 1 + 0j, 2 - 1j, 3 + 0.5j, 4 + 3j])
            return np.tile(base, (N + 7) // 8)[:N].astype(npdt)
        base = np.array([2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        return np.tile(base, (N + 7) // 8)[:N].astype(npdt)

    # (values, weights) dtype pairs whose np.average result is Single / Double / Complex (never Half):
    c8_pairs = [
        ("float64", "float64"), ("float32", "float32"), ("float32", "float64"),
        ("int64", "int64"), ("int32", "int32"), ("uint8", "uint8"), ("int8", "int8"),
        ("int64", "float64"), ("float32", "int64"), ("bool", "int64"),
        ("complex128", "complex128"), ("complex128", "float64"), ("float64", "complex128"),
        ("float16", "float32"),   # f2 values with wider f4 weights → f4 (Single); result is not Half
    ]
    for ln in c5_layouts:
        for (vt, wt) in c8_pairs:
            vb, vv = LAYOUTS[ln](np.dtype(vt))
            wb, wv = LAYOUTS[ln](np.dtype(wt))
            vb.reshape(-1)[:] = _c8_values(vb.size, np.dtype(vt))
            wb.reshape(-1)[:] = _c8_weights(wb.size, np.dtype(wt))
            isFlayout = (ln == "f_contiguous_2d")
            combos = [(None, False)]
            if vv.ndim > 0:
                combos.append((None, True))                          # flat + keepdims (plan P2 M5)
            if not isFlayout:
                combos += [(ax, False) for ax in range(vv.ndim)]
                if vv.ndim > 1:
                    combos.append((1, True))
            for ax, kd in combos:
                red = {"kind": "average", "axis": ax, "keepdims": kd, "weights": "in1"}
                emit("in0", [(vb, vv), (wb, wv)], ln, params={"reduce": red},
                     cid_tag=f"m4cavg/{vt},{wt}[{ax},{int(kd)}]")

    # WIDE-magnitude flat teeth: a naive / multi-accumulator sum diverges here where it matches on the
    # benign pool above — so these prove BOTH sums take the pairwise path (crosses PW_BLOCKSIZE 128).
    for N in (7, 8, 127, 128, 129, 257, 1000):
        for dt in ("float32", "float64", "complex128"):
            npdt = np.dtype(dt)
            vv = _wide_sum_pool(N).astype(npdt)
            if npdt.kind == "c":
                vv = vv + 1j * _wide_sum_pool(N)[::-1].astype(np.float64)
            ww = (np.abs(_wide_sum_pool(N)) + 0.5).astype(npdt)       # positive-real weights (Σ ≠ 0)
            vb = np.ascontiguousarray(vv)
            wb = np.ascontiguousarray(ww)
            red = {"kind": "average", "axis": None, "keepdims": False, "weights": "in1"}
            emit("in0", [(vb, vb), (wb, wb)], "c_contiguous_1d", params={"reduce": red},
                 cid_tag=f"m4cavgwide/{dt}/N={N}")

    # A FUSED values / weights tree (not identity) proves the materialize-then-average path end to end.
    for dt in ("int64", "float64", "complex128"):
        npdt = np.dtype(dt)
        vb, vv = LAYOUTS["c_contiguous_2d"](npdt)
        wb, wv = LAYOUTS["c_contiguous_2d"](npdt)
        vb.reshape(-1)[:] = _c8_values(vb.size, npdt)
        wb.reshape(-1)[:] = _c8_weights(wb.size, npdt)
        for ax, kd in [(None, False), (0, False), (1, True)]:
            red = {"kind": "average", "axis": ax, "keepdims": kd, "weights": "add(in1,in1)"}
            emit("mul(in0,in0)", [(vb, vv), (wb, wv)], "c_contiguous_2d", params={"reduce": red},
                 cid_tag=f"m4cavg/{dt}[{ax},{int(kd)}]/fused")

    # ---- D. out= (returned view + the whole base buffer behind out) -------------------------
    out_exprs = ["add(mul(in0,in1),in0)", "gt(in0,in1)", "where(gt(in0,in1),in0,in1)", "sqrt(abs(in0))"]
    for shape in [(8,), (4, 5)]:
        cnt = int(np.prod(shape))
        for dt in ["float64", "float32", "int32", "int64", "uint8"]:
            a = _fill(cnt, np.dtype(dt)).reshape(shape)
            b2 = np.roll(_fill(cnt, np.dtype(dt)), 1).reshape(shape)
            for expr in out_exprs:
                probe = np.asarray(_ev_eval(expr, [a, b2]))
                for out_kind in ["c", "f", "strided", "negstride", "offset"]:
                    # sorted(): a bare set of dtype names iterates in string-hash order, which Python randomizes per
                    # process (PYTHONHASHSEED) — the corpus order (and every later case id's running counter) must not.
                    for out_dt in sorted({probe.dtype.name, "float64" if probe.dtype.kind == "f" else probe.dtype.name}):
                        built = _out_view(shape, np.dtype(out_dt), out_kind)
                        if built is None:
                            continue
                        ob, ov = built
                        if not np.can_cast(probe.dtype, ov.dtype, casting="same_kind"):
                            continue
                        emit(expr, [(a, a), (b2, b2)], f"out_{out_kind}", out=(ob, ov),
                             cid_tag=f"{dt}->{out_dt}/{'x'.join(map(str, shape))}/{expr}")

    # ---- D2. where= — masked writes into out= (plan P4.5) ------------------------------------
    # np.evaluate(expr, out=dst, where=mask) writes the fused result only where the mask is True,
    # leaving masked-off dst slots at their PRIOR contents — NumPy's ufunc where= convention (the mask
    # rides the iterator as a trailing ARRAYMASK operand, the output is WRITEMASKED, and ForEach's
    # masked driver runs the fused kernel per mask-true run). The tuple result records BOTH the returned
    # view AND the whole out base, so a kernel that ignored the mask (wrote every slot) turns the gate
    # red. A FRESH out view is built per case so each starts from clean prior contents. The out dtype
    # stays same-kind-castable from the result (a widening float→float64 out gates the WRITEMASKED +
    # buffered-flush cast path; the C-undefined float→int mask+cast edge is unit-tested, not byte-gated,
    # matching the B5/out block policy). The mask sweep covers dense/alternating/all-true/all-false plus
    # strided, negative-stride, and — at rank 2 — broadcasting row/column masks.
    where_exprs = ["add(mul(in0,in1),in0)", "sqrt(abs(in0))"]
    where_masks_1d = ["checker", "alt", "all_true", "all_false", "strided", "negstride"]
    where_masks_2d = where_masks_1d + ["row", "col"]
    for shape in [(8,), (4, 5)]:
        cnt = int(np.prod(shape))
        mask_kinds = where_masks_2d if len(shape) == 2 else where_masks_1d
        for dt in ["float64", "float32", "int32", "int64"]:
            a = _fill(cnt, np.dtype(dt)).reshape(shape)
            b2 = np.roll(_fill(cnt, np.dtype(dt)), 1).reshape(shape)
            for expr in where_exprs:
                probe = np.asarray(_ev_eval(expr, [a, b2]))
                out_dts = {probe.dtype.name}
                if probe.dtype.kind == "f":
                    out_dts.add("float64")            # the masked buffered-flush cast path
                for out_dt in sorted(out_dts):
                    if not np.can_cast(probe.dtype, np.dtype(out_dt), casting="same_kind"):
                        continue
                    for out_kind in ["c", "strided"]:
                        for mk in mask_kinds:
                            built = _out_view(shape, np.dtype(out_dt), out_kind)  # FRESH prior per case
                            if built is None:
                                continue
                            mv = _mask_view(shape, mk)
                            if mv is None:
                                continue
                            emit(expr, [(a, a), (b2, b2)], f"where_{out_kind}", out=built, where=mv,
                                 cid_tag=f"{dt}->{out_dt}/{mk}/{'x'.join(map(str, shape))}/{expr}")

    # ---- C9. flat Min / Max at NumPy-EXACT bits (plan "Perf review 2026-09-23" item 3) -------------
    # (Appended LAST on purpose: every case id carries the running counter, so a block inserted earlier would
    # renumber every later case and turn a pure addition into a whole-file corpus diff.)
    # Block C folds signed zeros into +0 for min/max: the fused 4-accumulator fold matched NumPy's VALUE but not
    # WHICH ZERO SIGN survives a ±0 tie. The host now reproduces NumPy's own contiguous reduction schedule
    # (simd_reduce_c_{max,min}: 8-vector groups seeded with splat(x[0]), the canonical-NaN horizontal step, the
    # scalar tail) over the very buffer NumPy reduces, so the sign of a zero result is contractual HERE. The pools
    # put the extreme (±0) at a few LANE-STRUCTURED stream positions — different lanes, groups and stream scratch
    # blocks (1024 float64 / 2048 float32 elements) — with both sign assignments: exactly the cases a sequential
    # scan, a per-block restart or a wrong horizontal order gets wrong. Routes covered: a contiguous LEAF (in place:
    # C 1-D, C 2-D, F 2-D), a streamed product (C / F), a materialized product (strided / negstride operands). A
    # NON-contiguous bare leaf is deliberately absent: NumPy reduces it with its scalar 8-accumulator unroll, a
    # different schedule the host does not port (it keeps the fold). NaN pools pin propagation only — BitDiff
    # tokenizes NaN, so the canonical-vs-payload bits are unit-tested (NDEvaluateMinMaxTests).
    def _c9_pool(N, npdt, kind, flip):
        fill = -1.0 if kind == "max" else 1.0
        a = np.full(N, fill, dtype=np.float64)
        for k, s in enumerate((0, 3, 28, 61, 1025, 2049, 2900, 4990)):   # stream positions (array index s + 1)
            if s + 1 < N:
                a[s + 1] = -0.0 if ((k % 2 == 0) != flip) else 0.0
        return a.astype(npdt)

    def _c9_signs(N, npdt):   # ±1: flips zero signs through the product; both sides compute identical bits
        return np.where((np.arange(N) * 7) % 3 == 0, -1.0, 1.0).astype(npdt)

    # Sizing: operands are stored as hex, so a large pool is replicated per route; the grid keeps every lane /
    # group / block feature (N = 33 / 65 one float64 / float32 group, 129 several groups + tail, 1089 / 2177 across the
    # 1024 / 2048-element stream-block boundary with zeros in DIFFERENT lanes of adjacent blocks) but spends the
    # keepdims and both-sign variants only on the small sizes (keepdims only reshapes the one result element).
    def _c9_1d(dt, npdt, kind, N, flip, keepdims_too, routes_too):
        pool = _c9_pool(N, npdt, kind, flip)
        sg = _c9_signs(N, npdt)
        cb, sb = np.ascontiguousarray(pool), np.ascontiguousarray(sg)
        for kd in ((False, True) if keepdims_too else (False,)):
            red = {"kind": kind, "axis": None, "keepdims": kd}
            emit("in0", [(cb, cb)], "c_contiguous_1d", params={"reduce": red},
                 cid_tag=f"c9/{dt}/{kind}/leaf/N={N}/f{int(flip)}[{int(kd)}]")
            emit("mul(in0,in1)", [(cb, cb), (sb, sb)], "c_contiguous_1d", params={"reduce": red},
                 cid_tag=f"c9/{dt}/{kind}/mul/N={N}/f{int(flip)}[{int(kd)}]")
        if not routes_too:
            return
        # Strided operands (every other element of a 2N base) and reversed operands: the product is materialized in
        # logical order on both sides, then reduced with the schedule.
        red = {"kind": kind, "axis": None, "keepdims": False}
        stb = np.full(2 * N, 7.0 if kind == "min" else -7.0, dtype=npdt)
        stb[::2] = pool
        ssb = np.ones(2 * N, dtype=npdt)
        ssb[::2] = sg
        emit("mul(in0,in1)", [(stb, stb[::2]), (ssb, ssb[::2])], "strided_step2_1d",
             params={"reduce": red}, cid_tag=f"c9/{dt}/{kind}/mulstrided/N={N}/f{int(flip)}")
        nb, nsb = np.ascontiguousarray(pool[::-1]), np.ascontiguousarray(sg[::-1])
        emit("mul(in0,in1)", [(nb, nb[::-1]), (nsb, nsb[::-1])], "negstride_1d",
             params={"reduce": red}, cid_tag=f"c9/{dt}/{kind}/mulneg/N={N}/f{int(flip)}")

    for dt in ("float64", "float32"):
        npdt = np.dtype(dt)
        for kind in ("max", "min"):
            for N in (2, 3, 33, 65, 129):
                for flip in (False, True):
                    _c9_1d(dt, npdt, kind, N, flip, keepdims_too=True, routes_too=True)
            # The stream-block boundary: one sign assignment per size, the two sizes covering both.
            _c9_1d(dt, npdt, kind, 1089, False, keepdims_too=False, routes_too=True)
            _c9_1d(dt, npdt, kind, 2177, True, keepdims_too=False, routes_too=False)
            # 2-D: the C leaf and product reduce memory order = logical order; the F leaf and all-F product reduce
            # the F MEMORY order (NumPy coalesces an F-contiguous array into one inner loop over memory).
            for (r, c), flips, kds in (((9, 7), (False, True), (False, True)), ((33, 31), (False,), (False,))):
                for flip in flips:
                    m = _c9_pool(r * c, npdt, kind, flip).reshape(r, c)
                    s2 = _c9_signs(r * c, npdt).reshape(r, c)
                    cm, cs = np.ascontiguousarray(m), np.ascontiguousarray(s2)
                    fm, fs = np.asfortranarray(m), np.asfortranarray(s2)
                    for kd in kds:
                        red = {"kind": kind, "axis": None, "keepdims": kd}
                        emit("in0", [(cm, cm)], "c_contiguous_2d", params={"reduce": red},
                             cid_tag=f"c9/{dt}/{kind}/leaf2d/{r}x{c}/f{int(flip)}[{int(kd)}]")
                        emit("in0", [(fm, fm)], "f_contiguous_2d", params={"reduce": red},
                             cid_tag=f"c9/{dt}/{kind}/leafF/{r}x{c}/f{int(flip)}[{int(kd)}]")
                        emit("mul(in0,in1)", [(cm, cm), (cs, cs)], "c_contiguous_2d", params={"reduce": red},
                             cid_tag=f"c9/{dt}/{kind}/mul2d/{r}x{c}/f{int(flip)}[{int(kd)}]")
                        emit("mul(in0,in1)", [(fm, fm), (fs, fs)], "f_contiguous_2d", params={"reduce": red},
                             cid_tag=f"c9/{dt}/{kind}/mulF/{r}x{c}/f{int(flip)}[{int(kd)}]")
            # NaN propagation through every route (a vector-section NaN and a tail-only NaN).
            for N in (40, 1089):
                for pos in (5, N - 1):
                    a = _c9_pool(N, npdt, kind, False)
                    a[pos] = np.nan
                    ca, sa = np.ascontiguousarray(a), np.ascontiguousarray(_c9_signs(N, npdt))
                    red = {"kind": kind, "axis": None, "keepdims": False}
                    emit("in0", [(ca, ca)], "c_contiguous_1d", params={"reduce": red},
                         cid_tag=f"c9/{dt}/{kind}/nanleaf/N={N}/p{pos}")
                    emit("mul(in0,in1)", [(ca, ca), (sa, sa)], "c_contiguous_1d", params={"reduce": red},
                         cid_tag=f"c9/{dt}/{kind}/nanmul/N={N}/p{pos}")

    if skipped:
        print(f"  (skipped {skipped} evaluate cells: NumPy raised a non-verbatim error, or complex64 width)")
    return cases


# T-real_if_close — np.real_if_close (real_if_close.jsonl). The complex-collapse decision:
# `if all(absolute(a.imag) < tol): a = a.real`. NumSharp fuses NumPy's absolute+<+all into ONE
# early-exit scan over the imaginary lane (ImagCloseScan), so this tier must exercise BOTH outcomes
# (collapse -> float64 real view, no-collapse -> complex128 unchanged) across the tol modes
# (tol>1 => machine-epsilon multiples; tol<=1 => absolute; tol<=0 => nothing collapses) and every
# scan path (dense contiguous / F-contiguous / negative-stride / strided-inner gather / broadcast /
# 0-d / empty). The result is pure copies of stored bits (real lane or the array unchanged), so it is
# host-INDEPENDENT and byte-exact everywhere -> the portable RunCorpus tier. NaN/inf imaginary parts
# and the strict `<` boundary (imag == tol) are gated too.
def gen_real_if_close():
    cases = []
    n = [0]
    eps = float(np.finfo(np.complex128).eps)   # 2.22e-16 — the ONLY eps (NumSharp has one complex dtype)

    def reals(count):
        # Exact-float64 reals with alternating sign; (i+1)+0.5 is an exact binary fraction so the
        # collapsed real lane round-trips bit-for-bit through the corpus buffer.
        return np.array([((i + 1) + 0.5) * (1 if i % 2 == 0 else -1) for i in range(count)], dtype=np.float64)

    # Imaginary-lane patterns over a length-`count` array. Each returns the imaginary values; the
    # collapse outcome is decided per (pattern, tol) by NumPy itself when the case is emitted.
    def imag_tiny(count):
        # All strictly within eps*100 (2.22e-14): mix of +/- tiny, exact zero and NEGATIVE zero
        # (|-0.0| == 0 collapses) — collapses for every tol > 0, stays complex for tol <= 0.
        pool = [0.0, -0.0, 1e-15, -1e-16, 2e-15, -3e-15, 1e-14, -5e-16, 0.0, 4e-15, -2e-15, 7e-16]
        return np.array([pool[i % len(pool)] for i in range(count)], dtype=np.float64)

    def imag_onebig(count):
        # Tiny everywhere except one 0.5 — out of band for tol=100 (2.22e-14) but within an absolute
        # tol=1; exactly ON the boundary for tol=0.5 (strict `<` => still no collapse there).
        im = imag_tiny(count)
        im[count // 2] = 0.5
        return im

    def imag_nan(count):
        im = imag_tiny(count)
        im[min(2, count - 1)] = np.nan     # a NaN imaginary part -> never collapses
        return im

    def imag_inf(count):
        im = imag_tiny(count)
        im[min(1, count - 1)] = np.inf      # +inf / -inf imaginary parts -> never collapse
        im[min(count - 1, 3)] = -np.inf
        return im

    def imag_boundary(count):
        im = imag_tiny(count)
        im[0] = eps * 100                   # exactly the resolved tol at tol=100 -> strict `<` fails
        return im

    PATTERNS = [
        ("tiny", imag_tiny, False),          # collapses for tol>0
        ("onebig", imag_onebig, False),
        ("nan_imag", imag_nan, False),
        ("inf_imag", imag_inf, False),
        ("boundary", imag_boundary, False),
        ("nan_real", imag_tiny, True),       # reals carry a NaN; imag tiny -> collapses to a real NaN
    ]
    TOLS = [100.0, 1000.0, 1e6, 1.0, 0.5, 0.1, 0.0, -5.0]

    def emit(base, view, tol, layout):
        # NumPy is the oracle for BOTH the collapse decision and the resulting dtype/shape/bytes.
        r = np.real_if_close(view, tol=tol)
        exp_shape = [int(d) for d in r.shape]                 # read BEFORE ascontiguousarray (0-D safe)
        exp_buf = np.ascontiguousarray(r).tobytes().hex()
        cases.append({
            "id": f"real_if_close/{layout}/tol={tol}/{n[0]}",
            "op": "real_if_close",
            "params": {"tol": float(tol)},
            "operands": [describe(base, view)],
            "expected": {"dtype": r.dtype.name, "shape": exp_shape, "buffer": exp_buf},
            "layout": layout,
            "valueclass": "mixed",
        })
        n[0] += 1

    def make_base(count, imag_fn, nan_real):
        re = reals(count)
        if nan_real and count > 0:
            re[min(2, count - 1)] = np.nan
        return (re + 1j * imag_fn(count)).astype(np.complex128)   # C-contiguous complex128

    # ---- Main matrix: length-12 arrays across every scan path × pattern × tol ----------------
    N = 12
    for label, imag_fn, nan_real in PATTERNS:
        a = make_base(N, imag_fn, nan_real)                        # C-contiguous 1-D
        # Interleaved base whose EVEN elements are `a` and odd elements carry a large imag (0.9) that
        # the ::2 view never addresses — proves the strided scan reads only the view's own elements.
        inter = np.empty(2 * N, dtype=np.complex128)
        inter[0::2] = a
        inter[1::2] = np.array([9.0 + 0.9j] * N)
        a2d = a.reshape(3, 4)                                      # C-contiguous 2-D
        wide = np.empty((3, 8), dtype=np.complex128)               # for a strided-column view
        wide[:, 0::2] = a2d
        wide[:, 1::2] = np.array([9.0 + 0.9j])
        for tol in TOLS:
            emit(a, a, tol, f"c_1d/{label}")                       # dense contiguous
            emit(a, a[::-1], tol, f"negstride_1d/{label}")         # reversed run (ScanRun stride -1)
            emit(inter, inter[::2], tol, f"strided_step2_1d/{label}")  # inner stride 2 -> gather
            emit(a2d, a2d, tol, f"c_2d/{label}")                   # dense 2-D
            emit(a2d, a2d.T, tol, f"transposed_2d/{label}")        # transpose of C-contig => F-contig dense
            emit(wide, wide[:, ::2], tol, f"strided_cols_2d/{label}")  # odometer + inner gather
        # Broadcast: a single tiny-imag element stretched — collapses; a single big-imag element — stays.
        one_tiny = np.array([1.5 + 1e-15j], dtype=np.complex128)
        one_big = np.array([1.5 + 0.5j], dtype=np.complex128)
        for tol in (100.0, 1.0, 0.0):
            emit(one_tiny, np.broadcast_to(one_tiny, (6,)), tol, f"broadcast_1d/{label}")
            emit(one_big, np.broadcast_to(one_big, (6,)), tol, f"broadcast_big_1d/{label}")

    # ---- Edge lengths: 0-d scalar, empty, 1- and 2-element (SIMD tail / vacuous-all) ----------
    for tol in (100.0, 1.0, 0.0):
        z_tiny = np.array(1.5 + 1e-15j, dtype=np.complex128)       # 0-d, collapses (tol>0)
        z_big = np.array(1.5 + 0.5j, dtype=np.complex128)          # 0-d, stays complex at tol=100
        emit(z_tiny, z_tiny, tol, "scalar_0d_tiny")
        emit(z_big, z_big, tol, "scalar_0d_big")
        for count in (0, 1, 2, 3, 7):                              # empty (vacuous all -> collapse) + short tails
            a = make_base(count, imag_tiny, False)
            emit(a, a, tol, f"len{count}_tiny")
        one_big = make_base(3, imag_onebig, False)                 # short array with an out-of-band imag
        emit(one_big, one_big, tol, "len3_onebig")

    # ---- Non-complex inputs: returned UNCHANGED (the collapse only applies to complex) ---------
    for dt in ("int32", "int64", "float64", "float32", "bool", "uint8"):
        base = _cbase((6,), np.dtype(dt))
        for tol in (100.0, 0.0):
            emit(base, base, tol, f"noncomplex_{dt}")

    print(f"  (real_if_close: {n[0]} cases)")
    return cases


def write_jsonl(path, cases):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases -> {path}")


# ---- ndarray INSTANCE surface (coverage plan §D / G0) --------------------------------------
#
# The registry dispatched np.foo(a) ~300 times vs a.foo() ~5 — so instance-default and
# overload divergences (a.max(axis) vs np.max, a.reshape(-1), a.round(n), in-place a.sort())
# had NO differential coverage. Op keys carry the "ndarray." prefix (the ma.* convention), so
# OracleSurfaceCoverageTests can discover instance coverage from the corpus, and
# MisalignedRegistry strips the prefix so the shared excuse branches (float var/std order,
# complex ULP envelopes) apply to the instance spelling exactly as to the np.* one.
#
# Result kinds: plain methods -> array; a.item()/len(a)/property scalars -> scalar; a.nonzero()
# -> tuple; IN-PLACE mutators (sort/partition/fill/put) -> tuple of [post-call view bytes,
# post-call WHOLE base buffer] — the out_where two-slot contract, so a mutator that writes
# outside a strided view's window is caught, and NumPy's post-call operand IS the oracle
# (plan §D3's operand-after comparator, expressed with the existing tuple machinery).

INSTANCE_LAYOUTS = [
    "c_contiguous_1d", "c_contiguous_2d", "c_contiguous_3d", "f_contiguous_2d",
    "transposed_2d", "strided_step2_1d", "negstride_1d", "simple_slice_offset_1d",
    "scalar_0d", "one_element_1d",
]
# All 13 NumPy-expressible dtypes (dtype-spread gate): Char is woven via char_tier("instance")
# on the uint16 proxy; Decimal has no NumPy oracle - its instance coverage rides the same engine
# paths the decimal_* tiers gate (documented in OracleCoverageStrengthTests.FixedDtypeOps).
INSTANCE_DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64",
                   "uint64", "float16", "float32", "float64", "complex128"]


def gen_instance(dtypes=None):
    """dtypes=None sweeps INSTANCE_DTYPES; char_tier("instance") passes ["uint16"] so the whole
    tier re-runs on the Char proxy (dedicated dot/searchsorted/choose + resize jobs included)."""
    cases = []
    n = 0
    skipped = 0
    sweep = dtypes if dtypes is not None else INSTANCE_DTYPES
    dedicated = dtypes if dtypes is not None else ("int32", "uint8", "float64", "complex128")
    resize_dts = dtypes if dtypes is not None else ("int32", "float64", "float16", "complex128")

    def emit(op, params, operand_pairs, layout, dtname, tag, run, kind=None):
        """Record one instance case: run() returns the numpy result (an ndarray for kind=array,
        a python scalar for kind=scalar, a prebuilt expected dict for kind=tuple), or raises —
        raising cells become error-parity rows exactly like the errors_full tier."""
        nonlocal n, skipped
        operands = [describe(b, v) for (b, v) in operand_pairs]
        cid = f"{op}/{layout}/{dtname}/{tag}/{n}"
        try:
            with np.errstate(all="ignore"):
                r = run()
        except Exception as e:
            cases.append(_error_case(op, params, operands, e, layout, cid=cid, kind=kind))
            n += 1
            return
        if kind == "tuple":
            expected = r                      # prebuilt by the caller (_tuple_expected)
        elif kind == "scalar":
            expected = _arr_expected(np.asarray(r), kind="scalar")
        else:
            expected = _arr_expected(r)
        cases.append(_case(op, params, operands, expected, layout, "instance", cid=cid))
        n += 1

    # ---- dual-form value methods (plan §D1) + property reads (§D4) ----
    for ln in INSTANCE_LAYOUTS:
        for s in sweep:
            dt = np.dtype(s)
            base, view = LAYOUTS[ln](dt)
            pair = [(base, view)]

            # reductions through the INSTANCE defaults (a.max(axis=…) etc.)
            for opname in ("all", "any", "max", "min", "mean", "sum", "prod", "std", "var"):
                for ax in (None, 0):
                    if ax is not None and view.ndim == 0:
                        continue
                    emit(f"ndarray.{opname}", {"axis": ax}, pair, ln, s, f"axis={ax}",
                         lambda v=view, o=opname, ax=ax: np.asarray(getattr(v, o)(axis=ax)))
            for opname in ("argmax", "argmin", "argsort"):
                if view.ndim == 0:
                    continue                          # instance argmax/argmin need an axis in NumSharp
                for ax in (0, -1):
                    emit(f"ndarray.{opname}", {"axis": ax}, pair, ln, s, f"axis={ax}",
                         lambda v=view, o=opname, ax=ax: np.asarray(getattr(v, o)(axis=ax)))
            # ndarray.partition/argpartition are deliberately ABSENT: the arrangement BETWEEN kth
            # anchors is introselect-implementation-specific on BOTH sides (whole-output bytes are
            # not contractual — the sort tier pins the DERIVED kth-values instead), and the
            # instance spelling delegates to the same introselect np.partition already gates.

            # conversions / copies / reshapes
            for target in ("float64", "int32"):
                if s == "complex128" and target == "int32":
                    continue                          # complex->int discards imag (warning path)
                emit("ndarray.astype", {"dtype": target}, pair, ln, s, f"to={target}",
                     lambda v=view, t=target: v.astype(t))
            emit("ndarray.copy", {}, pair, ln, s, "c", lambda v=view: v.copy())
            emit("ndarray.ravel", {}, pair, ln, s, "c", lambda v=view: v.ravel())
            for order in ("C", "F"):
                emit("ndarray.flatten", {"order": order}, pair, ln, s, f"order={order}",
                     lambda v=view, o=order: v.flatten(order=o))
            emit("ndarray.reshape", {"shape": [-1]}, pair, ln, s, "flat",
                 lambda v=view: v.reshape(-1))
            if view.size >= 2 and view.size % 2 == 0:
                emit("ndarray.reshape", {"shape": [2, -1]}, pair, ln, s, "2xhalf",
                     lambda v=view: v.reshape(2, -1))
            emit("ndarray.squeeze", {}, pair, ln, s, "all", lambda v=view: v.squeeze())
            emit("ndarray.transpose", {}, pair, ln, s, "rev", lambda v=view: v.transpose())
            if view.ndim >= 2:
                emit("ndarray.swapaxes", {"a1": 0, "a2": 1}, pair, ln, s, "01",
                     lambda v=view: v.swapaxes(0, 1))
                perm = list(range(view.ndim))[::-1]
                emit("ndarray.transpose", {"axes": perm}, pair, ln, s, "perm",
                     lambda v=view, p=perm: v.transpose(p))
                for off in (0, 1):
                    emit("ndarray.diagonal", {"offset": off}, pair, ln, s, f"off={off}",
                         lambda v=view, o=off: np.asarray(v.diagonal(offset=o)))
                    emit("ndarray.trace", {"offset": off}, pair, ln, s, f"off={off}",
                         lambda v=view, o=off: np.asarray(v.trace(offset=o)))

            # elementwise / scan / selection instance forms
            emit("ndarray.conj", {}, pair, ln, s, "c", lambda v=view: v.conj())
            for ax in (None, 0):
                if ax is not None and view.ndim == 0:
                    continue
                emit("ndarray.cumsum", {"axis": ax}, pair, ln, s, f"axis={ax}",
                     lambda v=view, ax=ax: v.cumsum(axis=ax))
                emit("ndarray.cumprod", {"axis": ax}, pair, ln, s, f"axis={ax}",
                     lambda v=view, ax=ax: v.cumprod(axis=ax))
            if s not in ("complex128",):
                emit("ndarray.clip", {"lo": 0, "hi": 2}, pair, ln, s, "0..2",
                     lambda v=view: v.clip(0, 2))
            dec_ok = s not in ("bool",)               # bool round raises at any decimals in NumPy? dec=0 ok
            if dec_ok:
                for dec in (0, 1):
                    emit("ndarray.round", {"decimals": dec}, pair, ln, s, f"dec={dec}",
                         lambda v=view, d=dec: v.round(d))
            emit("ndarray.repeat", {"repeats": 2}, pair, ln, s, "r2",
                 lambda v=view: v.repeat(2))
            if view.ndim >= 1 and view.size > 0:
                cond = np.tile(np.array([True, False, True], dtype=bool),
                               (view.shape[0] + 2) // 3)[:view.shape[0]].copy()
                emit("ndarray.compress", {"axis": 0}, [(base, view), (cond, cond)], ln, s, "ax0",
                     lambda v=view, c=cond: v.compress(c, axis=0))
                idx = np.array([0, int(view.shape[0]) - 1], dtype=np.int64)
                emit("ndarray.take", {"axis": 0}, [(base, view), (idx, idx)], ln, s, "ends",
                     lambda v=view, i=idx: v.take(i, axis=0))

            # reinterpret family
            emit("ndarray.view", {}, pair, ln, s, "same", lambda v=view: v.view())
            reinterp = {"int32": "float32", "int64": "float64", "float32": "int32",
                        "float64": "int64", "float16": "uint16", "uint8": "bool",
                        "int16": "uint16", "uint16": "int16", "uint32": "float32",
                        "uint64": "float64"}
            if s in reinterp:
                emit("ndarray.view", {"dtype": reinterp[s]}, pair, ln, s, f"as={reinterp[s]}",
                     lambda v=view, t=reinterp[s]: v.view(t))
            emit("ndarray.byteswap", {}, pair, ln, s, "c", lambda v=view: v.byteswap())
            if s == "complex128":
                for off in (0, 8):
                    emit("ndarray.getfield", {"dtype": "float64", "offset": off}, pair, ln, s,
                         f"off={off}", lambda v=view, o=off: v.getfield(np.float64, o))

            # scalar-kind: item() (size-1 only -> value; otherwise NumPy raises -> error cell),
            # item(k), len(a) (0-d raises), and the D4 property scalars.
            if s != "uint64":
                emit("ndarray.item", {}, pair, ln, s, "flat", lambda v=view: v.item(),
                     kind="scalar")
                if view.size > 0 and view.ndim >= 1:
                    for k in (0, int(view.size) - 1):
                        emit("ndarray.item", {"index": k}, pair, ln, s, f"k={k}",
                             lambda v=view, k=k: v.item(k), kind="scalar")
            emit("ndarray.__len__", {}, pair, ln, s, "len", lambda v=view: len(view),
                 kind="scalar")
            emit("ndarray.nbytes", {}, pair, ln, s, "p", lambda v=view: v.nbytes, kind="scalar")
            emit("ndarray.itemsize", {}, pair, ln, s, "p", lambda v=view: v.itemsize, kind="scalar")
            emit("ndarray.ndim", {}, pair, ln, s, "p", lambda v=view: v.ndim, kind="scalar")
            emit("ndarray.size", {}, pair, ln, s, "p", lambda v=view: v.size, kind="scalar")
            emit("ndarray.strides", {}, pair, ln, s, "p",
                 lambda v=view: np.asarray(v.strides, dtype=np.int64))

            # D4 array-kind property reads
            emit("ndarray.T", {}, pair, ln, s, "p", lambda v=view: v.T)
            emit("ndarray.mT", {}, pair, ln, s, "p", lambda v=view: v.mT)
            emit("ndarray.real", {}, pair, ln, s, "p", lambda v=view: np.asarray(v.real))
            emit("ndarray.imag", {}, pair, ln, s, "p", lambda v=view: np.asarray(v.imag))
            emit("ndarray.flat", {}, pair, ln, s, "p", lambda v=view: np.asarray(v.flat))

            # tobytes: the raw C-order (and F-order) bytes as a uint8 vector.
            for order in (("C",) if view.ndim < 2 else ("C", "F")):
                emit("ndarray.tobytes", {"order": order}, pair, ln, s, f"order={order}",
                     lambda v=view, o=order: np.frombuffer(v.tobytes(order=o), dtype=np.uint8))

            # tuple-kind: nonzero (arity == ndim, asserted by CompareTuple).
            if view.ndim >= 1:
                emit("ndarray.nonzero", {}, pair, ln, s, "t",
                     lambda v=view: _tuple_expected(list(v.nonzero())), kind="tuple")

            # ---- in-place mutators (§D3): fresh (base, view) per job, operand described
            # BEFORE the call, expected = [post-call view, post-call WHOLE base buffer].
            def inplace(op, params, tag, mutate, extra_pairs=(), dtname=s, layout=ln):
                b2, v2 = LAYOUTS[layout](dt)
                pairs = [(b2, v2)] + list(extra_pairs)
                emit(op, params, pairs, layout, dtname, tag,
                     lambda: (mutate(v2), _tuple_expected([v2, b2.ravel()]))[1], kind="tuple")

            for ax in (-1, 0):
                if view.ndim == 0 and ax == 0:
                    continue
                inplace("ndarray.sort", {"axis": ax}, f"axis={ax}",
                        lambda v, ax=ax: v.sort(axis=ax))
            fillv = True if s == "bool" else 7
            inplace("ndarray.fill", {"value": (True if s == "bool" else 7)}, "v7",
                    lambda v, fv=fillv: v.fill(fv))
            if view.ndim >= 1 and view.size >= 2:
                pidx = np.array([0, int(view.size) - 1], dtype=np.int64)
                pval = np.array([3, 1], dtype=dt) if s != "bool" else np.array([True, False])
                inplace("ndarray.put", {"mode": "raise"}, "ends",
                        lambda v, i=pidx, w=pval: v.put(i, w, mode="raise"),
                        extra_pairs=[(pidx, pidx), (pval, pval)])

    # ---- dedicated small-exact jobs that need custom operands ----
    # uint8 joins so dot/choose clear the strength gate's >=4-cases floor (and it exercises the
    # unsigned lanes of the small-exact product/gather paths).
    for s in dedicated:
        dt = np.dtype(s)
        A = np.arange(6, dtype=np.float64).reshape(2, 3)
        B = (np.arange(6, dtype=np.float64) + 1).reshape(3, 2)
        if s == "complex128":
            A = (A + 1j * (A + 1)).astype(dt)
            B = (B - 1j * B).astype(dt)
        else:
            A = A.astype(dt)
            B = B.astype(dt)
        emit("ndarray.dot", {}, [(A, A), (B, B)], "mm_2x3_3x2", s, "d",
             lambda A=A, B=B: A.dot(B))

        sorted_a = np.sort(_fill(8, dt)) if s != "complex128" else np.sort(_fill(8, dt))
        probes = sorted_a[[0, 3, 7]].copy()
        for side in ("left", "right"):
            emit("ndarray.searchsorted", {"side": side}, [(sorted_a, sorted_a), (probes, probes)],
                 "sorted_1d", s, side,
                 lambda a=sorted_a, v=probes, sd=side: a.searchsorted(v, side=sd))

        idx = np.array([0, 1, 1, 0, 1, 0], dtype=np.int64)
        c0 = _fill(6, dt)
        c1 = _fill(6, dt)[::-1].copy()
        emit("ndarray.choose", {}, [(idx, idx), (c0, c0), (c1, c1)], "choose_1d", s, "2c",
             lambda i=idx, a=c0, b=c1: i.choose((a, b)))

    # resize: own-data contiguous only (a non-owning view raises on the NumPy side, and the
    # reconstructed NumSharp operand always owns its buffer — an asymmetric cell). The operand
    # is built directly so base IS the array being resized; numpy needs refcheck=False because
    # the generator's locals hold references. Result kind is ARRAY: the mutated array IS the
    # whole observable state after resize (the old base buffer no longer exists).
    for s in resize_dts:
        dt = np.dtype(s)
        for newshape in ([3], [12], [2, 4]):
            own = _fill(8, dt)
            emit("ndarray.resize", {"shape": newshape}, [(own, own)], "own_1d", s,
                 f"to={newshape}",
                 lambda v=own, ns=newshape: (v.resize(tuple(ns), refcheck=False), v)[1])

    if skipped:
        print(f"  (skipped {skipped})")
    print(f"  ({len(cases)} instance cases, "
          f"{sum(1 for c in cases if c.get('expects_throw'))} raising)")
    return cases


# ---- np.emath — the scimath module (coverage plan §A2/E5) ----------------------------------
#
# Promoted into the differential corpus rather than left sibling-owned: emath's whole point is
# the real->complex promotion DECISION (any(x<0) / |x|>1), which is exactly the kind of
# branchy, dtype-dependent contract the byte corpus gates best. Dtypes are restricted to the
# lanes whose NumPy promotion lands on complex128/float64 — int8/16/uint16/float32/float16
# promote to complex64 (numpy/lib/_scimath_impl._tocomplex), which NumSharp cannot represent
# (issue #569); those lanes stay on the np.emath.Test.cs sibling suite.
EMATH_UNARY = ["sqrt", "log", "log2", "log10", "arccos", "arcsin", "arctanh"]
EMATH_DTYPES = ["float64", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                "bool", "complex128"]
EMATH_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "f_contiguous_2d", "negstride_1d",
                 "strided_step2_1d", "scalar_0d", "one_element_1d", "empty_2d"]


def gen_emath():
    cases = []
    n = 0
    skipped = 0
    for ln in EMATH_LAYOUTS:
        for s in EMATH_DTYPES:
            base, view = LAYOUTS[ln](np.dtype(s))
            operand = describe(base, view)
            for opname in EMATH_UNARY:
                try:
                    with np.errstate(all="ignore"):
                        r = np.asarray(getattr(np.emath, opname)(view))
                except Exception:
                    skipped += 1
                    continue
                if r.dtype.name not in ("float16", "float32", "float64", "complex128",
                                        "int32", "int64", "bool", "uint8"):
                    skipped += 1                       # complex64 lane — sibling-owned (#569)
                    continue
                cases.append(_case(f"emath.{opname}", {}, [operand], _arr_expected(r), ln,
                                   "emath", cid=f"emath.{opname}/{ln}/{s}/{n}"))
                n += 1

    # logn(n, x) and power(x, p): pair operands (n/p as real arrays; negatives force complex).
    # Four dtypes so each op clears OracleCoverageStrengthTests' >=4-cases floor.
    for s in ("float64", "int32", "int64", "complex128"):
        dt = np.dtype(s)
        x = _fill(8, dt)
        nbase = np.array([2.0, 10.0, 0.5, 3.0, 2.0, 8.0, 4.0, 9.0], dtype=np.float64)
        for opname, second in (("logn", nbase), ("power", np.array([2, 3, 0, 1, 2, 3, 1, 2],
                                                                   dtype=np.int64))):
            try:
                with np.errstate(all="ignore"):
                    r = (np.asarray(np.emath.logn(second, x)) if opname == "logn"
                         else np.asarray(np.emath.power(x, second)))
            except Exception:
                skipped += 1
                continue
            if r.dtype.name not in ("float16", "float32", "float64", "complex128"):
                skipped += 1
                continue
            operands = ([describe(second, second), describe(np.ascontiguousarray(x), x)]
                        if opname == "logn"
                        else [describe(np.ascontiguousarray(x), x), describe(second, second)])
            cases.append(_case(f"emath.{opname}", {}, operands, _arr_expected(r),
                               "pp_contig", "emath", cid=f"emath.{opname}/pp/{s}/{n}"))
            n += 1

    if skipped:
        print(f"  (skipped {skipped} cells — NumPy raised or complex64 lane)")
    return cases


# ---- numpy.polynomial evaluation family (plan docs/plans/numpy-polynomial.md U3) -------------
#
# {p}val / {p}val2d / {p}val3d / {p}grid2d / {p}grid3d for the six bases, plus {p}valnd (NumPy's
# development-branch addition, oracle'd through the identical private polyutils._valnd).
# Op keys are MODULE-QUALIFIED ("chebyshev.chebval") because the package reuses the legacy
# np.polyval/polyder/... names with the OPPOSITE coefficient order (plan D5): a bare "polyval" key
# would credit the new facade with the legacy corpus, or the legacy function with this one.
#
# Every case records the x forms NumPy distinguishes, in params["xs"] (one entry per x/y/z/pts
# position, in order; the coefficients are always the LAST operand):
#   "a"                          -> the next operand (an ndarray; 0-d arrays included — STRONG)
#   {"kind": "int",  "int": "…"} -> a Python int (decimal text; may exceed 64 bits)
#   {"kind": "float","bits": …}  -> a Python float (its IEEE-754 bit pattern as 0x… text)
#   {"kind": "complex","re": …,"im": …}  -> a Python complex (component bit patterns)
#   {"kind": "bool", "bool": b}  -> a Python bool
# A Python scalar is WEAK (NEP 50): chebval(2.0, float32_c) is float32, and CPython computes the
# x-only subexpressions (2*x, x*0, 1 - x, (2*nd - 1) - x) before NumPy sees them.
# Cells whose NumPy result is complex64 (a Python complex against float16/float32 coefficients)
# are skipped: NumSharp has one complex width (#569).

POLY_MODULES = [
    ("polynomial", "poly"), ("chebyshev", "cheb"), ("legendre", "leg"),
    ("laguerre", "lag"), ("hermite", "herm"), ("hermite_e", "herme"),
]

_POLY_MODERATE = [0.5, -1.25, 2.0, -0.75, 1.5, -2.0, 0.25, 1.0, -0.5, 0.0, -0.0, 3.0,
                  1e-3, -1e-3, 0.9, -1.9, 1.25, -3.0, 0.125, 2.5, -0.375, 1.75, -1.0, 0.625]
_POLY_INTS = [0, 1, 2, 3, -1, -2, 5, 7, -4, 9, 11, -6, 4, 6, -3, 8, 10, -7, 12, 13, -9, 14, 15, -11]


def _poly_module(name):
    import numpy.polynomial as npp
    return getattr(npp, name)


def _f64_bits(v):
    import struct
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", v))[0]


def _poly_fill(n, dt, seed=0):
    """Deterministic MODERATE values (so high-degree series stay finite and every bit is informative)."""
    dt = np.dtype(dt)
    idx = [(i * 7 + seed) % len(_POLY_MODERATE) for i in range(n)]
    if dt.kind == "f":
        return np.array([_POLY_MODERATE[i] for i in idx], dtype=np.float64).astype(dt)
    if dt.kind == "c":
        re = np.array([_POLY_MODERATE[i] for i in idx], dtype=np.float64)
        im = np.array([_POLY_MODERATE[(i + 5) % len(_POLY_MODERATE)] for i in idx], dtype=np.float64)
        return (re + 1j * im).astype(dt)
    if dt.kind == "b":
        return np.array([(i + seed) % 3 != 1 for i in range(n)], dtype=bool)
    return np.array([_POLY_INTS[i] for i in idx], dtype=np.int64).astype(dt)


def _poly_coef(shape, dt, seed=1):
    n = int(np.prod(shape)) if len(shape) else 1
    return np.ascontiguousarray(_poly_fill(n, dt, seed).reshape(shape))


def _weak_spec(v):
    if isinstance(v, bool):
        return {"kind": "bool", "bool": v}
    if isinstance(v, int):
        return {"kind": "int", "int": str(v)}
    if isinstance(v, float):
        return {"kind": "float", "bits": _f64_bits(v)}
    if isinstance(v, complex):
        return {"kind": "complex", "re": _f64_bits(v.real), "im": _f64_bits(v.imag)}
    raise TypeError(type(v))


def _poly_exc(e):
    # The C# harness trims its own message, so trailing blanks (NumPy's broadcast error ends in a
    # space) are trimmed here too; everything else is NumPy's text verbatim.
    return {"type": type(e).__name__, "text": str(e).strip()}


def gen_polyeval():
    import numpy.polynomial.polyutils as pu
    cases = []
    n = 0
    skipped = 0
    ok_dtypes = set(ALL_DTYPES)
    A = _PSArr   # an ndarray argument inside a Python-typed spec (sections J and K)

    def emit(op, xs, operands, call, layout, cid, extra=None):
        nonlocal n, skipped
        params = {"xs": xs}
        if extra:
            params.update(extra)
        try:
            with np.errstate(all="ignore"):
                r = np.asarray(call())
        except Exception as e:
            cases.append({"id": f"{cid}/{n}", "op": op, "params": params, "operands": operands,
                          "expected": {}, "expects_throw": True, "error": _poly_exc(e),
                          "layout": layout, "valueclass": "error"})
            n += 1
            return
        if r.dtype.name not in ok_dtypes:
            skipped += 1          # complex64 lane (#569)
            return
        cases.append(_case(op, params, operands, _arr_expected(r), layout, "polyeval", cid=f"{cid}/{n}"))
        n += 1

    c_dtypes = ["float64", "float32", "float16", "complex128", "int64", "bool", "int8", "uint16"]
    counts = [1, 2, 3, 4, 7, 12]

    for modname, p in POLY_MODULES:
        mod = _poly_module(modname)
        val = getattr(mod, p + "val")
        op = f"{modname}.{p}val"

        # (A) dtype matrix: every x dtype x the coefficient dtypes x every coefficient-count class.
        for xs in ALL_DTYPES:
            xv = _poly_fill(9, xs)
            xd = describe(xv, xv)
            for cs in c_dtypes:
                for nc in counts:
                    cv = _poly_coef((nc,), cs)
                    emit(op, ["a"], [xd, describe(cv, cv)], lambda: val(xv, cv), "c_contiguous_1d",
                         f"{op}/dt/{xs}/{cs}/{nc}")

        # (B) x memory layouts (values overwritten with moderate ones; the layout is the point).
        for ln in sorted(LAYOUTS):
            for xs, cs, nc in (("float64", "float64", 5), ("float32", "float32", 6),
                               ("float16", "float16", 4), ("complex128", "float64", 7),
                               ("int32", "float64", 12)):
                base, view = LAYOUTS[ln](np.dtype(xs))
                base[...] = _poly_fill(base.size, xs).reshape(base.shape)
                cv = _poly_coef((nc,), cs)
                emit(op, ["a"], [describe(base, view), describe(cv, cv)], lambda: val(view, cv), ln,
                     f"{op}/lay/{ln}/{xs}/{cs}")

        # (C) N-D coefficients: tensor=True (every series at every point) and tensor=False (x
        # broadcast over the series), coefficient layouts (transposed / reversed / broadcast).
        for cs in ("float64", "float32", "complex128", "float16"):
            for nc in (1, 2, 3, 6):
                for cshape_tail, xshape, tensor in (((2,), (4,), True), ((3, 2), (4,), True),
                                                    ((2,), (2, 3), True), ((3,), (3,), False),
                                                    ((2, 3), (3,), False), ((2, 3), (2, 3), False),
                                                    ((4,), (1,), False), ((1,), (5,), False)):
                    cv = _poly_coef((nc,) + cshape_tail, cs)
                    xv = _poly_fill(int(np.prod(xshape)), "float64", seed=3).reshape(xshape)
                    emit(op, ["a"], [describe(xv, xv), describe(cv, cv)],
                         lambda: val(xv, cv, tensor=tensor), f"nd_c_{'tensor' if tensor else 'bcast'}",
                         f"{op}/nd/{cs}/{nc}/{cshape_tail}/{xshape}/{tensor}", {"tensor": tensor})
            # coefficient layouts: axis 0 strided (transposed), reversed, broadcast along the columns
            cb = _poly_coef((3, 5), cs)
            ct = cb.T                                   # (5, 3): series stride = 1 element
            cr = _poly_coef((5, 3), cs)[::-1]           # negative series stride
            csrc = _poly_coef((5, 1), cs)
            cbc = np.broadcast_to(csrc, (5, 3))
            xv = _poly_fill(3, "float64", seed=5)
            for cn, (cbase, cview) in (("c_transposed", (cb, ct)), ("c_reversed", (cr.base, cr)),
                                       ("c_broadcast", (csrc, cbc))):
                for tensor in (True, False):
                    emit(op, ["a"], [describe(xv, xv), describe(cbase, cview)],
                         lambda: val(xv, cview, tensor=tensor), cn, f"{op}/clay/{cs}/{cn}/{tensor}", {"tensor": tensor})
        # tensor=False broadcast mismatch: NumPy's ValueError (column shape first)
        cv = _poly_coef((3, 2), "float64")
        xv = _poly_fill(3, "float64")
        emit(op, ["a"], [describe(xv, xv), describe(cv, cv)], lambda: val(xv, cv, tensor=False),
             "nd_c_bcast", f"{op}/err/bcast", {"tensor": False})

        # (D) Python-scalar x (weak) against 1-D and 2-D series.
        weak_values = [0, 1, -3, 7, 2 ** 40, 2 ** 64 - 1, True, False,
                       0.5, -0.0, 2.25, -1.75, float("inf"), float("-inf"), float("nan"), 1e300, 5e-324,
                       complex(1, 2), complex(-0.5, 0.25), complex(float("inf"), 1.0), complex(0.0, -0.0)]
        for wv in weak_values:
            for cs in ("float64", "float32", "float16", "complex128", "int64", "bool"):
                for cshape in ((1,), (2,), (3,), (6,), (4, 3)):
                    cv = _poly_coef(cshape, cs)
                    emit(op, [_weak_spec(wv)], [describe(cv, cv)], lambda: val(wv, cv), "weak_scalar",
                         f"{op}/weak/{type(wv).__name__}/{cs}/{cshape}")

        # (J) Python-sequence x — NumPy's `if isinstance(x, (tuple, list)): x = np.asarray(x)`, nested to any depth,
        #     dtype discovered over every item (the C# replay: object[] / ValueTuple, NDPolySequence) — and Python ints
        #     past int64 (weak: CPython's int arithmetic on the x-only terms, then NumPy's conversion — inf, nan, or
        #     OverflowError past float64; the C# replay: BigInteger).
        seq_x = [("tuple", (0.5, 1.5, -2.0)), ("list_int", [1, 2, 3]), ("nested", [[0.5, 1.0], [2.0, 3.0]]),
                 ("nested_tuple", ((1, 2), (3, 4))), ("mixed", [(0.5, 1), [2, 3.5]]), ("empty", []),
                 ("complex", [1j, 0.5]), ("bool", [True, False]), ("ragged", [[1], [1, 2]]),
                 ("f16_item", [_PSArr(np.array(0.5, np.float16)), 2.0])]
        big_x = [("p70", 2 ** 70), ("n70", -(2 ** 70)), ("p200", 2 ** 200), ("p1030", 2 ** 1030), ("n1030", -(2 ** 1030))]
        for cs in ("float64", "float32", "float16", "complex128"):
            for cshape in ((3,), (4, 2)):
                cv = _poly_coef(cshape, cs, seed=31)
                for xname, xv in seq_x:
                    xops = []
                    spec = _ps_enc(xv, xops)
                    emit(op, [spec], xops + [describe(cv, cv)], lambda: val(_ps_py(xv), cv), "python_seq_x",
                         f"{op}/seqx/{xname}/{cs}/{cshape}")
                for bname, bv in big_x:
                    emit(op, [_weak_spec(bv)], [describe(cv, cv)], lambda: val(bv, cv), "big_int_x",
                         f"{op}/bigx/{bname}/{cs}/{cshape}")

        # (E) 0-d array x (STRONG), including complex 0-d x, whose ops mix ufunc and scalar math.
        for xs in ("float64", "float32", "float16", "complex128", "int32", "bool"):
            x0 = np.array(_poly_fill(1, xs, seed=2)[0])
            for cs in ("float64", "float32", "complex128"):
                for cshape in ((1,), (2,), (3,), (5,), (9,), (3, 2)):
                    cv = _poly_coef(cshape, cs, seed=4)
                    emit(op, ["a"], [describe(x0, x0), describe(cv, cv)], lambda: val(x0, cv), "scalar_0d",
                         f"{op}/x0d/{xs}/{cs}/{cshape}")

        # (F) special values in the coefficients (NaN/inf/-0) at a few x dtypes.
        cspecial = np.array([1.0, float("nan"), -0.0, float("inf"), 2.0, -float("inf"), 0.5], dtype=np.float64)
        for xs in ("float64", "float32", "float16", "complex128"):
            xv = _poly_fill(6, xs, seed=6)
            for nc in (1, 2, 3, 7):
                cv = np.ascontiguousarray(cspecial[:nc])
                emit(op, ["a"], [describe(xv, xv), describe(cv, cv)], lambda: val(xv, cv), "c_contiguous_1d",
                     f"{op}/cspec/{xs}/{nc}")

        # (H) vector lanes: every x dtype x every inexact series dtype at 45 points — enough for the U-chain
        # stage, the 1-chain vector stage and the scalar tail at every lane width (45 = 32+8+5 for 8 lanes,
        # 32+12+1 for 4, 40+4+1 for 2) — as a 1-D series and one series per point (tensor=False). The (A)
        # matrix's 9 points never reach the U-chain stage of a float64/float32 loop, and its per-point
        # shapes (C) are narrower than one vector: these cells are the byte gate of the mixed-dtype lane kinds.
        for xs in ALL_DTYPES:
            xv = _poly_fill(45, xs, seed=41)
            xd = describe(xv, xv)
            for cs in ("float16", "float32", "float64", "complex128"):
                cv = _poly_coef((6,), cs, seed=42)
                emit(op, ["a"], [xd, describe(cv, cv)], lambda: val(xv, cv), "c_contiguous_1d",
                     f"{op}/vl/{xs}/{cs}/1d")
                cp = _poly_coef((6, 45), cs, seed=43)
                emit(op, ["a"], [xd, describe(cp, cp)], lambda: val(xv, cp, tensor=False), "nd_c_bcast",
                     f"{op}/vl/{xs}/{cs}/pp", {"tensor": False})
        # ... at the dtype extremes: integer x at its bounds, so 2*x / (2*nd - 1) - x / 1 - x WRAP in x's
        # dtype lane for lane (the moderate values above never overflow a signed lane), and float x with
        # the specials (NaN, +-inf, +-0, subnormals, values that overflow float16/float32 mid-recurrence).
        for xs in ALL_DTYPES:
            dt = np.dtype(xs)
            if dt.kind in "iu":
                info = np.iinfo(dt)
                ext = [int(info.min), int(info.max), int(info.min) + 1, int(info.max) - 1, int(info.max) // 2 + 1,
                       int(info.min) // 2, 0, 1, 3]
                xv = np.array([ext[i % len(ext)] for i in range(45)], dtype=dt)
                series = ("float64", "float16")
            elif dt.kind in "fc":
                ext = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 5e-324, -1e-310, 1e300, -3e38,
                       65504.0, 1e5, 0.5, -1.25]
                re = np.array([ext[i % len(ext)] for i in range(45)])
                if dt.kind == "c":
                    xv = (re + 1j * np.array([ext[(i + 4) % len(ext)] for i in range(45)])).astype(dt)
                else:
                    with np.errstate(all="ignore"):
                        xv = re.astype(dt)
                series = ("float64", "complex128")
            else:
                continue
            xd = describe(xv, xv)
            for cs in series:
                cv = _poly_coef((6,), cs, seed=45)
                emit(op, ["a"], [xd, describe(cv, cv)], lambda: val(xv, cv), "c_contiguous_1d",
                     f"{op}/vlx/{xs}/{cs}/1d")
                cp = _poly_coef((6, 45), cs, seed=46)
                emit(op, ["a"], [xd, describe(cp, cp)], lambda: val(xv, cp, tensor=False), "nd_c_bcast",
                     f"{op}/vlx/{xs}/{cs}/pp", {"tensor": False})
        # ... and the same per-point series through a column-strided view: the kernel's scalar per-point
        # part, which must agree with the vector part above byte for byte.
        for xs, cs in (("float64", "float64"), ("complex128", "float16"), ("int8", "float32"),
                       ("float16", "complex128"), ("uint64", "float64"), ("bool", "float16")):
            xv = _poly_fill(45, xs, seed=41)
            cbase = _poly_coef((6, 90), cs, seed=44)
            cview = cbase[:, ::2]
            emit(op, ["a"], [describe(xv, xv), describe(cbase, cview)], lambda: val(xv, cview, tensor=False),
                 "nd_c_strided", f"{op}/vl/{xs}/{cs}/ppstrided", {"tensor": False})

        # (I) single-element broadcasts: an N-D series at a per-point x whose result has ONE element.
        # - When c[k] and x differ in ndim, NumPy's trivial ufunc loop refuses the operands.
        # - NpyIter then iterates the one element with every stride 0, so CDOUBLE_multiply runs its
        #   MSVC-contracted fallback loop, im = fma(ai, br, ar*bi), instead of simd_cmul,
        #   im = fma(ar, bi, ai*br).
        # - Once a value has absorbed x it has the deeper ndim. With the series shallower (tensor=False) only
        #   the first Clenshaw step differs; with it deeper, every step does.
        # - The forms only disagree on full-mantissa operands. This tier's moderate values are short dyadic
        #   rationals whose products are exact, so these cells draw seeded random values.
        # - There are three draws per cell, because a single product separates the two forms only ~1 time
        #   in 6.
        # - Real-x pairs are controls: a real-cast operand makes every form agree.
        # Measured: docs/plans/numpy-polynomial-review.md.
        urng = np.random.default_rng(20260927 + POLY_MODULES.index((modname, p)))

        def unit_fill(shape, dt):
            n = int(np.prod(shape)) if len(shape) else 1
            re = urng.uniform(-2.0, 2.0, n)
            if np.dtype(dt).kind == "c":
                re = re + 1j * urng.uniform(-2.0, 2.0, n)
            return np.ascontiguousarray(re.astype(dt).reshape(shape))

        for tail, xshape, tensor in (((1,), (1,), True), ((1, 1), (1,), True), ((1,), (1, 1), True),
                                     ((1, 1), (1,), False), ((1,), (1, 1), False), ((1,), (1, 1, 1), False),
                                     ((1,), (1,), False), ((1, 1), (1, 1), False)):
            for xs, cs in (("complex128", "complex128"), ("complex128", "float64"),
                           ("float64", "complex128"), ("float32", "complex128")):
                for nc in (1, 2, 3, 4, 5, 7):
                    for draw in range(3):
                        xv = unit_fill(xshape, xs)
                        cv = unit_fill((nc,) + tail, cs)
                        emit(op, ["a"], [describe(xv, xv), describe(cv, cv)], lambda: val(xv, cv, tensor=tensor),
                             f"unit_{'tensor' if tensor else 'bcast'}",
                             f"{op}/unit/{tail}/{xshape}/{tensor}/{xs}/{cs}/{nc}/{draw}", {"tensor": tensor})
        # ... and the near misses: the same ndim mismatches with MORE than one result element. NpyIter's
        # iteration then has a real stride, and NumPy runs simd_cmul. These gate the size-1 test on each side.
        for tail, xshape, tensor in (((1,), (5,), True), ((3,), (1,), True),
                                     ((1, 1), (4,), False), ((3,), (1, 1), False)):
            for xs, cs in (("complex128", "complex128"), ("complex128", "float64")):
                for nc in (2, 3, 5):
                    for draw in range(3):
                        xv = unit_fill(xshape, xs)
                        cv = unit_fill((nc,) + tail, cs)
                        emit(op, ["a"], [describe(xv, xv), describe(cv, cv)], lambda: val(xv, cv, tensor=tensor),
                             f"unit_near_{'tensor' if tensor else 'bcast'}",
                             f"{op}/unit_near/{tail}/{xshape}/{tensor}/{xs}/{cs}/{nc}/{draw}", {"tensor": tensor})
        # ... reached by the 2-D / 3-D compositions too: a unit trailing series axis makes a pass single-element.
        for kind in ("val2d", "val3d", "grid2d", "grid3d"):
            f = getattr(mod, p + kind)
            op2 = f"{modname}.{p}{kind}"
            dims = 2 if kind.endswith("2d") else 3
            for cshape in (((3, 1), (1, 3), (3, 4)) if dims == 2 else ((2, 1, 1), (2, 3, 1), (2, 1, 3))):
                for xs, cs in (("complex128", "complex128"), ("complex128", "float64")):
                    for draw in range(3):
                        pts = [unit_fill((1,), xs) for _ in range(dims)]
                        cv = unit_fill(cshape, cs)
                        emit(op2, ["a"] * dims, [describe(q, q) for q in pts] + [describe(cv, cv)],
                             lambda: f(*pts, cv), "unit_nd", f"{op2}/unit/{cshape}/{xs}/{cs}/{draw}")
        opn = f"{modname}.{p}valnd"
        for npts, cshape in ((1, (3, 1)), (2, (3, 1)), (3, (2, 1, 1)), (4, (2, 2, 1, 1))):
            for draw in range(3):
                pts = [unit_fill((1,), "complex128") for _ in range(npts)]
                cv = unit_fill(cshape, "complex128")
                emit(opn, ["a"] * npts, [describe(q, q) for q in pts] + [describe(cv, cv)],
                     lambda: pu._valnd(val, cv, *pts), "unit_nd", f"{opn}/unit/{npts}/{cshape}/{draw}")

        # (G) errors: empty series (IndexError at NumPy's first coefficient read)
        ce = np.zeros((0,), np.float64)
        xv = _poly_fill(3, "float64")
        emit(op, ["a"], [describe(xv, xv), describe(ce, ce)], lambda: val(xv, ce), "c_contiguous_1d", f"{op}/err/empty")
        ce2 = np.zeros((0, 3), np.float64)
        emit(op, ["a"], [describe(xv, xv), describe(ce2, ce2)], lambda: val(xv, ce2), "nd_c_tensor", f"{op}/err/empty2d")
        emit(op, [_weak_spec(2.0)], [describe(ce, ce)], lambda: val(2.0, ce), "weak_scalar", f"{op}/err/emptyweak")

        # ---- 2-D / 3-D / N-D ----
        for kind in ("val2d", "val3d", "grid2d", "grid3d"):
            f = getattr(mod, p + kind)
            op2 = f"{modname}.{p}{kind}"
            dims = 2 if kind.endswith("2d") else 3
            for cs in ("float64", "float32", "complex128", "float16", "int64"):
                cshape = (3, 4) if dims == 2 else (2, 3, 4)
                cv = _poly_coef(cshape, cs, seed=7)
                for xs in ("float64", "float32", "int32", "complex128"):
                    for pshape in ((5,), (2, 3)):
                        pts = [_poly_fill(int(np.prod(pshape)), xs, seed=10 + i).reshape(pshape) for i in range(dims)]
                        emit(op2, ["a"] * dims, [describe(q, q) for q in pts] + [describe(cv, cv)],
                             lambda: f(*pts, cv), "c_contiguous", f"{op2}/{cs}/{xs}/{pshape}")
            # trailing series axes on c
            cv = _poly_coef((3, 4, 2) if dims == 2 else (2, 3, 4, 2), "float64", seed=8)
            pts = [_poly_fill(4, "float64", seed=20 + i) for i in range(dims)]
            emit(op2, ["a"] * dims, [describe(q, q) for q in pts] + [describe(cv, cv)],
                 lambda: f(*pts, cv), "c_contiguous", f"{op2}/extraaxis")
            # scalar ordinates: STRONG 0-d arrays for val*d (np.asanyarray), WEAK for grid*d
            cv = _poly_coef((3, 4) if dims == 2 else (2, 3, 4), "float32", seed=9)
            svals = [0.5, 2, -1.25][:dims]
            if kind.startswith("val"):
                arrs = [np.asarray(v) for v in svals]
                emit(op2, ["a"] * dims, [describe(a, a) for a in arrs] + [describe(cv, cv)],
                     lambda: f(*svals, cv), "scalar_0d", f"{op2}/scalars")
            else:
                emit(op2, [_weak_spec(v) for v in svals], [describe(cv, cv)],
                     lambda: f(*svals, cv), "weak_scalar", f"{op2}/scalars")
                # mixed: weak first ordinate, array second
                yv = _poly_fill(3, "float64", seed=12)
                mixed = [svals[0], yv] + ([svals[2]] if dims == 3 else [])
                xs_spec = [_weak_spec(svals[0]), "a"] + ([_weak_spec(svals[2])] if dims == 3 else [])
                emit(op2, xs_spec, [describe(yv, yv), describe(cv, cv)],
                     lambda: f(*mixed, cv), "weak_scalar", f"{op2}/mixed")
            if kind.startswith("val"):
                # incompatible ordinate shapes
                pts = [_poly_fill(3, "float64"), _poly_fill(4, "float64")] + ([_poly_fill(3, "float64")] if dims == 3 else [])
                emit(op2, ["a"] * dims, [describe(q, q) for q in pts] + [describe(cv, cv)],
                     lambda: f(*pts, cv), "c_contiguous", f"{op2}/err/shapes")

        # {p}valnd (numpy main) == polyutils._valnd(val, c, *pts) on 2.4.2
        opn = f"{modname}.{p}valnd"
        for npts in (1, 2, 3, 4):
            cshape = tuple([3, 2, 2, 2][:npts])
            for cs, xs in (("float64", "float64"), ("float32", "int32"), ("complex128", "float32"),
                           ("float16", "float16"), ("int64", "complex128")):
                cv = _poly_coef(cshape, cs, seed=13)
                pts = [_poly_fill(4, xs, seed=30 + i) for i in range(npts)]
                emit(opn, ["a"] * npts, [describe(q, q) for q in pts] + [describe(cv, cv)],
                     lambda: pu._valnd(val, cv, *pts), "c_contiguous", f"{opn}/{npts}/{cs}/{xs}")
        cv = _poly_coef((3, 2, 2, 2), "float64")
        pts = [_poly_fill(4, "float64"), _poly_fill(4, "float64"), _poly_fill(3, "float64"), _poly_fill(4, "float64")]
        emit(opn, ["a"] * 4, [describe(q, q) for q in pts] + [describe(cv, cv)],
             lambda: pu._valnd(val, cv, *pts), "c_contiguous", f"{opn}/err/shapes")

        # (K) array_like c and ordinates. The C# replay turns a Python list into object[], a tuple into a ValueTuple and a
        #     NumPy scalar item into a 0-d array (params "c" holds the series spec when it is not an operand):
        #     * {p}val: c = np.array(c, ndmin=1) of any kind, BEFORE x (a ragged c's error wins over a ragged x);
        #     * {p}val2d / {p}val3d / valnd: the ordinates np.asanyarray'd first (a Python scalar is a STRONG 0-d array, a
        #       Python int an int64 one; a ragged ordinate raises before the shape check, the shape check before c);
        #     * {p}grid2d / {p}grid3d: the ordinates handed to {p}val as they are (a Python scalar stays WEAK), c converted by
        #       the first {p}val, before its x.
        #     Python ints past uint64 (NumPy's object arrays) are not used: NumSharp has no object dtype.
        c_forms = [("list", [1, 2.5, -0.5]), ("tuple", (1, 2.5, -0.5)), ("int_list", [1, 2, 3]),
                   ("bool_list", [True, False, True]), ("complex_list", [1, 2j, 0.5]), ("nested", [[1, 2], [3, 4], [0.5, -1]]),
                   ("nested_tuple", ((1, 2), (3, 4))), ("f16_items", [A(np.array(1.0, np.float16)), A(np.array(0.5, np.float16))]),
                   ("f16_py_mix", [A(np.array(1.0, np.float16)), 2.0]),
                   ("f32_rows", [A(np.array([1.0, 2.0], np.float32)), A(np.array([0.5, 0.25], np.float32))]),
                   ("scalar_int", 3), ("scalar_float", 2.5), ("scalar_bool", True), ("scalar_complex", 1.5j),
                   ("empty", []), ("ragged", [[1], [1, 2]]), ("u64", [2 ** 64 - 1, 1])]
        x_forms = [("weak", 0.5), ("weak_int", 2), ("arr", A(_poly_fill(3, "float64", seed=40))), ("list_x", [0.5, -1.5]),
                   ("ragged_x", [[[1], [1, 2]]]), ("f32_arr", A(_poly_fill(3, "float32", seed=41)))]
        # ragged_x's text ("after 2 dimensions ... (1, 2)") differs from a ragged c's ("after 1 dimensions ... (2,)"), so the
        # pair shows which one NumPy converts first.
        for cname, cform in c_forms:
            for xname, xform in x_forms:
                for tensor in (True, False):
                    # tensor only matters for an N-D series at an array x
                    if not tensor and not (xname in ("arr", "list_x", "f32_arr") and cname in ("nested", "nested_tuple", "f32_rows")):
                        continue
                    kops = []
                    xspec = _ps_enc(xform, kops)
                    extra = {"c": _ps_enc(cform, kops)}
                    if not tensor:
                        extra["tensor"] = False
                    emit(op, [xspec], kops, lambda: val(_ps_py(xform), _ps_py(cform), tensor=tensor), "array_like_c",
                         f"{op}/argc/{cname}/{xname}/{tensor}", extra=extra)
        for kind, dims in (("val2d", 2), ("val3d", 3), ("grid2d", 2), ("grid3d", 3)):
            f = getattr(mod, p + kind)
            op2 = f"{modname}.{p}{kind}"
            c64 = _poly_coef((3, 2) if dims == 2 else (2, 2, 2), "float64", seed=42)
            c32 = c64.astype(np.float32)
            ordinate_forms = [
                ("lists", [[0.5, 1.0], [1.0, 2.0], [-1.0, 0.25]]),
                ("tuples", [(0.5, 1.0), (1.0, 2.0), (-1.0, 0.25)]),
                ("scalars_float", [0.5, 1.5, -2.0]),
                ("scalars_int", [2, 3, -1]),
                ("scalars_bool", [True, False, True]),
                ("mixed", [A(_poly_fill(2, "float32", seed=43)), [1, 2], (0.5, 1.5)]),
                ("nested", [[[0.5], [1.0]], [[1.0], [2.0]], [[3.0], [4.0]]]),
                ("f16_items", [[A(np.array(0.5, np.float16))], [A(np.array(1.0, np.float16))], [A(np.array(2.0, np.float16))]]),
                ("empty", [[], [], []]),
                ("incompatible", [[0.5, 1.0], [1.0, 2.0, 3.0], [1.0, 2.0]]),
                ("ragged_first", [[[1], [1, 2]], [1.0, 2.0, 3.0], [1.0]]),
                ("ragged_second", [[1.0, 2.0], [[1], [1, 2]], [1.0, 2.0]]),
            ]
            for oname, ords in ordinate_forms:
                ords = ords[:dims]
                # the ragged c differs in shape from the ragged ordinates, so a case holding both shows the order
                for cname, cform in (("arr64", A(c64)), ("arr32", A(c32)), ("list", c64.tolist()), ("ragged", [[[1], [1, 2]]])):
                    kops = []
                    xs = [_ps_enc(o, kops) for o in ords]
                    extra = {"c": _ps_enc(cform, kops)}
                    emit(op2, xs, kops, lambda: f(*[_ps_py(o) for o in ords], _ps_py(cform)), "array_like_ordinates",
                         f"{op2}/argo/{oname}/{cname}", extra=extra)
        for oname, ords in (("lists", [[0.5, 1.0], (1.0, 2.0)]), ("scalars", [2, 0.5]), ("ragged", [[[1], [1, 2]], [1.0]]),
                            ("incompatible", [[0.5], [1.0, 2.0]])):
            c2 = _poly_coef((3, 2), "float64", seed=44)
            for cname, cform in (("arr", A(c2)), ("list", c2.tolist()), ("ragged", [[[1], [1, 2]]])):
                kops = []
                xs = [_ps_enc(o, kops) for o in ords]
                extra = {"c": _ps_enc(cform, kops)}
                emit(opn, xs, kops, lambda: pu._valnd(val, _ps_py(cform), *[_ps_py(o) for o in ords]), "array_like_ordinates",
                     f"{opn}/argo/{oname}/{cname}", extra=extra)

    # The recurrence's Python ints meeting an integer x: lagval's (2*nd - 1) - x overflows int8 at
    # 70 coefficients and uint8 at 130 (NumPy's OverflowError, raised before any shape check).
    for xs, nc in (("int8", 70), ("uint8", 130), ("int8", 65), ("int16", 70)):
        xv = _poly_fill(4, xs)
        cv = _poly_coef((nc,), "float64")
        emit("laguerre.lagval", ["a"], [describe(xv, xv), describe(cv, cv)],
             lambda: _poly_module("laguerre").lagval(xv, cv), "c_contiguous_1d", f"laguerre.lagval/overflow/{xs}/{nc}")

    # Char: NumSharp's uint16-like dtype has no NumPy analog; the uint16 x cells of the dtype matrix
    # are the bytes-exact oracle for it (the house char weave).
    cases += _relabel_dtype([c for c in cases if any(t in (c.get("id") or "") for t in ("/dt/uint16/", "/vl/uint16/", "/vlx/uint16/"))],
                            "uint16", "char")
    if skipped:
        print(f"  (skipped {skipped} complex64 cells — #569)")
    return cases


# ---- numpy.polynomial additive family + polyutils (plan docs/plans/numpy-polynomial.md U1) ---------------
#
# Op keys are module-qualified like the polyeval tier: "<basis module>.<name>" for the six bases
# (polynomial.polyadd, laguerre.lagline, hermite_e.hermedomain, …) and "polyutils.<name>" for the helpers.
# Arguments are recorded in params by NAME, each one of:
#   "a"                                   -> the next operand (an ndarray, any layout; a 0-d one is STRONG)
#   {"kind": "int"|"float"|"complex"|"bool", …}   -> a Python scalar (the _weak_spec form)
#   {"kind": "list"|"tuple", "items": [...]}       -> a Python list / tuple of the same (nested)
#   {"kind": "str", "str": s}             -> a Python str
# and operands are consumed in the order the C# side decodes the params (per op: c1,c2 / c,tol / off,scl /
# old,new / x,old,new / alist), items depth-first. The NumPy call receives exactly those objects, so a
# domain given as a list/tuple runs CPython arithmetic and one given as an array NumPy's scalar math.
# Results: arrays/scalars as (dtype, shape, bytes) — a Python float becomes 0-d float64 — and
# as_series/mapparms as tuples (arity asserted). complex64 results (a Python complex meeting a float16/
# float32 value) are skipped (#569), as are NumPy object arrays (NumSharp has no object dtype).


class _PSArr:
    """An ndarray argument together with the buffer it views (so its layout can be described)."""

    def __init__(self, base, view=None):
        self.base = base
        self.view = base if view is None else view


def _ps_enc(v, operands):
    if isinstance(v, _PSArr):
        operands.append(describe(v.base, v.view))
        return "a"
    if isinstance(v, list):
        return {"kind": "list", "items": [_ps_enc(i, operands) for i in v]}
    if isinstance(v, tuple):
        return {"kind": "tuple", "items": [_ps_enc(i, operands) for i in v]}
    if isinstance(v, str):
        return {"kind": "str", "str": v}
    if v is None:
        return {"kind": "none"}
    if isinstance(v, np.float16):
        # np.float16 is the one NumPy scalar with a C# spelling of its own (Half); every other NumPy scalar kind is
        # spelled as the Python value a C# primitive stands for.
        return {"kind": "npscalar", "dtype": "float16", "bits": "0x%04x" % int(np.array(v).view(np.uint16))}
    return _weak_spec(v)


def _ps_py(v):
    if isinstance(v, _PSArr):
        return v.view
    if isinstance(v, list):
        return [_ps_py(i) for i in v]
    if isinstance(v, tuple):
        return tuple(_ps_py(i) for i in v)
    return v


def _ps_series(n, dt, pattern="moderate", seed=0):
    """A 1-D coefficient array of dtype dt: moderate values, or one of the trim/special patterns."""
    dt = np.dtype(dt)
    base = _poly_fill(n, dt, seed)
    if pattern == "moderate":
        return base
    if pattern == "tz":                               # trailing zeros (and a -0.0 among them for floats)
        z = np.zeros(3, dt)
        if dt.kind in "fc":
            z[1] = -0.0
        return np.concatenate([base, z]).astype(dt)
    if pattern == "allzero":
        return np.zeros(n, dt)
    if pattern == "lastnz":                           # zeros except the first element
        a = np.zeros(n, dt)
        a[0] = base[0] if base[0] != 0 else 1
        return a
    if pattern == "special":
        if dt.kind == "f":
            vals = [float("nan"), 1.5, float("-inf"), -0.0, float("inf"), 0.0, 2.0]
        elif dt.kind == "c":
            vals = [complex(float("nan"), 1), 1.5 - 2j, complex(0, float("inf")), complex(-0.0, 0.0),
                    complex(float("inf"), float("nan")), 0j, 2 + 0j]
        elif dt.kind == "b":
            vals = [True, False, True, False]
        else:
            info = np.iinfo(dt)
            vals = [int(info.min), int(info.max), 0, 1, int(info.max) - 1, 0]
        with np.errstate(all="ignore"):
            return np.array([vals[i % len(vals)] for i in range(max(n, len(vals)))]).astype(dt)
    if pattern == "nantail":                          # a trailing NaN is NONZERO: never trimmed
        if dt.kind not in "fc":
            return base
        a = base.copy()
        a[-1] = np.nan
        return a
    raise ValueError(pattern)


def _ps_c64_exact(op, operands, a):
    """Whether a complex64 result is recorded (NumSharp carries it as complex128 with the SAME values): a 0-d result
    (mapparms' off/scl, mapdomain at a scalar x), a {p}line of scalar operands (np.array stacking complex64 scalars),
    and any polyutils.mapdomain result — its complex64 ARRAY loops (off + scl*x) run on NumSharp's float32 kernels
    with NumPy's product forms. NumSharp has no complex64 dtype (#569), so the replay compares the VALUES up-cast."""
    if a.dtype != np.complex64:
        return False
    if a.ndim == 0 or op == "polyutils.mapdomain":
        return True
    return op.endswith("line") and all(len(o["shape"]) == 0 for o in operands)


def _ps_layouts(arr):
    """(name, _PSArr) for the 1-D layouts of arr's values: contiguous, strided, reversed, offset slice,
    and — for a single value — stride-0 broadcast and 0-d."""
    n = arr.size
    out = [("c_contiguous_1d", _PSArr(arr.copy()))]
    s = np.zeros(2 * n, arr.dtype)
    s[::2] = arr
    out.append(("strided_1d", _PSArr(s, s[::2])))
    r = arr[::-1].copy()
    out.append(("reversed_1d", _PSArr(r, r[::-1])))
    o = np.concatenate([np.zeros(2, arr.dtype), arr, np.zeros(1, arr.dtype)])
    out.append(("offset_1d", _PSArr(o, o[2:2 + n])))
    return out


def gen_polyseries():
    import numpy.polynomial.polyutils as pu
    cases = []
    counter = [0]
    skipped = [0]
    ok_dtypes = set(ALL_DTYPES)

    def emit(op, args, call, layout, cid, kind="array"):
        operands = []
        params = {k: _ps_enc(v, operands) for k, v in args.items()}
        n = counter[0]
        counter[0] += 1
        try:
            with np.errstate(all="ignore"):
                r = call()
        except Exception as e:
            cases.append({"id": f"{cid}/{n}", "op": op, "params": params, "operands": operands,
                          "expected": {"kind": kind} if kind != "array" else {}, "expects_throw": True,
                          "error": _poly_exc(e), "layout": layout, "valueclass": "error"})
            return
        if kind == "tuple":
            arrs = [np.asarray(v) for v in r]
            if any(a.dtype.name not in ok_dtypes and not _ps_c64_exact(op, operands, a) for a in arrs):
                skipped[0] += 1
                return
            cases.append(_case(op, params, operands, _tuple_expected(arrs), layout, "polyseries", cid=f"{cid}/{n}"))
        else:
            a = np.asarray(r)
            if a.dtype.name not in ok_dtypes and not _ps_c64_exact(op, operands, a):
                skipped[0] += 1
                return
            cases.append(_case(op, params, operands, _arr_expected(a), layout, "polyseries", cid=f"{cid}/{n}"))

    A = _PSArr
    sub_dtypes = ["float64", "float32", "float16", "complex128", "int32", "uint64"]

    # ---------------- (A) {p}add / {p}sub ----------------
    for modname, p in POLY_MODULES:
        mod = _poly_module(modname)
        for opname in ("add", "sub"):
            f = getattr(mod, p + opname)
            op = f"{modname}.{p}{opname}"
            full = modname == "polynomial"
            dts = ALL_DTYPES if full else sub_dtypes
            for d1 in dts:
                for d2 in dts:
                    for n1, n2 in ((3, 5), (5, 3), (4, 4)):
                        c1 = _ps_series(n1, d1, seed=1)
                        c2 = _ps_series(n2, d2, seed=2)
                        emit(op, {"c1": A(c1), "c2": A(c2)}, lambda: f(c1, c2), "c_contiguous_1d",
                             f"{op}/dt/{d1}/{d2}/{n1}/{n2}")
            pat_dtypes = ["float64", "float32", "float16", "complex128", "int64", "uint8"] if full else ["float64", "complex128"]
            for d1 in pat_dtypes:
                for d2 in pat_dtypes:
                    for pat1, pat2 in (("tz", "moderate"), ("moderate", "tz"), ("allzero", "allzero"),
                                       ("special", "moderate"), ("moderate", "special"), ("nantail", "tz"),
                                       ("lastnz", "tz"), ("tz", "tz")):
                        for n1, n2 in ((3, 5), (5, 3), (4, 4)):
                            c1 = _ps_series(n1, d1, pat1, seed=3)
                            c2 = _ps_series(n2, d2, pat2, seed=4)
                            emit(op, {"c1": A(c1), "c2": A(c2)}, lambda: f(c1, c2), "c_contiguous_1d",
                                 f"{op}/pat/{d1}/{d2}/{pat1}/{pat2}/{n1}/{n2}")
            if full:
                # layouts of either operand
                for d in ("float64", "complex128", "float16", "int16"):
                    for n1, n2 in ((3, 5), (5, 3), (4, 4)):
                        v1 = _ps_series(n1, d, "tz" if n1 == 4 else "moderate", seed=5)
                        v2 = _ps_series(n2, d, seed=6)
                        for ln, a1 in _ps_layouts(v1):
                            for ln2, a2 in _ps_layouts(v2)[:2]:
                                emit(op, {"c1": a1, "c2": a2}, lambda: f(_ps_py(a1), _ps_py(a2)), f"{ln}+{ln2}",
                                     f"{op}/lay/{d}/{ln}/{ln2}/{n1}/{n2}")
                    # a 0-d array and a stride-0 broadcast are length-1 / length-n series
                    s0 = np.array(_ps_series(1, d, seed=7)[0])
                    emit(op, {"c1": A(s0), "c2": A(_ps_series(3, d, seed=8))},
                         lambda: f(s0, _ps_series(3, d, seed=8)), "scalar_0d", f"{op}/0d/{d}")
                    bsrc = _ps_series(1, d, seed=9)
                    bc = np.broadcast_to(bsrc, (4,))
                    emit(op, {"c1": A(bsrc, bc), "c2": A(_ps_series(2, d, seed=10))},
                         lambda: f(bc, _ps_series(2, d, seed=10)), "broadcast_1d", f"{op}/bcast/{d}")
            # Python scalars and lists
            py_args = [5, 2.5, -0.0, True, 1 + 2j, [1, 2, 0], [1.5, 2], [True, 2], [0, 0], [1, 2 ** 63],
                       [2 ** 64 - 1], [1 + 1j, 0j], 7]
            others = [A(_ps_series(3, "float64", seed=11)), A(_ps_series(4, "float32", seed=12)),
                      A(_ps_series(2, "float16", seed=13)), 3, [0], [0.5, -1.0, 0.0]]
            for a1 in py_args:
                for a2 in others:
                    emit(op, {"c1": a1, "c2": a2}, lambda: f(_ps_py(a1), _ps_py(a2)), "python",
                         f"{op}/py/{type(a1).__name__}/{type(a2).__name__}")
                    emit(op, {"c1": a2, "c2": a1}, lambda: f(_ps_py(a2), _ps_py(a1)), "python",
                         f"{op}/pyr/{type(a2).__name__}/{type(a1).__name__}")
            # errors: empty, 2-D, bool (no common type), order of the checks
            emp = np.zeros(0)
            two = np.zeros((2, 2))
            bl = np.array([True, False])
            for a1, a2, tag in ((A(emp), A(_ps_series(2, "float64")), "empty1"),
                                (A(_ps_series(2, "float64")), A(emp), "empty2"),
                                (A(two), A(_ps_series(2, "float64")), "2d"),
                                (A(two), A(emp), "2d_then_empty"),
                                (A(emp), A(two), "empty_then_2d"),
                                (A(bl), A(_ps_series(2, "float64")), "bool"),
                                (A(bl), A(emp), "bool_then_empty"),
                                ([[1, 2], [3, 4]], [1], "nested"),
                                ("ab", [1], "str")):
                emit(op, {"c1": a1, "c2": a2}, lambda: f(_ps_py(a1), _ps_py(a2)), "error", f"{op}/err/{tag}")

    # ---------------- (B) trimcoef / {p}trim ----------------
    tol_forms = [None, 0, 0.5, 1, 1e-3, 1.5, float("nan"), float("inf"), 2 ** 70, True, -0.0]
    for modname, p in [("polyutils", None)] + POLY_MODULES:
        if p is None:
            f = pu.trimcoef
            op = "polyutils.trimcoef"
            dts = ALL_DTYPES
            pats = ("moderate", "tz", "allzero", "special", "nantail", "lastnz")
        else:
            f = getattr(_poly_module(modname), p + "trim")
            op = f"{modname}.{p}trim"
            dts = ["float64", "float32", "complex128", "int64"]
            pats = ("tz", "allzero", "special")
        for d in dts:
            for pat in pats:
                c = _ps_series(5, d, pat, seed=14)
                for tol in tol_forms:
                    if tol is None:
                        emit(op, {"c": A(c)}, lambda: f(c), "c_contiguous_1d", f"{op}/dt/{d}/{pat}/default")
                    else:
                        emit(op, {"c": A(c), "tol": tol}, lambda: f(c, tol), "c_contiguous_1d",
                             f"{op}/dt/{d}/{pat}/{type(tol).__name__}/{tol!r}")
        if p is None:
            # strong (NumPy-scalar / array) tolerances: the comparison runs in the PROMOTED dtype
            for d in ("float16", "float32", "float64", "complex128"):
                c = np.array([1.0, 1.0009765625, 0.25, 1.0001], dtype=np.float64).astype(d)
                for tv, tdt in ((1.0001, "float64"), (1.0001, "float32"), (1.0001, "float16"), (0.25, "float16"),
                                (1.0, "int8"), (0.5, "complex128")):
                    t0 = np.array(tv, dtype=tdt)
                    emit(op, {"c": A(c), "tol": A(t0)}, lambda: f(c, t0), "strong_tol", f"{op}/strong/{d}/{tdt}/{tv}")
                    t1 = np.array([tv], dtype=tdt)
                    emit(op, {"c": A(c), "tol": A(t1)}, lambda: f(c, t1), "strong_tol_1d", f"{op}/strong1d/{d}/{tdt}/{tv}")
            # layouts of c
            for d in ("float64", "complex128", "float16"):
                for ln, ca in _ps_layouts(_ps_series(6, d, "tz", seed=15)):
                    emit(op, {"c": ca, "tol": 0.5}, lambda: f(_ps_py(ca), 0.5), ln, f"{op}/lay/{d}/{ln}")
            # Python-list series and a Python-scalar series
            for cv in ([0, 0, 3, 0, 5, 0, 0], [0.0, 1e-3, 0.0, 1e-5], [1 + 1j, 0j, 1e-4j], 5, 0, [True, False, True]):
                for tol in (0, 1e-3, 2):
                    emit(op, {"c": cv, "tol": tol}, lambda: f(cv, tol), "python", f"{op}/py/{cv!r}/{tol!r}")
            # errors, in NumPy's order (tol first)
            c = _ps_series(3, "float64")
            for tag, cv, tol in (("negtol", A(c), -1), ("negtol_float", A(c), -1e-300),
                                 ("negtol_strong", A(c), A(np.array(-0.5))),
                                 ("negtol_before_empty", A(np.zeros(0)), -1),
                                 ("complextol", A(c), 1j), ("tolarray2", A(c), A(np.array([1.0, 2.0]))),
                                 ("tolarray0", A(c), A(np.zeros(0))), ("toltoobig", A(c), 2 ** 1030),
                                 ("empty", A(np.zeros(0)), 0), ("2d", A(np.zeros((2, 2))), 0),
                                 ("bool", A(np.array([True, False])), 0)):
                emit(op, {"c": cv, "tol": tol}, lambda: f(_ps_py(cv), _ps_py(tol)), "error", f"{op}/err/{tag}")

    # ---------------- (C) polyutils.trimseq ----------------
    op = "polyutils.trimseq"
    for d in ALL_DTYPES:
        for pat in ("moderate", "tz", "allzero", "special", "nantail", "lastnz"):
            c = _ps_series(5, d, pat, seed=16)
            emit(op, {"seq": A(c)}, lambda: pu.trimseq(c), "c_contiguous_1d", f"{op}/dt/{d}/{pat}")
    for d in ("float64", "complex128", "float16"):
        for ln, ca in _ps_layouts(_ps_series(5, d, "tz", seed=17)):
            emit(op, {"seq": ca}, lambda: pu.trimseq(_ps_py(ca)), ln, f"{op}/lay/{d}/{ln}")
        col = _ps_series(4, d, "tz", seed=18).reshape(-1, 1)
        emit(op, {"seq": A(col)}, lambda: pu.trimseq(col), "column_2d", f"{op}/col/{d}")
        col3 = _ps_series(4, d, "tz", seed=18).reshape(-1, 1, 1)
        emit(op, {"seq": A(col3)}, lambda: pu.trimseq(col3), "column_3d", f"{op}/col3/{d}")
    for tag, s in (("empty", np.zeros(0)), ("rows2", np.array([[1.0, 2.0], [0.0, 0.0]])),
                   ("rows0", np.zeros((3, 0))), ("empty2d", np.zeros((0, 3))), ("zero_d", np.array(0.0)),
                   ("rows2_nonzero_last", np.array([[1.0, 2.0], [3.0, 4.0]]))):
        emit(op, {"seq": A(s)}, lambda: pu.trimseq(s), "error" if tag != "empty2d" else "empty",
             f"{op}/edge/{tag}")

    # ---------------- (D) polyutils.as_series ----------------
    op = "polyutils.as_series"
    for d1 in ALL_DTYPES:
        for d2 in ("float64", "float16", "float32", "complex128", "int8", "bool"):
            for trim in (True, False):
                c1 = _ps_series(4, d1, "tz", seed=19)
                c2 = _ps_series(3, d2, seed=20)
                emit(op, {"alist": [A(c1), A(c2)], "trim": trim}, lambda: pu.as_series([c1, c2], trim=trim),
                     "list_of_arrays", f"{op}/dt/{d1}/{d2}/{trim}", kind="tuple")
    for d in ("float64", "int64", "complex128", "float16", "bool"):
        m1 = _ps_series(4, d, "tz", seed=21)
        m2 = _ps_series(6, d, seed=22).reshape(2, 3)
        for trim in (True, False):
            emit(op, {"alist": A(m1), "trim": trim}, lambda: pu.as_series(m1, trim=trim), "ndarray_1d",
                 f"{op}/nd1/{d}/{trim}", kind="tuple")
            emit(op, {"alist": A(m2), "trim": trim}, lambda: pu.as_series(m2, trim=trim), "ndarray_2d",
                 f"{op}/nd2/{d}/{trim}", kind="tuple")
    ar8 = np.arange(8.0)
    for tag, al in (("pylists", [[1, 2, 0], [3.5]]), ("mixed", [2, [1.1, 0.0]]), ("scalars", [1, 2.5, 1j]),
                    ("single", [[1, 0, 0]]), ("empty_list", []), ("u64", [[2 ** 64 - 1]]),
                    ("big_mixed", [[1, 2 ** 63]]), ("bools_ints", [[True, 2]]), ("strided",
                    [A(ar8, ar8[::2]), [0.0]])):
        for trim in (True, False):
            emit(op, {"alist": al, "trim": trim}, lambda: pu.as_series(_ps_py(al), trim=trim), "python",
                 f"{op}/py/{tag}/{trim}", kind="tuple")
    for tag, al in (("empty_item", [[1], []]), ("nd_item", [[[1]], []]), ("bool", [[True, False]]),
                    ("bool_second", [[1.5], [True]]), ("zero_d", A(np.array(5.0))), ("scalar", 5),
                    ("str", "ab"), ("nd3", A(np.zeros((2, 2, 2)))), ("rows0", A(np.zeros((2, 0))))):
        emit(op, {"alist": al, "trim": True}, lambda: pu.as_series(_ps_py(al)), "error", f"{op}/err/{tag}", kind="tuple")

    # ---------------- (E) polyutils.getdomain ----------------
    op = "polyutils.getdomain"
    for d in ALL_DTYPES:
        for pat in ("moderate", "special", "allzero", "nantail"):
            x = _ps_series(6, d, pat, seed=23)
            emit(op, {"x": A(x)}, lambda: pu.getdomain(x), "c_contiguous_1d", f"{op}/dt/{d}/{pat}")
        if d in ("float64", "float32", "float16", "complex128", "int32"):
            for ln, xa in _ps_layouts(_ps_series(5, d, "special", seed=24)):
                emit(op, {"x": xa}, lambda: pu.getdomain(_ps_py(xa)), ln, f"{op}/lay/{d}/{ln}")
    for tag, xv in (("pz_nz", [0.0, -0.0]), ("nz_pz", [-0.0, 0.0]), ("nz_nz", [-0.0, -0.0]),
                    ("c_signs", [complex(0.0, -0.0), complex(-0.0, 0.0)]), ("c_nan", [1 + 1j, complex(float("nan"), 2)]),
                    ("one", [5.0]), ("pyint", 5), ("pylist", [3, -1, 2]), ("pycomplex", [1j, -2 + 0.5j])):
        emit(op, {"x": xv if not isinstance(xv, list) or True else xv}, lambda: pu.getdomain(xv), "python",
             f"{op}/py/{tag}")
        if isinstance(xv, list) and not isinstance(xv[0], int):
            arr = np.array(xv)
            emit(op, {"x": A(arr)}, lambda: pu.getdomain(arr), "c_contiguous_1d", f"{op}/arr/{tag}")
            rv = arr[::-1].copy()
            emit(op, {"x": A(rv, rv[::-1])}, lambda: pu.getdomain(rv[::-1]), "reversed_1d", f"{op}/rev/{tag}")
    for tag, xv in (("empty", A(np.zeros(0))), ("2d", A(np.zeros((2, 2)))), ("bool", A(np.array([True, False]))),
                    ("pybool", True), ("str", "ab")):
        emit(op, {"x": xv}, lambda: pu.getdomain(_ps_py(xv)), "error", f"{op}/err/{tag}")

    # ---------------- (F) polyutils.mapparms ----------------
    op = "polyutils.mapparms"
    py_domains = [(-1, 1), (0, 2), (1, -1), (1, 1), (0.5, 1.5), (-1.0, 1.0), (1e308, -1e308), (0, float("inf")),
                  (float("nan"), 1), (-1j, 1), (1j, 1j), (True, False), (0, 2 ** 64), (0, 10 ** 400), (10 ** 400, 0),
                  (-2 ** 53 - 1, 2 ** 53 + 3), (0.1, 0.7), (2, 3 + 4j)]
    for old in py_domains:
        for new in ((-1, 1), (0, 1), (0.25, 3.5), (1j, -1), (2 ** 70, 1)):
            for wrap in (tuple, list):
                o, nw = wrap(old), wrap(new)
                emit(op, {"old": o, "new": nw}, lambda: pu.mapparms(o, nw), "python",
                     f"{op}/py/{wrap.__name__}/{old!r}/{new!r}", kind="tuple")
    # The CPython int/float fast lane (NumSharp runs Python-real tuple domains on long/double): the edges where it
    # must hand over to exact integers (a long overflow in a difference or a product, ulong-range ints), the int ->
    # float conversions it must round like PyLong_AsDouble (2**62 + 513 is not a tie and not truncatable), the XOR
    # sign of a zero int quotient (0 / -5 == -0.0), and ints beyond 2**53 in a true division.
    lane_domains = [(2 ** 62, -2 ** 62), (-2 ** 63, 2 ** 63 - 1), (-5, 2 ** 63 - 1), (0, 2 ** 63 + 5),
                    (2 ** 31 + 1, 2 ** 32 + 7), (2 ** 62 + 513, 1.5), (2 ** 60 + 1, 0.5), (True, 0.5), (0, -5),
                    (1, -1), (-7, 3), (3, 10 ** 15 + 37)]
    for old in lane_domains:
        for new in ((0, 0), (3, 5), (-2 ** 62, 2 ** 61), (0.5, -0.0), (2 ** 53 + 1, 1)):
            emit(op, {"old": old, "new": new}, lambda: pu.mapparms(old, new), "python",
                 f"{op}/lane/{old!r}/{new!r}", kind="tuple")
    for tag, old, new in (("zerolen_mixed", (1, 1.0), (0, 1)), ("zerolen_negzero", (0.0, -0.0), (0, 1)),
                          ("zerolen_bool", (True, 1), (0, 1)), ("zerolen_big", (2 ** 62, 2 ** 62), (2 ** 62, 1)),
                          ("zerolen_float_new", (2.5, 2.5), (0.5, 1))):
        emit(op, {"old": old, "new": new}, lambda: pu.mapparms(_ps_py(old), _ps_py(new)), "error",
             f"{op}/err/{tag}", kind="tuple")
    for d in ALL_DTYPES:
        for vals in ((-1, 1), (0, 3), (2, 2), (1, 0)):
            with np.errstate(all="ignore"):
                old = np.array(vals).astype(d)
            for new in ((0, 1), (0.25, 3.5), (0, 300), (0, -1), (1j, 2), A(np.array([0.5, 2.0]))):
                emit(op, {"old": A(old), "new": new}, lambda: pu.mapparms(old, _ps_py(new)), "ndarray",
                     f"{op}/nd/{d}/{vals}/{new if not isinstance(new, _PSArr) else 'f64arr'}", kind="tuple")
            emit(op, {"old": (0, 2), "new": A(old)}, lambda: pu.mapparms((0, 2), old), "ndarray",
                 f"{op}/ndnew/{d}/{vals}", kind="tuple")
    # same-dtype array domains with specials, complex products (scalarmath is the NAIVE complex product)
    rng = np.random.default_rng(20260927)
    for d in ("complex128", "float64", "float32", "float16"):
        for draw in range(8):
            re = rng.uniform(-2, 2, 4)
            vals = re + 1j * rng.uniform(-2, 2, 4) if d == "complex128" else re
            old = np.array(vals[:2]).astype(d)
            new = np.array(vals[2:]).astype(d)
            emit(op, {"old": A(old), "new": A(new)}, lambda: pu.mapparms(old, new), "ndarray",
                 f"{op}/rand/{d}/{draw}", kind="tuple")
            # lists of 0-d arrays: every product is a ufunc (the fused complex product)
            ol = [A(np.array(old[0])), A(np.array(old[1]))]
            nl = [A(np.array(new[0])), A(np.array(new[1]))]
            emit(op, {"old": ol, "new": nl}, lambda: pu.mapparms(_ps_py(ol), _ps_py(nl)), "list_of_0d",
                 f"{op}/rand0d/{d}/{draw}", kind="tuple")
    # Same-dtype 1-D ndarray domains of every dtype (numpy.polynomial's own case: ABCPolyBase.domain/window are
    # arrays): four NumPy scalars of ONE dtype, which NumSharp runs as one fused kernel that must equal the
    # per-operation scalarmath chain — intermediates wrap / round in the dtype (int8 100 - -100 wraps), the two
    # divisions of an integer dtype are float64, a zero-length domain is inf/nan, never a raise. Strided and
    # reversed views read their elements through the strides.
    for d in ALL_DTYPES:
        if d == "bool":
            continue
        for ov, nv in (((-1, 1), (0, 3)), ((2, 2), (0, 1)), ((100, -100), (3, 7)), ((1, 0), (0, 0)),
                       ((0, 5), (250, 7)), ((-3, 9), (-120, 127)), ((60000, 7), (3, 65000))):
            with np.errstate(all="ignore"):
                old = np.array(ov).astype(d)
                new = np.array(nv).astype(d)
            emit(op, {"old": A(old), "new": A(new)}, lambda: pu.mapparms(old, new), "ndarray",
                 f"{op}/same/{d}/{ov}/{nv}", kind="tuple")
        with np.errstate(all="ignore"):
            ob = np.array([-1, 7, 1, 7]).astype(d)
            nb = np.array([3, 0]).astype(d)
        emit(op, {"old": A(ob, ob[::2]), "new": A(nb, nb[::-1])}, lambda: pu.mapparms(ob[::2], nb[::-1]), "strided",
             f"{op}/same_view/{d}", kind="tuple")
        if d in ("float16", "float32", "float64", "complex128"):
            for ov, nv in (((float("nan"), 1), (0, 1)), ((float("inf"), float("-inf")), (1, 2)),
                           ((0.1, 0.7), (1e308, -1e308)), ((-0.0, 0.0), (1, 1)), ((0.3, -0.7), (1.1, 2.9))):
                with np.errstate(all="ignore"):
                    old = np.array(ov).astype(d)
                    new = np.array(nv).astype(d)
                emit(op, {"old": A(old), "new": A(new)}, lambda: pu.mapparms(old, new), "ndarray",
                     f"{op}/same_special/{d}/{ov}/{nv}", kind="tuple")
    # Python complex against NumPy float64 scalars: `pycomplex op np.float64` is CPython's arithmetic (complex
    # accepts a float subclass), `np.float64 op pycomplex` NumPy's — full-mantissa values separate the divisions.
    for draw in range(12):
        of = np.array(rng.uniform(-3, 3, 2))
        nc = tuple(complex(v, w) for v, w in zip(rng.uniform(-3, 3, 2), rng.uniform(-3, 3, 2)))
        emit(op, {"old": A(of), "new": nc}, lambda: pu.mapparms(of, nc), "mixed_py_np", f"{op}/pycplx_npf64/{draw}",
             kind="tuple")
        emit(op, {"old": nc, "new": A(of)}, lambda: pu.mapparms(nc, of), "mixed_py_np", f"{op}/npf64_pycplx/{draw}",
             kind="tuple")
        pr = tuple(float(v) for v in rng.uniform(-3, 3, 2))
        emit(op, {"old": pr, "new": nc}, lambda: pu.mapparms(pr, nc), "python", f"{op}/pycplx_rand/{draw}",
             kind="tuple")
        emit(op, {"old": nc, "new": pr}, lambda: pu.mapparms(nc, pr), "python", f"{op}/pycplx_rand_r/{draw}",
             kind="tuple")
    for d in ("float64", "complex128"):
        with np.errstate(all="ignore"):
            old2 = np.array([[0.0, 1.0, -2.0], [2.0, 3.0, 5.0]]).astype(d)
            new2 = np.array([[1.0, 0.5, 4.0], [-1.0, 2.5, 1.0]]).astype(d)
        emit(op, {"old": A(old2), "new": A(new2)}, lambda: pu.mapparms(old2, new2), "ndarray_2d",
             f"{op}/nd2/{d}", kind="tuple")
        emit(op, {"old": A(old2), "new": (0, 1)}, lambda: pu.mapparms(old2, (0, 1)), "ndarray_2d",
             f"{op}/nd2py/{d}", kind="tuple")
        c21 = np.array([[1.0], [3.0]]).astype(d)
        emit(op, {"old": A(c21), "new": A(np.array([[0.5, 2.0], [1.0, 3.0]]).astype(d))},
             lambda: pu.mapparms(c21, np.array([[0.5, 2.0], [1.0, 3.0]]).astype(d)), "ndarray_2d",
             f"{op}/nd21/{d}", kind="tuple")
    for tag, old, new in (("short_tuple", (0,), (0, 1)), ("short_list", [0], (0, 1)),
                          ("short_array", A(np.array([0.0])), (0, 1)), ("short_new", (0, 1), [0]),
                          ("zero_d", A(np.array(5.0)), (0, 1)), ("pyint", 5, (0, 1)), ("pyfloat", 1.5, (0, 1)),
                          ("pycomplex", 1j, (0, 1)), ("pybool", True, (0, 1)), ("str2", "ab", (0, 1)),
                          ("str1", "a", (0, 1)), ("zerolen_int", (1, 1), (0, 1)), ("zerolen_float", (1.0, 1.0), (0, 1)),
                          ("zerolen_complex", (1j, 1j), (0, 1)), ("bool_array", A(np.array([False, True])), (0, 1)),
                          ("int8_oob", A(np.array([0, 1], np.int8)), (0, 300)),
                          ("uint8_neg", A(np.array([0, 1], np.uint8)), (0, -1)),
                          ("int64_big", A(np.array([0, 1], np.int64)), (0, 2 ** 63)),
                          ("uint64_neg", A(np.array([0, 1], np.uint64)), (0, -1)),
                          ("int32_big", A(np.array([0, 1], np.int32)), (0, 2 ** 31)),
                          ("uint32_big", A(np.array([0, 1], np.uint32)), (0, 2 ** 32)),
                          ("f32_huge", A(np.array([0, 1], np.float32)), (0, 2 ** 1030))):
        emit(op, {"old": old, "new": new}, lambda: pu.mapparms(_ps_py(old), _ps_py(new)), "error",
             f"{op}/err/{tag}", kind="tuple")

    # ---------------- (G) polyutils.mapdomain ----------------
    op = "polyutils.mapdomain"
    dom_forms = [("pyint", (-1, 1), (0, 2)), ("pyfloat", (-1.0, 1.0), (0.0, 3.0)), ("pycomplex", (-1, 1), (0, 1j)),
                 ("f64", A(np.array([-1.0, 1.0])), A(np.array([0.0, 2.0]))),
                 ("f32", A(np.array([-1.0, 1.0], np.float32)), A(np.array([0.0, 2.5], np.float32))),
                 ("f16", A(np.array([-1.0, 1.0], np.float16)), A(np.array([0.0, 3.0], np.float16))),
                 ("i8", A(np.array([-1, 1], np.int8)), A(np.array([0, 2], np.int8))),
                 ("c128", A(np.array([-1.0, 1.0 + 1j])), A(np.array([0.5j, 2.0]))),
                 ("mixed", A(np.array([-1.0, 1.0], np.float32)), (0, 300)),
                 # the int/float fast lane's hand-over edges (a long overflow in a product; a correctly rounded
                 # big int meeting a float) — appended so the index-based picks above keep their forms
                 ("lane_big", (2 ** 62, -2 ** 62), (3, 5)), ("lane_round", (2 ** 62 + 513, 1.5), (0, 1))]
    for d in ALL_DTYPES:
        x = _ps_series(7, d, "special" if d != "bool" else "moderate", seed=25)
        for tag, old, new in dom_forms:
            emit(op, {"x": A(x), "old": old, "new": new}, lambda: pu.mapdomain(x, _ps_py(old), _ps_py(new)),
                 "c_contiguous_1d", f"{op}/dt/{d}/{tag}")
        x0 = np.array(_ps_series(1, d, seed=26)[0])
        for tag, old, new in dom_forms[:6]:
            emit(op, {"x": A(x0), "old": old, "new": new}, lambda: pu.mapdomain(x0, _ps_py(old), _ps_py(new)),
                 "scalar_0d", f"{op}/0d/{d}/{tag}")
    for d in ("float64", "float32", "complex128", "int16"):
        for ln, xa in _ps_layouts(_ps_series(9, d, seed=27)):
            for tag, old, new in (dom_forms[1], dom_forms[3]):
                emit(op, {"x": xa, "old": old, "new": new}, lambda: pu.mapdomain(_ps_py(xa), _ps_py(old), _ps_py(new)),
                     ln, f"{op}/lay/{d}/{ln}/{tag}")
        m = _ps_series(12, d, seed=28).reshape(3, 4)
        emit(op, {"x": A(m), "old": dom_forms[1][1], "new": dom_forms[1][2]},
             lambda: pu.mapdomain(m, (-1.0, 1.0), (0.0, 3.0)), "c_contiguous_2d", f"{op}/2d/{d}")
        emit(op, {"x": A(m, m.T), "old": dom_forms[3][1], "new": dom_forms[3][2]},
             lambda: pu.mapdomain(m.T, np.array([-1.0, 1.0]), np.array([0.0, 2.0])), "transposed_2d", f"{op}/T/{d}")
    for xv in (0, 5, -3, 2 ** 70, 0.5, -0.0, float("nan"), 1j, complex(float("inf"), 1), True, [0.5, 1.0], [1, 2]):
        for tag, old, new in dom_forms:
            emit(op, {"x": xv, "old": old, "new": new}, lambda: pu.mapdomain(xv, _ps_py(old), _ps_py(new)),
                 "python", f"{op}/py/{xv!r}/{tag}")
    for tag, x, old, new in (("zerolen", 0.5, (1, 1), (0, 1)), ("zerolen_arr", A(np.array([0.5])), (1.0, 1.0), (0, 1)),
                             ("short", A(np.array([0.5])), (0,), (0, 1)), ("scalar_dom", A(np.array([0.5])), 1, (0, 1))):
        emit(op, {"x": x, "old": old, "new": new}, lambda: pu.mapdomain(_ps_py(x), _ps_py(old), _ps_py(new)), "error",
             f"{op}/err/{tag}")

    # ---------------- (H) {p}line ----------------
    offs = [0, 1, -1, 2, 2 ** 63, -2 ** 63, 2 ** 64 - 1, 0.0, -0.0, 1.5, float("nan"), float("inf"), 1j, 0j,
            complex(float("inf"), 1), True, False]
    scls = [0, 1, 2, 3, -1, 2 ** 63, 2 ** 60 + 1, 2 ** 64 - 1, 2 ** 1030, 0.0, -0.0, 2.5, float("nan"), float("inf"), 1j,
            0j, complex(0, float("nan")), complex(float("inf"), 1), True, False]
    for modname, p in POLY_MODULES:
        f = getattr(_poly_module(modname), p + "line")
        op = f"{modname}.{p}line"
        for off in offs:
            for scl in scls:
                emit(op, {"off": off, "scl": scl}, lambda: f(off, scl), "python", f"{op}/py/{off!r}/{scl!r}")
        # NumPy scalars (0-d arrays: strong dtype) of every dtype, against each other and Python numbers
        for d in ALL_DTYPES:
            for sv in (0, 1, 3, 100):
                with np.errstate(all="ignore"):
                    s0 = np.array(sv).astype(d)
                    o0 = np.array(1).astype(d)
                for off in (A(o0), 1, 2.5, 1j):
                    emit(op, {"off": off, "scl": A(s0)}, lambda: f(_ps_py(off), s0), "scalar_0d",
                         f"{op}/np/{d}/{sv}/{off if not isinstance(off, _PSArr) else 'np'}")
                for scl in (2, 300, -1, 2.5, 2 ** 63):
                    emit(op, {"off": A(o0), "scl": scl}, lambda: f(o0, scl), "scalar_0d", f"{op}/npoff/{d}/{scl!r}")
        for d1, d2 in (("int8", "uint8"), ("float16", "float32"), ("uint64", "int64"), ("float16", "int16"),
                       ("int32", "uint32"), ("bool", "int8"), ("float32", "complex128")):
            a0, b0 = np.array(1).astype(d1), np.array(3).astype(d2)
            emit(op, {"off": A(a0), "scl": A(b0)}, lambda: f(a0, b0), "scalar_0d", f"{op}/mix/{d1}/{d2}")
        # full-mantissa Python values: hermline's scl / 2 is CPython's _Py_c_quot / int true division
        lrng = np.random.default_rng(20260928 + POLY_MODULES.index((modname, p)))
        for draw in range(6):
            cv = complex(*lrng.uniform(-5, 5, 2))
            fv = float(lrng.uniform(-5, 5))
            iv = int(lrng.integers(-2 ** 62, 2 ** 62))
            for off, scl in ((fv, cv), (cv, fv), (iv, cv), (cv, iv), (fv, iv)):
                emit(op, {"off": off, "scl": scl}, lambda: f(off, scl), "python", f"{op}/rand/{draw}/{off!r}/{scl!r}")
        # array operands: 1-element arrays stack, others hit NumPy's truth-value / inhomogeneity errors
        for tag, off, scl in (("arr1_arr1", A(np.array([1])), A(np.array([2]))),
                              ("arr1_arr1_f", A(np.array([1.5])), A(np.array([2.5], np.float32))),
                              ("arr1_zero", 1, A(np.array([0]))), ("arr1_py", 1, A(np.array([2]))),
                              ("arr2", 1, A(np.array([1, 2]))), ("arr0", 1, A(np.zeros(0))),
                              ("arr11_arr1", A(np.array([[1]])), A(np.array([2]))),
                              ("arr2_arr1", A(np.array([1, 2])), A(np.array([3]))),
                              ("big_obj", 2 ** 64, 1), ("big_obj2", 2 ** 63, 2 ** 63)):
            emit(op, {"off": off, "scl": scl}, lambda: f(_ps_py(off), _ps_py(scl)), "array_args", f"{op}/arr/{tag}")

    # ---------------- (I) constants ----------------
    # Four observable facets of each module constant: its value, its identity (the SAME ndarray object on every
    # attribute access — a write through it persists), and its flags (writeable, owning its buffer). A constant has
    # no argument to vary, so the facets are the case axis.
    for modname, p in POLY_MODULES:
        mod = _poly_module(modname)
        for name in ("domain", "zero", "one", "x"):
            attr = p + name
            op = f"{modname}.{attr}"
            emit(op, {"facet": "value"}, lambda: getattr(mod, attr), "constant", f"{op}/value")
            emit(op, {"facet": "identity"}, lambda: np.bool_(getattr(mod, attr) is getattr(mod, attr)), "constant",
                 f"{op}/identity")
            emit(op, {"facet": "writeable"}, lambda: np.bool_(getattr(mod, attr).flags.writeable), "constant",
                 f"{op}/writeable")
            emit(op, {"facet": "owndata"}, lambda: np.bool_(getattr(mod, attr).flags.owndata), "constant",
                 f"{op}/owndata")

    # ---------------- (J) complex64 NumPy scalars ----------------
    # NEP 50 turns a float16/float32 NumPy value meeting a Python complex into a complex64 NumPy SCALAR, which stays
    # complex64 against Python numbers and narrow NumPy values. NumSharp has no complex64 dtype (#569) but emulates
    # that scalar arithmetic in float32 exactly, so a complex128 result built from it must match NumPy bit for bit,
    # and a complex64 result (recorded only where every step was scalar-sized, _ps_c64_exact) must carry NumPy's
    # values. Full-mantissa draws separate the forms: scalarmath's NAIVE complex product (mapdomain at a Python
    # complex x: complex64 scl times it), CFLOAT_divide's reciprocal-multiply Smith (mapparms' divisions by a float32
    # oldlen), a 0-d operand's ufunc (a 0-d float32 x), and the rounding of a Python complex into the loop. A 1-D
    # float32/float16 domain makes old[0]/old[1] genuine NumPy scalars (scalarmath); a 0-d array is a ufunc operand.
    crng = np.random.default_rng(20260928)

    def c64_draw():
        return tuple(complex(*crng.uniform(-3, 3, 2)) for _ in range(2))

    for draw in range(12):
        for dt in ("float32", "float16"):
            with np.errstate(all="ignore"):
                dom = np.array(crng.uniform(-3, 3, 2)).astype(dt)
            cdom = c64_draw()
            op = "polyutils.mapparms"
            emit(op, {"old": A(dom), "new": cdom}, lambda: pu.mapparms(dom, cdom), "complex64",
                 f"{op}/c64/{dt}/{draw}", kind="tuple")
            emit(op, {"old": cdom, "new": A(dom)}, lambda: pu.mapparms(cdom, dom), "complex64",
                 f"{op}/c64r/{dt}/{draw}", kind="tuple")
            ol = [A(np.array(dom[0])), A(np.array(dom[1]))]
            emit(op, {"old": ol, "new": cdom}, lambda: pu.mapparms(_ps_py(ol), cdom), "complex64",
                 f"{op}/c64_0d/{dt}/{draw}", kind="tuple")
            op = "polyutils.mapdomain"
            xc = complex(*crng.uniform(-3, 3, 2))
            xf = float(crng.uniform(-3, 3))
            x64 = crng.uniform(-3, 3, 5)
            xc128 = crng.uniform(-3, 3, 5) + 1j * crng.uniform(-3, 3, 5)
            x0 = np.array(crng.uniform(-3, 3)).astype(dt)
            for tag, xv in (("pycomplex", xc), ("pyfloat", xf), ("f64", A(x64)), ("c128", A(xc128)), ("0d", A(x0))):
                emit(op, {"x": xv, "old": A(dom), "new": cdom},
                     lambda: pu.mapdomain(_ps_py(xv), dom, cdom), "complex64", f"{op}/c64/{dt}/{tag}/{draw}")
            for modname, p in (("laguerre", "lag"), ("polynomial", "poly"), ("hermite", "herm")):
                f = getattr(_poly_module(modname), p + "line")
                op = f"{modname}.{p}line"
                s0 = np.array(crng.uniform(-3, 3)).astype(dt)
                cv = complex(*crng.uniform(-3, 3, 2))
                emit(op, {"off": A(s0), "scl": cv}, lambda: f(s0, cv), "complex64", f"{op}/c64/{dt}/{draw}")
                emit(op, {"off": cv, "scl": A(s0)}, lambda: f(cv, s0), "complex64", f"{op}/c64r/{dt}/{draw}")

    # ---------------- (K) long series ----------------
    # Every section above uses series of at most a dozen elements, all of which run the fused scalar kernels. From
    # NumSharp's PolyHouseKernelThreshold (64 elements) on, the SAME operations run through the house SIMD kernels
    # instead — memcpy / the astype cast kernels for as_series' conversion copy, the same-dtype SimdFull binary kernel
    # (plus the house negate for `c2 = -c2`) for _add/_sub — so this block pins those routes to NumPy's bits: lengths
    # on both sides of the threshold (63/64/65) and past a vector body into its remainder, every kind of update
    # (add, subtract with the minuend longer, subtract with the subtrahend at least as long), converted operands,
    # strided and reversed operands, results whose trailing zeros are trimmed back below the threshold, and specials
    # (NaN, ±inf, ±0, subnormals, overflowing sums) in both the updated prefix and the untouched tail.
    def long_series(n, dt, seed, tz=0):
        lr = np.random.default_rng(seed)
        dt = np.dtype(dt)
        if dt.kind in "fc":
            f = np.finfo(dt if dt.kind == "f" else np.float64)
            big, tiny = float(f.max) * 0.75, float(f.smallest_subnormal)
            spec = {3: float("nan"), 5: -0.0, 7: 0.0, 11: float("inf"), 13: tiny, 19: big, 20: big, 29: float("-inf"),
                    n - 7: float("nan"), n - 4: -big}
            if dt.kind == "f":
                vals = lr.uniform(-4, 4, n)
                for i, v in spec.items():
                    if 0 <= i < n:
                        vals[i] = v
                with np.errstate(all="ignore"):
                    a = vals.astype(dt)
            else:
                re, im = lr.uniform(-4, 4, n), lr.uniform(-4, 4, n)
                for i, v in spec.items():
                    if 0 <= i < n:
                        re[i] = v
                        im[i] = -v if i % 2 else 1.5
                a = re + 1j * im
        else:
            info = np.iinfo(dt)
            vals = lr.integers(max(int(info.min), -1000), min(int(info.max), 1000), n, endpoint=True)
            a = vals.astype(dt)
            for i, v in ((2, info.min), (9, info.max), (17, 0), (n - 3, info.max)):
                if 0 <= i < n:
                    a[i] = v
        if tz:
            z = np.zeros(tz, dt)
            if dt.kind in "fc":
                z[::2] = -0.0
            a = np.concatenate([a, z]).astype(dt)
        return a

    def long_layouts(arr, which=("c", "s", "r")):
        """(name, _PSArr) views of arr's values: contiguous, stride-2 and reversed (the three CopyInto routes)."""
        out = []
        if "c" in which:
            out.append(("c_contiguous_1d", A(arr.copy())))
        if "s" in which:
            s = np.zeros(2 * arr.size, arr.dtype)
            s[::2] = arr
            out.append(("strided_1d", A(s, s[::2])))
        if "r" in which:
            r = arr[::-1].copy()
            out.append(("reversed_1d", A(r, r[::-1])))
        return out

    long_pairs = ((63, 63), (64, 64), (65, 65), (70, 64), (64, 70), (71, 3), (3, 71), (257, 100), (100, 257),
                  (1000, 1000), (1003, 5), (5, 1003))
    for opname, f in (("polynomial.polyadd", _poly_module("polynomial").polyadd),
                      ("polynomial.polysub", _poly_module("polynomial").polysub),
                      ("legendre.legsub", _poly_module("legendre").legsub)):
        for d in ("float64", "float32", "float16", "complex128"):
            for n1, n2 in long_pairs:
                c1 = long_series(n1, d, 100 + n1)
                c2 = long_series(n2, d, 200 + n2)
                emit(opname, {"c1": A(c1), "c2": A(c2)}, lambda: f(c1, c2), "c_contiguous_1d",
                     f"{opname}/long/{d}/{n1}/{n2}")
        for d1, d2 in (("float16", "float32"), ("float32", "float16"), ("float32", "float64"), ("int32", "float64"),
                       ("int8", "float16"), ("uint8", "uint8"), ("int64", "int64"), ("uint64", "float64"),
                       ("complex128", "float64"), ("float16", "complex128"), ("uint16", "float32"),
                       ("float16", "float16")):
            for n1, n2 in ((65, 70), (70, 65), (257, 257)):
                c1 = long_series(n1, d1, 300 + n1)
                c2 = long_series(n2, d2, 400 + n2)
                emit(opname, {"c1": A(c1), "c2": A(c2)}, lambda: f(c1, c2), "c_contiguous_1d",
                     f"{opname}/longmix/{d1}/{d2}/{n1}/{n2}")
        for d1, d2 in (("float64", "float64"), ("float32", "float32"), ("float16", "float16"), ("float16", "float32"),
                       ("int16", "float64")):
            for n1, n2 in ((100, 70), (70, 100), (100, 100)):
                for ln1, a1 in long_layouts(long_series(n1, d1, 500 + n1)):
                    for ln2, a2 in long_layouts(long_series(n2, d2, 600 + n2)):
                        emit(opname, {"c1": a1, "c2": a2}, lambda: f(_ps_py(a1), _ps_py(a2)), f"{ln1}+{ln2}",
                             f"{opname}/longlay/{d1}/{d2}/{ln1}/{ln2}/{n1}/{n2}")
        # Trimming: the operands' own trailing zeros (trimmed BEFORE the update, possibly below the threshold) and a
        # result whose tail cancels (trimmed AFTER it, to a view).
        for d in ("float64", "float32", "float16", "complex128"):
            for n, tz in ((60, 30), (64, 1), (100, 40)):
                c1 = long_series(n, d, 700 + n, tz=tz)
                c2 = long_series(n // 2, d, 800 + n)
                emit(opname, {"c1": A(c1), "c2": A(c2)}, lambda: f(c1, c2), "c_contiguous_1d",
                     f"{opname}/longtz/{d}/{n}/{tz}")
                emit(opname, {"c1": A(c2), "c2": A(c1)}, lambda: f(c2, c1), "c_contiguous_1d",
                     f"{opname}/longtzr/{d}/{n}/{tz}")
            base = long_series(120, d, 900)
            other = base.copy()
            other[:100] = long_series(100, d, 901)
            with np.errstate(all="ignore"):
                cancel = -other if opname.endswith("add") else other
            emit(opname, {"c1": A(base), "c2": A(cancel)}, lambda: f(base, cancel), "c_contiguous_1d",
                 f"{opname}/longcancel/{d}")

    op = "polyutils.as_series"
    for d1 in ALL_DTYPES:
        if d1 == "bool":
            continue
        for d2 in ("float64", "float16", "complex128"):
            for trim in (True, False):
                for ln, a1 in long_layouts(long_series(100, d1, 1000, tz=3)):
                    a2 = A(long_series(80, d2, 1001))
                    emit(op, {"alist": [a1, a2], "trim": trim}, lambda: pu.as_series([_ps_py(a1), _ps_py(a2)], trim=trim),
                         "list_of_arrays", f"{op}/long/{d1}/{d2}/{ln}/{trim}", kind="tuple")

    op = "polyutils.trimcoef"
    for d in ("float64", "float32", "float16", "complex128", "int32", "uint8"):
        for ln, ca in long_layouts(long_series(100, d, 1100, tz=5), ("c", "s")):
            for tol in (0, 0.5):
                emit(op, {"c": ca, "tol": tol}, lambda: pu.trimcoef(_ps_py(ca), tol), ln, f"{op}/long/{d}/{ln}/{tol}")

    op = "polyutils.getdomain"
    for d in ALL_DTYPES:
        if d == "bool":
            continue
        for n in (100, 1000):
            if n == 1000 and d not in ("float64", "float32", "float16", "complex128", "int64"):
                continue
            for ln, xa in long_layouts(long_series(n, d, 1200 + n)):
                emit(op, {"x": xa}, lambda: pu.getdomain(_ps_py(xa)), ln, f"{op}/long/{d}/{n}/{ln}")

    op = "polyutils.mapdomain"
    for d in ALL_DTYPES:
        if d == "bool":
            continue
        for n in (100, 1000):
            if n == 1000 and d not in ("float64", "float32", "float16", "complex128", "uint8"):
                continue
            for ln, xa in long_layouts(long_series(n, d, 1300 + n), ("c", "s", "r") if n == 100 else ("c",)):
                for tag, old, new in dom_forms[:9]:
                    emit(op, {"x": xa, "old": old, "new": new},
                         lambda: pu.mapdomain(_ps_py(xa), _ps_py(old), _ps_py(new)), ln, f"{op}/long/{d}/{n}/{ln}/{tag}")

    # ---------------- (L) complex64 ARRAY loops of mapdomain ----------------
    # A complex64 off/scl (mapparms of a float32/float16 domain and a Python complex tuple) with every narrow x NEP 50
    # keeps in complex64, and a Python complex tuple domain with a float16/float32 x. Lengths 1 (NpyIter's stride-0
    # one-element iteration: CFLOAT_multiply's loop_scalar product form), 5 and 37 (the vector loop's body and
    # remainder), plus strided and reversed views. Appended after (K) so the running case counter of every earlier
    # case is unchanged.
    op = "polyutils.mapdomain"
    for draw in range(6):
        for dt in ("float32", "float16"):
            arng = np.random.default_rng(20260930 + 17 * draw + (dt == "float16"))
            with np.errstate(all="ignore"):
                dom = np.array(arng.uniform(-3, 3, 2)).astype(dt)
                cdom = tuple(complex(*arng.uniform(-3, 3, 2)) for _ in range(2))
                for xdt in ("float32", "float16", "int8", "uint8", "int16", "uint16", "bool"):
                    for n in (1, 5, 37):
                        raw = arng.uniform(-3, 3, n)
                        xs = (raw > 0 if xdt == "bool" else raw * (40 if xdt[0] in "iu" else 1)).astype(xdt)
                        emit(op, {"x": A(xs), "old": A(dom), "new": cdom},
                             lambda: pu.mapdomain(xs, dom, cdom), "complex64", f"{op}/c64arr/{dt}/{xdt}/{n}/{draw}")
                        if xdt in ("float32", "float16"):
                            pc0 = complex(*arng.uniform(-3, 3, 2))
                            pc1 = complex(*arng.uniform(-3, 3, 2))
                            pold, pnew = (pc0, pc0 + 2), (0, pc1)
                            emit(op, {"x": A(xs), "old": pold, "new": pnew},
                                 lambda: pu.mapdomain(xs, pold, pnew), "complex64", f"{op}/c64py/{dt}/{xdt}/{n}/{draw}")
                    xl = (arng.uniform(-3, 3, 23) * (40 if xdt[0] in "iu" else 1)).astype(xdt if xdt != "bool" else "float32")
                    rb = np.repeat(xl, 2)
                    rv = xl[::-1].copy()
                    for ln, xa in (("strided_1d", A(rb, rb[::2])), ("reversed_1d", A(rv, rv[::-1]))):
                        emit(op, {"x": xa, "old": A(dom), "new": cdom},
                             lambda: pu.mapdomain(_ps_py(xa), dom, cdom), "complex64", f"{op}/c64lay/{dt}/{xdt}/{ln}/{draw}")

    # ---------------- (M) block and staging boundaries ----------------
    # (K) stops near 1,000 elements, below every boundary the long routes have beyond the 64-element house threshold:
    # the 1,024-element blocks in which the in-place combine materializes a converted, strided or negated operand
    # through a stack scratch, in which a strided source is copied then cast, and in which mapdomain's affine route
    # converts its points (a float64 / float32 / complex128 loop over 1-D or C-contiguous points) before mapping them;
    # and the fused pass's staging of an N-D non-contiguous x (from 8,192 points). One case per route past its
    # boundary — several whole blocks and a partial last one — in the dtypes whose conversion takes that route. The
    # series come from long_series (specials included), with seeds of their own; appended after (L) so every earlier
    # case keeps its running counter.
    for opname, f in (("polynomial.polyadd", _poly_module("polynomial").polyadd),
                      ("polynomial.polysub", _poly_module("polynomial").polysub)):
        for d1, d2, lay1 in (("int32", "float64", "c"), ("float64", "int32", "c"), ("float64", "float64", "c"),
                             ("float32", "float32", "s"), ("int16", "float64", "r"), ("float16", "float32", "c")):
            for n1, n2 in ((2100, 2150), (2150, 2100)):
                (ln1, a1), = long_layouts(long_series(n1, d1, 2000 + n1), (lay1,))
                a2 = A(long_series(n2, d2, 2100 + n2))
                emit(opname, {"c1": a1, "c2": a2}, lambda: f(_ps_py(a1), _ps_py(a2)), f"{ln1}+c_contiguous_1d",
                     f"{opname}/block/{d1}/{d2}/{ln1}/{n1}/{n2}")

    op = "polyutils.as_series"
    for d1, lay in (("int16", "s"), ("int64", "r"), ("uint32", "s"), ("float32", "r")):
        (ln, a1), = long_layouts(long_series(2100, d1, 2200), (lay,))
        a2 = A(long_series(10, "float64", 2201))
        emit(op, {"alist": [a1, a2], "trim": True}, lambda: pu.as_series([_ps_py(a1), _ps_py(a2)]),
             "list_of_arrays", f"{op}/block/{d1}/{ln}", kind="tuple")

    op = "polyutils.mapdomain"
    f64dom, f32dom = np.array([-2.0, 3.0]), np.array([-2.0, 3.0], np.float32)
    new64, new32 = np.array([0.5, 4.0]), np.array([0.5, 4.0], np.float32)
    # A contiguous x converted block by block into a float64 loop (Python-int domains) and into a float32 loop
    # (float32 array domains).
    for d, n, old, new, tag in (("int8", 2100, (-1, 1), (0, 2), "pyint"), ("uint8", 2100, (-1, 1), (0, 2), "pyint"),
                                ("int16", 2100, (-1, 1), (0, 2), "pyint"), ("uint16", 2100, (-1, 1), (0, 2), "pyint"),
                                ("int64", 2100, (-1, 1), (0, 2), "pyint"), ("uint64", 2100, (-1, 1), (0, 2), "pyint"),
                                ("float16", 600, f64dom, new64, "f64"), ("int8", 1030, f32dom, new32, "f32"),
                                ("uint8", 1030, f32dom, new32, "f32")):
        xa = A(long_series(n, d, 2300 + n))
        emit(op, {"x": xa, "old": old if isinstance(old, tuple) else A(old), "new": new if isinstance(new, tuple) else A(new)},
             lambda: pu.mapdomain(_ps_py(xa), old, new), "c_contiguous_1d", f"{op}/stage/{d}/{n}/{tag}")
    # A strided / reversed 1-D x: the house SIMD strided copy (then cast) of each block.
    for d, lay, old, new, tag in (("int64", "r", (-1, 1), (0, 2), "pyint"), ("int16", "s", (-1, 1), (0, 2), "pyint"),
                                  ("uint8", "s", (-1, 1), (0, 2), "pyint"), ("int32", "s", (-1, 1), (0, 2), "pyint"),
                                  ("float32", "s", f64dom, new64, "f64")):
        (ln, xa), = long_layouts(long_series(2100, d, 2400), (lay,))
        emit(op, {"x": xa, "old": old if isinstance(old, tuple) else A(old), "new": new if isinstance(new, tuple) else A(new)},
             lambda: pu.mapdomain(_ps_py(xa), old, new), ln, f"{op}/stage/{d}/{ln}/{tag}")
    # A transposed 2-D x (N-D, not contiguous): packed contiguous before the fused pass, then cast.
    for d in ("int16", "float32"):
        base2 = long_series(96 * 90, d, 2500).reshape(96, 90)
        xa = A(base2, base2.T)
        emit(op, {"x": xa, "old": (-1, 1), "new": (0, 2)}, lambda: pu.mapdomain(_ps_py(xa), (-1, 1), (0, 2)),
             "transposed_2d", f"{op}/stage/{d}/transposed_2d/pyint")

    # ---------------- (N) getdomain across the blocked pass's windows ----------------
    # NumSharp's getdomain reduces NumPy's copy in 8 KB windows (8,192 / itemsize elements, counted after element 0)
    # and carries the min/max accumulators from window to window: read in place (contiguous; a reversed INTEGER x as
    # the forward run it covers), packed by the SIMD strided copy otherwise, uint64 converted to float64 as it is
    # packed. (K) stops at 1,000 points — inside one window — so these cases cross a window boundary (a full window
    # folded group by group, then the partial last one finished: both halves of the pass) in every dtype the pass
    # serves and every layout it reads differently, and pin the schedule facts the carried accumulators
    # must preserve: which lane's zero survives a ±0 tie (a -0.0 early in the HIGHEST lane against a +0.0 late in
    # lane 0: NumPy's horizontal cascade returns the early one, where a sequential fold would return the late one)
    # and whether a NaN comes back canonical (met inside the vector section) or with its payload (met in the scalar
    # tail). Appended after (M) so every earlier case keeps its running counter.
    op = "polyutils.getdomain"
    for d in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float64"):
        window = 8192 // np.dtype(d).itemsize
        n = window + 37
        for ln, xa in long_layouts(long_series(n, d, 2600 + n)):
            emit(op, {"x": xa}, lambda: pu.getdomain(_ps_py(xa)), ln, f"{op}/window/{d}/{n}/{ln}")
    for d in ("int64", "float64", "uint8"):
        # Stride 3: the AVX2 gather (8-byte) / the scalar sub-word copy.
        window = 8192 // np.dtype(d).itemsize
        arr = long_series(window + 37, d, 2650)
        b3 = np.zeros(3 * arr.size, arr.dtype)
        b3[::3] = arr
        xa = A(b3, b3[::3])
        emit(op, {"x": xa}, lambda: pu.getdomain(_ps_py(xa)), "strided3_1d", f"{op}/window/{d}/strided3_1d")
    for d in ("float64", "float32"):
        lanes = 32 // np.dtype(d).itemsize
        window = 8192 // np.dtype(d).itemsize
        n = 2 * window + 1          # a whole number of vectors after element 0: no scalar tail, the cascade decides
        late = 1 + lanes * ((n - 1) // lanes - 1)       # lane 0 of the last vector (the last window)
        for early in (lanes, window + lanes):           # the highest lane, in window 0 / in window 1
            for tag, fill, zeros in (("max", -1.0, (-0.0, 0.0)), ("min", 1.0, (0.0, -0.0))):
                x = np.full(n, fill, d)
                x[early], x[late] = zeros
                for ln, xa in long_layouts(x):
                    emit(op, {"x": xa}, lambda: pu.getdomain(_ps_py(xa)), ln, f"{op}/lanes/{d}/{tag}/{early}/{ln}")
        # One NaN with a payload (negative sign, 0xbad), on the first element of window 1 (inside the vector section:
        # canonicalized by the horizontal reduce) and on the last element of a series whose length leaves a scalar
        # tail (kept as is). The series' own NaNs are zeroed so the planted one is the only NaN.
        payload = np.array([0xfff8000000000bad], "u8").view("f8")[0] if d == "float64" else \
            np.array([0xffc00bad], "u4").view("f4")[0]
        for m, at in ((n, window + 1), (n - 1, n - 2)):
            x = long_series(m, d, 2700 + at)
            x[np.isnan(x)] = 0
            x[at] = payload
            for ln, xa in long_layouts(x):
                emit(op, {"x": xa}, lambda: pu.getdomain(_ps_py(xa)), ln, f"{op}/nan/{d}/{m}/{at}/{ln}")

    # ---------------- (O) Python tuples and nested sequences ----------------
    # The C# boundary's ValueTuple (a Python tuple) and nested object[] (lists), coerced by np.array's walk (NDPolySequence):
    # a tuple is a sequence like a list, nesting makes an N-D array (as_series / {p}add / getdomain then say "not 1-d"),
    # a ragged one raises np.array's inhomogeneous text, an empty one is float64 (0,) — and as_series iterates a tuple.
    f32a = np.array([1.0, 2.0, 0.0], np.float32)
    f32b = np.array([3.0, 4.0], np.float32)
    tuple_coefs = [("tuple", (1, 2)), ("tuple_f", (1.5, -2.0, 0.0)), ("tuple_mixed", (1, 2.5, 1j)),
                   ("tuple_nested", ((1, 2), (3, 4))), ("list_of_tuples", [(1, 2), (3, 4)]),
                   ("tuple_arrays", (A(f32a), A(np.array([3.0, 4.0, 5.0], np.float32)))), ("empty_tuple", ()),
                   ("ragged", [[1], [1, 2]]), ("ragged_deep", [[[1], [1, 2]], [[1], [1, 2]]]), ("nested3", [[[1.0]]])]
    for modname, p in (("polynomial", "poly"), ("chebyshev", "cheb")):
        mod = _poly_module(modname)
        for opname in ("add", "sub"):
            f = getattr(mod, p + opname)
            op = f"{modname}.{p}{opname}"
            for tname, tv in tuple_coefs:
                for other in ((3, 4, 5), [0.5]):
                    emit(op, {"c1": tv, "c2": other}, lambda: f(_ps_py(tv), other), "python_tuple",
                         f"{op}/tuple/{tname}/{len(other)}")
                    emit(op, {"c1": other, "c2": tv}, lambda: f(other, _ps_py(tv)), "python_tuple",
                         f"{op}/tupler/{tname}/{len(other)}")
    for tag, alist, trim in (("tuple_arrays", (A(f32a), A(f32b)), False), ("tuple_arrays_trim", (A(f32a), A(f32b)), True),
                             ("tuple_tuples", ((1, 2, 0), (3.5, 4)), True), ("tuple_lists", ([1, 2], [3, 4, 0]), True),
                             ("list_tuples", [(1, 2), (3, 4, 0)], False), ("tuple_nested_err", ([[1, 2]], [3]), False),
                             ("tuple_ragged_err", ([[1], [1, 2]], [3]), False), ("empty_tuple", (), True),
                             ("tuple_scalars", (1, 2.5), True), ("tuple_empty_item", ((), [1]), True)):
        emit("polyutils.as_series", {"alist": alist, "trim": trim}, lambda: pu.as_series(_ps_py(alist), trim=trim),
             "python_tuple", f"polyutils.as_series/tuple/{tag}", kind="tuple")
    for tag, cv in (("tuple", (1, 2, 0, 0)), ("tuple_f", (0.5, 1e-3, 0.0)), ("tuple_nested", ((1, 2), (0, 0))),
                    ("ragged", [[1], [1, 2]])):
        emit("polyutils.trimcoef", {"c": cv, "tol": 1e-2}, lambda: pu.trimcoef(_ps_py(cv), 1e-2), "python_tuple",
             f"polyutils.trimcoef/tuple/{tag}")
    for tag, xv in (("tuple", (1.0, 5.0, -2.0)), ("tuple_int", (3, -7, 2)), ("tuple_nested", ((1.0, 5.0),)),
                    ("ragged", [[1], [1, 2]]), ("empty_tuple", ())):
        emit("polyutils.getdomain", {"x": xv}, lambda: pu.getdomain(_ps_py(xv)), "python_tuple",
             f"polyutils.getdomain/tuple/{tag}")
    for tag, xv in (("tuple", (0.5, 1.5)), ("nested", [[0.5, 1.5], [2, 3]]), ("nested_tuple", ((1, 2), (3, 4))),
                    ("ragged", [[1], [1, 2]]), ("empty", [])):
        emit("polyutils.mapdomain", {"x": xv, "old": (0, 2), "new": (-1, 1)},
             lambda: pu.mapdomain(_ps_py(xv), (0, 2), (-1, 1)), "python_tuple", f"polyutils.mapdomain/tuple/{tag}")

    # ---------------- (P) trimseq of Python sequences ----------------
    # NumPy returns the KIND it was given — the list / tuple itself, or its slice seq[:i+1] (a list / tuple); the corpus
    # records np.asarray of it. An item's zero-ness is Python's `item != 0`: a number by value (NaN nonzero, -0.0 zero, a
    # complex zero only when both parts are, a bool itself), a NumPy scalar likewise, a one-element array by its truth
    # value (an empty or longer array raises NumPy's ValueError when TESTED — seq[0] included), a nested list never 0.
    for tag, seq in (("list", [1, 2, 0]), ("list_keep", [1, 2]), ("list_allzero", [0, 0, 0]), ("list_empty", []),
                     ("list_nan", [1, float("nan")]), ("list_negzero", [1.5, 0.0, -0.0]), ("list_complex", [1, 0j]),
                     ("list_complex_im", [1, 1j, 0]), ("list_bool", [True, False]), ("list_mixed", [1, 2.5, 0, 0.0]),
                     ("list_one_zero", [0]), ("tuple", (1, 0, 0)), ("tuple_keep", (1, 2)), ("tuple_empty", ()),
                     ("tuple_allzero", (0, 0)), ("tuple_long", (1, 2, 3, 4, 5, 6, 7, 8, 9, 0)),
                     ("list_f16", [A(np.array(1.0, np.float16)), A(np.array(0.0, np.float16))]),
                     ("list_f16_negzero", [A(np.array(2.0, np.float16)), A(np.array(-0.0, np.float16))]),
                     ("list_1elem_arrays", [A(np.array([1.0])), A(np.array([0.0]))]),
                     ("list_2d_1elem", [A(np.array([[3]])), A(np.array([[0]]))]),
                     ("list_multi_array", [A(np.array([1.0, 2.0])), 0]),
                     ("list_multi_array_last", [1, A(np.array([0.0, 0.0]))]),
                     ("list_empty_array", [A(np.array([1.0])), A(np.zeros(0))]),
                     ("list_nested", [[1], [0]]), ("list_u64", [2 ** 64 - 1, 0]), ("list_bigneg", [-(2 ** 63), 0])):
        emit("polyutils.trimseq", {"seq": seq}, lambda: pu.trimseq(_ps_py(seq)), "python_seq", f"polyutils.trimseq/seq/{tag}")

    # ---------------- (Q) the deferred object / str refusal ----------------
    # A None / non-numeric / oversized-int item makes NumPy's array an OBJECT one and a str item a str one; as_series still
    # checks every argument's size and dims, then fails the common type (str, bool) or computes with Python objects (then
    # NumSharp refuses — not recorded). Only outcomes NumPy reaches before any object arithmetic are recorded.
    obj1 = [1.0, None]
    strl = [1.0, "a"]
    two = np.zeros((2, 2))
    for modname, p in (("polynomial", "poly"), ("chebyshev", "cheb")):
        mod = _poly_module(modname)
        for opname in ("add", "sub"):
            f = getattr(mod, p + opname)
            op = f"{modname}.{p}{opname}"
            for tag, c1v, c2v in (("obj_empty", None, A(np.zeros(0))), ("empty_obj", A(np.zeros(0)), obj1),
                                  ("obj_2d", obj1, A(two)), ("obj_ragged", None, [[1, 2], [3]]),
                                  ("ragged_obj", [[1, 2], [3]], None), ("str_f", strl, [0.5]), ("f_str", [0.5], strl),
                                  ("str_bool", strl, A(np.array([True]))), ("str_2d", [["a", 1.0]], [0.5]),
                                  ("str_obj", strl, obj1)):
                call = (lambda f=f, c1v=c1v, c2v=c2v: f(_ps_py(c1v), _ps_py(c2v)))
                if _pa_object_land(call):
                    continue
                emit(op, {"c1": c1v, "c2": c2v}, call, "object_order", f"{op}/objorder/{tag}")
    for tag, alist in (("obj_empty", [None, A(np.zeros(0))]), ("str_2d", [strl, A(two)]), ("str_only", [strl]),
                       ("obj_str", [obj1, strl]), ("empty_str", [A(np.zeros(0)), strl]), ("str_tuple", (["x"], [1.0]))):
        call = (lambda alist=alist: pu.as_series(_ps_py(alist)))
        if _pa_object_land(call):
            continue
        emit("polyutils.as_series", {"alist": alist}, call, "object_order", f"polyutils.as_series/objorder/{tag}",
             kind="tuple")
    for tag, xv in (("obj", obj1), ("str", strl), ("str_2d", [["a", 1.0]]), ("obj_empty_list", [None, []])):
        call = (lambda xv=xv: pu.getdomain(_ps_py(xv)))
        if not _pa_object_land(call):
            emit("polyutils.getdomain", {"x": xv}, call, "object_order", f"polyutils.getdomain/objorder/{tag}")
        call = (lambda xv=xv: pu.trimcoef(_ps_py(xv)))
        if not _pa_object_land(call):
            emit("polyutils.trimcoef", {"c": xv}, call, "object_order", f"polyutils.trimcoef/objorder/{tag}")
        call = (lambda xv=xv: pu.trimcoef(_ps_py(xv), -1.0))
        if not _pa_object_land(call):
            emit("polyutils.trimcoef", {"c": xv, "tol": -1.0}, call, "object_order",
                 f"polyutils.trimcoef/objorder/{tag}_negtol")

    # Char: NumSharp's uint16-like dtype — the uint16 cells relabelled (bytes-exact oracle, the house weave).
    cases += _relabel_dtype([c for c in cases if "/uint16" in (c.get("id") or "") and not c.get("expects_throw")],
                            "uint16", "char")
    if skipped[0]:
        print(f"  (skipped {skipped[0]} complex64 / object-dtype cells — #569 / no object dtype)")
    return cases


# ---- numpy.polynomial calculus family (plan docs/plans/numpy-polynomial.md U4) ----------------------------
#
# {p}der(c, m=1, scl=1, axis=0) and {p}int(c, m=1, k=[], lbnd=0, scl=1, axis=0) for the six bases. Op keys are
# module-qualified ("chebyshev.chebder"); arguments are recorded by NAME with the polyseries encoding (_ps_enc:
# "a" = the next operand, a 0-d array being NumPy's STRONG scalar; a Python scalar / list spec otherwise), except
# the two ints m and axis, written as plain JSON numbers (the C# facade takes them as ints). An absent argument is
# NumPy's default (the C# facade's null / default). A "facet": "flags" case records the RESULT's layout as the
# bool array [C_CONTIGUOUS, F_CONTIGUOUS, OWNDATA] instead of its values — NumPy returns moveaxis views of fresh
# C-order buffers, K-order copies (m == 0) and NpyIter-allocated c[:1]*0 arrays, and those layouts are part of
# the contract. The generator's value choices, section by section:
#   (A) every dtype x length x order at the defaults — the recurrences' j ranges, NumPy's `if n > 1` branches,
#       m == 0 (the copy), m >= len(c) (c[:1]*0), ints/bool -> float64;
#   (B) scl kinds — Python int/float/complex (weak: adopts the series dtype), 0-d arrays (strong: may widen,
#       then cast back same_kind), and the UFuncTypeError of a complex scl on a real series;
#   (C) array scl on derivatives (it broadcasts in place; NumPy's two broadcast error texts);
#   (D) integration constants: scalars, Python lists, typed arrays (NumPy scalars), 0-d arrays, N-D rows, and
#       the too-many / sequence / broadcast / cast errors;
#   (E) lbnd kinds (weak, strong 0-d, complex into a real series: 1-D keeps the real part, N-D raises);
#   (F) N-D series at every axis and memory layout, values and result flags;
#   (G) specials (NaN/inf/-0/subnormal) 1-D and N-D — 1-D runs scalarmath, N-D ufuncs;
#   (H) full-mantissa complex values, 1-D vs N-D: the naive (scalarmath) and fused (simd_cmul) products differ
#       in the last bit only on such operands;
#   (I) long series and wide N-D series — the kernel's vector loops, scalar tails and column blocks, and float16's
#       constant rounding (2*(j+1) > 2048) and overflow (j > 65504);
#   (J) argument errors in NumPy's check order; (K) Python-list series; (L) the n == 1 zero branch;
#   (M) C# boundary argument kinds — tuples, nested / ragged / empty lists, lists holding arrays, Python ints past
#       int64 (inf / nan / OverflowError), str k / lbnd timing, the ndim checks on sequences;
#   (N) zero-size and 5-D series (values and flags), extreme integers through the converting direct load;
#   (O) the kernel's widened c *= scl — float16 products just past a float16 tie, NaN / inf coefficients and scales,
#       a float16 series' exact float32 scale.
# complex64 and object results (a Python int past uint64 in a series) would be skipped (#569 / no object dtype).

def _pc_random(shape, dt, seed):
    """Seeded full-mantissa values (the product forms of complex arithmetic only differ on such operands)."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape)) if len(shape) else 1
    dt = np.dtype(dt)
    re = rng.uniform(-3.0, 3.0, n)
    if dt.kind == "c":
        re = re + 1j * rng.uniform(-3.0, 3.0, n)
    with np.errstate(all="ignore"):
        return np.ascontiguousarray(re.astype(dt).reshape(shape))


def _pc_special(shape, dt, seed=0):
    """Coefficients mixing NaN, +-inf, +-0, subnormals and ordinary values."""
    dt = np.dtype(dt)
    n = int(np.prod(shape)) if len(shape) else 1
    vals = [1.5, float("nan"), -0.0, float("inf"), 2.0, float("-inf"), 0.0, -0.75, 5e-324, 3.0, -1e-310, 0.5]
    re = np.array([vals[(i * 5 + seed) % len(vals)] for i in range(n)])
    if dt.kind == "c":
        im = np.array([vals[(i * 7 + seed + 3) % len(vals)] for i in range(n)])
        arr = re + 1j * im
    else:
        arr = re
    with np.errstate(all="ignore"):
        return np.ascontiguousarray(arr.astype(dt).reshape(shape))


def _pc_f16_hazards():
    """(scale, float16 coefficients) pairs whose float64 product lands just past a float16 TIE: NumPy's in-place
    multiply rounds it f64 -> f16 once, an f64 -> f32 -> f16 chain would round it to even. Every finite float16 is
    scanned against scales built to put the powers of two (and more) on such a band."""
    h = np.arange(0, 65536, dtype=np.uint16).view(np.float16)
    h = h[np.isfinite(h)]
    sets = []
    for s in (1 + 2.0 ** -11 + 2.0 ** -40, 1 + 2.0 ** -11 - 2.0 ** -40, 1.5 * (1 + 2.0 ** -12 + 2.0 ** -41),
              -(1 + 2.0 ** -11 + 2.0 ** -38), 0.75 * (1 + 2.0 ** -11 + 2.0 ** -40)):
        with np.errstate(all="ignore"):
            y = h.astype(np.float64) * s
            hazard = y.astype(np.float16).view(np.uint16) != y.astype(np.float32).astype(np.float16).view(np.uint16)
        found = h[hazard]
        if len(found) >= 8:
            pick = found[np.linspace(0, len(found) - 1, min(len(found), 37)).astype(int)]
            sets.append((s, np.concatenate([pick, np.array([1.5, -2.25, 0.0], np.float16)]).astype(np.float16)))
    return sets


def gen_polycalc():
    cases = []
    counter = [0]
    skipped = [0]
    ok_dtypes = set(ALL_DTYPES)
    A = _PSArr

    def emit(op, args, call, layout, cid, facet=None):
        operands = []
        params = {}
        # Canonical order: the C# replay decodes c, k, lbnd, scl in this order and consumes operands as it goes.
        for key in ("c", "m", "k", "lbnd", "scl", "axis"):
            if key in args:
                v = args[key]
                params[key] = v if key in ("m", "axis") else _ps_enc(v, operands)
        if facet:
            params["facet"] = facet
        n = counter[0]
        counter[0] += 1
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = call()
        except Exception as e:
            cases.append({"id": f"{cid}/{n}", "op": op, "params": params, "operands": operands,
                          "expected": {}, "expects_throw": True, "error": _poly_exc(e),
                          "layout": layout, "valueclass": "error"})
            return
        a = np.asarray(r)
        if facet == "flags":
            a = np.array([a.flags.c_contiguous, a.flags.f_contiguous, a.flags.owndata])
        if a.dtype.name not in ok_dtypes:
            skipped[0] += 1
            return
        cases.append(_case(op, params, operands, _arr_expected(a), layout, "polycalc", cid=f"{cid}/{n}"))

    float_dtypes = ("float64", "float32", "float16", "complex128")
    hazard_sets = _pc_f16_hazards()

    for modname, p in POLY_MODULES:
        mod = _poly_module(modname)
        for kind in ("der", "int"):
            f = getattr(mod, p + kind)
            op = f"{modname}.{p}{kind}"
            integ = kind == "int"

            # (A) dtype x length x order, default arguments
            for dt in ALL_DTYPES:
                inexact = np.dtype(dt).kind in "fc"
                lengths = (0, 1, 2, 3, 4, 5, 8, 13) if inexact else (0, 1, 3, 5, 13)
                for n in lengths:
                    orders = sorted({0, 1, 2, 3, max(n - 1, 0), n, n + 1}) if inexact else sorted({0, 1, 2, n})
                    c = _poly_coef((n,), dt, seed=11) if n else np.zeros(0, dt)
                    for m in orders:
                        emit(op, {"c": A(c), "m": m}, lambda: f(c, m), "c_contiguous_1d", f"{op}/dt/{dt}/{n}/{m}")

            # (B) scl kinds on 1-D series (and a few on N-D)
            weak_scl = [2, -1, 0, 3, True, 0.5, -1.25, 0.1, -0.0, float("nan"), float("inf"), 1.5j, complex(1, -2)]
            strong_scl = [np.array(0.5), np.array(0.1), np.array(0.1, np.float32), np.array(0.1, np.float16),
                          np.array(2, np.int8), np.array(True), np.array(1 + 1j), np.array(-3, np.int64)]
            for dt in float_dtypes:
                c = _poly_coef((6,), dt, seed=12)
                for m in (1, 2):
                    for s in weak_scl:
                        emit(op, {"c": A(c), "m": m, "scl": s}, lambda: f(c, m, scl=s), "c_contiguous_1d",
                             f"{op}/sclw/{dt}/{m}/{s!r}")
                    for s in strong_scl:
                        emit(op, {"c": A(c), "m": m, "scl": A(s)}, lambda: f(c, m, scl=s), "c_contiguous_1d",
                             f"{op}/scls/{dt}/{m}/{s.dtype}/{s.item()!r}")
                c2 = _poly_coef((4, 3), dt, seed=13)
                for s in (0.5, 0.1, 1.5j, np.array(0.1), np.array(0.1, np.float16), np.array(1 + 1j)):
                    sv = A(s) if isinstance(s, np.ndarray) else s
                    emit(op, {"c": A(c2), "m": 2, "scl": sv}, lambda: f(c2, 2, scl=s), "c_contiguous_2d",
                         f"{op}/scl2d/{dt}/{s!r}")

            # (C) array scl on derivatives (broadcast in place, per order); the int form rejects it
            c2 = _poly_coef((5, 3), "float64", seed=14)
            c1 = _poly_coef((5,), "float64", seed=15)
            for sname, s, cc, m in (("col", np.array([2.0, -0.5, 3.0]), c2, 2), ("row1", np.array([[2.0, 0.5, -1.0]]), c2, 1),
                                    ("perrow_m1", np.array([[2.0], [3.0], [4.0], [5.0], [6.0]]), c2, 1),
                                    ("perrow_m2", np.array([[2.0], [3.0], [4.0], [5.0], [6.0]]), c2, 2),
                                    ("bad", np.array([1.0, 2.0]), c2, 1), ("stretch", np.ones((2, 5, 3)), c2, 1),
                                    ("vec1d_m1", np.array([2.0, 3.0, 4.0, 5.0, 6.0]), c1, 1),
                                    ("vec1d_m2", np.array([2.0, 3.0, 4.0, 5.0, 6.0]), c1, 2),
                                    ("complex_col", np.array([1j, 1.0, 2.0]), c2, 1),
                                    ("f32_col", np.array([0.1, 0.2, 0.3], np.float32), c2, 1)):
                emit(op, {"c": A(cc), "m": m, "scl": A(s)}, lambda: f(cc, m, scl=s), "scl_array",
                     f"{op}/sclarr/{sname}")

            if integ:
                # (D) integration constants
                for dt in float_dtypes:
                    c = _poly_coef((4,), dt, seed=16)
                    for kname, kv in (("int", 3), ("float", 2.5), ("complex", 1j), ("list1", [1]), ("list2", [1, 2]),
                                      ("list3", [1, 2, 3]), ("listf", [1.5, -2.5]), ("listc", [0.5j, 1]),
                                      ("f64arr", np.array([1.0, 2.0])), ("i8arr", np.array([1, 2], np.int8)),
                                      ("f32arr", np.array([0.1], np.float32)), ("c128arr", np.array([1 + 1j])),
                                      ("zerod", np.array(5.0)), ("zerod32", np.array(0.1, np.float32)),
                                      ("toomany", [1, 2, 3, 4]), ("seq", [np.array([1.0])]), ("seq0d", [np.array(2.0)]),
                                      ("bool", True), ("bigint", 2 ** 40)):
                        kenc = A(kv) if isinstance(kv, np.ndarray) else ([A(e) if isinstance(e, np.ndarray) else e for e in kv]
                                                                        if isinstance(kv, list) else kv)
                        for m in (1, 2, 3):
                            emit(op, {"c": A(c), "m": m, "k": kenc}, lambda: f(c, m, k=kv), "c_contiguous_1d",
                                 f"{op}/k/{dt}/{kname}/{m}")
                for dt in ("float64", "complex128", "float32"):
                    c = _poly_coef((4, 3), dt, seed=17)
                    for kname, kv in (("rows_list", [[1, 2, 3]]), ("rows_arr", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                                      ("scalar", 2.5), ("arr_bcast", [np.array([1.0, 2.0])]), ("arr_2d", [np.ones((2, 3))]),
                                      ("complex", [1j]), ("complex_arr", [np.array([1j, 2, 3])]), ("f64", [np.array(0.1)])):
                        kenc = A(kv) if isinstance(kv, np.ndarray) else ([A(e) if isinstance(e, np.ndarray) else e for e in kv]
                                                                        if isinstance(kv, list) else kv)
                        for m in (1, 2):
                            emit(op, {"c": A(c), "m": m, "k": kenc}, lambda: f(c, m, k=kv), "c_contiguous_2d",
                                 f"{op}/k2d/{dt}/{kname}/{m}")

                # (E) lbnd kinds
                lbnds = [-1, 0.5, -0.0, 2.5, float("inf"), float("nan"), 1j, complex(0.5, -1),
                         np.array(0.3), np.array(0.3, np.float32), np.array(0.5, np.float16), np.array(1j), np.array([0.0])]
                for dt in float_dtypes:
                    for shape in ((4,), (4, 3)):
                        c = _poly_coef(shape, dt, seed=18)
                        for lb in lbnds:
                            lenc = A(lb) if isinstance(lb, np.ndarray) else lb
                            for m, kv in ((1, None), (2, [1, -2])):
                                args = {"c": A(c), "m": m, "lbnd": lenc}
                                if kv is not None:
                                    args["k"] = kv
                                emit(op, args, lambda: f(c, m, k=kv if kv is not None else [], lbnd=lb),
                                     f"c_contiguous_{len(shape)}d", f"{op}/lbnd/{dt}/{shape}/{lb!r}/{m}")
                # scl must be a scalar for integrals
                c = _poly_coef((4,), "float64", seed=19)
                emit(op, {"c": A(c), "m": 1, "scl": A(np.array([2.0]))}, lambda: f(c, 1, scl=np.array([2.0])),
                     "c_contiguous_1d", f"{op}/err/sclarr")

            # (F) N-D series: every axis, memory layouts, values + result flags
            # Values for float64 and complex128 (int32 on the C/F views: the converting direct load), flags for
            # float64 — the layout does not depend on the dtype.
            for shape in ((6, 3), (3, 6), (4, 3, 2), (2, 5, 3), (1, 4), (4, 1)):
                for dt in ("float64", "complex128", "int32"):
                    base = _poly_coef(shape, dt, seed=20)
                    views = [("c", base, base)]
                    # describe() serializes the BASE in C order, so every base must be C-contiguous and the layout
                    # expressed as the view: F order = the transpose of a C buffer holding base.T.
                    fb = np.ascontiguousarray(base.T)
                    views.append(("f", fb, fb.T))
                    if len(shape) >= 3:
                        perm = (1, 0) + tuple(range(2, len(shape)))
                        tb = np.ascontiguousarray(base.transpose(perm))
                        views.append(("t", tb, tb.transpose(perm)))
                    if len(shape) >= 2:
                        wide = np.zeros(shape[:-1] + (2 * shape[-1],), dt)
                        wide[..., ::2] = base
                        views.append(("strided", wide, wide[..., ::2]))
                        rb = np.ascontiguousarray(base[::-1])
                        views.append(("reversed", rb, rb[::-1]))
                    for vname, vb, vv in views:
                        if dt == "int32" and vname not in ("c", "f"):
                            continue
                        for ax in range(-1, len(shape)):
                            n = shape[ax]
                            for m in sorted({1, n}):
                                layout = f"nd_{vname}"
                                args = {"c": A(vb, vv), "m": m, "axis": ax}
                                emit(op, args, lambda: f(vv, m, axis=ax), layout, f"{op}/nd/{shape}/{dt}/{vname}/{ax}/{m}")
                                if dt == "float64":
                                    emit(op, args, lambda: f(vv, m, axis=ax), layout,
                                         f"{op}/ndflags/{shape}/{dt}/{vname}/{ax}/{m}", facet="flags")
                            if dt == "float64":
                                emit(op, {"c": A(vb, vv), "m": 0, "axis": ax}, lambda: f(vv, 0, axis=ax), f"nd_{vname}",
                                     f"{op}/ndflags0/{shape}/{dt}/{vname}/{ax}", facet="flags")
            # broadcast (stride 0) series and a 0-d series
            src = _poly_coef((1, 3), "float64", seed=21)
            bc = np.broadcast_to(src, (5, 3))
            for ax in (0, 1):
                for m in (1, 2, 5):
                    emit(op, {"c": A(src, bc), "m": m, "axis": ax}, lambda: f(bc, m, axis=ax), "nd_broadcast",
                         f"{op}/ndbc/{ax}/{m}")
                    emit(op, {"c": A(src, bc), "m": m, "axis": ax}, lambda: f(bc, m, axis=ax), "nd_broadcast",
                         f"{op}/ndbcflags/{ax}/{m}", facet="flags")
            src1 = _poly_coef((1,), "float64", seed=22)
            bc1 = np.broadcast_to(src1, (6,))
            emit(op, {"c": A(src1, bc1), "m": 2}, lambda: f(bc1, 2), "broadcast_1d", f"{op}/bc1d")
            z0 = np.array(2.5)
            for m in (0, 1, 2):
                emit(op, {"c": A(z0), "m": m}, lambda: f(z0, m), "scalar_0d", f"{op}/zerod/{m}")
            for ax in (0, -1, 1, -2):
                emit(op, {"c": A(z0), "m": 1, "axis": ax}, lambda: f(z0, 1, axis=ax), "scalar_0d", f"{op}/zerod_ax/{ax}")

            # (G) special values, 1-D (scalarmath) and N-D (ufuncs)
            for dt in float_dtypes:
                for shape in ((7,), (7, 1), (7, 3)):
                    c = _pc_special(shape, dt, seed=3)
                    for m in (1, 2):
                        emit(op, {"c": A(c), "m": m}, lambda: f(c, m), f"special_{len(shape)}d", f"{op}/spec/{dt}/{shape}/{m}")
                        emit(op, {"c": A(c), "m": m, "scl": 0.5}, lambda: f(c, m, scl=0.5), f"special_{len(shape)}d",
                             f"{op}/specs/{dt}/{shape}/{m}")
                        if integ:
                            kv = [1.5, -0.0][:m]
                            emit(op, {"c": A(c), "m": m, "k": kv, "lbnd": 0.5}, lambda: f(c, m, k=kv, lbnd=0.5),
                                 f"special_{len(shape)}d", f"{op}/speck/{dt}/{shape}/{m}")

            # (H) full-mantissa complex values: 1-D scalarmath (naive product) vs N-D ufuncs (simd_cmul)
            for shape in ((6,), (9,), (6, 1), (6, 2), (9, 5)):
                for draw in range(3):
                    c = _pc_random(shape, "complex128", 900 + 31 * draw + len(shape))
                    for m in (1, 2):
                        emit(op, {"c": A(c), "m": m}, lambda: f(c, m), f"cfull_{len(shape)}d", f"{op}/cfull/{shape}/{draw}/{m}")
                        s = complex(*_pc_random((2,), "float64", 1000 + draw))
                        emit(op, {"c": A(c), "m": m, "scl": s}, lambda: f(c, m, scl=s), f"cfull_{len(shape)}d",
                             f"{op}/cfulls/{shape}/{draw}/{m}")
                        if integ:
                            lb = float(_pc_random((1,), "float64", 1100 + draw)[0])
                            emit(op, {"c": A(c), "m": m, "lbnd": lb, "k": [1.25, -0.5][:m]},
                                 lambda: f(c, m, k=[1.25, -0.5][:m], lbnd=lb), f"cfull_{len(shape)}d",
                                 f"{op}/cfullk/{shape}/{draw}/{m}")

            # (I) long series and wide N-D series (vector loops, tails, column blocks)
            for dt in float_dtypes:
                for n in (64, 257):
                    c = _pc_random((n,), dt, 1200 + n)
                    for m in (1, 3):
                        emit(op, {"c": A(c), "m": m}, lambda: f(c, m), "long_1d", f"{op}/long/{dt}/{n}/{m}")
                # widths around every lane count (W = 2, 4, 8): vector loop only, loop + tail, tail only
                for cols in (1, 3, 7, 8, 9, 45):
                    c = _pc_random((6, cols), dt, 1300 + cols)
                    for m in (1, 2):
                        emit(op, {"c": A(c), "m": m}, lambda: f(c, m), "wide_2d", f"{op}/wide/{dt}/{cols}/{m}")
                        if integ and cols in (3, 9, 45):
                            emit(op, {"c": A(c), "m": m, "k": [0.5, 1.5][:m], "lbnd": -0.75},
                                 lambda: f(c, m, k=[0.5, 1.5][:m], lbnd=-0.75), "wide_2d", f"{op}/widek/{dt}/{cols}/{m}")
            c = _pc_random((1000,), "float64", 2200)
            emit(op, {"c": A(c), "m": 2}, lambda: f(c, 2), "long_1d", f"{op}/long/float64/1000/2")
            if p in ("cheb", "leg"):
                # More columns than one kernel block (4096) — the block loop and its partial last block. The loop is
                # the same for every basis, so two bases carry it.
                cb = _pc_random((3, 4200), "float32", 1400)
                emit(op, {"c": A(cb), "m": 2}, lambda: f(cb, 2), "blocks_2d", f"{op}/blocks/float32")
                cb = _pc_random((2, 4200), "complex128", 1401)
                emit(op, {"c": A(cb), "m": 1}, lambda: f(cb, 1), "blocks_2d", f"{op}/blocks/complex128")

            # (J) argument errors, NumPy's check order
            c = _poly_coef((4,), "float64", seed=23)
            emit(op, {"c": A(c), "m": -1}, lambda: f(c, -1), "c_contiguous_1d", f"{op}/err/mneg")
            emit(op, {"c": A(c), "m": 1, "axis": 1}, lambda: f(c, 1, axis=1), "c_contiguous_1d", f"{op}/err/axis")
            emit(op, {"c": A(c), "m": 1, "axis": -2}, lambda: f(c, 1, axis=-2), "c_contiguous_1d", f"{op}/err/axisneg")
            emit(op, {"c": A(c), "m": -1, "axis": 5}, lambda: f(c, -1, axis=5), "c_contiguous_1d", f"{op}/err/order")
            emit(op, {"c": A(c), "m": 0, "axis": 5}, lambda: f(c, 0, axis=5), "c_contiguous_1d", f"{op}/err/axism0")
            emit(op, {"c": A(c), "m": 1, "scl": 1j}, lambda: f(c, 1, scl=1j), "c_contiguous_1d", f"{op}/err/sclc")
            emit(op, {"c": A(c), "m": 9, "scl": 1j}, lambda: f(c, 9, scl=1j), "c_contiguous_1d", f"{op}/err/sclc_big")
            e0 = np.zeros(0)
            emit(op, {"c": A(e0), "m": 1, "scl": 1j}, lambda: f(e0, 1, scl=1j), "c_contiguous_1d", f"{op}/err/sclc_empty")
            if integ:
                emit(op, {"c": A(c), "m": -1, "k": [1, 2]}, lambda: f(c, -1, k=[1, 2]), "c_contiguous_1d", f"{op}/err/mneg_k")
                emit(op, {"c": A(c), "m": 1, "k": [1, 2], "lbnd": A(np.array([0.0]))},
                     lambda: f(c, 1, k=[1, 2], lbnd=np.array([0.0])), "c_contiguous_1d", f"{op}/err/k_before_lbnd")
                emit(op, {"c": A(c), "m": 1, "lbnd": A(np.array([0.0])), "scl": A(np.array([1.0]))},
                     lambda: f(c, 1, lbnd=np.array([0.0]), scl=np.array([1.0])), "c_contiguous_1d", f"{op}/err/lbnd_before_scl")
                emit(op, {"c": A(c), "m": 1, "scl": A(np.array([1.0])), "axis": 3},
                     lambda: f(c, 1, scl=np.array([1.0]), axis=3), "c_contiguous_1d", f"{op}/err/scl_before_axis")
                emit(op, {"c": A(c), "m": 0, "k": [1]}, lambda: f(c, 0, k=[1]), "c_contiguous_1d", f"{op}/err/k_m0")

            # (K) Python-list series
            for lname, lv in (("ints", [1, 2, 3]), ("mixed", [1, 2.5, -3]), ("complex", [1j, 2]), ("bools", [True, False, True]),
                              ("nested", [[1, 2], [3, 4], [5, 6]]), ("empty", []), ("scalar", 5), ("fscalar", 2.5)):
                for m in (0, 1, 2):
                    emit(op, {"c": lv, "m": m}, lambda: f(lv, m), "python_list", f"{op}/list/{lname}/{m}")

            # (L) the n == 1 branch of integrals (and one-coefficient derivatives)
            for cname, cv in (("zero", np.zeros(1)), ("negzero", np.array([-0.0])), ("nan", np.array([np.nan])),
                              ("czero", np.zeros(1, complex)), ("two", np.array([2.0])), ("zeros_row", np.zeros((1, 3))),
                              ("mixed_row", np.array([[0.0, 1.0, 0.0]])), ("f16zero", np.zeros(1, np.float16))):
                for m, kv, s in ((1, None, None), (2, [0, 3], None), (3, [0, 0, 3], None), (2, [3], None), (1, [0], 0),
                                 (2, [1, 2], 0), (3, [0, 1], None)):
                    if not integ and kv is not None:
                        continue
                    args = {"c": A(cv), "m": m}
                    kw = {}
                    if kv is not None:
                        args["k"] = kv
                        kw["k"] = kv
                    if s is not None:
                        args["scl"] = s
                        kw["scl"] = s
                    emit(op, args, lambda: f(cv, m, **kw), "one_coef", f"{op}/one/{cname}/{m}/{kv}/{s}")
                    emit(op, args, lambda: f(cv, m, **kw), "one_coef", f"{op}/oneflags/{cname}/{m}/{kv}/{s}", facet="flags")
            for (shape, ax) in (((1, 3, 2), 0), ((3, 1, 2), 1), ((2, 3, 1), 2)):
                zb = np.zeros(shape[::-1])      # C buffer; its transpose is the F-ordered series
                zf = zb.T
                for m, kv in ((1, [3]), (2, [0, 3]), (2, [3, 4])):
                    if not integ:
                        kv = None
                    args = {"c": A(zb, zf), "m": m, "axis": ax}
                    if kv is not None:
                        args["k"] = kv
                    call = (lambda: f(zf, m, k=kv, axis=ax)) if kv is not None else (lambda: f(zf, m, axis=ax))
                    emit(op, args, call, "one_coef_nd", f"{op}/onend/{shape}/{ax}/{m}/{kv}")
                    emit(op, args, call, "one_coef_nd", f"{op}/onendflags/{shape}/{ax}/{m}/{kv}", facet="flags")

            # (M) C# boundary argument kinds. A Python tuple replays as a ValueTuple, a list as object[], a big Python int as
            #     ulong / BigInteger, a NumPy scalar item as a 0-d array (the same dtype discovery). np.array's coercion walk
            #     (NDPolySequence): nesting to any depth, dtype discovery over every leaf, NumPy's ragged texts (the shape the
            #     walk still agreed on), an empty sequence ending the dims, and the ndim checks; k's items converted only when
            #     their order uses them (after that order's {p}val), a str lbnd refused only where NumPy evaluates at it.
            f32a = np.array([1.0, 2.0], np.float32)
            f32b = np.array([3.0, 4.0], np.float32)
            c_kinds = [
                ("tuple", (1.0, 2.0, 3.0)), ("tuple_int", (1, 2, 3)), ("tuple_nested", ((1, 2), (3, 4))),
                ("mixed_list_tuple", [(1, 2), [3, 4]]),
                ("nested3", [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]),
                ("tuple_arrays_f32", (A(f32a), A(f32b))), ("list_array_tuple", [A(f32a), (3, 4)]),
                ("list_f16_0d", [A(np.array(1.0, np.float16)), A(np.array(2.5, np.float16))]),
                ("list_f16_py", [A(np.array(1.0, np.float16)), 2.0]), ("bool_int_rows", [[True, False], [2, 3]]),
                ("complex_rows", [[1j, 2], [3, 4 - 1j]]), ("u64", [2 ** 64 - 1, 1]), ("u64_neg", [2 ** 63, -1]),
                ("empty", []), ("empty_tuple", ()), ("empties", [[], []]), ("empties_tuple", ((), ())),
                ("empty_deep", [[[]], [[]]]), ("empty_f32", [A(np.zeros(0, np.float32)), []]),
                ("ragged", [[1], [1, 2]]), ("ragged_deep", [[[1], [1, 2]], [[1], [1, 2]]]), ("ragged_depth", [[1], [[2]]]),
                ("ragged_scalar_seq", [1, [2, 3]]), ("ragged_seq_scalar", [[2, 3], 1]), ("ragged_empty", [[], [1]]),
                ("ragged_empty2", [[1], []]), ("ragged_arrays", [A(np.zeros(2)), A(np.zeros(3))]),
                ("ragged_array_scalar", [A(np.zeros(2)), 5]), ("empty_broadcast", [A(np.zeros((0, 3))), []]),
            ]
            for cname, cv in c_kinds:
                try:
                    nd = np.asarray(_ps_py(cv)).ndim
                except Exception:
                    nd = 1                      # ragged: axis 0 only (the conversion raises before the axis matters)
                for m in (1, 2):
                    for ax in sorted({0, nd - 1}):
                        emit(op, {"c": cv, "m": m, "axis": ax}, lambda: f(_ps_py(cv), m, axis=ax), "arg_kinds",
                             f"{op}/argc/{cname}/{m}/{ax}")
            big_ints = [("big20", 2 ** 20), ("big200", 2 ** 200), ("bigneg70", -(2 ** 70)), ("u64", 2 ** 64 - 1),
                        ("overflow", 2 ** 1030)]
            for sname, s in [("tuple1", (2.0,)), ("list1", [2.0]), ("nested_col", [[2.0], [3.0], [4.0]]),
                             ("tuple_row", ((2.0, 3.0),)), ("ragged", [[1], [1, 2]])] + big_ints:
                for dt in float_dtypes:
                    for shape in ((3,), (3, 2)):
                        cc = _poly_coef(shape, dt, seed=28)
                        for m in (1, 5):       # 5 >= len(c): a derivative returns before it scales
                            emit(op, {"c": A(cc), "m": m, "scl": s}, lambda: f(cc, m, scl=s), "arg_kinds",
                                 f"{op}/argscl/{sname}/{dt}/{shape}/{m}")
            if integ:
                c1 = _poly_coef((3,), "float64", seed=25)
                c2 = _poly_coef((3, 2), "float64", seed=24)
                c3 = _poly_coef((3, 2, 2), "float64", seed=26)
                for kname, cc, m, kv in (
                        ("tuple", c1, 2, (1, 2)), ("tuple1", c1, 1, (0.25,)), ("tuple_mixed", c1, 3, (1, 2.5, 1j)),
                        ("tuple_empty", c1, 2, ()), ("tuple_toomany", c1, 1, (1, 2)),
                        ("tuple_rows", c2, 2, ((1, 2), (3, 4))), ("list_of_tuples", c2, 2, [(1, 2), (3, 4)]),
                        ("tuple_row", c2, 1, ((1.0, 2.0),)), ("tuple_scalars_nd", c2, 2, (1, 2)),
                        ("nested3", c3, 1, [[[1, 2], [3, 4]]]), ("big70", c1, 1, 2 ** 70),
                        ("big_list", c1, 2, [2 ** 200, -(2 ** 70)]), ("overflow", c1, 1, 2 ** 1030), ("u64", c1, 1, 2 ** 64 - 1),
                        ("ragged_item", c2, 1, [[[1], [1, 2]]]), ("ragged_item_mneg", c2, -1, [[[1], [1, 2]]]),
                        ("ragged_item_toomany", c2, 1, [[[1], [1, 2]], 1]), ("ragged_second", c2, 2, [[1, 2], [[1], [1, 2]]]),
                        ("str_mneg", c1, -1, "ab"), ("str_toomany", c1, 1, "ab"), ("str_m0", c1, 0, "")):
                    emit(op, {"c": A(cc), "m": m, "k": kv}, lambda: f(cc, m, k=kv), "arg_kinds", f"{op}/argk/{kname}")
                # the order's lbnd is evaluated before its constant is converted
                emit(op, {"c": A(c2), "m": 1, "k": [[[1], [1, 2]]], "lbnd": 2 ** 1030},
                     lambda: f(c2, 1, k=[[[1], [1, 2]]], lbnd=2 ** 1030), "arg_kinds", f"{op}/argk/ragged_after_lbnd")
                for lname, lb in [("tuple", (0,)), ("list", [0.5]), ("ragged", [[1], [1, 2]]), ("empty_tuple", ())] + big_ints:
                    for dt in float_dtypes:
                        for shape in ((3,), (3, 2)):
                            cc = _poly_coef(shape, dt, seed=27)
                            emit(op, {"c": A(cc), "m": 1, "lbnd": lb}, lambda: f(cc, 1, lbnd=lb), "arg_kinds",
                                 f"{op}/arglbnd/{lname}/{dt}/{shape}")
                emit(op, {"c": A(c1), "m": 1, "lbnd": [[1], [1, 2]], "scl": [1]},
                     lambda: f(c1, 1, lbnd=[[1], [1, 2]], scl=[1]), "arg_kinds", f"{op}/arglbnd/ragged_before_scl")
                # A str lbnd: np.ndim('a') is 0, and NumPy only fails where it EVALUATES at it — never on m == 0, never in
                # the one-coefficient zero branch, never before a later argument's own error.
                z1 = np.zeros(1)
                for lname, cc, m, extra in (("str_m0", c1, 0, {}), ("str_zero1", z1, 1, {}), ("str_zero2", z1, 2, {}),
                                            ("str_sclarr", c1, 1, {"scl": [1]}), ("str_axis", c1, 1, {"axis": 3})):
                    args = {"c": A(cc), "m": m, "lbnd": "a"}
                    args.update(extra)
                    emit(op, args, lambda: f(cc, m, lbnd="a", **extra), "arg_kinds", f"{op}/arglbnd/{lname}")

            # (N) zero-size and 5-D series, and extreme integers through the converting direct load (vector lanes + tail)
            for shape in ((3, 0), (0, 3), (1, 0), (3, 0, 2), (2, 0, 0)):
                z = np.zeros(shape)
                for ax in range(len(shape)):
                    for m in (1, 2):
                        emit(op, {"c": A(z), "m": m, "axis": ax}, lambda: f(z, m, axis=ax), "zero_size",
                             f"{op}/zsize/{shape}/{ax}/{m}")
                        emit(op, {"c": A(z), "m": m, "axis": ax}, lambda: f(z, m, axis=ax), "zero_size",
                             f"{op}/zsizeflags/{shape}/{ax}/{m}", facet="flags")
                        if integ:
                            kv = [1, 2][:m]
                            emit(op, {"c": A(z), "m": m, "k": kv, "axis": ax}, lambda: f(z, m, k=kv, axis=ax), "zero_size",
                                 f"{op}/zsizek/{shape}/{ax}/{m}")
            for dt in ("float64", "complex128"):
                c5 = _pc_random((2, 3, 1, 2, 3), dt, 2300)
                for ax in range(-1, 5):
                    for m in (1, 2):
                        emit(op, {"c": A(c5), "m": m, "axis": ax}, lambda: f(c5, m, axis=ax), "rank5", f"{op}/rank5/{dt}/{ax}/{m}")
                        if dt == "float64":
                            emit(op, {"c": A(c5), "m": m, "axis": ax}, lambda: f(c5, m, axis=ax), "rank5",
                                 f"{op}/rank5flags/{ax}/{m}", facet="flags")
            for dt in ("int64", "uint64", "int32", "uint32", "int16", "uint16", "int8", "uint8"):
                info = np.iinfo(dt)
                vals = [int(info.min), int(info.max), int(info.max) - 1, int(info.min) + 1, 0, 1,
                        int(info.max) // 3, int(info.min) // 3]
                c1 = np.array([vals[i % len(vals)] for i in range(19)], dt)
                c2 = np.array([[vals[(i * 3 + j) % len(vals)] for j in range(9)] for i in range(7)], dt)
                for m in (1, 2):
                    emit(op, {"c": A(c1), "m": m}, lambda: f(c1, m), "extreme_int", f"{op}/xint/{dt}/1d/{m}")
                    for ax in (0, 1):
                        emit(op, {"c": A(c2), "m": m, "axis": ax}, lambda: f(c2, m, axis=ax), "extreme_int",
                             f"{op}/xint/{dt}/2d/{ax}/{m}")

            # (O) the kernel's WIDENED c *= scl — a strong scalar promoting a float16 / float32 series runs NumPy's wider loop
            #     and casts back: float16 products just past a float16 tie (rounded f64 -> f16 once), NaN / inf coefficients
            #     and scales (the lane helpers' per-lane NaN route, the series' NaN winning), and a float16 series' float32
            #     loop, whose scale is the EXACT float32 value (int16 2049 and float32 0.1 are not float16 values). 1-D series
            #     run the scalar tails, (3, n) series the vector lanes.
            for hs, hc in hazard_sets:
                for shape in ((len(hc),), (3, len(hc))):
                    c = np.ascontiguousarray(np.broadcast_to(hc, shape) if len(shape) == 2 else hc)
                    for m in (1, 2):
                        sa = np.array(hs)
                        emit(op, {"c": A(c), "m": m, "scl": A(sa)}, lambda: f(c, m, scl=sa), "widened_scale",
                             f"{op}/widen/f16tie/{hs!r}/{len(shape)}d/{m}")
            for dt, scls in (("float16", (np.array(0.75), np.array(np.nan), np.array(-np.inf), np.array(3, np.int16),
                                          np.array(2049, np.int16), np.array(0.1, np.float32), np.array(70000, np.int32))),
                             ("float32", (np.array(0.75), np.array(np.nan), np.array(1e300), np.array(-3, np.int64),
                                          np.array(2 ** 40 + 1, np.int64)))):
                for shape in ((13,), (3, 13)):
                    c = _pc_special(shape, dt, seed=5)
                    for sa in scls:
                        for m in (1, 2):
                            emit(op, {"c": A(c), "m": m, "scl": A(sa)}, lambda: f(c, m, scl=sa), "widened_scale",
                                 f"{op}/widen/{dt}/{sa.dtype}/{sa.item()!r}/{len(shape)}d/{m}")

        # float16 constants past float16's exact integers (2*(j+1) > 2048: the Python int rounds to nearest-even) for
        # every basis, and past its range (a constant > 65504 is inf) where a basis reaches it at a corpus-sized length:
        # hermder's 2*j at j = 32753, polyint's j + 1 at j = 65504.
        for kind, n in (("der", 2100), ("int", 2100)):
            f = getattr(mod, p + kind)
            op = f"{modname}.{p}{kind}"
            c = np.ones(n, np.float16)
            emit(op, {"c": A(c), "m": 1}, lambda: f(c, 1), "long_1d", f"{op}/f16range/{n}")
        for kind, n in ({"herm": (("der", 33000),), "poly": (("int", 66000),)}.get(p, ())):
            f = getattr(mod, p + kind)
            op = f"{modname}.{p}{kind}"
            c = np.ones(n, np.float16)
            emit(op, {"c": A(c), "m": 1}, lambda: f(c, 1), "long_1d", f"{op}/f16range/{n}")

    # Char: NumSharp's uint16-like dtype converts to float64 exactly as uint16 does (the house weave).
    cases += _relabel_dtype([c for c in cases if "/dt/uint16/" in (c.get("id") or "") and not c.get("expects_throw")],
                            "uint16", "char")
    if skipped[0]:
        print(f"  (skipped {skipped[0]} complex64 cells — #569)")
    return cases


# ---- numpy.polynomial Vandermonde family (plan docs/plans/numpy-polynomial.md U5) ---------------------------------
#
# Op keys "<basis module>.<p>vander" / "<p>vander2d" / "<p>vander3d". Arguments by NAME in call order — x (the 1-D form)
# or x, y[, z] (2-D / 3-D), then deg — each a _ps_enc spec: "a" (the next operand, any layout; a 0-d one is STRONG), a
# Python scalar / list / tuple / str / None, or an np.float16 (C#'s Half). Results: the matrix (dtype, shape, bytes), or
# with "facet": "flags" its [C_CONTIGUOUS, F_CONTIGUOUS, OWNDATA] (NumPy's moveaxis / reshape views: OWNDATA is False).
# Sections:
#   (A) dtype x length x degree, every basis, 1-D contiguous x (the load stage's conversions, the vector loops' tails);
#   (B) full-mantissa random values (complex128's simd_cmul and Smith forms, float16's HALF loops);
#   (C) special values — quiet / signalling / negative NaNs, +-inf, +-0, subnormals, the largest finite values;
#   (D) memory layouts of x (1-D strided / reversed / offset / broadcast / 0-d; N-D C / F / transposed / strided /
#       reversed) — values, and the result's flags (plus zero-size and size-1 shapes);
#   (E) float16's Python-int constants past its exact integers (rounded to even) and past its range (inf);
#   (F) 2-D / 3-D: every dtype (and mixed pairs — np.asarray's strong promotion), point shapes 0-d / 1-D / N-D, layouts
#       (each point read in place with its own strides), degree combinations, flags, Python-typed points (stacked);
#   (G) argument kinds and errors in NumPy's order: deg kinds (operator.index, and the f-string text of a refused value),
#       x kinds (Python sequences), deg-before-x, 2-D / 3-D deg containers (len(), the count), points stacking (ragged),
#       empty points (the reshape error), huge degrees (npy_intp, "array is too big");
#   (H) Char (the uint16 proxy, relabelled);
#   (I) inputs longer than one kernel block (the block loop, the partial last block, the scratch reuse).
# Not recorded: complex64 x (NumSharp has one complex width, #569), object results (a Python int past uint64), NumPy's
# str / object `x + 0.0` failures (NumSharp refuses str / object arrays — unit-tested), MemoryError (its text and even
# its occurrence depend on the machine) and degrees that make NumPy's Python loop run for hours (2**40 over empty x).

def _pv_special(dt):
    """Points mixing quiet / signalling / negative-payload NaNs, +-inf, +-0, subnormals, the largest finite value and
    ordinary values — as bit patterns, so the signalling NaN survives into the operand."""
    dt = np.dtype(dt)
    table = {
        "float64": (np.uint64, [0x7ff8000000000000, 0x7ff4000000000001, 0xfff8000000000123, 0x7ff0000000000000,
                                0xfff0000000000000, 0, 0x8000000000000000, 1, 0x800fffffffffffff, 0x7fefffffffffffff,
                                0x3fe0000000000000, 0xc008000000000000, 0x3ff8000000000000]),
        "float32": (np.uint32, [0x7fc00000, 0x7fa00001, 0xffc00123, 0x7f800000, 0xff800000, 0, 0x80000000, 1, 0x807fffff,
                                0x7f7fffff, 0x3f000000, 0xc0400000, 0x3fc00000]),
        "float16": (np.uint16, [0x7e00, 0x7d01, 0xfe23, 0x7c00, 0xfc00, 0, 0x8000, 1, 0x83ff, 0x7bff, 0x3800, 0xc200, 0x3e00]),
    }
    if dt.kind == "c":
        re = _pv_special("float64")
        c = np.zeros(len(re), np.complex128)
        # Bit copies (no arithmetic): the payloads, the signalling NaN and the signed zeros reach the operand as is.
        c.view(np.float64)[0::2] = re
        c.view(np.float64)[1::2] = np.roll(re, 4)
        return c
    ut, bits = table[dt.name]
    return np.array(bits, dtype=ut).view(dt).copy()


def _pv_refused(e):
    """NumPy failures NumSharp answers with its own refusal (not recorded): `x + 0.0` on a str array (UFuncTypeError)
    or an object array holding None / a non-numeric object / a str (the object stack's per-element add), and
    MemoryError."""
    t = type(e).__name__
    m = str(e)
    return (t in ("UFuncTypeError", "MemoryError") or m.startswith("unsupported operand type(s) for +")
            or m.startswith("can only concatenate str"))


def gen_polyvander():
    cases = []
    counter = [0]
    skipped = [0]
    ok_dtypes = set(ALL_DTYPES)
    A = _PSArr

    def emit(op, args, call, layout, cid, facet=None):
        operands = []
        params = {}
        # Canonical order: the C# replay decodes x, y, z, deg in this order and consumes operands as it goes.
        for key in ("x", "y", "z", "deg"):
            if key in args:
                params[key] = _ps_enc(args[key], operands)
        if facet:
            params["facet"] = facet
        n = counter[0]
        counter[0] += 1
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = call()
        except Exception as e:
            if _pv_refused(e):
                skipped[0] += 1
                return
            cases.append({"id": f"{cid}/{n}", "op": op, "params": params, "operands": operands,
                          "expected": {}, "expects_throw": True, "error": _poly_exc(e),
                          "layout": layout, "valueclass": "error"})
            return
        a = np.asarray(r)
        if facet == "flags":
            a = np.array([a.flags.c_contiguous, a.flags.f_contiguous, a.flags.owndata])
        if a.dtype.name not in ok_dtypes:
            skipped[0] += 1
            return
        cases.append(_case(op, params, operands, _arr_expected(a), layout, "polyvander", cid=f"{cid}/{n}"))

    def nd_views(base):
        """(name, base, view) layouts of an N-D array's values (describe() serializes the BASE in C order, so every base
        is C-contiguous and the layout lives in the view: F order is the transpose of a C buffer holding base.T)."""
        shape = base.shape
        out = [("c", base, base)]
        fb = np.ascontiguousarray(base.T)
        out.append(("f", fb, fb.T))
        if len(shape) >= 3:
            perm = (1, 0) + tuple(range(2, len(shape)))
            tb = np.ascontiguousarray(base.transpose(perm))
            out.append(("t", tb, tb.transpose(perm)))
        if len(shape) >= 2:
            wide = np.zeros(shape[:-1] + (2 * shape[-1],), base.dtype)
            wide[..., ::2] = base
            out.append(("strided", wide, wide[..., ::2]))
            rb = np.ascontiguousarray(base[::-1])
            out.append(("reversed", rb, rb[::-1]))
        return out

    float_dtypes = ("float64", "float32", "float16", "complex128")

    for modname, p in POLY_MODULES:
        mod = _poly_module(modname)
        v1 = getattr(mod, p + "vander")
        v2 = getattr(mod, p + "vander2d")
        v3 = getattr(mod, p + "vander3d")
        op1, op2, op3 = f"{modname}.{p}vander", f"{modname}.{p}vander2d", f"{modname}.{p}vander3d"

        # (A) dtype x length x degree, 1-D contiguous x
        for dt in ALL_DTYPES:
            inexact = np.dtype(dt).kind in "fc"
            lengths = (0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33) if inexact else (0, 1, 3, 5, 9, 17)
            degs = (0, 1, 2, 3, 5, 9, 14) if inexact else (0, 1, 2, 5, 9)
            for n in lengths:
                x = _poly_fill(n, dt, seed=3) if n else np.zeros(0, dt)
                for d in degs:
                    emit(op1, {"x": A(x), "deg": d}, lambda: v1(x, d), "c_contiguous_1d", f"{op1}/dt/{dt}/{n}/{d}")

        # (B) full-mantissa random values around the lane widths (W = 2 complex, 4 float64, 8 float32 / float16)
        for dt in float_dtypes:
            for n in (1, 2, 5, 8, 9, 17, 33):
                for draw in range(2):
                    x = _pc_random((n,), dt, 3000 + 17 * draw + n)
                    for d in (4, 8):
                        emit(op1, {"x": A(x), "deg": d}, lambda: v1(x, d), "cfull_1d", f"{op1}/rand/{dt}/{n}/{draw}/{d}")

        # (C) special values
        for dt in float_dtypes:
            x = _pv_special(dt)
            for d in (0, 1, 2, 3, 6):
                emit(op1, {"x": A(x), "deg": d}, lambda: v1(x, d), "special_1d", f"{op1}/spec/{dt}/{d}")

        # (D) memory layouts of x, values and flags
        for dt in ("float64", "complex128", "int32", "float16", "bool"):
            base = _poly_fill(7, dt, seed=4)
            for lname, ax in _ps_layouts(base):
                for d in (0, 2, 5):
                    emit(op1, {"x": ax, "deg": d}, lambda: v1(ax.view, d), lname, f"{op1}/lay1/{dt}/{lname}/{d}")
                    if dt == "float64":
                        emit(op1, {"x": ax, "deg": d}, lambda: v1(ax.view, d), lname, f"{op1}/lay1flags/{lname}/{d}",
                             facet="flags")
            src = base[:1].copy()
            bc = np.broadcast_to(src, (6,))
            z0 = np.array(base[1])
            for d in (0, 3):
                emit(op1, {"x": A(src, bc), "deg": d}, lambda: v1(bc, d), "broadcast_1d", f"{op1}/lay1/{dt}/broadcast/{d}")
                emit(op1, {"x": A(z0), "deg": d}, lambda: v1(z0, d), "scalar_0d", f"{op1}/lay1/{dt}/0d/{d}")
        for shape in ((2, 3), (3, 1, 2), (1, 4), (4, 1), (2, 3, 4)):
            for dt in ("float64", "complex128", "int32", "float16"):
                base = _poly_coef(shape, dt, seed=5)
                for vname, vb, vv in nd_views(base):
                    for d in (0, 2, 5):
                        emit(op1, {"x": A(vb, vv), "deg": d}, lambda: v1(vv, d), f"nd_{vname}",
                             f"{op1}/nd/{shape}/{dt}/{vname}/{d}")
                        if dt == "float64":
                            emit(op1, {"x": A(vb, vv), "deg": d}, lambda: v1(vv, d), f"nd_{vname}",
                                 f"{op1}/ndflags/{shape}/{vname}/{d}", facet="flags")
        for shape in ((), (1,), (5,), (0,), (0, 3), (3, 0), (1, 1), (4, 1, 1), (2, 0, 2)):
            x = np.zeros(shape) + 0.5
            for d in (0, 1, 3):
                emit(op1, {"x": A(x), "deg": d}, lambda: v1(x, d), "flags", f"{op1}/flags/{shape}/{d}", facet="flags")
                emit(op1, {"x": A(x), "deg": d}, lambda: v1(x, d), "zero_size", f"{op1}/shape/{shape}/{d}")

        # (E) float16's Python-int constants past its exact integers (rounded to even), and past its range (inf) where the
        #     basis reaches 65504 at a corpus-sized degree (leg / lag: 2*i - 1, herm: 2*(i - 1), herme: i - 1).
        x16 = np.array([0.5, -0.25, 0.75], np.float16)
        emit(op1, {"x": A(x16), "deg": 2100}, lambda: v1(x16, 2100), "long_deg", f"{op1}/f16range/2100")
        x16b = np.array([0.5], np.float16)
        big = {"leg": 33000, "lag": 33000, "herm": 33000, "herme": 65600}.get(p)
        if big:
            emit(op1, {"x": A(x16b), "deg": big}, lambda: v1(x16b, big), "long_deg", f"{op1}/f16range/{big}")
        for dt in ("float32", "float64", "complex128"):
            xl = _pc_random((3,), dt, 3500) / 3.0
            emit(op1, {"x": A(xl), "deg": 300}, lambda: v1(xl, 300), "long_deg", f"{op1}/longdeg/{dt}/300")

        # (F) 2-D / 3-D
        pts_shapes = ((), (1,), (3,), (8,), (9,), (17,), (2, 3))
        deg2 = ((0, 0), (1, 0), (0, 2), (2, 3), (3, 1), (4, 4))
        deg3 = ((0, 0, 0), (1, 2, 1), (2, 0, 3), (3, 3, 2))
        for dt in ALL_DTYPES:
            for shape in pts_shapes:
                x = _poly_coef(shape, dt, seed=6)
                y = _poly_coef(shape, dt, seed=7)
                z = _poly_coef(shape, dt, seed=8)
                for dg in deg2:
                    dl = list(dg)
                    emit(op2, {"x": A(x), "y": A(y), "deg": dl}, lambda: v2(x, y, dl), f"nd2_{len(shape)}d",
                         f"{op2}/dt/{dt}/{shape}/{dg}")
                for dg in deg3:
                    dl = list(dg)
                    emit(op3, {"x": A(x), "y": A(y), "z": A(z), "deg": dl}, lambda: v3(x, y, z, dl), f"nd3_{len(shape)}d",
                         f"{op3}/dt/{dt}/{shape}/{dg}")
        mixed = [("float32", "float64"), ("float16", "int8"), ("float16", "uint8"), ("float16", "int16"),
                 ("float32", "int32"), ("int8", "uint8"), ("bool", "float16"), ("complex128", "float64"),
                 ("int64", "uint64"), ("float32", "int16"), ("uint16", "int32"), ("complex128", "int32"),
                 ("float16", "float32"), ("bool", "int8")]
        for dx, dy in mixed:
            for shape in ((), (3,), (9,)):
                x = _poly_coef(shape, dx, seed=9)
                y = _poly_coef(shape, dy, seed=10)
                z = _poly_coef(shape, dy, seed=11)
                for dg in ((1, 2), (3, 3)):
                    dl = list(dg)
                    emit(op2, {"x": A(x), "y": A(y), "deg": dl}, lambda: v2(x, y, dl), "mixed_dtypes",
                         f"{op2}/mixed/{dx}/{dy}/{shape}/{dg}")
                emit(op3, {"x": A(x), "y": A(y), "z": A(z), "deg": [1, 2, 1]}, lambda: v3(x, y, z, [1, 2, 1]),
                     "mixed_dtypes", f"{op3}/mixed/{dx}/{dy}/{shape}")
                emit(op3, {"x": A(y), "y": A(x), "z": A(x), "deg": [2, 1, 1]}, lambda: v3(y, x, x, [2, 1, 1]),
                     "mixed_dtypes", f"{op3}/mixed_rev/{dx}/{dy}/{shape}")
        # special and full-mantissa values in both coordinates (the products' NaN / inf / complex forms)
        for dt in float_dtypes:
            sx = _pv_special(dt)
            sy = np.roll(sx, 5).copy()
            sz = np.roll(sx, 2).copy()
            for dg in ((0, 0), (1, 1), (2, 3)):
                dl = list(dg)
                emit(op2, {"x": A(sx), "y": A(sy), "deg": dl}, lambda: v2(sx, sy, dl), "special_2d", f"{op2}/spec/{dt}/{dg}")
            emit(op3, {"x": A(sx), "y": A(sy), "z": A(sz), "deg": [1, 2, 1]}, lambda: v3(sx, sy, sz, [1, 2, 1]),
                 "special_3d", f"{op3}/spec/{dt}")
            for n in (1, 3, 9):
                rx = _pc_random((n,), dt, 3600 + n)
                ry = _pc_random((n,), dt, 3700 + n)
                rz = _pc_random((n,), dt, 3800 + n)
                emit(op2, {"x": A(rx), "y": A(ry), "deg": [3, 4]}, lambda: v2(rx, ry, [3, 4]), "cfull_2d", f"{op2}/rand/{dt}/{n}")
                emit(op3, {"x": A(rx), "y": A(ry), "z": A(rz), "deg": [2, 2, 3]}, lambda: v3(rx, ry, rz, [2, 2, 3]),
                     "cfull_3d", f"{op3}/rand/{dt}/{n}")
        # point layouts: each coordinate read in place with its own strides (a non-flat one copied first)
        for dt in ("float64", "complex128", "int32"):
            for shape in ((3, 4), (2, 3, 2)):
                bx = _poly_coef(shape, dt, seed=12)
                by = _poly_coef(shape, dt, seed=13)
                vx = nd_views(bx)
                vy = nd_views(by)
                for (nx, bxb, bxv), (ny, byb, byv) in zip(vx, vy[1:] + vy[:1]):
                    emit(op2, {"x": A(bxb, bxv), "y": A(byb, byv), "deg": [2, 1]}, lambda: v2(bxv, byv, [2, 1]),
                         f"nd2_{nx}_{ny}", f"{op2}/lay/{dt}/{shape}/{nx}/{ny}")
                    emit(op3, {"x": A(bxb, bxv), "y": A(byb, byv), "z": A(bxb, bxv), "deg": [1, 1, 2]},
                         lambda: v3(bxv, byv, bxv, [1, 1, 2]), f"nd3_{nx}_{ny}", f"{op3}/lay/{dt}/{shape}/{nx}/{ny}")
            b1 = _poly_fill(7, dt, seed=14)
            b2 = _poly_fill(7, dt, seed=15)
            lx = _ps_layouts(b1)
            ly = _ps_layouts(b2)
            for (nx, ax), (ny, ay) in zip(lx, ly[2:] + ly[:2]):
                emit(op2, {"x": ax, "y": ay, "deg": [2, 2]}, lambda: v2(ax.view, ay.view, [2, 2]), f"lay1_{nx}_{ny}",
                     f"{op2}/lay1/{dt}/{nx}/{ny}")
            src = b1[:1].copy()
            bc = np.broadcast_to(src, (7,))
            emit(op2, {"x": A(src, bc), "y": A(b2), "deg": [1, 3]}, lambda: v2(bc, b2, [1, 3]), "broadcast_1d",
                 f"{op2}/lay1/{dt}/broadcast")
        for shape in ((), (1,), (5,), (2, 3), (1, 1), (3, 1), (1, 4)):
            x = np.zeros(shape) + 0.5
            for dg in ((0, 0), (1, 2), (2, 0)):
                dl = list(dg)
                emit(op2, {"x": A(x), "y": A(x), "deg": dl}, lambda: v2(x, x, dl), "flags", f"{op2}/flags/{shape}/{dg}",
                     facet="flags")
            for dg in ((0, 0, 0), (1, 1, 2)):
                dl = list(dg)
                emit(op3, {"x": A(x), "y": A(x), "z": A(x), "deg": dl}, lambda: v3(x, x, x, dl), "flags",
                     f"{op3}/flags/{shape}/{dg}", facet="flags")
        # Python-typed points (stacked by np.asarray's coercion; a Python float is float64 there, np.float16 is not)
        for pname, xv, yv in (("pyfloat", 0.5, 0.25), ("pyint", 1, 2), ("pybool", True, False), ("pycomplex", 1j, 2),
                              ("list", [0.5, 1.5], [2, 3]), ("tuple", (0.5, 1.5), (2.0, 3.0)),
                              ("nested", [[1, 2], [3, 4]], [[5, 6], [7, 8]]),
                              ("f16_list", A(np.array([0.5, 1.5], np.float16)), [0.5, 1.0]),
                              ("f16_0d_py", A(np.array(0.5, np.float16)), 0.5),
                              ("npf16", np.float16(0.5), np.float16(1.5)), ("npf16_py", np.float16(0.5), 1.5),
                              ("u64", [2 ** 64 - 1, 3], [1, 2]), ("arr_list", A(np.array([1.0, 2.0])), [3, 4]),
                              ("f32_0d_pair", A(np.array(0.5, np.float32)), A(np.array(0.25, np.float32))),
                              ("empty_lists", [], []), ("list_of_0d", [A(np.array(1.0)), A(np.array(2.0))], [3, 4])):
            emit(op2, {"x": xv, "y": yv, "deg": [2, 1]}, lambda: v2(_ps_py(xv), _ps_py(yv), [2, 1]), "arg_points",
                 f"{op2}/argpts/{pname}")
            emit(op3, {"x": xv, "y": yv, "z": xv, "deg": [1, 1, 1]}, lambda: v3(_ps_py(xv), _ps_py(yv), _ps_py(xv), [1, 1, 1]),
                 "arg_points", f"{op3}/argpts/{pname}")

        # (G) argument kinds and errors, NumPy's order
        xg = np.array([0.5, -1.5, 2.0])
        deg_kinds = [
            ("int", 3), ("true", True), ("false", False), ("zero", 0), ("neg", -1), ("neg_big", -2 ** 70),
            ("0d_i8", A(np.array(2, np.int8))), ("0d_u64", A(np.array(2, np.uint64))), ("0d_i32", A(np.array(2, np.int32))),
            ("0d_bool", A(np.array(True))), ("0d_f64", A(np.array(2.0))), ("0d_f16", A(np.array(2.0, np.float16))),
            ("0d_f32", A(np.array(2.5, np.float32))), ("0d_c", A(np.array(2 + 0j))), ("0d_neg", A(np.array(-1))),
            ("1d_i", A(np.array([2]))), ("1d_i2", A(np.array([1, 2]))), ("2d_i", A(np.array([[2]]))), ("1d_f", A(np.array([2.0]))),
            ("npf16", np.float16(2)), ("npf16_1000", np.float16(1000)), ("float", 2.0), ("float_frac", 1.5),
            ("negzero", -0.0), ("f_1e20", 1e20), ("f_inf", float("inf")), ("f_nan", float("nan")), ("f_neg", -1.0),
            ("f_small", 1e-7), ("complex", 2 + 0j), ("complex_j", 1j), ("complex_neg0", complex(-0.0, 1.0)),
            ("str", "2"), ("str_abc", "abc"), ("none", None), ("list", [2]), ("list2", [1, 2]), ("tuple", (2,)),
            ("tuple2", (1, 2)), ("list_f", [2.0]), ("list_mixed", [1, "a", None, 2.5]), ("tuple_empty", ()),
            ("list_empty", []), ("list_nested", [[1, 2], [3]]), ("list_arr", [A(np.array([1.5]))]),
            ("list_0d", [A(np.array(2.0))]), ("list_npf16", [np.float16(2)]), ("tuple_str", ("a", "b'c")),
            ("big_max", 2 ** 63 - 1), ("big_max2", 2 ** 63 - 2), ("big_62", 2 ** 62), ("big_60", 2 ** 60),
            ("big_64", 2 ** 64), ("big_100", 2 ** 100),
        ]
        for dname, dv in deg_kinds:
            emit(op1, {"x": A(xg), "deg": dv}, lambda: v1(xg, _ps_py(dv)), "arg_deg", f"{op1}/argdeg/{dname}")
        e0 = np.zeros(0)
        for dname, dv in (("big_62", 2 ** 62), ("big_max", 2 ** 63 - 1), ("big_60_2x0", 2 ** 60)):
            xe = e0 if dname != "big_60_2x0" else np.zeros((2, 0))
            emit(op1, {"x": A(xe), "deg": dv}, lambda: v1(xe, dv), "arg_deg", f"{op1}/argdeg_empty/{dname}")
        # the degree is checked before x is converted (a VALID degree would let NumPy compute `x + 0.0` with the str /
        # object array — where NumSharp refuses it — so only the degree errors that come first are recorded here)
        for dname, dv in (("neg", -1), ("frac", 1.5), ("none", None)):
            for xname, xv in (("str", "abc"), ("none", None), ("ragged", [[1, 2], [3]]), ("obj", [1, None]),
                              ("bigint", [2 ** 70])):
                emit(op1, {"x": xv, "deg": dv}, lambda: v1(xv, _ps_py(dv)), "arg_order", f"{op1}/order/{xname}/{dname}")
        for xname, xv in (("ragged", [[1, 2], [3]]), ("ragged_deep", [[[1], [1, 2]]]), ("ragged_depth", [[1], [[2]]]),
                          ("ragged_arrays", [A(np.zeros(2)), A(np.zeros(3))])):
            for dv in (2, 2 ** 63 - 1):
                emit(op1, {"x": xv, "deg": dv}, lambda: v1(_ps_py(xv), dv), "arg_order", f"{op1}/ragged/{xname}/{dv}")
        x_kinds = [("pyint", 3), ("pyfloat", 0.5), ("pycomplex", 0.5j), ("pybool", True), ("list_int", [1, 2]),
                   ("list_float", [0.5, -1.5]), ("tuple", (0.5, 1.0)), ("nested", [[0.5, 1], [2, 3]]),
                   ("list_bool", [True, False]), ("list_npf16", [np.float16(0.5), np.float16(1)]),
                   ("list_npf16_py", [np.float16(0.5), 1.0]), ("empty", []), ("empty_nested", [[], []]),
                   ("u64max", [2 ** 64 - 1]), ("list_complex", [1j, 2]), ("npf16", np.float16(0.5)),
                   ("list_0d", [A(np.array(0.5, np.float32)), A(np.array(1.5, np.float32))]),
                   ("list_arrays", [A(np.array([1.0, 2.0])), A(np.array([3.0, 4.0]))]),
                   ("tuple_nested", ((1, 2), (3, 4))), ("negzero", [-0.0, 0.0])]
        for xname, xv in x_kinds:
            for d in (0, 2):
                emit(op1, {"x": xv, "deg": d}, lambda: v1(_ps_py(xv), d), "arg_x", f"{op1}/argx/{xname}/{d}")
        # 2-D / 3-D degree containers
        a2 = np.array([0.5, 0.25])
        b2 = np.array([0.1, 0.2])
        deg2_kinds = [
            ("int", 2), ("float", 2.0), ("none", None), ("0d", A(np.array(2))), ("str_ab", "ab"), ("str_12", "12"),
            ("str_1", "1"), ("bool", True), ("complex", 1j), ("npf16", np.float16(2)), ("list3", [1, 1, 1]), ("list1", [1]),
            ("list0", []), ("tuple", (1, 2)), ("arr_f", A(np.array([1.0, 2.0]))), ("arr_bool", A(np.array([True, False]))),
            ("arr_2d_col", A(np.array([[1], [2]]))), ("arr_2d_row", A(np.array([[1, 2]]))),
            ("arr_u8", A(np.array([1, 2], np.uint8))), ("arr_i64", A(np.array([2, 1]))), ("list_nested", [[1], [2]]),
            ("list_f", [1, 2.0]), ("list_neg_first", [-1, 1.5]), ("list_f_first", [1.5, -1]), ("list_ok", [1, 2]),
            ("list_none", [None, 1]), ("list_str", ["1", 2]), ("list_big", [2 ** 63, 1]), ("list_true", [True, 2]),
            ("list_npf16", [np.float16(1), 1]), ("list_0d", [A(np.array(1)), A(np.array(2, np.int16))]),
            ("list_neg_second", [1, -1]), ("list_maxdim_second", [1, 2 ** 63 - 1]), ("list_toobig", [2 ** 61, 1]),
            ("list_toobig_then_frac", [2 ** 61, 1.5]), ("list_frac_then_toobig", [1.5, 2 ** 61]), ("list_c", [1j, 1]),
        ]
        for dname, dv in deg2_kinds:
            emit(op2, {"x": A(a2), "y": A(b2), "deg": dv}, lambda: v2(a2, b2, _ps_py(dv)), "arg_deg2", f"{op2}/argdeg/{dname}")
        deg3_kinds = [("int", 2), ("none", None), ("0d", A(np.array(1))), ("str_abc", "abc"), ("str_12", "12"), ("list2", [1, 2]),
                      ("list4", [1, 2, 3, 4]), ("tuple", (1, 2, 0)), ("arr_i", A(np.array([1, 0, 2]))),
                      ("arr_f", A(np.array([1.0, 0.0, 2.0]))), ("list_neg_third", [1, 1, -1]), ("list_frac_third", [1, 1, 0.5]),
                      ("list_maxdim_third", [1, 1, 2 ** 63 - 1]), ("list_toobig_first", [2 ** 61, 1, 1.5])]
        for dname, dv in deg3_kinds:
            emit(op3, {"x": A(a2), "y": A(b2), "z": A(a2), "deg": dv}, lambda: v3(a2, b2, a2, _ps_py(dv)), "arg_deg3",
                 f"{op3}/argdeg/{dname}")
        # points: shape mismatches (np.asarray's ragged text), error order, empty points (the reshape error)
        z2 = np.zeros(0)
        for oname, xv, yv, dv in (
                ("ragged_bad_deg", [1, 2], [3, 4, 5], [1, -1]), ("ragged_deg_int", [1, 2], [3], 2),
                ("ragged_deg_len3", [1, 2], [3], [1, 1, 1]), ("str_deg_int", "ab", "cd", 2),
                ("str_bad_deg", "ab", "cd", [1, -1]), ("none_deg_len1", None, None, [1]),
                ("empty_bad_deg2", A(z2), A(z2), [1, -1]), ("empty_bad_deg1", A(z2), A(z2), [-1, 1]),
                ("empty_ok", A(z2), A(z2), [1, 2]), ("empty_0x3", A(np.zeros((0, 3))), A(np.zeros((0, 3))), [1, 1]),
                ("empty_2x0", A(np.zeros((2, 0))), A(np.zeros((2, 0))), [1, 2]), ("empty_lists", [], [], [0, 0]),
                ("scalar_1d", 0.5, [1.0, 2.0], [1, 1]), ("0d_1d1", A(np.array(0.5)), A(np.array([2.0])), [1, 1]),
                ("1d_2d", [0.5, 1], [[1.0, 2.0]], [1, 1]), ("arrays_2_3", A(np.zeros(2)), A(np.zeros(3)), [1, 1]),
                ("arrays_23_24", A(np.zeros((2, 3))), A(np.zeros((2, 4))), [1, 1]),
                ("empty_huge", A(z2), A(z2), [2 ** 62, 1]), ("empty_maxdim", A(z2), A(z2), [1, 2 ** 63 - 1]),
                ("ragged_huge", [1, 2], [3], [2 ** 63 - 1, 1])):
            emit(op2, {"x": xv, "y": yv, "deg": dv}, lambda: v2(_ps_py(xv), _ps_py(yv), _ps_py(dv)), "arg_order2",
                 f"{op2}/order/{oname}")
        for oname, xv, yv, zv, dv in (
                ("empty_ok", A(z2), A(z2), A(z2), [1, 2, 1]), ("empty_3x0x2", A(np.zeros((3, 0, 2))), A(np.zeros((3, 0, 2))),
                                                                A(np.zeros((3, 0, 2))), [1, 2, 1]),
                ("ragged_third", [1, 2], [3, 4], [5], [1, 1, 1]), ("empty_bad_deg3", A(z2), A(z2), A(z2), [1, 1, -1])):
            emit(op3, {"x": xv, "y": yv, "z": zv, "deg": dv}, lambda: v3(_ps_py(xv), _ps_py(yv), _ps_py(zv), _ps_py(dv)),
                 "arg_order3", f"{op3}/order/{oname}")

        # (I) inputs longer than one kernel block — two bases carry it (the block loop is the same for every basis)
        if p in ("cheb", "lag"):
            for dt, n, d in (("float32", 5000, 3), ("float16", 9000, 2), ("complex128", 1700, 3), ("float64", 3100, 4),
                             ("int16", 3500, 3)):
                xb = _pc_random((n,), dt, 4000 + n) if np.dtype(dt).kind in "fc" else _poly_fill(n, dt, 5)
                emit(op1, {"x": A(xb), "deg": d}, lambda: v1(xb, d), "blocks_1d", f"{op1}/blocks/{dt}/{n}/{d}")
            xs = _pc_random((3000,), "float32", 4100)
            ss = np.zeros(6000, np.float32)
            ss[::2] = xs
            emit(op1, {"x": A(ss, ss[::2]), "deg": 3}, lambda: v1(ss[::2], 3), "blocks_strided", f"{op1}/blocks/strided")

    # Section J runs in its own pass over the modules, after A-I, so the running id counter of every earlier case is
    # unchanged by it (the ids end in that counter).
    for modname, p in POLY_MODULES:
        v2 = getattr(_poly_module(modname), p + "vander2d")
        v3 = getattr(_poly_module(modname), p + "vander3d")
        op2, op3 = f"{modname}.{p}vander2d", f"{modname}.{p}vander3d"
        # (J) the OBJECT stack of scalars (wholeness pass 2026-09-30): a Python int past uint64 among scalar points makes
        #     np.asarray((x, y[, z])) an object array, whose `+ 0.0` then runs per element in Python and hands every
        #     dimension its OWN number — float(int) (OverflowError past the float range), a Python float / bool / complex,
        #     a NumPy scalar or 0-d array keeping NEP 50's dtype — so the dimensions' matrices can have different dtypes
        #     (a float16 point's matrix computed in float16, then promoted by the outer product's multiply). Both orders,
        #     2-D and 3-D, the stack's OverflowError before any degree error, and the degree errors after it.
        # float(int) is CORRECTLY ROUNDED (round-half-even; .NET's BigInteger -> double truncates): values that round
        # (2**70 + 1 down, 2**64 + 2**11 + 1 up, 3**45, one just below the overflow tie rounding down to the largest
        # double) next to exactly representable ones, the tie that rounds up to 2**1024 (OverflowError) and 2**1024.
        bigs = [("big70", 2 ** 70), ("negbig70", -(2 ** 70)), ("big64", 2 ** 64), ("bigmaxf", 2 ** 1023 + 2 ** 1022),
                ("bigover", 2 ** 1024), ("bigtie", 2 ** 1024 - 2 ** 970), ("bigbelow", 2 ** 1024 - 2 ** 971),
                ("big70p1", 2 ** 70 + 1), ("bigup", 2 ** 64 + 2 ** 11 + 1), ("bigodd", 3 ** 45), ("negbigodd", -(3 ** 45)),
                ("belowtie", 2 ** 1024 - 2 ** 970 - 1)]
        others = [("int", 3), ("u64max", 2 ** 64 - 1), ("f", 1.5), ("fnan", float("nan")), ("finf", float("-inf")),
                  ("fneg0", -0.0), ("c", complex(0.5, -1.0)), ("cnan", complex(float("nan"), 1.0)), ("b", True), ("bf", False),
                  ("h", np.float16(0.5)), ("hnan", np.float16("nan")), ("hneg0", np.float16(-0.0)),
                  ("nd0_f32", A(np.array(0.1, np.float32))), ("nd0_i8", A(np.array(-3, np.int8))), ("nd0_b", A(np.array(True))),
                  ("nd0_f16", A(np.array(0.25, np.float16))), ("nd0_c", A(np.array(1 + 2j))),
                  ("nd0_u64max", A(np.array(2 ** 64 - 1, np.uint64))), ("nd0_f64nan", A(np.array(float("nan")))), ("big", 2 ** 80),
                  ("none", None), ("str", "a")]
        for bname, bv in bigs:
            for oname, ov in others:
                for dg in ((1, 2), (0, 0), (3, 1)):
                    dl = list(dg)
                    emit(op2, {"x": bv, "y": ov, "deg": dl}, lambda: v2(bv, _ps_py(ov), dl), "objstack",
                         f"{op2}/objstack/{bname}_{oname}/{dg}")
                    emit(op2, {"x": ov, "y": bv, "deg": dl}, lambda: v2(_ps_py(ov), bv, dl), "objstack",
                         f"{op2}/objstack/{oname}_{bname}/{dg}")
                emit(op3, {"x": bv, "y": ov, "z": ov, "deg": [1, 2, 1]}, lambda: v3(bv, _ps_py(ov), _ps_py(ov), [1, 2, 1]),
                     "objstack", f"{op3}/objstack/{bname}_{oname}")
                emit(op3, {"x": ov, "y": bv, "z": np.float16(0.75), "deg": [2, 1, 2]},
                     lambda: v3(_ps_py(ov), bv, np.float16(0.75), [2, 1, 2]), "objstack", f"{op3}/objstack/{oname}_{bname}_h")
                emit(op3, {"x": ov, "y": A(np.array(1.5, np.float32)), "z": bv, "deg": [1, 1, 3]},
                     lambda: v3(_ps_py(ov), np.array(1.5, np.float32), bv, [1, 1, 3]), "objstack",
                     f"{op3}/objstack/{oname}_nd0f32_{bname}")
            for dname, dv in (("neg_first", [-1, 1]), ("frac_second", [1, 1.5]), ("len3", [1, 1, 1]), ("int", 2),
                              ("maxdim_second", [1, 2 ** 63 - 1]), ("toobig", [2 ** 61, 1])):
                emit(op2, {"x": bv, "y": np.float16(0.5), "deg": dv}, lambda: v2(bv, np.float16(0.5), _ps_py(dv)),
                     "objstack_deg", f"{op2}/objstack_deg/{bname}/{dname}")
                emit(op2, {"x": bv, "y": 0.5, "deg": dv}, lambda: v2(bv, 0.5, _ps_py(dv)), "objstack_deg",
                     f"{op2}/objstack_deg/{bname}_f/{dname}")


    # (H) Char: NumSharp's uint16-like dtype converts to float64 exactly as uint16 does (the house weave).
    cases += _relabel_dtype([c for c in cases if "/dt/uint16/" in (c.get("id") or "") and not c.get("expects_throw")],
                            "uint16", "char")
    if skipped[0]:
        print(f"  (skipped {skipped[0]} refused / complex64 / object cells)")
    return cases


# ---- numpy.polynomial series algebra (plan docs/plans/numpy-polynomial.md U2) -------------------------------------
#
# {p}mulx(c), {p}mul(c1, c2), {p}div(c1, c2) -> (quo, rem), {p}pow(c, pow, maxpower), {p}fromroots(roots) for the six
# bases, and X2poly(c) / poly2X(pol) for the five non-power ones. Op keys are module-qualified ("legendre.legmul",
# "chebyshev.poly2cheb"); arguments are recorded by NAME with the polyseries encoding (_ps_enc), in the fixed order
# c, c1, c2, roots, pol, pow — the order the C# replay decodes them and consumes operands in — except maxpower: a
# plain JSON int, or JSON null for an explicit None (absent = NumPy's default: None for polypow, 16 for the others).
# {p}div is a tuple case (both slots recorded, arity asserted). A "facet": "flags" case records each result's
# [C_CONTIGUOUS, F_CONTIGUOUS, OWNDATA] instead of its values: NumPy returns trimseq VIEWS (polymul's trimmed
# product, polydiv's remainder, _div's trimmed remainder) next to fresh arrays and as_series copies, and which one
# comes back is part of the contract.
#
# TWO FILES. np.convolve (polymul, chebmul's z-series product, polypow, chebpow, polyfromroots, chebfromroots) reduces
# every output position with NumPy's per-dtype dot: small_correlate's sequential sum for float/double kernels of at
# most 11 terms, else cblas ?dot / ?dotu — sequential below the OpenBLAS kernel's vector block (ddot: 16 terms; sdot:
# 32, products summed in a double; zdotu: 8, four separate sums), a vector kernel from there on. NumSharp's managed
# sliding engine reproduces the sequential regime byte for byte; the vector regime is reproducible only through the
# scipy-openblas NumSharp.Interop.OpenBLAS bundles. Every call runs under a np.convolve recorder, and a case whose
# products reach the vector regime — float64 with both factors of 16 or more coefficients, float32 32 or more,
# complex128 8 or more, or a complex product of non-finite values (the contiguous zdotu mixes lanes, so an infinity
# puts a NaN in the real part) — goes to polyalgebra_parity.jsonl, host-pinned like linalg_parity (threads = 1).
# Everything else — the four recurrence bases (no convolution), the divisions, the conversions and every short
# product — is portable, in polyalgebra.jsonl.
#
# Sections: (A) dtype x length at the defaults; (B) full-mantissa values; (C) the trim / special patterns; (D) layouts;
# (E) Python-typed arguments; (F) errors in NumPy's check order; (G) pow / maxpower kinds; (H) long series across the
# managed / BLAS boundary; (I) result flags; (J) fromroots' root kinds; (K) underflow and overflow.
# complex64 and object results would be skipped (#569 / no object dtype).

class _PAConvRecorder:
    """Records every np.convolve call a polynomial function makes. The polynomial modules reach np.convolve through the
    numpy module at call time (polymul's `np.convolve(c1, c2)`, chebyshev's `_zseries_mul`, polypow's
    `pu._pow(np.convolve, ...)`), so replacing the module attribute for the duration of one call sees every product."""

    def __init__(self):
        self.calls = []
        self._orig = None

    def __enter__(self):
        self._orig = np.convolve
        orig = self._orig
        calls = self.calls

        def conv(a, v, mode="full"):
            aa = np.asarray(a)
            vv = np.asarray(v)
            with np.errstate(all="ignore"):
                finite = bool(np.all(np.isfinite(aa))) and bool(np.all(np.isfinite(vv)))
            calls.append((int(aa.size), int(vv.size), np.result_type(aa, vv), finite))
            return orig(a, v, mode)

        np.convolve = conv
        return self

    def __exit__(self, *exc):
        np.convolve = self._orig
        return False

    def blas_bound(self):
        """Whether any recorded product reaches OpenBLAS's vector regime (see the section comment): the longest dot of
        a full convolution is the shorter operand's length. A complex product holding infinities or NaNs used to count
        too; NumSharp's managed dot now reproduces zdotu's C99 result construction (an infinite / NaN imaginary part
        turns the real part into NaN) and CDOUBLE_dot's plain loop for a one-element kernel, so only the vector regime
        is BLAS-bound. (`finite` is still recorded: it documents which cases hold non-finite values.)"""
        for n1, n2, dt, finite in self.calls:
            m = min(n1, n2)
            if dt == np.float64 and m >= 16:
                return True
            if dt == np.float32 and m >= 32:
                return True
            if dt == np.complex128 and m >= 8:
                return True
        return False


# Roots with ONE zero only: NumPy's SIMD sort orders +0.0 / -0.0 by CPU, NumSharp's puts -0.0 first, and the product
# order then decides a zero's sign (a documented divergence, pinned by a unit test, kept out of the corpus).
_PA_ROOTS = [0.5, -1.25, 2.0, -0.75, 1.5, -2.0, 0.25, 1.0, -0.5, 3.0, -3.0, 0.125, 1.75, -1.5, 2.5, -0.375, 0.0,
             0.625, -2.5, 1.25]
_PA_IROOTS = [1, -2, 3, 0, -1, 2, 5, -4, 7, 4, -3, 6, -5, 8, -6, 9]


def _pa_roots(k, dt, seed=0):
    """k distinct-ish roots of dtype dt (repeats are harmless — equal values have equal bits — but never both zeros)."""
    dt = np.dtype(dt)
    if k == 0:
        return np.zeros(0, dt)
    if dt.kind == "f":
        return np.array([_PA_ROOTS[(i * 3 + seed) % len(_PA_ROOTS)] for i in range(k)]).astype(dt)
    if dt.kind == "c":
        re = [_PA_ROOTS[(i * 3 + seed) % len(_PA_ROOTS)] for i in range(k)]
        im = [_PA_ROOTS[(i * 5 + seed + 7) % len(_PA_ROOTS)] for i in range(k)]
        return (np.array(re) + 1j * np.array(im)).astype(dt)
    if dt.kind == "b":
        return np.array([(i + seed) % 2 == 0 for i in range(k)], dtype=bool)
    with np.errstate(all="ignore"):
        return np.array([_PA_IROOTS[(i * 3 + seed) % len(_PA_IROOTS)] for i in range(k)]).astype(dt)


def _pa_random(n, dt, seed, scale=1.0):
    """Seeded full-mantissa series in [-scale, scale] (the complex product forms and the dot schedules differ only on
    such operands)."""
    rng = np.random.default_rng(seed)
    dt = np.dtype(dt)
    v = rng.uniform(-scale, scale, n)
    if dt.kind == "c":
        v = v + 1j * rng.uniform(-scale, scale, n)
    with np.errstate(all="ignore"):
        return v.astype(dt)


def _pa_object_land(call):
    """Whether NumPy's outcome of `call` is a computation with Python objects: an object-array result, or a TypeError the
    object arithmetic raises (None * 0.0, str * float, sorting None). NumSharp has no object dtype and refuses there with
    NotSupportedException (the documented divergence), so such a case cannot be gated; every error NumPy raises BEFORE it
    computes with objects — as_series' checks, np.array's raggedness, the zero divisor, the power checks, int(pow) and the
    maxpower comparison — can, and is kept."""
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = call()
    except TypeError as e:
        m = str(e)
        return not (m.startswith("int() argument must be") or m.startswith("'>' not supported between instances of 'int'")
                    or m.startswith("only 0-dimensional arrays") or m.startswith("object of type"))
    except Exception:
        return False
    rs = r if isinstance(r, (tuple, list)) else (r,)
    return any(np.asarray(x).dtype == object for x in rs)


def gen_polyalgebra():
    cases = []
    host = []
    counter = [0]
    skipped = [0]
    ok_dtypes = set(ALL_DTYPES)
    A = _PSArr
    arg_order = ("c", "c1", "c2", "roots", "pol", "pow", "maxpower")

    def flags(a):
        return np.array([a.flags.c_contiguous, a.flags.f_contiguous, a.flags.owndata])

    def emit(op, args, call, layout, cid, kind="array", facet=None):
        operands = []
        params = {}
        for key in arg_order:
            if key in args:
                v = args[key]
                # maxpower stays a plain JSON int / null where the facades' int? parameter takes it; any other kind
                # (a float, a bool, a str, an ndarray, an int past int32, np.float16) travels as a spec.
                plain = v is None or (type(v) is int and -2 ** 31 <= v < 2 ** 31)
                params[key] = v if key == "maxpower" and plain else _ps_enc(v, operands)
        if facet:
            params["facet"] = facet
        n = counter[0]
        counter[0] += 1
        rec = _PAConvRecorder()
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings(), rec:
                warnings.simplefilter("ignore")
                r = call()
        except Exception as e:
            dest = host if rec.blas_bound() else cases
            dest.append({"id": f"{cid}/{n}", "op": op, "params": params, "operands": operands,
                         "expected": {"kind": kind} if kind != "array" else {}, "expects_throw": True,
                         "error": _poly_exc(e), "layout": layout, "valueclass": "error"})
            return
        dest = host if rec.blas_bound() else cases
        if kind == "tuple":
            arrs = [flags(v) if facet == "flags" else np.asarray(v) for v in r]
            if any(a.dtype.name not in ok_dtypes for a in arrs):
                skipped[0] += 1
                return
            dest.append(_case(op, params, operands, _tuple_expected(arrs), layout, "polyalgebra", cid=f"{cid}/{n}"))
        else:
            a = flags(r) if facet == "flags" else np.asarray(r)
            if a.dtype.name not in ok_dtypes:
                skipped[0] += 1
                return
            dest.append(_case(op, params, operands, _arr_expected(a), layout, "polyalgebra", cid=f"{cid}/{n}"))

    sub_dtypes = ["float64", "float32", "float16", "complex128", "int32", "uint64", "bool"]
    patterns = ("tz", "allzero", "lastnz", "special", "nantail")
    pat_dtypes = ("float64", "float32", "float16", "complex128", "int64")

    for modname, p in POLY_MODULES:
        mod = _poly_module(modname)
        power = modname == "polynomial"
        full = modname in ("polynomial", "chebyshev", "legendre")
        dts = ALL_DTYPES if full else sub_dtypes
        mulx = getattr(mod, p + "mulx")
        mul = getattr(mod, p + "mul")
        div = getattr(mod, p + "div")
        pw = getattr(mod, p + "pow")
        fr = getattr(mod, p + "fromroots")
        o_mulx, o_mul, o_div = f"{modname}.{p}mulx", f"{modname}.{p}mul", f"{modname}.{p}div"
        o_pow, o_fr = f"{modname}.{p}pow", f"{modname}.{p}fromroots"
        # The two conversions of a non-power basis: (op key, function, argument name).
        convs = [] if power else [(f"{modname}.{p}2poly", getattr(mod, p + "2poly"), "c"),
                                  (f"{modname}.poly2{p}", getattr(mod, "poly2" + p), "pol")]
        # A NumPy-default maxpower, spelled explicitly only where a case needs it (polypow's None would loop forever
        # on a huge power, so those cases pass 16 by hand).
        cap = 16 if power else None

        # ---------------- (A) dtype x length at the defaults ----------------
        for dt in ALL_DTYPES:
            for n in (1, 2, 3, 4, 5, 8, 13):
                c = _ps_series(n, dt, seed=1)
                emit(o_mulx, {"c": A(c)}, lambda: mulx(c), "c_contiguous_1d", f"{o_mulx}/dt/{dt}/{n}")
                for cop, cf, cname in convs:
                    emit(cop, {cname: A(c)}, lambda: cf(c), "c_contiguous_1d", f"{cop}/dt/{dt}/{n}")
            for n in (1, 2, 3, 4):
                c = _ps_series(n, dt, seed=2)
                for k in (0, 1, 2, 3, 4):
                    emit(o_pow, {"c": A(c), "pow": k}, lambda: pw(c, k), "c_contiguous_1d", f"{o_pow}/dt/{dt}/{n}/{k}")
            for k in (0, 1, 2, 3, 4, 5, 7, 9, 12):
                roots = _pa_roots(k, dt, seed=3)
                emit(o_fr, {"roots": A(roots)}, lambda: fr(roots), "c_contiguous_1d", f"{o_fr}/dt/{dt}/{k}")
        for d1 in dts:
            for d2 in dts:
                for n1, n2 in ((1, 1), (1, 4), (4, 1), (3, 5), (5, 3), (4, 4)):
                    c1 = _ps_series(n1, d1, seed=4)
                    c2 = _ps_series(n2, d2, seed=5)
                    emit(o_mul, {"c1": A(c1), "c2": A(c2)}, lambda: mul(c1, c2), "c_contiguous_1d",
                         f"{o_mul}/dt/{d1}/{d2}/{n1}/{n2}")
                for n1, n2 in ((1, 1), (4, 1), (1, 4), (5, 3), (3, 5), (4, 4), (6, 2)):
                    c1 = _ps_series(n1, d1, seed=6)
                    c2 = _ps_series(n2, d2, seed=7)
                    emit(o_div, {"c1": A(c1), "c2": A(c2)}, lambda: div(c1, c2), "c_contiguous_1d",
                         f"{o_div}/dt/{d1}/{d2}/{n1}/{n2}", kind="tuple")

        # ---------------- (B) full-mantissa values ----------------
        for dt in ("float64", "float32", "float16", "complex128"):
            for n in (1, 2, 3, 7, 17, 64):
                c = _pa_random(n, dt, seed=100 + n)
                emit(o_mulx, {"c": A(c)}, lambda: mulx(c), "c_contiguous_1d", f"{o_mulx}/rand/{dt}/{n}")
            for n1, n2 in ((1, 1), (2, 3), (5, 5), (7, 7), (8, 8), (11, 11), (12, 12), (15, 15), (16, 16), (12, 30),
                           (31, 31), (32, 32), (3, 40)):
                c1 = _pa_random(n1, dt, seed=200 + n1)
                c2 = _pa_random(n2, dt, seed=300 + n2)
                emit(o_mul, {"c1": A(c1), "c2": A(c2)}, lambda: mul(c1, c2), "c_contiguous_1d",
                     f"{o_mul}/rand/{dt}/{n1}/{n2}")
            for n1, n2 in ((5, 3), (9, 4), (12, 12), (17, 5), (20, 11), (3, 3), (2, 7)):
                c1 = _pa_random(n1, dt, seed=400 + n1)
                c2 = _pa_random(n2, dt, seed=500 + n2)
                emit(o_div, {"c1": A(c1), "c2": A(c2)}, lambda: div(c1, c2), "c_contiguous_1d",
                     f"{o_div}/rand/{dt}/{n1}/{n2}", kind="tuple")
            for n in (2, 3, 5):
                c = _pa_random(n, dt, seed=600 + n)
                for k in (2, 3, 5):
                    emit(o_pow, {"c": A(c), "pow": k}, lambda: pw(c, k), "c_contiguous_1d", f"{o_pow}/rand/{dt}/{n}/{k}")
            for k in (2, 5, 8, 13, 20):
                roots = _pa_random(k, dt, seed=700 + k, scale=2.0)
                emit(o_fr, {"roots": A(roots)}, lambda: fr(roots), "c_contiguous_1d", f"{o_fr}/rand/{dt}/{k}")
            for cop, cf, cname in convs:
                for n in (3, 6, 11, 20):
                    c = _pa_random(n, dt, seed=800 + n)
                    emit(cop, {cname: A(c)}, lambda: cf(c), "c_contiguous_1d", f"{cop}/rand/{dt}/{n}")

        # ---------------- (C) trim / special patterns ----------------
        for dt in pat_dtypes:
            for pat in patterns:
                for n in (3, 5):
                    c = _ps_series(n, dt, pat, seed=8)
                    emit(o_mulx, {"c": A(c)}, lambda: mulx(c), "c_contiguous_1d", f"{o_mulx}/pat/{dt}/{pat}/{n}")
                    for k in (0, 1, 2, 3):
                        emit(o_pow, {"c": A(c), "pow": k}, lambda: pw(c, k), "c_contiguous_1d",
                             f"{o_pow}/pat/{dt}/{pat}/{n}/{k}")
                    for cop, cf, cname in convs:
                        emit(cop, {cname: A(c)}, lambda: cf(c), "c_contiguous_1d", f"{cop}/pat/{dt}/{pat}/{n}")
                for n1, n2 in ((3, 5), (4, 4), (5, 2)):
                    mod1 = _ps_series(n1, dt, seed=9)
                    mod2 = _ps_series(n2, dt, seed=10)
                    pt1 = _ps_series(n1, dt, pat, seed=11)
                    pt2 = _ps_series(n2, dt, pat, seed=12)
                    for tag, c1, c2 in (("pm", pt1, mod2), ("mp", mod1, pt2), ("pp", pt1, pt2)):
                        emit(o_mul, {"c1": A(c1), "c2": A(c2)}, lambda: mul(c1, c2), "c_contiguous_1d",
                             f"{o_mul}/pat/{dt}/{pat}/{tag}/{n1}/{n2}")
                        emit(o_div, {"c1": A(c1), "c2": A(c2)}, lambda: div(c1, c2), "c_contiguous_1d",
                             f"{o_div}/pat/{dt}/{pat}/{tag}/{n1}/{n2}", kind="tuple")

        # ---------------- (D) layouts ----------------
        for d in ("float64", "complex128", "float16", "int16"):
            v = _ps_series(5, d, "tz", seed=13)
            w = _ps_series(3, d, seed=14)
            for ln, a in _ps_layouts(v):
                emit(o_mulx, {"c": a}, lambda: mulx(_ps_py(a)), ln, f"{o_mulx}/lay/{d}/{ln}")
                emit(o_pow, {"c": a, "pow": 2}, lambda: pw(_ps_py(a), 2), ln, f"{o_pow}/lay/{d}/{ln}")
                emit(o_fr, {"roots": a}, lambda: fr(_ps_py(a)), ln, f"{o_fr}/lay/{d}/{ln}")
                for cop, cf, cname in convs:
                    emit(cop, {cname: a}, lambda: cf(_ps_py(a)), ln, f"{cop}/lay/{d}/{ln}")
                for ln2, b in _ps_layouts(w)[:2]:
                    emit(o_mul, {"c1": a, "c2": b}, lambda: mul(_ps_py(a), _ps_py(b)), f"{ln}+{ln2}",
                         f"{o_mul}/lay/{d}/{ln}/{ln2}")
                    emit(o_mul, {"c1": b, "c2": a}, lambda: mul(_ps_py(b), _ps_py(a)), f"{ln2}+{ln}",
                         f"{o_mul}/layr/{d}/{ln2}/{ln}")
                    emit(o_div, {"c1": a, "c2": b}, lambda: div(_ps_py(a), _ps_py(b)), f"{ln}+{ln2}",
                         f"{o_div}/lay/{d}/{ln}/{ln2}", kind="tuple")
                    emit(o_div, {"c1": b, "c2": a}, lambda: div(_ps_py(b), _ps_py(a)), f"{ln2}+{ln}",
                         f"{o_div}/layr/{d}/{ln2}/{ln}", kind="tuple")
            # A 0-d array is a length-1 series; a stride-0 broadcast a length-n one.
            s0 = np.array(_ps_series(1, d, seed=15)[0])
            emit(o_mulx, {"c": A(s0)}, lambda: mulx(s0), "scalar_0d", f"{o_mulx}/0d/{d}")
            emit(o_mul, {"c1": A(s0), "c2": A(w)}, lambda: mul(s0, w), "scalar_0d", f"{o_mul}/0d/{d}")
            emit(o_div, {"c1": A(w), "c2": A(s0)}, lambda: div(w, s0), "scalar_0d", f"{o_div}/0d/{d}", kind="tuple")
            emit(o_pow, {"c": A(s0), "pow": 3}, lambda: pw(s0, 3), "scalar_0d", f"{o_pow}/0d/{d}")
            emit(o_fr, {"roots": A(s0)}, lambda: fr(s0), "scalar_0d", f"{o_fr}/0d/{d}")
            for cop, cf, cname in convs:
                emit(cop, {cname: A(s0)}, lambda: cf(s0), "scalar_0d", f"{cop}/0d/{d}")
            bsrc = _ps_series(1, d, seed=16)
            if bsrc[0] == 0:
                bsrc = np.ones(1, d)
            bc = np.broadcast_to(bsrc, (4,))
            emit(o_mulx, {"c": A(bsrc, bc)}, lambda: mulx(bc), "broadcast_1d", f"{o_mulx}/bcast/{d}")
            emit(o_mul, {"c1": A(bsrc, bc), "c2": A(w)}, lambda: mul(bc, w), "broadcast_1d", f"{o_mul}/bcast/{d}")
            emit(o_div, {"c1": A(bsrc, bc), "c2": A(w)}, lambda: div(bc, w), "broadcast_1d", f"{o_div}/bcast/{d}",
                 kind="tuple")
            emit(o_pow, {"c": A(bsrc, bc), "pow": 2}, lambda: pw(bc, 2), "broadcast_1d", f"{o_pow}/bcast/{d}")
            emit(o_fr, {"roots": A(bsrc, bc)}, lambda: fr(bc), "broadcast_1d", f"{o_fr}/bcast/{d}")
            for cop, cf, cname in convs:
                emit(cop, {cname: A(bsrc, bc)}, lambda: cf(bc), "broadcast_1d", f"{cop}/bcast/{d}")

        # ---------------- (E) Python-typed arguments ----------------
        py_series = [5, 2.5, -0.0, 1 + 2j, True, [1, 2, 0], [1.5, 2], [True, 2], [0, 0], [1, 2 ** 63],
                     [2 ** 64 - 1], [1 + 1j, 0j], (1, 2, 3), (0.5,), [3], (2 + 0j, -1), [[1.0, 2.0]]]
        others = [A(_ps_series(3, "float64", seed=17)), A(_ps_series(4, "float32", seed=18)),
                  A(_ps_series(2, "float16", seed=19)), 3, [0.5, -1.0, 0.0], (1, 1)]
        for a1 in py_series:
            tn = type(a1).__name__
            emit(o_mulx, {"c": a1}, lambda: mulx(_ps_py(a1)), "python", f"{o_mulx}/py/{tn}")
            emit(o_pow, {"c": a1, "pow": 2}, lambda: pw(_ps_py(a1), 2), "python", f"{o_pow}/py/{tn}")
            for cop, cf, cname in convs:
                emit(cop, {cname: a1}, lambda: cf(_ps_py(a1)), "python", f"{cop}/py/{tn}")
            for a2 in others:
                t2 = type(a2).__name__
                emit(o_mul, {"c1": a1, "c2": a2}, lambda: mul(_ps_py(a1), _ps_py(a2)), "python", f"{o_mul}/py/{tn}/{t2}")
                emit(o_mul, {"c1": a2, "c2": a1}, lambda: mul(_ps_py(a2), _ps_py(a1)), "python", f"{o_mul}/pyr/{t2}/{tn}")
                emit(o_div, {"c1": a1, "c2": a2}, lambda: div(_ps_py(a1), _ps_py(a2)), "python", f"{o_div}/py/{tn}/{t2}",
                     kind="tuple")
                emit(o_div, {"c1": a2, "c2": a1}, lambda: div(_ps_py(a2), _ps_py(a1)), "python",
                     f"{o_div}/pyr/{t2}/{tn}", kind="tuple")
        for roots in ([1, 2, 3], (1, -2), [1.5, 2j], [2 ** 63], [], (), [0.5], (2.5, -1.0, 4.0), [True, False],
                      [1, 2.5, 3 + 1j], 5, 2.5, 1j, True, 2 ** 70, "ab", [[1.0, 2.0], [3.0, 4.0]], [[1.0]], [[]]):
            tn = type(roots).__name__
            emit(o_fr, {"roots": roots}, lambda: fr(_ps_py(roots)), "python", f"{o_fr}/py/{tn}/{len(str(roots))}")

        # ---------------- (F) errors in NumPy's check order ----------------
        emp = np.zeros(0)
        two = np.zeros((2, 2))
        bl = np.array([True, False])
        good = _ps_series(3, "float64", seed=20)
        z0 = np.zeros(1)
        z2 = np.array([0.0, -0.0])
        zc = np.zeros(2, np.complex128)
        for tag, x in (("empty", emp), ("2d", two), ("bool", bl), ("str", "ab"), ("ragged", [[1, 2], [3]]),
                       ("emptylist", []), ("nested_empty", [[]])):
            xa = A(x) if isinstance(x, np.ndarray) else x
            emit(o_mulx, {"c": xa}, lambda: mulx(_ps_py(xa)), "error", f"{o_mulx}/err/{tag}")
            emit(o_pow, {"c": xa, "pow": 2}, lambda: pw(_ps_py(xa), 2), "error", f"{o_pow}/err/{tag}")
            for cop, cf, cname in convs:
                emit(cop, {cname: xa}, lambda: cf(_ps_py(xa)), "error", f"{cop}/err/{tag}")
            for tag2, a1, a2 in ((f"{tag}_first", xa, A(good)), (f"{tag}_second", A(good), xa)):
                emit(o_mul, {"c1": a1, "c2": a2}, lambda: mul(_ps_py(a1), _ps_py(a2)), "error", f"{o_mul}/err/{tag2}")
                emit(o_div, {"c1": a1, "c2": a2}, lambda: div(_ps_py(a1), _ps_py(a2)), "error", f"{o_div}/err/{tag2}",
                     kind="tuple")
        for tag, a1, a2 in (("2d_then_empty", A(two), A(emp)), ("empty_then_2d", A(emp), A(two)),
                            ("bool_then_empty", A(bl), A(emp))):
            emit(o_mul, {"c1": a1, "c2": a2}, lambda: mul(_ps_py(a1), _ps_py(a2)), "error", f"{o_mul}/err/{tag}")
            emit(o_div, {"c1": a1, "c2": a2}, lambda: div(_ps_py(a1), _ps_py(a2)), "error", f"{o_div}/err/{tag}",
                 kind="tuple")
        # Division by a zero series (ZeroDivisionError with an empty message), checked before the lengths; a NaN or
        # -0.0-only divisor, and the empty dividend against a zero divisor (as_series first).
        for tag, a1, a2 in (("zero", A(good), A(z0)), ("zeros", A(good), A(z2)), ("czero", A(good), A(zc)),
                            ("zero_short_dividend", A(z0), A(z0)), ("empty_vs_zero", A(emp), A(z0)),
                            ("negzero", A(good), A(np.array([-0.0]))), ("nan", A(good), A(np.array([1.0, np.nan]))),
                            ("int_zero", A(good), [0]), ("py_zero", [1, 2], 0), ("zero_vs_zero", 0, 0)):
            emit(o_div, {"c1": a1, "c2": a2}, lambda: div(_ps_py(a1), _ps_py(a2)), "error", f"{o_div}/err/{tag}",
                 kind="tuple")
        # fromroots: len() first (a scalar / 0-d array raise TypeError before as_series), then as_series' checks.
        for tag, x in (("0d", A(np.array(2.0))), ("0d_int", A(np.array(3))), ("2d", A(two)), ("2d_empty", A(np.zeros((0, 3)))),
                       ("bool", A(bl)), ("empty", A(emp)), ("str", "ab"), ("ragged", [[1, 2], [3]])):
            emit(o_fr, {"roots": x}, lambda: fr(_ps_py(x)), "error", f"{o_fr}/err/{tag}")

        # ---------------- (G) pow / maxpower kinds ----------------
        c = _ps_series(3, "float64", seed=21)
        for pv in (2.0, 2.5, -1, -1.0, -0.0, 0.0, 1.0, float("nan"), float("inf"), float("-inf"), 3.0000000000000004,
                   1e300, 2 ** 40, 17, 16, -2 ** 40):
            # polypow's default maxpower is None: a huge VALID power would loop for ever, so those cases cap it by hand.
            huge = (isinstance(pv, float) and pv >= 1e300) or (isinstance(pv, int) and pv >= 2 ** 40)
            kw = {"maxpower": 16} if power and huge else {}
            args = {"c": A(c), "pow": pv}
            args.update(kw)
            emit(o_pow, args, lambda: pw(c, pv, **kw), "pow_kind", f"{o_pow}/kind/{pv!r}/{kw}")
        for mp in (None, 0, 3, -1, 16, 5):
            for pv in (0, 1, 3, 5, 6, 3.0):
                emit(o_pow, {"c": A(c), "pow": pv, "maxpower": mp}, lambda: pw(c, pv, mp), "pow_kind",
                     f"{o_pow}/maxpower/{mp}/{pv!r}")
        # The as_series checks come before the power checks.
        for tag, cc, pv in (("empty_nan", A(emp), float("nan")), ("empty_neg", A(emp), -1), ("2d_frac", A(two), 2.5),
                            ("bool_big", A(bl), 1e300)):
            emit(o_pow, {"c": cc, "pow": pv, "maxpower": 16}, lambda: pw(_ps_py(cc), pv, 16), "error",
                 f"{o_pow}/err/{tag}")

        # ---------------- (H) long series across the managed / BLAS boundary ----------------
        if modname in ("polynomial", "chebyshev"):
            # cheb multiplies z-series of 2n - 1 coefficients: the boundaries sit at n = 8/9 (float64), 16/17 (float32),
            # 4/5 (complex128).
            if power:
                pairs = {"float64": ((15, 15), (16, 16), (15, 100), (16, 100), (40, 40), (1, 200), (100, 100)),
                         "float32": ((31, 31), (32, 32), (20, 200), (64, 64)),
                         "complex128": ((7, 7), (8, 8), (7, 50), (30, 30)),
                         "float16": ((40, 40), (100, 100)), "int64": ((50, 50),)}
            else:
                pairs = {"float64": ((8, 8), (9, 9), (8, 60), (9, 60), (30, 30)),
                         "float32": ((16, 16), (17, 17), (10, 80)),
                         "complex128": ((4, 4), (5, 5), (4, 30), (20, 20)),
                         "float16": ((20, 20), (50, 50)), "int64": ((25, 25),)}
            for dt, prs in pairs.items():
                for n1, n2 in prs:
                    c1 = _pa_random(n1, dt, seed=900 + n1) if dt != "int64" else _ps_series(n1, dt, seed=22)
                    c2 = _pa_random(n2, dt, seed=950 + n2) if dt != "int64" else _ps_series(n2, dt, seed=23)
                    emit(o_mul, {"c1": A(c1), "c2": A(c2)}, lambda: mul(c1, c2), "long_1d", f"{o_mul}/long/{dt}/{n1}/{n2}")
                    emit(o_mul, {"c1": A(c2), "c2": A(c1)}, lambda: mul(c2, c1), "long_1d", f"{o_mul}/longr/{dt}/{n2}/{n1}")
            for dt, ns in (("float64", (8, 9, 15, 16)), ("float32", (16, 17, 31, 32)), ("complex128", (4, 5, 7, 8)),
                           ("float16", (12,))):
                for n in ns:
                    c = _pa_random(n, dt, seed=1000 + n)
                    for k in (2, 3):
                        emit(o_pow, {"c": A(c), "pow": k}, lambda: pw(c, k), "long_1d", f"{o_pow}/long/{dt}/{n}/{k}")
            for dt, ks in (("float64", (20, 31, 32, 40)), ("float32", (40, 64, 65)), ("complex128", (14, 15, 16, 17)),
                           ("float16", (24,))):
                for k in ks:
                    roots = _pa_random(k, dt, seed=1100 + k, scale=1.5)
                    emit(o_fr, {"roots": A(roots)}, lambda: fr(roots), "long_1d", f"{o_fr}/long/{dt}/{k}")
        else:
            # The recurrence bases: no convolution, long series stay portable (the arena's growth, the recurrence
            # loop's Keep moves, the division's unit series).
            for dt in ("float64", "complex128", "float32", "float16"):
                for n1, n2 in ((20, 20), (33, 12), (12, 33), (40, 40)):
                    c1 = _pa_random(n1, dt, seed=1200 + n1, scale=0.5)
                    c2 = _pa_random(n2, dt, seed=1250 + n2, scale=0.5)
                    emit(o_mul, {"c1": A(c1), "c2": A(c2)}, lambda: mul(c1, c2), "long_1d", f"{o_mul}/long/{dt}/{n1}/{n2}")
                for n1, n2 in ((40, 13), (33, 32), (25, 2)):
                    c1 = _pa_random(n1, dt, seed=1300 + n1, scale=0.5)
                    c2 = _pa_random(n2, dt, seed=1350 + n2, scale=0.5)
                    emit(o_div, {"c1": A(c1), "c2": A(c2)}, lambda: div(c1, c2), "long_1d",
                         f"{o_div}/long/{dt}/{n1}/{n2}", kind="tuple")
                c = _pa_random(6, dt, seed=1400, scale=0.5)
                emit(o_pow, {"c": A(c), "pow": 5}, lambda: pw(c, 5), "long_1d", f"{o_pow}/long/{dt}")
                roots = _pa_random(24, dt, seed=1450, scale=1.0)
                emit(o_fr, {"roots": A(roots)}, lambda: fr(roots), "long_1d", f"{o_fr}/long/{dt}")
        for dt in ("float64", "complex128", "float32", "float16"):
            for n in (257, 1000):
                c = _pa_random(n, dt, seed=1500 + n)
                emit(o_mulx, {"c": A(c)}, lambda: mulx(c), "long_1d", f"{o_mulx}/long/{dt}/{n}")
            for cop, cf, cname in convs:
                for n in (30, 64):
                    c = _pa_random(n, dt, seed=1600 + n, scale=0.25)
                    emit(cop, {cname: A(c)}, lambda: cf(c), "long_1d", f"{cop}/long/{dt}/{n}")
            if power:
                for n1, n2 in ((200, 7), (60, 59), (100, 3)):
                    c1 = _pa_random(n1, dt, seed=1700 + n1)
                    c2 = _pa_random(n2, dt, seed=1750 + n2)
                    emit(o_div, {"c1": A(c1), "c2": A(c2)}, lambda: div(c1, c2), "long_1d",
                         f"{o_div}/long/{dt}/{n1}/{n2}", kind="tuple")
            elif modname == "chebyshev":
                for n1, n2 in ((60, 7), (40, 39), (50, 3)):
                    c1 = _pa_random(n1, dt, seed=1800 + n1)
                    c2 = _pa_random(n2, dt, seed=1850 + n2)
                    emit(o_div, {"c1": A(c1), "c2": A(c2)}, lambda: div(c1, c2), "long_1d",
                         f"{o_div}/long/{dt}/{n1}/{n2}", kind="tuple")

        # ---------------- (I) result flags ----------------
        for dt in ("float64", "float16", "complex128", "int64"):
            for pat in ("moderate", "tz", "lastnz", "allzero"):
                c = _ps_series(4, dt, pat, seed=24)
                d = _ps_series(3, dt, seed=25)
                emit(o_mulx, {"c": A(c)}, lambda: mulx(c), "flags", f"{o_mulx}/flags/{dt}/{pat}", facet="flags")
                for k in (0, 1, 2):
                    emit(o_pow, {"c": A(c), "pow": k}, lambda: pw(c, k), "flags", f"{o_pow}/flags/{dt}/{pat}/{k}",
                         facet="flags")
                emit(o_mul, {"c1": A(c), "c2": A(d)}, lambda: mul(c, d), "flags", f"{o_mul}/flags/{dt}/{pat}",
                     facet="flags")
                emit(o_div, {"c1": A(c), "c2": A(d)}, lambda: div(c, d), "flags", f"{o_div}/flags/{dt}/{pat}",
                     kind="tuple", facet="flags")
                emit(o_div, {"c1": A(d), "c2": A(c)}, lambda: div(d, c), "flags", f"{o_div}/flagsr/{dt}/{pat}",
                     kind="tuple", facet="flags")
                for cop, cf, cname in convs:
                    emit(cop, {cname: A(c)}, lambda: cf(c), "flags", f"{cop}/flags/{dt}/{pat}", facet="flags")
            for k in (0, 1, 3):
                roots = _pa_roots(k, dt, seed=26)
                emit(o_fr, {"roots": A(roots)}, lambda: fr(roots), "flags", f"{o_fr}/flags/{dt}/{k}", facet="flags")
        # The product whose trailing coefficient cancels (trimseq returns a VIEW) and a division whose remainder trims.
        for dt in ("float64", "complex128"):
            c1 = np.array([1.0, 1.0], dt)
            c2 = np.array([1.0, -1.0], dt)
            emit(o_mul, {"c1": A(c1), "c2": A(c2)}, lambda: mul(c1, c2), "flags", f"{o_mul}/flags_cancel/{dt}",
                 facet="flags")
            emit(o_mul, {"c1": A(c1), "c2": A(c2)}, lambda: mul(c1, c2), "c_contiguous_1d", f"{o_mul}/cancel/{dt}")
            c3 = np.array([2.0, 3.0, 1.0], dt)
            emit(o_div, {"c1": A(c3), "c2": A(c1)}, lambda: div(c3, c1), "flags", f"{o_div}/flags_exact/{dt}",
                 kind="tuple", facet="flags")
            emit(o_div, {"c1": A(c3), "c2": A(c1)}, lambda: div(c3, c1), "c_contiguous_1d", f"{o_div}/exact/{dt}",
                 kind="tuple")

        # ---------------- (J) fromroots' root kinds ----------------
        rkinds = [("nan", [1.0, np.nan, 2.0]), ("inf", [1.0, np.inf, -2.0]), ("neginf", [-np.inf, 0.5]),
                  ("repeat", [2.0, 2.0, 2.0, -1.0]), ("allnegzero", [-0.0, -0.0, -0.0]), ("allzero", [0.0, 0.0]),
                  ("unsorted", [3.0, -1.0, 2.0, -5.0, 0.5, 4.0, -2.5]), ("big", [1e200, 1e200, -3e150]),
                  ("subnormal", [5e-324, -1e-310, 2.0]), ("ints_as_float", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]
        for tag, vals in rkinds:
            for dt in ("float64", "float32", "float16"):
                roots = np.array(vals).astype(dt)
                emit(o_fr, {"roots": A(roots)}, lambda: fr(roots), "roots_kind", f"{o_fr}/rk/{tag}/{dt}")
        ckinds = [("conj", [1 + 2j, 1 - 2j, -0.5 + 0j]), ("lexi", [1 + 3j, 1 - 1j, 0.5 + 9j, 1 + 0.5j]),
                  ("realonly", [2 + 0j, -1 + 0j, 0.5 + 0j]), ("nan", [complex(np.nan, 1), 1 + 1j]),
                  ("inf", [complex(np.inf, 0), 1 + 1j])]
        for tag, vals in ckinds:
            roots = np.array(vals, np.complex128)
            emit(o_fr, {"roots": A(roots)}, lambda: fr(roots), "roots_kind", f"{o_fr}/rk/c/{tag}")

        # ---------------- (K) underflow and overflow ----------------
        for dt in ("float64", "float32", "complex128"):
            tiny = np.array([1.0, 1e-200 if dt != "float32" else 1e-30], dt)
            huge = np.array([1e200 if dt != "float32" else 1e30, 1e200 if dt != "float32" else 1e30], dt)
            for tag, c in (("tiny", tiny), ("huge", huge)):
                for k in (2, 3):
                    emit(o_pow, {"c": A(c), "pow": k}, lambda: pw(c, k), "extreme", f"{o_pow}/ext/{dt}/{tag}/{k}")
                emit(o_mul, {"c1": A(c), "c2": A(c)}, lambda: mul(c, c), "extreme", f"{o_mul}/ext/{dt}/{tag}")
                emit(o_div, {"c1": A(np.concatenate([c, c])), "c2": A(c)}, lambda: div(np.concatenate([c, c]), c),
                     "extreme", f"{o_div}/ext/{dt}/{tag}", kind="tuple")
                emit(o_mulx, {"c": A(c)}, lambda: mulx(c), "extreme", f"{o_mulx}/ext/{dt}/{tag}")
                for cop, cf, cname in convs:
                    emit(cop, {cname: A(c)}, lambda: cf(c), "extreme", f"{cop}/ext/{dt}/{tag}")

        # ---------------- (L) chebmulx's fused kernel: special values through its vector stage ----------------
        if modname == "chebyshev":
            # NumSharp runs chebmulx's three array statements as one pass whose vector stage halves by a multiply by 0.5
            # (float16: then rounds the half to the float16 grid, which only a subnormal half leaves; complex: Smith's
            # branch with the divisor 2+0j prepared once). Series long enough for that stage, holding the special
            # values, both float16 halving regimes, and float32 / float64 subnormals.
            for dt in ("float64", "float32", "float16", "complex128"):
                for n in (40, 101):
                    c = _ps_series(n, dt, "special")
                    emit(o_mulx, {"c": A(c)}, lambda: mulx(c), "long_1d", f"{o_mulx}/vspecial/{dt}/{n}")
            # Every float16 bit pattern below 2^-12 (exponent fields 0-2, both signs: the halves that round and their
            # neighbours) plus the infinities, NaNs and the largest finite values, each once as c[j-1] and once as c[j+1]
            # of an output, beside seeded random partners; a final 1.0 keeps as_series' trim away.
            pats = np.concatenate([np.arange(0x0000, 0x0C00), np.arange(0x8000, 0x8C00),
                                   np.array([0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0x7BFF, 0xFBFF])]).astype(np.uint16)
            rng = np.random.default_rng(1900)
            for shift in (0, 1):
                bits = np.empty(2 * pats.size + 2, np.uint16)
                bits[shift:shift + 2 * pats.size:2] = pats
                bits[1 - shift:1 - shift + 2 * pats.size:2] = rng.integers(0, 65536, pats.size).astype(np.uint16)
                bits[-2:] = 0x3C00
                c = bits.view(np.float16)
                emit(o_mulx, {"c": A(c)}, lambda: mulx(c), "long_1d", f"{o_mulx}/f16grid/{shift}")
            for dt, lo, hi in (("float64", -1e-307, 1e-307), ("float32", -1e-37, 1e-37)):
                c = np.random.default_rng(1950).uniform(lo, hi, 300).astype(dt)
                c[-1] = 1.0
                emit(o_mulx, {"c": A(c)}, lambda: mulx(c), "long_1d", f"{o_mulx}/subnormal/{dt}")

        # ---------------- (M) argument kinds (the wholeness pass) ----------------
        # (M1) pow through the object-typed overloads: int(pow) and `power != pow` with Python's / NumPy's semantics for
        # every kind a C# caller can spell (NDPolyPowerArgument.cs) — bool, np.float16 (Half), a str (int()'s grammar,
        # then never equal), None / list / tuple / complex (int()'s TypeError), 0-d arrays of every kind (a complex one
        # int()'s TypeError), arrays of one or more dims (TypeError whatever their size), Python ints past int64.
        one = np.array([1.5])                  # one coefficient: every product of a power of it is one cheap term
        three = _ps_series(3, "float64", seed=27)
        f16 = np.float16
        pkinds = [("true", True), ("false", False), ("h2", f16(2.0)), ("h2_5", f16(2.5)), ("hnan", f16("nan")),
                  ("hinf", f16("inf")), ("s3", "3"), ("s_sp", " 3 "), ("s_us", "3_0"), ("sx", "x"), ("s_empty", ""),
                  ("s_neg", "-1"), ("none", None), ("list", [3]), ("tuple", (3,)), ("cplx", 2 + 0j),
                  ("nd0_i64", A(np.array(3))), ("nd0_f64", A(np.array(2.0))), ("nd0_f64_frac", A(np.array(2.5))),
                  ("nd0_bool", A(np.array(True))), ("nd0_c128", A(np.array(2 + 0j))),
                  ("nd0_u8", A(np.array(250, np.uint8))), ("nd0_f16", A(np.array(2, np.float16))),
                  ("nd0_i8_neg", A(np.array(-1, np.int8))), ("nd0_nan", A(np.array(np.nan))),
                  ("nd0_f32_inf", A(np.array(np.inf, np.float32))), ("nd1", A(np.array([3]))), ("nd2", A(np.array([[3]]))),
                  ("nd_2elem", A(np.array([2, 3]))), ("nd_empty", A(np.zeros(0))), ("big70", 2 ** 70),
                  ("u64max", 2 ** 64 - 1), ("nd0_u64max", A(np.array(2 ** 64 - 1, np.uint64)))]
        for tag, pv in pkinds:
            big = tag in ("nd0_u8", "big70", "u64max", "nd0_u64max")
            cc = one if big else three
            # A huge VALID power with no limit (polypow's None) would loop for ever: those cap it at 16 like section G.
            args = {"c": A(cc), "pow": pv}
            kw = {}
            if power and big:
                args["maxpower"] = 16
                kw = {"maxpower": 16}
            emit(o_pow, args, lambda: pw(cc, _ps_py(pv), **kw), "pow_kind", f"{o_pow}/argkind/{tag}")
        # (M2) maxpower through the object-typed overload: `power > maxpower` exactly for Python ints / bools / floats,
        # in the dtype of np.float16 or of an ndarray (NEP 50: the int rounded to float16 / float32 first, complex
        # compared lexicographically, bool as int64), the ndarray result then tested for truth (one element only).
        mkinds = [("f2_5", 2.5), ("fnan", float("nan")), ("fninf", float("-inf")), ("finf", float("inf")), ("big70", 2 ** 70),
                  ("true", True), ("false", False), ("h1_5", f16(1.5)), ("hnan", f16("nan")), ("cplx", 1 + 0j), ("s", "5"),
                  ("list", [5]), ("tuple", (5,)), ("nd0_5", A(np.array(5))), ("nd0_1", A(np.array(1))),
                  ("nd0_nan", A(np.array(np.nan))), ("nd0_c5", A(np.array(5 + 0j))), ("nd0_c1", A(np.array(1 + 0j))),
                  ("nd0_c2m5", A(np.array(2 - 5j))), ("nd0_c2p5", A(np.array(2 + 5j))), ("nd0_bool", A(np.array(True))),
                  ("nd1_5", A(np.array([5]))), ("nd2_1", A(np.array([[1]]))), ("nd_2elem", A(np.array([1, 5]))),
                  ("nd_empty", A(np.zeros(0))), ("nd0_u64max", A(np.array(2 ** 64 - 1, np.uint64))),
                  ("nd0_i8", A(np.array(2, np.int8))), ("nd0_f16", A(np.array(2.5, np.float16))),
                  ("nd0_f32", A(np.array(2.5, np.float32)))]
        for mtag, mv in mkinds:
            for ptag, pv in (("int2", 2), ("int3", 3), ("f2", 2.0), ("nd0_2", A(np.array(2))), ("true", True)):
                emit(o_pow, {"c": A(three), "pow": pv, "maxpower": mv}, lambda: pw(three, _ps_py(pv), _ps_py(mv)),
                     "pow_kind", f"{o_pow}/maxkind/{mtag}/{ptag}")
        # The comparison's dtype decides at the boundaries: 2049 rounds to float16 2048 (not exceeded, so the power is
        # computed), 70000 to float16 inf; the weak int's conversion into a float / complex loop overflows past 2**1024
        # (OverflowError, before an ndarray's truth test), into a bool one past int64; integer ndarrays and Python floats
        # compare exactly.
        for tag, pv, mv in (("h2048_2049", 2049, f16(2048)), ("nd_h2048_2049", 2049, A(np.array(2048, np.float16))),
                            ("h65504_70000", 70000, f16(65504)), ("nd_h65504_70000", 70000, A(np.array(65504, np.float16))),
                            ("h_big1100", 2 ** 1100, f16(1)), ("nd_f64_big1100", 2 ** 1100, A(np.array(1.0))),
                            ("nd_c_big1100", 2 ** 1100, A(np.array(1 + 0j))), ("nd_bool_big63", 2 ** 63, A(np.array(True))),
                            ("nd_bool_big62", 2 ** 62, A(np.array(True))),
                            ("nd_u64_big1100", 2 ** 1100, A(np.array(2 ** 64 - 1, np.uint64))),
                            ("nd_i8_big70", 2 ** 70, A(np.array(5, np.int8))), ("f_big1100", 2 ** 1100, 1.0),
                            ("f_exact", 2 ** 53 + 1, float(2 ** 53)), ("nd_empty_big1100", 2 ** 1100, A(np.zeros(0))),
                            ("nd_2elem_big1100", 2 ** 1100, A(np.zeros(2))), ("nd_empty_int", 3, A(np.zeros(0, np.int64)))):
            emit(o_pow, {"c": A(one), "pow": pv, "maxpower": mv}, lambda: pw(one, pv, _ps_py(mv)), "pow_kind",
                 f"{o_pow}/maxedge/{tag}")
        # (M3) the object / str refusal is DEFERRED to where NumPy computes with Python objects: as_series' checks of
        # every argument, np.array's raggedness, div's zero-divisor test (run on the object copies) and pow's checks come
        # first, in NumPy's order. Only outcomes NumPy reaches before any object arithmetic are recorded.
        obj1 = [1.0, None]
        strl = [1.0, "a"]
        NO = object()
        for tag, c1v, c2v in (("obj_empty", None, A(np.zeros(0))), ("empty_obj", A(np.zeros(0)), obj1),
                              ("obj_2d", obj1, A(np.zeros((2, 2)))), ("obj_ragged", None, [[1, 2], [3]]),
                              ("ragged_obj", [[1, 2], [3]], None), ("str_f", strl, A(three)), ("f_str", A(three), strl),
                              ("str_bool", strl, A(np.array([True]))), ("str_obj", strl, obj1),
                              ("obj_zero", None, A(np.zeros(1))), ("obj_zeros", obj1, A(np.array([0.0, -0.0]))),
                              ("obj_false", None, A(np.array([False]))), ("obj_c0", None, A(np.zeros(1, np.complex128))),
                              ("obj_i0", None, A(np.zeros(1, np.int32))), ("big_zero", 2 ** 70, A(np.zeros(1))),
                              ("str_2d", [["a", 1.0]], A(three)), ("str_tuple", ("x",), A(three)),
                              ("str_empty", strl, A(np.zeros(0)))):
            for opx, fx, kd in ((o_mul, mul, "array"), (o_div, div, "tuple")):
                call = (lambda fx=fx, c1v=c1v, c2v=c2v: fx(_ps_py(c1v), _ps_py(c2v)))
                if _pa_object_land(call):
                    continue
                emit(opx, {"c1": c1v, "c2": c2v}, call, "object_order", f"{opx}/objorder/{tag}", kind=kd)
        for tag, cv, pv, mv in (("obj_neg", None, -1, NO), ("obj_frac", None, 2.5, NO), ("obj_toobig", obj1, 17, 16),
                                ("obj_strx", None, "x", NO), ("obj_cplx", None, 1 + 0j, NO), ("obj_strmax", None, 2, "5"),
                                ("obj_nanpow", None, float("nan"), NO), ("str_neg", strl, -1, NO), ("big_neg", 2 ** 70, -1, NO),
                                ("empty_strpow", A(np.zeros(0)), "x", NO), ("obj_huge_cap", obj1, 1e19, 16),
                                ("str_2d_pow", [["a", 1.0]], 2, NO)):
            args = {"c": cv, "pow": pv}
            if mv is not NO:
                args["maxpower"] = mv
            call = (lambda cv=cv, pv=pv, mv=mv: pw(_ps_py(cv), pv) if mv is NO else pw(_ps_py(cv), pv, mv))
            if _pa_object_land(call):
                continue
            emit(o_pow, args, call, "object_order", f"{o_pow}/objorder/{tag}")
        for tag, cv in (("obj_2d", [[None, 1.0]]), ("str_2d", [["a", 1.0]]), ("str_only", ["a", "b"]), ("str_tuple", ("x",)),
                        ("str_list_one", ["x"])):
            singles = [(o_mulx, mulx, "c")] + [(cop, cf, cname) for cop, cf, cname in convs]
            for opx, fx, cname in singles:
                call = (lambda fx=fx, cv=cv: fx(_ps_py(cv)))
                if _pa_object_land(call):
                    continue
                emit(opx, {cname: cv}, call, "object_order", f"{opx}/objorder/{tag}")
            call = (lambda cv=cv: fr(_ps_py(cv)))
            if not _pa_object_land(call):
                emit(o_fr, {"roots": cv}, call, "object_order", f"{o_fr}/objorder/{tag}")
        # (M4) complex series holding infinities and NaNs. cblas' zdotu builds its result with C99 complex arithmetic
        # (re + im*_Complex_I), so an infinite / NaN imaginary part turns the real part into NaN — while a ONE-term factor
        # is np.convolve's reversed one-element kernel, whose negative stride keeps CDOUBLE_dot off cblas (plain loop, no
        # such NaN). Below zdotu's vector block both are managed-exact, hence portable.
        cvals = [("inf_re", [complex(np.inf, 0), 1 + 1j, -2 + 0.5j]), ("ninf_im", [complex(0, -np.inf), 1 + 1j]),
                 ("inf_both", [complex(np.inf, np.inf), 2 - 1j, 0.5j]), ("big", [complex(1e308, 1e308), 3 + 1j, -1 + 2j]),
                 ("nan_im", [complex(1, np.nan), 1 + 1j, 3 - 2j]), ("negzero_inf", [complex(-0.0, np.inf), -0.5 + 0j])]
        fac2 = np.array([2 - 1j, 1 + 3j])
        fac1 = np.array([2 + 1j])
        for tag, vals in cvals:
            cv = np.array(vals, np.complex128)
            single = cv[:1].copy()
            emit(o_fr, {"roots": A(cv)}, lambda: fr(cv), "cspecial", f"{o_fr}/cspecial/{tag}")
            emit(o_mul, {"c1": A(cv), "c2": A(fac2)}, lambda: mul(cv, fac2), "cspecial", f"{o_mul}/cspecial/{tag}")
            emit(o_mul, {"c1": A(fac2), "c2": A(cv)}, lambda: mul(fac2, cv), "cspecial", f"{o_mul}/cspecialr/{tag}")
            emit(o_mul, {"c1": A(cv), "c2": A(fac1)}, lambda: mul(cv, fac1), "cspecial", f"{o_mul}/cspecial1/{tag}")
            emit(o_mul, {"c1": A(fac1), "c2": A(cv)}, lambda: mul(fac1, cv), "cspecial", f"{o_mul}/cspecial1r/{tag}")
            emit(o_mul, {"c1": A(single), "c2": A(fac1)}, lambda: mul(single, fac1), "cspecial", f"{o_mul}/cspecial11/{tag}")
            emit(o_pow, {"c": A(cv), "pow": 3}, lambda: pw(cv, 3), "cspecial", f"{o_pow}/cspecial/{tag}")
            emit(o_pow, {"c": A(single), "pow": 3}, lambda: pw(single, 3), "cspecial", f"{o_pow}/cspecial1/{tag}")
            emit(o_div, {"c1": A(np.concatenate([cv, fac2])), "c2": A(fac2)},
                 lambda: div(np.concatenate([cv, fac2]), fac2), "cspecial", f"{o_div}/cspecial/{tag}", kind="tuple")
            emit(o_mulx, {"c": A(cv)}, lambda: mulx(cv), "cspecial", f"{o_mulx}/cspecial/{tag}")
            for cop, cf, cname in convs:
                emit(cop, {cname: A(cv)}, lambda: cf(cv), "cspecial", f"{cop}/cspecial/{tag}")

    # Char: NumSharp's uint16-like dtype converts to float64 exactly as uint16 does (the house weave).
    cases += _relabel_dtype([c for c in cases if "/uint16/" in (c.get("id") or "") and not c.get("expects_throw")],
                            "uint16", "char")
    if skipped[0]:
        print(f"  (skipped {skipped[0]} complex64 / object-dtype cells — #569 / no object dtype)")
    print(f"  portable {len(cases)}, host-pinned {len(host)}")
    return cases, host


# ---- numpy.polynomial companion matrices and roots (plan U7) -----------------------------------------------------------
#
# {p}companion(c) and {p}roots(c) for the six bases. The companion is NumPy's statements over the as_series copy — the
# zero matrix, diagonals assigned through mat.reshape(-1)[k::n+1], one in-place update of the last column, whose float64
# helper vectors (cheb / leg / herm / herme) put a float16 / float32 series' last column through float64 and one rounding
# — so it is portable. {p}roots is np.linalg.eigvals of the companion (ROTATED [::-1, ::-1] for every basis but the power
# series) sorted in place: a constant series' empty array, a linear series' one scalarmath root and every error NumPy
# raises before LAPACK (float16's linalg TypeError, a non-finite companion's LinAlgError) are portable; a call that reaches
# LAPACK geev goes to polyroots_parity.jsonl, host-pinned like linalg_parity (threads = 1).
#
# Recorded as NumPy produces them: real roots of a float64 series are a VIEW of eigvals' complex result (OWNDATA false,
# the flags facet records it); a float32 series' complex roots are complex64, which the replay compares by VALUE (NumSharp
# carries them as complex128 — np.linalg.eigvals rounds a float32 operand's complex result to complex64 values exactly).
# Results whose sorted order is not fixed by value — two elements equal by value but different in bits (+0.0 / -0.0) —
# are skipped: NumPy's SIMD sort orders them by CPU, NumSharp's sort deterministically (the documented divergence of
# fromroots, pinned by a unit test).
#
# Sections: (A) dtype x length; (B) full-mantissa values; (C) trim / special patterns; (D) layouts; (E) Python-typed and
# object arguments (only outcomes NumPy reaches before object arithmetic); (F) errors in NumPy's order; (G) roots of known
# polynomials (real, repeated, conjugate pairs, clustered, wide); (H) result flags; (I) long series; (J) the linear
# roots' scalarmath with special values.

class _PREigRecorder:
    """Records whether a polynomial call reached LAPACK: the modules call np.linalg.eigvals through the numpy module at
    call time, so replacing the attribute for one call sees the companion matrix eigvals received. eigvals asserts
    finiteness, then rejects float16 / object dtypes, and only then runs geev — so geev ran exactly when the matrix is
    finite and float32 / float64 / complex128."""

    def __init__(self):
        self.lapack = False
        self._orig = None

    def __enter__(self):
        self._orig = np.linalg.eigvals
        orig = self._orig
        rec = self

        def eig(a):
            m = np.asarray(a)
            with np.errstate(all="ignore"):
                if m.dtype in (np.float32, np.float64, np.complex128) and m.size and bool(np.isfinite(m).all()):
                    rec.lapack = True
            return orig(a)

        np.linalg.eigvals = eig
        return self

    def __exit__(self, *exc):
        np.linalg.eigvals = self._orig
        return False


def _pr_order_fixed(a):
    """Whether a 1-D result's order is decided by values alone: no two elements equal by value but different in bits."""
    a = np.asarray(a)
    if a.ndim != 1 or a.size < 2 or a.dtype.kind not in "fc":
        return True
    raw = np.ascontiguousarray(a).view(np.uint8).reshape(a.size, -1)
    for i in range(a.size):
        for j in range(i + 1, a.size):
            if a[i] == a[j] and not np.array_equal(raw[i], raw[j]):
                return False
    return True


def gen_polyroots():
    cases = []
    host = []
    counter = [0]
    skipped = {"dtype": 0, "order": 0, "object": 0}
    ok_dtypes = set(ALL_DTYPES) | {"complex64"}
    A = _PSArr

    def flags(a):
        return np.array([a.flags.c_contiguous, a.flags.f_contiguous, a.flags.owndata])

    def emit(op, cv, call, layout, cid, facet=None):
        operands = []
        params = {"c": _ps_enc(cv, operands)}
        if facet:
            params["facet"] = facet
        n = counter[0]
        counter[0] += 1
        rec = _PREigRecorder()
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings(), rec:
                warnings.simplefilter("ignore")
                r = call()
        except Exception as e:
            (host if rec.lapack else cases).append(
                {"id": f"{cid}/{n}", "op": op, "params": params, "operands": operands, "expected": {},
                 "expects_throw": True, "error": _poly_exc(e), "layout": layout, "valueclass": "error"})
            return
        dest = host if rec.lapack else cases
        a = np.asarray(r)
        if facet == "flags":
            a = flags(a)
        elif a.dtype.name not in ok_dtypes:
            skipped["dtype"] += 1
            return
        elif not _pr_order_fixed(a):
            skipped["order"] += 1
            return
        dest.append(_case(op, params, operands, _arr_expected(a), layout, "polyroots", cid=f"{cid}/{n}"))

    def both(ops, cv, layout, cid, facet=None):
        """Emits companion and roots of one argument (`cv`: an _PSArr or a Python-typed value)."""
        for op, f in ops:
            emit(op, cv, lambda: f(_ps_py(cv)), layout, f"{op}/{cid}", facet)

    sub_dtypes = ["float64", "float32", "float16", "complex128", "int32", "uint64", "bool", "int8"]
    patterns = ("tz", "allzero", "lastnz", "special", "nantail")
    pat_dtypes = ("float64", "float32", "float16", "complex128", "int64")

    for modname, p in POLY_MODULES:
        mod = _poly_module(modname)
        comp = getattr(mod, p + "companion")
        roots = getattr(mod, p + "roots")
        fromroots = getattr(mod, p + "fromroots")
        o_comp, o_roots = f"{modname}.{p}companion", f"{modname}.{p}roots"
        ops = ((o_comp, comp), (o_roots, roots))
        full = modname in ("polynomial", "chebyshev", "legendre")
        dts = ALL_DTYPES if full else sub_dtypes

        # ---------------- (A) dtype x length ----------------
        for dt in dts:
            for n in (1, 2, 3, 4, 5, 6, 9, 13):
                c = _ps_series(n, dt, seed=1)
                both(ops, A(c), "c_contiguous_1d", f"dt/{dt}/{n}")

        # ---------------- (B) full-mantissa values ----------------
        for dt in ("float64", "float32", "float16", "complex128"):
            for n in (2, 3, 4, 5, 7, 12, 20, 33):
                c = _pa_random(n, dt, seed=100 + n)
                both(ops, A(c), "c_contiguous_1d", f"rand/{dt}/{n}")
                # A small leading coefficient: the quotient c[:-1] / c[-1] grows large (float16 overflows to inf).
                c2 = c.copy()
                c2[-1] = c2[-1] * np.asarray(1e-3, dtype=c2.dtype)
                both(ops, A(c2), "c_contiguous_1d", f"randsmall/{dt}/{n}")

        # ---------------- (C) trim / special patterns ----------------
        for dt in pat_dtypes:
            for pat in patterns:
                for n in (3, 5, 8):
                    c = _ps_series(n, dt, pat, seed=8)
                    both(ops, A(c), "c_contiguous_1d", f"pat/{dt}/{pat}/{n}")

        # ---------------- (D) layouts ----------------
        for d in ("float64", "complex128", "float32", "float16", "int16"):
            for tag, v in (("tz", _ps_series(6, d, "tz", seed=13)),
                           ("rand", _pa_random(7, d, seed=14) if d != "int16" else _ps_series(7, d, seed=14))):
                for ln, a in _ps_layouts(v):
                    both(ops, a, ln, f"lay/{d}/{tag}/{ln}")
            # A 0-d array is a one-term series; a stride-0 broadcast a series of equal terms.
            s0 = np.array(_ps_series(1, d, seed=15)[0])
            both(ops, A(s0), "scalar_0d", f"0d/{d}")
            bsrc = _ps_series(1, d, seed=16)
            if bsrc[0] == 0:
                bsrc = np.ones(1, d)
            for k in (2, 3, 5):
                bc = np.broadcast_to(bsrc, (k,))
                both(ops, A(bsrc, bc), "broadcast_1d", f"bcast/{d}/{k}")

        # ---------------- (E) Python-typed and object arguments ----------------
        f16 = np.float16
        py_series = [5, 2.5, -0.0, 1 + 2j, [1, 2, 3], [1.5, -2, 0.5], (1, -2, 0.5), [1.5, 2j, 1], [2 ** 63, 1, 1],
                     [2 ** 64 - 1, 3], [0, 0, 0], [1, 2, 0, 0], [0.0, -0.0, 1.0], (2 + 0j, -1), [3], (0.5,), (),
                     [f16(1.5), 2.0, f16(-0.5)], [True, 2, 3], [1, 2 ** 62], (1, 1, 1, 1)]
        for cv in py_series:
            both(ops, cv, "python", f"py/{type(cv).__name__}/{len(str(cv))}")
        # The object / str refusal is deferred to where NumPy computes with Python objects: the length check runs on the
        # TRIMMED object array first (so [None, 0] is NumPy's ValueError), as_series' common-type check rejects a str array.
        for cv in ([None], [None, 0], [None, 0, 0], [None, 0.0, -0.0], [2 ** 70, 0], [2 ** 70, 0, 0], [0, None],
                   [None, 1, 0], [2 ** 70, 1, 2], ["a", 1], ["a"], [1.0, "x", 0], None, "ab", [None, 1.5],
                   [[None, 1.0]], [[1.0, "a"]], [f16(0), None]):
            for op, f in ops:
                call = (lambda f=f, cv=cv: f(_ps_py(cv)))
                if _pa_object_land(call):
                    skipped["object"] += 1
                    continue
                emit(op, cv, call, "object_order", f"{op}/obj/{len(str(cv))}/{type(cv).__name__}")

        # ---------------- (F) errors in NumPy's check order ----------------
        for tag, x in (("empty", A(np.zeros(0))), ("2d", A(np.zeros((2, 2)))), ("2d_empty", A(np.zeros((0, 3)))),
                       ("3d", A(np.ones((1, 2, 2)))), ("bool", A(np.array([True, False, True]))), ("str", "ab"),
                       ("ragged", [[1, 2], [3]]), ("emptylist", []), ("nested_empty", [[]]),
                       ("f16_nan", A(np.array([1.0, np.nan, 2.0], np.float16))),
                       ("f16_inf", A(np.array([np.inf, 1.0, 2.0], np.float16)))):
            both(ops, x, "error", f"err/{tag}")

        # ---------------- (G) roots of known polynomials ----------------
        known = [("distinct", [-1.5, 0.25, 2.0]), ("repeat2", [1.0, 1.0, -2.0]), ("repeat3", [0.5, 0.5, 0.5, -1.0]),
                 ("clustered", [1.0, 1.0001, 1.0002, -3.0]), ("wide", [1e-3, 2.0, 1e3]),
                 ("wilkinson", [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]), ("conj", [1 + 2j, 1 - 2j, -0.5 + 0j]),
                 ("lexi", [1 + 3j, 1 - 1j, 0.5 + 9j, 1 + 0.5j]), ("imag", [2j, -2j]), ("mixed", [3.0, -1 + 1j, -1 - 1j, 0.5])]
        for tag, rts in known:
            iscplx = any(isinstance(r, complex) for r in rts)
            for dt in (("complex128",) if iscplx else ("float64", "float32", "complex128")):
                with np.errstate(all="ignore"):
                    c = np.asarray(fromroots(np.array(rts, dtype=dt))).astype(dt)
                both(ops, A(c), "roots_kind", f"known/{tag}/{dt}")
        # Real coefficients with complex roots in float32: NumPy's complex64 result (value-compared by the replay).
        for n in (3, 4, 6, 9):
            c = _pa_random(n, "float32", seed=1600 + n)
            emit(o_roots, A(c), lambda: roots(c), "roots_kind", f"{o_roots}/f32mix/{n}")

        # ---------------- (H) result flags ----------------
        for dt in ("float64", "float32", "float16", "complex128", "int64"):
            for n in (1, 2, 3, 5):
                c = _ps_series(n, dt, seed=24)
                both(ops, A(c), "flags", f"flags/{dt}/{n}", facet="flags")
        for tag, rts in (("real", [1.0, -2.0, 0.5]), ("cplx", [1 + 1j, 1 - 1j, 2 + 0j])):
            for dt in ("float64", "float32"):
                with np.errstate(all="ignore"):
                    c = np.real_if_close(np.asarray(fromroots(np.array(rts)))).astype(dt)
                emit(o_roots, A(c), lambda: roots(c), "flags", f"{o_roots}/flags/{tag}/{dt}", facet="flags")

        # ---------------- (I) long series ----------------
        for dt in ("float64", "complex128", "float32", "float16"):
            for n in (40, 65):
                c = _pa_random(n, dt, seed=1800 + n, scale=0.5)
                emit(o_comp, A(c), lambda: comp(c), "long_1d", f"{o_comp}/long/{dt}/{n}")
            if dt != "float16":
                for n in (25, 51):
                    c = _pa_random(n, dt, seed=1850 + n, scale=0.5)
                    emit(o_roots, A(c), lambda: roots(c), "long_1d", f"{o_roots}/long/{dt}/{n}")

        # ---------------- (J) the linear roots' scalarmath ----------------
        sp = [1.0, -0.0, 0.0, np.inf, -np.inf, np.nan, 5e-324, 1e308, -2.5]
        for dt in ("float64", "float32", "float16"):
            for a0 in sp:
                for a1 in sp:
                    with np.errstate(all="ignore"):   # 1e308 / 5e-324 overflow / underflow the narrow dtypes
                        c = np.array([a0, a1], dtype=dt)
                    both(ops, A(c), "linear", f"lin/{dt}/{a0!r}/{a1!r}")
        csp = [complex(1, 2), complex(-0.0, 0.0), complex(np.inf, 1), complex(1, np.nan), complex(0, 1e308), complex(3, 0)]
        for a0 in csp:
            for a1 in csp:
                c = np.array([a0, a1], dtype=np.complex128)
                both(ops, A(c), "linear", f"lin/c/{a0!r}/{a1!r}")

    # Char: NumSharp's uint16-like dtype converts to float64 exactly as uint16 does (the house weave) — in both tiers,
    # since a uint16 series' roots reach LAPACK like any other integer series'.
    cases += _relabel_dtype([c for c in cases if "/uint16/" in (c.get("id") or "") and not c.get("expects_throw")],
                            "uint16", "char")
    host += _relabel_dtype([c for c in host if "/uint16/" in (c.get("id") or "") and not c.get("expects_throw")],
                           "uint16", "char")
    print(f"  portable {len(cases)}, host-pinned {len(host)}  (skipped: {skipped['dtype']} object-dtype results, "
          f"{skipped['order']} value-tied orders, {skipped['object']} object-arithmetic outcomes)")
    return cases, host


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.normpath(os.path.join(here, "..", "NumSharp.Tests.Oracle", "Fuzz", "corpus"))
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if mode == "conversion":
        cases = gen_conversion(ALL_DTYPES)
        cases += _relabel_dtype(gen_conversion(["uint16"]), "uint16", "char")
        write_jsonl(os.path.join(corpus_dir, "conversion.jsonl"), cases)
    elif mode == "creation":
        cases = gen_creation(ALL_DTYPES)
        cases += _relabel_dtype(gen_creation(["uint16"]), "uint16", "char")
        write_jsonl(os.path.join(corpus_dir, "creation.jsonl"), cases)
    elif mode == "multioutput":
        write_jsonl(os.path.join(corpus_dir, "multioutput.jsonl"), gen_multioutput())
    elif mode == "iter":
        # Iterator traces — see gen_iter. Order/layout resolution has no other gate.
        write_jsonl(os.path.join(corpus_dir, "iter.jsonl"), gen_iter())
    elif mode == "dtype_text":
        write_jsonl(os.path.join(corpus_dir, "dtype_text.jsonl"), gen_dtype_text())
    elif mode == "out_where":
        write_jsonl(os.path.join(corpus_dir, "out_where.jsonl"), gen_out_where())
    elif mode == "errors_full":
        write_jsonl(os.path.join(corpus_dir, "errors_full.jsonl"), gen_errors_full())
    elif mode == "smoke":
        srcs = ["float64", "int32", "float32"]
        dsts = ["int32", "float64", "uint8", "int16"]
        layouts = list(LAYOUTS.keys())
        cases = gen_astype(srcs, dsts, layouts)
        write_jsonl(os.path.join(corpus_dir, "astype_smoke.jsonl"), cases)
    elif mode == "astype_full":
        cases = gen_astype(ALL_DTYPES, ALL_DTYPES, list(LAYOUTS.keys()))
        cases += char_tier("astype_full")
        write_jsonl(os.path.join(corpus_dir, "astype_full.jsonl"), cases)
    elif mode == "binary":
        cases = gen_binary(BINARY_OPS, DT_PAIRS, list(PAIR_LAYOUTS.keys()))
        cases += char_tier("binary")
        write_jsonl(os.path.join(corpus_dir, "binary_arith.jsonl"), cases)
    elif mode == "divmod_power":
        cases = gen_binary(DIVMOD_POWER_OPS, DT_PAIRS, list(PAIR_LAYOUTS.keys()))
        cases += char_tier("divmod_power")
        write_jsonl(os.path.join(corpus_dir, "binary_divmod_power.jsonl"), cases)
    elif mode == "comparison":
        cases = gen_binary(COMPARISON_OPS, DT_PAIRS, list(PAIR_LAYOUTS.keys()))
        cases += char_tier("comparison")
        write_jsonl(os.path.join(corpus_dir, "comparison.jsonl"), cases)
    elif mode == "unary":
        cases = gen_unary(UNARY_OPS, UNARY_DTYPES, list(LAYOUTS.keys()))
        cases += char_tier("unary")
        write_jsonl(os.path.join(corpus_dir, "unary.jsonl"), cases)
    elif mode == "reduce":
        cases = gen_reduce(REDUCE_OPS, REDUCE_DTYPES, REDUCE_LAYOUTS)
        cases += char_tier("reduce")
        write_jsonl(os.path.join(corpus_dir, "reduce.jsonl"), cases)
    elif mode == "where":
        cases = gen_where(WHERE_DT_PAIRS, list(WHERE_LAYOUTS.keys()))
        cases += gen_where_cond(WHERE_COND_DTYPES, WHERE_COND_XY_PAIRS)   # G4: non-bool cond
        cases += char_tier("where")                                        # G9
        write_jsonl(os.path.join(corpus_dir, "where.jsonl"), cases)
    elif mode == "place":
        cases = gen_place(PLACE_DTYPES, PLACE_LAYOUTS)
        write_jsonl(os.path.join(corpus_dir, "place.jsonl"), cases)
    elif mode == "putmask":
        cases = gen_putmask(PUTMASK_DTYPES, PUTMASK_LAYOUTS)
        write_jsonl(os.path.join(corpus_dir, "putmask.jsonl"), cases)
    elif mode == "matmul":
        cases = gen_matmul(MATMUL_SHAPE_CASES, MATMUL_DTYPES, MATMUL_LAYOUTS)
        cases += gen_matmul_edges(MATMUL_EDGE_DTYPES)                  # G14: negstride + k=0
        cases += gen_matmul_zerodim(MATMUL_EDGE_DTYPES)                # G15: stacked zero extents
        cases += gen_matmul_half_depth()                               # G16: float16 contraction depth
        cases += gen_trace_diag(TRACE_DTYPES)                          # Group A: trace/diagonal
        cases += gen_diag_tri(TRACE_DTYPES)                            # diag/tri family
        cases += char_tier("matmul")                                   # G9
        write_jsonl(os.path.join(corpus_dir, "matmul.jsonl"), cases)
    elif mode == "rounding":
        cases = gen_round(ROUND_DTYPES, list(LAYOUTS.keys()))          # Group A: round_/around
        cases += char_tier("rounding")                                 # G9
        write_jsonl(os.path.join(corpus_dir, "rounding.jsonl"), cases)
    elif mode == "bitwise":
        cases = gen_binary(BITWISE_BIN_OPS, BITWISE_DT_PAIRS, list(PAIR_LAYOUTS.keys()))
        cases += gen_unary(INVERT_OP, INT_BOOL_DTYPES, list(LAYOUTS.keys()))
        cases += gen_unary(BITWISE_COUNT_OP, INT_BOOL_DTYPES, list(LAYOUTS.keys()))
        cases += gen_shift(SHIFT_OPS, SHIFT_DTYPES)
        cases += char_tier("bitwise")
        write_jsonl(os.path.join(corpus_dir, "bitwise.jsonl"), cases)
    elif mode == "gcd":
        # np.gcd / np.lcm — integer-only binary ufuncs across every valid integer dtype pair × layout.
        # Invalid pairs (float/complex/bool+bool, uint64+signed -> float64) are skipped by gen_binary;
        # their no-loop errors are gated in errors_full. Char woven via char_tier.
        cases = gen_binary(GCDLCM_OPS, GCDLCM_DT_PAIRS, list(PAIR_LAYOUTS.keys()))
        cases += char_tier("gcd")
        write_jsonl(os.path.join(corpus_dir, "gcd.jsonl"), cases)
    elif mode == "unary_extra":
        cases = gen_unary(UNARY_EXTRA_OPS, ALL_DTYPES, list(LAYOUTS.keys()))
        cases += char_tier("unary_extra")
        write_jsonl(os.path.join(corpus_dir, "unary_extra.jsonl"), cases)
    elif mode == "sinc":
        # sinc over every REAL dtype × all layouts (complex128 excluded — see SINC_DTYPES).
        cases = gen_unary(SINC_OP, SINC_DTYPES, list(LAYOUTS.keys()))
        cases += char_tier("sinc")
        write_jsonl(os.path.join(corpus_dir, "sinc.jsonl"), cases)
    elif mode == "i0":
        # i0 over every REAL dtype × all layouts (complex128 excluded — NumPy rejects it).
        cases = gen_unary(I0_OP, I0_DTYPES, list(LAYOUTS.keys()))
        cases += char_tier("i0")
        write_jsonl(os.path.join(corpus_dir, "i0.jsonl"), cases)
    elif mode == "unwrap":
        # np.unwrap over the scan layouts x every dtype x the period/discont/axis sweep.
        # Char rides the uint16 proxy (float-period cases; its integer-period cases OverflowError
        # like every unsigned and are skipped in-generator).
        cases = gen_unwrap(SCAN_DTYPES, SCAN_LAYOUTS)
        cases += char_tier("unwrap")
        write_jsonl(os.path.join(corpus_dir, "unwrap.jsonl"), cases)
    elif mode == "nanreduce":
        cases = gen_reduce(NAN_REDUCE_OPS, NAN_REDUCE_DTYPES, REDUCE_LAYOUTS)
        cases += gen_nanquantile(NANQ_DTYPES)                           # Group A: nanpercentile/nanquantile
        write_jsonl(os.path.join(corpus_dir, "nanreduce.jsonl"), cases)
    elif mode == "scan":
        cases = gen_scan(SCAN_OPS, SCAN_DTYPES, SCAN_LAYOUTS)
        cases += gen_diff(SCAN_DTYPES, SCAN_LAYOUTS)
        cases += gen_ediff1d(EDIFF_DTYPES, list(LAYOUTS.keys()))        # Group A: ediff1d
        cases += char_tier("scan")
        write_jsonl(os.path.join(corpus_dir, "scan.jsonl"), cases)
    elif mode == "nanscan":
        # nancumsum is complex ADD (portable, bit-exact); nancumprod is complex MULTIPLY, whose
        # win-amd64 NumPy result is MSVC-FMA-contracted (numpy computes in1r*in2i + in1i*in2r and MSVC
        # fuses one product), ~1-2 ULP off .NET's non-fused System.Numerics.Complex operator* — the same
        # host-FMA class as the GEMM/float-kernel pins, and genuinely non-portable-bit-exact for a complex
        # product without BLAS. complex128 is therefore CARVED from nancumprod (plain cumprod hides this by
        # short-circuiting the NaN-laced pool to NaN; nancumprod replaces NaN->1 and exposes the real
        # product chain) and covered by the NanCumProd_Complex unit test on exact-representable values.
        # complex128 nancumsum and every other dtype for both ops stay bit-exact.
        cases = gen_scan(NAN_SCAN_OPS, [d for d in SCAN_DTYPES if d != "complex128"], SCAN_LAYOUTS)
        cases += gen_scan({"nancumsum": NAN_SCAN_OPS["nancumsum"]}, ["complex128"], SCAN_LAYOUTS)
        write_jsonl(os.path.join(corpus_dir, "nanscan.jsonl"), cases)
    elif mode == "stat":
        cases = gen_reduce(STAT_REDUCE_OPS, STAT_DTYPES, STAT_LAYOUTS)
        cases += gen_count_nonzero(CNZ_DTYPES, STAT_LAYOUTS)
        cases += gen_quantile(QUANTILE_SPECS, STAT_DTYPES, STAT_LAYOUTS)
        cases += gen_clip(CLIP_DTYPES, STAT_LAYOUTS)
        cases += gen_interp()                                  # np.interp (1-D linear interpolation)
        cases += char_tier("stat")
        write_jsonl(os.path.join(corpus_dir, "stat.jsonl"), cases)
    elif mode == "logic":
        cases = gen_unary(LOGIC_UNARY_OPS, LOGIC_UNARY_DTYPES, list(LAYOUTS.keys()))
        cases += gen_binary(LOGIC_BIN_OPS, LOGIC_BIN_PAIRS, list(PAIR_LAYOUTS.keys()))
        cases += gen_binary(LOGICAL_BIN_OPS, LOGICAL_PAIRS, list(PAIR_LAYOUTS.keys()))   # Group A B1
        cases += gen_unary(LOGICAL_NOT_OP, ALL_DTYPES, list(LAYOUTS.keys()))             # Group A B1
        cases += gen_binary(ARCTAN2_OP, ARCTAN2_PAIRS, list(PAIR_LAYOUTS.keys()))        # Group A B1
        cases += gen_binary(LOGADDEXP_OPS, ARCTAN2_PAIRS, list(PAIR_LAYOUTS.keys()))      # logaddexp/logaddexp2
        cases += gen_binary(NEXTAFTER_OP, ARCTAN2_PAIRS, list(PAIR_LAYOUTS.keys()))       # nextafter (bit-exact)
        cases += gen_binary(COPYSIGN_OP, ARCTAN2_PAIRS, list(PAIR_LAYOUTS.keys()))        # copysign (bit-exact)
        cases += gen_binary(HYPOT_OP, ARCTAN2_PAIRS, list(PAIR_LAYOUTS.keys()))           # hypot (f32/f16 exact, f64 <=1 ULP)
        cases += gen_binary(HEAVISIDE_OP, ARCTAN2_PAIRS, list(PAIR_LAYOUTS.keys()))       # heaviside (bit-exact, all dtypes)
        cases += gen_binary(ALLCLOSE_OPS, ALLCLOSE_PAIRS, list(PAIR_LAYOUTS.keys()))     # Group A B3
        cases += gen_unary(ISCOMPLEX_OPS, ISCOMPLEX_DTYPES, list(LAYOUTS.keys()))         # G5 (full)
        cases += char_tier("logic")                                                       # G9
        write_jsonl(os.path.join(corpus_dir, "logic.jsonl"), cases)
    elif mode == "modf":
        cases = gen_modf(MODF_DTYPES, MODF_LAYOUTS)
        cases += char_tier("modf")
        write_jsonl(os.path.join(corpus_dir, "modf.jsonl"), cases)
    elif mode == "manip":
        cases = gen_manip(MANIP_DTYPES, list(LAYOUTS.keys()))
        cases += gen_concat_stack(MANIP_DTYPES)
        cases += gen_pad(MANIP_DTYPES)
        cases += gen_index_tricks(MANIP_DTYPES)        # r_ / c_ / ix_ index-expression DSL
        cases += char_tier("manip")
        write_jsonl(os.path.join(corpus_dir, "manip.jsonl"), cases)
    elif mode == "sort":
        cases = gen_argsort(SORT_DTYPES)
        cases += gen_sort(SORT_DTYPES)                                  # Group A B2: value sort
        cases += gen_searchsorted(SORT_DTYPES)
        cases += gen_digitize(DIGITIZE_DTYPES)                          # digitize = searchsorted + monotonicity
        cases += gen_nonzero(SORT_DTYPES)
        cases += gen_bincount(BINCOUNT_DTYPES)                          # counting: bincount (+ weights/minlength)
        cases += gen_unary(NZ_OPS, NZ_DTYPES, list(LAYOUTS.keys()))     # Group A B3: flatnonzero/argwhere
        cases += gen_unique(["bool", "int32", "uint8", "int64", "float64", "float32", "complex128"])  # B3: unique
        cases += gen_sort_special()                                     # G11: NaN + strided/negstride
        cases += gen_partition_family(SORT_DTYPES)                      # G12: issue #623 kth-values compare
        cases += gen_partition_nan()                                    # G12: float/complex NaN partition
        cases += gen_lexsort(SORT_DTYPES)                               # G12: stable multi-key indices
        cases += gen_sort_complex(SORT_COMPLEX_DTYPES)                  # G12: sort in own dtype -> Complex
        cases += char_tier("sort")
        write_jsonl(os.path.join(corpus_dir, "sort.jsonl"), cases)
    elif mode == "tail":
        cases = gen_tail(TAIL_DTYPES)
        cases += char_tier("tail")
        write_jsonl(os.path.join(corpus_dir, "tail.jsonl"), cases)
    elif mode == "params":
        cases = gen_params(PARAM_DTYPES)
        write_jsonl(os.path.join(corpus_dir, "params.jsonl"), cases)
    elif mode == "aliasing":
        cases = gen_aliasing(ALIAS_DTYPES)
        write_jsonl(os.path.join(corpus_dir, "aliasing.jsonl"), cases)
    elif mode == "copyto":
        cases = gen_copyto(COPYTO_OVERLAP_DTYPES, COPYTO_CROSS)
        cases += char_tier("copyto")                                   # G9
        write_jsonl(os.path.join(corpus_dir, "copyto.jsonl"), cases)
    elif mode == "errors":
        cases = gen_errors()
        write_jsonl(os.path.join(corpus_dir, "errors.jsonl"), cases)
    elif mode == "groupa":
        cases = gen_groupa()                                            # Group A B4-6
        write_jsonl(os.path.join(corpus_dir, "groupa.jsonl"), cases)
    elif mode == "numpy_f32":
        cases = gen_numpy_f32_kernels()                                 # bit-exact float32 kernel tier
        write_jsonl(os.path.join(corpus_dir, "numpy_f32_kernels.jsonl"), cases)
        cases = gen_numpy_f64_kernels()                                 # bit-exact float64 kernel tier (tanh)
        write_jsonl(os.path.join(corpus_dir, "numpy_f64_kernels.jsonl"), cases)
    elif mode == "matmul_parity":
        cases = gen_matmul_parity()                                     # np.parity_matmul byte gate
        write_jsonl(os.path.join(corpus_dir, "matmul_parity.jsonl"), cases)
        # The host pin travels with the corpus: these bytes are only reproducible on a
        # host whose BLAS binary + dispatched kernel + thread count match. The C# gate
        # reports Inconclusive (never red) when they do not.
        write_jsonl(os.path.join(corpus_dir, "matmul_parity.host.jsonl"), [blas_identity()])
    elif mode == "linalg_parity":
        cases = gen_linalg_parity()                                     # LAPACK factorisation byte gate
        write_jsonl(os.path.join(corpus_dir, "linalg_parity.jsonl"), cases)
        # Host-pinned like matmul_parity, but at threads=1 (the deterministic config the
        # interop live-parity suite proves) — gen_linalg_parity() forces single-thread, so
        # blas_identity() records blas_threads=1 and the C# gate enables the backend at 1.
        write_jsonl(os.path.join(corpus_dir, "linalg_parity.host.jsonl"), [blas_identity()])
    elif mode == "poly":
        cases = gen_poly()                                             # portable polynomial family
        write_jsonl(os.path.join(corpus_dir, "poly.jsonl"), cases)
    elif mode == "einsum":
        cases = gen_einsum()                                           # np.einsum + einsum_path
        write_jsonl(os.path.join(corpus_dir, "einsum.jsonl"), cases)
    elif mode == "specials":
        cases = gen_specials()                                          # IEEE special-value parity tier
        write_jsonl(os.path.join(corpus_dir, "specials.jsonl"), cases)
    elif mode == "precision":
        cases = gen_precision()                                         # truthful-vs-precise tier (needs mpmath)
        write_jsonl(os.path.join(corpus_dir, "precision.jsonl"), cases)
    elif mode == "products":
        cases = gen_products()                                          # CBLAS product family values
        write_jsonl(os.path.join(corpus_dir, "products.jsonl"), cases)
    elif mode == "random_parity":
        portable, host = gen_random_parity()                            # seeded MT19937 stream bytes
        write_jsonl(os.path.join(corpus_dir, "random_parity.jsonl"), portable)
        write_jsonl(os.path.join(corpus_dir, "random_parity_host.jsonl"), host)
    elif mode == "generator_parity":
        portable, host = gen_generator_parity()                         # seeded PCG64 Generator stream bytes
        write_jsonl(os.path.join(corpus_dir, "generator_parity.jsonl"), portable)
        write_jsonl(os.path.join(corpus_dir, "generator_parity_host.jsonl"), host)
    elif mode == "fft":
        cases = gen_fft()                                               # np.fft.* differential tier
        write_jsonl(os.path.join(corpus_dir, "fft.jsonl"), cases)
    elif mode == "windows":
        cases = gen_windows()                                           # bartlett/blackman/hamming/hanning/kaiser
        write_jsonl(os.path.join(corpus_dir, "windows.jsonl"), cases)
    elif mode == "real_if_close":
        cases = gen_real_if_close()                                     # complex -> real collapse decision
        write_jsonl(os.path.join(corpus_dir, "real_if_close.jsonl"), cases)
    elif mode == "evaluate":
        cases = gen_evaluate()                                          # np.evaluate / NDExpr fused trees
        write_jsonl(os.path.join(corpus_dir, "evaluate.jsonl"), cases)
    elif mode == "instance":
        cases = gen_instance()                                          # ndarray.* instance surface (plan §D)
        cases += char_tier("instance")
        write_jsonl(os.path.join(corpus_dir, "instance.jsonl"), cases)
    elif mode == "emath":
        cases = gen_emath()                                             # np.emath scimath module (plan §A2/E5)
        write_jsonl(os.path.join(corpus_dir, "emath.jsonl"), cases)
    elif mode == "polyeval":
        cases = gen_polyeval()                                          # numpy.polynomial {p}val family (U3)
        write_jsonl(os.path.join(corpus_dir, "polyeval.jsonl"), cases)
    elif mode == "polyseries":
        cases = gen_polyseries()                                        # numpy.polynomial additive family + polyutils (U1)
        write_jsonl(os.path.join(corpus_dir, "polyseries.jsonl"), cases)
    elif mode == "polycalc":
        cases = gen_polycalc()                                          # numpy.polynomial calculus: {p}der / {p}int (U4)
        write_jsonl(os.path.join(corpus_dir, "polycalc.jsonl"), cases)
    elif mode == "polyvander":
        cases = gen_polyvander()                                        # numpy.polynomial Vandermonde family (U5)
        write_jsonl(os.path.join(corpus_dir, "polyvander.jsonl"), cases)
    elif mode == "polyalgebra":
        # The BLAS-bound products (np.convolve's vector-kernel regime) are host-pinned like linalg_parity: the bytes
        # come out of numpy's own scipy-openblas at threads=1 (forced first), so the pin travels with them.
        _set_openblas_threads(1)
        cases, host = gen_polyalgebra()                                 # numpy.polynomial series algebra (U2)
        write_jsonl(os.path.join(corpus_dir, "polyalgebra.jsonl"), cases)
        write_jsonl(os.path.join(corpus_dir, "polyalgebra_parity.jsonl"), host)
        write_jsonl(os.path.join(corpus_dir, "polyalgebra_parity.host.jsonl"), [blas_identity()])
    elif mode == "polyroots":
        # numpy.polynomial companion matrices and roots (U7): portable + host-pinned (LAPACK geev at threads = 1).
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            if os.environ.get(var) != "1":
                sys.exit(f"polyroots needs {var}=1 set before Python starts (eigvals' bits depend on the thread count)")
        cases, host = gen_polyroots()
        write_jsonl(os.path.join(corpus_dir, "polyroots.jsonl"), cases)
        write_jsonl(os.path.join(corpus_dir, "polyroots_parity.jsonl"), host)
        write_jsonl(os.path.join(corpus_dir, "polyroots_parity.host.jsonl"), [blas_identity()])
    else:
        print(f"unknown mode '{mode}' (expected: conversion | creation | multioutput | smoke | astype_full | binary | divmod_power | comparison | unary | reduce | where | place | putmask | matmul | rounding | bitwise | unary_extra | sinc | i0 | nanreduce | scan | nanscan | stat | logic | modf | manip | sort | tail | params | aliasing | copyto | errors | groupa | numpy_f32 | matmul_parity | linalg_parity | poly | einsum | specials | precision | random_parity | generator_parity | products | fft | windows | evaluate | real_if_close | instance | emath | polyeval | polyseries | polycalc | polyvander | polyalgebra | polyroots)")
        sys.exit(2)


if __name__ == "__main__":
    main()
