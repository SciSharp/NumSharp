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
import json
import os
import sys
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
    "power": lambda a, b: a ** b,
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
    "negative": np.negative, "abs": np.abs, "sign": np.sign,
    "sqrt": np.sqrt, "cbrt": np.cbrt, "square": np.square, "reciprocal": np.reciprocal,
    "floor": np.floor, "ceil": np.ceil, "trunc": np.trunc,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
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
    "deg2rad": np.deg2rad, "rad2deg": np.rad2deg,
    "positive": np.positive,
    "rint": np.rint,   # round-half-to-even; float-tier dtype like the others in this group
}


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
# [1+2j, 2+0j, 0+0j]); included below. bool is CARVED OUT: NumSharp's general
# (strided/transposed/F-contig) clip kernel throws "clip not supported for
# Boolean" (only the contiguous path handles bool) — reproduced under [OpenBugs] (Clip_Bool_Strided_*).
CLIP_DTYPES = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
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
            lo_v, hi_v = (1, 100) if dt.kind == "u" else (-10, 10)
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
LOGIC_UNARY_OPS = {"isnan": np.isnan, "isinf": np.isinf, "isfinite": np.isfinite}
LOGIC_UNARY_DTYPES = list(ALL_DTYPES)  # widened: isnan/isinf/isfinite defined on every dtype
LOGIC_BIN_OPS = {
    "maximum": np.maximum, "minimum": np.minimum,
    "fmax": np.fmax, "fmin": np.fmin, "isclose": np.isclose,
}
LOGIC_BIN_PAIRS = [
    ("float32", "float32"), ("float64", "float64"), ("float16", "float16"),
    ("int32", "int32"), ("int32", "float64"), ("uint8", "int8"), ("int32", "int64"),
    ("complex128", "complex128"),
]

# G5 (F5) — iscomplex/isreal: REAL dtypes × CONTIGUOUS layouts ONLY (this is the verified-green
# envelope). CARVED (both documented bugs already pinned in OpenBugs.DtypeCoverage.cs —
# IsComplex_IgnoresImaginaryPart / IsReal_IgnoresImaginaryPart, ≈:119/:130):
#   * complex128 input — NumSharp never inspects the imaginary part (iscomplex -> all False,
#     isreal -> all True, both wrong for nonzero-imag values);
#   * strided/F-contiguous/transposed REAL input — same op emits garbage bytes on the
#     non-contiguous path.
ISCOMPLEX_OPS = {"iscomplex": np.iscomplex, "isreal": np.isreal}
ISCOMPLEX_DTYPES = [d for d in ALL_DTYPES if d != "complex128"]
ISCOMPLEX_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "c_contiguous_3d",
                     "one_element_1d", "scalar_0d"]

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
INT_BOOL_DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]
BITWISE_DT_PAIRS = [
    ("int32", "int32"), ("uint8", "uint8"), ("int8", "int8"), ("int16", "int16"),
    ("uint16", "uint16"), ("uint32", "uint32"), ("int64", "int64"), ("uint64", "uint64"),
    ("bool", "bool"), ("int32", "int64"), ("uint8", "int8"), ("int32", "uint32"),
    ("bool", "int32"), ("int8", "int16"), ("uint16", "uint32"), ("int64", "uint64"),
]

SHIFT_OPS = {"left_shift": np.left_shift, "right_shift": np.right_shift}
SHIFT_DTYPES = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]


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
            ]
            if sz > 0:
                jobs.append(("reshape", {"shape": [sz]}, lambda v, sz=sz: v.reshape(sz)))
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
MODF_DTYPES = ["float16", "float32", "float64", "int32"]
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
    cases = []
    n = 0
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
# bool `-`) and complex128 (the a*a self-multiply of a large _cbase value hits catastrophic
# cancellation in a^2-b^2, where NumPy's ARRAY ufunc and the naive ac-bd formula round differently;
# NumSharp matches NumPy's SCALAR multiply exactly, so this is a ULP/ill-conditioned artifact, NOT a bug).
ALIAS_DTYPES = [d for d in ALL_DTYPES if d not in ("bool", "complex128")]


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
        if isinstance(exp, dict) and exp.get("dtype") == frm:
            exp["dtype"] = to
        for k, v in list(c.get("params", {}).items()):
            if v == frm:
                c["params"][k] = to
        c["id"] = c["id"].replace(frm, to)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Group A Batch 2 generators: sort / round_ / trace / diagonal / ediff1d / nan-quantile.
# ---------------------------------------------------------------------------
# bool is CARVED OUT: np.round(bool, 0) -> float16 [0,1] in NumPy (rint float-tier), while
# NumSharp's round_ resolves bool -> Double — dtype divergence pinned under [OpenBugs]
# (OpenBugs.FuzzGaps.cs: Round_Bool_Dtype_Diverges). (bool with decimals!=0 raises in NumPy.)
# complex128 is included at decimals=0 ONLY: NumSharp's round_ with decimals!=0 is a NO-OP
# identity for Complex (NumPy rounds re+im via multiply->rint->divide: round(1.55+2.45j, 1)
# -> 1.6+2.4j) — pinned under [OpenBugs] (OpenBugs.FuzzGaps.cs: Round_Complex_NonzeroDecimals_NoOp);
# the dec!=0 complex carve lives in gen_round.
ROUND_DTYPES = ["int8", "uint8", "int16", "int32", "int64", "uint16", "uint32", "uint64",
                "float16", "float32", "float64", "complex128"]
# uint8 CARVED: trace of an unsigned dtype upcasts to Int64 in NumSharp but uint64 in NumPy -> [OpenBugs].
TRACE_DTYPES = ["int16", "int32", "int64", "float16", "float32", "float64", "complex128"]
EDIFF_DTYPES = ["int16", "int32", "int64", "uint8", "float32", "float64", "complex128"]  # no bool (NumPy bans bool `-`)
NANQ_DTYPES = ["float16", "float32", "float64"]  # NaN only exists in float; pools already carry NaN/inf

# Group A Batch 3: searching (flatnonzero/argwhere -> int64 coords) + whole-array bool reductions
# (allclose/array_equal, wrapped to a 0-D bool via np.asarray). All GREEN.
# CARVED (-> [OpenBugs]): iscomplex/isreal (NumSharp ignores the imaginary part for complex input and
# emits garbage bytes on strided real input) and unique (mishandles offset/strided views + NaN-complex
# ordering). flatnonzero/argwhere stay.
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
    CARVE-OUTS (-> [OpenBugs]): dec=-1 (NumSharp's Math.Round rejects negative digits for ints and
    mis-rounds floats), float16 with dec>=1 (float16 fractional rounding diverges), and
    complex128 with dec>=1 (NumSharp round_ is a no-op identity for Complex when decimals!=0;
    OpenBugs.FuzzGaps.cs: Round_Complex_NonzeroDecimals_NoOp)."""
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            for dec in (0, 1, 2):                         # dec=-1 carved (negative-decimals bug)
                if s == "float16" and dec != 0:           # float16 fractional rounding carved
                    continue
                if s == "complex128" and dec != 0:        # complex dec!=0 carved (NumSharp no-op bug)
                    continue
                try:
                    r = np.asarray(np.round(view, dec))
                except Exception:
                    skipped += 1
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
        base1 = np.array([3.5, -2.0, np.nan, 7.25, 0.0, -9.5, 4.0, np.nan, 1.5, 6.0, -3.0, 2.5], dtype=dt)
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

        # append — flatten form (axis=None) + along axis 0.
        vals1 = _cbase((4,), d)
        emit("append", {}, [describe(a2, a2), describe(vals1, vals1)], np.append(a2, vals1))
        row = _cbase((1, 4), d)
        emit("append", {"axis": 0}, [describe(a2, a2), describe(row, row)], np.append(a2, row, 0))

        # insert — insert a row at obj=1 along axis 0.
        insvals = _cbase((4,), d)
        emit("insert", {"obj": 1, "axis": 0}, [describe(a2, a2), describe(insvals, insvals)], np.insert(a2, 1, insvals, 0))

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

    # ravel_multi_index / unravel_index — index<->coord transforms (int64, dtype-independent).
    row = np.array([0, 1, 2, 0], dtype=np.int64)
    col = np.array([1, 3, 0, 2], dtype=np.int64)
    emit("ravel_multi_index", {"dims": [3, 4]}, [describe(row, row), describe(col, col)],
         np.ravel_multi_index((row, col), (3, 4)))
    flat = np.array([0, 5, 11, 7], dtype=np.int64)
    for pi, part in enumerate(np.unravel_index(flat, (3, 4))):
        emit("unravel_index", {"shape": [3, 4], "piece": pi}, [describe(flat, flat)], part)

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

# Power crashes on any char operand; reciprocal mis-types char -> excluded per-op.
_CHAR_DIVMOD_OPS = {k: v for k, v in DIVMOD_POWER_OPS.items() if k != "power"}
_CHAR_UNARY_OPS  = {k: v for k, v in UNARY_OPS.items() if k != "reciprocal"}

# G9 (F8) — pairs/op-sets for the additionally woven modes. The uint16 slot IS the Char;
# uint8/bool partners stay carved (char-promote bug), power/reciprocal/invert stay carved.
CHAR_WHERE_PAIRS = [(_C, _C), (_C, "int32"), ("float64", _C)]   # cond stays bool
CHAR_EXTREMA_OPS = {"maximum": np.maximum, "minimum": np.minimum, "fmax": np.fmax, "fmin": np.fmin}
CHAR_LOGIC_UNARY = {"isnan": np.isnan, "isinf": np.isinf, "isfinite": np.isfinite,
                    "logical_not": np.logical_not}
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
    elif mode == "bitwise":
        raw = gen_binary(BITWISE_BIN_OPS, CHAR_BIT_PAIRS, PL)
        raw += gen_shift(SHIFT_OPS, [_C])                          # invert(char) carved (SIMD gap)
    elif mode == "reduce":
        raw = gen_reduce(REDUCE_OPS, [_C], REDUCE_LAYOUTS)
    elif mode == "scan":
        raw = gen_scan(SCAN_OPS, [_C], SCAN_LAYOUTS) + gen_diff([_C], SCAN_LAYOUTS)
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
MATMUL_PARITY_DTYPES = ["float32", "float64"]

# k values: 1..4 (agreeing region), the powers of two and their +-1 neighbours
# (OpenBLAS panel edges), NumSharp's own KC=256 boundary, and the MLP's 784.
MATMUL_PARITY_KS = [1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                    127, 128, 129, 255, 256, 257, 511, 512, 784]


def _mp_values(shape, dt, rng, valueclass="normal"):
    """Operand values. Random by default — regular ramps hide reassociation error."""
    n = int(np.prod(shape)) if shape else 1
    if valueclass == "wide":
        # Magnitudes spanning ~40 decades: summation order dominates the result.
        mant = rng.standard_normal(n)
        expo = rng.randint(-18, 18, n)
        a = (mant * (10.0 ** expo))
    elif valueclass == "specials":
        a = rng.standard_normal(n)
        if n >= 4:
            a[0] = np.inf
            a[1] = -np.inf
            a[2] = np.nan
            a[3] = 0.0
    else:
        a = rng.standard_normal(n)
    return np.ascontiguousarray(a.astype(np.dtype(dt)).reshape(shape))


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
                    cases.append(_case("nditer_values", {"order": order}, [operand],
                                       _arr_expected(vals), ln, "iter",
                                       cid=f"nditer_values/{ln}/{s}/{order}/{n}"))
                    n += 1
                except Exception as e:
                    cases.append(_error_case("nditer_values", {"order": order}, [operand], e, ln,
                                             cid=f"nditer_values/{ln}/{s}/{order}/{n}"))
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
    cases += gen_nditer_pair(DT_PAIRS[:12], list(PAIR_LAYOUTS.keys()), ["C", "K"])
    cases += gen_broadcast(DT_PAIRS[:12], list(PAIR_LAYOUTS.keys()))
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

    # --- scalar-returning predicates (wrapped 0-d, the np.allclose pattern) ---
    for frm in ALL_DTYPES:
        for to in ALL_DTYPES:
            for rule in CASTING_RULES:
                r = np.can_cast(np.dtype(frm), np.dtype(to), casting=rule)
                cases.append(_case("can_cast", {"from": frm, "to": to, "casting": rule}, [],
                                   _arr_expected(np.bool_(r), "scalar"), "generator", "predicate",
                                   cid=f"can_cast/{frm}->{to}/{rule}/{n}"))
                n += 1

    for ln in DTYPE_TEXT_LAYOUTS:
        fn = LAYOUTS[ln]
        for s in ITER_DTYPES:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            for opname, val in (("isscalar", np.isscalar(view)),
                                ("iscomplexobj", np.iscomplexobj(view)),
                                ("isrealobj", np.isrealobj(view))):
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
OUT_BINARY_UFUNCS = {
    "add": ["int32", "float64", "float32", "uint8"],
    "subtract": ["int32", "float64"],
    "multiply": ["int64", "float32"],
    "divide": ["float64", "int32"],
    "power": ["float64", "int32"],
    "mod": ["int32", "float64"],
    "floor_divide": ["int32", "float64"],
    "arctan2": ["float64", "float32"],
    "bitwise_and": ["int32", "uint8", "bool"],
    "bitwise_or": ["int64"],
    "bitwise_xor": ["uint16"],
    "less": ["int32", "float64"],
    "greater_equal": ["float32"],
    "equal": ["int32"],
}

OUT_UNARY_UFUNCS = {
    "sqrt": ["float64", "float32"],
    "negative": ["int32", "float64"],
    "abs": ["int32", "float64"],
    "square": ["float64", "int32"],
    "exp": ["float64", "float32"],
    "log": ["float64"],
    "sin": ["float64", "float32"],
    "floor": ["float64"],
    "ceil": ["float32"],
    "rint": ["float64"],
    "sign": ["int32", "float64"],
    "reciprocal": ["float64", "int32"],
    "invert": ["int32", "uint8"],
    "isnan": ["float64"],
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
    "deg2rad": np.deg2rad, "rad2deg": np.rad2deg, "isnan": np.isnan, "isinf": np.isinf, "isfinite": np.isfinite,
}
SPECIAL_BINARY_OPS = {
    "add": lambda a, b: a + b, "subtract": lambda a, b: a - b, "multiply": lambda a, b: a * b, "divide": lambda a, b: a / b,
    "floor_divide": lambda a, b: a // b, "mod": lambda a, b: a % b, "power": lambda a, b: a ** b, "arctan2": np.arctan2,
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
# host-dtype-pinned by NumPy itself: legacy RandomState uses C long, so this corpus
# (authored on win-amd64) records int32 — regenerating on Linux would record int64.
# NumSharp's int32 matches the Windows authoring host. Sequential-draw cases
# ("draws": 2) pin stream ADVANCEMENT, not just the first block.
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
    ("gamma", [2.0, 3.0]), ("gamma", [2.0, 1.0]),   # shape>=1 matches (scale path included);
                                                    # shape<1 via gamma() is CARVED, see below
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
    ("geometric", [0.35]),
    ("zipf", [3.0]),
    ("logseries", [0.6]),
    ("noncentral_chisquare", [3.0, 1.5]),
    ("noncentral_f", [5.0, 7.0, 1.5]),
]
# Stream-advancement pins: draw the same spec twice, record the SECOND block.
_RND_DRAWS2 = {"uniform", "randint", "normal", "standard_gamma", "poisson"}

# CARVED — the tier's findings on arrival (2026-08-14): eight samplers where NumSharp's
# STREAM diverges from NumPy's (a different algorithm / draw order / accept-reject
# boundary, not mere rounding), so the documented 1-to-1 claim does not hold for them
# today. Each is carved from the green corpus and pinned under [OpenBugs]
# (OpenBugs.Random.cs, RandomParity_* — remove the pin and re-add the spec when fixed):
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
# noise on an IDENTICAL stream, measured ≤5/≤24/≤3/≤3 ULP) stay IN the tier under the
# scoped R1 registry envelope (8 ULP; 32 for wald).
#
# int-output samplers: legacy RandomState returns C long — int32 on win-amd64, int64 on
# Linux — while NumSharp fixes int64 (the Linux-NumPy shape). The corpus records these
# WIDENED to int64 so the VALUE stream stays hard-gated and the dtype policy is
# documented here rather than silently failing per-host. Two exceptions, both matching
# win-amd64 NumPy unwidened: randint (NumSharp defaults int32) and PLAIN choice —
# NumSharp's choice returns int32 without `p` but int64 WITH `p` (an internal
# inconsistency worth its own fix; the with-p cases are widened so their values gate).
_RND_INT64_CAST = {"permutation", "poisson", "zipf", "logseries", "hypergeometric", "geometric"}


def gen_random_parity():
    portable = []
    host = []
    n = 0

    def emit(into, dist, params, r):
        nonlocal n
        r = np.asarray(r)
        if dist in _RND_INT64_CAST or (dist == "choice" and "p" in params):
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

    # --- host-libm: transform / rejection samplers ------------------------------------
    # standard_cauchy / multinomial / multivariate_normal are CARVED (see _RND_INT64_CAST
    # comment block) — gross stream divergence, pinned in OpenBugs.Random.cs.
    for dist, args in _RND_HOST_GENERIC:
        cases_for(host, dist, args)
    cases_for(host, "randn", [])
    cases_for(host, "standard_normal", [])
    cases_for(host, "standard_exponential", [])
    cases_for(host, "hypergeometric", [10, 7, 8])
    cases_for(host, "dirichlet", [], extra={"alpha": [2.0, 3.0, 5.0]})
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

    # --- F-layout spot checks (the stride-aware read path) ----------------------------
    for dt in ("int32", "float64"):
        A = _mm_fill((3, 4), dt); B = _mm_fill((2, 4), dt)
        baseA, viewA = _mm_layout(A, "F")
        try_emit("inner", {}, [(baseA, viewA), (np.ascontiguousarray(B), B)],
                 lambda: np.inner(viewA, B), "F", dt)

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

    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


def write_jsonl(path, cases):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases -> {path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.normpath(os.path.join(here, "..", "NumSharp.UnitTest", "Fuzz", "corpus"))
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if mode == "iter":
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
        cases += gen_shift(SHIFT_OPS, SHIFT_DTYPES)
        cases += char_tier("bitwise")
        write_jsonl(os.path.join(corpus_dir, "bitwise.jsonl"), cases)
    elif mode == "unary_extra":
        cases = gen_unary(UNARY_EXTRA_OPS, ALL_DTYPES, list(LAYOUTS.keys()))
        cases += char_tier("unary_extra")
        write_jsonl(os.path.join(corpus_dir, "unary_extra.jsonl"), cases)
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
    elif mode == "stat":
        cases = gen_reduce(STAT_REDUCE_OPS, STAT_DTYPES, STAT_LAYOUTS)
        cases += gen_count_nonzero(CNZ_DTYPES, STAT_LAYOUTS)
        cases += gen_quantile(QUANTILE_SPECS, STAT_DTYPES, STAT_LAYOUTS)
        cases += gen_clip(CLIP_DTYPES, STAT_LAYOUTS)
        cases += char_tier("stat")
        write_jsonl(os.path.join(corpus_dir, "stat.jsonl"), cases)
    elif mode == "logic":
        cases = gen_unary(LOGIC_UNARY_OPS, LOGIC_UNARY_DTYPES, list(LAYOUTS.keys()))
        cases += gen_binary(LOGIC_BIN_OPS, LOGIC_BIN_PAIRS, list(PAIR_LAYOUTS.keys()))
        cases += gen_binary(LOGICAL_BIN_OPS, LOGICAL_PAIRS, list(PAIR_LAYOUTS.keys()))   # Group A B1
        cases += gen_unary(LOGICAL_NOT_OP, ALL_DTYPES, list(LAYOUTS.keys()))             # Group A B1
        cases += gen_binary(ARCTAN2_OP, ARCTAN2_PAIRS, list(PAIR_LAYOUTS.keys()))        # Group A B1
        cases += gen_binary(ALLCLOSE_OPS, ALLCLOSE_PAIRS, list(PAIR_LAYOUTS.keys()))     # Group A B3
        cases += gen_unary(ISCOMPLEX_OPS, ISCOMPLEX_DTYPES, ISCOMPLEX_LAYOUTS)           # G5
        cases += char_tier("logic")                                                       # G9
        write_jsonl(os.path.join(corpus_dir, "logic.jsonl"), cases)
    elif mode == "modf":
        cases = gen_modf(MODF_DTYPES, MODF_LAYOUTS)
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
        cases += gen_nonzero(SORT_DTYPES)
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
    else:
        print(f"unknown mode '{mode}' (expected: smoke | astype_full | binary | divmod_power | comparison | unary | reduce | where | place | matmul | rounding | bitwise | unary_extra | nanreduce | scan | stat | logic | modf | manip | sort | tail | params | aliasing | copyto | errors | groupa | numpy_f32 | matmul_parity | specials | precision | random_parity | products)")
        sys.exit(2)


if __name__ == "__main__":
    main()
