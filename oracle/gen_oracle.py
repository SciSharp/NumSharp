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
                  "empty_2d", "one_element_1d"]


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
                    if opname in ("argmax", "argmin") and axis is None:
                        continue  # NumSharp has no flatten-argmax overload
                    for keepdims in (False, True):
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
}
NAN_REDUCE_DTYPES = ["float16", "float32", "float64", "int32", "complex128"]


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
STAT_DTYPES = ["int16", "int32", "int64", "uint8", "float16", "float32", "float64"]
STAT_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "c_contiguous_3d", "f_contiguous_2d",
                "transposed_3d", "strided_2d_cols", "one_element_1d"]
CNZ_DTYPES = ["bool", "int32", "uint8", "float64", "complex128"]
CLIP_DTYPES = ["int8", "uint8", "int16", "int32", "int64", "float16", "float32", "float64"]
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


# T11 — cumulative scans (cumsum/cumprod) and finite differences (diff). NumPy is the oracle for
# value, NEP50 accumulator dtype (cumsum(int32)->int64), and the diff output shape (shrinks by n).
SCAN_OPS = {
    "cumsum": lambda a, ax: np.cumsum(a, axis=ax),
    "cumprod": lambda a, ax: np.cumprod(a, axis=ax),
}
SCAN_DTYPES = ["int16", "int32", "int64", "uint8", "uint16", "float32", "float64", "complex128"]
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
LOGIC_UNARY_DTYPES = ["int32", "uint8", "float16", "float32", "float64", "complex128"]
LOGIC_BIN_OPS = {
    "maximum": np.maximum, "minimum": np.minimum,
    "fmax": np.fmax, "fmin": np.fmin, "isclose": np.isclose,
}
LOGIC_BIN_PAIRS = [
    ("float32", "float32"), ("float64", "float64"), ("float16", "float16"),
    ("int32", "int32"), ("int32", "float64"), ("uint8", "int8"), ("int32", "int64"),
    ("complex128", "complex128"),
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
MATMUL_DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                 "float16", "float32", "float64", "complex128"]

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
MANIP_DTYPES = ["int32", "float64", "uint8", "complex128"]


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
            ]
            if sz > 0:
                jobs.append(("reshape", {"shape": [sz]}, lambda v, sz=sz: v.reshape(sz)))
            if nd >= 2:
                jobs.append(("swapaxes", {"a1": 0, "a2": nd - 1}, lambda v, nd=nd: np.swapaxes(v, 0, nd - 1)))
                jobs.append(("moveaxis", {"src": 0, "dst": nd - 1}, lambda v, nd=nd: np.moveaxis(v, 0, nd - 1)))
                jobs.append(("delete", {"obj": 0, "axis": 0}, lambda v: np.delete(v, 0, axis=0)))
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
SORT_DTYPES = ["int32", "int64", "uint8", "float32", "float64"]


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
TAIL_DTYPES = ["int32", "int64", "uint8", "float32", "float64"]


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


def write_jsonl(path, cases):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases -> {path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.normpath(os.path.join(here, "..", "test", "NumSharp.UnitTest", "Fuzz", "corpus"))
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if mode == "smoke":
        srcs = ["float64", "int32", "float32"]
        dsts = ["int32", "float64", "uint8", "int16"]
        layouts = list(LAYOUTS.keys())
        cases = gen_astype(srcs, dsts, layouts)
        write_jsonl(os.path.join(corpus_dir, "astype_smoke.jsonl"), cases)
    elif mode == "astype_full":
        cases = gen_astype(ALL_DTYPES, ALL_DTYPES, list(LAYOUTS.keys()))
        write_jsonl(os.path.join(corpus_dir, "astype_full.jsonl"), cases)
    elif mode == "binary":
        cases = gen_binary(BINARY_OPS, DT_PAIRS, list(PAIR_LAYOUTS.keys()))
        write_jsonl(os.path.join(corpus_dir, "binary_arith.jsonl"), cases)
    elif mode == "divmod_power":
        cases = gen_binary(DIVMOD_POWER_OPS, DT_PAIRS, list(PAIR_LAYOUTS.keys()))
        write_jsonl(os.path.join(corpus_dir, "binary_divmod_power.jsonl"), cases)
    elif mode == "comparison":
        cases = gen_binary(COMPARISON_OPS, DT_PAIRS, list(PAIR_LAYOUTS.keys()))
        write_jsonl(os.path.join(corpus_dir, "comparison.jsonl"), cases)
    elif mode == "unary":
        cases = gen_unary(UNARY_OPS, UNARY_DTYPES, list(LAYOUTS.keys()))
        write_jsonl(os.path.join(corpus_dir, "unary.jsonl"), cases)
    elif mode == "reduce":
        cases = gen_reduce(REDUCE_OPS, REDUCE_DTYPES, REDUCE_LAYOUTS)
        write_jsonl(os.path.join(corpus_dir, "reduce.jsonl"), cases)
    elif mode == "where":
        cases = gen_where(WHERE_DT_PAIRS, list(WHERE_LAYOUTS.keys()))
        write_jsonl(os.path.join(corpus_dir, "where.jsonl"), cases)
    elif mode == "place":
        cases = gen_place(PLACE_DTYPES, PLACE_LAYOUTS)
        write_jsonl(os.path.join(corpus_dir, "place.jsonl"), cases)
    elif mode == "matmul":
        cases = gen_matmul(MATMUL_SHAPE_CASES, MATMUL_DTYPES, MATMUL_LAYOUTS)
        write_jsonl(os.path.join(corpus_dir, "matmul.jsonl"), cases)
    elif mode == "bitwise":
        cases = gen_binary(BITWISE_BIN_OPS, BITWISE_DT_PAIRS, list(PAIR_LAYOUTS.keys()))
        cases += gen_unary(INVERT_OP, INT_BOOL_DTYPES, list(LAYOUTS.keys()))
        cases += gen_shift(SHIFT_OPS, SHIFT_DTYPES)
        write_jsonl(os.path.join(corpus_dir, "bitwise.jsonl"), cases)
    elif mode == "unary_extra":
        cases = gen_unary(UNARY_EXTRA_OPS, ALL_DTYPES, list(LAYOUTS.keys()))
        write_jsonl(os.path.join(corpus_dir, "unary_extra.jsonl"), cases)
    elif mode == "nanreduce":
        cases = gen_reduce(NAN_REDUCE_OPS, NAN_REDUCE_DTYPES, REDUCE_LAYOUTS)
        write_jsonl(os.path.join(corpus_dir, "nanreduce.jsonl"), cases)
    elif mode == "scan":
        cases = gen_scan(SCAN_OPS, SCAN_DTYPES, SCAN_LAYOUTS)
        cases += gen_diff(SCAN_DTYPES, SCAN_LAYOUTS)
        write_jsonl(os.path.join(corpus_dir, "scan.jsonl"), cases)
    elif mode == "stat":
        cases = gen_reduce(STAT_REDUCE_OPS, STAT_DTYPES, STAT_LAYOUTS)
        cases += gen_count_nonzero(CNZ_DTYPES, STAT_LAYOUTS)
        cases += gen_quantile(QUANTILE_SPECS, STAT_DTYPES, STAT_LAYOUTS)
        cases += gen_clip(CLIP_DTYPES, STAT_LAYOUTS)
        write_jsonl(os.path.join(corpus_dir, "stat.jsonl"), cases)
    elif mode == "logic":
        cases = gen_unary(LOGIC_UNARY_OPS, LOGIC_UNARY_DTYPES, list(LAYOUTS.keys()))
        cases += gen_binary(LOGIC_BIN_OPS, LOGIC_BIN_PAIRS, list(PAIR_LAYOUTS.keys()))
        write_jsonl(os.path.join(corpus_dir, "logic.jsonl"), cases)
    elif mode == "modf":
        cases = gen_modf(MODF_DTYPES, MODF_LAYOUTS)
        write_jsonl(os.path.join(corpus_dir, "modf.jsonl"), cases)
    elif mode == "manip":
        cases = gen_manip(MANIP_DTYPES, list(LAYOUTS.keys()))
        cases += gen_concat_stack(MANIP_DTYPES)
        cases += gen_pad(MANIP_DTYPES)
        write_jsonl(os.path.join(corpus_dir, "manip.jsonl"), cases)
    elif mode == "sort":
        cases = gen_argsort(SORT_DTYPES)
        cases += gen_searchsorted(SORT_DTYPES)
        cases += gen_nonzero(SORT_DTYPES)
        write_jsonl(os.path.join(corpus_dir, "sort.jsonl"), cases)
    elif mode == "tail":
        cases = gen_tail(TAIL_DTYPES)
        write_jsonl(os.path.join(corpus_dir, "tail.jsonl"), cases)
    else:
        print(f"unknown mode '{mode}' (expected: smoke | astype_full | binary | divmod_power | comparison | unary | reduce | where | place | matmul | bitwise | unary_extra | nanreduce)")
        sys.exit(2)


if __name__ == "__main__":
    main()
