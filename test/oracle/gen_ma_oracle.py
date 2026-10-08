"""
gen_ma_oracle.py — emit the committed, bytes-exact NumPy 2.4.2 oracle corpus for numpy.ma
(masked arrays). Standalone, like gen_nan_oracle.py / gen_index_oracle.py: it owns its own
numbering and writes ma_*.jsonl into ../NumSharp.Tests.Oracle/Fuzz/corpus/.

A MaskedArray is (data + optional bool mask), which the single-buffer NDArray corpus cannot
carry, so this corpus extends the schema minimally (see docs/MA_ORACLE_DESIGN.md):

  operand  = { <the DATA descriptor from layout_catalog.describe>, mask? }   # mask = C-contig bool hex
  expected = { kind:"masked", dtype, shape, buffer, mask }                   # buffer = filled(0); mask = getmaskarray

The C# harness (FuzzCorpusTests.Ma.RunMaCorpus) rebuilds np.ma.masked_array(dataView, cmask),
runs the ma.* op, and compares filled(0) bytes (NaN-tokenized) + getmaskarray bytes (bit-exact).
Op keys are prefixed "ma." so they never collide with the np.* key of the same name.

Run: python gen_ma_oracle.py [all|unary|binary|reduce|scan|manip|construct|select|sortsetops|extras]
Requires numpy==2.4.2.
"""
import json
import os
import sys
import warnings

import numpy as np

np.seterr(all="ignore")
warnings.simplefilter("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout_catalog import LAYOUTS, PAIR_LAYOUTS, describe, _cbase, _fill  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Axes. Mask propagation is per-element (dtype-independent), and np.ma delegates the VALUE to the
# identical np.* op (already gated by the ordinary corpus), so the ma corpus's unique job is the
# MASK behaviour + the ma-specific dtype rules. A compact but representative layout × dtype × mask
# cross-product carries that without exploding the corpus.
# ---------------------------------------------------------------------------------------------
MA_LAYOUTS = ["c_contiguous_1d", "c_contiguous_2d", "f_contiguous_2d", "transposed_2d",
              "strided_step2_1d", "negstride_1d", "simple_slice_offset_1d"]
# PASS 2 — edge layouts that exercise rank>2, broadcast (read-only) views, degenerate shapes and
# 2-D non-contiguity, the paths most likely to hide more strided/rank/empty mask bugs (the DimsOf
# class). Run these through the non-mutating tiers (unary/reduce/manip) on top of MA_LAYOUTS.
MA_LAYOUTS_EDGE = ["c_contiguous_3d", "transposed_3d", "f_contiguous_3d",
                   "broadcast_1d_to_2d", "scalar_0d", "one_element_1d",
                   "empty_2d", "negstride_2d_offset", "strided_2d_cols", "reshape_view_2d"]
MA_DTYPES = ["bool", "uint8", "int32", "int64", "float16", "float32", "float64", "complex128"]
MA_MASKS = ["none", "some", "all"]

# Pairwise: a compact dtype-pair set (same-type + int/float crossing + complex + bool promotion)
# and a representative pair-layout subset (contiguous, one-Fortran, inner-strided, negative-stride,
# broadcast-row). Mask COMBOS exercise OR-propagation: neither / both / one-sided / all|some.
MA_DT_PAIRS = [("int32", "int32"), ("int32", "float64"), ("float32", "float32"),
               ("float64", "float64"), ("complex128", "complex128"), ("int64", "int64"),
               ("uint8", "uint8"), ("float16", "float16"), ("bool", "int32"),
               # PASS 2 — narrow-int width mixing, signed/unsigned crossing, int/complex + float widths
               ("int8", "int8"), ("int16", "uint16"), ("uint32", "int64"),
               ("int32", "complex128"), ("float16", "float64"), ("uint8", "int8")]
# PASS 2 — pp_broadcast_col (the other broadcast direction) + pp_scalar_right/left (a 0-d operand,
# the strong-scalar-widening path pass 12 found a bug in — more coverage across all ops/dtypes).
MA_PAIR_LAYOUTS = ["pp_contig_contig", "pp_contig_fortran", "pp_contig_strided",
                   "pp_negstride_both", "pp_broadcast_row", "pp_broadcast_col",
                   "pp_scalar_right", "pp_scalar_left"]
MA_MASK_COMBOS = [("none", "none"), ("some", "some"), ("some", "none"), ("all", "some")]


# ---------------------------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------------------------
def _ma_mask(shape, pattern):
    """A C-contiguous bool mask of `shape`: none->None (nomask), all->all True, some->F F T T ...
    (a mixed pattern that hits both the SIMD-lane interior and the edges)."""
    n = int(np.prod(shape)) if len(shape) else 1
    if pattern == "none":
        return None
    if pattern == "all":
        return np.ones(shape, dtype=bool) if len(shape) else np.array(True)
    flat = (np.arange(n) % 4 >= 2)
    return flat.reshape(shape) if len(shape) else np.array(bool(flat[0]))


def _ma_operand(base, view, mask):
    """Operand descriptor: the DATA view (describe) + optional C-contiguous bool mask hex."""
    d = describe(base, view)
    if mask is not None:
        d = dict(d)
        d["mask"] = np.ascontiguousarray(np.asarray(mask, dtype=bool)).tobytes().hex()
    return d


def _ma_build(view, mask):
    """The numpy MaskedArray the operand describes (nomask when mask is None)."""
    return np.ma.array(view, mask=(np.ma.nomask if mask is None else mask))


def _ma_expected(r):
    """Expected for a masked result: filled(0) data bytes + getmaskarray bytes, both C-contiguous.
    Robust to a MaskedArray, the `masked` singleton, or a bare numpy scalar (partial-mask flat
    reduction) — filled/getmaskarray/shape are all defined on each."""
    filled = np.asarray(np.ma.filled(r, 0))
    marr = np.asarray(np.ma.getmaskarray(r), dtype=bool)
    return {
        "kind": "masked",
        "dtype": filled.dtype.name,
        "shape": [int(d) for d in np.shape(r)],
        "buffer": np.ascontiguousarray(filled).tobytes().hex(),
        "mask": np.ascontiguousarray(marr).tobytes().hex(),
    }


def _plain_expected(r):
    """Expected for a bare-NDArray-returning ma op (array kind)."""
    a = np.asarray(r)
    return {"dtype": a.dtype.name, "shape": [int(d) for d in a.shape],
            "buffer": np.ascontiguousarray(a).tobytes().hex()}


def _scalar_expected(v):
    """Expected for a scalar/bool-returning ma predicate (scalar kind)."""
    a = np.asarray(v)
    return {"kind": "scalar", "dtype": a.dtype.name, "shape": [int(d) for d in a.shape],
            "buffer": np.ascontiguousarray(a).tobytes().hex()}


def _dtype_expected(dt):
    """Expected for a dtype-returning ma op (make_mask_descr)."""
    return {"kind": "dtype", "value": np.dtype(dt).name}


class _N:
    """A mutable running id counter shared across a tier's emit calls."""
    def __init__(self):
        self.v = 0

    def next(self):
        self.v += 1
        return self.v


def _case(opname, params, operands, expected, layout, valueclass, n):
    return {
        "id": f"ma.{opname}/{layout}/{valueclass}/{n.next()}",
        "op": f"ma.{opname}",
        "params": params,
        "operands": operands,
        "expected": expected,
        "layout": layout,
        "valueclass": valueclass,
    }


# ---------------------------------------------------------------------------------------------
# Op tables (NumPy oracle callables)
# ---------------------------------------------------------------------------------------------
def _unary_fns():
    ma = np.ma
    names = ["negative", "absolute", "conjugate", "sqrt", "exp", "log", "log2", "log10",
             "sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
             "arcsinh", "arccosh", "arctanh", "floor", "ceil", "fabs", "angle", "logical_not"]
    d = {nm: getattr(ma, nm) for nm in names if getattr(ma, nm, None) is not None}
    d["abs"] = lambda m: abs(m)          # __abs__ -> masked abs (np.ma has no top-level `abs`)
    d["around"] = lambda m: ma.around(m)  # decimals=0; C# default matches
    return d


def _binary_fns():
    ma = np.ma
    names = ["add", "subtract", "multiply", "divide", "true_divide", "floor_divide", "mod",
             "remainder", "fmod", "power", "arctan2", "hypot", "equal", "not_equal", "less",
             "less_equal", "greater", "greater_equal", "logical_and", "logical_or", "logical_xor",
             "bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift",
             "maximum", "minimum"]
    return {nm: getattr(ma, nm) for nm in names if getattr(ma, nm, None) is not None}


# ---------------------------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------------------------
def gen_ma_unary():
    cases, n, skipped = [], _N(), 0
    ops = _unary_fns()
    for ln in MA_LAYOUTS + MA_LAYOUTS_EDGE:
        fn = LAYOUTS[ln]
        for s in MA_DTYPES:
            base, view = fn(np.dtype(s))
            for mp in MA_MASKS:
                mask = _ma_mask(view.shape, mp)
                operand = _ma_operand(base, view, mask)
                m = _ma_build(view, mask)
                for opname, f in ops.items():
                    try:
                        r = f(m)
                    except Exception:
                        skipped += 1
                        continue
                    cases.append(_case(opname, {}, [operand], _ma_expected(r), ln, f"{s}.{mp}", n))
    if skipped:
        print(f"  (ma_unary skipped {skipped} where NumPy raised)")
    return cases


def gen_ma_binary():
    cases, n, skipped = [], _N(), 0
    ops = _binary_fns()
    for ln in MA_PAIR_LAYOUTS:
        fn = PAIR_LAYOUTS[ln]
        for da, db in MA_DT_PAIRS:
            ba, va, bb, vb = fn(np.dtype(da), np.dtype(db))
            for mpa, mpb in MA_MASK_COMBOS:
                ma_mask = _ma_mask(va.shape, mpa)
                mb_mask = _ma_mask(vb.shape, mpb)
                oa = _ma_operand(ba, va, ma_mask)
                ob = _ma_operand(bb, vb, mb_mask)
                A = _ma_build(va, ma_mask)
                B = _ma_build(vb, mb_mask)
                for opname, f in ops.items():
                    try:
                        r = f(A, B)
                    except Exception:
                        skipped += 1
                        continue
                    cases.append(_case(opname, {}, [oa, ob], _ma_expected(r),
                                       ln, f"{da}_{db}.{mpa}{mpb}", n))
    if skipped:
        print(f"  (ma_binary skipped {skipped} where NumPy raised)")
    return cases


def _axes(ndim):
    if ndim <= 0:
        return [None]
    if ndim == 1:
        return [None, 0]
    return [None, 0, ndim - 1]


def gen_ma_reduce():
    cases, n, skipped = [], _N(), 0
    ma = np.ma
    masked_ops = {
        "sum": (lambda m, ax, kd: ma.sum(m, axis=ax, keepdims=kd), True),
        "prod": (lambda m, ax, kd: ma.prod(m, axis=ax, keepdims=kd), True),
        "mean": (lambda m, ax, kd: ma.mean(m, axis=ax, keepdims=kd), True),
        "min": (lambda m, ax, kd: ma.min(m, axis=ax, keepdims=kd), True),
        "max": (lambda m, ax, kd: ma.max(m, axis=ax, keepdims=kd), True),
        "ptp": (lambda m, ax, kd: ma.ptp(m, axis=ax, keepdims=kd), True),
        "std": (lambda m, ax, kd: ma.std(m, axis=ax, keepdims=kd), True),
        "var": (lambda m, ax, kd: ma.var(m, axis=ax, keepdims=kd), True),
        "all": (lambda m, ax, kd: ma.all(m, axis=ax, keepdims=kd), True),
        "any": (lambda m, ax, kd: ma.any(m, axis=ax, keepdims=kd), True),
        "median": (lambda m, ax, kd: ma.median(m, axis=ax, keepdims=kd), True),
        "average": (lambda m, ax, kd: ma.average(m, axis=ax, keepdims=kd), True),
        "anom": (lambda m, ax, kd: ma.anom(m, axis=ax), False),   # no keepdims
    }
    array_ops = {
        "count": lambda m, ax: ma.count(m, axis=ax),
        "count_masked": lambda m, ax: ma.count_masked(m, axis=ax),
        "argmin": lambda m, ax: ma.argmin(m, axis=ax),
        "argmax": lambda m, ax: ma.argmax(m, axis=ax),
    }
    for ln in MA_LAYOUTS + MA_LAYOUTS_EDGE:
        fn = LAYOUTS[ln]
        for s in MA_DTYPES:
            base, view = fn(np.dtype(s))
            for mp in MA_MASKS:
                mask = _ma_mask(view.shape, mp)
                operand = _ma_operand(base, view, mask)
                m = _ma_build(view, mask)
                for ax in _axes(view.ndim):
                    axp = {} if ax is None else {"axis": int(ax)}
                    for opname, (f, has_kd) in masked_ops.items():
                        # KNOWN GAP (documented): np.ma all/any/median ignore keepdims on the flat
                        # (axis=None) path — NumSharp returns a 0-d result where NumPy returns (1,)*ndim.
                        # Exclude that one combo so the byte-exact surface stays gated; every axis'd and
                        # every other-op keepdims case is kept. See docs/MA_ORACLE_DESIGN.md "Known gaps".
                        kds = [False, True] if has_kd else [False]
                        if ax is None and opname in ("all", "any", "median"):
                            kds = [False]
                        for kd in kds:
                            try:
                                r = f(m, ax, kd)
                            except Exception:
                                skipped += 1
                                continue
                            p = dict(axp)
                            if kd:
                                p["keepdims"] = True
                            cases.append(_case(opname, p, [operand], _ma_expected(r),
                                               ln, f"{s}.{mp}", n))
                    for opname, f in array_ops.items():
                        try:
                            r = f(m, ax)
                        except Exception:
                            skipped += 1
                            continue
                        cases.append(_case(opname, dict(axp), [operand], _plain_expected(r),
                                           ln, f"{s}.{mp}", n))
    if skipped:
        print(f"  (ma_reduce skipped {skipped} where NumPy raised)")
    return cases


def gen_ma_scan():
    cases, n, skipped = [], _N(), 0
    ma = np.ma
    for ln in MA_LAYOUTS + MA_LAYOUTS_EDGE:
        fn = LAYOUTS[ln]
        for s in MA_DTYPES:
            base, view = fn(np.dtype(s))
            for mp in MA_MASKS:
                mask = _ma_mask(view.shape, mp)
                operand = _ma_operand(base, view, mask)
                m = _ma_build(view, mask)
                jobs = [("cumsum", {}, lambda mm: ma.cumsum(mm)),
                        ("cumprod", {}, lambda mm: ma.cumprod(mm)),
                        ("ediff1d", {}, lambda mm: ma.ediff1d(mm))]
                if view.ndim >= 1:
                    jobs += [("diff", {"n": 1}, lambda mm: ma.diff(mm, 1)),
                             ("diff", {"n": 2}, lambda mm: ma.diff(mm, 2)),
                             ("diff", {"n": 3}, lambda mm: ma.diff(mm, 3))]
                    if view.ndim >= 2:
                        jobs += [("cumsum", {"axis": 0}, lambda mm: ma.cumsum(mm, axis=0)),
                                 ("cumprod", {"axis": 1}, lambda mm: ma.cumprod(mm, axis=1)),
                                 ("diff", {"axis": 0}, lambda mm: ma.diff(mm, 1, axis=0)),
                                 ("cumsum", {"axis": 1}, lambda mm: ma.cumsum(mm, axis=1)),
                                 ("diff", {"n": 2, "axis": 1}, lambda mm: ma.diff(mm, 2, axis=1))]
                for opname, p, f in jobs:
                    try:
                        r = f(m)
                    except Exception:
                        skipped += 1
                        continue
                    cases.append(_case(opname, p, [operand], _ma_expected(r), ln, f"{s}.{mp}", n))
    if skipped:
        print(f"  (ma_scan skipped {skipped} where NumPy raised)")
    return cases


def gen_ma_manip():
    cases, n, skipped = [], _N(), 0
    ma = np.ma
    for ln in MA_LAYOUTS + MA_LAYOUTS_EDGE:
        fn = LAYOUTS[ln]
        for s in MA_DTYPES:
            base, view = fn(np.dtype(s))
            nd = view.ndim
            sz = int(view.size)
            for mp in MA_MASKS:
                mask = _ma_mask(view.shape, mp)
                operand = _ma_operand(base, view, mask)
                m = _ma_build(view, mask)
                # single-operand masked-result jobs
                jobs = [
                    ("ravel", {}, lambda mm: ma.ravel(mm)),
                    ("flatten", {}, lambda mm: mm.flatten()),
                    ("transpose", {}, lambda mm: ma.transpose(mm)),
                    ("squeeze", {}, lambda mm: ma.squeeze(mm)),
                    ("atleast_1d", {}, lambda mm: ma.atleast_1d(mm)),
                    ("atleast_2d", {}, lambda mm: ma.atleast_2d(mm)),
                    ("atleast_3d", {}, lambda mm: ma.atleast_3d(mm)),
                    ("reshape", {"shape": [sz]}, lambda mm: ma.reshape(mm, (sz,))),
                    ("expand_dims", {"axis": 0}, lambda mm: ma.expand_dims(mm, 0)),
                    ("repeat", {"repeats": 2}, lambda mm: ma.repeat(mm, 2)),
                    ("diag", {"k": 0}, lambda mm: ma.diag(mm)),
                ]
                if nd >= 1:
                    jobs += [("moveaxis", {"source": 0, "destination": nd - 1},
                              lambda mm, nd=nd: ma.moveaxis(mm, 0, nd - 1)),
                             ("expand_dims", {"axis": nd}, lambda mm, nd=nd: ma.expand_dims(mm, nd))]  # trailing axis
                if nd == 1:
                    jobs += [("diag", {"k": 1}, lambda mm: ma.diag(mm, 1))]   # 1-D -> offset diagonal matrix
                if nd >= 2:
                    jobs += [("swapaxes", {"axis1": 0, "axis2": 1}, lambda mm: ma.swapaxes(mm, 0, 1)),
                             ("diagonal", {}, lambda mm: ma.diagonal(mm)),
                             ("diagonal", {"offset": 1}, lambda mm: ma.diagonal(mm, 1)),
                             ("moveaxis", {"source": nd - 1, "destination": 0},
                              lambda mm, nd=nd: ma.moveaxis(mm, nd - 1, 0))]
                for opname, p, f in jobs:
                    try:
                        r = f(m)
                    except Exception:
                        skipped += 1
                        continue
                    cases.append(_case(opname, p, [operand], _ma_expected(r), ln, f"{s}.{mp}", n))
                # array-returning accessors
                for opname, f in [("compressed", ma.compressed), ("getdata", ma.getdata),
                                  ("getmaskarray", ma.getmaskarray)]:
                    try:
                        r = f(m)
                    except Exception:
                        skipped += 1
                        continue
                    cases.append(_case(opname, {}, [operand], _plain_expected(r), ln, f"{s}.{mp}", n))
                try:
                    cases.append(_case("filled", {"fill": 0.0}, [operand],
                                       _plain_expected(ma.filled(m, 0)), ln, f"{s}.{mp}", n))
                except Exception:
                    skipped += 1
    # multi-operand joins (same view twice, two mask patterns) + hsplit tuple
    for ln in ["c_contiguous_1d", "c_contiguous_2d"]:
        fn = LAYOUTS[ln]
        for s in ["int32", "float64", "complex128", "uint8"]:
            base, view = fn(np.dtype(s))
            m1_mask = _ma_mask(view.shape, "some")
            m2_mask = _ma_mask(view.shape, "all")
            o1 = _ma_operand(base, view, m1_mask)
            o2 = _ma_operand(base, view, m2_mask)
            A = _ma_build(view, m1_mask)
            B = _ma_build(view, m2_mask)
            joins = [("concatenate", {"axis": 0}, lambda a, b: ma.concatenate([a, b], axis=0)),
                     ("stack", {"axis": 0}, lambda a, b: ma.stack([a, b], axis=0)),
                     ("hstack", {}, lambda a, b: ma.hstack([a, b])),
                     ("vstack", {}, lambda a, b: ma.vstack([a, b])),
                     ("append", {}, lambda a, b: ma.append(a, b))]
            for opname, p, f in joins:
                try:
                    r = f(A, B)
                except Exception:
                    skipped += 1
                    continue
                cases.append(_case(opname, p, [o1, o2], _ma_expected(r), ln, f"{s}.somall", n))
            # hsplit -> masked tuple (needs a 2-D whose last axis divides)
            if view.ndim == 2 and view.shape[1] % 2 == 0:
                try:
                    parts = ma.hsplit(A, 2)
                    cases.append({
                        "id": f"ma.hsplit/{ln}/{s}.some/{n.next()}",
                        "op": "ma.hsplit",
                        "params": {"sections": 2},
                        "operands": [o1],
                        "expected": {"kind": "masked_tuple", "slots": [_ma_expected(p) for p in parts]},
                        "layout": ln, "valueclass": f"{s}.some",
                    })
                except Exception:
                    skipped += 1
    if skipped:
        print(f"  (ma_manip skipped {skipped} where NumPy raised)")
    return cases


def gen_ma_construct():
    cases, n, skipped = [], _N(), 0
    ma = np.ma
    # constructors operate on a base (mostly nomask, plus a pre-masked base to test composition)
    for ln in MA_LAYOUTS + MA_LAYOUTS_EDGE:
        fn = LAYOUTS[ln]
        for s in MA_DTYPES:
            base, view = fn(np.dtype(s))
            for mp in ["none", "some"]:
                mask = _ma_mask(view.shape, mp)
                operand = _ma_operand(base, view, mask)
                m = _ma_build(view, mask)
                cond_arr = np.asarray(np.ascontiguousarray(view) > 0, dtype=bool)
                cond_op = _ma_operand(np.ascontiguousarray(cond_arr), cond_arr, None)
                jobs = [
                    ("masked_greater", {"value": 0.0}, [operand], lambda: ma.masked_greater(m, 0.0)),
                    ("masked_less", {"value": 0.0}, [operand], lambda: ma.masked_less(m, 0.0)),
                    ("masked_greater_equal", {"value": 0.0}, [operand], lambda: ma.masked_greater_equal(m, 0.0)),
                    ("masked_less_equal", {"value": 0.0}, [operand], lambda: ma.masked_less_equal(m, 0.0)),
                    ("masked_equal", {"value": 1.0}, [operand], lambda: ma.masked_equal(m, 1.0)),
                    ("masked_not_equal", {"value": 1.0}, [operand], lambda: ma.masked_not_equal(m, 1.0)),
                    ("masked_inside", {"v1": -1.0, "v2": 1.0}, [operand], lambda: ma.masked_inside(m, -1.0, 1.0)),
                    ("masked_outside", {"v1": -1.0, "v2": 1.0}, [operand], lambda: ma.masked_outside(m, -1.0, 1.0)),
                    ("masked_values", {"value": 1.0}, [operand], lambda: ma.masked_values(m, 1.0)),
                    ("masked_invalid", {}, [operand], lambda: ma.masked_invalid(m)),
                    ("fix_invalid", {}, [operand], lambda: ma.fix_invalid(m)),
                    ("masked_all_like", {}, [operand], lambda: ma.masked_all_like(m)),
                    ("array", {}, [operand], lambda: ma.array(m)),
                    ("masked_array", {}, [operand], lambda: ma.masked_array(m)),
                    ("asarray", {}, [operand], lambda: ma.asarray(m)),
                    ("copy", {}, [operand], lambda: ma.copy(m)),
                    ("masked_where", {}, [cond_op, operand], lambda: ma.masked_where(cond_arr, m)),
                ]
                for opname, p, opnds, f in jobs:
                    try:
                        r = f()
                    except Exception:
                        skipped += 1
                        continue
                    cases.append(_case(opname, p, opnds, _ma_expected(r), ln, f"{s}.{mp}", n))
    # masked_all / masked_all_like shapes (no operand)
    for shape in [[3], [2, 3], [4]]:
        for s in ["int32", "float64", "complex128", "uint8"]:
            r = ma.masked_all(tuple(shape), dtype=np.dtype(s))
            cases.append({
                "id": f"ma.masked_all/creation/{s}/{n.next()}",
                "op": "ma.masked_all",
                "params": {"shape": shape, "dtype": s},
                "operands": [],
                "expected": _ma_expected(r),
                "layout": "creation", "valueclass": f"{s}",
            })
    if skipped:
        print(f"  (ma_construct skipped {skipped} where NumPy raised)")
    return cases


def gen_ma_select():
    cases, n, skipped = [], _N(), 0
    ma = np.ma
    for ln in MA_LAYOUTS:
        fn = LAYOUTS[ln]
        for s in MA_DTYPES:
            base, view = fn(np.dtype(s))
            sz = int(view.size)
            if sz == 0:
                continue
            for mp in MA_MASKS:
                mask = _ma_mask(view.shape, mp)
                operand = _ma_operand(base, view, mask)
                m = _ma_build(view, mask)
                # clip (float threshold; on ints numpy clips fine)
                try:
                    cases.append(_case("clip", {"min": -1.0, "max": 1.0}, [operand],
                                       _ma_expected(ma.clip(m, -1.0, 1.0)), ln, f"{s}.{mp}", n))
                except Exception:
                    skipped += 1
                # take: gather over a flat index set (raveled logical order)
                idx = np.array([0, sz - 1, sz // 2, 0], dtype=np.int32)
                idx_op = _ma_operand(np.ascontiguousarray(idx), idx, None)
                try:
                    cases.append(_case("take", {}, [operand, idx_op],
                                       _ma_expected(ma.take(m, idx)), ln, f"{s}.{mp}", n))
                except Exception:
                    skipped += 1
                # compress: boolean condition over the raveled data
                cond = (np.arange(sz) % 2 == 0)
                cond_op = _ma_operand(np.ascontiguousarray(cond), cond, None)
                try:
                    cases.append(_case("compress", {}, [cond_op, operand],
                                       _ma_expected(ma.compress(cond, m)), ln, f"{s}.{mp}", n))
                except Exception:
                    skipped += 1
                # take with OUT-OF-RANGE indices under wrap/clip modes (raise mode is the default take above)
                oob = np.array([0, sz, sz + 1, -1, -(sz + 1)], dtype=np.int32)
                oob_op = _ma_operand(np.ascontiguousarray(oob), oob, None)
                for mode in ("wrap", "clip"):
                    try:
                        cases.append(_case("take", {"mode": mode}, [operand, oob_op],
                                           _ma_expected(ma.take(m, oob, mode=mode)), ln, f"{s}.{mp}", n))
                    except Exception:
                        skipped += 1
    # where(cond, x, y): 3 masked operands over a contiguous shape
    for ln in ["c_contiguous_1d", "c_contiguous_2d"]:
        fn = LAYOUTS[ln]
        for s in ["int32", "float64", "complex128"]:
            base, view = fn(np.dtype(s))
            cond = np.asarray(np.ascontiguousarray(view) > 0, dtype=bool)
            cond_op = _ma_operand(np.ascontiguousarray(cond), cond, None)
            xm = _ma_mask(view.shape, "some")
            ym = _ma_mask(view.shape, "all")
            ox = _ma_operand(base, view, xm)
            oy = _ma_operand(base, view, ym)
            X = _ma_build(view, xm)
            Y = _ma_build(view, ym)
            try:
                r = ma.where(cond, X, Y)
                cases.append(_case("where", {}, [cond_op, ox, oy], _ma_expected(r), ln, f"{s}.wh", n))
            except Exception:
                skipped += 1
    # choose(indices, [choice arrays]) — index selects among N masked choice arrays (1-D)
    for s in ["int32", "float64", "complex128", "uint8"]:
        base, view = LAYOUTS["c_contiguous_1d"](np.dtype(s))
        szc = int(view.size)
        idxc = (np.arange(szc) % 3).astype(np.int32)
        idxc_op = _ma_operand(np.ascontiguousarray(idxc), idxc, None)
        chs, ch_ops = [], []
        for k in range(3):
            cv = _fill(szc, np.dtype(s))
            cm = _ma_mask(cv.shape, "some" if k == 1 else "none")   # one masked choice exercises mask-OR
            chs.append(_ma_build(cv, cm)); ch_ops.append(_ma_operand(np.ascontiguousarray(cv), cv, cm))
        try:
            r = ma.choose(np.ma.array(idxc), chs)
            cases.append(_case("choose", {}, [idxc_op] + ch_ops, _ma_expected(r), "c_contiguous_1d", f"{s}.choose", n))
        except Exception:
            skipped += 1
    # put / putmask (mutating -> compare the mutated masked array)
    for ln in ["c_contiguous_1d"]:
        fn = LAYOUTS[ln]
        for s in ["int32", "float64", "uint8"]:
            base, view = fn(np.dtype(s))
            sz = int(view.size)
            for mp in ["none", "some"]:
                mask = _ma_mask(view.shape, mp)
                idx = np.array([0, 2, sz - 1], dtype=np.int32)
                idx_op = _ma_operand(np.ascontiguousarray(idx), idx, None)
                vals = np.array([9, 8, 7], dtype=np.dtype(s))
                vals_op = _ma_operand(np.ascontiguousarray(vals), vals, None)
                # put
                try:
                    mm = _ma_build(view, mask).copy()
                    ma.put(mm, idx, vals)
                    op0 = _ma_operand(base, view, mask)
                    cases.append(_case("put", {}, [op0, idx_op, vals_op], _ma_expected(mm), ln, f"{s}.{mp}", n))
                except Exception:
                    skipped += 1
                # putmask. NOTE (2026-10-08): this block has never emitted a case: val1_op describes two DIFFERENT
                # arrays as (base, view), so layout_catalog.describe raises and the except below skips it. putmask is
                # covered by the numpy.ma surface tier (gen_ma_surface_oracle.py, section "putmask") instead; left as is
                # so this tier's committed output does not change.
                try:
                    mm = _ma_build(view, mask).copy()
                    pmask = (np.arange(sz) % 3 == 0)
                    pmask_op = _ma_operand(np.ascontiguousarray(pmask), pmask, None)
                    ma.putmask(mm, pmask, np.array([5], dtype=np.dtype(s)))
                    op0 = _ma_operand(base, view, mask)
                    val1_op = _ma_operand(np.array([5], dtype=np.dtype(s)),
                                          np.array([5], dtype=np.dtype(s)), None)
                    cases.append(_case("putmask", {}, [op0, pmask_op, val1_op], _ma_expected(mm), ln, f"{s}.{mp}", n))
                except Exception:
                    skipped += 1
    if skipped:
        print(f"  (ma_select skipped {skipped} where NumPy raised)")
    return cases


def gen_ma_sortsetops():
    cases, n, skipped = [], _N(), 0
    ma = np.ma
    for ln in MA_LAYOUTS + MA_LAYOUTS_EDGE:
        fn = LAYOUTS[ln]
        for s in MA_DTYPES:
            base, view = fn(np.dtype(s))
            for mp in MA_MASKS:
                mask = _ma_mask(view.shape, mp)
                operand = _ma_operand(base, view, mask)
                m = _ma_build(view, mask)
                for ew in [True, False]:
                    try:
                        cases.append(_case("sort", {"endwith": ew}, [operand],
                                           _ma_expected(ma.sort(m, endwith=ew)), ln, f"{s}.{mp}", n))
                    except Exception:
                        skipped += 1
                    # KNOWN GAP (documented): np.ma.argsort's CURRENT default axis is None (flatten), but
                    # numpy 2.4.2 emits a FutureWarning that it WILL become -1 to match np.argsort; NumSharp
                    # has no axis=None flatten and uses -1 (numpy's documented FUTURE default). They agree
                    # for 1-D (axis -1 == None), so restrict argsort to 1-D. See docs/MA_ORACLE_DESIGN.md.
                    # KNOWN GAP (documented): masked argsort's tie-break / masked-slot INDEX ordering
                    # diverges (NumPy fills masked with a sentinel then argsorts; the order among equal
                    # values and masked positions differs from NumSharp's stable radix), and even nomask
                    # argsort's tie order differs on a REVERSED (negative-stride) float view. The
                    # contiguous/strided/offset nomask path is byte-exact, so restrict argsort there.
                    # See docs/MA_ORACLE_DESIGN.md "Known gaps".
                    if view.ndim == 1 and mp == "none" and ln != "negstride_1d":
                        try:
                            cases.append(_case("argsort", {"endwith": ew}, [operand],
                                               _plain_expected(ma.argsort(m, endwith=ew)), ln, f"{s}.{mp}", n))
                        except Exception:
                            skipped += 1
                # KNOWN GAP (documented): np.ma.unique treats each MASKED element as a distinct value
                # (its underlying datum participates), an ill-specified numpy quirk NumSharp does not
                # reproduce; the NOMASK path is byte-exact, so restrict unique to nomask inputs here and
                # cover masked-unique by unit tests. See docs/MA_ORACLE_DESIGN.md "Known gaps".
                if mp == "none":
                    try:
                        cases.append(_case("unique", {}, [operand], _ma_expected(ma.unique(m)), ln, f"{s}.{mp}", n))
                    except Exception:
                        skipped += 1
    # set ops over two 1-D masked arrays
    for s in ["int32", "int64", "float64", "uint8"]:
        a = np.array([1, 2, 3, 4, 2, 5], dtype=np.dtype(s))
        b = np.array([3, 4, 5, 6, 7, 3], dtype=np.dtype(s))
        am = _ma_mask(a.shape, "some")
        bm = _ma_mask(b.shape, "some")
        oa = _ma_operand(np.ascontiguousarray(a), a, am)
        ob = _ma_operand(np.ascontiguousarray(b), b, bm)
        A = _ma_build(a, am)
        B = _ma_build(b, bm)
        for opname, f in [("intersect1d", ma.intersect1d), ("union1d", ma.union1d),
                          ("setxor1d", ma.setxor1d), ("setdiff1d", ma.setdiff1d)]:
            try:
                cases.append(_case(opname, {}, [oa, ob], _ma_expected(f(A, B)), "setops", f"{s}", n))
            except Exception:
                skipped += 1
        for opname, f in [("isin", ma.isin), ("in1d", ma.in1d)]:
            try:
                cases.append(_case(opname, {}, [oa, ob], _ma_expected(f(A, B)), "setops", f"{s}", n))
            except Exception:
                skipped += 1
    if skipped:
        print(f"  (ma_sortsetops skipped {skipped} where NumPy raised)")
    return cases


def gen_ma_extras():
    cases, n, skipped = [], _N(), 0
    ma = np.ma
    # small exact products (dot/inner/outer) — kept tiny so float sums are exact -> byte-parity
    for s in ["int32", "int64", "float64", "float32", "complex128"]:
        a = _fill(6, np.dtype(s)).reshape(2, 3)
        b = _fill(6, np.dtype(s)).reshape(3, 2)
        am = _ma_mask(a.shape, "some")
        bm = _ma_mask(b.shape, "some")
        oa = _ma_operand(np.ascontiguousarray(a), a, am)
        ob = _ma_operand(np.ascontiguousarray(b), b, bm)
        A = _ma_build(a, am)
        B = _ma_build(b, bm)
        try:
            cases.append(_case("dot", {}, [oa, ob], _ma_expected(ma.dot(A, B)), "extras", f"{s}", n))
        except Exception:
            skipped += 1
        v1 = _fill(4, np.dtype(s))
        v2 = _fill(4, np.dtype(s))
        vm1 = _ma_mask(v1.shape, "some")
        vm2 = _ma_mask(v2.shape, "some")
        ov1 = _ma_operand(np.ascontiguousarray(v1), v1, vm1)
        ov2 = _ma_operand(np.ascontiguousarray(v2), v2, vm2)
        V1 = _ma_build(v1, vm1)
        V2 = _ma_build(v2, vm2)
        for opname, f in [("inner", ma.inner), ("outer", ma.outer)]:
            try:
                cases.append(_case(opname, {}, [ov1, ov2], _ma_expected(f(V1, V2)), "extras", f"{s}", n))
            except Exception:
                skipped += 1
    # triangular-mask family + trace/vander (2-D) — array & masked results
    for s in ["int32", "float64", "uint8"]:
        base2 = _cbase((3, 4), np.dtype(s))
        m2mask = _ma_mask(base2.shape, "some")
        o2 = _ma_operand(np.ascontiguousarray(base2), base2, m2mask)
        M2 = _ma_build(base2, m2mask)
        for opname, kind, f in [("mask_rows", "m", ma.mask_rows), ("mask_cols", "m", ma.mask_cols),
                                ("mask_rowcols", "m", ma.mask_rowcols),
                                ("compress_rows", "a", ma.compress_rows),
                                ("compress_cols", "a", ma.compress_cols),
                                ("trace", "a", lambda x: ma.trace(x))]:
            try:
                r = f(M2)
                exp = _ma_expected(r) if kind == "m" else _plain_expected(r)
                cases.append(_case(opname, {}, [o2], exp, "extras", f"{s}", n))
            except Exception:
                skipped += 1
        # vander over a 1-D
        v = _fill(4, np.dtype(s))
        vm = _ma_mask(v.shape, "some")
        ov = _ma_operand(np.ascontiguousarray(v), v, vm)
        try:
            cases.append(_case("vander", {"n": 3}, [ov], _plain_expected(ma.vander(_ma_build(v, vm), 3)),
                               "extras", f"{s}", n))
        except Exception:
            skipped += 1
    # predicates (scalar) + mask helpers (array/dtype)
    for s in ["int32", "float64", "complex128"]:
        a = _fill(6, np.dtype(s))
        for mp in ["none", "some", "all"]:
            am = _ma_mask(a.shape, mp)
            oa = _ma_operand(np.ascontiguousarray(a), a, am)
            A = _ma_build(a, am)
            try:
                cases.append(_case("is_masked", {}, [oa], _scalar_expected(bool(ma.is_masked(A))),
                                   "extras", f"{s}.{mp}", n))
            except Exception:
                skipped += 1
            try:
                cases.append(_case("flatten_mask", {}, [oa],
                                   _plain_expected(ma.flatten_mask(ma.getmaskarray(A))), "extras", f"{s}.{mp}", n))
            except Exception:
                skipped += 1
        # allclose / allequal over two arrays
        b = _fill(6, np.dtype(s))
        am = _ma_mask(a.shape, "some")
        bm = _ma_mask(b.shape, "some")
        oa = _ma_operand(np.ascontiguousarray(a), a, am)
        ob = _ma_operand(np.ascontiguousarray(b), b, bm)
        A = _ma_build(a, am)
        B = _ma_build(b, bm)
        for opname, f in [("allclose", ma.allclose), ("allequal", ma.allequal)]:
            try:
                cases.append(_case(opname, {}, [oa, ob], _scalar_expected(bool(f(A, B))), "extras", f"{s}", n))
            except Exception:
                skipped += 1
        # mask_or over two masks
        try:
            cases.append(_case("mask_or", {}, [oa, ob],
                               _plain_expected(ma.mask_or(ma.getmaskarray(A), ma.getmaskarray(B))),
                               "extras", f"{s}", n))
        except Exception:
            skipped += 1
    # make_mask / make_mask_none / make_mask_descr
    for s in ["int32", "float64", "uint8", "bool"]:
        mvals = np.array([0, 1, 0, 2], dtype=np.dtype(s)) if s != "bool" else np.array([True, False, True, False])
        omv = _ma_operand(np.ascontiguousarray(mvals), mvals, None)
        try:
            cases.append(_case("make_mask", {}, [omv], _plain_expected(ma.make_mask(mvals)), "extras", f"{s}", n))
        except Exception:
            skipped += 1
        cases.append(_case("make_mask_descr", {"dtype": s}, [], _dtype_expected(ma.make_mask_descr(np.dtype(s))),
                           "extras", f"{s}", n))
    for shape in [[3], [2, 3]]:
        cases.append({
            "id": f"ma.make_mask_none/extras/shape/{n.next()}",
            "op": "ma.make_mask_none",
            "params": {"shape": shape},
            "operands": [],
            "expected": _plain_expected(ma.make_mask_none(tuple(shape))),
            "layout": "extras", "valueclass": "shape",
        })
    if skipped:
        print(f"  (ma_extras skipped {skipped} where NumPy raised)")
    return cases


# ---------------------------------------------------------------------------------------------
# ma.mean / ma.var / ma.std / ma.anom, held to numpy.ma's bits with NO excuse (plan api-parameter-parity A6).
#
# NumSharp computes these four as numpy.ma's own statements (src/NumSharp.Core/Ma/NDMaskedArray.cs). Without a mask,
# ndarray's _mean / _var run ON the masked array, so both of their true_divide(out=...) steps pass through
# MaskedArray.__array_wrap__'s divide domain. With a mask (an all-False one included) the mean is `sum * 1. / count`
# through ma.true_divide, the anomalies are `self - mean` through ma.subtract, the square is taken in place (complex:
# abs(danom) ** 2 through ma.power), the sum is divided by count - ddof, and the result's mask is replaced. std is
# ma.sqrt of the variance (NumPy ignores its mean=, so does NumSharp); anom is `self - mean`.
#
# The other ma tiers compare filled(0) bytes and getmaskarray bytes. Every case here records two more fields, which
# FuzzCorpusTests.Ma compares for valueclass "mameanvar" (MisalignedRegistry never excuses that valueclass, and the
# ma harness's own excuses do not run for it):
#   raw  - the result's data at EVERY slot in C order: masked slots too, where NumPy leaves its domain fills and the
#          minuend's put-back data (NaN tokenized, like every value comparison of the oracle);
#   form - the object NumPy returned: "masked" (the singleton), "scalar" (a NumPy scalar, or a 0-d result without a
#          mask), "nomask" (an array whose _mask is nomask) or "mask" (an array with a mask array, all-False included).
# NumPy's errors are cases too (expects_throw, with NumPy's exception type and text).
#
# Sections: L every catalog layout x the 13 dtypes x {nomask, all-False, some} (mean, var, anom); K the mask classes
# (all masked, one unmasked element per row, a masked first row, some) with keepdims and ddof 0 / 1 / 2 / 5; T dtype=
# for every input in eight loops; V the mean= forms of var (and of std, which ignores them); S special values
# (non-finite, huge, -0.0, integer extremes); M sizes where the pairwise order of the sums shows (130 / 1031 elements,
# a 19x13 block in C, F and a permuted 3-D layout); E empty inputs; Z 0-d inputs with an axis and out-of-range axes.
#
# Carved (_mv_carved, counted per rule): a case whose mean sums the operand VIEW itself, because the operand has no
# mask or an all-False one (filled(0) returns the data unchanged then; a mask with a True slot makes a K-order copy,
# whose forward strides np.sum walks in NumPy's order), on a layout np.sum still sums in another order:
#   finding5  - an axis reduction over a view with a reversed axis and several outputs (plan finding 5,
#               OpenBugsReductionTests.Sum_ReversedViewAxis_WalksTheLogicalOrder);
#   broadcast - a broadcast operand reduced flat (plan finding 10, Sum_BroadcastOperand_Flat_IsNumPysCallSchedule) or
#               along a stride-0 axis (plan X12, Sum_AlongBroadcastAxis_IsNumPysOrder);
# each only where that sum is not exact in every order (_mv_order_free).
# ---------------------------------------------------------------------------------------------
MV_DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
             "float16", "float32", "float64", "complex128"]
# The dtype= loops of section T: every kind of arithmetic a loop selects (float widths, complex, integer widths, bool).
MV_LOOPS = ["float16", "float32", "float64", "complex128", "int32", "int64", "uint8", "bool"]
# The largest integer each float loop holds exactly: a float sum of integers inside it is exact in every order.
_MV_EXACT_INT = {"float16": 2.0 ** 11, "float32": 2.0 ** 24, "float64": 2.0 ** 53, "complex128": 2.0 ** 53}
# A mean= argument left out (None is a valid NumPy value for nothing here, so a sentinel marks "not passed").
_MV_NO_MEAN = object()


def _mv_values(dtype, n, rng, special="plain"):
    """`n` values of `dtype` drawn from `rng` (deterministic for a seeded generator).

    plain: floats with fractional parts, so a float sum's bits depend on its order; small integers; random bools.
    nonfinite: a quarter of the slots inf / -inf / nan (complex: in either part). huge: most squares overflow while the
    sums stay finite (float64 and complex times 1e155, float32 1e19, float16 1e3), so var meets inf squares: masked by
    ma.power on the complex path, in the divide domain without a mask. negzero: half the slots -0.0. extreme: an integer
    dtype's min / max / 0 / 1.
    """
    dt = np.dtype(dtype)
    k = dt.kind
    if k == "b":
        return rng.integers(0, 2, n).astype(bool)
    if k in "iu":
        if special == "extreme":
            info = np.iinfo(dt)
            return np.array([info.min, info.max, 0, 1], dtype=dt)[rng.integers(0, 4, n)]
        lo, hi = (0, 50) if k == "u" else (-50, 50)
        return rng.integers(lo, hi, n).astype(dt)
    if k == "f":
        v = rng.standard_normal(n) * 10
        if special == "nonfinite" and n > 0:
            idx = rng.choice(n, size=max(1, n // 4), replace=False)
            v[idx] = rng.choice([np.inf, -np.inf, np.nan], size=len(idx))
        elif special == "huge":
            v = v * {"float64": 1e155, "float32": 1e19, "float16": 1e3}[dt.name]
        elif special == "negzero":
            v[: n // 2] = -0.0
        return v.astype(dt)
    re_ = rng.standard_normal(n) * 10
    im_ = rng.standard_normal(n) * 10
    if special == "nonfinite" and n > 0:
        idx = rng.choice(n, size=max(1, n // 4), replace=False)
        re_[idx] = rng.choice([np.inf, -np.inf, np.nan], size=len(idx))
        idx2 = rng.choice(n, size=max(1, n // 5), replace=False)
        im_[idx2] = rng.choice([np.inf, np.nan], size=len(idx2))
    elif special == "huge":
        re_ *= 1e155
        im_ *= 1e155
    elif special == "negzero":
        re_[: n // 2] = -0.0
        im_[n // 4: (3 * n) // 4] = -0.0
    return (re_ + 1j * im_).astype(dt)


def _mv_mask_fits(shape, kind):
    """Whether mask class `kind` means something for `shape` (row classes need rows)."""
    if kind == "row1":
        return len(shape) >= 1 and shape[-1] > 0
    if kind == "row0":
        return len(shape) >= 2 and shape[0] > 0
    return True


def _mv_mask(shape, kind, rng):
    """The operand's mask for class `kind`: none (nomask), false (an all-False array: numpy.ma's masked path, but
    filled(0) returns the data itself), some (about a third masked), all, row1 (one unmasked element per row: index 0
    of the last axis), row0 (some, with the first row along axis 0 fully masked)."""
    if kind == "none":
        return None
    if kind == "false":
        return np.zeros(shape, dtype=bool)
    if kind == "all":
        return np.ones(shape, dtype=bool)
    if kind == "some":
        return rng.random(shape) < 0.35 if len(shape) else np.array(bool(rng.random() < 0.35))
    if kind == "row1":
        m = np.ones(shape, dtype=bool)
        m[..., 0] = False
        return m
    if kind == "row0":
        m = rng.random(shape) < 0.3
        m[0] = True
        return m
    raise ValueError(f"unknown mask class {kind!r}")


def _mv_expected(r):
    """The ma tiers' masked expectation (filled(0) bytes + getmaskarray bytes) plus this tier's `raw` (the data at
    every slot, C order) and `form` (masked / scalar / nomask / mask: which object NumPy returned)."""
    e = _ma_expected(r)
    if r is np.ma.masked:
        form, raw = "masked", np.asarray(r.data)
    elif isinstance(r, np.ma.MaskedArray):
        raw = np.asarray(r.data)
        if r._mask is np.ma.nomask:
            form = "scalar" if raw.ndim == 0 else "nomask"
        else:
            form = "mask"
    else:
        form, raw = "scalar", np.asarray(r)
    e["raw"] = np.ascontiguousarray(raw).tobytes().hex()
    e["form"] = form
    return e


def _mv_order_free(view, axis, loop):
    """Whether np.sum(view, axis, dtype=loop) is exact in every summation order: an integer / bool loop (modular add,
    logical OR), or a float loop where every element cast to it is finite and every partial sum is representable. The
    elements are all multiples of 2**E (E: the exponent of the lowest set bit over every element), so every partial sum
    is a multiple of 2**E no larger than extent * max|x|; with K = max|x| / 2**E, extent * K <= 2**precision makes every
    partial sum exact, and then no addition rounds in any order. (gen_oracle._mean_order_free, generalized from integers
    to any binary scale for this generator's carve: a float16 mean sums in float32, where such sums are exact.)"""
    if np.dtype(loop).kind in "biu" or view.size == 0:
        return True
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        v = view.astype(loop)
        parts = [v.real, v.imag] if np.dtype(loop).kind == "c" else [v]
        extent = view.size if axis is None else view.shape[axis]
        for p in parts:
            p64 = np.asarray(p, dtype=np.float64).ravel()
            if not np.all(np.isfinite(p64)):
                return False
            nz = p64[p64 != 0]
            if nz.size == 0:
                continue
            mant, exp = np.frexp(np.abs(nz))                           # |x| = mant * 2**exp, 0.5 <= mant < 1
            ints = (mant * 2.0 ** 53).astype(np.int64)                  # the 53-bit significand as an integer
            low = np.log2((ints & -ints).astype(np.float64)).astype(np.int64)   # its trailing zero bits
            e_min = int(np.min(exp - 53 + low))                          # every element is a multiple of 2**e_min
            k = float(np.max(np.abs(nz))) / 2.0 ** e_min                 # max|x| in units of 2**e_min (exact)
            if k * extent > _MV_EXACT_INT[np.dtype(loop).name]:
                return False
            # A partial sum past the loop's range overflows in some orders and not in others (inf - inf is nan).
            if float(np.max(np.abs(nz))) * extent > float(np.finfo(np.dtype(loop)).max):
                return False
    return True


def _mv_sum_loops(view, dtype, op, has_mask):
    """The loop dtypes of the sums that walk the operand view: the mean's (numpy.ma's masked mean and ndarray's _mean:
    dtype=, else float64 for bool / integers, float32 for float16, else the input's) and, for var / std without a
    mask, _var's first sum (dtype=, else float64 for bool / integers, else the input's: float16 sums in float16)."""
    k = view.dtype.kind
    if dtype is not None:
        return [np.dtype(dtype).name]
    loops = ["float64" if k in "biu" else ("float32" if view.dtype == np.float16 else view.dtype.name)]
    if op in ("var", "std") and not has_mask:
        loops.append("float64" if k in "biu" else view.dtype.name)
    return loops


def _mv_carved(view, mask, axis, dtype, op):
    """The np.sum residual a case inherits ("finding5" / "broadcast", see the section header), or None."""
    if mask is not None and mask.any():
        return None                       # filled(0) is a forward K-order copy: np.sum walks it in NumPy's order
    if view.ndim == 0 or view.size == 0:
        return None
    ax = axis
    if ax is not None:
        if not -view.ndim <= ax < view.ndim:
            return None                   # NumPy raises before any sum
        ax %= view.ndim
    if all(_mv_order_free(view, ax, loop) for loop in _mv_sum_loops(view, dtype, op, mask is not None)):
        return None
    if ax is not None and view.ndim >= 2 and any(st < 0 and sh > 1 for st, sh in zip(view.strides, view.shape)):
        return "finding5"
    if any(st == 0 and sh > 1 for st, sh in zip(view.strides, view.shape)):
        if ax is None or (view.strides[ax] == 0 and view.shape[ax] > 1):
            return "broadcast"
    return None


def _mv_call(op, m, params, mean_obj):
    """NumPy's call for one case: np.ma.<op>(m, axis=, dtype=, ddof=, keepdims=, mean=) with only the keywords the
    case sets (keepdims only when True; ddof always for var / std; mean only when a form is given)."""
    kw = {}
    if "axis" in params:
        kw["axis"] = params["axis"]
    if "dtype" in params:
        kw["dtype"] = params["dtype"]
    if op in ("var", "std"):
        kw["ddof"] = params.get("ddof", 0)
    if params.get("keepdims"):
        kw["keepdims"] = True
    if mean_obj is not _MV_NO_MEAN:
        kw["mean"] = mean_obj
    return getattr(np.ma, op)(m, **kw)


def _mv_axes(view, last=True):
    """The axes a section reduces over: None, every axis of a view up to 3-D (0, 2 and the last of a 5-D one), and -1
    when `last` (the negative spelling of the last axis)."""
    nd = view.ndim
    if nd == 0:
        return [None]
    axes = [None] + (list(range(nd)) if nd <= 3 else [0, 2, nd - 1])
    if last and nd >= 2:
        axes.append(-1)
    return axes


def gen_ma_meanvar(carve=True):
    """ma.mean / ma.var / ma.std / ma.anom cases (valueclass "mameanvar"); see the section header.

    carve: True (the committed tier) leaves out the cells _mv_carved names; False emits them too, for a check that
    every carved cell is one the replay gets wrong for the named reason (python gen_ma_oracle.py meanvar --uncarved
    <path>).
    """
    cases, n = [], _N()
    carved = {"finding5": 0, "broadcast": 0}

    def emit(op, base, view, mask, params, ln, tag, mean=None):
        """Record one case. `mean` is None (no mean=) or (param descriptor, NumPy object, extra operand or None)."""
        mean_param, mean_obj, extra = (None, _MV_NO_MEAN, None) if mean is None else mean
        if carve and mean is None:
            # A given mean= replaces the computed mean (var's masked path) or reaches _var as given (nomask path), so
            # no sum walks the view in the mean's loop; only a computed mean inherits np.sum's order.
            rule = _mv_carved(view, mask, params.get("axis"), params.get("dtype"), op)
            if rule is not None:
                carved[rule] += 1
                return
        if carve and mean is not None and op == "std":
            rule = _mv_carved(view, mask, params.get("axis"), params.get("dtype"), op)
            if rule is not None:          # std ignores mean=: its variance computes the mean
                carved[rule] += 1
                return
        p = dict(params)
        if mean_param is not None:
            p["mean"] = mean_param
        operands = [_ma_operand(base, view, mask)]
        if extra is not None:
            operands.append(extra)
        m = _ma_build(view, mask)
        cid = f"ma.{op}/{tag}/{ln}/{n.next()}"
        try:
            r = _mv_call(op, m, p, mean_obj)
        except Exception as e:  # noqa: BLE001 — NumPy's error is the expectation
            cases.append({"id": cid, "op": f"ma.{op}", "params": p, "operands": operands, "expected": {},
                          "expects_throw": True, "error": {"type": type(e).__name__, "text": str(e)},
                          "layout": ln, "valueclass": "mameanvar"})
            return
        cases.append({"id": cid, "op": f"ma.{op}", "params": p, "operands": operands, "expected": _mv_expected(r),
                      "layout": ln, "valueclass": "mameanvar"})

    def refill(base, s, rng, special="plain"):
        """Overwrite the catalog's base with this tier's values (the catalog pools open with NaN / inf)."""
        if base.size:
            base[...] = _mv_values(s, base.size, rng, special).reshape(base.shape)

    # Section L: every catalog layout, every dtype, nomask / all-False / some; mean, var and anom over each axis (std is
    # ma.sqrt of the same variance: the other sections cover it).
    rng = np.random.default_rng(20261006)
    for ln, fn_layout in LAYOUTS.items():
        for s in MV_DTYPES:
            base, view = fn_layout(np.dtype(s))
            refill(base, s, rng)
            for mk in ("none", "false", "some"):
                mask = _mv_mask(view.shape, mk, rng)
                tag = f"L/{s}.{mk}"
                for ax in _mv_axes(view, last=False):
                    axp = {} if ax is None else {"axis": ax}
                    emit("mean", base, view, mask, axp, ln, tag)
                    emit("var", base, view, mask, axp, ln, tag)
                    emit("anom", base, view, mask, axp, ln, tag)

    # Section K: the mask classes with keepdims and ddof 0 / 1 / 2 / 5 (5 is past every slice: cnt - ddof <= 0).
    k_dtypes = ["bool", "int32", "int64", "uint64", "float16", "float32", "float64", "complex128"]
    for ln in ("c_contiguous_2d", "f_contiguous_2d", "strided_2d_cols", "transposed_3d", "c_contiguous_1d"):
        for s in k_dtypes:
            base, view = LAYOUTS[ln](np.dtype(s))
            refill(base, s, rng)
            for mk in ("all", "row1", "row0", "some"):
                if not _mv_mask_fits(view.shape, mk):
                    continue
                mask = _mv_mask(view.shape, mk, rng)
                tag = f"K/{s}.{mk}"
                axes = _mv_axes(view) if view.ndim <= 2 else [None, 0, view.ndim - 1, -1]
                for ax in axes:
                    axp = {} if ax is None else {"axis": ax}
                    for kd in (False, True):
                        emit("mean", base, view, mask, dict(axp, keepdims=True) if kd else axp, ln, tag)
                    for ddof in (0, 1, 2, 5):
                        emit("var", base, view, mask, dict(axp, ddof=ddof), ln, tag)
                    for ddof in (0, 5):
                        emit("std", base, view, mask, dict(axp, ddof=ddof), ln, tag)
                for ax in (None, 0):
                    axp = {} if ax is None else {"axis": ax}
                    emit("var", base, view, mask, dict(axp, ddof=1, keepdims=True), ln, tag)
                emit("std", base, view, mask, {"keepdims": True}, ln, tag)

    # Section T: dtype= for every input in eight loops; nomask and some.
    for s in MV_DTYPES:
        for shape, ln in (((3, 5), "c_contiguous_2d"), ((7,), "c_contiguous_1d")):
            base = np.ascontiguousarray(_mv_values(s, int(np.prod(shape)), rng).reshape(shape))
            for mk in ("none", "some"):
                mask = _mv_mask(shape, mk, rng)
                for loop in MV_LOOPS:
                    tag = f"T/{s}->{loop}.{mk}"
                    for ax in [None] + list(range(len(shape))):
                        axp = {"dtype": loop} if ax is None else {"axis": ax, "dtype": loop}
                        emit("mean", base, base, mask, axp, ln, tag)
                        emit("var", base, base, mask, axp, ln, tag)
                        emit("anom", base, base, mask, axp, ln, tag)
                    emit("std", base, base, mask, {"dtype": loop, "ddof": 1}, ln, tag)

    # Section V: the mean= forms of var (and of std, which ignores them).
    def mean_forms(view, ax, s):
        """(descriptor, NumPy object, extra operand) for each mean= form: a Python float / int (weak without a mask,
        strong float64 / int64 through ma.subtract's getdata with one), a Python complex where NumPy's weak complex
        stays complex128 (not with float16 / float32), np.float32 / np.float64 scalars (strong), and arrays of the
        keepdims shape — plain, masked, 0-d — carried as a second operand."""
        yield {"kind": "pyfloat", "value": 1.5}, 1.5, None
        yield {"kind": "pyint", "value": 2}, 2, None
        yield {"kind": "npf32", "value": 1.25}, np.float32(1.25), None
        yield {"kind": "npf64", "value": -0.75}, np.float64(-0.75), None
        if s not in ("float16", "float32"):
            yield {"kind": "pycomplex", "real": 1.0, "imag": -2.0}, complex(1.0, -2.0), None
        a0 = np.array(np.float32(0.5))
        yield {"kind": "array"}, a0, _ma_operand(a0, a0, None)
        if view.ndim == 0:
            return
        kshape = tuple(1 if (ax is None or i == ax % view.ndim) else d for i, d in enumerate(view.shape))
        arr = rng.standard_normal(kshape) * 3
        if s == "complex128":
            arr = arr + 1j * rng.standard_normal(kshape)
        arr = np.ascontiguousarray(arr)
        yield {"kind": "array"}, arr, _ma_operand(arr, arr, None)
        am = rng.random(kshape) < 0.4
        yield {"kind": "masked"}, np.ma.array(arr, mask=am), _ma_operand(arr, arr, am)

    for ln in ("c_contiguous_2d", "f_contiguous_2d", "c_contiguous_1d", "scalar_0d"):
        for s in MV_DTYPES:
            base, view = LAYOUTS[ln](np.dtype(s))
            refill(base, s, rng)
            for mk in ("none", "false", "some") + (("all",) if ln == "c_contiguous_2d" else ()):
                mask = _mv_mask(view.shape, mk, rng)
                tag = f"V/{s}.{mk}"
                for ax in ([None] if view.ndim == 0 else [None, 0, -1]):
                    axp = {} if ax is None else {"axis": ax}
                    forms = list(mean_forms(view, ax, s))
                    for form in forms:
                        emit("var", base, view, mask, axp, ln, tag, mean=form)
                    # std ignores mean=: one scalar and the last (keepdims-shaped or 0-d) array form pin that.
                    for form in (forms[0], forms[-1]):
                        emit("std", base, view, mask, dict(axp, ddof=1), ln, tag, mean=form)
            if ln == "c_contiguous_2d":
                # A mean of the reduced shape without keepdims does not broadcast back: NumPy's subtract raises.
                bad = np.ascontiguousarray(rng.standard_normal(view.shape[:1]))
                for mk in ("none", "some"):
                    mask = _mv_mask(view.shape, mk, rng)
                    emit("var", base, view, mask, {"axis": 1}, ln, f"V/{s}.{mk}",
                         mean=({"kind": "array"}, bad, _ma_operand(bad, bad, None)))

    # Section S: special values.
    special_sets = [(s, sp) for sp in ("nonfinite", "huge", "negzero") for s in ("float16", "float32", "float64",
                                                                                  "complex128")]
    special_sets += [(s, "extreme") for s in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64",
                                             "uint64")]
    for s, sp in special_sets:
        for ln in ("c_contiguous_2d", "f_contiguous_2d", "c_contiguous_1d", "scalar_0d", "one_element_1d"):
            base, view = LAYOUTS[ln](np.dtype(s))
            refill(base, s, rng, sp)
            for mk in ("none", "false", "some"):
                mask = _mv_mask(view.shape, mk, rng)
                tag = f"S/{s}.{sp}.{mk}"
                for ax in ([None] if view.ndim == 0 else [None, 0, -1]):
                    axp = {} if ax is None else {"axis": ax}
                    emit("mean", base, view, mask, axp, ln, tag)
                    emit("var", base, view, mask, axp, ln, tag)
                    emit("std", base, view, mask, axp, ln, tag)
                    emit("anom", base, view, mask, axp, ln, tag)
                emit("var", base, view, mask, {"ddof": 1}, ln, tag)

    # Section M: sizes where the pairwise sums show their order: 130 elements (past NumPy's 128-element pairwise block),
    # 1031 (several levels of the recursion; float64 only, the operand bytes are large) and a 19x13 block (flat past the
    # block; rows of 13 and columns of 19 in the 8-accumulator loop) in C, F and a permuted 3-D layout.
    def big(s, size):
        """`size` values whose sums cancel: near 1000 (float16 near 8), complex with a large imaginary offset, int64
        near +-2**52 (their float64 sums round)."""
        if s == "int64":
            return rng.integers(-2 ** 52, 2 ** 52, size).astype(s)
        vals = 1000.0 + rng.normal(0.0, 3.0, size) if s != "float16" else 8.0 + rng.normal(0.0, 1.5, size)
        if s == "complex128":
            vals = vals + 1j * (-250.0 + rng.normal(0.0, 7.0, size))
        return np.ascontiguousarray(np.asarray(vals).astype(s))

    for s in ("float16", "float32", "float64", "complex128", "int64"):
        for size in ((130, 1031) if s == "float64" else (130,)):
            base = big(s, size)
            views = [("c_contiguous_1d", base), ("negstride_1d", base[::-1])]
            if size == 130:
                views.append(("strided_step2_1d", base[::2]))
            for ln, view in views:
                for mk in ("none", "some"):
                    mask = _mv_mask(view.shape, mk, rng)
                    tag = f"M/{s}/{size}.{mk}"
                    emit("mean", base, view, mask, {}, ln, tag)
                    emit("var", base, view, mask, {}, ln, tag)
                    emit("var", base, view, mask, {"axis": 0, "ddof": 1}, ln, tag)
                    emit("std", base, view, mask, {}, ln, tag)
                    emit("anom", base, view, mask, {}, ln, tag)
        base = big(s, 19 * 13)
        for ln, view in (("c_contiguous_2d", base.reshape(19, 13)), ("f_contiguous_2d", base.reshape(13, 19).T),
                         ("transposed_3d", base[: 18 * 13].reshape(3, 6, 13).transpose(2, 0, 1))):
            for mk in ("none", "some"):
                mask = _mv_mask(view.shape, mk, rng)
                tag = f"M/{s}/block.{mk}"
                for ax in [None] + list(range(view.ndim)):
                    axp = {} if ax is None else {"axis": ax}
                    emit("mean", base, view, mask, axp, ln, tag)
                    emit("var", base, view, mask, axp, ln, tag)
                emit("std", base, view, mask, {"axis": 0}, ln, tag)
                emit("anom", base, view, mask, {"axis": view.ndim - 1}, ln, tag)

    # Section E: empty inputs (count 0: masked results, nan, NumPy's own error texts).
    for s in ("bool", "int32", "uint8", "float16", "float32", "float64", "complex128"):
        for shape in ((0,), (0, 3), (3, 0)):
            base = np.zeros(shape, dtype=s)
            for mk in ("none", "false", "all"):
                mask = _mv_mask(shape, mk, rng)
                tag = f"E/{s}.{mk}"
                for ax in [None] + list(range(len(shape))):
                    axp = {} if ax is None else {"axis": ax}
                    emit("mean", base, base, mask, axp, "empty", tag)
                    emit("mean", base, base, mask, dict(axp, keepdims=True), "empty", tag)
                    emit("var", base, base, mask, axp, "empty", tag)
                    emit("var", base, base, mask, dict(axp, ddof=1), "empty", tag)
                    emit("std", base, base, mask, axp, "empty", tag)
                    emit("anom", base, base, mask, axp, "empty", tag)

    # Section Z: 0-d inputs with an axis (the nomask path normalizes it against rank 0; the masked path's ufunc
    # reduction takes 0 / -1), and out-of-range axes.
    for s in ("bool", "int32", "float16", "float32", "float64", "complex128"):
        base = np.array(_mv_values(s, 1, rng)[0])
        for mk in ("none", "false", "some", "all"):
            mask = _mv_mask((), mk, rng)
            tag = f"Z/{s}.{mk}"
            for ax in (0, -1, 1):
                for op in ("mean", "var", "std", "anom"):
                    emit(op, base, base, mask, {"axis": ax}, "scalar_0d", tag)
        b2 = np.ascontiguousarray(_mv_values(s, 6, rng).reshape(2, 3))
        for mk in ("none", "some"):
            mask = _mv_mask((2, 3), mk, rng)
            for ax in (2, -3):
                for op in ("mean", "var", "std", "anom"):
                    emit(op, b2, b2, mask, {"axis": ax}, "c_contiguous_2d", f"Z/{s}.{mk}")

    print(f"  (ma_meanvar carved: {carved})")
    return cases


# ---------------------------------------------------------------------------------------------
# Tier `setops` (ma_setops.jsonl): numpy.ma's arraysetops as NumPy's statements — unique (with return_index /
# return_inverse), in1d, isin, intersect1d, setxor1d, union1d, setdiff1d — over every dtype, value pool, mask class,
# hard masks, plain operands, layouts and mixed dtypes. Held to ZERO excuses (valueclass "masetops").
#
# NumPy's 'quicksort' is unstable (host-dependent among equal keys) and NumSharp's sort is stable, so a case enters
# the corpus only when NumPy's observable answer (filled(0) data, mask, dtype, shape, form) equals the stable
# reference's (ma_setops_reference.py: NumPy's statements with stable sorts). Every dropped case must be one the
# reference marked SENSITIVE (an unstable sort met a group of equal keys whose members are not interchangeable); a
# dropped case that is not would mean the reference is not NumPy's statements, and the generator stops.
# ---------------------------------------------------------------------------------------------
SO_DTYPES = ["bool", "uint8", "int8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
             "float16", "float32", "float64", "complex128"]
SO_PAIRS = [("int32", "float64"), ("bool", "int8"), ("uint8", "int8"), ("float16", "float64"),
            ("int64", "complex128"), ("uint64", "int64"), ("float32", "int16"), ("int16", "uint16")]
SO_SIZES = [0, 1, 2, 3, 5, 8, 13, 17, 24, 40]


def _so_values(dt, n, rng, pool):
    """N values of DT from POOL: 'small' (few distinct, many duplicates), 'special' (NaN / ±inf / ±0 / extremes),
    'wide' (many distinct)."""
    d = np.dtype(dt)
    if d.kind == "b":
        return rng.integers(0, 2, n).astype(bool)
    if pool == "small":
        vals = rng.integers(0, 4, n)
        if d.kind == "c":
            return (vals + 1j * rng.integers(0, 2, n)).astype(d)
        return vals.astype(d)
    if pool == "wide":
        if d.kind in "ui":
            info = np.iinfo(d)
            lo, hi = max(int(info.min), -10 ** 6), min(int(info.max), 10 ** 6)
            return rng.integers(lo, hi + 1, n).astype(d)
        if d.kind == "c":
            return (rng.integers(-50, 50, n) + 1j * rng.integers(-3, 3, n)).astype(d)
        return rng.integers(-50, 50, n).astype(d)
    # special
    if d.kind in "ui":
        info = np.iinfo(d)
        pick = [0, 1, 2, int(info.max), int(info.min), int(info.max) - 1]
        return np.array([pick[i] for i in rng.integers(0, len(pick), n)], dtype=d)
    if d.kind == "f":
        pick = [0.0, -0.0, 1.0, 2.0, -1.0, np.nan, np.inf, -np.inf]
        return np.array([pick[i] for i in rng.integers(0, len(pick), n)], dtype=d)
    pick = [0, 1, 1j, -0.0, complex(np.nan, 0), complex(0, np.nan), complex(np.inf, np.inf), complex(1, np.inf)]
    return np.array([pick[i] for i in rng.integers(0, len(pick), n)], dtype=d)


def _so_mask(shape, kind, rng):
    """A mask of KIND for SHAPE: 'none' (nomask), 'false' (an all-False array), 'some' (random), 'all'."""
    if kind == "none":
        return None
    if kind == "false":
        return np.zeros(shape, dtype=bool)
    if kind == "all":
        return np.ones(shape, dtype=bool)
    return rng.random(shape) < 0.35 if len(shape) else np.array(bool(rng.random() < 0.35))


def _so_layout(vals, layout):
    """(base, view) holding VALS in LAYOUT: c1d / strided / reversed / c2d / t2d / 0d."""
    n = vals.shape[0]
    if layout == "strided":
        base = np.empty(2 * n, dtype=vals.dtype)
        base[::2] = vals
        base[1::2] = vals[::-1] if n else vals
        return base, base[::2]
    if layout == "reversed":
        base = vals[::-1].copy()
        return base, base[::-1]
    if layout in ("c2d", "t2d"):
        r = 2 if n % 2 == 0 else 1
        if layout == "c2d":
            base = vals.copy()
            return base, base.reshape(r, n // r)
        base = vals.reshape(n // r, r).T.copy().reshape(-1)
        return base, base.reshape(r, n // r).T
    if layout == "0d":
        base = np.array(vals[0] if n else vals.dtype.type(0))
        return base, base
    base = vals.copy()
    return base, base


def _so_form(r):
    """The result's object kind: masked (the singleton), scalar (0-d, no mask), nomask, or mask (a mask array)."""
    if r is np.ma.masked:
        return "masked"
    if isinstance(r, np.ma.MaskedArray):
        if r._mask is np.ma.nomask:
            return "scalar" if r.ndim == 0 else "nomask"
        return "mask"
    return "scalar"


def _so_expected(r, raw):
    """The masked expectation plus the form; RAW adds the data at every slot (deterministic only for in1d / isin). A
    plain ndarray (a tuple's index slot) records no form: the C# side carries it in a nomask masked array."""
    e = _ma_expected(r)
    if isinstance(r, np.ma.MaskedArray) or not isinstance(r, np.ndarray):
        e["form"] = _so_form(r)
    if raw:
        d = np.asarray(r.data) if isinstance(r, np.ma.MaskedArray) else np.asarray(r)
        e["raw"] = np.ascontiguousarray(d).tobytes().hex()
    return e


def _so_observe(r, raw):
    """What the corpus compares of a result (a tuple slot by slot), for the NumPy-vs-reference agreement check: the raw
    data too when the case records it (in1d / isin), since a tie order can move it while filled(0) and the mask agree."""
    if isinstance(r, tuple):
        return tuple(_so_observe(x, False) for x in r)
    e = _so_expected(r, raw=raw)
    return (e["dtype"], tuple(e["shape"]), e["buffer"], e["mask"], e.get("form"), e.get("raw"))


def gen_ma_setops():
    import ma_setops_reference as R

    cases, n = [], _N()
    ma = np.ma
    stats = {"kept": 0, "tie_dropped": 0}
    rng = np.random.default_rng(20261008)

    # op key -> (NumPy callable, reference callable, params, raw?)
    def op_table():
        t = []
        for ri, rx in [(False, False), (True, False), (False, True), (True, True)]:
            p = {}
            if ri:
                p["return_index"] = True
            if rx:
                p["return_inverse"] = True
            t.append(("unique", (lambda ri=ri, rx=rx: (lambda a, b: ma.unique(a, return_index=ri, return_inverse=rx)))(),
                      (lambda ri=ri, rx=rx: (lambda a, b: R.unique(a, return_index=ri, return_inverse=rx)))(), p, False, 1))
        for au in (False, True):
            for inv in (False, True):
                p = {}
                if au:
                    p["assume_unique"] = True
                if inv:
                    p["invert"] = True
                t.append(("in1d", (lambda au=au, inv=inv: (lambda a, b: ma.in1d(a, b, assume_unique=au, invert=inv)))(),
                          (lambda au=au, inv=inv: (lambda a, b: R.in1d(a, b, assume_unique=au, invert=inv)))(), p, True, 2))
                t.append(("isin", (lambda au=au, inv=inv: (lambda a, b: ma.isin(a, b, assume_unique=au, invert=inv)))(),
                          (lambda au=au, inv=inv: (lambda a, b: R.isin(a, b, assume_unique=au, invert=inv)))(), p, True, 2))
        for au in (False, True):
            p = {"assume_unique": True} if au else {}
            t.append(("intersect1d", (lambda au=au: (lambda a, b: ma.intersect1d(a, b, assume_unique=au)))(),
                      (lambda au=au: (lambda a, b: R.intersect1d(a, b, assume_unique=au)))(), p, False, 2))
            t.append(("setxor1d", (lambda au=au: (lambda a, b: ma.setxor1d(a, b, assume_unique=au)))(),
                      (lambda au=au: (lambda a, b: R.setxor1d(a, b, assume_unique=au)))(), p, False, 2))
            t.append(("setdiff1d", (lambda au=au: (lambda a, b: ma.setdiff1d(a, b, assume_unique=au)))(),
                      (lambda au=au: (lambda a, b: R.setdiff1d(a, b, assume_unique=au)))(), p, False, 2))
        t.append(("union1d", lambda a, b: ma.union1d(a, b), lambda a, b: R.union1d(a, b), {}, False, 2))
        return t

    ops = op_table()

    def build(base, view, mask, plain, hard):
        """The NumPy operand: the plain view, or a masked array over it (hardened when asked)."""
        if plain:
            return view
        m = np.ma.array(view, mask=(np.ma.nomask if mask is None else mask))
        if hard:
            m.harden_mask()
        return m

    def emit(opname, fnp, fref, params, raw, nops, operands_spec, tag):
        """Runs NumPy and the reference on fresh operands; records the case when their observable answers agree."""
        def fresh():
            return [build(*spec) for spec in operands_spec]
        a = fresh()
        try:
            want = fnp(a[0], a[1] if nops > 1 else None)
            want_err = None
        except Exception as e:  # noqa: BLE001 — NumPy's error is an expectation
            want, want_err = None, e
        a = fresh()
        got, sensitive = R.run(fref, a[0], a[1] if nops > 1 else None)
        if want_err is not None:
            same = isinstance(got, Exception) and type(got) is type(want_err) and str(got) == str(want_err)
        else:
            same = not isinstance(got, Exception) and _so_observe(want, raw) == _so_observe(got, raw)
        if not same:
            if not sensitive:
                raise AssertionError(f"setops: NumPy and the stable reference disagree without a sensitive tie: "
                                     f"{opname} {params} {tag}\n  numpy: {want if want_err is None else want_err!r}"
                                     f"\n  ref:   {got!r}")
            stats["tie_dropped"] += 1
            return
        p = dict(params)
        plain = [i for i, s in enumerate(operands_spec[:nops]) if s[3]]
        hard = [i for i, s in enumerate(operands_spec[:nops]) if s[4]]
        if plain:
            p["plain"] = plain
        if hard:
            p["hard"] = hard
        operands = [_ma_operand(s[0], s[1], None if s[3] else s[2]) for s in operands_spec[:nops]]
        cid = f"ma.{opname}/{tag}/{n.next()}"
        stats["kept"] += 1
        if want_err is not None:
            cases.append({"id": cid, "op": f"ma.{opname}", "params": p, "operands": operands, "expected": {},
                          "expects_throw": True, "error": {"type": type(want_err).__name__, "text": str(want_err)},
                          "layout": tag, "valueclass": "masetops"})
            return
        if isinstance(want, tuple):
            expected = {"kind": "masked_tuple", "slots": [_so_expected(x, raw=False) for x in want]}
        else:
            expected = _so_expected(want, raw)
        cases.append({"id": cid, "op": f"ma.{opname}", "params": p, "operands": operands, "expected": expected,
                      "layout": tag, "valueclass": "masetops"})

    def spec(vals, layout, mask_kind, plain=False, hard=False):
        base, view = _so_layout(vals, layout)
        mask = None if plain else _so_mask(view.shape, mask_kind, rng)
        return (base, view, mask, plain, hard)

    # Section H: NumPy's own TestArraySetOps inputs (numpy/ma/tests/test_extras.py).
    def mk(data, mask=None, dt="int64"):
        v = np.array(data, dtype=dt)
        return (v, v, None if mask is None else np.array(mask, dtype=bool), mask is None, False)
    hand = [
        ("unique", [mk([1, 1, 1, 2, 2, 3], [0, 0, 1, 0, 1, 0])]),
        ("unique", [mk([1, 1, 1], [1, 1, 1])]),
        ("intersect1d", [mk([1, 3, 3, 3], [0, 0, 0, 1]), mk([3, 1, 1, 1], [0, 0, 0, 1])]),
        ("setxor1d", [mk([1, 2, 5, 7, -1], [0, 0, 0, 0, 1]), mk([1, 2, 3, 4, 5, -1], [0, 0, 0, 0, 0, 1])]),
        ("setxor1d", [mk([1, 2, 5, 7, -1], [0, 0, 0, 0, 1]), mk([1, 2, 3, 4, 5])]),
        ("setxor1d", [mk([1, 8, 2, 3], [0, 1, 0, 0]), mk([6, 5, 4, 8], [0, 0, 0, 1])]),
        ("in1d", [mk([1, 2, 5, 7, -1], [0, 0, 0, 0, 1]), mk([1, 2, 3, 4, 5, -1], [0, 0, 0, 0, 0, 1])]),
        ("in1d", [mk([5, 5, 2, 1, -1], [0, 0, 0, 0, 1]), mk([1, 5, -1], [0, 0, 1])]),
        ("in1d", [mk([3, 5], [0, 0]), mk([3, 9], [0, 1])]),
        ("union1d", [mk([1, 2, 5, 7, 5, -1], [0, 0, 0, 0, 0, 1]), mk([1, 2, 3, 4, 5, -1], [0, 0, 0, 0, 0, 1])]),
        ("setdiff1d", [mk([1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 1, 0, 0, 1]), mk([2, 4, 3, 3, 2, 1, 5], [0, 0, 0, 0, 1, 0, 0])]),
    ]
    for opname, specs in hand:
        for (o, fnp, fref, params, raw, nops) in ops:
            if o == opname and not params:
                emit(o, fnp, fref, params, raw, nops, specs + [specs[0]] * (2 - len(specs)), "numpy_tests")
    # isin over NumPy's 3-D test input (a 2x3x4 element with one masked slot).
    a3 = np.arange(24, dtype=np.int64)
    m3 = np.zeros(24, dtype=bool)
    m3[1 * 12 + 2 * 4 + 0] = True
    b9 = np.array([0, 10, 20, 30, 1, 3, 11, 22, 33], dtype=np.int64)
    s3 = (a3, a3.reshape(2, 3, 4), m3.reshape(2, 3, 4), False, False)
    s9 = (b9, b9, np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool), False, False)
    for (o, fnp, fref, params, raw, nops) in ops:
        if o in ("isin", "in1d") and not params:
            emit(o, fnp, fref, params, raw, nops, [s3, s9], "numpy_tests_3d")

    # Section R: random operands, every dtype x op.
    layouts0 = ["c1d", "c1d", "strided", "reversed", "c2d", "t2d", "0d"]
    for dt in SO_DTYPES:
        for (opname, fnp, fref, params, raw, nops) in ops:
            for draw in range(26):
                n1 = int(rng.choice(SO_SIZES))
                n2 = int(rng.choice(SO_SIZES))
                pool1 = str(rng.choice(["small", "small", "special", "wide"]))
                pool2 = str(rng.choice(["small", "special", "wide"]))
                lay = str(rng.choice(layouts0))
                if lay == "0d":
                    n1 = 1
                v1 = _so_values(dt, n1, rng, pool1)
                v2 = _so_values(dt, n2, rng, pool2)
                m1 = str(rng.choice(["none", "false", "some", "some", "all"]))
                m2 = str(rng.choice(["none", "false", "some", "all"]))
                plain1 = bool(rng.random() < 0.12)
                plain2 = bool(rng.random() < 0.25)
                hard1 = bool(rng.random() < 0.15) and not plain1
                hard2 = bool(rng.random() < 0.15) and not plain2
                s1 = spec(v1, lay, m1, plain1, hard1)
                s2 = spec(v2, str(rng.choice(["c1d", "c1d", "strided", "reversed"])), m2, plain2, hard2)
                emit(opname, fnp, fref, params, raw, nops, [s1, s2], f"{dt}.{lay}.{pool1}.{m1}")

    # Section P: mixed dtypes (the concatenation promotes).
    for (d1, d2) in SO_PAIRS:
        for (opname, fnp, fref, params, raw, nops) in ops:
            if nops < 2:
                continue
            for draw in range(6):
                n1 = int(rng.choice(SO_SIZES[1:]))
                n2 = int(rng.choice(SO_SIZES[1:]))
                v1 = _so_values(d1, n1, rng, "small")
                v2 = _so_values(d2, n2, rng, str(rng.choice(["small", "special"])))
                s1 = spec(v1, "c1d", str(rng.choice(["none", "some"])))
                s2 = spec(v2, "c1d", str(rng.choice(["none", "some", "all"])))
                emit(opname, fnp, fref, params, raw, nops, [s1, s2], f"{d1}+{d2}")

    # Section B: large operands (4,500 / 600 elements) of the integer dtypes, past the size from which NumSharp argsorts a
    # masked integer array as "the unmasked values, then the masked slots" (MaxKeyPartitionArgsort, 4,096 elements); the
    # 'special' pool puts the dtype maximum among the unmasked values, which sends NumSharp back to the general path.
    for dt in ["uint8", "int16", "uint16", "int32", "int64", "uint64"]:
        for pool in ["wide", "special"]:
            for (opname, fnp, fref, params, raw, nops) in ops:
                if opname not in ("unique", "in1d", "intersect1d", "setxor1d", "union1d") or params.get("invert") \
                        or params.get("assume_unique"):
                    continue
                v1 = _so_values(dt, 4500, rng, pool)
                v2 = _so_values(dt, 600, rng, pool)
                s1 = spec(v1, "c1d", "some")
                s2 = spec(v2, "c1d", "some")
                emit(opname, fnp, fref, params, raw, nops, [s1, s2], f"big.{dt}.{pool}")

    print(f"  (ma_setops kept {stats['kept']}, dropped {stats['tie_dropped']} that depend on NumPy's quicksort tie order)")
    return cases


# ---------------------------------------------------------------------------------------------
# ma.put as numpy.ma's statements (2026-10-08): MaskedArray.put / ma.put over every dtype, target layout (1-D, strided,
# reversed, 2-D, F-order, 3-D, (1, n), (n, 1), 0-d, empty), mask kind (nomask / all-False / some / all), hard and soft
# masks, plain ndarray targets, index kinds (int64 / int32 / int8 / uint64 / bool / float / 2-D / 0-d / empty, in range,
# negative, out of bounds), value kinds (Python scalars, Python lists, ndarrays NumPy owns or views — C, F, strided,
# 0-d, (k, 1) —, masked arrays, the masked constant) and modes (raise / wrap / clip / a bad one). NumPy's errors are
# cases (type + text). Under a hard mask NumPy runs mask = self._mask[indices]; values.resize(indices.shape);
# indices[~mask]: an N-D target always raises IndexError, short values are zero-padded, masked values raise
# ValueError, an ndarray of another size raises ndarray.resize's errors.
#
# Params (OpRegistry.Ma "put"): mode; hard (harden the target); plain (the target is the plain data view); vkind —
# "array" (the ndarray operand; vown: NumPy owned it, so the replay copies the reconstructed view), "list" (a Python
# list: the operand holds np.array(list)), "scalar" (value + vtype in params), "masked" (the masked-array operand),
# "constant" (ma.masked).
# ---------------------------------------------------------------------------------------------
PUT_LAYOUTS = ["c1d", "c1d", "strided", "reversed", "c2d", "t2d", "c3d", "row", "col", "0d", "empty", "empty2d"]


def _put_target(vals, layout):
    """(base, view) of a put target holding VALS (reshaped as needed) in LAYOUT; the base is always C-contiguous
    (describe() serializes a base in C order)."""
    if layout in ("c1d", "strided", "reversed", "c2d", "t2d", "0d"):
        return _so_layout(vals, layout)
    if layout == "c3d":
        base = vals[:8].copy()
        return base, base.reshape(2, 2, 2)
    if layout == "row":
        base = vals.copy()
        return base, base.reshape(1, -1)
    if layout == "col":
        base = vals.copy()
        return base, base.reshape(-1, 1)
    if layout == "empty":
        base = vals[:0].copy()
        return base, base
    base = np.zeros((0, 3), dtype=vals.dtype)          # empty2d
    return base, base


def _put_scalar_param(v):
    """A Python scalar as JSON params (value + vtype); NaN / inf as strings, complex as [re, im]."""
    def num(x):
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return "nan" if np.isnan(x) else ("inf" if x > 0 else "-inf")
        return x
    if isinstance(v, bool):
        return {"vtype": "bool", "value": v}
    if isinstance(v, int):
        return {"vtype": "int", "value": v}
    if isinstance(v, float):
        return {"vtype": "float", "value": num(v)}
    return {"vtype": "complex", "value": [num(v.real), num(v.imag)]}


def _put_array_values(src, layout):
    """An ndarray of SRC's values in LAYOUT, as (numpy object, C base, view describing it, owndata). The (base, view)
    pair describes the same logical values and layout flags with a C-contiguous base."""
    k = src.shape[0]
    if layout == "f" and k >= 2 and k % 2 == 0:
        arr = np.asfortranarray(src.reshape(2, k // 2))
        cbuf = np.ascontiguousarray(arr.T)
        return arr, cbuf, cbuf.T, True
    if layout == "strided":
        big = np.empty(2 * k, dtype=src.dtype)
        big[::2] = src
        big[1::2] = src
        return big[::2], big, big[::2], False
    if layout == "view":
        big = np.concatenate([src, src])
        return big[:k], big, big[:k], False
    if layout == "col":
        arr = src.reshape(k, 1).copy()
        return arr, arr, arr, True
    if layout == "0d":
        arr = np.array(src[0])
        return arr, arr, arr, True
    arr = src.copy()
    return arr, arr, arr, True


def gen_ma_put():
    cases, n = [], _N()
    ma = np.ma
    rng = np.random.default_rng(20261008 + 7)
    stats = {"kept": 0, "errors": 0}

    def emit(tag, tvals, layout, mask, kind, hard, idx, values, mode):
        """Describes the operands, then runs NumPy on fresh objects and records the mutated target or NumPy's error.
        kind: 'masked' / 'plain'; values: ('array', (obj, base, view, own)) / ('list', list) / ('scalar', v) /
        ('masked', MaskedArray) / ('constant', None)."""
        base, view = _put_target(tvals, layout)
        if mask is not None and mask.shape != view.shape:
            mask = _so_mask(view.shape, "some", rng)
        p = {"mode": mode}
        if kind == "plain":
            p["plain"] = True
        if hard:
            p["hard"] = True
        vk, vv = values
        p["vkind"] = vk
        operands = [_ma_operand(base, view, None if kind == "plain" else mask)]
        # np.array(idx) keeps a 0-d index 0-d (np.ascontiguousarray would return it as shape (1,)).
        ib = np.array(idx)
        operands.append(_ma_operand(ib, ib, None))
        if vk == "array":
            vobj, vbase, vview, vown = vv
            operands.append(_ma_operand(vbase, vview, None))
            p["vown"] = bool(vown)
        elif vk == "list":
            la = np.array(vv)
            operands.append(_ma_operand(la, la, None))
        elif vk == "masked":
            d = np.ascontiguousarray(vv.data)
            vm = ma.getmask(vv)
            operands.append(_ma_operand(d, d, None if vm is ma.nomask else vm))
        elif vk == "scalar":
            p.update(_put_scalar_param(vv))

        # NumPy runs on fresh objects: a copy of the target, and of every values object NumPy may resize.
        b2, v2 = _put_target(tvals.copy(), layout)
        if kind == "plain":
            target = v2
        else:
            target = ma.array(v2, mask=(ma.nomask if mask is None else mask.copy()))
            if hard:
                target.harden_mask()
        if vk == "array":
            vobj = vv[0]
            vnp = vobj.copy(order="K") if vv[3] else vobj
        elif vk == "masked":
            vm = ma.getmask(vv)
            vnp = ma.array(vv.data.copy(), mask=(ma.nomask if vm is ma.nomask else vm.copy()))
        elif vk == "constant":
            vnp = ma.masked
        else:
            vnp = vv
        try:
            if kind == "plain":
                ma.put(target, idx, vnp, mode=mode)
            else:
                target.put(idx, vnp, mode=mode)
            err = None
        except Exception as e:  # noqa: BLE001 — NumPy's error is an expectation
            err = e

        cid = f"ma.put/{tag}/{n.next()}"
        stats["kept"] += 1
        if err is not None:
            stats["errors"] += 1
            cases.append({"id": cid, "op": "ma.put", "params": p, "operands": operands, "expected": {},
                          "expects_throw": True, "error": {"type": type(err).__name__, "text": str(err)},
                          "layout": layout, "valueclass": "maput"})
            return
        expected = _plain_expected(target) if kind == "plain" else _so_expected(target, raw=True)
        cases.append({"id": cid, "op": "ma.put", "params": p, "operands": operands, "expected": expected,
                      "layout": layout, "valueclass": "maput"})

    def i64(*v):
        return np.array(v, dtype=np.int64)

    def arr(a, layout="c"):
        a = np.asarray(a)
        return ("array", _put_array_values(a.ravel(), layout) if a.ndim <= 1 else (a, a, a, True))

    # Section H: the rundown's cases and the probes behind the port (numpy 2.4.2 outputs, every one).
    t = np.array([1, 2, 3, 4], dtype=np.int64)
    m1 = np.array([False, True, False, False])
    m0 = np.array([True, False, False, False])
    t6 = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    m6 = np.array([[False, True, False], [False, False, True]])
    hand = [
        # 1-D: zero-padded / truncated values, index forms, modes
        ("short", t, "c1d", m1, i64(0, 1, 2, 3), ("list", [10, 20]), "raise"),
        ("scalar", t, "c1d", m1, i64(0, 2, 3), ("scalar", 5), "raise"),
        ("long", t, "c1d", m1, i64(0, 2), ("list", [10, 20, 30, 40]), "raise"),
        ("idx0d", t, "c1d", m1, np.array(0), ("scalar", 5), "raise"),
        ("idx0d.masked", t, "c1d", m1, np.array(1), ("scalar", 5), "raise"),
        ("neg", t, "c1d", m1, i64(-1, -3), ("list", [7, 8]), "raise"),
        ("oob.raise", t, "c1d", m1, i64(0, 9), ("list", [7, 8]), "raise"),
        ("oob.wrap", t, "c1d", m1, i64(0, 9), ("list", [7, 8]), "wrap"),
        ("oob.clip", t, "c1d", m1, i64(0, 9), ("list", [7, 8]), "clip"),
        ("clip.neg", t, "c1d", m0, i64(-1), ("list", [7]), "clip"),
        ("clip.neg4", t, "c1d", m0, i64(-4), ("list", [7]), "clip"),
        ("wrap.neg", t, "c1d", m0, i64(-1), ("list", [7]), "wrap"),
        ("badmode", t, "c1d", m1, i64(0), ("list", [7]), "bogus"),
        ("badmode.oob", t, "c1d", m1, i64(9), ("list", [7]), "bogus"),
        ("badmode.Raise", t, "c1d", m1, i64(0), ("list", [7]), "Raise"),
        ("2didx", t, "c1d", m1, np.array([[0, 2], [3, 1]], dtype=np.int64), ("list", [7, 8]), "raise"),
        ("boolidx", t, "c1d", m1, np.array([True, False, True, False]), ("list", [7, 8]), "raise"),
        ("boolidx.all", t, "c1d", m0, np.array([True, True, True, True]), ("list", [7, 8, 9, 10]), "raise"),
        ("floatidx", t, "c1d", m1, np.array([0.0, 2.0]), ("list", [7, 8]), "raise"),
        ("emptyidx", t, "c1d", m1, np.array([], dtype=np.int64), ("list", [7]), "raise"),
        ("emptyfloatidx", t, "c1d", m1, np.array([], dtype=np.float64), ("list", [7]), "raise"),
        ("u64idx", t, "c1d", m1, np.array([0, 2], dtype=np.uint64), ("list", [7, 8]), "raise"),
        ("emptyvals", t, "c1d", m1, i64(1, 2), ("list", []), "raise"),
        # ndarray values (the caller holds them)
        ("arr.eq", t, "c1d", m1, i64(0, 2), arr(i64(7, 8)), "raise"),
        ("arr.short", t, "c1d", m1, i64(0, 2, 3), arr(i64(7, 8)), "raise"),
        ("arr.col", t, "c1d", m1, i64(0, 2), arr(np.array([7, 8], dtype=np.int64), "col"), "raise"),
        ("arr.float", t, "c1d", m1, i64(0, 2), arr(np.array([7.5, 8.5])), "raise"),
        ("arr.0d", t, "c1d", m1, i64(0, 2), arr(np.array([5]), "0d"), "raise"),
        ("arr.0d.one", t, "c1d", m1, i64(0), arr(np.array([5]), "0d"), "raise"),
        ("arr.F", t, "c1d", np.zeros(4, bool), i64(0, 1, 2, 3), arr(np.array([10, 12, 11, 13]), "f"), "raise"),
        ("arr.strided", t, "c1d", np.zeros(4, bool), i64(0, 1), arr(np.arange(2), "strided"), "raise"),
        ("arr.view", t, "c1d", np.zeros(4, bool), i64(0, 1, 2), arr(np.arange(2), "view"), "raise"),
        ("arr.view.eq", t, "c1d", np.zeros(4, bool), i64(0, 1), arr(np.arange(2), "view"), "raise"),
        # masked values
        ("masked", t, "c1d", m1, i64(0, 2), ("masked", ma.array([7, 8], mask=[0, 1])), "raise"),
        ("masked.nomask", t, "c1d", m1, i64(0, 2), ("masked", ma.array([7, 8])), "raise"),
        ("masked.cycle", t, "c1d", m1, i64(0, 1, 2, 3), ("masked", ma.array([7, 8], mask=[1, 0])), "raise"),
        ("constant", t, "c1d", m1, i64(0, 2), ("constant", None), "raise"),
        ("allfalse.list", t, "c1d", np.zeros(4, bool), i64(0, 2), ("list", [7, 8]), "raise"),
        ("allfalse.constant", t, "c1d", np.zeros(4, bool), i64(0, 2), ("constant", None), "raise"),
        # N-D: under a hard mask always IndexError
        ("2d.oob", t6, "c2d", m6, i64(0, 4), ("list", [7, 8]), "raise"),
        ("2d.rows", t6, "c2d", m6, i64(0, 1), ("list", [7, 8]), "raise"),
        ("2d.row", t6, "c2d", m6, i64(1), ("list", [7]), "raise"),
        ("2d.idx0d", t6, "c2d", m6, np.array(0), ("scalar", 7), "raise"),
        ("2d.idx0d.oob", t6, "c2d", m6, np.array(4), ("scalar", 7), "raise"),
        ("2d.neg", t6, "c2d", m6, i64(-1), ("list", [7]), "raise"),
        ("2d.wrap", t6, "c2d", m6, i64(0, 1), ("list", [7, 8]), "wrap"),
        ("2d.wrap.oob", t6, "c2d", m6, i64(0, 9), ("list", [7, 8]), "wrap"),
        ("2d.2didx", t6, "c2d", m6, np.array([[0], [1]], dtype=np.int64), ("list", [7, 8]), "raise"),
        ("2d.floatidx", t6, "c2d", m6, np.array([0.0]), ("list", [7]), "raise"),
        ("2d.masked", t6, "c2d", m6, i64(0, 1), ("masked", ma.array([7, 8], mask=[0, 1])), "raise"),
        ("2d.arr.short", t6, "c2d", m6, i64(0, 1), arr(i64(7, 8, 9)), "raise"),
        ("2d.arr.col", t6, "c2d", m6, i64(0, 1), arr(np.array([7, 8], dtype=np.int64), "col"), "raise"),
        ("2d.allfalse", t6, "c2d", np.zeros((2, 3), bool), i64(0, 1), ("list", [7, 8]), "raise"),
        ("2d.soft.short", t6, "c2d", m6, i64(0, 2, 4), ("list", [7, 8]), "raise"),
        ("2d.soft.constant", t6, "c2d", m6, i64(1, 4), ("constant", None), "raise"),
    ]
    # Every hand case under a hard mask, a soft mask, a nomask hard array and as a plain target.
    for tag, tv, lay, mk, idx, vals, mode in hand:
        emit("h." + tag, tv, lay, mk, "masked", True, idx, vals, mode)
        emit("s." + tag, tv, lay, mk, "masked", False, idx, vals, mode)
        emit("n." + tag, tv, lay, None, "masked", True, idx, vals, mode)
        emit("p." + tag, tv, lay, None, "plain", False, idx, vals, mode)
    # 3-D, (1, n), (n, 1), 0-d and empty targets.
    t8 = np.arange(8, dtype=np.int64)
    m8 = (np.arange(8) % 3 == 0).reshape(2, 2, 2)
    for hard in (True, False):
        emit(f"3d.rows.{hard}", t8, "c3d", m8, "masked", hard, i64(0, 1), ("list", [7, 8]), "raise")
        emit(f"3d.oob.{hard}", t8, "c3d", m8, "masked", hard, i64(0, 5), ("list", [7, 8]), "raise")
        emit(f"row.{hard}", t, "row", np.array([[False, True, False, False]]), "masked", hard, i64(0), ("list", [7]), "raise")
        emit(f"row.oob.{hard}", t, "row", np.array([[False, True, False, False]]), "masked", hard, i64(2), ("list", [7]), "raise")
        emit(f"col.{hard}", t, "col", np.array([[False], [True], [False], [False]]), "masked", hard, i64(0, 2), ("list", [7, 8]), "raise")
        for mk0 in (np.array(False), np.array(True)):
            emit(f"0d.{hard}", t, "0d", mk0, "masked", hard, np.array(0), ("scalar", 9), "raise")
            emit(f"0d.idx1.{hard}", t, "0d", mk0, "masked", hard, i64(0), ("scalar", 9), "raise")
        emit(f"empty.{hard}", t, "empty", np.zeros(0, bool), "masked", hard, i64(0), ("list", [1]), "raise")
        emit(f"empty.noidx.{hard}", t, "empty", np.zeros(0, bool), "masked", hard, np.array([], dtype=np.int64), ("list", [1]), "raise")
    emit("0d.nomask", t, "0d", None, "masked", True, np.array(0), ("scalar", 9), "raise")
    # Value conversion: the hard branch converts first (a literal wraps), the soft branch packs it as a Python scalar.
    t8i = np.array([1, 2, 3, 4], dtype=np.int8)
    m8i = np.array([False, False, True, False])
    for hard in (True, False):
        for v in (300, -129, 2.7, float("nan"), float("inf"), True):
            emit(f"conv8.{hard}", t8i, "c1d", m8i, "masked", hard, i64(0), ("scalar", v), "raise")
            emit(f"conv64.{hard}", t, "c1d", m1, "masked", hard, i64(0), ("scalar", v), "raise")
            emit(f"convf.{hard}", np.array([1.0, 2.0, 3.0, 4.0]), "c1d", m1, "masked", hard, i64(0), ("scalar", v), "raise")
    emit("conv8.list.hard", t8i, "c1d", m8i, "masked", True, i64(0, 1), ("list", [300, 7]), "raise")

    # Section R: random cases over every dtype.
    for dt in SO_DTYPES:
        d = np.dtype(dt)
        for draw in range(260):
            lay = str(rng.choice(PUT_LAYOUTS))
            size = 8 if lay == "c3d" else int(rng.choice([1, 2, 4, 5, 6, 8, 9]))
            tv = _so_values(dt, size, rng, str(rng.choice(["small", "small", "special", "wide"])))
            _, view = _put_target(tv, lay)
            mask = _so_mask(view.shape, str(rng.choice(["none", "false", "some", "some", "all"])), rng)
            kind = "plain" if rng.random() < 0.12 else "masked"
            hard = kind == "masked" and rng.random() < 0.55
            vsz = max(int(view.size), 1)

            ik = str(rng.choice(["i64", "i64", "i64", "i32", "i8", "u64", "bool", "float", "2d", "0d", "empty", "oob"]))
            if ik == "bool":
                idx = rng.random(int(rng.choice([vsz, vsz, 3]))) < 0.5
            elif ik == "float":
                idx = rng.integers(0, vsz, int(rng.integers(1, 4))).astype(np.float64)
            elif ik == "2d":
                idx = rng.integers(-vsz, vsz, (2, 2)).astype(np.int64)
            elif ik == "0d":
                idx = np.array(int(rng.integers(-vsz, vsz)), dtype=np.int64)
            elif ik == "empty":
                idx = np.array([], dtype=np.int64)
            elif ik == "oob":
                idx = np.array([0, vsz + int(rng.integers(0, 3))], dtype=np.int64)
            elif ik == "u64":
                idx = rng.integers(0, vsz, int(rng.integers(1, 4))).astype(np.uint64)
            else:
                idx = rng.integers(-vsz, vsz, int(rng.integers(1, 6))).astype(
                    {"i64": np.int64, "i32": np.int32, "i8": np.int8}[ik])
            mode = str(rng.choice(["raise", "raise", "wrap", "clip"]))
            ni = int(np.size(idx))

            vk = str(rng.choice(["scalar", "list", "list", "array", "array", "masked", "constant"]))
            if vk == "scalar":
                if d.kind == "b":
                    vv = bool(rng.integers(0, 2))
                elif d.kind == "c":
                    vv = complex(int(rng.integers(0, 9)), int(rng.integers(-2, 3)))
                elif d.kind == "f":
                    vv = float(rng.choice([0.5, 1.0, -2.0, 3.25]))
                else:
                    vv = int(rng.integers(0, 50))
                values = ("scalar", vv)
            elif vk == "list":
                k = int(rng.choice([max(ni, 1), max(ni, 1), 1, 3, 6]))
                values = ("list", _so_values(dt, k, rng, "small").tolist())
            elif vk == "array":
                k = int(rng.choice([max(ni, 1), max(ni, 1), 1, 2, 5]))
                lay_v = str(rng.choice(["c", "c", "f", "strided", "view", "col", "0d"]))
                src = _so_values(dt, max(k, 1), rng, str(rng.choice(["small", "special"])))
                values = ("array", _put_array_values(src, lay_v))
            elif vk == "masked":
                k = int(rng.choice([max(ni, 1), 1, 3]))
                mv = _so_values(dt, k, rng, "small")
                values = ("masked", ma.array(mv, mask=_so_mask(mv.shape, str(rng.choice(["none", "some", "all"])), rng)))
            else:
                values = ("constant", None)
            emit(f"{dt}.{lay}.{ik}.{vk}", tv, lay, mask, kind, hard, idx, values, mode)

    print(f"  (ma_put kept {stats['kept']}, {stats['errors']} of them NumPy errors)")
    return cases


# ---------------------------------------------------------------------------------------------
TIERS = {
    "unary": ("ma_unary.jsonl", gen_ma_unary),
    "binary": ("ma_binary.jsonl", gen_ma_binary),
    "reduce": ("ma_reduce.jsonl", gen_ma_reduce),
    "scan": ("ma_scan.jsonl", gen_ma_scan),
    "manip": ("ma_manip.jsonl", gen_ma_manip),
    "construct": ("ma_construct.jsonl", gen_ma_construct),
    "select": ("ma_select.jsonl", gen_ma_select),
    "sortsetops": ("ma_sortsetops.jsonl", gen_ma_sortsetops),
    "extras": ("ma_extras.jsonl", gen_ma_extras),
    "meanvar": ("ma_meanvar.jsonl", gen_ma_meanvar),
    "setops": ("ma_setops.jsonl", gen_ma_setops),
    "put": ("ma_put.jsonl", gen_ma_put),
}


def write_jsonl(path, cases):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases -> {os.path.basename(path)}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.normpath(os.path.join(here, "..", "NumSharp.Tests.Oracle", "Fuzz", "corpus"))
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "meanvar" and len(sys.argv) > 3 and sys.argv[2] == "--uncarved":
        # The carve check: every cell, carved ones included, written to the given path (never the corpus).
        write_jsonl(os.path.abspath(sys.argv[3]), gen_ma_meanvar(carve=False))
        return
    tiers = TIERS.keys() if mode == "all" else [mode]
    for t in tiers:
        if t not in TIERS:
            print(f"unknown ma tier '{t}' (expected: all | {' | '.join(TIERS)})")
            sys.exit(2)
        fname, fn = TIERS[t]
        write_jsonl(os.path.join(corpus_dir, fname), fn())


if __name__ == "__main__":
    main()
