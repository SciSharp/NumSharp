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
MA_DTYPES = ["bool", "uint8", "int32", "int64", "float16", "float32", "float64", "complex128"]
MA_MASKS = ["none", "some", "all"]

# Pairwise: a compact dtype-pair set (same-type + int/float crossing + complex + bool promotion)
# and a representative pair-layout subset (contiguous, one-Fortran, inner-strided, negative-stride,
# broadcast-row). Mask COMBOS exercise OR-propagation: neither / both / one-sided / all|some.
MA_DT_PAIRS = [("int32", "int32"), ("int32", "float64"), ("float32", "float32"),
               ("float64", "float64"), ("complex128", "complex128"), ("int64", "int64"),
               ("uint8", "uint8"), ("float16", "float16"), ("bool", "int32")]
MA_PAIR_LAYOUTS = ["pp_contig_contig", "pp_contig_fortran", "pp_contig_strided",
                   "pp_negstride_both", "pp_broadcast_row"]
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
    for ln in MA_LAYOUTS:
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
    for ln in MA_LAYOUTS:
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
    for ln in MA_LAYOUTS:
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
                             ("diff", {"n": 2}, lambda mm: ma.diff(mm, 2))]
                    if view.ndim >= 2:
                        jobs += [("cumsum", {"axis": 0}, lambda mm: ma.cumsum(mm, axis=0)),
                                 ("cumprod", {"axis": 1}, lambda mm: ma.cumprod(mm, axis=1)),
                                 ("diff", {"axis": 0}, lambda mm: ma.diff(mm, 1, axis=0))]
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
    for ln in MA_LAYOUTS:
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
                ]
                if nd >= 1:
                    jobs += [("moveaxis", {"source": 0, "destination": nd - 1},
                              lambda mm, nd=nd: ma.moveaxis(mm, 0, nd - 1))]
                if nd >= 2:
                    jobs += [("swapaxes", {"axis1": 0, "axis2": 1}, lambda mm: ma.swapaxes(mm, 0, 1)),
                             ("diagonal", {}, lambda mm: ma.diagonal(mm))]
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
    for ln in MA_LAYOUTS:
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
                # putmask
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
    for ln in MA_LAYOUTS:
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
    tiers = TIERS.keys() if mode == "all" else [mode]
    for t in tiers:
        if t not in TIERS:
            print(f"unknown ma tier '{t}' (expected: all | {' | '.join(TIERS)})")
            sys.exit(2)
        fname, fn = TIERS[t]
        write_jsonl(os.path.join(corpus_dir, fname), fn())


if __name__ == "__main__":
    main()
