#!/usr/bin/env python3
"""
gen_ma_surface_oracle.py -- oracle contracts for the numpy.ma functions the coverage inventory lists as AVAILABLE but
UNPROVEN.

The coverage artifact (coverage/generate_coverage.py) joins every committed NumPy oracle contract onto its API row. After
the surface tier (gen_surface_oracle.py, 2026-10-08) it still listed 51 numpy.ma functions that NumSharp exposes
(np.ma.<name>) with no contract: the creation wrappers (arange, zeros, ones, empty, the *_like forms, identity,
indices, frombuffer, fromfunction), the joins (column_stack, dstack, hsplit, diagflat, resize), the fill-value
helpers (default_fill_value, minimum_fill_value, maximum_fill_value, set_fill_value, common_fill_value), the
predicates and queries (getmask, is_mask, isarray, isMA, isMaskedArray, ndim, shape, size, nonzero, ids, harden_mask,
soften_mask), masked_object and putmask, the products (convolve, correlate, cov, corrcoef), the edge finders
(clump_masked, clump_unmasked, flatnotmasked_contiguous, flatnotmasked_edges, notmasked_contiguous, notmasked_edges),
compress_nd, compress_rowcols, ndenumerate, apply_along_axis, apply_over_axes and polyfit. A few of them were even
registered in OpRegistry.Ma.cs, but the ma generator never produced a case for them (a putmask block failed to build
its operands and was skipped, every catalog 2-D layout has an odd last axis, so hsplit never ran).

Inputs the ma corpus could not express ride params, as in the surface tier:

  * a FUNCTION argument is a name from a fixed menu (ALONG_FUNCS, OVER_FUNCS, FROMFUNCTION_FUNCS) both sides map to the
    same computation;
  * a Python scalar (a fill value, a putmask value, an arange bound) is a JSON param, typed as the C# literal NumPy's
    Python object stands for (int -> long, float -> double, bool, complex -> {"re", "im"}, None -> null, str);
  * an object's state is the case's steps (set a fill value, harden the mask, assign through the indexer) applied to
    the reconstructed operand before the call.

Every masked operand is a DATA view in one of the relayout tier's ndarray kinds (C, F, permuted, stepped and C views at
an offset, reversed, stepped-and-reversed, read-only, broadcast, one-element 1-D, empty, an empty view inside a buffer)
plus a C-contiguous mask in one of the MASK patterns (nomask, an explicit all-False mask, a mixed pattern, all
masked, only the first or last element, alternating, runs). NumPy computes every record (or its error type and text)
on operands rebuilt from the very descriptors NumSharp reconstructs.

Results are compared by kind. A masked result records filled(0) data and getmaskarray (the ma tiers' contract), plus
the FORM (the masked singleton, a scalar, an array with nomask or with a mask array) and, where the masked slots hold
data NumPy defines (copies, joins, putmask writes, products), the RAW data at every slot. A Python object result (a
fill value, a list of slices, an edge tuple, a shape tuple, None) is its normalized Python repr (see pyrepr).

Output (../NumSharp.Tests.Oracle/Fuzz/corpus/):

  ma_surface.jsonl              replayed by FuzzCorpusTests.MaSurface (valueclass "masurface": ZERO excuses, neither
                                MisalignedRegistry's nor the ma harness's own)
  gaps/ma_surface_gaps.jsonl    the cases a carve rule (_gap_of) names: NumSharp is known to differ there; replayed by
                                an [OpenBugs] test that names each gap

Run (needs numpy==2.4.2; deterministic: a rerun is byte-identical):
    python test/oracle/gen_ma_surface_oracle.py [--uncarved DIR]
"""
import argparse
import collections
import itertools
import json
import os
import sys
import warnings

import numpy as np

np.seterr(all="ignore")
warnings.simplefilter("ignore")

assert np.__version__ == "2.4.2", f"corpus must be generated with numpy 2.4.2, got {np.__version__}"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_surface_oracle as sf  # noqa: E402 -- values(), operand(), kinds_of(): the surface tier's operand kinds
import gen_relayout_oracle as rl  # noqa: E402 -- rebuild(): a data view exactly as FuzzCorpus.Reconstruct builds it

ma = np.ma
CORPUS = sf.CORPUS
GAP_DIR = sf.GAP_DIR

INTS = sf.INTS
FLOATS = sf.FLOATS
ALL13 = sf.ALL13
# The dtypes of the arithmetic families: one of each storage class plus the narrow and unsigned edges.
ARITH = ["int32", "int64", "int8", "uint8", "float16", "float32", "float64", "complex128", "bool"]

# The data kinds a masked operand meets. "c" first: the plain C view every older ma tier used.
DATA_KINDS = ["c", "c_offset", "f", "f_offset", "permuted", "permuted_offset", "reversed", "strided",
              "strided_offset", "strided_neg", "readonly", "bcast", "bcast_offset", "scalar_bcast", "one_1d",
              "0d_owner", "empty", "empty_view"]
# A short list for the families whose cost is in the params rather than the layout.
FEW_KINDS = ["c", "f", "reversed", "strided_offset", "bcast", "empty_view"]

MASK_PATTERNS = ["none", "false", "some", "all", "first", "last", "alt", "runs"]
FEW_MASKS = ["none", "false", "some", "all"]


# =============================================================================================================
# Operands
# =============================================================================================================


def mask_for(shape, pattern):
    """A C-contiguous bool mask of ``shape`` in one of MASK_PATTERNS, or None for "none" (NumPy's nomask).

    Parameters:
        shape: the data shape (a tuple or list; () for a 0-d operand).
        pattern: "none", "false" (an explicit all-False mask: NumPy keeps it as an array, unlike nomask), "some"
            (F F T T F F T T ...), "all", "first" / "last" (one masked element), "alt" (every other element),
            "runs" (alternating runs of three).

    Returns:
        A NumPy bool array of ``shape``, or None.
    """
    shape = tuple(shape)
    n = int(np.prod(shape)) if len(shape) else 1
    if pattern == "none":
        return None
    k = np.arange(n)
    flat = {
        "false": np.zeros(n, dtype=bool),
        "all": np.ones(n, dtype=bool),
        "some": k % 4 >= 2,
        "first": k == 0,
        "last": k == n - 1,
        "alt": k % 2 == 1,
        "runs": (k // 3) % 2 == 0,
    }[pattern]
    return flat.reshape(shape)


def with_mask(desc, pattern):
    """A copy of the data descriptor ``desc`` with the mask hex of ``pattern`` over its shape (no key for "none").

    The mask is always C-contiguous over the VIEW's shape: FuzzCorpus.ReconstructMask builds it that way, so a
    broadcast data view still carries one mask bit per element.
    """
    m = mask_for(desc["shape"], pattern)
    d = dict(desc)
    if m is not None:
        d["mask"] = np.ascontiguousarray(m).tobytes().hex()
    return d


def rebuild_masked(d):
    """The NumPy MaskedArray a masked descriptor names, built exactly as FuzzCorpusTests.RunMaCorpus builds NumSharp's:
    the data view (gen_relayout_oracle.rebuild) wrapped by np.ma.masked_array with the mask (nomask when absent)."""
    data = rl.rebuild(d)
    if "mask" not in d:
        return ma.masked_array(data, mask=ma.nomask)
    m = np.frombuffer(bytes.fromhex(d["mask"]), dtype=bool).copy().reshape(tuple(d["shape"]))
    return ma.masked_array(data, mask=m)


def mops(descs):
    """The masked operands the descriptors name (fresh per case, so a mutating call never leaks into the next)."""
    return [rebuild_masked(d) for d in descs]


def mkinds(vals, kinds, patterns):
    """Every (label, masked descriptor) of ``vals`` over the data ``kinds`` and mask ``patterns``.

    Parameters:
        vals: the logical values (a NumPy array).
        kinds: data kind names (sf.operand: "c", "own" or a relayout transform).
        patterns: mask pattern names.

    Returns:
        A list of (label, descriptor); the label is "<dtype>.<kind>.<pattern>". A transform that does not apply to the
        shape (an F view of a 1-D array) is skipped, as is a second transform landing on an already-emitted kind.
    """
    out = []
    for kind, d in sf.kinds_of(vals, kinds):
        for pat in patterns:
            out.append((f"{np.asarray(vals).dtype.name}.{kind}.{pat}", with_mask(d, pat)))
    return out


def plain(vals):
    """The C descriptor of ``vals`` without a mask (a second operand whose layout is not the point)."""
    v = np.array(vals, copy=True, order="C")
    return sf.describe(v, v)


def matrix(runs, seed):
    """(dtype, shape, kind, label, masked descriptor) over a list of runs, each (dtypes, shapes, kinds, patterns).

    The families below use one run for the C view over the full dtype x shape x mask matrix and smaller runs for the
    other data kinds (a layout is dtype-independent) and the rarer mask patterns, so the corpus stays in the tens of
    thousands while every combination class is met.

    Parameters:
        runs: the (dtypes, shapes, kinds, patterns) tuples.
        seed: the values() seed.

    Yields:
        (dtype, shape, kind, label, descriptor); kind is the data kind the transform landed on.
    """
    for dtypes, shapes, kinds, pats in runs:
        for s, shape in itertools.product(dtypes, shapes):
            for label, d in mkinds(sf.values(shape, s, seed=seed), kinds, pats):
                yield s, shape, label.split(".")[1], label, d


# =============================================================================================================
# Python repr, normalized (the "text" result kind)
# =============================================================================================================


def pyrepr(v):
    """Python's repr of a result NumPy returns as a Python object, normalized the way OpRegistry.MaSurface.MaRepr
    formats NumSharp's answer.

    Normalization: a NumPy scalar or a 0-d array is its ``.item()`` (a fill value's Python number, not
    ``np.float64(...)``: C# has one boxed double for both); a slice step of None is written 1 (C#'s Slice has no None
    step; the selection is the same). Everything else is CPython's repr: ints, floats (shortest round trip), complex,
    bool, None, str, lists, tuples, and an N-d array's ``array(...)`` repr.

    Parameters:
        v: the value.

    Returns:
        The text.

    Raises:
        TypeError: a value kind the C# side has no counterpart for (a generator bug).
    """
    if v is None:
        return "None"
    if v is ma.masked:
        return "masked"
    if isinstance(v, np.ma.MaskedArray):
        raise TypeError("a MaskedArray inside a Python-object result has no text form here")
    if isinstance(v, np.generic):
        return pyrepr(v.item())
    if isinstance(v, np.ndarray):
        return repr(v) if v.ndim else pyrepr(v.item())
    if isinstance(v, (bool, int, float, complex, str)):
        return repr(v)
    if isinstance(v, slice):
        step = 1 if v.step is None else v.step
        return f"slice({pyrepr(v.start)}, {pyrepr(v.stop)}, {pyrepr(step)})"
    if isinstance(v, list):
        return "[" + ", ".join(pyrepr(x) for x in v) + "]"
    if isinstance(v, tuple):
        parts = [pyrepr(x) for x in v]
        return "(" + parts[0] + ",)" if len(parts) == 1 else "(" + ", ".join(parts) + ")"
    raise TypeError(f"no normalized repr for {type(v).__name__}")


def jvalue(v):
    """A Python scalar as its JSON param form: int / float / bool / str / None as themselves (a non-finite float as
    "nan" / "inf" / "-inf", strict JSON has none), a complex as {"re", "im"} (non-finite parts as strings too)."""
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return v if np.isfinite(v) else ("nan" if np.isnan(v) else ("inf" if v > 0 else "-inf"))
    if isinstance(v, complex):
        return {"re": jvalue(v.real), "im": jvalue(v.imag)}
    raise TypeError(type(v).__name__)


# =============================================================================================================
# Case emission
# =============================================================================================================

CASES = []
_N = itertools.count()


def _array_record(r):
    """dtype / shape / C-order bytes of an array result (the "array" kind and every tuple slot)."""
    a = np.asarray(r)
    return {"dtype": a.dtype.name, "shape": [int(x) for x in a.shape], "buffer": np.ascontiguousarray(a).tobytes().hex()}


def _form(r):
    """Which object NumPy returned: "masked" (the singleton), "scalar" (no MaskedArray, or a 0-d one with nomask),
    "nomask" or "mask" (an array whose mask is nomask / a mask array, an all-False one included)."""
    if r is ma.masked:
        return "masked"
    if isinstance(r, np.ma.MaskedArray):
        if r._mask is ma.nomask:
            return "scalar" if np.ndim(r) == 0 else "nomask"
        return "mask"
    return "scalar"


def _masked_record(r, raw):
    """A masked result: filled(0) bytes + getmaskarray bytes + the form, and the raw data at every slot when ``raw``."""
    filled = np.asarray(ma.filled(r, 0))
    marr = np.asarray(ma.getmaskarray(r), dtype=bool)
    e = {"kind": "masked", "dtype": filled.dtype.name, "shape": [int(d) for d in np.shape(r)],
         "buffer": np.ascontiguousarray(filled).tobytes().hex(), "mask": np.ascontiguousarray(marr).tobytes().hex(),
         "form": _form(r)}
    if raw:
        data = np.asarray(r.data) if isinstance(r, np.ma.MaskedArray) else np.asarray(r)
        e["raw"] = np.ascontiguousarray(data).tobytes().hex()
    return e


def emit(section, op, label, descs, params, kind, fn, raw=False):
    """Run ``fn`` (NumPy's call over masked operands rebuilt from ``descs``) and record the case.

    Parameters:
        section: the generator section (part of the id and the layout tag).
        op: the bare numpy.ma function name (the corpus key is "ma.<op>": the coverage join's "ma" prefix rule).
        label: a short label of the operand kinds / variant.
        descs: the masked operand descriptors (data + optional mask).
        params: the case params (JSON-serializable).
        kind: the result kind: "masked", "masked_tuple", "array", "scalar", "tuple", "text" or "dtype".
        fn: callable(ops) -> NumPy's result; it may raise (an error case, recorded verbatim).
        raw: also record the data at masked slots (only where NumPy defines it).

    Returns:
        The case dict (already appended to CASES).
    """
    n = next(_N)
    case = {"id": f"ma.{op}/msurf:{section}:{label}/{n}", "op": f"ma.{op}", "params": params, "operands": descs,
            "layout": f"msurf:{section}:{label}", "valueclass": "masurface"}
    try:
        r = fn(mops(descs))
    except Exception as e:  # noqa: BLE001 -- NumPy's refusal is the outcome
        case["expected"] = {} if kind in ("masked", "array") else {"kind": kind}
        case["expects_throw"] = True
        case["error"] = {"type": type(e).__name__, "text": str(e)}
        CASES.append(case)
        return case
    if kind == "masked":
        case["expected"] = _masked_record(r, raw)
    elif kind == "masked_tuple":
        case["expected"] = {"kind": "masked_tuple", "slots": [_masked_record(x, raw) for x in r]}
    elif kind == "array":
        case["expected"] = _array_record(r)
    elif kind == "scalar":
        case["expected"] = {"kind": "scalar", **_array_record(r)}
    elif kind == "tuple":
        case["expected"] = {"kind": "tuple", "slots": [_array_record(x) for x in r]}
    elif kind == "text":
        case["expected"] = {"kind": "text", "value": pyrepr(r)}
    elif kind == "dtype":
        case["expected"] = {"kind": "dtype", "value": np.dtype(r).name}
    else:
        raise ValueError(kind)
    CASES.append(case)
    return case


# =============================================================================================================
# Section "create": the _convert2ma wrappers (arange, zeros, ones, empty, *_like, identity, indices, frombuffer,
# fromfunction)
# =============================================================================================================

SHAPES_CREATE = [(), (0,), (3,), (2, 3), (2, 0, 3), (1, 1, 1), (2, 1, 3, 1)]

# fromfunction's callable menu: NumPy calls function(*indices(shape, dtype)). No int literals: a C# literal and a Python
# literal promote differently against a narrow grid, which would test the operators, not ma.fromfunction.
FROMFUNCTION_FUNCS = {
    "first": (1, lambda i, *rest: i),
    "add2": (2, lambda i, j: i + j),
    "mul2": (2, lambda i, j: i * j),
    "sub2": (2, lambda i, j: i - j),
    "gt2": (2, lambda i, j: i > j),
    "sq2": (2, lambda i, j: i * i + j),
    "add3": (3, lambda i, j, k: i + j + k),
}


def _nodata(r):
    """The empty / empty_like facet: the result with its uninitialized data replaced by zeros of its dtype and its mask
    (or nomask) kept, so shape, dtype, mask and form are compared and the arbitrary bytes are not."""
    return ma.masked_array(np.zeros(r.shape, r.dtype), mask=r._mask)


def gen_create():
    """arange / zeros / ones / empty / identity / indices / frombuffer / fromfunction (no operands) and the *_like
    forms over masked operands of every kind and mask pattern."""
    sec = "create"
    # arange: the C# overloads are (int stop), (int start, int stop, int step = 1) and (double start, double stop,
    # double step = 1); a lone float bound and dtype= have no overload (the registry refuses them).
    for args in [(0,), (1,), (5,), (-3,), (2, 7), (7, 2), (2, 7, 2), (7, 2, -1), (0, 10, 3), (5, 5), (-4, 4, 3),
                 (10, 0, -3), (1.0, 2.0, 0.25), (0.0, 1.0, 0.1), (-1.5, 1.5, 0.5), (2.5, -1.0, -0.75), (0.0, 0.0, 1.0),
                 (1.0, 4.0), (1, 2.5), (0.0, 3.0, 0.5), (5.5,), (0, 10, 0), (0.0, 1.0, 0.0), (1e18, 1e18 + 4, 1.0)]:
        emit(sec, "arange", "args", [], {"args": [jvalue(a) for a in args]}, "masked",
             lambda ops, a=args: ma.arange(*a))
    for args, dt in [((5,), "int8"), ((0, 3), "float32"), ((1.0, 2.0, 0.5), "int64"), ((4,), "complex128")]:
        emit(sec, "arange", "dtype", [], {"args": [jvalue(a) for a in args], "dtype": dt}, "masked",
             lambda ops, a=args, d=dt: ma.arange(*a, dtype=d))
    for op in ["zeros", "ones", "empty"]:
        for shape in SHAPES_CREATE:
            for dt in [None] + ALL13:
                fn = getattr(ma, op)
                p = {"shape": list(shape)}
                if dt is not None:
                    p["dtype"] = dt
                if op == "empty":
                    p["facet"] = "nodata"
                emit(sec, op, f"{dt or 'default'}.{len(shape)}d", [], p, "masked",
                     lambda ops, f=fn, s=shape, d=dt, o=op: (_nodata if o == "empty" else (lambda r: r))(
                         f(s) if d is None else f(s, dtype=d)), raw=True)
        # order= is a NumPy parameter NumSharp's signature lacks (the values are the same; the case documents it).
        p = {"shape": [2, 3], "order": "F"}
        if op == "empty":
            p["facet"] = "nodata"
        emit(sec, op, "order", [], p, "masked",
             lambda ops, f=getattr(ma, op), o=op: (_nodata if o == "empty" else (lambda r: r))(f((2, 3), order="F")))
    like_runs = [(ALL13, [(6,), (2, 3), (2, 3, 2), (), (0, 3)], ["c"], FEW_MASKS),
                 (["float64"], [(6,), (2, 3)], DATA_KINDS[1:], ["some"]),
                 (["float64", "int32"], [(2, 3)], ["c"], ["first", "last", "alt", "runs"])]
    for op in ["zeros_like", "ones_like", "empty_like"]:
        fn = getattr(ma, op)
        for s, shape, kind, label, d in matrix(like_runs, 1):
            # dtype= on the C view of the 2-D shape of three dtypes; the prototype's dtype everywhere else.
            dts = [None, "int8", "float32", "complex128", "bool"] if (
                kind == "c" and shape == (2, 3) and s in ("float64", "int32", "bool")) else [None]
            for dt in dts:
                p = {} if dt is None else {"dtype": dt}
                if op == "empty_like":
                    p["facet"] = "nodata"
                emit(sec, op, f"{label}.{dt or 'same'}", [d], p, "masked",
                     lambda ops, f=fn, d_=dt, o=op: (_nodata if o == "empty_like" else (lambda r: r))(
                         f(ops[0]) if d_ is None else f(ops[0], dtype=d_)), raw=True)
    for op in ["zeros_like", "ones_like", "empty_like"]:
        fn = getattr(ma, op)
        # A plain ndarray prototype (no MaskedArray at all): NumPy's result has nomask.
        for s in ["float64", "int8"]:
            d = plain(sf.values((2, 3), s))
            p = {"plain": True}
            if op == "empty_like":
                p["facet"] = "nodata"
            emit(sec, op, f"{s}.plain", [d], p, "masked",
                 lambda ops, f=fn, o=op: (_nodata if o == "empty_like" else (lambda r: r))(f(ops[0].data)), raw=True)
    for n in [0, 1, 3, 5]:
        for dt in [None, "bool", "int8", "int32", "uint64", "float16", "float32", "complex128"]:
            p = {"n": n}
            if dt is not None:
                p["dtype"] = dt
            emit(sec, "identity", f"{dt or 'default'}", [], p, "masked",
                 lambda ops, n_=n, d=dt: ma.identity(n_) if d is None else ma.identity(n_, dtype=d), raw=True)
    for dims in [(), (0,), (3,), (2, 3), (0, 2), (2, 1, 2), (1,)]:
        for dt in [None, "int8", "int32", "uint16", "float64", "float32", "complex128"]:
            p = {"dimensions": list(dims)}
            if dt is not None:
                p["dtype"] = dt
            emit(sec, "indices", f"{dt or 'default'}", [], p, "masked",
                 lambda ops, dm=dims, d=dt: ma.indices(dm) if d is None else ma.indices(dm, dtype=d), raw=True)
    emit(sec, "indices", "sparse", [], {"dimensions": [2, 3], "sparse": True}, "masked",
         lambda ops: ma.indices((2, 3), sparse=True))
    # frombuffer over the bytes of an int16 / float64 ramp read back as every dtype; count / offset in and out of range.
    raw16 = np.arange(24, dtype=np.uint8).tobytes()
    for dt in [None] + ALL13:
        isz = np.dtype(dt or "float64").itemsize
        for count, offset in [(-1, 0), (2, 0), (1, isz), (-1, isz), (0, 0), (3, 2 * isz), (-1, 1), (100, 0), (1, 25),
                              (-1, 24), (2, -1)]:
            p = {"buffer": raw16.hex(), "count": count, "offset": offset}
            if dt is not None:
                p["dtype"] = dt
            emit(sec, "frombuffer", f"{dt or 'default'}.c{count}.o{offset}", [], p, "masked",
                 lambda ops, d=dt, c=count, o=offset: ma.frombuffer(raw16, dtype=(d or float), count=c, offset=o),
                 raw=True)
    for name, (arity, f) in FROMFUNCTION_FUNCS.items():
        shapes = {1: [(4,), (0,), (1,)], 2: [(2, 3), (3, 1), (0, 2), (1, 1)], 3: [(2, 3, 2), (1, 2, 1)]}[arity]
        for shape in shapes:
            for dt in [None, "int64", "int8", "uint8", "float32", "complex128"]:
                p = {"func": name, "shape": list(shape)}
                if dt is not None:
                    p["dtype"] = dt
                emit(sec, "fromfunction", f"{name}.{dt or 'default'}", [], p, "masked",
                     lambda ops, f_=f, s=shape, d=dt: ma.fromfunction(f_, s) if d is None else ma.fromfunction(f_, s, dtype=d),
                     raw=True)


# =============================================================================================================
# Section "join": column_stack / dstack / hsplit / diagflat / resize
# =============================================================================================================


def gen_join():
    """The _fromnxfunction wrappers (each applies the np function to the data and to the mask) and ma.resize."""
    sec = "join"
    pairs = [((6,), (6,)), ((3, 2), (3, 4)), ((3,), (3, 2)), ((), ()), ((2, 3, 2), (2, 3, 1)), ((0,), (0,)),
             ((6,), (3,)), ((3, 2), (2, 2))]
    dt_pairs = [("float64", "float64"), ("int32", "float64"), ("bool", "int8"), ("uint8", "uint8"),
                ("complex128", "int64"), ("float16", "float32")]
    for op in ["column_stack", "dstack"]:
        f = getattr(ma, op)
        for sa, sb in pairs:
            for s, t in dt_pairs:
                va, vb = sf.values(sa, s, seed=1), sf.values(sb, t, seed=2)
                # The C view of the first operand meets four mask patterns against two; the other data kinds (only for
                # the float64 pair) one pattern against nomask.
                runs = [(["c"], ["none", "some", "all", "false"], ["none", "some"])]
                if (s, t) == ("float64", "float64"):
                    runs.append((DATA_KINDS[1:], ["some"], ["none"]))
                for kinds, pas, pbs in runs:
                    for label, da in mkinds(va, kinds, pas):
                        for pb in pbs:
                            db = with_mask(plain(vb), pb)
                            emit(sec, op, f"{label}+{t}.{pb}", [da, db], {}, "masked",
                                 lambda ops, f_=f: f_(tuple(ops)), raw=True)
        # Three operands.
        for s in ["float64", "int64"]:
            ds = [with_mask(plain(sf.values((4,), s, seed=i)), p) for i, p in enumerate(["some", "none", "all"])]
            emit(sec, op, f"{s}.three", ds, {}, "masked", lambda ops, f_=f: f_(tuple(ops)), raw=True)
    split_runs = [(["float64", "int32", "bool", "complex128", "uint16"], [(3, 4), (6,), (2, 4, 2), (4, 0), (1, 6), ()],
                   ["c"], FEW_MASKS + ["first"]),
                  (["float64"], [(3, 4), (6,)], DATA_KINDS[1:], ["some"])]
    for s, shape, kind, label, d in matrix(split_runs, 3):
        for sec_arg in [1, 2, 4, 3]:
            emit(sec, "hsplit", f"{label}.n{sec_arg}", [d], {"sections": sec_arg}, "masked_tuple",
                 lambda ops, k=sec_arg: ma.hsplit(ops[0], k), raw=True)
        for idx in [[1, 3], [0], [5], [2, 2], [-1], []]:
            emit(sec, "hsplit", f"{label}.idx", [d], {"indices": idx}, "masked_tuple",
                 lambda ops, ix=idx: ma.hsplit(ops[0], ix), raw=True)
    diag_runs = [(["float64", "int32", "bool", "complex128", "int8", "float16"], [(4,), (2, 2), (), (0,), (1,), (2, 3)],
                  ["c"], FEW_MASKS + ["last"]),
                 (["float64"], [(4,), (2, 2)], DATA_KINDS[1:], ["some"])]
    for s, shape, kind, label, d in matrix(diag_runs, 4):
        for k in [0, 1, -2, 5]:
            emit(sec, "diagflat", f"{label}.k{k}", [d], {"k": k}, "masked",
                 lambda ops, k_=k: ma.diagflat(ops[0], k_), raw=True)
    resize_runs = [(["float64", "int64", "bool", "complex128", "uint8"], [(6,), (2, 3), (), (0,), (1,), (2, 3, 2)],
                    ["c"], FEW_MASKS + ["first"]),
                   (["float64"], [(6,), (2, 3)], DATA_KINDS[1:], ["some"])]
    for s, shape, kind, label, d in matrix(resize_runs, 5):
        for new in [(2, 3), (4,), (9,), (0,), (), (3, 1, 2), (1,), (2, -1)]:
            emit(sec, "resize", f"{label}.to{len(new)}d", [d], {"shape": list(new)}, "masked",
                 lambda ops, nw=new: ma.resize(ops[0], nw), raw=True)


# =============================================================================================================
# Section "fill": default_fill_value / minimum_fill_value / maximum_fill_value / set_fill_value / common_fill_value
# =============================================================================================================

FILL_VALUES = [7, 0, -1, 300, -129, 2 ** 40, 2 ** 63, 1.5, -2.5, 0.1, float("nan"), float("inf"), True, False,
               complex(1, 2), complex(3, 0), None, "abc", "5"]


def gen_fill():
    """The fill-value helpers. Every result is the normalized Python repr of the value NumPy returns (or of the array's
    fill_value after set_fill_value), so the VALUE and its Python kind (bool / int / float / complex / None) are
    compared, not the C# boxing type."""
    sec = "fill"
    for spec in ALL13 + ["f4", "i2", "<f8", "c16", "?", "u8", "e"]:
        emit(sec, "default_fill_value", f"dtype.{spec}", [], {"dtype": spec}, "text",
             lambda ops, sp=spec: ma.default_fill_value(np.dtype(sp)))
    for op in ["default_fill_value", "minimum_fill_value", "maximum_fill_value"]:
        f = getattr(ma, op)
        for s in ALL13:
            for label, d in mkinds(sf.values((3,), s), ["c", "reversed", "bcast", "empty"], ["none", "some"]):
                emit(sec, op, f"{label}", [d], {}, "text", lambda ops, f_=f: f_(ops[0]))
        # A Python scalar (NumPy reads its asanyarray dtype) and a dtype object.
        for v in [1, 1.5, True, complex(0, 1), -7]:
            emit(sec, op, f"scalar.{type(v).__name__}", [], {"value": jvalue(v)}, "text", lambda ops, f_=f, v_=v: f_(v_))
        if op != "default_fill_value":
            for spec in ["float64", "int8", "bool", "complex128", "uint64", "float16"]:
                emit(sec, op, f"dtype.{spec}", [], {"dtype": spec}, "text",
                     lambda ops, f_=f, sp=spec: f_(np.dtype(sp)))
    for s in ALL13:
        for fv in FILL_VALUES:
            d = with_mask(plain(sf.values((4,), s)), "some")

            def setter(ops, v=fv):
                ma.set_fill_value(ops[0], v)
                return ops[0].fill_value
            emit(sec, "set_fill_value", f"{s}.{type(fv).__name__}", [d], {"value": jvalue(fv)}, "text", setter)
    # set_fill_value on a plain ndarray does nothing (and returns None).
    d = plain(sf.values((4,), "float64"))
    emit(sec, "set_fill_value", "plain", [d], {"value": 5, "plain": True}, "text",
         lambda ops: ma.set_fill_value(ops[0].data, 5))
    # common_fill_value over two masked arrays whose fill values were set first (None: the dtype default).
    for s, t in [("float64", "float64"), ("int64", "int64"), ("int32", "float64"), ("float32", "float64"),
                 ("bool", "bool"), ("int8", "int64"), ("complex128", "float64"), ("uint8", "int16")]:
        for fa, fb in [(None, None), (3, 3), (3, 3.0), (3, 4), (1.5, 1.5), (None, 999999), (None, 1e20), (True, 1),
                       (complex(1, 0), 1)]:
            da = with_mask(plain(sf.values((3,), s)), "some")
            db = with_mask(plain(sf.values((3,), t)), "none")

            def common(ops, a=fa, b=fb):
                if a is not None:
                    ma.set_fill_value(ops[0], a)
                if b is not None:
                    ma.set_fill_value(ops[1], b)
                return ma.common_fill_value(ops[0], ops[1])
            emit(sec, "common_fill_value", f"{s}+{t}", [da, db], {"fa": jvalue(fa), "fb": jvalue(fb)}, "text", common)


# =============================================================================================================
# Section "query": getmask / is_mask / isarray / isMA / isMaskedArray / ndim / shape / size / nonzero / ids /
# harden_mask / soften_mask / masked_object
# =============================================================================================================

# The argument forms of the predicates: the masked operand, its data ndarray, its full mask, getmask(op) (nomask or the
# mask), a Python list, a Python int, the masked singleton and nomask itself.
PRED_ARGS = ["masked", "data", "maskarray", "getmask", "list", "int", "singleton", "nomask"]


def _pred_arg(ops, arg):
    """The NumPy object a predicate case passes (see PRED_ARGS); OpRegistry.MaSurface builds the same in C#."""
    return {
        "masked": lambda: ops[0],
        "data": lambda: ops[0].data,
        "maskarray": lambda: ma.getmaskarray(ops[0]),
        "getmask": lambda: ma.getmask(ops[0]),
        "list": lambda: [True, False],
        "int": lambda: 5,
        "singleton": lambda: ma.masked,
        "nomask": lambda: ma.nomask,
    }[arg]()


# View recipes for ids: each is a NumPy index both sides apply to the masked operand (an int keeps a 1-D view, "T"
# transposes).
IDS_VIEWS = {
    1: [("tail", "1:"), ("step", "::2"), ("rev", "::-1"), ("mid", "2:5")],
    2: [("rows", "1:"), ("cols", ":, 1:"), ("T", "T"), ("row1", "1"), ("rev", "::-1, ::-1")],
}


def _apply_view(m, recipe):
    """``m`` indexed by an IDS_VIEWS recipe: "T" is the transpose, any other string a comma-separated slice tuple."""
    if recipe == "T":
        return m.T
    parts = []
    for p in recipe.split(","):
        p = p.strip()
        if ":" in p:
            bits = [int(b) if b else None for b in p.split(":")]
            parts.append(slice(*bits))
        else:
            parts.append(int(p))
    return m[tuple(parts)]


def gen_query():
    """The predicates and the shape / mask queries, ids by its relational facet, the hard-mask switches by their effect
    on assignment, and masked_object."""
    sec = "query"
    q_runs = [(ALL13, [(6,), (2, 3), (), (0, 3)], ["c"], FEW_MASKS),
              (["float64"], [(6,), (2, 3)], DATA_KINDS[1:], ["some", "false"]),
              (["float64"], [(2, 3)], ["c"], ["first", "last", "alt", "runs"])]
    for s, shape, kind, label, d in matrix(q_runs, 2):
        emit(sec, "getmask", label, [d], {}, "array", lambda ops: ma.getmask(ops[0]))
        emit(sec, "nonzero", label, [d], {}, "tuple", lambda ops: ma.nonzero(ops[0]))
        emit(sec, "ndim", label, [d], {}, "scalar", lambda ops: ma.ndim(ops[0]))
        emit(sec, "shape", label, [d], {}, "text", lambda ops: ma.shape(ops[0]))
        emit(sec, "size", label, [d], {}, "scalar", lambda ops: ma.size(ops[0]))
        if kind == "c":
            for ax in [0, -1, 1, 3]:
                emit(sec, "size", f"{label}.ax{ax}", [d], {"axis": ax}, "scalar", lambda ops, a=ax: ma.size(ops[0], a))
    for s in ALL13:
        d = with_mask(plain(sf.values((2, 3), s)), "some")
        for arg in PRED_ARGS:
            for op in ["is_mask", "isarray", "isMA", "isMaskedArray"]:
                emit(sec, op, f"{s}.{arg}", [d], {"arg": arg}, "scalar",
                     lambda ops, a=arg, f=getattr(ma, op): bool(f(_pred_arg(ops, a))))
        emit(sec, "getmask", f"{s}.plain", [d], {"plain": True}, "array", lambda ops: ma.getmask(ops[0].data))
        for op in ["ndim", "shape", "size"]:
            for arg in ["data", "list", "int"]:
                emit(sec, op, f"{s}.{arg}", [d], {"arg": arg}, "text" if op == "shape" else "scalar",
                     lambda ops, a=arg, f=getattr(ma, op): f(_pred_arg(ops, a)))
    # ids: the address deltas between a view and its masked array, in bytes ([data, mask]). NumPy's ids is
    # (self.ctypes.data, self._mask.ctypes.data), or id(nomask) for an unmasked array: equal for the view and the array.
    for s in ["float64", "int16", "bool", "complex128"]:
        for nd, shape in [(1, (8,)), (2, (3, 4))]:
            for label, d in mkinds(sf.values(shape, s), ["c", "f", "reversed", "strided_offset", "c_offset"],
                                   ["none", "some", "false"]):
                for vname, recipe in IDS_VIEWS[nd]:
                    def delta(ops, r=recipe):
                        v = _apply_view(ops[0], r)
                        a, b = ma.ids(ops[0]), ma.ids(v)
                        return np.array([b[0] - a[0], b[1] - a[1]], dtype=np.int64)
                    emit(sec, "ids", f"{label}.{vname}", [d], {"facet": "delta", "view": recipe}, "array", delta)
    # harden_mask / soften_mask: they return the array itself and switch whether an assignment can unmask.
    assigns = [("0", 99), ("1:3", 99), (":", 77), ("0", "masked"), ("1:3", "marr")]
    hard_runs = [(["float64", "int32", "bool", "complex128", "uint8"], [(6,), (2, 3)], ["c"], FEW_MASKS),
                 (["float64"], [(6,), (2, 3)], ["f", "reversed", "strided_offset", "bcast"], ["some", "none"])]
    for s, shape, kind, label, d in matrix(hard_runs, 0):
        emit(sec, "harden_mask", f"{label}.self", [d], {"facet": "self"}, "scalar",
             lambda ops: ma.harden_mask(ops[0]) is ops[0])
        emit(sec, "soften_mask", f"{label}.self", [d], {"facet": "self"}, "scalar",
             lambda ops: ma.soften_mask(ops[0]) is ops[0])
        emit(sec, "harden_mask", f"{label}.flag", [d], {"facet": "hardmask", "steps": "hard"}, "scalar",
             lambda ops: (ma.harden_mask(ops[0]), ops[0].hardmask)[1])
        emit(sec, "soften_mask", f"{label}.flag", [d], {"facet": "hardmask", "steps": "hard_soft"}, "scalar",
             lambda ops: (ma.harden_mask(ops[0]), ma.soften_mask(ops[0]), ops[0].hardmask)[2])
        for idx, val in assigns:
            for seq in ["hard", "hard_soft"]:
                def assign(ops, i=idx, v=val, sq=seq):
                    ma.harden_mask(ops[0])
                    if sq == "hard_soft":
                        ma.soften_mask(ops[0])
                    key = _apply_view_key(i)
                    if v == "masked":
                        ops[0][key] = ma.masked
                    elif v == "marr":
                        ops[0][key] = ma.array([5, 6], mask=[True, False]).astype(ops[0].dtype)
                    else:
                        ops[0][key] = v
                    return ops[0]
                emit(sec, "harden_mask" if seq == "hard" else "soften_mask", f"{label}.{seq}.{idx}.{val}", [d],
                     {"facet": "assign", "steps": seq, "index": idx, "value": val}, "masked", assign, raw=True)
    # masked_object(x, value, copy=True, shrink=True): masked_equal plus fill_value=value.
    for s in ["int64", "float64", "int8", "bool", "complex128", "float32", "uint16"]:
        vals = sf.values((6,), s, seed=1)
        for label, d in mkinds(vals, DATA_KINDS if s == "float64" else FEW_KINDS, ["none", "some", "false"]):
            present = vals.ravel()[0].item() if vals.size else 0
            for v in [present, 12345, float("nan")]:
                emit(sec, "masked_object", f"{label}.{type(v).__name__}", [d], {"value": jvalue(v)}, "masked",
                     lambda ops, v_=v: ma.masked_object(ops[0], v_), raw=True)
                emit(sec, "masked_object", f"{label}.fill", [d], {"value": jvalue(v), "facet": "fill_value"}, "text",
                     lambda ops, v_=v: ma.masked_object(ops[0], v_).fill_value)
            emit(sec, "masked_object", f"{label}.noshrink", [d], {"value": jvalue(present), "shrink": False}, "masked",
                 lambda ops, v_=present: ma.masked_object(ops[0], v_, shrink=False), raw=True)


def _apply_view_key(idx):
    """A harden-facet index as the NumPy key: "0" -> 0, "1:3" -> slice(1, 3), ":" -> slice(None)."""
    if ":" not in idx:
        return int(idx)
    bits = [int(b) if b else None for b in idx.split(":")]
    return slice(*bits)


# =============================================================================================================
# Section "putmask": ma.putmask(a, mask, values) -- the mutated array is the result
# =============================================================================================================


def gen_putmask():
    """putmask over every data kind (read-only ones raise), every mask pattern of ``a``, soft and hard masks, and the
    value forms: a Python scalar, a full array, a row that broadcasts, a short array that does not, a masked array, the
    masked singleton, a value of another dtype."""
    sec = "putmask"
    runs = [(["float64", "int32", "int8", "uint8", "bool", "complex128", "float16", "int64"], [(6,), (2, 3), ()], ["c"],
             FEW_MASKS),
            (["float64"], [(6,), (2, 3)], DATA_KINDS[1:], ["some"]),
            (["float64"], [(2, 3)], ["c"], ["first", "last", "alt", "runs"])]
    for s, shape, kind, label, da in matrix(runs, 1):
        ashape = tuple(da["shape"])
        where = (np.arange(int(np.prod(ashape)) if ashape else 1) % 3 == 0).reshape(ashape)
        dm = plain(where)
        forms = []
        # Python scalars: an int, a float, a bool, a complex (NumPy's getdata makes each a 0-d array).
        for v in [5, 2.5, True, complex(1, 1)]:
            forms.append((f"py.{type(v).__name__}", [], {"value": jvalue(v)}, lambda m, v_=v: v_))
        full = sf.values(ashape, s, seed=7)
        forms.append(("full", [plain(full)], {}, None))
        forms.append(("fullmasked", [with_mask(plain(full), "alt")], {}, None))
        if len(ashape) == 2:
            forms.append(("row", [with_mask(plain(sf.values(ashape[1:], s, seed=8)), "first")], {}, None))
        if ashape and ashape[0] > 2:
            forms.append(("short", [plain(sf.values((2,), s, seed=9))], {}, None))
        other = "float64" if s != "float64" else "int64"
        forms.append(("otherdtype", [plain(sf.values(ashape, other, seed=10))], {}, None))
        forms.append(("singleton", [], {"value": "masked"}, lambda m: ma.masked))
        if kind != "c":
            # The other data kinds meet the value forms that read or write through every element.
            forms = [f for f in forms if f[0] in ("py.float", "full", "fullmasked", "singleton")]
        for fname, extra, p0, valfn in forms:
            for hard in [False, True]:
                p = dict(p0)
                if hard:
                    p["hard"] = True

                def call(ops, vf=valfn, h=hard):
                    a = ops[0]
                    if h:
                        a.harden_mask()
                    values = ops[2] if vf is None else vf(a)
                    ma.putmask(a, ops[1].data, values)
                    return a
                emit(sec, "putmask", f"{label}.{fname}.{'hard' if hard else 'soft'}", [da, dm] + extra, p, "masked",
                     call, raw=True)


# =============================================================================================================
# Section "products": convolve / correlate / cov / corrcoef
# =============================================================================================================


def _exact_cov_values(shape, dtype, mask, rowvar, seed):
    """Small integers in ``dtype`` whose every variable's unmasked mean is exact (each variable's unmasked sum is a
    multiple of its unmasked count), so every product and sum in the covariance is exact and any summation order --
    NumPy's dgemm, NumSharp's managed GEMM -- gives the same bits; only the final division rounds, once, on both sides.

    Parameters:
        shape: the 2-D (or 1-D) data shape.
        dtype: the data dtype (cov computes in float64 whatever it is).
        mask: the bool mask (or None) the operand carries.
        rowvar: whether a variable is a row (else a column).
        seed: varies the pattern.

    Returns:
        The C-contiguous values.
    """
    v = sf.values(shape, "int64", seed=seed, lo=-3, hi=4).astype(np.int64)
    v2 = np.atleast_2d(v).copy()
    m2 = np.zeros(v2.shape, bool) if mask is None else np.atleast_2d(mask)
    rv = rowvar or v2.shape[0] == 1
    lines = range(v2.shape[0]) if rv else range(v2.shape[1])
    for i in lines:
        line = v2[i] if rv else v2[:, i]
        keep = ~(m2[i] if rv else m2[:, i])
        cnt = int(keep.sum())
        if cnt:
            last = np.flatnonzero(keep)[-1]
            line[last] -= int(line[keep].sum()) % cnt
    out = v2.reshape(np.shape(v)).astype(dtype)
    return out


def gen_products():
    """convolve / correlate (mode x propagate_mask x mask combos x dtypes, the empty and 2-D refusals) and cov /
    corrcoef (rowvar x bias x ddof x allow_masked x an optional y, on exact-mean data)."""
    sec = "products"
    # (mode, propagate_mask) combos; None is "not passed" (the defaults differ: convolve "full", correlate "valid").
    combos_full = [(m, p) for m in [None, "full", "same", "valid"] for p in [None, True, False]]
    combos_few = [(None, None), ("full", False), ("same", True), ("valid", False)]
    # (a mask, v mask) pairs: neither, each side alone, both, an all-masked a, an explicit all-False a, edge bits.
    mask_pairs = [("none", "none"), ("some", "none"), ("none", "some"), ("some", "last"), ("all", "none"),
                  ("false", "none"), ("first", "some")]
    for op in ["convolve", "correlate"]:
        f = getattr(ma, op)
        for s in ["int32", "int64", "float64", "float32", "complex128", "bool", "uint8", "int8", "float16"]:
            for la, lv in [(5, 3), (1, 1), (3, 5), (6, 6)]:
                va, vv = sf.values((la,), s, seed=1), sf.values((lv,), s, seed=2)
                # The C view meets every mask pair; every (mode, propagate_mask) for four dtypes, four combos for the
                # rest. The other data kinds (float64 / int32 only: the layout is dtype-independent) meet two mask
                # pairs and four combos.
                full = s in ("int32", "float64", "complex128", "bool")
                runs = [(["c"], mask_pairs, combos_full if full else combos_few)]
                if s in ("float64", "int32"):
                    runs.append((DATA_KINDS[1:], [("some", "none"), ("none", "last")], combos_few))
                for kinds, pairs, combos in runs:
                    for pa, pv in pairs:
                        for label, da in mkinds(va, kinds, [pa]):
                            dv = with_mask(plain(vv), pv)
                            for mode, prop in combos:
                                p = {}
                                if mode is not None:
                                    p["mode"] = mode
                                if prop is not None:
                                    p["propagate_mask"] = prop
                                kw = dict(p)
                                emit(sec, op, f"{label}+{lv}.{pv}.{mode or 'dflt'}.{prop}", [da, dv], p, "masked",
                                     lambda ops, f_=f, k=kw: f_(ops[0], ops[1], **k), raw=True)
        for s in ["float64", "int32"]:
            emit(sec, op, f"{s}.emptya", [plain(np.zeros(0, s)), plain(sf.values((3,), s))], {}, "masked",
                 lambda ops, f_=f: f_(ops[0], ops[1]))
            emit(sec, op, f"{s}.emptyv", [plain(sf.values((3,), s)), plain(np.zeros(0, s))], {}, "masked",
                 lambda ops, f_=f: f_(ops[0], ops[1]))
            emit(sec, op, f"{s}.2d", [plain(sf.values((2, 3), s)), plain(sf.values((3,), s))], {}, "masked",
                 lambda ops, f_=f: f_(ops[0], ops[1]))
            emit(sec, op, f"{s}.badmode", [plain(sf.values((4,), s)), plain(sf.values((3,), s))], {"mode": "bad"},
                 "masked", lambda ops, f_=f: f_(ops[0], ops[1], mode="bad"))
    for op in ["cov", "corrcoef"]:
        f = getattr(ma, op)
        for shape in [(3, 4), (2, 6), (4, 2), (6,), (1, 5), (3, 1)]:
            # int16, not bool: a bool cast would undo _exact_cov_values' exact means (True / False counts over an
            # odd number of observations are inexact), and the sums would then depend on the GEMM order.
            for s in ["float64", "int32", "float32", "int16"]:
                for pat in ["none", "false", "some", "first", "all"]:
                    for rowvar in [True, False]:
                        m = mask_for(shape, pat)
                        vals = _exact_cov_values(shape, s, m, rowvar, seed=len(shape) + 1)
                        kinds = DATA_KINDS if (s == "float64" and pat == "some") else ["c", "reversed"]
                        for kind, d in sf.kinds_of(vals, kinds):
                            d = with_mask(d, pat)
                            label = f"{s}.{kind}.{pat}.rv{int(rowvar)}"
                            variants = [{}]
                            if op == "cov":
                                variants += [{"bias": True}, {"ddof": 0}, {"ddof": 2}, {"ddof": 9}]
                            variants += [{"allow_masked": False}]
                            for extra in variants:
                                p = {"rowvar": rowvar, **extra}
                                emit(sec, op, f"{label}.{'.'.join(f'{k}{v}' for k, v in extra.items()) or 'plain'}",
                                     [d], p, "masked", lambda ops, f_=f, k=dict(p): f_(ops[0], **k), raw=True)
        # y: a same-shape masked y (a common mask), a different-shape one (no common mask), 1-D pairs.
        for sx, sy, px, py in [((3, 4), (3, 4), "some", "first"), ((3, 4), (2, 4), "some", "none"),
                               ((6,), (6,), "first", "last"), ((6,), (6,), "none", "none"), ((3, 4), (3, 4), "none", "alt")]:
            for rowvar in [True, False]:
                if not rowvar and sx != sy and len(sx) == 2:
                    continue
                mx, my = mask_for(sx, px), mask_for(sy, py)
                common = None
                if sx == sy:
                    cm = np.zeros(sx, bool)
                    if mx is not None:
                        cm |= mx
                    if my is not None:
                        cm |= my
                    common = cm
                vx = _exact_cov_values(sx, "float64", common if common is not None else mx, rowvar, seed=3)
                vy = _exact_cov_values(sy, "float64", common if common is not None else my, rowvar, seed=4)
                dx, dy = with_mask(plain(vx), px), with_mask(plain(vy), py)
                emit(sec, op, f"y.{len(sx)}d.{px}{py}.rv{int(rowvar)}", [dx, dy], {"rowvar": rowvar, "y": True},
                     "masked", lambda ops, f_=f, r=rowvar: f_(ops[0], ops[1], rowvar=r), raw=True)


# =============================================================================================================
# Section "edges": clump_* / *notmasked_* / compress_nd / compress_rowcols / ndenumerate
# =============================================================================================================


def gen_edges():
    """The run / edge finders, the compressors and ndenumerate, over every mask pattern (the runs and edges ARE the
    mask) and the shapes each accepts or refuses."""
    sec = "edges"
    # Three runs: every mask pattern and shape on the float64 C view (the runs and edges ARE the mask); every data kind
    # on two shapes with the "some" / "runs" masks (compress_nd and ndenumerate read the data through its strides); and
    # every dtype on the C view of a 2-D shape (the value repr of ndenumerate, the compressed dtype).
    runs = [(["float64"], [(8,), (3, 4), (2, 3, 4), (), (0,), (0, 3), (1,), (4, 1)], ["c"], MASK_PATTERNS),
            (["float64", "int32"], [(8,), (3, 4)], DATA_KINDS[1:], ["some", "runs"]),
            ([s for s in ALL13 if s != "float64"], [(3, 4)], ["c"], ["none", "some", "all"])]
    for dtypes, shapes, kinds, pats in runs:
        for s, shape in itertools.product(dtypes, shapes):
            for label, d in mkinds(sf.values(shape, s, seed=3), kinds, pats):
                _edges_cases(sec, label, d)


def _edges_cases(sec, label, d):
    """Every edges-family call over one masked operand descriptor (see gen_edges)."""
    nd = len(d["shape"])
    for op in ["clump_masked", "clump_unmasked", "flatnotmasked_contiguous", "flatnotmasked_edges"]:
        emit(sec, op, label, [d], {}, "text", lambda ops, f=getattr(ma, op): f(ops[0]))
    for op in ["notmasked_contiguous", "notmasked_edges"]:
        for ax in [None, 0, 1, -1, 2]:
            p = {} if ax is None else {"axis": ax}
            emit(sec, op, f"{label}.ax{ax}", [d], p, "text", lambda ops, f=getattr(ma, op), a=ax: f(ops[0], axis=a))
    for ax in [None, 0, 1, -1, 3, [0, 1]]:
        p = {} if ax is None else {"axis": ax}
        emit(sec, "compress_nd", f"{label}.ax{ax}", [d], p, "array",
             lambda ops, a=ax: ma.compress_nd(ops[0], axis=(tuple(a) if isinstance(a, list) else a)))
        if not isinstance(ax, list):
            emit(sec, "compress_rowcols", f"{label}.ax{ax}", [d], p, "array",
                 lambda ops, a=ax: ma.compress_rowcols(ops[0], axis=a))
    emit(sec, "ndenumerate", label, [d], {}, "text", lambda ops: list(ma.ndenumerate(ops[0])))
    if nd <= 2:
        emit(sec, "ndenumerate", f"{label}.full", [d], {"compressed": False}, "text",
             lambda ops: list(ma.ndenumerate(ops[0], compressed=False)))


# =============================================================================================================
# Section "apply": apply_along_axis / apply_over_axes
# =============================================================================================================

# func1d menu for apply_along_axis: each maps a masked 1-D slice to a masked result (NumPy's func1d(arr[i])).
ALONG_FUNCS = {
    "sum": lambda x: ma.sum(x),
    "max": lambda x: ma.max(x),
    "min": lambda x: ma.min(x),
    "mean": lambda x: ma.mean(x),
    "mul2": lambda x: ma.multiply(x, 2),
    "cumsum": lambda x: ma.cumsum(x),
    "rev": lambda x: x[::-1],
    "sort": lambda x: ma.sort(x),
}

# func menu for apply_over_axes: func(a, axis), a reduction (re-expanded by apply_over_axes) or a same-rank result.
OVER_FUNCS = {
    "sum": lambda x, a: ma.sum(x, axis=a),
    "max": lambda x, a: ma.max(x, axis=a),
    "mean": lambda x, a: ma.mean(x, axis=a),
    "sumkd": lambda x, a: ma.sum(x, axis=a, keepdims=True),
    "cumsum": lambda x, a: ma.cumsum(x, axis=a),
}


def gen_apply():
    """apply_along_axis (every menu function, every axis incl. negative and out of range) and apply_over_axes (int and
    list axes), on small integers so every sum is exact."""
    sec = "apply"
    runs = [(["int32", "float64", "int64", "complex128", "uint8", "bool", "float32"], [(3, 4), (2, 3, 2), (5,), (2, 0)],
             ["c"], ["none", "some", "all"]),
            (["int32", "float64"], [(3, 4), (2, 3, 2)], DATA_KINDS[1:], ["some"]),
            (["float64"], [(3, 4)], ["c"], ["first", "runs", "false"])]
    for s, shape, kind, label, d in matrix(runs, 5):
        nd = len(d["shape"])
        along = ALONG_FUNCS if kind == "c" else {k: ALONG_FUNCS[k] for k in ("sum", "mul2", "rev")}
        over = OVER_FUNCS if kind == "c" else {k: OVER_FUNCS[k] for k in ("sum", "cumsum")}
        for name, fn in along.items():
            for ax in [0, -1, 1, nd]:
                if ax == 1 and nd < 2:
                    continue
                emit(sec, "apply_along_axis", f"{label}.{name}.ax{ax}", [d], {"func": name, "axis": ax}, "masked",
                     lambda ops, f=fn, a=ax: ma.apply_along_axis(f, a, ops[0]))
        for name, fn in over.items():
            for axes in [0, -1, [0, 1], [1, 0], [nd]]:
                emit(sec, "apply_over_axes", f"{label}.{name}.{axes}", [d], {"func": name, "axes": axes}, "masked",
                     lambda ops, f=fn, ax=axes: ma.apply_over_axes(f, ops[0], ax))


SECTIONS = [gen_create, gen_join, gen_fill, gen_query, gen_putmask, gen_products, gen_edges, gen_apply]


# =============================================================================================================
# The carve
# =============================================================================================================

# Gap ids -> one line each (the [OpenBugs] replay prints the ids; OpenBugs.MaSurface.cs holds a public-call repro per
# gap that a single call can show).
GAPS = {
    "MS1": "convolve / correlate / cov / apply_along_axis drop an all-False result mask to nomask; NumPy keeps the mask "
           "array (masked_array(data, mask=mask) never shrinks)",
    "MS2": "column_stack / dstack / hsplit / diagflat over nomask operands keep nomask; NumPy's _fromnxfunction wrappers "
           "build the result mask with getmaskarray (a full all-False mask)",
    "MS3": "zeros_like / ones_like / empty_like drop the prototype's mask; NumPy's result keeps a copy of it",
    "MS4": "ma.resize to a 0-d shape keeps the mask; NumPy sets the mask only when the result has dimensions (0-d: nomask)",
    "MS5": "ma.apply_over_axes returns a masked array where NumPy returns a plain ndarray (an unmasked reduction's scalar "
           "re-expanded by expand_dims)",
    "MS6": "ma.apply_along_axis computes a narrow-integer / bool result where NumPy raises TypeError (it sets the default "
           "fill value 999999, which does not fit the dtype)",
    "MS7": "ma.masked_object leaves the dtype's default fill value; NumPy sets fill_value=value (converted to the dtype)",
    "MS8": "ma.masked_object accepts a value that does not fit the dtype; NumPy raises TypeError (Cannot convert fill_value)",
    "MS9": "ma.masked_object over a nomask array with no match keeps an all-False mask; NumPy's make_mask(shrink=True) "
           "returns nomask",
    "MS10": "np.ma.masked_object has no shrink parameter",
    "MS11": "ma.putmask casts the values unsafely; NumPy's np.copyto(where=mask) uses casting='same_kind' and raises "
            "TypeError (a float into an int array, a complex into a float array, the masked singleton's 0.0 into an int "
            "array, an int into a bool array)",
    "MS12": "ma.set_fill_value stores the raw value: no conversion to the dtype (1.5 stays 1.5 on an int array, 7 stays 7 on "
            "a bool array) and no TypeError for a value that does not fit or a str; None leaves NumSharp's raw default",
    "MS13": "ma.common_fill_value compares the raw stored values (C# Equals: 3 != 3.0) where NumPy compares the converted "
            "fill values, and inherits MS12's missing TypeErrors",
    "MS14": "minimum_fill_value / maximum_fill_value of a bool array return True / False; NumPy returns the Python ints 1 "
            "/ 0 (a bool array plus the fill promotes to int64 in NumPy, stays bool in NumSharp)",
    "MS15": "minimum_fill_value / maximum_fill_value refuse a dtype argument (NotSupportedException); NumPy reads it",
    "MS16": "np.ma.default_fill_value takes a DType only; NumPy also takes an array or a scalar (and reads its dtype)",
    "MS17": "ma.is_mask(a masked bool array) is False; NumPy's is_mask reads the dtype (bool -> True)",
    "MS18": "np.ma.size has no axis parameter",
    "MS19": "np.ma.ndenumerate has no compressed parameter (NumPy's compressed=False yields masked entries too)",
    "MS20": "creation wrappers lack NumPy parameters: ma.arange dtype= and a lone float stop, ma.zeros / ones / empty "
            "order=, ma.indices sparse=",
    "MS21": "ma.arange with a zero step raises ArgumentException('step can't be 0'); NumPy raises ZeroDivisionError",
    "MS22": "ma.frombuffer requires the WHOLE buffer to be a multiple of the item size before count / offset select a "
            "range; NumPy checks the selected range (count=0 is an empty array, a short range 'buffer is smaller than "
            "requested size')",
    "MS23": "ma.nonzero of a 0-d array returns a result; NumPy raises ValueError ('Calling nonzero on 0d arrays is not "
            "allowed')",
    "MS24": "clump_masked / clump_unmasked of a 0-d or empty array with a mask return a list; NumPy's _ezclump raises "
            "IndexError",
    "MS25": "flatnotmasked_edges / notmasked_edges of an empty array return None; NumPy returns array([0, -1])",
    "MS26": "notmasked_edges of a 0-d array with an axis raises AxisError; NumPy returns [(), ()]",
    "MS27": "notmasked_contiguous with an axis out of range raises AxisError; NumPy raises IndexError (list assignment / "
            "tuple index out of range)",
    "MS28": "compress_nd / compress_rowcols do not validate the axis: NumPy raises AxisError for an out-of-range axis even "
            "without a mask, NumSharp returns the data (nomask) or raises ArgumentException('condition must be a 1-d "
            "array')",
    "MS29": "np.ma.compress_nd takes an int axis only; NumPy also takes a tuple",
    "MS30": "ma.hsplit into parts of different shapes returns the parts; NumPy's _fromnxfunction_single wraps the list "
            "in masked_array, which raises the inhomogeneous-shape ValueError",
    "MS31": "a hard-masked setitem of several elements casts unsafely; NumPy's hard path is np.copyto(..., where=~mask) "
            "with casting='same_kind' (an int into a bool array raises TypeError)",
    "MS32": "masked setitem broadcast errors: NumSharp's text / type differ from NumPy's ('operands could not be "
            "broadcast together with shapes (1,3) (2,) ', 'could not broadcast input array from shape (2,) into shape "
            "(1,3)')",
    "MS33": "ma.corrcoef of a single variable throws ArgumentException (diag of a 0-d array); NumPy returns the masked "
            "singleton",
    "MS34": "ma.corrcoef's masked slots hold +0.0; NumPy's in-place corr /= outer(std, std) divides masked slots by 1, "
            "keeping cov's data there (-0.0, nan)",
    "MS35": "ma.convolve / ma.correlate refuse a 0-d first operand (NumPy's convolve reads it as one element; correlate "
            "raises 'object of too small depth') and word the 2-D refusal differently ('object too deep for desired "
            "array')",
    "MS37": "ma.cov / ma.corrcoef with y decide rowvar after concatenating y: a 1-D x with rowvar=False becomes ONE "
            "variable of 2N observations (cov 0-d, corrcoef throws), where NumPy forces rowvar=True for a one-row x first "
            "(two variables)",
    "MS36": "ma.ids reports the storage's buffer start, not the first element's address, for a strided or reversed view "
            "and for any view of an operand that is itself a view at an offset (NumPy's ids is ctypes.data)",
}

# The data views ids' delta is right for: a contiguous slice re-seats the storage address, so ids of such a view of a
# C operand at offset 0 is the first element's address; every other view keeps the buffer start (MS: ids, see below).
_IDS_CONTIGUOUS_VIEWS = {"tail", "mid", "rows", "row1"}


def _label(c):
    """The case label (the third field of the layout tag): '<dtype>.<kind>.<pattern>[...]' for operand cases."""
    return c["layout"].split(":", 2)[2]


def _mask_any(e):
    """Whether a masked expected record has any True mask byte."""
    m = e.get("mask") or ""
    return any(int(m[i:i + 2], 16) for i in range(0, len(m), 2))


def _err_text(c):
    """NumPy's error text of an error case ("" for a success case)."""
    return (c.get("error") or {}).get("text", "") if c.get("expects_throw") else ""


def _ns_default_repr(dtype):
    """NumSharp's default fill value as MaRepr prints it: default_fill_value returns a raw C# value, never converted to
    the array's dtype (True, the long 999999, the double 1e20, the complex 1e20+0j)."""
    k = np.dtype(dtype).kind
    if k == "b":
        return "True"
    if k in "iu":
        return "999999"
    if k == "c":
        return "(1e+20+0j)"
    return "1e+20"


def _py_param(v):
    """A JSON param as the Python object the C# side's MaPyValue boxes (the strings "nan" / "inf" / "-inf" are floats,
    {"re", "im"} a complex)."""
    if isinstance(v, dict):
        return complex(_py_param(v["re"]), _py_param(v["im"]))
    if v == "nan":
        return float("nan")
    if v == "inf":
        return float("inf")
    if v == "-inf":
        return float("-inf")
    return v


def _ns_fill_repr(v, dtype):
    """What NumSharp's ma reports for a fill value it stored raw (MS12): the value's own repr, or the raw default for
    None."""
    return _ns_default_repr(dtype) if v is None else pyrepr(v)


def _ns_fill_key(v, dtype):
    """The (CLR type, value) NumSharp compares in common_fill_value (C# Equals on the boxed values: a long and a double
    never compare equal; a double NaN equals itself)."""
    if v is None:
        k = np.dtype(dtype).kind
        return ("bool", True) if k == "b" else ("long", 999999) if k in "iu" else \
               ("complex", complex(1e20, 0)) if k == "c" else ("double", 1e20)
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, int):
        return ("long" if -2 ** 63 <= v < 2 ** 63 else "bigint", v)
    if isinstance(v, float):
        return ("double", "nan" if v != v else v)
    if isinstance(v, complex):
        return ("complex", v)
    return ("str", v)


def _gap_of(c):
    """The gap id a case belongs to, or None to keep it.

    Each rule is a fixed predicate over the case (op, params, operand kinds and masks, NumPy's recorded outcome) or a model
    of NumSharp's behavior where the gap is not a fixed set (MS12 / MS13: NumSharp's raw fill values; ids: which views
    re-seat the storage address). The carve is exact: replayed uncarved, every carved case diverges and no kept case does
    (README "The numpy.ma surface tier").
    """
    op = c["op"][3:]
    p = c["params"]
    e = c.get("expected") or {}
    err = _err_text(c)
    label = _label(c)
    parts = label.split(".")
    ops = c["operands"]
    o0_mask = bool(ops) and "mask" in ops[0]
    if op in ("convolve", "correlate") and ops and (len(ops[0]["shape"]) == 0 or len(ops[0]["shape"]) >= 2):
        return "MS35"
    if op in ("cov", "corrcoef") and p.get("y") and len(ops[0]["shape"]) == 1 and not p.get("rowvar", True):
        return "MS37"
    if op in ("convolve", "correlate", "cov", "apply_along_axis") and not err and e.get("form") == "mask" \
            and not _mask_any(e):
        return "MS1"
    if op in ("column_stack", "dstack", "diagflat") and not err and e.get("form") == "mask" \
            and not any("mask" in o for o in ops):
        return "MS2"
    if op == "hsplit" and not err and not o0_mask and any(s.get("form") == "mask" for s in e.get("slots", [])):
        return "MS2"
    if op in ("zeros_like", "ones_like", "empty_like") and o0_mask and not p.get("plain"):
        return "MS3"
    if op == "resize" and p.get("shape") == [] and o0_mask and not err:
        return "MS4"
    if op == "apply_over_axes" and e.get("form") == "scalar" and len(e.get("shape", [])) >= 1:
        return "MS5"
    if op == "apply_along_axis" and err.startswith("Cannot convert fill_value"):
        return "MS6"
    if op == "masked_object":
        if "shrink" in p:
            return "MS10"
        if err:
            return "MS8"
        if p.get("facet") == "fill_value":
            return "MS7" if e.get("value") != _ns_default_repr(ops[0]["dtype"]) else None
        if not o0_mask and e.get("form") in ("nomask", "scalar"):
            return "MS9"
    if op == "putmask" and err.startswith("Cannot cast"):
        return "MS11"
    if op == "set_fill_value":
        if p.get("plain"):
            return None
        v = _py_param(p["value"])
        if err or e.get("value") != _ns_fill_repr(v, ops[0]["dtype"]):
            return "MS12"
    if op == "common_fill_value":
        if err:
            return "MS13"
        fa, fb = _py_param(p["fa"]), _py_param(p["fb"])
        ka, kb = _ns_fill_key(fa, ops[0]["dtype"]), _ns_fill_key(fb, ops[1]["dtype"])
        ns = _ns_fill_repr(fa, ops[0]["dtype"]) if ka == kb else "None"
        if e.get("value") != ns:
            return "MS13"
    if op in ("minimum_fill_value", "maximum_fill_value"):
        if "dtype" in p:
            return "MS15"
        if (ops and ops[0]["dtype"] == "bool") or p.get("value") is True:
            return "MS14"
    if op == "default_fill_value" and "dtype" not in p:
        return "MS16"
    if op == "is_mask" and p.get("arg") == "masked" and ops[0]["dtype"] == "bool":
        return "MS17"
    if op == "size" and "axis" in p:
        return "MS18"
    if op == "ndenumerate" and "compressed" in p:
        return "MS19"
    if (op == "arange" and ("dtype" in p or (len(p["args"]) == 1 and not isinstance(p["args"][0], int)))) \
            or (op in ("zeros", "ones", "empty") and "order" in p) or (op == "indices" and "sparse" in p):
        return "MS20"
    if op == "arange" and c.get("error", {}).get("type") == "ZeroDivisionError":
        return "MS21"
    if op == "frombuffer" and p.get("dtype") == "complex128" and p["count"] >= 0 and 0 <= p["offset"] <= 24:
        return "MS22"
    if op == "nonzero" and err:
        return "MS23"
    if op in ("clump_masked", "clump_unmasked") and err:
        return "MS24"
    if op in ("flatnotmasked_edges", "notmasked_edges") and e.get("value") == "array([ 0, -1])":
        return "MS25"
    if op == "notmasked_edges" and e.get("value") == "[(), ()]":
        return "MS26"
    if op == "notmasked_contiguous" and c.get("error", {}).get("type") == "IndexError":
        return "MS27"
    if op in ("compress_nd", "compress_rowcols"):
        if isinstance(p.get("axis"), list):
            return "MS29"
        if c.get("error", {}).get("type") == "AxisError":
            return "MS28"
    if op == "hsplit" and "inhomogeneous" in err:
        return "MS30"
    if op in ("harden_mask", "soften_mask") and p.get("facet") == "assign" and err:
        if err.startswith("Cannot cast"):
            return "MS31"
        # NumSharp's setitem raises NumPy's soft-path text for a C operand ("could not broadcast input array from
        # shape (2,) into shape (1,3)"); NumPy's hard path words it "operands could not be broadcast together ...", and
        # a non-C operand gets NumSharp's IncorrectShapeException text.
        if "broadcast" in err and (err.startswith("operands could not be broadcast together") or parts[1] != "c"):
            return "MS32"
    if op == "corrcoef":
        if e.get("form") == "masked":
            return "MS33"
        if _corrcoef_masked_raw_nonzero(e):
            return "MS34"
    if op == "ids":
        delta = int.from_bytes(bytes.fromhex(e["buffer"])[:8], "little", signed=True)
        view = parts[3]
        if delta != 0 and not (parts[1] == "c" and view in _IDS_CONTIGUOUS_VIEWS):
            return "MS36"
    return None


def _corrcoef_masked_raw_nonzero(e):
    """Whether a corrcoef result keeps -0.0 at a masked slot (MS34: NumSharp writes +0.0 there; a NaN slot compares
    tokenized and matches)."""
    if e.get("kind") != "masked" or "raw" not in e or not _mask_any(e):
        return False
    raw = bytes.fromhex(e["raw"])
    mask = bytes.fromhex(e["mask"])
    negzero = bytes(7) + b"\x80"
    return any(mask[i] and raw[8 * i:8 * i + 8] == negzero for i in range(len(mask)))


def write_jsonl(path, cases):
    """Write ``cases`` one JSON object per line (LF, no NaN literals: every non-finite param is a string)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"wrote {len(cases)} cases -> {os.path.basename(path)}")


def main():
    """Generate the numpy.ma surface tier (see the module docstring for the flags)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--uncarved", metavar="DIR", help="write the whole matrix (no carve) into DIR instead")
    args = ap.parse_args()
    for section in SECTIONS:
        section()
    counts = collections.Counter(c["op"] for c in CASES)
    print(f"{len(CASES)} cases over {len(counts)} op keys")
    if args.uncarved:
        write_jsonl(os.path.join(args.uncarved, "ma_surface.jsonl"), CASES)
        return
    kept, carved = [], []
    for c in CASES:
        gap = _gap_of(c)
        if gap:
            c["gap"] = gap
            carved.append(c)
        else:
            kept.append(c)
    write_jsonl(os.path.join(CORPUS, "ma_surface.jsonl"), kept)
    write_jsonl(os.path.join(GAP_DIR, "ma_surface_gaps.jsonl"), carved)


if __name__ == "__main__":
    main()
