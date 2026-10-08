#!/usr/bin/env python3
"""
gen_surface_oracle.py -- oracle contracts for the APIs the coverage inventory lists as AVAILABLE but UNPROVEN.

The coverage artifact (coverage/generate_coverage.py over NumSharp.Tools.ApiInventory) joins every committed NumPy
oracle contract onto its API row. On 2026-10-08 it listed 40 headline APIs (and about 200 object attributes) that
NumSharp exposes but no contract proves: the np.linalg Array-API forms, the histogram family, array_equiv, the memory
predicates, broadcast_shapes, apply_along_axis / apply_over_axes, bmat, the text formatters, the ndarray attributes and
methods tolist / tofile / setfield / base / dtype / shape / device / to_device / data, the print and buffer options,
frompyfunc, the cumulative_* aliases and numpy.shape, and the members of finfo, iinfo, dtype, ndarray.flags, poly1d,
np.broadcast, flatiter and np.nditer. Most were classified "sibling-owned" because the operand corpus had no form for
them (a tuple, a text, a shape list, a function argument, two views of one buffer, an object stepped k times); the
harness has had tuple, text, scalar and dtype results for a while, and this generator supplies the rest through params:

  * a FUNCTION argument is a name from a fixed menu both sides map to the same computation (APPLY_FUNCS, OVER_FUNCS,
    FROMPY_FUNCS);
  * two VIEWS of one buffer are recipes (VIEW_PAIRS, apply_steps) both sides apply to the same owner operand;
  * an object's state is its construction params plus a number of steps (the np.broadcast / flatiter / np.nditer
    members), and global options are set and restored inside the case (print options, buffer size);
  * a SHAPE list, a dtype string, a format option or a Python int is a param.

Every array operand is built in several ndarray KINDS (C, F, permuted, stepped and C views at an offset, reversed,
stepped-and-reversed, read-only, broadcast, 0-d, empty, an empty view inside a buffer) through the relayout tier's
transforms, so each API also meets the layouts the earlier tiers proved risky. NumPy computes every expected record
(or its exact error type and text) on operands rebuilt from the very descriptors NumSharp reconstructs.

Output (../NumSharp.Tests.Oracle/Fuzz/corpus/):

  surface.jsonl                 replayed by FuzzCorpusTests.Surface (valueclass "surface": ZERO registry excuses)
  gaps/surface_gaps.jsonl       the cases a carve rule (_gap_of) names: NumSharp is known to differ there; replayed by
                                an [OpenBugs] test that names each gap

Run (needs numpy==2.4.2; deterministic: a rerun is byte-identical):
    python test/oracle/gen_surface_oracle.py [--uncarved DIR]
"""
import argparse
import collections
import itertools
import json
import os
import sys
import tempfile
import warnings

import numpy as np

np.seterr(all="ignore")
warnings.simplefilter("ignore")

assert np.__version__ == "2.4.2", f"corpus must be generated with numpy 2.4.2, got {np.__version__}"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from layout_catalog import describe  # noqa: E402
import gen_relayout_oracle as rl  # noqa: E402 -- the kind transforms, kind_of and rebuild, shared on purpose

FUZZ = os.path.normpath(os.path.join(HERE, "..", "NumSharp.Tests.Oracle", "Fuzz"))
CORPUS = os.path.join(FUZZ, "corpus")
GAP_DIR = os.path.join(CORPUS, "gaps")

INTS = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
FLOATS = ["float16", "float32", "float64"]
ALL13 = ["bool"] + INTS + FLOATS + ["complex128"]

# =============================================================================================================
# Values and operands
# =============================================================================================================


def values(shape, dtype, seed=0, lo=-3, hi=5):
    """Deterministic values of ``shape`` in ``dtype``: small integers (with fractions for float / complex), so every
    dtype holds them exactly and a wrong element read shows as a wrong value.

    Parameters:
        shape: the array shape.
        dtype: a NumPy dtype name.
        seed: varies the pattern between operands.
        lo, hi: the integer range (unsigned dtypes take abs).

    Returns:
        A C-contiguous NumPy array.
    """
    n = int(np.prod(shape)) if len(shape) else 1
    k = np.arange(n, dtype=np.int64)
    span = hi - lo + 1
    ints = ((k * 7 + seed * 3) % span) + lo
    dt = np.dtype(dtype)
    if dt.kind == "b":
        return ((k * 3 + seed) % 3 != 0).reshape(shape)
    if dt.kind == "u":
        return np.abs(ints).astype(dt).reshape(shape)
    if dt.kind == "i":
        return ints.astype(dt).reshape(shape)
    if dt.kind == "f":
        return (ints + ((k + seed) % 4) * 0.25).astype(dt).reshape(shape)
    if dt.kind == "c":
        return (ints + 1j * (((k * 5 + seed) % span) + lo) * 0.5).astype(dt).reshape(shape)
    raise ValueError(dtype)


# Kinds an operand can be rebuilt in. "c" is a view over a C buffer at offset 0 (how every corpus operand is rebuilt);
# "own" is an owner (descriptor flag own: true); the rest are the relayout tier's transforms.
VALUE_KINDS = ["c", "c_offset", "f", "f_offset", "permuted", "permuted_offset", "reversed", "strided",
               "strided_offset", "strided_neg", "readonly"]
DERIVED_KINDS = ["bcast", "bcast_offset", "scalar_bcast", "scalar_bcast_offset", "empty", "empty_view"]


def operand(vals, name):
    """One operand descriptor holding ``vals`` in ndarray kind ``name``.

    Parameters:
        vals: the logical values (a NumPy array).
        name: "c", "own" or a gen_relayout_oracle transform name.

    Returns:
        (kind, descriptor), or None when the transform does not apply to this shape (an F view of a 1-D array, a
        permuted view of a 2-D one, a stepped view of a 0-d one).
    """
    v = np.asarray(vals)
    if name in ("c", "own"):
        # np.array(copy, order='C'), not np.ascontiguousarray: the latter returns a (1,) array for a 0-d input.
        base = np.array(v, copy=True, order="C")
        d = describe(base, base)
        if name == "own":
            d["own"] = True
        return ("own" if name == "own" else rl.kind_of(d)[0]), d
    builder, min_nd, _ = rl.TRANSFORMS[name]
    if v.ndim < min_nd or (v.size == 0 and name not in ("empty", "empty_view")):
        return None
    try:
        base, view = builder(v)
        d = describe(base, view)
    except Exception:  # noqa: BLE001 -- the transform does not apply to this operand
        return None
    if name in rl.OWNING_TRANSFORMS:
        d["own"] = True
    kind = "readonly" if name == "readonly" else rl.kind_of(d)[0]
    if kind in ("c",) and name not in ("c",):
        return None  # e.g. a strided view of a one-element axis collapses to C: nothing new
    return kind, d


def kinds_of(vals, names):
    """Every applicable (kind, descriptor) of ``vals`` among ``names``, one per distinct kind."""
    out, seen = [], set()
    for name in names:
        r = operand(vals, name)
        if r is None or r[0] in seen:
            continue
        seen.add(r[0])
        out.append(r)
    return out


def np_ops(descs):
    """The NumPy operands the descriptors name, rebuilt exactly as the C# side rebuilds them."""
    return [rl.rebuild(d) for d in descs]


# =============================================================================================================
# Case emission
# =============================================================================================================

CASES = []
_N = itertools.count()


def _array_record(r):
    a = np.asarray(r)
    return {"dtype": a.dtype.name, "shape": [int(x) for x in a.shape], "buffer": np.ascontiguousarray(a).tobytes().hex()}


def emit(section, op, layout, descs, params, kind, fn):
    """Run ``fn`` (NumPy's call over operands rebuilt from ``descs``) and record the case.

    Parameters:
        section: the generator section (part of the id).
        op: the corpus op key (resolved by the coverage join and dispatched by OpRegistry.Surface.cs).
        layout: a short label of the operand kinds / variant (part of the id and the layout tag).
        descs: the operand descriptors.
        params: the case params (JSON-serializable).
        kind: the expected result kind: "array", "scalar", "tuple", "text" or "dtype".
        fn: callable(ops) -> NumPy's result: an array / scalar, a list of arrays (tuple), a str (text), or a
            dtype (dtype); it may raise (an error case, recorded verbatim).
    """
    n = next(_N)
    cid = f"surface/{section}/{op}/{layout}/{n}"
    case = {"id": cid, "op": op, "params": params, "operands": descs, "layout": f"surface:{section}:{layout}",
            "valueclass": "surface"}
    try:
        r = fn(np_ops(descs))
    except Exception as e:  # noqa: BLE001 -- NumPy's refusal is the outcome
        case["expected"] = {} if kind == "array" else {"kind": kind}
        case["expects_throw"] = True
        case["error"] = {"type": type(e).__name__, "text": str(e)}
        CASES.append(case)
        return case
    if kind == "array":
        case["expected"] = _array_record(r)
    elif kind == "scalar":
        case["expected"] = {"kind": "scalar", **_array_record(r)}
    elif kind == "tuple":
        case["expected"] = {"kind": "tuple", "slots": [_array_record(x) for x in r]}
    elif kind == "text":
        case["expected"] = {"kind": "text", "value": r}
    elif kind == "dtype":
        case["expected"] = {"kind": "dtype", "value": np.dtype(r).name}
    else:
        raise ValueError(kind)
    CASES.append(case)
    return case


# =============================================================================================================
# S1-S4: the np.linalg Array-API forms (last-two-axes / 1-D / 3-vector contracts, not the np.* forms)
# =============================================================================================================


def gen_linalg():
    """np.linalg.cross / diagonal / outer / trace over every dtype, operand kinds and their refusals."""
    # cross: (3,) vectors, stacks, a broadcast pair, axis=0, and the refusals.
    for dt in ["int8", "int16", "int32", "int64", "uint8", "uint32", "float16", "float32", "float64", "complex128",
               "bool"]:
        a, b = values((4, 3), dt, 1), values((4, 3), dt, 2)
        for kind, d in kinds_of(a, VALUE_KINDS + ["bcast", "scalar_bcast"]):
            emit("linalg", "linalg.cross", f"{dt}:{kind}@0", [d, operand(b, "c")[1]], {"axis": -1}, "array",
                 lambda o: np.linalg.cross(o[0], o[1], axis=-1))
        for kind, d in kinds_of(b, ["c_offset", "reversed", "strided_neg"]):
            emit("linalg", "linalg.cross", f"{dt}:{kind}@1", [operand(a, "c")[1], d], {"axis": -1}, "array",
                 lambda o: np.linalg.cross(o[0], o[1], axis=-1))
        emit("linalg", "linalg.cross", f"{dt}:vec", [operand(a[0], "c")[1], operand(b[1], "c")[1]], {"axis": -1},
             "array", lambda o: np.linalg.cross(o[0], o[1]))
        emit("linalg", "linalg.cross", f"{dt}:bcast_pair", [operand(a[:1], "c")[1], operand(b, "c")[1]],
             {"axis": -1}, "array", lambda o: np.linalg.cross(o[0], o[1]))
        emit("linalg", "linalg.cross", f"{dt}:axis0", [operand(a.T.copy(), "c")[1], operand(b.T.copy(), "c")[1]],
             {"axis": 0}, "array", lambda o: np.linalg.cross(o[0], o[1], axis=0))
        emit("linalg", "linalg.cross", f"{dt}:stack3d", [operand(values((2, 4, 3), dt, 3), "c")[1],
                                                         operand(b, "c")[1]],
             {"axis": -1}, "array", lambda o: np.linalg.cross(o[0], o[1]))
    i64 = values((4, 3), "int64")
    for label, x, y, axis in [("vec2", values((2,), "int64"), values((2,), "int64"), -1),
                              ("vec2x3", values((4, 2), "int64"), values((4, 3), "int64"), -1),
                              ("bcast_bad", i64, values((5, 3), "int64"), -1),
                              ("axis5", i64, i64, 5), ("axis_neg3", i64, i64, -3),
                              ("zerod", np.array(3), i64, -1),
                              ("empty_stack", values((0, 3), "float64"), values((0, 3), "float64"), -1),
                              ("mixed_i8_f4", values((4, 3), "int8"), values((4, 3), "float32", 1), -1),
                              ("mixed_u8_i8", values((4, 3), "uint8"), values((4, 3), "int8", 1), -1),
                              ("wrap_i8", np.full((2, 3), 100, np.int8), np.array([[100, -100, 7], [1, 2, 3]],
                                                                                  np.int8), -1)]:
        emit("linalg", "linalg.cross", label, [operand(x, "c")[1], operand(y, "c")[1]], {"axis": axis}, "array",
             lambda o, axis=axis: np.linalg.cross(o[0], o[1], axis=axis))

    # diagonal: the LAST two axes, any offset, every kind; a view in NumPy (values compared here).
    for dt in ALL13:
        for shape in [(3, 4), (2, 3, 4), (4, 3)]:
            x = values(shape, dt, 4)
            for kind, d in kinds_of(x, VALUE_KINDS + ["bcast", "bcast_offset", "empty_view"]):
                for off in ([0] if kind != "c" else [-3, -1, 0, 1, 2, 9]):
                    emit("linalg", "linalg.diagonal", f"{dt}:{shape}:{kind}:off{off}", [d], {"offset": off}, "array",
                         lambda o, off=off: np.linalg.diagonal(o[0], offset=off))
    for label, x in [("1d", values((3,), "int32")), ("0d", np.array(2.5)), ("4d", values((2, 2, 3, 3), "int16")),
                     ("empty_rows", values((0, 3), "float64")), ("empty_cols", values((3, 0), "int8"))]:
        for off in [0, 1, -1]:
            emit("linalg", "linalg.diagonal", f"{label}:off{off}", [operand(x, "c")[1]], {"offset": off}, "array",
                 lambda o, off=off: np.linalg.diagonal(o[0], offset=off))

    # outer: 1-D only; the result dtype is the multiply's promotion.
    for dt in ALL13:
        a, b = values((4,), dt, 5), values((3,), dt, 6)
        for kind, d in kinds_of(a, ["c", "c_offset", "reversed", "strided", "strided_offset", "strided_neg",
                                    "readonly", "scalar_bcast", "scalar_bcast_offset", "one_1d", "empty",
                                    "empty_view"]):
            emit("linalg", "linalg.outer", f"{dt}:{kind}@0", [d, operand(b, "c")[1]], {}, "array",
                 lambda o: np.linalg.outer(o[0], o[1]))
        for kind, d in kinds_of(b, ["c_offset", "strided_neg", "empty"]):
            emit("linalg", "linalg.outer", f"{dt}:{kind}@1", [operand(a, "c")[1], d], {}, "array",
                 lambda o: np.linalg.outer(o[0], o[1]))
    for label, x, y in [("2d", values((2, 2), "float64"), values((3,), "float64")),
                        ("0d", np.array(2.0), values((3,), "float64")),
                        ("both2d", values((2, 2), "int32"), values((2, 2), "int32")),
                        ("i8_f2", values((3,), "int8"), values((2,), "float16")),
                        ("u8_i16", values((3,), "uint8"), values((2,), "int16")),
                        ("b_c", values((3,), "bool"), values((2,), "complex128"))]:
        emit("linalg", "linalg.outer", label, [operand(x, "c")[1], operand(y, "c")[1]], {}, "array",
             lambda o: np.linalg.outer(o[0], o[1]))

    # trace: the LAST two axes, NumPy's sum promotion, dtype=.
    for dt in ALL13:
        for shape in [(3, 3), (2, 3, 4)]:
            x = values(shape, dt, 7)
            for kind, d in kinds_of(x, VALUE_KINDS + ["bcast", "empty_view"]):
                emit("linalg", "linalg.trace", f"{dt}:{shape}:{kind}", [d], {"offset": 0, "dtype": None}, "array",
                     lambda o: np.linalg.trace(o[0]))
        x = values((3, 4), dt, 8)
        for off in [-2, -1, 1, 3, 7]:
            emit("linalg", "linalg.trace", f"{dt}:off{off}", [operand(x, "c")[1]], {"offset": off, "dtype": None},
                 "array", lambda o, off=off: np.linalg.trace(o[0], offset=off))
        for req in ["int8", "uint8", "int64", "float32", "float64", "complex128", "bool"]:
            emit("linalg", "linalg.trace", f"{dt}:dtype={req}", [operand(x, "c")[1]], {"offset": 0, "dtype": req},
                 "array", lambda o, req=req: np.linalg.trace(o[0], dtype=req))
    for label, x in [("1d", values((3,), "int32")), ("0d", np.array(2)), ("empty", values((0, 0), "float64")),
                     ("empty_stack", values((2, 0, 0), "int64")), ("wrap_i8", np.full((3, 3), 100, np.int8))]:
        emit("linalg", "linalg.trace", label, [operand(x, "c")[1]], {"offset": 0, "dtype": None}, "array",
             lambda o: np.linalg.trace(o[0]))


# =============================================================================================================
# S5-S8: the histogram family (tuples over int / estimator / edge-array bins, range, density, weights)
# =============================================================================================================

ESTIMATORS = ["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]


def _jf(x):
    """A float for a JSON param: the number itself, or "nan" / "inf" / "-inf" (strict JSON has no non-finite numbers;
    OpRegistry.Surface.PyFloat reads both forms)."""
    return x if np.isfinite(x) else repr(float(x))


def _hist_params(bins=None, bins_operand=None, rng=None, density=False, weights_operand=None):
    p = {"bins": bins, "range": None if rng is None else [_jf(r) for r in rng], "density": density}
    if bins_operand is not None:
        p["bins_operand"] = bins_operand
    if weights_operand is not None:
        p["weights_operand"] = weights_operand
    return p


def _hist_call(fn, o, p, density=True):
    kw = {}
    kw["bins"] = o[p["bins_operand"]] if "bins_operand" in p else (10 if p["bins"] is None else p["bins"])
    if p["range"] is not None:
        kw["range"] = tuple(float(r) for r in p["range"])
    if "weights_operand" in p:
        kw["weights"] = o[p["weights_operand"]]
    if density:
        kw["density"] = p["density"]
    return fn(o[0], **kw)


def gen_histogram():
    """np.histogram / np.histogram_bin_edges: every dtype, int and estimator bins, edge arrays, range, density,
    weights, operand kinds, the degenerate samples and every refusal."""
    sample = np.array([0.5, 1.5, 1.5, 2.25, 3.0, 9.0, -1.0, 4.5, 4.5, 4.5, 0.0, 2.0])
    for dt in ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16",
               "float32", "float64", "complex128"]:
        if dt == "bool":
            x = sample > 1.0
        elif np.dtype(dt).kind == "u":
            x = np.abs(sample).astype(dt)
        else:
            x = sample.astype(dt)
        for bins in [1, 3, 10]:
            p = _hist_params(bins)
            emit("histogram", "histogram", f"{dt}:bins{bins}", [operand(x, "c")[1]], p, "tuple",
                 lambda o, p=p: list(_hist_call(np.histogram, o, p)))
            emit("histogram", "histogram_bin_edges", f"{dt}:bins{bins}", [operand(x, "c")[1]], p, "array",
                 lambda o, p=p: _hist_call(np.histogram_bin_edges, o, p, density=False))
        for est in ESTIMATORS:
            p = _hist_params(est)
            emit("histogram", "histogram", f"{dt}:{est}", [operand(x, "c")[1]], p, "tuple",
                 lambda o, p=p: list(_hist_call(np.histogram, o, p)))
        p = _hist_params(4, rng=[0.0, 5.0])
        emit("histogram", "histogram", f"{dt}:range", [operand(x, "c")[1]], p, "tuple",
             lambda o, p=p: list(_hist_call(np.histogram, o, p)))
        p = _hist_params(4, density=True)
        emit("histogram", "histogram", f"{dt}:density", [operand(x, "c")[1]], p, "tuple",
             lambda o, p=p: list(_hist_call(np.histogram, o, p)))
        edges = np.array([-1.0, 0.0, 1.5, 4.5, 9.0])
        p = _hist_params(bins_operand=1)
        emit("histogram", "histogram", f"{dt}:edges", [operand(x, "c")[1], operand(edges, "c")[1]], p, "tuple",
             lambda o, p=p: list(_hist_call(np.histogram, o, p)))
        w = values(x.shape, "float64", 3)
        p = _hist_params(3, weights_operand=1)
        emit("histogram", "histogram", f"{dt}:weights", [operand(x, "c")[1], operand(w, "c")[1]], p, "tuple",
             lambda o, p=p: list(_hist_call(np.histogram, o, p)))
    # operand kinds of the sample (2-D samples are flattened), and of the weights / edges
    x2 = sample.reshape(3, 4)
    for kind, d in kinds_of(x2, VALUE_KINDS + DERIVED_KINDS):
        for bins in [4, "auto"]:
            p = _hist_params(bins)
            emit("histogram", "histogram", f"kind:{kind}:{bins}", [d], p, "tuple",
                 lambda o, p=p: list(_hist_call(np.histogram, o, p)))
        p = _hist_params(3)
        emit("histogram", "histogram_bin_edges", f"kind:{kind}", [d], p, "array",
             lambda o, p=p: _hist_call(np.histogram_bin_edges, o, p, density=False))
    w2 = values((3, 4), "int32", 2)
    for kind, d in kinds_of(w2, ["c_offset", "f", "reversed", "strided_neg", "bcast"]):
        p = _hist_params(3, weights_operand=1)
        emit("histogram", "histogram", f"weights:{kind}", [operand(x2, "c")[1], d], p, "tuple",
             lambda o, p=p: list(_hist_call(np.histogram, o, p)))
    edges = np.array([-1.0, 0.0, 1.5, 4.5, 9.0])
    for kind, d in kinds_of(edges, ["c_offset", "strided", "strided_neg", "readonly"]):
        p = _hist_params(bins_operand=1)
        emit("histogram", "histogram", f"edges:{kind}", [operand(sample, "c")[1], d], p, "tuple",
             lambda o, p=p: list(_hist_call(np.histogram, o, p)))
    # degenerate samples and refusals
    for label, x, p, extra in [
            ("all_equal", np.full(5, 2.0), _hist_params(4), []),
            ("all_equal_auto", np.full(5, 2.0), _hist_params("auto"), []),
            ("single", np.array([3.0]), _hist_params(3), []),
            ("empty", np.zeros(0), _hist_params(3), []),
            ("empty_auto", np.zeros(0), _hist_params("auto"), []),
            ("zerod", np.array(2.5), _hist_params(2), []),
            ("big_int64", np.array([2 ** 62, 2 ** 62 + 4096, -2 ** 62], np.int64), _hist_params(4), []),
            ("nan", np.array([1.0, np.nan, 2.0]), _hist_params(3), []),
            ("inf", np.array([1.0, np.inf]), _hist_params(3), []),
            ("nan_range", np.array([1.0, np.nan, 2.0]), _hist_params(3, rng=[0.0, 3.0]), []),
            ("bins0", sample, _hist_params(0), []),
            ("bins_neg", sample, _hist_params(-2), []),
            ("bogus", sample, _hist_params("bogus"), []),
            ("range_bad", sample, _hist_params(3, rng=[3.0, 1.0]), []),
            ("range_nan", sample, _hist_params(3, rng=[float("nan"), 1.0]), []),
            ("range_equal", sample, _hist_params(3, rng=[2.0, 2.0]), []),
            ("edges_desc", sample, _hist_params(bins_operand=1), [np.array([3.0, 1.0, 2.0])]),
            ("edges_2d", sample, _hist_params(bins_operand=1), [np.array([[0.0, 1.0], [2.0, 3.0]])]),
            ("weights_shape", sample, _hist_params(3, weights_operand=1), [np.ones(3)]),
            ("weights_density", sample, _hist_params(3, density=True, weights_operand=1), [values((12,), "float32")]),
            ("weights_complex", sample, _hist_params(3, weights_operand=1), [values((12,), "complex128")]),
            ("weights_bool", sample, _hist_params(3, weights_operand=1), [values((12,), "bool")]),
            ("edges_int", sample, _hist_params(bins_operand=1), [np.array([-1, 0, 2, 9], np.int64)]),
            ("estimator_range", sample, _hist_params("fd", rng=[0.0, 4.0]), []),
            ("estimator_weights", sample, _hist_params("auto", weights_operand=1), [np.ones(12)])]:
        descs = [operand(x, "c")[1]] + [operand(e, "c")[1] for e in extra]
        emit("histogram", "histogram", label, descs, p, "tuple",
             lambda o, p=p: list(_hist_call(np.histogram, o, p)))

    # histogram2d: two 1-D samples; bins int / pair / edges / pair of edges; range; density; weights.
    xs = np.array([0.5, 1.5, 1.5, 2.25, 3.0, 4.0])
    ys = np.array([1.0, 1.0, 2.5, 2.0, 0.5, 3.0])
    for dt in ["int16", "int64", "uint8", "float16", "float32", "float64"]:
        x, y = (np.abs(xs) if np.dtype(dt).kind == "u" else xs).astype(dt), ys.astype(dt)
        for label, p, extra in [("bins2", {"bins": 2}, []), ("bins_pair", {"bins": [2, 3]}, []),
                                ("range", {"bins": 2, "range": [[0.0, 4.0], [0.0, 3.0]]}, []),
                                ("density", {"bins": 3, "density": True}, []),
                                ("weights", {"bins": 2, "weights_operand": 2}, [values((6,), "float64", 4)]),
                                ("edges", {"bins_operand": 2}, [np.array([0.0, 1.0, 2.0, 5.0])]),
                                ("edges_pair", {"bins_operands": [2, 3]}, [np.array([0.0, 2.0, 5.0]),
                                                                           np.array([0.0, 1.0, 1.5, 4.0])])]:
            descs = [operand(x, "c")[1], operand(y, "c")[1]] + [operand(e, "c")[1] for e in extra]
            emit("histogram", "histogram2d", f"{dt}:{label}", descs, p, "tuple",
                 lambda o, p=p: list(_hist2d(o, p)))
    for kind, d in kinds_of(xs, ["c_offset", "reversed", "strided", "strided_neg", "readonly", "scalar_bcast",
                                 "empty", "empty_view"]):
        yy = ys[:0] if kind.startswith("empty") else ys
        emit("histogram", "histogram2d", f"kind:{kind}", [d, operand(yy, "c")[1]], {"bins": 2}, "tuple",
             lambda o: list(_hist2d(o, {"bins": 2})))
    for label, x, y, p in [("len_mismatch", xs, ys[:4], {"bins": 2}),
                           # A bins list whose length is not 2 is one edge list for BOTH axes. Floats on purpose: a
                           # Python int list is int64 edges, a C# int[] int32 ones (the language-wide literal width).
                           ("bins_edges3", xs, ys, {"bins_edges": [0.0, 2.0, 5.0]}),
                           ("nan", np.array([1.0, np.nan]), np.array([1.0, 2.0]), {"bins": 2})]:
        emit("histogram", "histogram2d", label, [operand(x, "c")[1], operand(y, "c")[1]], p, "tuple",
             lambda o, p=p: list(_hist2d(o, p)))

    # histogramdd: an (N, D) sample or a list of D 1-D samples; bins int / per-axis ints / per-axis edges.
    sd = np.stack([xs, ys, xs[::-1] * 0.5], axis=1)
    for dt in ["int32", "int64", "uint16", "float16", "float32", "float64"]:
        s = (np.abs(sd) if np.dtype(dt).kind == "u" else sd).astype(dt)
        for label, p, extra in [("bins2", {"bins": 2}, []), ("bins_list", {"bins": [2, 3, 1]}, []),
                                ("range", {"bins": 2, "range": [[0.0, 4.0], [0.0, 3.0], [0.0, 2.0]]}, []),
                                ("density", {"bins": 2, "density": True}, []),
                                ("weights", {"bins": 2, "weights_operand": 1}, [values((6,), "float64", 5)]),
                                ("edges_list", {"bins_operands": [1, 2, 3]},
                                 [np.array([0.0, 2.0, 5.0]), np.array([0.0, 1.0, 4.0]), np.array([0.0, 3.0])])]:
            descs = [operand(s, "c")[1]] + [operand(e, "c")[1] for e in extra]
            emit("histogram", "histogramdd", f"{dt}:{label}", descs, p, "tuple",
                 lambda o, p=p: _histdd(o, p))
        p = {"bins": 2, "sample_list": True}
        emit("histogram", "histogramdd", f"{dt}:sample_list",
             [operand(s[:, 0].copy(), "c")[1], operand(s[:, 1].copy(), "c")[1]], p, "tuple",
             lambda o, p=p: _histdd(o, p))
    for kind, d in kinds_of(sd, VALUE_KINDS + ["bcast", "empty", "empty_view"]):
        emit("histogram", "histogramdd", f"kind:{kind}", [d], {"bins": 2}, "tuple",
             lambda o: _histdd(o, {"bins": 2}))
    for label, s, p in [("bins_wrong_len", sd, {"bins": [2, 3]}), ("one_d_sample", xs, {"bins": 3}),
                        ("bins0", sd, {"bins": 0}), ("nan", np.array([[1.0, np.nan], [2.0, 3.0]]), {"bins": 2})]:
        emit("histogram", "histogramdd", label, [operand(s, "c")[1]], p, "tuple", lambda o, p=p: _histdd(o, p))


def _hist2d(o, p):
    kw = {}
    if "bins_operand" in p:
        kw["bins"] = o[p["bins_operand"]]
    elif "bins_operands" in p:
        kw["bins"] = [o[i] for i in p["bins_operands"]]
    elif "bins_edges" in p:
        kw["bins"] = list(p["bins_edges"])
    else:
        kw["bins"] = p["bins"]
    if p.get("range") is not None:
        kw["range"] = [tuple(r) for r in p["range"]]
    if p.get("density"):
        kw["density"] = True
    if "weights_operand" in p:
        kw["weights"] = o[p["weights_operand"]]
    h, xe, ye = np.histogram2d(o[0], o[1], **kw)
    return [h, xe, ye]


def _histdd(o, p):
    kw = {}
    if "bins_operands" in p:
        kw["bins"] = [o[i] for i in p["bins_operands"]]
    else:
        kw["bins"] = p["bins"]
    if p.get("range") is not None:
        kw["range"] = [tuple(r) for r in p["range"]]
    if p.get("density"):
        kw["density"] = True
    if "weights_operand" in p:
        kw["weights"] = o[p["weights_operand"]]
    sample = list(o) if p.get("sample_list") else o[0]
    h, edges = np.histogramdd(sample, **kw)
    return [h] + list(edges)


# =============================================================================================================
# S9-S11: array_equiv, the memory predicates over two views of one buffer, broadcast_shapes
# =============================================================================================================


def apply_steps(a, steps):
    """Apply a view recipe (VIEW_STEPS) to a NumPy array, exactly as OpRegistry.Surface.ApplySteps does in C#.

    Parameters:
        a: the owner.
        steps: a list of steps: ["s", "<NumSharp slice string>"], ["T"], ["r", [dims]], ["sw", a1, a2], ["rev"].

    Returns:
        The view (or, for a reshape NumPy must copy, the copy).
    """
    for st in steps:
        if st[0] == "s":
            a = a[_parse_index(st[1])]
        elif st[0] == "T":
            a = a.T
        elif st[0] == "r":
            a = a.reshape(tuple(st[1]))
        elif st[0] == "sw":
            a = a.swapaxes(st[1], st[2])
        elif st[0] == "rev":
            a = a[::-1]
        else:
            raise ValueError(st)
    return a


def _parse_index(text):
    """A NumSharp slice string ("1:5, ::2, 3") as a NumPy index tuple (start:stop:step parts or integers)."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if ":" in part:
            bits = [b.strip() for b in part.split(":")]
            bits += [""] * (3 - len(bits))
            out.append(slice(*(int(b) if b else None for b in bits)))
        else:
            out.append(int(part))
    return tuple(out)


VIEW_PAIRS = [
    ("overlap", [["s", "0:6"]], [["s", "5:9"]]),
    ("disjoint", [["s", "0:6"]], [["s", "6:12"]]),
    ("interleaved", [["s", "0:12:2"]], [["s", "1:12:2"]]),
    ("same", [], []),
    ("rev_vs_fwd", [["rev"]], [["s", "3:4"]]),
    ("rows_cols", [["r", [3, 4]], ["s", ":, 0"]], [["r", [3, 4]], ["s", "1, :"]]),
    ("cols_even_odd", [["r", [3, 4]], ["s", ":, ::2"]], [["r", [3, 4]], ["s", ":, 1::2"]]),
    ("transposed", [["r", [3, 4]], ["T"]], [["s", "11:12"]]),
    ("empty", [["s", "3:3"]], [["s", "0:12"]]),
    ("swapaxes_block", [["r", [2, 2, 3]], ["sw", 0, 2], ["s", "0:1"]], [["r", [2, 2, 3]], ["s", "1:2"]]),
]


def gen_equiv_memory_shapes():
    """np.array_equiv over broadcast pairs; np.shares_memory / np.may_share_memory over two views of one owner;
    np.broadcast_shapes over shape lists."""
    for dt in ALL13:
        a = values((3,), dt, 1)
        b = np.broadcast_to(a, (2, 3)).copy()
        for label, x, y in [("bcast_eq", a, b), ("same", a, a.copy()), ("diff", a, values((3,), dt, 2)),
                            ("nonbcast", a, values((2, 4), dt)), ("empty", a[:0], b[:, :0]),
                            ("zerod", np.asarray(a[0]), np.full((2, 2), a[0])), ("bcast_ne", a, values((2, 3), dt))]:
            emit("memory", "array_equiv", f"{dt}:{label}", [operand(x, "c")[1], operand(y, "c")[1]], {}, "scalar",
                 lambda o: np.bool_(np.array_equiv(o[0], o[1])))
        for kind, d in kinds_of(b, VALUE_KINDS + ["bcast"]):
            emit("memory", "array_equiv", f"{dt}:{kind}", [operand(a, "c")[1], d], {}, "scalar",
                 lambda o: np.bool_(np.array_equiv(o[0], o[1])))
    for label, x, y in [("nan", np.array([np.nan, 1.0]), np.array([np.nan, 1.0])),
                        ("negzero", np.array([-0.0]), np.array([0.0])),
                        ("mixed", np.array([1, 2], np.int8), np.array([1.0, 2.0])),
                        ("cplx_real", np.array([1 + 0j]), np.array([1.0]))]:
        emit("memory", "array_equiv", label, [operand(x, "c")[1], operand(y, "c")[1]], {}, "scalar",
             lambda o: np.bool_(np.array_equiv(o[0], o[1])))

    owner = values((12,), "int64")
    for fn_name, fn in [("shares_memory", np.shares_memory), ("may_share_memory", np.may_share_memory)]:
        for label, va, vb in VIEW_PAIRS:
            for mw_label, mw in [("default", "absent"), ("exact", -1), ("bounds", 0), ("work1", 1), ("work1000", 1000)]:
                p = {"views": [va, vb]}
                if mw != "absent":
                    p["max_work"] = mw
                emit("memory", fn_name, f"{label}:{mw_label}", [operand(owner, "own")[1]], p, "scalar",
                     lambda o, p=p, fn=fn: np.bool_(_mem_call(fn, o, p)))
    for label, va, vb in [("f8_rows", [["r", [3, 4]], ["s", "0:1"]], [["r", [3, 4]], ["s", "1:3"]])]:
        p = {"views": [va, vb]}
        emit("memory", "shares_memory", f"f8:{label}", [operand(values((12,), "float64"), "own")[1]], p, "scalar",
             lambda o, p=p: np.bool_(_mem_call(np.shares_memory, o, p)))
    # The same view pairs over owners of other element sizes (the overlap test works in bytes).
    for dt in ["int8", "float32", "complex128", "bool"]:
        dt_owner = values((12,), dt, 1)
        for fn_name, fn in [("shares_memory", np.shares_memory), ("may_share_memory", np.may_share_memory)]:
            for label, va, vb in VIEW_PAIRS:
                p = {"views": [va, vb]}
                emit("memory", fn_name, f"{dt}:{label}", [operand(dt_owner, "own")[1]], p, "scalar",
                     lambda o, p=p, fn=fn: np.bool_(_mem_call(fn, o, p)))

    for label, shapes in [("pair", [[2, 1], [1, 3]]), ("three", [[1], [4, 1], [2, 1, 1]]), ("none", []),
                          ("one", [[5, 0]]), ("scalar_shape", [[], [3]]), ("zero_bcast", [[0, 1], [1, 4]]),
                          ("mismatch", [[2], [3]]), ("mismatch3", [[2, 3], [3, 3], [4, 3]]), ("neg", [[-1], [2]]),
                          ("high_rank", [[1, 2, 1, 4, 1, 6], [3, 1, 5, 1, 7, 1]]), ("zero_vs_one", [[0], [1]]),
                          ("zero_vs_two", [[0], [2]])]:
        p = {"shapes": shapes}
        emit("memory", "broadcast_shapes", label, [], p, "array",
             lambda o, p=p: np.array(np.broadcast_shapes(*[tuple(s) for s in p["shapes"]]), dtype=np.int64))


def _mem_call(fn, o, p):
    a = apply_steps(o[0], p["views"][0])
    b = apply_steps(o[0], p["views"][1])
    if "max_work" in p:
        return fn(a, b, max_work=p["max_work"])
    return fn(a, b)


# =============================================================================================================
# S12-S14: apply_along_axis / apply_over_axes (a menu of functions), bmat
# =============================================================================================================

# name -> NumPy callable on a 1-D slice. OpRegistry.Surface.cs maps each name to the same NumSharp computation.
APPLY_FUNCS = {
    "sum": np.sum,
    "mean": np.mean,
    "max": np.max,
    "cumsum": np.cumsum,
    "sort": np.sort,
    "first": lambda v: v[0],
    "outer": lambda v: np.outer(v, v),
    "diff": np.diff,
    "argmax": np.argmax,
    "double": lambda v: v * 2,
}

# name -> NumPy callable (a, axis) for apply_over_axes.
OVER_FUNCS = {
    "sum": np.sum,
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
    "prod": np.prod,
    "cumsum": np.cumsum,
}


def gen_apply_bmat():
    """apply_along_axis / apply_over_axes over a function menu, every dtype and operand kinds; np.bmat's nested list
    and string forms."""
    for dt in ALL13:
        x = values((2, 3, 4), dt, 2)
        for fname in APPLY_FUNCS:
            for axis in [0, 1, -1]:
                p = {"func": fname, "axis": axis}
                emit("apply", "apply_along_axis", f"{dt}:{fname}:ax{axis}", [operand(x, "c")[1]], p, "array",
                     lambda o, p=p: np.apply_along_axis(APPLY_FUNCS[p["func"]], p["axis"], o[0]))
        for fname in OVER_FUNCS:
            for axes in [0, [0, 2], [-1], []]:
                p = {"func": fname, "axes": axes}
                emit("apply", "apply_over_axes", f"{dt}:{fname}:{axes}", [operand(x, "c")[1]], p, "array",
                     lambda o, p=p: np.apply_over_axes(OVER_FUNCS[p["func"]], o[0], p["axes"]))
    x = values((3, 4), "float64", 3)
    for kind, d in kinds_of(x, VALUE_KINDS + DERIVED_KINDS):
        for fname in ["sum", "cumsum", "first", "outer"]:
            p = {"func": fname, "axis": 1}
            emit("apply", "apply_along_axis", f"kind:{kind}:{fname}", [d], p, "array",
                 lambda o, p=p: np.apply_along_axis(APPLY_FUNCS[p["func"]], p["axis"], o[0]))
        p = {"func": "sum", "axes": [0]}
        emit("apply", "apply_over_axes", f"kind:{kind}", [d], p, "array",
             lambda o, p=p: np.apply_over_axes(OVER_FUNCS[p["func"]], o[0], p["axes"]))
    for label, arr, p in [("axis_oob", x, {"func": "sum", "axis": 2}), ("axis_neg_oob", x, {"func": "sum", "axis": -3}),
                          ("one_d", values((5,), "int32"), {"func": "cumsum", "axis": 0}),
                          ("zerod", np.array(3.0), {"func": "sum", "axis": 0}),
                          ("empty_iter", values((0, 3), "float64"), {"func": "sum", "axis": 1}),
                          ("empty_axis", values((3, 0), "float64"), {"func": "sum", "axis": 1})]:
        emit("apply", "apply_along_axis", label, [operand(arr, "c")[1]], p, "array",
             lambda o, p=p: np.apply_along_axis(APPLY_FUNCS[p["func"]], p["axis"], o[0]))
    for label, arr, p in [("axes_oob", x, {"func": "sum", "axes": [2]}), ("one_d", values((5,), "int32"),
                                                                           {"func": "sum", "axes": [0]}),
                          ("zerod", np.array(3.0), {"func": "sum", "axes": [0]})]:
        emit("apply", "apply_over_axes", label, [operand(arr, "c")[1]], p, "array",
             lambda o, p=p: np.apply_over_axes(OVER_FUNCS[p["func"]], o[0], p["axes"]))

    blocks = {
        "2x2": ([[0, 1], [2, 3]], [values((1, 2), "float64"), values((1, 1), "float64", 1),
                                   values((2, 2), "float64", 2), values((2, 1), "float64", 3)]),
        "row": ([0, 1], [values((2, 2), "int32"), values((2, 3), "int32", 1)]),
        "mixed": ([[0, 1]], [values((2, 2), "int8"), values((2, 1), "float32", 1)]),
        "bad": ([[0, 1], [2, 3]], [values((1, 2), "float64"), values((1, 1), "float64", 1),
                                   values((2, 1), "float64", 2), values((2, 1), "float64", 3)]),
        "single": ([[0]], [values((2, 3), "int16")]),
    }
    for label, (layout, arrs) in blocks.items():
        p = {"layout": layout}
        emit("apply", "bmat", f"list:{label}", [operand(a, "c")[1] for a in arrs], p, "array",
             lambda o, p=p: np.asarray(np.bmat(_bmat_obj(o, p["layout"]))))
    for kind, d in kinds_of(values((2, 2), "float64", 4), ["c_offset", "f", "reversed", "strided_neg", "readonly",
                                                           "bcast"]):
        p = {"layout": [[0, 1]]}
        emit("apply", "bmat", f"list:kind:{kind}", [d, operand(values((2, 1), "float64"), "c")[1]], p, "array",
             lambda o, p=p: np.asarray(np.bmat(_bmat_obj(o, p["layout"]))))
    for label, text, local, glob in [("both", "A,B;C,D", {"A": 0, "B": 1}, {"C": 2, "D": 3}),
                                     ("glob_only", "A B; C D", {}, {"A": 0, "B": 1, "C": 2, "D": 3}),
                                     ("local_shadows", "A,B", {"A": 1}, {"A": 0, "B": 1}),
                                     ("missing", "A,X", {"A": 0}, {"B": 1})]:
        arrs = [values((1, 1), "float64"), values((1, 1), "float64", 1), values((1, 1), "float64", 2),
                values((1, 1), "float64", 3)]
        p = {"string": text, "ldict": local, "gdict": glob}
        emit("apply", "bmat", f"str:{label}", [operand(a, "c")[1] for a in arrs], p, "array",
             lambda o, p=p: np.asarray(np.bmat(p["string"], ldict={k: o[v] for k, v in p["ldict"].items()},
                                               gdict={k: o[v] for k, v in p["gdict"].items()})))


def _bmat_obj(o, layout):
    if layout and isinstance(layout[0], list):
        return [[o[i] for i in row] for row in layout]
    return [o[i] for i in layout]


# =============================================================================================================
# S15-S21: the text formatters
# =============================================================================================================


def gen_text():
    """array2string with every non-default option, format_float_positional / format_float_scientific per float
    width, base_repr, binary_repr, typename, datetime_data."""
    opts = [({}, "default"), ({"separator": ", "}, "sep"), ({"prefix": "x(", "suffix": ")"}, "prefix_suffix"),
            ({"precision": 2}, "precision"), ({"suppress_small": True, "precision": 3}, "suppress"),
            ({"threshold": 5, "edgeitems": 1}, "threshold"), ({"max_line_width": 20}, "linewidth"),
            ({"sign": "+"}, "sign_plus"), ({"sign": " "}, "sign_space"), ({"floatmode": "fixed", "precision": 3},
                                                                          "fixed"),
            ({"floatmode": "unique"}, "unique"), ({"floatmode": "maxprec_equal", "precision": 4}, "maxprec_equal")]
    for dt in ALL13:
        x = values((3, 4), dt, 1)
        if np.dtype(dt).kind == "f":
            x = (x / 3).astype(dt)
        for p, label in opts:
            emit("text", "array2string", f"{dt}:{label}", [operand(x, "c")[1]], p, "text",
                 lambda o, p=p: np.array2string(o[0], **p))
        for kind, d in kinds_of(x, ["f", "c_offset", "reversed", "strided_neg", "bcast", "empty", "empty_view"]):
            emit("text", "array2string", f"{dt}:{kind}", [d], {}, "text", lambda o: np.array2string(o[0]))
        emit("text", "array2string", f"{dt}:zerod", [operand(np.asarray(x.flat[1]), "c")[1]], {}, "text",
             lambda o: np.array2string(o[0]))
        emit("text", "array2string", f"{dt}:zerod_view", [operand(np.asarray(x.flat[6]), "c_offset")[1]], {}, "text",
             lambda o: np.array2string(o[0]))
    for label, x in [("big", np.arange(2000, dtype=np.int64)), ("specials", np.array([np.nan, -np.inf, -0.0, 1e300])),
                     ("small", np.array([1e-20, 1.0])), ("cplx_specials", np.array([complex(np.nan, -0.0), 1j]))]:
        for p, lab in [({}, "default"), ({"threshold": 10, "edgeitems": 2}, "summ"), ({"precision": 3}, "p3")]:
            emit("text", "array2string", f"{label}:{lab}", [operand(x, "c")[1]], p, "text",
                 lambda o, p=p: np.array2string(o[0], **p))

    floats = [0.1, 1.0, 3.25, -2.5, 1e-5, 123456789.125, 1e300, 5e-324, 65504.0, float("nan"), float("inf"), -0.0]
    pos_opts = [({}, "default"), ({"precision": 3}, "p3"), ({"precision": 0}, "p0"), ({"unique": False, "precision": 5},
                                                                                       "nonunique"),
                ({"fractional": False, "precision": 3}, "nonfrac"), ({"trim": "-"}, "trim-"), ({"trim": "0"}, "trim0"),
                ({"trim": "."}, "trimdot"), ({"sign": True}, "sign"), ({"pad_left": 5, "pad_right": 4}, "pad"),
                ({"min_digits": 4}, "mindigits"), ({"precision": -1}, "prec_neg")]
    sci_opts = [({}, "default"), ({"precision": 3}, "p3"), ({"unique": False, "precision": 4}, "nonunique"),
                ({"trim": "-"}, "trim-"), ({"sign": True}, "sign"), ({"pad_left": 4}, "pad"), ({"exp_digits": 4},
                                                                                              "expdigits"),
                ({"min_digits": 3}, "mindigits"), ({"precision": -2}, "prec_neg")]
    for width in ["float16", "float32", "float64"]:
        for v in floats:
            xv = np.asarray(v, dtype=width)
            vlabel = f"{width}:{v!r}"
            for p, label in pos_opts:
                emit("text", "format_float_positional", f"{vlabel}:{label}", [operand(xv, "c")[1]], p, "text",
                     lambda o, p=p: np.format_float_positional(o[0][()], **p))
            for p, label in sci_opts:
                emit("text", "format_float_scientific", f"{vlabel}:{label}", [operand(xv, "c")[1]], p, "text",
                     lambda o, p=p: np.format_float_scientific(o[0][()], **p))
    # The precision cutoff's rounding (float64) under every trim mode: digits that carry through 9s (1.99999), round
    # down onto zeros (0.1001), tie (0.125, 2.675) or sit just below a power of ten (0.0999755859375).
    for v in [1.99999, 0.0995, 9.9999, 999.9996, 0.1001, 0.125, 2.675, 0.0999755859375, 9.999999747378752e-06]:
        xv = np.asarray(v, dtype="float64")
        for prec in [0, 1, 2, 3, 5]:
            for trim in ["k", ".", "0", "-"]:
                p = {"precision": prec, "trim": trim}
                emit("text", "format_float_positional", f"round:{v!r}:p{prec}:{trim}", [operand(xv, "c")[1]], p,
                     "text", lambda o, p=p: np.format_float_positional(o[0][()], **p))
                emit("text", "format_float_scientific", f"round:{v!r}:p{prec}:{trim}", [operand(xv, "c")[1]], p,
                     "text", lambda o, p=p: np.format_float_scientific(o[0][()], **p))

    for num in [0, 1, 255, -255, 2 ** 31, -2 ** 63, 2 ** 64 - 1, 3 ** 50, -(5 ** 40)]:
        for base in [2, 7, 16, 36]:
            for padding in [0, 3]:
                p = {"number": num, "base": base, "padding": padding}
                emit("text", "base_repr", f"{num}:{base}:{padding}", [], p, "text",
                     lambda o, p=p: np.base_repr(p["number"], p["base"], p["padding"]))
    for base in [1, 0, 37, -2]:
        p = {"number": 10, "base": base, "padding": 0}
        emit("text", "base_repr", f"bad_base{base}", [], p, "text",
             lambda o, p=p: np.base_repr(p["number"], p["base"], p["padding"]))
    for num in [0, 5, -5, 127, -128, 2 ** 40, -(2 ** 40), 2 ** 70, -1]:
        for width in [None, 1, 4, 8, 16, 80]:
            p = {"num": num, "width": width}
            emit("text", "binary_repr", f"{num}:{width}", [], p, "text",
                 lambda o, p=p: np.binary_repr(p["num"], width=p["width"]))
    for ch in ["S1", "?", "B", "D", "G", "F", "I", "L", "O", "Q", "S", "U", "V", "b", "c", "d", "f", "g", "h",
               "i", "l", "q", "e", "x", "", "s1"]:
        p = {"char": ch}
        emit("text", "typename", f"char:{ch or 'empty'}", [], p, "text", lambda o, p=p: np.typename(p["char"]))
    for spec in ["M8[ms]", "m8[5s]", "M8", "m8[W]", "M8[10us]", "m8[3D]", "f8", "i4", "M8[Y]", "m8[as]"]:
        p = {"dtype": spec}
        emit("text", "datetime_data", spec, [], p, "text", lambda o, p=p: repr(np.datetime_data(np.dtype(p["dtype"]))))


# =============================================================================================================
# S22-S27: ndarray methods and attributes (tolist, tofile, setfield, base, dtype, shape, device, to_device)
# =============================================================================================================


def _tofile(a, p):
    """NumPy's ndarray.tofile into a temporary file, read back: raw bytes (sep="") as uint8, text as str."""
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    try:
        a.tofile(path, sep=p["sep"], format=p["format"])
        with open(path, "rb") as f:
            data = f.read()
    finally:
        os.remove(path)
    if p["sep"] == "":
        return np.frombuffer(data, dtype=np.uint8).copy()
    return data.decode("latin-1")


def gen_ndarray():
    """The ndarray members: tolist (Python repr), tofile (raw and text), setfield, base, dtype, shape, device,
    to_device; over every dtype and operand kinds."""
    for dt in ALL13:
        x = values((2, 3), dt, 1)
        # "zerod_view" is a 0-d view at a non-zero offset (x.flat[2] of a buffer), the layout an integer index makes.
        for kind, d in kinds_of(x, VALUE_KINDS + DERIVED_KINDS) + [("zerod", operand(np.asarray(x.flat[2]), "c")[1]),
                                                                    ("zerod_view",
                                                                     operand(np.asarray(x.flat[2]), "c_offset")[1]),
                                                                    ("own", operand(x, "own")[1])]:
            emit("ndarray", "ndarray.tolist", f"{dt}:{kind}", [d], {}, "text", lambda o: repr(o[0].tolist()))
            emit("ndarray", "ndarray.dtype", f"{dt}:{kind}", [d], {}, "dtype", lambda o: o[0].dtype)
            emit("ndarray", "ndarray.shape", f"{dt}:{kind}", [d], {}, "array",
                 lambda o: np.array(o[0].shape, dtype=np.int64))
            emit("ndarray", "ndarray.tofile", f"{dt}:{kind}:raw", [d], {"sep": "", "format": "%s"}, "array",
                 lambda o: _tofile(o[0], {"sep": "", "format": "%s"}))
            if dt in ("int32", "float64", "bool", "uint8"):
                for sep, fmt in [(" ", "%s"), (",", "%.3f"), ("\n", "%d")]:
                    p = {"sep": sep, "format": fmt}
                    emit("ndarray", "ndarray.tofile", f"{dt}:{kind}:{sep!r}:{fmt}", [d], p, "text",
                         lambda o, p=p: _tofile(o[0], p))
        emit("ndarray", "ndarray.device", f"{dt}", [operand(x, "c")[1]], {}, "text", lambda o: o[0].device)
        for kind, d in kinds_of(x, ["c", "f", "reversed", "readonly", "bcast"]):
            emit("ndarray", "ndarray.to_device", f"{dt}:{kind}", [d], {"device": "cpu"}, "array",
                 lambda o: o[0].to_device("cpu"))
    for label, p in [("gpu", {"device": "gpu"}), ("stream", {"device": "cpu", "stream": 1}),
                     ("cuda", {"device": "cuda:0"})]:
        emit("ndarray", "ndarray.to_device", label, [operand(values((3,), "int32"), "c")[1]], p, "array",
             lambda o, p=p: o[0].to_device(p["device"], stream=p.get("stream")))
    for label, x in [("specials", np.array([np.nan, -np.inf, -0.0, 1e-5, 1e16, 0.1])),
                     ("specials_f4", np.array([np.nan, -0.0, 1e-5, 1e16, 0.1], dtype=np.float32)),
                     ("specials_f2", np.array([np.nan, -0.0, 1e-5, 65504, 0.1], dtype=np.float16)),
                     ("cplx", np.array([1 + 2j, complex(-0.0, -1), 0j, complex(np.nan, np.inf), 1e-5j, complex(1e16, 0)])),
                     ("int_extremes", np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max])),
                     ("u64_max", np.array([np.iinfo(np.uint64).max], dtype=np.uint64)),
                     ("deep", values((2, 1, 2, 1, 2), "int16")), ("empty_nested", np.zeros((2, 0, 3)))]:
        emit("ndarray", "ndarray.tolist", label, [operand(x, "c")[1]], {}, "text", lambda o: repr(o[0].tolist()))

    # setfield: write a field of another dtype into the bytes of each element.
    for dt, fields in [("int32", [("int16", 0), ("int16", 2), ("int8", 3), ("uint8", 1)]),
                       ("int64", [("int32", 4), ("float32", 0), ("int16", 6)]),
                       ("float64", [("float32", 4), ("int32", 0)]),
                       ("uint16", [("uint8", 1), ("int8", 0)]),
                       ("complex128", [("float64", 8), ("float64", 0)])]:
        x = values((2, 3), dt, 2)
        for fdt, off in fields:
            for val in [-1, 7, 0.5]:
                p = {"value": val, "dtype": fdt, "offset": off}
                emit("ndarray", "ndarray.setfield", f"{dt}:{fdt}@{off}:{val!r}", [operand(x, "own")[1]], p, "array",
                     lambda o, p=p: _setfield(o[0], p))
        for kind, d in kinds_of(x, ["c_offset", "f", "reversed", "strided"]):
            p = {"value": 3, "dtype": fields[0][0], "offset": fields[0][1]}
            emit("ndarray", "ndarray.setfield", f"{dt}:{kind}", [d], p, "array", lambda o, p=p: _setfield(o[0], p))
    for label, dt, p in [("too_big", "int32", {"value": 1, "dtype": "int16", "offset": 3}),
                         ("neg_offset", "int32", {"value": 1, "dtype": "int16", "offset": -1}),
                         ("wider", "int16", {"value": 1, "dtype": "int64", "offset": 0}),
                         ("readonly", "int32", {"value": 1, "dtype": "int16", "offset": 0})]:
        d = operand(values((3,), dt), "readonly" if label == "readonly" else "own")[1]
        emit("ndarray", "ndarray.setfield", label, [d], p, "array", lambda o, p=p: _setfield(o[0], p))
    p = {"value_operand": 1, "dtype": "int16", "offset": 0}
    emit("ndarray", "ndarray.setfield", "array_value", [operand(values((2, 3), "int32"), "own")[1],
                                                         operand(values((3,), "int16", 4), "c")[1]], p, "array",
         lambda o, p=p: _setfield(o[0], p, o[1]))

    # base: the owner a view of an owner reports (NumPy collapses the chain), or None for the owner itself.
    for dt in ["float64", "int8", "bool", "complex128"]:
        owner = values((12,), dt)
        for label, steps in [("owner", []), ("slice", [["s", "2:8"]]), ("reshape", [["r", [3, 4]]]),
                             ("chain", [["r", [3, 4]], ["s", "1:, ::2"], ["T"]]), ("rev", [["rev"]])]:
            p = {"views": [steps]}
            emit("ndarray", "ndarray.base", f"{dt}:{label}", [operand(owner, "own")[1]], p, "tuple",
                 lambda o, p=p: _base_slots(apply_steps(o[0], p["views"][0])))


def _setfield(a, p, val=None):
    a.setfield(p["value"] if val is None else val, p["dtype"], p["offset"])
    return a


def _base_slots(v):
    return [] if v.base is None else [v.base]


# =============================================================================================================
# S29-S34: the attributes of finfo, iinfo, dtype, ndarray.flags and poly1d
# =============================================================================================================

FINFO_FLOAT = ["eps", "epsneg", "max", "min", "tiny", "smallest_normal", "smallest_subnormal", "resolution"]
FINFO_INT = ["bits", "precision", "maxexp", "minexp"]
DTYPE_SPECS = ALL13 + ["b1", "u2", "f4", "c16", ">i4", "<f8", "=i2", "|u1", ">c16", "M8[ms]", "m8[5s]"]
# Spellings of the finfo / iinfo dtypes beyond the names (char codes, type-code strings, aliases): each must resolve to
# the same machine limits. Platform-dependent C-long codes ('l', 'L', 'long', 'ulong', 'g') are left out on purpose.
FINFO_SPECS = ["float16", "float32", "float64", "complex128", "int32", "bool", "uint8", "half", "e", "f2", "single",
               "f", "f4", "double", "d", "f8", "c16", "cdouble"]
IINFO_SPECS = INTS + ["bool", "float64", "complex128", "i1", "u2", "i8", "u8", "intp", "uintp", "short", "ushort",
                      "byte", "ubyte", "longlong", "ulonglong"]
DTYPE_ATTRS = ["alignment", "base", "byteorder", "char", "descr", "fields", "flags", "hasobject", "isbuiltin",
               "isnative", "itemsize", "kind", "metadata", "name", "names", "ndim", "num", "shape", "str", "subdtype",
               "newbyteorder"]
FLAG_ATTRS = ["aligned", "behaved", "c_contiguous", "carray", "contiguous", "f_contiguous", "farray", "fnc", "forc",
              "fortran", "num", "owndata", "writeable", "writebackifcopy"]


def _dtype_attr_text(dt, attr):
    """The Python repr of a dtype attribute, with strings shown raw (the C# side formats them the same way)."""
    v = dt.newbyteorder() if attr == "newbyteorder" else getattr(dt, attr)
    if isinstance(v, str):
        return v
    return repr(v)


def gen_attributes():
    """finfo / iinfo attributes per dtype (with the refusals), dtype attributes per dtype string, ndarray.flags
    attributes per operand kind, poly1d members."""
    for dt in FINFO_SPECS:
        for attr in FINFO_FLOAT:
            p = {"dtype": dt}
            emit("attrs", f"finfo.{attr}", dt, [], p, "scalar",
                 lambda o, attr=attr, dt=dt: np.float64(getattr(np.finfo(dt), attr)))
        for attr in FINFO_INT:
            p = {"dtype": dt}
            emit("attrs", f"finfo.{attr}", dt, [], p, "scalar",
                 lambda o, attr=attr, dt=dt: np.int64(getattr(np.finfo(dt), attr)))
        emit("attrs", "finfo.dtype", dt, [], {"dtype": dt}, "dtype", lambda o, dt=dt: np.finfo(dt).dtype)
    for dt in IINFO_SPECS:
        p = {"dtype": dt}
        emit("attrs", "iinfo.bits", dt, [], p, "scalar", lambda o, dt=dt: np.int64(np.iinfo(dt).bits))
        emit("attrs", "iinfo.min", dt, [], p, "scalar", lambda o, dt=dt: np.int64(np.iinfo(dt).min))
        emit("attrs", "iinfo.max", dt, [], p, "scalar",
             lambda o, dt=dt: (np.uint64 if np.dtype(dt).kind == "u" else np.int64)(np.iinfo(dt).max))
        emit("attrs", "iinfo.kind", dt, [], p, "text", lambda o, dt=dt: np.iinfo(dt).kind)
        emit("attrs", "iinfo.dtype", dt, [], p, "dtype", lambda o, dt=dt: np.iinfo(dt).dtype)
    for spec in DTYPE_SPECS:
        for attr in DTYPE_ATTRS:
            # A 4- or 8-byte integer's type char and type number follow the C long (int32 is 'l' / 7 on win-amd64, 'i'
            # / 5 on LP64), on NumPy's side and NumSharp's: host-pinned, so they stay out of the portable tier.
            if attr in ("char", "num") and np.dtype(spec).name in PLATFORM_FORMATS:
                continue
            p = {"dtype": spec}
            emit("attrs", f"dtype.{attr}", spec, [], p, "text",
                 lambda o, attr=attr, spec=spec: _dtype_attr_text(np.dtype(spec), attr))
    for dt in ["int32", "float64", "bool", "uint8", "float16", "complex128"]:
        x = values((3, 4), dt)
        for kind, d in kinds_of(x, VALUE_KINDS + DERIVED_KINDS) + [("own", operand(x, "own")[1]),
                                                                    ("zerod", operand(np.asarray(x.flat[0]), "c")[1]),
                                                                    ("zerod_view",
                                                                     operand(np.asarray(x.flat[5]), "c_offset")[1]),
                                                                    ("one_1d", operand(x.reshape(-1)[:1].copy(),
                                                                                       "c")[1])]:
            for attr in FLAG_ATTRS:
                emit("attrs", f"ndarray.flags.{attr}", f"{dt}:{kind}", [d], {}, "scalar",
                     lambda o, attr=attr: (np.int64 if attr == "num" else np.bool_)(getattr(o[0].flags, attr)))
    for dt in ["int64", "float64", "complex128", "int8", "float32"]:
        c = values((4,), dt, 1)
        if dt != "complex128":
            c[0] = 2
        for kind, d in kinds_of(c, ["c", "c_offset", "reversed", "strided_neg"]):
            emit("attrs", "poly1d.coeffs", f"{dt}:{kind}", [d], {}, "array", lambda o: np.poly1d(o[0]).coeffs)
            for attr in ["c", "coef", "coefficients"]:
                emit("attrs", f"poly1d.{attr}", f"{dt}:{kind}", [d], {}, "array",
                     lambda o, attr=attr: getattr(np.poly1d(o[0]), attr))
            for attr in ["order", "o"]:
                emit("attrs", f"poly1d.{attr}", f"{dt}:{kind}", [d], {}, "scalar",
                     lambda o, attr=attr: np.int64(getattr(np.poly1d(o[0]), attr)))
            for m in [0, 1, 2, 5]:
                emit("attrs", "poly1d.deriv", f"{dt}:{kind}:m{m}", [d], {"m": m}, "array",
                     lambda o, m=m: np.poly1d(o[0]).deriv(m).coeffs)
            for m, k in [(1, None), (2, None), (2, [1.0, -2.0]), (1, [3.0])]:
                p = {"m": m, "k": k}
                emit("attrs", "poly1d.integ", f"{dt}:{kind}:m{m}:k{k}", [d], p, "array",
                     lambda o, m=m, k=k: np.poly1d(o[0]).integ(m, k=0 if k is None else k).coeffs)
        for var in [None, "z", "t"]:
            p = {"variable": var}
            emit("attrs", "poly1d.variable", f"{dt}:{var}", [operand(c, "c")[1]], p, "text",
                 lambda o, var=var: np.poly1d(o[0], variable=var).variable)
    for label, c, p in [("leading_zeros", np.array([0.0, 0.0, 1.0, 2.0]), {"m": 1}),
                        ("zero_poly", np.array([0.0]), {"m": 1}), ("deriv_neg", np.array([1.0, 2.0]), {"m": -1}),
                        ("scalar_coeff", np.array([5.0]), {"m": 2})]:
        emit("attrs", "poly1d.deriv", label, [operand(c, "c")[1]], p, "array",
             lambda o, p=p: np.poly1d(o[0]).deriv(p["m"]).coeffs)
    # poly1d.r / poly1d.roots = np.roots(coeffs): only the polynomials whose stripped coefficients have at most one
    # term, so np.roots never builds a companion matrix (eigvals needs a LAPACK backend, which this portable tier does
    # not load): the trailing zeros are the roots, a constant has none, the zero polynomial has none.
    for cname, coeffs in [("const", [5.0]), ("linear0", [3.0, 0.0]), ("pow2", [2.0, 0.0, 0.0]), ("lead0", [0.0, 4.0]),
                          ("lead0pow", [0.0, 0.0, 7.0, 0.0]), ("neg", [-2.5]), ("zero", [0.0]), ("zeros", [0.0, 0.0]),
                          ("pow5", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])]:
        for dt in ["float64", "int64", "complex128", "float32", "int8"]:
            c = np.array(coeffs, dtype=dt)
            for kind, d in kinds_of(c, ["c", "c_offset", "reversed", "strided_neg"]):
                for attr in ["r", "roots"]:
                    emit("attrs", f"poly1d.{attr}", f"{cname}:{dt}:{kind}", [d], {}, "array",
                         lambda o, attr=attr: getattr(np.poly1d(o[0]), attr))


# =============================================================================================================
# S35-S42: the iteration objects (np.broadcast, flatiter, np.nditer), the print and buffer options, ndarray.data
# (the memoryview), dtype.type and np.frompyfunc
# =============================================================================================================

# label -> operand values broadcast together (an empty list is np.broadcast() with no operand: a 0-d broadcast).
BCAST_SETS = [
    ("row_col", [values((3,), "int64", 1), values((2, 1), "int64", 2)]),
    ("three", [values((2, 1, 3), "float64", 1), values((4, 1), "float64", 2), values((3,), "float64", 3)]),
    ("scalar", [np.asarray(np.int32(5)), values((2, 2), "int32")]),
    ("one", [values((2, 3), "int16")]),
    ("mixed", [values((2, 3), "uint8"), values((3,), "complex128", 1)]),
    ("empty", [values((0, 3), "float32"), values((3,), "float32")]),
    ("none", []),
]


def _bcast_after(o, k):
    """np.broadcast(*o) advanced k steps (next(b) k times)."""
    b = np.broadcast(*o)
    for _ in range(k):
        next(b)
    return b


def _bcast_reset(o, k):
    """[index after reset] + the first value tuple after reset (as 0-d arrays; none for an empty broadcast)."""
    b = _bcast_after(o, k)
    b.reset()
    slots = [np.int64(b.index)]
    if b.size:
        slots += [np.asarray(v, dtype=o[i].dtype) for i, v in enumerate(next(b))]
    return slots


def _flat_after(a, k):
    """a.flat advanced k steps (next(f) k times)."""
    f = a.flat
    for _ in range(k):
        next(f)
    return f


# label -> np.nditer keyword arguments (flags / op_flags / op_dtypes / order / casting), as params.
NDITER_CONFIGS = [
    ("plain", {"flags": []}),
    ("multi", {"flags": ["multi_index"]}),
    ("c_index", {"flags": ["c_index"]}),
    ("f_index", {"flags": ["f_index"]}),
    ("extloop", {"flags": ["external_loop"]}),
    ("order_c", {"flags": [], "order": "C"}),
    ("order_f", {"flags": ["multi_index"], "order": "F"}),
    ("buffered", {"flags": ["buffered"], "op_dtypes": ["float64"], "casting": "safe"}),
    ("ranged", {"flags": ["ranged"]}),
]


def _nditer(o, p):
    """np.nditer over the operands with the case's keyword params."""
    kw = {"flags": list(p.get("flags", [])), "order": p.get("order", "K"), "casting": p.get("casting", "safe")}
    if p.get("op_dtypes") is not None:
        kw["op_dtypes"] = list(p["op_dtypes"]) * len(o)
    if p.get("op_flags") is not None:
        kw["op_flags"] = [list(f) for f in p["op_flags"]]
    if p.get("buffersize"):
        kw["buffersize"] = p["buffersize"]
    return np.nditer(list(o) if len(o) > 1 else o[0], **kw)


def _nd_values(it):
    """nditer.value as tuple slots (NumPy returns a bare array for one operand): copies, so they outlive the iterator."""
    v = it.value
    return [np.array(x, copy=True) for x in (v if isinstance(v, tuple) else (v,))]


def _nditer_attr(o, p):
    """One np.nditer member after the case's steps: the value NumPy returns, shaped as the case's result kind."""
    it = _nditer(o, p)
    with it:
        for _ in range(p.get("steps", 0)):
            it.iternext()
        # "next": elements consumed through the Python iteration protocol (NumPy's __next__ advances ON ENTRY: after k
        # calls the cursor is at element k - 1, the one last returned).
        for _ in range(p.get("next", 0)):
            next(it)
        attr = p["attr"]
        if attr == "iternext":
            return np.bool_(it.iternext())
        if attr == "value":
            return _nd_values(it)
        if attr in ("operands", "itviews"):
            return [np.array(x, copy=True) for x in getattr(it, attr)]
        if attr == "copy":
            c = it.copy()
            return [np.int64(c.iterindex)] + _nd_values(c)
        if attr == "reset":
            it.reset()
            return [np.int64(it.iterindex)] + _nd_values(it)
        if attr == "remove_axis":
            it.remove_axis(p["axis"])
            return np.array(it.shape, dtype=np.int64)
        if attr == "remove_multi_index":
            # The value at the first four positions after the removal shows the order the iterator then follows.
            it.remove_multi_index()
            slots = [np.bool_(it.has_multi_index), np.array(it.shape, dtype=np.int64)]
            for _ in range(4):
                slots += _nd_values(it)
                it.iternext()
            return slots
        if attr == "enable_external_loop":
            it.enable_external_loop()
            return _nd_values(it)
        if attr == "shape":
            return np.array(it.shape, dtype=np.int64)
        if attr == "multi_index":
            return np.array(it.multi_index, dtype=np.int64)
        if attr in ("iterrange", "dtypes"):
            return repr(getattr(it, attr))
        if attr in ("finished", "has_index", "has_multi_index", "has_delayed_bufalloc", "iterationneedsapi"):
            return np.bool_(getattr(it, attr))
        return np.int64(getattr(it, attr))  # ndim, nop, itersize, iterindex


# The configs the dtype variants of the C operand run under (the buffered float64 cast would refuse complex).
DTYPE_CONFIGS = {"plain", "multi", "extloop", "order_f"}

NDITER_KINDS = {"iternext": "scalar", "value": "tuple", "operands": "tuple", "itviews": "tuple", "copy": "tuple",
                "reset": "tuple", "remove_axis": "array", "remove_multi_index": "tuple", "enable_external_loop": "tuple",
                "shape": "array", "iterrange": "text", "dtypes": "text", "multi_index": "array"}

# The print options NumSharp's set_printoptions / printoptions take, with values NumPy accepts and refuses.
PRINT_OPTS = [({"precision": 3}, "p3"), ({"threshold": 5, "edgeitems": 1}, "summ"), ({"linewidth": 20}, "lw20"),
              ({"suppress": True}, "suppress"), ({"nanstr": "NaN", "infstr": "Inf"}, "naninf"), ({"sign": "+"}, "sign+"),
              ({"sign": " "}, "sign_sp"), ({"floatmode": "fixed", "precision": 2}, "fixed"),
              ({"floatmode": "unique"}, "unique"), ({"floatmode": "maxprec_equal", "precision": 3}, "maxprec_eq"),
              ({"precision": 0}, "p0"), ({"precision": -1}, "p_neg"), ({"sign": "x"}, "sign_bad"),
              ({"floatmode": "bogus"}, "fm_bad"), ({"threshold": 3, "edgeitems": 0}, "edge0")]
PRINT_KEYS = ["precision", "threshold", "edgeitems", "linewidth", "suppress", "nanstr", "infstr", "sign", "floatmode",
              "legacy", "formatter", "override_repr"]

# name -> (nin, nout, NumPy callable). OpRegistry.Surface.FromPyFunc maps each name to the same C# delegate over doubles.
FROMPY_FUNCS = {
    # float(): NumSharp's delegate takes doubles, so an int element is a double there (-float(0) is -0.0, -0 is 0).
    "add": (2, 1, lambda x, y: float(x) + float(y)),
    "neg": (1, 1, lambda x: -float(x)),
    "addmul": (2, 2, lambda x, y: (float(x) + float(y), float(x) * float(y))),
}

MEMVIEW_ATTRS = ["tobytes", "itemsize", "ndim", "nbytes", "readonly", "shape", "strides", "c_contiguous",
                 "f_contiguous", "contiguous", "hex", "format"]
MEMVIEW_KINDS = {"tobytes": "array", "shape": "array", "strides": "array", "hex": "text", "format": "text",
                 "readonly": "scalar", "c_contiguous": "scalar", "f_contiguous": "scalar", "contiguous": "scalar"}
# The struct codes of these dtypes are platform-dependent ('l' is a C long), so their memoryview.format is host-pinned.
PLATFORM_FORMATS = {"int32", "uint32", "int64", "uint64"}


def _memview_attr(a, attr):
    m = a.data
    if attr == "tobytes":
        return np.frombuffer(m.tobytes(), dtype=np.uint8).copy()
    if attr in ("shape", "strides"):
        return np.array(getattr(m, attr), dtype=np.int64)
    if attr == "hex":
        return m.hex()
    if attr == "format":
        return m.format
    if attr in ("readonly", "c_contiguous", "f_contiguous", "contiguous"):
        return np.bool_(getattr(m, attr))
    return np.int64(getattr(m, attr))


def gen_objects():
    """np.broadcast / flatiter / np.nditer members after k steps over operand kinds, set_printoptions /
    printoptions / get_printoptions, setbufsize / getbufsize, ndarray.data, dtype.type and np.frompyfunc."""
    # ---- np.broadcast ------------------------------------------------------------------------------------------
    for label, vals in BCAST_SETS:
        descs = [operand(v, "c")[1] for v in vals]
        for attr in ["nd", "ndim", "size", "numiter"]:
            emit("objects", f"broadcast.{attr}", label, descs, {}, "scalar",
                 lambda o, attr=attr: np.int64(getattr(np.broadcast(*o), attr)))
        size = np.broadcast(*np_ops(descs)).size
        for k in sorted({0, 1, size // 2, size} - ({1} if size == 0 else set())):
            p = {"steps": k}
            emit("objects", "broadcast.index", f"{label}:k{k}", descs, p, "scalar",
                 lambda o, k=k: np.int64(_bcast_after(o, k).index))
            emit("objects", "broadcast.reset", f"{label}:k{k}", descs, p, "tuple", lambda o, k=k: _bcast_reset(o, k))
            if vals:
                emit("objects", "broadcast.iters", f"{label}:k{k}", descs, p, "tuple",
                     lambda o, k=k: [np.array(list(it), dtype=o[i].dtype)
                                     for i, it in enumerate(_bcast_after(o, k).iters)])
    x = values((2, 3), "float64", 2)
    for kind, d in kinds_of(x, VALUE_KINDS + ["bcast", "empty_view"]):
        descs = [d, operand(values((3,), "float64", 3), "c")[1]]
        emit("objects", "broadcast.iters", f"kind:{kind}", descs, {"steps": 0}, "tuple",
             lambda o: [np.array(list(it), dtype=o[i].dtype) for i, it in enumerate(np.broadcast(*o).iters)])
        emit("objects", "broadcast.size", f"kind:{kind}", descs, {}, "scalar", lambda o: np.int64(np.broadcast(*o).size))

    # ---- flatiter ------------------------------------------------------------------------------------------------
    for dt in ["int16", "float64", "complex128", "bool"]:
        x = values((2, 3), dt, 1)
        for kind, d in kinds_of(x, VALUE_KINDS + DERIVED_KINDS) + [
                ("zerod", operand(np.asarray(x.flat[2]), "c")[1]),
                ("zerod_view", operand(np.asarray(x.flat[2]), "c_offset")[1])]:
            n = rl.rebuild(d).size
            for k in sorted({0, 1, n // 2, n} - ({1} if n == 0 else set())):
                p = {"steps": k}
                emit("objects", "flatiter.index", f"{dt}:{kind}:k{k}", [d], p, "scalar",
                     lambda o, k=k: np.int64(_flat_after(o[0], k).index))
                emit("objects", "flatiter.coords", f"{dt}:{kind}:k{k}", [d], p, "text",
                     lambda o, k=k: repr(_flat_after(o[0], k).coords))
            emit("objects", "flatiter.copy", f"{dt}:{kind}", [d], {}, "array", lambda o: o[0].flat.copy())
            emit("objects", "flatiter.base", f"{dt}:{kind}", [d], {}, "array", lambda o: o[0].flat.base)

    # ---- np.nditer -----------------------------------------------------------------------------------------------
    x = values((2, 3), "int32", 4)
    operand_sets = [(f"one:{kind}", [d]) for kind, d in kinds_of(x, ["c", "f", "reversed", "strided", "bcast",
                                                                      "c_offset"])]
    operand_sets.append(("pair", [operand(x, "c")[1], operand(values((3,), "int32", 5), "c")[1]]))
    operand_sets.append(("pair_f", [operand(x, "f")[1], operand(values((2, 1), "int16", 6), "c")[1]]))
    # The C operand in other dtypes (one:<dtype>), under the configs that need no cast (DTYPE_CONFIGS).
    operand_sets += [(f"one:{dt}", [operand(values((2, 3), dt, 7), "c")[1]])
                     for dt in ["bool", "uint8", "float16", "float64", "complex128", "int64"]]
    for set_label, descs in operand_sets:
        for cfg_label, cfg in NDITER_CONFIGS:
            if set_label.split(":")[-1] in ALL13 and cfg_label not in DTYPE_CONFIGS:
                continue
            label = f"{set_label}:{cfg_label}"
            multi = "multi_index" in cfg["flags"]
            buffered = "buffered" in cfg["flags"]
            for attr in ["ndim", "nop", "shape", "itersize", "iterrange", "has_index", "has_multi_index",
                         "has_delayed_bufalloc", "iterationneedsapi", "dtypes", "operands"]:
                p = {**cfg, "attr": attr}
                emit("objects", f"nditer.{attr}", label, descs, p, NDITER_KINDS.get(attr, "scalar"),
                     lambda o, p=p: _nditer_attr(o, p))
            if not buffered:
                p = {**cfg, "attr": "itviews"}
                emit("objects", "nditer.itviews", label, descs, p, "tuple", lambda o, p=p: _nditer_attr(o, p))
            for k in [0, 1, 3, 100]:
                for attr in ["iterindex", "finished", "value", "iternext", "copy", "reset"]:
                    if attr in ("value", "copy") and k == 100:
                        continue  # past the end: NumPy's value read is its own error, covered by close below
                    p = {**cfg, "attr": attr, "steps": k}
                    emit("objects", f"nditer.{attr}", f"{label}:k{k}", descs, p, NDITER_KINDS.get(attr, "scalar"),
                         lambda o, p=p: _nditer_attr(o, p))
            if multi:
                for axis in [0, 1]:
                    p = {**cfg, "attr": "remove_axis", "axis": axis}
                    emit("objects", "nditer.remove_axis", f"{label}:ax{axis}", descs, p, "array",
                         lambda o, p=p: _nditer_attr(o, p))
                p = {**cfg, "attr": "remove_multi_index"}
                emit("objects", "nditer.remove_multi_index", label, descs, p, "tuple", lambda o, p=p: _nditer_attr(o, p))
            elif cfg_label in ("plain", "order_c"):
                p = {**cfg, "attr": "enable_external_loop"}
                emit("objects", "nditer.enable_external_loop", label, descs, p, "tuple",
                     lambda o, p=p: _nditer_attr(o, p))
    # refusals and the closed / delayed iterator
    single = [operand(x, "c")[1]]
    for label, p in [("remove_axis_no_multi", {"flags": [], "attr": "remove_axis", "axis": 0}),
                     ("ext_with_multi", {"flags": ["multi_index"], "attr": "enable_external_loop"}),
                     ("delayed_value", {"flags": ["buffered", "delay_bufalloc"], "op_dtypes": ["float64"],
                                        "attr": "value"}),
                     ("delayed_reset", {"flags": ["buffered", "delay_bufalloc"], "op_dtypes": ["float64"],
                                        "attr": "reset"}),
                     ("delayed_flag", {"flags": ["buffered", "delay_bufalloc"], "op_dtypes": ["float64"],
                                       "attr": "has_delayed_bufalloc"})]:
        emit("objects", f"nditer.{p['attr']}", label, single, p, NDITER_KINDS.get(p["attr"], "scalar"),
             lambda o, p=p: _nditer_attr(o, p))
    for after in ["iternext", "ndim", "value", "iterindex"]:
        p = {"flags": [], "after": after}
        emit("objects", "nditer.close", f"then:{after}", single, p,
             {"iternext": "scalar", "value": "tuple"}.get(after, "scalar"), lambda o, p=p: _nditer_closed(o, p))
    # The iteration protocol: k next() calls (C#: k MoveNext of the enumerator), then iterindex / value / multi_index.
    proto_vals = np.arange(6, dtype=np.float32).reshape(2, 3)
    for cfg_label, cfg in [("plain", {"flags": []}), ("multi", {"flags": ["multi_index"]}),
                           ("buffered", {"flags": ["buffered"], "op_dtypes": ["float64"], "casting": "same_kind"})]:
        for k in [1, 2, 3, 6]:
            for attr in ["iterindex", "value"] + (["multi_index"] if cfg_label == "multi" else []):
                p = {**cfg, "next": k, "attr": attr}
                emit("objects", f"nditer.{attr}", f"protocol:{cfg_label}:next{k}", [operand(proto_vals, "c")[1]], p,
                     NDITER_KINDS.get(attr, "scalar"), lambda o, p=p: _nditer_attr(o, p))
    # close() resolves the write-backs: a copy (updateifcopy) reaches the operand only at close, a buffered operand by the
    # end of the loop, a plain readwrite one directly. Each config writes x * 2 + 1 through the iterator and records the
    # operand after the loop ("before" close) and after close() — called twice (NumPy allows it).
    wb_vals = np.arange(6, dtype=np.float32).reshape(2, 3)
    for cfg_label, cfg, vals in [
            ("updateifcopy", {"flags": [], "op_flags": [["readwrite", "updateifcopy"]], "op_dtypes": ["float64"],
                              "casting": "same_kind"}, wb_vals),
            ("buffered", {"flags": ["buffered"], "op_flags": [["readwrite"]], "op_dtypes": ["float64"],
                          "casting": "same_kind"}, wb_vals),
            ("buffered_ext", {"flags": ["buffered", "external_loop"], "op_flags": [["readwrite"]],
                              "op_dtypes": ["float64"], "casting": "same_kind", "buffersize": 4}, wb_vals),
            ("plain", {"flags": [], "op_flags": [["readwrite"]]}, wb_vals),
            # A plain readwrite iterator over other dtypes: no copy, no cast, the writes land directly.
            ("plain_f8", {"flags": [], "op_flags": [["readwrite"]]}, wb_vals.astype(np.float64)),
            ("plain_i8", {"flags": [], "op_flags": [["readwrite"]]}, wb_vals.astype(np.int64)),
            ("plain_i2", {"flags": [], "op_flags": [["readwrite"]]}, wb_vals.astype(np.int16)),
            ("plain_u1", {"flags": [], "op_flags": [["readwrite"]]}, wb_vals.astype(np.uint8)),
            ("int_unsafe", {"flags": [], "op_flags": [["readwrite", "updateifcopy"]], "op_dtypes": ["float64"],
                            "casting": "unsafe"}, np.arange(6, dtype=np.int32).reshape(2, 3)),
            ("int_same_kind", {"flags": [], "op_flags": [["readwrite", "updateifcopy"]], "op_dtypes": ["float64"],
                               "casting": "same_kind"}, np.arange(6, dtype=np.int32).reshape(2, 3))]:
        for kind, d in kinds_of(vals, ["c", "f", "reversed", "strided_offset", "c_offset", "readonly"]):
            for when in ["before", "after"]:
                p = {**cfg, "writeback": when}
                emit("objects", "nditer.close", f"writeback:{cfg_label}:{kind}:{when}", [d], p, "array",
                     lambda o, p=p: _nditer_writeback(o, p))

    # ---- print options -------------------------------------------------------------------------------------------
    arrays = [("f8", np.array([1 / 3, 1e-9, np.nan, np.inf, -2.5, 1e5])), ("i4", np.arange(12, dtype=np.int32)),
              ("c16", np.array([1 + 1j / 3, complex(np.nan, 1), -2j])), ("f4_2d", (np.arange(6, dtype=np.float32) / 7)
                                                                         .reshape(2, 3))]
    for alabel, arr in arrays:
        d = [operand(arr, "c")[1]]
        for opts, olabel in PRINT_OPTS:
            emit("print", "set_printoptions", f"{alabel}:{olabel}", d, {"opts": opts}, "text",
                 lambda o, opts=opts: _with_set_printoptions(opts, lambda: np.array2string(o[0])))
            emit("print", "printoptions", f"{alabel}:{olabel}", d, {"opts": opts, "phase": "inside"}, "text",
                 lambda o, opts=opts: _in_printoptions(opts, lambda: np.array2string(o[0])))
        emit("print", "printoptions", f"{alabel}:restored", d, {"opts": {"precision": 2, "sign": "+"},
                                                               "phase": "after"}, "text",
             lambda o: _after_printoptions({"precision": 2, "sign": "+"}, lambda: np.array2string(o[0])))
    for opts, olabel in [({}, "default")] + PRINT_OPTS[:11]:
        for key in PRINT_KEYS:
            emit("print", "get_printoptions", f"{olabel}:{key}", [], {"opts": opts, "key": key}, "text",
                 lambda o, opts=opts, key=key: _in_printoptions(opts, lambda: repr(np.get_printoptions()[key])))
        emit("print", "get_printoptions", f"{olabel}:keys", [], {"opts": opts, "key": None}, "text",
             lambda o, opts=opts: _in_printoptions(opts, lambda: repr(sorted(np.get_printoptions()))))

    # ---- buffer size ---------------------------------------------------------------------------------------------
    for n in [16, 4096, 8192, 10_000_000, 10_000_016, 4, 17, -1, 0, 8, 160]:
        emit("print", "setbufsize", f"n{n}", [], {"size": n}, "scalar", lambda o, n=n: _with_bufsize(n, lambda old: old))
        emit("print", "getbufsize", f"after{n}", [], {"size": n}, "scalar",
             lambda o, n=n: _with_bufsize(n, lambda old: np.int64(np.getbufsize())))
    emit("print", "getbufsize", "default", [], {"size": None}, "scalar", lambda o: np.int64(np.getbufsize()))

    # ---- ndarray.data (the memoryview) ---------------------------------------------------------------------------
    for dt in ALL13:
        x = values((2, 3), dt, 3)
        for kind, d in kinds_of(x, ["c", "c_offset", "f", "reversed", "strided", "strided_neg", "bcast", "readonly",
                                    "empty", "empty_view"]) + [("zerod", operand(np.asarray(x.flat[1]), "c")[1]),
                                                               ("zerod_view",
                                                                operand(np.asarray(x.flat[4]), "c_offset")[1])]:
            for attr in MEMVIEW_ATTRS:
                if attr == "format" and dt in PLATFORM_FORMATS:
                    continue
                emit("ndarray", "ndarray.data", f"{dt}:{kind}:{attr}", [d], {"attr": attr},
                     MEMVIEW_KINDS.get(attr, "scalar"), lambda o, attr=attr: _memview_attr(o[0], attr))

    # ---- dtype.type ----------------------------------------------------------------------------------------------
    for spec in DTYPE_SPECS:
        emit("attrs", "dtype.type", spec, [], {"dtype": spec}, "text", lambda o, spec=spec: repr(np.dtype(spec).type))

    # ---- np.frompyfunc -------------------------------------------------------------------------------------------
    for dt in ["float64", "int32", "float32", "int8", "uint8", "int64", "float16"]:
        a = values((2, 3), dt, 1)
        b = values((3,), dt, 2)
        for name, (nin, nout, f) in FROMPY_FUNCS.items():
            kind_list = kinds_of(a, ["c", "f", "reversed", "strided", "bcast", "empty"]) + [
                ("zerod", operand(np.asarray(a.flat[1]), "c")[1])]
            for kind, d in kind_list:
                descs = [d] if nin == 1 else [d, operand(b if rl.rebuild(d).ndim else np.asarray(b[0]), "c")[1]]
                p = {"func": name, "nin": nin, "nout": nout}
                emit("objects", "frompyfunc", f"{dt}:{name}:{kind}", descs, p, "tuple" if nout > 1 else "array",
                     lambda o, f=f, nin=nin, nout=nout: _frompy(f, nin, nout, o))


def gen_aliases():
    """The rows the inventory credits through an alias or a property: numpy.cumulative_sum / cumulative_prod (aliased
    to np.cumsum / np.cumprod) and numpy.shape (NDArray.shape) — the shared subset and the parameters the alias lacks."""
    for op, fn in [("cumulative_sum", np.cumulative_sum), ("cumulative_prod", np.cumulative_prod)]:
        for dt in ALL13:
            x1 = values((5,), dt, 1, lo=-2, hi=3)
            x2 = values((2, 3), dt, 2, lo=-2, hi=3)
            for kind, d in kinds_of(x1, ["c", "c_offset", "reversed", "strided", "strided_neg", "readonly",
                                         "scalar_bcast", "empty", "empty_view"]):
                emit("aliases", op, f"{dt}:1d:{kind}", [d], {"axis": None}, "array", lambda o, fn=fn: fn(o[0]))
            for kind, d in kinds_of(x2, ["c", "f", "reversed", "strided", "bcast"]):
                for axis in [0, 1, -1]:
                    emit("aliases", op, f"{dt}:2d:{kind}:ax{axis}", [d], {"axis": axis}, "array",
                         lambda o, fn=fn, axis=axis: fn(o[0], axis=axis))
            emit("aliases", op, f"{dt}:2d:noaxis", [operand(x2, "c")[1]], {"axis": None}, "array",
                 lambda o, fn=fn: fn(o[0]))
            emit("aliases", op, f"{dt}:zerod", [operand(np.asarray(x1[1]), "c")[1]], {"axis": None}, "array",
                 lambda o, fn=fn: fn(o[0]))
            emit("aliases", op, f"{dt}:include_initial", [operand(x1, "c")[1]], {"axis": None, "include_initial": True},
                 "array", lambda o, fn=fn: fn(o[0], include_initial=True))
        for req in ["float32", "int64", "complex128", "int8", "bool"]:
            emit("aliases", op, f"dtype={req}", [operand(values((5,), "int32", 3), "c")[1]],
                 {"axis": None, "dtype": req}, "array", lambda o, fn=fn, req=req: fn(o[0], dtype=req))
        emit("aliases", op, "axis_oob", [operand(values((5,), "int32", 3), "c")[1]], {"axis": 1}, "array",
             lambda o, fn=fn: fn(o[0], axis=1))
    for dt in ALL13:
        x = values((2, 3), dt, 4)
        for kind, d in kinds_of(x, ["c", "f", "strided", "bcast", "empty", "empty_view"]) + [
                ("zerod", operand(np.asarray(x.flat[1]), "c")[1])]:
            emit("aliases", "shape", f"{dt}:{kind}", [d], {}, "array",
                 lambda o: np.array(np.shape(o[0]), dtype=np.int64))


def _frompy(f, nin, nout, o):
    """np.frompyfunc's object result(s) as float64 arrays (NumSharp's typed analog returns the delegate's double)."""
    r = np.frompyfunc(f, nin, nout)(*o)
    if nout == 1:
        return np.asarray(r).astype(np.float64)
    return [np.asarray(x).astype(np.float64) for x in r]


def _nditer_closed(o, p):
    """np.nditer closed, then one member read: NumPy's answer for a closed iterator."""
    it = _nditer(o, p)
    it.close()
    after = p["after"]
    if after == "iternext":
        return np.bool_(it.iternext())
    if after == "value":
        return _nd_values(it)
    return np.int64(getattr(it, after))


def _nditer_writeback(o, p):
    """np.nditer over the operand with the case's params, x[...] = x * 2 + 1 for every element (or chunk), then the
    operand copied after the loop (writeback "before" close) or after close() called twice (writeback "after")."""
    it = _nditer(o, p)
    for x in it:
        x[...] = x * 2 + 1
    if p["writeback"] == "before":
        snap = np.array(o[0], copy=True)
        it.close()
        return snap
    it.close()
    it.close()
    return np.array(o[0], copy=True)


def _with_set_printoptions(opts, fn):
    """np.set_printoptions(**opts), then fn(), the previous options restored either way."""
    with np.printoptions():
        np.set_printoptions(**opts)
        return fn()


def _in_printoptions(opts, fn):
    """fn() inside np.printoptions(**opts)."""
    with np.printoptions(**opts):
        return fn()


def _after_printoptions(opts, fn):
    """fn() after a np.printoptions(**opts) block has exited (the options it set must be gone)."""
    with np.printoptions(**opts):
        pass
    return fn()


def _with_bufsize(n, fn):
    """np.setbufsize(n) (its return value handed to fn as a 0-d int64), then the previous size restored."""
    prev = np.getbufsize()
    try:
        old = np.setbufsize(n)
        return fn(np.int64(old))
    finally:
        np.setbufsize(prev)


# =============================================================================================================
# Carving (filled from the replay of the uncarved matrix)
# =============================================================================================================

# gap id -> description. FuzzCorpusTests.Surface.cs replays gaps/surface_gaps.jsonl under [TestCategory("OpenBugs")].
# When a gap is fixed, drop its rule and regenerate.
GAPS = {
    "SU1": "np.linalg.cross with an axis outside the operands' rank (or a 0-d operand) raises AxisError 'axis 5 is out "
           "of bounds for array of dimension 2'; NumPy raises IndexError 'tuple index out of range' (its a.shape[axis] "
           "lookup)",
    "SU2": "np.linalg.cross over operands that cannot broadcast raises 'operands could not be broadcast together with "
           "shapes (4,) (5,)'; NumPy's linalg.cross raises broadcast_shapes' 'shape mismatch: objects cannot be "
           "broadcast to a single shape.  Mismatch is between arg 0 with shape (4,) and arg 1 with shape (5,).'",
    "SU3": "np.histogram / np.histogram_bin_edges of a complex128 sample with a bin COUNT return float64 edges; NumPy's "
           "edges are complex128 (bin_type = result_type(first_edge, last_edge, a))",
    "SU4": "np.histogram of a complex128 sample with an estimator name ('auto', 'fd', ...) returns a result; NumPy "
           "raises TypeError (the estimators need real data)",
    "SU5": "np.apply_along_axis over a broadcast view whose every stride is 0 at a non-zero Shape offset (np.broadcast_to "
           "of a one-element np.split child) throws ArgumentOutOfRangeException (Parameter 'length')",
    "SU6": "np.format_float_positional / np.format_float_scientific of inf or nan print .NET text ('Infinity.', "
           "'NaN.', 'I.nfinitye+07') or throw ArgumentOutOfRangeException; NumPy prints 'inf', 'nan', '+inf'",
    "SU7": "np.format_float_positional / np.format_float_scientific take only a double, so a float16 / float32 value "
           "is printed with float64's shortest digits ('0.0999755859375' for np.float16(0.1)); NumPy prints the "
           "value's own dtype ('0.1')",
    "SU8": "np.format_float_positional / np.format_float_scientific accept a negative precision; NumPy raises "
           "ValueError 'precision must be >= 0'",
    "SU9": "np.format_float_positional ignores fractional=False (precision counts SIGNIFICANT digits there: NumPy "
           "prints 123456789.125 as '123000000.' with precision=3)",
    "SU10": "ndarray.setfield with a Python int outside the field dtype's range wraps it (-1 into a uint8 field is "
            "255); NumPy raises OverflowError 'Python integer -1 out of bounds for uint8'",
    "SU11": "np.finfo of a non-inexact dtype raises 'data type 'int32' not inexact'; NumPy's text names the scalar "
            "class: \"data type <class 'numpy.int32'> not inexact\"",
    "SU12": "np.finfo(float32 / float64 / complex128).minexp is one too high (-125 / -1021); NumPy's is -126 / -1022",
    "SU13": "np.finfo(float16 / float32).resolution is the decimal 10**-precision as a double; NumPy's is that value "
            "rounded to the dtype (np.float32(1e-06) == 9.99999997e-07)",
    "SU14": "np.finfo(float32).tiny / smallest_normal is the double 1.17549435e-38, not exactly 2**-126",
    "SU15": "np.iinfo(bool) returns an iinfo; NumPy raises ValueError \"Invalid integer data type 'b'.\"",
    "SU16": "np.iinfo of a float / complex dtype raises \"Invalid integer data type 'float64'\"; NumPy's text names the "
            "kind character and ends with a period: \"Invalid integer data type 'f'.\"",
    "SU17": "dtype.newbyteorder() of a big-endian descriptor returns the canonical native descriptor (repr "
            "dtype('int32')); NumPy keeps the explicit '<' byte order (repr dtype('<i4'))",
    "SU18": "np.format_float_positional / np.format_float_scientific (unique=True, trim='k') keep the zeros a precision "
            "cutoff carries into: format_float_positional(1.99999, precision=2) is '2.00' and format_float_scientific"
            "(0.0995, precision=1) '1.0e-01'; NumPy's Dragon4 drops the carried digits ('2.', '1.e-01')",
    "SU19": "np.format_float_scientific with trim='-' drops the decimal point when the cutoff leaves only zero "
            "fraction digits: format_float_scientific(0.1001, precision=2, trim='-') is '1e-01'; NumPy keeps it "
            "('1.e-01': its DptZeros mode removes the point only when no fraction digit was generated)",
    "SU20": "np.broadcast(...).iters are independent iterators that start at element 0; NumPy's share the broadcast's "
            "cursor, so after k steps of the broadcast they yield the remaining size - k elements",
    "SU21": "np.get_printoptions() lacks NumPy's 'formatter' and 'override_repr' keys (both None by default): reading "
            "them throws KeyNotFoundException, and the key list differs",
    "SU22": "np.set_printoptions / np.printoptions refuse precision=-1 at once; NumPy stores it and raises 'precision "
            "must be >= 0' only when it formats a float, so an integer array prints",
    "SU23": "array2string with edgeitems=0 under summarization picks the float format from the last element only; "
            "NumPy's _leading_trailing(a, 0) takes a[-0:] (the whole array), so 1e-9 and 1e5 make it scientific",
    "SU24": "dtype.type of a datetime64 / timedelta64 descriptor is null (no CLR element type before Stage C); NumPy's "
            "is numpy.datetime64 / numpy.timedelta64",
    "SU25": "np.nditer in K order over an F-ordered operand and an operand broadcast along an axis iterates C order "
            "(NumSharp sorts the axes by the smallest stride over all operands); NumPy lets each operand vote and the "
            "stride-0 operand abstain, so it iterates F order: other values, chunks and itviews",
    "SU26": "np.nditer.shape without multi_index lists the iterator's axes outermost first; NumPy's NpyIter_GetShape "
            "lists its axisdata innermost first ((3, 2) for a (2, 3) array it cannot coalesce)",
    "SU27": "np.nditer with c_index / f_index never coalesces axes; NumPy coalesces when the tracked index stays "
            "linear (a C array with c_index is a 1-D (6,) iterator), so ndim, shape and itviews differ",
    "SU28": "np.nditer members read past the end (iterindex, value, copy after the last iternext) return a value; "
            "NumPy raises ValueError 'Iterator is past the end'",
    "SU41": "np.nditer refuses a readwrite updateifcopy (copy) operand of another dtype without buffering ('Casting "
            "between different dtypes requires the BUFFERED flag'); NumPy makes a write-back copy that close() resolves "
            "(and raises its own same_kind TypeError for an int32 operand)",
    "SU42": "np.nditer's iteration protocol publishes the current element and THEN advances, where NumPy's __next__ "
            "advances on entry: after k next() calls iterindex / value / multi_index are one element ahead, and a "
            "buffered readwrite loop's write into the last element of each buffer (or each external-loop chunk) is "
            "flushed before the loop body runs, so it never reaches the operand",
    "SU29": "np.nditer after close() throws InvalidOperationException 'Iterator is closed' for every member; NumPy's "
            "iternext() returns False, ndim raises 'Iterator is invalid', value / iterindex 'Iterator is past the end'",
    "SU30": "np.nditer.remove_axis throws IndexOutOfRangeException when the iteration order is not the operands' C "
            "order (order='F', or an F-ordered operand in K order)",
    "SU31": "np.nditer.remove_multi_index re-sorts the axes by stride and coalesces: an order='F' iterator over a C "
            "array becomes a 1-D C walk, and a broadcast row moves its stride-0 axis innermost; NumPy keeps the "
            "iteration order it had",
    "SU32": "np.nditer with delay_bufalloc returns a value before reset(); NumPy raises 'Iterator construction used "
            "delayed buffer allocation, and no reset has been done yet'",
    "SU33": "np.nditer.remove_axis without multi_index is accepted; NumPy raises RuntimeError 'Iterator RemoveAxis may "
            "only be called if a multi-index is being tracked'",
    "SU34": "np.nditer.enable_external_loop while a multi-index is tracked is accepted; NumPy raises 'Iterator flag "
            "EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked'",
    "SU35": "the dtype name 'cdouble' (NumPy's alias of complex128) is not understood: np.dtype / np.finfo raise "
            "\"data type 'cdouble' not understood\"",
    "SU36":"numpy.cumulative_sum / cumulative_prod are aliased to np.cumsum / np.cumprod, which have no "
            "include_initial (NumPy prepends the identity: cumulative_sum([1, 2], include_initial=True) is [0, 1, 3])",
    "SU37": "np.cumsum / np.cumprod (the cumulative_* aliases) flatten an N-D input without axis; numpy.cumulative_sum / "
            "cumulative_prod raise ValueError 'For arrays which have more than one dimension ``axis`` argument is "
            "required.'",
    "SU38": "np.cumsum / np.cumprod of an integer input with dtype=complex128 return the input's raw bytes read as "
            "complex values (the cast to the accumulator dtype is skipped): silent wrong values",
    "SU39": "np.cumsum(x, dtype=bool) accumulates counts into the bool result (bytes 2, 3, ...); NumPy's bool add loop is "
            "logical OR, so the result is True from the first nonzero element on",
    "SU40": "np.cumprod with an axis out of range raises ArgumentOutOfRangeException 'Specified argument was out of the "
            "range of valid values.'; NumPy raises AxisError 'axis 1 is out of bounds for array of dimension 1' "
            "(np.cumsum matches)",
}

# The np.nditer configs that keep K order (order_c / order_f force one) and the members whose answer follows the
# iteration order.
_K_CONFIGS = {"plain", "multi", "c_index", "f_index", "extloop", "buffered", "ranged"}
_ORDER_MEMBERS = {"value", "copy", "reset", "iternext", "iterindex", "itviews", "enable_external_loop"}


def _nditer_gap(case):
    """The gap an np.nditer case falls into (SU25-SU34), or None.

    Parameters:
        case: an nditer.* case (layout surface:objects:<set>:<config>[:k<steps> | :ax<axis>], where <set> is
            one:<kind>, pair or pair_f).

    Returns:
        A GAPS id or None.
    """
    op, p, err = case["op"], case["params"], case.get("error")
    if op == "nditer.close" and "after" in p:
        return "SU29"
    if op == "nditer.close" and "writeback" in p:
        flags = [f for fl in (p.get("op_flags") or []) for f in fl]
        if "updateifcopy" in flags and "buffered" not in p.get("flags", []) and (
                not err or err["text"].startswith("Iterator requested dtype could not be cast")):
            return "SU41"
        if "buffered" in p.get("flags", []) and not err:
            return "SU42"
        return None  # the plain readwrite write-back and the read-only refusals match NumPy
    if op.startswith("nditer.") and p.get("next") and not err:
        # After the last element NumSharp's buffered iterator is past the end but its buffer still holds that element,
        # so value matches NumPy's (which has not advanced yet) there; iterindex does not.
        if p["attr"] == "value" and "buffered" in p.get("flags", []) and p["next"] == 6:
            return None
        return "SU42"
    if err:
        text = err["text"]
        if text == "Iterator is past the end":
            return "SU28"
        if text.startswith("Iterator construction used delayed buffer allocation"):
            return "SU32"
        if text.startswith("Iterator RemoveAxis may only be called"):
            return "SU33"
        if text.startswith("Iterator flag EXTERNAL_LOOP cannot be used"):
            return "SU34"
        return None
    parts = case["layout"].split(":")[2:]
    if len(parts) < 2:
        return None  # the single-label refusal / delayed-allocation cases: only their errors above diverge
    set_label = ":".join(parts[:2]) if parts[0] == "one" else parts[0]
    cfg = parts[2] if parts[0] == "one" else parts[1]
    attr = p["attr"]
    multi = "multi_index" in p["flags"]
    o = np_ops(case["operands"])
    if attr == "remove_axis":
        return "SU30" if cfg == "order_f" or set_label == "one:f" else None
    if attr == "remove_multi_index":
        if (cfg == "order_f" and set_label != "one:f") or set_label in ("pair", "pair_f"):
            return "SU31"
        return "SU26" if set_label == "one:bcast" else None
    if set_label == "pair_f" and cfg in _K_CONFIGS and attr in _ORDER_MEMBERS:
        # NumSharp iterates this set in C order: NumPy's answer with order='C' is NumSharp's.
        as_c = _nditer_attr(o, {**p, "order": "C"})
        same = (len(as_c) == len(case["expected"]["slots"]) and all(
            _array_record(a) == s for a, s in zip(as_c, case["expected"]["slots"]))) if isinstance(as_c, list) \
            else _array_record(as_c) == {k: case["expected"][k] for k in ("dtype", "shape", "buffer")}
        return None if same else "SU25"
    if attr in ("ndim", "shape", "itviews") and cfg in ("c_index", "f_index"):
        if _nditer(o, p).ndim < np.broadcast(*o).ndim:
            return "SU27"
    if attr == "shape" and not multi and not (set_label == "pair_f" and cfg in _K_CONFIGS):
        shape = tuple(int(x) for x in np.frombuffer(bytes.fromhex(case["expected"]["buffer"]), dtype=np.int64))
        if len(shape) >= 2 and shape != shape[::-1]:
            return "SU26"
    return None


def _edge0_model(a, opts):
    """NumSharp's array2string of a 1-D array under threshold / edgeitems=0 summarization, modelled: '[... ' + the
    format of the LAST element alone (NumPy's text for a 2-D array or no summarization is returned as NumSharp's)."""
    if a.ndim != 1 or a.size <= opts.get("threshold", 1000):
        return None
    return "[... " + np.array2string(a[-1:])[1:]


def _rounding_carries(fn, x, prec):
    """Whether cutting the double ``x`` at ``prec`` digits rounds UP through a trailing 9 (1.99999 at one digit is
    2.0): the case where NumSharp's Dragon4 keeps the carried zeros (SU18).

    Parameters:
        fn: np.format_float_positional or np.format_float_scientific (decides where the digits are counted from).
        x: the value (a Python float).
        prec: the precision (>= 1; a zero precision has no fraction digit to keep).

    Returns:
        True when the shortest round-trip digits are longer than the precision (so unique mode cuts them at all), the
        correctly rounded digits are the truncated digits plus one, and the last truncated digit is 9.
    """
    from decimal import Decimal
    shortest = fn(x, trim="-").lstrip("+-").partition("e")[0]
    if len(shortest.partition(".")[2]) <= prec:
        return False  # unique mode prints the shortest digits whole: no cutoff, nothing carried
    s = fn(x, precision=prec, unique=False, trim="k")
    mant, _, exp = s.lstrip("+-").partition("e")
    d = Decimal(abs(x))
    scientific = fn is np.format_float_scientific
    e = d.adjusted() if scientific else 0
    truncated = int(d.scaleb(prec - e))
    rounded = int(mant.replace(".", "")) * (10 ** (int(exp) - e) if scientific else 1)
    return rounded == truncated + 1 and truncated % 10 == 9


def _ns_format_model(fn, x, p):
    """NumSharp's formatter, modelled: NumPy's on the double ``x`` (fractional=False ignored), with NumSharp's two
    Dragon4 differences applied.

    Parameters:
        fn: np.format_float_positional or np.format_float_scientific.
        x: the value (a Python float — NumSharp's only overload takes a double).
        p: the case params.

    Returns:
        (text, rule): the modelled text and the GAPS id of the difference applied ("SU18" / "SU19"), or None.
    """
    q = {k: v for k, v in p.items() if k != "fractional"}
    out = fn(x, **q)
    prec = q.get("precision")
    if (q.get("unique", True) and prec is not None and prec >= 1 and q.get("trim", "k") == "k"
            and _rounding_carries(fn, x, prec)):
        return fn(x, **{**q, "unique": False}), "SU18"
    if fn is np.format_float_scientific and q.get("trim") == "-" and ".e" in out:
        return out.replace(".e", "e"), "SU19"
    return out, None


def _format_float_gap(case):
    """The gap a format_float_positional / format_float_scientific case falls into, decided by modelling NumSharp's
    double-only formatter (_ns_format_model) and comparing with the recorded NumPy text.

    Parameters:
        case: a format_float_* case.

    Returns:
        SU6 / SU7 / SU8 / SU9 / SU18 / SU19, or None when the model matches NumPy.
    """
    p = case["params"]
    if p.get("precision") is not None and p["precision"] < 0:
        return "SU8"
    v = rl.rebuild(case["operands"][0])[()]
    if not np.isfinite(v):
        return "SU6"
    fn = np.format_float_positional if case["op"] == "format_float_positional" else np.format_float_scientific
    expected = case["expected"]["value"]
    modelled, rule = _ns_format_model(fn, float(v), p)
    if modelled == expected:
        return None
    as_double = {k: x for k, x in p.items() if k != "fractional"}
    if p.get("fractional") is False and fn(float(v), **p) != fn(float(v), **as_double):
        return "SU9"
    if rule is not None:
        return rule
    return "SU7"


def _gap_of(case):
    """The documented gap a case falls into (a GAPS id), or None when NumSharp is expected to match NumPy.

    Parameters:
        case: the finished case.

    Returns:
        A GAPS id or None.

    Notes:
        Every rule is exact: it names precisely the cases NumSharp gets wrong today (checked by replaying the uncarved
        matrix), so a fix shows up as a gap whose carved cases all match NumPy.
    """
    op, p, err = case["op"], case["params"], case.get("error")
    # finfo / iinfo take dtype SPELLINGS ('half', 'f4', 'intp', ...): the rules key on the dtype they name.
    dt = np.dtype(p["dtype"]).name if op.startswith(("finfo.", "iinfo.")) else p.get("dtype")
    if op == "linalg.cross" and err:
        if err["type"] == "IndexError":
            return "SU1"
        if err["text"].startswith("shape mismatch"):
            return "SU2"
    if op in ("histogram", "histogram_bin_edges") and case["operands"][0]["dtype"] == "complex128":
        if isinstance(p.get("bins"), str):
            return "SU4"
        if "bins_operand" not in p:
            return "SU3"
    if op == "apply_along_axis" and ":scalar_bcast_offset:" in case["layout"]:
        return "SU5"
    if op in ("format_float_positional", "format_float_scientific"):
        return _format_float_gap(case)
    if op == "ndarray.setfield" and err and err["type"] == "OverflowError":
        return "SU10"
    if op.startswith("finfo."):
        if p["dtype"] == "cdouble":
            return "SU35"
        if err:
            return "SU11"
        if op == "finfo.minexp" and dt in ("float32", "float64", "complex128"):
            return "SU12"
        if op == "finfo.resolution" and dt in ("float16", "float32"):
            return "SU13"
        if op in ("finfo.tiny", "finfo.smallest_normal") and dt == "float32":
            return "SU14"
    if op.startswith("iinfo.") and err:
        return "SU15" if dt == "bool" else "SU16"
    if op == "dtype.newbyteorder" and p["dtype"] in (">i4", ">c16"):
        return "SU17"
    if op == "broadcast.iters" and p.get("steps", 0) > 0:
        return "SU20"
    if op == "get_printoptions" and p["key"] in (None, "formatter", "override_repr"):
        return "SU21"
    if op in ("set_printoptions", "printoptions") and not err:
        opts = p["opts"]
        if opts.get("precision", 0) < 0:
            return "SU22"
        if opts.get("edgeitems") == 0 and p.get("phase") != "after":
            model = _edge0_model(rl.rebuild(case["operands"][0]), opts)
            if model is not None and model != case["expected"]["value"]:
                return "SU23"
    if op == "dtype.type" and p["dtype"][0] in "Mm":
        return "SU24"
    if op.startswith("nditer."):
        return _nditer_gap(case)
    if op in ("cumulative_sum", "cumulative_prod"):
        if p.get("include_initial"):
            return "SU36"
        if err and err["text"].startswith("For arrays which have more than one dimension"):
            return "SU37"
        if p.get("dtype") == "complex128":
            return "SU38"
        if op == "cumulative_sum" and p.get("dtype") == "bool":
            return "SU39"
        if op == "cumulative_prod" and err and err["type"] == "AxisError":
            return "SU40"
    return None


# =============================================================================================================
# Main
# =============================================================================================================

SECTIONS = [gen_linalg, gen_histogram, gen_equiv_memory_shapes, gen_apply_bmat, gen_text, gen_ndarray,
            gen_attributes, gen_objects, gen_aliases]


def write_jsonl(path, cases):
    """Write cases as JSONL (LF line ends, compact separators) and report the count."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        for c in cases:
            # allow_nan=False: the C# reader is strict JSON, so a NaN / inf that slipped into a param must fail here.
            f.write(json.dumps(c, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"wrote {len(cases)} cases -> {path}")


def main():
    """Generate the surface tier (see the module docstring for the flags)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--uncarved", metavar="DIR", help="write the whole matrix (no carve) into DIR instead")
    args = ap.parse_args()
    for section in SECTIONS:
        section()
    counts = collections.Counter(c["op"] for c in CASES)
    print(f"{len(CASES)} cases over {len(counts)} op keys")
    if args.uncarved:
        write_jsonl(os.path.join(args.uncarved, "surface.jsonl"), CASES)
        return
    kept, carved = [], []
    for c in CASES:
        gap = _gap_of(c)
        if gap:
            c["gap"] = gap
            carved.append(c)
        else:
            kept.append(c)
    write_jsonl(os.path.join(CORPUS, "surface.jsonl"), kept)
    write_jsonl(os.path.join(GAP_DIR, "surface_gaps.jsonl"), carved)


if __name__ == "__main__":
    main()
