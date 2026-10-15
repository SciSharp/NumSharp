#!/usr/bin/env python3
"""
gen_matrix_oracle.py -- the numpy.matrix oracle: NumPy 2.4.2's answers for the matrix class (NumSharp's NDMatrix).

numpy.matrix is an ndarray SUBCLASS, so a result here is more than (dtype, shape, bytes): the replay also holds
NumSharp to the CLASS of the returned object (a matrix, a plain ndarray, or a scalar), to its memory facet (view /
copy / identity against each operand, the five flags, the byte strides -- gen_viewflags_oracle's ``mem``), and
for the calls NumPy refuses to the exception type and its verbatim text.

Tier (one file under ../NumSharp.Tests.Oracle/Fuzz/corpus/, replayed by FuzzCorpusTests.Matrix.cs):

  matrix.jsonl   op keys are the API names, so the coverage join credits them to the numpy.matrix rows:
                   matrix                 np.matrix(data, dtype, copy) from an array, a matrix, a literal, a list
                   asmatrix               np.asmatrix(data, dtype)
                   matrix.<member>        the members: T H A A1 I getT.. , __getitem__ / __setitem__, the operators
                                          (__mul__ / __rmul__ / __imul__ / __pow__ / __ipow__ / __rpow__ / __add__ ...),
                                          the reductions, the methods ndarray hands down, __repr__ / __str__ / __iter__

Case schema (the ordinary FuzzCorpus schema plus gen_viewflags_oracle's ``own`` / ``mem``), and:

  operand.matrix   true when the operand is a numpy.matrix; the replay rebuilds it as an NDMatrix.
  expected.cls     "matrix" | "ndarray" | "scalar": the class NumPy returned. A scalar result carries no ``mem``
                   (NumSharp answers with a 0-d array, whose flags are a view's).
  params           per op; the twin of every reader is OpRegistry.Matrix.cs. A binary operator case carries
                   ``sym``, the C# operator applied to (operand 0, operand 1) in that order, so the key (which
                   names the Python method NumPy dispatched to) never has to be decoded.

Operand kinds. A matrix operand is either a VIEW of a 2-d source layout (``np.matrix(a, copy=False)``: C / F owners,
transposed, stepped, reversed and offset views, a read-only owner, a broadcast view, a row, a column, 1x1, empty)
or an OWNER (``np.matrix(a)``). Values are gen_viewflags_oracle._vals: small integers every dtype holds exactly,
so a value difference can only mean a wrong element, a wrong dtype or a wrong operation, never rounding.

Scalars. A NumPy scalar operand (``m * np.int8(2)``) is recorded as a 0-d operand and replayed as a 0-d NDArray.
An explicit 0-d ARRAY operand (``m * np.array(2)``, an ndarray to ``matrix.__mul__``: the product with a 1 x 1
matrix) is recorded the same way, and ``params.array0d`` lists its positions: the replay marks those operands as
np.array / np.asarray would (NDArray.IsExplicitZeroDimArray).
A Python scalar (``m * 2``) is recorded in ``params.py`` and replayed as the C# literal of the same kind (long,
double, bool, Complex), which converts implicitly to a 0-d NDMatrix.

What is NOT here, and why:
  * matrix.I of a general matrix, a negative matrix power, and every float product a BLAS would reorder: NumSharp's
    managed LU / GEMM agree with LAPACK / OpenBLAS to tolerance only. The inverse cases here are diagonal and
    permutation matrices, where every implementation is exact.
  * a matrix literal whose token is a NAME ("nan"): CPython's message prints the address of an ast node.
  * tuple axes of the reductions: NumSharp's reductions take one axis (a library-wide gap).
  * a Python LIST as an assigned value: NumPy's assignment through a matrix view has list-only quirks (a nested
    list raises RuntimeError, a flat list into m[0, 1:3] writes nothing); arrays, matrices and scalars are covered.

Carving (house rule: real product bugs are carved + pinned, never registry-excused): ``_gap_of`` names the gap a
case pins (GAPS); carved cases go to gaps/matrix_gaps.jsonl and are replayed by the [OpenBugs] test
FuzzCorpusTests.Matrix_KnownGaps. ``--uncarved DIR`` writes the whole matrix.

Run (needs numpy==2.4.2):
    python test/oracle/gen_matrix_oracle.py [--uncarved DIR]
"""
import argparse
import os
import warnings

import numpy as np

from gen_viewflags_oracle import (ALL_DTYPES, CORPUS, GAP_DIR, _desc, _exc, _mem, _off_view, _owner, _vals, _value,
                                  write_jsonl)

np.seterr(all="ignore")
warnings.simplefilter("ignore")

assert np.__version__ == "2.4.2", f"corpus must be generated with numpy 2.4.2, got {np.__version__}"

# The dtype subsets the sections iterate: everything, and small sets that still cross every dtype kind.
KINDS4 = ["bool", "int64", "float64", "complex128"]
KINDS6 = ["bool", "int8", "uint64", "int64", "float32", "complex128"]
INTS = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]
FLOATS = ["float16", "float32", "float64"]


# =============================================================================================================
# Sources
# =============================================================================================================

# name -> builder(dt) -> a 2-d ndarray layout. A matrix VIEW of it is np.matrix(a, copy=False).
SOURCES = {
    "c": lambda dt: _owner((3, 4), dt),
    "f": lambda dt: _owner((3, 4), dt, order="F"),
    "t": lambda dt: _owner((4, 3), dt).T,
    "step": lambda dt: _owner((6, 8), dt)[::2, ::2],
    "rev": lambda dt: _owner((3, 4), dt)[::-1, ::-1],
    "off": lambda dt: _off_view(dt),
    "ro": lambda dt: _owner((3, 4), dt, readonly=True),
    "bcast": lambda dt: np.broadcast_to(_owner((4,), dt), (3, 4)),
    "row": lambda dt: _owner((1, 4), dt),
    "col": lambda dt: _owner((4, 1), dt),
    "one": lambda dt: _owner((1, 1), dt),
    "e03": lambda dt: _owner((0, 3), dt),
    "e30": lambda dt: _owner((3, 0), dt),
    "sq": lambda dt: _owner((3, 3), dt),
}
ALL_SRC = list(SOURCES)
MAIN_SRC = ["c", "f", "t", "step", "rev", "off", "ro", "bcast"]
WRITABLE_SRC = ["c", "f", "t", "step", "rev", "off"]


def mview(name, dt):
    """A matrix VIEW of one source layout.

    Parameters:
        name: a key of SOURCES.
        dt: the dtype name.

    Returns:
        ``np.matrix(source, copy=False)``: never an owner, read-only when the source is. An EMPTY source is the
        exception: a view of no element has no buffer the corpus could record, and the replay rebuilds an empty
        operand as an owner, so the matrix is an owner here too (``flags.owndata`` then agrees on both sides).
    """
    source = SOURCES[name](dt)
    if source.size == 0:
        return np.matrix(source)
    return np.matrix(source, copy=False)


def mown(shape, dt, order="C"):
    """A matrix that OWNS its data.

    Parameters:
        shape: the (rows, columns) shape.
        dt: the dtype name.
        order: 'C' or 'F', the memory order of the owner.

    Returns:
        A writeable matrix with ``flags.owndata`` set, filled with ``_vals`` in logical C order.
    """
    out = np.matrix(_owner(shape, dt, order=order), copy=False).copy(order=order)
    assert out.flags.owndata
    return out


def desc(a):
    """Serialize one operand.

    Parameters:
        a: an ndarray or a numpy.matrix.

    Returns:
        gen_viewflags_oracle._desc's descriptor, plus ``matrix: true`` for a numpy.matrix.

    Raises:
        ValueError: the operand cannot be serialized as raw memory (see ``_desc``).
    """
    d = _desc(a)
    if isinstance(a, np.matrix):
        d["matrix"] = True
    return d


def expect(r, operands):
    """The expected record of one result.

    Parameters:
        r: what NumPy returned: a matrix, an ndarray, or a scalar (a NumPy scalar or a Python number).
        operands: the operand arrays the memory facet is taken against.

    Returns:
        (dtype, shape, C-order bytes) plus ``cls`` and, for an array, ``mem``.
    """
    if not isinstance(r, np.ndarray):
        e = _value(np.asarray(r))
        e["cls"] = "scalar"
        return e
    e = _value(r)
    e["mem"] = _mem(r, operands)
    e["cls"] = "matrix" if isinstance(r, np.matrix) else "ndarray"
    return e


class Emitter:
    """Collects the cases of the tier; ``call`` runs one NumPy call and records its result or its exception."""

    def __init__(self):
        """Start with no cases.

        Returns:
            None.
        """
        self.cases = []
        self.n = 0
        self.skipped = 0

    def _id(self, op, layout, dt):
        """The next case id.

        Parameters:
            op: the op key.
            layout: the layout label.
            dt: the dtype label.

        Returns:
            ``op/layout/dtype/n`` with a running counter, unique inside the tier.
        """
        self.n += 1
        return f"{op}/{layout}/{dt}/{self.n}"

    def call(self, op, params, ops, fn, layout, dt, kind=None):
        """Run ``fn(ops)`` and record it as one case.

        Parameters:
            op: the op key.
            params: the params dict the C# twin reads.
            ops: the operand arrays (described BEFORE the call, so a mutating call records its inputs).
            fn: callable(ops) -> the result: an ndarray / matrix / scalar, a str for kind "text", a sequence of
                arrays for kind "tuple".
            layout: the layout label of the case.
            dt: the dtype label of the case.
            kind: None (one array or scalar), "text" or "tuple".

        Returns:
            None. A NumPy exception becomes an error case (type and verbatim text); a case whose operand cannot
            be serialized, or whose error text holds a CPython object address, is skipped and counted.
        """
        try:
            descs = [desc(o) for o in ops]
        except ValueError:
            self.skipped += 1
            return
        cid = self._id(op, layout, dt)
        try:
            r = fn(ops)
        except Exception as e:  # NumPy refuses: the replay holds NumSharp to the type and the text
            if "object at 0x" in str(e):
                self.skipped += 1
                return
            expected = {"kind": kind} if kind else {}
            self.cases.append({"id": cid, "op": op, "params": params, "operands": descs, "expected": expected,
                               "expects_throw": True, "error": _exc(e), "layout": layout, "valueclass": "matrix"})
            return
        if kind == "text":
            expected = {"kind": "text", "value": r}
        elif kind == "tuple":
            expected = {"kind": "tuple", "slots": [expect(s, ops) for s in r]}
        else:
            expected = expect(r, ops)
        self.cases.append({"id": cid, "op": op, "params": params, "operands": descs, "expected": expected,
                           "layout": layout, "valueclass": "matrix"})


# =============================================================================================================
# Section C: construction (np.matrix, np.asmatrix)
# =============================================================================================================

def _nd_sources(dt):
    """The ndarray sources of the constructor: every rank it folds or refuses.

    Parameters:
        dt: the dtype name.

    Returns:
        {label: ndarray}: 0-d, 1-d and 2-d layouts, arrays with more than two axes that fold onto two, and ones
        that cannot.
    """
    return {
        "0d": _owner((), dt),
        "1d": _owner((5,), dt),
        "1d_step": _owner((10,), dt)[::2],
        "1d_rev": _owner((5,), dt)[::-1],
        "1d_one": _owner((1,), dt),
        "1d_empty": _owner((0,), dt),
        "2d_c": _owner((3, 4), dt),
        "2d_f": _owner((3, 4), dt, order="F"),
        "2d_t": _owner((4, 3), dt).T,
        "2d_step": _owner((6, 8), dt)[::2, ::2],
        "2d_rev": _owner((3, 4), dt)[::-1, ::-1],
        "2d_off": _off_view(dt),
        "2d_ro": _owner((3, 4), dt, readonly=True),
        "2d_bcast": np.broadcast_to(_owner((4,), dt), (3, 4)),
        "2d_e03": _owner((0, 3), dt),
        "3d_123": _owner((1, 2, 3), dt),
        "3d_231": _owner((2, 3, 1), dt),
        "3d_115": _owner((1, 1, 5), dt),
        "3d_111": _owner((1, 1, 1), dt),
        "4d_1221": _owner((1, 2, 2, 1), dt),
        "5d_13141": _owner((1, 3, 1, 4, 1), dt),
        "3d_222": _owner((2, 2, 2), dt),
        "3d_234": _owner((2, 3, 4), dt),
        "3d_031": _owner((0, 3, 1), dt),
        "3d_perm": _owner((3, 1, 2), dt).transpose(2, 1, 0),
    }


# (label, text): literals np.matrix / np.asmatrix accept or refuse. No NAME tokens (see the module docstring).
LITERALS = [
    ("ints", "1 2; 3 4"), ("commas", "1,2,3"), ("mixed_sep", "1, 2 3;4 5,6"), ("brackets", "[1 2; 3 4]"),
    ("float", "1.5 2"), ("leading_dot", ".5 1"), ("trailing_dot", "5. 1"), ("exp", "1e3 2"), ("exp_sign", "1e-3 2E+2"),
    ("signs", "+5 -3"), ("neg_zero", "-0.0 0.0"), ("inf_literal", "1e400 -1e400"), ("complex", "1+2j 3-4j"),
    ("imag", "2j 1"), ("imag_neg", "-2j 1"), ("complex_float", "1.5+2.5j 0"), ("paren", "(1) (2.5)"),
    ("paren_complex", "(1+2j) 3"), ("bools", "True False; False True"), ("bool_int", "True 2"),
    ("bool_float", "True 2.5"), ("hex", "0x10 0X1f"), ("oct", "0o17 0O7"), ("bin", "0b101 0B11"),
    ("underscore", "1_000 2_0"), ("zeros", "0 00 0_0"), ("big", "9223372036854775807 1"),
    ("uint64", "9223372036854775808 1"), ("uint64_max", "18446744073709551615"),
    ("uint64_int64", "9223372036854775808 -1"), ("int_min", "-9223372036854775808 0"),
    ("three_rows", "1 2;3 4;5 6"), ("empty", ""), ("spaces", "   "), ("only_brackets", "[]"),
    ("semicolon", ";"), ("two_semicolons", ";;"), ("trailing_semicolon", "1 2;"), ("tab_newline", "1\t2\n3 4"),
    ("ragged", "1 2; 3 4 5"), ("ragged_empty_first", "; 1"), ("py2_long", "1L 2"), ("letters", "12abc"),
    ("lone_plus", "1 + 2"), ("lone_minus", "- 1 2"), ("leading_zero", "012 3"), ("bad_hex", "0x 1"),
    ("bad_oct_digit", "0o8"), ("bad_bin_digit", "0b2"), ("double_underscore", "1__0"), ("trailing_underscore", "1_"),
    ("bad_exp", "1e 2"), ("open_paren", "(1 2"), ("close_paren", "1) 2"), ("neg_bool", "-True"),
    ("double_neg", "--1"), ("expr_mul", "2*3"), ("expr_sum3", "1+2j+3j"), ("imag_plus_real", "2j+1"),
    ("neg_paren", "-(1)"), ("float_overflow_neg", "-1e999"), ("complex_neg_real", "-1-2j"),
]

# (label, nested data): Python values np.matrix coerces. Leaves are tagged for the C# side by _leaf.
LISTS = [
    ("flat_int", [1, 2, 3]), ("flat_float", [1.5, 2.0]), ("flat_bool", [True, False]),
    ("flat_complex", [1 + 2j, 3j]), ("mixed", [1, 2.5]), ("bool_int", [True, 2]), ("rows", [[1, 2], [3, 4]]),
    ("rows_float", [[1.0, 2], [3, 4]]), ("empty", []), ("empty_row", [[]]), ("two_empty_rows", [[], []]),
    ("ragged", [[1, 2], [3]]), ("deep", [[[1, 2]]]), ("scalar_int", 5), ("scalar_float", 2.5),
    ("scalar_bool", True), ("big_uint", [2 ** 63, 1]), ("uint_and_neg", [2 ** 63, -1]),
]


def _leaf(v):
    """Tag one Python value for the C# twin.

    Parameters:
        v: a (nested) list, or a bool / int / float / complex.

    Returns:
        The same nesting with every scalar as ``{"b": bool}``, ``{"i": "digits"}``, ``{"f": "repr"}`` or
        ``{"c": ["re", "im"]}``, so the Python type survives JSON.
    """
    if isinstance(v, list):
        return [_leaf(x) for x in v]
    if isinstance(v, bool):
        return {"b": v}
    if isinstance(v, int):
        return {"i": str(v)}
    if isinstance(v, float):
        return {"f": repr(v)}
    return {"c": [repr(v.real), repr(v.imag)]}


def gen_construct(em):
    """np.matrix and np.asmatrix over arrays of every rank, matrices, literals and Python lists.

    Parameters:
        em: the tier's Emitter.

    Returns:
        None.
    """
    for dt in ALL_DTYPES:
        for label, a in _nd_sources(dt).items():
            if dt not in KINDS4 and label not in ("1d", "2d_c", "2d_f"):
                continue
            em.call("matrix", {"src": "op"}, [a], lambda o: np.matrix(o[0]), f"nd:{label}", dt)
            em.call("matrix", {"src": "op", "copy": False}, [a], lambda o: np.matrix(o[0], copy=False),
                    f"nd:{label}:nocopy", dt)
            em.call("asmatrix", {}, [a], lambda o: np.asmatrix(o[0]), f"nd:{label}", dt)
        # A matrix argument: identity without a copy, a K-order copy otherwise.
        for src in ("c", "f", "t", "ro", "row"):
            m = mview(src, dt)
            em.call("matrix", {"src": "op"}, [m], lambda o: np.matrix(o[0]), f"m:{src}", dt)
            em.call("matrix", {"src": "op", "copy": False}, [m], lambda o: np.matrix(o[0], copy=False),
                    f"m:{src}:nocopy", dt)
            em.call("asmatrix", {}, [m], lambda o: np.asmatrix(o[0]), f"m:{src}", dt)

    # dtype= : every source dtype into a few targets, for an array and for a matrix, with and without copy.
    for dt in ALL_DTYPES:
        for to in ("float64", "int32", "bool", "complex128", "float16", dt):
            a = _owner((2, 3), dt)
            m = np.matrix(_owner((2, 3), dt, order="F"), copy=False)
            for copy in (True, False):
                p = {"src": "op", "dtype": to, "copy": copy}
                em.call("matrix", p, [a], lambda o, to=to, copy=copy: np.matrix(o[0], dtype=to, copy=copy),
                        f"nd:dtype:{to}:{'copy' if copy else 'nocopy'}", dt)
                em.call("matrix", p, [m], lambda o, to=to, copy=copy: np.matrix(o[0], dtype=to, copy=copy),
                        f"m:dtype:{to}:{'copy' if copy else 'nocopy'}", dt)
            em.call("asmatrix", {"dtype": to}, [a], lambda o, to=to: np.asmatrix(o[0], dtype=to), f"nd:dtype:{to}", dt)

    for label, text in LITERALS:
        em.call("matrix", {"src": "text", "text": text}, [], lambda o, text=text: np.matrix(text), f"text:{label}", "-")
        em.call("asmatrix", {"text": text}, [], lambda o, text=text: np.asmatrix(text), f"text:{label}", "-")
    for label, text in (("ints", "1 2; 3 4"), ("float", "1.5 2"), ("complex", "1+2j 3"), ("bools", "True False"),
                        ("empty", "")):
        for to in ("float32", "int8", "complex128", "bool", "uint64"):
            em.call("matrix", {"src": "text", "text": text, "dtype": to}, [],
                    lambda o, text=text, to=to: np.matrix(text, dtype=to), f"text:{label}:{to}", "-")

    for label, data in LISTS:
        em.call("matrix", {"src": "list", "data": _leaf(data)}, [], lambda o, data=data: np.matrix(data),
                f"list:{label}", "-")
    for to in ("float32", "int8", "complex128"):
        em.call("matrix", {"src": "list", "data": _leaf([[1, 2], [3, 4]]), "dtype": to}, [],
                lambda o, to=to: np.matrix([[1, 2], [3, 4]], dtype=to), f"list:rows:{to}", "-")


# =============================================================================================================
# Section P: the matrix-only properties
# =============================================================================================================

def gen_properties(em):
    """T / H / A / A1 and their get-methods over every layout and dtype; I where every algorithm is exact.

    Parameters:
        em: the tier's Emitter.

    Returns:
        None.
    """
    for dt in ALL_DTYPES:
        for src in ALL_SRC:
            if dt not in KINDS6 and src not in ("c", "f", "t"):
                continue
            m = mview(src, dt)
            for name in ("T", "H", "A", "A1"):
                em.call(f"matrix.{name}", {}, [m], lambda o, name=name: getattr(o[0], name), src, dt)
            for name in ("getT", "getH", "getA", "getA1"):
                if src in ("c", "t"):
                    em.call(f"matrix.{name}", {}, [m], lambda o, name=name: getattr(o[0], name)(), src, dt)
        own = mown((3, 4), dt)
        for name in ("T", "H", "A", "A1"):
            em.call(f"matrix.{name}", {}, [own], lambda o, name=name: getattr(o[0], name), "own", dt)

    # The inverse of diagonal and permutation matrices is exact in LAPACK and in the managed LU alike.
    exact = {
        "identity": np.eye(3),
        "diag_pow2": np.diag([2.0, 4.0, 0.5]),
        "perm": np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        "one": np.array([[4.0]]),
        "neg_diag": np.diag([-1.0, 8.0]),
    }
    for label, a in exact.items():
        # No complex128: the managed LU and LAPACK agree on the values but not on the sign of a zero imaginary
        # part, which is a property of the two factorisations and not of the matrix class.
        for dt in ("float64", "float32", "int64", "bool", "int8", "uint8", "float16"):
            if dt == "bool" and label not in ("identity", "perm"):
                continue
            if dt == "uint8" and label == "neg_diag":
                continue
            if dt in INTS and label == "diag_pow2":
                continue
            m = np.matrix(a.astype(dt))
            em.call("matrix.I", {}, [m], lambda o: o[0].I, f"I:{label}", dt)
            em.call("matrix.getI", {}, [m], lambda o: o[0].getI(), f"I:{label}", dt)
    # Refusals of the inverse that need no factorisation result.
    em.call("matrix.I", {}, [np.matrix(np.zeros((2, 2)))], lambda o: o[0].I, "I:singular", "float64")
    em.call("matrix.I", {}, [np.matrix(np.zeros((0, 0)))], lambda o: o[0].I, "I:empty", "float64")


# =============================================================================================================
# Section G / S: indexing
# =============================================================================================================

def _tok(entry, extra):
    """Encode one index entry for the C# twin.

    Parameters:
        entry: an int, a slice, Ellipsis, None, an ndarray (an index or a mask) or a list of ints / bools.
        extra: the list an array entry is appended to; the token then names its operand position.

    Returns:
        An int, ``"..."``, ``"newaxis"``, ``{"s": [start, stop, step]}``, ``{"op": position}``, ``{"l": ints}``
        or ``{"lb": bools}``.
    """
    if entry is Ellipsis:
        return "..."
    if entry is None:
        return "newaxis"
    if isinstance(entry, slice):
        return {"s": [entry.start, entry.stop, entry.step]}
    if isinstance(entry, np.ndarray):
        extra.append(entry)
        return {"op": len(extra)}          # operand 0 is the matrix
    if isinstance(entry, list):
        return {"lb": entry} if entry and isinstance(entry[0], bool) else {"l": entry}
    return int(entry)


def _slice_text(entry):
    """Python slice notation of one basic index entry, for NumSharp's string indexer.

    Parameters:
        entry: an int, a slice, Ellipsis or None.

    Returns:
        The text: ``"3"``, ``"1:3"``, ``"::2"``, ``"..."`` or ``"np.newaxis"``.
    """
    if entry is Ellipsis:
        return "..."
    if entry is None:
        return "np.newaxis"
    if isinstance(entry, slice):
        part = lambda v: "" if v is None else str(v)
        text = f"{part(entry.start)}:{part(entry.stop)}"
        return text + (f":{entry.step}" if entry.step is not None else "")
    return str(int(entry))


S = slice
BASIC_INDEXES = [
    0, -1, 2, (0, 0), (-1, -1), (1, 2), (S(None), 0), (S(None), -1), (0, S(None)), (S(1, 3),), (S(1, 3), S(1, 3)),
    (S(None, None, 2), S(None, None, -1)), (Ellipsis, 0), (0, Ellipsis), (Ellipsis,), (S(None),), (None,),
    (0, None), (S(0, 0),), (S(None), S(0, 0)), (S(1, 3), 0), (0, S(1, 3)), (S(None, None, -1),), (-1, S(None, None, 2)),
    3, -4, (0, 4), (0, 0, 0), (S(None), S(None), 0), (S(None), 9), (Ellipsis, Ellipsis),
]


def _index_params(index):
    """The params and the extra operands of one index expression.

    Parameters:
        index: the Python index (one entry, or a tuple of entries).

    Returns:
        ({"index": [tokens], "tuple": bool}, [array operands the tokens reference]).
    """
    extra = []
    is_tuple = isinstance(index, tuple)
    items = index if is_tuple else (index,)
    return {"index": [_tok(x, extra) for x in items], "tuple": is_tuple}, extra


def _index_text(index):
    """The slice-notation string of a basic index, for the string indexer.

    Parameters:
        index: the Python index (one entry, or a tuple of entries), integers / slices / Ellipsis / None only.

    Returns:
        The entries joined with ``", "``.
    """
    items = index if isinstance(index, tuple) else (index,)
    return ", ".join(_slice_text(x) for x in items)


def gen_getitem(em):
    """matrix.__getitem__: the row / column / scalar rule over every index form and layout; iteration.

    Parameters:
        em: the tier's Emitter.

    Returns:
        None.
    """
    for dt in ALL_DTYPES:
        for src in ALL_SRC:
            if dt not in KINDS4 and src != "c":
                continue
            for index in BASIC_INDEXES:
                p, extra = _index_params(index)
                vias = ["obj"]
                if src in ("c", "t", "row", "col") and dt in ("int64", "float64"):
                    vias += ["typed", "str"]
                for via in vias:
                    q = dict(p, via=via)
                    if via == "str":
                        q["str"] = _index_text(index)
                    em.call("matrix.__getitem__", q, [mview(src, dt)] + extra, lambda o, index=index: o[0][index],
                            f"{src}:{via}", dt)

    # Index arrays, masks, lists: a 1-d result is a row unless entry #1 is a scalar.
    for dt in KINDS4:
        for src in ("c", "f", "t", "rev", "bcast", "row", "col", "sq"):
            r, c = mview(src, dt).shape
            rows = [0, r - 1] if r > 1 else [0, 0]
            cols = [0, c - 1] if c > 1 else [0, 0]
            adv = [
                rows, (rows, cols), (min(1, r - 1), cols + [0]), (rows + [0], min(1, c - 1)), (S(None), cols),
                (rows, S(None)), np.array(rows), (np.array(rows), np.array(cols)), np.array(min(1, r - 1)),
                (S(None, 2), np.array(-1)), np.array([rows, rows[::-1]]), (np.array(rows)[::-1],),
                (S(None), np.array(cols, dtype=np.int32)), [True] + [False] * (r - 1),
                (S(None), [True] + [False] * (c - 1)), np.array([i % 2 == 0 for i in range(r)]),
                np.array(_vals(r * c, "bool")).reshape(r, c), np.matrix(_vals(r * c, "bool").reshape(r, c)),
                (np.array([i % 2 == 0 for i in range(r)]), 0), [r], ([0], [c]),
            ]
            for index in adv:
                p, extra = _index_params(index)
                em.call("matrix.__getitem__", dict(p, via="obj"), [mview(src, dt)] + extra,
                        lambda o, index=index: o[0][index], f"{src}:adv", dt)

    # A mask from a comparison of the matrix itself, and iteration over the rows.
    for dt in ALL_DTYPES:
        for src in ("c", "t", "rev"):
            m = mview(src, dt)
            mask = np.asarray(m) != np.asarray(m).flat[0]
            em.call("matrix.__getitem__", {"index": [{"op": 1}], "tuple": False, "via": "obj"}, [m, np.matrix(mask)],
                    lambda o: o[0][o[1]], f"{src}:mask", dt)
            em.call("matrix.__iter__", {}, [m], lambda o: list(o[0]), src, dt, kind="tuple")
    for src in ("row", "col", "one", "e03", "e30"):
        em.call("matrix.__iter__", {}, [mview(src, "int64")], lambda o: list(o[0]), src, "int64", kind="tuple")


def _value_operands(rows, cols, dt):
    """The values assigned through an index: shapes that broadcast, or fail, against a row, a column, a block.

    Parameters:
        rows: the row count of the assigned-to matrix.
        cols: its column count.
        dt: the dtype name.

    Returns:
        {label: ndarray or matrix}.
    """
    return {
        "scalar": np.array(_vals(3, dt)[2]),
        "v_c": _owner((cols,), dt),
        "v_r": _owner((rows,), dt),
        "row": _owner((1, cols), dt),
        "col": _owner((rows, 1), dt),
        "one": _owner((1,), dt),
        "one2": _owner((1, 1), dt),
        "full": _owner((rows, cols), dt),
        "mrow": np.matrix(_owner((1, cols), dt)),
        "mcol": np.matrix(_owner((rows, 1), dt)),
        "v2": _owner((2,), dt),
        "b22": _owner((2, 2), dt),
        "c21": _owner((2, 1), dt),
    }


SET_INDEXES = [
    0, -1, (S(None), 0), (S(None), -1), (0, S(None)), (0, 0), (-1, -1), (S(None),), Ellipsis, (S(1, 3), S(1, 3)),
    (S(1, 3),), (0, S(1, 3)), (S(1, 3), 0), (S(None, None, 2), S(None, None, -1)), 3, (0, 0, 0),
]


def _set_advanced(rows, cols):
    """The index-array and mask indexes of the assignment section.

    Parameters:
        rows: the row count of the assigned-to matrix.
        cols: its column count.

    Returns:
        A list of Python indexes that take ndarray's plain assignment path.
    """
    return [
        [0, rows - 1], ([0, rows - 1], [1, cols - 1]), (S(None), [0, cols - 1]), (1, [0, cols - 1]),
        ([0, rows - 1], 1), np.array([i % 2 == 0 for i in range(rows)]),
        (S(None), np.array([i % 2 == 0 for i in range(cols)])),
    ]


def _assign(o, index):
    """Assign the last operand through ``index`` into the first and return the first.

    Parameters:
        o: the operands: the matrix, the index arrays (if any), the value.
        index: the Python index.

    Returns:
        The mutated matrix.
    """
    o[0][index] = o[len(o) - 1]
    return o[0]


def gen_setitem(em):
    """matrix.__setitem__: the value is broadcast into the view the matrix's own indexing returns.

    Parameters:
        em: the tier's Emitter.

    Returns:
        None.
    """
    for dt in KINDS4 + ["int8", "float32"]:
        for src in WRITABLE_SRC + ["ro"]:
            if src not in ("c", "t", "ro") and dt != "int64":
                continue
            r, c = mview(src, dt).shape
            values = _value_operands(r, c, dt)
            few = src != "c" or dt not in KINDS4
            for index in SET_INDEXES + _set_advanced(r, c):
                p, extra = _index_params(index)
                for vname, value in values.items():
                    if src == "ro" and vname not in ("scalar", "row"):
                        continue
                    if few and vname not in ("scalar", "v_c", "v_r", "row", "col", "mrow", "mcol"):
                        continue
                    em.call("matrix.__setitem__", dict(p, via="setitem"), [mview(src, dt)] + extra + [value],
                            lambda o, index=index: _assign(o, index), f"{src}:{vname}", dt)
            # A full mask from the matrix: a scalar, the exact count, a wrong count, a 2-d value.
            base = mview(src, dt)
            mask = np.matrix(np.asarray(base) != np.asarray(base).flat[0])
            k = int(mask.sum())
            for vname, value in (("scalar", np.array(_vals(3, dt)[2])), ("count", _owner((k,), dt)),
                                 ("short", _owner((max(k - 1, 0),), dt)), ("mrow", np.matrix(_owner((1, k), dt)))):
                if src == "ro" and vname != "scalar":
                    continue
                em.call("matrix.__setitem__", {"index": [{"op": 1}], "tuple": False, "via": "setitem"},
                        [mview(src, dt), mask, value], lambda o: _assign(o, o[1]), f"{src}:mask:{vname}", dt)

    # Cross-dtype stores are unsafe casts.
    for dt, vdt in (("int64", "float64"), ("int8", "int64"), ("float32", "float64"), ("bool", "int64"),
                    ("uint8", "int64"), ("float64", "complex128"), ("int64", "bool")):
        em.call("matrix.__setitem__", {"index": [0], "tuple": False, "via": "setitem"},
                [mown((3, 4), dt), _owner((4,), vdt)], lambda o: _assign(o, 0), f"cast:{vdt}", dt)

    # The typed indexers of the C# class take a matrix or a number.
    for dt in KINDS4:
        for index, vname in ((0, "mrow"), ((S(None), 0), "mcol"), ((S(1, 3), S(1, 3)), "b22m"), (0, "scalar"),
                             ((S(None), 0), "scalar"), ((0, 0), "scalar"), ((S(None), 0), "mrow_bad")):
            value = {"mrow": np.matrix(_owner((1, 4), dt)), "mcol": np.matrix(_owner((3, 1), dt)),
                     "b22m": np.matrix(_owner((2, 2), dt)), "scalar": np.array(_vals(3, dt)[2]),
                     "mrow_bad": np.matrix(_owner((1, 3), dt))}[vname]
            p, extra = _index_params(index)

            def run(o, index=index):
                o[0][index] = o[1][()] if o[1].ndim == 0 else o[1]
                return o[0]
            for via in ("typed", "str"):
                q = dict(p, via=via)
                if via == "str":
                    q["str"] = _index_text(index)
                em.call("matrix.__setitem__", q, [mown((3, 4), dt), value], run, f"{via}:{vname}", dt)


# =============================================================================================================
# Section O: operators
# =============================================================================================================

PY_SCALARS = [("int", 2), ("int", -1), ("int", 0), ("float", 2.5), ("float", -0.5), ("bool", True),
              ("complex", 1 + 2j)]


def _py(kind, v):
    """The params record of a Python scalar operand.

    Parameters:
        kind: "int", "float", "bool" or "complex".
        v: the value.

    Returns:
        ``{"k": kind, "v": value}`` (a complex as ``[re, im]``); replayed as a C# long / double / bool / Complex.
    """
    if kind == "complex":
        return {"k": kind, "v": [v.real, v.imag]}
    return {"k": kind, "v": v}


# (Python method, C# operator symbol, reflected method, callable)
ELEMENTWISE = [
    ("__add__", "+", "__radd__", lambda a, b: a + b), ("__sub__", "-", "__rsub__", lambda a, b: a - b),
    ("__truediv__", "/", "__rtruediv__", lambda a, b: a / b), ("__mod__", "%", "__rmod__", lambda a, b: a % b),
    ("__eq__", "==", "__eq__", lambda a, b: a == b), ("__ne__", "!=", "__ne__", lambda a, b: a != b),
    ("__lt__", "<", "__gt__", lambda a, b: a < b), ("__gt__", ">", "__lt__", lambda a, b: a > b),
    ("__le__", "<=", "__ge__", lambda a, b: a <= b), ("__ge__", ">=", "__le__", lambda a, b: a >= b),
]
BITWISE = [("__and__", "&", "__rand__", lambda a, b: a & b), ("__or__", "|", "__ror__", lambda a, b: a | b),
           ("__xor__", "^", "__rxor__", lambda a, b: a ^ b)]
SHIFTS = [("__lshift__", "<<", None, lambda a, b: a << b), ("__rshift__", ">>", None, lambda a, b: a >> b)]


def gen_mul(em):
    """matrix.__mul__ / __rmul__ / __imul__: the matrix product, the row rule, scaling in np.dot's dtype.

    Parameters:
        em: the tier's Emitter.

    Returns:
        None.
    """
    star = {"sym": "*"}
    # matrix * matrix over every dtype pair: small integer values, so every product is exact everywhere.
    for d1 in ALL_DTYPES:
        for d2 in ALL_DTYPES:
            em.call("matrix.__mul__", star, [mown((2, 3), d1), mown((3, 2), d2)], lambda o: o[0] * o[1],
                    f"mm:{d2}", d1)
    # Layouts of both operands, and shapes that align or do not.
    for dt in KINDS4 + ["float32", "int8"]:
        for s1 in MAIN_SRC:
            for s2 in ("t", "c"):
                a, b = mview(s1, dt), mview(s2, dt)
                em.call("matrix.__mul__", star, [a, b.T if a.shape[1] != b.shape[0] else b], lambda o: o[0] * o[1],
                        f"mm:{s1}*{s2}", dt)
        for (r1, c1), (r2, c2) in (((3, 4), (4, 2)), ((1, 4), (4, 1)), ((4, 1), (1, 4)), ((1, 1), (1, 1)),
                                   ((3, 4), (3, 4)), ((2, 3), (2, 3)), ((0, 3), (3, 2)), ((3, 0), (0, 2)),
                                   ((1, 1), (3, 3)), ((3, 3), (1, 1))):
            em.call("matrix.__mul__", star, [mown((r1, c1), dt), mown((r2, c2), dt)], lambda o: o[0] * o[1],
                    f"mm:{r1}x{c1}*{r2}x{c2}", dt)
        # matrix * ndarray: asmatrix(other) first; ndarray * matrix: the array as it is.
        for label, other in (("v4", _owner((4,), dt)), ("v3", _owner((3,), dt)), ("a43", _owner((4, 3), dt)),
                             ("a34", _owner((3, 4), dt)), ("a41", _owner((4, 1), dt)), ("a14", _owner((1, 4), dt)),
                             ("a143", _owner((1, 4, 3), dt)), ("a222", _owner((2, 2, 2), dt)),
                             ("v1", _owner((1,), dt)), ("a43_f", _owner((4, 3), dt, order="F")),
                             ("a43_t", _owner((3, 4), dt).T)):
            m = lambda: mown((3, 4), dt)
            em.call("matrix.__mul__", star, [m(), other], lambda o: o[0] * o[1], f"m*{label}", dt)
            em.call("matrix.__rmul__", star, [other, m()], lambda o: o[0] * o[1], f"{label}*m", dt)
            em.call("matrix.__mul__", dict(star, via="object"), [m(), other], lambda o: o[0] * o[1],
                    f"m*{label}:object", dt)
            em.call("matrix.__rmul__", dict(star, via="object"), [other, m()], lambda o: o[0] * o[1],
                    f"{label}*m:object", dt)
            em.call("matrix.__mul__", dict(star, via="method"), [m(), other], lambda o: o[0].__mul__(o[1]),
                    f"m*{label}:method", dt)
            em.call("matrix.__rmul__", dict(star, via="method"), [other, m()], lambda o: o[1].__rmul__(o[0]),
                    f"{label}*m:method", dt)

    # Scaling: NumPy scalars of every dtype (strong, np.dot's promotion) and Python scalars.
    for dt in ALL_DTYPES:
        for sdt in ALL_DTYPES:
            s = np.array(_vals(4, sdt)[3])        # 2 (True for bool, 2+0j ...)
            em.call("matrix.__mul__", dict(star, scalar=1), [mown((2, 2), dt), s], lambda o: o[0] * o[1][()],
                    f"*np:{sdt}", dt)
            em.call("matrix.__rmul__", dict(star, scalar=0), [s, mown((2, 2), dt)], lambda o: o[0][()] * o[1],
                    f"np:{sdt}*", dt)
        for src in ("t", "bcast", "rev"):
            s = np.array(_vals(4, dt)[3])
            em.call("matrix.__mul__", dict(star, scalar=1), [mview(src, dt), s], lambda o: o[0] * o[1][()],
                    f"{src}*np", dt)
        for kind, v in PY_SCALARS:
            em.call("matrix.__mul__", dict(star, py=_py(kind, v), side="right"), [mown((2, 3), dt)],
                    lambda o, v=v: o[0] * v, f"*py:{kind}:{v}", dt)
            em.call("matrix.__rmul__", dict(star, py=_py(kind, v), side="left"), [mown((2, 3), dt)],
                    lambda o, v=v: v * o[0], f"py:{kind}:{v}*", dt)

    # An explicit 0-d ARRAY operand (np.array(2)), which is NOT a scalar to matrix.__mul__: isinstance(other,
    # ndarray) is tested first, so m * a0 is N.dot(m, asmatrix(a0)), the product with a 1 x 1 matrix (it raises
    # unless m has one column), and a0 * m is N.dot(a0, m), np.dot's scalar route. params.array0d lists the operand
    # positions the replay marks as explicit 0-d arrays (NDArray.IsExplicitZeroDimArray); a 0-d operand without it
    # is a NumPy scalar, as in the block above. The values are non-zero, so no product depends on the sign of a
    # zero (np.dot's size-1 branch is a known gap of the array library, pinned in OpenBugs.Reductions.cs).
    for dt in ALL_DTYPES:
        for sdt in dict.fromkeys((dt, "bool", "int64", "float64", "complex128")):
            a0 = np.array(_vals(4, sdt)[3])
            for shape in ((2, 3), (3, 1), (1, 3), (1, 1)):
                label = f"{shape[0]}x{shape[1]}:{sdt}"
                em.call("matrix.__mul__", dict(star, array0d=[1]), [mown(shape, dt), a0],
                        lambda o: o[0] * o[1], f"*arr0d:{label}", dt)
                em.call("matrix.__rmul__", dict(star, array0d=[0]), [a0, mown(shape, dt)],
                        lambda o: o[0] * o[1], f"arr0d*:{label}", dt)
        a0 = np.array(_vals(4, dt)[3])
        for src in ("f", "t", "step", "rev", "off", "ro", "bcast"):
            em.call("matrix.__rmul__", dict(star, array0d=[0]), [a0, mview(src, dt)], lambda o: o[0] * o[1],
                    f"arr0d*:{src}", dt)
            em.call("matrix.__mul__", dict(star, array0d=[1]), [mview(src, dt), a0], lambda o: o[0] * o[1],
                    f"*arr0d:{src}", dt)
        for via in ("object", "method"):
            em.call("matrix.__mul__", dict(star, array0d=[1], via=via), [mown((3, 1), dt), a0],
                    (lambda o: o[0] * o[1]) if via == "object" else (lambda o: o[0].__mul__(o[1])),
                    f"*arr0d:{via}", dt)
            em.call("matrix.__rmul__", dict(star, array0d=[0], via=via), [a0, mown((2, 3), dt)],
                    (lambda o: o[0] * o[1]) if via == "object" else (lambda o: o[1].__rmul__(o[0])),
                    f"arr0d*:{via}", dt)
    # The sign of a zero through the left route: N.dot(a0, m) accumulates into zeros, so -0.0 products are +0.0.
    for dt in ("float32", "float64", "complex128"):
        zeros = np.matrix(np.array([[0.0, -0.0, 1.5], [-2.5, 0.0, -0.0]], dtype=dt))
        for factor in (-0.5, 0.5, -0.0, 0.0):
            em.call("matrix.__rmul__", dict(star, array0d=[0]), [np.array(factor, dtype=dt), zeros],
                    lambda o: o[0] * o[1], f"arr0d*:zeros:{factor}", dt)

    # In place: self[:] = self * other.
    def imul(o):
        r = o[0]
        r *= o[1]
        return r

    for dt in KINDS4 + ["int8", "float32"]:
        for label, other in (("m", mown((3, 3), dt)), ("a", _owner((3, 3), dt)), ("bad", mown((3, 2), dt))):
            em.call("matrix.__imul__", {}, [mown((3, 3), dt), other], imul, f"imul:{label}", dt)
        for kind, v in (("int", 2), ("float", 2.5), ("complex", 1 + 2j)):
            def imul_py(o, v=v):
                r = o[0]
                r *= v
                return r
            em.call("matrix.__imul__", {"py": _py(kind, v)}, [mown((2, 3), dt)], imul_py, f"imul:py:{kind}", dt)
        ro = np.matrix(_owner((2, 2), dt, readonly=True), copy=False)

        def imul2(o):
            r = o[0]
            r *= 2
            return r
        em.call("matrix.__imul__", {"py": _py("int", 2)}, [ro], imul2, "imul:readonly", dt)


def gen_elementwise(em):
    """The operators a matrix inherits from ndarray: elementwise, wrapped back into a matrix.

    Parameters:
        em: the tier's Emitter.

    Returns:
        None.
    """
    for dt in ALL_DTYPES:
        m = lambda: mown((3, 4), dt)
        others = {
            "m": mown((3, 4), dt), "row": mown((1, 4), dt), "col": mown((3, 1), dt), "a": _owner((3, 4), dt),
            "v": _owner((4,), dt), "a31": _owner((3, 1), dt), "bad": _owner((2, 4), dt),
            "big": _owner((2, 3, 4), dt), "unit3": _owner((1, 3, 4), dt),
        }
        for name, sym, reflected, fn in ELEMENTWISE:
            for label, other in others.items():
                if dt not in KINDS6 and label not in ("m", "a", "v"):
                    continue
                em.call(f"matrix.{name}", {"sym": sym}, [m(), other], lambda o, fn=fn: fn(o[0], o[1]),
                        f"m{sym}{label}", dt)
                if not isinstance(other, np.matrix):
                    em.call(f"matrix.{reflected}", {"sym": sym}, [other, m()], lambda o, fn=fn: fn(o[0], o[1]),
                            f"{label}{sym}m", dt)
                    if label in ("a", "v"):
                        em.call(f"matrix.{name}", {"sym": sym, "via": "object"}, [m(), other],
                                lambda o, fn=fn: fn(o[0], o[1]), f"m{sym}{label}:object", dt)
                        em.call(f"matrix.{reflected}", {"sym": sym, "via": "object"}, [other, m()],
                                lambda o, fn=fn: fn(o[0], o[1]), f"{label}{sym}m:object", dt)
            for kind, v in (("int", 2), ("float", 2.5), ("bool", True)):
                em.call(f"matrix.{name}", {"sym": sym, "py": _py(kind, v), "side": "right"}, [m()],
                        lambda o, fn=fn, v=v: fn(o[0], v), f"m{sym}py:{kind}", dt)
                em.call(f"matrix.{reflected}", {"sym": sym, "py": _py(kind, v), "side": "left"}, [m()],
                        lambda o, fn=fn, v=v: fn(v, o[0]), f"py:{kind}{sym}m", dt)
        if dt in INTS or dt == "bool":
            for name, sym, reflected, fn in BITWISE:
                for label in ("m", "a", "v", "row"):
                    other = others[label]
                    em.call(f"matrix.{name}", {"sym": sym}, [m(), other], lambda o, fn=fn: fn(o[0], o[1]),
                            f"m{sym}{label}", dt)
                    if not isinstance(other, np.matrix):
                        em.call(f"matrix.{reflected}", {"sym": sym}, [other, m()], lambda o, fn=fn: fn(o[0], o[1]),
                                f"{label}{sym}m", dt)
        if dt in INTS:
            for name, sym, _, fn in SHIFTS:
                for label in ("m", "a", "v"):
                    other = (np.abs(np.asarray(others[label]).astype("int64")) % 3).astype(dt)
                    if label == "m":
                        other = np.matrix(other)
                    em.call(f"matrix.{name}", {"sym": sym}, [m(), other], lambda o, fn=fn: fn(o[0], o[1]),
                            f"m{sym}{label}", dt)
        for name, sym, fn in (("__neg__", "-", lambda a: -a), ("__pos__", "+", lambda a: +a),
                              ("__invert__", "~", lambda a: ~a)):
            for src in ("c", "t", "rev"):
                em.call(f"matrix.{name}", {"sym": sym}, [mview(src, dt)], lambda o, fn=fn: fn(o[0]), f"{sym}:{src}", dt)
    # Layouts of the matrix operand.
    for dt in KINDS4:
        for src in MAIN_SRC + ["row", "col", "one", "e03"]:
            m = mview(src, dt)
            em.call("matrix.__add__", {"sym": "+"}, [m, m], lambda o: o[0] + o[1], f"add:{src}:self", dt)
            em.call("matrix.__gt__", {"sym": ">", "py": _py("int", 0), "side": "right"}, [mview(src, dt)],
                    lambda o: o[0] > 0, f"gt:{src}", dt)


def gen_pow(em):
    """matrix.__pow__ / __ipow__ / __rpow__: integer powers; negative powers only where the inverse is exact.

    Parameters:
        em: the tier's Emitter.

    Returns:
        None.
    """
    for dt in ALL_DTYPES:
        for n in (0, 1, 2, 3, 4, 5, 7):
            em.call("matrix.__pow__", {"n": n}, [mown((3, 3), dt)], lambda o, n=n: o[0] ** n, f"pow:{n}", dt)
        em.call("matrix.__pow__", {"n": 2}, [mview("c", dt)], lambda o: o[0] ** 2, "pow:nonsquare", dt)
        em.call("matrix.__pow__", {"n": 2}, [mown((1, 1), dt)], lambda o: o[0] ** 2, "pow:1x1", dt)
        em.call("matrix.__pow__", {"n": 0}, [mown((0, 0), dt)], lambda o: o[0] ** 0, "pow:empty", dt)
    for dt in KINDS4 + ["float32"]:
        sq = np.matrix(SOURCES["sq"](dt), copy=False)
        for label, m in (("f", np.matrix(_owner((3, 3), dt, order="F"), copy=False)), ("t", sq.T),
                         ("rev", sq[::-1, ::-1])):
            em.call("matrix.__pow__", {"n": 3}, [m], lambda o: o[0] ** 3, f"pow:3:{label}", dt)
    perm = np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])
    diag = np.diag([2.0, 4.0, 0.5])
    for dt in ("float64", "float32", "int64"):
        for label, a in (("perm", perm), ("diag", diag)):
            if dt == "int64" and label == "diag":
                continue
            for n in (-1, -2, -3):
                em.call("matrix.__pow__", {"n": n}, [np.matrix(a.astype(dt))], lambda o, n=n: o[0] ** n,
                        f"pow:{n}:{label}", dt)
    # Exponents that are not Python ints.
    for label, e, p in (("float", 2.0, {"exp": {"k": "float", "v": 2.0}}),
                        ("bool", True, {"exp": {"k": "bool", "v": True}}),
                        ("list", [2], {"exp": {"k": "list", "v": [2]}}),
                        ("np_int", np.int64(2), {"exp_op": 1}), ("np_float", np.float64(2), {"exp_op": 1}),
                        ("matrix", np.matrix([[2]]), {"exp_op": 1}), ("array0d", np.array(3), {"exp_op": 1}),
                        ("array1d", np.array([2]), {"exp_op": 1})):
        ops = [mown((2, 2), "int64")]
        if "exp_op" in p:
            ops.append(e if isinstance(e, np.matrix) else np.asarray(e))
        em.call("matrix.__pow__", p, ops, lambda o, e=e: o[0] ** e, f"pow:exp:{label}", "int64")
    em.call("matrix.__pow__", {"exp": {"k": "float", "v": 2.0}}, [mview("c", "int64")], lambda o: o[0] ** 2.0,
            "pow:nonsquare:float", "int64")

    def ipow(n):
        def run(o):
            r = o[0]
            r **= n
            return r
        return run

    for dt in KINDS4:
        em.call("matrix.__ipow__", {"n": 2}, [mown((3, 3), dt)], ipow(2), "ipow:2", dt)
        em.call("matrix.__ipow__", {"n": 1}, [mown((3, 3), dt)], ipow(1), "ipow:1", dt)
    for kind, v in (("int", 2), ("float", 2.5), ("bool", True), ("complex", 1j)):
        em.call("matrix.__rpow__", {"py": _py(kind, v)}, [mown((2, 2), "int64")], lambda o, v=v: v ** o[0],
                f"rpow:py:{kind}", "int64")
    # An ARRAY base is ndarray's own elementwise power, wrapped as a matrix. Exponents 0..3 keep every dtype exact.
    for dt in ("int64", "float64", "int8", "uint64", "float32", "complex128", "bool"):
        base = np.full((2, 2), 2).astype(dt)
        exponent = np.matrix(np.arange(4).reshape(2, 2).astype(dt))
        em.call("matrix.__rpow__", {}, [base, exponent], lambda o: o[0] ** o[1], "rpow:ndarray", dt)


# =============================================================================================================
# Section R: reductions
# =============================================================================================================

REDUCTIONS = ["sum", "prod", "mean", "std", "var", "any", "all", "max", "min", "argmax", "argmin", "ptp"]


def gen_reductions(em):
    """sum / prod / mean / std / var / any / all / max / min / argmax / argmin / ptp: scalar, row, column.

    Parameters:
        em: the tier's Emitter.

    Returns:
        None.
    """
    for dt in ALL_DTYPES:
        for src in ALL_SRC:
            if dt not in KINDS6 and src not in ("c", "t"):
                continue
            axes = [None, 0, 1, -1, -2, 2, -3] if src == "c" else [None, 0, 1, -1]
            for name in REDUCTIONS:
                for axis in axes:
                    em.call(f"matrix.{name}", {"axis": axis}, [mview(src, dt)],
                            lambda o, name=name, axis=axis: getattr(o[0], name)(axis), f"{src}:{axis}", dt)
                    # The (int? axis) overload of the C# class, for an axis that is given.
                    if src == "c" and axis in (0, 1) and name not in ("argmax", "argmin"):
                        em.call(f"matrix.{name}", {"axis": axis, "nullable": True}, [mview(src, dt)],
                                lambda o, name=name, axis=axis: getattr(o[0], name)(axis), f"{src}:{axis}:nullable", dt)
    # dtype= and ddof=.
    for dt in ALL_DTYPES:
        for name in ("sum", "prod", "mean", "std", "var"):
            for to in ("float64", "float32", "int64", "int8", "complex128", "bool"):
                if dt not in KINDS6 and to not in ("float64", "int64"):
                    continue
                for axis in (None, 1):
                    em.call(f"matrix.{name}", {"axis": axis, "dtype": to}, [mown((3, 4), dt)],
                            lambda o, name=name, axis=axis, to=to: getattr(o[0], name)(axis, dtype=to),
                            f"dtype:{to}:{axis}", dt)
        for name in ("std", "var"):
            for ddof in (1, 4, 12):
                for axis in (None, 0, 1):
                    em.call(f"matrix.{name}", {"axis": axis, "ddof": ddof}, [mown((3, 4), dt)],
                            lambda o, name=name, axis=axis, ddof=ddof: getattr(o[0], name)(axis, ddof=ddof),
                            f"ddof:{ddof}:{axis}", dt)
    # out=: a matrix of the kept shape. Without dtype= the reduce loop is promote_types(out.dtype, operand dtype),
    # and var / std hand out to their last sum only.
    for dt in KINDS6:
        shapes = {None: (1, 1), 0: (1, 4), 1: (3, 1), -1: (3, 1)}
        for name in ("sum", "prod", "mean", "max", "min", "var", "std"):
            for axis, shape in shapes.items():
                for odt in ("float64", "int64", "float32", "int8", "bool", "complex128", "float16"):
                    out = np.matrix(np.zeros(shape, dtype=odt))
                    em.call(f"matrix.{name}", {"axis": axis, "out": 1}, [mown((3, 4), dt), out],
                            lambda o, name=name, axis=axis: getattr(o[0], name)(axis, out=o[1]),
                            f"out:{odt}:{axis}", dt)
            # out of a wrong shape, and a read-only out.
            for label, shape in (("wrong", (4, 1)), ("full", (3, 4)), ("short", (1, 3)), ("readonly", (1, 4))):
                out = np.matrix(np.zeros(shape))
                if label == "readonly":
                    out.setflags(write=False)
                em.call(f"matrix.{name}", {"axis": 0, "out": 1}, [mown((3, 4), dt), out],
                        lambda o, name=name: getattr(o[0], name)(0, out=o[1]), f"out:{label}", dt)
        # The out NumPy's own code cannot use.
        for name in ("any", "all"):
            out = np.matrix(np.zeros((1, 4), dtype=bool))
            em.call(f"matrix.{name}", {"axis": 0, "out": 1}, [mown((3, 4), dt), out],
                    lambda o, name=name: getattr(o[0], name)(0, out=o[1]), "out:bool", dt)
        for name in ("argmax", "argmin"):
            for label, out in (("matrix", np.matrix(np.zeros((1, 4), dtype=np.int64))),
                               ("array", np.zeros((4,), dtype=np.int64)), ("short", np.zeros((3,), dtype=np.int64))):
                em.call(f"matrix.{name}", {"axis": 0, "out": 1}, [mown((3, 4), dt), out],
                        lambda o, name=name: getattr(o[0], name)(0, out=o[1]), f"out:{label}", dt)
        if dt != "bool":
            for label, out, axis in (("array", np.zeros((4,), dtype=dt), 0),
                                     ("matrix", np.matrix(np.zeros((1, 4), dtype=dt)), 0),
                                     ("scalar", np.zeros((), dtype=dt), None)):
                em.call("matrix.ptp", {"axis": axis, "out": 1}, [mown((3, 4), dt), out],
                        lambda o, axis=axis: o[0].ptp(axis, out=o[1]), f"out:{label}", dt)


# =============================================================================================================
# Section M: the methods ndarray hands down
# =============================================================================================================

def gen_methods(em):
    """squeeze / flatten / ravel and every inherited method that returns an array, as matrices.

    Parameters:
        em: the tier's Emitter.

    Returns:
        None.
    """
    for dt in ALL_DTYPES:
        for src in ALL_SRC:
            if dt not in KINDS4 and src not in ("c", "t"):
                continue
            full = dt in KINDS4
            m = lambda: mview(src, dt)
            one = lambda name, p, fn: em.call(f"matrix.{name}", p, [m()], fn, src, dt)
            one("squeeze", {}, lambda o: o[0].squeeze())
            for axis in (0, 1, -1):
                one("squeeze", {"axis": axis}, lambda o, axis=axis: o[0].squeeze(axis))
            for order in ("C", "F", "A", "K") if full else ("C", "F"):
                one("flatten", {"order": order}, lambda o, order=order: o[0].flatten(order))
                one("ravel", {"order": order}, lambda o, order=order: o[0].ravel(order))
            one("ravel", {}, lambda o: o[0].ravel())
            one("copy", {}, lambda o: o[0].copy())
            for order in ("F", "K", "A") if full else ("K",):
                one("copy", {"order": order}, lambda o, order=order: o[0].copy(order))
            one("transpose", {}, lambda o: o[0].transpose())
            one("transpose", {"axes": [1, 0]}, lambda o: o[0].transpose(1, 0))
            one("transpose", {"axes": [0, 1]}, lambda o: o[0].transpose(0, 1))
            one("swapaxes", {"a": 0, "b": 1}, lambda o: o[0].swapaxes(0, 1))
            one("mT", {}, lambda o: o[0].mT)
            one("view", {}, lambda o: o[0].view())
            one("diagonal", {}, lambda o: o[0].diagonal())
            for off in (1, -1):
                one("diagonal", {"offset": off}, lambda o, off=off: o[0].diagonal(off))
            one("trace", {}, lambda o: o[0].trace())
            one("conj", {}, lambda o: o[0].conj())
            one("conjugate", {}, lambda o: o[0].conjugate())
            one("real", {}, lambda o: o[0].real)
            one("imag", {}, lambda o: o[0].imag)
            one("astype", {"dtype": dt, "copy": False}, lambda o: o[0].astype(o[0].dtype, copy=False))
            for to in ("float64", "int32", "bool", "complex128") if full else ("float64",):
                one("astype", {"dtype": to}, lambda o, to=to: o[0].astype(to))
            one("cumsum", {}, lambda o: o[0].cumsum())
            one("cumprod", {}, lambda o: o[0].cumprod())
            for axis in (0, 1):
                one("cumsum", {"axis": axis}, lambda o, axis=axis: o[0].cumsum(axis))
                one("cumprod", {"axis": axis}, lambda o, axis=axis: o[0].cumprod(axis))
                # kind="stable": NumPy's default quicksort orders equal elements by its SIMD kernel's whim, while
                # NumSharp's argsort is always stable. The matrix contract is the class and shape of the result.
                one("argsort", {"axis": axis}, lambda o, axis=axis: o[0].argsort(axis, kind="stable"))
                one("repeat", {"repeats": 2, "axis": axis}, lambda o, axis=axis: o[0].repeat(2, axis))
            one("argsort", {}, lambda o: o[0].argsort(kind="stable"))
            one("argsort", {"axis": None}, lambda o: o[0].argsort(None, kind="stable"))
            one("repeat", {"repeats": 2}, lambda o: o[0].repeat(2))
            one("round", {}, lambda o: o[0].round())
            for dec in (1, -1, 0):
                one("round", {"decimals": dec}, lambda o, dec=dec: o[0].round(dec))
            size = m().size
            shapes = ((size,), (-1,), (1, size), (size, 1), (1, 1, size), (size, 1, 1), (2, -1), (1, 2, -1),
                      (2, 2, -1), (5,))
            for shape in shapes if full else shapes[:3]:
                one("reshape", {"shape": list(shape)}, lambda o, shape=shape: o[0].reshape(shape))
            if size:
                one("take", {"index": 1 % size}, lambda o: o[0].take(1 % o[0].size))
                em.call("matrix.take", {"indices": 1}, [m(), np.array([0, size - 1, 0])], lambda o: o[0].take(o[1]),
                        src, dt)
                for axis in (0, 1):
                    k = m().shape[axis]
                    em.call("matrix.take", {"indices": 1, "axis": axis}, [m(), np.array([0, k - 1])],
                            lambda o, axis=axis: o[0].take(o[1], axis), src, dt)
                    cond = np.array([i % 2 == 0 for i in range(k)])
                    em.call("matrix.compress", {"cond": 1, "axis": axis}, [m(), cond],
                            lambda o, axis=axis: o[0].compress(o[1], axis), src, dt)
                cond = np.array([i % 2 == 0 for i in range(size)])
                em.call("matrix.compress", {"cond": 1}, [m(), cond], lambda o: o[0].compress(o[1]), src, dt)
                em.call("matrix.take", {"indices": 1, "mode": "wrap"}, [m(), np.array([size + 1, -1])],
                        lambda o: o[0].take(o[1], mode="wrap"), src, dt)
                em.call("matrix.take", {"indices": 1}, [m(), np.array([size])], lambda o: o[0].take(o[1]), src, dt)
            if dt not in ("bool", "complex128"):
                lo, hi = np.array(_vals(4, dt)[1]), np.array(_vals(4, dt)[3])
                em.call("matrix.clip", {"lo": 1, "hi": 2}, [m(), lo, hi], lambda o: o[0].clip(o[1][()], o[2][()]),
                        src, dt)
            # dot through the array method: the right operand as it is; a 1-d product is folded to a row.
            c = m().shape[1]
            for label, other in (("v", _owner((c,), dt)), ("a", _owner((c, 2), dt)), ("m", mown((c, 2), dt)),
                                 ("bad", _owner((c + 1,), dt))):
                em.call("matrix.dot", {}, [m(), other], lambda o: o[0].dot(o[1]), f"{src}:dot:{label}", dt)

    # In-place shape assignment folds the new shape.
    def set_shape(shape):
        def run(o):
            o[0].shape = shape
            return o[0]
        return run

    for dt in KINDS4:
        for label, base, shape in (("c", lambda: mown((3, 4), dt), (12,)), ("c", lambda: mown((3, 4), dt), (2, 6)),
                                   ("c", lambda: mown((3, 4), dt), (1, 3, 4)), ("c", lambda: mown((3, 4), dt), (2, 2, 3)),
                                   ("c", lambda: mown((3, 4), dt), (5,)), ("c", lambda: mown((3, 4), dt), (-1,)),
                                   ("t", lambda: mview("t", dt), (12,)), ("row", lambda: mown((1, 4), dt), (4, 1)),
                                   ("c", lambda: mown((3, 4), dt), (12, 1, 1))):
            em.call("matrix.shape", {"shape": list(shape)}, [base()], set_shape(shape), f"shape:{label}:{shape}", dt)
    # view(dtype) reinterprets the bytes; byteswap.
    for dt, to in (("int64", "int32"), ("int32", "int64"), ("float64", "int64"), ("int8", "uint8"), ("int16", "int8"),
                   ("complex128", "float64"), ("float32", "int32"), ("int64", "int8")):
        em.call("matrix.view", {"dtype": to}, [mown((2, 4), dt)], lambda o, to=to: o[0].view(to), f"view:{to}", dt)
        em.call("matrix.view", {"dtype": to}, [mview("t", dt)], lambda o, to=to: o[0].view(to), f"view:t:{to}", dt)
    for dt in ("int16", "int64", "float32", "complex128", "bool"):
        em.call("matrix.byteswap", {}, [mown((2, 3), dt)], lambda o: o[0].byteswap(), "byteswap", dt)
        em.call("matrix.byteswap", {"inplace": True}, [mown((2, 3), dt)], lambda o: o[0].byteswap(True),
                "byteswap:inplace", dt)


# =============================================================================================================
# Section X: text
# =============================================================================================================

def gen_text(em):
    """repr prints the class name ``matrix`` and indents one column further than an array; str is the array's.

    Parameters:
        em: the tier's Emitter.

    Returns:
        None.
    """
    for dt in ALL_DTYPES:
        for src in ("c", "t", "row", "col", "one", "e03", "e30"):
            m = mview(src, dt)
            em.call("matrix.__repr__", {}, [m], lambda o: repr(o[0]), src, dt, kind="text")
            em.call("matrix.__str__", {}, [m], lambda o: str(o[0]), src, dt, kind="text")
        big = np.matrix(_vals(1200, dt).reshape(30, 40))
        em.call("matrix.__repr__", {}, [big], lambda o: repr(o[0]), "big", dt, kind="text")
        wide = _vals(40, dt).reshape(2, 20)
        if dt in FLOATS:
            wide = (wide.astype("float64") * 1234.5678).astype(dt)
        em.call("matrix.__repr__", {}, [np.matrix(wide)], lambda o: repr(o[0]), "wide", dt, kind="text")


# =============================================================================================================
# Carving
# =============================================================================================================

# gap id -> the documented divergence (each one has a pin in test/NumSharp.Tests/OpenBugs.Matrix.cs).
#
# Every gap here belongs to the ARRAY library below the matrix class: NDMatrix delegates the computation to an
# NDArray member or an np.* function, and that function differs from NumPy for the operand layout or dtype of
# the case. The matrix tier found them because it runs each method over layouts and dtypes the older tiers did
# not combine. The class itself (the finalize rule, the row / column rule, the product, the result class and
# ownership) is held to NumPy without exception.
GAPS = {
    "MX01": "a[:, index_array]: NumPy returns its transposed MapIter view (F-contiguous for two axes); NumSharp's "
            "fancy getter returns a C-ordered array. Values equal. Library-wide (OpenBugs.FancyIndex.cs).",
    "MX02": "np.cumsum / np.cumprod result layout for an operand that is not C-contiguous: NumPy's accumulate "
            "allocates in the operand's order (F for an F or transposed operand), NumSharp in C order; a "
            "broadcast bool operand goes the other way. Values equal.",
    "MX03": "ndarray.imag of a real F-contiguous array: NumPy allocates the zeros in Fortran order "
            "(PyArray_ISFORTRAN(self)), NumSharp in C order.",
    "MX04": "ndarray.round of a broadcast (stride-0) operand: the result layout differs (NumPy's KEEPORDER copy "
            "/ PyArray_Empty(ISFORTRAN) against NumSharp's). Values equal.",
    "MX05": "np.conjugate of a transposed strided or reversed complex128 view: NumPy's ufunc keeps the operand's "
            "K order, NumSharp writes C order. Reached through matrix.H. Values equal.",
    "MX06": "ndarray.reshape with a -1 dimension on a broadcast (stride-0) view throws NotSupportedException; "
            "NumPy copies and reshapes.",
    "MX07": "fancy assignment with two index entries into a C-contiguous view at a non-zero offset "
            "(a[rows, cols] = v, a[:, cols] = v) throws IndexOutOfRangeException: the offsets include the view's "
            "offset and the bound does not. The offset-view class of the array-object oracle.",
    "MX08": "np.prod in a cast loop: the flat form with dtype=bool throws NotSupportedException, with a "
            "complex128 or float16 loop over a real operand it returns NaN or garbage, and complex128 -> bool "
            "along an axis is wrong. Reached through matrix.prod(dtype=...) and matrix.prod(out=...).",
    "MX09": "complex128 np.prod over a reversed view: the sign of a zero part differs (NumPy's npy_cmul visiting "
            "order). The complex Prod gap of the reductions tier.",
    "MX10": "np.dot of an N-D operand with an M-D one (the generic DotNDMD route) computes every dtype in "
            "float64: a complex128 product loses its imaginary parts. Reached through ndarray(1,4,3) * matrix.",
    "MX11": "np.repeat of a C-contiguous view at a non-zero offset reads from the start of the parent buffer "
            "(the offset-view class of the array-object oracle).",
}


def _case_dtype(case):
    """The dtype label of a case.

    Parameters:
        case: one generated case.

    Returns:
        The third-from-last component of the case id (``op/layout/dtype/n``).
    """
    return case["id"].split("/")[-2]


def _is_advanced(entry):
    """Whether one serialized index entry is an index array or list.

    Parameters:
        entry: an item of ``params["index"]``.

    Returns:
        True for ``{"l": [...]}`` (a list) and ``{"op": k}`` (an operand array).
    """
    return isinstance(entry, dict) and ("l" in entry or "op" in entry)


def _gap_of(case):
    """The documented gap a case pins.

    Parameters:
        case: one generated case.

    Returns:
        A key of GAPS, or None when NumSharp is held to NumPy's answer for the case.
    """
    op, layout, dt, p = case["op"], case["layout"], _case_dtype(case), case.get("params") or {}
    raises = bool(case.get("expects_throw"))
    flags = (case.get("expected") or {}).get("mem", {}).get("flags")

    if op == "matrix.__getitem__" and not raises and flags and not flags[0] and flags[1]:
        index = p.get("index") or []
        if len(index) == 2 and any(_is_advanced(e) for e in index) and not all(_is_advanced(e) for e in index):
            return "MX01"

    if not raises and p.get("axis") in (0, 1):
        if op == "matrix.cumprod" and layout in ("f", "t"):
            return "MX02"
        if op == "matrix.cumsum" and layout == "bcast" and dt == "bool":
            return "MX02"

    if op == "matrix.imag" and layout in ("f", "t") and dt != "complex128" and not raises:
        return "MX03"

    if op == "matrix.round" and layout == "bcast" and not raises:
        decimals = p.get("decimals", 0)
        if (dt == "complex128" and decimals == 0) or (dt == "float64" and decimals in (-1, 1)) \
                or (dt == "int64" and decimals == -1):
            return "MX04"

    if op in ("matrix.H", "matrix.getH") and layout in ("rev", "step") and dt == "complex128":
        return "MX05"

    if op == "matrix.reshape" and layout == "bcast" and -1 in (p.get("shape") or []):
        return "MX06"

    if op == "matrix.__setitem__" and layout.startswith("off:") and not raises:
        # Only the assignments NumPy accepts reach the scatter that computes the offsets; a value NumPy refuses is
        # refused by NumSharp first, with the same text. An integer FIRST entry takes another route that is right.
        index = p.get("index") or []
        if len(index) == 2 and any(_is_advanced(e) for e in index) and not isinstance(index[0], int):
            return "MX07"

    if op == "matrix.prod":
        axis, loop = p.get("axis"), p.get("dtype")
        if "out" in p and layout.startswith("out:"):
            # The loop of a reduction with out= is promote_types(out, operand); the cases below name the pairs
            # where that promotion is the out dtype itself.
            loop = layout.split(":")[1]
        if axis is None and loop == "bool" and ("out" not in p or dt == "bool"):
            return "MX08"
        if axis is None and loop == "complex128" and dt in ("float32", "int64", "int8"):
            return "MX08"
        if axis is None and loop == "float16" and dt == "int8":
            return "MX08"
        if axis == 1 and loop == "bool" and "out" not in p and dt == "complex128":
            return "MX08"
        if layout.startswith("rev:") and dt == "complex128":
            return "MX09"

    if op == "matrix.__rmul__" and layout.startswith("a143*m") and dt == "complex128":
        return "MX10"

    if op == "matrix.repeat" and layout == "off":
        return "MX11"

    return None


SECTIONS = [gen_construct, gen_properties, gen_getitem, gen_setitem, gen_mul, gen_elementwise, gen_pow,
            gen_reductions, gen_methods, gen_text]


def gen(carve=True):
    """Generate the tier.

    Parameters:
        carve: split the documented gaps out of the gate file.

    Returns:
        (gate cases, carved gap cases).
    """
    em = Emitter()
    for section in SECTIONS:
        before = len(em.cases)
        section(em)
        print(f"  {section.__name__}: {len(em.cases) - before} cases")
    if em.skipped:
        print(f"  skipped (not serializable / address in an error text): {em.skipped}")
    if not carve:
        return em.cases, []
    gate, gaps = [], []
    for c in em.cases:
        g = _gap_of(c)
        if g is None:
            gate.append(c)
        else:
            gaps.append(dict(c, gap=g))
    return gate, gaps


def main():
    """Generate matrix.jsonl and gaps/matrix_gaps.jsonl; see the module docstring for the flags.

    Returns:
        None.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--uncarved", metavar="DIR", help="write the whole matrix (no carve) into DIR instead")
    args = ap.parse_args()
    if args.uncarved:
        cases, _ = gen(carve=False)
        write_jsonl(os.path.join(args.uncarved, "matrix.jsonl"), cases)
        return
    cases, gaps = gen(carve=True)
    write_jsonl(os.path.join(CORPUS, "matrix.jsonl"), cases)
    write_jsonl(os.path.join(GAP_DIR, "matrix_gaps.jsonl"), gaps)


if __name__ == "__main__":
    main()
