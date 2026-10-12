#!/usr/bin/env python3
"""NumPy 2.4.2 oracle for NumSharp's string dtypes: the fixed-width 'S' (bytes) and 'U' (str), and NEP 55 StringDType ('T').

Writes ``test/NumSharp.Tests.Oracle/Strings/corpus/strings.jsonl``: one case per line, replayed byte for byte by
``StringOracleTests`` (no Python at test time).

Each case records what NumPy 2.4.2 returns for one call:

* ``operands`` -- the arrays the call reads, serialized EXACTLY: dtype ``str`` ('<U3', '|S5', '|b1', '<i8'), shape,
  element strides, element offset and the whole base buffer, so a reversed / strided / transposed / broadcast / 0-d /
  empty view is rebuilt as that view (not as a copy).
* ``params`` -- the call's other arguments. Text never travels as a JSON string: a ``str`` is ``{"u": <UTF-32LE hex>}``
  (lone surrogates included, encoded with ``surrogatepass``) and a ``bytes`` is ``{"b": <hex>}``.
* ``expected`` -- one of: ``array`` (dtype str, shape, C-order bytes, element strides, C/F/writeable/owndata flags and
  whether the result shares memory with each operand), ``value`` (a Python value: str / bytes / nested list), ``text``
  (a printed string), ``bool``, ``dtype`` (a descriptor's str and repr), ``props`` (descriptor attributes),
  ``tuple`` (a list of array expectations) or ``mutated`` (an in-place call: the target view and its whole base buffer
  afterwards). A numeric NumPy SCALAR result is an ``array`` with ``"scalar": true`` and no writeable / owndata flags
  (a scalar is immutable and owns nothing; NumSharp answers with a read-only 0-d array).
* ``error`` -- when NumPy raises: the exception's first well-known class on its MRO and ``str(e)`` verbatim.
* NEP 55 ``StringDType`` (``T``, sections tstore / tcreate / tmove / tcast) travels by VALUE, never by cells (a cell holds
  an arena offset or a heap pointer of NumPy's allocator): an operand has ``"t": {"coerce", "na"}`` (the descriptor
  request) and ``"values"`` (the base's elements, ``{"na": true}`` for an NA cell, told apart by the cell's MISSING flag)
  instead of ``"buffer"``; an array expectation has ``"values"`` instead of ``"buffer"``; an in-place call's base is
  ``"basevalues"``; a ``tobytes()`` result is ``{"kind": "tcells"}`` with the heap pointers masked (``_mask_heap``).
* ``phase`` -- 1 for the calls NumSharp's string support covers now (storage, element access, conversion, copying,
  string-to-string casts, text <-> number conversion, indexing, printing, and every refusal NumPy itself raises); 2 for
  the calls NumPy answers by COMPUTING on text (sorting, searching and the rest of phase 2). A phase-2 case passes while
  NumSharp throws ``StringComputationNotSupportedException`` (counted as deferred) and is compared strictly as soon as
  NumSharp computes it.

Deterministic: no randomness; regenerate with ``python test/oracle/gen_strings_oracle.py`` (needs numpy==2.4.2).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

assert np.__version__ == "2.4.2", f"the string oracle is pinned to numpy 2.4.2, found {np.__version__}"

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "NumSharp.Tests.Oracle" / "Strings" / "corpus" / "strings.jsonl"

CASES = []
_IDS = set()


# ----------------------------------------------------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------------------------------------------------

def enc(v):
    """A Python value as JSON: text as hex (so lone surrogates and NULs survive), containers element by element."""
    if isinstance(v, (str, np.str_)):
        return {"u": str(v).encode("utf-32-le", "surrogatepass").hex()}
    if isinstance(v, (bytes, np.bytes_)):
        return {"b": bytes(v).hex()}
    if v is None:
        return {"none": True}
    if isinstance(v, (bool, np.bool_)):
        return {"bool": bool(v)}
    if isinstance(v, (int, np.integer)):
        return {"int": int(v)}
    if isinstance(v, (float, np.floating)):
        # A non-finite float travels as "nan" / "inf" / "-inf" (the corpus is written with allow_nan=False); a StringDType
        # NA object that is a float NaN reaches here through tolist() / item().
        return {"float": _jf(float(v))}
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
    raise TypeError(f"cannot encode {type(v)}")


# LinAlgError (a ValueError subclass in NumPy and NumSharp alike) is recorded under its own name, so the replay checks
# that NumSharp raises the linear-algebra class itself, not just any ValueError.
KNOWN_ERRORS = ["UnicodeEncodeError", "UnicodeDecodeError", "DTypePromotionError", "AxisError", "LinAlgError",
                "ValueError", "TypeError", "IndexError", "OverflowError", "SystemError", "KeyError", "ZeroDivisionError",
                "NotImplementedError", "RuntimeError", "MemoryError"]


def err_info(e):
    """The exception's first well-known class on its MRO (UFuncTypeError -> TypeError) and its text verbatim."""
    for cls in type(e).__mro__:
        if cls.__name__ in KNOWN_ERRORS:
            return {"type": cls.__name__, "text": str(e)}
    return {"type": type(e).__name__, "text": str(e)}


def describe(base, view):
    """A (base, view) pair as an operand: the view is rebuilt over the whole base buffer, in elements."""
    assert base.ndim == 1 and base.flags.c_contiguous, "a base must be a 1-D C-contiguous buffer"
    if base.dtype.kind == "T":
        return _describe_t(base, view)
    isz = base.itemsize
    off_bytes = view.__array_interface__["data"][0] - base.__array_interface__["data"][0] if view.size else 0
    assert off_bytes % isz == 0
    strides = []
    for s in view.strides:
        assert s % isz == 0, "a view stride must be a whole number of elements"
        strides.append(s // isz)
    d = {
        "dtype": view.dtype.str,
        "shape": [int(x) for x in view.shape],
        "strides": strides,
        "offset": int(off_bytes // isz),
        "bufferSize": int(base.size),
        "buffer": base.tobytes().hex(),
    }
    if not view.flags.writeable:
        d["writeable"] = False
    return d


def plain(arr):
    """An operand that is its own C-contiguous base (a fresh copy, reshaped as a view of a 1-D buffer)."""
    arr = np.asarray(arr)
    base = np.ascontiguousarray(arr).reshape(-1).copy()
    return base, base.reshape(arr.shape)


def own(arr):
    """An operand that owns its data where the replay can tell: a 1-D array is its own base. ``plain`` makes a reshaped
    VIEW of the base (OWNDATA False), and a 1-D view with the base's own shape, strides and offset is rebuilt as the base
    itself — so an in-place target or an out= array, whose flags are compared, is given as the base."""
    arr = np.asarray(arr)
    base = np.ascontiguousarray(arr).reshape(-1).copy()
    return base, (base if arr.ndim == 1 else base.reshape(arr.shape))


def target(arr):
    """An in-place target or out= operand whose OWNDATA the replay reproduces: a 1-D string array, or an empty 1-D numeric
    one, is its own base (``own``: the replay rebuilds those as the base); a non-empty numeric operand stays a view
    (``plain``: the general corpus' reconstruction builds a view over the buffer)."""
    arr = np.asarray(arr)
    if arr.ndim == 1 and (arr.dtype.kind in "SU" or arr.size == 0):
        return own(arr)
    return plain(arr)


def array_expect(r, operands):
    """An array result: dtype, shape, C-order bytes, element strides, flags and memory sharing with each operand."""
    r = np.asarray(r)
    if r.dtype.kind == "T":
        return _t_array_expect(r, operands)
    d = {
        "kind": "array",
        "dtype": r.dtype.str,
        "shape": [int(x) for x in r.shape],
        "buffer": np.ascontiguousarray(r).tobytes().hex(),
        "strides": [int(s // r.itemsize) for s in r.strides] if r.itemsize else [],
        "c": bool(r.flags.c_contiguous),
        "f": bool(r.flags.f_contiguous),
        "writeable": bool(r.flags.writeable),
        "owndata": bool(r.flags.owndata),
    }
    if operands:
        d["shares"] = [bool(np.shares_memory(r, o)) if isinstance(o, np.ndarray) else False for o in operands]
    return d


def _enc_fill(v):
    """A masked array's fill value as its getter returns it: text, bytes, a number (complex as a pair), or an array (the
    deprecated non-scalar fill)."""
    if isinstance(v, np.ndarray):
        return {"arr": array_expect(v, None)}
    if isinstance(v, np.generic) and not isinstance(v, (np.str_, np.bytes_)):
        v = v.item()
    if isinstance(v, complex):
        return {"complex": [v.real, v.imag]}
    return enc(v)


def masked_expect(r, uninit=False):
    """A masked result: its data (descriptor, shape and C-order bytes, masked slots included), its mask (``None`` for
    nomask, else the boolean bytes), the fill value its getter returns and the hard-mask flag. ``uninit`` marks data
    NumPy left UNINITIALIZED (``np.empty_like`` / ``masked_all`` / ``masked_all_like``): an ``S`` buffer then holds
    whatever the allocator recycled, so the buffer is recorded as ``None`` (not compared). A ``U`` buffer is still
    recorded: its dtype carries ``NPY_NEEDS_INIT``, so NumPy's ``np.empty`` zero-fills it."""
    data = np.asarray(r.data)
    m = r._mask
    buffer = None if uninit and data.dtype.kind == "S" else np.ascontiguousarray(data).tobytes().hex()
    return {"kind": "masked", "dtype": data.dtype.str, "shape": [int(x) for x in data.shape],
            "buffer": buffer,
            "mask": None if m is np.ma.nomask else np.ascontiguousarray(np.asarray(m, dtype=bool)).tobytes().hex(),
            "fill": _enc_fill(r.fill_value), "hard": bool(r._hardmask)}


class _UninitData:
    """A masked result whose data NumPy left uninitialized (see ``masked_expect``); ``expect_of`` unwraps it."""

    def __init__(self, r):
        self.r = r


def expect_of(r, operands):
    """The expectation for any result NumPy returned."""
    if isinstance(r, _UninitData):
        return masked_expect(r.r, uninit=True) if isinstance(r.r, np.ma.MaskedArray) and r.r is not np.ma.masked \
            else expect_of(r.r, operands)
    if r is np.ma.masked:
        return {"kind": "masked_constant"}
    if isinstance(r, np.ma.MaskedArray):
        return masked_expect(r)
    if r is None:
        return {"kind": "none"}
    if isinstance(r, tuple) and any(isinstance(x, np.ma.MaskedArray) for x in r):
        return {"kind": "seq", "items": [expect_of(x, operands) for x in r]}
    if isinstance(r, np.ndarray):
        return array_expect(r, operands)
    if isinstance(r, (tuple, list)) and r and all(isinstance(x, np.ndarray) for x in r):
        return {"kind": "tuple", "items": [array_expect(x, operands) for x in r]}
    if isinstance(r, _TProps):
        return tprops(r.d)
    if isinstance(r, _TCells):
        return {"kind": "tcells", "hex": _mask_heap(r.b)}
    if type(r) is float:
        # A StringDType element that is the float NaN NA object (a.item(i), a[i]); np.float64 is a subclass of float and
        # keeps the NumPy-scalar branch below.
        return {"kind": "value", "value": enc(r)}
    if isinstance(r, np.dtype):
        return {"kind": "dtype", "str": r.str, "repr": repr(r)}
    if isinstance(r, (bool, np.bool_)):
        return {"kind": "bool", "value": bool(r)}
    if isinstance(r, str) and not isinstance(r, np.str_):
        # UTF-32 hex like every other text: a printed lone surrogate is not valid JSON text.
        return {"kind": "text", "u": r.encode("utf-32-le", "surrogatepass").hex()}
    if isinstance(r, (np.str_, np.bytes_, bytes, list)):
        return {"kind": "value", "value": enc(r)}
    if isinstance(r, np.generic) or (isinstance(r, int) and not isinstance(r, bool)):
        # A numeric NumPy scalar (an index, a count): compared as a 0-d array. A scalar is immutable and owns nothing,
        # so the writeable / owndata flags np.asarray would invent are left out (NumSharp's 0-d reduction results are
        # read-only, its stand-in for the scalar). A plain Python int (np.linalg.matrix_rank of a 1-D operand returns
        # int(...)) is the int64 NumSharp's 0-d count holds.
        d = array_expect(np.asarray(r) if isinstance(r, np.generic) else np.asarray(r, dtype=np.int64), None)
        d.pop("writeable", None)
        d.pop("owndata", None)
        d["scalar"] = True
        return d
    raise TypeError(f"unexpected result type {type(r)}")


def add(section, op, *, params=None, operands=(), call, phase=1, layout="", mutate=None, note=""):
    """Run one NumPy call and record its outcome.

    ``operands`` is a list of (base, view) pairs; ``call(*views)`` makes the call. ``mutate`` names the operand an
    in-place call writes: its view and whole base buffer afterwards are the expectation. ``phase`` is 1, 2, or
    ``"compute"`` for a call NumPy may answer by computing on text: phase 2 when NumPy returns a result, phase 1 when it
    raises (NumPy's error is then final, so NumSharp must raise the same) -- a PARSE error (NumPy converting text to a
    number, ``could not convert string to float``) included: text -> number conversion is phase 1 since P1.1.
    """
    views = [v for _, v in operands]
    n = sum(1 for c in CASES if c["id"].startswith(f"{section}/{op}/")) + 1
    cid = f"{section}/{op}/{n}"
    assert cid not in _IDS
    _IDS.add(cid)
    case = {"id": cid, "op": op, "params": params or {}, "operands": [describe(b, v) for b, v in operands],
            "phase": phase, "layout": layout}
    if note:
        case["note"] = note
    try:
        r = call(*views)
        if mutate is not None:
            base, view = operands[mutate]
            case["expected"] = {"kind": "mutated", "operand": mutate, "view": array_expect(view, None)}
            # A StringDType base is compared by its values (its cells hold allocator offsets and heap pointers).
            if base.dtype.kind == "T":
                case["expected"]["basevalues"] = tvals(base)
            else:
                case["expected"]["base"] = base.tobytes().hex()
        else:
            case["expected"] = expect_of(r, views)
        if phase == "compute":
            case["phase"] = 2
    except Exception as e:  # noqa: BLE001 -- every NumPy exception is a recorded outcome
        case["error"] = err_info(e)
        if phase == "compute":
            case["phase"] = 2 if _is_parse_error(e) and not _parses_in_phase1(op) else 1
    CASES.append(case)


# Calls whose parse runs only after a computation of the later phase: np.sort_complex sorts the text, THEN casts it
# to complex (b.sort(); b.astype('D')), so its parse error is reachable only once text sorts.
_PARSE_AFTER_PHASE2 = {"sort_complex"}


def _parses_in_phase1(op):
    """True when a parse error of ``op`` is one NumSharp's phase 1 raises: text -> number conversion (P1.1) answers
    every failed int() / float() / complex() of a string and numpy.ma's TypeError around one (the RecursionError of
    numpy.ma.flatten_mask is a truth test, which phase 1 answers too) -- except where a phase-2 computation runs
    before the parse (``_PARSE_AFTER_PHASE2``)."""
    return op not in _PARSE_AFTER_PHASE2


def _is_parse_error(e):
    """An error whose outcome the TEXT decides, so a call that raises it belongs to the computational phase:

    * NumPy's text-to-number conversion failing on a value (``could not convert string to float: np.str_('a')``,
      ``invalid literal for int() with base 10``), also when numpy.ma re-raises it as its own TypeError (``Cannot convert
      fill_value [-- 'c' 'de'] to dtype float64`` is ``raise TypeError(...) from e`` around the parse error: the same text
      parses for a fill like ``'1.5'``);
    * the RecursionError of ``numpy.ma.flatten_mask`` over text: its ``_flatsequence`` recurses into every element that
      has ``__iter__``, and a NON-EMPTY string's characters are strings again, so whether it recurses forever depends on
      whether some element is empty (an all-empty array returns ``[]``) -- a truth test of the text.
    """
    if isinstance(e, RecursionError):
        return True
    if isinstance(e, ValueError) and str(e).startswith(
            ("could not convert string to", "invalid literal for int()", "complex() arg is a malformed string")):
        return True
    return e.__cause__ is not None and _is_parse_error(e.__cause__)


# ----------------------------------------------------------------------------------------------------------------------
# Value pools and layouts
# ----------------------------------------------------------------------------------------------------------------------

# Values chosen to exercise widths (truncation, NUL padding), quotes and escapes in printing, non-ASCII and
# supplementary-plane text (UCS-4 units), embedded NULs (kept) and trailing NULs (stripped on read).
U_POOL = ["a", "bc", "", "def", "Ü", "x y", "it's", 'q"z', "\t", "\U0001F600b", "中文字", "a\x00b", "\\", "Ab", "zz", "é"]
U_ASCII = ["a", "bc", "", "def", "xy", "it's", 'q"z', "\t", "Ab", "a\x00b", "\\", "zz", "Q", "hello", " ", "~"]
S_POOL = [b"a", b"bc", b"", b"def", b"\xff", b"x y", b"it's", b'q"z', b"\t", b"\x80b", b"ab\x00c", b"\\", b"Ab", b"zz",
          b"\x7f", b"\x00"]
S_ASCII = [b"a", b"bc", b"", b"def", b"xy", b"it's", b'q"z', b"\t", b"Ab", b"a\x00b", b"\\", b"zz", b"Q", b"hello",
           b" ", b"~"]


def pool_for(spec, ascii_only=False):
    if spec[0] in "U" or spec[1:2] == "U":
        return U_ASCII if ascii_only else U_POOL
    return S_ASCII if ascii_only else S_POOL


def base_of(spec, n, seed=0, ascii_only=False):
    """A 1-D base of ``n`` elements of ``spec``, its values cycling through the pool from ``seed``."""
    pool = pool_for(spec, ascii_only)
    return np.array([pool[(seed + i) % len(pool)] for i in range(n)], dtype=spec)


# name -> (base length, view builder). Every view addresses the base buffer only.
LAYOUTS = {
    "c1": (6, lambda b: b),
    "c2": (12, lambda b: b.reshape(3, 4)),
    "c3": (24, lambda b: b.reshape(2, 3, 4)),
    "f2": (12, lambda b: b.reshape(4, 3).T),
    "f3": (24, lambda b: b.reshape(4, 3, 2).T),
    "t3": (24, lambda b: b.reshape(2, 3, 4).transpose(1, 2, 0)),
    "s1": (12, lambda b: b[::2]),
    "s2": (24, lambda b: b.reshape(4, 6)[::2, 1::2]),
    "r1": (6, lambda b: b[::-1]),
    "r2": (12, lambda b: b.reshape(3, 4)[::-1, ::-1]),
    "o1": (8, lambda b: b[2:6]),
    "rs": (12, lambda b: b[10:0:-3]),
    "col": (12, lambda b: b.reshape(3, 4)[:, 1]),
    "nx": (12, lambda b: b.reshape(3, 4)[:, None, :]),
    "bc": (4, lambda b: np.broadcast_to(b, (3, 4))),
    "bcol": (3, lambda b: np.broadcast_to(b[:, None], (3, 4))),
    "z0": (5, lambda b: b[3, ...]),
    "one": (5, lambda b: b[2:3]),
    "d5": (12, lambda b: b.reshape(1, 2, 1, 3, 2)),
    "e1": (0, lambda b: b),
    "e2": (0, lambda b: b.reshape(0, 3)),
    "e3": (6, lambda b: b.reshape(2, 3)[:, 3:]),
}
CORE_LAYOUTS = ["c1", "c2", "f2", "t3", "s1", "r2", "o1", "col", "bc", "z0", "e2"]


def operand(spec, layout, seed=0, ascii_only=False):
    n, build = LAYOUTS[layout]
    base = base_of(spec, n, seed, ascii_only)
    return base, build(base)


def num_operand(arr):
    return plain(np.asarray(arr))


SPECS = ["<U1", "<U3", "<U5", "|S1", "|S3", "|S5"]
U_SPECS = ["<U1", "<U3", "<U5"]
S_SPECS = ["|S1", "|S3", "|S5"]


# ----------------------------------------------------------------------------------------------------------------------
# A. Descriptors, promotion, casting rules
# ----------------------------------------------------------------------------------------------------------------------

def dtype_props(spec):
    d = np.dtype(spec)
    return {"kind": "props", "str": d.str, "name": d.name, "char": d.char, "dkind": d.kind, "num": int(d.num),
            "itemsize": int(d.itemsize), "byteorder": d.byteorder, "isbuiltin": int(d.isbuiltin),
            "isnative": bool(d.isnative), "alignment": int(d.alignment), "repr": repr(d), "strfmt": str(d)}


def gen_dtypes():
    specs = ["S", "S0", "S1", "S10", "|S5", "<S5", ">S5", "=S5", "a", "a5", "c", "U", "U0", "U1", "U32", "<U3", ">U3",
             "=U2", "str", "str_", "bytes", "bytes_", "U1000", "S1000", "<U0", "|S0"]
    for spec in specs:
        case = {"id": f"dtype/props/{spec}", "op": "dtype_props", "params": {"spec": spec}, "operands": [],
                "phase": 1, "layout": ""}
        try:
            case["expected"] = dtype_props(spec)
        except Exception as e:  # noqa: BLE001
            case["error"] = err_info(e)
        CASES.append(case)
    for spec in ["U-1", "S-1", "Ux", "U3.5", "|U2", "S2147483648", "U1073741824", "string_", "unicode_", "a0x"]:
        case = {"id": f"dtype/props/err/{spec}", "op": "dtype_props", "params": {"spec": spec}, "operands": [],
                "phase": 1, "layout": ""}
        try:
            case["expected"] = dtype_props(spec)
        except Exception as e:  # noqa: BLE001
            case["error"] = err_info(e)
        CASES.append(case)

    strs = ["U1", "U5", "S1", "S5", "U", "S"]
    nums = ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "c16"]
    for a in strs:
        for b in strs + nums:
            for x, y in ((a, b), (b, a)):
                add("dtype", "promote_types", params={"a": x, "b": y},
                    call=lambda x=x, y=y: np.promote_types(x, y))
    for group in (["U2", "S5"], ["S3", "S7", "U1"], ["U4", "i8"], ["S2", "?", "u1"], ["U3", "U9", "U1"],
                  ["S4", "f8", "U1"], ["U", "S"], ["S", "i4"]):
        add("dtype", "result_type", params={"specs": group}, call=lambda g=group: np.result_type(*g))

    pairs = [("U3", "U5"), ("U5", "U3"), ("U3", "U3"), ("S3", "S5"), ("S5", "S3"), ("S3", "U3"), ("U3", "S3"),
             ("S3", "U2"), ("U3", "S5"), ("U", "U3"), ("U3", "U"), ("S", "U"), ("U", "S"), ("i4", "U"), ("i4", "U11"),
             ("i4", "U10"), ("u1", "S3"), ("u1", "S2"), ("?", "U5"), ("?", "S1"), ("f8", "U32"), ("f8", "U31"),
             ("i8", "S21"), ("u8", "U20"), ("c16", "U"), ("U3", "i4"), ("S3", "f8"), ("U3", "?"), (">U3", "<U3"),
             ("<U3", ">U3"), ("f2", "S"), ("i2", "U6")]
    for f, t in pairs:
        for casting in ("no", "equiv", "safe", "same_kind", "unsafe"):
            add("dtype", "can_cast", params={"from": f, "to": t, "casting": casting},
                call=lambda f=f, t=t, c=casting: np.can_cast(f, t, c))
    for a, b in [("U5", "str_"), ("S3", "bytes_"), ("U5", "character"), ("S3", "character"), ("U5", "flexible"),
                 ("S3", "flexible"), ("U5", "generic"), ("U5", "number"), ("S3", "integer"), ("i4", "character"),
                 ("U5", "bytes_"), ("S3", "str_"), ("U5", "U"), ("U5", "U3"), ("S3", "S"), ("U2", "S2")]:
        target = {"str_": np.str_, "bytes_": np.bytes_, "character": np.character, "flexible": np.flexible,
                  "generic": np.generic, "number": np.number, "integer": np.integer}.get(b, b)
        add("dtype", "issubdtype", params={"a": a, "b": b}, call=lambda a=a, t=target: np.issubdtype(a, t))


# ----------------------------------------------------------------------------------------------------------------------
# B. Creation from Python values
# ----------------------------------------------------------------------------------------------------------------------

def _has_leaf(v):
    """Whether a nested list holds any value at all."""
    if isinstance(v, list):
        return any(_has_leaf(x) for x in v)
    return True


def gen_creation():
    value_sets = [
        ["a", "bc", "def"], ["", ""], [""], ["Ü", "é"], ["\U0001F600", "ab"], ["a\x00b", "c"], ["ab\x00", "c\x00\x00"],
        ["\ud800", "x"], [["a", "bc"], ["def", ""]], [[["a"], ["b"]], [["cc"], ["d"]]], [b"a", b"bc"], [b"\xff", b""],
        [[b"a", b"b"], [b"cc", b"\x00d"]], ["a", b"bc"], [b"a", "bc"], [[]], [["a", "b"], ["c"]], ["it's", 'q"z'],
        ["x" * 40, "y"],
    ]
    dtypes = [None, "U", "S", "U2", "S2", "<U1", "|S1", "U8"]
    for vals in value_sets:
        for dt in dtypes:
            # A value set with no text in it ([[]]) is NumPy's float64 without a string dtype: not a string case.
            if dt is None and not _has_leaf(vals):
                continue
            add("creation", "array", params={"values": enc(vals), "dtype": dt},
                call=lambda v=vals, d=dt: np.array(v, dtype=d))
    for vals in (["a", "bc"], [["a", "bc"], ["d", "e"]]):
        for ndmin in (0, 1, 2, 3):
            add("creation", "array", params={"values": enc(vals), "dtype": None, "ndmin": ndmin},
                call=lambda v=vals, n=ndmin: np.array(v, ndmin=n))
        for order in ("C", "F"):
            add("creation", "array", params={"values": enc(vals), "dtype": "U4", "order": order},
                call=lambda v=vals, o=order: np.array(v, dtype="U4", order=o))
    # None stores its text under a string dtype (NumPy's str(None)); text into numbers / numbers into text convert.
    add("creation", "array", params={"values": enc([None, "a"]), "dtype": "U"},
        call=lambda: np.array([None, "a"], dtype="U"))
    add("creation", "array", params={"values": enc([None]), "dtype": "S5"}, call=lambda: np.array([None], dtype="S5"))
    add("creation", "array", params={"values": enc(["a", 1]), "dtype": None},
        call=lambda: np.array(["a", 1]))
    add("creation", "array", params={"values": enc(["12", "3"]), "dtype": "i8"},
        call=lambda: np.array(["12", "3"], dtype="i8"))
    add("creation", "array", params={"values": enc([1.5, 2.0]), "dtype": "U"},
        call=lambda: np.array([1.5, 2.0], dtype="U"))

    for v in ["abc", "", "a\x00", "\U0001F600", b"ab", b"", b"\x00x", "it's"]:
        for dt in (None, "U", "S", "U2", "S2", "U6"):
            add("creation", "array_scalar", params={"value": enc(v), "dtype": dt},
                call=lambda v=v, d=dt: np.array(v, dtype=d))

    for spec in ("U3", "S3", "U", "S", "U1", "<U2"):
        for shape in ((3,), (2, 3), (0,), (2, 0), ()):
            add("creation", "zeros", params={"shape": list(shape), "dtype": spec},
                call=lambda s=shape, d=spec: np.zeros(s, dtype=d))
    for spec in ("U3", "U", "<U5"):
        for shape in ((3,), (2, 3), (0, 2), ()):
            add("creation", "empty", params={"shape": list(shape), "dtype": spec},
                call=lambda s=shape, d=spec: np.empty(s, dtype=d))
    for fill in ("ab", "", "Ü", "\U0001F600z", b"xy", b"\xff", "a\x00"):
        for dt in (None, "U1", "U4", "S1", "S4"):
            for shape in ((3,), (2, 2)):
                add("creation", "full", params={"shape": list(shape), "fill": enc(fill), "dtype": dt},
                    call=lambda s=shape, f=fill, d=dt: np.full(s, f, dtype=d))
    add("creation", "full", params={"shape": [2], "fill": enc(7), "dtype": "U3"},
        call=lambda: np.full((2,), 7, dtype="U3"))
    add("creation", "ones", params={"shape": [2], "dtype": "U3"}, call=lambda: np.ones((2,), dtype="U3"))

    for spec in ("<U3", "|S3"):
        for lay in ("c2", "f2", "t3", "s2", "bc", "z0", "e2"):
            for dt in (None, "U5", "S2"):
                add("creation", "zeros_like", params={"dtype": dt}, operands=[operand(spec, lay)], layout=lay,
                    call=lambda a, d=dt: np.zeros_like(a, dtype=d))
                add("creation", "full_like", params={"fill": enc("qq" if spec[1] == "U" else b"qq"), "dtype": dt},
                    operands=[operand(spec, lay)], layout=lay,
                    call=lambda a, d=dt, s=spec: np.full_like(a, "qq" if s[1] == "U" else b"qq", dtype=d))
            if spec[1] == "U":
                add("creation", "empty_like", params={"dtype": None}, operands=[operand(spec, lay)], layout=lay,
                    call=lambda a: np.empty_like(a))
            add("creation", "ones_like", params={}, operands=[operand(spec, lay)], layout=lay,
                call=lambda a: np.ones_like(a))

    for spec in SPECS:
        for lay in CORE_LAYOUTS:
            for dt in (None, spec, "U8", "S8"):
                asc = dt is not None and dt[0] == "S" and spec[1] == "U"
                add("creation", "asarray", params={"dtype": dt}, operands=[operand(spec, lay, ascii_only=asc)],
                    layout=lay, call=lambda a, d=dt: np.asarray(a, dtype=d))
            add("creation", "ascontiguousarray", operands=[operand(spec, lay)], layout=lay,
                call=lambda a: np.ascontiguousarray(a))
            add("creation", "asfortranarray", operands=[operand(spec, lay)], layout=lay,
                call=lambda a: np.asfortranarray(a))

    for raw, dt, count, offset in [(b"abcdef", "S2", -1, 0), (b"abcdef", "S3", 1, 3), (b"abcdefg", "S2", -1, 0),
                                   ("hi!".encode("utf-32-le"), "U1", -1, 0), ("hi!".encode("utf-32-le"), "U3", -1, 0),
                                   ("abcd".encode("utf-32-le"), "U2", 1, 8), (b"ab", "U1", -1, 0)]:
        add("creation", "frombuffer", params={"buffer": raw.hex(), "dtype": dt, "count": count, "offset": offset},
            # A bytearray: mutable like a C# byte[], so NumPy's view is writeable as NumSharp's is.
            call=lambda r=raw, d=dt, c=count, o=offset: np.frombuffer(bytearray(r), dtype=d, count=c, offset=o))

    # np.frombuffer's checks in NumPy's order (an unsized class, the offset, a partial element, a count too large), the
    # empty results (count 0, offset at the end), NumPy's one-byte 'c' and the legacy 'a' alias. Each runs through the
    # descriptor overload ("frombuffer") and the dtype-STRING overload ("frombuffer_str"), whose own parser once read
    # 'c' / 'S1' as NumSharp's 2-byte Char. A big-endian '>U2' is left to the unit tests: NumSharp decodes it into a
    # native copy (no big-endian storage), so its descriptor and flags differ by design.
    fb_more = [(b"abcdef", "S2", 0, 0), (b"abcdef", "S3", -1, 6), (b"abcdef", "S2", 0, 6), (b"abcdef", "S", -1, 0),
               (b"abcdef", "U", -1, 0), (b"abcdef", "S2", -1, 7), (b"abcdef", "S2", -1, -1), (b"abcdef", "S4", -1, 0),
               (b"abcdef", "S4", 2, 0), (b"abcdef", "S3", 2, 1), (b"abcdef", "S3", 1, 1), (b"abcdef", "c", -1, 0),
               (b"abcdef", "S1", -1, 2), (b"abcdef", "|S2", 1, 4), ("hi".encode("utf-32-le"), "U1", 3, 0),
               (b"abcdefg", "U1", -1, 0), ("hey!".encode("utf-32-le"), "<U2", -1, 0), ("hey!".encode("utf-32-le"), "U1", 2, 4)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # 'a3' is a deprecated alias of 'S3' that still works
        fb_more.append((b"abcdef", "a3", -1, 0))
        for raw, dt, count, offset in fb_more:
            for op in ("frombuffer", "frombuffer_str"):
                add("creation", op, params={"buffer": raw.hex(), "dtype": dt, "count": count, "offset": offset},
                    call=lambda r=raw, d=dt, c=count, o=offset: np.frombuffer(bytearray(r), dtype=d, count=c, offset=o))

    # np.ndarray(shape, dtype, buffer=...): a string dtype lays fixed-width elements over the buffer's bytes - offsets,
    # explicit byte strides (a stride of 0 repeats an element and stays writeable), Fortran order, the shape (-1,), and
    # NumPy's two refusals (too small; strides outside the buffer). A U dtype over ASCII bytes stores code points above
    # U+10FFFF; their BYTES are compared, which NumPy produces without decoding them. An unsized 'S'/'U' (NumPy builds a
    # zero-width S0/U0 array, NumSharp refuses) is pinned by a unit test instead.
    nd_buf = b"abcdefghijkl"
    for shape, dt, offset, strides, order in [((4,), "S3", 0, None, "C"), ((2, 3), "S2", 0, None, "C"),
                                              ((3,), "S3", 3, None, "C"), ((2,), "S3", 0, (6,), "C"),
                                              ((2, 2), "S3", 0, None, "F"), ((-1,), "S3", 0, None, "C"),
                                              ((-1,), "S5", 2, None, "C"), ((3,), "S5", 0, None, "C"),
                                              ((2,), "S3", 0, (9,), "C"), ((3,), "U1", 0, None, "C"),
                                              ((2,), "S3", 0, (0,), "C"), ((0,), "S3", 12, None, "C"),
                                              ((2, 2), "S2", 6, (-6, 2), "C"), ((3,), "S4", 0, None, "A"),
                                              ((2, 3), "S2", 0, None, "F"), ((2,), "U2", 0, None, "C"),
                                              ((1, 2), "S2", 2, (4, 6), "C")]:
        add("creation", "ndarray",
            params={"shape": list(shape), "dtype": dt, "buffer": nd_buf.hex(), "offset": offset,
                    "strides": list(strides) if strides is not None else None, "order": order},
            call=lambda s=shape, d=dt, o=offset, st=strides, od=order:
                np.ndarray(s, d, buffer=bytearray(nd_buf), offset=o, strides=st, order=od))
    # Over a string ARRAY's bytes: the same descriptor shares the array's elements; another width or class reads the
    # bytes (a U1 view of U2 text, an S4 view of UCS-4 units, uint32 code points).
    for spec, shape, dt, offset in [("<U2", (2,), "<U2", 0), ("<U2", (2,), "<U2", 8), ("<U2", (4,), "<U1", 0),
                                    ("<U2", (4,), "S4", 0), ("<U2", (4,), "<u4", 0), ("|S4", (2,), "|S2", 4),
                                    ("|S4", (4,), "|S1", 2), ("<U3", (1,), "<U3", 12)]:
        add("creation", "ndarray", params={"shape": list(shape), "dtype": dt, "offset": offset},
            operands=[operand(spec, "c1", seed=3, ascii_only=True)], layout="c1",
            call=lambda a, s=shape, d=dt, o=offset: np.ndarray(s, d, buffer=a, offset=o))


# ----------------------------------------------------------------------------------------------------------------------
# C. Casts between string dtypes (and the text <-> number casts NumPy computes)
# ----------------------------------------------------------------------------------------------------------------------

def gen_astype():
    targets = ["<U1", "<U2", "<U3", "<U8", "|S1", "|S3", "|S8", "U", "S"]
    for spec in SPECS:
        for lay in CORE_LAYOUTS + ["c3", "r1", "rs", "d5", "e1"]:
            for t in targets:
                for ascii_only in (False, True):
                    if ascii_only and not ((spec[1] == "U" and t[-2:-1] in "S|" or t == "S") or
                                           (spec[1] == "S" and (t.startswith("<U") or t == "U"))):
                        continue
                    add("astype", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
                        operands=[operand(spec, lay, seed=3, ascii_only=ascii_only)], layout=lay,
                        call=lambda a, t=t: a.astype(t))
    for spec in ("<U3", "|S3"):
        for lay in ("c2", "f2", "t3", "r2", "bc"):
            for order in ("C", "F", "A", "K"):
                for t in ("<U5", "|S5", spec):
                    add("astype", "astype", params={"dtype": t, "casting": "unsafe", "order": order, "copy": True},
                        operands=[operand(spec, lay, ascii_only=True)], layout=lay,
                        call=lambda a, t=t, o=order: a.astype(t, order=o))
            for t in (spec, "<U5", "|S5"):
                add("astype", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": False},
                    operands=[operand(spec, lay, ascii_only=True)], layout=lay,
                    call=lambda a, t=t: a.astype(t, copy=False))
        for t in ("<U1", "<U3", "<U5", "|S1", "|S3", "|S5", "U", "S"):
            for casting in ("no", "equiv", "safe", "same_kind", "unsafe"):
                add("astype", "astype", params={"dtype": t, "casting": casting, "order": "K", "copy": True},
                    operands=[operand(spec, "c1", ascii_only=True)], layout="c1",
                    call=lambda a, t=t, c=casting: a.astype(t, casting=c))
    # Non-ASCII text into bytes and non-ASCII bytes into text raise; the position is within the element.
    for vals, t in [(["ab", "cé"], "S3"), (["\ud800"], "S2"), (["x", "\U0001F600"], "S"), ([b"a", b"b\xff"], "U2"),
                    ([b"\x80"], "U")]:
        add("astype", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
            operands=[plain(np.array(vals))], layout="c1", call=lambda a, t=t: a.astype(t))
    # Text <-> numbers: NumPy parses / formats (phase 1 since P1.1; section strconv has the full matrix).
    for vals, t in [(["12", "-3"], "i8"), (["1.5", "nan"], "f8"), (["ab"], "i4"), (["True", ""], "?"),
                    ([b"7", b"8"], "u1"), (["1+2j"], "c16")]:
        add("astype", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
            operands=[plain(np.array(vals))], layout="c1", call=lambda a, t=t: a.astype(t))
    for arr, t in [(np.array([1, -2], dtype="i8"), "U"), (np.array([1.5, 2.0]), "S"), (np.array([True]), "U5"),
                   (np.array([300], dtype="i2"), "S2")]:
        add("astype", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
            operands=[plain(arr)], layout="c1", call=lambda a, t=t: a.astype(t))


# ----------------------------------------------------------------------------------------------------------------------
# D. Views and reshaping
# ----------------------------------------------------------------------------------------------------------------------

def gen_views():
    for spec in ("<U3", "|S3", "<U1"):
        for lay in ("c2", "c3", "f2", "t3", "s2", "r2", "col", "nx", "bc", "z0", "e2", "d5"):
            op = operand(spec, lay)
            shape = op[1].shape
            add("views", "transpose", params={"axes": None}, operands=[op], layout=lay, call=lambda a: a.transpose())
            if len(shape) >= 2:
                add("views", "swapaxes", params={"a1": 0, "a2": -1}, operands=[op], layout=lay,
                    call=lambda a: np.swapaxes(a, 0, -1))
                add("views", "moveaxis", params={"src": 0, "dst": -1}, operands=[op], layout=lay,
                    call=lambda a: np.moveaxis(a, 0, -1))
                add("views", "rot90", params={"k": 1}, operands=[op], layout=lay, call=lambda a: np.rot90(a))
                add("views", "diagonal", params={"offset": 0}, operands=[op], layout=lay,
                    call=lambda a: np.diagonal(a))
            for order in ("C", "F", "A", "K"):
                add("views", "ravel", params={"order": order}, operands=[op], layout=lay,
                    call=lambda a, o=order: np.ravel(a, order=o))
            for order in ("C", "F"):
                add("views", "reshape", params={"shape": [-1], "order": order}, operands=[op], layout=lay,
                    call=lambda a, o=order: np.reshape(a, (-1,), order=o))
            add("views", "squeeze", params={"axis": None}, operands=[op], layout=lay, call=lambda a: np.squeeze(a))
            add("views", "expand_dims", params={"axis": 0}, operands=[op], layout=lay,
                call=lambda a: np.expand_dims(a, 0))
            add("views", "flip", params={"axis": None}, operands=[op], layout=lay, call=lambda a: np.flip(a))
            add("views", "atleast_2d", operands=[op], layout=lay, call=lambda a: np.atleast_2d(a))
            add("views", "atleast_3d", operands=[op], layout=lay, call=lambda a: np.atleast_3d(a))
        for lay in ("c1", "c2", "r1"):
            op = operand(spec, lay)
            add("views", "broadcast_to", params={"shape": [2] + list(op[1].shape)}, operands=[op], layout=lay,
                call=lambda a: np.broadcast_to(a, (2,) + a.shape))
        add("views", "reshape", params={"shape": [2, 2, 3], "order": "C"}, operands=[operand(spec, "c2")],
            layout="c2", call=lambda a: np.reshape(a, (2, 2, 3)))
        add("views", "reshape", params={"shape": [2, 6], "order": "C"}, operands=[operand(spec, "f2")],
            layout="f2", call=lambda a: np.reshape(a, (2, 6)))

    # Reinterpreting views: unit views, class changes of the same item size, and numeric views of the bytes.
    for spec, lay, t in [("<U3", "c1", "<U1"), ("<U4", "c2", "<U2"), ("<U2", "c1", "<U4"), ("<U1", "c1", "|S4"),
                         ("|S4", "c1", "<U1"), ("|S4", "c1", "|S2"), ("|S2", "c2", "|S4"), ("<U1", "c1", "<i4"),
                         ("<U2", "c1", "<u4"), ("|S4", "c1", "<i4"), ("|S2", "c1", "|u1"), ("|S8", "c1", "<f8"),
                         ("<U3", "f2", "<U1"), ("<U3", "s1", "<U1"), ("<U3", "z0", "<U1"), ("<U3", "z0", "<U3"),
                         ("<U3", "r1", "<U1"), ("|S6", "c1", "|S4"), ("<U2", "c1", "|S3"), ("<U3", "e1", "<U1"),
                         ("<U3", "c1", "U"), ("|S3", "c1", "S"), ("<U3", "bc", "<U1"), ("|S3", "col", "|S1")]:
        add("views", "view", params={"dtype": t}, operands=[operand(spec, lay, ascii_only=True)], layout=lay,
            call=lambda a, t=t: a.view(t))
    for arr, t in [(np.array([97, 98, 0x1F600], dtype="<u4"), "<U1"), (np.array([97, 98, 99, 100], dtype="<i4"), "<U2"),
                   (np.frombuffer(b"abcdef", dtype="u1"), "|S2"), (np.frombuffer(b"abcdef", dtype="u1"), "|S3"),
                   (np.array([0x110000], dtype="<u4"), "<U1")]:
        add("views", "view", params={"dtype": t}, operands=[plain(arr)], layout="c1", call=lambda a, t=t: a.view(t))

    # Basic and advanced indexing.
    idx_cases = {
        "c1": [[{"int": 1}], [{"int": -1}], [{"slice": [1, 4, None]}], [{"slice": [None, None, -2]}],
               [{"ellipsis": True}], [{"none": True}], [{"arr": [0, 2, -1]}], [{"arr": [[0, 1], [5, 4]]}],
               [{"mask": [True, False, True, False, False, True]}], [{"int": 7}], [{"arr": [6]}]],
        "c2": [[{"int": 1}], [{"int": 1}, {"int": 2}], [{"slice": [None, None, None]}, {"int": 0}],
               [{"ellipsis": True}, {"int": -1}], [{"arr": [2, 0]}], [{"slice": [None, None, None]}, {"arr": [3, 1]}],
               [{"arr": [0, 2]}, {"arr": [1, 3]}], [{"mask2": [[True, False, False, True], [False] * 4, [True] * 4]}],
               [{"none": True}, {"int": 0}], [{"int": 0}, {"ellipsis": True}]],
        "f2": [[{"int": 1}], [{"slice": [None, None, None]}, {"int": 2}], [{"arr": [2, 0]}], [{"arr": [0, 1]},
                                                                                              {"arr": [1, 0]}]],
        "r2": [[{"int": 0}], [{"arr": [1, 2]}], [{"slice": [None, None, 2]}, {"slice": [1, None, None]}]],
        "bc": [[{"int": 2}], [{"arr": [0, 0, 2]}], [{"slice": [None, None, None]}, {"int": 3}]],
        "z0": [[{"ellipsis": True}], [{"none": True}]],
    }
    for spec in ("<U3", "|S3"):
        for lay, idxs in idx_cases.items():
            for idx in idxs:
                key = tuple(_py_index(e) for e in idx)
                add("views", "getitem", params={"index": idx}, operands=[operand(spec, lay)], layout=lay,
                    call=lambda a, k=key: a[k])


def _py_index(e):
    if "int" in e:
        return e["int"]
    if "slice" in e:
        return slice(*e["slice"])
    if "none" in e:
        return None
    if "ellipsis" in e:
        return Ellipsis
    if "arr" in e:
        return np.array(e["arr"], dtype=np.int64)
    if "mask" in e:
        return np.array(e["mask"], dtype=bool)
    if "mask2" in e:
        return np.array(e["mask2"], dtype=bool)
    raise ValueError(e)


# ----------------------------------------------------------------------------------------------------------------------
# E. Copies and movement
# ----------------------------------------------------------------------------------------------------------------------

def gen_movement():
    for spec in ("<U3", "|S3"):
        for lay in CORE_LAYOUTS + ["c3", "f3", "r1", "rs", "nx", "bcol", "d5", "e3", "one"]:
            op = operand(spec, lay)
            nd = op[1].ndim
            for order in ("C", "F", "A", "K"):
                add("movement", "copy", params={"order": order}, operands=[op], layout=lay,
                    call=lambda a, o=order: np.copy(a, order=o))
            for order in ("C", "F"):
                add("movement", "flatten", params={"order": order}, operands=[op], layout=lay,
                    call=lambda a, o=order: a.flatten(order=o))
            add("movement", "repeat", params={"repeats": 2, "axis": None}, operands=[op], layout=lay,
                call=lambda a: np.repeat(a, 2))
            add("movement", "tile", params={"reps": [2, 1]}, operands=[op], layout=lay,
                call=lambda a: np.tile(a, (2, 1)))
            add("movement", "roll", params={"shift": 2, "axis": None}, operands=[op], layout=lay,
                call=lambda a: np.roll(a, 2))
            add("movement", "resize", params={"shape": [5]}, operands=[op], layout=lay,
                call=lambda a: np.resize(a, (5,)))
            add("movement", "take", params={"indices": [0, -1, 1], "axis": None, "mode": "raise"}, operands=[op],
                layout=lay, call=lambda a: np.take(a, [0, -1, 1]))
            add("movement", "take", params={"indices": [5, -7, 100], "axis": None, "mode": "wrap"}, operands=[op],
                layout=lay, call=lambda a: np.take(a, [5, -7, 100], mode="wrap"))
            add("movement", "take", params={"indices": [5, -7, 100], "axis": None, "mode": "clip"}, operands=[op],
                layout=lay, call=lambda a: np.take(a, [5, -7, 100], mode="clip"))
            if nd >= 1:
                add("movement", "repeat", params={"repeats": [2], "axis": 0}, operands=[op], layout=lay,
                    call=lambda a: np.repeat(a, [2], axis=0))
                add("movement", "roll", params={"shift": -1, "axis": 0}, operands=[op], layout=lay,
                    call=lambda a: np.roll(a, -1, axis=0))
                add("movement", "take", params={"indices": [1, 0], "axis": 0, "mode": "clip"}, operands=[op],
                    layout=lay, call=lambda a: np.take(a, [1, 0], axis=0, mode="clip"))
                add("movement", "compress", params={"condition": [True, False, True], "axis": 0}, operands=[op],
                    layout=lay, call=lambda a: np.compress([True, False, True], a, axis=0))
                add("movement", "flip", params={"axis": 0}, operands=[op], layout=lay, call=lambda a: np.flip(a, 0))
                add("movement", "delete", params={"obj": [0], "axis": 0}, operands=[op], layout=lay,
                    call=lambda a: np.delete(a, [0], axis=0))
                add("movement", "insert", params={"obj": 1, "values": enc("ins" if spec[1] == "U" else b"ins"),
                                                  "axis": 0}, operands=[op], layout=lay,
                    call=lambda a, s=spec: np.insert(a, 1, "ins" if s[1] == "U" else b"ins", axis=0))
                add("movement", "append", params={"values": enc(["zz"] if spec[1] == "U" else [b"zz"]), "axis": None},
                    operands=[op], layout=lay,
                    call=lambda a, s=spec: np.append(a, ["zz"] if s[1] == "U" else [b"zz"]))
            if nd >= 2:
                add("movement", "tril", params={"k": 0}, operands=[op], layout=lay, call=lambda a: np.tril(a))
                add("movement", "triu", params={"k": 1}, operands=[op], layout=lay, call=lambda a: np.triu(a, 1))
                add("movement", "take_along_axis", params={"indices": np.zeros((1,) * (nd - 1) + (2,), int).tolist(),
                                                           "axis": -1}, operands=[op], layout=lay,
                    call=lambda a: np.take_along_axis(a, np.zeros((1,) * (a.ndim - 1) + (2,), int), -1))
                add("movement", "split", params={"sections": 2, "axis": -1}, operands=[op], layout=lay,
                    call=lambda a: np.array_split(a, 2, axis=-1))
            if nd == 1:
                add("movement", "diag", params={"k": 0}, operands=[op], layout=lay, call=lambda a: np.diag(a))
                add("movement", "diag", params={"k": -1}, operands=[op], layout=lay, call=lambda a: np.diag(a, -1))
                add("movement", "diagflat", params={"k": 1}, operands=[op], layout=lay,
                    call=lambda a: np.diagflat(a, 1))
            # Padding: every mode that only moves elements, and constant mode with a text constant.
            text = "pd" if spec[1] == "U" else b"pd"
            for mode in ("edge", "reflect", "symmetric", "wrap"):
                add("movement", "pad", params={"pad_width": 1, "mode": mode, "constant": None}, operands=[op],
                    layout=lay, call=lambda a, m=mode: np.pad(a, 1, mode=m))
            add("movement", "pad", params={"pad_width": [[1, 2]] * nd if nd else 1, "mode": "constant",
                                           "constant": enc(text)}, operands=[op], layout=lay,
                call=lambda a, t=text: np.pad(a, [(1, 2)] * a.ndim if a.ndim else 1, mode="constant",
                                              constant_values=t))
            if spec[1] == "U":
                add("movement", "pad", params={"pad_width": 1, "mode": "empty", "constant": None}, operands=[op],
                    layout=lay, call=lambda a: np.pad(a, 1, mode="empty"))
            add("movement", "pad", params={"pad_width": 1, "mode": "constant", "constant": None}, operands=[op],
                layout=lay, call=lambda a: np.pad(a, 1))

        # Joining operands of different widths and classes: the result is the common string dtype.
        for other, lay2 in (("<U5", "c2"), ("|S5", "f2"), ("<U1", "r2"), ("|S1", "bc")):
            a = operand(spec, "c2")
            b = operand(other, lay2, seed=4, ascii_only=True)
            for axis in (0, 1, None):
                add("movement", "concatenate", params={"axis": axis}, operands=[a, b], layout=f"c2+{lay2}",
                    call=lambda x, y, ax=axis: np.concatenate([x, y], axis=ax))
            for fn in ("stack", "vstack", "hstack", "dstack", "column_stack"):
                add("movement", fn, params={"axis": 0}, operands=[a, b], layout=f"c2+{lay2}",
                    call=lambda x, y, f=fn: np.stack([x, y]) if f == "stack" else getattr(np, f)([x, y]))
            add("movement", "where", params={"cond": [[True, False, True, False], [False] * 4, [True] * 4]},
                operands=[a, b], layout=f"c2+{lay2}",
                call=lambda x, y: np.where(np.array([[True, False, True, False], [False] * 4, [True] * 4]), x, y))
            add("movement", "choose", params={"index": [[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]]},
                operands=[a, b], layout=f"c2+{lay2}",
                call=lambda x, y: np.choose(np.array([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]]), [x, y]))
        add("movement", "concatenate", params={"axis": 0}, operands=[operand(spec, "c1"), plain(np.arange(3))],
            layout="c1+int", call=lambda x, y: np.concatenate([x, y]))
        text3 = plain(np.array(["xy"] * 3 if spec[1] == "U" else [b"xy"] * 3))
        add("movement", "where", params={"cond": [True, False, True]}, operands=[text3, plain(np.array([1, 2, 3]))],
            layout="c1+int", call=lambda x, y: np.where(np.array([True, False, True]), x, y))
        # Scalar text operands of np.where (NumPy's weak-scalar promotion for str / bytes).
        for xv, yv in (("long text", None), (None, "q"), ("Ü", "w")):
            xs = xv if spec[1] == "U" else (xv.encode() if xv and xv.isascii() else xv)
            ys = yv if spec[1] == "U" else (yv.encode() if yv and yv.isascii() else yv)
            add("movement", "where_scalar", params={"cond": [True, False, True, False, True, False],
                                                    "x": enc(xs) if xs is not None else None,
                                                    "y": enc(ys) if ys is not None else None},
                operands=[operand(spec, "c1")], layout="c1",
                call=lambda a, x=xs, y=ys: np.where(np.array([True, False, True, False, True, False]),
                                                    a if x is None else x, a if y is None else y))
        a2 = operand(spec, "c2")
        b2 = operand(spec, "c2", seed=5)
        add("movement", "meshgrid", params={"indexing": "xy"}, operands=[operand(spec, "c1"), operand(spec, "o1", 2)],
            layout="c1+o1", call=lambda x, y: np.meshgrid(x, y))
        add("movement", "broadcast_arrays", operands=[operand(spec, "c2"), plain(base_of(spec, 4, 2))], layout="c2+c1",
            call=lambda x, y: np.broadcast_arrays(x, y))
        add("movement", "unstack", params={"axis": 1}, operands=[a2], layout="c2", call=lambda a: np.unstack(a, axis=1))
        add("movement", "extract", params={"condition": [[True, False, True, False], [False] * 4, [True] * 4]},
            operands=[b2], layout="c2",
            call=lambda a: np.extract(np.array([[True, False, True, False], [False] * 4, [True] * 4]), a))


# ----------------------------------------------------------------------------------------------------------------------
# F. In-place writes
# ----------------------------------------------------------------------------------------------------------------------

def _setitem(a, key, value):
    a[key] = value


def gen_mutation():
    # No None: a C# null assignment is a caller error, not Python's str(None) (creation keeps its None cases).
    u_values = ["zz", "", "longer text", "Ü", "\U0001F600", "a\x00", b"by"]
    s_values = [b"zz", b"", b"longer", b"\xff", "zz", "é", b"\x00x"]
    for spec in ("<U3", "|S3"):
        values = u_values if spec[1] == "U" else s_values
        for lay in ("c1", "c2", "f2", "s1", "r2", "col", "z0", "bc", "e2"):
            for v in values:
                for idx in ([{"int": 0}], [{"ellipsis": True}], [{"slice": [None, None, 2]}], [{"arr": [0, -1]}],
                            [{"mask": [True, False, True, False, False, True]}]):
                    if "mask" in idx[0] and lay != "c1":
                        continue
                    key = tuple(_py_index(e) for e in idx)
                    add("mutation", "setitem", params={"index": idx, "value": enc(v)},
                        operands=[operand(spec, lay)], layout=lay, mutate=0,
                        call=lambda a, k=key, v=v: _setitem(a, k, v))
            add("mutation", "fill", params={"value": enc(values[0])}, operands=[operand(spec, lay)], layout=lay,
                mutate=0, call=lambda a, v=values[0]: a.fill(v))
            add("mutation", "fill", params={"value": enc(values[3])}, operands=[operand(spec, lay)], layout=lay,
                mutate=0, call=lambda a, v=values[3]: a.fill(v))
            add("mutation", "setitem", params={"index": [{"int": 0}], "value": enc(5)}, operands=[operand(spec, lay)],
                layout=lay, mutate=0, call=lambda a: _setitem(a, 0, 5))
        # Array values: broadcasting, sequences into one element, other widths and classes.
        for lay in ("c2", "f2"):
            for vals in (["p", "qq", "rrrr", "s"], [["x"], ["y"], ["z"]], [[["only"] * 4] * 3], ["a", "b"]):
                vv = vals if spec[1] == "U" else _to_bytes(vals)
                add("mutation", "setitem_array", params={"index": [{"ellipsis": True}]},
                    operands=[operand(spec, lay), plain(np.array(vv))], layout=lay, mutate=0,
                    call=lambda a, v: _setitem(a, Ellipsis, v))
            add("mutation", "setitem_array", params={"index": [{"int": 1}]},
                operands=[operand(spec, lay), plain(np.array(["xyz", "w"] if spec[1] == "U" else [b"xyz", b"w"]))],
                layout=lay, mutate=0, call=lambda a, v: _setitem(a, 1, v))
            add("mutation", "setitem_array", params={"index": [{"arr": [2, 0]}]},
                operands=[operand(spec, lay), plain(np.array([["m"] * 4, ["n"] * 4]))],
                layout=lay, mutate=0, call=lambda a, v: _setitem(a, np.array([2, 0]), v))
        add("mutation", "setitem_array", params={"index": [{"int": 0}]},
            operands=[operand(spec, "c1"), plain(np.array(["a", "b"]))], layout="c1", mutate=0,
            call=lambda a, v: _setitem(a, 0, v))
        add("mutation", "setitem_array", params={"index": [{"mask": [True, False, True, False, False, True]}]},
            operands=[operand(spec, "c1"), plain(np.array(["u", "v"]))], layout="c1", mutate=0,
            call=lambda a, v: _setitem(a, np.array([True, False, True, False, False, True]), v))
        add("mutation", "setitem_array", params={"index": [{"ellipsis": True}]},
            operands=[operand(spec, "c1"), plain(np.arange(6))], layout="c1", mutate=0,
            call=lambda a, v: _setitem(a, Ellipsis, v))

        # put / place / putmask / copyto / put_along_axis.
        for lay in ("c1", "c2", "f2", "s1", "r2"):
            vals = ["P", "QQ", "RRRR"] if spec[1] == "U" else [b"P", b"QQ", b"RRRR"]
            for mode in ("raise", "wrap", "clip"):
                add("mutation", "put", params={"indices": [0, 4, -1, 9], "values": enc(vals), "mode": mode},
                    operands=[operand(spec, lay)], layout=lay, mutate=0,
                    call=lambda a, v=vals, m=mode: np.put(a, [0, 4, -1, 9], v, mode=m))
            mask = (np.arange(operand(spec, lay)[1].size) % 3 == 0).reshape(operand(spec, lay)[1].shape)
            add("mutation", "place", params={"mask": mask.tolist(), "values": enc(vals)},
                operands=[operand(spec, lay)], layout=lay, mutate=0,
                call=lambda a, m=mask, v=vals: np.place(a, m, v))
            add("mutation", "putmask", params={"mask": mask.tolist(), "values": enc(vals)},
                operands=[operand(spec, lay)], layout=lay, mutate=0,
                call=lambda a, m=mask, v=vals: np.putmask(a, m, v))
        for src_spec, src_lay in (("<U5", "c2"), ("<U1", "f2"), ("|S5", "r2"), ("|S2", "bc"), ("<U5", "c1")):
            for casting in ("no", "same_kind", "unsafe"):
                for where in (None, [[True, False, True, False], [False, True, False, True], [True] * 4]):
                    add("mutation", "copyto", params={"casting": casting, "where": where},
                        operands=[operand(spec, "c2"), operand(src_spec, src_lay, seed=6, ascii_only=True)],
                        layout=f"c2<-{src_lay}", mutate=0,
                        call=lambda d, s, c=casting, w=where: np.copyto(d, s, casting=c,
                                                                       where=True if w is None else np.array(w)))
        add("mutation", "copyto", params={"casting": "unsafe", "where": None},
            operands=[operand(spec, "c1"), plain(np.array(["é"] * 6))], layout="c1", mutate=0,
            call=lambda d, s: np.copyto(d, s, casting="unsafe"))
        add("mutation", "copyto", params={"casting": "unsafe", "where": None},
            operands=[operand(spec, "c1"), plain(np.arange(6))], layout="c1", mutate=0,
            call=lambda d, s: np.copyto(d, s, casting="unsafe"))
        add("mutation", "put_along_axis", params={"indices": [[1], [0], [3]], "axis": 1},
            operands=[operand(spec, "c2"), plain(np.array([["v"], ["w"], ["x"]] if spec[1] == "U" else
                                                          [[b"v"], [b"w"], [b"x"]]))], layout="c2", mutate=0,
            call=lambda a, v: np.put_along_axis(a, np.array([[1], [0], [3]]), v, axis=1))


def _to_bytes(v):
    if isinstance(v, list):
        return [_to_bytes(x) for x in v]
    return v.encode("ascii")


# ----------------------------------------------------------------------------------------------------------------------
# G. Element access and conversion to Python
# ----------------------------------------------------------------------------------------------------------------------

def gen_access():
    for spec in ("<U3", "|S3", "<U5"):
        for lay in CORE_LAYOUTS + ["c3", "r1", "rs", "e1"]:
            op = operand(spec, lay)
            add("access", "tolist", operands=[op], layout=lay, call=lambda a: a.tolist())
            if op[1].size:
                add("access", "item", params={"index": 0}, operands=[op], layout=lay, call=lambda a: a.item(0))
                add("access", "item", params={"index": op[1].size - 1}, operands=[op], layout=lay,
                    call=lambda a: a.item(a.size - 1))
            if op[1].size == 1:
                add("access", "item", params={"index": None}, operands=[op], layout=lay, call=lambda a: a.item())
            if op[1].ndim == 1:
                add("access", "iter", operands=[op], layout=lay, call=lambda a: [x for x in a])
    add("access", "item", params={"index": None}, operands=[operand("<U3", "c1")], layout="c1",
        call=lambda a: a.item())
    add("access", "item", params={"index": 99}, operands=[operand("<U3", "c1")], layout="c1",
        call=lambda a: a.item(99))
    add("access", "tolist", operands=[plain(np.array([0x110000], dtype="<u4").view("<U1"))], layout="c1",
        call=lambda a: a.tolist())


# ----------------------------------------------------------------------------------------------------------------------
# H. Printing
# ----------------------------------------------------------------------------------------------------------------------

PRINT_U = ["", "a", "it's", 'say "hi"', "both'\"", "\t\n\r", "\x00a", "\x01\x1f\x7f", "\x80\x9f", "\xa0", "\xe9",
           chr(0x4E2D), chr(0x1F600), chr(0xD800), chr(0xDFFF), chr(0x2028), chr(0x2029), chr(0xE000), chr(0xF0000),
           chr(0x0378), chr(0xFEFF), chr(0x200B), chr(0x00AD), "e" + chr(0x0301), chr(0x3000), "\\", "' \"",
           "x" * 30, "tab\there"]
PRINT_S = [b"", b"a", b"it's", b'say "hi"', b"both'\"", b"\t\n\r", b"\x00a", b"\x01\x7f", b"\x80\xff", b"\\",
           b"x" * 30, b"' \""]


def gen_printing():
    for pool, cls in ((PRINT_U, "U"), (PRINT_S, "S")):
        for v in pool:
            arr = np.array([v], dtype=f"{cls}{max(len(v), 1)}")
            for fn in ("repr", "str"):
                add("printing", fn, operands=[plain(arr)], layout="c1", call=lambda a, f=fn: repr(a) if f == "repr" else str(a))
            z = plain(np.array(v, dtype=f"{cls}{max(len(v), 1)}"))
            add("printing", "repr", operands=[z], layout="z0", call=lambda a: repr(a))
            add("printing", "str", operands=[z], layout="z0", call=lambda a: str(a))
        width = max(len(v) for v in pool)
        mixed = np.array(pool, dtype=f"{cls}{width}")
        for fn in ("repr", "str"):
            add("printing", fn, operands=[plain(mixed)], layout="c1", call=lambda a, f=fn: repr(a) if f == "repr" else str(a))
    for spec in ("<U3", "|S3", "<U1", "<U5"):
        for lay in CORE_LAYOUTS + ["c3", "f3", "nx", "d5", "e1", "e3", "rs"]:
            op = operand(spec, lay)
            add("printing", "repr", operands=[op], layout=lay, call=lambda a: repr(a))
            add("printing", "str", operands=[op], layout=lay, call=lambda a: str(a))
    long_u = plain(np.array([f"item{i}" for i in range(40)]))
    long_s = plain(np.array([f"b{i}".encode() for i in range(40)]))
    for op in (long_u, long_s):
        for th, ed, lw in ((10, 2, 75), (5, 1, 30), (1000, 3, 20), (2, 2, 200)):
            add("printing", "repr_opts", params={"threshold": th, "edgeitems": ed, "linewidth": lw}, operands=[op],
                layout="c1", call=lambda a, t=th, e=ed, l=lw: _with_opts(repr, a, t, e, l))
            add("printing", "str_opts", params={"threshold": th, "edgeitems": ed, "linewidth": lw}, operands=[op],
                layout="c1", call=lambda a, t=th, e=ed, l=lw: _with_opts(str, a, t, e, l))
        for lw in (20, 40):
            add("printing", "array2string", params={"max_line_width": lw, "separator": ", ", "prefix": "x("},
                operands=[op], layout="c1",
                call=lambda a, l=lw: np.array2string(a, max_line_width=l, separator=", ", prefix="x("))
            add("printing", "array_repr", params={"max_line_width": lw}, operands=[op], layout="c1",
                call=lambda a, l=lw: np.array_repr(a, max_line_width=l))
            add("printing", "array_str", params={"max_line_width": lw}, operands=[op], layout="c1",
                call=lambda a, l=lw: np.array_str(a, max_line_width=l))
    big = plain(np.array([["r%dc%d" % (i, j) for j in range(12)] for i in range(12)]))
    for th in (10, 1000):
        add("printing", "repr_opts", params={"threshold": th, "edgeitems": 2, "linewidth": 75}, operands=[big],
            layout="c2", call=lambda a, t=th: _with_opts(repr, a, t, 2, 75))
    wide = plain(np.array(["\U0001F600" * 6, "中" * 7, "ab"] * 4))
    add("printing", "repr_opts", params={"threshold": 1000, "edgeitems": 3, "linewidth": 40}, operands=[wide],
        layout="c1", call=lambda a: _with_opts(repr, a, 1000, 3, 40))


def _with_opts(fn, a, threshold, edgeitems, linewidth):
    with np.printoptions(threshold=threshold, edgeitems=edgeitems, linewidth=linewidth):
        return fn(a)


# ----------------------------------------------------------------------------------------------------------------------
# I. What NumPy refuses for text (phase 1: the error is final) and what it computes (phase 2)
# ----------------------------------------------------------------------------------------------------------------------

BINARY_UFUNCS = ["add", "subtract", "multiply", "divide", "true_divide", "floor_divide", "power", "float_power",
                 "remainder", "mod", "fmod", "divmod", "maximum", "minimum", "fmax", "fmin", "arctan2", "hypot",
                 "copysign", "nextafter", "logaddexp", "logaddexp2", "heaviside", "ldexp", "gcd", "lcm",
                 "bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift", "matmul", "equal",
                 "not_equal", "less", "less_equal", "greater", "greater_equal", "logical_and", "logical_or",
                 "logical_xor"]
UNARY_UFUNCS = ["negative", "positive", "absolute", "fabs", "sign", "sqrt", "cbrt", "square", "reciprocal", "exp",
                "exp2", "expm1", "log", "log2", "log10", "log1p", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
                "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "deg2rad", "rad2deg", "degrees", "radians",
                "floor", "ceil", "trunc", "rint", "isnan", "isinf", "isfinite", "signbit", "logical_not", "invert",
                "bitwise_not", "spacing", "modf", "frexp", "conjugate"]
COMPUTES = {"equal", "not_equal", "less", "less_equal", "greater", "greater_equal", "logical_and", "logical_or",
            "logical_xor", "logical_not"}


def _ufunc_phase(name, kinds):
    """Whether NumPy may compute this ufunc for these operand kinds ("compute") or must refuse it (phase 1)."""
    if name in COMPUTES:
        return "compute"
    if name == "add" and kinds in (("U", "U"), ("S", "S")):
        return "compute"
    return 1


def gen_refusals():
    u = operand("<U3", "c1", ascii_only=True)
    u2 = operand("<U2", "c1", seed=2, ascii_only=True)
    s = operand("|S3", "c1", ascii_only=True)
    s2 = operand("|S2", "c1", seed=2, ascii_only=True)
    i8 = num_operand(np.arange(6))
    f8 = num_operand(np.linspace(0, 1, 6))
    b1 = num_operand(np.array([True, False] * 3))
    u0 = plain(np.array("ab"))
    for name in BINARY_UFUNCS:
        fn = getattr(np, name)
        for (ka, a), (kb, b) in ((("U", u), ("U", u2)), (("S", s), ("S", s2)), (("U", u), ("S", s)),
                                 (("U", u), ("i", i8)), (("i", i8), ("U", u)), (("S", s), ("f", f8)),
                                 (("U", u), ("b", b1)), (("U", u0), ("i", i8))):
            add("ufuncs", "ufunc", params={"name": name}, operands=[a, b], layout=f"{ka}{kb}",
                phase=_ufunc_phase(name, (ka, kb)), call=lambda x, y, f=fn: f(x, y))
    for name in UNARY_UFUNCS:
        fn = getattr(np, name)
        for k, a in (("U", u), ("S", s)):
            add("ufuncs", "ufunc", params={"name": name}, operands=[a], layout=k, phase=_ufunc_phase(name, (k,)),
                call=lambda x, f=fn: f(x))
    # dtype= and out= of the string ufuncs.
    for name, dt in (("subtract", "f8"), ("add", "f8"), ("add", "U6"), ("maximum", "U3"), ("negative", "f8"),
                     ("sqrt", "f8"), ("equal", "?"), ("equal", "i8"), ("less", "f8")):
        fn = getattr(np, name)
        ops = [u] if name in ("negative", "sqrt") else [u, u2]
        add("ufuncs", "ufunc_dtype", params={"name": name, "dtype": dt}, operands=ops, layout="U",
            phase=_ufunc_phase(name, ("U",) * len(ops)), call=lambda *xs, f=fn, d=dt: f(*xs, dtype=d))
    add("ufuncs", "ufunc", params={"name": "multiply"}, operands=[u, num_operand(np.array([2]))], layout="Ui",
        call=lambda x, y: np.multiply(x, y))
    # The comparison operators: no loop for text against numbers answers 'never equal' (== / !=) or raises (<).
    for op_name in ("==", "!=", "<"):
        for (ka, a), (kb, b) in ((("U", u), ("i", i8)), (("U", u), ("U", u2)), (("S", s), ("S", s2)),
                                 (("U", u), ("S", s)), (("i", i8), ("S", s))):
            # Text against text compares elementwise (the computational phase); text against numbers has no loop, so
            # == / != answer 'never equal' and < raises -- final answers NumSharp gives now.
            phase = "compute" if ka in "US" and kb in "US" else 1
            add("ufuncs", "operator", params={"op": op_name}, operands=[a, b], layout=f"{ka}{kb}", phase=phase,
                call=lambda x, y, o=op_name: {"==": lambda: x == y, "!=": lambda: x != y, "<": lambda: x < y}[o]())

    reductions = ["sum", "prod", "mean", "std", "var", "max", "min", "ptp", "cumsum", "cumprod", "median", "average",
                  "nansum", "nanprod", "nanmean", "nanstd", "nanvar", "nanmax", "nanmin", "nanmedian", "nancumsum",
                  "nancumprod", "percentile", "quantile", "nanpercentile", "nanquantile", "argmax", "argmin", "all",
                  "any", "count_nonzero", "nonzero", "flatnonzero", "argwhere", "sort", "argsort", "unique",
                  "trim_zeros", "diff", "ediff1d", "gradient", "trace", "cov", "corrcoef", "sinc", "angle", "i0",
                  "unwrap", "vander", "round", "clip", "isreal", "iscomplex", "real", "imag", "nan_to_num",
                  "real_if_close", "fix", "around", "histogram", "lexsort", "searchsorted", "partition",
                  "argpartition", "isin", "union1d", "intersect1d", "setdiff1d", "setxor1d", "array_equal",
                  "array_equiv", "isclose", "allclose", "linalg.norm", "fft.fft", "nanargmax", "nanargmin",
                  "logical_not", "bincount", "digitize", "interp", "polyval", "trapezoid", "convolve", "correlate",
                  "dot", "inner", "vdot", "outer", "kron", "tensordot", "cross", "einsum", "byteswap", "conj"]
    # NumPy computes these on text: ordering, truth, searching, set logic, and the functions that parse the text as
    # numbers (i0 and linalg.norm cast to float first: phase 1 since P1.1, they parse the text).
    computes = {"argmax", "argmin", "nanargmax", "nanargmin", "all", "any", "count_nonzero", "nonzero", "flatnonzero",
                "argwhere", "sort", "argsort", "unique", "trim_zeros", "lexsort", "searchsorted", "partition",
                "argpartition", "isin", "union1d", "intersect1d", "setdiff1d", "setxor1d", "array_equal", "array_equiv",
                "logical_not"}
    parses = {"i0", "linalg.norm"}
    for name in reductions:
        for k, a in (("U", u), ("S", s)):
            phase = 1 if name in parses else "compute" if name in computes else 1
            add("reductions", "call", params={"name": name}, operands=[a], layout=k, phase=phase,
                call=lambda x, n=name: _call_named(n, x))


def _call_named(name, x):
    """The single-array calls of the refusal table, spelled as NumPy code would spell them."""
    other = x.copy()
    table = {
        "percentile": lambda: np.percentile(x, 50), "quantile": lambda: np.quantile(x, 0.5),
        "nanpercentile": lambda: np.nanpercentile(x, 50), "nanquantile": lambda: np.nanquantile(x, 0.5),
        "round": lambda: np.round(x), "around": lambda: np.around(x), "clip": lambda: np.clip(x, x[0], x[1]),
        "histogram": lambda: np.histogram(x), "lexsort": lambda: np.lexsort([x]),
        "searchsorted": lambda: np.searchsorted(np.sort(x), x[0]), "partition": lambda: np.partition(x, 1),
        "argpartition": lambda: np.argpartition(x, 1), "isin": lambda: np.isin(x, other[:2]),
        "union1d": lambda: np.union1d(x, other), "intersect1d": lambda: np.intersect1d(x, other),
        "setdiff1d": lambda: np.setdiff1d(x, other[:2]), "setxor1d": lambda: np.setxor1d(x, other[:2]),
        "array_equal": lambda: np.array_equal(x, other), "array_equiv": lambda: np.array_equiv(x, other),
        "isclose": lambda: np.isclose(x, other), "allclose": lambda: np.allclose(x, other),
        "linalg.norm": lambda: np.linalg.norm(x), "fft.fft": lambda: np.fft.fft(x),
        "digitize": lambda: np.digitize(np.arange(3.0), x), "interp": lambda: np.interp(np.arange(3.0), x, x),
        "polyval": lambda: np.polyval(x, 2.0), "trapezoid": lambda: np.trapezoid(x),
        "convolve": lambda: np.convolve(x, other), "correlate": lambda: np.correlate(x, other),
        "dot": lambda: np.dot(x, other), "inner": lambda: np.inner(x, other), "vdot": lambda: np.vdot(x, other),
        "outer": lambda: np.outer(x, other), "kron": lambda: np.kron(x, other),
        "tensordot": lambda: np.tensordot(x, other, 1), "cross": lambda: np.cross(x[:3], other[:3]),
        "einsum": lambda: np.einsum("i,i->i", x, other), "byteswap": lambda: x.byteswap(),
        "conj": lambda: np.conj(x), "vander": lambda: np.vander(x, 2), "bincount": lambda: np.bincount(x),
        "diff": lambda: np.diff(x), "ediff1d": lambda: np.ediff1d(x), "gradient": lambda: np.gradient(x),
        "trace": lambda: np.trace(np.reshape(x, (2, 3))), "cov": lambda: np.cov(x), "corrcoef": lambda: np.corrcoef(x),
        "unwrap": lambda: np.unwrap(x), "real_if_close": lambda: np.real_if_close(x),
    }
    if name in table:
        return table[name]()
    return getattr(np, name)(x)


# ----------------------------------------------------------------------------------------------------------------------
# J. Unsized string dtype= requests, and the order in which NumPy reports the problems of a call that involves text
# ----------------------------------------------------------------------------------------------------------------------

# Every numeric dtype NumSharp shares with NumPy, as a six-element operand (the string operands are six long too).
NUM_KINDS = {
    "b": np.array([True, False, True, True, False, True]), "i1": np.arange(1, 7, dtype=np.int8),
    "u1": np.arange(1, 7, dtype=np.uint8), "i2": np.arange(1, 7, dtype=np.int16),
    "u2": np.arange(1, 7, dtype=np.uint16), "i4": np.arange(1, 7, dtype=np.int32),
    "u4": np.arange(1, 7, dtype=np.uint32), "i8": np.arange(1, 7, dtype=np.int64),
    "u8": np.arange(1, 7, dtype=np.uint64), "f2": np.linspace(0.5, 3, 6).astype(np.float16),
    "f4": np.linspace(0.5, 3, 6).astype(np.float32), "f8": np.linspace(0.5, 3, 6),
    "c16": np.arange(1, 7) + 2j,
}
DTYPE_UNARY = ["isfinite", "isinf", "isnan", "signbit", "absolute", "arccos", "arccosh", "arcsin", "arcsinh", "arctan",
               "arctanh", "bitwise_count", "cbrt", "ceil", "conjugate", "cos", "cosh", "deg2rad", "exp", "exp2", "expm1",
               "fabs", "floor", "invert", "log", "log10", "log1p", "log2", "negative", "positive", "rad2deg",
               "reciprocal", "rint", "sign", "sin", "sinh", "spacing", "sqrt", "square", "tan", "tanh", "trunc",
               "radians", "degrees", "logical_not", "modf"]
DTYPE_BINARY = ["add", "arctan2", "divide", "divmod", "float_power", "floor_divide", "fmod", "gcd", "lcm", "logaddexp",
                "logaddexp2", "nextafter", "copysign", "hypot", "heaviside", "maximum", "minimum", "fmax", "fmin",
                "remainder", "multiply", "power", "left_shift", "right_shift", "subtract", "bitwise_and", "bitwise_or",
                "bitwise_xor", "equal", "not_equal", "less", "less_equal", "greater", "greater_equal", "logical_and",
                "logical_or", "logical_xor", "true_divide"]


def _kinds():
    """The operand kinds of section J: the two string classes and every numeric dtype (fresh (base, view) pairs)."""
    kinds = {"u": operand("<U3", "c1", ascii_only=True), "s": operand("|S3", "c1", ascii_only=True)}
    for k, v in NUM_KINDS.items():
        kinds[k] = num_operand(v)
    return kinds


def gen_string_requests():
    """A ufunc / reduction / gufunc called with an unsized string dtype= ('U', 'S'): NumPy's string loops (add formats
    numbers as text and concatenates) or its refusal, which names the requested CLASS (probed against 2.4.2)."""
    k = _kinds()
    pairs = [(x, x) for x in k] + [("u", "s"), ("s", "u"), ("u", "f8"), ("f8", "u"), ("s", "i8"), ("i8", "s"),
                                   ("f8", "i8"), ("i4", "f4"), ("b", "i1"), ("u", "i8"), ("i8", "u"), ("u", "i1"),
                                   ("u1", "s"), ("u", "b")]
    # The reductions: a string request reads a number as text of its REQUIRED_STR_LEN (float64 is '<U32'); a sized or
    # byte-swapped request is the class-only TypeError where NumPy runs _get_dtype (ufunc.reduce, np.mean's umr_sum)
    # and is taken as its class where it does not (the accumulations, np.trace).
    reduce_calls = {
        "sum": lambda a, d, ax: np.sum(a, axis=ax, dtype=d), "prod": lambda a, d, ax: np.prod(a, axis=ax, dtype=d),
        "mean": lambda a, d, ax: np.mean(a, axis=0 if ax is None else ax, dtype=d),
        "var": lambda a, d, ax: np.var(a, axis=ax, dtype=d), "std": lambda a, d, ax: np.std(a, axis=ax, dtype=d),
        "nanmean": lambda a, d, ax: np.nanmean(a, axis=ax, dtype=d),
        "nanvar": lambda a, d, ax: np.nanvar(a, axis=ax, dtype=d),
        "nanstd": lambda a, d, ax: np.nanstd(a, axis=ax, dtype=d),
        "cumsum": lambda a, d, ax: np.cumsum(a, axis=ax, dtype=d),
        "cumprod": lambda a, d, ax: np.cumprod(a, axis=ax, dtype=d),
        "nancumsum": lambda a, d, ax: np.nancumsum(a, axis=ax, dtype=d),
        "nancumprod": lambda a, d, ax: np.nancumprod(a, axis=ax, dtype=d),
        "maximum.reduce": lambda a, d, ax: np.maximum.reduce(a, axis=ax, dtype=d),
        "minimum.reduce": lambda a, d, ax: np.minimum.reduce(a, axis=ax, dtype=d),
    }
    for dt in ("U", "S", "U3", "S3", ">f8", "f8"):
        for name, fn in reduce_calls.items():
            for key, op in k.items():
                # A sized string request is the class-only TypeError for every input; a numeric request matters only
                # over text (a numeric reduction of numbers is the numeric corpus's business).
                if dt in ("U3", "S3") and key not in ("u", "s", "f8", "i8", "b"):
                    continue
                if dt in (">f8", "f8") and key not in ("u", "s"):
                    continue
                for ax in ((None, 9) if dt in ("U", "S") else (None,)):
                    add("strdtype", "reduce", params={"name": name, "dtype": dt, "axis": ax}, operands=[op],
                        layout=key, phase="compute", call=lambda a, f=fn, d=dt, x=ax: f(a, d, x))
    for dt in ("U", "S", "U3", ">f8", "f8"):
        for key, op in k.items():
            if dt in (">f8", "f8") and key not in ("u", "s"):
                continue
            add("strdtype", "trace", params={"dtype": dt}, operands=[op], layout=key, phase="compute",
                call=lambda a, d=dt: np.trace(np.stack([a, a]), dtype=d))
    # 0-d operands: ufunc.reduce reads axis 0 / -1 as every axis; np.mean's family refuses any axis.
    zero_d = {"u0": plain(np.array("ab")), "f0": plain(np.array(1.5)), "i0": plain(np.array(3))}
    for dt in ("U", "S", "U3"):
        for name in ("sum", "prod", "mean", "var", "cumsum", "cumprod", "maximum.reduce"):
            for key, op in zero_d.items():
                for ax in (0, -1, 9, None):
                    fn = reduce_calls[name]
                    if name == "mean" and ax is None:
                        continue
                    add("strdtype", "reduce", params={"name": name, "dtype": dt, "axis": ax}, operands=[op],
                        layout=key, phase="compute",
                        call=lambda a, f=fn, d=dt, x=ax, n=name: np.mean(a, axis=x, dtype=d) if n == "mean" else f(a, d, x))

    # The ufuncs run AFTER the reductions: NumPy caches a promoter's resolution per DType tuple
    # (ufunc._dispatch_cache), so np.multiply(u, i1, dtype='U') -- the string multiply promoter casting int8 to int64 --
    # would make a later np.prod(i1, dtype='U') find that loop ('out' kwarg text) where a fresh process names no loop.
    # The corpus records the fresh-process answer, which is NumSharp's.
    for dt in ("U", "S"):
        for name in DTYPE_UNARY:
            fn = getattr(np, name)
            for key, op in k.items():
                add("strdtype", "ufunc_dtype", params={"name": name, "dtype": dt}, operands=[op], layout=key,
                    phase="compute", call=lambda x, f=fn, d=dt: f(x, dtype=d))
        for name in DTYPE_BINARY:
            fn = getattr(np, name)
            for k1, k2 in pairs:
                add("strdtype", "ufunc_dtype", params={"name": name, "dtype": dt}, operands=[k[k1], k[k2]],
                    layout=f"{k1}{k2}", phase="compute", call=lambda x, y, f=fn, d=dt: f(x, y, dtype=d))

    # The gufuncs resolve the loop before any core-dimension / axis / keepdims check.
    two_d = {key: num_operand(np.stack([v, v])) for key, v in NUM_KINDS.items() if key in ("f8", "i8", "b")}
    two_d["u"] = plain(np.array([["ab", "c"], ["d", "e"]]))
    one_d = {key: num_operand(v[:2]) for key, v in NUM_KINDS.items() if key in ("f8", "i8", "b")}
    one_d["u"] = plain(np.array(["ab", "c"]))
    for dt in ("U", "S", "U3", None):
        for key in two_d:
            # Without a request only text operands are string cases.
            if dt is None and key != "u":
                continue
            for name, args in (("matmul", (two_d[key], two_d[key])), ("vecdot", (one_d[key], one_d[key])),
                               ("matvec", (two_d[key], one_d[key])), ("vecmat", (one_d[key], two_d[key]))):
                fn = getattr(np, name)
                add("strdtype", "gufunc", params={"name": name, "dtype": dt}, operands=list(args), layout=key,
                    phase="compute", call=lambda x, y, f=fn, d=dt: f(x, y, dtype=d))
            # A shape the core dimensions refuse, and an axis matvec refuses: the loop answer comes first.
            add("strdtype", "gufunc", params={"name": "matmul", "dtype": dt}, operands=[one_d[key], num_operand(np.ones(3))],
                layout=key + "-core", phase="compute", call=lambda x, y, d=dt: np.matmul(x, y, dtype=d))
            add("strdtype", "gufunc", params={"name": "matvec", "dtype": dt, "axis": 0},
                operands=[two_d[key], one_d[key]], layout=key + "-axis", phase="compute",
                call=lambda x, y, d=dt: np.matvec(x, y, axis=0, dtype=d))

    # clip routes by its bounds (positive / minimum / maximum / clip); a string request names the operand classes.
    for dt in ("U", "S"):
        for key, op in k.items():
            add("strdtype", "clip", params={"bounds": "both", "dtype": dt}, operands=[op, op, op], layout=key,
                phase="compute", call=lambda a, lo, hi, d=dt: np.clip(a, lo, hi, dtype=d))
            add("strdtype", "clip", params={"bounds": "min", "dtype": dt}, operands=[op, op], layout=key,
                phase="compute", call=lambda a, lo, d=dt: np.clip(a, lo, None, dtype=d))
            add("strdtype", "clip", params={"bounds": "max", "dtype": dt}, operands=[op, op], layout=key,
                phase="compute", call=lambda a, hi, d=dt: np.clip(a, None, hi, dtype=d))
            add("strdtype", "clip", params={"bounds": "none", "dtype": dt}, operands=[op], layout=key,
                phase="compute", call=lambda a, d=dt: np.clip(a, None, None, dtype=d))

    # einsum: the operand casts under casting=, then the bool-zero output fill, then the missing string loop.
    for dt in ("U", "S", "U1", "U5", "U32", "S5", None):
        for key in ("u", "s", "f8", "b", "i8"):
            # Without a request only text operands are string cases.
            if dt is None and key not in ("u", "s"):
                continue
            for casting in ("safe", "unsafe", "no"):
                op = k[key]
                add("strdtype", "einsum", params={"subscripts": "i,i", "dtype": dt, "casting": casting},
                    operands=[op, op], layout=key, phase="compute",
                    call=lambda x, y, d=dt, c=casting: np.einsum("i,i", x, y, dtype=d, casting=c))
        if dt is not None:
            add("strdtype", "einsum", params={"subscripts": "i->", "dtype": dt, "casting": "safe"},
                operands=[k["f8"]], layout="f8", phase="compute", call=lambda x, d=dt: np.einsum("i->", x, dtype=d))
    wide = plain(np.array(["abcde", "c"]))
    add("strdtype", "einsum", params={"subscripts": "i,i", "dtype": None, "casting": "safe"}, operands=[wide, wide],
        layout="U5", phase="compute", call=lambda x, y: np.einsum("i,i", x, y))
    add("strdtype", "einsum", params={"subscripts": "i", "dtype": "U", "casting": "safe"}, operands=[k["u"]],
        layout="view", phase="compute", call=lambda x: np.einsum("i", x, dtype="U"))
    add("strdtype", "einsum", params={"subscripts": "ij->ji", "dtype": None, "casting": "safe"},
        operands=[two_d["u"]], layout="view", phase="compute", call=lambda x: np.einsum("ij->ji", x))


def gen_text_precedence():
    """The order NumPy reports a text call's problems in: a read-only out= and a non-bool where= before the loop answer,
    a computing loop's output cast and broadcast after it, every reduction's axis before its answer."""
    u = operand("<U3", "c1", ascii_only=True)
    s = operand("|S3", "c1", ascii_only=True)
    u2 = operand("<U2", "c1", seed=3, ascii_only=True)
    i8 = num_operand(np.arange(6))
    f8 = num_operand(np.linspace(0, 1, 6))
    # The out= operands go through ``target``: NumPy returns a computing call's out itself, so its OWNDATA is compared, and
    # the replay rebuilds a 1-D string operand as its own base (a ``plain`` 1-D view would record OWNDATA False).
    outs = {"ro": (lambda: target(np.zeros(6)), True), "f8": (lambda: target(np.zeros(6)), False),
            "i8": (lambda: target(np.zeros(6, np.int64)), False), "b": (lambda: target(np.zeros(6, bool)), False),
            "U1": (lambda: target(np.zeros(6, "<U1")), False), "U9": (lambda: target(np.zeros(6, "<U9")), False),
            "S9": (lambda: target(np.zeros(6, "S9")), False), "f8x7": (lambda: target(np.zeros(7)), False),
            "U9x23": (lambda: target(np.zeros((2, 3), "<U9")), False)}
    wheres = {"int": num_operand(np.array([1, 0, 1, 0, 1, 0])), "b7": num_operand(np.ones(7, bool))}

    def ro_view(pair):
        base, view = pair
        view.flags.writeable = False
        return base, view

    calls = [("add", [u, u2]), ("add", [s, s]), ("equal", [u, u2]), ("less", [s, s]), ("logical_and", [u, i8]),
             ("multiply", [u, i8]), ("multiply", [i8, s]), ("subtract", [u, u2]), ("sqrt", [u]), ("negative", [s]),
             ("logical_not", [u]), ("add", [f8, f8, "U"]), ("sqrt", [f8, "U"]), ("isnan", [u])]
    for name, spec in calls:
        dt = spec[-1] if isinstance(spec[-1], str) else None
        ins = [x for x in spec if not isinstance(x, str)]
        fn = getattr(np, name)
        for oname, (make, readonly) in outs.items():
            out = make()
            if readonly:
                out = ro_view(out)
            add("strprec", "ufunc_kw", params={"name": name, "dtype": dt, "nin": len(ins), "out": len(ins)},
                operands=ins + [out], layout=oname, phase="compute",
                call=lambda *xs, f=fn, n=len(ins), d=dt: f(*xs[:n], out=xs[n], dtype=d))
        for wname, w in wheres.items():
            add("strprec", "ufunc_kw", params={"name": name, "dtype": dt, "nin": len(ins), "where": len(ins)},
                operands=ins + [w], layout="where-" + wname, phase="compute",
                call=lambda *xs, f=fn, n=len(ins), d=dt: f(*xs[:n], where=xs[n], dtype=d))
        # A read-only out AND a non-bool mask: the out wins.
        out = ro_view(outs["f8"][0]())
        add("strprec", "ufunc_kw", params={"name": name, "dtype": dt, "nin": len(ins), "out": len(ins),
                                           "where": len(ins) + 1},
            operands=ins + [out, wheres["int"]], layout="ro+where", phase="compute",
            call=lambda *xs, f=fn, n=len(ins), d=dt: f(*xs[:n], out=xs[n], where=xs[n + 1], dtype=d))
    # Inputs that do not broadcast: a computing loop broadcasts after its loop answer.
    u7 = operand("<U2", "c1", seed=1, ascii_only=True)
    u7 = (u7[0], u7[1][:4])
    for name in ("add", "equal", "logical_and", "subtract", "multiply"):
        fn = getattr(np, name)
        add("strprec", "ufunc_kw", params={"name": name, "dtype": None, "nin": 2}, operands=[u, u7], layout="bcast",
            phase="compute", call=lambda x, y, f=fn: f(x, y))
    # divmod / modf parse both outputs before the loop.
    for which in (0, 1):
        o1, o2 = plain(np.zeros(6)), plain(np.zeros(6))
        read_only = o1 if which == 0 else o2
        read_only[1].flags.writeable = False
        add("strprec", "divmod_out", params={"readonly": which}, operands=[u, u, o1, o2], layout=f"ro{which}",
            phase="compute", call=lambda x, y, q, r: np.divmod(x, y, out=(q, r)))

    # The reductions of text: every axis rule, before the refusal or the next phase.
    a1, a0, a2 = plain(np.array(["ab", "c", "de"])), plain(np.array("ab")), plain(np.array([["ab", "c"], ["d", "e"]]))
    s1 = plain(np.array([b"ab", b"c", b"de"]))
    e1, e2 = plain(np.array([], "<U2")), plain(np.empty((0, 2), "<U2"))
    reducers = {
        "sum": lambda a, ax: np.sum(a, axis=ax), "prod": lambda a, ax: np.prod(a, axis=ax),
        "mean": lambda a, ax: np.mean(a, axis=ax), "var": lambda a, ax: np.var(a, axis=ax),
        "std": lambda a, ax: np.std(a, axis=ax), "nansum": lambda a, ax: np.nansum(a, axis=ax),
        "nanprod": lambda a, ax: np.nanprod(a, axis=ax), "nanmean": lambda a, ax: np.nanmean(a, axis=ax),
        "nanvar": lambda a, ax: np.nanvar(a, axis=ax), "nanstd": lambda a, ax: np.nanstd(a, axis=ax),
        "cumsum": lambda a, ax: np.cumsum(a, axis=ax), "cumprod": lambda a, ax: np.cumprod(a, axis=ax),
        "nancumsum": lambda a, ax: np.nancumsum(a, axis=ax), "nancumprod": lambda a, ax: np.nancumprod(a, axis=ax),
        "max": lambda a, ax: np.max(a, axis=ax), "min": lambda a, ax: np.min(a, axis=ax),
        "amax": lambda a, ax: np.amax(a, axis=ax), "amin": lambda a, ax: np.amin(a, axis=ax),
        "nanmax": lambda a, ax: np.nanmax(a, axis=ax), "nanmin": lambda a, ax: np.nanmin(a, axis=ax),
        "ptp": lambda a, ax: np.ptp(a, axis=ax), "argmax": lambda a, ax: np.argmax(a, axis=ax),
        "argmin": lambda a, ax: np.argmin(a, axis=ax), "nanargmax": lambda a, ax: np.nanargmax(a, axis=ax),
        "nanargmin": lambda a, ax: np.nanargmin(a, axis=ax), "median": lambda a, ax: np.median(a, axis=ax),
        "nanmedian": lambda a, ax: np.nanmedian(a, axis=ax), "average": lambda a, ax: np.average(a, axis=ax),
        "any": lambda a, ax: np.any(a, axis=ax), "all": lambda a, ax: np.all(a, axis=ax),
        "count_nonzero": lambda a, ax: np.count_nonzero(a, axis=ax), "sort": lambda a, ax: np.sort(a, axis=ax),
        "argsort": lambda a, ax: np.argsort(a, axis=ax),
        "percentile": lambda a, ax: np.percentile(a, 50, axis=ax),
        "quantile": lambda a, ax: np.quantile(a, 0.5, axis=ax),
        "nanpercentile": lambda a, ax: np.nanpercentile(a, 50, axis=ax),
        "nanquantile": lambda a, ax: np.nanquantile(a, 0.5, axis=ax),
        "partition": lambda a, ax: np.partition(a, 0, axis=ax),
        "argpartition": lambda a, ax: np.argpartition(a, 0, axis=ax),
    }
    for name, fn in reducers.items():
        for key, op in (("u1", a1), ("u0", a0), ("u2d", a2), ("s1", s1), ("e1", e1), ("e2", e2)):
            for ax in (9, -3, 0, -1, 1, None):
                add("straxis", "reduce", params={"name": name, "axis": ax}, operands=[op], layout=key,
                    phase="compute", call=lambda a, f=fn, x=ax: f(a, x))
    # Tuple axes: ufunc.reduce's "duplicate value in 'axis'", normalize_axis_tuple's "repeated axis" (with average's
    # argument name), every axis checked before the repeat.
    tuple_reducers = {
        "all": lambda a, ax: np.all(a, axis=ax), "any": lambda a, ax: np.any(a, axis=ax),
        "ptp": lambda a, ax: np.ptp(a, axis=ax), "median": lambda a, ax: np.median(a, axis=ax),
        "nanmedian": lambda a, ax: np.nanmedian(a, axis=ax), "average": lambda a, ax: np.average(a, axis=ax),
        "percentile": lambda a, ax: np.percentile(a, 50, axis=ax), "quantile": lambda a, ax: np.quantile(a, 0.5, axis=ax),
        "nanpercentile": lambda a, ax: np.nanpercentile(a, 50, axis=ax),
        "nanquantile": lambda a, ax: np.nanquantile(a, 0.5, axis=ax),
    }
    for name, fn in tuple_reducers.items():
        for axes in ([0, 0], [1, -1], [0, 5], [5, 5], [0, 1], [0], [1]):
            add("straxis", "reduce", params={"name": name, "axes": axes}, operands=[a2], layout="u2d",
                phase="compute", call=lambda a, f=fn, x=tuple(axes): f(a, x))

    # The quantile family: q, then the axis, then the method, then the empty axis, then the interpolation.
    for name, qv, bad in (("percentile", 50, 150), ("quantile", 0.5, 2.0), ("nanpercentile", 50, 150),
                          ("nanquantile", 0.5, 2.0)):
        fn = getattr(np, name)
        for method in ("linear", "lower", "higher", "nearest", "midpoint", "inverted_cdf", "hazen", "weibull",
                       "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf",
                       "median_unbiased", "normal_unbiased"):
            for key, op, ax in (("u1", a1, None), ("u2d0", a2, 0), ("u2dN", a2, None), ("e1", e1, None),
                                ("e2-0", e2, 0), ("e2-1", e2, 1), ("u0", a0, None)):
                add("straxis", "quantile", params={"name": name, "q": qv, "axis": ax, "method": method},
                    operands=[op], layout=key, phase="compute",
                    call=lambda a, f=fn, q=qv, x=ax, m=method: f(a, q, axis=x, method=m))
        for key, op, ax, q in (("bad-q", a1, None, bad), ("bad-q-axis", a1, 9, bad), ("qarr", a1, None, [qv]),
                               ("qarr2", a1, None, [qv, qv]), ("axis9", a1, 9, qv)):
            add("straxis", "quantile", params={"name": name, "q": q, "axis": ax, "method": "linear"}, operands=[op],
                layout=key, phase="compute", call=lambda a, f=fn, qq=q, x=ax: f(a, qq, axis=x))
        add("straxis", "quantile", params={"name": name, "q": qv, "axis": 9, "method": "foo"}, operands=[a1],
            layout="bad-method-axis", phase="compute", call=lambda a, f=fn, q=qv: f(a, q, axis=9, method="foo"))
        add("straxis", "quantile", params={"name": name, "q": qv, "axis": None, "method": "linear", "keepdims": True},
            operands=[a1], layout="keepdims", phase="compute",
            call=lambda a, f=fn, q=qv: f(a, q, keepdims=True))

    # np.average with weights: the shape checks, then the weight sum's sized string descriptor.
    w3, w2 = num_operand(np.ones(3)), num_operand(np.ones(2))
    for key, op, ax, w in (("same", a1, None, w3), ("noaxis", a2, None, w2), ("axis1", a2, 1, w2),
                           ("badlen", a2, 1, w3), ("axis9", a1, 9, w3)):
        add("straxis", "average_w", params={"axis": ax}, operands=[op, w], layout=key, phase="compute",
            call=lambda a, wt, x=ax: np.average(a, axis=x, weights=wt))

    # np.cov / np.corrcoef: their statements up to the first reduction (the descriptor the observations convert to).
    for corr in (False, True):
        fn = np.corrcoef if corr else np.cov
        for key, op in (("u", a1), ("s", s1), ("u2d", a2), ("f8", num_operand(np.linspace(0, 1, 3))),
                        ("b", num_operand(np.array([True, False, True]))), ("e", e2),
                        ("u3d", plain(np.zeros((2, 2, 2), "<U1")))):
            for dt in (None, "U", "S", "U3", "f8"):
                # A numeric dtype= makes NumPy PARSE the text (array(m, dtype=float64)): phase 1 since P1.1.
                parses = dt == "f8" and key in ("u", "s", "u2d", "e", "u3d")
                add("strcov", "cov", params={"corr": corr, "dtype": dt}, operands=[op], layout=key,
                    phase=1 if parses else "compute", call=lambda a, f=fn, d=dt: f(a, dtype=d))
        add("strcov", "cov", params={"corr": corr, "dtype": None, "rowvar": False}, operands=[a2], layout="rowvar",
            phase="compute", call=lambda a, f=fn: f(a, rowvar=False))
        add("strcov", "cov", params={"corr": corr, "dtype": None, "y": 1}, operands=[a1, num_operand(np.ones(3))],
            layout="y-f8", phase="compute", call=lambda a, yy, f=fn: f(a, yy))
        add("strcov", "cov", params={"corr": corr, "dtype": None, "y": 1}, operands=[a1, num_operand(np.ones(2))],
            layout="y-short", phase="compute", call=lambda a, yy, f=fn: f(a, yy))
        add("strcov", "cov", params={"corr": corr, "dtype": "S"}, operands=[plain(np.array(["é", "a"]))],
            layout="nonascii-S", phase="compute", call=lambda a, f=fn: f(a, dtype="S"))
    for wkey, kw in (("fw", {"fweights": [1, 1, 1]}), ("aw", {"aweights": [1, 1, 1]}), ("fw-bad", {"fweights": [1, 1]}),
                     ("fw-frac", {"fweights": [1.5, 1, 1]})):
        add("strcov", "cov", params={"corr": False, "dtype": None, **{k: v for k, v in kw.items()}}, operands=[a1],
            layout=wkey, phase="compute", call=lambda a, kws=kw: np.cov(a, **kws))

    # sort / partition: the axis, then the kth bounds, before the comparison.
    for name, fn in (("partition", np.partition), ("argpartition", np.argpartition)):
        for key, op, kth, ax in (("u1", a1, 5, -1), ("u1", a1, -5, -1), ("u1", a1, 1, 9), ("u0", a0, 0, 0),
                                 ("u0", a0, 0, None), ("u2d", a2, 3, 0), ("e1", e1, 5, -1)):
            add("straxis", "partition", params={"name": name, "kth": kth, "axis": ax}, operands=[op], layout=key,
                phase="compute", call=lambda a, f=fn, k_=kth, x=ax: f(a, k_, axis=x))
    ro_sort = plain(np.array(["b", "a"]))
    ro_sort[1].flags.writeable = False
    add("straxis", "sort_inplace", params={"axis": 9}, operands=[ro_sort], layout="ro-axis9", phase="compute",
        call=lambda a: a.sort(axis=9))
    add("straxis", "sort_inplace", params={"axis": -1}, operands=[ro_sort], layout="ro", phase="compute",
        call=lambda a: a.sort(axis=-1))



# ----------------------------------------------------------------------------------------------------------------------
# Section K: the answers the SHAPE decides. NumPy computes these calls on text, but over an empty input, a line of one
# element or a pair the operators cannot compare, its loops read nothing (or nothing that changes the answer) — so
# NumSharp's string support answers them now (phase 1), and defers the rest (phase "compute").
# ----------------------------------------------------------------------------------------------------------------------

_ARG_LIKE = ("argmax", "argmin", "nanargmax", "nanargmin", "sort", "argsort", "partition", "argpartition")


def _reduced_line(a, axis):
    """The length of the line an arg-reduction / sort reads, or None for an out-of-range axis."""
    if axis is None or a.ndim == 0:
        return a.size
    if not -a.ndim <= axis < a.ndim:
        return None
    return a.shape[axis % a.ndim]


def _shape_decides(name, a, axis):
    """Mirrors NDStringUfuncs.ShapeDecides: an empty input, or (sorts / arg-reductions) a line of at most one element."""
    if a.size == 0:
        return True
    if name in _ARG_LIKE:
        line = _reduced_line(a, axis)
        return line is not None and line <= 1
    return False


def _unique_shape_decides(a, axis):
    """Mirrors NDArray.UniqueShapeDecides: at most one element / slab to order, or slabs of zero width."""
    if axis is None:
        return a.size <= 1
    if not -a.ndim <= axis < a.ndim:
        return True  # the AxisError is decided by the shape
    return a.size == 0 or a.shape[axis % a.ndim] <= 1


def gen_shape_decided():
    e1, e2, e3 = plain(np.zeros(0, "<U2")), plain(np.zeros((0, 3), "<U2")), plain(np.zeros((3, 0), "<U2"))
    e3v = operand("<U3", "e3")              # a (2, 0) view at an offset of a 6-element base
    es = plain(np.zeros(0, "S2"))
    u0 = plain(np.array("ab"))
    z0 = operand("<U3", "z0")               # a 0-d view of a 1-D base
    one = plain(np.array(["b"]))
    onev = operand("<U3", "one")            # a one-element view at an offset
    col = plain(np.array([["b"], ["a"]]))   # (2, 1): lines of one element along axis 1
    row = plain(np.array([["a", "b", "a"]]))  # (1, 3): one slab along axis 0
    # Its own base: diff(n=0) and trim_zeros(axis=()) return the input itself, so its OWNDATA is compared.
    u3 = own(np.array(["ab", "c", "de"]))
    s1, s2 = plain(np.array([b"q"])), plain(np.array([b"x", b"y"]))
    sx = plain(np.array([b"\xff", b"a"]))
    f2, ef = num_operand(np.array([1.5, 2.0])), num_operand(np.zeros(0))
    i3, i20 = num_operand(np.arange(3)), num_operand(np.arange(20))
    named = {"e1": e1, "e2": e2, "e3": e3, "e3v": e3v, "es": es, "u0": u0, "z0": z0, "one": one, "onev": onev,
             "col": col, "row": row, "u3": u3, "s1": s1}

    # Truth tests, counts and arg-reductions, every axis and keepdims.
    reducers = {
        "any": lambda a, ax, kd: np.any(a, axis=ax, keepdims=kd),
        "all": lambda a, ax, kd: np.all(a, axis=ax, keepdims=kd),
        "count_nonzero": lambda a, ax, kd: np.count_nonzero(a, axis=ax, keepdims=kd),
        "argmax": lambda a, ax, kd: np.argmax(a, axis=ax, keepdims=kd),
        "argmin": lambda a, ax, kd: np.argmin(a, axis=ax, keepdims=kd),
        "nanargmax": lambda a, ax, kd: np.nanargmax(a, axis=ax, keepdims=kd),
        "nanargmin": lambda a, ax, kd: np.nanargmin(a, axis=ax, keepdims=kd),
    }
    for name, fn in reducers.items():
        for key, op in named.items():
            v = op[1]
            for ax in (None, 0, 1, -1, 2):
                for kd in (False, True):
                    if name in ("count_nonzero", "argmax", "argmin") and ax is None and kd:
                        continue  # NumSharp has no axis=None + keepdims form of these (an API gap, not a text one)
                    add("strshape", "shape_reduce", params={"name": name, "axis": ax, "keepdims": kd}, operands=[op],
                        layout=key, phase=1 if _shape_decides(name, v, ax) else "compute",
                        call=lambda a, f=fn, x=ax, k=kd: f(a, x, k))
    # out= / where= on an empty input (any / all): the identity lands in out.
    for name in ("any", "all"):
        fn = getattr(np, name)
        for key, op, ax, oshape in (("e2", e2, 0, (3,)), ("e2", e2, 1, (0,)), ("e3", e3, 1, (3,))):
            add("strshape", "truth_out", params={"name": name, "axis": ax}, operands=[op, target(np.ones(oshape, bool))],
                layout=key, phase=1, mutate=1, call=lambda a, o, f=fn, x=ax: f(a, axis=x, out=o))
            add("strshape", "truth_where", params={"name": name, "axis": ax},
                operands=[op, plain(np.ones(op[1].shape, bool))], layout=key, phase=1,
                call=lambda a, w, f=fn, x=ax: f(a, axis=x, where=w))

    # Sorts and partitions.
    for name in ("sort", "argsort", "partition", "argpartition"):
        for key, op in named.items():
            v = op[1]
            for ax in (None, 0, 1, -1):
                kths = (0, 1) if name in ("partition", "argpartition") else (None,)
                for kth in kths:
                    if kth is None:
                        call = (lambda a, x=ax, n=name: getattr(np, n)(a, axis=x))
                    else:
                        call = (lambda a, x=ax, n=name, k=kth: getattr(np, n)(a, k, axis=x))
                    add("strshape", "shape_sort", params={"name": name, "axis": ax, "kth": kth}, operands=[op],
                        layout=key, phase=1 if _shape_decides(name, v, ax) else "compute", call=call)

    # The index searches.
    for name in ("nonzero", "flatnonzero", "argwhere", "where1"):
        fn = (lambda a: np.where(a)) if name == "where1" else getattr(np, name)
        for key, op in named.items():
            v = op[1]
            decided = v.size == 0 or (name in ("nonzero", "where1") and v.ndim == 0)
            add("strshape", "index_search", params={"name": name}, operands=[op], layout=key,
                phase=1 if decided else "compute", call=lambda a, f=fn: f(a))

    # np.unique and the Array-API family.
    for key, op in named.items():
        v = op[1]
        for ax in (None, 0, 1):
            for flags in ((False, False, False), (True, True, True), (False, True, False)):
                ri, rv, rc = flags
                add("strshape", "unique", params={"ri": ri, "rv": rv, "rc": rc, "axis": ax}, operands=[op], layout=key,
                    phase=1 if _unique_shape_decides(v, ax) else "compute",
                    call=lambda a, x=ax, i=ri, n=rv, c=rc: np.unique(a, return_index=i, return_inverse=n,
                                                                     return_counts=c, axis=x))
        for name in ("unique_values", "unique_counts", "unique_inverse", "unique_all"):
            fn = getattr(np, name)
            add("strshape", "unique_api", params={"name": name}, operands=[op], layout=key,
                phase=1 if v.size <= 1 else "compute", call=lambda a, f=fn: f(a))

    # isin: decided by the sizes and dtypes where NumPy reads no text (or compares a pair equal has no loop for).
    pairs = [("e1", e1, "u3", u3), ("e2", e2, "u3", u3), ("u3", u3, "e1", e1), ("u3", u3, "f2", f2), ("f2", f2, "u3", u3),
             ("u3", u3, "s2", s2), ("u3", u3, "i3", i3), ("u3", u3, "i20", i20), ("u3", u3, "u3", u3),
             ("es", es, "u3", u3), ("ef", ef, "u3", u3), ("e1", e1, "s2", s2), ("one", one, "s1", s1),
             ("u0", u0, "f2", f2)]
    for k1, x, k2, y in pairs:
        n1, n2 = x[1].size, y[1].size
        same = x[1].dtype.kind in "US" and y[1].dtype.kind == x[1].dtype.kind
        decided = n1 == 0 or (n2 < 10 * n1 ** 0.145 and (n2 == 0 or not same))
        for invert in (False, True):
            for au in (False, True):
                add("strshape", "isin", params={"invert": invert, "assume_unique": au, "kind": None},
                    operands=[x, y], layout=f"{k1}-{k2}", phase=1 if decided else "compute",
                    call=lambda a, b, i=invert, u=au: np.isin(a, b, invert=i, assume_unique=u))
    for kind in ("table", "bad", "sort"):
        add("strshape", "isin", params={"invert": False, "assume_unique": False, "kind": kind}, operands=[u3, u3],
            layout=f"kind-{kind}", phase="compute", call=lambda a, b, k=kind: np.isin(a, b, kind=k))
    # A decode of non-ASCII bytes NumPy reaches after sorting the bytes: deferred (its error names the first failing
    # element of the sorted text).
    add("strshape", "isin", params={"invert": False, "assume_unique": False, "kind": None}, operands=[e1, sx],
        layout="e1-sx", phase=2, call=lambda a, b: np.isin(a, b))

    # The set routines: (pair, phase per routine).
    set_cases = [
        ("e1-e1", e1, e1, {"union1d": 1, "intersect1d": 1, "setxor1d": 1, "setdiff1d": 1}),
        ("e1-one", e1, one, {"union1d": 1, "intersect1d": 1, "setxor1d": 1, "setdiff1d": 1}),
        ("u0-e1", u0, e1, {"union1d": 1, "intersect1d": 1, "setxor1d": 1, "setdiff1d": 1}),
        ("e1-ef", e1, ef, {"union1d": 1, "intersect1d": 1, "setxor1d": 1, "setdiff1d": 1}),
        ("e1-u3", e1, u3, {"union1d": "compute", "intersect1d": 1, "setxor1d": "compute", "setdiff1d": 1}),
        ("u3-ef", u3, ef, {"union1d": "compute", "intersect1d": 1, "setxor1d": "compute", "setdiff1d": "compute"}),
        ("e2-s2", e2, s2, {"union1d": "compute", "intersect1d": 1, "setxor1d": "compute", "setdiff1d": 1}),
        ("one-one", one, one, {"union1d": "compute", "intersect1d": "compute", "setxor1d": "compute",
                               "setdiff1d": "compute"}),
        ("one-f2", one, f2, {"union1d": "compute", "intersect1d": "compute", "setxor1d": "compute", "setdiff1d": 1}),
        ("i3-u3", i3, u3, {"union1d": "compute", "intersect1d": "compute", "setxor1d": "compute", "setdiff1d": 1}),
        ("u3-e1", u3, e1, {"union1d": "compute", "intersect1d": 1, "setxor1d": "compute", "setdiff1d": "compute"}),
        # union1d decodes in C order before any sort (NumSharp's concatenation raises the same error); the others sort
        # the bytes first, so the failing element NumPy names depends on the text (deferred).
        ("e1-sx", e1, sx, {"union1d": 1, "intersect1d": 2, "setxor1d": 2, "setdiff1d": 2}),
    ]
    for key, x, y, phases in set_cases:
        for name, ph in phases.items():
            fn = getattr(np, name)
            variants = [{}]
            if name in ("intersect1d", "setxor1d", "setdiff1d"):
                variants.append({"assume_unique": True})
            if name == "intersect1d":
                variants.append({"return_indices": True})
            for kw in variants:
                add("strshape", "setop", params={"name": name, **kw}, operands=[x, y], layout=key, phase=ph,
                    call=lambda a, b, f=fn, k=kw: f(a, b, **k))

    # Equality: shape and dtype decide unless two non-empty strings of one class are compared.
    # Two DIFFERENT objects with equal contents (array_equal's `a1 is a2` shortcut has its own case below).
    eq_cases = [("e1-e1", e1, plain(np.zeros(0, "<U2"))), ("e1-es", e1, es), ("u3-f3", u3, num_operand(np.zeros(3))),
                ("u3-u3", u3, plain(np.array(["ab", "c", "de"]))),
                ("u3-one", u3, one), ("e2-e1", e2, e1), ("u3-s3", u3, plain(np.array([b"ab", b"c", b"de"])))]
    for key, x, y in eq_cases:
        same = x[1].dtype.kind == y[1].dtype.kind
        for equal_nan in (False, True):
            if equal_nan:
                decided = True  # a shape mismatch, or isnan's TypeError
            else:
                decided = not (same and x[1].shape == y[1].shape and x[1].size)
            add("strshape", "equality", params={"name": "array_equal", "equal_nan": equal_nan, "same": False},
                operands=[x, y], layout=key, phase=1 if decided else "compute",
                call=lambda a, b, e=equal_nan: np.array_equal(a, b, equal_nan=e))
        try:
            bshape = np.broadcast_shapes(x[1].shape, y[1].shape)
            empty = int(np.prod(bshape)) == 0
            decided = empty or not same
        except ValueError:
            decided = True
        add("strshape", "equality", params={"name": "array_equiv", "equal_nan": False, "same": False},
            operands=[x, y], layout=key, phase=1 if decided else "compute", call=lambda a, b: np.array_equiv(a, b))
    # The same object: array_equal(a, a, equal_nan=True) is True without reading it (a1 is a2).
    add("strshape", "equality", params={"name": "array_equal", "equal_nan": True, "same": True}, operands=[u3],
        layout="u3-same", phase=1, call=lambda a: np.array_equal(a, a, equal_nan=True))

    # lexsort, sort_complex, trim_zeros, diff.
    for key, keys, ax, ph in (("col", [col], -1, 1), ("u3", [u3], -1, "compute"), ("i3-u3", [i3, u3], -1, "compute"),
                              ("col-col", [col, col], 0, "compute"), ("e1", [e1], -1, 1), ("one", [one], -1, 1),
                              ("e2", [e2], 0, 1)):
        add("strshape", "lexsort", params={"axis": ax}, operands=keys, layout=key, phase=ph,
            call=lambda *ks, x=ax: np.lexsort(list(ks), axis=x))
    for key, op in (("e1", e1), ("e2", e2), ("u0", u0), ("one", one), ("u3", u3)):
        add("strshape", "sort_complex", operands=[op], layout=key,
            phase=1 if op[1].size == 0 or op[1].ndim == 0 else "compute", call=lambda a: np.sort_complex(a))
    for key, op, trim, ax in (("e1", e1, "fb", None), ("e2", e2, "fb", None), ("e2", e2, "fb", 1), ("e2", e2, "f", [0]),
                              ("u0", u0, "fb", None), ("u3", u3, "fb", None), ("u3", u3, "x", None),
                              ("u3", u3, "fb", []), ("e3", e3, "b", None)):
        decided = op[1].size == 0 or op[1].ndim == 0 or ax == [] or trim == "x"
        add("strshape", "trim_zeros", params={"trim": trim, "axis": ax}, operands=[op], layout=key,
            phase=1 if decided else "compute", call=lambda a, t=trim, x=ax: np.trim_zeros(a, trim=t, axis=x))
    for key, ops, n, ax, pre in (("e1", [e1], 1, -1, False), ("u3", [u3], 1, -1, False), ("u3", [u3], 0, -1, False),
                                 ("e2", [e2], 1, 0, False), ("e2", [e2], 2, 1, False), ("e1-ef", [e1, ef], 1, -1, True),
                                 ("u0", [u0], 1, -1, False), ("u3", [u3], -1, -1, False)):
        add("strshape", "diff", params={"n": n, "axis": ax, "prepend": pre}, operands=ops, layout=key, phase=1,
            call=lambda a, *rest, nn=n, x=ax: np.diff(a, n=nn, axis=x, prepend=rest[0]) if rest else
            np.diff(a, n=nn, axis=x))

    # The binary loops over an empty broadcast compute nothing: NumPy returns the empty result (or out=).
    for name in ("equal", "not_equal", "less", "greater_equal", "add", "logical_and", "logical_or", "logical_xor"):
        fn = getattr(np, name)
        for key, x, y in (("e1-e1", e1, e1), ("e2-u3", e2, u3), ("e2-e1x", e2, plain(np.zeros((0, 1), "<U4"))),
                          ("e1-es", e1, es), ("e1-ef", e1, ef), ("u3-e1", u3, e1)):
            add("strshape", "ufunc", params={"name": name}, operands=[x, y], layout=key, phase="compute",
                call=lambda a, b, f=fn: f(a, b))
    for name, odt in (("equal", bool), ("add", "<U6"), ("logical_or", bool)):
        fn = getattr(np, name)
        add("strshape", "ufunc_kw", params={"name": name, "dtype": None, "nin": 2, "out": 2},
            operands=[e1, plain(np.zeros(0, "<U4")), target(np.zeros(0, odt))], layout="out", phase="compute",
            call=lambda a, b, o, f=fn: f(a, b, out=o))


# ----------------------------------------------------------------------------------------------------------------------
# Section L: the order of the checks NumPy makes before it converts or compares text — the validation of
# interp / digitize / repeat, empty conversions (astype, copyto, setitem, where, concatenate, choose), place / putmask
# with ndarray values, insert / delete with array obj, searchsorted.
# ----------------------------------------------------------------------------------------------------------------------

def gen_validation_order():
    F = num_operand(np.array([1.0, 2.0, 3.0]))
    U = plain(np.array(["ab", "c", "de"]))
    C3 = num_operand(np.array([1j, 2j, 3j]))
    Ue = plain(np.array([], "<U2"))
    Ue1 = plain(np.array([], "<U1"))
    u0 = plain(np.array("ab"))
    c0 = num_operand(np.array(1j))
    E = num_operand(np.zeros(0))

    # interp: fp's dtype picks the loop; then fp, xp and x each meet the depth check and the 'safe' cast to float64.
    interp_cases = [
        ("u0-F-F", [u0, F, F], {}), ("c0-F-F", [c0, F, F], {}), ("C-F-F", [num_operand(np.array([1j])), F, F], {}),
        ("F-C-F", [F, C3, F], {}), ("F-C2d-F", [F, num_operand(np.array([[1j, 2j, 3j]])), F], {}),
        ("F-F-C", [F, F, C3], {}), ("Ue-F-F-p", [Ue, F, F], {"period": 2.0}),
        ("F-Ue-Ue-p", [F, Ue1, Ue1], {"period": 2.0}), ("F-Ue-F-p", [F, Ue1, F], {"period": 2.0}),
        ("Ue-F-F", [Ue, F, F], {}), ("F-F-U-p0", [F, F, U], {"period": 0.0}),
        ("C-F-F-p", [num_operand(np.array([1j])), F, F], {"period": 2.0}),
        ("F-F-F-left-c", [F, F, F], {"left": 1 + 0j}), ("F-F-U-left-c", [F, F, U], {"left": 1 + 0j}),
        ("F-E-E-left-c", [F, E, E], {"left": 1 + 0j}),
    ]
    for key, ops, kw in interp_cases:
        params = {k: (v if not isinstance(v, complex) else [v.real, v.imag]) for k, v in kw.items()}
        params["complex_left"] = isinstance(kw.get("left"), complex)
        add("strvalid", "interp", params=params, operands=ops, layout=key, phase="compute",
            call=lambda x, xp, fp, k=kw: np.interp(x, xp, fp, **k))

    # digitize: complex x first, then the bins' depth, 'safe' cast and monotonicity, then the search.
    dig_cases = [("Ue-F", [Ue, F], False), ("Ue2-Frev", [plain(np.zeros((0, 3), "<U2")), num_operand(np.array([3.0, 2.0, 1.0]))], False),
                 ("U-empty", [plain(np.array(["a"])), E], False), ("U2d-empty", [plain(np.array([["a", "b"]])), E], False),
                 ("U-nonmono", [plain(np.array(["a"])), num_operand(np.array([3.0, 1.0, 2.0]))], False),
                 ("U-dec", [plain(np.array(["a"])), num_operand(np.array([3.0, 2.0, 1.0]))], False),
                 ("U0-F", [plain(np.array("a")), F], False), ("c0-F", [c0, F], False),
                 ("F-C", [F, num_operand(np.array([1j, 2j]))], False), ("U-Bempty", [plain(np.array(["a"])), num_operand(np.array([], bool))], False),
                 ("U-B", [plain(np.array(["a"])), num_operand(np.array([False, True]))], False),
                 ("B-U", [num_operand(np.array([True])), U], False), ("U-U", [U, U], False),
                 ("F-F2d", [F, num_operand(np.array([[1.0, 2.0]]))], False), ("F-F0", [F, num_operand(np.array(5.0))], False),
                 ("Ue-F-right", [Ue, F], True)]
    for key, ops, right in dig_cases:
        add("strvalid", "digitize", params={"right": right}, operands=ops, layout=key, phase="compute",
            call=lambda x, b, r=right: np.digitize(x, b, right=r))

    # repeat with array repeats of every kind: the depth check, then the 'safe' cast to intp, then the axis.
    rep_cases = [("U-I2d", [U, num_operand(np.array([[1]]))], None), ("U-F0", [U, num_operand(np.array(2.0))], None),
                 ("U-U0", [U, plain(np.array("a"))], None), ("Ue-Ue", [Ue, Ue1], None),
                 ("F-I2d", [F, num_operand(np.array([[1, 1, 1]]))], None),
                 ("F-u64", [F, num_operand(np.array([1, 1, 1], np.uint64))], None),
                 ("F-b", [F, num_operand(np.array([True, False, True]))], None),
                 ("F-I2d-ax9", [F, num_operand(np.array([[1]]))], 9), ("F-U-ax9", [F, U], 9),
                 ("F-F-ax9", [F, num_operand(np.array([1.0]))], 9), ("F-I-ax9", [F, num_operand(np.array([1]))], 9),
                 ("U-I", [U, num_operand(np.array([1, 0, 2]))], None), ("U-I-ax0", [U, num_operand(np.array([2]))], 0)]
    for key, ops, ax in rep_cases:
        add("strvalid", "repeat_arr", params={"axis": ax}, operands=ops, layout=key, phase="compute",
            call=lambda a, r, x=ax: np.repeat(a, r, axis=x))

    # Conversions of nothing: astype / copyto / setitem / where / concatenate / choose over empty selections.
    for key, op, dt, order, casting in (("U0-f8", Ue, "f8", "K", "unsafe"), ("U0x3-f8-F", plain(np.zeros((0, 3), "<U2")), "f8", "F", "unsafe"),
                                        ("f80-U", E, "U", "K", "unsafe"), ("f80-S5", E, "S5", "K", "unsafe"),
                                        ("f80-U5-safe", E, "U5", "K", "safe"), ("U0-f8-safe", Ue, "f8", "K", "safe"),
                                        ("U0-b", Ue, "?", "K", "unsafe"), ("U0-c16", Ue, "c16", "C", "unsafe")):
        add("strvalid", "astype", params={"dtype": dt, "copy": True, "order": order, "casting": casting}, operands=[op],
            layout=key, phase="compute", call=lambda a, d=dt, o=order, c=casting: a.astype(d, order=o, casting=c))

    copyto_cases = [
        ("F0-U", target(np.zeros(0)), plain(np.array(["x"])), "unsafe", None),
        ("F3-U2", target(np.zeros(3)), plain(np.array(["x", "y"])), "unsafe", None),
        ("F3-U-maskF", target(np.zeros(3)), plain(np.array(["x"])), "unsafe", [False, False, False]),
        ("F3-U-mask0F", target(np.zeros(3)), plain(np.array(["x"])), "unsafe", False),
        ("F3-U0-mask0F", target(np.zeros(3)), plain(np.array("x")), "unsafe", False),
        ("F1-U0-maskF", target(np.zeros(1)), plain(np.array("x")), "unsafe", [False]),
        ("F0-U0", target(np.zeros(0)), plain(np.array("x")), "unsafe", None),
        ("F0-U0-badmask", target(np.zeros(0)), plain(np.array("x")), "unsafe", [False, False]),
        ("U3-F-maskF", target(np.zeros(3, "<U5")), num_operand(np.array([1.5])), "same_kind", [False, False, False]),
        ("U3-F-safe", target(np.zeros(3, "<U5")), num_operand(np.array([1.5])), "safe", None),
        ("U3-F-badshape", target(np.zeros(3, "<U5")), num_operand(np.array([1.5, 2.5])), "same_kind", None),
        ("U3-F", target(np.zeros(3, "<U5")), num_operand(np.array([1.5])), "same_kind", None),
    ]
    for key, dst, src, casting, where in copyto_cases:
        add("strvalid", "copyto", params={"casting": casting, "where": where}, operands=[dst, src], layout=key,
            mutate=0, phase="compute",
            call=lambda d, s, c=casting, w=where: np.copyto(d, s, casting=c, where=True if w is None else np.array(w)))

    def setit(a, idx, v):
        a[idx] = v

    set_cases = [
        ("S0-ell-Ue", target(np.zeros(0, "S1")), [{"ellipsis": True}], plain(np.array(["é"]))),
        ("S3-fancy0-Ue", target(np.zeros(3, "S1")), [{"arr": []}], plain(np.array(["é"]))),
        ("S3-boolF-Ue", target(np.zeros(3, "S1")), [{"mask": [False, False, False]}], plain(np.array(["é"]))),
        ("U0-ell-F", target(np.zeros(0, "U1")), [{"ellipsis": True}], num_operand(np.array([1.5]))),
        ("U3-boolF-F", target(np.zeros(3, "U3")), [{"mask": [False, False, False]}], num_operand(np.array([1.5]))),
        ("U3-fancy0-F", target(np.zeros(3, "U3")), [{"arr": []}], num_operand(np.array([1.5]))),
        ("U0-ell-Fbad", target(np.zeros(0, "U1")), [{"ellipsis": True}], num_operand(np.array([1.5, 2.5]))),
        ("F0-ell-U", target(np.zeros(0)), [{"ellipsis": True}], plain(np.array(["x"]))),
        ("F3-boolF-U", target(np.zeros(3)), [{"mask": [False, False, False]}], plain(np.array(["x"]))),
        ("F3-fancy0-U", target(np.zeros(3)), [{"arr": []}], plain(np.array(["x"]))),
        ("F0-ell-Ubad", target(np.zeros(0)), [{"ellipsis": True}], plain(np.array(["x", "y"]))),
        ("F3-int-U0", target(np.zeros(3)), [{"int": 0}], plain(np.array("x"))),
    ]
    for key, a, idx, v in set_cases:
        k = tuple(_py_index(e) for e in idx)
        add("strvalid", "setitem_array", params={"index": idx}, operands=[a, v], layout=key, mutate=0,
            phase="compute", call=lambda arr, val, kk=k: setit(arr, kk, val))

    where_cases = [("empty-F-U", [num_operand(np.zeros(0, bool)), E, Ue]),
                   ("Ucond-empty-F-F", [Ue1, E, E]), ("Ucond-F-F", [plain(np.array(["a", ""])), num_operand(np.zeros(2)), num_operand(np.ones(2))]),
                   ("Ucond-empty-U-U", [Ue1, Ue, plain(np.zeros(0, "<U3"))]),
                   ("F-cond-F-U", [num_operand(np.array([True])), num_operand(np.array([1.5])), plain(np.array(["a"]))])]
    for key, ops in where_cases:
        add("strvalid", "where3", operands=ops, layout=key, phase="compute", call=lambda c, x, y: np.where(c, x, y))

    concat_cases = [
        ("F0-U0", [E, Ue], {}), ("F0-U1", [E, plain(np.array(["ab"]))], {}),
        ("F1-U1", [num_operand(np.ones(1)), plain(np.array(["ab"]))], {}),
        ("F0-U1-f8", [E, plain(np.array(["ab"]))], {"dtype": "f8", "casting": "unsafe"}),
        ("F1-U0-f8", [num_operand(np.ones(1)), Ue], {"dtype": "f8", "casting": "unsafe"}),
        ("Sx-U5-U3-safe", [plain(np.array([b"\xff"])), plain(np.array(["abcde"]))], {"dtype": "U3", "casting": "safe"}),
        ("Ue-F-S3", [plain(np.array(["é"])), num_operand(np.array([1]))], {"dtype": "S3", "casting": "unsafe"}),
        ("F-Ue-S3", [num_operand(np.array([1])), plain(np.array(["é"]))], {"dtype": "S3", "casting": "unsafe"}),
    ]
    for key, ops, kw in concat_cases:
        add("strvalid", "concat_kw", params={"dtype": kw.get("dtype"), "casting": kw.get("casting", "same_kind"),
                                             "out": None}, operands=ops, layout=key, phase="compute",
            call=lambda *xs, k=kw: np.concatenate(list(xs), **k))
    for key, ops, out_shape, out_dt in (("U-U-outbad", [plain(np.array(["a"])), plain(np.array(["b"]))], (3,), "S1"),
                                        ("U-U-outdim", [plain(np.array(["a"])), plain(np.array(["b"]))], (2, 1), "<U1"),
                                        ("U-U-out", [plain(np.array(["a"])), plain(np.array(["bc"]))], (2,), "<U1")):
        add("strvalid", "concat_kw", params={"dtype": None, "casting": "same_kind", "out": len(ops)},
            operands=ops + [target(np.zeros(out_shape, out_dt))], layout=key, phase="compute",
            call=lambda *xs: np.concatenate(list(xs[:-1]), out=xs[-1], casting="same_kind"))

    choose_cases = [("F-U-empty", [0] * 0, "raise", [E, Ue]), ("F-U", [0], "raise", [num_operand(np.zeros(1)), plain(np.array(["ab"]))]),
                    ("F-U-sel1", [1], "raise", [num_operand(np.zeros(1)), plain(np.array(["ab"]))]),
                    ("F-U-bad", [5], "raise", [num_operand(np.zeros(1)), plain(np.array(["ab"]))]),
                    ("F-U-wrap", [3], "wrap", [num_operand(np.zeros(1)), plain(np.array(["ab"]))]),
                    ("F-U-clip", [3], "clip", [num_operand(np.zeros(1)), plain(np.array(["ab"]))]),
                    ("F-U-clip0", [-3], "clip", [num_operand(np.zeros(1)), plain(np.array(["ab"]))]),
                    ("U-U", [1, 0], "raise", [plain(np.array(["a", "b"])), plain(np.array(["cc", "dd"]))])]
    for key, idx, mode, choices in choose_cases:
        add("strvalid", "choose_mode", params={"index": idx, "mode": mode}, operands=choices, layout=key,
            phase="compute", call=lambda *cs, i=idx, m=mode: np.choose(np.array(i, dtype=np.intp), list(cs), mode=m))

    # place / putmask with ndarray values: PyArray_FromAny without FORCECAST (the 'safe' rule), before the mask is read.
    for name in ("place", "putmask"):
        fn = getattr(np, name)
        for key, arr, mask, vals in (
                ("i-f8", target(np.arange(5, dtype=np.int32)), [True] * 5, num_operand(np.array([1.9, 2.9]))),
                ("b-i", target(np.zeros(4, bool)), [True] * 4, num_operand(np.array([0, 5, 0, 9], np.int32))),
                ("U3-U4", target(np.array(["a", "b", "c"])), [True, False, True], plain(np.array(["wxyz"]))),
                ("U3-S2", target(np.array(["a", "b", "c"], "<U3")), [True, False, True], plain(np.array([b"xy"]))),
                ("S2-U", target(np.array([b"a", b"b"])), [True, True], plain(np.array(["x"]))),
                ("F-U", target(np.zeros(3)), [False, False, False], plain(np.array(["x"]))),
                ("U-F-maskF", target(np.zeros(3, "<U32")), [False, False, False], num_operand(np.array([1.5]))),
                ("U-I-maskT", target(np.zeros(3, "<U32")), [True, True, True], num_operand(np.array([1]))),
                ("F-Uempty", target(np.zeros(3)), [True, False, True], plain(np.zeros(0, "<U1"))),
                ("Fe-F", target(np.zeros(0, np.int64)), [], num_operand(np.array([1.5]))),
                ("U-Umask", target(np.zeros(3)), None, num_operand(np.array([7.0])))):
            if mask is None:
                mask_op = plain(np.array(["a", "", "b"]))
                add("strvalid", f"{name}_arr", params={"mask": None}, operands=[arr, vals, mask_op], layout=key,
                    mutate=0, phase="compute", call=lambda a, v, m, f=fn: f(a, m, v))
            else:
                add("strvalid", f"{name}_arr", params={"mask": mask}, operands=[arr, vals], layout=key, mutate=0,
                    phase="compute", call=lambda a, v, m=mask, f=fn: f(a, np.array(m, dtype=bool), v))

    # insert / delete with an array obj, and searchsorted's argument checks.
    Fi = num_operand(np.array([1.5, 2.0, 3.0]))
    for key, obj, values in (("U3", U, num_operand(np.array(9.0))), ("U1", plain(np.array(["a"])), num_operand(np.array(9.0))),
                             ("S1", plain(np.array([b"a"])), num_operand(np.array(9.0))),
                             ("S3", plain(np.array([b"a", b"b", b"c"])), num_operand(np.array(9.0))),
                             ("U0d", plain(np.array("a")), num_operand(np.array(9.0))),
                             ("Uempty", plain(np.zeros(0, "<U1")), num_operand(np.array(9.0))),
                             ("F3", num_operand(np.array([0.5, 1.0, 2.0])), num_operand(np.array(9.0))),
                             ("F1in", num_operand(np.array([1.5])), num_operand(np.array(9.0))),
                             ("F1out", num_operand(np.array([5.5])), num_operand(np.array(9.0))),
                             ("Fempty", num_operand(np.zeros(0)), num_operand(np.array(9.0))),
                             ("C3", num_operand(np.array([1j, 2j, 0j])), num_operand(np.array(9.0))),
                             ("U2d", plain(np.array([["a"]])), num_operand(np.array(9.0))),
                             ("F1in-Uvals", num_operand(np.array([1.5])), U),
                             ("I0-Uvals", num_operand(np.array([0])), U)):
        add("strvalid", "insert_obj", params={"axis": None}, operands=[Fi, obj, values], layout=key, phase="compute",
            call=lambda a, o, v: np.insert(a, o, v))
    add("strvalid", "insert_obj", params={"axis": 9}, operands=[Fi, U, num_operand(np.array(9.0))], layout="U3-ax9",
        phase="compute", call=lambda a, o, v: np.insert(a, o, v, axis=9))
    for key, obj in (("U3", U), ("Uempty", plain(np.zeros(0, "<U1"))), ("U1", plain(np.array(["a"]))),
                     ("U0d", plain(np.array("a"))), ("F1", num_operand(np.array([1.5]))),
                     ("F3", num_operand(np.array([1.5, 0.5, 2.0]))), ("Fempty", num_operand(np.zeros(0))),
                     ("b-short", num_operand(np.array([True]))), ("U2d", plain(np.array([["a"]]))),
                     ("I2d", num_operand(np.array([[0]]))), ("C", num_operand(np.array([1j]))),
                     ("i8", num_operand(np.array([1], np.int8))), ("I2d-2", num_operand(np.array([[0, 2]])))):
        add("strvalid", "delete_obj", params={"axis": None}, operands=[Fi, obj], layout=key, phase="compute",
            call=lambda a, o: np.delete(a, o))
    add("strvalid", "delete_obj", params={"axis": 9}, operands=[Fi, U], layout="U3-ax9", phase="compute",
        call=lambda a, o: np.delete(a, o, axis=9))

    a3 = num_operand(np.array([1, 2, 3]))
    for key, a, v, side, sorter in (
            ("side-L", a3, num_operand(np.array(2)), "L", None), ("side-x", a3, num_operand(np.array(2)), "x", None),
            ("side-empty", a3, num_operand(np.array(2)), "", None), ("side-Right", a3, num_operand(np.array(2)), "Right", None),
            ("a0d", num_operand(np.array(5)), num_operand(np.array(2)), "left", None),
            ("sorter2d", a3, num_operand(np.array(2)), "left", num_operand(np.array([[0, 1, 2]]))),
            ("sorterF", a3, num_operand(np.array(2)), "left", num_operand(np.array([0.0, 1.0, 2.0]))),
            ("sorterB", a3, num_operand(np.array(2)), "left", num_operand(np.array([True, False, True]))),
            ("sorterU64", a3, num_operand(np.array(2)), "left", num_operand(np.array([0, 1, 2], np.uint64))),
            ("sorter0d", a3, num_operand(np.array(2)), "left", num_operand(np.array(0))),
            ("sorterSize", a3, num_operand(np.array(2)), "left", num_operand(np.array([0, 1]))),
            ("U-Fempty", U, E, "left", None), ("Uempty-U", Ue1, plain(np.array(["a", "b"])), "left", None),
            ("Uempty-U0", Ue1, plain(np.array("a")), "left", None), ("Fempty-U", E, plain(np.array(["a"])), "left", None),
            ("U2d-U", plain(np.array([["ab"], ["c"], ["de"]])), plain(np.array(["a"])), "left", None),
            ("U-Sx", U, plain(np.array([b"\xff"])), "left", None),
            ("U-U-sorterU", U, plain(np.array(["a"])), "left", U),
            ("F-U-sorterU", num_operand(np.array([1.0, 2.0])), num_operand(np.array(1.5)), "left", plain(np.array(["a", "b"]))),
            ("Uempty-I", Ue1, num_operand(np.array(3)), "left", None), ("U-I", U, num_operand(np.array(3)), "left", None),
            ("U-U", U, plain(np.array(["b"])), "left", None)):
        ops = [a, v] + ([sorter] if sorter is not None else [])
        add("strvalid", "searchsorted_kw", params={"side": side, "sorter": sorter is not None}, operands=ops,
            layout=key, phase="compute",
            call=lambda aa, vv, *s, sd=side: np.searchsorted(aa, vv, side=sd, sorter=s[0] if s else None))


# ----------------------------------------------------------------------------------------------------------------------
# M. The rest of the text surface: np.linalg, the products, random seeding and bounds, the histogram family, quantile q,
#    reductions and arg-reductions with out=, polynomials, emath, index conversions, dtype predicates, ediff1d, where=
#    masks, 0-d conversions to C# scalars and fill with a sequence. Every call is recorded with phase "compute": NumPy's
#    own refusals and early answers are final, a computation on the text is the next phase. A few numeric calls ride
#    along where the text work found a general divergence (histogramdd's bin specs, norm of an empty matrix).
# ----------------------------------------------------------------------------------------------------------------------

# name -> the np.linalg call over one operand (the C# replay spells each the way a ported NumPy line does).
LINALG1 = [
    ("cholesky", lambda a: np.linalg.cholesky(a)), ("cholesky_up", lambda a: np.linalg.cholesky(a, upper=True)),
    ("det", np.linalg.det), ("slogdet", np.linalg.slogdet), ("inv", np.linalg.inv), ("pinv", np.linalg.pinv),
    ("pinv_h", lambda a: np.linalg.pinv(a, hermitian=True)), ("eig", np.linalg.eig), ("eigvals", np.linalg.eigvals),
    ("eigh", np.linalg.eigh), ("eigh_U", lambda a: np.linalg.eigh(a, "U")), ("eigvalsh", np.linalg.eigvalsh),
    ("eigvalsh_U", lambda a: np.linalg.eigvalsh(a, "U")), ("svd", np.linalg.svd),
    ("svd_nuv", lambda a: np.linalg.svd(a, compute_uv=False)), ("svd_s", lambda a: np.linalg.svd(a, full_matrices=False)),
    ("svd_h", lambda a: np.linalg.svd(a, hermitian=True)), ("svdvals", np.linalg.svdvals),
    ("matrix_rank", np.linalg.matrix_rank), ("matrix_rank_tol", lambda a: np.linalg.matrix_rank(a, tol=0.5)),
    ("matrix_rank_h", lambda a: np.linalg.matrix_rank(a, hermitian=True)), ("cond", np.linalg.cond),
    ("cond_fro", lambda a: np.linalg.cond(a, "fro")), ("cond_1", lambda a: np.linalg.cond(a, 1)),
    ("cond_2", lambda a: np.linalg.cond(a, 2)), ("qr", np.linalg.qr), ("qr_r", lambda a: np.linalg.qr(a, mode="r")),
    ("qr_bad", lambda a: np.linalg.qr(a, mode="x")), ("norm", np.linalg.norm),
    ("norm_fro", lambda a: np.linalg.norm(a, "fro")), ("norm_nuc", lambda a: np.linalg.norm(a, "nuc")),
    ("norm_2", lambda a: np.linalg.norm(a, 2)), ("norm_inf", lambda a: np.linalg.norm(a, np.inf)),
    ("norm_1", lambda a: np.linalg.norm(a, 1)), ("norm_ax", lambda a: np.linalg.norm(a, axis=0)),
    ("norm_bad", lambda a: np.linalg.norm(a, "x")), ("matrix_norm", np.linalg.matrix_norm),
    ("matrix_norm_2", lambda a: np.linalg.matrix_norm(a, ord=2)), ("vector_norm", np.linalg.vector_norm),
    ("matrix_power0", lambda a: np.linalg.matrix_power(a, 0)), ("matrix_power1", lambda a: np.linalg.matrix_power(a, 1)),
    ("matrix_power2", lambda a: np.linalg.matrix_power(a, 2)), ("matrix_power3", lambda a: np.linalg.matrix_power(a, 3)),
    ("matrix_powerm1", lambda a: np.linalg.matrix_power(a, -1)),
    ("matrix_powerm2", lambda a: np.linalg.matrix_power(a, -2)),
    ("tensorinv1", lambda a: np.linalg.tensorinv(a, 1)), ("tensorinv2", lambda a: np.linalg.tensorinv(a, 2)),
    ("tensorinv0", lambda a: np.linalg.tensorinv(a, 0)), ("trace", np.linalg.trace), ("diagonal", np.linalg.diagonal),
    ("matrix_transpose", np.linalg.matrix_transpose),
]

# name -> the np.linalg call over two operands.
LINALG2 = [
    ("solve", np.linalg.solve), ("lstsq", np.linalg.lstsq), ("tensorsolve", np.linalg.tensorsolve),
    ("multi_dot", lambda a, b: np.linalg.multi_dot([a, b])), ("matmul", np.linalg.matmul), ("outer", np.linalg.outer),
    ("tensordot", np.linalg.tensordot), ("vecdot", np.linalg.vecdot), ("cross", np.linalg.cross),
]

# name -> the top-level product or sliding-window call over two operands.
PRODUCTS = [
    ("dot", np.dot), ("inner", np.inner), ("vdot", np.vdot), ("outer", np.outer),
    ("tensordot", lambda a, b: np.tensordot(a, b, 1)), ("kron", np.kron), ("matmul", np.matmul), ("cross", np.cross),
    ("convolve", np.convolve), ("correlate", np.correlate),
]


def _text(values, spec=None):
    """A C-contiguous text operand (its own base when 1-D, see ``own``)."""
    return own(np.array(values, dtype=spec) if spec else np.array(values))


def _view_operand(base, build):
    """A (base, view) pair whose view is ``build(base)`` over a 1-D base."""
    return base, build(base)


def gen_surface():
    U1 = plain(np.array(["ab", "c", "de"]))
    U2 = plain(np.array([["ab", "c"], ["de", "f"]]))
    U23 = plain(np.array([["ab", "c", "x"], ["de", "f", "y"]]))
    U3 = plain(np.zeros((2, 2, 2), "<U1"))
    S2 = plain(np.array([["ab", "c"], ["de", "f"]]).astype("S2"))
    U0 = plain(np.array("ab"))
    Ue = plain(np.zeros((0, 0), "<U1"))
    F2 = num_operand(np.eye(2))
    F1 = num_operand(np.ones(2))
    F3 = num_operand(np.array([1.5, 2.0, 3.0]))
    Ux2 = plain(np.array(["a", "b"]))

    # ---- np.linalg over one text operand, then over two operands ----
    for name, fn in LINALG1:
        for key, a in (("U1", U1), ("U2", U2), ("U23", U23), ("U3", U3), ("S2", S2), ("U0", U0), ("Ue", Ue)):
            # An EMPTY text matrix converts to an empty float64 one, whose QR / SVD NumPy computes through LAPACK; NumSharp
            # Core ships no QR / SVD and answers those through the optional OpenBLAS backend only (none in this replay),
            # so the five SVD / QR calls over Ue are left to the host-pinned linalg tier.
            if key == "Ue" and name in ("qr", "qr_r", "norm_nuc", "norm_2", "matrix_norm_2"):
                continue
            add("strsurface", "linalg1", params={"name": name}, operands=[a], layout=key, phase="compute",
                call=lambda x, f=fn: f(x))
    for name, fn in LINALG2:
        for key, a, b in (("U2,F2", U2, F2), ("F2,U2", F2, U2), ("U2,F1", U2, F1), ("F2,Ux2", F2, Ux2),
                          ("U2,U2", U2, U2), ("U1,U1", U1, U1), ("U1,F3", U1, num_operand(np.ones(3))), ("U23,F2", U23, F2)):
            add("strsurface", "linalg2", params={"name": name}, operands=[a, b], layout=key, phase="compute",
                call=lambda x, y, f=fn: f(x, y))

    # ---- the products and sliding windows: 0-d operands, alignment, empties, the two classes, depth ----
    S1 = plain(np.array([b"ab", b"c", b"de"]))
    Ue1 = plain(np.zeros(0, "<U1"))
    pairs = [("U1,F3", U1, F3), ("F3,U1", F3, U1), ("U0,F3", U0, F3), ("F3,U0", F3, U0), ("U0,U0", U0, U0),
             ("U2,F22", U2, num_operand(np.ones((2, 2)))), ("U2,F3", U2, F3), ("U23,U23", U23, U23),
             ("Ue1,Ue1", Ue1, Ue1), ("Ue1,Fe1", Ue1, num_operand(np.zeros(0))), ("U1,Ux2", U1, Ux2),
             ("S1,S1", S1, S1), ("U1,S1", U1, S1), ("U2,F32", U2, num_operand(np.ones((3, 2)))), ("U3,U2", U3, U2),
             ("U1,I3", U1, num_operand(np.array([1, 2, 3])))]
    for name, fn in PRODUCTS:
        for key, a, b in pairs:
            add("strsurface", "product", params={"name": name}, operands=[a, b], layout=key, phase="compute",
                call=lambda x, y, f=fn: f(x, y))
    for mode in ("full", "same", "valid", "x"):
        for name in ("convolve", "correlate"):
            add("strsurface", "sliding_mode", params={"name": name, "mode": mode}, operands=[U1, F3],
                layout=f"U1,F3,{mode}", phase="compute", call=lambda x, y, n=name, m=mode: getattr(np, n)(x, y, mode=m))
    # np.dot with out=: the out check runs after the 0-d / alignment checks and before the itemsize refusal.
    for key, a, b, out in (("U2,F22,F3", U2, num_operand(np.ones((2, 2))), target(np.zeros(3))),
                           ("U2,F22,F22", U2, num_operand(np.ones((2, 2))), target(np.zeros((2, 2)))),
                           ("U2,F22,I22", U2, num_operand(np.ones((2, 2))), target(np.zeros((2, 2), np.int64))),
                           ("U2,F22,U22", U2, num_operand(np.ones((2, 2))), target(np.zeros((2, 2), "<U3"))),
                           ("U2,F3,F22", U2, F3, target(np.zeros((2, 2)))),
                           ("U0,F3,F3", U0, F3, target(np.zeros(3))),
                           ("F22,F22,U22", num_operand(np.ones((2, 2))), num_operand(np.ones((2, 2))),
                            target(np.zeros((2, 2), "<U32")))):
        add("strsurface", "dot_out", operands=[a, b, out], layout=key, phase="compute", mutate=2,
            call=lambda x, y, o: np.dot(x, y, out=o))

    # ---- random: SeedSequence over text arrays (pool + generate_state), spawn keys, default_rng, Philox counters ----
    def u_base(values):
        return np.array(values)

    seed_cases = [
        ("U-12-0x10", _text(["12", "0x10"])), ("U-2d", plain(np.array([["1", "2"], ["3", "4"]]))),
        ("U-2dF", _view_operand(u_base(["1", "3", "2", "4"]), lambda b: b.reshape(2, 2, order="F"))),
        ("U-2dT", _view_operand(u_base(["1", "2", "3", "4"]), lambda b: b.reshape(2, 2).T)),
        ("U-rev", _view_operand(u_base(["1", "2", "3"]), lambda b: b[::-1])),
        ("U-step", _view_operand(u_base(["1", "2", "3", "4"]), lambda b: b[::2])),
        ("U-bcast", _view_operand(u_base(["7"]), lambda b: np.broadcast_to(b, (3,)))),
        ("U-empty-str", _text([""])), ("U-lead-sp", _text(["  5"])), ("U-trail-sp", _text(["5 "])),
        ("U-neg", _text(["-5"])), ("U-plus", _text(["+5"])), ("U-0x", _text(["0x"])), ("U-1a", _text(["1a"])),
        ("U-quote", _text(["1it's"])), ("U-long", _text(["1" + "z" * 300])), ("U-longhex", _text(["0x" + "z" * 300])),
        ("U-0X10", _text(["0X10"])), ("U-under", _text(["1_000"])), ("U-big", _text(["99999999999999999999999"])),
        ("U-0", _text(["0"])), ("U-0d", plain(np.array("12"))), ("U-none", _text([], "<U3")),
        ("U-2x0", plain(np.zeros((2, 0), "<U3"))), ("U-tab", _text(["1\t"])), ("U-nl", _text(["1\n2"])),
        ("U-arabic", _text(["٣"])), ("U-fullwidth", _text(["１２"])),
        ("S-ab-c", _text([b"ab", b"c"])), ("S-empty-str", _text([b""])), ("S-nul", _text([b"a\x00b"])),
        ("S-ff", _text([b"\xff"])), ("S-2d", plain(np.array([[b"ab", b"c"], [b"d", b""]]))),
        ("S-0d", plain(np.array(b"12"))), ("S-none", _text([], "S3")), ("S-trailnul", _text([b"a\x00\x00", b"bc"])),
    ]
    for key, x in seed_cases:
        add("strsurface", "seed_seq", operands=[x], layout=key, phase="compute",
            call=lambda v: (lambda ss: np.concatenate([ss.pool, ss.generate_state(4)]))(np.random.SeedSequence(v)))
    for key, x in (("U", _text(["1", "2"])), ("S", _text([b"ab"])), ("U0", plain(np.array("1"))),
                   ("U-1a", _text(["1a"]))):
        add("strsurface", "seed_spawn", params={"list": False}, operands=[x], layout=key, phase="compute",
            call=lambda v: np.random.SeedSequence(5, spawn_key=v).generate_state(4))
    add("strsurface", "seed_spawn", params={"list": True}, operands=[_text(["3"])], layout="list-U", phase="compute",
        call=lambda v: np.random.SeedSequence(5, spawn_key=[v]).generate_state(4))
    for key, x in (("U", _text(["12"])), ("U-1a", _text(["1a"])), ("S", _text([b"12"]))):
        add("strsurface", "default_rng", operands=[x], layout=key, phase="compute",
            call=lambda v: np.random.default_rng(v).random(3))
    for key, x in (("U0-12", plain(np.array("12"))), ("U0-1a", plain(np.array("1a"))), ("U0-sp", plain(np.array(" 7 "))),
                   ("U0-0x10", plain(np.array("0x10"))), ("U0-under", plain(np.array("1_0"))),
                   ("U0-neg", plain(np.array("-1"))), ("U0-empty", plain(np.array(""))),
                   ("U0-arabic", plain(np.array("1٢"))), ("S0-12", plain(np.array(b"12"))),
                   ("S0-1a", plain(np.array(b"1a"))), ("S0-sp", plain(np.array(b" 7 ")))):
        add("strsurface", "philox_counter", operands=[x], layout=key, phase="compute",
            call=lambda v: np.random.Philox(1, counter=v).state["state"]["counter"])

    # ---- random: integer bounds that are text (small dtypes through the comparison checks, 64-bit through int()) ----
    small = ["int8", "int16", "int32", "uint8", "uint16", "uint32", "bool"]
    small_bounds = [("U,H", _text(["1", "5"]), num_operand(np.array([9, 10]))),
                    ("I,U", num_operand(np.array([0, 1])), _text(["1", "5"])),
                    ("S,H", _text([b"1", b"5"]), num_operand(np.array([9, 10]))),
                    ("I,S", num_operand(np.array([0, 1])), _text([b"1", b"5"])),
                    ("U,U", _text(["1", "5"]), _text(["1", "5"])),
                    ("U0,H", plain(np.array("3")), num_operand(np.array([9, 10]))),
                    ("Ue,He", _text([], "<U1"), num_operand(np.array([], dtype=np.int64)))]
    apis = {
        "legacy": lambda lo, hi, dt: np.random.RandomState(1).randint(lo, hi, dtype=dt),
        "legacy_size0": lambda lo, hi, dt: np.random.RandomState(1).randint(lo, hi, size=0, dtype=dt),
        "gen": lambda lo, hi, dt: np.random.default_rng(1).integers(lo, hi, dtype=dt),
        "gen_closed": lambda lo, hi, dt: np.random.default_rng(1).integers(lo, hi, dtype=dt, endpoint=True),
    }
    for dt in small:
        for key, lo, hi in small_bounds:
            for api, fn in apis.items():
                add("strsurface", "randint", params={"api": api, "dtype": dt}, operands=[lo, hi], layout=f"{dt}:{key}",
                    phase="compute", call=lambda l, h, f=fn, d=dt: f(l, h, d))
    H = num_operand(np.array([9, 10]))
    wide_bounds = [
        ("0d U 5,9", plain(np.array("5")), plain(np.array("9"))), ("0d U 1a,9", plain(np.array("1a")), plain(np.array("9"))),
        ("0d U 5,1a", plain(np.array("5")), plain(np.array("1a"))), ("0d S 5,9", plain(np.array(b"5")), plain(np.array(b"9"))),
        ("0d S 1a,9", plain(np.array(b"1a")), plain(np.array(b"9"))),
        ("0d U sp7sp,9", plain(np.array(" 7 ")), plain(np.array("9"))),
        ("0d U arabic,99", plain(np.array("1٢")), plain(np.array("99"))),
        ("0d U 0x10,99", plain(np.array("0x10")), plain(np.array("99"))),
        ("0d U 9,5", plain(np.array("9")), plain(np.array("5"))), ("0d U empty,9", plain(np.array("")), plain(np.array("9"))),
        ("0d U 5,i9", plain(np.array("5")), num_operand(np.array(9))),
        ("1d U 1-5,H", _text(["1", "5"]), H), ("1d U 1a-5,H", _text(["1a", "5"]), H),
        ("1d S 1-5,H", _text([b"1", b"5"]), H), ("1d S 1a-5,H", _text([b"1a", b"5"]), H),
        ("1d I,U 9-10", num_operand(np.array([0, 1])), _text(["9", "10"])),
        ("1d I,U 9-1a", num_operand(np.array([0, 1])), _text(["9", "1a"])),
        ("1d I,U 0-10", num_operand(np.array([0, 1])), _text(["0", "10"])),
        ("1d U huge", _text(["99999999999999999999"]), H), ("1d U neg", _text(["-1", "0"]), H),
        ("1d U 5-9,H5-10", _text(["5", "9"]), num_operand(np.array([5, 10]))),
        ("2d U F", _view_operand(u_base(["1", "3", "2", "4"]), lambda b: b.reshape(2, 2, order="F")),
         num_operand(np.array([[9, 9], [9, 9]]))),
        ("1d U bcast", _view_operand(u_base(["3"]), lambda b: np.broadcast_to(b, (2,))), H),
        ("1d U empty", _text([], "<U1"), num_operand(np.array([], dtype=np.int64))),
    ]
    for dt in ("int64", "uint64", "int32", "uint8"):
        for key, lo, hi in wide_bounds:
            for api in ("legacy", "gen", "gen_closed"):
                fn = apis[api]
                add("strsurface", "randint", params={"api": api, "dtype": dt}, operands=[lo, hi], layout=f"{dt}:{key}",
                    phase="compute", call=lambda l, h, f=fn, d=dt: f(l, h, d))
    for key, lo, dt in (("U int64", _text(["5", "9"]), "int64"), ("U0 int64", plain(np.array("5")), "int64"),
                        ("U0 int32", plain(np.array("5")), "int32"), ("U0 uint8", plain(np.array("5")), "uint8")):
        add("strsurface", "randint_onearg", params={"dtype": dt}, operands=[lo], layout=key, phase="compute",
            call=lambda l, d=dt: np.random.RandomState(1).randint(l, dtype=d))

    # ---- random: choice's p meets the 'safe' cast to float64 before its ndim / size checks ----
    for key, p in (("U3", _text(["0.2", "0.3", "0.5"])), ("S3", _text([b"0.2", b"0.3", b"0.5"])),
                   ("U0", plain(np.array("1"))), ("Ue", _text([], "<U1")), ("U2d", plain(np.array([["0.5", "0.5"]]))),
                   ("C3", num_operand(np.array([0.2 + 0j, 0.3 + 0j, 0.5 + 0j])))):
        for api in ("gen", "legacy"):
            add("strsurface", "choice_p", params={"api": api}, operands=[p], layout=f"{api}:{key}", phase="compute",
                call=lambda pp, a=api: (np.random.default_rng(1) if a == "gen" else np.random.RandomState(1)).choice(3, p=pp))

    # ---- the histogram family over text data, bins, weights and ranges (the probe matrix) ----
    hist_a = {"U": lambda: _text(["ab", "c", "de"]), "S": lambda: _text([b"ab", b"c", b"de"]),
              "Ue": lambda: _text([], "<U2"), "U2d": lambda: plain(np.array([["ab", "c"], ["de", "f"]])),
              "U1": lambda: _text(["ab"]), "F": lambda: num_operand(np.array([1.5, 2.0, 3.0]))}
    hist_bins = {"10": 10, "0": 0, "auto": "auto", "bogus": "bogus", "Fbins": np.array([0.5, 1.0]),
                 "Fdec": np.array([1.0, 0.5]), "Ubins": np.array(["a", "b"]), "Udec": np.array(["b", "a"]),
                 "U1bins": np.array(["a"]), "Uebins": np.zeros(0, "<U1"), "F2d": np.array([[0.5, 1.0]]),
                 "U2dbins": np.array([["a", "b"]]), "I0d": np.array(3), "U0d": np.array("3")}
    hist_range = {"none": None, "01": (0.0, 1.0), "10r": (1.0, 0.0), "inf": (0.0, float("inf"))}

    def weights_for(kind, a):
        return {"ones": lambda: np.ones(a.shape), "bad": lambda: np.ones(7),
                "U": lambda: np.full(a.shape, "w")}[kind]()

    def bins_operand(b):
        # ``target``: the replay rebuilds a 1-D string (or empty) operand as its own base and any other as a view, so
        # NumPy must get the same kind of object for the edges it returns as-is to report the same flags.
        return target(b)

    for fn in ("histogram_bin_edges", "histogram"):
        for ak, mk in hist_a.items():
            for bk, bins in hist_bins.items():
                for rk, rng in hist_range.items():
                    for wk in ("none", "ones", "bad", "U"):
                        for dens in ((False,) if fn == "histogram_bin_edges" else (False, True)):
                            if (rk != "none" and wk != "none") or (dens and (wk != "none" or rk != "none")):
                                continue
                            a = mk()
                            ops = [a]
                            params = {"fn": fn, "range": _json_floats(rng) if rng else None, "density": dens}
                            if isinstance(bins, np.ndarray):
                                params["bins_op"] = len(ops)
                                ops.append(bins_operand(bins))
                            else:
                                params["bins"] = bins
                            if wk != "none":
                                params["weights_op"] = len(ops)
                                ops.append(num_operand(weights_for(wk, a[1])) if wk != "U" else plain(weights_for(wk, a[1])))

                            # The call reads the REAL range (rr): params carry inf as the string "inf" for JSON.
                            def call(*views, p=params, rr=rng):
                                x = views[0]
                                b = views[p["bins_op"]] if "bins_op" in p else p["bins"]
                                w = views[p["weights_op"]] if "weights_op" in p else None
                                if p["fn"] == "histogram_bin_edges":
                                    return np.histogram_bin_edges(x, b, rr, w)
                                return np.histogram(x, b, rr, p["density"], w)
                            add("strsurface", "hist", params=params, operands=ops,
                                layout=f"{fn}:a={ak}:bins={bk}:range={rk}:w={wk}:d={dens}",
                                phase=_bins_phase(bins), call=call)
    for fn in ("histogram2d", "histogramdd"):
        for ak in ("U", "S", "Ue", "F"):
            for bk in ("10", "Fbins", "Ubins", "U1bins", "I0d", "U0d"):
                for rk in ("none", "01"):
                    a = hist_a[ak]()
                    y = num_operand(np.array([1.0, 2.0, 3.0]) if a[1].size == 3 else np.zeros(a[1].shape))
                    bins = hist_bins[bk]
                    ops = [a, y]
                    params = {"fn": fn, "range": _json_floats(hist_range[rk]) if hist_range[rk] else None}
                    if isinstance(bins, np.ndarray):
                        params["bins_op"] = len(ops)
                        ops.append(bins_operand(bins))
                    else:
                        params["bins"] = bins

                    def call2(*views, p=params, r=hist_range[rk]):
                        x, yy = views[0], views[1]
                        b = views[p["bins_op"]] if "bins_op" in p else p["bins"]
                        if p["fn"] == "histogram2d":
                            return np.histogram2d(x, yy, b, (r, r) if r else None)
                        bb = [b, b] if isinstance(b, np.ndarray) and b.ndim == 1 else b
                        H, edges = np.histogramdd([x, yy], bb, (r, r) if r else None)
                        return (H,) + tuple(edges)
                    add("strsurface", "hist", params=params, operands=ops, layout=f"{fn}:a={ak}:bins={bk}:range={rk}",
                        phase=_bins_phase(bins), call=call2)

    # ---- histogramdd's bin specifications (mostly numeric; the per-dimension scalar checks the text work ported) ----
    # "dd": np.histogramdd([x, y], bins) with an int or a list spec; "2d": np.histogram2d(x, y, bins) with an ARRAY spec,
    # which NumPy hands to histogramdd unchanged when it has 2 entries (and as both dimensions' edges otherwise). A 0-d
    # array spec of histogramdd is D copies of it (NumPy's `bins = D*[bins]` after len() fails), spelled so in C#.
    hx = num_operand(np.array([1.0, 2.0, 3.0, 2.5]))
    hy = num_operand(np.array([1.0, 2.0, 3.0, 0.5]))
    hdd_cases = [
        ("dd", "int 2", 2), ("dd", "ints 2-3", [2, 3]), ("dd", "0d 3", np.array(3)),
        ("dd", "edges list", [np.array([0., 1, 2, 3]), np.array([0., 2, 4])]), ("dd", "mixed", [2, np.array([0., 2, 4])]),
        ("dd", "py float", [2.5, 3]), ("dd", "py bool", [True, 3]), ("dd", "0d bool", [np.array(True), 3]),
        ("dd", "zero", [0, 3]), ("dd", "None", [None, 3]), ("dd", "str", ["a", 3]), ("dd", "py complex", [1 + 0j, 3]),
        ("dd", "decreasing", [np.array([1., 0]), 2]), ("dd", "0d text", [np.array("2"), 3]),
        ("dd", "0d bytes", [np.array(b"2"), 3]), ("dd", "0d text edges", [np.array(["2"]), 3]),
        ("dd", "text edges", [np.array(["1", "2"]), 3]),
        ("2d", "1d ints", np.array([2, 3])), ("2d", "1d floats", np.array([2.0, 3.0])),
        ("2d", "1d uint8", np.array([2, 3], np.uint8)), ("2d", "1d complex", np.array([2 + 0j, 3])),
        ("2d", "2 rows", np.array([[0., 1, 2, 3], [0, 2, 4, 6]])), ("2d", "shared", np.array([0., 1, 2, 3])),
        ("2d", "0d", np.array(3)), ("2d", "1 elem", np.array([3])), ("2d", "1d text", np.array(["2", "3"])),
        ("2d", "1d bytes", np.array([b"2", b"3"])), ("2d", "1d bool", np.array([True, True])),
    ]
    for fn, key, spec in hdd_cases:
        ops = [hx, hy]
        enc_spec = _encode_bins(spec, ops)
        if fn == "dd":
            # NumPy gets the spec rebuilt over the recorded operand VIEWS (the objects the C# replay passes), so an
            # edges array it returns is the same object in both.
            call = (lambda x, yy, *rest, e=enc_spec: (lambda r: (r[0],) + tuple(r[1]))(
                np.histogramdd([x, yy], _decode_bins(e, (x, yy) + rest))))
        else:
            call = lambda x, yy, b: np.histogram2d(x, yy, b)
        add("strsurface", "hdd_bins", params={"fn": fn, "bins": enc_spec}, operands=ops, layout=f"{fn} {key}",
            phase=_bins_phase(spec), call=call)

    # ---- quantile q / percentile q over text, and text data with numeric q ----
    F = num_operand(np.array([1.5, 2.0, 3.0]))
    for name in ("percentile", "nanpercentile", "quantile", "nanquantile"):
        for key, a, q in (("F,U", F, _text(["ab", "c", "de"])), ("F,S", F, _text([b"ab", b"c", b"de"])),
                          ("F,U2d", F, plain(np.array([["ab", "c"], ["de", "f"]]))), ("F,U0", F, plain(np.array("ab"))),
                          ("F,U12", F, _text(["a"] * 12)), ("F,Ue", F, _text([], "<U1")),
                          ("U,U", _text(["ab", "c", "de"]), _text(["ab", "c", "de"])),
                          ("Ue,Ue", _text([], "<U1"), _text([], "<U1")), ("Ue,F", _text([], "<U1"), num_operand(np.array([0.5])))):
            add("strsurface", "quantile_q", params={"name": name}, operands=[a, q], layout=key, phase="compute",
                call=lambda x, qq, n=name: getattr(np, n)(x, qq))

    # ---- reductions and scans with out=: the loop resolved on (out, input) names both descriptors ----
    for name in ("max", "min", "amax", "amin", "nanmax", "nanmin", "ptp", "median", "nanmedian", "cumsum", "cumprod",
                 "nancumsum", "nancumprod"):
        scan = name.startswith(("cum", "nancum"))
        oshape = (3,) if scan else ()
        for key, a, out in (("U,F", _text(["ab", "c", "de"]), target(np.zeros(oshape))),
                            ("U,U4", _text(["ab", "c", "de"]), target(np.zeros(oshape, "<U4"))),
                            ("F,U4", F, target(np.zeros(oshape, "<U4"))),
                            ("U,I", _text(["ab", "c", "de"]), target(np.zeros(oshape, np.int64))),
                            ("S,S4", _text([b"ab", b"c", b"de"]), target(np.zeros(oshape, "S4")))):
            add("strsurface", "reduce_out", params={"name": name}, operands=[a, out], layout=key, phase="compute",
                mutate=1, call=lambda x, o, n=name: getattr(np, n)(x, out=o))

    # ---- arg-reductions with out=: the out shape, then the out's safe cast to intp, before the computation ----
    U = _text(["ab", "c", "de"])
    U2_ = plain(np.array([["ab", "c", "x"], ["de", "f", "y"]]))
    for key, name, a, axis, out in (("nanargmax,U,o3", "nanargmax", U, None, target(np.zeros(3, np.int64))),
                                    ("nanargmax,U,o0", "nanargmax", U, None, target(np.zeros((), np.int64))),
                                    ("nanargmax,U,of", "nanargmax", U, None, target(np.zeros((), np.float64))),
                                    ("nanargmin,U2,ax0,o3", "nanargmin", U2_, 0, target(np.zeros(4, np.int64))),
                                    ("nanargmax,Ue,o0", "nanargmax", _text([], "<U1"), None, target(np.zeros((), np.int64))),
                                    ("nanargmin,U1,o0", "nanargmin", _text(["ab"]), None, target(np.zeros((), np.int64)))):
        add("strsurface", "argreduce_out", params={"name": name, "axis": axis}, operands=[a, out], layout=key,
            phase="compute", mutate=1, call=lambda x, o, n=name, ax=axis: getattr(np, n)(x, axis=ax, out=o))
    for name in ("nanargmax", "nanargmin"):
        add("strsurface", "argreduce_out", params={"name": name, "axis": 5}, operands=[U], layout=f"{name},U,ax5",
            phase="compute", call=lambda x, n=name: getattr(np, n)(x, axis=5))

    # ---- polynomials over text ----
    for key, name, ops in (("poly,U2d", "poly", [plain(np.array([["ab", "c"], ["de", "f"]]))]), ("poly,U", "poly", [U]),
                           ("poly,U3d", "poly", [plain(np.zeros((2, 2, 2), "<U1"))]),
                           ("poly,U2x3", "poly", [plain(np.zeros((2, 3), "<U1"))]),
                           ("roots,U2d", "roots", [plain(np.array([["ab", "c"], ["de", "f"]]))]), ("roots,U", "roots", [U]),
                           ("polymul,U,U", "polymul", [U, U]), ("polymul,U,F", "polymul", [U, F]),
                           ("polymul,F,U", "polymul", [F, U]), ("polymul,Ue,F", "polymul", [_text([], "<U1"), F]),
                           ("polymul,U2d,F", "polymul", [plain(np.array([["ab", "c"], ["de", "f"]])), F]),
                           ("polyadd,U,U", "polyadd", [U, U]), ("polyval,U,F", "polyval", [U, F]),
                           ("polyval,F,U", "polyval", [F, U]), ("polyder,U", "polyder", [U]), ("polyint,U", "polyint", [U]),
                           ("polydiv,U,F", "polydiv", [U, F])):
        add("strsurface", "poly", params={"name": name}, operands=ops, layout=key, phase="compute",
            call=lambda *xs, n=name: getattr(np, n)(*xs))

    # ---- emath (the complex-result wrappers): the domain checks meet the text first ----
    for name in ("sqrt", "log", "log2", "log10", "arccos", "arcsin", "arctanh"):
        for key, x in (("U", U), ("U0", plain(np.array("ab"))), ("Ue", _text([], "<U1")), ("S", _text([b"ab"]))):
            add("strsurface", "emath", params={"name": name}, operands=[x], layout=key, phase="compute",
                call=lambda v, n=name: getattr(np.emath, n)(v))
    for key, a, b in (("U,F", U, F), ("F,U", F, U)):
        for name in ("logn", "power"):
            add("strsurface", "emath", params={"name": name}, operands=[a, b], layout=key, phase="compute",
                call=lambda x, y, n=name: getattr(np.emath, n)(x, y))

    # ---- index conversions: only integer (or bool) index arrays ----
    for key, x, shape in (("U", U, [3]), ("F", F, [3]), ("U-neg-shape", U, [-1]), ("B", num_operand(np.array([True, False])), [3]),
                          ("Ue", _text([], "<U1"), [3]), ("S", _text([b"1"]), [3])):
        add("strsurface", "unravel_ix", params={"shape": shape}, operands=[x], layout=key, phase="compute",
            call=lambda v, s=shape: np.unravel_index(v, tuple(s)))
    for key, xs, dims in (("U", [U], [3]), ("F", [F], [3]), ("U,U-len", [U, U], [3]), ("Ue", [_text([], "<U1")], [3])):
        add("strsurface", "ravel_mi", params={"dims": dims}, operands=xs, layout=key, phase="compute",
            call=lambda *v, d=dims: np.ravel_multi_index(tuple(v), tuple(d)))

    # ---- dtype predicates over arrays ----
    B = num_operand(np.array([True]))
    I = num_operand(np.array([1, 2, 3]))
    for key, ops in (("U", [U]), ("S", [_text([b"ab"])]), ("B", [B]), ("F,U", [F, U]), ("I", [I]), ("F", [F]),
                     ("I,F32", [I, num_operand(np.array([1.0], np.float32))])):
        add("strsurface", "common_type", operands=ops, layout=key, phase="compute",
            call=lambda *xs: np.dtype(np.common_type(*xs)))
    for key, x in (("U", U), ("S", _text([b"ab"])), ("U0", plain(np.array("ab"))), ("Ue", _text([], "<U3"))):
        add("strsurface", "min_scalar_type", operands=[x], layout=key, phase="compute", call=lambda v: np.min_scalar_type(v))
    for key, x, to, casting in (("U-U1-safe", U, "<U1", "safe"), ("U-U2-safe", U, "<U2", "safe"),
                                ("U-S2-same", U, "S2", "same_kind"), ("U-f8-unsafe", U, "f8", "unsafe"),
                                ("U-f8-safe", U, "f8", "safe"), ("S-U2-safe", _text([b"ab"]), "<U2", "safe"),
                                ("F-U32-safe", F, "<U32", "safe"), ("F-U31-safe", F, "<U31", "safe"),
                                ("I-U21-safe", I, "<U21", "safe"), ("B-U5-safe", B, "<U5", "safe"),
                                ("U0-U2-equiv", plain(np.array("ab")), "<U2", "equiv")):
        add("strsurface", "can_cast_arr", params={"to": to, "casting": casting}, operands=[x], layout=key,
            phase="compute", call=lambda v, t=to, c=casting: np.can_cast(v, t, c))

    # ---- ediff1d: to_begin / to_end meet the same_kind check before the subtraction ----
    for key, ary, to_end, to_begin in (("U", U, None, None), ("Ue", _text([], "<U1"), None, None),
                                       ("F,end=U", F, U, None), ("F,begin=U", F, None, U),
                                       ("U,end=F", U, F, None), ("U,end=U", U, _text(["x"]), None),
                                       ("U0", plain(np.array("ab")), None, None)):
        ops = [ary] + [x for x in (to_end, to_begin) if x is not None]
        params = {"end": 1 if to_end is not None else None,
                  "begin": (2 if to_end is not None else 1) if to_begin is not None else None}
        add("strsurface", "ediff1d", params=params, operands=ops, layout=key, phase="compute",
            call=lambda *v, p=params: np.ediff1d(v[0], to_end=v[p["end"]] if p["end"] else None,
                                                 to_begin=v[p["begin"]] if p["begin"] else None))

    # ---- where= masks that are text: the mask's 'safe' cast to bool fails before anything is read ----
    Fd = num_operand(np.array([1.0, 2.0, 3.0]))
    for mkey, mask in (("U", _text(["a", "", "b"])), ("S", _text([b"a", b"", b"b"])), ("U0", plain(np.array("a")))):
        for name in ("add", "subtract", "sqrt", "less", "logical_and", "max", "copyto"):
            ops = [target(np.array([1.0, 2.0, 3.0])), Fd, mask] if name == "copyto" else [Fd, Fd, mask]
            add("strsurface", "where_mask", params={"name": name}, operands=ops, layout=f"{name}:{mkey}", phase="compute",
                mutate=0 if name == "copyto" else None, call=lambda x, y, m, n=name: _where_call(n, x, y, m))

    # ---- 0-d conversions to Python scalars (the C# explicit conversions) ----
    for key, x, to in (("U0-ab-complex", plain(np.array("ab")), "complex"), ("U0-1-complex", plain(np.array("1")), "complex"),
                       ("S0-1-complex", plain(np.array(b"1")), "complex"), ("U0-ab-float", plain(np.array("ab")), "float"),
                       ("U0-1.5-float", plain(np.array("1.5")), "float"), ("U0-12-int", plain(np.array("12")), "int"),
                       ("U0-ab-int", plain(np.array("ab")), "int")):
        add("strsurface", "to_scalar", params={"to": to}, operands=[x], layout=key, phase="compute",
            call=lambda v, t=to: {"complex": lambda: np.complex128(complex(v)), "float": lambda: np.float64(float(v)),
                                  "int": lambda: np.int64(int(v))}[t]())

    # ---- fill with a sequence: the string target's text has no trailing period ----
    for key, a, v in (("U3,U2", target(np.array(["a", "b", "c"])), _text(["x", "y"])),
                      ("S3,S2", target(np.array([b"a", b"b", b"c"])), _text([b"x", b"y"])),
                      ("U3,F2", target(np.array(["a", "b", "c"])), num_operand(np.array([1.0, 2.0]))),
                      ("F3,F2", target(np.array([1.0, 2.0, 3.0])), num_operand(np.array([1.0, 2.0]))),
                      ("U3,U1", target(np.array(["a", "b", "c"])), _text(["x"]))):
        add("strsurface", "fill_seq", operands=[a, v], layout=key, phase="compute", mutate=0,
            call=lambda t, val: t.fill(val))


def _json_floats(values):
    """Floats as JSON: a finite value as a number, inf / -inf / nan as the strings "inf" / "-inf" / "nan" (JSON has
    no literal for them; the replay reads either form)."""
    return [float(v) if np.isfinite(v) else repr(float(v)) for v in values]


def _where_call(name, x, y, mask):
    """One where= call of section M: a binary / unary / comparison / logical ufunc, np.max with an initial, or copyto."""
    if name == "sqrt":
        return np.sqrt(x, where=mask)
    if name == "max":
        return np.max(x, initial=0, where=mask)
    if name == "copyto":
        return np.copyto(x, y, where=mask)
    return getattr(np, name)(x, y, where=mask)


def _bins_phase(spec):
    """The phase of a histogram call whose bins are ``spec``: 2 when a bins array is TEXT of two or more elements —
    NumPy's monotonicity check compares those elements, so which error (or answer) follows depends on the text; else
    "compute" (NumPy's error is final, its answer the next phase)."""
    items = spec if isinstance(spec, list) else [spec]
    for item in items:
        if isinstance(item, np.ndarray) and item.dtype.kind in "SU" and item.ndim == 1 and item.size >= 2:
            return 2
    return "compute"


def _decode_bins(enc_spec, views):
    """The inverse of ``_encode_bins`` over the recorded operand views: a 0-d array spec becomes its two copies (NumPy's
    own ``D*[bins]``, which the C# replay spells as an object[] of the two)."""
    if isinstance(enc_spec, dict):
        if "op" in enc_spec:
            v = views[enc_spec["op"]]
            return [v, v]
        return enc_spec["int"]
    out = []
    for e in enc_spec:
        if "op" in e:
            out.append(views[e["op"]])
        elif "int" in e:
            out.append(e["int"])
        elif "float" in e:
            out.append(e["float"])
        elif "bool" in e:
            out.append(e["bool"])
        elif "none" in e:
            out.append(None)
        elif "complex" in e:
            out.append(complex(*e["complex"]))
        elif "u" in e:
            out.append(bytes.fromhex(e["u"]).decode("utf-32-le", "surrogatepass"))
        else:
            raise ValueError(f"unknown bins entry {e}")
    return out


def _encode_bins(spec, ops):
    """A histogramdd bin specification as JSON: Python scalars by kind, arrays as appended operands ({"op": index})."""
    if isinstance(spec, np.ndarray):
        ops.append(target(spec))
        return {"op": len(ops) - 1}
    if isinstance(spec, list):
        return [_encode_bins(s, ops) for s in spec]
    if isinstance(spec, np.str_):
        return {"npstr": enc(str(spec))}
    if isinstance(spec, complex):
        return {"complex": [spec.real, spec.imag]}
    return enc(spec)


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# N. numpy.ma over text
# ----------------------------------------------------------------------------------------------------------------------

class MArg:
    """A masked-array argument of a ``strma`` case: data values (with an optional dtype), a mask (``None`` for nomask, a
    Python bool for NumPy's ``mask=True`` / ``mask=False``, or nested values), a fill value and the hard-mask flag."""

    def __init__(self, data, mask=None, spec=None, fill=None, hard=False):
        self.data, self.mask, self.spec, self.fill, self.hard = data, mask, spec, fill, hard


def _enc_value(v):
    """A plain Python argument of a ``strma`` case (a scalar, None, text, bytes or a complex)."""
    if isinstance(v, complex):
        return {"complex": [v.real, v.imag]}
    return enc(v)


def _is_text_arg(a):
    """Whether a ``strma`` argument (an ``MArg``, an array or a Python value) is text."""
    if isinstance(a, MArg):
        data = np.array(a.data, dtype=a.spec) if a.spec else np.array(a.data)
        return data.dtype.kind in "SU"
    if isinstance(a, np.ndarray):
        return a.dtype.kind in "SU"
    return isinstance(a, (str, bytes))


def add_ma(op, *, args, params=None, call, phase="compute", layout=""):
    """Record one numpy.ma call over masked arguments. Each ``MArg`` travels as a data operand plus (unless nomask or a
    bool) a boolean mask operand; ``params["mops"]`` says how to rebuild every argument (``{"d": data, "m": mask}``, a
    plain array ``{"a": i}`` or a value ``{"v": ...}``) and the replay builds the same masked arrays."""
    operands, mops = [], []
    for a in args:
        if isinstance(a, MArg):
            data = np.array(a.data, dtype=a.spec) if a.spec else np.array(a.data)
            entry = {"d": len(operands), "m": -1}
            # ``target``: an operand whose flags the replay reproduces (a 1-D string operand is its own base, a non-empty
            # numeric one a view), since a function may return the very array it was given (getmask, filled).
            operands.append(target(data))
            if isinstance(a.mask, bool):
                entry["mbool"] = a.mask
            elif a.mask is not None:
                # ``target`` like the data: numpy.ma ADOPTS the array passed as ``mask=`` (getmask returns it), so its
                # flags must be ones the replay reproduces (a non-empty bool operand is a view on both sides).
                entry["m"] = len(operands)
                operands.append(target(np.array(a.mask, dtype=bool)))
            if a.fill is not None:
                entry["fill"] = _enc_value(a.fill)
            if a.hard:
                entry["hard"] = True
            mops.append(entry)
        elif isinstance(a, np.ndarray):
            mops.append({"a": len(operands)})
            operands.append(target(a))
        else:
            mops.append({"v": _enc_value(a)})
    p = dict(params or {})
    p["mops"] = mops
    margs = [a for a in args]

    def wrapped(*views):
        objs = []
        for e, a in zip(mops, margs):
            if "d" in e:
                mask = e["mbool"] if "mbool" in e else (views[e["m"]] if e["m"] >= 0 else np.ma.nomask)
                objs.append(np.ma.array(views[e["d"]], mask=mask, fill_value=a.fill, hard_mask=a.hard))
            elif "a" in e:
                objs.append(views[e["a"]])
            else:
                objs.append(a)
        return call(*objs)

    add("strma", op, params=p, operands=operands, call=wrapped, phase=phase, layout=layout)


def gen_strma():
    ma = np.ma
    MU = MArg(["ab", "c", "de"], [1, 0, 0])
    MUn = MArg(["ab", "c", "de"])
    MU2 = MArg([["ab", "c"], ["de", "f"]], [[1, 0], [0, 0]])
    MUe = MArg(np.zeros(0, "<U1"), np.zeros(0, bool))
    MUe2 = MArg(np.zeros((0, 3), "<U1"))
    MS = MArg([b"ab", b"c", b"de"], [1, 0, 0])
    MU0 = MArg(np.array("ab"))
    MU0m = MArg(np.array("ab"), True)
    MUall = MArg(["ab", "c", "de"], True)
    MUf = MArg(["ab", "c", "de"], [1, 0, 0], fill="zz")
    MUh = MArg(["ab", "c", "de"], [1, 0, 0], hard=True)
    MF = MArg([1.5, 2.0, 3.0], [0, 1, 0])
    unary_variants = [("MU", MU), ("MUn", MUn), ("MU2", MU2), ("MUe", MUe), ("MUe2", MUe2), ("MS", MS), ("MU0", MU0),
                      ("MU0m", MU0m), ("MUall", MUall), ("MUf", MUf)]

    # ---- one masked text argument: every function numpy.ma applies to an array alone ----
    unary = {
        # masked ufuncs
        **{n: (lambda f: lambda x: f(x))(getattr(ma, n)) for n in (
            "abs", "absolute", "negative", "fabs", "conjugate", "angle", "around", "floor", "ceil", "exp", "sin", "cos",
            "sinh", "cosh", "tanh", "arctan", "arcsinh", "logical_not", "sqrt", "log", "log2", "log10", "tan", "arcsin",
            "arccos", "arccosh", "arctanh")},
        # reductions, scans, statistics
        **{n: (lambda f: lambda x: f(x))(getattr(ma, n)) for n in (
            "sum", "prod", "product", "min", "max", "amin", "amax", "ptp", "count", "mean", "var", "std", "anom",
            "anomalies", "cumsum", "cumprod", "argmin", "argmax", "all", "any", "alltrue", "sometrue", "median")},
        # masks, data, conversion, movement
        **{n: (lambda f: lambda x: f(x))(getattr(ma, n)) for n in (
            "getdata", "getmask", "getmaskarray", "is_mask", "is_masked", "copy", "ravel", "transpose",
            "squeeze", "atleast_1d", "atleast_2d", "atleast_3d", "compressed", "count_masked", "diff", "ediff1d", "sort",
            "argsort", "unique", "harden_mask", "soften_mask", "asarray", "asanyarray", "array",
            "masked_array", "masked_all_like", "zeros_like", "ones_like", "empty_like", "fix_invalid", "masked_invalid",
            "minimum_fill_value", "maximum_fill_value", "make_mask", "flatten_mask", "filled", "ndim", "size", "diag",
            "diagflat", "trace", "nonzero", "round", "round_", "compress_nd", "compress_rowcols", "compress_rows",
            "compress_cols", "mask_rowcols", "mask_rows", "mask_cols", "flatnotmasked_edges", "vander")},
        "flatten": lambda x: x.flatten(),
        "shrink_mask": lambda x: x.shrink_mask(),
        "shape": lambda x: list(ma.shape(x)),
        "default_fill_value": lambda x: ma.default_fill_value(x.dtype),
        "fill_value": lambda x: x.fill_value,
        "repr": lambda x: repr(x),
        "str": lambda x: str(x),
        "make_mask_noshrink": lambda x: ma.make_mask(x, shrink=False),
        "astype_U1": lambda x: x.astype("U1"),
        "astype_S2": lambda x: x.astype("S2"),
        "astype_f8": lambda x: x.astype("f8"),
        "set_fill_q": lambda x: (setattr(x, "fill_value", "qqq"), x)[1],
        "set_fill_none": lambda x: (setattr(x, "fill_value", None), x)[1],
        "set_fill_5": lambda x: (setattr(x, "fill_value", 5), x)[1],
        "set_fill_value_q": lambda x: (ma.set_fill_value(x, "qqq"), x)[1],
        "filled_q": lambda x: x.filled("q"),
        "filled_5": lambda x: x.filled(5),
        "filled_bytes": lambda x: x.filled(b"Q"),
        "print_disabled": lambda x: _ma_str_disabled(x),
        "masked_object_ab": lambda x: ma.masked_object(x, "ab"),
        "masked_values_ab": lambda x: ma.masked_values(x, "ab"),
        "masked_values_15": lambda x: ma.masked_values(x, 1.5),
        "masked_values_1": lambda x: ma.masked_values(x, 1),
        "masked_object_15": lambda x: ma.masked_object(x, 1.5),
        "masked_object_true": lambda x: ma.masked_object(x, True),
        "masked_equal_15": lambda x: ma.masked_equal(x, 1.5),
        "masked_greater_1": lambda x: ma.masked_greater(x, 1),
        "masked_less_15": lambda x: ma.masked_less(x, 1.5),
        "masked_inside_12": lambda x: ma.masked_inside(x, 1.0, 2.0),
        "masked_inside_21": lambda x: ma.masked_inside(x, 2, 1),
        "masked_outside_12": lambda x: ma.masked_outside(x, 1.0, 2.0),
        "masked_inside_ab": lambda x: ma.masked_inside(x, "a", "b"),
        "masked_outside_ab": lambda x: ma.masked_outside(x, "b", "a"),
        "masked_inside_mix": lambda x: ma.masked_inside(x, 1.0, "b"),
        "masked_inside_bytes": lambda x: ma.masked_inside(x, b"a", b"b"),
        "masked_inside_complex": lambda x: ma.masked_inside(x, 1j, 2j),
        "putmask_text": lambda x: (ma.putmask(x, np.ones(x.shape, bool), "xyz"), x)[1],
        "putmask_num": lambda x: (ma.putmask(x, np.ones(x.shape, bool), 5), x)[1],
        "putmask_none": lambda x: (ma.putmask(x, np.zeros(x.shape, bool), 1.5), x)[1],
        "putmask_textmask": lambda x: (ma.putmask(x, np.asarray(x.data), "q"), x)[1],
        "putmask_intmask": lambda x: (ma.putmask(x, np.zeros(x.shape, np.int64), "q"), x)[1],
        "putmask_masked": lambda x: (ma.putmask(x, np.ones(x.shape, bool), ma.array(np.full(x.shape, "y"), mask=True)), x)[1],
        "average_textw": lambda x: ma.average(np.ones(x.shape), weights=x),
        "average_numw": lambda x: ma.average(x, weights=np.ones(x.shape)),
        "average_short": lambda x: ma.average(x, weights=np.ones(1)),
        "average_ax0": lambda x: ma.average(x, axis=0, weights=np.ones(x.shape[0] if x.ndim else 1)),
        "average_ax5": lambda x: ma.average(x, axis=5),
        "masked_where_self": lambda x: ma.masked_where(x, np.zeros(x.shape)),
        "masked_where_short": lambda x: ma.masked_where(np.array([True, False]), x),
        "masked_where_0d": lambda x: ma.masked_where(np.array(True), x),
        "masked_where_bool": lambda x: ma.masked_where(np.zeros(x.shape, bool), x),
        "mask_or_bool": lambda x: ma.mask_or(x, np.zeros(x.shape, bool)),
        "mask_or_self": lambda x: ma.mask_or(x, x),
        "mask_or_false": lambda x: ma.mask_or(False, x),
        "mask_or_nomask": lambda x: ma.mask_or(ma.nomask, x),
        "compress_self": lambda x: ma.compress(x, np.arange(x.shape[0] if x.ndim else 1)),
        "fix_invalid_textmask": lambda x: ma.fix_invalid(np.zeros(x.shape), mask=np.asarray(x.data)),
        "fix_invalid_mask2": lambda x: ma.fix_invalid(x, mask=[True, False]),
        "common_fill_self": lambda x: ma.common_fill_value(x, x),
        "common_fill_float": lambda x: ma.common_fill_value(x, ma.array([1.0])),
        "take_0": lambda x: ma.take(x, np.array([0])),
        "repeat_2": lambda x: ma.repeat(x, 2),
        "reshape_flat": lambda x: ma.reshape(x, (x.size,)),
        "resize_4": lambda x: ma.resize(x, (4,)),
        "expand_dims_0": lambda x: ma.expand_dims(x, 0),
        "concatenate_self": lambda x: ma.concatenate([x, x]),
        "stack_self": lambda x: ma.stack([x, x]),
        "hstack_self": lambda x: ma.hstack([x, x]),
        "vstack_self": lambda x: ma.vstack([x, x]),
        "clip_ab": lambda x: ma.clip(x, "a", "b"),
        "clip_num": lambda x: ma.clip(x, 0, 1),
        "where_self": lambda x: ma.where(np.ones(x.shape, bool), x, x),
        "where_textcond": lambda x: ma.where(x, 1, 2),
        "sort_end_false": lambda x: ma.sort(x, endwith=False),
        "count_ax0": lambda x: ma.count(x, axis=0),
    }
    # np.empty_like / masked_all_like leave the data uninitialized: an S buffer is whatever the allocator recycled.
    unary["empty_like"] = lambda x: _UninitData(ma.empty_like(x))
    unary["masked_all_like"] = lambda x: _UninitData(ma.masked_all_like(x))
    for name, fn in unary.items():
        for key, arg in unary_variants:
            add_ma("ma1", params={"name": name}, args=[arg], layout=key, call=lambda x, f=fn: f(x))
    for key, arg in (("MUh", MUh), ("MU2", MU2)):
        add_ma("ma1", params={"name": "putmask_text"}, args=[arg], layout=key, call=lambda x: unary["putmask_text"](x))

    # ---- two arguments: the binary ufuncs, products, set routines, mask combination ----
    F = np.array([1.5, 2.0, 3.0])
    Ue1 = np.zeros(0, "<U1")
    binary = {n: (lambda f: lambda x, y: f(x, y))(getattr(ma, n)) for n in (
        "add", "subtract", "multiply", "divide", "true_divide", "floor_divide", "mod", "remainder", "fmod", "power",
        "hypot", "arctan2", "maximum", "minimum", "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
        "logical_and", "logical_or", "logical_xor", "bitwise_and", "bitwise_or", "bitwise_xor", "left_shift",
        "right_shift", "allequal", "allclose", "inner", "innerproduct", "outer", "outerproduct", "dot", "convolve",
        "correlate", "append", "isin", "in1d", "intersect1d", "union1d", "setdiff1d", "setxor1d", "mask_or",
        "common_fill_value", "masked_where", "masked_equal", "masked_not_equal", "masked_greater",
        "masked_greater_equal", "masked_less", "masked_less_equal", "masked_values", "masked_object", "compress",
        "filled")}
    binary["average"] = lambda x, y: ma.average(x, weights=y)
    # The bytes operand is a 0-d S array, not Python bytes: every function here reads its second argument as an
    # array_like (np.asarray(b'ab') is that same 0-d S2 array), and C#'s spelling of it is an NDArray -- a C# byte[] in an
    # array_like position is a uint8 ARRAY (the house map; byte[] is bytes only in VALUE positions: fills, compared
    # values, bounds). mask_or alone sees the difference (Python bytes has no dtype: 'Incompatible dtypes |S2<>None').
    pairs = [("MU,MU", MU, MU), ("MU,MUn", MU, MUn), ("MU,F", MU, F), ("F,MU", F, MU), ("MF,MU", MF, MU),
             ("MUe,MUe", MUe, MUe), ("MUe,Ue", MUe, Ue1), ("MU,'ab'", MU, "ab"), ("MU,1.5", MU, 1.5), ("MU,1", MU, 1),
             ("MU,True", MU, True), ("MS,MS", MS, MS), ("MU,MS", MU, MS), ("MS,b'ab'", MS, np.array(b"ab")),
             ("MUe,1.5", MUe, 1.5), ("MU2,MU2", MU2, MU2)]
    for name, fn in binary.items():
        for key, x, y in pairs:
            # ma.compress(condition, a): a TEXT condition selects the indices of its non-empty strings, so every outcome
            # (an IndexError included) follows from the text -- the computational phase.
            phase = 2 if name == "compress" and _is_text_arg(x) else "compute"
            add_ma("ma2", params={"name": name}, args=[x, y], layout=key, phase=phase, call=lambda a, b, f=fn: f(a, b))

    # ---- constructors: text masks, fill values, mask reconciliation ----
    U3 = np.array(["ab", "c", "de"])
    S3 = U3.astype("S2")
    ctor = [
        ("text_mask", lambda: ma.array(np.arange(3), mask=np.array(["a", "", "b"]))),
        ("text_mask_empty", lambda: ma.array(np.zeros(0), mask=np.zeros(0, "<U1"))),
        ("text_mask_size", lambda: ma.array(np.arange(3), mask=np.array(["a", ""]))),
        ("text_mask_one", lambda: ma.array(np.arange(3), mask=np.array(["a"]))),
        ("mask_size", lambda: ma.array(U3, mask=[True, False])),
        ("mask_one", lambda: ma.array(U3, mask=[True])),
        ("mask_reshape", lambda: ma.array(np.array([["ab", "c", "x"], ["de", "f", "y"]]), mask=np.zeros((3, 2), bool))),
        ("mask_true", lambda: ma.array(U3, mask=True)),
        ("mask_false", lambda: ma.array(U3, mask=False)),
        ("keep_mask", lambda: ma.array(ma.array(U3, mask=[1, 0, 0]), mask=[0, 1, 0])),
        ("keep_mask_false", lambda: ma.array(ma.array(U3, mask=[1, 0, 0]), mask=[0, 1, 0], keep_mask=False)),
        ("fill_hello", lambda: ma.array(U3, fill_value="hello")),
        ("fill_5", lambda: ma.array(U3, fill_value=5)),
        ("fill_bytes_u", lambda: ma.array(U3, fill_value=b"xyz")),
        ("fill_str_s", lambda: ma.array(S3, fill_value="xyz")),
        ("fill_nonascii_s", lambda: ma.array(S3, fill_value="é")),
        ("fill_nonascii_bytes_u", lambda: ma.array(U3, fill_value=b"\xe9")),
        ("fill_str_float", lambda: ma.array(F, fill_value="abc")),
        ("fill_str_int", lambda: ma.array(np.arange(3), fill_value="abc")),
        ("fill_str_bool", lambda: ma.array(np.array([True]), fill_value="abc")),
        ("fill_bytes_float", lambda: ma.array(F, fill_value=b"1.5")),
        ("fill_array_u", lambda: ma.array(U3, fill_value=np.array(["xyz"]))),
        ("fill_inherit", lambda: ma.array(ma.array(U3, fill_value="zz"))),
        ("fill_inherit_dtype", lambda: ma.array(ma.array(U3, fill_value="zz"), dtype="U1")),
        ("hard_inherit", lambda: ma.array(ma.array(U3, mask=[1, 0, 0], hard_mask=True))),
        ("masked_all_u2", lambda: _UninitData(ma.masked_all((3,), dtype="U2"))),
        ("masked_all_s2", lambda: _UninitData(ma.masked_all((2, 2), dtype="S2"))),
        ("default_fill_u", lambda: ma.default_fill_value(np.dtype("U2"))),
        ("default_fill_s", lambda: ma.default_fill_value(np.dtype("S2"))),
        ("make_mask_text", lambda: ma.make_mask(np.array(["a", ""]))),
        ("make_mask_text_empty", lambda: ma.make_mask(np.zeros(0, "<U1"), shrink=False)),
        ("mask_or_text_bool", lambda: ma.mask_or(U3, np.zeros(3, bool))),
        ("mask_or_bool_text", lambda: ma.mask_or(np.zeros(3, bool), U3)),
        ("mask_or_u_s", lambda: ma.mask_or(U3, S3)),
        ("mask_or_widths", lambda: ma.mask_or(U3, np.array(["a", "b", "c"]))),
        ("mask_or_empty", lambda: ma.mask_or(np.zeros(0, "<U1"), np.zeros(0, "<U1"))),
        ("mask_or_int_bool", lambda: ma.mask_or(np.array([1, 0, 1]), np.array([True, False, True]))),
        ("mask_or_0d_text", lambda: ma.mask_or(np.array(False), U3)),
        ("masked_where_short", lambda: ma.masked_where([True, False], U3)),
        ("masked_where_col", lambda: ma.masked_where(np.array([[True], [False]]), np.zeros((2, 3)))),
        ("masked_where_text", lambda: ma.masked_where(np.array(["a", "", "b"]), np.arange(3))),
        ("masked_where_text_short", lambda: ma.masked_where(np.array(["a", ""]), np.arange(3))),
        ("masked_where_empty", lambda: ma.masked_where(np.zeros(0, "<U1"), np.zeros(0))),
        ("masked_where_allfalse", lambda: ma.masked_where([False, False, False], U3)),
        ("putmask_plain", lambda: (ma.putmask(U3.copy(), [True, False, True], "zz"))),
        ("compress_2d_text", lambda: ma.compress(np.array([["a"]]), U3)),
        ("compress_rows_1d", lambda: ma.compress_rows(U3)),
        ("compress_cols_1d", lambda: ma.compress_cols(U3)),
        ("mask_rows_1d", lambda: ma.mask_rows(U3)),
        ("mask_cols_1d", lambda: ma.mask_cols(U3)),
        ("common_fill_u_s", lambda: ma.common_fill_value(ma.array(U3), ma.array(S3))),
        ("common_fill_u_u1", lambda: ma.common_fill_value(ma.array(U3), ma.array(np.array(["a"])))),
    ]
    for name, fn in ctor:
        add_ma("ma0", params={"name": name}, args=[], layout="", call=fn)


def _ma_str_disabled(x):
    """``str()`` of a masked array with ``masked_print_option`` disabled (masked slots print their fill value)."""
    np.ma.masked_print_option.enable(False)
    try:
        return str(x)
    finally:
        np.ma.masked_print_option.enable(True)



# ----------------------------------------------------------------------------------------------------------------------
# striter: np.nditer and np.nested_iters over fixed-width strings
# ----------------------------------------------------------------------------------------------------------------------
#
# Each case is a TRACE of one iteration, recorded as text: the iteration descriptors, every step's values (and its
# multi-index / flat index when tracked), the iterator's operands before close, and the caller's arrays after close (so
# copy / updateifcopy write-backs and writes through the views are part of the answer). Values travel as hex (u = UTF-32,
# b = bytes), so supplementary-plane text, embedded NULs and non-ASCII bytes survive. A step that raises ends the walk
# with "!Type: text"; a construction that raises is the case's error.

def _it_val(v):
    """One traced value: text / bytes as hex, numbers by kind, a 1-D chunk (external_loop) as a bracketed list."""
    if isinstance(v, list):
        return "[" + ",".join(_it_val(x) for x in v) + "]"
    if isinstance(v, (str, np.str_)):
        return "u" + str(v).encode("utf-32-le", "surrogatepass").hex()
    if isinstance(v, (bytes, np.bytes_)):
        return "b" + bytes(v).hex()
    if isinstance(v, (bool, np.bool_)):
        return "B1" if v else "B0"
    if isinstance(v, (int, np.integer)):
        return "i" + str(int(v))
    if isinstance(v, (float, np.floating)):
        return "f" + repr(float(v))
    raise TypeError(f"untraceable value {type(v)}")


def _it_trace(views, flags=None, op_flags=None, op_dtypes=None, order="K", casting="safe", buffersize=0, writes=None,
              copy_to=None, iterrange=None):
    kw = {}
    if flags is not None:
        kw["flags"] = flags
    if op_flags is not None:
        kw["op_flags"] = op_flags
    if op_dtypes is not None:
        kw["op_dtypes"] = [None if d is None else np.dtype(d) for d in op_dtypes]
    it = np.nditer(views if len(views) > 1 else views[0], order=order, casting=casting, buffersize=buffersize, **kw)
    lines = ["dtypes " + " ".join(d.str for d in it.dtypes)]
    # A write-only operand is never read by the iterator: an allocated one starts uninitialized (np.empty — an 'S'
    # buffer then holds whatever the allocator recycled), so its step values are rendered as "-", not traced.
    nop = len(views)
    if op_flags is None:
        per_op = [None] * nop
    elif isinstance(op_flags[0], list):
        per_op = op_flags
    else:
        per_op = [op_flags] * nop
    readable = []
    for i in range(nop):
        f = per_op[i] if per_op[i] is not None else (["writeonly", "allocate"] if views[i] is None else ["readonly"])
        readable.append("readonly" in f or "readwrite" in f)
    with it:
        if iterrange is not None:
            it.iterrange = tuple(iterrange)
        k = 0
        walked = False
        try:
            for x in it:
                xs = (x,) if it.nop == 1 else x
                row = []
                if it.has_multi_index:
                    row.append("m" + ",".join(str(i) for i in it.multi_index))
                if it.has_index:
                    row.append("x" + str(it.index))
                row += [_it_val(t.tolist()) if readable[i] else "-" for i, t in enumerate(xs)]
                lines.append(" ".join(row))
                if writes is not None:
                    xs[0][...] = writes[k % len(writes)]
                if copy_to is not None:
                    xs[copy_to][...] = xs[0]
                k += 1
            walked = True
        except Exception as e:  # noqa: BLE001 — a step's error is part of the trace
            lines.append(f"!{err_info(e)['type']}: {e}")
        # An allocated output a failed walk did not finish writing holds uninitialized elements: "-", not traced.
        lines.append("operands " + " ".join(
            _it_val(o.tolist()) if (walked or views[i] is not None) else "-" for i, o in enumerate(it.operands)))
    lines.append("after " + " ".join(_it_val(v.tolist()) for v in views if v is not None))
    return "\n".join(lines)


def _nested_trace(view, axes, flags=None, op_flags=None, op_dtypes=None, buffersize=0):
    kw = {}
    if flags is not None:
        kw["flags"] = flags
    if op_flags is not None:
        kw["op_flags"] = op_flags
    if op_dtypes is not None:
        kw["op_dtypes"] = [None if d is None else np.dtype(d) for d in op_dtypes]
    iters = np.nested_iters(view, axes, buffersize=buffersize, **kw)
    outer, inner = iters[0], iters[-1]
    lines = ["dtypes " + " ".join(d.str for d in inner.dtypes)]
    try:
        for _ in outer:
            row = []
            for y in inner:
                row.append(_it_val(y.tolist()))
            lines.append(" ".join(row))
    except Exception as e:  # noqa: BLE001
        lines.append(f"!{err_info(e)['type']}: {e}")
    for i in iters:
        i.close()
    return "\n".join(lines)


def _it_params(flags=None, op_flags=None, op_dtypes=None, order="K", casting="safe", buffersize=0, writes=None,
               copy_to=None, iterrange=None, none_ops=None):
    p = {"order": order, "casting": casting, "buffersize": buffersize}
    if flags is not None:
        p["flags"] = flags
    if op_flags is not None:
        p["op_flags"] = op_flags
    if op_dtypes is not None:
        p["op_dtypes"] = op_dtypes
    if writes is not None:
        p["writes"] = enc(writes)
    if copy_to is not None:
        p["copy_to"] = copy_to
    if iterrange is not None:
        p["iterrange"] = iterrange
    if none_ops is not None:
        p["none_ops"] = none_ops
    return p


def _it_add(name, pairs, phase=1, layout="", **kw):
    """One nditer trace case: pairs are (base, view) operands, None entries the outputs NumPy allocates."""
    views = [v for _, v in pairs]
    real = [(b, v) for b, v in pairs if b is not None]
    none_ops = [i for i, (b, _) in enumerate(pairs) if b is None] or None
    params = _it_params(none_ops=none_ops, **kw)

    def call(*rviews):
        it_views, j = [], 0
        for b, _ in pairs:
            if b is None:
                it_views.append(None)
            else:
                it_views.append(rviews[j])
                j += 1
        return _it_trace(it_views, **kw)

    add("striter", "nditer_" + name.replace("/", "_"), params=params, operands=real, call=call, phase=phase,
        layout=layout)


def gen_striter():
    flag_sets = [None, ["multi_index"], ["c_index"], ["f_index"], ["external_loop"], ["buffered"],
                 ["buffered", "external_loop"], ["multi_index", "c_index"]]

    # A. the value stream: every layout x both classes x the flag sets x the orders.
    for spec in ("<U3", "|S3"):
        for lname, (n, build) in LAYOUTS.items():
            for flags in flag_sets:
                for order in ("K", "C", "F", "A"):
                    base = base_of(spec, n, seed=len(lname))
                    _it_add("values", [(base, build(base))], layout=lname, flags=flags, order=order)

    # B. buffered requests: widths, the class conversions (ASCII only), unsized requests, every casting rule.
    sources = {
        "u_ascii": ("<U3", True), "u_text": ("<U3", False), "s_ascii": ("|S3", True), "s_bytes": ("|S3", False),
        "u1": ("<U1", True), "s5": ("|S5", True),
    }
    requests = ["<U1", "<U2", "<U3", "<U5", "<U8", "|S1", "|S3", "|S8", "U", "S"]
    for sname, (spec, ascii_only) in sources.items():
        for req in requests:
            for casting in ("no", "safe", "same_kind", "unsafe"):
                for lname in ("c1", "r2", "f2", "s2", "bc", "z0", "t3"):
                    n, build = LAYOUTS[lname]
                    base = base_of(spec, n, seed=3, ascii_only=ascii_only)
                    for bs in (0, 2):
                        _it_add("cast", [(base, build(base))], layout=lname, flags=["buffered"], op_dtypes=[req],
                                casting=casting, buffersize=bs)

    # C. copy routes: a 0-d read-only operand copied without being asked, copy / updateifcopy temporaries written back.
    for spec, ascii_only in (("<U3", True), ("<U3", False), ("|S3", True)):
        for req in ("<U5", "<U2", "|S3", "<U3", "U", "S"):
            for lname in ("c1", "r2", "f2", "z0", "t3", "e2"):
                n, build = LAYOUTS[lname]
                base = base_of(spec, n, seed=5, ascii_only=ascii_only)
                _it_add("copy_ro", [(base, build(base))], layout=lname, op_flags=[["readonly", "copy"]],
                        op_dtypes=[req], casting="unsafe")
                base = base_of(spec, n, seed=5, ascii_only=ascii_only)
                w = ["XYZW", "ab", ""] if req.lstrip("<|")[0] == "U" else [b"XYZW", b"ab", b""]
                _it_add("updateifcopy", [(base, build(base))], layout=lname,
                        op_flags=[["readwrite", "updateifcopy"]], op_dtypes=[req], casting="unsafe", writes=w)
                base = base_of(spec, n, seed=5, ascii_only=ascii_only)
                _it_add("zero_d_auto", [(base, build(base))], layout=lname, op_dtypes=[req], casting="unsafe")

    # D. writes through the views: unbuffered and buffered (a buffered window written back), truncation, NULs, the
    # other class (decoded / ASCII-encoded), and non-ASCII into bytes (NumPy's encode error mid-walk).
    write_sets = {
        "same": (["Q", "RS", "TUVW"], [b"Q", b"RS", b"TUVW"]),
        "nul": (["a\x00", "\x00b", ""], [b"a\x00", b"\x00b", b""]),
        "other": ([b"ab", b"c"], ["ab", "c"]),
        "nonascii": (["é", "ok"], [b"\xff", b"ok"]),
        "supp": (["\U0001F600", "x\U0001F600"], [b"x", b"y"]),
    }
    for spec in ("<U3", "|S3"):
        for wname, (for_u, for_s) in write_sets.items():
            w = for_u if spec[1] == "U" else for_s
            for lname in ("c1", "c2", "r2", "f2", "s2", "z0", "t3"):
                n, build = LAYOUTS[lname]
                for flags, req in ((None, None), (["buffered"], None), (["buffered"], "<U5" if spec[1] == "U" else "|S5"),
                                   (["external_loop"], None)):
                    base = base_of(spec, n, seed=7, ascii_only=True)
                    _it_add(f"write/{wname}", [(base, build(base))], layout=lname, flags=flags,
                            op_flags=[["readwrite"]], op_dtypes=None if req is None else [req], casting="unsafe",
                            writes=w)

    # E. allocated outputs: their descriptors (the common dtype, a sized or an unsized request), filled from the input.
    for spec, ascii_only in (("<U3", True), ("<U3", False), ("|S3", True)):
        for req in (None, "<U5", "<U2", "|S4", "<U3"):
            for flags in (None, ["buffered"], ["external_loop"]):
                for lname in ("c1", "c2", "r2", "f2", "z0", "e2"):
                    n, build = LAYOUTS[lname]
                    base = base_of(spec, n, seed=9, ascii_only=ascii_only)
                    _it_add("alloc", [(base, build(base)), (None, None)], layout=lname, flags=flags,
                            op_flags=[["readonly"], ["writeonly", "allocate"]], op_dtypes=[None, req], casting="unsafe",
                            copy_to=1)

    # F. two operands of different classes / widths: each iterated as its own, or with common_dtype / a request.
    for (sa, sb) in (("<U3", "|S3"), ("<U2", "<U5"), ("|S1", "|S4"), ("<U3", "<i8"), ("|S3", "<f8"), ("<U3", "?")):
        for flags in (None, ["common_dtype"], ["common_dtype", "buffered"], ["buffered"]):
            n, build = LAYOUTS["c2"]
            ba = base_of(sa, n, seed=2, ascii_only=True) if np.dtype(sa).kind in "US" else np.arange(n).astype(sa)
            bb = base_of(sb, 4, seed=4, ascii_only=True) if np.dtype(sb).kind in "US" else np.arange(4).astype(sb)
            _it_add("pair", [(ba, build(ba)), (bb, bb)], layout="pair", flags=flags, casting="unsafe",
                    phase="compute")

    # G. text <-> number through the buffers: NumPy parses / formats (phase 1 since P1.1).
    for spec, values in (("<U3", ["1", "22", "-3"]), ("|S3", [b"1", b"22", b"-3"]), ("<U3", ["1.5", "x", "2"])):
        base = np.array(values * 2, dtype=spec)
        for req in ("<i8", "<f8", "?"):
            for casting in ("safe", "unsafe"):
                _it_add("tonumber", [(base, base)], layout="c1", flags=["buffered"], op_dtypes=[req],
                        casting=casting, phase=1)
    for src in (np.arange(6, dtype=np.int64), np.array([1.5, -2.0, 3.25, 0.0, 1e20, -0.5]), np.array([True, False] * 3)):
        for req in ("<U5", "|S5", "U", "S", "<U2"):
            for casting in ("safe", "unsafe"):
                _it_add("fromnumber", [(src, src)], layout="c1", flags=["buffered"], op_dtypes=[req],
                        casting=casting, phase=1)

    # H. NumPy's refusals over strings: a cast neither copied nor buffered, a read-only broadcast written, ranges.
    for spec in ("<U3", "|S3"):
        n, build = LAYOUTS["bc"]
        base = base_of(spec, n, seed=1, ascii_only=True)
        _it_add("refuse/readonly_bcast", [(base, build(base))], layout="bc", op_flags=[["readwrite"]])
        n, build = LAYOUTS["c2"]
        base = base_of(spec, n, seed=1, ascii_only=True)
        other = "<U5" if spec[1] == "U" else "|S5"
        _it_add("refuse/unbuffered_cast", [(base, build(base))], layout="c2", op_dtypes=[other], casting="unsafe")
        _it_add("refuse/safe_narrow", [(base, build(base))], layout="c2", flags=["buffered"],
                op_dtypes=["<U1" if spec[1] == "U" else "|S1"], casting="safe")
        _it_add("refuse/class_safe", [(base, build(base))], layout="c2", flags=["buffered"],
                op_dtypes=["|S3" if spec[1] == "U" else "<U3"], casting="safe")
        for rng in ((1, 4), (0, 12), (5, 5)):
            _it_add("ranged", [(base, build(base))], layout="c2", flags=["buffered", "ranged"],
                    op_dtypes=["<U5" if spec[1] == "U" else "|S5"], casting="unsafe", iterrange=list(rng))

    # I. np.nested_iters over strings.
    for spec in ("<U3", "|S3"):
        for lname, axes in (("c2", [[0], [1]]), ("c2", [[1], [0]]), ("f2", [[0], [1]]), ("c3", [[0], [1, 2]]),
                            ("c3", [[2], [0, 1]]), ("t3", [[1], [0, 2]]), ("r2", [[1], [0]])):
            n, build = LAYOUTS[lname]
            base = base_of(spec, n, seed=11)
            view = build(base)
            add("striter", "nested_iters", params={"axes": axes}, operands=[(base, view)],
                call=lambda v, axes=axes: _nested_trace(v, axes), layout=lname)
            base = base_of(spec, n, seed=11, ascii_only=True)
            view = build(base)
            req = "<U5" if spec[1] == "U" else "|S5"
            add("striter", "nested_iters", params={"axes": axes, "flags": ["buffered"], "op_dtypes": [req]},
                operands=[(base, view)], call=lambda v, axes=axes, req=req: _nested_trace(
                    v, axes, flags=["buffered"], op_dtypes=[req]), layout=lname)


# ----------------------------------------------------------------------------------------------------------------------
# Section strcomp: comparisons and truth values (strings phase 2, P2.1) — strict (phase 1: NumSharp computes them)
# ----------------------------------------------------------------------------------------------------------------------

CMP_OPS = ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"]

# Element widths per class: equal widths, a longer second operand (its tail decides a zero-padded equal prefix), a
# shorter one, widths past one Vector256 (U8 = 32 bytes, S33), and past the kernels' unrolled chunk plan (U70 = 280
# bytes, S300), which switch to the loop form.
CMP_WIDTHS = {
    "U": [("U3", "U3"), ("U3", "U8"), ("U9", "U2"), ("U40", "U40"), ("U70", "U66")],
    "S": [("S3", "S3"), ("S2", "S9"), ("S33", "S8"), ("S300", "S290")],
}

# The code units the values are built from: a common pair, a non-ASCII and a supplementary-plane character (U), bytes
# on both sides of 0x80 (S, compared unsigned), and NUL (embedded NULs count, trailing ones do not).
CMP_UNITS = {"U": ["a", "b", "\x00", "Ü", "\U0001F600", "z"], "S": [b"a", b"b", b"\x00", b"\x80", b"\xff", b"\x7f"]}

# Same-layout pairs over every layout kind, then pairs of different layouts that broadcast (or must refuse to).
CMP_CROSS = [("c2", "f2"), ("f2", "c2"), ("c2", "r2"), ("r2", "bc"), ("c2", "z0"), ("z0", "c2"), ("nx", "c2"),
             ("one", "s1"), ("d5", "one"), ("e2", "z0"), ("e3", "one"), ("bcol", "f2"), ("col", "c2")]


def _cmp_values(kind, width, n, seed, like=None):
    """``n`` deterministic values of one class for a comparison operand, at most ``width`` code units each.

    Without ``like``: lengths cycle 0..width over the units of CMP_UNITS (offset by ``seed``). With ``like`` (the other
    operand's values): element i is, by ``(i + seed) % 5``, that value itself (equal), its prefix one unit shorter, the
    value with one unit changed, the value with an extra nonzero unit appended (a tail difference), or the value with a
    trailing NUL appended (equal again: a trailing NUL never counts).

    Args:
        kind: ``"U"`` (str values) or ``"S"`` (bytes values).
        width: the most code units a value keeps (the dtype's width).
        n: how many values.
        seed: an offset that varies the cycle between operands.
        like: the other operand's values to derive these from, or None.

    Returns:
        A list of ``n`` str / bytes values.
    """
    units = CMP_UNITS[kind]
    join = "".join if kind == "U" else b"".join
    out = []
    for i in range(n):
        if like is not None:
            v = like[i % len(like)]
            mode = (i + seed) % 5
            if mode == 1:
                v = v[:-1]
            elif mode == 2 and len(v) > 0:
                k = (i * 7 + seed) % len(v)
                v = v[:k] + units[(i + seed + 1) % len(units)] + v[k + 1:]
            elif mode == 3:
                v = v + units[(i + seed + 3) % len(units)].replace("\x00" if kind == "U" else b"\x00",
                                                                   "q" if kind == "U" else b"q")
            elif mode == 4:
                v = v + units[2]
            out.append(v[:width])
            continue
        length = (i * 3 + seed) % (width + 1)
        out.append(join(units[(i * 5 + k * 3 + seed) % len(units)] for k in range(length)))
    return out


def _cmp_operand(spec, layout, seed, like=None):
    """A (base, view) comparison operand of dtype ``spec`` in one of LAYOUTS, its base built by ``_cmp_values``.

    Args:
        spec: the dtype, ``"U3"`` / ``"S9"`` (class and width).
        layout: a LAYOUTS key (the base length and the view builder).
        seed: passed to ``_cmp_values``.
        like: the other operand's base values to derive these from (same base length), or None.

    Returns:
        The (base, view) pair ``describe`` serializes.
    """
    n, build = LAYOUTS[layout]
    kind = spec[0]
    width = int(spec[1:])
    base = np.array(_cmp_values(kind, width, n, seed, like), dtype=spec) if n else np.zeros(0, dtype=spec)
    return base, build(base)


def _units_operand(units, chars, shape=None):
    """A ``U{chars}`` operand over raw UCS-4 units (units above U+10FFFF are reachable only through such a view).

    Args:
        units: the code units, a multiple of ``chars`` of them.
        chars: the width of each element in units.
        shape: the view's shape, or None for the 1-D base itself.

    Returns:
        The (base, view) pair ``describe`` serializes.
    """
    base = np.array(units, dtype=np.uint32).view(f"U{chars}")
    base = np.ascontiguousarray(base)
    return base, (base if shape is None else base.reshape(shape))


def gen_strcomp():
    """Section strcomp: the six comparison ufuncs over two strings of one class and the truth readers, strict."""
    # A. The six comparisons: every layout kind (same layout for both operands) and the cross pairs, every width pair.
    for kind, pairs in CMP_WIDTHS.items():
        for s1, s2 in pairs:
            layouts = [(name, name) for name in LAYOUTS] + CMP_CROSS
            for l1, l2 in layouts:
                b1, v1 = _cmp_operand(s1, l1, seed=1)
                like = [x.item() if hasattr(x, "item") else x for x in b1.tolist()] if b1.size else None
                b2, v2 = _cmp_operand(s2, l2, seed=2, like=like if LAYOUTS[l1][0] == LAYOUTS[l2][0] else None)
                for name in CMP_OPS:
                    add("strcomp", "ufunc", params={"name": name}, operands=[(b1, v1), (b2, v2)],
                        call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"{s1}:{l1},{s2}:{l2}")

    # B. Special code units, every pair through an outer broadcast: U units on both sides of U+10FFFF and of the BMP
    #    (compared as unsigned 32-bit values, no error), S bytes on both sides of 0x80.
    u_units = [0, 0x41, 0xFFFF, 0x1F600, 0x10FFFF, 0x110000, 0xFFFFFFFF]
    s_bytes = [b"", b"\x01", b"\x7f", b"\x80", b"\xff", b"\x00\x01", b"a\x00"]
    for name in CMP_OPS:
        col = _units_operand(u_units, 1, (len(u_units), 1))
        row = _units_operand(u_units, 1, (1, len(u_units)))
        add("strcomp", "ufunc", params={"name": name}, operands=[col, row],
            call=lambda x, y, f=getattr(np, name): f(x, y), layout="U1:units")
        pairs = _units_operand([0x41, 0x110000, 0x42, 0x41, 0, 0x110000], 2, (3, 1))
        rowp = _units_operand([0x41, 0x42, 0x110000, 0], 2, (1, 2))
        add("strcomp", "ufunc", params={"name": name}, operands=[pairs, rowp],
            call=lambda x, y, f=getattr(np, name): f(x, y), layout="U2:units")
        cb = plain(np.array(s_bytes, dtype="S2").reshape(-1, 1))
        rb = plain(np.array(s_bytes, dtype="S3").reshape(1, -1))
        add("strcomp", "ufunc", params={"name": name}, operands=[cb, rb],
            call=lambda x, y, f=getattr(np, name): f(x, y), layout="S:bytes")

    # C. The C# operators with a string operand: NumPy's Python str (a U scalar) on either side; bytes against str.
    for spec, layout in [("U3", "c2"), ("U8", "f2"), ("U3", "z0"), ("S3", "c2"), ("U3", "e2")]:
        for sym in ["==", "!=", "<", "<=", ">", ">="]:
            for text in ["", "a", "b\x00", "zzzz"]:
                for side in ["right", "left"]:
                    b, v = _cmp_operand(spec, layout, seed=3)
                    add("strcomp", "operator_text", params={"op": sym, "text": enc(text), "side": side},
                        operands=[(b, v)], call=lambda a, s=sym, t=text, sd=side: _text_op(s, t, a, sd),
                        layout=f"{spec}:{layout}")

    # D. out= (bool and numeric dtypes, a broadcast-absorbing out), where= masks, dtype=bool.
    for name in CMP_OPS:
        for odt in ["?", "i1", "f8", "c16", "u8"]:
            b1, v1 = _cmp_operand("U4", "c2", seed=4)
            b2, v2 = _cmp_operand("U6", "bc", seed=5, like=None)
            out = target(np.zeros((3, 4), dtype=odt))
            add("strcomp", "ufunc_kw", params={"name": name, "nin": 2, "out": 2, "dtype": None},
                operands=[(b1, v1), (b2, v2), out], call=lambda x, y, o, f=getattr(np, name): f(x, y, out=o),
                layout=f"out:{odt}")
        for mask in [np.array([[1, 0, 1, 0]], bool), np.eye(3, 4, dtype=bool), np.zeros((3, 4), bool)]:
            b1, v1 = _cmp_operand("S5", "f2", seed=6)
            b2, v2 = _cmp_operand("S5", "c2", seed=7)
            out = target(np.ones((3, 4), dtype=bool))
            add("strcomp", "ufunc_kw", params={"name": name, "nin": 2, "out": 2, "where": 3, "dtype": None},
                operands=[(b1, v1), (b2, v2), out, plain(mask)],
                call=lambda x, y, o, w, f=getattr(np, name): f(x, y, out=o, where=w), layout="where")
        b1, v1 = _cmp_operand("U3", "t3", seed=8)
        b2, v2 = _cmp_operand("U3", "t3", seed=9)
        add("strcomp", "ufunc_kw", params={"name": name, "nin": 2, "dtype": "?"}, operands=[(b1, v1), (b2, v2)],
            call=lambda x, y, f=getattr(np, name): f(x, y, dtype="?"), layout="dtype:?")

    # E. Truth values. The cast route (astype(bool), any / all, the logical ufuncs) and the nonzero route (nonzero,
    #    count_nonzero, flatnonzero, argwhere) over every layout kind and width; units above U+10FFFF (the cast raises
    #    SystemError, the nonzero route does not).
    for kind, pairs in CMP_WIDTHS.items():
        for spec in sorted({s for p in pairs for s in p}):
            for layout in LAYOUTS:
                b, v = _cmp_operand(spec, layout, seed=10)
                for order in ["K", "C", "F", "A"]:
                    add("strcomp", "astype", params={"dtype": "?", "casting": "unsafe", "order": order, "copy": True},
                        operands=[(b, v)], call=lambda a, o=order: a.astype(bool, order=o, casting="unsafe", copy=True),
                        layout=f"{spec}:{layout}")
                for name in ["any", "all", "count_nonzero"] + ([] if v.ndim == 0 else ["nonzero", "flatnonzero",
                                                                                        "argwhere"]):
                    add("strcomp", "call", params={"name": name}, operands=[(b, v)],
                        call=lambda a, f=getattr(np, name): f(a), layout=f"{spec}:{layout}")
                if v.ndim >= 1:
                    for name in ["any", "all", "count_nonzero"]:
                        for axis in [0, -1]:
                            for keepdims in [False, True]:
                                add("strcomp", "truth_axis", params={"name": name, "axis": axis, "keepdims": keepdims},
                                    operands=[(b, v)],
                                    call=lambda a, f=getattr(np, name), ax=axis, kd=keepdims: f(a, axis=ax, keepdims=kd),
                                    layout=f"{spec}:{layout}")
    for name in ["logical_and", "logical_or", "logical_xor"]:
        for (s1, l1), (s2, l2) in [(("U3", "c2"), ("U3", "f2")), (("S3", "r2"), ("S3", "bc")), (("U3", "c2"), (None, "num")),
                                   (("U3", "z0"), ("S2", "c2"))]:
            b1, v1 = _cmp_operand(s1, l1, seed=11)
            o2 = num_operand(np.array([[0, 1, 2, 0]] * 3)) if s2 is None else _cmp_operand(s2, l2, seed=12)
            add("strcomp", "ufunc", params={"name": name}, operands=[(b1, v1), o2],
                call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"{l1},{l2}")
    big = [0x110000, 0x41, 0, 0x10FFFF]
    for chars, shape in [(1, None), (2, (2,))]:
        op = _units_operand(big, chars, shape)
        add("strcomp", "astype", params={"dtype": "?", "casting": "unsafe", "order": "K", "copy": True}, operands=[op],
            call=lambda a: a.astype(bool, order="K", casting="unsafe", copy=True), layout="units")
        for name in ["any", "all", "count_nonzero", "nonzero", "flatnonzero", "argwhere"]:
            add("strcomp", "call", params={"name": name}, operands=[op], call=lambda a, f=getattr(np, name): f(a),
                layout="units")
        add("strcomp", "ufunc", params={"name": "logical_or"}, operands=[op, num_operand(np.array(0))],
            call=lambda x, y: np.logical_or(x, y), layout="units")


def _text_op(sym, text, a, side):
    """``a OP text`` or ``text OP a`` for a Python str operand (on the left, Python reflects it to the array's operator).

    Args:
        sym: the operator, ``"=="`` ``"!="`` ``"<"`` ``"<="`` ``">"`` ``">="``.
        text: the Python str.
        a: the array.
        side: ``"right"`` for ``a OP text``, ``"left"`` for ``text OP a``.

    Returns:
        NumPy's result.
    """
    x, y = (a, text) if side == "right" else (text, a)
    return {"==": lambda: x == y, "!=": lambda: x != y, "<": lambda: x < y, "<=": lambda: x <= y,
            ">": lambda: x > y, ">=": lambda: x >= y, "+": lambda: x + y, "*": lambda: x * y}[sym]()


# ----------------------------------------------------------------------------------------------------------------------
# Section strarith: concatenation and repetition (strings phase 2, P2.3) — strict (phase 1: NumSharp computes them)
# ----------------------------------------------------------------------------------------------------------------------

def _count_operand(view, dtype, offset=-1):
    """A per-element repeat-count operand of ``view``'s shape: counts cycling from ``offset`` (negative counts repeat
    nothing), C-ordered, of integer ``dtype``.

    Args:
        view: the text operand the counts pair with (its shape).
        dtype: the counts' dtype.
        offset: the first count.

    Returns:
        The (base, view) pair of the counts.
    """
    counts = (np.arange(view.size, dtype=np.int64).reshape(view.shape) % 6 + offset).astype(dtype)
    return plain(counts)


def gen_strarith():
    """Section strarith: np.add over two strings of one class, the string repeat (np.multiply into out=), the C#
    operators + / * with a string operand, and np.strings.add / np.strings.multiply, strict."""
    # A. add over every layout kind (same layout for both operands) and the cross pairs, every width pair.
    for kind, pairs in CMP_WIDTHS.items():
        for s1, s2 in pairs:
            layouts = [(name, name) for name in LAYOUTS] + CMP_CROSS
            for l1, l2 in layouts:
                b1, v1 = _cmp_operand(s1, l1, seed=21)
                like = [x.item() if hasattr(x, "item") else x for x in b1.tolist()] if b1.size else None
                b2, v2 = _cmp_operand(s2, l2, seed=22, like=like if LAYOUTS[l1][0] == LAYOUTS[l2][0] else None)
                add("strarith", "ufunc", params={"name": "add"}, operands=[(b1, v1), (b2, v2)],
                    call=lambda x, y: np.add(x, y), layout=f"{s1}:{l1},{s2}:{l2}")

    # B. Special code units through an outer broadcast: embedded / trailing NULs, U units above U+10FFFF (just
    #    nonzero units here), S bytes on both sides of 0x80.
    u_units = [0, 0x41, 0xFFFF, 0x1F600, 0x10FFFF, 0x110000, 0xFFFFFFFF]
    add("strarith", "ufunc", params={"name": "add"},
        operands=[_units_operand(u_units, 1, (len(u_units), 1)), _units_operand(u_units, 1, (1, len(u_units)))],
        call=lambda x, y: np.add(x, y), layout="U1:units")
    add("strarith", "ufunc", params={"name": "add"},
        operands=[_units_operand([0x41, 0x110000, 0, 0x110000, 0x42, 0], 2, (3, 1)),
                  _units_operand([0x41, 0, 0, 0x110000], 2, (1, 2))],
        call=lambda x, y: np.add(x, y), layout="U2:units")
    s_bytes = [b"", b"\x01", b"\x7f", b"\x80", b"\xff", b"\x00\x01", b"a\x00"]
    add("strarith", "ufunc", params={"name": "add"},
        operands=[plain(np.array(s_bytes, dtype="S2").reshape(-1, 1)), plain(np.array(s_bytes, dtype="S3").reshape(1, -1))],
        call=lambda x, y: np.add(x, y), layout="S:bytes")

    # C. The repeat into out=: text layouts x count forms (0-d counts, per-element counts, a broadcast column), widths
    #    past one vector and past the unrolled plans, out widths that cut the repetition and that pad it, both operand
    #    orders.
    for spec in ["U3", "U8", "U70", "S3", "S9", "S300"]:
        width = int(spec[1:])
        for layout in LAYOUTS:
            b, v = _cmp_operand(spec, layout, seed=23)
            for count_form in ["scalar", "array"]:
                for owidth in [width * 3, max(1, width * 2 - 1)]:
                    if count_form == "scalar":
                        counts = [plain(np.array(c)) for c in (2, 0, -1, 3)]
                    else:
                        counts = [_count_operand(v, np.int64)]
                    for c in counts:
                        out = target(np.zeros(np.broadcast_shapes(v.shape, c[1].shape), dtype=f"{spec[0]}{owidth}"))
                        add("strarith", "ufunc_kw", params={"name": "multiply", "nin": 2, "out": 2, "dtype": None},
                            operands=[(b, v), c, out], call=lambda x, y, o: np.multiply(x, y, out=o),
                            layout=f"{spec}:{layout}:{count_form}:o{owidth}")
            # the count first (NumPy's int-str loop)
            if v.size:
                c = _count_operand(v, np.int64, offset=0)
                out = target(np.zeros(v.shape, dtype=f"{spec[0]}{width * 4}"))
                add("strarith", "ufunc_kw", params={"name": "multiply", "nin": 2, "out": 2, "dtype": None},
                    operands=[c, (b, v), out], call=lambda x, y, o: np.multiply(x, y, out=o), layout=f"{spec}:{layout}:intfirst")

    # D. Count dtypes and the repeat's refusals: every integer width (a uint64 count past 2^63 wraps negative), an
    #    int64 count whose repetition overflows (OverflowError after the loop), bool / float counts (no loop), no out=
    #    (the required-out TypeError), a U loop into an S out / a numeric out (the output cast), an S loop into a U out
    #    (the ASCII decode), where= masks.
    b, v = _cmp_operand("U3", "c2", seed=24)
    for dt in ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8"]:
        c = plain(np.array([2], dtype=dt))
        out = target(np.zeros(v.shape, dtype="U6"))
        add("strarith", "ufunc_kw", params={"name": "multiply", "nin": 2, "out": 2, "dtype": None},
            operands=[(b, v), c, out], call=lambda x, y, o: np.multiply(x, y, out=o), layout=f"count:{dt}")
    for value, dt in [(2**63 + 5, "u8"), (2**62, "i8"), (2**61 + 1, "i8"), (-(2**63), "i8")]:
        out = target(np.zeros(v.shape, dtype="U6"))
        add("strarith", "ufunc_kw", params={"name": "multiply", "nin": 2, "out": 2, "dtype": None},
            operands=[(b, v), plain(np.array(value, dtype=dt)), out], call=lambda x, y, o: np.multiply(x, y, out=o),
            layout=f"count:{value}")
    for c in [plain(np.array(True)), plain(np.array(2.0)), plain(np.array([[2.5]]))]:
        out = target(np.zeros(v.shape, dtype="U6"))
        add("strarith", "ufunc_kw", params={"name": "multiply", "nin": 2, "out": 2, "dtype": None},
            operands=[(b, v), c, out], call=lambda x, y, o: np.multiply(x, y, out=o), layout="count:nonint")
    add("strarith", "ufunc", params={"name": "multiply"}, operands=[(b, v), plain(np.array(2))],
        call=lambda x, y: np.multiply(x, y), layout="noout")
    for odt in ["S9", "f8", "?", "U2"]:
        out = target(np.zeros(v.shape, dtype=odt))
        add("strarith", "ufunc_kw", params={"name": "multiply", "nin": 2, "out": 2, "dtype": None},
            operands=[(b, v), plain(np.array(3)), out], call=lambda x, y, o: np.multiply(x, y, out=o), layout=f"out:{odt}")
    for spec, vals in [("S3", [b"ab", b"a\x00", b"", b"xyz"]), ("S3", [b"ab", b"\xff", b"", b"q"])]:
        sb = plain(np.array(vals, dtype=spec))
        out = target(np.zeros(4, dtype="U9"))
        add("strarith", "ufunc_kw", params={"name": "multiply", "nin": 2, "out": 2, "dtype": None},
            operands=[sb, plain(np.array(3)), out], call=lambda x, y, o: np.multiply(x, y, out=o), layout="S:into:U")
    for mask in [np.array([[1, 0, 1, 0]], bool), np.eye(3, 4, dtype=bool), np.zeros((3, 4), bool)]:
        out = target(np.array([["k"] * 4] * 3, dtype="U6"))
        add("strarith", "ufunc_kw", params={"name": "multiply", "nin": 2, "out": 2, "where": 3, "dtype": None},
            operands=[(b, v), plain(np.array(2)), out, plain(mask)],
            call=lambda x, y, o, w: np.multiply(x, y, out=o, where=w), layout="where")

    # E. add with out= (exact, narrower, wider, S into U, U into S, numeric), where=, dtype= (ignored by the string
    #    add promoter), the bytes / str mix, a number operand, 0-d operands.
    b1, v1 = _cmp_operand("U3", "c2", seed=25)
    b2, v2 = _cmp_operand("U4", "bc", seed=26)
    for odt in ["U7", "U3", "U1", "U12", "S7", "f8", "?"]:
        out = target(np.zeros((3, 4), dtype=odt))
        add("strarith", "ufunc_kw", params={"name": "add", "nin": 2, "out": 2, "dtype": None},
            operands=[(b1, v1), (b2, v2), out], call=lambda x, y, o: np.add(x, y, out=o), layout=f"out:{odt}")
    for vals in [[b"ab", b"a\x00", b""], [b"ab", b"\x80q", b""]]:
        sa = plain(np.array(vals, dtype="S2"))
        sb = plain(np.array([b"c", b"", b"de"], dtype="S2"))
        out = target(np.zeros(3, dtype="U4"))
        add("strarith", "ufunc_kw", params={"name": "add", "nin": 2, "out": 2, "dtype": None},
            operands=[sa, sb, out], call=lambda x, y, o: np.add(x, y, out=o), layout="S:into:U")
    for mask in [np.array([[1, 0, 1, 0]], bool), np.eye(3, 4, dtype=bool), np.zeros((3, 4), bool)]:
        out = target(np.array([["k"] * 4] * 3, dtype="U7"))
        add("strarith", "ufunc_kw", params={"name": "add", "nin": 2, "out": 2, "where": 3, "dtype": None},
            operands=[(b1, v1), (b2, v2), out, plain(mask)],
            call=lambda x, y, o, w: np.add(x, y, out=o, where=w), layout="where")
    for dt in ["f8", "U", "S", "i4"]:
        add("strarith", "ufunc_kw", params={"name": "add", "nin": 2, "dtype": dt}, operands=[(b1, v1), (b2, v2)],
            call=lambda x, y, d=dt: np.add(x, y, dtype=d), layout=f"dtype:{dt}")
    sx = _cmp_operand("S3", "c2", seed=27)
    add("strarith", "ufunc", params={"name": "add"}, operands=[(b1, v1), sx], call=lambda x, y: np.add(x, y), layout="U+S")
    add("strarith", "ufunc", params={"name": "add"}, operands=[sx, (b1, v1)], call=lambda x, y: np.add(x, y), layout="S+U")
    for num in [np.array(1), np.array([[1.5]]), np.array(True)]:
        add("strarith", "ufunc", params={"name": "add"}, operands=[(b1, v1), num_operand(num)],
            call=lambda x, y: np.add(x, y), layout="U+num")
        add("strarith", "ufunc", params={"name": "add"}, operands=[num_operand(num), (b1, v1)],
            call=lambda x, y: np.add(x, y), layout="num+U")
    add("strarith", "ufunc", params={"name": "add"}, operands=[plain(np.array("ab")), plain(np.array("c\x00d"))],
        call=lambda x, y: np.add(x, y), layout="0d")

    # F. The concatenation too wide for NumPy's byte limit (resolved before the out cast and the broadcast).
    for s1, s2 in [("S2147483647", "S1"), ("U536870911", "U1"), ("S2147483646", "S1")]:
        add("strarith", "ufunc", params={"name": "add"},
            operands=[plain(np.zeros(0, dtype=s1)), plain(np.zeros(0, dtype=s2))], call=lambda x, y: np.add(x, y),
            layout=f"limit:{s1}+{s2}")

    # G. The C# operators with a string operand: concatenation on either side (the string stays first on the left:
    #    Python's __radd__), bytes plus str, the repeat without out= (both orders), a string times a string.
    for spec, layout in [("U3", "c2"), ("U8", "f2"), ("U3", "z0"), ("S3", "c2"), ("U3", "e2"), ("U3", "t3")]:
        for sym in ["+", "*"]:
            for text in ["", "a", "b\x00", "zzzz"]:
                for side in ["right", "left"]:
                    b, v = _cmp_operand(spec, layout, seed=28)
                    add("strarith", "operator_text", params={"op": sym, "text": enc(text), "side": side},
                        operands=[(b, v)], call=lambda a, s=sym, t=text, sd=side: _text_op(s, t, a, sd),
                        layout=f"{spec}:{layout}")
    b, v = _cmp_operand("U3", "c2", seed=29)
    for order in ["text_first", "int_first"]:
        add("strarith", "ufunc", params={"name": "multiply"},
            operands=[(b, v), plain(np.array(2))] if order == "text_first" else [plain(np.array(2)), (b, v)],
            call=lambda x, y: x * y, layout=f"operator:{order}")

    # H. np.strings.add (the np.add ufunc itself) and np.strings.multiply (NumPy's two-pass wrapper: the count check,
    #    max(i, 0), str_len, the overflow check, the output width from the largest repetition, empty_like's layout).
    for l1, l2 in [("c2", "c2"), ("f2", "f2"), ("c2", "bc"), ("z0", "z0"), ("e2", "e2")]:
        b1, v1 = _cmp_operand("U4", l1, seed=30)
        b2, v2 = _cmp_operand("U3", l2, seed=31)
        add("strarith", "strings_add", operands=[(b1, v1), (b2, v2)], call=lambda x, y: np.strings.add(x, y),
            layout=f"{l1},{l2}")
    for spec in ["U3", "S3", "U70"]:
        for layout in ["c2", "f2", "t3", "r2", "s2", "bc", "z0", "e2", "one", "d5"]:
            b, v = _cmp_operand(spec, layout, seed=32)
            for c in [plain(np.array(2)), plain(np.array(0)), plain(np.array(-3)), _count_operand(v, np.int64)]:
                add("strarith", "strings_multiply", operands=[(b, v), c], call=lambda a, i: np.strings.multiply(a, i),
                    layout=f"{spec}:{layout}")
    b, v = _cmp_operand("U3", "c2", seed=33)
    for dt in ["i1", "u1", "i4", "u4", "i8", "u8"]:
        add("strarith", "strings_multiply", operands=[(b, v), plain(np.array(2, dtype=dt))],
            call=lambda a, i: np.strings.multiply(a, i), layout=f"count:{dt}")
    for c in [plain(np.array(2.0)), plain(np.array(True)), plain(np.array("x")), plain(np.array(2**62)),
              plain(np.array(2**63 + 1, dtype="u8")), plain(np.array([[1], [2], [3]])), plain(np.array([1, 2]))]:
        add("strarith", "strings_multiply", operands=[(b, v), c], call=lambda a, i: np.strings.multiply(a, i),
            layout="count:edge")
    add("strarith", "strings_multiply", operands=[num_operand(np.array([1, 2])), plain(np.array(2))],
        call=lambda a, i: np.strings.multiply(a, i), layout="a:numeric")
    add("strarith", "strings_multiply", operands=[plain(np.zeros(0, "U3")), plain(np.array(2))],
        call=lambda a, i: np.strings.multiply(a, i), layout="a:empty")
    for text in ["ab", "", "x\x00y"]:
        add("strarith", "strings_multiply_text", params={"text": enc(text)}, operands=[plain(np.array(3))],
            call=lambda i, t=text: np.strings.multiply(t, i), layout="a:str")


# ----------------------------------------------------------------------------------------------------------------------
# Section strconv: conversion between text and numbers (strings phase 1, P1.1), strict. NumPy's legacy casts: a number
# becomes its scalar's str() (truncated / NUL-padded to the width; an unsized request takes REQUIRED_STR_LEN), a text
# is parsed by int() / float() / complex() through the NumPy scalar (np.str_ / np.bytes_), with NumPy's errors.
# ----------------------------------------------------------------------------------------------------------------------

_INF, _NAN = float("inf"), float("nan")

# The value set per numeric dtype: zeros of both signs, extremes, the float positional / scientific thresholds,
# specials, values whose shortest repr needs every digit.
CONV_VALUES = {
    "?": [True, False],
    "i1": [0, 1, -1, 127, -128, 42],
    "u1": [0, 1, 255, 42],
    "i2": [0, -1, 32767, -32768, 1234],
    "u2": [0, 65535, 1234],
    "i4": [0, -1, 2 ** 31 - 1, -2 ** 31, 123456],
    "u4": [0, 2 ** 32 - 1, 123456],
    "i8": [0, -1, 2 ** 63 - 1, -2 ** 63, 1234567890123],
    "u8": [0, 2 ** 64 - 1, 2 ** 63, 1234567890123],
    "f2": [0.0, -0.0, 1.5, -2.25, 65504.0, 6e-08, 0.1, 1e4, _INF, -_INF, _NAN, 1 / 3, 2049.0, 0.00006103515625],
    "f4": [0.0, -0.0, 1.5, 0.1, 1e16, 1e-5, 1e-4, 3.4028235e38, 1.4e-45, 16777217.0, _INF, -_INF, _NAN, 1 / 3,
           9999999.0, 1e15, 0.0001],
    "f8": [0.0, -0.0, 1.5, -2.25, 0.1, 1e16, 9999999999999998.0, 1e15, 1e-4, 1e-5, 1e22, 5e-324,
           1.7976931348623157e308, _INF, -_INF, _NAN, 1 / 3, 123456789.125, 2.0 ** 53 + 2],
    "c16": [1 + 2j, complex(0.0, -0.0), complex(-0.0, 0.0), complex(_INF, _NAN), complex(_NAN, -_INF),
            complex(1e16, 1e-5), 0j, -1.5j, complex(_NAN, 0.0), complex(1.5, -0.0), complex(-1e-5, 1e22)],
}
CONV_LAYOUTS = ["c1", "r1", "s1", "c2", "f2", "bc", "z0", "e1", "one", "rs"]


def _num_operand(dt, layout, values, seed=0):
    """A numeric (base, view) operand of dtype ``dt`` in one of LAYOUTS, its base cycling through ``values``."""
    n, build = LAYOUTS[layout]
    base = np.array([values[(seed + i) % len(values)] for i in range(n)], dtype=dt) if n else np.zeros(0, dt)
    return base, build(base)


def _platform_text(text):
    """NumPy's C long / long long overflow texts differ between the Windows corpus host and LP64 (Linux / macOS):
    "int too big to convert" / "Python int too large to convert to C long" on Windows, "Python integer ... out of
    bounds for int32" on LP64 for an int32 / uint32 target between 2^31 and 2^63. NumSharp pins the LP64 texts (unit
    tests); such cases stay out of the corpus, the clip_pyint policy."""
    return "too large to convert" in text or "too big to convert" in text


def _conv(section, op, **kw):
    """``add`` for a case whose outcome must be portable: a case NumPy answers with a platform-dependent overflow text is
    dropped again (its id freed)."""
    add(section, op, **kw)
    c = CASES[-1]
    if "error" in c and _platform_text(c["error"]["text"]):
        CASES.pop()
        _IDS.discard(c["id"])


def _text_operand(values, spec):
    """A 1-D text operand of the given values (a str list for U, encoded for S) at ``spec`` (e.g. "U", "S", "U8")."""
    if spec.lstrip("<|")[0] == "S":
        vals = [v if isinstance(v, bytes) else v.encode("utf-8") for v in values]
    else:
        vals = values
    return plain(np.array(vals, dtype=spec))


# Texts and the target kinds they exercise. Each text is converted alone (so NumPy's error names it) and every target
# of the kind is tried.
INT_TEXTS = ["0", "-0", "+5", " 12 ", "\t7\n", "1_000", "00012", "١٢٣", "１２", "  5 　", "-1", "",
             " ", "1.0", "1e3", "0x10", "1__0", "_1", "1_", "+-1", "abc", "1 2", "٣.٥", "12\x00", "1\x002", "\x00",
             "127", "128", "-128", "-129", "255", "256", "32767", "32768", "65535", "65536", "2147483647",
             "2147483648", "4294967295", "4294967296", "9223372036854775807", "9223372036854775808",
             "18446744073709551615", "18446744073709551616", "-9223372036854775808", "-9223372036854775809",
             "\ud800", "\U0001F600", "Ⅻ", "²", "0b1", "1 _0", "- 1"]
FLOAT_TEXTS = ["1.5", "-0.0", "inf", "-Infinity", "nan", "-nan", "NaN", "+inf", "1e308", "1e309", "1e-400",
               "1_000.5", ".5", "5.", " 3.25 ", "٣.٥", "1E5", "infinity", "-iNf", "0.1", "65520", "65519.99",
               "3.4028236e38", "1e-45", "6e-8", "", "0x1p3", "1._5", "nan(123)", "infinit", "1e", "e5", "--1",
               "1.5.5", "1_e5", "1e_5", "_1.5", "1.5_", "١_٢", "inf ", " -nan", "+-1.5", "1,5", "1.5\x00", "\x001.5",
               "12345678901234567890", "-0", "1e1000", "0.000000000000000000000000000001"]
COMPLEX_TEXTS = ["1+2j", "(1+2j)", "-j", "j", "1e3j", "inf+nanj", "1+2J", " (3-4j) ", "2", "nan", "-0-0j",
                 "+1.5-2.5j", "(1)", " 1+2j ", "1+", "(1+2j", "1 + 2j", "1+2", "jj", "(", "", "1j2", "1+2jj", "( 1+2j )",
                 "1_0+2_0j", "infj", "-infj", "nanj", "1e5+1e-5j", "1.5", "  ", "(1+2j)\x00"]
TARGETS = {"int": ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8"], "float": ["f2", "f4", "f8"], "complex": ["c16"]}


def gen_strconv():
    """Section strconv: text <-> number conversions through every route that converts, strict."""
    # A. Numbers to text: every numeric dtype x value set x layout x unsized / sized / narrow targets.
    targets = ["U", "S", "U1", "U4", "U40", "S3", "S40"]
    for dt, values in CONV_VALUES.items():
        for lay in CONV_LAYOUTS:
            for t in targets:
                _conv("strconv", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
                      operands=[_num_operand(dt, lay, values, seed=1)], layout=f"{dt}:{lay}",
                      call=lambda a, t=t: a.astype(t))
        # Every value alone through each target: the scalar str() of each special.
        for i, v in enumerate(values):
            for t in ("U", "S", "U6"):
                _conv("strconv", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
                      operands=[plain(np.array([v], dtype=dt))], layout=f"{dt}:v{i}", call=lambda a, t=t: a.astype(t))
        # The casting rules: an unsized request adapts to REQUIRED_STR_LEN (safe); a narrower one is same_kind.
        for t in ("U", "U3", "S", "S3", "U64"):
            for casting in ("no", "equiv", "safe", "same_kind", "unsafe"):
                _conv("strconv", "astype", params={"dtype": t, "casting": casting, "order": "K", "copy": True},
                      operands=[plain(np.array(values[:2], dtype=dt))], layout=f"{dt}:{casting}",
                      call=lambda a, t=t, c=casting: a.astype(t, casting=c))
        # Orders over a 2-D numeric source.
        for order in ("C", "F", "A"):
            _conv("strconv", "astype", params={"dtype": "U", "casting": "unsafe", "order": order, "copy": True},
                  operands=[_num_operand(dt, "f2", values)], layout=f"{dt}:order{order}",
                  call=lambda a, o=order: a.astype("U", order=o))

    # B. Text to numbers: every text alone into every target of its kind (U and S), then layouts of valid texts, then the
    #    first failing element in NumPy's raw assignment order (reversed and stepped sources, two bad elements).
    for kind, texts in (("int", INT_TEXTS), ("float", FLOAT_TEXTS), ("complex", COMPLEX_TEXTS)):
        for i, text in enumerate(texts):
            for spec in ("U", "S"):
                if spec == "S":
                    try:
                        text.encode("ascii")
                    except UnicodeEncodeError:
                        continue
                for t in TARGETS[kind] + (["?"] if i % 7 == 0 else []):
                    _conv("strconv", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
                          operands=[_text_operand([text], spec)], layout=f"{kind}:{spec}{i}",
                          call=lambda a, t=t: a.astype(t))
    # Non-ASCII bytes: int() / float() of bytes run on the bytes; complex() decodes them as UTF-8 first.
    for raw in (b"\xff", b"\xc3\xa9", b"1\xff", b"\x80", b"\xe2\x82\xac1"):
        for t in ("i8", "f8", "c16"):
            _conv("strconv", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
                  operands=[plain(np.array([raw]))], layout="S:nonascii", call=lambda a, t=t: a.astype(t))
    # Long digit strings: CPython's 4300-digit limit for int(), none for float().
    for digits in (4299, 4300, 4301):
        txt = "1" * digits
        for t in ("i8", "f8", "c16"):
            _conv("strconv", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
                  operands=[plain(np.array([txt]))], layout=f"digits{digits}", call=lambda a, t=t: a.astype(t))
    # Valid texts through every layout.
    good = {"int": ["12", "-3", " 7", "1_0", "0", "99", "-128", "+4"],
            "float": ["1.5", "-0.0", "inf", "nan", "1e-5", " 2 ", "3.25", "1_0.5"],
            "complex": ["1+2j", "(3-4j)", "-j", "2", "nan", "1e3j", " 5 ", "0j"]}
    for kind, vals in good.items():
        for spec in ("U", "S"):
            for lay in CONV_LAYOUTS:
                n, build = LAYOUTS[lay]
                base = np.array([vals[i % len(vals)] for i in range(n)] if n else [], dtype=spec + "8")
                for t in TARGETS[kind]:
                    _conv("strconv", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
                          operands=[(base, build(base))], layout=f"{kind}:{spec}:{lay}",
                          call=lambda a, t=t: a.astype(t))
    # The first failing element: NumPy walks the destination's order (a reversed source is read backwards).
    for lay in ("c1", "r1", "s1", "rs", "f2", "t3"):
        n, build = LAYOUTS[lay]
        vals = ["1"] * n
        vals[1], vals[n - 2] = "x1", "x2"
        base = np.array(vals, dtype="U4")
        for t in ("i8", "f8", "c16", "u1"):
            _conv("strconv", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
                  operands=[(base, build(base))], layout=f"firstbad:{lay}", call=lambda a, t=t: a.astype(t))
    # The casting rules for text -> numbers: unsafe only.
    for t in ("i8", "f8", "c16", "?"):
        for casting in ("no", "equiv", "safe", "same_kind", "unsafe"):
            _conv("strconv", "astype", params={"dtype": t, "casting": casting, "order": "K", "copy": True},
                  operands=[_text_operand(["1", "2"], "U")], layout=f"text:{casting}",
                  call=lambda a, t=t, c=casting: a.astype(t, casting=c))

    # C. Element stores: numbers into text arrays (setitem / fill / full / ones), text into numeric arrays.
    num_values = [5, -1, 2.5, 1e16, True, False, 123456789, -9223372036854775808, 0.1, 1e-5, 1e22, -0.0]
    for spec in ("<U1", "<U3", "<U40", "|S3"):
        for v in num_values:
            for idx in ([{"int": 0}], [{"ellipsis": True}], [{"slice": [None, None, 2]}], [{"arr": [0, -1]}]):
                key = tuple(_py_index(e) for e in idx)
                _conv("strconv", "setitem", params={"index": idx, "value": enc(v)}, operands=[operand(spec, "c1")],
                      layout=spec, mutate=0, call=lambda a, k=key, v=v: _setitem(a, k, v))
            _conv("strconv", "fill", params={"value": enc(v)}, operands=[operand(spec, "c2")], layout=spec, mutate=0,
                  call=lambda a, v=v: a.fill(v))
            _conv("strconv", "full", params={"shape": [3], "fill": enc(v), "dtype": spec},
                  call=lambda v=v, d=spec: np.full((3,), v, dtype=d))
        for shape in ((3,), (2, 2), (0,), ()):
            _conv("strconv", "ones", params={"shape": list(shape), "dtype": spec},
                  call=lambda s=shape, d=spec: np.ones(s, dtype=d))
    for d in ("U", "S"):
        for v in (7, 2.5, True, -12):
            _conv("strconv", "full", params={"shape": [2], "fill": enc(v), "dtype": d},
                  call=lambda v=v, d=d: np.full((2,), v, dtype=d))
        _conv("strconv", "ones", params={"shape": [2], "dtype": d}, call=lambda d=d: np.ones((2,), dtype=d))
    # Text into numeric arrays: a Python str is parsed by the target's setitem (its error quotes the str itself).
    text_values = ["12", " 7 ", "x", "1.5", "300", "-1", "1+2j", "", "inf", "1_0", "١٢", b"34", b"x", b"\xff"]
    for dt in ("?", "i1", "u1", "i4", "i8", "u8", "f2", "f4", "f8", "c16"):
        for v in text_values:
            # The C# indexer takes a Python bytes only as the 0-d S array the replay builds (an ndarray): NumPy's error
            # text for an unparsable Python bytes quotes b'x', the array's np.bytes_(b'x'), so only a parsable one is
            # stored through the indexer (fill / full take the bytes object itself).
            if not (isinstance(v, bytes) and v != b"34"):
                _conv("strconv", "setitem", params={"index": [{"int": 0}], "value": enc(v)},
                      operands=[plain(np.zeros(3, dtype=dt))], layout=dt, mutate=0,
                      call=lambda a, v=v: _setitem(a, 0, v))
            _conv("strconv", "fill", params={"value": enc(v)}, operands=[plain(np.zeros(3, dtype=dt))], layout=dt,
                  mutate=0, call=lambda a, v=v: a.fill(v))
            _conv("strconv", "full", params={"shape": [2], "fill": enc(v), "dtype": dt},
                  call=lambda v=v, d=dt: np.full((2,), v, dtype=d))
        # A text ARRAY into a numeric array: the array cast (its errors quote the NumPy scalar).
        for vals in (["1", "2"], ["1", "x"], ["3.5", "nan"]):
            _conv("strconv", "setitem_array", params={"index": [{"ellipsis": True}]},
                  operands=[plain(np.zeros(2, dtype=dt)), plain(np.array(vals))], layout=dt, mutate=0,
                  call=lambda a, v: _setitem(a, Ellipsis, v))

    # D. np.copyto between text and numbers, every casting rule, with and without a mask, both directions.
    for dt, values in (("i8", CONV_VALUES["i8"]), ("f8", CONV_VALUES["f8"]), ("?", CONV_VALUES["?"]),
                       ("c16", CONV_VALUES["c16"]), ("f4", CONV_VALUES["f4"]), ("u1", CONV_VALUES["u1"])):
        for spec in ("<U3", "<U32", "|S8"):
            for casting in ("no", "safe", "same_kind", "unsafe"):
                for where in (None, [True, False, True, True, False, True]):
                    _conv("strconv", "copyto", params={"casting": casting, "where": where},
                          operands=[operand(spec, "c1"), _num_operand(dt, "c1", values)], layout=f"{spec}<-{dt}",
                          mutate=0, call=lambda d, s, c=casting, w=where: np.copyto(
                              d, s, casting=c, where=True if w is None else np.array(w)))
            # Reversed / broadcast sources.
            for lay in ("r1", "bc", "z0"):
                dst = operand(spec, "c2" if lay == "bc" else ("c1" if lay != "z0" else "c1"))
                _conv("strconv", "copyto", params={"casting": "unsafe", "where": None},
                      operands=[dst, _num_operand(dt, lay, values)], layout=f"{spec}<-{dt}:{lay}", mutate=0,
                      call=lambda d, s: np.copyto(d, s, casting="unsafe"))
    for dt in ("i8", "f8", "c16", "u1", "?"):
        for vals in (["1", "2", "3", "4", "5", "6"], ["1", "x", "3", "4", "5", "6"], [" 1", "-2", "3_0", "4", "5", "6"]):
            for casting in ("unsafe", "same_kind"):
                for where in (None, [True, False, True, True, False, True]):
                    _conv("strconv", "copyto", params={"casting": casting, "where": where},
                          operands=[plain(np.zeros(6, dtype=dt)), plain(np.array(vals))], layout=f"{dt}<-U", mutate=0,
                          call=lambda d, s, c=casting, w=where: np.copyto(
                              d, s, casting=c, where=True if w is None else np.array(w)))

    # E. np.array over mixed values and number <-> text dtypes.
    mixed = [["a", 1], [1, "ab"], [1.5, "ab"], [True, "x"], ["1", 2], [2, "1"], [1e16, "a"], [b"1", 2],
             [-2 ** 63, "a"], [[1, "a"], ["bc", 2.5]], ["12", "3"], [b"7", b"8"]]
    for vals in mixed:
        for dt in (None, "U", "S", "U3", "i8", "f8", "c16", "?"):
            _conv("strconv", "array", params={"values": enc(vals), "dtype": dt},
                  call=lambda v=vals, d=dt: np.array(v, dtype=d))
    for vals in ([1, 22, -333], [1.5, 1e16, 1e-5], [True, False]):
        for dt in ("U", "S", "U2", "S4"):
            _conv("strconv", "array", params={"values": enc(vals), "dtype": dt},
                  call=lambda v=vals, d=dt: np.array(v, dtype=d))

    # F. Ufuncs: numeric loops into a text out= (with and without where=), the string-request add over numbers, text
    #    comparisons and logical ufuncs into a text out=.
    f = num_operand(np.array([1.0, 2.5, -0.5, 4.0]))
    g = num_operand(np.array([1.0, 3.0, 0.5, 2.0]))
    ii = num_operand(np.array([1, 2, -3, 4]))
    uu = plain(np.array(["ab", "c", "de", "f"]))
    vv = plain(np.array(["ab", "x", "dd", "f"]))
    calls = [("add", [f, g]), ("add", [ii, ii]), ("subtract", [ii, f]), ("multiply", [f, ii]), ("power", [ii, ii]),
             ("floor_divide", [f, g]), ("maximum", [f, g]), ("minimum", [ii, ii]), ("equal", [f, g]), ("less", [ii, f]),
             ("logical_and", [f, ii]), ("logical_or", [ii, ii]), ("sqrt", [f]), ("negative", [ii]), ("absolute", [f]),
             ("isnan", [f]), ("isinf", [f]), ("rint", [f]), ("equal", [uu, vv]), ("less", [uu, vv]),
             ("logical_and", [uu, ii]), ("logical_xor", [uu, vv])]
    outs = ["U1", "U5", "U32", "S1", "S5", "S32"]
    for name, ins in calls:
        fn = getattr(np, name)
        for odt in outs:
            _conv("strconv", "ufunc_kw", params={"name": name, "dtype": None, "nin": len(ins), "out": len(ins)},
                  operands=ins + [target(np.zeros(4, odt))], layout=f"out{odt}", mutate=None,
                  call=lambda *xs, f=fn, n=len(ins): f(*xs[:n], out=xs[n]))
        # where= with an out whose text parses (selected slots written, masked-off kept) and one that does not.
        for wname, text in (("ok", ["7", "8", "9", "1"]), ("bad", ["7", "x", "9", "1"]), ("truth", ["", "a", "b", ""])):
            _conv("strconv", "ufunc_kw", params={"name": name, "dtype": None, "nin": len(ins), "out": len(ins),
                                                 "where": len(ins) + 1},
                  operands=ins + [target(np.array(text, dtype="U8")), num_operand(np.array([True, False, True, False]))],
                  layout=f"where:{wname}", call=lambda *xs, f=fn, n=len(ins): f(*xs[:n], out=xs[n], where=xs[n + 1]))
    # The out= broadcast (an out larger than the inputs) and a 0-d out.
    _conv("strconv", "ufunc_kw", params={"name": "add", "dtype": None, "nin": 2, "out": 2},
          operands=[f, num_operand(np.array(1.0)), target(np.zeros((3, 4), "U5"))], layout="out:bcast",
          call=lambda x, y, o: np.add(x, y, out=o))
    _conv("strconv", "ufunc_kw", params={"name": "add", "dtype": None, "nin": 2, "out": 2},
          operands=[num_operand(np.array(1.5)), num_operand(np.array(2.0)), target(np.zeros((), "U5"))], layout="out:0d",
          call=lambda x, y, o: np.add(x, y, out=o))
    # The string-request add: every numeric dtype with itself, with text, the casting refusals, an out.
    num_ops = {dt: num_operand(np.array(CONV_VALUES[dt][:4] if len(CONV_VALUES[dt]) >= 4 else
                                        (CONV_VALUES[dt] * 2)[:4], dtype=dt)) for dt in CONV_VALUES}
    for dt, op in num_ops.items():
        for req in ("U", "S"):
            _conv("strconv", "ufunc_kw", params={"name": "add", "dtype": req, "nin": 2}, operands=[op, op],
                  layout=f"req{req}:{dt}", call=lambda x, y, r=req: np.add(x, y, dtype=r))
            _conv("strconv", "ufunc_kw", params={"name": "add", "dtype": req, "nin": 2}, operands=[uu, op],
                  layout=f"req{req}:U+{dt}", call=lambda x, y, r=req: np.add(x, y, dtype=r))
            _conv("strconv", "ufunc_kw", params={"name": "add", "dtype": req, "nin": 2},
                  operands=[op, plain(np.array([b"ab", b"c", b"de", b"f"]))], layout=f"req{req}:{dt}+S",
                  call=lambda x, y, r=req: np.add(x, y, dtype=r))
        _conv("strconv", "ufunc_kw", params={"name": "add", "dtype": "U", "nin": 2, "out": 2},
              operands=[op, op, target(np.zeros(4, "U3"))], layout=f"reqU:{dt}:outU3",
              call=lambda x, y, o: np.add(x, y, out=o, dtype="U"))
        _conv("strconv", "ufunc_kw", params={"name": "add", "dtype": "U", "nin": 2, "out": 2, "where": 3},
              operands=[op, op, target(np.array(["p", "q", "r", "s"], dtype="U9")),
                        num_operand(np.array([True, False, True, False]))], layout=f"reqU:{dt}:where",
              call=lambda x, y, o, w: np.add(x, y, out=o, where=w, dtype="U"))
    # A numeric dtype= reduction over text parses first; numbers multiply / extremes compare.
    for vals in (["1.5", "2", "4"], ["1", "x"], ["3", "-1", "2"], [" 2 ", "1_0"], ["inf", "nan"], []):
        arr = plain(np.array(vals, dtype="U4"))
        for name in ("prod", "cumprod", "nancumprod", "maximum.reduce", "minimum.reduce"):
            for dt in ("f8", "i8", "c16"):
                # A complex product of inf and nan is NumPy's MSVC operand order per part (the NaN sign of each part),
                # a numeric residual pinned elsewhere ([OpenBugs] NaNOperandOrder): not this tier's business.
                if dt == "c16" and "nan" in vals:
                    continue
                _conv("strconv", "reduce", params={"name": name, "axis": None, "dtype": dt}, operands=[arr],
                      layout=f"{name}:{dt}", call=lambda a, n=name, d=dt: _strconv_reduce(n, a, d))
    # np.cov / np.corrcoef with a numeric dtype parse the text.
    # (A 2-D operand would end in the covariance's squeeze, whose OWNDATA is the array-object oracle's carved gap VA04.)
    for vals in (["1", "2", "4"], ["1", "x"], [" 1.5", "2_0", "4"], np.zeros((0, 2), "U1")):
        arr = plain(np.array(vals, dtype="U2") if not isinstance(vals, np.ndarray) else vals)
        for corr in (False, True):
            fn = np.corrcoef if corr else np.cov
            _conv("strconv", "cov", params={"corr": corr, "dtype": "f8"}, operands=[arr], layout="parse",
                  call=lambda a, f=fn: f(a, dtype="f8"))

    # G. Number <-> text inside movement routes: concatenate / where promote text with numbers (REQUIRED_STR_LEN),
    #    np.pad pads text with the str() of 0, matrix_power(text, 0) is an identity written as text.
    for dt in ("i8", "f8", "?", "c16", "u1"):
        op = num_ops[dt]
        _conv("strconv", "concatenate", params={"axis": 0}, operands=[uu, op], layout=f"U+{dt}",
              call=lambda x, y: np.concatenate([x, y]))
        _conv("strconv", "concatenate", params={"axis": 0}, operands=[op, plain(np.array([b"ab", b"c"]))],
              layout=f"{dt}+S", call=lambda x, y: np.concatenate([x, y]))
        _conv("strconv", "where", params={"cond": [True, False, True, False]}, operands=[uu, op], layout=f"U|{dt}",
              call=lambda x, y: np.where(np.array([True, False, True, False]), x, y))
    for spec in ("<U1", "<U3", "|S2"):
        for lay in ("c1", "c2"):
            _conv("strconv", "pad", params={"pad_width": 1, "mode": "constant", "constant": None},
                  operands=[operand(spec, lay)], layout=f"{spec}:{lay}", call=lambda a: np.pad(a, 1))
            for cv in (7, 2.5, True, -12):
                _conv("strconv", "pad", params={"pad_width": 1, "mode": "constant", "constant": enc(cv)},
                      operands=[operand(spec, lay)], layout=f"{spec}:{lay}:cv",
                      call=lambda a, cv=cv: np.pad(a, 1, constant_values=cv))
    for spec in ("<U1", "<U3", "|S2"):
        sq = plain(np.array([["a", "b"], ["c", "d"]], dtype=spec) if spec[1] == "U" else
                   np.array([[b"a", b"b"], [b"c", b"d"]], dtype=spec))
        _conv("strconv", "linalg1", params={"name": "matrix_power0"}, operands=[sq], layout=spec,
              call=lambda a: np.linalg.matrix_power(a, 0))


def _strconv_reduce(name, a, dt):
    """A reduction-style call by name with a dtype= request (the replay's ``reduce`` op names)."""
    if name == "maximum.reduce":
        return np.maximum.reduce(a, axis=None, dtype=dt)
    if name == "minimum.reduce":
        return np.minimum.reduce(a, axis=None, dtype=dt)
    return getattr(np, name)(a, axis=None, dtype=dt)


# ----------------------------------------------------------------------------------------------------------------------
# U. NEP 55 StringDType descriptors (strings phase 1, P1.2)
# ----------------------------------------------------------------------------------------------------------------------

_ABSENT = object()


def _jf(x):
    """A float as JSON (the corpus is written with ``allow_nan=False``).

    :param x: the float.
    :returns: ``"nan"``, ``"inf"`` or ``"-inf"`` for the non-finite values (the replay parses them back), else ``x``.
    """
    if x != x:
        return "nan"
    if x in (float("inf"), float("-inf")):
        return "inf" if x > 0 else "-inf"
    return x


def enc_na(v):
    """A StringDType ``na_object`` as JSON, so the replay rebuilds the C# value standing in for it.

    :param v: the Python object: anything ``enc`` encodes, a float (NaN / infinity included), an np.float16 (by its
        bits, so a NaN payload survives), a complex, a Python int past int64, a tuple or a list (element by element).
    :returns: the JSON-ready encoding.
    :raises TypeError: an object ``enc`` cannot encode either.
    """
    if isinstance(v, np.float16):
        return {"f2bits": int(np.array(v).view(np.uint16))}
    if isinstance(v, complex):
        return {"complex": [_jf(v.real), _jf(v.imag)]}
    if isinstance(v, float):
        return {"float": _jf(v)}
    if isinstance(v, int) and not isinstance(v, bool) and not (-2 ** 63 <= v < 2 ** 63):
        return {"bigint": str(v)}
    if isinstance(v, tuple):
        return {"tuple": [enc_na(x) for x in v]}
    if isinstance(v, list):
        return {"list": [enc_na(x) for x in v]}
    return enc(v)


class TSpec:
    """A StringDType descriptor request: ``np.dtypes.StringDType(na_object=..., coerce=...)`` with the sentinel absent
    unless given (NumPy distinguishes "not given" from ``None``). Building happens inside the recorded call, so a
    sentinel NumPy refuses (a lone surrogate) is a recorded error, not a generator crash.
    """

    def __init__(self, na=_ABSENT, coerce=True):
        """Records the request.

        :param na: the sentinel, or ``_ABSENT`` for none.
        :param coerce: NumPy's ``coerce``.
        """
        self.na, self.coerce = na, coerce

    def enc(self):
        """The request as JSON.

        :returns: ``{"T": {"coerce": bool, "na": <enc_na>}}``, ``na`` left out when the sentinel is absent.
        """
        d = {"coerce": self.coerce}
        if self.na is not _ABSENT:
            d["na"] = enc_na(self.na)
        return {"T": d}

    def build(self):
        """The NumPy descriptor.

        :returns: a fresh ``StringDType`` instance.
        :raises UnicodeEncodeError: a string sentinel (or a sentinel's ``str()``) with a lone surrogate.
        """
        kw = {"coerce": self.coerce}
        if self.na is not _ABSENT:
            kw["na_object"] = self.na
        return np.dtypes.StringDType(**kw)


def senc(s):
    """A descriptor request as JSON.

    :param s: a dtype string or a ``TSpec``.
    :returns: the string itself, or the TSpec's ``{"T": {...}}``.
    """
    return s if isinstance(s, str) else s.enc()


def sbuild(s):
    """The NumPy descriptor of a request.

    :param s: a dtype string or a ``TSpec``.
    :returns: ``np.dtype(s)`` for a string, the StringDType call for a TSpec.
    :raises TypeError: a dtype string NumPy does not understand (recorded as the case's error).
    """
    return np.dtype(s) if isinstance(s, str) else s.build()


def tprops(d):
    """The ``props`` expectation of a descriptor, with the attributes a StringDType adds.

    :param d: any NumPy descriptor.
    :returns: the attributes of ``dtype_props`` plus ``hasobject`` and ``flags``, and for a StringDType ``coerce``,
        ``has_na`` and (when the attribute exists) ``na_repr``, the sentinel's ``repr()``.
    """
    r = {"kind": "props", "str": d.str, "name": d.name, "char": d.char, "dkind": d.kind, "num": int(d.num),
         "itemsize": int(d.itemsize), "byteorder": d.byteorder, "isbuiltin": int(d.isbuiltin),
         "isnative": bool(d.isnative), "alignment": int(d.alignment), "repr": repr(d), "strfmt": str(d),
         "hasobject": bool(d.hasobject), "flags": int(d.flags)}
    if d.kind == "T":
        r["coerce"] = bool(d.coerce)
        r["has_na"] = hasattr(d, "na_object")
        if r["has_na"]:
            r["na_repr"] = repr(d.na_object)
    return r


_CATEGORIES = {"generic": np.generic, "character": np.character, "flexible": np.flexible, "number": np.number,
               "integer": np.integer, "inexact": np.inexact, "str_": np.str_, "bytes_": np.bytes_}


def gen_tdtype():
    """Section tdtype (strings phase 1, P1.2): the NEP 55 ``StringDType`` descriptor without arrays — its attributes
    for every kind of sentinel, equality, promotion (classes, common instances, sequences), the casting-rule table, the
    scalar hierarchy, ``np.isdtype``'s refusals, ``newbyteorder`` and the ufunc ``dtype=`` refusal. Every case is
    strict: none needs storage.
    """
    nan, inf = float("nan"), float("inf")
    sentinels = [None, nan, -nan, "NA", "", "None", "it's", 'q"z', "a\nb", "😀", 5, -7, 1.5, -0.0, 0.0, inf, -inf,
                 True, False, b"x", b"\x80", np.float16(1.5), np.float16(nan), np.float16(65504), complex(1, 2),
                 complex(nan, 0), complex(0, 1), 2 ** 70, -2 ** 64, (1,), (1, 2), (), [], ["a", None], ["\ud800"],
                 "\ud800", "a\udc00b", "x" * 40]
    t_specs = [TSpec(), TSpec(coerce=False)] + [TSpec(na) for na in sentinels] + \
              [TSpec(None, coerce=False), TSpec("NA", coerce=False), TSpec(nan, coerce=False)]

    # 1. descriptor attributes: np.dtype's spellings and every kind of sentinel (a lone surrogate in a string sentinel is
    #    CPython's encode error at construction).
    for spec in ["T", "<T", ">T", "=T", "|T", "T8", "T16", "StringDType", "StringDType()", "t", "TT", "T()", "T(2)",
                 "(2)T", "|T(2)", "U5", "f8"] + t_specs:
        add("tdtype", "t_props", params={"spec": senc(spec)}, call=lambda s=spec: _TProps(sbuild(s)))

    # 2. equality: every pair of a set of descriptors (NaN sentinels equal, 5 == 5.0, True == 1, None != absent,
    #    coerce differing, the same sentinel object vs another), and a descriptor against a dtype STRING (StringDType's
    #    __eq__ does not coerce one: T == 'T' is False, while U5 == 'U5' is True).
    eq_set = [TSpec(), TSpec(coerce=False), TSpec(None), TSpec(None, coerce=False), TSpec(nan), TSpec(-nan),
              TSpec("NA"), TSpec("NB"), TSpec(5), TSpec(5.0), TSpec(True), TSpec(1), TSpec(complex(1, 0)),
              TSpec(0.0), TSpec(-0.0), TSpec(b"x"), TSpec("x"), TSpec(np.float16(nan)), TSpec(2 ** 53 + 1),
              TSpec(float(2 ** 53)), TSpec((1, 2)), "T", "U5", "f8"]
    for a in eq_set:
        for b in eq_set:
            # "same": both descriptors are built from ONE sentinel object, whose identity na_eq_cmp tests first (an
            # np.float16 NaN equals itself only that way); the replay decodes the sentinel once for such a pair.
            prm = {"a": senc(a), "b": senc(b)}
            if a is b and isinstance(a, TSpec):
                prm["same"] = True
            add("tdtype", "t_eq", params=prm, call=lambda a=a, b=b: bool(sbuild(a) == sbuild(b)))
    for a in [TSpec(), TSpec(None), "U5", "f8"]:
        for b in ["T", "U", "U5", "str", "f8", "StringDType()"]:
            add("tdtype", "t_eq_str", params={"a": senc(a), "b": b}, call=lambda a=a, b=b: bool(sbuild(a) == b))

    # 3. promotion: T with every class (only U promotes), the common instance of two StringDTypes (compatible
    #    sentinels, coerce from the first), and sequences (NumPy's most-knowledgeable reduction, weak Python literals).
    others = ["U5", "U", "S5", "S", "?", "i1", "u1", "i8", "u8", "f2", "f4", "f8", "c16", "M8[s]", "m8[s]", "T"]
    for o in others:
        for x, y in (("T", o), (o, "T")):
            add("tdtype", "t_promote", params={"a": x, "b": y}, call=lambda x=x, y=y: np.promote_types(sbuild(x), sbuild(y)))
    inst = [TSpec(), TSpec(coerce=False), TSpec(None), TSpec(nan), TSpec("NA"), TSpec("NB"), TSpec(None, coerce=False),
            TSpec(5), TSpec(5.0), "U5"]
    for a in inst:
        for b in inst:
            add("tdtype", "t_promote", params={"a": senc(a), "b": senc(b)},
                call=lambda a=a, b=b: np.promote_types(sbuild(a), sbuild(b)))
    groups = [["T"], ["T", "U5"], ["U5", "T", "S2"], ["T", "U5", "i8"], ["i8", "T"], ["U5", "i8", "T"],
              ["T", "S5", "U5"], [TSpec(None), "T"], [TSpec(None), TSpec(nan)], ["T", TSpec(None, coerce=False), "U3"],
              ["S3", "T"], ["T", "T", "T"]]
    for g in groups:
        add("tdtype", "t_result", params={"specs": [senc(s) for s in g], "extra": []},
            call=lambda g=g: np.result_type(*[sbuild(s) for s in g]))
    for g, extra in [(["T"], [1]), (["T"], [1.5]), (["T"], [True]), (["T", "U5"], [1]), (["U5"], [1]),
                     (["T"], [complex(1, 2)])]:
        add("tdtype", "t_result", params={"specs": [senc(s) for s in g], "extra": [enc_na(e) for e in extra]},
            call=lambda g=g, extra=extra: np.result_type(*[sbuild(s) for s in g], *extra))

    # 4. casting rules: T against every class under every rule, and between StringDType instances (an NA to a no-NA
    #    descriptor is unsafe; any other pair is a no-cast).
    casts = ["U5", "U", "S5", "S", "?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "c16",
             "M8[s]", "m8[s]", "M8", "m8"]
    for o in casts:
        for x, y in (("T", o), (o, "T")):
            for casting in ("no", "equiv", "safe", "same_kind", "unsafe"):
                add("tdtype", "t_can_cast", params={"from": x, "to": y, "casting": casting},
                    call=lambda x=x, y=y, c=casting: bool(np.can_cast(sbuild(x), sbuild(y), c)))
    for a in [TSpec(), TSpec(None), TSpec(nan), TSpec("NA"), TSpec(coerce=False)]:
        for b in [TSpec(), TSpec(None), TSpec(nan), TSpec("NA"), TSpec(coerce=False)]:
            for casting in ("no", "equiv", "safe", "same_kind", "unsafe"):
                add("tdtype", "t_can_cast", params={"from": senc(a), "to": senc(b), "casting": casting},
                    call=lambda a=a, b=b, c=casting: bool(np.can_cast(sbuild(a), sbuild(b), c)))

    # 5. the scalar hierarchy: StringDType's type is Python's str (np.str_ derives from it), under no NumPy category.
    for a, b in [("T", "T"), ("U5", "T"), ("T", "U5"), ("S5", "T"), ("T", "S5"), ("i8", "T"), ("T", "i8"),
                 (TSpec(None), "T"), ("T", TSpec(coerce=False)), ("T", "U"), ("U3", "U")] + \
                [("T", c) for c in _CATEGORIES] + [("U5", c) for c in ("character", "str_")]:
        if isinstance(b, str) and b in _CATEGORIES:
            add("tdtype", "t_issubdtype", params={"a": senc(a), "category": b},
                call=lambda a=a, b=b: bool(np.issubdtype(sbuild(a), _CATEGORIES[b])))
        else:
            add("tdtype", "t_issubdtype", params={"a": senc(a), "b": senc(b)},
                call=lambda a=a, b=b: bool(np.issubdtype(sbuild(a), sbuild(b))))

    # 6. np.isdtype refuses a StringDType descriptor as the dtype (first) and as a kind.
    for dt, kinds in [("T", ["numeric"]), ("T", ["foo"]), ("T", []), ("f8", ["numeric"]), ("U5", ["numeric"])]:
        add("tdtype", "t_isdtype", params={"dtype": dt, "kinds": kinds},
            call=lambda dt=dt, k=kinds: bool(np.isdtype(sbuild(dt), tuple(k) if len(k) != 1 else k[0])))
    for dt, kinds in [("f8", ["T"]), ("f8", ["f8", "T"]), ("T", ["f8"])]:
        add("tdtype", "t_isdtype_dtypes", params={"dtype": dt, "kinds": kinds},
            call=lambda dt=dt, k=kinds: bool(np.isdtype(sbuild(dt), tuple(sbuild(x) for x in k) if len(k) != 1 else sbuild(k[0]))))

    # 7. byte order: new-style descriptors refuse newbyteorder (after the order code is parsed).
    for spec in ["T", TSpec(None)]:
        for order in ["S", "=", "<", ">", "|", "x"]:
            add("tdtype", "t_newbyteorder", params={"spec": senc(spec), "order": order},
                call=lambda s=spec, o=order: sbuild(s).newbyteorder(o))

    # 8. a ufunc refuses a new-style descriptor INSTANCE as dtype=.
    for spec in ["T", TSpec(None)]:
        add("tdtype", "t_ufunc_dtype", params={"spec": senc(spec), "name": "add"}, operands=[num_operand(np.arange(3))],
            call=lambda x, s=spec: np.add(x, x, dtype=sbuild(s)))


# ----------------------------------------------------------------------------------------------------------------------
# V. NEP 55 StringDType arrays (strings phase 1, P1.3 - P1.6)
# ----------------------------------------------------------------------------------------------------------------------

# Texts that reach every cell kind of static_string.c: short (<= 15 UTF-8 bytes, stored in the cell), medium (16 - 255
# bytes: a one-byte arena prefix) and long (> 255 bytes: an eight-byte prefix), both size boundaries (15 / 16, 255 / 256),
# multi-byte UTF-8 (the BYTE count decides the kind: 'é' * 8 is 16 bytes), embedded and trailing NULs (StringDType keeps
# them, unlike 'U'), quotes, tabs and supplementary-plane characters.
T_POOL = ["a", "m" * 16, "", "bc", "é" * 8, "def", "q" * 20, "Ü", "\U0001F600" * 5, "x y", "n" * 15, "中文",
          "p" * 255, "a\x00b", "r" * 256, "it's", "s" * 300, "\t", "長い文字列です" * 2, "ab\x00"]
# Short texts only: NumPy 2.4.2's put / putmask / choose copy StringDType cells RAW (no dup), which is correct only for
# short strings (an arena or heap cell then points into another array's allocator: garbage or a crash), so those calls
# get short values only; NumSharp's correct duplicate of every string is pinned by the unit tests for long ones.
T_SHORT = ["a", "bc", "", "def", "Ü", "x y", "it's", "\t", "中文", "a\x00b", "\U0001F600", "zz", "é", "ab\x00", "~", "Q"]

TD = TSpec()
TDN = TSpec(None)
TDNAN = TSpec(float("nan"))
TDS = TSpec("NA")
TDC = TSpec(coerce=False)
TDNC = TSpec(None, coerce=False)


def tspec_of(d):
    """The ``{"coerce", "na"}`` request that rebuilds a StringDType descriptor equal to ``d``.

    :param d: a StringDType descriptor.
    :returns: ``{"coerce": bool}`` plus ``"na"`` (``enc_na``) when the descriptor has an NA object.
    """
    r = {"coerce": bool(d.coerce)}
    if hasattr(d, "na_object"):
        r["na"] = enc_na(d.na_object)
    return r


def tvals(arr):
    """The elements of a StringDType array in C order: ``{"na": true}`` for an NA cell (its MISSING flag, read from the
    cells — a string NA object equal to some text is told apart this way), else the text.

    :param arr: a StringDType array of any layout.
    :returns: the encoded elements.
    """
    flat = np.array(arr, copy=True, order="C").reshape(-1)
    cells = flat.tobytes()
    return [{"na": True} if cells[16 * i + 15] & 0x80 else enc(flat[i]) for i in range(flat.size)]


def _describe_t(base, view):
    """A StringDType (base, view) pair as an operand: the descriptor's request, the base's VALUES (its cells are
    meaningful only with its own allocator) and the view's shape / element strides / element offset.

    :param base: the 1-D C-contiguous StringDType base.
    :param view: a view of it.
    :returns: the operand descriptor.
    """
    off = view.__array_interface__["data"][0] - base.__array_interface__["data"][0] if view.size else 0
    assert off % 16 == 0
    d = {"dtype": view.dtype.str, "t": tspec_of(base.dtype), "shape": [int(x) for x in view.shape],
         "strides": [int(s // 16) for s in view.strides], "offset": int(off // 16), "bufferSize": int(base.size),
         "values": tvals(base)}
    if not view.flags.writeable:
        d["writeable"] = False
    return d


def _t_array_expect(r, operands):
    """A StringDType array result: the descriptor (its ``str`` names the NA object and coerce), shape, the values with
    NA cells marked, element strides, flags and memory sharing with each operand.

    :param r: the StringDType array.
    :param operands: the call's operand views, or None.
    :returns: the expectation.
    """
    d = {"kind": "array", "dtype": r.dtype.str, "t": tspec_of(r.dtype), "shape": [int(x) for x in r.shape],
         "values": tvals(r), "strides": [int(s // 16) for s in r.strides], "c": bool(r.flags.c_contiguous),
         "f": bool(r.flags.f_contiguous), "writeable": bool(r.flags.writeable), "owndata": bool(r.flags.owndata)}
    if operands:
        d["shares"] = [bool(np.shares_memory(r, o)) if isinstance(o, np.ndarray) else False for o in operands]
    return d


class _TCells:
    """Marks a call whose expectation is the raw cells of a StringDType array (``tobytes()``), heap pointers masked."""

    def __init__(self, b):
        """Keeps the bytes.

        :param b: the cells as ``bytes``.
        """
        self.b = b


def _mask_heap(b):
    """StringDType cells with the bytes that hold a HEAP POINTER zeroed: a cell with ``OUTSIDE_ARENA | LONG`` (a heap
    string, or one freed into an empty / NA cell that keeps the stale pointer). Everything else is deterministic: a short
    string's text and length, an arena string's offset and size, the flags.

    :param b: the cells.
    :returns: the masked cells as hex.
    """
    out = bytearray(b)
    for i in range(0, len(out), 16):
        if out[i + 15] & 0x30 == 0x30:
            out[i:i + 8] = bytes(8)
    return out.hex()


def t_values(spec, n, seed=0, pool=None, na_every=4):
    """``n`` values cycling through ``pool`` from ``seed``; with an NA object, every ``na_every``-th one is that object.

    :param spec: the TSpec.
    :param n: the count.
    :param seed: the first pool index.
    :param pool: the texts (``T_POOL`` by default).
    :param na_every: the NA period (0 for none).
    :returns: the values.
    """
    pool = pool or T_POOL
    vals = [pool[(seed + i) % len(pool)] for i in range(n)]
    if spec.na is not _ABSENT and na_every:
        for i in range(n):
            if (seed + i) % na_every == na_every - 1:
                vals[i] = spec.na
    return vals


def t_operand(spec, layout, seed=0, pool=None, na_every=4):
    """A StringDType operand: a base built by ``np.array(values, dtype)`` and the layout's view of it.

    :param spec: the TSpec.
    :param layout: a ``LAYOUTS`` name.
    :param seed: the first pool index.
    :param pool: the texts.
    :param na_every: the NA period.
    :returns: the (base, view) pair.
    """
    n, build = LAYOUTS[layout]
    base = np.array(t_values(spec, n, seed, pool, na_every), dtype=spec.build())
    return base, build(base)


def t_plain(arr):
    """A StringDType operand of exactly ``arr``'s shape: a 1-D copy as the base and its reshaped view.

    :param arr: a StringDType array.
    :returns: the (base, view) pair.
    """
    base = np.array(arr, copy=True, order="C").reshape(-1)
    return base, base.reshape(arr.shape)


def _drop_platform_int_text():
    """Drops the case just added when NumPy's error names the C type of an int32 / uint32 target: casts.cpp names the
    target by its type number, which is NPY_LONG on Windows (``out of bounds for long``) and NPY_INT on LP64; NumSharp
    pins the LP64 text (unit-tested), the policy of ``_platform_text``."""
    err = CASES[-1].get("error")
    if err and err["text"].endswith(("is out of bounds for long", "is out of bounds for unsigned long")):
        _IDS.discard(CASES.pop()["id"])


def _t_writes(spec, values, writes):
    """Builds ``np.array(values, dtype)`` and applies each ``a[index] = value`` in order.

    :param spec: the TSpec.
    :param values: the initial values.
    :param writes: (index entries, value) pairs (index entries as ``_py_index`` reads them).
    :returns: the array.
    """
    a = np.array(values, dtype=spec.build())
    for idx, v in writes:
        a[tuple(_py_index(e) for e in idx)] = v
    return a


T_TEXT_SPECS = [TD, TDN, TDNAN, TDS]


def gen_tstore():
    """Section tstore (P1.3): the cells of StringDType arrays — what ``tobytes()`` shows after building an array from
    values (short / medium / long strings and their boundaries, NA cells, empty strings, 1-D and 2-D) and after write
    sequences that walk static_string.c's state machine (a shorter rewrite reuses its arena slot, a longer one moves to
    the heap, a short write forgets the arena slot, an NA write keeps it, the empty string keeps it); the values and
    layout of the same arrays; itemsize / nbytes / byte strides; and the raw cells of views. Heap pointers are masked
    (``_mask_heap``); everything else in a cell is compared byte for byte.
    """
    sets = [["a", "bc", "", "def"], ["n" * 15, "m" * 16, "o" * 255, "p" * 256, "q" * 300],
            ["é" * 7, "é" * 8, "\U0001F600" * 3, "\U0001F600" * 4, "中" * 5, "中" * 6], ["", "", ""],
            ["a\x00b", "ab\x00", "\x00", "x" * 15 + "\x00"], T_POOL, [["a", "m" * 16], ["", "q" * 20]],
            [[["x" * 20], ["y"]], [["z" * 30], [""]]], []]
    for spec in T_TEXT_SPECS + [TDC]:
        for vals in sets:
            if spec.na is not _ABSENT and vals is T_POOL:
                vals = t_values(spec, 20, 0, T_POOL, 3)
            for result in ("array", "cells"):
                add("tstore", "t_writes", params={"values": enc(vals), "dtype": senc(spec), "writes": [],
                                                  "result": result},
                    call=lambda s=spec, v=vals, r=result: _t_result(_t_writes(s, v, []), r))
        # NA cells among texts of every kind, built by np.array.
        if spec.na is not _ABSENT:
            vals = ["a", spec.na, "x" * 20, spec.na, "", "y" * 300, spec.na]
            for result in ("array", "cells"):
                add("tstore", "t_writes", params={"values": enc(vals), "dtype": senc(spec), "writes": [],
                                                  "result": result},
                    call=lambda s=spec, v=vals, r=result: _t_result(_t_writes(s, v, []), r))

    i0 = [{"int": 0}]
    seqs = [
        # a shorter rewrite of a medium string reuses its arena slot; an equal one too
        (["q" * 20, "a"], [(i0, "r" * 17)]), (["q" * 20, "a"], [(i0, "r" * 20)]),
        # a longer rewrite moves to the heap; the heap string is then rewritten shorter, longer, short, empty
        (["q" * 20, "a"], [(i0, "s" * 40)]), (["q" * 20, "a"], [(i0, "s" * 40), (i0, "t" * 30)]),
        (["q" * 20, "a"], [(i0, "s" * 40), (i0, "t" * 60)]), (["q" * 20, "a"], [(i0, "s" * 40), (i0, "u")]),
        (["q" * 20, "a"], [(i0, "s" * 40), (i0, "")]),
        # a short write forgets the arena slot: a later medium write goes to the heap
        (["q" * 20, "a"], [(i0, "x"), (i0, "y" * 18)]),
        # the empty string keeps an arena slot (size 0): a later medium write reuses it
        (["q" * 20, "a"], [(i0, ""), (i0, "w" * 16)]),
        # a long string's slot takes a medium rewrite; a longer one moves to the heap
        (["L" * 300, "a"], [(i0, "M" * 100)]), (["L" * 300, "a"], [(i0, "N" * 400)]),
        (["L" * 300, "a"], [(i0, "M" * 100), (i0, "O" * 290)]),
        # short to short, short to medium (arena space appended after the existing strings), medium after medium
        (["a", "b", "c"], [([{"int": 1}], "zz"), ([{"int": 2}], "k" * 17), (i0, "j" * 19)]),
        # a slice assignment packs each element; a fill, a negative index, a 2-D target
        (["a", "q" * 20, "b", "r" * 30], [([{"slice": [1, 3, None]}], "k" * 20)]),
        (["a", "q" * 20, "b", "r" * 30], [([{"slice": [None, None, -1]}], "z")]),
        (["a", "q" * 20, "b", "r" * 30], [([{"int": -1}], "v" * 25), ([{"int": -1}], "w" * 26)]),
        ([["a", "q" * 20], ["b", "r" * 30]], [([{"int": 1}, {"int": 0}], "c" * 40), ([{"int": 0}], "d")]),
        # non-ASCII rewrites (the BYTE size decides)
        (["é" * 8, "a"], [(i0, "é" * 9)]), (["é" * 8, "a"], [(i0, "e" * 16)]),
    ]
    for spec in (TD, TDN):
        for vals, writes in seqs:
            for result in ("array", "cells"):
                add("tstore", "t_writes",
                    params={"values": enc(vals), "dtype": senc(spec), "result": result,
                            "writes": [[idx, enc(v)] for idx, v in writes]},
                    call=lambda s=spec, v=vals, w=writes, r=result: _t_result(_t_writes(s, v, w), r))
    # NA writes keep the old allocation's metadata (a later write can reuse the arena slot); an NA overwritten by NA.
    na_seqs = [lambda na: (["q" * 20, "a"], [(i0, na)]), lambda na: (["q" * 20, "a"], [(i0, na), (i0, "z" * 19)]),
               lambda na: (["q" * 20, "a"], [(i0, na), (i0, "z" * 30)]), lambda na: (["a", "b"], [(i0, na), (i0, "c")]),
               lambda na: (["q" * 20, "a"], [(i0, "s" * 40), (i0, na)]),
               lambda na: (["q" * 20, "a"], [(i0, na), (i0, na)]),
               lambda na: (["a", "b", "c"], [([{"slice": [None, None, 2]}], na)])]
    for spec in (TDN, TDNAN, TDS):
        for mk in na_seqs:
            vals, writes = mk(spec.na)
            for result in ("array", "cells"):
                add("tstore", "t_writes",
                    params={"values": enc(vals), "dtype": senc(spec), "result": result,
                            "writes": [[idx, enc(v)] for idx, v in writes]},
                    call=lambda s=spec, v=vals, w=writes, r=result: _t_result(_t_writes(s, v, w), r))

    # itemsize / nbytes / byte strides (not of an empty array: NumPy reports all-zero strides for any zero-size array it
    # allocates, a library-wide divergence pinned in OpenBugs.EmptyStrides), and tobytes() of views: a buffer already in
    # the requested order gives its own cells, any other layout is RAVELLED first (a copy that duplicates the strings in
    # that order into a new arena, so the offsets show the walk order).
    for spec in (TD, TDN):
        for lay in ("c1", "c2", "f2", "t3", "s1", "r1", "r2", "o1", "bc", "z0", "d5"):
            add("tstore", "t_sizes", operands=[t_operand(spec, lay)], layout=lay,
                call=lambda a: [int(a.itemsize), int(a.nbytes)] + [int(s) for s in a.strides])
        for lay in ("c1", "c2", "f2", "t3", "s1", "r1", "r2", "o1", "bc", "z0", "e1", "e2", "d5"):
            for order in ("C", "F"):
                add("tstore", "t_tobytes", params={"order": order}, operands=[t_operand(spec, lay)], layout=lay,
                    call=lambda a, o=order: _TCells(a.tobytes(order=o)))
        # the cells of copies: np.copy duplicates the strings walking the destination's order
        for lay in ("c2", "f2", "t3", "s2", "r2", "bc"):
            for order in ("C", "F", "K"):
                add("tstore", "t_copy_cells", params={"order": order}, operands=[t_operand(spec, lay)], layout=lay,
                    call=lambda a, o=order: _TCells(np.copy(a, order=o).tobytes(order="A")))


def _t_result(a, result):
    """The array itself, or its cells.

    :param a: a StringDType array.
    :param result: ``"array"`` or ``"cells"``.
    :returns: ``a`` or ``_TCells(a.tobytes())``.
    """
    return a if result == "array" else _TCells(a.tobytes())


def gen_tcreate():
    """Section tcreate (P1.5): creating StringDType arrays (``np.array`` from values of every kind under every
    descriptor — numbers, bools and None coerced to their ``str()`` or refused by ``coerce=False``, the NA objects, bytes
    decoded as UTF-8, lone surrogates and invalid UTF-8 refused, nested / ragged / empty lists, ``ndmin``, ``order``;
    ``np.zeros`` / ``empty`` / ``ones`` / ``full`` and the ``_like`` forms, ``asarray`` / ``ascontiguousarray`` /
    ``asfortranarray``), element access (``tolist``, ``item``, iteration, integer indexing — the NA object comes back
    as itself), setitem of every value kind, ``fill``, printing (repr / str / array2string / array_repr / array_str with
    NA objects, escapes and summarization), ``view`` (NumPy's reference refusal) and ``np.nditer``'s ``refs_ok`` rule.
    """
    nan = float("nan")
    value_sets = [["a", "bc", "def"], ["", ""], ["x" * 20, "y" * 300, "z"], ["Ü", "é", "\U0001F600ab"],
                  ["a\x00b", "c\x00"], [["a", "bc"], ["def", ""]], [[["a"], ["b" * 20]], [["cc"], ["d"]]], [],
                  [[]], [["a", "b"], ["c"]], ["a", 1, 2.5, True, None], [5], [None], [nan, 1.5], [True, False],
                  [b"ab", b"\xc3\xa9"], [b"\x80"], ["\ud800"], ["ok", "\udc80x"], ["NA", "x"], [-0.0, 1e16, 1e-5],
                  [123456789, -7]]
    for spec in T_TEXT_SPECS + [TDC, TDNC]:
        for vals in value_sets:
            add("tcreate", "array", params={"values": enc(vals), "dtype": senc(spec)},
                call=lambda v=vals, s=spec: np.array(v, dtype=s.build()))
    for vals in (["a", "b" * 20], [["a", "bc"], ["d" * 20, "e"]]):
        for ndmin in (0, 1, 2, 3):
            add("tcreate", "array", params={"values": enc(vals), "dtype": senc(TD), "ndmin": ndmin},
                call=lambda v=vals, n=ndmin: np.array(v, dtype=TD.build(), ndmin=n))
        for order in ("C", "F"):
            add("tcreate", "array", params={"values": enc(vals), "dtype": senc(TDN), "order": order},
                call=lambda v=vals, o=order: np.array(v, dtype=TDN.build(), order=o))

    for v in ["abc", "", "x" * 20, None, 5, 2.5, True, b"ab", nan, "NA"]:
        for spec in T_TEXT_SPECS + [TDC]:
            add("tcreate", "array_scalar", params={"value": enc(v), "dtype": senc(spec)},
                call=lambda v=v, s=spec: np.array(v, dtype=s.build()))

    for spec in (TD, TDN, TDC):
        for shape in ((3,), (2, 3), (0,), (2, 0), ()):
            for fn in ("zeros", "empty", "ones"):
                add("tcreate", fn, params={"shape": list(shape), "dtype": senc(spec)},
                    call=lambda f=fn, sh=shape, s=spec: getattr(np, f)(sh, dtype=s.build()))
    for spec in T_TEXT_SPECS + [TDC, TDNC]:
        for fill in ("ab", "x" * 20, "", 7, -2.5, True, None, nan, b"ab", "NA"):
            for shape in ((3,), (2, 2)):
                add("tcreate", "full", params={"shape": list(shape), "fill": enc(fill), "dtype": senc(spec)},
                    call=lambda sh=shape, f=fill, s=spec: np.full(sh, f, dtype=s.build()))

    for spec in (TD, TDN):
        for lay in ("c2", "f2", "t3", "s2", "r2", "bc", "z0", "e2"):
            add("tcreate", "zeros_like", params={"dtype": None}, operands=[t_operand(spec, lay)], layout=lay,
                call=lambda a: np.zeros_like(a))
            add("tcreate", "empty_like", params={"dtype": None}, operands=[t_operand(spec, lay)], layout=lay,
                call=lambda a: np.empty_like(a))
            add("tcreate", "ones_like", operands=[t_operand(spec, lay)], layout=lay, call=lambda a: np.ones_like(a))
            for fill in ("qq", "w" * 20, 5, None):
                add("tcreate", "full_like", params={"fill": enc(fill), "dtype": None}, operands=[t_operand(spec, lay)],
                    layout=lay, call=lambda a, f=fill: np.full_like(a, f))
    for lay in ("c2", "f2", "bc"):
        for spec in (TD, TDN):
            add("tcreate", "zeros_like", params={"dtype": senc(spec)}, operands=[operand("<U3", lay)], layout=lay,
                call=lambda a, s=spec: np.zeros_like(a, dtype=s.build()))
            add("tcreate", "full_like", params={"fill": enc("zz"), "dtype": senc(spec)}, operands=[operand("<U3", lay)],
                layout=lay, call=lambda a, s=spec: np.full_like(a, "zz", dtype=s.build()))
            add("tcreate", "full_like", params={"fill": enc(None), "dtype": senc(spec)},
                operands=[plain(np.arange(6).reshape(2, 3))], layout="c2",
                call=lambda a, s=spec: np.full_like(a, None, dtype=s.build()))

    for spec in (TD, TDN):
        for lay in CORE_LAYOUTS + ["c3", "r1", "rs", "e1"]:
            add("tcreate", "asarray", params={"dtype": None}, operands=[t_operand(spec, lay)], layout=lay,
                call=lambda a: np.asarray(a))
            add("tcreate", "asarray", params={"dtype": senc(spec)}, operands=[t_operand(spec, lay)], layout=lay,
                call=lambda a, s=spec: np.asarray(a, dtype=s.build()))
            add("tcreate", "ascontiguousarray", operands=[t_operand(spec, lay)], layout=lay,
                call=lambda a: np.ascontiguousarray(a))
            add("tcreate", "asfortranarray", operands=[t_operand(spec, lay)], layout=lay,
                call=lambda a: np.asfortranarray(a))
        for lay in ("c1", "c2", "f2"):
            add("tcreate", "asarray", params={"dtype": senc(spec)}, operands=[operand("<U3", lay)], layout=lay,
                call=lambda a, s=spec: np.asarray(a, dtype=s.build()))

    # Element access: the NA object comes back as itself (None, nan, the string), text as a str.
    for spec in T_TEXT_SPECS:
        for lay in CORE_LAYOUTS + ["c3", "r1", "rs", "e1"]:
            op = t_operand(spec, lay)
            add("tcreate", "tolist", operands=[op], layout=lay, call=lambda a: a.tolist())
            if op[1].size:
                add("tcreate", "item", params={"index": 0}, operands=[op], layout=lay, call=lambda a: a.item(0))
                add("tcreate", "item", params={"index": op[1].size - 1}, operands=[op], layout=lay,
                    call=lambda a: a.item(a.size - 1))
            if op[1].size == 1:
                add("tcreate", "item", params={"index": None}, operands=[op], layout=lay, call=lambda a: a.item())
            if op[1].ndim == 1:
                add("tcreate", "iter", operands=[op], layout=lay, call=lambda a: [x for x in a])
                for k in range(min(op[1].size, 4)):
                    add("tcreate", "getitem", params={"index": [{"int": k}]}, operands=[op], layout=lay,
                        call=lambda a, k=k: a[k])
    add("tcreate", "item", params={"index": 99}, operands=[t_operand(TD, "c1")], layout="c1", call=lambda a: a.item(99))

    # setitem: every value kind into every descriptor (a literal is stringdtype_setitem's: str() of any value with coerce,
    # the NA object packs NA, coerce=False refuses anything but str, bytes decode as UTF-8), through every index kind.
    values = ["zz", "", "longer text here!", "Ü", "\U0001F600" * 4, "a\x00", 5, -2.5, True, None, nan, b"by", b"\xff",
              "\ud800", "NA"]
    for spec in T_TEXT_SPECS + [TDC]:
        for lay in ("c1", "c2", "f2", "s1", "r2", "bc", "z0", "e2"):
            for v in values:
                for idx in ([{"int": 0}], [{"ellipsis": True}], [{"slice": [None, None, 2]}], [{"arr": [0, -1]}],
                            [{"mask": [True, False, True, False, False, True]}]):
                    if "mask" in idx[0] and lay != "c1":
                        continue
                    if lay == "z0" and idx[0].keys() & {"int", "slice", "arr"}:
                        continue
                    # A Python bytes value only through an element index: C# reaches one element with the object-typed
                    # SetValue (Python's bytes is a byte[] there), while an NDArray-typed store has no spelling for a bytes
                    # object (a 0-d S array is NumPy's ndarray, which is CAST, not stored by setitem).
                    if isinstance(v, bytes) and not ("int" in idx[0] and lay in ("c1", "s1")):
                        continue
                    key = tuple(_py_index(e) for e in idx)
                    add("tcreate", "t_setitem", params={"index": idx, "value": enc(v)},
                        operands=[t_operand(spec, lay)], layout=lay, mutate=0,
                        call=lambda a, k=key, v=v: _setitem(a, k, v))
            for v in ("fill text over twenty", "f", 7, None, nan, b"\xc3\xa9", "NA"):
                add("tcreate", "fill", params={"value": enc(v)}, operands=[t_operand(spec, lay)], layout=lay,
                    mutate=0, call=lambda a, v=v: a.fill(v))
        # a whole array into one element is its str(); arrays into slices are cast
        add("tcreate", "setitem_array", params={"index": [{"int": 0}]},
            operands=[t_operand(spec, "c1"), plain(np.array(["zz", "y"]))], layout="c1", mutate=0,
            call=lambda a, v: _setitem(a, 0, v))
        add("tcreate", "setitem_array", params={"index": [{"slice": [None, 2, None]}]},
            operands=[t_operand(spec, "c1"), plain(np.array([1, 2]))], layout="c1", mutate=0,
            call=lambda a, v: _setitem(a, slice(None, 2), v))

    # Printing: escapes, quotes, NULs, long strings, the NA objects, 0-d, 2-D, empty, summarization.
    texts = ["", "a", "it's", 'say "hi"', "both'\"", "\t\n\r", "\x00a", "\x01\x1f\x7f", "\x80\x9f", "\xe9",
             chr(0x4E2D), chr(0x1F600), chr(0x2028), chr(0xFEFF), "\\", "x" * 30, "tab\there"]
    for spec in T_TEXT_SPECS:
        arrs = [np.array(texts, dtype=spec.build()), np.array(t_values(spec, 10, 0, T_POOL, 3), dtype=spec.build())]
        if spec.na is not _ABSENT:
            arrs.append(np.array([spec.na, "a", spec.na], dtype=spec.build()))
            arrs.append(np.array(spec.na, dtype=spec.build()))
        arrs += [np.array("abc", dtype=spec.build()), np.array([["a", "bb"], ["ccc", "d"]], dtype=spec.build()),
                 np.empty((0,), dtype=spec.build()), np.empty((2, 0), dtype=spec.build())]
        for arr in arrs:
            for fn in ("repr", "str"):
                add("tcreate", fn, operands=[plain(arr)], layout=f"{arr.ndim}d",
                    call=lambda a, f=fn: repr(a) if f == "repr" else str(a))
        for lay in ("c2", "f2", "t3", "r2", "bc", "z0", "e2", "d5"):
            op = t_operand(spec, lay)
            add("tcreate", "repr", operands=[op], layout=lay, call=lambda a: repr(a))
            add("tcreate", "str", operands=[op], layout=lay, call=lambda a: str(a))
    long_t = plain(np.array([f"item{i}" for i in range(40)] + [None], dtype=TDN.build()))
    for th, ed, lw in ((10, 2, 75), (5, 1, 30), (1000, 3, 20)):
        add("tcreate", "repr_opts", params={"threshold": th, "edgeitems": ed, "linewidth": lw}, operands=[long_t],
            layout="c1", call=lambda a, t=th, e=ed, l=lw: _with_opts(repr, a, t, e, l))
        add("tcreate", "str_opts", params={"threshold": th, "edgeitems": ed, "linewidth": lw}, operands=[long_t],
            layout="c1", call=lambda a, t=th, e=ed, l=lw: _with_opts(str, a, t, e, l))
    for lw in (20, 40):
        add("tcreate", "array2string", params={"max_line_width": lw, "separator": ", ", "prefix": "x("},
            operands=[long_t], layout="c1",
            call=lambda a, l=lw: np.array2string(a, max_line_width=l, separator=", ", prefix="x("))
        add("tcreate", "array_repr", params={"max_line_width": lw}, operands=[long_t], layout="c1",
            call=lambda a, l=lw: np.array_repr(a, max_line_width=l))
        add("tcreate", "array_str", params={"max_line_width": lw}, operands=[long_t], layout="c1",
            call=lambda a, l=lw: np.array_str(a, max_line_width=l))

    # view: the array's own descriptor is a view sharing its allocator; any other dtype is NumPy's reference refusal (both
    # directions). A DIFFERENT-but-equal StringDType instance is left out: NumPy 2.4.2 returns a view that reads its cells
    # through the NEW instance's empty allocator, so every arena string reads '' (NumSharp keeps the array's allocator and
    # reads the texts; unit-tested).
    for spec in (TD, TDN):
        for lay in ("c1", "f2", "s1", "bc", "z0"):
            add("tcreate", "t_view_same", operands=[t_operand(spec, lay)], layout=lay, call=lambda a: a.view(a.dtype))
            for t in (senc(TD if spec is TDN else TDN), "<U4", "|S16", "<i8", "<c16"):
                add("tcreate", "view", params={"dtype": t}, operands=[t_operand(spec, lay)], layout=lay,
                    call=lambda a, t=t: a.view(sbuild(t) if isinstance(t, str) else TSpec(**_tspec_kw(t)).build()))
    for t in (senc(TD), senc(TDN)):
        add("tcreate", "view", params={"dtype": t}, operands=[operand("<U4", "c1")], layout="c1",
            call=lambda a, t=t: a.view(TSpec(**_tspec_kw(t)).build()))
        add("tcreate", "view", params={"dtype": t}, operands=[plain(np.zeros(4, dtype="<c16"))], layout="c1",
            call=lambda a, t=t: a.view(TSpec(**_tspec_kw(t)).build()))

    # np.nditer: a StringDType operand needs refs_ok; with it, the walk yields 0-d views of the elements.
    for spec in (TD, TDN):
        for lay in ("c1", "c2", "f2", "r1", "bc", "e1"):
            add("tcreate", "t_nditer", params={"flags": []}, operands=[t_operand(spec, lay)], layout=lay,
                call=lambda a: [x[()] for x in np.nditer(a)])
            add("tcreate", "t_nditer", params={"flags": ["refs_ok"]}, operands=[t_operand(spec, lay)], layout=lay,
                call=lambda a: [x[()] for x in np.nditer(a, flags=["refs_ok"])])
            add("tcreate", "t_nditer", params={"flags": ["refs_ok", "zerosize_ok"]}, operands=[t_operand(spec, lay)],
                layout=lay, call=lambda a: [x[()] for x in np.nditer(a, flags=["refs_ok", "zerosize_ok"])])

    # Raw-buffer routes NumPy refuses for a reference dtype.
    add("tcreate", "t_memoryview", operands=[t_operand(TD, "c1")], layout="c1", call=lambda a: memoryview(a))
    add("tcreate", "t_frombuffer", params={"buffer": (b"\x00" * 32).hex(), "dtype": senc(TD)},
        call=lambda: np.frombuffer(bytearray(32), dtype=TD.build()))


def _tspec_kw(enc_spec):
    """The ``TSpec`` keywords of an encoded request (the inverse of ``TSpec.enc`` for the sentinels used here).

    :param enc_spec: ``{"T": {"coerce": bool, "na": ...}}``.
    :returns: ``{"coerce": ..., "na": ...}`` for ``TSpec(**kw)``.
    """
    t = enc_spec["T"]
    kw = {"coerce": t["coerce"]}
    if "na" in t:
        na = t["na"]
        kw["na"] = None if na == {"none": True} else float(na["float"]) if "float" in na else \
            bytes.fromhex(na["u"]).decode("utf-32-le", "surrogatepass")
    return kw


def gen_tmove():
    """Section tmove (P1.4): moving StringDType elements — the views (transposes, reshapes, flips, broadcasts, basic and
    advanced indexing: a view shares its base's allocator), the copies of every layout and order, and every movement
    function over operands holding short, arena and long strings and NA cells: repeat, tile, roll, resize, take, compress,
    delete, insert, append, tril / triu, take_along_axis, split, diag / diagflat, pad, concatenate and the stack family
    (StringDType with StringDType / U / S / numbers: NumPy's promotion and its common-instance error), where, choose,
    select, extract, meshgrid, broadcast_arrays, unstack; and the in-place writes: setitem of arrays, put, putmask,
    copyto (every casting rule, masks), put_along_axis, fill_diagonal, ndarray.resize. put / putmask / choose get short
    texts only (``T_SHORT``: NumPy 2.4.2 copies their cells raw).
    """
    specs = [TD, TDN]
    for spec in specs:
        for lay in ("c2", "c3", "f2", "t3", "s2", "r2", "col", "nx", "bc", "z0", "e2", "d5"):
            op = t_operand(spec, lay)
            shape = op[1].shape
            add("tmove", "transpose", params={"axes": None}, operands=[op], layout=lay, call=lambda a: a.transpose())
            if len(shape) >= 2:
                add("tmove", "swapaxes", params={"a1": 0, "a2": -1}, operands=[op], layout=lay,
                    call=lambda a: np.swapaxes(a, 0, -1))
                add("tmove", "moveaxis", params={"src": 0, "dst": -1}, operands=[op], layout=lay,
                    call=lambda a: np.moveaxis(a, 0, -1))
                add("tmove", "rot90", params={"k": 1}, operands=[op], layout=lay, call=lambda a: np.rot90(a))
                add("tmove", "diagonal", params={"offset": 0}, operands=[op], layout=lay, call=lambda a: np.diagonal(a))
            for order in ("C", "F", "A", "K"):
                add("tmove", "ravel", params={"order": order}, operands=[op], layout=lay,
                    call=lambda a, o=order: np.ravel(a, order=o))
            for order in ("C", "F"):
                add("tmove", "reshape", params={"shape": [-1], "order": order}, operands=[op], layout=lay,
                    call=lambda a, o=order: np.reshape(a, (-1,), order=o))
            add("tmove", "squeeze", params={"axis": None}, operands=[op], layout=lay, call=lambda a: np.squeeze(a))
            add("tmove", "expand_dims", params={"axis": 0}, operands=[op], layout=lay,
                call=lambda a: np.expand_dims(a, 0))
            add("tmove", "flip", params={"axis": None}, operands=[op], layout=lay, call=lambda a: np.flip(a))
            add("tmove", "atleast_2d", operands=[op], layout=lay, call=lambda a: np.atleast_2d(a))
            add("tmove", "atleast_3d", operands=[op], layout=lay, call=lambda a: np.atleast_3d(a))
        for lay in ("c1", "c2", "r1"):
            op = t_operand(spec, lay)
            add("tmove", "broadcast_to", params={"shape": [2] + list(op[1].shape)}, operands=[op], layout=lay,
                call=lambda a: np.broadcast_to(a, (2,) + a.shape))

        idx_cases = {
            "c1": [[{"int": 1}], [{"slice": [1, 4, None]}], [{"slice": [None, None, -2]}], [{"ellipsis": True}],
                   [{"none": True}], [{"arr": [0, 2, -1]}], [{"arr": [[0, 1], [5, 4]]}],
                   [{"mask": [True, False, True, False, False, True]}], [{"int": 7}], [{"arr": [6]}]],
            "c2": [[{"int": 1}], [{"slice": [None, None, None]}, {"int": 0}], [{"ellipsis": True}, {"int": -1}],
                   [{"arr": [2, 0]}], [{"arr": [0, 2]}, {"arr": [1, 3]}],
                   [{"mask2": [[True, False, False, True], [False] * 4, [True] * 4]}], [{"none": True}, {"int": 0}]],
            "f2": [[{"int": 1}], [{"slice": [None, None, None]}, {"int": 2}], [{"arr": [2, 0]}]],
            "r2": [[{"int": 0}], [{"arr": [1, 2]}], [{"slice": [None, None, 2]}, {"slice": [1, None, None]}]],
            "bc": [[{"int": 2}], [{"arr": [0, 0, 2]}], [{"slice": [None, None, None]}, {"int": 3}]],
            "z0": [[{"ellipsis": True}], [{"none": True}]],
        }
        for lay, idxs in idx_cases.items():
            for idx in idxs:
                key = tuple(_py_index(e) for e in idx)
                add("tmove", "getitem", params={"index": idx}, operands=[t_operand(spec, lay)], layout=lay,
                    call=lambda a, k=key: a[k])

        for lay in CORE_LAYOUTS + ["c3", "f3", "r1", "rs", "nx", "bcol", "d5", "e3", "one"]:
            op = t_operand(spec, lay)
            nd = op[1].ndim
            for order in ("C", "F", "A", "K"):
                add("tmove", "copy", params={"order": order}, operands=[op], layout=lay,
                    call=lambda a, o=order: np.copy(a, order=o))
            for order in ("C", "F"):
                add("tmove", "flatten", params={"order": order}, operands=[op], layout=lay,
                    call=lambda a, o=order: a.flatten(order=o))
            add("tmove", "repeat", params={"repeats": 2, "axis": None}, operands=[op], layout=lay,
                call=lambda a: np.repeat(a, 2))
            add("tmove", "tile", params={"reps": [2, 1]}, operands=[op], layout=lay, call=lambda a: np.tile(a, (2, 1)))
            add("tmove", "roll", params={"shift": 2, "axis": None}, operands=[op], layout=lay,
                call=lambda a: np.roll(a, 2))
            add("tmove", "resize", params={"shape": [5]}, operands=[op], layout=lay, call=lambda a: np.resize(a, (5,)))
            add("tmove", "take", params={"indices": [0, -1, 1], "axis": None, "mode": "raise"}, operands=[op],
                layout=lay, call=lambda a: np.take(a, [0, -1, 1]))
            add("tmove", "take", params={"indices": [5, -7, 100], "axis": None, "mode": "wrap"}, operands=[op],
                layout=lay, call=lambda a: np.take(a, [5, -7, 100], mode="wrap"))
            add("tmove", "take", params={"indices": [5, -7, 100], "axis": None, "mode": "clip"}, operands=[op],
                layout=lay, call=lambda a: np.take(a, [5, -7, 100], mode="clip"))
            if nd >= 1:
                add("tmove", "repeat", params={"repeats": [2], "axis": 0}, operands=[op], layout=lay,
                    call=lambda a: np.repeat(a, [2], axis=0))
                if op[1].shape[0] >= 1:
                    reps = [k % 3 for k in range(op[1].shape[0])]
                    add("tmove", "repeat", params={"repeats": reps, "axis": 0}, operands=[op], layout=lay,
                        call=lambda a, r=reps: np.repeat(a, r, axis=0))
                add("tmove", "roll", params={"shift": -1, "axis": 0}, operands=[op], layout=lay,
                    call=lambda a: np.roll(a, -1, axis=0))
                add("tmove", "take", params={"indices": [1, 0], "axis": 0, "mode": "clip"}, operands=[op],
                    layout=lay, call=lambda a: np.take(a, [1, 0], axis=0, mode="clip"))
                add("tmove", "compress", params={"condition": [True, False, True], "axis": 0}, operands=[op],
                    layout=lay, call=lambda a: np.compress([True, False, True], a, axis=0))
                add("tmove", "flip", params={"axis": 0}, operands=[op], layout=lay, call=lambda a: np.flip(a, 0))
                add("tmove", "delete", params={"obj": [0], "axis": 0}, operands=[op], layout=lay,
                    call=lambda a: np.delete(a, [0], axis=0))
                add("tmove", "insert", params={"obj": 1, "values": enc("inserted text value"), "axis": 0},
                    operands=[op], layout=lay, call=lambda a: np.insert(a, 1, "inserted text value", axis=0))
                add("tmove", "append", params={"values": enc(["zz", "w" * 20]), "axis": None}, operands=[op],
                    layout=lay, call=lambda a: np.append(a, ["zz", "w" * 20]))
            if nd >= 2:
                add("tmove", "tril", params={"k": 0}, operands=[op], layout=lay, call=lambda a: np.tril(a))
                add("tmove", "triu", params={"k": 1}, operands=[op], layout=lay, call=lambda a: np.triu(a, 1))
                add("tmove", "take_along_axis", params={"indices": np.zeros((1,) * (nd - 1) + (2,), int).tolist(),
                                                        "axis": -1}, operands=[op], layout=lay,
                    call=lambda a: np.take_along_axis(a, np.zeros((1,) * (a.ndim - 1) + (2,), int), -1))
                add("tmove", "split", params={"sections": 2, "axis": -1}, operands=[op], layout=lay,
                    call=lambda a: np.array_split(a, 2, axis=-1))
            if nd == 1:
                add("tmove", "diag", params={"k": 0}, operands=[op], layout=lay, call=lambda a: np.diag(a))
                add("tmove", "diag", params={"k": -1}, operands=[op], layout=lay, call=lambda a: np.diag(a, -1))
                add("tmove", "diagflat", params={"k": 1}, operands=[op], layout=lay, call=lambda a: np.diagflat(a, 1))
            for mode in ("edge", "reflect", "symmetric", "wrap"):
                add("tmove", "pad", params={"pad_width": 1, "mode": mode, "constant": None}, operands=[op],
                    layout=lay, call=lambda a, m=mode: np.pad(a, 1, mode=m))
            add("tmove", "pad", params={"pad_width": [[1, 2]] * nd if nd else 1, "mode": "constant",
                                        "constant": enc("pad text over 16 b")}, operands=[op], layout=lay,
                call=lambda a: np.pad(a, [(1, 2)] * a.ndim if a.ndim else 1, mode="constant",
                                      constant_values="pad text over 16 b"))
            add("tmove", "pad", params={"pad_width": 1, "mode": "constant", "constant": None}, operands=[op],
                layout=lay, call=lambda a: np.pad(a, 1))

        # Joining: StringDType with StringDType (the first descriptor wins; another NA object is NumPy's common-instance
        # error), with U (promotes to StringDType), with S / numbers (DTypePromotionError).
        a = t_operand(spec, "c2")
        for other_name, other in (("T", t_operand(spec, "f2", seed=4)), ("Tother", t_operand(TDN if spec is TD else TD, "r2", seed=5)),
                                  ("U", operand("<U5", "c2", seed=4)), ("S", operand("|S5", "c2", seed=4, ascii_only=True)),
                                  ("int", plain(np.arange(12).reshape(3, 4)))):
            for axis in (0, 1, None):
                add("tmove", "concatenate", params={"axis": axis}, operands=[a, other], layout=f"c2+{other_name}",
                    call=lambda x, y, ax=axis: np.concatenate([x, y], axis=ax))
            for fn in ("stack", "vstack", "hstack", "dstack", "column_stack"):
                add("tmove", fn, params={"axis": 0}, operands=[a, other], layout=f"c2+{other_name}",
                    call=lambda x, y, f=fn: np.stack([x, y]) if f == "stack" else getattr(np, f)([x, y]))
            add("tmove", "where", params={"cond": [[True, False, True, False], [False] * 4, [True] * 4]},
                operands=[a, other], layout=f"c2+{other_name}",
                call=lambda x, y: np.where(np.array([[True, False, True, False], [False] * 4, [True] * 4]), x, y))
        for xv, yv in (("long scalar text value", None), (None, "q"), ("Ü", "w" * 20)):
            add("tmove", "where_scalar", params={"cond": [True, False, True, False, True, False],
                                                 "x": enc(xv) if xv is not None else None,
                                                 "y": enc(yv) if yv is not None else None},
                operands=[t_operand(spec, "c1")], layout="c1",
                call=lambda a, x=xv, y=yv: np.where(np.array([True, False, True, False, True, False]),
                                                    a if x is None else x, a if y is None else y))
        sa = t_operand(spec, "c2", pool=T_SHORT)
        sb = t_operand(spec, "r2", seed=5, pool=T_SHORT)
        add("tmove", "choose", params={"index": [[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]]}, operands=[sa, sb],
            layout="c2+r2", call=lambda x, y: np.choose(np.array([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]]), [x, y]))
        add("tmove", "select", params={"conds": [[[True, False, True, False], [False] * 4, [True] * 4],
                                                 [[False] * 4, [True] * 4, [True] * 4]], "default": enc("dflt")},
            operands=[t_operand(spec, "c2"), t_operand(spec, "f2", seed=3)], layout="c2+f2",
            call=lambda x, y: np.select([np.array([[True, False, True, False], [False] * 4, [True] * 4]),
                                         np.array([[False] * 4, [True] * 4, [True] * 4])], [x, y], "dflt"))
        add("tmove", "meshgrid", params={"indexing": "xy"}, operands=[t_operand(spec, "c1"), t_operand(spec, "o1", 2)],
            layout="c1+o1", call=lambda x, y: np.meshgrid(x, y))
        add("tmove", "broadcast_arrays", operands=[t_operand(spec, "c2"), t_operand(spec, "one", 2)],
            layout="c2+one", call=lambda x, y: np.broadcast_arrays(x, y))
        add("tmove", "unstack", params={"axis": 1}, operands=[t_operand(spec, "c2")], layout="c2",
            call=lambda a: np.unstack(a, axis=1))
        add("tmove", "extract", params={"condition": [[True, False, True, False], [False] * 4, [True] * 4]},
            operands=[t_operand(spec, "c2", seed=5)], layout="c2",
            call=lambda a: np.extract(np.array([[True, False, True, False], [False] * 4, [True] * 4]), a))

        # In-place writes with array values: StringDType / U / S / numbers into StringDType targets of every layout.
        for lay in ("c2", "f2", "r2"):
            for vname, vop in (("T", t_operand(spec, "c2", seed=7)), ("Trow", t_operand(spec, "c1", seed=9)),
                               ("U", plain(np.array(["p", "qq", "rrrr", "s"]))),
                               ("S", plain(np.array([b"p", b"q" * 20, b"r", b"s"]))),
                               ("int", plain(np.array([[1], [2], [3]])))):
                if vname == "Trow":
                    vop = t_operand(spec, "o1", seed=9)
                add("tmove", "setitem_array", params={"index": [{"ellipsis": True}]},
                    operands=[t_operand(spec, lay), vop], layout=f"{lay}<-{vname}", mutate=0,
                    call=lambda t, v: _setitem(t, Ellipsis, v))
            add("tmove", "setitem_array", params={"index": [{"arr": [2, 0]}]},
                operands=[t_operand(spec, lay), t_plain(np.array(t_values(spec, 8, 3), dtype=spec.build()).reshape(2, 4))],
                layout=lay, mutate=0, call=lambda t, v: _setitem(t, np.array([2, 0]), v))
        add("tmove", "setitem_array", params={"index": [{"mask": [True, False, True, False, False, True]}]},
            operands=[t_operand(spec, "c1"), plain(np.array(["u" * 20, "v"]))], layout="c1", mutate=0,
            call=lambda t, v: _setitem(t, np.array([True, False, True, False, False, True]), v))
        add("tmove", "setitem_array", params={"index": [{"slice": [None, None, None]}]},
            operands=[t_operand(spec, "c1"), t_operand(spec, "r1", seed=2)], layout="c1<-r1", mutate=0,
            call=lambda t, v: _setitem(t, slice(None), v))

        for lay in ("c1", "c2", "f2", "s1", "r2"):
            vals = ["P", "QQ", "RRRR"]
            for mode in ("raise", "wrap", "clip"):
                add("tmove", "put", params={"indices": [0, 4, -1, 9], "values": enc(vals), "mode": mode},
                    operands=[t_operand(spec, lay, pool=T_SHORT)], layout=lay, mutate=0,
                    call=lambda t, v=vals, m=mode: np.put(t, [0, 4, -1, 9], v, mode=m))
            mask = (np.arange(t_operand(spec, lay)[1].size) % 3 == 0).reshape(t_operand(spec, lay)[1].shape)
            add("tmove", "putmask", params={"mask": mask.tolist(), "values": enc(vals)},
                operands=[t_operand(spec, lay, pool=T_SHORT)], layout=lay, mutate=0,
                call=lambda t, m=mask, v=vals: np.putmask(t, m, v))
        for src_name, src in (("T", t_operand(spec, "c2", seed=6)), ("Tf", t_operand(spec, "f2", seed=6)),
                              ("Tother", t_operand(TDN if spec is TD else TD, "c2", seed=6)),
                              ("Ubc", operand("<U5", "bc", seed=6)), ("U", operand("<U1", "f2", seed=6)),
                              ("S", operand("|S5", "r2", seed=6, ascii_only=True)),
                              ("int", plain(np.arange(12).reshape(3, 4))), ("f8", plain(np.linspace(0, 1, 12).reshape(3, 4))),
                              ("bool", plain(np.arange(12).reshape(3, 4) % 2 == 0))):
            for casting in ("no", "same_kind", "unsafe"):
                for where in (None, [[True, False, True, False], [False, True, False, True], [True] * 4]):
                    add("tmove", "copyto", params={"casting": casting, "where": where},
                        operands=[t_operand(spec, "c2"), src], layout=f"c2<-{src_name}", mutate=0,
                        call=lambda d, s, c=casting, w=where: np.copyto(d, s, casting=c,
                                                                       where=True if w is None else np.array(w)))
        # a StringDType source into U / S / numeric targets (the T -> other casts)
        for tgt_name, tgt in (("U8", plain(np.array(["x"] * 12, dtype="<U8").reshape(3, 4))),
                              ("S8", plain(np.array([b"x"] * 12, dtype="|S8").reshape(3, 4)))):
            for casting in ("same_kind", "unsafe"):
                add("tmove", "copyto", params={"casting": casting, "where": None},
                    operands=[tgt, t_operand(spec, "c2", pool=T_SHORT[:9] if tgt_name == "S8" else T_POOL)],
                    layout=f"{tgt_name}<-T", mutate=0,
                    call=lambda d, s, c=casting: np.copyto(d, s, casting=c))
        add("tmove", "put_along_axis", params={"indices": [[1], [0], [3]], "axis": 1},
            operands=[t_operand(spec, "c2"), t_plain(np.array(t_values(spec, 3, 4), dtype=spec.build()).reshape(3, 1))],
            layout="c2", mutate=0, call=lambda t, v: np.put_along_axis(t, np.array([[1], [0], [3]]), v, axis=1))
        for lay in ("c2", "f2"):
            for wrap in (False, True):
                add("tmove", "fill_diagonal", params={"value": enc("diagonal text value"), "wrap": wrap},
                    operands=[t_operand(spec, lay)], layout=lay, mutate=0,
                    call=lambda t, w=wrap: np.fill_diagonal(t, "diagonal text value", wrap=w))
        for shape in ((8,), (3,), (2, 2), (0,)):
            add("tmove", "t_resize_method", params={"shape": list(shape)}, operands=[t_operand(spec, "c1")],
                layout="c1", mutate=0, call=lambda t, sh=shape: t.resize(sh, refcheck=False))


def gen_tcast():
    """Section tcast (P1.6): the casts.cpp loops — StringDType to StringDType (another NA object: NA cells become the
    NA object's str() where the target has none), to U / S (unsized refused, sized cut by code points; S encodes ASCII),
    to bool (empty False, NA by the NA object's kind), to every number (CPython's int() / float() parse, NumPy's packing
    errors, NA refused); and U / S (UTF-8 decoded) / bool / every number (the NumPy scalar's str(); a float64 NaN is NA
    for a NaN-like NA object) to StringDType — over every layout, order, copy flag and casting rule.
    """
    targets = [senc(TD), senc(TDN), senc(TDNAN), senc(TDS), senc(TDC), "<U1", "<U3", "<U8", "U", "|S3", "|S8", "S",
               "?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8"]
    for spec in T_TEXT_SPECS:
        for lay in CORE_LAYOUTS + ["c3", "r1", "rs", "d5", "e1"]:
            for t in targets:
                ascii_only = isinstance(t, str) and t.lstrip("|<").startswith("S")
                pool = T_SHORT[:9] if ascii_only else T_POOL
                add("tcast", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
                    operands=[t_operand(spec, lay, seed=3, pool=pool)], layout=lay,
                    call=lambda a, t=t: a.astype(_tbuild(t)))
    # orders and copy=False
    for spec in (TD, TDN):
        for lay in ("c2", "f2", "t3", "r2", "bc"):
            for order in ("C", "F", "A", "K"):
                for t in (senc(spec), "<U8", senc(TDN if spec is TD else TD)):
                    add("tcast", "astype", params={"dtype": t, "casting": "unsafe", "order": order, "copy": True},
                        operands=[t_operand(spec, lay)], layout=lay,
                        call=lambda a, t=t, o=order: a.astype(_tbuild(t), order=o))
            for t in (senc(spec), senc(TDN if spec is TD else TD), "<U8"):
                add("tcast", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": False},
                    operands=[t_operand(spec, lay)], layout=lay,
                    call=lambda a, t=t: a.astype(_tbuild(t), copy=False))
        for t in (senc(spec), senc(TDN if spec is TD else TD), "<U8", "U", "|S8", "?", "i8", "f8"):
            for casting in ("no", "equiv", "safe", "same_kind", "unsafe"):
                add("tcast", "astype", params={"dtype": t, "casting": casting, "order": "K", "copy": True},
                    operands=[t_operand(spec, "c1", pool=T_SHORT[:9] if t == "|S8" else T_POOL)], layout="c1",
                    call=lambda a, t=t, c=casting: a.astype(_tbuild(t), casting=c))
    # Text that is not ASCII into S; NA into every number and bool.
    for spec in T_TEXT_SPECS:
        for vals, t in [(["ab", "cé"], "|S3"), (["x", "\U0001F600"], "|S8"), (["ab", "cé" * 10], "<U2"),
                        (["\U0001F600" * 4, "a"], "<U3")]:
            add("tcast", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
                operands=[plain(np.array(vals, dtype=spec.build()))], layout="c1", call=lambda a, t=t: a.astype(t))
    # Text <-> numbers: CPython's int() / float() grammar, Unicode digits and spaces, NumPy's packing errors.
    int_texts = ["0", "-0", "+5", " 12 ", "\t7\n", "1_000", "00012", "١٢٣", "１２", "-1", "", "x", "1.5", "300",
                 "-129", "255", "256", "65536", "2147483648", "9223372036854775807", "9223372036854775808",
                 "18446744073709551616", "1e3", "nan"]
    float_texts = ["1.5", "-0.0", "inf", "-Infinity", "nan", "-nan", "NaN", "+inf", "1e308", "1e309", "1e-400", " 2.5 ",
                   "1_0.5", "0x10", "", "abc", "1e", "６.５", "65504", "65520", "3.4028235e38", "3.5e38"]
    for spec in (TD, TDN):
        for texts, targets_ in ((int_texts, ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8"]),
                                (float_texts, ["f2", "f4", "f8"])):
            for t in targets_:
                for tx in texts:
                    add("tcast", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
                        operands=[plain(np.array([tx], dtype=spec.build()))], layout="c1",
                        call=lambda a, t=t: a.astype(t))
                    _drop_platform_int_text()
        for t in ("?", "i8", "u1", "f8", "f4", "c16"):
            # StringDType -> complex succeeds wrongly in NumPy 2.4.2 (the real part is written into the imaginary part too:
            # ['1', '2'] -> [1+1j, 2+2j]); only its NA refusal is NumPy's answer, the values are unit-tested.
            if t == "c16" and spec is not TDN:
                continue
            add("tcast", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
                operands=[plain(np.array(["1", None] if spec is TDN else ["1", "2"], dtype=spec.build()))],
                layout="c1", call=lambda a, t=t: a.astype(t))
    for spec in (TDNAN, TDS):
        for t in ("?", "i8", "f8", "c16"):
            add("tcast", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
                operands=[plain(np.array(["1", spec.na], dtype=spec.build()))], layout="c1",
                call=lambda a, t=t: a.astype(t))
    for spec in T_TEXT_SPECS:
        add("tcast", "astype", params={"dtype": "?", "casting": "unsafe", "order": "K", "copy": True},
            operands=[plain(np.array(["", "a", "0", "False", " ", "\x00"], dtype=spec.build()))], layout="c1",
            call=lambda a: a.astype("?"))

    # Into StringDType: U (any code point), S (UTF-8 decoded; invalid bytes refused), bool, every number.
    numbers = {
        "?": [True, False], "i1": [-128, 0, 127], "u1": [0, 255], "i2": [-32768, 32767], "u2": [65535],
        "i4": [-2147483648, 7], "u4": [4294967295], "i8": [-9223372036854775808, 9223372036854775807],
        "u8": [18446744073709551615, 0],
        "f2": [0.5, -0.0, float("inf"), float("nan"), 65504.0, 1e-7, 0.1],
        "f4": [0.1, 1e16, 1e-5, -0.0, float("nan"), float("-inf"), 3.4028235e38, 1.0, 123456789.0],
        "f8": [0.1, 1e16, 1e-5, -0.0, float("nan"), float("inf"), 1 / 3, 1e22, 5e-324, 2.5, 100.0],
        "c16": [1 + 2j, -0.0 - 1j, complex(float("nan"), 1), complex(1, float("inf")), 0j],
    }
    for dt, vals in numbers.items():
        arr = np.array(vals, dtype=dt)
        for spec in T_TEXT_SPECS + [TDC]:
            add("tcast", "astype", params={"dtype": senc(spec), "casting": "unsafe", "order": "K", "copy": True},
                operands=[plain(arr)], layout="c1", call=lambda a, s=spec: a.astype(s.build()))
    for spec in (TD, TDN):
        for src in ("<U1", "<U3", "<U5", "|S1", "|S3", "|S5"):
            for lay in CORE_LAYOUTS + ["c3", "r1", "e1"]:
                # An S source holds valid UTF-8 only: NumPy 2.4.2's bytes -> StringDType cast copies the bytes WITHOUT
                # validating them (casts.cpp bytes_to_string), so invalid UTF-8 is stored and raises on every later read;
                # NumSharp raises the same UnicodeDecodeError at the cast (unit-tested divergence).
                add("tcast", "astype", params={"dtype": senc(spec), "casting": "unsafe", "order": "K", "copy": True},
                    operands=[operand(src, lay, seed=3, ascii_only=src[1] == "S")], layout=lay,
                    call=lambda a, s=spec: a.astype(s.build()))
        for vals in ([b"\xc3\xa9", b"\xe2\x82\xac"], ["\ud800"], ["a", "\U0010FFFF"]):
            add("tcast", "astype", params={"dtype": senc(spec), "casting": "unsafe", "order": "K", "copy": True},
                operands=[plain(np.array(vals))], layout="c1", call=lambda a, s=spec: a.astype(s.build()))
        for casting in ("no", "equiv", "safe", "same_kind", "unsafe"):
            for src in (plain(np.array(["ab", "c"])), plain(np.array([b"ab", b"c"])), plain(np.array([1, 2])),
                        plain(np.array([True])), plain(np.array([1.5]))):
                add("tcast", "astype", params={"dtype": senc(spec), "casting": casting, "order": "K", "copy": True},
                    operands=[src], layout="c1", call=lambda a, s=spec, c=casting: a.astype(s.build(), casting=c))
    # np.array of a StringDType operand with a U / S / numeric dtype, and np.array of U with dtype=StringDType.
    for spec in (TD, TDN):
        for t in ("<U8", "|S8", "f8"):
            vals = ["1.5", "2", "-3"] if t == "f8" else ["ab", "c", "def"]
            add("tcast", "asarray", params={"dtype": t}, operands=[plain(np.array(vals, dtype=spec.build()))],
                layout="c1", call=lambda a, t=t: np.asarray(a, dtype=t))


def _tbuild(t):
    """A target dtype of section tcast: a dtype string, or an encoded StringDType request.

    :param t: the request.
    :returns: the NumPy descriptor.
    """
    return np.dtype(t) if isinstance(t, str) else TSpec(**_tspec_kw(t)).build()


class _TProps:
    """Marks a descriptor whose expectation is its full attribute set (``tprops``) rather than ``{"kind": "dtype"}``."""

    def __init__(self, d):
        """Wraps the descriptor.

        :param d: the NumPy descriptor the call returned.
        """
        self.d = d


GENERATORS = [gen_dtypes, gen_creation, gen_astype, gen_views, gen_movement, gen_mutation, gen_access, gen_printing,
              gen_refusals, gen_string_requests, gen_text_precedence, gen_shape_decided, gen_validation_order, gen_surface,
              gen_strma, gen_striter, gen_strcomp, gen_strarith, gen_strconv, gen_tdtype, gen_tstore, gen_tcreate,
              gen_tmove, gen_tcast]


def main():
    """Writes the whole corpus, or with ``--sections a,b`` regenerates only those sections' generators (``gen_<name>``)
    and splices their cases into the existing corpus in place of the old ones — a development shortcut (the whole corpus
    takes minutes); the committed corpus is always a full run."""
    only = None
    if len(sys.argv) > 2 and sys.argv[1] == "--sections":
        only = set(sys.argv[2].split(","))
    for gen in GENERATORS:
        if only is None or gen.__name__[len("gen_"):] in only:
            gen()
    lines = [json.dumps(c, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n"
             for c in CASES]
    if only is not None:
        # Keep the other sections' cases as they are; the regenerated sections replace their old cases.
        regenerated = {c["id"].split("/")[0] for c in CASES} | only
        kept = [ln for ln in open(OUT, "rb") if json.loads(ln)["id"].split("/")[0] not in regenerated]
        lines = kept + lines
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        # allow_nan=False above: a NaN / infinity in the params would be written as invalid JSON (Infinity); such a value
        # travels as a string instead (_json_floats).
        f.writelines(lines)
    by_section = {}
    for c in CASES:
        sec = c["id"].split("/")[0]
        by_section[sec] = by_section.get(sec, 0) + 1
    errs = sum(1 for c in CASES if "error" in c)
    p2 = sum(1 for c in CASES if c["phase"] == 2)
    print(f"wrote {len(CASES)} cases ({errs} NumPy errors, {p2} phase-2) to {OUT}")
    print("  " + ", ".join(f"{k} {v}" for k, v in by_section.items()))


if __name__ == "__main__":
    sys.exit(main())
