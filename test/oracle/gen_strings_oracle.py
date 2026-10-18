#!/usr/bin/env python3
"""NumPy 2.4.2 oracle for NumSharp's string dtypes: the fixed-width 'S' (bytes) and 'U' (str), and NEP 55 StringDType ('T').

Writes ``test/NumSharp.Tests.Oracle/Strings/corpus/strings_<section>.jsonl``, one file per section (the id's first
component), one case per line, replayed byte for byte by ``StringOracleTests`` (no Python at test time). One file per
section keeps every file far below GitHub's 100 MiB limit (the single ``strings.jsonl`` had reached 96.6 MiB); a full
run rewrites every section file and removes the files of sections that no longer exist.

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
import codecs
import json
import operator
import sys
import warnings
from pathlib import Path

import numpy as np

assert np.__version__ == "2.4.2", f"the string oracle is pinned to numpy 2.4.2, found {np.__version__}"

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE.parent / "NumSharp.Tests.Oracle" / "Strings" / "corpus"

# GitHub refuses a file over 100 MiB and warns over 50 MiB: a section file past the warning size is an error here, and the
# section is then split by the generator (a second id prefix), not committed.
MAX_SECTION_BYTES = 50 * 1024 * 1024


def section_path(section):
    """The corpus file of one section.

    :param section: the section name (a case id's first component).
    :returns: ``CORPUS_DIR / "strings_<section>.jsonl"``.
    """
    return CORPUS_DIR / f"strings_{section}.jsonl"

CASES = []
_IDS = set()
# The number of recorded cases per id prefix "section/op/": add() numbers the next case from it, and _pop_case() gives the
# number back, so the ids are exactly what a scan of CASES gives (the generator used to count with that scan per add(),
# quadratic in the corpus size).
_OP_COUNT = {}


def _pop_case():
    """Drops the case just recorded and frees its id and its number (the next case of its op takes the number again).

    :returns: the dropped case.
    """
    c = CASES.pop()
    _IDS.discard(c["id"])
    prefix = c["id"].rsplit("/", 1)[0] + "/"
    _OP_COUNT[prefix] -= 1
    return c


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
    if isinstance(v, dict):
        # A dict (translate's table): its items in insertion order, each key and value encoded (the replay builds a C#
        # dictionary with the keys' own types: a float key 97.0 stays a double, True a bool, 'a' a string).
        return {"dict": [[enc(k), enc(x)] for k, x in v.items()]}
    raise TypeError(f"cannot encode {type(v)}")


# LinAlgError (a ValueError subclass in NumPy and NumSharp alike) is recorded under its own name, so the replay checks
# that NumSharp raises the linear-algebra class itself, not just any ValueError.
KNOWN_ERRORS = ["UnicodeEncodeError", "UnicodeDecodeError", "UnicodeError", "DTypePromotionError", "AxisError", "LinAlgError",
                "ValueError", "TypeError", "IndexError", "OverflowError", "SystemError", "KeyError", "ZeroDivisionError",
                "NotImplementedError", "RuntimeError", "MemoryError", "OSError", "LookupError"]


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
    # The view must address the base's own buffer: a pair built from two separate arrays gives an offset between two
    # unrelated allocations, and the replay would write outside the rebuilt buffer (strsearch's first strided out= did).
    assert view.size == 0 or np.may_share_memory(base, view), "a view must address its base's buffer"
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
    """An in-place target or out= operand whose OWNDATA the replay reproduces: a 1-D string array (``S``, ``U`` or a
    ``StringDType``), or an empty 1-D numeric one, is its own base (``own``: the replay rebuilds those as the base); a
    non-empty numeric operand stays a view (``plain``: the general corpus' reconstruction builds a view over the
    buffer)."""
    arr = np.asarray(arr)
    if arr.ndim == 1 and (arr.dtype.kind in "SUT" or arr.size == 0):
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
    recorded: its dtype carries ``NPY_NEEDS_INIT``, so NumPy's ``np.empty`` zero-fills it. A ``StringDType`` data array
    (section tma) is recorded by its VALUES (``tvals``: NA cells marked) and its descriptor request (``t``), never by its
    cells, which hold allocator offsets and heap pointers; its ``np.empty`` is zero-filled too (every element ``''``)."""
    data = np.asarray(r.data)
    m = r._mask
    if data.dtype.kind == "T":
        return {"kind": "masked", "dtype": data.dtype.str, "t": tspec_of(data.dtype),
                "shape": [int(x) for x in data.shape], "values": tvals(data),
                "mask": None if m is np.ma.nomask else np.ascontiguousarray(np.asarray(m, dtype=bool)).tobytes().hex(),
                "fill": _enc_fill(r.fill_value), "hard": bool(r._hardmask)}
    buffer = None if uninit and data.dtype.kind == "S" else np.ascontiguousarray(data).tobytes().hex()
    return {"kind": "masked", "dtype": data.dtype.str, "shape": [int(x) for x in data.shape],
            "buffer": buffer,
            "mask": None if m is np.ma.nomask else np.ascontiguousarray(np.asarray(m, dtype=bool)).tobytes().hex(),
            "fill": _enc_fill(r.fill_value), "hard": bool(r._hardmask)}


class _UninitData:
    """A masked result whose data NumPy left uninitialized (see ``masked_expect``); ``expect_of`` unwraps it."""

    def __init__(self, r):
        self.r = r


class _AsciiText(str):
    """A text result that is ASCII by construction (a ``titer`` trace: every value in it is hex), stored as is:
    ``{"kind": "text", "a": text}`` — the UTF-32 hex of ``{"kind": "text", "u": ...}`` would be eight times its size."""


def expect_of(r, operands):
    """The expectation for any result NumPy returned."""
    if isinstance(r, _AsciiText):
        assert str(r).isascii(), "an _AsciiText result holds only ASCII"
        return {"kind": "text", "a": str(r)}
    if isinstance(r, _UninitData):
        return masked_expect(r.r, uninit=True) if isinstance(r.r, np.ma.MaskedArray) and r.r is not np.ma.masked \
            else expect_of(r.r, operands)
    if r is np.ma.masked:
        return {"kind": "masked_constant"}
    if r is NotImplemented:
        # numpy.char's comparisons (compare_chararrays) answer NotImplemented for an S / U pair; NumSharp returns null.
        return {"kind": "notimplemented"}
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
    if isinstance(r, tuple) and r and all(isinstance(x, (str, bytes, np.generic, np.ndarray)) for x in r):
        # A ufunc of several outputs over 0-d operands (np.strings.partition of a 0-d StringDType array): NumPy's NumPy
        # scalars / Python strs, each by its own kind.
        return {"kind": "items", "items": [expect_of(x, operands) for x in r]}
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
    prefix = f"{section}/{op}/"
    n = _OP_COUNT.get(prefix, 0) + 1
    _OP_COUNT[prefix] = n
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
        data = np.array(a.data, dtype=a.spec) if a.spec is not None else np.array(a.data)
        return data.dtype.kind in "SUT"
    if isinstance(a, np.ndarray):
        return a.dtype.kind in "SUT"
    return isinstance(a, (str, bytes))


def add_ma(op, *, args, params=None, call, phase="compute", layout="", section="strma", trace_t=False):
    """Record one numpy.ma call over masked arguments. Each ``MArg`` travels as a data operand plus (unless nomask or a
    bool) a boolean mask operand; ``params["mops"]`` says how to rebuild every argument (``{"d": data, "m": mask}``, a
    plain array ``{"a": i}`` or a value ``{"v": ...}``) and the replay builds the same masked arrays.

    :param op: the op key (``ma0`` / ``ma1`` / ``ma2``).
    :param args: the arguments: ``MArg``s, plain arrays or Python values.
    :param params: the case parameters (``name`` selects the function).
    :param call: the NumPy call over the rebuilt arguments.
    :param phase: the case's phase (``"compute"``: decided by NumPy's outcome, see ``add``).
    :param layout: the variant label.
    :param section: the corpus section (``strma`` for the fixed-width strings, ``tma`` for ``StringDType``).
    :param trace_t: classify a ``"compute"`` case by whether NumPy COMPUTED on ``StringDType`` text (section tma): the
        call is run a second time over fresh copies of the arguments whose ``StringDType`` arrays are ``_TTrace`` views,
        and the case is phase 1 (NumSharp must answer it now) unless that run computed on text (phase 2). Without it a
        ``"compute"`` case is phase 2 whenever NumPy returns a result.
    """

    def build():
        """The operands and the rebuild instructions, from fresh copies of every argument (a call may mutate them)."""
        operands, mops = [], []
        for a in args:
            if isinstance(a, MArg):
                # ``is not None``: a dtype OBJECT (a StringDType instance) is falsy, len() of an unstructured dtype being 0.
                data = np.array(a.data, dtype=a.spec) if a.spec is not None else np.array(a.data)
                entry = {"d": len(operands), "m": -1}
                # ``target``: an operand whose flags the replay reproduces (a 1-D string operand is its own base, a
                # non-empty numeric one a view), since a function may return the very array it was given (getmask,
                # filled).
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
        return operands, mops

    operands, mops = build()
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

    add(section, op, params=p, operands=operands, call=wrapped, phase=phase, layout=layout)
    if trace_t and phase == "compute":
        _t_trace_phase(CASES[-1], build, wrapped)


def _strma_unary_table():
    """The one-argument calls of sections strma and tma: every numpy.ma function applied to one masked array (the
    masked ufuncs, the reductions and statistics, the mask / data / conversion / movement functions) plus the
    method and attribute forms (``flatten``, ``fill_value``, ``repr`` ...), keyed by the name the replay switches on.

    :returns: a dict of name -> ``f(x)``; ``empty_like`` / ``masked_all_like`` wrap their result in ``_UninitData``
        (NumPy leaves an ``S`` buffer uninitialized).
    """
    ma = np.ma
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
    return unary


def _strma_binary_table():
    """The two-argument calls of sections strma and tma: the binary masked ufuncs, products, set routines, mask
    combination, the masked_* family against a value, compress / filled and ``average`` with weights.

    :returns: a dict of name -> ``f(x, y)``.
    """
    ma = np.ma
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
    return binary


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
    unary = _strma_unary_table()
    for name, fn in unary.items():
        for key, arg in unary_variants:
            add_ma("ma1", params={"name": name}, args=[arg], layout=key, call=lambda x, f=fn: f(x))
    for key, arg in (("MUh", MUh), ("MU2", MU2)):
        add_ma("ma1", params={"name": "putmask_text"}, args=[arg], layout=key, call=lambda x: unary["putmask_text"](x))

    # ---- two arguments: the binary ufuncs, products, set routines, mask combination ----
    F = np.array([1.5, 2.0, 3.0])
    Ue1 = np.zeros(0, "<U1")
    binary = _strma_binary_table()
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
# tma: numpy.ma over NEP 55 StringDType (strings phase 1, P1.7d)
# ----------------------------------------------------------------------------------------------------------------------
#
# The phase of a case is decided by whether NumPy COMPUTED on StringDType text, not by whether it returned: section tma's
# masked arrays are rebuilt a second time over ``_TTrace`` views (``_t_trace_phase``), which record every ufunc call that
# reaches a StringDType loop (an error the same call raises over benign texts is a setup error: dtype- or shape-decided,
# no computation) and every text-reading method (sort, argsort, partition, searchsorted, nonzero, argmax / argmin, truth).
# A case NumPy answers without computing on text is phase 1 — movement, printing, fills, masks, casts and the refusals
# NumPy raises before it reads a string — and NumSharp must answer it now; a case that computed is phase 2 (a deferral
# while NumSharp throws StringComputationNotSupportedException).

_TMA_TRACE_MISMATCH = []

# The calls NumPy computes on StringDType text through a path the tracer cannot see -- an unbound ndarray.sort on the
# masked array itself (MaskedArray.sort without a mask, so ma.sort / ma.unique and the set routines built on them), a
# plain np.asarray copy before the ufunc (np.outer), the truth of each text inside a C loop (nonzero, np.where's
# condition, ma.compress's condition): a result is the computational phase (phase 2), an error the outcome-decided phase
# (1, as in strma).
_TMA_UNTRACEABLE = {"sort", "sort_end_false", "unique", "nonzero", "compress_self", "where_textcond", "outer",
                    "outerproduct", "isin", "in1d", "intersect1d", "union1d", "setdiff1d", "setxor1d"}


class _TTrace(np.ndarray):
    """A ``StringDType`` array view that records whether NumPy COMPUTED on its text (section tma's phase rule).

    ``computed`` is a class flag: ``_t_trace_phase`` clears it, runs the call over ``_TTrace`` views of the arguments,
    and reads it. A ufunc call (any method: ``__call__``, ``reduce``, ``accumulate``, ...) with a ``StringDType`` input
    that runs a loop sets it; a ufunc call that raises sets it only when the same call over benign texts (every element
    ``'x'``, no NA) does not raise the same error (``Cannot add null that is not a nan-like value`` is decided by the
    values; ``did not contain a loop`` / ``not reorderable`` / ``zero-size array to reduction`` by the dtypes and shapes).
    The text-reading methods set it after they return.
    """

    computed = False

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Runs the ufunc over plain arrays and records a computation on text.

        :param ufunc: the ufunc.
        :param method: the ufunc method (``__call__``, ``reduce``, ``accumulate``, ``reduceat``, ``outer``, ``at``).
        :param inputs: the inputs (``_TTrace`` views are unwrapped).
        :param kwargs: the keywords (an ``out`` tuple's ``_TTrace`` views are unwrapped).
        :returns: the ufunc's result (plain arrays).
        :raises Exception: whatever the ufunc raises.
        """
        plain = tuple(np.asarray(x) if isinstance(x, _TTrace) else x for x in inputs)
        if kwargs.get("out") is not None:
            kwargs["out"] = tuple(np.asarray(o) if isinstance(o, _TTrace) else o for o in kwargs["out"])
        text = any(isinstance(x, np.ndarray) and x.dtype.kind == "T" for x in plain)
        try:
            r = getattr(ufunc, method)(*plain, **kwargs)
        except Exception as e:  # noqa: BLE001 -- re-raised; only classified here
            if text and not _t_setup_error(ufunc, method, plain, kwargs, e):
                _TTrace.computed = True
            raise
        if text:
            _TTrace.computed = True
        return r

    @staticmethod
    def _mark(x):
        """Records a text-reading method of a ``StringDType`` array.

        :param x: the array the method ran on.
        """
        if x.dtype.kind == "T":
            _TTrace.computed = True

    def sort(self, *a, **k):
        """``ndarray.sort``, recorded as a computation once it returns."""
        np.ndarray.sort(self, *a, **k)
        _TTrace._mark(self)

    def argsort(self, *a, **k):
        """``ndarray.argsort``, recorded as a computation once it returns."""
        r = np.ndarray.argsort(self, *a, **k)
        _TTrace._mark(self)
        return r

    def partition(self, *a, **k):
        """``ndarray.partition``, recorded as a computation once it returns."""
        np.ndarray.partition(self, *a, **k)
        _TTrace._mark(self)

    def argpartition(self, *a, **k):
        """``ndarray.argpartition``, recorded as a computation once it returns."""
        r = np.ndarray.argpartition(self, *a, **k)
        _TTrace._mark(self)
        return r

    def searchsorted(self, *a, **k):
        """``ndarray.searchsorted``, recorded as a computation once it returns."""
        r = np.ndarray.searchsorted(self, *a, **k)
        _TTrace._mark(self)
        return r

    def nonzero(self):
        """``ndarray.nonzero`` (the truth of each text), recorded as a computation once it returns."""
        r = np.ndarray.nonzero(self)
        _TTrace._mark(self)
        return r

    def argmax(self, *a, **k):
        """``ndarray.argmax``, recorded as a computation once it returns."""
        r = np.ndarray.argmax(self, *a, **k)
        _TTrace._mark(self)
        return r

    def argmin(self, *a, **k):
        """``ndarray.argmin``, recorded as a computation once it returns."""
        r = np.ndarray.argmin(self, *a, **k)
        _TTrace._mark(self)
        return r

    def __bool__(self):
        """The truth of a one-element array (the truth of its text), recorded as a computation once it returns."""
        r = np.ndarray.__bool__(self)
        _TTrace._mark(self)
        return r


def _t_benign(x):
    """A benign stand-in for a ufunc input: a ``StringDType`` array of the same descriptor and shape whose every element is
    ``'x'`` (no NA, not empty); any other input as is.

    :param x: the input.
    :returns: the stand-in.
    """
    if isinstance(x, np.ndarray) and x.dtype.kind == "T":
        return np.full(x.shape, "x", dtype=x.dtype)
    return x


def _t_setup_error(ufunc, method, inputs, kwargs, e):
    """Whether a ufunc error over ``StringDType`` inputs is decided by the dtypes and shapes alone (no computation on the
    text): the same call over benign texts raises the same error.

    :param ufunc: the ufunc.
    :param method: the ufunc method.
    :param inputs: the plain inputs.
    :param kwargs: the keywords (a ``StringDType`` ``out`` is replaced by a fresh one).
    :param e: the error the real call raised.
    :returns: True for a setup error.
    """
    kw = dict(kwargs)
    if kw.get("out") is not None:
        kw["out"] = tuple(np.empty_like(o) if isinstance(o, np.ndarray) and o.dtype.kind == "T" else o
                          for o in kw["out"])
    try:
        getattr(ufunc, method)(*(_t_benign(x) for x in inputs), **kw)
    except Exception as e2:  # noqa: BLE001 -- compared, not raised
        return type(e2) is type(e) and str(e2) == str(e)
    return False


def _ma_traced_ufunc(self, ufunc, method, *inputs, **kwargs):
    """The ``__array_ufunc__`` ``_t_trace_phase`` installs on ``MaskedArray`` for the traced run: a ufunc that receives a
    masked array ITSELF (``ndarray.mean`` / ``var`` of a nomask masked array call ``umr_sum`` on the masked array, not on
    its ``_TTrace`` data) is recorded like ``_TTrace.__array_ufunc__`` records it, and runs exactly as without the hook
    (the hook removes itself for the call, so masked arrays keep their ``__array_wrap__`` semantics).

    :param self: the masked array the ufunc dispatched on.
    :param ufunc: the ufunc.
    :param method: the ufunc method.
    :param inputs: the inputs, unchanged.
    :param kwargs: the keywords, unchanged.
    :returns: the ufunc's result.
    :raises Exception: whatever the ufunc raises.
    """
    text = any(isinstance(x, np.ndarray) and x.dtype.kind == "T" for x in inputs)
    del np.ma.MaskedArray.__array_ufunc__
    try:
        r = getattr(ufunc, method)(*inputs, **kwargs)
    except Exception as e:  # noqa: BLE001 -- re-raised; only classified here
        plain = tuple(np.asarray(x) if isinstance(x, np.ndarray) else x for x in inputs)
        if text and not _t_setup_error(ufunc, method, plain, kwargs, e):
            _TTrace.computed = True
        raise
    finally:
        np.ma.MaskedArray.__array_ufunc__ = _ma_traced_ufunc
    if text:
        _TTrace.computed = True
    return r


def _t_trace_phase(case, build, call):
    """Sets a section-tma case's phase by whether NumPy computed on ``StringDType`` text: the call runs again over fresh
    arguments (``build``) whose ``StringDType`` operands are ``_TTrace`` views, with ``_ma_traced_ufunc`` installed on
    ``MaskedArray`` for the run; phase 2 when it computed, else 1. A case the outcome already made phase 2 for an error (a
    parse error, ``_is_parse_error``) keeps it. When the traced run's outcome differs from the recorded one (the tracing
    changed what NumPy does) the outcome-decided phase is kept and the case is listed in ``_TMA_TRACE_MISMATCH`` (main
    prints them).

    :param case: the case just recorded.
    :param build: returns fresh (operands, mops) for the call.
    :param call: the wrapped call over the operand views.
    """
    want = case.get("error")
    if want is not None and case["phase"] == 2:
        return
    operands, _ = build()
    views = [v.view(_TTrace) if v.dtype.kind == "T" else v for _, v in operands]
    _TTrace.computed = False
    assert "__array_ufunc__" not in np.ma.MaskedArray.__dict__, "MaskedArray must inherit ndarray's __array_ufunc__"
    np.ma.MaskedArray.__array_ufunc__ = _ma_traced_ufunc
    try:
        call(*views)
        got = None
    except Exception as e:  # noqa: BLE001 -- a recorded outcome
        got = err_info(e)
    finally:
        del np.ma.MaskedArray.__array_ufunc__
    if (got is None) != (want is None) or (got is not None and got != want):
        _TMA_TRACE_MISMATCH.append(case["id"])
        return
    case["phase"] = 2 if _TTrace.computed else 1


def _tma_unary_extra():
    """The one-argument calls section tma adds to strma's table: element access and stores (an NA store, a masked
    store), ``tolist``, the ``data`` / ``mask`` attributes, casts to and from ``StringDType`` (another NA object, ``U`` /
    ``S`` / numbers), filled with an explicit text, ``where`` / ``concatenate`` beside ``U`` text.

    :returns: a dict of name -> ``f(x)``.
    """
    ma = np.ma

    def store(x, k, v):
        x[k] = v
        return x

    return {
        "getitem_0": lambda x: x[0],
        "getitem_1": lambda x: x[1],
        "getitem_last": lambda x: x[-1],
        "getitem_slice": lambda x: x[1:],
        "getitem_rev": lambda x: x[::-1],
        "getitem_mask": lambda x: x[np.ones(x.shape, bool)],
        "getitem_fancy": lambda x: x[np.array([0, 0])],
        "setitem_text": lambda x: store(x, 1, "zz"),
        "setitem_long": lambda x: store(x, 0, "w" * 40),
        "setitem_masked": lambda x: store(x, 1, ma.masked),
        "setitem_none": lambda x: store(x, 1, None),
        "setitem_slice": lambda x: store(x, slice(None), "y"),
        "setitem_num": lambda x: store(x, 1, 5),
        "tolist": lambda x: x.tolist(),
        "data": lambda x: x.data,
        "mask_attr": lambda x: x.mask,
        "astype_T": lambda x: x.astype(np.dtypes.StringDType()),
        "astype_TN": lambda x: x.astype(np.dtypes.StringDType(na_object=None)),
        "astype_U3": lambda x: x.astype("U3"),
        "astype_S3": lambda x: x.astype("S3"),
        "astype_i8": lambda x: x.astype("i8"),
        "astype_b": lambda x: x.astype(bool),
        "filled_long": lambda x: x.filled("v" * 30),
        "filled_none": lambda x: x.filled(None),
        "concat_u": lambda x: ma.concatenate([x, ma.array(np.array(["q"]), mask=[True])]) if x.ndim == 1
        else ma.concatenate([x, x]),
        "where_text": lambda x: ma.where(np.ones(x.shape, bool), x, "zz"),
        "copy_method": lambda x: x.copy(),
        "view_method": lambda x: x.view(),
        "reshape_method": lambda x: x.reshape(-1),
        "swapaxes": lambda x: ma.swapaxes(x, 0, -1) if x.ndim else x,
    }


def gen_tma():
    """Section tma (strings phase 1, P1.7d): numpy.ma over NEP 55 ``StringDType``. The strma tables over ``T`` arguments —
    every numpy.ma function applied to one masked array (14 variants: 1-D / 2-D / empty / 0-d / all masked / a fill value /
    long texts / the NA descriptors ``na_object=None`` / ``nan`` / ``'NA'`` with an NA element / ``coerce=False``), the
    binary functions over ``T`` / ``U`` / ``S`` / numeric / value pairs, and the constructors (``T`` masks, fill values,
    ``masked_all``, the default fill value ``'N/A'`` of every ``T`` descriptor, mask reconciliation) — plus the element
    access, stores, casts and attributes of ``_tma_unary_extra``. The phase follows ``_t_trace_phase``: phase 1 unless
    NumPy computed on the text.
    """
    ma = np.ma
    T, TN, TNAN, TS, TC = (s.build() for s in (TD, TDN, TDNAN, TDS, TDC))
    MT = MArg(["ab", "c", "de"], [1, 0, 0], spec=T)
    MTn = MArg(["ab", "c", "de"], spec=T)
    MT2 = MArg([["ab", "c"], ["de", "f"]], [[1, 0], [0, 0]], spec=T)
    MTe = MArg(np.zeros(0, T), np.zeros(0, bool))
    MTe2 = MArg(np.zeros((0, 3), T))
    MT0 = MArg(np.array("ab", T))
    MT0m = MArg(np.array("ab", T), True)
    MTall = MArg(["ab", "c", "de"], True, spec=T)
    MTf = MArg(["ab", "c", "de"], [1, 0, 0], spec=T, fill="zz")
    MTh = MArg(["ab", "c", "de"], [1, 0, 0], spec=T, hard=True)
    MTl = MArg(["m" * 16, "q" * 300, "é" * 8, ""], [0, 1, 0, 0], spec=T)
    MTN = MArg(["ab", None, "de"], [1, 0, 0], spec=TN)
    MTnan = MArg(["ab", float("nan"), "de"], [0, 0, 1], spec=TNAN)
    MTs = MArg(["ab", "NA", "de"], [1, 0, 0], spec=TS)
    MTc = MArg(["ab", "c", "de"], [1, 0, 0], spec=TC)
    unary_variants = [("MT", MT), ("MTn", MTn), ("MT2", MT2), ("MTe", MTe), ("MTe2", MTe2), ("MT0", MT0),
                      ("MT0m", MT0m), ("MTall", MTall), ("MTf", MTf), ("MTl", MTl), ("MTN", MTN), ("MTnan", MTnan),
                      ("MTs", MTs), ("MTc", MTc)]

    # ---- one masked StringDType argument ----
    unary = _strma_unary_table()
    unary.update(_tma_unary_extra())
    for name, fn in unary.items():
        for key, arg in unary_variants:
            add_ma("ma1", params={"name": name}, args=[arg], layout=key, call=lambda x, f=fn: f(x), section="tma",
                   trace_t=name not in _TMA_UNTRACEABLE)
    for key, arg in (("MTh", MTh), ("MT2", MT2)):
        for name in ("putmask_text", "setitem_text", "setitem_masked", "setitem_slice"):
            add_ma("ma1", params={"name": name}, args=[arg], layout=key, call=lambda x, f=unary[name]: f(x),
                   section="tma", trace_t=True)

    # ---- two arguments ----
    binary = _strma_binary_table()
    F = np.array([1.5, 2.0, 3.0])
    Te1 = np.zeros(0, T)
    MU = MArg(["ab", "c", "de"], [1, 0, 0])
    MS = MArg([b"ab", b"c", b"de"], [1, 0, 0])
    MF = MArg([1.5, 2.0, 3.0], [0, 1, 0])
    pairs = [("MT,MT", MT, MT), ("MT,MTn", MT, MTn), ("MT,F", MT, F), ("F,MT", F, MT), ("MF,MT", MF, MT),
             ("MTe,MTe", MTe, MTe), ("MTe,Te", MTe, Te1), ("MT,'ab'", MT, "ab"), ("MT,1.5", MT, 1.5), ("MT,1", MT, 1),
             ("MT,True", MT, True), ("MT,MU", MT, MU), ("MU,MT", MU, MT), ("MT,MS", MT, MS), ("MS,MT", MS, MT),
             ("MT2,MT2", MT2, MT2), ("MTN,MTN", MTN, MTN), ("MT,MTN", MT, MTN), ("MTnan,MTnan", MTnan, MTnan),
             ("MTe,1.5", MTe, 1.5)]
    for name, fn in binary.items():
        for key, x, y in pairs:
            # ma.compress(condition, a): a TEXT condition selects the indices of its non-empty strings (np.compress's C
            # loop, which the tracer cannot see): the computational phase, as in strma.
            phase = 2 if name == "compress" and _is_text_arg(x) else "compute"
            add_ma("ma2", params={"name": name}, args=[x, y], layout=key, phase=phase, call=lambda a, b, f=fn: f(a, b),
                   section="tma", trace_t=name not in _TMA_UNTRACEABLE)

    # ---- constructors over StringDType: masks, fill values, mask reconciliation ----
    def t3(dt=T):
        """A fresh ``['ab', 'c', 'de']`` array of the descriptor ``dt``."""
        return np.array(["ab", "c", "de"], dt)

    ctor = [
        ("t_mask", lambda: ma.array(np.arange(3), mask=np.array(["a", "", "b"], T))),
        ("t_mask_na", lambda: ma.array(np.arange(3), mask=np.array(["a", None, "b"], TN))),
        ("t_mask_empty", lambda: ma.array(np.zeros(0), mask=np.zeros(0, T))),
        ("t_mask_size", lambda: ma.array(np.arange(3), mask=np.array(["a", ""], T))),
        ("mask_size", lambda: ma.array(t3(), mask=[True, False])),
        ("mask_one", lambda: ma.array(t3(), mask=[True])),
        ("mask_reshape", lambda: ma.array(np.array([["ab", "c", "x"], ["de", "f", "y"]], T), mask=np.zeros((3, 2), bool))),
        ("mask_true", lambda: ma.array(t3(), mask=True)),
        ("mask_false", lambda: ma.array(t3(), mask=False)),
        ("keep_mask", lambda: ma.array(ma.array(t3(), mask=[1, 0, 0]), mask=[0, 1, 0])),
        ("keep_mask_false", lambda: ma.array(ma.array(t3(), mask=[1, 0, 0]), mask=[0, 1, 0], keep_mask=False)),
        ("fill_hello", lambda: ma.array(t3(), fill_value="hello")),
        ("fill_long", lambda: ma.array(t3(), fill_value="f" * 40)),
        ("fill_5", lambda: ma.array(t3(), fill_value=5)),
        ("fill_15", lambda: ma.array(t3(), fill_value=1.5)),
        ("fill_true", lambda: ma.array(t3(), fill_value=True)),
        ("fill_bytes", lambda: ma.array(t3(), fill_value=b"xyz")),
        ("fill_5_coerce_false", lambda: ma.array(t3(TC), fill_value=5)),
        ("fill_text_coerce_false", lambda: ma.array(t3(TC), fill_value="q")),
        ("fill_none_na", lambda: ma.array(t3(TN), fill_value="zz")),
        ("fill_array_t", lambda: ma.array(t3(), fill_value=np.array(["xyz"], T))),
        ("fill_array_u", lambda: ma.array(t3(), fill_value=np.array(["xyz"]))),
        ("fill_0d_t", lambda: ma.array(t3(), fill_value=np.array("xyz", T))),
        ("fill_t_into_u", lambda: ma.array(np.array(["ab", "c"]), fill_value=np.array("xyz", T))),
        ("fill_t_into_f", lambda: ma.array(np.array([1.5]), fill_value=np.array("2.5", T))),
        ("fill_inherit", lambda: ma.array(ma.array(t3(), fill_value="zz"))),
        ("fill_inherit_dtype_u", lambda: ma.array(ma.array(t3(), fill_value="zz"), dtype="U1")),
        ("fill_inherit_dtype_t", lambda: ma.array(ma.array(np.array(["ab"]), fill_value="zz"), dtype=T)),
        ("fill_inherit_dtype_tn", lambda: ma.array(ma.array(t3(), fill_value="zz"), dtype=TN)),
        ("hard_inherit", lambda: ma.array(ma.array(t3(), mask=[1, 0, 0], hard_mask=True))),
        ("masked_all_t", lambda: ma.masked_all((3,), dtype=T)),
        ("masked_all_tn", lambda: ma.masked_all((2, 2), dtype=TN)),
        ("masked_all_tc", lambda: ma.masked_all((2,), dtype=TC)),
        ("default_fill_t", lambda: ma.default_fill_value(T)),
        ("default_fill_tn", lambda: ma.default_fill_value(TN)),
        ("default_fill_tnan", lambda: ma.default_fill_value(TNAN)),
        ("default_fill_tc", lambda: ma.default_fill_value(TC)),
        ("default_fill_t_arr", lambda: ma.default_fill_value(t3())),
        ("minimum_fill_t", lambda: ma.minimum_fill_value(t3())),
        ("maximum_fill_t", lambda: ma.maximum_fill_value(t3(TN))),
        ("make_mask_t", lambda: ma.make_mask(np.array(["a", ""], T))),
        ("make_mask_t_empty", lambda: ma.make_mask(np.zeros(0, T), shrink=False)),
        ("make_mask_tn", lambda: ma.make_mask(np.array(["a", None], TN))),
        ("mask_or_t_bool", lambda: ma.mask_or(t3(), np.zeros(3, bool))),
        ("mask_or_bool_t", lambda: ma.mask_or(np.zeros(3, bool), t3())),
        ("mask_or_t_u", lambda: ma.mask_or(t3(), np.array(["a", "b", "c"]))),
        ("mask_or_t_t", lambda: ma.mask_or(t3(), t3())),
        ("mask_or_t_tn", lambda: ma.mask_or(t3(), t3(TN))),
        ("masked_where_t", lambda: ma.masked_where(np.array(["a", "", "b"], T), np.arange(3))),
        ("masked_where_t_short", lambda: ma.masked_where(np.array(["a", ""], T), np.arange(3))),
        ("masked_where_short", lambda: ma.masked_where([True, False], t3())),
        ("masked_where_allfalse", lambda: ma.masked_where([False, False, False], t3())),
        ("masked_where_copy_false", lambda: ma.masked_where([True, False, False], t3(), copy=False)),
        ("putmask_plain", lambda: ma.putmask(t3(), [True, False, True], "zz")),
        ("compress_rows_1d", lambda: ma.compress_rows(t3())),
        ("compress_cols_1d", lambda: ma.compress_cols(t3())),
        ("mask_rows_1d", lambda: ma.mask_rows(t3())),
        ("mask_cols_1d", lambda: ma.mask_cols(t3())),
        ("common_fill_t_u", lambda: ma.common_fill_value(ma.array(t3()), ma.array(np.array(["a"])))),
        ("common_fill_t_t", lambda: ma.common_fill_value(ma.array(t3()), ma.array(t3()))),
        ("common_fill_t_tn", lambda: ma.common_fill_value(ma.array(t3()), ma.array(t3(TN)))),
        ("common_fill_t_f", lambda: ma.common_fill_value(ma.array(t3()), ma.array(np.array([1.5])))),
        ("array_list", lambda: ma.array(["ab", "c"], dtype=T, mask=[0, 1])),
        ("array_list_na", lambda: ma.array(["ab", None], dtype=TN, mask=[0, 0])),
        ("array_list_num", lambda: ma.array([1, 2.5], dtype=T)),
        ("array_list_num_coerce_false", lambda: ma.array([1, 2.5], dtype=TC)),
        ("masked_array_t", lambda: ma.masked_array(t3(), mask=[0, 0, 1], fill_value="w")),
        ("asarray_t", lambda: ma.asarray(t3())),
        ("asanyarray_t", lambda: ma.asanyarray(t3())),
        ("filled_plain_t", lambda: ma.filled(t3())),
        ("getdata_t", lambda: ma.getdata(t3())),
        ("getmaskarray_t", lambda: ma.getmaskarray(t3())),
        ("zeros_t", lambda: ma.zeros(3, dtype=T)),
        ("ones_t", lambda: ma.ones(2, dtype=T)),
        ("empty_t", lambda: ma.empty(2, dtype=T)),
        ("masked_values_t", lambda: ma.masked_values(t3(), "c")),
        ("masked_object_t", lambda: ma.masked_object(t3(), "c")),
        ("masked_equal_t", lambda: ma.masked_equal(t3(), "c")),
        ("fix_invalid_t", lambda: ma.fix_invalid(t3())),
        ("masked_invalid_t", lambda: ma.masked_invalid(t3())),
        ("concatenate_t_u", lambda: ma.concatenate([ma.array(t3(), mask=[1, 0, 0]), ma.array(np.array(["q"]))])),
        ("concatenate_u_t", lambda: ma.concatenate([ma.array(np.array(["q"])), ma.array(t3(), mask=[1, 0, 0])])),
        ("concatenate_t_tn", lambda: ma.concatenate([ma.array(t3()), ma.array(t3(TN))])),
        ("concatenate_t_s", lambda: ma.concatenate([ma.array(t3()), ma.array(np.array([b"q"]))])),
        ("stack_t", lambda: ma.stack([ma.array(t3(), mask=[1, 0, 0]), ma.array(t3())])),
        ("where_t_u", lambda: ma.where([True, False, True], ma.array(t3(), mask=[1, 0, 0]), np.array(["q", "r", "s"]))),
        ("where_t_num", lambda: ma.where([True, False, True], ma.array(t3()), 1.5)),
    ]
    for name, fn in ctor:
        # No operand to trace: these build their arrays inside the call and none reads a text (casts, fills, stores and
        # refusals are phase 1); the masked_values / masked_object / masked_equal of a text compare it (phase 2).
        # Op tma0, not ma0: the names repeat strma's constructors over StringDType arrays, and the replay spells each.
        phase = 2 if name in ("masked_values_t", "masked_object_t", "masked_equal_t", "mask_or_t_t") else 1
        add_ma("tma0", params={"name": name}, args=[], layout="", call=fn, section="tma", phase=phase)



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
    """One traced value: text / bytes as hex, numbers by kind, a 1-D chunk (external_loop) as a bracketed list, and a
    StringDType element that is the NA object ``None`` as ``N`` (a NaN NA object is the float ``fnan``)."""
    if isinstance(v, list):
        return "[" + ",".join(_it_val(x) for x in v) + "]"
    if v is None:
        return "N"
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
    if isinstance(v, (complex, np.complexfloating)):
        return "c" + repr(float(v.real)) + "," + repr(float(v.imag))
    raise TypeError(f"untraceable value {type(v)}")


def _it_trace(views, flags=None, op_flags=None, op_dtypes=None, order="K", casting="safe", buffersize=0, writes=None,
              copy_to=None, iterrange=None, op_axes=None, setter=False):
    """One iteration traced as text (see the section comment above).

    :param views: the operands (None for an output the iterator allocates).
    :param flags: the global flags.
    :param op_flags: the per-operand flags.
    :param op_dtypes: the requested descriptors: dtype strings or ``TSpec`` (StringDType) requests, None entries kept.
    :param order: the iteration order.
    :param casting: the casting rule.
    :param buffersize: the buffer size (0: NumPy's default).
    :param writes: values written into operand 0 at each step (cycled), or None.
    :param copy_to: an operand that receives operand 0's view at each step (``y[...] = x``), or None.
    :param iterrange: ``it.iterrange`` to set before the walk, or None.
    :param op_axes: the per-operand axis maps, or None.
    :param setter: write ``writes`` through the iterator's setter (``it[0] = w``) instead of the view (``x[...] = w``).
    :returns: the trace.
    """
    kw = {}
    if flags is not None:
        kw["flags"] = flags
    if op_flags is not None:
        kw["op_flags"] = op_flags
    if op_dtypes is not None:
        kw["op_dtypes"] = [None if d is None else sbuild(d) for d in op_dtypes]
    if op_axes is not None:
        kw["op_axes"] = op_axes
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
                    if setter:
                        it[0] = writes[k % len(writes)]
                    else:
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
        kw["op_dtypes"] = [None if d is None else sbuild(d) for d in op_dtypes]
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
               copy_to=None, iterrange=None, none_ops=None, op_axes=None, setter=False):
    p = {"order": order, "casting": casting, "buffersize": buffersize}
    if flags is not None:
        p["flags"] = flags
    if op_flags is not None:
        p["op_flags"] = op_flags
    if op_dtypes is not None:
        # A StringDType request travels as its TSpec encoding ({"T": {...}}), a dtype string as itself.
        p["op_dtypes"] = [None if d is None else senc(d) for d in op_dtypes]
    if op_axes is not None:
        p["op_axes"] = op_axes
    if setter:
        p["setter"] = True
    if writes is not None:
        p["writes"] = enc(writes)
    if copy_to is not None:
        p["copy_to"] = copy_to
    if iterrange is not None:
        p["iterrange"] = iterrange
    if none_ops is not None:
        p["none_ops"] = none_ops
    return p


def _it_add(name, pairs, phase=1, layout="", section="striter", ascii_trace=False, **kw):
    """One nditer trace case: pairs are (base, view) operands, None entries the outputs NumPy allocates; ``section``
    names the case's section (``striter`` for the fixed-width traces, ``titer`` for StringDType); ``ascii_trace`` stores
    the trace as ASCII text (``_AsciiText``) instead of UTF-32 hex."""
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
        trace = _it_trace(it_views, **kw)
        return _AsciiText(trace) if ascii_trace else trace

    add(section, "nditer_" + name.replace("/", "_"), params=params, operands=real, call=call, phase=phase,
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
        _pop_case()


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
    iip = num_operand(np.array([1, 2, 3, 4]))
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
        # where= with an out whose text parses (selected slots written, masked-off kept) and one that does not. power
        # reads non-negative exponents here: NumPy 2.4.2's answer to a negative integer exponent under where= with a text
        # out depended on the process's history (the same call raised "Integers to negative integer powers are not
        # allowed." alone and returned without raising after the other sections ran), so it cannot be a contract; the
        # plain out= cases above keep that error.
        wins = [iip if x is ii else x for x in ins] if name == "power" else ins
        for wname, text in (("ok", ["7", "8", "9", "1"]), ("bad", ["7", "x", "9", "1"]), ("truth", ["", "a", "b", ""])):
            _conv("strconv", "ufunc_kw", params={"name": name, "dtype": None, "nin": len(wins), "out": len(wins),
                                                 "where": len(wins) + 1},
                  operands=wins + [target(np.array(text, dtype="U8")), num_operand(np.array([True, False, True, False]))],
                  layout=f"where:{wname}", call=lambda *xs, f=fn, n=len(wins): f(*xs[:n], out=xs[n], where=xs[n + 1]))
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
        _pop_case()


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


# ----------------------------------------------------------------------------------------------------------------------
# Section titer: np.nditer and np.nested_iters over NEP 55 StringDType (strings phase 1, P1.7)
# ----------------------------------------------------------------------------------------------------------------------
#
# The striter trace (_it_trace) over StringDType operands: a T base travels by value (_describe_t), an NA element is
# traced as "N" (a NaN NA object as "fnan"), and a StringDType request is an encoded TSpec. NumPy 2.4.2 raises MemoryError
# ("Failed to load string in StringDType getitem", then "String deallocation failed in clear loop" at close) for a
# buffered iteration with a StringDType input and an ALLOCATED StringDType output; those cases are not recorded (a NumPy
# bug: NumSharp's values are pinned by StringDTypeIterationTests).

# Short texts (in the cell), arena texts (16+ UTF-8 bytes: the shortest ones, to keep the traces small), the empty
# string, non-ASCII and supplementary-plane text.
T_ITER_POOL = ["a", "bb", "abcdefghijklmnop", "", "é" * 8, "\U0001F600", "q" * 17, "x y"]
# Texts NumPy parses as numbers (CPython int(): spaces, underscores and a sign allowed).
T_NUM_POOL = ["12", "-7", "3", " 4 ", "1_0", "+5"]
# The same numbers as complex texts whose two parts are equal, for a complex128 request: NumPy 2.4.2's StringDType ->
# complex loop writes the REAL part into the imaginary part too (casts.cpp: npy_csetimagfunc(out, complex_value.real)),
# so only a text with real == imag gives the value complex() parses — the value NumSharp writes (NDTextParse's
# ParseVStringElement; the bug itself is pinned by unit tests, never recorded).
T_CNUM_POOL = ["12+12j", "-7-7j", "(3+3j)", " 4+4j ", "1_0+1_0j", "+5+5j"]
# The titer cases NumPy's allocated-output MemoryError removed (reported by main()).
_TITER_DROPPED = []


def _titer_add(name, pairs, **kw):
    """One ``titer`` trace case (``_it_add`` into section ``titer``), dropped again when NumPy raises its StringDType
    allocated-output MemoryError.

    :param name: the case name (op key ``nditer_<name>``).
    :param pairs: (base, view) operands, ``(None, None)`` for an output the iterator allocates.
    :param kw: ``_it_trace`` arguments.
    """
    _it_add(name, pairs, section="titer", ascii_trace=True, **kw)
    last = CASES[-1]
    err = last.get("error")
    exp = last.get("expected") or {}
    text = exp.get("a", "") if exp.get("kind") == "text" else ""
    if (err is not None and err["type"] == "MemoryError") or "!MemoryError" in text:
        _pop_case()
        _TITER_DROPPED.append(name)


def _zerosize(view):
    """The flag an empty operand needs: NumPy refuses to iterate a zero-size operand without ``zerosize_ok``.

    :param view: the operand view.
    :returns: ``["zerosize_ok"]`` for an empty view, else ``[]``.
    """
    return ["zerosize_ok"] if view.size == 0 else []


def _t_op(spec, layout, seed, pool=None):
    """A StringDType operand over ``T_ITER_POOL`` (or ``pool``), every fourth element NA for a descriptor with an NA
    object: see ``t_operand``.

    :param spec: the TSpec.
    :param layout: a ``LAYOUTS`` name.
    :param seed: the first pool index.
    :param pool: the texts, or None for ``T_ITER_POOL``.
    :returns: the (base, view) pair.
    """
    return t_operand(spec, layout, seed=seed, pool=pool or T_ITER_POOL)


def _other_t(spec):
    """The StringDType request that differs from ``spec``: the NA descriptor for one without an NA object, and
    ``StringDType()`` for one with.

    :param spec: the TSpec.
    :returns: TDN or TD.
    """
    return TDN if spec.na is _ABSENT else TD


def gen_titer():
    """Section titer (strings phase 1, P1.7): np.nditer and np.nested_iters over StringDType operands — the value
    stream of every layout, order and flag set; buffered casts out of and into StringDType (U, S, numbers, bool, complex,
    the NA descriptors) under every casting rule and two buffer sizes; the copy routes; writes through the views and
    through the iterator's setter (texts, arena and long strings, None, numbers, coerce=False refusals), plain, buffered
    and through cast buffers; allocated StringDType outputs; operand pairs with common_dtype; reductions (op_axes, the
    stride-0 external-loop chunk, a cast reduction operand, a float64 one); nested iterators; NumPy's refusals.
    """
    rok = ["refs_ok"]
    specs = [TD, TDN]

    # A. the value stream: every layout x both descriptors x the flag sets x three orders. (A buffered index is
    # uninitialized memory in NumPy, so c_index runs unbuffered only.)
    flag_sets = [rok, rok + ["multi_index"], rok + ["c_index"], rok + ["external_loop"], rok + ["buffered"],
                 rok + ["buffered", "external_loop"]]
    layouts = ["c1", "c2", "c3", "f2", "t3", "s1", "s2", "r1", "r2", "col", "nx", "bc", "bcol", "z0", "d5", "e2"]
    for spec in specs:
        for lname in layouts:
            for flags in flag_sets:
                # The orders differ only over two or more axes (a 1-D walk is the same in C and F; K follows memory).
                op = _t_op(spec, lname, seed=len(lname))
                orders = ("K", "C", "F") if op[1].ndim >= 2 else ("K", "C")
                for order in orders:
                    op = _t_op(spec, lname, seed=len(lname))
                    _titer_add("values", [op], layout=lname, flags=flags + _zerosize(op[1]), order=order)

    # B. buffered casts out of StringDType: texts, NA cells and number texts into the other StringDType descriptors, U / S
    # of two widths, numbers, bool and complex. Every casting rule on one layout (the rule decides whether the cast runs);
    # the cast that runs over every other layout, with the default buffer and one of two elements (a refill every two
    # steps).
    sources = [("t", TD, None), ("tn", TDN, None), ("tnum", TD, T_NUM_POOL), ("tnnum", TDN, T_NUM_POOL)]
    requests = [TD, TDN, TDNAN, TDC, "<U5", "<U30", "|S5", "<i8", "<f8", "?", "<c16"]
    for sname, spec, pool in sources:
        for req in requests:
            # A number source cast to complex128 reads complex texts with equal parts (T_CNUM_POOL: NumPy's bug is
            # invisible there); the other texts raise complex()'s ValueError at their first element in both libraries.
            rpool = T_CNUM_POOL if pool is T_NUM_POOL and req == "<c16" else pool
            for casting in ("no", "safe", "same_kind", "unsafe"):
                op = _t_op(spec, "c1", seed=1, pool=rpool)
                _titer_add(f"cast_from/{sname}", [op], layout="c1", flags=rok + ["buffered"], op_dtypes=[req],
                           casting=casting)
            for lname in ("r2", "f2", "s2", "bc", "z0", "t3"):
                for bs in (0, 2):
                    op = _t_op(spec, lname, seed=1, pool=rpool)
                    _titer_add(f"cast_from/{sname}", [op], layout=lname, flags=rok + ["buffered"], op_dtypes=[req],
                               casting="unsafe", buffersize=bs)

    # C. buffered casts into StringDType: numbers and bool become their str(), U / S text keeps its characters, and a
    # NaN meets a NaN NA object.
    num_sources = {
        "i8": np.array([1, 22, -3, 0, 2 ** 40, -2 ** 63], dtype=np.int64),
        "u1": np.array([7, 200, 0, 255, 1, 2], dtype=np.uint8),
        "f8": np.array([1.5, -2.0, float("nan"), 1e20, -0.0, float("inf")]),
        "f2": np.array([0.1, 65504, -1, 0.5, float("nan"), 2], dtype=np.float16),
        "b1": np.array([True, False, True, True, False, False]),
        "c16": np.array([1 + 2j, -1j, 0, complex(float("nan"), 1), 2, 3j]),
    }
    views = (("c1", lambda b: b), ("r1", lambda b: b[::-1]), ("s1", lambda b: b[::2]))
    for sname, src in num_sources.items():
        for req in (TD, TDN, TDNAN, TDC):
            for casting in ("safe", "same_kind", "unsafe"):
                for lname, view in views:
                    for bs in (0, 2):
                        base = src.copy()
                        _titer_add(f"cast_to/{sname}", [(base, view(base))], layout=lname, flags=rok + ["buffered"],
                                   op_dtypes=[req], casting=casting, buffersize=bs)
    for spec, ascii_only in (("<U3", False), ("|S3", True)):
        for req in (TD, TDN, TDC):
            for casting in ("safe", "same_kind", "unsafe"):
                for lname in ("c1", "r2", "f2", "bc", "z0"):
                    for bs in (0, 2):
                        op = operand(spec, lname, seed=3, ascii_only=ascii_only)
                        _titer_add("cast_to/text", [op], layout=lname, flags=rok + ["buffered"], op_dtypes=[req],
                                   casting=casting, buffersize=bs)

    # D. the copy routes: a copy operand reads a temporary of the request, an updateifcopy one writes it back at close,
    # a read-only 0-d operand is copied without being asked; StringDType requests over U operands too.
    for spec in specs:
        for lname in ("c1", "r2", "f2", "z0", "t3", "e2"):
            for req in (_other_t(spec), "<U5", "<U30", "|S5"):
                op = _t_op(spec, lname, seed=2)
                _titer_add("copy_ro", [op], layout=lname, flags=rok + _zerosize(op[1]),
                           op_flags=[["readonly", "copy"]], op_dtypes=[req], casting="unsafe")
                op = _t_op(spec, lname, seed=2)
                _titer_add("updateifcopy", [op], layout=lname, flags=rok + _zerosize(op[1]),
                           op_flags=[["readwrite", "updateifcopy"]], op_dtypes=[req], casting="unsafe",
                           writes=["XYZW", "ab", ""])
                op = _t_op(spec, lname, seed=2)
                _titer_add("zero_d_auto", [op], layout=lname, flags=rok + _zerosize(op[1]), op_dtypes=[req],
                           casting="unsafe")
    for lname in ("c1", "r2", "f2", "z0"):
        for req in (TD, TDN):
            op = operand("<U3", lname, seed=4)
            _titer_add("copy_ro/text", [op], layout=lname, flags=rok, op_flags=[["readonly", "copy"]], op_dtypes=[req],
                       casting="unsafe")
            op = operand("<U3", lname, seed=4)
            _titer_add("updateifcopy/text", [op], layout=lname, flags=rok, op_flags=[["readwrite", "updateifcopy"]],
                       op_dtypes=[req], casting="unsafe", writes=["zz", "a much longer string than fits", ""])

    # E. writes through the views and through the setter: texts, arena and long strings (a rewrite past the slot moves
    # to the heap), None (the NA object, or 'None'), numbers and bool (their str(); refused under coerce=False), plain,
    # buffered, through a StringDType or a U40 buffer, under external_loop.
    write_sets = {
        "same": ["Q", "RS", "TUVW"],
        "long": ["a string past the slot", "", "h" * 40],
        "na": [None, "x"],
        "text": ["é", "\U0001F600", "a\x00b"],
        "number": [5, 1.5, True],
    }
    modes = [("plain", [], None), ("buffered", ["buffered"], None), ("cast", ["buffered"], "other"),
             ("castu", ["buffered"], "<U40"), ("ext", ["external_loop"], None)]
    for spec in (TD, TDN, TDC):
        for wname, w in write_sets.items():
            for lname in ("c1", "c2", "r2", "f2", "s2", "z0", "t3"):
                for mname, mflags, req in modes:
                    for setter in (False, True):
                        op = _t_op(spec, lname, seed=5)
                        r = None if req is None else (_other_t(spec) if req == "other" else req)
                        _titer_add(f"write/{wname}/{mname}", [op], layout=lname, flags=rok + mflags,
                                   op_flags=[["readwrite"]], op_dtypes=None if r is None else [r], casting="unsafe",
                                   writes=w, setter=setter)

    # F. allocated outputs, filled from the input (a StringDType output buffered behind a StringDType input is NumPy's
    # MemoryError, dropped by _titer_add).
    for spec in specs:
        for req in (None, TD, TDN, "<U5"):
            for flags in (rok, rok + ["buffered"], rok + ["external_loop"]):
                for lname in ("c1", "c2", "r2", "f2", "z0", "e2"):
                    op = _t_op(spec, lname, seed=3)
                    _titer_add("alloc", [op, (None, None)], layout=lname, flags=flags + _zerosize(op[1]),
                               op_flags=[["readonly"], ["writeonly", "allocate"]], op_dtypes=[None, req],
                               casting="unsafe", copy_to=1)
    for req in (TD, TDN):
        for flags in (rok, rok + ["buffered"]):
            op = operand("<U3", "c2", seed=6)
            _titer_add("alloc/text", [op, (None, None)], layout="c2", flags=flags,
                       op_flags=[["readonly"], ["writeonly", "allocate"]], op_dtypes=[None, req], casting="unsafe",
                       copy_to=1)

    # G. operand pairs: a StringDType operand with U / S / numbers / bool / the other descriptor, each iterated as its
    # own, or with common_dtype (U is cast to StringDType; numbers have no common DType with it).
    others = {
        "u3": lambda: operand("<U3", "c1", seed=7)[0][:4],
        "s3": lambda: operand("|S3", "c1", seed=7, ascii_only=True)[0][:4],
        "i8": lambda: np.arange(4, dtype=np.int64),
        "f8": lambda: np.arange(4, dtype=np.float64) / 2,
        "b1": lambda: np.array([True, False, True, False]),
        "tn": lambda: np.array(["p", None, "q", "r"], dtype=TDN.build()),
    }
    for oname, make in others.items():
        for flags in (rok, rok + ["common_dtype"], rok + ["common_dtype", "buffered"], rok + ["buffered"]):
            a = _t_op(TD, "c2", seed=8)
            b = make().copy()
            _titer_add(f"pair/{oname}", [a, (b, b)], layout="pair", flags=flags, casting="unsafe")

    # H. reductions: a StringDType reduction operand (op_axes; the output follows the iterator's axis 1, or axis 0),
    # unbuffered with and without external_loop (the stride-0 chunk takes y[...] = x and keeps its last element), and
    # buffered, plain or through a cast buffer: NumPy's reduce double loop at the default size (one fill of 3 rows) and
    # with a buffer of 8 (fills of 2 rows, then 1), and buffers of 2 and 3 elements, shorter than a row (no double loop:
    # the core is one element); a float64 reduction operand's stride-0 chunk takes the input chunk and a constant
    # (through the view and through the setter).
    for spec in specs:
        for axes, outn in (([[0, 1], [-1, 0]], 4), ([[0, 1], [0, -1]], 3)):
            for flags, req in ((rok + ["reduce_ok"], None), (rok + ["reduce_ok", "external_loop"], None),
                               (rok + ["reduce_ok", "buffered"], None),
                               (rok + ["reduce_ok", "buffered"], "other"), (rok + ["reduce_ok", "buffered"], "<U8"),
                               (rok + ["reduce_ok", "buffered", "external_loop"], "other")):
                for bs in ((0, 2, 3, 8) if "buffered" in flags else (0,)):
                    a = _t_op(spec, "c2", seed=9)
                    out = np.array([""] * outn, dtype=spec.build())
                    r = None if req is None else (_other_t(spec) if req == "other" else req)
                    _titer_add("reduce", [a, (out, out)], layout="reduce", flags=flags,
                               op_flags=[["readonly"], ["readwrite"]], op_axes=axes, op_dtypes=[None, r],
                               casting="unsafe", copy_to=1, buffersize=bs)
    for axes, outn in (([[0, 1], [-1, 0]], 4), ([[0, 1], [0, -1]], 3)):
        a = np.arange(12, dtype=np.float64)
        out = np.zeros(outn)
        _titer_add("reduce/f8", [(a, a.reshape(3, 4)), (out, out)], layout="reduce",
                   flags=["reduce_ok", "external_loop"], op_flags=[["readonly"], ["readwrite"]], op_axes=axes,
                   copy_to=1)
        for setter in (False, True):
            a = np.arange(12, dtype=np.float64)
            out = np.zeros(outn)
            _titer_add("reduce/f8_const", [(out, out), (a, a.reshape(3, 4))], layout="reduce",
                       flags=["reduce_ok", "external_loop"], op_flags=[["readwrite"], ["readonly"]],
                       op_axes=[axes[1], axes[0]], writes=[7.5], setter=setter)

    # I. np.nested_iters over StringDType: plain, buffered, buffered through the other descriptor and through U30.
    for spec in specs:
        for lname, axes in (("c2", [[0], [1]]), ("c2", [[1], [0]]), ("f2", [[0], [1]]), ("c3", [[0], [1, 2]]),
                            ("c3", [[2], [0, 1]]), ("t3", [[1], [0, 2]]), ("r2", [[1], [0]])):
            for flags, req in ((rok, None), (rok + ["buffered"], None), (rok + ["buffered"], "other"),
                               (rok + ["buffered"], "<U30")):
                op = _t_op(spec, lname, seed=10)
                r = None if req is None else (_other_t(spec) if req == "other" else req)
                params = {"axes": axes, "flags": flags}
                if r is not None:
                    params["op_dtypes"] = [senc(r)]
                add("titer", "nested_iters", params=params, operands=[op], layout=lname,
                    call=lambda v, axes=axes, flags=flags, r=r: _AsciiText(_nested_trace(
                        v, axes, flags=flags, op_dtypes=None if r is None else [r])))

    # J. NumPy's refusals: no refs_ok (for an operand, for a request), a cast neither copied nor buffered (also between
    # two StringDType descriptors), a read-only broadcast written, and the casting rules.
    _titer_add("refuse/norefs", [_t_op(TD, "c2", seed=11)], layout="c2")
    ints = np.arange(4)
    _titer_add("refuse/norefs_request", [(ints, ints)], layout="c1", flags=["buffered"], op_dtypes=[TD])
    for req in (TDN, "<U5", "<i8"):
        _titer_add("refuse/unbuffered_cast", [_t_op(TD, "c2", seed=11)], layout="c2", flags=rok, op_dtypes=[req],
                   casting="unsafe")
    _titer_add("refuse/readonly_bcast", [_t_op(TD, "bc", seed=11)], layout="bc", flags=rok, op_flags=[["readwrite"]])
    for req, casting in (("<U5", "safe"), ("<i8", "safe"), ("<U5", "no"), (TDN, "no"), (TDN, "equiv"), ("?", "safe")):
        _titer_add("refuse/casting", [_t_op(TD, "c1", seed=11)], layout="c1", flags=rok + ["buffered"],
                   op_dtypes=[req], casting=casting)
    for casting in ("no", "equiv", "safe"):
        op = operand("<U3", "c1", seed=11)
        _titer_add("refuse/casting_text", [op], layout="c1", flags=rok + ["buffered"], op_dtypes=[TD], casting=casting)


# ----------------------------------------------------------------------------------------------------------------------
# Section cvtext: NumSharp's Converts API against NumPy's text rules (2026-10-13)
# ----------------------------------------------------------------------------------------------------------------------
#
# Converts (NumSharp's conversion helpers: the typed and object To* / ToString overloads, ChangeType by NPTypeCode and
# generic) speaks NumPy's text. cv_tostring records str() of the NumPy scalar given by its bytes; cv_fromstring records
# np.zeros(1, dtype) after a[0] = text (NumPy's store of a Python str: CPython int() / float() / complex(), NumPy's range
# check, a bool target taking the str's truth) or NumPy's error. The replay runs every Converts route and requires
# them to agree. A platform-dependent overflow text (Windows' C long) is dropped (_conv).

CV_VALUES = {
    "bool": [True, False],
    "int8": [0, 1, -1, -128, 127],
    "uint8": [0, 1, 255],
    "int16": [-32768, 32767, 7],
    "uint16": [65535, 0, 300],
    "int32": [-2 ** 31, 2 ** 31 - 1, 42],
    "uint32": [2 ** 32 - 1, 0, 123456],
    "int64": [-2 ** 63, 2 ** 63 - 1, -5],
    "uint64": [2 ** 64 - 1, 0, 2 ** 63],
    "float16": [0.0, -0.0, 0.1, 1.0, 65504.0, 6e-08, 1e4, 1.5, 0.3333, 2048.0, 100.0, float("nan"), float("inf"),
                float("-inf")],
    "float32": [0.1, 1e20, 1e16, 1e15, 3.4028235e38, 1e-45, -0.0, 16777217.0, 123456.789, 1e-5, 1e-4, 1.0,
                float("nan"), float("-inf")],
    "float64": [0.1, 1e20, 1e16, 1e15, 1e-5, 1e-4, 1.7976931348623157e308, 5e-324, -0.0, 1 / 3, 2.5, 100.0,
                12345678901234567890.0, float("nan"), float("inf")],
    "complex128": [1 + 2j, complex(-0.0, -1), 2j, 1e20 + 1e-5j, complex(float("nan"), 0),
                   complex(float("inf"), float("-inf")), 0j, complex(0.0, -0.0), 1.5 + 0j, complex(-1.5, 1e-300)],
}

CV_TEXTS = {
    "int": ["12", " -7 ", "1_000", "+5", "٣٢", "1.5", "0x10", "", "300", "-1", "127", "-128", "255", "256",
            "65535", "65536", "2147483647", "9223372036854775807", "18446744073709551615", "-9223372036854775808",
            "1e3", "nan", " ", "12abc", "0b11", "1__0", "_1", "--1", "\t8\n"],
    "float": ["1.5", " 2.5 ", "1e3", "-Infinity", "inf", "nan", "-nan", "1_000.5", "0x1p3", "1e400", "1e-400", "", "abc",
              "65520", "65504", "0.1", "3.4028235e38", "3.5e38", "  -0  ", "1e-45", "infinity", "NaN", "+1.0", "1.",
              ".5", "1e", "1_e3", "١.٥"],
    "complex": ["1+2j", "(1+2j)", "2j", "j", "1+", "(1)", " 1+2j ", "inf+nanj", "1e3-2.5e-3j", "", "1 + 2j", "-j",
                "(-0-0j)", "nan", "1+2J", "(1+2j", "1+2jj"],
    "bool": ["", "False", "0", "True", " ", "no", "\x00"],
}

CV_TARGETS = {
    "int": ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"],
    "float": ["float16", "float32", "float64"],
    "complex": ["complex128"],
    "bool": ["bool"],
}


def _cv_store(text, name):
    """NumPy's store of a Python str into a one-element array of ``name`` (overflow warnings silenced).

    :param text: the str.
    :param name: the dtype name.
    :returns: the array after ``a[0] = text``.
    """
    a = np.zeros(1, dtype=name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a[0] = text
    return a


def gen_cvtext():
    """Section cvtext: Converts' number -> text (the NumPy scalar's str() of every dtype's edge values) and text ->
    number (every integer width, the three float widths, complex and bool, over texts CPython's grammar accepts and
    refuses, past each width's range and with the non-ASCII digits CPython reads)."""
    for name, vals in CV_VALUES.items():
        for v in vals:
            raw = np.array([v], dtype=name).tobytes().hex()
            add("cvtext", "cv_tostring", params={"dtype": name, "bits": raw},
                call=lambda raw=raw, name=name: str(np.frombuffer(bytes.fromhex(raw), dtype=name)[0]))
    for group, names in CV_TARGETS.items():
        texts = CV_TEXTS[group] if group == "int" else CV_TEXTS[group] + CV_TEXTS["int"][:8]
        for name in names:
            for text in texts:
                _conv("cvtext", "cv_fromstring", params={"dtype": name, "text": enc(text)},
                      call=lambda text=text, name=name: _cv_store(text, name))


# ----------------------------------------------------------------------------------------------------------------------
# Section promote: promotion over three or more dtypes, and NEP 50's weak Python scalar beside text and numbers
# (2026-10-13)
# ----------------------------------------------------------------------------------------------------------------------

PROMO_DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32",
                "float64", "complex128"]

# Literals a C# caller writes for a Python scalar: ints in and out of a small dtype's range, a float, a bool.
PROMO_LITERALS = [1, -1, 300, 1000, 1.5, True]


def gen_promote():
    """Section promote: ``np.result_type`` of every ordered triple of the 13 NumPy dtypes and a stride of the ordered
    quadruples (NumPy's order-independent sequence promotion); ``np.concatenate`` of every triple of one-element arrays
    and ``np.stack`` / ``np.vstack`` / ``np.hstack`` of every fourth one; and a Python scalar (a C# literal) beside a
    text or numeric array in ``np.where`` (either side), ``np.choose``, ``np.select``'s default and ``np.r_`` — text
    beside a number is NumPy's DTypePromotionError, a number beside numbers is weak."""
    import itertools
    triples = list(itertools.product(PROMO_DTYPES, repeat=3))
    for t in triples:
        add("promote", "result_type", params={"specs": list(t)}, call=lambda t=t: np.result_type(*t))
    for t in list(itertools.product(PROMO_DTYPES, repeat=4))[::47]:
        add("promote", "result_type", params={"specs": list(t)}, call=lambda t=t: np.result_type(*t))
    for i, t in enumerate(triples):
        ops = [plain(np.array([v], dtype=d)) for v, d in zip((1, 2, 3), t)]
        add("promote", "promote_concatenate", operands=ops, layout="three", call=lambda *v: np.concatenate(v))
        if i % 4 == 0:
            add("promote", "promote_stack", operands=ops, layout="three", call=lambda *v: np.stack(v))
            add("promote", "promote_vstack", operands=ops, layout="three", call=lambda *v: np.vstack(v))
            add("promote", "promote_hstack", operands=ops, layout="three", call=lambda *v: np.hstack(v))

    arrays = {
        "u3": lambda: np.array(["ab", "c", "def", ""], dtype="<U3"),
        "s3": lambda: np.array([b"ab", b"c", b"def", b""], dtype="|S3"),
        "t": lambda: np.array(["ab", "c", "def", ""], dtype=TD.build()),
        "i1": lambda: np.array([1, -2, 3, 127], dtype=np.int8),
        "u1": lambda: np.array([1, 2, 3, 255], dtype=np.uint8),
        "f4": lambda: np.array([1.5, -2.0, 0.1, 3.0], dtype=np.float32),
        "f2": lambda: np.array([1.5, -2.0, 0.1, 3.0], dtype=np.float16),
        "b1": lambda: np.array([True, False, True, False]),
    }
    cond = np.array([True, False, False, True])
    idx = np.array([0, 1, 0, 1])
    for aname, make in arrays.items():
        for lit in PROMO_LITERALS:
            arr = make()
            add("promote", "promote_where_lit", params={"literal": enc(lit), "side": "y"},
                operands=[plain(cond), plain(arr)], layout=aname,
                call=lambda c, a, lit=lit: np.where(c, a, lit))
            arr = make()
            add("promote", "promote_where_lit", params={"literal": enc(lit), "side": "x"},
                operands=[plain(cond), plain(arr)], layout=aname,
                call=lambda c, a, lit=lit: np.where(c, lit, a))
            arr = make()
            add("promote", "promote_choose_lit", params={"literal": enc(lit)}, operands=[plain(idx), plain(arr)],
                layout=aname, call=lambda i, a, lit=lit: np.choose(i, [a, lit]))
            arr = make()
            add("promote", "promote_select_lit", params={"literal": enc(lit)}, operands=[plain(cond), plain(arr)],
                layout=aname, call=lambda c, a, lit=lit: np.select([c], [a], default=lit))
            arr = make()
            add("promote", "promote_r_lit", params={"literal": enc(lit)}, operands=[plain(arr)], layout=aname,
                call=lambda a, lit=lit: np.r_[a, lit])


# ----------------------------------------------------------------------------------------------------------------------
# Section storenone: Python objects stored into arrays — None into text through every index form, ndarray.__setitem__,
# and stores through a sub-array of a writeable broadcast (2026-10-13)
# ----------------------------------------------------------------------------------------------------------------------

STORE_INDICES_1D = [[{"int": 0}], [{"int": -1}], [{"ellipsis": True}], [{"slice": [None, 1, None]}],
                    [{"slice": [None, None, 2]}], [{"arr": [1, 0]}], [{"mask": [True, False, True, False, False, True]}],
                    [{"arr": []}], [{"int": 6}]]
STORE_INDICES_2D = [[{"int": 1}], [{"int": 1}, {"int": 2}], [{"slice": [None, None, None]}, {"int": 0}],
                    [{"mask2": [[True, False, False, True], [False] * 4, [True] * 4]}]]


def _set(a, key, value):
    """``a[key] = value`` with a one-entry key unwrapped (``a[(0,)]`` is ``a[0]`` for NumPy too, but the case reads
    better).

    :param a: the array.
    :param key: the index tuple.
    :param value: the value.
    """
    a[key if len(key) > 1 else key[0]] = value


def gen_storenone():
    """Section storenone: ``None`` stored into S / U arrays (``str(None)``, truncated to the width) and into StringDType
    arrays (the NA object, ``'None'``, or coerce=False's ValueError) through integers, an ellipsis, slices, index arrays,
    masks, an empty index and an out-of-range one, 1-D and 2-D; ``ndarray.__setitem__`` with an int or a slice and a
    Python object (None, ints in and out of range, a float into an integer array, text into numbers and numbers into
    text); and a store through a sub-array of an ``np.broadcast_arrays`` result whose stride is 0 (writeable, as NumPy's
    views inherit WRITEABLE: the store writes the source)."""
    for spec in ("<U2", "<U8", "|S2", "|S8"):
        for lname, indices in (("c1", STORE_INDICES_1D), ("c2", STORE_INDICES_2D)):
            for idx in indices:
                key = tuple(_py_index(e) for e in idx)
                op = operand(spec, lname, seed=1, ascii_only=True)
                add("storenone", "setitem", params={"index": idx, "value": enc(None)}, operands=[op], layout=lname,
                    mutate=0, call=lambda a, k=key: _set(a, k, None))
    for spec in (TD, TDN, TDC):
        for lname, indices in (("c1", STORE_INDICES_1D), ("c2", STORE_INDICES_2D)):
            for idx in indices:
                key = tuple(_py_index(e) for e in idx)
                op = _t_op(spec, lname, seed=2)
                add("storenone", "setitem", params={"index": idx, "value": enc(None)}, operands=[op], layout=lname,
                    mutate=0, call=lambda a, k=key: _set(a, k, None))

    # ndarray.__setitem__(key, value): the C# __setitem__(int / long / string, object) overloads.
    targets = {
        "u2": lambda: np.array(["ab", "cd", "ef"], dtype="<U2"),
        "s2": lambda: np.array([b"ab", b"cd", b"ef"], dtype="|S2"),
        "t": lambda: np.array(["ab", "cd", "ef"], dtype=TD.build()),
        "tn": lambda: np.array(["ab", "cd", "ef"], dtype=TDN.build()),
        "i1": lambda: np.array([1, 2, 3], dtype=np.int8),
        "i4": lambda: np.array([1, 2, 3], dtype=np.int32),
        "i8": lambda: np.array([1, 2, 3], dtype=np.int64),
        "f8": lambda: np.array([1.0, 2.0, 3.0]),
    }
    values = [None, 5, 1000, -1, 2.9, "xyz", "12", True, ""]
    keys = [{"int": 0}, {"int": -1}, {"slice": [None, 2, None]}, {"slice": [None, None, -1]}]
    for tname, make in targets.items():
        for key in keys:
            for v in values:
                # None into a number is NaN / False / int()'s TypeError in NumPy and NumSharp alike since P1.7b.
                arr = make()
                # A 1-D text target is its own base, so the replay rebuilds its OWNDATA; a numeric one stays a view (the
                # general corpus' reconstruction builds a view over the buffer): target()'s rule, with StringDType text.
                op = own(arr) if arr.dtype.kind in "SUT" else plain(arr)
                k = _py_index(key)
                add("storenone", "setitem_method", params={"key": key, "value": enc(v)}, operands=[op], layout=tname,
                    mutate=0, call=lambda a, k=k, v=v: a.__setitem__(k, v))

    # A store through a sub-array of a writeable broadcast (np.broadcast_arrays' results are writeable in NumPy 2.4.2,
    # with a DeprecationWarning, and their views inherit WRITEABLE even where the stride is 0).
    for which in ("a", "b"):
        for row in (0, 1, 2):
            for value in (9, [7, 8, 9]):
                x = np.arange(3, dtype=np.int64)

                def store(x, which=which, row=row, value=value):
                    """``broadcast_arrays(x[:, None], x)[which][row][...] = value``: a whole-view store through one row of
                    a writeable broadcast. ``a``'s rows are stride-0 views of ``x[row]`` (the row's offset decides the
                    element; every slot aliases it, so the last value stays), ``b``'s rows are views of all of ``x``.

                    :param x: the int64 operand, mutated in place (the case records it).
                    :param which: ``"a"`` (the stretched column) or ``"b"`` (the stretched row).
                    :param row: the row index into the (3, 3) broadcast.
                    :param value: a Python int or a list of three.
                    """
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        a, b = np.broadcast_arrays(x[:, None], x)
                        (a if which == "a" else b)[row][...] = value

                add("storenone", "bcast_subview_store", params={"which": which, "row": row, "value": enc(value)},
                    operands=[plain(x)], layout="c1", mutate=0, call=store)

    _gen_none_routes()


# P1.7b: None into NUMERIC arrays (and into text through the routes the index forms above do not reach). NumPy's setitem of
# None is NaN for a float dtype (NPY_NAN, the POSITIVE quiet NaN: the corpus compares bytes, so the sign is checked),
# nan+nanj for complex, False for bool, and int()'s TypeError for every integer dtype.
NONE_NUM_SPECS = ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "c16"]
NONE_TEXT_SPECS = ["<U4", "<U2", "|S2", TD, TDN, TDC]


def _none_num_op(spec, layout):
    """A numeric operand for the None routes: the base holds 1, 2, 3, ... (bool: alternating), so a stored NaN / False is
    visible against it. The replay rebuilds a non-empty numeric operand as a VIEW over its buffer (``target``'s rule), so
    a layout whose view IS the base is given as a reshape view (OWNDATA False on both sides), and an empty operand only as
    the 1-D base itself (``e1``: the replay owns it, as NumPy's base does) — ``e2`` / ``e3`` map to ``e1``.

    :param spec: a numeric dtype string.
    :param layout: a ``LAYOUTS`` name.
    :returns: the (base, view) pair.
    """
    if layout in ("e2", "e3"):
        layout = "e1"
    n, build = LAYOUTS[layout]
    base = (np.arange(n) % 2 == 1) if spec == "?" else np.arange(1, n + 1).astype(spec)
    view = build(base)
    if view is base and base.size > 0:
        view = base.reshape(base.shape)
    return base, view


def _none_op(spec, layout):
    """An operand of any None-route target: numeric (``_none_num_op``), fixed-width text (ASCII pool values, so the
    stored ``'None'`` and its truncation are the only text that varies) or StringDType (``_t_op``).

    :param spec: a dtype string or a TSpec.
    :param layout: a ``LAYOUTS`` name.
    :returns: the (base, view) pair.
    """
    if isinstance(spec, TSpec):
        return _t_op(spec, layout, seed=2, pool=T_SHORT)
    if np.dtype(spec).kind in "SU":
        return operand(spec, layout, seed=1, ascii_only=True)
    return _none_num_op(spec, layout)


def _none_shaped(spec, shape):
    """A C-ordered operand of ``shape`` for the fill_diagonal / put_along_axis routes.

    :param spec: a dtype string or a TSpec.
    :param shape: the dimensions.
    :returns: the (base, view) pair.
    """
    n = int(np.prod(shape))
    text = isinstance(spec, TSpec) or np.dtype(spec).kind in "SU"
    if isinstance(spec, TSpec):
        base = np.array([T_SHORT[i % len(T_SHORT)] for i in range(n)], dtype=spec.build())
    elif text:
        base = np.array([U_ASCII[(i + 1) % len(U_ASCII)] for i in range(n)], dtype=spec)
    elif spec == "?":
        base = np.arange(n) % 2 == 1
    else:
        base = np.arange(1, n + 1).astype(spec)
    # The replay rebuilds a 1-D text operand with its base's shape as the base itself (OWNDATA True), a numeric one
    # as a view over the buffer: the same objects here.
    return base, (base if text and len(shape) == 1 else base.reshape(shape))


def _none_spec_name(spec):
    """The layout label of a target spec (a TSpec has no string of its own).

    :param spec: a dtype string or a TSpec.
    :returns: the label.
    """
    if isinstance(spec, TSpec):
        return {id(TD): "T", id(TDN): "Tna", id(TDC): "Tnc"}[id(spec)]
    return spec


def _gen_none_routes():
    """None through every store route NumPy has, into numbers and text: the index forms (numeric targets: the text ones
    are above), ``fill``, ``np.full`` / ``np.full_like``, ``np.copyto`` (every casting rule; ``'unsafe'`` with each mask form
    and destination size, which decide whether the object cast runs up front, per selected element, or not at all),
    ``np.put`` (in-bounds, negative, out-of-bounds, empty), ``np.putmask``, ``np.place`` (not StringDType: NumPy 2.4.2 crashes
    there), ``np.insert`` (scalar, sequence, empty, slice, array and mask positions, axes, an out-of-bounds index), the flat
    iterator (an integer index rewords any failure as "Error setting single item of array.", a slice or index list does
    not), ``np.fill_diagonal`` (2-D, tall with wrap, 3-D, empty), ``np.put_along_axis``, ``ndarray.setfield``,
    ``np.asanyarray`` / ``np.asarray`` with a dtype, and ``np.array`` of a list holding None. Every case records NumPy's
    outcome: the mutated operand (bytes, so NaN's sign is compared) or the result, or NumPy's error and its order against
    the other checks (read-only, bounds, mask size)."""
    num = NONE_NUM_SPECS
    alls = NONE_NUM_SPECS + NONE_TEXT_SPECS

    def ns(route, spec, layout_label, operands, call, mutate=0, **params):
        add("storenone", "none_store", params={"route": route, **params}, operands=operands,
            layout=f"{_none_spec_name(spec)}/{layout_label}", mutate=mutate, call=call)

    # The index forms beyond the store-index lists: views, 0-d, empty, read-only, F-ordered, transposed.
    for spec in num:
        for lname, idx in (("c1", [{"int": 6}]), ("s1", [{"ellipsis": True}]), ("r1", [{"int": 0}]),
                           ("col", [{"slice": [1, None, None]}]), ("z0", [{"ellipsis": True}]),
                           ("e2", [{"ellipsis": True}]), ("bc", [{"int": 0}]), ("f2", [{"int": 1}]),
                           ("t3", [{"slice": [None, None, None]}, {"int": 0}]), ("c1", [{"slice": [6, None, None]}]),
                           ("c1", [{"arr": [6]}]), ("c1", [{"mask": [True, False, True]}])):
            key = tuple(_py_index(e) for e in idx)
            add("storenone", "setitem", params={"index": idx, "value": enc(None)},
                operands=[_none_num_op(spec, lname)], layout=f"{spec}/{lname}", mutate=0,
                call=lambda a, k=key: _set(a, k, None))
        for lname, indices in (("c1", STORE_INDICES_1D), ("c2", STORE_INDICES_2D)):
            for idx in indices:
                key = tuple(_py_index(e) for e in idx)
                add("storenone", "setitem", params={"index": idx, "value": enc(None)},
                    operands=[_none_num_op(spec, lname)], layout=f"{spec}/{lname}", mutate=0,
                    call=lambda a, k=key: _set(a, k, None))

    for spec in alls:
        # fill: PyArray_FillWithScalar packs first (an empty integer array raises), then writes every element.
        for lname in ("c1", "c2", "s1", "r2", "z0", "e1", "e2", "bc"):
            ns("fill", spec, lname, [_none_op(spec, lname)], lambda a: a.fill(None))

        # np.full / np.full_like: empty(...) then copyto(..., None, casting='unsafe') — nothing is converted for an empty
        # result.
        dt = sbuild(spec)
        for dims in ([2], [2, 3], [], [0], [0, 3], [1]):
            ns("full", spec, "-", [], lambda dims=dims, dt=dt: np.full(tuple(dims), None, dtype=dt), mutate=None,
               dims=dims, dtype=senc(spec))
        for lname in ("c1", "f2", "s1", "e2"):
            ns("full_like", spec, lname, [_none_op(spec, lname)], lambda a: np.full_like(a, None), mutate=None)

        # np.copyto(dst, None): a 0-d OBJECT source — only 'unsafe' casts it; more than one destination element converts
        # it up front (an integer destination raises under any mask), one element per selected element, none for empty.
        for casting in ("no", "equiv", "safe", "same_kind"):
            ns("copyto", spec, "c1", [_none_op(spec, "c1")], lambda a, c=casting: np.copyto(a, None, casting=c),
               casting=casting, where=None)
        masks = {"c1": [None, [True, False, True, False, False, True], [False] * 6, True, False, [True, False, True]],
                 "one": [None, [True], [False], [True, False, True]],
                 "z0": [None, True, False],
                 "e1": [None, [True, False, True]]}
        for lname, wheres in masks.items():
            for w in wheres:
                ns("copyto", spec, lname, [_none_op(spec, lname)],
                   # where=None would be NumPy's mask None (False: nothing copied), so no mask passes no where=.
                   lambda a, w=w: np.copyto(a, None, casting="unsafe") if w is None
                   else np.copyto(a, None, casting="unsafe", where=np.array(w)),
                   casting="unsafe", where=w)
        for casting in ("same_kind", "unsafe"):
            ns("copyto", spec, "bc", [_none_op(spec, "bc")], lambda a, c=casting: np.copyto(a, None, casting=c),
               casting=casting, where=None)

        # np.put: the values are converted after the read-only, index-cast and empty-target checks and before the bounds.
        for lname, idx, mode in (("c1", [0, 2], "raise"), ("c1", [-1], "raise"), ("c1", [6], "raise"), ("c1", [], "raise"),
                                 ("c1", [6], "wrap"), ("c1", [-9], "clip"), ("s1", [0, 2], "raise"), ("e1", [0], "raise"),
                                 ("e1", [], "raise"), ("bc", [0], "raise")):
            ns("put", spec, lname, [_none_op(spec, lname)], lambda a, i=idx, m=mode: np.put(a, i, None, mode=m),
               indices=idx, mode=mode)

        # np.putmask / np.place: the mask's size is checked first, then None is converted whatever the mask holds.
        for lname, mask in (("c1", [True, False, True, False, False, True]), ("c1", [False] * 6), ("c1", [True, False, True]),
                            ("c2", [[True, False, False, True], [False] * 4, [True] * 4]), ("s1", [True] * 6),
                            ("bc", [True] * 12)):
            ns("putmask", spec, lname, [_none_op(spec, lname)], lambda a, m=mask: np.putmask(a, np.array(m), None), mask=mask)
            if not isinstance(spec, TSpec):
                ns("place", spec, lname, [_none_op(spec, lname)], lambda a, m=mask: np.place(a, np.array(m), None), mask=mask)

        # np.insert: a scalar position is bounds-checked before the values are converted; a sequence's values are
        # converted by the fancy assignment (an empty sequence too) after its positions are checked.
        for lname, obj, axis in (("c1", {"int": 1}, None), ("c1", {"int": -1}, None), ("c1", {"int": 10}, None),
                                 ("c1", {"list": [1, 3]}, None), ("c1", {"list": []}, None), ("c1", {"list": [2]}, None),
                                 ("c1", {"slice": [1, 3, None]}, None), ("c1", {"arr": [1, 2]}, None),
                                 ("c1", {"mask": [True, False, True, False, False, True]}, None),
                                 ("c2", {"int": 1}, 0), ("c2", {"int": 1}, 1), ("c2", {"int": 2}, None)):
            pyobj = (obj["int"] if "int" in obj else list(obj["list"]) if "list" in obj else slice(*obj["slice"])
                     if "slice" in obj else np.array(obj["arr"], dtype=np.int64) if "arr" in obj else np.array(obj["mask"]))
            ns("insert", spec, lname, [_none_op(spec, lname)],
               lambda a, o=pyobj, ax=axis: np.insert(a, o, None, axis=ax), mutate=None, obj=obj, axis=axis)

        # The flat iterator: an integer index rewords any failure; a slice or an index list keeps the setitem's own error.
        for lname, kind, idx in (("c1", "int", 1), ("c1", "int", -1), ("c1", "int", 6), ("r2", "int", 1),
                                 ("c1", "slice", "1:3"), ("c1", "slice", "::2"), ("c1", "slice", "4:4"),
                                 ("c1", "fancy", [1, 3]), ("c1", "fancy", [6]), ("c1", "fancy", []), ("bc", "int", 0)):
            if kind == "slice":
                pyk = slice(*[None if s == "" else int(s) for s in idx.split(":")])
            else:
                pyk = idx
            ns("flat_" + kind, spec, lname, [_none_op(spec, lname)],
               lambda a, k=pyk: a.flat.__setitem__(k, None), index=idx)

        # np.fill_diagonal: `a.flat[:end:step] = None` (a slice: the setitem's own errors), even over an empty diagonal.
        for shape, wrap in (((3, 3), False), ((3, 4), False), ((5, 2), True), ((5, 2), False), ((2, 2, 2), False),
                            ((0, 0), False)):
            op = _none_shaped(spec, shape)
            # An empty target is compared by its outcome alone: the replay rebuilds an empty numeric operand as an
            # owning array, where a NumPy reshape of an empty base is a view.
            ns("fill_diagonal", spec, "x".join(map(str, shape)), [op],
               lambda a, w=wrap: np.fill_diagonal(a, None, wrap=w), mutate=None if 0 in shape else 0, wrap=wrap)

        # np.put_along_axis: the fancy assignment converts None before the indices' bounds are checked.
        for shape, idx, axis in (((3, 4), [[0], [1], [2]], 1), ((4,), [1], 0), ((2, 2), [0, 3], None),
                                 ((4,), [10], 0)):
            op = _none_shaped(spec, shape)
            ns("put_along_axis", spec, "x".join(map(str, shape)), [op],
               lambda a, i=idx, ax=axis: np.put_along_axis(a, np.array(i), None, axis=ax), indices=idx, axis=axis)

        # np.asanyarray / np.asarray(None, dtype): the 0-d setitem value; np.array of a list holding None.
        ns("asanyarray", spec, "-", [], lambda dt=dt: np.asanyarray(None, dtype=dt), mutate=None, dtype=senc(spec))
        ns("asarray", spec, "-", [], lambda dt=dt: np.asarray(None, dtype=dt), mutate=None, dtype=senc(spec))
        for values in ([None, 1], [[None, 1], [2, None]], [1, None]):
            ns("array_list", spec, "-", [], lambda v=values, dt=dt: np.array(v, dtype=dt), mutate=None,
               values=enc(values), dtype=senc(spec))

    # ndarray.setfield(None, dtype, offset): None is packed into the FIELD's dtype (NumSharp refuses setfield over
    # StringDType for now: the whole-surface package P1.7f).
    for spec in num + ["<U4", "|S2"]:
        ns("setfield", spec, "c1", [_none_op(spec, "c1")], lambda a, s=spec: a.setfield(None, np.dtype(s), 0),
           dtype=spec, offset=0)
    for spec, field, offset in (("f8", "f4", 0), ("f8", "i4", 4), ("i8", "?", 0), ("c16", "f8", 8), ("c16", "c16", 0),
                                ("u8", "f2", 2)):
        ns("setfield", spec, f"c1/{field}@{offset}", [_none_op(spec, "c1")],
           lambda a, f=field, o=offset: a.setfield(None, np.dtype(f), o), dtype=field, offset=offset)

    # Mixed lists: the elements are packed in C order, the first failure wins.
    for spec, values in (("i1", [1e300, None]), ("i1", [None, 1e300]), ("i1", ["x", None]), ("i1", [None, "x"]),
                         ("f8", [None, "1.5"]), ("?", [None, "x", 0])):
        ns("array_list", spec, "mixed", [], lambda v=values, s=spec: np.array(v, dtype=s), mutate=None,
           values=enc(values), dtype=spec)


# ----------------------------------------------------------------------------------------------------------------------
# Section tio: text I/O over StringDType (strings phase 1, P1.7c)
# ----------------------------------------------------------------------------------------------------------------------
#
# np.loadtxt reads every field of a StringDType through npy_to_generic: PyArray_Pack of the field's str, so a field equal
# to a string na_object is an NA element and a converter's value meets stringdtype_setitem (the NA object first, then
# as_pystring's coerce rule; a 0-d array is CAST into the descriptor, so coerce does not apply to it). np.savetxt formats
# a StringDType row as Python objects (getitem's str, or the na_object itself), ndarray.tofile refuses the binary mode and
# formats each element as a Python object in text mode, and np.fromfile / np.fromstring / np.frombuffer refuse the dtype.
# The section also replays the reader's options for numeric and fixed-width dtypes: np.loadtxt's tokenizer, sources and
# converters are one port for every dtype.
#
# Sources (``params.source``): "lines" (a Python list of str), "reader" (io.StringIO, iterated line by line), "stream"
# (io.BytesIO: bytes lines decoded one by one, Latin-1 by default) and "file" (a temporary file read by NAME: universal
# newlines, decoded with the given encoding; always an explicit encoding here, because NumPy's default is the locale's,
# which NumSharp pins to UTF-8). np.savetxt writes to "writer" (io.StringIO) or "stream" (io.BytesIO, Latin-1 by
# default), never to a file name, whose newline translation is the host's. ndarray.tofile writes ASCII-only arrays:
# NumPy 2.4.2 crashes (an access violation) on the error path of a non-ASCII StringDType element in text mode.

# Texts for ndarray.tofile: ASCII only (see above), across the cell kinds (short, medium, long) and the reprs that differ
# from the text (quotes, NUL, tab).
T_ASCII_POOL = ["a", "m" * 16, "", "bc", "def", "q" * 20, "x y", "n" * 15, "p" * 255, "a\x00b", "r" * 256, "it's",
                "s" * 300, "\t", "~", "zz"]

# The converter menu: both sides map a name to the same computation (the C# twin is TioConverter in
# StringOracleOps.TextIO.cs). Each name covers one kind of value a converter may return.
_TIO_CONVERTERS = {
    "upper": lambda s: s.upper(),
    "len": lambda s: len(s),
    "none": lambda s: None,
    "nan": lambda s: float("nan"),
    "na": lambda s: "NA",
    "int": lambda s: int(s),
    "float3": lambda s: float(s) / 3,
    "bool": lambda s: s == "a",
    "complex": lambda s: complex(1, 2),
    "utf8": lambda s: s.encode("utf-8"),
    "badbytes": lambda s: b"\x80",
    "list": lambda s: [s, s],
    "tuple": lambda s: (s,),
    "dict": lambda s: {s: 1},
    "arr0": lambda s: np.array(5),
    "arr0f": lambda s: np.array(1.5),
    "arr1": lambda s: np.array([5, 6]),
    "raise": lambda s: 1 // 0,
    "surrogate": lambda s: "\ud800",
    "same": lambda s: s,
    "half": lambda s: np.float16(0.5),
}


def _encodes(text, name):
    """Whether ``text`` encodes strictly under the codec ``name``.

    :param text: the text.
    :param name: a codec name.
    :returns: False for an unencodable text, an unknown name or a codec that raises on every call.
    """
    try:
        text.encode(name)
        return True
    except (UnicodeError, LookupError):
        return False


def _tio_conv(conv):
    """The NumPy ``converters`` argument of an encoded request.

    :param conv: None, a converter name, ``{"<col>": name}`` (a dict; a name of ``None`` is a non-callable value), or
        ``"notcallable"`` (``converters=5``: neither a dict nor a callable).
    :returns: what ``np.loadtxt``'s ``converters`` receives.
    """
    if conv is None:
        return None
    if conv == "notcallable":
        return 5
    if isinstance(conv, str):
        return _TIO_CONVERTERS[conv]
    return {int(k): (None if v is None else _TIO_CONVERTERS[v]) for k, v in conv.items()}


def _tio_source(source, data, tmpdir):
    """The NumPy input of one loadtxt case.

    :param source: "lines", "reader", "stream" or "file".
    :param data: a list of str (lines), a str (reader) or bytes (stream / file).
    :param tmpdir: the directory of the file sources.
    :returns: the object ``np.loadtxt`` reads.
    """
    import io
    import os
    if source == "lines":
        return list(data)
    if source == "reader":
        return io.StringIO(data)
    if source == "stream":
        return io.BytesIO(data)
    path = os.path.join(tmpdir, f"tio_{len(CASES)}.txt")
    with open(path, "wb") as f:
        f.write(data)
    return path


class _TioRepeat:
    """A stream / file source of ``head + unit * count + tail`` bytes, recorded as that recipe instead of the bytes (the
    large files that put a decode error into a later TextIOWrapper chunk would otherwise be tens of kilobytes each).
    """

    def __init__(self, unit, count, tail, head=b""):
        """Records the recipe.

        :param unit: the repeated bytes.
        :param count: how many times.
        :param tail: the bytes after them.
        :param head: the bytes before them.
        """
        self.unit, self.count, self.tail, self.head = unit, count, tail, head

    def bytes(self):
        """The source's bytes.

        :returns: ``head + unit * count + tail``.
        """
        return self.head + self.unit * self.count + self.tail

    def enc(self):
        """The recipe as JSON.

        :returns: ``{"repeat": [unit hex, count, tail hex]}``, with the head's hex as a fourth item when there is one.
        """
        recipe = [self.unit.hex(), self.count, self.tail.hex()]
        if self.head:
            recipe.append(self.head.hex())
        return {"repeat": recipe}


def _tio_load(source, data, dtype, tmpdir, *, result="array", **kw):
    """Records one ``np.loadtxt`` case (op ``t_loadtxt``).

    :param source: the source kind (``_tio_source``).
    :param data: the source's data.
    :param dtype: a dtype string or a ``TSpec``.
    :param tmpdir: the directory of the file sources.
    :param result: "array", or "cells" for a StringDType result's raw cells (their arena offsets show the order NumPy
        packed the strings in).
    :param kw: loadtxt's other arguments: comments / delimiter / quotechar (a str, a list of str or None), skiprows,
        usecols (an int or a list), unpack, ndmin, max_rows, encoding, and conv (a ``_tio_conv`` request).
    """
    if isinstance(data, _TioRepeat):
        encoded, data = data.enc(), data.bytes()
    else:
        encoded = enc(list(data)) if source == "lines" else enc(data)
    params = {"source": source, "data": encoded, "dtype": senc(dtype), "result": result}
    call_kw = {}
    for name in ("comments", "delimiter", "quotechar"):
        if name in kw:
            params[name] = enc(kw[name])
            call_kw[name] = kw[name]
    for name in ("skiprows", "usecols", "unpack", "ndmin", "max_rows", "encoding"):
        if name in kw:
            params[name] = kw[name]
            call_kw[name] = kw[name]
    if "conv" in kw:
        params["conv"] = kw["conv"]
        call_kw["converters"] = _tio_conv(kw["conv"])

    def call():
        with warnings.catch_warnings():
            # loadtxt's "input contained no data" / empty-line warnings: NumSharp has no warnings channel.
            warnings.simplefilter("ignore")
            arr = np.loadtxt(_tio_source(source, data, tmpdir), dtype=sbuild(dtype), **call_kw)
        return _TCells(arr.tobytes()) if result == "cells" else arr

    add("tio", "t_loadtxt", params=params, call=call)


def _tio_write(arr_pair, *, target, written=False, tmpdir=None, **kw):
    """Records one ``np.savetxt`` case: the text (writer) or the bytes (stream, file) written, or NumPy's error.

    :param arr_pair: the (base, view) operand.
    :param target: "writer" (io.StringIO), "stream" (io.BytesIO) or "file" (a fresh file name: NumPy's TextIOWrapper,
        whose newline translation makes a case platform-dependent unless nothing written holds a '\\n').
    :param written: True for op ``t_savetxt_written``: the output is recorded even when NumPy raised (what reached the
        target before the error); False for ``t_savetxt``: the output, or the error.
    :param tmpdir: the directory of a file target.
    :param kw: savetxt's other arguments (fmt as a str or a list of str, delimiter, newline, header, footer, comments,
        encoding).
    """
    import io
    import os
    params = {"target": target}
    for name in ("fmt", "delimiter", "newline", "header", "footer", "comments"):
        if name in kw:
            params[name] = enc(kw[name])
    if "encoding" in kw:
        params["encoding"] = kw["encoding"]

    def call(a):
        if target == "file":
            path = os.path.join(tmpdir, f"tio_save_{len(CASES)}.txt")
            try:
                np.savetxt(path, a, **kw)
            except Exception:  # noqa: BLE001 -- the _written form records the file NumPy left behind
                if not written:
                    raise
            with open(path, "rb") as f:
                return f.read()
        fh = io.StringIO() if target == "writer" else io.BytesIO()
        try:
            np.savetxt(fh, a, **kw)
        except Exception:  # noqa: BLE001 -- the _written form records what was written before the error
            if not written:
                raise
        return fh.getvalue()

    add("tio", "t_savetxt_written" if written else "t_savetxt", params=params, operands=[arr_pair], call=call)


def _tio_tofile(arr_pair, tmpdir, *, sep, fmt, written=False):
    """Records one ``ndarray.tofile`` case: the file's bytes, or NumPy's error.

    :param arr_pair: the (base, view) operand.
    :param tmpdir: the output directory (a fresh file per case).
    :param sep: the separator ("" for the binary mode).
    :param fmt: the format.
    :param written: True for op ``t_tofile_written``: the file's bytes are recorded even when NumPy raised (NumPy
        creates or truncates the file before it writes).
    """
    import os

    def call(a):
        path = os.path.join(tmpdir, f"tio_out_{len(CASES)}.bin")
        try:
            a.tofile(path, sep=sep, format=fmt)
        except Exception:  # noqa: BLE001 -- the _written form records the file NumPy left behind
            if not written:
                raise
        with open(path, "rb") as f:
            return f.read()

    add("tio", "t_tofile_written" if written else "t_tofile", params={"sep": enc(sep), "format": enc(fmt)},
        operands=[arr_pair], call=call)


def _tio_fromfile(data, spec, tmpdir, *, sep, count, offset):
    """Records one ``np.fromfile`` case over a temporary file holding ``data`` (op ``t_fromfile``).

    :param data: the file's bytes.
    :param spec: the descriptor request.
    :param tmpdir: the directory.
    :param sep: the separator ("" for binary).
    :param count: the item count.
    :param offset: the byte offset.
    """
    import os

    def call():
        path = os.path.join(tmpdir, f"tio_in_{len(CASES)}.bin")
        with open(path, "wb") as f:
            f.write(data)
        return np.fromfile(path, dtype=sbuild(spec), count=count, sep=sep, offset=offset)

    add("tio", "t_fromfile", params={"data": enc(data), "dtype": senc(spec), "sep": enc(sep), "count": count,
                                     "offset": offset}, call=call)


def gen_tio():
    """Section tio (strings phase 1, P1.7c): text I/O over NEP 55 StringDType.

    * ``np.loadtxt`` with every option — the four sources, delimiters, comments (one, several, multi-character),
      quoting (fields across lines included), skiprows, usecols, unpack, ndmin, max_rows, encodings (the stream default,
      'bytes', undecodable bytes, an unknown codec) and converters of every value kind (the NA objects, coerce=False,
      bytes, 0-d and N-d arrays, lists, tuples, a raising converter) under every descriptor — its result's layout and
      its cells (NumPy's packing order), and its errors in NumPy's order and texts;
    * the same reader for numeric and fixed-width dtypes;
    * ``np.savetxt`` of StringDType arrays: the NA element, Python objects' %-conversions, the Mismatch error, the
      Latin-1 default of a byte stream, an unknown codec and what reached the target before an error;
    * ``ndarray.tofile``: the binary refusal and the text mode's Python objects;
    * the refusals of ``np.fromfile`` / ``np.fromstring`` / ``np.frombuffer``;
    * loadtxt's comments as Python objects (non-iterables, None and non-str elements, ``delimiter in comments``);
    * the NA element as every kind of Python object (ints, a big int, floats, a complex, np.float16, None, a str) under
      every %-conversion, through savetxt and tofile;
    * the % formats of savetxt (NumPy scalars) and tofile (getitem's Python objects) over every NumPy dtype: CPython's
      parse and its errors, '*' arguments, NumPy scalar reprs, and tofile's checks of sep and format;
    * Python's codecs: names resolved as ``codecs.lookup`` resolves them, every codec family over a file and a byte
      stream, decode errors in later TextIOWrapper chunks, and savetxt to a byte stream and to a file name (a file
      target writes no '\\n', so the corpus does not depend on the platform's line separator).

    Every case is strict.
    """
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="numsharp_tio_")
    specs = [TD, TDN, TDNAN, TDS, TDC, TDNC]
    long_a, long_b = "a" * 20, "b" * 300
    datasets = {
        "basic": ["ab cd", "ef gh"],
        "na": ["ab NA", "NA cd", "None nan"],
        "long": [f"{long_a} x", f"y {long_b}", "z w"],
        "uni": ["é ü", "中 文", "😀 x"],
        "single": ["a", "bb", "ccc"],
        "one": ["only"],
        "blank": ["a b", "", "   ", "c d"],
        "comment": ["a b # c", "# full line", "c d"],
        "ws": [" a  b ", "\tc d\t"],
        "crlf": ["a b\r\n", "c d\r\n"],
        "empty": [],
        "allcomment": ["# x", "# y"],
    }
    # ---- every descriptor over the datasets, from each source ----------------------------------------------------
    for spec in specs:
        for lines in datasets.values():
            _tio_load("lines", lines, spec, tmpdir)
            _tio_load("reader", "\n".join(lines), spec, tmpdir)
            if all(s.isascii() for s in lines):
                _tio_load("stream", "\n".join(lines).encode("ascii"), spec, tmpdir)
            _tio_load("file", "\n".join(lines).encode("utf-8"), spec, tmpdir, encoding="utf-8")
            if spec in (TD, TDS):
                _tio_load("lines", lines, spec, tmpdir, result="cells")
    # ---- the result's cells: short / medium / long strings, NA cells, row-major packing under usecols --------------
    cell_sets = [["ab " + "m" * 16, "x" * 15 + " y", "z" * 256 + " q"], ["NA a", "b NA"], ["", "a"],
                 ["é" * 8 + " ü", "中" * 6 + " 文"], ['"" a', 'b ""']]
    for spec in (TD, TDN, TDS):
        for lines in cell_sets:
            for usecols in (None, [1, 0], [-1]):
                kw = {} if usecols is None else {"usecols": usecols}
                _tio_load("lines", lines, spec, tmpdir, result="cells", quotechar='"', **kw)
    # ---- options ---------------------------------------------------------------------------------------------------
    for spec in (TD, TDS):
        csv = ["a,b,c", "d,,f", ",h,"]
        for delimiter in (",", ";", " ", None):
            _tio_load("lines", csv, spec, tmpdir, delimiter=delimiter)
        for usecols in ([0], [2, 0], [-1], [0, 0], [5], 1, [], [-4]):
            _tio_load("lines", csv, spec, tmpdir, delimiter=",", usecols=usecols)
        for skiprows in (0, 1, 2, 5):
            for max_rows in (None, 0, 1, 2):
                _tio_load("lines", csv, spec, tmpdir, delimiter=",", skiprows=skiprows, max_rows=max_rows)
        _tio_load("lines", ["a,b", "", "c,d", "e,f"], spec, tmpdir, delimiter=",", max_rows=2)
        for unpack in (False, True):
            for ndmin in (0, 1, 2):
                for lines in (csv, ["a,b"], ["a"], ["a", "b"], []):
                    _tio_load("lines", lines, spec, tmpdir, delimiter=",", unpack=unpack, ndmin=ndmin)
        for comments in ("#", "%", None, ["#", "%"], "//", [], ["//", "#"], ["#"]):
            lines = ["a b # c", "d e % f", "// g h", "i j"]
            _tio_load("lines", lines, spec, tmpdir, comments=comments)
            _tio_load("reader", "\n".join(lines), spec, tmpdir, comments=comments)
            _tio_load("file", "\n".join(lines).encode("utf-8"), spec, tmpdir, comments=comments, encoding="utf-8")
        quoted = ['"a b" c', '"x""y" z', '"" q', "'s t' u"]
        for quotechar in ('"', "'", None):
            _tio_load("lines", quoted, spec, tmpdir, quotechar=quotechar)
            _tio_load("lines", ['"a,b",c', 'd,"e"'], spec, tmpdir, quotechar=quotechar, delimiter=",")
        # A quoted field spans lines (list elements, a text stream's lines, a file's universal newlines).
        _tio_load("lines", ['"a', 'b" c'], spec, tmpdir, quotechar='"')
        _tio_load("reader", '"a\nb" c\nd e', spec, tmpdir, quotechar='"')
        _tio_load("file", b'"a\r\nb" c\r\nd e\r\n', spec, tmpdir, quotechar='"', encoding="utf-8")
        _tio_load("file", b"a b\rc d\re f", spec, tmpdir, encoding="utf-8")
        _tio_load("stream", b'"a\nb" c\nd e\n', spec, tmpdir, quotechar='"')
        # Whitespace before a quote: data under a delimiter, skipped in whitespace mode.
        _tio_load("lines", [' "a b",c'], spec, tmpdir, quotechar='"', delimiter=",")
        _tio_load("lines", ['  "a b" c '], spec, tmpdir, quotechar='"')
        _tio_load("lines", ['"a" b '], spec, tmpdir, quotechar='"')
        _tio_load("lines", ['"a"b c'], spec, tmpdir, quotechar='"')
        _tio_load("lines", ['a"b" c'], spec, tmpdir, quotechar='"')
    # ---- encodings ---------------------------------------------------------------------------------------------------
    for spec in (TD, TDN):
        data = "é x\nü 中\n".encode("utf-8")
        for encoding in (None, "utf-8", "latin-1", "ascii", "bytes"):
            _tio_load("stream", data, spec, tmpdir, **({} if encoding is None else {"encoding": encoding}))
            # A file NAME with 'bytes' is read with the locale's encoding (host-dependent): not recorded.
            if encoding not in (None, "bytes"):
                _tio_load("file", data, spec, tmpdir, encoding=encoding)
        _tio_load("stream", "é x".encode("latin-1"), spec, tmpdir, encoding="latin-1")
        _tio_load("stream", b"\xef\xbb\xbfa b", spec, tmpdir, encoding="utf-8")
        _tio_load("file", b"\xef\xbb\xbfa b", spec, tmpdir, encoding="utf-8")
        _tio_load("stream", b"a b\r\nc d\r\n", spec, tmpdir)
        _tio_load("stream", b"a\xff b\n", spec, tmpdir, encoding="utf-8")
        _tio_load("file", b"ok\na\xff b\n", spec, tmpdir, encoding="utf-8")
        _tio_load("stream", "a b\nc 中\n".encode("utf-8"), spec, tmpdir, encoding="ascii")
        # An unknown codec: a byte stream meets it at its first line (none: no error), a file name at the open.
        _tio_load("stream", b"a b\n", spec, tmpdir, encoding="nosuchcodec")
        _tio_load("stream", b"", spec, tmpdir, encoding="nosuchcodec")
        _tio_load("file", b"a b\n", spec, tmpdir, encoding="nosuchcodec")
        _tio_load("lines", ["a b"], spec, tmpdir, encoding="nosuchcodec")
        _tio_load("reader", "a b\n", spec, tmpdir, encoding="nosuchcodec")
    # ---- converters of every value kind under every descriptor -----------------------------------------------------
    for spec in specs:
        for name in _TIO_CONVERTERS:
            data = ["1 2", "3 4"] if name in ("int", "float3") else ["a 1", "b NA"]
            _tio_load("lines", data, spec, tmpdir, conv=name)
        for conv in ({"0": "upper"}, {"-1": "len"}, {"1": "none", "0": "na"}, {"5": "upper"}, {"-3": "upper"},
                     {"0": None}, "notcallable", {"0": "upper", "-2": "len"}):
            _tio_load("lines", ["a b", "c d"], spec, tmpdir, conv=conv)
            _tio_load("lines", ["a b", "c d"], spec, tmpdir, conv=conv, usecols=[1])
            _tio_load("lines", ["a b", "c d"], spec, tmpdir, conv=conv, usecols=[-1, 0])
        # Converters are created at the first data row: an empty input never meets a bad one.
        _tio_load("lines", [], spec, tmpdir, conv={"5": "upper"})
        _tio_load("lines", [], spec, tmpdir, conv="notcallable")
        # encoding='bytes': a converter receives the field's Latin-1 bytes in NumPy (a C# converter the same characters),
        # so only converters that do not return their input are recorded; a field past U+00FF fails the encode.
        for name in ("len", "none", "na"):
            _tio_load("lines", ["a b"], spec, tmpdir, conv=name, encoding="bytes")
        _tio_load("lines", ["中 b"], spec, tmpdir, conv="len", encoding="bytes")
        _tio_load("lines", ["中 b"], spec, tmpdir, encoding="bytes")
    # ---- errors ----------------------------------------------------------------------------------------------------
    for spec in (TD, TDC):
        _tio_load("lines", ["a b", "c"], spec, tmpdir)
        _tio_load("lines", ["a b", "c d e"], spec, tmpdir)
        _tio_load("lines", ["a b", "c"], spec, tmpdir, usecols=[1])
        _tio_load("lines", ["a\nb c"], spec, tmpdir)
        _tio_load("lines", ["a\rb"], spec, tmpdir)
        _tio_load("reader", "a b\rc d\n", spec, tmpdir)
        _tio_load("lines", ["\ud800 x"], spec, tmpdir)
        _tio_load("lines", ["a b"], spec, tmpdir, ndmin=3)
        _tio_load("lines", ["a b"], spec, tmpdir, skiprows=-1)
        _tio_load("lines", ["a b"], spec, tmpdir, max_rows=-1)
        _tio_load("lines", ["a b"], spec, tmpdir, comments="")
        _tio_load("lines", ["a b"], spec, tmpdir, comments=["#", ""])
        _tio_load("lines", ["a b"], spec, tmpdir, comments=["#", ","], delimiter=",")
        _tio_load("lines", ["a b"], spec, tmpdir, comments="//", quotechar='"')
        for delimiter, quotechar, comments in ((", ", None, "#"), ("\n", None, "#"), (",", ",", "#"), (",", None, ","),
                                               (None, " ", "#"), (None, None, " "), (",", "\r", "#"), ("ab", None, "#"),
                                               (",", '"', '"'), (None, "\t", "#"), ("", None, "#"), (None, "", "#")):
            _tio_load("lines", ["a b"], spec, tmpdir, delimiter=delimiter, quotechar=quotechar, comments=comments)
    # ---- the same reader for the numeric and fixed-width dtypes -----------------------------------------------------
    for dt in ("float64", "int32", "<U3", "|S2", "<U", "|S", "bool", "complex128"):
        for lines in (["1 2", "3 4"], ["1,2,", ",3,4"], ["1 2 # x", "", "3 4"], ['"1" 2'], ["é 1"], ["中 1"],
                      ["abc defg", "h i"], []):
            for kw in ({}, {"delimiter": ","}, {"quotechar": '"'}, {"usecols": [1]}, {"ndmin": 2}, {"unpack": True}):
                _tio_load("lines", lines, dt, tmpdir, **kw)
        _tio_load("stream", "1 2\né 3\n".encode("latin-1"), dt, tmpdir)
        _tio_load("file", b"1 2\r\n3 4\r\n", dt, tmpdir, encoding="utf-8")
        for name in ("len", "none", "raise", "same"):
            _tio_load("lines", ["1 2"], dt, tmpdir, conv=name)
    # ---- np.savetxt of StringDType arrays ------------------------------------------------------------------------------
    for spec in (TD, TDN, TDNAN, TDS, TDC):
        for lay in ("c1", "c2", "f2", "r1", "e1", "z0"):
            op = t_operand(spec, lay)
            for fmt in ("%s", "%r", "%a", "%5s", "%-5s|", "%.1s", "%c", "%d", "%.18e", "%x"):
                _tio_write(op, target="writer", fmt=fmt)
            _tio_write(op, target="writer")
            _tio_write(op, target="writer", fmt="%s", delimiter=";", newline="|", header="h\nh2", footer="f",
                       comments="## ")
            _tio_write(op, target="writer", fmt="%s", encoding="nosuchcodec")
            _tio_write(op, target="stream", fmt="%s")
            _tio_write(op, target="stream", fmt="%s", encoding="utf-8")
            ncol = op[1].shape[1] if op[1].ndim == 2 else 1
            _tio_write(op, target="writer", fmt=["%s", "%r", "%3s", "%-2s"][:ncol])
            _tio_write(op, target="writer", fmt=["%s", "%r", "%3s", "%-2s", "%s"][:ncol + 1])
            _tio_write(op, target="writer", fmt="|".join(["%s", "%r", "%3s", "%-2s"][:ncol]))
            _tio_write(op, target="writer", fmt="|".join(["%s", "%r", "%3s", "%-2s", "%s"][:ncol + 1]))
    for spec in (TD, TDN):
        base = np.array(["ab", "é", "中", "x"], dtype=spec.build())
        pair = (base, base)
        for encoding in (None, "utf-8", "latin-1", "ascii", "bytes", "nosuchcodec"):
            kw = {} if encoding is None else {"encoding": encoding}
            _tio_write(pair, target="stream", fmt="%s", header="H", **kw)
            _tio_write(pair, target="stream", written=True, fmt="%s", header="H", **kw)
        _tio_write(pair, target="stream", fmt="%s", header="h€")
        _tio_write(pair, target="stream", written=True, fmt="%s", header="h€")
        _tio_write(pair, target="stream", written=True, fmt="%s€")
        _tio_write(pair, target="writer", written=True, fmt="%d", header="H", footer="F")
        # An empty array writes no row: an unknown codec is never met without a header.
        e = np.array([], dtype=spec.build())
        _tio_write((e, e), target="stream", fmt="%s", encoding="nosuchcodec")
        _tio_write(t_plain(np.zeros((2, 0), dtype=spec.build())), target="writer")
    na_arr = np.array(["ab", None, "c", None], dtype=TDN.build())
    _tio_write((na_arr, na_arr), target="writer", written=True, fmt="%d", header="H")
    _tio_write((na_arr, na_arr), target="writer", fmt="%s")
    nan_arr = np.array(["1.5", float("nan"), "2"], dtype=TDNAN.build())
    _tio_write((nan_arr, nan_arr), target="writer", written=True, fmt="%d")
    # ---- ndarray.tofile --------------------------------------------------------------------------------------------
    for spec in (TD, TDN, TDNAN, TDS):
        for lay in ("c1", "c2", "f2", "r1", "e1", "z0"):
            op = t_operand(spec, lay, pool=T_ASCII_POOL)
            for sep, fmt in (("", "%s"), (",", "%s"), (" ", "%r"), ("\n", "%5s"), (",", ""), (";", "%-3s|"),
                             (",", "%d"), (",", "%c")):
                _tio_tofile(op, tmpdir, sep=sep, fmt=fmt)
                _tio_tofile(op, tmpdir, sep=sep, fmt=fmt, written=True)
    # ---- the refusals of the readers ------------------------------------------------------------------------------------
    for spec in (TD, TDN, TDC):
        for sep, count, offset in (("", -1, 0), (" ", -1, 0), (" ", 0, 0), (" ", -1, 1), ("", -1, 1), (",", 2, 0)):
            _tio_fromfile(b"a b c", spec, tmpdir, sep=sep, count=count, offset=offset)
        for text, sep in (("a b", " "), ("a b", None), ("", " "), ("", None), ("a,b", ",")):
            add("tio", "t_fromstring", params={"text": enc(text), "dtype": senc(spec), "sep": enc(sep)},
                call=lambda t=text, s=sep, d=spec: np.fromstring(t, dtype=d.build(), sep=s) if s is not None
                else np.fromstring(t, dtype=d.build()))
        # tcreate's t_frombuffer op (its params: the buffer as hex, the descriptor request), over more buffers and
        # descriptors.
        for data in (b"", b"a" * 16, b"a" * 32):
            add("tio", "t_frombuffer", params={"buffer": data.hex(), "dtype": senc(spec)},
                call=lambda b=data, d=spec: np.frombuffer(b, dtype=d.build()))
    # ---- loadtxt's comments as Python objects -------------------------------------------------------------------------
    # loadtxt's comprehension over the argument (a non-iterable raises "'int' object is not iterable" before _read's
    # checks), _read's checks over the tuple ("" in comment, the one-character fast path, `delimiter in comments` with
    # Python's `in`: None is in it when an element is None), and _preprocess_comments' `line.split(c, 1)[0]` per line: a
    # None element keeps the first whitespace-separated word (IndexError on an empty or blank line, skipped lines
    # included), any other non-str raises str.split's TypeError at the first line read (never for an input without lines).
    for comments in (5, 2.5, True, [5], [5, "#"], ["#", 5], [2.5], [True], ["#", None], [None], [None, "%"],
                     [None, None], [b"#"], [b"#", "%"], b"#", b"%%", ("#",), ("#", "%"), [], ["#", "\n"], ["##"],
                     ["\n"], "\n"):
        for lines in (["a b # c", "d e"], ["  x y", ""], ["a,b # c"], ["1 2 % x", "3 4"], []):
            for spec in (TD, "float64"):
                _tio_load("lines", lines, spec, tmpdir, comments=comments)
    for kw in ({"delimiter": ","}, {"quotechar": '"'}, {"delimiter": "ab"}, {"max_rows": 1}, {"skiprows": 1},
               {"ndmin": 3}, {"usecols": [0]}, {"delimiter": ",", "skiprows": 1}):
        for comments in ([None], [None, 5], [5, ","], [5], 5, [None, ","]):
            _tio_load("lines", ["a b,c d", " x y,z", "u v w\x1cq,r"], TD, tmpdir, comments=comments, **kw)
            _tio_load("lines", ["", "a b,c d"], TD, tmpdir, comments=comments, **kw)
    for source, data in (("reader", "a b\n\n c d\n"), ("reader", "a b\nc d"), ("stream", b"a b\n\n c d\n"),
                         ("stream", b"a b\nc d"), ("file", b"a b\r\n\r\n c d\r\n"), ("file", b"a b\nc d"),
                         ("file", b"a b\r c d\r")):
        for comments in ([None], [None, "%"], ["%", "#"], [5], ["b", None]):
            _tio_load(source, data, TD, tmpdir, comments=comments)
            _tio_load(source, data, TD, tmpdir, comments=comments, delimiter=",")
    for comments in (["#", "%"], "#", [None], ["#"]):
        _tio_load("lines", ["a", None], TD, tmpdir, comments=comments)
    # ---- the NA element as a Python object under every conversion ----------------------------------------------------
    # savetxt's `fmt % tuple(row)` and tofile's `format % obj` meet getitem's na_object itself: an int (a bool, a big
    # int, a code point for %c), a float (signed zero, NaN, infinity, past float32), a complex, an np.float16, None or a
    # str, each through every conversion with flags, width and precision. savetxt wraps a TypeError into its "Mismatch"
    # text and lets ValueError / OverflowError through; tofile raises them all as they are. NumPy's tofile writes its
    # text with PyUnicode_AsASCIIString and segfaults on a NULL result, so no tofile case formats a non-ASCII %c.
    na_objects = (7, -3, True, False, 2 ** 70, 2 ** 1030, 65, 0x10FFFF, 0x110000, -1, 0xD800, 2.5, -0.0, 1e300,
                  float("inf"), float("-inf"), float("nan"), complex(1, 2), np.float16(1.5), None, "NA")
    na_fmts = ("%d", "%i", "%u", "%x", "%X", "%#o", "%c", "%e", "%E", "%f", "%F", "%g", "%G", "%s", "%r", "%a",
               "%+08.2f", "%-6d|", "%.3x", "%5c", "% g", "%#x", "%.0e", "%10.4G")
    for na in na_objects:
        arr = np.array([na, na], dtype=TSpec(na=na).build())
        for fmt in na_fmts:
            _tio_write((arr, arr), target="writer", fmt=fmt)
        _tio_write((arr, arr), target="stream", fmt="%s")
        _tio_write((arr, arr), target="writer", fmt=["%s"])
        non_ascii_char = isinstance(na, int) and not isinstance(na, bool) and 128 <= na < 0x110000
        for fmt in ("%d", "%x", "%c", "%e", "%g", "%s", "%r", "%a", "%5.1f", ""):
            if non_ascii_char and fmt == "%c":
                continue
            _tio_tofile((arr, arr), tmpdir, sep=",", fmt=fmt)
    # ---- the % formats of savetxt and tofile over every dtype ----------------------------------------------------------
    # CPython 3.12's PyUnicode_Format, as NumPy calls it: np.savetxt formats tuple(row) of NUMPY SCALARS (a StringDType
    # row: Python objects; a complex row: np.float64 pairs), ndarray.tofile formats (item,) of the PYTHON OBJECT getitem
    # returns (a float for every float dtype, an int, a bool, a complex, a str / bytes). The matrix covers the parse —
    # flags, '*' widths and precisions (a Python int only: a NumPy scalar is "* wants int"), one length modifier, '%%',
    # '%(key)' without a mapping, an incomplete or unsupported spec and its code-point index, the argument taken before
    # the conversion is checked, a leftover argument — and every conversion against every element type: a NumPy scalar's
    # repr names its type, a TypeError quotes it, a NumPy bool is not an int where a Python bool is. tofile's sep and
    # format pass CPython's "s" converter (no NUL, no lone surrogate) before the file is opened. NumPy 2.4.2's tofile
    # crashes on non-ASCII text (it does not check PyUnicode_AsASCIIString's NULL), so no tofile case reaches one.
    pct_fmts = ["%y", "%5y", "%", "a%", "%5", "%.", "%-", "%(x)s", "%*d", "%.*f", "%ld", "%lld", "%hd", "%Ld", "%5%",
                "%-5%", "%é", "%5.2l", "%#", "%l", "%x", "%c", "%d", "%s", "%r", "%a", "%e", "%5.1f", "%%", "%o", "%X",
                "%#x", "%+d", "%-6.2s|", "%.*s", "%*s", "%10.3e", "%#o", "% d", "%+.2g", "%G", "%i", "%u", "%.0f",
                "%#.3x", "%05.1f", "%-8r|", "%.3r", "%c%c", "%d %d", "%%%d", "%\U0001F600", "\U0001F600%y", "%\x00",
                "% ", "%(", "%**d", "%.5.3f", "%-*d", "%0*d", "%*-d", "%9223372036854775808d", "%.2147483648d",
                "%.2147483647d", "%.2147483647e", "%.2147483647g", "%.2147483647s", "%.300f", "%.40g", "%#.20g",
                "%\x1f", "%\x1e", "%~", "%\x7f"]
    pct_arrays = [
        np.array([7, -2, 0]), np.array([65, 200], dtype=np.uint8), np.array([2 ** 64 - 1, 2 ** 63 + 1025], dtype=np.uint64),
        np.array([-128], dtype=np.int8), np.array([1.5, np.nan, -np.inf, -0.0]), np.array([0.1], dtype=np.float32),
        np.array([1.5, 65504], dtype=np.float16), np.array([True, False]), np.array([1 + 2j, complex(np.nan, 0)]),
        np.array(["ab", "é"]), np.array([b"ab", b"\x80"]), np.array(["ab", "é"], dtype=TD.build()),
        np.array([None, "ab"], dtype=TDN.build()), np.array([7, "ab"], dtype=TSpec(na=7).build()),
        np.array([[3, 7]]), np.array([[1.5, 2.5]]), np.array([[1, 2, 3]]), np.array([[3, 7]], dtype=np.uint16),
        np.array([[4, "ab"]], dtype=TSpec(na=4).build()), np.array([["5", "ab"]]),
        np.array([[-2 ** 63, -2 ** 63]], dtype=TSpec(na=-2 ** 63).build()), np.array([[-5, -5]], dtype=TSpec(na=-5).build()),
    ]

    def tofile_crashes(a, fmt):
        """Whether NumPy's tofile meets a non-ASCII element text (it crashes there) before it raises or finishes.

        :param a: the array.
        :param fmt: the format.
        :returns: True when an element's ``fmt % (item,)`` succeeds with non-ASCII text before any element raises.
        """
        for i in range(a.size):
            item = a.reshape(-1)[i].item() if a.dtype.kind != "T" else a.reshape(-1)[i]
            try:
                text = fmt % (item,) if fmt else str(item)
            except Exception:  # noqa: BLE001 -- NumPy raises at this element: no crash
                return False
            if not text.isascii():
                return True
        return False

    for a in pct_arrays:
        pair = t_plain(a) if a.dtype.kind == "T" else plain(a)
        for fmt in pct_fmts:
            _tio_write(pair, target="writer", fmt=fmt)
            if a.ndim == 1 and not tofile_crashes(a, fmt):
                _tio_tofile(pair, tmpdir, sep=",", fmt=fmt)
    for a in pct_arrays:
        if a.ndim == 2:
            pair = t_plain(a) if a.dtype.kind == "T" else plain(a)
            for fmt in (["%*s", ""], ["%*d", ""], ["%-*d|", ""], ["%s", "%*d"], ["%d", "%s"], ["%x", "%.*s"], ["%.*d", ""],
                        ["%*.*f", ""],
                        ["%d%%", ""], ["%r", "%a"], ["%c", "%s"]):
                if len(fmt) == a.shape[1]:
                    _tio_write(pair, target="writer", fmt=fmt)
            _tio_write(pair, target="writer", fmt="%d%%")
    # The "s" converter of tofile's sep and format: checked before the file is opened (sep first).
    small = plain(np.array([1, 2]))
    for sep, fmt in ((",", "%s\x00"), ("\x00", "%s"), (",", "\udc80%s"), ("\ud800", "%s"), ("\x00", "\ud800"),
                     ("", "%s\x00"), ("é", "%d"), ("\U0001F600", "%s"), (",", "%s\ud800\x00")):
        _tio_tofile(small, tmpdir, sep=sep, fmt=fmt)
    # ---- Python's codecs ----------------------------------------------------------------------------------------------
    # NumPy decodes a byte stream's lines with bytes.decode (a BOM codec reads a BOM per line, native little-endian without
    # one) and a file name through TextIOWrapper (the BOM codecs require a BOM; positions in a decode error count within
    # the chunk TextIOWrapper decodes, whose size read(16384) derives from the bytes-per-character ratio and a line
    # iteration fixes at 8192); it encodes a byte stream's writes one by one with str.encode (a BOM per write) and a file's
    # through TextIOWrapper's encoder (one BOM; 'utf-16' names itself so in the first write's error, 'utf-16-le' after).
    # Names resolve as codecs.lookup resolves them, after the C fast paths for one-shot calls (an iso8859_1 spelling
    # encodes with the 'latin-1' text). The codecs NumSharp refuses (the multi-byte CJK ones, utf-7, …) are not cases.
    codec_names = ["utf-8", "UTF8", "utf_8_sig", "utf-16", "utf-16-le", "UTF-16BE", "u16", "utf-32", "utf_32_le",
                   "utf-32-be", "latin-1", "L1", "iso8859_1", "iso-8859-1", "ascii", "646", "cp1252", "Windows-1252",
                   "iso8859_15", "koi8_r", "cp037", "mac_roman", "cp437", "undefined", "base64", "rot13", "nosuch", "",
                   "utf.8", "iso.8859.1", "locale", "cp65001"]
    texts = ["a b\né 中\n", "x\r\ny z\r", "€ 1\n", "\U0001F600 b\n", "", "p q"]
    for name in codec_names:
        try:
            codecs_info = codecs.lookup(name)
            text_codec = getattr(codecs_info, "_is_text_encoding", True) and name != "undefined"
        except LookupError:
            text_codec = False
        # 'locale' names the locale's encoding only for open(): a file then reads with it (the ANSI code page on the
        # Windows host that generates the corpus, UTF-8 in NumSharp), so only its byte-stream cases (an unknown name
        # there) are portable.
        sources = ("stream",) if name == "locale" else ("file", "stream")
        for text in texts:
            data = text.encode(name) if text_codec and _encodes(text, name) else text.encode("utf-8")
            for source in sources:
                _tio_load(source, data, TD, tmpdir, encoding=name)
            if "file" in sources:
                _tio_load("file", data, TD, tmpdir, encoding=name, comments=["#", "%"])
        for data in (b"\xff\xfe\x00\xdcx\x00", b"a\x00b", b"a\x81b\n", b"\x80 1\n", b"\xef\xbb", b"\xef\xbb\xbf\xff",
                     b"\xff", b"\x00\x00\x11\x00", b"\x00\xd8\x00\x00", b"\xfe\xff\x00a\x00\n", b"a\x00\x00\xd8", b"x y\n"):
            for source in sources:
                _tio_load(source, data, TD, tmpdir, encoding=name)
    # Decode errors in later chunks of a file: TextIOWrapper's chunk sizes decide the position.
    for name, unit, bad in (("utf-8", b"a", b"\xff"), ("utf-8", "é".encode(), b"\xff"), ("utf-8", "a\r\n".encode(), b"\xff"),
                            ("utf-16-le", b"a\x00", b"\x00\xdc"), ("utf-16-le", "中".encode("utf-16-le"), b"\x00\xdc"),
                            ("cp1252", b"a", b"\x81"), ("ascii", b"a", b"\x80"), ("utf-32-le", b"a\x00\x00\x00", b"\x00\x00\x11\x00"),
                            ("utf-8", b"\xe4\xb8\xad", b"\xe4\xb8")):
        for n in (8190, 8192, 10000, 16383, 16384, 20000, 40000):
            data = _TioRepeat(unit, n // len(unit), bad + b"\n")
            _tio_load("file", data, TD, tmpdir, encoding=name)
            _tio_load("file", data, TD, tmpdir, encoding=name, comments=["#", "%"])
    # A chunk that ends INSIDE a UTF-8 sequence (the first read(16384) chunk, or a line iteration's 8192): CPython's
    # incremental decoder carries a valid start to the next chunk and raises at once for an invalid one — an overlong or
    # surrogate second byte, a code point past U+10FFFF, a lead it never accepts — counting the position in that chunk;
    # an invalid byte earlier in the chunk is reported first. Every source ends in an error (a short corpus case).
    for head, cut, after in ((b"", b"\xe0\x80", b""), (b"", b"\xed\xa0", b""), (b"", b"\xf4\x90", b""),
                             (b"", b"\xf0\x80", b""), (b"", b"\xc0", b""), (b"", b"\xe4\xb8", b"\xad\xff"),
                             (b"", b"\xf0\x9f\x98", b"\x80\xff"), (b"", b"\xe4", b"\xb8\xad\xff"),
                             (b"\xff", b"\xe4\xb8", b"\xad"), (b"\xff", b"\xf0\x9f", b"\x98\x80")):
        for size in (8192, 16384):
            data = _TioRepeat(b"a", size - len(head) - len(cut), cut + after + b"\n", head=head)
            _tio_load("file", data, TD, tmpdir, encoding="utf-8")
            _tio_load("file", data, TD, tmpdir, encoding="utf-8", comments=["#", "%"])
    # A whole (non-final) first chunk of continuation bytes only, shorter than a UTF-8 sequence: no lead to carry.
    for data in (b"\x80\x80", b"\x80", b"\xbf\x80\x80", b"\x80\x80\x80\x80\n"):
        _tio_load("file", data, TD, tmpdir, encoding="utf-8")
    # Complex fields: exponents, the inf / nan words in either part, a missing parenthesis.
    for field in ("1e3+2e-1j", "1E+2-3E-2j", "inf+nanj", "-infj", "nanj", "-nan-infj", "(1+2j", "1e+j", "1e", "infinityj",
                  "(nan)", "INF-INFj", "1.5e-3j", "(-0-0j)"):
        _tio_load("lines", [field], "complex128", tmpdir)
    # savetxt to a byte stream (WriteWrap: str.encode per write) and to a file name (TextIOWrapper); nothing written holds
    # a '\n' in the file cases, so the corpus does not depend on the platform's line separator.
    t_ok = np.array(["ab", "é", "中", "\U0001F600"], dtype=TD.build())
    t_ascii = np.array(["ab", "cd"], dtype=TD.build())
    # A lone surrogate (only UCS-4 U can hold one: StringDType stores UTF-8) is what UTF-8 / 16 / 32 refuse to encode.
    t_bad = np.array(["ab", "x\ud800y"], dtype="<U3")
    # A run of unencodable characters holding a surrogate pair: one code point of the run, as CPython counts it.
    t_run = np.array(["a中\U0001F600b", "\U0001F600\U0001F600x"], dtype=TD.build())
    for name in ("cp1252", "cp437", "koi8_r", "latin-1", "ascii", "mac_roman"):
        _tio_write((t_run, t_run), target="stream", fmt="%s", encoding=name)
        _tio_write((t_run, t_run), target="file", tmpdir=tmpdir, fmt="%s", newline="|", encoding=name)
    for name in codec_names + ["utf-16", "bytes"]:
        for arr in (t_ok, t_ascii, t_bad):
            for written in (False, True):
                _tio_write((arr, arr), target="stream", written=written, fmt="%s", header="H", encoding=name)
                if name != "locale":
                    _tio_write((arr, arr), target="file", written=written, tmpdir=tmpdir, fmt="%s", newline="|",
                               header="H", footer="F", comments="#", encoding=name)
        if name == "locale":
            continue
        empty = np.array([], dtype=TD.build())
        _tio_write((empty, empty), target="file", tmpdir=tmpdir, fmt="%s", newline="|", encoding=name)
        # Two rows of no columns with newline="": two empty writes, the first of which a BOM codec encodes to its BOM.
        _tio_write(t_plain(np.zeros((2, 0), dtype=TD.build())), target="file", tmpdir=tmpdir, fmt="%s", newline="",
                   encoding=name)


# ----------------------------------------------------------------------------------------------------------------------
# strsort. Sorting text: np.sort / np.argsort / ndarray.sort / np.partition / np.argpartition (strings phase 2, P2.2)
# ----------------------------------------------------------------------------------------------------------------------

# Every spelling of the sort keywords NumPy accepts: the default, the four kind names, single letters of either case (NumPy
# parses the first letter only) and the stable keyword both ways.
_SORT_KWS = [{}, {"kind": "quicksort"}, {"kind": "heapsort"}, {"kind": "stable"}, {"kind": "mergesort"}, {"kind": "Q"},
             {"kind": "h"}, {"stable": True}, {"stable": False}]

# The keywords whose argsort answers differ over text: the default introsort, the heapsort kind (the same answer in
# NumPy 2.4.2) and the stable sort.
_SORT_KWS_CORE = [{}, {"kind": "heapsort"}, {"kind": "stable"}]

# NumPy's SMALL_QUICKSORT (npysort/npysort_common.h): a range of at most 16 elements is insertion-sorted.
_SMALL_QUICKSORT = 15


def _sort_params(kw, **extra):
    """The params of a sort case: the three sort keywords as given (None for an omitted one) plus ``extra``.

    :param kw: the sort keywords (``kind`` / ``stable`` / ``order``).
    :param extra: other params (``axis``, ``kth``).
    :returns: the params dict.
    """
    p = {"kind": kw.get("kind"), "stable": kw.get("stable"), "order": kw.get("order")}
    p.update(extra)
    return p


def _msb(n):
    """NumPy's ``npy_get_msb``: the position of the highest set bit of ``n`` (0 for 0 and 1).

    :param n: a non-negative count.
    :returns: the bit position.
    """
    d = 0
    while n > 1:
        n >>= 1
        d += 1
    return d


def _heapsort_idx(a, less):
    """NumPy's ``string_aheapsort_`` over a list of indices, in place (``_introsort_idx``'s fallback).

    :param a: the indices.
    :param less: the strict comparator ``less(i, j)``.
    """
    n = len(a)
    a.insert(0, None)  # NumPy's heapsort indexes from 1 (``a = tosort - 1``)
    for l in range(n >> 1, 0, -1):
        tmp = a[l]
        i, j = l, l << 1
        while j <= n:
            if j < n and less(a[j], a[j + 1]):
                j += 1
            if less(tmp, a[j]):
                a[i] = a[j]
                i = j
                j += j
            else:
                break
        a[i] = tmp
    while n > 1:
        tmp = a[n]
        a[n] = a[1]
        n -= 1
        i, j = 1, 2
        while j <= n:
            if j < n and less(a[j], a[j + 1]):
                j += 1
            if less(tmp, a[j]):
                a[i] = a[j]
                i = j
                j += j
            else:
                break
        a[i] = tmp
    a.pop(0)


def _introsort_idx(tosort, less):
    """NumPy's ``string_aquicksort_`` (npysort/quicksort.cpp) over a list of indices with a comparator, in place.

    A transcription used only to BUILD inputs (McIlroy's adversary) and to count how often an input reaches the heapsort
    fallback; the recorded answers are NumPy's own.

    :param tosort: the indices (sorted in place).
    :param less: the strict comparator ``less(i, j)``.
    :returns: how many ranges went to the heapsort fallback.
    """
    fallbacks = 0
    pl, pr = 0, len(tosort) - 1
    stack = []
    cdepth = _msb(len(tosort)) * 2
    while True:
        if cdepth < 0:
            fallbacks += 1
            seg = tosort[pl:pr + 1]
            _heapsort_idx(seg, less)
            tosort[pl:pr + 1] = seg
        else:
            while pr - pl > _SMALL_QUICKSORT:
                pm = pl + ((pr - pl) >> 1)
                if less(tosort[pm], tosort[pl]):
                    tosort[pm], tosort[pl] = tosort[pl], tosort[pm]
                if less(tosort[pr], tosort[pm]):
                    tosort[pr], tosort[pm] = tosort[pm], tosort[pr]
                if less(tosort[pm], tosort[pl]):
                    tosort[pm], tosort[pl] = tosort[pl], tosort[pm]
                vp = tosort[pm]  # NumPy keeps a pointer to the pivot's DATA, which survives the swap below
                pi, pj = pl, pr - 1
                tosort[pm], tosort[pj] = tosort[pj], tosort[pm]
                while True:
                    pi += 1
                    while less(tosort[pi], vp):
                        pi += 1
                    pj -= 1
                    while less(vp, tosort[pj]):
                        pj -= 1
                    if pi >= pj:
                        break
                    tosort[pi], tosort[pj] = tosort[pj], tosort[pi]
                pk = pr - 1
                tosort[pi], tosort[pk] = tosort[pk], tosort[pi]
                # NumPy pushes the larger partition with the decremented depth and continues with the smaller one.
                cdepth -= 1
                if pi - pl < pr - pi:
                    stack.append((pi + 1, pr, cdepth))
                    pr = pi - 1
                else:
                    stack.append((pl, pi - 1, cdepth))
                    pl = pi + 1
            for i in range(pl + 1, pr + 1):
                vi = tosort[i]
                j = i
                while j > pl and less(vi, tosort[j - 1]):
                    tosort[j] = tosort[j - 1]
                    j -= 1
                tosort[j] = vi
        if not stack:
            break
        pl, pr, cdepth = stack.pop()
    return fallbacks


def _adversary(n):
    """McIlroy's "A killer adversary for quicksort" against ``_introsort_idx``: values frozen lazily so that every pivot
    the introsort picks is small, which drives it into its heapsort fallback.

    :param n: the length.
    :returns: the values, a permutation of ``range(n)``.
    """
    gas = n
    val = [gas] * n
    state = {"nsolid": 0, "candidate": 0}

    def freeze(x):
        val[x] = state["nsolid"]
        state["nsolid"] += 1

    def less(x, y):
        if val[x] == gas and val[y] == gas:
            freeze(x if x == state["candidate"] else y)
        if val[x] == gas:
            state["candidate"] = x
        elif val[y] == gas:
            state["candidate"] = y
        return val[x] < val[y]

    _introsort_idx(list(range(n)), less)
    for i in range(n):
        if val[i] == gas:
            freeze(i)
    return val


def _units_array(rows, width):
    """A ``U`` array from raw code units (any 32-bit value, above U+10FFFF included), each row zero-padded to ``width``.

    :param rows: the code units of each element.
    :param width: the dtype's width in units.
    :returns: the array.
    """
    raw = np.zeros((len(rows), width), dtype=np.uint32)
    for i, r in enumerate(rows):
        raw[i, :len(r)] = r
    return np.frombuffer(raw.tobytes(), dtype=f"<U{width}").copy()


def _categorical(rng, n, k, width, kind, alphabet="abcdefghij"):
    """``n`` strings drawn from ``k`` distinct values of at most ``width`` characters (the tie-heavy data whose argsort
    exposes the sort's tie order); ``k`` is capped by how many distinct values the width allows.

    :param rng: the generator.
    :param n: the count.
    :param k: the number of distinct values.
    :param width: the most characters a value has (and the dtype's width).
    :param kind: ``"U"`` or ``"S"``.
    :param alphabet: the characters.
    :returns: the array.
    """
    k = min(k, sum(len(alphabet) ** length for length in range(width + 1)))
    pool = []
    while len(pool) < k:
        s = "".join(rng.choice(list(alphabet), int(rng.integers(0, width + 1))))
        if s not in pool:
            pool.append(s)
    values = [pool[int(i)] for i in rng.integers(0, k, n)]
    if kind == "S":
        return np.array([v.encode() for v in values], dtype=f"S{width}")
    return np.array(values, dtype=f"U{width}")


def _sort_all(make, layout, axes, kws, *, sort_kws=({},), partition=True, inplace_kws=()):
    """Records the sort family over one operand.

    :param make: returns a fresh (base, view) pair (called once for the calls that only read it, and once more for each
        in-place case, whose base the call rewrites).
    :param layout: the layout label.
    :param axes: the axes to record (``None`` flattens).
    :param kws: the keyword sets of ``np.argsort``.
    :param sort_kws: the keyword sets of ``np.sort`` (a value sort gives one answer for every kind over text).
    :param partition: also record ``np.partition`` / ``np.argpartition`` with ``kth`` -1 and 0.
    :param inplace_kws: the keyword sets of ``ndarray.sort`` (an int axis only; the view must be writeable).
    """
    op = make()
    view = op[1]
    for ax in axes:
        for kw in kws:
            add("strsort", "argsort_kw", params=_sort_params(kw, axis=ax), operands=[op], layout=layout,
                call=lambda a, x=ax, k=kw: np.argsort(a, axis=x, **k))
        for kw in sort_kws:
            add("strsort", "sort_kw", params=_sort_params(kw, axis=ax), operands=[op], layout=layout,
                call=lambda a, x=ax, k=kw: np.sort(a, axis=x, **k))
        if partition and view.ndim > 0:
            n_axis = view.size if ax is None else view.shape[ax]
            for kth in ([-1, 0] if n_axis > 1 else [0]):
                add("strsort", "partition_kw", params={"axis": ax, "kth": kth}, operands=[op], layout=layout,
                    call=lambda a, x=ax, k=kth: np.partition(a, k, axis=x))
                add("strsort", "argpartition_kw", params={"axis": ax, "kth": kth}, operands=[op], layout=layout,
                    call=lambda a, x=ax, k=kth: np.argpartition(a, k, axis=x))
        if ax is not None and view.ndim > 0 and view.flags.writeable:
            for kw in inplace_kws:
                add("strsort", "ndsort_kw", params=_sort_params(kw, axis=ax), operands=[make()], layout=layout, mutate=0,
                    call=lambda a, x=ax, k=kw: a.sort(axis=x, **k))


def _fixed(arr):
    """A ``make`` callable for ``_sort_all`` over a C-contiguous operand of ``arr``'s shape (a fresh copy per call).

    :param arr: the array.
    :returns: the callable.
    """
    return lambda: plain(arr)


def gen_strsort():
    """Section strsort: sorting text (P2.2). Every case is phase 1: NumSharp sorts ``S`` / ``U`` text, so it must match
    NumPy now.

    * NumPy's introsort tie order: categorical data at sizes around its insertion-only boundary (16 / 17 elements) and
      beyond, ``S`` and ``U`` of several widths, every kind spelling and the stable keyword;
    * distinct strings, long common prefixes (keys of the first units tie), every ``U`` key packing (Latin-1, the BMP,
      astral code points, units above U+10FFFF), a non-ASCII unit after eight ASCII ones, NULs and high bytes;
    * every layout of the shared catalog and wider lines over every axis, ``ndarray.sort`` in place (the whole base
      buffer compared);
    * McIlroy's adversary, which drives the introsort into its heapsort fallback (ties included);
    * NumPy's argument errors in its order, and ``StringDType``'s refusal of the heapsort kind.
    """
    rng = np.random.default_rng(20261015)

    # Tie order: categorical data (NumPy insertion-sorts a line of at most 16 elements, so 16 / 17 is its boundary).
    for n in (2, 15, 16, 17, 18, 33, 100, 257, 1000):
        variants = (("U", 3), ("U", 9), ("S", 8)) if n == 1000 else \
            (("U", 1), ("U", 3), ("U", 8), ("U", 9), ("S", 3), ("S", 8), ("S", 9))
        for kd, width in variants:
            for k in (2, 5):
                arr = _categorical(rng, n, k, width, kd)
                every = n in (17, 100)
                _sort_all(_fixed(arr), f"cat{kd}{width}k{k}n{n}", (-1,), _SORT_KWS if every else _SORT_KWS_CORE,
                          sort_kws=_SORT_KWS if n == 17 else ({},), partition=n in (16, 17, 100))
    for kd, width in (("U", 5), ("S", 5)):
        arr = _categorical(rng, 3000, 6, width, kd)
        _sort_all(_fixed(arr), f"cat{kd}{width}k6n3000", (-1,), _SORT_KWS_CORE, partition=False)

    # Distinct strings (any correct sort is NumPy's answer) and long common prefixes (the first units of every key tie).
    for n in (17, 300, 2000):
        draws = ("".join(rng.choice(list("abcdefghijklmnop"), int(rng.integers(1, 9)))) for _ in range(n * 3))
        values = list(dict.fromkeys(draws))[:n]
        for kd in ("U", "S"):
            arr = np.array([v.encode() for v in values], dtype="S8") if kd == "S" else np.array(values, dtype="U8")
            _sort_all(_fixed(arr), f"distinct{kd}n{n}", (-1,), _SORT_KWS_CORE, partition=False)
    for n in (40, 400):
        for kd, width in (("U", 24), ("S", 24), ("U", 40)):
            pool = list(dict.fromkeys("https://example.org/" + "".join(rng.choice(list("ab"), int(rng.integers(0, 5))))
                                      for _ in range(12)))
            values = [pool[int(i)] for i in rng.integers(0, len(pool), n)]
            arr = np.array([v.encode() for v in values], dtype=f"S{width}") if kd == "S" else \
                np.array(values, dtype=f"U{width}")
            _sort_all(_fixed(arr), f"prefix{kd}{width}n{n}", (-1,), _SORT_KWS_CORE, partition=False)

    # Every U key packing (NumSharp keys 8 / 16 / 21 / 32 bits a unit by the widest unit present), the late non-ASCII
    # unit, NULs (embedded and trailing) and high bytes.
    pools = {"latin1": [0, 0x41, 0x7F, 0x80, 0xE9, 0xFF], "bmp": [0x41, 0x100, 0x4E2D, 0xFFFF],
             "astral": [0x41, 0x10000, 0x1F600, 0x10FFFF], "over": [0x41, 0x110000, 0xFFFFFFFE, 0xFFFFFFFF]}
    for name, pool in pools.items():
        for width in (1, 2, 3, 5, 9):
            for n in (30, 200):
                rows = [[int(rng.choice(pool)) for _ in range(int(rng.integers(0, width + 1)))] for _ in range(n)]
                _sort_all(_fixed(_units_array(rows, width)), f"units-{name}-w{width}n{n}", (-1,), _SORT_KWS_CORE,
                          partition=False)
        rows = [[0x61] * 8 + [int(rng.choice(pool))] for _ in range(60)]
        _sort_all(_fixed(_units_array(rows, 9)), f"units-{name}-late", (-1,), _SORT_KWS_CORE, partition=False)
    for n in (20, 200):
        rows = [[int(rng.choice([0, 0x61, 0x62])) for _ in range(4)] for _ in range(n)]
        _sort_all(_fixed(_units_array(rows, 4)), f"nulU4n{n}", (-1,), _SORT_KWS_CORE, partition=False)
        raw = rng.choice(np.array([0, 0x61, 0x62, 0x7F, 0x80, 0xFF], dtype=np.uint8), (n, 5))
        _sort_all(_fixed(np.frombuffer(raw.tobytes(), dtype="S5").copy()), f"nulS5n{n}", (-1,), _SORT_KWS_CORE,
                  partition=False)

    # The catalog's layouts over the shared pools (U with non-ASCII text, S with high bytes). Its core layouts get every
    # axis (None flattens) and an in-place sort along each; the others the flattened and the last axis.
    for spec in SPECS:
        for lay in LAYOUTS:
            def make(s=spec, l=lay):
                return operand(s, l, seed=3)
            ndim = make()[1].ndim
            if ndim == 0:
                axes = [None]
            elif lay in CORE_LAYOUTS:
                axes = [None] + list(range(-1, ndim))
            else:
                axes = [None, -1]
            _sort_all(make, lay, axes, _SORT_KWS_CORE, inplace_kws=({},))
    # Wider lines (longer than 16 elements, so the introsort partitions) over C / F / reversed / stepped / permuted views.
    for spec in ("<U4", "|S4"):
        cat = _categorical(rng, 120, 3, 4, spec[1])
        builders = {"c": lambda b: b.reshape(4, 30), "f": lambda b: b.reshape(30, 4).T,
                    "r": lambda b: b.reshape(4, 30)[::-1, ::-1], "s": lambda b: b.reshape(2, 60)[:, ::2],
                    "t3": lambda b: b.reshape(2, 3, 20).transpose(1, 2, 0)}
        for key, build in builders.items():
            def make(c=cat, bld=build):
                base = c.copy()
                return base, bld(base)
            ndim = make()[1].ndim
            _sort_all(make, f"wide-{key}{spec[1]}", [None] + list(range(-1, ndim)), _SORT_KWS_CORE,
                      inplace_kws=({}, {"kind": "stable"}, {"stable": True}))

    # The heapsort fallback: McIlroy's adversary, its values halved (q = 2) so that the fallback orders ties too (a
    # coarser division no longer reaches it).
    for n in (64, 200, 500):
        values = _adversary(n)
        for q in (1, 2):
            qv = [v // q for v in values]
            fallbacks = _introsort_idx(list(range(n)), lambda x, y: qv[x] < qv[y])
            assert fallbacks > 0, f"the adversary of {n} / {q} does not reach the heapsort fallback"
            for kd in ("U", "S"):
                strs = [f"{v:04d}" for v in qv]
                arr = np.array([s.encode() for s in strs], dtype="S4") if kd == "S" else np.array(strs, dtype="U4")
                _sort_all(_fixed(arr), f"adversary{kd}n{n}q{q}fb{fallbacks}", (-1,), _SORT_KWS_CORE)

    # Prefix keys (8 units / bytes) that do not cover the strings and rarely repeat: NumSharp then runs NumPy's introsort
    # with its string compare straight over (key, index) records. Each line has a few duplicated strings (NumPy's tie
    # order decides them) and a few pairs equal in their first 8 characters only (the compare goes past the key).
    def tail_line(n, width, seed, dups, prefix_pairs, kd):
        r = np.random.default_rng(seed)
        values = []
        while len(values) < n - dups - 2 * prefix_pairs:
            values.append("".join(r.choice(list("abcdefghijklmnopqrstuvwxyz"), width)))
        for _ in range(prefix_pairs):
            head = "".join(r.choice(list("abcdefghij"), 8))
            values.append(head + "a" * (width - 8))
            values.append(head + "b" + "a" * (width - 9))
        for _ in range(dups):
            values.append(values[int(r.integers(0, len(values)))])
        order = r.permutation(len(values))
        values = [values[int(i)] for i in order]
        if kd == "S":
            return np.array([v.encode() for v in values], dtype=f"S{width}")
        return np.array(values, dtype=f"U{width}")

    for n, dups in ((40, 2), (300, 20), (1000, 150)):
        for kd, width in (("U", 9), ("U", 12), ("S", 10)):
            arr = tail_line(n, width, 7 * n + width, dups, 2, kd)
            _sort_all(_fixed(arr), f"tail{kd}{width}n{n}", (-1,), _SORT_KWS_CORE, partition=n == 40)
            if n == 300:
                # Two-dimensional: lines of 30 along the last axis, and the same lines along axis 0 (stepped columns).
                m = arr.reshape(10, 30)
                _sort_all(_fixed(m), f"tail{kd}{width}rows", (-1,), _SORT_KWS_CORE, partition=False)
                _sort_all(_fixed(m.T.copy()), f"tail{kd}{width}cols", (0,), _SORT_KWS_CORE, partition=False)
    # Short rows (at most 16 elements: NumSharp counts their ranks) whose 8-unit keys tie inside every row: the order
    # comes from the unit after the key. Rows laid end to end, F-ordered rows (strided) and the columns (axis 0).
    for kd in ("U", "S"):
        for w in (2, 3, 4, 7, 16):
            rr = np.random.default_rng(100 + w)
            rows = []
            for _ in range(12):
                head = "".join(rr.choice(list("ab"), 8))
                rows.append([head + str(rr.choice(list("zyxa"))) for _ in range(w)])
            arr = np.array(rows, dtype="U9")
            if kd == "S":
                arr = arr.astype("S9")
            _sort_all(_fixed(arr), f"rowties{kd}w{w}", (-1, 0), _SORT_KWS_CORE, partition=False)

            def make_f(a=arr):
                base = np.ascontiguousarray(a.T).reshape(-1).copy()
                return base, base.reshape(a.shape[1], a.shape[0]).T
            _sort_all(make_f, f"rowtiesF{kd}w{w}", (-1,), _SORT_KWS_CORE, partition=False)

    # The adversary through the same route: distinct 8-character keys (the heapsort fallback orders them) and a 9th unit.
    for n in (64, 200, 500):
        values = _adversary(n)
        fallbacks = _introsort_idx(list(range(n)), lambda x, y: values[x] < values[y])
        assert fallbacks > 0, f"the adversary of {n} does not reach the heapsort fallback"
        for kd in ("U", "S"):
            strs = [f"{v:08d}z" for v in values]
            arr = np.array([s.encode() for s in strs], dtype="S9") if kd == "S" else np.array(strs, dtype="U9")
            _sort_all(_fixed(arr), f"adversary-tail{kd}n{n}fb{fallbacks}", (-1,), _SORT_KWS_CORE, partition=False)

    # NumPy's argument errors, in its order: the kind (parsed first: its first letter, quoted by repr), the kind with
    # stable, the order, then the axis. Numbers refuse the same arguments.
    row = plain(np.array(["b", "a", "c"]))
    bad = [{"kind": "timsort"}, {"kind": ""}, {"kind": "x"}, {"kind": "it's"}, {"kind": 'a"b'}, {"kind": "radix"},
           {"kind": "stable", "stable": True}, {"kind": "q", "stable": False}, {"order": "x"}, {"order": ["x"]},
           {"kind": "x", "order": "x"}, {"kind": "stable", "stable": True, "order": "x"},
           {"kind": "x", "stable": True}]
    for kw in bad:
        for ax in (-1, 9, None):
            add("strsort", "argsort_kw", params=_sort_params(kw, axis=ax), operands=[row], layout="row",
                call=lambda a, x=ax, k=kw: np.argsort(a, axis=x, **k))
            add("strsort", "sort_kw", params=_sort_params(kw, axis=ax), operands=[row], layout="row",
                call=lambda a, x=ax, k=kw: np.sort(a, axis=x, **k))
        add("strsort", "ndsort_kw", params=_sort_params(kw, axis=-1), operands=[target(np.array(["b", "a", "c"]))],
            layout="row", mutate=0, call=lambda a, k=kw: a.sort(axis=-1, **k))
    for kw in ({"kind": "x"}, {"kind": "stable", "stable": True}, {"order": "f"}):
        num = num_operand(np.array([3, 1, 2]))
        add("strsort", "argsort_kw", params=_sort_params(kw, axis=-1), operands=[num], layout="int",
            call=lambda a, k=kw: np.argsort(a, **k))
        add("strsort", "sort_kw", params=_sort_params(kw, axis=-1), operands=[num], layout="int",
            call=lambda a, k=kw: np.sort(a, **k))
    zero_d = plain(np.array("ab"))
    for ax in (-1, 0, 1, None):
        add("strsort", "sort_kw", params=_sort_params({}, axis=ax), operands=[zero_d], layout="0d",
            call=lambda a, x=ax: np.sort(a, axis=x))
        add("strsort", "argsort_kw", params=_sort_params({}, axis=ax), operands=[zero_d], layout="0d",
            call=lambda a, x=ax: np.argsort(a, axis=x))
    for ax in (-1, 1, -3):
        add("strsort", "ndsort_kw", params=_sort_params({}, axis=ax), operands=[plain(np.array([["b", "a"], ["d", "c"]]))],
            layout="2d", mutate=0, call=lambda a, x=ax: a.sort(axis=x))
    for kw in ({}, {"kind": "x"}):
        add("strsort", "ndsort_kw", params=_sort_params(kw, axis=-1), operands=[operand("<U3", "bc")], layout="bc",
            mutate=0, call=lambda a, k=kw: a.sort(axis=-1, **k))

    # StringDType refuses the heapsort kind: a sort after its axis (and writeability) checks, an argsort before them; one
    # element and an empty array included.
    for values in (["b", "a"], ["q"], []):
        for ax in (-1, 5):
            t = t_plain(np.array(values, dtype=TD.build()))
            add("strsort", "sort_kw", params=_sort_params({"kind": "heapsort"}, axis=ax), operands=[t],
                layout=f"T{len(values)}", call=lambda a, x=ax: np.sort(a, axis=x, kind="heapsort"))
            add("strsort", "argsort_kw", params=_sort_params({"kind": "heapsort"}, axis=ax), operands=[t],
                layout=f"T{len(values)}", call=lambda a, x=ax: np.argsort(a, axis=x, kind="heapsort"))


# ----------------------------------------------------------------------------------------------------------------------
# Section strchar: numpy.char's module functions (np-char plan, phase 1) — the six comparisons and compare_chararrays that
# strip trailing whitespace, np.char.add and np.char.multiply — strict, except a StringDType add / multiply NumPy computes
# (phase 2: NumSharp's StringDType arithmetic is a later subphase).
# ----------------------------------------------------------------------------------------------------------------------

# The units numpy's Buffer::rstrip removes (NUL and NumPyOS_ascii_isspace), and units it must KEEP: the other whitespace
# (0x1c, 0x85, 0xa0, U+3000), neighbours of the strip set ('\b', '!'), high bytes, a supplementary-plane character.
STRIP_UNITS = {"U": ["\x00", " ", "\t", "\n", "\x0b", "\x0c", "\r"],
               "S": [b"\x00", b" ", b"\t", b"\n", b"\x0b", b"\x0c", b"\r"]}
KEEP_UNITS = {"U": ["a", "b", "\x1c", "\x85", "\xa0", "　", "\U0001F600", "\x08", "!"],
              "S": [b"a", b"b", b"\x1c", b"\x85", b"\xa0", b"\x80", b"\xff", b"\x08", b"!"]}
STRIP_SET = {"U": "\x00 \t\n\x0b\x0c\r", "S": b"\x00 \t\n\x0b\x0c\r"}

# Width pairs per class: equal widths, a longer second operand, a shorter one, the bit-mask kernel's 64-unit limit on
# both sides (U64 / U65, S64 / S65) and two wide pairs past it.
STRIP_WIDTHS = {
    "U": [("U3", "U3"), ("U3", "U8"), ("U9", "U2"), ("U64", "U64"), ("U65", "U60"), ("U70", "U66")],
    "S": [("S3", "S3"), ("S2", "S9"), ("S33", "S8"), ("S64", "S65"), ("S300", "S290")],
}

CHAR_CMP_OPS = ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"]


def _strip_values(kind, width, n, seed, like=None):
    """``n`` values of one class, at most ``width`` code units each, built so NumPy's strip rule decides many pairs.

    Without ``like``: content (keep units, now and then a strip unit INSIDE the string) followed by a trailing run of
    strip units, the lengths cycling. With ``like`` (the other operand's values), two values in three are derived from the
    value at the same index: its stripped content with another trailing run (equal once stripped), with one unit changed,
    or with one more content unit.

    Args:
        kind: ``"U"`` (str) or ``"S"`` (bytes).
        width: the dtype's width in code units.
        n: how many values.
        seed: an offset that varies the cycles between operands.
        like: the other operand's base values, or None.

    Returns:
        A list of ``n`` str / bytes values.
    """
    strip, keep, strip_set = STRIP_UNITS[kind], KEEP_UNITS[kind], STRIP_SET[kind]
    join = "".join if kind == "U" else b"".join
    out = []
    for i in range(n):
        if like is not None and (i + seed) % 3 != 0:
            core = like[i % len(like)].rstrip(strip_set)[:width]
            mode = (i * 7 + seed) % 4
            if mode == 1 and core:
                k = (i * 5 + seed) % len(core)
                core = core[:k] + keep[(i + seed) % len(keep)] + core[k + 1:]
            elif mode == 2 and len(core) < width:
                core = core + keep[(i * 3 + seed) % len(keep)]
            room = width - len(core)
            trail = join(strip[(i * 3 + k + seed) % len(strip)] for k in range((i * 2 + seed) % (room + 1)))
            out.append(core + trail)
            continue
        clen = (i * 3 + seed) % (width + 1)
        content = join(keep[(i * 5 + k * 3 + seed) % len(keep)] if (i + k + seed) % 9 else strip[(i + k) % len(strip)]
                       for k in range(clen))
        tlen = (i * 2 + seed) % (width - clen + 1)
        out.append(content + join(strip[(i * 3 + k + seed) % len(strip)] for k in range(tlen)))
    return out


def _strip_operand(spec, layout, seed, like=None):
    """A (base, view) operand of dtype ``spec`` in one of LAYOUTS, its base built by ``_strip_values``.

    Args:
        spec: the dtype, ``"U3"`` / ``"S9"``.
        layout: a LAYOUTS key.
        seed: passed to ``_strip_values``.
        like: the other operand's base values (same base length), or None.

    Returns:
        The (base, view) pair.
    """
    n, build = LAYOUTS[layout]
    kind, width = spec[0], int(spec[1:])
    base = np.array(_strip_values(kind, width, n, seed, like), dtype=spec) if n else np.zeros(0, dtype=spec)
    return base, build(base)


def _char_arg(v, operands):
    """One argument of a numpy.char call as the replay rebuilds it: a (base, view) pair becomes an operand (``{"op": k}``),
    a Python value travels encoded (``{"v": ...}``: a str, bytes, a list of them, a number, a bool, None).

    Args:
        v: the argument.
        operands: the case's operand list (appended to).

    Returns:
        The argument's spec.
    """
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], np.ndarray):
        operands.append(v)
        return {"op": len(operands) - 1}
    return {"v": enc(v)}


def _char_call(op, args, call, *, layout="", phase=1, extra=None):
    """Records one numpy.char call whose arguments may be operands or Python values (``_char_arg``).

    Args:
        op: the corpus op key (``char_cmp``, ``char_compare``, ``char_add``, ``char_multiply``).
        args: the call's positional arguments, in order (operand pairs or Python values).
        call: the NumPy call, taking the arguments in the same order (operand views substituted).
        layout: the case's layout label.
        phase: 1, or ``"compute"`` for a StringDType add / multiply NumPy computes.
        extra: more params (the comparison's name, compare_chararrays' cmp / rstrip).
    """
    operands = []
    specs = [_char_arg(a, operands) for a in args]
    params = dict(extra or {})
    params["args"] = specs

    def run(*views):
        it = iter(views)
        real = [next(it) if "op" in s else a for s, a in zip(specs, args)]
        return call(*real)

    add("strchar", op, params=params, operands=operands, call=run, phase=phase, layout=layout)


def gen_strchar():
    """Section strchar: np.char's stripping comparisons and compare_chararrays, np.char.add and np.char.multiply."""
    # A. The six comparisons over every layout kind (same layout for both operands) and the cross pairs, every width pair.
    for kind, pairs in STRIP_WIDTHS.items():
        for s1, s2 in pairs:
            for l1, l2 in [(name, name) for name in LAYOUTS] + CMP_CROSS:
                b1, v1 = _strip_operand(s1, l1, seed=41)
                like = [x for x in b1.tolist()] if b1.size else None
                b2, v2 = _strip_operand(s2, l2, seed=42, like=like if LAYOUTS[l1][0] == LAYOUTS[l2][0] else None)
                for name in CHAR_CMP_OPS:
                    _char_call("char_cmp", [(b1, v1), (b2, v2)], lambda x, y, f=getattr(np.char, name): f(x, y),
                               layout=f"{s1}:{l1},{s2}:{l2}", extra={"name": name})

    # B. Special units through an outer broadcast: every strip unit, the whitespace NumPy keeps, units past U+10FFFF
    #    (unsigned values), and strip units inside a string / before it.
    u_units = [0, 9, 10, 11, 12, 13, 32, 0x1C, 0x85, 0xA0, 0x3000, 0x110000, 0xFFFFFFFF, 0x41]
    u_pairs = [0x41, 0x20, 0x41, 0x09, 0x41, 0x00, 0x20, 0x41, 0x41, 0x1C, 0x41, 0x110000, 0x41, 0x0D, 0x00, 0x41]
    u_row = [0x41, 0x00, 0x41, 0x20, 0x20, 0x00, 0x41, 0x0B]
    s_bytes = [b"", b" ", b"\t", b"\x0b", b"\x1c", b"\x85", b"\xa0", b"\x80", b"\xff", b"a", b"a ", b"a\t", b"a\x00",
               b" a"]
    for name in CHAR_CMP_OPS:
        f = getattr(np.char, name)
        _char_call("char_cmp", [_units_operand(u_units, 1, (len(u_units), 1)), _units_operand(u_units, 1, (1, len(u_units)))],
                   lambda x, y, f=f: f(x, y), layout="U1:units", extra={"name": name})
        _char_call("char_cmp", [_units_operand(u_pairs, 2, (8, 1)), _units_operand(u_row, 2, (1, 4))],
                   lambda x, y, f=f: f(x, y), layout="U2:units", extra={"name": name})
        _char_call("char_cmp", [plain(np.array(s_bytes, dtype="S2").reshape(-1, 1)),
                                plain(np.array(s_bytes, dtype="S3").reshape(1, -1))],
                   lambda x, y, f=f: f(x, y), layout="S:bytes", extra={"name": name})
        # The first differing unit differs only in a HIGHER byte (U+3100 / U+3000, U+10000 / U+20000, U+0100 / U+0200)
        # and the next units order the other way: a compare must read whole units, on both sides of the 64-unit limit.
        for chars in [8, 70]:
            pad = chars - 8
            a_rows = [[0x61] * (4 + pad) + [0x3100, 0x61, 0x20, 0x00], [0x61] * (4 + pad) + [0x10000, 0x7A, 0x09, 0x00],
                      [0x61] * (5 + pad) + [0x0100, 0x7A, 0x20]]
            b_rows = [[0x61] * (4 + pad) + [0x3000, 0x7A, 0x00, 0x00], [0x61] * (4 + pad) + [0x20000, 0x61, 0x00, 0x00],
                      [0x61] * (5 + pad) + [0x0200, 0x61, 0x00]]
            _char_call("char_cmp", [_units_operand([u for r in a_rows for u in r], chars),
                                    _units_operand([u for r in b_rows for u in r], chars)],
                       lambda x, y, f=f: f(x, y), layout=f"U{chars}:highbyte", extra={"name": name})

    # C. The arguments as Python values (NumPy's np.asanyarray / PyArray_FROM_O of them): a str, bytes, lists of each,
    #    numbers and bools (no string array), and the refusals in NumPy's order (non-string, StringDType, the S / U mix
    #    that answers NotImplemented, the broadcast).
    u2 = _strip_operand("U4", "c2", seed=43)
    s2 = _strip_operand("S4", "c2", seed=44)
    t_spec = TSpec()
    t1 = t_operand(t_spec, "c1", seed=3)
    u_row3 = plain(np.array(["a ", "b\t", "c"]))
    for name in CHAR_CMP_OPS:
        f = getattr(np.char, name)
        for x, y, label in [(u2, "a  ", "U,str"), ("b", u2, "str,U"), ("a ", "a\x00\t", "str,str"), (s2, b"a ", "S,bytes"),
                            (b"\x80 ", s2, "bytes,S"), (b"a", b"a\t ", "bytes,bytes"), (["a ", "b"], "a", "list,str"),
                            ([b"a ", b"b"], [b"a", b"b "], "blist,blist"), (u2, s2, "U,S"), (s2, u2, "S,U"), (u2, b"a", "U,bytes"),
                            ("a", s2, "str,S"), (u2, 1, "U,int"), (1.5, u2, "float,U"), (True, False, "bool,bool"),
                            (u2, [1, 2], "U,list"), (t1, t1, "T,T"), (u_row3, t1, "U,T"), (t1, "a", "T,str"),
                            (plain(np.array(["a", "b"])), plain(np.array(["a", "b", "c"])), "U:2,U:3"),
                            (plain(np.array([b"a", b"b"])), plain(np.array(["a", "b", "c"])), "S:2,U:3"),
                            (plain(np.zeros((0, 3), "U2")), "a", "empty,str"), (plain(np.array("ab ")), "ab", "0d,str")]:
            _char_call("char_cmp", [x, y], lambda a, b, f=f: f(a, b), layout=label, extra={"name": name})

    # D. compare_chararrays: every cmp spelling (str and bytes; the six comparisons and the refused ones, a lone
    #    surrogate, None) x rstrip, then the refusals' order (a bad cmp before everything else).
    cmps = ["==", "!=", "<", "<=", ">", ">=", b"==", b"!=", b"<", b"<=", b">", b">=", "=", "=<", "<>", "", "===", "! ",
            "é", "=é", "\ud800", "\x00", "=\x00", "==\x00", " =", "=>", b"\xff", b"", b"=", None]
    pairs = [(_strip_operand("U3", "c2", seed=45), _strip_operand("U5", "bc", seed=46), "U3:c2,U5:bc"),
             (_strip_operand("S3", "f2", seed=47), _strip_operand("S4", "c2", seed=48), "S3:f2,S4:c2")]
    for cmp in cmps:
        for rstrip in [True, False]:
            for a1, a2, label in pairs:
                _char_call("char_compare", [a1, a2, cmp, rstrip],
                           lambda x, y, c, r: np.char.compare_chararrays(x, y, c, r), layout=label)
    for a1, a2, label in [(t1, t1, "T,T"), (u2, s2, "U,S"), (u2, plain(np.array([1, 2])), "U,int"),
                          (plain(np.array(["a", "b"])), plain(np.array(["a", "b", "c"])), "U:2,U:3"),
                          (plain(np.array([b"a", b"b"])), plain(np.array(["a", "b", "c"])), "S:2,U:3")]:
        for cmp in ["==", "=", None]:
            _char_call("char_compare", [a1, a2, cmp, True], lambda x, y, c, r: np.char.compare_chararrays(x, y, c, r),
                       layout=f"order:{label}")

    # E. np.char.add (np.add itself): layouts, the str / bytes arguments, the mix and no-loop errors, a StringDType
    #    operand (NumPy computes; phase 2 for NumSharp).
    for l1, l2 in [("c2", "c2"), ("f2", "f2"), ("c2", "bc"), ("z0", "z0"), ("e2", "e2"), ("r2", "bcol"), ("t3", "t3")]:
        for spec1, spec2 in [("U4", "U3"), ("S4", "S3")]:
            _char_call("char_add", [_strip_operand(spec1, l1, seed=49), _strip_operand(spec2, l2, seed=50)],
                       lambda x, y: np.char.add(x, y), layout=f"{spec1}:{l1},{spec2}:{l2}")
    for x, y, label in [("a", "b ", "str,str"), (u2, "z", "U,str"), ("z", u2, "str,U"), (s2, b"z\x00", "S,bytes"),
                        (b"", b"", "bytes,bytes"), (u2, s2, "U,S"), (u2, b"z", "U,bytes"), (u2, 1, "U,int"),
                        (["a", "bc"], ["d"], "list,list")]:
        _char_call("char_add", [x, y], lambda a, b: np.char.add(a, b), layout=label)
    for x, y, label in [(t1, t1, "T,T"), (t1, "z", "T,str")]:
        _char_call("char_add", [x, y], lambda a, b: np.char.add(a, b), layout=label, phase="compute")

    # F. np.char.multiply: np.strings.multiply with every TypeError re-raised as ValueError("Can only multiply by
    #    integers") — layouts and counts, the count kinds, a numeric / empty / str operand, the overflow (an
    #    OverflowError passes through), a StringDType operand (NumPy computes an integer count; a float count is the
    #    ValueError first).
    for spec in ["U3", "S3", "U70"]:
        for layout in ["c2", "f2", "t3", "r2", "s2", "bc", "z0", "one", "d5"]:
            b, v = _strip_operand(spec, layout, seed=51)
            for c in [plain(np.array(2)), plain(np.array(0)), plain(np.array(-3)), _count_operand(v, np.int64), 3]:
                _char_call("char_multiply", [(b, v), c], lambda a, i: np.char.multiply(a, i), layout=f"{spec}:{layout}")
    for c, label in [(2.5, "float"), (True, "bool"), ("x", "str"), (b"x", "bytes"), (plain(np.array(2, dtype="u8")), "u8"),
                     (2 ** 62, "2**62"), (plain(np.array(2.0)), "f8"), ([1, 2, 3], "list:3"), (plain(np.array([[1], [2]])), "arr:col"),
                     (plain(np.array(2, dtype="i1")), "i1")]:
        _char_call("char_multiply", [u2, c], lambda a, i: np.char.multiply(a, i), layout=f"count:{label}")
    for a, c, label in [([1, 2], 2, "a:list-int"), (plain(np.zeros(0, "U3")), 2, "a:empty"), ("ab", 3, "a:str"),
                        (b"ab", 2, "a:bytes"), ("abc", 2 ** 62, "overflow"), (t1, 2.5, "T:float")]:
        _char_call("char_multiply", [a, c], lambda x, i: np.char.multiply(x, i), layout=label)
    _char_call("char_multiply", [t1, 2], lambda x, i: np.char.multiply(x, i), layout="T:int", phase="compute")


# ----------------------------------------------------------------------------------------------------------------------
# strsearch. Searching text: np.argmax / np.argmin / np.nanargmax / np.nanargmin and the ndarray methods, np.searchsorted
# and ndarray.searchsorted, np.digitize of text (strings phase 2, P2.2 milestone B)
# ----------------------------------------------------------------------------------------------------------------------

def _search_text(values, spec):
    """A C-contiguous text operand of the given str values at ``spec`` (encoded as UTF-8 for an ``S`` spec).

    :param values: the str values.
    :param spec: the dtype, e.g. ``"U8"`` or ``"S8"``.
    :returns: the (base, view) pair.
    """
    if spec.lstrip("<|")[0] == "S":
        return plain(np.array([v.encode() for v in values], dtype=spec))
    return plain(np.array(values, dtype=spec))


def _rand_strings(rng, n, width, alphabet, minlen=0):
    """``n`` strings of ``minlen`` to ``width`` characters drawn from ``alphabet``.

    :param rng: the generator.
    :param n: the count.
    :param width: the most characters.
    :param alphabet: the characters.
    :param minlen: the fewest characters.
    :returns: the list.
    """
    return ["".join(rng.choice(list(alphabet), int(rng.integers(minlen, width + 1)))) for _ in range(n)]


def _pick(rng, n, pool):
    """``n`` values drawn from ``pool`` (categorical data: the extreme value recurs).

    :param rng: the generator.
    :param n: the count.
    :param pool: the values.
    :returns: the list.
    """
    return [pool[int(i)] for i in rng.integers(0, len(pool), n)]


def _argred(layout, op, axes, *, names=("argmax", "argmin"), keepdims=(False,), methods=False):
    """Records the arg-reductions of one operand.

    :param layout: the layout label.
    :param op: the (base, view) operand.
    :param axes: the axes (``None`` flattens).
    :param names: the functions (``argmax`` / ``argmin`` / ``nanargmax`` / ``nanargmin``).
    :param keepdims: the keepdims values.
    :param methods: also record ``ndarray.argmax`` / ``ndarray.argmin``.
    """
    for ax in axes:
        for kd in keepdims:
            for name in names:
                add("strsearch", "argred_kw", params={"name": name, "axis": ax, "keepdims": kd, "method": False,
                                                      "out": False},
                    operands=[op], layout=layout, call=lambda a, f=getattr(np, name), x=ax, k=kd: f(a, axis=x, keepdims=k))
            if methods:
                for name in ("argmax", "argmin"):
                    add("strsearch", "argred_kw", params={"name": name, "axis": ax, "keepdims": kd, "method": True,
                                                          "out": False},
                        operands=[op], layout=layout, call=lambda a, n=name, x=ax, k=kd: getattr(a, n)(axis=x, keepdims=k))


def _argred_out(layout, op, out, name, ax, kd, method=False):
    """Records one arg-reduction into an ``out=`` array (its view and whole base compared afterwards).

    :param layout: the layout label.
    :param op: the (base, view) text operand.
    :param out: the (base, view) out operand.
    :param name: the function or method name.
    :param ax: the axis (``None`` flattens).
    :param kd: keepdims.
    :param method: the ndarray method instead of the function.
    """
    if method:
        call = lambda a, o, n=name, x=ax, k=kd: getattr(a, n)(axis=x, out=o, keepdims=k)
    else:
        call = lambda a, o, f=getattr(np, name), x=ax, k=kd: f(a, axis=x, out=o, keepdims=k)
    add("strsearch", "argred_kw", params={"name": name, "axis": ax, "keepdims": kd, "method": method, "out": True},
        operands=[op, out], layout=layout, mutate=1, call=call)


def _search(layout, hay, keys, side="left", sorter=None, method=False):
    """Records one search (``np.searchsorted``, or ``ndarray.searchsorted`` with ``method``).

    :param layout: the layout label.
    :param hay: the (base, view) haystack.
    :param keys: the (base, view) keys.
    :param side: the side name.
    :param sorter: the (base, view) sorter, or ``None``.
    :param method: the ndarray method instead of the function.
    """
    ops = [hay, keys] + ([sorter] if sorter is not None else [])
    if method:
        call = lambda h, k, *s, sd=side: h.searchsorted(k, side=sd, sorter=s[0] if s else None)
    else:
        call = lambda h, k, *s, sd=side: np.searchsorted(h, k, side=sd, sorter=s[0] if s else None)
    add("strsearch", "ndsearchsorted" if method else "searchsorted_kw", params={"side": side, "sorter": sorter is not None},
        operands=ops, layout=layout, call=call)


def gen_strsearch():
    """Section strsearch: searching text (P2.2 milestone B). Every case is phase 1: NumSharp searches ``S`` / ``U`` text,
    so it must match NumPy now.

    * arg-reductions: NumPy's first extreme element over random, categorical (the extreme recurs: the first copy wins),
      long-prefix (the prefix skip), all-equal and NUL / high-byte data, every ``U`` key packing, block boundaries (1,023 /
      1,024 / 1,025 / 2,049 elements, the extreme repeated in later blocks), every axis of 2-D / 3-D arrays with keepdims,
      the catalog's layouts, broadcast views, ``out=`` of every kind (NumPy's checks: the empty-sequence error, the shape,
      the 'safe' cast to intp), the ndarray methods, and the axis errors;
    * searches: sorted and unsorted haystacks (NumPy's exact path, its monotonic carry), both sides, sorted / descending /
      random keys, sorters (valid, out of range where the search reaches it and where it does not, every integer kind,
      the refused kinds), few keys against a long haystack, mixed widths, shared prefixes that reach the narrower width,
      ``S`` with ``U``, numbers with text, keys of every shape and layout, haystack views, every ``U`` packing, units above
      U+10FFFF, keys that end before the strings with ties in the haystack, the method, and the argument errors;
    * digitize: text ``x`` with increasing, decreasing, repeated, float and one-element bins, both ``right``, 0-d / 2-D /
      empty ``x``, and the refusals of text bins.
    """
    rng = np.random.default_rng(20261016)
    names4 = ("argmax", "argmin", "nanargmax", "nanargmin")

    # ---- arg-reductions ------------------------------------------------------------------------------------------------
    # The catalog's layouts: every axis (None flattens) and keepdims over the core layouts, the methods there too.
    for spec in SPECS:
        for lay in LAYOUTS:
            op = operand(spec, lay, seed=5)
            ndim = op[1].ndim
            if ndim == 0:
                axes = [None, 0, -1, 1]
            elif lay in CORE_LAYOUTS:
                axes = [None] + list(range(-ndim, ndim)) + [ndim]
            else:
                axes = [None, -1]
            core = lay in CORE_LAYOUTS
            _argred(lay, op, axes, names=names4 if core else ("argmax", "argmin"),
                    keepdims=(False, True) if core else (False,), methods=core)

    # Lines: random, categorical and long-prefix text at sizes around NumSharp's 16-element short lines and 1,024-element
    # blocks.
    big_flavors = {1023: ("cat",), 1025: ("rand", "cat", "prefix"), 2049: ("cat", "prefix")}
    for n in (2, 3, 16, 17, 31, 100, 1023, 1025, 2049):
        variants = (("U", 8), ("S", 8)) if n >= 1000 else \
            (("U", 1), ("U", 3), ("U", 8), ("U", 12), ("S", 1), ("S", 3), ("S", 8), ("S", 12))
        for kd, width in variants:
            for flavor in big_flavors.get(n, ("rand", "cat", "prefix")):
                if flavor == "rand":
                    vals = _rand_strings(rng, n, width, "abcdefghijklmnop")
                elif flavor == "cat":
                    vals = _pick(rng, n, list(dict.fromkeys(_rand_strings(rng, 12, width, "abcd")))[:4])
                else:
                    pre = "x" * max(0, width - 2)
                    vals = [pre + s for s in _rand_strings(rng, n, 2, "ab")]
                op = _search_text(vals, f"{kd}{width}")
                _argred(f"line-{flavor}{kd}{width}n{n}", op, [None])
                if n == 100:
                    for view, build in (("step3", lambda b: b[::3]), ("rev", lambda b: b[::-1])):
                        _argred(f"line-{flavor}{kd}{width}n{n}-{view}", (op[0], build(op[1])), [None])

    # Categorical words, the best word recurring one element in eight (NumSharp's U filter switches to keys once a block
    # finds that many copies of its best): the first copy wins, whichever filter runs.
    words = ["apple", "banana", "cherry", "date", "elder", "fig", "grape", "honeydew"]
    for n in (40, 2100):
        for kd in ("U", "S"):
            op = _search_text(_pick(rng, n, words), f"{kd}8")
            _argred(f"words{kd}n{n}", op, [None])
            if n == 2100:
                op2 = (op[0], op[1].reshape(210, 10))
                _argred(f"words{kd}n{n}-2d", op2, [0, 1])

    # Every U key packing (NumSharp keys 8 / 16 / 21 / 32 bits a unit), units above U+10FFFF, NULs and high bytes.
    alphabets = {"ascii": "abcdefghij", "latin": "aàbéz\xff", "bmp": "aäā中文￿", "astral": "a\U0001F600\U00010000中"}
    for name, alpha in alphabets.items():
        for n in (5, 40, 1100):
            for width in (2, 5, 9):
                _argred(f"pack-{name}-w{width}n{n}", _search_text(_rand_strings(rng, n, width, alpha), f"U{width}"), [None])
    hi = np.frombuffer(np.array([0x41, 0x110000, 0x7FFFFFFF, 0xFFFFFFFF, 0, 0x10FFFF] * 50, dtype="<u4").tobytes(),
                       dtype="<U2").copy()
    _argred("units-above-10ffff", plain(hi), [None])
    _argred("S-nul-high", plain(np.array([b"a", b"a\x00b", b"a\x00", b"\xff", b"\xff\x00", b"", b"\x00\x01"] * 20,
                                         dtype="S3")), [None])
    _argred("U-nul", plain(np.array(["a", "a\x00b", "a\x00", "￿", "", "\x00\x01"] * 30, dtype="U3")), [None])
    # All-equal lines (every element the same string), and lines equal but for one element at a block edge.
    for n in (20, 1100):
        for kd in ("U", "S"):
            _argred(f"equal{kd}n{n}", _search_text(["same"] * n, f"{kd}4"), [None])
            vals = ["same"] * n
            vals[n - 1] = "samf"
            vals[min(1024, n - 2)] = "samf"
            _argred(f"equal-but{kd}n{n}", _search_text(vals, f"{kd}4"), [None])

    # Block boundaries: the global extreme repeated in later blocks (the earliest copy wins), also after a shared prefix.
    for n in (1025, 2049):
        for kd in ("U", "S"):
            vals = _rand_strings(rng, n, 6, "abcde")
            for pos in (0, 1023, 1024, 2048, n - 1):
                if pos < n:
                    vals[pos] = "zzzzzz"
            _argred(f"blocks{kd}n{n}", _search_text(vals, f"{kd}6"), [None])
            long_pre = ["https://site.example/" + v for v in vals]
            _argred(f"blocks-prefix{kd}n{n}", _search_text(long_pre, f"{kd}27"), [None])

    # Every axis of 2-D / 3-D arrays (short rows and long ones), keepdims, and views over them.
    for shape in ((6, 10), (10, 6), (4, 5, 6), (3, 40, 2), (400, 3), (3, 400)):
        for kd in ("U", "S"):
            size = int(np.prod(shape))
            arr = np.array(_pick(rng, size, ["ab", "b", "abc", "ba", "c"]), dtype=f"{kd}4") if kd == "U" else \
                np.array([v.encode() for v in _pick(rng, size, ["ab", "b", "abc", "ba", "c"])], dtype="S4")
            m = arr.reshape(shape)
            nd = len(shape)
            label = "x".join(map(str, shape))
            big = size > 1000
            _argred(f"axes{kd}{label}", plain(m), [None] + list(range(-nd, nd)),
                    names=("argmax", "argmin") if big else names4, keepdims=(False,) if big else (False, True))
            if big:
                continue
            for view, build in (("T", lambda b: b.T), ("rev", lambda b: b[::-1]), ("step", lambda b: b[::2]),
                                ("lastrev", lambda b: b[..., ::-1])):
                base = plain(m)
                _argred(f"axes{kd}{label}-{view}", (base[0], build(base[1])), [None, 0, -1])
    bc = np.array(_pick(rng, 8, ["abc", "b", "ab"]), dtype="U3")
    _argred("bcast", (bc, np.broadcast_to(bc, (5, 8))), [None, 0, 1])

    # out=: the right shape and every accepted integer dtype (NumPy casts the indices 'unsafe' after a 'safe' check of the
    # dtype: bool through uint32 pass, uint64 and floats do not), keepdims shapes, a wrong shape, and the empty-sequence
    # error that comes first; the methods take the same out.
    a2 = plain(np.array([["b", "a", "c"], ["a", "c", "c"]], dtype="U2"))
    for name in names4:
        for dt in ("int64", "int32", "int8", "uint8", "uint32", "bool", "uint64", "float64"):
            _argred_out(f"out-flat-{dt}", a2, target(np.zeros((), dt)), name, None, False)
            _argred_out(f"out-ax1-{dt}", a2, target(np.zeros(2, dt)), name, 1, False)
        _argred_out("out-ax0-kd", a2, target(np.zeros((1, 3), np.int64)), name, 0, True)
        _argred_out("out-flat-kd", a2, target(np.zeros((1, 1), np.int64)), name, None, True)
        _argred_out("out-badshape", a2, target(np.zeros(3, np.int64)), name, 1, False)
        _argred_out("out-badshape-kd", a2, target(np.zeros(2, np.int64)), name, 1, True)
        _argred_out("out-empty", plain(np.zeros((0, 3), "U2")), target(np.zeros(5, np.int64)), name, 0, False)
        _argred_out("out-empty-flat", plain(np.zeros(0, "U2")), target(np.zeros((), np.float64)), name, None, False)
        zb = np.zeros(6, np.int64)
        _argred_out("out-strided", a2, (zb, zb[::3]), name, 1, False)
    for name in ("argmax", "argmin"):
        _argred_out("out-method", a2, target(np.zeros(3, np.int64)), name, 0, False, method=True)
        _argred_out("out-method-bad", a2, target(np.zeros(3, np.float64)), name, 0, False, method=True)

    # Errors: an empty line, an empty axis, an axis out of range, 0-d arrays.
    for name in names4:
        for shape, ax in (((0,), None), ((0, 3), 0), ((0, 3), 1), ((3, 0), 1), ((3, 0), None), ((2, 3), 2), ((2, 3), -3)):
            _argred(f"err{shape}ax{ax}", plain(np.zeros(shape, "U2")), [ax], names=(name,))
    _argred("zero-d", plain(np.array("ab")), [None, 0, -1, 1], names=names4, keepdims=(False, True))

    # ---- searches -----------------------------------------------------------------------------------------------------
    # The search matrix: random, categorical and shared-prefix text; sorted and unsorted haystacks (NumPy's exact path);
    # random, sorted and descending keys; a sorter; reversed and stepped haystack views. Sizes cross NumSharp's cost rule
    # (few keys against a long haystack compare strings; many keys key the haystack).
    for kd in ("U", "S"):
        for flavor in ("rand", "cat", "prefix"):
            for n, m in ((1, 1), (1, 5), (2, 5), (5, 5), (5, 100), (17, 1), (17, 100), (100, 1), (100, 5), (100, 120),
                         (300, 200), (1000, 1), (1000, 5), (1000, 100)):
                    if flavor == "rand":
                        hv, kv, width = _rand_strings(rng, n, 6, "abcdefg"), _rand_strings(rng, m, 6, "abcdefgh"), 6
                    elif flavor == "cat":
                        pool_h = list(dict.fromkeys(_rand_strings(rng, 20, 4, "abcde")))[:5]
                        pool_k = pool_h[:3] + list(dict.fromkeys(_rand_strings(rng, 20, 4, "abcdef")))[:4]
                        hv, kv, width = _pick(rng, n, pool_h), _pick(rng, m, pool_k), 4
                    else:
                        hv = ["https://site.example/" + v for v in _rand_strings(rng, n, 3, "abc")]
                        kv = ["https://site.example/" + v for v in _rand_strings(rng, m, 3, "abcd")]
                        width = 24
                    hay = _search_text(hv, f"{kd}{width}")
                    keys = _search_text(kv, f"{kd}{width}")
                    srt = plain(np.sort(hay[1]))
                    label = f"{flavor}{kd}n{n}m{m}"
                    for side in ("left", "right"):
                        _search(f"{label}-sorted", srt, keys, side)
                        _search(f"{label}-unsorted", hay, keys, side)
                        if m <= 100:
                            _search(f"{label}-sortedkeys", srt, plain(np.sort(keys[1])), side)
                            _search(f"{label}-desckeys", hay, plain(np.sort(keys[1])[::-1].copy()), side)
                        _search(f"{label}-sorter", hay, keys, side, num_operand(np.argsort(hay[1], kind="stable")))
                    if n >= 5:
                        _search(f"{label}-rev", (srt[0], srt[1][::-1]), keys, "left")
                        _search(f"{label}-step2", (srt[0], srt[1][::2]), keys, "right")
    # Few keys against a long haystack (the strings compared directly), sorted and not, both sides.
    for kd, width, alpha in (("U", 8, "abcdefghij"), ("S", 8, "abcdefghij"), ("U", 1, "abcd"), ("U", 24, None)):
        if alpha is None:
            hv = ["https://site.example/" + v for v in _rand_strings(rng, 1200, 3, "abc")]
            kv = ["https://site.example/" + v for v in _rand_strings(rng, 9, 3, "abcd")]
        else:
            hv, kv = _rand_strings(rng, 1200, width, alpha), _rand_strings(rng, 9, width, alpha)
        hay, keys = _search_text(hv, f"{kd}{width}"), _search_text(kv, f"{kd}{width}")
        for side in ("left", "right"):
            _search(f"few{kd}{width}-sorted", plain(np.sort(hay[1])), keys, side)
            _search(f"few{kd}{width}-unsorted", hay, keys, side)
            _search(f"few{kd}{width}-sorter", hay, keys, side, num_operand(np.argsort(hay[1], kind="stable")))

    # Keys that end before the strings do (U with a BMP alphabet keys 4 of 5 units; S16 / S32 key 8 bytes), with ties in
    # the haystack past the key: the sorted haystack's checks and the searches compare the units after the keys.
    for kd, width, alpha in (("U", 5, "aäā中文"), ("S", 16, "ab"), ("S", 32, "abc")):
        hv = _rand_strings(rng, 700, width, alpha, minlen=width - 1)
        hv += [hv[i][:8] + "a" * (width - 8) for i in range(0, 200, 7)]
        kv = _rand_strings(rng, 300, width, alpha, minlen=width - 1) + hv[:50]
        hay, keys = _search_text(hv, f"{kd}{width}"), _search_text(kv, f"{kd}{width}")
        srt = plain(np.sort(hay[1]))
        for side in ("left", "right"):
            _search(f"ties{kd}{width}-sorted", srt, keys, side)
            _search(f"ties{kd}{width}-sortedkeys", srt, plain(np.sort(keys[1])), side)
            _search(f"ties{kd}{width}-sorter", hay, keys, side, num_operand(np.argsort(hay[1], kind="stable")))
            _search(f"ties{kd}{width}-unsorted", hay, keys, side)

    # Sorter values reached and not reached by the search, every integer kind, and the refused sorters.
    hay = own(np.array(["d", "a", "c", "b", "e"], dtype="U1"))
    for sname, srt in (("ok", [1, 3, 2, 0, 4]), ("oob-mid", [1, 3, 9, 0, 4]), ("oob-edge", [1, 3, 2, 0, 9]),
                       ("neg", [1, -1, 2, 0, 4]), ("neg-first", [-1, 3, 2, 0, 4]), ("dup", [1, 1, 1, 1, 1])):
        for dt in (np.int64, np.int32, np.int8, np.uint16):
            if np.any(np.array(srt) < 0) and np.dtype(dt).kind == "u":
                continue
            for kv in (["c"], ["a"], ["z"], ["b", "e", "a"]):
                for side in ("left", "right"):
                    _search(f"sorter-{sname}-{np.dtype(dt).name}-{''.join(kv)}", hay, _search_text(kv, "U1"), side,
                            num_operand(np.array(srt, dtype=dt)))
    for sname, srt in (("uint64", np.array([1, 3, 2, 0, 4], np.uint64)), ("float", np.array([1.0, 3, 2, 0, 4])),
                       ("bool", np.array([True, False, True, False, True])), ("2d", np.array([[1, 3, 2, 0, 4]])),
                       ("size", np.array([1, 0])), ("0d", np.array(2))):
        _search(f"sorter-bad-{sname}", hay, _search_text(["c"], "U1"), "left", num_operand(srt))

    # Widths, kinds and numbers: mixed widths (the narrower zero-padded), shared prefixes reaching the narrower width,
    # NULs, S with U (bytes decoded; a non-ASCII byte refused), numbers with text both ways.
    _search("width-hay3-keys8", plain(np.sort(np.array(_rand_strings(rng, 200, 3, "abc"), dtype="U3"))),
            _search_text(_rand_strings(rng, 50, 8, "abc"), "U8"))
    _search("width-hay8-keys2", plain(np.sort(np.array(_rand_strings(rng, 200, 8, "abc"), dtype="U8"))),
            _search_text(_rand_strings(rng, 50, 2, "abc"), "U2"))
    _search("width-S-hay3-keys8", plain(np.sort(np.array([v.encode() for v in _rand_strings(rng, 200, 3, "abc")],
                                                          dtype="S3"))),
            _search_text(_rand_strings(rng, 50, 8, "abc"), "S8"), "right")
    pre_h = ["https://site.example/" + v for v in _rand_strings(rng, 300, 3, "abc")]
    for kname, kvals, kw in (("exact-prefix", ["https://site.example/"] * 7, 21),
                             ("short", ["https"] * 3 + ["https://site"] * 4, 12),
                             ("longer", ["https://site.example/" + v for v in _rand_strings(rng, 40, 6, "abcd")], 27)):
        for kd in ("U", "S"):
            hay = _search_text(pre_h, f"{kd}24")
            keys = _search_text(kvals, f"{kd}{kw}")
            for side in ("left", "right"):
                _search(f"prefix-width-{kname}{kd}-sorted", plain(np.sort(hay[1])), keys, side)
                _search(f"prefix-width-{kname}{kd}-unsorted", hay, keys, side)
                _search(f"prefix-width-{kname}{kd}-swapped", plain(np.sort(keys[1])), hay, side)
    _search("nul-U", plain(np.sort(np.array(["a", "a\x00b", "a\x00", "b", "", "a\x00\x00c"] * 5, dtype="U4"))),
            plain(np.array(["a", "a\x00", "a\x00b", "a\x00a", "\x00", ""], dtype="U3")))
    _search("nul-S", plain(np.sort(np.array([b"a", b"a\x00b", b"\xff", b"\x00", b""] * 5, dtype="S3"))),
            plain(np.array([b"a\x00", b"\xfe", b"\xff\x00", b""], dtype="S2")))
    _search("kinds-S-hay-U-keys", plain(np.sort(np.array([b"ab", b"b", b"cd"], dtype="S2"))),
            plain(np.array(["b", "ba", "c", ""], dtype="U2")))
    _search("kinds-U-hay-S-keys", plain(np.sort(np.array(["ab", "b", "cd"], dtype="U2"))),
            plain(np.array([b"b", b"ba", b"c", b""], dtype="S2")), "right")
    _search("kinds-U-hay-S-nonascii", plain(np.array(["ab", "b"], dtype="U2")), plain(np.array([b"\xff"], dtype="S1")))
    _search("kinds-S-hay-nonascii-U-keys", plain(np.array([b"a", b"\xff"], dtype="S1")), plain(np.array(["a"], dtype="U1")))
    _search("num-int-key", plain(np.array(["1", "10", "2", "5"], dtype="U2")), num_operand(np.array(3)))
    _search("num-int-keys", plain(np.array(["1", "10", "2", "5"], dtype="U2")), num_operand(np.array([3, 10, 100, -1])))
    _search("num-float-keys", plain(np.array(["1.5", "2", "2.25"], dtype="U4")), num_operand(np.array([1.75, 2.0, -0.5])))
    _search("num-text-keys-int-hay", num_operand(np.array([1, 5, 9, 12])), plain(np.array(["3", "10", "9"], dtype="U2")))
    _search("num-text-keys-float-hay", num_operand(np.array([0.5, 1.5, 2.5])),
            plain(np.array(["1", "1.5", "2.0"], dtype="U3")), "right")
    _search("num-S-hay-int-key", plain(np.array([b"1", b"10", b"2"], dtype="S2")), num_operand(np.array([3, 11])))
    _search("num-bool-key", plain(np.array(["False", "True"], dtype="U5")), num_operand(np.array([True, False])))

    # Keys of every shape and layout, and haystack views.
    hay = plain(np.sort(np.array(_rand_strings(rng, 300, 4, "abcd"), dtype="U4")))
    k2 = np.array(_rand_strings(rng, 60, 4, "abcde"), dtype="U4").reshape(6, 10)
    _search("keys-2d", hay, plain(k2))
    k2f = np.ascontiguousarray(k2.T).reshape(-1).copy()
    _search("keys-2d-F", hay, (k2f, k2f.reshape(10, 6).T))
    _search("keys-0d", hay, plain(np.array("bcd", dtype="U4")))
    _search("keys-0d-right", hay, plain(np.array("bcd", dtype="U4")), "right")
    kb = np.array(_rand_strings(rng, 30, 4, "abcde"), dtype="U4")
    _search("keys-strided", hay, (kb, kb[::3]))
    _search("keys-reversed", hay, (kb, kb[::-1]))
    _search("keys-3d", hay, plain(np.array(_rand_strings(rng, 24, 4, "abcde"), dtype="U4").reshape(2, 3, 4)))
    _search("keys-empty", hay, plain(np.zeros(0, "U4")))
    _search("keys-empty-2d", hay, plain(np.zeros((2, 0), "U4")))
    _search("hay-empty", plain(np.zeros(0, "U4")), plain(np.array(["a", "b"], dtype="U1")))
    _search("hay-empty-0d", plain(np.zeros(0, "U4")), plain(np.array("a", dtype="U1")))
    _search("hay-one", plain(np.array(["m"], dtype="U1")), plain(np.array(["a", "m", "z"], dtype="U1")), "right")
    big = np.sort(np.array(_rand_strings(rng, 600, 5, "abcdef"), dtype="U5"))
    for vname, build in (("rev", lambda b: b[::-1]), ("step3", lambda b: b[::3]), ("offset", lambda b: b[100:500])):
        for side in ("left", "right"):
            _search(f"hay-{vname}", (big, build(big)), plain(np.array(_rand_strings(rng, 80, 5, "abcdefg"), dtype="U5")),
                    side)

    # Every U packing and units above U+10FFFF.
    for name, alpha in alphabets.items():
        hv = np.sort(np.array(_rand_strings(rng, 400, 5, alpha), dtype="U5"))
        for side in ("left", "right"):
            _search(f"pack-{name}", plain(hv), plain(np.array(_rand_strings(rng, 300, 5, alpha), dtype="U5")), side)
            _search(f"pack-{name}-unsorted", plain(np.array(_rand_strings(rng, 400, 5, alpha), dtype="U5")),
                    plain(np.array(_rand_strings(rng, 300, 5, alpha), dtype="U5")), side)
    _search("units-above-10ffff", plain(np.sort(hi)), plain(hi[:40].copy()))

    # The method form, and the argument errors in NumPy's order: the side, the keys' conversion, the haystack's rank, the
    # sorter.
    hay = plain(np.array(["a", "b", "c"], dtype="U1"))
    for side in ("left", "right"):
        _search("method", hay, plain(np.array(["b", "bb", ""], dtype="U2")), side, method=True)
    _search("method-sorter", own(np.array(["c", "a", "b"], dtype="U1")), plain(np.array(["b"], dtype="U1")), "left",
            num_operand(np.array([1, 2, 0])), method=True)
    for side in ("lft", "Left", "R", "", "LEFT", "x", "rightish"):
        _search(f"side-{side or 'empty'}", hay, plain(np.array(["b"], dtype="U1")), side)
    _search("side-before-sorter", hay, plain(np.array(["b"], dtype="U1")), "bad", num_operand(np.array([[1]])))
    _search("hay-2d", plain(np.array([["a", "b"], ["c", "d"]], dtype="U1")), plain(np.array(["b"], dtype="U1")))
    _search("hay-0d", plain(np.array("a", dtype="U1")), plain(np.array(["b"], dtype="U1")))
    _search("hay-2d-keys-nonascii", plain(np.array([["a", "b"]], dtype="U1")), plain(np.array([b"\xff"], dtype="S1")))

    # ---- edges a mutation pass found unguarded --------------------------------------------------------------------------
    # Short rows whose keys tie before the row's end (the 9th unit decides); units that differ in a HIGH byte (U+0101 against
    # U+0201) with the next units ordered the other way, as block winners and as search comparisons; a haystack sorted but
    # for one descent at every position modulo 8; equal 8-unit keys whose tails descend; keys one unit short of the width.
    for kd in ("U", "S"):
        rows = np.array(["aaaaaaaab", "aaaaaaaac", "a", "zzzzzzzzb", "zzzzzzzza", "b", "m", "aaaaaaaaz", "aaaaaaaay"],
                        dtype=f"{kd}9").reshape(3, 3)
        _argred(f"rowties{kd}", plain(rows), [1, 0, None])
    hb = ["a"] * 2000
    hb[10] = "āz"
    hb[1500] = "ȁa"
    _argred("highbyte-blocks", plain(np.array(hb)), [None])
    _search("highbyte", plain(np.array(["āz", "ȁa"])), plain(np.array(["ȁ\x01", "ā\x7f", "ǿz"])))
    base = sorted(set(_rand_strings(rng, 80, 4, "abcdefgh", minlen=4)))
    dkeys = _rand_strings(rng, 120, 4, "abcdefgh", minlen=4)
    for pos in range(0, 18):
        hay = list(base)
        hay[pos], hay[pos + 1] = hay[pos + 1], hay[pos]
        _search(f"one-descent-{pos}", plain(np.array(hay, dtype="U4")), plain(np.array(dkeys, dtype="U4")))
    for kd in ("U", "S"):
        hay = _search_text(["aaaaaaaa", "bbbbbbbbz", "bbbbbbbba", "cccccccc", "dddddddd"], f"{kd}12")
        keys = _search_text(["bbbbbbbbm", "bbbbbbbbz", "c", "bbbbbbbba", "e"], f"{kd}12")
        for side in ("left", "right"):
            _search(f"desc-tails{kd}", hay, keys, side)
        short_h = ["aaaaaaaa" + c for c in rng.choice(list("abc"), 300)]
        short_k = ["aaaaaaaa" + c for c in rng.choice(list("abcd"), 100)]
        for side in ("left", "right"):
            _search(f"one-unit-short{kd}-sorted", _search_text(sorted(short_h), f"{kd}9"), _search_text(short_k, f"{kd}9"), side)
            _search(f"one-unit-short{kd}-sortedkeys", _search_text(sorted(short_h), f"{kd}9"),
                    _search_text(sorted(short_k), f"{kd}9"), side)
            _search(f"one-unit-short{kd}-unsorted", _search_text(["b"] + short_h[1:], f"{kd}9"), _search_text(short_k, f"{kd}9"),
                    side)

    # ---- digitize ----------------------------------------------------------------------------------------------------
    xs_u = plain(np.array([str(v) for v in rng.integers(-5, 120, 200)], dtype="U4"))
    xs_s = plain(np.array([str(v).encode() for v in rng.integers(-5, 120, 200)], dtype="S4"))
    for bname, bins in (("inc", np.arange(0, 100, 10)), ("dec", np.arange(100, 0, -10)), ("rep", np.array([1, 1, 5, 5, 20])),
                        ("float", np.array([0.5, 2.25, 10.0, 99.5])), ("one", np.array([7])), ("neg", np.array([-3, 0, 3])),
                        ("big", np.arange(0, 1000, 7)), ("rep-dec", np.array([20, 5, 5, 1, 1]))):
        for right in (False, True):
            for xname, x in (("U", xs_u), ("S", xs_s)):
                add("strsearch", "digitize", params={"right": right}, operands=[x, num_operand(bins)],
                    layout=f"dig-{xname}-{bname}", call=lambda xx, b, r=right: np.digitize(xx, b, right=r))
    for xname, x in (("2d", plain(np.array([str(v) for v in rng.integers(0, 100, 20)], dtype="U3").reshape(4, 5))),
                     ("0d", plain(np.array("42", dtype="U2"))), ("empty", plain(np.zeros(0, "U2"))),
                     ("empty-2d", plain(np.zeros((0, 3), "U2"))), ("words", plain(np.array(["b", "a", "zz", ""], dtype="U2")))):
        for right in (False, True):
            add("strsearch", "digitize", params={"right": right}, operands=[x, num_operand(np.arange(0, 100, 10))],
                layout=f"dig-{xname}", call=lambda xx, b, r=right: np.digitize(xx, b, right=r))
    for xname, x, bins in (("text-bins", num_operand(np.array([1.5, 3.0])), plain(np.array(["1", "2"], dtype="U1"))),
                           ("text-both", plain(np.array(["1"], dtype="U1")), plain(np.array(["1", "2"], dtype="U1"))),
                           ("nonmono", plain(np.array(["1"], dtype="U1")), num_operand(np.array([3, 1, 2]))),
                           ("bins-2d", plain(np.array(["1"], dtype="U1")), num_operand(np.array([[1, 2]])))):
        add("strsearch", "digitize", params={"right": False}, operands=[x, bins], layout=f"dig-{xname}",
            call=lambda xx, b: np.digitize(xx, b))

    # ---- the bin search of histogram2d / histogramdd over a text sample whose columns differ in kind ------------------
    # histogramdd's first statement stacks the sample (atleast_2d(sample).T): bytes beside str are decoded there, so a byte
    # that is not ASCII raises UnicodeDecodeError before a bad bin count, decreasing edges or good edges are read; bytes
    # beside bytes or numbers stay bytes. Otherwise the stacked text is searched in the numeric edges converted to text.
    hist_samples = {
        "Sx-U": (np.array([b"a\xff", b"b"], dtype="S2"), np.array(["c", "d"], dtype="U1")),
        "Sascii-U": (np.array([b"10", b"3"], dtype="S2"), np.array(["1.5", "0.5"], dtype="U3")),
        "U-Sx": (np.array(["1.5", "0.5"], dtype="U3"), np.array([b"a\xff", b"b"], dtype="S2")),
        "Sx-S": (np.array([b"a\xff", b"10"], dtype="S2"), np.array([b"1.5", b"0.5"], dtype="S3")),
        "Sx-F": (np.array([b"\xff", b"10"], dtype="S2"), np.array([1.5, 0.5])),
    }
    hist_bins_specs = {"2": (2, None), "2-range": (2, (0.0, 100.0)), "edges": (np.array([0.0, 1.0, 2.0, 30.0]), None),
                       "dec": (np.array([30.0, 2.0, 1.0, 0.5]), None)}
    for fn in ("histogram2d", "histogramdd"):
        for sk, (sx, sy) in hist_samples.items():
            for bk, (bins, rng_) in hist_bins_specs.items():
                ops = [plain(sx), plain(sy) if sy.dtype.kind in "SU" else num_operand(sy)]
                params = {"fn": fn, "range": _json_floats(rng_) if rng_ else None}
                if isinstance(bins, np.ndarray):
                    params["bins_op"] = len(ops)
                    ops.append(num_operand(bins))
                else:
                    params["bins"] = bins

                def call_hist(*views, p=params, r=rng_):
                    x, yy = views[0], views[1]
                    b = views[p["bins_op"]] if "bins_op" in p else p["bins"]
                    if p["fn"] == "histogram2d":
                        return np.histogram2d(x, yy, b, (r, r) if r else None)
                    bb = [b, b] if isinstance(b, np.ndarray) and b.ndim == 1 else b
                    H, edges = np.histogramdd([x, yy], bb, (r, r) if r else None)
                    return (H,) + tuple(edges)
                add("strsearch", "hist", params=params, operands=ops, layout=f"{fn}:sample={sk}:bins={bk}", call=call_hist)


# ----------------------------------------------------------------------------------------------------------------------
# Section strchararray: numpy.char.chararray as data (the numpy.char plan's phase 2)
# ----------------------------------------------------------------------------------------------------------------------

CHA = "strchararray"

# str.rstrip() with no argument removes the code points str.isspace() accepts: 29 in CPython 3.12 (Unicode 15.0).
# chararray.__getitem__ strips a str element with them (Python's own method), never with np.char's ASCII set.
PY_WS = [chr(c) for c in range(0x110000) if chr(c).isspace()]
assert len(PY_WS) == 29, "CPython's whitespace set changed: NDCharArray's strip table must follow it"
# bytes.rstrip() removes six ASCII bytes: \t \n \v \f \r and the space.
BYTES_WS = [b for b in range(256) if bytes([b]).isspace()]
assert len(BYTES_WS) == 6
# Characters rstrip() keeps although they print like whitespace or are controls, the NUL (kept unless it is trailing), a
# letter and a supplementary-plane character.
PY_KEEP = ["\u200b", "\u180e", "\ufeff", "\u2060", "\x01", "\x08", "\x7f", "\x00", "a", "\U0001F600"]

# The element pools of the chararray operands: every whitespace class in trailing position, kept look-alikes, NULs before
# and after whitespace (rstrip keeps the 'a\x00' of 'a\x00 '), leading whitespace, empty values, non-ASCII and
# supplementary-plane text. Values longer than an operand's width are truncated by np.array, as NumPy stores them.
CHA_U = ["ab  ", "c\t", " d ", "e\x1c", "f\xa0", "g\u3000", "", "h", "x  ", "y \u3000 ", "\U0001F600 ", "a\x00 ",
         "a \x00", "\u200b", "z\u180e", "q\ufeff", "  ", "\t\n", "\xe9\x85", "w\u2028", "m\u205f ", "n\u1680",
         "k\x1f\x1e\x1d", "p\r\x0c\x0b", "   ", "s     ", "\u4e2d\u6587 "]
CHA_S = [b"ab  ", b"c\t", b" d ", b"e\x1c", b"f\xa0", b"g\x85", b"", b"h", b"x\r\n", b"\x0b\x0c", b"a\x00 ", b"a \x00",
         b"\xff ", b"q\x01", b"  ", b"\x1d\x1e\x1f", b"t\x0b"]


class _ChaText(str):
    """A printed result (``repr`` / ``str`` of a chararray), recorded as ``{"kind": "text"}``."""


class _WithSelf:
    """A result together with the object the call was made on, so the case records NumPy's ``result is receiver``
    (``astype(copy=False)``, ``real``, ``squeeze()``, ``to_device``, ``np.char.asarray`` of a chararray)."""

    def __init__(self, result, receiver):
        """Keeps both.

        Args:
            result: the call's result.
            receiver: the object whose identity the result is compared with (None: no identity recorded).
        """
        self.result, self.receiver = result, receiver


def _cha_operand(spec, layout, seed=0):
    """A (base, view) operand of ``spec`` in one of LAYOUTS, its base cycling through the chararray pool from ``seed``.

    Args:
        spec: the dtype, ``"<U4"`` / ``"|S4"``.
        layout: a LAYOUTS key.
        seed: the first pool index.

    Returns:
        The (base, view) pair.
    """
    n, build = LAYOUTS[layout]
    pool = CHA_U if "U" in spec else CHA_S
    base = np.array([pool[(seed + i) % len(pool)] for i in range(n)], dtype=spec) if n else np.zeros(0, dtype=spec)
    return base, build(base)


def _fview(arr):
    """An F-contiguous operand equal to ``arr`` (2-D): a C-contiguous base holding ``arr`` in F order and its transposed
    view (the replay rebuilds a view over its base buffer, so the layout survives).

    Args:
        arr: a 2-D array.

    Returns:
        The (base, view) pair.
    """
    base = np.ascontiguousarray(arr.T).reshape(-1).copy()
    return base, base.reshape(arr.shape[::-1]).T


def _cha_enc(v):
    """``enc`` plus what np.char.array's Python arguments add: a complex, and a Python int past int64.

    Args:
        v: the Python value (lists element by element).

    Returns:
        The JSON-ready encoding.
    """
    if isinstance(v, complex):
        return {"complex": [_jf(v.real), _jf(v.imag)]}
    if isinstance(v, int) and not isinstance(v, bool) and not (-2 ** 63 <= v < 2 ** 63):
        return {"bigint": str(v)}
    if isinstance(v, (list, tuple)):
        return [_cha_enc(x) for x in v]
    return enc(v)


def _cha_expect(r, views, receiver=None, value_kind=False):
    """The expectation of a chararray call's result.

    Args:
        r: NumPy's result.
        views: the case's operand views (for memory sharing).
        receiver: the object the call was made on, or None.
        value_kind: True for a Python value (``item()`` / ``tolist()``), compared by value with Python's integer and
            float semantics (``pyvalue``).

    Returns:
        ``text`` for a printed string; ``pyvalue``; ``array`` with its CLASS (``chararray`` / ``ndarray`` / ``matrix``),
        its OWNDATA read from the result itself (``array_expect`` reads a base-class view of a subclass) and, given a
        receiver, ``self`` (NumPy's identity); ``items`` for a list (iteration); ``element`` for a scalar: its Python /
        NumPy type name (``str`` is stripped by ``__getitem__``, ``str_`` is NumPy's unstripped scalar, ``uint32`` a
        number) and its value.

    Raises:
        TypeError: a result kind the section does not record.
    """
    if isinstance(r, _ChaText):
        return {"kind": "text", "u": str(r).encode("utf-32-le", "surrogatepass").hex()}
    if value_kind:
        return {"kind": "pyvalue", "value": enc(r)}
    if isinstance(r, np.ndarray):
        d = array_expect(r, views)
        d["owndata"] = bool(r.flags.owndata)
        d["class"] = type(r).__name__
        if receiver is not None:
            d["self"] = r is receiver
        return d
    if isinstance(r, list):
        return {"kind": "items", "items": [_cha_expect(x, views) for x in r]}
    if isinstance(r, (str, bytes, np.generic)):
        return {"kind": "element", "py": type(r).__name__, "value": enc(r)}
    raise TypeError(f"unexpected chararray result {type(r)}")


def _cha_run(call, views, value_kind=False, uninit=False):
    """Runs one call and returns its outcome.

    Args:
        call: the NumPy call, taking the operand views.
        views: the operand views.
        value_kind: see ``_cha_expect``.
        uninit: the result's bytes are uninitialized (an ``S`` chararray without a buffer): its buffer is recorded as
            None (not compared).

    Returns:
        ``{"expected": ...}`` or ``{"error": ...}``.
    """
    try:
        r = call(*views)
        receiver = None
        if isinstance(r, _WithSelf):
            r, receiver = r.result, r.receiver
        exp = _cha_expect(r, views, receiver, value_kind)
        if uninit and exp.get("kind") == "array":
            exp["buffer"] = None
        return {"expected": exp}
    except Exception as e:  # noqa: BLE001 -- every NumPy exception is a recorded outcome
        return {"error": err_info(e)}


# The cases whose outcome a fresh process records (``add_cha(..., isolate=payload)``): NumPy 2.4.2 corrupts its heap when
# a chararray's __array_finalize__ raises on a path that ALLOCATED the result first (astype / copy / flatten / byteswap /
# imag / a flat slice of a chararray over numbers): the failure frees the buffer through a bad handler capsule
# ("Exception ignored in: 'PyDataMem_UserFREE'") and a later call crashes the process.
_CHA_PENDING = []


def add_cha(op, *, params=None, operands=(), call=None, isolate=None, layout="", value_kind=False, uninit=False):
    """Records one chararray case.

    Args:
        op: the corpus op key.
        params: the case parameters.
        operands: (base, view) pairs.
        call: the NumPy call (``call(*views)``), when ``isolate`` is None.
        isolate: a ``_cha_member_outcome`` payload: the outcome is recorded by a fresh process (``_cha_flush``).
        layout: the case's layout label.
        value_kind: see ``_cha_expect``.
        uninit: see ``_cha_run``.
    """
    views = [v for _, v in operands]
    prefix = f"{CHA}/{op}/"
    n = _OP_COUNT.get(prefix, 0) + 1
    _OP_COUNT[prefix] = n
    cid = f"{CHA}/{op}/{n}"
    assert cid not in _IDS
    _IDS.add(cid)
    case = {"id": cid, "op": op, "params": params or {}, "operands": [describe(b, v) for b, v in operands],
            "phase": 1, "layout": layout}
    if isolate is not None:
        _CHA_PENDING.append((case, isolate))
    else:
        case.update(_cha_run(call, views, value_kind, uninit))
    CASES.append(case)


def _cha_arg_value(a):
    """One JSON argument of a member call as the Python value NumPy receives.

    Args:
        a: ``{"dtype": s}`` (a dtype string, or ``ndarray`` / ``chararray`` / ``matrix`` for the classes),
            ``{"type": "int" | "str"}`` (a Python type), ``{"list": [...]}`` (a list), or a JSON scalar (itself).

    Returns:
        The Python value.
    """
    if isinstance(a, dict):
        if "dtype" in a:
            return {"ndarray": np.ndarray, "chararray": np.char.chararray, "matrix": np.matrix}.get(a["dtype"], a["dtype"])
        if "type" in a:
            return {"int": int, "str": str}[a["type"]]
        if "list" in a:
            return a["list"]
    return a


def _cha_member_call(name, args):
    """A member call of section E by name.

    Args:
        name: the member: a method name, a property (``T`` / ``mT`` / ``real`` / ``imag``) or a flat-iterator access
            (``flat_index`` / ``flat_slice`` / ``flat_fancy`` / ``flat_copy``).
        args: its positional arguments as JSON, an optional trailing ``{"kw": {...}}`` holding keywords.

    Returns:
        A function of the receiver chararray that makes the call.
    """
    kw = {}
    if args and isinstance(args[-1], dict) and "kw" in args[-1]:
        kw = {k: _cha_arg_value(v) for k, v in args[-1]["kw"].items()}
        args = args[:-1]
    real = [_cha_arg_value(a) for a in args]
    if name in ("T", "mT", "real", "imag"):
        return lambda c: getattr(c, name)
    if name in ("flat_index", "flat_fancy"):
        return lambda c: c.flat[real[0]]
    if name == "flat_slice":
        return lambda c: c.flat[slice(*real)]
    if name == "flat_copy":
        return lambda c: c.flat.copy()
    return lambda c: getattr(c, name)(*real, **kw)


def _cha_receiver(view, recv):
    """The chararray a call is made on.

    Args:
        view: the operand view.
        recv: None, or a dtype the chararray is viewed as first (``view(np.uint32)`` keeps the class: NumPy's finalize
            reads the string dtype the array has, then the dtype changes).

    Returns:
        The chararray.
    """
    c = view.view(np.char.chararray)
    return c.view(recv) if recv else c


def _cha_member_outcome(payload):
    """Rebuilds a member case's operand, makes the call and records its outcome (run by ``_cha_flush``'s worker
    processes).

    Args:
        payload: ``{"spec", "layout", "seed", "name", "args", "recv"}``.

    Returns:
        ``{"expected": ...}`` or ``{"error": ...}``.
    """
    _, view = _cha_operand(payload["spec"], payload["layout"], payload["seed"])
    member = _cha_member_call(payload["name"], payload["args"])

    def call(v):
        c = _cha_receiver(v, payload["recv"])
        return _WithSelf(member(c), c)

    return _cha_run(call, [view], payload["name"] in ("item", "tolist"))


# The worker of an isolated case: imports this module, records the outcome, writes it as one JSON line and leaves without
# the interpreter's teardown (where a heap NumPy corrupted would crash after the line is written).
_CHA_WORKER = ("import json, os, sys\n"
               "sys.path.insert(0, sys.argv[1])\n"
               "import gen_strings_oracle as g\n"
               "out = g._cha_member_outcome(json.loads(sys.stdin.read()))\n"
               "sys.stdout.write(json.dumps(out, allow_nan=False) + '\\n')\n"
               "sys.stdout.flush()\n"
               "os._exit(0)\n")


def _cha_isolated(payload):
    """Records one isolated case in a fresh Python process.

    Args:
        payload: the ``_cha_member_outcome`` payload.

    Returns:
        The outcome.

    Raises:
        AssertionError: the worker wrote no outcome (it crashed during the call itself).
    """
    import subprocess
    out = subprocess.run([sys.executable, "-I", "-c", _CHA_WORKER, str(HERE)], input=json.dumps(payload),
                         capture_output=True, text=True, encoding="utf-8")
    lines = out.stdout.strip().splitlines()
    assert lines and lines[-1].startswith("{"), \
        f"isolated chararray case {payload} wrote no outcome: rc={out.returncode} {out.stderr[-600:]}"
    return json.loads(lines[-1])


def _cha_flush():
    """Records the outcome of every pending isolated case, eight worker processes at a time (each case alone in its
    process); the cases keep their place and ids."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        outcomes = list(ex.map(lambda cp: _cha_isolated(cp[1]), _CHA_PENDING))
    for (case, _), outcome in zip(_CHA_PENDING, outcomes):
        case.update(outcome)
    _CHA_PENDING.clear()


def add_cha_member(spec, layout, seed, name, args, recv=None, isolate=False, values=None):
    """Records one member call of section E on a chararray view of an operand.

    Args:
        spec: the operand's dtype.
        layout: a LAYOUTS key (ignored with ``values``).
        seed: the operand's first pool index.
        name: see ``_cha_member_call``.
        args: see ``_cha_member_call``.
        recv: see ``_cha_receiver``.
        isolate: True to record the outcome in a fresh process (a call that can raise the finalize error after
            allocating); not with ``values``.
        values: explicit base values (a 1-D operand), instead of the pool.
    """
    if values is not None:
        operand_pair = plain(np.array(values, dtype=spec))
        label = f"{spec}:values"
    else:
        operand_pair = _cha_operand(spec, layout, seed)
        label = f"{spec}:{layout}" + (f":{recv}" if recv else "")
    params = {"name": name, "args": args, "recv": recv}
    if isolate:
        assert values is None
        payload = {"spec": spec, "layout": layout, "seed": seed, "name": name, "args": args, "recv": recv}
        add_cha("cha_member", params=params, operands=[operand_pair], isolate=payload, layout=label)
        return
    member = _cha_member_call(name, args)

    def call(v):
        c = _cha_receiver(v, recv)
        return _WithSelf(member(c), c)

    add_cha("cha_member", params=params, operands=[operand_pair], call=call, layout=label,
            value_kind=name in ("item", "tolist"))


def _cha_index_py(spec, views):
    """A JSON index as the Python index.

    Args:
        spec: entries ``{"i": n}`` (an int), ``{"s": [start, stop, step]}`` (a slice), ``{"e": 1}`` (Ellipsis), ``{"n": 1}``
            (None, a new axis), ``{"b": bool}`` (a bool), ``{"a": [...]}`` (an int64 index array), ``{"z": n}`` (a 0-d
            int64 array), ``{"m": k}`` (the k-th operand, a bool mask).
        views: the case's operand views.

    Returns:
        The index tuple.
    """
    out = []
    for e in spec:
        if "i" in e:
            out.append(e["i"])
        elif "s" in e:
            out.append(slice(*e["s"]))
        elif "e" in e:
            out.append(Ellipsis)
        elif "n" in e:
            out.append(None)
        elif "b" in e:
            out.append(e["b"])
        elif "a" in e:
            out.append(np.array(e["a"], dtype=np.int64))
        elif "z" in e:
            out.append(np.array(e["z"], dtype=np.int64))
        elif "m" in e:
            out.append(views[e["m"]])
    return tuple(out)


def _cha_basic(spec):
    """Whether an index has at least one entry and holds only ints, slices, Ellipsis and new axes (the C#
    slice-notation and Slice[] indexers take it).

    Args:
        spec: the JSON index.

    Returns:
        The answer.
    """
    return bool(spec) and all(("i" in e) or ("s" in e) or ("e" in e) or ("n" in e) for e in spec)


def add_cha_item(base, view, index, layout, extra_ops=(), recv=None):
    """Records ``c[index]`` for the operand viewed as a chararray, once per C# spelling of the index: the general
    ``object[]`` indexer always, the slice-notation string and ``Slice[]`` indexers when the index is basic.

    Args:
        base: the operand's base.
        view: the operand view.
        index: the JSON index (``_cha_index_py``).
        layout: the layout label.
        extra_ops: more (base, view) operands an index entry names (a mask).
        recv: see ``_cha_receiver``.
    """
    forms = ["objects"] + (["notation", "slices"] if _cha_basic(index) else [])
    for form in forms:
        def call(v, *rest, index=index):
            return _cha_receiver(v, recv)[_cha_index_py(index, [v] + list(rest))]

        add_cha("cha_item", params={"index": index, "form": form, "recv": recv},
                operands=[(base, view)] + list(extra_ops), call=call, layout=layout)


def _cha_array_cases():
    """Section B: np.char.array / np.char.asarray over every argument kind NumPy's ``array`` branches on."""

    def case(op, obj_spec, make_obj, operands, label, **kw):
        """Records ``np.char.array / asarray(make_obj(*views), **kw)``, the None keywords left out (NumPy's defaults)."""
        given = {k: v for k, v in kw.items() if v is not None}
        params = {"obj": obj_spec, **given}
        f = np.char.array if op == "cha_array" else np.char.asarray

        def call(*vs):
            obj = make_obj(*vs)
            return _WithSelf(f(obj, **given), obj if isinstance(obj, np.ndarray) else None)

        add_cha(op, params=params, operands=operands, call=call, layout=label)

    for op in ["cha_array", "cha_asarray"]:
        copies = [None, False] if op == "cha_array" else [None]
        # A str / bytes: itemsize defaults to its length, the shape is len // itemsize (a ZeroDivisionError at 0), a str
        # FILLS the chararray, bytes are its buffer (read-only).
        for text in ["abcdef", "", "ab  ", "\xe9", "\U0001F600xy", "a\x00b", "  "]:
            for itemsize in [None, 1, 2, 4, 7, 0, -1]:
                for unicode in [None, False, True]:
                    for order in [None, "F", "Q"]:
                        case(op, {"v": enc(text)}, lambda text=text: text, [], "str", itemsize=itemsize, unicode=unicode,
                             order=order)
        for data in [b"abcd", b"", b"ab \x00", b"\xff\xfe", b"a\x00\x00\x00b\x00\x00\x00"]:
            for itemsize in [None, 1, 2, 3, 0]:
                for unicode in [None, False, True]:
                    for order in [None, "F"]:
                        case(op, {"v": enc(data)}, lambda data=data: data, [], "bytes", itemsize=itemsize,
                             unicode=unicode, order=order)
        # Lists (np.asarray first): strings take the string-array branch, anything else the tolist() route.
        lists = [["hello", "world "], [b"ab", b"c"], ["a", 1], [["a", "bb"], ["c", "d"]], [], [["a"], "b"], [1, 22],
                 [1.5, 2.25, 1e16, -0.0, float("nan"), float("-inf")], [True, False], [b"\xe9"], ["\xe9"],
                 [[1, 2], [3, 4]], [1 + 2j, -0.5j, complex(float("nan"), float("inf"))], [2 ** 63 - 1, -2 ** 63],
                 [b"ab", "c"]]
        for lst in lists:
            for itemsize in [None, 2, 0]:
                for unicode in [None, False, True]:
                    for order in [None, "F"]:
                        for copy in copies:
                            case(op, {"v": _cha_enc(lst)}, lambda lst=lst: lst, [], "list", itemsize=itemsize,
                                 unicode=unicode, order=order, copy=copy)
        # S / U ndarrays of every layout kind: a chararray view, the width in characters, a cast when the width or the
        # class changes (U always: its itemsize counts bytes), np.asarray for an order (NumPy's result is then a plain
        # ndarray: the order quirk, excused by the replay).
        for spec in ["<U3", "|S3"]:
            for layout in ["c1", "c2", "f2", "t3", "s1", "r1", "o1", "bc", "z0", "e1", "e2", "e3", "nx"]:
                for itemsize in [None, 1, 5, 0]:
                    for unicode in [None, False, True]:
                        for order in [None, "C", "F", "K"]:
                            for copy in copies:
                                b, v = operand(spec, layout, seed=3, ascii_only=True)
                                case(op, {"op": 0}, lambda v: v, [(b, v)], f"{spec}:{layout}", itemsize=itemsize,
                                     unicode=unicode, order=order, copy=copy)
        # Numbers and bools: the tolist() route without an itemsize (value-based widths: float32 prints its double
        # value), a cast at the width with one (itemsize 0: the unsized string, int64 -> S21).
        num_arrays = {
            "i8": plain(np.array([1, 22, -333])), "i1": plain(np.array([-128, 5, 127], dtype=np.int8)),
            "u1": plain(np.array([255, 0, 7], dtype=np.uint8)), "u8": plain(np.array([2 ** 64 - 1, 0], dtype=np.uint64)),
            "f4": plain(np.array([0.1, 2, -1e-5], dtype=np.float32)),
            "f2": plain(np.array([0.1, 2, 65504], dtype=np.float16)),
            "f8": plain(np.array([0.1, 1e300, np.nan, -np.inf, -0.0, 1e16])),
            "c16": plain(np.array([1 + 2j, -0.5j, complex(float("nan"), float("inf"))])),
            "b": plain(np.array([True, False])), "i8:2d": plain(np.array([[1, 2], [3, 4]])),
            "i8:F": _fview(np.array([[1, 2], [3, 4]])), "i8:e2": plain(np.zeros((0, 3), dtype=np.int64)),
            "i8:20": plain(np.zeros((2, 0), dtype=np.int64)), "i8:0d": plain(np.array(7)), "f8:0d": plain(np.array(0.5)),
        }
        for label, (b, v) in num_arrays.items():
            for itemsize in [None, 3, 0]:
                for unicode in [None, True]:
                    for order in [None, "F"]:
                        case(op, {"op": 0}, lambda v: v, [(b, v)], label, itemsize=itemsize, unicode=unicode,
                             order=order)
        b, v = plain(np.array([[1, 2], [3, 4]]))
        for itemsize in [None, 3]:
            case(op, {"op": 0, "as": "T"}, lambda v: v.T, [(b, v)], "i8:T", itemsize=itemsize)
        # StringDType: not character data, so the tolist() route (an NA object becomes its text).
        for t_layout in ["c1", "f2", "e1"]:
            for itemsize in [None, 2]:
                for unicode in [None, True]:
                    tb, tv = t_operand(TD, t_layout, seed=1, pool=["ab", "c", "", "def", "x y"], na_every=0)
                    case(op, {"op": 0}, lambda v: v, [(tb, tv)], f"T:{t_layout}", itemsize=itemsize, unicode=unicode)
        tb, tv = t_operand(TD, "c1", seed=0, pool=["\xe9", "a"], na_every=0)
        case(op, {"op": 0}, lambda v: v, [(tb, tv)], "T:nonascii")
        tb, tv = t_operand(TDN, "c1", seed=0, pool=["ab", "c"], na_every=2)
        case(op, {"op": 0}, lambda v: v, [(tb, tv)], "T:na")
        # Chararray inputs: an S chararray is answered ITSELF under copy=False; a U one is always cast.
        for spec in ["|S3", "<U3"]:
            for itemsize in [None, 3, 2]:
                for copy in copies:
                    b, v = operand(spec, "c1", seed=2, ascii_only=True)
                    case(op, {"op": 0, "as": "chararray"}, lambda v: v.view(np.char.chararray), [(b, v)],
                         f"chararray:{spec}", itemsize=itemsize, copy=copy)
        # Python scalars: np.array(obj, dtype=bytes_ / str_), value-based.
        for value in [5, -3, 2.5, -0.0, 1e20, float("nan"), True, False, None, 2 ** 70, 12345]:
            for itemsize in [None, 2, 0]:
                for unicode in [None, True]:
                    case(op, {"v": _cha_enc(value)}, lambda value=value: value, [], "scalar", itemsize=itemsize,
                         unicode=unicode)


# The members of section E (name, args): the views and copies that keep the class, the identities, the finalize errors,
# the Python values and the flat iterator.
CHA_MEMBERS = [("T", []), ("mT", []), ("transpose", []), ("swapaxes", [0, -1]), ("reshape", [-1]),
               ("reshape", [-1, {"kw": {"order": "F"}}]), ("ravel", []), ("ravel", ["F"]), ("flatten", []),
               ("flatten", ["F"]), ("squeeze", []), ("copy", []), ("copy", ["F"]), ("copy", ["K"]),
               ("take", [{"list": [0, -1]}]), ("take", [0]), ("take", [{"list": [[1]]}, {"kw": {"axis": -1}}]),
               ("repeat", [2]), ("repeat", [2, {"kw": {"axis": 0}}]), ("compress", [{"list": [True, False]}]),
               ("diagonal", []), ("real", []), ("imag", []), ("view", []), ("view", [{"dtype": "ndarray"}]),
               ("astype", [{"dtype": "U5"}]), ("astype", [{"dtype": "S6"}]), ("astype", [{"dtype": "U"}]),
               ("byteswap", []), ("item", [0]), ("tolist", []), ("to_device", ["cpu"]), ("flat_index", [0]),
               ("flat_index", [-1]), ("flat_slice", [0, 2]), ("flat_fancy", [{"list": [0, -1]}]), ("flat_copy", [])]


def gen_strchararray():
    """Section strchararray: numpy.char.chararray as data — the constructor, np.char.array / asarray, element access and
    iteration (stripped), the members that keep the class or raise the finalize error, view(type=), printing and the
    flat iterator."""
    # A. The constructor: shapes, widths, both classes, orders; a str buffer (a filler), bytes / bytearray / ndarray
    #    buffers with offsets and strides; NumPy's errors.
    for unicode in (False, True):
        for shape in [3, (), (2, 0), (2, 3), [2, 2], -1, (2, -1)]:
            for itemsize in [1, 3]:
                for order in ["C", "F"]:
                    add_cha("cha_new", params={"shape": shape, "itemsize": itemsize, "unicode": unicode, "order": order},
                            call=lambda shape=shape, itemsize=itemsize, unicode=unicode, order=order:
                            np.char.chararray(shape, itemsize=itemsize, unicode=unicode, order=order),
                            layout="nobuffer", uninit=not unicode)
        for itemsize in [-1, 2 ** 31, 2 ** 29, 2 ** 28, 5]:
            add_cha("cha_new", params={"shape": 0, "itemsize": itemsize, "unicode": unicode},
                    call=lambda itemsize=itemsize, unicode=unicode: np.char.chararray(0, itemsize=itemsize,
                                                                                      unicode=unicode),
                    layout="itemsize")
        for order in ["A", "K", "f", "a", "Q", "c"]:
            add_cha("cha_new", params={"shape": [2, 3], "itemsize": 2, "unicode": unicode, "order": order},
                    call=lambda unicode=unicode, order=order: np.char.chararray((2, 3), itemsize=2, unicode=unicode,
                                                                                order=order),
                    layout="order", uninit=not unicode)
        for filler in ["ab", "abcdef", "", "a  ", "\xe9", "\U0001F600x", "a\x00b", "  \t"]:
            for shape, order in [(3, "C"), ((2, 2), "F"), ((), "C"), ((0,), "C")]:
                add_cha("cha_new", params={"shape": shape, "itemsize": 3, "unicode": unicode, "order": order,
                                           "buffer": {"str": enc(filler)}, "offset": 5},
                        call=lambda filler=filler, shape=shape, order=order, unicode=unicode:
                        np.char.chararray(shape, itemsize=3, unicode=unicode, buffer=filler, offset=5, order=order),
                        layout="filler")
    raw = bytes(range(97, 97 + 16))
    utf32 = "abcd".encode("utf-32-le")
    for kind in ["bytes", "bytearray"]:
        for shape, itemsize, unicode, offset, strides, order in [
                (2, 2, False, 0, None, "C"), (1, 2, False, 2, None, "C"), (1, 2, False, 3, None, "C"),
                (2, 1, False, 0, [2], "C"), (2, 1, False, 3, [-2], "C"), ((2, 2), 1, False, 0, None, "F"),
                ((2, 3), 2, False, 1, [6, 2], "C"), (3, 6, False, 0, None, "C"), (1, 2, True, 0, None, "C"),
                (2, 1, True, 4, None, "C"), (2, 2, True, 0, None, "C"), ((), 4, False, 2, None, "C"),
                (2, 1, True, 2, None, "C")]:
            data = utf32 if unicode else raw

            def call(shape=shape, itemsize=itemsize, unicode=unicode, offset=offset, strides=strides, order=order,
                     data=data, kind=kind):
                buf = bytes(data) if kind == "bytes" else bytearray(data)
                return np.char.chararray(shape, itemsize=itemsize, unicode=unicode, buffer=buf, offset=offset,
                                         strides=strides, order=order)

            add_cha("cha_new", params={"shape": shape, "itemsize": itemsize, "unicode": unicode, "offset": offset,
                                       "strides": strides, "order": order, "buffer": {kind: data.hex()}},
                    call=call, layout=kind)
    for spec, values, itemsize, unicode in [("|S2", [b"ab", b"cd"], 2, False), ("<U1", ["a", "b"], 1, True),
                                            ("<i4", [0x64636261, 0x68676665], 4, False), ("<U2", ["ab", "cd"], 1, True)]:
        b, v = plain(np.array(values, dtype=spec))
        add_cha("cha_new", params={"shape": 2, "itemsize": itemsize, "unicode": unicode, "buffer": {"op": 0}},
                operands=[(b, v)], call=lambda v, itemsize=itemsize, unicode=unicode:
                np.char.chararray(2, itemsize=itemsize, unicode=unicode, buffer=v), layout="ndarray")
    add_cha("cha_new", params={"shape": 2, "itemsize": 2, "buffer": {"list": enc([1, 2])}},
            call=lambda: np.char.chararray(2, itemsize=2, buffer=[1, 2]), layout="list")

    # B. np.char.array / np.char.asarray.
    _cha_array_cases()

    # C. Element access: every whitespace code point after a letter (U) and every byte (S), then every index kind over
    #    every layout kind — a full integer index (0-d integer arrays count as integers) is the STRIPPED scalar, every
    #    other index a chararray (a copy for a 0-d integer array beside a slice, NumPy's HAS_SCALAR_ARRAY).
    b = np.array(["a" + c for c in PY_WS + PY_KEEP] + ["", " ", "\U0001F600 ", " a", "a\x00 ", "\x00 ", "a \x00 "],
                 dtype="<U4")
    for i in range(b.size):
        add_cha_item(b, b, [{"i": i}], "U:ws")
    b = np.array([b"a" + bytes([x]) for x in range(256)] + [b"", b" ", b" a", b"a\x00 ", b"\x00 "], dtype="|S3")
    for i in range(b.size):
        add_cha_item(b, b, [{"i": i}], "S:ws")
    for spec in ["<U4", "|S4"]:
        for layout in LAYOUTS:
            base, view = _cha_operand(spec, layout, seed=1)
            nd = view.ndim
            if nd == 0:
                indices = [[], [{"e": 1}], [{"n": 1}], [{"i": 0}], [{"e": 1}, {"n": 1}], [{"b": True}]]
            else:
                indices = [[{"i": int(x)} for x in np.unravel_index(k, view.shape)] for k in range(min(view.size, 6))]
                indices += [[{"i": -1}] * nd, [{"i": view.shape[0]}] + [{"i": 0}] * (nd - 1),
                            [{"i": 0}], [{"s": [1, None, None]}], [{"s": [None, None, -1]}], [{"e": 1}], [],
                            [{"e": 1}, {"i": 0}], [{"i": 0}, {"e": 1}], [{"n": 1}], [{"b": True}], [{"b": False}],
                            [{"a": [0, -1]}], [{"z": 0}], [{"z": 0}] * nd, [{"i": 0}] * (nd - 1) + [{"z": 0}],
                            [{"i": 0}] * (nd + 1)]
                if nd >= 2:
                    indices += [[{"s": [None, None, None]}, {"i": 0}], [{"z": 0}, {"s": [None, None, None]}],
                                [{"a": [0]}, {"a": [0]}] + [{"i": 0}] * (nd - 2)]
            for index in indices:
                add_cha_item(base, view, index, f"{spec}:{layout}")
            if view.size and nd:
                mask = np.zeros(view.shape, dtype=bool)
                mask.flat[::2] = True
                add_cha_item(base, view, [{"m": 1}], f"{spec}:{layout}:mask", extra_ops=[plain(mask)])
    # A chararray over numbers (view(np.uint32) / view(np.int8)): an element is the NumPy number, a row view raises.
    for spec, recv in [("<U2", "uint32"), ("|S4", "int8")]:
        base, view = _cha_operand(spec, "c2", seed=0)
        for index in [[{"i": 0}, {"i": 1}], [{"i": 0}], [{"s": [0, 1, None]}], [{"i": -1}, {"i": -1}]]:
            add_cha_item(base, view, index, f"{spec}:c2:{recv}", recv=recv)

    # D. Iteration: the stripped elements of a 1-d chararray, chararray rows of an N-d one, the 0-d TypeError.
    for spec in ["<U4", "|S4"]:
        for layout in LAYOUTS:
            base, view = _cha_operand(spec, layout, seed=2)
            add_cha("cha_iter", params={"recv": None}, operands=[(base, view)],
                    call=lambda v: list(v.view(np.char.chararray)), layout=f"{spec}:{layout}")
    for shape in [(2, 0), (0, 2), (0,)]:
        b, v = plain(np.zeros(shape, dtype="<U1"))
        add_cha("cha_iter", params={"recv": None}, operands=[(b, v)], call=lambda v: list(v.view(np.char.chararray)),
                layout=f"empty{shape}")
    for spec, recv in [("<U2", "uint32"), ("|S4", "int8")]:
        base, view = _cha_operand(spec, "c1", seed=0)
        add_cha("cha_iter", params={"recv": recv}, operands=[(base, view)],
                call=lambda v, recv=recv: list(_cha_receiver(v, recv)), layout=f"{spec}:c1:{recv}")

    # E. The members: views and copies keep the class (OWNDATA as NumPy), NumPy's identities, the finalize errors after
    #    a non-string astype / getfield (isolated), item() / tolist() unstripped, the flat iterator.
    for spec in ["<U4", "|S4"]:
        cls = spec[1]
        for layout in ["c1", "c2", "f2", "t3", "s2", "r2", "o1", "col", "bc", "z0", "one", "e2", "e3", "d5"]:
            for name, args in CHA_MEMBERS:
                add_cha_member(spec, layout, 3, name, args)
            add_cha_member(spec, layout, 3, "astype", [{"dtype": spec}, {"kw": {"copy": False}}])
            add_cha_member(spec, layout, 3, "view", [{"dtype": f"{cls}1"}])
            add_cha_member(spec, layout, 3, "getfield", [{"dtype": f"{cls}1"}, 0])
            add_cha_member(spec, layout, 3, "getfield", [{"dtype": f"{cls}2"}, 4 if cls == "U" else 2])
            add_cha_member(spec, layout, 3, "view", [{"dtype": "uint32" if cls == "U" else "int8"}])
            for target in ["int8", "float64", "T"]:
                add_cha_member(spec, layout, 3, "astype", [{"dtype": target}], isolate=True)
            add_cha_member(spec, layout, 3, "getfield", [{"dtype": "uint8"}, 0], isolate=True)
    # Digit text cast to int8: the one numeric dtype the finalize rule accepts ('b'), so an int8 chararray.
    for spec, values in [("<U2", ["1", "-5", "7"]), ("|S2", ["12", "3", "-1"])]:
        add_cha_member(spec, None, 0, "astype", [{"dtype": "int8"}], values=values)
    # A chararray over numbers: every view and copy raises the finalize error; element access and item() do not.
    for spec, recv in [("<U2", "uint32"), ("|S4", "int8")]:
        for layout in ["c1", "c2"]:
            for name, args in CHA_MEMBERS + [("astype", [{"dtype": "U3"}]), ("view", [{"dtype": f"{spec[1]}1"}]),
                                              ("astype", [{"dtype": recv}, {"kw": {"copy": False}}])]:
                add_cha_member(spec, layout, 0, name, args, recv=recv, isolate=True)

    # F. view(type=): the finalize rule reads the CURRENT dtype; a type that is not an ndarray class is a ValueError. A
    #    None first argument stands for an omitted dtype (the keyword-only type=), as C#'s null does.
    view_sources = {"U2": plain(np.array(["ab", "c"])), "S4": plain(np.array([b"abcd"])), "i8": plain(np.arange(3)),
                    "i1": plain(np.array([65, 32], dtype=np.int8)), "u1": plain(np.array([65], dtype=np.uint8)),
                    "b": plain(np.array([True])), "i4": plain(np.arange(2, dtype=np.int32)),
                    "f8": plain(np.array([1.5])), "U2:2d": plain(np.array([["ab", "c"], ["d", "ef"]])),
                    "U2:F": _fview(np.array([["ab", "c"], ["d", "ef"]]))}
    view_args = [[{"dtype": "chararray"}], [{"dtype": "ndarray"}], [{"dtype": "matrix"}], [None, {"dtype": "chararray"}],
                 [{"dtype": "int32"}, {"dtype": "chararray"}], [{"dtype": "S4"}, {"dtype": "chararray"}],
                 [{"dtype": "U1"}, {"dtype": "chararray"}], [None, {"type": "int"}], [None, {"type": "str"}]]
    for label, (b, v) in view_sources.items():
        for args in view_args:
            def call(v, args=args):
                real = [None if a is None else _cha_arg_value(a) for a in args]
                if len(real) == 1:
                    return v.view(real[0])
                if real[0] is None:
                    return v.view(type=real[1])
                return v.view(real[0], real[1])

            add_cha("cha_viewtype", params={"args": args}, operands=[(b, v)], call=call, layout=label)
    for spec in ["<U3", "|S3"]:
        b, v = _cha_operand(spec, "c2", seed=4)
        for target in ["ndarray", "chararray", "matrix"]:
            add_cha("cha_viewtype", params={"args": [{"dtype": target}], "chararray": True}, operands=[(b, v)],
                    call=lambda v, target=target: v.view(np.char.chararray).view(_cha_arg_value({"dtype": target})),
                    layout=f"{spec}:chararray")

    # G. Printing: repr (the class name, its continuation indent, stripped elements; a 0-d chararray unstripped) and str.
    for spec in ["<U4", "|S4", "<U1", "<U12"]:
        for layout in LAYOUTS:
            base, view = _cha_operand(spec, layout, seed=5)
            add_cha("cha_repr", params={"recv": None}, operands=[(base, view)],
                    call=lambda v: _ChaText(repr(v.view(np.char.chararray))), layout=f"{spec}:{layout}")
            add_cha("cha_str", params={"recv": None}, operands=[(base, view)],
                    call=lambda v: _ChaText(str(v.view(np.char.chararray))), layout=f"{spec}:{layout}")
    for n in [30, 1001, 2000]:
        for spec in ["<U5", "|S5"]:
            vals = [f"{i} " for i in range(n)] if spec[1] == "U" else [f"{i}\t".encode() for i in range(n)]
            b, v = plain(np.array(vals, dtype=spec))
            add_cha("cha_repr", params={"recv": None}, operands=[(b, v)],
                    call=lambda v: _ChaText(repr(v.view(np.char.chararray))), layout=f"n{n}")
            add_cha("cha_str", params={"recv": None}, operands=[(b, v)],
                    call=lambda v: _ChaText(str(v.view(np.char.chararray))), layout=f"n{n}")
    for spec, recv in [("<U2", "uint32"), ("|S4", "int8")]:
        b, v = _cha_operand(spec, "c2", seed=0)
        add_cha("cha_repr", params={"recv": recv}, operands=[(b, v)],
                call=lambda v, recv=recv: _ChaText(repr(_cha_receiver(v, recv))), layout=f"{spec}:{recv}")

    _cha_flush()


# ----------------------------------------------------------------------------------------------------------------------
# Sections strcharops / strcharfn: numpy.char.chararray BEHAVIOR (the numpy.char plan's phase 3) — the operators with
# Python's dispatch, the members that refuse (the finalize error) or partition the strings, and the np.* functions that
# keep the class, wrap their result, drop the class or refuse a chararray.
# ----------------------------------------------------------------------------------------------------------------------

CHO = "strcharops"
CHF = "strcharfn"

# __array_finalize__'s ValueError. NumPy 2.4.2 may corrupt its heap freeing an array whose finalize raised (the
# strchararray section's note), so a worker process that recorded this error is replaced before the next case.
FINALIZE_TEXT = "Can only create a chararray from string data."

_CHX_OPS = {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt,
            ">=": operator.ge, "+": operator.add, "*": operator.mul, "%": operator.mod}


def _chx_dec(v):
    """The inverse of ``enc``: the Python value a JSON encoding stands for.

    Args:
        v: an ``enc`` result (text and bytes as hex, None, bool, int, float, lists, dicts: translate's tables).

    Returns:
        The value.

    Raises:
        TypeError: not an ``enc`` encoding.
    """
    if isinstance(v, list):
        return [_chx_dec(x) for x in v]
    if isinstance(v, dict):
        if "dict" in v:
            # A dict in insertion order, its keys and values decoded (translate's str.translate tables).
            return {_chx_dec(k): _chx_dec(x) for k, x in v["dict"]}
        if "u" in v:
            return bytes.fromhex(v["u"]).decode("utf-32-le", "surrogatepass")
        if "b" in v:
            return bytes.fromhex(v["b"])
        if "none" in v:
            return None
        if "bool" in v:
            return v["bool"]
        if "int" in v:
            return v["int"]
        if "float" in v:
            f = v["float"]
            return float(f) if isinstance(f, str) else f
    raise TypeError(f"not an enc() value: {v!r}")


def _chx_pool(spec, layout, seed=1):
    """The recipe of a chararray-pool operand (``_cha_operand``)."""
    return {"pool": [spec, layout, seed]}


def _chx_vals(values, dtype, out=False):
    """The recipe of an operand holding ``np.array(values, dtype)`` (its own C-contiguous base); ``out`` for an out=
    operand, whose OWNDATA the replay must reproduce (``target``)."""
    d = {"values": enc(values), "dtype": dtype}
    if out:
        d["out"] = True
    return d


def _chx_build(recipe):
    """A (base, view) operand from its recipe — the same in the generator and in a worker process.

    Args:
        recipe: ``{"pool": [spec, layout, seed]}`` (``_cha_operand``), ``{"nonempty": [spec, layout, seed]}``
            (``_cha_operand`` over the pool without its empty string: NumPy 2.4.2's zfill of an empty element reads
            uninitialized memory, see ``_fn_nonempty``), ``{"fmt": [spec, layout, pool, seed]}`` (``_cha_operand``'s
            layout over the % format pool ``CHM_FMT_POOLS[pool]``, latin-1 bytes for an S spec),
            ``{"values": enc, "dtype": spec}`` (``np.array(values, dtype)``) or ``{"T": enc}`` (a StringDType array of
            the values).

    Returns:
        The (base, view) pair.
    """
    if "pool" in recipe:
        spec, layout, seed = recipe["pool"]
        return _cha_operand(spec, layout, seed)
    if "nonempty" in recipe:
        spec, layout, seed = recipe["nonempty"]
        n, build = LAYOUTS[layout]
        pool = [v for v in (CHA_U if "U" in spec else CHA_S) if v]
        base = (np.array([pool[(seed + i) % len(pool)] for i in range(n)], dtype=spec) if n
                else np.zeros(0, dtype=spec))
        return base, build(base)
    if "T" in recipe:
        return t_plain(np.array(_chx_dec(recipe["T"]), dtype=np.dtypes.StringDType()))
    if "fmt" in recipe:
        # A pool of % formats (section strcharmeth's mod cases), laid out like a chararray-pool operand.
        spec, layout, pool_name, seed = recipe["fmt"]
        n, build = LAYOUTS[layout]
        pool = CHM_FMT_POOLS[pool_name]
        if spec[1] == "S":
            pool = [f.encode("latin-1") for f in pool]
        base = (np.array([pool[(seed + i) % len(pool)] for i in range(n)], dtype=spec) if n
                else np.zeros(0, dtype=spec))
        return base, build(base)
    arr = np.array(_chx_dec(recipe["values"]), dtype=recipe["dtype"])
    return target(arr) if recipe.get("out") else plain(arr)


def _chx_cha(k=0):
    """The argument spec of operand ``k`` viewed as a chararray."""
    return {"op": k, "cha": True}


def _chx_arr(k):
    """The argument spec of operand ``k`` as the plain array it is."""
    return {"op": k}


def _chx_v(value):
    """The argument spec of a Python value."""
    return {"v": enc(value)}


def _chx_arg(spec, views, chars):
    """One argument of a call as the Python value NumPy receives.

    Args:
        spec: ``{"op": k}`` (operand k), ``{"op": k, "cha": true}`` (operand k's chararray view — one object per operand,
            so a call that names it twice passes the same chararray), ``{"elem": [k, [i, ...]]}`` (an element of operand
            k's chararray view: the stripped Python str / bytes), ``{"v": enc}`` (a Python value), ``{"dt": spec}`` (a
            dtype) or ``{"tuple": [spec, ...]}`` (a tuple).
        views: the operand views.
        chars: the chararray view of each operand used so far, by index.

    Returns:
        The value.

    Raises:
        TypeError: an unknown spec.
    """
    if "op" in spec:
        k = spec["op"]
        if not spec.get("cha"):
            return views[k]
        if k not in chars:
            chars[k] = views[k].view(np.char.chararray)
        return chars[k]
    if "elem" in spec:
        k, index = spec["elem"]
        if k not in chars:
            chars[k] = views[k].view(np.char.chararray)
        return chars[k][tuple(index)]
    if "v" in spec:
        return _chx_dec(spec["v"])
    if "dt" in spec:
        return np.dtype(spec["dt"])
    if "tuple" in spec:
        return tuple(_chx_arg(x, views, chars) for x in spec["tuple"])
    raise TypeError(f"unknown argument spec {spec!r}")


def _chx_resolve(fn):
    """A NumPy function by its dotted name under ``np`` (``reshape``, ``strings.add``, ``char.multiply``)."""
    obj = np
    for part in fn.split("."):
        obj = getattr(obj, part)
    return obj


def _chx_call(op, params, views):
    """Makes one case's call.

    Args:
        op: ``cha_op`` (an operator: ``params.sym`` over ``params.args``), ``cha_call`` (a member of operand
            ``params.recv``'s chararray view: ``params.name`` with ``args`` / ``kw``; ``sort`` returns the sorted
            receiver) or ``cha_fn`` (the NumPy function ``params.fn`` with ``args`` / ``kw``).
        params: the case parameters.
        views: the operand views.

    Returns:
        The result; a ``_WithSelf`` when NumPy's identity with the receiver is recorded (a member, or a function whose
        first argument is a chararray).
    """
    chars = {}
    if op == "cha_call":
        k = params["recv"]
        chars[k] = views[k].view(np.char.chararray)
    args = [_chx_arg(a, views, chars) for a in params.get("args", [])]
    kw = {name: _chx_arg(a, views, chars) for name, a in params.get("kw", {}).items()}
    if op == "cha_op":
        return _CHX_OPS[params["sym"]](*args)
    if op == "cha_call":
        recv = chars[params["recv"]]
        if params["name"] == "sort":
            recv.sort(*args, **kw)
            return _WithSelf(recv, recv)
        return _WithSelf(getattr(recv, params["name"])(*args, **kw), recv)
    r = _chx_resolve(params["fn"])(*args, **kw)
    first = args[0] if args else None
    return _WithSelf(r, first) if isinstance(first, np.char.chararray) else r


def _chx_expect(r, views, receiver=None, uninit=False):
    """The expectation of a behavior case's result.

    Args:
        r: NumPy's result.
        views: the operand views (memory sharing).
        receiver: the object NumPy's identity is compared with, or None.
        uninit: the result's text is uninitialized (``np.empty_like``): its buffer is not compared.

    Returns:
        ``array`` with its CLASS, its OWNDATA read from the result itself and, given a receiver, ``self``; ``items`` for
        a tuple / list (a namedtuple too); ``element`` for a str / bytes (``__getitem__``'s stripped value) or a
        np.str_ / np.bytes_ (an unstripped NumPy scalar); ``bool`` for a Python bool (Python's fallback of a comparison
        nothing answers); a 0-d ``scalar`` array for a numeric NumPy scalar.

    Raises:
        TypeError: a result kind the sections do not record.
    """
    if isinstance(r, np.ndarray):
        d = array_expect(r, views)
        d["owndata"] = bool(r.flags.owndata)
        d["class"] = type(r).__name__
        if receiver is not None:
            d["self"] = r is receiver
        if uninit:
            d["buffer"] = None
        return d
    if isinstance(r, (list, tuple)):
        return {"kind": "items", "items": [_chx_expect(x, views) for x in r]}
    if isinstance(r, (str, bytes)):
        return {"kind": "element", "py": type(r).__name__, "value": enc(r)}
    if isinstance(r, bool):
        return {"kind": "bool", "value": r}
    if isinstance(r, (np.generic, int)):
        # A numeric NumPy scalar, or a Python int (np.count_nonzero): a 0-d array of its dtype (int64 for a Python int).
        d = array_expect(np.asarray(r) if isinstance(r, np.generic) else np.asarray(r, dtype=np.int64), None)
        d.pop("writeable", None)
        d.pop("owndata", None)
        d["scalar"] = True
        return d
    raise TypeError(f"unexpected behavior result {type(r)}")


def _chx_outcome(payload):
    """Rebuilds a case's operands, makes its call and records the outcome (run by the worker processes).

    Args:
        payload: ``{"op", "params", "recipes"}``.

    Returns:
        ``{"expected": ...}`` or ``{"error": ...}``.
    """
    views = [_chx_build(r)[1] for r in payload["recipes"]]
    try:
        r = _chx_call(payload["op"], payload["params"], views)
        receiver = None
        if isinstance(r, _WithSelf):
            r, receiver = r.result, r.receiver
        return {"expected": _chx_expect(r, views, receiver, payload["params"].get("uninit", False))}
    except Exception as e:  # noqa: BLE001 -- every NumPy exception is a recorded outcome
        return {"error": err_info(e)}


# The cases of sections strcharops / strcharfn whose outcome a worker process records (_chx_flush).
_CHX_PENDING = []


def add_chx(section, op, params, recipes, layout="", phase=1):
    """Records one behavior case; its outcome is computed later by ``_chx_flush``.

    Args:
        section: ``CHO`` or ``CHF``.
        op: ``cha_op``, ``cha_call`` or ``cha_fn`` (see ``_chx_call``).
        params: the case parameters (JSON: argument specs, see ``_chx_arg``).
        recipes: the operands' recipes (``_chx_build``).
        layout: the case's layout label.
        phase: 1, or 2 for a call NumPy answers by computing what NumSharp's string support defers (the string
            partition of ``chararray.partition``).
    """
    pairs = [_chx_build(r) for r in recipes]
    prefix = f"{section}/{op}/"
    n = _OP_COUNT.get(prefix, 0) + 1
    _OP_COUNT[prefix] = n
    cid = f"{section}/{op}/{n}"
    assert cid not in _IDS
    _IDS.add(cid)
    case = {"id": cid, "op": op, "params": params, "operands": [describe(b, v) for b, v in pairs], "phase": phase,
            "layout": layout}
    _CHX_PENDING.append((case, {"op": op, "params": params, "recipes": recipes}))
    CASES.append(case)


# A worker answers cases read from stdin, one JSON line each, until a case raises the finalize error: it leaves then
# (without the interpreter's teardown), and the driver starts a fresh one for the next case.
_CHX_WORKER = ("import json, os, sys\n"
               "sys.path.insert(0, sys.argv[1])\n"
               "import gen_strings_oracle as g\n"
               "for line in sys.stdin:\n"
               "    out = g._chx_outcome(json.loads(line))\n"
               "    sys.stdout.write(json.dumps(out, allow_nan=False) + '\\n')\n"
               "    sys.stdout.flush()\n"
               "    if out.get('error', {}).get('text') == g.FINALIZE_TEXT:\n"
               "        os._exit(0)\n"
               "os._exit(0)\n")


def _chx_run_chunk(payloads):
    """The outcomes of a run of cases, in order, from a sequence of worker processes (a fresh one after each finalize
    error).

    Args:
        payloads: the cases' payloads.

    Returns:
        The outcomes.

    Raises:
        AssertionError: a worker wrote no outcome for a case (it crashed during the call).
    """
    import subprocess
    outcomes = []
    proc = None
    for payload in payloads:
        if proc is None:
            proc = subprocess.Popen([sys.executable, "-I", "-c", _CHX_WORKER, str(HERE)], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        assert line.startswith("{"), f"behavior worker wrote no outcome for {payload}: rc={proc.poll()}"
        out = json.loads(line)
        outcomes.append(out)
        if out.get("error", {}).get("text") == FINALIZE_TEXT:
            proc.stdin.close()
            proc.wait()
            proc = None
    if proc is not None:
        proc.stdin.close()
        proc.wait()
    return outcomes


def _chx_flush():
    """Records the outcome of every pending behavior case: eight runs of consecutive cases, each through its own worker
    sequence; the cases keep their place and ids."""
    from concurrent.futures import ThreadPoolExecutor
    payloads = [p for _, p in _CHX_PENDING]
    k = max(1, (len(payloads) + 7) // 8)
    chunks = [payloads[i:i + k] for i in range(0, len(payloads), k)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_chx_run_chunk, chunks))
    outcomes = [o for chunk in results for o in chunk]
    for (case, _), outcome in zip(_CHX_PENDING, outcomes):
        case.update(outcome)
    _CHX_PENDING.clear()


def _chx_shape(recipe):
    """The view shape of an operand recipe."""
    return _chx_build(recipe)[1].shape


def gen_strcharops():
    """Section strcharops: the chararray operators — the six stripping comparisons with Python's binary-operator
    dispatch (a plain array on either side, NotImplemented for an S / U mix and Python's fallback), + (np.add, a chararray
    view) and * (np.char.asarray(np.char.multiply(...)), a copy for U) — and the members phase 3 adds: argmax / argmin /
    argpartition (the finalize error after NumPy's checks; out= computes), partition (the STRING partition; its sort form
    is NumPy's TypeError), sort (in place), argsort and searchsorted (plain)."""
    syms = ["==", "!=", "<", "<=", ">", ">="]
    for spec in ["<U4", "|S4"]:
        is_u = spec[1] == "U"
        other = "|S4" if is_u else "<U4"
        word = (lambda s: s) if is_u else (lambda s: s.encode("latin-1"))
        alien = (lambda s: s.encode("latin-1")) if is_u else (lambda s: s)
        for layout in ["c1", "c2", "f2", "s1", "r1", "z0", "e1", "one", "bc", "d5"]:
            cha = _chx_pool(spec, layout, 1)
            shape = _chx_shape(cha)
            label = f"{spec}:{layout}"
            C = _chx_cha(0)
            nd = len(shape)
            size = int(np.prod(shape)) if shape else 1
            # The partners: (recipes after the chararray, argument spec, label, syms to cover).
            partners = [([], _chx_v(word("ab")), "word", syms), ([], _chx_v(word("ab  ")), "word_ws", syms),
                        ([], _chx_v(word("")), "empty_word", ["==", "<"]),
                        ([_chx_pool(spec, layout, 5)], _chx_arr(1), "plain", syms),
                        ([_chx_pool(spec, layout, 5)], _chx_cha(1), "chararray", syms),
                        ([], _chx_v(alien("ab")), "alien_word", syms),
                        ([_chx_pool(other, layout, 5)], _chx_arr(1), "alien_plain", syms),
                        ([], _chx_v(5), "int", ["==", "<"]), ([], _chx_v(1.5), "float", ["=="]),
                        ([], _chx_v(True), "bool", ["=="]),
                        ([{"T": enc(["ab"])}], _chx_arr(1), "stringdtype", ["==", "<"])]
            if nd >= 1:
                partners.append(([], _chx_v([word(w) for w in ["ab", "c", " d", ""] * 3][:shape[-1]]), "list", ["==", ">"]))
            if nd >= 1 and size > 0:
                partners.append(([_chx_vals([], spec)], _chx_arr(1), "empty_plain", ["=="]))
                partners.append(([], {"elem": [0, [0] * nd]}, "elem", ["==", "<"]))
            for extra, partner, plabel, cover in partners:
                for sym in cover:
                    for left_first in (True, False):
                        args = [C, partner] if left_first else [partner, C]
                        add_chx(CHO, "cha_op", {"sym": sym, "args": args}, [cha] + extra,
                                layout=f"{label}:{plabel}:{'L' if left_first else 'R'}")
            # + : np.add in the written order.
            add_partners = [([], _chx_v(word("x")), "word"), ([_chx_pool(spec, layout, 5)], _chx_arr(1), "plain"),
                            ([_chx_pool(spec, layout, 5)], _chx_cha(1), "chararray"),
                            ([], _chx_v(alien("x")), "alien_word"), ([], _chx_v(1), "int"), ([], _chx_v(None), "none"),
                            ([{"T": enc(["q"])}], _chx_arr(1), "stringdtype")]
            if nd >= 1:
                add_partners.append(([], _chx_v([word(w) for w in ["x", "yy", "", "z"] * 3][:shape[-1]]), "list"))
            if nd >= 1 and size > 0:
                add_partners.append(([], {"elem": [0, [0] * nd]}, "elem"))
            for extra, partner, plabel in add_partners:
                for left_first in (True, False):
                    args = [C, partner] if left_first else [partner, C]
                    add_chx(CHO, "cha_op", {"sym": "+", "args": args}, [cha] + extra,
                            layout=f"{label}:{plabel}:{'L' if left_first else 'R'}")
            # * : np.char.multiply, then np.char.asarray.
            mul_partners = [([], _chx_v(2), "two"), ([], _chx_v(0), "zero"), ([], _chx_v(-1), "minus"),
                            ([], _chx_v(2 ** 62), "huge"), ([], _chx_v(2.5), "float"), ([], _chx_v(word("x")), "word"),
                            ([_chx_pool(spec, layout, 5)], _chx_arr(1), "plain"), ([], _chx_v(True), "bool"),
                            ([], _chx_v(None), "none")]
            if size > 0:
                counts = (np.arange(size) % 4).reshape(shape).tolist() if nd else 3
                mul_partners.append(([_chx_vals(counts, "int64")], _chx_arr(1), "counts"))
            for extra, partner, plabel in mul_partners:
                for left_first in (True, False):
                    if not left_first and plabel in ("float", "word", "plain", "bool", "none", "minus", "zero"):
                        continue
                    args = [C, partner] if left_first else [partner, C]
                    add_chx(CHO, "cha_op", {"sym": "*", "args": args}, [cha] + extra,
                            layout=f"{label}:{plabel}:{'L' if left_first else 'R'}")
        # A chararray past NumPy's buffer size: the trivial loop no longer copies it, so None meets CPython's own error.
        big = _chx_vals([word("ab")] * 9000, spec)
        for args in ([_chx_cha(0), _chx_v(None)], [_chx_v(None), _chx_cha(0)]):
            add_chx(CHO, "cha_op", {"sym": "+", "args": args}, [big], layout=f"{spec}:big:none")

    # The members.
    for spec in ["<U4", "|S4"]:
        word = (lambda s: s) if spec[1] == "U" else (lambda s: s.encode("latin-1"))
        for layout in ["c1", "c2", "f2", "s1", "r2", "z0", "e1", "e2", "one", "bc", "d5"]:
            cha = _chx_pool(spec, layout, 2)
            shape = _chx_shape(cha)
            nd = len(shape)
            label = f"{spec}:{layout}"

            def member(name, args=(), kw=None, extra=(), phase=1, sub=""):
                add_chx(CHO, "cha_call", {"recv": 0, "name": name, "args": list(args), "kw": kw or {}},
                        [cha] + list(extra), layout=f"{label}{sub}", phase=phase)

            for name in ["argmax", "argmin"]:
                member(name)
                member(name, kw={"axis": _chx_v(0)})
                member(name, kw={"axis": _chx_v(-1)})
                member(name, kw={"axis": _chx_v(5)})
                member(name, kw={"keepdims": _chx_v(True)})
                if nd == 2:
                    member(name, kw={"axis": _chx_v(1), "out": _chx_arr(1)},
                           extra=[_chx_vals([0] * shape[0], "int64", out=True)], sub=":out")
                    member(name, kw={"axis": _chx_v(1), "keepdims": _chx_v(True), "out": _chx_arr(1)},
                           extra=[_chx_vals([[0]] * shape[0], "int64", out=True)], sub=":out_keepdims")
            member("argmax", kw={"out": _chx_arr(1)}, extra=[_chx_vals(0, "int64", out=True)], sub=":out")
            member("argmax", kw={"out": _chx_arr(1)}, extra=[_chx_vals([0, 0], "int64", out=True)], sub=":out_shape")
            member("argmin", kw={"out": _chx_arr(1)}, extra=[_chx_vals(0.0, "float64", out=True)], sub=":out_float")
            member("argpartition", [_chx_v(1)])
            member("argpartition", [_chx_v(9)])
            member("argpartition", [_chx_v(1)], {"axis": _chx_v(3)})
            member("argpartition", [_chx_v(1)], {"kind": _chx_v("bad")})
            member("argpartition", [_chx_v([0, 1])])
            member("argpartition", [_chx_v(0)], {"axis": _chx_v(None)})
            member("partition", [_chx_v(1)], {"axis": _chx_v(0)})
            member("partition", [_chx_v(1)], {"axis": _chx_v(None)})
            member("partition", [_chx_v([0, 1])], {"axis": _chx_v(-1)})
            # chararray.partition(sep): the STRING partition (np.char.partition = np.stack(np.strings.partition(a, sep),
            # axis=-1), seen as a chararray), strict since P2.5.
            member("partition", [_chx_v(word("a"))])
            member("sort")
            member("sort", kw={"axis": _chx_v(0)})
            member("sort", kw={"kind": _chx_v("stable")})
            member("argsort")
            member("argsort", kw={"axis": _chx_v(0)})
            member("argsort", kw={"kind": _chx_v("stable")})
            if nd == 1:
                member("searchsorted", [_chx_v(word("b"))])
                member("searchsorted", [_chx_v([word("a"), word("x")])])
                member("searchsorted", [_chx_v(word("b"))], {"side": _chx_v("right")})
                if shape[0]:
                    member("searchsorted", [_chx_v(word("b"))], {"sorter": _chx_arr(1)},
                           extra=[_chx_vals(list(range(shape[0]))[::-1], "int64")], sub=":sorter")
            member("nonzero")
    _chx_flush()


def gen_strcharfn():
    """Section strcharfn: the np.* functions with chararray operands — the ones that keep the class (Kept: methods and
    np.asanyarray views; their identities: np.asanyarray(c) is c), wrap a text result (Wrapped: the ufuncs, insert /
    delete / diagflat / take_along_axis / unique / setdiff1d, OWNDATA False; a chararray out= comes back as a new view),
    drop it (np.asarray, np.concatenate, np.where, the set operations), create a non-string result as the subclass (the
    finalize error: the index functions without out=, np.zeros_like with a numeric dtype, the trivial-loop copy of a short
    input into bool / StringDType) or call the string partition (np.partition, np.median, np.quantile, np.percentile)."""
    def fn(name, args=(), kw=None, recipes=(), label="", phase=1, uninit=False):
        params = {"fn": name, "args": list(args), "kw": kw or {}}
        if uninit:
            params["uninit"] = True
        add_chx(CHF, "cha_fn", params, list(recipes), layout=label, phase=phase)

    C = _chx_cha(0)
    for spec in ["<U4", "|S4"]:
        is_u = spec[1] == "U"
        word = (lambda s: s) if is_u else (lambda s: s.encode("latin-1"))
        layouts = ["c1", "c2", "f2", "t3", "s2", "r2", "o1", "bc", "z0", "e2", "one", "d5"] if is_u else ["c1", "c2", "z0"]
        for layout in layouts:
            cha = _chx_pool(spec, layout, 3)
            shape = _chx_shape(cha)
            nd = len(shape)
            size = int(np.prod(shape)) if shape else 1
            L = f"{spec}:{layout}"

            def one(name, args=(), kw=None, extra=(), phase=1, uninit=False, sub=""):
                fn(name, [C] + list(args), kw, [cha] + list(extra), f"{L}{sub}", phase, uninit)

            # Kept: methods and np.asanyarray views.
            one("reshape", [_chx_v(-1)])
            one("reshape", [_chx_v(shape if shape else ())])
            one("ravel")
            one("ravel", kw={"order": _chx_v("F")})
            one("transpose")
            one("permute_dims")
            one("squeeze")
            one("expand_dims", [_chx_v(0)])
            one("atleast_1d")
            one("atleast_2d")
            one("atleast_3d")
            one("flip")
            one("roll", [_chx_v(1)])
            one("tile", [_chx_v(2)])
            one("tile", [_chx_v(1)])
            one("repeat", [_chx_v(2)])
            one("sort", kw={"axis": _chx_v(None)})
            one("require")
            one("require", kw={"requirements": _chx_v("F")})
            one("asanyarray")
            one("diagflat")
            one("take", [_chx_v(0)])
            one("unique")
            one("unique", kw={"return_counts": _chx_v(True)})
            one("unique", kw={"return_index": _chx_v(True), "return_inverse": _chx_v(True), "return_counts": _chx_v(True)})
            one("unique_counts")
            one("unique_inverse")
            one("unique_all")
            one("empty_like", uninit=True)
            one("zeros_like")
            one("ones_like")
            one("full_like", [_chx_v(word("xy"))])
            one("full_like", [_chx_v(5)])
            for dt in ["int8", "bool", "float64", "U2", "S2", "int64"]:
                one("zeros_like", kw={"dtype": {"dt": dt}}, sub=f":{dt}")
            one("ones_like", kw={"dtype": {"dt": "float64"}}, sub=":float64")
            one("full_like", [_chx_v(word("x"))], kw={"dtype": {"dt": "float64"}}, sub=":float64")
            one("full_like", [_chx_v(3)], kw={"dtype": {"dt": "int8"}}, sub=":int8")
            # Plain: np.asarray and the conversions, the functions that concatenate or broadcast.
            one("asarray")
            one("ascontiguousarray")
            one("asfortranarray")
            one("copy")
            one("array")
            fn("concatenate", [{"tuple": [C, C]}], {"axis": _chx_v(None)}, [cha], f"{L}:pair")
            fn("stack", [{"tuple": [C, C]}], {}, [cha], L)
            fn("where", [_chx_v(True), C, C], {}, [cha], L)
            one("resize", [_chx_v(3)])
            one("broadcast_to", [_chx_v((2,) + tuple(shape))])
            one("argsort", kw={"axis": _chx_v(None)})
            one("count_nonzero")
            fn("union1d", [C, C], {}, [cha], L)
            fn("intersect1d", [C, C], {}, [cha], L)
            fn("setxor1d", [C, C], {}, [cha], L)
            fn("isin", [C, C], {}, [cha], L)
            fn("setdiff1d", [C, _chx_arr(1)], {}, [cha, _chx_pool(spec, "c1", 6)], f"{L}:ar1")
            fn("setdiff1d", [_chx_arr(1), C], {}, [cha, _chx_pool(spec, "c1", 6)], f"{L}:ar2")
            fn("setdiff1d", [C, _chx_arr(1)], {"assume_unique": _chx_v(True)}, [cha, _chx_pool(spec, "c1", 6)],
               f"{L}:assume_unique")
            # More functions whose result is plain (they end in np.concatenate / np.where / an index computation) while
            # their inner calls meet chararray rules (np.append's ravel, the stack family's atleast_*, np.flatnonzero's
            # ravel): the class, and — through the boundary audit — that NumSharp keeps those inner calls plain.
            one("append", [C])
            if nd:
                one("append", [C], {"axis": _chx_v(0)})
            fn("dstack", [{"tuple": [C, C]}], {}, [cha], L)
            fn("column_stack", [{"tuple": [C, C]}], {}, [cha], L)
            one("tril")
            one("triu")
            one("pad", [_chx_v(1)], {"mode": _chx_v("edge")})
            fn("broadcast_arrays", [C, C], {}, [cha], L)
            one("nonzero")
            one("argwhere")
            one("flatnonzero")
            if nd >= 2:
                one("lexsort")
            fn("choose", [_chx_arr(1), {"tuple": [C, C]}], {}, [cha, _chx_vals(np.zeros(shape, dtype=np.int64).tolist(), "int64")],
               L)
            # np.copy keeps the class only with subok=True.
            one("copy", kw={"subok": _chx_v(True)}, sub=":subok")
            # np.meshgrid keeps EACH input's class (np.asanyarray, reshape, broadcast_arrays(subok=True), copy).
            if nd == 1:
                fn("meshgrid", [C, C], {}, [cha], L)
                fn("meshgrid", [C, C], {"sparse": _chx_v(True)}, [cha], f"{L}:sparse")
                fn("meshgrid", [C, C], {"copy": _chx_v(False)}, [cha], f"{L}:nocopy")
                fn("meshgrid", [C, _chx_arr(1)], {"indexing": _chx_v("ij")}, [cha, _chx_vals([word("p"), word("q")], spec)],
                   f"{L}:mixed")
            # The index functions: the finalize error after their checks; out= computes.
            for name in ["argmax", "argmin", "nanargmax", "nanargmin"]:
                one(name)
                one(name, kw={"axis": _chx_v(0)})
                one(name, kw={"axis": _chx_v(3)})
                one(name, kw={"out": _chx_arr(1)}, extra=[_chx_vals(0, "int64", out=True)], sub=":out")
            if nd:
                one("argmax", kw={"keepdims": _chx_v(True), "out": _chx_arr(1)},
                    extra=[_chx_vals(np.zeros([1] * nd, dtype=np.int64).tolist(), "int64", out=True)],
                    sub=":out_keepdims")
            one("argpartition", [_chx_v(0)])
            one("argpartition", [_chx_v(9)])
            one("argpartition", [_chx_v(0)], {"axis": _chx_v(None)})
            # The string partition: np.partition's a.partition(kth, axis=...) and the statistics that call it.
            one("partition", [_chx_v(0)])
            one("partition", [_chx_v(9)], {"axis": _chx_v(3)})
            one("median")
            one("median", kw={"axis": _chx_v(0)})
            one("median", kw={"axis": _chx_v(3)})
            one("median", kw={"overwrite_input": _chx_v(True)})
            one("median", kw={"keepdims": _chx_v(True)})
            one("quantile", [_chx_v(0.5)])
            one("quantile", [_chx_v(0.5)], {"method": _chx_v("lower")})
            one("quantile", [_chx_v(0.5)], {"method": _chx_v("bad")})
            one("quantile", [_chx_v(2.0)], {"method": _chx_v("lower")})
            one("quantile", [_chx_v([0.25, 0.75])], {"method": _chx_v("nearest"), "axis": _chx_v(0)})
            one("percentile", [_chx_v(50)], {"method": _chx_v("higher")})
            one("percentile", [_chx_v(150)])
            # The ufuncs: chararray.__array_wrap__'s view of a text result; a plain one for a bool result.
            one("add", [_chx_v(word("x"))])
            fn("add", [_chx_v(word("x")), C], {}, [cha], f"{L}:R")
            fn("add", [C, C], {}, [cha], f"{L}:self")
            fn("add", [C, _chx_arr(1)], {}, [cha, {"T": enc(["q"])}], f"{L}:stringdtype")
            fn("add", [_chx_arr(1), C], {}, [cha, {"T": enc(["q"])}], f"{L}:stringdtype:R")
            if size:
                mask = (np.arange(size) % 2 == 0).reshape(shape).tolist() if nd else True
                # A fresh result's masked-off elements are uninitialized in NumPy.
                fn("add", [C, _chx_v(word("x"))], {"where": _chx_arr(1)}, [cha, _chx_vals(mask, "bool")], f"{L}:where",
                   uninit=True)
                # With out= every element is defined: the selected ones computed, the others out='s own.
                fn("add", [C, _chx_v(word("x"))], {"where": _chx_arr(1), "out": _chx_arr(2)},
                   [cha, _chx_vals(mask, "bool"), _chx_vals(np.full(shape, word("o"), dtype=spec[1] + "8").tolist(),
                                                            spec[1] + "8", out=True)], f"{L}:where_out")
            # where=True (the Python object) is no mask; where=False is a mask that selects nothing.
            fn("add", [C, _chx_v(word("x"))], {"where": _chx_v(True)}, [cha], f"{L}:where_true")
            fn("add", [C, C], {"out": _chx_arr(1)}, [cha, _chx_vals(np.zeros(shape, dtype=spec[1] + "8").tolist(),
                                                                    spec[1] + "8", out=True)], f"{L}:out")
            fn("add", [C, C], {"out": _chx_cha(1)}, [cha, _chx_vals(np.zeros(shape, dtype=spec[1] + "8").tolist(),
                                                                    spec[1] + "8", out=True)], f"{L}:out_chararray")
            fn("multiply", [C, _chx_v(2)], {"out": _chx_arr(1)}, [cha, _chx_vals(np.zeros(shape, dtype=spec[1] + "8").tolist(),
                                                                                spec[1] + "8", out=True)], f"{L}:out")
            one("strings.add", [_chx_v(word("x"))])
            one("strings.multiply", [_chx_v(2)])
            one("char.add", [_chx_v(word("x"))])
            one("char.multiply", [_chx_v(2)])
            one("char.multiply", [_chx_v(2.5)])
            for name in ["equal", "less"]:
                fn(name, [C, C], {}, [cha], f"{L}:self")
                fn(name, [C, _chx_arr(1)], {}, [cha, {"T": enc(["q"])}], f"{L}:stringdtype")
            for name in ["logical_and", "logical_or", "logical_xor"]:
                fn(name, [C, C], {}, [cha], f"{L}:self")
                fn(name, [C, _chx_v(5)], {}, [cha], f"{L}:int")
                fn(name, [_chx_v(True), C], {}, [cha], f"{L}:bool:R")
            if size:
                mask = (np.arange(size) % 2 == 0).reshape(shape).tolist() if nd else True
                fn("logical_and", [C, C], {"where": _chx_arr(1)}, [cha, _chx_vals(mask, "bool")], f"{L}:where", uninit=True)
                fn("logical_and", [C, C], {"where": _chx_arr(1), "out": _chx_arr(2)},
                   [cha, _chx_vals(mask, "bool"), _chx_vals(np.ones(shape, dtype=bool).tolist(), "bool", out=True)],
                   f"{L}:where_out")
            # _wheremask_converter: only the object True is no mask (the trivial loop then copies a short chararray into
            # bool: the finalize error); False is a mask, so the masked loop runs and never copies.
            fn("logical_and", [C, C], {"where": _chx_v(True)}, [cha], f"{L}:where_true")
            fn("logical_and", [C, C], {"where": _chx_v(False)}, [cha], f"{L}:where_false", uninit=True)
            fn("logical_and", [C, C], {"out": _chx_arr(1)}, [cha, _chx_vals(np.zeros(shape, dtype=bool).tolist(), "bool",
                                                                            out=True)], f"{L}:out")
            if nd >= 1:
                # Rank-dependent forms.
                one("swapaxes", [_chx_v(0), _chx_v(-1)])
                one("moveaxis", [_chx_v(0), _chx_v(-1)])
                one("rollaxis", [_chx_v(-1)])
                one("flipud")
                one("roll", [_chx_v(1)], {"axis": _chx_v(0)})
                one("repeat", [_chx_v(2)], {"axis": _chx_v(0)})
                one("sort")
                one("sort", kw={"axis": _chx_v(0)})
                if shape[0]:
                    one("take", [_chx_v([0, -1])], kw={"axis": _chx_v(0)})
                one("unstack")
                one("unstack", kw={"axis": _chx_v(-1)})
                one("split", [_chx_v(1)])
                one("array_split", [_chx_v(2)])
                one("hsplit", [_chx_v(1)])
                one("insert", [_chx_v(0), _chx_v(word("z"))])
                if shape[0]:
                    one("delete", [_chx_v(0)])
                fn("compress", [_chx_v(([True, False] * 4)[:shape[0]]), C], {"axis": _chx_v(0)}, [cha], f"{L}")
                fn("compress", [_chx_v([True] * (shape[0] + 1)), C], {"axis": _chx_v(0)}, [cha], f"{L}:long")
                one("trim_zeros")
                one("unique", kw={"axis": _chx_v(0)})
                one("argsort")
                one("partition", [_chx_v(0)], {"axis": _chx_v(0)})
                one("median", kw={"axis": _chx_v(-1), "overwrite_input": _chx_v(True)})
                idx = np.zeros(shape, dtype=np.int64).tolist()
                one("take_along_axis", [_chx_arr(1)], {"axis": _chx_v(-1)}, extra=[_chx_vals(idx, "int64")])
                fn("extract", [_chx_arr(1), C], {}, [cha, _chx_vals(np.ones(shape, dtype=bool).tolist(), "bool")],
                   f"{L}:order")
                fn("hstack", [{"tuple": [C, C]}], {}, [cha], L)
                fn("vstack", [{"tuple": [C, C]}], {}, [cha], L)
                if nd == 1:
                    fn("searchsorted", [C, _chx_v(word("b"))], {}, [cha], L)
                    one("diag")
            if nd >= 2:
                one("matrix_transpose")
                one("fliplr")
                one("rot90")
                one("diagonal")
                one("diag")
                one("vsplit", [_chx_v(1)])
                one("unique", kw={"axis": _chx_v(1)})
                one("median", kw={"axis": _chx_v((0, 1))})
                one("quantile", [_chx_v(0.5)], {"axis": _chx_v((0, 1)), "method": _chx_v("lower")})
            # The stand-in (NumPy's Python str / bytes) is no chararray operand.
            if size and nd:
                fn("atleast_2d", [{"elem": [0, [0] * nd]}], {}, [cha], f"{L}:elem")
                fn("add", [{"elem": [0, [0] * nd]}, C], {}, [cha], f"{L}:elem")
                fn("logical_and", [{"elem": [0, [0] * nd]}, {"elem": [0, [-1] * nd]}], {}, [cha], f"{L}:elems")
        # Past NumPy's buffer size the trivial loop no longer copies a 1-d input, and a 2-d one is never copied.
        for n in [8192, 8193]:
            big = _chx_vals([word("ab")] * n, spec)
            fn("logical_and", [C, C], {}, [big], f"{spec}:n{n}")
            fn("add", [C, _chx_arr(1)], {}, [big, {"T": enc(["q"])}], f"{spec}:n{n}:stringdtype")
        fn("logical_and", [_chx_arr(1), C], {}, [_chx_pool(spec, "c1", 3), _chx_pool(spec, "c2", 4)], f"{spec}:2d_first")
        fn("logical_and", [C, _chx_arr(1)], {}, [_chx_pool(spec, "c1", 3), _chx_pool(spec, "c2", 4)], f"{spec}:1d_first")
    _chx_flush()


# ----------------------------------------------------------------------------------------------------------------------
# strsets. Set routines and statistics over text: np.unique with every flag and the Array-API unique family, np.isin,
# intersect1d / union1d / setdiff1d / setxor1d, np.lexsort, np.sort_complex, np.histogram / histogram_bin_edges, and
# np.quantile / np.percentile with the methods NumPy computes on text (strings phase 2, P2.2 milestone C)
# ----------------------------------------------------------------------------------------------------------------------

SETS = "strsets"


def _set_values(rng, n, flavor, kind):
    """``n`` str values of one data flavor (encoded later for an ``S`` operand).

    :param rng: the generator.
    :param n: the count.
    :param flavor: ``cat`` (a few values repeated), ``rand`` (mostly distinct), ``nul`` (embedded and trailing NULs, the
        empty string), ``prefix`` (a long shared prefix), ``wide`` (non-ASCII: Latin-1 for ``S`` stand-ins, astral
        and BMP units for ``U``).
    :param kind: ``U`` or ``S`` (an ``S`` flavor keeps to code points below 256, encoded as Latin-1).
    :returns: the list.
    """
    if flavor == "cat":
        return _pick(rng, n, ["pear", "fig", "apple", "fig2", "kiwi", ""])
    if flavor == "rand":
        return _rand_strings(rng, n, 6, "abcdefgh")
    if flavor == "nul":
        return _pick(rng, n, ["a", "a\x00b", "a\x00", "", "\x00", "ab", "b\x00\x00c"])
    if flavor == "prefix":
        return ["xxxxxxx" + s for s in _rand_strings(rng, n, 2, "ab")]
    pool = ["\xe9", "a\xff", "\xe0b", "z", "\x80"] if kind == "S" else ["\xe9", "\U0001F600", "中", "a￿", "z"]
    return _pick(rng, n, pool)


def _set_operand(rng, n, flavor, spec):
    """A 1-D C-contiguous text operand of ``n`` values of one flavor.

    :param rng: the generator.
    :param n: the count.
    :param flavor: see ``_set_values``.
    :param spec: the dtype (``U6``, ``S6`` ...).
    :returns: the (base, view) pair.
    """
    kind = spec.lstrip("<|")[0]
    vals = _set_values(rng, n, flavor, kind)
    if kind == "S":
        return own(np.array([v.encode("latin-1") for v in vals], dtype=spec))
    return own(np.array(vals, dtype=spec))


def _quant_call(name, a, q, axis, method, keepdims, out, weights=None):
    """``np.quantile`` / ``np.percentile`` (or a nan form) with the given keywords (``out`` and ``weights`` omitted when
    None, as a NumPy caller leaves them out)."""
    kw = {"axis": axis, "method": method, "keepdims": keepdims}
    if out is not None:
        kw["out"] = out
    if weights is not None:
        kw["weights"] = weights
    return getattr(np, name)(a, q, **kw)


def _quant(layout, op, *, name="quantile", q=0.5, qkind="f", axis=None, method="lower", keepdims=False, out=None,
           extra=(), weights=None, section=None):
    """Records one quantile-family call (op ``quant_kw``).

    :param layout: the layout label.
    :param op: the (base, view) data operand.
    :param name: ``quantile``, ``percentile``, ``nanquantile`` or ``nanpercentile``.
    :param q: the q value: a float, a list of floats, a Python int, or a (base, view) operand for ``qkind='arr'``.
    :param qkind: ``f`` (a Python float), ``fl`` (a list of floats), ``i`` (a Python int), ``arr`` (an ndarray q).
    :param axis: None, an int or a tuple of ints.
    :param method: the method name.
    :param keepdims: keepdims.
    :param out: an out (base, view) operand, or None.
    :param extra: nothing (reserved).
    :param weights: a weights (base, view) operand (NumPy's ``weights=`` keyword), or None to leave it out.
    :param section: the section (``SETS`` by default).
    """
    ops = [op]
    params = {"name": name, "qkind": qkind, "method": method, "keepdims": keepdims,
              "axis": list(axis) if isinstance(axis, tuple) else axis}
    if qkind == "arr":
        params["q_op"] = len(ops)
        ops.append(q)
    else:
        params["q"] = q
    if out is not None:
        params["out_op"] = len(ops)
        ops.append(out)
    if weights is not None:
        params["w_op"] = len(ops)
        ops.append(weights)

    def call(*views, p=params):
        a = views[0]
        qq = views[p["q_op"]] if p["qkind"] == "arr" else p["q"]
        o = views[p["out_op"]] if "out_op" in p else None
        w = views[p["w_op"]] if "w_op" in p else None
        ax = tuple(p["axis"]) if isinstance(p["axis"], list) else p["axis"]
        return _quant_call(p["name"], a, qq, ax, p["method"], p["keepdims"], o, w)

    add(section or SETS, "quant_kw", params=params, operands=ops, layout=layout,
        mutate=params.get("out_op"), call=call)


def gen_strsets():
    """Section strsets: set routines and statistics over text (P2.2 milestone C). Every case is phase 1: NumSharp computes
    ``S`` / ``U`` text, so it must match NumPy now.

    * ``np.unique``: every flag combination over categorical, distinct, NUL-holding (``'a\\x00b'`` beside ``'a'``,
      trailing NULs that read away), long-prefix and non-ASCII data at sizes around NumSharp's block boundaries; the
      catalog's layouts; every axis of 2-D / 3-D arrays (NumPy's structured row path: rows compared field by field); the
      ``equal_nan`` / ``sorted`` keywords (``sorted=False`` without flags is NumPy's hash order: an excused order); the
      Array-API family (``unique_values`` is NumPy's hash order: an excused order);
    * ``np.isin``: both of ``_isin``'s paths (the loop below ``10 * len(ar1) ** 0.145`` test elements, the sort path at and
      above it), ``S`` with ``U`` (no loop: the mask stays; the sort path decodes the bytes, raising NumPy's
      UnicodeDecodeError for a non-ASCII byte), text with numbers (written as text on the sort path), ``invert``,
      ``assume_unique`` with duplicates (NumPy's undefined-but-exact answer), the ``table`` refusal, empty and 0-d / 2-D
      operands;
    * the set routines: ``intersect1d`` (``assume_unique``, ``return_indices``: the stable argsort's first occurrences),
      ``union1d``, ``setdiff1d`` / ``setxor1d`` (``assume_unique``), over widths, classes and numbers;
    * ``np.lexsort``: text keys, mixed text / number keys, 2-D keys along every axis, ties broken by the earlier keys;
    * ``np.sort_complex``: text sorted as text and then parsed as complex (the order is the text's), the parse errors;
    * ``np.histogram`` / ``histogram_bin_edges``: text data with text edges and with number edges (searched as text:
      counts can be negative), number data with text edges, weights of every kind, density, data past NumPy's
      65,536-element block, decreasing edges;
    * ``np.quantile`` / ``np.percentile``: every method (the take path's selections, 'linear' with an integer q, the
      interpolating methods' refusals and empty-axis lookups), q as a Python float / list / int and as arrays of every
      dtype and rank up to 3, ``axis`` as None / int / tuple, ``keepdims``, ``out=`` of every width and class (NumPy's
      take checks), over the catalog's layouts and empty axes.
    """
    rng = np.random.default_rng(20261016 + 3)
    flags_all = [(False, False, False), (True, False, False), (False, True, False), (False, False, True),
                 (True, True, True), (True, False, True)]

    # ---- np.unique and the Array-API family ----------------------------------------------------------------------------
    for spec in SPECS:
        for lay in LAYOUTS:
            op = operand(spec, lay, seed=7)
            v = op[1]
            axes = [None] + (list(range(-v.ndim, v.ndim)) if v.ndim and lay in CORE_LAYOUTS else [])
            for ax in axes:
                for ri, rv, rc in (flags_all if lay in CORE_LAYOUTS else flags_all[:2] + flags_all[4:5]):
                    add(SETS, "unique", params={"ri": ri, "rv": rv, "rc": rc, "axis": ax}, operands=[op], layout=lay,
                        call=lambda a, x=ax, i=ri, n=rv, c=rc: np.unique(a, return_index=i, return_inverse=n,
                                                                         return_counts=c, axis=x))
            for name in ("unique_values", "unique_counts", "unique_inverse", "unique_all"):
                add(SETS, "unique_api", params={"name": name}, operands=[op], layout=lay,
                    call=lambda a, f=getattr(np, name): f(a))

    for n in (2, 7, 16, 17, 100, 1025):
        for spec in ("<U6", "|S6", "<U9", "|S9"):
            for flavor in ("cat", "rand", "nul", "prefix", "wide"):
                if n >= 1000 and (flavor not in ("cat", "rand") or spec in ("<U9", "|S9")):
                    continue
                op = _set_operand(rng, n, flavor, spec)
                lay = f"{flavor}{spec.lstrip('<|')}n{n}"
                for ri, rv, rc in (flags_all if n < 1000 else [(False, False, False), (True, True, True)]):
                    add(SETS, "unique", params={"ri": ri, "rv": rv, "rc": rc, "axis": None}, operands=[op], layout=lay,
                        call=lambda a, i=ri, nn=rv, c=rc: np.unique(a, return_index=i, return_inverse=nn, return_counts=c))
                if n in (17, 1025):
                    for name in ("unique_values", "unique_counts", "unique_inverse", "unique_all"):
                        add(SETS, "unique_api", params={"name": name}, operands=[op], layout=lay,
                            call=lambda a, f=getattr(np, name): f(a))
                    # sorted=False: NumPy hashes the text when no index output is asked (an excused order), sorts it
                    # otherwise; equal_nan has no effect on text.
                    for ri, rv, rc in ((False, False, False), (True, True, True)):
                        for en, so in ((False, True), (True, False)):
                            add(SETS, "unique", params={"ri": ri, "rv": rv, "rc": rc, "axis": None, "equal_nan": en,
                                                        "sorted": so},
                                operands=[op], layout=lay,
                                call=lambda a, i=ri, nn=rv, c=rc, e=en, s=so: np.unique(
                                    a, return_index=i, return_inverse=nn, return_counts=c, equal_nan=e, sorted=s))

    # Rows: 2-D and 3-D arrays along every axis (NumPy's structured row path), duplicate rows and NUL padding.
    for spec in ("<U3", "|S3"):
        kind = spec.lstrip("<|")[0]
        rows = [["ab", "c", "d"], ["ab", "c", "d"], ["a", "c\x00", "d"], ["ab", "", "d"], ["b", "a", "z"],
                ["ab", "c", "d"], ["a", "c", "d"], ["", "", ""]]
        enc_rows = [[x.encode() for x in r] for r in rows] if kind == "S" else rows
        base2 = np.array(enc_rows, dtype=spec)
        cases = {"r2": base2, "r2T": base2.T, "r3": base2.reshape(2, 4, 3), "r2s": base2[::2], "r2rev": base2[::-1]}
        for key, arr in cases.items():
            op = plain(arr)
            for ax in range(-arr.ndim, arr.ndim):
                for ri, rv, rc in flags_all:
                    add(SETS, "unique", params={"ri": ri, "rv": rv, "rc": rc, "axis": ax}, operands=[op],
                        layout=f"{key}{spec.lstrip('<|')}",
                        call=lambda a, x=ax, i=ri, n=rv, c=rc: np.unique(a, return_index=i, return_inverse=n,
                                                                         return_counts=c, axis=x))
    for key, arr in {"cols1000": np.array(_pick(rng, 3000, ["x", "y", "xy", ""]), dtype="<U2").reshape(1000, 3),
                     "colsS1000": np.array([s.encode() for s in _pick(rng, 3000, ["x", "y", "xy", ""])],
                                           dtype="|S2").reshape(1000, 3)}.items():
        op = plain(arr)
        for ax in (0, 1):
            for ri, rv, rc in ((False, False, False), (True, True, True)):
                add(SETS, "unique", params={"ri": ri, "rv": rv, "rc": rc, "axis": ax}, operands=[op], layout=key,
                    call=lambda a, x=ax, i=ri, n=rv, c=rc: np.unique(a, return_index=i, return_inverse=n,
                                                                     return_counts=c, axis=x))

    # ---- np.isin ---------------------------------------------------------------------------------------------------------
    def isin_case(layout, x, y, *, assume_unique=False, invert=False, kind=None):
        params = {"assume_unique": assume_unique, "invert": invert, "kind": kind}
        add(SETS, "isin", params=params, operands=[x, y], layout=layout,
            call=lambda a, b, au=assume_unique, iv=invert, k=kind: np.isin(a, b, assume_unique=au, invert=iv, kind=k))

    u_pool = ["pear", "fig", "apple", "kiwi", "", "fi", "fig\x00x"]
    for n1, n2s in ((1, (1, 9, 10, 11)), (100, (5, 19, 20, 40)), (2000, (30, 31)), (7, (1, 13, 14))):
        x_u = own(np.array(_pick(rng, n1, u_pool), dtype="<U6"))
        x_s = own(np.array([s.encode() for s in _pick(rng, n1, u_pool)], dtype="|S6"))
        for n2 in n2s:
            y_u = own(np.array(_pick(rng, n2, u_pool + ["zz", "plum"]), dtype="<U5"))
            y_s = own(np.array([s.encode() for s in _pick(rng, n2, u_pool + ["zz", "plum"])], dtype="|S5"))
            y_i = own(np.arange(n2) - 2)
            y_sx = own(np.array([b"\xff" if i % 3 == 0 else b"fig" for i in range(n2)], dtype="|S3"))
            lay = f"{n1}x{n2}"
            for iv in ((False, True) if n1 < 1000 else (False,)):
                isin_case(f"UU{lay}", x_u, y_u, invert=iv)
                isin_case(f"SS{lay}", x_s, y_s, invert=iv)
                isin_case(f"US{lay}", x_u, y_s, invert=iv)
                isin_case(f"SU{lay}", x_s, y_u, invert=iv)
                isin_case(f"Ui{lay}", x_u, y_i, invert=iv)
                isin_case(f"iU{lay}", own(np.arange(n1) - 1), y_u, invert=iv)
                isin_case(f"USx{lay}", x_u, y_sx, invert=iv)
            isin_case(f"UUau{lay}", x_u, y_u, assume_unique=True)
            isin_case(f"UUsort{lay}", x_u, y_u, kind="sort")
    nums = own(np.array(["1", "2", "10", "3.5", "-1", "nan", "1e3"]))
    for y, key in ((own(np.arange(-3, 20)), "ints"), (own(np.array([3.5, 1000.0, np.nan, -1.0] * 6)), "floats"),
                   (own(np.array([True, False] * 12)), "bools"), (own(np.array([1 + 0j, 3.5 + 0j] * 12)), "complex")):
        isin_case(f"nums-{key}", nums, y)
        isin_case(f"nums-{key}-inv", nums, y, invert=True)
    two_d = plain(np.array(_pick(rng, 12, u_pool), dtype="<U6").reshape(3, 4))
    zero_d = plain(np.array("fig"))
    for key, x in (("2d", two_d), ("0d", zero_d), ("e", own(np.zeros(0, "<U3"))),
                   ("t3", operand("<U3", "t3", seed=2)), ("bc", operand("<U3", "bc", seed=4))):
        for y, ykey in ((own(np.array(["fig", "pear", "a"], dtype="<U4")), "small"),
                        (own(np.array(_pick(rng, 60, u_pool), dtype="<U6")), "big"),
                        (operand("<U3", "s1", seed=1), "s1")):
            isin_case(f"{key}-{ykey}", x, y)
    isin_case("table", own(np.array(["a", "b"])), own(np.array(["a"])), kind="table")
    isin_case("badkind", own(np.array(["a", "b"])), own(np.array(["a"])), kind="quick")
    isin_case("au-dups", own(np.array(["b", "a", "b", "c", "a"] * 8)), own(np.array(["a", "b"] * 30)), assume_unique=True)
    isin_case("au-dups-inv", own(np.array(["b", "a", "b", "c", "a"] * 8)), own(np.array(["a", "b"] * 30)),
              assume_unique=True, invert=True)
    isin_case("empty-S-nonascii", own(np.zeros(0, "<U2")), own(np.array([b"\xff", b"a"], dtype="|S1")))

    # ---- the set routines ------------------------------------------------------------------------------------------------
    def setop(layout, name, x, y, assume_unique=False, return_indices=False):
        params = {"name": name, "assume_unique": assume_unique, "return_indices": return_indices}
        if name == "intersect1d":
            call = lambda a, b, au=assume_unique, ri=return_indices: np.intersect1d(a, b, assume_unique=au,
                                                                                     return_indices=ri)
        elif name == "union1d":
            call = lambda a, b: np.union1d(a, b)
        else:
            call = lambda a, b, f=getattr(np, name), au=assume_unique: f(a, b, assume_unique=au)
        add(SETS, "setop", params=params, operands=[x, y], layout=layout, call=call)

    set_pairs = []
    for n1, n2 in ((5, 4), (40, 33), (600, 400)):
        a_u = own(np.array(_pick(rng, n1, u_pool), dtype="<U6"))
        b_u = own(np.array(_pick(rng, n2, u_pool + ["plum", "zz", "fig\x00"]), dtype="<U4"))
        a_s = own(np.array([s.encode() for s in _pick(rng, n1, u_pool)], dtype="|S6"))
        b_s = own(np.array([s.encode() for s in _pick(rng, n2, u_pool + ["plum"])], dtype="|S7"))
        set_pairs += [(f"UU{n1}", a_u, b_u), (f"SS{n1}", a_s, b_s), (f"US{n1}", a_u, b_s), (f"SU{n1}", a_s, b_u),
                      (f"Ui{n1}", a_u, own(np.arange(n2) % 7)), (f"iU{n1}", own(np.arange(n1) % 5), b_u)]
    set_pairs += [("Ue", own(np.array(["b", "a", "b"])), own(np.zeros(0, "<U3"))),
                  ("eU", own(np.zeros(0, "<U3")), own(np.array(["b", "a"]))),
                  ("2d", plain(np.array([["b", "a"], ["c", "a"]])), plain(np.array([["a", "z"]]))),
                  ("0d", plain(np.array("b")), own(np.array(["b", "c"]))),
                  ("t3", operand("<U3", "t3", seed=3), operand("<U3", "r2", seed=5)),
                  ("Sx", own(np.array(["a", "b"])), own(np.array([b"\xff", b"b"], dtype="|S1"))),
                  ("nul", own(np.array(["a", "a\x00b", "", "\x00b"])), own(np.array(["a\x00b", "b", "", "a"]))),
                  ("floats", own(np.array(["1.5", "nan", "2"])), own(np.array([1.5, np.nan, 2.0])))]
    for key, x, y in set_pairs:
        for name in ("intersect1d", "union1d", "setdiff1d", "setxor1d"):
            setop(key, name, x, y)
            if name != "union1d":
                setop(key, name, x, y, assume_unique=True)
        setop(key, "intersect1d", x, y, return_indices=True)
        setop(key, "intersect1d", x, y, assume_unique=True, return_indices=True)

    # ---- np.lexsort --------------------------------------------------------------------------------------------------------
    def lexsort_case(layout, keys, axis=-1):
        add(SETS, "lexsort", params={"axis": axis}, operands=keys, layout=layout,
            call=lambda *ks, x=axis: np.lexsort(ks, axis=x))

    for n in (5, 40, 1025):
        k1 = own(np.array(_pick(rng, n, ["b", "a", "c", "ab", ""]), dtype="<U2"))
        k2 = own(np.array(_pick(rng, n, ["x", "y", "xy"]), dtype="<U2"))
        k3 = own(np.array([s.encode() for s in _pick(rng, n, ["q", "p", "pq", ""])], dtype="|S2"))
        ki = own(rng.integers(0, 3, n))
        kf = own(rng.integers(0, 4, n) / 2.0)
        lexsort_case(f"U{n}", [k1])
        lexsort_case(f"UU{n}", [k1, k2])
        lexsort_case(f"UUS{n}", [k1, k2, k3])
        lexsort_case(f"Ui{n}", [k1, ki])
        lexsort_case(f"iU{n}", [ki, k1])
        lexsort_case(f"fSU{n}", [kf, k3, k1])
    m1 = plain(np.array(_pick(rng, 12, ["b", "a", "c", "ab"]), dtype="<U2").reshape(3, 4))
    m2 = plain(np.array(_pick(rng, 12, ["x", "y"]), dtype="<U1").reshape(3, 4))
    for ax in (0, 1, -1, 2, -3):
        lexsort_case(f"2d-ax{ax}", [m1, m2], axis=ax)
    lexsort_case("0d", [plain(np.array("b"))], axis=0)
    lexsort_case("0d-1", [plain(np.array("b"))], axis=-1)
    lexsort_case("e", [own(np.zeros(0, "<U2")), own(np.zeros(0, "<U2"))])
    lexsort_case("shapes", [own(np.array(["a", "b"])), own(np.array(["a", "b", "c"]))])
    lexsort_case("view", [operand("<U3", "s1", seed=1), operand("<U3", "r1", seed=2)])

    # ---- np.sort_complex -----------------------------------------------------------------------------------------------
    good = ["1+2j", "3", "-1j", "2.5", "1e1", "inf", "-inf", "nan", " 4 ", "+1-1j", "(1+1j)", "10", "9", "1+j", "-0"]
    for spec in ("<U7", "|S7"):
        kind = spec.lstrip("<|")[0]
        enc_ = (lambda vs: [v.encode() for v in vs]) if kind == "S" else (lambda vs: vs)
        cases = {"good": good, "bad": good[:4] + ["0x10"], "bad2": ["1+2j", "j1", "3"], "e": [], "one": ["7"],
                 "spaces": ["  1", "1  ", "1 + 2j"]}
        for key, vals in cases.items():
            arr = np.array(enc_(vals), dtype=spec)
            add(SETS, "sort_complex", operands=[own(arr)], layout=f"{key}{kind}", call=lambda a: np.sort_complex(a))
        arr2 = np.array(enc_(good[:12]), dtype=spec).reshape(3, 4)
        add(SETS, "sort_complex", operands=[plain(arr2)], layout=f"2d{kind}", call=lambda a: np.sort_complex(a))
        add(SETS, "sort_complex", operands=[plain(arr2.T)], layout=f"2dT{kind}", call=lambda a: np.sort_complex(a))
        add(SETS, "sort_complex", operands=[plain(np.array(enc_(["3"])[0], dtype=spec))], layout=f"0d{kind}",
            call=lambda a: np.sort_complex(a))

    # ---- np.histogram / histogram_bin_edges ------------------------------------------------------------------------------
    def hist(layout, fn, data, bins, *, weights=None, density=False, rng_=None):
        ops = [data]
        params = {"fn": fn, "range": list(rng_) if rng_ is not None else None, "density": density}
        if isinstance(bins, tuple):
            params["bins_op"] = len(ops)
            ops.append(bins)
            params["bins"] = None
        else:
            params["bins"] = bins
        if weights is not None:
            params["weights_op"] = len(ops)
            ops.append(weights)

        def call(*views, p=params):
            a = views[0]
            b = views[p["bins_op"]] if "bins_op" in p else p["bins"]
            w = views[p["weights_op"]] if "weights_op" in p else None
            r = tuple(p["range"]) if p["range"] is not None else None
            if p["fn"] == "histogram_bin_edges":
                return np.histogram_bin_edges(a, b, r, w)
            return np.histogram(a, b, r, density=p["density"], weights=w)

        add(SETS, "hist", params=params, operands=ops, layout=layout, call=call)

    # Words of at most two characters keep the 65,537-element operands (one past NumPy's block) small in the corpus.
    words = ["ap", "fi", "ki", "pe", "pl", "da", "", "fi"]
    for n in (8, 100, 65537):
        for kind in ("U", "S"):
            vals = _pick(rng, n, words)
            data = own(np.array([v.encode() for v in vals] if kind == "S" else vals, dtype=f"{'|' if kind == 'S' else '<'}{kind}2"))
            edges_t = own(np.array([b"ap", b"da", b"ki", b"pe"] if kind == "S" else ["ap", "da", "ki", "pe"]))
            lay = f"{kind}n{n}"
            hist(lay, "histogram", data, edges_t)
            hist(lay, "histogram", data, edges_t, weights=own(rng.integers(1, 5, n).astype(np.float64)))
            if n < 65537:
                hist(lay, "histogram", data, edges_t, density=True)
            if n < 65537 or kind == "S":
                hist(lay, "histogram", data, edges_t, weights=own(rng.integers(-3, 4, n)))
            if n in (8, 100):
                hist(lay, "histogram", data, edges_t, weights=own((np.arange(n) % 3 == 0)))
                hist(lay, "histogram", data, edges_t, weights=own(np.arange(n) * (1 + 1j)))
                hist(lay, "histogram", data, edges_t, weights=own(np.arange(n, dtype=np.float32)))
                hist(lay, "histogram_bin_edges", data, edges_t)
    numbers = own(np.array(["1", "2.5", "10", "3", "-1", "0.5", "100", "2"]))
    # Numeric edges come back as the operand itself: `plain` makes NumPy's operand the view the replay rebuilds (a 1-D
    # numeric base is rebuilt as a view, see `target`), so their OWNDATA agree.
    for edges, key in ((plain(np.array([0, 2, 5, 11])), "ints"), (plain(np.array([0.0, 2.5, 5.0])), "floats"),
                       (plain(np.array([1, 10, 100])), "pow"), (own(np.array(["1", "2", "3"])), "text"),
                       (own(np.array([b"1", b"2"])), "bytes")):
        hist(f"nums-{key}", "histogram", numbers, edges)
        hist(f"nums-{key}-w", "histogram", numbers, edges, weights=own(np.arange(8) * 0.5))
        hist(f"nums-{key}-d", "histogram", numbers, edges, density=True)
    for data, key in ((own(np.array([1, 2, 3, 10, 20])), "ints"), (own(np.array([0.5, 2.5, 30.0])), "floats")):
        hist(f"{key}-textedges", "histogram", data, own(np.array(["1", "2", "3"])))
    hist("decreasing", "histogram", own(np.array(["a", "b"])), own(np.array(["b", "a"])))
    hist("int-bins", "histogram", own(np.array(["a", "b"])), 3)
    hist("str-bins", "histogram", own(np.array(["a", "b"])), "auto")
    hist("2d-data", "histogram", plain(np.array([["b", "a"], ["c", "d"]])), own(np.array(["a", "c", "e"])))
    hist("t3-data", "histogram", operand("<U3", "t3", seed=4), own(np.array(["", "b", "d", "z"])))
    hist("empty", "histogram", own(np.zeros(0, "<U2")), own(np.array(["a", "b"])))
    hist("one-edge", "histogram", own(np.array(["a", "b"])), own(np.array(["a"])))
    hist("w-shape", "histogram", own(np.array(["a", "b"])), own(np.array(["a", "c"])), weights=own(np.ones(3)))

    # ---- np.quantile / np.percentile ------------------------------------------------------------------------------------
    methods = ["lower", "higher", "nearest", "inverted_cdf", "closest_observation", "linear", "midpoint",
               "averaged_inverted_cdf", "interpolated_inverted_cdf", "hazen", "weibull", "median_unbiased",
               "normal_unbiased"]
    t1 = _set_operand(rng, 9, "rand", "<U6")
    s1_ = _set_operand(rng, 9, "rand", "|S6")
    for key, op in (("U9", t1), ("S9", s1_)):
        for m in methods:
            for q in (0.0, 0.3, 0.5, 1.0):
                _quant(key, op, q=q, method=m)
            _quant(key, op, q=[0.0, 0.25, 0.8, 1.0], qkind="fl", method=m)
            _quant(key, op, q=1, qkind="i", method=m)
            _quant(key, op, q=0, qkind="i", method=m)
            _quant(key, op, name="percentile", q=50.0, method=m)
            _quant(key, op, name="percentile", q=[10.0, 90.0], qkind="fl", method=m)
    big = _set_operand(rng, 3001, "cat", "<U6")
    for m in ("lower", "higher", "nearest", "inverted_cdf", "closest_observation"):
        _quant("U3001", big, q=[0.0, 0.1, 0.5, 0.999, 1.0], qkind="fl", method=m)
    for qarr, qkey in ((np.array([0, 1]), "i64"), (np.array([True, False]), "bool"), (np.array(1, np.int8), "i8-0d"),
                       (np.array([[0.0, 0.5], [1.0, 0.25]]), "f2d"), (np.array([[0, 1]], np.uint8), "u8-2d"),
                       (np.zeros((1, 1, 1)), "f3d"), (np.zeros((0, 2)), "e2d"), (np.zeros(0), "e1d"),
                       (np.array([0.5], np.float16), "f16"), (np.array([0.25, 0.75], np.float32), "f32"),
                       (np.array([1.5]), "outside"), (np.array([np.nan]), "nan")):
        for m in ("lower", "linear", "nearest", "midpoint"):
            _quant(f"U9-{qkey}", t1, q=plain(qarr), qkind="arr", method=m)
        _quant(f"U9-{qkey}-pct", t1, name="percentile", q=plain(qarr * 100 if qarr.dtype.kind == "f" else qarr),
               qkind="arr", method="lower")
    # Axes: one axis, a tuple, every layout, keepdims.
    for lay in ("c2", "c3", "f2", "f3", "t3", "s2", "r2", "col", "bc", "bcol", "nx", "d5", "z0", "one", "e1", "e2", "e3"):
        for spec in ("<U3", "|S3"):
            op = operand(spec, lay, seed=8)
            v = op[1]
            axes = [None] + list(range(-v.ndim, v.ndim)) + ([(0, v.ndim - 1)] if v.ndim >= 2 else []) + \
                ([(0, 1, 2)] if v.ndim >= 3 else []) + ([()] if v.ndim >= 1 else [])
            for ax in axes:
                for kd in (False, True):
                    _quant(lay, op, q=0.5, axis=ax, method="lower", keepdims=kd)
                    _quant(lay, op, q=[0.0, 1.0], qkind="fl", axis=ax, method="nearest", keepdims=kd)
                _quant(lay, op, q=1, qkind="i", axis=ax, method="linear")
                _quant(lay, op, q=0.5, axis=ax, method="linear")
                _quant(lay, op, q=plain(np.array([[0.0, 1.0]])), qkind="arr", axis=ax, method="higher")
            _quant(lay, op, q=0.5, axis=v.ndim + 1, method="lower")
            if v.ndim >= 2:
                _quant(lay, op, q=0.5, axis=(0, 0), method="lower")
    # out=: NumPy's take checks (shape, the safe cast from out's dtype to the data's), the unsafe writeback.
    t32 = plain(np.array([["d", "b", "x"], ["a", "c", "y"]]))
    s32 = plain(np.array([[b"d", b"b", b"x"], [b"a", b"c", b"y"]]))
    for key, op in (("U", t32), ("S", s32)):
        # np.full, never np.empty: the out's prior bytes are part of the case and must not change between regenerations.
        for out_arr, okey in ((np.full(3, "zzzzz", "<U5"), "U5"), (np.full(3, "z", "<U1"), "U1"),
                              (np.full(3, b"zz", "|S2"), "S2"), (np.full(3, b"z", "|S1"), "S1"),
                              (np.full(2, "zz", "<U3"), "short"), (np.full((1, 3), "zz", "<U3"), "kd"),
                              (np.full(3, 7, np.int64), "int"), (np.full(3, 7.5, np.float64), "float"),
                              (np.full((2, 3), "zz", "<U3"), "q2"), (None, "rev")):
            if okey == "rev":
                ob = np.full(3, "zz", "<U3")
                o = (ob, ob[::-1])
            else:
                o = target(out_arr)
            _quant(f"{key}-out{okey}", op, q=0.5, axis=0, method="lower", out=o)
            _quant(f"{key}-out{okey}-kd", op, q=0.5, axis=0, method="lower", keepdims=True, out=o)
            _quant(f"{key}-out{okey}-q2", op, q=[0.0, 1.0], qkind="fl", axis=0, method="higher", out=o)
            _quant(f"{key}-out{okey}-lin", op, q=0.5, axis=0, method="linear", out=o)
        _quant(f"{key}-out0d", op, q=0.5, axis=None, method="lower", out=target(np.full((), "zz", "<U4")))
        _quant(f"{key}-out0d-kd", op, q=0.5, axis=None, method="lower", keepdims=True,
               out=target(np.full((1, 1), "zz", "<U4")))
        _quant(f"{key}-kd-scalar", op, q=0.5, axis=None, method="lower", keepdims=True)
        _quant(f"{key}-kd-scalar-ax", op, q=0.5, axis=(0, 1), method="lower", keepdims=True)
    # Empty reduced axes: the take path's IndexErrors and the interpolating methods' neighbour lookups.
    for shape, axes in (((0,), [None, 0]), ((0, 3), [None, 0, 1, (0, 1)]), ((3, 0), [0, 1]), ((0, 0), [0, 1]),
                        ((0, 2, 3), [(1, 2), (0, 2), 0])):
        op = plain(np.zeros(shape, "<U2"))
        for ax in axes:
            for m in methods:
                for q, qk in ((0.0, "f"), (0.5, "f"), (1.0, "f"), ([0.0, 0.5], "fl"), ([1.0, 0.0], "fl"), (1, "i"),
                              (0, "i"), ([], "fl")):
                    if m not in ("lower", "linear", "midpoint", "weibull", "inverted_cdf") and qk == "fl" and q:
                        continue
                    _quant(f"e{shape}", op, q=q, qkind=qk, axis=ax, method=m)
    _quant("badmethod", t1, q=0.5, method="foo")
    _quant("badmethod-q3d", t1, q=plain(np.zeros((1, 1, 1))), qkind="arr", method="foo")
    _quant("q-outside", t1, q=1.5, method="lower")
    _quant("pct-outside", t1, name="percentile", q=101.0, method="lower")


# ----------------------------------------------------------------------------------------------------------------------
# Sections tcomp / tarith / treduce: computing on NEP 55 StringDType text (strings phase 2, P2.T), strict
# ----------------------------------------------------------------------------------------------------------------------

# Every NA kind the StringDType loops tell apart: no NA object, None (the ordering / add / multiply refusals), a NaN (the
# missing value orders last and propagates), a sentinel string (it reads as its text), and coerce off.
T_COMPUTE_SPECS = [TD, TDN, TDNAN, TDS, TDC]
# Same-layout kinds the computing loops walk differently: contiguous 1-D / 2-D / 3-D, F-ordered, transposed, stepped,
# reversed, offset, a reversed step, a column, a new axis, broadcast rows / columns, 0-d, one element, 5-D, empty.
T_COMPUTE_LAYOUTS = ["c1", "c2", "c3", "f2", "t3", "s1", "s2", "r1", "r2", "o1", "rs", "col", "nx", "bc", "bcol", "z0",
                     "one", "d5", "e1", "e2", "e3"]
# The texts a comparison pair meets: short (in the cell), medium and long (arena / heap), the 15 / 16 boundary, NULs,
# non-ASCII and supplementary-plane characters, and the empty string.
T_CMP_POOL = ["a", "ab", "", "abc", "m" * 16, "m" * 15, "é" * 8, "q" * 20, "a\x00", "\U0001F600", "中文", "zz", "s" * 300,
              "a\x00b", "Ü", "x y"]


def _t_label(spec):
    """A short label of a TSpec for case layouts: ``T``, ``TN`` (None), ``TNAN``, ``TS`` (a string sentinel), ``TC``."""
    if spec.na is _ABSENT:
        return "T" if spec.coerce else "TC"
    if spec.na is None:
        return "TN" if spec.coerce else "TNC"
    if isinstance(spec.na, float):
        return "TNAN"
    return "TS"


def _t_derived(spec, src, seed):
    """Values for a binary loop's second StringDType operand, position by position from the first operand's values: the
    same text, its prefix one character shorter, the text with a character appended, or the text with a NUL appended (a
    longer text that compares greater), so equal texts, prefixes and longer texts meet; an NA object stays the first
    operand's NA object (a value of another kind for a descriptor without one: its ``str()``).

    :param spec: the second operand's TSpec.
    :param src: the first operand's values (texts and NA objects).
    :param seed: the derivation's phase.
    :returns: the derived values.
    """
    out = []
    for i, v in enumerate(src):
        if not isinstance(v, str):
            out.append(v if spec.na is not _ABSENT and spec.coerce else str(v))
            continue
        k = (i + seed) % 4
        out.append(v if k == 0 else v[:-1] if k == 1 else v + "z" if k == 2 else v + "\x00")
    return out


def _t_pair(spec1, l1, spec2, l2, seed=0, pool=None):
    """Two StringDType operands for a binary loop: the first from ``t_values`` over ``pool``, the second derived from it
    position by position (``_t_derived``) when both layouts have the same base length, else built independently.

    :param spec1: the first TSpec.
    :param l1: the first layout.
    :param spec2: the second TSpec.
    :param l2: the second layout.
    :param seed: the pool / derivation phase.
    :param pool: the texts (``T_CMP_POOL`` by default).
    :returns: two (base, view) pairs.
    """
    pool = pool or T_CMP_POOL
    n1, build1 = LAYOUTS[l1]
    n2, build2 = LAYOUTS[l2]
    v1 = t_values(spec1, n1, seed, pool, na_every=4)
    v2 = _t_derived(spec2, v1, seed) if n1 == n2 else t_values(spec2, n2, seed + 3, pool, na_every=5)
    b1 = np.array(v1, dtype=spec1.build()) if n1 else np.zeros(0, dtype=spec1.build())
    b2 = np.array(v2, dtype=spec2.build()) if n2 else np.zeros(0, dtype=spec2.build())
    return (b1, build1(b1)), (b2, build2(b2))


def gen_tcomp():
    """Section tcomp: the six comparisons over StringDType operands (both StringDType, StringDType against U on either
    side, the operators with a Python str), out= / where= / dtype=, NumPy's refusals, the truth readers and np.isnan."""
    # A. Both operands StringDType of one descriptor: every NA kind x every layout kind (the second operand derived from the
    #    first: equal texts, prefixes, longer texts, NA cells meeting NA cells and texts).
    for spec in T_COMPUTE_SPECS:
        for layout in T_COMPUTE_LAYOUTS:
            o1, o2 = _t_pair(spec, layout, spec, layout, seed=1)
            for name in CMP_OPS:
                add("tcomp", "ufunc", params={"name": name}, operands=[o1, o2],
                    call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"{_t_label(spec)}:{layout}")
    # B. Layout pairs that broadcast (or must refuse to), for the no-NA and the NaN-NA descriptors.
    for spec in [TD, TDNAN]:
        for l1, l2 in CMP_CROSS:
            o1, o2 = _t_pair(spec, l1, spec, l2, seed=2)
            for name in CMP_OPS:
                add("tcomp", "ufunc", params={"name": name}, operands=[o1, o2],
                    call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"{_t_label(spec)}:{l1},{l2}")
    # C. Two descriptors: compatible NA objects (one absent), incompatible ones (NumPy's TypeError), coerce mixes, and
    #    the NA rule of the FIRST descriptor deciding.
    for s1, s2 in [(TD, TDN), (TDN, TD), (TDN, TDNAN), (TDNAN, TDN), (TDS, TD), (TD, TDS), (TDNAN, TDS), (TDS, TDNAN),
                   (TDS, TSpec("other")), (TDN, TDNC), (TDC, TD), (TD, TDC), (TDNAN, TSpec(float("nan")))]:
        o1, o2 = _t_pair(s1, "c1", s2, "c1", seed=3)
        for name in CMP_OPS:
            add("tcomp", "ufunc", params={"name": name}, operands=[o1, o2],
                call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"{_t_label(s1)},{_t_label(s2)}")
    # D. StringDType against U (a U operand is cast to the default StringDType) on either side, against S and numbers (no
    #    loop), over the layouts that broadcast.
    for spec in [TD, TDN, TDNAN, TDS]:
        for layout in ["c2", "bc", "z0", "e2", "f2"]:
            o1 = t_operand(spec, layout, seed=4, pool=T_CMP_POOL)
            vals = [v if isinstance(v, str) else "NA" for v in t_values(spec, LAYOUTS[layout][0], 5, T_CMP_POOL, na_every=4)]
            bu = np.array([v.replace("\x00", "") for v in vals], dtype="U20") if vals else np.zeros(0, "U20")
            ou = (bu, LAYOUTS[layout][1](bu))
            for name in CMP_OPS:
                add("tcomp", "ufunc", params={"name": name}, operands=[o1, ou],
                    call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"{_t_label(spec)}:{layout},U")
                add("tcomp", "ufunc", params={"name": name}, operands=[ou, o1],
                    call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"U,{_t_label(spec)}:{layout}")
    o1 = t_operand(TD, "c2", seed=6, pool=T_CMP_POOL)
    for other in [plain(np.array([b"a", b"bc", b""] * 4, dtype="S3").reshape(3, 4)), num_operand(np.arange(12).reshape(3, 4)),
                  num_operand(np.array(1.5)), num_operand(np.array([True, False, True, True]))]:
        for name in ["equal", "less"]:
            add("tcomp", "ufunc", params={"name": name}, operands=[o1, other],
                call=lambda x, y, f=getattr(np, name): f(x, y), layout="T,other")
            add("tcomp", "ufunc", params={"name": name}, operands=[other, o1],
                call=lambda x, y, f=getattr(np, name): f(x, y), layout="other,T")
    # E. The operators with a Python str on either side (NumPy's weak str: a StringDType operand compares it as text).
    for spec in [TD, TDN, TDNAN, TDS]:
        for layout in ["c2", "z0", "e2"]:
            for sym in ["==", "!=", "<", "<=", ">", ">="]:
                for text in ["", "a", "NA", "nan", "é" * 8]:
                    for side in ["right", "left"]:
                        b, v = t_operand(spec, layout, seed=7, pool=T_CMP_POOL)
                        add("tcomp", "operator_text", params={"op": sym, "text": enc(text), "side": side},
                            operands=[(b, v)], call=lambda a, s=sym, t=text, sd=side: _text_op(s, t, a, sd),
                            layout=f"{_t_label(spec)}:{layout}")
    # F. out= (bool, integer, float, complex, text outs: the bool loop's output cast), where= masks, dtype=.
    for spec in [TD, TDNAN]:
        for name in CMP_OPS:
            o1, o2 = _t_pair(spec, "c2", spec, "bc", seed=8)
            for odt in ["?", "i1", "f8", "c16", "U5"]:
                out = target(np.zeros((3, 4), dtype=odt))
                add("tcomp", "ufunc_kw", params={"name": name, "nin": 2, "out": 2, "dtype": None},
                    operands=[o1, o2, out], call=lambda x, y, o, f=getattr(np, name): f(x, y, out=o),
                    layout=f"{_t_label(spec)}:out:{odt}")
            for mask in [np.array([[1, 0, 1, 0]], bool), np.eye(3, 4, dtype=bool), np.zeros((3, 4), bool)]:
                out = target(np.ones((3, 4), dtype=bool))
                add("tcomp", "ufunc_kw", params={"name": name, "nin": 2, "out": 2, "where": 3, "dtype": None},
                    operands=[o1, o2, out, plain(mask)],
                    call=lambda x, y, o, w, f=getattr(np, name): f(x, y, out=o, where=w), layout=f"{_t_label(spec)}:where")
            for dt in ["?", "f8"]:
                add("tcomp", "ufunc_kw", params={"name": name, "nin": 2, "dtype": dt}, operands=[o1, o2],
                    call=lambda x, y, d=dt, f=getattr(np, name): f(x, y, dtype=d), layout=f"{_t_label(spec)}:dtype:{dt}")
    # G. Truth values: astype(bool) (NumPy's truth of an NA: True for NaN, the string NA's own text, the default otherwise),
    #    any / all / count_nonzero flat and along an axis, nonzero / flatnonzero / argwhere, the logical ufuncs, isnan.
    for spec in T_COMPUTE_SPECS:
        for layout in T_COMPUTE_LAYOUTS:
            b, v = t_operand(spec, layout, seed=9, pool=["", "a", "", "bc", "m" * 16, "\x00", "q" * 20])
            add("tcomp", "astype", params={"dtype": "?", "casting": "unsafe", "order": "K", "copy": True},
                operands=[(b, v)], call=lambda a: a.astype(bool, order="K", casting="unsafe", copy=True),
                layout=f"{_t_label(spec)}:{layout}")
            for name in ["any", "all", "count_nonzero"] + ([] if v.ndim == 0 else ["nonzero", "flatnonzero", "argwhere"]):
                add("tcomp", "call", params={"name": name}, operands=[(b, v)],
                    call=lambda a, f=getattr(np, name): f(a), layout=f"{_t_label(spec)}:{layout}")
            if v.ndim >= 1:
                for name in ["any", "all", "count_nonzero"]:
                    for axis in [0, -1]:
                        for keepdims in [False, True]:
                            add("tcomp", "truth_axis", params={"name": name, "axis": axis, "keepdims": keepdims},
                                operands=[(b, v)],
                                call=lambda a, f=getattr(np, name), ax=axis, kd=keepdims: f(a, axis=ax, keepdims=kd),
                                layout=f"{_t_label(spec)}:{layout}")
            add("tcomp", "ufunc", params={"name": "isnan"}, operands=[(b, v)], call=lambda a: np.isnan(a),
                layout=f"{_t_label(spec)}:{layout}")
            add("tcomp", "ufunc", params={"name": "logical_not"}, operands=[(b, v)], call=lambda a: np.logical_not(a),
                layout=f"{_t_label(spec)}:{layout}")
    for name in ["logical_and", "logical_or", "logical_xor"]:
        for spec in [TD, TDN, TDNAN, TDS]:
            o1, o2 = _t_pair(spec, "c2", spec, "bc", seed=10, pool=["", "a", "", "bc"])
            add("tcomp", "ufunc", params={"name": name}, operands=[o1, o2], call=lambda x, y, f=getattr(np, name): f(x, y),
                layout=f"{_t_label(spec)}:T,T")
            add("tcomp", "ufunc", params={"name": name}, operands=[o1, num_operand(np.array([[0, 1, 2, 0]] * 3))],
                call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"{_t_label(spec)}:T,num")
            add("tcomp", "ufunc", params={"name": name}, operands=[o1, plain(np.array([["", "a", "b", ""]] * 3))],
                call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"{_t_label(spec)}:T,U")
    for spec in [TDNAN, TDN, TD]:
        b, v = t_operand(spec, "c2", seed=11, pool=T_CMP_POOL)
        for odt in ["?", "i1", "f8"]:
            out = target(np.ones((3, 4), dtype=odt))
            add("tcomp", "ufunc_kw", params={"name": "isnan", "nin": 1, "out": 1, "dtype": None}, operands=[(b, v), out],
                call=lambda x, o: np.isnan(x, out=o), layout=f"{_t_label(spec)}:isnan:out:{odt}")
        out = target(np.ones((3, 4), dtype=bool))
        add("tcomp", "ufunc_kw", params={"name": "isnan", "nin": 1, "out": 1, "where": 2, "dtype": None},
            operands=[(b, v), out, plain(np.eye(3, 4, dtype=bool))], call=lambda x, o, w: np.isnan(x, out=o, where=w),
            layout=f"{_t_label(spec)}:isnan:where")


def gen_tarith():
    """Section tarith: add / multiply / maximum / minimum over StringDType operands, the operators, out= / where= and the
    refusals."""
    # A. add and maximum / minimum, both operands StringDType of one descriptor, every NA kind x every layout kind.
    for spec in T_COMPUTE_SPECS:
        for layout in T_COMPUTE_LAYOUTS:
            o1, o2 = _t_pair(spec, layout, spec, layout, seed=21, pool=T_POOL)
            for name in ["add", "maximum", "minimum"]:
                add("tarith", "ufunc", params={"name": name}, operands=[o1, o2],
                    call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"{_t_label(spec)}:{layout}")
    for spec in [TD, TDNAN]:
        for l1, l2 in CMP_CROSS:
            o1, o2 = _t_pair(spec, l1, spec, l2, seed=22, pool=T_POOL)
            for name in ["add", "maximum", "minimum"]:
                add("tarith", "ufunc", params={"name": name}, operands=[o1, o2],
                    call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"{_t_label(spec)}:{l1},{l2}")
    # B. Two descriptors (the output takes the compatible NA object and the FIRST operand's rules), U on either side, S /
    #    numbers / bool (no loop).
    for s1, s2 in [(TD, TDN), (TDN, TD), (TDN, TDNAN), (TDS, TD), (TD, TDS), (TDNAN, TDS), (TDC, TD), (TD, TDC),
                   (TDN, TDNC)]:
        o1, o2 = _t_pair(s1, "c1", s2, "c1", seed=23, pool=T_POOL)
        for name in ["add", "maximum", "minimum"]:
            add("tarith", "ufunc", params={"name": name}, operands=[o1, o2],
                call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"{_t_label(s1)},{_t_label(s2)}")
    for spec in [TD, TDN, TDNAN, TDS]:
        o1 = t_operand(spec, "c2", seed=24, pool=T_POOL)
        ou = plain(np.array([["x", "", "yz", "é"]] * 3, dtype="U3"))
        for name in ["add", "maximum", "minimum"]:
            add("tarith", "ufunc", params={"name": name}, operands=[o1, ou],
                call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"{_t_label(spec)},U")
            add("tarith", "ufunc", params={"name": name}, operands=[ou, o1],
                call=lambda x, y, f=getattr(np, name): f(x, y), layout=f"U,{_t_label(spec)}")
    o1 = t_operand(TD, "c2", seed=25, pool=T_POOL)
    for other in [plain(np.array([[b"a", b"", b"cd", b"e"]] * 3, dtype="S2")), num_operand(np.arange(12).reshape(3, 4)),
                  num_operand(np.array(1.5)), num_operand(np.array(True))]:
        for name in ["add", "maximum", "minimum", "subtract"]:
            add("tarith", "ufunc", params={"name": name}, operands=[o1, other],
                call=lambda x, y, f=getattr(np, name): f(x, y), layout="T,other")
            add("tarith", "ufunc", params={"name": name}, operands=[other, o1],
                call=lambda x, y, f=getattr(np, name): f(x, y), layout="other,T")
    # C. multiply: the text times a count (0-d counts, per-element counts, a broadcast column, every integer width, the
    #    count first), the NA rules (None refuses, NaN stays NA, a string NA repeats its text), the refusals (a float /
    #    bool / text count, the repetition past the size limit).
    for spec in T_COMPUTE_SPECS:
        for layout in ["c1", "c2", "f2", "t3", "r2", "bc", "z0", "one", "e2", "d5"]:
            b, v = t_operand(spec, layout, seed=26, pool=T_POOL)
            counts = [plain(np.array(c)) for c in (2, 0, -1, 3)] + [plain((np.arange(v.size).reshape(v.shape) % 5) - 1)]
            for c in counts:
                add("tarith", "ufunc", params={"name": "multiply"}, operands=[(b, v), c],
                    call=lambda x, y: np.multiply(x, y), layout=f"{_t_label(spec)}:{layout}")
            add("tarith", "ufunc", params={"name": "multiply"}, operands=[plain(np.array(2)), (b, v)],
                call=lambda x, y: np.multiply(x, y), layout=f"{_t_label(spec)}:{layout}:countfirst")
    b, v = t_operand(TD, "c2", seed=27, pool=T_POOL)
    for dt in ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8"]:
        add("tarith", "ufunc", params={"name": "multiply"}, operands=[(b, v), plain(np.array([3], dtype=dt))],
            call=lambda x, y: np.multiply(x, y), layout=f"count:{dt}")
    for value, dt in [(2**63 + 5, "u8"), (2**62, "i8"), (2**61 + 1, "i8"), (-(2**63), "i8"), (2**40, "i8")]:
        add("tarith", "ufunc", params={"name": "multiply"}, operands=[(b, v), plain(np.array(value, dtype=dt))],
            call=lambda x, y: np.multiply(x, y), layout=f"count:{value}")
    for c in [plain(np.array(True)), plain(np.array(2.0)), plain(np.array([[2.5]])), t_operand(TD, "c1", seed=28)]:
        add("tarith", "ufunc", params={"name": "multiply"}, operands=[(b, v), c], call=lambda x, y: np.multiply(x, y),
            layout="count:other")
    # D. The operators: + with a Python str on either side, * with a Python int on either side.
    for spec in [TD, TDN, TDNAN, TDS]:
        for layout in ["c2", "z0", "e2", "t3"]:
            for text in ["", "a", "NA", "é" * 8, "q" * 20]:
                for side in ["right", "left"]:
                    b, v = t_operand(spec, layout, seed=29, pool=T_POOL)
                    add("tarith", "operator_text", params={"op": "+", "text": enc(text), "side": side},
                        operands=[(b, v)], call=lambda a, t=text, sd=side: _text_op("+", t, a, sd),
                        layout=f"{_t_label(spec)}:{layout}")
            for count in [0, 1, 3, -2]:
                for side in ["right", "left"]:
                    b, v = t_operand(spec, layout, seed=30, pool=T_POOL)
                    add("tarith", "t_operator_int", params={"op": "*", "value": count, "side": side}, operands=[(b, v)],
                        call=lambda a, c=count, sd=side: a * c if sd == "right" else c * a,
                        layout=f"{_t_label(spec)}:{layout}")
    # E. out=: a StringDType out of the result's descriptor and of other NA kinds, an S / U out (the output cast), a
    #    numeric out, where= masks (masked-off cells keep their text).
    for name, second in [("add", "text"), ("maximum", "text"), ("minimum", "text"), ("multiply", "count")]:
        for spec in [TD, TDNAN, TDN]:
            o1 = t_operand(spec, "c2", seed=31, pool=T_POOL)
            o2 = t_operand(spec, "bc", seed=32, pool=T_POOL) if second == "text" else plain(np.array([[0, 1, 2, 3]] * 3))
            for out_spec in ["T", "TN", "TNAN", "U9", "U2", "S9", "f8"]:
                if out_spec in ("T", "TN", "TNAN"):
                    ospec = {"T": TD, "TN": TDN, "TNAN": TDNAN}[out_spec]
                    out = t_plain(np.array([["k"] * 4] * 3, dtype=ospec.build()))
                else:
                    out = target(np.zeros((3, 4), dtype=out_spec))
                add("tarith", "ufunc_kw", params={"name": name, "nin": 2, "out": 2, "dtype": None}, operands=[o1, o2, out],
                    call=lambda x, y, o, f=getattr(np, name): f(x, y, out=o), layout=f"{_t_label(spec)}:out:{out_spec}")
                if CASES[-1].get("error", {}).get("type") == "SystemError":
                    _pop_case()
            for mask in [np.array([[1, 0, 1, 0]], bool), np.eye(3, 4, dtype=bool), np.zeros((3, 4), bool)]:
                out = t_plain(np.array([["k" * 20] * 4] * 3, dtype=spec.build()))
                add("tarith", "ufunc_kw", params={"name": name, "nin": 2, "out": 2, "where": 3, "dtype": None},
                    operands=[o1, o2, out, plain(mask)],
                    call=lambda x, y, o, w, f=getattr(np, name): f(x, y, out=o, where=w), layout=f"{_t_label(spec)}:where")
                # A cast out under where=: NumPy's masked loop runs only where the mask is set, so a refused NA that the
                # mask hides raises nothing and the masked-off cells keep their text.
                uout = target(np.full((3, 4), "q", dtype="U9"))
                add("tarith", "ufunc_kw", params={"name": name, "nin": 2, "out": 2, "where": 3, "dtype": None},
                    operands=[o1, o2, uout, plain(mask)],
                    call=lambda x, y, o, w, f=getattr(np, name): f(x, y, out=o, where=w),
                    layout=f"{_t_label(spec)}:where:U9")
    for spec in [TDN, TDNAN]:
        base = np.array([spec.na, "a", "bc"], dtype=spec.build())
        for name, second in [("add", "text"), ("maximum", "text"), ("multiply", "count")]:
            other = (base, base) if second == "text" else plain(np.array([2, 3, 1]))
            for mask in [[False, True, True], [True, False, True]]:
                for odt in ["U9", TD, TDN]:
                    out = target(np.full(3, "q", dtype=odt if isinstance(odt, str) else odt.build()))
                    add("tarith", "ufunc_kw", params={"name": name, "nin": 2, "out": 2, "where": 3, "dtype": None},
                        operands=[(base, base), other, out, plain(np.array(mask))],
                        call=lambda x, y, o, w, f=getattr(np, name): f(x, y, out=o, where=w),
                        layout=f"{_t_label(spec)}:where:hiddenNA:{odt if isinstance(odt, str) else _t_label(odt)}")
    # F. dtype= requests: the StringDType loops have no other output (a class or a descriptor instance).
    o1, o2 = _t_pair(TD, "c1", TD, "c1", seed=33)
    for name in ["add", "maximum", "minimum"]:
        for dt in ["f8", "U", TD]:
            add("tarith", "ufunc_kw", params={"name": name, "nin": 2, "dtype": senc(dt)}, operands=[o1, o2],
                call=lambda x, y, d=dt, f=getattr(np, name): f(x, y, dtype=d if isinstance(d, str) else sbuild(d)),
                layout=f"dtype:{dt if isinstance(dt, str) else 'T'}")


def _t_red(name, a, axis, keepdims, dtype, out, initial, where):
    """One reduction of section treduce, spelled with exactly the keywords the C# replay passes (``t_red``): an absent
    ``initial`` / ``where`` / ``out`` / ``dtype`` is left out of the call.

    :param name: the NumPy function.
    :param a: the array.
    :param axis: the axis, or None.
    :param keepdims: keepdims (only for the functions that take it).
    :param dtype: the dtype request (a dtype string or a TSpec) or None.
    :param out: the out array or None.
    :param initial: the initial value, or ``_ABSENT``.
    :param where: the mask, or None.
    :returns: NumPy's result.
    """
    f = getattr(np, name)
    kw = {"axis": axis}
    if dtype is not None:
        kw["dtype"] = dtype if isinstance(dtype, str) else sbuild(dtype)
    if out is not None:
        kw["out"] = out
    if name in ("sum", "max", "min", "amax", "amin", "nanmax", "nanmin", "argmax", "argmin", "nanargmax", "nanargmin",
                "nansum", "prod", "nanprod"):
        kw["keepdims"] = keepdims
    if initial is not _ABSENT:
        kw["initial"] = initial
    if where is not None:
        kw["where"] = where
    return f(a, **kw)


def _t_red_case(name, operands, axis=None, keepdims=False, dtype=None, out_index=None, initial=_ABSENT, where_index=None,
                layout=""):
    """Records one ``t_red`` case of section treduce.

    :param name: the NumPy function.
    :param operands: the operands (the array first; the out and the mask at their indices).
    :param axis: the axis, or None.
    :param keepdims: keepdims.
    :param dtype: the dtype request, or None.
    :param out_index: the out's operand index, or None.
    :param initial: the initial value, or ``_ABSENT``.
    :param where_index: the mask's operand index, or None.
    :param layout: the case's layout label.
    """
    params = {"name": name, "axis": axis, "keepdims": keepdims}
    if dtype is not None:
        params["dtype"] = senc(dtype)
    if out_index is not None:
        params["out"] = out_index
    if initial is not _ABSENT:
        params["initial"] = enc(initial)
    if where_index is not None:
        params["where"] = where_index

    def call(*ops):
        return _t_red(name, ops[0], axis, keepdims, dtype, ops[out_index] if out_index is not None else None, initial,
                      ops[where_index] if where_index is not None else None)

    add("treduce", "t_red", params=params, operands=operands, call=call, layout=layout)


def _t_stat(name, a, axis, keepdims, dtype, ddof, out, weights):
    """One statistic of section treduce, spelled as the C# replay's ``t_stat`` calls it.

    :param name: the NumPy function (mean / average / nanmean / var / std / nanvar / nanstd / ptp / median).
    :param a: the array.
    :param axis: the axis, or None.
    :param keepdims: keepdims.
    :param dtype: the dtype request, or None.
    :param ddof: the ddof (var / std forms), or None.
    :param out: the out (ptp), or None.
    :param weights: the weights (average), or None.
    :returns: NumPy's result.
    """
    f = getattr(np, name)
    kw = {"axis": axis}
    if name != "average" or keepdims:
        kw["keepdims"] = keepdims
    if dtype is not None:
        kw["dtype"] = dtype
    if ddof is not None:
        kw["ddof"] = ddof
    if out is not None:
        kw["out"] = out
    if weights is not None:
        kw["weights"] = weights
    return f(a, **kw)


def _t_stat_case(name, operands, axis=None, keepdims=False, dtype=None, ddof=None, out_index=None, weights_index=None,
                 layout="", section="treduce"):
    """Records one ``t_stat`` case of section treduce, or of ``section`` (tsort's median / nanmedian; see ``_t_stat``)."""
    params = {"name": name, "axis": axis, "keepdims": keepdims}
    if dtype is not None:
        params["dtype"] = dtype
    if ddof is not None:
        params["ddof"] = ddof
    if out_index is not None:
        params["out"] = out_index
    if weights_index is not None:
        params["weights"] = weights_index

    def call(*ops):
        return _t_stat(name, ops[0], axis, keepdims, dtype, ddof, ops[out_index] if out_index is not None else None,
                       ops[weights_index] if weights_index is not None else None)

    add(section, "t_stat", params=params, operands=operands, call=call, layout=layout)


def gen_treduce():
    """Section treduce: np.sum / cumsum / max / min / argmax / argmin and their nan forms over StringDType along every
    axis, the keywords (initial=, where=, out=, dtype=), np.trace, and the statistics that compute first and refuse after."""
    full_layouts = ["c1", "c2", "c3", "f2", "t3", "s2", "r2", "rs", "bc", "bcol", "nx", "z0", "one", "d5", "e1", "e2", "e3"]
    # A. Every reduction along every axis (None, 0, -1, 1 for two or more dimensions) with and without keepdims, every
    #    NA kind (a None-like NA refuses at its first addition / comparison, a NaN-like one propagates / orders last).
    for spec in T_COMPUTE_SPECS:
        for layout in full_layouts:
            b, v = t_operand(spec, layout, seed=41, pool=T_POOL, na_every=4 if spec is not TDN else 7)
            axes = [None, 0, -1] + ([1] if v.ndim >= 2 else [])
            for axis in axes:
                for name in ["sum", "max", "min", "argmax", "argmin"]:
                    for keepdims in [False, True]:
                        _t_red_case(name, [(b, v)], axis=axis, keepdims=keepdims, layout=f"{_t_label(spec)}:{layout}")
                _t_red_case("cumsum", [(b, v)], axis=axis, layout=f"{_t_label(spec)}:{layout}")
            for name in ["nansum", "nancumsum", "nanargmax", "nanargmin", "amax", "amin"]:
                _t_red_case(name, [(b, v)], axis=None if v.ndim != 1 else 0, layout=f"{_t_label(spec)}:{layout}")
    # NA lanes that the rules decide: a lone NA, NA first / last, all NA, the None-like NA alone in a lane.
    for spec in [TDN, TDNAN, TDS]:
        for vals in [[spec.na], [spec.na, "a"], ["a", spec.na], [spec.na, spec.na], ["b", spec.na, "c", spec.na, "a"]]:
            base = np.array(vals, dtype=spec.build())
            for name in ["sum", "max", "min", "argmax", "argmin", "cumsum"]:
                _t_red_case(name, [(base, base)], layout=f"{_t_label(spec)}:na{len(vals)}")
            m = np.array([vals, vals[::-1]], dtype=spec.build())
            mb, mv = t_plain(m)
            for name in ["sum", "max", "min", "argmax", "argmin", "cumsum"]:
                for axis in [0, 1]:
                    _t_red_case(name, [(mb, mv)], axis=axis, layout=f"{_t_label(spec)}:na2d")
    # B. Keywords: initial= (a text, an integer under coerce, the NA object, a text under coerce=False), where= with and
    #    without initial=, out= (StringDType of the same / another NA kind, S / U, numeric, wrong shape), dtype=.
    for spec in [TD, TDNAN, TDS, TDC]:
        b, v = t_operand(spec, "c2", seed=42, pool=T_POOL)
        for name in ["max", "min"]:
            for initial in ["", "zz", "m" * 20, 5, spec.na if spec.na is not _ABSENT else "NA"]:
                for axis in [None, 0, 1] if spec is TD else [1]:
                    _t_red_case(name, [(b, v)], axis=axis, initial=initial, layout=f"{_t_label(spec)}:initial")
            for mask in [np.array([[1, 0, 1, 0]], bool), np.eye(3, 4, dtype=bool), np.zeros((3, 4), bool)]:
                for initial in [_ABSENT, "", "zz"]:
                    _t_red_case(name, [(b, v), plain(mask)], axis=1, initial=initial, where_index=1,
                                layout=f"{_t_label(spec)}:where")
            eb, ev = t_operand(spec, "e2", seed=43, pool=T_POOL)
            _t_red_case(name, [(eb, ev)], axis=0, initial="q", layout=f"{_t_label(spec)}:initial:empty")
    for spec in [TD, TDNAN, TDN]:
        b, v = t_operand(spec, "c2", seed=44, pool=T_POOL, na_every=4 if spec is not TDN else 0)
        for name in ["max", "min"]:
            for out_spec in ["T", "TN", "TNAN", "U9", "f8", "T3"]:
                if out_spec in ("T", "TN", "TNAN"):
                    ospec = {"T": TD, "TN": TDN, "TNAN": TDNAN}[out_spec]
                    out = target(np.array(["k"] * 3, dtype=ospec.build()))
                elif out_spec == "T3":
                    out = target(np.array(["k"] * 4, dtype=TD.build()))
                else:
                    out = target(np.zeros(3, dtype=out_spec))
                _t_red_case(name, [(b, v), out], axis=1, out_index=1, layout=f"{_t_label(spec)}:out:{out_spec}")
        for name in ["argmax", "argmin"]:
            for odt in ["i8", "i4", "u8", "f8"]:
                out = target(np.zeros(3, dtype=odt))
                _t_red_case(name, [(b, v), out], axis=1, out_index=1, layout=f"{_t_label(spec)}:out:{odt}")
            out = target(np.zeros((3, 1), dtype="i8"))
            _t_red_case(name, [(b, v), out], axis=1, keepdims=True, out_index=1, layout=f"{_t_label(spec)}:out:keepdims")
        for name in ["cumsum", "nancumsum"]:
            for out_spec in ["T", "TNAN", "U30", "U2", "wrong"]:
                if out_spec in ("T", "TNAN"):
                    ospec = {"T": TD, "TNAN": TDNAN}[out_spec]
                    out = t_plain(np.array([["k"] * 4] * 3, dtype=ospec.build()))
                elif out_spec == "wrong":
                    out = target(np.array(["k"] * 12, dtype=TD.build()))
                else:
                    out = target(np.zeros((3, 4), dtype=out_spec))
                _t_red_case(name, [(b, v), out], axis=1, out_index=1, layout=f"{_t_label(spec)}:out:{out_spec}")
        for name in ["sum", "cumsum"]:
            for dt in ["f8", "U", TD]:
                _t_red_case(name, [(b, v)], axis=1, dtype=dt, layout=f"{_t_label(spec)}:dtype")
    # The accumulation's out= on NumPy's two paths. The DIRECT path (1-D, a StringDType out) returns at once for an empty
    # input or an empty out and otherwise wants the input's exact dimensions; the ITERATOR path (2-D, a U out) raises the
    # iterator's op_axes / remapped-broadcast / non-broadcastable texts, accepts an out with more dimensions (its
    # trailing axes stay at index 0), broadcasts the input on the other axes and writes only the input's extent along
    # the axis. A U out casts the default StringDType() result: an NA first element is the input's na_name, a later NA
    # reads as ''. (A 0-d StringDType out over a 1-D input crashes NumPy 2.4.2 on its direct path: not generated.)
    m2 = np.array([["a", "b"], ["c", "d"]], dtype=TD.build())
    col = np.array([["a"], ["bb"]], dtype=TD.build())
    row = np.array([["x", "yy", "z"]], dtype=TD.build())
    v3 = np.array(["a", "b", "c"], dtype=TD.build())
    e0 = np.zeros(0, dtype=TD.build())
    kfill = lambda shape, dt=TD: np.full(shape, "k", dtype=dt.build() if isinstance(dt, TSpec) else dt)
    for arr, axis, out_arr, label in [
            (m2, 0, kfill(2), "acc:2d:out1d"), (m2, 0, kfill((2, 3)), "acc:2d:outwide"),
            (m2, 0, kfill((3, 2)), "acc:2d:outtall"), (m2, 0, kfill((2, 2, 2)), "acc:2d:out3d"),
            (m2, 1, kfill((2, 2, 3)), "acc:2d:out3d:axis1"), (m2, 0, kfill((1, 2)), "acc:2d:outrow"),
            (m2, 1, kfill((2, 1)), "acc:2d:outcol"), (m2, 1, kfill((2, 2), "U5"), "acc:2d:U5"),
            (m2, 0, kfill((2, 2), "U1"), "acc:2d:U1"), (m2, None, kfill(4), "acc:flat:out4"),
            (m2, None, kfill(3), "acc:flat:out3"), (m2, None, kfill((2, 2)), "acc:flat:out2d"),
            (col, 0, kfill((2, 3)), "acc:bcast:col"), (row, 0, kfill((2, 3)), "acc:bcast:rowaxis"),
            (row, 1, kfill((4, 3)), "acc:bcast:row"), (v3, None, kfill(2), "acc:1d:short"),
            (v3, None, kfill(4), "acc:1d:long"), (v3, None, kfill((1, 3)), "acc:1d:2dout"),
            (v3, None, kfill(0), "acc:1d:emptyout"), (e0, None, kfill(5), "acc:emptyin"),
            (v3, None, kfill(2, "U5"), "acc:1d:U:short"), (v3, None, kfill((1, 3), "U5"), "acc:1d:U:row"),
            (v3, None, np.array("k", dtype="U5"), "acc:1d:U:0d"), (v3, 0, kfill((3, 2), "U5"), "acc:1d:U:extra")]:
        ab, av = t_plain(arr)
        _t_red_case("cumsum", [(ab, av), target(out_arr)], axis=axis, out_index=1, layout=f"T:{label}")
    for spec in [TDNAN, TDN, TDS]:
        for vals in [[spec.na, "a", "b"], ["a", spec.na, "b"], [spec.na, spec.na]]:
            base = np.array(vals, dtype=spec.build())
            for odt in ["U9", TD]:
                out = target(np.full(len(vals), "k", dtype=odt if isinstance(odt, str) else odt.build()))
                _t_red_case("cumsum", [(base, base), out], out_index=1,
                            layout=f"{_t_label(spec)}:acc:firstna:{odt if isinstance(odt, str) else 'T'}")
            # np.trace's add.reduce into a U out: the same first-element cast (the diagonal starts with the NA).
            sq = np.array([vals[0], "x", "y", vals[1]], dtype=spec.build())
            sb, sv = t_plain(sq.reshape(2, 2))
            add("treduce", "t_trace", params={"offset": 0, "axis1": 0, "axis2": 1, "dtype": None, "out": 1},
                operands=[(sb, sv), target(np.array("", dtype="U9"))], call=lambda a, o: np.trace(a, out=o),
                layout=f"{_t_label(spec)}:trace:firstna:U9")
    # C. The statistics: the sum / extremes compute, then NumPy's next statement refuses in the operands' words (an ndarray
    #    against the divide / subtract loops, a 0-d result as its Python object); the NaN-like NA's mean is np.float64(nan).
    for spec in [TD, TDN, TDNAN, TDS]:
        for layout in ["c1", "c2", "f2", "bc", "z0", "one", "e1", "e2"]:
            for vals_seed, na_every in [(45, 4), (46, 0)]:
                b, v = t_operand(spec, layout, seed=vals_seed, pool=T_POOL, na_every=na_every)
                for name in ["mean", "average", "nanmean", "var", "std", "nanvar", "nanstd", "ptp"]:
                    for axis in [None, 0] if v.ndim >= 1 else [None]:
                        for keepdims in [False, True]:
                            _t_stat_case(name, [(b, v)], axis=axis, keepdims=keepdims, layout=f"{_t_label(spec)}:{layout}")
        for vals in [["x"]] + ([[spec.na]] if spec.na is not _ABSENT else []):
            base = np.array(vals, dtype=spec.build())
            z = base.reshape(())
            for name in ["mean", "average", "nanmean", "var", "std", "nanvar", "nanstd", "ptp"]:
                _t_stat_case(name, [(base, base)], layout=f"{_t_label(spec)}:lone")
                _t_stat_case(name, [(base, z)], layout=f"{_t_label(spec)}:lone0d")
    b, v = t_operand(TD, "c2", seed=47, pool=T_POOL)
    for name in ["mean", "var", "std", "nanmean", "nanvar", "nanstd"]:
        _t_stat_case(name, [(b, v)], axis=0, dtype="f8", layout="T:dtype")
    for ddof in [0, 1, 5]:
        _t_stat_case("var", [(b, v)], axis=1, ddof=ddof, layout="T:ddof")
    for wdt in ["i8", "f8"]:
        _t_stat_case("average", [(b, v), plain(np.ones((3, 4), dtype=wdt))], axis=1, weights_index=1, layout=f"T:weights:{wdt}")
    _t_stat_case("average", [(b, v), t_plain(np.array([["a"] * 4] * 3, dtype=TD.build()))], axis=1, weights_index=1,
                 layout="T:weights:T")
    for out_spec in ["T", "U5"]:
        out = target(np.array(["k"] * 3, dtype=TD.build())) if out_spec == "T" else target(np.zeros(3, dtype=out_spec))
        _t_stat_case("ptp", [(b, v), out], axis=1, out_index=1, layout=f"T:ptp:out:{out_spec}")
    # D. np.trace: the diagonal's add.reduce along its last axis (2-D, 3-D, offsets, out=, dtype=), every NA kind.
    for spec in [TD, TDN, TDNAN, TDS]:
        for shape, axes in [((3, 3), (0, 1)), ((3, 4), (0, 1)), ((2, 3, 3), (1, 2)), ((3, 2, 3), (0, 2)), ((0, 3), (0, 1))]:
            n = int(np.prod(shape))
            base = np.array(t_values(spec, n, 48, T_POOL, na_every=5), dtype=spec.build()) if n else np.zeros(0, spec.build())
            view = base.reshape(shape)
            for offset in [0, 1, -1]:
                add("treduce", "t_trace", params={"offset": offset, "axis1": axes[0], "axis2": axes[1], "dtype": None},
                    operands=[(base, view)], call=lambda a, o=offset, ax=axes: np.trace(a, offset=o, axis1=ax[0], axis2=ax[1]),
                    layout=f"{_t_label(spec)}:{shape}")
    base = np.array(t_values(TD, 9, 49, T_POOL), dtype=TD.build())
    view = base.reshape(3, 3)
    for out_spec in ["T", "TN", "U9", "wrong"]:
        if out_spec in ("T", "TN"):
            out = t_plain(np.array("k", dtype=(TD if out_spec == "T" else TDN).build()))
        elif out_spec == "wrong":
            out = target(np.array(["k", "k"], dtype=TD.build()))
        else:
            out = target(np.array("", dtype="U9"))
        add("treduce", "t_trace", params={"offset": 0, "axis1": 0, "axis2": 1, "dtype": None, "out": 1},
            operands=[(base, view), out], call=lambda a, o: np.trace(a, out=o), layout=f"T:trace:out:{out_spec}")
    for dt in ["f8", TD]:
        add("treduce", "t_trace", params={"offset": 0, "axis1": 0, "axis2": 1, "dtype": senc(dt)}, operands=[(base, view)],
            call=lambda a, d=dt: np.trace(a, dtype=d if isinstance(d, str) else sbuild(d)), layout="T:trace:dtype")
    # E. The reductions with no StringDType loop: multiply / fmax / fmin.
    for spec in T_COMPUTE_SPECS:
        b, v = t_operand(spec, "c1", seed=50, pool=T_POOL)
        for name in ["prod", "nanprod", "cumprod", "nancumprod", "nanmax", "nanmin"]:
            _t_red_case(name, [(b, v)], layout=f"{_t_label(spec)}:noloop")


# ----------------------------------------------------------------------------------------------------------------------
# Section tsort: sorting, searching, the set routines and the order statistics over NEP 55 StringDType (strings phase 2,
# P2.2 over StringDType)
# ----------------------------------------------------------------------------------------------------------------------

# Texts for the sorts: every cell kind (short, arena, heap), embedded NULs ('a' < 'a\x00' < 'ab'), non-ASCII and
# supplementary-plane text, and texts equal in their first 8 / 15 bytes.
T_SORT_POOL = ["", "a", "a\x00", "ab", "abc", "b", "bb", "c", "x" * 15, "x" * 14, "é", "NB", "m" * 16, "q" * 300, "中文",
               "\U0001F600", "ab\x00", "xxxxxxxxy"]
# The bulk of the sort cases: the same kinds of text without the heap-sized one (every case carries its operand's and its
# result's values, so a 300-character text repeated over the catalog would dominate the section's size).
T_SORT_SHORT = [v for v in T_SORT_POOL if len(v) < 100]
# The layouts whose sorts also get the heap-sized text.
T_SORT_LONG_LAYOUTS = ("c1", "c2", "t3")
# Haystack texts of the searches stay in their cells (at most 15 UTF-8 bytes): NumPy 2.4.2's searchsorted loads haystack
# cells with the KEYS' allocator, so an arena or heap haystack text raises MemoryError or reads another array's arena — a
# NumPy bug NumSharp does not reproduce. Keys may be any length.
T_SEARCH_POOL = ["", "a", "a\x00", "ab", "abc", "b", "bb", "c", "x" * 15, "x" * 14, "é", "NB", "中文"]
T_SET_POOL = ["pear", "fig", "apple", "", "fi", "fig\x00x", "kiwi", "m" * 20, "é"]
T_ORDER_SPECS = [TD, TDN, TDNAN, TDS, TDC]


def _ts_values(rng, n, pool, spec, na_rate):
    """``n`` values drawn from ``pool``, each an NA (the spec's NA object) with probability ``na_rate`` when it has one."""
    vals = []
    for _ in range(n):
        if spec.na is not _ABSENT and rng.random() < na_rate:
            vals.append(spec.na)
        else:
            vals.append(pool[int(rng.integers(0, len(pool)))])
    return vals


def _ts_array(vals, spec):
    """A StringDType array of ``vals`` (an empty one keeps the descriptor)."""
    return np.array(vals, dtype=spec.build()) if vals else np.zeros(0, dtype=spec.build())


def _ts_sorted(arr):
    """``arr`` sorted with NumPy's stable sort when it can be (a None-like NA refuses: then as is)."""
    try:
        return arr[np.argsort(arr, kind="stable")]
    except Exception:  # noqa: BLE001 -- an unsortable haystack is searched as is
        return arr


def gen_tsort():
    """Section tsort: StringDType's part of P2.2. Every case is phase 1: NumSharp sorts, searches and runs the set routines
    and the order statistics over ``StringDType``, so it must match NumPy now.

    * the sort family (``np.sort`` / ``np.argsort`` / ``np.partition`` / ``np.argpartition`` / ``ndarray.sort``) over every
      NA kind and every layout of the catalog, every axis, the kinds (``heapsort`` refused), NumPy's ordering of missing
      values (a NaN-like NA last, a string NA as its text, a None-like NA refused in a line of two or more), categorical
      data around the introsort's insertion boundary and the mergesort's runs;
    * ``np.lexsort`` (NaN-like NAs in adjacent runs: NumPy's timsort keeps them reversed), mixed with numeric and ``U``
      keys;
    * ``np.searchsorted`` (sides, sorters, ``U`` / numeric / bytes keys and their refusals, NA keys of every kind) and
      ``np.digitize``;
    * ``np.unique`` (every flag, ``equal_nan``, ``sorted``), the Array-API family (``unique_values`` is NumPy's hash order:
      an excused order), ``np.isin`` and the four set routines between StringDType, ``U``, bytes and numbers;
    * ``np.sort_complex`` (its parse errors and missing data; NumPy 2.4.2's StringDType-to-complex cast stores the real part
      in both parts, so only zeros are compared as values);
    * ``np.histogram`` / ``histogram_bin_edges`` / ``histogram2d`` / ``histogramdd`` over StringDType data, edges and weights;
    * ``np.quantile`` / ``np.percentile`` / ``np.nanquantile`` / ``np.nanpercentile`` (every method, q forms, axes,
      keepdims, ``out=``, a StringDType q), ``np.median`` / ``np.nanmedian``;
    * numpy.ma's ``sort`` / ``argsort`` / ``median`` / ``unique`` over StringDType masked arrays.
    """
    rng = np.random.default_rng(20261016 + 7)

    # ---- the sort family over the catalog ------------------------------------------------------------------------------
    sort_layouts = ["c1", "c2", "c3", "f2", "t3", "s1", "s2", "r1", "r2", "rs", "o1", "col", "nx", "bc", "bcol", "z0",
                    "one", "d5", "e1", "e2", "e3"]
    argsort_kws = [{}, {"kind": "stable"}, {"kind": "heapsort"}, {"stable": True}]
    sort_kws = [{}, {"kind": "stable"}, {"kind": "heapsort"}]
    for spec in T_ORDER_SPECS:
        for na_every in ([5, 0] if spec is TDN else [5]):
            for lay in sort_layouts:
                # A broadcast layout holds texts of at most 15 bytes: NumPy 2.4.2's argsort of a broadcast StringDType view
                # misorders arena texts (np.argsort(np.broadcast_to(b, (3, 4)), axis=None) puts a 16-byte text before
                # shorter, smaller ones; the same array copied sorts right) — a NumPy bug NumSharp does not reproduce.
                def make(s=spec, l=lay, ne=na_every):
                    pool = T_SEARCH_POOL if l in ("bc", "bcol") else T_SORT_POOL if l in T_SORT_LONG_LAYOUTS else T_SORT_SHORT
                    return t_operand(s, l, seed=11, pool=pool, na_every=ne)
                op = make()
                view = op[1]
                axes = [None] + list(range(-1, view.ndim)) if view.ndim else [None, -1]
                label = f"{_t_label(spec)}{'' if na_every else '-nona'}:{lay}"
                for ax in axes:
                    for kw in argsort_kws:
                        add("tsort", "argsort_kw", params=_sort_params(kw, axis=ax), operands=[op], layout=label,
                            call=lambda a, x=ax, k=kw: np.argsort(a, axis=x, **k))
                    for kw in sort_kws:
                        add("tsort", "sort_kw", params=_sort_params(kw, axis=ax), operands=[op], layout=label,
                            call=lambda a, x=ax, k=kw: np.sort(a, axis=x, **k))
                    if view.ndim > 0:
                        n_axis = view.size if ax is None else view.shape[ax]
                        for kth in ([-1, 0] if n_axis > 1 else [0]):
                            add("tsort", "partition_kw", params={"axis": ax, "kth": kth}, operands=[op], layout=label,
                                call=lambda a, x=ax, k=kth: np.partition(a, k, axis=x))
                            add("tsort", "argpartition_kw", params={"axis": ax, "kth": kth}, operands=[op], layout=label,
                                call=lambda a, x=ax, k=kth: np.argpartition(a, k, axis=x))
                    if ax is not None and view.ndim > 0 and view.flags.writeable:
                        # One value sort (every kind sorts the same values) and the heapsort refusal.
                        for kw in ({}, {"kind": "heapsort"}):
                            add("tsort", "ndsort_kw", params=_sort_params(kw, axis=ax), operands=[make()], layout=label,
                                mutate=0, call=lambda a, x=ax, k=kw: a.sort(axis=x, **k))

    # Categorical data with duplicates and NAs: the insertion-only boundary (16 / 17), the introsort's partitions and the
    # mergesort's runs; NaN-like NAs at the ends, in the middle and everywhere.
    for spec in (TD, TDNAN, TDS):
        for n in (2, 15, 16, 17, 18, 33, 100, 1000):
            for k in (2, 6):
                for na_rate in ((0.0, 0.3) if spec.na is not _ABSENT else (0.0,)):
                    vals = _ts_values(rng, n, T_SORT_SHORT[:k] + ["zz"], spec, na_rate)
                    arr = _ts_array(vals, spec)
                    lab = f"{_t_label(spec)}:cat-n{n}k{k}na{na_rate}"
                    kws = argsort_kws if n in (17, 100) else [{}, {"kind": "stable"}]
                    for kw in kws:
                        add("tsort", "argsort_kw", params=_sort_params(kw, axis=-1), operands=[plain(arr)], layout=lab,
                            call=lambda a, k=kw: np.argsort(a, **k))
                    add("tsort", "sort_kw", params=_sort_params({}, axis=-1), operands=[plain(arr)], layout=lab,
                        call=lambda a: np.sort(a))
                    if n in (16, 17, 100):
                        for kth in (0, n // 2, -1):
                            add("tsort", "partition_kw", params={"axis": -1, "kth": kth}, operands=[plain(arr)],
                                layout=lab, call=lambda a, kk=kth: np.partition(a, kk))
                            add("tsort", "argpartition_kw", params={"axis": -1, "kth": kth}, operands=[plain(arr)],
                                layout=lab, call=lambda a, kk=kth: np.argpartition(a, kk))
    # NA placement on the edges: a lone NA, NAs first / last / interleaved, NA-only lines, a None-like NA alone (no compare).
    for spec in (TDN, TDNAN, TDS):
        for vals in ([spec.na], [spec.na, "a"], ["a", spec.na], [spec.na, spec.na], ["b", spec.na, "c", spec.na, "a"],
                     [spec.na, "", spec.na, "a\x00"]):
            arr = _ts_array(vals, spec)
            lab = f"{_t_label(spec)}:na{len(vals)}"
            for kw in argsort_kws:
                add("tsort", "argsort_kw", params=_sort_params(kw, axis=-1), operands=[plain(arr)], layout=lab,
                    call=lambda a, k=kw: np.argsort(a, **k))
            add("tsort", "sort_kw", params=_sort_params({}, axis=-1), operands=[plain(arr)], layout=lab,
                call=lambda a: np.sort(a))
            m = np.array([vals, vals[::-1]], dtype=spec.build())
            for ax in (0, 1):
                add("tsort", "argsort_kw", params=_sort_params({"kind": "stable"}, axis=ax), operands=[t_plain(m)],
                    layout=f"{lab}:2d", call=lambda a, x=ax: np.argsort(a, axis=x, kind="stable"))

    # ---- np.lexsort ----------------------------------------------------------------------------------------------------
    def lexsort_case(layout, keys, axis=-1):
        add("tsort", "lexsort", params={"axis": axis}, operands=keys, layout=layout,
            call=lambda *ks, x=axis: np.lexsort(ks, axis=x))

    for spec in T_ORDER_SPECS:
        for n in (5, 17, 40, 300):
            k1 = own(_ts_array(_ts_values(rng, n, ["b", "a", "c", "ab", ""], spec, 0.25), spec))
            k2 = own(_ts_array(_ts_values(rng, n, ["x", "y"], spec, 0.25), spec))
            kn = own(rng.integers(0, 3, n))
            ku = own(np.array(_pick(rng, n, ["q", "p", "pq"]), dtype="<U2"))
            lab = f"{_t_label(spec)}:n{n}"
            lexsort_case(f"{lab}:T", [k1])
            lexsort_case(f"{lab}:TT", [k1, k2])
            lexsort_case(f"{lab}:Ti", [k1, kn])
            lexsort_case(f"{lab}:iT", [kn, k1])
            lexsort_case(f"{lab}:UT", [ku, k1])
            lexsort_case(f"{lab}:TU", [k1, ku])
    # Adjacent NaN-like NAs in the primary key (NumPy's timsort run detection reverses them) and in a secondary key.
    for vals in ([np.nan, np.nan, "a"], ["a", np.nan, np.nan, np.nan], [np.nan] * 5, ["b", np.nan, "a", np.nan, np.nan]):
        k = own(np.array(vals, dtype=TDNAN.build()))
        lexsort_case(f"T-NaN:{len(vals)}runs", [k])
        lexsort_case(f"T-NaN:{len(vals)}runs:sec", [own(np.arange(len(vals))[::-1].copy()), k])
    # Keys contiguous along the axis only: NumPy 2.4.2 CRASHES (access violation) in np.lexsort when a StringDType key is
    # not — axis 0 of a C-ordered 2-D key, a stepped or reversed 1-D view — on its copy-to-buffer path (a NumPy bug;
    # NumSharp's answer is pinned by the unit tests).
    m1 = t_plain(np.array(_pick(rng, 12, ["b", "a", "c", "ab"]), dtype=TD.build()).reshape(3, 4))
    m2 = t_plain(np.array(_pick(rng, 12, ["x", "y", np.nan]), dtype=TDNAN.build()).reshape(3, 4))
    for ax in (1, -1, 2):
        lexsort_case(f"T:2d-ax{ax}", [m1, m2], axis=ax)
    lexsort_case("T:0d", [t_plain(np.array("b", dtype=TD.build()))], axis=0)
    lexsort_case("T:e", [own(np.zeros(0, TD.build())), own(np.zeros(0, TD.build()))])

    # ---- np.searchsorted / np.digitize ---------------------------------------------------------------------------------
    def search(layout, hay, keys, side="left", sorter=None, method=False):
        ops = [hay, keys] + ([sorter] if sorter is not None else [])
        if method:
            call = lambda h, k, *s, sd=side: h.searchsorted(k, side=sd, sorter=s[0] if s else None)
        else:
            call = lambda h, k, *s, sd=side: np.searchsorted(h, k, side=sd, sorter=s[0] if s else None)
        add("tsort", "ndsearchsorted" if method else "searchsorted_kw",
            params={"side": side, "sorter": sorter is not None}, operands=ops, layout=layout, call=call)

    key_pool = T_SEARCH_POOL + ["x" * 20, "y" * 300, "é" * 9]
    for hspec in (TD, TDN, TDNAN, TDS):
        for n in (0, 1, 5, 16, 40):
            for na_rate in ((0.0, 0.3) if hspec.na is not _ABSENT else (0.0,)):
                hay_arr = _ts_sorted(_ts_array(_ts_values(rng, n, T_SEARCH_POOL, hspec, na_rate), hspec))
                hay = own(hay_arr)
                for kspec in (TD, TDN, TDNAN, TDS):
                    keys = own(_ts_array(_ts_values(rng, 8, key_pool, kspec, 0.3), kspec))
                    for side in ("left", "right"):
                        search(f"{_t_label(hspec)}{n}na{na_rate}:{_t_label(kspec)}", hay, keys, side)
                search(f"{_t_label(hspec)}{n}na{na_rate}:T0d", hay, t_plain(np.array("b", dtype=TD.build())))
                search(f"{_t_label(hspec)}{n}na{na_rate}:T2d", hay, t_plain(_ts_array(
                    _ts_values(rng, 6, key_pool, hspec, 0.2), hspec).reshape(2, 3)))
                search(f"{_t_label(hspec)}{n}na{na_rate}:U", hay, own(np.array(["b", "a"], dtype="<U1")))
                search(f"{_t_label(hspec)}{n}na{na_rate}:i", hay, own(np.array([1, 5])))
                search(f"{_t_label(hspec)}{n}na{na_rate}:f", hay, own(np.array([1.5, np.nan])))
                search(f"{_t_label(hspec)}{n}na{na_rate}:S", hay, own(np.array([b"a"], dtype="|S1")))
                search(f"{_t_label(hspec)}{n}na{na_rate}:method", hay, own(np.array(["b", "zz"], dtype=TD.build())),
                       method=True)
                if n:
                    perm = rng.permutation(n)
                    search(f"{_t_label(hspec)}{n}na{na_rate}:sorter", own(hay_arr[perm].copy()), own(
                        np.array(["b", "a"], dtype=TD.build())), "left", own(np.argsort(perm).astype(np.int64)))
                    bad = np.argsort(perm).astype(np.int64)
                    bad[-1] = n + 3
                    search(f"{_t_label(hspec)}{n}na{na_rate}:badsorter", hay, own(np.array(["b"], dtype=TD.build())),
                           "right", own(bad))
    for haystack, key in ((np.array(["a", "b"]), np.array(["b"], dtype=TDNAN.build())),
                          (np.array([1.0, 2.0]), np.array(["a"], dtype=TD.build())),
                          (np.array([b"a", b"b"]), np.array(["a"], dtype=TD.build())),
                          (np.array([], dtype=float), np.array(["a"], dtype=TD.build()))):
        search(f"{haystack.dtype.str}-in-T", own(haystack), own(key))
    for xspec in (TD, TDN, TDNAN):
        for n in (0, 1, 4):
            x = own(_ts_array(_ts_values(rng, n, T_SEARCH_POOL, xspec, 0.3), xspec))
            for bins in (np.array([1.0, 2.0]), np.array([2.0, 1.0]), np.array([1.0, 1.0]), np.zeros(0)):
                for right in (False, True):
                    add("tsort", "digitize", params={"right": right}, operands=[x, num_operand(bins)],
                        layout=f"{_t_label(xspec)}{n}:bins{len(bins)}", call=lambda a, b, r=right: np.digitize(a, b, right=r))
            for bins_arr in (np.array(["a", "b", "x" * 15], dtype=TD.build()), np.array(["x", "b"], dtype=TD.build()),
                             np.array(["a", np.nan], dtype=TDNAN.build())):
                add("tsort", "digitize", params={"right": False}, operands=[x, own(bins_arr)],
                    layout=f"{_t_label(xspec)}{n}:Tbins", call=lambda a, b: np.digitize(a, b))
    add("tsort", "digitize", params={"right": False}, operands=[own(np.array([1.0, 2.5])),
                                                                own(np.array(["1", "2"], dtype=TD.build()))],
        layout="f8-x:Tbins", call=lambda a, b: np.digitize(a, b))

    # ---- np.unique, the Array-API family, np.isin and the set routines -------------------------------------------------
    flags_all = [(False, False, False), (True, False, False), (False, True, False), (False, False, True),
                 (True, True, True), (True, False, True)]
    for spec in T_ORDER_SPECS:
        for n, na_rate in ((0, 0.0), (1, 0.0), (7, 0.3), (40, 0.25), (300, 0.1), (500, 0.05)):
            if spec is TDN and na_rate and n > 7:
                na_rate = 0.0 if n == 300 else na_rate
            arr = _ts_array(_ts_values(rng, n, T_SET_POOL, spec, na_rate), spec)
            op = own(arr)
            lab = f"{_t_label(spec)}:n{n}na{na_rate}"
            for ri, rv, rc in (flags_all if n in (7, 40, 300) else flags_all[:1] + flags_all[4:5]):
                add("tsort", "unique", params={"ri": ri, "rv": rv, "rc": rc, "axis": None}, operands=[op], layout=lab,
                    call=lambda a, i=ri, v=rv, c=rc: np.unique(a, return_index=i, return_inverse=v, return_counts=c))
            for en in (True, False):
                for srt in (True, False):
                    add("tsort", "unique", params={"ri": False, "rv": False, "rc": False, "axis": None, "equal_nan": en,
                                                   "sorted": srt}, operands=[op], layout=lab,
                        call=lambda a, e=en, s=srt: np.unique(a, equal_nan=e, sorted=s))
            for name in ("unique_values", "unique_counts", "unique_inverse", "unique_all"):
                add("tsort", "unique_api", params={"name": name}, operands=[op], layout=lab,
                    call=lambda a, f=getattr(np, name): f(a))
        for lay in ("c2", "f2", "t3", "s2", "r2", "bc", "z0", "e2", "nx"):
            op = t_operand(spec, lay, seed=21, pool=T_SET_POOL, na_every=5 if spec is not TDN else 0)
            v = op[1]
            for ax in [None] + ([0, -1] if v.ndim else []):
                add("tsort", "unique", params={"ri": True, "rv": True, "rc": True, "axis": ax}, operands=[op],
                    layout=f"{_t_label(spec)}:{lay}",
                    call=lambda a, x=ax: np.unique(a, return_index=True, return_inverse=True, return_counts=True, axis=x))

    def isin_case(layout, x, y, *, assume_unique=False, invert=False, kind=None):
        params = {"assume_unique": assume_unique, "invert": invert, "kind": kind}
        add("tsort", "isin", params=params, operands=[x, y], layout=layout,
            call=lambda a, b, au=assume_unique, iv=invert, k=kind: np.isin(a, b, assume_unique=au, invert=iv, kind=k))

    for xspec in T_ORDER_SPECS:
        x = own(_ts_array(_ts_values(rng, 12, T_SET_POOL, xspec, 0.25), xspec))
        for yspec in (TD, TDN, TDNAN, TDS):
            y = own(_ts_array(_ts_values(rng, 5, T_SET_POOL, yspec, 0.3), yspec))
            for iv in (False, True):
                isin_case(f"{_t_label(xspec)}-{_t_label(yspec)}", x, y, invert=iv)
            isin_case(f"{_t_label(xspec)}-{_t_label(yspec)}:sort", x, y, kind="sort")
        for y, ykey in ((own(np.array(["fig", "pear\x00", "zz"], dtype="<U5")), "U"),
                        (own(np.array([b"fig", b"x"], dtype="|S3")), "S"), (own(np.arange(-2, 3)), "i"),
                        (own(np.array([np.nan, 1.5])), "f")):
            isin_case(f"{_t_label(xspec)}-{ykey}", x, y)
            isin_case(f"{ykey}-{_t_label(xspec)}", y, x)
        isin_case(f"{_t_label(xspec)}-table", x, own(np.array(["a"], dtype=TD.build())), kind="table")
        isin_case(f"{_t_label(xspec)}-au", x, x, assume_unique=True)
    for key, x in (("2d", t_plain(np.array(_pick(rng, 12, T_SET_POOL), dtype=TD.build()).reshape(3, 4))),
                   ("0d", t_plain(np.array("fig", dtype=TD.build()))), ("e", own(np.zeros(0, TD.build()))),
                   ("t3", t_operand(TD, "t3", seed=2, pool=T_SET_POOL))):
        isin_case(f"T{key}", x, own(np.array(["fig", "pear", "a"], dtype=TD.build())))
        isin_case(f"T{key}-big", x, own(np.array(_pick(rng, 60, T_SET_POOL), dtype=TD.build())))

    def setop(layout, name, x, y, assume_unique=False, return_indices=False):
        params = {"name": name, "assume_unique": assume_unique, "return_indices": return_indices}
        if name == "intersect1d":
            call = lambda a, b, au=assume_unique, ri=return_indices: np.intersect1d(a, b, assume_unique=au,
                                                                                     return_indices=ri)
        elif name == "union1d":
            call = lambda a, b: np.union1d(a, b)
        else:
            call = lambda a, b, f=getattr(np, name), au=assume_unique: f(a, b, assume_unique=au)
        add("tsort", "setop", params=params, operands=[x, y], layout=layout, call=call)

    set_pairs = []
    for spec in T_ORDER_SPECS:
        for n1, n2 in ((5, 4), (40, 33), (600, 400)):
            na1 = 0.0 if spec is TDN and n1 > 5 else 0.2
            a = own(_ts_array(_ts_values(rng, n1, T_SET_POOL, spec, na1), spec))
            b = own(_ts_array(_ts_values(rng, n2, T_SET_POOL + ["plum"], spec, na1), spec))
            set_pairs.append((f"{_t_label(spec)}{n1}", a, b))
        set_pairs.append((f"{_t_label(spec)}-U", own(_ts_array(_ts_values(rng, 6, T_SET_POOL, spec, 0.2), spec)),
                          own(np.array(["fig", "zz"], dtype="<U3"))))
        set_pairs.append((f"U-{_t_label(spec)}", own(np.array(["fig", "zz", "fig"], dtype="<U3")),
                          own(_ts_array(_ts_values(rng, 6, T_SET_POOL, spec, 0.2), spec))))
    set_pairs += [("NaN-None", own(np.array(["a", np.nan], dtype=TDNAN.build())), own(np.array(["a", None], dtype=TDN.build()))),
                  ("NaN-NA", own(np.array(["a", np.nan], dtype=TDNAN.build())), own(np.array(["a", "NA"], dtype=TDS.build()))),
                  ("T-e", own(np.array(["b", "a", "b"], dtype=TD.build())), own(np.zeros(0, TD.build()))),
                  ("T-2d", t_plain(np.array([["b", "a"], ["c", "a"]], dtype=TD.build())),
                   t_plain(np.array([["a", "z"]], dtype=TD.build()))),
                  ("T-i", own(np.array(["1", "2"], dtype=TD.build())), own(np.array([1, 2])))]
    for key, x, y in set_pairs:
        for name in ("intersect1d", "union1d", "setdiff1d", "setxor1d"):
            setop(key, name, x, y)
            if name != "union1d":
                setop(key, name, x, y, assume_unique=True)
        setop(key, "intersect1d", x, y, return_indices=True)

    # ---- np.sort_complex -----------------------------------------------------------------------------------------------
    # NumPy 2.4.2's StringDType-to-complex cast stores the REAL part in both parts ('3' becomes 3+3j), a bug NumSharp does
    # not reproduce: the values compared are zeros; everything else is an error NumPy raises before or in the cast.
    for spec in (TD, TDN, TDNAN, TDS):
        for vals in (["0", "0", "+0"], ["0"], [], ["1+2j", "j1"], ["x"], ["0", spec.na] if spec.na is not _ABSENT else
                     ["0", "0x1"], ["0", "1e1e1"], [spec.na] if spec.na is not _ABSENT else ["  "]):
            arr = _ts_array(vals, spec)
            add("tsort", "sort_complex", operands=[own(arr)], layout=f"{_t_label(spec)}:{len(vals)}",
                call=lambda a: np.sort_complex(a))
        arr2 = _ts_array(["0", "0", "0", "0"], spec).reshape(2, 2)
        add("tsort", "sort_complex", operands=[t_plain(arr2)], layout=f"{_t_label(spec)}:2d", call=lambda a: np.sort_complex(a))
        add("tsort", "sort_complex", operands=[t_plain(np.array("0", dtype=spec.build()))], layout=f"{_t_label(spec)}:0d",
            call=lambda a: np.sort_complex(a))

    # ---- histograms ----------------------------------------------------------------------------------------------------
    def hist(layout, fn, data, bins, *, weights=None, density=False, rng_=None, second=None):
        ops = [data] + ([second] if second is not None else [])
        params = {"fn": fn, "range": _json_floats(rng_) if rng_ is not None else None, "density": density}
        if isinstance(bins, tuple):
            params["bins_op"] = len(ops)
            ops.append(bins)
            params["bins"] = None
        else:
            params["bins"] = bins
        if weights is not None:
            params["weights_op"] = len(ops)
            ops.append(weights)

        def call(*views, p=params, r=rng_):
            a = views[0]
            b = views[p["bins_op"]] if "bins_op" in p else p["bins"]
            w = views[p["weights_op"]] if "weights_op" in p else None
            rr = tuple(r) if r is not None else None
            if p["fn"] == "histogram_bin_edges":
                return np.histogram_bin_edges(a, b, rr, w)
            if p["fn"] == "histogram":
                return np.histogram(a, b, rr, density=p["density"], weights=w)
            yy = views[1]
            if p["fn"] == "histogram2d":
                return np.histogram2d(a, yy, b, (rr, rr) if rr else None)
            bb = [b, b] if isinstance(b, np.ndarray) and b.ndim == 1 else b
            H, edges = np.histogramdd([a, yy], bb, (rr, rr) if rr else None)
            return (H,) + tuple(edges)

        add("tsort", "hist", params=params, operands=ops, layout=layout, call=call)

    for spec in (TD, TDN, TDNAN, TDS):
        for vals in (["b", "a", "zz", "", "a"], ["b"], [], [spec.na, spec.na] if spec.na is not _ABSENT else ["a", "a"],
                     ["b", spec.na, "c"] if spec.na is not _ABSENT else ["b", "c"]):
            data = own(_ts_array(vals, spec))
            lab = f"{_t_label(spec)}:{len(vals)}"
            for edges in (np.array(["a", "b", "z"], dtype=spec.build()), np.array(["c", "a"], dtype=spec.build()),
                          np.array(["a"], dtype=spec.build())):
                hist(f"{lab}:Tedges{len(edges)}", "histogram", data, own(edges))
                hist(f"{lab}:Tedges{len(edges)}:density", "histogram", data, own(edges), density=True)
                hist(f"{lab}:Tedges{len(edges)}:edges", "histogram_bin_edges", data, own(edges))
            hist(f"{lab}:Uedges", "histogram", data, own(np.array(["a", "c"])))
            hist(f"{lab}:fedges", "histogram", data, num_operand(np.array([0.0, 1.0])))
            for b in (3, "auto"):
                hist(f"{lab}:bins{b}", "histogram", data, b)
                hist(f"{lab}:bins{b}:range", "histogram", data, b, rng_=(0.0, 1.0))
            if vals:
                w = own(np.ones(len(vals)))
                hist(f"{lab}:w", "histogram", data, own(np.array(["a", "zz"], dtype=spec.build())), weights=w)
                hist(f"{lab}:Tw", "histogram", data, own(np.array(["a", "zz"], dtype=spec.build())),
                     weights=own(_ts_array(["x"] * len(vals), spec)))
    hist("U-data:Tedges", "histogram", own(np.array(["b", "a"])), own(np.array(["a", "c"], dtype=TD.build())))
    hist("f-data:Tedges", "histogram", num_operand(np.array([0.5, 1.5])), own(np.array(["a", "c"], dtype=TD.build())))
    hist("f-data:Tw", "histogram", num_operand(np.array([0.5, 1.5])), 2, weights=own(np.array(["x", "y"], dtype=TD.build())))
    for fn in ("histogram2d", "histogramdd"):
        for xs, ys, key in ((np.array(["b", "a"], dtype=TD.build()), np.array(["c", "d"], dtype=TD.build()), "TT"),
                            (np.array(["b", "a"], dtype=TD.build()), np.array([1.0, 2.0]), "Tf"),
                            (np.array([1.0, 2.0]), np.array([0.5, 1.5]), "ff")):
            sec = own(ys) if ys.dtype.kind == "T" else num_operand(ys)
            first = own(xs) if xs.dtype.kind == "T" else num_operand(xs)
            hist(f"{fn}:{key}:2", fn, first, 2, second=sec)
            hist(f"{fn}:{key}:Tedges", fn, first, own(np.array(["a", "c", "e"], dtype=TD.build())), second=sec)
            hist(f"{fn}:{key}:range", fn, first, 2, rng_=(0.0, 1.0), second=sec)

    # ---- the quantile family, np.median and np.nanmedian -----------------------------------------------------------------
    methods = ["inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen",
               "weibull", "linear", "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint", "nearest"]

    def quant(layout, op, *, name="quantile", q=0.5, qkind="f", axis=None, method="lower", keepdims=False, out=None):
        ops = [op]
        params = {"name": name, "qkind": qkind, "method": method, "keepdims": keepdims,
                  "axis": list(axis) if isinstance(axis, tuple) else axis}
        if qkind == "arr":
            params["q_op"] = len(ops)
            ops.append(q)
        else:
            params["q"] = q
        if out is not None:
            params["out_op"] = len(ops)
            ops.append(out)

        def call(*views, p=params):
            a = views[0]
            qq = views[p["q_op"]] if p["qkind"] == "arr" else p["q"]
            o = views[p["out_op"]] if "out_op" in p else None
            ax = tuple(p["axis"]) if isinstance(p["axis"], list) else p["axis"]
            kw = {"axis": ax, "method": p["method"], "keepdims": p["keepdims"]}
            if o is not None:
                kw["out"] = o
            return getattr(np, p["name"])(a, qq, **kw)

        add("tsort", "quant_kw", params=params, operands=ops, layout=layout, mutate=params.get("out_op"), call=call)

    quant_data = {
        "T5": lambda s: _ts_array(["b", "a", "zz", "", "m" * 20], s),
        "NaN-mid": lambda s: _ts_array(["b", s.na, "a"] if s.na is not _ABSENT else ["b", "c", "a"], s),
        "NaN-all": lambda s: _ts_array([s.na, s.na] if s.na is not _ABSENT else ["q", "q"], s),
        "one": lambda s: _ts_array(["x"], s),
        "oneNA": lambda s: _ts_array([s.na] if s.na is not _ABSENT else ["y"], s),
        "e": lambda s: _ts_array([], s),
    }
    for spec in (TD, TDN, TDNAN, TDS):
        for dkey, build in quant_data.items():
            arr = build(spec)
            lab = f"{_t_label(spec)}:{dkey}"
            for name in ("quantile", "nanquantile"):
                for method in methods:
                    quant(lab, own(arr), name=name, q=0.5, method=method)
                    quant(lab, own(arr), name=name, q=1.0, method=method)
                for method in ("lower", "linear", "hazen"):
                    quant(lab, own(arr), name=name, q=[0.0, 0.75], qkind="fl", method=method)
                    quant(lab, own(arr), name=name, q=0.5, method=method, keepdims=True)
                    quant(lab, own(arr), name=name, q=own(np.array([0, 1])), qkind="arr", method=method)
                    quant(lab, own(arr), name=name, q=own(np.array(1)), qkind="arr", method=method)
                if name == "quantile":
                    # A Python int q: 'linear' takes the element (nanquantile's C# overloads have no integer q).
                    quant(lab, own(arr), name=name, q=1, qkind="i", method="linear")
            for name in ("percentile", "nanpercentile"):
                for method in ("lower", "nearest", "linear", "midpoint"):
                    quant(lab, own(arr), name=name, q=50.0, method=method)
                    quant(lab, own(arr), name=name, q=[0.0, 100.0], qkind="fl", method=method)
        for lay in ("c2", "f2", "t3", "bc", "e2", "nx"):
            op = t_operand(spec, lay, seed=31, pool=T_SORT_SHORT, na_every=4 if spec is not TDN else 0)
            v = op[1]
            axes = [None, 0, -1] + ([(0, 1)] if v.ndim >= 2 else [])
            for ax in axes:
                for name in ("quantile", "nanquantile"):
                    for method in ("lower", "linear", "nearest"):
                        quant(f"{_t_label(spec)}:{lay}", op, name=name, q=0.5, axis=ax, method=method)
                        quant(f"{_t_label(spec)}:{lay}", op, name=name, q=[0.25, 1.0], qkind="fl", axis=ax, method=method)
                    quant(f"{_t_label(spec)}:{lay}", op, name=name, q=0.5, axis=ax, method="lower", keepdims=True)
    # out=: the take path's checks (the out's class and descriptor), the interpolation's add / subtract into the out, the
    # nan forms' setitem.
    for dspec, dvals in ((TD, ["b", "a"]), (TDNAN, [np.nan, np.nan]), (TDNAN, ["b", np.nan, "a"]), (TDN, ["b", None])):
        arr = _ts_array(dvals, dspec)
        for odt in ("T", "TNAN", "TN", "U3", "U1", "S3", "f8", "f4", "i8", "?", "c16"):
            o = target(np.array("k", dtype={"T": TD, "TNAN": TDNAN, "TN": TDN}[odt].build())) if odt.startswith("T") else \
                target(np.zeros((), dtype=odt))
            for name, method in (("quantile", "lower"), ("quantile", "linear"), ("nanquantile", "lower"),
                                 ("nanquantile", "linear")):
                quant(f"{_t_label(dspec)}{len(dvals)}:out:{odt}", own(arr), name=name, q=0.5, method=method, out=o)
        bad = target(np.zeros(2, dtype="f8"))
        quant(f"{_t_label(dspec)}:out:wrongshape", own(arr), name="quantile", q=0.5, method="lower", out=bad)
    # A StringDType q: _quantile_is_valid's comparisons and minimum, then the virtual index over the text.
    for qvals, qkey in ((["0.5"], "str1"), ([np.nan, "a"], "nanstr"), (["a", np.nan], "strnan"), ([], "empty"),
                        (["0.5"] * 12, "str12")):
        qspec = TDNAN if any(isinstance(v, float) for v in qvals) else TD
        q_op = own(_ts_array(qvals, qspec))
        for name in ("quantile", "nanquantile", "percentile", "nanpercentile"):
            for method in (["linear", "lower", "higher", "nearest", "inverted_cdf", "hazen", "weibull",
                            "interpolated_inverted_cdf", "bogus"] if qkey == "empty" else ["linear"]):
                for data in (num_operand(np.array([1.0, 2.0, 3.0])), own(np.array(["b", "a"], dtype=TD.build()))):
                    quant(f"Tq:{qkey}", data, name=name, q=q_op, qkind="arr", method=method)
    quant("Tq:0d", num_operand(np.array([1.0, 2.0])), q=t_plain(np.array("0.5", dtype=TD.build())), qkind="arr")
    quant("Tq:2d", num_operand(np.array([1.0, 2.0])), q=t_plain(np.array([["0.5"]], dtype=TD.build())), qkind="arr")

    def stat(layout, name, op, axis=None, keepdims=False, out_index=None, extra=()):
        _t_stat_case(name, [op] + list(extra), axis=axis, keepdims=keepdims, out_index=out_index, layout=layout,
                     section="tsort")

    for spec in (TD, TDN, TDNAN, TDS):
        for dkey, build in quant_data.items():
            arr = build(spec)
            for name in ("median", "nanmedian"):
                stat(f"{_t_label(spec)}:{dkey}", name, own(arr))
                stat(f"{_t_label(spec)}:{dkey}", name, own(arr), keepdims=True)
        for lay in ("c2", "f2", "t3", "bc", "e2", "e3", "z0"):
            op = t_operand(spec, lay, seed=33, pool=T_SORT_SHORT, na_every=4 if spec is not TDN else 0)
            for name in ("median", "nanmedian"):
                for ax in [None, 0, -1] if op[1].ndim else [None]:
                    stat(f"{_t_label(spec)}:{lay}", name, op, axis=ax)
    for odt in ("T", "U3", "f8"):
        o = target(np.array("k", dtype=TD.build())) if odt == "T" else target(np.zeros((), dtype=odt))
        for dvals, dspec in ((["b"], TD), ([np.nan, np.nan], TDNAN), (["b", None], TDN)):
            for name in ("median", "nanmedian"):
                stat(f"{_t_label(dspec)}:out:{odt}", name, own(_ts_array(dvals, dspec)), out_index=1, extra=[o])
    for n in (599, 600):
        wide = np.array([[np.nan] * n, ["b"] * n], dtype=TDNAN.build())
        stat(f"T-NaN:{n}cols", "nanmedian", t_plain(wide), axis=1)
        stat(f"T-NaN:{n}cols:nan", "nanmedian", t_plain(wide[:1]), axis=1)

    # ---- numpy.ma over StringDType -------------------------------------------------------------------------------------
    unary = _strma_unary_table()
    T, TN, TNAN, TS = (s.build() for s in (TD, TDN, TDNAN, TDS))
    margs = [("nomask", MArg(["b", "a", "c"], spec=T)), ("mask", MArg(["b", "a", "c"], [0, 1, 0], spec=T)),
             ("allfalse", MArg(["b", "a", "c"], [0, 0, 0], spec=T)), ("2d", MArg([["b", "a"], ["d", "c"]], spec=T)),
             ("nan", MArg(["b", float("nan"), "a", float("nan")], spec=TNAN)), ("none", MArg(["b", None], spec=TN)),
             ("noneone", MArg([None], spec=TN)), ("sna", MArg(["b", "NA", "a"], spec=TS)),
             ("e", MArg(np.zeros(0, T))), ("e2", MArg(np.zeros((0, 2), T))), ("one", MArg(["x"], spec=T))]
    for name in ("sort", "argsort", "median", "unique", "sort_end_false"):
        for key, arg in margs:
            add_ma("ma1", params={"name": name}, args=[arg], layout=key, call=lambda x, f=unary[name]: f(x),
                   section="tsort", phase=1)


# Section strcharobj: numpy.char's OBJECT path (the numpy.char plan's phase 4) — np.char.split / rsplit / splitlines /
# join and the chararray members over them. _vec_string per element: the operands' NumPy scalars, CPython's str / bytes
# methods in CPython's order (the descriptor check, maxsplit's __index__, keepends' truth, the separator's type, the empty
# separator, the pieces), the first failing element (C order) ending the call; the object results recorded piece by
# piece with every np.str_ piece (unicodeobject.c's [self] shortcut) marked; join's _to_bytes_or_str_array results and
# NumPy 2.4.2's ObjectDType(n) failure over an object seq.
# ----------------------------------------------------------------------------------------------------------------------

COBJ = "strcharobj"

# Texts for the split family: whitespace runs at either end, every class of whitespace (Python's 29 code points for str,
# the 6 ASCII bytes for bytes: \x1c-\x1f, \x85 and \xa0 are whitespace to str only), embedded and trailing NULs (a U
# element's trailing NULs are padding, a StringDType element keeps them), separators that repeat and overlap, the
# [self] shapes (a separator longer than the text or of a wider kind), and supplementary-plane text.
COBJ_SPLIT_U = ["a b  c", "  lead trail  ", "", "x\ty\nz", "one", "w\x00 x\x00", "a\x1cb\x85c\xa0d", "\xe9 \xfc\u3000\xe7",
                "\U0001F600 a\U0001F600b", "\t\n\x0b\x0c\r ", "a,b,,c,", "aaa", "ab", "abc", "  ", "x"]
COBJ_SPLIT_S = [b"a b  c", b"  lead trail  ", b"", b"x\ty\nz", b"one", b"w\x00 x\x00", b"a\x1cb\x85c\xa0d", b"\xe9 \xfc",
                b"a,b,,c,", b"aaa", b"ab", b"abc", b"  ", b"x", b"\t\n\x0b\x0c\r ", b"\xff\x80 \x7f"]
COBJ_LINES_U = ["a\nb\r\nc\rd", "x\n", "", "\n", "a\x0bb\x0cc\x1cd\x1de\x1ef\x85g\u2028h\u2029i", "no newline",
                "\r\n\r\n", "\r", "a\r\n", "\n\n", "x", "\r\r\n\n"]
COBJ_LINES_S = [b"a\nb\r\nc\rd", b"x\n", b"", b"\n", b"a\x0bb\x0cc\x1cd\x85e", b"no newline", b"\r\n\r\n", b"\r", b"a\r\n",
                b"\n\n", b"x", b"\r\r\n\n"]
COBJ_JOIN_U = ["osd", "ab", "", "c", "x y", "\xe9\U0001F600", "a\x00b", "z"]
# bytes.join iterates a bytes object as ints, so only empty S sequences join; the second pool raises at its first
# non-empty element.
COBJ_JOIN_S = [b"", b"", b""]
COBJ_JOIN_S2 = [b"", b"ab", b""]


def _cobj_u(s):
    """A str (or np.str_) as JSON code points (UTF-32LE hex). A NumPy str may hold a code point above U+10FFFF: the codec
    copies it, while iterating the str would create it as a one-character str, which CPython refuses (SystemError)."""
    return {"u": s.encode("utf-32-le", "surrogatepass").hex()}


def _cobj_piece(p):
    """One piece of a list: text as code points, bytes as hex."""
    if isinstance(p, (bytes, np.bytes_)):
        return {"b": bytes(p).hex()}
    return _cobj_u(p)


def _cobj_expect(r, views):
    """The expectation of an object-path result: an object array of lists as ``lists`` (its shape, every element's pieces in
    C order, and the C-order positions whose list holds an np.str_ piece — CPython's [self], which NumPy prints as
    ``np.str_('...')``), any other result as ``expect_of`` records it.

    Args:
        r: NumPy's result.
        views: the operand views.

    Returns:
        The expectation.

    Raises:
        AssertionError: an object array whose element is not a list (the object path makes lists only).
    """
    if isinstance(r, np.ndarray) and r.dtype == object:
        lists, selfs = [], []
        for pos, el in enumerate(r.flat):
            assert isinstance(el, list), f"an object-path element that is not a list: {el!r}"
            lists.append([_cobj_piece(p) for p in el])
            if any(type(p) is np.str_ for p in el):
                selfs.append(pos)
        return {"kind": "lists", "shape": [int(x) for x in r.shape], "lists": lists, "strself": selfs}
    return expect_of(r, views)


def _cobj_value(spec, views):
    """One argument of a case as the Python value NumPy receives.

    Args:
        spec: ``{"op": k}`` (operand k), ``{"cha": k}`` (operand k's chararray view), ``{"v": enc}`` (a Python value),
            ``{"split": [spec, ...]}`` (``np.char.split`` of those arguments: an object array of lists), ``{"view":
            [spec, index]}`` (a basic index of an argument's object array, the index in NumSharp's slice notation).
        views: the operand views.

    Returns:
        The value.

    Raises:
        TypeError: an unknown spec.
    """
    if "op" in spec:
        return views[spec["op"]]
    if "cha" in spec:
        return views[spec["cha"]].view(np.char.chararray)
    if "v" in spec:
        return _cobj_dec(spec["v"])
    if "split" in spec:
        return np.char.split(*[_cobj_value(s, views) for s in spec["split"]])
    if "view" in spec:
        target, index = spec["view"]
        return _cobj_value(target, views)[_cobj_index(index)]
    raise TypeError(f"unknown argument spec {spec!r}")


def _cobj_dec(v):
    """``_chx_dec`` with complex numbers (``{"complex": [re, im]}``) and Python ints past int64."""
    if isinstance(v, dict) and "complex" in v:
        re, im = v["complex"]
        return complex(float(re) if isinstance(re, str) else re, float(im) if isinstance(im, str) else im)
    if isinstance(v, list):
        return [_cobj_dec(x) for x in v]
    return _chx_dec(v)


def _cobj_enc(v):
    """``enc`` with complex numbers (the generator's own extension; NumSharp's replay decodes them)."""
    if isinstance(v, (complex, np.complexfloating)):
        return {"complex": [_jf(v.real), _jf(v.imag)]}
    if isinstance(v, (list, tuple)):
        return [_cobj_enc(x) for x in v]
    return enc(v)


def _cobj_index(text):
    """A basic index in NumSharp's slice notation (``"1:, ::-1"``, ``"0"``) as the Python index."""
    parts = []
    for part in text.split(","):
        part = part.strip()
        if ":" in part:
            parts.append(slice(*[int(x) if x.strip() else None for x in part.split(":")]))
        else:
            parts.append(int(part))
    return tuple(parts)


def _cobj_call(params, views):
    """Makes one case's call: ``np.char.<fn>`` (``params.fn``) or a chararray member (``params.member`` of operand
    ``params.recv``'s chararray view), with ``args`` / ``kw``; ``print`` adds repr (true) or str (false) of the result;
    ``index`` a basic index of the result."""
    args = [_cobj_value(a, views) for a in params.get("args", [])]
    kw = {name: _cobj_value(a, views) for name, a in params.get("kw", {}).items()}
    if "member" in params:
        r = getattr(views[params["recv"]].view(np.char.chararray), params["member"])(*args, **kw)
    else:
        r = getattr(np.char, params["fn"])(*args, **kw)
    if "index" in params:
        r = r[_cobj_index(params["index"])]
    if "print" in params:
        return repr(r) if params["print"] else str(r)
    return r


def add_cobj(op, params, operands=(), layout=""):
    """Records one object-path case (``_cobj_call``), its object results as ``lists``.

    Args:
        op: ``cobj_fn`` (a module function), ``cobj_member`` (a chararray member) or ``cobj_print`` (repr / str of a
            result).
        params: the case parameters (argument specs: ``_cobj_value``).
        operands: (base, view) pairs.
        layout: the case's layout label.
    """
    views = [v for _, v in operands]
    prefix = f"{COBJ}/{op}/"
    n = _OP_COUNT.get(prefix, 0) + 1
    _OP_COUNT[prefix] = n
    cid = f"{COBJ}/{op}/{n}"
    assert cid not in _IDS
    _IDS.add(cid)
    case = {"id": cid, "op": op, "params": params, "operands": [describe(b, v) for b, v in operands], "phase": 1,
            "layout": layout}
    try:
        case["expected"] = _cobj_expect(_cobj_call(params, views), views)
    except Exception as e:  # noqa: BLE001 -- every NumPy exception is a recorded outcome
        case["error"] = err_info(e)
    CASES.append(case)


def _cobj_operand(kind, pool, layout, seed=0, na=_ABSENT, na_every=0):
    """A text operand of class ``kind`` (``U`` / ``S`` / ``T``) whose base cycles through ``pool`` from ``seed``, viewed
    through ``LAYOUTS[layout]``; a StringDType operand with an NA object holds it every ``na_every``-th element."""
    n, build = LAYOUTS[layout]
    vals = [pool[(seed + i) % len(pool)] for i in range(n)]
    if kind == "T":
        dt = np.dtypes.StringDType() if na is _ABSENT else np.dtypes.StringDType(na_object=na)
        if na is not _ABSENT and na_every:
            vals = [na if (seed + i) % na_every == na_every - 1 else v for i, v in enumerate(vals)]
        base = np.array(vals, dtype=dt)
    elif kind == "U":
        base = np.array(vals, dtype=f"<U{max(1, max((len(v) for v in vals), default=1))}")
    else:
        base = np.array(vals, dtype=f"|S{max(1, max((len(v) for v in vals), default=1))}")
    return base, build(base)


def _cobj_like(kind, value):
    """A str value as the text of class ``kind``: bytes (Latin-1) for ``S``, the str itself otherwise."""
    return value.encode("latin-1") if kind == "S" and isinstance(value, str) else value


def _cobj_v(value):
    """The argument spec of a Python value."""
    return {"v": _cobj_enc(value)}


def gen_strcharobj():
    """Section strcharobj: np.char.split / rsplit / splitlines / join and the chararray members — every text class
    (S / U / StringDType, also with NA objects) over the core layouts with every separator and limit form, the argument
    kinds of every dtype (numbers, NumPy scalars, text of the other class, None, lists holding None, ints past int64), the
    errors in NumPy's order (the non-string first argument, the descriptor check, __index__, the separator's type, the
    empty separator, broadcasting, ragged lists, the one-unit str above U+10FFFF), the whitespace and line-boundary sets
    exhaustively, the result's views and printing, and join over the object result."""
    pools = {"U": COBJ_SPLIT_U, "S": COBJ_SPLIT_S, "T": COBJ_SPLIT_U}
    lines = {"U": COBJ_LINES_U, "S": COBJ_LINES_S, "T": COBJ_LINES_U}
    seps = [None, " ", ",", "aa", "  ", "x", "\U0001F600", "Ā"]
    limits = [_ABSENT, 0, 1, 2, -1]
    # 1. split / rsplit: class x layout x separator x maxsplit.
    for fn in ("split", "rsplit"):
        for kind in ("U", "S", "T"):
            for li, layout in enumerate(CORE_LAYOUTS):
                ops = [_cobj_operand(kind, pools[kind], layout, seed=li)]
                for sep in seps:
                    if kind == "S" and sep is not None and not sep.isascii():
                        continue
                    for lim in limits:
                        args = [{"op": 0}]
                        kw = {}
                        if sep is not None:
                            kw["sep"] = _cobj_v(_cobj_like(kind, sep))
                        if lim is not _ABSENT:
                            kw["maxsplit"] = _cobj_v(lim)
                        add_cobj("cobj_fn", {"fn": fn, "args": args, "kw": kw}, ops, layout=f"{kind}:{layout}")
    # 2. splitlines: class x layout x keepends (Python's truth rule).
    keeps = [_ABSENT, True, False, 0, 1, 2, 0.5, "", "x", None, float("nan")]
    for kind in ("U", "S", "T"):
        for li, layout in enumerate(CORE_LAYOUTS):
            ops = [_cobj_operand(kind, lines[kind], layout, seed=li)]
            for keep in keeps:
                kw = {} if keep is _ABSENT else {"keepends": _cobj_v(keep)}
                add_cobj("cobj_fn", {"fn": "splitlines", "args": [{"op": 0}], "kw": kw}, ops, layout=f"{kind}:{layout}")
    # 3. join: separators x sequences of every class and layout.
    join_seps = {"U": ["-", "", "--", "\U0001F600", " "], "S": [b"-", b""], "T": ["-", "", "~~"]}
    for skind in ("U", "S", "T"):
        for qkind in ("U", "S", "T"):
            for li, layout in enumerate(CORE_LAYOUTS):
                pool = (COBJ_JOIN_S if li % 2 == 0 else COBJ_JOIN_S2) if qkind == "S" else COBJ_JOIN_U
                ops = [_cobj_operand(qkind, pool, layout, seed=li)]
                for sep in join_seps[skind]:
                    sep_arg = _cobj_v(sep) if skind != "T" else None
                    if skind == "T":
                        sb, sv = plain(np.array([sep], dtype=np.dtypes.StringDType()))
                        ops2 = ops + [(sb, sv)]
                        add_cobj("cobj_fn", {"fn": "join", "args": [{"op": 1}, {"op": 0}]}, ops2,
                                 layout=f"{skind}{qkind}:{layout}")
                    else:
                        add_cobj("cobj_fn", {"fn": "join", "args": [sep_arg, {"op": 0}]}, ops,
                                 layout=f"{skind}{qkind}:{layout}")
    # A separator array of every layout against a sequence array (broadcast both ways).
    for li, layout in enumerate(CORE_LAYOUTS):
        sb = _cobj_operand("U", ["-", "", "+", "~~", "\xe9"], layout, seed=li)
        qb = _cobj_operand("U", COBJ_JOIN_U, "c1", seed=li)
        add_cobj("cobj_fn", {"fn": "join", "args": [{"op": 0}, {"op": 0}]}, [sb], layout=f"UU:{layout}:self")
        add_cobj("cobj_fn", {"fn": "join", "args": [{"op": 0}, _cobj_v("abc")]}, [sb], layout=f"UU:{layout}:str")
    # 4. Argument kinds: maxsplit / keepends / separators of every dtype, NumPy scalars, text of the other class, None,
    # object lists, ints past int64, and the separator / maxsplit as broadcast arrays.
    u2 = plain(np.array(["a b c d", "e f g"]))
    s2 = plain(np.array([b"a b c d", b"e\x05f g"]))
    for dt in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64",
               "complex128", "bool"):
        num = plain(np.array([1, 0], dtype=dt))
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 0}, _cobj_v(" "), {"op": 1}]}, [u2, num], layout=f"max:{dt}")
        add_cobj("cobj_fn", {"fn": "rsplit", "args": [{"op": 0}, _cobj_v(None), {"op": 1}]}, [u2, num], layout=f"max:{dt}")
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 0}, {"op": 1}]}, [u2, num], layout=f"sep:{dt}")
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 0}, {"op": 1}]}, [s2, num], layout=f"bsep:{dt}")
        lines2 = plain(np.array(["a\nb", "c\r\nd"]))
        add_cobj("cobj_fn", {"fn": "splitlines", "args": [{"op": 0}, {"op": 1}]}, [lines2, num], layout=f"keep:{dt}")
    for value in [1.5, True, False, 5, -3, 2 ** 62, 2 ** 63, 2 ** 64, -(2 ** 63), -(2 ** 63) - 1, 1 + 0j, "1", b"1",
                  None, [1, None], [None, None], [" ", None], [None]]:
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 0}, _cobj_v(" "), _cobj_v(value)]}, [u2], layout="max:py")
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 0}, _cobj_v(value)]}, [u2], layout="sep:py")
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 0}, _cobj_v(value)]}, [s2], layout="bsep:py")
        add_cobj("cobj_fn", {"fn": "splitlines", "args": [{"op": 0}, _cobj_v(value)]}, [u2], layout="keep:py")
        add_cobj("cobj_fn", {"fn": "join", "args": [_cobj_v("-"), _cobj_v(value)]}, [], layout="seq:py")
        add_cobj("cobj_fn", {"fn": "join", "args": [_cobj_v(value), _cobj_v("ab")]}, [], layout="jsep:py")
    for tspec in (_ABSENT, None, float("nan"), "NA", ""):
        tb = _cobj_operand("T", ["a b", "c,d", "", "x\ny", "\xe9 \xfc"], "c1", na=tspec, na_every=3)
        lab = "T" if tspec is _ABSENT else f"T(na={tspec!r})"
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 0}]}, [tb], layout=lab)
        add_cobj("cobj_fn", {"fn": "rsplit", "args": [{"op": 0}, _cobj_v(","), _cobj_v(1)]}, [tb], layout=lab)
        add_cobj("cobj_fn", {"fn": "splitlines", "args": [{"op": 0}, _cobj_v(True)]}, [tb], layout=lab)
        add_cobj("cobj_fn", {"fn": "join", "args": [_cobj_v("-"), {"op": 0}]}, [tb], layout=lab)
        add_cobj("cobj_fn", {"fn": "join", "args": [{"op": 0}, _cobj_v("xy")]}, [tb], layout=lab)
        # The StringDType array as the separator / maxsplit / keepends of a U text (its NA elements: None is
        # whitespace, other NA objects their own type's errors).
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 1}, {"op": 0}]},
                 [tb, plain(np.array(["a b", "c,d", "e f", "g h", "i j", "k l"]))], layout=lab + ":sep")
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 1}, _cobj_v(" "), {"op": 0}]},
                 [tb, plain(np.array(["a b c"] * 6))], layout=lab + ":max")
        add_cobj("cobj_fn", {"fn": "splitlines", "args": [{"op": 1}, {"op": 0}]},
                 [tb, plain(np.array(["a\nb"] * 6))], layout=lab + ":keep")
    # Broadcasting of the arguments, and its error (NumPy's multi-iterator text).
    m = plain(np.array([["a b", "c,d", "e f g"], ["h", "i j", ",k"]]))
    for spec in ([" ", ","], [[" "], [","]], [" ", ",", "x"], [[" ", ","]], [" "]):
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 0}, _cobj_v(spec)]}, [m], layout="bcast:sep")
    for spec in ([0, 1, -1], [[0], [1]], [0, 1], [[1, 0, 2]]):
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 0}, _cobj_v(None), _cobj_v(spec)]}, [m], layout="bcast:max")
        add_cobj("cobj_fn", {"fn": "rsplit", "args": [{"op": 0}, _cobj_v(" "), _cobj_v(spec)]}, [m], layout="bcast:max")
        add_cobj("cobj_fn", {"fn": "splitlines", "args": [{"op": 0}, _cobj_v(spec)]}, [m], layout="bcast:keep")
    for sspec in (["-", "+"], [["-"], ["+"]], ["-", "+", "*"]):
        add_cobj("cobj_fn", {"fn": "join", "args": [_cobj_v(sspec), {"op": 0}]}, [m], layout="bcast:join")
    # 5. Errors in NumPy's order.
    for a in (1.5, 5, True, None, [1, 2], [None, "a"], [["a"], ["b", "c"]], ["a b", None]):
        add_cobj("cobj_fn", {"fn": "split", "args": [_cobj_v(a)]}, [], layout="a:py")
        add_cobj("cobj_fn", {"fn": "splitlines", "args": [_cobj_v(a)]}, [], layout="a:py")
        add_cobj("cobj_fn", {"fn": "join", "args": [_cobj_v(a), _cobj_v("ab")]}, [], layout="a:py")
    for args in ([["a", "b"], "", [0, 1, 2]], [["a", "b"], ["", "x"]], [["a", "b"], ["x", ""]], [["a", "b"], " ", [1.5, 1]],
                 [["a", "b"], ["", "x"], [1.5, 1]], [["a", "b"], [b"x", b""], 1], [["a", "b"], 5], [[b"a b"], 5],
                 [[b"a b"], 1.5], [[b"a b"], ""], [[b"a b"], b""], [[b"a b"], None, 1.5], [["a"], True], [[b"a\x01b"], True],
                 [[b"ab"], 2 ** 63], [[b"ab"], 2 ** 64], [["ab"], 2 ** 64]):
        add_cobj("cobj_fn", {"fn": "split", "args": [_cobj_v(x) for x in args]}, [], layout="order")
        add_cobj("cobj_fn", {"fn": "rsplit", "args": [_cobj_v(x) for x in args]}, [], layout="order")
    for args in ([["-", "+"], [1, 2]], [b"-", [b"ab"]], [b"-", [b""]], ["-", [b"ab"]], [b"-", ["ab"]], [b"-", [b"", b"x"]],
                 [5, "ab"], ["-", 5], ["-", [1, 2]], ["-", [True]], ["-", [1.5]], [["-", "+", "*"], ["ab", "cd"]],
                 ["-", [["a"], ["b", "c"]]], [["-", None], "ab"], ["-", ["ab", None]], [b"-", [""]], ["-", [b""]]):
        add_cobj("cobj_fn", {"fn": "join", "args": [_cobj_v(x) for x in args]}, [], layout="order")
    # A one-unit str above U+10FFFF (CPython's PyUnicode_New refusal) as the text, a separator, a piece, a line, a join
    # sequence (iterated character by character) or a separator (only a ONE-unit text is refused).
    def bad(units):
        return plain(np.array(units, dtype=np.uint32).view(f"U{len(units)}"))
    for units in ([0x61, 0x20, 0x110000, 0x62], [0x110000, 0, 0, 0], [0x61, 0x20, 0x110000, 0], [0x110000, 0x20, 0x110000, 0x62],
                  [0x61, 0x0A, 0x110000, 0], [0x78, 0x110000]):
        b = bad(units)
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 0}]}, [b], layout="u110000")
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 0}, _cobj_v(" ")]}, [b], layout="u110000")
        add_cobj("cobj_fn", {"fn": "rsplit", "args": [{"op": 0}]}, [b], layout="u110000")
        add_cobj("cobj_fn", {"fn": "splitlines", "args": [{"op": 0}]}, [b], layout="u110000")
        add_cobj("cobj_fn", {"fn": "join", "args": [_cobj_v("-"), {"op": 0}]}, [b], layout="u110000")
        add_cobj("cobj_fn", {"fn": "join", "args": [{"op": 0}, _cobj_v("ab")]}, [b], layout="u110000")
        add_cobj("cobj_fn", {"fn": "split", "args": [_cobj_v(["ab"]), {"op": 0}]}, [b], layout="u110000:sep")
    # 6. The whitespace and line-boundary sets, exhaustively: every code point below U+3010 (str) / byte (bytes) between
    # two letters, split and split into lines.
    every_u = "".join("a" + chr(c) for c in range(1, 0x3010)) + "a"
    every_s = b"".join(b"a" + bytes([c]) for c in range(1, 256)) + b"a"
    for text in (every_u, every_s):
        add_cobj("cobj_fn", {"fn": "split", "args": [_cobj_v([text])]}, [], layout="every")
        add_cobj("cobj_fn", {"fn": "rsplit", "args": [_cobj_v([text]), _cobj_v(None), _cobj_v(7)]}, [], layout="every")
        add_cobj("cobj_fn", {"fn": "splitlines", "args": [_cobj_v([text])]}, [], layout="every")
        add_cobj("cobj_fn", {"fn": "splitlines", "args": [_cobj_v([text]), _cobj_v(True)]}, [], layout="every")
    tb = plain(np.array([every_u], dtype=np.dtypes.StringDType()))
    add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 0}]}, [tb], layout="every:T")
    add_cobj("cobj_fn", {"fn": "splitlines", "args": [{"op": 0}, _cobj_v(True)]}, [tb], layout="every:T")
    # 7. Long texts: many pieces, long separators and limits from either end (the searches' vector paths).
    big = " ".join(f"w{i}" for i in range(400))
    for kind in ("U", "S", "T"):
        bb = _cobj_operand(kind, [_cobj_like(kind, big), _cobj_like(kind, big.replace(" ", ",, ")), _cobj_like(kind, "x" * 300)],
                           "c1")
        for sep, lim in ((None, _ABSENT), (" ", _ABSENT), (",,", _ABSENT), ("w29", _ABSENT), (" ", 5), (None, 5),
                         ("x", 3), ("xx", _ABSENT), ("w1 w2", _ABSENT)):
            for fn in ("split", "rsplit"):
                kw = {} if sep is None else {"sep": _cobj_v(_cobj_like(kind, sep))}
                if lim is not _ABSENT:
                    kw["maxsplit"] = _cobj_v(lim)
                add_cobj("cobj_fn", {"fn": fn, "args": [{"op": 0}], "kw": kw}, [bb], layout=f"long:{kind}")
        add_cobj("cobj_fn", {"fn": "join", "args": [_cobj_v(_cobj_like(kind, "--") if kind != "S" else "--"), {"op": 0}]},
                 [bb], layout=f"long:{kind}")
    # 8. The chararray members (the elements unstripped: _vec_string reads the NumPy scalars).
    for kind in ("U", "S"):
        cb = _cobj_operand(kind, [_cobj_like(kind, v) for v in ["a b ", "c  d", " e\n", "f\r\n", ""]], "c1")
        for member, args in (("split", []), ("split", [_cobj_v(_cobj_like(kind, " "))]), ("rsplit", [_cobj_v(None), _cobj_v(1)]),
                             ("splitlines", []), ("splitlines", [_cobj_v(True)]), ("join", [_cobj_v(["xy", "z", "", "ab", "c"])]),
                             ("join", [_cobj_v("xy")]), ("join", [_cobj_v(_cobj_like(kind, "xy"))])):
            add_cobj("cobj_member", {"member": member, "recv": 0, "args": args}, [cb], layout=f"cha:{kind}")
        add_cobj("cobj_fn", {"fn": "split", "args": [{"cha": 0}]}, [cb], layout=f"cha:{kind}")
        add_cobj("cobj_fn", {"fn": "join", "args": [_cobj_v("-"), {"cha": 0}]}, [cb], layout=f"cha:{kind}")
        add_cobj("cobj_fn", {"fn": "join", "args": [{"cha": 0}, _cobj_v("ab")]}, [cb], layout=f"cha:{kind}")
    # 9. The object result: its views, printing, and join over it (NumPy 2.4.2's ObjectDType(n) failure; an empty
    # broadcast is an empty object array).
    mb = plain(np.array([["a b", "c"], ["d e f", ""], ["g", "h i"]]))
    for index in ("1:", "::-1", "1, ::-1", "::2, 1", "0", "-1, -1", "1:1", "2:, 1:"):
        add_cobj("cobj_fn", {"fn": "split", "args": [{"op": 0}], "index": index}, [mb], layout="view")
    sb = plain(np.array([b"a b", b"", b"c\x00 d"]))
    for target, ops in (([{"op": 0}], [mb]), ([{"op": 0}], [sb]), ([_cobj_v("a b")], []), ([_cobj_v([])], []),
                        ([{"op": 0}, _cobj_v(",")], [mb]), ([_cobj_v(["ab"]), _cobj_v("abc")], []),
                        ([_cobj_v(["ab", "abc"]), _cobj_v("Ā")], []),
                        ([_cobj_v(["a b c d e f g h i j k l m n o p q r s t u v w x y z"] * 3)], []),
                        ([_cobj_v(["x"] * 1001)], []), ([_cobj_v([["a", ""], ["", "b"]])], []),
                        ([_cobj_v(["it's", 'q"z', "\\", "\t\x00"])], [])):
        for repr_ in (True, False):
            add_cobj("cobj_print", {"fn": "split", "args": target, "print": repr_}, ops, layout="print")
        add_cobj("cobj_print", {"fn": "splitlines", "args": target[:1], "print": True}, ops, layout="print")
    for sep_args, ops in (([_cobj_v(" "), {"split": [{"op": 0}]}], [mb]), ([_cobj_v(["-", "+", "*"]), {"split": [{"op": 0}]}], [mb]),
                          ([_cobj_v(b" "), {"split": [{"op": 0}]}], [sb]), ([_cobj_v(" "), {"split": [{"op": 0}]}], [sb]),
                          ([_cobj_v(b"-"), {"split": [{"op": 0}]}], [mb]), ([_cobj_v(5), {"split": [{"op": 0}]}], [mb]),
                          ([_cobj_v("-"), {"split": [_cobj_v([])]}], []),
                          ([_cobj_v(["-", "+"]), {"split": [{"op": 0}]}], [plain(np.zeros((0, 2), dtype="U2"))]),
                          ([_cobj_v(["-", "+", "*"]), {"split": [_cobj_v(["a b", ""])]}], []),
                          ([_cobj_v("-"), {"view": [{"split": [{"op": 0}]}, "1:1"]}], [mb])):
        add_cobj("cobj_fn", {"fn": "join", "args": sep_args}, ops, layout="join:lists")


# ----------------------------------------------------------------------------------------------------------------------
# Section strcharlist: the ndarray members of numpy.char's object result (C-D3's NDListArray) that move the lists —
# transposition, reshaping, copies and views, selection, and the attributes of the outer array
# ----------------------------------------------------------------------------------------------------------------------
#
# The receiver is np.char.split's object array (or a basic index of it); a member's object result is compared piece by
# piece (_cobj_expect) together with its layout: the byte strides, C / F contiguity, OWNDATA, WRITEABLE and how its base
# relates to the receiver (NumPy's object array holds 8-byte pointers, NumSharp's list array int64 ranges of the same
# layout, so all of it must agree).

CLST = "strcharlist"


def _clst_base(r, recv):
    """How a result's base relates to the receiver.

    Args:
        r: the result (an object array).
        recv: the receiver.

    Returns:
        ``recv`` (the receiver itself), ``recvbase`` (the receiver's own base: a view of a view collapses to the owner),
        ``none`` (the result owns its pointers) or ``other`` (NumPy's intermediate copy, as for a reshape in the other order).
    """
    if r.base is None:
        return "none"
    if r.base is recv:
        return "recv"
    if recv.base is not None and r.base is recv.base:
        return "recvbase"
    return "other"


def _clst_expect(r, recv):
    """The expectation of a list-array member's result.

    Args:
        r: NumPy's result.
        recv: the receiver (an object array of lists).

    Returns:
        An object array as ``lists`` (``_cobj_expect``) with its ``layout`` facets; a Python list — NumPy's one-index take,
        which returns the element itself — as the ``lists`` of a 0-d array (NumSharp's 0-d list array stands for it); a
        Python bool / int / str / tuple (an attribute, an identity test) as ``pyvalue``; anything else as ``expect_of``.
    """
    if isinstance(r, np.ndarray) and r.dtype == object:
        d = _cobj_expect(r, [])
        d["layout"] = {"strides": [int(s) for s in r.strides], "c": bool(r.flags.c_contiguous),
                       "f": bool(r.flags.f_contiguous), "own": bool(r.flags.owndata), "w": bool(r.flags.writeable),
                       "base": _clst_base(r, recv)}
        if r.size == 0:
            # NumPy 2.x gives every zero-size array it allocates all-zero strides, NumSharp the C strides of its shape —
            # the library-wide open bug pinned in OpenBugs.EmptyStrides.cs, not a list-array behavior.
            del d["layout"]["strides"]
        return d
    if isinstance(r, list):
        selfs = [0] if any(type(p) is np.str_ for p in r) else []
        return {"kind": "lists", "shape": [], "lists": [[_cobj_piece(p) for p in r]], "strself": selfs}
    if isinstance(r, (bool, int, str, tuple)):
        return {"kind": "pyvalue", "value": enc(list(r) if isinstance(r, tuple) else r)}
    return expect_of(r, [])


def _clst_call(params, views):
    """Makes one case's call: the member ``params.name`` of the receiver (``_cobj_value(params.recv)``) with ``args`` /
    ``kw``; an attribute (no call) when ``params.attr`` is true; ``is_recv`` reports whether the result IS the receiver.

    Returns:
        (result, receiver).
    """
    recv = _cobj_value(params["recv"], views)
    attr = getattr(recv, params["name"])
    if params.get("attr"):
        r = attr
    else:
        args = [_cobj_value(a, views) for a in params.get("args", [])]
        kw = {name: _cobj_value(a, views) for name, a in params.get("kw", {}).items()}
        r = attr(*args, **kw)
    if params.get("is_recv"):
        r = r is recv
    return r, recv


def add_clst(params, operands=(), layout=""):
    """Records one list-array member case (``_clst_call``)."""
    views = [v for _, v in operands]
    prefix = f"{CLST}/clst/"
    n = _OP_COUNT.get(prefix, 0) + 1
    _OP_COUNT[prefix] = n
    cid = f"{CLST}/clst/{n}"
    assert cid not in _IDS
    _IDS.add(cid)
    case = {"id": cid, "op": "clst", "params": params, "operands": [describe(b, v) for b, v in operands], "phase": 1,
            "layout": layout}
    try:
        r, recv = _clst_call(params, views)
        case["expected"] = _clst_expect(r, recv)
    except Exception as e:  # noqa: BLE001 -- every NumPy exception is a recorded outcome
        case["error"] = err_info(e)
    CASES.append(case)


def gen_strcharlist():
    """Section strcharlist: NDListArray's structural ndarray members over np.char.split results of every class (U / S /
    StringDType) and core layout, and over views of them (a transpose, a reversed slice, a 0-d element view): T / mT /
    transpose / swapaxes, reshape (C and F order) / ravel / flatten / squeeze, copy / view, take / compress / repeat /
    diagonal with their arguments and errors, and the attributes strides / itemsize / nbytes / device / to_device."""
    for kind in ("U", "S", "T"):
        pool = COBJ_SPLIT_S if kind == "S" else COBJ_SPLIT_U
        for li, layout in enumerate(CORE_LAYOUTS + ["c3", "d5", "one", "e1"]):
            ops = [_cobj_operand(kind, pool, layout, seed=li)]
            shape = ops[0][1].shape
            nd = len(shape)
            size = int(np.prod(shape)) if shape else 1
            receivers = [({"split": [{"op": 0}]}, "")]
            if nd >= 1 and size > 0:
                receivers.append(({"view": [{"split": [{"op": 0}]}, "::-1" + ", :" * (nd - 1)]}, ":rev"))
            for recv, rl in receivers:
                label = f"{kind}:{layout}{rl}"

                def m(name, args=(), kw=None, sub="", attr=False, is_recv=False):
                    params = {"recv": recv, "name": name, "args": list(args), "kw": kw or {}}
                    if attr:
                        params["attr"] = True
                    if is_recv:
                        params["is_recv"] = True
                    add_clst(params, ops, layout=f"{label}{sub}")

                v = _cobj_v
                # ---- attributes ----
                for name in ("T", "mT", "strides", "itemsize", "nbytes", "device", "shape", "ndim", "size"):
                    # An empty array's strides are NumPy's all-zero strides of a zero-size allocation (OpenBugs.EmptyStrides).
                    if name == "strides" and size == 0:
                        continue
                    m(name, attr=True)
                m("to_device", [v("cpu")], is_recv=True)
                m("to_device", [v("gpu")])
                # ---- transposition ----
                m("transpose")
                if nd >= 2:
                    m("transpose", list(map(v, range(nd)))[::-1], sub=":rev")
                    m("transpose", [v(0)] * nd, sub=":repeat")
                    m("swapaxes", [v(0), v(nd - 1)])
                    m("swapaxes", [v(-1), v(0)], sub=":neg")
                m("transpose", [v(nd)] + [v(i) for i in range(1, nd)] if nd else [v(0)], sub=":oob")
                m("swapaxes", [v(0), v(nd + 1)], sub=":oob")
                # ---- reshaping ----
                m("reshape", [v(-1)])
                m("reshape", [v(size), v(1)] if size else [v(0), v(1)], sub=":col")
                m("reshape", [v(size + 1)], sub=":bad")
                m("reshape", [v(-1), v(-1)], sub=":two_unknown")
                if size % 2 == 0 and size:
                    m("reshape", [v(2), v(-1)], sub=":two")
                    m("reshape", [v([2, size // 2])], {"order": v("F")}, sub=":F")
                for order in ("C", "F", "A", "K"):
                    m("ravel", [v(order)], sub=f":{order}")
                    m("flatten", [v(order)], sub=f":{order}")
                    m("copy", [v(order)], sub=f":{order}")
                m("squeeze", is_recv=True, sub=":is")
                m("squeeze")
                if nd >= 1:
                    m("squeeze", [v(0)], sub=":axis0")
                m("squeeze", [v(nd + 2)], sub=":oob")
                m("view")
                # ---- selection ----
                if size > 0:
                    m("take", [v(0)])
                    m("take", [v([0, size - 1, 0])], sub=":list")
                    m("take", [v(size + 1)], sub=":oob")
                    m("take", [v(size + 1)], {"mode": v("wrap")}, sub=":wrap")
                    m("take", [v(-1)], {"mode": v("clip")}, sub=":clip")
                    m("take", [v([[0], [size - 1]])], sub=":2d")
                    if nd >= 2:
                        m("take", [v([1, 0])], {"axis": v(nd - 1)}, sub=":axis")
                        m("take", [v([0])], {"axis": v(nd)}, sub=":axis_oob")
                    m("compress", [v(([True, False] * size)[:size])])
                    m("compress", [v([False] * size + [True])], sub=":past")
                    m("repeat", [v(2)])
                    m("repeat", [v(0)], sub=":zero")
                    if nd >= 1:
                        m("repeat", [v(list(range(shape[0])))], {"axis": v(0)}, sub=":counts")
                    if nd >= 2:
                        m("compress", [v([True, False, True][: shape[-1]])], {"axis": v(nd - 1)}, sub=":axis")
                        m("diagonal")
                        m("diagonal", [v(1)], sub=":1")
                        m("diagonal", [v(-1)], sub=":-1")
                        m("diagonal", [v(0), v(nd - 1), v(0)], sub=":axes")
                m("diagonal", [v(0), v(0), v(0)], sub=":same")
    # A view of a view (base collapses to the owner), the element view of a 0-d index, and a temporary's base.
    m2 = _cobj_operand("U", COBJ_SPLIT_U, "c2", seed=3)
    for recv, rl in (({"view": [{"split": [{"op": 0}]}, "1:, ::-1"]}, "view"), ({"view": [{"split": [{"op": 0}]}, "0, 1:"]}, "row")):
        for name, args, kw in (("T", None, None), ("reshape", [-1], None), ("ravel", ["K"], None), ("copy", [], None),
                               ("diagonal", [], None), ("take", [[0]], None), ("squeeze", [], None)):
            params = {"recv": recv, "name": name, "args": [_cobj_v(x) for x in (args or [])],
                      "kw": {k: _cobj_v(x) for k, x in (kw or {}).items()}}
            if args is None:
                params["attr"] = True
            add_clst(params, [m2], layout=f"U:c2:{rl}")


# ----------------------------------------------------------------------------------------------------------------------
# Section strcharmeth: numpy.char's string functions (its partition / rpartition wrappers and the functions it
# re-exports from numpy.strings) and the chararray string members over them (the numpy.char plan's C4 / C5)
# ----------------------------------------------------------------------------------------------------------------------
#
# Each case records the result's CLASS, OWNDATA, strides and memory sharing (_chx_expect), so the three return rules of
# the chararray members are checked, not only their values: asarray(...) (a U result owns its data, an S result views the
# computed array), the ufunc's chararray view (chararray.__array_wrap__), or a plain int64 / bool array. The functions run
# over plain and chararray operands of every core layout of both classes, over StringDType, over Python values (str,
# bytes, lists, numbers) and into NumPy's refusals; a chararray in a position other than the text keeps the result plain
# where NumPy's out= comes from np.empty_like(a). A C# null is NumPy's default for an optional argument and an
# ArgumentNullException for a required one (the module's convention), so no case passes Python's None.

CHM = "strcharmeth"
CHM_LAYOUTS = ["c1", "c2", "f2", "t3", "s1", "r2", "o1", "col", "bc", "z0", "e1", "e2", "one", "d5"]
# StringDType texts for the module functions: separators, tabs, signs for zfill, non-ASCII text.
CHM_T = ["ab cd", "x", "", "a\tb c", "-12", "\xe9 中", "+7", "q-r-s"]


def _chm_words(spec):
    """The text constructors of a section-strcharmeth operand class.

    Args:
        spec: ``"<U4"`` or ``"|S4"``.

    Returns:
        ``(word, alien)``: a str to a value of spec's class, and a str to a value of the other class.
    """
    if spec[1] == "U":
        return (lambda s: s), (lambda s: s.encode("latin-1"))
    return (lambda s: s.encode("latin-1")), (lambda s: s)


def _chm_member(recipes, name, args=(), kw=None, layout=""):
    """Records one chararray member call on operand 0's chararray view.

    Args:
        recipes: the operands' recipes (operand 0 is the receiver's base).
        name: the member.
        args: the positional argument specs.
        kw: the keyword argument specs.
        layout: the case's layout label.
    """
    add_chx(CHM, "cha_call", {"recv": 0, "name": name, "args": list(args), "kw": kw or {}}, recipes, layout=layout)


def _chm_fn(recipes, fn, args, kw=None, layout=""):
    """Records one numpy.char function call.

    Args:
        recipes: the operands' recipes.
        fn: the function's name under ``np.char``.
        args: the positional argument specs.
        kw: the keyword argument specs.
        layout: the case's layout label.
    """
    add_chx(CHM, "cha_fn", {"fn": "char." + fn, "args": list(args), "kw": kw or {}}, recipes, layout=layout)


def _chm_transform():
    """numpy.char's partition / rpartition (NumPy's np.stack wrappers) and center / ljust / rjust / zfill / expandtabs
    (numpy.strings' functions), with the chararray members of the six names (C4, and C5's first batch)."""
    for spec in ["<U4", "|S4"]:
        word, alien = _chm_words(spec)
        for layout in CHM_LAYOUTS:
            cha = _chx_pool(spec, layout, 3)
            shape = _chx_shape(cha)
            nd = len(shape)
            size = int(np.prod(shape)) if shape else 1
            row = shape[-1] if nd else 0
            label = f"{spec}:{layout}"
            # ---- the members: chararray.partition / rpartition -> asarray(np.char.partition(self, sep)) ----
            for name in ["partition", "rpartition"]:
                for sep, sl in [(word(" "), "space"), (word("a"), "a"), (word("ab"), "ab"), (alien("a"), "alien"),
                                (1, "int"), (word(""), "empty")]:
                    _chm_member([cha], name, [_chx_v(sep)], layout=f"{label}:{sl}")
                _chm_member([cha], name, [], {"sep": _chx_v(word("c"))}, layout=f"{label}:kw")
                if nd >= 1 and size > 0:
                    seps = [word(w) for w in [" ", "a", "b", "c d", "\t"] * 3][:row]
                    _chm_member([cha], name, [_chx_v(seps)], layout=f"{label}:list")
                    _chm_member([cha], name, [_chx_v(seps + [word("x")])], layout=f"{label}:list_longer")
            # ---- center / ljust / rjust -> asarray(f(self, width, fillchar)) ----
            for name in ["center", "ljust", "rjust"]:
                for width in [0, 3, 9, -1]:
                    _chm_member([cha], name, [_chx_v(width)], layout=f"{label}:w{width}")
                _chm_member([cha], name, [_chx_v(9), _chx_v(word("*"))], layout=f"{label}:fill")
                _chm_member([cha], name, [_chx_v(9)], {"fillchar": _chx_v(alien("*"))}, layout=f"{label}:alien_fill")
                _chm_member([cha], name, [], {"width": _chx_v(8), "fillchar": _chx_v(word("."))}, layout=f"{label}:kw")
                _chm_member([cha], name, [_chx_v(9), _chx_v(word("ab"))], layout=f"{label}:long_fill")
                _chm_member([cha], name, [_chx_v(2.5)], layout=f"{label}:float")
                _chm_member([cha], name, [_chx_v(word("3"))], layout=f"{label}:text")
                _chm_member([cha], name, [_chx_v(True)], layout=f"{label}:bool")
                if spec[1] == "S":
                    _chm_member([cha], name, [_chx_v(9), _chx_v("\xe9")], layout=f"{label}:nonascii_fill")
                if nd >= 1 and size > 0:
                    widths = ([2, 5, 9, 0] * 3)[:row]
                    _chm_member([cha], name, [_chx_v(widths)], layout=f"{label}:widths")
                    fills = [word(f) for f in ["*", "-", ".", "#"] * 3][:row]
                    _chm_member([cha], name, [_chx_v(7), _chx_v(fills)], layout=f"{label}:fills")
            # ---- zfill -> asarray(zfill(self, width)), over a receiver without an empty element (NumPy 2.4.2's
            # zfill of an empty element reads uninitialized memory: _fn_nonempty) ----
            zcha = {"nonempty": [spec, layout, 3]}
            for width in [0, 3, 7]:
                _chm_member([zcha], "zfill", [_chx_v(width)], layout=f"{label}:w{width}")
            _chm_member([zcha], "zfill", [], {"width": _chx_v(5)}, layout=f"{label}:kw")
            _chm_member([zcha], "zfill", [_chx_v(2.5)], layout=f"{label}:float")
            _chm_member([zcha], "zfill", [_chx_v(word("3"))], layout=f"{label}:text")
            if nd >= 1 and size > 0:
                _chm_member([zcha], "zfill", [_chx_v(([1, 4, 6] * 4)[:row])], layout=f"{label}:widths")
            # ---- expandtabs -> asarray(expandtabs(self, tabsize)) ----
            _chm_member([cha], "expandtabs", [], layout=f"{label}:default")
            for tab in [3, 0, -1]:
                _chm_member([cha], "expandtabs", [_chx_v(tab)], layout=f"{label}:t{tab}")
            _chm_member([cha], "expandtabs", [], {"tabsize": _chx_v(2)}, layout=f"{label}:kw")
            _chm_member([cha], "expandtabs", [_chx_v(2.0)], layout=f"{label}:float")
        # zfill's sign rule and expandtabs' columns over values that hold them (C order and a reversed view).
        signs = [word(v) for v in ["-7", "+3", "12", "x", "-", "+-1", " -2", "0"]]
        tabs = [word(v) for v in ["a\tb", "\t", "ab\tc\td", "x\ny\tz", "\r\t", "", "\t\t", "abcd\te"]]
        for values, vl in [(signs, "signs"), (tabs, "tabs")]:
            recipe = _chx_vals(values, spec.replace("4", "8"))
            if vl == "signs":
                for width in [1, 5]:
                    _chm_member([recipe], "zfill", [_chx_v(width)], layout=f"{spec}:{vl}:zfill{width}")
            for tab in [8, 4, 1, 0]:
                _chm_member([recipe], "expandtabs", [_chx_v(tab)], layout=f"{spec}:{vl}:tab{tab}")
            _chm_member([recipe], "center", [_chx_v(11), _chx_v(word("~"))], layout=f"{spec}:{vl}:center")
            _chm_member([recipe], "partition", [_chx_v(word("\t"))], layout=f"{spec}:{vl}:partition")
            _chm_member([recipe], "rpartition", [_chx_v(word("-"))], layout=f"{spec}:{vl}:rpartition")

    # ---- the module functions over plain and chararray operands (numpy.strings' class rules) ----
    for spec in ["<U4", "|S4"]:
        word, alien = _chm_words(spec)
        for layout in CHM_LAYOUTS:
            base = _chx_pool(spec, layout, 4)
            zbase = {"nonempty": [spec, layout, 4]}
            shape = _chx_shape(base)
            nd = len(shape)
            size = int(np.prod(shape)) if shape else 1
            label = f"{spec}:{layout}"
            for view, vl in [(_chx_arr(0), "plain"), (_chx_cha(0), "chararray")]:
                for fn in ["partition", "rpartition"]:
                    _chm_fn([base], fn, [view, _chx_v(word(" "))], layout=f"{label}:{vl}")
                    _chm_fn([base], fn, [view], {"sep": _chx_v(word("a"))}, layout=f"{label}:{vl}:kw")
                    _chm_fn([base], fn, [view, _chx_v(word(""))], layout=f"{label}:{vl}:empty_sep")
                _chm_fn([base], "center", [view, _chx_v(9)], layout=f"{label}:{vl}")
                _chm_fn([base], "center", [view, _chx_v(9), _chx_v(word("*"))], layout=f"{label}:{vl}:fill")
                _chm_fn([base], "ljust", [view, _chx_v(7)], {"fillchar": _chx_v(word("-"))}, layout=f"{label}:{vl}")
                _chm_fn([base], "rjust", [view, _chx_v(-2)], layout=f"{label}:{vl}")
                _chm_fn([zbase], "zfill", [view, _chx_v(6)], layout=f"{label}:{vl}")
                _chm_fn([base], "expandtabs", [view], layout=f"{label}:{vl}")
                _chm_fn([base], "expandtabs", [view, _chx_v(2)], layout=f"{label}:{vl}:tab2")
            if nd >= 1 and size > 0:
                # A chararray in a non-text position: numpy.strings' out= is np.empty_like(a) (plain), and np.stack
                # keeps no subclass, so the results stay plain.
                fill = _chx_vals([word("*")], spec)
                sep = _chx_vals([word(" ")], spec)
                _chm_fn([base, fill], "center", [_chx_arr(0), _chx_v(9), _chx_cha(1)], layout=f"{label}:chararray_fill")
                _chm_fn([base, sep], "partition", [_chx_arr(0), _chx_cha(1)], layout=f"{label}:chararray_sep")
                _chm_fn([base, sep], "rpartition", [_chx_cha(0), _chx_arr(1)], layout=f"{label}:plain_sep")
    # StringDType text (a StringDType separator: NumPy 2.4.2 has no (StringDType, str) partition loop).
    t_text = {"T": enc(CHM_T)}
    t_sep = {"T": enc([" "])}
    for fn in ["partition", "rpartition"]:
        _chm_fn([t_text, t_sep], fn, [_chx_arr(0), _chx_arr(1)], layout="T:T_sep")
        _chm_fn([t_text], fn, [_chx_arr(0), _chx_v(" ")], layout="T:str_sep")
        _chm_fn([t_text], fn, [_chx_arr(0), _chx_v(b" ")], layout="T:bytes_sep")
    for fn, args in [("center", [_chx_v(9)]), ("center", [_chx_v(9), _chx_v("*")]), ("ljust", [_chx_v(6)]),
                     ("rjust", [_chx_v(6), _chx_v("0")]), ("expandtabs", []), ("expandtabs", [_chx_v(3)]),
                     ("center", [_chx_v(2.5)]), ("center", [_chx_v(9), _chx_v("**")])]:
        _chm_fn([t_text], fn, [_chx_arr(0)] + args, layout="T")
    # zfill over StringDType without an empty text (NumPy reads past the written buffer for a sign: _fn_nonempty).
    _chm_fn([{"T": enc([v for v in CHM_T if v])}], "zfill", [_chx_arr(0), _chx_v(5)], layout="T")
    # Python values as the text: a str, bytes, lists, a number (cast to text, or NumPy's refusal).
    for value, vl in [("ab cd", "str"), (b"ab cd", "bytes"), (["ab cd", "x y"], "list"), ([b"a-b", b"c"], "bytes_list"),
                      ([["a b", "c"], ["d", "e f"]], "nested"), (5, "int")]:
        sep = b" " if isinstance(value, bytes) or (isinstance(value, list) and isinstance(value[0], bytes)) else " "
        for fn in ["partition", "rpartition"]:
            _chm_fn([], fn, [_chx_v(value), _chx_v(sep)], layout=f"value:{vl}")
        _chm_fn([], "center", [_chx_v(value), _chx_v(8)], layout=f"value:{vl}")
        _chm_fn([], "zfill", [_chx_v(value), _chx_v(4)], layout=f"value:{vl}")
        _chm_fn([], "expandtabs", [_chx_v(value)], layout=f"value:{vl}")
    # Broadcasting: a column of texts against a row of separators / widths, and a shape mismatch.
    col = _chx_vals([["ab cd"], ["x-y z"], ["q"]], "<U5")
    _chm_fn([col], "partition", [_chx_arr(0), _chx_v([" ", "-"])], layout="broadcast")
    _chm_fn([col], "rpartition", [_chx_cha(0), _chx_v([" ", "-"])], layout="broadcast:chararray")
    _chm_fn([col], "center", [_chx_arr(0), _chx_v([4, 9])], layout="broadcast")
    _chm_fn([col], "partition", [_chx_arr(0), _chx_v([[" ", "-"], ["y", "z"]])], layout="broadcast:mismatch")


def _chm_search():
    """numpy.char's length and search functions (numpy.strings' str_len, count, find, rfind, index, rindex, startswith,
    endswith) and the chararray members of the seven search names (C5's second batch): plain int64 / bool results."""
    search = [("count", "sub"), ("find", "sub"), ("rfind", "sub"), ("index", "sub"), ("rindex", "sub"),
              ("startswith", "prefix"), ("endswith", "suffix")]
    for spec in ["<U4", "|S4"]:
        word, alien = _chm_words(spec)
        for layout in CHM_LAYOUTS:
            cha = _chx_pool(spec, layout, 3)
            shape = _chx_shape(cha)
            nd = len(shape)
            size = int(np.prod(shape)) if shape else 1
            row = shape[-1] if nd else 0
            label = f"{spec}:{layout}"
            # ---- the members: chararray.<name>(sub, start, end) -> <name>(self, sub, start, end), a plain result ----
            for name, argname in search:
                # The stored text is searched unstripped: a space or a NUL finds the trailing whitespace c[i] hides.
                for sub, sl in [(word("a"), "a"), (word(" "), "space"), (word(""), "empty"), (word("ab"), "ab"),
                                (word("\x00"), "nul"), (word("  "), "two_spaces")]:
                    _chm_member([cha], name, [_chx_v(sub)], layout=f"{label}:{sl}")
                _chm_member([cha], name, [_chx_v(word(" ")), _chx_v(1)], layout=f"{label}:start1")
                _chm_member([cha], name, [_chx_v(word(" ")), _chx_v(0), _chx_v(2)], layout=f"{label}:end2")
                _chm_member([cha], name, [_chx_v(word(" "))], {"end": _chx_v(-1)}, layout=f"{label}:end_neg")
                _chm_member([cha], name, [_chx_v(word(" ")), _chx_v(-3), _chx_v(100)], layout=f"{label}:wide_range")
                _chm_member([cha], name, [_chx_v(word("")), _chx_v(5), _chx_v(2)], layout=f"{label}:empty_range")
                _chm_member([cha], name, [], {argname: _chx_v(word(" ")), "start": _chx_v(1)}, layout=f"{label}:kw")
                # NumPy's refusals: no loop for a substring of the other class or a number, a start that cannot reach
                # int64 under same_kind (a float, text).
                _chm_member([cha], name, [_chx_v(alien("a"))], layout=f"{label}:alien")
                _chm_member([cha], name, [_chx_v(1)], layout=f"{label}:int_sub")
                _chm_member([cha], name, [_chx_v(word("a")), _chx_v(1.5)], layout=f"{label}:float_start")
                _chm_member([cha], name, [_chx_v(word("a")), _chx_v(word("1"))], layout=f"{label}:text_start")
                if nd >= 1 and size > 0:
                    subs = [word(w) for w in ["a", " ", "", "c d"] * 3][:row]
                    _chm_member([cha], name, [_chx_v(subs)], layout=f"{label}:subs")
                    _chm_member([cha], name, [_chx_v(word(" ")), _chx_v(([0, 1, -1, 2] * 3)[:row])],
                                layout=f"{label}:starts")
                    _chm_member([cha], name, [_chx_v(word(" ")), _chx_v(0), _chx_v(([1, 3, -2, 9] * 3)[:row])],
                                layout=f"{label}:ends")
                    _chm_member([cha], name, [_chx_v(subs + [word("x")])], layout=f"{label}:subs_longer")
        # Long texts: the search kernels' vector paths (a match past the first vector, a miss over a whole element).
        longs = [word(v) for v in ["x" * 40 + "ab" + "y" * 20, "ab" * 33, "q" * 64, "a b " * 15 + "ab", "", "ab"]]
        recipe = _chx_vals(longs, spec.replace("4", "70"))
        for name, _ in search:
            _chm_member([recipe], name, [_chx_v(word("ab"))], layout=f"{spec}:long")
            _chm_member([recipe], name, [_chx_v(word("ab")), _chx_v(10), _chx_v(60)], layout=f"{spec}:long:range")
        for name, _ in search:
            _chm_member([recipe], name, [_chx_v(word("b "))], layout=f"{spec}:long:other")
            _chm_member([recipe], name, [_chx_v(word("y"))], {"start": _chx_v(-30)}, layout=f"{spec}:long:tail")

    # ---- the module functions over plain and chararray operands (numpy.strings' class rules: always plain) ----
    for spec in ["<U4", "|S4"]:
        word, alien = _chm_words(spec)
        for layout in CHM_LAYOUTS:
            base = _chx_pool(spec, layout, 4)
            shape = _chx_shape(base)
            nd = len(shape)
            size = int(np.prod(shape)) if shape else 1
            label = f"{spec}:{layout}"
            for view, vl in [(_chx_arr(0), "plain"), (_chx_cha(0), "chararray")]:
                _chm_fn([base], "str_len", [view], layout=f"{label}:{vl}")
                for name, argname in search:
                    _chm_fn([base], name, [view, _chx_v(word(" "))], layout=f"{label}:{vl}")
                    _chm_fn([base], name, [view, _chx_v(word("a")), _chx_v(1)], layout=f"{label}:{vl}:start")
                    _chm_fn([base], name, [view], {argname: _chx_v(word("")), "end": _chx_v(2)},
                            layout=f"{label}:{vl}:kw")
            if nd >= 1 and size > 0:
                # A chararray in a non-text position (the substring): the result stays plain.
                sub = _chx_vals([word(" ")], spec)
                for name, _ in search:
                    _chm_fn([base, sub], name, [_chx_arr(0), _chx_cha(1)], layout=f"{label}:chararray_sub")
                # str_len's ufunc keywords: out= (a same_kind int32 out; a chararray out is not int64-castable text),
                # where= with an out, dtype= (only int64 has a loop).
                out = _chx_vals(np.zeros(shape, dtype=np.int32).tolist(), "int32", out=True)
                _chm_fn([base, out], "str_len", [_chx_cha(0)], {"out": _chx_arr(1)}, layout=f"{label}:out")
                outz = _chx_vals(np.full(shape, 7, dtype=np.int64).tolist(), "int64", out=True)
                mask = _chx_vals((np.arange(size) % 2 == 0).reshape(shape).tolist(), "bool")
                _chm_fn([base, outz, mask], "str_len", [_chx_arr(0)], {"out": _chx_arr(1), "where": _chx_arr(2)},
                        layout=f"{label}:where")
                _chm_fn([base], "str_len", [_chx_arr(0)], {"dtype": {"dt": "int64"}}, layout=f"{label}:dtype")
                _chm_fn([base], "str_len", [_chx_arr(0)], {"dtype": {"dt": "int32"}}, layout=f"{label}:dtype_noloop")
    # StringDType text (a StringDType or str substring; a bytes one has no loop).
    t_text = {"T": enc(CHM_T)}
    t_sub = {"T": enc([" "])}
    _chm_fn([t_text], "str_len", [_chx_arr(0)], layout="T")
    for name, _ in search:
        _chm_fn([t_text, t_sub], name, [_chx_arr(0), _chx_arr(1)], layout="T:T_sub")
        _chm_fn([t_text], name, [_chx_arr(0), _chx_v(" ")], layout="T:str_sub")
        _chm_fn([t_text], name, [_chx_arr(0), _chx_v(b" ")], layout="T:bytes_sub")
        _chm_fn([t_text], name, [_chx_arr(0), _chx_v("b"), _chx_v(1), _chx_v(-1)], layout="T:range")
    # Python values as the text: a str, bytes, lists, numbers (no loop: str_len, a one-input ufunc, names a number by its
    # default DType, Int64DType / Float64DType; the four-input search ufuncs name the weak _PyLongDType / _PyFloatDType).
    for value, vl in [("ab cd", "str"), (b"ab cd", "bytes"), (["ab cd", "x y"], "list"), ([b"a-b", b"c"], "bytes_list"),
                      ([["a b", "c"], ["d", "e f"]], "nested"), (5, "int"), (5.5, "float")]:
        sub = b" " if isinstance(value, bytes) or (isinstance(value, list) and isinstance(value[0], bytes)) else " "
        _chm_fn([], "str_len", [_chx_v(value)], layout=f"value:{vl}")
        for name, _ in search:
            _chm_fn([], name, [_chx_v(value), _chx_v(sub)], layout=f"value:{vl}")
    # Broadcasting: a column of texts against a row of substrings / starts, and a shape mismatch.
    col = _chx_vals([["ab cd"], ["x-y z"], ["q"]], "<U5")
    for name, _ in search:
        _chm_fn([col], name, [_chx_arr(0), _chx_v([" ", "-"])], layout="broadcast")
        _chm_fn([col], name, [_chx_cha(0), _chx_v("y"), _chx_v([0, 3])], layout="broadcast:chararray")
        _chm_fn([col], name, [_chx_arr(0), _chx_v([[" ", "-"], ["y", "z"]])], layout="broadcast:mismatch")


def _chm_strip():
    """numpy.char's strip / lstrip / rstrip / replace (numpy.strings' functions) and the chararray members of the four
    names (C5's third batch): NumPy's rule 2, the function's own chararray view."""
    strips = ["strip", "lstrip", "rstrip"]
    for spec in ["<U4", "|S4"]:
        word, alien = _chm_words(spec)
        for layout in CHM_LAYOUTS:
            cha = _chx_pool(spec, layout, 3)
            shape = _chx_shape(cha)
            nd = len(shape)
            size = int(np.prod(shape)) if shape else 1
            row = shape[-1] if nd else 0
            label = f"{spec}:{layout}"
            # ---- the members: chararray.<strip>(chars) -> <strip>(self, chars), a chararray view ----
            for name in strips:
                _chm_member([cha], name, [], layout=f"{label}:default")
                # The stored text: whitespace, NULs and the pool's other code points the element access strips.
                for chars, cl in [(word(" "), "space"), (word("ab"), "ab"), (word(" \t"), "space_tab"), (word("x"), "x"),
                                  (word(""), "empty"), (word("\x00 "), "nul_space")]:
                    _chm_member([cha], name, [_chx_v(chars)], layout=f"{label}:{cl}")
                _chm_member([cha], name, [], {"chars": _chx_v(word("a "))}, layout=f"{label}:kw")
                # NumPy's refusals: a set of the other class, a number.
                _chm_member([cha], name, [_chx_v(alien("a"))], layout=f"{label}:alien")
                _chm_member([cha], name, [_chx_v(5)], layout=f"{label}:int")
                if nd >= 1 and size > 0:
                    sets = [word(w) for w in [" ", "ab", "", "\t "] * 3][:row]
                    _chm_member([cha], name, [_chx_v(sets)], layout=f"{label}:sets")
                    _chm_member([cha], name, [_chx_v(sets + [word("x")])], layout=f"{label}:sets_longer")
            # ---- chararray.replace(old, new, count=None) -> replace(self, old, new, count or -1) ----
            for old, new, rl in [(word(" "), word("_"), "space"), (word("a"), word("xyz"), "grow"),
                                 (word("ab"), word(""), "delete"), (word(""), word("-"), "empty_old"),
                                 (word("  "), word("."), "two_spaces"), (word("q"), word("r"), "absent")]:
                _chm_member([cha], "replace", [_chx_v(old), _chx_v(new)], layout=f"{label}:{rl}")
            for count in [0, 1, -1, 3]:
                _chm_member([cha], "replace", [_chx_v(word(" ")), _chx_v(word("#")), _chx_v(count)],
                            layout=f"{label}:count{count}")
            _chm_member([cha], "replace", [], {"old": _chx_v(word("a")), "new": _chx_v(word("A")), "count": _chx_v(1)},
                        layout=f"{label}:kw")
            # The cast of a Python value of the other class (old / new have no dtype: astype(a.dtype)), a count of
            # another kind (NumPy's TypeError).
            _chm_member([cha], "replace", [_chx_v(alien("a")), _chx_v(word("b"))], layout=f"{label}:alien_old")
            _chm_member([cha], "replace", [_chx_v(word("a")), _chx_v(alien("b"))], layout=f"{label}:alien_new")
            _chm_member([cha], "replace", [_chx_v(word("a")), _chx_v(word("b")), _chx_v(1.5)], layout=f"{label}:float_count")
            if spec[1] == "S":
                _chm_member([cha], "replace", [_chx_v("\xe9"), _chx_v(word("b"))], layout=f"{label}:nonascii_old")
            if nd >= 1 and size > 0:
                counts = ([0, 1, -1, 2] * 3)[:row]
                _chm_member([cha], "replace", [_chx_v(word(" ")), _chx_v(word("+")), _chx_v(counts)],
                            layout=f"{label}:counts")
                olds = [word(w) for w in ["a", " ", "", "b"] * 3][:row]
                _chm_member([cha], "replace", [_chx_v(olds), _chx_v(word("*"))], layout=f"{label}:olds")

    # ---- the module functions over plain and chararray operands (numpy.strings' class rules) ----
    for spec in ["<U4", "|S4"]:
        word, alien = _chm_words(spec)
        for layout in CHM_LAYOUTS:
            base = _chx_pool(spec, layout, 4)
            shape = _chx_shape(base)
            nd = len(shape)
            size = int(np.prod(shape)) if shape else 1
            label = f"{spec}:{layout}"
            for view, vl in [(_chx_arr(0), "plain"), (_chx_cha(0), "chararray")]:
                for name in strips:
                    _chm_fn([base], name, [view], layout=f"{label}:{vl}")
                    _chm_fn([base], name, [view, _chx_v(word(" a"))], layout=f"{label}:{vl}:chars")
                _chm_fn([base], "replace", [view, _chx_v(word(" ")), _chx_v(word("__"))], layout=f"{label}:{vl}")
                _chm_fn([base], "replace", [view, _chx_v(word("a")), _chx_v(word("")), _chx_v(1)],
                        layout=f"{label}:{vl}:count")
                _chm_fn([base], "replace", [view], {"old": _chx_v(word("")), "new": _chx_v(word("."))},
                        layout=f"{label}:{vl}:kw")
            if nd >= 1 and size > 0:
                # A chararray in a position other than the text: the result follows the text's class.
                chars = _chx_vals([word(" ")], spec)
                old = _chx_vals([word("a")], spec)
                _chm_fn([base, chars], "strip", [_chx_arr(0), _chx_cha(1)], layout=f"{label}:chararray_chars")
                _chm_fn([base, old], "replace", [_chx_arr(0), _chx_cha(1), _chx_v(word("z"))],
                        layout=f"{label}:chararray_old")
    # StringDType text (a StringDType or str argument; a bytes one has no loop) and a chararray with StringDType
    # arguments (NumPy's finalize error: the result is not text a chararray holds).
    t_text = {"T": enc(CHM_T)}
    t_chars = {"T": enc([" a"])}
    for name in strips:
        _chm_fn([t_text], name, [_chx_arr(0)], layout="T")
        _chm_fn([t_text, t_chars], name, [_chx_arr(0), _chx_arr(1)], layout="T:T_chars")
        _chm_fn([t_text], name, [_chx_arr(0), _chx_v(" a")], layout="T:str_chars")
        _chm_fn([t_text], name, [_chx_arr(0), _chx_v(b" ")], layout="T:bytes_chars")
        _chm_member([_chx_pool("<U4", "c1", 2), t_chars], name, [_chx_arr(1)], layout="U:T_chars")
    _chm_fn([t_text], "replace", [_chx_arr(0), _chx_v(" "), _chx_v("__")], layout="T")
    _chm_fn([t_text], "replace", [_chx_arr(0), _chx_v("-"), _chx_v(""), _chx_v(1)], layout="T:count")
    _chm_fn([t_text, t_chars], "replace", [_chx_arr(0), _chx_arr(1), _chx_v("+")], layout="T:T_old")
    _chm_member([_chx_pool("<U4", "c1", 2), t_chars], "replace", [_chx_arr(1), _chx_v("q")], layout="U:T_old")
    _chm_member([_chx_pool("<U4", "c1", 2), t_chars], "replace", [_chx_v("a"), _chx_arr(1)], layout="U:T_new")
    # Wide U elements (44 units, past the 64-byte lane cap: the chunked chars strip): stripped runs inside one 8-unit chunk,
    # across chunk boundaries and over the whole text, a set holding a non-ASCII space or NUL, an empty element.
    wide = _chx_vals(["aaaaaaaaaaaaaaaaaaaa middle ||||||||||||||||", "|a|a|a|a|a|a|a|a|a|a|a", "no strip here at all, none",
                      "", "a| \u3000a|" * 4 + "x" + " |a" * 7, "\u3000" * 17 + "q" + "\u3000" * 20, "| a" * 14 + "|"],
                     "<U44")
    for name in strips:
        _chm_fn([wide], name, [_chx_arr(0)], layout="wide:default")
        for chars, cl in [("a|", "a_bar"), ("a| ", "a_bar_space"), ("|", "bar"), ("\u3000 a|", "wide_space"),
                          ("xyz", "absent"), ("\x00a|", "nul_set"), ("a| \u3000mxqiddle,nostrph", "many")]:
            _chm_fn([wide], name, [_chx_arr(0), _chx_v(chars)], layout=f"wide:{cl}")
            _chm_member([wide], name, [_chx_v(chars)], layout=f"wide:{cl}")
    # Python values as the text: a str, bytes, lists, numbers (no loop for the strips; replace casts a number to text).
    for value, vl in [("  ab cd ", "str"), (b" ab cd  ", "bytes"), (["ab cd ", " x y"], "list"),
                      ([b"a-b ", b" c"], "bytes_list"), ([["a b ", "c"], [" d", "e f"]], "nested"), (5, "int"),
                      (5.5, "float")]:
        isb = isinstance(value, bytes) or (isinstance(value, list) and isinstance(value[0], bytes))
        sp, us = (b" ", b"_") if isb else (" ", "_")
        for name in strips:
            _chm_fn([], name, [_chx_v(value)], layout=f"value:{vl}")
        _chm_fn([], "replace", [_chx_v(value), _chx_v(sp), _chx_v(us)], layout=f"value:{vl}")
    # Broadcasting: a column of texts against a row of sets / counts, and a shape mismatch.
    col = _chx_vals([["ab cd "], [" x-y z"], ["q"]], "<U6")
    for name in strips:
        _chm_fn([col], name, [_chx_arr(0), _chx_v([" ", "a "])], layout="broadcast")
        _chm_fn([col], name, [_chx_arr(0), _chx_v([[" ", "-"], ["y", "z"]])], layout="broadcast:mismatch")
    _chm_fn([col], "replace", [_chx_cha(0), _chx_v(" "), _chx_v("."), _chx_v([0, 1])], layout="broadcast:chararray")
    _chm_fn([col], "replace", [_chx_arr(0), _chx_v([" ", "x"]), _chx_v(".")], layout="broadcast")
    _chm_fn([col], "replace", [_chx_arr(0), _chx_v([[" ", "-"], ["y", "z"]]), _chx_v(".")], layout="broadcast:mismatch")


def _chm_tcopy():
    """NumPy's trivial-loop copy of a chararray INPUT beside a StringDType argument (C5's third batch, for every function
    of the section with a second text input): the search functions, the strips with a set, replace and the pads with a
    fill resolve a StringDType loop, and ``check_for_trivial_loop`` copies a 0-d or 1-d input of at most
    ``np.getbufsize()`` elements into ``StringDType()`` AS ITS CLASS, which a chararray's ``__array_finalize__`` refuses
    ("Can only create a chararray from string data."). An input of more dimensions is not copied: it refuses the trivial
    loop, no later input is copied, and the function computes a plain StringDType / int64 / bool array (a rule-2
    member's plain result is the excused NotSupportedException; a pad member converts it with np.char.asarray, an S
    chararray). A 1-d chararray past 8,192 elements takes the same branch; the unit tests pin that length with
    np.setbufsize instead of an 8,193-element operand per case.

    Every text position: the chararray as the text (members and functions, over the layouts), as the set / sub / old /
    new / fill beside a StringDType text, after a 2-d text or a 2-d int32 width that refuses the trivial loop first, and
    an S chararray (no loop for bytes beside StringDType: NumPy's TypeError comes before any copy)."""
    t1 = {"T": enc(["a"])}
    tf = {"T": enc([" "])}
    for layout in ["c1", "c2", "t3", "z0", "e1", "one", "bc", "s1", "col", "r2"]:
        cha = _chx_pool("<U4", layout, 5)
        label = f"tcopy:{layout}"
        for name in ["strip", "lstrip", "rstrip"]:
            _chm_member([cha, t1], name, [_chx_arr(1)], layout=label)
        _chm_member([cha, t1], "replace", [_chx_arr(1), _chx_v("q")], layout=f"{label}:T_old")
        _chm_member([cha, t1], "replace", [_chx_v("a"), _chx_arr(1)], layout=f"{label}:T_new")
        for name in ["find", "rfind", "count", "startswith", "endswith", "index", "rindex"]:
            _chm_member([cha, t1], name, [_chx_arr(1)], layout=label)
        for name in ["center", "ljust", "rjust"]:
            _chm_member([cha, tf], name, [_chx_v(9), _chx_arr(1)], layout=label)
        # The functions with the chararray as the text: the same copy, and NumPy's plain results past it.
        _chm_fn([cha, t1], "strip", [_chx_cha(0), _chx_arr(1)], layout=label)
        _chm_fn([cha, t1], "find", [_chx_cha(0), _chx_arr(1)], layout=label)
        _chm_fn([cha, t1], "endswith", [_chx_cha(0), _chx_arr(1)], layout=label)
        _chm_fn([cha, t1], "replace", [_chx_cha(0), _chx_arr(1), _chx_v("x")], layout=label)
        _chm_fn([cha, tf], "center", [_chx_cha(0), _chx_v(9), _chx_arr(1)], layout=label)
    # A non-ASCII StringDType fill: the pad member's np.char.asarray encodes the result as ASCII bytes (2-d: computed).
    _chm_member([_chx_pool("<U4", "c2", 5), {"T": enc(["\xe9"])}], "center", [_chx_v(9), _chx_arr(1)],
                layout="tcopy:c2:nonascii_fill")
    # The chararray in another text position, beside a StringDType text: one element (1-d: copied) or a 2-d one.
    t_text = {"T": enc(CHM_T)}
    for shape_label, wrap in [("1d", lambda v: [v]), ("2d", lambda v: [[v]])]:
        sub = _chx_vals(wrap("a"), "<U4")
        fill = _chx_vals(wrap(" "), "<U4")
        label = f"tcopy:second:{shape_label}"
        for name in ["find", "count", "startswith", "rindex"]:
            _chm_fn([t_text, sub], name, [_chx_arr(0), _chx_cha(1)], layout=label)
        for name in ["strip", "rstrip"]:
            _chm_fn([t_text, sub], name, [_chx_arr(0), _chx_cha(1)], layout=label)
        _chm_fn([t_text, sub], "replace", [_chx_arr(0), _chx_cha(1), _chx_v("x")], layout=f"{label}:old")
        _chm_fn([t_text, sub], "replace", [_chx_arr(0), _chx_v("a"), _chx_cha(1)], layout=f"{label}:new")
        for name in ["center", "ljust", "rjust"]:
            _chm_fn([t_text, fill], name, [_chx_arr(0), _chx_v(9), _chx_cha(1)], layout=label)
    # The order of the inputs: a 2-d int32 width (cast to int64, so not trivially iterable) refuses the trivial loop
    # BEFORE the fill, so a 1-d chararray fill is not copied; a 1-d width is copied and the fill after it is refused.
    fill1 = _chx_vals([" "], "<U4")
    for wl, width in [("1d", _chx_vals([9], "int32")), ("2d", _chx_vals([[9]], "int32"))]:
        _chm_fn([t_text, width, fill1], "center", [_chx_arr(0), _chx_arr(1), _chx_cha(2)], layout=f"tcopy:width{wl}")
    # A plain U text first: a short one is copied (plain) and the chararray old after it is refused; a 2-d one refuses
    # the trivial loop first, and the replace computes a plain StringDType array.
    t_new = {"T": enc(["Q"])}
    old1 = _chx_vals(["a"], "<U4")
    for ul, text in [("short", _chx_vals(["ab", "ba"], "<U4")), ("2d", _chx_vals([["ab", "ba"]], "<U4"))]:
        _chm_fn([text, old1, t_new], "replace", [_chx_arr(0), _chx_cha(1), _chx_arr(2)], layout=f"tcopy:U_{ul}:old")
        _chm_fn([text, old1, t_new], "replace", [_chx_arr(0), _chx_arr(2), _chx_cha(1)], layout=f"tcopy:U_{ul}:new")
    # An S chararray beside a StringDType argument: no loop (the dispatcher's TypeError before any copy).
    s_cha = _chx_pool("|S4", "c1", 5)
    for name in ["strip", "find", "startswith"]:
        _chm_member([s_cha, t1], name, [_chx_arr(1)], layout="tcopy:S")
    _chm_member([s_cha, t1], "replace", [_chx_arr(1), _chx_v(b"q")], layout="tcopy:S")
    _chm_member([s_cha, tf], "center", [_chx_v(9), _chx_arr(1)], layout="tcopy:S")


CHM_PREDICATES = ["isalnum", "isalpha", "isdecimal", "isdigit", "islower", "isnumeric", "isspace", "istitle", "isupper"]
# Texts for the predicates: cases, titles, ASCII and Unicode digits (decimal U+0661, digit-only U+00B2, numeric-only
# U+00BD / U+2155), whitespace (ASCII, U+3000, U+1680), an empty text, an embedded NUL, a titlecase letter (U+01C5),
# letters without case (U+4E2D), trailing whitespace the element access strips.
CHM_PRED_U = ["abc", "ABC", "Abc Def", "123", "\u0661\u0662", "\u00b2", "\u00bd", "\u2155", " \t", "", "a1", "A b",
              "\u01c5ab", "Ab  ", "  ", "x\x00y", "\u3000", "\u1680\u2028", "\u00df", "\u4e2d1", "AB  ", "12 ",
              "Ab Cd ", "aBc"]
CHM_PRED_S = [b"abc", b"ABC", b"Abc Def", b"123", b" \t", b"", b"a1", b"Ab  ", b"\xe9", b"x\x00y", b"AB  ", b"12 ",
              b"\x0b\x0c", b"Ab Cd ", b"\xb2", b"aBc"]


def _chm_predicates():
    """numpy.char's nine predicates (numpy.strings' ufuncs, re-exported: np.char.isalpha IS the ufunc) and the chararray
    members of the nine names (C5's fourth batch): `isalpha(self)`, a PLAIN bool result over the stored text (a 0-d
    array for a 0-d chararray: chararray.__array_wrap__ keeps the class only for text, and ignores return_scalar), the
    ufunc's out= / where= / dtype=, and its refusals (isdecimal / isnumeric have no bytes loop; a number has no loop,
    named by its default DType: a one-input ufunc marks no operand weak)."""
    for spec in ["<U4", "|S4"]:
        for layout in CHM_LAYOUTS:
            cha = _chx_pool(spec, layout, 5)
            label = f"{spec}:{layout}"
            for name in CHM_PREDICATES:
                _chm_member([cha], name, [], layout=label)
                _chm_fn([cha], name, [_chx_cha(0)], layout=f"{label}:chararray")
                _chm_fn([cha], name, [_chx_arr(0)], layout=f"{label}:plain")
    # The predicate texts, as members and functions (the stored text decides: 'Ab  ' is not alpha, 'AB  ' is upper).
    for spec, texts in [("<U8", CHM_PRED_U), ("|S8", CHM_PRED_S)]:
        recipe = _chx_vals(texts, spec)
        col = _chx_vals([[t] for t in texts], spec)
        for name in CHM_PREDICATES:
            _chm_member([recipe], name, [], layout=f"{spec}:texts")
            _chm_member([col], name, [], layout=f"{spec}:texts:column")
            _chm_fn([recipe], name, [_chx_cha(0)], layout=f"{spec}:texts:chararray")
    # The ufunc's keywords over a chararray: out= (bool, a same_kind int32, a text chararray out takes 'True' / 'False'
    # cut to its width and comes back as a new chararray view), where= with an out, dtype= (only bool has a loop).
    for spec in ["<U4", "|S4"]:
        base = _chx_pool(spec, "c2", 6)
        size, shape = 12, (3, 4)
        mask = _chx_vals((np.arange(size) % 3 != 0).reshape(shape).tolist(), "bool")
        for name in CHM_PREDICATES:
            if spec[1] == "S" and name in ("isdecimal", "isnumeric"):
                continue
            for odt, ol in [("bool", "bool"), ("int32", "int32")]:
                out = _chx_vals(np.zeros(shape, dtype=odt).tolist(), odt, out=True)
                _chm_fn([base, out], name, [_chx_cha(0)], {"out": _chx_arr(1)}, layout=f"{spec}:out_{ol}")
                _chm_fn([base, out], name, [_chx_cha(0), _chx_arr(1)], layout=f"{spec}:out_{ol}:positional")
            outt = _chx_vals(np.full(shape, "zzzz").tolist(), "<U4", out=True)
            _chm_fn([base, outt], name, [_chx_cha(0)], {"out": _chx_cha(1)}, layout=f"{spec}:out_chararray")
            outz = _chx_vals(np.ones(shape, dtype=bool).tolist(), "bool", out=True)
            _chm_fn([base, outz, mask], name, [_chx_cha(0)], {"out": _chx_arr(1), "where": _chx_arr(2)},
                    layout=f"{spec}:where")
            _chm_fn([base], name, [_chx_cha(0)], {"dtype": {"dt": "bool"}}, layout=f"{spec}:dtype")
            _chm_fn([base], name, [_chx_cha(0)], {"dtype": {"dt": "int64"}}, layout=f"{spec}:dtype_noloop")
            outf = _chx_vals(np.zeros((2, 4), dtype=bool).tolist(), "bool", out=True)
            _chm_fn([base, outf], name, [_chx_cha(0)], {"out": _chx_arr(1)}, layout=f"{spec}:out_shape")
    # StringDType text (the predicates' NA rules are section tprop's: P2.6 milestone 1).
    t_text = {"T": enc(CHM_PRED_U)}
    for name in CHM_PREDICATES:
        _chm_fn([t_text], name, [_chx_arr(0)], layout="T")
    # Python values: a str, bytes, lists, numbers (a number has no loop, named Int64DType / Float64DType).
    for value, vl in [("Abc", "str"), (b"123", "bytes"), (["ab", "AB", "1 "], "list"), ([b"ab", b"12"], "bytes_list"),
                      ([["a", "B"], ["1", " "]], "nested"), (5, "int"), (5.5, "float"), ("", "empty_str")]:
        for name in CHM_PREDICATES:
            _chm_fn([], name, [_chx_v(value)], layout=f"value:{vl}")


CHM_CASE = ["capitalize", "lower", "swapcase", "title", "upper"]
# Texts for the case maps: mixed case and titles, the full mappings that grow (U+00DF, U+FB03, U+0149, U+0130), the final
# sigma, titlecase letters (U+01C5, U+01C6), uncased characters between cased ones, digits, an embedded NUL, trailing
# whitespace a chararray's element access strips.
CHM_CASE_U = ["aBc", "ab cD", "\u00df", "\ufb03x", "\u0149", "\u0391\u03a3", "\u0391\u03a3 \u0392", "\u01c5emal",
              "\u01c6x", "they're bill's", "hello2world", "12a", "x\x00y", "", "  ", "\u0130", "Ab  ", "abc  "]
CHM_CASE_S = [b"aBc", b"ab cD", b"\xe9a", b"they're", b"hello2world", b"12a", b"x\x00y", b"", b"  ", b"Ab  ", b"\xdfX",
              b"MiXeD 9"]
# translate's tables: a str.translate dict (a str value, None deletes, an int names a code point) and a bytes.translate
# table of 256 bytes ('a' -> 'x', 'b' -> 'y'; no trailing NUL, which np.bytes_ would drop).
CHM_TR_U = {97: "XY", 98: None, 99: 90, 0x3a3: "s"}
CHM_TR_S = bytes(range(256)).replace(b"a", b"x").replace(b"b", b"y")


def _chm_case():
    """numpy.char's case maps (capitalize, lower, swapcase, title, upper) and translate, numpy.strings' functions
    re-exported, and the chararray members of the six names (C5's fifth batch). The functions are `np.asarray(a)` and
    _vec_string: a PLAIN result of a's dtype (a chararray operand too), 0-d for a 0-d operand or a Python str. The members
    are NumPy's rule 1, `asarray(upper(self))`: a U result an owning chararray, an S result a chararray view."""
    for spec in ["<U4", "|S4"]:
        table = CHM_TR_U if spec[1] == "U" else CHM_TR_S
        for layout in CHM_LAYOUTS:
            cha = _chx_pool(spec, layout, 7)
            label = f"{spec}:{layout}"
            for name in CHM_CASE:
                _chm_member([cha], name, [], layout=label)
                _chm_fn([cha], name, [_chx_cha(0)], layout=f"{label}:chararray")
                _chm_fn([cha], name, [_chx_arr(0)], layout=f"{label}:plain")
            # ---- translate -> asarray(translate(self, table, deletechars)) ----
            _chm_member([cha], "translate", [_chx_v(table)], layout=label)
            _chm_member([cha], "translate", [], {"table": _chx_v(table)}, layout=f"{label}:kw")
            _chm_fn([cha], "translate", [_chx_cha(0), _chx_v(table)], layout=f"{label}:chararray")
            _chm_fn([cha], "translate", [_chx_arr(0), _chx_v(table)], layout=f"{label}:plain")
            if spec[1] == "S":
                _chm_member([cha], "translate", [_chx_v(table), _chx_v(b"cd")], layout=f"{label}:delete")
                _chm_member([cha], "translate", [_chx_v(None), _chx_v(b"a")], layout=f"{label}:none_table")
            else:
                # NumPy passes deletechars to bytes.translate only: a U chararray ignores it.
                _chm_member([cha], "translate", [_chx_v(table), _chx_v("ab")], layout=f"{label}:delete_ignored")
    # The case texts as members and functions, as a row, a column and at a width that cuts a growing mapping.
    for spec, texts in [("<U8", CHM_CASE_U), ("|S8", CHM_CASE_S), ("<U2", CHM_CASE_U), ("|S2", CHM_CASE_S)]:
        recipe = _chx_vals(texts, spec)
        col = _chx_vals([[t] for t in texts], spec)
        for name in CHM_CASE:
            _chm_member([recipe], name, [], layout=f"{spec}:texts")
            _chm_member([col], name, [], layout=f"{spec}:texts:column")
            _chm_fn([recipe], name, [_chx_cha(0)], layout=f"{spec}:texts:chararray")
        table = CHM_TR_U if spec[1] == "U" else CHM_TR_S
        _chm_member([recipe], "translate", [_chx_v(table)], layout=f"{spec}:texts")
        _chm_fn([recipe], "translate", [_chx_cha(0), _chx_v(table)], layout=f"{spec}:texts:chararray")
    # translate's table kinds and refusals over a chararray: a str table for U, None and a delete set for S, the other
    # class's table (bytes.translate wants a bytes-like table), a 255-byte table, a deletechars str for S.
    u_base = _chx_vals(["abc", "cab ", "\u03a3a", ""], "<U4")
    s_base = _chx_vals([b"abc", b"cab ", b"\xe9a", b""], "|S4")
    for member, args, kw, lab in [
            (True, [_chx_v("zyx")], None, "U:str_table"),
            (True, [_chx_v({97: 0x110000})], None, "U:out_of_range"),
            (True, [_chx_v({97: 1.5})], None, "U:float_value"),
            (True, [_chx_v(None)], None, "U:none_table"),
            (False, [_chx_v(CHM_TR_U)], None, "S:dict_table"),
            (False, [_chx_v(CHM_TR_S[:255])], None, "S:short_table"),
            (False, [_chx_v(CHM_TR_S), _chx_v("bc")], None, "S:str_delete"),
            (False, [], {"table": _chx_v(None), "deletechars": _chx_v(b"ab")}, "S:kw")]:
        base = u_base if member else s_base
        _chm_member([base], "translate", args, kw, layout=lab)
        _chm_fn([base], "translate", [_chx_cha(0)] + args, kw, layout=f"{lab}:fn")
    # StringDType text: the case maps, translate, and translate's deletechars refusal (str.translate takes one argument).
    t_text = {"T": enc(CHM_CASE_U)}
    for name in CHM_CASE:
        _chm_fn([t_text], name, [_chx_arr(0)], layout="T")
    _chm_fn([t_text], "translate", [_chx_arr(0), _chx_v(CHM_TR_U)], layout="T")
    _chm_fn([t_text], "translate", [_chx_arr(0), _chx_v(CHM_TR_U), _chx_v("a")], layout="T:deletechars")
    # Python values: a str, bytes, lists, an empty str, numbers (NumPy's `string operation on non-string array`).
    for value, vl in [("aBc d", "str"), (b"aBc d", "bytes"), (["ab", "Cd e"], "list"), ([b"ab", b"Cd"], "bytes_list"),
                      ([["a b", "C"], ["d", "E f"]], "nested"), ("", "empty_str"), (5, "int"), (5.5, "float")]:
        for name in CHM_CASE:
            _chm_fn([], name, [_chx_v(value)], layout=f"value:{vl}")
        is_bytes = isinstance(value, bytes) or (isinstance(value, list) and value and isinstance(value[0], bytes))
        _chm_fn([], "translate", [_chx_v(value), _chx_v(CHM_TR_S if is_bytes else CHM_TR_U)], layout=f"value:{vl}")


# The % format pools of _chm_mod (the "fmt" recipe of _chx_build): formats that take one value of any kind (str(), repr(),
# ascii(), a width, a precision cut, a literal %, a stored trailing blank — the formats are the STORED text —, a non-ASCII
# character), and formats that take one NUMBER (each integer conversion, the float ones, %c).
CHM_FMT_POOLS = {
    "text": ["%s", "<%s>", "%-3s|", "%r", "%.1s", "%%%s", "%3s", "%a", "x%s  ", "\xe9%s"],
    "num": ["%d", "%x", "%05.1f", "%e", "%+i", "%o", "%#x", "%-4d|", "%g", "%c"],
}
# The str formats of `fmt % chararray` (CPython's str.__mod__ with the chararray as ONE argument object, a mapping).
CHM_MOD_STR_FORMATS = ["%s", "%r", "%a", "%d", "%x", "%f", "%c", "%(k)s", "abc", "%s %s", "%5s|", "%.3s", "%%", "%-30s|",
                       "%*s"]
# The bytes formats of `fmt % chararray` (bytes.__mod__: %s / %b are the chararray's C-order buffer).
CHM_MOD_BYTES_FORMATS = [b"%s", b"%b", b"%r", b"%a", b"%d", b"abc", b"%5s|", b"%.3s", b"%c", b"%f", b"%x", b"%(k)s",
                         b"%s %s"]


def _chm_mod():
    """numpy.char's mod (numpy.strings' function re-exported) and the chararray % operator (C5's sixth batch).
    np.char.mod(a, values) is np.asarray(a) and _vec_string with str.__mod__ / bytes.__mod__ per element: a PLAIN result
    (a chararray operand too). chararray.__mod__(self, i) is asarray(mod(self, i)) — a U result an owning chararray, an S
    result a chararray view — and __rmod__ returns NotImplemented, so with the chararray on the right the left operand
    answers alone: a str / bytes (or an element, a str / bytes too) formats the chararray as its ONE argument (a mapping:
    str() / repr(), int() / float() of a 0-d chararray only, NumPy's IndexError for %(key)s), an ndarray runs
    np.remainder (no text loop: NumPy's TypeError), a Python number / None ends in CPython's TypeError."""
    for spec in ["<U8", "|S8"]:
        word, alien = _chm_words(spec)
        for layout in CHM_LAYOUTS:
            text = {"fmt": [spec, layout, "text", 0]}
            num = {"fmt": [spec, layout, "num", 0]}
            shape = _chx_shape(text)
            nd = len(shape)
            size = int(np.prod(shape)) if shape else 1
            label = f"{spec}:{layout}"
            C = _chx_cha(0)

            def op(recipes, args, sub):
                add_chx(CHM, "cha_op", {"sym": "%", "args": args}, recipes, layout=f"{label}:{sub}")

            # ---- c % v: the text formats with one value of each kind ----
            for v, vl in [(_chx_v("xy"), "str"), (_chx_v(b"xy"), "bytes"), (_chx_v(5), "int"), (_chx_v(1.5), "float"),
                          (_chx_v(None), "none"), (_chx_v(True), "bool"), (_chx_v(word("q\xe9")), "word"),
                          (_chx_v(alien("q")), "alien"), (_chx_v({"k": 1}), "dict")]:
                op([text], [C, v], f"text:{vl}")
            # ---- c % v: the number formats ----
            for v, vl in [(_chx_v(5), "int"), (_chx_v(255), "int255"), (_chx_v(-3), "neg"), (_chx_v(2.5), "float"),
                          (_chx_v(True), "bool"), (_chx_v("x"), "str"), (_chx_v(None), "none")]:
                op([num], [C, v], f"num:{vl}")
            # ---- c % arrays: the values element by element (same shape, a row broadcast, a StringDType value) ----
            if nd >= 1 and size > 0:
                flat = [str(i) for i in range(size)]
                vals = np.array(flat).reshape(shape).tolist()
                ints = (np.arange(size) * 7 - 3).reshape(shape).tolist()
                floats = (np.arange(size) * 1.25 - 2).reshape(shape).tolist()
                row = [f"r{i}" for i in range(shape[-1])]
                op([text, _chx_vals(vals, "<U3")], [C, _chx_arr(1)], "text:plain")
                op([text, _chx_vals(vals, "<U3")], [C, _chx_cha(1)], "text:chararray")
                op([text, _chx_vals(row, "<U3")], [C, _chx_arr(1)], "text:row")
                op([text], [C, _chx_v(row)], "text:list")
                op([num, _chx_vals(ints, "int64")], [C, _chx_arr(1)], "num:ints")
                op([num, _chx_vals(floats, "float64")], [C, _chx_arr(1)], "num:floats")
            op([text, {"T": enc(["t\xe9"])}], [C, _chx_arr(1)], "text:stringdtype")
            # ---- the chararray on the RIGHT: Python's str / bytes formatting, an element, numbers, None, arrays ----
            if layout in ("c1", "c2", "f2", "z0", "e1", "one", "bc", "d5", "r2", "s1"):
                for fmt in CHM_MOD_STR_FORMATS:
                    op([text], [_chx_v(fmt), C], f"R:str:{fmt}")
                for fmt in CHM_MOD_BYTES_FORMATS:
                    op([text], [_chx_v(fmt), C], f"R:bytes:{fmt.decode()}")
                for v, vl in [(_chx_v(5), "int"), (_chx_v(1.5), "float"), (_chx_v(True), "bool"), (_chx_v(None), "none")]:
                    op([text], [v, C], f"R:{vl}")
                op([text, _chx_vals(["%s"] * max(size, 1) if not shape else np.full(shape, "%s").tolist(), "<U2")],
                   [_chx_arr(1), C], "R:plain_text")
                op([text, _chx_vals([1] * 3, "int64")], [_chx_arr(1), C], "R:plain_int")
                op([text, {"T": enc(["%s"])}], [_chx_arr(1), C], "R:stringdtype")
                if nd >= 1 and size > 0:
                    elem = {"elem": [0, [0] * nd]}
                    op([text], [elem, C], "R:elem")
                    for v, vl in [(_chx_v(5), "int"), (_chx_v(word("v")), "word"), (_chx_v(None), "none")]:
                        op([text], [elem, v], f"elem:{vl}")
    # ---- int() / float() / __index__ of a 0-d chararray as str / bytes formatting's ONE argument ----
    for spec, values in [("<U32", ["7", " 7 ", "x", "1.5", "1_000", "123456789012345678901234567890", "١٢", ""]),
                         ("|S8", [b"7", b"1.5", b"x", b" 9 "])]:
        for value in values:
            zero_d = _chx_vals(value, spec)
            for fmt in ["%d", "%f", "%x", "%c", "%s", "%r", "%i", "%e"]:
                add_chx(CHM, "cha_op", {"sym": "%", "args": [_chx_v(fmt), _chx_cha(0)]}, [zero_d],
                        layout=f"0d:{spec}:str:{fmt}")
            for fmt in [b"%d", b"%f", b"%s", b"%c", b"%x"]:
                add_chx(CHM, "cha_op", {"sym": "%", "args": [_chx_v(fmt), _chx_cha(0)]}, [zero_d],
                        layout=f"0d:{spec}:bytes:{fmt.decode()}")
    # ---- an element format over a plain 0-d / 1-d array argument (int(a), float(a), __index__; bytes formatting's
    # buffer, its %c byte and its float conversion's own TypeError; a StringDType array has no buffer) ----
    for spec, formats in [("<U2", ["%x", "%d", "%f", "%c", "%s", "%r"]), ("|S2", [b"%x", b"%d", b"%f", b"%c", b"%s", b"%r"])]:
        elems = _chx_vals(formats, spec)
        for arg, al in [(_chx_vals(255, "int64"), "i255"), (_chx_vals(-3, "int64"), "ineg"),
                        (_chx_vals(5.7, "float64"), "f"), (_chx_vals(True, "bool"), "b"), (_chx_vals(65, "uint8"), "u8"),
                        (_chx_vals([5], "int64"), "1d"), (_chx_vals(float("nan"), "float64"), "nan"),
                        (_chx_vals(float("inf"), "float64"), "inf"), (_chx_vals("7", "<U2"), "u7"),
                        (_chx_vals("x", "<U2"), "ux"), (_chx_vals(b"1.5", "|S3"), "s15"), ({"T": enc(["9"])}, "t1d")]:
            for i in range(len(formats)):
                add_chx(CHM, "cha_op", {"sym": "%", "args": [{"elem": [0, [i]]}, _chx_arr(1)]}, [elems, arg],
                        layout=f"elem_arg:{spec}:{al}:{i}")
    # ---- an element format over a Python value: the value is the Python object itself (a Python int has no buffer for
    # bytes formatting's %s, its repr is not a NumPy scalar's) ----
    for spec, formats in [("<U2", ["%x", "%d", "%f", "%c", "%s", "%r"]), ("|S2", [b"%x", b"%d", b"%f", b"%c", b"%s", b"%r"])]:
        elems = _chx_vals(formats, spec)
        for v, vl in [(_chx_v(5), "int"), (_chx_v(65), "int65"), (_chx_v(2.5), "float"), (_chx_v(True), "bool"),
                      (_chx_v(None), "none"), (_chx_v("q"), "str"), (_chx_v(b"q"), "bytes")]:
            for i in range(len(formats)):
                add_chx(CHM, "cha_op", {"sym": "%", "args": [{"elem": [0, [i]]}, v]}, [elems],
                        layout=f"elem_value:{spec}:{vl}:{i}")
    # ---- np.char.mod: chararray and plain operands (a PLAIN result), Python values, StringDType ----
    for spec in ["<U8", "|S8"]:
        for layout in ["c1", "c2", "f2", "z0", "e2", "bc", "r2", "d5"]:
            text = {"fmt": [spec, layout, "text", 2]}
            num = {"fmt": [spec, layout, "num", 2]}
            label = f"{spec}:{layout}"
            _chm_fn([text], "mod", [_chx_cha(0), _chx_v("v")], layout=f"{label}:chararray")
            _chm_fn([text], "mod", [_chx_arr(0), _chx_v("v")], layout=f"{label}:plain")
            _chm_fn([text], "mod", [_chx_cha(0)], {"values": _chx_v(None)}, layout=f"{label}:kw")
            _chm_fn([num], "mod", [_chx_cha(0), _chx_v(7)], layout=f"{label}:num")
            _chm_fn([num], "mod", [_chx_arr(0), _chx_v(2.5)], layout=f"{label}:num_float")
    for value, vl, values in [("a%sb", "str", "X"), (b"%d", "bytes", 5), (["%d", "%x"], "list", 255),
                              ([b"%s", b"%r"], "bytes_list", b"q"), ([["%s", "%d"], ["%x", "%o"]], "nested", 9),
                              ("", "empty", "x"), ("%s", "none", None), ("%(k)s", "dict", {"k": "v"}), (5, "int", "x"),
                              (5.5, "float", 1), ("%d", "str_value", "x")]:
        _chm_fn([], "mod", [_chx_v(value), _chx_v(values)], layout=f"value:{vl}")
    t_formats = {"T": enc(["%s!", "%r", "<%5s>", "%d"])}
    _chm_fn([t_formats], "mod", [_chx_arr(0), _chx_v(5)], layout="T")
    _chm_fn([t_formats], "mod", [_chx_arr(0), _chx_v("x")], layout="T:str")


def gen_strcharmeth():
    """Section strcharmeth: numpy.char's string functions and the chararray string members (the numpy.char plan's C4 /
    C5): partition / rpartition, the padding functions, str_len, the search functions, the strips and replace,
    NumPy's trivial-loop copy of a chararray beside a StringDType argument, the nine predicates, the case maps with
    translate, and mod with the chararray % operator."""
    _chm_transform()
    _chm_search()
    _chm_strip()
    _chm_tcopy()
    _chm_predicates()
    _chm_case()
    _chm_mod()
    _chx_flush()

# ----------------------------------------------------------------------------------------------------------------------
# Section qfamily: the quantile family's keywords over every dtype (the validation of strings phase 2, 2026-10-16)
# ----------------------------------------------------------------------------------------------------------------------

QFAM = "qfamily"
# The numeric dtypes the weighted branch is gated on, then the text kinds (fixed-width and StringDType with each NA kind
# the branch treats differently: none, NaN-like, a string).
QF_NUM = ["float64", "float32", "float16", "int8", "uint8", "int32", "int64", "uint64", "bool"]
QF_TEXT = ["<U3", "|S3", "T", "Tnan", "Tstr"]
QF_KINDS = QF_NUM + QF_TEXT
QF_FUNCS = ("quantile", "percentile", "nanquantile", "nanpercentile")
QF_TSPEC = {"T": TD, "Tnan": TDNAN, "Tstr": TDS}


def _qf_num(dtype, layout, seed):
    """A numeric data operand of ``LAYOUTS[layout]``: DISTINCT values (bool aside), with no zero of either sign.

    NumPy's default argsort leaves tied values in an order its dispatched sort decides (introsort for 8- and 16-bit types,
    x86-simd-sort on AVX2 / AVX-512 hosts for 32- and 64-bit ones), NumSharp's stable one in input order; the weighted
    branch adds the weights of a tied run in that order, so a run of ties with fractional weights could round differently.
    Distinct values have one sorted order, and their weights may be any floats.

    :param dtype: the numeric dtype name.
    :param layout: a ``LAYOUTS`` name.
    :param seed: the permutation's seed.
    :returns: the (base, view) pair.
    """
    n, build = LAYOUTS[layout]
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    if dt.kind == "b":
        base = rng.integers(0, 2, n).astype(bool)
    else:
        perm = rng.permutation(n)
        if dt.kind == "f":
            vals = perm * 1.25 - n * 0.5 + 0.375          # multiples of 1/8: exact in float16, never 0
        elif dt.kind == "u":
            vals = perm * 3 + 1
        else:
            vals = perm * 3 - n
        base = np.asarray(vals).astype(dt)
    return base, build(base)


def _qf_data(kind, layout, seed):
    """A data operand of ``kind`` (``QF_KINDS``) and ``layout``.

    :param kind: a numeric dtype name, ``<U3`` / ``|S3``, or ``T`` / ``Tnan`` / ``Tstr``.
    :param layout: a ``LAYOUTS`` name.
    :param seed: the values' seed.
    :returns: the (base, view) pair.
    """
    if kind in QF_NUM:
        return _qf_num(kind, layout, seed)
    if kind in QF_TSPEC:
        return t_operand(QF_TSPEC[kind], layout, seed)
    return operand(kind, layout, seed)


def _qf_weights(kind, shape, seed, integral=None):
    """A weights operand of ``shape``: fractional positive floats (their sums round), or small positive integers for bool
    data (its two values tie in every run).

    :param kind: the data kind.
    :param shape: the weights' shape.
    :param seed: the values' seed.
    :param integral: force (True) or refuse (False) integer weights; None decides by the data kind.
    :returns: the (base, view) pair.
    """
    rng = np.random.default_rng(seed + 7919)
    if integral is None:
        integral = kind == "bool"
    if integral:
        w = rng.integers(1, 5, shape).astype(np.float64)
    else:
        w = np.round(rng.uniform(0.05, 3.0, shape), 3)
    return plain(w)


def _qf_q(fn, values):
    """q values scaled for ``fn`` (a percentile multiplies them by 100).

    :param fn: the function name.
    :param values: the quantiles in [0, 1].
    :returns: the scaled list.
    """
    return [v * 100 for v in values] if fn.endswith("percentile") else list(values)


def gen_qfamily():
    """Section qfamily: the quantile family's ``weights=`` keyword (NumPy 2.4.2, method ``inverted_cdf`` only) over every
    dtype — numbers and text — through ``np.quantile`` / ``np.percentile`` / ``np.nanquantile`` / ``np.nanpercentile``
    (op ``quant_kw`` with a ``w_op`` operand). Every case is phase 1: NumSharp computes them.

    * the core matrix: 14 data kinds (9 numeric dtypes; ``U``, ``S``; StringDType without an NA, with a NaN-like NA,
      with a string NA) x 7 layouts x the 4 functions x every axis form (None, each axis, -1, every axis in order and
      reversed) x weights of the data's shape and of the reduced axes' shape (the nan forms' NumPy 2.4.2 IndexError for
      the latter over a kept axis longer than 1 included);
    * q forms: a Python float, a list, a Python int, float64 / float32 / float16 / int / bool / 0-d / 2-d / 3-d arrays;
    * weight dtypes (int, uint8, bool, float16, float32, complex, text) and values (negative, NaN, inf, all zero,
      leading zeros, a zero on the picked element, an overflowing sum);
    * NumPy's error order (q, method, axis, the weights' shape, their sign, q's rank), out= (shape, keepdims, another
      dtype, text parsed into numbers, the nan forms' unchecked per-position store), empty data, 0-d data, ``axis=()``,
      NaN data and all-NaN lines in both families, and text with many ties under fractional weights (NumPy's introsort
      order of equal strings decides the cumulative sum's rounding).
    """
    # --- The core matrix.
    for ki, kind in enumerate(QF_KINDS):
        for li, layout in enumerate(["c1", "c2", "f2", "t3", "r2", "col", "s2"]):
            seed = 101 * ki + 13 * li
            data = _qf_data(kind, layout, seed)
            v = data[1]
            nd = v.ndim
            axes = [None] + list(range(nd))
            if nd > 1:
                axes += [-1, tuple(range(nd)), tuple(reversed(range(nd)))]
            for fn in QF_FUNCS:
                q = _qf_q(fn, [0.0, 0.3, 0.5, 0.71, 1.0])
                for ax in axes:
                    forms = ["full"] if ax is None else ["full", "reduced"]
                    for form in forms:
                        if form == "full":
                            wshape = v.shape
                        else:
                            axt = ax if isinstance(ax, tuple) else (ax,)
                            wshape = tuple(v.shape[a] for a in axt)
                        w = _qf_weights(kind, wshape, seed + len(wshape))
                        _quant(f"{kind}-{layout}-{form}", data, name=fn, q=q, qkind="fl", axis=ax,
                               method="inverted_cdf", weights=w, section=QFAM)

    # --- q forms, keepdims.
    for ki, kind in enumerate(QF_KINDS):
        data = _qf_data(kind, "c2", 7 + ki)
        v = data[1]
        w = _qf_weights(kind, v.shape, 7 + ki)
        for fn in QF_FUNCS:
            pct = fn.endswith("percentile")
            s = 100 if pct else 1
            forms = [("f", 0.5 * s), ("f", 0.0), ("f", 1.0 * s), ("fl", [0.1 * s, 0.9 * s]), ("i", 1 * s), ("i", 0),
                     ("arr", plain(np.array([0.1, 0.4, 0.9]) * s)),
                     ("arr", plain((np.array([0.1, 0.4, 0.9]) * s).astype(np.float32))),
                     ("arr", plain((np.array([0.1, 0.4, 0.9]) * s).astype(np.float16))),
                     ("arr", plain(np.array([0, 1]) * s)),
                     ("arr", plain(np.array(0.3 * s))),
                     ("arr", plain(np.array([[0.2, 0.4], [0.6, 0.8]]) * s)),
                     ("arr", plain(np.zeros((1, 1, 1))))]
            if not pct:
                forms.append(("arr", plain(np.array([True, False]))))
            for qi, (qk, qv) in enumerate(forms):
                for ax in (None, 0, 1):
                    wq = w if ax is None else _qf_weights(kind, (v.shape[ax],), 3 + qi)
                    _quant(f"{kind}-q{qi}", data, name=fn, q=qv, qkind=qk, axis=ax, method="inverted_cdf", weights=wq,
                           section=QFAM)
                _quant(f"{kind}-q{qi}-kd", data, name=fn, q=qv, qkind=qk, axis=1, method="inverted_cdf",
                       weights=_qf_weights(kind, (v.shape[1],), 5 + qi), keepdims=True, section=QFAM)
                _quant(f"{kind}-q{qi}-kdN", data, name=fn, q=qv, qkind=qk, axis=None, method="inverted_cdf",
                       weights=w, keepdims=True, section=QFAM)

    # --- Weight dtypes and values (float64 data, every function).
    data = _qf_data("float64", "c2", 3)
    v = data[1]
    rng = np.random.default_rng(5)
    wdts = {
        "i8": plain(rng.integers(1, 9, v.shape).astype(np.int64)),
        "i1": plain(rng.integers(1, 9, v.shape).astype(np.int8)),
        "u1": plain(rng.integers(1, 9, v.shape).astype(np.uint8)),
        "b": plain(np.array([[True, False, True, True], [True, True, False, True], [False, True, True, True]])),
        "f2": plain(np.round(rng.uniform(0.1, 3, v.shape), 2).astype(np.float16)),
        "f4": plain(np.round(rng.uniform(0.1, 3, v.shape), 2).astype(np.float32)),
        "c16": plain((rng.integers(1, 9, v.shape) + 1j * rng.integers(-3, 3, v.shape)).astype(np.complex128)),
        "U": plain(np.array([["a", "b", "c", "d"]] * 3)),
        "T": t_plain(np.array([["a", "b", "c", "d"]] * 3, dtype=TD.build())),
    }
    for fn in QF_FUNCS:
        q = _qf_q(fn, [0.0, 0.25, 0.5, 1.0])
        for key, w in wdts.items():
            _quant(f"w{key}", data, name=fn, q=q, qkind="fl", axis=None, method="inverted_cdf", weights=w, section=QFAM)
            _quant(f"w{key}-ax1", data, name=fn, q=q, qkind="fl", axis=1, method="inverted_cdf", weights=w,
                   section=QFAM)
        base_w = np.round(rng.uniform(0.5, 2.0, v.shape), 2)
        values = {
            "neg": lambda b: b.__setitem__((1, 2), -0.5),
            "negzero": lambda b: b.__setitem__((1, 2), -0.0),
            "nan": lambda b: b.__setitem__((2, 1), np.nan),
            "inf": lambda b: b.__setitem__((0, 0), np.inf),
            "zerorow": lambda b: b.__setitem__((1,), 0.0),
            "allzero": lambda b: b.__setitem__((slice(None),), 0.0),
            "lead0": lambda b: b.__setitem__((slice(None), 0), 0.0),
            "huge": lambda b: b.__setitem__((slice(None),), 1e308),
        }
        for key, mutate in values.items():
            wv = base_w.copy()
            mutate(wv)
            for ax in (None, 1):
                _quant(f"v{key}", data, name=fn, q=q, qkind="fl", axis=ax, method="inverted_cdf", weights=plain(wv),
                       section=QFAM)
        # A zero weight on the element q = 0 would pick, and a q that lands exactly on a cdf step.
        _quant("step", plain(np.array([1.0, 2.0, 3.0, 4.0])), name=fn, q=_qf_q(fn, [0.0, 0.25, 0.5, 0.75, 1.0]),
               qkind="fl", method="inverted_cdf", weights=plain(np.array([0.0, 1.0, 1.0, 2.0])), section=QFAM)
        _quant("step-f4", plain(np.array([1.0, 2.0, 3.0, 4.0])),
               name=fn, q=plain((np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * (100 if fn.endswith("percentile") else 1))
                                .astype(np.float32)),
               qkind="arr", method="inverted_cdf", weights=plain(np.array([0.1, 0.2, 0.3, 0.4])), section=QFAM)
        # A float q searches the cdf cast to ITS dtype: equal weights put cdf steps at k/10, which float32 / float16 round
        # onto (or past) the q values themselves (probed: np.quantile(arange(10.), np.float32([0.3]), weights=ones) is 2.0,
        # 3.0 with a float64 q of the same value); a percentile's q is divided by 100 in its dtype first.
        for qdt in (np.float64, np.float32, np.float16):
            qv = np.arange(1, 10) / 10 * (100 if fn.endswith("percentile") else 1)
            _quant(f"step-tenths-{np.dtype(qdt).name}", plain(np.arange(10.0)), name=fn,
                   q=plain(qv.astype(qdt)), qkind="arr", method="inverted_cdf", weights=plain(np.ones(10)),
                   section=QFAM)
            _quant(f"step-thirds-{np.dtype(qdt).name}", plain(np.arange(9.0)), name=fn,
                   q=plain((np.arange(0, 10) / 9 * (100 if fn.endswith("percentile") else 1)).astype(qdt)),
                   qkind="arr", method="inverted_cdf", weights=plain(np.full(9, 1.0 / 3)), section=QFAM)
        # One line's search at its boundaries, for every q dtype: leading zero weights put the sorted cdf at
        # [0, 0, .25, .5, 1] (NumPy's -1 rule takes the zeros, so q = 0 skips them; .25 and .5 are hit exactly), and a
        # first weight that divides to 0.0 beside huge ones (5e-324 / 1e300) falls under the -1 rule too although it is not
        # zero. NumSharp divides only the entries its search probes when the search compares in float64 (float64, integer
        # and bool q) and runs NumPy's statements otherwise (float16 / float32 q), so every q dtype is here.
        pc = 100 if fn.endswith("percentile") else 1
        lead = plain(np.array([30.0, 10.0, 50.0, 20.0, 40.0]))
        lead_w = plain(np.array([1.0, 0.0, 2.0, 0.0, 1.0]))
        for qdt in (np.float64, np.float32, np.float16):
            _quant(f"lead0-{np.dtype(qdt).name}", lead, name=fn,
                   q=plain((np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * pc).astype(qdt)), qkind="arr",
                   method="inverted_cdf", weights=lead_w, section=QFAM)
        lead_q = {"i8": np.array([0, 1]) * pc, "u8": (np.array([0, 1]) * pc).astype(np.uint64), "0d": np.array(0)}
        if pc == 1:
            # A bool q of a percentile is divided by 100 first (a float q); the quantile searches it as bool.
            lead_q["b"] = np.array([False, True])
        for key, qv in lead_q.items():
            _quant(f"lead0-q{key}", lead, name=fn, q=plain(qv), qkind="arr", method="inverted_cdf", weights=lead_w,
                   section=QFAM)
        _quant("underflow", plain(np.array([1.0, 2.0, 3.0])), name=fn, q=_qf_q(fn, [0.0, 1e-310, 0.5, 1.0]),
               qkind="fl", method="inverted_cdf", weights=plain(np.array([5e-324, 1e300, 1e300])), section=QFAM)

    # --- NumPy's order of checks.
    a1 = plain(np.array([3.0, 1.0, 2.0, 5.0, 4.0]))
    w1 = plain(np.array([1.0, 2.0, 1.0, 1.0, 1.0]))
    m = plain(np.arange(6.0).reshape(2, 3))
    for fn in QF_FUNCS:
        s = 100 if fn.endswith("percentile") else 1
        _quant("chk-linear", a1, name=fn, q=0.5 * s, method="linear", weights=w1, section=QFAM)
        _quant("chk-unknown", a1, name=fn, q=0.5 * s, method="nope", weights=w1, section=QFAM)
        _quant("chk-q-range", a1, name=fn, q=1.5 * s, method="nope", weights=plain(-np.array([1.0] * 5)), section=QFAM)
        _quant("chk-axis", a1, name=fn, q=0.5 * s, axis=3, method="inverted_cdf", weights=w1, section=QFAM)
        _quant("chk-axis-linear", a1, name=fn, q=0.5 * s, axis=3, method="linear", weights=w1, section=QFAM)
        _quant("chk-axis-shape", a1, name=fn, q=0.5 * s, axis=4, method="inverted_cdf",
               weights=plain(np.array([1.0, 2.0])), section=QFAM)
        _quant("chk-shape-none", a1, name=fn, q=0.5 * s, method="inverted_cdf", weights=plain(np.array([1.0, 2.0])),
               section=QFAM)
        _quant("chk-shape-axis", a1, name=fn, q=0.5 * s, axis=0, method="inverted_cdf",
               weights=plain(-np.array([1.0, 2.0])), section=QFAM)
        _quant("chk-shape-2d", a1, name=fn, q=0.5 * s, axis=0, method="inverted_cdf",
               weights=plain(np.ones((1, 5))), section=QFAM)
        _quant("chk-0d-w", a1, name=fn, q=0.5 * s, method="inverted_cdf", weights=plain(np.array(1.0)), section=QFAM)
        _quant("chk-q3d-neg", a1, name=fn, q=plain(np.zeros((1, 1, 1))), qkind="arr", method="inverted_cdf",
               weights=plain(-np.ones(5)), section=QFAM)
        _quant("chk-q3d-axis", a1, name=fn, q=plain(np.zeros((1, 1, 1))), qkind="arr", axis=3, method="inverted_cdf",
               weights=w1, section=QFAM)
        _quant("chk-axes-oob", m, name=fn, q=0.5 * s, axis=(0, 5), method="inverted_cdf", weights=plain(np.ones((2, 3))),
               section=QFAM)
        _quant("chk-axes-rep", m, name=fn, q=0.5 * s, axis=(1, -1), method="inverted_cdf",
               weights=plain(np.ones((2, 3))), section=QFAM)
        _quant("chk-axes-rep-plain", m, name=fn, q=0.5 * s, axis=(0, 0), method="inverted_cdf", section=QFAM)
        _quant("chk-complex", plain(np.array([1 + 1j, 2 + 0j])), name=fn, q=0.5 * s, method="linear",
               weights=plain(np.ones(2)), section=QFAM)
        _quant("chk-wtext-shape", a1, name=fn, q=0.5 * s, method="inverted_cdf",
               weights=plain(np.array(["a", "b", "c"])), section=QFAM)

    # --- out=.
    m23 = plain(np.array([[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]]))
    w3 = plain(np.array([1.0, 2.0, 3.0]))
    wf = plain(np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]))
    for fn in QF_FUNCS:
        s = 100 if fn.endswith("percentile") else 1
        outs = {
            "ok": (target(np.zeros(2)), 1, False, w3),
            "okf": (target(np.zeros(2)), 1, False, wf),
            "wrong": (target(np.zeros(3)), 1, False, wf),
            "small": (target(np.zeros(1)), 1, False, wf),
            "wide": (target(np.zeros((3, 2))), 1, False, wf),
            "int": (target(np.zeros(2, np.int64)), 1, False, wf),
            "f4": (target(np.zeros(2, np.float32)), 1, False, wf),
            "u5": (target(np.zeros(2, "<U5")), 1, False, wf),
            "kd": (target(np.zeros((2, 1))), 1, True, wf),
            "kdbad": (target(np.zeros(2)), 1, True, wf),
            "kdN": (target(np.zeros((1, 1))), None, True, wf),
            "none0d": (target(np.zeros(())), None, False, wf),
            "strided": (plain(np.zeros((1, 4)))[0], 1, False, wf),
        }
        for key, (o, ax, kd, w) in outs.items():
            if key == "strided":
                base = np.zeros(4)
                o = (base, base[::2])
            _quant(f"out-{key}", m23, name=fn, q=0.5 * s, axis=ax, method="inverted_cdf", keepdims=kd, out=o,
                   weights=w, section=QFAM)
        # A list q into a wrong-length out: the flat nan path stores by broadcast.
        _quant("out-flat-wrong", a1, name=fn, q=[0.5 * s], qkind="fl", method="inverted_cdf",
               out=target(np.zeros(3)), weights=w1, section=QFAM)
        _quant("out-flat-bad", a1, name=fn, q=[0.5 * s, 0.6 * s], qkind="fl", method="inverted_cdf",
               out=target(np.zeros(3)), weights=w1, section=QFAM)
        # Text data into a numeric out: parsed (NumPy's setitem), or refused with the parse's text.
        _quant("out-text-num", plain(np.array(["10", "2", "33"])), name=fn, q=[0.5 * s], qkind="fl",
               method="inverted_cdf", out=target(np.zeros(1)), weights=plain(np.ones(3)), section=QFAM)
        _quant("out-text-bad", plain(np.array(["b", "a", "c"])), name=fn, q=[0.5 * s, 1.0 * s], qkind="fl",
               method="inverted_cdf", out=target(np.zeros(2)), weights=plain(np.ones(3)), section=QFAM)
        _quant("out-text-bad0d", plain(np.array(["b", "a", "c"])), name=fn, q=0.5 * s,
               method="inverted_cdf", out=target(np.zeros(())), weights=plain(np.ones(3)), section=QFAM)
        _quant("out-nan-int", plain(np.array([[1.0, np.nan], [2.0, 3.0]])), name=fn, q=0.5 * s, axis=1,
               method="inverted_cdf", out=target(np.zeros(2, np.int64)), weights=plain(np.ones((2, 2))), section=QFAM)

    # --- Empty, 0-d and axis=().
    for fn in QF_FUNCS:
        s = 100 if fn.endswith("percentile") else 1
        for key, (arr, ax, wshape) in {
            "e1": (np.zeros(0), None, (0,)),
            "e2-ax1": (np.zeros((2, 0)), 1, (2, 0)),
            "e2-ax0": (np.zeros((2, 0)), 0, (2, 0)),
            "e2-ax1r": (np.zeros((2, 0)), 1, (0,)),
            "e3-axes": (np.zeros((0, 2, 3)), (1, 2), (0, 2, 3)),
            "z0": (np.array(3.0), None, ()),
            "z0-ax0": (np.array(3.0), 0, ()),
            "noaxes-0d": (np.arange(6.0).reshape(2, 3), (), ()),
            "noaxes-full": (np.arange(6.0).reshape(2, 3), (), (2, 3)),
            "noaxes-1d": (np.arange(6.0).reshape(2, 3), (), (3,)),
        }.items():
            w = plain(np.ones(wshape)) if wshape else plain(np.array(2.0))
            _quant(f"empty-{key}", plain(arr), name=fn, q=0.5 * s, axis=ax, method="inverted_cdf", weights=w,
                   section=QFAM)
            _quant(f"empty-{key}-kd", plain(arr), name=fn, q=[0.5 * s], qkind="fl", axis=ax, method="inverted_cdf",
                   keepdims=True, weights=w, section=QFAM)

    # --- NaN data in both families, all-NaN lines, NaN-like StringDType NAs, None-like NAs.
    nan_data = {
        "f8": np.array([[3.0, np.nan, 2.0, 5.0], [np.nan, np.nan, np.nan, np.nan], [1.0, 4.0, np.nan, 0.5]]),
        "f4": np.array([[3.0, np.nan, 2.0, 5.0], [7.0, 1.0, 6.0, 2.5], [1.0, 4.0, np.nan, 0.5]], np.float32),
        "f2": np.array([[np.nan, np.nan], [2.0, 1.0]], np.float16),
        "allnan": np.full((2, 3), np.nan),
        "T": np.array([["b", np.nan, "a", "zz"], [np.nan, np.nan, np.nan, np.nan], ["c", "d" * 20, np.nan, ""]],
                      dtype=TDNAN.build()),
        "Tall": np.array([np.nan, np.nan], dtype=TDNAN.build()),
    }
    for fn in QF_FUNCS:
        s = 100 if fn.endswith("percentile") else 1
        for key, arr in nan_data.items():
            op = t_plain(arr) if arr.dtype.kind == "T" else plain(arr)
            for ax in (None, 0, -1) if arr.ndim > 1 else (None, 0):
                wshape = arr.shape
                rng = np.random.default_rng(len(key) * 31 + (0 if ax is None else ax + 5))
                w = plain(np.round(rng.uniform(0.2, 2.0, wshape), 2))
                _quant(f"nan-{key}", op, name=fn, q=0.5 * s, axis=ax, method="inverted_cdf", weights=w, section=QFAM)
                _quant(f"nan-{key}-q", op, name=fn, q=[0.0, 0.4 * s, 1.0 * s], qkind="fl", axis=ax,
                       method="inverted_cdf", weights=w, section=QFAM)
                _quant(f"nan-{key}-kd", op, name=fn, q=0.5 * s, axis=ax, method="inverted_cdf", keepdims=True,
                       weights=w, section=QFAM)
        tnone = t_plain(np.array(["b", None, "c", "a"], dtype=TDN.build()))
        _quant("none-na", tnone, name=fn, q=0.5 * s, method="inverted_cdf", weights=plain(np.ones(4)), section=QFAM)
        _quant("none-na-1", t_plain(np.array([None], dtype=TDN.build())), name=fn, q=0.5 * s, method="inverted_cdf",
               weights=plain(np.ones(1)), section=QFAM)

    # --- Text with many ties under fractional weights: NumPy's introsort order of equal strings (lines longer than 16)
    # decides the order the cumulative sum adds their weights in.
    rng = np.random.default_rng(77)
    for ti, (kind, n) in enumerate([("<U3", 24), ("|S3", 40), ("T", 33), ("Tnan", 50), ("<U3", 200)]):
        pool = ["a", "b", "a\x00", "c", "bb", "a\x00b"]
        if kind in QF_TSPEC:
            # StringDType compares with NpyString_cmp (strncmp: a NUL both texts share ends the comparison), so texts of
            # one size that differ only past a shared NUL are EQUAL and the sort's order of them decides which bytes the
            # answer holds (the NaN removal's order of the nan forms too).
            pool = ["a", "b", "a\x00b", "a\x00c", "a\x00d", "bb", "c"]
        vals = [pool[i] for i in rng.integers(0, len(pool), n)]
        if kind == "Tnan":
            # NaN-like NAs among the ties: the nan forms remove them in _remove_nan_1d's order, which reorders the texts
            # (and their weights) the sort then meets.
            for i in rng.integers(0, n, n // 5):
                vals[i] = np.nan
        if kind == "|S3":
            arr = np.array([x.encode() for x in vals], dtype="|S3")
        elif kind in QF_TSPEC:
            arr = np.array(vals, dtype=QF_TSPEC[kind].build())
        else:
            arr = np.array(vals, dtype=kind)
        op = t_plain(arr) if arr.dtype.kind == "T" else plain(arr)
        w = plain(np.round(rng.uniform(0.01, 1.0, n), 3))
        qs = np.round(rng.uniform(0, 1, 9), 4).tolist() + [0.0, 1.0]
        for fn in ("quantile", "nanquantile"):
            _quant(f"ties{ti}", op, name=fn, q=qs, qkind="fl", method="inverted_cdf", weights=w, section=QFAM)


# ----------------------------------------------------------------------------------------------------------------------
# Section qprec: the quantile family's index arithmetic in q's dtype, its out= routes, an integer q's integer weight and
# empty data (the validation of strings phase 2, 2026-10-16)
# ----------------------------------------------------------------------------------------------------------------------

QPREC = "qprec"
QP_METHODS = ["inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen",
              "weibull", "linear", "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint", "nearest"]
QP_NUM = ["float64", "float32", "float16", "int8", "uint8", "int16", "uint16", "int32", "int64", "uint64", "bool"]
QP_FUNCS = ("quantile", "percentile", "nanquantile", "nanpercentile")


def _qp_values(dtype, n, seed, nan_every=0):
    """``n`` values of ``dtype`` that make the lerp's rounding visible: floats with full mantissas (in float16's range for
    float16), integers over their whole range (the wrapping ``b - a`` and the int64 / uint64 rounding to float64 included),
    random bools; every ``nan_every``-th float replaced by NaN.

    :param dtype: the dtype name.
    :param n: how many values.
    :param seed: the seed.
    :param nan_every: put a NaN at every multiple of it (floats only; 0 for none).
    :returns: a 1-D array.
    """
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    if dt.kind == "b":
        vals = rng.integers(0, 2, n).astype(bool)
    elif dt.kind == "f":
        vals = rng.uniform(-60.0, 60.0, n).astype(dt)
        if nan_every:
            vals[::nan_every] = np.nan
    else:
        info = np.iinfo(dt)
        vals = rng.integers(int(info.min), int(info.max), n, dtype=dt, endpoint=True)
        vals[0] = info.min
        vals[-1] = info.max
    return vals


def _qp_q(fn, values, dtype):
    """A q ARRAY of ``dtype`` (float16 / float32 / float64, or an integer / bool dtype) holding ``values`` (scaled by 100
    for a percentile, before the conversion: NumPy divides in q's dtype).

    :param fn: the function name.
    :param values: the quantiles in [0, 1] (a scalar for a 0-d q).
    :param dtype: q's dtype.
    :returns: the (base, view) operand.
    """
    arr = np.asarray(values, dtype=np.float64)
    if fn.endswith("percentile"):
        arr = arr * 100
    return plain(arr.astype(dtype))


def _qp_broadcast(dtype, value, shape):
    """A read-only broadcast view of one element: a long line that costs one element of corpus.

    :param dtype: the dtype name.
    :param value: the element.
    :param shape: the view's shape.
    :returns: the (base, view) operand.
    """
    base = np.array([value], dtype=dtype)
    return base, np.broadcast_to(base, shape)


def _qp_out(dtype, shape, writeable=True):
    """An out operand of ``dtype`` and ``shape`` (zeros), read-only when ``writeable`` is False.

    :param dtype: the dtype name.
    :param shape: the shape.
    :param writeable: False for a read-only view.
    :returns: the (base, view) operand.
    """
    # An empty out is its own base: a view of a zero-size buffer is rebuilt as an owner by the replay.
    base, view = own(np.zeros(shape, dtype=dtype)) if 0 in tuple(shape) else plain(np.zeros(shape, dtype=dtype))
    if not writeable:
        view = view.view()
        view.flags.writeable = False
    return base, view


def gen_qprec():
    """Section qprec: the quantile family's index arithmetic in q's dtype, and the routes around it (op ``quant_kw``).
    Every case is phase 1.

    * q's precision: data of 11 dtypes x float16 / float32 q arrays (1-d and 0-d) x the 13 methods x the four functions x
      a flat line, rows along an axis and a transposed operand. NumPy computes the virtual index, the neighbours, the weight
      t and ``1 - t`` in q's dtype, and the lerp in ``result_type(data, q)`` (float16 for int8 data under a float16 q);
    * the partition's kth validation: float16 q over lines of 2047 to 4097 elements and of 65519 / 65520 / 65521 / 70000
      (broadcast lines, one element of corpus): ``n - 1`` rounds in float16 (2051 is 2052) or overflows to inf, so an index
      reaches n (``kth(=2052) out of bounds (2052)``), or is NaN / inf (``kth(=-9223372036854705808) out of bounds
      (70000)``, or NaN results over float data); the nan forms per slice, a long slice after a short one and the reverse;
    * percentile's ``np.true_divide(q, 100)`` in q's dtype, and the range check on the quotient (float16 ``100.0625`` is
      refused, ``-2**-24`` and float64 ``-5e-324`` pass);
    * out=: the take route (exact shape, a safe cast from out to the data, ``WRITEBACKIFCOPY base is read-only``, the
      NaN slices' same_kind ``np.copyto``), the lerp route (the add's same_kind cast, the broadcast into a larger out, the
      t >= 0.5 branch computed in out's dtype, ``output array is read-only``), the nan forms' assignment, keepdims' out
      index — over out dtypes float16 / float32 / float64 / complex128 / int64 / int32 / bool / uint8;
    * an integer q ARRAY under weibull / averaged_inverted_cdf / interpolated_inverted_cdf: an int64 weight, so integer
      data gives an int64 result (uint64 and float data float64), with out=;
    * empty data: the take's ``cannot do a non-empty take from an empty axes.`` and index checks, the neighbour lookup's
      ``index 0`` / ``index -1``, float data's ``arr[-1, ...]``, bool data's IndexError before the subtract, empty results
      with out=;
    * text with a float16 q: the take path's kth errors over 2052 strings, the interpolating refusal after them, an
      inverted_cdf index that a float16 q moves, StringDType's NaN-like NA lerp in float16 / float32.
    """
    # --- q's precision over every dtype and method.
    layouts = [("flat", (37,), None), ("rows", (5, 23), 1), ("cols", (23, 5), 0)]
    qvals = [0.0, 0.1, 0.3, 0.33, 0.5, 0.7, 0.999, 1.0]
    for di, dtype in enumerate(QP_NUM):
        for li, (lname, shape, ax) in enumerate(layouts):
            n = int(np.prod(shape))
            vals = _qp_values(dtype, n, 1000 + 31 * di + li)
            data = plain(vals.reshape(shape)) if lname != "cols" else plain(vals.reshape(shape[::-1]))
            if lname == "cols":
                base, view = data
                data = (base, view.T)
            for qdt in ("float16", "float32"):
                for fn in QP_FUNCS:
                    q = _qp_q(fn, qvals, qdt)
                    for m in QP_METHODS:
                        _quant(f"{dtype}-{lname}-q{qdt}", data, name=fn, q=q, qkind="arr", axis=ax, method=m,
                               section=QPREC)
        # 0-d q arrays (the scalar result) and keepdims.
        data = plain(_qp_values(dtype, 29, 77 + di))
        for qdt in ("float16", "float32"):
            for fn in ("quantile", "nanquantile"):
                for m in QP_METHODS:
                    _quant(f"{dtype}-0d-q{qdt}", data, name=fn, q=_qp_q(fn, 0.33, qdt), qkind="arr", method=m,
                           keepdims=m in ("linear", "lower"), section=QPREC)

    # --- the partition's kth validation and the float16 overflow of n - 1.
    for n in (2047, 2048, 2049, 2050, 2051, 2052, 2053, 2056, 2060, 4097):
        for dtype in ("int16", "float32"):
            vals = (np.arange(n) * 7 % n).astype(dtype)      # distinct values, a known order
            data = plain(vals)
            for m in QP_METHODS:
                _quant(f"kth{n}-{dtype}", data, name="quantile", q=_qp_q("quantile", [0.5, 0.999, 0.9995, 1.0], "float16"),
                       qkind="arr", method=m, section=QPREC)
                for qv in (0.9995, 1.0):
                    _quant(f"kth{n}-{dtype}-q{qv}", data, name="quantile", q=_qp_q("quantile", [qv], "float16"),
                           qkind="arr", method=m, section=QPREC)
            if n in (2051, 2052, 2056):
                for m in ("lower", "inverted_cdf", "linear", "midpoint"):
                    _quant(f"kth{n}-{dtype}-pct", data, name="percentile", q=_qp_q("percentile", [0.999, 1.0], "float16"),
                           qkind="arr", method=m, section=QPREC)
                    _quant(f"kth{n}-{dtype}-q32", data, name="quantile", q=_qp_q("quantile", [0.999, 1.0], "float32"),
                           qkind="arr", method=m, section=QPREC)
    for n in (65519, 65520, 65521, 70000):
        for dtype, value in (("int8", 5), ("float64", 2.5), ("float16", 1.5), ("bool", True)):
            for shape, ax in (((n,), None), ((2, n), 1)):
                data = _qp_broadcast(dtype, value, shape)
                for m in QP_METHODS:
                    for qv in ([0.0], [0.5], [1.0], [0.5, 0.0]):
                        _quant(f"ovf{n}-{dtype}-{len(shape)}d", data, name="quantile", q=_qp_q("quantile", qv, "float16"),
                               qkind="arr", axis=ax, method=m, section=QPREC)
                if dtype == "float64":
                    for fn in ("nanquantile", "percentile"):
                        _quant(f"ovf{n}-{dtype}-{fn}", data, name=fn, q=_qp_q(fn, [0.0, 0.5], "float16"), qkind="arr",
                               axis=ax, method="linear", section=QPREC)
    # The nan forms decide per slice: a slice of 2048 values and one of 2052 (NaN-padded rows of 2100), in both orders.
    for order in ("short-first", "long-first"):
        rows = np.empty((2, 2100), dtype=np.float32)
        rows[0] = np.arange(2100) * 0.5
        rows[1] = np.arange(2100) * 0.25
        short, long_ = (0, 1) if order == "short-first" else (1, 0)
        rows[short, 2048:] = np.nan
        rows[long_, 2052:] = np.nan
        data = plain(rows)
        for m in QP_METHODS:
            for fn in ("nanquantile", "nanpercentile"):
                _quant(f"nanslices-{order}", data, name=fn, q=_qp_q(fn, [0.5, 1.0], "float16"), qkind="arr", axis=1,
                       method=m, section=QPREC)

    # --- percentile: the quotient in q's dtype and the range check on it.
    pdata = plain(np.arange(9.0) * 1.5)
    pqs = [("float16", [100.0]), ("float16", [100.0625]), ("float16", [100.125]), ("float16", [99.95]),
           ("float16", [-0.0]), ("float16", [-2.0 ** -24]), ("float16", [33.3, 66.7]), ("float16", [0.1]),
           ("float32", [100.0]), ("float32", [float(np.nextafter(np.float32(100), np.float32(200)))]),
           ("float32", [-1e-45]), ("float32", [33.3]), ("float64", [-5e-324]), ("float64", [33.3])]
    for qdt, qv in pqs:
        for fn in ("percentile", "nanpercentile"):
            for m in ("linear", "lower", "hazen"):
                _quant(f"pct-{qdt}", pdata, name=fn, q=plain(np.array(qv, dtype=qdt)), qkind="arr", method=m,
                       section=QPREC)
    for fn in ("percentile", "nanpercentile"):
        _quant("pct-py", pdata, name=fn, q=-5e-324, qkind="f", method="linear", section=QPREC)
        _quant("pct-pylist", pdata, name=fn, q=[-5e-324, 50.0], qkind="fl", method="linear", section=QPREC)

    # --- out=: the take, the lerp and the nan forms' routes over every out dtype.
    out_dtypes = ["float16", "float32", "float64", "complex128", "int64", "int32", "bool", "uint8"]
    for di, dtype in enumerate(["float64", "float32", "float16", "int32", "int64", "uint64", "bool"]):
        vals = _qp_values(dtype, 11, 500 + di)
        nanvals = _qp_values(dtype, 11, 500 + di, nan_every=4) if dtype.startswith("float") else None
        for dset, dv in (("v", vals), ("nan", nanvals)):
            if dv is None:
                continue
            data = plain(dv)
            for qform in ("weak", "f64", "f16", "f32-0d"):
                for fn in ("quantile", "nanquantile"):
                    if qform == "weak":
                        q, qk, rshape = 0.3, "f", ()
                    elif qform == "f64":
                        q, qk, rshape = _qp_q(fn, [0.3, 0.65], "float64"), "arr", (2,)
                    elif qform == "f16":
                        q, qk, rshape = _qp_q(fn, [0.3, 0.65], "float16"), "arr", (2,)
                    else:
                        q, qk, rshape = _qp_q(fn, 0.3, "float32"), "arr", ()
                    for m in ("linear", "lower", "hazen", "midpoint"):
                        for odt in out_dtypes:
                            # The nan forms store a 0-d answer as a NumPy SCALAR (PyArray_Pack): a SIGNED integer out
                            # converts it with int() through C long, so a value past 2^31 into int32 raises "Python int
                            # too large to convert to C long" on Windows (32-bit long) but "Python integer ... out of
                            # bounds for int32" on LP64, and a uint64 past 2^63 into int64 raises "int too big to
                            # convert" on Windows (C long long) but "... to C long" on LP64 — LP64 is the answer NumSharp
                            # pins, left to the unit tests. An unsigned out casts the scalar: the same on every platform.
                            if fn.startswith("nan") and rshape == () and (
                                    (dtype in ("int64", "uint64") and odt == "int32")
                                    or (dtype == "uint64" and odt == "int64")):
                                continue
                            _quant(f"out-{dtype}-{dset}-{qform}-{odt}", data, name=fn, q=q, qkind=qk, method=m,
                                   out=_qp_out(odt, rshape), section=QPREC)
    # out's shapes, read-only outs and keepdims' out index, per route.
    data = plain(_qp_values("float64", 9, 901))
    datan = plain(_qp_values("float64", 9, 902, nan_every=3))
    data2 = plain(_qp_values("float64", 12, 903).reshape(3, 4))
    shape_cases = [((3, 2), "f8"), ((3,), "f8"), ((2, 1), "f8"), ((1, 2), "f8"), ((2, 3), "f8"), ((), "f8"),
                   ((3,), "i8"), ((), "i8"), ((3, 2), "i8")]
    for fn in ("quantile", "nanquantile"):
        for m in ("linear", "lower"):
            for oshape, odt in shape_cases:
                _quant(f"outshape-{oshape}-{odt}", data, name=fn, q=_qp_q(fn, [0.3, 0.6], "float64"), qkind="arr",
                       method=m, out=_qp_out(np.dtype(odt).name, oshape), section=QPREC)
            _quant("outshape-scalar-into-1", data, name=fn, q=0.3, qkind="f", method=m, out=_qp_out("float64", (1,)),
                   section=QPREC)
            _quant("outshape-scalar-into-3x1", data, name=fn, q=0.3, qkind="f", method=m, out=_qp_out("float64", (3, 1)),
                   section=QPREC)
            for odt in ("float64", "float32", "int64"):
                _quant(f"readonly-{odt}", data, name=fn, q=_qp_q(fn, [0.3], "float64"), qkind="arr", method=m,
                       out=_qp_out(odt, (1,), writeable=False), section=QPREC)
                _quant(f"readonly-nan-{odt}", datan, name=fn, q=_qp_q(fn, [0.3], "float64"), qkind="arr", method=m,
                       out=_qp_out(odt, (1,), writeable=False), section=QPREC)
            for ax, oshape in ((1, (3, 1)), (1, (1, 3, 1)), (1, (3,)), (0, (1, 4)), (0, (2, 1, 4)), (None, (1, 1)),
                               (None, (2,)), (1, (0, 1)), (1, (3, 1, 2))):
                for odt in ("float64", "float32", "int64"):
                    _quant(f"keepdims-out-{ax}-{oshape}-{odt}", data2, name=fn, q=0.3, qkind="f", axis=ax, method=m,
                           keepdims=True, out=_qp_out(odt, oshape), section=QPREC)
        _quant("bool-lerp-readonly", plain(np.array([True, False, True])), name=fn, q=_qp_q(fn, [0.5], "float64"),
               qkind="arr", method="linear", out=_qp_out("float64", (1,), writeable=False), section=QPREC)
    # The take path's NaN slices into an out the data does not reach under same_kind.
    for fn in ("quantile", "percentile"):
        nan2 = plain(np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 6.0]]))
        for odt in ("int64", "bool", "float16", "complex128", "uint8"):
            _quant(f"take-nan-{odt}-flat", datan, name=fn, q=_qp_q(fn, [0.5], "float64"), qkind="arr", method="lower",
                   out=_qp_out(odt, (1,)), section=QPREC)
            _quant(f"take-nan-{odt}-0d", datan, name=fn, q=0.5 * (100 if fn == "percentile" else 1), qkind="f",
                   method="nearest", out=_qp_out(odt, ()), section=QPREC)
            _quant(f"take-nan-{odt}-axis", nan2, name=fn, q=_qp_q(fn, [0.5], "float64"), qkind="arr", axis=1,
                   method="lower", out=_qp_out(odt, (1, 2)), section=QPREC)

    # --- an integer q ARRAY: the int64 weight of weibull / averaged_inverted_cdf / interpolated_inverted_cdf.
    int_dtypes = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "bool", "float16", "float32",
                  "float64"]
    for di, dtype in enumerate(int_dtypes):
        vals = _qp_values(dtype, 8, 300 + di)
        data = plain(vals)
        data2 = plain(_qp_values(dtype, 12, 400 + di).reshape(3, 4))
        for qv, qdt in (([0, 1], "int64"), ([1], "int32"), (1, "int64"), ([False, True], "bool"), ([1, 0, 1], "uint8")):
            for fn in ("quantile", "nanquantile"):
                for m in QP_METHODS:
                    _quant(f"intq-{dtype}-{qdt}", data, name=fn, q=plain(np.array(qv, dtype=qdt)), qkind="arr",
                           method=m, section=QPREC)
                for m in ("weibull", "averaged_inverted_cdf", "interpolated_inverted_cdf"):
                    _quant(f"intq-axis-{dtype}-{qdt}", data2, name=fn, q=plain(np.array(qv, dtype=qdt)), qkind="arr",
                           axis=1, method=m, section=QPREC)
        for m in ("weibull", "averaged_inverted_cdf"):
            _quant(f"intq-py-{dtype}", data, name="quantile", q=1, qkind="i", method=m, section=QPREC)
            for odt in ("int32", "bool", "float32", "uint8", "complex128", "float16"):
                _quant(f"intq-out-{dtype}-{odt}", data, name="quantile", q=plain(np.array([0, 1])), qkind="arr",
                       method=m, out=_qp_out(odt, (2,)), section=QPREC)
    big = plain(np.array([2 ** 60 + 1, 3, -(2 ** 62) - 7], dtype=np.int64))
    for odt in ("float32", "float16", "int64"):
        _quant(f"intq-int64-big-{odt}", big, name="quantile", q=plain(np.array([0, 1])), qkind="arr", method="weibull",
               out=_qp_out(odt, (2,)), section=QPREC)
    # An integer q over a line longer than its dtype: NEP 50 converts the Python int n (or n - 1) to q's dtype first and
    # raises (``Python integer 128 out of bounds for int8``) — at n for the methods that multiply n, at n - 1 for the
    # others, so the line of exactly 128 / 256 / 32768 / 65536 elements raises for some methods only. Broadcast lines: one
    # element of corpus each.
    for qdt, n in (("int8", 128), ("int8", 200), ("uint8", 256), ("uint8", 300), ("int16", 32768), ("int16", 40000),
                   ("uint16", 65536), ("uint16", 70000)):
        for dtype, value in (("float64", 2.5), ("int32", 5)):
            data = _qp_broadcast(dtype, value, (n,))
            for fn in ("quantile", "nanquantile"):
                for m in QP_METHODS:
                    _quant(f"intq-long-{qdt}-{n}-{dtype}", data, name=fn, q=plain(np.array([0, 1], dtype=qdt)),
                           qkind="arr", method=m, section=QPREC)

    # --- the nan forms' 0-d answer stored into an out by PyArray_Pack: a SIGNED integer out converts the NumPy scalar
    # with int() (a NaN answer — an all-NaN line, or inf - inf in the lerp — raises ``cannot convert float NaN to
    # integer``, an infinite one ``cannot convert float infinity to integer``, a value past the out's range ``Python integer
    # 400 out of bounds for int8``, one past C long ``Python int too large to convert to C long``); every other out casts
    # it (an unsigned out wraps, and takes 0 or 2^63 for NaN / infinity; a bool out its truth). These outcomes are the same
    # on every platform but one: 1e20 into int64 names C long long on Windows (skipped).
    nan_lines = [("allnan", [np.nan, np.nan, np.nan], 0.5, "linear"), ("infdiff", [np.inf, 1.0], 1.0, "linear"),
                 ("inf", [np.inf, np.nan], 0.5, "linear"), ("neginf", [-np.inf, np.nan, 2.0], 0.0, "lower"),
                 ("finite", [1.0, np.nan, 4.0], 0.5, "linear"), ("big", [300.0, np.nan, 500.0], 0.5, "linear"),
                 ("negative", [-7.5, np.nan, -2.5], 0.5, "linear"), ("huge", [1e20, np.nan, 3e20], 0.0, "lower")]
    for lname, line, qv, m in nan_lines:
        for dtype in ("float64", "float32", "float16"):
            data = plain(np.array(line, dtype=dtype))
            for fn in ("nanquantile", "nanpercentile"):
                for odt in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"):
                    if lname == "huge" and dtype != "float16" and odt == "int64":
                        continue
                    _quant(f"nan0d-{lname}-{dtype}-{odt}", data, name=fn, q=qv * (100 if fn == "nanpercentile" else 1),
                           qkind="f", method=m, out=_qp_out(odt, ()), section=QPREC)
                _quant(f"nan0d-{lname}-{dtype}-readonly", data, name=fn, q=qv * (100 if fn == "nanpercentile" else 1),
                       qkind="f", method=m, out=_qp_out("int8", (), writeable=False), section=QPREC)

    # --- empty data.
    empty_shapes = [("e1", (0,), None), ("e0x3-0", (0, 3), 0), ("e0x3-1", (0, 3), 1), ("e3x0-0", (3, 0), 0),
                    ("e3x0-1", (3, 0), 1), ("e0x0-0", (0, 0), 0), ("e0x0-1", (0, 0), 1)]
    empty_methods = ["linear", "lower", "higher", "nearest", "inverted_cdf", "closest_observation", "weibull",
                     "midpoint", "averaged_inverted_cdf", "hazen"]
    for dtype in ("int32", "bool", "float64", "float16", "uint8", "int64"):
        for sname, shape, ax in empty_shapes:
            data = plain(np.zeros(shape, dtype=dtype))
            for qname, q, qk in (("q[]", [], "fl"), ("q.5", [0.5], "fl"), ("q0", [0.0], "fl"), ("qs", 0.5, "f"),
                                 ("q.5,0", [0.5, 0.0], "fl"), ("q16", None, "arr")):
                for fn in ("quantile", "percentile"):
                    if qk == "arr":
                        qq = _qp_q(fn, [0.5, 1.0], "float16")
                    elif fn == "percentile":
                        qq = q * 100 if qk == "f" else [v * 100 for v in q]
                    else:
                        qq = q
                    for m in empty_methods:
                        _quant(f"empty-{dtype}-{sname}-{qname}", data, name=fn, q=qq, qkind=qk, axis=ax, method=m,
                               section=QPREC)
        data = plain(np.zeros((0,), dtype=dtype))
        for m in ("linear", "lower"):
            for oname, oshape, odt in (("i8", (1,), "int64"), ("f4", (1,), "float32"), ("i1", (1,), "int8"),
                                       ("bad", (2,), dtype), ("ro", (1,), dtype)):
                _quant(f"empty-out-{dtype}-{oname}", data, name="quantile", q=[0.5], qkind="fl", method=m,
                       out=_qp_out(odt, oshape, writeable=oname != "ro"), section=QPREC)
            for odt in ("int64", "float32", dtype):
                _quant(f"empty-out-q[]-{dtype}-{odt}", data, name="quantile", q=[], qkind="fl", method=m,
                       out=_qp_out(odt, (0,)), section=QPREC)

    # --- text with a float16 q.
    for spec in ("<U5", "|S5"):
        strs = ["%05d" % i for i in range(2052)]
        arr = np.array([s.encode() for s in strs], dtype=spec) if spec == "|S5" else np.array(strs, dtype=spec)
        data = plain(arr)
        for m in QP_METHODS:
            _quant(f"text2052-{spec}", data, name="quantile", q=_qp_q("quantile", [1.0], "float16"), qkind="arr",
                   method=m, section=QPREC)
        for m in ("lower", "inverted_cdf"):
            _quant(f"text2052-{spec}-pct", data, name="percentile", q=_qp_q("percentile", [0.999], "float16"),
                   qkind="arr", method=m, section=QPREC)
    strs = ["%05d" % i for i in range(3001)]
    data = plain(np.array(strs, dtype="<U5"))
    for m in ("lower", "higher", "nearest", "inverted_cdf", "closest_observation"):
        for qdt in ("float16", "float32", "float64"):
            _quant(f"text3001-{qdt}", data, name="quantile", q=_qp_q("quantile", [0.3, 0.7, 0.999], qdt), qkind="arr",
                   method=m, section=QPREC)
    _quant("text-ovf", _qp_broadcast("<U3", "abc", (70000,)), name="quantile", q=_qp_q("quantile", [0.0], "float16"),
           qkind="arr", method="lower", section=QPREC)
    _quant("text-ovf-linear", _qp_broadcast("<U3", "abc", (70000,)), name="quantile",
           q=_qp_q("quantile", [0.5, 0.0], "float16"), qkind="arr", method="linear", section=QPREC)
    # StringDType: the take path in float16, and the NaN-like NA lerp in q's dtype.
    tspec = QF_TSPEC["Tnan"]
    tnan = t_plain(np.array([np.nan, np.nan, np.nan], dtype=tspec.build()))
    tstr = t_plain(np.array(["a", "b", "c", "d"], dtype=QF_TSPEC["T"].build()))
    for qdt in ("float16", "float32", "float64"):
        for m in ("linear", "hazen", "midpoint"):
            _quant(f"T-nan-{qdt}", tnan, name="quantile", q=plain(np.array(0.3, dtype=qdt)), qkind="arr", method=m,
                   section=QPREC)
        for m in ("lower", "inverted_cdf", "nearest"):
            _quant(f"T-str-{qdt}", tstr, name="quantile", q=plain(np.array([0.3, 0.9], dtype=qdt)), qkind="arr",
                   method=m, section=QPREC)
        _quant(f"T-str-lerp-{qdt}", tstr, name="quantile", q=plain(np.array(0.5, dtype=qdt)), qkind="arr",
               method="linear", section=QPREC)
        _quant(f"T-nan-out-{qdt}", tnan, name="quantile", q=plain(np.array(0.3, dtype=qdt)), qkind="arr",
               method="linear", out=_qp_out("float64", ()), section=QPREC)


# ----------------------------------------------------------------------------------------------------------------------
# Sections strfn / tfn: numpy.strings' search, strip, pad, replace, partition and slice functions over S / U (strfn) and
# NEP 55 StringDType (tfn), strings phase 2, P2.5, strict
# ----------------------------------------------------------------------------------------------------------------------

# Texts that reach every rule of the P2.5 loops over S / U: the whitespace of each class (ASCII for S, Python's
# str.isspace for U — \x1c-\x1f, NBSP and IDEOGRAPHIC SPACE included), embedded NULs, signs for zfill, tabs and line breaks
# for expandtabs, repeated patterns for count / replace (non-overlapping greedy matches), bytes past 0x7f for S, non-ASCII
# and supplementary-plane characters for U, and the empty string. np.array truncates a value longer than the descriptor.
FN_U_POOL = ["", " ", "a", "ab", "abc", "aaaa", "abab", "  x y  ", "\tx\n", "x\ty\tz", "-12", "+3.5", "-", "+", "0",
             "a\x00b", "é日本", "\U0001F600x", "it's", "Hello World", "\x1cz\x1f", "\xa0a　", "abcabcab", "zz\r\nq"]
FN_S_POOL = [b"", b" ", b"a", b"ab", b"abc", b"aaaa", b"abab", b"  x y  ", b"\tx\n", b"x\ty\tz", b"-12", b"+3.5", b"-",
             b"+", b"0", b"a\x00b", b"\x80\xff", b"\xe9t\xe9", b"it's", b"Hello World", b"\x0bv\x0c", b"abcabcab",
             b"zz\r\nq", b"\x1cz"]
FN_SPECS = ["<U5", "<U12", "|S5", "|S12"]


def _fn_nonempty(spec):
    """The P2.5 pool of ``spec``'s class without the empty string: NumPy 2.4.2's zfill of an EMPTY element reads the
    output byte at the width for a sign before the loop zero-fills it (``string_zfill``'s ``*tmp`` past the copied text:
    uninitialized ``np.empty_like`` memory), so its answer depends on what the allocator recycled; NumSharp's ``'0' *
    width`` (Python's ``''.zfill``) is pinned by the unit tests instead.

    :param spec: an S / U descriptor string.
    :returns: the texts.
    """
    return [v for v in (FN_U_POOL if spec[1] == "U" else FN_S_POOL) if v]


def fn_operand(spec, layout, seed=0, pool=None):
    """An S / U operand over the P2.5 pools: a base cycling through the texts from ``seed`` and the layout's view of it.

    :param spec: the descriptor string.
    :param layout: a ``LAYOUTS`` name.
    :param seed: the first pool index.
    :param pool: the texts (``FN_U_POOL`` / ``FN_S_POOL`` by the class by default).
    :returns: the (base, view) pair.
    """
    n, build = LAYOUTS[layout]
    pool = pool or (FN_U_POOL if spec[1] == "U" else FN_S_POOL)
    base = np.array([pool[(seed + i) % len(pool)] for i in range(n)], dtype=spec) if n else np.zeros(0, dtype=spec)
    return base, build(base)


class _Op:
    """An argument of a numpy.strings call that is the case's operand ``i`` (its view)."""

    def __init__(self, i):
        """Keeps the operand index.

        :param i: the operand index.
        """
        self.i = i


class _NpStr:
    """A NumPy ``np.str_`` argument: a 0-d text that HAS a dtype (NumSharp's ``NDArray.Scalar(string)``), where a Python
    str has none (NumSharp's implicit ``string`` conversion) — replace casts only the dtype-less one to the array's
    descriptor."""

    def __init__(self, text):
        """Keeps the text.

        :param text: the str.
        """
        self.text = text


def _fn_arg_enc(x):
    """One argument of a numpy.strings call as the replay reads it: ``{"op": i}`` for an operand, ``{"npstr": …}`` for a
    np.str_, otherwise ``{"py": enc(value)}`` (a str is a Python str, bytes are np.bytes_ — a dtype-carrying scalar, as
    NumSharp's ``NDArray.Scalar(byte[])`` is — an int a Python int, None NumSharp's null default).

    :param x: the argument.
    :returns: the encoding.
    """
    if isinstance(x, _Op):
        return {"op": x.i}
    if isinstance(x, _NpStr):
        return {"npstr": enc(x.text)}
    return {"py": enc(x)}


def _fn_arg_value(x, views):
    """One argument of a numpy.strings call as NumPy receives it.

    :param x: the argument.
    :param views: the operand views.
    :returns: the value (bytes become np.bytes_, a np.str_ marker its np.str_).
    """
    if isinstance(x, _Op):
        return views[x.i]
    if isinstance(x, _NpStr):
        return np.str_(x.text)
    if isinstance(x, bytes):
        return np.bytes_(x)
    return x


def _fn_zero_width_parts():
    """Records the S / U partition parts NumPy sizes at width 0 (``'U0'`` / ``'S0'``, a structured field of no bytes) as
    the width-1 arrays of empty strings NumSharp returns: NumSharp has no zero-width string dtype, the one documented
    divergence of these sections (its layout — NumPy's three field views of one structured array — is the replay's
    excused layout). Applied to the case just added."""
    exp = CASES[-1].get("expected")
    if not exp or exp.get("kind") != "tuple":
        return
    for item in exp["items"]:
        if item.get("dtype") in ("<U0", "|S0"):
            n = 1
            for d in item["shape"]:
                n *= d
            item["dtype"] = "<U1" if item["dtype"] == "<U0" else "|S1"
            item["buffer"] = "00" * (n * (4 if item["dtype"] == "<U1" else 1))
            CASES[-1]["note"] = "a zero-width part recorded as width 1 (NumSharp has no zero-width string dtype)"


def fn_case(section, name, operands, args, layout, kw=None):
    """Records one numpy.strings call: ``np.strings.<name>(*args, **kw)`` (op ``str_fn``), each ``_Op(i)`` argument the
    view of operand ``i``, every other argument a Python value (``_fn_arg_value``); ``kw`` names operands for out= /
    where= and a dtype string for dtype=.

    :param section: the section.
    :param name: the numpy.strings function.
    :param operands: the (base, view) pairs.
    :param args: the positional arguments.
    :param layout: the case's layout label.
    :param kw: ``{"out": i, "where": i, "dtype": str}`` (each optional), or None.
    """
    params = {"name": name, "args": [_fn_arg_enc(x) for x in args]}
    if kw:
        params["kw"] = kw

    def call(*views):
        values = [_fn_arg_value(x, views) for x in args]
        kwargs = {}
        if kw:
            if "out" in kw:
                kwargs["out"] = views[kw["out"]]
            if "where" in kw:
                kwargs["where"] = views[kw["where"]]
            if "dtype" in kw:
                kwargs["dtype"] = kw["dtype"]
        return getattr(np.strings, name)(*values, **kwargs)

    add(section, "str_fn", params=params, operands=operands, call=call, layout=layout)
    if "error" in CASES[-1] and _platform_text(CASES[-1]["error"]["text"]):
        # NumPy's C long overflow text is the host's (Windows: "int too big to convert"); NumSharp pins the LP64 text in
        # its unit tests, the policy of _platform_text.
        _pop_case()
        return
    if name in ("partition", "rpartition"):
        _fn_zero_width_parts()


def _fn_t(spec):
    """The text-parameter maker of a class: the str itself for U (and StringDType), its latin-1 bytes for S.

    :param spec: an S / U descriptor string, or ``"T"``.
    :returns: the maker.
    """
    return (lambda s: s.encode("latin-1")) if spec[1:2] == "S" else (lambda s: s)


_FN_FIND = ["find", "rfind", "index", "rindex", "count", "startswith", "endswith"]
_FN_RANGES = [(), (0,), (1,), (-3,), (10,), (0, 3), (2, -1), (5, 2), (-10, -1), (1, None), (0, 2**62), (-(2**62), 3)]
_FN_SLICES = [(), (3,), (None,), (-2,), (1, 4), (None, 3), (2, None), (-3, -1), (4, 1), (1, 4, 2), (None, None, -1),
              (None, None, -2), (5, 0, -1), (-1, None, -3), (0, 100, 3), (-100, 100, 1), (2**62, None, -1),
              (None, None, 2**62), (None, None, -(2**62)), (3, 3), (-1, -100, -1)]


def gen_strfn():
    """Section strfn: numpy.strings' P2.5 functions over S / U text, strict — str_len, the find family, startswith /
    endswith, the strips, the pads, zfill, expandtabs, replace, partition / rpartition, slice and the comparison names:
    every layout kind of both classes, every parameter form (Python ints, arrays of each integer width, broadcast operands,
    the defaults), NumPy's quirks (expandtabs sizes U four times wide, partition's empty separator, the S buffer / U fill
    casts), and the wrappers' and ufuncs' refusals in NumPy's order."""
    S = "strfn"
    # A. Every function over every layout kind, both classes, two widths (a representative parameter set each).
    for spec in FN_SPECS:
        t = _fn_t(spec)
        for layout in LAYOUTS:
            op = fn_operand(spec, layout, seed=1)
            lab = f"{spec}:{layout}"
            fn_case(S, "str_len", [op], [_Op(0)], lab)
            for name in ["find", "rfind", "count", "index", "rindex"]:
                fn_case(S, name, [op], [_Op(0), t("ab")], lab)
                fn_case(S, name, [op], [_Op(0), t("a"), 1, -1], lab)
            for name in ["startswith", "endswith"]:
                fn_case(S, name, [op], [_Op(0), t("a")], lab)
                fn_case(S, name, [op], [_Op(0), t("b"), 0, 2], lab)
            for name in ["strip", "lstrip", "rstrip"]:
                fn_case(S, name, [op], [_Op(0)], lab)
                fn_case(S, name, [op], [_Op(0), t(" a")], lab)
            for name in ["center", "ljust", "rjust"]:
                fn_case(S, name, [op], [_Op(0), 9], lab)
                fn_case(S, name, [op], [_Op(0), 4, t("*")], lab)
            zop = fn_operand(spec, layout, seed=1, pool=_fn_nonempty(spec))
            fn_case(S, "zfill", [zop], [_Op(0), 7], lab)
            fn_case(S, "expandtabs", [op], [_Op(0)], lab)
            fn_case(S, "expandtabs", [op], [_Op(0), 3], lab)
            fn_case(S, "replace", [op], [_Op(0), t("a"), t("XY")], lab)
            fn_case(S, "replace", [op], [_Op(0), t("ab"), t(""), 1], lab)
            for name in ["partition", "rpartition"]:
                fn_case(S, name, [op], [_Op(0), t("b")], lab)
            fn_case(S, "slice", [op], [_Op(0), 3], lab)
            fn_case(S, "slice", [op], [_Op(0), 1, None, 2], lab)
            fn_case(S, "slice", [op], [_Op(0), None, None, -1], lab)

    # B. The find family and the tail matches: every needle kind (empty, longer than the text, a trailing NUL — which
    #    never counts — and a lone NUL, non-ASCII) x every start / end form, over a contiguous and an F-ordered operand.
    for spec in ["<U8", "|S8"]:
        t = _fn_t(spec)
        for layout in ["c2", "f2"]:
            op = fn_operand(spec, layout, seed=3)
            for name in _FN_FIND:
                for sub in ["a", "ab", "", "abc", "x y", "b\x00", "\x00", "é", "aa", "Z"]:
                    for rng in _FN_RANGES:
                        fn_case(S, name, [op], [_Op(0), t(sub), *rng], f"{spec}:{layout}:{sub!r}:{rng}")
    # Start / end arrays of every integer width (and bool, which casts safely), broadcast columns, a 3-D end that
    # broadcasts the text up, per-element needles, a needle row.
    for spec in ["<U6", "|S6"]:
        t = _fn_t(spec)
        op = fn_operand(spec, "c2", seed=4)
        for dt in ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "?"]:
            vals = [0, 1, 2, 3] if dt[0] in "u?" else [0, 1, -2, 3]
            st = plain(np.array(vals, dtype=dt))
            for name in ["find", "rfind", "count", "startswith", "endswith"]:
                fn_case(S, name, [op, st], [_Op(0), t("a"), _Op(1)], f"{spec}:start:{dt}")
                fn_case(S, name, [op, st], [_Op(0), t("b"), 0, _Op(1)], f"{spec}:end:{dt}")
        col = plain(np.array([[0], [1], [-1]]))
        e3 = plain(np.arange(8).reshape(2, 1, 4) - 3)
        needles = plain(np.array([t("a"), t("b"), t(""), t("ab")], dtype=spec))
        per = plain(np.array([[t("a"), t("x"), t(""), t("ab")]] * 3, dtype="<U2" if spec[1] == "U" else "|S2"))
        for name in _FN_FIND:
            fn_case(S, name, [op, col], [_Op(0), t("b"), _Op(1)], f"{spec}:startcol")
            fn_case(S, name, [op, e3], [_Op(0), t("b"), 0, _Op(1)], f"{spec}:end3d")
            fn_case(S, name, [op, needles], [_Op(0), _Op(1)], f"{spec}:needlerow")
            fn_case(S, name, [op, per], [_Op(0), _Op(1), 1], f"{spec}:needles")
    # The refusals: a float / complex start (the input cast), a Python int past int64, a bytes / str mix, a number, a
    # needle row that does not broadcast, index's miss.
    for spec in ["<U6", "|S6"]:
        t = _fn_t(spec)
        other = (lambda s: s.encode("latin-1")) if spec[1] == "U" else (lambda s: s)
        op = fn_operand(spec, "c2", seed=5)
        for name in _FN_FIND:
            fn_case(S, name, [op], [_Op(0), t("a"), 1.5], f"{spec}:float")
            fn_case(S, name, [op], [_Op(0), t("a"), 0, 2.5], f"{spec}:floatend")
            fn_case(S, name, [op], [_Op(0), t("a"), 2**63], f"{spec}:bigint")
            fn_case(S, name, [op, plain(np.array([0.5] * 4))], [_Op(0), t("a"), _Op(1)], f"{spec}:floatarr")
            fn_case(S, name, [op, plain(np.array([1j]))], [_Op(0), t("a"), _Op(1)], f"{spec}:complexarr")
            fn_case(S, name, [op], [_Op(0), other("a")], f"{spec}:mix")
            fn_case(S, name, [op, num_operand(np.arange(4))], [_Op(1), t("a")], f"{spec}:numeric")
            fn_case(S, name, [op, num_operand(np.arange(4))], [_Op(0), _Op(1)], f"{spec}:numericsub")
            fn_case(S, name, [op, plain(np.array([t("a")] * 5, dtype=spec))], [_Op(0), _Op(1)], f"{spec}:nobroadcast")
            fn_case(S, name, [op], [_Op(0), t("zzz")], f"{spec}:miss")
            fn_case(S, name, [op], [_Op(0), 5], f"{spec}:intsub")

    # C. The strips: every set (None = whitespace, empty, a NUL, non-ASCII, one character, several), the whitespace of
    #    each class (U strips Python's whitespace, S only the ASCII six), per-element sets, the refusals.
    ws_u = ["\x1c\x1d\x1e\x1fa\x85", " b　", "\x0b\x0cc\r", " \t\n", "a \x00b ", "\xa0\xa0", "x​y"]
    ws_s = [b"\x1c\x1d\x1e\x1fa\x85", b"\xa0b\xa0", b"\x0b\x0cc\r", b" \t\n", b"a \x00b ", b"\x85\x85", b"x\x0by"]
    for spec, ws in [("<U8", ws_u), ("|S8", ws_s)]:
        t = _fn_t(spec)
        wop = plain(np.array(ws, dtype=spec))
        for layout in ["c2", "r2", "bc", "z0", "e2"]:
            op = fn_operand(spec, layout, seed=6)
            for name in ["strip", "lstrip", "rstrip"]:
                for chars in [None, " ", "a", "ab", "", " \t\n", "\x00", "xa ", "é", "a\x00", "\x1c"]:
                    fn_case(S, name, [op], [_Op(0), t(chars) if chars is not None else None],
                            f"{spec}:{layout}:{chars!r}")
        for name in ["strip", "lstrip", "rstrip"]:
            fn_case(S, name, [wop], [_Op(0)], f"{spec}:whitespace")
            fn_case(S, name, [wop], [_Op(0), t("\x1c\x85 ")], f"{spec}:whitespace:set")
            sets = plain(np.array([t("a"), t(" "), t("xy"), t("")], dtype=spec))
            op = fn_operand(spec, "c2", seed=7)
            fn_case(S, name, [op, sets], [_Op(0), _Op(1)], f"{spec}:setrow")
            other = (lambda s: s.encode("latin-1")) if spec[1] == "U" else (lambda s: s)
            fn_case(S, name, [op], [_Op(0), other("a")], f"{spec}:mix")
            fn_case(S, name, [num_operand(np.arange(3))], [_Op(0)], f"{spec}:numeric")
            fn_case(S, name, [num_operand(np.arange(3))], [_Op(0), t("a")], f"{spec}:numericset")
            fn_case(S, name, [op], [_Op(0), 5], f"{spec}:intset")
            fn_case(S, name, [op, plain(np.array([t("a")] * 5, dtype=spec))], [_Op(0), _Op(1)], f"{spec}:nobroadcast")

    # D. The pads: every width (negative, zero, shorter and longer than the texts, widths of each integer dtype — a uint64
    #    one makes NumPy's descriptor 'U12.0' —, per-element widths), every fill (the default, a character, a non-ASCII
    #    character, a fill of the other class: cast to an S buffer, a U fill decodes), per-element fills, and the refusals
    #    (a float / bool width, a fill that is not one character, a number, the order of center's descriptor and
    #    ljust / rjust's broadcast_shapes).
    for spec in ["<U8", "|S8"]:
        t = _fn_t(spec)
        other = (lambda s: s.encode("latin-1")) if spec[1] == "U" else (lambda s: s)
        for layout in ["c2", "f2", "s1", "bc", "z0"]:
            op = fn_operand(spec, layout, seed=8)
            for name in ["center", "ljust", "rjust"]:
                for width in [-1, 0, 1, 3, 5, 8, 12, 20]:
                    fn_case(S, name, [op], [_Op(0), width], f"{spec}:{layout}:{width}")
                for fill in ["*", "é", "\x00", "ab", "", "-"]:
                    fn_case(S, name, [op], [_Op(0), 10, t(fill)], f"{spec}:{layout}:fill{fill!r}")
                    fn_case(S, name, [op], [_Op(0), 10, other(fill)], f"{spec}:{layout}:otherfill{fill!r}")
        op = fn_operand(spec, "c2", seed=9)
        for name in ["center", "ljust", "rjust"]:
            for dt in ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8"]:
                fn_case(S, name, [op, plain(np.array([3, 6, 9, 12], dtype=dt))], [_Op(0), _Op(1)], f"{spec}:width:{dt}")
                fn_case(S, name, [op, plain(np.array(7, dtype=dt))], [_Op(0), _Op(1), t("~")], f"{spec}:width0d:{dt}")
            fn_case(S, name, [op, plain(np.array([[4], [8], [11]]))], [_Op(0), _Op(1)], f"{spec}:widthcol")
            fn_case(S, name, [op, plain(np.array(["*", "-", "+", "~"], dtype=spec))], [_Op(0), 7, _Op(1)],
                    f"{spec}:fillrow")
            fn_case(S, name, [op], [_Op(0), 2.5], f"{spec}:floatwidth")
            fn_case(S, name, [op], [_Op(0), True], f"{spec}:boolwidth")
            fn_case(S, name, [op], [_Op(0), 6, 5], f"{spec}:numericfill")
            fn_case(S, name, [num_operand(np.arange(3))], [_Op(0), 6], f"{spec}:numeric")
            fn_case(S, name, [op, plain(np.array([t("*")] * 5, dtype=spec))], [_Op(0), 7, _Op(1)], f"{spec}:fillbad")
            fn_case(S, name, [op, plain(np.array([3] * 5))], [_Op(0), _Op(1)], f"{spec}:widthbad")
            e2 = fn_operand(spec, "e2", seed=9)
            fn_case(S, name, [e2, plain(np.array([t("*")] * 5, dtype=spec))], [_Op(0), 7, _Op(1)], f"{spec}:emptyfillbad")
            fn_case(S, name, [e2], [_Op(0), 7], f"{spec}:empty")
            fn_case(S, name, [op], [_Op(0), 2**40], f"{spec}:hugewidth")

    # E. zfill: every width over texts with and without signs (no empty text: see _fn_nonempty), integer widths of each
    #    dtype, per-element widths, the refusals.
    for spec in ["<U8", "|S8"]:
        pool = _fn_nonempty(spec)
        for layout in ["c2", "f2", "r2", "bc", "z0", "e2"]:
            op = fn_operand(spec, layout, seed=10, pool=pool)
            for width in [-2, 0, 2, 5, 9, 15]:
                fn_case(S, "zfill", [op], [_Op(0), width], f"{spec}:{layout}:{width}")
        op = fn_operand(spec, "c2", seed=11, pool=pool)
        for dt in ["i1", "u1", "i4", "u4", "i8", "u8"]:
            fn_case(S, "zfill", [op, plain(np.array([3, 6, 9, 1], dtype=dt))], [_Op(0), _Op(1)], f"{spec}:width:{dt}")
        fn_case(S, "zfill", [op, plain(np.array([[4], [8], [11]]))], [_Op(0), _Op(1)], f"{spec}:widthcol")
        fn_case(S, "zfill", [op], [_Op(0), 2.5], f"{spec}:floatwidth")
        fn_case(S, "zfill", [op], [_Op(0), False], f"{spec}:boolwidth")
        fn_case(S, "zfill", [num_operand(np.arange(3))], [_Op(0), 6], f"{spec}:numeric")
        fn_case(S, "zfill", [op, plain(np.array([3] * 5))], [_Op(0), _Op(1)], f"{spec}:widthbad")
        signs = plain(np.array(["+1", "-1", "+", "-", "1+", "+-2", "--", "a-1"], dtype=spec))
        for width in [0, 1, 2, 4]:
            fn_case(S, "zfill", [signs], [_Op(0), width], f"{spec}:signs:{width}")

    # F. expandtabs: every tab size (the default 8, 0 and negative remove tabs, sizes per element, of each integer dtype),
    #    column resets at \n and \r, NumPy's U sizing (four characters per non-tab character), the overflow past 2^31 - 1,
    #    the refusals.
    tabs_u = ["a\tb", "\t", "ab\tc\td", "x\ny\tz", "1\r\t2", "\t\t\t", "abcdefg\th", "é\t日"]
    for spec, vals in [("<U12", tabs_u), ("|S12", [v.encode("utf-8") for v in tabs_u])]:
        tab = plain(np.array(vals, dtype=spec))
        for size in [None, 0, 1, 2, 4, 8, -1, 13]:
            fn_case(S, "expandtabs", [tab], [_Op(0)] + ([] if size is None else [size]), f"{spec}:tabs:{size}")
        for layout in ["c2", "f2", "r2", "bc", "z0", "e2"]:
            op = fn_operand(spec, layout, seed=12)
            fn_case(S, "expandtabs", [op], [_Op(0), 4], f"{spec}:{layout}")
        for dt in ["i1", "u1", "i4", "u8"]:
            fn_case(S, "expandtabs", [tab, plain(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dt))], [_Op(0), _Op(1)],
                    f"{spec}:tabsize:{dt}")
        fn_case(S, "expandtabs", [tab], [_Op(0), 2.5], f"{spec}:floattab")
        fn_case(S, "expandtabs", [tab], [_Op(0), 2**31], f"{spec}:overflow")
        fn_case(S, "expandtabs", [tab], [_Op(0), 2**63], f"{spec}:bigint")
        fn_case(S, "expandtabs", [num_operand(np.arange(3))], [_Op(0)], f"{spec}:numeric")
        fn_case(S, "expandtabs", [tab, plain(np.array([2] * 5))], [_Op(0), _Op(1)], f"{spec}:nobroadcast")

    # G. replace: every old / new / count form — an empty old (interleaving), an old longer than the descriptor (a Python
    #    str is cast to the array's dtype, truncated; a np.str_ keeps its own), a count of each sign, per-element counts of
    #    each dtype (uint64: NumPy's float sizes), broadcast olds / news / counts, the classes' mix, the refusals.
    for spec in ["<U6", "|S6"]:
        t = _fn_t(spec)
        other = (lambda s: s.encode("latin-1")) if spec[1] == "U" else (lambda s: s)
        for layout in ["c2", "f2", "r2", "bc", "z0", "e2"]:
            op = fn_operand(spec, layout, seed=13)
            for old in ["a", "ab", "", "aa", "b", "x", "abcdefgh"]:
                for new in ["", "Q", "XY", "longer!"]:
                    for count in [(), (0,), (1,), (2,), (-1,), (-5,)]:
                        fn_case(S, "replace", [op], [_Op(0), t(old), t(new), *count],
                                f"{spec}:{layout}:{old!r}->{new!r}:{count}")
        op = fn_operand(spec, "c2", seed=14)
        if spec[1] == "U":
            for old, new in [("ab", "Z"), ("abcdefgh", "Q"), ("a", "longer!"), ("", "-")]:
                fn_case(S, "replace", [op], [_Op(0), _NpStr(old), t(new)], f"{spec}:npstr-old:{old!r}")
                fn_case(S, "replace", [op], [_Op(0), t(old), _NpStr(new)], f"{spec}:npstr-new:{new!r}")
        for dt in ["i1", "u1", "i4", "u4", "i8", "u8"]:
            fn_case(S, "replace", [op, plain(np.array([0, 1, 2, 3], dtype=dt))], [_Op(0), t("a"), t("XY"), _Op(1)],
                    f"{spec}:count:{dt}")
        olds = plain(np.array([t("a"), t("b"), t(""), t("ab")], dtype=spec))
        news = plain(np.array([[t("1")], [t("22")], [t("")]], dtype=spec))
        counts = plain(np.array([[[1]], [[-1]]]))
        fn_case(S, "replace", [op, olds], [_Op(0), _Op(1), t("Z")], f"{spec}:oldrow")
        fn_case(S, "replace", [op, news], [_Op(0), t("a"), _Op(1)], f"{spec}:newcol")
        fn_case(S, "replace", [op, olds, news, counts], [_Op(0), _Op(1), _Op(2), _Op(3)], f"{spec}:all")
        fn_case(S, "replace", [op], [_Op(0), other("a"), t("Z")], f"{spec}:mixold")
        fn_case(S, "replace", [op], [_Op(0), t("a"), other("Z")], f"{spec}:mixnew")
        fn_case(S, "replace", [op], [_Op(0), "é", "Z"], f"{spec}:nonascii")
        fn_case(S, "replace", [op], [_Op(0), t("a"), t("b"), 1.5], f"{spec}:floatcount")
        fn_case(S, "replace", [op], [_Op(0), t("a"), t("b"), True], f"{spec}:boolcount")
        fn_case(S, "replace", [num_operand(np.arange(3))], [_Op(0), t("a"), t("b")], f"{spec}:numeric")
        fn_case(S, "replace", [op], [_Op(0), 5, t("b")], f"{spec}:numericold")
        fn_case(S, "replace", [op, plain(np.array([t("a")] * 5, dtype=spec))], [_Op(0), _Op(1), t("b")],
                f"{spec}:oldbad")
        fn_case(S, "replace", [op, plain(np.array([1] * 5))], [_Op(0), t("a"), t("b"), _Op(1)], f"{spec}:countbad")
        fn_case(S, "replace", [op, plain(np.array([t("b")] * 5, dtype=spec))], [_Op(0), t("a"), _Op(1)],
                f"{spec}:newbad")

    # H. partition / rpartition: every separator (found once, several times, nowhere, a space, a NUL: empty after its
    #    trailing-NUL strip — 'empty separator'), per-element separators, a separator of the other class (cast to the
    #    array's descriptor), the refusals.
    for spec in ["<U6", "|S6"]:
        t = _fn_t(spec)
        other = (lambda s: s.encode("latin-1")) if spec[1] == "U" else (lambda s: s)
        for layout in ["c2", "f2", "r2", "bc", "z0", "e2", "d5"]:
            op = fn_operand(spec, layout, seed=15)
            for name in ["partition", "rpartition"]:
                for sep in ["a", "b", "ab", "x y", " ", "zz", "\x00", "", "abcdefgh"]:
                    fn_case(S, name, [op], [_Op(0), t(sep)], f"{spec}:{layout}:{sep!r}")
        op = fn_operand(spec, "c2", seed=16)
        seps = plain(np.array([t("a"), t("b"), t(" "), t("c")], dtype=spec))
        for name in ["partition", "rpartition"]:
            fn_case(S, name, [op, seps], [_Op(0), _Op(1)], f"{spec}:seprow")
            fn_case(S, name, [op], [_Op(0), other("a")], f"{spec}:mix")
            fn_case(S, name, [num_operand(np.arange(3))], [_Op(0), t("1")], f"{spec}:numeric")
            fn_case(S, name, [num_operand(np.arange(3))], [_Op(0), t("x")], f"{spec}:numericparse")
            fn_case(S, name, [op, plain(np.array([t("a")] * 5, dtype=spec))], [_Op(0), _Op(1)], f"{spec}:sepbad")

    # I. slice: every arity (one argument is the stop), negative and huge bounds, every step sign, per-element bounds and
    #    steps of each integer dtype, broadcast bounds, the refusals (step 0, a float / bool step, a float bound).
    for spec in ["<U8", "|S8"]:
        for layout in ["c2", "f2", "r2", "bc", "z0", "e2"]:
            op = fn_operand(spec, layout, seed=17)
            for args in _FN_SLICES:
                fn_case(S, "slice", [op], [_Op(0), *args], f"{spec}:{layout}:{args}")
        op = fn_operand(spec, "c2", seed=18)
        for dt in ["i1", "u1", "i4", "u4", "i8", "u8"]:
            vals = [1, 2, 3, 1] if dt[0] == "u" else [1, -1, 2, -3]
            arr = plain(np.array(vals, dtype=dt))
            fn_case(S, "slice", [op, arr], [_Op(0), None, None, _Op(1)], f"{spec}:step:{dt}")
            fn_case(S, "slice", [op, arr], [_Op(0), _Op(1), None], f"{spec}:start:{dt}")
            fn_case(S, "slice", [op, arr], [_Op(0), _Op(1)], f"{spec}:stop:{dt}")
        fn_case(S, "slice", [op, plain(np.array([[1], [-1], [2]]))], [_Op(0), None, None, _Op(1)], f"{spec}:stepcol")
        fn_case(S, "slice", [op, plain(np.array([0, 1, -1, 2]))], [_Op(0), None, None, _Op(1)], f"{spec}:stepzero")
        fn_case(S, "slice", [op], [_Op(0), 0, 5, 0], f"{spec}:zero")
        fn_case(S, "slice", [op], [_Op(0), 0, 5, 1.5], f"{spec}:floatstep")
        fn_case(S, "slice", [op], [_Op(0), 0, 5, True], f"{spec}:boolstep")
        fn_case(S, "slice", [op], [_Op(0), 1.5, 5], f"{spec}:floatstart")
        fn_case(S, "slice", [op], [_Op(0), 2**63], f"{spec}:bigstop")
        fn_case(S, "slice", [num_operand(np.arange(3))], [_Op(0), 1], f"{spec}:numeric")
        fn_case(S, "slice", [op, plain(np.array([1] * 5))], [_Op(0), _Op(1)], f"{spec}:stopbad")

    # J. str_len as a ufunc (out= / where= / dtype=), and the comparison names (np.strings.equal IS np.equal).
    for spec in ["<U5", "|S5"]:
        t = _fn_t(spec)
        op = fn_operand(spec, "c2", seed=19)
        for odt in ["i8", "i4", "f8", "i1", "?", "u8"]:
            out = target(np.zeros((3, 4), dtype=odt))
            fn_case(S, "str_len", [op, out], [_Op(0)], f"{spec}:out:{odt}", kw={"out": 1})
        mask = plain(np.array([[True, False, True, False]]))
        out = target(np.full((3, 4), -7, dtype="i8"))
        fn_case(S, "str_len", [op, out, mask], [_Op(0)], f"{spec}:where", kw={"out": 1, "where": 2})
        fn_case(S, "str_len", [op], [_Op(0)], f"{spec}:dtype:i8", kw={"dtype": "i8"})
        fn_case(S, "str_len", [op], [_Op(0)], f"{spec}:dtype:f8", kw={"dtype": "f8"})
        fn_case(S, "str_len", [num_operand(np.arange(3))], [_Op(0)], f"{spec}:numeric")
        fn_case(S, "str_len", [op, target(np.zeros((2, 4), dtype="i8"))], [_Op(0)], f"{spec}:outbad", kw={"out": 1})
        rhs = fn_operand(spec, "c2", seed=20)
        for name in ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"]:
            fn_case(S, name, [op, rhs], [_Op(0), _Op(1)], f"{spec}:compare")
            fn_case(S, name, [op], [_Op(0), t("ab")], f"{spec}:compare-text")


# The texts of section tfn: the P2.5 pool's rules plus the cell kinds of StringDType — short (<= 15 UTF-8 bytes), medium
# (16 - 255) and long (> 255) — trailing NULs (StringDType keeps them: they never count for a length, and NumPy's UTF-8
# loops handle them in their own ways, see NDVStringMethods), and multi-byte characters.
TFN_POOL = ["", " ", "a", "ab", "abc", "aaaa", "abab", "  x y  ", "\tx\n", "x\ty\tz", "-12", "+3.5", "-", "a\x00b", "ab\x00",
            "é日本", "\U0001F600x", "Hello World", "\x1cz\x1f", "\xa0a　", "abcabcab" * 3, "q" * 20, "m" * 15,
            "s" * 300, "  " + "é" * 10 + "  ", "zz\r\nq", "\U0001F600" * 5, "-" + "9" * 20]
# zfill's texts: no empty text and no text that is only NULs (NumPy reads the byte after the written buffer for a sign: an
# arena / heap neighbour, see _fn_nonempty); tail-match needles: no trailing NULs (NumPy compares the needle's full bytes and
# reads past a shorter haystack); replace's old: never only NULs (NumPy sizes its buffer by bytes and writes past it).
TFN_ZPOOL = [v for v in TFN_POOL if v.strip("\x00")]
TFN_NEEDLES = [v for v in TFN_POOL if not v.endswith("\x00")]
# Slices whose step NumPy's StringDType loop can walk: its outsize loop counts with a C int (`int i += step`), which a step
# past 2^31 overflows (an endless loop on 2.4.2), so tfn keeps the steps in range.
TFN_SLICES = [a for a in _FN_SLICES if not (len(a) == 3 and a[2] is not None and abs(a[2]) >= 2**31)]


def _t_scalar(spec, text):
    """A 0-d StringDType operand of ``spec`` holding ``text`` (or the NA object).

    :param spec: the TSpec.
    :param text: the value.
    :returns: the (base, view) pair.
    """
    base = np.array([text], dtype=spec.build())
    return base, base.reshape(())


def gen_tfn():
    """Section tfn: numpy.strings' P2.5 functions over NEP 55 StringDType, strict — every function x every NA kind
    (none, None, NaN, a string sentinel, coerce off) x the layout kinds, the U operands the promoters cast, the S operands
    with no loop, two descriptors (the compatible NA object, or NumPy's TypeError), NumPy's UTF-8 rules (code points,
    trailing NULs, the strip quirks, slice's buffer end), the cell kinds of the results (short / arena / heap), str_len's
    mask (an NA where the mask is off never raises), and the refusals."""
    S = "tfn"
    layouts = ["c1", "c2", "f2", "t3", "s1", "r2", "o1", "col", "bc", "z0", "one", "d5", "e1", "e2"]
    for spec in T_COMPUTE_SPECS:
        lab0 = _t_label(spec)
        for layout in layouts:
            b, v = t_operand(spec, layout, seed=1, pool=TFN_POOL, na_every=5)
            op = (b, v)
            lab = f"{lab0}:{layout}"
            fn_case(S, "str_len", [op], [_Op(0)], lab)
            for name in ["find", "rfind", "count", "index", "rindex"]:
                fn_case(S, name, [op], [_Op(0), "ab"], lab)
                fn_case(S, name, [op], [_Op(0), "a", 1, -1], lab)
                fn_case(S, name, [op], [_Op(0), "b\x00"], lab)
            for name in ["startswith", "endswith"]:
                fn_case(S, name, [op], [_Op(0), "a"], lab)
                fn_case(S, name, [op], [_Op(0), "é", 0, 3], lab)
            for name in ["strip", "lstrip", "rstrip"]:
                fn_case(S, name, [op], [_Op(0)], lab)
                fn_case(S, name, [op], [_Op(0), " aé"], lab)
            for name in ["center", "ljust", "rjust"]:
                fn_case(S, name, [op], [_Op(0), 9], lab)
                fn_case(S, name, [op], [_Op(0), 30, "é"], lab)
            zb, zv = t_operand(spec, layout, seed=1, pool=TFN_ZPOOL, na_every=5)
            fn_case(S, "zfill", [(zb, zv)], [_Op(0), 7], lab)
            fn_case(S, "zfill", [(zb, zv)], [_Op(0), 25], lab)
            fn_case(S, "expandtabs", [op], [_Op(0)], lab)
            fn_case(S, "expandtabs", [op], [_Op(0), 3], lab)
            fn_case(S, "replace", [op], [_Op(0), "a", "XY"], lab)
            fn_case(S, "replace", [op], [_Op(0), "", "-", 3], lab)
            fn_case(S, "replace", [op], [_Op(0), "ab", "é" * 9], lab)
            sep = _t_scalar(spec, "b")
            for name in ["partition", "rpartition"]:
                fn_case(S, name, [op, sep], [_Op(0), _Op(1)], lab)
            fn_case(S, "slice", [op], [_Op(0), 3], lab)
            fn_case(S, "slice", [op], [_Op(0), 1, None, 2], lab)
            fn_case(S, "slice", [op], [_Op(0), None, None, -1], lab)
            fn_case(S, "slice", [op], [_Op(0), 2, 100], lab)

    # B. Parameter sweeps over a contiguous operand of each NA kind: needles x ranges, the strip sets, the pad widths /
    #    fills, zfill's widths, tab sizes, replace's forms, separators, slices.
    for spec in T_COMPUTE_SPECS:
        lab0 = _t_label(spec)
        b, v = t_operand(spec, "c2", seed=3, pool=TFN_POOL, na_every=6)
        op = (b, v)
        for name in _FN_FIND:
            for sub in ["a", "ab", "", "abc", "x y", "\x00", "é", "日本", "\U0001F600", "q" * 18, "Z"]:
                for rng in [(), (1,), (-3,), (0, 3), (2, -1), (5, 2), (1, None), (0, 2**62)]:
                    fn_case(S, name, [op], [_Op(0), sub, *rng], f"{lab0}:{sub!r}:{rng}")
        for name in ["find", "rfind", "count", "index", "rindex"]:
            fn_case(S, name, [op], [_Op(0), "ab\x00"], f"{lab0}:trailingnul")
            fn_case(S, name, [op], [_Op(0), "\x00\x00"], f"{lab0}:nuls")
        for name in ["strip", "lstrip", "rstrip"]:
            for chars in [None, " ", "a", "ab", "", " \t\n", "\x00", "xa ", "é", "a\x00", "\x1c", "日é", "\U0001F600"]:
                fn_case(S, name, [op], [_Op(0), chars], f"{lab0}:{chars!r}")
        for name in ["center", "ljust", "rjust"]:
            for width in [-1, 0, 3, 8, 16, 300]:
                fn_case(S, name, [op], [_Op(0), width], f"{lab0}:{width}")
            for fill in ["*", "é", "\U0001F600", "ab", "", "\x00"]:
                fn_case(S, name, [op], [_Op(0), 12, fill], f"{lab0}:fill{fill!r}")
        zb, zv = t_operand(spec, "c2", seed=3, pool=TFN_ZPOOL, na_every=6)
        for width in [-2, 0, 5, 15, 16, 40]:
            fn_case(S, "zfill", [(zb, zv)], [_Op(0), width], f"{lab0}:{width}")
        for size in [0, 1, 4, 8, -1, 13]:
            fn_case(S, "expandtabs", [op], [_Op(0), size], f"{lab0}:{size}")
        for old in ["a", "ab", "", "é", "q", "b"]:
            for new in ["", "Q", "é" * 8, "longer than fifteen!"]:
                for count in [(), (0,), (1,), (-1,), (-5,)]:
                    # NumPy 2.4.2's StringDType loop sizes its buffer as size + count * growth with a size_t product: a
                    # negative count other than -1 (which replaces nothing there) wraps it, the calloc fails and the copy
                    # writes through NULL — an access violation whenever the replacement is longer than the old text.
                    # NumSharp's copy of the text is pinned by the unit tests.
                    if count == (-5,) and old and len(new.encode()) > len(old.encode()):
                        continue
                    fn_case(S, "replace", [op], [_Op(0), old, new, *count], f"{lab0}:{old!r}->{new!r}:{count}")
        for name in ["partition", "rpartition"]:
            for text in ["a", "b", "ab", " ", "é", "q" * 5, "zz", "\x00", ""]:
                fn_case(S, name, [op, _t_scalar(spec, text)], [_Op(0), _Op(1)], f"{lab0}:{text!r}")
        for args in TFN_SLICES:
            fn_case(S, "slice", [op], [_Op(0), *args], f"{lab0}:{args}")

    # C. U operands beside StringDType (the promoters cast them to StringDType()), S operands (no loop), two descriptors
    #    (compatible NA objects, or NumPy's TypeError), a descriptor whose NA reads as a string beside one that does not.
    for spec in T_COMPUTE_SPECS:
        lab0 = _t_label(spec)
        b, v = t_operand(spec, "c2", seed=4, pool=TFN_POOL, na_every=4)
        op = (b, v)
        u = plain(np.array([["a", "é", "", "ab"]] * 3, dtype="U2"))
        s = plain(np.array([[b"a", b"b", b"", b"ab"]] * 3, dtype="S2"))
        for name in _FN_FIND:
            fn_case(S, name, [op, u], [_Op(0), _Op(1)], f"{lab0},U")
            fn_case(S, name, [op, u], [_Op(1), _Op(0)], f"U,{lab0}")
            fn_case(S, name, [op, s], [_Op(0), _Op(1)], f"{lab0},S")
        for name in ["strip", "lstrip", "rstrip"]:
            fn_case(S, name, [op, u], [_Op(0), _Op(1)], f"{lab0},U")
            fn_case(S, name, [op, u], [_Op(1), _Op(0)], f"U,{lab0}")
            fn_case(S, name, [op, s], [_Op(0), _Op(1)], f"{lab0},S")
        for name in ["center", "ljust", "rjust"]:
            fn_case(S, name, [op, plain(np.array("*", dtype="U1"))], [_Op(0), 6, _Op(1)], f"{lab0},Ufill")
            fn_case(S, name, [op, plain(np.array(b"*", dtype="S1"))], [_Op(0), 6, _Op(1)], f"{lab0},Sfill")
            fn_case(S, name, [u, _t_scalar(spec, "*")], [_Op(0), 6, _Op(1)], f"U,{lab0}fill")
        fn_case(S, "replace", [op, u], [_Op(0), _Op(1), "Z"], f"{lab0},Uold")
        fn_case(S, "replace", [op, u], [_Op(1), "a", _Op(0)], f"U,{lab0}new")
        fn_case(S, "replace", [op, s], [_Op(0), _Op(1), "Z"], f"{lab0},Sold")
        for name in ["partition", "rpartition"]:
            fn_case(S, name, [op, u], [_Op(0), _Op(1)], f"{lab0},U")
            fn_case(S, name, [op], [_Op(0), "b"], f"{lab0},str")
            fn_case(S, name, [op, u], [_Op(1), _Op(0)], f"U,{lab0}")
        for other in T_COMPUTE_SPECS:
            ob, ov = t_operand(other, "c2", seed=5, pool=TFN_POOL, na_every=3)
            nb, nv = t_operand(other, "c2", seed=5, pool=TFN_NEEDLES, na_every=3)
            pair = f"{lab0},{_t_label(other)}"
            fn_case(S, "find", [op, (ob, ov)], [_Op(0), _Op(1)], pair)
            fn_case(S, "startswith", [op, (nb, nv)], [_Op(0), _Op(1)], pair)
            fn_case(S, "strip", [op, (ob, ov)], [_Op(0), _Op(1)], pair)
            fn_case(S, "replace", [op, (ob, ov)], [_Op(0), _Op(1), "Z"], pair)
            fn_case(S, "center", [op, _t_scalar(other, "*")], [_Op(0), 6, _Op(1)], pair)
            fn_case(S, "partition", [op, _t_scalar(other, "b")], [_Op(0), _Op(1)], pair)

    # D. str_len with where= (an NA where the mask is off is never read) and out= of other dtypes; the refusals of the
    #    integer parameters and of the shapes.
    for spec in [TDN, TDNAN, TDS, TD]:
        lab0 = _t_label(spec)
        b, v = t_operand(spec, "c2", seed=6, pool=TFN_POOL, na_every=4)
        op = (b, v)
        na_mask = plain(np.array([[(i * 4 + j) % 4 != 3 for j in range(4)] for i in range(3)]))
        out = target(np.full((3, 4), -7, dtype="i8"))
        fn_case(S, "str_len", [op, out, na_mask], [_Op(0)], f"{lab0}:where", kw={"out": 1, "where": 2})
        for odt in ["i4", "f8", "?"]:
            fn_case(S, "str_len", [op, target(np.zeros((3, 4), dtype=odt))], [_Op(0)], f"{lab0}:out:{odt}", kw={"out": 1})
        fn_case(S, "find", [op], [_Op(0), "a", 1.5], f"{lab0}:float")
        fn_case(S, "find", [op], [_Op(0), "a", 2**63], f"{lab0}:bigint")
        fn_case(S, "find", [op, plain(np.array(["a"] * 5))], [_Op(0), _Op(1)], f"{lab0}:nobroadcast")
        fn_case(S, "center", [op], [_Op(0), 2.5], f"{lab0}:floatwidth")
        fn_case(S, "center", [op, plain(np.array([3, 4, 5, 6], dtype="u8"))], [_Op(0), _Op(1)], f"{lab0}:u8width")
        fn_case(S, "zfill", [op, plain(np.array([3, 4, 5, 6], dtype="u8"))], [_Op(0), _Op(1)], f"{lab0}:u8width")
        fn_case(S, "expandtabs", [op], [_Op(0), 2.5], f"{lab0}:floattab")
        fn_case(S, "replace", [op], [_Op(0), "a", "b", 1.5], f"{lab0}:floatcount")
        fn_case(S, "replace", [op, plain(np.array([1, 2, 3, 4], dtype="u8"))], [_Op(0), "a", "b", _Op(1)], f"{lab0}:u8count")
        fn_case(S, "slice", [op], [_Op(0), 0, 5, 0], f"{lab0}:zero")
        fn_case(S, "slice", [op], [_Op(0), 1.5], f"{lab0}:floatstop")
        tab = t_plain(np.array(["\t" * 3, "a\tb"], dtype=spec.build()))
        fn_case(S, "expandtabs", [tab], [_Op(0), 2**31], f"{lab0}:overflow")


# ----------------------------------------------------------------------------------------------------------------------
# numpy.strings' predicates (strprop over S / U, tprop over StringDType), strings phase 2, P2.6, strict
# ----------------------------------------------------------------------------------------------------------------------

PROP_NAMES = ["isalpha", "isdigit", "isspace", "isalnum", "islower", "isupper", "istitle", "isdecimal", "isnumeric"]
# The texts that separate the predicates: each class alone and mixed, Python's extra whitespace (U+001C, NEL, NBSP, the
# ideographic space), digits that are not decimal ('²'), numerics that are not digits ('½', 'Ⅻ', '万'), titlecase letters
# ('ǅ'), cased characters that are not letters ('ʰ' is lowercase, 'Ⅻ' uppercase), embedded / trailing NULs, astral
# letters ('𐐀' / '𐐨', an uppercase / lowercase pair) and digits ('𝟘'), and the empty string.
PROP_U_POOL = ["", "a", "abc", "ABC", "Abc", "aBc", "abc1", "123", "1", "٣", "²", "½", "Ⅻ", "万", " ", "\t\n", "\x1c",
               "\x85", "\xa0", "　", "a b", "Hello World", "Hello world", "HELLO", "hello", "ǅ", "ǅa", "ǈungla", "ʰ",
               "ª", "ß", "é", "Éa", "a\x00b", "ab\x00", "\x00", "\U0001F600", "\U00010400", "\U00010428", "𝟘",
               "x y Z", "It's", "A1b", "1A", "_", "-", "!", "A B c", "Ab1 Cd", "ΑΣ"]
PROP_S_POOL = [b"", b"a", b"abc", b"ABC", b"Abc", b"aBc", b"abc1", b"123", b"1", b" ", b"\t\n", b"\x0b\x0c", b"\x1c",
               b"\x85", b"\xe9", b"\xc9a", b"a b", b"Hello World", b"Hello world", b"HELLO", b"hello", b"a\x00b",
               b"ab\x00", b"\x00", b"1A", b"A1b", b"It's", b"\x80\xff", b"_", b"A B c", b"Ab1 Cd"]
PROP_SPECS = ["<U5", "<U12", "<U70", "|S5", "|S12", "|S70"]
# The unit widths of the width sweep: every lane-mask chunk class (below 8 / 16 / 32 bytes, whole and overlapping
# 32-byte chunks, the 64-unit cap) and the stream past it.
PROP_WIDTHS = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 40, 63, 64, 65, 66, 70, 96, 100, 129, 300]
# Character classes the random texts are drawn from (U; the S sweep keeps the ASCII ones).
_PROP_ALPHABETS = {
    "lower": "abcdefghijklmnopqrstuvwxyz",
    "upper": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "letters": "abcXYZqwERTyu",
    "digits": "0123456789",
    "spaces": " \t\n\r\x0b\x0c",
    "alnum": "ab1C9z0",
    "mixed": "aZ 1!\x00é",
    "uni_letters": "éßΩωЖжǅ",
    "uni_digits": "٣٤²½Ⅻ万𝟘",
    "uni_spaces": " \x1c\x1f\x85\xa0　 ",
    "uni_case": "aAéÉǅʰªΩω𐐀𐐨",
}


def _prop_text(rng, alphabet, length):
    """A random text of ``length`` characters from ``alphabet``.

    :param rng: the random.Random.
    :param alphabet: the characters.
    :param length: the length.
    :returns: the text.
    """
    return "".join(rng.choice(alphabet) for _ in range(length))


def _prop_title_text(rng, length, unicode):
    """A random text of words, each title-cased or not, separated by spaces or digits — istitle's interesting inputs.

    :param rng: the random.Random.
    :param length: the target length.
    :param unicode: whether non-ASCII cased letters and titlecase letters may appear.
    :returns: the text (cut to ``length``).
    """
    lowers = "abcdefgh" + ("éωж" if unicode else "")
    uppers = "ABCDEFGH" + ("ÉΩЖǅ" if unicode else "")
    out = []
    while len("".join(out)) < length:
        word = rng.choice(uppers) + "".join(rng.choice(lowers) for _ in range(rng.randint(0, 6)))
        r = rng.random()
        if r < 0.15:
            word = word.lower()
        elif r < 0.25:
            word = word[0] + word[1:].upper()
        out.append(word)
        out.append(rng.choice([" ", " ", "1", "-", "'"]))
    return "".join(out)[:length]


def _prop_texts(rng, width, n, unicode):
    """``n`` texts for a width-sweep array: each array draws its texts from one character class (so every-unit predicates
    are often True), some perturbed by one character of another class at a random position (any lane, any chunk), some
    title-cased words, some empty, some with trailing NULs.

    :param rng: the random.Random.
    :param width: the unit width.
    :param n: the count.
    :param unicode: whether the non-ASCII classes may appear.
    :returns: the texts.
    """
    names = [k for k in _PROP_ALPHABETS if unicode or not k.startswith("uni")]
    texts = []
    for i in range(n):
        kind = rng.random()
        length = rng.choice([width, width, width, rng.randint(0, width)])
        if kind < 0.25:
            t = _prop_title_text(rng, length, unicode)
        else:
            alphabet = _PROP_ALPHABETS[rng.choice(names)]
            t = _prop_text(rng, alphabet, length)
            if t and rng.random() < 0.5:
                pos = rng.randrange(len(t))
                other = _PROP_ALPHABETS[rng.choice(names)]
                t = t[:pos] + rng.choice(other) + t[pos + 1:]
        if t and rng.random() < 0.1:
            t = t[:-1] + "\x00"
        texts.append(t)
    return texts


def _prop_bytes(texts):
    """The S form of sweep texts: each character's code point modulo 256 as a byte (latin-1 when it fits).

    :param texts: the texts.
    :returns: the bytes values.
    """
    return [bytes(ord(c) % 256 for c in t) for t in texts]


def gen_strprop():
    """Section strprop: numpy.strings' predicate ufuncs over S / U, strict — the nine predicates (S has no isdecimal /
    isnumeric loop) over every layout kind and three widths of both classes; a width sweep of random texts over every
    lane-mask chunk class and the stream past 64 units, all-ASCII arrays and arrays where a non-ASCII element comes first,
    midway or last (the call's escalation); out= of other dtypes, where= masks, dtype= requests and the refusals in
    NumPy's order (out / where parsing, no loop, the output cast, the broadcast); units past U+10FFFF (record 0)."""
    import random
    S = "strprop"
    # A. Every predicate x every layout x three widths of both classes.
    for spec in PROP_SPECS:
        pool = PROP_U_POOL if spec[1] == "U" else PROP_S_POOL
        for layout in LAYOUTS:
            op = fn_operand(spec, layout, seed=3, pool=pool)
            for name in PROP_NAMES:
                fn_case(S, name, [op], [_Op(0)], f"{spec}:{layout}")
    # B. The pools whole, one array per predicate and width (every text of the pool in one call).
    for spec in PROP_SPECS:
        pool = PROP_U_POOL if spec[1] == "U" else PROP_S_POOL
        op = plain(np.array(pool, dtype=spec))
        for name in PROP_NAMES:
            fn_case(S, name, [op], [_Op(0)], f"{spec}:pool")
    # C. The width sweep: random texts per width, U (Unicode classes) and S (ASCII classes), plus arrays whose first /
    #    middle / last element is the first non-ASCII one (the escalation mid-call) and all-ASCII U arrays.
    rng = random.Random(2606)
    for w in PROP_WIDTHS:
        for rep in range(3):
            utexts = _prop_texts(rng, w, 24, unicode=True)
            atexts = _prop_texts(rng, w, 24, unicode=False)
            ops = [(f"<U{w}", plain(np.array(utexts, dtype=f"<U{w}")), "uni"),
                   (f"<U{w}", plain(np.array(atexts, dtype=f"<U{w}")), "ascii"),
                   (f"|S{w}", plain(np.array(_prop_bytes(atexts), dtype=f"|S{w}")), "ascii"),
                   (f"|S{w}", plain(np.array(_prop_bytes(utexts), dtype=f"|S{w}")), "bytes")]
            for spec, op, tag in ops:
                for name in PROP_NAMES:
                    fn_case(S, name, [op], [_Op(0)], f"{spec}:sweep{rep}:{tag}")
        for where_first in (0, 11, 23):
            texts = _prop_texts(rng, w, 24, unicode=False)
            texts[where_first] = "é" + texts[where_first][1:]
            op = plain(np.array(texts, dtype=f"<U{w}"))
            for name in PROP_NAMES:
                fn_case(S, name, [op], [_Op(0)], f"<U{w}:escalate@{where_first}")
    # D. out= / where= / dtype= and the refusals, over a contiguous 2-D operand of each class.
    for spec in ["<U5", "|S5"]:
        pool = PROP_U_POOL if spec[1] == "U" else PROP_S_POOL
        op = fn_operand(spec, "c2", seed=5, pool=pool)
        mask = plain(np.array([[True, False, True, True], [False, False, True, False], [True, True, True, False]]))
        for name in ["isalpha", "istitle", "isspace", "isdecimal"]:
            for odt in ["?", "i1", "u8", "f2", "f8", "c16", "U1", "U5", "S1", "S5"]:
                fn_case(S, name, [op, target(np.zeros((3, 4), dtype=odt))], [_Op(0)], f"{spec}:out:{odt}", kw={"out": 1})
            fn_case(S, name, [op, target(np.full((3, 4), 7, dtype="i1")), mask], [_Op(0)], f"{spec}:where",
                    kw={"out": 1, "where": 2})
            fn_case(S, name, [op, target(np.full((3, 4), True)), mask], [_Op(0)], f"{spec}:wherebool",
                    kw={"out": 1, "where": 2})
            # (no "O": NumSharp has no object dtype to pass)
            for dt in ["?", "i8", "u1", "f8", "U1", "U", "S", ">?", ">i8", "M8", "M8[s]", "T"]:
                fn_case(S, name, [op], [_Op(0)], f"{spec}:dtype:{dt}", kw={"dtype": dt})
            ro_where = plain(np.zeros((3, 4), dtype="?"))
            ro_where[1].flags.writeable = False
            fn_case(S, name, [op, ro_where], [_Op(0)], f"{spec}:readonly+dtype", kw={"out": 1, "dtype": "U1"})
            fn_case(S, name, [op, plain(np.zeros((3, 4), dtype="i8"))], [_Op(0)], f"{spec}:whereint+dtype",
                    kw={"where": 1, "dtype": "U1"})
            ro = plain(np.zeros((3, 4), dtype="?"))
            ro[1].flags.writeable = False
            fn_case(S, name, [op, ro], [_Op(0)], f"{spec}:readonly", kw={"out": 1})
            fn_case(S, name, [op, plain(np.zeros((3, 4), dtype="i8"))], [_Op(0)], f"{spec}:where-int",
                    kw={"where": 1})
            fn_case(S, name, [op, target(np.zeros((2, 4), dtype="?"))], [_Op(0)], f"{spec}:outshape", kw={"out": 1})
            fn_case(S, name, [op, plain(np.array([True, False, True]))], [_Op(0)], f"{spec}:whereshape", kw={"where": 1})
            fn_case(S, name, [op, target(np.zeros((2, 3, 4), dtype="?"))], [_Op(0)], f"{spec}:outbroadcast",
                    kw={"out": 1})
    # E. Non-text operands (no loop), Python strs / bytes as the operand, 0-d operands.
    for name in PROP_NAMES:
        for arr in [np.array([1, 2]), np.array([1.5]), np.array([True]), np.zeros((2, 0), dtype="i4")]:
            fn_case(S, name, [plain(arr)], [_Op(0)], f"num:{arr.dtype}")
        fn_case(S, name, [], ["Abc"], "pystr")
        fn_case(S, name, [], [b"Abc"], "pybytes")
        fn_case(S, name, [], ["  "], "pystr:space")
    # F. Units past U+10FFFF (a U array over raw code units): CPython's record 0, no property.
    raw = np.frombuffer(np.array([0x110000, 0x61, 0x61, 0xFFFFFFFF, 0x41, 0x110000, 0x61, 0x62, 0x10FFFF, 0x0, 0, 0],
                                 dtype="<u4").tobytes(), dtype="<U3")
    for name in PROP_NAMES:
        fn_case(S, name, [plain(raw.copy())], [_Op(0)], "U:past-unicode")


# StringDType predicate texts: short (in the cell) and long (arena), ASCII and not, trailing / embedded NULs, the classes.
PROP_T_POOL = ["", "a", "abc", "ABC", "Abc", "aBc", "123", "٣", "²", "½", " \t", "\x1c", "　", "Hello World", "Hello world",
               "ǅa", "ʰ", "é", "a\x00b", "ab\x00", "\U00010400", "Hello World From A Long Title Text", "a" * 40,
               "A" * 33 + "b", "1" * 20, " " * 18 + "\x85", "ΑΣ ΒΣ ΓΣ ΔΣ ΕΣ ΖΣ", "x" * 15, "x" * 16, "Abc Def Ghi Jkl Mno",
               "日本語のテキスト", "Ünïcödé Wörds Here"]


def gen_tprop():
    """Section tprop: numpy.strings' predicate ufuncs over NEP 55 StringDType, strict — every predicate x every NA kind
    (none, None, NaN, a string sentinel, coerce off) x the layout kinds over short / long / ASCII / non-ASCII texts; the
    NA rules (a NaN-like NA is False, a string NA reads as its string, any other NA raises — but only where the mask
    selects it); out= of other dtypes, dtype= requests; a sweep of random texts by length (the short-cell vector, the
    32-byte chunk stream, the decoded path)."""
    import random
    S = "tprop"
    layouts = ["c1", "c2", "f2", "t3", "s1", "r2", "o1", "col", "bc", "z0", "one", "d5", "e1", "e2"]
    for spec in T_COMPUTE_SPECS:
        lab0 = _t_label(spec)
        for layout in layouts:
            op = t_operand(spec, layout, seed=1, pool=PROP_T_POOL, na_every=5)
            for name in PROP_NAMES:
                fn_case(S, name, [op], [_Op(0)], f"{lab0}:{layout}")
        # The whole pool in one call per NA kind (no NA) and with an NA every 3rd element.
        for na_every in (0, 3):
            op = (np.array(t_values(spec, len(PROP_T_POOL), 0, PROP_T_POOL, na_every), dtype=spec.build()), None)
            op = (op[0], op[0])
            for name in PROP_NAMES:
                fn_case(S, name, [op], [_Op(0)], f"{lab0}:pool:na{na_every}")
    # The mask: an NA in a masked-off position is never read (no error even under a refusing NA kind); out= of other
    # dtypes; dtype= requests.
    for spec in [TDN, TDNAN, TDS, TD]:
        lab0 = _t_label(spec)
        b, v = t_operand(spec, "c2", seed=6, pool=PROP_T_POOL, na_every=4)
        op = (b, v)
        na_mask = plain(np.array([[(i * 4 + j) % 4 != 1 for j in range(4)] for i in range(3)]))
        for name in ["isalpha", "istitle", "isspace", "isnumeric"]:
            fn_case(S, name, [op, target(np.full((3, 4), True)), na_mask], [_Op(0)], f"{lab0}:where",
                    kw={"out": 1, "where": 2})
            fn_case(S, name, [op, target(np.full((3, 4), 5, dtype="i2")), na_mask], [_Op(0)], f"{lab0}:where:i2",
                    kw={"out": 1, "where": 2})
            for odt in ["i4", "f8", "U2"]:
                fn_case(S, name, [op, target(np.zeros((3, 4), dtype=odt))], [_Op(0)], f"{lab0}:out:{odt}",
                        kw={"out": 1})
            for dt in ["?", "i8", "U1", "T"]:
                fn_case(S, name, [op], [_Op(0)], f"{lab0}:dtype:{dt}", kw={"dtype": dt})
    # The length sweep: random texts of every length class (short cells up to 15 bytes, the 16 / 31 / 32 / 33 byte
    # boundaries of the stream, long texts), ASCII and not.
    rng = random.Random(2607)
    for spec in [TD, TDS]:
        lab0 = _t_label(spec)
        for w in [0, 1, 3, 7, 8, 14, 15, 16, 17, 31, 32, 33, 40, 63, 64, 65, 100, 300]:
            for unicode in (False, True):
                texts = _prop_texts(rng, w, 24, unicode)
                op = (np.array(texts, dtype=spec.build()), None)
                op = (op[0], op[0])
                for name in PROP_NAMES:
                    fn_case(S, name, [op], [_Op(0)], f"{lab0}:len{w}:{'uni' if unicode else 'ascii'}")


# ----------------------------------------------------------------------------------------------------------------------
# numpy.strings' case maps (strcase over S / U, tcase over StringDType), strings phase 2, P2.6 milestone 2, strict
# ----------------------------------------------------------------------------------------------------------------------

CASE_NAMES = ["upper", "lower", "swapcase", "capitalize", "title"]
# The texts that separate the maps: ASCII words with apostrophes and digits (title's "any uncased character starts a
# word"), leading non-letters (capitalize), the full mappings that grow ('ß' -> 'SS', ligatures, 'ŉ', 'ΐ', 'ᾳ', 'İ'), the
# titlecase digraphs ('ǆ' / 'ǅ' / 'Ǆ', 'ǉ'), the final sigma's contexts (case-ignorable marks around it, a NUL after it),
# cased characters that are not letters ('ʰ', 'ª', U+0345), mappings that change the UTF-8 size ('ı', 'ſ', 'ⱥ', 'ɐ'),
# Cherokee (far deltas), astral letters, embedded / leading / trailing NULs and the empty string.
CASE_U_POOL = ["", "a", "abc", "ABC", "Abc", "aBc", "hello world", "Hello World", "HELLO", "they're bill's",
               "hello2world", "a1b2", "1abc", " lead", "trail ", "ß", "straße", "ﬃ", "ﬁx", "ŉ", "ǆ", "ǅ", "Ǆ",
               "ǉubljana", "ΐ", "ᾳ", "ᾀ", "ᾈ", "İ", "ı", "ſ", "ⱥ", "ɐ", "ΑΣ", "ΌΣΟΣ", "aΣ", "aΣa", "aΣ́",
               "aΣ́b", ".Σ", "ΣΣ", "aΣͅb", "ʰa", "ª", "é", "Éa", "ÀÉÎÕÜ", "àéîõü", "Ωω", "Жж", "a\x00b",
               "\x00ab", "ab\x00", "\U00010400\U00010428", "\U0001D400", "\U0001F600a", "xͅy", "ǈungla Ǉ", "ﬀ",
               "ﬓ", "ẞ", "ꭰ", "Ꭰ", "ㄱ", "٣a", "²a", "_x_", "-a-b", "aΣ\x00"]
CASE_S_POOL = [b"", b"a", b"abc", b"ABC", b"Abc", b"aBc", b"hello world", b"Hello World", b"they're bill's",
               b"hello2world", b"1abc", b" lead", b"\xe9t\xe9", b"\xc9A", b"a\x00b", b"\x00ab", b"ab\x00", b"\x80\xff",
               b"@[`{", b"_x_", b"A B c", b"aZ09 zA"]
CASE_SPECS = ["<U5", "<U12", "<U70", "|S5", "|S12", "|S70"]
# The unit widths of the width sweep: one padded chunk (up to 32 bytes), whole chunks plus an overlapping last one, the
# 64-unit lane cap and past it.
CASE_WIDTHS = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 40, 63, 64, 65, 70, 96, 100, 129, 300]
_CASE_ALPHABETS = {
    "lower": "abcdefghijklmnopqrstuvwxyz",
    "upper": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "mixed": "aBcDeF gH1!'.\x00",
    "letters": "abcXYZqwERTyu",
    "uni": "éÉßẞΩωЖжǅǆǄıſⱥɐ",
    "grow": "ßﬃŉΐᾳİabc",
    "sigma": "ΑΣαΣ́ͅ. ",
    "astral": "\U00010400\U00010428a",
}


def _case_texts(rng, width, n, unicode):
    """``n`` texts for a case-map sweep array: each drawn from one character class (some title-cased words), some cut
    short, some with a trailing NUL; the S form keeps the ASCII classes.

    :param rng: the random.Random.
    :param width: the unit width.
    :param n: the count.
    :param unicode: whether the non-ASCII classes may appear.
    :returns: the texts.
    """
    names = [k for k in _CASE_ALPHABETS if unicode or k in ("lower", "upper", "mixed", "letters")]
    texts = []
    for i in range(n):
        length = rng.choice([width, width, width, rng.randint(0, width)])
        if rng.random() < 0.25:
            t = _prop_title_text(rng, length, unicode)
        else:
            t = _prop_text(rng, _CASE_ALPHABETS[rng.choice(names)], length)
        if t and rng.random() < 0.1:
            t = t[:-1] + "\x00"
        texts.append(t)
    return texts


def gen_strcase():
    """Section strcase: numpy.strings' case maps over S / U, strict — upper, lower, swapcase, capitalize and title
    (``_vec_string``: a fresh C-ordered array of the operand's dtype, each element the bytes / str method's result cut to
    the width) over every layout kind and three widths of both classes; a width sweep over every chunk class and the
    stream past 64 units, with all-ASCII arrays and arrays whose first non-ASCII element comes first, midway or last (the
    call's escalation); the full mappings cut at the width; the final sigma; U units past U+10FFFF (CPython's
    SystemError); lone surrogates; Python strs / bytes as the operand; non-text operands."""
    import random
    S = "strcase"
    # A. Every map x every layout x three widths of both classes.
    for spec in CASE_SPECS:
        pool = CASE_U_POOL if spec[1] == "U" else CASE_S_POOL
        for layout in LAYOUTS:
            op = fn_operand(spec, layout, seed=3, pool=pool)
            for name in CASE_NAMES:
                fn_case(S, name, [op], [_Op(0)], f"{spec}:{layout}")
    # B. The pools whole, per map and width (every text in one call; the narrow widths cut the mappings that grow).
    for spec in CASE_SPECS + ["<U1", "<U2", "<U3", "|S1", "|S2"]:
        pool = CASE_U_POOL if spec[1] == "U" else CASE_S_POOL
        op = plain(np.array(pool, dtype=spec))
        for name in CASE_NAMES:
            fn_case(S, name, [op], [_Op(0)], f"{spec}:pool")
    # C. The width sweep: random texts per width, U (every class) and S (the ASCII classes), plus U arrays whose first
    #    non-ASCII element is first, midway or last.
    rng = random.Random(2616)
    for w in CASE_WIDTHS:
        for rep in range(2):
            utexts = _case_texts(rng, w, 24, unicode=True)
            atexts = _case_texts(rng, w, 24, unicode=False)
            ops = [(f"<U{w}", plain(np.array(utexts, dtype=f"<U{w}")), "uni"),
                   (f"<U{w}", plain(np.array(atexts, dtype=f"<U{w}")), "ascii"),
                   (f"|S{w}", plain(np.array(_prop_bytes(atexts), dtype=f"|S{w}")), "ascii"),
                   (f"|S{w}", plain(np.array(_prop_bytes(utexts), dtype=f"|S{w}")), "bytes")]
            for spec, op, tag in ops:
                for name in CASE_NAMES:
                    fn_case(S, name, [op], [_Op(0)], f"{spec}:sweep{rep}:{tag}")
        for where_first in (0, 11, 23):
            texts = _case_texts(rng, w, 24, unicode=False)
            texts[where_first] = "é" + texts[where_first][1:]
            op = plain(np.array(texts, dtype=f"<U{w}"))
            for name in CASE_NAMES:
                fn_case(S, name, [op], [_Op(0)], f"<U{w}:escalate@{where_first}")
    # D. Units past U+10FFFF (a one-unit text raises at the scalar, a longer one at the result; any element does), and
    #    lone surrogates (code points with no property, kept).
    for units, label in [([0x61, 0x62, 0x110000, 0, 0x63, 0], "one-unit"),
                         ([0x61, 0x62, 0x63, 0x110000, 0x41, 0x42], "multi-unit"),
                         ([0x61, 0x110000, 0x63, 0x64, 0x65, 0x66], "first"),
                         ([0x61, 0x62, 0x63, 0x64, 0x65, 0xFFFFFFFF], "last"),
                         ([0xD800, 0x61, 0xDC00, 0x41, 0xDFFF, 0x62], "surrogates"),
                         ([0x10FFFF, 0x61, 0x10FFFF, 0x41, 0, 0], "max")]:
        raw = np.frombuffer(np.array(units, dtype="<u4").tobytes(), dtype="<U2")
        for name in CASE_NAMES:
            fn_case(S, name, [plain(raw.copy())], [_Op(0)], f"U:{label}")
    # E. Python strs / bytes as the operand, 0-d and empty arrays, non-text operands.
    for name in CASE_NAMES:
        fn_case(S, name, [], ["hello World"], "pystr")
        fn_case(S, name, [], [b"hello World"], "pybytes")
        fn_case(S, name, [], ["straße ǅ"], "pystr:uni")
        for arr in [np.array([1, 2]), np.array([1.5]), np.array([True]), np.zeros((2, 0), dtype="i4")]:
            fn_case(S, name, [plain(arr)], [_Op(0)], f"num:{arr.dtype}")


# StringDType case-map texts: short (in the cell) and long (arena / heap), ASCII and not, mappings that change the UTF-8
# size both ways, the final sigma, NULs (characters of a StringDType text), the empty string.
CASE_T_POOL = ["", "a", "abc", "ABC", "Abc", "hello world", "Hello World From A Long Title Text", "they're bill's",
               "ß", "straße", "ﬃx" * 5, "ı" * 8, "ſtraße", "ⱥ" * 10, "ɐ" * 10, "ΌΣΟΣ ΌΣΟΣ", "aΣ", "aΣ\x00", "a\x00",
               "\x00ab", "ǆemal ǈungla", "İstanbul", "x" * 15, "x" * 16, "é" * 20, "\U00010400\U00010428" * 3,
               "日本語のテキスト", "Ünïcödé Wörds Here", "a\x00b", "ᾳ" * 6]


def gen_tcase():
    """Section tcase: numpy.strings' case maps over NEP 55 StringDType, strict — every map x every NA kind (a str method
    refuses an NA object that is not a str; a string NA is computed as its text, and any result equal to it is stored as
    the NA) x the layout kinds over short / long / ASCII / non-ASCII texts, results that change the UTF-8 size, the NA
    equality quirk with sentinels equal to results ('ABC', ''), coerce off, and a length sweep over the short-cell
    vector, the arena chunks and the decoded path."""
    import random
    S = "tcase"
    for spec in T_COMPUTE_SPECS:
        lab0 = _t_label(spec)
        for layout in T_COMPUTE_LAYOUTS:
            op = t_operand(spec, layout, seed=2, pool=CASE_T_POOL, na_every=5)
            for name in CASE_NAMES:
                fn_case(S, name, [op], [_Op(0)], f"{lab0}:{layout}")
        # The whole pool in one call, without NAs and with an NA every 3rd element.
        for na_every in (0, 3):
            base = np.array(t_values(spec, len(CASE_T_POOL), 0, CASE_T_POOL, na_every), dtype=spec.build())
            for name in CASE_NAMES:
                fn_case(S, name, [(base, base)], [_Op(0)], f"{lab0}:pool:na{na_every}")
    # Sentinels equal to some results: setitem stores a result equal to the NA string as the NA.
    for sentinel, values in [("ABC", ["abc", "ABC", "Abc", "x", "aBC", "ABCD"]),
                             ("", ["", "a", "", "B"]),
                             ("Abc", ["abc", "ABC", "aBC", "abC"]),
                             ("SS", ["ß", "ss", "SS", "ſſ"])]:
        spec = TSpec(sentinel)
        base = np.array(values, dtype=spec.build())
        for name in CASE_NAMES:
            fn_case(S, name, [(base, base)], [_Op(0)], f"TS[{sentinel}]:eq")
    # The length sweep: random texts of every length class, ASCII and not.
    rng = random.Random(2617)
    for spec in [TD, TDS]:
        lab0 = _t_label(spec)
        for w in [0, 1, 3, 7, 8, 14, 15, 16, 17, 31, 32, 33, 40, 63, 64, 65, 100, 300]:
            for unicode in (False, True):
                texts = _case_texts(rng, w, 24, unicode)
                base = np.array(texts, dtype=spec.build())
                for name in CASE_NAMES:
                    fn_case(S, name, [(base, base)], [_Op(0)], f"{lab0}:len{w}:{'uni' if unicode else 'ascii'}")


# ----------------------------------------------------------------------------------------------------------------------
# numpy.strings.translate (strtrans over S / U, ttrans over StringDType), strings phase 2, P2.6 milestone 3, strict
# ----------------------------------------------------------------------------------------------------------------------

# str.translate tables (U, StringDType): maketrans maps (ASCII one-to-one, with deletions), replacements that grow or
# delete, values of every kind (None, '', ints, a bool, the maximum code point, a lone surrogate), keys that equal a code
# point (97.0, True) or never do ('a') or lie past the BMP / U+10FFFF, a NUL mapped / produced, and the values that raise
# (a float, an int out of range or huge, bytes) — only for a text holding that key.
TR_U_TABLES = {
    "mk": str.maketrans("abcdef", "uvwxyz"),
    "mkdel": str.maketrans("abc", "xyz", "def"),
    "exp": {97: "XYZ", 98: "Q", 99: ""},
    "none": {97: None, 32: None},
    "uni": {0xE9: "e", 0x4E2D: "Z", 97: "á", 0x10400: "\U00010428"},
    "astral": {0x10400: "x", 99: "\U00010401", 0x1F600: None},
    "keys": {97.0: "z", True: "T", "b": "never", 2**70: "big", 99: True},
    "nul": {0: "Z"},
    "toNul": {97: 0},
    "maxcp": {97: 0x10FFFF},
    "surr": {97: 0xDC00, 98: "\ud800"},
    # A key of 2^31 or more is looked up as a negative C long on Windows (32-bit long) and as itself on Linux / macOS
    # (NumSharp models the 64-bit long): the keys past U+10FFFF stay below 2^31, where every platform agrees.
    "past": {0x110000: "p", 0x7FFFFFF0: None},
    "empty": {},
    "err_float": {97: 3.5},
    "err_range": {97: 0x110000},
    "err_neg": {98: -1},
    "err_big": {99: 2**70},
    "err_bytes": {97: b"z"},
}
# Texts the tables separate: their keys present and absent, non-ASCII and astral characters, NULs inside and at the end,
# lone surrogates, the empty string.
TR_U_POOL = ["", "a", "abc", "cab", "abcdef", "def", "xyz", "aaaa", "a b c", "ABC", "hello world", "é中文", "aé中b",
             "\U00010400\U00010401c", "a\U0001F600b", "a\x00b\x00", "\x00a", "zzz", "ab", "ç", "\ud800a", "Q"]
TR_S_POOL = [b"", b"a", b"abc", b"cab", b"abcdef", b"def", b"xyz", b"aaaa", b"a b c", b"ABC", b"\x00ab", b"a\x00b\x00",
             b"\xe9t\xe9", b"\xff\x80", b"zz", b"hello"]


def _tr_bytes_map(src, dst):
    """bytes.maketrans(src, dst) (a 256-byte table)."""
    return bytes.maketrans(src, dst)


def gen_strtrans():
    """Section strtrans: numpy.strings.translate over S / U, strict — every layout kind of both classes with the main
    tables; every str.translate table kind (dicts of every key / value kind, str / bytes / None / number tables, arrays
    and object lists of tables, broadcast tables) and every bytes.translate table / deletechars kind (256-byte tables,
    wrong lengths incl. a trailing NUL dropped by np.bytes_, np.str_ tables read as UCS-4 bytes, numbers, dicts, None,
    element-wise delete sets); the C-order first error; U units past U+10FFFF mapped, deleted, kept and alone; the gather
    route's width sweep; deletechars ignored for U; Python str / bytes operands, 0-d, empty and non-text operands."""
    import random
    S = "strtrans"
    mk = _tr_bytes_map(b"abcdef", b"uvwxyz")
    rev = bytes(range(256))[::-1]
    # A. Every layout kind x both classes x the main tables.
    for spec in ["<U5", "<U12", "|S5", "|S12"]:
        pool = TR_U_POOL if spec[1] == "U" else TR_S_POOL
        for layout in LAYOUTS:
            op = fn_operand(spec, layout, seed=2, pool=pool)
            lab = f"{spec}:{layout}"
            if spec[1] == "U":
                for tname in ["mk", "mkdel", "exp", "uni"]:
                    fn_case(S, "translate", [op], [_Op(0), TR_U_TABLES[tname]], f"{lab}:{tname}")
            else:
                fn_case(S, "translate", [op], [_Op(0), mk], f"{lab}:mk")
                fn_case(S, "translate", [op], [_Op(0), mk, b"de"], f"{lab}:mkdel")
                fn_case(S, "translate", [op], [_Op(0), None, b"a\x00"], f"{lab}:del")
    # B. Every str.translate table over the U pool (one call each), at three widths.
    for spec in ["<U1", "<U3", "<U12"]:
        op = plain(np.array(TR_U_POOL, dtype=spec))
        for tname, table in TR_U_TABLES.items():
            fn_case(S, "translate", [op], [_Op(0), table], f"{spec}:pool:{tname}")
        for label, table in [("str", "xyz"), ("str100", "x" * 100), ("strNul", "q\x00\x00"),
                             ("strAstral", "x\U00010400y" * 40), ("strSurr", "\ud800" * 120), ("bytes100", b"x" * 100),
                             ("bytesNul", b"\x00\x01xyz" * 30), ("none", None), ("int", 5), ("float", 1.5),
                             ("bool", True), ("bigint", 2**70), ("strLong", "".join(chr(0x4E00 + i % 50) for i in range(300)))]:
            fn_case(S, "translate", [op], [_Op(0), table], f"{spec}:pool:{label}")
    # C. Arrays of tables: lists, U arrays (operands), object lists of dicts / None / ints / Python bytes / strs, a 2-D
    #    object list (the broadcast grows), mismatched shapes.
    two = plain(np.array(["abc", "de"], dtype="<U3"))
    fn_case(S, "translate", [two], [_Op(0), ["x" * 100, "y" * 120]], "list:str")
    fn_case(S, "translate", [two, plain(np.array(["x" * 100]))], [_Op(0), _Op(1)], "arr:U:one")
    fn_case(S, "translate", [two, plain(np.array(["x" * 100, "y" * 101]))], [_Op(0), _Op(1)], "arr:U:two")
    fn_case(S, "translate", [two, plain(np.array([1, 2, 3]))], [_Op(0), _Op(1)], "arr:int:mismatch")
    fn_case(S, "translate", [plain(np.array(["abc"]))], [_Op(0), [{97: "z"}, None]], "obj:dict,none")
    fn_case(S, "translate", [plain(np.array(["", "b"]))], [_Op(0), [None, {98: "q"}]], "obj:none,dict")
    fn_case(S, "translate", [plain(np.array(["abc"]))], [_Op(0), [5, 2**70]], "obj:int,bigint")
    fn_case(S, "translate", [plain(np.array(["abc"]))], [_Op(0), [b"\x00\x01xyz" * 30, None]], "obj:bytes,none")
    fn_case(S, "translate", [plain(np.array(["abc"]))], [_Op(0), ["xyz\x00" * 30, None]], "obj:str,none")
    fn_case(S, "translate", [plain(np.array(["abc"]))], [_Op(0), [1.5, None]], "obj:float,none")
    fn_case(S, "translate", [plain(np.array(["abc", "bcd"]))], [_Op(0), [[{97: "z"}], [{98: "y", 99: None}]]], "obj:2d")
    fn_case(S, "translate", [plain(np.array(["abc", "bcd", "x"]))], [_Op(0), [{97: "z"}, None]], "obj:mismatch")
    fn_case(S, "translate", [plain(np.array([["abc"], ["cab"]]))], [_Op(0), [{97: "z"}, {99: "q"}, {98: None}]], "obj:bcast")
    # D. The C-order first error: elements raising different errors in several layouts.
    errs = {97: 3.5, 99: -1, 100: 0x110000}
    for layout in ["c1", "c2", "f2", "t3", "r2", "s1", "col"]:
        n, build = LAYOUTS[layout]
        texts = [["bd", "ab", "cc", "xyz"][i % 4] for i in range(n)]
        base = np.array(texts, dtype="<U4") if n else np.zeros(0, dtype="<U4")
        fn_case(S, "translate", [(base, build(base))], [_Op(0), errs], f"errorder:{layout}")
    # E. U units past U+10FFFF: alone (the scalar refuses it), kept (the writer refuses it), mapped / deleted (fine).
    for units, label in [([0x61, 0x110000], "kept"), ([0x110000, 0], "alone"), ([0x110000, 0x61], "first"),
                         ([0x61, 0x110000, 0x62, 0], "mixed"), ([0x61, 0x62, 0x63, 0xFFFFFFF0], "last"), ([0x61, 0x62, 0x63, 0x7FFFFFF0], "last31"),
                         ([0xD800, 0x61], "surrogate")]:
        raw = plain(np.frombuffer(np.array(units, dtype="<u4").tobytes(), dtype="<U2").copy())
        for tname in ["mk", "past", "uni", "empty"]:
            fn_case(S, "translate", [raw], [_Op(0), TR_U_TABLES[tname]], f"U:{label}:{tname}")
        fn_case(S, "translate", [raw], [_Op(0), {0x110000: "x"}], f"U:{label}:mapped")
        fn_case(S, "translate", [raw], [_Op(0), {0x110000: None, 0x7FFFFFF0: "y"}], f"U:{label}:deleted")
    # The translate mutation pass's pins: a non-integral float key (it equals no code point), a str TABLE holding a unit
    # past U+10FFFF (CPython's SystemError when a character indexes it there, nothing otherwise), and a key exactly at the
    # dense table's end read through the eight-unit gather (U+0100 when the call compiles 256 dense entries).
    fn_case(S, "translate", [plain(np.array(["abc", "xbz"]))], [_Op(0), {97.5: "z", 98.0: "y"}], "U:floatkeys")
    fn_case(S, "translate", [plain(np.array(["\x01\x00a"])), _u_units([0x61, 0x110000], 2)], [_Op(0), _Op(1)],
            "U:strtable-past:indexed")
    fn_case(S, "translate", [plain(np.array(["ab"])), _u_units([0x61, 0x110000], 2)], [_Op(0), _Op(1)],
            "U:strtable-past:kept")
    fn_case(S, "translate", [plain(np.array(["Ā" * 8, "Āÿ" * 4]))], [_Op(0), {256: "X", 255: "Y"}], "U:dense-end")
    fn_case(S, "translate", [plain(np.array(["Ā" * 8, "aĀbĀcĀdĀ"]))], [_Op(0), {97: "b"}], "U:dense-end:gather")
    # F. bytes.translate: tables, wrong lengths, np.str_ / number / dict tables, delete sets of every kind.
    sop = plain(np.array(TR_S_POOL, dtype="|S6"))
    for label, table in [("mk", mk), ("rev", rev), ("rev255", rev[:-1] + b"\x01"), ("x255", b"x" * 255),
                         ("x256", b"x" * 256), ("x257", b"x" * 257), ("nulEnd", bytes(range(255)) + b"\x00"),
                         ("none", None), ("str64", "xyzw" * 16), ("str256", "x" * 256), ("int", 5), ("dict", {97: 98}),
                         ("float", 1.5)]:
        fn_case(S, "translate", [sop], [_Op(0), table], f"S:table:{label}")
    for label, delete in [("bytes", b"ab"), ("nul", b"\x00"), ("empty", b""), ("str", "a"), ("strAcc", "\xe9"),
                          ("int", 97), ("bool", True), ("dict", {1: 2}), ("float", 2.5), ("list", [b"a", b"b"]),
                          ("objNone", [None, b"a"]), ("objBool", [True, None]), ("long", b"abc" * 100)]:
        sub = plain(np.array([b"abc", b"cab"], dtype="|S4"))
        fn_case(S, "translate", [sub], [_Op(0), mk, delete], f"S:delete:{label}")
        fn_case(S, "translate", [sub], [_Op(0), None, delete], f"S:delete:{label}:notable")
    fn_case(S, "translate", [plain(np.array([b"abc"]))], [_Op(0), b"x" * 10, {1: 2}], "S:order:table-first")
    fn_case(S, "translate", [plain(np.array([b"abc"]))], [_Op(0), {1: 2}, b"x" * 10], "S:order:dict-table")
    fn_case(S, "translate", [plain(np.zeros(0, dtype="|S3"))], [_Op(0), b"x" * 255], "S:empty:badtable")
    fn_case(S, "translate", [plain(np.array([b""]))], [_Op(0), b"x" * 255], "S:emptytext:badtable")
    fn_case(S, "translate", [plain(np.array([b"abc"]))], [_Op(0), [b"x" * 256, b"y" * 256]], "S:list:tables")
    fn_case(S, "translate", [plain(np.array([b"abc", b"cab"]))], [_Op(0), None, [b"a", b"b", b"c"]], "S:delete:mismatch")
    # G. The width sweep (the U gather route's chunking and the S whole-width loops): random texts per width.
    rng = random.Random(2618)
    letters = "abcdefghijklmnopqrstuvwxyz é中"
    for w in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 100]:
        texts = ["".join(rng.choice(letters) for _ in range(rng.choice([w, rng.randint(0, w)]))) for _ in range(20)]
        uop = plain(np.array(texts, dtype=f"<U{w}"))
        for tname in ["mk", "mkdel", "exp", "uni", "nul", "toNul"]:
            fn_case(S, "translate", [uop], [_Op(0), TR_U_TABLES[tname]], f"<U{w}:sweep:{tname}")
        stexts = [t.encode("latin-1", "replace").replace(b"?", b"q") for t in texts]
        sop2 = plain(np.array(stexts, dtype=f"|S{w}"))
        fn_case(S, "translate", [sop2], [_Op(0), mk], f"|S{w}:sweep:mk")
        fn_case(S, "translate", [sop2], [_Op(0), mk, b"de "], f"|S{w}:sweep:mkdel")
        fn_case(S, "translate", [sop2], [_Op(0), _tr_bytes_map(b"\x00a", b"Zb")], f"|S{w}:sweep:nulmap")
        fn_case(S, "translate", [sop2], [_Op(0), None, b"\x00a"], f"|S{w}:sweep:nuldel")
    # H. deletechars is ignored for U (even one that would not broadcast); Python operands; 0-d, empty, non-text.
    u = plain(np.array(["abc", "cab"]))
    for delete in ["b", b"b", [1, 2, 3], None, {1: 2}]:
        fn_case(S, "translate", [u], [_Op(0), TR_U_TABLES["mk"], delete], f"U:deleteIgnored:{type(delete).__name__}")
    fn_case(S, "translate", [], ["abc", TR_U_TABLES["mk"]], "pystr")
    fn_case(S, "translate", [], [b"abc", mk], "pybytes")
    fn_case(S, "translate", [plain(np.array("abc"))], [_Op(0), {97: "z"}], "0d")
    fn_case(S, "translate", [plain(np.zeros((2, 0), dtype="<U3"))], [_Op(0), {97: "z"}], "empty")
    fn_case(S, "translate", [plain(np.zeros((2, 0), dtype="<U3"))], [_Op(0), None], "empty:none")
    for arr in [np.array([1, 2]), np.array([1.5]), np.array([True]), np.zeros((2, 0), dtype="i4")]:
        fn_case(S, "translate", [plain(arr)], [_Op(0), {97: "z"}], f"num:{arr.dtype}")
    # R. Random tables over random texts, widths and layouts (seeded): dicts of 1-8 entries whose values are characters,
    #    replacements of 0-4 characters, None, code points and (rarely) a value that raises; random bytes.maketrans tables
    #    with random delete sets.
    rng = random.Random(2620)
    for i in range(240):
        unicode = i % 3 != 2
        w = rng.choice([1, 2, 3, 5, 8, 9, 16, 17, 33])
        layout = rng.choice(list(LAYOUTS))
        if unicode:
            pool = [_tr_random_text(rng, w, True) for _ in range(8)]
            spec = f"<U{w}"
            op = fn_operand(spec, layout, seed=0, pool=pool)
            fn_case(S, "translate", [op], [_Op(0), _tr_random_table(rng)], f"rand{i}:{spec}:{layout}")
        else:
            pool = [_tr_random_text(rng, w, False).encode("latin-1") for _ in range(8)]
            spec = f"|S{w}"
            op = fn_operand(spec, layout, seed=0, pool=pool)
            frm = bytes(rng.sample(range(256), rng.randint(0, 12)))
            to = bytes(rng.randrange(256) for _ in frm)
            delete = bytes(rng.sample(range(256), rng.randint(0, 4))) if rng.random() < 0.6 else None
            fn_case(S, "translate", [op], [_Op(0), bytes.maketrans(frm, to), delete], f"rand{i}:{spec}:{layout}")


# The characters random translate texts and tables draw from: ASCII letters and space, NUL, Latin-1 and BMP characters
# (two- and three-byte UTF-8), an astral one.
_TR_CHARS_U = "abcdefxyz ABC\x00éßñ中文\U00010400"
_TR_CHARS_S = "abcdefxyz ABC\x00\xe9\xff"


def _tr_random_text(rng, width, unicode):
    """A random translate text of up to ``width`` characters (often exactly ``width``).

    :param rng: the random.Random.
    :param width: the most characters.
    :param unicode: whether non-Latin-1 characters may appear.
    :returns: the text.
    """
    chars = _TR_CHARS_U if unicode else _TR_CHARS_S
    n = rng.choice([width, width, rng.randint(0, width)])
    return "".join(rng.choice(chars) for _ in range(n))


def _tr_random_table(rng):
    """A random str.translate dict: 1-8 keys drawn from the text characters, each value a character, a replacement of 0-4
    characters, None, a code point or True, and rarely (4 %) a value that raises (a float, a negative int).

    :param rng: the random.Random.
    :returns: the dict.
    """
    table = {}
    for _ in range(rng.randint(1, 8)):
        key = ord(rng.choice(_TR_CHARS_U))
        r = rng.random()
        if r < 0.3:
            table[key] = rng.choice(_TR_CHARS_U)
        elif r < 0.55:
            table[key] = "".join(rng.choice(_TR_CHARS_U) for _ in range(rng.randint(0, 4)))
        elif r < 0.7:
            table[key] = None
        elif r < 0.9:
            table[key] = ord(rng.choice(_TR_CHARS_U))
        elif r < 0.96:
            table[key] = True
        else:
            table[key] = rng.choice([3.5, -1])
    return table


def gen_ttrans():
    """Section ttrans: numpy.strings.translate over NEP 55 StringDType, strict — every table kind x every NA kind (a str
    method refuses an NA object that is not a str; a string NA is translated as its text, and any result equal to it is
    stored as the NA) x the layout kinds over short / long / ASCII / non-ASCII texts; deletechars (str.translate takes one
    argument), surrogate results (the setitem failure _vec_string reports as a TypeError), the C-order first error,
    StringDType arrays of tables, and a length sweep over the byte routes and the UTF-8 route."""
    import random
    S = "ttrans"
    tpool = ["", "a", "abc", "cab", "abcdef", "hello world", "a\x00b\x00", "\x00", "é中文", "aé中b", "x" * 15, "x" * 16,
             "abc" * 10, "é" * 20, "\U00010400\U00010401c", "a\U0001F600b", "Q", "ABC", "def def def"]
    for spec in T_COMPUTE_SPECS:
        lab0 = _t_label(spec)
        for layout in T_COMPUTE_LAYOUTS:
            op = t_operand(spec, layout, seed=1, pool=tpool, na_every=5)
            for tname in ["mk", "mkdel", "exp", "uni"]:
                fn_case(S, "translate", [op], [_Op(0), TR_U_TABLES[tname]], f"{lab0}:{layout}:{tname}")
        for na_every in (0, 3):
            base = np.array(t_values(spec, len(tpool), 0, tpool, na_every), dtype=spec.build())
            for tname, table in TR_U_TABLES.items():
                fn_case(S, "translate", [(base, base)], [_Op(0), table], f"{lab0}:pool:na{na_every}:{tname}")
            for label, table in [("str", "xyz"), ("str100", "x" * 100), ("bytes", b"x" * 100), ("none", None),
                                 ("int", 5), ("float", 1.5), ("bigint", 2**70)]:
                fn_case(S, "translate", [(base, base)], [_Op(0), table], f"{lab0}:pool:na{na_every}:{label}")
    # deletechars: every element raises (the descriptor's check first for an NA that is not a str); none raise when the
    # broadcast is empty; None is not passed.
    for spec in [TD, TDN, TDNAN, TDS]:
        lab0 = _t_label(spec)
        variants = [["abc", "x"]]
        if spec.na is not _ABSENT:
            variants += [[spec.na, "abc"], ["abc", spec.na]]
        for k, values in enumerate(variants):
            base = np.array(values, dtype=spec.build())
            fn_case(S, "translate", [(base, base)], [_Op(0), {97: "z"}, "b"], f"{lab0}:deletechars{k}")
            fn_case(S, "translate", [(base, base)], [_Op(0), {97: "z"}, None], f"{lab0}:deletechars{k}:none")
        empty = np.array([], dtype=spec.build())
        fn_case(S, "translate", [(empty, empty)], [_Op(0), {97: "z"}, "b"], f"{lab0}:deletechars:empty")
    # NA first / later with tables that raise or not (the C-order first error).
    for spec in [TDN, TDNAN]:
        lab0 = _t_label(spec)
        for values in (["abc", spec.na], [spec.na, "abc"], ["", spec.na]):
            base = np.array(values, dtype=spec.build())
            for label, table in [("ok", {97: "XYZ"}), ("none", None), ("err", {97: 3.5})]:
                fn_case(S, "translate", [(base, base)], [_Op(0), table], f"{lab0}:naorder:{label}")
    # Sentinels equal to some results: setitem stores a result equal to the NA string as the NA.
    for sentinel, values, table in [("Q", ["abc", "Q", "x"], {97: "Q", 98: None, 99: None}),
                                    ("", ["abc", "", "z"], {97: None, 98: None, 99: None}),
                                    ("xyz", ["abc", "xyz", "aBc"], {97: "x", 98: "y", 99: "z"})]:
        spec = TSpec(sentinel)
        base = np.array(values, dtype=spec.build())
        fn_case(S, "translate", [(base, base)], [_Op(0), table], f"TS[{sentinel}]:eq")
    # Surrogate results (raised at setitem, after the element's own translate errors), and their order.
    base = np.array(["abc", "b"], dtype=TD.build())
    fn_case(S, "translate", [(base, base)], [_Op(0), {97: 0xD800}], "surr:int")
    fn_case(S, "translate", [(base, base)], [_Op(0), {97: "\ud800"}], "surr:str")
    fn_case(S, "translate", [(base, base)], [_Op(0), {97: 0xD800, 98: 3.5}], "surr:then-err")
    base2 = np.array(["xyz", "abc"], dtype=TD.build())
    fn_case(S, "translate", [(base2, base2)], [_Op(0), {97: 0xD800, 98: 3.5}], "surr:err-first")
    fn_case(S, "translate", [(base, base)], [_Op(0), "x\ud800yz" * 30], "surr:strtable")
    # StringDType tables (Python strs: no buffer, so a bytes.translate refuses them; str.translate indexes them).
    tab = np.array(["xy" * 60, "z" * 120], dtype=TD.build())
    two = np.array(["abc", "de"], dtype=TD.build())
    fn_case(S, "translate", [(two, two), (tab, tab)], [_Op(0), _Op(1)], "Ttable")
    fn_case(S, "translate", [plain(np.array([b"abc", b"cab"])), (tab, tab)], [_Op(0), None, _Op(1)], "S:Tdelete")
    # The length sweep: random texts of every length class over each route (one-to-one / compaction byte maps, ASCII
    # replacements, the UTF-8 route).
    rng = random.Random(2619)
    for spec in [TD, TDS]:
        lab0 = _t_label(spec)
        for w in [0, 1, 3, 7, 8, 14, 15, 16, 17, 31, 32, 33, 64, 100]:
            for unicode in (False, True):
                letters = "abcdefghij é中文ßñ\U00010400" if unicode else "abcdefghij xyz"
                texts = ["".join(rng.choice(letters) for _ in range(rng.choice([w, rng.randint(0, w)]))) for _ in range(20)]
                base = np.array(texts, dtype=spec.build())
                for tname in ["mk", "mkdel", "exp", "uni", "astral", "nul"]:
                    fn_case(S, "translate", [(base, base)], [_Op(0), TR_U_TABLES[tname]],
                            f"{lab0}:len{w}:{'uni' if unicode else 'ascii'}:{tname}")
    # Random tables over random texts, NA kinds and layouts (seeded).
    rng = random.Random(2621)
    specs = [TD, TDS, TDN, TDNAN, TSpec("ab")]
    for i in range(200):
        spec = rng.choice(specs)
        w = rng.choice([0, 3, 8, 15, 16, 24, 40])
        layout = rng.choice(T_COMPUTE_LAYOUTS)
        pool = [_tr_random_text(rng, w, True) for _ in range(8)]
        op = t_operand(spec, layout, seed=0, pool=pool, na_every=rng.choice([0, 0, 4]))
        fn_case(S, "translate", [op], [_Op(0), _tr_random_table(rng)], f"rand{i}:{_t_label(spec)}:{layout}")


def _u_units(units, width):
    """A U array holding raw UCS-4 units (rows of ``width`` units), as an operand: units past U+10FFFF and lone surrogates
    included, which no Python str literal builds.

    :param units: the units, row after row.
    :param width: the element width in units.
    :returns: the (base, view) pair.
    """
    return plain(np.frombuffer(np.array(units, dtype="<u4").tobytes(), dtype=f"<U{width}").copy())


# numpy.strings.mod's format pools. MOD_NUM_FORMATS succeed for every Python int / float value (so a layout case over them
# never ends at an element error): decimal, str / repr, fixed and general floats, flags, a literal-only format (a NumPy
# scalar is a mapping, so no "not all arguments converted"), %% and a length modifier.
MOD_NUM_FORMATS = ["%d", "%s", "%r", "%5.1f|", "v=%g", "%e", "%+d", "%-6s|", "%%%d", "abc", "", "%ld", "%.3s", "[%10.2E]"]
# The value dtypes the per-dtype matrix crosses with every conversion (NumPy's scalar of each element meets %).
MOD_VALUE_ARRAYS = [
    ("i1", [-128, 0, 5, 127]), ("u1", [0, 65, 255]), ("i2", [-32768, 7, 32767]), ("u2", [0, 66, 65535]),
    ("i4", [-(2**31), 9, 2**31 - 1]), ("u4", [0, 67, 2**32 - 1]), ("i8", [-(2**63), -1, 0, 68, 2**63 - 1]),
    ("u8", [0, 69, 2**63, 2**64 - 1]), ("f2", [0.1, -0.0, 65504.0, 6e-08, float("inf"), float("nan")]),
    ("f4", [0.1, 1e16, 1e-05, 123456789.0, -2.5, float("-inf")]),
    ("f8", [0.1, -0.0, 1e16, 1e-05, 0.30000000000000004, 123456789012345680.0, 5e-324, float("nan")]),
    ("c16", [1 + 2j, -0.0 + 1e300j, complex(float("nan"), float("inf")), 3 + 0j]), ("?", [True, False]),
    ("<U4", ["", "a", "é中", "\U00010400x"]), ("|S3", [b"", b"a", b"\xff\x00x"]),
]
MOD_CONVERSIONS = ["%s", "%r", "%a", "%d", "%i", "%x", "%X", "%o", "%c", "%e", "%f", "%g", "%G", "%5s|", "%-8.3s|", "%u",
                   "%b", "%#x"]


def _mod_flag_formats(convs, flags=("", "-", "+", " ", "0", "#", "-0", "+0", "- ", "#0", "-#", "+#"),
                      widths=("", "1", "7"), precs=("", ".0", ".3")):
    """Every flag x width x precision combination of the given conversion characters, as one format list.

    :param convs: the conversion characters.
    :param flags: the flag strings.
    :param widths: the width strings.
    :param precs: the precision strings.
    :returns: the formats.
    """
    return [f"%{fl}{w}{p}{c}|" for c in convs for fl in flags for w in widths for p in precs]


def gen_strmod():
    """Section strmod: numpy.strings.mod over S / U formats, strict — every layout kind of both classes over formats that
    succeed for numbers; every NumPy value dtype x every conversion (NumPy's scalar meets %: np.float64 repr, %d truncating,
    %x refusing floats and np.bool, %s over S writing a scalar's raw buffer); the flag / width / precision matrix of the
    integer, float and text conversions; mappings (dicts with str / bytes keys and values of every kind, NumPy scalars'
    IndexError, the format-requires-a-mapping and not-all-arguments rules per value kind); CPython's parse errors in their
    order; units past U+10FFFF; result widths (trailing NULs, empty, 0-d, broadcast); long results past the 64-element sample
    NumSharp lays its result rows out from (section L); Python operands; random formats."""
    import random
    S = "strmod"
    # A. Every layout kind x both classes x number values (Python int, float, a broadcast int array).
    for spec in ["<U8", "<U16", "|S8", "|S16"]:
        t = _fn_t(spec)
        pool = [t(f) for f in MOD_NUM_FORMATS]
        for layout in LAYOUTS:
            op = fn_operand(spec, layout, seed=1, pool=pool)
            lab = f"{spec}:{layout}"
            fn_case(S, "mod", [op], [_Op(0), 42], f"{lab}:int")
            fn_case(S, "mod", [op], [_Op(0), -3.75], f"{lab}:float")
            fn_case(S, "mod", [op, plain(np.array([7, -8, 900], dtype="i8"))], [_Op(0), _Op(1)], f"{lab}:arr")
    # B. Every value dtype x every conversion, both classes (one format per case: its first failing element is the case's
    #    error, so every conversion's own answer is recorded).
    for cls in ["U", "S"]:
        for fmt in MOD_CONVERSIONS:
            f = fmt.encode("latin-1") if cls == "S" else fmt
            fop = plain(np.array([f]))
            for dt, vals in MOD_VALUE_ARRAYS:
                vop = plain(np.array(vals, dtype=dt))
                fn_case(S, "mod", [fop, vop], [_Op(0), _Op(1)], f"{cls}:{fmt}:{dt}")
            tvals = np.array(["", "ab", "é", "a\x00"], dtype=TD.build())
            fn_case(S, "mod", [fop, (tvals, tvals)], [_Op(0), _Op(1)], f"{cls}:{fmt}:T")
            for label, value in [("pyint", 65), ("pybig", 2**70), ("pyneg", -(2**63) - 1), ("pyfloat", 2.5),
                                 ("pybool", True), ("pystr", "z"), ("pybytes", b"q"), ("none", None)]:
                fn_case(S, "mod", [fop], [_Op(0), value], f"{cls}:{fmt}:{label}")
    # C. The flag / width / precision matrix: one case per (class, value) over every combination's format.
    int_formats = _mod_flag_formats("dixXoeEfFgGsra")
    float_formats = _mod_flag_formats("deEfFgGsra")
    nonfinite_formats = _mod_flag_formats("eEfFgGsr")
    text_formats = _mod_flag_formats("sra")
    for cls in ["U", "S"]:
        t = (lambda s: s.encode("latin-1")) if cls == "S" else (lambda s: s)
        for value in [0, 42, -42, 2**40, -(2**63), 2**64 - 1]:
            fn_case(S, "mod", [plain(np.array([t(f) for f in int_formats]))], [_Op(0), value], f"{cls}:flags:int{value}")
        for value in [0.0, -0.0, 3.14159, -1234.5678, 1e20, 1e-07, 0.5, 2.5]:
            fn_case(S, "mod", [plain(np.array([t(f) for f in float_formats]))], [_Op(0), value], f"{cls}:flags:float{value}")
        for value in [float("inf"), float("-inf"), float("nan")]:
            fn_case(S, "mod", [plain(np.array([t(f) for f in nonfinite_formats]))], [_Op(0), value],
                    f"{cls}:flags:{value}")
        for value in ["", "ab", "é\U00010400z"] if cls == "U" else [b"", b"ab", b"\xe9z"]:
            fn_case(S, "mod", [plain(np.array([t(f) for f in text_formats]))], [_Op(0), value],
                    f"{cls}:flags:text{len(value)}")
        cfmts = [t(f) for f in _mod_flag_formats("c", precs=("", ".2"))]
        fn_case(S, "mod", [plain(np.array(cfmts))], [_Op(0), 65], f"{cls}:flags:c")
    # D. Mappings: dicts (str keys for U, bytes keys for S; values of every kind), NumPy scalars, bytes / str / None.
    udict = {"a": 1, "b": "xyz", "c": None, "d": 2.5, "e": b"q", "f": [1, 2], "g": {"h": 1}, "a(b)": 7, "": 0,
             "big": 2**70, "t": True, "u": "é\U00010400"}
    for fmt in ["%(a)s", "%(a)d|%(b)s|%(c)r", "%(d).1f", "%(e)s", "%(f)s", "%(g)r", "%(a(b))s", "%()d", "%(zz)s",
                "%(a)s %s", "%s %(a)s", "%(a", "%(a)", "%(a)*d", "%(big)x", "%(t)c", "%(u)s|%(u)r|%(u)a", "%(b)c",
                "%(a)-5d|", "abc", "%s", "%(b)5.2s|"]:
        fn_case(S, "mod", [plain(np.array([fmt]))], [_Op(0), udict], f"U:dict:{fmt}")
        fn_case(S, "mod", [plain(np.array([fmt.encode("utf-8")]))], [_Op(0), udict], f"S:strdict:{fmt}")
    bdict = {b"a": 1, b"b": b"xyz", b"c": None, b"d": 2.5, b"s": "str", b"": 0}
    for fmt in [b"%(a)d", b"%(b)s", b"%(b)r", b"%(c)r", b"%(d)f", b"%(s)s", b"%(s)r", b"%()d", b"%(zz)s", b"abc", b"%s",
                b"%(a)c", b"%(b)b"]:
        fn_case(S, "mod", [plain(np.array([fmt]))], [_Op(0), bdict], f"S:dict:{fmt!r}")
    for cls in ["U", "S"]:
        t = (lambda s: s.encode("latin-1")) if cls == "S" else (lambda s: s)
        for label, value in [("int", 5), ("float", 1.5), ("bool", True), ("u64", np.array([5], dtype="u8")),
                             ("str", "x"), ("bytes", b"x"), ("none", None), ("big", 2**70), ("objdict", [{"a": 1}, None]),
                             ("f2", np.array([1.5], dtype="f2")), ("c16", np.array([1j]))]:
            vargs = [_Op(0), _Op(1)] if isinstance(value, np.ndarray) else [_Op(0), value]
            ops = [plain(np.array([t("abc"), t("%(a)s"), t("x%sy")]))]
            if isinstance(value, np.ndarray):
                ops.append(plain(value))
            for k, f in enumerate([t("abc"), t("%(a)s"), t("%(a")]):
                fops = [plain(np.array([f]))] + ops[1:]
                fn_case(S, "mod", fops, vargs, f"{cls}:mapping:{label}:{k}")
    # E. CPython's parse errors and their order (the argument taken before the conversion character is checked).
    for fmt in ["%y", "%", "%(a", "%5", "%.", "%l", "%-", "%ld|%hd|%Ld", "%lld", "%9999999999999999999d",
                "%.9999999999d", "%.2147483647d", "%s %s", "%s %y", "%y %s", "%s %s %y", "%*d", "%.*f", "%*", "%%|%%",
                "%s %%", "%\x01", "%\x7f", "%(a)", "% 5d|% -5d|", "%5%", "%-%", "%0s", "%#s", "%#c"]:
        for cls in ["U", "S"]:
            f = fmt.encode("latin-1") if cls == "S" else fmt
            fn_case(S, "mod", [plain(np.array([f]))], [_Op(0), 5], f"{cls}:parse:{fmt!r}")
            fn_case(S, "mod", [plain(np.array([f]))], [_Op(0), "x"], f"{cls}:parse:{fmt!r}:str")
    for fmt in [b"%\xff", b"%\x80", b"ab%\x00x", b"\xe9\x80%s", b"%b", b"%5b|", b"%.1b|", b"%a"]:
        fn_case(S, "mod", [plain(np.array([fmt]))], [_Op(0), b"xy"], f"S:bytesparse:{fmt!r}")
        fn_case(S, "mod", [plain(np.array([fmt]))], [_Op(0), 5], f"S:bytesparse:{fmt!r}:int")
    for fmt in ["%Ā", "%\U00010400", "a\U00010400%y"]:
        fn_case(S, "mod", [plain(np.array([fmt]))], [_Op(0), 5], f"U:parse:astral:{ascii(fmt)}")
    # The C-order first error: formats raising different errors in several layouts.
    for layout in ["c1", "c2", "f2", "t3", "r2", "s1", "col"]:
        n, build = LAYOUTS[layout]
        fmts = [["%d", "%y", "%x", "%s %s"][i % 4] for i in range(n)]
        base = np.array(fmts, dtype="<U5") if n else np.zeros(0, dtype="<U5")
        fn_case(S, "mod", [(base, build(base))], [_Op(0), "x"], f"errorder:{layout}")
        fn_case(S, "mod", [(base, build(base))], [_Op(0), 2.5], f"errorder:{layout}:float")
    # F. U units past U+10FFFF and lone surrogates in formats and values.
    fn_case(S, "mod", [_u_units([0x25, 0x73, 0x110000, 0], 2)], [_Op(0), 5], "U:past:fmt-alone")
    fn_case(S, "mod", [_u_units([0x25, 0x73, 0x61, 0x110000], 2)], [_Op(0), 5], "U:past:fmt-kept")
    fn_case(S, "mod", [_u_units([0x25, 0x110000, 0x25, 0x64], 2)], [_Op(0), 5], "U:past:conversion")
    fn_case(S, "mod", [plain(np.array(["%s", "%r", "%a", "%5s|"])), _u_units([0x61, 0x110000], 2)], [_Op(0), _Op(1)],
            "U:past:value-kept")
    fn_case(S, "mod", [plain(np.array(["%s"])), _u_units([0x110000, 0, 0x61, 0], 2)], [_Op(0), _Op(1)], "U:past:value-alone")
    fn_case(S, "mod", [_u_units([0x110000, 0], 2), _u_units([0x110000, 0], 2)], [_Op(0), _Op(1)], "U:past:both")
    fn_case(S, "mod", [plain(np.array(["%s", "%r", "%c"])), _u_units([0xD800, 0x61, 0xDC00, 0], 2)], [_Op(0), _Op(1)],
            "U:surrogate:value")
    fn_case(S, "mod", [_u_units([0x25, 0x73, 0xD800, 0], 2)], [_Op(0), 5], "U:surrogate:fmt")
    fn_case(S, "mod", [plain(np.array([b"%s", b"%r"])), _u_units([0x61, 0x110000], 2)], [_Op(0), _Op(1)], "S:past:value")
    # G. bytes formatting specifics: %c of bytes / ints, buffers of every scalar, U / StringDType values, float errors.
    for label, value in [("u1", np.array([0, 65, 255], dtype="u1")), ("i8", np.array([65, 256, -1])),
                         ("S", np.array([b"x", b"", b"xy"])), ("bool", np.array([True])), ("f8", np.array([1.5])),
                         ("U", np.array(["x"]))]:
        fn_case(S, "mod", [plain(np.array([b"%c"])), plain(value)], [_Op(0), _Op(1)], f"S:c:{label}")
    for label, value in [("big", 2**2000), ("str", "x"), ("none", None), ("bytes", b"x"), ("bool", True), ("int", 7)]:
        for f in [b"%f", b"%e", b"%g", b"%d", b"%x"]:
            fn_case(S, "mod", [plain(np.array([f]))], [_Op(0), value], f"S:numconv:{f!r}:{label}")
        fn_case(S, "mod", [plain(np.array(["%f"]))], [_Op(0), value], f"U:numconv:%f:{label}")
    tvals = np.array(["x", "ab"], dtype=TD.build())
    tnone = np.array(["x", None], dtype=TDN.build())
    for f in [b"%s", b"%r", b"%a", b"%b"]:
        fn_case(S, "mod", [plain(np.array([f])), (tvals, tvals)], [_Op(0), _Op(1)], f"S:T:{f!r}")
        fn_case(S, "mod", [plain(np.array([f])), (tnone, tnone)], [_Op(0), _Op(1)], f"S:TN:{f!r}")
    # H. Result widths: trailing NULs (counted by np.asarray, dropped by the element), empty, 0-d, broadcasting.
    tnul = np.array(["a\x00", "a\x00b\x00\x00", "\x00"], dtype=TD.build())
    for f in ["%s", "%s|", "[%r]"]:
        fn_case(S, "mod", [plain(np.array([f])), (tnul, tnul)], [_Op(0), _Op(1)], f"U:width:nul:{f}")
    for dt in ["i8", "i2", "u1", "f4", "c16", "?"]:
        fn_case(S, "mod", [plain(np.array([b"%s"])), plain(np.array([5], dtype=dt))], [_Op(0), _Op(1)], f"S:width:buf:{dt}")
    fn_case(S, "mod", [plain(np.array(["%s"])), plain(np.array(["\x00"]))], [_Op(0), _Op(1)], "U:width:empty-result")
    fn_case(S, "mod", [plain(np.array(["ab%s"])), plain(np.array(["", "xyz"]))], [_Op(0), _Op(1)], "U:width:grow")
    fn_case(S, "mod", [plain(np.zeros(0, dtype="<U3"))], [_Op(0), 5], "U:empty")
    fn_case(S, "mod", [plain(np.zeros((2, 0), dtype="|S3"))], [_Op(0), 5], "S:empty2d")
    fn_case(S, "mod", [plain(np.array(["%s"])), plain(np.zeros(0, dtype="i8"))], [_Op(0), _Op(1)], "U:empty-values")
    fn_case(S, "mod", [plain(np.array(["%s"])), plain(np.zeros((2, 0)))], [_Op(0), _Op(1)], "U:empty-values2d")
    fn_case(S, "mod", [plain(np.array("%d"))], [_Op(0), 5], "U:0d")
    fn_case(S, "mod", [plain(np.array(["%d", "%s"])), plain(np.array([[1], [22], [333]]))], [_Op(0), _Op(1)], "U:bcast2d")
    fn_case(S, "mod", [plain(np.array([[b"%d"], [b"%s"]]).T), plain(np.array([1, 22]))], [_Op(0), _Op(1)], "S:bcastT")
    fn_case(S, "mod", [plain(np.array(["%s", "%s"]))], [_Op(0), [1, 2, 3]], "U:mismatch")
    fn_case(S, "mod", [plain(np.array(["%s"]))], [_Op(0), [[1, 2], [3]]], "U:ragged")
    fn_case(S, "mod", [plain(np.array(["%s"]))], [_Op(0), [1, None]], "U:objlist")
    fn_case(S, "mod", [plain(np.array(["%r"]))], [_Op(0), [1, 2**63]], "U:list:i8u8")
    fn_case(S, "mod", [plain(np.array(["%r"]))], [_Op(0), [2**63, 2**64 - 1]], "U:list:u8")
    fn_case(S, "mod", [plain(np.array(["%r"]))], [_Op(0), [True, 2]], "U:list:boolint")
    fn_case(S, "mod", [plain(np.array(["%r"]))], [_Op(0), [1, "a"]], "U:list:intstr")
    fn_case(S, "mod", [plain(np.array(["%r"]))], [_Op(0), [b"a", "b"]], "U:list:bytesstr")
    fn_case(S, "mod", [plain(np.array(["%r"]))], [_Op(0), [b"a", b"bc"]], "U:list:bytes")
    fn_case(S, "mod", [plain(np.array(["%s"]))], [_Op(0), (1, 2)], "U:tuple")
    fn_case(S, "mod", [plain(np.array(["%s", "%s"]))], [_Op(0), [(1, 2), (3, 4)]], "U:tuples2d")
    fn_case(S, "mod", [plain(np.array(["%s"]))], [_Op(0), [(1, 2), None]], "U:tuple-none-ragged")
    # I. Python operands, non-text operands.
    fn_case(S, "mod", [], ["%d", 5], "pystr")
    fn_case(S, "mod", [], [b"%d|%s", 5], "pybytes")
    for arr in [np.array([1, 2]), np.array([1.5]), np.array([True]), np.zeros((2, 0), dtype="i4")]:
        fn_case(S, "mod", [plain(arr)], [_Op(0), 5], f"num:{arr.dtype}")
    fn_case(S, "mod", [plain(np.array([1]))], [_Op(0), [[1], [2, 3]]], "num:ragged-values")
    # L. Long results, past the 64-element sample NumSharp lays its result rows out from: widths that keep growing after
    #    the sample (the rows re-laid out, then narrowed to NumPy's width), every value class through one format (the
    #    chunk path), a format that changes per element (the element path), a 2-D broadcast, every float width, uint64
    #    past int64, and an element error late in the walk.
    lrng = random.Random(2624)
    grow = np.array([1] * 64 + [(-1) ** k * 10 ** min(18, k // 4) for k in range(66)], dtype="i8")
    floats = np.array([lrng.uniform(-1, 1) * 10.0 ** lrng.randint(-8, 15) for _ in range(130)])
    texts = np.array(["".join(lrng.choice("abcé中") for _ in range(min(12, k // 10))) for k in range(130)])
    for cls in ["U", "S"]:
        t = (lambda s: s.encode("latin-1")) if cls == "S" else (lambda s: s)
        for f in ["%d", "<%d>", "%-12d|", "%x", "%r"]:
            fn_case(S, "mod", [plain(np.array([t(f)])), plain(grow)], [_Op(0), _Op(1)], f"long:{cls}:int:{f}")
        for f in ["%.2f", "%.3e", "%g", "%10.4f", "%r"] + (["%s"] if cls == "U" else []):
            fn_case(S, "mod", [plain(np.array([t(f)])), plain(floats)], [_Op(0), _Op(1)], f"long:{cls}:float:{f}")
        late = floats.copy()
        late[100] = np.nan
        fn_case(S, "mod", [plain(np.array([t("%d")])), plain(late)], [_Op(0), _Op(1)], f"long:{cls}:late-nan")
    for f in ["%s", "[%-8s]", "%r", "%.3s"]:
        fn_case(S, "mod", [plain(np.array([f])), plain(texts)], [_Op(0), _Op(1)], f"long:U:text:{f}")
    stexts = np.array([x.encode("utf-8") for x in texts])
    for f in [b"%s", b"[%-8s]", b"%.3b"]:
        fn_case(S, "mod", [plain(np.array([f])), plain(stexts)], [_Op(0), _Op(1)], f"long:S:text:{f!r}")
    cyc = ["%d", "%5d", "%-6d|", "%+d", "%x"]
    fn_case(S, "mod", [plain(np.array([cyc[k % 5] for k in range(130)])), plain(grow)], [_Op(0), _Op(1)],
            "long:U:perelement")
    fn_case(S, "mod", [plain(np.array(["%d", "<%5x>"])), plain(grow.reshape(65, 2))], [_Op(0), _Op(1)], "long:U:rows2d")
    for dt in ["f2", "f4"]:
        vals = np.array([lrng.uniform(-1, 1) * 10.0 ** lrng.randint(-4, 4) for _ in range(130)], dtype=dt)
        for f in ["%s", "%r", "%.3f", "%e"]:
            fn_case(S, "mod", [plain(np.array([f])), plain(vals)], [_Op(0), _Op(1)], f"long:U:{dt}:{f}")
    big = np.array([2**64 - 1 - k * 123456789012345 for k in range(130)], dtype="u8")
    for f in ["%d", "%x", "%e", "%.3f"]:
        fn_case(S, "mod", [plain(np.array([f])), plain(big)], [_Op(0), _Op(1)], f"long:U:u8:{f}")
    # R. Random formats over random values and layouts (seeded): 1-3 specs of random flags / width / precision /
    #    conversion between literal runs, against ints, floats or strs (an element error ends a case: still a case).
    rng = random.Random(2622)
    lits = ["", "a", "x=", " ", "é", "%%", "|", "ab cd"]
    for i in range(240):
        cls = rng.choice(["U", "U", "S"])
        kind = rng.choice(["int", "float", "str"])
        convs = {"int": "dixXoeEfgGsrac", "float": "eEfFgGsrd", "str": "sra"}[kind]
        n = rng.choice([4, 6, 12])
        fmts = []
        for _ in range(n):
            f = rng.choice(lits)
            if rng.random() < 0.9:
                f += "%" + "".join(rng.sample("-+ 0#", rng.randint(0, 2))) + rng.choice(["", "", "3", "12"]) + \
                     rng.choice(["", "", ".0", ".2", ".7"]) + rng.choice(convs) + rng.choice(lits)
            fmts.append(f)
        if cls == "S":
            fmts = [f.encode("utf-8") for f in fmts]
        if kind == "int":
            if rng.random() < 0.5:
                # One Python int (a code point, so %c succeeds too) broadcast over every format.
                vals = rng.randint(0, 0x10FFF)
            else:
                dt = rng.choice(["i8", "i4", "u2"])
                lo = 0 if dt == "u2" else -(10**4)
                vals = np.array([rng.randint(lo, 60000) for _ in range(n)], dtype=dt)
        elif kind == "float":
            vals = np.array([rng.uniform(-1e4, 1e4) for _ in range(n)], dtype=rng.choice(["f8", "f4", "f2"]))
        else:
            vals = np.array(["".join(rng.choice("abcé中\U00010400") for _ in range(rng.randint(0, 5))) for _ in range(n)])
        fop = plain(np.array(fmts))
        if isinstance(vals, np.ndarray):
            fn_case(S, "mod", [fop, plain(vals)], [_Op(0), _Op(1)], f"rand{i}:{cls}:{kind}")
        else:
            fn_case(S, "mod", [fop], [_Op(0), vals], f"rand{i}:{cls}:{kind}:scalar")


def gen_tmod():
    """Section tmod: numpy.strings.mod over NEP 55 StringDType formats, strict — the layout kinds x every NA kind (str.__mod__
    refuses an NA object that is not a str; a string NA is formatted as its text) with number values; StringDType values
    and their NA objects into U and StringDType formats; the result's fresh default descriptor (the NA dropped), its U
    round trip (trailing NULs dropped, a surrogate or a unit past U+10FFFF refused after every element was formatted, in
    C order behind any formatting error); empty operands keeping their descriptor; a length sweep; long results (130
    elements, packed cell by cell, a surrogate deferred behind them); random formats."""
    import random
    S = "tmod"
    tpool = ["%d", "%s", "%r", "%5.1f|", "v=%g", "%e", "%+d", "%-6s|", "%%%d", "abc", "", "%.3s", "é%s中", "x" * 20 + "%d",
             "%s\U0001F600", "Q", "NA"]
    for spec in T_COMPUTE_SPECS:
        lab0 = _t_label(spec)
        for layout in T_COMPUTE_LAYOUTS:
            for na_every in (0, 5):
                op = t_operand(spec, layout, seed=1, pool=tpool, na_every=na_every)
                fn_case(S, "mod", [op], [_Op(0), 42], f"{lab0}:{layout}:na{na_every}:int")
                fn_case(S, "mod", [op], [_Op(0), -3.75], f"{lab0}:{layout}:na{na_every}:float")
    # StringDType values (str elements and NA objects) into U, S and StringDType formats.
    for vspec in [TD, TDN, TDNAN, TDS]:
        vals = np.array(t_values(vspec, 6, 0, ["", "ab", "é中", "a\x00", "x" * 20, "\U0001F600"], 3), dtype=vspec.build())
        for fspec in [TD, TDN]:
            for f in ["%s", "%r", "%a", "%5s|", "%d", "%c", "%(a)s", "abc"]:
                fbase = np.array([f], dtype=fspec.build())
                fn_case(S, "mod", [(fbase, fbase), (vals, vals)], [_Op(0), _Op(1)], f"{_t_label(fspec)}<-{_t_label(vspec)}:{f}")
        for f in ["%s", "%r", "%d"]:
            fn_case(S, "mod", [plain(np.array([f])), (vals, vals)], [_Op(0), _Op(1)], f"U<-{_t_label(vspec)}:{f}")
    # NA formats: a non-str NA first / later; a string NA formatted as its text; values that raise or not.
    for spec in [TDN, TDNAN]:
        for values in (["%d", spec.na], [spec.na, "%d"], ["", spec.na]):
            base = np.array(values, dtype=spec.build())
            for label, value in [("ok", 5), ("err", "x")]:
                fn_case(S, "mod", [(base, base)], [_Op(0), value], f"{_t_label(spec)}:naorder:{label}")
    for sentinel in ["Q", "", "%d"]:
        spec = TSpec(sentinel)
        base = np.array(["%d", sentinel, "x"], dtype=spec.build())
        fn_case(S, "mod", [(base, base)], [_Op(0), 5], f"TS[{sentinel}]:sentinel")
        empty = np.array([], dtype=spec.build())
        fn_case(S, "mod", [(empty, empty)], [_Op(0), 5], f"TS[{sentinel}]:empty")
    # The U round trip: surrogates and units past U+10FFFF refused after formatting, behind a formatting error.
    base = np.array(["%c", "x"], dtype=TD.build())
    fn_case(S, "mod", [(base, base)], [_Op(0), 0xD800], "surr:c")
    fn_case(S, "mod", [(base, base), plain(np.array([0xD800, 5]))], [_Op(0), _Op(1)], "surr:c-arr")
    fmt2 = np.array(["%s"], dtype=TD.build())
    fn_case(S, "mod", [(fmt2, fmt2), _u_units([0xD800, 0x61], 2)], [_Op(0), _Op(1)], "surr:value")
    fn_case(S, "mod", [(fmt2, fmt2), _u_units([0x61, 0x110000], 2)], [_Op(0), _Op(1)], "past:value")
    fmt3 = np.array(["%s", "%d"], dtype=TD.build())
    fn_case(S, "mod", [(fmt3, fmt3), _u_units([0xD800, 0x61, 0x78, 0], 2)], [_Op(0), _Op(1)], "surr:then-err")
    tnul = np.array(["a\x00", "a\x00b\x00\x00", "\x00"], dtype=TD.build())
    fn_case(S, "mod", [(fmt2, fmt2), (tnul, tnul)], [_Op(0), _Op(1)], "nul:stripped")
    for spec in [TD, TDN, TDC]:
        empty = np.zeros((0, 3), dtype=spec.build())
        fn_case(S, "mod", [t_plain(empty)], [_Op(0), 5], f"{_t_label(spec)}:empty")
    zero = np.array("%d", dtype=TD.build())
    fn_case(S, "mod", [t_plain(zero)], [_Op(0), 5], "0d")
    # The length sweep: formats and values of every length class (cell / arena texts).
    rng = random.Random(2623)
    for w in [0, 1, 7, 14, 15, 16, 17, 31, 64, 100]:
        fmts = ["".join(rng.choice("abc é中\U00010400") for _ in range(w)) + "%s" for _ in range(6)]
        vals = ["".join(rng.choice("xyz é") for _ in range(rng.randint(0, w))) for _ in range(6)]
        fbase = np.array(fmts, dtype=TD.build())
        vbase = np.array(vals, dtype=TD.build())
        fn_case(S, "mod", [(fbase, fbase), (vbase, vbase)], [_Op(0), _Op(1)], f"len{w}")
    # Long results (past NumSharp's 64-element row sample; the result packed cell by cell): growing widths, floats, texts
    # with non-ASCII characters, and a surrogate deferred behind a later formatting error.
    lrng = random.Random(2625)
    grow = np.array([1] * 64 + [(-1) ** k * 10 ** min(18, k // 4) for k in range(66)], dtype="i8")
    floats = np.array([lrng.uniform(-1, 1) * 10.0 ** lrng.randint(-8, 15) for _ in range(130)])
    texts = np.array(["".join(lrng.choice("abcé中") for _ in range(min(12, k // 10))) for k in range(130)])
    for f, vals, kind in [("%d", grow, "int"), ("%-12d|", grow, "int"), ("<%s>", grow, "int"), ("%.2f", floats, "float"),
                          ("%s", floats, "float"), ("%g", floats, "float"), ("<%s>", texts, "text"), ("%r", texts, "text")]:
        fbase = np.array([f], dtype=TD.build())
        fn_case(S, "mod", [(fbase, fbase), plain(vals)], [_Op(0), _Op(1)], f"long:{kind}:{f}")
    chars = np.array(["k"] * 130, dtype="<U2")
    chars[120] = "\ud800"
    cfmt = np.array(["%c"], dtype=TD.build())
    fn_case(S, "mod", [(cfmt, cfmt), plain(chars)], [_Op(0), _Op(1)], "long:surr")
    chars2 = chars.copy()
    chars2[125] = "ab"
    fn_case(S, "mod", [(cfmt, cfmt), plain(chars2)], [_Op(0), _Op(1)], "long:surr-then-err")
    # Random formats over random values (seeded).
    for i in range(160):
        spec = rng.choice([TD, TDS, TDN, TDNAN])
        kind = rng.choice(["int", "float", "str"])
        convs = {"int": "dxXoeEfgGsr", "float": "eEfFgGsrd", "str": "sra"}[kind]
        n = rng.choice([4, 6])
        fmts = []
        for _ in range(n):
            f = rng.choice(["", "é", "x=", "%%", "|"]) + "%" + "".join(rng.sample("-+ 0#", rng.randint(0, 2))) + \
                rng.choice(["", "4", "11"]) + rng.choice(["", ".1", ".5"]) + rng.choice(convs)
            fmts.append(f)
        fbase = np.array(t_values(spec, n, 0, fmts, rng.choice([0, 0, 3])), dtype=spec.build())
        if kind == "int":
            vop = plain(np.array([rng.randint(-10**5, 10**5) for _ in range(n)]))
        elif kind == "float":
            vop = plain(np.array([rng.uniform(-1e5, 1e5) for _ in range(n)]))
        else:
            vb = np.array(["".join(rng.choice("abé中") for _ in range(rng.randint(0, 6))) for _ in range(n)], dtype=TD.build())
            vop = (vb, vb)
        fn_case(S, "mod", [(fbase, fbase), vop], [_Op(0), _Op(1)], f"rand{i}:{_t_label(spec)}:{kind}")


# ----------------------------------------------------------------------------------------------------------------------
# Sections strcodec / tcodec: numpy.strings.encode / decode (strings phase 2, P2.6 milestone 4), strict
# ----------------------------------------------------------------------------------------------------------------------

# The codecs NumSharp implements, by a canonical name each (the C fast paths, the registry's Unicode and charmap modules,
# 'undefined'), and the spellings that reach them through _Py_normalize_encoding / the alias table. The multi-byte CJK
# codecs, utf-7, the escape codecs, idna, punycode and mbcs / oem are refused with NotSupportedException (unit-tested), so
# they stay out of the corpus.
CODEC_NAMES = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le", "utf-32-be", "latin-1",
               "ascii", "cp1252", "cp437", "cp037", "koi8_r", "mac_roman", "iso8859_15", "cp1251", "hp_roman8", "cp864",
               "undefined"]
CODEC_ALIASES = ["UTF8", "U8", "utf_8", "UTF-8", "  utf 8 ", "utf--8", "Latin-1", "latin1", "iso-8859-1", "iso8859_1", "8859",
                 "L1", "cp65001", "UTF-16LE", "u16", "utf_32_be", "windows-1252", "US-ASCII", "646", "ebcdic-cp-us",
                 "csASCII", "iso_8859_15:1998", "UTF-8-SIG"]
# Every error-handler name: the absent one (None: strict, and the errors argument dropped), the built-ins, an unknown
# name and a wrong case (both LookupError at the first error only).
CODEC_HANDLERS = [None, "strict", "ignore", "replace", "backslashreplace", "xmlcharrefreplace", "namereplace",
                  "surrogateescape", "surrogatepass", "bogus", "Strict"]
# The texts an encode meets: ASCII (with an embedded NUL), the Latin-1 range, the BMP (code-page characters, Cyrillic,
# CJK), the supplementary planes (an emoji, Deseret), lone surrogates (escapable U+DC80..U+DCFF runs, high / low, a pair
# held as two code points), named and unnamed characters for namereplace (aliases in the PUA plane 15, a Hangul syllable,
# a CJK unified / compatibility ideograph, Tangut, unassigned U+0378), and the empty str.
CODEC_U_POOL = ["", "a", "abc", "Hello World", "x\x00y", "é", "café", "\xff\xa0", "€", "Привет", "中文字", "Ā",
                "\U0001F600", "a\U00010400b", "aé€\U0001F600", "\ud800", "a\udc80b", "\udcff\udc80", "😀",
                "\udfffz", "x\udc80\ud800y", "\U000F0000\U000F0001", "가", "豈\U0002F800", "\U00017000", "͸",
                "\x80\x9f", "ß ", "€" * 12]
# The bytes a decode meets: ASCII (an embedded NUL), valid UTF-8 of every length, every invalid UTF-8 shape (a stray
# continuation byte, a truncated sequence, an overlong lead, an encoded surrogate, past U+10FFFF, F5..FF), UTF-16 / UTF-32
# data (whose trailing NUL byte NumPy's np.bytes_ drops), BOMs, Latin-1 and code-page bytes, and the empty bytes.
CODEC_S_POOL = [b"", b"a", b"abc", b"x\x00y", b"caf\xc3\xa9", b"\xe2\x82\xac", b"\xf0\x9f\x98\x80", b"\xff", b"\x80",
                b"a\xc3", b"\xe2\x82", b"\xed\xa0\x80", b"\xc0\xaf", b"\xf4\x90\x80\x80", b"\xf0\x80\x80\x80", b"\xf5z",
                b"\xe0\x80b", b"\xef\xbb\xbfabc", b"\xff\xfea\x00b\x00", b"\xfe\xff\x00a\x00b", b"a\x00\x00\xd8b\x00",
                b"a\x00\x00\xdc", b"\xff\xfe\x00\x00a\x00\x00\x00", b"a\x00\x00\x00\x00\xd8\x00\x00", b"\xe9\xff", b"\x81\x8d",
                b"\x00\x00\x11\x00", b"ab\xedcd\xa0"]


def gen_strcodec():
    """Section strcodec: numpy.strings.encode over U and decode over S, strict — every layout kind of both classes;
    every codec NumSharp implements x every error handler x every text of the pools, one element per case (so each text's
    own answer, error or result, is recorded); the codec spellings; the arguments' conversion and Argument Clinic's checks
    in CPython's order (a non-str encoding / errors of every kind, NULs, surrogates, long names cut at 400 bytes, unknown
    and non-text codecs, the empty-bytes shortcut of decode); _clean_args (errors without an encoding); arrays of encodings
    and errors broadcast; empty, 0-d and Python operands; units past U+10FFFF; long arrays past the 64-element row sample
    NumSharp lays its result out from (growing widths, fast and codec routes mixed); random texts x codecs x handlers."""
    import random
    S = "strcodec"
    # A. Every layout kind x both classes, the default codec and two others.
    for spec, name, pool in [("<U6", "encode", ["abc", "é", "", "x\x00y", "Hello", "€"]),
                             ("<U12", "encode", ["abc", "naïve", "", "Привет", "a\U0001F600", "zz"]),
                             ("|S6", "decode", [b"abc", b"\xc3\xa9", b"", b"x\x00y", b"Hello", b"\xe2\x82\xac"]),
                             ("|S12", "decode", [b"abc", b"na\xc3\xafve", b"", b"\xd0\x9f", b"a\xf0\x9f\x98\x80", b"zz"])]:
        for layout in LAYOUTS:
            op = fn_operand(spec, layout, seed=1, pool=pool)
            lab = f"{spec}:{layout}"
            fn_case(S, name, [op], [_Op(0)], lab)
            fn_case(S, name, [op], [_Op(0), "latin-1", "replace"], f"{lab}:latin1")
            fn_case(S, name, [op], [_Op(0), "utf-16"], f"{lab}:utf16")
    # B. Every codec x handler x text, one element per case.
    for codec in CODEC_NAMES:
        for handler in CODEC_HANDLERS:
            args = [codec] if handler is None else [codec, handler]
            for k, text in enumerate(CODEC_U_POOL):
                op = plain(np.array([text], dtype=f"<U{max(1, len(text))}"))
                fn_case(S, "encode", [op], [_Op(0)] + args, f"enc:{codec}:{handler}:{k}")
            for k, data in enumerate(CODEC_S_POOL):
                op = plain(np.array([data], dtype=f"|S{max(1, len(data))}"))
                fn_case(S, "decode", [op], [_Op(0)] + args, f"dec:{codec}:{handler}:{k}")
    # C. The codec spellings: each over a pool array under three handlers.
    upool = ["abc", "é", "€", "Ā", "\ud800", ""]
    spool = [b"abc", b"\xc3\xa9", b"\xe9", b"\xff\xfea\x00", b""]
    for alias in CODEC_ALIASES:
        for handler in [None, "replace", "backslashreplace"]:
            args = [alias] if handler is None else [alias, handler]
            fn_case(S, "encode", [plain(np.array(upool))], [_Op(0)] + args, f"alias:enc:{alias}:{handler}")
            fn_case(S, "decode", [plain(np.array(spool))], [_Op(0)] + args, f"alias:dec:{alias}:{handler}")
    # D. Argument Clinic's checks, in CPython's order (the a-scalar first, the encoding before the errors, the empty-bytes
    #    shortcut of decode before the lookup), _clean_args, and the codec lookup's errors.
    u3 = plain(np.array(["abc", "é", ""]))
    s3 = plain(np.array([b"abc", b"\xc3\xa9", b""]))
    se = plain(np.array([b"", b""]))
    sx = plain(np.array([b"", b"x"]))
    for name, op in [("encode", u3), ("decode", s3), ("decode", se), ("decode", sx)]:
        lab = f"{name}:{op[1].dtype.str}:{op[1].size}"
        for label, enc, err in [("bytes-enc", b"utf-8", None), ("int-enc", 5, None), ("float-enc", 1.5, None),
                                ("bool-enc", True, None), ("bytes-err", "utf-8", b"strict"), ("int-err", "utf-8", 5),
                                ("float-err", "utf-8", 2.5), ("bool-err", "ascii", False), ("both-bad", 5, b"x"),
                                ("nul-enc", "utf\x008", None), ("nul-err", "utf-8", "str\x00ict"),
                                ("nul-tail-enc", "utf-8\x00", None),
                                ("surr-enc", "\ud800", None), ("surr-enc2", "utf\udfff\ud800-8", None),
                                ("surr-err", "utf-8", "x\udfff"), ("unknown", "nope", None), ("empty-name", "", None),
                                ("base64", "base64", None), ("rot13", "rot13", "strict"), ("hex", "hex", None),
                                ("zlib", "zlib_codec", None), ("long-enc", "b" * 450, None),
                                ("long-err", "ascii", "x" * 450), ("long-nontext", "base64" + " " * 420, None),
                                ("long-err-utf8", "ascii", "é" * 300), ("none-enc", None, "ignore"),
                                ("none-err", "ascii", None), ("strict-name", "ascii", "strict"),
                                ("odd-name", "ascii", "Replace"), ("astral-enc", "a\U00010400b", None)]:
            args = [enc] if err is None else [enc, err]
            fn_case(S, name, [op], [_Op(0)] + args, f"{lab}:{label}")
    # Object lists as arguments (NumPy's object arrays: a str element passes, None / numbers do not).
    for name, op in [("encode", u3), ("decode", s3)]:
        for label, enc in [("obj-none", ["utf-8", None, "ascii"]), ("obj-int", ["utf-8", "ascii", 5]),
                           ("obj-ok", ["utf-8", "latin-1", "ascii"]), ("obj-big", ["utf-8", 2**70, "ascii"])]:
            fn_case(S, name, [op], [_Op(0), enc], f"{name}:args:{label}")
            fn_case(S, name, [op], [_Op(0), "ascii", enc], f"{name}:args:{label}:errors")
    # E. Arrays of encodings and errors, broadcast (per-element codecs, a varying handler, 2-D, a mismatch, a ragged list).
    fn_case(S, "encode", [plain(np.array(["a", "b", "é"])), plain(np.array(["utf-8", "utf-16", "latin-1"]))],
                [_Op(0), _Op(1)], "arr:enc")
    fn_case(S, "encode", [plain(np.array(["é", "€"])), plain(np.array([["replace"], ["ignore"]]))],
                [_Op(0), "ascii", _Op(1)], "arr:err2d")
    fn_case(S, "encode", [plain(np.array([["a"], ["bé"]])), plain(np.array(["utf-8", "latin-1"]))],
                [_Op(0), _Op(1)], "arr:bcast2d")
    fn_case(S, "encode", [plain(np.array(["a", "b"])), plain(np.array(["utf-8", "ascii", "latin-1"]))],
                [_Op(0), _Op(1)], "arr:mismatch")
    fn_case(S, "encode", [plain(np.array(["a"]))], [_Op(0), [["a", "b"], ["c"]]], "arr:ragged")
    fn_case(S, "encode", [plain(np.array(["é", "é", "é", "é"]))],
                [_Op(0), ["ascii", "latin-1", "ascii", "nope"], ["replace", "strict", "backslashreplace", "ignore"]],
                "arr:mixed-later-error")
    fn_case(S, "decode", [plain(np.array([b"a\xff", b"b\xff"]))], [_Op(0), "utf-8", ["ignore", "replace"]], "arr:dec-err")
    fn_case(S, "decode", [plain(np.array([b"a", b"\xff"]))], [_Op(0), ["utf-8", "nope"]], "arr:dec-first-err")
    fn_case(S, "decode", [plain(np.array([b"", b"\xff", b""]))], [_Op(0), ["nope", "latin-1", "base64"]],
                "arr:dec-empty-skips-lookup")
    rng = random.Random(4413)
    for i in range(12):
        names = [rng.choice(["utf-8", "latin-1", "ascii", "utf-16", "cp1252", "utf-32-be", "koi8_r"]) for _ in range(8)]
        handlers = [rng.choice(["replace", "ignore", "backslashreplace", "xmlcharrefreplace", "namereplace"]) for _ in range(8)]
        texts = [rng.choice(["abc", "é", "€", "Привет", "中", "", "a\U0001F600"]) for _ in range(8)]
        fn_case(S, "encode", [plain(np.array(texts)), plain(np.array(names)), plain(np.array(handlers))],
                    [_Op(0), _Op(1), _Op(2)], f"arr:rand{i}")
    # F. Empty, 0-d and Python operands.
    for name, spec in [("encode", "<U3"), ("decode", "|S3")]:
        fn_case(S, name, [plain(np.zeros(0, dtype=spec))], [_Op(0)], f"{name}:empty")
        fn_case(S, name, [plain(np.zeros((2, 0), dtype=spec))], [_Op(0), "nope", 5], f"{name}:empty2d-bad-args")
        fn_case(S, name, [plain(np.zeros(3, dtype=spec))], [_Op(0), "utf-16"], f"{name}:all-empty-elements")
    fn_case(S, "encode", [plain(np.zeros((0, 3), dtype="<U2")), plain(np.array(["a", "b", "c"]))], [_Op(0), _Op(1)],
            "encode:empty-bcast")
    fn_case(S, "encode", [plain(np.array("aé€"))], [_Op(0), "latin-1", "xmlcharrefreplace"], "encode:0d")
    fn_case(S, "decode", [plain(np.array(b"a\xffb"))], [_Op(0), "ascii", "surrogateescape"], "decode:0d")
    fn_case(S, "encode", [], ["abc\U0001F600", "utf-16-be"], "encode:pystr")
    fn_case(S, "decode", [], [b"abc\xc3\xa9", "utf-8"], "decode:pybytes")
    fn_case(S, "encode", [plain(np.array([1, 2]))], [_Op(0)], "encode:int")
    fn_case(S, "decode", [plain(np.array([1.5]))], [_Op(0)], "decode:float")
    fn_case(S, "encode", [plain(np.array([b"a"]))], [_Op(0), "nope"], "encode:S")
    fn_case(S, "decode", [plain(np.array(["a"]))], [_Op(0), "nope"], "decode:U")
    # G. U units past U+10FFFF: a one-unit element is CPython's SystemError at the scalar; a longer one carries the unit
    #    through each codec's arithmetic (UTF-8's truncated four-byte pattern, UTF-16's "surrogates", UTF-32's value).
    for codec, handler in [("utf-8", None), ("utf-16", None), ("utf-32-be", None), ("latin-1", "backslashreplace"),
                           ("ascii", "xmlcharrefreplace"), ("ascii", "namereplace"), ("cp1252", "replace"),
                           ("cp1252", "backslashreplace"), ("utf-16-le", "strict"), ("ascii", None)]:
        args = [codec] if handler is None else [codec, handler]
        fn_case(S, "encode", [_u_units([0x61, 0x110000, 0x62, 0xFFFFFFFF], 2)], [_Op(0)] + args,
                    f"past:kept:{codec}:{handler}")
        fn_case(S, "encode", [_u_units([0x110000, 0, 0x61, 0], 2)], [_Op(0)] + args, f"past:alone:{codec}:{handler}")
        fn_case(S, "encode", [_u_units([0x61, 0x7FFFFFFF, 0x62, 0x80000000], 2)], [_Op(0)] + args,
                    f"past:big:{codec}:{handler}")
    # H. Long arrays: past the 64-element sample, widths growing after it (the rows re-laid out), the direct row paths
    #    and the codec route mixed element by element.
    lrng = random.Random(4417)
    long_u = [("a" * min(40, k // 3)) + ("é" if k % 7 == 0 else "") + ("€" if k % 11 == 0 else "") for k in range(150)]
    long_s = [x.encode("utf-8") for x in long_u]
    for codec, handler in [("utf-8", None), ("ascii", "replace"), ("latin-1", "xmlcharrefreplace"), ("utf-16", None),
                           ("utf-32-le", None), ("cp1252", "namereplace"), ("ascii", "backslashreplace"),
                           ("utf-8-sig", None), ("koi8_r", "replace")]:
        args = [codec] if handler is None else [codec, handler]
        fn_case(S, "encode", [plain(np.array(long_u))], [_Op(0)] + args, f"long:enc:{codec}:{handler}")
    for codec, handler in [("utf-8", None), ("ascii", "replace"), ("latin-1", None), ("utf-8", "surrogateescape"),
                           ("utf-16-le", "replace"), ("cp1252", "backslashreplace"), ("ascii", "ignore")]:
        args = [codec] if handler is None else [codec, handler]
        fn_case(S, "decode", [plain(np.array(long_s))], [_Op(0)] + args, f"long:dec:{codec}:{handler}")
    shrink_u = ["x" * 30 + "é"] * 70 + ["ab"] * 80
    fn_case(S, "encode", [plain(np.array(shrink_u))], [_Op(0)], "long:enc:sample-wider")
    late = ["a"] * 100 + ["é" * 50] + ["b"] * 30
    fn_case(S, "encode", [plain(np.array(late))], [_Op(0)], "long:enc:late-grow")
    fn_case(S, "decode", [plain(np.array([x.encode() for x in late]))], [_Op(0)], "long:dec:late-grow")
    late_err = ["ok"] * 120 + ["\ud800"] + ["ok"] * 9
    fn_case(S, "encode", [plain(np.array(late_err))], [_Op(0)], "long:enc:late-error")
    fn_case(S, "decode", [plain(np.array([b"ok"] * 120 + [b"\xff"] + [b"ok"] * 9))], [_Op(0)], "long:dec:late-error")
    lens = [lrng.randint(0, 24) for _ in range(200)]
    rand_u = ["".join(lrng.choice("abcdefgh") for _ in range(n)) for n in lens]
    rand_u[150] = rand_u[150] + "é"
    rand_u[170] = "€" * 20
    for codec in ["utf-8", "latin-1", "ascii", "utf-16-be"]:
        fn_case(S, "encode", [plain(np.array(rand_u))], [_Op(0), codec, "replace"], f"long:enc:rand:{codec}")
    rand_s = [x.encode("utf-8") for x in rand_u]
    for codec in ["utf-8", "latin-1", "ascii", "utf-16-le"]:
        fn_case(S, "decode", [plain(np.array(rand_s))], [_Op(0), codec, "replace"], f"long:dec:rand:{codec}")
    # I. Random texts x codecs x handlers (seeded): code points drawn from ASCII, Latin-1, the BMP, surrogates and the
    #    supplementary planes; bytes drawn from ASCII, continuation, lead and invalid bytes.
    cp_ranges = [(0x20, 0x7E), (0x80, 0xFF), (0x100, 0x17F), (0x400, 0x4FF), (0x4E00, 0x4E40), (0xD800, 0xDFFF),
                 (0xDC80, 0xDCFF), (0x1F600, 0x1F64F), (0xF0000, 0xF0010), (0xAC00, 0xAC10)]
    for i in range(300):
        n = rng.randint(1, 6)
        texts = ["".join(chr(rng.randint(*rng.choice(cp_ranges))) for _ in range(rng.randint(0, 8))) for _ in range(n)]
        width = max(1, max(len(t) for t in texts))
        codec = rng.choice(CODEC_NAMES[:-1])
        handler = rng.choice(CODEC_HANDLERS[1:9])
        fn_case(S, "encode", [plain(np.array(texts, dtype=f"<U{width}"))], [_Op(0), codec, handler],
                    f"rand:enc{i}:{codec}:{handler}")
    byte_kinds = [range(0x20, 0x7F), range(0x80, 0xC0), range(0xC2, 0xF5), [0xC0, 0xC1, 0xF5, 0xFF, 0xED, 0xE0, 0xF0, 0xF4],
                  [0x00]]
    for i in range(300):
        n = rng.randint(1, 6)
        datas = [bytes(rng.choice(list(rng.choice(byte_kinds))) for _ in range(rng.randint(0, 10))) for _ in range(n)]
        width = max(1, max(len(d) for d in datas))
        codec = rng.choice(CODEC_NAMES[:-1])
        handler = rng.choice(CODEC_HANDLERS[1:9])
        fn_case(S, "decode", [plain(np.array(datas, dtype=f"|S{width}"))], [_Op(0), codec, handler],
                    f"rand:dec{i}:{codec}:{handler}")


def gen_tcodec():
    """Section tcodec: numpy.strings.encode over NEP 55 StringDType text, strict — the layout kinds x every NA kind (an NA
    object that is not a str is the method descriptor's TypeError at its element; a string NA encodes as its text; no NA
    object reads as the empty str); every codec x handler over StringDType pools (cells short and long, non-ASCII,
    supplementary-plane characters); StringDType arrays as the encoding / errors arguments (a None NA element is the clinic
    TypeError); the descriptor check before the arguments' checks; empty and 0-d operands; long arrays; decode refused."""
    import random
    S = "tcodec"
    tpool = ["abc", "é", "", "Привет", "a\U0001F600", "x" * 20, "中文", "a\x00b", "€uro", "zz"]
    for spec in T_COMPUTE_SPECS:
        lab0 = _t_label(spec)
        for layout in T_COMPUTE_LAYOUTS:
            for na_every in (0, 5):
                op = t_operand(spec, layout, seed=1, pool=tpool, na_every=na_every)
                fn_case(S, "encode", [op], [_Op(0)], f"{lab0}:{layout}:na{na_every}")
                fn_case(S, "encode", [op], [_Op(0), "latin-1", "replace"], f"{lab0}:{layout}:na{na_every}:latin1")
    for codec in CODEC_NAMES:
        for handler in CODEC_HANDLERS:
            args = [codec] if handler is None else [codec, handler]
            for spec in [TD, TDS]:
                vals = np.array(t_values(spec, 10, 0, tpool, 4), dtype=spec.build())
                fn_case(S, "encode", [(vals, vals)], [_Op(0)] + args, f"{_t_label(spec)}:{codec}:{handler}")
            for k, text in enumerate(["é", "€", "\U0001F600", "中", "x" * 40 + "€", "Ā"]):
                one = np.array([text], dtype=TD.build())
                fn_case(S, "encode", [(one, one)], [_Op(0)] + args, f"T1:{codec}:{handler}:{k}")
    # NA order: a non-str NA object before / after an element the codec refuses; with a bad encoding argument.
    for spec in [TDN, TDNAN]:
        for values in (["a", spec.na], [spec.na, "a"], ["é", spec.na]):
            base = np.array(values, dtype=spec.build())
            for label, args in [("ok", []), ("ascii", ["ascii"]), ("bad-enc", [5]), ("unknown", ["nope"])]:
                fn_case(S, "encode", [(base, base)], [_Op(0)] + args, f"{_t_label(spec)}:naorder:{label}:{values!r}")
    # StringDType arguments.
    ops = plain(np.array(["a", "b", "é"]))
    for spec, values in [(TD, ["utf-16", "ascii", "latin-1"]), (TDN, ["utf-8", None, "ascii"]),
                         (TDS, ["utf-8", "NA", "ascii"])]:
        arg = np.array(values, dtype=spec.build())
        fn_case(S, "encode", [ops, (arg, arg)], [_Op(0), _Op(1)], f"Targ:{_t_label(spec)}:encoding")
        fn_case(S, "encode", [ops, (arg, arg)], [_Op(0), "ascii", _Op(1)], f"Targ:{_t_label(spec)}:errors")
    for spec in [TD, TDN]:
        empty = np.zeros((0, 3), dtype=spec.build())
        fn_case(S, "encode", [t_plain(empty)], [_Op(0), "nope"], f"{_t_label(spec)}:empty")
    zero = np.array("aé€", dtype=TD.build())
    fn_case(S, "encode", [t_plain(zero)], [_Op(0), "ascii", "namereplace"], "0d")
    tb = np.array(["a"], dtype=TD.build())
    fn_case(S, "decode", [(tb, tb)], [_Op(0)], "decode:T")
    # Long arrays (past the 64-element row sample; cells copied straight into the rows, the codec route mixed in).
    lrng = random.Random(4421)
    vals = ["".join(lrng.choice("abcdefgh") for _ in range(lrng.randint(0, 30))) for _ in range(160)]
    vals[100] = vals[100] + "€"
    vals[140] = "é" * 40
    for spec in [TD, TDS]:
        arr = np.array(t_values(spec, 160, 0, vals, 0 if spec is TD else 9), dtype=spec.build())
        for codec, handler in [("utf-8", None), ("ascii", "replace"), ("latin-1", "backslashreplace"), ("utf-16", None),
                               ("cp1252", "xmlcharrefreplace")]:
            args = [codec] if handler is None else [codec, handler]
            fn_case(S, "encode", [(arr, arr)], [_Op(0)] + args, f"long:{_t_label(spec)}:{codec}:{handler}")


GENERATORS = [gen_dtypes, gen_creation, gen_astype, gen_views, gen_movement, gen_mutation, gen_access, gen_printing,
              gen_refusals, gen_string_requests, gen_text_precedence, gen_shape_decided, gen_validation_order, gen_surface,
              gen_strma, gen_striter, gen_strcomp, gen_strarith, gen_strconv, gen_tdtype, gen_tstore, gen_tcreate,
              gen_tmove, gen_tcast, gen_titer, gen_cvtext, gen_promote, gen_storenone, gen_tma, gen_tio,
              gen_strsort, gen_strchar, gen_strsearch, gen_strchararray, gen_strcharops, gen_strcharfn, gen_strcharobj,
              gen_strcharmeth, gen_strcharlist, gen_strsets, gen_tcomp, gen_tarith, gen_treduce, gen_tsort, gen_qfamily, gen_qprec, gen_strfn,
              gen_tfn, gen_strprop, gen_tprop, gen_strcase, gen_tcase, gen_strtrans, gen_ttrans, gen_strmod, gen_tmod,
              gen_strcodec, gen_tcodec]


def main():
    """Writes the whole corpus, one file per section, or with ``--sections a,b`` runs only those generators
    (``gen_<name>``) and rewrites only the files of the sections they produce — a development shortcut (the whole corpus
    takes minutes); the committed corpus is always a full run. A full run also removes the files of sections that no
    longer exist (the pre-split ``strings.jsonl`` included).

    :raises AssertionError: a section file larger than ``MAX_SECTION_BYTES``.
    """
    only = None
    if len(sys.argv) > 2 and sys.argv[1] == "--sections":
        only = set(sys.argv[2].split(","))
    for gen in GENERATORS:
        if only is None or gen.__name__[len("gen_"):] in only:
            gen()
    by_section = {}
    for c in CASES:
        by_section.setdefault(c["id"].split("/")[0], []).append(c)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    if only is None:
        for old in CORPUS_DIR.glob("strings*.jsonl"):
            if old.stem[len("strings_"):] not in by_section:
                old.unlink()
    sizes = {}
    for sec, cases in by_section.items():
        # allow_nan=False: a NaN / infinity in the params would be written as invalid JSON (Infinity); such a value
        # travels as a string instead (_json_floats).
        lines = [json.dumps(c, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n"
                 for c in cases]
        data = b"".join(lines)
        assert len(data) <= MAX_SECTION_BYTES, f"section {sec} is {len(data)} bytes, past {MAX_SECTION_BYTES}: split it"
        with open(section_path(sec), "wb") as f:
            f.write(data)
        sizes[sec] = len(data)
    errs = sum(1 for c in CASES if "error" in c)
    p2 = sum(1 for c in CASES if c["phase"] == 2)
    print(f"wrote {len(CASES)} cases ({errs} NumPy errors, {p2} phase-2) in {len(by_section)} files under {CORPUS_DIR}")
    print("  " + ", ".join(f"{k} {len(v)} ({sizes[k] / 1048576:.1f} MiB)" for k, v in by_section.items()))
    if _TITER_DROPPED:
        print(f"  titer: {len(_TITER_DROPPED)} cases dropped for NumPy 2.4.2's StringDType allocated-output MemoryError")
    if _TMA_TRACE_MISMATCH:
        print(f"  tma: {len(_TMA_TRACE_MISMATCH)} cases whose traced run differed (phase kept from the outcome): "
              + ", ".join(_TMA_TRACE_MISMATCH[:20]))


if __name__ == "__main__":
    sys.exit(main())
