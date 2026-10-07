#!/usr/bin/env python3
"""NumPy 2.4.2 oracle for NumSharp's fixed-width string dtypes ('S' bytes, 'U' str).

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
  afterwards).
* ``error`` -- when NumPy raises: the exception's first well-known class on its MRO and ``str(e)`` verbatim.
* ``phase`` -- 1 for the calls NumSharp's string support covers now (storage, element access, conversion, copying,
  string-to-string casts, indexing, printing, and every refusal NumPy itself raises); 2 for the calls NumPy answers by
  COMPUTING on text (comparisons, concatenation, truth tests, sorting and searching, text <-> number conversion). A
  phase-2 case passes while NumSharp throws ``StringComputationNotSupportedException`` (counted as deferred) and is
  compared strictly as soon as NumSharp computes it.

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
        return {"float": float(v)}
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
    raise TypeError(f"cannot encode {type(v)}")


KNOWN_ERRORS = ["UnicodeEncodeError", "UnicodeDecodeError", "DTypePromotionError", "AxisError", "ValueError",
                "TypeError", "IndexError", "OverflowError", "SystemError", "KeyError", "ZeroDivisionError",
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


def array_expect(r, operands):
    """An array result: dtype, shape, C-order bytes, element strides, flags and memory sharing with each operand."""
    r = np.asarray(r)
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


def expect_of(r, operands):
    """The expectation for any result NumPy returned."""
    if isinstance(r, np.ndarray):
        return array_expect(r, operands)
    if isinstance(r, (tuple, list)) and r and all(isinstance(x, np.ndarray) for x in r):
        return {"kind": "tuple", "items": [array_expect(x, operands) for x in r]}
    if isinstance(r, np.dtype):
        return {"kind": "dtype", "str": r.str, "repr": repr(r)}
    if isinstance(r, (bool, np.bool_)):
        return {"kind": "bool", "value": bool(r)}
    if isinstance(r, str) and not isinstance(r, np.str_):
        # UTF-32 hex like every other text: a printed lone surrogate is not valid JSON text.
        return {"kind": "text", "u": r.encode("utf-32-le", "surrogatepass").hex()}
    if isinstance(r, (np.str_, np.bytes_, bytes, list)):
        return {"kind": "value", "value": enc(r)}
    if isinstance(r, np.generic):
        # A numeric NumPy scalar (an index, a count): compared as a 0-d array.
        return array_expect(np.asarray(r), None)
    raise TypeError(f"unexpected result type {type(r)}")


def add(section, op, *, params=None, operands=(), call, phase=1, layout="", mutate=None, note=""):
    """Run one NumPy call and record its outcome.

    ``operands`` is a list of (base, view) pairs; ``call(*views)`` makes the call. ``mutate`` names the operand an
    in-place call writes: its view and whole base buffer afterwards are the expectation. ``phase`` is 1, 2, or
    ``"compute"`` for a call NumPy may answer by computing on text: phase 2 when NumPy returns a result, phase 1 when it
    raises (NumPy's error is then final, so NumSharp must raise the same) -- except a PARSE error (NumPy converting text
    to a number, ``could not convert string to float``), whose outcome depends on the text and is phase 2 too.
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
            case["expected"] = {"kind": "mutated", "operand": mutate, "view": array_expect(view, None),
                                "base": base.tobytes().hex()}
        else:
            case["expected"] = expect_of(r, views)
        if phase == "compute":
            case["phase"] = 2
    except Exception as e:  # noqa: BLE001 -- every NumPy exception is a recorded outcome
        case["error"] = err_info(e)
        if phase == "compute":
            case["phase"] = 2 if _is_parse_error(e) else 1
    CASES.append(case)


def _is_parse_error(e):
    """NumPy's text-to-number conversion failing on a value (``could not convert string to float: np.str_('a')``,
    ``invalid literal for int() with base 10``): an outcome of the computational phase, decided by the text."""
    return isinstance(e, ValueError) and str(e).startswith(
        ("could not convert string to", "invalid literal for int()", "complex() arg is a malformed string"))


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
    # None stores its text under a string dtype (NumPy's str(None)); text into numbers / numbers into text compute.
    add("creation", "array", params={"values": enc([None, "a"]), "dtype": "U"},
        call=lambda: np.array([None, "a"], dtype="U"))
    add("creation", "array", params={"values": enc([None]), "dtype": "S5"}, call=lambda: np.array([None], dtype="S5"))
    add("creation", "array", params={"values": enc(["a", 1]), "dtype": None}, phase=2,
        call=lambda: np.array(["a", 1]))
    add("creation", "array", params={"values": enc(["12", "3"]), "dtype": "i8"}, phase=2,
        call=lambda: np.array(["12", "3"], dtype="i8"))
    add("creation", "array", params={"values": enc([1.5, 2.0]), "dtype": "U"}, phase=2,
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
    add("creation", "full", params={"shape": [2], "fill": enc(7), "dtype": "U3"}, phase=2,
        call=lambda: np.full((2,), 7, dtype="U3"))
    add("creation", "ones", params={"shape": [2], "dtype": "U3"}, phase=2, call=lambda: np.ones((2,), dtype="U3"))

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
            add("creation", "ones_like", params={}, operands=[operand(spec, lay)], layout=lay, phase=2,
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
    # Text <-> numbers: NumPy parses / formats (the computational phase).
    for vals, t in [(["12", "-3"], "i8"), (["1.5", "nan"], "f8"), (["ab"], "i4"), (["True", ""], "?"),
                    ([b"7", b"8"], "u1"), (["1+2j"], "c16")]:
        add("astype", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
            operands=[plain(np.array(vals))], layout="c1", phase=2, call=lambda a, t=t: a.astype(t))
    for arr, t in [(np.array([1, -2], dtype="i8"), "U"), (np.array([1.5, 2.0]), "S"), (np.array([True]), "U5"),
                   (np.array([300], dtype="i2"), "S2")]:
        add("astype", "astype", params={"dtype": t, "casting": "unsafe", "order": "K", "copy": True},
            operands=[plain(arr)], layout="c1", phase=2, call=lambda a, t=t: a.astype(t))


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
                layout=lay, phase=2, call=lambda a: np.pad(a, 1))

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
            layout="c1+int", phase=2, call=lambda x, y: np.concatenate([x, y]))
        text3 = plain(np.array(["xy"] * 3 if spec[1] == "U" else [b"xy"] * 3))
        add("movement", "where", params={"cond": [True, False, True]}, operands=[text3, plain(np.array([1, 2, 3]))],
            layout="c1+int", phase=2, call=lambda x, y: np.where(np.array([True, False, True]), x, y))
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
                layout=lay, mutate=0, phase=2, call=lambda a: _setitem(a, 0, 5))
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
            operands=[operand(spec, "c1"), plain(np.arange(6))], layout="c1", mutate=0, phase=2,
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
            operands=[operand(spec, "c1"), plain(np.arange(6))], layout="c1", mutate=0, phase=2,
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
    # numbers (i0 and linalg.norm cast to float first) — phase 2 when NumPy returns or raises a parsing error.
    computes = {"argmax", "argmin", "nanargmax", "nanargmin", "all", "any", "count_nonzero", "nonzero", "flatnonzero",
                "argwhere", "sort", "argsort", "unique", "trim_zeros", "lexsort", "searchsorted", "partition",
                "argpartition", "isin", "union1d", "intersect1d", "setdiff1d", "setxor1d", "array_equal", "array_equiv",
                "logical_not"}
    parses = {"i0", "linalg.norm"}
    for name in reductions:
        for k, a in (("U", u), ("S", s)):
            phase = 2 if name in parses else "compute" if name in computes else 1
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
    outs = {"ro": (lambda: plain(np.zeros(6)), True), "f8": (lambda: plain(np.zeros(6)), False),
            "i8": (lambda: plain(np.zeros(6, np.int64)), False), "b": (lambda: plain(np.zeros(6, bool)), False),
            "U1": (lambda: plain(np.zeros(6, "<U1")), False), "U9": (lambda: plain(np.zeros(6, "<U9")), False),
            "S9": (lambda: plain(np.zeros(6, "S9")), False), "f8x7": (lambda: plain(np.zeros(7)), False),
            "U9x23": (lambda: plain(np.zeros((2, 3), "<U9")), False)}
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
        target = o1 if which == 0 else o2
        target[1].flags.writeable = False
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
                # A numeric dtype= makes NumPy PARSE the text (array(m, dtype=float64)): the computational phase.
                parses = dt == "f8" and key in ("u", "s", "u2d", "e", "u3d")
                add("strcov", "cov", params={"corr": corr, "dtype": dt}, operands=[op], layout=key,
                    phase=2 if parses else "compute", call=lambda a, f=fn, d=dt: f(a, dtype=d))
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

def main():
    gen_dtypes()
    gen_creation()
    gen_astype()
    gen_views()
    gen_movement()
    gen_mutation()
    gen_access()
    gen_printing()
    gen_refusals()
    gen_string_requests()
    gen_text_precedence()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        for c in CASES:
            f.write(json.dumps(c, ensure_ascii=True, separators=(",", ":")).encode("ascii") + b"\n")
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
