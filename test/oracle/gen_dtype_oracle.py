"""numpy.dtype / DType-class (DTypeMeta) differential oracle — NumPy 2.4.2 is the source of truth.

Emits test/NumSharp.Tests/DTypes/corpus/dtype_oracle.<model>.jsonl, one JSON object per line, where <model> is the C
`long` model of the host NumPy runs on: `llp64` (Windows, C long is 32 bits) or `lp64` (Linux / macOS, 64 bits). The
two files differ only where NumPy itself does — the type numbers and characters of the 4- and 8-byte integers, and
which class the C-name aliases (IntDType, LongDType, LongLongDType, ...) are — and DTypeOracleTests.cs replays the
file of its own host's model, so every platform CI runs on is gated in full. Refresh both:

    python test/oracle/gen_dtype_oracle.py                                   (Windows NumPy 2.4.2 -> llp64)
    wsl -d Ubuntu -- ~/np242/bin/python test/oracle/gen_dtype_oracle.py      (Linux NumPy 2.4.2 -> lp64)

The DType class's C names (PyArray_DTypeMeta's fields, NPY_DType_Slots, the NEP 42 slot ids, the NPY_DT_* flags) are
read from NumPy's own headers: refs/numpy of this repository, or the source root named by NUMPY_SRC.

Record kinds (field "kind"):
  members  the names NumPy defines for the two objects: numpy.dtype's public attributes and dunders, numpy._DTypeMeta's
           own attributes, numpy.dtypes' public names, the C fields and slots of the DType class, the NPY_DT_* flag
           values and the byte-order macros. The replay asserts each exists in C# under NumPy's spelling.
  descr    one descriptor built by a recipe and every numpy.dtype attribute of it, plus str / repr / len, the class
           name and the KeyError text of d[0].
  error    a recipe NumPy refuses: the exception type and str(e).
  compare  two descriptors: ==, !=, <, <=, >, >= and whether their hashes are equal.
  promote  np.promote_types(a, b) or np.result_type(a): repr, isbuiltin, byteorder and metadata of the result.
  class    one numpy.dtypes class: __name__, __module__, repr, type.__name__, the four flag attributes and the
           no-argument class call (the instance's repr and whether it is the singleton, or the error).
  cast     one (from class, to class) pair: whether NumPy has a casting implementation, its name, and what its
           resolve_descriptors makes of the given descriptors (casting, loop descriptors, view_offset).

Recipes (field "recipe") build a descriptor step by step, and the C# replay has a twin of each step:
  ["dtype", spec, kw]      np.dtype(spec, **kw)      (kw may hold align / copy / metadata)
  ["dtype_of", kw]         np.dtype(d, **kw)
  ["newbyteorder", code]   d.newbyteorder(code)      (code None means the default 'S')
  ["instantiate", name]    getattr(np.dtypes, name)()

What is left out on purpose: specs whose NumPy class is not the canonical class of its width (NumPy's IntDType on
Windows, LongLongDType on Linux, LongDoubleDType everywhere — NumSharp has one class per width and kind, so those
spellings resolve to the canonical class there by design); every dtype NumSharp has no storage for (complex64,
strings, bytes, void, object); and the metadata=None TypeError (C#'s null means "not given").

Regenerate:  python test/oracle/gen_dtype_oracle.py   (requires numpy==2.4.2)
"""
import itertools
import json
import os
import re
import sys

import numpy as np
from numpy._core._multiarray_umath import _get_castingimpl

assert np.__version__ == "2.4.2", f"oracle must be generated with numpy 2.4.2, got {np.__version__}"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
CLONG_BITS = np.dtype("l").itemsize * 8
MODEL = "llp64" if CLONG_BITS == 32 else "lp64"
OUT = os.path.join(REPO, "test", "NumSharp.Tests", "DTypes", "corpus", f"dtype_oracle.{MODEL}.jsonl")
NUMPY_SRC = os.environ.get("NUMPY_SRC") or os.path.join(REPO, "refs", "numpy")

# The DType classes NumSharp has storage or descriptors for, canonical NumPy class first.
CLASSES = ["BoolDType", "Int8DType", "UInt8DType", "Int16DType", "UInt16DType", "Int32DType", "UInt32DType",
           "Int64DType", "UInt64DType", "Float16DType", "Float32DType", "Float64DType", "Complex128DType",
           "DateTime64DType", "TimeDelta64DType"]
NAMES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32",
         "float64", "complex128"]
SIZED = ["b1", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "c16"]
CHARS = ["?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "e", "f", "d", "D"]
UNITS = ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as"]
ATTRS = ["alignment", "base", "byteorder", "char", "descr", "fields", "flags", "hasobject", "isalignedstruct",
         "isbuiltin", "isnative", "itemsize", "kind", "metadata", "name", "names", "ndim", "num", "shape", "str",
         "subdtype", "type"]
BYTEORDER_CODES = [None, "S", "s", "<", ">", "=", "|", "L", "l", "B", "b", "N", "n", "I", "i", "little", "big",
                   "native", "swap", "ignore", "lx", "Sx"]

cases = []
seen_ids = set()


def emit(kind, cid, **fields):
    """Appends one record; ids are unique so a failing replay names exactly one case."""
    cid = f"{kind}/{cid}"
    if cid in seen_ids:
        return
    seen_ids.add(cid)
    rec = {"kind": kind, "id": cid}
    rec.update(fields)
    cases.append(rec)


def run(recipe):
    """Builds the descriptor a recipe describes. Every step has a twin in DTypeOracleTests.Build (C#)."""
    d = None
    for step in recipe:
        op = step[0]
        if op == "dtype":
            d = np.dtype(step[1], **(step[2] if len(step) > 2 else {}))
        elif op == "dtype_of":
            d = np.dtype(d, **step[1])
        elif op == "newbyteorder":
            d = d.newbyteorder() if step[1] is None else d.newbyteorder(step[1])
        elif op == "instantiate":
            d = getattr(np.dtypes, step[1])()
        else:
            raise ValueError(op)
    return d


def err(fn):
    """(type name, str(e)) of the exception fn raises, or None when it returns."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — the oracle records whatever NumPy raises
        return {"type": type(e).__name__, "text": str(e)}
    return None


def attr_text(d, a):
    """The Python text of one attribute: strings raw, the scalar type by name, metadata as a dict repr, else repr."""
    v = getattr(d, a)
    if a == "type":
        return v.__name__
    if a == "metadata":
        return None if v is None else repr(dict(v))
    if isinstance(v, str):
        return v
    return repr(v)


def canonical_class(d):
    """True when d's class is the class NumPy itself uses for d's name — the only kind NumSharp models (one class
    per width and kind). False for the C-name classes that exist beside it (IntDType on Windows, LongLongDType on
    Linux, LongDoubleDType)."""
    return type(d) is type(np.dtype(d.name))


def descr_record(cid, recipe):
    """One `descr` record, or an `error` record when NumPy refuses the recipe."""
    try:
        d = run(recipe)
    except Exception as e:  # noqa: BLE001
        emit("error", cid, recipe=recipe, error={"type": type(e).__name__, "text": str(e)})
        return None
    if not canonical_class(d):
        return None
    emit("descr", cid, recipe=recipe,
         attrs={a: attr_text(d, a) for a in ATTRS},
         str=str(d), repr=repr(d), len=len(d), cls=type(d).__name__,
         getitem=err(lambda: d[0]))
    return d


# ---------------------------------------------------------------------------------------------------------------------
# members
# ---------------------------------------------------------------------------------------------------------------------

def struct_fields(text, typedef_name):
    """The member names of `typedef struct { ... } <typedef_name>;` (function pointers and arrays included)."""
    # [^{}]* keeps the match inside ONE struct body (a lazy .*? would run on from an earlier typedef struct).
    m = re.search(r"typedef struct[^{}]*\{([^{}]*)\}\s*" + re.escape(typedef_name) + r"\s*;", text, re.S)
    if not m:
        raise SystemExit(f"struct {typedef_name} not found in NumPy's headers")
    body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
    names = []
    for line in body.split(";"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        mm = re.search(r"\*?\s*(\w+)\s*(\[[^\]]*\])?\s*$", line)
        if mm:
            names.append(mm.group(1))
    return names


def members():
    """The `members` record: every name NumPy defines for numpy.dtype and the DType class."""
    api = open(os.path.join(NUMPY_SRC, "numpy", "_core", "include", "numpy", "dtype_api.h"), encoding="utf-8").read()
    meta_h = open(os.path.join(NUMPY_SRC, "numpy", "_core", "src", "multiarray", "dtypemeta.h"), encoding="utf-8").read()
    slots = [m.group(2) for m in re.finditer(r"#define (_)?NPY_DT_([a-z_]+)\s+(\d+)\s*$", api, re.M)]
    flags = {}
    for m in re.finditer(r"#define (NPY_DT_[A-Z_]+)\s+1\s*<<\s*(\d+)", api + "\n" + meta_h):
        flags[m.group(1)] = 1 << int(m.group(2))
    meta_fields = [f for f in struct_fields(api, "PyArray_DTypeMeta") if f != "super"]
    slot_fields = struct_fields(meta_h, "NPY_DType_Slots")

    d = np.dtype("f8")
    obj = set(dir(object))
    public = sorted(a for a in dir(d) if not a.startswith("_"))
    dunders = sorted(a for a in dir(np.dtype)
                     if a.startswith("__") and (a not in obj or getattr(np.dtype, a, None) is not getattr(object, a, None)))
    meta = type(np.dtype)
    meta_own = sorted(a for a in dir(meta) if a not in set(dir(type)))
    module = sorted(a for a in dir(np.dtypes) if not a.startswith("_"))
    native = np.dtype(int).newbyteorder("S").newbyteorder("S").byteorder
    swapped = np.dtype(int).newbyteorder("S").byteorder
    emit("members", "all",
         dtype_public=public, dtype_dunders=dunders, dtypemeta_own=meta_own, dtypes_module=module,
         dt_slots=slots, dtypemeta_fields=meta_fields, dtype_slot_fields=slot_fields, dt_flags=flags,
         byteorder_macros={"NPY_LITTLE": "<", "NPY_BIG": ">", "NPY_NATIVE": "=", "NPY_SWAP": "s", "NPY_IGNORE": "|",
                           "NPY_NATBYTE": native, "NPY_OPPBYTE": swapped})


# ---------------------------------------------------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------------------------------------------------

def descriptors():
    """Every descriptor NumSharp can build: names, sized and character codes, byte orders, datetime units, the
    newbyteorder codes over a mix of descriptors, the copy / align / metadata keywords and the class calls."""
    for spec in NAMES + SIZED + CHARS:
        descr_record(f"spec:{spec}", [["dtype", spec]])
    for spec in SIZED:
        for bo in "<>=|":
            descr_record(f"spec:{bo}{spec}", [["dtype", bo + spec]])
    for letter in "Mm":
        descr_record(f"spec:{letter}", [["dtype", letter]])
        descr_record(f"spec:{letter}8", [["dtype", letter + "8"]])
        for unit in UNITS:
            descr_record(f"spec:{letter}8[{unit}]", [["dtype", f"{letter}8[{unit}]"]])
        for extra in ["[10ns]", "[5s]", "[2Y]", "[3h]", "[25us]", "[1000ms]"]:
            descr_record(f"spec:{letter}8{extra}", [["dtype", letter + "8" + extra]])
        for bo in "<>=|":
            descr_record(f"spec:{bo}{letter}8[ns]", [["dtype", f"{bo}{letter}8[ns]"]])
            descr_record(f"spec:{bo}{letter}8", [["dtype", f"{bo}{letter}8"]])
    for spec in ["datetime64", "datetime64[ns]", "timedelta64", "timedelta64[25us]"]:
        descr_record(f"spec:{spec}", [["dtype", spec]])

    # newbyteorder: every code over native / explicit / non-native / single-byte / datetime descriptors, and chains.
    bases = ["f8", ">f8", "<f8", "i4", ">i4", "c16", ">c16", "f2", "u2", "b1", "u1", "i1", "M8[ns]", ">M8[ns]",
             "m8[s]", ">m8"]
    for base in bases:
        for code in BYTEORDER_CODES:
            descr_record(f"nbo:{base}:{code}", [["dtype", base], ["newbyteorder", code]])
        for c1, c2 in itertools.product(["S", "<", ">", "=", "|"], repeat=2):
            descr_record(f"nbo2:{base}:{c1}{c2}", [["dtype", base], ["newbyteorder", c1], ["newbyteorder", c2]])
        for code in ["X", "", "Xyz", "'", "1", " "]:
            descr_record(f"nbo_err:{base}:{code!r}", [["dtype", base], ["newbyteorder", code]])

    # The np.dtype keywords: copy copies only a singleton, align is a no-op for these dtypes, metadata always copies and
    # merges into the descriptor's own metadata without overriding existing keys.
    for spec in ["f8", ">f8", "b1", "i8", "c16", "M8[ns]", "m8", "f2"]:
        descr_record(f"kw:{spec}:copy", [["dtype", spec, {"copy": True}]])
        descr_record(f"kw:{spec}:align", [["dtype", spec, {"align": True}]])
        descr_record(f"kw:{spec}:align_copy", [["dtype", spec, {"align": True, "copy": True}]])
        descr_record(f"kw:{spec}:meta", [["dtype", spec, {"metadata": {"a": 1}}]])
        descr_record(f"kw:{spec}:meta_empty", [["dtype", spec, {"metadata": {}}]])
        descr_record(f"kw:{spec}:meta_mixed", [["dtype", spec, {"metadata": {"k": "v", "n": 2.5, "b": True}}]])
        descr_record(f"kw:{spec}:copy_of_copy", [["dtype", spec, {"copy": True}], ["dtype_of", {"copy": True}]])
        descr_record(f"kw:{spec}:dtype_of", [["dtype", spec], ["dtype_of", {}]])
        descr_record(f"kw:{spec}:meta_merge",
                     [["dtype", spec, {"metadata": {"a": 1}}], ["dtype_of", {"metadata": {"a": 5, "c": 3}}]])
        descr_record(f"kw:{spec}:meta_then_copy", [["dtype", spec, {"metadata": {"a": 1}}], ["dtype_of", {"copy": True}]])
        descr_record(f"kw:{spec}:meta_then_nbo", [["dtype", spec, {"metadata": {"a": 1}}], ["newbyteorder", "S"]])
        descr_record(f"kw:{spec}:nbo_then_meta", [["dtype", spec], ["newbyteorder", "<"], ["dtype_of", {"metadata": {"z": 0}}]])

    for name in CLASSES:
        descr_record(f"call:{name}", [["instantiate", name]])


# ---------------------------------------------------------------------------------------------------------------------
# comparisons and promotion
# ---------------------------------------------------------------------------------------------------------------------

COMPARE = [
    [["dtype", "f8"]], [["dtype", ">f8"]], [["dtype", "f8"], ["newbyteorder", "<"]], [["dtype", "f8"], ["newbyteorder", "="]],
    [["dtype", "f4"]], [["dtype", "f2"]], [["dtype", "c16"]], [["dtype", ">c16"]], [["dtype", "i1"]], [["dtype", "u1"]],
    [["dtype", "i2"]], [["dtype", "u2"]], [["dtype", "i4"]], [["dtype", ">i4"]], [["dtype", "u4"]], [["dtype", "i8"]],
    [["dtype", "u8"]], [["dtype", "b1"]], [["dtype", "M8"]], [["dtype", "M8[ns]"]], [["dtype", "M8[s]"]],
    [["dtype", ">M8[ns]"]], [["dtype", "M8[1000ms]"]], [["dtype", "m8"]], [["dtype", "m8[ns]"]], [["dtype", "m8[Y]"]],
    [["dtype", "f8", {"metadata": {"a": 1}}]], [["dtype", "f8", {"copy": True}]],
]


def label(recipe):
    """A short readable label for a recipe, used in case ids."""
    parts = []
    for step in recipe:
        if step[0] == "dtype":
            kw = step[2] if len(step) > 2 else {}
            parts.append(step[1] + ("" if not kw else "{" + ",".join(sorted(kw)) + "}"))
        elif step[0] == "newbyteorder":
            parts.append(f".nbo({step[1]})")
        else:
            parts.append(f".{step[0]}")
    return "".join(parts)


def comparisons():
    """Every ordered pair of COMPARE: the six rich comparisons and hash equality."""
    for ra, rb in itertools.product(COMPARE, repeat=2):
        a, b = run(ra), run(rb)
        emit("compare", f"{label(ra)}|{label(rb)}", a=ra, b=rb,
             eq=bool(a == b), ne=bool(a != b), lt=bool(a < b), le=bool(a <= b), gt=bool(a > b), ge=bool(a >= b),
             hash_eq=hash(a) == hash(b))


def promote_result(fn):
    """The facts of a promotion result (or the error)."""
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001
        return {"error": {"type": type(e).__name__, "text": str(e)}}
    return {"repr": repr(r), "isbuiltin": r.isbuiltin, "byteorder": r.byteorder,
            "metadata": None if r.metadata is None else repr(dict(r.metadata))}


def promotions():
    """np.promote_types over every pair of COMPARE, and np.result_type of each one alone."""
    for ra, rb in itertools.product(COMPARE, repeat=2):
        a, b = run(ra), run(rb)
        emit("promote", f"promote_types:{label(ra)}|{label(rb)}", op="promote_types", a=ra, b=rb,
             result=promote_result(lambda: np.promote_types(a, b)))
    for ra in COMPARE:
        a = run(ra)
        emit("promote", f"result_type:{label(ra)}", op="result_type", a=ra, b=None,
             result=promote_result(lambda: np.result_type(a)))


# ---------------------------------------------------------------------------------------------------------------------
# DType classes
# ---------------------------------------------------------------------------------------------------------------------

def classes():
    """Every public numpy.dtypes class: its Python surface and the class call."""
    for name in sorted(a for a in dir(np.dtypes) if not a.startswith("_")):
        cls = getattr(np.dtypes, name)
        try:
            inst = cls()
            call = {"repr": repr(inst), "is_singleton": inst is cls()}
        except Exception as e:  # noqa: BLE001
            call = {"error": {"type": type(e).__name__, "text": str(e)}}
        canonical = None
        if "repr" in call:
            try:
                canonical = type(np.dtype(cls().name)).__name__
            except Exception:  # noqa: BLE001 — a class whose instance name does not parse (StringDType) has no canonical twin
                canonical = None
        emit("class", name, name=name, __name__=cls.__name__, __module__=cls.__module__, repr=repr(cls),
             type_name=cls.type.__name__, _abstract=cls._abstract, _parametric=cls._parametric, _legacy=cls._legacy,
             _is_numeric=cls._is_numeric, call=call, canonical=canonical)


# ---------------------------------------------------------------------------------------------------------------------
# casting implementations
# ---------------------------------------------------------------------------------------------------------------------

NUMERIC_INSTANCES = {
    "BoolDType": ["b1"], "Int8DType": ["i1"], "UInt8DType": ["u1"], "Int16DType": ["i2", ">i2"],
    "UInt16DType": ["u2", ">u2"], "Int32DType": ["i4", ">i4"], "UInt32DType": ["u4", ">u4"],
    "Int64DType": ["i8", ">i8"], "UInt64DType": ["u8", ">u8"], "Float16DType": ["f2", ">f2"],
    "Float32DType": ["f4", ">f4"], "Float64DType": ["f8", ">f8"], "Complex128DType": ["c16", ">c16"],
}
TIME_INSTANCES = {
    "DateTime64DType": ["M8", "M8[ns]", ">M8[ns]", "M8[s]", "M8[Y]", "M8[10ms]", "M8[1000ms]", "M8[D]"],
    "TimeDelta64DType": ["m8", "m8[ns]", ">m8[ns]", "m8[s]", "m8[Y]", "m8[10ms]", "m8[1000ms]", "m8[D]"],
}


def instances(cls_name):
    """The descriptor specs a cast record gives for a class."""
    return NUMERIC_INSTANCES.get(cls_name) or TIME_INSTANCES[cls_name]


def casts():
    """Every ordered pair of CLASSES: the implementation's existence and name, and resolve_descriptors over the given
    descriptors (the destination None = "please pick", or each instance of the destination class)."""
    for fn, tn in itertools.product(CLASSES, CLASSES):
        impl = _get_castingimpl(getattr(np.dtypes, fn), getattr(np.dtypes, tn))
        name = None if impl is None else re.search(r"`(.*?)`", repr(impl)).group(1)
        emit("cast", f"{fn}->{tn}", frm=fn, to=tn, exists=impl is not None, name=name, given=None, resolved=None)
        if impl is None:
            continue
        froms = instances(fn)
        tos = [None] + instances(tn)
        # Keep the cross-kind datetime pairs small: every from with None, and the first two of each side otherwise.
        if fn in TIME_INSTANCES or tn in TIME_INSTANCES:
            pairs = [(f, None) for f in froms] + [(f, t) for f in froms[:3] for t in tos[1:]]
        else:
            pairs = list(itertools.product(froms, tos))
        for f, t in pairs:
            given = (np.dtype(f), None if t is None else np.dtype(t))
            try:
                casting, loop, view = impl._resolve_descriptors(given)
                resolved = {"casting": casting & 63, "same_value": bool(casting & 64),
                            "loop": [repr(loop[0]), repr(loop[1])], "view_offset": view}
            except Exception as e:  # noqa: BLE001
                resolved = {"error": {"type": type(e).__name__, "text": str(e)}}
            emit("cast", f"{fn}->{tn}:{f},{t}", frm=fn, to=tn, exists=True, name=name, given=[f, t], resolved=resolved)


def main():
    members()
    descriptors()
    comparisons()
    promotions()
    classes()
    casts()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"kind": "meta", "id": "meta/host", "numpy": np.__version__, "platform": sys.platform,
                             "clong_bits": CLONG_BITS, "model": MODEL, "byteorder": sys.byteorder},
                            separators=(",", ":")) + "\n")
        for c in cases:
            fh.write(json.dumps(c, separators=(",", ":"), sort_keys=False) + "\n")
    kinds = {}
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"wrote {len(cases)} cases {kinds} -> {OUT}")


if __name__ == "__main__":
    main()
