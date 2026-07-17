"""
gen_npy_oracle.py — emit a committed, bytes-exact NumPy 2.4.2 oracle for the .npy/.npz format.

NumPy is the oracle. This script writes REAL `np.save` / `np.savez` / `np.savez_compressed`
output into a single zip; the C# harness (`test/NumSharp.UnitTest/IO/NpyOracleTests.cs`)
replays it with NO Python at test time or in CI.

Output: test/NumSharp.UnitTest/IO/corpus/npy_oracle.zip
  manifest.json          the case list (schema below)
  cases/<name>.npy|.npz  the exact bytes NumPy produced

Each case asserts up to three independent things:

  read   — NumSharp loads the file; dtype + shape + raw C-order buffer must equal `ns_bytes`.
           `ns_bytes` is the array's LOGICAL values in NumSharp's native in-memory form, so
           big-endian files must byte-swap and fortran_order files must transpose to match.
  write  — NumSharp rebuilds the array from `ns_bytes` and saves it; the produced file must be
           BYTE-IDENTICAL to NumPy's. Only set where NumSharp can natively represent the dtype
           and layout (see `write_exact` below).
  error  — NumSharp must throw, and the message must contain `load_error` (verbatim NumPy text).

Case schema:
  {
    "name":          "float64_2d_c",
    "file":          "cases/float64_2d_c.npy",   # entry inside this zip
    "kind":          "npy" | "npz" | "raw",
    "descr":         "<f8",                      # dtype descriptor in the header
    "np_dtype":      "float64",
    "ns_dtype":      "Double",                   # NPTypeCode name NumSharp must load as (null => error case)
    "shape":         [2, 3],
    "fortran_order": false,
    "version":       [1, 0],
    "data_offset":   128,                        # where the data starts (always 64-aligned)
    "ns_bytes":      "<hex>",                    # logical values, native-endian, C-order
    "write_exact":   true,                       # if true, NumSharp's writer must reproduce `file` byte-for-byte
    "load_error":    null,                       # if set, loading MUST throw containing this text
    "note":          "..."                       # why this case exists
  }

NPZ cases carry `entries`: {name -> {ns_dtype, shape, ns_bytes}} instead of a single array.

Regenerate (deterministic; needs numpy==2.4.2):
    python test/oracle/gen_npy_oracle.py
"""
import io
import json
import os
import struct
import sys
import warnings
import zipfile

import numpy as np
from numpy.lib import _format_impl as fmt

np.seterr(all="ignore")
warnings.simplefilter("ignore")  # the 2.0/3.0 "stored array in format X" warnings are expected here

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "NumSharp.UnitTest", "IO", "corpus", "npy_oracle.zip",
)

# NumPy dtype -> NPTypeCode. Char (UTF-16) and Decimal have no NumPy analog and are handled
# specially: Char rides NumPy's 4-byte '<U1', Decimal cannot be written at all.
NS_DTYPE = {
    "bool": "Boolean",
    "uint8": "Byte",
    "int8": "SByte",
    "int16": "Int16",
    "uint16": "UInt16",
    "int32": "Int32",
    "uint32": "UInt32",
    "int64": "Int64",
    "uint64": "UInt64",
    "float16": "Half",
    "float32": "Single",
    "float64": "Double",
    "complex128": "Complex",
}

# dtypes NumSharp writes natively, in NumPy's declaration order
NATIVE_DTYPES = list(NS_DTYPE.keys())

CASES = []


def _ns_bytes(arr):
    """The logical values as NumSharp holds them in memory: native-endian, C-order.

    This is what `read` compares against, so it must be independent of how the file stored
    them (byte order / fortran_order / striding).
    """
    if arr.dtype.kind == "U":
        # NumSharp's Char is a 2-byte UTF-16 code unit; NumPy's '<U1' is 4-byte UCS-4.
        #
        # Deliberately NOT "".join(arr.tolist()): NumPy's '<U' dtype treats NUL as string padding,
        # so an element holding U+0000 reads back as '' and joining silently drops it — the
        # expectation would then be one element short of what the file actually holds. Read the raw
        # code points instead.
        buf = np.ascontiguousarray(arr.astype("<U1")).tobytes()
        cps = struct.unpack(f"<{len(buf) // 4}I", buf)
        return b"".join(struct.pack("<H", c) for c in cps).hex()
    native = arr.dtype.newbyteorder("=")
    if native == np.dtype("complex64"):
        native = np.dtype("complex128")  # NumSharp widens c8 -> System.Numerics.Complex
    return np.ascontiguousarray(arr.astype(native)).tobytes("C").hex()


def _ns_dtype_of(arr):
    if arr.dtype.kind == "U":
        return "Char"
    native = arr.dtype.newbyteorder("=")
    if native == np.dtype("complex64"):
        return "Complex"  # NumSharp has no 64-bit complex; c8 widens on read
    return NS_DTYPE[native.name]


def add_npy(name, arr, *, version=None, write_exact=True, note=""):
    """Save `arr` with real NumPy and register it as a case."""
    buf = io.BytesIO()
    fmt.write_array(buf, arr, version=version)
    raw = buf.getvalue()

    major = raw[6]
    hlen_size = 2 if major == 1 else 4
    hlen = int.from_bytes(raw[8:8 + hlen_size], "little")

    CASES.append({
        "name": name,
        "file": f"cases/{name}.npy",
        "kind": "npy",
        "bytes": raw,  # popped before manifest serialization
        "descr": fmt.dtype_to_descr(arr.dtype),
        "np_dtype": arr.dtype.name,
        "ns_dtype": _ns_dtype_of(arr),
        "shape": [int(d) for d in arr.shape],
        "fortran_order": bool(arr.flags.f_contiguous and not arr.flags.c_contiguous),
        "version": [raw[6], raw[7]],
        "data_offset": 8 + hlen_size + hlen,
        "ns_bytes": _ns_bytes(arr),
        "write_exact": write_exact,
        "load_error": None,
        "note": note,
    })


def add_raw(name, raw, *, load_error, note="", ext="npy", load_via="load", max_header_size=None):
    """A hand-built (usually malformed) file that must fail to load with `load_error`.

    `load_via` picks the entry point, because the two disagree on purpose: np.load dispatches on
    magic bytes and treats anything unrecognized as a pickle, so a CORRUPT magic never reaches
    read_magic and surfaces the pickle message instead. numpy.lib.format.read_array (NumSharp's
    np.load_npy) is handed a .npy directly and does report the magic error.
    """
    CASES.append({
        "name": name,
        "file": f"cases/{name}.{ext}",
        "kind": "raw",
        "bytes": raw,
        "descr": None, "np_dtype": None, "ns_dtype": None,
        "shape": None, "fortran_order": None, "version": None, "data_offset": None,
        "ns_bytes": None, "write_exact": False,
        "load_error": load_error,
        "load_via": load_via,
        "max_header_size": max_header_size,
        "note": note,
    })


def add_npz(name, arrays, *, compressed=False, note="", write_exact=False):
    buf = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(buf, **arrays)
    CASES.append({
        "name": name,
        "file": f"cases/{name}.npz",
        "kind": "npz",
        "bytes": buf.getvalue(),
        "compressed": compressed,
        "descr": None, "np_dtype": None, "ns_dtype": None,
        "shape": None, "fortran_order": None, "version": None, "data_offset": None,
        "ns_bytes": None,
        "write_exact": write_exact,
        "load_error": None,
        "entries": {
            k: {"ns_dtype": _ns_dtype_of(v), "shape": [int(d) for d in v.shape], "ns_bytes": _ns_bytes(v)}
            for k, v in arrays.items()
        },
        "note": note,
    })


# ---------------------------------------------------------------------------------------------
# 1. Every NumSharp dtype x representative shapes, C-order, version 1.0 (the default write path)
# ---------------------------------------------------------------------------------------------
SHAPES = {
    "scalar": (),        # 0-d: count == 1, no growth padding (len(shape) == 0)
    "empty": (0,),       # zero elements, still a full 128-byte header
    "1d": (5,),          # trailing comma in the shape tuple: (5,)
    "2d": (2, 3),
    "3d": (2, 3, 4),
    "unit": (1,),
    "empty2d": (0, 4),   # zero-size but multi-dim
    "empty3d": (5, 0, 3),
}


def _values(dt, n):
    """Deterministic, dtype-appropriate values spanning the interesting range."""
    d = np.dtype(dt)
    if d.kind == "b":
        return np.arange(n) % 3 == 0
    if d.kind in "iu":
        info = np.iinfo(d)
        pool = [0, 1, info.max, info.min, info.max - 1] + list(range(2, 40))
        return np.array([pool[i % len(pool)] for i in range(n)], dtype=d)
    if d.kind == "f":
        finfo = np.finfo(d)
        pool = [0.0, -0.0, 1.0, -1.0, np.nan, np.inf, -np.inf,
                float(finfo.max), float(finfo.tiny), 0.5, -2.25, 3.14159]
        return np.array([pool[i % len(pool)] for i in range(n)], dtype=d)
    if d.kind == "c":
        re = _values(d.type(0).real.dtype, n)
        im = _values(d.type(0).real.dtype, n)[::-1]
        return (re + 1j * im).astype(d)
    if d.kind == "U":
        pool = list("aZ0 ~é中")
        return np.array([pool[i % len(pool)] for i in range(n)], dtype=d)
    raise AssertionError(dt)


for dt in NATIVE_DTYPES:
    for sname, shape in SHAPES.items():
        n = int(np.prod(shape)) if shape else 1
        arr = _values(dt, n).reshape(shape)
        add_npy(f"{dt}_{sname}", arr, note=f"{dt} {sname} C-order v1.0 — the default np.save path")

# Char rides '<U1' (2-byte UTF-16 in NumSharp <-> 4-byte UCS-4 in NumPy)
for sname, shape in SHAPES.items():
    n = int(np.prod(shape)) if shape else 1
    add_npy(f"char_{sname}", _values("<U1", n).reshape(shape),
            note="NumSharp Char <-> NumPy '<U1' (2-byte UTF-16 widened to 4-byte UCS-4)")

# ---------------------------------------------------------------------------------------------
# 2. Fortran order — the layout the old implementation rejected outright
# ---------------------------------------------------------------------------------------------
for dt in ["int32", "float64", "complex128", "bool", "float16"]:
    for shape in [(2, 3), (3, 2), (2, 3, 4), (1, 5), (0, 4)]:
        n = int(np.prod(shape))
        arr = np.asfortranarray(_values(dt, n).reshape(shape))
        sname = "x".join(map(str, shape))
        add_npy(f"fortran_{dt}_{sname}", arr, note=f"F-contiguous {dt} {shape} -> fortran_order: True")

# A transposed view is F-contiguous: NumPy stores fortran_order=True and writes the base bytes.
add_npy("fortran_transposed_view", np.arange(12, dtype=np.int32).reshape(3, 4).T,
        note="transposed view is F-contiguous -> fortran_order: True")

# 1-D and 0-d are BOTH C- and F-contiguous; the C check runs first, so fortran_order stays False.
add_npy("fortran_1d_is_c", np.asfortranarray(np.arange(5, dtype=np.int32)),
        note="1-D is both C- and F-contiguous; C is tested first -> fortran_order: False")
# NOTE: asfortranarray is documented "ndim >= 1", so it PROMOTES a 0-d input to shape (1,). This case
# therefore pins that promotion, not 0-d contiguity — the true 0-d case is `int32_scalar`.
_promoted = np.asfortranarray(np.array(7, dtype=np.int32))
assert _promoted.shape == (1,)
add_npy("fortran_scalar_promoted_to_1d", _promoted,
        note="asfortranarray promotes 0-d to shape (1,) (it is documented ndim>=1); the result is both "
             "C- and F-contiguous -> fortran_order: False. True 0-d coverage is the *_scalar cases.")

# ---------------------------------------------------------------------------------------------
# 3. Non-contiguous sources — NumPy copies them to C-order on write (fortran_order: False)
# ---------------------------------------------------------------------------------------------
add_npy("strided_step2", np.arange(12, dtype=np.int32)[::2],
        note="strided slice is neither C- nor F-contiguous -> written as a C-order copy")
add_npy("strided_reversed", np.arange(6, dtype=np.float64)[::-1],
        note="negative-stride view -> written as a C-order copy")
add_npy("strided_2d_col", np.arange(12, dtype=np.int32).reshape(3, 4)[:, ::2],
        note="column-strided 2-D view -> C-order copy")
add_npy("strided_offset_row", np.arange(12, dtype=np.int32).reshape(3, 4)[1:, :],
        note="sliced view with a non-zero base offset")
add_npy("broadcast_view", np.broadcast_to(np.arange(3, dtype=np.int32), (4, 3)),
        note="stride-0 broadcast view -> materialized C-order on write")

# ---------------------------------------------------------------------------------------------
# 4. Format versions 2.0 and 3.0 — rejected outright by the old implementation
# ---------------------------------------------------------------------------------------------
for ver in [(1, 0), (2, 0), (3, 0)]:
    tag = f"v{ver[0]}_{ver[1]}"
    add_npy(f"version_{tag}_int32", np.arange(6, dtype=np.int32).reshape(2, 3), version=ver,
            note=f"explicit format version {ver} (4-byte header length for 2.0/3.0; utf8 for 3.0)")
    add_npy(f"version_{tag}_fortran", np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3)),
            version=ver, note=f"format {ver} + fortran_order")

# A header big enough to FORCE version 2.0 (> 65535 bytes) via a many-field structured dtype is
# out of scope (structured dtypes unsupported), so force the version explicitly instead and also
# ship a genuine >64KB-header file for the 4-byte length-field path.
big_names = [(f"f{i:04d}", "u1") for i in range(12000)]
big_struct = np.zeros(1, dtype=np.dtype(big_names))
buf = io.BytesIO()
fmt.write_array(buf, big_struct)
assert buf.getvalue()[6] == 2, "12000 fields should force an auto-selected version 2.0 header"
# Raise the limit past this 218100-byte header so the load gets far enough to reject the dtype:
# this is the case that proves the 4-byte v2.0 length field is parsed at all.
add_raw("version_2_0_auto_large_header", buf.getvalue(),
        load_error="Structured dtypes are not supported", max_header_size=500_000,
        note="12000-field structured dtype forces an auto-selected v2.0 header (>65535 bytes). With the "
             "size guard raised, NumSharp must parse the 4-byte length and THEN reject the structured descr")
add_raw("version_2_0_auto_large_header_default_limit", buf.getvalue(),
        load_error="Header info length (218100) is large and may not be safe to load securely",
        note="the same file at the DEFAULT max_header_size: the size guard fires before the dtype is even parsed")

# ---------------------------------------------------------------------------------------------
# 5. Byte order — big-endian files must byte-swap to native on read
# ---------------------------------------------------------------------------------------------
for dt in ["int16", "int32", "int64", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex128"]:
    be = np.dtype(dt).newbyteorder(">")
    arr = _values(dt, 6).astype(be)
    add_npy(f"bigendian_{dt}", arr, write_exact=False,
            note=f"'>{np.dtype(dt).str[1:]}' big-endian -> byte-swapped to native on read; "
                 f"NumSharp always writes native, so it cannot reproduce these bytes")

# '=' (native) and '|' (not-applicable) descriptors must parse as native
add_raw_native = np.arange(3, dtype=np.int32)
for prefix, dt in [("=", "i4"), ("<", "i4")]:
    raw = fmt.magic(1, 0)
    body = ("{'descr': '%s%s', 'fortran_order': False, 'shape': (3,), }" % (prefix, dt)).encode()
    pad = 64 - ((8 + 2 + len(body) + 1) % 64)
    raw += struct.pack("<H", len(body) + 1 + pad) + body + b" " * pad + b"\n"
    raw += add_raw_native.tobytes()
    CASES.append({
        "name": f"endian_prefix_{'native' if prefix == '=' else 'little'}",
        "file": f"cases/endian_prefix_{'native' if prefix == '=' else 'little'}.npy",
        "kind": "npy", "bytes": raw,
        "descr": f"{prefix}{dt}", "np_dtype": "int32", "ns_dtype": "Int32",
        "shape": [3], "fortran_order": False, "version": [1, 0],
        "data_offset": 10 + len(body) + 1 + pad,
        "ns_bytes": add_raw_native.tobytes().hex(),
        "write_exact": False,
        "load_error": None,
        "note": f"'{prefix}' endian prefix must parse as native order",
    })

# ---------------------------------------------------------------------------------------------
# 6. complex64 -> widened to System.Numerics.Complex (read-only; NumSharp writes '<c16')
# ---------------------------------------------------------------------------------------------
add_npy("complex64_1d", _values("complex64", 6), write_exact=False,
        note="'<c8' complex64 widens to NumSharp's 128-bit Complex on read")
add_npy("complex64_2d", _values("complex64", 6).reshape(2, 3), write_exact=False,
        note="'<c8' complex64 2-D widen")

# ---------------------------------------------------------------------------------------------
# 7. Large-ish data — exercises the chunked read/write path (BUFFER_SIZE = 256 KB)
# ---------------------------------------------------------------------------------------------
add_npy("large_float64_100k", np.arange(100_000, dtype=np.float64),
        note="800 KB of data: >3 chunks through the 256 KB BUFFER_SIZE read loop")
add_npy("large_int8_300k", np.arange(300_000, dtype=np.int8),
        note="300 KB of 1-byte data: chunk boundary not a multiple of itemsize")
add_npy("large_fortran_2d", np.asfortranarray(np.arange(50_000, dtype=np.float64).reshape(500, 100)),
        note="400 KB F-order: chunked Fortran write path")

# ---------------------------------------------------------------------------------------------
# 8. Header edge cases
# ---------------------------------------------------------------------------------------------
add_npy("header_many_dims", np.zeros((1,) * 19, dtype=np.int8),
        note="19 dims pushes the header past 64 bytes -> hlen jumps 118 -> 182")
add_npy("header_big_growth_axis", np.zeros((100_000,), dtype=np.int8),
        note="6-digit growth axis shrinks the growth padding (21 - len(repr(dim)))")

# Already-64-aligned header: NumPy adds a FULL 64 bytes of padding rather than zero.
for extra in range(0, 80):
    body = ("{'descr': '|i1', 'fortran_order': False, 'shape': (1,), }" + " " * extra).encode()
    if (8 + 2 + len(body) + 1) % 64 == 0:
        wrapped = fmt._wrap_header(body.decode(), (1, 0))
        assert len(wrapped) % 64 == 0 and struct.unpack("<H", wrapped[8:10])[0] == len(body) + 1 + 64
        add_raw("header_exact_align_full_pad", wrapped + b"\x07",
                load_error=None,
                note="header already 64-aligned -> NumPy emits a FULL 64 bytes of padding, never 0")
        CASES[-1].update({
            "kind": "npy", "descr": "|i1", "np_dtype": "int8", "ns_dtype": "SByte",
            "shape": [1], "fortran_order": False, "version": [1, 0],
            "data_offset": len(wrapped), "ns_bytes": b"\x07".hex(),
        })
        break

# Whitespace / key-order / no-trailing-comma variants a reader MUST accept ("a reader MUST NOT
# depend on" the writer's alphabetical key order).
for vname, body_str in [
    ("header_reordered_keys", "{'shape': (3,), 'descr': '<i4', 'fortran_order': False}"),
    ("header_no_trailing_comma", "{'descr': '<i4', 'fortran_order': False, 'shape': (3,)}"),
    ("header_extra_whitespace", "{  'descr' :  '<i4' ,  'fortran_order' :  False ,  'shape' :  ( 3 , ) , }"),
    ("header_double_quotes", '{"descr": "<i4", "fortran_order": False, "shape": (3,), }'),
    ("header_py2_L_suffix", "{'descr': '<i4', 'fortran_order': False, 'shape': (3L,), }"),
]:
    body = body_str.encode()
    pad = 64 - ((8 + 2 + len(body) + 1) % 64)
    raw = fmt.magic(1, 0) + struct.pack("<H", len(body) + 1 + pad) + body + b" " * pad + b"\n"
    raw += np.arange(3, dtype=np.int32).tobytes()
    # Prove NumPy itself accepts each of these before demanding NumSharp does.
    assert np.array_equal(fmt.read_array(io.BytesIO(raw)), np.arange(3, dtype=np.int32)), vname
    CASES.append({
        "name": vname, "file": f"cases/{vname}.npy", "kind": "npy", "bytes": raw,
        "descr": "<i4", "np_dtype": "int32", "ns_dtype": "Int32",
        "shape": [3], "fortran_order": False, "version": [1, 0],
        "data_offset": 10 + len(body) + 1 + pad,
        "ns_bytes": np.arange(3, dtype=np.int32).tobytes().hex(),
        "write_exact": False, "load_error": None,
        "note": f"header variant NumPy's ast.literal_eval accepts: {body_str}",
    })

# ---------------------------------------------------------------------------------------------
# 9. Malformed / unsupported — NumSharp must throw NumPy's verbatim message
# ---------------------------------------------------------------------------------------------
def _hdr(body_str, ver=(1, 0), data=b""):
    body = body_str.encode()
    pad = 64 - ((8 + 2 + len(body) + 1) % 64)
    lenfmt = "<H" if ver[0] == 1 else "<I"
    return fmt.magic(*ver) + struct.pack(lenfmt, len(body) + 1 + pad) + body + b" " * pad + b"\n" + data


add_raw("bad_magic", b"\x93NUMPX\x01\x00" + b"\x00" * 120, load_via="load_npy",
        load_error="the magic string is not correct; expected b'\\x93NUMPY', got b'\\x93NUMPX'",
        note="magic \\x93 is byte 147, not the char '?' the old reader compared against")
add_raw("short_magic", b"\x93NU", load_via="load_npy",
        load_error="EOF: reading magic string, expected 8 bytes got 3",
        note="file shorter than the magic")
# ...but np.load dispatches on magic and never reaches read_magic for a corrupt one: an unrecognized
# magic is assumed to be a bare pickle. Same bytes, different entry point, different message.
add_raw("bad_magic_via_load", b"\x93NUMPX\x01\x00" + b"\x00" * 120,
        load_error="This file contains pickled (object) data",
        note="np.load treats an unrecognized magic as a pickle rather than reporting the magic error")
add_raw("short_magic_via_load", b"\x93NU",
        load_error="This file contains pickled (object) data",
        note="np.load on a too-short file: unrecognized magic -> pickle branch")
add_raw("bad_version_4_0", fmt.magic(4, 0) + struct.pack("<H", 118) + b" " * 117 + b"\n",
        load_error="we only support format version (1,0), (2,0), and (3,0), not (4, 0)",
        note="unknown major version")
add_raw("bad_version_1_1", fmt.magic(1, 1) + struct.pack("<H", 118) + b" " * 117 + b"\n",
        load_error="we only support format version (1,0), (2,0), and (3,0), not (1, 1)",
        note="known major, unknown minor")

_good = io.BytesIO()
fmt.write_array(_good, np.arange(3, dtype=np.int32))
add_raw("truncated_data", _good.getvalue()[:-4],
        load_error="EOF: reading array data, expected 12 bytes got 8",
        note="file ends mid-data")
add_raw("truncated_header", fmt.magic(1, 0) + struct.pack("<H", 118) + b"{'descr'",
        load_error="EOF: reading array header, expected 118 bytes got 8",
        note="file ends mid-header")
add_raw("truncated_header_length", fmt.magic(1, 0) + b"\x76",
        load_error="EOF: reading array header length, expected 2 bytes got 1",
        note="file ends inside the 2-byte header-length field")

add_raw("header_wrong_keys", _hdr("{'descr': '<i4', 'shape': (3,), }"),
        load_error="Header does not contain the correct keys: ['descr', 'shape']",
        note="missing fortran_order")
add_raw("header_extra_key", _hdr("{'descr': '<i4', 'fortran_order': False, 'shape': (3,), 'x': 1, }"),
        load_error="Header does not contain the correct keys: ['descr', 'fortran_order', 'shape', 'x']",
        note="unexpected 4th key")
add_raw("header_not_a_dict", _hdr("[1, 2, 3]"),
        load_error="Header is not a dictionary: [1, 2, 3]",
        note="header parses but is a list")
add_raw("header_bad_fortran", _hdr("{'descr': '<i4', 'fortran_order': 5, 'shape': (3,), }"),
        load_error="fortran_order is not a valid bool: 5",
        note="fortran_order must be a bool")
add_raw("header_shape_not_tuple", _hdr("{'descr': '<i4', 'fortran_order': False, 'shape': 3, }"),
        load_error="shape is not valid: 3",
        note="shape must be a tuple")
add_raw("header_shape_float", _hdr("{'descr': '<i4', 'fortran_order': False, 'shape': (3.5,), }"),
        load_error="shape is not valid: (3.5,)",
        note="shape entries must be ints")
add_raw("header_negative_shape", _hdr("{'descr': '<i4', 'fortran_order': False, 'shape': (-3,), }"),
        load_error="negative dimensions are not allowed",
        note="negative dim rejected at array construction")
add_raw("header_bad_descr", _hdr("{'descr': 'zz', 'fortran_order': False, 'shape': (3,), }"),
        load_error="descr is not a valid dtype descriptor: 'zz'",
        note="unparsable dtype descriptor")
add_raw("header_unparsable", _hdr("{'descr': "),
        load_error="Cannot parse header",
        note="truncated python literal")

# A tiny file whose header-length field claims a huge header. The message must name the CLAIMED
# size, and — the real point — reading it must not reserve that claim: NumPy shrugs these off in
# ~2 KB because Python's fp.read(n) only allocates what it returns, so allocating up front would
# turn a 28-byte file into a multi-gigabyte spike. NumSharp's ReadBytes grows as it reads.
add_raw("hostile_header_len_4gb", fmt.magic(2, 0) + struct.pack("<I", 0xFFFFFFF0) + b"{'descr': '|i1'}",
        load_via="load_npy", load_error="EOF: reading array header, expected 4294967280 bytes got 16",
        note="28-byte file claiming a ~4 GB header (v2.0's 4-byte length field)")
add_raw("hostile_header_len_1gb", fmt.magic(2, 0) + struct.pack("<I", 1_000_000_000) + b"{'descr': '|i1'}",
        load_via="load_npy", load_error="EOF: reading array header, expected 1000000000 bytes got 16",
        note="same trick just under int.MaxValue, where a naive size guard would not fire")
add_raw("hostile_header_len_64k_v1", fmt.magic(1, 0) + struct.pack("<H", 0xFFFF) + b"{'descr': '|i1'}",
        load_via="load_npy", load_error="EOF: reading array header, expected 65535 bytes got 16",
        note="v1.0's 2-byte length field claiming the maximum 65535")

# max_header_size: this file is legal but its header exceeds the 10000-byte default.
_pad_body = "{'descr': '|i1', 'fortran_order': False, 'shape': (1,), }" + " " * 11000
add_raw("header_over_max_size", _hdr(_pad_body) + b"\x01",
        load_error="is large and may not be safe to load securely",
        note="header > max_header_size (10000) -> refuse to parse; raise the limit or allow_pickle to bypass")

# Unsupported-but-valid dtypes: parse the header, then reject with a precise message.
for name, arr, msg in [
    ("dtype_object", np.array([{"a": 1}], dtype=object), "Object arrays cannot be loaded when allow_pickle=False"),
    ("dtype_structured", np.zeros(2, dtype=[("x", "<i4"), ("y", "<f8")]), "Structured dtypes are not supported"),
    ("dtype_datetime64", np.array(["2020-01-01"], dtype="datetime64[D]"), "datetime64"),
    ("dtype_timedelta64", np.array([5], dtype="timedelta64[s]"), "timedelta64"),
    ("dtype_bytestring", np.array([b"hello"], dtype="|S10"), "Byte-string dtypes"),
    ("dtype_unicode_multi", np.array(["hello"], dtype="<U5"), "only '<U1' maps to NumSharp's Char"),
    ("dtype_void", np.zeros(2, dtype="|V16"), "Void dtypes"),
]:
    b = io.BytesIO()
    fmt.write_array(b, arr)
    add_raw(name, b.getvalue(), load_error=msg,
            note=f"valid NumPy file NumSharp cannot represent: {fmt.dtype_to_descr(arr.dtype)}")

# float128/complex256 only exist on some platforms; emit hand-built headers so the corpus is
# identical on every OS.
add_raw("dtype_float128", _hdr("{'descr': '<f16', 'fortran_order': False, 'shape': (2,), }") + b"\x00" * 32,
        load_error="float128", note="'<f16' has no .NET analog")
add_raw("dtype_complex256", _hdr("{'descr': '<c32', 'fortran_order': False, 'shape': (2,), }") + b"\x00" * 64,
        load_error="complex256", note="'<c32' has no .NET analog")

# np.load-level file-type detection
add_raw("not_npy_not_zip", b"this is just some text, definitely not a numpy file at all",
        load_error="This file contains pickled (object) data",
        note="np.load falls through to pickle for unknown magic; NumSharp has no pickle -> same message")
add_raw("empty_file", b"",
        load_error="No data left in file",
        note="np.load on an empty file raises EOFError")

# ---------------------------------------------------------------------------------------------
# 9b. Value fidelity — exotic bit patterns that a naive conversion would round or normalize away
# ---------------------------------------------------------------------------------------------
def _f8(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _f4(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# NaN is not one value: the payload and sign bits must survive verbatim, and a signaling NaN must
# not be quieted. A raw byte copy gets this right; anything that routes through arithmetic will not.
_f8_nans = np.array([_f8(0x7FF8000000000000), _f8(0xFFF8000000000000),
                     _f8(0x7FF0000000000001), _f8(0x7FFDEADBEEFCAFE)], dtype="<f8")
add_npy("value_f8_nan_payloads", _f8_nans, note="float64 NaN payloads: quiet, negative, signaling, custom")
add_npy("value_f4_nan_payloads",
        np.array([_f4(0x7FC00000), _f4(0xFFC00000), _f4(0x7F800001), _f4(0x7FDEADBE)], dtype="<f4"),
        note="float32 NaN payloads incl. sNaN")
add_npy("value_f2_nan_payloads",
        np.frombuffer(struct.pack("<4H", 0x7E00, 0xFE00, 0x7C01, 0x7DAD), dtype="<f2"),
        note="float16 NaN payloads incl. sNaN")

add_npy("value_f8_subnormal",
        np.array([5e-324, -5e-324, 2.2250738585072014e-308, 1.1125369292536007e-308,
                  0.0, -0.0, np.finfo(np.float64).max, np.finfo(np.float64).min], dtype="<f8"),
        note="float64 smallest subnormal, tiny, +/-0.0 (the sign bit must survive), extremes")
add_npy("value_f4_subnormal",
        np.array([1.4e-45, -1.4e-45, 1.1754944e-38, 0.0, -0.0,
                  np.finfo(np.float32).max, np.finfo(np.float32).min], dtype="<f4"),
        note="float32 subnormals and signed zero")
add_npy("value_f2_subnormal", np.array([6e-8, -6e-8, 6.104e-5, 0.0, -0.0, 65504.0, -65504.0], dtype="<f2"),
        note="float16 subnormals, signed zero, max finite")

for _dt in ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]:
    _i = np.iinfo(_dt)
    add_npy(f"value_{_dt}_extremes",
            np.array([_i.min, _i.min + 1, -1 if _i.min < 0 else 0, 0, 1, _i.max - 1, _i.max], dtype=_dt),
            note=f"{_dt} min/max boundaries")

_cvals = [complex(0.0, -0.0), complex(-0.0, 0.0), complex(np.nan, 1), complex(1, np.nan),
          complex(np.inf, -np.inf), complex(np.nan, np.nan), complex(5e-324, 1e308)]
add_npy("value_c16_exotic", np.array(_cvals, dtype="<c16"),
        note="complex128 NaN/inf per component, signed zero, subnormal")
add_npy("value_c8_exotic", np.array(_cvals, dtype="<c8"), write_exact=False,
        note="complex64 -> Complex is a real WIDENING conversion, not a copy: NaN/inf must survive it")

# Big-endian + the conversion paths together: the swap unit is a COMPONENT, not an element.
add_npy("value_be_c16_exotic", np.array(_cvals, dtype=">c16"), write_exact=False,
        note="'>c16' swaps in 8s (per component), NOT in 16s — swapping the whole element corrupts it")
add_npy("value_be_c8_exotic", np.array(_cvals, dtype=">c8"), write_exact=False,
        note="'>c8' must byte-swap in 4s AND widen to complex128")
add_npy("value_be_f2_nan",
        np.frombuffer(struct.pack("<4H", 0x7E00, 0xFE00, 0x7C01, 0x7DAD), dtype="<f2").astype(">f2"),
        write_exact=False, note="'>f2' half NaN payloads survive the swap")
add_npy("value_be_f8_nan", _f8_nans.astype(">f8"), write_exact=False,
        note="'>f8' NaN payload survives the swap")
add_npy("value_be_bool", np.array([True, False, True], dtype=">b1"), write_exact=False,
        note="'>b1' is 1 byte: the swap must be a no-op, not a reversal")
add_npy("value_be_i1", np.array([-128, 0, 127], dtype=">i1"), write_exact=False,
        note="'>i1' is 1 byte: no-op swap")
add_npy("value_be_U1", np.array(["a", chr(0xFFFF), chr(0xE9), "Z"], dtype=">U1"), write_exact=False,
        note="'>U1': 4-byte code points swapped, then narrowed to UTF-16")

# Char/UCS-4 seams. U+0000 is the interesting one: NumPy's '<U' treats NUL as string padding, so
# NumPy itself reports this element as '' — the bytes are there, its Python-level view just hides it.
add_npy("value_U1_bmp_edges",
        np.array([chr(c) for c in [0x20, 0x00, 0x01, 0x7F, 0x80, 0xFF, 0x7FF, 0x800,
                                   0xD7FF, 0xE000, 0xFFFD, 0xFFFF]], dtype="<U1"),
        note="BMP seams: U+0000 (which NumPy's <U reports as ''), the latin-1/UTF-8 boundaries, "
             "U+D7FF/U+E000 either side of the surrogate block, and U+FFFF")

# Non-BMP fits NumPy's 4-byte <U1 but needs a surrogate PAIR in UTF-16, so it cannot fit one Char.
_astral = io.BytesIO()
fmt.write_array(_astral, np.array([chr(0x1F600)], dtype="<U1"))
add_raw("value_U1_astral_rejected", _astral.getvalue(), load_via="load_npy",
        load_error="outside the Basic Multilingual Plane",
        note="U+1F600 fits a 4-byte '<U1' but not a 2-byte Char — must be refused, never truncated")
add_raw("value_U1_lone_surrogate_rejected",
        _hdr("{'descr': '<U1', 'fortran_order': False, 'shape': (2,), }") + struct.pack("<2I", 0xD83D, 0x61),
        load_via="load_npy", load_error="unpaired surrogate",
        note="a lone surrogate is not a valid code point, even though the 4 bytes decode to one")

# ---------------------------------------------------------------------------------------------
# 9c. Python-grammar traps in the header (the parser stands in for ast.literal_eval)
# ---------------------------------------------------------------------------------------------
add_raw("header_tuple_no_comma", _hdr("{'descr': '|i1', 'fortran_order': False, 'shape': (1), }", data=b"\x07"),
        load_error="shape is not valid: 1",
        note="(1) is the INT 1 in grouping parens, not a 1-tuple — only a comma makes a tuple in "
             "Python, so NumPy sees an int here and rejects it")
add_raw("header_shape_bool", _hdr("{'descr': '|i1', 'fortran_order': False, 'shape': (True,), }", data=b"\x07"),
        load_error="shape is not valid",
        note="bool is a subclass of int in Python so NumPy's isinstance check passes, but building the "
             "array then fails; NumSharp rejects it up front — both refuse, the text differs")

# Zero-width dtypes: real NumPy dtypes with itemsize 0 that NumSharp has no analog for. They must be
# reported as unsupported, NOT as malformed descriptors.
for _name, _d, _msg in [
    ("dtype_U0_zero_width", "<U0", "only '<U1' maps to NumSharp's Char"),
    ("dtype_S0_zero_width", "|S0", "Byte-string dtypes"),
    ("dtype_V0_zero_width", "|V0", "Void dtypes"),
]:
    add_raw(_name, _hdr("{'descr': '%s', 'fortran_order': False, 'shape': (0,), }" % _d),
            load_via="load_npy", load_error=_msg,
            note=f"'{_d}' is a VALID zero-width NumPy dtype with no NumSharp analog — 'unsupported', not 'invalid'")

# ---------------------------------------------------------------------------------------------
# 9d. Header-only cases — writer logic no real array can reach
#
# These drive _write_array_header with a dict directly, because the shapes involved would need
# 10**17 elements to allocate. Two branches live here and nowhere else:
#
#   * The growth padding (21 - len(repr(growth_axis_dim)); first axis in C-order, LAST in F-order)
#     is normally INVISIBLE: shrink the body by 5 chars and the alignment padding grows by 5, so the
#     file is unchanged. It only shows when it tips the header across a 64-byte bucket. Using the
#     wrong axis would therefore pass every ordinary test — these are the shapes that catch it.
#   * The v1.0 -> v2.0 auto-selection boundary: NumSharp must switch at the same byte NumPy does.
# ---------------------------------------------------------------------------------------------
def add_header(name, d, note=""):
    buf = io.BytesIO()
    fmt._write_array_header(buf, d, None)  # None => auto-select the oldest version that fits
    raw = buf.getvalue()
    CASES.append({
        "name": name, "file": f"cases/{name}.hdr", "kind": "header", "bytes": raw,
        "descr": d["descr"], "np_dtype": None, "ns_dtype": None,
        "shape": [int(x) for x in d["shape"]], "fortran_order": d["fortran_order"],
        "version": [raw[6], raw[7]], "data_offset": None, "ns_bytes": None,
        "write_exact": True, "load_error": None, "note": note,
    })


def _hlen(shape, fortran):
    b = io.BytesIO()
    fmt._write_array_header(b, {"descr": "|i1", "fortran_order": fortran, "shape": shape}, None)
    return int.from_bytes(b.getvalue()[8:10], "little")


_observable = []
for _ndim in range(1, 12):
    for _df in range(1, 20):
        for _dl in range(1, 20):
            if _ndim == 1 and _df != _dl:
                continue
            _s = tuple(([10 ** (_df - 1)] + [1] * (_ndim - 2) + ([10 ** (_dl - 1)] if _ndim > 1 else []))[:_ndim])
            if _hlen(_s, False) != _hlen(_s, True):
                _observable.append(_s)

assert _observable, "expected shapes where the growth axis tips the header across a 64-byte bucket"
for _i, _s in enumerate(_observable[:12]):
    for _fo in (False, True):
        add_header(f"header_growth_axis_{_i}_{'F' if _fo else 'C'}",
                   {"descr": "|i1", "fortran_order": _fo, "shape": _s},
                   note=f"growth axis is OBSERVABLE here: shape {_s} in {'F' if _fo else 'C'}-order tips the "
                        f"header across a 64-byte bucket (C hlen={_hlen(_s, False)} vs F hlen={_hlen(_s, True)}), "
                        f"so using the wrong axis changes the file")

# The exact v1.0 -> v2.0 auto-selection boundary (found by bisection: 21817 dims is the last v1.0).
_lo, _hi = 1, 60000
while _lo + 1 < _hi:
    _mid = (_lo + _hi) // 2
    b = io.BytesIO()
    fmt._write_array_header(b, {"descr": "|u1", "fortran_order": False, "shape": (1,) * _mid}, None)
    if b.getvalue()[6] == 1:
        _lo = _mid
    else:
        _hi = _mid
for _nd, _tag in [(1, "tiny"), (_lo - 1, "just_under"), (_lo, "last_v1_0"), (_hi, "first_v2_0"), (_hi + 500, "well_into_v2_0")]:
    add_header(f"header_version_boundary_{_tag}",
               {"descr": "|u1", "fortran_order": False, "shape": (1,) * _nd},
               note=f"{_nd} dims: NumSharp's version auto-selection must flip to 2.0 at exactly the byte "
                    f"NumPy does (the last v1.0 header is {_lo} dims)")

# ---------------------------------------------------------------------------------------------
# 10. NPZ archives
# ---------------------------------------------------------------------------------------------
add_npz("npz_single", {"arr_0": np.arange(6, dtype=np.int32).reshape(2, 3)},
        note="single array, ZIP_STORED")
add_npz("npz_multi", {
    "arr_0": np.arange(6, dtype=np.int32).reshape(2, 3),
    "arr_1": _values("float64", 4),
    "weights": _values("float32", 8).reshape(2, 4),
    "flag": np.array(True),
}, note="mixed dtypes/shapes; .files strips '.npy'; both 'weights' and 'weights.npy' must resolve")
add_npz("npz_compressed", {
    "arr_0": np.arange(1000, dtype=np.int64),
    "text": _values("<U1", 5),
}, compressed=True, note="ZIP_DEFLATED (savez_compressed)")
add_npz("npz_fortran", {"f": np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3))},
        note="F-order array inside an npz")
add_npz("npz_empty_archive", {}, note="archive with no entries (PK\\x05\\x06 empty-zip magic)")
add_npz("npz_scalar_and_empty", {"s": np.array(42, dtype=np.int32), "e": np.array([], dtype=np.float64)},
        note="0-d and 0-element arrays inside an npz")
add_npz("npz_large", {"big": np.arange(100_000, dtype=np.float64)},
        note="800 KB entry: the ZipExtFile short-read path _read_bytes exists for")

# Duplicate member names. A zip may legally hold two entries with the same name, and the two
# runtimes disagree about which one a lookup finds: .NET's ZipArchive.GetEntry returns the FIRST,
# while Python's zipfile builds a name->info dict as it scans, so the LAST wins. NumPy's .files
# still lists BOTH. Verified against zipfile: z['dup'] -> [2].
_dup = io.BytesIO()
with zipfile.ZipFile(_dup, "w") as zf:
    for _v in (1, 2):
        _m = io.BytesIO()
        fmt.write_array(_m, np.array([_v], dtype=np.int8))
        zf.writestr("dup.npy", _m.getvalue())
_dup.seek(0)
with np.load(_dup) as _z:
    assert list(_z.files) == ["dup", "dup"] and _z["dup"].tolist() == [2], "zipfile: last duplicate wins"
CASES.append({
    "name": "npz_duplicate_names", "file": "cases/npz_duplicate_names.npz", "kind": "npz",
    "bytes": _dup.getvalue(), "compressed": False,
    "descr": None, "np_dtype": None, "ns_dtype": None, "shape": None,
    "fortran_order": None, "version": None, "data_offset": None, "ns_bytes": None,
    "write_exact": False, "load_error": None,
    "entries": {"dup": {"ns_dtype": "SByte", "shape": [1], "ns_bytes": np.array([2], dtype=np.int8).tobytes().hex()}},
    "files": ["dup", "dup"],
    "note": "duplicate zip member names: .files lists both, but a lookup must resolve to the LAST — "
            "Python's zipfile keeps the last in its name map, while .NET's GetEntry returns the first",
})

# An npz holding a non-.npy member: NumPy returns its raw bytes.
_mixed = io.BytesIO()
with zipfile.ZipFile(_mixed, "w", zipfile.ZIP_DEFLATED) as zf:
    b = io.BytesIO()
    fmt.write_array(b, np.arange(3, dtype=np.int32))
    zf.writestr("data.npy", b.getvalue())
    zf.writestr("readme.txt", b"not a numpy file")
CASES.append({
    "name": "npz_non_npy_member", "file": "cases/npz_non_npy_member.npz", "kind": "npz",
    "bytes": _mixed.getvalue(), "compressed": True,
    "descr": None, "np_dtype": None, "ns_dtype": None, "shape": None,
    "fortran_order": None, "version": None, "data_offset": None, "ns_bytes": None,
    "write_exact": False, "load_error": None,
    "entries": {"data": {"ns_dtype": "Int32", "shape": [3], "ns_bytes": np.arange(3, dtype=np.int32).tobytes().hex()}},
    "raw_entries": {"readme.txt": b"not a numpy file".hex()},
    "note": "npz members that are not .npy are returned as raw bytes by NumPy",
})

# ---------------------------------------------------------------------------------------------
# 11. Multiple arrays appended to ONE .npy stream (np.save(f, a); np.save(f, b))
# ---------------------------------------------------------------------------------------------
_multi = io.BytesIO()
_seq = [np.arange(3, dtype=np.int32), _values("float64", 4).reshape(2, 2), np.array(9, dtype=np.int8)]
for a in _seq:
    fmt.write_array(_multi, a)
CASES.append({
    "name": "stream_multi_array", "file": "cases/stream_multi_array.npy", "kind": "sequence",
    "bytes": _multi.getvalue(),
    "descr": None, "np_dtype": None, "ns_dtype": None, "shape": None,
    "fortran_order": None, "version": None, "data_offset": None, "ns_bytes": None,
    "write_exact": True, "load_error": None,
    "sequence": [
        {"ns_dtype": _ns_dtype_of(a), "shape": [int(d) for d in a.shape], "ns_bytes": _ns_bytes(a)}
        for a in _seq
    ],
    "note": "three arrays appended to one stream; each np.load reads exactly one and leaves the "
            "position at the next. A 4th load must raise EOFError.",
})


# ---------------------------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    manifest = []
    # ZIP_DEFLATED + a fixed date_time keeps the committed zip byte-stable across regenerations.
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for case in CASES:
            case.setdefault("load_via", "load")
            case.setdefault("max_header_size", None)
            raw = case.pop("bytes")
            info = zipfile.ZipInfo(case["file"], date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, raw)
            case["size"] = len(raw)
            manifest.append(case)

        info = zipfile.ZipInfo("manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, json.dumps({
            "numpy_version": np.__version__,
            "generator": "test/oracle/gen_npy_oracle.py",
            "cases": manifest,
        }, indent=1, sort_keys=True))

    kinds = {}
    for c in manifest:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"wrote {os.path.relpath(OUT)}  ({os.path.getsize(OUT):,} bytes)")
    print(f"  cases        : {len(manifest)}")
    for k, v in sorted(kinds.items()):
        print(f"    {k:10s} : {v}")
    print(f"  write_exact  : {sum(1 for c in manifest if c['write_exact'])}")
    print(f"  load_error   : {sum(1 for c in manifest if c['load_error'])}")
    print(f"  numpy        : {np.__version__}")


if __name__ == "__main__":
    sys.exit(main())
