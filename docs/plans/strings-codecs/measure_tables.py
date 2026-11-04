"""Measures the data a port of CPython's refused codecs would carry (``docs/plans/strings-codecs.md``, sections 3.5 and
3.7): the CJK charset tables, read from the running interpreter, and the Unicode 3.2.0 data IDNA's nameprep needs.

CJK. Each ``_codecs_<lang>`` module exposes its charset tables as ``__map_<charset>`` attributes: a PyCapsule named
``multibytecodec.map`` over ``struct dbcs_map {const char *charset; const struct unim_index *encmap; const struct
dbcs_index *decmap;}`` (Modules/cjkcodecs/cjkcodecs.h). Each index is 256 rows of ``{const T *map; unsigned char bottom,
top;}``: a decode row per lead byte (``ucs2_t``, ``Py_UCS4`` for the JIS X 0213 pairs), an encode row per high byte of a
BMP code point (``DBCHAR``). ctypes reads them exactly as CPython's coders do, so a generator can dump them from the
oracle's interpreter without CPython's source, as tools/gen_python_codecs.py dumps the charmap tables. The sizes are
measured three ways: the plain values, the values as deltas from the previous value of their row, and those deltas split
into a low-byte and a high-byte plane (both zlib level 9, the compression PyUnicodeNames.Data.cs already uses).

IDNA. ``encodings.idna`` runs RFC 3491's nameprep over ``unicodedata.ucd_3_2_0``: the 3.2.0 category (Zs / Cc / Co / Cs /
Cn) and bidirectional class (R / AL / L) of every code point, and NFKC with the CURRENT decomposition data gated to the
code points 3.2.0 assigned, the current combining classes and composition pairs, and the 3.2.0 normalization
corrections (Modules/unicodedata.c: get_decomp_record, nfd_nfkd, nfc_nfkc).

Run with the CPython the oracles run (3.12; the CJK part needs CPython's ctypes ABI):

    python -I docs/plans/strings-codecs/measure_tables.py
"""
import base64
import ctypes
import struct
import sys
import unicodedata
import zlib
from unicodedata import ucd_3_2_0

import _codecs_cn
import _codecs_hk
import _codecs_jp
import _codecs_kr
import _codecs_tw

_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_GetPointer.restype = ctypes.c_void_p
_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


class _DbcsMap(ctypes.Structure):
    """``struct dbcs_map``: a charset's name and its encode / decode indexes (either may be NULL)."""
    _fields_ = [("charset", ctypes.c_char_p), ("encmap", ctypes.c_void_p), ("decmap", ctypes.c_void_p)]


class _Row(ctypes.Structure):
    """One row of ``struct dbcs_index`` / ``unim_index`` / ``widedbcs_index``: the values of the second byte (or low
    byte) ``bottom .. top``, or a NULL map for an absent row."""
    _fields_ = [("map", ctypes.c_void_p), ("bottom", ctypes.c_ubyte), ("top", ctypes.c_ubyte)]


def read_index(address, element):
    """Reads a 256-row index and the values of every present row.

    :param address: the index's address (from a ``dbcs_map``).
    :param element: the ctypes type of a value (``c_uint16`` for ``ucs2_t`` / ``DBCHAR``, ``c_uint32`` for ``Py_UCS4``).
    :returns: 256 entries, each ``(bottom, top, [values])`` or None for an absent row.
    """
    rows = (_Row * 256).from_address(address)
    return [None if not r.map else (r.bottom, r.top, list((element * (r.top - r.bottom + 1)).from_address(r.map)))
            for r in rows]


def charset_tables(module):
    """Every charset index a ``_codecs_<lang>`` module exposes.

    :param module: the imported module.
    :returns: ``[(charset, 'dec' | 'enc', width in bytes, rows)]``; the JIS X 0213 pair ENCODE map is a flat
        ``pair_encodemap[46]`` array, not an index, and is left out (46 entries).
    """
    out = []
    for attr in sorted(a for a in dir(module) if a.startswith("__map_")):
        m = _DbcsMap.from_address(_GetPointer(getattr(module, attr), b"multibytecodec.map"))
        cs = m.charset.decode()
        for kind, address in (("dec", m.decmap), ("enc", m.encmap)):
            if not address or (cs == "jisx0213_pair" and kind == "enc"):
                continue
            width = 4 if (cs == "jisx0213_pair" and kind == "dec") else 2
            out.append((cs, kind, width, read_index(address, ctypes.c_uint32 if width == 4 else ctypes.c_uint16)))
    return out


def serialize(rows, width, delta):
    """One index as bytes: per row its bottom / top (two zero bytes for an absent row), then its values.

    :param rows: the index (``read_index``'s result).
    :param width: the bytes per value.
    :param delta: write each value as its difference from the previous value of its row (modulo the width).
    :returns: the bytes.
    """
    out = bytearray()
    for r in rows:
        if r is None:
            out += b"\x00\x00"
            continue
        out += bytes([r[0], r[1]])
        prev = 0
        for v in r[2]:
            out += (((v - prev) if delta else v) & ((1 << (8 * width)) - 1)).to_bytes(width, "little")
            prev = v
    return bytes(out)


def measure_cjk():
    """Prints every charset table and the per-language and total sizes, plain and delta + byte planes."""
    total_plain = total_planes = 0
    for module in (_codecs_cn, _codecs_hk, _codecs_jp, _codecs_kr, _codecs_tw):
        plain = bytearray()
        delta = bytearray()
        for cs, kind, width, rows in charset_tables(module):
            n = sum(len(r[2]) for r in rows if r)
            print(f"  {module.__name__:11s} {cs:18s} {kind}  rows {sum(1 for r in rows if r):3d}  values {n:6d}")
            plain += serialize(rows, width, delta=False)
            delta += serialize(rows, width, delta=True)
        planes = zlib.compress(bytes(delta[0::2]) + bytes(delta[1::2]), 9)
        total_plain += len(plain)
        total_planes += len(planes)
        print(f"{module.__name__}: raw {len(plain)} B, zlib {len(zlib.compress(bytes(plain), 9))} B, "
              f"delta + planes zlib {len(planes)} B (base64 {len(base64.b64encode(planes))} B)")
    print(f"CJK total: raw {total_plain} B, delta + planes zlib {total_planes} B (base64 ~{total_planes * 4 // 3} B)")


def measure_idna():
    """Prints the sizes of the Unicode 3.2.0 data nameprep reads and checks the decomposition gating rule."""
    def klass(cp):
        ch = chr(cp)
        cat = {"Zs": 1, "Cc": 2, "Co": 3, "Cs": 4, "Cn": 5}.get(ucd_3_2_0.category(ch), 0)
        bidi = ucd_3_2_0.bidirectional(ch)
        return cat * 3 + (1 if bidi in ("R", "AL") else 2 if bidi == "L" else 0)

    ranges, prev = [], None
    for cp in range(0x110000):
        k = klass(cp)
        if k != prev:
            ranges.append((cp, k))
            prev = k
    raw = b"".join(struct.pack("<IB", s, k) for s, k in ranges)
    print(f"IDNA 3.2.0 category x bidi classes: {len(ranges)} ranges, {len(raw)} B, zlib {len(zlib.compress(raw, 9))} B")
    decomp = {}
    mismatched = 0
    for cp in range(0x110000):
        ch = chr(cp)
        gated = unicodedata.decomposition(ch) if ucd_3_2_0.category(ch) != "Cn" else ""
        if gated != ucd_3_2_0.decomposition(ch):
            mismatched += 1   # get_decomp_record's rule: the current record, empty where 3.2.0 had no character
        if gated:
            parts = gated.split()
            compat = parts[0].startswith("<")
            decomp[cp] = (compat, [int(p, 16) for p in (parts[1:] if compat else parts)])
    raw = b"".join(struct.pack("<IBB", cp, int(c), len(v)) + b"".join(struct.pack("<I", x) for x in v)
                   for cp, (c, v) in sorted(decomp.items()))
    print(f"gated decompositions: {len(decomp)}, {len(raw)} B, zlib {len(zlib.compress(raw, 9))} B "
          f"(ucd_3_2_0.decomposition disagrees with the gating rule at {mismatched} code points)")
    combining = sum(1 for cp in range(0x110000) if unicodedata.combining(chr(cp)))
    pairs = 0
    for cp in range(0x110000):
        d = unicodedata.decomposition(chr(cp))
        if d and not d.startswith("<") and len(d.split()) == 2:
            a, b = (chr(int(p, 16)) for p in d.split())
            pairs += unicodedata.normalize("NFC", a + b) == chr(cp)
    corrections = [hex(cp) for cp in range(0x110000) if ucd_3_2_0.category(chr(cp)) != "Cn"
                   and ucd_3_2_0.normalize("NFD", chr(cp)) != unicodedata.normalize("NFD", chr(cp))]
    print(f"current non-zero combining classes: {combining}; composition pairs: {pairs}; "
          f"3.2.0 normalization corrections: {corrections}")


def main():
    """Prints both measurements."""
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"CPython {sys.version.split()[0]}, Unicode {unicodedata.unidata_version}")
    measure_cjk()
    measure_idna()


if __name__ == "__main__":
    main()
