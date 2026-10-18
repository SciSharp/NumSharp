"""Generates src/NumSharp.Core/Backends/Strings/PyUnicodeNames.Data.cs from the CPython that runs it.

np.strings.encode(a, encoding, 'namereplace') calls CPython's namereplace error handler (PyCodec_NameReplaceErrors),
which writes an unencodable character as \\N{NAME} — the name unicodedata's C API returns for it, WITH aliases and named
sequences (getname(code, buffer, size, with_alias_and_seq=1)) — and as \\xNN / \\uNNNN / \\UNNNNNNNN when it has none.
NumSharp reproduces those names from a table pinned to the CPython the NumPy 2.4.2 oracle runs (3.12, Unicode 15.0.0);
.NET has no character-name API at all.

What the script records, read from the RUNNING interpreter's behavior (the handler itself, called for every code point,
not the UCD files), so the table is what CPython does:

* every code point 0 .. 0x10FFFF that codecs.namereplace_errors names (143,975 at Unicode 15.0.0), split into
  - the two ranges CPython's _getucname computes rather than stores: the Hangul syllables U+AC00 .. U+D7A3 ("HANGUL
    SYLLABLE " + the romanized jamo of Unicode's syllable algorithm) and the CJK unified ideographs of
    is_unified_ideograph ("CJK UNIFIED IDEOGRAPH-%X" over nine ranges) — the script asserts both rules over every
    code point, so NumSharp computes them the same way;
  - every other name (35,757), stored. They include the names of the Private Use Area plane-15 code points U+F0000 ..
    where CPython keeps its name aliases and named sequences ("NULL", "KEYCAP NUMBER SIGN"): with_alias_and_seq=1 makes
    getname return those for the PUA characters themselves, so namereplace of U+F0000 is \\N{NULL};
* the stored names as one blob: per name in code-point order the code point's distance from the previous one (an
  unsigned LEB128 varint, the first from -1), the name's length (one byte) and its ASCII bytes; the blob deflated
  (zlib, level 9 — .NET's ZLibStream reads it) and written as base64 lines of a UTF-8 raw string literal, which NumSharp
  decodes and inflates once, at the first namereplace that needs a name.

The generated file also carries a SHA-256 over every code point's answer (the name's bytes then '\\n', or '\\n' alone),
which NumSharp's pin test recomputes from its runtime lookups.

Run with the CPython the oracles run (3.12, the NumPy 2.4.2 oracle interpreter):  python tools/gen_unicode_names.py
"""
import base64
import codecs
import hashlib
import os
import sys
import unicodedata
import zlib

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "NumSharp.Core", "Backends", "Strings",
                   "PyUnicodeNames.Data.cs")

MAX = 0x110000

# CPython 3.12's unicodedata.c: hangul_syllables (the initial, medial and final jamo romanizations; None where the column
# has fewer entries) and the syllable constants.
_HANGUL = [
    ("G", "A", ""), ("GG", "AE", "G"), ("N", "YA", "GG"), ("D", "YAE", "GS"), ("DD", "EO", "N"), ("R", "E", "NJ"),
    ("M", "YEO", "NH"), ("B", "YE", "D"), ("BB", "O", "L"), ("S", "WA", "LG"), ("SS", "WAE", "LM"), ("", "OE", "LB"),
    ("J", "YO", "LS"), ("JJ", "U", "LT"), ("C", "WEO", "LP"), ("K", "WE", "LH"), ("T", "WI", "M"), ("P", "YU", "B"),
    ("H", "EU", "BS"), (None, "YI", "S"), (None, "I", "SS"), (None, None, "NG"), (None, None, "J"), (None, None, "C"),
    (None, None, "K"), (None, None, "T"), (None, None, "P"), (None, None, "H"),
]
_SBASE, _LCOUNT, _VCOUNT, _TCOUNT = 0xAC00, 19, 21, 28
_NCOUNT = _VCOUNT * _TCOUNT
_SCOUNT = _LCOUNT * _NCOUNT

# CPython 3.12's is_unified_ideograph (unicodedata.c, matching makeunicodedata.py's cjk_ranges).
_CJK = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0x20000, 0x2A6DF), (0x2A700, 0x2B739), (0x2B740, 0x2B81D),
        (0x2B820, 0x2CEA1), (0x2CEB0, 0x2EBE0), (0x30000, 0x3134A), (0x31350, 0x323AF)]


def _hangul_name(cp):
    """The Hangul syllable name CPython's _getucname computes.

    :param cp: a code point in U+AC00 .. U+D7A3.
    :returns: the name.
    """
    s = cp - _SBASE
    l, v, t = s // _NCOUNT, (s % _NCOUNT) // _TCOUNT, s % _TCOUNT
    return "HANGUL SYLLABLE " + _HANGUL[l][0] + _HANGUL[v][1] + _HANGUL[t][2]


def _is_unified_ideograph(cp):
    """CPython's is_unified_ideograph.

    :param cp: the code point.
    :returns: True inside one of the nine ranges.
    """
    return any(a <= cp <= b for a, b in _CJK)


def _read():
    """The name namereplace writes for every code point, or None.

    :returns: a list indexed by code point.
    """
    names = [None] * MAX
    for cp in range(MAX):
        rep, _ = codecs.namereplace_errors(UnicodeEncodeError("ascii", chr(cp), 0, 1, "x"))
        if rep.startswith("\\N{"):
            assert rep.endswith("}"), (hex(cp), rep)
            names[cp] = rep[3:-1]
    return names


def _canonical_hash(names):
    """The SHA-256 the runtime recomputes: per code point the name's bytes then '\\n', or '\\n' alone.

    :param names: the names by code point.
    :returns: the hex digest.
    """
    h = hashlib.sha256()
    for name in names:
        if name is not None:
            h.update(name.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _blob(stored):
    """The stored names as the blob described in the module docstring.

    :param stored: (code point, name) pairs in code-point order.
    :returns: the blob.
    """
    out = bytearray()
    prev = -1
    for cp, name in stored:
        d = cp - prev
        prev = cp
        while d >= 0x80:
            out.append((d & 0x7F) | 0x80)
            d >>= 7
        out.append(d)
        b = name.encode("ascii")
        assert 0 < len(b) < 256, (hex(cp), name)
        out.append(len(b))
        out += b
    return bytes(out)


def main():
    assert unicodedata.unidata_version == "15.0.0", unicodedata.unidata_version
    names = _read()
    stored = []
    hangul = cjk = 0
    longest = 0
    for cp, name in enumerate(names):
        if name is None:
            # The algorithmic ranges name every code point in them.
            assert not (_SBASE <= cp < _SBASE + _SCOUNT) and not _is_unified_ideograph(cp), hex(cp)
            continue
        longest = max(longest, len(name))
        if _SBASE <= cp < _SBASE + _SCOUNT:
            assert name == _hangul_name(cp), (hex(cp), name)
            hangul += 1
        elif _is_unified_ideograph(cp):
            assert name == "CJK UNIFIED IDEOGRAPH-%X" % cp, (hex(cp), name)
            cjk += 1
        else:
            assert not name.startswith("CJK UNIFIED IDEOGRAPH-"), (hex(cp), name)
            assert all(c.isupper() or c.isdigit() or c in " -" for c in name), (hex(cp), name)
            stored.append((cp, name))
    digest = _canonical_hash(names)
    blob = _blob(stored)
    packed = zlib.compress(blob, 9)
    b64 = base64.b64encode(packed).decode("ascii")
    lines = [b64[i:i + 120] for i in range(0, len(b64), 120)]

    version = "%d.%d.%d" % sys.version_info[:3]
    out = []
    out.append("// <auto-generated>")
    out.append("// CPython %s's Unicode character names (Unicode %s) as its namereplace error handler writes them." %
               (version, unicodedata.unidata_version))
    out.append("// Generated by tools/gen_unicode_names.py from the CPython the oracles run: do not edit by hand, re-run the script.")
    out.append("//   CompressedNames  every stored name in code-point order (varint distance from the previous code point, one")
    out.append("//                    length byte, the ASCII name), zlib-deflated, base64 in lines of 120 characters.")
    out.append("//   The Hangul syllables and the CJK unified ideographs are computed, as CPython's _getucname computes them.")
    out.append("// </auto-generated>")
    out.append("using System;")
    out.append("")
    out.append("namespace NumSharp.Backends.Strings")
    out.append("{")
    out.append("    internal static partial class PyUnicodeNames")
    out.append("    {")
    out.append("        /// <summary>The Unicode version of the names: <c>unicodedata.unidata_version</c> of the generating CPython.</summary>")
    out.append("        internal const string UnicodeVersion = \"%s\";" % unicodedata.unidata_version)
    out.append("")
    out.append("        /// <summary>The CPython the names were read from.</summary>")
    out.append("        internal const string CPythonVersion = \"%s\";" % version)
    out.append("")
    out.append("        /// <summary>The SHA-256 of every code point's answer (see <see cref=\"ComputeNamesHash\"/>), recorded by the generator.</summary>")
    out.append("        internal const string NamesHash = \"%s\";" % digest)
    out.append("")
    out.append("        /// <summary>The number of stored names (the blob's records).</summary>")
    out.append("        private const int StoredCount = %d;" % len(stored))
    out.append("")
    out.append("        /// <summary>The size of the inflated blob in bytes.</summary>")
    out.append("        private const int BlobSize = %d;" % len(blob))
    out.append("")
    out.append("        /// <summary>The longest name (the replacement buffer CPython sizes at 256 bytes never truncates one).</summary>")
    out.append("        internal const int LongestName = %d;" % longest)
    out.append("")
    out.append("        /// <summary>The stored names: zlib-deflated, base64 (the file header describes the records).</summary>")
    out.append("        private static ReadOnlySpan<byte> CompressedNames => \"\"\"")
    for line in lines:
        out.append("            " + line)
    out.append("            \"\"\"u8;")
    out.append("    }")
    out.append("}")
    text = "\n".join(out) + "\n"
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("named=%d hangul=%d cjk=%d stored=%d longest=%d blob=%d deflated=%d base64=%d hash=%s" %
          (sum(1 for n in names if n is not None), hangul, cjk, len(stored), longest, len(blob), len(packed), len(b64),
           digest))


if __name__ == "__main__":
    main()
