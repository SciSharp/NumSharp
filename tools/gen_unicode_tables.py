"""Generates src/NumSharp.Core/Backends/Strings/PyUnicodeData.Tables.cs from the CPython that runs it.

NumPy's string ufuncs (np.strings.isalpha ... isnumeric) read CPython's Unicode type records (Py_UNICODE_ISALPHA and its
siblings) per code point, and its case maps, translate, mod and codecs call CPython's str methods, which read the same
records (_PyUnicode_ToUpperFull, _PyUnicode_IsCased, ...). NumSharp reproduces them from tables pinned to the CPython the
NumPy 2.4.2 oracle runs (3.12, Unicode 15.0.0), never from .NET's tables, which follow the running .NET's Unicode version
(.NET 10 reads Unicode 16.0).

What the script records, per code point 0 .. 0x10FFFF, read from the RUNNING interpreter's behavior (not from the UCD
files), so the tables are what CPython does:

* the flags of _PyUnicode_TypeRecord that NumSharp reads: ALPHA (str.isalpha), DECIMAL (str.isdecimal), DIGIT
  (str.isdigit), NUMERIC (str.isnumeric), SPACE (str.isspace), LOWER (str.islower of one character is Py_UNICODE_ISLOWER),
  UPPER (likewise), TITLE (general category Lt, makeunicodedata.py's rule: str.istitle of one character also counts an
  uppercase one), CASED (_PyUnicode_IsCased: read from str.title, whose state is "the previous character is cased", and
  checked against Unicode's definition Lowercase | Uppercase | Lt), CASE_IGNORABLE (_PyUnicode_IsCaseIgnorable: read from
  str.lower's final-sigma rule), PRINTABLE (str.isprintable), XID_START (_PyUnicode_IsXidStart: read as c.isidentifier(),
  which is XID_Start or the LOW LINE U+005F, the one character CPython's _PyUnicode_ScanIdentifier admits besides; the LOW
  LINE itself is excluded, it is not XID_Start) and XID_CONTINUE (_PyUnicode_IsXidContinue: read as ("a" + c).isidentifier()
  — str.isidentifier tests the first character for XID_Start and every other for XID_Continue, without NFKC);
* the decimal value (unicodedata.decimal: Py_UNICODE_TODECIMAL);
* the full case mappings: str.upper / str.lower / str.title of the one character, which are _PyUnicode_ToUpperFull /
  _PyUnicode_ToLowerFull / _PyUnicode_ToTitleFull (one to three code points; str.lower of a lone U+03A3 is sigma, the
  final-sigma rule is applied by the code that reads the table, as CPython applies it in lower_ucs4).

Layout (CPython's makeunicodedata.py layout): a record table (flags, decimal, and per mapping either the delta to the
mapped code point or an index into the extended mappings, flagged EXTENDED_CASE) and a two-stage index, split by
splitbins at the shift that minimizes the total size. The generated file also carries a SHA-256 of the per-code-point
data, which NumSharp's pin test recomputes from the runtime lookups.

Run with the CPython the oracles run (3.12, the NumPy 2.4.2 oracle interpreter):  python tools/gen_unicode_tables.py
"""
import hashlib
import os
import struct
import sys
import unicodedata

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "NumSharp.Core", "Backends", "Strings",
                   "PyUnicodeData.Tables.cs")

MAX = 0x110000

# The flag bits (PyUnicodeData's constants mirror these).
ALPHA = 1 << 0
DECIMAL = 1 << 1
DIGIT = 1 << 2
NUMERIC = 1 << 3
SPACE = 1 << 4
LOWER = 1 << 5
UPPER = 1 << 6
TITLE = 1 << 7
CASED = 1 << 8
CASE_IGNORABLE = 1 << 9
PRINTABLE = 1 << 10
EXTENDED_CASE = 1 << 11
XID_START = 1 << 12
XID_CONTINUE = 1 << 13

SIGMA = "Σ"
FINAL_SIGMA = "ς"


def _case_ignorable(c, cased):
    """_PyUnicode_IsCaseIgnorable of one character, read from str.lower's final-sigma rule (handle_capital_sigma).

    'X' + c + SIGMA lowers the sigma to the final form when, scanning back from the sigma over case-ignorable characters,
    the first other character is cased. With a cased prefix 'A' the sigma is final iff c is case-ignorable (skipped, then
    'A') or cased (it stops the scan and is cased); with an uncased, not case-ignorable prefix '1' it is final iff c is
    cased and not case-ignorable. So: an uncased c is case-ignorable iff 'A' + c + SIGMA ends in the final sigma; a cased
    c iff '1' + c + SIGMA does not.

    :param c: the character.
    :param cased: whether c is cased.
    :returns: True for a case-ignorable character.
    """
    if cased:
        return ("1" + c + SIGMA).lower()[-1] != FINAL_SIGMA
    return ("A" + c + SIGMA).lower()[-1] == FINAL_SIGMA


def _read():
    """Every code point's flags, decimal value and full case mappings, as CPython computes them.

    :returns: (flags, decimals, uppers, lowers, titles) lists indexed by code point.
    :raises AssertionError: CPython's behavior departs from an assumption the tables rely on.
    """
    flags = [0] * MAX
    decimals = [-1] * MAX
    uppers, lowers, titles = [None] * MAX, [None] * MAX, [None] * MAX
    for cp in range(MAX):
        c = chr(cp)
        f = 0
        if c.isalpha():
            f |= ALPHA
        if c.isdecimal():
            f |= DECIMAL
        if c.isdigit():
            f |= DIGIT
        if c.isnumeric():
            f |= NUMERIC
        if c.isspace():
            f |= SPACE
        if c.islower():
            f |= LOWER
        if c.isupper():
            f |= UPPER
        if unicodedata.category(c) == "Lt":
            f |= TITLE
        # _PyUnicode_IsCased, as str.title reads it: 'a' after c stays lower exactly when c is cased.
        cased = (c + "a").title()[-1] == "a"
        assert cased == bool(f & (LOWER | UPPER | TITLE)), hex(cp)
        if cased:
            f |= CASED
        if _case_ignorable(c, cased):
            f |= CASE_IGNORABLE
        if c.isprintable():
            f |= PRINTABLE
        # _PyUnicode_IsXidStart / _PyUnicode_IsXidContinue, as str.isidentifier reads them (CPython's
        # _PyUnicode_ScanIdentifier: XID_Start or U+005F first, XID_Continue after, no NFKC): a one-character identifier
        # starts with XID_Start unless it is the LOW LINE, which isidentifier admits on its own; any character after "a"
        # continues an identifier exactly when it is XID_Continue.
        if c.isidentifier() and cp != 0x5F:
            f |= XID_START
        if ("a" + c).isidentifier():
            f |= XID_CONTINUE
        d = unicodedata.decimal(c, -1)
        assert (d >= 0) == bool(f & DECIMAL), hex(cp)
        decimals[cp] = d
        uppers[cp] = tuple(ord(x) for x in c.upper())
        lowers[cp] = tuple(ord(x) for x in c.lower())
        titles[cp] = tuple(ord(x) for x in c.title())
        assert 1 <= len(uppers[cp]) <= 3 and 1 <= len(lowers[cp]) <= 3 and 1 <= len(titles[cp]) <= 3, hex(cp)
        flags[cp] = f
    return flags, decimals, uppers, lowers, titles


def _getsize(data):
    """The C# element size that holds every value (1, 2 or 4 bytes)."""
    maxdata = max(data)
    if maxdata < 256:
        return 1
    if maxdata < 65536:
        return 2
    return 4


def _splitbins(t):
    """makeunicodedata.py's splitbins: t[i] == t2[(t1[i >> shift] << shift) + (i & mask)] at the smallest total size.

    :param t: the per-code-point record indices.
    :returns: (t1, t2, shift).
    """
    maxshift = 0
    n = len(t) - 1
    while n >> 1:
        n >>= 1
        maxshift += 1
    best = None
    best_bytes = sys.maxsize
    t = tuple(t)
    for shift in range(maxshift + 1):
        t1, t2 = [], []
        size = 2 ** shift
        bincache = {}
        for i in range(0, len(t), size):
            b = t[i:i + size]
            index = bincache.get(b)
            if index is None:
                index = len(t2)
                bincache[b] = index
                t2.extend(b)
            t1.append(index >> shift)
        nbytes = len(t1) * _getsize(t1) + len(t2) * _getsize(t2)
        if nbytes < best_bytes:
            best = t1, t2, shift
            best_bytes = nbytes
    t1, t2, shift = best
    mask = (1 << shift) - 1
    for i in range(len(t)):
        assert t[i] == t2[(t1[i >> shift] << shift) + (i & mask)]
    return best


def _canonical_hash(flags, decimals, uppers, lowers, titles):
    """The SHA-256 PyUnicodeData's pin test recomputes: per code point the flags (uint16 LE, EXTENDED_CASE cleared: a
    layout bit, not a property), the decimal value (int8) and each full mapping as its length then its code points
    (int32 LE).

    :returns: the hex digest.
    """
    h = hashlib.sha256()
    for cp in range(MAX):
        h.update(struct.pack("<Hb", flags[cp] & ~EXTENDED_CASE, decimals[cp]))
        for m in (uppers[cp], lowers[cp], titles[cp]):
            h.update(struct.pack("<B", len(m)))
            h.update(struct.pack("<%di" % len(m), *m))
    return h.hexdigest()


def _records(flags, decimals, uppers, lowers, titles):
    """The record table and each code point's record index.

    A simple mapping (one code point) is stored as its delta from the code point, so runs of letters share a record; a
    multi-code-point mapping sets EXTENDED_CASE and stores (offset | length << 24) into the extended array (all three
    mappings of the record are then extended entries).

    :returns: (records, index, extended) — records as tuples (flags, decimal, upper, lower, title).
    """
    extended = []
    ext_cache = {}

    def ext(seq):
        key = tuple(seq)
        if key not in ext_cache:
            ext_cache[key] = len(extended)
            extended.extend(seq)
        return ext_cache[key] | (len(seq) << 24)

    records = []
    record_cache = {}
    index = [0] * MAX
    for cp in range(MAX):
        f = flags[cp]
        up, lo, ti = uppers[cp], lowers[cp], titles[cp]
        if len(up) == 1 and len(lo) == 1 and len(ti) == 1:
            rec = (f, decimals[cp], up[0] - cp, lo[0] - cp, ti[0] - cp)
        else:
            rec = (f | EXTENDED_CASE, decimals[cp], ext(up), ext(lo), ext(ti))
        i = record_cache.get(rec)
        if i is None:
            i = len(records)
            record_cache[rec] = i
            records.append(rec)
        index[cp] = i
    return records, index, extended


def _cs_array(name, ctype, values, doc, per_line=24):
    """One generated array as a ReadOnlySpan property over a constant array (no allocation, data in the image).

    :returns: the C# text.
    """
    lines = ["        /// <summary>%s</summary>" % doc,
             "        private static ReadOnlySpan<%s> %s => new %s[]" % (ctype, name, ctype),
             "        {"]
    for i in range(0, len(values), per_line):
        chunk = values[i:i + per_line]
        lines.append("            " + ", ".join(str(v) for v in chunk) + ",")
    lines.append("        };")
    return "\n".join(lines)


def _ctype(size, signed=False):
    if signed:
        return {1: "sbyte", 2: "short", 4: "int"}[size]
    return {1: "byte", 2: "ushort", 4: "uint"}[size]


def main():
    assert unicodedata.unidata_version == "15.0.0", unicodedata.unidata_version
    flags, decimals, uppers, lowers, titles = _read()
    digest = _canonical_hash(flags, decimals, uppers, lowers, titles)
    records, index, extended = _records(flags, decimals, uppers, lowers, titles)
    # Record 0 is the record of a code point with no property that maps to itself (U+0000's): the lookup's answer for a
    # unit past U+10FFFF, CPython's gettyperecord.
    assert records[0] == (0, -1, 0, 0, 0), records[0]
    t1, t2, shift = _splitbins(index)
    i1size, i2size = _getsize(t1), _getsize(t2)

    rec_flags = [r[0] for r in records]
    rec_dec = [r[1] for r in records]
    rec_up = [r[2] for r in records]
    rec_lo = [r[3] for r in records]
    rec_ti = [r[4] for r in records]

    version = "%d.%d.%d" % sys.version_info[:3]
    out = []
    out.append("// <auto-generated>")
    out.append("// CPython %s's Unicode type records (Unicode %s), as NumPy's string ufuncs and CPython's str methods read them." %
               (version, unicodedata.unidata_version))
    out.append("// Generated by tools/gen_unicode_tables.py from the CPython the oracles run: do not edit by hand, re-run the script.")
    out.append("//   Records       per distinct (flags, decimal, upper, lower, title): a simple mapping as the delta to the mapped")
    out.append("//                 code point, a multi-code-point one (EXTENDED_CASE) as offset | length << 24 into ExtendedCase.")
    out.append("//   Index1/Index2 the two-stage index (makeunicodedata.py's splitbins): record = Index2[(Index1[cp >> Shift] << Shift)")
    out.append("//                 + (cp & ((1 << Shift) - 1))].")
    out.append("// </auto-generated>")
    out.append("using System;")
    out.append("")
    out.append("namespace NumSharp.Backends.Strings")
    out.append("{")
    out.append("    internal static partial class PyUnicodeData")
    out.append("    {")
    out.append("        /// <summary>The Unicode version of the tables: <c>unicodedata.unidata_version</c> of the generating CPython.</summary>")
    out.append("        internal const string UnicodeVersion = \"%s\";" % unicodedata.unidata_version)
    out.append("")
    out.append("        /// <summary>The CPython the tables were read from.</summary>")
    out.append("        internal const string CPythonVersion = \"%s\";" % version)
    out.append("")
    out.append("        /// <summary>The SHA-256 of the per-code-point data (see <see cref=\"ComputeTableHash\"/>), recorded by the generator.</summary>")
    out.append("        internal const string TableHash = \"%s\";" % digest)
    out.append("")
    out.append("        /// <summary>The index split: the low <c>Shift</c> bits of a code point select within its block.</summary>")
    out.append("        private const int Shift = %d;" % shift)
    out.append("")
    out.append("        /// <summary>The number of records.</summary>")
    out.append("        private const int RecordCount = %d;" % len(records))
    out.append("")
    out.append(_cs_array("Index1", _ctype(i1size), t1, "The first index stage: a code point's block (<c>cp &gt;&gt; Shift</c>) to its block in Index2."))
    out.append("")
    out.append(_cs_array("Index2", _ctype(i2size), t2, "The second index stage: the record of every code point of every distinct block."))
    out.append("")
    out.append(_cs_array("RecordFlags", "ushort", rec_flags, "Each record's flags (<see cref=\"AlphaMask\"/> ... <see cref=\"XidContinueMask\"/>)."))
    out.append("")
    out.append(_cs_array("RecordDecimal", "sbyte", rec_dec, "Each record's decimal value (0..9), or -1."))
    out.append("")
    out.append(_cs_array("RecordUpper", "int", rec_up, "Each record's uppercase mapping: a delta, or offset | length &lt;&lt; 24 when extended."))
    out.append("")
    out.append(_cs_array("RecordLower", "int", rec_lo, "Each record's lowercase mapping: a delta, or offset | length &lt;&lt; 24 when extended."))
    out.append("")
    out.append(_cs_array("RecordTitle", "int", rec_ti, "Each record's titlecase mapping: a delta, or offset | length &lt;&lt; 24 when extended."))
    out.append("")
    out.append(_cs_array("ExtendedCase", "int", extended, "The code points of the multi-code-point mappings."))
    out.append("    }")
    out.append("}")
    text = "\n".join(out) + "\n"
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("records=%d shift=%d index1=%d x %dB index2=%d x %dB extended=%d hash=%s" %
          (len(records), shift, len(t1), i1size, len(t2), i2size, len(extended), digest))


if __name__ == "__main__":
    main()
