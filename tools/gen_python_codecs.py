"""Generates src/NumSharp.Core/IO/TextReading/PythonCodecs.Tables.cs from the CPython that runs it.

NumPy's text I/O (np.loadtxt / np.savetxt) decodes and encodes through Python's codec registry, so NumSharp resolves an
encoding NAME the way CPython does and reproduces the codecs it implements exactly. This script records what that needs:

* ``Aliases`` -- ``encodings.aliases.aliases``: the normalized alias -> codec module table ``encodings.search_function``
  consults first;
* ``Modules`` -- every module of the ``encodings`` package with the codec name ``codecs.lookup(module).name`` reports and
  the kind NumSharp gives it (a Unicode codec, Latin-1, ASCII, a single-byte charmap, ``undefined``, a non-text
  bytes-to-bytes codec, or a codec NumSharp does not implement);
* ``CharmapTables`` -- the decoding table of every single-byte charmap codec: 256 characters, U+FFFE for a byte the codec
  cannot decode (CPython's own marker). The encode map is the inverse of that table, the LAST byte winning where two bytes
  decode to one character (``PyUnicode_BuildEncodingMap``); the script verifies that rule against ``str.encode`` over the
  whole BMP for every codec and fails when a codec departs from it.

Run with the CPython the oracles run (3.12, the NumPy 2.4.2 oracle interpreter):  python tools/gen_python_codecs.py
"""
import codecs
import importlib
import os
import pkgutil
import sys
import unicodedata

import encodings
from encodings import aliases

# The modules whose codec NumSharp implements statement for statement (the kind names match PythonCodecKind).
_UNICODE_KINDS = {
    "utf_8": "Utf8", "utf_8_sig": "Utf8Sig",
    "utf_16": "Utf16", "utf_16_le": "Utf16LE", "utf_16_be": "Utf16BE",
    "utf_32": "Utf32", "utf_32_le": "Utf32LE", "utf_32_be": "Utf32BE",
    "latin_1": "Latin1", "ascii": "Ascii", "undefined": "Undefined",
}
# The modules the registry lists that are not codecs.
_NOT_CODECS = {"aliases"}


def _classify(module):
    """The kind of one encodings module.

    :param module: the module name.
    :returns: (kind, codec name), or None for a module that is not a codec on this platform.
    """
    if module in _NOT_CODECS:
        return None
    try:
        info = codecs.lookup(module)
    except LookupError:
        return None
    if not getattr(info, "_is_text_encoding", True):
        return "NotText", info.name
    if module in _UNICODE_KINDS:
        return _UNICODE_KINDS[module], info.name
    mod = importlib.import_module("encodings." + module)
    if hasattr(mod, "decoding_table") or hasattr(mod, "decoding_map"):
        single = True
        for b in range(256):
            try:
                if len(bytes([b]).decode(module)) > 1:
                    single = False
                    break
            except UnicodeDecodeError:
                pass
        if single:
            return "Charmap", info.name
    return "Unsupported", info.name


def _decoding_table(module):
    """The codec's 256-entry decoding table, U+FFFE for an undefined byte.

    :param module: a single-byte charmap module.
    :returns: the table as a str of 256 characters.
    :raises AssertionError: a byte decodes to U+FFFE itself (the marker would be ambiguous).
    """
    out = []
    for b in range(256):
        try:
            ch = bytes([b]).decode(module)
            assert ch != "\ufffe", (module, b)
            out.append(ch)
        except UnicodeDecodeError:
            out.append("\ufffe")
    return "".join(out)


def _verify_encode_rule(module, table):
    """Checks that ``str.encode`` over the BMP is the inverse of the table, the last byte winning.

    :param module: the codec module.
    :param table: its decoding table.
    :raises AssertionError: a character encodes differently from the rule.
    """
    inverse = {}
    for b, ch in enumerate(table):
        if ch != "\ufffe":
            inverse[ch] = b  # a later byte overwrites an earlier one
    for cp in range(0x10000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        try:
            got = ch.encode(module)
        except UnicodeEncodeError:
            got = None
        want = bytes([inverse[ch]]) if ch in inverse else None
        assert got == want, (module, hex(cp), got, want)


def _cs_string(s):
    """A C# string literal of ``s``: printable characters as they are, every other one as a \\u escape.

    :param s: the text.
    :returns: the literal, quotes included.
    """
    out = ['"']
    for ch in s:
        cp = ord(ch)
        cat = unicodedata.category(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif 0x20 <= cp < 0x7F:
            out.append(ch)
        elif cp < 0x80 or cat[0] in ("C", "Z", "M") or cp == 0xFFFE:
            out.append(f"\\u{cp:04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def main():
    """Writes the table file next to TextIOCodec.cs."""
    modules = sorted(m.name for m in pkgutil.iter_modules(encodings.__path__))
    kinds = {}
    for m in modules:
        k = _classify(m)
        if k is not None:
            kinds[m] = k
    charmaps = {m: _decoding_table(m) for m, (k, _) in kinds.items() if k == "Charmap"}
    for m, t in charmaps.items():
        _verify_encode_rule(m, t)
    # Every alias must name a module this platform knows, or the lookup would fall through to the alias's own name.
    alias_items = sorted(aliases.aliases.items())

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "src", "NumSharp.Core", "IO", "TextReading", "PythonCodecs.Tables.cs")
    lines = []
    w = lines.append
    w("// <auto-generated>")
    w(f"// CPython {sys.version.split()[0]}'s codec registry, as NumSharp's text I/O resolves an encoding name against it.")
    w("// Generated by tools/gen_python_codecs.py from the CPython the oracles run: do not edit by hand, re-run the script.")
    w("//   Aliases       encodings.aliases.aliases (normalized alias -> codec module).")
    w("//   Modules       every module of the encodings package: codecs.lookup(module).name and the kind NumSharp gives it.")
    w("//   CharmapTables the decoding table of every single-byte charmap codec ('\\uFFFE' marks a byte it cannot decode);")
    w("//                 the encode map is its inverse, the last byte winning (verified over the BMP by the script).")
    w("// </auto-generated>")
    w("using System.Collections.Generic;")
    w("")
    w("namespace NumSharp.IO")
    w("{")
    w("    internal static partial class PythonCodecs")
    w("    {")
    w("        /// <summary>encodings.aliases.aliases: a normalized alias and the codec module it names.</summary>")
    w("        internal static readonly Dictionary<string, string> Aliases = new Dictionary<string, string>(System.StringComparer.Ordinal)")
    w("        {")
    for k, v in alias_items:
        w(f"            [{_cs_string(k)}] = {_cs_string(v)},")
    w("        };")
    w("")
    w("        /// <summary>Every codec module of the encodings package: its kind and the name codecs.lookup reports.</summary>")
    w("        internal static readonly Dictionary<string, (PythonCodecKind Kind, string Name)> Modules =")
    w("            new Dictionary<string, (PythonCodecKind Kind, string Name)>(System.StringComparer.Ordinal)")
    w("        {")
    for m in sorted(kinds):
        k, name = kinds[m]
        w(f"            [{_cs_string(m)}] = (PythonCodecKind.{k}, {_cs_string(name)}),")
    w("        };")
    w("")
    w("        /// <summary>The decoding table of every single-byte charmap module (256 characters, U+FFFE undefined).</summary>")
    w("        internal static readonly Dictionary<string, string> CharmapTables = new Dictionary<string, string>(System.StringComparer.Ordinal)")
    w("        {")
    for m in sorted(charmaps):
        w(f"            [{_cs_string(m)}] =")
        t = charmaps[m]
        for i in range(0, 256, 64):
            sep = " +" if i < 192 else ","
            w(f"                {_cs_string(t[i:i + 64])}{sep}")
    w("        };")
    w("    }")
    w("}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}: {len(alias_items)} aliases, {len(kinds)} modules, {len(charmaps)} charmap tables")


if __name__ == "__main__":
    main()
