using System;
using System.Text;

namespace NumSharp.IO
{
    /// <summary>The kind of codec a module of Python's <c>encodings</c> package provides (<see cref="PythonCodecs.Modules"/>).</summary>
    internal enum PythonCodecKind
    {
        /// <summary><c>utf_8</c>: CPython's strict UTF-8 codec.</summary>
        Utf8,
        /// <summary><c>utf_8_sig</c>: UTF-8 whose decoder drops a leading BOM and whose encoder writes one.</summary>
        Utf8Sig,
        /// <summary><c>utf_16</c>: UTF-16 with a byte-order mark (read to choose the order, written native little-endian).</summary>
        Utf16,
        /// <summary><c>utf_16_le</c>: little-endian UTF-16 without a BOM.</summary>
        Utf16LE,
        /// <summary><c>utf_16_be</c>: big-endian UTF-16 without a BOM.</summary>
        Utf16BE,
        /// <summary><c>utf_32</c>: UTF-32 with a byte-order mark.</summary>
        Utf32,
        /// <summary><c>utf_32_le</c>: little-endian UTF-32 without a BOM.</summary>
        Utf32LE,
        /// <summary><c>utf_32_be</c>: big-endian UTF-32 without a BOM.</summary>
        Utf32BE,
        /// <summary><c>latin_1</c>: every byte is the code point of its value.</summary>
        Latin1,
        /// <summary><c>ascii</c>: bytes and code points below 128.</summary>
        Ascii,
        /// <summary>A single-byte charmap codec (<c>cp1252</c>, <c>iso8859_15</c>, <c>koi8_r</c>, …) with a 256-entry table.</summary>
        Charmap,
        /// <summary><c>undefined</c>: every decode and encode raises <c>UnicodeError("undefined encoding")</c>.</summary>
        Undefined,
        /// <summary>A bytes-to-bytes codec (<c>base64</c>, <c>hex</c>, <c>rot13</c>, …): not a text encoding.</summary>
        NotText,
        /// <summary>A text codec NumSharp does not implement (the multi-byte CJK codecs, <c>utf_7</c>, <c>idna</c>, …).</summary>
        Unsupported,
    }

    /// <summary>
    ///     Where NumPy's text I/O looks a codec up, which decides two things CPython does differently by caller: whether the
    ///     C fast paths run first, and the advice a non-text codec's <c>LookupError</c> gives.
    /// </summary>
    internal enum CodecLookupContext
    {
        /// <summary><c>bytes.decode(encoding)</c> — a line of a byte stream (<c>PyUnicode_FromEncodedObject</c>).</summary>
        Decode,
        /// <summary><c>str.encode(encoding)</c> — one <c>np.savetxt</c> write to a byte stream (<c>WriteWrap</c>).</summary>
        Encode,
        /// <summary><c>open(path, mode, encoding=…)</c> — a file NumPy opens from a name (TextIOWrapper).</summary>
        Open,
    }

    /// <summary>
    ///     CPython 3.12's codec registry as NumPy's text I/O reaches it: the name normalization of <c>codecs.lookup</c>, the
    ///     alias table and the codec modules of the <c>encodings</c> package (the generated <c>PythonCodecs.Tables.cs</c>).
    /// </summary>
    internal static partial class PythonCodecs
    {
        /// <summary>
        ///     CPython's <c>_Py_normalize_encoding</c>: ASCII letters, digits and <c>'.'</c> are kept (lowercased); every run of
        ///     other characters becomes ONE <c>'_'</c> between kept characters, none at either end (<c>"  UTF--8 "</c> is
        ///     <c>utf_8</c>, <c>"ISO-8859-1"</c> <c>iso_8859_1</c>).
        /// </summary>
        /// <param name="name">The encoding name as given.</param>
        /// <returns>The normalized name (empty when nothing is kept).</returns>
        internal static string Normalize(string name)
        {
            var sb = new StringBuilder(name.Length);
            bool punct = false;
            foreach (char c in name)
            {
                bool keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.';
                if (keep)
                {
                    if (punct && sb.Length > 0)
                        sb.Append('_');
                    punct = false;
                    sb.Append(char.ToLowerInvariant(c));
                }
                else
                {
                    // A non-ASCII character is punctuation too: the C function sees its UTF-8 bytes, none of them alphanumeric.
                    punct = true;
                }
            }
            return sb.ToString();
        }

        /// <summary>
        ///     The codec module <c>encodings.search_function</c> imports for a normalized name: the alias's module first (the
        ///     alias looked up as it is, then with <c>'.'</c> read as <c>'_'</c>), then the name itself; a module name holding
        ///     a <c>'.'</c> is never imported (<c>'utf.8'</c> is unknown, <c>'iso.8859.1'</c> is <c>latin_1</c> by alias).
        /// </summary>
        /// <param name="normalized">The normalized name.</param>
        /// <returns>The module name, or null when no module of that name exists (CPython's <c>unknown encoding</c>).</returns>
        internal static string FindModule(string normalized)
        {
            if (!Aliases.TryGetValue(normalized, out string aliased))
                Aliases.TryGetValue(normalized.Replace('.', '_'), out aliased);
            if (aliased != null && aliased.Length > 0 && aliased.IndexOf('.') < 0 && Modules.ContainsKey(aliased))
                return aliased;
            if (normalized.Length > 0 && normalized.IndexOf('.') < 0 && Modules.ContainsKey(normalized))
                return normalized;
            return null;
        }

        /// <summary>
        ///     The kind of codec <c>PyUnicode_Decode</c> / <c>PyUnicode_AsEncodedString</c> pick WITHOUT the registry for a
        ///     name (the C fast paths, over <c>_Py_normalize_encoding</c> into an 11-byte buffer, so a name normalizing to
        ///     more than 10 characters never takes one): <c>utf8</c> / <c>utf_8</c>, <c>utf16</c> / <c>utf_16</c>,
        ///     <c>utf32</c> / <c>utf_32</c>, <c>ascii</c> / <c>us_ascii</c>, and <c>latin1</c> / <c>latin_1</c> /
        ///     <c>iso_8859_1</c> / <c>iso8859_1</c> — the last of which the registry would resolve to the CHARMAP module
        ///     <c>iso8859_1</c>, whose encode errors read <c>'charmap'</c> where the fast path's read <c>'latin-1'</c>.
        /// </summary>
        /// <param name="normalized">The normalized name.</param>
        /// <param name="kind">The fast path's codec.</param>
        /// <returns>True when a fast path applies.</returns>
        internal static bool TryFastPath(string normalized, out PythonCodecKind kind)
        {
            kind = default;
            if (normalized.Length > 10)
                return false;
            switch (normalized)
            {
                case "utf8":
                case "utf_8":
                    kind = PythonCodecKind.Utf8;
                    return true;
                case "utf16":
                case "utf_16":
                    kind = PythonCodecKind.Utf16;
                    return true;
                case "utf32":
                case "utf_32":
                    kind = PythonCodecKind.Utf32;
                    return true;
                case "ascii":
                case "us_ascii":
                    kind = PythonCodecKind.Ascii;
                    return true;
                case "latin1":
                case "latin_1":
                case "iso_8859_1":
                case "iso8859_1":
                    kind = PythonCodecKind.Latin1;
                    return true;
                default:
                    return false;
            }
        }
    }
}
