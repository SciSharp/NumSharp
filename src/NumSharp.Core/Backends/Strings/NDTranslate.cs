using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

// =============================================================================
// NDTranslate.cs — the compiled tables of np.strings.translate (strings phase 2, P2.6 milestone 3) and CPython's
// translate algorithms over them
// =============================================================================
//
// NUMPY'S STRUCTURE (numpy/_core/strings.py translate, multiarraymodule.c _vec_string_with_args)
//   translate(a, table, deletechars=None) is _vec_string(a_arr, a_arr.dtype, 'translate', args) with args = (table,) for
//   U (dtype.type np.str_: deletechars is IGNORED, not even converted) and [table] + _clean_args(deletechars) for S and
//   StringDType (deletechars passed only when it is not None). _vec_string broadcasts a and the arguments with a
//   multi-iterator (each argument np.asarray'd: a dict is a 0-d object array holding it) and calls, per element in C
//   order, bytes.translate / str.translate with every operand's scalar (NDCharObjectPath reads them: np.str_ / np.bytes_
//   without their trailing NULs, a StringDType element as a Python str or its NA object, a NumPy numeric scalar, an object
//   element as itself).
//
// THE PYTHON METHODS (CPython 3.12)
//   str.translate(table) (unicodeobject.c _PyUnicode_TranslateCharmap, charmaptranslate_lookup / _output): each
//   character's mapping is table[ord(c)]: a LookupError (KeyError, IndexError) keeps the character; None deletes it; an int
//   is a code point (outside range(0x110000), a huge int included: ValueError "character mapping must be in
//   range(0x110000)"); a str replaces it (an empty one deletes it); anything else is TypeError "character mapping must
//   return integer, None or str". A table that cannot be subscripted raises TypeError "'NoneType' object is not
//   subscriptable" at the first character looked up (an empty text looks nothing up). A dict looks its key up by Python
//   equality (97.0 and True == 1 are keys of code points, 'a' never is); a str table is indexed by code point (a one-char
//   str, a unit past U+10FFFF raising CPython's SystemError when it is made), a bytes table gives the byte, a NumPy numeric
//   scalar raises IndexError for every key (so it keeps everything). A U unit past U+10FFFF that is written as itself
//   raises SystemError "invalid maximum character passed to PyUnicode_New" (the writer cannot hold it); one the table maps
//   or deletes does not.
//   bytes.translate(table, delete=b'') (bytesobject.c bytes_translate_impl): the table is None or a buffer of exactly 256
//   bytes (ValueError "translation table must be 256 characters long"), delete a buffer (TypeError "a bytes-like object
//   is required, not 'dict'"), both checked before the input is read — so every call raises, even an empty element's.
//
// THE COMPILED FORMS
//   StrTranslateTable: one int entry per code point below DenseLength — the code point the character becomes, Delete, or
//   -3 - k for extended entry k (a replacement of several code points in the pool, or an error) — then dict keys past the
//   dense range, the rest of a long str / bytes table, and the entry of everything else (the character itself, or the
//   unsubscriptable table's error). The dense length follows the call's work (DenseLimitFor): a 256 KB table compiled for
//   one short text would cost more than the call. BytesTranslateTable: bytes.translate's trans_table[256] (-1 = delete),
//   or the error every call raises. TranslateState carries a table's pinned arrays into a kernel's auxdata.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    /// <summary>One Python exception a translate call raises: its class and its text.</summary>
    internal sealed class TranslateError
    {
        /// <summary>The Python class: <c>TypeError</c>, <c>ValueError</c> or <c>SystemError</c>.</summary>
        internal readonly string PyType;

        /// <summary>The text.</summary>
        internal readonly string Message;

        /// <summary>Creates one.</summary>
        /// <param name="pyType">The Python class.</param>
        /// <param name="message">The text.</param>
        private TranslateError(string pyType, string message)
        {
            PyType = pyType;
            Message = message;
        }

        /// <summary>A <c>TypeError</c>.</summary>
        /// <param name="message">The text.</param>
        /// <returns>The error.</returns>
        internal static TranslateError Type(string message) => new TranslateError("TypeError", message);

        /// <summary>A <c>ValueError</c>.</summary>
        /// <param name="message">The text.</param>
        /// <returns>The error.</returns>
        internal static TranslateError Value(string message) => new TranslateError("ValueError", message);

        /// <summary>CPython's <c>SystemError: invalid maximum character passed to PyUnicode_New</c>: a <c>U</c> unit past
        ///     U+10FFFF in a str the call creates.</summary>
        internal static readonly TranslateError InvalidMaxChar = new TranslateError("SystemError", NDTextElements.InvalidMaxCharText);

        /// <summary><c>ValueError: character mapping must be in range(0x110000)</c>.</summary>
        internal static readonly TranslateError MappingRange = Value("character mapping must be in range(0x110000)");

        /// <summary><c>TypeError: character mapping must return integer, None or str</c>.</summary>
        internal static readonly TranslateError MappingType = Type("character mapping must return integer, None or str");

        /// <summary><c>ValueError: translation table must be 256 characters long</c> (bytes.translate).</summary>
        internal static readonly TranslateError TableLength = Value("translation table must be 256 characters long");

        /// <summary>The C# exception NumPy's Python exception corresponds to (NumSharp's <see cref="TypeError"/>,
        ///     <see cref="ValueError"/> and <see cref="SystemError"/>).</summary>
        /// <returns>The exception (not thrown).</returns>
        internal Exception ToException() => PyType switch
        {
            "TypeError" => new TypeError(Message),
            "ValueError" => new ValueError(Message),
            _ => new SystemError(Message),
        };
    }

    /// <summary>
    ///     A <c>str.translate</c> table compiled for the kernels and the element walk (see the file header): dense entries,
    ///     dict keys past them, a long table's tail, the entry of every other character, and the replacements and errors the
    ///     extended entries index.
    /// </summary>
    /// <remarks>
    ///     An entry is a code point (0 to U+10FFFF), <see cref="Delete"/>, <see cref="Identity"/> (from
    ///     <see cref="Lookup"/> only: the character is kept) or <c>-3 - k</c> for extended entry <c>k</c>, whose pool range
    ///     (<see cref="TranslateState.Index"/>: start, length) holds a replacement, or whose length -1 marks an error
    ///     (<see cref="ErrorOf"/>). Every array a kernel reads is pinned for the table's life, and the table is complete once a
    ///     factory returns (a lookup never adds an entry, so a <see cref="TranslateState"/> never goes stale).
    /// </remarks>
    internal sealed unsafe class StrTranslateTable
    {
        /// <summary>The entry of a deleted character (a <c>None</c> mapping, or an empty str).</summary>
        internal const int Delete = -1;

        /// <summary>The entry of a kept character (a LookupError), past the dense range.</summary>
        internal const int Identity = -2;

        /// <summary>The longest dense range: every BMP code point (256 KB of entries).</summary>
        internal const int MaxDenseLength = 0x10000;

        /// <summary>The dense entries (pinned; empty for a table compiled for one element).</summary>
        private int[] _dense = Array.Empty<int>();

        /// <summary>The entries of dict keys at or past the dense range, or null.</summary>
        private Dictionary<uint, int> _keys;

        /// <summary>A str table's code points when it is longer than the dense range, or null.</summary>
        private uint[] _sequencePoints;

        /// <summary>A bytes table's bytes when it is longer than the dense range, or null.</summary>
        private byte[] _sequenceBytes;

        /// <summary>The extended entry of a str table's unit past U+10FFFF (made once), or 0.</summary>
        private int _invalidEntry;

        /// <summary>The entry of a character past every range: <see cref="Identity"/>, or an unsubscriptable table's error.</summary>
        private int _default = Identity;

        /// <summary>The replacements' code points, while the table is built.</summary>
        private readonly List<uint> _poolList = new();

        /// <summary>Start and length per extended entry (length -1 for an error), while the table is built.</summary>
        private readonly List<long> _indexList = new();

        /// <summary>The error of each extended entry (null for a replacement).</summary>
        private readonly List<TranslateError> _errors = new();

        /// <summary>The replacements' code points (pinned).</summary>
        private uint[] _pool;

        /// <summary>Start and length per extended entry (pinned).</summary>
        private long[] _index;

        /// <summary>The byte map of a table that maps every ASCII character to one ASCII character or deletes it (0xFF:
        ///     deleted), 256 entries whose upper half copies its byte (pinned), or null.</summary>
        private byte[] _ascii;

        /// <summary>The longest replacement in code points (at least 1).</summary>
        internal long MaxReplacement { get; private set; } = 1;

        /// <summary>The number of dense entries.</summary>
        internal int DenseLength => _dense.Length;

        /// <summary>The table's entries are complete; built only by the factories.</summary>
        private StrTranslateTable() { }

        /// <summary>
        ///     Whether the gather path applies: every character past the dense range is kept (no dict key, table tail or
        ///     error past it), NUL maps to itself (a padded or trailing NUL lane then stays 0), and no ASCII character is
        ///     deleted or replaced by several (a table that changes the length of common text would send most elements down
        ///     the gather path and back).
        /// </summary>
        internal bool VectorOk
        {
            get
            {
                if (_keys is not null || _sequencePoints is not null || _sequenceBytes is not null || _default != Identity || _dense.Length == 0 || _dense[0] != 0)
                    return false;
                int n = Math.Min(128, _dense.Length);
                for (int c = 0; c < n; c++)
                    if (_dense[c] < 0)
                        return false;
                return true;
            }
        }

        /// <summary>
        ///     The dense length a call compiles its table with: its work in characters (a <c>U</c> element's units times the
        ///     elements; 16 per <c>StringDType</c> cell), between 256 and <see cref="MaxDenseLength"/>. NumPy builds no table at
        ///     all (it looks each character up in the Python object), so the compile must stay small next to the call.
        /// </summary>
        /// <param name="characters">The characters the call reads (an estimate).</param>
        /// <returns>The dense limit.</returns>
        internal static int DenseLimitFor(long characters) => (int)Math.Clamp(characters, 256, MaxDenseLength);

        // =============================================================================================================
        // The factories (one per kind of table object)
        // =============================================================================================================

        /// <summary>
        ///     A dict table: each key that equals an integer in <c>[0, 2^32)</c> under Python's equality gets the entry of its
        ///     value; an equal later key overwrites an earlier one (a Python dict keeps one value per key; a C# dictionary
        ///     enumerates in insertion order).
        /// </summary>
        /// <param name="dict">The mapping (the <see cref="NDPyObject"/> boundary map gives its keys and values).</param>
        /// <param name="denseLimit">The longest dense range (0: no dense entries, every key in the key map).</param>
        /// <returns>The table.</returns>
        internal static StrTranslateTable FromDict(IDictionary dict, int denseLimit)
        {
            var values = new Dictionary<uint, object>();
            uint maxKey = 0;
            foreach (DictionaryEntry e in dict)
            {
                if (!TryCodePointKey(e.Key, out uint key))
                    continue;
                values[key] = e.Value;
                if (key > maxKey)
                    maxKey = key;
            }
            var t = new StrTranslateTable();
            int dense = denseLimit == 0 ? 0 : (int)Math.Min(Math.Max(256L, (long)maxKey + 1), denseLimit);
            t._dense = GC.AllocateUninitializedArray<int>(dense, pinned: true);
            for (int c = 0; c < dense; c++)
                t._dense[c] = c;
            foreach (var kv in values)
            {
                int entry = t.EntryOf(kv.Value);
                if (kv.Key < (uint)dense)
                {
                    t._dense[kv.Key] = entry;
                }
                else
                {
                    t._keys ??= new Dictionary<uint, int>();
                    t._keys[kv.Key] = entry;
                }
            }
            return t.Finish();
        }

        /// <summary>
        ///     A str table (<c>np.str_</c> or a Python str): the character with code point <c>c &lt; len(table)</c> becomes
        ///     <c>table[c]</c>, any other is kept; a table unit past U+10FFFF is the SystemError of making it a str.
        /// </summary>
        /// <param name="points">The table's code points.</param>
        /// <param name="denseLimit">The longest dense range.</param>
        /// <returns>The table.</returns>
        internal static StrTranslateTable FromCodePoints(ReadOnlySpan<uint> points, int denseLimit)
        {
            var t = new StrTranslateTable();
            foreach (uint cp in points)
                if (cp > 0x10FFFF)
                {
                    // Made once, before any entry refers to it (Lookup must never add entries).
                    t._invalidEntry = t.AddExtended(null, TranslateError.InvalidMaxChar);
                    break;
                }
            int dense = denseLimit == 0 ? 0 : Math.Max(256, Math.Min(points.Length, denseLimit));
            t._dense = GC.AllocateUninitializedArray<int>(dense, pinned: true);
            for (int c = 0; c < dense; c++)
                t._dense[c] = c < points.Length ? t.PointEntry(points[c]) : c;
            if (points.Length > dense)
                t._sequencePoints = points.ToArray();
            return t.Finish();
        }

        /// <summary>
        ///     A bytes table (<c>np.bytes_</c> or Python bytes): the character with code point <c>c &lt; len(table)</c>
        ///     becomes the code point <c>table[c]</c> (the byte's value), any other is kept.
        /// </summary>
        /// <param name="bytes">The table's bytes.</param>
        /// <param name="denseLimit">The longest dense range.</param>
        /// <returns>The table.</returns>
        internal static StrTranslateTable FromBytes(ReadOnlySpan<byte> bytes, int denseLimit)
        {
            var t = new StrTranslateTable();
            int dense = denseLimit == 0 ? 0 : Math.Max(256, Math.Min(bytes.Length, denseLimit));
            t._dense = GC.AllocateUninitializedArray<int>(dense, pinned: true);
            for (int c = 0; c < dense; c++)
                t._dense[c] = c < bytes.Length ? bytes[c] : c;
            if (bytes.Length > dense)
                t._sequenceBytes = bytes.ToArray();
            return t.Finish();
        }

        /// <summary>
        ///     A Python list or tuple table (an object-array element): the character with code point <c>c &lt; len(table)</c>
        ///     takes the entry of <c>table[c]</c> as a dict value would (None deletes, an int is a code point, a str replaces),
        ///     any other is kept (IndexError).
        /// </summary>
        /// <param name="items">The items, by position.</param>
        /// <param name="denseLimit">The longest dense range.</param>
        /// <returns>The table.</returns>
        internal static StrTranslateTable FromList(IList items, int denseLimit)
        {
            var t = new StrTranslateTable();
            int count = items.Count;
            int dense = denseLimit == 0 ? 0 : Math.Max(256, Math.Min(count, denseLimit));
            t._dense = GC.AllocateUninitializedArray<int>(dense, pinned: true);
            for (int c = 0; c < dense; c++)
                t._dense[c] = c < count ? t.EntryOf(items[c]) : c;
            // Positions past the dense range are keys like a dict's, so a lookup never computes (or adds) an entry.
            for (int c = dense; c < count; c++)
            {
                t._keys ??= new Dictionary<uint, int>();
                t._keys[(uint)c] = t.EntryOf(items[c]);
            }
            return t.Finish();
        }

        /// <summary>A table whose every lookup raises IndexError (a NumPy numeric scalar): every character is kept.</summary>
        /// <param name="denseLimit">The longest dense range (0: no dense entries).</param>
        /// <returns>The table.</returns>
        internal static StrTranslateTable KeepAll(int denseLimit)
        {
            var t = new StrTranslateTable();
            int dense = denseLimit == 0 ? 0 : 256;
            t._dense = GC.AllocateUninitializedArray<int>(dense, pinned: true);
            for (int c = 0; c < dense; c++)
                t._dense[c] = c;
            return t.Finish();
        }

        /// <summary>
        ///     A table <c>PyObject_GetItem</c> refuses (None, a Python int or float, ...): the first character looked up raises
        ///     <c>TypeError: '{type}' object is not subscriptable</c>.
        /// </summary>
        /// <param name="typeName">The object's Python type name (<c>NoneType</c>, <c>int</c>, <c>float</c>, ...).</param>
        /// <returns>The table.</returns>
        internal static StrTranslateTable Unsubscriptable(string typeName)
        {
            var t = new StrTranslateTable();
            t._default = t.AddExtended(null, TranslateError.Type($"'{typeName}' object is not subscriptable"));
            return t.Finish();
        }

        /// <summary>The pinned arrays of a built table: the pool, the index, and the ASCII map when one applies.</summary>
        /// <returns>The table.</returns>
        private StrTranslateTable Finish()
        {
            // At least one element each, so the pinned addresses are always real ones.
            _pool = GC.AllocateUninitializedArray<uint>(Math.Max(1, _poolList.Count), pinned: true);
            _poolList.CopyTo(_pool);
            _index = GC.AllocateUninitializedArray<long>(Math.Max(2, _indexList.Count), pinned: true);
            _indexList.CopyTo(_index);
            if (_dense.Length >= 128)
            {
                // Every ASCII character to one ASCII character, or deleted (0xFF in the map: no ASCII byte has it).
                bool ascii = true;
                for (int c = 0; c < 128 && ascii; c++)
                    ascii = (uint)_dense[c] < 128 || _dense[c] == Delete;
                if (ascii)
                {
                    // 256 entries: the upper half copies its byte, so a table that touches ASCII only maps any UTF-8 text
                    // through it (0xFF, the deletion marker, never occurs in UTF-8).
                    _ascii = GC.AllocateUninitializedArray<byte>(256, pinned: true);
                    for (int c = 0; c < 128; c++)
                    {
                        _ascii[c] = _dense[c] == Delete ? (byte)0xFF : (byte)_dense[c];
                        AsciiDeletes |= _dense[c] == Delete;
                    }
                    for (int c = 128; c < 256; c++)
                        _ascii[c] = (byte)c;
                }
            }
            SurrogateOutput = MapsToSurrogate(_dense) || HasSurrogate(_pool) || HasSurrogate(_sequencePoints);
            if (!SurrogateOutput && _keys is not null)
                foreach (int e in _keys.Values)
                    SurrogateOutput |= IsSurrogate((uint)e);
            AsciiOnly = ComputeAsciiOnly();
            return this;
        }

        /// <summary>
        ///     Whether some character can become a surrogate code point (U+D800–U+DFFF): a <c>StringDType</c> result cannot hold
        ///     one (its UTF-8 encoding refuses it), so its elements are checked after translating.
        /// </summary>
        internal bool SurrogateOutput { get; private set; }

        /// <summary>Whether the ASCII map deletes some ASCII character (it is then a compaction map, not one-to-one).</summary>
        internal bool AsciiDeletes { get; private set; }

        /// <summary>
        ///     Whether every character past the dense range is kept: no dict key, table tail or error lives past it, so a
        ///     lookup there never needs the managed table.
        /// </summary>
        internal bool HighIdentity => _keys is null && _sequencePoints is null && _sequenceBytes is null && _default == Identity;

        /// <summary>
        ///     Whether the table touches ASCII only: every non-ASCII character is kept, and every ASCII character becomes ASCII
        ///     text (one character, a replacement of ASCII characters, nothing) or raises. Then a UTF-8 text translates byte
        ///     for byte — a byte of 0x80 or more belongs to a kept character and is copied — with no decoding (the
        ///     <c>str.maketrans</c> tables of ASCII text are all of this kind).
        /// </summary>
        internal bool AsciiOnly { get; private set; }

        /// <summary>
        ///     Computes <see cref="AsciiOnly"/> (see there) once the entries are complete.
        /// </summary>
        /// <returns>True when the table touches ASCII only.</returns>
        private bool ComputeAsciiOnly()
        {
            if (!HighIdentity || _dense.Length < 128)
                return false;
            for (int c = 128; c < _dense.Length; c++)
                if (_dense[c] != c)
                    return false;
            for (int c = 0; c < 128; c++)
            {
                int e = _dense[c];
                if (e >= 128)
                    return false;
                if (e < Delete)
                {
                    // A replacement must be ASCII; an error entry raises whatever the text, so it does not matter here.
                    int x = -3 - e;
                    long length = _indexList[2 * x + 1];
                    for (long j = 0; j < length; j++)
                        if (_poolList[(int)(_indexList[2 * x] + j)] >= 128)
                            return false;
                }
            }
            return true;
        }

        /// <summary>Whether a value (an entry or a code point) is a surrogate code point.</summary>
        /// <param name="v">The value (a negative entry, cast to <see cref="uint"/>, is past every surrogate).</param>
        /// <returns>True for U+D800–U+DFFF.</returns>
        private static bool IsSurrogate(uint v) => v - 0xD800u < 0x800u;

        /// <summary>
        ///     Whether a dense entry maps a character to a surrogate code point other than itself (an unmapped character's
        ///     entry is the character, and a <c>StringDType</c> text never holds a surrogate to keep).
        /// </summary>
        /// <param name="dense">The dense entries.</param>
        /// <returns>True when one maps to a surrogate.</returns>
        private static bool MapsToSurrogate(int[] dense)
        {
            for (int c = 0; c < dense.Length; c++)
                if (dense[c] != c && IsSurrogate((uint)dense[c]))
                    return true;
            return false;
        }

        /// <summary>Whether an array of code points holds a surrogate.</summary>
        /// <param name="values">The code points (null for none).</param>
        /// <returns>True when one is a surrogate.</returns>
        private static bool HasSurrogate(uint[] values)
        {
            if (values is null)
                return false;
            foreach (uint v in values)
                if (IsSurrogate(v))
                    return true;
            return false;
        }

        // =============================================================================================================
        // Entries
        // =============================================================================================================

        /// <summary>The entry of a str table's code point: itself, or the SystemError of a unit past U+10FFFF.</summary>
        /// <param name="cp">The code point.</param>
        /// <returns>The entry.</returns>
        private int PointEntry(uint cp) => cp <= 0x10FFFF ? (int)cp : _invalidEntry;

        /// <summary>
        ///     The entry of a dict value (CPython's <c>charmaptranslate_lookup</c> on what the lookup returned): None deletes,
        ///     a Python int (a <see cref="bool"/> is one) is a code point (an error entry outside <c>range(0x110000)</c>), a str
        ///     replaces (an empty one deletes; a <see cref="char"/> is a one-unit str), anything else is an error entry.
        /// </summary>
        /// <param name="value">The C# value standing in for the Python object (the <see cref="NDPyObject"/> map).</param>
        /// <returns>The entry.</returns>
        private int EntryOf(object value)
        {
            switch (value)
            {
                case null:
                    return Delete;
                case string s:
                {
                    var cps = CodePointsOf(s);
                    return cps.Length switch
                    {
                        0 => Delete,
                        1 => (int)cps[0],
                        _ => AddExtended(cps, null),
                    };
                }
                case char ch:
                    return ch;
                case bool b:
                    return b ? 1 : 0;
                case sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger:
                {
                    var v = value is BigInteger big ? big : new BigInteger(Convert.ToDecimal(value));
                    // PyLong_AsLong overflows for a huge int; CPython then reports the range error, not the overflow.
                    return v >= 0 && v <= 0x10FFFF ? (int)v : AddExtended(null, TranslateError.MappingRange);
                }
                default:
                    return AddExtended(null, TranslateError.MappingType);
            }
        }

        /// <summary>Adds an extended entry (a replacement, or an error).</summary>
        /// <param name="points">A replacement's code points, or null for an error.</param>
        /// <param name="error">The error, or null for a replacement.</param>
        /// <returns>The entry that indexes it.</returns>
        private int AddExtended(uint[] points, TranslateError error)
        {
            if (error is not null)
            {
                _indexList.Add(0);
                _indexList.Add(-1);
            }
            else
            {
                _indexList.Add(_poolList.Count);
                _indexList.Add(points.Length);
                _poolList.AddRange(points);
                if (points.Length > MaxReplacement)
                    MaxReplacement = points.Length;
            }
            _errors.Add(error);
            return -3 - (_errors.Count - 1);
        }

        /// <summary>The error an extended entry raises.</summary>
        /// <param name="entry">An extended entry whose length is -1.</param>
        /// <returns>The error.</returns>
        internal TranslateError ErrorOf(int entry) => _errors[-3 - entry];

        /// <summary>
        ///     Whether a dict key equals an integer in <c>[0, 2^32)</c> under Python's <c>==</c>, with an equal hash (the dict
        ///     lookup of a code point): an integer or a Python int of that value, a bool (False is 0, True is 1), a float /
        ///     <see cref="Half"/> / <see cref="decimal"/> / <see cref="Complex"/> of that integral value (no imaginary part);
        ///     a str (a C# <see cref="string"/> or <see cref="char"/>) never equals an int.
        /// </summary>
        /// <param name="key">The C# key.</param>
        /// <param name="cp">Receives the integer.</param>
        /// <returns>True when the key can match a code point lookup.</returns>
        /// <remarks>
        ///     Keys past U+10FFFF still matter: a <c>U</c> unit past it is looked up by its value. CPython converts the unit
        ///     with <c>PyLong_FromLong((long)c)</c>, a 32-bit C long on Windows, so a unit of 2^31 or more looks up a NEGATIVE
        ///     key there; NumSharp models the 64-bit long (Linux / macOS), whose key is the unit's value.
        /// </remarks>
        private static bool TryCodePointKey(object key, out uint cp)
        {
            cp = 0;
            double d;
            switch (key)
            {
                case bool b:
                    cp = b ? 1u : 0u;
                    return true;
                case sbyte or byte or short or ushort or int or uint or long or ulong:
                {
                    var v = Convert.ToDecimal(key);
                    if (v < 0 || v > uint.MaxValue)
                        return false;
                    cp = (uint)v;
                    return true;
                }
                case BigInteger big:
                    if (big.Sign < 0 || big > uint.MaxValue)
                        return false;
                    cp = (uint)big;
                    return true;
                case float f:
                    d = f;
                    break;
                case double dd:
                    d = dd;
                    break;
                case Half h:
                    d = (double)h;
                    break;
                case decimal m:
                    if (m != decimal.Truncate(m) || m < 0 || m > uint.MaxValue)
                        return false;
                    cp = (uint)m;
                    return true;
                case Complex z:
                    if (z.Imaginary != 0)
                        return false;
                    d = z.Real;
                    break;
                default:
                    return false;
            }
            // 97.0 == 97 and their hashes agree; -0.0 == 0. NaN and infinities equal no integer.
            if (!(d >= 0 && d <= uint.MaxValue) || d != Math.Floor(d))
                return false;
            cp = (uint)d;
            return true;
        }

        /// <summary>The code points of a C# string as Python's <c>str</c>: a surrogate pair is one code point, a lone
        ///     surrogate stays the code point it is.</summary>
        /// <param name="s">The string.</param>
        /// <returns>The code points.</returns>
        internal static uint[] CodePointsOf(string s)
        {
            var list = new List<uint>(s.Length);
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                if (char.IsHighSurrogate(c) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                {
                    list.Add((uint)char.ConvertToUtf32(c, s[i + 1]));
                    i++;
                }
                else
                {
                    list.Add(c);
                }
            }
            return list.ToArray();
        }

        // =============================================================================================================
        // Lookups and the managed translate (the element walk)
        // =============================================================================================================

        /// <summary>The entry of a character (<c>PyObject_GetItem(table, ord(c))</c>, compiled).</summary>
        /// <param name="c">The code point (a <c>U</c> unit can be past U+10FFFF).</param>
        /// <returns>The entry; <see cref="Identity"/> for a kept character past the dense range.</returns>
        internal int Lookup(uint c)
        {
            if (c < (uint)_dense.Length)
                return _dense[c];
            if (_keys is not null)
                return _keys.TryGetValue(c, out int e) ? e : _default;
            if (_sequencePoints is not null)
                return c < (uint)_sequencePoints.Length ? PointEntry(_sequencePoints[c]) : Identity;
            if (_sequenceBytes is not null)
                return c < (uint)_sequenceBytes.Length ? _sequenceBytes[c] : Identity;
            return _default;
        }

        /// <summary>
        ///     CPython's <c>_PyUnicode_TranslateCharmap</c> over code points: each character's entry appended to
        ///     <paramref name="dst"/> (characters past its capacity counted, not written), stopping at the first error.
        /// </summary>
        /// <param name="s">The text's code points.</param>
        /// <param name="dst">The output (its length is the capacity).</param>
        /// <param name="total">Receives the length of the whole result (it can exceed the capacity).</param>
        /// <returns>The error the call raises, or null.</returns>
        /// <remarks>The scan continues past the capacity, as CPython builds the whole string before setitem cuts it: an error
        ///     after the cut still raises. A character kept as itself past U+10FFFF raises SystemError (the writer cannot
        ///     hold it); a mapped or deleted one does not.</remarks>
        internal TranslateError TranslateInto(ReadOnlySpan<uint> s, Span<uint> dst, out long total)
        {
            long k = 0;
            for (int i = 0; i < s.Length; i++)
            {
                uint c = s[i];
                int e = Lookup(c);
                if (e >= 0)
                {
                    if (k < dst.Length)
                        dst[(int)k] = (uint)e;
                    k++;
                }
                else if (e == Identity)
                {
                    if (c > 0x10FFFF)
                    {
                        total = k;
                        return TranslateError.InvalidMaxChar;
                    }
                    if (k < dst.Length)
                        dst[(int)k] = c;
                    k++;
                }
                else if (e != Delete)
                {
                    int x = -3 - e;
                    long length = _index[2 * x + 1];
                    if (length < 0)
                    {
                        total = k;
                        return _errors[x];
                    }
                    long start = _index[2 * x];
                    for (long j = 0; j < length; j++)
                    {
                        if (k < dst.Length)
                            dst[(int)k] = _pool[start + j];
                        k++;
                    }
                }
            }
            total = k;
            return null;
        }

        /// <summary>
        ///     Fills a kernel's state with the table's pinned arrays and flags; the caller sets <see cref="TranslateState.Table"/>
        ///     to a <see cref="GCHandle"/> of this table (the slow lookups past the dense range).
        /// </summary>
        /// <param name="st">The state.</param>
        internal void Prepare(TranslateState* st)
        {
            st->Status = 0;
            st->Dense = _dense.Length == 0 ? null : (int*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_dense));
            st->DenseLength = _dense.Length;
            st->Pool = (uint*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_pool));
            st->Index = (long*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_index));
            st->AsciiMap = _ascii is null ? null : (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_ascii));
            st->MaxReplacement = MaxReplacement;
            var flags = TranslateFlags.None;
            if (VectorOk)
                flags |= TranslateFlags.VectorOk;
            if (_ascii is not null)
                flags |= AsciiDeletes ? TranslateFlags.AsciiCompact : TranslateFlags.AsciiOneToOne;
            if (SurrogateOutput)
                flags |= TranslateFlags.SurrogateOutput;
            if (HighIdentity)
                flags |= TranslateFlags.HighIdentity;
            if (AsciiOnly)
                flags |= TranslateFlags.AsciiOnly;
            // NUL to NUL or deleted: a U element's trailing NULs may be translated with the rest (see NulPreserved).
            if (_dense.Length > 0 && (_dense[0] == 0 || _dense[0] == Delete))
                flags |= TranslateFlags.NulPreserved;
            st->Flags = (int)flags;
        }
    }

    /// <summary>
    ///     A <c>bytes.translate</c> table and delete set (see the file header): the 256 entries, a deleted byte's -1; or the
    ///     error every call with these arguments raises.
    /// </summary>
    internal sealed unsafe class BytesTranslateTable
    {
        /// <summary>The 256 entries (pinned): the byte a byte becomes, or -1 when it is deleted.</summary>
        private readonly int[] _trans;

        /// <summary>Whether no byte is deleted (an element keeps its length).</summary>
        internal readonly bool NoDeletes;

        /// <summary>The error <c>bytes.translate</c> raises on every call with these arguments, or null.</summary>
        internal readonly TranslateError Error;

        /// <summary>Creates one.</summary>
        /// <param name="trans">The entries (null with an error).</param>
        /// <param name="noDeletes">Whether no byte is deleted.</param>
        /// <param name="error">The call's error, or null.</param>
        private BytesTranslateTable(int[] trans, bool noDeletes, TranslateError error)
        {
            _trans = trans;
            NoDeletes = noDeletes;
            Error = error;
        }

        /// <summary>The table of a call that raises: <paramref name="error"/> before any input is read.</summary>
        /// <param name="error">The error.</param>
        /// <returns>The table.</returns>
        internal static BytesTranslateTable Failed(TranslateError error) => new BytesTranslateTable(null, true, error);

        /// <summary>
        ///     <c>bytes.translate(table, delete)</c>'s trans_table: the table's 256 bytes (a None table maps every byte to
        ///     itself), then -1 for every byte of the delete set.
        /// </summary>
        /// <param name="table">The table's bytes (exactly 256: the caller checked), or an empty span with
        ///     <paramref name="hasTable"/> false for None.</param>
        /// <param name="hasTable">False for a None table.</param>
        /// <param name="delete">The delete set's bytes (empty when NumPy passes none).</param>
        /// <returns>The table.</returns>
        internal static BytesTranslateTable Create(ReadOnlySpan<byte> table, bool hasTable, ReadOnlySpan<byte> delete)
        {
            var trans = GC.AllocateUninitializedArray<int>(256, pinned: true);
            for (int c = 0; c < 256; c++)
                trans[c] = hasTable ? table[c] : c;
            foreach (byte b in delete)
                trans[b] = -1;
            return new BytesTranslateTable(trans, delete.IsEmpty, null);
        }

        /// <summary>One <c>S</c> element through <c>bytes.translate</c>: each byte mapped or deleted.</summary>
        /// <param name="s">The element's bytes (trailing NULs already dropped: the <c>np.bytes_</c> scalar).</param>
        /// <param name="dst">The output (at least <paramref name="s"/>'s length).</param>
        /// <returns>The output length.</returns>
        internal int TranslateInto(ReadOnlySpan<byte> s, Span<byte> dst)
        {
            int k = 0;
            foreach (byte b in s)
            {
                int v = _trans[b];
                if (v >= 0)
                    dst[k++] = (byte)v;
            }
            return k;
        }

        /// <summary>Fills a kernel's state: the 256 entries and the no-delete flag.</summary>
        /// <param name="st">The state.</param>
        internal void Prepare(TranslateState* st)
        {
            st->Status = 0;
            st->Dense = (int*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_trans));
            st->DenseLength = 256;
            var flags = NoDeletes ? TranslateFlags.NoDeletes : TranslateFlags.None;
            if (_trans[0] is 0 or -1)
                flags |= TranslateFlags.NulPreserved;
            st->Flags = (int)flags;
        }
    }

    /// <summary>
    ///     The per-call state a translate kernel reads (its auxdata; inside the <c>StringDType</c> loop's
    ///     <see cref="VStringMethodAux.Extra"/>): a compiled table's pinned arrays, and the error word the bodies set.
    /// </summary>
    /// <remarks>One block per call on the driver's stack; the managed table it points into stays reachable through the
    ///     driver (and <see cref="Table"/>'s handle) for the whole call.</remarks>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct TranslateState
    {
        /// <summary>0, or 1 once an element raised (the driver then finds NumPy's first error by a C-order walk).</summary>
        internal long Status;

        /// <summary><c>S</c>: the 256 byte entries (-1 deleted); <c>U</c> / <c>StringDType</c>: the dense entries (null when
        ///     the table has none).</summary>
        internal int* Dense;

        /// <summary>The number of dense entries.</summary>
        internal int DenseLength;

        /// <summary>The <see cref="TranslateFlags"/>.</summary>
        internal int Flags;

        /// <summary>The replacements' code points.</summary>
        internal uint* Pool;

        /// <summary>Start and length per extended entry (length -1: the entry raises).</summary>
        internal long* Index;

        /// <summary>The byte map of a table that maps every ASCII character to one ASCII character or deletes it (0xFF:
        ///     deleted; see <see cref="TranslateFlags.AsciiOneToOne"/> / <see cref="TranslateFlags.AsciiCompact"/>): 256
        ///     entries, the upper half copying its byte; or null.</summary>
        internal byte* AsciiMap;

        /// <summary>The longest replacement in code points (at least 1): sizes a <c>StringDType</c> result's scratch.</summary>
        internal long MaxReplacement;

        /// <summary>The <see cref="GCHandle"/> of the managed <see cref="StrTranslateTable"/> (lookups past the dense range).</summary>
        internal IntPtr Table;
    }

    /// <summary>The <see cref="TranslateState.Flags"/> bits.</summary>
    [Flags]
    internal enum TranslateFlags
    {
        /// <summary>None.</summary>
        None = 0,

        /// <summary><c>U</c>: the gather path applies (<see cref="StrTranslateTable.VectorOk"/>).</summary>
        VectorOk = 1,

        /// <summary><c>S</c>: no byte is deleted.</summary>
        NoDeletes = 2,

        /// <summary><c>StringDType</c>: every ASCII character becomes one ASCII character, so an ASCII text translates byte for
        ///     byte through <see cref="TranslateState.AsciiMap"/> at its own size.</summary>
        AsciiOneToOne = 4,

        /// <summary>
        ///     <c>StringDType</c>: the table can produce a surrogate code point (an int value or a replacement in
        ///     U+D800–U+DFFF), which a <c>StringDType</c> element cannot hold — its setitem fails, and <c>_vec_string</c> reports
        ///     that as <c>TypeError: result array type does not match underlying function</c>.
        /// </summary>
        SurrogateOutput = 8,

        /// <summary>
        ///     <c>U</c> / <c>StringDType</c>: every character past the dense range is kept (no dict key, table tail or error
        ///     there), so the kernels never ask the managed table (a kept unit past U+10FFFF still raises).
        /// </summary>
        HighIdentity = 16,

        /// <summary>
        ///     <c>StringDType</c>: every ASCII character becomes one ASCII character or is deleted, at least one is deleted:
        ///     an ASCII text is compacted byte for byte through <see cref="TranslateState.AsciiMap"/> (0xFF = delete).
        /// </summary>
        AsciiCompact = 32,

        /// <summary>
        ///     <c>S</c> / <c>U</c>: NUL maps to NUL or is deleted, so translating an element's trailing NULs changes nothing the
        ///     zero padding would not: the kernel maps the whole width without first finding the text's end.
        /// </summary>
        NulPreserved = 64,

        /// <summary><c>StringDType</c>: the table touches ASCII only (<see cref="StrTranslateTable.AsciiOnly"/>): any UTF-8 text
        ///     translates byte for byte, its non-ASCII bytes copied.</summary>
        AsciiOnly = 128,
    }
}
