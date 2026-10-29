using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using NumSharp.Backends;
using NumSharp.IO; // PyLiteral.Repr: NumPy's error texts quote the dtype string with %R

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the character for the minimum-size type to which given types can be safely cast.
        ///     The returned type character must represent the smallest size dtype such that an array of the returned type can handle the data from an array of all types in typechars(or if typechars is an array, then its dtype.char).
        /// </summary>
        /// <param name="typechars">every character represents a type. see <see cref="DType.@char"/></param>
        /// <param name="typeset">The set of characters that the returned character is chosen from. The default set is ‘GDFgdf’.</param>
        /// <param name="default">The default character, this is returned if none of the characters in typechars matches a character in typeset.</param>
        /// <returns>The character representing the minimum-size type that was found.</returns>
        /// <exception cref="TypeError"><paramref name="typechars"/> is null, or <paramref name="typeset"/> is null while there is a character to test against it (Python's iteration / <c>in</c> errors).</exception>
        /// <exception cref="ValueError">
        ///     A matching character is not one of NumPy's sized codes <c>'GDFgdfQqLlIiHhBb?'</c> (<c>'e'</c>, <c>'n'</c>,
        ///     <c>'p'</c>, …) and the answer is not the <c>'F'</c> + <c>'d'</c> rule: NumPy picks the minimum with
        ///     <c>_typecodes_by_elsize.index</c>, whose <c>str.index</c> raises "substring not found" for it.
        /// </exception>
        /// <remarks>
        ///     Port of <c>numpy/lib/_type_check_impl.py::mintypecode</c>: the characters of <paramref name="typechars"/> that
        ///     are in <paramref name="typeset"/>, the default when there are none, <c>'D'</c> when both <c>'F'</c> and
        ///     <c>'d'</c> are among them, else the one listed first in <c>'GDFgdfQqLlIiHhBb?'</c>.
        /// </remarks>
        public static char mintypecode(string typechars, string typeset = "GDFgdf", char @default = 'd')
        {
            if (typechars == null)
                throw new TypeError("'NoneType' object is not iterable");
            return MinTypeCode(typechars, typeset, @default);
        }

        /// <summary>
        ///     Return the character for the minimum-size type to which given types can be safely cast.
        ///     The returned type character must represent the smallest size dtype such that an array of the returned type can handle the data from an array of all types in typechars(or if typechars is an array, then its dtype.char).
        /// </summary>
        /// <param name="typechars">The type characters, one per type (see <see cref="DType.@char"/>).</param>
        /// <param name="typeset">The set of characters that the returned character is chosen from. The default set is ‘GDFgdf’.</param>
        /// <param name="default">The default character, this is returned if none of the characters in typechars matches a character in typeset.</param>
        /// <returns>The character representing the minimum-size type that was found.</returns>
        /// <exception cref="TypeError"><paramref name="typechars"/> is null, or <paramref name="typeset"/> is null while there is a character to test against it.</exception>
        /// <exception cref="ValueError">A matching character is not one of NumPy's sized codes (see <see cref="mintypecode(string, string, char)"/>).</exception>
        public static char mintypecode(char[] typechars, string typeset = "GDFgdf", char @default = 'd')
        {
            if (typechars == null)
                throw new TypeError("'NoneType' object is not iterable");
            return MinTypeCode(typechars, typeset, @default);
        }

        /// <summary>
        ///     The body of <see cref="mintypecode(string, string, char)"/>, statement for statement NumPy's: the set of
        ///     characters found in <paramref name="typeset"/>, the default for an empty set, the <c>'F'</c> + <c>'d'</c> →
        ///     <c>'D'</c> rule, then the minimum by position in <c>_typecodes_by_elsize</c>.
        /// </summary>
        /// <param name="typechars">The type characters to test.</param>
        /// <param name="typeset">The characters the answer may be; null raises only when a character reaches the test (Python's lazy set comprehension).</param>
        /// <param name="default">The answer for an empty intersection.</param>
        /// <returns>The minimum-size type character.</returns>
        /// <exception cref="TypeError"><paramref name="typeset"/> is null and a character is tested against it.</exception>
        /// <exception cref="ValueError">A character of the intersection is not in <c>_typecodes_by_elsize</c>.</exception>
        private static char MinTypeCode(IEnumerable<char> typechars, string typeset, char @default)
        {
            const string _typecodes_by_elsize = "GDFgdfQqLlIiHhBb?";

            var intersection = new HashSet<char>();
            foreach (char t in typechars)
            {
                // `t in typeset` — evaluated per character, so a None typeset fails only once there is a character.
                if (typeset == null)
                    throw new TypeError("argument of type 'NoneType' is not iterable");
                if (typeset.IndexOf(t) >= 0)
                    intersection.Add(t);
            }
            if (intersection.Count == 0)
                return @default;
            if (intersection.Contains('F') && intersection.Contains('d'))
                return 'D';

            // min(intersection, key=_typecodes_by_elsize.index): Python's min evaluates the key of EVERY element, so any
            // element missing from the string raises, even when another element would have been the minimum.
            char best = '\0';
            int bestIndex = int.MaxValue;
            foreach (char t in intersection)
            {
                int index = _typecodes_by_elsize.IndexOf(t);
                if (index < 0)
                    throw new ValueError("substring not found");
                if (index < bestIndex)
                {
                    bestIndex = index;
                    best = t;
                }
            }
            return best;
        }

        // ---- Platform-detected types (MUST be declared BEFORE _dtype_string_map since
        //      BuildDtypeStringMap() reads them, and static initializers run top-down) ----

        /// <summary>
        ///     Platform-detected C <c>long</c> type. MSVC (Windows) = 32-bit,
        ///     gcc/clang (Linux/Mac) on 64-bit = 64-bit. NumPy follows the native C convention.
        /// </summary>
        private static readonly Type _cLongType =
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? typeof(int)
                : (IntPtr.Size == 8 ? typeof(long) : typeof(int));

        private static readonly Type _cULongType =
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? typeof(uint)
                : (IntPtr.Size == 8 ? typeof(ulong) : typeof(uint));

        /// <summary>
        ///     Platform-detected pointer-sized integer (<c>intp</c>). Always matches
        ///     <see cref="IntPtr.Size"/> (8 bytes on 64-bit, 4 bytes on 32-bit).
        /// </summary>
        private static readonly Type _intpType  = IntPtr.Size == 8 ? typeof(long)  : typeof(int);
        private static readonly Type _uintpType = IntPtr.Size == 8 ? typeof(ulong) : typeof(uint);

        /// <summary>
        ///     NumPy's <c>typeDict</c>: the dtype NAME lookup (<c>"float64"</c>, <c>"intc"</c>, <c>"longlong"</c>, …) plus the
        ///     single-character codes and NumSharp's own aliases. Built to match <c>numpy.dtype(str)</c> exactly, with
        ///     NumSharp-specific adaptations:
        ///     <list type="bullet">
        ///       <item>NumPy types NumSharp doesn't implement (S/U/V/O/a) throw NotSupportedException.</item>
        ///       <item>complex64 ('F'/'c8'/'complex64') throws NotSupportedException — NumSharp only has complex128.</item>
        ///       <item>'l'/'L'/'long'/'ulong' are platform-detected to match NumPy's C-long convention:
        ///             32-bit on Windows (MSVC), 64-bit on 64-bit Linux/Mac (gcc LP64).</item>
        ///       <item>'int'/'int_'/'intp' → int64 on 64-bit (matches NumPy 2.x where int_ == intp).</item>
        ///       <item>Aliases unique to .NET (SByte/Decimal/Char) are accepted.</item>
        ///     </list>
        ///     The sized codes (<c>"i4"</c>, <c>"f8"</c>) are NOT looked up here: NumPy parses <c>kind + size</c>
        ///     arithmetically (<c>PyArray_TypestrConvert</c>) before consulting the name table, and so does
        ///     <see cref="dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/>.
        /// </summary>
        private static readonly FrozenDictionary<string, Type> _dtype_string_map = BuildDtypeStringMap();

        private static FrozenDictionary<string, Type> BuildDtypeStringMap()
        {
            var map = new Dictionary<string, Type>(StringComparer.Ordinal);

            void Add(string key, Type t) => map[key] = t;

            // ---- single-char NumPy type codes ----
            // bool
            Add("?",  typeof(bool));
            // signed int
            Add("b",  typeof(sbyte));
            Add("h",  typeof(short));
            Add("i",  typeof(int));
            Add("l",  _cLongType);       // C long: 32-bit on Windows (MSVC), 64-bit on *nix (gcc LP64)
            Add("q",  typeof(long));
            // NumPy 2.x's two intp letters (arraytypes.c.src's LETTER_TO_NUM): 'n' is np.intp (the size of size_t, the
            // letter NumPy 2.0 moved intp to) and 'p' stays pointer-sized; on every platform NumSharp runs on they are
            // the same integer.
            Add("n",  _intpType);
            Add("p",  _intpType);        // intptr
            // unsigned int
            Add("B",  typeof(byte));
            Add("H",  typeof(ushort));
            Add("I",  typeof(uint));
            Add("L",  _cULongType);      // C unsigned long: same platform rule as 'l'
            Add("Q",  typeof(ulong));
            Add("N",  _uintpType);       // np.uintp (see 'n')
            Add("P",  _uintpType);       // uintptr
            // float
            Add("e",  typeof(Half));
            Add("f",  typeof(float));
            Add("d",  typeof(double));
            Add("g",  typeof(double));   // long double collapses to double
            // complex — NumSharp only has complex128 (System.Numerics.Complex = 2 × float64).
            // complex64 ('F', 'c8', 'complex64') is NOT supported and throws NotSupportedException
            // via _unsupported_numpy_typechars / _unsupported_numpy_names below — users must explicitly opt into complex128.
            Add("D",  typeof(Complex));
            Add("G",  typeof(Complex));  // long-double complex collapses to complex128

            // ---- NumPy lowercase names ----
            Add("bool",       typeof(bool));
            Add("int8",       typeof(sbyte));
            Add("uint8",      typeof(byte));
            Add("int16",      typeof(short));
            Add("uint16",     typeof(ushort));
            Add("int32",      typeof(int));
            Add("uint32",     typeof(uint));
            Add("int64",      typeof(long));
            Add("uint64",     typeof(ulong));
            Add("float16",    typeof(Half));
            Add("half",       typeof(Half));
            Add("float32",    typeof(float));
            Add("single",     typeof(float));
            Add("float64",    typeof(double));
            Add("double",     typeof(double));
            Add("float",      typeof(double)); // NumPy: np.dtype('float') → float64
            // Note: "complex64" is NOT in the map — it's in _unsupported_numpy_names so
            // accessing it throws NotSupportedException. NumSharp only has complex128.
            Add("complex128", typeof(Complex));
            Add("cdouble",    typeof(Complex));  // np.cdouble, the C name of complex128 (in NumPy's sctypeDict)
            Add("complex",    typeof(Complex));
            Add("byte",       typeof(sbyte));   // NumPy: np.dtype('byte') → int8
            Add("ubyte",      typeof(byte));    // NumPy: np.dtype('ubyte') → uint8
            Add("short",      typeof(short));
            Add("ushort",     typeof(ushort));
            Add("intc",       typeof(int));
            Add("uintc",      typeof(uint));
            // NumPy 2.x: int_ and intp are both pointer-sized (no longer C-long).
            Add("int_",       _intpType);       // int64 on 64-bit, int32 on 32-bit
            Add("intp",       _intpType);
            Add("uintp",      _uintpType);
            Add("bool_",      typeof(bool));    // NumPy alias for bool
            // NumPy 2.x: 'int' resolves to intp (pointer-sized), not C-long.
            Add("int",        _intpType);
            Add("uint",       _uintpType);
            // NumPy 'long'/'ulong' follow the C-long platform rule (Windows=32, *nix LP64=64).
            Add("long",       _cLongType);
            Add("ulong",      _cULongType);
            // long long is always 64-bit.
            Add("longlong",   typeof(long));
            Add("ulonglong",  typeof(ulong));
            Add("longdouble",  typeof(double));  // collapses to float64
            Add("clongdouble", typeof(Complex)); // collapses to complex128

            // ---- NumSharp-only friendly aliases (unique to .NET) ----
            Add("sbyte",   typeof(sbyte));
            Add("SByte",   typeof(sbyte));
            Add("Byte",    typeof(byte));
            Add("UByte",   typeof(byte));
            Add("Int16",   typeof(short));
            Add("UInt16",  typeof(ushort));
            Add("Int32",   typeof(int));
            Add("UInt32",  typeof(uint));
            Add("Int64",   typeof(long));
            Add("UInt64",  typeof(ulong));
            Add("Half",    typeof(Half));
            Add("Single",  typeof(float));
            Add("Float",   typeof(float));
            Add("Double",  typeof(double));
            Add("Complex", typeof(Complex));
            Add("Bool",    typeof(bool));
            Add("Boolean", typeof(bool));
            Add("boolean", typeof(bool));
            Add("Char",    typeof(char));
            Add("char",    typeof(char));
            Add("decimal", typeof(decimal));
            Add("Decimal", typeof(decimal));
            Add("string",  typeof(string));
            Add("String",  typeof(string));

            return map.ToFrozenDictionary();
        }

        /// <summary>
        ///     The one-character type codes NumPy's <c>PyArray_DescrFromType</c> accepts (<c>LETTER_TO_NUM</c>, plus
        ///     <c>'c'</c> and <c>'T'</c>) whose class NumSharp has no storage for: complex64 (<c>'F'</c>), object
        ///     (<c>'O'</c>), the unsized bytes / str / void (<c>'S'</c>, <c>'U'</c>, <c>'V'</c>), the one-byte string
        ///     <c>'c'</c> (<c>S1</c>, not NumSharp's <c>char</c>) and NEP 55's <c>StringDType</c> (<c>'T'</c>). They are
        ///     valid dtypes, so the refusal says "not supported" rather than NumPy's "not understood". A byte-order prefix
        ///     does not change that, because NumPy strips the prefix before it reads a type code (<c>'&lt;S'</c> is
        ///     <c>dtype('S')</c> there).
        /// </summary>
        private static readonly FrozenSet<char> _unsupported_numpy_typechars = new HashSet<char>
        {
            'F', 'O', 'S', 'U', 'V', 'c', 'T',
        }.ToFrozenSet();

        /// <summary>
        ///     The names of NumPy's <c>typeDict</c> (<c>np.sctypeDict</c>'s string keys) whose class NumSharp has no storage
        ///     for: complex64, bytes, str, void and object. NumPy looks a name up by the WHOLE string, so only the bare name
        ///     is a valid dtype (<c>'object'</c>); with a byte-order prefix (<c>'&lt;object'</c>) it is NumPy's "not understood".
        ///     <c>'a'</c> is the deprecated one-letter alias of <c>'S'</c>, which NumPy reaches through this table (it has no
        ///     <c>LETTER_TO_NUM</c> entry), so <c>'|a'</c> is "not understood" too.
        /// </summary>
        private static readonly FrozenSet<string> _unsupported_numpy_names = new HashSet<string>(StringComparer.Ordinal)
        {
            "a", "bytes", "bytes_",            // the bytes class 'S'
            "complex64", "csingle",            // complex64 — NumSharp has no 32-bit complex
            "object", "object_",
            "str", "str_", "unicode",          // the str class 'U'
            "void",
        }.ToFrozenSet();

        // NumPy 2.0 removed these bit-suffixed aliases; the message is NumPy's verbatim.
        private static readonly FrozenSet<string> _removed_numpy_aliases = new HashSet<string>(StringComparer.Ordinal)
        {
            "int0", "uint0", "void0", "object0", "str0", "bytes0", "bool8",
        }.ToFrozenSet();

        /// <summary>
        ///     Create a data type object from a NumPy dtype string — the port of <c>descriptor.c</c>'s
        ///     <c>_convert_from_str</c> (NumPy 2.4.2), the coercion point behind <c>np.dtype('…')</c> and every
        ///     <c>dtype=</c> keyword.
        /// </summary>
        /// <param name="dtype">
        ///     Any NumPy-style dtype string: a single type code (<c>"d"</c>, <c>"?"</c>, <c>"q"</c>), a kind plus byte size
        ///     (<c>"i4"</c>, <c>"f8"</c>, <c>"c16"</c>, <c>"b1"</c>), a name (<c>"float64"</c>, <c>"intc"</c>, <c>"longlong"</c>,
        ///     <c>"complex128"</c>), a datetime typestr (<c>"M8[ns]"</c>, <c>"m8"</c>, <c>"datetime64[10ns]"</c>,
        ///     <c>"timedelta64[s/2]"</c>), optionally prefixed with a byte order (<c>"&lt;i4"</c>, <c>"&gt;f8"</c>, <c>"=u2"</c>,
        ///     <c>"|b1"</c>), or one of NumSharp's PascalCase aliases (<c>"Int32"</c>, <c>"Single"</c>). Null is NumPy's
        ///     <c>None</c>: the default float64 descriptor (<see cref="DefaultTypeDescr"/>).
        /// </param>
        /// <param name="align">
        ///     NumPy's <c>align=</c>: pad a STRUCTURED dtype like a C struct. Every NumSharp dtype is non-structured, where
        ///     NumPy ignores it too (<c>np.dtype('f8', align=True) is np.dtype('f8')</c>), so it is accepted and has no effect.
        /// </param>
        /// <param name="copy">
        ///     NumPy's <c>copy=</c>: return a fresh copy instead of the shared builtin singleton. Only a singleton is copied
        ///     (NumPy tests <c>fields is None</c>, which only the static builtin descriptors satisfy): a descriptor that is
        ///     already a fresh object (a non-native byte order, a datetime unit) comes back as itself.
        /// </param>
        /// <param name="metadata">
        ///     NumPy's <c>metadata=</c>: a dictionary to attach (see <see cref="DType.metadata"/>). The result is always a
        ///     fresh descriptor; when the parsed descriptor already carries metadata the two are merged without
        ///     overwriting the keys it already has (NumPy's <c>PyDict_Merge(..., override=0)</c>). Null means "not given".
        /// </param>
        /// <returns>
        ///     The matching descriptor: the class's canonical instance for a native builtin (so <c>np.dtype("i8")</c>
        ///     is the same object every call), a fresh instance for a non-native byte order (<c>"&gt;i4"</c> keeps
        ///     <c>byteorder == '&gt;'</c>, <c>isnative == false</c>) and for every datetime64/timedelta64 descriptor — or a
        ///     fresh copy when <paramref name="copy"/> or <paramref name="metadata"/> asks for one. An explicit host byte
        ///     order is read as native (<c>"&lt;f8"</c> is the <c>float64</c> singleton on a little-endian host), as NumPy's
        ///     parser reads it. A null string gives the float64 singleton (NumPy's <c>np.dtype(None)</c>).
        /// </returns>
        /// <exception cref="NotSupportedException">
        ///     A valid NumPy dtype NumSharp does not implement — bytes/str (S, U, a, c), void, object, complex64,
        ///     structured / sub-array / comma-string dtypes — or an invalid string, reported with NumPy's own
        ///     <c>data type 'X' not understood</c> / <c>Alias 'bool8' was removed in NumPy 2.0. …</c> texts.
        /// </exception>
        /// <exception cref="TypeError">A malformed datetime unit (<c>Invalid datetime metadata string "[5]" at position 2</c>, verbatim).</exception>
        /// <exception cref="ValueError">A datetime divisor that is not a multiple of a lower unit (verbatim).</exception>
        /// <remarks>
        ///     Grammar, in NumPy's order: a comma string (a top-level comma, a leading digit or a leading <c>()</c>) is a
        ///     structured / sub-array dtype (unsupported); the byte-order character is consumed (<c>'|'</c> reads as native,
        ///     an explicit host order folds into <c>'='</c>, and a lone byte-order character is invalid); a datetime typestr
        ///     (<c>M8</c>/<c>m8</c>/<c>datetime64</c>/<c>timedelta64</c> + metadata) is parsed by
        ///     <see cref="DatetimeMetaData.Parse"/>; a one-character code is a type code (a control character below
        ///     <c>'\x18'</c> is a TYPE NUMBER: <c>'\x0c'</c> is float64); a code whose tail is an integer is
        ///     <c>kind + size</c> (<c>PyArray_TypestrConvert</c> — so <c>"b1"</c> is bool while <c>"b"</c> is int8,
        ///     <c>"i3"</c>, <c>"f16"</c> and <c>"?1"</c> are invalid); anything else is a NAME looked up in the type
        ///     dictionary by the WHOLE string, so a byte-order prefix before a name (<c>"&lt;float64"</c>,
        ///     <c>"|bool"</c>) is "not understood" while it is accepted before a code (<c>"&lt;f8"</c>, <c>"&gt;d"</c>).
        ///     Case matters everywhere (<c>"I4"</c> is not <c>"i4"</c>), whitespace is never stripped. The keywords then
        ///     apply in <c>arraydescr_new</c>'s order: <paramref name="copy"/>, then <paramref name="metadata"/>.
        ///     <para>
        ///     Null is NumPy's <c>None</c> in every <c>np.dtype</c> overload: the float64 singleton, before the keywords
        ///     (<c>copy: true</c> gives a fresh float64, <c>metadata</c> a fresh float64 carrying it). A bare <c>null</c>
        ///     literal matches the string and <see cref="Type"/> overloads alike (CS0121), so write it typed:
        ///     <c>np.dtype((string)null)</c>, <c>np.dtype((DType)null)</c>.
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/reference/arrays.dtypes.html</para>
        /// </remarks>
        public static DType dtype(string dtype, bool align = false, bool copy = false, IReadOnlyDictionary<string, object> metadata = null)
            => ApplyDescrKeywords(dtype == null ? DefaultTypeDescr : ConvertFromStr(dtype), copy, metadata);

        /// <summary>The parse half of <see cref="dtype(string, bool, bool, IReadOnlyDictionary{string, object})"/> — NumPy's <c>_convert_from_str</c>.</summary>
        /// <param name="dtype">The dtype string.</param>
        /// <returns>The parsed descriptor (the shared singleton for a native builtin).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="NotSupportedException">An unsupported or invalid dtype string (NumPy's texts).</exception>
        /// <exception cref="TypeError">A malformed datetime unit.</exception>
        /// <exception cref="ValueError">A datetime divisor that is not a multiple of a lower unit.</exception>
        private static DType ConvertFromStr(string dtype)
        {
            if (dtype == null)
                throw new ArgumentNullException(nameof(dtype));

            string type = dtype;

            /* Empty string is invalid */
            if (type.Length == 0)
                throw NotUnderstood(dtype);

            /* check for commas present or first (or second) element a digit */
            if (CheckForCommaString(type))
                throw new NotSupportedException("NumSharp does not support structured (comma-string / field list) dtypes");
            // A parenthesis outside a comma string is no sub-array in NumPy: '(2)f8', '( )f8' and 'f8()' go on through the
            // grammar below, which names no type with a '(', and are "not understood" (a sub-array string is always a
            // comma string: '(2,)f8' and '2f8' are caught above).

            /* Process the endian character. '|' is replaced by '='*/
            char endian = DType.NPY_NATIVE;
            switch (type[0])
            {
                case '>':
                case '<':
                case '=':
                    endian = type[0];
                    type = type.Substring(1);
                    break;

                case '|':
                    endian = DType.NPY_NATIVE;
                    type = type.Substring(1);
                    break;
            }

            /* Just an endian character is invalid */
            if (type.Length == 0)
                throw NotUnderstood(dtype);

            // NumPy stores a parsed host order as '=' (`if (endian != '=' && PyArray_ISNBO(endian)) endian = '=';` at
            // `finish`, and the datetime branch only stores a non-native order). Folding it here, before any branch builds
            // a descriptor, keeps "<f8" the float64 singleton and "<M8[ns]" a '=' descriptor; only the opposite order is
            // stored explicitly. (newbyteorder, by contrast, keeps an explicit host order — DType.NormalizeByteOrder does
            // not fold it.)
            if (endian != DType.NPY_NATIVE && endian != DType.NPY_OPPBYTE)
                endian = DType.NPY_NATIVE;

            /* Check for datetime format */
            if (IsDatetimeTypestr(type))
                return ParseDatetimeTypestr(type).WithByteOrder(endian);

            // The descriptor a type NUMBER gave (a control character, see below), else null; `resolved` is the C# type a
            // type code, a kind + size or a name gave. One of them is set once the grammar has read the string.
            DType numbered = null;
            Type resolved = null;
            /* A typecode like 'd' */
            if (type.Length == 1)
            {
                char code = type[0];
                // NumPy's PyArray_DescrFromType maps the datetime LETTERS to the generic-unit descriptors
                // (np.dtype('M') is dtype('<M8'), np.dtype('m') is dtype('<m8')) — the static BUILTIN descriptor itself,
                // so np.dtype('M').isbuiltin is 1 while the typestr np.dtype('M8') builds a fresh one (isbuiltin 0); only
                // the opposite byte order copies it (the `finish` block's DESCR_REPLACE).
                if (code == 'M' || code == 'm')
                {
                    var letter = (code == 'M' ? DTypeRegistry.DateTime64 : DTypeRegistry.TimeDelta64).singleton;
                    return endian == DType.NPY_NATIVE ? letter : letter.WithByteOrder(endian);
                }
                // `check_num = (unsigned char)type[0]` is handed to PyArray_DescrFromType, which reads a value below
                // NPY_NTYPES_LEGACY (24) as a TYPE NUMBER, not a letter: a lone control character is a builtin dtype
                // ('\x00' is bool, '\x0c' float64, '\x17' float16). DType.From(NPY_TYPES) is that lookup for NumSharp's
                // classes, and refuses the numbers NumSharp has no class for (complex64, object, bytes, void). The str
                // class's number is refused here too: NumSharp's vestigial unsized 'U' slot holds no text, and every other
                // spelling of that class ('U', 'str', 'U5') is refused as unsupported as well.
                if (code == (char)NPY_TYPES.NPY_UNICODE)
                    throw new NotSupportedException($"NumPy dtype {PyLiteral.Repr(type)} (type number 19, str) is not supported by NumSharp");
                if (code < (char)NPY_TYPES.NPY_NTYPES_LEGACY)
                    numbered = DType.From((NPY_TYPES)code);
                else if (_unsupported_numpy_typechars.Contains(code))
                    throw new NotSupportedException($"NumPy dtype '{type}' is not supported by NumSharp");
                else if (!_dtype_string_map.TryGetValue(type, out resolved))
                    resolved = null; // not a type code: NumPy clears the error and tries the name table below
            }
            /* Possibly a kind + size like 'f8' but also could be 'bool' */
            else if (TryParseSizeSuffix(type, out int elsize))
            {
                char kind = type[0];
                switch (kind)
                {
                    case 'S':
                    case 'a':
                    case 'U':
                    case 'V':
                        throw new NotSupportedException($"NumPy dtype '{type}' is not supported by NumSharp");
                    default:
                        if (elsize != 0)
                        {
                            resolved = TypestrConvert(elsize, kind, type);
                            if (resolved == null)
                                throw NotUnderstood(dtype);
                        }
                        break;
                }
            }

            if (numbered == null && resolved == null)
            {
                /* Now check to see if the object is registered in typeDict */
                // NumPy looks the WHOLE original string up (`PyDict_GetItemWithError(typeDict, obj)`), not the
                // endian-stripped one, and no name starts with a byte-order character: '<float64', '|bool' and '>double'
                // are "not understood" while the bare names are dtypes. Only the removed-alias test reads the stripped
                // string, and its message quotes the whole one ("Alias '<int0' was removed ...").
                if (_unsupported_numpy_names.Contains(dtype))
                    throw new NotSupportedException($"NumPy dtype '{dtype}' is not supported by NumSharp");
                if (!_dtype_string_map.TryGetValue(dtype, out resolved))
                {
                    if (_removed_numpy_aliases.Contains(type))
                        throw new NotSupportedException($"Alias {PyLiteral.Repr(dtype)} was removed in NumPy 2.0. Use a name without a digit at the end.");
                    throw NotUnderstood(dtype);
                }
            }

            var descr = numbered ?? DType.From(resolved);
            if (endian != DType.NPY_NATIVE && endian != DType.NPY_NATBYTE && descr.byteorder != DType.NPY_IGNORE)
                return descr.WithByteOrder(endian);
            return descr;
        }

        /// <summary>The descriptor of a C# <see cref="System.Type"/> — <c>np.dtype(typeof(int))</c> (NumPy's <c>np.dtype(np.int32)</c>).</summary>
        /// <param name="type">The C# element type, or null — NumPy's <c>None</c>, the default float64 descriptor (<see cref="DefaultTypeDescr"/>).</param>
        /// <param name="align">NumPy's <c>align=</c>; no effect on a non-structured dtype (see <see cref="dtype(string, bool, bool, IReadOnlyDictionary{string, object})"/>).</param>
        /// <param name="copy">NumPy's <c>copy=</c>: a fresh copy of the class singleton instead of the singleton.</param>
        /// <param name="metadata">NumPy's <c>metadata=</c>: a dictionary to attach (always gives a fresh descriptor); null means "not given".</param>
        /// <returns>The class's canonical instance (float64 for a null type), or a fresh copy when <paramref name="copy"/> or <paramref name="metadata"/> asks for one.</returns>
        /// <exception cref="NotSupportedException"><paramref name="type"/> is not one of NumSharp's element types.</exception>
        public static DType dtype(Type type, bool align = false, bool copy = false, IReadOnlyDictionary<string, object> metadata = null)
            => ApplyDescrKeywords(type == null ? DefaultTypeDescr : DType.From(type), copy, metadata);

        /// <summary>The descriptor of an <see cref="NPTypeCode"/> — NumSharp's storage enum spelling.</summary>
        /// <param name="typecode">The storage code.</param>
        /// <param name="align">NumPy's <c>align=</c>; no effect on a non-structured dtype.</param>
        /// <param name="copy">NumPy's <c>copy=</c>: a fresh copy of the class singleton instead of the singleton.</param>
        /// <param name="metadata">NumPy's <c>metadata=</c>: a dictionary to attach (always gives a fresh descriptor); null means "not given".</param>
        /// <returns>The class's canonical instance, or a fresh copy when <paramref name="copy"/> or <paramref name="metadata"/> asks for one.</returns>
        /// <exception cref="ArgumentException"><paramref name="typecode"/> is <see cref="NPTypeCode.Empty"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="typecode"/> has no registered class.</exception>
        public static DType dtype(NPTypeCode typecode, bool align = false, bool copy = false, IReadOnlyDictionary<string, object> metadata = null)
            => ApplyDescrKeywords(DType.From(typecode), copy, metadata);

        /// <summary>A descriptor converts to itself — <c>np.dtype(np.dtype('f8'))</c> is the same object, unless a keyword asks for a copy.</summary>
        /// <param name="dtype">
        ///     The descriptor, or null — NumPy's <c>None</c>, the default float64 descriptor (<see cref="DefaultTypeDescr"/>).
        ///     A null <see cref="Nullable{T}"/> <see cref="NPTypeCode"/> reaches this overload through its implicit conversion
        ///     and reads the same way.
        /// </param>
        /// <param name="align">NumPy's <c>align=</c>; no effect on a non-structured dtype.</param>
        /// <param name="copy">NumPy's <c>copy=</c>: a fresh copy when <paramref name="dtype"/> is a class singleton (any other descriptor comes back as itself, as in NumPy).</param>
        /// <param name="metadata">NumPy's <c>metadata=</c>: merged into the descriptor's own metadata without overwriting its keys, in a fresh descriptor; null means "not given".</param>
        /// <returns><paramref name="dtype"/> itself (the float64 singleton for null), or a fresh copy when a keyword asks for one.</returns>
        public static DType dtype(DType dtype, bool align = false, bool copy = false, IReadOnlyDictionary<string, object> metadata = null)
            => ApplyDescrKeywords(dtype ?? DefaultTypeDescr, copy, metadata);

        /// <summary>
        ///     The descriptor of a NumPy scalar CLASS — <c>np.dtype(np.float64)</c> where <c>np.float64</c> is the class: a
        ///     concrete class gives its builtin descriptor (the same object <c>np.dtype("f8")</c> returns; the generic
        ///     <c>dtype('&lt;M8')</c> singleton for <c>datetime64</c>), and an ABSTRACT class raises NumPy's
        ///     <see cref="TypeError"/>, since it has no dtype.
        /// </summary>
        /// <param name="sctype">
        ///     The scalar class (<c>(NDScalarType)np.float64</c>, an element of <c>np.floating.__subclasses__()</c>,
        ///     <see cref="integer"/>, ...), or null — NumPy's <c>None</c>, the default float64 descriptor
        ///     (<see cref="DefaultTypeDescr"/>).
        /// </param>
        /// <param name="align">NumPy's <c>align=</c>; no effect on a non-structured dtype.</param>
        /// <param name="copy">NumPy's <c>copy=</c>: a fresh copy of the class singleton instead of the singleton.</param>
        /// <param name="metadata">NumPy's <c>metadata=</c>: a dictionary to attach (always gives a fresh descriptor); null means "not given".</param>
        /// <returns>The class's canonical descriptor (float64 for null), or a fresh copy when <paramref name="copy"/> or <paramref name="metadata"/> asks for one.</returns>
        /// <exception cref="TypeError">
        ///     <paramref name="sctype"/> is abstract — NumPy's texts, verbatim: <c>Converting 'np.integer' or
        ///     'np.signedinteger' to a dtype is not allowed</c>, <c>Converting `np.inexact` or `np.floating` to a dtype not
        ///     allowed</c> (also for <c>np.number</c>), <c>Converting `np.unsignedinteger` to a dtype is not allowed</c>,
        ///     <c>Converting `np.complex` to a dtype is not allowed.</c>, <c>Converting `np.character` to a dtype is not
        ///     allowed</c>, <c>Converting `np.generic` to a dtype is not allowed.</c> (also for <c>np.flexible</c>).
        /// </exception>
        public static DType dtype(NDScalarType sctype, bool align = false, bool copy = false, IReadOnlyDictionary<string, object> metadata = null)
            => ApplyDescrKeywords(sctype is null ? DefaultTypeDescr : sctype.ToDescriptor(), copy, metadata);

        /// <summary>
        ///     NumPy's <c>None</c> as a dtype — the first test of <c>descriptor.c</c>'s <c>_convert_from_any</c>
        ///     (<c>if (obj == Py_None) return PyArray_DescrFromType(NPY_DEFAULT_TYPE)</c>, <c>NPY_DEFAULT_TYPE</c> being
        ///     <c>NPY_DOUBLE</c>): the float64 class singleton, so <c>np.dtype(None)</c> is the same object as
        ///     <c>np.dtype('f8')</c>. NumSharp reads a null argument this way wherever NumPy runs that converter
        ///     (<c>PyArray_DescrConverter</c>): every <c>np.dtype</c> overload and every <c>np.result_type</c> operand
        ///     (<c>np.result_type(None, np.int8)</c> is float64, a STRONG operand).
        /// </summary>
        /// <remarks>
        ///     Not everywhere a dtype is read: NumPy's <c>PyArray_DescrConverter2</c> turns None into NO descriptor, which
        ///     <c>np.promote_types</c> and <c>np.can_cast</c> refuse with their TypeErrors (NumSharp does the same), and a
        ///     <c>DType dtype = null</c> PARAMETER is "not given", whose default each function decides
        ///     (<c>np.array</c> infers). Equality cannot read it either: <see cref="DType.Equals(object)"/> answers false
        ///     for null where NumPy's <c>np.dtype('f8') == None</c> is True, because <c>descr == null</c> is the engine's
        ///     null test.
        /// </remarks>
        internal static DType DefaultTypeDescr => DType.Double;

        /// <summary>
        ///     The keyword half of NumPy's <c>arraydescr_new</c> (<c>np.dtype(obj, align, copy, metadata)</c>), applied to the
        ///     descriptor the positional argument converted to, in NumPy's statement order: <c>copy</c> copies only a class
        ///     singleton (NumPy's <c>fields == None</c> test); <c>metadata</c> then always yields a fresh descriptor whose
        ///     metadata is the descriptor's own dict merged with the given one, existing keys winning.
        /// </summary>
        /// <param name="conv">The converted descriptor.</param>
        /// <param name="copy">The <c>copy=</c> keyword.</param>
        /// <param name="metadata">The <c>metadata=</c> keyword, or null when not given.</param>
        /// <returns><paramref name="conv"/>, or the copy the keywords ask for.</returns>
        /// <exception cref="TypeError"><paramref name="metadata"/> is given for a non-legacy class (NumPy's "cannot attach metadata to new style DType").</exception>
        private static DType ApplyDescrKeywords(DType conv, bool copy, IReadOnlyDictionary<string, object> metadata)
        {
            // NumPy: `if (copy && PyDataType_FIELDS(conv) == Py_None)` — only the static builtin descriptor has fields None,
            // so a descriptor that is already a fresh object is returned as is even with copy=True.
            if (copy && ReferenceEquals(conv, conv.Meta.singleton))
                conv = conv.Copy();

            if (metadata == null)
                return conv;
            if (!conv.Meta._legacy)
                throw new TypeError("cannot attach metadata to new style DType");

            // NumPy copies the descriptor (unless the copy step already did) and gives it its own dict: a copy of the existing
            // metadata merged with the keyword's WITHOUT overriding keys already present (PyDict_Merge override=0), else a
            // copy of the keyword's dict. Dictionary keeps insertion order when nothing is removed, which is Python's order.
            // WithMetadata always builds a fresh descriptor, which stands for both of NumPy's copies (the result is a new
            // object either way; nothing else holds the intermediate copy).
            var merged = new Dictionary<string, object>(StringComparer.Ordinal);
            if (conv.metadata != null)
                foreach (var kv in conv.metadata)
                    merged[kv.Key] = kv.Value;
            foreach (var kv in metadata)
                merged.TryAdd(kv.Key, kv.Value);
            return conv.WithMetadata(new System.Collections.ObjectModel.ReadOnlyDictionary<string, object>(merged));
        }

        /// <summary>
        ///     NumPy's <c>_convert_from_str</c> failure: <c>data type %R not understood</c>, the string quoted by Python's
        ///     <c>repr</c> — so a quote inside switches to double quotes (<c>data type "a'b" not understood</c>) and a control
        ///     character is escaped (<c>data type 'i4\n' not understood</c>). NumPy raises <c>TypeError</c>; NumSharp keeps
        ///     its long-standing <see cref="NotSupportedException"/> for every dtype string it cannot use, with NumPy's text.
        /// </summary>
        /// <param name="dtype">The WHOLE string the caller passed (byte-order prefix included).</param>
        /// <returns>The exception to throw.</returns>
        private static NotSupportedException NotUnderstood(string dtype)
            => new NotSupportedException($"data type {PyLiteral.Repr(dtype)} not understood");

        /// <summary>
        ///     NumPy's <c>_check_for_commastring</c>: a comma outside square brackets, a leading digit (after an optional
        ///     byte-order character), or a leading empty tuple <c>()</c> (also after a byte-order character) marks a
        ///     structured "comma string" dtype (<c>"i4,f8"</c>, <c>"3f8"</c>, <c>"()f8"</c>, <c>"&lt;()f8"</c>).
        /// </summary>
        /// <param name="type">The whole dtype string.</param>
        /// <returns>True when NumPy would hand the string to <c>_convert_from_commastring</c>.</returns>
        private static bool CheckForCommaString(string type)
        {
            /* Check for ints at start of string */
            int start = 0;
            if (type.Length > 1 && IsByteOrderChar(type[0]))
                start = 1;
            if (start < type.Length && type[start] >= '0' && type[start] <= '9')
                return true;
            /* Check for empty tuple */
            if ((type.Length > 1 && type[0] == '(' && type[1] == ')')
                || (type.Length > 3 && IsByteOrderChar(type[0]) && type[1] == '(' && type[2] == ')'))
                return true;
            /* Check for presence of commas outside square [] brackets */
            int sqbracket = 0;
            foreach (char c in type)
            {
                switch (c)
                {
                    case ',':
                        if (sqbracket == 0) return true;
                        break;
                    case '[':
                        ++sqbracket;
                        break;
                    case ']':
                        --sqbracket;
                        break;
                }
            }
            return false;
        }

        /// <summary>NumPy's <c>_chk_byteorder</c>: one of the four byte-order characters a dtype string may start with.</summary>
        /// <param name="c">The character.</param>
        /// <returns>True for <c>'&gt;'</c>, <c>'&lt;'</c>, <c>'|'</c> and <c>'='</c>.</returns>
        private static bool IsByteOrderChar(char c) => c == '>' || c == '<' || c == '|' || c == '=';

        /// <summary>
        ///     C's <c>isspace</c> in the "C" locale, the whitespace <c>strtol</c> skips: space, tab, line feed, vertical tab,
        ///     form feed and carriage return. NumPy parses the size of <c>"f 8"</c> with <c>strtol</c>, so only these six
        ///     count — a no-break space (U+00A0) or any other Unicode space makes the string "not understood", which
        ///     <see cref="char.IsWhiteSpace(char)"/> would wrongly accept.
        /// </summary>
        /// <param name="c">The character.</param>
        /// <returns>True for the six ASCII whitespace characters.</returns>
        private static bool IsCSpace(char c) => c == ' ' || (c >= '\t' && c <= '\r');

        /// <summary>NumPy's <c>is_datetime_typestr</c>: <c>M8…</c>, <c>m8…</c>, <c>datetime64…</c>, <c>timedelta64…</c>.</summary>
        private static bool IsDatetimeTypestr(string type)
        {
            if (type.Length < 2)
                return false;
            if (type[1] == '8' && (type[0] == 'M' || type[0] == 'm'))
                return true;
            if (type.Length < 10)
                return false;
            if (string.CompareOrdinal(type, 0, "datetime64", 0, 10) == 0)
                return true;
            if (type.Length < 11)
                return false;
            return string.CompareOrdinal(type, 0, "timedelta64", 0, 11) == 0;
        }

        /// <summary>NumPy's <c>parse_dtype_from_datetime_typestr</c>: split the root from the metadata string and parse the latter.</summary>
        private static DType ParseDatetimeTypestr(string typestr)
        {
            bool isTimedelta;
            string metastr;
            if (typestr[0] == 'm' && typestr[1] == '8')
            {
                isTimedelta = true;
                metastr = typestr.Substring(2);
            }
            else if (typestr[0] == 'M' && typestr[1] == '8')
            {
                isTimedelta = false;
                metastr = typestr.Substring(2);
            }
            else if (typestr.Length >= 11 && string.CompareOrdinal(typestr, 0, "timedelta64", 0, 11) == 0)
            {
                isTimedelta = true;
                metastr = typestr.Substring(11);
            }
            else if (typestr.Length >= 10 && string.CompareOrdinal(typestr, 0, "datetime64", 0, 10) == 0)
            {
                isTimedelta = false;
                metastr = typestr.Substring(10);
            }
            else
            {
                throw new TypeError($"Invalid datetime typestr \"{typestr}\"");
            }

            var meta = DatetimeMetaData.Parse(metastr);
            return (isTimedelta ? DTypeRegistry.TimeDelta64 : DTypeRegistry.DateTime64).Descr(meta);
        }

        /// <summary>
        ///     C <c>strtol(type + 1, &amp;typeend, 10)</c> with NumPy's acceptance test: the tail after the kind letter
        ///     must parse entirely as a non-negative integer (leading whitespace and an explicit sign are what
        ///     <c>strtol</c> accepts, so <c>"i 4"</c> and <c>"i+4"</c> read as <c>"i4"</c> in NumPy too; the whitespace is
        ///     C's <see cref="IsCSpace"/>, never a Unicode space).
        /// </summary>
        /// <param name="type">The dtype string after its byte-order character (at least two characters).</param>
        /// <param name="elsize">The parsed size, or 0 when the tail is not an integer.</param>
        /// <returns>True when the whole tail is one non-negative integer that fits an <see cref="int"/>.</returns>
        private static bool TryParseSizeSuffix(string type, out int elsize)
        {
            elsize = 0;
            int i = 1;
            while (i < type.Length && IsCSpace(type[i]))
                i++;
            bool negative = false;
            if (i < type.Length && (type[i] == '+' || type[i] == '-'))
            {
                negative = type[i] == '-';
                i++;
            }
            int digitsStart = i;
            long value = 0;
            while (i < type.Length && type[i] >= '0' && type[i] <= '9')
            {
                value = value * 10 + (type[i] - '0');
                if (value > int.MaxValue)
                    return false; // overflow → "not understood"
                i++;
            }
            if (i == digitsStart || i != type.Length)
                return false;
            if (negative && value != 0)
                return false; // "make sure it doesn't overflow or go negative"
            elsize = (int)value;
            return true;
        }

        /// <summary>
        ///     NumPy's <c>PyArray_TypestrConvert</c> restricted to NumSharp's types: kind letter + byte size → type.
        ///     Sizes with no NumSharp type return null ("not understood"); NumPy-valid-but-unsupported ones
        ///     (<c>c8</c> complex64, <c>O8</c> object) raise <see cref="NotSupportedException"/>.
        /// </summary>
        private static Type TypestrConvert(int itemsize, char gentype, string type)
        {
            switch (gentype)
            {
                case 'b':
                    return itemsize == 1 ? typeof(bool) : null;
                case 'i':
                    switch (itemsize)
                    {
                        case 1: return typeof(sbyte);
                        case 2: return typeof(short);
                        case 4: return typeof(int);
                        case 8: return typeof(long);
                        default: return null;
                    }
                case 'u':
                    switch (itemsize)
                    {
                        case 1: return typeof(byte);
                        case 2: return typeof(ushort);
                        case 4: return typeof(uint);
                        case 8: return typeof(ulong);
                        default: return null;
                    }
                case 'f':
                    switch (itemsize)
                    {
                        case 2: return typeof(Half);
                        case 4: return typeof(float);
                        case 8: return typeof(double);
                        default: return null; // no extended precision (f16 is invalid on NumPy's win-amd64 build too)
                    }
                case 'c':
                    switch (itemsize)
                    {
                        case 8: throw new NotSupportedException($"NumPy dtype '{type}' is not supported by NumSharp");
                        case 16: return typeof(Complex);
                        default: return null;
                    }
                case 'O':
                    // PyArray_TypestrConvert takes 'O4' AND 'O8' as object on every platform (the deprecation of the two
                    // spellings expired, they still parse); any other size is not a type.
                    if (itemsize == 4 || itemsize == 8)
                        throw new NotSupportedException($"NumPy dtype '{type}' is not supported by NumSharp");
                    return null;
                default:
                    return null;
            }
        }
    }

    /// <summary>
    ///     NumPy's <c>NPY_SCALARKIND</c> enum (<c>numpy/_core/include/numpy/ndarraytypes.h</c>, NumPy 2.4.2): the kind of a
    ///     scalar for the LEGACY value-based casting of NumPy 1.x, the answer of <c>PyArray_ScalarKind</c> that
    ///     <c>PyArray_CanCoerceScalar</c> reads. The same members, order and values as the header.
    /// </summary>
    /// <remarks>
    ///     NumPy 2 keeps it in its C API for compatibility only: NEP 50 promotion never asks for a scalar's kind, and neither
    ///     does NumSharp, whose promotion is the NEP 50 tables (<see cref="np.result_type(DType[])"/>). It is mirrored so
    ///     code ported from NumPy's C API compiles; nothing in NumSharp reads it. NumPy's <c>NPY_NSCALARKINDS</c> define is
    ///     <c>NPY_OBJECT_SCALAR + 1</c>.
    /// </remarks>
    public enum NPY_SCALARKIND
    {
        /// <summary>Not a scalar (an array of one or more dimensions): <c>PyArray_CanCoerceScalar</c> then requires a safe cast.</summary>
        NPY_NOSCALAR = -1,
        /// <summary>A bool scalar.</summary>
        NPY_BOOL_SCALAR,
        /// <summary>
        ///     A non-negative integer: every unsigned integer scalar, and a signed one whose sign bit is clear. It may be coerced
        ///     to signed and unsigned integer types.
        /// </summary>
        NPY_INTPOS_SCALAR,
        /// <summary>A negative signed integer, which may not be coerced to an unsigned integer type.</summary>
        NPY_INTNEG_SCALAR,
        /// <summary>A floating-point scalar.</summary>
        NPY_FLOAT_SCALAR,
        /// <summary>A complex scalar.</summary>
        NPY_COMPLEX_SCALAR,
        /// <summary>An object scalar, also NumPy's default kind for every other builtin type: it coerces only by a safe cast.</summary>
        NPY_OBJECT_SCALAR
    };

    /// <summary>
    ///     NumPy's <c>NPY_TYPES</c> type-number enum (<c>numpy/_core/include/numpy/ndarraytypes.h</c>) — the C-API
    ///     identity behind every descriptor's <see cref="DType.num"/> / <see cref="DType.type_num"/>. Byte-identical to
    ///     NumPy's header: the members and their integer values are exactly NumPy's, so <c>(int)NPY_TYPES.NPY_CDOUBLE</c>
    ///     is 15 on every platform, and <c>a.dtype.type_num</c> equals <c>np.dtype(...).num</c> on the host it runs on.
    /// </summary>
    /// <remarks>
    /// <para>
    ///     <b>Identity, not the computational switch.</b> <see cref="NPTypeCode"/> remains NumSharp's storage/kernel
    ///     discriminator (the value every backend <c>switch</c> and IL generator dispatches on); this enum is the
    ///     NumPy-facing <i>identity</i> that rides ALONGSIDE it, the exact split NumPy draws between its C
    ///     <c>type_num</c> and the descriptor it keys. It is a pure mirror — see <see cref="DType.type_num"/> for the
    ///     accessor and <see cref="DType.From(NPY_TYPES)"/> for the <c>PyArray_DescrFromType</c> reverse.
    /// </para>
    /// <para>
    ///     <b>C long is platform-dependent, and so is the descriptor a fixed-width spelling resolves to</b> (verified
    ///     against numpy 2.4.2): on <b>LP64</b> (Linux/macOS) <c>np.dtype('int32')</c> is <see cref="NPY_INT"/> (5) and
    ///     <c>np.dtype('int64')</c> is <see cref="NPY_LONG"/> (7); on <b>LLP64</b> (Windows / any 32-bit process, where C
    ///     <c>long</c> is 32-bit) <c>np.dtype('int32')</c> is <see cref="NPY_LONG"/> (7) and <c>np.dtype('int64')</c> is
    ///     <see cref="NPY_LONGLONG"/> (9). This enum carries ALL of those members regardless of platform (it is the whole
    ///     C enum); it is <see cref="DTypeRegistry"/> that maps NumSharp's single Int32/Int64 onto the platform-correct
    ///     member — the same <c>CLongIs32Bit</c> switch the <c>LongDType</c>/<c>LongLongDType</c> aliases already use.
    /// </para>
    /// <para>
    ///     <b>Members NumSharp has no storage for are still present</b> (for byte-identity and for
    ///     <see cref="DType.From(NPY_TYPES)"/> to reject them precisely): <see cref="NPY_LONGLONG"/>/<see cref="NPY_ULONGLONG"/>
    ///     on LP64, <see cref="NPY_LONGDOUBLE"/>/<see cref="NPY_CLONGDOUBLE"/> (NumSharp collapses to float64/complex128),
    ///     <see cref="NPY_CFLOAT"/> (complex64 — unsupported), <see cref="NPY_OBJECT"/>, <see cref="NPY_STRING"/> (bytes),
    ///     <see cref="NPY_VOID"/>. <see cref="NPY_CHAR"/> (24) is NumPy's <i>deprecated</i> byte-char and is NOT NumSharp's
    ///     <c>char</c> dtype (which is a NumSharp extension at <see cref="NUMSHARP_CHAR"/> in the user range).
    /// </para>
    /// </remarks>
    /// <remarks>https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types</remarks>
    public enum NPY_TYPES
    {
        /// <summary>Boolean (<see cref="System.Boolean"/> / NumSharp <see cref="NPTypeCode.Boolean"/>).</summary>
        NPY_BOOL = 0,
        /// <summary>Signed 8-bit (C <c>signed char</c>; NumSharp <see cref="NPTypeCode.SByte"/> / int8).</summary>
        NPY_BYTE = 1,
        /// <summary>Unsigned 8-bit (C <c>unsigned char</c>; NumSharp <see cref="NPTypeCode.Byte"/> / uint8).</summary>
        NPY_UBYTE = 2,
        /// <summary>Signed 16-bit (C <c>short</c>; NumSharp <see cref="NPTypeCode.Int16"/>).</summary>
        NPY_SHORT = 3,
        /// <summary>Unsigned 16-bit (C <c>unsigned short</c>; NumSharp <see cref="NPTypeCode.UInt16"/>).</summary>
        NPY_USHORT = 4,
        /// <summary>C <c>int</c> — always 32-bit; NumPy's <c>intc</c>. NumSharp's Int32 maps here ONLY on LP64 (on LLP64 it is <see cref="NPY_LONG"/>).</summary>
        NPY_INT = 5,
        /// <summary>C <c>unsigned int</c> — NumPy's <c>uintc</c>. NumSharp's UInt32 maps here only on LP64.</summary>
        NPY_UINT = 6,
        /// <summary>C <c>long</c> — 64-bit on LP64, 32-bit on LLP64. NumSharp's Int64 maps here on LP64; its Int32 maps here on LLP64.</summary>
        NPY_LONG = 7,
        /// <summary>C <c>unsigned long</c>. NumSharp's UInt64 maps here on LP64; its UInt32 on LLP64.</summary>
        NPY_ULONG = 8,
        /// <summary>C <c>long long</c> — always 64-bit. NumSharp's Int64 maps here on LLP64 (unused on LP64, where Int64 is <see cref="NPY_LONG"/>).</summary>
        NPY_LONGLONG = 9,
        /// <summary>C <c>unsigned long long</c>. NumSharp's UInt64 maps here on LLP64 (unused on LP64).</summary>
        NPY_ULONGLONG = 10,
        /// <summary>float32 (NumSharp <see cref="NPTypeCode.Single"/>) — platform-independent.</summary>
        NPY_FLOAT = 11,
        /// <summary>float64 (NumSharp <see cref="NPTypeCode.Double"/>) — platform-independent.</summary>
        NPY_DOUBLE = 12,
        /// <summary>C <c>long double</c> (extended precision). NumSharp has none — its <c>LongDoubleDType</c> collapses onto float64, so no descriptor reports this num.</summary>
        NPY_LONGDOUBLE = 13,
        /// <summary>complex64 — <b>unsupported in NumSharp</b> (single complex width is complex128); no descriptor reports this num.</summary>
        NPY_CFLOAT = 14,
        /// <summary>complex128 (NumSharp <see cref="NPTypeCode.Complex"/>) — platform-independent.</summary>
        NPY_CDOUBLE = 15,
        /// <summary>C <c>long double</c> complex — collapses onto complex128 in NumSharp; no descriptor reports this num.</summary>
        NPY_CLONGDOUBLE = 16,
        /// <summary>Python object array — <b>unsupported in NumSharp</b>.</summary>
        NPY_OBJECT = 17,
        /// <summary>Fixed-width byte string (<c>|S</c>) — <b>unsupported in NumSharp</b> (Stage D).</summary>
        NPY_STRING = 18,
        /// <summary>Fixed-width Unicode (<c>&lt;U</c>) — the vestigial <see cref="NPTypeCode.String"/> slot; no storage (Stage D).</summary>
        NPY_UNICODE = 19,
        /// <summary>Structured / void (<c>V</c>) — <b>unsupported in NumSharp</b>.</summary>
        NPY_VOID = 20,
        /// <summary>datetime64 (<c>M8</c>) — descriptor-level only until Stage C (no storage lane yet).</summary>
        NPY_DATETIME = 21,
        /// <summary>timedelta64 (<c>m8</c>) — descriptor-level only until Stage C.</summary>
        NPY_TIMEDELTA = 22,
        /// <summary>float16 (NumSharp <see cref="NPTypeCode.Half"/>) — appended out of precision order (23), platform-independent.</summary>
        NPY_HALF = 23,
        /// <summary>NumPy's <b>deprecated</b> byte-char (raises if used). NOT NumSharp's <c>char</c> dtype — that is <see cref="NUMSHARP_CHAR"/>.</summary>
        NPY_CHAR = 24,
        /// <summary>Sentinel: the count of legacy dtypes (shares value 24 with <see cref="NPY_CHAR"/>).</summary>
        NPY_NTYPES_LEGACY = 24,
        /// <summary>Sentinel: "no type" (a deliberately high, stable value so appending dtypes does not move it).</summary>
        NPY_NOTYPE = 25,
        /// <summary>The first user-defined type number — everything at or above is a registered user dtype (NumSharp's Decimal/Char live here).</summary>
        NPY_USERDEF = 256,
        /// <summary>Sentinel: the count of types excluding the 1.6 additions (shares value 21 with <see cref="NPY_DATETIME"/>).</summary>
        NPY_NTYPES_ABI_COMPATIBLE = 21,
        /// <summary>NEP 55 variable-width <c>StringDType</c> — the first non-legacy-layout dtype (reserved; Stage D).</summary>
        NPY_VSTRING = 2056,

        // ---- NumSharp extensions (no NumPy analog) — the user-defined range, exactly as NumPy numbers a user dtype ----

        /// <summary>NumSharp-only <see cref="System.Decimal"/> dtype (NumSharp <see cref="NPTypeCode.Decimal"/>) — the first user slot (aliases <see cref="NPY_USERDEF"/> = 256).</summary>
        NUMSHARP_DECIMAL = NPY_USERDEF,
        /// <summary>NumSharp-only <see cref="System.Char"/> dtype (NumSharp <see cref="NPTypeCode.Char"/>; promotes as uint16) — user slot 257.</summary>
        NUMSHARP_CHAR = NPY_USERDEF + 1,
    }

    /// <summary>
    ///     NumPy's <c>NPY_TYPECHAR</c> enum (<c>numpy/_core/include/numpy/ndarraytypes.h</c>, NumPy 2.4.2): the one-character
    ///     type codes of the builtin dtypes (<see cref="DType.@char"/>, the strings <c>np.dtype('d')</c> parses) followed by the
    ///     dtype KIND letters (<see cref="DType.kind"/>). Byte-identical to the header: the same members, in the same order,
    ///     with the same values, so <c>(char)NPY_TYPECHAR.NPY_CDOUBLELTR</c> is <c>'D'</c> exactly as in C.
    /// </summary>
    /// <remarks>
    /// <para>
    ///     <b>Several members share a value, as in C</b>, because a kind letter reuses a type character:
    ///     <see cref="NPY_GENBOOLLTR"/> is <see cref="NPY_BYTELTR"/> (<c>'b'</c>), <see cref="NPY_SIGNEDLTR"/> is
    ///     <see cref="NPY_INTLTR"/> (<c>'i'</c>), <see cref="NPY_FLOATINGLTR"/> is <see cref="NPY_FLOATLTR"/> (<c>'f'</c>) and
    ///     <see cref="NPY_COMPLEXLTR"/> is <see cref="NPY_CHARLTR"/> (<c>'c'</c>). The CHARACTER is the identity: compare
    ///     values, never names. <c>ToString()</c> of a shared value is either name (.NET leaves the choice unspecified; on
    ///     .NET 10 <c>'b'</c> prints <c>NPY_GENBOOLLTR</c>), and a <c>switch</c> can list only one member of each pair.
    /// </para>
    /// <para>
    ///     <b>Valid type characters without a member.</b> <c>'p'</c>/<c>'P'</c> (pointer-sized integers) and <c>'n'</c>/<c>'N'</c>
    ///     (NumPy 2's <c>intp</c>/<c>uintp</c>, the size of <c>size_t</c>) are accepted by <c>np.dtype</c> but have no member:
    ///     NumPy 2.0 removed <c>NPY_INTPLTR</c>/<c>NPY_UINTPLTR</c> when <c>intp</c> moved from <c>'p'</c> to <c>'n'</c>, and
    ///     defines all four explicitly in <c>arraytypes.c.src</c>. Cast the character to reach them:
    ///     <c>((NPY_TYPECHAR)'n').ToTypeCode()</c>.
    /// </para>
    /// <para>
    ///     <b>Platform-dependent meaning.</b> <see cref="NPY_LONGLTR"/>/<see cref="NPY_ULONGLTR"/> name C <c>long</c>: 32-bit on
    ///     Windows, 64-bit on Linux and macOS. So <c>np.int32</c>'s character is <c>'l'</c> on Windows and <c>'i'</c> elsewhere,
    ///     and <c>np.int64</c>'s is <c>'q'</c> on Windows and <c>'l'</c> elsewhere; use
    ///     <see cref="NPTypeCodeExtensions.ToTypeCode(NPY_TYPECHAR)"/> and <see cref="NPTypeCodeExtensions.ToTYPECHAR"/> rather
    ///     than fixed pairs.
    /// </para>
    /// <para>
    ///     A character naming a dtype NumSharp cannot store (<see cref="NPY_CFLOATLTR"/> complex64, the string and void kinds,
    ///     <see cref="NPY_OBJECTLTR"/>, <see cref="NPY_VSTRINGLTR"/>) is still a member, so code ported from NumPy compiles and
    ///     the refusal happens where NumSharp meets the dtype (<see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/>).
    ///     NumSharp's own <see cref="NPTypeCode.Decimal"/> and <see cref="NPTypeCode.Char"/> have no character here: NumPy has
    ///     no such dtypes.
    /// </para>
    /// </remarks>
    /// <remarks>https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types</remarks>
    public enum NPY_TYPECHAR
    {
        /// <summary><c>'?'</c>: bool (<see cref="NPTypeCode.Boolean"/>).</summary>
        NPY_BOOLLTR = '?',
        /// <summary><c>'b'</c>: C <c>signed char</c>, int8 (<see cref="NPTypeCode.SByte"/>). Same value as the kind letter <see cref="NPY_GENBOOLLTR"/>.</summary>
        NPY_BYTELTR = 'b',
        /// <summary><c>'B'</c>: C <c>unsigned char</c>, uint8 (<see cref="NPTypeCode.Byte"/>).</summary>
        NPY_UBYTELTR = 'B',
        /// <summary><c>'h'</c>: C <c>short</c>, int16 (<see cref="NPTypeCode.Int16"/>).</summary>
        NPY_SHORTLTR = 'h',
        /// <summary><c>'H'</c>: C <c>unsigned short</c>, uint16 (<see cref="NPTypeCode.UInt16"/>).</summary>
        NPY_USHORTLTR = 'H',
        /// <summary><c>'i'</c>: C <c>int</c> (<c>np.intc</c>), always 32-bit (<see cref="NPTypeCode.Int32"/>). Same value as the kind letter <see cref="NPY_SIGNEDLTR"/>.</summary>
        NPY_INTLTR = 'i',
        /// <summary><c>'I'</c>: C <c>unsigned int</c> (<c>np.uintc</c>), always 32-bit (<see cref="NPTypeCode.UInt32"/>).</summary>
        NPY_UINTLTR = 'I',
        /// <summary><c>'l'</c>: C <c>long</c> (<c>np.long</c>): int32 on Windows, int64 on Linux and macOS.</summary>
        NPY_LONGLTR = 'l',
        /// <summary><c>'L'</c>: C <c>unsigned long</c> (<c>np.ulong</c>): uint32 on Windows, uint64 on Linux and macOS.</summary>
        NPY_ULONGLTR = 'L',
        /// <summary><c>'q'</c>: C <c>long long</c> (<c>np.longlong</c>), always 64-bit (<see cref="NPTypeCode.Int64"/>).</summary>
        NPY_LONGLONGLTR = 'q',
        /// <summary><c>'Q'</c>: C <c>unsigned long long</c> (<c>np.ulonglong</c>), always 64-bit (<see cref="NPTypeCode.UInt64"/>).</summary>
        NPY_ULONGLONGLTR = 'Q',
        /// <summary><c>'e'</c>: float16 (<see cref="NPTypeCode.Half"/>).</summary>
        NPY_HALFLTR = 'e',
        /// <summary><c>'f'</c>: float32 (<see cref="NPTypeCode.Single"/>). Same value as the kind letter <see cref="NPY_FLOATINGLTR"/>.</summary>
        NPY_FLOATLTR = 'f',
        /// <summary><c>'d'</c>: float64 (<see cref="NPTypeCode.Double"/>).</summary>
        NPY_DOUBLELTR = 'd',
        /// <summary><c>'g'</c>: C <c>long double</c>; NumSharp stores it as float64 (<see cref="NPTypeCode.Double"/>), Windows' form on every host.</summary>
        NPY_LONGDOUBLELTR = 'g',
        /// <summary><c>'F'</c>: complex64. NumSharp has no complex64, so <c>np.dtype('F')</c> refuses it.</summary>
        NPY_CFLOATLTR = 'F',
        /// <summary><c>'D'</c>: complex128 (<see cref="NPTypeCode.Complex"/>).</summary>
        NPY_CDOUBLELTR = 'D',
        /// <summary><c>'G'</c>: C <c>long double</c> complex; NumSharp stores it as complex128 (<see cref="NPTypeCode.Complex"/>).</summary>
        NPY_CLONGDOUBLELTR = 'G',
        /// <summary><c>'O'</c>: object (Python objects). Not supported by NumSharp.</summary>
        NPY_OBJECTLTR = 'O',
        /// <summary><c>'S'</c>: fixed-width bytes (<c>np.bytes_</c>, <c>|S5</c>). Not supported by NumSharp.</summary>
        NPY_STRINGLTR = 'S',
        /// <summary><c>'a'</c>: the deprecated alias of <see cref="NPY_STRINGLTR"/>; NumPy 2 accepts it with a DeprecationWarning ("Use the 'S' alias instead").</summary>
        NPY_DEPRECATED_STRINGLTR2 = 'a',
        /// <summary><c>'U'</c>: fixed-width Unicode (<c>np.str_</c>, <c>&lt;U5</c>), the slot of <see cref="NPTypeCode.String"/>. Not supported by <c>np.dtype</c>.</summary>
        NPY_UNICODELTR = 'U',
        /// <summary><c>'V'</c>: void (raw bytes and structured records). Not supported by NumSharp.</summary>
        NPY_VOIDLTR = 'V',
        /// <summary><c>'M'</c>: datetime64. Parses to the generic-unit descriptor, which has no NumSharp storage yet.</summary>
        NPY_DATETIMELTR = 'M',
        /// <summary><c>'m'</c>: timedelta64. Parses to the generic-unit descriptor, which has no NumSharp storage yet.</summary>
        NPY_TIMEDELTALTR = 'm',
        /// <summary>
        ///     <c>'c'</c>: NumPy's one-byte string, <c>np.dtype('c')</c> = <c>S1</c> (<c>np.typecodes['Character']</c>). Not
        ///     NumSharp's <see cref="NPTypeCode.Char"/>, and not complex: the complex KIND letter <see cref="NPY_COMPLEXLTR"/>
        ///     shares the value.
        /// </summary>
        NPY_CHARLTR = 'c',

        /*
         * New non-legacy DTypes
         */

        /// <summary><c>'T'</c>: NEP 55's variable-width <c>StringDType</c>. Not supported by NumSharp.</summary>
        NPY_VSTRINGLTR = 'T',

        /*
         * Note, we removed `NPY_INTPLTR` due to changing its definition
         * to 'n', rather than 'p'.  On any typical platform this is the
         * same integer.  'n' should be used for the `np.intp` with the same
         * size as `size_t` while 'p' remains pointer sized.
         *
         * 'p', 'P', 'n', and 'N' are valid and defined explicitly
         * in `arraytypes.c.src`.
         */

        /*
         * These are for dtype 'kinds', not dtype 'typecodes'
         * as the above are for.
         */

        /// <summary>Kind letter <c>'b'</c>: booleans (<c>np.dtype(bool).kind</c>). Shares its value with the type character <see cref="NPY_BYTELTR"/>.</summary>
        NPY_GENBOOLLTR = 'b',
        /// <summary>Kind letter <c>'i'</c>: signed integers. Shares its value with the type character <see cref="NPY_INTLTR"/>.</summary>
        NPY_SIGNEDLTR = 'i',
        /// <summary>Kind letter <c>'u'</c>: unsigned integers. Not a type character: <c>np.dtype('u')</c> is "data type 'u' not understood".</summary>
        NPY_UNSIGNEDLTR = 'u',
        /// <summary>Kind letter <c>'f'</c>: floating point. Shares its value with the type character <see cref="NPY_FLOATLTR"/>.</summary>
        NPY_FLOATINGLTR = 'f',
        /// <summary>Kind letter <c>'c'</c>: complex. Shares its value with the type character <see cref="NPY_CHARLTR"/> (<c>S1</c>), so as a TYPE character it names bytes, not complex.</summary>
        NPY_COMPLEXLTR = 'c',
    };

}
