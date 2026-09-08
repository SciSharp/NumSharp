using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using NumSharp.Backends;

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
        public static char mintypecode(string typechars, string typeset = "GDFgdf", char @default = 'd')
        {
            const string _typecodes_by_elsize = "GDFgdfQqLlIiHhBb?";

            var chars = typechars.ToCharArray();
            var intersect = chars.Intersect(typeset.ToCharArray()).ToArray();
            if (intersect.Length == 0)
                return @default;
            if (intersect.Contains('F') && intersect.Contains('d'))
                return 'D';

            return intersect.OrderBy(c => _typecodes_by_elsize.IndexOf(c)).First();
        }

        /// <summary>
        ///     Return the character for the minimum-size type to which given types can be safely cast.
        ///     The returned type character must represent the smallest size dtype such that an array of the returned type can handle the data from an array of all types in typechars(or if typechars is an array, then its dtype.char).
        /// </summary>
        /// <param name="typechars"></param>
        /// <param name="typeset">The set of characters that the returned character is chosen from. The default set is ‘GDFgdf’.</param>
        /// <param name="default">The default character, this is returned if none of the characters in typechars matches a character in typeset.</param>
        /// <returns>The character representing the minimum-size type that was found.</returns>
        public static char mintypecode(char[] typechars, string typeset = "GDFgdf", char @default = 'd')
        {
            const string _typecodes_by_elsize = "GDFgdfQqLlIiHhBb?";

            var chars = typechars;
            var intersect = chars.Intersect(typeset.ToCharArray()).ToArray();
            if (intersect.Length == 0)
                return @default;
            if (intersect.Contains('F') && intersect.Contains('d'))
                return 'D';

            return intersect.OrderBy(c => _typecodes_by_elsize.IndexOf(c)).First();
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
        ///     <see cref="dtype(string)"/>.
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
            Add("p",  _intpType);        // intptr
            // unsigned int
            Add("B",  typeof(byte));
            Add("H",  typeof(ushort));
            Add("I",  typeof(uint));
            Add("L",  _cULongType);      // C unsigned long: same platform rule as 'l'
            Add("Q",  typeof(ulong));
            Add("P",  _uintpType);       // uintptr
            // float
            Add("e",  typeof(Half));
            Add("f",  typeof(float));
            Add("d",  typeof(double));
            Add("g",  typeof(double));   // long double collapses to double
            // complex — NumSharp only has complex128 (System.Numerics.Complex = 2 × float64).
            // complex64 ('F', 'c8', 'complex64') is NOT supported and throws NotSupportedException
            // via _unsupported_numpy_codes below — users must explicitly opt into complex128.
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
            // Note: "complex64" is NOT in the map — it's in _unsupported_numpy_codes so
            // accessing it throws NotSupportedException. NumSharp only has complex128.
            Add("complex128", typeof(Complex));
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

        // NumPy dtype codes that are valid in NumPy but NumSharp does not implement.
        // Route to clear NotSupportedException instead of silent misbehavior.
        // Note: 'F', 'c8', 'complex64' — NumSharp refuses these since it only has complex128.
        // Users should explicitly use 'complex128' / 'D' / 'c16' / 'complex'.
        private static readonly FrozenSet<string> _unsupported_numpy_codes = new HashSet<string>(StringComparer.Ordinal)
        {
            "S", "U", "V", "O", "a", "c", // c = S1 (1-byte string), NOT complex
            "F", "c8", "complex64",       // complex64 — NumSharp has no 32-bit complex
            "object", "object_", "bytes_", "str_", "str", "void", "unicode", "bytes",
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
        ///     <c>"|b1"</c>), or one of NumSharp's PascalCase aliases (<c>"Int32"</c>, <c>"Single"</c>).
        /// </param>
        /// <returns>
        ///     The matching descriptor: the class's canonical instance for a native builtin (so <c>np.dtype("i8")</c>
        ///     is the same object every call), a fresh instance for a non-native byte order (<c>"&gt;i4"</c> keeps
        ///     <c>byteorder == '&gt;'</c>, <c>isnative == false</c>) and for every datetime64/timedelta64 descriptor.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="NotSupportedException">
        ///     A valid NumPy dtype NumSharp does not implement — bytes/str (S, U, a, c), void, object, complex64,
        ///     structured / sub-array / comma-string dtypes — or an invalid string, reported with NumPy's own
        ///     <c>data type 'X' not understood</c> / <c>Alias 'bool8' was removed in NumPy 2.0. …</c> texts.
        /// </exception>
        /// <exception cref="TypeError">A malformed datetime unit (<c>Invalid datetime metadata string "[5]" at position 2</c>, verbatim).</exception>
        /// <exception cref="ValueError">A datetime divisor that is not a multiple of a lower unit (verbatim).</exception>
        /// <remarks>
        ///     Grammar, in NumPy's order: a comma-string / parenthesised sub-array is a structured dtype (unsupported);
        ///     the byte-order character is consumed (<c>'|'</c> reads as native, and a lone byte-order character is
        ///     invalid); a datetime typestr (<c>M8</c>/<c>m8</c>/<c>datetime64</c>/<c>timedelta64</c> + metadata) is
        ///     parsed by <see cref="DatetimeMetaData.Parse"/>; a one-character code is a type code; a code whose tail is
        ///     an integer is <c>kind + size</c> (<c>PyArray_TypestrConvert</c> — so <c>"b1"</c> is bool while <c>"b"</c> is
        ///     int8, <c>"i3"</c>, <c>"f16"</c> and <c>"?1"</c> are invalid); anything else is a NAME looked up in the
        ///     type dictionary. Case matters everywhere (<c>"I4"</c> is not <c>"i4"</c>), whitespace is never stripped.
        ///     <para>https://numpy.org/doc/stable/reference/arrays.dtypes.html</para>
        /// </remarks>
        public static DType dtype(string dtype)
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
            if (type.Contains('('))
                throw new NotSupportedException("NumSharp does not support custom nested array dtypes");

            /* Process the endian character. '|' is replaced by '='*/
            char endian = DType.NativeByteOrder;
            switch (type[0])
            {
                case '>':
                case '<':
                case '=':
                    endian = type[0];
                    type = type.Substring(1);
                    break;

                case '|':
                    endian = DType.NativeByteOrder;
                    type = type.Substring(1);
                    break;
            }

            /* Just an endian character is invalid */
            if (type.Length == 0)
                throw NotUnderstood(dtype);

            /* Check for datetime format */
            if (IsDatetimeTypestr(type))
                return ParseDatetimeTypestr(type).WithByteOrder(endian);

            Type resolved = null;
            /* A typecode like 'd' */
            if (type.Length == 1)
            {
                // NumPy's PyArray_DescrFromType maps the datetime LETTERS to the generic-unit descriptors
                // (np.dtype('M') is dtype('<M8'), np.dtype('m') is dtype('<m8')).
                if (type[0] == 'M')
                    return DTypeRegistry.DateTime64.DefaultDescr().WithByteOrder(endian);
                if (type[0] == 'm')
                    return DTypeRegistry.TimeDelta64.DefaultDescr().WithByteOrder(endian);
                if (!_dtype_string_map.TryGetValue(type, out resolved))
                    resolved = null;
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

            if (resolved == null)
            {
                /* Now check to see if the object is registered in typeDict */
                if (_unsupported_numpy_codes.Contains(type))
                    throw new NotSupportedException($"NumPy dtype '{type}' is not supported by NumSharp");
                if (!_dtype_string_map.TryGetValue(type, out resolved))
                {
                    if (_removed_numpy_aliases.Contains(type))
                        throw new NotSupportedException($"Alias '{type}' was removed in NumPy 2.0. Use a name without a digit at the end.");
                    throw NotUnderstood(dtype);
                }
            }

            var descr = DType.From(resolved);
            if (endian != DType.NativeByteOrder && endian != DType.HostByteOrder && descr.byteorder != DType.NotApplicableByteOrder)
                return descr.WithByteOrder(endian);
            return descr;
        }

        /// <summary>The descriptor of a C# <see cref="System.Type"/> — <c>np.dtype(typeof(int))</c> (NumPy's <c>np.dtype(np.int32)</c>).</summary>
        public static DType dtype(Type type) => DType.From(type ?? throw new ArgumentNullException(nameof(type)));

        /// <summary>The descriptor of an <see cref="NPTypeCode"/> — NumSharp's storage enum spelling.</summary>
        public static DType dtype(NPTypeCode typecode) => DType.From(typecode);

        /// <summary>A descriptor converts to itself — <c>np.dtype(np.dtype('f8'))</c>.</summary>
        public static DType dtype(DType dtype) => dtype ?? throw new ArgumentNullException(nameof(dtype));

        private static NotSupportedException NotUnderstood(string dtype)
            => new NotSupportedException($"data type '{dtype}' not understood");

        /// <summary>
        ///     NumPy's <c>_check_for_commastring</c>: a comma anywhere, or a leading digit (after an optional byte-order
        ///     character), marks a structured "comma string" dtype (<c>"i4,f8"</c>, <c>"3f8"</c>).
        /// </summary>
        private static bool CheckForCommaString(string type)
        {
            /* Check for ints at start of string */
            int start = 0;
            if (type.Length > 1 && (type[0] == '>' || type[0] == '<' || type[0] == '|' || type[0] == '='))
                start = 1;
            if (start < type.Length && type[start] >= '0' && type[start] <= '9')
                return true;
            /* Check for empty tuple */
            if (type.Length > 1 && type[0] == '(' && type[1] == ')')
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
        ///     <c>strtol</c> accepts, so <c>"i 4"</c> and <c>"i+4"</c> read as <c>"i4"</c> in NumPy too).
        /// </summary>
        private static bool TryParseSizeSuffix(string type, out int elsize)
        {
            elsize = 0;
            int i = 1;
            while (i < type.Length && char.IsWhiteSpace(type[i]))
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
                    if (itemsize == IntPtr.Size)
                        throw new NotSupportedException($"NumPy dtype '{type}' is not supported by NumSharp");
                    return null;
                default:
                    return null;
            }
        }
    }

    public enum NPY_SCALARKIND
    {
        NPY_NOSCALAR = -1,
        NPY_BOOL_SCALAR,
        NPY_INTPOS_SCALAR,
        NPY_INTNEG_SCALAR,
        NPY_FLOAT_SCALAR,
        NPY_COMPLEX_SCALAR,
        NPY_OBJECT_SCALAR
    };

    /// <summary>
    ///     https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types
    /// </summary>
    public enum NPY_TYPECHAR
    {
        NPY_BOOLLTR = '?',
        NPY_BYTELTR = 'b',
        NPY_UBYTELTR = 'B',
        NPY_SHORTLTR = 'h',
        NPY_USHORTLTR = 'H',
        NPY_INTLTR = 'i',
        NPY_UINTLTR = 'I',
        NPY_LONGLTR = 'l',
        NPY_ULONGLTR = 'L',
        NPY_LONGLONGLTR = 'q',
        NPY_ULONGLONGLTR = 'Q',
        NPY_HALFLTR = 'e',
        NPY_FLOATLTR = 'f',
        NPY_DOUBLELTR = 'd',
        NPY_LONGDOUBLELTR = 'g',
        NPY_CFLOATLTR = 'F',
        NPY_CDOUBLELTR = 'D',
        NPY_CLONGDOUBLELTR = 'G',
        NPY_OBJECTLTR = 'O',
        NPY_STRINGLTR = 'S',
        NPY_STRINGLTR2 = 'a',
        NPY_UNICODELTR = 'U',
        NPY_VOIDLTR = 'V',
        NPY_DATETIMELTR = 'M',
        NPY_TIMEDELTALTR = 'm',
        NPY_CHARLTR = 'c',

        /*
         * No Descriptor, just a define -- this let's
         * Python users specify an array of integers
         * large enough to hold a pointer on the
         * platform
         */
        NPY_INTPLTR = 'p',
        NPY_UINTPLTR = 'P',

        /*
         * These are for dtype 'kinds', not dtype 'typecodes'
         * as the above are for.
         */
        NPY_GENBOOLLTR = 'b',
        NPY_SIGNEDLTR = 'i',
        NPY_UNSIGNEDLTR = 'u',
        NPY_FLOATINGLTR = 'f',
        NPY_COMPLEXLTR = 'c'
    };

}
