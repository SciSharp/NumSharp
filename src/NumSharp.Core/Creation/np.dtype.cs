using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using NumSharp.Backends;

namespace NumSharp
{
    /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype</remarks>
    public class DType
    {
        protected internal static readonly FrozenDictionary<NPTypeCode, char> _kind_list_map = new Dictionary<NPTypeCode, char>()
        {
            {NPTypeCode.Complex, 'c'},
            {NPTypeCode.Boolean, '?'},
            {NPTypeCode.SByte, 'i'},
            {NPTypeCode.Byte, 'u'},
            {NPTypeCode.Int16, 'i'},
            {NPTypeCode.UInt16, 'u'},
            {NPTypeCode.Int32, 'i'},
            {NPTypeCode.UInt32, 'u'},
            {NPTypeCode.Int64, 'i'},
            {NPTypeCode.UInt64, 'u'},
            {NPTypeCode.Char, 'S'},
            {NPTypeCode.Half, 'f'},
            {NPTypeCode.Double, 'f'},
            {NPTypeCode.Single, 'f'},
            {NPTypeCode.Decimal, 'f'},
            {NPTypeCode.String, 'S'},
        }.ToFrozenDictionary();

        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
        public DType(Type type)
        {
            this.type = type ?? throw new ArgumentNullException(nameof(type));
            typecode = type.GetTypeCode();
            name = type.Name;
            byteorder = '=';
            itemsize = typecode.SizeOf();
            TYPECHAR = typecode.ToTYPECHAR();
            kind = _kind_list_map[typecode];
        }

        /// <summary>
        ///     A character indicating the byte-order of this data-type object.<br></br>
        ///     One of:<br></br>
        ///     
        ///     '='	native<br></br>
        ///     '\&lt;'	little-endian<br></br>
        ///     '&gt;'	big-endian<br></br>
        ///     '|'	not applicable<br></br>
        /// </summary>
        public char byteorder;

        /// <summary>
        ///     The size of the dtype in bytes.
        /// </summary>
        public int itemsize;

        /// <summary>
        ///     The name of this dtype.
        /// </summary>
        public string name;

        /// <summary>
        ///     The actual type this dtype represents.
        /// </summary>
        public Type type;

        /// <summary>
        ///     The NumSharp type code.
        /// </summary>
        public NPTypeCode typecode;

        /// <summary>
        ///     A unique character code for each of the 21 different built-in types.
        /// </summary>
        internal NPY_TYPECHAR TYPECHAR;

        /// <summary>
        ///     A character code (one of ‘biufcmMOSUV’) identifying the general kind of data.<br></br><br></br>
        ///     b boolean<br></br>
        ///     i signed integer<br></br>
        ///     u   unsigned integer<br></br>
        ///     f floating-point<br></br>
        ///     c   complex floating-point<br></br>
        ///     m   timedelta<br></br>
        ///     M   datetime<br></br>
        ///     O   object<br></br>
        ///     S(byte-)string<br></br>
        ///     U   Unicode<br></br>
        ///     V   void<br></br>
        /// </summary>
        public char kind;

        /// <summary>
        /// A unique character code for each of the 21 different built-in types.
        /// </summary>
        public char @char => (char)TYPECHAR;

        /// <summary>
        ///     Return a new dtype with a different byte order.
        ///     Changes are also made in all fields and sub-arrays of the data type.
        /// </summary>
        /// <param name="new_order">
        ///     Byte order to force; a value from the byte order specifications below.<br></br> The default value (‘S’) results in swapping the current byte order.<br></br> new_order codes can be any of:<br></br>
        ///     ‘S’ - swap dtype from current to opposite endian<br></br>
        ///     '='	- native order<br></br>
        ///     '\&lt;'	- little-endian<br></br>
        ///     '&gt;' - big-endian<br></br>
        ///     '|'	- ignore(no change to byte order)<br></br>
        ///     The code does a case-insensitive check on the first letter of new_order for these alternatives.<br></br>For example, any of ‘>’ or ‘B’ or ‘b’ or ‘brian’ are valid to specify big-endian.
        /// </param>
        /// <returns>New dtype object with the given change to the byte order.</returns>
        public DType newbyteorder(char new_order = 'S')
        {
            throw new NotSupportedException();
        }
    }

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
        ///     Full NumPy 2.x dtype string → Type lookup. Built to match
        ///     <c>numpy.dtype(str)</c> exactly, with NumSharp-specific adaptations:
        ///     <list type="bullet">
        ///       <item>NumPy types NumSharp doesn't implement (S/U/M/m/O/V/a) throw NotSupportedException.</item>
        ///       <item>complex64 ('F'/'c8'/'complex64') throws NotSupportedException — NumSharp only has complex128.</item>
        ///       <item>'l'/'L'/'long'/'ulong' are platform-detected to match NumPy's C-long convention:
        ///             32-bit on Windows (MSVC), 64-bit on 64-bit Linux/Mac (gcc LP64).</item>
        ///       <item>'int'/'int_'/'intp' → int64 on 64-bit (matches NumPy 2.x where int_ == intp).</item>
        ///       <item>Aliases unique to .NET (SByte/Decimal/Char) are accepted.</item>
        ///     </list>
        /// </summary>
        private static readonly FrozenDictionary<string, Type> _dtype_string_map = BuildDtypeStringMap();

        private static FrozenDictionary<string, Type> BuildDtypeStringMap()
        {
            var map = new Dictionary<string, Type>(StringComparer.Ordinal);

            void Add(string key, Type t) => map[key] = t;

            // ---- single-char NumPy type codes (sized OR unsized forms) ----
            // bool
            Add("?",  typeof(bool));     Add("b1", typeof(bool));
            // signed int
            Add("b",  typeof(sbyte));    Add("i1", typeof(sbyte));
            Add("h",  typeof(short));    Add("i2", typeof(short));
            Add("i",  typeof(int));      Add("i4", typeof(int));
            Add("l",  _cLongType);       // C long: 32-bit on Windows (MSVC), 64-bit on *nix (gcc LP64)
            Add("q",  typeof(long));     Add("i8", typeof(long));
            Add("p",  _intpType);        // intptr
            // unsigned int
            Add("B",  typeof(byte));     Add("u1", typeof(byte));
            Add("H",  typeof(ushort));   Add("u2", typeof(ushort));
            Add("I",  typeof(uint));     Add("u4", typeof(uint));
            Add("L",  _cULongType);      // C unsigned long: same platform rule as 'l'
            Add("Q",  typeof(ulong));    Add("u8", typeof(ulong));
            Add("P",  _uintpType);       // uintptr
            // float
            Add("e",  typeof(Half));     Add("f2", typeof(Half));
            Add("f",  typeof(float));    Add("f4", typeof(float));
            Add("d",  typeof(double));   Add("f8", typeof(double));
            Add("g",  typeof(double));   // long double collapses to double
            // complex — NumSharp only has complex128 (System.Numerics.Complex = 2 × float64).
            // complex64 ('F', 'c8', 'complex64') is NOT supported and throws NotSupportedException
            // via _unsupported_numpy_codes below — users must explicitly opt into complex128.
            Add("D",  typeof(Complex));  Add("c16", typeof(Complex));
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
            "S", "U", "V", "O", "M", "m", "a", "c", // c = S1 (1-byte string), NOT complex
            "F", "c8", "complex64",                 // complex64 — NumSharp has no 32-bit complex
            "datetime64", "timedelta64", "object", "object_", "bytes_", "str_", "str", "void", "unicode",
        }.ToFrozenSet();

        /// <summary>
        ///     Parse a string into a <see cref="DType"/>. 1:1 NumPy 2.x parity (with adaptations
        ///     documented in <see cref="_dtype_string_map"/>).
        /// </summary>
        /// <param name="dtype">Any NumPy-style dtype string (e.g. "int8", "f4", "&lt;i2", "complex128").</param>
        /// <returns>Matching <see cref="DType"/>.</returns>
        /// <exception cref="NotSupportedException">
        ///     Thrown for valid-NumPy types NumSharp doesn't implement (S, U, M, m, O, V, a, c=S1),
        ///     or for syntactically invalid strings (e.g. "f16", "b4", "xyz").
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/arrays.dtypes.html</remarks>
        public static DType dtype(string dtype)
        {
            if (dtype == null)
                throw new ArgumentNullException(nameof(dtype));

            if (dtype.Contains("("))
                throw new NotSupportedException("NumSharp does not support custom nested array dtypes");

            // NumPy accepts byte-order prefixes (<, >, =, |). Strip before lookup — NumSharp is
            // host-endian only.
            string key = dtype;
            if (key.Length > 1 && (key[0] == '<' || key[0] == '>' || key[0] == '=' || key[0] == '|'))
                key = key.Substring(1);

            // Prefer the lookup first so c8/c16 resolve to Complex before any "unsupported" check
            // intercepts 'c' as S1.
            if (_dtype_string_map.TryGetValue(key, out Type t))
                return new DType(t);

            // Reject valid-NumPy codes NumSharp doesn't implement.
            if (_unsupported_numpy_codes.Contains(key))
                throw new NotSupportedException($"NumPy dtype '{key}' is not supported by NumSharp");

            // Bytestring/unicode/void/datetime with size suffix: "S10", "U32", "V16", "a5", "M8", "m8".
            // (c is excluded because c8/c16 are complex sizes — already caught by the map above.)
            if (key.Length > 1 && char.IsDigit(key[1]))
            {
                char first = key[0];
                if (first == 'S' || first == 'U' || first == 'V' || first == 'a' ||
                    first == 'M' || first == 'm')
                    throw new NotSupportedException($"NumPy dtype '{key}' is not supported by NumSharp");
            }

            // Fall back to C# Enum name (handles "Int32", "Complex", etc. — redundant with aliases
            // above but belt-and-suspenders for case-insensitive eng names).
            if (Enum.TryParse<NPTypeCode>(key, out var code) && code != NPTypeCode.Empty)
            {
                var resolved = code.AsType();
                if (resolved != null)
                    return new DType(resolved);
            }

            throw new NotSupportedException($"NumSharp cannot parse dtype '{dtype}' — not a recognized NumPy type string");
        }
    }

    internal enum NPY_SCALARKIND
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
    internal enum NPY_TYPECHAR
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
