using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using NumSharp.Backends;

namespace NumSharp
{
    /// <summary>
    ///     NumSharp's data-type descriptor — the reference type that stands in for NumPy's <c>numpy.dtype</c>,
    ///     and the <b>single dtype spelling</b> every dtype-taking API in NumSharp accepts.
    /// </summary>
    /// <remarks>
    /// <para>
    ///     <b>Why it exists.</b> NumPy funnels every <c>dtype=</c> argument through one coercion point,
    ///     <c>numpy.dtype(...)</c>, so a Python type (<c>float</c>), a NumPy scalar type (<c>np.float32</c>),
    ///     a <c>np.dtype</c> instance and a dtype <b>string</b> (<c>'float32'</c>, <c>'f4'</c>, <c>'&lt;f8'</c>)
    ///     are ALL valid there. NumSharp historically had three separate spellings — a C# <see cref="System.Type"/>,
    ///     an <see cref="NPTypeCode"/> enum, and this descriptor from <see cref="np.dtype(string)"/> — which forced
    ///     two-or-three overloads per function. <see cref="DType"/> collapses them: it is the one type every
    ///     dtype-taking overload accepts, and each of the other spellings converts to it IMPLICITLY, so a caller
    ///     writes whichever is convenient and it binds the single overload — mirroring NumPy's one <c>dtype=</c>.
    /// </para>
    /// <para>
    ///     <b>The four spellings, all implicit.</b> The following are equivalent and all bind the one
    ///     <c>DType</c> overload (e.g. <c>np.sqrt(x, dtype: …)</c>):
    ///     <list type="bullet">
    ///       <item><description><b><see cref="System.Type"/></b> — <c>typeof(float)</c> (NumPy's Python/NumPy scalar type).</description></item>
    ///       <item><description><b><see cref="NPTypeCode"/></b> — <c>NPTypeCode.Single</c> (NumSharp's compact enum; no NumPy counterpart).</description></item>
    ///       <item><description><b>NumPy dtype string</b> — <c>"float32"</c> / <c>"f4"</c> / <c>"&lt;f8"</c> (NumPy's <c>dtype='float32'</c> — see the casing rules below).</description></item>
    ///       <item><description><b><see cref="DType"/> itself</b> — <c>DType.Single</c>, <c>np.dtype("f4")</c>, <c>DType.From(...)</c>.</description></item>
    ///     </list>
    ///     A <see cref="DType"/> also converts back to <see cref="System.Type"/> and <see cref="NPTypeCode"/>
    ///     implicitly, so it drops straight into code expecting either.
    /// </para>
    /// <para>
    ///     <b>None / infer is <see langword="null"/>.</b> A <see langword="null"/> <see cref="DType"/> is the
    ///     "none/infer" state — the analog of NumPy's <c>dtype=None</c>. This is precisely why <see cref="DType"/>
    ///     is a <b>class, not a struct</b>: a nullable <c>DType dtype = null</c> parameter is a drop-in replacement
    ///     for the old <c>Type dtype = null</c> parameter, so the engine's existing null idioms keep working
    ///     verbatim — <c>dtype?.GetTypeCode()</c> yields <c>NPTypeCode?</c> (null when none) and <c>dtype == null</c>
    ///     tests the none state. Converting a <see langword="null"/>/<see cref="NPTypeCode.Empty"/> spelling yields
    ///     a <see langword="null"/> <see cref="DType"/> (never a throwing conversion); the explicit
    ///     <see cref="DType(System.Type)"/> / <see cref="DType(NPTypeCode)"/> constructors, by contrast, reject
    ///     null/Empty (use a <see langword="null"/> <see cref="DType"/> for none).
    /// </para>
    /// <para>
    ///     <b>NumPy string casing (source of truth: NumPy 2.4.2).</b> Strings are parsed by
    ///     <see cref="np.dtype(string)"/> with NumPy's exact, <b>case-sensitive</b> spelling — the single-character
    ///     codes differ by case:
    ///     <list type="table">
    ///       <listheader><term>code</term><description>type</description></listheader>
    ///       <item><term>?</term><description>bool</description></item>
    ///       <item><term>b / B</term><description>int8 / uint8</description></item>
    ///       <item><term>h / H</term><description>int16 / uint16</description></item>
    ///       <item><term>i / I</term><description>int32 / uint32</description></item>
    ///       <item><term>q / Q</term><description>int64 / uint64</description></item>
    ///       <item><term>e / f / d</term><description>float16 / float32 / float64</description></item>
    ///       <item><term>D</term><description>complex128</description></item>
    ///     </list>
    ///     Sized forms (<c>"i4"</c>, <c>"f8"</c>, <c>"c16"</c>), lowercase names (<c>"float64"</c>, <c>"int32"</c>,
    ///     <c>"complex128"</c>) and byte-order prefixes (<c>"&lt;f8"</c>, <c>"&gt;i4"</c>, <c>"=u2"</c>, <c>"|b1"</c>)
    ///     are all accepted; the prefix is stripped because NumSharp is host-endian only.
    /// </para>
    /// <para>
    ///     <b>Deliberately narrowed to NumSharp's capability</b> (the same narrowing as <see cref="np.dtype(string)"/>):
    ///     only NumSharp's <b>15 element types</b> are representable. Two consequences differ from NumPy and are
    ///     intentional — (1) <b>complex64</b> (<c>'F'</c>, <c>"c8"</c>, <c>"complex64"</c>) is rejected with
    ///     <see cref="NotSupportedException"/> (NumSharp has only complex128), as are structured / datetime /
    ///     void / object / (byte)string dtypes; (2) NumSharp additionally accepts a <b>superset</b> of NumPy's
    ///     casing — the C# / <see cref="NPTypeCode"/> PascalCase names (<c>"Int32"</c>, <c>"Single"</c>,
    ///     <c>"Boolean"</c>, <c>"SByte"</c>, <c>"Decimal"</c>, <c>"Char"</c>) that NumPy 2.4.2 rejects — as a
    ///     convenience for C# callers. Byte order is always native (<c>'='</c>); <see cref="newbyteorder"/> throws.
    /// </para>
    /// <para>
    ///     <b>Descriptor surface</b> (mirrors <c>numpy.dtype</c>): <see cref="type"/> (the C# <see cref="System.Type"/>),
    ///     <see cref="typecode"/> (<see cref="NPTypeCode"/>), <see cref="name"/>, <see cref="kind"/>
    ///     ('b'/'i'/'u'/'f'/'c'/'S'…), <see cref="char"/> (the type char code), <see cref="itemsize"/> (bytes) and
    ///     <see cref="byteorder"/>. Equality is by <see cref="typecode"/> and null-safe.
    /// </para>
    /// <example>
    /// <code>
    /// np.sqrt(x, dtype: typeof(float));      // Type
    /// np.sqrt(x, dtype: NPTypeCode.Single);  // NPTypeCode enum
    /// np.sqrt(x, dtype: "float32");          // NumPy string  (== NumPy's dtype='float32')
    /// np.sqrt(x, dtype: DType.Single);       // DType spelling
    /// np.sqrt(x);                            // dtype omitted  == None / infer
    ///
    /// DType d  = np.dtype("&lt;f8");           // full descriptor: d.type==typeof(double), d.kind=='f', d.itemsize==8
    /// Type t   = DType.Double;               // implicit DType -&gt; Type
    /// NPTypeCode c = (DType)"int32";         // implicit string -&gt; DType -&gt; NPTypeCode
    /// bool same = DType.Single == (DType)"f4"; // true — value equality by typecode
    /// </code>
    /// </example>
    /// </remarks>
    /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype</remarks>
    public class DType : IEquatable<DType>
    {
        internal static readonly FrozenDictionary<NPTypeCode, char> _kind_list_map = new Dictionary<NPTypeCode, char>()
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

        /// <summary>
        ///     Builds a descriptor for a C# <see cref="System.Type"/>. Identical to <see cref="From(System.Type)"/>.
        ///     For the "none/infer" state use a <see langword="null"/> <see cref="DType"/> (this ctor throws on a
        ///     <see langword="null"/> type; the implicit <c>Type</c>→<c>DType</c> conversion yields <see langword="null"/>).
        /// </summary>
        public DType(Type type)
        {
            this.type = type ?? throw new ArgumentNullException(nameof(type));
            this.typecode = type.GetTypeCode();
            this.name = NumpyName(this.typecode);
            this.byteorder = '=';
            this.itemsize = this.typecode.SizeOf();
            this.TYPECHAR = this.typecode.ToTYPECHAR();
            this.kind = _kind_list_map[this.typecode];
        }

        /// <summary>
        ///     Builds a descriptor for an <see cref="NPTypeCode"/>. Identical to <see cref="From(NPTypeCode)"/>.
        ///     For the "none/infer" state use a <see langword="null"/> <see cref="DType"/> (this ctor throws on
        ///     <see cref="NPTypeCode.Empty"/>; the implicit <c>NPTypeCode</c>→<c>DType</c> conversion yields <see langword="null"/>).
        /// </summary>
        public DType(NPTypeCode typecode)
        {
            if (typecode == NPTypeCode.Empty)
                throw new ArgumentException("NPTypeCode.Empty has no dtype; use a null DType for the none/infer state.", nameof(typecode));

            this.typecode = typecode;
            this.type = typecode.AsType();
            this.name = NumpyName(typecode);
            this.byteorder = '=';
            this.itemsize = typecode.SizeOf();
            this.TYPECHAR = typecode.ToTYPECHAR();
            this.kind = _kind_list_map[typecode];
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
        public readonly char byteorder;

        /// <summary>
        ///     The size of the dtype in bytes.
        /// </summary>
        public readonly int itemsize;

        /// <summary>
        ///     The name of this dtype.
        /// </summary>
        public readonly string name;

        /// <summary>
        ///     NumPy's <c>dtype.name</c> spelling (<c>"float32"</c>, <c>"int64"</c>, <c>"bool"</c>, <c>"complex128"</c>) —
        ///     the CLR <see cref="Type.Name"/> (<c>"Single"</c>) is not a NumPy name and does not round-trip through
        ///     <see cref="np.dtype(string)"/>. The three NumSharp-only dtypes have no NumPy analog and get lowercase
        ///     names of their own rather than the nearest NumPy stand-in (<c>uint16</c>/<c>float64</c>), which would misreport them.
        /// </summary>
        private static string NumpyName(NPTypeCode typecode) => typecode switch
        {
            NPTypeCode.Char => "char",
            NPTypeCode.Decimal => "decimal",
            NPTypeCode.String => "str",
            _ => typecode.AsNumpyDtypeName(),
        };

        /// <summary>
        ///     The actual type this dtype represents.
        /// </summary>
        public readonly Type type;

        /// <summary>
        ///     The NumSharp type code.
        /// </summary>
        public readonly NPTypeCode typecode;

        /// <summary>
        ///     A unique character code for each of the 21 different built-in types.
        /// </summary>
        internal readonly NPY_TYPECHAR TYPECHAR;

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
        public readonly char kind;

        /// <summary>
        /// A unique character code for each of the 21 different built-in types.
        /// </summary>
        public char @char => (char)TYPECHAR;

        // ---- factories (public ctors are identical) ----

        /// <summary>Builds a descriptor from a C# <see cref="System.Type"/>. Same as <c>new DType(type)</c>.</summary>
        public static DType From(Type type) => new DType(type);

        /// <summary>Builds a descriptor from an <see cref="NPTypeCode"/>. Same as <c>new DType(typecode)</c>.</summary>
        public static DType From(NPTypeCode typecode) => new DType(typecode);

        /// <summary>Builds a descriptor from a NumPy dtype <b>string</b> (NumPy's case-sensitive spelling). Same as <see cref="np.dtype(string)"/>.</summary>
        public static DType From(string dtype) => np.dtype(dtype);

        /// <summary>
        ///     Returns this descriptor's <see cref="NPTypeCode"/>. Combined with a nullable <see cref="DType"/>
        ///     this keeps the old <c>Type dtype</c> idiom drop-in: <c>dtype?.GetTypeCode()</c> yields
        ///     <c>NPTypeCode?</c> (<see langword="null"/> for the none/infer state where <c>dtype == null</c>).
        /// </summary>
        public NPTypeCode GetTypeCode() => typecode;

        // ---- one static spelling per NPTypeCode (the 15 NumSharp element types) ----

        /// <summary>The <see cref="System.Boolean"/> descriptor.</summary>
        public static DType Boolean => new DType(NPTypeCode.Boolean);
        /// <summary>The <see cref="System.Byte"/> (uint8) descriptor.</summary>
        public static DType Byte => new DType(NPTypeCode.Byte);
        /// <summary>The <see cref="System.SByte"/> (int8) descriptor.</summary>
        public static DType SByte => new DType(NPTypeCode.SByte);
        /// <summary>The <see cref="System.Int16"/> descriptor.</summary>
        public static DType Int16 => new DType(NPTypeCode.Int16);
        /// <summary>The <see cref="System.UInt16"/> descriptor.</summary>
        public static DType UInt16 => new DType(NPTypeCode.UInt16);
        /// <summary>The <see cref="System.Int32"/> descriptor.</summary>
        public static DType Int32 => new DType(NPTypeCode.Int32);
        /// <summary>The <see cref="System.UInt32"/> descriptor.</summary>
        public static DType UInt32 => new DType(NPTypeCode.UInt32);
        /// <summary>The <see cref="System.Int64"/> descriptor.</summary>
        public static DType Int64 => new DType(NPTypeCode.Int64);
        /// <summary>The <see cref="System.UInt64"/> descriptor.</summary>
        public static DType UInt64 => new DType(NPTypeCode.UInt64);
        /// <summary>The <see cref="System.Char"/> descriptor.</summary>
        public static DType Char => new DType(NPTypeCode.Char);
        /// <summary>The <see cref="System.Half"/> (float16) descriptor.</summary>
        public static DType Half => new DType(NPTypeCode.Half);
        /// <summary>The <see cref="System.Single"/> (float32) descriptor.</summary>
        public static DType Single => new DType(NPTypeCode.Single);
        /// <summary>The <see cref="System.Double"/> (float64) descriptor.</summary>
        public static DType Double => new DType(NPTypeCode.Double);
        /// <summary>The <see cref="System.Decimal"/> descriptor.</summary>
        public static DType Decimal => new DType(NPTypeCode.Decimal);
        /// <summary>The <see cref="System.Numerics.Complex"/> (complex128) descriptor.</summary>
        public static DType Complex => new DType(NPTypeCode.Complex);

        // ---- implicit conversions: DType is the single spelling Type / NPTypeCode / NumPy-string collapse into ----

        /// <summary>A C# <see cref="System.Type"/> converts to a descriptor (<see langword="null"/> ⇒ none).</summary>
        public static implicit operator DType(Type type) => type == null ? null : new DType(type);

        /// <summary>An <see cref="NPTypeCode"/> converts to a descriptor (<see cref="NPTypeCode.Empty"/> ⇒ none).</summary>
        public static implicit operator DType(NPTypeCode typecode) => typecode == NPTypeCode.Empty ? null : new DType(typecode);

        /// <summary>A nullable <see cref="NPTypeCode"/> converts to a descriptor (<see langword="null"/>/<see cref="NPTypeCode.Empty"/> ⇒ none).</summary>
        public static implicit operator DType(NPTypeCode? typecode) => typecode.HasValue ? (DType)typecode.Value : null;

        /// <summary>
        ///     A NumPy dtype <b>string</b> converts to a descriptor via <see cref="np.dtype(string)"/> — NumPy's exact,
        ///     case-sensitive spelling (<c>"f4"</c>, <c>"float32"</c>, <c>"&lt;f8"</c>, <c>"F"</c>). <see langword="null"/> ⇒ none.
        /// </summary>
        public static implicit operator DType(string dtype) => dtype == null ? null : np.dtype(dtype);

        /// <summary>A descriptor converts back to its <see cref="System.Type"/> (none/<see langword="null"/> ⇒ null).</summary>
        public static implicit operator Type(DType dtype) => dtype?.type;

        /// <summary>A descriptor converts back to its <see cref="NPTypeCode"/> (none/<see langword="null"/> ⇒ <see cref="NPTypeCode.Empty"/>).</summary>
        public static implicit operator NPTypeCode(DType dtype) => dtype is null ? NPTypeCode.Empty : dtype.typecode;

        // ---- value equality (by typecode, null-safe) ----

        /// <inheritdoc/>
        public bool Equals(DType other) => other is not null && typecode == other.typecode;

        /// <inheritdoc/>
        public override bool Equals(object obj) => Equals(obj as DType);

        /// <inheritdoc/>
        public override int GetHashCode() => (int)typecode;

        /// <summary>Value equality by <see cref="typecode"/> (null-safe).</summary>
        public static bool operator ==(DType left, DType right)
        {
            if (left is null) return right is null;
            return right is not null && left.typecode == right.typecode;
        }

        /// <summary>Value inequality by <see cref="typecode"/> (null-safe).</summary>
        public static bool operator !=(DType left, DType right) => !(left == right);

        /// <inheritdoc/>
        public override string ToString() => name ?? "None";

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
