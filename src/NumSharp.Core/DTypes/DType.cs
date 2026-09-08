using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     NumSharp's data-type descriptor — the INSTANCE of a <see cref="DTypeMeta"/> class, standing in for NumPy's
    ///     <c>numpy.dtype</c> object (<c>np.dtype('f8')</c>, <c>np.dtype('M8[ns]')</c>), and the <b>single dtype
    ///     spelling</b> every dtype-taking API in NumSharp accepts.
    /// </summary>
    /// <remarks>
    /// <para>
    ///     <b>Two levels (NEP 41/42).</b> The CLASS (<see cref="Meta"/> — <c>np.dtypes.Float64DType</c>) owns the
    ///     behaviour: promotion, casting, the default instance; the INSTANCE owns the parameters: <see cref="byteorder"/>
    ///     and, for the parametric datetime pair, the unit (<see cref="DatetimeMetadata"/>). The 15 storage-backed
    ///     builtins have one canonical instance each (<c>np.dtype("i8")</c> is the same object every time —
    ///     <see cref="isbuiltin"/> == 1); a datetime descriptor or a non-native-byte-order one is a fresh object
    ///     (<see cref="isbuiltin"/> == 0), exactly as in NumPy.
    /// </para>
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
    ///       <item><description><b><see cref="NPTypeCode"/></b> — <c>NPTypeCode.Single</c> (NumSharp's compact storage enum; no NumPy counterpart).</description></item>
    ///       <item><description><b>NumPy dtype string</b> — <c>"float32"</c> / <c>"f4"</c> / <c>"&lt;f8"</c> / <c>"M8[ns]"</c> (NumPy's <c>dtype='float32'</c> — see the casing rules below).</description></item>
    ///       <item><description><b><see cref="DType"/> itself</b> — <c>DType.Single</c>, <c>np.dtype("f4")</c>, <c>DType.From(...)</c>.</description></item>
    ///     </list>
    ///     A <see cref="DType"/> also converts back to <see cref="System.Type"/> and <see cref="NPTypeCode"/>
    ///     implicitly, so it drops straight into code expecting either — as long as its class has storage: a
    ///     datetime descriptor has none yet (Stage A), and converting one raises <see cref="NotSupportedException"/>
    ///     rather than silently degrading to the "infer" state.
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
    ///     <see cref="np.dtype(string)"/> with NumPy's exact, <b>case-sensitive</b> grammar — the single-character
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
    ///       <item><term>M8[unit] / m8[unit]</term><description>datetime64 / timedelta64 (descriptor-level in Stage A)</description></item>
    ///     </list>
    ///     Sized forms (<c>"i4"</c>, <c>"f8"</c>, <c>"c16"</c>), lowercase names (<c>"float64"</c>, <c>"int32"</c>,
    ///     <c>"complex128"</c>, <c>"datetime64[ns]"</c>) and byte-order prefixes (<c>"&lt;f8"</c>, <c>"&gt;i4"</c>,
    ///     <c>"=u2"</c>, <c>"|b1"</c>) are all accepted; a non-native prefix (<c>"&gt;i4"</c> on this little-endian host) is
    ///     KEPT on the descriptor (<see cref="byteorder"/> <c>'&gt;'</c>, <see cref="isnative"/> false) as NumPy does.
    /// </para>
    /// <para>
    ///     <b>Deliberately narrowed to NumSharp's capability</b> (the same narrowing as <see cref="np.dtype(string)"/>):
    ///     only NumSharp's <b>15 element types</b> can back an array. Three consequences differ from NumPy and are
    ///     intentional — (1) <b>complex64</b> (<c>'F'</c>, <c>"c8"</c>, <c>"complex64"</c>) is rejected with
    ///     <see cref="NotSupportedException"/> (NumSharp has only complex128), as are structured / void / object /
    ///     (byte)string dtypes; (2) NumSharp additionally accepts a <b>superset</b> of NumPy's
    ///     casing — the C# / <see cref="NPTypeCode"/> PascalCase names (<c>"Int32"</c>, <c>"Single"</c>,
    ///     <c>"Boolean"</c>, <c>"SByte"</c>, <c>"Decimal"</c>, <c>"Char"</c>) that NumPy 2.4.2 rejects — as a
    ///     convenience for C# callers; (3) <c>datetime64</c>/<c>timedelta64</c> descriptors exist (parse, format,
    ///     promote, cast-check) but cannot allocate storage yet.
    /// </para>
    /// <para>
    ///     <b>Descriptor surface</b> (mirrors <c>numpy.dtype</c>): <see cref="type"/> (the C# <see cref="System.Type"/>),
    ///     <see cref="typecode"/> (<see cref="NPTypeCode"/>), <see cref="Meta"/>, <see cref="num"/>, <see cref="name"/>,
    ///     <see cref="kind"/> (<c>'b'/'i'/'u'/'f'/'c'/'M'/'m'</c>), <see cref="char"/>, <see cref="itemsize"/>,
    ///     <see cref="alignment"/>, <see cref="byteorder"/>, <see cref="str"/>, <see cref="descr"/>, <see cref="isbuiltin"/>,
    ///     <see cref="isnative"/>, <see cref="hasobject"/>, <see cref="flags"/>, <see cref="metadata"/>,
    ///     <see cref="newbyteorder(string)"/>. Equality is structural (class + byte order + parameters) and null-safe;
    ///     <see cref="Equals(object)"/> additionally coerces a <see cref="System.Type"/>, an <see cref="NPTypeCode"/> or a
    ///     dtype string the way <c>np.dtype('i4') == 'i4'</c> does. <see cref="ToString()"/> is NumPy's <c>str(dtype)</c>
    ///     (<c>int32</c>, <c>&gt;i4</c>, <c>datetime64[ns]</c>); <see cref="ToString(bool)"/> with <c>repr: true</c> is
    ///     <c>repr(dtype)</c> (<c>dtype('int32')</c>, <c>dtype('&lt;M8[ns]')</c>).
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
    /// bool same = DType.Single == (DType)"f4"; // true — structural equality
    /// DType m  = np.dtype("M8[10ns]");       // datetime64[10ns]: m.kind=='M', np.datetime_data(m) == ("ns", 10)
    /// </code>
    /// </example>
    /// </remarks>
    /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype</remarks>
    public sealed class DType : IEquatable<DType>
    {
        /// <summary>
        ///     The LEGACY kind map the old promotion engine (<see cref="np._FindCommonType(NPTypeCode[], NPTypeCode[])"/>)
        ///     keys its kind-ordering on. Kept byte-for-byte (<c>'?'</c> for bool, <c>'S'</c> for char) so that engine's
        ///     answers do not move; the descriptor's own <see cref="kind"/> comes from <see cref="DTypeMeta.Kind"/>.
        /// </summary>
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

        /// <summary>NumPy's <c>'='</c>: the byte order of this host.</summary>
        public const char NativeByteOrder = '=';

        /// <summary>NumPy's <c>'|'</c>: byte order is not applicable (single-byte types).</summary>
        public const char NotApplicableByteOrder = '|';

        /// <summary>The explicit character of this host's byte order (<c>'&lt;'</c> on little-endian machines) — what <see cref="str"/> renders for a native descriptor.</summary>
        public static char HostByteOrder => BitConverter.IsLittleEndian ? '<' : '>';

        /// <summary>The character of the OTHER byte order (<c>'&gt;'</c> on little-endian machines) — what a non-native descriptor carries.</summary>
        public static char SwappedByteOrder => BitConverter.IsLittleEndian ? '>' : '<';

        private readonly DatetimeMetaData? _datetime;

        // ---- construction ----

        /// <summary>
        ///     Builds a fresh descriptor for a C# <see cref="System.Type"/> — a COPY of the class's canonical instance
        ///     (prefer <see cref="From(System.Type)"/> / the implicit conversion, which return the singleton itself).
        ///     For the "none/infer" state use a <see langword="null"/> <see cref="DType"/> (this ctor throws on a
        ///     <see langword="null"/> type; the implicit <c>Type</c>→<c>DType</c> conversion yields <see langword="null"/>).
        /// </summary>
        public DType(Type type) : this(MetaOf(type ?? throw new ArgumentNullException(nameof(type))), NativeByteOrder, null)
        {
        }

        /// <summary>
        ///     Builds a fresh descriptor for an <see cref="NPTypeCode"/> — a COPY of the class's canonical instance
        ///     (prefer <see cref="From(NPTypeCode)"/> / the implicit conversion, which return the singleton itself).
        ///     For the "none/infer" state use a <see langword="null"/> <see cref="DType"/> (this ctor throws on
        ///     <see cref="NPTypeCode.Empty"/>; the implicit <c>NPTypeCode</c>→<c>DType</c> conversion yields <see langword="null"/>).
        /// </summary>
        public DType(NPTypeCode typecode) : this(MetaOf(typecode), NativeByteOrder, null)
        {
        }

        /// <summary>The descriptor constructor proper: a class, a byte order and the class's instance parameters.</summary>
        internal DType(DTypeMeta meta, char byteorder, DatetimeMetaData? datetime)
        {
            Meta = meta ?? throw new ArgumentNullException(nameof(meta));
            if (meta.IsAbstract)
                throw new TypeError($"Cannot instantiate abstract DType {meta}");
            if (meta is DatetimeDTypeMeta != datetime.HasValue)
                throw new ArgumentException($"{meta} {(datetime.HasValue ? "takes no" : "requires")} datetime metadata.", nameof(datetime));

            _datetime = datetime;
            this.type = meta.ScalarType;
            this.typecode = meta.TypeCode;
            this.itemsize = meta.ItemSize;
            this.kind = meta.Kind;
            this.byteorder = NormalizeByteOrder(meta, byteorder);
            this.TYPECHAR = (NPY_TYPECHAR)meta.TypeChar;
            this.name = ComputeName();
        }

        private static DTypeMeta MetaOf(Type type)
        {
            var meta = DTypeRegistry.FromScalarType(type);
            if (meta == null)
                throw new NotSupportedException($"Cannot interpret '{type}' as a data type: it is not one of NumSharp's element types.");
            return meta;
        }

        private static DTypeMeta MetaOf(NPTypeCode typecode)
        {
            if (typecode == NPTypeCode.Empty)
                throw new ArgumentException("NPTypeCode.Empty has no dtype; use a null DType for the none/infer state.", nameof(typecode));
            var meta = DTypeRegistry.FromTypeCode(typecode);
            if (meta == null)
                throw new NotSupportedException($"NPTypeCode.{typecode} has no registered DType class.");
            return meta;
        }

        /// <summary>
        ///     NumPy's byte-order normalisation: single-byte types are always <c>'|'</c>; the host's own order (or
        ///     <c>'|'</c>, or <c>'='</c>) is <c>'='</c>; only the OTHER order survives as an explicit character.
        /// </summary>
        private static char NormalizeByteOrder(DTypeMeta meta, char byteorder)
        {
            if (meta.ItemSize == 1)
                return NotApplicableByteOrder;
            switch (byteorder)
            {
                case NativeByteOrder:
                case NotApplicableByteOrder:
                    return NativeByteOrder;
                case '<':
                case '>':
                    return byteorder == HostByteOrder ? NativeByteOrder : byteorder;
                default:
                    throw new ArgumentException($"byteorder not recognized (got '{byteorder}')", nameof(byteorder));
            }
        }

        /// <summary>
        ///     NumPy's <c>_name_get</c>: <c>"bool"</c>; kind-stem plus bit count for the numeric builtins (<c>"int32"</c>,
        ///     <c>"float64"</c>, <c>"complex128"</c>); the scalar-type name for a user-range class (<c>"decimal"</c>,
        ///     <c>"char"</c>); the class name plus unit for datetimes (<c>"datetime64[ns]"</c>).
        /// </summary>
        private string ComputeName()
        {
            if (Meta.TypeNum >= 256)
                return Meta.ScalarName; // user dtypes don't promise to do anything special
            if (Meta is DatetimeDTypeMeta)
                return Meta.ScalarName + _datetime.Value.ToString(skipBrackets: false);
            switch (kind)
            {
                case 'b': return "bool"; // implied
                case 'U': return "str";  // the vestigial unsized flexible class: no bit suffix
                case 'u': return "uint" + itemsize * 8;
                case 'i': return "int" + itemsize * 8;
                case 'c': return "complex" + itemsize * 8;
                case 'f': return "float" + itemsize * 8;
                default: throw new RuntimeError($"internal dtype error, unknown kind '{kind}'");
            }
        }

        // ---- the descriptor surface ----

        /// <summary>The DType CLASS this descriptor instantiates (<c>type(np.dtype('f8'))</c>): <c>np.dtypes.Float64DType</c>.</summary>
        public DTypeMeta Meta { get; }

        /// <summary>
        ///     A character indicating the byte-order of this data-type object:
        ///     <c>'='</c> native, <c>'&lt;'</c> little-endian, <c>'&gt;'</c> big-endian, <c>'|'</c> not applicable (single-byte types).
        ///     Only the NON-native order is ever stored explicitly (NumPy normalises the host's own order to <c>'='</c>).
        /// </summary>
        public readonly char byteorder;

        /// <summary>The size of the dtype in bytes (<c>itemsize</c>).</summary>
        public readonly int itemsize;

        /// <summary>
        ///     NumPy's <c>dtype.name</c> spelling (<c>"float32"</c>, <c>"int64"</c>, <c>"bool"</c>, <c>"complex128"</c>,
        ///     <c>"datetime64[ns]"</c>) — the CLR <see cref="Type.Name"/> (<c>"Single"</c>) is not a NumPy name and does not
        ///     round-trip through <see cref="np.dtype(string)"/>. The NumSharp-only dtypes have no NumPy analog and get
        ///     lowercase names of their own (<c>"decimal"</c>, <c>"char"</c>) rather than the nearest NumPy stand-in.
        /// </summary>
        public readonly string name;

        /// <summary>The C# scalar type of one element (NumPy's <c>dtype.type</c>); null for a class with no C# scalar yet (datetime64 in Stage A).</summary>
        public readonly Type type;

        /// <summary>The NumSharp storage discriminator; <see cref="NPTypeCode.Empty"/> for a class without storage (datetime64 in Stage A).</summary>
        public readonly NPTypeCode typecode;

        /// <summary>The NumPy type character as the legacy enum (<see cref="char"/> is the public form).</summary>
        internal readonly NPY_TYPECHAR TYPECHAR;

        /// <summary>
        ///     A character code (one of ‘biufcmMOSUV’) identifying the general kind of data:
        ///     <c>b</c> boolean, <c>i</c> signed integer, <c>u</c> unsigned integer, <c>f</c> floating-point,
        ///     <c>c</c> complex floating-point, <c>m</c> timedelta, <c>M</c> datetime, <c>O</c> object,
        ///     <c>S</c> (byte-)string, <c>U</c> Unicode, <c>V</c> void.
        /// </summary>
        public readonly char kind;

        /// <summary>A unique character code for each of the built-in types (<c>'?' 'b' 'B' 'h' 'H' 'i' 'I' 'l' 'L' 'e' 'f' 'd' 'D' 'M' 'm'</c>).</summary>
        public char @char => Meta.TypeChar;

        /// <summary>NumPy's unique type number (<c>dtype.num</c>).</summary>
        public int num => Meta.TypeNum;

        /// <summary>The required alignment (bytes) of this data-type (<c>dtype.alignment</c>).</summary>
        public int alignment => Meta.Alignment;

        /// <summary>The instance parameters of a <c>datetime64</c>/<c>timedelta64</c> descriptor (NumPy's <c>c_metadata</c>); null for every other class.</summary>
        public DatetimeMetaData? DatetimeMetadata => _datetime;

        /// <summary>
        ///     The array-interface typestring (<c>dtype.str</c>): explicit byte order, kind and size — <c>"&lt;i4"</c>,
        ///     <c>"|b1"</c>, <c>"&gt;f8"</c>, <c>"&lt;M8[ns]"</c>, <c>"&lt;m8"</c>.
        /// </summary>
        public string str
        {
            get
            {
                char endian = byteorder == NativeByteOrder ? HostByteOrder : byteorder;
                if (Meta is DatetimeDTypeMeta)
                    return endian.ToString() + Meta.TypeChar + "8" + _datetime.Value.ToString(skipBrackets: false);
                int size = kind == 'U' ? itemsize >> 2 : itemsize;
                return endian.ToString() + kind + size;
            }
        }

        /// <summary>The array-interface description (<c>dtype.descr</c>): one unnamed field carrying <see cref="str"/> — <c>[('', '&lt;i4')]</c>.</summary>
        public IReadOnlyList<(string name, string typestr)> descr => new[] { ("", str) };

        /// <summary>
        ///     <c>dtype.isbuiltin</c>: 1 for the canonical instance of a builtin class, 2 for a user-defined class (the
        ///     NumSharp-only Decimal/Char), 0 for any other instance — a datetime descriptor, a non-native byte order, or a
        ///     copy made with the public constructors.
        /// </summary>
        public int isbuiltin => Meta.TypeNum >= 256 ? 2 : (ReferenceEquals(this, Meta.Singleton) ? 1 : 0);

        /// <summary><c>dtype.isnative</c>: true unless the byte order is explicitly the non-host one.</summary>
        public bool isnative => byteorder != '<' && byteorder != '>';

        /// <summary><c>dtype.hasobject</c>: whether items hold references that need clearing (false for every current class).</summary>
        public bool hasobject => Meta.HasReferences;

        /// <summary><c>dtype.flags</c>: the item flags (NumPy's <c>NPY_ITEM_REFCOUNT | NPY_NEEDS_INIT | …</c> = 63 for a reference-holding dtype, 0 otherwise).</summary>
        public int flags => Meta.HasReferences ? 63 : 0;

        /// <summary><c>dtype.metadata</c>: user metadata (never set in NumSharp — always null).</summary>
        public object metadata => null;

        /// <summary><c>dtype.shape</c>: the sub-array shape — always empty (no sub-array dtypes).</summary>
        public int[] shape => Array.Empty<int>();

        /// <summary><c>dtype.ndim</c>: the sub-array rank — always 0.</summary>
        public int ndim => 0;

        /// <summary><c>dtype.base</c>: the base of a sub-array dtype — this descriptor itself.</summary>
        public DType @base => this;

        /// <summary><c>dtype.subdtype</c>: always null (no sub-array dtypes).</summary>
        public (DType, int[])? subdtype => null;

        /// <summary><c>dtype.fields</c>: always null (no structured dtypes).</summary>
        public IReadOnlyDictionary<string, (DType, int)> fields => null;

        /// <summary><c>dtype.names</c>: always null (no structured dtypes).</summary>
        public string[] names => null;

        // ---- factories (public ctors are copies; these return the canonical instance) ----

        /// <summary>The descriptor of a C# <see cref="System.Type"/> — the class's canonical instance (<c>np.dtype(typeof(int))</c>).</summary>
        public static DType From(Type type)
        {
            if (type == null)
                throw new ArgumentNullException(nameof(type));
            return MetaOf(type).Singleton;
        }

        /// <summary>The descriptor of an <see cref="NPTypeCode"/> — the class's canonical instance.</summary>
        public static DType From(NPTypeCode typecode) => MetaOf(typecode).Singleton;

        /// <summary>Builds a descriptor from a NumPy dtype <b>string</b> (NumPy's case-sensitive grammar). Same as <see cref="np.dtype(string)"/>.</summary>
        public static DType From(string dtype) => np.dtype(dtype);

        /// <summary>
        ///     Returns this descriptor's <see cref="NPTypeCode"/>. Combined with a nullable <see cref="DType"/>
        ///     this keeps the old <c>Type dtype</c> idiom drop-in: <c>dtype?.GetTypeCode()</c> yields
        ///     <c>NPTypeCode?</c> (<see langword="null"/> for the none/infer state where <c>dtype == null</c>).
        /// </summary>
        /// <exception cref="NotSupportedException">The class has no storage yet (a datetime64/timedelta64 descriptor in Stage A).</exception>
        public NPTypeCode GetTypeCode()
        {
            if (typecode == NPTypeCode.Empty)
                throw NoStorage();
            return typecode;
        }

        private NotSupportedException NoStorage()
            => new NotSupportedException(
                $"dtype {ToString(repr: true)} has no NumSharp storage yet: {Meta} exists at the descriptor level (parsing, promotion, casting rules) " +
                "but arrays of it cannot be allocated until its storage lane and kernels land (Stage C of docs/plans/dtype-system.md).");

        // ---- one static spelling per NPTypeCode (the 15 NumSharp element types): the canonical instances ----

        /// <summary>The <see cref="System.Boolean"/> descriptor.</summary>
        public static DType Boolean => DTypeRegistry.Bool.Singleton;
        /// <summary>The <see cref="System.Byte"/> (uint8) descriptor.</summary>
        public static DType Byte => DTypeRegistry.UInt8.Singleton;
        /// <summary>The <see cref="System.SByte"/> (int8) descriptor.</summary>
        public static DType SByte => DTypeRegistry.Int8.Singleton;
        /// <summary>The <see cref="System.Int16"/> descriptor.</summary>
        public static DType Int16 => DTypeRegistry.Int16.Singleton;
        /// <summary>The <see cref="System.UInt16"/> descriptor.</summary>
        public static DType UInt16 => DTypeRegistry.UInt16.Singleton;
        /// <summary>The <see cref="System.Int32"/> descriptor.</summary>
        public static DType Int32 => DTypeRegistry.Int32.Singleton;
        /// <summary>The <see cref="System.UInt32"/> descriptor.</summary>
        public static DType UInt32 => DTypeRegistry.UInt32.Singleton;
        /// <summary>The <see cref="System.Int64"/> descriptor.</summary>
        public static DType Int64 => DTypeRegistry.Int64.Singleton;
        /// <summary>The <see cref="System.UInt64"/> descriptor.</summary>
        public static DType UInt64 => DTypeRegistry.UInt64.Singleton;
        /// <summary>The <see cref="System.Char"/> descriptor.</summary>
        public static DType Char => DTypeRegistry.Char.Singleton;
        /// <summary>The <see cref="System.Half"/> (float16) descriptor.</summary>
        public static DType Half => DTypeRegistry.Half.Singleton;
        /// <summary>The <see cref="System.Single"/> (float32) descriptor.</summary>
        public static DType Single => DTypeRegistry.Single.Singleton;
        /// <summary>The <see cref="System.Double"/> (float64) descriptor.</summary>
        public static DType Double => DTypeRegistry.Double.Singleton;
        /// <summary>The <see cref="System.Decimal"/> descriptor.</summary>
        public static DType Decimal => DTypeRegistry.Decimal.Singleton;
        /// <summary>The <see cref="System.Numerics.Complex"/> (complex128) descriptor.</summary>
        public static DType Complex => DTypeRegistry.Complex128.Singleton;

        // ---- implicit conversions: DType is the single spelling Type / NPTypeCode / NumPy-string collapse into ----

        /// <summary>A C# <see cref="System.Type"/> converts to its canonical descriptor (<see langword="null"/> ⇒ none).</summary>
        public static implicit operator DType(Type type) => type == null ? null : From(type);

        /// <summary>An <see cref="NPTypeCode"/> converts to its canonical descriptor (<see cref="NPTypeCode.Empty"/> ⇒ none).</summary>
        public static implicit operator DType(NPTypeCode typecode) => typecode == NPTypeCode.Empty ? null : From(typecode);

        /// <summary>A nullable <see cref="NPTypeCode"/> converts to a descriptor (<see langword="null"/>/<see cref="NPTypeCode.Empty"/> ⇒ none).</summary>
        public static implicit operator DType(NPTypeCode? typecode) => typecode.HasValue ? (DType)typecode.Value : null;

        /// <summary>
        ///     A NumPy dtype <b>string</b> converts to a descriptor via <see cref="np.dtype(string)"/> — NumPy's exact,
        ///     case-sensitive spelling (<c>"f4"</c>, <c>"float32"</c>, <c>"&lt;f8"</c>, <c>"M8[ns]"</c>). <see langword="null"/> ⇒ none.
        /// </summary>
        public static implicit operator DType(string dtype) => dtype == null ? null : np.dtype(dtype);

        /// <summary>A descriptor converts back to its <see cref="System.Type"/> (none/<see langword="null"/> ⇒ null).</summary>
        /// <exception cref="NotSupportedException">The class has no C# storage type yet (datetime64/timedelta64 in Stage A).</exception>
        public static implicit operator Type(DType dtype)
        {
            if (dtype is null)
                return null;
            if (dtype.type == null)
                throw dtype.NoStorage();
            return dtype.type;
        }

        /// <summary>A descriptor converts back to its <see cref="NPTypeCode"/> (none/<see langword="null"/> ⇒ <see cref="NPTypeCode.Empty"/>).</summary>
        /// <exception cref="NotSupportedException">The class has no storage yet (datetime64/timedelta64 in Stage A).</exception>
        public static implicit operator NPTypeCode(DType dtype) => dtype is null ? NPTypeCode.Empty : dtype.GetTypeCode();

        // ---- equality: structural (class + byte order + parameters), null-safe; coercing for foreign operands ----

        /// <summary>Structural equality: the same class, byte order and instance parameters (<c>M8[ns] != M8[s]</c>, <c>&gt;i4 != i4</c>).</summary>
        public bool Equals(DType other)
            => other is not null && ReferenceEquals(Meta, other.Meta) && byteorder == other.byteorder && Nullable.Equals(_datetime, other._datetime);

        /// <summary>
        ///     NumPy's coercing <c>dtype.__eq__</c>: the operand is converted to a descriptor first, so a
        ///     <see cref="System.Type"/> (<c>typeof(int)</c>), an <see cref="NPTypeCode"/> or a dtype string
        ///     (<c>"i4"</c>, <c>"int32"</c>) compare equal to the descriptor they denote; anything that is not a dtype
        ///     (a bad string, a <see cref="DTypeMeta"/> class object, null) compares unequal without raising.
        /// </summary>
        public override bool Equals(object obj)
        {
            switch (obj)
            {
                case null:
                    return false;
                case DType d:
                    return Equals(d);
                case Type t:
                    return DTypeRegistry.FromScalarType(t) is { } meta && Equals(meta.Singleton);
                case NPTypeCode tc:
                    return tc != NPTypeCode.Empty && DTypeRegistry.FromTypeCode(tc) is { } m2 && Equals(m2.Singleton);
                case string s:
                    try
                    {
                        return Equals(np.dtype(s));
                    }
                    catch (Exception e) when (e is NotSupportedException || e is TypeError || e is ValueError || e is ArgumentException)
                    {
                        return false;
                    }
                default:
                    return false;
            }
        }

        /// <summary>
        ///     Coercing equality with a dtype STRING (<c>np.dtype('i4') == 'i4'</c>): an unparseable string is simply unequal
        ///     (<c>np.dtype('i4') == 'garbage'</c> is False, not an error). This overload exists because a string would
        ///     otherwise bind <see cref="Equals(DType)"/> through the implicit string→<see cref="DType"/> conversion, which
        ///     raises on a bad string.
        /// </summary>
        public bool Equals(string dtype) => Equals((object)dtype);

        /// <summary>Coercing equality with a C# <see cref="System.Type"/> (<c>np.dtype('i4') == np.int32</c>); a non-NumSharp type is unequal.</summary>
        public bool Equals(Type type) => Equals((object)type);

        /// <summary>Coercing equality with an <see cref="NPTypeCode"/>; <see cref="NPTypeCode.Empty"/> is unequal.</summary>
        public bool Equals(NPTypeCode typecode) => Equals((object)typecode);

        /// <summary>Consistent with <see cref="Equals(DType)"/>: the class and byte order, not the parameters (NumPy: <c>hash(M8[ns]) == hash(M8[s])</c>).</summary>
        public override int GetHashCode() => HashCode.Combine(Meta.TypeNum, byteorder);

        // NOTE: there are deliberately NO `==(DType, string)` operators. A `null` literal would bind them (`descr == null`
        // silently became "false" for a null descriptor inside the engine), so the coercing string comparison lives on
        // Equals(string) only; `dtype == "i4"` still works through the implicit string→DType conversion (and, like that
        // conversion, raises on a string that is not a dtype).

        /// <summary>Structural equality (null-safe).</summary>
        public static bool operator ==(DType left, DType right)
        {
            if (left is null) return right is null;
            return left.Equals(right);
        }

        /// <summary>Structural inequality (null-safe).</summary>
        public static bool operator !=(DType left, DType right) => !(left == right);

        /// <summary>NumPy's <c>dtype &lt; other</c>: strictly promotes — not equal, and <paramref name="left"/> casts safely to <paramref name="right"/>.</summary>
        public static bool operator <(DType left, DType right) => Check(left, right) && !left.Equals(right) && DTypeCasting.CanCastTypeTo(left, right, NPY_CASTING.NPY_SAFE_CASTING);

        /// <summary>NumPy's <c>dtype &lt;= other</c>: <paramref name="left"/> casts safely to <paramref name="right"/>.</summary>
        public static bool operator <=(DType left, DType right) => Check(left, right) && DTypeCasting.CanCastTypeTo(left, right, NPY_CASTING.NPY_SAFE_CASTING);

        /// <summary>NumPy's <c>dtype &gt; other</c>: not equal, and <paramref name="right"/> casts safely to <paramref name="left"/>.</summary>
        public static bool operator >(DType left, DType right) => Check(left, right) && !left.Equals(right) && DTypeCasting.CanCastTypeTo(right, left, NPY_CASTING.NPY_SAFE_CASTING);

        /// <summary>NumPy's <c>dtype &gt;= other</c>: <paramref name="right"/> casts safely to <paramref name="left"/>.</summary>
        public static bool operator >=(DType left, DType right) => Check(left, right) && DTypeCasting.CanCastTypeTo(right, left, NPY_CASTING.NPY_SAFE_CASTING);

        private static bool Check(DType left, DType right)
        {
            if (left is null) throw new ArgumentNullException(nameof(left));
            if (right is null) throw new ArgumentNullException(nameof(right));
            return true;
        }

        // ---- rendering ----

        /// <summary>
        ///     NumPy's <c>str(dtype)</c>: the <see cref="name"/> for a native descriptor (<c>int32</c>, <c>datetime64[ns]</c>,
        ///     <c>decimal</c>), the typestring for a non-native or flexible one (<c>&gt;i4</c>).
        /// </summary>
        public override string ToString() => (!isnative || kind == 'U' || kind == 'S' || kind == 'V') ? str : name;

        /// <summary>
        ///     <paramref name="repr"/> false: <c>str(dtype)</c>. <paramref name="repr"/> true: NumPy's <c>repr(dtype)</c> —
        ///     <c>dtype('int32')</c>, <c>dtype('bool')</c>, <c>dtype('&gt;i4')</c>, <c>dtype('&lt;M8[ns]')</c>, and for the
        ///     NumSharp-only user-range classes NumPy's unquoted user form <c>dtype(decimal)</c>.
        /// </summary>
        public string ToString(bool repr) => repr ? "dtype(" + ConstructionRepr() + ")" : ToString();

        /// <summary>NumPy's <c>_construction_repr</c> / <c>_scalar_str</c>.</summary>
        private string ConstructionRepr()
        {
            if (kind == 'b')
                return "'bool'";
            if (Meta is DatetimeDTypeMeta)
                return "'" + str + "'";
            if (Meta.TypeNum >= 256)
                return Meta.ScalarName;
            if (kind == 'U' && itemsize == 0)
                return "'" + (byteorder == NativeByteOrder ? HostByteOrder : byteorder) + "U'"; // NumPy's unsized `dtype('<U')`
            if (kind == 'U' || kind == 'S' || kind == 'V' || !isnative)
                return "'" + str + "'";
            return "'" + name + "'";
        }

        // ---- byte order ----

        /// <summary>A copy of this descriptor carrying <paramref name="newByteOrder"/> (normalised; single-byte classes stay <c>'|'</c>).</summary>
        internal DType WithByteOrder(char newByteOrder) => new DType(Meta, newByteOrder, _datetime);

        /// <summary>
        ///     Return a new dtype with a different byte order (NumPy's <c>dtype.newbyteorder</c>).
        /// </summary>
        /// <param name="new_order">
        ///     Byte order to force. The default (<c>'S'</c>) swaps the current byte order. Codes (first letter, case-insensitive):
        ///     <c>'S'</c> swap; <c>'&lt;'</c>/<c>'L'</c>/<c>'little'</c> little-endian; <c>'&gt;'</c>/<c>'B'</c>/<c>'big'</c> big-endian;
        ///     <c>'='</c>/<c>'N'</c>/<c>'native'</c> native; <c>'|'</c>/<c>'I'</c>/<c>'ignore'</c> no change.
        /// </param>
        /// <returns>A new descriptor with the requested byte order (single-byte classes are unaffected).</returns>
        /// <exception cref="ValueError"><c>byteorder not recognized (got '…')</c>.</exception>
        public DType newbyteorder(string new_order = "S")
        {
            if (string.IsNullOrEmpty(new_order))
                throw new ValueError($"byteorder not recognized (got '{new_order}')");
            return newbyteorder(new_order[0]);
        }

        /// <summary>See <see cref="newbyteorder(string)"/>.</summary>
        public DType newbyteorder(char new_order)
        {
            char target;
            switch (new_order)
            {
                case '>':
                case 'B':
                case 'b':
                    target = '>';
                    break;
                case '<':
                case 'L':
                    case 'l':
                    target = '<';
                    break;
                case '=':
                case 'N':
                case 'n':
                    target = NativeByteOrder;
                    break;
                case 'S':
                case 's':
                    target = isnative ? SwappedByteOrder : NativeByteOrder;
                    break;
                case '|':
                case 'I':
                case 'i':
                    target = byteorder == NotApplicableByteOrder ? NativeByteOrder : byteorder;
                    break;
                default:
                    throw new ValueError($"byteorder not recognized (got '{new_order}')");
            }
            return WithByteOrder(target);
        }
    }
}
