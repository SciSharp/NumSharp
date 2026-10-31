using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends.Iteration;
using NumSharp.IO; // PyLiteral.Repr: the Python repr a KeyError's str() shows

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
    ///     an <see cref="NPTypeCode"/> enum, and this descriptor from <see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/> — which forced
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
    ///     <see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/> with NumPy's exact, <b>case-sensitive</b> grammar — the single-character
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
    ///     <b>Deliberately narrowed to NumSharp's capability</b> (the same narrowing as <see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/>):
    ///     only NumSharp's <b>15 element types</b> and the two fixed-width string classes (<c>S</c> bytes, <c>U</c>
    ///     Unicode — one <see cref="NPTypeCode.String"/> storage lane, see <see cref="StringDTypeMeta"/>) can back an array.
    ///     Three consequences differ from NumPy and are intentional — (1) <b>complex64</b> (<c>'F'</c>, <c>"c8"</c>,
    ///     <c>"complex64"</c>) is rejected with <see cref="NotSupportedException"/> (NumSharp has only complex128), as are
    ///     structured / void / object dtypes; (2) NumSharp additionally accepts a <b>superset</b> of NumPy's
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

        // ---- NumPy's byte-order characters, under their C macro names (ndarraytypes.h) ----

        /// <summary>
        ///     NumPy's <c>NPY_NATIVE</c> (<c>'='</c>): "the byte order of this host", the form every parsed native descriptor
        ///     stores. A descriptor can also carry the host order EXPLICITLY (<see cref="NPY_NATBYTE"/>, what
        ///     <c>newbyteorder('&lt;')</c> leaves on a little-endian host); both are native and compare equal, but the
        ///     explicit form shows in <c>repr</c> (<c>dtype('&lt;f8')</c> instead of <c>dtype('float64')</c>).
        /// </summary>
        public const char NPY_NATIVE = '=';

        /// <summary>NumPy's <c>NPY_IGNORE</c> (<c>'|'</c>): byte order is not applicable — what every single-byte descriptor stores.</summary>
        public const char NPY_IGNORE = '|';

        /// <summary>NumPy's <c>NPY_SWAP</c> (<c>'s'</c>): the <c>newbyteorder</c> request "swap the current order".</summary>
        public const char NPY_SWAP = 's';

        /// <summary>NumPy's <c>NPY_LITTLE</c> (<c>'&lt;'</c>): little-endian.</summary>
        public const char NPY_LITTLE = '<';

        /// <summary>NumPy's <c>NPY_BIG</c> (<c>'&gt;'</c>): big-endian.</summary>
        public const char NPY_BIG = '>';

        /// <summary>
        ///     NumPy's <c>NPY_NATBYTE</c>: the explicit character of this host's byte order (<c>'&lt;'</c> on little-endian
        ///     machines) — what <see cref="str"/> renders for a native descriptor and what swapping a non-native descriptor
        ///     produces. A <c>static readonly</c> field rather than a <c>const</c>: C fixes it at compile time, .NET at type
        ///     initialisation, which the JIT then treats as a constant.
        /// </summary>
        public static readonly char NPY_NATBYTE = BitConverter.IsLittleEndian ? NPY_LITTLE : NPY_BIG;

        /// <summary>
        ///     NumPy's <c>NPY_OPPBYTE</c>: the character of the OTHER byte order (<c>'&gt;'</c> on little-endian machines) —
        ///     the only byte order that makes a descriptor non-native (<c>PyArray_ISNBO(c)</c> is <c>c != NPY_OPPBYTE</c>).
        /// </summary>
        public static readonly char NPY_OPPBYTE = BitConverter.IsLittleEndian ? NPY_BIG : NPY_LITTLE;

        /// <summary>The datetime unit of a datetime64 / timedelta64 descriptor (NumPy's <c>c_metadata</c>); null for every other class.</summary>
        private readonly DatetimeMetaData? _datetime;

        /// <summary>
        ///     NumPy's <c>PyArray_Descr.metadata</c>: the user dictionary <c>np.dtype(..., metadata=...)</c> attached, or null.
        ///     Immutable once built (NumPy hands it out as a read-only <c>mappingproxy</c>), so copies of the descriptor may
        ///     share it — as NumPy's <c>PyArray_DescrNew</c> shares the dict between a descriptor and its copies.
        /// </summary>
        private readonly IReadOnlyDictionary<string, object> _metadata;

        /// <summary>
        ///     The instance parameters of a NEP 55 <c>StringDType</c> descriptor (<c>na_object</c>, <c>coerce</c> and the values
        ///     NumPy derives from them); null for every other class.
        /// </summary>
        private readonly VStringParameters _vstring;

        // ---- construction ----

        /// <summary>
        ///     Builds a fresh descriptor for a C# <see cref="System.Type"/> — a COPY of the class's canonical instance
        ///     (prefer <see cref="From(System.Type)"/> / the implicit conversion, which return the singleton itself).
        ///     For the "none/infer" state use a <see langword="null"/> <see cref="DType"/> (this ctor throws on a
        ///     <see langword="null"/> type; the implicit <c>Type</c>→<c>DType</c> conversion yields <see langword="null"/>).
        /// </summary>
        public DType(Type type) : this(MetaOf(type ?? throw new ArgumentNullException(nameof(type))), NPY_NATIVE, null)
        {
        }

        /// <summary>
        ///     Builds a fresh descriptor for an <see cref="NPTypeCode"/> — a COPY of the class's canonical instance
        ///     (prefer <see cref="From(NPTypeCode)"/> / the implicit conversion, which return the singleton itself).
        ///     For the "none/infer" state use a <see langword="null"/> <see cref="DType"/> (this ctor throws on
        ///     <see cref="NPTypeCode.Empty"/>; the implicit <c>NPTypeCode</c>→<c>DType</c> conversion yields <see langword="null"/>).
        /// </summary>
        public DType(NPTypeCode typecode) : this(MetaOf(typecode), NPY_NATIVE, null)
        {
        }

        /// <summary>
        ///     The descriptor constructor for a FIXED-SIZE class — NumPy's <c>PyArray_DescrNew</c> plus the fields it sets: a
        ///     class, a byte order, the class's instance parameters and the user metadata. The item size and type character
        ///     are the class's own (<see cref="DTypeMeta.ItemSize"/>, <see cref="DTypeMeta.TypeChar"/>). Every descriptor it
        ///     builds is a fresh object, so it reports <c>isbuiltin == 0</c> unless the registry installs it as the class
        ///     <see cref="DTypeMeta.singleton"/>.
        /// </summary>
        /// <param name="meta">The class; must not be abstract, and must not be a flexible string class.</param>
        /// <param name="byteorder">
        ///     The byte order to STORE, normalised as NumPy stores it: <c>'|'</c> for a class whose byte order is not
        ///     applicable whatever is asked; <c>'='</c> or <c>'|'</c> for a multi-byte class become <c>'='</c>;
        ///     <c>'&lt;'</c> and <c>'&gt;'</c> are kept as given, the host's own order included — folding an explicit host
        ///     order into <c>'='</c> is the PARSER's job
        ///     (<see cref="np.dtype(string, bool, bool, IReadOnlyDictionary{string, object})"/>), while
        ///     <see cref="newbyteorder(string)"/> keeps it, exactly as NumPy's <c>_convert_from_str</c> and
        ///     <c>PyArray_DescrNewByteorder</c> differ.
        /// </param>
        /// <param name="datetime">The unit of a datetime64 / timedelta64 descriptor; must be given for those classes only.</param>
        /// <param name="metadata">The user metadata to carry (already an immutable copy), or null.</param>
        /// <exception cref="ArgumentNullException"><paramref name="meta"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="meta"/> is abstract (NumPy's "Cannot instantiate abstract DType …").</exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="datetime"/> is missing for a datetime class or given for another class, the class is flexible
        ///     (a string class needs the item-size overload), or <paramref name="byteorder"/> is not one of
        ///     <c>'=' '|' '&lt;' '&gt;'</c>.
        /// </exception>
        internal DType(DTypeMeta meta, char byteorder, DatetimeMetaData? datetime, IReadOnlyDictionary<string, object> metadata = null)
            : this(meta, byteorder, datetime, ClassItemSize(meta), meta?.TypeChar ?? '\0', null, metadata)
        {
        }

        /// <summary>
        ///     The descriptor constructor proper: a class, a byte order, the class's instance parameters, the user metadata,
        ///     and — for a FLEXIBLE class (<c>S</c> / <c>U</c>) — the instance's own item size and type character. NumPy
        ///     stores <c>elsize</c> and <c>type</c> on every descriptor; for the fixed-size classes they always equal the
        ///     class's, for the string classes they are the length (<c>S5</c> → 5 bytes, <c>U5</c> → 20 bytes) and the
        ///     spelling (<c>np.dtype('c')</c> is an <c>S1</c> whose <c>char</c> is <c>'c'</c>).
        /// </summary>
        /// <param name="meta">The DType class this descriptor instantiates; must not be abstract.</param>
        /// <param name="byteorder">The requested byte order (normalised: see <see cref="NormalizeByteOrder"/>).</param>
        /// <param name="datetime">The unit of a <c>datetime64</c>/<c>timedelta64</c> descriptor; null for every other class.</param>
        /// <param name="itemsize">Bytes per element. Must equal <paramref name="meta"/>'s size for a fixed-size class; for a
        /// string class any non-negative size (a multiple of 4 for <c>U</c>), where 0 is NumPy's unsized <c>S0</c>/<c>U0</c>.</param>
        /// <param name="typeChar">NumPy's <c>dtype.char</c>; the class's own character, or <c>'c'</c> for a one-byte <c>S</c>.</param>
        /// <param name="vstring">The parameters of a <c>StringDType</c> descriptor (required for that class, null for every other).</param>
        /// <param name="metadata">The user metadata to carry (already an immutable copy), or null.</param>
        /// <exception cref="ArgumentNullException"><paramref name="meta"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="meta"/> is abstract (NumPy's "Cannot instantiate abstract DType …").</exception>
        /// <exception cref="ArgumentException">The datetime metadata or the <c>StringDType</c> parameters do not match the
        /// class, the item size / type character is not one this class can have, or <paramref name="byteorder"/> is not one
        /// of <c>'=' '|' '&lt;' '&gt;'</c>.</exception>
        internal DType(DTypeMeta meta, char byteorder, DatetimeMetaData? datetime, int itemsize, char typeChar,
            VStringParameters vstring = null, IReadOnlyDictionary<string, object> metadata = null)
        {
            Meta = meta ?? throw new ArgumentNullException(nameof(meta));
            if (meta._abstract)
                throw new TypeError($"Cannot instantiate abstract DType {meta}");
            if (meta is DatetimeDTypeMeta != datetime.HasValue)
                throw new ArgumentException($"{meta} {(datetime.HasValue ? "takes no" : "requires")} datetime metadata.", nameof(datetime));
            // A StringDType descriptor IS its parameters (na_object, coerce): one without them would compare, hash and print
            // as no NumPy descriptor does, so the class and the parameters must come together.
            if (meta is VStringDTypeMeta != (vstring != null))
                throw new ArgumentException($"{meta} {(vstring != null ? "takes no" : "requires")} StringDType parameters.", nameof(vstring));
            _vstring = vstring;

            // Only a flexible class varies its element size per instance. Every other class has ONE layout, and
            // accepting a different size there would let a descriptor disagree with the storage lane it selects.
            if (meta is StringDTypeMeta str)
            {
                if (itemsize < 0 || itemsize % str.UnitSize != 0)
                    throw new ArgumentException($"{meta} cannot have an item size of {itemsize} bytes (a non-negative multiple of {str.UnitSize} is required).", nameof(itemsize));
                // 'c' is NumPy's legacy one-character S1 spelling (NPY_CHARLTR): it renders and compares as S1, but
                // keeps its own dtype.char.
                bool charSpelling = !str.IsUnicode && typeChar == (char)NPY_TYPECHAR.NPY_CHARLTR && itemsize == 1;
                if (typeChar != meta.TypeChar && !charSpelling)
                    throw new ArgumentException($"{meta} descriptors cannot use the type character '{typeChar}'.", nameof(typeChar));
            }
            else if (itemsize != meta.ItemSize || typeChar != meta.TypeChar)
            {
                throw new ArgumentException($"{meta} has a fixed layout ({meta.ItemSize} bytes, char '{meta.TypeChar}'); got {itemsize} bytes, char '{typeChar}'.");
            }

            _datetime = datetime;
            _metadata = metadata;
            _typeChar = typeChar;
            this.type = meta.type;
            this.typecode = meta.TypeCode;
            this.itemsize = itemsize;
            this.kind = meta.Kind;
            this.byteorder = NormalizeByteOrder(meta, byteorder);
            this.TYPECHAR = (NPY_TYPECHAR)typeChar;
            this.name = ComputeName();
        }

        /// <summary>
        ///     A NEP 55 <c>StringDType</c> descriptor with the given parameters (NumPy's <c>new_stringdtype_instance</c>): a
        ///     16-byte cell, alignment 8, byte order <c>'|'</c>, type character <c>'T'</c>.
        /// </summary>
        /// <param name="meta">The <c>StringDType</c> class.</param>
        /// <param name="parameters">The descriptor's <c>na_object</c> / <c>coerce</c> parameters.</param>
        /// <exception cref="ArgumentNullException"><paramref name="meta"/> or <paramref name="parameters"/> is null.</exception>
        internal DType(VStringDTypeMeta meta, VStringParameters parameters)
            : this(meta, NPY_IGNORE, null, VStringDTypeMeta.SIZEOF_NPY_PACKED_STATIC_STRING, 'T',
                parameters ?? throw new ArgumentNullException(nameof(parameters)))
        {
        }

        /// <summary>
        ///     The item size the fixed-size constructor fills in — the class's own. A flexible (string) class has no single
        ///     size, so building one of its descriptors without a size is a programming error, reported here rather than
        ///     silently producing an unsized descriptor.
        /// </summary>
        /// <param name="meta">The class being instantiated (null is reported by the main constructor).</param>
        /// <returns>The class's item size (0 for a null class, which the main constructor then rejects).</returns>
        /// <exception cref="ArgumentException"><paramref name="meta"/> is a string class.</exception>
        private static int ClassItemSize(DTypeMeta meta)
        {
            if (meta is StringDTypeMeta)
                throw new ArgumentException($"{meta} is flexible: its descriptors need an explicit item size.", nameof(meta));
            return meta?.ItemSize ?? 0;
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
        ///     The byte order a descriptor STORES, as NumPy's descriptors store it: a class whose byte order is not applicable
        ///     — the single-byte types, byte strings (<c>S</c>, any width) and <c>StringDType</c> — is always <c>'|'</c>
        ///     (NumPy's descriptors for them carry <c>NPY_IGNORE</c>, which no byte-order request changes); for any other
        ///     class <c>'='</c> and <c>'|'</c> mean native and are stored as <c>'='</c>, and an explicit <c>'&lt;'</c> /
        ///     <c>'&gt;'</c> is stored as given — including the host's own order, which NumPy's
        ///     <c>PyArray_DescrNewByteorder</c> keeps explicit (<c>np.dtype('f8').newbyteorder('&lt;').byteorder</c> is
        ///     <c>'&lt;'</c>). The parser folds an explicit host order into <c>'='</c> before it gets here.
        /// </summary>
        /// <param name="meta">The class whose policy applies (<see cref="DTypeMeta.ByteOrderNotApplicable"/>).</param>
        /// <param name="byteorder">The requested byte order.</param>
        /// <returns>The byte order to store.</returns>
        /// <exception cref="ArgumentException"><paramref name="byteorder"/> is not one of <c>'=' '|' '&lt;' '&gt;'</c>.</exception>
        private static char NormalizeByteOrder(DTypeMeta meta, char byteorder)
        {
            if (meta.ByteOrderNotApplicable)
                return NPY_IGNORE;
            switch (byteorder)
            {
                case NPY_NATIVE:
                case NPY_IGNORE:
                    return NPY_NATIVE;
                case NPY_LITTLE:
                case NPY_BIG:
                    return byteorder;
                default:
                    throw new ArgumentException($"byteorder not recognized (got '{byteorder}')", nameof(byteorder));
            }
        }

        /// <summary>
        ///     The byte order with <c>'='</c> spelled as the host's explicit character — NumPy's <c>_normalize_byteorder</c>
        ///     (<c>hashdescr.c</c>), the form equality and hashing compare so that <c>'='</c> and an explicit host order are
        ///     one byte order while the opposite order stays distinct.
        /// </summary>
        private char CanonicalByteOrder => byteorder == NPY_NATIVE ? NPY_NATBYTE : byteorder;

        /// <summary>
        ///     NumPy's <c>_name_get</c>: <c>"bool"</c>; kind-stem plus bit count for the numeric builtins (<c>"int32"</c>,
        ///     <c>"float64"</c>, <c>"complex128"</c>) and for a SIZED string (<c>"str160"</c> for <c>U5</c>, <c>"bytes40"</c> for
        ///     <c>S5</c>); the bare stem for an unsized string (<c>"str"</c>, <c>"bytes"</c>); the scalar-type name for a
        ///     user-range class (<c>"decimal"</c>, <c>"char"</c>); the class name plus unit for datetimes (<c>"datetime64[ns]"</c>).
        /// </summary>
        /// <returns>The <see cref="name"/> string.</returns>
        /// <exception cref="RuntimeError">An unknown kind (an internal inconsistency).</exception>
        /// <remarks>
        ///     A non-legacy (NEP 42) class is named after the class itself plus the bit count, because its scalar type is not
        ///     flexible (<c>_name_get</c>: <c>type(dtype).__name__</c>, then <c>_name_includes_bit_suffix</c>): a
        ///     <c>StringDType</c> descriptor is <c>'StringDType128'</c>. This runs before the user-range branch, whose
        ///     threshold (<c>type_num &gt;= 256</c>) the NEP 55 class number 2056 also passes.
        /// </remarks>
        private string ComputeName()
        {
            if (!Meta._legacy)
                return Meta.Name + (long)itemsize * 8;
            if (Meta.type_num >= 256)
                return Meta.ScalarName; // user dtypes don't promise to do anything special
            if (Meta is DatetimeDTypeMeta)
                return Meta.ScalarName + _datetime.Value.ToString(skipBrackets: false);
            switch (kind)
            {
                case 'b': return "bool"; // implied
                // _name_includes_bit_suffix: an UNSIZED flexible dtype has no suffix. The bit count is computed in 64 bits
                // because Python's int does not overflow: 'S2147483647' is named 'bytes17179869176'.
                case 'U': return itemsize == 0 ? "str" : "str" + (long)itemsize * 8;
                case 'S': return itemsize == 0 ? "bytes" : "bytes" + (long)itemsize * 8;
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

        /// <summary>
        ///     The size of the dtype in bytes (<c>itemsize</c>). Per instance for the string classes (<c>S5</c> → 5,
        ///     <c>U5</c> → 20, the unsized <c>S0</c>/<c>U0</c> → 0); the class's fixed size for every other class.
        /// </summary>
        public readonly int itemsize;

        /// <summary>
        ///     NumPy's <c>dtype.name</c> spelling (<c>"float32"</c>, <c>"int64"</c>, <c>"bool"</c>, <c>"complex128"</c>,
        ///     <c>"datetime64[ns]"</c>) — the CLR <see cref="System.Reflection.MemberInfo.Name">Type.Name</see> (<c>"Single"</c>) is not a NumPy name and does not
        ///     round-trip through <see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/>. The NumSharp-only dtypes have no NumPy analog and get
        ///     lowercase names of their own (<c>"decimal"</c>, <c>"char"</c>) rather than the nearest NumPy stand-in.
        /// </summary>
        public readonly string name;

        /// <summary>
        ///     The C# scalar type of one element (NumPy's <c>dtype.type</c>): <see cref="string"/> for <c>U</c> (NumPy's
        ///     <c>np.str_</c>), <c>byte[]</c> for <c>S</c> (NumPy's <c>np.bytes_</c>); null for a class with no C# scalar yet
        ///     (datetime64 in Stage A).
        /// </summary>
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

        /// <summary>
        ///     A unique character code for each of the built-in types (<c>'?' 'b' 'B' 'h' 'H' 'i' 'I' 'l' 'L' 'e' 'f' 'd' 'D'
        ///     'M' 'm' 'S' 'U'</c>). Stored per descriptor, as in NumPy: the legacy spelling <c>np.dtype('c')</c> is an
        ///     <c>S1</c> whose <c>char</c> is <c>'c'</c> while <c>np.dtype('S1').char</c> is <c>'S'</c> (the two still compare equal).
        /// </summary>
        public char @char => _typeChar;

        /// <summary>The per-descriptor <c>dtype.char</c> (see <see cref="char"/>).</summary>
        private readonly char _typeChar;

        /// <summary>NumPy's unique type number (<c>dtype.num</c>) as a raw <see cref="int"/> — exactly what NumPy's Python surface returns.</summary>
        public int num => Meta.type_num;

        /// <summary>
        ///     The same number as <see cref="num"/>, typed as NumPy's C <see cref="NPY_TYPES"/> enum — the NumPy-facing
        ///     dtype IDENTITY that rides alongside the storage/kernel discriminator <see cref="typecode"/>
        ///     (<see cref="NPTypeCode"/>). Byte-identical to <c>np.dtype(...).num</c> on the host it runs on, including
        ///     the platform-dependent integer resolution (on Windows/LLP64 <c>int32.type_num</c> is
        ///     <see cref="NPY_TYPES.NPY_LONG"/> and <c>int64.type_num</c> is <see cref="NPY_TYPES.NPY_LONGLONG"/>; on LP64
        ///     they are <see cref="NPY_TYPES.NPY_INT"/> and <see cref="NPY_TYPES.NPY_LONG"/>). NumSharp's two non-NumPy
        ///     dtypes report the user-range <see cref="NPY_TYPES.NUMSHARP_DECIMAL"/> / <see cref="NPY_TYPES.NUMSHARP_CHAR"/>.
        /// </summary>
        public NPY_TYPES type_num => (NPY_TYPES)Meta.type_num;

        /// <summary>The required alignment (bytes) of this data-type (<c>dtype.alignment</c>).</summary>
        public int alignment => Meta.Alignment;

        /// <summary>The instance parameters of a <c>datetime64</c>/<c>timedelta64</c> descriptor (NumPy's <c>c_metadata</c>); null for every other class.</summary>
        public DatetimeMetaData? DatetimeMetadata => _datetime;

        /// <summary>
        ///     The array-interface typestring (<c>dtype.str</c>): explicit byte order, kind and size — <c>"&lt;i4"</c>,
        ///     <c>"|b1"</c>, <c>"&gt;f8"</c>, <c>"&lt;M8[ns]"</c>, <c>"&lt;m8"</c>, and for strings the CHARACTER count
        ///     (<c>"|S5"</c> is 5 bytes, <c>"&lt;U5"</c> is 5 code points = 20 bytes; the unsized forms are <c>"|S0"</c>/<c>"&lt;U0"</c>).
        /// </summary>
        public string str
        {
            get
            {
                // A non-legacy class's typestring is its str() (arraydescr_protocol_typestr_get): 'StringDType()'.
                if (_vstring != null)
                    return _vstring.Repr();
                char endian = byteorder == NPY_NATIVE ? NPY_NATBYTE : byteorder;
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
        ///     NumSharp-only Decimal/Char), 0 for any other instance — a datetime descriptor, a non-native byte order, a
        ///     copy made with the public constructors, or any descriptor of a non-legacy class (<c>StringDType</c>: NumPy's
        ///     getter reads the legacy <c>fields</c> slot, which a new-style descriptor does not have).
        /// </summary>
        public int isbuiltin => !Meta._legacy ? 0 : Meta.type_num >= 256 ? 2 : (ReferenceEquals(this, Meta.singleton) ? 1 : 0);

        /// <summary>
        ///     <c>dtype.isnative</c> (NumPy's <c>PyArray_ISNBO</c>): true unless the byte order is the host's OPPOSITE order —
        ///     so <c>'='</c>, <c>'|'</c> and an explicit host order (<c>'&lt;'</c> on a little-endian machine) are all native.
        /// </summary>
        public bool isnative => byteorder != NPY_OPPBYTE;

        /// <summary>
        ///     <c>dtype.hasobject</c>: whether items hold references that need clearing — true for <c>StringDType</c> (a cell
        ///     can own heap memory its array must free), false for every other class.
        /// </summary>
        public bool hasobject => Meta.HasReferences;

        /// <summary>
        ///     <c>dtype.flags</c>: the item flags (NumPy's <c>NPY_ITEM_REFCOUNT | NPY_NEEDS_INIT | …</c>) — 107 for
        ///     <c>StringDType</c>, 8 (<c>NPY_NEEDS_INIT</c>) for <c>U</c>, 0 otherwise (<see cref="DTypeMeta.ItemFlags"/>).
        /// </summary>
        public int flags => Meta.ItemFlags;

        /// <summary>
        ///     <c>dtype.metadata</c>: the user metadata attached with <c>np.dtype(..., metadata=...)</c>, or null when none
        ///     was (NumPy's <c>None</c>). Read-only, like NumPy's <c>mappingproxy</c>; an empty attachment is an empty
        ///     dictionary, not null. Metadata never takes part in equality, hashing, <c>str</c> or <c>repr</c>.
        /// </summary>
        /// <remarks>
        ///     Keys are strings (NumPy accepts any hashable key; string keys are what NumPy's own users attach — h5py's
        ///     <c>'vlen'</c>, <c>'h5py_encoding'</c>). A <see cref="newbyteorder(string)"/> or <see cref="DTypeMeta.ensure_canonical"/>
        ///     copy keeps it; promotion of non-parametric classes drops it (the result is the class singleton), as in NumPy.
        /// </remarks>
        public IReadOnlyDictionary<string, object> metadata => _metadata;

        /// <summary>
        ///     <c>dtype.isalignedstruct</c>: whether a STRUCTURED dtype was built with <c>align=True</c> (C-struct
        ///     padding). False for every NumSharp descriptor, since none is structured — NumPy answers False for every
        ///     non-structured dtype too.
        /// </summary>
        public bool isalignedstruct => false;

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

        // ---- the StringDType parameters (NEP 55) ----

        /// <summary>
        ///     The parameters of a <c>StringDType</c> descriptor; null for every other class. Internal readers (the cast and
        ///     store paths) take the cached derived fields from here instead of re-running Python protocols per element.
        /// </summary>
        internal VStringParameters VString => _vstring;

        /// <summary>
        ///     True when this is a <c>StringDType</c> descriptor that has a missing-data object — the C# stand-in for
        ///     <c>hasattr(dtype, 'na_object')</c>, which tells "no sentinel" apart from a <c>None</c> sentinel (both make
        ///     <see cref="na_object"/> look like <see langword="null"/>, but only one of them exists).
        /// </summary>
        public bool HasNAObject => _vstring?.HasNAObject == true;

        /// <summary>
        ///     <c>dtype.na_object</c> of a <c>StringDType</c> descriptor: the missing-data sentinel it was built with
        ///     (<see langword="null"/> is Python's <c>None</c>; a <see cref="char"/> sentinel reads back as a one-character
        ///     <see cref="string"/>). Use <see cref="HasNAObject"/> to test for it without the exception.
        /// </summary>
        /// <exception cref="AttributeError">The descriptor has no missing-data object, or is not a <c>StringDType</c>:
        ///     NumPy's <c>'numpy.dtypes.StringDType' object has no attribute 'na_object'</c>.</exception>
        public object na_object
        {
            get
            {
                if (_vstring is null || !_vstring.HasNAObject)
                    throw new AttributeError($"'{Meta.FullName}' object has no attribute 'na_object'");
                return _vstring.na_object;
            }
        }

        /// <summary>
        ///     <c>dtype.coerce</c> of a <c>StringDType</c> descriptor: whether a stored value that is not a <c>str</c> is
        ///     converted to its <c>str()</c> (true, the default) or refused with NumPy's <c>ValueError</c> (false).
        /// </summary>
        /// <exception cref="AttributeError">The descriptor is not a <c>StringDType</c>: NumPy's
        ///     <c>'numpy.dtypes.Float64DType' object has no attribute 'coerce'</c>.</exception>
        public bool coerce
        {
            get
            {
                if (_vstring is null)
                    throw new AttributeError($"'{Meta.FullName}' object has no attribute 'coerce'");
                return _vstring.coerce;
            }
        }

        // ---- the Python container protocol NumPy gives a dtype (its fields) ----

        /// <summary>
        ///     Python's <c>len(dtype)</c>: the number of FIELDS of a structured dtype — 0 for every NumSharp descriptor,
        ///     which is NumPy's answer for every non-structured dtype too (<c>len(np.dtype('f8')) == 0</c>). It is not the
        ///     item size; use <see cref="itemsize"/> for that.
        /// </summary>
        /// <returns>0.</returns>
        public int __len__() => 0;

        /// <summary>
        ///     Python's <c>dtype[key]</c>: a field lookup on a structured dtype. Every NumSharp descriptor is
        ///     non-structured, so every key raises NumPy's <c>KeyError</c>, whatever its type (<c>dt[0]</c>,
        ///     <c>dt['x']</c> and <c>dt[None]</c> all raise the same).
        /// </summary>
        /// <param name="key">The field name or index NumPy would look up; never inspected.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="KeyError">
        ///     Always — NumPy's <c>descr_subscript</c> text, <c>There are no fields in dtype float64.</c> (the dtype as
        ///     <c>str(dtype)</c> renders it: <c>&gt;f8</c>, <c>datetime64[ns]</c>), quoted as Python's <c>str(KeyError)</c>
        ///     shows it — the convention <c>np.typename</c>'s <see cref="KeyError"/> follows too.
        /// </exception>
        public DType __getitem__(object key)
            => throw new KeyError(PyLiteral.Repr($"There are no fields in dtype {ToString()}."));

        // ---- factories (public ctors are copies; these return the canonical instance) ----

        /// <summary>The descriptor of a C# <see cref="System.Type"/> — the class's canonical instance (<c>np.dtype(typeof(int))</c>).</summary>
        public static DType From(Type type)
        {
            if (type == null)
                throw new ArgumentNullException(nameof(type));
            return MetaOf(type).singleton;
        }

        /// <summary>
        ///     The descriptor of an <see cref="NPTypeCode"/> — the class's canonical instance. Every NumSharp API that takes a
        ///     <see cref="DType"/> also takes an <see cref="NPTypeCode"/> through this conversion.
        /// </summary>
        /// <param name="typecode">A storage lane with a fixed element type.</param>
        /// <returns>The lane's descriptor.</returns>
        /// <exception cref="ArgumentException"><paramref name="typecode"/> is <see cref="NPTypeCode.Empty"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="typecode"/> is <see cref="NPTypeCode.String"/>: the
        ///     fixed-width string lane names no dtype (it holds <c>S</c> and <c>U</c> arrays of every width), so code that
        ///     allocates "an array like <c>a</c>" by <c>a.typecode</c> would silently get a one-character <c>&lt;U1</c> and
        ///     write <c>a</c>'s wider elements past its end. Refusing here turns every such allocation — an operation that
        ///     does not support strings yet — into an error; a string-aware operation passes <c>a.dtype</c>.</exception>
        public static DType From(NPTypeCode typecode)
        {
            if (typecode == NPTypeCode.String)
                throw new NotSupportedException(StringLaneHasNoDType);
            return MetaOf(typecode).singleton;
        }

        /// <summary>The refusal of <see cref="From(NPTypeCode)"/> for the string lane (shared with the storage allocator).</summary>
        internal const string StringLaneHasNoDType =
            "This operation does not support fixed-width string arrays: it asked for a dtype by storage lane " +
            "(NPTypeCode.String), which carries neither the string class (bytes 'S' / str 'U') nor the width. " +
            "Use the array's own dtype (a.dtype), a dtype string such as 'U5' or 'S3', or typeof(string) / np.str_ for NumPy's default '<U1'.";

        /// <summary>Builds a descriptor from a NumPy dtype <b>string</b> (NumPy's case-sensitive grammar). Same as <see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/>.</summary>
        public static DType From(string dtype) => np.dtype(dtype);

        /// <summary>
        ///     NumPy's <c>PyArray_DescrFromType</c>: the builtin descriptor for a NumPy type number — the reverse of
        ///     <see cref="type_num"/>. Every C-integer number has a class of its own, as in NumPy: on Windows
        ///     <see cref="NPY_TYPES.NPY_INT"/> is <c>intc</c> (char <c>'i'</c>) beside <see cref="NPY_TYPES.NPY_LONG"/>'s
        ///     <c>int32</c>, on Linux <see cref="NPY_TYPES.NPY_LONGLONG"/> is <c>longlong</c> (char <c>'q'</c>) beside
        ///     <see cref="NPY_TYPES.NPY_LONG"/>'s <c>int64</c>, so the returned descriptor's <see cref="num"/> is always the
        ///     number asked for. <see cref="NPY_TYPES.NPY_LONGDOUBLE"/> / <see cref="NPY_TYPES.NPY_CLONGDOUBLE"/> are NumSharp's
        ///     8 / 16-byte <c>longdouble</c> / <c>clongdouble</c>. Numbers with NO NumSharp class
        ///     (<see cref="NPY_TYPES.NPY_CFLOAT"/> complex64, <see cref="NPY_TYPES.NPY_OBJECT"/>, <see cref="NPY_TYPES.NPY_VOID"/>,
        ///     the sentinels) are rejected, matching how the parser refuses a dtype NumSharp cannot represent.
        /// </summary>
        /// <param name="type_num">A NumPy type number.</param>
        /// <returns>The class's canonical instance (for the datetime pair, the generic-unit descriptor).</returns>
        /// <exception cref="NotSupportedException">NumSharp has no class of this number (complex64, object, void, or a sentinel).</exception>
        public static DType From(NPY_TYPES type_num)
        {
            var meta = DTypeRegistry.FromTypeNum((int)type_num);
            if (meta == null)
                throw new NotSupportedException(
                    $"NumPy type number {(int)type_num} (NPY_TYPES.{type_num}) has no NumSharp dtype class: " +
                    "NumSharp has no complex64/object/void types, a single complex width, and no void storage.");
            return meta.singleton;
        }

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
        public static DType Boolean => DTypeRegistry.Bool.singleton;
        /// <summary>The <see cref="System.Byte"/> (uint8) descriptor.</summary>
        public static DType Byte => DTypeRegistry.UInt8.singleton;
        /// <summary>The <see cref="System.SByte"/> (int8) descriptor.</summary>
        public static DType SByte => DTypeRegistry.Int8.singleton;
        /// <summary>The <see cref="System.Int16"/> descriptor.</summary>
        public static DType Int16 => DTypeRegistry.Int16.singleton;
        /// <summary>The <see cref="System.UInt16"/> descriptor.</summary>
        public static DType UInt16 => DTypeRegistry.UInt16.singleton;
        /// <summary>The <see cref="System.Int32"/> descriptor.</summary>
        public static DType Int32 => DTypeRegistry.Int32.singleton;
        /// <summary>The <see cref="System.UInt32"/> descriptor.</summary>
        public static DType UInt32 => DTypeRegistry.UInt32.singleton;
        /// <summary>The <see cref="System.Int64"/> descriptor.</summary>
        public static DType Int64 => DTypeRegistry.Int64.singleton;
        /// <summary>The <see cref="System.UInt64"/> descriptor.</summary>
        public static DType UInt64 => DTypeRegistry.UInt64.singleton;
        /// <summary>The <see cref="System.Char"/> descriptor.</summary>
        public static DType Char => DTypeRegistry.Char.singleton;
        /// <summary>The <see cref="System.Half"/> (float16) descriptor.</summary>
        public static DType Half => DTypeRegistry.Half.singleton;
        /// <summary>The <see cref="System.Single"/> (float32) descriptor.</summary>
        public static DType Single => DTypeRegistry.Single.singleton;
        /// <summary>The <see cref="System.Double"/> (float64) descriptor.</summary>
        public static DType Double => DTypeRegistry.Double.singleton;
        /// <summary>The <see cref="System.Decimal"/> descriptor.</summary>
        public static DType Decimal => DTypeRegistry.Decimal.singleton;
        /// <summary>The <see cref="System.Numerics.Complex"/> (complex128) descriptor.</summary>
        public static DType Complex => DTypeRegistry.Complex128.singleton;

        // ---- implicit conversions: DType is the single spelling Type / NPTypeCode / NumPy-string collapse into ----
        //
        // LOAD-BEARING — do not remove these operators. Every dtype-taking overload across the library declares ONE
        // `DType` parameter; these conversions are what let a caller pass a `Type`, an `NPTypeCode` (or `NPTypeCode?`),
        // or a dtype string to it UNCHANGED — the argument converts implicitly and binds that single overload. The
        // whole np.* / NDArray surface AND the interop packages (System.Numerics.Tensors / OnnxRuntime / MLNet /
        // pythonnet dtype maps take `DType` and rely on `nd.typecode` / `NPTypeCode.X` converting to it) depend on this
        // for source-compatibility, so dropping an operator silently breaks every such call site. The forward direction
        // (Type / NPTypeCode / string -> DType) is implicit by design; the reverse (DType -> Type) is EXPLICIT — see
        // the operator below for why.

        /// <summary>A C# <see cref="System.Type"/> converts to its canonical descriptor (<see langword="null"/> ⇒ none).</summary>
        public static implicit operator DType(Type type) => type == null ? null : From(type);

        /// <summary>An <see cref="NPTypeCode"/> converts to its canonical descriptor (<see cref="NPTypeCode.Empty"/> ⇒ none).</summary>
        public static implicit operator DType(NPTypeCode typecode) => typecode == NPTypeCode.Empty ? null : From(typecode);

        /// <summary>A nullable <see cref="NPTypeCode"/> converts to a descriptor (<see langword="null"/>/<see cref="NPTypeCode.Empty"/> ⇒ none).</summary>
        public static implicit operator DType(NPTypeCode? typecode) => typecode.HasValue ? (DType)typecode.Value : null;

        /// <summary>
        ///     A NumPy dtype <b>string</b> converts to a descriptor via <see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/> — NumPy's exact,
        ///     case-sensitive spelling (<c>"f4"</c>, <c>"float32"</c>, <c>"&lt;f8"</c>, <c>"M8[ns]"</c>). <see langword="null"/> ⇒ none.
        /// </summary>
        public static implicit operator DType(string dtype) => dtype == null ? null : np.dtype(dtype);

        /// <summary>
        ///     A NumPy type number (<see cref="NPY_TYPES"/>) converts to its canonical descriptor — the fifth dtype
        ///     spelling, so <c>np.zeros(3, NPY_TYPES.NPY_DOUBLE)</c> binds the one <c>DType</c> overload like the others.
        ///     Delegates to <see cref="From(NPY_TYPES)"/>, so a number with no NumSharp class on this platform
        ///     (e.g. <see cref="NPY_TYPES.NPY_CFLOAT"/>) throws <see cref="NotSupportedException"/> — the same
        ///     throw-on-invalid the string conversion has (a bad <c>"complex64"</c> string). Unlike the reference-type
        ///     spellings there is no <see langword="null"/> case: <see cref="NPY_TYPES"/> is a value type, which is also
        ///     why it cannot reintroduce the <c>null</c>-literal binding hazard that bars <c>==(DType, string)</c>
        ///     operators. The reverse direction is the <see cref="type_num"/> property (not an operator, to keep
        ///     <c>DType</c>'s back-conversions to <see cref="System.Type"/>/<see cref="NPTypeCode"/> unambiguous).
        /// </summary>
        /// <exception cref="NotSupportedException">The number has no NumSharp dtype class on the current platform.</exception>
        public static implicit operator DType(NPY_TYPES type_num) => From(type_num);

        /// <summary>
        ///     A descriptor converts back to its <see cref="System.Type"/> (none/<see langword="null"/> ⇒ null) — EXPLICITLY:
        ///     <c>(Type)a.dtype</c>, or the NumPy field <c>a.dtype.type</c>. It is deliberately not implicit, because
        ///     <see cref="System.Type"/> declares its own <c>==</c>: with conversions in BOTH directions,
        ///     <c>a.dtype == typeof(double)</c> would be ambiguous between <c>Type.==</c> and <c>DType.==</c> (CS0034), and adding
        ///     <c>==(DType, Type)</c> overloads instead would make <c>descr == null</c> ambiguous (neither target is better). With
        ///     only the <see cref="Type"/>→<see cref="DType"/> direction implicit, <c>a.dtype == typeof(double)</c>,
        ///     <c>typeof(double) == a.dtype</c>, <c>Assert.AreEqual(typeof(double), a.dtype)</c> (T infers as <see cref="DType"/>)
        ///     and <c>a.dtype.Should().Be(typeof(double))</c> all resolve to the structural, coercing DType equality.
        /// </summary>
        /// <exception cref="NotSupportedException">The class has no C# storage type yet (datetime64/timedelta64 in Stage A).</exception>
        public static explicit operator Type(DType dtype)
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

        /// <summary>
        ///     Structural equality — NumPy's <c>dtype.__eq__</c> (<c>PyArray_EquivTypes</c>) for the descriptors NumSharp has:
        ///     the same class, the same byte order once <c>'='</c> reads as the host's own order (so
        ///     <c>np.dtype('f8').newbyteorder('&lt;')</c> equals <c>np.dtype('f8')</c> on a little-endian host while
        ///     <c>&gt;f8</c> does not), the same item size (<c>U5 != U3</c>) and the same instance parameters
        ///     (<c>M8[ns] != M8[s]</c>). The type character is NOT compared, as in NumPy: <c>np.dtype('c') == np.dtype('S1')</c>.
        ///     Two <c>StringDType</c> descriptors are equal when they coerce alike and their missing-data objects are the same
        ///     missing value (NumPy's <c>stringdtype_richcompare</c>: <c>StringDType(na_object=nan) == StringDType(na_object=np.nan)</c>).
        ///     Metadata is ignored, as NumPy ignores it.
        /// </summary>
        /// <param name="other">The descriptor to compare with; null is unequal.</param>
        /// <returns>True when both describe the same element layout and parameters.</returns>
        /// <remarks>
        ///     <para>
        ///     Two DIFFERENT NumPy numeric classes of one kind and one size are equal too, as in NumPy, whose cast between
        ///     them is no cast at all: <c>np.dtype('g') == np.dtype('d')</c> (long double is a 64-bit double in NumSharp),
        ///     and on Windows <c>np.dtype(np.intc) == np.dtype(np.int32)</c>. NumSharp's own char and uint16 are not
        ///     (<see cref="LegacyBuiltinDTypeMeta.IsEquivalentNumeric"/>).
        ///     </para>
        ///     <para>
        ///     One known difference: NumPy's relation folds exact metric-prefix units one way only (<c>M8[1000ms] == M8[s]</c>
        ///     but not the reverse); this equality is symmetric and keeps them apart. <see cref="DTypeCasting.EquivTypes"/> is
        ///     NumPy's relation.
        ///     </para>
        /// </remarks>
        public bool Equals(DType other)
            // The identity test first and alone, so the common case (a singleton against itself) stays small enough to
            // inline into ==; the structural comparison is a call.
            => ReferenceEquals(this, other) || (other is not null && StructurallyEqual(other));

        /// <summary>
        ///     The structural half of <see cref="Equals(DType)"/>: the same class with the same canonical byte order, size
        ///     and parameters, or two equivalent NumPy numeric classes (<see cref="LegacyBuiltinDTypeMeta.IsEquivalentNumeric"/>:
        ///     one kind and one size, as <c>PyArray_EquivTypes</c> reads them) with the same canonical byte order.
        /// </summary>
        /// <param name="other">A descriptor that is not this one (and not null).</param>
        /// <returns>True when the two describe the same data.</returns>
        private bool StructurallyEqual(DType other)
        {
            // Every equal pair has one item size, so this rejects most unequal pairs (float64 against float32) before
            // any class test.
            if (itemsize != other.itemsize)
                return false;
            var mine = Meta;
            var theirs = other.Meta;
            if (ReferenceEquals(mine, theirs))
                return CanonicalByteOrder == other.CanonicalByteOrder && Nullable.Equals(_datetime, other._datetime)
                       && VStringParameters.AreEqual(_vstring, other._vstring);
            return mine is LegacyBuiltinDTypeMeta a && theirs is LegacyBuiltinDTypeMeta b && a.IsEquivalentNumeric(b)
                   && CanonicalByteOrder == other.CanonicalByteOrder;
        }

        /// <summary>
        ///     NumPy's coercing <c>dtype.__eq__</c>: the operand is converted to a descriptor first, so a
        ///     <see cref="System.Type"/> (<c>typeof(int)</c>), an <see cref="NPTypeCode"/> or a dtype string
        ///     (<c>"i4"</c>, <c>"int32"</c>) compare equal to the descriptor they denote; anything that is not a dtype
        ///     (a bad string, a <see cref="DTypeMeta"/> class object, null) compares unequal without raising. A
        ///     <c>StringDType</c> descriptor does NOT coerce: its <c>__eq__</c> returns <c>NotImplemented</c> for anything but
        ///     another <c>StringDType</c> instance, so <c>np.dtype('T') == 'T'</c> and <c>np.dtype('T') == str</c> are False.
        /// </summary>
        /// <param name="obj">The operand: a <see cref="DType"/>, a C# <see cref="System.Type"/>, an <see cref="NPTypeCode"/> or a dtype string.</param>
        /// <returns>True when <paramref name="obj"/> denotes an equal descriptor.</returns>
        /// <remarks>
        ///     NumPy reads <c>None</c> as <c>float64</c> here too (<c>np.dtype('f8') == None</c> is True), and NumSharp's
        ///     converters do (<c>np.dtype((DType)null)</c> and <c>np.result_type((DType)null)</c> are float64), but equality
        ///     cannot: <c>descr == null</c> is how the engine tests for a missing descriptor, so null is unequal to every
        ///     descriptor by design.
        /// </remarks>
        public override bool Equals(object obj)
        {
            if (_vstring != null && obj is not DType)
                return false; // stringdtype_richcompare: Py_TYPE(other) != Py_TYPE(self) -> NotImplemented -> identity -> False
            switch (obj)
            {
                case null:
                    return false;
                case DType d:
                    return Equals(d);
                case Type t:
                    return DTypeRegistry.FromScalarType(t) is { } meta && Equals(meta.singleton);
                case NPTypeCode tc:
                    return tc != NPTypeCode.Empty && DTypeRegistry.FromTypeCode(tc) is { } m2 && Equals(m2.singleton);
                case string s:
                    try
                    {
                        return Equals(np.dtype(s));
                    }
                    catch (Exception e) when (e is NotSupportedException || e is TypeError || e is ValueError || e is ArgumentException)
                    {
                        // NumPy's __eq__ returns False for a string that is not a dtype instead of raising.
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
        /// <param name="dtype">A NumPy dtype string; null or unparseable is unequal.</param>
        /// <returns>True when the string denotes an equal descriptor.</returns>
        public bool Equals(string dtype) => Equals((object)dtype);

        /// <summary>Coercing equality with a C# <see cref="System.Type"/> (<c>np.dtype('i4') == np.int32</c>); a non-NumSharp type is unequal.</summary>
        /// <param name="type">The C# element type; null or a non-NumSharp type is unequal.</param>
        /// <returns>True when <paramref name="type"/>'s canonical descriptor equals this one.</returns>
        public bool Equals(Type type) => Equals((object)type);

        /// <summary>Coercing equality with an <see cref="NPTypeCode"/>; <see cref="NPTypeCode.Empty"/> is unequal.</summary>
        /// <param name="typecode">The storage code.</param>
        /// <returns>True when the code's canonical descriptor equals this one.</returns>
        public bool Equals(NPTypeCode typecode) => Equals((object)typecode);

        /// <summary>
        ///     Consistent with <see cref="Equals(DType)"/>: the class and the canonical byte order — NumPy's
        ///     <c>_array_descr_builtin</c> hash, which reads <c>'='</c> as the host's order too
        ///     (<c>hash(np.dtype('f8').newbyteorder('&lt;')) == hash(np.dtype('f8'))</c>) and ignores the datetime unit
        ///     (<c>hash(M8[ns]) == hash(M8[s])</c>) — plus the item size for the flexible string classes (NumPy:
        ///     <c>hash(U5) != hash(U10)</c>). Fixed-size classes keep the two-field hash.
        /// </summary>
        /// <returns>The hash code.</returns>
        /// <remarks>
        ///     A <c>StringDType</c> descriptor hashes its parameters (NumPy hashes the tuple <c>(coerce, na_object)</c>). A
        ///     NumPy numeric class hashes its kind and size instead of its type number, as NumPy's <c>_array_descr_builtin</c>
        ///     does (<c>(kind, byteorder, flags, elsize, alignment)</c>), so the equal descriptors of two equivalent classes
        ///     (<c>hash(np.dtype('g')) == hash(np.dtype('d'))</c>) hash alike.
        /// </remarks>
        public override int GetHashCode()
            => _vstring != null ? HashCode.Combine(Meta.type_num, _vstring.Hash())
                : Meta is StringDTypeMeta ? HashCode.Combine(Meta.type_num, CanonicalByteOrder, itemsize)
                : Meta is LegacyBuiltinDTypeMeta { IsNumPyNumeric: true } ? HashCode.Combine(kind, itemsize, CanonicalByteOrder)
                : HashCode.Combine(Meta.type_num, CanonicalByteOrder);

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
        ///     <c>decimal</c>), the typestring for a non-native or flexible one (<c>&gt;i4</c>), and a <c>StringDType</c>
        ///     descriptor's own repr (<c>StringDType()</c>, <c>StringDType(na_object=None)</c>).
        /// </summary>
        /// <returns>The <c>str()</c> text.</returns>
        public override string ToString()
            => _vstring != null ? _vstring.Repr() : (!isnative || kind == 'U' || kind == 'S' || kind == 'V') ? str : name;

        /// <summary>
        ///     <paramref name="repr"/> false: <c>str(dtype)</c>. <paramref name="repr"/> true: NumPy's <c>repr(dtype)</c> —
        ///     <c>dtype('int32')</c>, <c>dtype('bool')</c>, <c>dtype('&gt;i4')</c>, <c>dtype('&lt;M8[ns]')</c>, and for the
        ///     NumSharp-only user-range classes NumPy's unquoted user form <c>dtype(decimal)</c>. A <c>StringDType</c>
        ///     descriptor prints the same text both ways, with no <c>dtype(…)</c> wrapper (<c>StringDType(coerce=False)</c>):
        ///     NumPy gives the class its own <c>tp_repr</c> and <c>tp_str</c>.
        /// </summary>
        /// <param name="repr">True for <c>repr(dtype)</c>, false for <c>str(dtype)</c>.</param>
        /// <returns>The text.</returns>
        public string ToString(bool repr)
            => _vstring != null ? _vstring.Repr() : repr ? "dtype(" + ConstructionRepr() + ")" : ToString();

        /// <summary>
        ///     NumPy's <c>_construction_repr</c> / <c>_scalar_str</c>: the argument <c>repr(dtype)</c> puts inside
        ///     <c>dtype(...)</c> — <c>'bool'</c>; <c>'S'</c> / <c>'S5'</c> for byte strings (no byte-order character — it is
        ///     never applicable); <c>'&lt;U'</c> / <c>'&lt;U5'</c> for Unicode (the byte order is explicit); the typestring
        ///     for datetimes and explicitly ordered numerics; the bare scalar name for the user-range classes
        ///     (<c>decimal</c>); otherwise the quoted <see cref="name"/>.
        /// </summary>
        /// <returns>The quoted (or, for a user class, bare) construction string.</returns>
        private string ConstructionRepr()
        {
            if (kind == 'b')
                return "'bool'";
            if (Meta is DatetimeDTypeMeta)
                return "'" + str + "'";
            if (Meta.type_num >= 256)
                return Meta.ScalarName;
            if (kind == 'S')
                return itemsize == 0 ? "'S'" : "'S" + itemsize + "'"; // _scalar_str: "'S%d'" % dtype.itemsize, no byte order
            if (kind == 'U' && itemsize == 0)
                return "'" + (byteorder == NPY_NATIVE ? NPY_NATBYTE : byteorder) + "U'"; // NumPy's unsized `dtype('<U')`
            // _scalar_str takes the short '<f8' form whenever the STORED byte order is explicit — the opposite order, and
            // the host's own order when newbyteorder left it explicit (dtype('<f8')) — and the long 'float64' form only
            // for '=' (and '|'). isnative cannot decide this: an explicit host order is native yet prints short.
            if (kind == 'U' || kind == 'V' || (byteorder != NPY_NATIVE && byteorder != NPY_IGNORE))
                return "'" + str + "'";
            return "'" + name + "'";
        }

        // ---- byte order, copies, metadata ----

        /// <summary>
        ///     A copy of this descriptor carrying <paramref name="newByteOrder"/> — NumPy's <c>PyArray_DescrNew</c> followed
        ///     by a byte-order store. The copy keeps the unit and the metadata (NumPy shares the metadata dict with copies)
        ///     and is never the class singleton, so it reports <c>isbuiltin == 0</c>.
        /// </summary>
        /// <param name="newByteOrder">The byte order to store (normalised by the constructor: single-byte classes, byte
        ///     strings and <c>StringDType</c> stay <c>'|'</c>).</param>
        /// <returns>A fresh descriptor. The item size and type character are kept, so <c>U5.newbyteorder()</c> is
        ///     <c>&gt;U5</c> and <c>np.dtype('c').newbyteorder()</c> still reports <c>char == 'c'</c>; a <c>StringDType</c>
        ///     descriptor keeps its parameters.</returns>
        /// <exception cref="ArgumentException"><paramref name="newByteOrder"/> is not one of <c>'=' '|' '&lt;' '&gt;'</c>.</exception>
        internal DType WithByteOrder(char newByteOrder) => new DType(Meta, newByteOrder, _datetime, itemsize, _typeChar, _vstring, _metadata);

        /// <summary>
        ///     A copy of this descriptor (NumPy's <c>PyArray_DescrNew</c>, what <c>np.dtype(d, copy=True)</c> makes of a
        ///     builtin singleton): the same class, byte order, item size, type character, unit, parameters and metadata in
        ///     a fresh object.
        /// </summary>
        /// <returns>A fresh descriptor equal to this one, reporting <c>isbuiltin == 0</c> for a builtin class.</returns>
        internal DType Copy() => new DType(Meta, byteorder, _datetime, itemsize, _typeChar, _vstring, _metadata);

        /// <summary>
        ///     A copy of this descriptor carrying <paramref name="metadata"/> instead of its own — the step of NumPy's
        ///     <c>arraydescr_new</c> that installs the merged metadata dict on the (already copied) descriptor.
        /// </summary>
        /// <param name="metadata">The complete metadata to carry (an immutable dictionary the caller built), or null.</param>
        /// <returns>A fresh descriptor.</returns>
        internal DType WithMetadata(IReadOnlyDictionary<string, object> metadata)
            => new DType(Meta, byteorder, _datetime, itemsize, _typeChar, _vstring, metadata);

        /// <summary>
        ///     Return a new dtype with a different byte order — NumPy's <c>dtype.newbyteorder</c>
        ///     (<c>PyArray_ByteorderConverter</c> + <c>PyArray_DescrNewByteorder</c>), statement for statement.
        /// </summary>
        /// <param name="new_order">
        ///     The byte order to force; only the FIRST character is read (<c>"little"</c>, <c>"lx"</c> and <c>"&lt;"</c> are
        ///     one request). <c>'S'</c>/<c>'s'</c> (the default) swaps; <c>'&lt;'</c>/<c>'L'</c>/<c>'l'</c> little-endian;
        ///     <c>'&gt;'</c>/<c>'B'</c>/<c>'b'</c> big-endian; <c>'='</c>/<c>'N'</c>/<c>'n'</c> native;
        ///     <c>'|'</c>/<c>'I'</c>/<c>'i'</c> leaves the byte order as it is.
        /// </param>
        /// <returns>
        ///     A NEW descriptor (never this one, never the class singleton: <c>isbuiltin == 0</c>), keeping the unit and the
        ///     metadata. A requested byte order is stored as given — <c>'&lt;'</c> on a little-endian host stays an explicit
        ///     <c>'&lt;'</c> (native, equal to <c>'='</c>, but <c>repr</c> shows <c>dtype('&lt;f8')</c>); a swap of a native
        ///     descriptor gives the opposite order and a swap of a non-native one gives the host's explicit order. A
        ///     single-byte class keeps <c>'|'</c> whatever is requested.
        /// </returns>
        /// <exception cref="TypeError"><paramref name="new_order"/> is null (NumPy's <c>byteorder must be str, not NoneType</c>);
        ///     or the descriptor is of a non-legacy class (<c>StringDType</c>): NumPy's
        ///     <c>Cannot use DescrNewByteOrder for this new style DTypes.</c>, raised after the order itself is parsed.</exception>
        /// <exception cref="ValueError"><paramref name="new_order"/> is empty or starts with another character (NumPy's <c>byteorder not recognized (got 'X')</c>, quoting the whole string).</exception>
        public DType newbyteorder(string new_order = "S")
        {
            if (new_order == null)
                throw new TypeError("byteorder must be str, not NoneType");
            // byteorder_parser reads only str[0]; an empty string is "not recognized", quoting the whole argument with %R.
            if (new_order.Length == 0 || !TryParseByteOrderCode(new_order[0], out char newendian))
                throw new ValueError($"byteorder not recognized (got {PyLiteral.Repr(new_order)})");
            return NewByteorder(newendian);
        }

        /// <summary>
        ///     <see cref="newbyteorder(string)"/> with the request as one C# character (a NumSharp convenience — Python has no
        ///     character type); the same codes and the same results.
        /// </summary>
        /// <param name="new_order">The byte-order code (see <see cref="newbyteorder(string)"/>).</param>
        /// <returns>A new descriptor with the requested byte order.</returns>
        /// <exception cref="ValueError"><paramref name="new_order"/> is not a byte-order code (NumPy's text).</exception>
        /// <exception cref="TypeError">A descriptor of a non-legacy class (<c>StringDType</c>).</exception>
        public DType newbyteorder(char new_order)
        {
            if (!TryParseByteOrderCode(new_order, out char newendian))
                throw new ValueError($"byteorder not recognized (got {PyLiteral.Repr(new_order.ToString())})");
            return NewByteorder(newendian);
        }

        /// <summary>
        ///     NumPy's <c>byteorder_parser</c>: maps the first character of a byte-order request onto a NumPy byte-order
        ///     character (<see cref="NPY_BIG"/>, <see cref="NPY_LITTLE"/>, <see cref="NPY_NATIVE"/>,
        ///     <see cref="NPY_IGNORE"/> or <see cref="NPY_SWAP"/>).
        /// </summary>
        /// <param name="c">The first character of the request.</param>
        /// <param name="endian">The NumPy byte-order character, when recognised.</param>
        /// <returns>False when <paramref name="c"/> is not a byte-order code.</returns>
        private static bool TryParseByteOrderCode(char c, out char endian)
        {
            switch (c)
            {
                case NPY_BIG:
                case NPY_LITTLE:
                case NPY_NATIVE:
                case NPY_IGNORE:
                    endian = c;
                    return true;
                case 'b':
                case 'B':
                    endian = NPY_BIG;
                    return true;
                case 'l':
                case 'L':
                    endian = NPY_LITTLE;
                    return true;
                case 'n':
                case 'N':
                    endian = NPY_NATIVE;
                    return true;
                case 'i':
                case 'I':
                    endian = NPY_IGNORE;
                    return true;
                case 's':
                case 'S':
                    endian = NPY_SWAP;
                    return true;
                default:
                    endian = '\0';
                    return false;
            }
        }

        /// <summary>
        ///     NumPy's <c>PyArray_DescrNewByteorder</c>: a copy whose byte order follows <paramref name="newendian"/> —
        ///     unchanged for a single-byte descriptor (<c>'|'</c>) and for <see cref="NPY_IGNORE"/>; swapped for
        ///     <see cref="NPY_SWAP"/> (a native descriptor takes <see cref="NPY_OPPBYTE"/>, a non-native one the host's
        ///     explicit <see cref="NPY_NATBYTE"/>); otherwise exactly <paramref name="newendian"/>.
        /// </summary>
        /// <param name="newendian">A NumPy byte-order character from <see cref="TryParseByteOrderCode"/>.</param>
        /// <returns>A fresh descriptor.</returns>
        /// <exception cref="TypeError">A descriptor of a non-legacy class (<c>StringDType</c>).</exception>
        private DType NewByteorder(char newendian)
        {
            // PyArray_DescrNewByteorder refuses every new-style descriptor; NumPy's argument converter has parsed the order
            // first, so a bad code is still the ValueError of the callers.
            if (!Meta._legacy)
                throw new TypeError("Cannot use DescrNewByteOrder for this new style DTypes.");
            char endian = byteorder;
            if (endian != NPY_IGNORE)
            {
                if (newendian == NPY_SWAP)
                    endian = isnative ? NPY_OPPBYTE : NPY_NATBYTE;
                else if (newendian != NPY_IGNORE)
                    endian = newendian;
            }
            return WithByteOrder(endian);
        }
    }
}
