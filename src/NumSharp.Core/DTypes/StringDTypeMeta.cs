using System;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's two fixed-width string DType classes — <c>numpy.dtypes.BytesDType</c> (kind <c>'S'</c>, type number 18,
    ///     one byte per character, NumPy's <c>np.bytes_</c>) and <c>numpy.dtypes.StrDType</c> (kind <c>'U'</c>, type number
    ///     19, one UCS-4 code point = 4 bytes per character, NumPy's <c>np.str_</c>). Both are PARAMETRIC: the length lives
    ///     on the descriptor (<c>S5</c> → <c>itemsize</c> 5, <c>U5</c> → 20), so promotion runs <see cref="CommonInstance"/>
    ///     (the longer string wins) and every cast resolves the destination length (<see cref="ToStringCastingImpl"/>,
    ///     <see cref="StringToStringCastingImpl"/>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>One storage lane, two classes.</b> Both classes back arrays through the single <see cref="NPTypeCode.String"/>
    ///     lane: the storage holds raw bytes (an <c>S</c> element is <c>itemsize</c> bytes, a <c>U</c> element
    ///     <c>itemsize / 4</c> native-endian <c>uint32</c> code points), NUL-padded, and the descriptor says how to read
    ///     them. Moving data (copy, view, broadcast, fancy indexing, string↔string casts) never decodes — it runs the
    ///     existing <c>uint8</c>/<c>uint32</c> kernels over a "unit view" with one extra trailing axis — so it is byte
    ///     exact by construction. Only element access (C# <see cref="string"/> / <c>byte[]</c> in and out) runs the codec
    ///     (<c>NDStringCodec</c>).
    ///     </para>
    ///     <para>
    ///     <b>The slots</b> are ports of <c>dtypemeta.c</c>: <c>string_and_unicode_default_descr</c> (a FRESH 1-character
    ///     descriptor, which is why <c>np.zeros(3, 'U')</c> is <c>&lt;U1</c>), <c>string_unicode_common_dtype</c>
    ///     (a string class absorbs every numeric class, <c>U</c> absorbs <c>S</c>, anything else defers),
    ///     <c>string_unicode_common_instance</c> (the longer descriptor, made canonical),
    ///     <c>string_discover_descr_from_pyobject</c> (a scalar's length) and <c>string_unicode_new</c>
    ///     (<c>StrDType(5)</c> → <c>dtype('&lt;U5')</c>).
    ///     </para>
    ///     <para>
    ///     <b>Registry policy.</b> <c>StrDType</c> owns <see cref="NPTypeCode.String"/> and <see cref="string"/>
    ///     (<c>(DType)NPTypeCode.String</c> and <c>np.dtype(typeof(string))</c> are <c>dtype('&lt;U')</c>);
    ///     <c>BytesDType</c> owns neither — its C# element <c>byte[]</c> must keep resolving to <c>uint8</c> the way every
    ///     C# array type resolves to its element class — so an <c>S</c> dtype is requested by name (<c>"S5"</c>,
    ///     <see cref="np.bytes_"/>, <see cref="np.dtypes.BytesDType"/>).
    ///     </para>
    /// </remarks>
    public sealed class StringDTypeMeta : DTypeMeta
    {
        /// <summary>
        ///     Builds one of the two string classes. Called once per class by <see cref="DTypeRegistry"/>; the registry
        ///     then installs the unsized singleton (<c>S0</c> / <c>U0</c>).
        /// </summary>
        /// <param name="isUnicode">True for <c>StrDType</c> (<c>U</c>), false for <c>BytesDType</c> (<c>S</c>).</param>
        internal StringDTypeMeta(bool isUnicode)
            : base(isUnicode ? "StrDType" : "BytesDType", "numpy.dtypes", isUnicode ? 19 : 18,
                isUnicode ? typeof(string) : typeof(byte[]), isUnicode ? "str_" : "bytes_", NPTypeCode.String,
                isUnicode ? 'U' : 'S', isUnicode ? 'U' : 'S', itemSize: 0, alignment: isUnicode ? 4 : 1,
                DTypeFlags.Legacy | DTypeFlags.Parametric)
        {
            IsUnicode = isUnicode;
        }

        /// <summary>True for <c>StrDType</c> (kind <c>'U'</c>, UCS-4 code points), false for <c>BytesDType</c> (kind <c>'S'</c>, bytes).</summary>
        public bool IsUnicode { get; }

        /// <summary>Bytes per character — the storage "unit": 4 for <c>U</c> (one <c>uint32</c> code point), 1 for <c>S</c>.</summary>
        public int UnitSize => IsUnicode ? 4 : 1;

        /// <summary>
        ///     The largest character count a descriptor of this class can hold — NumPy's limit that the item size fits a C
        ///     <c>int</c>: <c>2147483647</c> for <c>S</c>, <c>536870911</c> (<c>INT_MAX / 4</c>) for <c>U</c>.
        /// </summary>
        public int MaxChars => IsUnicode ? int.MaxValue / 4 : int.MaxValue;

        /// <summary>NumPy flags <c>U</c> items <c>NPY_NEEDS_INIT</c> (8) — NUL padding is part of the value; <c>S</c> reports 0.</summary>
        public override int ItemFlags => IsUnicode ? 8 : 0;

        /// <summary>A byte string is never byte-swapped, so every <c>S</c> descriptor reports <c>'|'</c>; <c>U</c> code points are.</summary>
        internal override bool ByteOrderNotApplicable => !IsUnicode;

        /// <summary><c>StrDType</c> is the canonical class of the shared <see cref="NPTypeCode.String"/> lane; <c>BytesDType</c> is not.</summary>
        internal override bool OwnsTypeCode => IsUnicode;

        /// <summary><c>byte[]</c> keeps resolving to <c>uint8</c>, so only <c>StrDType</c> registers its scalar type (<see cref="string"/>).</summary>
        internal override bool OwnsScalarType => IsUnicode;

        /// <summary>
        ///     A FRESH native descriptor of this class holding <paramref name="chars"/> characters — the building block of
        ///     every sized <c>S</c>/<c>U</c> descriptor (NumPy's <c>PyArray_DescrNewFromType</c> + <c>elsize</c>).
        /// </summary>
        /// <param name="chars">Characters per element (0 = NumPy's unsized <c>S0</c>/<c>U0</c>, as a non-builtin copy).</param>
        /// <returns>A descriptor with <c>itemsize == chars * UnitSize</c>; never the singleton, so its <c>isbuiltin</c> is 0.</returns>
        /// <exception cref="TypeError"><paramref name="chars"/> exceeds <see cref="MaxChars"/> (NumPy's
        /// <c>string of length N is too large to store inside array.</c>).</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="chars"/> is negative.</exception>
        public DType Descr(long chars)
        {
            if (chars < 0)
                throw new ArgumentOutOfRangeException(nameof(chars), chars, "A string length cannot be negative.");
            if (chars > MaxChars)
                throw new TypeError($"string of length {chars} is too large to store inside array.");
            return new DType(this, DType.NativeByteOrder, null, (int)(chars * UnitSize), TypeChar);
        }

        /// <summary>
        ///     <c>string_and_unicode_default_descr</c>: a FRESH one-character descriptor (<c>S1</c> / <c>U1</c>) — what a bare
        ///     class request turns into when an array is allocated (<c>np.zeros(3, 'U')</c> is <c>&lt;U1</c>,
        ///     <c>np.array([], dtype='S')</c> is <c>|S1</c>). It is NOT the unsized singleton, which only names the class.
        /// </summary>
        /// <returns>A new <c>S1</c> / <c>U1</c> descriptor.</returns>
        public override DType DefaultDescr() => Descr(1);

        /// <summary>
        ///     <c>string_unicode_common_dtype</c>: a string class is the common class of itself and any NUMERIC legacy class
        ///     (bool, the integers, the floats, complex — and NumSharp's Decimal/Char, which are numeric storage classes),
        ///     and <c>StrDType</c> absorbs <c>BytesDType</c>. Anything else — the datetime pair, the abstract NEP 50 literal
        ///     classes (<c>np.result_type(U5_array, 3)</c> raises), <c>BytesDType</c> asked about <c>StrDType</c> — defers
        ///     (<see langword="null"/>, NumPy's <c>NotImplemented</c>).
        /// </summary>
        /// <param name="other">The class to promote with.</param>
        /// <returns>This class, or null when it does not know <paramref name="other"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="other"/> is null.</exception>
        public override DTypeMeta CommonDType(DTypeMeta other)
        {
            if (other == null)
                throw new ArgumentNullException(nameof(other));
            if (ReferenceEquals(other, this))
                return this;
            bool otherIsNumber = other.IsLegacy && other.IsNumeric;
            bool unicodeOverBytes = IsUnicode && other is StringDTypeMeta otherString && !otherString.IsUnicode;
            /* Not numeric so defer unless cls is unicode and other is string */
            if (!other.IsLegacy || (!otherIsNumber && !unicodeOverBytes))
                return null;
            /*
             * The builtin types are ordered by complexity (aside from object) here.
             * Arguably, we should not consider numbers and strings "common", but
             * we currently do.
             */
            return this;
        }

        /// <summary>
        ///     <c>string_unicode_common_instance</c>: the LONGER of two descriptors of this class (the first on a tie), in
        ///     native byte order — <c>promote_types('&gt;U5', '&gt;U3')</c> is <c>&lt;U5</c>.
        /// </summary>
        /// <param name="descr1">A descriptor of this class.</param>
        /// <param name="descr2">A descriptor of this class.</param>
        /// <returns>The canonical form of the longer one.</returns>
        /// <exception cref="ArgumentNullException">Either descriptor is null.</exception>
        public override DType CommonInstance(DType descr1, DType descr2)
        {
            if (descr1 == null) throw new ArgumentNullException(nameof(descr1));
            if (descr2 == null) throw new ArgumentNullException(nameof(descr2));
            return descr1.itemsize >= descr2.itemsize ? EnsureCanonical(descr1) : EnsureCanonical(descr2);
        }

        /// <summary>
        ///     <c>string_discover_descr_from_pyobject</c>: the descriptor a C# scalar needs to be stored whole — a
        ///     <see cref="string"/> holds as many characters as it has CODE POINTS (Python's <c>len(str)</c>: a surrogate
        ///     pair is one, a lone surrogate is one), a <c>byte[]</c> as many as it has bytes. For <c>U</c> the item size is
        ///     4 bytes per character.
        /// </summary>
        /// <param name="obj">A <see cref="string"/> or <c>byte[]</c> element value.</param>
        /// <returns>A fresh sized descriptor of this class.</returns>
        /// <exception cref="TypeError">The value is too long for a descriptor of this class (NumPy's
        /// <c>string too large to store inside array.</c>).</exception>
        /// <exception cref="NotSupportedException">Any other value: discovering a length from <c>str(value)</c> (a number
        /// stored into a string array) is NumPy's number→text conversion, which arrives with the computational phase.</exception>
        public override DType DiscoverDescrFromObject(object obj)
        {
            long length;
            switch (obj)
            {
                case string s:
                    length = NDStringCodec.CodePointCount(s);
                    break;
                case byte[] b:
                    length = b.Length;
                    break;
                default:
                    throw new StringComputationNotSupportedException(
                        $"Cannot discover a {Name} length from a {obj?.GetType().Name ?? "null"} value: converting numbers to text " +
                        "(NumPy's str(value) discovery) is not part of NumSharp's fixed-width string storage phase yet.");
            }
            if (length > MaxChars)
                throw new TypeError("string too large to store inside array.");
            return Descr(length);
        }

        /// <summary>
        ///     <c>string_known_scalar_types</c>, narrowed to what this phase stores: a C# <see cref="string"/> (Python
        ///     <c>str</c>) and a <c>byte[]</c> (Python <c>bytes</c>) are scalars of either string class. NumPy also accepts
        ///     the Python numbers (stored as their <c>str()</c>); NumSharp refuses those until number↔text conversion lands.
        /// </summary>
        /// <param name="type">The C# type of a candidate element value.</param>
        /// <returns>True for <see cref="string"/> and <c>byte[]</c>.</returns>
        public override bool IsKnownScalarType(Type type) => type == typeof(string) || type == typeof(byte[]);

        /// <summary>
        ///     The no-argument class call <c>StrDType()</c>: NumPy's string classes require the length argument, so this
        ///     raises NumPy's verbatim <c>function takes exactly 1 positional argument (0 given)</c>.
        /// </summary>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError">Always.</exception>
        public override DType Instantiate() => throw new TypeError("function takes exactly 1 positional argument (0 given)");

        /// <summary>
        ///     The class call with a length — NumPy's <c>string_unicode_new</c>: <c>np.dtypes.StrDType(5)</c> is
        ///     <c>dtype('&lt;U5')</c>, <c>np.dtypes.BytesDType(3)</c> is <c>dtype('S3')</c>, and a length of 0 is a FRESH unsized
        ///     descriptor (<c>dtype('&lt;U')</c>, <c>isbuiltin == 0</c>).
        /// </summary>
        /// <param name="size">The length in characters (NumPy's Python <c>int</c> argument).</param>
        /// <returns>A fresh native descriptor.</returns>
        /// <exception cref="ValueError">A negative length: <c>Strings cannot have a negative size but a size of -1 was given</c>.</exception>
        /// <exception cref="TypeError">A length whose item size does not fit a C <c>int</c>: <c>Strings too large to store inside array.</c></exception>
        public DType Instantiate(long size)
        {
            if (size < 0)
                throw new ValueError($"Strings cannot have a negative size but a size of {size} was given");
            if (IsUnicode)
            {
                // unicode strings are 4 bytes per character; npy_mul_sizes_with_overflow on intp
                if (size > long.MaxValue / 4)
                    throw new TypeError("Strings too large to store inside array.");
                size *= 4;
            }
            if (size > int.MaxValue)
                throw new TypeError("Strings too large to store inside array.");
            return new DType(this, DType.NativeByteOrder, null, (int)size, TypeChar);
        }
    }
}
