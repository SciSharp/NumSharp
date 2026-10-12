using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     <c>numpy.dtypes</c> (NEP 56 / NumPy 1.25+): the DType CLASSES — <c>np.dtypes.Float64DType</c> is
        ///     <c>type(np.dtype('f8'))</c>. Each member is the live <see cref="DTypeMeta"/> singleton; its
        ///     <see cref="DTypeMeta.Instantiate"/> is NumPy's class call (<c>Int8DType()</c> → <c>dtype('int8')</c>;
        ///     <c>DateTime64DType()</c> raises the "Preliminary-API … can only be instantiated using `np.dtype(...)`"
        ///     <see cref="TypeError"/>), and <see cref="DType.Meta"/> is <c>type(dtype)</c>.
        /// </summary>
        /// <remarks>
        ///     Names and aliases are NumPy 2.4.2's <c>numpy.dtypes.__all__</c>: <c>ByteDType</c>/<c>UByteDType</c> for int8/uint8,
        ///     <c>ShortDType</c>/<c>UShortDType</c> for int16/uint16, <c>IntDType</c>/<c>UIntDType</c> for int32/uint32,
        ///     <c>LongLongDType</c>/<c>ULongLongDType</c> for int64/uint64, and the platform-dependent <c>LongDType</c>/<c>ULongDType</c>
        ///     (C <c>long</c>: 32-bit on Windows and 32-bit hosts, 64-bit on LP64 Unix). <c>LongDoubleDType</c> and
        ///     <c>CLongDoubleDType</c> alias the 64-bit float / complex classes — NumSharp has no extended precision, exactly
        ///     as NumPy's win-amd64 build. <c>StrDType</c> and <c>BytesDType</c> are the fixed-width string classes
        ///     (<see cref="StringDTypeMeta"/>). <c>StringDType</c> (NEP 55) is NumPy's one class you CALL with keyword
        ///     parameters; C# cannot make a member both a class and a call, so it is the pair of factory methods
        ///     <see cref="StringDType(bool)"/> / <see cref="StringDType(object, bool)"/>, and the class itself is
        ///     <see cref="DTypeRegistry.VString"/> (<c>type(dt)</c> is <see cref="DType.Meta"/>). The classes NumPy has and
        ///     NumSharp does not (<c>Complex64DType</c>, <c>ObjectDType</c>, <c>VoidDType</c>) are absent rather than stubbed;
        ///     <c>DecimalDType</c> and <c>CharDType</c> are NumSharp-only.
        /// </remarks>
        [ModuleName("np.dtypes")]
        public static class dtypes
        {
            /// <summary><c>numpy.dtypes.BoolDType</c>.</summary>
            public static LegacyBuiltinDTypeMeta BoolDType => DTypeRegistry.Bool;
            /// <summary><c>numpy.dtypes.Int8DType</c>.</summary>
            public static LegacyBuiltinDTypeMeta Int8DType => DTypeRegistry.Int8;
            /// <summary><c>numpy.dtypes.UInt8DType</c>.</summary>
            public static LegacyBuiltinDTypeMeta UInt8DType => DTypeRegistry.UInt8;
            /// <summary><c>numpy.dtypes.Int16DType</c>.</summary>
            public static LegacyBuiltinDTypeMeta Int16DType => DTypeRegistry.Int16;
            /// <summary><c>numpy.dtypes.UInt16DType</c>.</summary>
            public static LegacyBuiltinDTypeMeta UInt16DType => DTypeRegistry.UInt16;
            /// <summary><c>numpy.dtypes.Int32DType</c>.</summary>
            public static LegacyBuiltinDTypeMeta Int32DType => DTypeRegistry.Int32;
            /// <summary><c>numpy.dtypes.UInt32DType</c>.</summary>
            public static LegacyBuiltinDTypeMeta UInt32DType => DTypeRegistry.UInt32;
            /// <summary><c>numpy.dtypes.Int64DType</c>.</summary>
            public static LegacyBuiltinDTypeMeta Int64DType => DTypeRegistry.Int64;
            /// <summary><c>numpy.dtypes.UInt64DType</c>.</summary>
            public static LegacyBuiltinDTypeMeta UInt64DType => DTypeRegistry.UInt64;
            /// <summary><c>numpy.dtypes.Float16DType</c>.</summary>
            public static LegacyBuiltinDTypeMeta Float16DType => DTypeRegistry.Half;
            /// <summary><c>numpy.dtypes.Float32DType</c>.</summary>
            public static LegacyBuiltinDTypeMeta Float32DType => DTypeRegistry.Single;
            /// <summary><c>numpy.dtypes.Float64DType</c>.</summary>
            public static LegacyBuiltinDTypeMeta Float64DType => DTypeRegistry.Double;
            /// <summary><c>numpy.dtypes.Complex128DType</c>.</summary>
            public static LegacyBuiltinDTypeMeta Complex128DType => DTypeRegistry.Complex128;
            /// <summary><c>numpy.dtypes.DateTime64DType</c> — parametric; descriptor-level in Stage A.</summary>
            public static DatetimeDTypeMeta DateTime64DType => DTypeRegistry.DateTime64;
            /// <summary><c>numpy.dtypes.TimeDelta64DType</c> — parametric; descriptor-level in Stage A.</summary>
            public static DatetimeDTypeMeta TimeDelta64DType => DTypeRegistry.TimeDelta64;
            /// <summary>
            ///     <c>numpy.dtypes.StrDType</c> — fixed-width Unicode (<c>U</c>), parametric in the length:
            ///     <c>StrDType.Instantiate(5)</c> is NumPy's <c>StrDType(5)</c> → <c>dtype('&lt;U5')</c>.
            /// </summary>
            public static StringDTypeMeta StrDType => DTypeRegistry.Str;
            /// <summary>
            ///     <c>numpy.dtypes.BytesDType</c> — fixed-width bytes (<c>S</c>), parametric in the length:
            ///     <c>BytesDType.Instantiate(3)</c> is NumPy's <c>BytesDType(3)</c> → <c>dtype('S3')</c>.
            /// </summary>
            public static StringDTypeMeta BytesDType => DTypeRegistry.Bytes;

            // ---- C-name aliases (numpy.dtypes exports each builtin under its C name too) ----

            /// <summary><c>numpy.dtypes.ByteDType</c> — alias of <see cref="Int8DType"/>.</summary>
            public static LegacyBuiltinDTypeMeta ByteDType => DTypeRegistry.Int8;
            /// <summary><c>numpy.dtypes.UByteDType</c> — alias of <see cref="UInt8DType"/>.</summary>
            public static LegacyBuiltinDTypeMeta UByteDType => DTypeRegistry.UInt8;
            /// <summary><c>numpy.dtypes.ShortDType</c> — alias of <see cref="Int16DType"/>.</summary>
            public static LegacyBuiltinDTypeMeta ShortDType => DTypeRegistry.Int16;
            /// <summary><c>numpy.dtypes.UShortDType</c> — alias of <see cref="UInt16DType"/>.</summary>
            public static LegacyBuiltinDTypeMeta UShortDType => DTypeRegistry.UInt16;
            /// <summary><c>numpy.dtypes.IntDType</c> (C <c>int</c>) — alias of <see cref="Int32DType"/>.</summary>
            public static LegacyBuiltinDTypeMeta IntDType => DTypeRegistry.Int32;
            /// <summary><c>numpy.dtypes.UIntDType</c> (C <c>unsigned int</c>) — alias of <see cref="UInt32DType"/>.</summary>
            public static LegacyBuiltinDTypeMeta UIntDType => DTypeRegistry.UInt32;
            /// <summary><c>numpy.dtypes.LongDType</c> (C <c>long</c>): <see cref="Int32DType"/> on Windows / 32-bit hosts, <see cref="Int64DType"/> on LP64 Unix.</summary>
            public static LegacyBuiltinDTypeMeta LongDType => DTypeRegistry.CLongIs32Bit ? DTypeRegistry.Int32 : DTypeRegistry.Int64;
            /// <summary><c>numpy.dtypes.ULongDType</c> (C <c>unsigned long</c>): <see cref="UInt32DType"/> on Windows / 32-bit hosts, <see cref="UInt64DType"/> on LP64 Unix.</summary>
            public static LegacyBuiltinDTypeMeta ULongDType => DTypeRegistry.CLongIs32Bit ? DTypeRegistry.UInt32 : DTypeRegistry.UInt64;
            /// <summary><c>numpy.dtypes.LongLongDType</c> — alias of <see cref="Int64DType"/>.</summary>
            public static LegacyBuiltinDTypeMeta LongLongDType => DTypeRegistry.Int64;
            /// <summary><c>numpy.dtypes.ULongLongDType</c> — alias of <see cref="UInt64DType"/>.</summary>
            public static LegacyBuiltinDTypeMeta ULongLongDType => DTypeRegistry.UInt64;
            /// <summary><c>numpy.dtypes.LongDoubleDType</c> — collapses onto <see cref="Float64DType"/> (no extended precision, as on NumPy's win-amd64 build).</summary>
            public static LegacyBuiltinDTypeMeta LongDoubleDType => DTypeRegistry.Double;
            /// <summary><c>numpy.dtypes.CLongDoubleDType</c> — collapses onto <see cref="Complex128DType"/>.</summary>
            public static LegacyBuiltinDTypeMeta CLongDoubleDType => DTypeRegistry.Complex128;

            // ---- NumSharp-only classes ----

            /// <summary>NumSharp-only: the <see cref="decimal"/> class (no NumPy analog; type number 256, <c>isbuiltin == 2</c>).</summary>
            public static LegacyBuiltinDTypeMeta DecimalDType => DTypeRegistry.Decimal;

            // ---- NEP 55 StringDType: called with parameters ----

            /// <summary>
            ///     <c>np.dtypes.StringDType(coerce=True)</c>: a NEW variable-width string descriptor without a missing-data
            ///     object. Every call returns a fresh descriptor (NumPy's <c>StringDType() is StringDType()</c> is False),
            ///     equal to any other with the same parameters.
            /// </summary>
            /// <param name="coerce">NumPy's <c>coerce</c>: true (the default) stores a value that is not a <c>str</c> as its
            ///     <c>str()</c> (<c>3.25</c> → <c>'3.25'</c>); false refuses it with NumPy's
            ///     <c>ValueError: StringDType only allows string data when string coercion is disabled.</c></param>
            /// <returns>A fresh descriptor: <c>StringDType()</c> or <c>StringDType(coerce=False)</c>.</returns>
            /// <remarks>
            ///     The overload without <c>na_object</c> keeps NumPy's "not given" apart from <c>na_object=None</c>
            ///     (decision D5): <c>StringDType()</c> has no missing-data object at all, while
            ///     <c>StringDType(na_object: null)</c> has <c>None</c> as its sentinel.
            /// </remarks>
            public static DType StringDType(bool coerce = true) => DTypeRegistry.VString.Descr(false, null, coerce);

            /// <summary>
            ///     <c>np.dtypes.StringDType(na_object=…, coerce=True)</c>: a NEW variable-width string descriptor with a
            ///     missing-data sentinel.
            /// </summary>
            /// <param name="na_object">The sentinel (decision D5): <see langword="null"/> is Python's <c>None</c>; a
            ///     <see cref="double"/> / <see cref="float"/> NaN is <c>nan</c> (NaN-like: <c>StringDType(na_object=nan)</c>
            ///     equals <c>StringDType(na_object=np.nan)</c>); a <see cref="string"/> (or a <see cref="char"/>) is a string
            ///     sentinel, which missing values read as in string operations; any other value is an "other" sentinel compared
            ///     with Python's <c>==</c> (numbers by value).</param>
            /// <param name="coerce">NumPy's <c>coerce</c> (see <see cref="StringDType(bool)"/>).</param>
            /// <returns>A fresh descriptor, printing as <c>StringDType(na_object=None)</c>,
            ///     <c>StringDType(na_object='NA', coerce=False)</c>, …</returns>
            /// <exception cref="UnicodeEncodeError">A string sentinel (or the <c>str()</c> of another sentinel) holds a lone
            ///     surrogate: NumPy stores it as UTF-8 and raises CPython's
            ///     <c>'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed</c>.</exception>
            /// <remarks>
            ///     Pass a <see cref="bool"/> sentinel BY NAME (<c>StringDType(na_object: true)</c>): a positional bool binds
            ///     the <see cref="StringDType(bool)"/> overload and means <c>coerce</c>.
            /// </remarks>
            public static DType StringDType(object na_object, bool coerce = true) => DTypeRegistry.VString.Descr(true, na_object, coerce);

            /// <summary>NumSharp-only: the <see cref="char"/> class (no NumPy analog; promotes as uint16; type number 257).</summary>
            public static LegacyBuiltinDTypeMeta CharDType => DTypeRegistry.Char;
        }
    }
}
