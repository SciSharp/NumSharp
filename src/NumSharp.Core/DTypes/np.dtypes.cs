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
        ///     as NumPy's win-amd64 build. The classes NumPy has and NumSharp does not (<c>Complex64DType</c>, <c>StrDType</c>,
        ///     <c>BytesDType</c>, <c>StringDType</c>, <c>ObjectDType</c>, <c>VoidDType</c>) are absent rather than stubbed;
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
            /// <summary>NumSharp-only: the <see cref="char"/> class (no NumPy analog; promotes as uint16; type number 257).</summary>
            public static LegacyBuiltinDTypeMeta CharDType => DTypeRegistry.Char;
        }
    }
}
