using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    /// <summary>
    /// Machine limits for integer types.
    /// </summary>
    /// <remarks>
    /// https://numpy.org/doc/stable/reference/generated/numpy.iinfo.html
    /// </remarks>
    public sealed class iinfo
    {
        /// <summary>
        /// Number of bits occupied by the type.
        /// </summary>
        public int bits { get; }

        /// <summary>
        /// Minimum value of given dtype.
        /// </summary>
        public long min { get; }

        /// <summary>
        /// Maximum value of given dtype.
        /// </summary>
        /// <remarks>
        /// For unsigned 64-bit integers, the actual max is ulong.MaxValue which
        /// exceeds long.MaxValue. Use <see cref="maxUnsigned"/> to get the true value.
        /// </remarks>
        public long max { get; }

        /// <summary>
        /// Maximum value for unsigned types as ulong.
        /// For signed types, this equals <see cref="max"/>.
        /// </summary>
        public ulong maxUnsigned { get; }

        /// <summary>
        /// The NPTypeCode of this integer type.
        /// </summary>
        public NPTypeCode dtype { get; }

        /// <summary>
        /// Character code for this type.
        /// 'i' for signed integers, 'u' for unsigned integers, 'b' for boolean.
        /// </summary>
        public char kind { get; }

        /// <summary>
        /// Create iinfo for the specified NPTypeCode.
        /// </summary>
        /// <param name="typeCode">An integer type code.</param>
        /// <exception cref="ArgumentException">Thrown if typeCode is not an integer type.</exception>
        public iinfo(NPTypeCode typeCode)
        {
            if (!IsIntegerType(typeCode))
                throw new ArgumentException($"Invalid integer data type '{typeCode.AsNumpyDtypeName()}'", nameof(typeCode));

            dtype = typeCode;
            (bits, min, max, maxUnsigned, kind) = GetTypeInfo(typeCode);
        }

        /// <summary>
        /// Create iinfo for the specified CLR type.
        /// </summary>
        /// <param name="type">A CLR integer type.</param>
        /// <exception cref="ArgumentException">Thrown if type is not an integer type.</exception>
        public iinfo(Type type) : this(type.GetTypeCode())
        {
        }

        /// <inheritdoc />
        public override string ToString()
        {
            return $"iinfo(min={min}, max={max}, dtype={dtype.AsNumpyDtypeName()})";
        }

        private static bool IsIntegerType(NPTypeCode typeCode)
        {
            return typeCode switch
            {
                NPTypeCode.Boolean => true,  // NumPy treats bool as integer-like for iinfo
                NPTypeCode.Byte => true,
                NPTypeCode.Int16 => true,
                NPTypeCode.UInt16 => true,
                NPTypeCode.Int32 => true,
                NPTypeCode.UInt32 => true,
                NPTypeCode.Int64 => true,
                NPTypeCode.UInt64 => true,
                NPTypeCode.Char => true,  // Char is treated as uint16 equivalent
                _ => false
            };
        }

        private static (int bits, long min, long max, ulong maxUnsigned, char kind) GetTypeInfo(NPTypeCode typeCode)
        {
            return typeCode switch
            {
                NPTypeCode.Boolean => (8, 0, 1, 1, 'b'),
                NPTypeCode.Byte => (8, 0, byte.MaxValue, byte.MaxValue, 'u'),
                NPTypeCode.Int16 => (16, short.MinValue, short.MaxValue, (ulong)short.MaxValue, 'i'),
                NPTypeCode.UInt16 => (16, 0, ushort.MaxValue, ushort.MaxValue, 'u'),
                NPTypeCode.Int32 => (32, int.MinValue, int.MaxValue, (ulong)int.MaxValue, 'i'),
                NPTypeCode.UInt32 => (32, 0, uint.MaxValue, uint.MaxValue, 'u'),
                NPTypeCode.Int64 => (64, long.MinValue, long.MaxValue, (ulong)long.MaxValue, 'i'),
                NPTypeCode.UInt64 => (64, 0, long.MaxValue, ulong.MaxValue, 'u'),  // max clamped to long.MaxValue
                NPTypeCode.Char => (16, 0, char.MaxValue, char.MaxValue, 'u'),  // Char treated as uint16
                _ => throw new ArgumentException($"Invalid integer data type '{typeCode}'")
            };
        }
    }

    public partial class np
    {
        /// <summary>
        /// Machine limits for integer types.
        /// </summary>
        /// <param name="typeCode">An integer type code.</param>
        /// <returns>An iinfo object describing the integer type limits.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.iinfo.html
        /// </remarks>
        /// <example>
        /// <code>
        /// var info = np.iinfo(NPTypeCode.Int32);
        /// Console.WriteLine(info.bits);  // 32
        /// Console.WriteLine(info.min);   // -2147483648
        /// Console.WriteLine(info.max);   // 2147483647
        /// </code>
        /// </example>
        public static iinfo iinfo(NPTypeCode typeCode) => new iinfo(typeCode);

        /// <summary>
        /// Machine limits for integer types.
        /// </summary>
        /// <param name="type">A CLR integer type.</param>
        /// <returns>An iinfo object describing the integer type limits.</returns>
        public static iinfo iinfo(Type type) => new iinfo(type);
    }
}
