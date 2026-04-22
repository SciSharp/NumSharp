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
                NPTypeCode.Boolean => true,  // NumSharp extension — NumPy 2.x throws ValueError
                NPTypeCode.SByte => true,
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
                NPTypeCode.SByte => (8, sbyte.MinValue, sbyte.MaxValue, (ulong)sbyte.MaxValue, 'i'),
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

        /// <summary>
        /// Machine limits for integer types.
        /// </summary>
        /// <typeparam name="T">An integer type (bool, byte, short, ushort, int, uint, long, ulong, char).</typeparam>
        /// <returns>An iinfo object describing the integer type limits.</returns>
        /// <example>
        /// <code>
        /// var info = np.iinfo&lt;int&gt;();
        /// Console.WriteLine(info.bits);  // 32
        /// </code>
        /// </example>
        public static iinfo iinfo<T>() where T : struct => new iinfo(typeof(T));

        /// <summary>
        /// Machine limits for integer types.
        /// </summary>
        /// <param name="arr">An NDArray with integer dtype.</param>
        /// <returns>An iinfo object describing the array's integer type limits.</returns>
        /// <exception cref="ArgumentNullException">Thrown if arr is null.</exception>
        /// <example>
        /// <code>
        /// var a = np.array(new int[] {1, 2, 3});
        /// var info = np.iinfo(a);
        /// Console.WriteLine(info.bits);  // 32
        /// </code>
        /// </example>
        public static iinfo iinfo(NDArray arr)
        {
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));
            return new iinfo(arr.GetTypeCode);
        }

        /// <summary>
        /// Machine limits for integer types.
        /// </summary>
        /// <param name="dtypeName">A dtype string (e.g., "int32", "uint8", "bool").</param>
        /// <returns>An iinfo object describing the integer type limits.</returns>
        /// <exception cref="ArgumentException">Thrown if dtypeName is not a valid integer dtype.</exception>
        /// <example>
        /// <code>
        /// var info = np.iinfo("int32");
        /// Console.WriteLine(info.bits);  // 32
        /// </code>
        /// </example>
        public static iinfo iinfo(string dtypeName)
        {
            if (string.IsNullOrEmpty(dtypeName))
                throw new ArgumentException("dtype name cannot be null or empty", nameof(dtypeName));
            return new iinfo(dtype(dtypeName).typecode);
        }
    }
}
