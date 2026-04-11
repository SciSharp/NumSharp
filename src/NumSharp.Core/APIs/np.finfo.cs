using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    /// <summary>
    /// Machine limits for floating point types.
    /// </summary>
    /// <remarks>
    /// https://numpy.org/doc/stable/reference/generated/numpy.finfo.html
    /// </remarks>
    public sealed class finfo
    {
        /// <summary>
        /// Number of bits occupied by the type.
        /// </summary>
        public int bits { get; }

        /// <summary>
        /// The difference between 1.0 and the next smallest representable float larger than 1.0.
        /// </summary>
        public double eps { get; }

        /// <summary>
        /// The difference between 1.0 and the next smallest representable float smaller than 1.0.
        /// </summary>
        public double epsneg { get; }

        /// <summary>
        /// The largest representable number.
        /// </summary>
        public double max { get; }

        /// <summary>
        /// The smallest representable number (typically -max).
        /// </summary>
        public double min { get; }

        /// <summary>
        /// The smallest positive usable number (alias for smallest_normal).
        /// </summary>
        public double tiny { get; }

        /// <summary>
        /// The smallest positive normal number.
        /// </summary>
        public double smallest_normal { get; }

        /// <summary>
        /// The smallest positive subnormal number.
        /// </summary>
        public double smallest_subnormal { get; }

        /// <summary>
        /// The approximate number of decimal digits to which this kind of float is precise.
        /// </summary>
        public int precision { get; }

        /// <summary>
        /// The approximate decimal resolution of this type (10^-precision).
        /// </summary>
        public double resolution { get; }

        /// <summary>
        /// The smallest positive power of the base (2) that causes overflow.
        /// </summary>
        public int maxexp { get; }

        /// <summary>
        /// The most negative power of the base (2) consistent with there being no leading 0s in the mantissa.
        /// </summary>
        public int minexp { get; }

        /// <summary>
        /// The NPTypeCode of this floating point type.
        /// </summary>
        public NPTypeCode dtype { get; }

        /// <summary>
        /// Create finfo for the specified NPTypeCode.
        /// </summary>
        /// <param name="typeCode">A floating point type code.</param>
        /// <exception cref="ArgumentException">Thrown if typeCode is not a floating point type.</exception>
        public finfo(NPTypeCode typeCode)
        {
            if (!IsFloatType(typeCode))
                throw new ArgumentException($"data type '{typeCode.AsNumpyDtypeName()}' not inexact", nameof(typeCode));

            dtype = typeCode;

            switch (typeCode)
            {
                case NPTypeCode.Single:
                    bits = 32;
                    // float.Epsilon is the smallest subnormal
                    // Machine epsilon for float: MathF.BitIncrement(1.0f) - 1.0f ≈ 1.1920929e-07
                    // Note: Must use MathF, not Math (which only works on double)
                    eps = MathF.BitIncrement(1.0f) - 1.0f;
                    epsneg = 1.0f - MathF.BitDecrement(1.0f);
                    max = float.MaxValue;
                    min = -float.MaxValue;
                    smallest_normal = 1.175494351e-38;  // 2^-126
                    smallest_subnormal = float.Epsilon;  // ~1.4e-45
                    tiny = smallest_normal;
                    precision = 6;
                    resolution = 1e-6;
                    maxexp = 128;   // 2^127 * (2-eps) = MaxValue
                    minexp = -125;  // 2^-126 = smallest normal
                    break;

                case NPTypeCode.Double:
                    bits = 64;
                    eps = Math.BitIncrement(1.0) - 1.0;  // ~2.22e-16
                    epsneg = 1.0 - Math.BitDecrement(1.0);
                    max = double.MaxValue;
                    min = -double.MaxValue;
                    smallest_normal = 2.2250738585072014e-308;  // 2^-1022
                    smallest_subnormal = double.Epsilon;  // ~5e-324
                    tiny = smallest_normal;
                    precision = 15;
                    resolution = 1e-15;
                    maxexp = 1024;   // 2^1023 * (2-eps) = MaxValue
                    minexp = -1021;  // 2^-1022 = smallest normal
                    break;

                case NPTypeCode.Decimal:
                    // Decimal is a 128-bit type with different semantics
                    // It doesn't have subnormals like IEEE floats
                    bits = 128;
                    eps = 1e-28;  // Approximately 1 / 10^28
                    epsneg = 1e-28;
                    max = (double)decimal.MaxValue;
                    min = (double)decimal.MinValue;
                    smallest_normal = 1e-28;  // Smallest positive decimal
                    smallest_subnormal = 1e-28;  // No subnormals for decimal
                    tiny = smallest_normal;
                    precision = 28;
                    resolution = 1e-28;
                    maxexp = 96;   // Approximate - decimal has different representation
                    minexp = -28;  // Approximate
                    break;

                default:
                    throw new ArgumentException($"Unsupported float type: {typeCode}");
            }
        }

        /// <summary>
        /// Create finfo for the specified CLR type.
        /// </summary>
        /// <param name="type">A CLR floating point type.</param>
        /// <exception cref="ArgumentException">Thrown if type is not a floating point type.</exception>
        public finfo(Type type) : this(type.GetTypeCode())
        {
        }

        /// <inheritdoc />
        public override string ToString()
        {
            return $"finfo(resolution={resolution}, min={min:E}, max={max:E}, dtype={dtype.AsNumpyDtypeName()})";
        }

        private static bool IsFloatType(NPTypeCode typeCode)
        {
            return typeCode switch
            {
                NPTypeCode.Single => true,
                NPTypeCode.Double => true,
                NPTypeCode.Decimal => true,  // Partial support - no subnormals
                _ => false
            };
        }
    }

    public partial class np
    {
        /// <summary>
        /// Machine limits for floating point types.
        /// </summary>
        /// <param name="typeCode">A floating point type code.</param>
        /// <returns>An finfo object describing the floating point type limits.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.finfo.html
        /// </remarks>
        /// <example>
        /// <code>
        /// var info = np.finfo(NPTypeCode.Double);
        /// Console.WriteLine(info.bits);       // 64
        /// Console.WriteLine(info.eps);        // ~2.22e-16
        /// Console.WriteLine(info.precision);  // 15
        /// </code>
        /// </example>
        public static finfo finfo(NPTypeCode typeCode) => new finfo(typeCode);

        /// <summary>
        /// Machine limits for floating point types.
        /// </summary>
        /// <param name="type">A CLR floating point type.</param>
        /// <returns>An finfo object describing the floating point type limits.</returns>
        public static finfo finfo(Type type) => new finfo(type);

        /// <summary>
        /// Machine limits for floating point types.
        /// </summary>
        /// <typeparam name="T">A floating point type (float, double, decimal).</typeparam>
        /// <returns>An finfo object describing the floating point type limits.</returns>
        /// <example>
        /// <code>
        /// var info = np.finfo&lt;double&gt;();
        /// Console.WriteLine(info.bits);  // 64
        /// </code>
        /// </example>
        public static finfo finfo<T>() where T : struct => new finfo(typeof(T));

        /// <summary>
        /// Machine limits for floating point types.
        /// </summary>
        /// <param name="arr">An NDArray with floating point dtype.</param>
        /// <returns>An finfo object describing the array's floating point type limits.</returns>
        /// <exception cref="ArgumentNullException">Thrown if arr is null.</exception>
        /// <example>
        /// <code>
        /// var a = np.array(new double[] {1.0, 2.0, 3.0});
        /// var info = np.finfo(a);
        /// Console.WriteLine(info.bits);  // 64
        /// </code>
        /// </example>
        public static finfo finfo(NDArray arr)
        {
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));
            return new finfo(arr.GetTypeCode);
        }

        /// <summary>
        /// Machine limits for floating point types.
        /// </summary>
        /// <param name="dtypeName">A dtype string (e.g., "float32", "float64", "double").</param>
        /// <returns>An finfo object describing the floating point type limits.</returns>
        /// <exception cref="ArgumentException">Thrown if dtypeName is not a valid floating point dtype.</exception>
        /// <example>
        /// <code>
        /// var info = np.finfo("float64");
        /// Console.WriteLine(info.bits);  // 64
        /// </code>
        /// </example>
        public static finfo finfo(string dtypeName)
        {
            if (string.IsNullOrEmpty(dtypeName))
                throw new ArgumentException("dtype name cannot be null or empty", nameof(dtypeName));
            return new finfo(dtype(dtypeName).typecode);
        }
    }
}
