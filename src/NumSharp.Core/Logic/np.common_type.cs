using System;
using System.Linq;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Return a scalar type which is common to the input arrays.
        /// </summary>
        /// <param name="arrays">Input arrays.</param>
        /// <returns>The common scalar type as CLR Type.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.common_type.html
        ///
        /// The return type will always be a floating-point type (minimum float64 for integers).
        /// This differs from result_type which may return an integer type.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.common_type(np.array(new int[] {1, 2}))           // typeof(double)
        /// np.common_type(a_float32, a_float64)                 // typeof(double)
        /// </code>
        /// </example>
        public static Type common_type(params NDArray[] arrays)
        {
            if (arrays == null || arrays.Length == 0)
                throw new ArgumentException("At least one array must be provided", nameof(arrays));

            return common_type_code(arrays).AsType();
        }

        /// <summary>
        /// Return a scalar type code which is common to the input arrays.
        /// </summary>
        /// <param name="arrays">Input arrays.</param>
        /// <returns>The common scalar type as NPTypeCode.</returns>
        /// <remarks>
        /// The return type will always be a floating-point type (minimum Double for integers).
        /// </remarks>
        public static NPTypeCode common_type_code(params NDArray[] arrays)
        {
            if (arrays == null || arrays.Length == 0)
                throw new ArgumentException("At least one array must be provided", nameof(arrays));

            return common_type_code(arrays.Select(a => a.GetTypeCode).ToArray());
        }

        /// <summary>
        /// Return a scalar type code which is common to the input type codes.
        /// </summary>
        /// <param name="types">Input type codes.</param>
        /// <returns>The common scalar type as NPTypeCode.</returns>
        /// <remarks>
        /// NumPy common_type rules:
        /// - Any Complex input -> Complex (complex128).
        /// - Any Decimal input (NumSharp extension) -> Decimal.
        /// - Any integer/bool/char input -> Double (any int presence forces float64).
        /// - Otherwise (all float16/float32/float64): return max-precision float.
        /// </remarks>
        public static NPTypeCode common_type_code(params NPTypeCode[] types)
        {
            if (types == null || types.Length == 0)
                throw new ArgumentException("At least one type must be provided", nameof(types));

            bool hasComplex = false;
            bool hasDecimal = false;
            bool hasInt = false;
            // Rank pure floats: Half=1, Single=2, Double=3.
            int maxFloatRank = 0;

            foreach (var t in types)
            {
                switch (t)
                {
                    case NPTypeCode.Boolean:
                        // NumPy parity: np.common_type rejects bool as "non-numeric".
                        throw new TypeError("can't get common type for non-numeric array");
                    case NPTypeCode.Complex:
                        hasComplex = true;
                        break;
                    case NPTypeCode.Decimal:
                        hasDecimal = true;
                        break;
                    case NPTypeCode.Half:
                        if (maxFloatRank < 1) maxFloatRank = 1;
                        break;
                    case NPTypeCode.Single:
                        if (maxFloatRank < 2) maxFloatRank = 2;
                        break;
                    case NPTypeCode.Double:
                        if (maxFloatRank < 3) maxFloatRank = 3;
                        break;
                    // byte/sbyte, int16/uint16, int32/uint32, int64/uint64, char
                    default:
                        hasInt = true;
                        break;
                }
            }

            if (hasComplex) return NPTypeCode.Complex;
            if (hasDecimal) return NPTypeCode.Decimal;
            // NumPy parity: any integer presence promotes to at least float64, overriding
            // smaller float precision seen elsewhere in the inputs.
            if (hasInt) return NPTypeCode.Double;

            // All pure floats. Pick max precision.
            return maxFloatRank switch
            {
                1 => NPTypeCode.Half,
                2 => NPTypeCode.Single,
                _ => NPTypeCode.Double,
            };
        }
    }
}
