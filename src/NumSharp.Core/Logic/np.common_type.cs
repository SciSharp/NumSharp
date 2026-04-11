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

            // Get the result type from all arrays
            var types = arrays.Select(a => a.GetTypeCode).ToArray();

            NPTypeCode result;
            if (types.Length == 1)
            {
                result = types[0];
            }
            else
            {
                result = _FindCommonType_Array(types);
            }

            // common_type always returns a floating point type
            // Integers promote to at least Double
            return result switch
            {
                NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or
                NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 or
                NPTypeCode.Char => NPTypeCode.Double,

                NPTypeCode.Single => NPTypeCode.Single,  // Keep float32 if all inputs are float32
                NPTypeCode.Double => NPTypeCode.Double,
                NPTypeCode.Decimal => NPTypeCode.Decimal,
                NPTypeCode.Complex => NPTypeCode.Complex,  // Complex stays complex

                _ => NPTypeCode.Double  // Default to double for unknown types
            };
        }

        /// <summary>
        /// Return a scalar type code which is common to the input type codes.
        /// </summary>
        /// <param name="types">Input type codes.</param>
        /// <returns>The common scalar type as NPTypeCode.</returns>
        public static NPTypeCode common_type_code(params NPTypeCode[] types)
        {
            if (types == null || types.Length == 0)
                throw new ArgumentException("At least one type must be provided", nameof(types));

            NPTypeCode result;
            if (types.Length == 1)
            {
                result = types[0];
            }
            else
            {
                result = _FindCommonType_Array(types);
            }

            // Always return floating point type
            return result switch
            {
                NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or
                NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 or
                NPTypeCode.Char => NPTypeCode.Double,

                NPTypeCode.Single => NPTypeCode.Single,
                NPTypeCode.Double => NPTypeCode.Double,
                NPTypeCode.Decimal => NPTypeCode.Decimal,
                NPTypeCode.Complex => NPTypeCode.Complex,

                _ => NPTypeCode.Double
            };
        }
    }
}
