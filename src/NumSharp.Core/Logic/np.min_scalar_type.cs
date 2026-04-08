using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// For scalar value, returns the data type with the smallest size and smallest scalar
        /// kind which can hold its value.
        /// </summary>
        /// <param name="value">The scalar value to check.</param>
        /// <returns>The minimum dtype that can represent the value.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.min_scalar_type.html
        ///
        /// For integers, finds the smallest integer type that can hold the value.
        /// For floats, finds the smallest float type that can represent the value exactly.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.min_scalar_type(10)      // Byte (uint8)
        /// np.min_scalar_type(-10)     // Int16 (no int8 in NumSharp)
        /// np.min_scalar_type(1000)    // UInt16
        /// np.min_scalar_type(1.0)     // Single (float32)
        /// np.min_scalar_type(true)    // Boolean
        /// </code>
        /// </example>
        public static NPTypeCode min_scalar_type(object value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));

            return value switch
            {
                bool _ => NPTypeCode.Boolean,
                byte _ => NPTypeCode.Byte,
                sbyte sb => sb >= 0 ? NPTypeCode.Byte : NPTypeCode.Int16,  // No int8 in NumSharp
                short s => MinTypeForSignedInt(s),
                ushort us => MinTypeForUnsignedInt(us),
                int i => MinTypeForSignedInt(i),
                uint ui => MinTypeForUnsignedInt(ui),
                long l => MinTypeForSignedInt(l),
                ulong ul => MinTypeForUnsignedInt(ul),
                float f => MinTypeForFloat(f),
                double d => MinTypeForDouble(d),
                decimal _ => NPTypeCode.Decimal,
                char c => MinTypeForUnsignedInt(c),
                _ => value.GetType().GetTypeCode()
            };
        }

        /// <summary>
        /// Find minimum type for a signed integer value.
        /// </summary>
        private static NPTypeCode MinTypeForSignedInt(long value)
        {
            if (value >= 0)
            {
                // Non-negative: prefer unsigned
                return MinTypeForUnsignedInt((ulong)value);
            }
            else
            {
                // Negative: need signed type
                // Note: NumSharp doesn't have int8/sbyte, so smallest is int16
                if (value >= short.MinValue)
                    return NPTypeCode.Int16;
                if (value >= int.MinValue)
                    return NPTypeCode.Int32;
                return NPTypeCode.Int64;
            }
        }

        /// <summary>
        /// Find minimum type for an unsigned integer value.
        /// </summary>
        private static NPTypeCode MinTypeForUnsignedInt(ulong value)
        {
            if (value <= byte.MaxValue)
                return NPTypeCode.Byte;
            if (value <= ushort.MaxValue)
                return NPTypeCode.UInt16;
            if (value <= uint.MaxValue)
                return NPTypeCode.UInt32;
            return NPTypeCode.UInt64;
        }

        /// <summary>
        /// Find minimum float type for a float value.
        /// </summary>
        private static NPTypeCode MinTypeForFloat(float value)
        {
            // Float32 is already the smallest float type
            return NPTypeCode.Single;
        }

        /// <summary>
        /// Find minimum float type for a double value.
        /// </summary>
        private static NPTypeCode MinTypeForDouble(double value)
        {
            // Check if value can be represented exactly as float32
            if (!double.IsFinite(value))
            {
                // Infinity and NaN can be represented in float32
                return NPTypeCode.Single;
            }

            // Check if converting to float and back preserves the value
            float asFloat = (float)value;
            if ((double)asFloat == value)
            {
                return NPTypeCode.Single;
            }

            return NPTypeCode.Double;
        }
    }
}
