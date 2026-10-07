using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// For scalar value, returns the data type with the smallest size and smallest scalar
        /// kind which can hold its value. For an array of one or more dimensions (or of a non-number dtype such as
        /// bool or a string class), returns the array's own dtype unmodified, as NumPy does.
        /// </summary>
        /// <param name="value">The scalar value to check, a C# string (a Python <c>str</c>: <c>&lt;U</c> of its code
        /// points), or an <see cref="NDArray"/>.</param>
        /// <returns>The minimum dtype that can represent the value; an array's dtype for an array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.min_scalar_type.html
        ///
        /// For integers, finds the smallest integer type that can hold the value.
        /// For floats, finds the smallest float type whose RANGE (magnitude) can hold the
        /// value — matching NumPy, precision loss is allowed (e.g. 1e-40 reports float16).
        /// </remarks>
        /// <example>
        /// <code>
        /// np.min_scalar_type(10)      // Byte (uint8)
        /// np.min_scalar_type(-10)     // SByte (int8)
        /// np.min_scalar_type(1000)    // UInt16
        /// np.min_scalar_type(1.0)     // Half (float16)
        /// np.min_scalar_type(true)    // Boolean
        /// </code>
        /// </example>
        public static DType min_scalar_type(object value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));

            // NumPy's PyArray_MinScalarType: an array of one or more dimensions, or of a non-number dtype (bool, the
            // string classes, ...), reports its own descriptor UNMODIFIED; only a 0-d NUMBER array is inspected by value
            // (np.min_scalar_type(np.array(10)) is uint8). A C# string is a Python str: its 0-d array's descriptor,
            // <U(code points) — np.min_scalar_type('ab') is dtype('<U2').
            if (value is NDArray array)
                return MinScalarTypeOfArray(array);
            if (value is string text)
            {
                using var scalar = NDArray.Scalar(text);
                return scalar.dtype;
            }

            return value switch
            {
                bool _ => NPTypeCode.Boolean,
                byte _ => NPTypeCode.Byte,
                // Signed 8-bit: NumSharp DOES have int8 (SByte). Negative -> int8; non-negative
                // demotes to the smallest unsigned type (NumPy: min_scalar_type(np.int8(5)) == uint8).
                sbyte sb => MinTypeForSignedInt(sb),
                short s => MinTypeForSignedInt(s),
                ushort us => MinTypeForUnsignedInt(us),
                int i => MinTypeForSignedInt(i),
                uint ui => MinTypeForUnsignedInt(ui),
                long l => MinTypeForSignedInt(l),
                ulong ul => MinTypeForUnsignedInt(ul),
                System.Half _ => NPTypeCode.Half,   // NumPy NPY_HALF: float16 stays float16
                float f => MinTypeForFloat(f),
                double d => MinTypeForDouble(d),
                decimal _ => NPTypeCode.Decimal,
                char c => MinTypeForUnsignedInt(c),
                _ => value.GetType().GetTypeCode()
            };
        }

        /// <summary>
        ///     NumPy's <c>PyArray_MinScalarType</c> for an array argument: the array's own descriptor, except a 0-d array
        ///     of a NUMBER dtype, which is reduced by its value like the scalar it holds.
        /// </summary>
        /// <param name="array">The array (any dtype, any layout).</param>
        /// <returns>The array's dtype (a string array keeps its class and width), or the smallest dtype holding its single
        ///     value when it is a 0-d number.</returns>
        /// <remarks>
        ///     NumPy's rule is <c>PyArray_NDIM(arr) &gt; 0 || !PyTypeNum_ISNUMBER(type_num)</c> → the dtype unchanged.
        ///     Bool is not a number there (its value would answer bool anyway); NumSharp's <c>Char</c> and <c>Decimal</c>
        ///     lanes are numbers (their scalars already have value rules above).
        /// </remarks>
        private static DType MinScalarTypeOfArray(NDArray array)
        {
            var code = array.typecode;
            if (array.ndim > 0 || code == NPTypeCode.Boolean || code == NPTypeCode.String)
                return array.dtype;
            // A 0-d number: its single element, boxed as its own CLR type, takes the scalar rules.
            return min_scalar_type(array.GetAtIndex(0));
        }

        /// <summary>
        /// Find minimum type for a signed integer value.
        /// </summary>
        private static NPTypeCode MinTypeForSignedInt(long value)
        {
            if (value >= 0)
            {
                // Non-negative signed scalars demote to the smallest UNSIGNED type
                // (NumPy: min_scalar_type(np.int32(10)) == uint8).
                return MinTypeForUnsignedInt((ulong)value);
            }

            // Negative: smallest SIGNED type whose range holds the value.
            // NumPy min_scalar_type_num demotes negatives down to int8.
            if (value >= sbyte.MinValue)   // >= -128   -> int8
                return NPTypeCode.SByte;
            if (value >= short.MinValue)   // >= -32768 -> int16
                return NPTypeCode.Int16;
            if (value >= int.MinValue)
                return NPTypeCode.Int32;
            return NPTypeCode.Int64;
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
        /// Find minimum float type for a float32 value (NumPy min_scalar_type_num, NPY_FLOAT).
        /// </summary>
        private static NPTypeCode MinTypeForFloat(float value)
        {
            // A value inside float16's finite RANGE (|v| &lt; 65000) — or any non-finite
            // value (±inf / NaN) — demotes to float16; otherwise it stays float32.
            // Exclusive bounds copied verbatim from numpy convert_datatype.c.
            if (!float.IsFinite(value) || (value > -65000f && value < 65000f))
                return NPTypeCode.Half;
            return NPTypeCode.Single;
        }

        /// <summary>
        /// Find minimum float type for a float64 value (NumPy min_scalar_type_num, NPY_DOUBLE).
        /// </summary>
        private static NPTypeCode MinTypeForDouble(double value)
        {
            // NumPy demotes floats by RANGE (magnitude), NEVER by exact representability —
            // even values that underflow float16 (e.g. 1e-40) report float16. Bounds are
            // exclusive and copied verbatim from numpy convert_datatype.c:
            //   |v| < 65000 (or non-finite) -> float16 ; |v| < 3.4e38 -> float32 ; else float64.
            if (!double.IsFinite(value) || (value > -65000.0 && value < 65000.0))
                return NPTypeCode.Half;
            if (value > -3.4e38 && value < 3.4e38)
                return NPTypeCode.Single;
            return NPTypeCode.Double;
        }
    }
}
