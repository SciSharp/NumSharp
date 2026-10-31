using System;
using System.Linq;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Return a scalar type which is common to the input arrays.
        /// </summary>
        /// <param name="arrays">Input arrays; none is allowed, as in NumPy.</param>
        /// <returns>
        /// The common scalar type: always inexact (float64 at least for integers), complex when any input is complex,
        /// and NumPy's starting precision — float16 — when no array is given (<c>np.common_type()</c> is
        /// <c>numpy.float16</c>, the <c>array_type[0][0]</c> its loop starts from).
        /// </returns>
        /// <exception cref="TypeError">An input is not numeric (a bool array: NumPy's "can't get common type for non-numeric array").</exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.common_type.html
        ///
        /// The return type will always be a floating-point type (minimum float64 for integers).
        /// This differs from result_type which may return an integer type.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.common_type(np.array(new int[] {1, 2}))           // float64
        /// np.common_type(a_float32, a_float64)                 // float64
        /// np.common_type()                                     // float16
        /// </code>
        /// </example>
        public static DType common_type(params NDArray[] arrays)
        {
            // A null params array is the caller passing (NDArray[])null; NumPy has no such spelling, so it stays an error.
            if (arrays == null)
                throw new ArgumentNullException(nameof(arrays));

            return common_type_code(arrays);
        }

        /// <summary>
        /// Return a scalar type code which is common to the input arrays.
        /// </summary>
        /// <param name="arrays">Input arrays; none gives float16, as <see cref="common_type(NDArray[])"/> does.</param>
        /// <returns>The common scalar type as NPTypeCode.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arrays"/> is null.</exception>
        /// <exception cref="TypeError">An input is a bool array (not numeric for <c>common_type</c>).</exception>
        /// <remarks>
        /// The return type will always be a floating-point type (minimum Double for integers).
        /// </remarks>
        public static NPTypeCode common_type_code(params NDArray[] arrays)
        {
            if (arrays == null)
                throw new ArgumentNullException(nameof(arrays));

            return common_type_code(arrays.Select(a => a.GetTypeCode).ToArray());
        }

        /// <summary>
        /// Return a scalar type code which is common to the input type codes.
        /// </summary>
        /// <param name="types">Input type codes; none gives <see cref="NPTypeCode.Half"/>.</param>
        /// <returns>The common scalar type as NPTypeCode.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="types"/> is null.</exception>
        /// <exception cref="TypeError">A type is <see cref="NPTypeCode.Boolean"/> (not numeric for <c>common_type</c>).</exception>
        /// <remarks>
        /// NumPy common_type rules:
        /// - Any Complex input -> Complex (complex128).
        /// - Any Decimal input (NumSharp extension) -> Decimal.
        /// - Any integer/bool/char input -> Double (any int presence forces float64).
        /// - Otherwise (all float16/float32/float64): return max-precision float — float16 when there is no input at
        ///   all, because NumPy's loop starts from precision 0 (<c>half</c>) and never raises it.
        /// </remarks>
        public static NPTypeCode common_type_code(params NPTypeCode[] types)
        {
            if (types == null)
                throw new ArgumentNullException(nameof(types));

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
                    case NPTypeCode.String:
                        // NumPy parity: np.common_type rejects bool and text as "non-numeric" (neither is an np.integer
                        // nor in its array_precision table). Text used to fall through to the integer case: float64.
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

            // All pure floats. Pick max precision. Rank 0 means no input at all: NumPy's `precision = 0` start, which
            // indexes array_type[0][0] = half.
            return maxFloatRank switch
            {
                0 or 1 => NPTypeCode.Half,
                2 => NPTypeCode.Single,
                _ => NPTypeCode.Double,
            };
        }
    }
}
