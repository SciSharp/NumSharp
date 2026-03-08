using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Compute the truth value of x1 AND x2 element-wise.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Boolean result of the logical AND operation applied to the elements of x1 and x2; the shape is determined by broadcasting.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html</remarks>
        public static NDArray<bool> logical_and(in NDArray x1, in NDArray x2)
        {
            // Convert to boolean (nonzero = true) then AND
            var b1 = x1.typecode == NPTypeCode.Boolean ? x1 : (x1 != 0);
            var b2 = x2.typecode == NPTypeCode.Boolean ? x2 : (x2 != 0);
            return (b1 & b2).MakeGeneric<bool>();
        }

        /// <summary>
        /// Compute the truth value of x1 OR x2 element-wise.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Boolean result of the logical OR operation applied to the elements of x1 and x2; the shape is determined by broadcasting.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html</remarks>
        public static NDArray<bool> logical_or(in NDArray x1, in NDArray x2)
        {
            // Convert to boolean (nonzero = true) then OR
            var b1 = x1.typecode == NPTypeCode.Boolean ? x1 : (x1 != 0);
            var b2 = x2.typecode == NPTypeCode.Boolean ? x2 : (x2 != 0);
            return (b1 | b2).MakeGeneric<bool>();
        }

        /// <summary>
        /// Compute the truth value of NOT x element-wise.
        /// </summary>
        /// <param name="x">Logical NOT is applied to the elements of x.</param>
        /// <returns>Boolean result with the same shape as x.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html</remarks>
        public static NDArray<bool> logical_not(in NDArray x)
        {
            // For boolean arrays, use LogicalNot (via Negate which routes to LogicalNot for bool)
            // For other types, nonzero becomes False, zero becomes True
            if (x.typecode == NPTypeCode.Boolean)
            {
                return x.TensorEngine.Negate(x).MakeGeneric<bool>();
            }
            return (x == 0).MakeGeneric<bool>();
        }

        /// <summary>
        /// Compute the truth value of x1 XOR x2 element-wise.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Boolean result of the logical XOR operation applied to the elements of x1 and x2; the shape is determined by broadcasting.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html</remarks>
        public static NDArray<bool> logical_xor(in NDArray x1, in NDArray x2)
        {
            // Convert to boolean (nonzero = true) then XOR
            NDArray b1 = x1.typecode == NPTypeCode.Boolean ? x1 : (x1 != 0);
            NDArray b2 = x2.typecode == NPTypeCode.Boolean ? x2 : (x2 != 0);
            return b1.TensorEngine.BitwiseXor(b1, b2).MakeGeneric<bool>();
        }
    }
}
