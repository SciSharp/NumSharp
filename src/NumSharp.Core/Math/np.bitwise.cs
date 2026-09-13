using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Compute the bit-wise AND of two arrays element-wise.
        /// Only integer and boolean types are handled (NumPy: float/complex inputs raise the no-loop TypeError).
        /// </summary>
        /// <param name="x1">First input array.</param>
        /// <param name="x2">Second input array.</param>
        /// <param name="@out">A location into which the result is stored (must broadcast with the inputs without being stretched; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <returns>Result. This is a scalar if both x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.bitwise_and.html</remarks>
        public static NDArray bitwise_and(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.BitwiseAnd(x1, x2, dtype, @out, where);

        /// <summary>
        /// Compute the bit-wise OR of two arrays element-wise.
        /// Only integer and boolean types are handled (NumPy: float/complex inputs raise the no-loop TypeError).
        /// </summary>
        /// <param name="x1">First input array.</param>
        /// <param name="x2">Second input array.</param>
        /// <param name="@out">A location into which the result is stored (must broadcast with the inputs without being stretched; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <returns>Result. This is a scalar if both x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.bitwise_or.html</remarks>
        public static NDArray bitwise_or(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.BitwiseOr(x1, x2, dtype, @out, where);

        /// <summary>
        /// Compute the bit-wise XOR of two arrays element-wise.
        /// Only integer and boolean types are handled (NumPy: float/complex inputs raise the no-loop TypeError).
        /// </summary>
        /// <param name="x1">First input array.</param>
        /// <param name="x2">Second input array.</param>
        /// <param name="@out">A location into which the result is stored (must broadcast with the inputs without being stretched; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <returns>Result. This is a scalar if both x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.bitwise_xor.html</remarks>
        public static NDArray bitwise_xor(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.BitwiseXor(x1, x2, dtype, @out, where);

        /// <summary>
        /// Compute the number of 1-bits in the absolute value of each element (NumPy 2.0's np.bitwise_count).
        /// Only integer and boolean types are handled — float/complex/decimal inputs raise a TypeError, and a
        /// <paramref name="dtype"/> other than uint8 raises the no-loop TypeError (every loop outputs uint8).
        /// Signed negatives count the MAGNITUDE, so bitwise_count(-1) == 1 (not 8), and the signed minimum
        /// counts as 1 (its two's-complement negation overflows to itself, matching NumPy's <c>a &lt; 0 ? -a : a</c>).
        /// </summary>
        /// <param name="x">Input array. Only integer and boolean types are handled.</param>
        /// <param name="@out">A location into which the result is stored (uint8 loop result cast same_kind into it; a wider integer out is accepted).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Output dtype request; only uint8 (or null) is a valid loop.</param>
        /// <returns>A uint8 array of set-bit counts. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.bitwise_count.html</remarks>
        public static NDArray bitwise_count(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.BitwiseCount(x, dtype, @out, where);
    }
}
