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
        public static NDArray bitwise_and(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null)
            => x1.TensorEngine.BitwiseAnd(x1, x2, @out, where);

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
        public static NDArray bitwise_or(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null)
            => x1.TensorEngine.BitwiseOr(x1, x2, @out, where);

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
        public static NDArray bitwise_xor(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null)
            => x1.TensorEngine.BitwiseXor(x1, x2, @out, where);
    }
}
