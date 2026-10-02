using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Element-wise minimum of array elements (NaN-PROPAGATING).
        ///     Compare two arrays and return a new array containing the element-wise minima. If either
        ///     element being compared is a NaN the NaN is returned (both NaN → the first); for complex
        ///     NaNs a NaN in the real OR imaginary part counts, so NaNs propagate. Use
        ///     <see cref="fmin(NDArray, NDArray, NDArray, NDArray, DType)"/> to IGNORE NaNs instead.
        ///     Mirrors NumPy's ufunc signature: <c>minimum(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">First operand. If <paramref name="x1"/>.shape != <paramref name="x2"/>.shape they must broadcast to a common shape (the output shape).</param>
        /// <param name="x2">Second operand. If shapes differ they must broadcast to a common shape.</param>
        /// <param name="@out">A location into which the result is stored (NumPy ufunc out=): it joins the broadcast without being stretched, must be same_kind-castable from the loop dtype, and is returned as-is (reference identity). A read-only/broadcast out is refused.</param>
        /// <param name="where">Boolean mask (NumPy ufunc where=): only mask-true elements are computed/written; masked-off <paramref name="@out"/> slots keep their prior contents. Must be a bool array (a non-bool mask raises NumPy's 'safe'-cast error).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the comparison runs at this precision.</param>
        /// <returns>The element-wise minimum of <paramref name="x1"/> and <paramref name="x2"/> (or <paramref name="@out"/> when supplied). A scalar if both inputs are scalars.</returns>
        /// <exception cref="ArgumentException"><paramref name="@out"/> cannot be same_kind-cast from the loop dtype, <paramref name="where"/> is not bool, or the shapes do not broadcast (NumPy-verbatim texts).</exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
        /// <para>
        /// BREAKING: the third positional argument is now <paramref name="@out"/> (NumPy's order), not
        /// <c>dtype</c>; pass <c>dtype:</c> by name. Every in-repo call form is unaffected — callers use
        /// either a positional <paramref name="@out"/> or the <c>dtype:</c> keyword.
        /// </para>
        /// </remarks>
        public static NDArray minimum(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Minimum(x1, x2, dtype, @out, where);

        /// <summary>
        ///     Element-wise minimum of array elements, IGNORING NaNs.
        ///     Compare two arrays and return a new array containing the element-wise minima. If one element
        ///     is a NaN the non-NaN element is returned (both NaN → the first), so NaNs are ignored where
        ///     possible. Use <see cref="minimum(NDArray, NDArray, NDArray, NDArray, DType)"/> to propagate
        ///     NaNs instead. Mirrors NumPy's ufunc signature: <c>fmin(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">First operand. If shapes differ they must broadcast to a common shape.</param>
        /// <param name="x2">Second operand. If shapes differ they must broadcast to a common shape.</param>
        /// <param name="@out">A location into which the result is stored (NumPy ufunc out=): joins the broadcast without being stretched, same_kind-castable from the loop dtype, returned as-is. A read-only/broadcast out is refused.</param>
        /// <param name="where">Boolean mask (NumPy ufunc where=): only mask-true elements are computed/written; masked-off <paramref name="@out"/> slots keep prior contents. Must be a bool array.</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the comparison runs at this precision.</param>
        /// <returns>The element-wise NaN-ignoring minimum (or <paramref name="@out"/> when supplied). A scalar if both inputs are scalars.</returns>
        /// <exception cref="ArgumentException"><paramref name="@out"/> is not same_kind-castable, <paramref name="where"/> is not bool, or the shapes do not broadcast (NumPy-verbatim texts).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fmin.html</remarks>
        public static NDArray fmin(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.FMin(x1, x2, dtype, @out, where);
    }
}
