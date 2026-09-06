using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the non-negative square-root of an array, element-wise.
        ///     Mirrors NumPy's ufunc signature: <c>sqrt(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">The values whose square-roots are required.</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the computation runs at this precision; integer/bool requests raise NumPy's "No loop matching" error.</param>
        /// <returns>An array of the same shape as x, containing the positive square-root of each element in x. If any element in x is complex, a complex array is returned (and the square-roots of negative reals are calculated). If all of the elements in x are real, so is y, with negative elements returning nan. If out was provided, y is a reference to it. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html</remarks>
        public static NDArray sqrt(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.Sqrt(x, dtype, @out, where);
    }
}
