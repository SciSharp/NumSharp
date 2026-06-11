using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return the cube-root of an array, element-wise.
        /// </summary>
        /// <param name="x">The values whose cube-roots are required.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>An array of the same shape as x, containing the cube root of each element.
        /// If x contains negative values, the result contains the (negative) real cube root.
        /// This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cbrt.html</remarks>
        public static NDArray cbrt(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x.TensorEngine.Cbrt(x, dtype, @out, where);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience
        ///     overload (NumPy accepts dtype only as a keyword).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The loop dtype the computation should run in.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cbrt.html</remarks>
        public static NDArray cbrt(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.Cbrt(x, dtype);

        /// <summary>
        /// Return the cube-root of an array, element-wise.
        /// </summary>
        /// <param name="x">The values whose cube-roots are required.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>An array of the same shape as x, containing the cube root of each element.
        /// If x contains negative values, the result contains the (negative) real cube root.
        /// This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cbrt.html</remarks>
        public static NDArray cbrt(NDArray x, Type dtype)
            => x.TensorEngine.Cbrt(x, dtype);
    }
}
