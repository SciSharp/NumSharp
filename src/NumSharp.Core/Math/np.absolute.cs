using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Calculate the absolute value element-wise. <br></br>
        ///     np.abs is a shorthand for this function.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An ndarray containing the absolute value of each element in x.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.absolute.html</remarks>
        public static NDArray absolute(NDArray a, Type dtype) => a.TensorEngine.Abs(a, dtype);

        /// <summary>
        ///     Calculate the absolute value element-wise. <br></br>
        ///     np.abs is a shorthand for this function.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An ndarray containing the absolute value of each element in x.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.absolute.html</remarks>
        public static NDArray absolute(NDArray a, NPTypeCode? dtype) => a.TensorEngine.Abs(a, dtype);

        /// <summary>
        ///     Calculate the absolute value element-wise. <br></br>
        ///     np.abs is a shorthand for this function.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <returns>An ndarray containing the absolute value of each element in x.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.absolute.html</remarks>
        public static NDArray absolute(NDArray a) => a.TensorEngine.Abs(a);

        /// <summary>
        ///     Calculate the absolute value element-wise.
        ///     Mirrors NumPy's ufunc signature: <c>absolute(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the input must be same_kind-castable to it; for complex input it selects the magnitude dtype (float kinds only).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.absolute.html</remarks>
        public static NDArray absolute(NDArray a, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => a.TensorEngine.Abs(a, dtype, @out, where);

        /// <summary>
        ///     Calculate the absolute value element-wise. <br></br>
        ///     np.abs is a shorthand for this function.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An ndarray containing the absolute value of each element in x.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.absolute.html</remarks>
        public static NDArray abs(NDArray a, Type dtype) => a.TensorEngine.Abs(a, dtype);

        /// <summary>
        ///     Calculate the absolute value element-wise. <br></br>
        ///     np.abs is a shorthand for this function.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An ndarray containing the absolute value of each element in x.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.absolute.html</remarks>
        public static NDArray abs(NDArray a, NPTypeCode? dtype) => a.TensorEngine.Abs(a, dtype);

        /// <summary>
        ///     Calculate the absolute value element-wise. <br></br>
        ///     np.abs is a shorthand for this function.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <returns>An ndarray containing the absolute value of each element in x.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.absolute.html</remarks>
        public static NDArray abs(NDArray a) => a.TensorEngine.Abs(a);

        /// <summary>
        ///     Calculate the absolute value element-wise (alias of <see cref="absolute(NDArray, NDArray, NDArray, NPTypeCode?)"/>).
        ///     Mirrors NumPy's ufunc signature: <c>absolute(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the input must be same_kind-castable to it; for complex input it selects the magnitude dtype (float kinds only).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.absolute.html</remarks>
        public static NDArray abs(NDArray a, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => a.TensorEngine.Abs(a, dtype, @out, where);
    }
}
