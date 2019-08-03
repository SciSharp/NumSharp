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
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An ndarray containing the absolute value of each element in x.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.absolute.html</remarks>
        public static NDArray absolute(in NDArray a, Type outType) => a.TensorEngine.Abs(a, outType);

        /// <summary>
        ///     Calculate the absolute value element-wise. <br></br>
        ///     np.abs is a shorthand for this function.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An ndarray containing the absolute value of each element in x.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.absolute.html</remarks>
        public static NDArray absolute(in NDArray a, NPTypeCode? outType) => a.TensorEngine.Abs(a, outType);

        /// <summary>
        ///     Calculate the absolute value element-wise. <br></br>
        ///     np.abs is a shorthand for this function.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <returns>An ndarray containing the absolute value of each element in x.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.absolute.html</remarks>
        public static NDArray absolute(in NDArray a) => a.TensorEngine.Abs(a);

        /// <summary>
        ///     Calculate the absolute value element-wise. <br></br>
        ///     np.abs is a shorthand for this function.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An ndarray containing the absolute value of each element in x.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.absolute.html</remarks>
        public static NDArray abs(in NDArray a, Type outType) => a.TensorEngine.Abs(a, outType);

        /// <summary>
        ///     Calculate the absolute value element-wise. <br></br>
        ///     np.abs is a shorthand for this function.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An ndarray containing the absolute value of each element in x.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.absolute.html</remarks>
        public static NDArray abs(in NDArray a, NPTypeCode? outType) => a.TensorEngine.Abs(a, outType);

        /// <summary>
        ///     Calculate the absolute value element-wise. <br></br>
        ///     np.abs is a shorthand for this function.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <returns>An ndarray containing the absolute value of each element in x.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.absolute.html</remarks>
        public static NDArray abs(in NDArray a) => a.TensorEngine.Abs(a);
    }
}
