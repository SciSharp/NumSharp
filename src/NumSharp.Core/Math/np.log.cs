using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     Natural logarithm, element-wise.
        ///     The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x.
        ///     The natural logarithm is logarithm in base e.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The natural logarithm of x, element-wise. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html</remarks>
        public static NDArray log(in NDArray x, Type outType) => x.TensorEngine.Log(x);

        /// <summary>
        ///     Natural logarithm, element-wise.
        ///     The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x.
        ///     The natural logarithm is logarithm in base e.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The natural logarithm of x, element-wise. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html</remarks>
        public static NDArray log(in NDArray x, NPTypeCode? outType = null) => x.TensorEngine.Log(x, outType);

        /// <summary>
        ///     Natural logarithm, element-wise.
        ///     The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x.
        ///     The natural logarithm is logarithm in base e.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>The natural logarithm of x, element-wise. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html</remarks>
        public static NDArray log(in NDArray x) => x.TensorEngine.Log(x);

    }
}
