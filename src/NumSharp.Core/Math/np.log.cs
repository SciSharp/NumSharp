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

        /// <summary>
        ///     Base-2 logarithm of x.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>Base-2 logarithm of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log2.html</remarks>
        public static NDArray log2(in NDArray x, Type outType) => x.TensorEngine.Log2(x);

        /// <summary>
        ///     Base-2 logarithm of x.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>Base-2 logarithm of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log2.html</remarks>
        public static NDArray log2(in NDArray x, NPTypeCode? outType = null) => x.TensorEngine.Log2(x, outType);

        /// <summary>
        ///     Base-2 logarithm of x.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>Base-2 logarithm of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log2.html</remarks>
        public static NDArray log2(in NDArray x) => x.TensorEngine.Log2(x);
                
        /// <summary>
        ///     Return the base 10 logarithm of the input array, element-wise.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>The logarithm to the base 10 of x, element-wise. NaNs are returned where x is negative. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log10.html</remarks>
        public static NDArray log10(in NDArray x, Type outType) => x.TensorEngine.Log10(x);

        /// <summary>
        ///     Return the base 10 logarithm of the input array, element-wise.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>The logarithm to the base 10 of x, element-wise. NaNs are returned where x is negative. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log10.html</remarks>
        public static NDArray log10(in NDArray x, NPTypeCode? outType = null) => x.TensorEngine.Log10(x, outType);

        /// <summary>
        ///     Return the base 10 logarithm of the input array, element-wise.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>The logarithm to the base 10 of x, element-wise. NaNs are returned where x is negative. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log10.html</remarks>
        public static NDArray log10(in NDArray x) => x.TensorEngine.Log10(x);

        /// <summary>
        ///     Return the natural logarithm of one plus the input array, element-wise.<br></br>
        ///     Calculates log(1 + x).
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>Natural logarithm of 1 + x, element-wise. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log1p.html</remarks>
        public static NDArray log1p(in NDArray x, Type outType) => x.TensorEngine.Log1p(x);

        /// <summary>
        ///     Return the natural logarithm of one plus the input array, element-wise.<br></br>
        ///     Calculates log(1 + x).
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>Natural logarithm of 1 + x, element-wise. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log1p.html</remarks>
        public static NDArray log1p(in NDArray x, NPTypeCode? outType = null) => x.TensorEngine.Log1p(x, outType);

        /// <summary>
        ///     Return the natural logarithm of one plus the input array, element-wise.<br></br>
        ///     Calculates log(1 + x).
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>Natural logarithm of 1 + x, element-wise. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log1p.html</remarks>
        public static NDArray log1p(in NDArray x) => x.TensorEngine.Log1p(x);
    }
}
