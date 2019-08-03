using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     Base-e exponential, element-wise.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="dtype">The dtype of the returned NDArray</param>
        /// <returns>The natural logarithm of x, element-wise. This is a scalar NDArray.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html</remarks>
        public static NDArray exp(in NDArray a, Type dtype) => a.TensorEngine.Exp(a, dtype);

        /// <summary>
        ///     Base-e exponential, element-wise.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="typeCode">The dtype of the returned NDArray</param>
        /// <returns>The natural logarithm of x, element-wise. This is a scalar NDArray.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html</remarks>
        public static NDArray exp(in NDArray a, NPTypeCode typeCode) => a.TensorEngine.Exp(a, typeCode);

        /// <summary>
        ///     Base-e exponential, element-wise.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <returns>The natural logarithm of x, element-wise. This is a scalar NDArray.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html</remarks>
        public static NDArray exp(in NDArray a) => a.TensorEngine.Exp(a);
    }
}
