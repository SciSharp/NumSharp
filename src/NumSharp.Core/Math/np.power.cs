using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents.</param>
        /// <param name="dtype">The dtype of the returned NDArray</param>
        /// <returns>The bases in x1 raised to the exponents in x2. This is a scalar NDArray if both x1 and x2 are scalars.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.power.html</remarks>
        public static NDArray power(in NDArray x1, in ValueType x2, Type dtype) => x1.TensorEngine.Power(x1, x2, dtype);

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents.</param>
        /// <param name="typeCode">The dtype of the returned NDArray</param>
        /// <returns>The bases in x1 raised to the exponents in x2. This is a scalar NDArray if both x1 and x2 are scalars.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.power.html</remarks>
        public static NDArray power(in NDArray x1, in ValueType x2, NPTypeCode typeCode) => x1.TensorEngine.Power(x1, x2, typeCode);

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents.</param>
        /// <returns>The bases in x1 raised to the exponents in x2. This is a scalar NDArray if both x1 and x2 are scalars.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.power.html</remarks>
        public static NDArray power(in NDArray x1, in ValueType x2) => x1.TensorEngine.Power(x1, x2);

        /// <summary>
        ///     Return the element-wise square of the input.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <returns>Element-wise x*x, of the same shape and dtype as x. Returns scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html</remarks>
        public static NDArray square(in NDArray x) => x.TensorEngine.Power(x, 2);
    }
}
