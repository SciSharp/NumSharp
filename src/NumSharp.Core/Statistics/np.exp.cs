using System;
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
        
        /// <summary>
        ///     Calculate 2**p for all p in the input array.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <returns>Element-wise 2 to the power x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp2.html</remarks>
        public static NDArray exp2(in NDArray a, Type dtype) => a.TensorEngine.Exp2(a, dtype);

        /// <summary>
        ///     Calculate 2**p for all p in the input array.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <returns>Element-wise 2 to the power x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp2.html</remarks>
        public static NDArray exp2(in NDArray a, NPTypeCode typeCode) => a.TensorEngine.Exp2(a, typeCode);

        /// <summary>
        ///     Calculate 2**p for all p in the input array.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <returns>Element-wise 2 to the power x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp2.html</remarks>
        public static NDArray exp2(in NDArray a) => a.TensorEngine.Exp2(a);        

        /// <summary>
        ///     Calculate exp(x) - 1 for all elements in the array.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <returns>Element-wise exponential minus one: out = exp(x) - 1. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.expm1.html</remarks>
        public static NDArray expm1(in NDArray a, Type dtype) => a.TensorEngine.Expm1(a, dtype);

        /// <summary>
        ///     Calculate exp(x) - 1 for all elements in the array.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <returns>Element-wise exponential minus one: out = exp(x) - 1. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.expm1.html</remarks>
        public static NDArray expm1(in NDArray a, NPTypeCode typeCode) => a.TensorEngine.Expm1(a, typeCode);

        /// <summary>
        ///     Calculate exp(x) - 1 for all elements in the array.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <returns>Element-wise exponential minus one: out = exp(x) - 1. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.expm1.html</remarks>
        public static NDArray expm1(in NDArray a) => a.TensorEngine.Expm1(a);
    }
}
