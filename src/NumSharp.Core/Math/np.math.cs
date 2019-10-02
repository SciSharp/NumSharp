using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.add.html</remarks>
        public static NDArray add(in NDArray x1, in NDArray x2)
            => x1.TensorEngine.Add(x1, x2);

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.divide.html</remarks>
        public static NDArray divide(in NDArray x1, in NDArray x2)
            => x1.TensorEngine.Divide(x1, x2);

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.true_divide.html</remarks>
        public static NDArray true_divide(in NDArray x1, in NDArray x2) 
            => x1.TensorEngine.Divide(x1, x2);

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html</remarks>
        public static NDArray multiply(in NDArray x1, in NDArray x2)
            => x1.TensorEngine.Multiply(x1, x2);

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.subtract.html</remarks>
        public static NDArray subtract(in NDArray x1, in NDArray x2)
            => x1.TensorEngine.Subtract(x1, x2);

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.mod.html</remarks>
        public static NDArray mod(in NDArray x1, in NDArray x2)
            => x1.TensorEngine.Mod(x1, x2);

        public static NDArray mod(in NDArray x1, in float x2)
            => x1.TensorEngine.Mod(x1, x2);

        /// <summary>
        /// Returns the discrete, linear convolution of two one-dimensional sequences.
        ///
        /// The convolution operator is often seen in signal processing, where it models the effect of a linear time-invariant system on a signal[1]. In probability theory, the sum of two independent random variables is distributed according to the convolution of their individual distributions.
        /// 
        /// If v is longer than a, the arrays are swapped before computation.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="v"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static NDArray convolve(NDArray a, NDArray v, string mode = "full")
            => a.convolve(v, mode);

        /// <summary>
        ///     Return the product of array elements over a given axis.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis or axes along which a product is performed. The default, axis=None, will calculate the product of all the elements in the input array. If axis is negative it counts from the last to the first axis.</param>
        /// <param name="dtype">The type of the returned array, as well as of the accumulator in which the elements are multiplied. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.</param>
        /// <returns>An array shaped as a but with the specified axis removed.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.prod.html</remarks>
        public static NDArray prod(in NDArray a, int? axis = null, Type dtype = null, bool keepdims = false) //todo impl a version with keepdims
            => a.TensorEngine.ReduceProduct(a, axis, keepdims, dtype != null ? dtype.GetTypeCode() : (NPTypeCode?)null);

        /// <summary>
        ///     Numerical positive, element-wise.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.positive.html</remarks>
        public static NDArray positive(in NDArray nd)
            => nd.positive();

        /// <summary>
        ///     Numerical negative, element-wise.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.negative.html</remarks>
        public static NDArray negative(in NDArray nd)
            => nd.negative();
    }
}
