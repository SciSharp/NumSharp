using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray add(in NDArray x, in NDArray y)
            => BackendFactory.GetEngine().Add(x, y);

        public static NDArray divide(in NDArray x, in NDArray y)
            => BackendFactory.GetEngine().Divide(x, y);

        public static NDArray multiply(in NDArray x, in NDArray y)
            => BackendFactory.GetEngine().Multiply(x, y);

        public static NDArray subtract(in NDArray x, in NDArray y)
            => BackendFactory.GetEngine().Subtract(x, y);

        /// <summary>
        /// Sum of array elements over a given axis.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static NDArray sum(NDArray x, int? axis = null)
            => BackendFactory.GetEngine().Sum(x, axis: axis);

        /// <summary>
        /// Calculate the absolute value element-wise.
        /// np.abs is a shorthand for this function.
        /// </summary>
        public static NDArray absolute(NDArray x)
            => x.absolute();

        /// <summary>
        /// Calculate the absolute value element-wise.
        /// </summary>
        public static NDArray abs(NDArray x)
            => x.absolute();

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
        /// Natural logarithm, element-wise.
        ///
        /// The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x.The natural logarithm is logarithm in base e.
        /// </summary>
        /// <returns></returns>
        public static NDArray log(NDArray x)
            => x.log();

        /// <summary>
        /// Return the non-negative square-root of an array, element-wise.
        /// </summary>
        public static NDArray sqrt(NDArray x)
            => x.sqrt();

        /// <summary>
        /// Array elements raised to given powers, element-wise.
        /// </summary>
        public static NDArray power(NDArray x, ValueType y)
            => BackendFactory.GetEngine().Power(x, y);

        //public static NDArray power<T>(NDArray nd, T exponent)
        //    => nd.power<T>(exponent);

        /// <summary>
        /// Trigonometric sine, element-wise.
        /// </summary>
        public static NDArray sin(NDArray nd)
            => nd.sin();

        /// <summary>
        /// Return the product of array elements over a given axis.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.1/reference/generated/numpy.prod.html</remarks>
        public static NDArray prod(NDArray nd, int axis = -1, Type dtype = null) //todo impl a version with keepdims
            => nd.prod(axis, dtype);

        /// <summary>
        /// Numerical positive, element-wise.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.positive.html</remarks>
        public static NDArray positive(NDArray nd)
            => nd.positive();

        /// <summary>
        /// Numerical negative, element-wise.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.negative.html</remarks>
        public static NDArray negative(NDArray nd)
            => nd.negative();
    }
}
