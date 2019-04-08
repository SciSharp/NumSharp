using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray add(NDArray x, NDArray y)
            => BackendFactory.GetEngine().Add(x, y);

        public static NDArray divide(NDArray x, NDArray y)
            => BackendFactory.GetEngine().Add(x, y);

        public static NDArray multiply(NDArray x, NDArray y)
            => BackendFactory.GetEngine().Multiply(x, y);

        public static NDArray subtract(NDArray x, NDArray y)
            => BackendFactory.GetEngine().Sub(x, y);

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
        public static NDArray prod(NDArray nd, int axis = -1, Type dtype = null)
            => nd.prod(axis, dtype);
    }
}
