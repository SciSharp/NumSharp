using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.add.html</remarks>
        public static NDArray add(in NDArray x1, in NDArray x2)
            => BackendFactory.GetEngine().Add(x1, x2);

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.divide.html</remarks>
        public static NDArray divide(in NDArray x1, in NDArray x2)
            => BackendFactory.GetEngine().Divide(x1, x2);

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.true_divide.html</remarks>
        public static NDArray true_divide(in NDArray x1, in NDArray x2)
        {
            return BackendFactory.GetEngine().Divide(x1, x2);
        }

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html</remarks>
        public static NDArray multiply(in NDArray x1, in NDArray x2)
            => BackendFactory.GetEngine().Multiply(x1, x2);

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.subtract.html</remarks>
        public static NDArray subtract(in NDArray x1, in NDArray x2)
            => BackendFactory.GetEngine().Subtract(x1, x2);

        /// <summary>
        ///     Sum of array elements over a given axis.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis. </param>
        /// <param name="dtype">The type of the returned array and of the accumulator in which the elements are summed. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. If the default value is passed, then keepdims will not be passed through to the sum method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.</param>
        /// <returns>An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a scalar is returned. If an output array is specified, a reference to out is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html</remarks>
        public static NDArray sum(in NDArray a)
            => BackendFactory.GetEngine().Sum(a, axis: null, typeCode: null, keepdims: false);

        /// <summary>
        ///     Sum of array elements over a given axis.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis. </param>
        /// <param name="dtype">The type of the returned array and of the accumulator in which the elements are summed. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. If the default value is passed, then keepdims will not be passed through to the sum method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.</param>
        /// <returns>An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a scalar is returned. If an output array is specified, a reference to out is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html</remarks>
        public static NDArray sum(in NDArray a, int axis)
            => BackendFactory.GetEngine().Sum(a, axis: axis, typeCode: null, keepdims: false);

        /// <summary>
        ///     Sum of array elements over a given axis.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis. </param>
        /// <param name="dtype">The type of the returned array and of the accumulator in which the elements are summed. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. If the default value is passed, then keepdims will not be passed through to the sum method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.</param>
        /// <returns>An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a scalar is returned. If an output array is specified, a reference to out is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html</remarks>
        public static NDArray sum(in NDArray a, bool keepdims)
            => BackendFactory.GetEngine().Sum(a, axis: null, typeCode: null, keepdims: keepdims);

        /// <summary>
        ///     Sum of array elements over a given axis.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis. </param>
        /// <param name="dtype">The type of the returned array and of the accumulator in which the elements are summed. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. If the default value is passed, then keepdims will not be passed through to the sum method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.</param>
        /// <returns>An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a scalar is returned. If an output array is specified, a reference to out is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html</remarks>
        public static NDArray sum(in NDArray a, int? axis, bool keepdims)
            => BackendFactory.GetEngine().Sum(a, axis: axis, typeCode: null, keepdims: keepdims);

        /// <summary>
        ///     Sum of array elements over a given axis.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis. </param>
        /// <param name="dtype">The type of the returned array and of the accumulator in which the elements are summed. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. If the default value is passed, then keepdims will not be passed through to the sum method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.</param>
        /// <returns>An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a scalar is returned. If an output array is specified, a reference to out is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html</remarks>
        public static NDArray sum(in NDArray a, int? axis, bool keepdims, Type dtype)
            => BackendFactory.GetEngine().Sum(a, axis: axis, typeCode: dtype?.GetTypeCode(), keepdims: keepdims);

        /// <summary>
        ///     Sum of array elements over a given axis.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis. </param>
        /// <param name="typeCode">The type of the returned array and of the accumulator in which the elements are summed. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. If the default value is passed, then keepdims will not be passed through to the sum method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.</param>
        /// <returns>An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a scalar is returned. If an output array is specified, a reference to out is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html</remarks>
        public static NDArray sum(in NDArray a, int? axis, bool keepdims, NPTypeCode? typeCode)
            => BackendFactory.GetEngine().Sum(a, axis: axis, typeCode: typeCode, keepdims: keepdims);

        /// <summary>
        ///     Sum of array elements over a given axis.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis. </param>
        /// <param name="dtype">The type of the returned array and of the accumulator in which the elements are summed. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. If the default value is passed, then keepdims will not be passed through to the sum method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.</param>
        /// <returns>An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a scalar is returned. If an output array is specified, a reference to out is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html</remarks>
        public static NDArray sum(in NDArray a, int? axis, Type dtype)
            => BackendFactory.GetEngine().Sum(a, axis: axis, typeCode: dtype?.GetTypeCode(), keepdims: false);

        /// <summary>
        ///     Sum of array elements over a given axis.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis. </param>
        /// <param name="typeCode">The type of the returned array and of the accumulator in which the elements are summed. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.</param>
        /// <returns>An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a scalar is returned. If an output array is specified, a reference to out is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html</remarks>
        public static NDArray sum(in NDArray a, int? axis, NPTypeCode? typeCode)
            => BackendFactory.GetEngine().Sum(a, axis: axis, typeCode: typeCode, keepdims: false);

        /// <summary>
        ///     Sum of array elements over a given axis.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis. </param>
        /// <param name="dtype">The type of the returned array and of the accumulator in which the elements are summed. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. If the default value is passed, then keepdims will not be passed through to the sum method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.</param>
        /// <returns>An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a scalar is returned. If an output array is specified, a reference to out is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html</remarks>
        public static NDArray sum(in NDArray a, Type dtype)
            => BackendFactory.GetEngine().Sum(a, axis: null, typeCode: dtype?.GetTypeCode(), keepdims: false);

        /// <summary>
        ///     Sum of array elements over a given axis.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis. </param>
        /// <param name="typeCode">The type of the returned array and of the accumulator in which the elements are summed. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. If the default value is passed, then keepdims will not be passed through to the sum method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.</param>
        /// <returns>An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a scalar is returned. If an output array is specified, a reference to out is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html</remarks>
        public static NDArray sum(in NDArray a, NPTypeCode? typeCode)
            => BackendFactory.GetEngine().Sum(a, axis: null, typeCode: typeCode, keepdims: false);

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
