using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Compute the arithmetic mean along the specified axis.
        ///     Returns the average of the array elements.
        ///     The average is taken over the flattened array by default, otherwise over the specified axis.
        ///     float64 intermediate and return values are used for integer inputs.
        /// </summary>
        /// <param name="nd">Array containing numbers whose mean is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.</param>
        /// <param name="type">Type to use in computing the mean. For integer inputs, the default is float64; for floating point inputs, it is the same as the input dtype.</param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.
        ///     If the default value is passed, then keepdims will not be passed through to the mean method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.
        /// </param>
        /// <returns> returns a new array containing the mean values, otherwise a reference to the output array is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html</remarks>
        public NDArray mean()
            => TensorEngine.Mean(this);

        /// <summary>
        ///     Compute the arithmetic mean along the specified axis.
        ///     Returns the average of the array elements.
        ///     The average is taken over the flattened array by default, otherwise over the specified axis.
        ///     float64 intermediate and return values are used for integer inputs.
        /// </summary>
        /// <param name="nd">Array containing numbers whose mean is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.</param>
        /// <param name="type">Type to use in computing the mean. For integer inputs, the default is float64; for floating point inputs, it is the same as the input dtype.</param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.
        ///     If the default value is passed, then keepdims will not be passed through to the mean method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.
        /// </param>
        /// <returns> returns a new array containing the mean values, otherwise a reference to the output array is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html</remarks>
        public NDArray mean(int axis)
            => TensorEngine.Mean(this, axis);

        /// <summary>
        ///     Compute the arithmetic mean along the specified axis.
        ///     Returns the average of the array elements.
        ///     The average is taken over the flattened array by default, otherwise over the specified axis.
        ///     float64 intermediate and return values are used for integer inputs.
        /// </summary>
        /// <param name="nd">Array containing numbers whose mean is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.</param>
        /// <param name="type">Type to use in computing the mean. For integer inputs, the default is float64; for floating point inputs, it is the same as the input dtype.</param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.
        ///     If the default value is passed, then keepdims will not be passed through to the mean method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.
        /// </param>
        /// <returns> returns a new array containing the mean values, otherwise a reference to the output array is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html</remarks>
        public NDArray mean(int axis, Type type, bool keepdims = false)
            => TensorEngine.Mean(this, axis, dtype, keepdims);

        /// <summary>
        ///     Compute the arithmetic mean along the specified axis.
        ///     Returns the average of the array elements.
        ///     The average is taken over the flattened array by default, otherwise over the specified axis.
        ///     float64 intermediate and return values are used for integer inputs.
        /// </summary>
        /// <param name="nd">Array containing numbers whose mean is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.</param>
        /// <param name="type">Type to use in computing the mean. For integer inputs, the default is float64; for floating point inputs, it is the same as the input dtype.</param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.
        ///     If the default value is passed, then keepdims will not be passed through to the mean method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.
        /// </param>
        /// <returns> returns a new array containing the mean values, otherwise a reference to the output array is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html</remarks>
        public NDArray mean(int axis, NPTypeCode type, bool keepdims = false)
            => TensorEngine.Mean(this, axis, type, keepdims);

        /// <summary>
        ///     Compute the arithmetic mean along the specified axis.
        ///     Returns the average of the array elements.
        ///     The average is taken over the flattened array by default, otherwise over the specified axis.
        ///     float64 intermediate and return values are used for integer inputs.
        /// </summary>
        /// <param name="nd">Array containing numbers whose mean is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.</param>
        /// <param name="type">Type to use in computing the mean. For integer inputs, the default is float64; for floating point inputs, it is the same as the input dtype.</param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.
        ///     If the default value is passed, then keepdims will not be passed through to the mean method of sub-classes of ndarray, however any non-default value will be.If the sub-class’ method does not implement keepdims any exceptions will be raised.
        /// </param>
        /// <returns> returns a new array containing the mean values, otherwise a reference to the output array is returned.</returns>
        public NDArray mean(int axis, bool keepdims)
            => TensorEngine.Mean(this, axis, null, keepdims);
        
    }
}
