using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the variance along the specified axis.
        ///     Returns the variance of the array elements, a measure of the spread of a distribution.
        ///     The variance is computed for the flattened array by default, otherwise over the specified axis.
        /// </summary>
        /// <param name="a">Array containing numbers whose variance is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the variance is computed. The default is to compute the variance of the flattened array.</param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        ///     With this option, the result will broadcast correctly against the input array.
        /// </param>
        /// <param name="ddof">Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.</param>
        /// <returns> returns a new array containing the var values, otherwise a reference to the output array is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html</remarks>
        public static NDArray var(NDArray a, bool keepdims = false, int? ddof = null, NPTypeCode? dtype = null)
        {
            return a.TensorEngine.ReduceVar(a, null, keepdims, ddof, dtype);
        }

        /// <summary>
        ///     Compute the variance along the specified axis.
        ///     Returns the variance of the array elements, a measure of the spread of a distribution.
        ///     The variance is computed for the flattened array by default, otherwise over the specified axis.
        /// </summary>
        /// <param name="a">Array containing numbers whose variance is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the variance is computed. The default is to compute the variance of the flattened array.</param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        ///     With this option, the result will broadcast correctly against the input array.
        /// </param>
        /// <param name="ddof">Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.</param>
        /// <returns> returns a new array containing the var values, otherwise a reference to the output array is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html</remarks>
        public static NDArray var(in NDArray a, int? ddof = null)
        => a.TensorEngine.ReduceVar(a, null);

        /// <summary>
        ///     Compute the variance along the specified axis.
        ///     Returns the variance of the array elements, a measure of the spread of a distribution.
        ///     The variance is computed for the flattened array by default, otherwise over the specified axis.
        /// </summary>
        /// <param name="a">Array containing numbers whose variance is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the variance is computed. The default is to compute the variance of the flattened array.</param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        ///     With this option, the result will broadcast correctly against the input array.
        /// </param>
        /// <param name="ddof">Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.</param>
        /// <returns> returns a new array containing the var values, otherwise a reference to the output array is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html</remarks>
        public static NDArray var(in NDArray a, int axis, int? ddof = null)
            => a.TensorEngine.ReduceVar(a, axis, ddof: ddof);

        /// <summary>
        ///     Compute the variance along the specified axis.
        ///     Returns the variance of the array elements, a measure of the spread of a distribution.
        ///     The variance is computed for the flattened array by default, otherwise over the specified axis.
        /// </summary>
        /// <param name="a">Array containing numbers whose variance is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the variance is computed. The default is to compute the variance of the flattened array.</param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        ///     With this option, the result will broadcast correctly against the input array.
        /// </param>
        /// <param name="ddof">Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.</param>
        /// <returns> returns a new array containing the var values, otherwise a reference to the output array is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html</remarks>
        public static NDArray var(in NDArray a, bool keepdims, int? ddof = null)
            => a.TensorEngine.ReduceVar(a, null, keepdims, ddof: ddof);

        /// <summary>
        ///     Compute the variance along the specified axis.
        ///     Returns the variance of the array elements, a measure of the spread of a distribution.
        ///     The variance is computed for the flattened array by default, otherwise over the specified axis.
        /// </summary>
        /// <param name="a">Array containing numbers whose variance is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the variance is computed. The default is to compute the variance of the flattened array.</param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        ///     With this option, the result will broadcast correctly against the input array.
        /// </param>
        /// <param name="ddof">Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.</param>
        /// <returns> returns a new array containing the var values, otherwise a reference to the output array is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html</remarks>
        public static NDArray var(in NDArray a, int axis, Type dtype, bool keepdims = false, int? ddof = null)
            => a.TensorEngine.ReduceVar(a, axis, keepdims, ddof, dtype != null ? dtype.GetTypeCode() : (NPTypeCode?)null);

        /// <summary>
        ///     Compute the variance along the specified axis.
        ///     Returns the variance of the array elements, a measure of the spread of a distribution.
        ///     The variance is computed for the flattened array by default, otherwise over the specified axis.
        /// </summary>
        /// <param name="a">Array containing numbers whose variance is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the variance is computed. The default is to compute the variance of the flattened array.</param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        ///     With this option, the result will broadcast correctly against the input array.
        /// </param>
        /// <param name="ddof">Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.</param>
        /// <returns> returns a new array containing the var values, otherwise a reference to the output array is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html</remarks>
        public static NDArray var(in NDArray a, int axis, NPTypeCode type, bool keepdims = false, int? ddof = null)
            => a.TensorEngine.ReduceVar(a, axis, keepdims, ddof, type);

        /// <summary>
        ///     Compute the variance along the specified axis.
        ///     Returns the variance of the array elements, a measure of the spread of a distribution.
        ///     The variance is computed for the flattened array by default, otherwise over the specified axis.
        /// </summary>
        /// <param name="a">Array containing numbers whose variance is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the variance is computed. The default is to compute the variance of the flattened array.</param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        ///     With this option, the result will broadcast correctly against the input array.
        /// </param>
        /// <param name="ddof">Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.</param>
        /// <returns> returns a new array containing the var values, otherwise a reference to the output array is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html</remarks>
        public static NDArray var(in NDArray a, int axis, bool keepdims, int? ddof = null)
            => a.TensorEngine.ReduceVar(a, axis, keepdims, ddof: ddof);

        /// <summary>
        ///     Compute the variance along the specified axis.
        ///     Returns the variance of the array elements, a measure of the spread of a distribution.
        ///     The variance is computed for the flattened array by default, otherwise over the specified axis.
        /// </summary>
        /// <param name="a">Array containing numbers whose variance is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the variance is computed. The default is to compute the variance of the flattened array.</param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        ///     With this option, the result will broadcast correctly against the input array.
        /// </param>
        /// <param name="ddof">Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.</param>
        /// <returns> returns a new array containing the var values, otherwise a reference to the output array is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html</remarks>
        public static NDArray var(NDArray a, int axis, bool keepdims = false, int? ddof = null, NPTypeCode? dtype = null)
        {
            return a.TensorEngine.ReduceVar(a, axis, keepdims, ddof, dtype);
        }
    }
}
