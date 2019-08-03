using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the maximum of an array or maximum along an axis.
        /// </summary>
        /// <typeparam name="T">the type expected as a return, cast is performed if necessary.</typeparam>
        /// <returns>Maximum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html</remarks>
        public static T amax<T>(NDArray a) where T : unmanaged => a.amax<T>();

        /// <summary>
        ///     Return the maximum of an array or maximum along an axis.
        /// </summary>
        /// <param name="axis">Axis or axes along which to operate.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.</param>
        /// <param name="dtype">the type expected as a return, null will remain the same dtype.</param>
        /// <returns>Maximum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html</remarks>
        public static NDArray amax(NDArray a, int? axis = null, bool keepdims = false, Type dtype = null)
        {
            if (!axis.HasValue)
                return a.amax(dtype);
            return a.amax(axis.Value, keepdims, dtype);
        }

        /// <summary>
        ///     Return the maximum of an array or maximum along an axis.
        /// </summary>
        /// <param name="axis">Axis or axes along which to operate.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.</param>
        /// <param name="dtype">the type expected as a return, null will remain the same dtype.</param>
        /// <returns>Maximum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html</remarks>
        public static NDArray max(NDArray a, int? axis = null, bool keepdims = false, Type dtype = null)
        {
            if (!axis.HasValue)
                return a.amax(dtype);

            return a.amax(axis.Value, keepdims, dtype);
        }
    }
}