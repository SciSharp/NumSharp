using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the minimum of an array or minimum along an axis.
        /// </summary>
        /// <param name="a">Input data. </param>
        /// <typeparam name="T">the type expected as a return, cast is performed if necessary.</typeparam>
        /// <returns>Minimum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html</remarks>
        public static T amin<T>(in NDArray a) where T : unmanaged => a.amin<T>();

        /// <summary>
        ///     Return the minimum of an array or minimum along an axis.
        /// </summary>
        /// <param name="a">Input data. </param>
        /// <param name="axis">Axis or axes along which to operate. </param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.</param>
        /// <param name="dtype">the type expected as a return, null will remain the same dtype.</param>
        /// <returns>Minimum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html</remarks>
        public static NDArray amin(in NDArray a, int? axis = null, bool keepdims = false, Type dtype = null)
        {
            // Delegate to TensorEngine which handles keepdims for axis=null
            return a.TensorEngine.ReduceAMin(a, axis, keepdims, dtype?.GetTypeCode());
        }

        /// <summary>
        ///     Return the minimum of an array or minimum along an axis.
        /// </summary>
        /// <param name="a">Input data. </param>
        /// <param name="axis">Axis or axes along which to operate. </param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.</param>
        /// <param name="dtype">the type expected as a return, null will remain the same dtype.</param>
        /// <returns>Minimum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html</remarks>
        public static NDArray min(in NDArray a, int? axis = null, bool keepdims = false, Type dtype = null)
        {
            // Delegate to TensorEngine which handles keepdims for axis=null
            return a.TensorEngine.ReduceAMin(a, axis, keepdims, dtype?.GetTypeCode());
        }
    }
}
