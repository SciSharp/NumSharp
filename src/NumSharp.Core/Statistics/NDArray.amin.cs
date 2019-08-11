using System;
using System.Diagnostics.CodeAnalysis;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return the minimum of an array or minimum along an axis.
        /// </summary>
        /// <typeparam name="T">The expected return type, cast will be performed if necessary.</typeparam>
        /// <returns>Minimum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html</remarks>
        public T amin<T>() where T : unmanaged
        {
            return np.asscalar<T>(TensorEngine.AMin(this, null, typeof(T).GetTypeCode(), false));
        }
        
        /// <summary>
        ///     Return the minimum of an array or minimum along an axis.
        /// </summary>
        /// <param name="axis">Axis or axes along which to operate. </param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.</param>
        /// <param name="dtype">the type expected as a return, null will remain the same dtype.</param>
        /// <returns>Minimum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html</remarks>
        [SuppressMessage("ReSharper", "TooWideLocalVariableScope")]
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public NDArray amin(int axis, bool keepdims = false, Type dtype = null)
        {
            return TensorEngine.AMin(this, axis, dtype, keepdims);
        }

        /// <summary>
        ///     Return the minimum of an array or minimum along an axis.
        /// </summary>
        /// <param name="dtype">the type expected as a return, null will remain the same dtype.</param>
        /// <returns>Minimum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html</remarks>
        [SuppressMessage("ReSharper", "TooWideLocalVariableScope")]
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public NDArray amin(Type dtype = null)
        {
            return TensorEngine.AMin(this, null, dtype?.GetTypeCode(), false);
        }

        /// <summary>
        ///     Return the minimum of an array or minimum along an axis.
        /// </summary>
        /// <typeparam name="T">The expected return type, cast will be performed if necessary.</typeparam>
        /// <returns>Minimum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html</remarks>
        public T min<T>() where T : unmanaged
        {
            return amin<T>();
        }
        
        /// <summary>
        ///     Return the minimum of an array or minimum along an axis.
        /// </summary>
        /// <param name="axis">Axis or axes along which to operate. </param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.</param>
        /// <param name="dtype">the type expected as a return, null will remain the same dtype.</param>
        /// <returns>Minimum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html</remarks>
        [SuppressMessage("ReSharper", "TooWideLocalVariableScope")]
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public NDArray min(int axis, bool keepdims = false, Type dtype = null)
        {
            return amin(axis, keepdims, dtype);
        }

        /// <summary>
        ///     Return the minimum of an array or minimum along an axis.
        /// </summary>
        /// <param name="dtype">the type expected as a return, null will remain the same dtype.</param>
        /// <returns>Minimum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html</remarks>
        [SuppressMessage("ReSharper", "TooWideLocalVariableScope")]
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public NDArray min(Type dtype = null)
        {
            return amin(dtype);
        }
    }
}
