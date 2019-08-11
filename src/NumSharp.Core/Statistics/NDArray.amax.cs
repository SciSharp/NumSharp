using System;
using System.Diagnostics.CodeAnalysis;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return the maximum of an array or maximum along an axis.
        /// </summary>
        /// <typeparam name="T">The expected return type, cast will be performed if necessary.</typeparam>
        /// <returns>Maximum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html</remarks>
        public T amax<T>() where T : unmanaged
        {
            return np.asscalar<T>(TensorEngine.AMax(this, null, typeof(T).GetTypeCode(), false));
        }

        /// <summary>
        ///     Return the maximum of an array or maximum along an axis.
        /// </summary>
        /// <param name="axis">Axis or axes along which to operate.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.</param>
        /// <param name="dtype">the type expected as a return, null will remain the same dtype.</param>
        /// <returns>Maximum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html</remarks>
        [SuppressMessage("ReSharper", "TooWideLocalVariableScope")]
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public NDArray amax(int axis, bool keepdims = false, Type dtype = null)
        {
            return TensorEngine.AMax(this, axis, dtype, keepdims);
        }

        /// <summary>
        ///     Return the maximum of an array or maximum along an axis.
        /// </summary>
        /// <param name="dtype">the type expected as a return, null will remain the same dtype.</param>
        /// <returns>Maximum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html</remarks>
        [SuppressMessage("ReSharper", "TooWideLocalVariableScope")]
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public NDArray amax(Type dtype = null)
        {
            return TensorEngine.AMax(this, null, dtype?.GetTypeCode(), false);
        }

        /// <summary>
        ///     Return the maximum of an array or maximum along an axis.
        /// </summary>
        /// <typeparam name="T">The expected return type, cast will be performed if necessary.</typeparam>
        /// <returns>Maximum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html</remarks>
        public T max<T>() where T : unmanaged
        {
            return amax<T>();
        }

        /// <summary>
        ///     Return the maximum of an array or maximum along an axis.
        /// </summary>
        /// <param name="axis">Axis or axes along which to operate.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.</param>
        /// <param name="dtype">the type expected as a return, null will remain the same dtype.</param>
        /// <returns>Maximum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html</remarks>
        [SuppressMessage("ReSharper", "TooWideLocalVariableScope")]
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public NDArray max(int axis, bool keepdims = false, Type dtype = null)
        {
            return amax(axis, keepdims, dtype);
        }

        /// <summary>
        ///     Return the maximum of an array or maximum along an axis.
        /// </summary>
        /// <param name="dtype">the type expected as a return, null will remain the same dtype.</param>
        /// <returns>Maximum of a. If axis is None, the result is a scalar value. If axis is given, the result is an array of dimension a.ndim - 1.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html</remarks>
        [SuppressMessage("ReSharper", "TooWideLocalVariableScope")]
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public NDArray max(Type dtype = null)
        {
            return amax(dtype);
        }
    }
}
