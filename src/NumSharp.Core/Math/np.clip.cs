using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_max">Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="a_min">Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array with the elements of a, but where values &lt; a_min are replaced with a_min, and those &gt; a_max with a_max.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.clip.html</remarks>
        public static NDArray clip(in NDArray a, ValueType a_min, ValueType a_max, NPTypeCode? outType = null)
            => a.TensorEngine.Clip(a, a_min, a_max, outType);

        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_max">Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="a_min">Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array with the elements of a, but where values &lt; a_min are replaced with a_min, and those &gt; a_max with a_max.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.clip.html</remarks>
        public static NDArray clip(in NDArray a, ValueType a_min, ValueType a_max, Type outType)
            => a.TensorEngine.Clip(a, a_min, a_max, outType);

        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_max">Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="a_min">Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array with the elements of a, but where values &lt; a_min are replaced with a_min, and those &gt; a_max with a_max.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.clip.html</remarks>
        public static NDArray clip(in NDArray a, NDArray a_min, NDArray a_max, NPTypeCode? outType = null)
            => a.TensorEngine.ClipNDArray(a, a_min, a_max, outType);

        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_max">Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="a_min">Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array with the elements of a, but where values &lt; a_min are replaced with a_min, and those &gt; a_max with a_max.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.clip.html</remarks>
        public static NDArray clip(in NDArray a, NDArray a_min, NDArray a_max, Type outType)
            => a.TensorEngine.ClipNDArray(a, a_min, a_max, outType);

        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_max">Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="a_min">Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array with the elements of a, but where values &lt; a_min are replaced with a_min, and those &gt; a_max with a_max.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.clip.html</remarks>
        public static NDArray clip<TMax>(in NDArray a, NDArray a_min, TMax a_max, NPTypeCode? outType = null) where TMax : unmanaged
            => a.TensorEngine.ClipNDArray(a, a_min, NDArray.Scalar(a_max), outType);

        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_max">Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="a_min">Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array with the elements of a, but where values &lt; a_min are replaced with a_min, and those &gt; a_max with a_max.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.clip.html</remarks>
        public static NDArray clip<TMax>(in NDArray a, NDArray a_min, TMax a_max, Type outType) where TMax : unmanaged
            => a.TensorEngine.ClipNDArray(a, a_min, NDArray.Scalar(a_max), outType);

        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_max">Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="a_min">Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array with the elements of a, but where values &lt; a_min are replaced with a_min, and those &gt; a_max with a_max.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.clip.html</remarks>
        public static NDArray clip<TMin>(in NDArray a, TMin a_min, NDArray a_max, NPTypeCode? outType = null) where TMin : unmanaged
            => a.TensorEngine.ClipNDArray(a, NDArray.Scalar(a_min), a_max, outType);

        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_max">Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="a_min">Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array with the elements of a, but where values &lt; a_min are replaced with a_min, and those &gt; a_max with a_max.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.clip.html</remarks>
        public static NDArray clip<TMin>(in NDArray a, TMin a_min, NDArray a_max, Type outType) where TMin : unmanaged
            => a.TensorEngine.ClipNDArray(a, NDArray.Scalar(a_min), a_max, outType);
    }
}
