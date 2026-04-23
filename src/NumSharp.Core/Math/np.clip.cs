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
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array with the elements of a, but where values &lt; a_min are replaced with a_min, and those &gt; a_max with a_max.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.clip.html</remarks>
        public static NDArray clip(NDArray a, NDArray a_min, NDArray a_max, NPTypeCode? dtype = null)
        {
            var result = a.TensorEngine.ClipNDArray(a, a_min, a_max, dtype);
            return PreserveFContigFromSource(a, result);
        }

        // Internal helper: after an element-wise op whose output inherits a's layout,
        // relay out to F-contig when the source is strictly F-contig and the result
        // came back as C-contig (current engine default).
        private static NDArray PreserveFContigFromSource(NDArray a, NDArray result)
        {
            // Note: NDArray overloads operator!=, so reference-compare via ReferenceEquals.
            if (!ReferenceEquals(result, null)
                && a.Shape.NDim > 1 && a.size > 1
                && a.Shape.IsFContiguous && !a.Shape.IsContiguous
                && result.Shape.NDim > 1 && !result.Shape.IsFContiguous)
            {
                return result.copy('F');
            }
            return result;
        }

        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_max">Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="a_min">Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array with the elements of a, but where values &lt; a_min are replaced with a_min, and those &gt; a_max with a_max.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.clip.html</remarks>
        public static NDArray clip(NDArray a, NDArray a_min, NDArray a_max, Type dtype)
            => a.TensorEngine.ClipNDArray(a, a_min, a_max, dtype);

        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_max">Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="a_min">Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and a_max may be None.</param>
        /// <param name="out">The results will be placed in this array. It may be the input array for in-place clipping. out must be of the right shape to hold the output. Its type is preserved.</param>
        /// <returns>An array with the elements of a, but where values &lt; a_min are replaced with a_min, and those &gt; a_max with a_max.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.clip.html</remarks>
        public static NDArray clip(NDArray a, NDArray a_min, NDArray a_max, NDArray @out)
            => a.TensorEngine.ClipNDArray(a, a_min, a_max, (NPTypeCode?)null, @out);
    }
}
