using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.<br></br>
        ///     Matches NumPy 2.x signature: <c>clip(a, a_min=None, a_max=None, out=None, *, min=None, max=None)</c>. Either or both bounds may be <c>null</c>. The <paramref name="min"/> and <paramref name="max"/> keyword aliases (added in NumPy 2.0) are accepted; mixing <paramref name="a_min"/> with <paramref name="min"/> (or <paramref name="a_max"/> with <paramref name="max"/>) throws.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_min">Minimum value. If null, clipping is not performed on lower interval edge.</param>
        /// <param name="a_max">Maximum value. If null, clipping is not performed on upper interval edge.</param>
        /// <param name="out">The results will be placed in this array. It may be the input array for in-place clipping. <paramref name="out"/> must be of the right shape to hold the output. Its type is preserved.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <param name="min">NumPy 2.x keyword alias for <paramref name="a_min"/>. Cannot be combined with <paramref name="a_min"/>.</param>
        /// <param name="max">NumPy 2.x keyword alias for <paramref name="a_max"/>. Cannot be combined with <paramref name="a_max"/>.</param>
        /// <returns>An array with the elements of a, but where values &lt; min are replaced with min, and those &gt; max with max.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.clip.html</remarks>
        public static NDArray clip(
            NDArray a,
            NDArray a_min = null,
            NDArray a_max = null,
            NDArray @out = null,
            NPTypeCode? dtype = null,
            NDArray min = null,
            NDArray max = null)
        {
            if (a_min is not null && min is not null)
                throw new ArgumentException("clip(): cannot specify both 'a_min' and 'min'.");
            if (a_max is not null && max is not null)
                throw new ArgumentException("clip(): cannot specify both 'a_max' and 'max'.");

            var lo = a_min ?? min;
            var hi = a_max ?? max;
            var result = a.TensorEngine.ClipNDArray(a, lo, hi, dtype, @out);
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
        ///     Clip (limit) the values in an array, returning a result of the requested CLR <see cref="Type"/>.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_min">Minimum value. If null, clipping is not performed on lower interval edge.</param>
        /// <param name="a_max">Maximum value. If null, clipping is not performed on upper interval edge.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>An array with the elements of a, but where values &lt; a_min are replaced with a_min, and those &gt; a_max with a_max.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.clip.html</remarks>
        public static NDArray clip(NDArray a, NDArray a_min, NDArray a_max, Type dtype)
            => a.TensorEngine.ClipNDArray(a, a_min, a_max, dtype);
    }
}
