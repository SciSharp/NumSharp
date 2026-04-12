using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Split an array into multiple sub-arrays vertically (row-wise).
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices_or_sections">
        /// If an integer, N, the array will be divided into N equal arrays along axis 0.
        /// If such a split is not possible, an error is raised.
        /// </param>
        /// <returns>A list of sub-arrays as views into ary.</returns>
        /// <remarks>
        /// Equivalent to split with axis=0.
        /// Array must have ndim >= 2.
        /// https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html
        /// </remarks>
        public static NDArray[] vsplit(NDArray ary, int indices_or_sections)
        {
            if (ary.ndim < 2)
                throw new ArgumentException("vsplit only works on arrays of 2 or more dimensions");

            return split(ary, indices_or_sections, axis: 0);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays vertically (row-wise).
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// A 1-D array of sorted integers indicating where along axis 0 the array is split.
        /// For example, [2, 3] would result in ary[:2], ary[2:3], ary[3:].
        /// </param>
        /// <returns>A list of sub-arrays as views into ary.</returns>
        /// <remarks>
        /// Equivalent to split with axis=0.
        /// Array must have ndim >= 2.
        /// https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html
        /// </remarks>
        public static NDArray[] vsplit(NDArray ary, int[] indices)
        {
            if (ary.ndim < 2)
                throw new ArgumentException("vsplit only works on arrays of 2 or more dimensions");

            return split(ary, indices, axis: 0);
        }
    }
}
