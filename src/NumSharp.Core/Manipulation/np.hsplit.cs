using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Split an array into multiple sub-arrays horizontally (column-wise).
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices_or_sections">
        /// If an integer, N, the array will be divided into N equal arrays along the axis.
        /// If such a split is not possible, an error is raised.
        /// </param>
        /// <returns>A list of sub-arrays as views into ary.</returns>
        /// <remarks>
        /// For 1-D arrays, splits along axis 0.
        /// For 2-D+ arrays, splits along axis 1 (columns).
        /// https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html
        /// </remarks>
        public static NDArray[] hsplit(NDArray ary, int indices_or_sections)
        {
            if (ary.ndim == 0)
                throw new ArgumentException("hsplit only works on arrays of 1 or more dimensions");

            int axis = ary.ndim > 1 ? 1 : 0;
            return split(ary, indices_or_sections, axis);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays horizontally (column-wise).
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// A 1-D array of sorted integers indicating where along the axis the array is split.
        /// For example, [2, 3] would result in ary[:2], ary[2:3], ary[3:].
        /// </param>
        /// <returns>A list of sub-arrays as views into ary.</returns>
        /// <remarks>
        /// For 1-D arrays, splits along axis 0.
        /// For 2-D+ arrays, splits along axis 1 (columns).
        /// https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html
        /// </remarks>
        public static NDArray[] hsplit(NDArray ary, int[] indices)
        {
            if (ary.ndim == 0)
                throw new ArgumentException("hsplit only works on arrays of 1 or more dimensions");

            int axis = ary.ndim > 1 ? 1 : 0;
            return split(ary, indices, axis);
        }
    }
}
