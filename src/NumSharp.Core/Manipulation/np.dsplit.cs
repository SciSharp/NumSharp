using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Split array into multiple sub-arrays along the 3rd axis (depth).
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices_or_sections">
        /// If an integer, N, the array will be divided into N equal arrays along axis 2.
        /// If such a split is not possible, an error is raised.
        /// </param>
        /// <returns>A list of sub-arrays as views into ary.</returns>
        /// <remarks>
        /// Equivalent to split with axis=2.
        /// Array must have ndim >= 3.
        /// https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html
        /// </remarks>
        public static NDArray[] dsplit(NDArray ary, int indices_or_sections)
        {
            if (ary.ndim < 3)
                throw new ArgumentException("dsplit only works on arrays of 3 or more dimensions");

            return split(ary, indices_or_sections, axis: 2);
        }

        /// <summary>
        /// Split array into multiple sub-arrays along the 3rd axis (depth).
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// A 1-D array of sorted integers indicating where along axis 2 the array is split.
        /// For example, [2, 3] would result in ary[:,:,:2], ary[:,:,2:3], ary[:,:,3:].
        /// </param>
        /// <returns>A list of sub-arrays as views into ary.</returns>
        /// <remarks>
        /// Equivalent to split with axis=2.
        /// Array must have ndim >= 3.
        /// https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html
        /// </remarks>
        public static NDArray[] dsplit(NDArray ary, int[] indices)
        {
            if (ary.ndim < 3)
                throw new ArgumentException("dsplit only works on arrays of 3 or more dimensions");

            return split(ary, indices, axis: 2);
        }
    }
}
