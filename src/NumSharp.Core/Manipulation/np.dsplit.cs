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
        /// <returns>A list of sub-arrays as views into ary.
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).</returns>
        /// <remarks>
        /// Equivalent to split with axis=2.
        /// Array must have ndim >= 3.
        /// https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html
        /// </remarks>
        /// <exception cref="ArgumentException"><paramref name="ary"/> has fewer than three dimensions ("dsplit only works on
        ///     arrays of 3 or more dimensions"; NumPy raises the text as a ValueError) - a <c>numpy.matrix</c> always
        ///     does.</exception>
        /// <exception cref="ValueError">The sections are not positive ("number sections must be larger than 0."), or do not
        ///     divide the axis ("array split does not result in an equal division").</exception>
        /// <exception cref="NullReferenceException"><paramref name="ary"/> is null (its rank is read first).</exception>
        public static NDArray[] dsplit(NDArray ary, int indices_or_sections)
        {
            // NumPy's dsplit ends in split, whose class rules applied here when this called np.split: a numpy.matrix
            // operand comes back as matrices (NDMatrixBoundary; it never gets that far, a matrix having two axes), and a
            // numpy.char.chararray operand as chararrays (NDCharArrayBoundary).
            if (NDMatrixBoundary.Sees(ary))
                return NDMatrixBoundary.WrappedAll((ary, indices_or_sections), static s => dsplit(s.ary, s.indices_or_sections));
            if (NDCharArrayBoundary.Sees(ary))
                return NDCharArrayBoundary.KeptAll((ary, indices_or_sections), static s => dsplit(s.ary, s.indices_or_sections));

            // NumPy's statements (the rank check, split along axis 2) are the array object's.
            return NDArray.dsplit(ary, indices_or_sections);
        }

        /// <summary>
        /// Split array into multiple sub-arrays along the 3rd axis (depth).
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// A 1-D array of sorted integers indicating where along axis 2 the array is split.
        /// For example, [2, 3] would result in ary[:,:,:2], ary[:,:,2:3], ary[:,:,3:].
        /// </param>
        /// <returns>A list of sub-arrays as views into ary.
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).</returns>
        /// <remarks>
        /// Equivalent to split with axis=2.
        /// Array must have ndim >= 3.
        /// https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html
        /// </remarks>
        /// <exception cref="ArgumentException"><paramref name="ary"/> has fewer than three dimensions ("dsplit only works on
        ///     arrays of 3 or more dimensions"; NumPy raises the text as a ValueError) - a <c>numpy.matrix</c> always
        ///     does.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="indices"/> is null (checked after the rank).</exception>
        /// <exception cref="NullReferenceException"><paramref name="ary"/> is null (its rank is read first).</exception>
        public static NDArray[] dsplit(NDArray ary, int[] indices)
        {
            // np.split's class rules (see the section-count overload): a matrix never gets past the rank check, a
            // chararray operand gives chararrays.
            if (NDMatrixBoundary.Sees(ary))
                return NDMatrixBoundary.WrappedAll((ary, indices), static s => dsplit(s.ary, s.indices));
            if (NDCharArrayBoundary.Sees(ary))
                return NDCharArrayBoundary.KeptAll((ary, indices), static s => dsplit(s.ary, s.indices));

            // NumPy's statements (the rank check, split along axis 2) are the array object's.
            return NDArray.dsplit(ary, indices);
        }
    }
}
