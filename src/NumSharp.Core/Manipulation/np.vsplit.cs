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
        /// <returns>A list of sub-arrays as views into ary.
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).</returns>
        /// <remarks>
        /// Equivalent to split with axis=0.
        /// Array must have ndim >= 2.
        /// https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="ary"/> has fewer than two dimensions ("vsplit only works on
        ///     arrays of 2 or more dimensions"; NumPy raises the text as a ValueError).</exception>
        /// <exception cref="ValueError">The sections are not positive ("number sections must be larger than 0."), or do not
        ///     divide the axis ("array split does not result in an equal division").</exception>
        /// <exception cref="NullReferenceException"><paramref name="ary"/> is null (its rank is read first).</exception>
        public static NDArray[] vsplit(NDArray ary, int indices_or_sections)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(ary))
                return NDMatrixBoundary.WrappedAll((ary, indices_or_sections), static s => vsplit(s.ary, s.indices_or_sections));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(ary))
                return NDCharArrayBoundary.KeptAll((ary, indices_or_sections), static s => vsplit(s.ary, s.indices_or_sections));

            // NumPy's statements (the rank check, split along axis 0) are the array object's.
            return NDArray.vsplit(ary, indices_or_sections);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays vertically (row-wise).
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// A 1-D array of sorted integers indicating where along axis 0 the array is split.
        /// For example, [2, 3] would result in ary[:2], ary[2:3], ary[3:].
        /// </param>
        /// <returns>A list of sub-arrays as views into ary.
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).</returns>
        /// <remarks>
        /// Equivalent to split with axis=0.
        /// Array must have ndim >= 2.
        /// https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="ary"/> has fewer than two dimensions ("vsplit only works on
        ///     arrays of 2 or more dimensions"; NumPy raises the text as a ValueError).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="indices"/> is null (checked after the rank).</exception>
        /// <exception cref="NullReferenceException"><paramref name="ary"/> is null (its rank is read first).</exception>
        public static NDArray[] vsplit(NDArray ary, int[] indices)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(ary))
                return NDMatrixBoundary.WrappedAll((ary, indices), static s => vsplit(s.ary, s.indices));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(ary))
                return NDCharArrayBoundary.KeptAll((ary, indices), static s => vsplit(s.ary, s.indices));

            // NumPy's statements (the rank check, split along axis 0) are the array object's.
            return NDArray.vsplit(ary, indices);
        }
    }
}
