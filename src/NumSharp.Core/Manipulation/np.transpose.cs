namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Permute the dimensions of an array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="premute">By default, reverse the dimensions, otherwise permute the axes according to the values given.</param>
        /// <returns>a with its axes permuted. A view is returned whenever possible.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.transpose.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray transpose(NDArray a, int[] premute = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((a, premute), static s => transpose(s.a, s.premute))
            : a.TensorEngine.Transpose(a, premute);
    }
}