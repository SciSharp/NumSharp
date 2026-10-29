namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Returns the indices that would sort an array.
        ///
        /// Perform an indirect sort along the given axis using the algorithm specified by the kind keyword.It returns an array of indices of the same shape as a that index data along the given axis in sorted order.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="nd"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static NDArray argsort<T>(NDArray nd, int axis = -1) where T : unmanaged
            => nd.argsort<T>(axis);

        /// <summary>
        ///     Returns the indices that would sort <paramref name="nd"/> along <paramref name="axis"/>
        ///     (default last; <c>null</c> flattens). Returns int64 indices. NumPy <c>np.argsort</c>.
        /// </summary>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray argsort(NDArray nd, int? axis = -1)
            => NDMatrixBoundary.Sees(nd)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((nd, axis), static s => argsort(s.nd, s.axis))
            : nd.argsort(axis);
    }
}
