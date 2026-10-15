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
        ///     Returns the indices that would sort <paramref name="a"/> along <paramref name="axis"/> (default last;
        ///     <c>null</c> flattens). Returns int64 indices. NumPy's
        ///     <c>np.argsort(a, axis=-1, kind=None, order=None, *, stable=None)</c>.
        /// </summary>
        /// <param name="a">Array to sort.</param>
        /// <param name="axis">Axis to sort along; null sorts the flattened array.</param>
        /// <param name="kind">The sort algorithm (see <see cref="NDArray.argsort(int?, string, object, bool?)"/>): over text
        ///     the default and heapsort kinds give NumPy's introsort tie order, the stable kinds the stable one.</param>
        /// <param name="order">Field names; any value is refused (no NumSharp dtype has fields).</param>
        /// <param name="stable">True asks for the stable kind; cannot be combined with <paramref name="kind"/>.</param>
        /// <returns>A fresh, C-contiguous int64 array of indices.</returns>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/>, <paramref name="kind"/> given with
        ///     <paramref name="stable"/>, or an <paramref name="order"/>; or a <c>numpy.matrix</c> operand whose result
        ///     cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="AxisError">The axis is out of bounds.</exception>
        /// <exception cref="RuntimeError">A <c>StringDType</c> array with the heapsort kind (<c>unsupported sort kind</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">A <c>StringDType</c> array whose lines hold two or more
        ///     elements (the <c>StringDType</c> part of NumSharp's next phase).</exception>
        public static NDArray argsort(NDArray a, int? axis = -1, string kind = null, object order = null, bool? stable = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((a, axis, kind, order, stable), static s => argsort(s.a, s.axis, s.kind, s.order, s.stable))
            : a.argsort(axis, kind, order, stable);
    }
}
