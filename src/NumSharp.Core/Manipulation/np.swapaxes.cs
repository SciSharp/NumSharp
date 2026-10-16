namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Interchange two axes of an array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis1">First axis.</param>
        /// <param name="axis2">Second axis.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.
        /// </remarks>
        public static NDArray swapaxes(NDArray a, int axis1, int axis2)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((a, axis1, axis2), static s => swapaxes(s.a, s.axis1, s.axis2))
            : NDCharArrayBoundary.Sees(a)   // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            ? NDCharArrayBoundary.Kept((a, axis1, axis2), static s => swapaxes(s.a, s.axis1, s.axis2))
            : a.TensorEngine.SwapAxes(a, axis1, axis2);
    }
}