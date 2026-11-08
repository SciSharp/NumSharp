namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="nd">Array to be reshaped.</param>
        /// <param name="shape">The new shape should be compatible with the original shape. </param>
        /// <returns>original <paramref name="nd"/> reshaped without copying.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray reshape(NDArray nd, int[] shape)
        {
            return nd.reshape(shape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="nd">Array to be reshaped.</param>
        /// <param name="shape">The new shape should be compatible with the original shape. </param>
        /// <returns>original <paramref name="nd"/> reshaped without copying.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray reshape(NDArray nd, long[] shape)
        {
            return nd.reshape(shape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="nd">Array to be reshaped.</param>
        /// <param name="shape">The new shape should be compatible with the original shape. </param>
        /// <returns>original <paramref name="nd"/> reshaped without copying.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray reshape(NDArray nd, Shape shape)
        {
            return nd.reshape(shape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="nd">Array to be reshaped.</param>
        /// <param name="shape">The new shape should be compatible with the original shape. </param>
        /// <returns>original <paramref name="nd"/> reshaped without copying.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public static NDArray reshape(NDArray nd, ref Shape shape)
        {
            return nd.reshape(ref shape);
        }
    }
}
