namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="nd">Array to be reshaped.</param>
        /// <param name="shape">The new shape should be compatible with the original shape. </param>
        /// <returns>original <paramref name="nd"/> reshaped without copying.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public static NDArray reshape(NDArray nd, int[] shape)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(nd))
                return NDMatrixBoundary.Wrapped((nd, shape), static s => reshape(s.nd, s.shape));
            return nd.reshape(shape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="nd">Array to be reshaped.</param>
        /// <param name="shape">The new shape should be compatible with the original shape. </param>
        /// <returns>original <paramref name="nd"/> reshaped without copying.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public static NDArray reshape(NDArray nd, long[] shape)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(nd))
                return NDMatrixBoundary.Wrapped((nd, shape), static s => reshape(s.nd, s.shape));
            return nd.reshape(shape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="nd">Array to be reshaped.</param>
        /// <param name="shape">The new shape should be compatible with the original shape. </param>
        /// <returns>original <paramref name="nd"/> reshaped without copying.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public static NDArray reshape(NDArray nd, Shape shape)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(nd))
                return NDMatrixBoundary.Wrapped((nd, shape), static s => reshape(s.nd, s.shape));
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
            // a numpy.matrix operand comes back as a matrix: the by-value overload's rule (a Shape is immutable, so
            // the copy it takes loses nothing)
            if (NDMatrixBoundary.Sees(nd))
                return reshape(nd, shape);
            return nd.reshape(ref shape);
        }
    }
}
