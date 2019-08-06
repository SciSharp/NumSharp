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
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.reshape.html</remarks>
        public static NDArray reshape(NDArray nd, params int[] shape)
        {
            return nd.reshape(shape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="nd">Array to be reshaped.</param>
        /// <param name="shape">The new shape should be compatible with the original shape. </param>
        /// <returns>original <paramref name="nd"/> reshaped without copying.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.reshape.html</remarks>
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
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.reshape.html</remarks>
        public static NDArray reshape(NDArray nd, ref Shape shape)
        {
            return nd.reshape(ref shape);
        }
    }
}
