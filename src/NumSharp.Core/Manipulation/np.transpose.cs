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
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html</remarks>
        public static NDArray transpose(in NDArray a, int[] premute = null)
            => a.TensorEngine.Transpose(a, premute);
    }
}