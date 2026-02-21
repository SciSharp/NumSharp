namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Permute the dimensions of an array.
        /// </summary>
        /// <param name="premute">By default, reverse the dimensions, otherwise permute the axes according to the values given.</param>
        /// <returns>a with its axes permuted. A view is returned whenever possible.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.transpose.html</remarks>
        public NDArray transpose(int[] premute = null)
            => np.transpose(this, premute);
    }
}
