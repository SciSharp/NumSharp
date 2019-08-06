namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return a copy of the array.
        /// </summary>
        /// <param name="order"></param>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.copy.html</remarks>
        public NDArray copy(char order = 'C') => Clone(); //TODO order support
    }
}
