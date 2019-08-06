namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     Return a copy of the array.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="order"></param>
        /// <returns>Array interpretation of a.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.copy.html</remarks>
        public static NDArray copy(NDArray a, char order = 'C') => a.copy(); //TODO order support
    }
}
