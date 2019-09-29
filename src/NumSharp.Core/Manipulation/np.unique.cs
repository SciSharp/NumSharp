namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the unique elements of an array.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html</remarks>
        public static NDArray unique(in NDArray a)
            => a.unique();
    }
}