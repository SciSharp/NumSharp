namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the unique elements of an array.<br></br>
        ///     
        ///     Returns the sorted unique elements of an array.There are three optional outputs in addition to the unique elements:<br></br>
        ///     * the indices of the input array that give the unique values<br></br>
        ///     * the indices of the unique array that reconstruct the input array<br></br>
        ///     * the number of times each unique value comes up in the input array<br></br>
        /// </summary>
        /// <returns>The sorted unique values.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html</remarks>
        public static NDArray unique(in NDArray a)
            => a.unique();
    }
}
