namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return this array with each element rounded to the given number of decimals (round half to even).
        ///     Refer to <see cref="np.around(NDArray, int, NDArray)"/> for full documentation.
        /// </summary>
        /// <param name="decimals">Number of decimal places to round to (default 0). Negative values round to positions left of the decimal point.</param>
        /// <param name="out">Alternate output array in which to place the result. It must have the same shape as the expected output.</param>
        /// <returns>An array of the same type as this array, containing the rounded values.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.round.html</remarks>
        public NDArray round(int decimals = 0, NDArray @out = null)
            => np.around(this, decimals, @out);
    }
}
