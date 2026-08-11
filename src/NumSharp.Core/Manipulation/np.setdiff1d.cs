namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the set difference of two arrays.<br></br>
        ///     Return the unique values in <paramref name="ar1"/> that are not in <paramref name="ar2"/>.
        /// </summary>
        /// <param name="ar1">Input array.</param>
        /// <param name="ar2">Input comparison array.</param>
        /// <param name="assume_unique">If True, the input arrays are both assumed to be unique, which can speed up
        ///     the calculation. Default is False.</param>
        /// <returns>1-D array of values in <paramref name="ar1"/> that are not in <paramref name="ar2"/>. Sorted
        ///     when <paramref name="assume_unique"/> is False.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.setdiff1d.html</remarks>
        public static NDArray setdiff1d(NDArray ar1, NDArray ar2, bool assume_unique = false)
        {
            if (assume_unique)
            {
                ar1 = np.ravel(ar1);
            }
            else
            {
                ar1 = np.unique(ar1);
                ar2 = np.unique(ar2);
            }

            // ar1[isin(ar1, ar2, assume_unique=True, invert=True)] — keep only the values absent from ar2.
            NDArray mask = np.isin(ar1, ar2, assume_unique: true, invert: true);
            return ar1[mask];
        }
    }
}
