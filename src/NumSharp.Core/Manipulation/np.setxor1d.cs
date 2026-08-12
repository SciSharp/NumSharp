namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the set exclusive-or of two arrays.<br></br>
        ///     Return the sorted, unique values that are in only one (not both) of the input arrays.
        /// </summary>
        /// <param name="ar1">Input array.</param>
        /// <param name="ar2">Input array.</param>
        /// <param name="assume_unique">If True, the input arrays are both assumed to be unique, which can speed up
        ///     the calculation. Default is False.</param>
        /// <returns>Sorted 1-D array of unique values that are in only one of the input arrays.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.setxor1d.html</remarks>
        public static NDArray setxor1d(NDArray ar1, NDArray ar2, bool assume_unique = false)
        {
            if (!assume_unique)
            {
                ar1 = np.unique(ar1);
                ar2 = np.unique(ar2);
            }

            // concatenate((ar1, ar2), axis=None) — flattens both.
            NDArray aux = np.concatenate((np.ravel(ar1), np.ravel(ar2)), 0);
            if (aux.size == 0)
                return aux;

            aux = np.sort(aux);

            // flag = concatenate(([True], aux[1:] != aux[:-1], [True]))
            NDArray trueOne = np.array(new[] { true });
            NDArray neq = aux["1:"] != aux[":-1"];
            NDArray flag = np.concatenate((trueOne, neq, trueOne), 0);

            // return aux[flag[1:] & flag[:-1]]  — values appearing exactly once survive
            NDArray sel = flag["1:"] & flag[":-1"];
            // aux came from np.sort; NumPy's sort canonicalises a surviving float32/float64 NaN (see np.setops.cs).
            return CanonicalizeSetOpNaN(aux[sel]);
        }
    }
}
