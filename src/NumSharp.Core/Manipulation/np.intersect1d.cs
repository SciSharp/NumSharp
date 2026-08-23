namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the intersection of two arrays (the bare-return form of <c>np.intersect1d</c>).<br></br>
        ///     Return the sorted, unique values that are in both of the input arrays.
        /// </summary>
        /// <param name="ar1">Input array (flattened if not already 1-D).</param>
        /// <param name="ar2">Input array (flattened if not already 1-D).</param>
        /// <returns>Sorted 1-D array of common, unique elements.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html</remarks>
        public static NDArray intersect1d(NDArray ar1, NDArray ar2)
            => IntersectCore(ar1, ar2, false, false)[0];

        /// <summary>
        ///     Find the intersection of two arrays (bare-return, with the <c>assume_unique</c> speed hint).<br></br>
        ///     Return the sorted, unique values that are in both of the input arrays.
        /// </summary>
        /// <param name="ar1">Input array (flattened if not already 1-D).</param>
        /// <param name="ar2">Input array (flattened if not already 1-D).</param>
        /// <param name="assume_unique">If True, the input arrays are both assumed to be unique, which can speed up
        ///     the calculation. Default is False.</param>
        /// <returns>Sorted 1-D array of common, unique elements.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html</remarks>
        public static NDArray intersect1d(NDArray ar1, NDArray ar2, bool assume_unique)
            => IntersectCore(ar1, ar2, assume_unique, false)[0];

        /// <summary>
        ///     Find the intersection of two arrays, returning the indices of the common values.<br></br>
        ///     Mirrors NumPy's tuple return: <c>np.intersect1d(a, b, return_indices=True)</c> — both
        ///     <paramref name="assume_unique"/> and <paramref name="return_indices"/> are optional so the
        ///     keyword-only <c>return_indices</c> call binds here (the bare overloads above have no
        ///     <c>return_indices</c> parameter, so they cannot capture it — no ambiguity with this one).
        /// </summary>
        /// <param name="ar1">Input array (flattened if not already 1-D).</param>
        /// <param name="ar2">Input array (flattened if not already 1-D).</param>
        /// <param name="assume_unique">If True, the input arrays are both assumed to be unique.</param>
        /// <param name="return_indices">If True, also return the indices of the first occurrences of the common
        ///     values in <paramref name="ar1"/> and <paramref name="ar2"/>.</param>
        /// <returns><c>[values]</c> when <paramref name="return_indices"/> is False, otherwise
        ///     <c>[values, comm1, comm2]</c>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html</remarks>
        public static NDArray[] intersect1d(NDArray ar1, NDArray ar2, bool assume_unique = false, bool return_indices = false)
            => IntersectCore(ar1, ar2, assume_unique, return_indices);

        [NDScoped]
        private static NDArray[] IntersectCore(NDArray ar1, NDArray ar2, bool assume_unique, bool return_indices)
        {
            // Boundary scope: the unique/ravel/concatenate/argsort/take temps and the adjacency
            // mask are reclaimed at exit; the returned tuple's arrays are yielded.

            NDArray ind1 = null, ind2 = null;
            if (!assume_unique)
            {
                if (return_indices)
                {
                    NDArray[] u1 = np.unique(ar1, true); ar1 = u1[0]; ind1 = u1[1];
                    NDArray[] u2 = np.unique(ar2, true); ar2 = u2[0]; ind2 = u2[1];
                }
                else
                {
                    ar1 = np.unique(ar1);
                    ar2 = np.unique(ar2);
                }
            }
            else
            {
                ar1 = np.ravel(ar1);
                ar2 = np.ravel(ar2);
            }

            long ar1Size = ar1.size;
            NDArray cat = np.concatenate((ar1, ar2), 0);   // promotes to the common dtype

            NDArray aux, auxSortIndices = null;
            if (return_indices)
            {
                auxSortIndices = np.argsort(cat);          // stable (NumPy uses mergesort)
                aux = np.take(cat, auxSortIndices);
            }
            else
            {
                aux = np.sort(cat);
            }

            // mask = aux[1:] == aux[:-1]  — an adjacent equal pair means the value is in both arrays.
            NDArray auxLeft = aux[":-1"];
            NDArray mask = aux["1:"] == auxLeft;
            NDArray int1d = auxLeft[mask];

            if (!return_indices)
                return new[] { int1d };

            NDArray ar1_indices = auxSortIndices[":-1"][mask];
            NDArray ar2_indices = auxSortIndices["1:"][mask] - (NDArray)ar1Size;
            if (!assume_unique)
            {
                ar1_indices = np.take(ind1, ar1_indices);
                ar2_indices = np.take(ind2, ar2_indices);
            }

            return new[] { int1d, ar1_indices, ar2_indices };
        }
    }
}
