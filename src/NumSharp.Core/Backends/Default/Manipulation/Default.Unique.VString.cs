using NumSharp.Backends.Sorting;

// =============================================================================
// Default.Unique.VString.cs — np.unique over NEP 55 StringDType text (T), NumPy 2.4.2's statements
// =============================================================================
//
// WHAT NUMPY DOES (numpy/lib/_arraysetops_impl.py, unique + _unique1d; unique.cpp unique_vstring)
//   * The flat path (axis=None, or a 1-D input whose axis passed normalize_axis_index): `ar = ar.flatten()`. Without a
//     return flag `_unique_hash` hashes the cells (an NA cell's buffer is NULL whatever the NA object: one entry under
//     equal_nan, one per cell otherwise, never equal to a text) and, with sorted=True, `hash_unique.sort()` orders the
//     packed entries with the default kind (a None-like NA among two or more entries raises the sort's ValueError).
//   * With a flag: `perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')`, `aux = ar[perm]`,
//     `mask[1:] = aux[1:] != aux[:-1]` — equal_nan is ignored (its NaN branch tests `aux.dtype.kind in "cfmM"`, and
//     StringDType's kind is 'T'), and the StringDType `!=` of a NaN-like NA is False, so the NAs (sorted last) join the
//     run of the greatest text — then values `aux[mask]`, index `perm[mask]`, inverse `cumsum(mask) - 1` scattered by
//     perm, counts `diff(nonzero(mask) + [n])`.
//   * An axis of an N-D input takes the structured path, which refuses StringDType (Default.Unique.cs).
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   The values-only route is VStringDistinct (the distinct texts by hashing, the NA entries apart, NumPy's sorted order);
//   the flag route is NumPy's statements over the StringDType kernels: the argsort of the kind NumPy picks
//   (VStringSort), the gather, the `!=` (P2.T's comparison kernel with the NA rules), the selection, and the int64 outputs
//   NumPy's _unique1d builds (UniqueOutputs, shared with the fixed-width route).
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy 2.4.2's flat <c>np.unique</c> over <c>StringDType</c> text (the caller checked the axis: the structured
        ///     path refuses StringDType), see the file header.
        /// </summary>
        /// <param name="a">The <c>StringDType</c> array (only read).</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="equal_nan">NumPy's switch for the values-only route: the NA cells form one entry when true, one entry
        ///     each when false. The flag route ignores it, as NumPy's does for StringDType.</param>
        /// <param name="sorted">NumPy's <c>sorted</c> (values-only route): false is NumPy's hash order, which NumSharp does
        ///     not reproduce — the values come back sorted, NAs last, and a None-like NA is not compared.</param>
        /// <returns>The outputs in NumPy's order, [values, index?, inverse?, counts?]: the values in <paramref name="a"/>'s
        ///     descriptor, every index output int64; the inverse 1-D (the caller reshapes it to the input's shape).</returns>
        /// <exception cref="ValueError">A None-like NA where NumPy's sort compares it: two or more entries in the values-only
        ///     route, two or more elements in the flag route (<c>Cannot compare null that is not a nan-like value</c>).</exception>
        /// <remarks>Scoped: the flattened copy, the permutation, the gathered strings and the mask are reclaimed at exit; only
        ///     the returned outputs survive.</remarks>
        [NDScoped]
        private static NDArray[] UniqueVString(NDArray a, bool return_index, bool return_inverse, bool return_counts,
            bool equal_nan, bool sorted)
        {
            if (!return_index && !return_inverse && !return_counts)
            {
                // _unique_hash, then hash_unique.sort() (sorted=True): the distinct entries in NumPy's order, gathered.
                // NumPy flattens into a copy first; this route only READS the elements (the values are gathered into a new
                // array either way), so a 1-D operand (any stride) or a C-contiguous one is read in place — its ravel is a
                // view — and only another layout pays for the C-order copy.
                NDArray flat = a.ndim == 1 ? a : a.Shape.IsContiguous ? a.reshape(-1) : a.flatten();
                long[] picks = VStringDistinct.DistinctOrder(flat, equal_nan, sorted);
                NDArray values = picks.Length == 0
                    ? new NDArray(flat.dtype, Shape.Vector(0), true)
                    // np.take(flat, np.array(picks)): np.take's TakeFrom (mode 'raise') over a 1-D array of the picks.
                    : UnmanagedStorage.TakeFrom(flat, new NDArray(Unmanaged.ArraySlice.FromArray(picks, true), Shape.Vector(picks.Length)),
                        null, null, ClipModes.ClipModeRaise);
                return new[] { values };
            }

            // `ar = np.asanyarray(ar).flatten()`: a C-order copy (its own allocator).
            NDArray ar = a.flatten();

            // `perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')`: the stable order serves return_index
            // (first occurrences), the default kind otherwise (the group starts it picks are NumPy's cells).
            NDArray perm = AxisSort.ArgSortVString(ar, 0, stable: return_index);
            NDArray aux = ar[perm];
            // mask[:1] = True; mask[1:] = aux[1:] != aux[:-1] — the StringDType != with the descriptor's NA rule.
            NDArray mask = RunStarts(aux);
            return UniqueOutputs(aux[mask], perm, mask, return_index, return_inverse, return_counts);
        }
    }
}
