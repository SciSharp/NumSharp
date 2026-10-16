using NumSharp.Backends;
using NumSharp.Backends.Sorting;

// =============================================================================
// NDArray.unique.Text.cs — np.unique over fixed-width text (S / U), NumPy 2.4.2's statements
// =============================================================================
//
// WHAT NUMPY DOES (numpy/lib/_arraysetops_impl.py, unique + _unique1d)
//   * axis=None, or a 1-D input: `ar = np.asanyarray(ar).flatten()`. Without a return flag `_unique_hash` hashes the text
//     (unique.cpp unique_string: FNV-1a over the whole item, memcmp equality) and `hash_unique.sort()` orders the set
//     (sorted=True); with a flag, `perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')`, `aux = ar[perm]`,
//     `mask[:1] = True; mask[1:] = aux[1:] != aux[:-1]`, then values `aux[mask]`, index `perm[mask]`, inverse
//     `inv_idx[perm] = cumsum(mask) - 1` reshaped to the input's shape, counts `diff(concatenate(nonzero(mask), [n]))`.
//   * an axis of an N-D input: the axis is moved first and the rows (every other axis flattened) are viewed as ONE
//     structured scalar each; the structured sort compares field by field with the text compare (lexicographic rows), the
//     mask compares whole rows, and the unique rows are viewed back, reshaped and moved to the axis (a view).
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   * Every output is order-free except through the ORDER of equal strings, and equal strings are byte-identical at one
//     width: the values are the sorted distinct strings whatever sort ran, the inverse and the counts do not depend on how
//     ties were ordered, and the stable order gives return_index NumPy's first occurrences (its mergesort). So one STABLE
//     order (TextSort's radix with its tie pass) serves every flag. The values-only call hashes like NumPy (TextDistinct:
//     the distinct strings, then a sort of only those) while at most max(256, n / 16) are distinct; past that it sorts the
//     flattened copy in place and keeps the first string of each run — the hash set's sorted values without hashing, and
//     the faster route for mostly distinct text.
//   * The rows of the axis path are ordered by np.lexsort over their columns (the last column first, so column 0 decides):
//     the stable lexicographic order of the structured sort, every column a stable text argsort.
//   * Everything after the order is NumPy's statement over a NumSharp kernel: the text gather (fancy index), the text
//     not_equal written into the mask's tail, the boolean selection, cumsum, the index scatter, nonzero, diff.
//   * sorted=False is not reproduced: NumPy returns its std::unordered_set's iteration order, which differs by C++ library
//     (MSVC and libstdc++) and is unspecified; NumSharp returns the sorted values (the integer / complex unique_values
//     divergence, documented on np.unique and np.unique_values).
// =============================================================================

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy 2.4.2's <c>np.unique</c> over fixed-width text (<c>S</c> / <c>U</c>): the flat path for
        ///     <c>axis=None</c> or a 1-D input, the row path for an axis of an N-D input (see the file header).
        /// </summary>
        /// <param name="a">The text array (only read).</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="axis">The axis, already range-checked by the caller (<see cref="ThrowIfTextUnique"/>), or null.</param>
        /// <returns>The outputs in NumPy's order, [values, index?, inverse?, counts?]: the values in <paramref name="a"/>'s
        ///     dtype, every index output int64. The flat path's inverse is still 1-D (the caller reshapes it to the input's
        ///     shape, as NumPy's <c>inv_idx.reshape(ar.shape)</c>); the row path's stays 1-D, as NumPy returns it.</returns>
        /// <remarks>Scoped: the flattened copy, the permutation, the gathered strings and the masks are reclaimed at exit;
        ///     only the returned outputs (and the buffers their views hold) survive.</remarks>
        [NDScoped]
        private static NDArray[] UniqueText(NDArray a, bool return_index, bool return_inverse, bool return_counts, int? axis)
        {
            // NumPy: `if axis is None or ar.ndim == 1` takes the flat path (a 1-D input's axis was normalized, and checked,
            // by the caller); any other axis takes the row path.
            if (axis is { } given && a.ndim != 1)
                return UniqueTextRows(a, given < 0 ? given + a.ndim : given, return_index, return_inverse, return_counts);

            // `ar = np.asanyarray(ar).flatten()`: a C-order copy that the values-only route may sort in place.
            NDArray ar = a.flatten();
            if (!return_index && !return_inverse && !return_counts)
            {
                // NumPy hashes the text and sorts the hash set. With few distinct strings the same hash pass collects them
                // and only they are sorted (TextDistinct); once too many are distinct it gives up, and the flattened copy
                // sorted, keeping the first string of every run of equal strings, holds the same values in the same order.
                NDArray hashed = TextDistinct.TryDistinct(ar);
                if (hashed is not null)
                {
                    TextSort.SortLines(hashed, 0);
                    return new[] { hashed };
                }
                TextSort.SortLines(ar, 0);
                using var runStarts = RunStarts(ar);
                return new[] { ar[runStarts] };
            }

            // One stable order serves every flag: return_index needs NumPy's mergesort (the first occurrence of each
            // string), and the inverse and the counts do not depend on how equal strings were ordered.
            NDArray perm = AxisSort.ArgSortText(ar, 0, stable: true);
            NDArray aux = ar[perm];
            // A numpy.char.chararray operand of the user's call: NumPy's aux is the chararray's gather, so its
            // `aux[1:] != aux[:-1]` is the chararray's STRIPPING comparison (compare_chararrays, rstrip=True): neighbours that
            // differ only in trailing ASCII whitespace / NULs are one run, kept as its first (smallest) string.
            NDArray mask = RunStarts(aux, NDCharArrayBoundary.StripUniqueRuns && a is NDCharArray);
            return UniqueOutputs(aux[mask], perm, mask, return_index, return_inverse, return_counts);
        }

        /// <summary>
        ///     NumPy's row path of <c>np.unique</c> over text (an axis of an N-D input): the axis moved first, every row (the
        ///     other axes flattened, C order) one structured scalar, the rows ordered lexicographically by their columns'
        ///     text, the distinct rows kept and moved back to the axis.
        /// </summary>
        /// <param name="a">The text array (two or more dimensions).</param>
        /// <param name="axis">The normalized axis.</param>
        /// <param name="return_index">Also return the first row of each distinct row.</param>
        /// <param name="return_inverse">Also return the reconstruction indices (1-D, as NumPy returns them with an axis).</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <returns>[values, index?, inverse?, counts?]; the values are NumPy's <c>reshape_uniq</c> result, a
        ///     <c>np.moveaxis</c> view of the distinct rows' C-contiguous buffer.</returns>
        /// <remarks>
        ///     <para>
        ///     NumPy's structured sort compares the fields (the columns) in order with the text compare, so the rows' order is
        ///     the stable lexicographic one np.lexsort gives with the LAST column passed first. Rows of zero width compare
        ///     equal (NumPy's field-less structured scalars), so every row joins the first one: the identity order.
        ///     </para>
        ///     <para>
        ///     <c>np.unique(np.zeros((3, 0), 'U2'), axis=0)</c> is one <c>(1, 0)</c> row with count 3, and an empty axis
        ///     gives no row at all.
        ///     </para>
        /// </remarks>
        private static NDArray[] UniqueTextRows(NDArray a, int axis, bool return_index, bool return_inverse, bool return_counts)
        {
            // `ar = np.moveaxis(ar, axis, 0)`, then `ar.reshape(n, prod(rest))` made C-contiguous: one row per index.
            NDArray moved = np.moveaxis(a, axis, 0);
            var movedDims = moved.Shape.dimensions;
            long n = movedDims[0];
            long width = 1;
            for (int d = 1; d < movedDims.Length; d++)
                width *= movedDims[d];
            NDArray rows = np.ascontiguousarray(moved.reshape(n, width));

            // The lexicographic order of the rows: np.lexsort's LAST key is its primary one, so the columns are passed from
            // the last to the first. Rows that cannot differ (no column) or cannot be reordered (at most one row) keep their
            // order.
            NDArray perm;
            if (width == 0 || n <= 1)
                perm = np.arange(n);
            else
            {
                // rows.T is (width, n): its rows are the columns; reversed, its last key is column 0.
                NDArray keys = rows.T["::-1"];
                perm = np.lexsort(keys, -1);
            }

            NDArray aux = np.take(rows, perm, 0);
            // mask[0] = True; mask[1:] = any(aux[1:] != aux[:-1], axis=1): a row starts a run when any column differs.
            var mask = new NDArray(NPTypeCode.Boolean, Shape.Vector(n), true);
            if (n > 0)
            {
                unsafe { *(bool*)mask.Address = true; }
                if (n > 1)
                {
                    NDArray differs = aux["1:"] != aux[":-1"];
                    np.any(differs, 1, @out: mask["1:"]);
                }
            }

            // NumPy's reshape_uniq: the distinct rows viewed back as the input's dtype, reshaped to (k, *rest) and moved
            // to the axis — views of the selected rows' fresh buffer.
            NDArray distinct = aux[mask];
            var shape = new long[movedDims.Length];
            shape[0] = distinct.Shape.dimensions[0];
            for (int d = 1; d < movedDims.Length; d++)
                shape[d] = movedDims[d];
            NDArray values = np.moveaxis(distinct.reshape(Shape.Adopt(shape)), 0, axis);
            return UniqueOutputs(values, perm, mask, return_index, return_inverse, return_counts);
        }

        /// <summary>
        ///     NumPy's <c>mask</c> of <c>_unique1d</c> over a sorted 1-D text array: True where an element starts a run of
        ///     equal strings (<c>mask[:1] = True; mask[1:] = aux[1:] != aux[:-1]</c>), the text <c>not_equal</c> written
        ///     straight into the mask's tail.
        /// </summary>
        /// <param name="sorted">The sorted 1-D text array.</param>
        /// <param name="strip">True to compare neighbours as a chararray does (trailing ASCII whitespace and NULs stripped
        ///     first: <c>compare_chararrays(..., '!=', rstrip=True)</c>).</param>
        /// <returns>A fresh bool array of <paramref name="sorted"/>'s length.</returns>
        private static NDArray RunStarts(NDArray sorted, bool strip = false)
        {
            long n = sorted.size;
            var mask = new NDArray(NPTypeCode.Boolean, Shape.Vector(n), false);
            if (n > 0)
            {
                unsafe { *(bool*)mask.Address = true; }
                if (n > 1)
                {
                    using var tail = mask["1:"];
                    using var later = sorted["1:"];
                    using var earlier = sorted[":-1"];
                    if (!strip)
                        np.not_equal(later, earlier, @out: tail);
                    else
                    {
                        // Both neighbours are one class, so the stripping comparison always answers (never NotImplemented).
                        using var differs = Backends.Strings.NDStringComparison.CompareChararrays("not_equal", later, earlier, rstrip: true);
                        np.copyto(tail, differs);
                    }
                }
            }
            return mask;
        }

        /// <summary>
        ///     The outputs NumPy's <c>_unique1d</c> builds once its order and mask exist: the values, then
        ///     <c>perm[mask]</c>, the inverse <c>inv_idx[perm] = cumsum(mask) - 1</c>, and the counts
        ///     <c>diff(concatenate((nonzero(mask)[0], [mask.size])))</c>.
        /// </summary>
        /// <param name="values">The distinct values (already selected).</param>
        /// <param name="perm">The order (int64) that sorted the input.</param>
        /// <param name="mask">The run starts in that order.</param>
        /// <param name="return_index">Whether the first occurrences are returned.</param>
        /// <param name="return_inverse">Whether the reconstruction indices are returned.</param>
        /// <param name="return_counts">Whether the counts are returned.</param>
        /// <returns>[values, index?, inverse?, counts?]; the inverse is 1-D.</returns>
        private static NDArray[] UniqueOutputs(NDArray values, NDArray perm, NDArray mask, bool return_index,
            bool return_inverse, bool return_counts)
        {
            int count = 1 + (return_index ? 1 : 0) + (return_inverse ? 1 : 0) + (return_counts ? 1 : 0);
            var outputs = new NDArray[count];
            int at = 0;
            outputs[at++] = values;
            if (return_index)
                outputs[at++] = perm[mask];
            if (return_inverse)
            {
                // imask = np.cumsum(mask) - 1; inv_idx = np.empty(n, intp); inv_idx[perm] = imask
                NDArray imask = np.cumsum(mask) - 1L;
                var inverse = new NDArray(NPTypeCode.Int64, Shape.Vector(mask.size), false);
                inverse[perm] = imask;
                outputs[at++] = inverse;
            }
            if (return_counts)
            {
                // idx = np.concatenate(np.nonzero(mask) + ([mask.size],)); np.diff(idx)
                NDArray starts = np.nonzero(mask)[0];
                NDArray idx = np.concatenate((starts, np.array(new[] { mask.size })), 0);
                outputs[at] = np.diff(idx);
            }
            return outputs;
        }
    }
}
