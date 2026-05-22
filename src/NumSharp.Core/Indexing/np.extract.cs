using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the elements of <paramref name="arr"/> that satisfy some
        ///     <paramref name="condition"/>. Equivalent to
        ///     <c>np.take(np.ravel(arr), np.flatnonzero(np.ravel(condition)))</c> —
        ///     i.e. <c>arr.ravel()[condition.ravel()]</c> when condition is boolean.
        /// </summary>
        /// <param name="condition">
        ///     Array whose nonzero / True entries indicate the elements of
        ///     <paramref name="arr"/> to extract. May be any dtype (treated as
        ///     truthy via NumPy's "nonzero" semantics). May be any shape — it is
        ///     ravel'd before alignment with <paramref name="arr"/>.
        /// </param>
        /// <param name="arr">Input array. May be any shape; it is ravel'd.</param>
        /// <returns>
        ///     Rank-1 <see cref="NDArray"/> of values from <paramref name="arr"/>
        ///     where the corresponding ravel'd <paramref name="condition"/> entry
        ///     is truthy. Dtype matches <paramref name="arr"/>.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.extract.html
        ///     <para>
        ///     Mirrors NumPy's two-step chain: <c>flatnonzero(condition.ravel())</c>
        ///     produces flat indices, then <c>take(arr.ravel(), idx)</c> gathers.
        ///     When the ravel'd condition is longer than <c>arr.size</c> and any
        ///     True falls beyond <c>arr.size</c>, the underlying take raises
        ///     <see cref="IndexOutOfRangeException"/>, mirroring NumPy's
        ///     <c>IndexError</c>.
        ///     </para>
        ///     <para>
        ///     Note that <see cref="place"/> is the inverse operation.
        ///     </para>
        /// </remarks>
        public static NDArray extract(NDArray condition, NDArray arr)
        {
            if (condition is null) throw new ArgumentNullException(nameof(condition));
            if (arr is null) throw new ArgumentNullException(nameof(arr));

            // flatnonzero already ravels its input and handles non-bool (treats as
            // truthy), 0-d (returns [0] or empty), and empty (returns empty 1-D).
            var indices = np.flatnonzero(condition);
            try
            {
                // ravel(arr) so take's axis=None flat path operates on the flat view.
                // For 0-d arr, np.take(scalar, ...) takes the 0-d → 1-element 1-D
                // route already; ravel just normalises so the kernel sees 1-D.
                var flatArr = np.ravel(arr);
                return np.take(flatArr, indices);
            }
            finally
            {
                indices.Dispose();
            }
        }
    }
}
