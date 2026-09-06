using System;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the indices to access the main diagonal of an array.
        /// </summary>
        /// <param name="n">The size, along each dimension, of the arrays for which the returned indices can be used.</param>
        /// <param name="ndim">The number of dimensions. Default 2.</param>
        /// <returns>
        ///     <paramref name="ndim"/> index arrays, each <c>arange(n)</c>, suitable for
        ///     indexing the main diagonal of an array of shape <c>(n,) * ndim</c>.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.diag_indices.html
        ///     <para>
        ///     NumPy's body is <c>idx = np.arange(n); return (idx,) * ndim</c> — the tuple holds
        ///     <b>the same array object</b> repeated, so writing through one entry is visible
        ///     through all of them (probed against 2.4.2). NumSharp reproduces that aliasing by
        ///     returning the identical <see cref="NDArray{T}"/> instance in every slot rather
        ///     than <c>ndim</c> independent copies.
        ///     </para>
        ///     <para>
        ///     A non-positive <paramref name="ndim"/> yields an empty array, mirroring Python's
        ///     <c>(idx,) * 0</c>; a negative <paramref name="n"/> yields empty index arrays.
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray<long>[] diag_indices(int n, int ndim = 2)
        {
            if (ndim <= 0)
                return Array.Empty<NDArray<long>>();

            var idx = np.arange(n < 0 ? 0 : n).MakeGeneric<long>();

            var result = new NDArray<long>[ndim];
            for (int d = 0; d < ndim; d++)
                result[d] = idx; // same instance, as NumPy's `(idx,) * ndim`

            return result;
        }

        /// <summary>
        ///     Return the indices to access the main diagonal of an n-dimensional array.
        /// </summary>
        /// <param name="arr">Array, at least 2-D, whose dimensions must all be of equal length.</param>
        /// <returns>Index arrays addressing <paramref name="arr"/>'s main diagonal.</returns>
        /// <exception cref="ArgumentException">
        ///     <c>input array must be at least 2-d</c> when <c>ndim &lt; 2</c>, or
        ///     <c>All dimensions of input must be of equal length</c> when the shape is not
        ///     hyper-cubic (both NumPy <c>ValueError</c>s, verbatim).
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.diag_indices_from.html</remarks>
        public static NDArray<long>[] diag_indices_from(NDArray arr)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));

            if (arr.ndim < 2)
                throw new ArgumentException("input array must be at least 2-d");

            // For more than d=2 the strided formula is only valid when every dimension
            // matches, so NumPy checks it up front regardless of ndim.
            RequireEqualDimensions(arr);

            return diag_indices(checked((int)arr.shape[0]), arr.ndim);
        }

        /// <summary>
        ///     NumPy's <c>if not np.all(diff(a.shape) == 0)</c> guard, shared by
        ///     <see cref="diag_indices_from"/> and <see cref="fill_diagonal"/>.
        /// </summary>
        internal static void RequireEqualDimensions(NDArray a)
        {
            var shape = a.shape;
            for (int d = 1; d < shape.Length; d++)
            {
                if (shape[d] != shape[0])
                    throw new ArgumentException("All dimensions of input must be of equal length");
            }
        }
    }
}
