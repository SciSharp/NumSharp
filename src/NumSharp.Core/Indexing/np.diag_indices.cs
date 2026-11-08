using System;
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
        ///     <para>
        ///     The array OWNS its data (<c>flags.owndata</c> True), as NumPy's fresh <c>np.arange</c> does; it was an alias of
        ///     its own buffer (<c>MakeGeneric</c>), which reported OWNDATA False (the open-bugs rundown's bug 8 sweep).
        ///     </para>
        ///     <para>NumPy's statements are the array object's (<c>NDArray.diag_indices</c>); it has no operand, so no class
        ///     rule applies.</para>
        /// </remarks>
        public static NDArray<long>[] diag_indices(int n, int ndim = 2) => NDArray.diag_indices(n, ndim);

        /// <summary>
        ///     Return the indices to access the main diagonal of an n-dimensional array.
        /// </summary>
        /// <param name="arr">Array, at least 2-D, whose dimensions must all be of equal length.</param>
        /// <returns>Index arrays addressing <paramref name="arr"/>'s main diagonal.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> is null.</exception>
        /// <exception cref="ArgumentException">
        ///     <c>input array must be at least 2-d</c> when <c>ndim &lt; 2</c>, or
        ///     <c>All dimensions of input must be of equal length</c> when the shape is not
        ///     hyper-cubic (both NumPy <c>ValueError</c>s, verbatim).
        /// </exception>
        /// <exception cref="OverflowException">The dimension exceeds <see cref="int.MaxValue"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.diag_indices_from.html
        ///     <para>Only the shape is read, so no class rule applies; NumPy's statements are the array object's
        ///     (<c>NDArray.diag_indices_from</c>).</para></remarks>
        public static NDArray<long>[] diag_indices_from(NDArray arr)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));

            return NDArray.diag_indices_from(arr);
        }
    }
}
