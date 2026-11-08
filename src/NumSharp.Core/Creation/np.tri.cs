using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     An array with ones at and below the given diagonal and zeros elsewhere.
        /// </summary>
        /// <param name="N">Number of rows in the array.</param>
        /// <param name="M">Number of columns in the array. By default, <paramref name="M"/> is taken equal to <paramref name="N"/>.</param>
        /// <param name="k">
        ///     The sub-diagonal at and below which the array is filled. <c>k = 0</c> is the main
        ///     diagonal, while <c>k &lt; 0</c> is below it, and <c>k &gt; 0</c> is above.
        /// </param>
        /// <param name="dtype">Data type of the returned array. Defaults to <see cref="double"/>.</param>
        /// <returns>
        ///     Array with shape <c>(N, M)</c> whose lower triangle — filled with ones — is
        ///     at and below the <paramref name="k"/>-th diagonal. Always a freshly allocated,
        ///     C-contiguous, writeable array.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.tri.html
        ///     <para>
        ///     NumPy computes this as <c>greater_equal.outer(arange(N), arange(-k, M-k))</c>
        ///     followed by an <c>astype</c> — two full N×M passes plus a bool temporary.
        ///     Because row <c>i</c> is exactly <c>c_i = clamp(i + k + 1, 0, M)</c> ones followed
        ///     by zeros, NumSharp instead allocates the (already zeroed) result and blits a
        ///     prefix of a single pre-built ones row into each row. That is one
        ///     <see cref="Buffer.MemoryCopy(void*, void*, long, long)"/> per row over an untouched zero tail — no
        ///     comparison kernel, no mask temporary, and completely dtype-agnostic (the copy
        ///     only ever sees bytes).
        ///     </para>
        ///     <para>
        ///     Negative <paramref name="N"/>/<paramref name="M"/> clamp to zero, matching
        ///     NumPy (whose <c>arange</c> of a negative count yields an empty axis).
        ///     </para>
        ///     <para>The dtype is read as its storage lane (<see cref="DType.GetTypeCode"/>), which the array object's tri
        ///     fills; it has no operand, so no class rule applies.</para>
        /// </remarks>
        public static NDArray tri(int N, int? M = null, int k = 0, DType dtype = null)
            => NDArray.tri(N, M, k, dtype?.GetTypeCode() ?? NPTypeCode.Double);
    }
}
