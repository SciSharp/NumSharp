using System;
using NumSharp.Backends;

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
        ///     <see cref="Buffer.MemoryCopy"/> per row over an untouched zero tail — no
        ///     comparison kernel, no mask temporary, and completely dtype-agnostic (the copy
        ///     only ever sees bytes).
        ///     </para>
        ///     <para>
        ///     Negative <paramref name="N"/>/<paramref name="M"/> clamp to zero, matching
        ///     NumPy (whose <c>arange</c> of a negative count yields an empty axis).
        ///     </para>
        /// </remarks>
        public static NDArray tri(int N, int? M = null, int k = 0, Type dtype = null)
            => tri(N, M, k, (dtype ?? typeof(double)).GetTypeCode());

        /// <summary>
        ///     An array with ones at and below the given diagonal and zeros elsewhere.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.tri.html</remarks>
        public static unsafe NDArray tri(int N, int? M, int k, NPTypeCode dtype)
        {
            // NumPy builds the row/column axes with arange, so a negative extent simply
            // produces an empty axis rather than an error.
            int rows = N < 0 ? 0 : N;
            int cols = (M ?? N) < 0 ? 0 : (M ?? N);

            if (rows == 0 || cols == 0)
                return np.zeros(Shape.Matrix(rows, cols), dtype);

            int elemSize = dtype.SizeOf();
            bool writeOnce = PrefersWriteOnce((long)rows * cols * elemSize);

            // Below the threshold the buffer comes back dirty, so pre-zeroing it would be a
            // wasted pass — allocate uninitialised and write every byte exactly once (ones
            // prefix, then zero suffix). Above it np.zeros is free, so only the ones prefix
            // is written. See PrefersWriteOnce.
            var res = writeOnce
                ? new NDArray(dtype, Shape.Matrix(rows, cols), false)
                : np.zeros(Shape.Matrix(rows, cols), dtype);

            // One row of ones, blitted as a prefix into every output row.
            var onesRow = np.ones(new Shape(cols), dtype);
            byte* src = onesRow.Storage.Address + onesRow.Shape.Offset * elemSize;
            byte* dst = res.Storage.Address + res.Shape.Offset * elemSize;
            long rowBytes = (long)cols * elemSize;

            for (long i = 0; i < rows; i++)
            {
                // Columns j <= i + k are kept => the first clamp(i + k + 1, 0, cols) entries.
                long keep = i + k + 1;
                if (keep < 0) keep = 0;
                else if (keep > cols) keep = cols;

                byte* row = dst + i * rowBytes;
                long onesBytes = keep * elemSize;

                if (onesBytes > 0)
                    Buffer.MemoryCopy(src, row, onesBytes, onesBytes);

                if (writeOnce)
                    ZeroBytes(row + onesBytes, rowBytes - onesBytes);
            }

            return res;
        }

        /// <summary>
        ///     Byte size at and below which a freshly allocated buffer comes back <b>dirty</b>,
        ///     making an <c>np.zeros</c> pre-fill a genuine extra write over the whole array.
        /// </summary>
        /// <remarks>
        ///     Measured, not guessed. Timing the two strategies for <c>tril</c> across
        ///     0.8 MB → 200 MB shows a sharp cliff at 64 MiB: at 64 MB write-once takes 6.0 ms
        ///     vs zeros+blit's 11.0 ms, while at 72 MB it inverts to 18.9 ms vs 13.5 ms. The
        ///     allocator hands out fresh (OS-zeroed) pages above that size, so there the
        ///     zero-fill costs nothing and only the retained half need be touched; below it the
        ///     memory is recycled and pre-zeroing doubles the traffic.
        /// </remarks>
        internal const long WriteOnceMaxBytes = 64L * 1024 * 1024;

        /// <summary>
        ///     Whether to allocate uninitialised and write every byte once (true), or to lean on
        ///     <c>np.zeros</c> and blit only the retained span (false).
        /// </summary>
        internal static bool PrefersWriteOnce(long totalBytes) => totalBytes <= WriteOnceMaxBytes;

        /// <summary>
        ///     Zero <paramref name="count"/> bytes at <paramref name="p"/>. Shared by the
        ///     write-once triangular fills (<see cref="tri(int,int?,int,NPTypeCode)"/>,
        ///     <see cref="tril"/>/<see cref="triu"/>), which allocate uninitialised and must
        ///     clear the dropped span themselves. Chunked because <see cref="Span{T}"/> is
        ///     int-length while a single row can in principle exceed 2 GB.
        /// </summary>
        internal static unsafe void ZeroBytes(byte* p, long count)
        {
            while (count > 0)
            {
                int chunk = count > int.MaxValue ? int.MaxValue : (int)count;
                new Span<byte>(p, chunk).Clear();
                p += chunk;
                count -= chunk;
            }
        }
    }
}
