using System;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the indices for the lower-triangle of an <c>(n, m)</c> array.
        /// </summary>
        /// <param name="n">The row dimension of the arrays for which the returned indices will be valid.</param>
        /// <param name="k">Diagonal offset. <c>k = 0</c> (default) is the main diagonal.</param>
        /// <param name="m">The column dimension. By default <paramref name="m"/> is taken equal to <paramref name="n"/>.</param>
        /// <returns>The row and column indices of the lower triangle, in C (row-major) order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.tril_indices.html</remarks>
        public static NDArray<long>[] tril_indices(int n, int k = 0, int? m = null)
            => TriangleIndices(n, k, m, lower: true);

        /// <summary>
        ///     Return the indices for the upper-triangle of an <c>(n, m)</c> array.
        /// </summary>
        /// <param name="n">The size of the arrays for which the returned indices will be valid.</param>
        /// <param name="k">Diagonal offset. <c>k = 0</c> (default) is the main diagonal.</param>
        /// <param name="m">The column dimension. By default <paramref name="m"/> is taken equal to <paramref name="n"/>.</param>
        /// <returns>The row and column indices of the upper triangle, in C (row-major) order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.triu_indices.html</remarks>
        public static NDArray<long>[] triu_indices(int n, int k = 0, int? m = null)
            => TriangleIndices(n, k, m, lower: false);

        /// <summary>
        ///     Return the indices for the lower-triangle of <paramref name="arr"/>.
        /// </summary>
        /// <param name="arr">The 2-D array whose shape supplies the dimensions.</param>
        /// <param name="k">Diagonal offset.</param>
        /// <exception cref="ArgumentException"><c>input array must be 2-d</c> (NumPy <c>ValueError</c>, verbatim).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.tril_indices_from.html</remarks>
        public static NDArray<long>[] tril_indices_from(NDArray arr, int k = 0)
        {
            RequireTwoDimensional(arr);
            return tril_indices(checked((int)arr.shape[arr.ndim - 2]), k, checked((int)arr.shape[arr.ndim - 1]));
        }

        /// <summary>
        ///     Return the indices for the upper-triangle of <paramref name="arr"/>.
        /// </summary>
        /// <param name="arr">The 2-D array whose shape supplies the dimensions.</param>
        /// <param name="k">Diagonal offset.</param>
        /// <exception cref="ArgumentException"><c>input array must be 2-d</c> (NumPy <c>ValueError</c>, verbatim).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.triu_indices_from.html</remarks>
        public static NDArray<long>[] triu_indices_from(NDArray arr, int k = 0)
        {
            RequireTwoDimensional(arr);
            return triu_indices(checked((int)arr.shape[arr.ndim - 2]), k, checked((int)arr.shape[arr.ndim - 1]));
        }

        private static void RequireTwoDimensional(NDArray arr)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (arr.ndim != 2)
                throw new ArgumentException("input array must be 2-d");
        }

        /// <summary>
        ///     Shared generator behind <see cref="tril_indices"/> / <see cref="triu_indices"/>.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumPy materialises an <c>n×m</c> boolean mask via <see cref="tri"/>, broadcasts
        ///     both sparse coordinate grids to that shape, and runs two boolean gathers:
        ///     <c>tuple(broadcast_to(inds, tri_.shape)[tri_] for inds in indices(tri_.shape, sparse=True))</c>.
        ///     That is <c>O(n·m)</c> work plus three temporaries even when the answer is short
        ///     (a <c>k = -n</c> query returns nothing after scanning the whole mask).
        ///     </para>
        ///     <para>
        ///     The triangle is <em>separable</em>, so none of that is necessary: the kept columns
        ///     of row <c>i</c> are the single contiguous run <c>[0, c_i)</c> (lower) or
        ///     <c>[lo_i, m)</c> (upper). NumSharp sums those run lengths to size the output
        ///     exactly (the house pre-size-then-fill pattern), then emits each run with bulk
        ///     operations only — a <see cref="Span{T}.Fill"/> of the constant row index and a
        ///     <see cref="Buffer.MemoryCopy"/> from a shared <c>0..m-1</c> iota for the columns.
        ///     Cost is <c>O(output)</c>, never <c>O(n·m)</c>.
        ///     </para>
        /// </remarks>
        private static unsafe NDArray<long>[] TriangleIndices(int n, int k, int? m, bool lower)
        {
            // NumPy reaches these extents through tri()/arange(), where a negative count is an
            // empty axis rather than an error.
            long rows = n < 0 ? 0 : n;
            long cols = (m ?? n) < 0 ? 0 : (m ?? n);

            // Pass 1 — exact size.
            long total = 0;
            for (long i = 0; i < rows; i++)
                total += RunLength(i, k, cols, lower);

            var rowsResult = new NDArray(NPTypeCode.Int64, new Shape(total), false);
            var colsResult = new NDArray(NPTypeCode.Int64, new Shape(total), false);
            var result = new[] {rowsResult.MakeGeneric<long>(), colsResult.MakeGeneric<long>()};

            if (total == 0)
                return result;

            // Shared 0..cols-1 column ladder; every row's column run is a slice of it.
            var iota = np.arange(cols);
            long* iotaPtr = (long*)iota.Storage.Address + iota.Shape.Offset;
            long* rp = (long*)rowsResult.Storage.Address;
            long* cp = (long*)colsResult.Storage.Address;

            // Pass 2 — emit, bulk-wise.
            long w = 0;
            for (long i = 0; i < rows; i++)
            {
                long lo = lower ? 0 : Clamp(i + k, 0, cols);
                long hi = lower ? Clamp(i + k + 1, 0, cols) : cols;
                long len = hi - lo;
                if (len <= 0) continue;

                new Span<long>(rp + w, (int)len).Fill(i);

                long bytes = len * sizeof(long);
                Buffer.MemoryCopy(
                    source: iotaPtr + lo,
                    destination: cp + w,
                    destinationSizeInBytes: bytes,
                    sourceBytesToCopy: bytes);

                w += len;
            }

            return result;
        }

        /// <summary>Number of retained columns in row <paramref name="i"/>.</summary>
        private static long RunLength(long i, int k, long cols, bool lower)
        {
            if (lower)
                return Clamp(i + k + 1, 0, cols);

            return cols - Clamp(i + k, 0, cols);
        }

        private static long Clamp(long value, long min, long max)
            => value < min ? min : (value > max ? max : value);

        /// <summary>
        ///     Return the indices to access <c>(n, n)</c> arrays, given a masking function.
        /// </summary>
        /// <param name="n">The returned indices will be valid to access arrays of shape <c>(n, n)</c>.</param>
        /// <param name="mask_func">
        ///     A function whose call signature is <c>(arr, k)</c> and which returns
        ///     <paramref name="n"/>-by-<paramref name="n"/> masked arrays — e.g.
        ///     <see cref="triu"/> or <see cref="tril"/>.
        /// </param>
        /// <param name="k">An optional argument passed through to <paramref name="mask_func"/>.</param>
        /// <returns>The <c>N</c> index arrays of the locations where the mask function is non-zero.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.mask_indices.html
        ///     <para>
        ///     The returned arity follows <paramref name="mask_func"/>'s output rank, not
        ///     <c>n</c>: passing <see cref="diag"/> — which reduces a 2-D input to its 1-D
        ///     diagonal — yields a <b>single</b> index array, matching NumPy (probed).
        ///     </para>
        /// </remarks>
        public static NDArray<long>[] mask_indices(int n, Func<NDArray, int, NDArray> mask_func, int k = 0)
        {
            if (mask_func is null) throw new ArgumentNullException(nameof(mask_func));

            var m = np.ones(Shape.Matrix(n < 0 ? 0 : n, n < 0 ? 0 : n), NPTypeCode.Int64);
            var a = mask_func(m, k);

            return np.nonzero(np.not_equal(a, np.zeros(new Shape(1), a.typecode)));
        }
    }
}
