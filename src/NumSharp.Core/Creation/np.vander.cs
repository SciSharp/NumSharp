using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Generate a Vandermonde matrix.
        ///     The columns of the output matrix are powers of the input vector. When
        ///     <paramref name="increasing"/> is <c>false</c> (the default) the <c>i</c>-th output column is
        ///     the input vector raised element-wise to the power of <c>N - i - 1</c>.
        /// </summary>
        /// <param name="x">1-D input array.</param>
        /// <param name="N">Number of columns. If <c>null</c>, a square matrix is returned (<c>N = len(x)</c>).</param>
        /// <param name="increasing">
        ///     If <c>true</c> the powers increase left to right (<c>x^0 ... x^(N-1)</c>); if <c>false</c>
        ///     they are reversed (first column <c>x^(N-1)</c>).
        /// </param>
        /// <returns>The Vandermonde matrix, shape <c>(len(x), N)</c>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vander.html</remarks>
        [NDScoped]
        public static NDArray vander(NDArray x, int? N = null, bool increasing = false)
        {
            x = np.asarray(x);
            if (x.ndim != 1)
                throw new ValueError("x must be a one-dimensional array or sequence.");

            long m = x.size;
            int n = N ?? checked((int)m);

            // NumPy: dtype = promote_types(x.dtype, int) — `int` is intp (int64 on 64-bit).
            NPTypeCode dt = np.promote_types(x.typecode, NPTypeCode.Int64);

            // NumPy allocates `empty((len(x), N))`; a negative N surfaces the allocator's own error.
            if (n <= 0)
                return np.empty(new Shape(m, n), dt);

            NDArray onesCol = np.ones(new Shape(m, 1), dt);
            NDArray incr;
            if (n == 1)
            {
                incr = onesCol;
            }
            else
            {
                // Each of the trailing N-1 columns starts as x, then a cumulative product along the
                // columns turns column k into x^(k+1) — bit-identical to NumPy's `multiply.accumulate`.
                var xcol = np.expand_dims(x.astype(dt), -1);                 // (M, 1)
                var repeated = np.broadcast_to(xcol, new Shape(m, n - 1));   // (M, N-1), each column = x
                var powered = np.cumprod(repeated, 1, dt);                   // (M, N-1), column k = x^(k+1)
                incr = np.concatenate(new[] { onesCol, powered }, 1);        // (M, N), increasing powers
            }

            if (increasing)
                return incr;

            // Reverse the columns so the first is x^(N-1); copy to a C-contiguous buffer like NumPy's.
            return incr[":, ::-1"].copy();
        }
    }
}
