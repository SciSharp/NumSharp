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
        /// <exception cref="ValueError"><paramref name="x"/> is not one-dimensional, or <paramref name="N"/> is negative
        ///     (the allocation's <c>negative dimensions are not allowed</c>).</exception>
        /// <exception cref="TypeError">A text <paramref name="x"/> with two or more columns: the powers come from
        ///     <c>multiply.accumulate</c>, which has no string loop.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vander.html
        /// <para>NumPy's statements: <c>v = empty((len(x), N), dtype=promote_types(x.dtype, int))</c>; <c>if N &gt; 0:
        /// tmp[:, 0] = 1</c>; <c>if N &gt; 1: tmp[:, 1:] = x[:, None]; multiply.accumulate(tmp[:, 1:], ...)</c>. A text
        /// <paramref name="x"/> promotes to a 21-character string (int64 formatted), so its matrix is the empty allocation
        /// for no columns or no rows with one column, the formatted <c>'1'</c> column otherwise for one column, and the
        /// multiply's refusal from two columns on — for an empty <paramref name="x"/> too (the dtype decides).</para></remarks>
        [NDScoped]
        public static NDArray vander(NDArray x, int? N = null, bool increasing = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(x);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            x = np.asarray(x);
            if (x.ndim != 1)
                throw new ValueError("x must be a one-dimensional array or sequence.");

            long m = x.size;
            int n = N ?? checked((int)m);

            // NumPy: dtype = promote_types(x.dtype, int) — `int` is intp (int64 on 64-bit). Text absorbs int64 as a
            // 21-character string, and the powers come from multiply.accumulate, which has no string loop.
            if (Backends.Strings.NDStringUfuncs.IsString(x))
            {
                var text = DTypePromotion.PromoteTypes(x.dtype, np.int64);
                // `if N > 1: ... multiply.accumulate(...)` — the dtype refusal, whatever the rows.
                if (n > 1)
                    throw Backends.Strings.NDStringUfuncs.NoLoop("multiply", text, text);
                // `if N > 0: tmp[:, 0] = 1` writes the number one as its str() — the text '1' — into every row of the single
                // column (np.ones' string route).
                if (n == 1)
                    return np.ones(new Shape(m, n), text);
                // `empty((len(x), N), dtype)`: no column to write — the allocation is the matrix, and a negative N is its
                // own error. A U matrix is zero-filled, as NumPy's NEEDS_INIT allocation is.
                return np.empty(new Shape(m, n), text);
            }
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
