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
        /// <exception cref="DTypePromotionError">A 1-D <c>StringDType</c> <paramref name="x"/>: <c>promote_types(x.dtype,
        ///     int)</c> finds no common DType (<c>The DTypes &lt;class 'numpy.dtypes.StringDType'&gt; and &lt;class
        ///     'numpy.dtypes.Int64DType'&gt; do not have a common DType. ...</c>), whatever <paramref name="N"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vander.html
        /// <para>NumPy's statements: <c>v = empty((len(x), N), dtype=promote_types(x.dtype, int))</c>; <c>if N &gt; 0:
        /// tmp[:, 0] = 1</c>; <c>if N &gt; 1: tmp[:, 1:] = x[:, None]; multiply.accumulate(tmp[:, 1:], ...)</c>. A text
        /// <paramref name="x"/> promotes to a 21-character string (int64 formatted), so its matrix is the empty allocation
        /// for no columns or no rows with one column, the formatted <c>'1'</c> column otherwise for one column, and the
        /// multiply's refusal from two columns on — for an empty <paramref name="x"/> too (the dtype decides).</para></remarks>
        [NDScoped]
        public static NDArray vander(NDArray x, int? N = null, bool increasing = false)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            using var plain = NDMatrixBoundary.PlainFor(x);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return x.TensorEngine.Vander(x, N, increasing);
        }
    }
}
