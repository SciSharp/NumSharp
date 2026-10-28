using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the indices of the bins to which each value in input array <paramref name="x"/> belongs.
        ///
        ///     <list type="table">
        ///       <listheader><term>right</term><term>bins order</term><term>returned index i satisfies</term></listheader>
        ///       <item><term>false</term><term>increasing</term><term><c>bins[i-1] &lt;= x &lt; bins[i]</c></term></item>
        ///       <item><term>true</term><term>increasing</term><term><c>bins[i-1] &lt; x &lt;= bins[i]</c></term></item>
        ///       <item><term>false</term><term>decreasing</term><term><c>bins[i-1] &gt; x &gt;= bins[i]</c></term></item>
        ///       <item><term>true</term><term>decreasing</term><term><c>bins[i-1] &gt;= x &gt; bins[i]</c></term></item>
        ///     </list>
        ///
        ///     Values in <paramref name="x"/> beyond the bounds of <paramref name="bins"/> return 0 or
        ///     <c>len(bins)</c> as appropriate. Implemented in terms of <see cref="searchsorted(NDArray, NDArray, string, NDArray)"/>.
        /// </summary>
        /// <param name="x">Input array to be binned. May have any shape; the result has the same shape.</param>
        /// <param name="bins">1-D monotonic (increasing or decreasing) array of bin edges.</param>
        /// <param name="right">Whether the intervals include the right or the left bin edge. Default is left-closed (false).</param>
        /// <returns>Array of int64 indices, of the same shape as <paramref name="x"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> or <paramref name="bins"/> is null.</exception>
        /// <exception cref="IncorrectTypeException"><paramref name="x"/> is complex (<c>x may not be complex</c>, checked first).</exception>
        /// <exception cref="ValueError"><paramref name="bins"/> is not 1-D (<c>object too deep for desired array</c> /
        ///     <c>object of too small depth for desired array</c>), or, after the cast below, not monotonic (<c>bins must be
        ///     monotonically increasing or decreasing</c>).</exception>
        /// <exception cref="InvalidCastException"><paramref name="bins"/> does not cast 'safely' to float64 (text, StringDType
        ///     included, or complex; NumPy's TypeError text).</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> <paramref name="x"/>: NumPy's searchsorted compares the numeric bins
        ///     and the text as Python objects and refuses the first mixed pair (<c>'&lt;' not supported between instances of
        ///     'float' and 'str'</c>), or two keys Python cannot order.</exception>

        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
        ///     <para>
        ///     <b>Check order (NumPy's).</b> A complex x first; then <c>_monotonicity(bins)</c>, whose
        ///     <c>PyArray_FROMANY(bins, NPY_DOUBLE, 1, 1)</c> checks the bins' rank before their 'safe' cast to float64; then the
        ///     monotonicity; then <c>searchsorted</c>. So a text x is only reached once the (numeric) bins passed — and when x has
        ///     no elements, or the bins are empty, NumPy's answer needs no comparison: an empty int64 array, or all zeros.
        ///     </para>
        ///     <para>
        ///     <b>Text x.</b> <c>searchsorted(bins, x)</c> converts the numeric bins to x's string kind (their text, as NumPy
        ///     formats them) and compares text: <c>digitize(['3', '10'], [2, 5])</c> is <c>[1, 0]</c>, because <c>'10'</c>
        ///     sorts before <c>'2'</c>. The bins' text need not be in order (the monotonicity was checked on the numbers); the
        ///     search's exact comparisons, ported from NumPy, give NumPy's indices anyway.
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray digitize(NDArray x, NDArray bins, bool right = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(x, bins);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return np.EngineOf(x).Digitize(x, bins, right);
        }
    }
}
