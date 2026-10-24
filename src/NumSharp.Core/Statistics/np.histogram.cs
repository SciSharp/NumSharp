using System;
using NumSharp.Backends;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The two arrays NumPy's <c>histogram</c> returns: the per-bin values and the bin edges.
        ///     Converts implicitly to <see cref="NDArray"/> (the histogram values — the primary result)
        ///     and to <see cref="NDArray"/><c>[]</c> (<c>[hist, bin_edges]</c>), and deconstructs as
        ///     <c>var (hist, edges) = np.histogram(a);</c> so ported NumPy call shapes read verbatim.
        /// </summary>
        public readonly struct HistogramResult : INDArrayCarrier
        {
            /// <summary>The histogram values: sample counts (int64) or, with weights, summed weights.</summary>
            public readonly NDArray hist;
            /// <summary>The bin edges (length = <c>hist.size + 1</c>).</summary>
            public readonly NDArray bin_edges;

            /// <summary>Pairs the two results; <see cref="Statistics.HistogramEngine"/> builds them.</summary>
            /// <param name="hist">The histogram values (the result owns them).</param>
            /// <param name="binEdges">The bin edges (an explicit edge array the caller passed comes back as is).</param>
            internal HistogramResult(NDArray hist, NDArray binEdges) { this.hist = hist; bin_edges = binEdges; }

            /// <summary>Deconstructs into <c>(hist, bin_edges)</c>.</summary>
            /// <param name="hist">Receives the histogram values.</param>
            /// <param name="binEdges">Receives the bin edges.</param>
            public void Deconstruct(out NDArray hist, out NDArray binEdges) { hist = this.hist; binEdges = bin_edges; }

            /// <summary>Yields the primary result — the histogram values.</summary>
            /// <param name="r">The result to unwrap.</param>
            public static implicit operator NDArray(HistogramResult r) => r.hist;

            /// <summary>Yields both arrays in NumPy field order <c>[hist, bin_edges]</c>.</summary>
            /// <param name="r">The result to unwrap.</param>
            public static implicit operator NDArray[](HistogramResult r) => new[] { r.hist, r.bin_edges };

            void INDArrayCarrier.YieldTo(NDScope scope) { scope.Returns(hist); scope.Returns(bin_edges); }
        }

        // ---------------------------------------------------------------------
        //  histogram_bin_edges
        // ---------------------------------------------------------------------

        /// <summary>
        ///     Compute only the bin edges <see cref="histogram(NDArray,int,ValueTuple{double,double}?,bool,NDArray)"/>
        ///     would use, so one set of edges can be shared across several histograms. With an integer
        ///     <paramref name="bins"/> the edges are equal-width across the range.
        /// </summary>
        /// <param name="a">Input data; the histogram is computed over the flattened array.</param>
        /// <param name="bins">Number of equal-width bins (default 10).</param>
        /// <param name="range">Optional <c>(lower, upper)</c> outer edges; defaults to <c>(a.min(), a.max())</c>.</param>
        /// <param name="weights">Unused for edge computation (accepted for signature parity, only shape-checked).</param>
        /// <returns>The bin edges (length <paramref name="bins"/> + 1), dtype float (or the input's float width).</returns>
        /// <exception cref="ValueError"><paramref name="bins"/> &lt; 1, or the range is invalid/non-finite.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html <para>Scoped (<c>[NDScoped]</c>): the raveled input, any weights ravel and every estimator/linspace intermediate are reclaimed; only the edge array leaves (an explicit edge array the caller passed through unchanged stays the caller's).</para></remarks>
        [NDScoped]
        public static NDArray histogram_bin_edges(NDArray a, int bins = 10, (double, double)? range = null, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return HistogramEngine.HistogramBinEdges(a, bins, range, weights);
        }

        /// <summary>
        ///     Compute the bin edges chosen by an automatic estimator (<c>"auto"</c>, <c>"fd"</c>,
        ///     <c>"doane"</c>, <c>"scott"</c>, <c>"stone"</c>, <c>"rice"</c>, <c>"sturges"</c>, <c>"sqrt"</c>).
        ///     The bin width is optimal for the data within the range while the bin count fills the whole range.
        /// </summary>
        /// <param name="a">Input data; the histogram is computed over the flattened array.</param>
        /// <param name="bins">The estimator name.</param>
        /// <param name="range">Optional <c>(lower, upper)</c> outer edges; also constrains the data the estimator sees.</param>
        /// <param name="weights">Weighted data is not supported for automatic estimation.</param>
        /// <returns>The estimator-chosen bin edges.</returns>
        /// <exception cref="ValueError"><paramref name="bins"/> is not a known estimator, or the range is invalid.</exception>
        /// <exception cref="TypeError"><paramref name="weights"/> is supplied (unsupported for estimators).</exception>
        /// <remarks>Scoped (<c>[NDScoped]</c>): the raveled input, any weights ravel and every estimator/linspace intermediate are reclaimed; only the edge array leaves (an explicit edge array the caller passed through unchanged stays the caller's).</remarks>
        [NDScoped]
        public static NDArray histogram_bin_edges(NDArray a, string bins, (double, double)? range = null, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return HistogramEngine.HistogramBinEdges(a, bins, range, weights);
        }

        /// <summary>
        ///     Pass a pre-computed, monotonically-increasing edge array through unmodified (after a
        ///     monotonicity check) — the identity form that lets histogram/histogram_bin_edges accept
        ///     explicit non-uniform edges.
        /// </summary>
        /// <param name="a">Input data (only its dtype/emptiness matters here).</param>
        /// <param name="bins">The explicit bin edges (must increase monotonically; dtype preserved).</param>
        /// <param name="range">Ignored when explicit edges are given (accepted for signature parity).</param>
        /// <param name="weights">Ignored for edge computation.</param>
        /// <returns>The edges array (dtype preserved).</returns>
        /// <exception cref="ValueError">The edges are not 1-D or not monotonically increasing.</exception>
        /// <remarks>Scoped (<c>[NDScoped]</c>): the raveled input, any weights ravel and every estimator/linspace intermediate are reclaimed; only the edge array leaves (an explicit edge array the caller passed through unchanged stays the caller's).</remarks>
        [NDScoped]
        public static NDArray histogram_bin_edges(NDArray a, NDArray bins, (double, double)? range = null, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, bins, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return HistogramEngine.HistogramBinEdges(a, bins, range, weights);
        }

        /// <summary>
        ///     Convenience overload accepting explicit edges as a C# <c>double[]</c> — equivalent to
        ///     <see cref="histogram_bin_edges(NDArray,NDArray,ValueTuple{double,double}?,NDArray)"/> with
        ///     <c>np.array(bins)</c>.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="bins">The explicit bin edges.</param>
        /// <param name="range">Ignored when explicit edges are given.</param>
        /// <param name="weights">Ignored for edge computation.</param>
        /// <returns>The edges array (float64).</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>): the raveled input, any weights ravel and every estimator/linspace intermediate are reclaimed; only the edge array leaves (an explicit edge array the caller passed through unchanged stays the caller's).</remarks>
        [NDScoped]
        public static NDArray histogram_bin_edges(NDArray a, double[] bins, (double, double)? range = null, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return HistogramEngine.HistogramBinEdges(a, (NDArray)np.array(bins), range, weights);
        }

        // ---------------------------------------------------------------------
        //  histogram
        // ---------------------------------------------------------------------

        /// <summary>
        ///     Compute the histogram of a dataset over <paramref name="bins"/> equal-width bins.
        ///     All bins are half-open <c>[l, r)</c> except the last, which is closed <c>[l, r]</c>;
        ///     samples outside the range are ignored.
        /// </summary>
        /// <param name="a">Input data; the histogram is computed over the flattened array.</param>
        /// <param name="bins">Number of equal-width bins (default 10).</param>
        /// <param name="range">Optional <c>(lower, upper)</c> outer edges; defaults to <c>(a.min(), a.max())</c>.</param>
        /// <param name="density">
        ///     When true, return the probability density (<c>count / total / bin_width</c>) whose integral over
        ///     the range is 1, instead of raw counts.
        /// </param>
        /// <param name="weights">
        ///     Optional per-sample weights of the same shape as <paramref name="a"/>; each sample then contributes
        ///     its weight instead of 1, and <c>hist.dtype</c> follows <paramref name="weights"/>.
        /// </param>
        /// <returns>A <see cref="HistogramResult"/> carrying the histogram values and the bin edges.</returns>
        /// <exception cref="ValueError"><paramref name="bins"/> &lt; 1, or the range is invalid/non-finite.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.histogram.html <para>Scoped (<c>[NDScoped]</c>; <see cref="HistogramResult"/> is an <see cref="INDArrayCarrier"/>): the raveled input and weights, the per-sample bin indices, the bincount/cumulative scratch and the density quotients are reclaimed; only the histogram and its edges leave.</para></remarks>
        [NDScoped]
        public static HistogramResult histogram(NDArray a, int bins = 10, (double, double)? range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return HistogramEngine.Histogram(a, bins, range, density, weights);
        }

        /// <summary>
        ///     Compute the histogram using an automatic bin-count estimator (see
        ///     <see cref="histogram_bin_edges(NDArray,string,ValueTuple{double,double}?,NDArray)"/> for the names).
        /// </summary>
        /// <param name="a">Input data; the histogram is computed over the flattened array.</param>
        /// <param name="bins">The estimator name.</param>
        /// <param name="range">Optional <c>(lower, upper)</c> outer edges; also constrains the data the estimator sees.</param>
        /// <param name="density">When true, return the probability density instead of raw counts.</param>
        /// <param name="weights">Weighted data is not supported for automatic estimation.</param>
        /// <returns>A <see cref="HistogramResult"/> carrying the histogram values and the bin edges.</returns>
        /// <exception cref="TypeError"><paramref name="weights"/> is supplied together with an estimator name.</exception>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramResult"/> is an <see cref="INDArrayCarrier"/>): the raveled input and weights, the per-sample bin indices, the bincount/cumulative scratch and the density quotients are reclaimed; only the histogram and its edges leave.</remarks>
        [NDScoped]
        public static HistogramResult histogram(NDArray a, string bins, (double, double)? range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return HistogramEngine.Histogram(a, bins, range, density, weights);
        }

        /// <summary>
        ///     Compute the histogram over explicit, possibly non-uniform, monotonically-increasing bin edges.
        /// </summary>
        /// <param name="a">Input data; the histogram is computed over the flattened array.</param>
        /// <param name="bins">The explicit bin edges (must increase monotonically).</param>
        /// <param name="range">Ignored when explicit edges are given (accepted for signature parity).</param>
        /// <param name="density">When true, return the probability density instead of raw counts.</param>
        /// <param name="weights">Optional per-sample weights (same shape as <paramref name="a"/>).</param>
        /// <returns>A <see cref="HistogramResult"/> carrying the histogram values and the bin edges.</returns>
        /// <exception cref="ValueError">The edges are not 1-D or not monotonically increasing.</exception>
        /// <exception cref="ArgumentException">Bool <paramref name="weights"/> over data: NumPy's UFuncTypeError <c>Cannot
        ///     cast ufunc 'add' output from dtype('int64') to dtype('bool') with casting rule 'same_kind'</c> (NumPy adds the
        ///     weights' int64 cumulative sum into a bool total in place).</exception>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramResult"/> is an <see cref="INDArrayCarrier"/>): the raveled input and weights, the per-sample bin indices, the bincount/cumulative scratch and the density quotients are reclaimed; only the histogram and its edges leave.</remarks>
        [NDScoped]
        public static HistogramResult histogram(NDArray a, NDArray bins, (double, double)? range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, bins, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return HistogramEngine.Histogram(a, bins, range, density, weights);
        }

        /// <summary>
        ///     Convenience overload accepting explicit edges as a C# <c>double[]</c>.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="bins">The explicit bin edges.</param>
        /// <param name="range">Ignored when explicit edges are given.</param>
        /// <param name="density">When true, return the probability density instead of raw counts.</param>
        /// <param name="weights">Optional per-sample weights.</param>
        /// <returns>A <see cref="HistogramResult"/> carrying the histogram values and the bin edges.</returns>
        /// <exception cref="ValueError">The edges are not monotonically increasing.</exception>
        /// <exception cref="ArgumentException">Bool <paramref name="weights"/> over data: NumPy's UFuncTypeError <c>Cannot
        ///     cast ufunc 'add' output from dtype('int64') to dtype('bool') with casting rule 'same_kind'</c> (NumPy adds the
        ///     weights' int64 cumulative sum into a bool total in place).</exception>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramResult"/> is an <see cref="INDArrayCarrier"/>): the raveled input and weights, the per-sample bin indices, the bincount/cumulative scratch and the density quotients are reclaimed; only the histogram and its edges leave.</remarks>
        [NDScoped]
        public static HistogramResult histogram(NDArray a, double[] bins, (double, double)? range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return HistogramEngine.Histogram(a, (NDArray)np.array(bins), range, density, weights);
        }
    }
}
