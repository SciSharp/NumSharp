using System;
using NumSharp.Backends;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The multidimensional histogram plus its per-dimension bin edges — the two values NumPy's
        ///     <c>histogramdd</c> returns. Converts implicitly to <see cref="NDArray"/> (the histogram <c>H</c>,
        ///     always float64) and deconstructs as <c>var (H, edges) = np.histogramdd(sample);</c>.
        /// </summary>
        public readonly struct HistogramddResult : INDArrayCarrier
        {
            /// <summary>The D-dimensional histogram, shape <c>(nbin_0, …, nbin_{D-1})</c>, dtype float64.</summary>
            public readonly NDArray H;
            /// <summary>The D bin-edge arrays, one per dimension.</summary>
            public readonly NDArray[] edges;

            /// <summary>Pairs the two results; the engine member <see cref="TensorEngine.Histogramdd"/> builds them.</summary>
            /// <param name="h">The histogram (the result owns it).</param>
            /// <param name="edges">The per-dimension edges (an explicit 1-D edge array the caller passed comes back as is).</param>
            internal HistogramddResult(NDArray h, NDArray[] edges) { H = h; this.edges = edges; }

            /// <summary>Deconstructs into <c>(H, edges)</c>.</summary>
            /// <param name="h">Receives the histogram.</param>
            /// <param name="edges">Receives the per-dimension edge arrays.</param>
            public void Deconstruct(out NDArray h, out NDArray[] edges) { h = H; edges = this.edges; }

            /// <summary>Yields the primary result — the histogram <c>H</c>.</summary>
            /// <param name="r">The result to unwrap.</param>
            public static implicit operator NDArray(HistogramddResult r) => r.H;

            void INDArrayCarrier.YieldTo(NDScope scope) { scope.Returns(H); scope.Returns(edges); }
        }

        // ---------------------------------------------------------------------
        //  histogramdd — sample given as an (N, D) array
        // ---------------------------------------------------------------------

        /// <summary>
        ///     Compute the D-dimensional histogram of <paramref name="sample"/> with the same integer bin count
        ///     along every dimension. Each row of <paramref name="sample"/> is one D-dimensional coordinate.
        /// </summary>
        /// <param name="sample">An <c>(N, D)</c> array of N points in D-dimensional space.</param>
        /// <param name="bins">Number of equal-width bins along every dimension (default 10).</param>
        /// <param name="range">Optional per-dimension <c>(lower, upper)</c> outer edges; a null entry auto-detects that dim.</param>
        /// <param name="density">When true, return the probability density (<c>count / total / bin_volume</c>).</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="sample"/> is null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html <para>Scoped (<c>[NDScoped]</c>; <see cref="HistogramddResult"/> is an <see cref="INDArrayCarrier"/>): the sample column views, the promoted columns, the per-dimension bin indices, the raveled multi-index and the density quotients are reclaimed; only <c>H</c> and the edge arrays leave.</para></remarks>
        [NDScoped]
        public static HistogramddResult histogramdd(NDArray sample, int bins = 10, (double, double)?[] range = null, bool density = false, NDArray weights = null)
        {
            if (sample is null) throw new ArgumentNullException(nameof(sample));
            using var plain = NDMatrixBoundary.PlainFor(sample, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return sample.TensorEngine.Histogramdd(DefaultEngine.SampleColumns(sample), bins, range, density, weights);
        }

        /// <summary>Compute the D-dimensional histogram with a per-dimension bin count.</summary>
        /// <param name="sample">An <c>(N, D)</c> array of N points in D-dimensional space.</param>
        /// <param name="bins">Per-dimension bin counts (length must equal D).</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="sample"/> is null.</exception>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramddResult"/> is an <see cref="INDArrayCarrier"/>): the sample column views, the promoted columns, the per-dimension bin indices, the raveled multi-index and the density quotients are reclaimed; only <c>H</c> and the edge arrays leave.</remarks>
        [NDScoped]
        public static HistogramddResult histogramdd(NDArray sample, int[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
        {
            if (sample is null) throw new ArgumentNullException(nameof(sample));
            using var plain = NDMatrixBoundary.PlainFor(sample, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return sample.TensorEngine.Histogramdd(DefaultEngine.SampleColumns(sample), bins, range, density, weights);
        }

        /// <summary>Compute the D-dimensional histogram with explicit per-dimension bin edges.</summary>
        /// <param name="sample">An <c>(N, D)</c> array of N points in D-dimensional space.</param>
        /// <param name="bins">Per-dimension edge arrays (length must equal D; a 0-d entry is read as a bin count).</param>
        /// <param name="range">Optional per-dimension outer edges (used only where <paramref name="bins"/> gives a count).</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="sample"/> is null.</exception>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramddResult"/> is an <see cref="INDArrayCarrier"/>): the sample column views, the promoted columns, the per-dimension bin indices, the raveled multi-index and the density quotients are reclaimed; only <c>H</c> and the edge arrays leave.</remarks>
        [NDScoped]
        public static HistogramddResult histogramdd(NDArray sample, NDArray[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
        {
            if (sample is null) throw new ArgumentNullException(nameof(sample));
            using var plain = NDMatrixBoundary.PlainFor(sample, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            using var plain2 = NDMatrixBoundary.PlainFor(bins);
            return sample.TensorEngine.Histogramdd(DefaultEngine.SampleColumns(sample), bins, range, density, weights);
        }

        /// <summary>Compute the D-dimensional histogram with a mixed per-dimension bin spec (each entry an int count or an edge array).</summary>
        /// <param name="sample">An <c>(N, D)</c> array of N points in D-dimensional space.</param>
        /// <param name="bins">Per-dimension spec: each entry an <see cref="int"/> count, an <see cref="NDArray"/> / <c>double[]</c> edge array.</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="sample"/> is null.</exception>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramddResult"/> is an <see cref="INDArrayCarrier"/>): the sample column views, the promoted columns, the per-dimension bin indices, the raveled multi-index and the density quotients are reclaimed; only <c>H</c> and the edge arrays leave.</remarks>
        [NDScoped]
        public static HistogramddResult histogramdd(NDArray sample, object[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
        {
            if (sample is null) throw new ArgumentNullException(nameof(sample));
            using var plain = NDMatrixBoundary.PlainFor(sample, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return sample.TensorEngine.Histogramdd(DefaultEngine.SampleColumns(sample), bins, range, density, weights);
        }

        // ---------------------------------------------------------------------
        //  histogramdd — sample given as a sequence of D column arrays
        // ---------------------------------------------------------------------

        /// <summary>
        ///     Compute the D-dimensional histogram from a sequence of D coordinate arrays (each element is the
        ///     list of values for one coordinate — NumPy's <c>histogramdd((X, Y, Z))</c> form).
        /// </summary>
        /// <param name="sample">D coordinate arrays, each of length N.</param>
        /// <param name="bins">Number of equal-width bins along every dimension (default 10).</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramddResult"/> is an <see cref="INDArrayCarrier"/>): the sample column views, the promoted columns, the per-dimension bin indices, the raveled multi-index and the density quotients are reclaimed; only <c>H</c> and the edge arrays leave.</remarks>
        [NDScoped]
        public static HistogramddResult histogramdd(NDArray[] sample, int bins = 10, (double, double)?[] range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            using var plain2 = NDMatrixBoundary.PlainFor(sample);
            return SequenceEngine(sample).Histogramdd(DefaultEngine.ColumnsFromSequence(sample), bins, range, density, weights);
        }

        /// <summary>Sequence-of-columns form with a per-dimension bin count.</summary>
        /// <param name="sample">D coordinate arrays, each of length N.</param>
        /// <param name="bins">Per-dimension bin counts (length must equal D).</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramddResult"/> is an <see cref="INDArrayCarrier"/>): the sample column views, the promoted columns, the per-dimension bin indices, the raveled multi-index and the density quotients are reclaimed; only <c>H</c> and the edge arrays leave.</remarks>
        [NDScoped]
        public static HistogramddResult histogramdd(NDArray[] sample, int[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            using var plain2 = NDMatrixBoundary.PlainFor(sample);
            return SequenceEngine(sample).Histogramdd(DefaultEngine.ColumnsFromSequence(sample), bins, range, density, weights);
        }

        /// <summary>Sequence-of-columns form with explicit per-dimension edge arrays.</summary>
        /// <param name="sample">D coordinate arrays, each of length N.</param>
        /// <param name="bins">Per-dimension edge arrays.</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramddResult"/> is an <see cref="INDArrayCarrier"/>): the sample column views, the promoted columns, the per-dimension bin indices, the raveled multi-index and the density quotients are reclaimed; only <c>H</c> and the edge arrays leave.</remarks>
        [NDScoped]
        public static HistogramddResult histogramdd(NDArray[] sample, NDArray[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            using var plain2 = NDMatrixBoundary.PlainFor(sample);
            using var plain3 = NDMatrixBoundary.PlainFor(bins);
            return SequenceEngine(sample).Histogramdd(DefaultEngine.ColumnsFromSequence(sample), bins, range, density, weights);
        }

        /// <summary>Sequence-of-columns form with a mixed per-dimension bin spec.</summary>
        /// <param name="sample">D coordinate arrays, each of length N.</param>
        /// <param name="bins">Per-dimension spec: each entry an int count or an edge array.</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramddResult"/> is an <see cref="INDArrayCarrier"/>): the sample column views, the promoted columns, the per-dimension bin indices, the raveled multi-index and the density quotients are reclaimed; only <c>H</c> and the edge arrays leave.</remarks>
        [NDScoped]
        public static HistogramddResult histogramdd(NDArray[] sample, object[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            using var plain2 = NDMatrixBoundary.PlainFor(sample);
            return SequenceEngine(sample).Histogramdd(DefaultEngine.ColumnsFromSequence(sample), bins, range, density, weights);
        }

        /// <summary>
        ///     The engine a sequence sample's <c>np.histogramdd</c> call computes on: the first column's engine (the operand
        ///     NumPy's statements read first), or <see cref="TensorEngine.Default"/> for a null, empty or null-led sequence,
        ///     which has no operand to decide and whose refusal the column conversion then reports as before.
        /// </summary>
        /// <param name="sample">The caller's sequence of D coordinate arrays, or null.</param>
        /// <returns>The engine of <c>sample[0]</c>, or <see cref="TensorEngine.Default"/>.</returns>
        /// <remarks>Chosen before the columns are converted (the conversion may make new arrays, bound to the default
        ///     engine), and without side effects, so the conversion's errors keep their order.</remarks>
        private static TensorEngine SequenceEngine(NDArray[] sample)
            => sample is { Length: > 0 } && sample[0] is { } first ? first.TensorEngine : TensorEngine.Default;
    }
}
