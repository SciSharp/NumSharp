using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The bi-dimensional histogram plus its two edge arrays — the three values NumPy's
        ///     <c>histogram2d</c> returns. Converts implicitly to <see cref="NDArray"/> (the histogram <c>H</c>)
        ///     and deconstructs as <c>var (H, xedges, yedges) = np.histogram2d(x, y);</c>.
        /// </summary>
        public readonly struct Histogram2dResult : INDArrayCarrier
        {
            /// <summary>The 2-D histogram, shape <c>(nx, ny)</c>, dtype float64.</summary>
            public readonly NDArray H;
            /// <summary>The bin edges along the first (x) dimension.</summary>
            public readonly NDArray xedges;
            /// <summary>The bin edges along the second (y) dimension.</summary>
            public readonly NDArray yedges;

            internal Histogram2dResult(NDArray h, NDArray xedges, NDArray yedges) { H = h; this.xedges = xedges; this.yedges = yedges; }

            /// <summary>Deconstructs into <c>(H, xedges, yedges)</c>.</summary>
            /// <param name="h">Receives the histogram.</param>
            /// <param name="xedges">Receives the x edges.</param>
            /// <param name="yedges">Receives the y edges.</param>
            public void Deconstruct(out NDArray h, out NDArray xedges, out NDArray yedges) { h = H; xedges = this.xedges; yedges = this.yedges; }

            /// <summary>Yields the primary result — the histogram <c>H</c>.</summary>
            /// <param name="r">The result to unwrap.</param>
            public static implicit operator NDArray(Histogram2dResult r) => r.H;

            void INDArrayCarrier.YieldTo(NDScope scope) { scope.Returns(H); scope.Returns(xedges); scope.Returns(yedges); }
        }

        /// <summary>
        ///     Compute the 2-D histogram of two data samples with <paramref name="bins"/> equal-width bins along
        ///     each dimension. Note NumPy's convention: <paramref name="x"/> is histogrammed along the first
        ///     (row) axis and <paramref name="y"/> along the second (column) axis.
        /// </summary>
        /// <param name="x">The x coordinates, shape <c>(N,)</c>.</param>
        /// <param name="y">The y coordinates, shape <c>(N,)</c>.</param>
        /// <param name="bins">Number of equal-width bins along both dimensions (default 10).</param>
        /// <param name="range">Optional per-dimension <c>(lower, upper)</c> outer edges (<c>[[xmin,xmax],[ymin,ymax]]</c>).</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="Histogram2dResult"/> carrying <c>H</c>, the x edges and the y edges.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html <para>Scoped (<c>[NDScoped]</c>; <see cref="Histogram2dResult"/> is an <see cref="INDArrayCarrier"/>): the shared-edge conversions, the promoted coordinate columns and every per-dimension binning intermediate are reclaimed; only <c>H</c> and the two edge arrays leave.</para></remarks>
        [NDScoped]
        public static Histogram2dResult histogram2d(NDArray x, NDArray y, int bins = 10, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => Histogram2dImpl(x, y, bins, range, density, weights);

        /// <summary>2-D histogram with a per-dimension bin count <c>[nx, ny]</c>.</summary>
        /// <param name="x">The x coordinates, shape <c>(N,)</c>.</param>
        /// <param name="y">The y coordinates, shape <c>(N,)</c>.</param>
        /// <param name="bins">Per-dimension bin counts <c>[nx, ny]</c>.</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="Histogram2dResult"/> carrying <c>H</c>, the x edges and the y edges.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="Histogram2dResult"/> is an <see cref="INDArrayCarrier"/>): the shared-edge conversions, the promoted coordinate columns and every per-dimension binning intermediate are reclaimed; only <c>H</c> and the two edge arrays leave.</remarks>
        [NDScoped]
        public static Histogram2dResult histogram2d(NDArray x, NDArray y, int[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => Histogram2dImpl(x, y, bins, range, density, weights);

        /// <summary>2-D histogram with a single edge array shared by both dimensions (<c>x_edges = y_edges = bins</c>).</summary>
        /// <param name="x">The x coordinates, shape <c>(N,)</c>.</param>
        /// <param name="y">The y coordinates, shape <c>(N,)</c>.</param>
        /// <param name="bins">Shared bin edges for both dimensions (must have more than 2 entries — a 2-entry array is read as <c>[nx, ny]</c>).</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="Histogram2dResult"/> carrying <c>H</c>, the x edges and the y edges.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="Histogram2dResult"/> is an <see cref="INDArrayCarrier"/>): the shared-edge conversions, the promoted coordinate columns and every per-dimension binning intermediate are reclaimed; only <c>H</c> and the two edge arrays leave.</remarks>
        [NDScoped]
        public static Histogram2dResult histogram2d(NDArray x, NDArray y, NDArray bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => Histogram2dImpl(x, y, bins, range, density, weights);

        /// <summary>2-D histogram with explicit per-dimension edge arrays <c>[x_edges, y_edges]</c> (or a mixed <c>[int, array]</c>).</summary>
        /// <param name="x">The x coordinates, shape <c>(N,)</c>.</param>
        /// <param name="y">The y coordinates, shape <c>(N,)</c>.</param>
        /// <param name="bins">Per-dimension edge arrays.</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="Histogram2dResult"/> carrying <c>H</c>, the x edges and the y edges.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="Histogram2dResult"/> is an <see cref="INDArrayCarrier"/>): the shared-edge conversions, the promoted coordinate columns and every per-dimension binning intermediate are reclaimed; only <c>H</c> and the two edge arrays leave.</remarks>
        [NDScoped]
        public static Histogram2dResult histogram2d(NDArray x, NDArray y, NDArray[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => Histogram2dImpl(x, y, bins, range, density, weights);

        /// <summary>2-D histogram with a mixed per-dimension bin spec <c>[nx, y_edges]</c> or <c>[x_edges, ny]</c>.</summary>
        /// <param name="x">The x coordinates, shape <c>(N,)</c>.</param>
        /// <param name="y">The y coordinates, shape <c>(N,)</c>.</param>
        /// <param name="bins">Per-dimension spec, each entry an int count or an edge array.</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="Histogram2dResult"/> carrying <c>H</c>, the x edges and the y edges.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="Histogram2dResult"/> is an <see cref="INDArrayCarrier"/>): the shared-edge conversions, the promoted coordinate columns and every per-dimension binning intermediate are reclaimed; only <c>H</c> and the two edge arrays leave.</remarks>
        [NDScoped]
        public static Histogram2dResult histogram2d(NDArray x, NDArray y, object[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => Histogram2dImpl(x, y, bins, range, density, weights);

        /// <summary>2-D histogram with a single edge array (C# <c>double[]</c>) shared by both dimensions.</summary>
        /// <param name="x">The x coordinates, shape <c>(N,)</c>.</param>
        /// <param name="y">The y coordinates, shape <c>(N,)</c>.</param>
        /// <param name="bins">Shared bin edges (more than 2 entries).</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="Histogram2dResult"/> carrying <c>H</c>, the x edges and the y edges.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="Histogram2dResult"/> is an <see cref="INDArrayCarrier"/>): the shared-edge conversions, the promoted coordinate columns and every per-dimension binning intermediate are reclaimed; only <c>H</c> and the two edge arrays leave.</remarks>
        [NDScoped]
        public static Histogram2dResult histogram2d(NDArray x, NDArray y, double[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => Histogram2dImpl(x, y, bins, range, density, weights);

        /// <summary>
        ///     Shared core: reproduce NumPy's <c>len(bins)</c> dispatch — a bin spec that is not a 1- or 2-element
        ///     sequence is a single edge array shared by both dimensions — then delegate to the histogramdd engine.
        /// </summary>
        /// <param name="x">The x coordinates.</param>
        /// <param name="y">The y coordinates.</param>
        /// <param name="bins">The polymorphic bin argument.</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional per-point weights.</param>
        /// <returns>The 2-D histogram and its two edge arrays.</returns>
        /// <exception cref="ValueError"><paramref name="x"/> and <paramref name="y"/> differ in length.</exception>
        private static Histogram2dResult Histogram2dImpl(NDArray x, NDArray y, object bins, (double, double)?[] range, bool density, NDArray weights)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (y is null) throw new ArgumentNullException(nameof(y));
            if (x.size != y.size)
                throw new ValueError("x and y must have the same length.");

            int n = BinsSeqLength(bins);
            object ddBins = (n != 1 && n != 2)
                ? new NDArray[] { AsEdgeArray(bins), AsEdgeArray(bins) } // xedges = yedges = asarray(bins)
                : bins;

            HistogramddResult dd = HistogramddImpl(new[] { x, y }, ddBins, range, density, weights);
            return new Histogram2dResult(dd.H, dd.edges[0], dd.edges[1]);
        }

        /// <summary>
        ///     NumPy's <c>try: N = len(bins) except TypeError: N = 1</c> — a scalar bin count reports length 1;
        ///     any sequence (or edge array) reports its element count.
        /// </summary>
        private static int BinsSeqLength(object bins) => bins switch
        {
            int => 1,
            int[] a => a.Length,
            long[] a => a.Length,
            double[] a => a.Length,
            NDArray[] a => a.Length,
            object[] a => a.Length,
            NDArray nd => nd.ndim == 0 ? 1 : (int)nd.size,
            _ => throw new ValueError("`bins` must be an int or a sequence")
        };

        /// <summary>Coerce a shared-edge bin argument to an <see cref="NDArray"/> of edges.</summary>
        private static NDArray AsEdgeArray(object bins) => bins switch
        {
            NDArray nd => nd,
            double[] a => np.array(a),
            int[] a => np.array(a),
            long[] a => np.array(a),
            _ => throw new ValueError("shared `bins` must be a 1-D edge array")
        };
    }
}
