using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html</remarks>
        public static HistogramddResult histogramdd(NDArray sample, int bins = 10, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => HistogramddImpl(SampleColumns(sample), bins, range, density, weights);

        /// <summary>Compute the D-dimensional histogram with a per-dimension bin count.</summary>
        /// <param name="sample">An <c>(N, D)</c> array of N points in D-dimensional space.</param>
        /// <param name="bins">Per-dimension bin counts (length must equal D).</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        public static HistogramddResult histogramdd(NDArray sample, int[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => HistogramddImpl(SampleColumns(sample), bins, range, density, weights);

        /// <summary>Compute the D-dimensional histogram with explicit per-dimension bin edges.</summary>
        /// <param name="sample">An <c>(N, D)</c> array of N points in D-dimensional space.</param>
        /// <param name="bins">Per-dimension edge arrays (length must equal D; a 0-d entry is read as a bin count).</param>
        /// <param name="range">Optional per-dimension outer edges (used only where <paramref name="bins"/> gives a count).</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        public static HistogramddResult histogramdd(NDArray sample, NDArray[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => HistogramddImpl(SampleColumns(sample), bins, range, density, weights);

        /// <summary>Compute the D-dimensional histogram with a mixed per-dimension bin spec (each entry an int count or an edge array).</summary>
        /// <param name="sample">An <c>(N, D)</c> array of N points in D-dimensional space.</param>
        /// <param name="bins">Per-dimension spec: each entry an <see cref="int"/> count, an <see cref="NDArray"/> / <c>double[]</c> edge array.</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        public static HistogramddResult histogramdd(NDArray sample, object[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => HistogramddImpl(SampleColumns(sample), bins, range, density, weights);

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
        public static HistogramddResult histogramdd(NDArray[] sample, int bins = 10, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => HistogramddImpl(ColumnsFromSequence(sample), bins, range, density, weights);

        /// <summary>Sequence-of-columns form with a per-dimension bin count.</summary>
        /// <param name="sample">D coordinate arrays, each of length N.</param>
        /// <param name="bins">Per-dimension bin counts (length must equal D).</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        public static HistogramddResult histogramdd(NDArray[] sample, int[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => HistogramddImpl(ColumnsFromSequence(sample), bins, range, density, weights);

        /// <summary>Sequence-of-columns form with explicit per-dimension edge arrays.</summary>
        /// <param name="sample">D coordinate arrays, each of length N.</param>
        /// <param name="bins">Per-dimension edge arrays.</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        public static HistogramddResult histogramdd(NDArray[] sample, NDArray[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => HistogramddImpl(ColumnsFromSequence(sample), bins, range, density, weights);

        /// <summary>Sequence-of-columns form with a mixed per-dimension bin spec.</summary>
        /// <param name="sample">D coordinate arrays, each of length N.</param>
        /// <param name="bins">Per-dimension spec: each entry an int count or an edge array.</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        public static HistogramddResult histogramdd(NDArray[] sample, object[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
            => HistogramddImpl(ColumnsFromSequence(sample), bins, range, density, weights);

        // ---------------------------------------------------------------------
        //  Core
        // ---------------------------------------------------------------------

        /// <summary>
        ///     The histogramdd algorithm (port of numpy/lib/_histograms_impl.py::histogramdd): build per-dimension
        ///     edges, assign each point to a flattened bin via <c>searchsorted</c> + <c>ravel_multi_index</c>, tally
        ///     with <c>bincount</c>, reshape, strip the per-dimension outlier bins, and optionally normalize.
        /// </summary>
        /// <param name="cols">The D coordinate columns (each 1-D, length N).</param>
        /// <param name="bins">The bin spec (int, int[], NDArray[], or object[]).</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, normalize to a probability density.</param>
        /// <param name="weights">Optional per-point weights.</param>
        /// <returns>The histogram and its per-dimension edges.</returns>
        /// <exception cref="ValueError">A dimension mismatch, a non-positive integer bin, or non-monotonic edges.</exception>
        private static HistogramddResult HistogramddImpl(NDArray[] cols, object bins, (double, double)?[] range, bool density, NDArray weights)
        {
            int D = cols.Length;
            long N = D == 0 ? 0 : cols[0].size;

            object[] binSpec = NormalizeDdBins(bins, D);

            // Normalize the range argument to one (nullable) entry per dimension.
            if (range != null && range.Length != D)
                throw new ValueError("range argument must have one entry per dimension");

            var edges = new NDArray[D];
            var dedges = new NDArray[D];
            var nbin = new int[D];

            for (int i = 0; i < D; i++)
            {
                (double, double)? ri = range?[i];
                object spec = binSpec[i];

                if (spec is NDArray earr && earr.ndim == 1)
                {
                    edges[i] = earr;
                    if (AnyDecreasing(edges[i]))
                        throw new ValueError($"`bins[{i}]` must be monotonically increasing, when an array");
                }
                else if (spec is NDArray earr0 && earr0.ndim > 1)
                {
                    throw new ValueError($"`bins[{i}]` must be a scalar or 1d array");
                }
                else
                {
                    // Scalar bin count (int or 0-d integer array).
                    int nb = AsBinCount(spec);
                    if (nb < 1)
                        throw new ValueError($"`bins[{i}]` must be positive, when an integer");
                    (double smin, double smax) = GetOuterEdges(cols[i], ri);
                    // NumPy's `np.linspace(smin, smax, n+1)` runs in the sample dtype's float width:
                    // int/float64 => float64, float32 => float32, float16 => float16 (NEP50 weak-float `num`
                    // never widens a strong float column). Compute IN that width for bit-exact edges.
                    edges[i] = HistogramLinspace(smin, smax, nb + 1, FloatPromoteBinType(cols[i].typecode));
                }

                nbin[i] = (int)edges[i].size + 1; // +1 for an outlier bin on each end (shared low+high span)
                dedges[i] = np.diff(edges[i]);
            }

            // Per-dimension bin number for every point: searchsorted(edges, x, 'right'), then pull points that
            // sit exactly on the rightmost edge back into the last real bin (else they count as an outlier).
            // A fused SIMD/binary search-index kernel does this in one pass per dimension (the general
            // np.searchsorted has heavy per-key overhead — the whole dd hot loop otherwise); Complex edges
            // (no ordering) fall back to the composed searchsorted + on-edge subtract.
            var Ncount = new NDArray[D];
            for (int i = 0; i < D; i++)
            {
                NPTypeCode cmp = promote_types(edges[i].typecode, cols[i].typecode);
                var searchIdx = DirectILKernelGenerator.GetHistogramSearchIndexKernel(cmp);
                if (searchIdx != null)
                {
                    NDArray colC = EnsureContiguousDtype(cols[i], cmp);
                    NDArray edgeC = EnsureContiguousDtype(edges[i], cmp);
                    var ncnt = new NDArray(NPTypeCode.Int64, Shape.Vector(N), fillZeros: false);
                    unsafe { searchIdx((void*)colC.Address, N, (void*)edgeC.Address, (int)edges[i].size, (long*)ncnt.Address); }
                    Ncount[i] = ncnt;
                }
                else
                {
                    NDArray idx = np.searchsorted(edges[i], cols[i], "right");
                    NDArray onEdge = cols[i] == edges[i]["-1"];
                    Ncount[i] = idx - onEdge.astype(NPTypeCode.Int64);
                }
            }

            // Flatten the D bin numbers into one index into the padded histogram, tally, reshape, drop outliers.
            NDArray xy = np.ravel_multi_index(Ncount, nbin);

            long minlen = 1;
            foreach (int nb in nbin) minlen *= nb;

            NDArray w1d = weights is null ? null : np.ravel(weights);
            NDArray flat = bincount(xy, w1d, minlength: (int)minlen);
            NDArray hist = flat.reshape(ToLongShape(nbin));
            hist = hist.astype(NPTypeCode.Double); // NumPy: hist.astype(float, casting='safe') — H is always float64

            // Strip the leading/trailing outlier bin on every dimension: hist[1:-1, 1:-1, ...].
            string core = string.Join(", ", System.Linq.Enumerable.Repeat("1:-1", D));
            hist = D == 0 ? hist : hist[core].copy();

            if (density)
            {
                NDArray s = np.sum(hist);
                for (int i = 0; i < D; i++)
                {
                    long[] shape = new long[D];
                    for (int d = 0; d < D; d++) shape[d] = 1;
                    shape[i] = nbin[i] - 2;
                    hist = hist / dedges[i].reshape(shape);
                }
                hist = hist / s;
            }

            return new HistogramddResult(hist, edges);
        }

        // ---------------------------------------------------------------------
        //  Sample / bins normalization helpers
        // ---------------------------------------------------------------------

        /// <summary>
        ///     Split an <c>(N, D)</c> sample array into D coordinate columns (each a 1-D view). A 1-D sample is
        ///     read as a single column (D = 1), matching NumPy's <c>atleast_2d(sample).T</c> fallback.
        /// </summary>
        private static NDArray[] SampleColumns(NDArray sample)
        {
            if (sample is null) throw new ArgumentNullException(nameof(sample));
            if (sample.ndim == 2)
            {
                int d = (int)sample.shape[1];
                var cols = new NDArray[d];
                for (int i = 0; i < d; i++)
                    cols[i] = sample[$":, {i}"];
                return cols;
            }
            if (sample.ndim == 1)
                return new[] { sample }; // atleast_2d((N,)).T == (N,1) => a single column
            throw new ValueError("sample must be a 2-D (N, D) array or a sequence of 1-D arrays");
        }

        /// <summary>Use a sequence of D coordinate arrays directly as the columns.</summary>
        private static NDArray[] ColumnsFromSequence(NDArray[] sample)
        {
            if (sample is null) throw new ArgumentNullException(nameof(sample));
            return sample;
        }

        /// <summary>
        ///     Normalize the polymorphic <c>bins</c> argument to exactly D per-dimension specs, each an
        ///     <see cref="int"/> count or an <see cref="NDArray"/> edge array. Mirrors NumPy's
        ///     "int broadcasts to all dims; a sequence must match D".
        /// </summary>
        private static object[] NormalizeDdBins(object bins, int D)
        {
            var result = new object[D];
            switch (bins)
            {
                case int n:
                    for (int i = 0; i < D; i++) result[i] = n;
                    return result;
                case int[] arr:
                    RequireDdLength(arr.Length, D);
                    for (int i = 0; i < D; i++) result[i] = arr[i];
                    return result;
                case NDArray[] narr:
                    RequireDdLength(narr.Length, D);
                    for (int i = 0; i < D; i++) result[i] = narr[i];
                    return result;
                case object[] oarr:
                    RequireDdLength(oarr.Length, D);
                    for (int i = 0; i < D; i++) result[i] = NormalizeDdEntry(oarr[i]);
                    return result;
                case NDArray nd0 when nd0.ndim == 0:
                    // A 0-d scalar count broadcasts to every dimension (operator.index).
                    for (int i = 0; i < D; i++) result[i] = AsBinCount(nd0);
                    return result;
                case NDArray nd1:
                    // A bare 1-D array is NumPy's `len(bins)==D` per-dimension form: EACH element is a scalar
                    // bin count (NOT shared edges — histogram2d pre-expands shared edges into an NDArray[]).
                    RequireDdLength((int)nd1.size, D);
                    for (int i = 0; i < D; i++) result[i] = (int)Convert.ToInt64(nd1.GetAtIndex(i));
                    return result;
                default:
                    throw new ValueError("`bins` must be an int, a sequence of ints, or a sequence of edge arrays");
            }
        }

        /// <summary>Coerce one entry of an <c>object[]</c> bin spec to an int count or an NDArray edge array.</summary>
        private static object NormalizeDdEntry(object entry) => entry switch
        {
            int n => n,
            NDArray nd => nd,
            double[] d => (object)np.array(d),
            long[] l => (object)np.array(l),
            int[] ii => (object)np.array(ii),
            _ => throw new ValueError("each `bins` entry must be an int or a 1-D edge array")
        };

        /// <summary>Raise NumPy's dimension-mismatch error when a bin sequence length differs from the sample rank.</summary>
        private static void RequireDdLength(int have, int D)
        {
            if (have != D)
                throw new ValueError("The dimension of bins must be equal to the dimension of the sample x.");
        }

        /// <summary>Convert an int[] shape to the long[] the reshape API expects.</summary>
        private static long[] ToLongShape(int[] dims)
        {
            var r = new long[dims.Length];
            for (int i = 0; i < dims.Length; i++) r[i] = dims[i];
            return r;
        }
    }
}
