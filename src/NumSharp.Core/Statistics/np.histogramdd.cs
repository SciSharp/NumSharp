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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html <para>Scoped (<c>[NDScoped]</c>; <see cref="HistogramddResult"/> is an <see cref="INDArrayCarrier"/>): the sample column views, the promoted columns, the per-dimension bin indices, the raveled multi-index and the density quotients are reclaimed; only <c>H</c> and the edge arrays leave.</para></remarks>
        [NDScoped]
        public static HistogramddResult histogramdd(NDArray sample, int bins = 10, (double, double)?[] range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(sample, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return HistogramddImpl(SampleColumns(sample), bins, range, density, weights);
        }

        /// <summary>Compute the D-dimensional histogram with a per-dimension bin count.</summary>
        /// <param name="sample">An <c>(N, D)</c> array of N points in D-dimensional space.</param>
        /// <param name="bins">Per-dimension bin counts (length must equal D).</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramddResult"/> is an <see cref="INDArrayCarrier"/>): the sample column views, the promoted columns, the per-dimension bin indices, the raveled multi-index and the density quotients are reclaimed; only <c>H</c> and the edge arrays leave.</remarks>
        [NDScoped]
        public static HistogramddResult histogramdd(NDArray sample, int[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(sample, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return HistogramddImpl(SampleColumns(sample), bins, range, density, weights);
        }

        /// <summary>Compute the D-dimensional histogram with explicit per-dimension bin edges.</summary>
        /// <param name="sample">An <c>(N, D)</c> array of N points in D-dimensional space.</param>
        /// <param name="bins">Per-dimension edge arrays (length must equal D; a 0-d entry is read as a bin count).</param>
        /// <param name="range">Optional per-dimension outer edges (used only where <paramref name="bins"/> gives a count).</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramddResult"/> is an <see cref="INDArrayCarrier"/>): the sample column views, the promoted columns, the per-dimension bin indices, the raveled multi-index and the density quotients are reclaimed; only <c>H</c> and the edge arrays leave.</remarks>
        [NDScoped]
        public static HistogramddResult histogramdd(NDArray sample, NDArray[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(sample, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            using var plain2 = NDMatrixBoundary.PlainFor(bins);
            return HistogramddImpl(SampleColumns(sample), bins, range, density, weights);
        }

        /// <summary>Compute the D-dimensional histogram with a mixed per-dimension bin spec (each entry an int count or an edge array).</summary>
        /// <param name="sample">An <c>(N, D)</c> array of N points in D-dimensional space.</param>
        /// <param name="bins">Per-dimension spec: each entry an <see cref="int"/> count, an <see cref="NDArray"/> / <c>double[]</c> edge array.</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional <c>(N,)</c> per-point weights.</param>
        /// <returns>A <see cref="HistogramddResult"/> carrying <c>H</c> and the per-dimension edges.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramddResult"/> is an <see cref="INDArrayCarrier"/>): the sample column views, the promoted columns, the per-dimension bin indices, the raveled multi-index and the density quotients are reclaimed; only <c>H</c> and the edge arrays leave.</remarks>
        [NDScoped]
        public static HistogramddResult histogramdd(NDArray sample, object[] bins, (double, double)?[] range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(sample, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return HistogramddImpl(SampleColumns(sample), bins, range, density, weights);
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
            return HistogramddImpl(ColumnsFromSequence(sample), bins, range, density, weights);
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
            return HistogramddImpl(ColumnsFromSequence(sample), bins, range, density, weights);
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
            return HistogramddImpl(ColumnsFromSequence(sample), bins, range, density, weights);
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
            return HistogramddImpl(ColumnsFromSequence(sample), bins, range, density, weights);
        }

        // ---------------------------------------------------------------------
        //  Core
        // ---------------------------------------------------------------------

        /// <summary>
        ///     The histogramdd algorithm (port of numpy/lib/_histograms_impl.py::histogramdd): build per-dimension
        ///     edges, assign each point to a flattened bin via <c>searchsorted</c> + <c>ravel_multi_index</c>, tally
        ///     with <c>bincount</c>, reshape, strip the per-dimension outlier bins, and optionally normalize.
        /// </summary>
        /// <param name="cols">The D coordinate columns (each 1-D, length N).</param>
        /// <param name="bins">The bin spec (int, int[], NDArray, NDArray[], or object[]).</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, normalize to a probability density.</param>
        /// <param name="weights">Optional per-point weights.</param>
        /// <returns>The histogram and its per-dimension edges. Without density, H is NumPy's VIEW of the padded counts
        ///     (strides of the padded shape, not C-contiguous, OWNDATA False); a 1-D edges array the caller passed is
        ///     returned as that same array.</returns>
        /// <exception cref="ValueError">A dimension mismatch, a non-positive scalar bin, edges that are not a scalar or 1-D,
        ///     or non-monotonic edges.</exception>
        /// <exception cref="TypeError">A scalar bin that is not an integer (<c>`bins[i]` must be an integer, when a scalar</c>)
        ///     or that does not compare with 1 (text: <see cref="DdScalarBinBelowOne"/>). Text samples: the column's
        ///     <c>min()</c> (<c>minimum</c> has no text loop, named with the STACKED dtype) when the dimension has a scalar bin
        ///     spec and no range; text edges: their widths' <c>subtract</c> refusal; text weights with points: bincount's
        ///     safe cast to float64.</exception>
        /// <exception cref="StringComputationNotSupportedException">Text edges of two or more elements (NumPy compares them),
        ///     or a text sample with points to bin (NumPy searches the edges, comparing text).</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy stacks the columns into ONE array (<c>np.atleast_2d(sample).T</c>), so a text column makes the whole
        ///     sample text: <c>U2</c> beside float64 is <c>&lt;U32</c>, beside int64 <c>&lt;U21</c>, beside bool
        ///     <c>&lt;U5</c> (NumPy formats the numbers; the answers below never need the formatted text, only its dtype).
        ///     </para>
        ///     <para>
        ///     The per-dimension statements run in NumPy's order — the scalar bin's <c>bins[i] &lt; 1</c>, the outer edges,
        ///     <c>operator.index</c>, the edges, their widths — one dimension at a time, so an error of dimension 0 wins
        ///     over every error of dimension 1. An EMPTY text sample bins nothing: NumPy's searches and comparisons on it
        ///     change no count, so its histogram is the all-zero one of an empty numeric sample.
        ///     </para>
        /// </remarks>
        private static HistogramddResult HistogramddImpl(NDArray[] cols, object bins, (double, double)?[] range, bool density, NDArray weights)
        {
            // NumPy builds the internal sample via `atleast_2d(sample).T`, which STACKS the D coordinate
            // columns into ONE array and thereby promotes every column to their common dtype. An (N, D)
            // array (or any single-dtype sequence) is already uniform, so this is a no-op there; a sequence
            // of DIFFERENTLY-typed columns — e.g. histogram2d(float32_x, float64_y) — must be promoted so its
            // per-dimension edges AND the binning comparison run in the one common dtype, matching NumPy
            // (whose stacked sample makes both edges float64). Done first, before any edge/bin computation
            // reads cols[i].typecode. A text sample keeps its columns: only the stacked DTYPE is needed.
            DType stacked = StackedSampleDescr(cols);
            bool textSample = stacked?.Meta is StringDTypeMeta;
            if (!textSample)
                cols = PromoteColumnsToCommon(cols);

            int D = cols.Length;
            long N = D == 0 ? 0 : cols[0].size;

            object[] binSpec = DdBinSpecs(bins, D);

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
                    // Fewer than two text edges have nothing to compare; two or more are compared by the string greater kernel.
                    if (Backends.Strings.NDStringUfuncs.IsString(earr))
                    {
                        if (earr.size > 1 && TextEdgesDecrease(earr))
                            throw new ValueError($"`bins[{i}]` must be monotonically increasing, when an array");
                    }
                    else if (AnyDecreasing(edges[i]))
                        throw new ValueError($"`bins[{i}]` must be monotonically increasing, when an array");
                }
                else if (spec is NDArray earr0 && earr0.ndim > 1)
                {
                    throw new ValueError($"`bins[{i}]` must be a scalar or 1d array");
                }
                else
                {
                    // `np.ndim(bins[i]) == 0`: `if bins[i] < 1: raise`, the outer edges, then `operator.index(bins[i])`.
                    if (DdScalarBinBelowOne(spec))
                        throw new ValueError($"`bins[{i}]` must be positive, when an integer");
                    // A text sample's column is the STACKED text column: its min() names that dtype (<U32 beside float64).
                    if (textSample && !ri.HasValue && N > 0)
                        throw Backends.Strings.NDStringUfuncs.NoLoop("minimum", stacked, stacked);
                    (double smin, double smax) = GetOuterEdges(cols[i], ri);
                    int nb = DdOperatorIndex(spec, i);
                    // histogramdd calls `np.linspace(smin, smax, n+1)` WITHOUT a dtype, so the edges take
                    // result_type(smin, smax, n+1) — which, unlike np.histogram, has NO array to keep it strong.
                    // The column's own float width survives ONLY when the endpoints are strong array scalars
                    // (non-empty AND no range → smin/smax = col.min()/max()); a supplied range (Python floats) or an
                    // empty column (the 0/1 int defaults) makes the endpoints weak, collapsing the edges to float64.
                    // So a ranged or empty float32/float16 column yields float64 edges — matching NumPy (a
                    // float32-column-with-strong-endpoints keeps float32). Compute AND result are the same dtype here.
                    bool weakEndpoints = ri.HasValue || N == 0;
                    NPTypeCode edgeType = weakEndpoints ? NPTypeCode.Double : FloatPromoteBinType(cols[i].typecode);
                    edges[i] = HistogramLinspace(smin, smax, nb + 1, edgeType, edgeType);
                }

                nbin[i] = (int)edges[i].size + 1; // +1 for an outlier bin on each end (shared low+high span)
                // `dedges[i] = np.diff(edges[i])`: text edges have no subtract loop (any length — even an empty diff resolves it).
                if (Backends.Strings.NDStringUfuncs.IsString(edges[i]))
                    throw Backends.Strings.NDStringUfuncs.NoLoop("subtract", edges[i].dtype, edges[i].dtype);
                dedges[i] = np.diff(edges[i]);
            }

            if (textSample)
            {
                // NumPy now searches every edge array for the text points (string computation). With no points the searches,
                // the rightmost-edge comparisons and the bincount see nothing, so the counts are the empty numeric sample's.
                if (N > 0)
                    throw Backends.Strings.NDStringUfuncs.NextPhase("searchsorted", stacked);
                cols = new NDArray[D];
                for (int i = 0; i < D; i++)
                    cols[i] = np.zeros(Shape.Vector(0), NPTypeCode.Double);
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
            if (Backends.Strings.NDStringUfuncs.IsString(w1d))
            {
                // np.bincount converts the weights with a SAFE cast to float64 — after its empty-input early return, which
                // never reads them (so no points means the plain zero counts).
                if (xy.size > 0)
                    throw new TypeError($"Cannot cast array data from {w1d.dtype.ToString(true)} to dtype('float64') according to the rule 'safe'");
                w1d = null;
            }
            NDArray flat = bincount(xy, w1d, minlength: (int)minlen);
            NDArray hist = flat.reshape(Shape.Adopt(ToLongShape(nbin)));
            hist = hist.astype(NPTypeCode.Double); // NumPy: hist.astype(float, casting='safe') — H is always float64

            // Strip the leading/trailing outlier bin on every dimension: hist[1:-1, 1:-1, ...]. NumPy returns that slice as a
            // VIEW of the padded counts (non-contiguous, OWNDATA False — the padded buffer stays alive behind it), and so
            // does NumSharp: the view takes its own reference on the buffer, so the scope reclaiming the padded array
            // releases only its reference. The density branch below divides into fresh arrays, as NumPy's does.
            string core = string.Join(", ", System.Linq.Enumerable.Repeat("1:-1", D));
            hist = D == 0 ? hist : hist[core];

            if (density)
            {
                NDArray s = np.sum(hist);
                for (int i = 0; i < D; i++)
                {
                    long[] shape = new long[D];
                    for (int d = 0; d < D; d++) shape[d] = 1;
                    shape[i] = nbin[i] - 2;
                    hist = hist / dedges[i].reshape(Shape.Adopt(shape));
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
                int d = (int)sample.Shape.dimensions[1];
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
        ///     Promote a set of coordinate columns to their common dtype, reproducing the dtype promotion
        ///     NumPy's <c>atleast_2d(sample).T</c> stacking performs. Columns that already share a dtype
        ///     (the overwhelmingly common case, and always true for the <c>(N, D)</c>-array sample form) are
        ///     returned UNCHANGED with no copy; only a genuinely mixed-dtype sequence — the case that
        ///     otherwise diverges from NumPy, e.g. <c>histogram2d(float32, float64)</c> whose edges must both
        ///     be float64 — is cast, each column to <c>result_type</c> of all columns. Promoting up front is
        ///     what makes both the computed edges' dtype/values AND the per-dimension binning comparison run
        ///     in the one dtype NumPy's stacked sample uses.
        /// </summary>
        /// <param name="cols">The coordinate columns (each 1-D); may be same- or mixed-dtype.</param>
        /// <returns>The originals when already uniform; otherwise each column cast to their common dtype.</returns>
        private static NDArray[] PromoteColumnsToCommon(NDArray[] cols)
        {
            if (cols.Length < 2) return cols; // 0-D/1-D histogramdd: nothing to promote against

            // Fold the columns' dtypes through NEP50 promotion, exactly the result_type NumPy's stack uses.
            NPTypeCode common = cols[0].typecode;
            for (int i = 1; i < cols.Length; i++)
                common = promote_types(common, cols[i].typecode);

            // Uniform input needs no cast — return the originals so the common path allocates nothing.
            bool needsCast = false;
            for (int i = 0; i < cols.Length; i++)
                if (cols[i].typecode != common) { needsCast = true; break; }
            if (!needsCast) return cols;

            var promoted = new NDArray[cols.Length];
            for (int i = 0; i < cols.Length; i++)
                promoted[i] = cols[i].typecode == common ? cols[i] : cols[i].astype(common);
            return promoted;
        }

        /// <summary>
        ///     NumPy's <c>bins[i]</c> for each dimension: <c>try: M = len(bins) … except TypeError: bins = D * [bins]</c>. A
        ///     scalar (or a 0-d array, which has no <c>len</c>) is repeated for every dimension; a sequence must have D entries
        ///     and each entry is kept AS GIVEN — an <see cref="int"/> or other C# scalar (a Python scalar), an
        ///     <see cref="NDArray"/> of any rank (a 1-D one is that dimension's edges), or, for the entries of one
        ///     <see cref="NDArray"/> <paramref name="bins"/>, what indexing it gives: a NumPy scalar (<see cref="DdScalarItem"/>)
        ///     from a 1-D array, a row view from a deeper one.
        /// </summary>
        /// <param name="bins">The bin spec as received.</param>
        /// <param name="D">The sample's dimension count.</param>
        /// <returns>One raw spec per dimension; <see cref="HistogramddImpl"/> checks each in NumPy's order.</returns>
        /// <exception cref="ValueError">A sequence whose length is not D (<c>The dimension of bins must be equal to the
        ///     dimension of the sample x.</c>), or an entry of an unsupported C# type.</exception>
        /// <remarks>
        ///     Nothing is converted here: NumPy compares each scalar with 1, reads the outer edges and only then converts the
        ///     scalar with <c>operator.index</c> — dimension by dimension — so a float entry like <c>2.0</c> is NumPy's
        ///     <c>`bins[0]` must be an integer, when a scalar</c> (it used to be rounded to a count), and a text entry is the
        ///     comparison's TypeError (it used to reach <c>Convert.ToInt64</c>, a <see cref="FormatException"/>).
        /// </remarks>
        private static object[] DdBinSpecs(object bins, int D)
        {
            var result = new object[D];
            switch (bins)
            {
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
                    // len() of a 0-d array is a TypeError: the array itself is every dimension's scalar spec.
                    for (int i = 0; i < D; i++) result[i] = nd0;
                    return result;
                case NDArray nd:
                    // len(bins) is the FIRST dimension, and bins[i] indexes it: a NumPy scalar of a 1-D array, a row of a
                    // deeper one (a 2-D (D, k) array is one edge row per dimension).
                    RequireDdLength(checked((int)nd.Shape.dimensions[0]), D);
                    for (int i = 0; i < D; i++)
                        result[i] = nd.ndim == 1 ? new DdScalarItem(nd.GetAtIndex(i), nd.dtype) : nd[i];
                    return result;
                default:
                    // A C# int (or any other scalar NumPy would take as a Python scalar): no len(), so D copies.
                    for (int i = 0; i < D; i++) result[i] = NormalizeDdEntry(bins);
                    return result;
            }
        }

        /// <summary>
        ///     One entry of an <c>object[]</c> bin spec as NumPy sees it: C# scalars are Python scalars (kept for the
        ///     per-dimension checks), C# number arrays become 1-D edge arrays, NDArrays are kept.
        /// </summary>
        /// <param name="entry">The entry.</param>
        /// <returns>The entry, or the <see cref="NDArray"/> of a C# number array.</returns>
        /// <exception cref="ValueError">An entry of a C# type NumPy has no counterpart for.</exception>
        private static object NormalizeDdEntry(object entry) => entry switch
        {
            null => null,
            NDArray nd => nd,
            double[] d => np.array(d),
            long[] l => np.array(l),
            int[] ii => np.array(ii),
            bool or sbyte or byte or short or ushort or int or uint or long or ulong => entry,
            double or float or Half or decimal or System.Numerics.Complex or string or byte[] => entry,
            _ => throw new ValueError("each `bins` entry must be an int or a 1-D edge array")
        };

        /// <summary>
        ///     A NumPy scalar taken out of a 1-D <see cref="NDArray"/> bin spec (<c>bins[i]</c>): its value as
        ///     <see cref="NDArray.GetAtIndex(long)"/> gives it and the array's dtype — distinct from a 0-d array, whose
        ///     comparison with 1 is a ufunc call with a different error text.
        /// </summary>
        private sealed class DdScalarItem
        {
            /// <summary>Wraps one element.</summary>
            /// <param name="value">The element's C# value (a <see cref="string"/> / <c>byte[]</c> for text).</param>
            /// <param name="descr">The array's dtype.</param>
            internal DdScalarItem(object value, DType descr)
            {
                Value = value;
                Descr = descr;
            }

            /// <summary>The element's C# value.</summary>
            internal object Value { get; }

            /// <summary>The dtype of the array it came from.</summary>
            internal DType Descr { get; }
        }

        /// <summary>
        ///     NumPy's <c>if bins[i] &lt; 1</c> for a scalar per-dimension spec: a numeric value compares numerically (complex
        ///     lexicographically, NaN never below), anything else raises the comparison's TypeError.
        /// </summary>
        /// <param name="spec">A C# scalar (Python scalar), a <see cref="DdScalarItem"/> (NumPy scalar) or a 0-d
        ///     <see cref="NDArray"/>.</param>
        /// <returns>True when the spec is below 1 (NumPy's <c>`bins[i]` must be positive, when an integer</c>).</returns>
        /// <exception cref="TypeError">Text, a Python complex or <c>None</c>: Python's <c>'&lt;' not supported between instances
        ///     of 'numpy.str_' and 'int'</c> (<c>numpy.bytes_</c>, <c>str</c>, <c>bytes</c>, <c>complex</c>, <c>NoneType</c>); a
        ///     0-d text array: the ufunc's <c>ufunc 'less' did not contain a loop with signature matching types (&lt;class
        ///     'numpy.dtypes.StrDType'&gt;, &lt;class 'numpy.dtypes._PyLongDType'&gt;) -&gt; None</c>.</exception>
        private static bool DdScalarBinBelowOne(object spec)
        {
            switch (spec)
            {
                case DdScalarItem item:
                    return item.Value switch
                    {
                        string => throw new TypeError("'<' not supported between instances of 'numpy.str_' and 'int'"),
                        byte[] => throw new TypeError("'<' not supported between instances of 'numpy.bytes_' and 'int'"),
                        _ => NumericBelowOne(item.Value),
                    };
                case NDArray nd:
                    if (Backends.Strings.NDStringUfuncs.IsString(nd))
                        throw new TypeError(
                            $"ufunc 'less' did not contain a loop with signature matching types ({nd.dtype.Meta}, {DTypeRegistry.PyLong}) -> None");
                    return NumericBelowOne(nd.GetAtIndex(0));
                case null:
                    throw new TypeError("'<' not supported between instances of 'NoneType' and 'int'");
                case string:
                    throw new TypeError("'<' not supported between instances of 'str' and 'int'");
                case byte[]:
                    throw new TypeError("'<' not supported between instances of 'bytes' and 'int'");
                case System.Numerics.Complex:
                    // A Python complex has no ordering (NumPy's complex scalars do: see NumericBelowOne).
                    throw new TypeError("'<' not supported between instances of 'complex' and 'int'");
                default:
                    return NumericBelowOne(spec);
            }
        }

        /// <summary><c>v &lt; 1</c> for a numeric C# value: NaN is never below; a NumPy complex orders lexicographically.</summary>
        /// <param name="v">The value (bool, integer, char, float, decimal or <see cref="System.Numerics.Complex"/>).</param>
        /// <returns>Whether it is below 1.</returns>
        private static bool NumericBelowOne(object v) => v switch
        {
            bool b => !b,
            System.Numerics.Complex z => z.Real < 1 || (z.Real == 1 && z.Imaginary < 0),
            Half h => (double)h < 1,
            decimal m => m < 1,
            double d => d < 1,
            float f => f < 1,
            ulong u => u < 1,
            _ => Convert.ToInt64(v, System.Globalization.CultureInfo.InvariantCulture) < 1,
        };

        /// <summary>
        ///     NumPy's <c>operator.index(bins[i])</c> for a scalar per-dimension spec: a Python int (any C# integer type or
        ///     bool — a Python bool IS an int), or an INTEGER NumPy scalar / 0-d array (NumPy's bool is not an index).
        /// </summary>
        /// <param name="spec">The scalar spec (already compared with 1).</param>
        /// <param name="i">The dimension, for the message.</param>
        /// <returns>The bin count.</returns>
        /// <exception cref="TypeError">Anything else: <c>`bins[i]` must be an integer, when a scalar</c>.</exception>
        private static int DdOperatorIndex(object spec, int i)
        {
            switch (spec)
            {
                case bool b:
                    return b ? 1 : 0;
                case sbyte or byte or short or ushort or int or uint or long or ulong:
                    return checked((int)Convert.ToInt64(spec, System.Globalization.CultureInfo.InvariantCulture));
                case DdScalarItem item when item.Descr.typecode.IsInteger():
                    return checked((int)Convert.ToInt64(item.Value, System.Globalization.CultureInfo.InvariantCulture));
                case NDArray nd when nd.ndim == 0 && nd.typecode.IsInteger():
                    return checked((int)Convert.ToInt64(nd.GetAtIndex(0), System.Globalization.CultureInfo.InvariantCulture));
                default:
                    throw new TypeError($"`bins[{i}]` must be an integer, when a scalar");
            }
        }

        /// <summary>
        ///     The dtype of NumPy's stacked sample (<c>np.atleast_2d(sample).T</c>): the common dtype of all columns, through
        ///     the descriptor promotion that keeps string widths (<c>&lt;U2</c> + float64 is <c>&lt;U32</c>).
        /// </summary>
        /// <param name="cols">The columns.</param>
        /// <returns>The common descriptor, or null for no columns.</returns>
        private static DType StackedSampleDescr(NDArray[] cols)
        {
            if (cols.Length == 0)
                return null;
            DType common = cols[0].dtype;
            for (int i = 1; i < cols.Length; i++)
                common = DTypePromotion.PromoteTypes(common, cols[i].dtype);
            return common;
        }

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
