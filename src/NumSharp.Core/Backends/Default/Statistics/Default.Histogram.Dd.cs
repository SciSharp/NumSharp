using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // ---------------------------------------------------------------------
        //  histogramdd / histogram2d
        // ---------------------------------------------------------------------
        //
        // The bodies of NumPy's histogramdd and histogram2d (numpy/lib/_histograms_impl.py, numpy/lib/_twodim_base_impl.py):
        // the sample statement (np.atleast_2d(sample).T), the per-dimension bin specs and edges, the bin search, the
        // flattened bincount, the outlier strip and the density, and histogram2d's length check and bins dispatch. The
        // public np.histogramdd / np.histogram2d overloads only open the matrix boundary's plain region and call the
        // Histogramdd member (through SampleColumns or ColumnsFromSequence) / the Histogram2d member.
        //
        // Moved from Statistics/np.histogramdd.cs and Statistics/np.histogram2d.cs statement for statement by the np-layer
        // dtype plan's P2 (an operation class, decision D1: NumPy writes these functions in Python); its phase PE made them
        // DefaultEngine members over the static bodies HistogramddFunction / Histogram2dFunction. They share np.histogram's
        // outer edges, edge linspace, monotonic checks and contiguous cast (GetOuterEdges, HistogramLinspace, AnyDecreasing,
        // TextEdgesDecrease, EnsureContiguousDtype, FloatPromoteBinType).

        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.Histogramdd"/>: <see cref="HistogramddFunction"/> under this member's
        ///     scope.
        /// </summary>
        /// <param name="sample">The D coordinate columns (each 1-D, length N): the facades' <c>SampleColumns</c> /
        ///     <c>ColumnsFromSequence</c> form of NumPy's sample.</param>
        /// <param name="bins">The bin spec (int, int[], NDArray, NDArray[], or object[]).</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, normalize to a probability density.</param>
        /// <param name="weights">Optional per-point weights.</param>
        /// <returns>The histogram and its per-dimension edges.</returns>
        /// <exception cref="Exception">The errors of <see cref="HistogramddFunction"/>, in its order.</exception>
        /// <remarks>Owns its scope (a public entry since the plan's phase PE; the body was <c>[NDScopedCovered]</c> under
        ///     np.histogramdd's scope): every binning temporary is reclaimed, the histogram and the edges leave.</remarks>
        [NDScoped]
        public override np.HistogramddResult Histogramdd(NDArray[] sample, object bins, (double, double)?[] range = null,
            bool density = false, NDArray weights = null)
            => HistogramddFunction(sample, bins, range, density, weights);

        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.Histogram2d"/>: <see cref="Histogram2dFunction"/> under this member's
        ///     scope.
        /// </summary>
        /// <param name="x">The x coordinates.</param>
        /// <param name="y">The y coordinates.</param>
        /// <param name="bins">The polymorphic bin argument.</param>
        /// <param name="range">Optional per-dimension outer edges.</param>
        /// <param name="density">When true, return the probability density.</param>
        /// <param name="weights">Optional per-point weights.</param>
        /// <returns>The 2-D histogram and its two edge arrays.</returns>
        /// <exception cref="Exception">The errors of <see cref="Histogram2dFunction"/>, in its order.</exception>
        /// <remarks>Owns its scope (a public entry since the plan's phase PE; the body was <c>[NDScopedCovered]</c> under
        ///     np.histogram2d's scope): the stacked column and every binning temporary are reclaimed.</remarks>
        [NDScoped]
        public override np.Histogram2dResult Histogram2d(NDArray x, NDArray y, object bins, (double, double)?[] range = null,
            bool density = false, NDArray weights = null)
            => Histogram2dFunction(x, y, bins, range, density, weights);

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
        /// <param name="weights">Optional per-point weights: a 1-D array of N values, read as given (NumPy never ravels them).</param>
        /// <returns>The histogram and its per-dimension edges. Without density, H is NumPy's VIEW of the padded counts
        ///     (strides of the padded shape, not C-contiguous, OWNDATA False); a 1-D edges array the caller passed is
        ///     returned as that same array.</returns>
        /// <exception cref="ValueError">A dimension mismatch, a non-positive scalar bin, edges that are not a scalar or 1-D,
        ///     or non-monotonic edges.</exception>
        /// <exception cref="IncorrectShapeException">With points, weights that are not 1-D: bincount's depth check
        ///     (<c>object of too small depth for desired array</c> for 0-d weights, <c>object too deep for desired array</c>
        ///     for deeper ones; NumPy's ValueError, under the house dimension-error type).</exception>
        /// <exception cref="ValueError">With points, 1-D weights of another length than the points (bincount's
        ///     <c>The weights and list don't have the same length.</c>).</exception>
        /// <exception cref="InvalidCastException">With points, complex weights (bincount's safe cast to float64).</exception>
        /// <exception cref="IndexError">An empty edge array in any dimension, with or without points: NumPy's
        ///     <c>edges[i][-1]</c> (<c>index -1 is out of bounds for axis 0 with size 0</c>), read after every dimension's
        ///     search, so a search's refusal comes first.</exception>
        /// <exception cref="TypeError">A scalar bin that is not an integer (<c>`bins[i]` must be an integer, when a scalar</c>)
        ///     or that does not compare with 1 (text: <see cref="DdScalarBinBelowOne"/>). Text samples: the column's
        ///     <c>min()</c> (<c>minimum</c> has no text loop, named with the STACKED dtype) when the dimension has a scalar bin
        ///     spec and no range; text edges: their widths' <c>subtract</c> refusal; 1-D text weights with points: bincount's
        ///     safe cast to float64.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> column beside a <c>U</c> column holding a byte that is not ASCII:
        ///     NumPy's stacking, its first statement, decodes the bytes before any bin is read, so this wins over every bin
        ///     error.</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy stacks the columns into ONE array (<c>np.atleast_2d(sample).T</c>), so a text column makes the whole
        ///     sample text: <c>U2</c> beside float64 is <c>&lt;U32</c>, beside int64 <c>&lt;U21</c>, beside bool
        ///     <c>&lt;U5</c> (NumPy formats the numbers), <c>S2</c> beside <c>U1</c> <c>&lt;U2</c> (NumPy decodes the bytes
        ///     as ASCII). Every column is cast to that stacked dtype first (<see cref="CastColumnsToStacked"/>), as NumPy's
        ///     stacking does, and the formatted text decides the bins: <c>searchsorted</c> runs in the text descriptor common to
        ///     the column and the edges (numeric edges become text), so <c>"10"</c> sorts before <c>"3"</c>, and the
        ///     rightmost-edge test compares text with the last edge, which never equals a number.
        ///     </para>
        ///     <para>
        ///     Text edges of two or more elements are compared by the string <c>greater</c> kernel (NumPy's monotonicity
        ///     check); their widths then meet <c>subtract</c>'s refusal, a <see cref="TypeError"/>.
        ///     </para>
        ///     <para>
        ///     The per-dimension statements run in NumPy's order — the scalar bin's <c>bins[i] &lt; 1</c>, the outer edges,
        ///     <c>operator.index</c>, the edges, their widths — one dimension at a time, so an error of dimension 0 wins
        ///     over every error of dimension 1. An EMPTY text sample bins nothing: NumPy's searches and comparisons on it
        ///     change no count, so its histogram is the all-zero one of an empty numeric sample.
        ///     </para>
        /// </remarks>
        [NDScopedCovered] // only callers: the [NDScoped] Histogramdd and Histogram2d overrides (Histogram2dFunction)
        internal static np.HistogramddResult HistogramddFunction(NDArray[] cols, object bins, (double, double)?[] range,
            bool density, NDArray weights)
        {
            // NumPy builds the internal sample via `atleast_2d(sample).T`, which STACKS the D coordinate
            // columns into ONE array and thereby promotes every column to their common dtype. An (N, D)
            // array (or any single-dtype sequence) is already uniform, so this is a no-op there; a sequence
            // of DIFFERENTLY-typed columns — e.g. histogram2d(float32_x, float64_y) — must be promoted so its
            // per-dimension edges AND the binning comparison run in the one common dtype, matching NumPy
            // (whose stacked sample makes both edges float64). Done first, before any edge/bin computation
            // reads cols[i].typecode. A text sample is cast to its stacked text dtype here too (CastColumnsToStacked):
            // NumPy's stacking formats the number columns and decodes bytes beside str before it reads a bin.
            DType stacked = StackedSampleDescr(cols);
            bool textSample = stacked?.Meta is StringDTypeMeta;
            // A StringDType column: NumPy's stack is that StringDType (with StringDType / U columns) or, beside numbers, an
            // object array whose columns keep their own values — every column then computes in its own dtype here.
            bool vstringSample = false;
            foreach (var col in cols)
                vstringSample |= Backends.Strings.NDStringUfuncs.IsVString(col);
            if (vstringSample)
                cols = stacked is not null ? CastColumnsToStacked(cols, stacked) : cols;
            else
                cols = textSample ? CastColumnsToStacked(cols, stacked) : PromoteColumnsToCommon(cols);

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
                    if (Backends.Strings.NDStringUfuncs.IsText(earr))
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
                    // A text sample's column is the STACKED text column (CastColumnsToStacked), so the outer edges' a.min()
                    // (np.amin, in GetOuterEdges) names that dtype when its minimum loop refuses (<U32 beside float64).
                    if (cols[i].typecode == NPTypeCode.Complex)
                    {
                        // A complex column: NumPy's same statements in complex128. `_get_outer_edges` gives the column's
                        // lexicographic min / max (complex128 scalars, both parts checked finite, a zero width widened by
                        // 0.5), so `np.linspace(smin, smax, n + 1)` — no dtype — runs in complex128; a range's Python floats
                        // or an empty column's ints 0 and 1 have no array to make them complex, so those edges stay float64.
                        ComplexOuterEdges ce = GetComplexOuterEdges(cols[i], ri);
                        int cnb = DdOperatorIndex(spec, i);
                        edges[i] = ce.Strong
                            ? ComplexEqualWidthEdges(ce, cnb)
                            : HistogramLinspace(ce.First.Real, ce.Last.Real, cnb + 1, NPTypeCode.Double, NPTypeCode.Double);
                    }
                    else
                    {
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
                }

                nbin[i] = (int)edges[i].size + 1; // +1 for an outlier bin on each end (shared low+high span)
                // `dedges[i] = np.diff(edges[i])`: text edges have no subtract loop, and np.diff's subtraction refuses them at
                // any length (an empty difference resolves the loop too).
                dedges[i] = edges[i].TensorEngine.Diff(edges[i], 1, -1, null, null);
            }

            if ((textSample || vstringSample) && N == 0)
            {
                // With no points the searches, the rightmost-edge comparisons and the bincount see nothing, so the
                // counts are the empty numeric sample's. (With points, every column already is the stacked text dtype:
                // NumPy searches the numeric edges converted to text, so the bins follow string order, "10" before "3".)
                cols = new NDArray[D];
                for (int i = 0; i < D; i++)
                    cols[i] = new NDArray(NPTypeCode.Double, Shape.Vector(0), true);
            }

            // Per-dimension bin number for every point: searchsorted(edges, x, 'right'), then pull points that
            // sit exactly on the rightmost edge back into the last real bin (else they count as an outlier).
            // A fused SIMD/binary search-index kernel does this in one pass per dimension (the general
            // np.searchsorted has heavy per-key overhead — the whole dd hot loop otherwise); Complex edges
            // (no ordering) fall back to the composed searchsorted + on-edge subtract.
            //
            // NumPy writes these as two loops, and their order is observable: `Ncount = tuple(np.searchsorted(edges[i],
            // sample[:, i], side='right') for i in range(D))` searches EVERY dimension first, and only then `for i in
            // range(D): on_edge = (sample[:, i] == edges[i][-1])` reads each rightmost edge. So a search's refusal in any
            // dimension (StringDType text against numeric edges) wins over an EMPTY edge array's IndexError, which
            // `edges[i][-1]` raises in the second loop — with or without points.
            var Ncount = new NDArray[D];
            // The dimensions whose search left the rightmost-edge correction to the second loop (text and complex columns);
            // the fused kernel's dimensions are complete after the first.
            var needsOnEdge = new bool[D];
            for (int i = 0; i < D; i++)
            {
                if ((textSample || Backends.Strings.NDStringUfuncs.IsVString(cols[i])) && N > 0)
                {
                    // A text column: NumPy's searchsorted in the text descriptor common to the column and the edges.
                    Ncount[i] = edges[i].TensorEngine.SearchSorted(edges[i], cols[i], "right", null);
                    needsOnEdge[i] = true;
                    continue;
                }
                // An empty edge array: its search finds nothing, and the second loop's `edges[i][-1]` refuses it. The fused
                // kernel must not run on it: it reads the last edge, which an empty array does not have.
                if (edges[i].size == 0)
                    continue;
                NPTypeCode cmp = DTypePromotion.PromoteTypes(edges[i].typecode, cols[i].typecode);
                var searchIdx = DirectILKernelGenerator.GetHistogramSearchIndexKernel(cmp);
                // The fused kernel counts the edges at or below each point, which is NumPy's binary search only over SORTED
                // edges. An edge array holding a NaN passes the monotonic check (no comparison with a NaN is true) but is not
                // sorted for the search, whose ordering puts NaN last, so NumPy's searchsorted itself decides those indices.
                if (searchIdx != null && !EdgesHoldNaN(edges[i]))
                {
                    NDArray colC = EnsureContiguousDtype(cols[i], cmp);
                    NDArray edgeC = EnsureContiguousDtype(edges[i], cmp);
                    var ncnt = new NDArray(NPTypeCode.Int64, Shape.Vector(N), fillZeros: false);
                    unsafe { searchIdx((void*)colC.Address, N, (void*)edgeC.Address, (int)edges[i].size, (long*)ncnt.Address); }
                    Ncount[i] = ncnt;
                }
                else
                {
                    Ncount[i] = edges[i].TensorEngine.SearchSorted(edges[i], cols[i], "right", null);
                    needsOnEdge[i] = true;
                }
            }
            for (int i = 0; i < D; i++)
            {
                // `edges[i][-1]`, for every dimension: an empty edge array raises NumPy's IndexError here ("index -1 is out of
                // bounds for axis 0 with size 0").
                using NDArray lastEdge = edges[i]["-1"];
                if (!needsOnEdge[i])
                    continue;
                // Text against a number compares unequal everywhere, as NumPy's == answers. The search result and the masks
                // are this dimension's own temporaries: only the difference is kept.
                using NDArray found = Ncount[i];
                using NDArray onEdge = cols[i] == lastEdge;
                using NDArray onEdge64 = onEdge.astype(NPTypeCode.Int64);
                Ncount[i] = found - onEdge64;
            }

            // Flatten the D bin numbers into one index into the padded histogram, tally, reshape, drop outliers.
            // np.ravel_multi_index(Ncount, nbin)'s statement: the first coordinate array's engine (the default one without
            // dimensions), mode 'raise', C order.
            TensorEngine indexEngine = D > 0 ? Ncount[0].TensorEngine : BackendFactory.GetEngine();
            NDArray xy = indexEngine.RavelMultiIndex(Ncount, nbin, "raise", 'C');

            long minlen = 1;
            foreach (int nb in nbin) minlen *= nb;

            // `np.bincount(xy, weights, minlength=nbin.prod())`: NumPy hands the weights over AS GIVEN (np.asarray only), and
            // bincount reads them only after its empty-input early return, as a 1-D float64 array — so a 0-d or deeper
            // weights array is refused by its depth check (object of too small depth / object too deep for desired array)
            // before the safe cast refuses a complex or text one, and the length is compared last. A text array of the right
            // depth meets that cast here, with NumPy's text (NumSharp's bincount does not convert text).
            if (weights is not null && weights.ndim == 1 && xy.size > 0 && Backends.Strings.NDStringUfuncs.IsText(weights))
                throw new TypeError($"Cannot cast array data from {weights.dtype.ToString(true)} to dtype('float64') according to the rule 'safe'");
            NDArray flat = xy.TensorEngine.Bincount(xy, weights, (int)minlen);
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
                NDArray s = hist.TensorEngine.Sum(hist, null, null, false);
                for (int i = 0; i < D; i++)
                {
                    long[] shape = new long[D];
                    for (int d = 0; d < D; d++) shape[d] = 1;
                    shape[i] = nbin[i] - 2;
                    hist = hist / dedges[i].reshape(Shape.Adopt(shape));
                }
                hist = hist / s;
            }

            return new np.HistogramddResult(hist, edges);
        }

        // ---------------------------------------------------------------------
        //  Sample / bins normalization helpers
        // ---------------------------------------------------------------------

        /// <summary>
        ///     NumPy's first statement for an array sample: <c>N, D = sample.shape</c>, and where that unpacking fails (a
        ///     sample that is not 2-D), <c>sample = np.atleast_2d(sample).T</c> and <c>N, D = sample.shape</c> again. An
        ///     <c>(N, D)</c> sample is its D column views; a 1-D sample becomes <c>(N, 1)</c>, one column of N points, and a 0-d
        ///     sample <c>(1, 1)</c>, one column of one point; a deeper sample keeps its rank through <c>atleast_2d</c>, so the
        ///     second unpacking fails too.
        /// </summary>
        /// <param name="sample">The sample array.</param>
        /// <returns>The D coordinate columns, each a 1-D view of <paramref name="sample"/> (a 1-D sample is its own column).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="sample"/> is null.</exception>
        /// <exception cref="ValueError">A sample of three or more dimensions: Python's <c>too many values to unpack
        ///     (expected 2)</c>, raised before any bin, range or weight is read.</exception>
        internal static NDArray[] SampleColumns(NDArray sample)
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
            // atleast_2d(()).T is (1, 1): the 0-d value is one point of one column, read through a (1,) view of it.
            if (sample.ndim == 0)
                return new[] { sample.reshape(1) };
            throw new ValueError("too many values to unpack (expected 2)");
        }

        /// <summary>
        ///     NumPy's first statement for a sequence sample: <c>sample = np.atleast_2d(sample).T</c>, then
        ///     <c>N, D = sample.shape</c>. The stacking is array coercion of the list of arrays, which checks every entry's
        ///     shape against the first one's; <c>atleast_2d</c> and <c>.T</c> then decide what a column is.
        /// </summary>
        /// <param name="sample">The sequence's entries (NumPy's list or tuple of arrays). Not modified.</param>
        /// <returns><paramref name="sample"/> itself when its entries are D 1-D arrays of one length (each entry is one column,
        ///     <c>(D, N)</c> transposed); a new one-element array holding ONE column when the entries are 0-d (the stack is
        ///     <c>(n,)</c>, which <c>atleast_2d(..).T</c> makes <c>(n, 1)</c>: n points of one dimension) or when there are no
        ///     entries (NumPy's float64 <c>(0,)</c> stack, one empty column).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="sample"/> is null.</exception>
        /// <exception cref="ValueError">Entries of different shapes (array coercion's <c>setting an array element with a
        ///     sequence. The requested array has an inhomogeneous shape after N dimensions. The detected shape was (...) +
        ///     inhomogeneous part.</c>), or entries of two or more dimensions (<c>too many values to unpack (expected 2)</c>):
        ///     both before any bin, range or weight is read.</exception>
        /// <exception cref="NotSupportedException">0-d entries NumPy stacks into an OBJECT array — a null entry (None), or
        ///     StringDType text beside a number — which NumPy then computes with as Python objects; NumSharp has no object
        ///     dtype.</exception>
        /// <remarks>
        ///     The shape check runs FIRST, before an entry's values are read: the columns used to be read with the first
        ///     entry's length, which read past the end of a shorter entry. A null entry beside arrays is a scalar of the
        ///     coercion, so it is ragged like a 0-d array would be.
        /// </remarks>
        internal static NDArray[] ColumnsFromSequence(NDArray[] sample)
        {
            if (sample is null) throw new ArgumentNullException(nameof(sample));
            // np.atleast_2d(sample) coerces the list first: its shape walk (array_coercion.c's update_shape, ported as
            // Backends.UnmanagedStorage.SequenceNDim) compares every entry's dims with the first entry's, so entries of different shapes raise
            // NumPy's inhomogeneous ValueError here, before anything is read from them.
            int ndim = Backends.UnmanagedStorage.SequenceNDim(sample);
            // `N, D = sample.shape` after `.T`: atleast_2d leaves a stack of two or more dims as it is, so entries of two or
            // more dims leave more than two values to unpack.
            if (ndim > 2)
                throw new ValueError("too many values to unpack (expected 2)");
            // (D, N) transposed: every entry is one column of N points (the stacking's dtype promotion runs in Histogramdd).
            if (ndim == 2)
                return sample;
            // (n,) -> (1, n) -> (n, 1): the entries are 0-d, the n values of ONE column.
            return new[] { StackScalarEntries(sample) };
        }

        /// <summary>
        ///     The one column NumPy's stack makes of 0-d entries: <c>np.asanyarray(sample)</c> is an <c>(n,)</c> array of the
        ///     entries' common dtype — promoted as arrays are (every entry strong, not NEP 50), a text entry making the whole
        ///     column text (<c>&lt;U32</c> beside a float64) — and no entries at all make NumPy's float64 <c>(0,)</c> array.
        /// </summary>
        /// <param name="entries">The 0-d entries (or none). Not modified.</param>
        /// <returns>A new 1-D array of the n entry values in the stacked dtype.</returns>
        /// <exception cref="NotSupportedException">A null entry, or StringDType text beside a number: NumPy's stack is then
        ///     an object array, a dtype NumSharp does not have.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> entry holding a byte that is not ASCII beside a <c>U</c> entry
        ///     (the stack decodes the bytes, as for columns).</exception>
        private static NDArray StackScalarEntries(NDArray[] entries)
        {
            // np.asanyarray(()) is an empty float64 array: one column without points.
            if (entries.Length == 0)
                return new NDArray(NPTypeCode.Double, Shape.Vector(0), false);
            foreach (var e in entries)
                if (e is null)
                    throw new NotSupportedException("None has no NumSharp equivalent: NumPy stacks the sample into an object " +
                                                    "array, a dtype NumSharp does not have");
            // The common dtype of the stack, through the descriptor promotion the columns use (string widths kept); null when
            // StringDType text meets a number, where NumPy's stack is an object array of Python values.
            DType stacked = StackedSampleDescr(entries);
            if (stacked is null)
                throw new NotSupportedException("StringDType text beside a number makes NumPy stack the sample into an " +
                                                "object array, which it computes with as Python objects; NumSharp has no " +
                                                "object dtype");
            // Each entry converted to the stacked dtype (a number formatted as text, bytes decoded), then the n values side by
            // side: every part already has the stacked dtype, so the concatenation promotes nothing.
            NDArray[] cast = CastColumnsToStacked(entries, stacked);
            var parts = new NDArray[cast.Length];
            for (int i = 0; i < cast.Length; i++)
                parts[i] = cast[i].reshape(1);
            return NDArray.concatenate(parts, 0, null, null, "same_kind");
        }

        /// <summary>
        ///     NumPy's <c>atleast_2d(sample).T</c> for a TEXT sample: every column converted to the stacked text dtype, a
        ///     number column formatted as NumPy formats it and an <c>S</c> column beside a <c>U</c> one decoded as ASCII.
        ///     The stacking is histogramdd's first statement, so its decode error comes before any bin specification is read;
        ///     and the columns it returns are what the bin search compares, since the formatted text decides the bins.
        /// </summary>
        /// <param name="cols">The sample's columns, as the caller passed them (the array may be the caller's own).</param>
        /// <param name="stacked">The stacked text dtype (<see cref="StackedSampleDescr"/>).</param>
        /// <returns><paramref name="cols"/> itself when every column already has the stacked dtype; otherwise a NEW array
        ///     holding the converted columns (the caller's array is never written: <c>histogramdd(NDArray[] sample, …)</c>
        ///     hands its own array in).</returns>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> column holding a byte that is not ASCII, converted to a <c>U</c>
        ///     stacked dtype.</exception>
        private static NDArray[] CastColumnsToStacked(NDArray[] cols, DType stacked)
        {
            NDArray[] cast = null;
            for (int i = 0; i < cols.Length; i++)
            {
                if (cols[i].dtype == stacked)
                    continue;
                // Copy the array on the first column that changes, so a uniform sample allocates nothing.
                cast ??= (NDArray[])cols.Clone();
                cast[i] = cols[i].astype(stacked);
            }
            return cast ?? cols;
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
                common = DTypePromotion.PromoteTypes(common, cols[i].typecode);

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
        /// <returns>One raw spec per dimension; <see cref="HistogramddFunction"/> checks each in NumPy's order.</returns>
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
            double[] d => new NDArray(Unmanaged.ArraySlice.FromArray(d, true), Shape.Vector(d.Length)),
            long[] l => new NDArray(Unmanaged.ArraySlice.FromArray(l, true), Shape.Vector(l.Length)),
            int[] ii => new NDArray(Unmanaged.ArraySlice.FromArray(ii, true), Shape.Vector(ii.Length)),
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
                        // NumPy hands out a StringDType element as a Python str, a U element as an np.str_.
                        string => throw new TypeError(
                            $"'<' not supported between instances of '{(item.Descr.Meta is VStringDTypeMeta ? "str" : "numpy.str_")}' and 'int'"),
                        // A StringDType element that is the NA object None.
                        null => throw new TypeError("'<' not supported between instances of 'NoneType' and 'int'"),
                        byte[] => throw new TypeError("'<' not supported between instances of 'numpy.bytes_' and 'int'"),
                        _ => NumericBelowOne(item.Value),
                    };
                case NDArray nd:
                    if (Backends.Strings.NDStringUfuncs.IsText(nd))
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
            {
                try
                {
                    common = DTypePromotion.PromoteTypes(common, cols[i].dtype);
                }
                catch (DTypePromotionError) when (AnyVString(cols))
                {
                    // StringDType beside a number has no common DType: NumPy's stack is then an OBJECT array, whose columns
                    // keep their own values (null: no stacked descriptor).
                    return null;
                }
            }
            return common;
        }

        /// <summary>Whether any column holds NEP 55 <c>StringDType</c> text.</summary>
        /// <param name="cols">The columns.</param>
        /// <returns>True when one does.</returns>
        private static bool AnyVString(NDArray[] cols)
        {
            foreach (var col in cols)
                if (Backends.Strings.NDStringUfuncs.IsVString(col))
                    return true;
            return false;
        }

        /// <summary>Raise NumPy's dimension-mismatch error when a bin sequence length differs from the sample rank.</summary>
        /// <param name="have">The bin sequence's length (NumPy's <c>len(bins)</c>).</param>
        /// <param name="D">The sample's dimension count.</param>
        /// <exception cref="ValueError">The lengths differ (<c>The dimension of bins must be equal to the dimension of the
        ///     sample x.</c>).</exception>
        private static void RequireDdLength(int have, int D)
        {
            if (have != D)
                throw new ValueError("The dimension of bins must be equal to the dimension of the sample x.");
        }

        /// <summary>Convert an int[] shape to the long[] the reshape API expects.</summary>
        /// <param name="dims">The per-dimension bin counts (outlier bins included).</param>
        /// <returns>A new array of the same values.</returns>
        private static long[] ToLongShape(int[] dims)
        {
            var r = new long[dims.Length];
            for (int i = 0; i < dims.Length; i++) r[i] = dims[i];
            return r;
        }

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
        /// <exception cref="ArgumentNullException"><paramref name="x"/> or <paramref name="y"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="x"/> or <paramref name="y"/> is 0-d: NumPy's first statement takes
        ///     <c>len()</c> of both (<c>len() of unsized object</c>), before any bin, range or weight is read.</exception>
        /// <exception cref="ValueError"><paramref name="x"/> and <paramref name="y"/> differ in length — their FIRST
        ///     dimension, not their size (<c>x and y must have the same length.</c>); arrays of one length but different
        ///     shapes, which the stacking of <c>[x, y]</c> refuses (<c>setting an array element with a sequence. The requested
        ///     array has an inhomogeneous shape after N dimensions. ...</c>); two 2-D arrays of one shape
        ///     (<c>too many values to unpack (expected 2)</c>); and every refusal of the histogramdd engine the two arrays and
        ///     the arguments then meet.</exception>
        [NDScopedCovered] // only caller: the [NDScoped] Histogram2d override (the stacked column and every binning temporary are reclaimed)
        internal static np.Histogram2dResult Histogram2dFunction(NDArray x, NDArray y, object bins, (double, double)?[] range,
            bool density, NDArray weights)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (y is null) throw new ArgumentNullException(nameof(y));
            // NumPy's first statement, `if len(x) != len(y)`: len() of a 0-d array raises (x's first, as Python evaluates it
            // first), and len() is the FIRST dimension rather than the size — so a (2, 3) x and a (3, 2) y differ here, while a
            // (2, 3) x and a (2, 4) y pass on to histogramdd, whose stacking of [x, y] refuses them.
            if (x.__len__() != y.__len__())
                throw new ValueError("x and y must have the same length.");

            int n = BinsSeqLength(bins);
            object ddBins = bins;
            if (n != 1 && n != 2)
            {
                // `xedges = yedges = asarray(bins)`: ONE conversion, the same array for both dimensions (NumPy returns that
                // one object as xedges and as yedges). It runs before histogramdd, so its own refusal — a ragged list's
                // inhomogeneous shape — comes before the sample's.
                NDArray shared = AsEdgeArray(bins);
                ddBins = new NDArray[] { shared, shared };
            }

            // `histogramdd([x, y], ...)`: the pair is a sequence sample, so its stacking statement runs on it — two arrays of
            // one length but different shapes are refused there, and two 2-D arrays of one shape leave three values to unpack.
            np.HistogramddResult dd = HistogramddFunction(ColumnsFromSequence(new[] { x, y }), ddBins, range, density, weights);
            return new np.Histogram2dResult(dd.H, dd.edges[0], dd.edges[1]);
        }

        /// <summary>
        ///     NumPy's <c>try: N = len(bins) except TypeError: N = 1</c> — a scalar bin count (or a 0-d array, which has no
        ///     <c>len</c>) reports length 1; a sequence reports its element count and an array its FIRST dimension.
        /// </summary>
        /// <param name="bins">The bin spec.</param>
        /// <returns>NumPy's <c>N</c>.</returns>
        /// <exception cref="ValueError">A bin spec of an unsupported C# type.</exception>
        /// <remarks>
        ///     <c>len</c> of a 2-D array is its row count, not its size: a <c>(2, k)</c> array is one edge row per dimension
        ///     (histogramdd indexes it), where reading its size made it a shared edge array and failed.
        /// </remarks>
        private static int BinsSeqLength(object bins) => bins switch
        {
            int => 1,
            int[] a => a.Length,
            long[] a => a.Length,
            double[] a => a.Length,
            NDArray[] a => a.Length,
            object[] a => a.Length,
            NDArray nd => nd.ndim == 0 ? 1 : checked((int)nd.Shape.dimensions[0]),
            _ => throw new ValueError("`bins` must be an int or a sequence")
        };

        /// <summary>
        ///     NumPy's <c>asarray(bins)</c> for histogram2d's shared edges: an <see cref="NDArray"/> is itself, a typed C# array
        ///     is an array of its dtype, and an <c>NDArray[]</c> / <c>object[]</c> is a Python list, converted by array
        ///     coercion — stacked when its entries have one shape (a list of edge arrays is a 2-D array, which histogramdd
        ///     then refuses per dimension), refused with NumPy's inhomogeneous-shape ValueError when they do not (a count
        ///     beside an edge array).
        /// </summary>
        /// <param name="bins">histogram2d's bin argument, when its length is not 1 or 2.</param>
        /// <returns>The argument itself for an <see cref="NDArray"/>, else a new array.</returns>
        /// <exception cref="ValueError">A list whose entries have different shapes (<c>setting an array element with a
        ///     sequence. The requested array has an inhomogeneous shape after N dimensions. ...</c>), or a C# form NumPy has no
        ///     counterpart for (<c>shared `bins` must be a 1-D edge array</c>).</exception>
        /// <exception cref="NotSupportedException">A list holding text or None: NumPy's array is then a str or object one,
        ///     which NumSharp's histogram edges do not model.</exception>
        private static NDArray AsEdgeArray(object bins) => bins switch
        {
            NDArray nd => nd,
            double[] a => new NDArray(Unmanaged.ArraySlice.FromArray(a, true), Shape.Vector(a.Length)),
            int[] a => new NDArray(Unmanaged.ArraySlice.FromArray(a, true), Shape.Vector(a.Length)),
            long[] a => new NDArray(Unmanaged.ArraySlice.FromArray(a, true), Shape.Vector(a.Length)),
            NDArray[] or object[] => SharedEdgesFromList(bins),
            _ => throw new ValueError("shared `bins` must be a 1-D edge array")
        };

        /// <summary>
        ///     <c>np.asarray</c> of histogram2d's list of bins (an <c>NDArray[]</c> or <c>object[]</c>): array coercion's shape
        ///     walk first, then the numeric array its entries promote to.
        /// </summary>
        /// <param name="list">The list.</param>
        /// <returns>A new C-contiguous array (float64 <c>(0,)</c> for an empty list).</returns>
        /// <exception cref="ValueError">Entries of different shapes: NumPy's inhomogeneous-shape text, which the walk raises
        ///     before any entry's dtype is read.</exception>
        /// <exception cref="NotSupportedException">An entry that is text (a str, bytes or a text array) or None.</exception>
        private static NDArray SharedEdgesFromList(object list)
        {
            // The shape walk alone (Backends.UnmanagedStorage.SequenceNDim, array_coercion.c's update_shape): NumPy reports raggedness before it
            // looks at what the entries hold.
            Backends.UnmanagedStorage.SequenceNDim(list);
            // A text array entry: the coercion's numeric fill cannot read it (NumPy's array is a str one, see below).
            foreach (object item in Backends.UnmanagedStorage.SequenceItems(list))
                if (item is NDArray a && Backends.Strings.NDStringUfuncs.IsText(a))
                    throw StrOrObjectEdges();
            // null: NumPy's str or object array (a str / None item anywhere in the list).
            return Backends.UnmanagedStorage.FromSequenceOrNonNumeric(list, out _) ?? throw StrOrObjectEdges();

            static NotSupportedException StrOrObjectEdges() => new NotSupportedException(
                "histogram2d's bins list holds text or None: NumPy's np.asarray(bins) makes it a str or object array, " +
                "which NumSharp's histogram edges do not model");
        }
    }
}
