using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        // Block size for the chunked accumulation paths, verbatim from NumPy's
        // `histogram` (numpy/lib/_histograms_impl.py). It is NOT just a memory lever:
        // weighted float sums are accumulated per 65536-sample block and only then
        // folded into the running total, so replicating this block boundary is what
        // keeps weighted histograms BIT-IDENTICAL to NumPy (a single-pass sum diverges
        // in the last ULP — verified against 2.4.2).
        private const int HistogramBlock = 65536;

        /// <summary>
        ///     The bin edges of the histogram plus (for equal-width bins) the parameters of the
        ///     optimized uniform path. Purely internal plumbing shared by
        ///     <see cref="histogram_bin_edges(NDArray,int,ValueTuple{double,double}?,NDArray)"/> and
        ///     <see cref="histogram(NDArray,int,ValueTuple{double,double}?,bool,NDArray)"/>.
        /// </summary>
        private readonly struct BinEdgesResult
        {
            /// <summary>The computed bin edges (length = nbins + 1).</summary>
            public readonly NDArray Edges;
            /// <summary>True when the edges are equal-width (the fast <c>bincount</c> path applies).</summary>
            public readonly bool Uniform;
            /// <summary>Lower outer edge — valid only when <see cref="Uniform"/>.</summary>
            public readonly double First;
            /// <summary>Upper outer edge — valid only when <see cref="Uniform"/>.</summary>
            public readonly double Last;
            /// <summary>Number of equal-width bins — valid only when <see cref="Uniform"/>.</summary>
            public readonly int NBins;
            /// <summary>The compute/bin dtype B (Half/Single/Double) — valid only when <see cref="Uniform"/>.</summary>
            public readonly NPTypeCode BinDtype;

            public BinEdgesResult(NDArray edges, bool uniform, double first, double last, int nbins, NPTypeCode binDtype)
            {
                Edges = edges; Uniform = uniform; First = first; Last = last; NBins = nbins; BinDtype = binDtype;
            }
        }

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
            return HistogramBinEdgesImpl(a, bins, range, weights);
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
            return HistogramBinEdgesImpl(a, bins, range, weights);
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
            return HistogramBinEdgesImpl(a, bins, range, weights);
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
            return HistogramBinEdgesImpl(a, (NDArray)np.array(bins), range, weights);
        }

        /// <summary>
        ///     <c>histogram_bin_edges</c>: NumPy's <c>_ravel_and_check_weights</c> then <c>_get_bin_edges</c>. Text data or text
        ///     edges are answered inside those two steps, at the statement where NumPy's code first touches them (see
        ///     <see cref="GetBinEdges"/>): explicit edges never read the data, so a text <paramref name="a"/> with numeric
        ///     edges returns the edges.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="bins">An <see cref="int"/>, an estimator name, or an <see cref="NDArray"/> (0-d count or 1-D edges).</param>
        /// <param name="range">The optional outer edges.</param>
        /// <param name="weights">The optional weights (shape-checked only).</param>
        /// <returns>The bin edges.</returns>
        /// <exception cref="ValueError">See <see cref="GetBinEdges"/>; a weights shape that differs from <paramref name="a"/>'s.</exception>
        /// <exception cref="TypeError">See <see cref="GetBinEdges"/>.</exception>
        private static NDArray HistogramBinEdgesImpl(NDArray a, object bins, (double, double)? range, NDArray weights)
        {
            (NDArray raveled, NDArray w) = RavelAndCheckWeights(a, weights);
            return GetBinEdges(raveled, bins, range, w).Edges;
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
            return HistogramImpl(a, bins, range, density, weights);
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
            return HistogramImpl(a, bins, range, density, weights);
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
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramResult"/> is an <see cref="INDArrayCarrier"/>): the raveled input and weights, the per-sample bin indices, the bincount/cumulative scratch and the density quotients are reclaimed; only the histogram and its edges leave.</remarks>
        [NDScoped]
        public static HistogramResult histogram(NDArray a, NDArray bins, (double, double)? range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, bins, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return HistogramImpl(a, bins, range, density, weights);
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
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="HistogramResult"/> is an <see cref="INDArrayCarrier"/>): the raveled input and weights, the per-sample bin indices, the bincount/cumulative scratch and the density quotients are reclaimed; only the histogram and its edges leave.</remarks>
        [NDScoped]
        public static HistogramResult histogram(NDArray a, double[] bins, (double, double)? range = null, bool density = false, NDArray weights = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return HistogramImpl(a, (NDArray)np.array(bins), range, density, weights);
        }

        /// <summary>
        ///     <c>histogram</c>: the edges as <see cref="HistogramBinEdgesImpl"/> finds them, then NumPy's equal-width
        ///     (bincount) path or its cumulative (sort + searchsorted) path, then the optional density.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="bins">An <see cref="int"/>, an estimator name, or an <see cref="NDArray"/> (0-d count or 1-D edges).</param>
        /// <param name="range">The optional outer edges.</param>
        /// <param name="density">Whether to normalize to a density.</param>
        /// <param name="weights">The optional weights.</param>
        /// <returns>The counts (or summed weights, or density) and the edges.</returns>
        /// <exception cref="ValueError">See <see cref="GetBinEdges"/>; a weights shape that differs from <paramref name="a"/>'s.</exception>
        /// <exception cref="TypeError">See <see cref="GetBinEdges"/> and <see cref="HistogramTextAnswer"/>.</exception>
        /// <exception cref="StringComputationNotSupportedException">NumPy sorts or searches text to count (<see cref="HistogramTextAnswer"/>).</exception>
        private static HistogramResult HistogramImpl(NDArray a, object bins, (double, double)? range, bool density, NDArray weights)
        {
            (NDArray raveled, NDArray w) = RavelAndCheckWeights(a, weights);
            BinEdgesResult be = GetBinEdges(raveled, bins, range, w);
            NDArray binEdges = be.Edges;

            // Text data, edges or weights only ever reach NumPy's cumulative path (text data never gets equal-width edges,
            // text weights are never "simple"); its answers are decided by sizes and dtypes alone, or need text computation.
            if (Backends.Strings.NDStringUfuncs.IsString(raveled) || Backends.Strings.NDStringUfuncs.IsString(binEdges)
                || Backends.Strings.NDStringUfuncs.IsString(w))
                return HistogramTextAnswer(raveled, w, binEdges, density);

            // ntype: int64 counts when unweighted, else the weights' dtype (NumPy: hist.dtype == weights.dtype).
            NPTypeCode ntype = w is null ? NPTypeCode.Int64 : w.typecode;

            // The bincount fast path only applies when the weights can accumulate as float64/complex128.
            bool simpleWeights = w is null
                || can_cast(w.typecode, NPTypeCode.Double)
                || can_cast(w.typecode, NPTypeCode.Complex);

            NDArray n = (be.Uniform && simpleWeights)
                ? HistogramUniform(raveled, w, be, ntype)
                : HistogramCumulative(raveled, w, binEdges, ntype);

            if (density)
            {
                // db = float64 bin widths; result = n / db / n.sum() — NumPy's exact density formula. n stays
                // int/float/complex and promotes to float/complex. The raw histogram n, db and the edges are
                // bit-identical to NumPy; the ONLY residual is that np.sum(n)'s reduction ORDER can differ from
                // NumPy's pairwise summation by ≤ a few ULP (and complex division amplifies that in the imaginary
                // quotient when n.sum()'s imaginary part is small), so a density result may sit within a handful of
                // ULP of NumPy's. That is the library-wide np.sum reduction-order characteristic, not a histogram
                // divergence — matching it would mean reimplementing NumPy's pairwise sum, an np.sum concern.
                NDArray db = np.diff(binEdges).astype(NPTypeCode.Double);
                NDArray dens = n / db / np.sum(n);
                return new HistogramResult(dens, binEdges);
            }

            return new HistogramResult(n, binEdges);
        }

        /// <summary>
        ///     NumPy's cumulative histogram path when the data, the edges or the weights are text — the only path text
        ///     reaches. Every answer it can give without comparing text is given; the rest needs NumSharp's string
        ///     computation.
        /// </summary>
        /// <param name="raveled">The raveled data.</param>
        /// <param name="w">The raveled weights, or null.</param>
        /// <param name="binEdges">The edges <see cref="GetBinEdges"/> returned (text ones have at most one element: two or more
        ///     were compared there).</param>
        /// <param name="density">Whether to normalize to a density.</param>
        /// <returns>The counts (empty when there are at most one edge; zeros when there is no data) and the edges.</returns>
        /// <exception cref="TypeError">Text weights: <c>ufunc 'subtract' did not contain a loop …</c> when there is no data
        ///     (NumPy's <c>np.diff</c> of its text-typed running total), else NumPy's add.accumulate refusal (the weights'
        ///     <c>cumsum</c>); text edges with <paramref name="density"/>: the bin widths' <c>subtract</c> refusal.</exception>
        /// <exception cref="StringComputationNotSupportedException">There is data and at least two edges, with text data:
        ///     NumPy sorts the data and searches it for every edge, comparing text.</exception>
        /// <remarks>
        ///     NumPy's loop runs once per 65536-element block of the data: with no data it never runs, so the running total
        ///     stays <c>np.zeros(edges.shape, ntype)</c>. With at most one edge the counts are <c>np.diff</c> of at most one
        ///     element, empty whatever the comparisons found — so those answers need no text computation. The sort and
        ///     searches cannot raise; the first statement that can is the weights' <c>cumsum</c>, then the density's
        ///     <c>np.diff(bin_edges)</c>.
        /// </remarks>
        private static HistogramResult HistogramTextAnswer(NDArray raveled, NDArray w, NDArray binEdges, bool density)
        {
            long nb = System.Math.Max(binEdges.size - 1, 0);
            NDArray n;
            if (w is null)
            {
                if (raveled.size > 0 && binEdges.size > 1)
                    throw Backends.Strings.NDStringUfuncs.NextPhase("histogram", raveled);
                n = np.zeros(Shape.Vector(nb), NPTypeCode.Int64);
            }
            else if (raveled.size == 0)
            {
                // cum_n = np.zeros(bin_edges.shape, weights.dtype); n = np.diff(cum_n): subtract has no text loop.
                if (Backends.Strings.NDStringUfuncs.IsString(w))
                    throw Backends.Strings.NDStringUfuncs.NoLoop("subtract", w.dtype, w.dtype);
                n = np.zeros(Shape.Vector(nb), w.dtype);
            }
            else
            {
                // np.argsort / np.sort of the block cannot fail; `sw.cumsum()` is the first statement that can.
                if (Backends.Strings.NDStringUfuncs.IsString(w))
                    throw Backends.Strings.NDStringUfuncs.AddNotCompatible(w.dtype, "accumulate");
                if (binEdges.size > 1)
                    throw Backends.Strings.NDStringUfuncs.NextPhase("histogram", raveled);
                n = np.zeros(Shape.Vector(nb), w.dtype);
            }

            if (density)
            {
                // db = np.array(np.diff(bin_edges), float): text edges have no subtract loop.
                if (Backends.Strings.NDStringUfuncs.IsString(binEdges))
                    throw Backends.Strings.NDStringUfuncs.NoLoop("subtract", binEdges.dtype, binEdges.dtype);
                NDArray db = np.diff(binEdges).astype(NPTypeCode.Double);
                return new HistogramResult(n / db / np.sum(n), binEdges);
            }
            return new HistogramResult(n, binEdges);
        }

        // ---------------------------------------------------------------------
        //  Fast (equal-width) path — fused index kernel + bincount
        // ---------------------------------------------------------------------

        /// <summary>
        ///     The uniform (equal-width) histogram path: compute every sample's bin index with the fused
        ///     kernel, then accumulate with <see cref="bincount(NDArray,NDArray,int)"/>. Out-of-range samples
        ///     ride a sentinel bin that is sliced off — identical to NumPy's <c>keep</c>-mask counts, and the
        ///     weighted accumulation is blocked at <see cref="HistogramBlock"/> for bit-exact float sums.
        /// </summary>
        private static unsafe NDArray HistogramUniform(NDArray raveled, NDArray w, BinEdgesResult be, NPTypeCode ntype)
        {
            long len = raveled.size;
            int nbins = be.NBins;

            // Empty input: no samples, all-zero histogram of the right dtype.
            if (len == 0)
                return np.zeros(Shape.Vector(nbins), ntype);

            // Cast samples and edges to the compute dtype B, contiguous, matching NumPy's
            // `tmp_a.astype(bin_edges.dtype)` — the index arithmetic then runs bit-for-bit in B.
            NDArray aB = EnsureContiguousDtype(raveled, be.BinDtype);
            NDArray edgesB = be.Edges.typecode == be.BinDtype ? be.Edges : be.Edges.astype(be.BinDtype);
            edgesB = EnsureContiguousDtype(edgesB, be.BinDtype);

            if (w is null)
            {
                // Unweighted: ONE fused pass — compute each index and increment its count directly.
                // No int64 indices array, no separate bincount. Integer counts are order-independent.
                NDArray counts = np.zeros(Shape.Vector(nbins), NPTypeCode.Int64);
                var countKernel = DirectILKernelGenerator.GetHistogramCountKernel(be.BinDtype);
                countKernel((void*)aB.Address, len, (void*)edgesB.Address, be.First, be.Last, nbins, (long*)counts.Address);
                return counts;
            }

            // Weighted: fused per-block scatter into a float64 running total, then cast to the weight dtype
            // and fold into the result — replicating NumPy's per-block `n += bincount(block, w).astype(ntype)`.
            // The float64-accumulate-per-block-then-cast AND the 65536 block boundary are both load-bearing
            // for bit-exactness (a single-pass sum diverges in the last ULP; verified against NumPy 2.4.2).
            bool complexWeights = ntype == NPTypeCode.Complex;
            NDArray wReal = EnsureContiguousDtype(complexWeights ? np.real(w) : w, NPTypeCode.Double);
            NDArray wImag = complexWeights ? EnsureContiguousDtype(np.imag(w), NPTypeCode.Double) : null;

            var wkernel = DirectILKernelGenerator.GetHistogramWeightedKernel(be.BinDtype);
            int elemB = DirectILKernelGenerator.GetTypeSize(be.BinDtype);
            byte* aBase = (byte*)aB.Address;
            double* wrBase = (double*)wReal.Address;
            double* wiBase = complexWeights ? (double*)wImag.Address : null;

            NDArray n = np.zeros(Shape.Vector(nbins), ntype);
            for (long start = 0; start < len; start += HistogramBlock)
            {
                long blockLen = System.Math.Min(HistogramBlock, len - start);
                void* sBlk = aBase + start * elemB;

                // Real part (or the whole weight, for real dtypes): one fused float64 scatter over the block.
                NDArray tempR = np.zeros(Shape.Vector(nbins), NPTypeCode.Double);
                wkernel(sBlk, wrBase + start, blockLen, (void*)edgesB.Address, be.First, be.Last, nbins, (double*)tempR.Address);

                if (complexWeights)
                {
                    // Imaginary part accumulates through a second scatter — NumPy's two-bincount split.
                    NDArray tempI = np.zeros(Shape.Vector(nbins), NPTypeCode.Double);
                    wkernel(sBlk, wiBase + start, blockLen, (void*)edgesB.Address, be.First, be.Last, nbins, (double*)tempI.Address);
                    n = n + MakeComplex(tempR, tempI);
                }
                else
                {
                    n = n + tempR.astype(ntype);
                }
            }
            return n;
        }

        // ---------------------------------------------------------------------
        //  Slow (arbitrary-edge) path — sort + inclusive searchsorted, cumulative
        // ---------------------------------------------------------------------

        /// <summary>
        ///     The general histogram path for non-uniform edges (or weights that cannot bincount): a port of
        ///     NumPy's cumulative-histogram algorithm. Unweighted uses sorted inclusive searchsorted counts;
        ///     weighted uses the sorted cumulative-weight differences, both blocked at <see cref="HistogramBlock"/>.
        /// </summary>
        private static NDArray HistogramCumulative(NDArray raveled, NDArray w, NDArray binEdges, NPTypeCode ntype)
        {
            long len = raveled.size;

            if (w is null)
            {
                // Unweighted non-uniform: bin each sample directly, NO sample sort. Integer counts are
                // identical to NumPy's cumulative sort path (a value is in exactly one bin), but one fused
                // binary-search-and-count pass replaces NumPy's per-block O(n log n) sort. The compare dtype
                // matches searchsorted's (result_type of samples and edges), so boundary decisions match too.
                int nb = (int)(binEdges.size - 1);
                NPTypeCode cmp = promote_types(raveled.typecode, binEdges.typecode);
                var searchKernel = DirectILKernelGenerator.GetHistogramSearchCountKernel(cmp);
                if (searchKernel != null)
                {
                    NDArray aC = EnsureContiguousDtype(raveled, cmp);
                    NDArray eC = EnsureContiguousDtype(binEdges, cmp);
                    NDArray counts = np.zeros(Shape.Vector(nb), NPTypeCode.Int64);
                    unsafe { searchKernel((void*)aC.Address, raveled.size, (void*)eC.Address, nb, (long*)counts.Address); }
                    return counts;
                }

                // Complex (no ordering): compose via searchsorted + bincount (rare). searchsorted('right')
                // yields [0, nbins+1]; a value equal to the last edge is pulled into the last bin, then
                // bincount slices off both outlier bins.
                NDArray idx = np.searchsorted(binEdges, raveled, "right");
                NDArray onEdge = raveled == binEdges["-1"];
                idx = idx - onEdge.astype(NPTypeCode.Int64);
                return bincount(idx, minlength: nb + 2)[$"1:{nb + 1}"].copy();
            }

            NDArray cumN = np.zeros(Shape.Vector(binEdges.size), ntype);
            {
                NDArray zero = np.zeros(Shape.Vector(1), ntype);
                for (long start = 0; start < len; start += HistogramBlock)
                {
                    long blockLen = System.Math.Min(HistogramBlock, len - start);
                    string slice = $"{start}:{start + blockLen}";
                    NDArray tmpA = raveled[slice];
                    NDArray tmpW = w[slice];
                    NDArray sortIdx = np.argsort(tmpA);
                    NDArray sa = tmpA[sortIdx];
                    NDArray sw = tmpW[sortIdx];
                    // cw = [0, cumsum(sorted weights)]; cum_n += cw[bin_index].
                    NDArray cw = np.concatenate((zero, np.cumsum(sw)));
                    NDArray binIndex = SearchSortedInclusive(sa, binEdges);
                    cumN = cumN + np.take(cw, binIndex);
                }
            }

            return np.diff(cumN);
        }

        /// <summary>
        ///     NumPy's <c>_search_sorted_inclusive</c>: <c>searchsorted(a, v[:-1], 'left')</c> concatenated with
        ///     <c>searchsorted(a, v[-1:], 'right')</c>, so the last bin edge is inclusive (a value equal to the
        ///     final edge lands in the last bin, not past it).
        /// </summary>
        private static NDArray SearchSortedInclusive(NDArray a, NDArray v)
        {
            long m = v.size;
            NDArray left = np.searchsorted(a, v[$"0:{m - 1}"], "left");
            NDArray right = np.searchsorted(a, v[$"{m - 1}:{m}"], "right");
            return np.concatenate((left, right));
        }

        // ---------------------------------------------------------------------
        //  _get_bin_edges  (port of numpy/lib/_histograms_impl.py)
        // ---------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>_get_bin_edges</c>: the edges (and, for equal-width bins, the outer edges and count) for an
        ///     estimator name, a bin count, or explicit 1-D edges — in NumPy's statement order, which decides every error.
        /// </summary>
        /// <param name="a">The raveled data.</param>
        /// <param name="bins">An estimator name, an <see cref="int"/>, or an <see cref="NDArray"/> (0-d count, 1-D edges).</param>
        /// <param name="range">The optional outer edges.</param>
        /// <param name="weights">The raveled weights, or null (an estimator refuses them).</param>
        /// <returns>The edges, plus the equal-width parameters when computed.</returns>
        /// <exception cref="ValueError">An unknown estimator; a count below 1; edges that are not 1-D or decrease; an invalid
        ///     range; too many bins for the range.</exception>
        /// <exception cref="TypeError">Weights with an estimator; a non-integer scalar <paramref name="bins"/>. Text data:
        ///     its <c>a.min()</c> (<c>minimum</c> has no text loop) when no range is given and there is data; with a range and an
        ///     estimator, the in-range mask's <c>greater_equal</c> against the Python float edge; otherwise the edges' dtype,
        ///     <c>np.result_type(first_edge, last_edge, a)</c>, which has no common DType for text and Python numbers
        ///     (<see cref="DTypePromotionError"/>).</exception>
        /// <exception cref="StringComputationNotSupportedException">Text edges of two or more elements: NumPy's monotonicity
        ///     check compares them.</exception>
        /// <remarks>
        ///     Explicit edges never read the data, so text data with explicit edges gets its edges back here; the histogram's
        ///     counting step decides the rest (<see cref="HistogramTextAnswer"/>).
        /// </remarks>
        private static BinEdgesResult GetBinEdges(NDArray a, object bins, (double, double)? range, NDArray weights)
        {
            int? nEqualBins = null;
            NDArray binEdges = null;
            double first = 0, last = 0;
            bool textData = Backends.Strings.NDStringUfuncs.IsString(a);

            if (bins is string binName)
            {
                if (!IsKnownEstimator(binName))
                    throw new ValueError($"'{binName}' is not a valid estimator for `bins`");
                if (weights is not null)
                    throw new TypeError("Automated estimation of the number of bins is not supported for weighted data");

                (first, last) = GetOuterEdges(a, range);

                // Truncate the range if needed — the estimator sees only in-range data.
                NDArray x = a;
                if (range.HasValue)
                {
                    // `keep = (a >= first_edge)` compares the data with a Python float: no loop for text, even with no data.
                    if (textData)
                        throw new TypeError(
                            $"ufunc 'greater_equal' did not contain a loop with signature matching types ({a.dtype.Meta}, " +
                            $"{DTypeRegistry.PyFloat}) -> None");
                    NDArray keep = (a >= NDArray.Scalar(first)) & (a <= NDArray.Scalar(last));
                    if (!AllTrue(keep))
                        x = a[keep];
                }

                if (x.size == 0)
                    nEqualBins = 1;
                else
                {
                    double width = EstimateBinWidth(binName, x, first, last, range);
                    if (width > 0)
                    {
                        // Integer data never gets a sub-unit bin width.
                        if (a.typecode.IsInteger() && width < 1) width = 1;
                        double delta = last - first;
                        nEqualBins = (int)System.Math.Ceiling(delta / width);
                    }
                    else
                    {
                        nEqualBins = 1; // e.g. FD when the IQR is zero
                    }
                }
            }
            else if (bins is NDArray nd && nd.ndim >= 1)
            {
                if (nd.ndim != 1)
                    throw new ValueError("`bins` must be 1d, when an array");
                binEdges = nd;
                // `np.any(bin_edges[:-1] > bin_edges[1:])`: with fewer than two edges there is nothing to compare, so text
                // edges pass as they are; two or more text edges are compared by the string greater kernel.
                if (Backends.Strings.NDStringUfuncs.IsString(binEdges))
                {
                    if (binEdges.size > 1 && TextEdgesDecrease(binEdges))
                        throw new ValueError("`bins` must increase monotonically, when an array");
                }
                else if (AnyDecreasing(binEdges))
                    throw new ValueError("`bins` must increase monotonically, when an array");
            }
            else
            {
                // Scalar bin count: a C# int, or a 0-d integer array (operator.index).
                int count = AsBinCount(bins);
                if (count < 1)
                    throw new ValueError("`bins` must be positive, when an integer");
                nEqualBins = count;
                (first, last) = GetOuterEdges(a, range);
            }

            if (nEqualBins.HasValue)
            {
                // `bin_type = np.result_type(first_edge, last_edge, a)`: text data has no common DType with the weak
                // Python edges (ints 0 / 1 for empty data, floats from a range), so NumPy's promotion error is the answer.
                if (textData)
                    ThrowTextEdgeType(a, range.HasValue);

                // The edges' RESULT dtype (gh-10322): float16 for float16 input, float32 for float32, float64 for
                // everything else — because the input array is in NumPy's result_type(first, last, a), so a weak
                // endpoint never widens a strong float array. This is also the dtype the uniform fast-path kernel
                // casts samples to (NumPy's `tmp_a.astype(bin_edges.dtype)`).
                NPTypeCode resultType = FloatPromoteBinType(a.typecode);
                // The COMPUTE dtype, however, is float64 whenever the endpoints are weak — a supplied range (Python
                // floats) or an empty input (the 0/1 int defaults) — since those cannot widen the linspace past
                // float64; only strong array-scalar endpoints (non-empty, no range → a.min()/a.max()) keep the
                // arithmetic in the narrow float width. Getting this wrong left a ~1 ULP error on ranged/empty
                // float32/float16 edges (NumPy computes them in float64, then casts to the result width).
                bool weakEndpoints = range.HasValue || a.size == 0;
                NPTypeCode computeType = weakEndpoints ? NPTypeCode.Double : resultType;
                int nb = nEqualBins.Value;
                NDArray edges = HistogramLinspace(first, last, nb + 1, computeType, resultType);
                if (AnyNonIncreasing(edges))
                    throw new ValueError($"Too many bins for data range. Cannot create {nb} finite-sized bins.");
                return new BinEdgesResult(edges, true, first, last, nb, resultType);
            }

            return new BinEdgesResult(binEdges, false, 0, 0, 0, NPTypeCode.Double);
        }

        /// <summary>
        ///     NumPy's <c>_get_outer_edges</c>: resolve the outer bin edges from <paramref name="range"/> or from
        ///     the data's min/max, expanding a degenerate zero-width range by ±0.5 and rejecting non-finite bounds.
        /// </summary>
        /// <param name="a">The data (text data reaches only the range checks or the empty default without an error).</param>
        /// <param name="range">The optional outer edges.</param>
        /// <returns>The outer edges.</returns>
        /// <exception cref="ValueError">A range whose lower edge exceeds its upper one, or a non-finite range or data range.</exception>
        /// <exception cref="TypeError">Text data with no range and at least one element: <c>a.min()</c> has no text loop
        ///     (<c>ufunc 'minimum' did not contain a loop with signature matching types (dtype('&lt;U2'), dtype('&lt;U2')) -&gt;
        ///     None</c>).</exception>
        private static (double first, double last) GetOuterEdges(NDArray a, (double, double)? range)
        {
            double first, last;
            if (range.HasValue)
            {
                (first, last) = range.Value;
                if (first > last)
                    throw new ValueError("max must be larger than min in range parameter.");
                if (!(IsFinite(first) && IsFinite(last)))
                    throw new ValueError($"supplied range of [{FormatEdge(first)}, {FormatEdge(last)}] is not finite");
            }
            else if (a.size == 0)
            {
                first = 0; last = 1; // can't infer a range from no data
            }
            else if (Backends.Strings.NDStringUfuncs.IsString(a))
            {
                // `a.min()` is minimum.reduce, which has no text loop.
                throw Backends.Strings.NDStringUfuncs.NoLoop("minimum", a.dtype, a.dtype);
            }
            else
            {
                first = ScalarDouble(np.amin(a));
                last = ScalarDouble(np.amax(a));
                if (!(IsFinite(first) && IsFinite(last)))
                    throw new ValueError($"autodetected range of [{FormatEdge(first)}, {FormatEdge(last)}] is not finite");
            }

            if (first == last) { first -= 0.5; last += 0.5; }
            return (first, last);
        }

        // ---------------------------------------------------------------------
        //  Bin-width estimators (port of the _hist_bin_* selectors)
        // ---------------------------------------------------------------------

        private static bool IsKnownEstimator(string name) => name switch
        {
            "stone" or "auto" or "doane" or "fd" or "rice" or "scott" or "sqrt" or "sturges" => true,
            _ => false
        };

        /// <summary>
        ///     Dispatch to the requested bin-width estimator. <paramref name="x"/> is the range-trimmed,
        ///     non-empty data; <paramref name="first"/>/<paramref name="last"/> are the outer edges (only
        ///     <c>stone</c> uses the range).
        /// </summary>
        private static double EstimateBinWidth(string name, NDArray x, double first, double last, (double, double)? range)
        {
            long n = x.size;
            switch (name)
            {
                case "sqrt":
                    return Ptp(x) / System.Math.Sqrt(n);
                case "sturges":
                    return Ptp(x) / (System.Math.Log2(n) + 1.0);
                case "rice":
                    return Ptp(x) / (2.0 * System.Math.Cbrt(n));
                case "scott":
                    return System.Math.Cbrt(24.0 * System.Math.Sqrt(System.Math.PI) / n) * ScalarDouble(np.std(x));
                case "fd":
                    return BinWidthFd(x, n);
                case "doane":
                    return BinWidthDoane(x, n);
                case "auto":
                {
                    double fd = BinWidthFd(x, n);
                    double sturges = Ptp(x) / (System.Math.Log2(n) + 1.0);
                    double sqrt = Ptp(x) / System.Math.Sqrt(n);
                    double fdCorrected = System.Math.Max(fd, sqrt / 2);
                    return System.Math.Min(fdCorrected, sturges);
                }
                case "stone":
                    return BinWidthStone(x, first, last, range);
                default:
                    throw new ValueError($"'{name}' is not a valid estimator for `bins`");
            }
        }

        /// <summary>Freedman–Diaconis: <c>2 · IQR · n^(-1/3)</c> (robust to outliers; 0 when the IQR is 0).</summary>
        private static double BinWidthFd(NDArray x, long n)
        {
            NDArray q = np.percentile(x, new double[] { 75, 25 });
            double iqr = ScalarDouble(q[0]) - ScalarDouble(q[1]);
            return 2.0 * iqr * System.Math.Pow(n, -1.0 / 3.0);
        }

        /// <summary>Doane's improved Sturges rule; accounts for skew. Returns 0 when std is 0 or n ≤ 2.</summary>
        private static double BinWidthDoane(NDArray x, long n)
        {
            if (n > 2)
            {
                double sg1 = System.Math.Sqrt(6.0 * (n - 2) / ((n + 1.0) * (n + 3)));
                double sigma = ScalarDouble(np.std(x));
                if (sigma > 0.0)
                {
                    double mean = ScalarDouble(np.mean(x));
                    // g1 = mean(((x - mean) / sigma)^3).
                    NDArray temp = (x.astype(NPTypeCode.Double) - NDArray.Scalar(mean)) / NDArray.Scalar(sigma);
                    temp = np.power(temp, NDArray.Scalar(3.0));
                    double g1 = ScalarDouble(np.mean(temp));
                    return Ptp(x) / (1.0 + System.Math.Log2(n) + System.Math.Log2(1.0 + System.Math.Abs(g1) / sg1));
                }
            }
            return 0.0;
        }

        /// <summary>
        ///     Stone's rule: choose the bin count that minimizes the leave-one-out cross-validation estimate of
        ///     the integrated squared error. Evaluates the histogram for each candidate count (as NumPy does).
        /// </summary>
        private static double BinWidthStone(NDArray x, double first, double last, (double, double)? range)
        {
            long n = x.size;
            double ptp = Ptp(x);
            if (n <= 1 || ptp == 0) return 0;

            (double, double) rng = range ?? (first, last);
            int upperBound = System.Math.Max(100, (int)System.Math.Sqrt(n));

            double bestJ = double.PositiveInfinity;
            int bestNbins = 1;
            for (int nbins = 1; nbins <= upperBound; nbins++)
            {
                double hh = ptp / nbins;
                NDArray counts = histogram(x, nbins, rng).hist; // pk = counts / n
                double sumSq = ScalarDouble(np.sum(np.power(counts.astype(NPTypeCode.Double), NDArray.Scalar(2.0))));
                double pkDot = sumSq / ((double)n * n); // p_k · p_k
                double j = (2 - (n + 1) * pkDot) / hh;
                if (j < bestJ) { bestJ = j; bestNbins = nbins; }
            }
            return ptp / bestNbins;
        }

        // ---------------------------------------------------------------------
        //  Small helpers
        // ---------------------------------------------------------------------

        /// <summary>Peak-to-peak (max − min) of <paramref name="x"/> computed in double (no signed-int overflow).</summary>
        private static double Ptp(NDArray x) => ScalarDouble(np.amax(x)) - ScalarDouble(np.amin(x));

        /// <summary>
        ///     The float width the equal-width bin edges (and the index arithmetic) use, matching NumPy's
        ///     <c>result_type(first_edge, last_edge, a)</c> with weak-float endpoints: float16/float32 keep their
        ///     width, everything else (int/bool/char/float64/decimal/complex) resolves to float64.
        /// </summary>
        private static NPTypeCode FloatPromoteBinType(NPTypeCode inputType) => inputType switch
        {
            NPTypeCode.Half => NPTypeCode.Half,
            NPTypeCode.Single => NPTypeCode.Single,
            _ => NPTypeCode.Double
        };

        /// <summary>
        ///     Equal-width bin edges reproducing NumPy's <c>linspace(first, last, num, dtype=resultType)</c>
        ///     BIT-FOR-BIT, honouring the subtle split between the dtype the arithmetic RUNS in
        ///     (<paramref name="computeType"/>) and the dtype the edges END UP in (<paramref name="resultType"/>).
        /// </summary>
        /// <remarks>
        ///     NumPy's <c>linspace</c> computes internally in <c>result_type(first, last, num)</c> and only then
        ///     casts to the requested dtype. The two dtypes DIVERGE exactly when the endpoints are "weak" — a
        ///     supplied <c>range</c> (Python floats) or an empty input (the 0/1 defaults, Python ints) — because a
        ///     weak scalar cannot widen the result past float64, so the arithmetic runs in float64 and is then cast
        ///     down to a float32/float16 result. They COINCIDE when the endpoints are "strong" array scalars
        ///     (non-empty, no range → <c>a.min()/a.max()</c>), where the whole computation stays in the array's own
        ///     float width. Passing them separately is what makes both the strong path (in-width, e.g. a float32
        ///     array's own edges) and the weak path (float64 arithmetic → float32/float16 cast) bit-exact; conflating
        ///     them left a ~1 ULP error on ranged/empty float32/float16 histograms.
        /// </remarks>
        /// <param name="first">Lower outer edge.</param>
        /// <param name="last">Upper outer edge.</param>
        /// <param name="num">Number of edge points (<c>nbins + 1</c>).</param>
        /// <param name="computeType">The dtype the linspace arithmetic runs in (Half/Single/Double).</param>
        /// <param name="resultType">The dtype the returned edges must have (Half/Single/Double).</param>
        /// <returns>The <paramref name="num"/> edges in dtype <paramref name="resultType"/>.</returns>
        private static unsafe NDArray HistogramLinspace(double first, double last, int num, NPTypeCode computeType, NPTypeCode resultType)
        {
            // Compute in the arithmetic dtype, then cast to the result dtype (a no-op when they coincide).
            NDArray computed = ComputeLinspaceInDtype(first, last, num, computeType);
            return computeType == resultType ? computed : computed.astype(resultType);
        }

        /// <summary>
        ///     The single-dtype linspace core: float64 defers to <see cref="linspace"/> (already double-internal),
        ///     while float32/float16 run the arithmetic in their OWN width (widen-op-narrow for float16, matching
        ///     NumPy's npy_half ufuncs) so a strong-endpoint narrow-float histogram is bit-exact rather than the
        ///     ~1 ULP off a double-internal linspace would give.
        /// </summary>
        /// <param name="first">Lower outer edge.</param>
        /// <param name="last">Upper outer edge.</param>
        /// <param name="num">Number of edge points (<c>nbins + 1</c>).</param>
        /// <param name="dtype">The dtype the arithmetic runs in AND the result has (Half/Single/Double).</param>
        /// <returns>The <paramref name="num"/> edges in dtype <paramref name="dtype"/>.</returns>
        private static unsafe NDArray ComputeLinspaceInDtype(double first, double last, int num, NPTypeCode dtype)
        {
            if (dtype == NPTypeCode.Double)
                return np.linspace(first, last, num, endpoint: true, dtype: NPTypeCode.Double);

            NPTypeCode binType = dtype;
            var edges = new NDArray(binType, Shape.Vector(num), false);
            int div = num - 1; // histogram always has endpoint=true and num = nbins+1 >= 2, so div >= 1

            if (binType == NPTypeCode.Single)
            {
                float* p = (float*)edges.Address;
                float start = (float)first, stop = (float)last;
                float delta = stop - start;
                float step = delta / div;
                bool stepZero = step == 0f;
                for (int i = 0; i < num; i++)
                {
                    // NumPy order: y = arange*step (or arange/div*delta if step underflowed); y += start; y[-1] = stop.
                    if (i == num - 1) { p[i] = stop; continue; }
                    p[i] = stepZero ? ((float)i / div) * delta + start : (float)i * step + start;
                }
            }
            else // Half — each op widens to float32 and narrows, matching NumPy's npy_half linspace.
            {
                Half* p = (Half*)edges.Address;
                Half start = (Half)first, stop = (Half)last;
                Half delta = (Half)((float)stop - (float)start);
                Half step = (Half)((float)delta / div);
                bool stepZero = (float)step == 0f;
                for (int i = 0; i < num; i++)
                {
                    if (i == num - 1) { p[i] = stop; continue; }
                    Half yi = (Half)(float)i;
                    Half scaled = stepZero
                        ? (Half)((float)(Half)((float)yi / div) * (float)delta)
                        : (Half)((float)yi * (float)step);
                    p[i] = (Half)((float)scaled + (float)start);
                }
            }
            return edges;
        }

        /// <summary>Read a size-1/0-d array as a double, converting from any numeric dtype.</summary>
        private static double ScalarDouble(NDArray nd)
            => nd.astype(NPTypeCode.Double).GetAtIndex<double>(0);

        /// <summary>
        ///     Ensure <paramref name="a"/> is a C-contiguous array of dtype <paramref name="tc"/> whose
        ///     <b>base address is its logical element 0</b>, so a kernel may read <c>(void*)result.Address</c>
        ///     directly. Copies only when needed.
        /// </summary>
        /// <remarks>
        ///     The <c>Shape.offset == 0</c> guard is load-bearing, not redundant with <see cref="Shape.IsContiguous"/>:
        ///     a size-1 array is <em>trivially</em> contiguous whatever its stride/offset, and a strided view such as
        ///     <c>np.imag(complexWeights)</c> (stride 2, offset 1) keeps the buffer's BASE address rather than
        ///     re-seating it (only a simple contiguous SLICE re-seats the address and zeroes the offset). Without the
        ///     guard, a single complex weight's imaginary part was read from offset 0 — the REAL part — so
        ///     <c>np.histogram([x], bins, weights=[re+im·j])</c> returned <c>&lt;re, re&gt;</c> instead of
        ///     <c>&lt;re, im&gt;</c>. Requiring offset 0 forces the copy that folds the offset into a fresh buffer.
        /// </remarks>
        /// <param name="a">The array to normalize.</param>
        /// <param name="tc">The dtype the result must have.</param>
        /// <returns><paramref name="a"/> itself when already contiguous-from-zero in dtype <paramref name="tc"/>; otherwise a fresh copy.</returns>
        private static NDArray EnsureContiguousDtype(NDArray a, NPTypeCode tc)
        {
            NDArray typed = a.typecode == tc ? a : a.astype(tc);
            return (typed.Shape.IsContiguous && typed.Shape.offset == 0) ? typed : typed.copy();
        }

        /// <summary>
        ///     NumPy's <c>operator.index(bins)</c> for a scalar <c>bins</c>: a C# int, or a 0-d INTEGER array (a bool, float,
        ///     complex or text 0-d array is not an index — NumPy 2 refuses <c>operator.index(np.array(True))</c> too).
        /// </summary>
        /// <param name="bins">The scalar bin spec.</param>
        /// <returns>The bin count.</returns>
        /// <exception cref="TypeError">Not an integer: <c>`bins` must be an integer, a string, or an array</c>.</exception>
        private static int AsBinCount(object bins)
        {
            if (bins is int i) return i;
            if (bins is NDArray nd && nd.ndim == 0 && nd.typecode.IsInteger())
                return (int)Convert.ToInt64(nd.GetAtIndex(0));
            throw new TypeError("`bins` must be an integer, a string, or an array");
        }

        /// <summary>
        ///     NumPy's <c>np.result_type(first_edge, last_edge, a)</c> for text data: the outer edges are weak Python numbers
        ///     (the ints <c>0</c> / <c>1</c> NumPy defaults to for empty data, or the floats of a range), and no common DType
        ///     exists between them and text, so this always raises NumPy's promotion error.
        /// </summary>
        /// <param name="a">The text data.</param>
        /// <param name="fromRange">True when the edges came from a range (Python floats), false for the empty-data defaults.</param>
        /// <exception cref="DTypePromotionError">Always: <c>The DType &lt;class 'numpy.dtypes.StrDType'&gt; could not be promoted
        ///     by &lt;class 'numpy.dtypes._PyLongDType'&gt;. …</c> (or <c>_PyFloatDType</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">Defensive: a promotion that succeeded (none does today).</exception>
        private static void ThrowTextEdgeType(NDArray a, bool fromRange)
        {
            // Boxed C# integer / double literals are the weak Python ints / floats of DTypePromotion.ResultType.
            if (fromRange)
                DTypePromotion.ResultType(0.0, 1.0, a.dtype);
            else
                DTypePromotion.ResultType(0L, 1L, a.dtype);
            throw Backends.Strings.NDStringUfuncs.NextPhase("histogram", a);
        }

        /// <summary>True when every element of the boolean mask is true (NumPy's <c>logical_and.reduce</c>).</summary>
        private static bool AllTrue(NDArray mask) => np.all(mask);

        /// <summary>
        ///     NumPy's <c>np.any(bin_edges[:-1] &gt; bin_edges[1:])</c> over 1-D TEXT edges (two or more): the string
        ///     <c>greater</c> comparison of each edge with the next (unsigned code units, zero-padded), then <c>any</c>.
        ///     Shared by <c>np.histogram</c> and <c>np.histogramdd</c>, whose monotonic checks are this statement.
        /// </summary>
        /// <param name="edges">The 1-D string edges (at least two).</param>
        /// <returns>True when some edge is greater than the one after it (the edges do not increase monotonically).</returns>
        internal static bool TextEdgesDecrease(NDArray edges)
        {
            using var lower = edges[":-1"];
            using var upper = edges["1:"];
            using var greater = lower > upper;
            return np.any(greater);
        }

        /// <summary>True when any adjacent pair strictly decreases (rejects non-monotonic explicit edges; equal is allowed).</summary>
        private static bool AnyDecreasing(NDArray edges)
        {
            long m = edges.size;
            if (m < 2) return false;
            return np.any(edges[$"0:{m - 1}"] > edges[$"1:{m}"]);
        }

        /// <summary>True when any adjacent pair is non-increasing (edges collapsed by too many bins over a tiny range).</summary>
        private static bool AnyNonIncreasing(NDArray edges)
        {
            long m = edges.size;
            if (m < 2) return false;
            return np.any(edges[$"0:{m - 1}"] >= edges[$"1:{m}"]);
        }

        /// <summary>Build a complex array from separate real and imaginary parts (used for complex weights).</summary>
        private static NDArray MakeComplex(NDArray re, NDArray im)
        {
            var result = new NDArray(NPTypeCode.Complex, re.Shape, fillZeros: false);
            long n = re.size;
            for (long i = 0; i < n; i++)
                result.SetAtIndex(new System.Numerics.Complex(re.GetAtIndex<double>(i), im.GetAtIndex<double>(i)), i);
            return result;
        }

        private static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

        /// <summary>
        ///     Render an edge value the way NumPy's f-string prints it inside the "supplied/autodetected range
        ///     of [.,.] is not finite" messages — i.e. Python's <c>str(float)</c> / <c>str(np.float64)</c>, NOT
        ///     .NET's default <see cref="double.ToString()"/>. The difference is load-bearing for verbatim error
        ///     parity: Python appends a trailing <c>.0</c> to whole numbers (<c>1.0</c>, <c>100.0</c>, <c>0.0</c>)
        ///     and switches to <c>e±NN</c> for large/small magnitudes, where .NET's default drops the <c>.0</c>
        ///     (<c>1.0</c> → "1"), so a non-finite range with a whole-number bound diverged from NumPy.
        /// </summary>
        /// <param name="v">The edge value to render.</param>
        /// <returns>The NumPy/Python scalar-repr string for <paramref name="v"/>.</returns>
        private static string FormatEdge(double v)
        {
            // Non-finite tokens are spelled the same by NumPy and are cheap to emit without allocating.
            if (double.IsNaN(v)) return "nan";
            if (double.IsPositiveInfinity(v)) return "inf";
            if (double.IsNegativeInfinity(v)) return "-inf";
            // A finite bound goes through NumSharp's 0-d scalar str, which is a byte-exact port of NumPy's
            // float repr (verified to equal Python's str(float) across whole/decimal/scientific magnitudes).
            using NDArray scalar = NDArray.Scalar(v);
            return scalar.ToString(false);
        }

        /// <summary>
        ///     NumPy's <c>_ravel_and_check_weights</c>: flatten <paramref name="a"/> (bool → uint8 for
        ///     subtractability) and, if given, verify <paramref name="weights"/> matches <paramref name="a"/>'s
        ///     shape before flattening it too.
        /// </summary>
        private static (NDArray a, NDArray weights) RavelAndCheckWeights(NDArray a, NDArray weights)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));

            // bool is not "subtractable" for the bin arithmetic; NumPy converts it to uint8.
            NDArray arr = a.typecode == NPTypeCode.Boolean ? a.astype(NPTypeCode.Byte) : a;

            NDArray w = null;
            if (weights is not null)
            {
                if (!ShapeEquals(weights, a))
                    throw new ValueError("weights should have the same shape as a.");
                w = np.ravel(weights);
            }
            return (np.ravel(arr), w);
        }

        /// <summary>True when two arrays have identical shapes (dimension-by-dimension).</summary>
        private static bool ShapeEquals(NDArray x, NDArray y)
        {
            var xs = x.Shape.dimensions; var ys = y.Shape.dimensions;
            if (xs.Length != ys.Length) return false;
            for (int i = 0; i < xs.Length; i++)
                if (xs[i] != ys[i]) return false;
            return true;
        }
    }
}
