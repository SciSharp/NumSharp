using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;
// The moved statements call the public np.* functions they called inside np (np.histogram for Stone's estimator,
// np.bincount, np.can_cast, np.promote_types, ...); importing np's statics keeps them verbatim.
using static NumSharp.np;

namespace NumSharp.Statistics
{
    /// <summary>
    ///     The bodies of NumPy's <c>histogram</c> and <c>histogram_bin_edges</c> (<c>numpy/lib/_histograms_impl.py</c>):
    ///     <c>_ravel_and_check_weights</c>, <c>_get_bin_edges</c> with its outer edges and bin-width estimators, the
    ///     equal-width (bincount) and cumulative (sort + searchsorted) counting paths, the text routes and the density.
    ///     The public <c>np.histogram</c> / <c>np.histogram_bin_edges</c> overloads only open the matrix boundary's plain
    ///     region, convert their C# <c>bins</c> form and call <see cref="Histogram"/> / <see cref="HistogramBinEdges"/>.
    /// </summary>
    /// <remarks>
    ///     Moved from <c>Statistics/np.histogram.cs</c> statement for statement by the np-layer dtype plan's P2 (an operation
    ///     class, decision D1: NumPy writes these functions in Python). <c>np.histogramdd</c> shares the outer edges, the
    ///     edge linspace, the monotonic checks and the contiguous cast (<see cref="GetOuterEdges"/>,
    ///     <see cref="HistogramLinspace"/>, <see cref="AnyDecreasing"/>, <see cref="TextEdgesDecrease"/>,
    ///     <see cref="EnsureContiguousDtype"/>, <see cref="FloatPromoteBinType"/>).
    /// </remarks>
    internal static partial class HistogramEngine
    {
        /// <summary>
        ///     NumPy's <c>BLOCK</c> of <c>histogram</c> (65536 samples): the counting paths walk the data in blocks of this
        ///     many samples, and the weighted ones fold each block's sum into the running total.
        /// </summary>
        /// <remarks>
        ///     Not just a memory lever: weighted float sums are accumulated per block and only then folded into the running
        ///     total, so this block boundary is what keeps weighted histograms BIT-IDENTICAL to NumPy (a single-pass sum
        ///     diverges in the last ULP, verified against 2.4.2). Changing it changes answers.
        /// </remarks>
        private const int HistogramBlock = 65536;

        /// <summary>
        ///     The bin edges of the histogram plus (for equal-width bins) the parameters of the
        ///     optimized uniform path. Purely internal plumbing shared by
        ///     <see cref="np.histogram_bin_edges(NDArray,int,ValueTuple{double,double}?,NDArray)"/> and
        ///     <see cref="np.histogram(NDArray,int,ValueTuple{double,double}?,bool,NDArray)"/>.
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
            /// <summary>The compute/bin dtype B (Half/Single/Double, or Complex for complex data) — valid only when
            ///     <see cref="Uniform"/>.</summary>
            public readonly NPTypeCode BinDtype;
            /// <summary>Complex data's outer edges (<see cref="First"/> / <see cref="Last"/> are unused then) — valid only when
            ///     <see cref="Uniform"/> and <see cref="BinDtype"/> is Complex.</summary>
            public readonly ComplexOuterEdges ComplexEdges;
            /// <summary>True when the outer edges are a range's Python floats rather than the data's own minimum and maximum:
            ///     NumPy's <c>norm_denom</c> is then their float64 difference (<see cref="ZeroBinsBlockHasNaNIndex"/>).</summary>
            public readonly bool RangeEdges;

            /// <summary>Holds the edges and, for equal-width bins, the parameters of the uniform counting path.</summary>
            /// <param name="edges">The bin edges (nbins + 1 of them).</param>
            /// <param name="uniform">Whether the edges are equal-width (only then do the other parameters mean anything).</param>
            /// <param name="first">The lower outer edge.</param>
            /// <param name="last">The upper outer edge.</param>
            /// <param name="nbins">The number of equal-width bins.</param>
            /// <param name="binDtype">The dtype the edges are in and the samples are cast to (Half, Single or Double; Complex
            ///     for complex data, whose counting runs <see cref="HistogramUniformComplexData"/>).</param>
            /// <param name="complexEdges">Complex data's outer edges, or the default for every other dtype.</param>
            /// <param name="rangeEdges">Whether the outer edges are a range's Python floats.</param>
            public BinEdgesResult(NDArray edges, bool uniform, double first, double last, int nbins, NPTypeCode binDtype,
                ComplexOuterEdges complexEdges = default, bool rangeEdges = false)
            {
                Edges = edges; Uniform = uniform; First = first; Last = last; NBins = nbins; BinDtype = binDtype;
                ComplexEdges = complexEdges; RangeEdges = rangeEdges;
            }
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
        internal static NDArray HistogramBinEdges(NDArray a, object bins, (double, double)? range, NDArray weights)
        {
            (NDArray raveled, NDArray w) = RavelAndCheckWeights(a, weights);
            return GetBinEdges(raveled, bins, range, w).Edges;
        }

        /// <summary>
        ///     <c>histogram</c>: the edges as <see cref="HistogramBinEdges"/> finds them, then NumPy's equal-width
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
        /// <exception cref="UnicodeDecodeError">A text search converts <c>S</c> text holding a byte that is not ASCII to a
        ///     <c>U</c> common descriptor (<see cref="HistogramTextAnswer"/>).</exception>
        /// <exception cref="ArgumentException">Bool weights over data on explicit edges: the in-place update of the bool total
        ///     (<see cref="HistogramCumulative"/>, <see cref="HistogramCumulativeLoop"/>).</exception>
        internal static HistogramResult Histogram(NDArray a, object bins, (double, double)? range, bool density, NDArray weights)
        {
            (NDArray raveled, NDArray w) = RavelAndCheckWeights(a, weights);
            BinEdgesResult be = GetBinEdges(raveled, bins, range, w);
            NDArray binEdges = be.Edges;

            // Text data, edges or weights only ever reach NumPy's cumulative path (text data never gets equal-width edges,
            // text weights are never "simple"); its answers are decided by sizes and dtypes alone, or need text computation.
            if (Backends.Strings.NDStringUfuncs.IsText(raveled) || Backends.Strings.NDStringUfuncs.IsText(binEdges)
                || Backends.Strings.NDStringUfuncs.IsText(w))
                return HistogramTextAnswer(raveled, w, binEdges, density);

            // ntype: int64 counts when unweighted, else the weights' dtype (NumPy: hist.dtype == weights.dtype).
            NPTypeCode ntype = w is null ? NPTypeCode.Int64 : w.typecode;

            // The bincount fast path only applies when the weights can accumulate as float64/complex128.
            bool simpleWeights = w is null
                || can_cast(w.typecode, NPTypeCode.Double)
                || can_cast(w.typecode, NPTypeCode.Complex);

            // Complex data's equal-width counting is NumPy's statements in complex128 (HistogramEngine.Complex.cs); the fused
            // index kernel of the real dtypes cannot express a complex quotient.
            NDArray n = (be.Uniform && simpleWeights)
                ? (be.BinDtype == NPTypeCode.Complex
                    ? HistogramUniformComplexData(raveled, w, be, ntype)
                    : HistogramUniform(raveled, w, be, ntype))
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
        ///     reaches (text data never gets equal-width edges, text weights are never "simple", text edges are an explicit
        ///     array): the block loop of <see cref="HistogramCumulativeLoop"/>, then the optional density.
        /// </summary>
        /// <param name="raveled">The raveled data.</param>
        /// <param name="w">The raveled weights, or null.</param>
        /// <param name="binEdges">The edges <see cref="GetBinEdges"/> returned (text edges were compared there, NumPy's
        ///     monotonic check).</param>
        /// <param name="density">Whether to normalize to a density.</param>
        /// <returns>The counts (or summed weights, or density) and the edges.</returns>
        /// <exception cref="TypeError">Text weights: <c>ufunc 'subtract' did not contain a loop …</c> when there is no data
        ///     (NumPy's <c>np.diff</c> of its text-typed running total), else NumPy's add.accumulate refusal (the weights'
        ///     <c>cumsum</c>, the first statement of a block that can fail); text edges with <paramref name="density"/>: the bin
        ///     widths' <c>subtract</c> refusal.</exception>
        /// <exception cref="UnicodeDecodeError">A search that converts <c>S</c> text beside <c>U</c> text to their common
        ///     <c>U</c> descriptor, and a byte is not ASCII.</exception>
        /// <remarks>
        ///     NumPy's loop runs once per 65536-element block of the data: with no data it never runs, so the running total
        ///     stays <c>np.zeros(edges.shape, ntype)</c> (<c>ntype</c>: intp without weights, the weights' dtype with them).
        ///     The counts can come out NEGATIVE where the edges increase as numbers but not as text (NumPy validates numeric
        ///     edges as numbers and searches them as text: edges <c>[1, 2, 10]</c> become <c>'1'</c>, <c>'2'</c>, <c>'10'</c>);
        ///     this is NumPy's answer and the loop below gives it, where binning each sample would not.
        /// </remarks>
        private static HistogramResult HistogramTextAnswer(NDArray raveled, NDArray w, NDArray binEdges, bool density)
        {
            if (w is not null && Backends.Strings.NDStringUfuncs.IsVString(w))
            {
                // StringDType weights: cum_n = np.zeros(bin_edges.shape, weights.dtype) holds empty texts, and NumPy's block
                // loop COMPUTES over them (the weights' cumsum and the in-place add concatenate text), so the counts'
                // n = np.diff(cum_n) is what refuses — unless a statement before it raises first (a None-like NA in the
                // data's argsort, the weights' cumsum or the add). Running the statements gives either answer.
                HistogramCumulativeLoop(raveled, w, binEdges, w.dtype);
                throw new InvalidOperationException("np.histogram: StringDType weights reached a subtraction NumPy refuses.");
            }
            if (w is not null && Backends.Strings.NDStringUfuncs.IsString(w))
            {
                // cum_n = np.zeros(bin_edges.shape, weights.dtype) holds text. With data, a block's `sw.cumsum()`
                // (add.accumulate) refuses it first (the argsort and the gathers before it cannot fail); without data the
                // loop never runs and n = np.diff(cum_n) refuses the text subtraction.
                if (raveled.size == 0)
                    throw Backends.Strings.NDStringUfuncs.NoLoop("subtract", w.dtype, w.dtype);
                throw Backends.Strings.NDStringUfuncs.AddNotCompatible(w.dtype, "accumulate");
            }

            NDArray n = HistogramCumulativeLoop(raveled, w, binEdges, w is null ? DType.Int64 : w.dtype);

            if (density)
            {
                // db = np.array(np.diff(bin_edges), float): text edges (StringDType included) have no subtract loop.
                if (Backends.Strings.NDStringUfuncs.IsText(binEdges))
                    throw Backends.Strings.NDStringUfuncs.NoLoop("subtract", binEdges.dtype, binEdges.dtype);
                NDArray db = np.diff(binEdges).astype(NPTypeCode.Double);
                return new HistogramResult(n / db / np.sum(n), binEdges);
            }
            return new HistogramResult(n, binEdges);
        }

        /// <summary>
        ///     NumPy's cumulative histogram loop, statement for statement: per block of 65536 samples, the block sorted (with
        ///     weights: its argsort, NumPy's default kind) and searched for every edge (<see cref="SearchSortedInclusive"/>),
        ///     the running total updated in place, and the counts taken as <c>np.diff</c> of the total. It runs wherever no
        ///     shortcut gives NumPy's answer: data, edges or weights of which one is text (<see cref="HistogramTextAnswer"/>),
        ///     and unweighted numeric data over edges holding a NaN, or over no edge at all (<see cref="HistogramCumulative"/>).
        /// </summary>
        /// <param name="a">The raveled data (text or numbers).</param>
        /// <param name="w">The raveled numeric weights, or null.</param>
        /// <param name="binEdges">The edges (text or numbers, at least as NumPy's monotonic check left them).</param>
        /// <param name="ntype">NumPy's <c>ntype</c>: int64 (intp) without weights, the weights' descriptor with them (a
        ///     <c>StringDType</c> one included: its running total holds texts, and the final <c>np.diff</c> refuses it).</param>
        /// <returns>The counts (or summed weights), one per bin: <c>np.diff(cum_n)</c>. They can be NEGATIVE: they are
        ///     differences of search positions, which only increase along edges that are ordered the way the search compares
        ///     them.</returns>
        /// <exception cref="UnicodeDecodeError">A search converts <c>S</c> text holding a byte that is not ASCII to a <c>U</c>
        ///     common descriptor.</exception>
        /// <remarks>
        ///     <para>
        ///     Every search runs in the text descriptor common to the block and the edges (np.searchsorted's rule): numeric
        ///     edges are searched as their text, and a numeric block — sorted as numbers — is searched as its text with
        ///     NumPy's exact binary search, whatever order the text has.
        ///     </para>
        ///     <para>
        ///     A NaN edge passes NumPy's monotonic check (no comparison with it is True), and the search places it after every
        ///     number (NaN sorts last): an inner NaN edge makes the next count negative (edges <c>[0, nan, 2]</c> over twelve
        ///     samples, eight of them at least 0: <c>[8, -5]</c>), and a NaN last edge counts the NaN samples in the last bin.
        ///     </para>
        ///     <para>
        ///     The weighted form argsorts each block with the DEFAULT kind on purpose: NumPy's introsort decides the order of
        ///     equal strings, hence the order in which their weights enter the cumulative sum, and so the last bits of a float
        ///     total (TextSort reproduces that order).
        ///     </para>
        ///     <para>Scoped: every block's sorted copy, order, gathers and searches are reclaimed; only the counts
        ///     survive.</para>
        /// </remarks>
        [NDScoped]
        private static NDArray HistogramCumulativeLoop(NDArray a, NDArray w, NDArray binEdges, DType ntype)
        {
            long len = a.size;
            // cum_n = np.zeros(bin_edges.shape, ntype)
            NDArray cumN = np.zeros(Shape.Vector(binEdges.size), ntype);
            NDArray zero = w is null ? null : np.zeros(Shape.Vector(1), ntype);
            for (long start = 0; start < len; start += HistogramBlock)
            {
                long blockLen = System.Math.Min(HistogramBlock, len - start);
                string slice = $"{start}:{start + blockLen}";
                NDArray tmpA = a[slice];
                if (w is null)
                {
                    // sa = np.sort(a[i:i + BLOCK]); cum_n += _search_sorted_inclusive(sa, bin_edges)
                    NDArray sa = np.sort(tmpA);
                    np.add(cumN, SearchSortedInclusive(sa, binEdges), @out: cumN);
                }
                else
                {
                    // sorting_index = np.argsort(tmp_a); sa = tmp_a[sorting_index]; sw = tmp_w[sorting_index]
                    NDArray sortIdx = np.argsort(tmpA);
                    NDArray sa = tmpA[sortIdx];
                    NDArray sw = w[slice][sortIdx];
                    // cw = np.concatenate((zero, sw.cumsum())); cum_n += cw[bin_index]
                    NDArray cw = np.concatenate((zero, np.cumsum(sw)), 0);
                    NDArray binIndex = SearchSortedInclusive(sa, binEdges);
                    np.add(cumN, np.take(cw, binIndex), @out: cumN);
                }
            }
            // n = np.diff(cum_n)
            return np.diff(cumN);
        }

        // ---------------------------------------------------------------------
        //  Fast (equal-width) path — fused index kernel + bincount
        // ---------------------------------------------------------------------

        /// <summary>
        ///     The uniform (equal-width) histogram path: compute every sample's bin index with the fused
        ///     kernel, then accumulate with <see cref="np.bincount(NDArray,NDArray,int)"/>. Out-of-range samples
        ///     ride a sentinel bin that is sliced off — identical to NumPy's <c>keep</c>-mask counts, and the
        ///     weighted accumulation is blocked at <see cref="HistogramBlock"/> for bit-exact float sums.
        /// </summary>
        /// <param name="raveled">The raveled numeric data.</param>
        /// <param name="w">The raveled weights (castable to float64 or complex128), or null for counts.</param>
        /// <param name="be">The equal-width edges and their outer edges, count and dtype.</param>
        /// <param name="ntype">The result's dtype: int64 without weights, the weights' dtype with them.</param>
        /// <returns>The counts (or summed weights), one per bin, in <paramref name="ntype"/>.</returns>
        /// <exception cref="ArgumentException">Zero bins with a sample between the outer edges: NumPy's np.bincount refusal
        ///     (<c>'list' argument must have no negative elements</c>, a ValueError in NumPy; NumSharp's np.bincount raises
        ///     it as an ArgumentException), see <see cref="HistogramZeroBins"/>.</exception>
        /// <remarks>The kernels read raw addresses, so the samples, edges and weights are first made contiguous from their
        ///     element 0 in the kernel's dtype (<see cref="EnsureContiguousDtype"/>). They assume at least one bin.</remarks>
        private static unsafe NDArray HistogramUniform(NDArray raveled, NDArray w, BinEdgesResult be, NPTypeCode ntype)
        {
            long len = raveled.size;
            int nbins = be.NBins;

            // Empty input: no samples, all-zero histogram of the right dtype.
            if (len == 0)
                return np.zeros(Shape.Vector(nbins), ntype);

            // Zero bins: the kernels would count every sample outside the zero-length result (a heap corruption); NumPy's
            // statements refuse such samples instead.
            if (nbins == 0)
                return HistogramZeroBins(raveled, w, be, ntype);

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

        /// <summary>
        ///     NumPy's equal-width counting loop when there are no bins: an estimator's <c>int(np.ceil(delta / width))</c> is
        ///     0 when the width is infinite and the range is not (<c>scott</c>'s width is a standard deviation in the data's
        ///     dtype, whose squares overflow once a float16 sample lies 256 or more from the mean, a float32 one about
        ///     1.8e19, a float64 one about 1.3e154), or when the outer edges have no width left: data with no spread whose
        ///     ±0.5 widening rounds back to the value itself (a float16 1026, a float32 1e9, a float64 or integer of 2^53 or
        ///     more), where scott's np.std is the rounding error of the mean. A kept sample's index is
        ///     <c>(_unsigned_subtract(tmp_a, first_edge) / norm_denom) * 0</c>: a NaN when the quotient is not finite (a zero
        ///     <c>norm_denom</c> gives 0 / 0), which <c>astype(np.intp)</c> makes INT64_MIN and the fancy index
        ///     <c>bin_edges[indices]</c> refuses; else 0, which <c>indices[indices == n_equal_bins] -= 1</c> turns into -1 and
        ///     the block's <c>np.bincount</c> refuses. With no sample between the outer edges the loop adds nothing and the
        ///     counts are empty.
        /// </summary>
        /// <param name="raveled">The raveled real data (at least one sample).</param>
        /// <param name="w">The raveled weights, or null (an estimator refuses weights, so this is null in practice).</param>
        /// <param name="be">The equal-width edges (one edge) and the outer edges.</param>
        /// <param name="ntype">The result's dtype.</param>
        /// <returns>The empty counts, when no sample lies between the outer edges.</returns>
        /// <exception cref="IndexError">A kept sample of the first block holding one whose quotient is not finite:
        ///     <c>index -9223372036854775808 is out of bounds for axis 0 with size 1</c>.</exception>
        /// <exception cref="ArgumentException">Otherwise a sample between the outer edges: np.bincount's refusal of the -1
        ///     indices (<c>'list' argument must have no negative elements</c>; NumSharp's np.bincount raises NumPy's
        ///     ValueError text as an ArgumentException).</exception>
        /// <remarks>
        ///     Zero bins come only from an estimator, which ran over the samples between the outer edges and found at least
        ///     one, so NumPy's answer is in practice always one of the refusals. The keep mask is the estimator branch's own
        ///     comparison with the outer edges, so it keeps the same samples. Decimal data (no NumPy dtype) skip the quotient
        ///     and take the bincount refusal.
        /// </remarks>
        private static NDArray HistogramZeroBins(NDArray raveled, NDArray w, BinEdgesResult be, NPTypeCode ntype)
        {
            // keep = (tmp_a >= first_edge) & (tmp_a <= last_edge) over every block: the first block holding a kept sample
            // raises, so whether any sample is kept decides the answer.
            using NDArray first = NDArray.Scalar(be.First);
            using NDArray last = NDArray.Scalar(be.Last);
            using NDArray keep = (raveled >= first) & (raveled <= last);
            long kept = np.count_nonzero(keep);
            if (kept > 0)
            {
                // `decrement = tmp_a < bin_edges[indices]` runs before the bincount: a NaN index of that first block is the
                // first refusal.
                if (raveled.typecode != NPTypeCode.Decimal && ZeroBinsBlockHasNaNIndex(raveled, keep, be))
                    throw new IndexError("index -9223372036854775808 is out of bounds for axis 0 with size 1");
                // indices = f_indices.astype(np.intp) is 0 (a product with n_equal_bins = 0), then -1; np.bincount refuses.
                using NDArray indices = np.full(Shape.Vector(kept), -1L, np.int64);
                using NDArray weights = w?[keep];
                bincount(indices, weights, 0);
                throw new InvalidOperationException("np.histogram: np.bincount accepted negative indices.");
            }
            return np.zeros(Shape.Vector(0), ntype);
        }

        /// <summary>
        ///     Whether a kept sample of the first block of <see cref="HistogramBlock"/> samples holding one gets a NaN index
        ///     from NumPy's <c>f_indices = (_unsigned_subtract(tmp_a, first_edge) / norm_denom) * n_equal_bins</c> with zero
        ///     bins: the quotient computed in NumPy's dtypes is not finite.
        /// </summary>
        /// <param name="raveled">The raveled float16 / float32 / float64 / integer data.</param>
        /// <param name="keep">The samples between the outer edges (at least one).</param>
        /// <param name="be">The outer edges, the edges' dtype and whether the outer edges are a range's.</param>
        /// <returns>True when NumPy's fancy index meets INT64_MIN in that block.</returns>
        /// <remarks>
        ///     The dtypes, as NEP 50 gives them: <c>tmp_a.astype(bin_edges.dtype)</c> and its difference with the lower edge
        ///     are in the edges' dtype (the data's float16 / float32 / float64, float64 for integer data; a range's Python
        ///     float rounds to it, an integer edge converts to float64). <c>norm_denom</c> is the range's float64 difference,
        ///     or the data's own edges' (<see cref="DataEdgesDelta"/>). The quotient keeps float16 / float32 only when both
        ///     operands have it, and is float64 otherwise.
        /// </remarks>
        private static bool ZeroBinsBlockHasNaNIndex(NDArray raveled, NDArray keep, BinEdgesResult be)
        {
            long len = raveled.size;
            long firstKept = 0;
            while (!keep.GetAtIndex<bool>(firstKept))
                firstKept++;
            long blockStart = firstKept / HistogramBlock * HistogramBlock;
            long blockEnd = Math.Min(blockStart + HistogramBlock, len);

            NpScalar normDenom = be.RangeEdges ? NpScalar.Float64(be.Last - be.First) : DataEdgesDelta(raveled);
            NPTypeCode b = be.BinDtype;
            double firstEdge = RoundToEdgeDtype(be.First, b);
            // The samples as float64: exact for float data, tmp_a.astype(float64) for integer data (the edges' dtype then).
            using NDArray samples = raveled[$"{blockStart}:{blockEnd}"].astype(NPTypeCode.Double);
            for (long i = blockStart; i < blockEnd; i++)
            {
                if (!keep.GetAtIndex<bool>(i))
                    continue;
                double x = samples.GetAtIndex<double>(i - blockStart);
                // _unsigned_subtract(tmp_a, first_edge) in the edges' dtype (half and float32 arithmetic rounded back).
                double diff = b switch
                {
                    NPTypeCode.Half => (double)(Half)((float)x - (float)firstEdge),
                    NPTypeCode.Single => (float)x - (float)firstEdge,
                    _ => x - firstEdge,
                };
                double quotient = DivideScalars(new NpScalar(diff, b == NPTypeCode.Half || b == NPTypeCode.Single ? b : NPTypeCode.Double), normDenom);
                if (!double.IsFinite(quotient))
                    return true;
            }
            return false;
        }

        /// <summary>A value rounded to the edges' dtype (a Python float meeting float16 / float32 data takes its dtype).</summary>
        /// <param name="value">The value.</param>
        /// <param name="dtype">Half, Single or Double.</param>
        /// <returns>The rounded value.</returns>
        private static double RoundToEdgeDtype(double value, NPTypeCode dtype) => dtype switch
        {
            NPTypeCode.Half => (double)(Half)value,
            NPTypeCode.Single => (float)value,
            _ => value,
        };

        // ---------------------------------------------------------------------
        //  Slow (arbitrary-edge) path — sort + inclusive searchsorted, cumulative
        // ---------------------------------------------------------------------

        /// <summary>
        ///     The general histogram path for non-uniform edges (or weights that cannot bincount): a port of
        ///     NumPy's cumulative-histogram algorithm. Unweighted uses sorted inclusive searchsorted counts;
        ///     weighted uses the sorted cumulative-weight differences, both blocked at <see cref="HistogramBlock"/>.
        /// </summary>
        /// <param name="raveled">The raveled numeric data.</param>
        /// <param name="w">The raveled numeric weights, or null for counts.</param>
        /// <param name="binEdges">The explicit edges (NumPy's monotonic check passed them, which a NaN edge does).</param>
        /// <param name="ntype">The result's dtype: int64 without weights, the weights' dtype with them.</param>
        /// <returns>The counts (or summed weights), one per bin; edges holding a NaN can give negative counts (NumPy's
        ///     answer, see <see cref="HistogramCumulativeLoop"/>). Summed weights keep the weights' dtype, wrapping in a
        ///     narrow integer dtype. No edges at all give an empty array (int64 counts, or the weights' dtype).</returns>
        /// <exception cref="ArgumentException">Bool weights with data: NumPy's UFuncTypeError <c>Cannot cast ufunc 'add'
        ///     output from dtype('int64') to dtype('bool') with casting rule 'same_kind'</c> (the in-place update of the
        ///     bool total by the weights' int64 cumulative sum).</exception>
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

                // No edges at all (an empty array passes NumPy's checks): NumPy's loop searches nothing and its counts are
                // np.diff of an empty total, an empty int64 array, whatever the data; the binning would ask for -1 bins.
                if (nb < 0)
                    return HistogramCumulativeLoop(raveled, null, binEdges, DType.Int64);

                NPTypeCode cmp = promote_types(raveled.typecode, binEdges.typecode);

                // Binning each sample gives NumPy's counts only while the edges are ordered for the search NumPy runs
                // instead. A NaN edge is not: it passes the monotonic check, the search places it after every number,
                // and NumPy's counts become differences that can be negative, or count the NaN samples under a NaN last
                // edge. Such edges run NumPy's statements themselves. Decimal data search in decimal, where a NaN edge
                // has no value (NumPy has no decimal dtype to follow), so they keep the binning.
                if (cmp != NPTypeCode.Decimal && EdgesHoldNaN(binEdges))
                    return HistogramCumulativeLoop(raveled, null, binEdges, DType.Int64);

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
                    // In place, as NumPy's `+=`: the total keeps the weights' dtype (ntype), so narrow integer weights
                    // wrap in their own dtype, and bool weights refuse the int64 cumulative sum under same_kind
                    // ("Cannot cast ufunc 'add' output from dtype('int64') to dtype('bool') ..."), even when the edges
                    // leave nothing to add.
                    np.add(cumN, np.take(cw, binIndex), @out: cumN);
                }
            }

            return np.diff(cumN);
        }

        /// <summary>
        ///     NumPy's <c>_search_sorted_inclusive</c>: <c>searchsorted(a, v[:-1], 'left')</c> concatenated with
        ///     <c>searchsorted(a, v[-1:], 'right')</c>, so the last bin edge is inclusive (a value equal to the
        ///     final edge lands in the last bin, not past it).
        /// </summary>
        /// <param name="a">The sorted samples of one block.</param>
        /// <param name="v">The edges.</param>
        /// <returns>For each edge, the number of samples before it (the last edge counting the samples equal to it).</returns>
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
        /// <exception cref="ValueError">Text edges that decrease (<c>`bins` must increase monotonically, when an array</c>): the
        ///     string <c>greater</c> comparison of each edge with the next, NumPy's monotonicity check.</exception>
        /// <remarks>
        ///     Explicit edges never read the data, so text data with explicit edges gets its edges back here; the histogram's
        ///     counting step decides the rest (<see cref="HistogramTextAnswer"/>).
        /// </remarks>
        private static BinEdgesResult GetBinEdges(NDArray a, object bins, (double, double)? range, NDArray weights)
        {
            int? nEqualBins = null;
            NDArray binEdges = null;
            double first = 0, last = 0;
            bool textData = Backends.Strings.NDStringUfuncs.IsText(a);
            // Complex data meet the same statements in complex128 (HistogramEngine.Complex.cs): complex outer edges, complex
            // equal-width edges, and estimators whose complex widths stop at int(np.ceil(...)).
            bool complexData = a.typecode == NPTypeCode.Complex;
            ComplexOuterEdges complexEdges = default;

            if (bins is string binName)
            {
                if (!IsKnownEstimator(binName))
                    throw new ValueError($"'{binName}' is not a valid estimator for `bins`");
                if (weights is not null)
                    throw new TypeError("Automated estimation of the number of bins is not supported for weighted data");

                if (complexData)
                    return ComplexEqualWidthResult(ComplexEstimatorBinCount(a, binName, range, out complexEdges), complexEdges);

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

                // The edges' linspace computes in float64 for a range's Python floats, else in the data's width.
                NPTypeCode linspaceDtype = range.HasValue ? NPTypeCode.Double : FloatPromoteBinType(a.typecode);
                if (x.size == 0)
                    nEqualBins = 1;
                else if (a.typecode == NPTypeCode.Decimal)
                {
                    // NumPy has no decimal dtype: decimal data keep NumSharp's own estimators, in double.
                    double width = EstimateBinWidth(binName, x, first, last, range);
                    if (width > 0)
                        nEqualBins = EstimatorBinCount((last - first) / width, linspaceDtype);
                    else
                        nEqualBins = 1; // e.g. FD when the IQR is zero
                }
                else
                {
                    // width = _hist_bin_selectors[bin_name](a, (first_edge, last_edge)), in NumPy's dtypes.
                    NpScalar width = EstimateWidthNumPy(binName, x, range);
                    if (width.IsTruthy)
                    {
                        // Integer data never gets a sub-unit bin width: NumPy's `np.issubdtype(a.dtype, np.integer)`, which
                        // holds for NumSharp's char too (an unsignedinteger in its scalar hierarchy, the uint16 of the oracle);
                        // the width is then the Python int 1.
                        if (np.issubdtype(a.dtype, np.integer) && width.Value < 1)
                            width = new NpScalar(1, NPTypeCode.UInt64);
                        // delta = _unsigned_subtract(last_edge, first_edge): a range's Python floats subtract in float64, the
                        // data's own edges in the data's dtype (an integer data's exact difference; with no spread, the
                        // ±0.5 widening's difference).
                        NpScalar delta = range.HasValue ? NpScalar.Float64(last - first) : DataEdgesDelta(a);
                        nEqualBins = EstimatorBinCount(DivideScalars(delta, width), linspaceDtype);
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
                // edges pass as they are; two or more text edges are compared by the string greater kernel (StringDType's
                // with its NA rules: a NaN-like NA compares False, a None-like one raises).
                if (Backends.Strings.NDStringUfuncs.IsText(binEdges))
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
                if (complexData)
                    return ComplexEqualWidthResult(count, GetComplexOuterEdges(a, range));
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
                return new BinEdgesResult(edges, true, first, last, nb, resultType, rangeEdges: range.HasValue);
            }

            return new BinEdgesResult(binEdges, false, 0, 0, 0, NPTypeCode.Double);
        }

        /// <summary>
        ///     NumPy's <c>n_equal_bins = int(np.ceil(delta / width))</c> after an estimator, with what the
        ///     <c>np.linspace(first_edge, last_edge, n_equal_bins + 1, ...)</c> that follows makes of a count too large to
        ///     allocate: Python's int() refuses a NaN and an infinity, and np.arange / the allocator refuse a huge edge count.
        /// </summary>
        /// <param name="quotient"><c>delta / width</c>.</param>
        /// <param name="linspaceDtype">The dtype linspace computes the edges in (float64 for Python-number outer edges and
        ///     integer data, the data's float16 / float32 width for its own outer edges): its arange's dtype.</param>
        /// <returns>The number of equal-width bins, when the edges can be allocated.</returns>
        /// <exception cref="ValueError">A NaN quotient (<c>cannot convert float NaN to integer</c>, an infinite delta over an
        ///     infinite width); an edge count whose float64 value exceeds 2^63 (np.arange's <c>Maximum allowed size
        ///     exceeded</c>); an edge count whose bytes exceed int64 (<c>array is too big; …</c>).</exception>
        /// <exception cref="OverflowException">An infinite quotient (<c>cannot convert float infinity to integer</c>).</exception>
        /// <exception cref="IndexError">An edge count whose float64 value is exactly 2^63: np.arange's length cast wraps to a
        ///     negative number, the arange is empty, and linspace's <c>y[-1] = stop</c> fails (<c>index -1 is out of bounds
        ///     for axis 0 with size 0</c>).</exception>
        /// <exception cref="OutOfMemoryException">An edge count NumSharp's int bin count cannot hold: NumPy's MemoryError for
        ///     the arange (<c>Unable to allocate 64.0 GiB for an array with shape (8589934593,) and data type float64</c>).
        ///     NumPy raises it when the memory is not there; with enough memory it allocates and goes on, which NumSharp's
        ///     histogram (an int bin count) does not.</exception>
        /// <remarks>
        ///     Reachable by real data: integer data whose estimated width is clamped to 1 count one bin per unit of their
        ///     range (int64 nanosecond timestamps clustered in time give billions), and float data with a tiny
        ///     interquartile range and a far outlier give a count past 2^64. .NET's saturating double-to-int conversion would
        ///     have turned a NaN into 0 bins and a huge count into int.MaxValue, whose edge count wraps negative.
        /// </remarks>
        private static int EstimatorBinCount(double quotient, NPTypeCode linspaceDtype)
        {
            double count = Math.Ceiling(quotient);
            if (double.IsNaN(count))
                throw new ValueError("cannot convert float NaN to integer");
            if (double.IsInfinity(count))
                throw new OverflowException("cannot convert float infinity to integer");
            if (count < int.MaxValue)
                return (int)count; // count + 1 edges fit an int

            // np.linspace(..., num=count + 1): its arange(0, num) computes the length as a double, ceils it and checks it
            // against intp (`_arange_safe_ceil_to_intp`, whose bound (double)NPY_MAX_INTP is 2^63 itself).
            // The length is Python's `(num - 0) / 1`, a correctly rounded int division (.NET's BigInteger-to-double cast
            // truncates instead).
            var edges = new System.Numerics.BigInteger(count) + 1;
            double length = PyScalar.IntTrueDivide(edges, System.Numerics.BigInteger.One);
            if (length > 9223372036854775808.0)
                throw new ValueError("Maximum allowed size exceeded");
            if (length == 9223372036854775808.0)
                throw new IndexError("index -1 is out of bounds for axis 0 with size 0");

            // The allocation of length elements: a byte count past intp is "array is too big", anything else NumPy's
            // MemoryError when the memory is not there.
            var nbytes = new System.Numerics.BigInteger(length) * linspaceDtype.SizeOf();
            if (nbytes > long.MaxValue)
                throw new ValueError(AllocationGuard.TooBigMessage);
            throw new OutOfMemoryException(
                $"Unable to allocate {NumPySizeString(nbytes)} for an array with shape ({new System.Numerics.BigInteger(length)},) " +
                $"and data type {linspaceDtype.AsNumpyDtypeName()}");
        }

        /// <summary>
        ///     NumPy's <c>_ArrayMemoryError._size_to_string</c>: a byte count in binary units (bytes, KiB, … EiB), with three
        ///     significant figures in Python's alternate form (<c>'#.3g'</c>: <c>8.00</c>, <c>16.0</c>, <c>149.</c>) below 1000
        ///     units and every integer digit (<c>'#.0f'</c>) from there on.
        /// </summary>
        /// <param name="numBytes">The byte count (positive).</param>
        /// <returns>The size text, e.g. <c>64.0 GiB</c>.</returns>
        private static string NumPySizeString(System.Numerics.BigInteger numBytes)
        {
            const int log2Step = 10, step = 1024;
            string[] units = { "bytes", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB" };

            int unitI = (int)(Math.Max(numBytes.GetBitLength() - 1, 1) / log2Step);
            // n_units = num_bytes / unit_val: Python's int / int, correctly rounded.
            double nUnits = PyScalar.IntTrueDivide(numBytes, System.Numerics.BigInteger.One << (unitI * log2Step));

            // Ensure the unit is right after rounding (Python's round() is half-to-even).
            if (Math.Round(nUnits, MidpointRounding.ToEven) == step)
            {
                unitI++;
                nUnits /= step;
            }
            // Sizes too large for the units: scale back to the last one.
            if (unitI >= units.Length)
            {
                int newUnitI = units.Length - 1;
                nUnits *= Math.Pow(2, (unitI - newUnitI) * log2Step);
                unitI = newUnitI;
            }

            string unitName = units[unitI];
            string number = unitI == 0
                ? Backends.Printing.PrintfFormatter.Format("%.0f", nUnits, Backends.Printing.PercentFormat.PyObject)
                : Math.Round(nUnits, MidpointRounding.ToEven) < 1000
                    ? Backends.Printing.PrintfFormatter.Format("%#.3g", nUnits, Backends.Printing.PercentFormat.PyObject)
                    : Backends.Printing.PrintfFormatter.Format("%#.0f", nUnits, Backends.Printing.PercentFormat.PyObject);
            return number + " " + unitName;
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
        internal static (double first, double last) GetOuterEdges(NDArray a, (double, double)? range)
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
            else if (Backends.Strings.NDStringUfuncs.IsVString(a))
            {
                // `first_edge, last_edge = a.min(), a.max()` COMPUTE over StringDType (a None-like NA raises there), then
                // `np.isfinite(first_edge)`: a str has no isfinite loop; a NaN-like NA (the minimum only when every element
                // is one) is not finite, and the `and` stops there, reporting both edges.
                object lo, hi;
                using (var mn = np.amin(a))
                    lo = mn.Storage.GetAtIndex(0);
                using (var mx = np.amax(a))
                    hi = mx.Storage.GetAtIndex(0);
                if (lo is string || lo is char)
                    throw Backends.Strings.NDStringUfuncs.NotSupportedForInputTypes("isfinite");
                throw new ValueError($"autodetected range of [{Backends.Strings.NDPyObject.Str(lo)}, " +
                                     $"{Backends.Strings.NDPyObject.Str(hi)}] is not finite");
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

        /// <summary>Whether <paramref name="name"/> is one of NumPy's <c>_hist_bin_selectors</c> keys.</summary>
        /// <param name="name">The estimator name as given (case-sensitive, as in NumPy).</param>
        /// <returns>True for <c>stone</c>, <c>auto</c>, <c>doane</c>, <c>fd</c>, <c>rice</c>, <c>scott</c>, <c>sqrt</c> and
        ///     <c>sturges</c>.</returns>
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
        /// <param name="name">A known estimator name (<see cref="IsKnownEstimator"/>).</param>
        /// <param name="x">The range-trimmed, non-empty data.</param>
        /// <param name="first">The lower outer edge.</param>
        /// <param name="last">The upper outer edge.</param>
        /// <param name="range">The caller's range, or null (Stone's rule evaluates histograms over it).</param>
        /// <returns>The bin width; 0 or less means one bin (<see cref="GetBinEdges"/> decides).</returns>
        /// <exception cref="ValueError">An unknown name (unreachable: the caller checked it first).</exception>
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
        /// <param name="x">The range-trimmed, non-empty data.</param>
        /// <param name="n">Its element count.</param>
        /// <returns>The bin width.</returns>
        private static double BinWidthFd(NDArray x, long n)
        {
            NDArray q = np.percentile(x, new double[] { 75, 25 });
            double iqr = ScalarDouble(q[0]) - ScalarDouble(q[1]);
            return 2.0 * iqr * System.Math.Pow(n, -1.0 / 3.0);
        }

        /// <summary>Doane's improved Sturges rule; accounts for skew. Returns 0 when std is 0 or n ≤ 2.</summary>
        /// <param name="x">The range-trimmed, non-empty data.</param>
        /// <param name="n">Its element count.</param>
        /// <returns>The bin width, or 0 (one bin).</returns>
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
        /// <param name="x">The range-trimmed, non-empty data.</param>
        /// <param name="first">The lower outer edge (the histograms' range when <paramref name="range"/> is null).</param>
        /// <param name="last">The upper outer edge.</param>
        /// <param name="range">The caller's range, or null.</param>
        /// <returns>The bin width of the best count, or 0 (one bin) for a single sample or constant data.</returns>
        /// <remarks>Each candidate count is a call of the public <c>np.histogram</c> (NumPy's own estimator calls
        ///     <c>np.histogram</c> too), so this is up to max(100, sqrt(n)) histograms.</remarks>
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
        /// <param name="x">The non-empty data.</param>
        /// <returns>The maximum minus the minimum, both read as float64.</returns>
        private static double Ptp(NDArray x) => ScalarDouble(np.amax(x)) - ScalarDouble(np.amin(x));

        /// <summary>
        ///     The float width the equal-width bin edges (and the index arithmetic) use, matching NumPy's
        ///     <c>result_type(first_edge, last_edge, a)</c> with weak-float endpoints: float16/float32 keep their
        ///     width, everything else (int/bool/char/float64/decimal/complex) resolves to float64.
        /// </summary>
        /// <param name="inputType">The data's dtype.</param>
        /// <returns>Half, Single, or Double.</returns>
        internal static NPTypeCode FloatPromoteBinType(NPTypeCode inputType) => inputType switch
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
        internal static unsafe NDArray HistogramLinspace(double first, double last, int num, NPTypeCode computeType, NPTypeCode resultType)
        {
            // Compute in the arithmetic dtype, then cast to the result dtype (a no-op when they coincide).
            NDArray computed = ComputeLinspaceInDtype(first, last, num, computeType);
            return computeType == resultType ? computed : computed.astype(resultType);
        }

        /// <summary>
        ///     The single-dtype linspace core: float64 defers to <see cref="np.linspace(double,double,int,bool,DType,string)"/> (already double-internal),
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
            int div = num - 1; // histogram always has endpoint=true and num = nbins + 1, so div >= 0

            if (div == 0)
            {
                // num == 1 (zero bins, an estimator whose width came out infinite): NumPy's linspace takes
                // `step = nan; y = y * delta` (y = [0]), then `y += start`, and sets `y[-1] = stop` only when num > 1. So the
                // single edge is 0 * delta + start, in the dtype's own arithmetic (float16: each operation through float32,
                // rounded back), not stop.
                if (binType == NPTypeCode.Single)
                {
                    float start = (float)first, stop = (float)last;
                    ((float*)edges.Address)[0] = 0f * (stop - start) + start;
                }
                else
                {
                    Half start = (Half)first, stop = (Half)last;
                    Half delta = (Half)((float)stop - (float)start);
                    Half scaled = (Half)(0f * (float)delta);
                    ((Half*)edges.Address)[0] = (Half)((float)scaled + (float)start);
                }
                return edges;
            }

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
        /// <param name="nd">A one-element numeric array (a reduction's 0-d result).</param>
        /// <returns>Its element as float64 (NumSharp's cast: a complex keeps its real part).</returns>
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
        internal static NDArray EnsureContiguousDtype(NDArray a, NPTypeCode tc)
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
        /// <exception cref="InvalidOperationException">Defensive: a promotion that succeeded (none can: NumPy 2.4.2 has no
        ///     common DType of text and a Python number).</exception>
        private static void ThrowTextEdgeType(NDArray a, bool fromRange)
        {
            // Boxed C# integer / double literals are the weak Python ints / floats of DTypePromotion.ResultType.
            if (fromRange)
                DTypePromotion.ResultType(0.0, 1.0, a.dtype);
            else
                DTypePromotion.ResultType(0L, 1L, a.dtype);
            throw new InvalidOperationException($"np.histogram: the edges' dtype of {a.dtype} data promoted unexpectedly");
        }

        /// <summary>True when every element of the boolean mask is true (NumPy's <c>logical_and.reduce</c>).</summary>
        /// <param name="mask">The bool mask.</param>
        /// <returns>True when no element is False (an empty mask is all True).</returns>
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
        /// <param name="edges">The 1-D numeric edges.</param>
        /// <returns>NumPy's <c>np.any(bin_edges[:-1] &gt; bin_edges[1:])</c>; false for fewer than two edges, and a NaN
        ///     makes no comparison True.</returns>
        internal static bool AnyDecreasing(NDArray edges)
        {
            long m = edges.size;
            if (m < 2) return false;
            return np.any(edges[$"0:{m - 1}"] > edges[$"1:{m}"]);
        }

        /// <summary>
        ///     True when a float or complex edge array holds a NaN (a complex edge with a NaN part): edges NumPy's monotonic
        ///     check accepts, but whose counts only NumPy's cumulative statements give (<see cref="HistogramCumulativeLoop"/>).
        /// </summary>
        /// <param name="edges">The 1-D numeric edges.</param>
        /// <returns>False for the integer, bool, char and decimal dtypes, which hold no NaN, without reading them.</returns>
        private static bool EdgesHoldNaN(NDArray edges)
        {
            switch (edges.typecode)
            {
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Complex:
                    using (NDArray nan = np.isnan(edges))
                        return np.any(nan);
                default:
                    return false;
            }
        }

        /// <summary>True when any adjacent pair is non-increasing (edges collapsed by too many bins over a tiny range).</summary>
        /// <param name="edges">The computed equal-width edges.</param>
        /// <returns>NumPy's <c>np.any(bin_edges[:-1] &gt;= bin_edges[1:])</c>; false for fewer than two edges.</returns>
        private static bool AnyNonIncreasing(NDArray edges)
        {
            long m = edges.size;
            if (m < 2) return false;
            return np.any(edges[$"0:{m - 1}"] >= edges[$"1:{m}"]);
        }

        /// <summary>Build a complex array from separate real and imaginary parts (used for complex weights).</summary>
        /// <param name="re">The float64 real parts.</param>
        /// <param name="im">The float64 imaginary parts, of <paramref name="re"/>'s shape.</param>
        /// <returns>A fresh complex128 array of <paramref name="re"/>'s shape.</returns>
        private static NDArray MakeComplex(NDArray re, NDArray im)
        {
            var result = new NDArray(NPTypeCode.Complex, re.Shape, fillZeros: false);
            long n = re.size;
            for (long i = 0; i < n; i++)
                result.SetAtIndex(new System.Numerics.Complex(re.GetAtIndex<double>(i), im.GetAtIndex<double>(i)), i);
            return result;
        }

        /// <summary>Python's <c>math.isfinite</c> of an outer edge.</summary>
        /// <param name="v">The edge.</param>
        /// <returns>False for NaN and both infinities.</returns>
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
        /// <param name="a">The data (any shape).</param>
        /// <param name="weights">The weights, or null.</param>
        /// <returns>The raveled data (uint8 for a bool input) and the raveled weights (null without weights).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="weights"/> has another shape than <paramref name="a"/> ("weights
        ///     should have the same shape as a.").</exception>
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
        /// <param name="x">The first array.</param>
        /// <param name="y">The second array.</param>
        /// <returns>True when the ranks and every dimension agree (strides and layout are not compared).</returns>
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
