using System;
using System.Numerics;
using NumSharp.Backends;
using static NumSharp.np;

namespace NumSharp.Statistics
{
    internal static partial class HistogramEngine
    {
        // ---------------------------------------------------------------------
        //  Complex data: NumPy's equal-width statements in complex128
        // ---------------------------------------------------------------------
        //
        // NumPy has no complex path of its own: its Python code meets complex data with the same statements as real data,
        // and they compute in complex128 because a.min() / a.max() are complex scalars and result_type(first_edge,
        // last_edge, a) is complex128. So the edges are complex, every comparison is lexicographic (the real parts first),
        // the index arithmetic is complex and its cast to intp keeps the real part, and an estimator whose width or range
        // is complex stops at int(np.ceil(...)), which has no complex loop. The members below are those statements over
        // NumSharp's complex ufuncs, which give NumPy's bits (division by Smith's method, the fused simd_cmul product).

        /// <summary>
        ///     The outer edges of complex data as NumPy's <c>_get_outer_edges</c> leaves them: complex128 scalars from the data's
        ///     lexicographic <c>a.min()</c> / <c>a.max()</c>, or the Python numbers of a range (floats) or of empty data
        ///     (the ints 0 and 1).
        /// </summary>
        /// <remarks>
        ///     Which of the two it is decides the edges' arithmetic, not only their values: a linspace of Python numbers runs
        ///     in float64 and is cast to complex128 at the end, a linspace of complex128 scalars runs in complex128, whose
        ///     step is a Smith division by <c>(div + 0j)</c> (a multiplication by its reciprocal), so even complex edges
        ///     with zero imaginary parts differ from the float64 ones in the last bit.
        /// </remarks>
        private readonly struct ComplexOuterEdges
        {
            /// <summary>The lower outer edge (a real number when <see cref="Strong"/> is false).</summary>
            public readonly Complex First;
            /// <summary>The upper outer edge (a real number when <see cref="Strong"/> is false).</summary>
            public readonly Complex Last;
            /// <summary>True when the edges are the data's complex128 scalars; false for the Python numbers of a range or of
            ///     empty data.</summary>
            public readonly bool Strong;

            /// <summary>Holds the outer edges and where they came from.</summary>
            /// <param name="first">The lower outer edge.</param>
            /// <param name="last">The upper outer edge.</param>
            /// <param name="strong">Whether they are complex128 scalars of the data.</param>
            public ComplexOuterEdges(Complex first, Complex last, bool strong)
            {
                First = first; Last = last; Strong = strong;
            }

            /// <summary>The lower edge as NumPy's statements hold it: a complex128 scalar, or a float64 one for a Python
            ///     number (every operation with complex data promotes either to complex128).</summary>
            /// <returns>A fresh 0-d array.</returns>
            public NDArray FirstScalar() => Strong ? NDArray.Scalar(First) : NDArray.Scalar(First.Real);

            /// <summary>The upper edge as NumPy's statements hold it (see <see cref="FirstScalar"/>).</summary>
            /// <returns>A fresh 0-d array.</returns>
            public NDArray LastScalar() => Strong ? NDArray.Scalar(Last) : NDArray.Scalar(Last.Real);
        }

        /// <summary>
        ///     NumPy's <c>_get_outer_edges</c> for complex data: the range as given (checked as real numbers), the ints 0 and 1
        ///     for empty data, else the data's lexicographic minimum and maximum, which must both be finite (both parts); a
        ///     zero-width range widens by 0.5 on each side.
        /// </summary>
        /// <param name="a">The raveled complex data.</param>
        /// <param name="range">The optional outer edges.</param>
        /// <returns>The outer edges.</returns>
        /// <exception cref="ValueError">A range whose lower edge exceeds its upper one; a non-finite range; a data minimum or
        ///     maximum with a non-finite part, printed as NumPy prints complex scalars
        ///     (<c>autodetected range of [(nan+0j), (nan+0j)] is not finite</c>, <c>[infj, (1+1j)]</c>).</exception>
        private static ComplexOuterEdges GetComplexOuterEdges(NDArray a, (double, double)? range)
        {
            if (range.HasValue)
            {
                (double first, double last) = range.Value;
                if (first > last)
                    throw new ValueError("max must be larger than min in range parameter.");
                if (!(IsFinite(first) && IsFinite(last)))
                    throw new ValueError($"supplied range of [{FormatEdge(first)}, {FormatEdge(last)}] is not finite");
                // Python floats: the widening is real arithmetic.
                if (first == last) { first -= 0.5; last += 0.5; }
                return new ComplexOuterEdges(first, last, strong: false);
            }
            if (a.size == 0)
                return new ComplexOuterEdges(0, 1, strong: false); // can't infer a range from no data

            using NDArray min = np.amin(a);
            using NDArray max = np.amax(a);
            Complex lo = min.GetAtIndex<Complex>(0), hi = max.GetAtIndex<Complex>(0);
            if (!(IsFinite(lo) && IsFinite(hi)))
                throw new ValueError($"autodetected range of [{min.ToString(false)}, {max.ToString(false)}] is not finite");
            if (lo == hi)
            {
                // complex128 scalar minus / plus a Python float: the float is (0.5 + 0j), so the imaginary part takes
                // `- 0.0` / `+ 0.0` (a -0.0 imaginary part becomes +0.0 on the upper edge, as in IEEE addition).
                lo = new Complex(lo.Real - 0.5, lo.Imaginary - 0.0);
                hi = new Complex(hi.Real + 0.5, hi.Imaginary + 0.0);
            }
            return new ComplexOuterEdges(lo, hi, strong: true);
        }

        /// <summary>
        ///     The equal-width tail of NumPy's <c>_get_bin_edges</c> for complex data: <c>bin_type = np.result_type(first_edge,
        ///     last_edge, a)</c> is complex128, the edges are its linspace, and the monotonic check compares them
        ///     lexicographically.
        /// </summary>
        /// <param name="nbins">The number of equal-width bins.</param>
        /// <param name="edges">The outer edges.</param>
        /// <returns>The complex128 edges with the parameters of the equal-width counting.</returns>
        /// <exception cref="ValueError">Two consecutive edges that do not increase (<c>Too many bins for data range. Cannot
        ///     create {nbins} finite-sized bins.</c>).</exception>
        private static BinEdgesResult ComplexEqualWidthResult(int nbins, ComplexOuterEdges edges)
        {
            NDArray binEdges = ComplexEqualWidthEdges(edges, nbins);
            if (AnyNonIncreasing(binEdges))
                throw new ValueError($"Too many bins for data range. Cannot create {nbins} finite-sized bins.");
            return new BinEdgesResult(binEdges, true, 0, 0, nbins, NPTypeCode.Complex, edges);
        }

        /// <summary>NumPy's <c>np.isfinite</c> of a complex scalar: both parts finite.</summary>
        /// <param name="z">The scalar.</param>
        /// <returns>False when either part is NaN or infinite.</returns>
        private static bool IsFinite(Complex z) => IsFinite(z.Real) && IsFinite(z.Imaginary);

        /// <summary>
        ///     The estimator branch of NumPy's <c>_get_bin_edges</c> for complex data: the outer edges, the data trimmed to a
        ///     range (lexicographic comparisons), then <c>width = selector(x)</c> and <c>int(np.ceil(delta / width))</c>, in
        ///     NumPy's order.
        /// </summary>
        /// <param name="a">The raveled complex data.</param>
        /// <param name="binName">A known estimator name.</param>
        /// <param name="range">The optional outer edges.</param>
        /// <param name="edges">The outer edges the equal-width edges are built from.</param>
        /// <returns>The number of equal-width bins: 1 when no data is left in the range or the width is zero.</returns>
        /// <exception cref="ValueError">The outer edges' errors (<see cref="GetComplexOuterEdges"/>); an inner histogram's
        ///     (<c>stone</c>).</exception>
        /// <exception cref="TypeError"><c>fd</c> / <c>auto</c>: <c>np.percentile</c>'s <c>a must be an array of real
        ///     numbers</c>. Every other estimator whose width is complex (<c>sqrt</c>, <c>sturges</c>, <c>rice</c>, <c>doane</c>,
        ///     <c>stone</c>: the peak-to-peak value is complex), and <c>scott</c> without a range (its width is real, the
        ///     range from the data's complex minimum and maximum is not): <c>ufunc 'ceil' not supported for the input types,
        ///     …</c>.</exception>
        /// <remarks>
        ///     A zero width ends in one bin before the ceil: <c>sqrt</c> / <c>sturges</c> / <c>rice</c> when every sample is
        ///     equal (the peak-to-peak value is <c>0j</c>), <c>doane</c> with two samples or fewer or no spread, <c>stone</c>
        ///     with one sample or no spread, <c>scott</c> with no spread. <c>stone</c> otherwise evaluates a histogram for
        ///     every candidate count before it picks one, as NumPy does, so an error of the complex counting surfaces from
        ///     there first.
        /// </remarks>
        private static int ComplexEstimatorBinCount(NDArray a, string binName, (double, double)? range, out ComplexOuterEdges edges)
        {
            edges = GetComplexOuterEdges(a, range);

            // Truncate the range if needed: `keep = (a >= first_edge); keep &= (a <= last_edge)`, lexicographic.
            NDArray x = a;
            if (range.HasValue)
            {
                using NDArray lo = edges.FirstScalar();
                using NDArray hi = edges.LastScalar();
                NDArray keep = (a >= lo) & (a <= hi);
                if (!AllTrue(keep))
                    x = a[keep];
            }
            if (x.size == 0)
                return 1;

            // The selectors, as they meet complex data. Only a width that is real AND a range of Python floats reach the
            // ceil as real numbers; any complex operand of `delta / width` makes np.ceil refuse.
            long n = x.size;
            switch (binName)
            {
                case "fd":
                case "auto":
                    // _hist_bin_fd: np.percentile(x, [75, 25]) refuses complex data ("a must be an array of real numbers").
                    np.percentile(x, new double[] { 75, 25 });
                    throw new InvalidOperationException("np.histogram: np.percentile accepted complex data.");
                case "scott":
                {
                    // np.std of complex data is real, so the width is real.
                    double width = EstimateBinWidth("scott", x, 0, 0, range);
                    if (width == 0)
                        return 1;
                    if (edges.Strong)
                        throw Backends.Strings.NDStringUfuncs.NotSupportedForInputTypes("ceil");
                    return (int)Math.Ceiling((edges.Last.Real - edges.First.Real) / width);
                }
                case "sqrt":
                case "sturges":
                case "rice":
                    // _ptp(x) / (a real number): complex, and zero only when the samples are all equal.
                    if (ComplexPtpIsZero(x))
                        return 1;
                    throw Backends.Strings.NDStringUfuncs.NotSupportedForInputTypes("ceil");
                case "doane":
                    // A complex width when there are more than two samples with a nonzero standard deviation (they are then
                    // not all equal, so the peak-to-peak value is nonzero); else the float 0.0.
                    if (n > 2)
                    {
                        using NDArray sigma = np.std(x);
                        if (sigma.GetAtIndex<double>(0) > 0.0)
                            throw Backends.Strings.NDStringUfuncs.NotSupportedForInputTypes("ceil");
                    }
                    return 1;
                case "stone":
                {
                    if (n <= 1 || ComplexPtpIsZero(x))
                        return 1;
                    // jhat(nbins) runs np.histogram(x, bins=nbins, range=(first_edge, last_edge)) for every candidate count
                    // before min() picks one; the count it picks then gives the complex width ptp_x / nbins. The outer edges
                    // are the data's own minimum and maximum (no range: x is a) or the caller's range, widened by 0.5 when it
                    // has no width, which the inner histogram widens the same way: the caller's range gives the same calls.
                    int upperBound = Math.Max(100, (int)Math.Sqrt(n));
                    for (int nbins = 1; nbins <= upperBound; nbins++)
                        Histogram(x, nbins, range, false, null);
                    throw Backends.Strings.NDStringUfuncs.NotSupportedForInputTypes("ceil");
                }
                default:
                    throw new ValueError($"'{binName}' is not a valid estimator for `bins`");
            }
        }

        /// <summary>Whether NumPy's <c>_ptp(x) = x.max() - x.min()</c> of complex data is <c>0j</c> (all samples equal).</summary>
        /// <param name="x">The non-empty complex data.</param>
        /// <returns>True when the lexicographic maximum and minimum are equal.</returns>
        private static bool ComplexPtpIsZero(NDArray x)
        {
            using NDArray max = np.amax(x);
            using NDArray min = np.amin(x);
            using NDArray ptp = np.subtract(max, min);
            return ptp.GetAtIndex<Complex>(0) == Complex.Zero;
        }

        /// <summary>
        ///     NumPy's <c>np.linspace(first_edge, last_edge, n_equal_bins + 1, endpoint=True, dtype=complex128)</c>: a float64
        ///     linspace cast to complex128 for Python-number edges, else linspace's own complex128 statements.
        /// </summary>
        /// <param name="edges">The outer edges.</param>
        /// <param name="nbins">The number of bins (the edges are one more).</param>
        /// <returns>The complex128 edges.</returns>
        /// <remarks>
        ///     The complex statements: <c>delta = stop - start</c>; <c>step = delta / div</c> (the scalar complex division,
        ///     Smith's method by <c>(div + 0j)</c>); <c>y = arange(num)</c> as complex128, then <c>y *= step</c> (the fused complex
        ///     product with a scalar operand), or, when the step underflows to zero, <c>y /= div; y *= delta</c>; then
        ///     <c>y += start</c> and <c>y[-1] = stop</c>.
        /// </remarks>
        private static NDArray ComplexEqualWidthEdges(ComplexOuterEdges edges, int nbins)
        {
            int num = nbins + 1;
            if (!edges.Strong)
            {
                // Python numbers: linspace computes in float64 (their result type) and casts to the requested complex128.
                using NDArray real = HistogramLinspace(edges.First.Real, edges.Last.Real, num, NPTypeCode.Double, NPTypeCode.Double);
                return real.astype(NPTypeCode.Complex);
            }

            int div = num - 1; // num >= 2
            using NDArray start = NDArray.Scalar(edges.First);
            using NDArray stop = NDArray.Scalar(edges.Last);
            using NDArray delta = np.subtract(stop, start);
            using NDArray divisor = NDArray.Scalar(new Complex(div, 0));
            using NDArray step = np.divide(delta, divisor);
            NDArray y = np.arange(num).astype(NPTypeCode.Complex);
            if (step.GetAtIndex<Complex>(0) == Complex.Zero)
            {
                // Special handling for denormal steps (NumPy's gh-5437): divide first, then scale by the whole delta.
                np.divide(y, divisor, @out: y);
                np.multiply(y, delta, @out: y);
            }
            else
                np.multiply(y, step, @out: y);
            np.add(y, start, @out: y);
            y.SetAtIndex(edges.Last, num - 1);
            return y;
        }

        /// <summary>
        ///     NumPy's equal-width counting loop over complex data, statement for statement: per block of 65536 samples, the
        ///     samples outside the outer edges dropped (lexicographic), each index computed in complex arithmetic and cast to
        ///     intp (the real part, truncated), corrected by one where a sample sits below its bin's lower edge or at or
        ///     above the next edge, and the block counted with <c>np.bincount</c>.
        /// </summary>
        /// <param name="raveled">The raveled complex data.</param>
        /// <param name="w">The raveled weights (castable to float64 or complex128), or null.</param>
        /// <param name="be">The complex equal-width edges, their count and the outer edges.</param>
        /// <param name="ntype">The result's dtype: int64 without weights, the weights' dtype with them.</param>
        /// <returns>The counts (or summed weights), one per bin.</returns>
        /// <exception cref="IndexError">An index outside the edges (a sample whose imaginary part pulls the complex quotient
        ///     past them): NumPy's fancy-index error.</exception>
        /// <exception cref="ArgumentException">A negative index left for <c>np.bincount</c> (its <c>'list' argument must have
        ///     no negative elements</c>, a ValueError in NumPy).</exception>
        /// <remarks>
        ///     The counts are NumPy's even where they are not a histogram of the real parts: the index is the real part of
        ///     <c>(x - first_edge) / (last_edge - first_edge) * n</c>, which mixes in the imaginary parts whenever the outer
        ///     edges have different imaginary parts.
        /// </remarks>
        private static NDArray HistogramUniformComplexData(NDArray raveled, NDArray w, BinEdgesResult be, NPTypeCode ntype)
        {
            long len = raveled.size;
            int nbins = be.NBins;
            NDArray binEdges = be.Edges;
            NDArray n = np.zeros(Shape.Vector(nbins), ntype);
            if (len == 0)
                return n;

            using NDArray firstEdge = be.ComplexEdges.FirstScalar();
            using NDArray lastEdge = be.ComplexEdges.LastScalar();
            // norm_numerator = n_equal_bins; norm_denom = _unsigned_subtract(last_edge, first_edge): complex128 for the data's
            // edges, float64 for a range's (no signed-to-unsigned mapping applies to either).
            using NDArray normDenom = np.subtract(lastEdge, firstEdge);
            using NDArray numerator = NDArray.Scalar((long)nbins);
            using NDArray one = NDArray.Scalar(1L);
            using NDArray lastIndex = NDArray.Scalar((long)nbins - 1);
            bool complexWeights = ntype == NPTypeCode.Complex;

            for (long start = 0; start < len; start += HistogramBlock)
            {
                long blockLen = Math.Min(HistogramBlock, len - start);
                string slice = $"{start}:{start + blockLen}";
                NDArray tmpA = raveled[slice];
                NDArray tmpW = w is null ? null : w[slice];

                // Only include values in the right range.
                NDArray keep = (tmpA >= firstEdge) & (tmpA <= lastEdge);
                if (!AllTrue(keep))
                {
                    tmpA = tmpA[keep];
                    if (tmpW is not null)
                        tmpW = tmpW[keep];
                }

                // tmp_a.astype(bin_edges.dtype) is complex128 already. The bin indices: the cast to intp keeps the real part.
                NDArray fIndices = np.multiply(np.divide(np.subtract(tmpA, firstEdge), normDenom), numerator);
                NDArray indices = fIndices.astype(NPTypeCode.Int64);
                // indices[indices == n_equal_bins] -= 1
                np.subtract(indices, one, @out: indices, where: indices == numerator);
                // decrement = tmp_a < bin_edges[indices]; indices[decrement] -= 1
                NDArray decrement = tmpA < binEdges[indices];
                np.subtract(indices, one, @out: indices, where: decrement);
                // increment = (tmp_a >= bin_edges[indices + 1]) & (indices != n_equal_bins - 1); indices[increment] += 1
                NDArray increment = (tmpA >= binEdges[indices + one]) & (indices != lastIndex);
                np.add(indices, one, @out: indices, where: increment);

                if (complexWeights)
                {
                    // n.real += bincount(..., weights=tmp_w.real); n.imag += bincount(..., weights=tmp_w.imag): componentwise.
                    NDArray re = bincount(indices, np.real(tmpW), nbins);
                    NDArray im = bincount(indices, np.imag(tmpW), nbins);
                    n = n + MakeComplex(re, im);
                }
                else
                {
                    // n += np.bincount(indices, weights=tmp_w, minlength=n_equal_bins).astype(ntype)
                    np.add(n, bincount(indices, tmpW, nbins).astype(ntype), @out: n);
                }
            }
            return n;
        }
    }
}
