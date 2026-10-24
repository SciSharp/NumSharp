using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using static NumSharp.np;

namespace NumSharp.Statistics
{
    internal static partial class HistogramEngine
    {
        // ---------------------------------------------------------------------
        //  The bin-width estimators in NumPy's dtypes
        // ---------------------------------------------------------------------
        //
        // NumPy's selectors (`_hist_bin_*` in numpy/lib/_histograms_impl.py) are Python statements over NumPy scalars, so
        // each operation runs in the dtype NEP 50 gives it: the peak-to-peak value and the range of float16 / float32 data
        // stay in the data's dtype (a float16 range past 65504 overflows), a Python float meeting such a scalar takes its
        // dtype (rice's and scott's widths are float16 / float32), and np.sqrt / np.log2 of Python numbers, np.percentile of
        // the data and every integer division are float64. The members below carry each value with its dtype through those
        // statements, and `_get_bin_edges`' `int(np.ceil(delta / width))` divides in the dtype the two give.

        /// <summary>
        ///     A value of NumPy's estimator statements with the dtype NumPy holds it in: <see cref="NPTypeCode.Half"/> /
        ///     <see cref="NPTypeCode.Single"/> / <see cref="NPTypeCode.Double"/> scalars (the value is exact in that dtype), or
        ///     <see cref="NPTypeCode.UInt64"/> for an integer: an integer data's exact peak-to-peak value or the Python int 1 of
        ///     a clamped width, held as its correctly rounded float64 (the only arithmetic an integer meets here is float64).
        /// </summary>
        private readonly struct NpScalar
        {
            /// <summary>The value (exact in <see cref="Dtype"/>, or the integer's correctly rounded float64).</summary>
            public readonly double Value;
            /// <summary>The dtype NumPy computes the value in: Half, Single, Double, or UInt64 for an integer.</summary>
            public readonly NPTypeCode Dtype;

            /// <summary>Holds a value with its NumPy dtype.</summary>
            /// <param name="value">The value.</param>
            /// <param name="dtype">Its dtype.</param>
            public NpScalar(double value, NPTypeCode dtype)
            {
                Value = value; Dtype = dtype;
            }

            /// <summary>A float64 scalar (np.float64, a Python float, the float64 of an integer division).</summary>
            /// <param name="value">The value.</param>
            /// <returns>The scalar.</returns>
            public static NpScalar Float64(double value) => new NpScalar(value, NPTypeCode.Double);

            /// <summary>Python's truthiness of the value: nonzero (a NaN is true), the <c>if width:</c> of NumPy's code.</summary>
            public bool IsTruthy => Value != 0;
        }

        /// <summary>
        ///     NumPy's <c>scalar / n</c> for a NumPy scalar and a Python int or float <paramref name="weak"/>: NEP 50 gives the
        ///     float scalar's dtype (the Python number converted to it first), and float64 to an integer scalar (true division).
        /// </summary>
        /// <param name="a">The NumPy scalar.</param>
        /// <param name="weak">The Python int or float.</param>
        /// <returns>The quotient in its dtype.</returns>
        private static NpScalar DivideByPython(NpScalar a, double weak) => a.Dtype switch
        {
            // HALF_divide: both operands widen to float32, the float32 quotient rounds to float16.
            NPTypeCode.Half => new NpScalar((double)(Half)((float)a.Value / (float)(Half)weak), NPTypeCode.Half),
            NPTypeCode.Single => new NpScalar((float)a.Value / (float)weak, NPTypeCode.Single),
            _ => NpScalar.Float64(a.Value / weak),
        };

        /// <summary>
        ///     NumPy's quotient of two NumPy scalars (<c>delta / width</c>): float16 / float16 and float32 / float32 stay in
        ///     their dtype, every other pair (one of them float64, or an integer) divides in float64.
        /// </summary>
        /// <param name="a">The dividend.</param>
        /// <param name="b">The divisor.</param>
        /// <returns>The quotient as a double holding its exact value in the result dtype.</returns>
        private static double DivideScalars(NpScalar a, NpScalar b)
        {
            if (a.Dtype == NPTypeCode.Half && b.Dtype == NPTypeCode.Half)
                return (double)(Half)((float)a.Value / (float)b.Value);
            if (a.Dtype == NPTypeCode.Single && b.Dtype == NPTypeCode.Single)
                return (float)a.Value / (float)b.Value;
            return a.Value / b.Value;
        }

        /// <summary>The class of the data's dtype in NumPy's estimator statements.</summary>
        /// <param name="x">The raveled data (bool data were made uint8; NumSharp's char is the oracle's uint16).</param>
        /// <returns>Half, Single, Double, or UInt64 for every integer dtype.</returns>
        private static NPTypeCode EstimatorDtype(NDArray x) => x.typecode switch
        {
            NPTypeCode.Half => NPTypeCode.Half,
            NPTypeCode.Single => NPTypeCode.Single,
            NPTypeCode.Double => NPTypeCode.Double,
            _ => NPTypeCode.UInt64,
        };

        /// <summary>
        ///     NumPy's <c>_ptp(x) = _unsigned_subtract(x.max(), x.min())</c>: an integer data's exact difference (an unsigned
        ///     subtraction, which no signed range overflows), a float data's subtraction in its own dtype (float16 through
        ///     float32, rounded back, so a float16 range past 65504 is inf).
        /// </summary>
        /// <param name="x">The non-empty data.</param>
        /// <returns>The peak-to-peak value with its dtype.</returns>
        private static NpScalar PtpScalar(NDArray x)
        {
            using NDArray max = np.amax(x);
            using NDArray min = np.amin(x);
            switch (EstimatorDtype(x))
            {
                case NPTypeCode.Half:
                    return new NpScalar((double)(Half)((float)max.GetAtIndex<Half>(0) - (float)min.GetAtIndex<Half>(0)), NPTypeCode.Half);
                case NPTypeCode.Single:
                    return new NpScalar(max.GetAtIndex<float>(0) - min.GetAtIndex<float>(0), NPTypeCode.Single);
                case NPTypeCode.Double:
                    return NpScalar.Float64(max.GetAtIndex<double>(0) - min.GetAtIndex<double>(0));
                default:
                    // The exact integer difference, converted to float64 once (Python's int-to-float is correctly rounded).
                    BigInteger difference = IntegerScalar(max) - IntegerScalar(min);
                    return new NpScalar(PyScalar.IntTrueDivide(difference, BigInteger.One), NPTypeCode.UInt64);
            }
        }

        /// <summary>
        ///     NumPy's <c>delta = _unsigned_subtract(last_edge, first_edge)</c> for the data's own outer edges (no range):
        ///     <c>a.max() - a.min()</c> as <see cref="PtpScalar"/> computes it, or, for data with no spread,
        ///     the difference of <c>_get_outer_edges</c>' widened edges <c>a.min() - 0.5</c> and <c>a.max() + 0.5</c> — computed
        ///     as NEP 50 computes them, in the float16 / float32 data's own dtype, in float64 for float64 and integer data (an
        ///     integer scalar minus a Python float is a float64).
        /// </summary>
        /// <param name="a">The non-empty raveled data.</param>
        /// <returns>The difference with its dtype.</returns>
        /// <remarks>
        ///     The widened edges round in a narrow dtype: a float32 1e9 widens to 1e9 itself on both sides (a zero difference,
        ///     so an estimator's <c>int(np.ceil(delta / width))</c> is 0 bins), a float16 1025 to 1024 and 1026 (a difference of
        ///     2). The difference matters only when an estimator finds a nonzero width for data with no spread, which
        ///     <c>scott</c> does: np.std of identical values is the rounding error of their mean (np.std of three float64 0.1
        ///     is 1.39e-17).
        /// </remarks>
        private static NpScalar DataEdgesDelta(NDArray a)
        {
            using NDArray max = np.amax(a);
            using NDArray min = np.amin(a);
            switch (EstimatorDtype(a))
            {
                case NPTypeCode.Half:
                {
                    float hi = (float)max.GetAtIndex<Half>(0), lo = (float)min.GetAtIndex<Half>(0);
                    if (hi == lo)
                    {
                        // first_edge - 0.5 / last_edge + 0.5: half scalar math (float32, rounded to float16).
                        lo = (float)(Half)(lo - 0.5f);
                        hi = (float)(Half)(hi + 0.5f);
                    }
                    return new NpScalar((double)(Half)(hi - lo), NPTypeCode.Half);
                }
                case NPTypeCode.Single:
                {
                    float hi = max.GetAtIndex<float>(0), lo = min.GetAtIndex<float>(0);
                    if (hi == lo)
                    {
                        lo -= 0.5f;
                        hi += 0.5f;
                    }
                    return new NpScalar(hi - lo, NPTypeCode.Single);
                }
                case NPTypeCode.Double:
                {
                    double hi = max.GetAtIndex<double>(0), lo = min.GetAtIndex<double>(0);
                    if (hi == lo)
                    {
                        lo -= 0.5;
                        hi += 0.5;
                    }
                    return NpScalar.Float64(hi - lo);
                }
                default:
                {
                    BigInteger hi = IntegerScalar(max), lo = IntegerScalar(min);
                    if (hi == lo)
                    {
                        // np.int64(c) - 0.5: the integer converted to float64 (correctly rounded), then the Python float.
                        double c = PyScalar.IntTrueDivide(hi, BigInteger.One);
                        return NpScalar.Float64((c + 0.5) - (c - 0.5));
                    }
                    return new NpScalar(PyScalar.IntTrueDivide(hi - lo, BigInteger.One), NPTypeCode.UInt64);
                }
            }
        }

        /// <summary>The exact value of a 0-d integer array (every integer dtype, and NumSharp's char as its code unit).</summary>
        /// <param name="scalar">The 0-d array.</param>
        /// <returns>The value.</returns>
        /// <exception cref="NotSupportedException">A dtype that is not an integer (unreachable: the caller checked).</exception>
        private static BigInteger IntegerScalar(NDArray scalar) => scalar.typecode switch
        {
            NPTypeCode.Byte => scalar.GetAtIndex<byte>(0),
            NPTypeCode.SByte => scalar.GetAtIndex<sbyte>(0),
            NPTypeCode.Int16 => scalar.GetAtIndex<short>(0),
            NPTypeCode.UInt16 => scalar.GetAtIndex<ushort>(0),
            NPTypeCode.Char => scalar.GetAtIndex<char>(0),
            NPTypeCode.Int32 => scalar.GetAtIndex<int>(0),
            NPTypeCode.UInt32 => scalar.GetAtIndex<uint>(0),
            NPTypeCode.Int64 => scalar.GetAtIndex<long>(0),
            NPTypeCode.UInt64 => scalar.GetAtIndex<ulong>(0),
            _ => throw new NotSupportedException($"np.histogram: {scalar.dtype} is not an integer dtype."),
        };

        /// <summary>
        ///     NumPy's bin-width selectors (<c>_hist_bin_selectors[name](x, (first_edge, last_edge))</c>) for integer, float16,
        ///     float32 and float64 data, each statement in NumPy's dtype.
        /// </summary>
        /// <param name="name">A known estimator name.</param>
        /// <param name="x">The range-trimmed, non-empty data.</param>
        /// <param name="range">The caller's range, or null: stone's histograms are taken over it (NumPy passes the outer
        ///     edges, which give the same histograms).</param>
        /// <returns>The width with its dtype; a zero (Python's 0 / 0.0) means one bin.</returns>
        /// <exception cref="ValueError">An unknown name (unreachable: the caller checked it first); an error of stone's
        ///     histograms.</exception>
        private static NpScalar EstimateWidthNumPy(string name, NDArray x, (double, double)? range)
        {
            switch (name)
            {
                case "sqrt": return SqrtWidth(x);
                case "sturges": return SturgesWidth(x);
                case "rice": return RiceWidth(x);
                case "scott": return ScottWidth(x);
                case "fd": return FdWidth(x);
                case "doane": return DoaneWidth(x);
                case "stone": return StoneWidth(x, range);
                case "auto":
                {
                    // fd_bw_corrected = max(fd_bw, sqrt_bw / 2); return min(fd_bw_corrected, sturges_bw): Python's max / min
                    // keep the first operand unless the second compares greater / smaller. All three are float64.
                    NpScalar fd = FdWidth(x), sturges = SturgesWidth(x), sqrt = SqrtWidth(x);
                    double halfSqrt = sqrt.Value / 2;
                    double corrected = halfSqrt > fd.Value ? halfSqrt : fd.Value;
                    return NpScalar.Float64(sturges.Value < corrected ? sturges.Value : corrected);
                }
                default:
                    throw new ValueError($"'{name}' is not a valid estimator for `bins`");
            }
        }

        /// <summary><c>_hist_bin_sqrt</c>: <c>_ptp(x) / np.sqrt(x.size)</c>, a float64 (np.sqrt gives a float64 scalar).</summary>
        /// <param name="x">The non-empty data.</param>
        /// <returns>The width.</returns>
        private static NpScalar SqrtWidth(NDArray x) => NpScalar.Float64(PtpScalar(x).Value / Math.Sqrt(x.size));

        /// <summary><c>_hist_bin_sturges</c>: <c>_ptp(x) / (np.log2(x.size) + 1.0)</c>, a float64.</summary>
        /// <param name="x">The non-empty data.</param>
        /// <returns>The width.</returns>
        private static NpScalar SturgesWidth(NDArray x) => NpScalar.Float64(PtpScalar(x).Value / (Math.Log2(x.size) + 1.0));

        /// <summary>
        ///     <c>_hist_bin_rice</c>: <c>_ptp(x) / (2.0 * x.size ** (1.0 / 3))</c>; the denominator is a Python float, so a
        ///     float16 / float32 peak-to-peak value keeps its dtype.
        /// </summary>
        /// <param name="x">The non-empty data.</param>
        /// <returns>The width.</returns>
        private static NpScalar RiceWidth(NDArray x) => DivideByPython(PtpScalar(x), 2.0 * Math.Pow(x.size, 1.0 / 3));

        /// <summary>
        ///     <c>_hist_bin_scott</c>: <c>(24.0 * np.pi**0.5 / x.size)**(1.0 / 3.0) * np.std(x)</c>; the factor is a Python
        ///     float, so the width has np.std's dtype (float16 / float32 for such data, float64 otherwise).
        /// </summary>
        /// <param name="x">The non-empty data (complex data's np.std is float64).</param>
        /// <returns>The width.</returns>
        private static NpScalar ScottWidth(NDArray x)
        {
            double factor = Math.Pow(24.0 * Math.Pow(Math.PI, 0.5) / x.size, 1.0 / 3.0);
            using NDArray std = np.std(x);
            return std.typecode switch
            {
                NPTypeCode.Half => new NpScalar((double)(Half)((float)(Half)factor * (float)std.GetAtIndex<Half>(0)), NPTypeCode.Half),
                NPTypeCode.Single => new NpScalar((float)factor * std.GetAtIndex<float>(0), NPTypeCode.Single),
                _ => NpScalar.Float64(factor * std.GetAtIndex<double>(0)),
            };
        }

        /// <summary>
        ///     <c>_hist_bin_fd</c>: <c>2.0 * iqr * x.size ** (-1.0 / 3.0)</c> with <c>iqr = np.subtract(*np.percentile(x,
        ///     [75, 25]))</c>; np.percentile is float64 for every data dtype, so the width is float64.
        /// </summary>
        /// <param name="x">The non-empty real data.</param>
        /// <returns>The width.</returns>
        /// <exception cref="TypeError">Complex data (np.percentile: <c>a must be an array of real numbers</c>).</exception>
        private static NpScalar FdWidth(NDArray x)
        {
            using NDArray q = np.percentile(x, new double[] { 75, 25 });
            double iqr = q.GetAtIndex<double>(0) - q.GetAtIndex<double>(1);
            return NpScalar.Float64(2.0 * iqr * Math.Pow(x.size, -1.0 / 3.0));
        }

        /// <summary>
        ///     <c>_hist_bin_doane</c>: with more than two samples and a positive standard deviation, the skewness
        ///     <c>g1 = mean(((x - mean(x)) / std(x)) ** 3)</c> computed in place in the data's float dtype (float64 for integer
        ///     data), then <c>_ptp(x) / (1.0 + np.log2(x.size) + np.log2(1.0 + np.absolute(g1) / sg1))</c>, a float64; else
        ///     the float 0.0.
        /// </summary>
        /// <param name="x">The non-empty data.</param>
        /// <returns>The width.</returns>
        private static NpScalar DoaneWidth(NDArray x)
        {
            long n = x.size;
            if (n > 2)
            {
                double sg1 = Math.Sqrt(6.0 * (n - 2) / ((n + 1.0) * (n + 3)));
                using NDArray sigma = np.std(x);
                if (sigma.astype(NPTypeCode.Double).GetAtIndex<double>(0) > 0.0)
                {
                    // temp = x - np.mean(x); np.true_divide(temp, sigma, temp); np.power(temp, 3, temp); g1 = np.mean(temp).
                    using NDArray mean = np.mean(x);
                    using NDArray temp = np.subtract(x, mean);
                    np.true_divide(temp, sigma, @out: temp);
                    using (NDArray three = NDArray.Scalar(3))
                        np.power(temp, three, @out: temp, dtype: temp.dtype); // a Python int exponent: the data's loop
                    using NDArray g1 = np.mean(temp);
                    double absG1 = Math.Abs(g1.astype(NPTypeCode.Double).GetAtIndex<double>(0));
                    double denominator = 1.0 + Math.Log2(n) + Math.Log2(1.0 + absG1 / sg1);
                    return NpScalar.Float64(PtpScalar(x).Value / denominator);
                }
            }
            return NpScalar.Float64(0.0);
        }

        /// <summary>
        ///     <c>_hist_bin_stone</c>: the bin count of 1 .. <c>max(100, int(np.sqrt(n)))</c> minimizing
        ///     <c>jhat(nbins) = (2 - (n + 1) * p_k.dot(p_k)) / (ptp_x / nbins)</c> with <c>p_k = np.histogram(x, bins=nbins,
        ///     range=...)[0] / n</c> (Python's min: the first value smaller than every one before), then <c>ptp_x / nbins</c> in
        ///     the peak-to-peak value's dtype; the int 0 for one sample or no spread.
        /// </summary>
        /// <param name="x">The non-empty data.</param>
        /// <param name="range">The caller's range, or null (the histograms then take the data's minimum and maximum, the
        ///     strong outer edges NumPy passes).</param>
        /// <returns>The width.</returns>
        /// <exception cref="ValueError">An error of the candidate histograms.</exception>
        /// <remarks>
        ///     <c>p_k.dot(p_k)</c> is NumPy's cblas ddot; NumSharp's np.dot gives its bits with the OpenBLAS backend and the
        ///     managed sum's otherwise, which differ in the last bit on 16 bins or more: two candidate counts whose jhat values
        ///     are that close can then be picked differently.
        /// </remarks>
        private static NpScalar StoneWidth(NDArray x, (double, double)? range)
        {
            long n = x.size;
            NpScalar ptp = PtpScalar(x);
            if (n <= 1 || ptp.Value == 0)
                return new NpScalar(0, NPTypeCode.UInt64); // Python's 0

            int upperBound = Math.Max(100, (int)Math.Sqrt(n));
            int best = 1;
            double bestJ = 0;
            for (int nbins = 1; nbins <= upperBound; nbins++)
            {
                NpScalar hh = DivideByPython(ptp, nbins);
                double j;
                HistogramResult h = Histogram(x, nbins, range, false, null);
                using (h.hist)
                using (h.bin_edges)
                using (NDArray count = NDArray.Scalar((double)n))
                using (NDArray pk = np.divide(h.hist, count))
                using (NDArray dot = np.dot(pk, pk))
                    j = (2 - (n + 1) * dot.GetAtIndex<double>(0)) / hh.Value;
                if (nbins == 1 || j < bestJ)
                {
                    bestJ = j;
                    best = nbins;
                }
            }
            return DivideByPython(ptp, best);
        }
    }
}
