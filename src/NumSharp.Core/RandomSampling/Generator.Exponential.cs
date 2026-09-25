using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     Draw samples from the standard exponential distribution.
        /// </summary>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value; <c>()</c> returns a 0-d array.</param>
        /// <param name="dtype">Native <c>float64</c> (default) or <c>float32</c>.</param>
        /// <param name="method"><c>"zig"</c> (ziggurat, default); any other string selects the inverse-CDF sampler, as in NumPy.</param>
        /// <param name="out">Optional C- or F-contiguous output array, filled in memory order and returned.</param>
        /// <returns>The draws; a float64 0-d array for <c>size=None</c> even when <paramref name="dtype"/> is float32.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is not a native float32/float64, or <paramref name="out"/> has the wrong dtype.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is not contiguous/writeable, or its shape disagrees with <paramref name="size"/>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_exponential.html
        /// </remarks>
        public NDArray standard_exponential(Shape size = default, DType dtype = null, string method = "zig", NDArray @out = null)
        {
            dtype ??= DType.Double;
            NPTypeCode tc = ResolveFloatDtype(dtype, "standard_exponential");
            bool zig = method == "zig"; // NumPy: anything not 'zig' uses the inverse-CDF sampler.

            Func<double> d = zig ? NextStandardExponential : (Func<double>)NextStandardExponentialInv;
            Func<float> f = zig ? NextStandardExponentialF : (Func<float>)NextStandardExponentialInvF;

            lock (_bitGenerator.@lock)
            {
                if (@out is not null)
                {
                    ValidateOut(@out, size, tc, requireCContiguous: false);
                    unsafe
                    {
                        FillStandardExponentialInto(OutStart(@out), @out.size, tc, zig);
                    }
                    return @out;
                }

                if (IsNoSize(size))
                    // size=None returns a float64 scalar even for dtype=float32 (NumPy float_fill widens
                    // the float32 draw to a Python float); only sized/()/out= stay float32.
                    return tc == NPTypeCode.Single ? NDArray.Scalar((double)f()) : NDArray.Scalar(d());

                var ret = new NDArray(tc == NPTypeCode.Single ? typeof(float) : typeof(double), size, false);
                unsafe
                {
                    FillStandardExponentialInto((byte*)ret.Address, ret.size, tc, zig);
                }
                return ret;
            }
        }

        /// <summary>
        ///     Fills <paramref name="n"/> standard exponentials in the loop dtype: the ziggurat through its read-ahead fill, or
        ///     NumPy's <c>method='inv'</c> sampler — one uniform per output, so a bulk uniform fill transformed in place
        ///     (<c>-log1p(-u)</c>; the float32 form takes the double log1p of the float draw, as NumPy's
        ///     <c>random_standard_exponential_inv_fill_f</c> does).
        /// </summary>
        /// <param name="p">The destination (float32 or float64 per <paramref name="tc"/>).</param>
        /// <param name="n">The number of draws.</param>
        /// <param name="tc">The loop dtype.</param>
        /// <param name="zig">True for the ziggurat, false for the inverse-CDF sampler.</param>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        private unsafe void FillStandardExponentialInto(byte* p, long n, NPTypeCode tc, bool zig)
        {
            if (tc == NPTypeCode.Single)
            {
                var fp = (float*)p;
                if (zig)
                {
                    FillStandardExponentialF(fp, n);
                    return;
                }
                _bitGenerator.FillFloat(fp, n);
                for (long i = 0; i < n; i++)
                    fp[i] = (float)(-Log1p(-(double)fp[i]));
                return;
            }
            var dp = (double*)p;
            if (zig)
            {
                FillStandardExponential(dp, n);
                return;
            }
            _bitGenerator.FillDouble(dp, n);
            for (long i = 0; i < n; i++)
                dp[i] = -Log1p(-dp[i]);
        }

        /// <summary>
        ///     Draw samples from an exponential distribution.
        /// </summary>
        /// <param name="scale">The scale parameter (1/rate); sign bit must be clear, NaN propagates.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (NumPy's <c>scale &lt; 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html
        ///     <br/><c>scale * standard_exponential()</c> (ziggurat), byte-identical to NumPy.
        /// </remarks>
        public NDArray exponential(double scale = 1.0, Shape size = default)
        {
            CheckNonNegative(scale, "scale");

            lock (_bitGenerator.@lock)
            {
                if (IsNoSize(size))
                    return NDArray.Scalar(scale * NextStandardExponential());

                var ret = new NDArray(typeof(double), size, false);
                unsafe
                {
                    var p = (double*)ret.Address;
                    long n = ret.size;
                    FillStandardExponential(p, n);
                    for (long i = 0; i < n; i++)
                        p[i] = scale * p[i];
                }
                return ret;
            }
        }
    }
}
