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
                    if (tc == NPTypeCode.Single) FillFloatDistInto(@out, f);
                    else FillDoubleDistInto(@out, d);
                    return @out;
                }

                if (IsNoSize(size))
                    // size=None returns a float64 scalar even for dtype=float32 (NumPy float_fill widens
                    // the float32 draw to a Python float); only sized/()/out= stay float32.
                    return tc == NPTypeCode.Single ? NDArray.Scalar((double)f()) : NDArray.Scalar(d());

                return tc == NPTypeCode.Single ? FillFloatDist(size, f) : FillDoubleDist(size, d);
            }
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

                return FillDoubleDist(size, () => scale * NextStandardExponential());
            }
        }
    }
}
