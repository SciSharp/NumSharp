using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     Draw samples from the standard exponential distribution.
        /// </summary>
        /// <param name="size">Output shape.</param>
        /// <param name="dtype"><c>float64</c> (default) or <c>float32</c>.</param>
        /// <param name="method">Either <c>"zig"</c> (ziggurat, default) or <c>"inv"</c> (inverse CDF).</param>
        /// <param name="out">Optional output array.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_exponential.html
        /// </remarks>
        public NDArray standard_exponential(Shape size = default, Type dtype = null, string method = "zig", NDArray @out = null)
        {
            dtype = dtype ?? typeof(double);
            NPTypeCode tc = ResolveFloatDtype(dtype, "standard_exponential");
            bool zig = method == "zig"; // NumPy: anything not 'zig' uses the inverse-CDF sampler.

            Func<double> d = zig ? NextStandardExponential : (Func<double>)NextStandardExponentialInv;
            Func<float> f = zig ? NextStandardExponentialF : (Func<float>)NextStandardExponentialInvF;

            if (@out is not null)
            {
                ValidateOut(@out, size, tc, "standard_exponential");
                if (tc == NPTypeCode.Single) FillFloatDistInto(@out, f);
                else FillDoubleDistInto(@out, d);
                return @out;
            }

            if (IsNoSize(size))
                // size=None returns a float64 scalar even for dtype=float32 (NumPy float_fill widens
                // the float32 draw to a Python float); only sized/out= stay float32.
                return tc == NPTypeCode.Single ? NDArray.Scalar((double)f()) : NDArray.Scalar(d());

            return tc == NPTypeCode.Single ? FillFloatDist(size, f) : FillDoubleDist(size, d);
        }

        /// <summary>
        ///     Draw samples from an exponential distribution.
        /// </summary>
        /// <param name="scale">The scale parameter (1/rate). Must be non-negative.</param>
        /// <param name="size">Output shape.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html
        ///     <br/><c>scale * standard_exponential()</c> (ziggurat), byte-identical to NumPy.
        /// </remarks>
        public NDArray exponential(double scale = 1.0, Shape size = default)
        {
            if (scale < 0)
                throw new ValueError("scale < 0");

            if (IsNoSize(size))
                return NDArray.Scalar(scale * NextStandardExponential());

            return FillDoubleDist(size, () => scale * NextStandardExponential());
        }
    }
}
