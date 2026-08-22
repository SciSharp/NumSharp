using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     Draw samples from a standard Gamma distribution (scale = 1).
        /// </summary>
        /// <param name="shape">The shape parameter (must be non-negative).</param>
        /// <param name="size">Output shape.</param>
        /// <param name="dtype"><c>float64</c> (default). <c>float32</c> is not yet supported.</param>
        /// <param name="out">Optional output array.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_gamma.html
        /// </remarks>
        public NDArray standard_gamma(double shape, Shape size = default, Type dtype = null, NDArray @out = null)
        {
            dtype = dtype ?? typeof(double);
            NPTypeCode tc = ResolveFloatDtype(dtype, "standard_gamma");
            if (tc != NPTypeCode.Double)
                throw new NotSupportedException("standard_gamma currently supports only float64 in NumSharp; float32 is not yet ported.");
            if (shape < 0)
                throw new ValueError("shape < 0");

            if (@out is not null)
            {
                ValidateOut(@out, size, tc, "standard_gamma");
                FillDoubleDistInto(@out, () => NextStandardGamma(shape));
                return @out;
            }

            if (IsNoSize(size))
                return NDArray.Scalar(NextStandardGamma(shape));

            return FillDoubleDist(size, () => NextStandardGamma(shape));
        }

        /// <summary>
        ///     Draw samples from a Gamma distribution.
        /// </summary>
        /// <param name="shape">The shape parameter (must be non-negative).</param>
        /// <param name="scale">The scale parameter (must be non-negative). Default 1.</param>
        /// <param name="size">Output shape.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.gamma.html
        ///     <br/><c>scale * standard_gamma(shape)</c>, byte-identical to NumPy.
        /// </remarks>
        public NDArray gamma(double shape, double scale = 1.0, Shape size = default)
        {
            if (shape < 0)
                throw new ValueError("shape < 0");
            if (scale < 0)
                throw new ValueError("scale < 0");

            if (IsNoSize(size))
                return NDArray.Scalar(scale * NextStandardGamma(shape));

            return FillDoubleDist(size, () => scale * NextStandardGamma(shape));
        }
    }
}
