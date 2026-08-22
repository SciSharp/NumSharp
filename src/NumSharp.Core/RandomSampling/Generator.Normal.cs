using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     Draw samples from a standard Normal distribution (mean 0, stdev 1).
        /// </summary>
        /// <param name="size">Output shape. If default/scalar a single value is returned.</param>
        /// <param name="dtype">Desired dtype — <c>float64</c> (default) or <c>float32</c>.</param>
        /// <param name="out">Optional output array.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_normal.html
        ///     <br/>Uses NumPy's ziggurat sampler, so the stream matches <c>default_rng(seed).standard_normal(...)</c>.
        /// </remarks>
        public NDArray standard_normal(Shape size = default, Type dtype = null, NDArray @out = null)
        {
            dtype = dtype ?? typeof(double);
            NPTypeCode tc = ResolveFloatDtype(dtype, "standard_normal");

            if (@out is not null)
            {
                ValidateOut(@out, size, tc, "standard_normal");
                if (tc == NPTypeCode.Single) FillFloatDistInto(@out, NextStandardNormalF);
                else FillDoubleDistInto(@out, NextStandardNormal);
                return @out;
            }

            if (IsNoSize(size))
                return tc == NPTypeCode.Single
                    ? NDArray.Scalar(NextStandardNormalF())
                    : NDArray.Scalar(NextStandardNormal());

            return tc == NPTypeCode.Single
                ? FillFloatDist(size, NextStandardNormalF)
                : FillDoubleDist(size, NextStandardNormal);
        }

        /// <summary>
        ///     Draw samples from a normal (Gaussian) distribution.
        /// </summary>
        /// <param name="loc">Mean of the distribution.</param>
        /// <param name="scale">Standard deviation (must be non-negative).</param>
        /// <param name="size">Output shape.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html
        ///     <br/><c>loc + scale * standard_normal()</c>, byte-identical to NumPy.
        /// </remarks>
        public NDArray normal(double loc = 0.0, double scale = 1.0, Shape size = default)
        {
            if (scale < 0)
                throw new ValueError("scale < 0");

            if (IsNoSize(size))
                return NDArray.Scalar(loc + scale * NextStandardNormal());

            return FillDoubleDist(size, () => loc + scale * NextStandardNormal());
        }
    }
}
