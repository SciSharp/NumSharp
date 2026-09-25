using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     Draw samples from a standard Normal distribution (mean 0, stdev 1).
        /// </summary>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value; <c>()</c> returns a 0-d array.</param>
        /// <param name="dtype">Desired dtype — native <c>float64</c> (default) or <c>float32</c>.</param>
        /// <param name="out">Optional C- or F-contiguous output array, filled in memory order and returned.</param>
        /// <returns>The draws; a float64 0-d array for <c>size=None</c> even when <paramref name="dtype"/> is float32.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is not a native float32/float64, or <paramref name="out"/> has the wrong dtype.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is not contiguous/writeable, or its shape disagrees with <paramref name="size"/>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_normal.html
        ///     <br/>Uses NumPy's ziggurat sampler, so the stream matches <c>default_rng(seed).standard_normal(...)</c>.
        /// </remarks>
        public NDArray standard_normal(Shape size = default, DType dtype = null, NDArray @out = null)
        {
            dtype ??= DType.Double;
            NPTypeCode tc = ResolveFloatDtype(dtype, "standard_normal");

            lock (_bitGenerator.@lock)
            {
                if (@out is not null)
                {
                    ValidateOut(@out, size, tc, requireCContiguous: false);
                    if (tc == NPTypeCode.Single) FillFloatDistInto(@out, NextStandardNormalF);
                    else FillDoubleDistInto(@out, NextStandardNormal);
                    return @out;
                }

                if (IsNoSize(size))
                    // size=None returns a float64 scalar even for dtype=float32 (NumPy float_fill widens
                    // the float32 draw to a Python float); only sized/()/out= stay float32.
                    return tc == NPTypeCode.Single
                        ? NDArray.Scalar((double)NextStandardNormalF())
                        : NDArray.Scalar(NextStandardNormal());

                return tc == NPTypeCode.Single
                    ? FillFloatDist(size, NextStandardNormalF)
                    : FillDoubleDist(size, NextStandardNormal);
            }
        }

        /// <summary>
        ///     Draw samples from a normal (Gaussian) distribution.
        /// </summary>
        /// <param name="loc">Mean of the distribution.</param>
        /// <param name="scale">Standard deviation (sign bit must be clear; NaN propagates).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (NumPy's <c>scale &lt; 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html
        ///     <br/><c>loc + scale * standard_normal()</c>, byte-identical to NumPy.
        /// </remarks>
        public NDArray normal(double loc = 0.0, double scale = 1.0, Shape size = default)
        {
            CheckNonNegative(scale, "scale");

            lock (_bitGenerator.@lock)
            {
                if (IsNoSize(size))
                    return NDArray.Scalar(loc + scale * NextStandardNormal());

                return FillDoubleDist(size, () => loc + scale * NextStandardNormal());
            }
        }
    }
}
