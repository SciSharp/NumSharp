using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     Draw samples from a standard Gamma distribution (scale = 1).
        /// </summary>
        /// <param name="shape">The shape parameter; sign bit must be clear (<c>-0.0</c> rejected), NaN propagates.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value; <c>()</c> returns a 0-d array.</param>
        /// <param name="dtype">Native <c>float64</c> (default) or <c>float32</c>.</param>
        /// <param name="out">Optional C-CONTIGUOUS output array (NumPy's <c>cont</c>/<c>cont_f</c> reject F order), returned when given.</param>
        /// <returns>The draws; a float64 0-d array for <c>size=None</c> even when <paramref name="dtype"/> is float32.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is not a native float32/float64, or <paramref name="out"/> has the wrong dtype.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is not C-contiguous/writeable or mis-shaped, or <paramref name="shape"/> is negative.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_gamma.html
        ///     <br/>Validation order follows NumPy's <c>cont</c>: dtype → <c>out</c> (<c>check_output</c> runs
        ///     first inside <c>cont</c>) → the <c>shape</c> constraint, so a call that is wrong in several ways
        ///     reports NumPy's first error.
        /// </remarks>
        public NDArray standard_gamma(double shape, Shape size = default, DType dtype = null, NDArray @out = null)
        {
            dtype ??= DType.Double;
            NPTypeCode tc = ResolveFloatDtype(dtype, "standard_gamma");
            if (@out is not null)
                ValidateOut(@out, size, tc, requireCContiguous: true);
            CheckNonNegative(shape, "shape");

            bool f32 = tc == NPTypeCode.Single;
            float shapeF = (float)shape;

            lock (_bitGenerator.@lock)
            {
                if (@out is not null)
                {
                    if (f32) FillFloatDistInto(@out, () => NextStandardGammaF(shapeF));
                    else FillDoubleDistInto(@out, () => NextStandardGamma(shape));
                    return @out;
                }

                if (IsNoSize(size))
                    // size=None returns a float64 scalar even for dtype=float32 (NumPy cont_f returns the C
                    // float, which Python widens to a float); only sized/()/out= stay float32.
                    return f32 ? NDArray.Scalar((double)NextStandardGammaF(shapeF)) : NDArray.Scalar(NextStandardGamma(shape));

                return f32
                    ? FillFloatDist(size, () => NextStandardGammaF(shapeF))
                    : FillDoubleDist(size, () => NextStandardGamma(shape));
            }
        }

        /// <summary>
        ///     Draw samples from a Gamma distribution.
        /// </summary>
        /// <param name="shape">The shape parameter; sign bit must be clear (<c>-0.0</c> rejected), NaN propagates.</param>
        /// <param name="scale">The scale parameter (default 1); sign bit must be clear, NaN propagates.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="ValueError"><paramref name="shape"/> or <paramref name="scale"/> is negative (checked in that order).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.gamma.html
        ///     <br/><c>scale * standard_gamma(shape)</c>, byte-identical to NumPy.
        /// </remarks>
        public NDArray gamma(double shape, double scale = 1.0, Shape size = default)
        {
            CheckNonNegative(shape, "shape");
            CheckNonNegative(scale, "scale");

            lock (_bitGenerator.@lock)
            {
                if (IsNoSize(size))
                    return NDArray.Scalar(scale * NextStandardGamma(shape));

                return FillDoubleDist(size, () => scale * NextStandardGamma(shape));
            }
        }
    }
}
