using System;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from an F distribution.
        /// </summary>
        /// <param name="dfnum">Degrees of freedom in numerator. Must be > 0.</param>
        /// <param name="dfden">Degrees of freedom in denominator. Must be > 0.</param>
        /// <param name="size">Output shape. If null, a single value is returned.</param>
        /// <returns>Drawn samples from the F distribution.</returns>
        /// <exception cref="ArgumentException">If dfnum or dfden is <= 0.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.f.html
        ///     <br/>
        ///     The F distribution is the ratio of two chi-square variates:
        ///     F = (chi2_num / dfnum) / (chi2_den / dfden)
        ///     <br/>
        ///     Also known as the Fisher distribution. Used in ANOVA tests.
        ///     <br/>
        ///     For dfden > 2, mean = dfden / (dfden - 2).
        /// </remarks>
        public NDArray f(double dfnum, double dfden, Shape? size = null)
        {
            if (dfnum <= 0)
                throw new ArgumentException("dfnum <= 0", nameof(dfnum));
            if (dfden <= 0)
                throw new ArgumentException("dfden <= 0", nameof(dfden));

            if (size == null)
            {
                // Return scalar
                return NDArray.Scalar(FSample(dfnum, dfden));
            }

            return f(dfnum, dfden, size.Value.dimensions);
        }

        /// <summary>
        ///     Draw samples from an F distribution.
        /// </summary>
        /// <param name="dfnum">Degrees of freedom in numerator. Must be > 0.</param>
        /// <param name="dfden">Degrees of freedom in denominator. Must be > 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the F distribution.</returns>
        /// <exception cref="ArgumentException">If dfnum or dfden is <= 0.</exception>
        public NDArray f(double dfnum, double dfden, Shape size) => f(dfnum, dfden, size.dimensions);

        /// <summary>
        ///     Draw samples from an F distribution.
        /// </summary>
        /// <param name="dfnum">Degrees of freedom in numerator. Must be > 0.</param>
        /// <param name="dfden">Degrees of freedom in denominator. Must be > 0.</param>
        /// <param name="size">Output shape as a single dimension.</param>
        /// <returns>Drawn samples from the F distribution.</returns>
        /// <exception cref="ArgumentException">If dfnum or dfden is <= 0.</exception>
        public NDArray f(double dfnum, double dfden, int size) => f(dfnum, dfden, new[] { size });

        /// <summary>
        ///     Draw samples from an F distribution.
        /// </summary>
        /// <param name="dfnum">Degrees of freedom in numerator. Must be > 0.</param>
        /// <param name="dfden">Degrees of freedom in denominator. Must be > 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the F distribution.</returns>
        /// <exception cref="ArgumentException">If dfnum or dfden is <= 0.</exception>
        public NDArray f(double dfnum, double dfden, int[] size)
        {
            if (dfnum <= 0)
                throw new ArgumentException("dfnum <= 0", nameof(dfnum));
            if (dfden <= 0)
                throw new ArgumentException("dfden <= 0", nameof(dfden));

            // F = (chi2_num * dfden) / (chi2_den * dfnum)
            // where chi2 = 2 * gamma(df/2, 1) = gamma(df/2, 2)
            var chi2_num = chisquare(dfnum, size);
            var chi2_den = chisquare(dfden, size);

            // Element-wise: (chi2_num * dfden) / (chi2_den * dfnum)
            return (chi2_num * dfden) / (chi2_den * dfnum);
        }

        /// <summary>
        ///     Draw samples from an F distribution.
        /// </summary>
        /// <param name="dfnum">Degrees of freedom in numerator. Must be > 0.</param>
        /// <param name="dfden">Degrees of freedom in denominator. Must be > 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the F distribution.</returns>
        /// <exception cref="ArgumentException">If dfnum or dfden is <= 0.</exception>
        public NDArray f(double dfnum, double dfden, params long[] size)
            => f(dfnum, dfden, Array.ConvertAll(size, d => (int)d));

        /// <summary>
        ///     Generate a single sample from the F distribution.
        /// </summary>
        private double FSample(double dfnum, double dfden)
        {
            // Generate two chi-square samples
            double chi2_num = GammaSample(dfnum / 2) * 2;
            double chi2_den = GammaSample(dfden / 2) * 2;

            return (chi2_num * dfden) / (chi2_den * dfnum);
        }

        /// <summary>
        ///     Generate a single gamma sample using Marsaglia's method.
        /// </summary>
        private double GammaSample(double shape)
        {
            if (shape < 1)
            {
                double d = shape + 1.0 - 1.0 / 3.0;
                double c = (1.0 / 3.0) / Math.Sqrt(d);
                double u = randomizer.NextDouble();
                return MarsagliaSample(d, c) * Math.Pow(u, 1.0 / shape);
            }
            else
            {
                double d = shape - 1.0 / 3.0;
                double c = (1.0 / 3.0) / Math.Sqrt(d);
                return MarsagliaSample(d, c);
            }
        }

        /// <summary>
        ///     Single sample from Marsaglia's gamma method.
        /// </summary>
        private double MarsagliaSample(double d, double c)
        {
            while (true)
            {
                double x, t, v;

                do
                {
                    x = NextGaussian();
                    t = 1.0 + c * x;
                    v = t * t * t;
                } while (v <= 0);

                double U = randomizer.NextDouble();
                double x2 = x * x;

                if (U < 1 - 0.0331 * x2 * x2)
                    return d * v;

                if (Math.Log(U) < 0.5 * x2 + d * (1.0 - v + Math.Log(v)))
                    return d * v;
            }
        }
    }
}
