using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a noncentral chi-square distribution.
        /// </summary>
        /// <param name="df">Degrees of freedom, must be > 0.</param>
        /// <param name="nonc">Non-centrality parameter, must be >= 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized noncentral chi-square distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.noncentral_chisquare.html
        ///     <br/>
        ///     The noncentral chi-square distribution is a generalization of the chi-square distribution.
        ///     <br/>
        ///     Mean = df + nonc
        /// </remarks>
        public NDArray noncentral_chisquare(double df, double nonc, Shape size)
        {
            if (df <= 0)
                throw new ArgumentException("df <= 0", nameof(df));
            if (nonc < 0)
                throw new ArgumentException("nonc < 0", nameof(nonc));

            var ret = new NDArray<double>(size);
            ArraySlice<double> data = ret.Data<double>();

            for (int i = 0; i < ret.size; i++)
            {
                data[i] = SampleNoncentralChisquare(df, nonc);
            }

            return ret;
        }

        /// <summary>
        ///     Draw samples from a noncentral chi-square distribution.
        /// </summary>
        /// <param name="df">Degrees of freedom, must be > 0.</param>
        /// <param name="nonc">Non-centrality parameter, must be >= 0.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the parameterized noncentral chi-square distribution.</returns>
        public NDArray noncentral_chisquare(double df, double nonc, int[] size)
            => noncentral_chisquare(df, nonc, new Shape(size));

        /// <summary>
        ///     Draw samples from a noncentral chi-square distribution.
        /// </summary>
        /// <param name="df">Degrees of freedom, must be > 0.</param>
        /// <param name="nonc">Non-centrality parameter, must be >= 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized noncentral chi-square distribution.</returns>
        public NDArray noncentral_chisquare(double df, double nonc, params long[] size)
            => noncentral_chisquare(df, nonc, new Shape(size));

        /// <summary>
        ///     Draw samples from a noncentral chi-square distribution.
        /// </summary>
        /// <param name="df">Degrees of freedom, must be > 0.</param>
        /// <param name="nonc">Non-centrality parameter, must be >= 0.</param>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the parameterized noncentral chi-square distribution.</returns>
        public NDArray noncentral_chisquare(double df, double nonc, int size)
            => noncentral_chisquare(df, nonc, new Shape(size));

        /// <summary>
        ///     Draw a single sample from a noncentral chi-square distribution.
        /// </summary>
        /// <param name="df">Degrees of freedom, must be > 0.</param>
        /// <param name="nonc">Non-centrality parameter, must be >= 0.</param>
        /// <returns>A single sample from the noncentral chi-square distribution.</returns>
        public double noncentral_chisquare(double df, double nonc)
        {
            if (df <= 0)
                throw new ArgumentException("df <= 0", nameof(df));
            if (nonc < 0)
                throw new ArgumentException("nonc < 0", nameof(nonc));

            return SampleNoncentralChisquare(df, nonc);
        }

        /// <summary>
        ///     Sample from the noncentral chi-square distribution.
        /// </summary>
        /// <remarks>
        ///     NumPy algorithm from distributions.c:
        ///     - If nonc == 0: return chisquare(df)
        ///     - If df > 1: return chisquare(df-1) + (N(0,1) + sqrt(nonc))^2
        ///     - If df <= 1: return chisquare(df + 2*Poisson(nonc/2))
        /// </remarks>
        private double SampleNoncentralChisquare(double df, double nonc)
        {
            if (nonc == 0)
            {
                // Central chi-square
                return SampleChisquare(df);
            }

            if (df > 1)
            {
                // df > 1: Chi2(df-1) + (N(0,1) + sqrt(nonc))^2
                double chi2 = SampleChisquare(df - 1);
                double n = NextGaussian() + Math.Sqrt(nonc);
                return chi2 + n * n;
            }
            else
            {
                // df <= 1: Chi2(df + 2*Poisson(nonc/2))
                int i = Knuth(nonc / 2.0);
                return SampleChisquare(df + 2 * i);
            }
        }

        /// <summary>
        ///     Sample a single value from the chi-square distribution.
        ///     Chi-square(df) = 2 * standard_gamma(df/2)
        /// </summary>
        private double SampleChisquare(double df)
        {
            return 2.0 * SampleStandardGamma(df / 2.0);
        }

        // Note: SampleStandardGamma() and SampleMarsaglia() are defined in np.random.standard_t.cs
    }
}
