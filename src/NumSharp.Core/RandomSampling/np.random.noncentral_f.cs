using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from the noncentral F distribution.
        /// </summary>
        /// <param name="dfnum">Numerator degrees of freedom, must be > 0.</param>
        /// <param name="dfden">Denominator degrees of freedom, must be > 0.</param>
        /// <param name="nonc">Non-centrality parameter, must be >= 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized noncentral F distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.noncentral_f.html
        ///     <br/>
        ///     When calculating the power of an experiment, the non-central F statistic
        ///     becomes important. When the null hypothesis is true, the F statistic follows
        ///     a central F distribution. When the null hypothesis is not true, it follows
        ///     a non-central F distribution.
        /// </remarks>
        public NDArray noncentral_f(double dfnum, double dfden, double nonc, Shape size)
            => noncentral_f(dfnum, dfden, nonc, size.dimensions);

        /// <summary>
        ///     Draw samples from the noncentral F distribution.
        /// </summary>
        /// <param name="dfnum">Numerator degrees of freedom, must be > 0.</param>
        /// <param name="dfden">Denominator degrees of freedom, must be > 0.</param>
        /// <param name="nonc">Non-centrality parameter, must be >= 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized noncentral F distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.noncentral_f.html
        /// </remarks>
        public NDArray noncentral_f(double dfnum, double dfden, double nonc, params int[] size)
        {
            // Parameter validation (matches NumPy error messages)
            if (dfnum <= 0)
                throw new ArgumentException("dfnum <= 0", nameof(dfnum));
            if (dfden <= 0)
                throw new ArgumentException("dfden <= 0", nameof(dfden));
            if (nonc < 0)
                throw new ArgumentException("nonc < 0", nameof(nonc));

            if (size == null || size.Length == 0)
            {
                return NDArray.Scalar(SampleNoncentralF(dfnum, dfden, nonc));
            }

            var result = new NDArray<double>(size);
            ArraySlice<double> resultArray = result.Data<double>();

            for (int i = 0; i < result.size; ++i)
                resultArray[i] = SampleNoncentralF(dfnum, dfden, nonc);

            result.ReplaceData(resultArray);
            return result;
        }

        /// <summary>
        ///     Sample a single value from the noncentral F distribution.
        /// </summary>
        /// <remarks>
        ///     Algorithm from NumPy's random_noncentral_f in distributions.c:
        ///     t = noncentral_chisquare(dfnum, nonc) * dfden
        ///     return t / (chisquare(dfden) * dfnum)
        /// </remarks>
        private double SampleNoncentralF(double dfnum, double dfden, double nonc)
        {
            // Use helper methods from np.random.noncentral_chisquare.cs
            double t = SampleNoncentralChisquare(dfnum, nonc) * dfden;
            return t / (SampleChisquare(dfden) * dfnum);
        }

        // Note: SampleChisquare() and SampleNoncentralChisquare() are defined in np.random.noncentral_chisquare.cs
    }
}
