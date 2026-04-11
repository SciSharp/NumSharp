using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a Hypergeometric distribution.
        /// </summary>
        /// <param name="ngood">Number of ways to make a good selection. Must be non-negative.</param>
        /// <param name="nbad">Number of ways to make a bad selection. Must be non-negative.</param>
        /// <param name="nsample">Number of items sampled. Must be >= 1 and &lt;= ngood + nbad.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the hypergeometric distribution (number of good items in sample).</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.hypergeometric.html
        ///     <br/>
        ///     Consider an urn with ngood white marbles and nbad black marbles.
        ///     If you draw nsample balls without replacement, the hypergeometric distribution
        ///     describes the distribution of white balls in the drawn sample.
        ///     <br/>
        ///     Mean = nsample * ngood / (ngood + nbad)
        /// </remarks>
        public NDArray hypergeometric(long ngood, long nbad, long nsample, Shape? size = null)
        {
            ValidateHypergeometricParams(ngood, nbad, nsample);

            if (size == null)
                return NDArray.Scalar(SampleHypergeometric(ngood, nbad, nsample));

            var ret = new NDArray<long>(size.Value);
            ArraySlice<long> data = ret.Data<long>();

            for (int i = 0; i < ret.size; i++)
            {
                data[i] = SampleHypergeometric(ngood, nbad, nsample);
            }

            return ret;
        }

        /// <summary>
        ///     Draw samples from a Hypergeometric distribution.
        /// </summary>
        /// <param name="ngood">Number of ways to make a good selection. Must be non-negative.</param>
        /// <param name="nbad">Number of ways to make a bad selection. Must be non-negative.</param>
        /// <param name="nsample">Number of items sampled. Must be >= 1 and &lt;= ngood + nbad.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the hypergeometric distribution.</returns>
        public NDArray hypergeometric(long ngood, long nbad, long nsample, int[] size)
            => hypergeometric(ngood, nbad, nsample, new Shape(size));

        /// <summary>
        ///     Draw samples from a Hypergeometric distribution.
        /// </summary>
        /// <param name="ngood">Number of ways to make a good selection. Must be non-negative.</param>
        /// <param name="nbad">Number of ways to make a bad selection. Must be non-negative.</param>
        /// <param name="nsample">Number of items sampled. Must be >= 1 and &lt;= ngood + nbad.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the hypergeometric distribution.</returns>
        public NDArray hypergeometric(long ngood, long nbad, long nsample, long[] size)
            => hypergeometric(ngood, nbad, nsample, new Shape(size));

        /// <summary>
        ///     Draw samples from a Hypergeometric distribution.
        /// </summary>
        /// <param name="ngood">Number of ways to make a good selection. Must be non-negative.</param>
        /// <param name="nbad">Number of ways to make a bad selection. Must be non-negative.</param>
        /// <param name="nsample">Number of items sampled. Must be >= 1 and &lt;= ngood + nbad.</param>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the hypergeometric distribution.</returns>
        public NDArray hypergeometric(long ngood, long nbad, long nsample, int size)
            => hypergeometric(ngood, nbad, nsample, new int[] { size });

        /// <summary>
        ///     Draw a single sample from a Hypergeometric distribution.
        /// </summary>
        /// <param name="ngood">Number of ways to make a good selection. Must be non-negative.</param>
        /// <param name="nbad">Number of ways to make a bad selection. Must be non-negative.</param>
        /// <param name="nsample">Number of items sampled. Must be >= 1 and &lt;= ngood + nbad.</param>
        /// <returns>A single sample from the hypergeometric distribution.</returns>
        public long hypergeometric(long ngood, long nbad, long nsample)
        {
            ValidateHypergeometricParams(ngood, nbad, nsample);
            return SampleHypergeometric(ngood, nbad, nsample);
        }

        private void ValidateHypergeometricParams(long ngood, long nbad, long nsample)
        {
            if (ngood < 0)
                throw new ArgumentException("ngood < 0", nameof(ngood));
            if (nbad < 0)
                throw new ArgumentException("nbad < 0", nameof(nbad));
            if (nsample < 1)
                throw new ArgumentException("nsample < 1", nameof(nsample));
            if (nsample > ngood + nbad)
                throw new ArgumentException("ngood + nbad < nsample");
        }

        /// <summary>
        ///     Sample from the hypergeometric distribution using the basic algorithm.
        /// </summary>
        /// <remarks>
        ///     This is the basic sampling algorithm from NumPy's hypergeometric_sample():
        ///     - If sample > total/2, we select the complement
        ///     - Loop: for each item in computed_sample, randomly decide if it's "good"
        ///     - Probability of good = remaining_good / remaining_total
        ///     - Early exit if only good items remain
        /// </remarks>
        private long SampleHypergeometric(long good, long bad, long sample)
        {
            long total = good + bad;
            long computedSample;
            long remainingTotal = total;
            long remainingGood = good;

            // Optimization: if sample > half, select complement
            if (sample > total / 2)
            {
                computedSample = total - sample;
            }
            else
            {
                computedSample = sample;
            }

            // Main sampling loop
            while (computedSample > 0 && remainingGood > 0 && remainingTotal > remainingGood)
            {
                --remainingTotal;
                // Random integer in [0, remainingTotal]
                long randomValue = (long)(randomizer.NextDouble() * (remainingTotal + 1));
                if (randomValue < remainingGood)
                {
                    // Selected a "good" one
                    --remainingGood;
                }
                --computedSample;
            }

            // If only "good" choices left, take what's needed
            if (remainingTotal == remainingGood)
            {
                remainingGood -= computedSample;
            }

            // Return result based on whether we used complement
            if (sample > total / 2)
            {
                return remainingGood;
            }
            else
            {
                return good - remainingGood;
            }
        }
    }
}
