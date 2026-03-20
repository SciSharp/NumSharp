using System;
using System.Linq;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a multinomial distribution.
        /// </summary>
        /// <param name="n">Number of experiments (>= 0).</param>
        /// <param name="pvals">Probabilities of each of the k different outcomes. Must sum to ~1.</param>
        /// <param name="size">Output shape. Result will have shape (*size, k).</param>
        /// <returns>Drawn samples with shape (*size, k), where each row sums to n.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html
        ///     <br/>
        ///     The multinomial distribution is a multivariate generalization of the binomial
        ///     distribution. Each sample represents n experiments, where each experiment
        ///     results in one of k possible outcomes.
        /// </remarks>
        public NDArray multinomial(int n, double[] pvals, Shape size)
            => multinomial(n, pvals, size.dimensions);

        /// <summary>
        ///     Draw samples from a multinomial distribution.
        /// </summary>
        /// <param name="n">Number of experiments (>= 0).</param>
        /// <param name="pvals">Probabilities of each of the k different outcomes. Must sum to ~1.</param>
        /// <param name="size">Output shape. Result will have shape (*size, k).</param>
        /// <returns>Drawn samples with shape (*size, k), where each row sums to n.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html
        /// </remarks>
        public NDArray multinomial(int n, double[] pvals, params int[] size)
        {
            // Parameter validation
            if (n < 0)
                throw new ArgumentException("n < 0", nameof(n));
            if (pvals == null || pvals.Length == 0)
                throw new ArgumentException("pvals is empty", nameof(pvals));

            // Check for negative probabilities
            for (int i = 0; i < pvals.Length; i++)
            {
                if (pvals[i] < 0 || pvals[i] > 1 || double.IsNaN(pvals[i]))
                    throw new ArgumentException("pvals < 0, pvals > 1 or pvals contains NaNs", nameof(pvals));
            }

            // Check sum of pvals[:-1] <= 1.0
            double sumExceptLast = 0;
            for (int i = 0; i < pvals.Length - 1; i++)
                sumExceptLast += pvals[i];
            if (sumExceptLast > 1.0 + 1e-10)
                throw new ArgumentException("sum(pvals[:-1]) > 1.0", nameof(pvals));

            int k = pvals.Length;

            // Determine output shape: (*size, k)
            int[] outputShape;
            int numSamples;
            if (size == null || size.Length == 0)
            {
                outputShape = new int[] { k };
                numSamples = 1;
            }
            else
            {
                outputShape = new int[size.Length + 1];
                Array.Copy(size, outputShape, size.Length);
                outputShape[size.Length] = k;
                numSamples = size.Aggregate(1, (a, b) => a * b);
            }

            var result = new NDArray(NPTypeCode.Int32, new Shape(outputShape));
            var resultArray = result.Data<int>();

            // Generate samples
            for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
            {
                int offset = sampleIdx * k;
                SampleMultinomial(n, pvals, resultArray, offset);
            }

            return result;
        }

        /// <summary>
        ///     Draw a single multinomial sample.
        /// </summary>
        /// <remarks>
        ///     Algorithm from NumPy's random_multinomial in distributions.c:
        ///     For each category i (except last):
        ///         counts[i] = binomial(remaining, pvals[i] / remaining_p)
        ///         remaining -= counts[i]
        ///         remaining_p -= pvals[i]
        ///     counts[k-1] = remaining
        /// </remarks>
        private void SampleMultinomial(int n, double[] pvals, ArraySlice<int> output, int offset)
        {
            int k = pvals.Length;
            double remainingP = 1.0;
            int remaining = n;

            // Initialize all counts to 0
            for (int j = 0; j < k; j++)
                output[offset + j] = 0;

            for (int j = 0; j < k - 1; j++)
            {
                if (remaining <= 0)
                    break;

                double p = pvals[j] / remainingP;
                if (p > 1.0) p = 1.0;
                if (p < 0.0) p = 0.0;

                int count = SampleBinomial(remaining, p);
                output[offset + j] = count;
                remaining -= count;
                remainingP -= pvals[j];

                if (remainingP < 0)
                    remainingP = 0;
            }

            // Last category gets the remainder
            if (remaining > 0)
                output[offset + k - 1] = remaining;
        }

        /// <summary>
        ///     Sample a single value from the binomial distribution.
        ///     Uses the BTPE algorithm for large n*p, direct method otherwise.
        /// </summary>
        private int SampleBinomial(int n, double p)
        {
            if (n <= 0 || p <= 0)
                return 0;
            if (p >= 1)
                return n;

            // For small n, use direct simulation
            if (n < 20)
            {
                int count = 0;
                for (int i = 0; i < n; i++)
                {
                    if (randomizer.NextDouble() < p)
                        count++;
                }
                return count;
            }

            // For larger n, use inverse transform with normal approximation
            // when n*p*(1-p) is large enough
            double np = n * p;
            double q = 1 - p;

            if (np < 10 || n * q < 10)
            {
                // Direct simulation for moderate n
                int count = 0;
                for (int i = 0; i < n; i++)
                {
                    if (randomizer.NextDouble() < p)
                        count++;
                }
                return count;
            }

            // Normal approximation for large n*p*(1-p)
            double mean = np;
            double stddev = Math.Sqrt(np * q);
            double x = mean + stddev * NextGaussian();
            int result = (int)Math.Round(x);
            if (result < 0) result = 0;
            if (result > n) result = n;
            return result;
        }
    }
}
