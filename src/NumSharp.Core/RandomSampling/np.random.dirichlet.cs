using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from the Dirichlet distribution.
        /// </summary>
        /// <param name="alpha">Concentration parameters of the distribution (k > 0 elements, each > 0).</param>
        /// <param name="size">Output shape. The output has shape (*size, k) where k is the length of alpha.</param>
        /// <returns>Drawn samples from the Dirichlet distribution. Each row sums to 1.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.dirichlet.html
        ///     <br/>
        ///     The Dirichlet distribution is a distribution over vectors x that fulfil:
        ///     - x_i > 0
        ///     - sum(x) = 1
        ///     <br/>
        ///     The probability density function is:
        ///     p(x) = (1/B(alpha)) * prod(x_i^(alpha_i - 1))
        ///     <br/>
        ///     Algorithm: For each sample, draw Y_i ~ Gamma(alpha_i, 1), then X = Y / sum(Y).
        /// </remarks>
        public NDArray dirichlet(double[] alpha, Shape size = default)
        {
            // Validation
            if (alpha == null || alpha.Length == 0)
                throw new ArgumentException("alpha must be a non-empty array", nameof(alpha));

            long k = alpha.Length;  // alpha.Length is int, implicitly converts to long

            for (long i = 0; i < k; i++)
            {
                if (alpha[i] <= 0 || double.IsNaN(alpha[i]))
                    throw new ArgumentException("alpha <= 0", nameof(alpha));
            }

            // Copy alpha to unmanaged storage for consistent long indexing
            var alphaBlock = new UnmanagedMemoryBlock<double>(k);
            var alphaSlice = new ArraySlice<double>(alphaBlock);
            for (long i = 0; i < k; i++)
                alphaSlice[i] = alpha[i];

            if (size.IsEmpty)
            {
                // Return single sample with shape (k,)
                var result = new NDArray<double>(new Shape(k));
                ArraySlice<double> data = result.Data<double>();
                SampleDirichletUnmanaged(alphaSlice, k, data, 0);
                return result;
            }

            // Output shape is (*size, k)
            long[] outputDims = new long[size.NDim + 1];
            for (long i = 0; i < size.NDim; i++)
                outputDims[i] = size.dimensions[i];
            outputDims[size.NDim] = k;

            var ret = new NDArray<double>(outputDims);
            ArraySlice<double> retData = ret.Data<double>();

            // Number of samples is product of size dimensions
            long numSamples = size.size;

            for (long s = 0; s < numSamples; s++)
            {
                SampleDirichletUnmanaged(alphaSlice, k, retData, s * k);
            }

            return ret;
        }

        /// <summary>
        ///     Draw samples from the Dirichlet distribution.
        /// </summary>
        /// <param name="alpha">Concentration parameters as NDArray.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the Dirichlet distribution.</returns>
        public NDArray dirichlet(NDArray alpha, Shape size = default)
        {
            long k = alpha.size;

            // Copy alpha to unmanaged storage
            var alphaBlock = new UnmanagedMemoryBlock<double>(k);
            var alphaSlice = new ArraySlice<double>(alphaBlock);
            long idx = 0;
            foreach (var val in alpha.AsIterator<double>())
            {
                alphaSlice[idx++] = val;
            }

            // Validate
            for (long i = 0; i < k; i++)
            {
                if (alphaSlice[i] <= 0 || double.IsNaN(alphaSlice[i]))
                    throw new ArgumentException("alpha <= 0", nameof(alpha));
            }

            if (size.IsEmpty)
            {
                // Return single sample with shape (k,)
                var result = new NDArray<double>(new Shape(k));
                ArraySlice<double> data = result.Data<double>();
                SampleDirichletUnmanaged(alphaSlice, k, data, 0);
                return result;
            }

            // Output shape is (*size, k)
            long[] outputDims = new long[size.NDim + 1];
            for (long i = 0; i < size.NDim; i++)
                outputDims[i] = size.dimensions[i];
            outputDims[size.NDim] = k;

            var ret = new NDArray<double>(outputDims);
            ArraySlice<double> retData = ret.Data<double>();

            // Number of samples is product of size dimensions
            long numSamples = size.size;

            for (long s = 0; s < numSamples; s++)
            {
                SampleDirichletUnmanaged(alphaSlice, k, retData, s * k);
            }

            return ret;
        }

        /// <summary>
        ///     Draw samples from the Dirichlet distribution.
        /// </summary>
        /// <param name="alpha">Concentration parameters.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the Dirichlet distribution.</returns>
        public NDArray dirichlet(double[] alpha, params int[] size)
            => dirichlet(alpha, new Shape(size));

        /// <summary>
        ///     Draw samples from the Dirichlet distribution.
        /// </summary>
        /// <param name="alpha">Concentration parameters.</param>
        /// <param name="size">Number of samples to draw.</param>
        /// <returns>Drawn samples from the Dirichlet distribution.</returns>
        public NDArray dirichlet(double[] alpha, int size)
            => dirichlet(alpha, new Shape(size));

        /// <summary>
        ///     Sample a single Dirichlet vector and store at the given offset.
        ///     Uses unmanaged storage with long indexing.
        /// </summary>
        /// <remarks>
        ///     Algorithm from NumPy: Y_i ~ Gamma(alpha_i, 1), X = Y / sum(Y)
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SampleDirichletUnmanaged(ArraySlice<double> alpha, long k, ArraySlice<double> data, long offset)
        {
            double sum = 0.0;

            // Draw Gamma(alpha[i], 1) for each i
            for (long i = 0; i < k; i++)
            {
                double y = SampleStandardGamma(alpha[i]);
                data[offset + i] = y;
                sum += y;
            }

            // Normalize to sum to 1
            if (sum > 0)
            {
                for (long i = 0; i < k; i++)
                {
                    data[offset + i] /= sum;
                }
            }
            else
            {
                // Edge case: all gammas were 0 (shouldn't happen with valid alpha > 0)
                // Set uniform distribution
                double uniform = 1.0 / k;
                for (long i = 0; i < k; i++)
                {
                    data[offset + i] = uniform;
                }
            }
        }
    }
}
