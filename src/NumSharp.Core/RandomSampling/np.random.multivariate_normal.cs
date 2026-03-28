using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">Mean of the N-dimensional distribution (1D array of length N).</param>
        /// <param name="cov">Covariance matrix of the distribution (N x N symmetric positive-semidefinite).</param>
        /// <param name="size">Output shape. Given a shape of (m, n, k), m*n*k samples are generated with output shape (m, n, k, N).</param>
        /// <param name="check_valid">Behavior when the covariance matrix is not positive semidefinite: "warn", "raise", or "ignore".</param>
        /// <param name="tol">Tolerance when checking covariance matrix validity.</param>
        /// <returns>Drawn samples of shape (*size, N) where N is the length of mean.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html
        ///     <br/>
        ///     The multivariate normal distribution is a generalization of the 1D normal distribution
        ///     to higher dimensions. It is specified by its mean vector and covariance matrix.
        ///     <br/>
        ///     Algorithm: Uses Cholesky decomposition of the covariance matrix.
        ///     L = cholesky(cov), then X = mean + Z @ L.T where Z ~ N(0, I).
        /// </remarks>
        public unsafe NDArray multivariate_normal(double[] mean, double[,] cov, Shape size = default,
            string check_valid = "warn", double tol = 1e-8)
        {
            // Validation
            if (mean == null || mean.Length == 0)
                throw new ArgumentException("mean must be a non-empty array", nameof(mean));
            if (cov == null)
                throw new ArgumentException("cov must not be null", nameof(cov));

            long n = mean.Length;

            // Check cov is square
            if (cov.GetLength(0) != cov.GetLength(1))
                throw new ArgumentException("cov must be 2 dimensional and square", nameof(cov));
            if (cov.GetLength(0) != n)
                throw new ArgumentException("mean and cov must have same length", nameof(cov));

            // Validate check_valid parameter
            if (check_valid != "warn" && check_valid != "raise" && check_valid != "ignore")
                throw new ArgumentException("check_valid must equal 'warn', 'raise', or 'ignore'", nameof(check_valid));

            // Copy mean to unmanaged storage
            var meanBlock = new UnmanagedMemoryBlock<double>(n);
            var meanSlice = new ArraySlice<double>(meanBlock);
            for (long i = 0; i < n; i++)
                meanSlice[i] = mean[i];

            // Copy cov to unmanaged storage (row-major: cov[i,j] -> covSlice[i*n+j])
            var covBlock = new UnmanagedMemoryBlock<double>(n * n);
            var covSlice = new ArraySlice<double>(covBlock);
            for (long i = 0; i < n; i++)
            {
                for (long j = 0; j < n; j++)
                    covSlice[i * n + j] = cov[i, j];
            }

            // Perform Cholesky decomposition into unmanaged storage
            var LBlock = new UnmanagedMemoryBlock<double>(n * n);
            var L = new ArraySlice<double>(LBlock);

            bool success;
            try
            {
                success = CholeskyDecompositionUnmanaged(covSlice, L, n);
            }
            catch (ArgumentException) when (check_valid == "ignore")
            {
                success = false;
            }

            if (!success)
            {
                if (check_valid == "raise")
                    throw new ArgumentException("covariance is not symmetric positive-semidefinite.", nameof(cov));
                // warn/ignore - try fallback
                CholeskyDecompositionFallbackUnmanaged(covSlice, L, n);
            }

            // Allocate scratch space for z vector
            var zBlock = new UnmanagedMemoryBlock<double>(n);
            var z = new ArraySlice<double>(zBlock);

            if (size.IsEmpty)
            {
                // Return single sample with shape (n,)
                var result = new NDArray<double>(new Shape(n));
                ArraySlice<double> data = result.Data<double>();
                SampleMultivariateNormalUnmanaged(meanSlice, L, n, z, data, 0);
                return result;
            }

            // Output shape is (*size, n)
            long[] outputDims = new long[size.NDim + 1];
            for (int i = 0; i < size.NDim; i++)
                outputDims[i] = size.dimensions[i];
            outputDims[size.NDim] = n;

            var ret = new NDArray<double>(outputDims);
            ArraySlice<double> retData = ret.Data<double>();

            // Number of samples is product of size dimensions
            long numSamples = size.size;

            for (long s = 0; s < numSamples; s++)
            {
                SampleMultivariateNormalUnmanaged(meanSlice, L, n, z, retData, s * n);
            }

            return ret;
        }

        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">Mean as NDArray (1D).</param>
        /// <param name="cov">Covariance matrix as NDArray (2D).</param>
        /// <param name="size">Output shape.</param>
        /// <param name="check_valid">Behavior when cov is not positive semidefinite.</param>
        /// <param name="tol">Tolerance for validity check.</param>
        /// <returns>Drawn samples.</returns>
        public unsafe NDArray multivariate_normal(NDArray mean, NDArray cov, Shape size = default,
            string check_valid = "warn", double tol = 1e-8)
        {
            // Validate dimensions
            if (mean.ndim != 1)
                throw new ArgumentException("mean must be 1 dimensional", nameof(mean));
            if (cov.ndim != 2)
                throw new ArgumentException("cov must be 2 dimensional and square", nameof(cov));

            long n = mean.size;

            // Copy mean to unmanaged storage
            var meanBlock = new UnmanagedMemoryBlock<double>(n);
            var meanSlice = new ArraySlice<double>(meanBlock);
            long idx = 0;
            foreach (var val in mean.AsIterator<double>())
            {
                meanSlice[idx++] = val;
            }

            // Copy cov to unmanaged storage (row-major)
            var covBlock = new UnmanagedMemoryBlock<double>(n * n);
            var covSlice = new ArraySlice<double>(covBlock);
            for (long i = 0; i < cov.shape[0]; i++)
            {
                for (long j = 0; j < cov.shape[1]; j++)
                {
                    covSlice[i * n + j] = cov.GetDouble(i, j);
                }
            }

            // Validate check_valid parameter
            if (check_valid != "warn" && check_valid != "raise" && check_valid != "ignore")
                throw new ArgumentException("check_valid must equal 'warn', 'raise', or 'ignore'", nameof(check_valid));

            // Check cov is square
            if (cov.shape[0] != cov.shape[1])
                throw new ArgumentException("cov must be 2 dimensional and square", nameof(cov));
            if (cov.shape[0] != n)
                throw new ArgumentException("mean and cov must have same length", nameof(cov));

            // Perform Cholesky decomposition into unmanaged storage
            var LBlock = new UnmanagedMemoryBlock<double>(n * n);
            var L = new ArraySlice<double>(LBlock);

            bool success;
            try
            {
                success = CholeskyDecompositionUnmanaged(covSlice, L, n);
            }
            catch (ArgumentException) when (check_valid == "ignore")
            {
                success = false;
            }

            if (!success)
            {
                if (check_valid == "raise")
                    throw new ArgumentException("covariance is not symmetric positive-semidefinite.", nameof(cov));
                // warn/ignore - try fallback
                CholeskyDecompositionFallbackUnmanaged(covSlice, L, n);
            }

            // Allocate scratch space for z vector
            var zBlock = new UnmanagedMemoryBlock<double>(n);
            var z = new ArraySlice<double>(zBlock);

            if (size.IsEmpty)
            {
                // Return single sample with shape (n,)
                var result = new NDArray<double>(new Shape(n));
                ArraySlice<double> data = result.Data<double>();
                SampleMultivariateNormalUnmanaged(meanSlice, L, n, z, data, 0);
                return result;
            }

            // Output shape is (*size, n)
            long[] outputDims = new long[size.NDim + 1];
            for (long i = 0; i < size.NDim; i++)
                outputDims[i] = size.dimensions[i];
            outputDims[size.NDim] = n;

            var ret = new NDArray<double>(outputDims);
            ArraySlice<double> retData = ret.Data<double>();

            // Number of samples is product of size dimensions
            long numSamples = size.size;

            for (long s = 0; s < numSamples; s++)
            {
                SampleMultivariateNormalUnmanaged(meanSlice, L, n, z, retData, s * n);
            }

            return ret;
        }

        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">Mean vector.</param>
        /// <param name="cov">Covariance matrix.</param>
        /// <param name="size">Number of samples.</param>
        /// <param name="check_valid">Behavior when cov is not positive semidefinite.</param>
        /// <param name="tol">Tolerance for validity check.</param>
        /// <returns>Drawn samples.</returns>
        public NDArray multivariate_normal(double[] mean, double[,] cov, int size,
            string check_valid = "warn", double tol = 1e-8)
            => multivariate_normal(mean, cov, new Shape(size), check_valid, tol);

        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">Mean vector.</param>
        /// <param name="cov">Covariance matrix.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples.</returns>
        public NDArray multivariate_normal(double[] mean, double[,] cov, params int[] size)
            => multivariate_normal(mean, cov, new Shape(size));

        /// <summary>
        ///     Sample a single multivariate normal vector and store at the given offset.
        ///     Uses unmanaged storage with row-major 2D indexing.
        /// </summary>
        /// <remarks>
        ///     Algorithm: X = mean + Z @ L.T where Z ~ N(0, I) and L = cholesky(cov).
        ///     The transformation gives us samples with the desired covariance.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SampleMultivariateNormalUnmanaged(ArraySlice<double> mean, ArraySlice<double> L, long n,
            ArraySlice<double> z, ArraySlice<double> data, long offset)
        {
            // Generate standard normal samples into z
            for (long i = 0; i < n; i++)
            {
                z[i] = NextGaussian();
            }

            // Compute mean + L @ Z
            // L is lower triangular, stored row-major: L[i,j] = L[i*n+j]
            // X = mean + L @ Z
            for (long i = 0; i < n; i++)
            {
                double sum = mean[i];
                for (long j = 0; j <= i; j++)  // L is lower triangular
                {
                    sum += L[i * n + j] * z[j];
                }
                data[offset + i] = sum;
            }
        }

        /// <summary>
        ///     Compute the Cholesky decomposition of a symmetric positive-definite matrix.
        ///     Uses unmanaged storage with row-major indexing.
        /// </summary>
        /// <param name="A">The input matrix (row-major, n*n elements).</param>
        /// <param name="L">Output lower triangular matrix (row-major, n*n elements).</param>
        /// <param name="n">The dimension of the matrix.</param>
        /// <returns>True if successful, false if not positive-definite.</returns>
        private static bool CholeskyDecompositionUnmanaged(ArraySlice<double> A, ArraySlice<double> L, long n)
        {
            // Initialize L to zero
            for (long i = 0; i < n * n; i++)
                L[i] = 0;

            for (long i = 0; i < n; i++)
            {
                for (long j = 0; j <= i; j++)
                {
                    double sum = 0;
                    for (long k = 0; k < j; k++)
                    {
                        sum += L[i * n + k] * L[j * n + k];
                    }

                    if (i == j)
                    {
                        double diag = A[i * n + i] - sum;
                        if (diag < 0)
                        {
                            return false; // Not positive-definite
                        }
                        L[i * n + j] = Math.Sqrt(diag);
                    }
                    else
                    {
                        double Ljj = L[j * n + j];
                        if (Math.Abs(Ljj) < 1e-15)
                        {
                            L[i * n + j] = 0;
                        }
                        else
                        {
                            L[i * n + j] = (A[i * n + j] - sum) / Ljj;
                        }
                    }
                }
            }

            return true;
        }

        /// <summary>
        ///     Fallback Cholesky decomposition for nearly positive-semidefinite matrices.
        ///     Uses absolute value for negative diagonals to avoid failure.
        ///     Uses unmanaged storage with row-major indexing.
        /// </summary>
        private static void CholeskyDecompositionFallbackUnmanaged(ArraySlice<double> A, ArraySlice<double> L, long n)
        {
            // Initialize L to zero
            for (long i = 0; i < n * n; i++)
                L[i] = 0;

            for (long i = 0; i < n; i++)
            {
                for (long j = 0; j <= i; j++)
                {
                    double sum = 0;
                    for (long k = 0; k < j; k++)
                    {
                        sum += L[i * n + k] * L[j * n + k];
                    }

                    if (i == j)
                    {
                        double diag = A[i * n + i] - sum;
                        // Use absolute value to handle slightly negative values
                        L[i * n + j] = Math.Sqrt(Math.Abs(diag));
                    }
                    else
                    {
                        double Ljj = L[j * n + j];
                        if (Math.Abs(Ljj) < 1e-15)
                        {
                            L[i * n + j] = 0;
                        }
                        else
                        {
                            L[i * n + j] = (A[i * n + j] - sum) / Ljj;
                        }
                    }
                }
            }
        }
    }
}
