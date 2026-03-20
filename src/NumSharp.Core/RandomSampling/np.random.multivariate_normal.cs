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
        public NDArray multivariate_normal(double[] mean, double[,] cov, Shape size = default,
            string check_valid = "warn", double tol = 1e-8)
        {
            // Validation
            if (mean == null || mean.Length == 0)
                throw new ArgumentException("mean must be a non-empty array", nameof(mean));
            if (cov == null)
                throw new ArgumentException("cov must not be null", nameof(cov));

            int n = mean.Length;

            // Check cov is square
            if (cov.GetLength(0) != cov.GetLength(1))
                throw new ArgumentException("cov must be 2 dimensional and square", nameof(cov));
            if (cov.GetLength(0) != n)
                throw new ArgumentException("mean and cov must have same length", nameof(cov));

            // Validate check_valid parameter
            if (check_valid != "warn" && check_valid != "raise" && check_valid != "ignore")
                throw new ArgumentException("check_valid must equal 'warn', 'raise', or 'ignore'", nameof(check_valid));

            // Perform Cholesky decomposition
            double[,] L;
            try
            {
                L = CholeskyDecomposition(cov, n);
            }
            catch (ArgumentException ex) when (check_valid == "ignore")
            {
                // If not positive definite and check_valid is "ignore", try to make it work
                // by using the absolute values of diagonal and zeroing problematic elements
                L = CholeskyDecompositionFallback(cov, n);
            }
            catch (ArgumentException ex)
            {
                if (check_valid == "raise")
                    throw new ArgumentException("covariance is not symmetric positive-semidefinite.", nameof(cov));
                // warn - we continue but log warning (in C# we just continue)
                // Try fallback
                L = CholeskyDecompositionFallback(cov, n);
            }

            if (size.IsEmpty)
            {
                // Return single sample with shape (n,)
                var result = new NDArray<double>(new Shape(n));
                ArraySlice<double> data = result.Data<double>();
                SampleMultivariateNormal(mean, L, n, data, 0);
                return result;
            }

            // Output shape is (*size, n)
            int[] outputDims = new int[size.NDim + 1];
            for (int i = 0; i < size.NDim; i++)
                outputDims[i] = size.dimensions[i];
            outputDims[size.NDim] = n;

            var ret = new NDArray<double>(outputDims);
            ArraySlice<double> retData = ret.Data<double>();

            // Number of samples is product of size dimensions
            int numSamples = size.size;

            for (int s = 0; s < numSamples; s++)
            {
                SampleMultivariateNormal(mean, L, n, retData, s * n);
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
        public NDArray multivariate_normal(NDArray mean, NDArray cov, Shape size = default,
            string check_valid = "warn", double tol = 1e-8)
        {
            // Validate dimensions
            if (mean.ndim != 1)
                throw new ArgumentException("mean must be 1 dimensional", nameof(mean));
            if (cov.ndim != 2)
                throw new ArgumentException("cov must be 2 dimensional and square", nameof(cov));

            int n = mean.size;

            // Convert mean to double[]
            double[] meanArray = new double[n];
            int idx = 0;
            foreach (var val in mean.AsIterator<double>())
            {
                meanArray[idx++] = val;
            }

            // Convert cov to double[,]
            double[,] covArray = new double[cov.shape[0], cov.shape[1]];
            for (int i = 0; i < cov.shape[0]; i++)
            {
                for (int j = 0; j < cov.shape[1]; j++)
                {
                    covArray[i, j] = cov.GetDouble(i, j);
                }
            }

            return multivariate_normal(meanArray, covArray, size, check_valid, tol);
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
        /// </summary>
        /// <remarks>
        ///     Algorithm: X = mean + Z @ L.T where Z ~ N(0, I) and L = cholesky(cov).
        ///     The transformation gives us samples with the desired covariance.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SampleMultivariateNormal(double[] mean, double[,] L, int n, ArraySlice<double> data, int offset)
        {
            // Generate standard normal samples
            double[] z = new double[n];
            for (int i = 0; i < n; i++)
            {
                z[i] = NextGaussian();
            }

            // Compute mean + Z @ L.T
            // (Z @ L.T)[i] = sum_j Z[j] * L[i, j] (since L.T[j, i] = L[i, j])
            // But actually L @ Z.T gives us a column vector, so we want L @ Z
            // X = mean + L @ Z
            for (int i = 0; i < n; i++)
            {
                double sum = mean[i];
                for (int j = 0; j <= i; j++)  // L is lower triangular
                {
                    sum += L[i, j] * z[j];
                }
                data[offset + i] = sum;
            }
        }

        /// <summary>
        ///     Compute the Cholesky decomposition of a symmetric positive-definite matrix.
        /// </summary>
        /// <param name="A">The input matrix (must be symmetric positive-definite).</param>
        /// <param name="n">The dimension of the matrix.</param>
        /// <returns>The lower triangular matrix L such that A = L @ L.T.</returns>
        /// <exception cref="ArgumentException">If the matrix is not positive-definite.</exception>
        private static double[,] CholeskyDecomposition(double[,] A, int n)
        {
            double[,] L = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < j; k++)
                    {
                        sum += L[i, k] * L[j, k];
                    }

                    if (i == j)
                    {
                        double diag = A[i, i] - sum;
                        if (diag < 0)
                        {
                            throw new ArgumentException(
                                "Matrix is not positive-definite (negative diagonal encountered).");
                        }
                        L[i, j] = Math.Sqrt(diag);
                    }
                    else
                    {
                        if (Math.Abs(L[j, j]) < 1e-15)
                        {
                            L[i, j] = 0;
                        }
                        else
                        {
                            L[i, j] = (A[i, j] - sum) / L[j, j];
                        }
                    }
                }
            }

            return L;
        }

        /// <summary>
        ///     Fallback Cholesky decomposition for nearly positive-semidefinite matrices.
        ///     Uses absolute value for negative diagonals to avoid failure.
        /// </summary>
        private static double[,] CholeskyDecompositionFallback(double[,] A, int n)
        {
            double[,] L = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < j; k++)
                    {
                        sum += L[i, k] * L[j, k];
                    }

                    if (i == j)
                    {
                        double diag = A[i, i] - sum;
                        // Use absolute value to handle slightly negative values
                        L[i, j] = Math.Sqrt(Math.Abs(diag));
                    }
                    else
                    {
                        if (Math.Abs(L[j, j]) < 1e-15)
                        {
                            L[i, j] = 0;
                        }
                        else
                        {
                            L[i, j] = (A[i, j] - sum) / L[j, j];
                        }
                    }
                }
            }

            return L;
        }
    }
}
