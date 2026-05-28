using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
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
        ///     Algorithm: Uses Jacobi eigendecomposition of the covariance matrix with sign normalization
        ///     to match NumPy's SVD-based approach. Transform = U @ sqrt(S), then X = mean + Transform @ Z
        ///     where Z ~ N(0, I).
        ///     <br/>
        ///     NumPy Compatibility: Produces 1-to-1 matching samples with NumPy for most common cases
        ///     (identity, diagonal, and correlated covariance matrices up to 4x4). For some larger matrices
        ///     (5x5+), the samples are statistically correct (same distribution) but may differ in exact
        ///     sequence due to differences in eigenvector sign conventions between Jacobi and LAPACK's
        ///     divide-and-conquer algorithms.
        /// </remarks>
        public unsafe NDArray multivariate_normal(double[] mean, double[,] cov, Shape? size = null,
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

            // Compute SVD transform matrix: U @ sqrt(S)
            // For symmetric matrices, this matches NumPy's approach
            var transformBlock = new UnmanagedMemoryBlock<double>(n * n);
            var transform = new ArraySlice<double>(transformBlock);

            bool success = ComputeSvdTransform(covSlice, transform, n, tol);
            if (!success)
            {
                if (check_valid == "raise")
                    throw new ArgumentException("covariance is not symmetric positive-semidefinite.", nameof(cov));
                // For warn/ignore, we still computed a fallback transform
            }

            // Allocate scratch space for z vector
            var zBlock = new UnmanagedMemoryBlock<double>(n);
            var z = new ArraySlice<double>(zBlock);

            if (size == null)
            {
                // Return single sample with shape (n,)
                var result = new NDArray<double>(new Shape(n));
                ArraySlice<double> data = result.Data<double>();
                SampleMultivariateNormalSvd(meanSlice, transform, n, z, data, 0);
                return result;
            }

            // Output shape is (*size, n)
            var sizeVal = size.Value;
            long[] outputDims = new long[sizeVal.NDim + 1];
            for (long i = 0; i < sizeVal.NDim; i++)
                outputDims[i] = sizeVal.dimensions[i];
            outputDims[sizeVal.NDim] = n;

            var ret = new NDArray<double>(outputDims);
            ArraySlice<double> retData = ret.Data<double>();

            // Number of samples is product of size dimensions
            long numSamples = sizeVal.size;

            for (long s = 0; s < numSamples; s++)
            {
                SampleMultivariateNormalSvd(meanSlice, transform, n, z, retData, s * n);
            }

            return ret;
        }

        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        public unsafe NDArray multivariate_normal(NDArray mean, NDArray cov, Shape? size = null,
            string check_valid = "warn", double tol = 1e-8)
        {
            // Validate dimensions
            if (mean.ndim != 1)
                throw new ArgumentException("mean must be 1 dimensional", nameof(mean));
            if (cov.ndim != 2)
                throw new ArgumentException("cov must be 2 dimensional and square", nameof(cov));

            long n = mean.size;

            // Copy mean (any layout) into a flat double buffer via NpyIter.Copy.
            var meanBlock = new UnmanagedMemoryBlock<double>(n);
            var meanSlice = new ArraySlice<double>(meanBlock);
            var meanStorage = new UnmanagedStorage(meanSlice, new Shape(n));
            NpyIter.Copy(meanStorage, mean.Storage);

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

            // Compute SVD transform matrix
            var transformBlock = new UnmanagedMemoryBlock<double>(n * n);
            var transform = new ArraySlice<double>(transformBlock);

            bool success = ComputeSvdTransform(covSlice, transform, n, tol);
            if (!success)
            {
                if (check_valid == "raise")
                    throw new ArgumentException("covariance is not symmetric positive-semidefinite.", nameof(cov));
            }

            // Allocate scratch space for z vector
            var zBlock = new UnmanagedMemoryBlock<double>(n);
            var z = new ArraySlice<double>(zBlock);

            if (size == null)
            {
                // Return single sample with shape (n,)
                var result = new NDArray<double>(new Shape(n));
                ArraySlice<double> data = result.Data<double>();
                SampleMultivariateNormalSvd(meanSlice, transform, n, z, data, 0);
                return result;
            }

            // Output shape is (*size, n)
            var sizeVal2 = size.Value;
            long[] outputDims = new long[sizeVal2.NDim + 1];
            for (long i = 0; i < sizeVal2.NDim; i++)
                outputDims[i] = sizeVal2.dimensions[i];
            outputDims[sizeVal2.NDim] = n;

            var ret = new NDArray<double>(outputDims);
            ArraySlice<double> retData = ret.Data<double>();

            // Number of samples is product of size dimensions
            long numSamples = sizeVal2.size;

            for (long s = 0; s < numSamples; s++)
            {
                SampleMultivariateNormalSvd(meanSlice, transform, n, z, retData, s * n);
            }

            return ret;
        }

        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        public NDArray multivariate_normal(double[] mean, double[,] cov, int size,
            string check_valid = "warn", double tol = 1e-8)
            => multivariate_normal(mean, cov, new Shape(size), check_valid, tol);

        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        public NDArray multivariate_normal(double[] mean, double[,] cov, int[] size)
            => multivariate_normal(mean, cov, new Shape(size));

        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        public NDArray multivariate_normal(double[] mean, double[,] cov, long[] size)
            => multivariate_normal(mean, cov, new Shape(size));

        /// <summary>
        ///     Sample a single multivariate normal vector using SVD transform.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SampleMultivariateNormalSvd(ArraySlice<double> mean, ArraySlice<double> transform, long n,
            ArraySlice<double> z, ArraySlice<double> data, long offset)
        {
            // Generate standard normal samples into z
            for (long i = 0; i < n; i++)
            {
                z[i] = NextGaussian();
            }

            // Compute mean + Transform @ Z
            // Transform is row-major: Transform[i,j] = transform[i*n+j]
            for (long i = 0; i < n; i++)
            {
                double sum = mean[i];
                for (long j = 0; j < n; j++)
                {
                    sum += transform[i * n + j] * z[j];
                }
                data[offset + i] = sum;
            }
        }

        /// <summary>
        ///     Compute the SVD-based transform matrix for multivariate normal sampling.
        ///     For a symmetric covariance matrix, computes U @ sqrt(S) where cov = U @ S @ U.T.
        ///     Uses Jacobi eigendecomposition for symmetric matrices.
        /// </summary>
        /// <returns>True if successful, false if matrix has negative eigenvalues.</returns>
        private static bool ComputeSvdTransform(ArraySlice<double> cov, ArraySlice<double> transform, long n, double tol)
        {
            // For symmetric matrices, SVD gives the same result as eigendecomposition
            // cov = U @ S @ V.T where U = V for symmetric matrices
            // So we use Jacobi eigendecomposition

            // Allocate working arrays
            var eigenvectorsBlock = new UnmanagedMemoryBlock<double>(n * n);
            var eigenvectors = new ArraySlice<double>(eigenvectorsBlock);
            var eigenvaluesBlock = new UnmanagedMemoryBlock<double>(n);
            var eigenvalues = new ArraySlice<double>(eigenvaluesBlock);
            var workBlock = new UnmanagedMemoryBlock<double>(n * n);
            var work = new ArraySlice<double>(workBlock);

            // Copy cov to work matrix
            for (long i = 0; i < n * n; i++)
                work[i] = cov[i];

            // Initialize eigenvectors to identity
            for (long i = 0; i < n; i++)
            {
                for (long j = 0; j < n; j++)
                    eigenvectors[i * n + j] = (i == j) ? 1.0 : 0.0;
            }

            // Jacobi eigendecomposition
            bool hasNegative = false;
            JacobiEigendecomposition(work, eigenvectors, eigenvalues, n, 100, 1e-12);

            // Check for negative eigenvalues
            for (long i = 0; i < n; i++)
            {
                if (eigenvalues[i] < -tol)
                    hasNegative = true;
            }

            // Sort eigenvalues in DESCENDING order (to match NumPy SVD)
            // and reorder eigenvectors accordingly
            SortEigenDescending(eigenvalues, eigenvectors, n);

            // Normalize eigenvector signs to match NumPy/LAPACK SVD convention:
            // The element with largest absolute value in each column should be NEGATIVE
            NormalizeEigenvectorSigns(eigenvectors, n);

            // Compute transform = eigenvectors @ diag(sqrt(abs(eigenvalues)))
            // NumPy uses abs for robustness with nearly singular matrices
            for (long i = 0; i < n; i++)
            {
                double sqrtEig = Math.Sqrt(Math.Abs(eigenvalues[i]));
                for (long j = 0; j < n; j++)
                {
                    transform[j * n + i] = eigenvectors[j * n + i] * sqrtEig;
                }
            }

            return !hasNegative;
        }

        /// <summary>
        ///     Jacobi eigendecomposition for symmetric matrices.
        ///     Uses the classical Jacobi algorithm with Schur2 rotations.
        /// </summary>
        private static void JacobiEigendecomposition(ArraySlice<double> A, ArraySlice<double> V,
            ArraySlice<double> eigenvalues, long n, int maxIterations, double tolerance)
        {
            // Classical Jacobi algorithm
            for (int iter = 0; iter < maxIterations * n * n; iter++)
            {
                // Find the largest off-diagonal element
                double maxOffDiag = 0;
                long p = 0, q = 1;

                for (long i = 0; i < n; i++)
                {
                    for (long j = i + 1; j < n; j++)
                    {
                        double absVal = Math.Abs(A[i * n + j]);
                        if (absVal > maxOffDiag)
                        {
                            maxOffDiag = absVal;
                            p = i;
                            q = j;
                        }
                    }
                }

                // Check for convergence
                if (maxOffDiag < tolerance)
                    break;

                // Compute Schur2 rotation
                double App = A[p * n + p];
                double Aqq = A[q * n + q];
                double Apq = A[p * n + q];

                double c, s;
                if (Math.Abs(Apq) < tolerance)
                {
                    c = 1.0;
                    s = 0.0;
                }
                else
                {
                    double tau = (Aqq - App) / (2.0 * Apq);
                    double t;
                    if (tau >= 0)
                        t = 1.0 / (tau + Math.Sqrt(1.0 + tau * tau));
                    else
                        t = 1.0 / (tau - Math.Sqrt(1.0 + tau * tau));
                    c = 1.0 / Math.Sqrt(1.0 + t * t);
                    s = t * c;
                }

                // Apply rotation to A: A' = J.T @ A @ J
                // This zeroes out A[p,q] and A[q,p]
                double newApp = c * c * App - 2.0 * s * c * Apq + s * s * Aqq;
                double newAqq = s * s * App + 2.0 * s * c * Apq + c * c * Aqq;

                A[p * n + p] = newApp;
                A[q * n + q] = newAqq;
                A[p * n + q] = 0.0;
                A[q * n + p] = 0.0;

                // Update other rows/columns
                for (long k = 0; k < n; k++)
                {
                    if (k != p && k != q)
                    {
                        double Akp = A[k * n + p];
                        double Akq = A[k * n + q];
                        A[k * n + p] = c * Akp - s * Akq;
                        A[p * n + k] = A[k * n + p];
                        A[k * n + q] = s * Akp + c * Akq;
                        A[q * n + k] = A[k * n + q];
                    }
                }

                // Update eigenvector matrix: V' = V @ J
                for (long k = 0; k < n; k++)
                {
                    double Vkp = V[k * n + p];
                    double Vkq = V[k * n + q];
                    V[k * n + p] = c * Vkp - s * Vkq;
                    V[k * n + q] = s * Vkp + c * Vkq;
                }
            }

            // Extract eigenvalues from diagonal
            for (long i = 0; i < n; i++)
                eigenvalues[i] = A[i * n + i];
        }

        /// <summary>
        ///     Sort eigenvalues in descending order and reorder eigenvectors accordingly.
        /// </summary>
        private static void SortEigenDescending(ArraySlice<double> eigenvalues, ArraySlice<double> eigenvectors, long n)
        {
            // Simple insertion sort (n is typically small for covariance matrices)
            for (long i = 1; i < n; i++)
            {
                double keyVal = eigenvalues[i];
                long j = i - 1;

                // Sort descending: move larger values to front
                while (j >= 0 && eigenvalues[j] < keyVal)
                {
                    // Swap eigenvalues
                    eigenvalues[j + 1] = eigenvalues[j];
                    eigenvalues[j] = keyVal;

                    // Swap corresponding eigenvector columns
                    for (long k = 0; k < n; k++)
                    {
                        double temp = eigenvectors[k * n + (j + 1)];
                        eigenvectors[k * n + (j + 1)] = eigenvectors[k * n + j];
                        eigenvectors[k * n + j] = temp;
                    }

                    j--;
                }
            }
        }

        /// <summary>
        ///     Normalize eigenvector signs to approximate NumPy/LAPACK SVD convention.
        ///
        ///     LAPACK's divide-and-conquer SVD (DGESDD) determines eigenvector signs based on
        ///     internal algorithm state (specifically DLAED3), which is deterministic but not
        ///     predictable from external matrix properties. This heuristic matches NumPy for
        ///     most common cases (identity, diagonal, correlated matrices up to 4x4).
        ///
        ///     Two-step process:
        ///     1. Make the element with largest absolute value in each column NEGATIVE
        ///        (skip standard basis vectors for identity matrices)
        ///     2. Ensure determinant matches NumPy convention: +1 for odd n, -1 for even n
        ///        (skip for identity-like matrices where all columns are standard basis vectors)
        /// </summary>
        private static void NormalizeEigenvectorSigns(ArraySlice<double> eigenvectors, long n)
        {
            // Step 1: Make largest element in each column negative
            // Exception: don't flip standard basis vectors (only one non-zero element)
            long standardBasisCount = 0;
            for (long col = 0; col < n; col++)
            {
                // Find the element with largest absolute value and count non-zero elements
                long maxRow = 0;
                double maxAbs = 0;
                int nonZeroCount = 0;
                for (long row = 0; row < n; row++)
                {
                    double absVal = Math.Abs(eigenvectors[row * n + col]);
                    if (absVal > 1e-10)
                        nonZeroCount++;
                    if (absVal > maxAbs)
                    {
                        maxAbs = absVal;
                        maxRow = row;
                    }
                }

                // Skip flipping for standard basis vectors (identity matrix eigenvectors)
                if (nonZeroCount == 1)
                {
                    standardBasisCount++;
                    continue;
                }

                // If the largest element is positive, flip the entire column
                if (eigenvectors[maxRow * n + col] > 0)
                {
                    for (long row = 0; row < n; row++)
                    {
                        eigenvectors[row * n + col] = -eigenvectors[row * n + col];
                    }
                }
            }

            // Step 2: Adjust determinant to match NumPy convention
            // NumPy's SVD has det(U) = +1 for odd n, -1 for even n
            // BUT: This only applies when eigenvalues are distinct
            // For identity-like matrices (all standard basis vectors), skip this step
            if (standardBasisCount == n)
                return;

            double det = ComputeDeterminant(eigenvectors, n);
            double expectedDet = (n % 2 == 1) ? 1.0 : -1.0;

            // If determinant has wrong sign, flip the last column
            if ((det > 0 && expectedDet < 0) || (det < 0 && expectedDet > 0))
            {
                for (long row = 0; row < n; row++)
                {
                    eigenvectors[row * n + (n - 1)] = -eigenvectors[row * n + (n - 1)];
                }
            }
        }

        /// <summary>
        ///     Compute determinant of an n×n matrix (stored row-major).
        ///     Uses LU decomposition for efficiency.
        /// </summary>
        private static double ComputeDeterminant(ArraySlice<double> matrix, long n)
        {
            if (n == 1)
                return matrix[0];

            if (n == 2)
                return matrix[0] * matrix[3] - matrix[1] * matrix[2];

            if (n == 3)
            {
                // Sarrus rule for 3x3
                return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
                     - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
                     + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
            }

            // For larger matrices, use LU decomposition with partial pivoting
            // Copy matrix to avoid modification
            var lu = new double[n * n];
            for (long i = 0; i < n * n; i++)
                lu[i] = matrix[i];

            double det = 1.0;
            int swaps = 0;

            for (long k = 0; k < n; k++)
            {
                // Find pivot
                long maxRow = k;
                double maxVal = Math.Abs(lu[k * n + k]);
                for (long i = k + 1; i < n; i++)
                {
                    double absVal = Math.Abs(lu[i * n + k]);
                    if (absVal > maxVal)
                    {
                        maxVal = absVal;
                        maxRow = i;
                    }
                }

                if (maxVal < 1e-15)
                    return 0.0; // Singular matrix

                // Swap rows if needed
                if (maxRow != k)
                {
                    for (long j = 0; j < n; j++)
                    {
                        double temp = lu[k * n + j];
                        lu[k * n + j] = lu[maxRow * n + j];
                        lu[maxRow * n + j] = temp;
                    }
                    swaps++;
                }

                det *= lu[k * n + k];

                // Eliminate below
                for (long i = k + 1; i < n; i++)
                {
                    double factor = lu[i * n + k] / lu[k * n + k];
                    for (long j = k + 1; j < n; j++)
                    {
                        lu[i * n + j] -= factor * lu[k * n + j];
                    }
                }
            }

            return (swaps % 2 == 0) ? det : -det;
        }

    }
}
