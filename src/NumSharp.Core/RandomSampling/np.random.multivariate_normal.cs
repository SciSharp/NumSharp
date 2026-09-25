using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">Mean of the N-dimensional distribution (1D array of length N).</param>
        /// <param name="cov">Covariance matrix of the distribution (N x N symmetric positive-semidefinite).</param>
        /// <param name="size">Output shape. Given a shape of (m, n, k), m*n*k samples are generated with output shape (m, n, k, N);
        ///     null (NumPy's <c>None</c>) draws one sample of shape (N,).</param>
        /// <param name="check_valid">Behavior when the covariance matrix is not positive semidefinite: "warn", "raise", or "ignore".</param>
        /// <param name="tol">Tolerance when checking covariance matrix validity.</param>
        /// <returns>Drawn samples of shape (*size, N) where N is the length of mean.</returns>
        /// <exception cref="ValueError">
        ///     <c>mean must be 1 dimensional</c> (a null mean included, as NumPy's <c>np.array(None)</c> is 0-d),
        ///     <c>cov must be 2 dimensional and square</c> (likewise for a null cov), <c>mean and cov must have same
        ///     length</c> (checked before anything is drawn); an invalid <paramref name="check_valid"/>, or — for
        ///     <c>"raise"</c> — <c>covariance is not symmetric positive-semidefinite.</c> (both checked AFTER the normals are
        ///     drawn, as in NumPy, so the stream has advanced); or a negative size dimension.
        /// </exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html
        ///     <br/>
        ///     The multivariate normal distribution is a generalization of the 1D normal distribution
        ///     to higher dimensions. It is specified by its mean vector and covariance matrix.
        ///     <br/>
        ///     Algorithm — NumPy's legacy <c>multivariate_normal</c> step for step: draw <c>standard_normal((*size, N))</c>
        ///     FIRST, factor <c>(u, s, v) = svd(cov)</c>, check <c>allclose(dot(v.T * s, v), cov, rtol=tol, atol=tol)</c>
        ///     unless <c>check_valid="ignore"</c>, then return <c>dot(x, sqrt(s)[:, None] * v) + mean</c>.
        ///     <br/>
        ///     NumPy Compatibility: with a BLAS/LAPACK backend installed (<c>NumSharp.Interop.OpenBLAS</c>) the SVD and both
        ///     products run the very <c>gesdd</c>/<c>gemm</c> NumPy calls, so samples are byte-identical to
        ///     <c>np.random.RandomState(seed).multivariate_normal</c>. Without a backend the SVD falls back to a managed Jacobi
        ///     eigendecomposition with LAPACK-like sign normalization: the stream of normals is identical and the samples
        ///     match NumPy for most small covariances (identity, diagonal, correlated up to ~4x4), but a singular-vector sign
        ///     or last-ULP difference can appear for larger matrices. <c>check_valid="warn"</c> cannot warn (NumSharp has no
        ///     warnings channel) and proceeds silently.
        /// </remarks>
        public NDArray multivariate_normal(double[] mean, double[,] cov, Shape? size = null,
            string check_valid = "warn", double tol = 1e-8)
        {
            // np.array(None) is a 0-d object array, so NumPy reports a null mean/cov through its shape checks.
            if (mean is null)
                throw new ValueError("mean must be 1 dimensional");
            if (cov is null)
                throw new ValueError("cov must be 2 dimensional and square");

            // np.array(mean) / np.array(cov): owned copies that die with this call.
            using var meanArr = np.array(mean);
            using var covArr = np.array(cov);
            return MultivariateNormalCore(meanArr, covArr, size, check_valid, tol);
        }

        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">Mean of the N-dimensional distribution (1-D; any numeric dtype and layout).</param>
        /// <param name="cov">Covariance matrix (N x N; any numeric dtype and layout — cast to float64 as NumPy's
        ///     <c>cov.astype(np.double)</c>).</param>
        /// <param name="size">Output shape; null draws one sample of shape (N,).</param>
        /// <param name="check_valid">Behavior when the covariance matrix is not positive semidefinite: "warn", "raise", or "ignore".</param>
        /// <param name="tol">Tolerance when checking covariance matrix validity.</param>
        /// <returns>Drawn samples of shape (*size, N).</returns>
        /// <exception cref="ValueError">See <see cref="multivariate_normal(double[], double[,], Shape?, string, double)"/>.</exception>
        /// <remarks>
        ///     <c>x += mean</c> is NumPy's in-place add into the float64 samples, so an integer or float32 mean is cast up while
        ///     a complex mean is refused by the ufunc's same-kind output cast, as in NumPy.
        /// </remarks>
        public NDArray multivariate_normal(NDArray mean, NDArray cov, Shape? size = null,
            string check_valid = "warn", double tol = 1e-8)
        {
            // np.array(None) is a 0-d object array, so NumPy reports a null mean/cov through its shape checks.
            if (mean is null)
                throw new ValueError("mean must be 1 dimensional");
            if (cov is null)
                throw new ValueError("cov must be 2 dimensional and square");

            return MultivariateNormalCore(mean, cov, size, check_valid, tol);
        }

        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">Mean of the N-dimensional distribution.</param>
        /// <param name="cov">Covariance matrix of the distribution.</param>
        /// <param name="size">Number of samples to draw (NumPy's integer <c>size</c>).</param>
        /// <param name="check_valid">Behavior when the covariance matrix is not positive semidefinite.</param>
        /// <param name="tol">Tolerance when checking covariance matrix validity.</param>
        /// <returns>Drawn samples of shape (size, N).</returns>
        /// <exception cref="ValueError">See <see cref="multivariate_normal(double[], double[,], Shape?, string, double)"/>.</exception>
        public NDArray multivariate_normal(double[] mean, double[,] cov, int size,
            string check_valid = "warn", double tol = 1e-8)
            => multivariate_normal(mean, cov, new Shape(size), check_valid, tol);

        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">Mean of the N-dimensional distribution.</param>
        /// <param name="cov">Covariance matrix of the distribution.</param>
        /// <param name="size">Sample shape as int array.</param>
        /// <returns>Drawn samples of shape (*size, N).</returns>
        /// <exception cref="ValueError">See <see cref="multivariate_normal(double[], double[,], Shape?, string, double)"/>.</exception>
        public NDArray multivariate_normal(double[] mean, double[,] cov, int[] size)
            => multivariate_normal(mean, cov, new Shape(size));

        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">Mean of the N-dimensional distribution.</param>
        /// <param name="cov">Covariance matrix of the distribution.</param>
        /// <param name="size">Sample shape as long array.</param>
        /// <returns>Drawn samples of shape (*size, N).</returns>
        /// <exception cref="ValueError">See <see cref="multivariate_normal(double[], double[,], Shape?, string, double)"/>.</exception>
        public NDArray multivariate_normal(double[] mean, double[,] cov, long[] size)
            => multivariate_normal(mean, cov, new Shape(size));

        /// <summary>
        ///     NumPy's legacy <c>multivariate_normal</c> body over validated-shape NDArrays: draw, factor, check, transform.
        /// </summary>
        /// <param name="mean">The mean (1-D expected).</param>
        /// <param name="cov">The covariance (square 2-D expected).</param>
        /// <param name="size">The sample shape, or null for one sample.</param>
        /// <param name="check_valid">"warn", "raise" or "ignore".</param>
        /// <param name="tol">The PSD check's rtol and atol.</param>
        /// <returns>The samples, shape (*size, N).</returns>
        /// <exception cref="ValueError">A shape, <paramref name="check_valid"/> or positive-semidefiniteness check fails.</exception>
        [NDScoped] // reclaims the normals, the float64 cov, the SVD factors and the product temporaries
        private NDArray MultivariateNormalCore(NDArray mean, NDArray cov, Shape? size, string check_valid, double tol)
        {
            // Check preconditions on arguments (NumPy's order; nothing drawn yet).
            if (mean.ndim != 1)
                throw new ValueError("mean must be 1 dimensional");
            if (cov.ndim != 2 || cov.shape[0] != cov.shape[1])
                throw new ValueError("cov must be 2 dimensional and square");
            if (mean.shape[0] != cov.shape[0])
                throw new ValueError("mean and cov must have same length");

            long n = mean.shape[0];

            // final_shape = list(size) + [N]; the normals are drawn BEFORE the SVD, so every later failure has already
            // advanced the stream, exactly as in NumPy.
            long[] finalShape;
            // A default Shape is NumSharp's spelling of NumPy's size=None, like a null one.
            if (size is null || size.Value.IsEmpty)
            {
                finalShape = new[] { n };
            }
            else
            {
                var sizeVal = size.Value;
                finalShape = new long[sizeVal.NDim + 1];
                for (int d = 0; d < sizeVal.NDim; d++)
                    finalShape[d] = sizeVal.dimensions[d];
                finalShape[sizeVal.NDim] = n;
            }
            NDArray x = standard_normal(new Shape(finalShape)).reshape(-1, n);

            // GH10839, ensure double to make tol meaningful
            NDArray covD = cov.astype(np.float64, copy: true);
            var (s, v) = SvdForMultivariateNormal(covD, n);

            if (check_valid != "ignore")
            {
                if (check_valid != "warn" && check_valid != "raise")
                    throw new ValueError("check_valid must equal 'warn', 'raise', or 'ignore'");
                bool psd = np.allclose(np.dot(v.T * s, v), covD, rtol: tol, atol: tol);
                // NumPy warns (RuntimeWarning) for "warn"; NumSharp has no warnings channel, so only "raise" is observable.
                if (!psd && check_valid == "raise")
                    throw new ValueError("covariance is not symmetric positive-semidefinite.");
            }

            NDArray transformed = np.dot(x, np.sqrt(s)[":", np.newaxis] * v);
            np.add(transformed, mean, @out: transformed);   // x += mean
            return transformed.reshape(new Shape(finalShape));
        }

        /// <summary>
        ///     <c>(s, v)</c> of NumPy's <c>svd(cov)</c> for the multivariate normal: the installed LAPACK backend's
        ///     <c>gesdd</c> when there is one (byte-identical to NumPy), otherwise the managed Jacobi fallback
        ///     (<see cref="ComputeManagedSvd"/>).
        /// </summary>
        /// <param name="covD">The float64 covariance, N x N.</param>
        /// <param name="n">N.</param>
        /// <returns>The singular values (descending, length N) and <c>vh</c> (N x N).</returns>
        private static (NDArray s, NDArray v) SvdForMultivariateNormal(NDArray covD, long n)
        {
            if (n == 0)
                return (np.empty(new Shape(0L), np.float64), np.empty(new Shape(0L, 0L), np.float64));

            // Read the backend into a local once (a concurrent Disable() must not null it between test and call).
            var blas = covD.TensorEngine.Blas;
            if (blas != null && blas.TrySvd(covD, true, true, out var u, out var s, out var vh))
            {
                u?.Dispose();
                return (s, vh);
            }

            return ComputeManagedSvd(covD, n);
        }

        /// <summary>
        ///     The managed stand-in for LAPACK's SVD of a symmetric covariance: a Jacobi eigendecomposition
        ///     (<see cref="JacobiEigendecomposition"/>), eigenpairs sorted descending, eigenvector signs normalized toward
        ///     LAPACK's convention, then <c>s = |eigenvalue|</c> and <c>vh = eigenvectors^T</c>.
        /// </summary>
        /// <param name="covD">The float64 covariance, N x N (N &gt; 0).</param>
        /// <param name="n">N.</param>
        /// <returns>The singular values (descending) and <c>vh</c>.</returns>
        /// <remarks>
        ///     For a symmetric matrix <c>cov = V diag(λ) V^T</c>, the SVD is <c>U diag(|λ|) V^T</c> — so <c>vh = V^T</c> and the
        ///     singular values are <c>|λ|</c> (a negative eigenvalue flips the matching column of U, which the sampler never
        ///     reads). The PSD test then happens uniformly on <c>dot(v.T * s, v)</c>, as NumPy's does. The Jacobi work arrays
        ///     are raw pooled buffers private to this call, freed on every path.
        /// </remarks>
        private static (NDArray s, NDArray v) ComputeManagedSvd(NDArray covD, long n)
        {
            var eigenvectors = new ArraySlice<double>(new UnmanagedMemoryBlock<double>(n * n));
            var eigenvalues = new ArraySlice<double>(new UnmanagedMemoryBlock<double>(n));
            var work = new ArraySlice<double>(new UnmanagedMemoryBlock<double>(n * n));
            try
            {
                // Copy cov (C order, whatever covD's layout) into the work matrix.
                for (long i = 0; i < n; i++)
                    for (long j = 0; j < n; j++)
                        work[i * n + j] = covD.GetDouble(i, j);

                // Initialize eigenvectors to identity
                for (long i = 0; i < n; i++)
                    for (long j = 0; j < n; j++)
                        eigenvectors[i * n + j] = (i == j) ? 1.0 : 0.0;

                JacobiEigendecomposition(work, eigenvectors, eigenvalues, n, 100, 1e-12);

                // Sort eigenvalues in DESCENDING order (to match NumPy SVD) and reorder eigenvectors accordingly.
                SortEigenDescending(eigenvalues, eigenvectors, n);

                // Normalize eigenvector signs to match NumPy/LAPACK SVD convention.
                NormalizeEigenvectorSigns(eigenvectors, n);

                var s = new NDArray(NPTypeCode.Double, new Shape(n), false);
                var vh = new NDArray(NPTypeCode.Double, new Shape(n, n), false);
                unsafe
                {
                    var sp = (double*)s.Address;
                    var vp = (double*)vh.Address;
                    for (long i = 0; i < n; i++)
                    {
                        sp[i] = Math.Abs(eigenvalues[i]);
                        // vh[i, j] = V[j, i]: row i of vh is eigenvector i.
                        for (long j = 0; j < n; j++)
                            vp[i * n + j] = eigenvectors[j * n + i];
                    }
                }
                return (s, vh);
            }
            finally
            {
                work.DangerousFree();
                eigenvalues.DangerousFree();
                eigenvectors.DangerousFree();
            }
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
