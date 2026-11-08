using NumSharp;

namespace NumSharpOpenBlasDefaultExample;

/// <summary>
///     The np.linalg functions that need LAPACK, which is inside the OpenBLAS library of the
///     package: linear systems, the four factorizations, least squares, rank and condition number.
/// </summary>
/// <remarks>
///     NumSharp alone has an LU solver (solve, inv, det, slogdet), but no Cholesky, QR, eigenvalue
///     or singular value routine. Those functions raise <c>OpenBlasMissingBackendException</c>
///     until the backend of the package is installed. That is the reason for the first statement
///     of Program.cs.
/// </remarks>
internal static class LinearAlgebra
{
    /// <summary>
    ///     Runs each function once and measures how well each result satisfies the identity that
    ///     defines it.
    /// </summary>
    /// <returns>
    ///     The largest of those errors, as a Frobenius norm. It is a small multiple of 1e-16, and
    ///     its last digits differ between machines, because OpenBLAS selects its kernels for the
    ///     processor and splits the work across threads.
    /// </returns>
    /// <exception cref="OpenBlasMissingBackendException">
    ///     The backend is not installed, so no library can compute a factorization.
    /// </exception>
    public static double Run()
    {
        using var scope = NDScope.Open();
        double worst = 0;

        NDArray a = np.array(new double[,] { { 4, 1, 2 }, { 1, 3, 0 }, { 2, 0, 5 } });   // symmetric and positive definite
        NDArray b = np.array(new double[] { 7, 4, 7 });

        // ---- a linear system ------------------------------------------------------------------
        NDArray x = np.linalg.solve(a, b);                               // [1, 1, 1]
        NDArray determinant = np.linalg.det(a);                          // 43.0
        NDArray inverse = np.linalg.inv(a);
        (NDArray sign, NDArray logDeterminant) = np.linalg.slogdet(a);   // 1.0 and 3.7612, which is log(43)
        worst = Math.Max(worst, Error(np.dot(a, x) - b));
        worst = Math.Max(worst, Error(np.matmul(a, inverse) - np.eye(3)));

        // ---- the factorizations ---------------------------------------------------------------
        NDArray lower = np.linalg.cholesky(a);                           // lower triangle: [[2, 0, 0], [0.5, 1.6583, 0], [1, -0.3015, 1.9771]]
        worst = Math.Max(worst, Error(np.matmul(lower, lower.T) - a));

        (NDArray q, NDArray r) = np.linalg.qr(a);                        // q has orthonormal columns, r is upper triangular
        worst = Math.Max(worst, Error(np.matmul(q, r) - a));

        (NDArray eigenvalues, NDArray eigenvectors) = np.linalg.eigh(a); // [1.8549, 3.4760, 6.6691], in ascending order
        worst = Math.Max(worst, Error(np.matmul(a, eigenvectors) - eigenvectors * eigenvalues));

        (NDArray u, NDArray singular, NDArray vt) = np.linalg.svd(a);    // [6.6691, 3.4760, 1.8549], in descending order
        worst = Math.Max(worst, Error(np.matmul(u * singular, vt) - a));

        // ---- a least-squares fit of a line ----------------------------------------------------
        NDArray t = np.arange(10.0);
        NDArray y = 2.0 * t + 1.0;
        NDArray design = np.stack(new[] { t, np.ones_like(t) }, 1);      // (10, 2): one row [t, 1] for each sample
        (NDArray line, NDArray squaredError, NDArray rank, NDArray spectrum) = np.linalg.lstsq(design, y);
        //    line is [2, 1]: slope and intercept.  rank is 2.  spectrum is [17.0932, 1.6804].
        worst = Math.Max(worst, Error(line - np.array(new double[] { 2, 1 })));

        NDArray pseudoInverse = np.linalg.pinv(design);                  // (2, 10); pseudoInverse @ y is [2, 1] also

        // ---- rank, condition number, norm -----------------------------------------------------
        NDArray fullRank = np.linalg.matrix_rank(a);                     // 3
        NDArray rankOne = np.linalg.matrix_rank(np.ones((3, 3)));        // 1
        NDArray condition = np.linalg.cond(a);                           // 3.5954 = 6.6691 / 1.8549
        NDArray spectralNorm = np.linalg.norm(a, 2);                     // 6.6691, the largest singular value

        // ---- a matrix that is not symmetric can have complex eigenvalues -----------------------
        NDArray quarterTurn = np.array(new double[,] { { 0, -1 }, { 1, 0 } });
        (NDArray roots, NDArray modes) = np.linalg.eig(quarterTurn);     // complex128 [0+1j, 0-1j]

        return worst;
    }

    /// <summary>The Frobenius norm of <paramref name="difference"/>: how far a result is from its identity.</summary>
    /// <param name="difference">The computed value minus the value that the identity requires.</param>
    /// <returns>The square root of the sum of the squared elements.</returns>
    private static double Error(NDArray difference) => np.linalg.norm(difference).GetDouble();
}
