namespace NumSharp.Backends
{
    /// <summary>
    ///     The linear-algebra half of the BLAS seam: the products NumPy routes through CBLAS beyond
    ///     <c>dot</c>/<c>matmul</c>, and the factorisations it routes through LAPACK.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Every member here is a DEFAULT implementation returning false.</b> That is what lets
    ///     this file exist at all without breaking the one backend that ships today
    ///     (<c>NumSharp.Interop.OpenBLAS</c>'s <c>OpenBlasBackend</c>, which answers only for
    ///     <c>dot</c>/<c>matmul</c> at float32/float64) and any third-party implementation — a
    ///     backend mentions only the members it serves, exactly as with
    ///     <see cref="IBlasBackend.TryMatMulBatched"/>.
    ///     </para>
    ///     <para>
    ///     <b>These two groups differ in one important way, and it is not their shape.</b> The
    ///     product family (<c>inner</c>/<c>vdot</c>/<c>vecdot</c>/<c>matvec</c>/<c>vecmat</c>) keeps
    ///     the invariant the rest of this interface states: NumSharp computes them with its own
    ///     managed kernels, so a backend changes WHICH implementation runs, never WHETHER the answer
    ///     can be produced. The factorisation family CANNOT keep it — NumSharp.Core ships no managed
    ///     LU, QR, SVD or eigensolver, so with no backend installed (or one that declines) the engine
    ///     has nothing to fall back to and raises <see cref="NumSharp.OpenBlasMissingBackendException"/>.
    ///     Callers of
    ///     <c>np.linalg</c> must expect that; callers of the product family need not.
    ///     </para>
    ///     <para>
    ///     <b>Reading the operands</b> follows the rule on <see cref="IBlasBackend"/> exactly:
    ///     <c>(T*)a.GetData().Address + a.Shape.Offset</c> paired with <c>a.Shape.Strides</c>
    ///     (element strides, not bytes). LAPACK additionally wants column-major input, so an
    ///     implementation will usually transpose or ask LAPACKE for row-major order; the engine hands
    ///     over the operands untouched rather than guessing which a backend prefers.
    ///     </para>
    ///     <para>
    ///     <b>Stacked operands.</b> NumPy's linalg entry points are gufuncs over the trailing one or
    ///     two axes, so every operand here may carry leading batch dimensions. A backend that only
    ///     serves a single matrix returns false for a stacked one.
    ///     </para>
    /// </remarks>
    public partial interface IBlasBackend
    {
        #region Products (CBLAS)

        /// <summary>
        ///     Computes <c>np.inner(a, b)</c> — a sum product over the LAST axis of both operands.
        /// </summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryInner(NDArray a, NDArray b, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.vdot(a, b)</c> — both operands flattened to 1-D, <paramref name="a"/>
        ///     conjugated when complex, then a vector dot product. Always a 0-d result.
        /// </summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryVdot(NDArray a, NDArray b, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes the <c>np.vecdot</c> gufunc <c>(n),(n)->()</c>, allocating the result —
        ///     <paramref name="x1"/> conjugated when complex.
        /// </summary>
        /// <remarks>
        ///     Allocating rather than filling a caller-provided buffer (the shape
        ///     <see cref="IBlasBackend.TryMatMul2D"/> takes) is deliberate for the three gufuncs: the
        ///     loop shape is the broadcast of the operands' leading axes, and making the engine
        ///     derive it up front would compute it twice on the path where the backend declines.
        /// </remarks>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryVecdot(NDArray x1, NDArray x2, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes the <c>np.matvec</c> gufunc <c>(m,n),(n)->(m)</c>, allocating the result —
        ///     NumPy's <c>gemv</c> route.
        /// </summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryMatvec(NDArray x1, NDArray x2, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes the <c>np.vecmat</c> gufunc <c>(n),(n,m)->(m)</c>, allocating the result —
        ///     <paramref name="x1"/> conjugated when complex.
        /// </summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryVecmat(NDArray x1, NDArray x2, out NDArray result)
        {
            result = null;
            return false;
        }

        #endregion

        #region Factorisations (LAPACK)

        /// <summary>Computes <c>np.linalg.cholesky</c> (LAPACK <c>potrf</c>).</summary>
        /// <param name="upper">False for the lower-triangular factor (NumPy's default).</param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryCholesky(NDArray a, bool upper, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>Computes <c>np.linalg.det</c> (LAPACK <c>getrf</c>).</summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryDet(NDArray a, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.slogdet</c> (LAPACK <c>getrf</c>) — the sign of the determinant
        ///     and the natural log of its absolute value, which stays finite where <c>det</c>
        ///     overflows.
        /// </summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TrySlogdet(NDArray a, out NDArray sign, out NDArray logabsdet)
        {
            sign = null;
            logabsdet = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.eig</c> / <c>np.linalg.eigvals</c> (LAPACK <c>geev</c>) for a
        ///     general, not necessarily symmetric, matrix.
        /// </summary>
        /// <param name="computeVectors">
        ///     False for <c>eigvals</c> — <paramref name="eigenvectors"/> is then null.
        /// </param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryEig(NDArray a, bool computeVectors, out NDArray eigenvalues, out NDArray eigenvectors)
        {
            eigenvalues = null;
            eigenvectors = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.eigh</c> / <c>np.linalg.eigvalsh</c> (LAPACK <c>syevd</c> for a
        ///     real symmetric operand, <c>heevd</c> for a complex Hermitian one).
        /// </summary>
        /// <param name="uplo">'L' or 'U' — which triangle holds the data.</param>
        /// <param name="computeVectors">False for <c>eigvalsh</c>.</param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryEigh(NDArray a, char uplo, bool computeVectors, out NDArray eigenvalues, out NDArray eigenvectors)
        {
            eigenvalues = null;
            eigenvectors = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.inv</c> — NumPy solves <c>a x = I</c> with <c>gesv</c> rather
        ///     than calling an explicit inversion routine.
        /// </summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryInv(NDArray a, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.lstsq</c> (LAPACK <c>gelsd</c>) — the least-squares solution
        ///     plus residuals, the effective rank and the singular values.
        /// </summary>
        /// <param name="rcond">Singular-value cutoff, relative to the largest singular value.</param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryLstsq(NDArray a, NDArray b, double rcond,
            out NDArray solution, out NDArray residuals, out NDArray rank, out NDArray singularValues)
        {
            solution = null;
            residuals = null;
            rank = null;
            singularValues = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.qr</c> (LAPACK <c>geqrf</c> plus <c>orgqr</c>/<c>ungqr</c>).
        /// </summary>
        /// <param name="mode">
        ///     Already validated to one of "reduced", "complete", "r", "raw". For "r" only
        ///     <paramref name="r"/> is produced; for "raw" the pair is LAPACK's packed
        ///     (<c>h</c>, <c>tau</c>) rather than (Q, R).
        /// </param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryQr(NDArray a, string mode, out NDArray q, out NDArray r)
        {
            q = null;
            r = null;
            return false;
        }

        /// <summary>Computes <c>np.linalg.solve</c> (LAPACK <c>gesv</c>).</summary>
        /// <param name="oneDimensionalRhs">
        ///     True when <paramref name="b"/> entered as a single vector — NumPy dispatches that to
        ///     its <c>solve1</c> gufunc <c>(m,m),(m)->(m)</c> rather than <c>(m,m),(m,n)->(m,n)</c>.
        /// </param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TrySolve(NDArray a, NDArray b, bool oneDimensionalRhs, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.svd</c> / <c>np.linalg.svdvals</c> (LAPACK <c>gesdd</c>).
        /// </summary>
        /// <param name="computeUv">
        ///     False for the singular values alone — <paramref name="u"/> and <paramref name="vh"/>
        ///     are then null.
        /// </param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TrySvd(NDArray a, bool fullMatrices, bool computeUv, out NDArray u, out NDArray s, out NDArray vh)
        {
            u = null;
            s = null;
            vh = null;
            return false;
        }

        #endregion
    }
}
