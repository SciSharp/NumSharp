// The ten kernels of DefaultLinalgEngine: NumPy's _umath_linalg gufuncs (cholesky_lo / cholesky_up, det, slogdet,
// eig / eigvals, eigh / eigvalsh, inv, solve / solve1, lstsq, the qr gufuncs, the svd gufuncs), which numpy.linalg's
// Python statements call once they have validated and cast the operands. Each answers an empty operand itself, then
// asks the operand engine's Blas, then falls back (the managed LU for the four LU-based ones, an exception for the
// rest). They were TensorEngine's virtual members of the same names until phase PE's E2 of
// docs/plans/np-layer-dtype-logic.md moved them here; a provider overrides them to compute the factorisations another way.

using System;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <summary>
        ///     The <c>cholesky_lo</c> / <c>cholesky_up</c> gufunc: OpenBLAS <c>potrf</c> through the operand engine's
        ///     <see cref="TensorEngine.Blas"/>; an empty operand is <see cref="LinAlgEmpty.Cholesky"/>, with or without a
        ///     backend.
        /// </summary>
        /// <param name="a">The Hermitian positive-definite matrix or stack, validated (square, at least two axes) and cast
        ///     to the loop dtype (float32, float64 or complex128) by <see cref="Cholesky"/>.</param>
        /// <param name="upper">Return the upper factor (NumPy's <c>upper=True</c>) rather than the lower one.</param>
        /// <returns>The triangular factor, a fresh array of <paramref name="a"/>'s dtype and shape.</returns>
        /// <exception cref="LinAlgError">A matrix that is not positive definite (<c>Matrix is not positive definite</c>,
        ///     from the factorisation).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no backend serves: Core carries no
        ///     managed Cholesky.</exception>
        /// <remarks>An override replaces the factorisation for every <c>np.linalg.cholesky</c> call of the engine whose
        ///     <see cref="TensorEngine.Linalg"/> it is; it receives empty operands too (the base kernel answers them).</remarks>
        protected virtual NDArray CholeskyKernel(NDArray a, bool upper)
        {
            // An empty operand (an empty matrix or a zero-size batch) factorises nothing: NumPy's answer needs no LAPACK, so
            // Core gives it whether or not a backend is installed (LinAlgEmpty, probed against NumPy 2.4.2).
            if (a.size == 0)
                return LinAlgEmpty.Cholesky(a);
            // Read once into a local: Blas is a settable property, and a concurrent OpenBlasEngine.Disable() turns a
            // test-then-call into a null dereference (docs/stale-docs/GEMM_PARITY.md section 9).
            var blas = a.TensorEngine.Blas;
            if (blas != null && blas.TryCholesky(a, upper, out var result))
                return result;
            throw MissingBackend(a, "np.linalg.cholesky", nameof(IBlasBackend.TryCholesky));
        }

        /// <summary>
        ///     The <c>det</c> gufunc: OpenBLAS <c>getrf</c> through the operand engine's <see cref="TensorEngine.Blas"/>, or
        ///     NumSharp's managed LU (<see cref="ManagedLu"/>) when no backend serves the operand.
        /// </summary>
        /// <param name="a">The square matrix or stack, validated and cast to the loop dtype by <see cref="Det"/>.</param>
        /// <returns>The determinant: for one matrix NumPy's scalar (a read-only 0-d array, the gufunc's return step), for a
        ///     stack a writeable array of one determinant per matrix.</returns>
        /// <remarks>
        ///     NumPy's det is the gufunc <c>_umath_linalg.det</c>, and the ufunc machinery returns a 0-d output as a numpy
        ///     scalar; <see cref="NDArray.MarkReductionScalar"/> is that step for both routes (a no-op for a stack's 1-d or
        ///     larger result). The managed LU is allclose to NumPy's blocked LAPACK, not bit-exact: a matrix whose zero
        ///     pivot only appears after several eliminations can give a tiny determinant where LAPACK cancels to 0.
        /// </remarks>
        protected virtual NDArray DetKernel(NDArray a)
        {
            // One read of the settable property (see CholeskyKernel).
            var blas = a.TensorEngine.Blas;
            if (blas != null && blas.TryDet(a, out var result))
                return result.MarkReductionScalar();
            return ManagedLu.Det(a).MarkReductionScalar();
        }

        /// <summary>
        ///     The <c>slogdet</c> gufunc: OpenBLAS <c>getrf</c> through the operand engine's <see cref="TensorEngine.Blas"/>,
        ///     or NumSharp's managed LU (<see cref="ManagedLu"/>) when no backend serves the operand.
        /// </summary>
        /// <param name="a">The square matrix or stack, validated and cast to the loop dtype by <see cref="Slogdet"/>.</param>
        /// <returns>The sign and the log of the absolute determinant: NumPy's scalars for one matrix (read-only 0-d arrays,
        ///     the gufunc's return step), writeable arrays for a stack. A singular matrix gives <c>(0, -inf)</c>.</returns>
        /// <remarks>
        ///     NumPy's slogdet is the gufunc <c>_umath_linalg.slogdet</c>: both 0-d outputs come back as numpy scalars
        ///     (<see cref="NDArray.MarkReductionScalar"/>, a no-op for a stack's outputs).
        /// </remarks>
        protected virtual (NDArray sign, NDArray logabsdet) SlogdetKernel(NDArray a)
        {
            // One read of the settable property (see CholeskyKernel).
            var blas = a.TensorEngine.Blas;
            if (blas != null && blas.TrySlogdet(a, out var sign, out var logabsdet))
                return (sign.MarkReductionScalar(), logabsdet.MarkReductionScalar());
            var (managedSign, managedLogabsdet) = ManagedLu.Slogdet(a);
            return (managedSign.MarkReductionScalar(), managedLogabsdet.MarkReductionScalar());
        }

        /// <summary>
        ///     The <c>eig</c> / <c>eigvals</c> gufunc: OpenBLAS <c>geev</c> through the operand engine's
        ///     <see cref="TensorEngine.Blas"/>, always complex128 (<see cref="Eig"/> collapses a real result); an empty
        ///     operand is <see cref="LinAlgEmpty.Eig"/>, with or without a backend.
        /// </summary>
        /// <param name="a">The square matrix or stack, validated (finite included) and cast to the loop dtype.</param>
        /// <param name="computeVectors">Whether the eigenvectors are wanted; the second slot is null otherwise.</param>
        /// <returns>The complex128 eigenvalues and, when <paramref name="computeVectors"/> is set, the eigenvectors (the
        ///     columns).</returns>
        /// <exception cref="LinAlgError">The eigenvalue iteration did not converge (<c>Eigenvalues did not
        ///     converge</c>).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no backend serves: Core carries no
        ///     managed eigensolver.</exception>
        protected virtual (NDArray eigenvalues, NDArray eigenvectors) EigKernel(NDArray a, bool computeVectors)
        {
            // An empty operand (an empty matrix or a zero-size batch) factorises nothing: NumPy's answer needs no LAPACK, so
            // Core gives it whether or not a backend is installed (LinAlgEmpty, probed against NumPy 2.4.2).
            if (a.size == 0)
                return LinAlgEmpty.Eig(a, computeVectors);
            // One read of the settable property (see CholeskyKernel).
            var blas = a.TensorEngine.Blas;
            if (blas != null && blas.TryEig(a, computeVectors, out var w, out var v))
                return (w, v);
            throw MissingBackend(a, computeVectors ? "np.linalg.eig" : "np.linalg.eigvals", nameof(IBlasBackend.TryEig));
        }

        /// <summary>
        ///     The <c>eigh</c> / <c>eigvalsh</c> gufunc: OpenBLAS <c>syevd</c> (real symmetric) or <c>heevd</c> (complex
        ///     Hermitian) through the operand engine's <see cref="TensorEngine.Blas"/>; an empty operand is
        ///     <see cref="LinAlgEmpty.Eigh"/>, with or without a backend.
        /// </summary>
        /// <param name="a">The symmetric / Hermitian matrix or stack, validated and cast to the loop dtype.</param>
        /// <param name="uplo">The triangle read, <c>'L'</c> or <c>'U'</c> (upper-cased by <see cref="Eigh"/>).</param>
        /// <param name="computeVectors">Whether the eigenvectors are wanted; the second slot is null otherwise.</param>
        /// <returns>The real eigenvalues, ascending, and (when <paramref name="computeVectors"/> is set) the
        ///     eigenvectors.</returns>
        /// <exception cref="LinAlgError">The eigenvalue iteration did not converge.</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no backend serves.</exception>
        protected virtual (NDArray eigenvalues, NDArray eigenvectors) EighKernel(NDArray a, char uplo, bool computeVectors)
        {
            // An empty operand (an empty matrix or a zero-size batch) factorises nothing: NumPy's answer needs no LAPACK, so
            // Core gives it whether or not a backend is installed (LinAlgEmpty, probed against NumPy 2.4.2).
            if (a.size == 0)
                return LinAlgEmpty.Eigh(a, computeVectors);
            // One read of the settable property (see CholeskyKernel).
            var blas = a.TensorEngine.Blas;
            if (blas != null && blas.TryEigh(a, uplo, computeVectors, out var w, out var v))
                return (w, v);
            throw MissingBackend(a, computeVectors ? "np.linalg.eigh" : "np.linalg.eigvalsh", nameof(IBlasBackend.TryEigh));
        }

        /// <summary>
        ///     The <c>inv</c> gufunc: OpenBLAS <c>gesv</c> against the identity through the operand engine's
        ///     <see cref="TensorEngine.Blas"/>, or NumSharp's managed LU (<see cref="ManagedLu"/>) when no backend serves the
        ///     operand.
        /// </summary>
        /// <param name="a">The square matrix or stack, validated and cast to the loop dtype.</param>
        /// <returns>The inverse, a fresh array of <paramref name="a"/>'s dtype and shape.</returns>
        /// <exception cref="LinAlgError">A singular matrix (<c>Singular matrix</c>, from the factorisation, as NumPy's
        ///     <c>gesv</c> reports it: no determinant test).</exception>
        /// <remarks><see cref="Cond"/> calls this kernel too, per matrix, and turns a singular one into NaN as NumPy's
        ///     <c>errstate(all='ignore')</c> does: an override must keep the <c>Singular matrix</c> text of its
        ///     <see cref="LinAlgError"/> for that to work.</remarks>
        protected virtual NDArray InvKernel(NDArray a)
        {
            // One read of the settable property (see CholeskyKernel).
            var blas = a.TensorEngine.Blas;
            if (blas != null && blas.TryInv(a, out var result))
                return result;
            return ManagedLu.Inv(a);
        }

        /// <summary>
        ///     The <c>solve</c> / <c>solve1</c> gufunc: OpenBLAS <c>gesv</c> through the operand engine's
        ///     <see cref="TensorEngine.Blas"/>, or NumSharp's managed LU (<see cref="ManagedLu"/>) when no backend serves the
        ///     operands.
        /// </summary>
        /// <param name="a">The coefficient matrix or stack <c>(..., M, M)</c>, validated and cast to the loop dtype.</param>
        /// <param name="b">The right-hand side, cast to the loop dtype: <c>(..., M)</c> when
        ///     <paramref name="oneDimensionalRhs"/> is set, else <c>(..., M, K)</c>; its loop axes broadcast against
        ///     <paramref name="a"/>'s (checked by <see cref="Solve"/>).</param>
        /// <param name="oneDimensionalRhs">NumPy's choice of the <c>(m,m),(m)-&gt;(m)</c> gufunc: <paramref name="b"/> was
        ///     exactly one-dimensional.</param>
        /// <returns>The solution, a fresh array of the loop dtype.</returns>
        /// <exception cref="LinAlgError">A singular matrix (<c>Singular matrix</c>).</exception>
        protected virtual NDArray SolveKernel(NDArray a, NDArray b, bool oneDimensionalRhs)
        {
            // One read of the settable property (see CholeskyKernel).
            var blas = a.TensorEngine.Blas;
            if (blas != null && blas.TrySolve(a, b, oneDimensionalRhs, out var result))
                return result;
            return ManagedLu.Solve(a, b, oneDimensionalRhs);
        }

        /// <summary>
        ///     The <c>lstsq</c> gufunc: OpenBLAS <c>gelsd</c> through the operand engine's <see cref="TensorEngine.Blas"/>;
        ///     an empty coefficient matrix is <see cref="LinAlgEmpty.Lstsq"/>, with or without a backend.
        /// </summary>
        /// <param name="a">The coefficient matrix <c>(M, N)</c>, cast to the loop dtype.</param>
        /// <param name="b">The right-hand sides <c>(M, K)</c>, K &gt;= 1 (<see cref="Lstsq"/> pads K == 0), cast to the loop
        ///     dtype.</param>
        /// <param name="rcond">The singular-value cutoff ratio (NumPy's default already applied).</param>
        /// <returns>The solution, the residuals, the rank and the singular values, before <see cref="Lstsq"/>'s casts; the
        ///     rank is NumPy's scalar (a read-only 0-d array, the gufunc's return step).</returns>
        /// <exception cref="LinAlgError">The SVD did not converge (<c>SVD did not converge in Linear Least
        ///     Squares</c>).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty coefficient matrix that no backend serves.</exception>
        /// <remarks>
        ///     NumPy's lstsq calls the gufunc <c>_umath_linalg.lstsq</c>, whose 0-d rank output the ufunc machinery returns
        ///     as a numpy scalar (<c>np.intc</c>); <see cref="NDArray.MarkReductionScalar"/> is that step for both routes.
        /// </remarks>
        protected virtual (NDArray Solution, NDArray Residuals, NDArray Rank, NDArray SingularValues) LstsqKernel(
            NDArray a, NDArray b, double rcond)
        {
            // An empty operand (an empty matrix or a zero-size batch) factorises nothing: NumPy's answer needs no LAPACK, so
            // Core gives it whether or not a backend is installed (LinAlgEmpty, probed against NumPy 2.4.2).
            if (a.size == 0)
            {
                var (emptyX, emptyRes, emptyRank, emptyS) = LinAlgEmpty.Lstsq(a, b);
                return (emptyX, emptyRes, emptyRank.MarkReductionScalar(), emptyS);
            }
            // One read of the settable property (see CholeskyKernel).
            var blas = a.TensorEngine.Blas;
            if (blas != null && blas.TryLstsq(a, b, rcond, out var x, out var res, out var rank, out var s))
                return (x, res, rank.MarkReductionScalar(), s);
            throw MissingBackend(a, "np.linalg.lstsq", nameof(IBlasBackend.TryLstsq));
        }

        /// <summary>
        ///     The qr gufuncs (<c>qr_r_raw</c> plus <c>qr_reduced</c> / <c>qr_complete</c>): OpenBLAS <c>geqrf</c> plus
        ///     <c>orgqr</c> / <c>ungqr</c> through the operand engine's <see cref="TensorEngine.Blas"/>; an empty operand is
        ///     <see cref="LinAlgEmpty.Qr"/>, with or without a backend.
        /// </summary>
        /// <param name="a">The matrix or stack, validated and cast to the loop dtype.</param>
        /// <param name="mode">The resolved mode: <c>reduced</c>, <c>complete</c>, <c>r</c>, <c>raw</c> or
        ///     <c>economic</c>.</param>
        /// <returns>The mode's pair: (Q, R), (null, R), (h, tau) or (packed, null), C-ordered; <see cref="Qr"/> turns the raw
        ///     and economic factors into NumPy's objects.</returns>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no backend serves.</exception>
        protected virtual (NDArray Q, NDArray R) QrKernel(NDArray a, string mode)
        {
            // An empty operand (an empty matrix or a zero-size batch) factorises nothing: NumPy's answer needs no LAPACK, so
            // Core gives it whether or not a backend is installed (LinAlgEmpty, probed against NumPy 2.4.2).
            if (a.size == 0)
                return LinAlgEmpty.Qr(a, mode);
            // One read of the settable property (see CholeskyKernel).
            var blas = a.TensorEngine.Blas;
            if (blas != null && blas.TryQr(a, mode, out var q, out var r))
                return (q, r);
            throw MissingBackend(a, "np.linalg.qr", nameof(IBlasBackend.TryQr));
        }

        /// <summary>
        ///     The svd gufuncs (<c>svd_f</c>, <c>svd_s</c>, <c>svd</c>): OpenBLAS <c>gesdd</c> through the operand engine's
        ///     <see cref="TensorEngine.Blas"/>. Also behind pinv, matrix_rank, cond and the spectral / nuclear norms. An
        ///     empty operand is <see cref="LinAlgEmpty.Svd"/>, with or without a backend.
        /// </summary>
        /// <param name="a">The matrix or stack, validated and cast to the loop dtype.</param>
        /// <param name="fullMatrices">NumPy's <c>full_matrices</c>: square U and Vh, else <c>min(M, N)</c> columns /
        ///     rows.</param>
        /// <param name="computeUv">NumPy's <c>compute_uv</c>; the U and Vh slots are null otherwise.</param>
        /// <returns>U, the real singular values (descending) and Vh.</returns>
        /// <exception cref="LinAlgError">The SVD did not converge (<c>SVD did not converge</c>).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no backend serves: Core carries no
        ///     managed SVD.</exception>
        protected virtual (NDArray U, NDArray S, NDArray Vh) SvdKernel(NDArray a, bool fullMatrices, bool computeUv)
        {
            // An empty operand (an empty matrix or a zero-size batch) factorises nothing: NumPy's answer needs no LAPACK, so
            // Core gives it whether or not a backend is installed (LinAlgEmpty, probed against NumPy 2.4.2).
            if (a.size == 0)
                return LinAlgEmpty.Svd(a, fullMatrices, computeUv);
            // One read of the settable property (see CholeskyKernel).
            var blas = a.TensorEngine.Blas;
            if (blas != null && blas.TrySvd(a, fullMatrices, computeUv, out var u, out var s, out var vh))
                return (u, s, vh);
            throw MissingBackend(a, "np.linalg.svd", nameof(IBlasBackend.TrySvd));
        }

        /// <summary>
        ///     The one message every unserved factorisation raises. It names the NumPy API, the seam member a backend has to
        ///     implement, and the package that supplies one, so the exception says what to install rather than merely that
        ///     something is missing.
        /// </summary>
        /// <param name="a">The operand whose engine was asked: the message names that engine's type, the holder of the
        ///     <see cref="TensorEngine.Blas"/> property a backend is assigned to.</param>
        /// <param name="api">The public function (<c>np.linalg.svd</c>, ...).</param>
        /// <param name="seamMember">The <see cref="IBlasBackend"/> member that would have served the call.</param>
        /// <returns>The exception to throw (not thrown here, so the kernels' <c>throw</c> statements keep their stack).</returns>
        /// <remarks>
        ///     There is no separate seam for the factorisations: they ride the very same <see cref="TensorEngine.Blas"/>
        ///     property as the matrix products, filled by a single <c>NumSharp.Interop.OpenBLAS</c> reference, so the message
        ///     points at that package rather than at "a backend" in the abstract. Before phase PE's E2 the kernels were
        ///     <see cref="TensorEngine"/> members and the message named that engine's own type; naming the operand's engine
        ///     keeps the text the same.
        /// </remarks>
        private static OpenBlasMissingBackendException MissingBackend(NDArray a, string api, string seamMember)
            => new OpenBlasMissingBackendException(
                $"{api} requires a matrix backend, and {a.TensorEngine.GetType().Name} has none installed or the " +
                $"installed one declined these operands. NumSharp.Core is 100 % managed C# and ships " +
                $"no matrix factorisation of its own, so — unlike the matrix products, which always " +
                $"have a managed kernel to fall back to — there is nothing to compute this with. " +
                $"{OpenBlasMissingBackendException.HowToFix} The backend serves this operation through " +
                $"its IBlasBackend.{seamMember}.");
    }
}
