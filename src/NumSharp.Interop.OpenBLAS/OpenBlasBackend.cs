using System;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     The <see cref="IBlasBackend"/> this package installs: NumSharp's matrix products computed
    ///     by an external CBLAS library, through a route-for-route port of NumPy's two
    ///     matrix-product dispatchers.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     It answers for <c>float32</c> and <c>float64</c> only, and returns false for everything
    ///     else so the engine keeps computing it with its own managed kernels — integer and bool
    ///     products do not need a BLAS anyway (modular integer addition is associative, so summation
    ///     order cannot change the result).
    ///     </para>
    ///     <para>
    ///     <b>Why the two entry points are ported separately.</b> <c>np.dot</c> and <c>np.matmul</c>
    ///     are not the same C code in NumPy: <c>cblas_matrixproduct</c> (+ the N-D <c>dotfunc</c>
    ///     tail) versus the <c>@TYPE@_matmul</c> gufunc. They agree bit-for-bit on nearly every
    ///     input, but pick different routes when an operand is not blasable — a stride-2 matrix
    ///     times a vector gets gemv-on-a-copy from one and the portable loop from the other, and
    ///     278/300 elements differ. A backend claiming bit-parity must reproduce that split, which
    ///     is why <see cref="IBlasBackend"/> has both <see cref="TryDot"/> and
    ///     <see cref="TryMatMul2D"/> rather than one matrix-product method.
    ///     </para>
    /// </remarks>
    public sealed class OpenBlasBackend : IBlasBackend
    {
        /// <inheritdoc/>
        public string Info
        {
            get
            {
                if (!OpenBlasNative.IsLoaded)
                    return "<no CBLAS library loaded>";

                var config = OpenBlasNative.GetConfig();
                return $"{OpenBlasNative.LibraryPath} [symbols {OpenBlasNative.SymbolScheme}, " +
                       $"{(OpenBlasNative.IsIlp64 ? "ILP64" : "LP64")}, threads {OpenBlasNative.GetNumThreads()}]" +
                       (config == null ? string.Empty : " " + config);
            }
        }

        /// <inheritdoc/>
        public bool TryDot(NDArray left, NDArray right, out NDArray result)
            => OpenBlasEngine.TryDot(left, right, out result);

        /// <inheritdoc/>
        public bool TryMatMul2D(NDArray left, NDArray right, NDArray result)
            => OpenBlasEngine.TryMatmul2D(left, right, result);

        /// <inheritdoc/>
        public bool TryMatMulBatched(NDArray left, NDArray right, NDArray result)
            => OpenBlasEngine.TryMatmulBatched(left, right, result);

        #region LU factorisations (LAPACK getrf / gesv)

        // These answer only when the loaded library also exports LAPACK (a full OpenBLAS does; a bare
        // reference CBLAS does not). A miss returns false and the engine raises its "needs a LAPACK
        // backend" — the same outcome as no backend at all. float32 operands are computed in double
        // and cast back, exactly as NumPy's linalg does; every other dtype falls through.

        /// <inheritdoc/>
        public bool TryInv(NDArray a, out NDArray result)
            => OpenBlasEngine.TryInv(a, out result);

        /// <inheritdoc/>
        public bool TrySolve(NDArray a, NDArray b, bool oneDimensionalRhs, out NDArray result)
            => OpenBlasEngine.TrySolve(a, b, oneDimensionalRhs, out result);

        /// <inheritdoc/>
        public bool TryDet(NDArray a, out NDArray result)
            => OpenBlasEngine.TryDet(a, out result);

        /// <inheritdoc/>
        public bool TrySlogdet(NDArray a, out NDArray sign, out NDArray logabsdet)
            => OpenBlasEngine.TrySlogdet(a, out sign, out logabsdet);

        #endregion

        #region Cholesky / QR factorisations (LAPACK potrf / geqrf / orgqr)

        // Same story as the LU family: they answer only when the loaded library exports LAPACK, compute
        // float32 in double and cast back exactly as NumPy does, and decline every other dtype so the
        // engine raises its "needs a LAPACK backend". A not-positive-definite / bad-argument matrix is a
        // genuine LinAlgError from the routine (the operand WAS served), not a decline — it propagates.

        /// <inheritdoc/>
        public bool TryCholesky(NDArray a, bool upper, out NDArray result)
            => OpenBlasEngine.TryCholesky(a, upper, out result);

        /// <inheritdoc/>
        public bool TryQr(NDArray a, string mode, out NDArray q, out NDArray r)
            => OpenBlasEngine.TryQr(a, mode, out q, out r);

        #endregion

        #region SVD / least-squares factorisations (LAPACK gesdd / gelsd)

        // Same story again: they answer only when the loaded library exports LAPACK, compute float32 in
        // double and cast back exactly as NumPy does, and decline every other dtype so the engine raises
        // its "needs a LAPACK backend". A non-convergence is a genuine LinAlgError from the routine (the
        // operand WAS served), not a decline — it propagates. TrySvd is also what turns on pinv,
        // matrix_rank, cond and the spectral/nuclear matrix norms, which reconstruct on top of it.

        /// <inheritdoc/>
        public bool TrySvd(NDArray a, bool fullMatrices, bool computeUv, out NDArray u, out NDArray s, out NDArray vh)
            => OpenBlasEngine.TrySvd(a, fullMatrices, computeUv, out u, out s, out vh);

        /// <inheritdoc/>
        public bool TryLstsq(NDArray a, NDArray b, double rcond,
            out NDArray solution, out NDArray residuals, out NDArray rank, out NDArray singularValues)
            => OpenBlasEngine.TryLstsq(a, b, rcond, out solution, out residuals, out rank, out singularValues);

        #endregion

        /// <summary>The loaded library's own description, for diagnostics.</summary>
        public override string ToString() => "OpenBlasBackend " + Info;
    }
}
