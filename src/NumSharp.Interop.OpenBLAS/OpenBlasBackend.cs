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
    ///     It answers for <c>float32</c>, <c>float64</c> and <c>complex128</c> (the dtypes NumPy itself
    ///     routes through cblas), and returns false for everything else so the engine keeps computing
    ///     it with its own managed kernels — integer and bool products do not need a BLAS anyway
    ///     (modular integer addition is associative, so summation order cannot change the result;
    ///     complex float accumulation is NOT associative and genuinely needs <c>zgemm</c>/<c>zdotu</c>
    ///     for byte-parity). complex128 is served only when the loaded library exports the complex
    ///     products; a bare real-only CBLAS declines it back to the managed complex kernel.
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
    public sealed class OpenBlasBackend : IBlasBackend, ISlidingDotBackend
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

        #region Products (CBLAS inner / vdot / vecdot / matvec / vecmat)

        // The rest of what NumPy sends to cblas beyond dot/matmul. Each is a route-for-route port of
        // NumPy 2.4.2's own dispatcher (cblasfuncs.c / matmul.c.src / vdot.c), so where the managed
        // composition reassociates the sum differently — vecdot's Multiply+ReduceAdd, complex vdot's
        // conj-then-dotu, complex vecmat's conj-then-gemv — this computes the byte-identical answer.
        // They answer for float32/float64/complex128 (complex128 only when the complex products are
        // loaded) and decline every other dtype, keeping the managed composition as the fallback.

        /// <inheritdoc/>
        public bool TryInner(NDArray a, NDArray b, out NDArray result)
            => OpenBlasEngine.TryInner(a, b, out result);

        /// <inheritdoc/>
        public bool TryVdot(NDArray a, NDArray b, out NDArray result)
            => OpenBlasEngine.TryVdot(a, b, out result);

        /// <inheritdoc/>
        public bool TryVecdot(NDArray x1, NDArray x2, out NDArray result)
            => OpenBlasEngine.TryVecdot(x1, x2, out result);

        /// <inheritdoc/>
        public bool TryMatvec(NDArray x1, NDArray x2, out NDArray result)
            => OpenBlasEngine.TryMatvec(x1, x2, out result);

        /// <inheritdoc/>
        public bool TryVecmat(NDArray x1, NDArray x2, out NDArray result)
            => OpenBlasEngine.TryVecmat(x1, x2, out result);

        #endregion

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

        #region Eigen factorisations (LAPACK syevd / heevd / geev)

        // Same story as the other LAPACK families: they answer only when the loaded library exports
        // LAPACK, compute float32 in double and cast back exactly as NumPy does, and decline every other
        // dtype so the engine raises its "needs a LAPACK backend". A non-convergent element is a genuine
        // LinAlgError from the routine (the operand WAS served), not a decline — it propagates. TryEig
        // always hands back complex128; the np.linalg.eig wrapper collapses it to a real result when the
        // imaginary parts all vanish, exactly as NumPy's Python layer does.

        /// <inheritdoc/>
        public bool TryEigh(NDArray a, char uplo, bool computeVectors, out NDArray eigenvalues, out NDArray eigenvectors)
            => OpenBlasEngine.TryEigh(a, uplo, computeVectors, out eigenvalues, out eigenvectors);

        /// <inheritdoc/>
        public bool TryEig(NDArray a, bool computeVectors, out NDArray eigenvalues, out NDArray eigenvectors)
            => OpenBlasEngine.TryEig(a, computeVectors, out eigenvalues, out eigenvectors);

        #endregion

        #region Sliding dot (np.correlate / np.convolve) — the byte-parity dotfunc

        // np.correlate / np.convolve reduce every ramp position, and the middle whenever NumPy's
        // small_correlate declines (a real kernel longer than 11, or any complex kernel), with NumPy's
        // per-dtype dotfunc — the same double-accumulated chunked cblas ?dot (?dotu for complex) the
        // product family uses. Exposing it through the ISlidingDotBackend seam is what lets the sliding
        // kernels match NumPy byte-for-byte on the long float32/float64 and complex128 kernels the
        // managed reduction reorders; short real kernels stay on the managed path (already byte-exact),
        // and this declines every non-cblas dtype so those stay managed too.

        /// <inheritdoc/>
        bool ISlidingDotBackend.SupportsDot(NPTypeCode dtype)
            => OpenBlasEngine.SupportsSlidingDot(dtype);

        /// <inheritdoc/>
        unsafe void ISlidingDotBackend.Dot(NPTypeCode dtype,
            void* a, long strideA, void* b, long strideB, void* result, long count)
            => OpenBlasEngine.SlidingDot(dtype, a, strideA, b, strideB, result, count);

        /// <inheritdoc/>
        // Override the default per-position fallback: switch on the dtype ONCE and run the whole
        // fully-overlapping middle through the tight native ?dot loop, so np.correlate/np.convolve
        // pay the interface dispatch once for the region instead of once per output. Bit-identical
        // to the per-position path (same DoubleBlas.Dot / NumPy @name@_dot); only the dispatch moves.
        unsafe void ISlidingDotBackend.DotBatch(NPTypeCode dtype,
            void* a, void* b, void* result, long count, long n2)
            => OpenBlasEngine.SlidingDotBatch(dtype, a, b, result, count, n2);

        #endregion

        /// <summary>The loaded library's own description, for diagnostics.</summary>
        public override string ToString() => "OpenBlasBackend " + Info;
    }
}
