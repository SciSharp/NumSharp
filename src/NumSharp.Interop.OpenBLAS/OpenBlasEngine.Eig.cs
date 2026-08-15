using System;
using System.Numerics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     <c>np.linalg.eigh</c> / <c>eigvalsh</c> (LAPACK <c>syevd</c>/<c>heevd</c>) and
    ///     <c>np.linalg.eig</c> / <c>eigvals</c> (LAPACK <c>geev</c>) — a route-for-route port of NumPy
    ///     2.4.2's <c>eigh_wrapper</c> and <c>eig_wrapper</c> gufuncs (<c>umath_linalg.cpp</c>) calling
    ///     the SAME LAPACK the bundled scipy-openblas ships.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Computed in double precision like all of NumPy's linalg: a float32 operand is upcast to
    ///     float64 (exactly), decomposed with the double routine, and the outputs cast back. Only the
    ///     double and cdouble routines are ever called.
    ///     </para>
    ///     <para>
    ///     <b>eigh / eigvalsh</b> (<c>syevd</c>/<c>heevd</c>) return REAL eigenvalues (the real basetype,
    ///     float64) in ASCENDING order for a symmetric/Hermitian operand — only the <paramref name="uplo"/>
    ///     triangle is read. This mirrors the SVD/QR pattern: the engine returns the FINAL dtype (W the
    ///     real basetype, V the compute dtype), so the <c>np.linalg</c> wrapper is thin. The eigenvectors
    ///     are the columns of the overwritten column-major matrix, delinearized straight to C-order.
    ///     </para>
    ///     <para>
    ///     <b>eig / eigvals</b> (<c>geev</c>) ALWAYS produce complex128 outputs at the engine seam — a real
    ///     matrix can carry a complex-conjugate eigenpair, so the dtype is data-dependent and cannot be
    ///     resolved from the input dtype alone. NumPy's gufunc is likewise always-complex
    ///     (<c>'d->DD'</c>/<c>'D->DD'</c>); the collapse to a real result when every imaginary part is zero
    ///     is NumPy's PYTHON-layer step (<c>if not isComplexType(t) and all(w.imag == 0)</c>), reproduced
    ///     in the <c>np.linalg.eig</c> wrapper — which is also where <c>_assert_finite</c> lives (eig /
    ///     eigvals reject inf/NaN; eigh / eigvalsh do not). For a real operand the real <c>dgeev</c> writes
    ///     split real/imaginary parts that <see cref="AssembleGeevEigenvectors"/> reassembles into complex
    ///     conjugate-pair eigenvectors exactly as NumPy's <c>mk_geev_complex_eigenvectors</c> does.
    ///     </para>
    ///     <para>
    ///     Each 2-D matrix is copied into a fresh COLUMN-MAJOR buffer read through the operand's own
    ///     strides (NumPy's <c>linearize_matrix</c> with the swapped steps), so every layout works and the
    ///     operand is never mutated. Stacks loop element by element with the scratch and workspace hoisted
    ///     once, the way the gufunc's outer loop does. A non-convergent element raises
    ///     <see cref="LinAlgError"/>("Eigenvalues did not converge"), NumPy's exact message.
    ///     </para>
    /// </remarks>
    public static unsafe partial class OpenBlasEngine
    {
        /// <summary>Parity entry point for <c>np.linalg.eigh</c> / <c>eigvalsh</c> (LAPACK <c>syevd</c>/<c>heevd</c>).</summary>
        internal static bool TryEigh(NDArray a, char uplo, bool computeVectors,
            out NDArray eigenvalues, out NDArray eigenvectors)
        {
            eigenvalues = null;
            eigenvectors = null;
            if (!OpenBlasNative.IsLapackLoaded)
                return false;

            byte up = (byte)char.ToUpperInvariant(uplo); // 'L' or 'U' — the wrapper validated it

            switch (a.GetTypeCode)
            {
                case NPTypeCode.Double:
                    (eigenvalues, eigenvectors) = EighCore<double, DoubleLapack>(a, up, computeVectors);
                    return true;
                case NPTypeCode.Complex:
                    (eigenvalues, eigenvectors) = EighCore<Complex, ComplexLapack>(a, up, computeVectors);
                    return true;
                case NPTypeCode.Single:
                    // Compute in double then cast back — NumPy's 'd->dd' / 'D->dD' lite path. W is the real
                    // basetype (-> single), V the result type (-> single). eigh mirrors svd: FINAL dtype here.
                    var (w, v) = EighCore<double, DoubleLapack>(a.astype(NPTypeCode.Double), up, computeVectors);
                    eigenvalues = w.astype(NPTypeCode.Single);
                    eigenvectors = v?.astype(NPTypeCode.Single);
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>Parity entry point for <c>np.linalg.eig</c> / <c>eigvals</c> (LAPACK <c>geev</c>).</summary>
        /// <remarks>
        ///     Always returns complex128 <paramref name="eigenvalues"/> / <paramref name="eigenvectors"/>;
        ///     the real-output collapse and the width cast to the operand's result dtype are the
        ///     <c>np.linalg.eig</c> wrapper's job (NumPy's Python layer), because both are data-dependent.
        /// </remarks>
        internal static bool TryEig(NDArray a, bool computeVectors,
            out NDArray eigenvalues, out NDArray eigenvectors)
        {
            eigenvalues = null;
            eigenvectors = null;
            if (!OpenBlasNative.IsLapackLoaded)
                return false;

            switch (a.GetTypeCode)
            {
                case NPTypeCode.Double:
                    (eigenvalues, eigenvectors) = EigCore<double, DoubleLapack>(a, computeVectors);
                    return true;
                case NPTypeCode.Complex:
                    (eigenvalues, eigenvectors) = EigCore<Complex, ComplexLapack>(a, computeVectors);
                    return true;
                case NPTypeCode.Single:
                    // 'd->DD': compute in double, still complex128 out. The wrapper collapses to single /
                    // complex per the imaginary parts.
                    (eigenvalues, eigenvectors) = EigCore<double, DoubleLapack>(a.astype(NPTypeCode.Double), computeVectors);
                    return true;
                default:
                    return false;
            }
        }

        // ----------------------------------------------------------------------------------------
        //  Generic cores (T ∈ {double, Complex}).
        // ----------------------------------------------------------------------------------------

        private static (NDArray w, NDArray v) EighCore<T, TOps>(NDArray a, byte uplo, bool computeVectors)
            where T : unmanaged
            where TOps : struct, ILapackType<T>
        {
            var ops = default(TOps);
            var shape = a.Shape;
            int nd = shape.NDim;
            long m = shape.dimensions[nd - 1]; // square: rows == cols == m (validated by AssertStackedSquare)
            int nb = nd - 2;

            long count = 1;
            for (int i = 0; i < nb; i++)
                count *= shape.dimensions[i];

            // W is ALWAYS the real basetype (double); the core only instantiates for T ∈ {double, Complex}.
            var w = new NDArray(NPTypeCode.Double, MakeShape1(shape.dimensions, nb, m));
            NDArray v = computeVectors ? new NDArray(InfoOf<T>.NPTypeCode, MakeShape(shape.dimensions, nb, m, m)) : null;

            if (count == 0 || m == 0)
                return (w, v); // empty batch or a 0×0 element — empty outputs, no LAPACK call

            byte jobz = computeVectors ? (byte)'V' : (byte)'N';
            long lda = Math.Max(m, 1);
            bool isComplex = typeof(T) == typeof(Complex);

            double* wptr = (double*)w.Address + w.Shape.offset;
            T* vptr = v is null ? null : (T*)v.Address + v.Shape.offset;

            T* bufA = null, work = null, vrrUnused = null;
            double* wbuf = null, rwork = null;
            void* iwork = null, iworkQuery = null;
            try
            {
                bufA = Alloc<T>(m * m);
                wbuf = Alloc<double>(m);
                iworkQuery = AllocIpiv(1);

                // Workspace query (lwork/lrwork/liwork == -1). Reads only dimensions — the uninitialised
                // bufA is fine. dsyevd ignores the rwork query; zheevd writes it.
                T wq;
                double rwq = 0.0;
                ops.Syevd(jobz, uplo, m, bufA, lda, wbuf, &wq, -1, &rwq, -1, iworkQuery, -1);
                long lwork = Math.Max(1, ops.OptimalLwork(&wq));
                long liwork = Math.Max(1, OpenBlasNative.ReadPivot(iworkQuery, 0)); // query wrote LIWORK as a fortran_int
                long lrwork = isComplex ? Math.Max(1, (long)rwq) : 0;

                work = Alloc<T>(lwork);
                iwork = AllocIpiv(liwork);
                if (isComplex)
                    rwork = Alloc<double>(lrwork);

                var coord = new long[nb];
                long wBlock = m;
                long vBlock = m * m;
                for (long e = 0; e < count; e++)
                {
                    long aoff = shape.offset;
                    for (int i = 0; i < nb; i++)
                        aoff += coord[i] * shape.strides[i];

                    Linearize((T*)a.Address + aoff, shape.strides[nd - 2], shape.strides[nd - 1], bufA, m, m);
                    long info = ops.Syevd(jobz, uplo, m, bufA, lda, wbuf, work, lwork, rwork, lrwork, iwork, liwork);
                    if (info != 0)
                        throw new LinAlgError("Eigenvalues did not converge");

                    double* wdst = wptr + e * wBlock;
                    for (long t = 0; t < m; t++)
                        wdst[t] = wbuf[t];

                    // syevd/heevd overwrote A (column-major) with the orthonormal eigenvectors as its columns.
                    if (computeVectors)
                        Delinearize(bufA, m, m, vptr + e * vBlock);

                    AdvanceCoord(coord, shape.dimensions, nb);
                }
            }
            finally
            {
                Free(bufA);
                Free(work);
                Free(wbuf);
                Free(rwork);
                Free(vrrUnused);
                FreeIpiv(iwork);
                FreeIpiv(iworkQuery);
            }

            return (w, v);
        }

        private static (NDArray w, NDArray v) EigCore<T, TOps>(NDArray a, bool computeVectors)
            where T : unmanaged
            where TOps : struct, ILapackType<T>
        {
            var ops = default(TOps);
            var shape = a.Shape;
            int nd = shape.NDim;
            long n = shape.dimensions[nd - 1]; // square
            int nb = nd - 2;

            long count = 1;
            for (int i = 0; i < nb; i++)
                count *= shape.dimensions[i];

            // eig ALWAYS produces complex128 at the seam (a real matrix can carry a complex-conjugate
            // eigenpair). The wrapper collapses to real when every imaginary part is zero.
            var w = new NDArray(NPTypeCode.Complex, MakeShape1(shape.dimensions, nb, n));
            NDArray v = computeVectors ? new NDArray(NPTypeCode.Complex, MakeShape(shape.dimensions, nb, n, n)) : null;

            if (count == 0 || n == 0)
                return (w, v);

            byte jobvr = computeVectors ? (byte)'V' : (byte)'N';
            long lda = Math.Max(n, 1);
            bool isComplex = typeof(T) == typeof(Complex);

            Complex* wptr = (Complex*)w.Address + w.Shape.offset;
            Complex* vptr = v is null ? null : (Complex*)v.Address + v.Shape.offset;

            T* bufA = null, work = null, vrr = null;
            double* wr = null, wi = null, rwork = null;
            Complex* wC = null, vrC = null;
            try
            {
                bufA = Alloc<T>(n * n);
                wC = Alloc<Complex>(n);                                // complex eigenvalue assembly buffer
                vrC = computeVectors ? Alloc<Complex>(n * n) : null;   // complex eigenvector assembly buffer (column-major)
                if (isComplex)
                {
                    rwork = Alloc<double>(2 * n);                      // zgeev RWORK
                }
                else
                {
                    wr = Alloc<double>(n);                             // dgeev real parts
                    wi = Alloc<double>(n);
                    vrr = computeVectors ? Alloc<T>(n * n) : null;     // dgeev real eigenvectors
                }

                // Workspace query (lwork == -1) — writes work[0], skips the complex assembly.
                T wq;
                ops.Geev(jobvr, n, bufA, lda, wC, vrC, lda, &wq, -1, wr, wi, vrr, rwork);
                long lwork = Math.Max(1, ops.OptimalLwork(&wq)); // NumPy's "if work_count == 0, work_count = 1"
                work = Alloc<T>(lwork);

                var coord = new long[nb];
                long wBlock = n;
                long vBlock = n * n;
                for (long e = 0; e < count; e++)
                {
                    long aoff = shape.offset;
                    for (int i = 0; i < nb; i++)
                        aoff += coord[i] * shape.strides[i];

                    Linearize((T*)a.Address + aoff, shape.strides[nd - 2], shape.strides[nd - 1], bufA, n, n);
                    long info = ops.Geev(jobvr, n, bufA, lda, wC, vrC, lda, work, lwork, wr, wi, vrr, rwork);
                    if (info != 0)
                        throw new LinAlgError("Eigenvalues did not converge");

                    Complex* wdst = wptr + e * wBlock;
                    for (long t = 0; t < n; t++)
                        wdst[t] = wC[t];

                    if (computeVectors)
                        Delinearize(vrC, n, n, vptr + e * vBlock); // column-major complex eigenvectors → C-order

                    AdvanceCoord(coord, shape.dimensions, nb);
                }
            }
            finally
            {
                Free(bufA);
                Free(work);
                Free(vrr);
                Free(wr);
                Free(wi);
                Free(rwork);
                Free(wC);
                Free(vrC);
            }

            return (w, v);
        }

        /// <summary>
        ///     NumPy's <c>mk_geev_complex_eigenvectors</c>: builds the always-complex right eigenvectors
        ///     (column-major, LD == n) from <c>dgeev</c>'s real <paramref name="vrr"/> buffer and the
        ///     eigenvalue imaginary parts <paramref name="wi"/>. A real eigenvalue (<c>wi == 0</c>) copies
        ///     one real column as <c>(re, 0)</c>; a complex-conjugate pair (<c>wi != 0</c>) turns columns
        ///     <c>k</c> and <c>k+1</c> into <c>VRR[:,k] ± i·VRR[:,k+1]</c>. Columns are contiguous because
        ///     LAPACK's <c>LD = max(n, 1) = n</c> for <c>n ≥ 1</c>.
        /// </summary>
        private static void AssembleGeevEigenvectors(Complex* c, double* r, double* wi, long n)
        {
            long iter = 0;
            while (iter < n)
            {
                if (wi[iter] == 0.0)
                {
                    // eigenvalue real -> eigenvector real (mk_complex_array_from_real)
                    for (long k = 0; k < n; k++)
                        c[k] = new Complex(r[k], 0.0);
                    c += n;
                    r += n;
                    iter++;
                }
                else
                {
                    // eigenvalue complex -> a conjugate pair (mk_complex_array_conjugate_pair)
                    for (long k = 0; k < n; k++)
                    {
                        double re = r[k], im = r[k + n];
                        c[k] = new Complex(re, im);
                        c[k + n] = new Complex(re, -im);
                    }

                    c += 2 * n;
                    r += 2 * n;
                    iter += 2;
                }
            }
        }
    }
}
