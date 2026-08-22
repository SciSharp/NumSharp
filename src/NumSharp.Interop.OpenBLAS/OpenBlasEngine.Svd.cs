using System;
using System.Numerics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     <c>np.linalg.svd</c> / <c>np.linalg.svdvals</c> — a route-for-route port of NumPy 2.4.2's
    ///     <c>svd_N</c> / <c>svd_S</c> / <c>svd_A</c> gufuncs (<c>umath_linalg.cpp</c>, <c>svd_wrapper</c>)
    ///     calling the SAME LAPACK <c>gesdd</c> the bundled scipy-openblas ships. Also the engine behind
    ///     <c>pinv</c>, <c>matrix_rank</c>, <c>cond</c> and the spectral/nuclear matrix norms, which are
    ///     pure array reconstructions on top of this.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Computed in double precision like all of NumPy's linalg: a float32 operand is upcast to
    ///     float64 (exactly), decomposed with <c>dgesdd</c>, and the outputs cast back — the singular
    ///     values <c>S</c> are ALWAYS the real basetype (float64 for a complex operand, its own width
    ///     for a real one). Only the double and cdouble routines are ever called.
    ///     </para>
    ///     <para>
    ///     Each 2-D matrix is copied into a fresh COLUMN-MAJOR buffer read through the operand's own
    ///     strides (NumPy's <c>linearize_matrix</c> with the swapped steps), so every layout works and
    ///     the operand is never mutated. <c>gesdd</c> overwrites that buffer and fills <c>S</c> plus —
    ///     when <c>compute_uv</c> — the column-major factors <c>U</c> (M×K/M) and <c>VT</c> (K/N×N),
    ///     which are delinearized straight back to C-order. The <c>JOBZ</c> letter and the U/VT shapes
    ///     match NumPy's <c>compute_urows_vtcolumns</c> exactly:
    ///     <list type="bullet">
    ///     <item><c>compute_uv=false</c> → JOBZ 'N': only <c>S</c> <c>(…,K)</c>, K = min(M,N).</item>
    ///     <item><c>full_matrices=false</c> → JOBZ 'S': U <c>(…,M,K)</c>, VT <c>(…,K,N)</c>.</item>
    ///     <item><c>full_matrices=true</c> → JOBZ 'A': U <c>(…,M,M)</c>, VT <c>(…,N,N)</c>.</item>
    ///     </list>
    ///     The workspace sizes (<c>LWORK</c>, and the complex <c>RWORK</c> / integer <c>IWORK</c>) are
    ///     queried and sized exactly as NumPy's <c>init_gesdd</c> does. A zero-sized K with JOBZ 'A'
    ///     leaves U/VT uninitialised in LAPACK, so — matching NumPy's <c>identity_matrix</c> fallback —
    ///     an identity is written for each. Non-convergence raises
    ///     <see cref="LinAlgError"/>("SVD did not converge"), NumPy's exact message.
    ///     </para>
    /// </remarks>
    public static unsafe partial class OpenBlasEngine
    {
        /// <summary>Parity entry point for <c>np.linalg.svd</c> / <c>svdvals</c> (LAPACK <c>gesdd</c>).</summary>
        internal static bool TrySvd(NDArray a, bool fullMatrices, bool computeUv, out NDArray u, out NDArray s, out NDArray vh)
        {
            u = null;
            s = null;
            vh = null;
            if (!OpenBlasNative.IsLapackLoaded)
                return false;

            switch (a.GetTypeCode)
            {
                case NPTypeCode.Double:
                    (u, s, vh) = SvdCore<double, DoubleLapack>(a, fullMatrices, computeUv);
                    return true;
                case NPTypeCode.Complex:
                    (u, s, vh) = SvdCore<Complex, ComplexLapack>(a, fullMatrices, computeUv);
                    return true;
                case NPTypeCode.Single:
                    var (uu, ss, vv) = SvdCore<double, DoubleLapack>(a.astype(NPTypeCode.Double), fullMatrices, computeUv);
                    u = uu?.astype(NPTypeCode.Single);        // result_t = single
                    s = ss.astype(NPTypeCode.Single);         // real_t   = single
                    vh = vv?.astype(NPTypeCode.Single);
                    return true;
                default:
                    return false;
            }
        }

        private static (NDArray u, NDArray s, NDArray vh) SvdCore<T, TOps>(NDArray a, bool fullMatrices, bool computeUv)
            where T : unmanaged
            where TOps : struct, ILapackType<T>
        {
            var ops = default(TOps);
            var shape = a.Shape;
            int nd = shape.NDim;
            long m = shape.dimensions[nd - 2];
            long n = shape.dimensions[nd - 1];
            long mn = Math.Min(m, n);
            int nb = nd - 2;

            long count = 1;
            for (int i = 0; i < nb; i++)
                count *= shape.dimensions[i];

            byte jobz = !computeUv ? (byte)'N' : (fullMatrices ? (byte)'A' : (byte)'S');
            long uCols = !computeUv ? 0 : (fullMatrices ? m : mn); // U is (M, uCols)
            long vRows = !computeUv ? 0 : (fullMatrices ? n : mn); // VT is (vRows, N)

            // S is always the REAL basetype (double); the SvdCore only instantiates for T ∈ {double, Complex}.
            // fillZeros:false: S is fully written by the loop, U/VT by Delinearize (or the self-zeroing
            // FillIdentityStack for the mn==0 identity path).
            var s = new NDArray(NPTypeCode.Double, MakeShape1(shape.dimensions, nb, mn), fillZeros: false);
            NDArray u = null, vh = null;
            if (computeUv)
            {
                u = new NDArray(InfoOf<T>.NPTypeCode, MakeShape(shape.dimensions, nb, m, uCols), fillZeros: false);
                vh = new NDArray(InfoOf<T>.NPTypeCode, MakeShape(shape.dimensions, nb, vRows, n), fillZeros: false);
            }

            if (count == 0)
                return (u, s, vh); // empty batch — every output empty

            double* sptr = (double*)s.Address + s.Shape.offset;
            T* uptr = u is null ? null : (T*)u.Address + u.Shape.offset;
            T* vptr = vh is null ? null : (T*)vh.Address + vh.Shape.offset;

            // A zero-sized K (M == 0 or N == 0): LAPACK leaves U/VT uninitialised, so NumPy substitutes
            // an identity for whichever is non-empty (JOBZ 'A' only). S is empty; nothing to factorise.
            if (mn == 0)
            {
                if (fullMatrices && computeUv)
                {
                    if (u.size != 0)
                        FillIdentityStack(uptr, count, m, m, ops.One);
                    if (vh.size != 0)
                        FillIdentityStack(vptr, count, n, n, ops.One);
                }

                return (u, s, vh);
            }

            long lda = Math.Max(m, 1);
            long ldu = Math.Max(m, 1);
            long ldvt = Math.Max(1, vRows);
            bool isComplex = typeof(T) == typeof(Complex);

            T* bufA = null, ubuf = null, vtbuf = null, work = null;
            double* sbuf = null, rwork = null;
            void* iwork = null;
            try
            {
                bufA = Alloc<T>(m * n);
                sbuf = Alloc<double>(mn);
                ubuf = Alloc<T>(Math.Max(m * uCols, 1)); // 'N' leaves these unused; keep them non-null
                vtbuf = Alloc<T>(Math.Max(vRows * n, 1));
                iwork = AllocIpiv(8 * mn);               // NumPy's 8 * min(m,n) fortran_int
                if (isComplex)
                {
                    long rworkCount = jobz == (byte)'N' ? 7 * mn : 5 * mn * mn + 5 * mn; // NumPy's init_gesdd
                    rwork = Alloc<double>(rworkCount);
                }

                // Workspace query (LWORK == -1) — reads only the dimensions, so the uninitialised buffers are fine.
                T wq;
                ops.Gesdd(jobz, m, n, bufA, lda, sbuf, ubuf, ldu, vtbuf, ldvt, &wq, -1, rwork, iwork);
                long lwork = Math.Max(1, ops.OptimalLwork(&wq));
                work = Alloc<T>(lwork);

                var coord = new long[nb];
                long sBlock = mn;
                long uBlock = m * uCols;
                long vBlock = vRows * n;
                for (long e = 0; e < count; e++)
                {
                    long aoff = shape.offset;
                    for (int i = 0; i < nb; i++)
                        aoff += coord[i] * shape.strides[i];

                    Linearize((T*)a.Address + aoff, shape.strides[nd - 2], shape.strides[nd - 1], bufA, m, n);
                    long info = ops.Gesdd(jobz, m, n, bufA, lda, sbuf, ubuf, ldu, vtbuf, ldvt, work, lwork, rwork, iwork);
                    if (info != 0)
                        throw new LinAlgError("SVD did not converge");

                    double* sdst = sptr + e * sBlock;
                    for (long t = 0; t < mn; t++)
                        sdst[t] = sbuf[t];

                    if (computeUv)
                    {
                        Delinearize(ubuf, m, uCols, uptr + e * uBlock);   // LDU == M
                        Delinearize(vtbuf, vRows, n, vptr + e * vBlock);  // LDVT == vRows
                    }

                    AdvanceCoord(coord, shape.dimensions, nb);
                }
            }
            finally
            {
                Free(bufA);
                Free(sbuf);
                Free(ubuf);
                Free(vtbuf);
                Free(work);
                Free(rwork);
                FreeIpiv(iwork);
            }

            return (u, s, vh);
        }
    }
}
