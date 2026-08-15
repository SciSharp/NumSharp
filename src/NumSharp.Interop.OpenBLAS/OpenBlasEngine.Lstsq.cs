using System;
using System.Numerics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     <c>np.linalg.lstsq</c> — a route-for-route port of NumPy 2.4.2's <c>lstsq</c> gufunc
    ///     (<c>umath_linalg.cpp</c>) calling the SAME LAPACK <c>gelsd</c> the bundled scipy-openblas
    ///     ships: the minimum-norm least-squares solution of <c>A x = B</c> plus the residuals, the
    ///     effective rank and the singular values.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Computed in double precision like all of NumPy's linalg (float32 upcast then cast back), so
    ///     only <c>dgelsd</c> / <c>zgelsd</c> are called. The solution <c>x</c> keeps the operand's
    ///     (result) dtype; the residuals and singular values are the real basetype (double).
    ///     </para>
    ///     <para>
    ///     <c>a</c> is copied into a column-major buffer and <c>b</c> into one of leading dimension
    ///     <c>max(M,N)</c> (LAPACK overwrites its first N rows with the solution and, when the system is
    ///     over-determined, leaves the residual components in rows <c>[N, M)</c>). This mirrors NumPy's
    ///     <c>init_gelsd</c> + <c>lstsq</c> exactly, including the residual rule: the squared 2-norm of
    ///     each excess column is produced only when <c>M ≥ N</c> AND the rank is full (<c>rank == N</c>);
    ///     otherwise the residuals are NaN and the <c>np.linalg.lstsq</c> wrapper discards them. The
    ///     workspace, integer-workspace and (complex) real-workspace sizes are all queried the way
    ///     <c>init_gelsd</c> queries them. Non-convergence raises
    ///     <see cref="LinAlgError"/>("SVD did not converge in Linear Least Squares"), NumPy's message.
    ///     </para>
    ///     <para>
    ///     Unlike the other factorisations this does NOT stack: <c>np.linalg.lstsq</c> is 2-D only
    ///     (<c>_assert_2d</c>), so the operands arrive as a single <c>(M,N)</c> and <c>(M,nrhs)</c> pair
    ///     — the 1-D-<c>b</c> promotion, the <c>nrhs == 0</c> padding, the <c>M == 0</c> zeroing and the
    ///     residual/dtype coercion live in the wrapper, exactly where NumPy's Python layer puts them.
    ///     </para>
    /// </remarks>
    public static unsafe partial class OpenBlasEngine
    {
        /// <summary>Parity entry point for <c>np.linalg.lstsq</c> (LAPACK <c>gelsd</c>).</summary>
        internal static bool TryLstsq(NDArray a, NDArray b, double rcond,
            out NDArray solution, out NDArray residuals, out NDArray rank, out NDArray singularValues)
        {
            solution = null;
            residuals = null;
            rank = null;
            singularValues = null;
            if (!OpenBlasNative.IsLapackLoaded)
                return false;

            switch (a.GetTypeCode)
            {
                case NPTypeCode.Double:
                    (solution, residuals, rank, singularValues) = LstsqCore<double, DoubleLapack>(a, b, rcond);
                    return true;
                case NPTypeCode.Complex:
                    (solution, residuals, rank, singularValues) = LstsqCore<Complex, ComplexLapack>(a, b, rcond);
                    return true;
                case NPTypeCode.Single:
                    var (x, res, rk, s) = LstsqCore<double, DoubleLapack>(
                        a.astype(NPTypeCode.Double), b.astype(NPTypeCode.Double), rcond);
                    solution = x.astype(NPTypeCode.Single);      // result_t = single
                    residuals = res.astype(NPTypeCode.Single);   // real_t   = single
                    rank = rk;                                   // rank stays int32
                    singularValues = s.astype(NPTypeCode.Single);
                    return true;
                default:
                    return false;
            }
        }

        private static (NDArray x, NDArray resids, NDArray rank, NDArray s) LstsqCore<T, TOps>(NDArray a, NDArray b, double rcond)
            where T : unmanaged
            where TOps : struct, ILapackType<T>
        {
            var ops = default(TOps);
            var sa = a.Shape;
            var sb = b.Shape;
            long m = sa.dimensions[0];
            long n = sa.dimensions[1];
            long nrhs = sb.dimensions[1];
            long mn = Math.Min(m, n);
            long maxmn = Math.Max(m, n);
            bool isComplex = typeof(T) == typeof(Complex);

            var x = new NDArray(InfoOf<T>.NPTypeCode, new Shape(new long[] { n, nrhs }));   // zero-filled
            var resids = new NDArray(NPTypeCode.Double, new Shape(new long[] { nrhs }));
            var sarr = new NDArray(NPTypeCode.Double, new Shape(new long[] { mn }));

            double* residPtr = nrhs == 0 ? null : (double*)resids.Address + resids.Shape.offset;
            double* sPtr = mn == 0 ? null : (double*)sarr.Address + sarr.Shape.offset;

            long bSr = sb.strides[0], bSc = sb.strides[1];
            T* pb = (T*)b.Address + sb.offset;

            // Degenerate M == 0 or N == 0 (mn == 0): the min-norm solution is the zero x already
            // allocated, no singular values, rank 0. Residuals still follow NumPy's excess-rows formula
            // (empty ⇒ 0 for M == 0), and the wrapper discards them where NumPy does.
            if (mn == 0)
            {
                for (long i = 0; i < nrhs; i++)
                {
                    double acc = 0.0;
                    for (long r = n; r < m; r++)
                        acc += ops.Abs2(pb[r * bSr + i * bSc]);
                    residPtr[i] = acc;
                }

                return (x, resids, NDArray.Scalar(0), sarr);
            }

            long lda = Math.Max(1, m);
            long ldb = Math.Max(1, maxmn);

            T* abuf = Alloc<T>(m * n);
            T* bbuf = Alloc<T>(maxmn * nrhs);
            double* sbuf = Alloc<double>(mn);
            double* rwork = null;
            T* work = null;
            void* iwork = null;
            long rank;
            try
            {
                Linearize((T*)a.Address + sa.offset, sa.strides[0], sa.strides[1], abuf, m, n);

                // B into a column-major buffer of leading dimension max(M,N) — only the M rows are the RHS.
                for (long c = 0; c < nrhs; c++)
                {
                    T* col = bbuf + c * ldb;
                    T* src = pb + c * bSc;
                    for (long r = 0; r < m; r++)
                        col[r] = src[r * bSr];
                }

                // Workspace query — writes work[0], iwork[0], and (complex) rwork[0].
                void* iworkQuery = AllocIpiv(1);
                double rworkQuery = 0.0;
                double* rworkQueryPtr = isComplex ? &rworkQuery : null;
                T wq;
                ops.Gelsd(m, n, nrhs, abuf, lda, bbuf, ldb, sbuf, rcond, out _, &wq, -1, rworkQueryPtr, iworkQuery);
                long lwork = Math.Max(1, ops.OptimalLwork(&wq));
                long liwork = Math.Max(1, OpenBlasNative.ReadPivot(iworkQuery, 0));
                FreeIpiv(iworkQuery);
                long lrwork = isComplex ? Math.Max(1, (long)rworkQuery) : 0;

                work = Alloc<T>(lwork);
                iwork = AllocIpiv(liwork);
                rwork = isComplex ? Alloc<double>(lrwork) : null;

                long info = ops.Gelsd(m, n, nrhs, abuf, lda, bbuf, ldb, sbuf, rcond, out rank, work, lwork, rwork, iwork);
                if (info != 0)
                    throw new LinAlgError("SVD did not converge in Linear Least Squares");

                // Solution = the first N rows of the column-major B buffer → (N, nrhs) row-major.
                T* xptr = (T*)x.Address + x.Shape.offset;
                for (long r = 0; r < n; r++)
                {
                    T* row = xptr + r * nrhs;
                    for (long c = 0; c < nrhs; c++)
                        row[c] = bbuf[r + c * ldb];
                }

                for (long t = 0; t < mn; t++)
                    sPtr[t] = sbuf[t];

                // Residuals: the squared 2-norm of each excess column, only when over-determined AND
                // full-rank (NumPy's `excess >= 0 && rank == n`); NaN otherwise (the wrapper discards it).
                if (m >= n && rank == n)
                {
                    for (long i = 0; i < nrhs; i++)
                    {
                        double acc = 0.0;
                        for (long r = n; r < m; r++)
                            acc += ops.Abs2(bbuf[r + i * ldb]);
                        residPtr[i] = acc;
                    }
                }
                else
                {
                    for (long i = 0; i < nrhs; i++)
                        residPtr[i] = double.NaN;
                }
            }
            finally
            {
                Free(abuf);
                Free(bbuf);
                Free(sbuf);
                Free(work);
                Free(rwork);
                FreeIpiv(iwork);
            }

            return (x, resids, NDArray.Scalar((int)rank), sarr);
        }
    }
}
