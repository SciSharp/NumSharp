using System;
using System.Numerics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     <c>np.linalg.qr</c> — a route-for-route port of NumPy 2.4.2's <c>qr_r_raw</c> (LAPACK
    ///     <c>geqrf</c>) plus <c>qr_reduced</c>/<c>qr_complete</c> (LAPACK <c>orgqr</c>/<c>ungqr</c>)
    ///     gufuncs, calling the SAME LAPACK the bundled scipy-openblas ships.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Computed in double precision like all of NumPy's linalg: a float32 operand is upcast to
    ///     float64 (exactly), factorised, and the outputs cast back — bit-identical to NumPy. Only the
    ///     double and cdouble routines are ever called.
    ///     </para>
    ///     <para>
    ///     Each 2-D matrix is copied into a fresh COLUMN-MAJOR buffer read through the operand's own
    ///     strides (NumPy's <c>linearize_matrix</c> with the swapped steps), so every layout works and
    ///     the operand is never mutated. <c>geqrf</c> overwrites the buffer with the packed R (upper
    ///     triangle) plus the Householder reflectors (below the diagonal) and produces <c>tau</c>; R is
    ///     read straight out of that packed buffer with the upper triangle kept (NumPy's <c>triu</c>),
    ///     and — for the modes that return Q — <c>orgqr</c>/<c>ungqr</c> expand the reflectors into Q in
    ///     a second buffer. The four modes match NumPy exactly:
    ///     <list type="bullet">
    ///     <item><c>reduced</c> (default): Q <c>(…,M,K)</c>, R <c>(…,K,N)</c>, K = min(M,N).</item>
    ///     <item><c>complete</c>: Q <c>(…,M,M)</c>, R <c>(…,M,N)</c> when M &gt; N, else same as reduced.</item>
    ///     <item><c>r</c>: R only, <c>(…,K,N)</c> (returned in the R slot; Q is null).</item>
    ///     <item><c>raw</c>: the LAPACK pair <c>(h, tau)</c> — h <c>(…,N,M)</c> is the packed reflectors
    ///     transposed, tau <c>(…,K)</c> (returned as (Q=h, R=tau)).</item>
    ///     </list>
    ///     The deprecated <c>economic</c> mode returns the packed <c>geqrf</c> result as Q (R null).
    ///     </para>
    ///     <para>
    ///     The workspace sizes (<c>LWORK</c>) are queried and applied exactly as NumPy's <c>init_geqrf</c>
    ///     / <c>init_gqr_common</c> do — including <c>zungqr</c>'s use of the queried optimal directly
    ///     where <c>dorgqr</c> uses <c>max(max(1,N), optimal)</c> — so the blocked algorithm runs with
    ///     the same block size and the result is bit-identical.
    ///     </para>
    /// </remarks>
    public static unsafe partial class OpenBlasEngine
    {
        /// <summary>Parity entry point for <c>np.linalg.qr</c> (LAPACK <c>geqrf</c> + <c>orgqr</c>/<c>ungqr</c>).</summary>
        internal static bool TryQr(NDArray a, string mode, out NDArray q, out NDArray r)
        {
            q = null;
            r = null;
            if (!OpenBlasNative.IsLapackLoaded)
                return false;

            switch (a.GetTypeCode)
            {
                case NPTypeCode.Double:
                    (q, r) = QrCore<double, DoubleLapack>(a, mode);
                    return true;
                case NPTypeCode.Complex:
                    (q, r) = QrCore<Complex, ComplexLapack>(a, mode);
                    return true;
                case NPTypeCode.Single:
                    var (qq, rr) = QrCore<double, DoubleLapack>(a.astype(NPTypeCode.Double), mode);
                    q = qq?.astype(NPTypeCode.Single);
                    r = rr?.astype(NPTypeCode.Single);
                    return true;
                default:
                    return false;
            }
        }

        private static (NDArray q, NDArray r) QrCore<T, TOps>(NDArray a, string mode)
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

            bool wantQ = mode == "reduced" || mode == "complete";
            long qCols = (mode == "complete" && m > n) ? m : mn; // NumPy's mc

            // Output shapes (batch dims + the mode's trailing shape).
            NDArray q = null, r = null;
            switch (mode)
            {
                // fillZeros:false: Q is fully written by Delinearize / the transpose loop (or the
                // self-zeroing FillIdentityStack for the mn==0 identity path) and R by DelinearizeTriu.
                case "reduced":
                case "complete":
                    q = new NDArray(InfoOf<T>.NPTypeCode, MakeShape(shape.dimensions, nb, m, qCols), fillZeros: false);
                    r = new NDArray(InfoOf<T>.NPTypeCode, MakeShape(shape.dimensions, nb, qCols, n), fillZeros: false);
                    break;
                case "r":
                    r = new NDArray(InfoOf<T>.NPTypeCode, MakeShape(shape.dimensions, nb, mn, n), fillZeros: false);
                    break;
                case "raw":
                    q = new NDArray(InfoOf<T>.NPTypeCode, MakeShape(shape.dimensions, nb, n, m), fillZeros: false); // h
                    r = new NDArray(InfoOf<T>.NPTypeCode, MakeShape1(shape.dimensions, nb, mn), fillZeros: false);   // tau
                    break;
                case "economic":
                    q = new NDArray(InfoOf<T>.NPTypeCode, MakeShape(shape.dimensions, nb, m, n), fillZeros: false);
                    break;
                default:
                    // np.linalg.qr resolves the mode before the seam; anything else is a caller error.
                    throw new ArgumentException($"Unrecognized mode '{mode}'");
            }

            long qSize = q?.size ?? 0;
            long rSize = r?.size ?? 0;
            if (count == 0 || (qSize == 0 && rSize == 0 && mn == 0))
                return (q, r); // empty batch or every output empty

            T* qptr = q is null ? null : (T*)q.Address + q.Shape.offset;
            T* rptr = r is null ? null : (T*)r.Address + r.Shape.offset;

            // No reflectors (M == 0 or N == 0): geqrf/orgqr are no-ops. The only non-empty output is a
            // complete-mode Q, which orgqr(k=0) would return as the leading identity — written directly.
            if (mn == 0)
            {
                if (wantQ && qSize != 0)
                    FillIdentityStack(qptr, count, m, qCols, ops.One);
                return (q, r);
            }

            long lda = Math.Max(m, 1);
            bool isComplex = typeof(T) == typeof(Complex);

            T* abuf = Alloc<T>(m * n);
            T* tau = Alloc<T>(mn);

            // geqrf workspace query — reads only the dimensions, so the uninitialised abuf is fine.
            T wq;
            ops.Geqrf(m, n, abuf, lda, tau, &wq, -1);
            long lworkGeqrf = Math.Max(Math.Max(1, n), ops.OptimalLwork(&wq));
            T* work = Alloc<T>(lworkGeqrf);

            T* qbuf = null, work2 = null;
            long lworkOrgqrPass = 0;
            if (wantQ)
            {
                qbuf = Alloc<T>(m * qCols);
                T wq2;
                ops.Orgqr(m, qCols, mn, qbuf, lda, tau, &wq2, -1);
                long optimal = ops.OptimalLwork(&wq2);
                long lworkAlloc = Math.Max(Math.Max(1, n), optimal);
                // zungqr passes the queried optimal directly; dorgqr passes max(max(1,n), optimal).
                lworkOrgqrPass = isComplex ? optimal : lworkAlloc;
                work2 = Alloc<T>(lworkAlloc);
            }

            try
            {
                var coord = new long[nb];
                for (long e = 0; e < count; e++)
                {
                    long aoff = shape.offset;
                    for (int i = 0; i < nb; i++)
                        aoff += coord[i] * shape.strides[i];

                    Linearize((T*)a.Address + aoff, shape.strides[nd - 2], shape.strides[nd - 1], abuf, m, n);
                    long info = ops.Geqrf(m, n, abuf, lda, tau, work, lworkGeqrf);
                    if (info != 0)
                        throw new LinAlgError("Incorrect argument found while performing QR factorization");

                    switch (mode)
                    {
                        case "r":
                            DelinearizeTriu(abuf, mn, n, lda, rptr + e * (mn * n));
                            break;

                        case "raw":
                        {
                            // h (n, m) = transpose of the packed C-order result: h[i,j] = packed[j,i].
                            T* hdst = qptr + e * (n * m);
                            for (long i = 0; i < n; i++)
                                for (long j = 0; j < m; j++)
                                    hdst[i * m + j] = abuf[j + i * lda];

                            T* tdst = rptr + e * mn;
                            for (long t = 0; t < mn; t++)
                                tdst[t] = tau[t];
                            break;
                        }

                        case "economic":
                            Delinearize(abuf, m, n, qptr + e * (m * n));
                            break;

                        default: // reduced / complete
                        {
                            // Seed Q with the reflector columns (0..mn-1), then expand to qCols columns.
                            for (long c = 0; c < mn; c++)
                                for (long rr = 0; rr < m; rr++)
                                    qbuf[rr + c * lda] = abuf[rr + c * lda];

                            long info2 = ops.Orgqr(m, qCols, mn, qbuf, lda, tau, work2, lworkOrgqrPass);
                            if (info2 != 0)
                                throw new LinAlgError("Incorrect argument found while performing QR factorization");

                            Delinearize(qbuf, m, qCols, qptr + e * (m * qCols));
                            DelinearizeTriu(abuf, qCols, n, lda, rptr + e * (qCols * n));
                            break;
                        }
                    }

                    AdvanceCoord(coord, shape.dimensions, nb);
                }
            }
            finally
            {
                Free(abuf);
                Free(tau);
                Free(work);
                Free(qbuf);
                Free(work2);
            }

            return (q, r);
        }

        /// <summary>Builds a shape of the leading <paramref name="nb"/> batch dims plus two trailing dims.</summary>
        private static Shape MakeShape(long[] dims, int nb, long d0, long d1)
        {
            var outDims = new long[nb + 2];
            for (int i = 0; i < nb; i++)
                outDims[i] = dims[i];
            outDims[nb] = d0;
            outDims[nb + 1] = d1;
            return new Shape(outDims);
        }

        /// <summary>Builds a shape of the leading <paramref name="nb"/> batch dims plus one trailing dim.</summary>
        private static Shape MakeShape1(long[] dims, int nb, long d0)
        {
            var outDims = new long[nb + 1];
            for (int i = 0; i < nb; i++)
                outDims[i] = dims[i];
            outDims[nb] = d0;
            return new Shape(outDims);
        }

        /// <summary>Writes a stack of leading-identity matrices — orgqr(k=0)'s result — into a C-block.</summary>
        /// <remarks>Self-zeroing (writes the whole block, not just the diagonal) so callers can allocate
        /// the destination <c>fillZeros:false</c> — this is the only degenerate path that would otherwise
        /// rely on the ctor's zeros.</remarks>
        private static void FillIdentityStack<T>(T* dst, long count, long rows, long cols, T one)
            where T : unmanaged
        {
            long block = rows * cols;
            Zero(dst, count * block);
            for (long e = 0; e < count; e++)
            {
                T* mat = dst + e * block;
                long diag = Math.Min(rows, cols);
                for (long d = 0; d < diag; d++)
                    mat[d * cols + d] = one;
            }
        }

        /// <summary>
        ///     Copies the upper triangle of a column-major buffer into a contiguous row-major block,
        ///     zeroing below the diagonal — NumPy's <c>triu</c> fused into the delinearize.
        /// </summary>
        private static void DelinearizeTriu<T>(T* colSrc, long rows, long cols, long lda, T* dstRowMajor)
            where T : unmanaged
        {
            for (long r = 0; r < rows; r++)
            {
                T* row = dstRowMajor + r * cols;
                for (long c = 0; c < cols; c++)
                    row[c] = c >= r ? colSrc[r + c * lda] : default;
            }
        }
    }
}
