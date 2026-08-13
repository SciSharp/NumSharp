using System;
using NumSharp;
using NumSharp.Backends;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NumSharp.Interop.OpenBLAS
{
    internal static unsafe partial class BlasParity
    {
        /// <summary>
        ///     The route decision plus scratch buffers of <c>@TYPE@_matmul</c> (matmul.c.src). NumPy
        ///     computes these once from <c>steps[]</c> — which do not change across the gufunc's outer
        ///     (stacked) loop — and mallocs a single buffer for every temp it may need, so the plan is
        ///     hoisted here the same way and shared by every batch element.
        /// </summary>
        internal struct MatmulPlan<T> where T : unmanaged
        {
            public bool NoblasFallback;
            public bool SpecialCase;
            public bool ScalarOut;
            public bool ScalarVec;
            public bool VectorMatrix;
            public bool MatrixVector;
            public bool I1Blasable;
            public bool I2Blasable;
            public bool OBlasable;
            public bool I1Transpose;
            public bool I2Transpose;
            public bool OTranspose;
            public long TmpIs1M, TmpIs1N, TmpIs2N, TmpIs2P, TmpOsM, TmpOsP;
            public T* Tmp1, Tmp2, TmpO;
            public T* Buffer;
        }

        /// <summary>
        ///     Port of the flag block at the top of <c>@TYPE@_matmul</c>: classifies the operands into
        ///     one of the five routes (noblas / dot / gemv×2 / gemm) and allocates the copy buffers
        ///     the matrix–matrix route needs when an operand is not blasable (gh-12365, gh-23588).
        /// </summary>
        internal static MatmulPlan<T> BuildMatmulPlan<T>(
            long is1M, long is1N, long is2N, long is2P, long osM, long osP,
            long dm, long dn, long dp) where T : unmanaged
        {
            var plan = default(MatmulPlan<T>);

            plan.SpecialCase = dm == 1 || dn == 1 || dp == 1;
            bool anyZeroDim = dm == 0 || dn == 0 || dp == 0;
            plan.ScalarOut = dm == 1 && dp == 1;
            plan.ScalarVec = dn == 1 && (dp == 1 || dm == 1);
            long maxSize = CBlasNative.BlasMaxSize;
            bool tooBigForBlas = dm > maxSize || dn > maxSize || dp > maxSize;

            bool i1CBlasable = IsBlasable2d(is1M, is1N, dm, dn);
            bool i2CBlasable = IsBlasable2d(is2N, is2P, dn, dp);
            bool i1FBlasable = IsBlasable2d(is1N, is1M, dn, dm);
            bool i2FBlasable = IsBlasable2d(is2P, is2N, dp, dn);
            plan.I1Blasable = i1CBlasable || i1FBlasable;
            plan.I2Blasable = i2CBlasable || i2FBlasable;
            plan.OBlasable = IsBlasable2d(osM, osP, dm, dp) || IsBlasable2d(osP, osM, dp, dm);

            plan.VectorMatrix = dm == 1 && plan.I2Blasable && IsBlasable2d(is1N, 1, dn, 1);
            plan.MatrixVector = dp == 1 && plan.I1Blasable && IsBlasable2d(is2N, 1, dn, 1);
            plan.NoblasFallback = tooBigForBlas || anyZeroDim;

            bool matrixMatrix = !plan.NoblasFallback && !plan.SpecialCase;
            bool allocateBuffer = matrixMatrix && (!plan.I1Blasable || !plan.I2Blasable || !plan.OBlasable);

            plan.I1Transpose = Math.Abs(is1M) < Math.Abs(is1N);
            plan.I2Transpose = Math.Abs(is2N) < Math.Abs(is2P);
            plan.OTranspose = Math.Abs(osM) < Math.Abs(osP);
            plan.TmpIs1M = plan.I1Transpose ? 1 : dn;
            plan.TmpIs1N = plan.I1Transpose ? dm : 1;
            plan.TmpIs2N = plan.I2Transpose ? 1 : dp;
            plan.TmpIs2P = plan.I2Transpose ? dn : 1;
            plan.TmpOsM = plan.OTranspose ? 1 : dp;
            plan.TmpOsP = plan.OTranspose ? dm : 1;

            if (allocateBuffer)
            {
                long ip1Size = plan.I1Blasable ? 0 : dm * dn;
                long ip2Size = plan.I2Blasable ? 0 : dn * dp;
                long opSize = plan.OBlasable ? 0 : dm * dp;
                plan.Buffer = Alloc<T>(ip1Size + ip2Size + opSize);
                plan.Tmp1 = plan.Buffer;
                plan.Tmp2 = plan.Buffer + ip1Size;
                plan.TmpO = plan.Buffer + ip1Size + ip2Size;
            }

            return plan;
        }

        internal static void FreeMatmulPlan<T>(ref MatmulPlan<T> plan) where T : unmanaged
        {
            Free(plan.Buffer);
            plan.Buffer = null;
        }

        /// <summary>
        ///     Port of one iteration of <c>@TYPE@_matmul</c>'s outer loop — the five-way route split
        ///     (noblas fallback / row·column dot / vector·matrix gemv / matrix·vector gemv /
        ///     matrix·matrix gemm, the last copying non-blasable operands into the plan's scratch).
        /// </summary>
        internal static void MatmulCore<T, TOps>(ref MatmulPlan<T> plan,
            T* ip1, long is1M, long is1N,
            T* ip2, long is2N, long is2P,
            T* op, long osM, long osP,
            long dm, long dn, long dp)
            where T : unmanaged, INumberBase<T>
            where TOps : struct, IBlasType<T>
        {
            var ops = default(TOps);

            if (plan.NoblasFallback)
            {
                MatmulInnerNoBlas(ip1, is1M, is1N, ip2, is2N, is2P, op, osM, osP, dm, dn, dp);
                return;
            }

            if (plan.SpecialCase)
            {
                // Special case variants that have a 1 in the core dimensions.
                if (plan.ScalarOut)
                {
                    // row @ column, 1,1 output
                    ops.Dot(ip1, is1N, ip2, is2N, op, dn);
                }
                else if (plan.ScalarVec)
                {
                    // 1,1d @ vector or vector @ 1,1d — NumPy notes cblas_Xaxy would need a zeroed
                    // output and would not be faster, so it stays on the portable loop.
                    MatmulInnerNoBlas(ip1, is1M, is1N, ip2, is2N, is2P, op, osM, osP, dm, dn, dp);
                }
                else if (plan.VectorMatrix)
                {
                    // vector @ matrix, switch ip1, ip2, p and m
                    Gemv<T, TOps>(ip2, is2P, is2N, ip1, is1N, op, osP, dp, dn);
                }
                else if (plan.MatrixVector)
                {
                    // matrix @ vector
                    Gemv<T, TOps>(ip1, is1M, is1N, ip2, is2N, op, osM, dm, dn);
                }
                else
                {
                    // column @ row, 2d output, no blas needed or non-blas-able input
                    MatmulInnerNoBlas(ip1, is1M, is1N, ip2, is2N, is2P, op, osM, osP, dm, dn, dp);
                }

                return;
            }

            // matrix @ matrix — copy if not blasable, see gh-12365 & gh-23588.
            if (!plan.I1Blasable)
                MatrixCopy(plan.I1Transpose, ip1, is1M, is1N, plan.Tmp1, plan.TmpIs1M, plan.TmpIs1N, dm, dn);

            if (!plan.I2Blasable)
                MatrixCopy(plan.I2Transpose, ip2, is2N, is2P, plan.Tmp2, plan.TmpIs2N, plan.TmpIs2P, dn, dp);

            T* ip1_ = plan.I1Blasable ? ip1 : plan.Tmp1;
            T* ip2_ = plan.I2Blasable ? ip2 : plan.Tmp2;
            T* op_ = plan.OBlasable ? op : plan.TmpO;

            long is1M_ = plan.I1Blasable ? is1M : plan.TmpIs1M;
            long is1N_ = plan.I1Blasable ? is1N : plan.TmpIs1N;
            long is2N_ = plan.I2Blasable ? is2N : plan.TmpIs2N;
            long is2P_ = plan.I2Blasable ? is2P : plan.TmpIs2P;
            long osM_ = plan.OBlasable ? osM : plan.TmpOsM;
            long osP_ = plan.OBlasable ? osP : plan.TmpOsP;

            // Use the transpose equivalence matmul(a, b, o) == matmul(b.T, a.T, o.T).
            if (plan.OTranspose)
                MatmulMatrixMatrix<T, TOps>(ip2_, is2P_, is2N_, ip1_, is1N_, is1M_, op_, osP_, osM_, dp, dn, dm);
            else
                MatmulMatrixMatrix<T, TOps>(ip1_, is1M_, is1N_, ip2_, is2N_, is2P_, op_, osM_, osP_, dm, dn, dp);

            if (!plan.OBlasable)
                MatrixCopy(plan.OTranspose, plan.TmpO, plan.TmpOsM, plan.TmpOsP, op, osM, osP, dm, dp);
        }

        /// <summary>
        ///     Port of <c>@name@_gemv</c> (matmul.c.src): vector–matrix product, Level 2 BLAS. The
        ///     matrix is passed column-major when its row stride is the contiguous one, otherwise
        ///     row-major; either way the call is <c>CblasTrans</c> with N and M swapped.
        /// </summary>
        private static void Gemv<T, TOps>(T* ip1, long is1M, long is1N, T* ip2, long is2N,
            T* op, long opM, long m, long n)
            where T : unmanaged
            where TOps : struct, IBlasType<T>
        {
            var ops = default(TOps);
            CBlasOrder order;
            long lda;

            if (IsBlasable2d(is1M, is1N, m, n))
            {
                order = CBlasOrder.ColMajor;
                lda = is1M;
            }
            else
            {
                // If not ColMajor, caller should have ensured we are RowMajor.
                order = CBlasOrder.RowMajor;
                lda = is1N;
            }

            ops.Gemv(order, CBlasTranspose.Trans, n, m, ip1, lda, ip2, is2N, op, opM);
        }

        /// <summary>
        ///     Port of <c>@name@_matmul_matrixmatrix</c> (matmul.c.src): Level 3 BLAS, row-major, with
        ///     the per-operand transpose flag chosen from which axis is contiguous — and NumPy's
        ///     <c>syrk</c> shortcut when an operand is multiplied by its own transpose (the upper
        ///     triangle is computed and mirrored down).
        /// </summary>
        private static void MatmulMatrixMatrix<T, TOps>(T* ip1, long is1M, long is1N,
            T* ip2, long is2N, long is2P, T* op, long osM, long osP, long m, long n, long p)
            where T : unmanaged
            where TOps : struct, IBlasType<T>
        {
            var ops = default(TOps);
            const CBlasOrder order = CBlasOrder.RowMajor;
            CBlasTranspose trans1, trans2;
            long lda, ldb;
            long ldc = osM;

            if (IsBlasable2d(is1M, is1N, m, n))
            {
                trans1 = CBlasTranspose.NoTrans;
                lda = is1M;
            }
            else
            {
                trans1 = CBlasTranspose.Trans;
                lda = is1N;
            }

            if (IsBlasable2d(is2N, is2P, n, p))
            {
                trans2 = CBlasTranspose.NoTrans;
                ldb = is2N;
            }
            else
            {
                trans2 = CBlasTranspose.Trans;
                ldb = is2P;
            }

            // Use syrk if we have a case of a matrix times its transpose. Otherwise, use gemm.
            if (ip1 == ip2 && m == p && is1M == is2P && is1N == is2N && trans1 != trans2)
            {
                ops.Syrk(order, CBlasUpLo.Upper, trans1, p, n,
                    ip1, trans1 == CBlasTranspose.NoTrans ? lda : ldb, op, ldc);

                // Copy the triangle.
                for (long i = 0; i < p; i++)
                    for (long j = i + 1; j < p; j++)
                        op[j * ldc + i] = op[i * ldc + j];
            }
            else
            {
                ops.Gemm(order, trans1, trans2, m, p, n, ip1, lda, ip2, ldb, op, ldc);
            }
        }
    }
}
