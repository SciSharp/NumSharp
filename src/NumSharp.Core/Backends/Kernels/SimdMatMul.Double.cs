using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

// =============================================================================
// Stride-aware double GEMM (BLIS-style GEBP, mirrors SimdMatMul.Strided.cs)
// =============================================================================
//
// Same design as the float kernel with Vector256<double> (4 doubles per
// vector): the packing stage absorbs all stride variation — transposed /
// sliced views are copied into MR- and NR_D-packed micro-kernel panels, so the
// 8x8 micro-kernel reads only contiguous buffers and is stride-agnostic.
//
// The micro-kernel is MR=8 rows × NR_D=8 cols (2 vectors of 4 doubles) with 16
// Vector256<double> accumulators — the exact register pressure the float 8x16
// kernel (16 Vector256<float> accumulators) already proves works on AVX2.
//
// Fast paths in the packers (same set as the float packers):
//   PackA, aStride0 == 1  — transposed-contiguous A, two 4-wide loads per k.
//   PackB, bStride1 == 1  — row-contiguous B, two 4-wide loads per k.
//   PackB, bStride0 == 1  — transposed-contiguous B, K-long contiguous read
//                           per column, scalar scatter-store.
//
// Everything else falls through to scalar element access. Packing is
// O(M*K + K*N) while GEMM is O(M*N*K), so the ratio is 1/N + 1/M — for any
// matrix large enough to care about, packing is <3% of the total work.
//
// Small matrices (all dims <= BLOCKING_THRESHOLD) keep the simple IKJ SIMD
// loop when B's inner stride is 1 — zero packing/alloc overhead there. When
// bStride1 != 1 the simple path's inner loop is SCALAR, and blocked (whose
// packing restores SIMD) is faster from ~4096 MACs up (measured: 3.0x at
// 16^3 growing to 5.9x at 128^3; the worst mid-size cell, 8x128x128, is
// 1.04x), so mid-size strided-B routes to blocked above that floor. Below it
// the simple path stays — the two pack-buffer allocs (~0.5 us) would dominate
// micro-dots (e.g. batched 4x4 stacks), and the tiny bit-exact fuzz-corpus
// products keep their committed accumulation order.
//
// Measured on 500×2000 @ 2000×500 (Release, best-of-5): transposed B (the
// A@A.T pattern np.cov hits) 346 ms → 23 ms (2.9 → 44 GFLOP/s), contiguous
// 52 ms → 23 ms, transposed A 55 ms → 22 ms. K-loop unrolling stays at 4x:
// an 8x variant was measured 25% SLOWER at every shape/layout (doubled
// live-range pressure on 16 YMM registers spills accumulators).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class SimdMatMul
    {
        // Micro-kernel cols for double: 2 × Vector256<double> = 8 lanes.
        // (MR, MC, KC, BLOCKING_THRESHOLD are shared with the float kernels;
        // NR = 16 is the float micro-kernel width, so double needs its own.)
        private const int NR_D = 8;

        // When B's inner stride isn't 1 the simple path degenerates to a scalar
        // inner loop, so blocked (which repacks B into contiguous panels) wins
        // even below BLOCKING_THRESHOLD — measured faster from 16^3 (= 4096
        // MACs, 3.0x) upward. Below this floor the pack-buffer allocations
        // dominate and the simple path stays.
        private const int SCALAR_FALLBACK_MAX_WORK = 4096;

        /// <summary>
        /// Stride-aware double matrix multiply: C = A * B.
        /// A is logical (M, K) with strides (aStride0, aStride1) in elements.
        /// B is logical (K, N) with strides (bStride0, bStride1) in elements.
        /// C is written as M×N row-major contiguous (ldc = N).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void MatMulDouble(
            double* A, long aStride0, long aStride1,
            double* B, long bStride0, long bStride1,
            double* C,
            long M, long N, long K)
        {
            new UnmanagedSpan<double>(C, M * N).Clear();

            if (M == 0 || N == 0 || K == 0)
                return;

            if (M <= BLOCKING_THRESHOLD && N <= BLOCKING_THRESHOLD && K <= BLOCKING_THRESHOLD
                && (bStride1 == 1 || M * N * K < SCALAR_FALLBACK_MAX_WORK))
            {
                MatMulDoubleSimpleStrided(A, aStride0, aStride1, B, bStride0, bStride1, C, M, N, K);
                return;
            }

            MatMulDoubleBlockedStrided(A, aStride0, aStride1, B, bStride0, bStride1, C, M, N, K);
        }

        // =====================================================================
        // Simple IKJ path (small matrices)
        // =====================================================================

        /// <summary>
        /// Stride-aware IKJ SIMD kernel. Inner loop uses Vector256&lt;double&gt;
        /// (4 doubles per FMA) when <paramref name="bStride1"/> is 1; falls
        /// back to scalar otherwise.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulDoubleSimpleStrided(
            double* A, long aStride0, long aStride1,
            double* B, long bStride0, long bStride1,
            double* C, long M, long N, long K)
        {
            if (bStride1 == 1)
            {
                for (long i = 0; i < M; i++)
                {
                    double* cRow = C + i * N;
                    long aRowBase = i * aStride0;

                    for (long k = 0; k < K; k++)
                    {
                        double aik = A[aRowBase + k * aStride1];
                        var aikVec = Vector256.Create(aik);
                        double* bRow = B + k * bStride0;

                        long j = 0;
                        for (; j <= N - 4; j += 4)
                        {
                            var cVec = Vector256.Load(cRow + j);
                            var bVec = Vector256.Load(bRow + j);
                            cVec = Fma.IsSupported
                                ? Fma.MultiplyAdd(aikVec, bVec, cVec)
                                : Vector256.Add(cVec, Vector256.Multiply(aikVec, bVec));
                            Vector256.Store(cVec, cRow + j);
                        }
                        for (; j < N; j++)
                            cRow[j] += aik * bRow[j];
                    }
                }
            }
            else
            {
                // B strided on the inner axis — scalar inner loop. This is
                // the TransB case; for larger matrices the blocked path
                // (which packs into contiguous panels) restores SIMD speed.
                for (long i = 0; i < M; i++)
                {
                    double* cRow = C + i * N;
                    long aRowBase = i * aStride0;

                    for (long k = 0; k < K; k++)
                    {
                        double aik = A[aRowBase + k * aStride1];
                        long bRowBase = k * bStride0;
                        for (long j = 0; j < N; j++)
                            cRow[j] += aik * B[bRowBase + j * bStride1];
                    }
                }
            }
        }

        // =====================================================================
        // Blocked GEBP path (large matrices)
        // =====================================================================

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulDoubleBlockedStrided(
            double* A, long aStride0, long aStride1,
            double* B, long bStride0, long bStride1,
            double* C, long M, long N, long K)
        {
            long numNPanels = (N + NR_D - 1) / NR_D;

            double* packA = (double*)NativeMemory.AlignedAlloc((nuint)(MC * KC * sizeof(double)), 64);
            double* packB = (double*)NativeMemory.AlignedAlloc((nuint)(numNPanels * KC * NR_D * sizeof(double)), 64);

            try
            {
                for (long k0 = 0; k0 < K; k0 += KC)
                {
                    int kc = (int)Math.Min(KC, K - k0);

                    PackBDoublePanelsStrided(B, bStride0, bStride1, packB, N, k0, kc);

                    for (long i0 = 0; i0 < M; i0 += MC)
                    {
                        int mc = (int)Math.Min(MC, M - i0);

                        PackADoublePanelsStrided(A, aStride0, aStride1, packA, i0, k0, mc, kc);

                        for (int ip = 0; ip < mc; ip += MR)
                        {
                            int mr = Math.Min(MR, mc - ip);
                            double* aPanel = packA + (ip / MR) * kc * MR;

                            for (long jp = 0; jp < N; jp += NR_D)
                            {
                                int nr = (int)Math.Min(NR_D, N - jp);
                                double* bPanel = packB + (jp / NR_D) * kc * NR_D;

                                if (mr == MR && nr == NR_D)
                                    Microkernel8x8Packed(aPanel, bPanel, C, N, i0 + ip, jp, kc);
                                else
                                    MicrokernelGenericDoublePacked(aPanel, bPanel, C, N, i0 + ip, jp, kc, mr, nr);
                            }
                        }
                    }
                }
            }
            finally
            {
                NativeMemory.AlignedFree(packA);
                NativeMemory.AlignedFree(packB);
            }
        }

        // =====================================================================
        // Strided packers
        // =====================================================================

        /// <summary>
        /// Pack a slice of A (rows i0..i0+mc, cols k0..k0+kc) into MR-row
        /// interleaved panels. Layout matches PackAPanelsStrided:
        ///   aPanel[(ip/MR) * kc * MR + k * MR + row].
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void PackADoublePanelsStrided(
            double* A, long aStride0, long aStride1,
            double* packA, long i0, long k0, int mc, int kc)
        {
            for (int ip = 0; ip < mc; ip += MR)
            {
                int mr = Math.Min(MR, mc - ip);
                double* aPanel = packA + (ip / MR) * kc * MR;

                if (mr == MR)
                {
                    if (aStride0 == 1)
                    {
                        // Transposed-contiguous A: 8 consecutive logical rows
                        // sit at 8 consecutive memory addresses (per fixed k),
                        // because A[i, k] = A + i*1 + k*aStride1.
                        // Two Vector256 loads pack 8 rows.
                        for (int k = 0; k < kc; k++)
                        {
                            long srcOff = (i0 + ip) + (k0 + k) * aStride1;
                            double* dst = aPanel + k * MR;
                            Vector256.Store(Vector256.Load(A + srcOff), dst);
                            Vector256.Store(Vector256.Load(A + srcOff + 4), dst + 4);
                        }
                    }
                    else
                    {
                        for (int k = 0; k < kc; k++)
                        {
                            double* dst = aPanel + k * MR;
                            long kOff = (k0 + k) * aStride1;
                            dst[0] = A[(i0 + ip + 0) * aStride0 + kOff];
                            dst[1] = A[(i0 + ip + 1) * aStride0 + kOff];
                            dst[2] = A[(i0 + ip + 2) * aStride0 + kOff];
                            dst[3] = A[(i0 + ip + 3) * aStride0 + kOff];
                            dst[4] = A[(i0 + ip + 4) * aStride0 + kOff];
                            dst[5] = A[(i0 + ip + 5) * aStride0 + kOff];
                            dst[6] = A[(i0 + ip + 6) * aStride0 + kOff];
                            dst[7] = A[(i0 + ip + 7) * aStride0 + kOff];
                        }
                    }
                }
                else
                {
                    // Partial edge panel — zero-pad missing rows.
                    for (int k = 0; k < kc; k++)
                    {
                        double* dst = aPanel + k * MR;
                        long kOff = (k0 + k) * aStride1;
                        for (int ii = 0; ii < MR; ii++)
                            dst[ii] = ii < mr ? A[(i0 + ip + ii) * aStride0 + kOff] : 0d;
                    }
                }
            }
        }

        /// <summary>
        /// Pack a K-slice of B (rows k0..k0+kc, all N columns) into NR_D-column
        /// panels. Layout matches PackBPanelsStrided:
        ///   bPanel[(jp/NR_D) * kc * NR_D + k * NR_D + col].
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void PackBDoublePanelsStrided(
            double* B, long bStride0, long bStride1,
            double* packB, long N_total, long k0, int kc)
        {
            for (long jp = 0; jp < N_total; jp += NR_D)
            {
                int nr = (int)Math.Min(NR_D, N_total - jp);
                double* bPanel = packB + (jp / NR_D) * kc * NR_D;

                if (bStride1 == 1)
                {
                    // Row-contiguous B: 8 consecutive doubles per k.
                    if (nr == NR_D)
                    {
                        for (int k = 0; k < kc; k++)
                        {
                            double* src = B + (k0 + k) * bStride0 + jp;
                            double* dst = bPanel + k * NR_D;
                            Vector256.Store(Vector256.Load(src), dst);
                            Vector256.Store(Vector256.Load(src + 4), dst + 4);
                        }
                    }
                    else
                    {
                        for (int k = 0; k < kc; k++)
                        {
                            double* src = B + (k0 + k) * bStride0 + jp;
                            double* dst = bPanel + k * NR_D;
                            for (int jj = 0; jj < NR_D; jj++)
                                dst[jj] = jj < nr ? src[jj] : 0d;
                        }
                    }
                }
                else if (bStride0 == 1)
                {
                    // Transposed-contiguous B: each logical column is a
                    // contiguous K-long run in memory at offset j*bStride1.
                    // Zero the panel first (handles partial-panel padding),
                    // then fill column-by-column with contiguous reads.
                    long panelDoubles = (long)kc * NR_D;
                    new UnmanagedSpan<double>(bPanel, panelDoubles).Clear();

                    for (int jj = 0; jj < nr; jj++)
                    {
                        double* colStart = B + (jp + jj) * bStride1 + k0;
                        // Scalar scatter — writes have stride NR_D which isn't
                        // SIMD-friendly on AVX2, but reads are contiguous.
                        for (int k = 0; k < kc; k++)
                            bPanel[k * NR_D + jj] = colStart[k];
                    }
                }
                else
                {
                    // Fully general: scalar reads using both strides.
                    for (int k = 0; k < kc; k++)
                    {
                        double* dst = bPanel + k * NR_D;
                        long kOff = (k0 + k) * bStride0;
                        for (int jj = 0; jj < NR_D; jj++)
                            dst[jj] = jj < nr ? B[kOff + (jp + jj) * bStride1] : 0d;
                    }
                }
            }
        }

        // =====================================================================
        // Micro-kernels
        // =====================================================================

        /// <summary>
        /// 8x8 double micro-kernel with full panel packing and k-loop unrolling.
        /// Both A and B are in packed panel format for optimal cache access:
        /// - A panel: aPanel[k * MR + row] - 8 doubles contiguous per k
        /// - B panel: bPanel[k * NR_D + col] - 8 doubles contiguous per k
        /// Uses long for i, j, ldc to support large matrices.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void Microkernel8x8Packed(double* aPanel, double* bPanel, double* C, long ldc, long i, long j, int kc)
        {
            // Load C accumulators (8 rows x 2 vectors = 16 accumulators)
            var c00 = Vector256.Load(C + (i + 0) * ldc + j);
            var c01 = Vector256.Load(C + (i + 0) * ldc + j + 4);
            var c10 = Vector256.Load(C + (i + 1) * ldc + j);
            var c11 = Vector256.Load(C + (i + 1) * ldc + j + 4);
            var c20 = Vector256.Load(C + (i + 2) * ldc + j);
            var c21 = Vector256.Load(C + (i + 2) * ldc + j + 4);
            var c30 = Vector256.Load(C + (i + 3) * ldc + j);
            var c31 = Vector256.Load(C + (i + 3) * ldc + j + 4);
            var c40 = Vector256.Load(C + (i + 4) * ldc + j);
            var c41 = Vector256.Load(C + (i + 4) * ldc + j + 4);
            var c50 = Vector256.Load(C + (i + 5) * ldc + j);
            var c51 = Vector256.Load(C + (i + 5) * ldc + j + 4);
            var c60 = Vector256.Load(C + (i + 6) * ldc + j);
            var c61 = Vector256.Load(C + (i + 6) * ldc + j + 4);
            var c70 = Vector256.Load(C + (i + 7) * ldc + j);
            var c71 = Vector256.Load(C + (i + 7) * ldc + j + 4);

            // K-loop with 4x unrolling for instruction-level parallelism
            int k = 0;
            if (Fma.IsSupported)
            {
                for (; k <= kc - 4; k += 4)
                {
                    // Load B panel rows (contiguous: bPanel[k*8..k*8+7])
                    var b00 = Vector256.Load(bPanel + (k + 0) * NR_D);
                    var b01 = Vector256.Load(bPanel + (k + 0) * NR_D + 4);
                    var b10 = Vector256.Load(bPanel + (k + 1) * NR_D);
                    var b11 = Vector256.Load(bPanel + (k + 1) * NR_D + 4);
                    var b20 = Vector256.Load(bPanel + (k + 2) * NR_D);
                    var b21 = Vector256.Load(bPanel + (k + 2) * NR_D + 4);
                    var b30 = Vector256.Load(bPanel + (k + 3) * NR_D);
                    var b31 = Vector256.Load(bPanel + (k + 3) * NR_D + 4);

                    // A panel pointers (contiguous: aPanel[k*8..k*8+7])
                    double* ak0 = aPanel + (k + 0) * MR;
                    double* ak1 = aPanel + (k + 1) * MR;
                    double* ak2 = aPanel + (k + 2) * MR;
                    double* ak3 = aPanel + (k + 3) * MR;

                    // k+0: 16 FMAs (8 rows x 2 vectors)
                    c00 = Fma.MultiplyAdd(Vector256.Create(ak0[0]), b00, c00);
                    c01 = Fma.MultiplyAdd(Vector256.Create(ak0[0]), b01, c01);
                    c10 = Fma.MultiplyAdd(Vector256.Create(ak0[1]), b00, c10);
                    c11 = Fma.MultiplyAdd(Vector256.Create(ak0[1]), b01, c11);
                    c20 = Fma.MultiplyAdd(Vector256.Create(ak0[2]), b00, c20);
                    c21 = Fma.MultiplyAdd(Vector256.Create(ak0[2]), b01, c21);
                    c30 = Fma.MultiplyAdd(Vector256.Create(ak0[3]), b00, c30);
                    c31 = Fma.MultiplyAdd(Vector256.Create(ak0[3]), b01, c31);
                    c40 = Fma.MultiplyAdd(Vector256.Create(ak0[4]), b00, c40);
                    c41 = Fma.MultiplyAdd(Vector256.Create(ak0[4]), b01, c41);
                    c50 = Fma.MultiplyAdd(Vector256.Create(ak0[5]), b00, c50);
                    c51 = Fma.MultiplyAdd(Vector256.Create(ak0[5]), b01, c51);
                    c60 = Fma.MultiplyAdd(Vector256.Create(ak0[6]), b00, c60);
                    c61 = Fma.MultiplyAdd(Vector256.Create(ak0[6]), b01, c61);
                    c70 = Fma.MultiplyAdd(Vector256.Create(ak0[7]), b00, c70);
                    c71 = Fma.MultiplyAdd(Vector256.Create(ak0[7]), b01, c71);

                    // k+1
                    c00 = Fma.MultiplyAdd(Vector256.Create(ak1[0]), b10, c00);
                    c01 = Fma.MultiplyAdd(Vector256.Create(ak1[0]), b11, c01);
                    c10 = Fma.MultiplyAdd(Vector256.Create(ak1[1]), b10, c10);
                    c11 = Fma.MultiplyAdd(Vector256.Create(ak1[1]), b11, c11);
                    c20 = Fma.MultiplyAdd(Vector256.Create(ak1[2]), b10, c20);
                    c21 = Fma.MultiplyAdd(Vector256.Create(ak1[2]), b11, c21);
                    c30 = Fma.MultiplyAdd(Vector256.Create(ak1[3]), b10, c30);
                    c31 = Fma.MultiplyAdd(Vector256.Create(ak1[3]), b11, c31);
                    c40 = Fma.MultiplyAdd(Vector256.Create(ak1[4]), b10, c40);
                    c41 = Fma.MultiplyAdd(Vector256.Create(ak1[4]), b11, c41);
                    c50 = Fma.MultiplyAdd(Vector256.Create(ak1[5]), b10, c50);
                    c51 = Fma.MultiplyAdd(Vector256.Create(ak1[5]), b11, c51);
                    c60 = Fma.MultiplyAdd(Vector256.Create(ak1[6]), b10, c60);
                    c61 = Fma.MultiplyAdd(Vector256.Create(ak1[6]), b11, c61);
                    c70 = Fma.MultiplyAdd(Vector256.Create(ak1[7]), b10, c70);
                    c71 = Fma.MultiplyAdd(Vector256.Create(ak1[7]), b11, c71);

                    // k+2
                    c00 = Fma.MultiplyAdd(Vector256.Create(ak2[0]), b20, c00);
                    c01 = Fma.MultiplyAdd(Vector256.Create(ak2[0]), b21, c01);
                    c10 = Fma.MultiplyAdd(Vector256.Create(ak2[1]), b20, c10);
                    c11 = Fma.MultiplyAdd(Vector256.Create(ak2[1]), b21, c11);
                    c20 = Fma.MultiplyAdd(Vector256.Create(ak2[2]), b20, c20);
                    c21 = Fma.MultiplyAdd(Vector256.Create(ak2[2]), b21, c21);
                    c30 = Fma.MultiplyAdd(Vector256.Create(ak2[3]), b20, c30);
                    c31 = Fma.MultiplyAdd(Vector256.Create(ak2[3]), b21, c31);
                    c40 = Fma.MultiplyAdd(Vector256.Create(ak2[4]), b20, c40);
                    c41 = Fma.MultiplyAdd(Vector256.Create(ak2[4]), b21, c41);
                    c50 = Fma.MultiplyAdd(Vector256.Create(ak2[5]), b20, c50);
                    c51 = Fma.MultiplyAdd(Vector256.Create(ak2[5]), b21, c51);
                    c60 = Fma.MultiplyAdd(Vector256.Create(ak2[6]), b20, c60);
                    c61 = Fma.MultiplyAdd(Vector256.Create(ak2[6]), b21, c61);
                    c70 = Fma.MultiplyAdd(Vector256.Create(ak2[7]), b20, c70);
                    c71 = Fma.MultiplyAdd(Vector256.Create(ak2[7]), b21, c71);

                    // k+3
                    c00 = Fma.MultiplyAdd(Vector256.Create(ak3[0]), b30, c00);
                    c01 = Fma.MultiplyAdd(Vector256.Create(ak3[0]), b31, c01);
                    c10 = Fma.MultiplyAdd(Vector256.Create(ak3[1]), b30, c10);
                    c11 = Fma.MultiplyAdd(Vector256.Create(ak3[1]), b31, c11);
                    c20 = Fma.MultiplyAdd(Vector256.Create(ak3[2]), b30, c20);
                    c21 = Fma.MultiplyAdd(Vector256.Create(ak3[2]), b31, c21);
                    c30 = Fma.MultiplyAdd(Vector256.Create(ak3[3]), b30, c30);
                    c31 = Fma.MultiplyAdd(Vector256.Create(ak3[3]), b31, c31);
                    c40 = Fma.MultiplyAdd(Vector256.Create(ak3[4]), b30, c40);
                    c41 = Fma.MultiplyAdd(Vector256.Create(ak3[4]), b31, c41);
                    c50 = Fma.MultiplyAdd(Vector256.Create(ak3[5]), b30, c50);
                    c51 = Fma.MultiplyAdd(Vector256.Create(ak3[5]), b31, c51);
                    c60 = Fma.MultiplyAdd(Vector256.Create(ak3[6]), b30, c60);
                    c61 = Fma.MultiplyAdd(Vector256.Create(ak3[6]), b31, c61);
                    c70 = Fma.MultiplyAdd(Vector256.Create(ak3[7]), b30, c70);
                    c71 = Fma.MultiplyAdd(Vector256.Create(ak3[7]), b31, c71);
                }
            }

            // Remainder k iterations
            for (; k < kc; k++)
            {
                var b0 = Vector256.Load(bPanel + k * NR_D);
                var b1 = Vector256.Load(bPanel + k * NR_D + 4);
                double* ak = aPanel + k * MR;

                if (Fma.IsSupported)
                {
                    c00 = Fma.MultiplyAdd(Vector256.Create(ak[0]), b0, c00);
                    c01 = Fma.MultiplyAdd(Vector256.Create(ak[0]), b1, c01);
                    c10 = Fma.MultiplyAdd(Vector256.Create(ak[1]), b0, c10);
                    c11 = Fma.MultiplyAdd(Vector256.Create(ak[1]), b1, c11);
                    c20 = Fma.MultiplyAdd(Vector256.Create(ak[2]), b0, c20);
                    c21 = Fma.MultiplyAdd(Vector256.Create(ak[2]), b1, c21);
                    c30 = Fma.MultiplyAdd(Vector256.Create(ak[3]), b0, c30);
                    c31 = Fma.MultiplyAdd(Vector256.Create(ak[3]), b1, c31);
                    c40 = Fma.MultiplyAdd(Vector256.Create(ak[4]), b0, c40);
                    c41 = Fma.MultiplyAdd(Vector256.Create(ak[4]), b1, c41);
                    c50 = Fma.MultiplyAdd(Vector256.Create(ak[5]), b0, c50);
                    c51 = Fma.MultiplyAdd(Vector256.Create(ak[5]), b1, c51);
                    c60 = Fma.MultiplyAdd(Vector256.Create(ak[6]), b0, c60);
                    c61 = Fma.MultiplyAdd(Vector256.Create(ak[6]), b1, c61);
                    c70 = Fma.MultiplyAdd(Vector256.Create(ak[7]), b0, c70);
                    c71 = Fma.MultiplyAdd(Vector256.Create(ak[7]), b1, c71);
                }
                else
                {
                    c00 = Vector256.Add(c00, Vector256.Multiply(Vector256.Create(ak[0]), b0));
                    c01 = Vector256.Add(c01, Vector256.Multiply(Vector256.Create(ak[0]), b1));
                    c10 = Vector256.Add(c10, Vector256.Multiply(Vector256.Create(ak[1]), b0));
                    c11 = Vector256.Add(c11, Vector256.Multiply(Vector256.Create(ak[1]), b1));
                    c20 = Vector256.Add(c20, Vector256.Multiply(Vector256.Create(ak[2]), b0));
                    c21 = Vector256.Add(c21, Vector256.Multiply(Vector256.Create(ak[2]), b1));
                    c30 = Vector256.Add(c30, Vector256.Multiply(Vector256.Create(ak[3]), b0));
                    c31 = Vector256.Add(c31, Vector256.Multiply(Vector256.Create(ak[3]), b1));
                    c40 = Vector256.Add(c40, Vector256.Multiply(Vector256.Create(ak[4]), b0));
                    c41 = Vector256.Add(c41, Vector256.Multiply(Vector256.Create(ak[4]), b1));
                    c50 = Vector256.Add(c50, Vector256.Multiply(Vector256.Create(ak[5]), b0));
                    c51 = Vector256.Add(c51, Vector256.Multiply(Vector256.Create(ak[5]), b1));
                    c60 = Vector256.Add(c60, Vector256.Multiply(Vector256.Create(ak[6]), b0));
                    c61 = Vector256.Add(c61, Vector256.Multiply(Vector256.Create(ak[6]), b1));
                    c70 = Vector256.Add(c70, Vector256.Multiply(Vector256.Create(ak[7]), b0));
                    c71 = Vector256.Add(c71, Vector256.Multiply(Vector256.Create(ak[7]), b1));
                }
            }

            // Store results
            Vector256.Store(c00, C + (i + 0) * ldc + j);
            Vector256.Store(c01, C + (i + 0) * ldc + j + 4);
            Vector256.Store(c10, C + (i + 1) * ldc + j);
            Vector256.Store(c11, C + (i + 1) * ldc + j + 4);
            Vector256.Store(c20, C + (i + 2) * ldc + j);
            Vector256.Store(c21, C + (i + 2) * ldc + j + 4);
            Vector256.Store(c30, C + (i + 3) * ldc + j);
            Vector256.Store(c31, C + (i + 3) * ldc + j + 4);
            Vector256.Store(c40, C + (i + 4) * ldc + j);
            Vector256.Store(c41, C + (i + 4) * ldc + j + 4);
            Vector256.Store(c50, C + (i + 5) * ldc + j);
            Vector256.Store(c51, C + (i + 5) * ldc + j + 4);
            Vector256.Store(c60, C + (i + 6) * ldc + j);
            Vector256.Store(c61, C + (i + 6) * ldc + j + 4);
            Vector256.Store(c70, C + (i + 7) * ldc + j);
            Vector256.Store(c71, C + (i + 7) * ldc + j + 4);
        }

        /// <summary>
        /// Generic double micro-kernel for edge cases (partial rows/cols) with
        /// packed panels. Uses long for i, j, ldc to support large matrices.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MicrokernelGenericDoublePacked(double* aPanel, double* bPanel, double* C, long ldc, long i, long j, int kc, int mr, int nr)
        {
            for (int ii = 0; ii < mr; ii++)
            {
                double* cRow = C + (i + ii) * ldc + j;

                // Use SIMD for full vectors, scalar for remainder
                if (nr >= 4 && Fma.IsSupported)
                {
                    var c0 = Vector256.Load(cRow);
                    var c1 = nr >= 8 ? Vector256.Load(cRow + 4) : Vector256<double>.Zero;

                    for (int k = 0; k < kc; k++)
                    {
                        var aVal = Vector256.Create(aPanel[k * MR + ii]);
                        var b0 = Vector256.Load(bPanel + k * NR_D);
                        c0 = Fma.MultiplyAdd(aVal, b0, c0);
                        if (nr >= 8)
                        {
                            var b1 = Vector256.Load(bPanel + k * NR_D + 4);
                            c1 = Fma.MultiplyAdd(aVal, b1, c1);
                        }
                    }

                    Vector256.Store(c0, cRow);
                    if (nr >= 8)
                        Vector256.Store(c1, cRow + 4);
                    else
                    {
                        // Handle 4-7 columns: scalar for remainder
                        for (int jj = 4; jj < nr; jj++)
                        {
                            double sum = cRow[jj];
                            for (int k = 0; k < kc; k++)
                                sum += aPanel[k * MR + ii] * bPanel[k * NR_D + jj];
                            cRow[jj] = sum;
                        }
                    }
                }
                else
                {
                    // Full scalar fallback
                    for (int jj = 0; jj < nr; jj++)
                    {
                        double sum = cRow[jj];
                        for (int k = 0; k < kc; k++)
                            sum += aPanel[k * MR + ii] * bPanel[k * NR_D + jj];
                        cRow[jj] = sum;
                    }
                }
            }
        }
    }
}
