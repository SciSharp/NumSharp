using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

// =============================================================================
// Stride-aware float GEMM (BLAS-style, replaces explicit TransA/TransB flags)
// =============================================================================
//
// BLIS-inspired GEBP (General Block Panel) with strided packing. The packing
// stage absorbs all stride variation — transposed / sliced views are copied
// into MR- and NR-packed micro-kernel panels. The micro-kernel itself reads
// only from the packed contiguous buffers, so it's stride-agnostic and the
// existing Microkernel8x16Packed / MicrokernelGenericPacked are reused.
//
// Fast paths in the packers:
//   PackA, aStride0 == 1  — transposed-contiguous A, 8-row SIMD load per k.
//   PackB, bStride1 == 1  — row-contiguous B, 16-col SIMD load per k (same
//                           as the original contiguous path).
//   PackB, bStride0 == 1  — transposed-contiguous B, K-long contiguous read
//                           per column, scalar scatter-store.
//
// Everything else falls through to scalar element access. Packing is
// O(M*K + K*N) while GEMM is O(M*N*K), so the ratio is 1/N + 1/M — for any
// matrix large enough to care about, packing is <3% of the total work.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class SimdMatMul
    {
        /// <summary>
        /// Stride-aware matrix multiply: C = A * B.
        /// A is logical (M, K) with strides (aStride0, aStride1) in elements.
        /// B is logical (K, N) with strides (bStride0, bStride1) in elements.
        /// C is written as M×N row-major contiguous (ldc = N).
        ///
        /// Passing (aStride0=K, aStride1=1, bStride0=N, bStride1=1) reproduces
        /// the contiguous-input behavior of <see cref="MatMulFloat(float*,float*,float*,long,long,long)"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void MatMulFloat(
            float* A, long aStride0, long aStride1,
            float* B, long bStride0, long bStride1,
            float* C,
            long M, long N, long K)
        {
            // Zero output — kernels accumulate into it.
            new UnmanagedSpan<float>(C, M * N).Clear();

            if (M == 0 || N == 0 || K == 0)
                return;

            // Contiguous fast path: route through the already-validated
            // MatMulFloat(A,B,C,M,N,K) so we don't regress any benchmarks.
            if (aStride0 == K && aStride1 == 1 && bStride0 == N && bStride1 == 1)
            {
                MatMulFloatContiguousCore(A, B, C, M, N, K);
                return;
            }

            if (M <= BLOCKING_THRESHOLD && N <= BLOCKING_THRESHOLD && K <= BLOCKING_THRESHOLD)
            {
                MatMulFloatSimpleStrided(A, aStride0, aStride1, B, bStride0, bStride1, C, M, N, K);
                return;
            }

            MatMulFloatBlockedStrided(A, aStride0, aStride1, B, bStride0, bStride1, C, M, N, K);
        }

        /// <summary>
        /// Shared body of the contiguous fast path — dispatches simple vs
        /// blocked without re-zeroing C (the stride-aware entry already did).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void MatMulFloatContiguousCore(float* A, float* B, float* C, long M, long N, long K)
        {
            if (M <= BLOCKING_THRESHOLD && N <= BLOCKING_THRESHOLD && K <= BLOCKING_THRESHOLD)
                MatMulFloatSimple(A, B, C, M, N, K);
            else
                MatMulFloatBlocked(A, B, C, M, N, K);
        }

        // =====================================================================
        // Simple IKJ path (small matrices)
        // =====================================================================

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulFloatSimpleStrided(
            float* A, long aStride0, long aStride1,
            float* B, long bStride0, long bStride1,
            float* C, long M, long N, long K)
        {
            // Dispatch on B's inner stride — that's what controls whether
            // the inner SIMD loop is valid (it needs 8 consecutive floats).
            if (bStride1 == 1)
            {
                for (long i = 0; i < M; i++)
                {
                    float* cRow = C + i * N;
                    long aRowBase = i * aStride0;

                    for (long k = 0; k < K; k++)
                    {
                        float aik = A[aRowBase + k * aStride1];
                        var aikVec = Vector256.Create(aik);
                        float* bRow = B + k * bStride0;

                        long j = 0;
                        for (; j <= N - 8; j += 8)
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
                    float* cRow = C + i * N;
                    long aRowBase = i * aStride0;

                    for (long k = 0; k < K; k++)
                    {
                        float aik = A[aRowBase + k * aStride1];
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
        private static unsafe void MatMulFloatBlockedStrided(
            float* A, long aStride0, long aStride1,
            float* B, long bStride0, long bStride1,
            float* C, long M, long N, long K)
        {
            long numNPanels = (N + NR - 1) / NR;

            float* packA = (float*)NativeMemory.AlignedAlloc((nuint)(MC * KC * sizeof(float)), 64);
            float* packB = (float*)NativeMemory.AlignedAlloc((nuint)(numNPanels * KC * NR * sizeof(float)), 64);

            try
            {
                for (long k0 = 0; k0 < K; k0 += KC)
                {
                    int kc = (int)Math.Min(KC, K - k0);

                    PackBPanelsStrided(B, bStride0, bStride1, packB, N, k0, kc);

                    for (long i0 = 0; i0 < M; i0 += MC)
                    {
                        int mc = (int)Math.Min(MC, M - i0);

                        PackAPanelsStrided(A, aStride0, aStride1, packA, i0, k0, mc, kc);

                        for (int ip = 0; ip < mc; ip += MR)
                        {
                            int mr = Math.Min(MR, mc - ip);
                            float* aPanel = packA + (ip / MR) * kc * MR;

                            for (long jp = 0; jp < N; jp += NR)
                            {
                                int nr = (int)Math.Min(NR, N - jp);
                                float* bPanel = packB + (jp / NR) * kc * NR;

                                if (mr == MR && nr == NR)
                                    Microkernel8x16Packed(aPanel, bPanel, C, N, i0 + ip, jp, kc);
                                else
                                    MicrokernelGenericPacked(aPanel, bPanel, C, N, i0 + ip, jp, kc, mr, nr);
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
        /// interleaved panels. Layout matches PackAPanels:
        ///   aPanel[(ip/MR) * kc * MR + k * MR + row].
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void PackAPanelsStrided(
            float* A, long aStride0, long aStride1,
            float* packA, long i0, long k0, int mc, int kc)
        {
            for (int ip = 0; ip < mc; ip += MR)
            {
                int mr = Math.Min(MR, mc - ip);
                float* aPanel = packA + (ip / MR) * kc * MR;

                if (mr == MR)
                {
                    if (aStride0 == 1)
                    {
                        // Transposed-contiguous A: 8 consecutive logical rows
                        // sit at 8 consecutive memory addresses (per fixed k),
                        // because A[i, k] = A + i*1 + k*aStride1.
                        // One Vector256 load packs 8 rows.
                        for (int k = 0; k < kc; k++)
                        {
                            long srcOff = (i0 + ip) + (k0 + k) * aStride1;
                            Vector256.Store(Vector256.Load(A + srcOff), aPanel + k * MR);
                        }
                    }
                    else
                    {
                        for (int k = 0; k < kc; k++)
                        {
                            float* dst = aPanel + k * MR;
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
                        float* dst = aPanel + k * MR;
                        long kOff = (k0 + k) * aStride1;
                        for (int ii = 0; ii < MR; ii++)
                            dst[ii] = ii < mr ? A[(i0 + ip + ii) * aStride0 + kOff] : 0f;
                    }
                }
            }
        }

        /// <summary>
        /// Pack a K-slice of B (rows k0..k0+kc, all N columns) into NR-column
        /// panels. Layout matches PackBPanels:
        ///   bPanel[(jp/NR) * kc * NR + k * NR + col].
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void PackBPanelsStrided(
            float* B, long bStride0, long bStride1,
            float* packB, long N_total, long k0, int kc)
        {
            for (long jp = 0; jp < N_total; jp += NR)
            {
                int nr = (int)Math.Min(NR, N_total - jp);
                float* bPanel = packB + (jp / NR) * kc * NR;

                if (bStride1 == 1)
                {
                    // Row-contiguous B: 16 consecutive floats per k.
                    if (nr == NR)
                    {
                        for (int k = 0; k < kc; k++)
                        {
                            float* src = B + (k0 + k) * bStride0 + jp;
                            float* dst = bPanel + k * NR;
                            Vector256.Store(Vector256.Load(src), dst);
                            Vector256.Store(Vector256.Load(src + 8), dst + 8);
                        }
                    }
                    else
                    {
                        for (int k = 0; k < kc; k++)
                        {
                            float* src = B + (k0 + k) * bStride0 + jp;
                            float* dst = bPanel + k * NR;
                            for (int jj = 0; jj < NR; jj++)
                                dst[jj] = jj < nr ? src[jj] : 0f;
                        }
                    }
                }
                else if (bStride0 == 1)
                {
                    // Transposed-contiguous B: each logical column is a
                    // contiguous K-long run in memory at offset j*bStride1.
                    // Zero the panel first (handles partial-panel padding),
                    // then fill column-by-column with contiguous reads.
                    long panelFloats = (long)kc * NR;
                    new UnmanagedSpan<float>(bPanel, panelFloats).Clear();

                    for (int jj = 0; jj < nr; jj++)
                    {
                        float* colStart = B + (jp + jj) * bStride1 + k0;
                        // Scalar scatter — writes have stride NR which isn't
                        // SIMD-friendly on AVX2, but reads are contiguous.
                        for (int k = 0; k < kc; k++)
                            bPanel[k * NR + jj] = colStart[k];
                    }
                }
                else
                {
                    // Fully general: scalar reads using both strides.
                    for (int k = 0; k < kc; k++)
                    {
                        float* dst = bPanel + k * NR;
                        long kOff = (k0 + k) * bStride0;
                        for (int jj = 0; jj < NR; jj++)
                            dst[jj] = jj < nr ? B[kOff + (jp + jj) * bStride1] : 0f;
                    }
                }
            }
        }
    }
}
