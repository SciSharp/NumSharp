using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// High-performance SIMD matrix multiplication with cache blocking and panel packing.
    /// Single-threaded implementation achieving ~20 GFLOPS on modern CPUs.
    ///
    /// Key optimizations:
    /// - GEBP (General Block Panel) algorithm with cache blocking
    /// - Full panel packing: A as [kc][MR] panels, B as [kc][NR] panels
    /// - 8x16 micro-kernel with 16 Vector256 accumulators
    /// - FMA (Fused Multiply-Add) for 2x FLOP throughput
    /// - 4x k-loop unrolling for instruction-level parallelism
    ///
    /// Stride-aware variants (see SimdMatMul.Strided.cs / SimdMatMul.Double.cs)
    /// accept (stride0, stride1) for each operand so transposed / sliced NDArray
    /// views can be matmul'd without materializing contiguous copies.
    /// </summary>
    public static partial class SimdMatMul
    {
        // Cache blocking parameters tuned for typical L1=32KB, L2=256KB
        private const int MC = 64;   // Rows of A panel (fits in L2 with B panel)
        private const int KC = 256;  // K depth (A panel + B panel fit in L2)
        private const int MR = 8;    // Micro-kernel rows
        private const int NR = 16;   // Micro-kernel cols (2 vectors)

        // Threshold below which simple IKJ is faster (no blocking overhead)
        private const int BLOCKING_THRESHOLD = 128;

        /// <summary>
        /// Matrix multiply: C = A * B
        /// A is [M x K], B is [K x N], C is [M x N]
        /// All matrices must be row-major contiguous.
        /// </summary>
        /// <remarks>
        /// Supports long dimensions for arrays > 2B elements.
        /// Cache blocking (MC, KC, MR, NR) keeps inner loops within int range.
        /// Outer loops and index calculations use long arithmetic.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void MatMulFloat(float* A, float* B, float* C, long M, long N, long K)
        {

            // Zero output using UnmanagedSpan for long indexing support
            new UnmanagedSpan<float>(C, M * N).Clear();

            // Small matrices: use simple IKJ loop (blocking overhead not worth it)
            if (M <= BLOCKING_THRESHOLD && N <= BLOCKING_THRESHOLD && K <= BLOCKING_THRESHOLD)
            {
                MatMulFloatSimple(A, B, C, M, N, K);
                return;
            }

            // Large matrices: cache-blocked GEBP algorithm with full panel packing
            MatMulFloatBlocked(A, B, C, M, N, K);
        }

        /// <summary>
        /// Simple IKJ loop for small matrices.
        /// Outer loops use long to support dimensions > int.MaxValue.
        /// Inner SIMD loop uses long for j to support large N dimension.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulFloatSimple(float* A, float* B, float* C, long M, long N, long K)
        {
            for (long i = 0; i < M; i++)
            {
                float* cRow = C + i * N;
                float* aRow = A + i * K;

                for (long k = 0; k < K; k++)
                {
                    float aik = aRow[k];
                    var aikVec = Vector256.Create(aik);
                    float* bRow = B + k * N;

                    long j = 0;
                    // SIMD loop
                    for (; j <= N - 8; j += 8)
                    {
                        var cVec = Vector256.Load(cRow + j);
                        var bVec = Vector256.Load(bRow + j);
                        cVec = Fma.IsSupported
                            ? Fma.MultiplyAdd(aikVec, bVec, cVec)
                            : Vector256.Add(cVec, Vector256.Multiply(aikVec, bVec));
                        Vector256.Store(cVec, cRow + j);
                    }
                    // Scalar tail
                    for (; j < N; j++)
                        cRow[j] += aik * bRow[j];
                }
            }
        }

        /// <summary>
        /// Cache-blocked GEBP algorithm with full panel packing.
        /// Both A and B are packed into micro-kernel-friendly layouts:
        /// - A: [kc][MR] panels - MR rows interleaved per k value
        /// - B: [kc][NR] panels - NR columns contiguous per k value
        /// Outer loops use long to support dimensions > int.MaxValue.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulFloatBlocked(float* A, float* B, float* C, long M, long N, long K)
        {
            int numMPanels = (MC + MR - 1) / MR;
            long numNPanels = (N + NR - 1) / NR;

            // Allocate packing buffers with 64-byte alignment for cache line efficiency
            // Pack A as MR-row panels: [numMPanels][kc][MR]
            float* packA = (float*)NativeMemory.AlignedAlloc((nuint)(MC * KC * sizeof(float)), 64);
            // Pack B as NR-column panels: [numNPanels][kc][NR]
            float* packB = (float*)NativeMemory.AlignedAlloc((nuint)(numNPanels * KC * NR * sizeof(float)), 64);

            try
            {
                // Loop over K blocks (outermost for B panel reuse)
                for (long k0 = 0; k0 < K; k0 += KC)
                {
                    int kc = (int)Math.Min(KC, K - k0);

                    // Pack B into NR-column panels: each panel has kc rows of NR contiguous elements
                    PackBPanels(B, packB, N, k0, kc);

                    // Loop over M blocks
                    for (long i0 = 0; i0 < M; i0 += MC)
                    {
                        int mc = (int)Math.Min(MC, M - i0);

                        // Pack A into MR-row panels: each panel has kc columns with MR interleaved rows
                        PackAPanels(A, packA, K, i0, k0, mc, kc);

                        // Process micro-kernels with packed panels
                        for (int ip = 0; ip < mc; ip += MR)
                        {
                            int mr = Math.Min(MR, mc - ip);
                            float* aPanel = packA + (ip / MR) * kc * MR;

                            for (long jp = 0; jp < N; jp += NR)
                            {
                                int nr = (int)Math.Min(NR, N - jp);
                                float* bPanel = packB + (jp / NR) * kc * NR;

                                if (mr == MR && nr == NR)
                                {
                                    // Full 8x16 micro-kernel with panel access
                                    Microkernel8x16Packed(aPanel, bPanel, C, N, i0 + ip, jp, kc);
                                }
                                else
                                {
                                    // Edge case: partial micro-kernel
                                    MicrokernelGenericPacked(aPanel, bPanel, C, N, i0 + ip, jp, kc, mr, nr);
                                }
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

        /// <summary>
        /// Pack A into MR-row panels with interleaved layout.
        /// Layout: for each MR-row panel, store [k0][row0..row7], [k1][row0..row7], ...
        /// This gives contiguous access pattern: aPanel[k * MR + row]
        /// Uses long for i0, k0, lda to support large matrices.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void PackAPanels(float* A, float* packA, long lda, long i0, long k0, int mc, int kc)
        {
            for (int ip = 0; ip < mc; ip += MR)
            {
                int mr = Math.Min(MR, mc - ip);
                float* aPanel = packA + (ip / MR) * kc * MR;

                if (mr == MR)
                {
                    // Full panel - interleave 8 rows per k value
                    for (int k = 0; k < kc; k++)
                    {
                        float* dst = aPanel + k * MR;
                        // Use long arithmetic for index calculation
                        dst[0] = A[(i0 + ip + 0) * lda + k0 + k];
                        dst[1] = A[(i0 + ip + 1) * lda + k0 + k];
                        dst[2] = A[(i0 + ip + 2) * lda + k0 + k];
                        dst[3] = A[(i0 + ip + 3) * lda + k0 + k];
                        dst[4] = A[(i0 + ip + 4) * lda + k0 + k];
                        dst[5] = A[(i0 + ip + 5) * lda + k0 + k];
                        dst[6] = A[(i0 + ip + 6) * lda + k0 + k];
                        dst[7] = A[(i0 + ip + 7) * lda + k0 + k];
                    }
                }
                else
                {
                    // Partial panel - zero-pad
                    for (int k = 0; k < kc; k++)
                    {
                        float* dst = aPanel + k * MR;
                        for (int ii = 0; ii < MR; ii++)
                            dst[ii] = ii < mr ? A[(i0 + ip + ii) * lda + k0 + k] : 0f;
                    }
                }
            }
        }

        /// <summary>
        /// Pack B into NR-column panels with contiguous layout.
        /// Layout: for each NR-column panel, store [k0][col0..col15], [k1][col0..col15], ...
        /// This gives contiguous access pattern: bPanel[k * NR + col]
        /// Uses long for n, k0, ldb to support large matrices.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void PackBPanels(float* B, float* packB, long ldb, long k0, int kc)
        {
            long n = ldb; // N == ldb for row-major B
            for (long jp = 0; jp < n; jp += NR)
            {
                int nr = (int)Math.Min(NR, n - jp);
                float* bPanel = packB + (jp / NR) * kc * NR;

                if (nr == NR)
                {
                    // Full panel - vectorized copy of 16 contiguous floats per k
                    for (int k = 0; k < kc; k++)
                    {
                        // Use long arithmetic for index calculation
                        float* src = B + (k0 + k) * ldb + jp;
                        float* dst = bPanel + k * NR;
                        Vector256.Store(Vector256.Load(src), dst);
                        Vector256.Store(Vector256.Load(src + 8), dst + 8);
                    }
                }
                else
                {
                    // Partial panel - zero-pad
                    for (int k = 0; k < kc; k++)
                    {
                        float* src = B + (k0 + k) * ldb + jp;
                        float* dst = bPanel + k * NR;
                        for (int jj = 0; jj < NR; jj++)
                            dst[jj] = jj < nr ? src[jj] : 0f;
                    }
                }
            }
        }

        /// <summary>
        /// 8x16 micro-kernel with full panel packing and k-loop unrolling.
        /// Both A and B are in packed panel format for optimal cache access:
        /// - A panel: aPanel[k * MR + row] - 8 floats contiguous per k
        /// - B panel: bPanel[k * NR + col] - 16 floats contiguous per k
        /// Uses long for i, j, ldc to support large matrices.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Microkernel8x16Packed(float* aPanel, float* bPanel, float* C, long ldc, long i, long j, int kc)
        {
            // Load C accumulators (8 rows x 2 vectors = 16 accumulators)
            // Use long arithmetic for index calculation
            var c00 = Vector256.Load(C + (i + 0) * ldc + j);
            var c01 = Vector256.Load(C + (i + 0) * ldc + j + 8);
            var c10 = Vector256.Load(C + (i + 1) * ldc + j);
            var c11 = Vector256.Load(C + (i + 1) * ldc + j + 8);
            var c20 = Vector256.Load(C + (i + 2) * ldc + j);
            var c21 = Vector256.Load(C + (i + 2) * ldc + j + 8);
            var c30 = Vector256.Load(C + (i + 3) * ldc + j);
            var c31 = Vector256.Load(C + (i + 3) * ldc + j + 8);
            var c40 = Vector256.Load(C + (i + 4) * ldc + j);
            var c41 = Vector256.Load(C + (i + 4) * ldc + j + 8);
            var c50 = Vector256.Load(C + (i + 5) * ldc + j);
            var c51 = Vector256.Load(C + (i + 5) * ldc + j + 8);
            var c60 = Vector256.Load(C + (i + 6) * ldc + j);
            var c61 = Vector256.Load(C + (i + 6) * ldc + j + 8);
            var c70 = Vector256.Load(C + (i + 7) * ldc + j);
            var c71 = Vector256.Load(C + (i + 7) * ldc + j + 8);

            // K-loop with 4x unrolling for instruction-level parallelism
            int k = 0;
            if (Fma.IsSupported)
            {
                for (; k <= kc - 4; k += 4)
                {
                    // Load B panel rows (contiguous: bPanel[k*16..k*16+15])
                    var b00 = Vector256.Load(bPanel + (k + 0) * NR);
                    var b01 = Vector256.Load(bPanel + (k + 0) * NR + 8);
                    var b10 = Vector256.Load(bPanel + (k + 1) * NR);
                    var b11 = Vector256.Load(bPanel + (k + 1) * NR + 8);
                    var b20 = Vector256.Load(bPanel + (k + 2) * NR);
                    var b21 = Vector256.Load(bPanel + (k + 2) * NR + 8);
                    var b30 = Vector256.Load(bPanel + (k + 3) * NR);
                    var b31 = Vector256.Load(bPanel + (k + 3) * NR + 8);

                    // A panel pointers (contiguous: aPanel[k*8..k*8+7])
                    float* ak0 = aPanel + (k + 0) * MR;
                    float* ak1 = aPanel + (k + 1) * MR;
                    float* ak2 = aPanel + (k + 2) * MR;
                    float* ak3 = aPanel + (k + 3) * MR;

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
                var b0 = Vector256.Load(bPanel + k * NR);
                var b1 = Vector256.Load(bPanel + k * NR + 8);
                float* ak = aPanel + k * MR;

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
            Vector256.Store(c01, C + (i + 0) * ldc + j + 8);
            Vector256.Store(c10, C + (i + 1) * ldc + j);
            Vector256.Store(c11, C + (i + 1) * ldc + j + 8);
            Vector256.Store(c20, C + (i + 2) * ldc + j);
            Vector256.Store(c21, C + (i + 2) * ldc + j + 8);
            Vector256.Store(c30, C + (i + 3) * ldc + j);
            Vector256.Store(c31, C + (i + 3) * ldc + j + 8);
            Vector256.Store(c40, C + (i + 4) * ldc + j);
            Vector256.Store(c41, C + (i + 4) * ldc + j + 8);
            Vector256.Store(c50, C + (i + 5) * ldc + j);
            Vector256.Store(c51, C + (i + 5) * ldc + j + 8);
            Vector256.Store(c60, C + (i + 6) * ldc + j);
            Vector256.Store(c61, C + (i + 6) * ldc + j + 8);
            Vector256.Store(c70, C + (i + 7) * ldc + j);
            Vector256.Store(c71, C + (i + 7) * ldc + j + 8);
        }

        /// <summary>
        /// Generic micro-kernel for edge cases (partial rows/cols) with packed panels.
        /// Uses long for i, j, ldc to support large matrices.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MicrokernelGenericPacked(float* aPanel, float* bPanel, float* C, long ldc, long i, long j, int kc, int mr, int nr)
        {
            for (int ii = 0; ii < mr; ii++)
            {
                // Use long arithmetic for index calculation
                float* cRow = C + (i + ii) * ldc + j;

                // Use SIMD for full vectors, scalar for remainder
                if (nr >= 8 && Fma.IsSupported)
                {
                    var c0 = Vector256.Load(cRow);
                    var c1 = nr >= 16 ? Vector256.Load(cRow + 8) : Vector256<float>.Zero;

                    for (int k = 0; k < kc; k++)
                    {
                        var aVal = Vector256.Create(aPanel[k * MR + ii]);
                        var b0 = Vector256.Load(bPanel + k * NR);
                        c0 = Fma.MultiplyAdd(aVal, b0, c0);
                        if (nr >= 16)
                        {
                            var b1 = Vector256.Load(bPanel + k * NR + 8);
                            c1 = Fma.MultiplyAdd(aVal, b1, c1);
                        }
                    }

                    Vector256.Store(c0, cRow);
                    if (nr >= 16)
                        Vector256.Store(c1, cRow + 8);
                    else
                    {
                        // Handle 8-15 columns: scalar for remainder
                        for (int jj = 8; jj < nr; jj++)
                        {
                            float sum = cRow[jj];
                            for (int k = 0; k < kc; k++)
                                sum += aPanel[k * MR + ii] * bPanel[k * NR + jj];
                            cRow[jj] = sum;
                        }
                    }
                }
                else
                {
                    // Full scalar fallback
                    for (int jj = 0; jj < nr; jj++)
                    {
                        float sum = cRow[jj];
                        for (int k = 0; k < kc; k++)
                            sum += aPanel[k * MR + ii] * bPanel[k * NR + jj];
                        cRow[jj] = sum;
                    }
                }
            }
        }
    }
}
