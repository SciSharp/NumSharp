using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// High-performance SIMD matrix multiplication with cache blocking.
    /// Single-threaded implementation optimized for L1/L2 cache.
    /// </summary>
    public static class SimdMatMul
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
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void MatMulFloat(float* A, float* B, float* C, int M, int N, int K)
        {
            // Zero output
            new Span<float>(C, M * N).Clear();

            // Small matrices: use simple IKJ loop (blocking overhead not worth it)
            if (M <= BLOCKING_THRESHOLD && N <= BLOCKING_THRESHOLD && K <= BLOCKING_THRESHOLD)
            {
                MatMulFloatSimple(A, B, C, M, N, K);
                return;
            }

            // Large matrices: cache-blocked GEBP algorithm
            MatMulFloatBlocked(A, B, C, M, N, K);
        }

        /// <summary>
        /// Simple IKJ loop for small matrices.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulFloatSimple(float* A, float* B, float* C, int M, int N, int K)
        {
            for (int i = 0; i < M; i++)
            {
                float* cRow = C + i * N;
                float* aRow = A + i * K;

                for (int k = 0; k < K; k++)
                {
                    float aik = aRow[k];
                    var aikVec = Vector256.Create(aik);
                    float* bRow = B + k * N;

                    int j = 0;
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
        /// Cache-blocked GEBP algorithm for large matrices.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulFloatBlocked(float* A, float* B, float* C, int M, int N, int K)
        {
            // Allocate packing buffers on stack for small blocks, heap for large
            int packASize = MC * KC;
            int packBSize = KC * N;

            // Always use heap allocation for packing buffers (stackalloc has size limits)
            float* packA = (float*)NativeMemory.AlignedAlloc((nuint)(packASize * sizeof(float)), 64);
            float* packB = (float*)NativeMemory.AlignedAlloc((nuint)(packBSize * sizeof(float)), 64);

            try
            {
                // Loop over K blocks (outermost for B panel reuse)
                for (int k0 = 0; k0 < K; k0 += KC)
                {
                    int kc = Math.Min(KC, K - k0);

                    // Pack B panel [kc x N] - copy rows for sequential access
                    PackB(B, packB, N, k0, kc, N);

                    // Loop over M blocks
                    for (int i0 = 0; i0 < M; i0 += MC)
                    {
                        int mc = Math.Min(MC, M - i0);

                        // Pack A panel [mc x kc]
                        PackA(A, packA, K, i0, k0, mc, kc);

                        // Process micro-kernels
                        for (int i = 0; i < mc; i += MR)
                        {
                            int mr = Math.Min(MR, mc - i);

                            for (int j = 0; j < N; j += NR)
                            {
                                int nr = Math.Min(NR, N - j);

                                if (mr == MR && nr == NR)
                                {
                                    // Full 8x16 micro-kernel
                                    Microkernel8x16(packA + i * kc, packB, C, N, i0 + i, j, kc, N);
                                }
                                else
                                {
                                    // Edge case: partial micro-kernel
                                    MicrokernelGeneric(packA + i * kc, packB, C, N, i0 + i, j, kc, N, mr, nr);
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
        /// Pack A panel [mc x kc] into contiguous row-major memory.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void PackA(float* A, float* pack, int lda, int i0, int k0, int mc, int kc)
        {
            for (int i = 0; i < mc; i++)
            {
                float* src = A + (i0 + i) * lda + k0;
                float* dst = pack + i * kc;

                // Use vectorized copy for better performance
                int k = 0;
                for (; k <= kc - 8; k += 8)
                {
                    Vector256.Store(Vector256.Load(src + k), dst + k);
                }
                for (; k < kc; k++)
                {
                    dst[k] = src[k];
                }
            }
        }

        /// <summary>
        /// Pack B panel - just copy since B is already row-major.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void PackB(float* B, float* pack, int ldb, int k0, int kc, int n)
        {
            for (int k = 0; k < kc; k++)
            {
                float* src = B + (k0 + k) * ldb;
                float* dst = pack + k * n;
                Buffer.MemoryCopy(src, dst, n * sizeof(float), n * sizeof(float));
            }
        }

        /// <summary>
        /// 8x16 micro-kernel with k-loop unrolling.
        /// Processes 8 rows x 16 cols (2 vectors) using 16 accumulators.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Microkernel8x16(float* packA, float* packB, float* C, int ldc, int i, int j, int kc, int n)
        {
            // Load C accumulators (8 rows x 2 vectors = 16 accumulators)
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

            float* pA0 = packA + 0 * kc;
            float* pA1 = packA + 1 * kc;
            float* pA2 = packA + 2 * kc;
            float* pA3 = packA + 3 * kc;
            float* pA4 = packA + 4 * kc;
            float* pA5 = packA + 5 * kc;
            float* pA6 = packA + 6 * kc;
            float* pA7 = packA + 7 * kc;

            // K-loop with unrolling by 4
            int k = 0;
            if (Fma.IsSupported)
            {
                for (; k <= kc - 4; k += 4)
                {
                    // Load 4 B rows (8 vectors total)
                    var b00 = Vector256.Load(packB + (k + 0) * n + j);
                    var b01 = Vector256.Load(packB + (k + 0) * n + j + 8);
                    var b10 = Vector256.Load(packB + (k + 1) * n + j);
                    var b11 = Vector256.Load(packB + (k + 1) * n + j + 8);
                    var b20 = Vector256.Load(packB + (k + 2) * n + j);
                    var b21 = Vector256.Load(packB + (k + 2) * n + j + 8);
                    var b30 = Vector256.Load(packB + (k + 3) * n + j);
                    var b31 = Vector256.Load(packB + (k + 3) * n + j + 8);

                    // Row 0
                    c00 = Fma.MultiplyAdd(Vector256.Create(pA0[k + 0]), b00, c00);
                    c01 = Fma.MultiplyAdd(Vector256.Create(pA0[k + 0]), b01, c01);
                    c00 = Fma.MultiplyAdd(Vector256.Create(pA0[k + 1]), b10, c00);
                    c01 = Fma.MultiplyAdd(Vector256.Create(pA0[k + 1]), b11, c01);
                    c00 = Fma.MultiplyAdd(Vector256.Create(pA0[k + 2]), b20, c00);
                    c01 = Fma.MultiplyAdd(Vector256.Create(pA0[k + 2]), b21, c01);
                    c00 = Fma.MultiplyAdd(Vector256.Create(pA0[k + 3]), b30, c00);
                    c01 = Fma.MultiplyAdd(Vector256.Create(pA0[k + 3]), b31, c01);

                    // Row 1
                    c10 = Fma.MultiplyAdd(Vector256.Create(pA1[k + 0]), b00, c10);
                    c11 = Fma.MultiplyAdd(Vector256.Create(pA1[k + 0]), b01, c11);
                    c10 = Fma.MultiplyAdd(Vector256.Create(pA1[k + 1]), b10, c10);
                    c11 = Fma.MultiplyAdd(Vector256.Create(pA1[k + 1]), b11, c11);
                    c10 = Fma.MultiplyAdd(Vector256.Create(pA1[k + 2]), b20, c10);
                    c11 = Fma.MultiplyAdd(Vector256.Create(pA1[k + 2]), b21, c11);
                    c10 = Fma.MultiplyAdd(Vector256.Create(pA1[k + 3]), b30, c10);
                    c11 = Fma.MultiplyAdd(Vector256.Create(pA1[k + 3]), b31, c11);

                    // Row 2
                    c20 = Fma.MultiplyAdd(Vector256.Create(pA2[k + 0]), b00, c20);
                    c21 = Fma.MultiplyAdd(Vector256.Create(pA2[k + 0]), b01, c21);
                    c20 = Fma.MultiplyAdd(Vector256.Create(pA2[k + 1]), b10, c20);
                    c21 = Fma.MultiplyAdd(Vector256.Create(pA2[k + 1]), b11, c21);
                    c20 = Fma.MultiplyAdd(Vector256.Create(pA2[k + 2]), b20, c20);
                    c21 = Fma.MultiplyAdd(Vector256.Create(pA2[k + 2]), b21, c21);
                    c20 = Fma.MultiplyAdd(Vector256.Create(pA2[k + 3]), b30, c20);
                    c21 = Fma.MultiplyAdd(Vector256.Create(pA2[k + 3]), b31, c21);

                    // Row 3
                    c30 = Fma.MultiplyAdd(Vector256.Create(pA3[k + 0]), b00, c30);
                    c31 = Fma.MultiplyAdd(Vector256.Create(pA3[k + 0]), b01, c31);
                    c30 = Fma.MultiplyAdd(Vector256.Create(pA3[k + 1]), b10, c30);
                    c31 = Fma.MultiplyAdd(Vector256.Create(pA3[k + 1]), b11, c31);
                    c30 = Fma.MultiplyAdd(Vector256.Create(pA3[k + 2]), b20, c30);
                    c31 = Fma.MultiplyAdd(Vector256.Create(pA3[k + 2]), b21, c31);
                    c30 = Fma.MultiplyAdd(Vector256.Create(pA3[k + 3]), b30, c30);
                    c31 = Fma.MultiplyAdd(Vector256.Create(pA3[k + 3]), b31, c31);

                    // Row 4
                    c40 = Fma.MultiplyAdd(Vector256.Create(pA4[k + 0]), b00, c40);
                    c41 = Fma.MultiplyAdd(Vector256.Create(pA4[k + 0]), b01, c41);
                    c40 = Fma.MultiplyAdd(Vector256.Create(pA4[k + 1]), b10, c40);
                    c41 = Fma.MultiplyAdd(Vector256.Create(pA4[k + 1]), b11, c41);
                    c40 = Fma.MultiplyAdd(Vector256.Create(pA4[k + 2]), b20, c40);
                    c41 = Fma.MultiplyAdd(Vector256.Create(pA4[k + 2]), b21, c41);
                    c40 = Fma.MultiplyAdd(Vector256.Create(pA4[k + 3]), b30, c40);
                    c41 = Fma.MultiplyAdd(Vector256.Create(pA4[k + 3]), b31, c41);

                    // Row 5
                    c50 = Fma.MultiplyAdd(Vector256.Create(pA5[k + 0]), b00, c50);
                    c51 = Fma.MultiplyAdd(Vector256.Create(pA5[k + 0]), b01, c51);
                    c50 = Fma.MultiplyAdd(Vector256.Create(pA5[k + 1]), b10, c50);
                    c51 = Fma.MultiplyAdd(Vector256.Create(pA5[k + 1]), b11, c51);
                    c50 = Fma.MultiplyAdd(Vector256.Create(pA5[k + 2]), b20, c50);
                    c51 = Fma.MultiplyAdd(Vector256.Create(pA5[k + 2]), b21, c51);
                    c50 = Fma.MultiplyAdd(Vector256.Create(pA5[k + 3]), b30, c50);
                    c51 = Fma.MultiplyAdd(Vector256.Create(pA5[k + 3]), b31, c51);

                    // Row 6
                    c60 = Fma.MultiplyAdd(Vector256.Create(pA6[k + 0]), b00, c60);
                    c61 = Fma.MultiplyAdd(Vector256.Create(pA6[k + 0]), b01, c61);
                    c60 = Fma.MultiplyAdd(Vector256.Create(pA6[k + 1]), b10, c60);
                    c61 = Fma.MultiplyAdd(Vector256.Create(pA6[k + 1]), b11, c61);
                    c60 = Fma.MultiplyAdd(Vector256.Create(pA6[k + 2]), b20, c60);
                    c61 = Fma.MultiplyAdd(Vector256.Create(pA6[k + 2]), b21, c61);
                    c60 = Fma.MultiplyAdd(Vector256.Create(pA6[k + 3]), b30, c60);
                    c61 = Fma.MultiplyAdd(Vector256.Create(pA6[k + 3]), b31, c61);

                    // Row 7
                    c70 = Fma.MultiplyAdd(Vector256.Create(pA7[k + 0]), b00, c70);
                    c71 = Fma.MultiplyAdd(Vector256.Create(pA7[k + 0]), b01, c71);
                    c70 = Fma.MultiplyAdd(Vector256.Create(pA7[k + 1]), b10, c70);
                    c71 = Fma.MultiplyAdd(Vector256.Create(pA7[k + 1]), b11, c71);
                    c70 = Fma.MultiplyAdd(Vector256.Create(pA7[k + 2]), b20, c70);
                    c71 = Fma.MultiplyAdd(Vector256.Create(pA7[k + 2]), b21, c71);
                    c70 = Fma.MultiplyAdd(Vector256.Create(pA7[k + 3]), b30, c70);
                    c71 = Fma.MultiplyAdd(Vector256.Create(pA7[k + 3]), b31, c71);
                }
            }

            // Remainder k iterations
            for (; k < kc; k++)
            {
                var b0 = Vector256.Load(packB + k * n + j);
                var b1 = Vector256.Load(packB + k * n + j + 8);

                if (Fma.IsSupported)
                {
                    c00 = Fma.MultiplyAdd(Vector256.Create(pA0[k]), b0, c00);
                    c01 = Fma.MultiplyAdd(Vector256.Create(pA0[k]), b1, c01);
                    c10 = Fma.MultiplyAdd(Vector256.Create(pA1[k]), b0, c10);
                    c11 = Fma.MultiplyAdd(Vector256.Create(pA1[k]), b1, c11);
                    c20 = Fma.MultiplyAdd(Vector256.Create(pA2[k]), b0, c20);
                    c21 = Fma.MultiplyAdd(Vector256.Create(pA2[k]), b1, c21);
                    c30 = Fma.MultiplyAdd(Vector256.Create(pA3[k]), b0, c30);
                    c31 = Fma.MultiplyAdd(Vector256.Create(pA3[k]), b1, c31);
                    c40 = Fma.MultiplyAdd(Vector256.Create(pA4[k]), b0, c40);
                    c41 = Fma.MultiplyAdd(Vector256.Create(pA4[k]), b1, c41);
                    c50 = Fma.MultiplyAdd(Vector256.Create(pA5[k]), b0, c50);
                    c51 = Fma.MultiplyAdd(Vector256.Create(pA5[k]), b1, c51);
                    c60 = Fma.MultiplyAdd(Vector256.Create(pA6[k]), b0, c60);
                    c61 = Fma.MultiplyAdd(Vector256.Create(pA6[k]), b1, c61);
                    c70 = Fma.MultiplyAdd(Vector256.Create(pA7[k]), b0, c70);
                    c71 = Fma.MultiplyAdd(Vector256.Create(pA7[k]), b1, c71);
                }
                else
                {
                    c00 = Vector256.Add(c00, Vector256.Multiply(Vector256.Create(pA0[k]), b0));
                    c01 = Vector256.Add(c01, Vector256.Multiply(Vector256.Create(pA0[k]), b1));
                    c10 = Vector256.Add(c10, Vector256.Multiply(Vector256.Create(pA1[k]), b0));
                    c11 = Vector256.Add(c11, Vector256.Multiply(Vector256.Create(pA1[k]), b1));
                    c20 = Vector256.Add(c20, Vector256.Multiply(Vector256.Create(pA2[k]), b0));
                    c21 = Vector256.Add(c21, Vector256.Multiply(Vector256.Create(pA2[k]), b1));
                    c30 = Vector256.Add(c30, Vector256.Multiply(Vector256.Create(pA3[k]), b0));
                    c31 = Vector256.Add(c31, Vector256.Multiply(Vector256.Create(pA3[k]), b1));
                    c40 = Vector256.Add(c40, Vector256.Multiply(Vector256.Create(pA4[k]), b0));
                    c41 = Vector256.Add(c41, Vector256.Multiply(Vector256.Create(pA4[k]), b1));
                    c50 = Vector256.Add(c50, Vector256.Multiply(Vector256.Create(pA5[k]), b0));
                    c51 = Vector256.Add(c51, Vector256.Multiply(Vector256.Create(pA5[k]), b1));
                    c60 = Vector256.Add(c60, Vector256.Multiply(Vector256.Create(pA6[k]), b0));
                    c61 = Vector256.Add(c61, Vector256.Multiply(Vector256.Create(pA6[k]), b1));
                    c70 = Vector256.Add(c70, Vector256.Multiply(Vector256.Create(pA7[k]), b0));
                    c71 = Vector256.Add(c71, Vector256.Multiply(Vector256.Create(pA7[k]), b1));
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
        /// Generic micro-kernel for edge cases (partial rows/cols).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MicrokernelGeneric(float* packA, float* packB, float* C, int ldc, int i, int j, int kc, int n, int mr, int nr)
        {
            for (int ii = 0; ii < mr; ii++)
            {
                float* pA = packA + ii * kc;
                float* cRow = C + (i + ii) * ldc + j;

                for (int jj = 0; jj < nr; jj++)
                {
                    float sum = cRow[jj];
                    for (int k = 0; k < kc; k++)
                    {
                        sum += pA[k] * packB[k * n + j + jj];
                    }
                    cRow[jj] = sum;
                }
            }
        }
    }
}
