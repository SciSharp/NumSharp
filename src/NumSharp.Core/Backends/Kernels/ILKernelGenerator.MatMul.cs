using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// ILKernelGenerator.MatMul - IL-based SIMD matrix multiplication
// =============================================================================
//
// ARCHITECTURE OVERVIEW
// ---------------------
// Implements cache-blocked matrix multiplication with SIMD inner loops.
// Target: ~40-60% of BLAS performance for float/double matrices.
//
// ALGORITHM
// ---------
// Uses blocked (tiled) matrix multiplication with ikj loop ordering:
//
//   for i_block in [0, M, BLOCK_M]:
//     for j_block in [0, N, BLOCK_N]:
//       for k_block in [0, K, BLOCK_K]:
//         // Compute block: C[i:i+BM, j:j+BN] += A[i:i+BM, k:k+BK] * B[k:k+BK, j:j+BN]
//         for i in [i_block, min(i_block+BM, M)]:
//           for k in [k_block, min(k_block+BK, K)]:
//             a_ik = A[i, k]
//             for j in [j_block, min(j_block+BN, N)]:  // SIMD vectorized
//               C[i, j] += a_ik * B[k, j]
//
// WHY IKJ ORDER?
// - A[i,k] is loaded once per (i,k) pair, reused across all j
// - B[k,j:j+vector_width] is accessed sequentially (cache-friendly)
// - C[i,j:j+vector_width] is accessed sequentially
//
// SIMD STRATEGY
// - Broadcast A[i,k] to all vector lanes
// - Load sequential B[k, j:j+8] (or j:j+4 for double)
// - FMA: C[i,j:j+8] += broadcast(A[i,k]) * B[k,j:j+8]
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Kernel delegate for 2D matrix multiplication: C = A * B
    /// A is [M x K], B is [K x N], C is [M x N]
    /// All matrices are row-major contiguous.
    /// </summary>
    public unsafe delegate void MatMul2DKernel<T>(
        T* a, T* b, T* c,
        int M, int N, int K) where T : unmanaged;

    /// <summary>
    /// Matrix multiplication kernels using IL generation with SIMD and cache blocking.
    /// </summary>
    public sealed partial class ILKernelGenerator
    {
        // Block sizes tuned for L1/L2 cache (32KB L1, 256KB L2 typical)
        // 64x64 float block = 16KB, fits comfortably in L1
        // 64x64 double block = 32KB, fits in L1
        private const int BLOCK_M = 64;
        private const int BLOCK_N = 64;
        private const int BLOCK_K = 64;

        // Small matrix threshold - below this, blocking overhead isn't worth it
        private const int SMALL_MATRIX_THRESHOLD = 32;

        // Threshold for enabling parallelization (roughly 256x256 matrix)
        private const int PARALLEL_THRESHOLD = 65536;

        /// <summary>
        /// Cache of generated MatMul kernels by type.
        /// </summary>
        private static readonly ConcurrentDictionary<Type, Delegate> _matmulKernelCache = new();

        #region Public API

        /// <summary>
        /// Get or generate a high-performance MatMul kernel for the given type.
        /// Returns null if the type is not supported for SIMD optimization.
        /// </summary>
        public static unsafe MatMul2DKernel<T>? GetMatMulKernel<T>() where T : unmanaged
        {
            if (!Enabled)
                return null;

            // Only support float and double for SIMD matmul
            if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
                return null;

            var key = typeof(T);

            if (_matmulKernelCache.TryGetValue(key, out var cached))
                return (MatMul2DKernel<T>)cached;

            var kernel = GenerateMatMulKernel<T>();
            if (kernel == null)
                return null;

            if (_matmulKernelCache.TryAdd(key, kernel))
                return kernel;

            return (MatMul2DKernel<T>)_matmulKernelCache[key];
        }

        /// <summary>
        /// Get a parallel MatMul kernel for large matrices.
        /// Uses row-based parallelization.
        /// </summary>
        public static unsafe MatMul2DKernel<T>? GetParallelMatMulKernel<T>() where T : unmanaged
        {
            if (!Enabled)
                return null;

            if (typeof(T) == typeof(float))
                return (MatMul2DKernel<T>)(Delegate)(MatMul2DKernel<float>)MatMulParallel_Float;
            if (typeof(T) == typeof(double))
                return (MatMul2DKernel<T>)(Delegate)(MatMul2DKernel<double>)MatMulParallel_Double;

            return null;
        }

        /// <summary>
        /// Clear the MatMul kernel cache.
        /// </summary>
        public static void ClearMatMulCache() => _matmulKernelCache.Clear();

        #endregion

        #region Kernel Generation

        /// <summary>
        /// Generate a blocked SIMD matrix multiplication kernel.
        /// </summary>
        private static unsafe MatMul2DKernel<T>? GenerateMatMulKernel<T>() where T : unmanaged
        {
            try
            {
                if (typeof(T) == typeof(float))
                    return (MatMul2DKernel<T>)(Delegate)(MatMul2DKernel<float>)MatMulBlockedSimd_Float;
                if (typeof(T) == typeof(double))
                    return (MatMul2DKernel<T>)(Delegate)(MatMul2DKernel<double>)MatMulBlockedSimd_Double;

                return null;
            }
            catch
            {
                return null;
            }
        }

        #endregion

        #region Float Implementation

        /// <summary>
        /// Blocked SIMD matrix multiplication for float.
        /// C[M,N] = A[M,K] * B[K,N]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulBlockedSimd_Float(float* a, float* b, float* c, int M, int N, int K)
        {
            // Zero out C first
            var cSize = M * N;
            for (int i = 0; i < cSize; i++)
                c[i] = 0;

            // For small matrices, use simple SIMD without blocking
            if (M <= SMALL_MATRIX_THRESHOLD && N <= SMALL_MATRIX_THRESHOLD && K <= SMALL_MATRIX_THRESHOLD)
            {
                MatMulSimdSmall_Float(a, b, c, M, N, K);
                return;
            }

            // Blocked matrix multiplication
            for (int i0 = 0; i0 < M; i0 += BLOCK_M)
            {
                int iEnd = Math.Min(i0 + BLOCK_M, M);

                for (int k0 = 0; k0 < K; k0 += BLOCK_K)
                {
                    int kEnd = Math.Min(k0 + BLOCK_K, K);

                    for (int j0 = 0; j0 < N; j0 += BLOCK_N)
                    {
                        int jEnd = Math.Min(j0 + BLOCK_N, N);

                        // Process block: C[i0:iEnd, j0:jEnd] += A[i0:iEnd, k0:kEnd] * B[k0:kEnd, j0:jEnd]
                        MatMulBlockSimd_Float(a, b, c, N, K, i0, iEnd, j0, jEnd, k0, kEnd);
                    }
                }
            }
        }

        /// <summary>
        /// Process a single block of the matrix multiplication with SIMD.
        /// Uses ikj loop order for optimal cache utilization.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulBlockSimd_Float(
            float* a, float* b, float* c,
            int N, int K,
            int i0, int iEnd, int j0, int jEnd, int k0, int kEnd)
        {
            int vectorWidth = Vector256<float>.Count; // 8 floats
            bool useFma = Fma.IsSupported;

            for (int i = i0; i < iEnd; i++)
            {
                float* cRow = c + i * N;
                float* aRow = a + i * K;

                for (int k = k0; k < kEnd; k++)
                {
                    float aik = aRow[k];
                    float* bRow = b + k * N;

                    // Broadcast A[i,k] to all vector lanes
                    var aVec = Vector256.Create(aik);

                    int j = j0;

                    // SIMD loop: process 8 elements at a time
                    int jSimdEnd = j0 + ((jEnd - j0) / vectorWidth) * vectorWidth;
                    for (; j < jSimdEnd; j += vectorWidth)
                    {
                        // Load C[i, j:j+8]
                        var cVec = Vector256.Load(cRow + j);

                        // Load B[k, j:j+8]
                        var bVec = Vector256.Load(bRow + j);

                        // C[i,j:j+8] += A[i,k] * B[k,j:j+8]
                        // Use FMA if available, otherwise mul+add
                        if (useFma)
                            cVec = Fma.MultiplyAdd(aVec, bVec, cVec);
                        else
                            cVec = cVec + aVec * bVec;

                        // Store result
                        Vector256.Store(cVec, cRow + j);
                    }

                    // Scalar tail
                    for (; j < jEnd; j++)
                    {
                        cRow[j] += aik * bRow[j];
                    }
                }
            }
        }

        /// <summary>
        /// Simple SIMD matmul for small matrices (no blocking overhead).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulSimdSmall_Float(float* a, float* b, float* c, int M, int N, int K)
        {
            int vectorWidth = Vector256<float>.Count; // 8 floats
            bool useFma = Fma.IsSupported;

            for (int i = 0; i < M; i++)
            {
                float* cRow = c + i * N;
                float* aRow = a + i * K;

                for (int k = 0; k < K; k++)
                {
                    float aik = aRow[k];
                    float* bRow = b + k * N;

                    var aVec = Vector256.Create(aik);

                    int j = 0;
                    int jSimdEnd = (N / vectorWidth) * vectorWidth;

                    for (; j < jSimdEnd; j += vectorWidth)
                    {
                        var cVec = Vector256.Load(cRow + j);
                        var bVec = Vector256.Load(bRow + j);

                        if (useFma)
                            cVec = Fma.MultiplyAdd(aVec, bVec, cVec);
                        else
                            cVec = cVec + aVec * bVec;

                        Vector256.Store(cVec, cRow + j);
                    }

                    for (; j < N; j++)
                    {
                        cRow[j] += aik * bRow[j];
                    }
                }
            }
        }

        #endregion

        #region Double Implementation

        /// <summary>
        /// Blocked SIMD matrix multiplication for double.
        /// C[M,N] = A[M,K] * B[K,N]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulBlockedSimd_Double(double* a, double* b, double* c, int M, int N, int K)
        {
            // Zero out C first
            var cSize = M * N;
            for (int i = 0; i < cSize; i++)
                c[i] = 0;

            // For small matrices, use simple SIMD without blocking
            if (M <= SMALL_MATRIX_THRESHOLD && N <= SMALL_MATRIX_THRESHOLD && K <= SMALL_MATRIX_THRESHOLD)
            {
                MatMulSimdSmall_Double(a, b, c, M, N, K);
                return;
            }

            // Blocked matrix multiplication
            for (int i0 = 0; i0 < M; i0 += BLOCK_M)
            {
                int iEnd = Math.Min(i0 + BLOCK_M, M);

                for (int k0 = 0; k0 < K; k0 += BLOCK_K)
                {
                    int kEnd = Math.Min(k0 + BLOCK_K, K);

                    for (int j0 = 0; j0 < N; j0 += BLOCK_N)
                    {
                        int jEnd = Math.Min(j0 + BLOCK_N, N);

                        // Process block
                        MatMulBlockSimd_Double(a, b, c, N, K, i0, iEnd, j0, jEnd, k0, kEnd);
                    }
                }
            }
        }

        /// <summary>
        /// Process a single block of the matrix multiplication with SIMD.
        /// Uses ikj loop order for optimal cache utilization.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulBlockSimd_Double(
            double* a, double* b, double* c,
            int N, int K,
            int i0, int iEnd, int j0, int jEnd, int k0, int kEnd)
        {
            int vectorWidth = Vector256<double>.Count; // 4 doubles
            bool useFma = Fma.IsSupported;

            for (int i = i0; i < iEnd; i++)
            {
                double* cRow = c + i * N;
                double* aRow = a + i * K;

                for (int k = k0; k < kEnd; k++)
                {
                    double aik = aRow[k];
                    double* bRow = b + k * N;

                    // Broadcast A[i,k] to all vector lanes
                    var aVec = Vector256.Create(aik);

                    int j = j0;

                    // SIMD loop: process 4 elements at a time
                    int jSimdEnd = j0 + ((jEnd - j0) / vectorWidth) * vectorWidth;
                    for (; j < jSimdEnd; j += vectorWidth)
                    {
                        // Load C[i, j:j+4]
                        var cVec = Vector256.Load(cRow + j);

                        // Load B[k, j:j+4]
                        var bVec = Vector256.Load(bRow + j);

                        // C[i,j:j+4] += A[i,k] * B[k,j:j+4]
                        if (useFma)
                            cVec = Fma.MultiplyAdd(aVec, bVec, cVec);
                        else
                            cVec = cVec + aVec * bVec;

                        // Store result
                        Vector256.Store(cVec, cRow + j);
                    }

                    // Scalar tail
                    for (; j < jEnd; j++)
                    {
                        cRow[j] += aik * bRow[j];
                    }
                }
            }
        }

        /// <summary>
        /// Simple SIMD matmul for small matrices (no blocking overhead).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulSimdSmall_Double(double* a, double* b, double* c, int M, int N, int K)
        {
            int vectorWidth = Vector256<double>.Count; // 4 doubles
            bool useFma = Fma.IsSupported;

            for (int i = 0; i < M; i++)
            {
                double* cRow = c + i * N;
                double* aRow = a + i * K;

                for (int k = 0; k < K; k++)
                {
                    double aik = aRow[k];
                    double* bRow = b + k * N;

                    var aVec = Vector256.Create(aik);

                    int j = 0;
                    int jSimdEnd = (N / vectorWidth) * vectorWidth;

                    for (; j < jSimdEnd; j += vectorWidth)
                    {
                        var cVec = Vector256.Load(cRow + j);
                        var bVec = Vector256.Load(bRow + j);

                        if (useFma)
                            cVec = Fma.MultiplyAdd(aVec, bVec, cVec);
                        else
                            cVec = cVec + aVec * bVec;

                        Vector256.Store(cVec, cRow + j);
                    }

                    for (; j < N; j++)
                    {
                        cRow[j] += aik * bRow[j];
                    }
                }
            }
        }

        #endregion

        #region Parallel Variants (Large Matrices)

        /// <summary>
        /// Parallel blocked SIMD matrix multiplication for float.
        /// Parallelizes over row blocks.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulParallel_Float(float* a, float* b, float* c, int M, int N, int K)
        {
            // Zero out C first
            var cSize = M * N;
            for (int i = 0; i < cSize; i++)
                c[i] = 0;

            // For small matrices, use sequential version
            if ((long)M * N * K < PARALLEL_THRESHOLD)
            {
                if (M <= SMALL_MATRIX_THRESHOLD && N <= SMALL_MATRIX_THRESHOLD && K <= SMALL_MATRIX_THRESHOLD)
                    MatMulSimdSmall_Float(a, b, c, M, N, K);
                else
                    MatMulBlockedSimd_Float(a, b, c, M, N, K);
                return;
            }

            // Parallel over row blocks
            int numRowBlocks = (M + BLOCK_M - 1) / BLOCK_M;

            System.Threading.Tasks.Parallel.For(0, numRowBlocks, i0Idx =>
            {
                int i0 = i0Idx * BLOCK_M;
                int iEnd = Math.Min(i0 + BLOCK_M, M);

                for (int k0 = 0; k0 < K; k0 += BLOCK_K)
                {
                    int kEnd = Math.Min(k0 + BLOCK_K, K);

                    for (int j0 = 0; j0 < N; j0 += BLOCK_N)
                    {
                        int jEnd = Math.Min(j0 + BLOCK_N, N);

                        // Process block
                        MatMulBlockSimd_Float(a, b, c, N, K, i0, iEnd, j0, jEnd, k0, kEnd);
                    }
                }
            });
        }

        /// <summary>
        /// Parallel blocked SIMD matrix multiplication for double.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulParallel_Double(double* a, double* b, double* c, int M, int N, int K)
        {
            // Zero out C first
            var cSize = M * N;
            for (int i = 0; i < cSize; i++)
                c[i] = 0;

            // For small matrices, use sequential version
            if ((long)M * N * K < PARALLEL_THRESHOLD)
            {
                if (M <= SMALL_MATRIX_THRESHOLD && N <= SMALL_MATRIX_THRESHOLD && K <= SMALL_MATRIX_THRESHOLD)
                    MatMulSimdSmall_Double(a, b, c, M, N, K);
                else
                    MatMulBlockedSimd_Double(a, b, c, M, N, K);
                return;
            }

            // Parallel over row blocks
            int numRowBlocks = (M + BLOCK_M - 1) / BLOCK_M;

            System.Threading.Tasks.Parallel.For(0, numRowBlocks, i0Idx =>
            {
                int i0 = i0Idx * BLOCK_M;
                int iEnd = Math.Min(i0 + BLOCK_M, M);

                for (int k0 = 0; k0 < K; k0 += BLOCK_K)
                {
                    int kEnd = Math.Min(k0 + BLOCK_K, K);

                    for (int j0 = 0; j0 < N; j0 += BLOCK_N)
                    {
                        int jEnd = Math.Min(j0 + BLOCK_N, N);

                        // Process block
                        MatMulBlockSimd_Double(a, b, c, N, K, i0, iEnd, j0, jEnd, k0, kEnd);
                    }
                }
            });
        }

        #endregion
    }
}
