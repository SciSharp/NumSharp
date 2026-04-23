using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

// =============================================================================
// Stride-aware double GEMM
// =============================================================================
//
// Mirrors the float simple path but with Vector256<double> (4 doubles per
// vector). Large contiguous double matmul already has an IL-generated IKJ
// SIMD kernel (ILKernelGenerator.GetMatMulKernel<double>), so the job here
// is only to add a stride-aware entry point that handles transposed / sliced
// double views without materializing a contiguous copy.
//
// Small / medium matrices use a stride-aware IKJ SIMD loop. Large matrices
// fall back to the contiguous IL kernel after an (unavoidable) copy. If
// double transposed-matmul ever becomes a hot path, mirror SimdMatMul.Strided
// to add a full blocked double kernel; the packer design transfers 1:1.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class SimdMatMul
    {
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

            MatMulDoubleSimpleStrided(A, aStride0, aStride1, B, bStride0, bStride1, C, M, N, K);
        }

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
                // B strided on the inner axis — scalar inner loop.
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
    }
}
