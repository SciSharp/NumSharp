using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;
using NumSharp.Utilities.Maths;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        #region 2D x 2D Matrix Multiplication

        /// <summary>
        /// Matrix multiplication for 2D arrays: C = A @ B
        /// A is [M x K], B is [K x N], result is [M x N]
        /// </summary>
        /// <remarks>
        /// Every dtype takes a stride-native path — no copies are materialized
        /// for transposed or sliced operands:
        ///   float / double    → BLIS-style SIMD GEMM in SimdMatMul (packers
        ///                        handle arbitrary strides).
        ///   all other dtypes  → INumber&lt;T&gt; generic kernel in
        ///                        Default.MatMul.Strided.cs, scalar pointer
        ///                        arithmetic.
        /// </remarks>
        [SuppressMessage("ReSharper", "JoinDeclarationAndInitializer")]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        protected static NDArray MultiplyMatrix(NDArray left, NDArray right, NDArray @out = null)
        {
            Debug.Assert(left.Shape.NDim == 2);
            Debug.Assert(right.Shape.NDim == 2);

            long M = left.shape[0];   // rows of A
            long K = left.shape[1];   // cols of A = rows of B
            long N = right.shape[1];  // cols of B

            if (K != right.shape[0])
                throw new IncorrectShapeException(
                    $"shapes {left.Shape} and {right.Shape} not aligned: {K} (dim 1) != {right.shape[0]} (dim 0)");

            // Determine output type and create result array
            var resultType = np._FindCommonArrayType(left.GetTypeCode, right.GetTypeCode);
            NDArray result = @out ?? new NDArray(resultType, Shape.Matrix(M, N));

            if (!(@out is null))
            {
                if (@out.ndim != 2 || @out.shape[0] != M || @out.shape[1] != N)
                    throw new IncorrectShapeException(
                        $"Output shape {@out.Shape} incompatible with matmul result shape ({M}, {N})");
            }

            // Stride-aware SIMD path for same-type float / double.
            if (TryMatMulSimd(left, right, result, M, K, N))
                return result;

            // Stride-native generic kernel for everything else (no copies).
            MatMulStridedGeneric(left, right, result, M, K, N);

            return result;
        }

        /// <summary>
        /// SIMD-optimized matmul for same-type float / double, stride-aware.
        /// Passes (stride0, stride1) for each operand through to the BLIS-style
        /// kernel in <see cref="Kernels.SimdMatMul"/>, so transposed and
        /// sliced views take the fast path without materializing copies.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool TryMatMulSimd(NDArray left, NDArray right, NDArray result, long M, long K, long N)
        {
            if (!ILKernelGenerator.Enabled)
                return false;

            // C is written as row-major contiguous; the inputs can have
            // arbitrary strides (packers absorb them).
            if (!result.Shape.IsContiguous)
                return false;

            var typeCode = result.typecode;
            if (left.typecode != typeCode || right.typecode != typeCode)
                return false;

            var lShape = left.Shape;
            var rShape = right.Shape;
            long aStride0 = lShape.strides[0];
            long aStride1 = lShape.strides[1];
            long bStride0 = rShape.strides[0];
            long bStride1 = rShape.strides[1];

            switch (typeCode)
            {
                case NPTypeCode.Single:
                {
                    float* a = (float*)left.Address   + lShape.offset;
                    float* b = (float*)right.Address  + rShape.offset;
                    float* c = (float*)result.Address + result.Shape.offset;
                    SimdMatMul.MatMulFloat(a, aStride0, aStride1, b, bStride0, bStride1, c, M, N, K);
                    return true;
                }

                case NPTypeCode.Double:
                {
                    double* a = (double*)left.Address   + lShape.offset;
                    double* b = (double*)right.Address  + rShape.offset;
                    double* c = (double*)result.Address + result.Shape.offset;
                    SimdMatMul.MatMulDouble(a, aStride0, aStride1, b, bStride0, bStride1, c, M, N, K);
                    return true;
                }

                default:
                    return false;
            }
        }

        #endregion
    }
}
