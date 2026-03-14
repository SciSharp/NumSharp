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
        /// Implementation strategy:
        /// 1. SIMD fast path for contiguous float/double (40-100x faster)
        /// 2. Generic fallback using Unsafe pointer arithmetic for all types
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

            // ========== SIMD FAST PATH ==========
            // For contiguous same-type float/double matrices, use blocked SIMD kernel
            if (TryMatMulSimd(left, right, result, M, K, N))
                return result;

            // ========== GENERIC FALLBACK ==========
            // Handle all type combinations with pointer-based implementation
            MatMulGeneric(left, right, result, M, K, N);

            return result;
        }

        /// <summary>
        /// SIMD-optimized matrix multiplication for contiguous float/double arrays.
        /// Uses cache-blocked algorithm with Vector256 FMA operations.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool TryMatMulSimd(NDArray left, NDArray right, NDArray result, long M, long K, long N)
        {
            if (!ILKernelGenerator.Enabled)
                return false;

            // Require all arrays contiguous and same type
            if (!left.Shape.IsContiguous || !right.Shape.IsContiguous || !result.Shape.IsContiguous)
                return false;

            var typeCode = result.typecode;
            if (left.typecode != typeCode || right.typecode != typeCode)
                return false;

            switch (typeCode)
            {
                case NPTypeCode.Single:
                {
                    float* a = (float*)left.Address;
                    float* b = (float*)right.Address;
                    float* c = (float*)result.Address;

                    // Use cache-blocked implementation for better performance
                    SimdMatMul.MatMulFloat(a, b, c, M, N, K);
                    return true;
                }

                case NPTypeCode.Double:
                {
                    var kernel = ILKernelGenerator.GetMatMulKernel<double>();
                    if (kernel == null) return false;

                    double* a = (double*)left.Address;
                    double* b = (double*)right.Address;
                    double* c = (double*)result.Address;

                    kernel(a, b, c, M, N, K);
                    return true;
                }

                default:
                    return false;
            }
        }

        /// <summary>
        /// Generic matrix multiplication supporting all type combinations.
        /// Uses ikj loop order for better cache utilization.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulGeneric(NDArray left, NDArray right, NDArray result, long M, long K, long N)
        {
            // Dispatch based on result type for optimal inner loop
            switch (result.typecode)
            {
                case NPTypeCode.Boolean:
                    MatMulCore<bool>(left, right, result, M, K, N);
                    break;
                case NPTypeCode.Byte:
                    MatMulCore<byte>(left, right, result, M, K, N);
                    break;
                case NPTypeCode.Int16:
                    MatMulCore<short>(left, right, result, M, K, N);
                    break;
                case NPTypeCode.UInt16:
                    MatMulCore<ushort>(left, right, result, M, K, N);
                    break;
                case NPTypeCode.Int32:
                    MatMulCore<int>(left, right, result, M, K, N);
                    break;
                case NPTypeCode.UInt32:
                    MatMulCore<uint>(left, right, result, M, K, N);
                    break;
                case NPTypeCode.Int64:
                    MatMulCore<long>(left, right, result, M, K, N);
                    break;
                case NPTypeCode.UInt64:
                    MatMulCore<ulong>(left, right, result, M, K, N);
                    break;
                case NPTypeCode.Char:
                    MatMulCore<char>(left, right, result, M, K, N);
                    break;
                case NPTypeCode.Single:
                    MatMulCore<float>(left, right, result, M, K, N);
                    break;
                case NPTypeCode.Double:
                    MatMulCore<double>(left, right, result, M, K, N);
                    break;
                case NPTypeCode.Decimal:
                    MatMulCore<decimal>(left, right, result, M, K, N);
                    break;
                default:
                    throw new NotSupportedException($"MatMul not supported for type {result.typecode}");
            }
        }

        /// <summary>
        /// Core matrix multiplication with typed result array.
        /// Handles mixed input types by converting to double for computation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulCore<TResult>(NDArray left, NDArray right, NDArray result, long M, long K, long N)
            where TResult : unmanaged
        {
            // Get typed result pointer
            var resultPtr = (TResult*)result.Address;

            // Zero out result
            long resultSize = M * N;
            for (long i = 0; i < resultSize; i++)
                resultPtr[i] = default;

            // Check if we can use fast contiguous path (same types, contiguous)
            bool leftContiguous = left.Shape.IsContiguous;
            bool rightContiguous = right.Shape.IsContiguous;

            // For same-type contiguous arrays, use optimized pointer loop
            if (leftContiguous && rightContiguous &&
                left.typecode == result.typecode && right.typecode == result.typecode)
            {
                MatMulSameType<TResult>(left, right, resultPtr, M, K, N);
                return;
            }

            // General case: use GetAtIndex for strided access, compute in double
            MatMulMixedType<TResult>(left, right, resultPtr, M, K, N);
        }

        /// <summary>
        /// Optimized path for same-type contiguous matrices.
        /// Dispatches to type-specific implementation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulSameType<T>(NDArray left, NDArray right, T* result, long M, long K, long N)
            where T : unmanaged
        {
            // For same-type contiguous, dispatch to specific implementations
            // This avoids generic arithmetic overhead
            if (typeof(T) == typeof(float))
                MatMulContiguous((float*)left.Address, (float*)right.Address, (float*)(void*)result, M, K, N);
            else if (typeof(T) == typeof(double))
                MatMulContiguous((double*)left.Address, (double*)right.Address, (double*)(void*)result, M, K, N);
            else if (typeof(T) == typeof(int))
                MatMulContiguous((int*)left.Address, (int*)right.Address, (int*)(void*)result, M, K, N);
            else if (typeof(T) == typeof(long))
                MatMulContiguous((long*)left.Address, (long*)right.Address, (long*)(void*)result, M, K, N);
            else
                // Fall back to mixed-type path for other types
                MatMulMixedType(left, right, result, M, K, N);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulContiguous(float* a, float* b, float* result, long M, long K, long N)
        {
            for (long i = 0; i < M; i++)
            {
                float* resultRow = result + i * N;
                float* aRow = a + i * K;
                for (long k = 0; k < K; k++)
                {
                    float aik = aRow[k];
                    float* bRow = b + k * N;
                    for (long j = 0; j < N; j++)
                        resultRow[j] += aik * bRow[j];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulContiguous(double* a, double* b, double* result, long M, long K, long N)
        {
            for (long i = 0; i < M; i++)
            {
                double* resultRow = result + i * N;
                double* aRow = a + i * K;
                for (long k = 0; k < K; k++)
                {
                    double aik = aRow[k];
                    double* bRow = b + k * N;
                    for (long j = 0; j < N; j++)
                        resultRow[j] += aik * bRow[j];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulContiguous(int* a, int* b, int* result, long M, long K, long N)
        {
            for (long i = 0; i < M; i++)
            {
                int* resultRow = result + i * N;
                int* aRow = a + i * K;
                for (long k = 0; k < K; k++)
                {
                    int aik = aRow[k];
                    int* bRow = b + k * N;
                    for (long j = 0; j < N; j++)
                        resultRow[j] += aik * bRow[j];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulContiguous(long* a, long* b, long* result, long M, long K, long N)
        {
            for (long i = 0; i < M; i++)
            {
                long* resultRow = result + i * N;
                long* aRow = a + i * K;
                for (long k = 0; k < K; k++)
                {
                    long aik = aRow[k];
                    long* bRow = b + k * N;
                    for (long j = 0; j < N; j++)
                        resultRow[j] += aik * bRow[j];
                }
            }
        }

        /// <summary>
        /// General path for mixed types or strided arrays.
        /// Converts to double for computation, then back to result type.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulMixedType<TResult>(NDArray left, NDArray right, TResult* result, long M, long K, long N)
            where TResult : unmanaged
        {
            // Use double accumulator for precision
            var accumulator = new double[N];

            // Temporary arrays for coordinates to avoid allocation in inner loop
            var leftCoords = new long[2];
            var rightCoords = new long[2];

            for (long i = 0; i < M; i++)
            {
                // Clear accumulator for this row
                Array.Clear(accumulator, 0, (int)N);

                leftCoords[0] = i;
                for (long k = 0; k < K; k++)
                {
                    leftCoords[1] = k;
                    // Use GetValue which correctly handles strided/non-contiguous arrays
                    // Note: GetAtIndex with manual stride calculation was wrong for transposed arrays
                    // because GetAtIndex applies TransformOffset which double-transforms for non-contiguous
                    double aik = Convert.ToDouble(left.GetValue(leftCoords));

                    rightCoords[0] = k;
                    for (long j = 0; j < N; j++)
                    {
                        rightCoords[1] = j;
                        double bkj = Convert.ToDouble(right.GetValue(rightCoords));
                        accumulator[j] += aik * bkj;
                    }
                }

                // Write row to result with type conversion
                TResult* resultRow = result + i * N;
                for (long j = 0; j < N; j++)
                {
                    resultRow[j] = Converts.ChangeType<TResult>(accumulator[j]);
                }
            }
        }

        #endregion
    }
}
