using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Utilities;
using NumSharp.Utilities.Maths;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// N-D x M-D dot product: sum product over last axis of lhs and second-to-last axis of rhs.
        /// dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
        /// </summary>
        /// <remarks>
        /// This replaces the original ~15,880 line implementation that used nested switch statements
        /// for every combination of dimension counts. The new implementation uses dynamic iteration
        /// with optional SIMD optimization for contiguous float/double arrays.
        /// </remarks>
        public static NDArray DotNDMD(NDArray lhs, NDArray rhs)
        {
#if MINIMAL
            return null;
#else
            // Validate contracting dimension
            long contractDim = lhs.Shape[-1];
            if (contractDim != rhs.Shape[-2])
                throw new IncorrectShapeException(
                    $"shapes {lhs.Shape} and {rhs.Shape} not aligned: {contractDim} (dim {lhs.ndim - 1}) != {rhs.Shape[-2]} (dim {rhs.ndim - 2})");

            // Compute output shape:
            // lhs shape without last dim + rhs shape without second-to-last dim
            long[] lshape = lhs.shape.RemoveAt(lhs.ndim - 1);  // All but last
            long[] rshape = rhs.shape.RemoveAt(rhs.ndim - 2);  // All but second-to-last

            var retShape = new long[lshape.Length + rshape.Length];
            Array.Copy(lshape, 0, retShape, 0, lshape.Length);
            Array.Copy(rshape, 0, retShape, lshape.Length, rshape.Length);

            var resultType = np._FindCommonArrayType(lhs.typecode, rhs.typecode);
            var result = new NDArray(resultType, new Shape(retShape));

            // Try SIMD fast path for contiguous same-type float/double
            if (TryDotNDMDSimd(lhs, rhs, result, lshape, rshape, contractDim))
                return result;

            // Generic fallback - handles all types and strides
            DotNDMDGeneric(lhs, rhs, result, lshape, rshape, contractDim);
            return result;
#endif
        }

        /// <summary>
        /// SIMD-optimized N-D dot product for contiguous float/double arrays.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool TryDotNDMDSimd(NDArray lhs, NDArray rhs, NDArray result,
            long[] lshape, long[] rshape, long K)
        {
            // Require contiguous arrays and same type
            if (!lhs.Shape.IsContiguous || !rhs.Shape.IsContiguous || !result.Shape.IsContiguous)
                return false;

            var typeCode = result.typecode;
            if (lhs.typecode != typeCode || rhs.typecode != typeCode)
                return false;

            switch (typeCode)
            {
                case NPTypeCode.Single:
                    DotNDMDSimdFloat(lhs, rhs, result, lshape, rshape, K);
                    return true;

                case NPTypeCode.Double:
                    DotNDMDSimdDouble(lhs, rhs, result, lshape, rshape, K);
                    return true;

                default:
                    return false;
            }
        }

        /// <summary>
        /// SIMD float implementation of N-D dot product.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DotNDMDSimdFloat(NDArray lhs, NDArray rhs, NDArray result,
            long[] lshape, long[] rshape, long K)
        {
            float* lhsPtr = (float*)lhs.Address;
            float* rhsPtr = (float*)rhs.Address;
            float* resPtr = (float*)result.Address;

            int lhsNdim = lhs.ndim;
            int rhsNdim = rhs.ndim;

            // Pre-compute strides for iteration
            long lhsInnerStride = lhs.strides[lhsNdim - 1];  // Stride along contracting dim in lhs
            long rhsContractStride = rhs.strides[rhsNdim - 2];  // Stride along contracting dim in rhs
            long rhsInnerStride = rhs.strides[rhsNdim - 1];  // Stride along last dim in rhs

            // Total elements to compute
            long totalLhs = 1;
            for (int i = 0; i < lshape.Length; i++)
                totalLhs *= lshape[i];

            long totalRhs = 1;
            for (int i = 0; i < rshape.Length; i++)
                totalRhs *= rshape[i];

            // Compute lhs strides (for multi-index iteration)
            long[] lhsIterStrides = ComputeIterStrides64(lshape);
            long[] rhsIterStrides = ComputeIterStrides64(rshape);

            // For each position in lhs (excluding contract dim)
            for (long li = 0; li < totalLhs; li++)
            {
                // Compute lhs base offset: position in lhs without last dim
                long lhsBase = ComputeBaseOffset64(li, lhsIterStrides, lhs.strides, lshape.Length);

                // For each position in rhs (excluding contract dim)
                for (long ri = 0; ri < totalRhs; ri++)
                {
                    // Compute rhs base offset: we need to skip the contract dimension
                    long rhsBase = ComputeRhsBaseOffset64(ri, rhsIterStrides, rhs.strides, rshape, rhsNdim);

                    // Compute dot product along contracting dimension
                    float sum = DotProductFloat(
                        lhsPtr + lhsBase, lhsInnerStride,
                        rhsPtr + rhsBase, rhsContractStride,
                        K);

                    // Store result
                    resPtr[li * totalRhs + ri] = sum;
                }
            }
        }

        /// <summary>
        /// SIMD double implementation of N-D dot product.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DotNDMDSimdDouble(NDArray lhs, NDArray rhs, NDArray result,
            long[] lshape, long[] rshape, long K)
        {
            double* lhsPtr = (double*)lhs.Address;
            double* rhsPtr = (double*)rhs.Address;
            double* resPtr = (double*)result.Address;

            int lhsNdim = lhs.ndim;
            int rhsNdim = rhs.ndim;

            long lhsInnerStride = lhs.strides[lhsNdim - 1];
            long rhsContractStride = rhs.strides[rhsNdim - 2];
            long rhsInnerStride = rhs.strides[rhsNdim - 1];

            long totalLhs = 1;
            for (int i = 0; i < lshape.Length; i++)
                totalLhs *= lshape[i];

            long totalRhs = 1;
            for (int i = 0; i < rshape.Length; i++)
                totalRhs *= rshape[i];

            long[] lhsIterStrides = ComputeIterStrides64(lshape);
            long[] rhsIterStrides = ComputeIterStrides64(rshape);

            for (long li = 0; li < totalLhs; li++)
            {
                long lhsBase = ComputeBaseOffset64(li, lhsIterStrides, lhs.strides, lshape.Length);

                for (long ri = 0; ri < totalRhs; ri++)
                {
                    long rhsBase = ComputeRhsBaseOffset64(ri, rhsIterStrides, rhs.strides, rshape, rhsNdim);

                    double sum = DotProductDouble(
                        lhsPtr + lhsBase, lhsInnerStride,
                        rhsPtr + rhsBase, rhsContractStride,
                        K);

                    resPtr[li * totalRhs + ri] = sum;
                }
            }
        }

        /// <summary>
        /// Compute iteration strides for multi-index decomposition (64-bit).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long[] ComputeIterStrides64(long[] shape)
        {
            var strides = new long[shape.Length];
            long stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        }

        /// <summary>
        /// Compute base offset for lhs array from linear index (64-bit).
        /// Maps linear index over lshape to offset in lhs storage (using lhs strides, not contract dim).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long ComputeBaseOffset64(long linearIdx, long[] iterStrides, long[] arrayStrides, int ndim)
        {
            long offset = 0;
            for (int d = 0; d < ndim; d++)
            {
                long idx = linearIdx / iterStrides[d];
                linearIdx %= iterStrides[d];
                offset += idx * arrayStrides[d];
            }
            return offset;
        }

        /// <summary>
        /// Compute base offset for rhs array from linear index (64-bit).
        /// rshape excludes the contracting dimension (second-to-last in original rhs).
        /// We need to map indices back, skipping the contract dim.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long ComputeRhsBaseOffset64(long linearIdx, long[] iterStrides, long[] arrayStrides, long[] rshape, int rhsNdim)
        {
            long offset = 0;
            int contractDimIdx = rhsNdim - 2;  // The dimension we're contracting over

            int rshapeIdx = 0;
            for (int d = 0; d < rhsNdim; d++)
            {
                if (d == contractDimIdx)
                    continue;  // Skip contract dimension

                long idx = linearIdx / iterStrides[rshapeIdx];
                linearIdx %= iterStrides[rshapeIdx];
                offset += idx * arrayStrides[d];
                rshapeIdx++;
            }
            return offset;
        }

        /// <summary>
        /// SIMD dot product for float with arbitrary strides.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float DotProductFloat(float* a, long strideA, float* b, long strideB, long n)
        {
            float sum = 0;

            // Fast path: both contiguous (stride=1)
            if (strideA == 1 && strideB == 1)
            {
                long i = 0;

                // SIMD loop
                if (Vector256.IsHardwareAccelerated && n >= 8)
                {
                    var vsum = Vector256<float>.Zero;
                    for (; i <= n - 8; i += 8)
                    {
                        var va = Vector256.Load(a + i);
                        var vb = Vector256.Load(b + i);
                        vsum = Vector256.Add(vsum, Vector256.Multiply(va, vb));
                    }
                    sum = Vector256.Sum(vsum);
                }

                // Scalar remainder
                for (; i < n; i++)
                    sum += a[i] * b[i];
            }
            else
            {
                // Strided access
                for (long i = 0; i < n; i++)
                    sum += a[i * strideA] * b[i * strideB];
            }

            return sum;
        }

        /// <summary>
        /// SIMD dot product for double with arbitrary strides.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe double DotProductDouble(double* a, long strideA, double* b, long strideB, long n)
        {
            double sum = 0;

            if (strideA == 1 && strideB == 1)
            {
                long i = 0;

                if (Vector256.IsHardwareAccelerated && n >= 4)
                {
                    var vsum = Vector256<double>.Zero;
                    for (; i <= n - 4; i += 4)
                    {
                        var va = Vector256.Load(a + i);
                        var vb = Vector256.Load(b + i);
                        vsum = Vector256.Add(vsum, Vector256.Multiply(va, vb));
                    }
                    sum = Vector256.Sum(vsum);
                }

                for (; i < n; i++)
                    sum += a[i] * b[i];
            }
            else
            {
                for (long i = 0; i < n; i++)
                    sum += a[i * strideA] * b[i * strideB];
            }

            return sum;
        }

        /// <summary>
        /// Generic N-D dot product for all types and non-contiguous arrays.
        /// Uses GetAtIndex for strided access and computes in double for precision.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void DotNDMDGeneric(NDArray lhs, NDArray rhs, NDArray result,
            long[] lshape, long[] rshape, long K)
        {
            int lhsNdim = lhs.ndim;
            int rhsNdim = rhs.ndim;

            // Compute total iterations
            long totalLhs = 1;
            for (int i = 0; i < lshape.Length; i++)
                totalLhs *= lshape[i];

            long totalRhs = 1;
            for (int i = 0; i < rshape.Length; i++)
                totalRhs *= rshape[i];

            // Pre-compute strides for iteration
            long[] lhsIterStrides = ComputeIterStrides64(lshape);
            long[] rhsIterStrides = ComputeIterStrides64(rshape);

            // Temporary arrays for coordinate computation
            long[] lhsCoords = new long[lhsNdim];
            long[] rhsCoords = new long[rhsNdim];

            long resultIdx = 0;

            // Iterate over all lhs positions (excluding contract dim)
            for (long li = 0; li < totalLhs; li++)
            {
                // Decompose li into lhs coordinates (first ndim-1 dims)
                DecomposeIndex64(li, lhsIterStrides, lhsCoords, lshape.Length);

                // Iterate over all rhs positions (excluding contract dim)
                for (long ri = 0; ri < totalRhs; ri++)
                {
                    // Decompose ri into rhs coordinates (skip contract dim)
                    DecomposeRhsIndex64(ri, rhsIterStrides, rhsCoords, rshape, rhsNdim);

                    // Compute dot product along contracting dimension
                    double sum = 0;
                    for (long k = 0; k < K; k++)
                    {
                        // lhs[..., k] - last dim is contract dim
                        lhsCoords[lhsNdim - 1] = k;
                        // Use GetValue(coords) which correctly applies Shape.GetOffset internally
                        // Note: GetAtIndex(Shape.GetOffset(coords)) is wrong because GetAtIndex
                        // applies TransformOffset again, double-transforming for non-contiguous arrays
                        // Converts.ToDouble handles all 15 dtypes including Half/Complex (System.Convert throws on those).
                        double lVal = Converts.ToDouble(lhs.GetValue(lhsCoords));

                        // rhs[..., k, ...] - second-to-last dim is contract dim
                        rhsCoords[rhsNdim - 2] = k;
                        double rVal = Converts.ToDouble(rhs.GetValue(rhsCoords));

                        sum += lVal * rVal;
                    }

                    // Store result with type conversion
                    result.SetAtIndex(Converts.ChangeType(sum, result.typecode), resultIdx++);
                }
            }
        }

        /// <summary>
        /// Decompose linear index into coordinates for lhs (first ndim dims, 64-bit).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void DecomposeIndex64(long linearIdx, long[] iterStrides, long[] coords, int ndim)
        {
            for (int d = 0; d < ndim; d++)
            {
                coords[d] = linearIdx / iterStrides[d];
                linearIdx %= iterStrides[d];
            }
        }

        /// <summary>
        /// Decompose linear index into coordinates for rhs, skipping the contract dimension (64-bit).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void DecomposeRhsIndex64(long linearIdx, long[] iterStrides, long[] coords, long[] rshape, int rhsNdim)
        {
            int contractDimIdx = rhsNdim - 2;
            int rshapeIdx = 0;

            for (int d = 0; d < rhsNdim; d++)
            {
                if (d == contractDimIdx)
                {
                    coords[d] = 0;  // Will be set during iteration
                    continue;
                }

                coords[d] = linearIdx / iterStrides[rshapeIdx];
                linearIdx %= iterStrides[rshapeIdx];
                rshapeIdx++;
            }
        }
    }
}
