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
        public static NDArray DotNDMD(in NDArray lhs, in NDArray rhs)
        {
#if MINIMAL
            return null;
#else
            // Validate contracting dimension
            int contractDim = lhs.Shape[-1];
            if (contractDim != rhs.Shape[-2])
                throw new IncorrectShapeException(
                    $"shapes {lhs.Shape} and {rhs.Shape} not aligned: {contractDim} (dim {lhs.ndim - 1}) != {rhs.Shape[-2]} (dim {rhs.ndim - 2})");

            // Compute output shape:
            // lhs shape without last dim + rhs shape without second-to-last dim
            int[] lshape = lhs.shape.RemoveAt(lhs.ndim - 1);  // All but last
            int[] rshape = rhs.shape.RemoveAt(rhs.ndim - 2);  // All but second-to-last

            var retShape = new int[lshape.Length + rshape.Length];
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
            int[] lshape, int[] rshape, int K)
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
            int[] lshape, int[] rshape, int K)
        {
            float* lhsPtr = (float*)lhs.Address;
            float* rhsPtr = (float*)rhs.Address;
            float* resPtr = (float*)result.Address;

            int lhsNdim = lhs.ndim;
            int rhsNdim = rhs.ndim;

            // Pre-compute strides for iteration
            int lhsInnerStride = lhs.strides[lhsNdim - 1];  // Stride along contracting dim in lhs
            int rhsContractStride = rhs.strides[rhsNdim - 2];  // Stride along contracting dim in rhs
            int rhsInnerStride = rhs.strides[rhsNdim - 1];  // Stride along last dim in rhs

            // Total elements to compute
            int totalLhs = 1;
            for (int i = 0; i < lshape.Length; i++)
                totalLhs *= lshape[i];

            int totalRhs = 1;
            for (int i = 0; i < rshape.Length; i++)
                totalRhs *= rshape[i];

            // Compute lhs strides (for multi-index iteration)
            int[] lhsIterStrides = ComputeIterStrides(lshape);
            int[] rhsIterStrides = ComputeIterStrides(rshape);

            // For each position in lhs (excluding contract dim)
            for (int li = 0; li < totalLhs; li++)
            {
                // Compute lhs base offset: position in lhs without last dim
                int lhsBase = ComputeBaseOffset(li, lhsIterStrides, lhs.strides, lshape.Length);

                // For each position in rhs (excluding contract dim)
                for (int ri = 0; ri < totalRhs; ri++)
                {
                    // Compute rhs base offset: we need to skip the contract dimension
                    int rhsBase = ComputeRhsBaseOffset(ri, rhsIterStrides, rhs.strides, rshape, rhsNdim);

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
            int[] lshape, int[] rshape, int K)
        {
            double* lhsPtr = (double*)lhs.Address;
            double* rhsPtr = (double*)rhs.Address;
            double* resPtr = (double*)result.Address;

            int lhsNdim = lhs.ndim;
            int rhsNdim = rhs.ndim;

            int lhsInnerStride = lhs.strides[lhsNdim - 1];
            int rhsContractStride = rhs.strides[rhsNdim - 2];
            int rhsInnerStride = rhs.strides[rhsNdim - 1];

            int totalLhs = 1;
            for (int i = 0; i < lshape.Length; i++)
                totalLhs *= lshape[i];

            int totalRhs = 1;
            for (int i = 0; i < rshape.Length; i++)
                totalRhs *= rshape[i];

            int[] lhsIterStrides = ComputeIterStrides(lshape);
            int[] rhsIterStrides = ComputeIterStrides(rshape);

            for (int li = 0; li < totalLhs; li++)
            {
                int lhsBase = ComputeBaseOffset(li, lhsIterStrides, lhs.strides, lshape.Length);

                for (int ri = 0; ri < totalRhs; ri++)
                {
                    int rhsBase = ComputeRhsBaseOffset(ri, rhsIterStrides, rhs.strides, rshape, rhsNdim);

                    double sum = DotProductDouble(
                        lhsPtr + lhsBase, lhsInnerStride,
                        rhsPtr + rhsBase, rhsContractStride,
                        K);

                    resPtr[li * totalRhs + ri] = sum;
                }
            }
        }

        /// <summary>
        /// Compute iteration strides for multi-index decomposition.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int[] ComputeIterStrides(int[] shape)
        {
            var strides = new int[shape.Length];
            int stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        }

        /// <summary>
        /// Compute base offset for lhs array from linear index.
        /// Maps linear index over lshape to offset in lhs storage (using lhs strides, not contract dim).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ComputeBaseOffset(int linearIdx, int[] iterStrides, int[] arrayStrides, int ndim)
        {
            int offset = 0;
            for (int d = 0; d < ndim; d++)
            {
                int idx = linearIdx / iterStrides[d];
                linearIdx %= iterStrides[d];
                offset += idx * arrayStrides[d];
            }
            return offset;
        }

        /// <summary>
        /// Compute base offset for rhs array from linear index.
        /// rshape excludes the contracting dimension (second-to-last in original rhs).
        /// We need to map indices back, skipping the contract dim.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ComputeRhsBaseOffset(int linearIdx, int[] iterStrides, int[] arrayStrides, int[] rshape, int rhsNdim)
        {
            int offset = 0;
            int contractDimIdx = rhsNdim - 2;  // The dimension we're contracting over

            int rshapeIdx = 0;
            for (int d = 0; d < rhsNdim; d++)
            {
                if (d == contractDimIdx)
                    continue;  // Skip contract dimension

                int idx = linearIdx / iterStrides[rshapeIdx];
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
        private static unsafe float DotProductFloat(float* a, int strideA, float* b, int strideB, int n)
        {
            float sum = 0;

            // Fast path: both contiguous (stride=1)
            if (strideA == 1 && strideB == 1)
            {
                int i = 0;

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
                for (int i = 0; i < n; i++)
                    sum += a[i * strideA] * b[i * strideB];
            }

            return sum;
        }

        /// <summary>
        /// SIMD dot product for double with arbitrary strides.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe double DotProductDouble(double* a, int strideA, double* b, int strideB, int n)
        {
            double sum = 0;

            if (strideA == 1 && strideB == 1)
            {
                int i = 0;

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
                for (int i = 0; i < n; i++)
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
            int[] lshape, int[] rshape, int K)
        {
            int lhsNdim = lhs.ndim;
            int rhsNdim = rhs.ndim;

            // Compute total iterations
            int totalLhs = 1;
            for (int i = 0; i < lshape.Length; i++)
                totalLhs *= lshape[i];

            int totalRhs = 1;
            for (int i = 0; i < rshape.Length; i++)
                totalRhs *= rshape[i];

            // Pre-compute strides for iteration
            int[] lhsIterStrides = ComputeIterStrides(lshape);
            int[] rhsIterStrides = ComputeIterStrides(rshape);

            // Temporary arrays for coordinate computation
            int[] lhsCoords = new int[lhsNdim];
            int[] rhsCoords = new int[rhsNdim];

            int resultIdx = 0;

            // Iterate over all lhs positions (excluding contract dim)
            for (int li = 0; li < totalLhs; li++)
            {
                // Decompose li into lhs coordinates (first ndim-1 dims)
                DecomposeIndex(li, lhsIterStrides, lhsCoords, lshape.Length);

                // Iterate over all rhs positions (excluding contract dim)
                for (int ri = 0; ri < totalRhs; ri++)
                {
                    // Decompose ri into rhs coordinates (skip contract dim)
                    DecomposeRhsIndex(ri, rhsIterStrides, rhsCoords, rshape, rhsNdim);

                    // Compute dot product along contracting dimension
                    double sum = 0;
                    for (int k = 0; k < K; k++)
                    {
                        // lhs[..., k] - last dim is contract dim
                        lhsCoords[lhsNdim - 1] = k;
                        // Use GetValue(coords) which correctly applies Shape.GetOffset internally
                        // Note: GetAtIndex(Shape.GetOffset(coords)) is wrong because GetAtIndex
                        // applies TransformOffset again, double-transforming for non-contiguous arrays
                        double lVal = Convert.ToDouble(lhs.GetValue(lhsCoords));

                        // rhs[..., k, ...] - second-to-last dim is contract dim
                        rhsCoords[rhsNdim - 2] = k;
                        double rVal = Convert.ToDouble(rhs.GetValue(rhsCoords));

                        sum += lVal * rVal;
                    }

                    // Store result with type conversion
                    result.SetAtIndex(Converts.ChangeType(sum, result.typecode), resultIdx++);
                }
            }
        }

        /// <summary>
        /// Decompose linear index into coordinates for lhs (first ndim dims).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void DecomposeIndex(int linearIdx, int[] iterStrides, int[] coords, int ndim)
        {
            for (int d = 0; d < ndim; d++)
            {
                coords[d] = linearIdx / iterStrides[d];
                linearIdx %= iterStrides[d];
            }
        }

        /// <summary>
        /// Decompose linear index into coordinates for rhs, skipping the contract dimension.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void DecomposeRhsIndex(int linearIdx, int[] iterStrides, int[] coords, int[] rshape, int rhsNdim)
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
