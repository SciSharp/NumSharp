using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Stride-based pattern detection for selecting optimal SIMD execution paths.
    /// All methods are aggressively inlined for minimal dispatch overhead.
    /// </summary>
    public static class StrideDetector
    {
        /// <summary>
        /// Check if array is fully contiguous (C-order).
        /// An array is contiguous if strides match expected C-order values:
        /// strides[n-1] = 1, strides[i] = strides[i+1] * shape[i+1]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe bool IsContiguous(int* strides, int* shape, int ndim)
        {
            if (ndim == 0) return true;

            int expectedStride = 1;
            for (int d = ndim - 1; d >= 0; d--)
            {
                // Skip dimensions of size 1 (they don't affect contiguity)
                if (shape[d] > 1 && strides[d] != expectedStride)
                    return false;
                expectedStride *= shape[d];
            }
            return true;
        }

        /// <summary>
        /// Check if array is a scalar (all strides are zero).
        /// A scalar is broadcast to any shape - each element accesses the same value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe bool IsScalar(int* strides, int ndim)
        {
            for (int d = 0; d < ndim; d++)
            {
                if (strides[d] != 0)
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Check if inner dimension is suitable for SIMD chunking.
        /// Returns true if both operands have inner stride of 1 (contiguous) or 0 (broadcast).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe bool CanSimdChunk<T>(int* lhsStrides, int* rhsStrides, int* shape, int ndim)
            where T : unmanaged
        {
            if (ndim == 0) return false;

            int innerSize = shape[ndim - 1];
            int minVectorSize = Vector256<T>.Count;

            // Inner dimension must be large enough for SIMD
            if (innerSize < minVectorSize)
                return false;

            int lhsInner = lhsStrides[ndim - 1];
            int rhsInner = rhsStrides[ndim - 1];

            // Both must be contiguous (1) or broadcast (0) in inner dimension
            return (lhsInner == 1 || lhsInner == 0) &&
                   (rhsInner == 1 || rhsInner == 0);
        }

        /// <summary>
        /// Classify the binary operation into an execution path based on stride analysis.
        /// Classification priority:
        /// 1. SimdFull - both fully contiguous (fastest)
        /// 2. SimdScalarRight/Left - one operand is scalar
        /// 3. SimdChunk - inner dimension is contiguous/broadcast
        /// 4. General - fallback for arbitrary strides
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe ExecutionPath Classify<T>(
            int* lhsStrides,
            int* rhsStrides,
            int* shape,
            int ndim)
            where T : unmanaged
        {
            bool lhsContiguous = IsContiguous(lhsStrides, shape, ndim);
            bool rhsContiguous = IsContiguous(rhsStrides, shape, ndim);

            // PATH 1: Both fully contiguous - use flat SIMD loop
            if (lhsContiguous && rhsContiguous)
            {
                return ExecutionPath.SimdFull;
            }

            // PATH 2: Scalar broadcast - use scalar splat
            bool rhsScalar = IsScalar(rhsStrides, ndim);
            if (rhsScalar)
            {
                return ExecutionPath.SimdScalarRight;
            }

            bool lhsScalar = IsScalar(lhsStrides, ndim);
            if (lhsScalar)
            {
                return ExecutionPath.SimdScalarLeft;
            }

            // PATH 3: Inner dimension contiguous - use chunked SIMD
            if (CanSimdChunk<T>(lhsStrides, rhsStrides, shape, ndim))
            {
                return ExecutionPath.SimdChunk;
            }

            // PATH 4: General case - scalar loop with offset calculation
            return ExecutionPath.General;
        }
    }
}
