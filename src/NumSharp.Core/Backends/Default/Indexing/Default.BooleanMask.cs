using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        private static unsafe void CopyMaskedDispatch<T>(nint arr, nint mask, nint result, long size) where T : unmanaged
            => ILKernelGenerator.CopyMaskedElementsHelper((T*)arr, (bool*)mask, (T*)result, size);

        /// <summary>
        /// Apply a boolean mask to select elements from an array.
        /// </summary>
        /// <param name="arr">Source array.</param>
        /// <param name="mask">Boolean mask (must have same shape as arr).</param>
        /// <returns>1D array containing elements where mask is true.</returns>
        public override NDArray BooleanMask(NDArray arr, NDArray mask)
        {
            if (mask.typecode != NPTypeCode.Boolean)
                throw new ArgumentException("Mask must be boolean array", nameof(mask));

            // SIMD fast path: contiguous arrays of same size
            if (ILKernelGenerator.Enabled && ILKernelGenerator.VectorBits > 0 &&
                mask.Shape.IsContiguous && arr.Shape.IsContiguous)
            {
                return BooleanMaskSimd(arr, mask.MakeGeneric<bool>());
            }

            // Fallback: use nonzero + fancy indexing
            return BooleanMaskFallback(arr, mask.MakeGeneric<bool>());
        }

        /// <summary>
        /// SIMD-optimized boolean masking for contiguous arrays.
        /// </summary>
        private unsafe NDArray BooleanMaskSimd(NDArray arr, NDArray<bool> mask)
        {
            long size = arr.size;

            // Count true values using SIMD
            long trueCount = ILKernelGenerator.CountTrueSimdHelper((bool*)mask.Address, size);

            if (trueCount == 0)
                return new NDArray(arr.dtype, Shape.Empty(1)); // Empty 1D result

            // Create result array
            var result = new NDArray(arr.dtype, new Shape(trueCount));

            NpFunc.Invoke(arr.typecode, CopyMaskedDispatch<int>, (nint)arr.Address, (nint)mask.Address, (nint)result.Address, size);

            return result;
        }

        /// <summary>
        /// Fallback boolean masking using NpyIter-based iteration.
        /// Handles strided/broadcast arr and/or mask.
        /// </summary>
        private unsafe NDArray BooleanMaskFallback(NDArray arr, NDArray<bool> mask)
        {
            // Pass 1: Count true values in the mask (layout-aware via NpyIter).
            long trueCount;
            using (var maskIter = NpyIterRef.New(mask, NpyIterGlobalFlags.EXTERNAL_LOOP))
            {
                trueCount = maskIter.ExecuteReducing<CountNonZeroKernel<bool>, long>(default, 0L);
            }

            if (trueCount == 0)
                return new NDArray(arr.dtype, Shape.Empty(1));

            var result = new NDArray(arr.dtype, new Shape(trueCount));

            // Pass 2: Gather elements where mask is true into flat result.
            // NPY_CORDER forces logical C-order traversal (matching NumPy
            // boolean indexing semantics) instead of memory-efficient order.
            using (var iter = NpyIterRef.MultiNew(
                2, new[] { arr, (NDArray)mask },
                NpyIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_CORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY }))
            {
                var accum = new BooleanMaskGatherAccumulator
                {
                    DestPtr = (IntPtr)result.Address,
                    ElemSize = arr.dtypesize,
                    DestIdx = 0,
                };
                iter.ExecuteReducing<BooleanMaskGatherKernel, BooleanMaskGatherAccumulator>(default, accum);
            }

            return result;
        }

        /// <summary>
        /// Accumulator threading the destination byte pointer and write cursor
        /// through the multi-op gather loop.
        /// </summary>
        private struct BooleanMaskGatherAccumulator
        {
            public IntPtr DestPtr;
            public long DestIdx;
            public int ElemSize;
        }

        /// <summary>
        /// Inner loop: for each position, if mask is true, copy arr element
        /// into result[destIdx] and increment destIdx.
        /// </summary>
        private readonly struct BooleanMaskGatherKernel : INpyReducingInnerLoop<BooleanMaskGatherAccumulator>
        {
            public unsafe bool Execute(void** dataptrs, long* strides, long count, ref BooleanMaskGatherAccumulator accum)
            {
                byte* srcPtr = (byte*)dataptrs[0];
                byte* maskPtr = (byte*)dataptrs[1];
                long srcStride = strides[0];
                long maskStride = strides[1];
                byte* destBase = (byte*)accum.DestPtr;
                long destIdx = accum.DestIdx;
                int elemSize = accum.ElemSize;

                for (long i = 0; i < count; i++)
                {
                    bool m = *(bool*)(maskPtr + i * maskStride);
                    if (m)
                    {
                        System.Buffer.MemoryCopy(
                            srcPtr + i * srcStride,
                            destBase + destIdx * elemSize,
                            elemSize, elemSize);
                        destIdx++;
                    }
                }

                accum.DestIdx = destIdx;
                return true;
            }
        }
    }
}
