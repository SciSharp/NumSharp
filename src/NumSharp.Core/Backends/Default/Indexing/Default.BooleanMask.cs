using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
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

            // Copy elements where mask is true
            switch (arr.typecode)
            {
                case NPTypeCode.Boolean:
                    ILKernelGenerator.CopyMaskedElementsHelper((bool*)arr.Address, (bool*)mask.Address, (bool*)result.Address, size);
                    break;
                case NPTypeCode.Byte:
                    ILKernelGenerator.CopyMaskedElementsHelper((byte*)arr.Address, (bool*)mask.Address, (byte*)result.Address, size);
                    break;
                case NPTypeCode.SByte:
                    ILKernelGenerator.CopyMaskedElementsHelper((sbyte*)arr.Address, (bool*)mask.Address, (sbyte*)result.Address, size);
                    break;
                case NPTypeCode.Int16:
                    ILKernelGenerator.CopyMaskedElementsHelper((short*)arr.Address, (bool*)mask.Address, (short*)result.Address, size);
                    break;
                case NPTypeCode.UInt16:
                    ILKernelGenerator.CopyMaskedElementsHelper((ushort*)arr.Address, (bool*)mask.Address, (ushort*)result.Address, size);
                    break;
                case NPTypeCode.Int32:
                    ILKernelGenerator.CopyMaskedElementsHelper((int*)arr.Address, (bool*)mask.Address, (int*)result.Address, size);
                    break;
                case NPTypeCode.UInt32:
                    ILKernelGenerator.CopyMaskedElementsHelper((uint*)arr.Address, (bool*)mask.Address, (uint*)result.Address, size);
                    break;
                case NPTypeCode.Int64:
                    ILKernelGenerator.CopyMaskedElementsHelper((long*)arr.Address, (bool*)mask.Address, (long*)result.Address, size);
                    break;
                case NPTypeCode.UInt64:
                    ILKernelGenerator.CopyMaskedElementsHelper((ulong*)arr.Address, (bool*)mask.Address, (ulong*)result.Address, size);
                    break;
                case NPTypeCode.Char:
                    ILKernelGenerator.CopyMaskedElementsHelper((char*)arr.Address, (bool*)mask.Address, (char*)result.Address, size);
                    break;
                case NPTypeCode.Half:
                    ILKernelGenerator.CopyMaskedElementsHelper((Half*)arr.Address, (bool*)mask.Address, (Half*)result.Address, size);
                    break;
                case NPTypeCode.Single:
                    ILKernelGenerator.CopyMaskedElementsHelper((float*)arr.Address, (bool*)mask.Address, (float*)result.Address, size);
                    break;
                case NPTypeCode.Double:
                    ILKernelGenerator.CopyMaskedElementsHelper((double*)arr.Address, (bool*)mask.Address, (double*)result.Address, size);
                    break;
                case NPTypeCode.Decimal:
                    ILKernelGenerator.CopyMaskedElementsHelper((decimal*)arr.Address, (bool*)mask.Address, (decimal*)result.Address, size);
                    break;
                case NPTypeCode.Complex:
                    ILKernelGenerator.CopyMaskedElementsHelper((System.Numerics.Complex*)arr.Address, (bool*)mask.Address, (System.Numerics.Complex*)result.Address, size);
                    break;
                default:
                    throw new NotSupportedException($"Type {arr.typecode} not supported for boolean masking");
            }

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
