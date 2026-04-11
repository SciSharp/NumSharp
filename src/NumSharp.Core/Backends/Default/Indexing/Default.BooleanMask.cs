using System;
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
                case NPTypeCode.Single:
                    ILKernelGenerator.CopyMaskedElementsHelper((float*)arr.Address, (bool*)mask.Address, (float*)result.Address, size);
                    break;
                case NPTypeCode.Double:
                    ILKernelGenerator.CopyMaskedElementsHelper((double*)arr.Address, (bool*)mask.Address, (double*)result.Address, size);
                    break;
                case NPTypeCode.Decimal:
                    ILKernelGenerator.CopyMaskedElementsHelper((decimal*)arr.Address, (bool*)mask.Address, (decimal*)result.Address, size);
                    break;
                default:
                    throw new NotSupportedException($"Type {arr.typecode} not supported for boolean masking");
            }

            return result;
        }

        /// <summary>
        /// Fallback boolean masking using iteration.
        /// </summary>
        private unsafe NDArray BooleanMaskFallback(NDArray arr, NDArray<bool> mask)
        {
            // Count true values
            long trueCount = 0;
            var maskIter = mask.AsIterator<bool>();
            while (maskIter.HasNext())
            {
                if (maskIter.MoveNext())
                    trueCount++;
            }

            if (trueCount == 0)
                return new NDArray(arr.dtype, Shape.Empty(1));

            var result = new NDArray(arr.dtype, new Shape(trueCount));

            // Copy elements where mask is true
            maskIter.Reset();
            long destIdx = 0;
            long srcIdx = 0;
            while (maskIter.HasNext())
            {
                bool m = maskIter.MoveNext();
                if (m)
                {
                    result.SetAtIndex(arr.GetAtIndex(srcIdx), destIdx);
                    destIdx++;
                }
                srcIdx++;
            }

            return result;
        }
    }
}
