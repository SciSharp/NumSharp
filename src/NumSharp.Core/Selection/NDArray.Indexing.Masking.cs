using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Used to perform selection based on a boolean mask.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.17.0/user/basics.indexing.html</remarks>
        /// <exception cref="IndexOutOfRangeException">When one of the indices exceeds limits.</exception>
        /// <exception cref="ArgumentException">indices must be of Int type (byte, u/short, u/int, u/long).</exception>
        [SuppressMessage("ReSharper", "CoVariantArrayConversion")]
        public unsafe NDArray this[NDArray<bool> mask]
        {
            get
            {
                // NumPy boolean indexing rules:
                // 1. If mask.shape == arr.shape: element-wise selection, result is 1D
                // 2. If mask is 1D and mask.shape[0] == arr.shape[0]: select along axis 0
                // 3. Otherwise: error

                // Case 1: Full element masking (mask has same shape as array)
                if (mask.Shape.dimensions.SequenceEqual(this.Shape.dimensions))
                {
                    // SIMD fast path: contiguous arrays of same size
                    var kp = DefaultEngine.DefaultKernelProvider;
                    if (kp.Enabled && kp.VectorBits > 0 &&
                        mask.Shape.IsContiguous && this.Shape.IsContiguous)
                    {
                        return BooleanMaskFastPath(mask);
                    }

                    // Fallback: use nonzero + fancy indexing
                    return FetchIndices(this, np.nonzero(mask), null, true);
                }

                // Case 2: Axis-0 selection (1D mask selecting along first axis)
                if (mask.ndim == 1 && mask.shape[0] == this.shape[0])
                {
                    return BooleanMaskAxis0(mask);
                }

                // Error: mask doesn't match array shape
                throw new IndexOutOfRangeException(
                    $"boolean index did not match indexed array along axis 0; " +
                    $"size of axis is {this.shape[0]} but size of boolean index is {mask.shape[0]}");
            }
            set
            {
                // NumPy boolean indexing rules for setter
                // 1. If mask.shape == arr.shape: element-wise assignment
                // 2. If mask is 1D and mask.shape[0] == arr.shape[0]: assign along axis 0

                NumSharpException.ThrowIfNotWriteable(Shape);

                // Case 1: Full element masking
                if (mask.Shape.dimensions.SequenceEqual(this.Shape.dimensions))
                {
                    var indices = np.nonzero(mask);
                    SetIndices(this, indices, value);
                    return;
                }

                // Case 2: Axis-0 selection
                if (mask.ndim == 1 && mask.shape[0] == this.shape[0])
                {
                    SetBooleanMaskAxis0(mask, value);
                    return;
                }

                throw new IndexOutOfRangeException(
                    $"boolean index did not match indexed array along axis 0; " +
                    $"size of axis is {this.shape[0]} but size of boolean index is {mask.shape[0]}");
            }
        }

        /// <summary>
        /// Boolean masking along axis 0 (row selection for 2D, etc).
        /// </summary>
        private NDArray BooleanMaskAxis0(NDArray<bool> mask)
        {
            // Count true values
            int trueCount = 0;
            for (int i = 0; i < mask.size; i++)
            {
                if (mask.GetBoolean(i))
                    trueCount++;
            }

            if (trueCount == 0)
            {
                // Return empty array with appropriate shape
                // For 2D array with shape (n, m), result should be shape (0, m)
                var emptyShape = new int[this.ndim];
                emptyShape[0] = 0;
                for (int i = 1; i < this.ndim; i++)
                    emptyShape[i] = this.shape[i];
                return new NDArray(this.dtype, new Shape(emptyShape));
            }

            // Build result shape: [trueCount, shape[1], shape[2], ...]
            var resultShape = new int[this.ndim];
            resultShape[0] = trueCount;
            for (int i = 1; i < this.ndim; i++)
                resultShape[i] = this.shape[i];

            var result = new NDArray(this.dtype, new Shape(resultShape));

            // Copy selected slices
            int destIdx = 0;
            for (int srcIdx = 0; srcIdx < mask.size; srcIdx++)
            {
                if (mask.GetBoolean(srcIdx))
                {
                    // Get slice at index srcIdx and copy to result at destIdx
                    var srcSlice = this[srcIdx];
                    var destSlice = result[destIdx];
                    np.copyto(destSlice, srcSlice);
                    destIdx++;
                }
            }

            return result;
        }

        /// <summary>
        /// Boolean masking setter along axis 0.
        /// </summary>
        private void SetBooleanMaskAxis0(NDArray<bool> mask, NDArray value)
        {
            int valueIdx = 0;
            for (int i = 0; i < mask.size; i++)
            {
                if (mask.GetBoolean(i))
                {
                    var destSlice = this[i];
                    if (value.ndim == 0)
                    {
                        // Scalar broadcast
                        np.copyto(destSlice, value);
                    }
                    else if (value.ndim == this.ndim - 1)
                    {
                        // Each mask position gets a row from value
                        np.copyto(destSlice, value[valueIdx]);
                        valueIdx++;
                    }
                    else
                    {
                        // Broadcast value to destination
                        np.copyto(destSlice, value);
                    }
                }
            }
        }

        /// <summary>
        /// SIMD-optimized boolean masking for contiguous arrays.
        /// </summary>
        private unsafe NDArray BooleanMaskFastPath(NDArray<bool> mask)
        {
            int size = this.size;
            var kp = DefaultEngine.DefaultKernelProvider;

            // Count true values using SIMD
            int trueCount = kp.CountTrue((bool*)mask.Address, size);

            if (trueCount == 0)
                return new NDArray(this.dtype, Shape.Empty(1)); // Empty 1D result

            // Create result array
            var result = new NDArray(this.dtype, new Shape(trueCount));

            // Copy elements where mask is true
            switch (this.typecode)
            {
                case NPTypeCode.Boolean:
                    kp.CopyMasked((bool*)this.Address, (bool*)mask.Address, (bool*)result.Address, size);
                    break;
                case NPTypeCode.Byte:
                    kp.CopyMasked((byte*)this.Address, (bool*)mask.Address, (byte*)result.Address, size);
                    break;
                case NPTypeCode.Int16:
                    kp.CopyMasked((short*)this.Address, (bool*)mask.Address, (short*)result.Address, size);
                    break;
                case NPTypeCode.UInt16:
                    kp.CopyMasked((ushort*)this.Address, (bool*)mask.Address, (ushort*)result.Address, size);
                    break;
                case NPTypeCode.Int32:
                    kp.CopyMasked((int*)this.Address, (bool*)mask.Address, (int*)result.Address, size);
                    break;
                case NPTypeCode.UInt32:
                    kp.CopyMasked((uint*)this.Address, (bool*)mask.Address, (uint*)result.Address, size);
                    break;
                case NPTypeCode.Int64:
                    kp.CopyMasked((long*)this.Address, (bool*)mask.Address, (long*)result.Address, size);
                    break;
                case NPTypeCode.UInt64:
                    kp.CopyMasked((ulong*)this.Address, (bool*)mask.Address, (ulong*)result.Address, size);
                    break;
                case NPTypeCode.Char:
                    kp.CopyMasked((char*)this.Address, (bool*)mask.Address, (char*)result.Address, size);
                    break;
                case NPTypeCode.Single:
                    kp.CopyMasked((float*)this.Address, (bool*)mask.Address, (float*)result.Address, size);
                    break;
                case NPTypeCode.Double:
                    kp.CopyMasked((double*)this.Address, (bool*)mask.Address, (double*)result.Address, size);
                    break;
                case NPTypeCode.Decimal:
                    kp.CopyMasked((decimal*)this.Address, (bool*)mask.Address, (decimal*)result.Address, size);
                    break;
                default:
                    throw new NotSupportedException($"Type {this.typecode} not supported for boolean masking");
            }

            return result;
        }
    }
}
