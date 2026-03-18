using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Used to perform selection based on a boolean mask.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/user/basics.indexing.html</remarks>
        /// <exception cref="IndexOutOfRangeException">When one of the indices exceeds limits.</exception>
        /// <exception cref="ArgumentException">indices must be of Int type (byte, u/short, u/int, u/long).</exception>
        [SuppressMessage("ReSharper", "CoVariantArrayConversion")]
        public NDArray this[NDArray<bool> mask]
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
                    return this.TensorEngine.BooleanMask(this, mask);
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
    }
}
