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
                // NumPy boolean indexing rules (from numpy/core/src/multiarray/mapping.c):
                // 1. 0-D boolean (scalar True/False): arr[True] adds axis, arr[False] empty with axis
                // 2. If mask.shape == arr.shape: element-wise selection, result is 1D
                // 3. If mask is 1D and mask.shape[0] == arr.shape[0]: select along axis 0
                // 4. If mask.shape == arr.shape[:mask.ndim]: partial match, use nonzero + fancy indexing
                // 5. Otherwise: error

                // Case 1: 0-D boolean (scalar True/False)
                // NumSharp represents scalars as shape [1], so check size == 1
                if (mask.size == 1 && mask.ndim == 1)
                {
                    return BooleanScalarIndex(mask.GetBoolean(0));
                }

                // Case 2: Full element masking (mask has same shape as array)
                if (mask.Shape.dimensions.SequenceEqual(this.Shape.dimensions))
                {
                    return this.TensorEngine.BooleanMask(this, mask);
                }

                // Case 3: Axis-0 selection (1D mask selecting along first axis)
                if (mask.ndim == 1 && mask.shape[0] == this.shape[0])
                {
                    return BooleanMaskAxis0(mask);
                }

                // Case 4: Partial shape match (mask.shape == arr.shape[:mask.ndim])
                // This is converted to fancy indexing via nonzero internally by NumPy
                if (mask.ndim < this.ndim && IsPartialShapeMatch(mask))
                {
                    return BooleanMaskPartialShape(mask);
                }

                // Error: mask doesn't match array shape
                throw new IndexOutOfRangeException(
                    $"boolean index did not match indexed array along axis 0; " +
                    $"size of axis is {this.shape[0]} but size of boolean index is {mask.shape[0]}");
            }
            set
            {
                // NumPy boolean indexing rules for setter
                // 1. 0-D boolean: not typically used for assignment
                // 2. If mask.shape == arr.shape: element-wise assignment
                // 3. If mask is 1D and mask.shape[0] == arr.shape[0]: assign along axis 0
                // 4. If mask.shape == arr.shape[:mask.ndim]: partial match assignment

                NumSharpException.ThrowIfNotWriteable(Shape);

                // Case 1: 0-D boolean - treat as axis-0 with single element
                if (mask.size == 1 && mask.ndim == 1)
                {
                    if (mask.GetBoolean(0))
                    {
                        np.copyto(this, value);
                    }
                    return;
                }

                // Case 2: Full element masking
                if (mask.Shape.dimensions.SequenceEqual(this.Shape.dimensions))
                {
                    var indices = np.nonzero(mask);
                    SetIndices(this, indices, value);
                    return;
                }

                // Case 3: Axis-0 selection
                if (mask.ndim == 1 && mask.shape[0] == this.shape[0])
                {
                    SetBooleanMaskAxis0(mask, value);
                    return;
                }

                // Case 4: Partial shape match
                if (mask.ndim < this.ndim && IsPartialShapeMatch(mask))
                {
                    SetBooleanMaskPartialShape(mask, value);
                    return;
                }

                throw new IndexOutOfRangeException(
                    $"boolean index did not match indexed array along axis 0; " +
                    $"size of axis is {this.shape[0]} but size of boolean index is {mask.shape[0]}");
            }
        }

        /// <summary>
        /// Checks if mask.shape matches arr.shape[:mask.ndim] (partial shape match).
        /// </summary>
        private bool IsPartialShapeMatch(NDArray mask)
        {
            for (int i = 0; i < mask.ndim; i++)
            {
                if (mask.shape[i] != this.shape[i])
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Handle 0-D boolean indexing (scalar True/False).
        /// arr[True] adds an axis: result shape (1,) + arr.shape
        /// arr[False] returns empty with extra axis: result shape (0,) + arr.shape
        /// </summary>
        private NDArray BooleanScalarIndex(bool value)
        {
            // Build new shape: (1 or 0,) + this.shape
            var newShape = new long[this.ndim + 1];
            newShape[0] = value ? 1 : 0;
            for (int i = 0; i < this.ndim; i++)
                newShape[i + 1] = this.shape[i];

            if (value)
            {
                // True: return array with extra axis, containing all data
                // This is like arr[np.newaxis]
                return this.reshape(newShape);
            }
            else
            {
                // False: return empty array with the extra axis shape
                return new NDArray(this.dtype, new Shape(newShape));
            }
        }

        /// <summary>
        /// Handle partial shape match: mask.shape == arr.shape[:mask.ndim]
        /// Internally uses nonzero + fancy indexing (same as NumPy).
        /// Result shape: (count_true,) + arr.shape[mask.ndim:]
        /// </summary>
        private NDArray BooleanMaskPartialShape(NDArray<bool> mask)
        {
            // Get nonzero indices for each dimension of the mask
            var indices = np.nonzero(mask);

            // Count true values
            long trueCount = indices[0].size;

            if (trueCount == 0)
            {
                // Empty result with shape (0,) + arr.shape[mask.ndim:]
                var emptyShape = new long[1 + this.ndim - mask.ndim];
                emptyShape[0] = 0;
                for (int i = 0; i < this.ndim - mask.ndim; i++)
                    emptyShape[i + 1] = this.shape[mask.ndim + i];
                return new NDArray(this.dtype, new Shape(emptyShape));
            }

            // Build result shape: (trueCount,) + arr.shape[mask.ndim:]
            var resultShape = new long[1 + this.ndim - mask.ndim];
            resultShape[0] = trueCount;
            for (int i = 0; i < this.ndim - mask.ndim; i++)
                resultShape[i + 1] = this.shape[mask.ndim + i];

            var result = new NDArray(this.dtype, new Shape(resultShape));

            // Copy selected slices using the nonzero indices
            for (long idx = 0; idx < trueCount; idx++)
            {
                // Build the index tuple from nonzero results
                var srcSlice = this;
                for (int dim = 0; dim < mask.ndim; dim++)
                {
                    srcSlice = srcSlice[indices[dim].GetInt64(idx)];
                }
                np.copyto(result[idx], srcSlice);
            }

            return result;
        }

        /// <summary>
        /// Assignment for partial shape match.
        /// </summary>
        private void SetBooleanMaskPartialShape(NDArray<bool> mask, NDArray value)
        {
            var indices = np.nonzero(mask);
            long trueCount = indices[0].size;

            if (trueCount == 0)
                return;

            bool isScalarValue = value.size == 1;

            for (long idx = 0; idx < trueCount; idx++)
            {
                // Navigate to the target slice using nonzero indices
                var destSlice = this;
                for (int dim = 0; dim < mask.ndim; dim++)
                {
                    destSlice = destSlice[indices[dim].GetInt64(idx)];
                }

                if (isScalarValue)
                {
                    np.copyto(destSlice, value);
                }
                else if (value.ndim == this.ndim - mask.ndim)
                {
                    np.copyto(destSlice, value[idx]);
                }
                else
                {
                    np.copyto(destSlice, value);
                }
            }
        }

        /// <summary>
        /// Boolean masking along axis 0 (row selection for 2D, etc).
        /// </summary>
        private NDArray BooleanMaskAxis0(NDArray<bool> mask)
        {
            // Count true values
            long trueCount = 0;
            for (long i = 0; i < mask.size; i++)
            {
                if (mask.GetBoolean(i))
                    trueCount++;
            }

            if (trueCount == 0)
            {
                // Return empty array with appropriate shape
                // For 2D array with shape (n, m), result should be shape (0, m)
                var emptyShape = new long[this.ndim];
                emptyShape[0] = 0;
                for (int i = 1; i < this.ndim; i++)
                    emptyShape[i] = this.shape[i];
                return new NDArray(this.dtype, new Shape(emptyShape));
            }

            // Build result shape: [trueCount, shape[1], shape[2], ...]
            var resultShape = new long[this.ndim];
            resultShape[0] = trueCount;
            for (int i = 1; i < this.ndim; i++)
                resultShape[i] = this.shape[i];

            var result = new NDArray(this.dtype, new Shape(resultShape));

            // Copy selected slices
            long destIdx = 0;
            for (long srcIdx = 0; srcIdx < mask.size; srcIdx++)
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
            // Detect scalar-like values (size == 1)
            // NumSharp represents scalars as shape [1], not shape []
            bool isScalarValue = value.size == 1;

            long valueIdx = 0;
            for (long i = 0; i < mask.size; i++)
            {
                if (mask.GetBoolean(i))
                {
                    var destSlice = this[i];
                    if (isScalarValue)
                    {
                        // Scalar broadcast - value.size == 1
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
