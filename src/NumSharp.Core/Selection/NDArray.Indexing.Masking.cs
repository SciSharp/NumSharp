using System;
using System.Diagnostics.CodeAnalysis;
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
                // 2. mask.shape == arr.shape[:mask.ndim] (prefix match, mask.ndim <= arr.ndim):
                //    gather sub-tensors where True; result (count,) + arr.shape[mask.ndim:].
                //    This single case covers full element masks, axis-0 row masks, and partial
                //    masks — all routed through the unified NpyIter gather (DefaultEngine).
                // 3. Otherwise: error

                // Case 1: 0-D boolean (scalar True/False)
                // NumSharp represents scalars as shape [1], so check size == 1
                if (mask.size == 1 && mask.ndim == 1)
                {
                    return BooleanScalarIndex(mask.GetBoolean(0));
                }

                // Case 2: prefix-shape match (full / axis-0 / partial)
                if (mask.ndim <= this.ndim && IsPartialShapeMatch(mask))
                {
                    return this.TensorEngine.BooleanMask(this, mask);
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

                // Case 2: prefix-shape match (full / axis-0 / partial) — unified
                // NpyIter scatter streams value into the selected slots (DefaultEngine).
                if (mask.ndim <= this.ndim && IsPartialShapeMatch(mask))
                {
                    this.TensorEngine.BooleanMaskSet(this, mask, value);
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
    }
}
