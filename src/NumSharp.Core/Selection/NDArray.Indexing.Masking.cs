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
                // NumPy boolean indexing (numpy/_core/src/multiarray/mapping.c):
                // the mask matches a leading PREFIX of arr's shape — mask.ndim <= arr.ndim
                // and mask.shape == arr.shape[:mask.ndim]. For each True (in C-order) the
                // trailing sub-tensor arr.shape[mask.ndim:] is gathered. Result:
                // (count_true,) + arr.shape[mask.ndim:]. This single prefix-match case covers
                // them all through the unified NDIter gather (DefaultEngine.BooleanMask):
                //   • full element mask (mask.ndim == arr.ndim)         -> 1-D result;
                //   • axis-0 row mask / partial mask (mask.ndim < ndim) -> (count,)+trailing;
                //   • 0-D mask (arr[True]/arr[False], mask.ndim == 0)    -> (1,)/(0,)+arr.shape.
                // A 1-D length-1 mask is a NORMAL mask (not a 0-D scalar), so it is NOT
                // special-cased — NumSharp represents true 0-D arrays as ndim 0.
                if (mask.ndim <= this.ndim && IsPartialShapeMatch(mask))
                {
                    return this.TensorEngine.BooleanMask(this, mask);
                }

                // Error: mask doesn't match a leading prefix of the array shape.
                throw new IndexOutOfRangeException(
                    $"boolean index did not match indexed array along axis 0; " +
                    $"size of axis is {this.shape[0]} but size of boolean index is {mask.shape[0]}");
            }
            set
            {
                NumSharpException.ThrowIfNotWriteable(Shape);

                // Prefix-match (full / axis-0 / partial / 0-D), symmetric with the getter.
                // The unified NDIter scatter streams value into the selected slots; value
                // broadcasts to the selection shape (count,)+arr.shape[mask.ndim:] by NumPy
                // rules (raises on an incompatible value). A 1-D length-1 mask is a normal
                // mask; a 0-D mask (arr[True]=v / arr[False]=v) assigns all / nothing.
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
        /// Checks if mask.shape matches arr.shape[:mask.ndim] (partial shape match). A mask axis
        /// of length 0 is EXEMPT from the match — a zero-length boolean axis selects nothing and is
        /// valid against an axis of any size (NumPy: A[np.zeros(0,bool)] -> (0,)+trailing on a
        /// size-3 axis, A[np.zeros((3,0),bool)] -> (0,)); only a NON-zero mask axis must equal the
        /// array axis (A[np.zeros((0,2),bool)] still raises on the mismatched size-2 axis).
        /// </summary>
        private bool IsPartialShapeMatch(NDArray mask)
        {
            for (int i = 0; i < mask.ndim; i++)
            {
                if (mask.shape[i] != this.shape[i] && mask.shape[i] != 0)
                    return false;
            }
            return true;
        }
    }
}
