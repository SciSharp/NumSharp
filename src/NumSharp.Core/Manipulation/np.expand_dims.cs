using System;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray expand_dims(NDArray a, int axis)
        {
            // Only the uninitialized-shape sentinel is passed through. A genuine empty array
            // (size == 0 but with real dimensions, e.g. (0, 3)) MUST still gain the new axis —
            // NumPy: expand_dims((0, 3), 0) -> (1, 0, 3). Guarding on a.size == 0 here made
            // expand_dims a no-op for every empty array (and, transitively, broke np.stack and
            // np.atleast_2d / np.vstack, which build on it).
            if (a.Shape.IsEmpty)
                return a;

            return new NDArray(a.Storage.Alias(a.Shape.ExpandDimension(axis))) { TensorEngine = a.TensorEngine };
        }

        /// <summary>
        ///     Expand the shape of an array. Insert new axes that will appear at
        ///     the <paramref name="axis"/> positions in the expanded output.
        /// </summary>
        /// <remarks>
        ///     Matches NumPy 2.x: each axis in the tuple is normalized against
        ///     the FINAL output ndim (<c>a.ndim + axis.Length</c>). Duplicate
        ///     normalized positions raise <see cref="ArgumentException"/>
        ///     ("repeated axis"); out-of-range axes throw the same.
        ///     Empty <paramref name="axis"/> returns the input unchanged.
        /// </remarks>
        public static NDArray expand_dims(NDArray a, int[] axis)
        {
            if (axis == null || axis.Length == 0)
                return a;

            // See the single-axis overload: a size == 0 array must still gain the new axes.
            if (a.Shape.IsEmpty)
                return a;

            return new NDArray(a.Storage.Alias(a.Shape.ExpandDimensions(axis))) { TensorEngine = a.TensorEngine };
        }

        /// <summary>
        ///     Sequence overload — accepts any <see cref="IEnumerable{Int32}"/>,
        ///     materializes to an array, and delegates to the tuple-axis path.
        /// </summary>
        public static NDArray expand_dims(NDArray a, IEnumerable<int> axis)
        {
            if (axis == null)
                return a;

            return expand_dims(a, axis as int[] ?? axis.ToArray());
        }
    }
}
