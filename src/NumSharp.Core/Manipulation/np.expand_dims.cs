using System;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray expand_dims(NDArray a, int axis)
        {
            //test if the ndarray is empty.
            if (a.size == 0 || a.Shape.IsEmpty)
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

            if (a.size == 0 || a.Shape.IsEmpty)
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
