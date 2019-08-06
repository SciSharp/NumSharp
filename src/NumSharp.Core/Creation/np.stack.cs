using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Join a sequence of arrays along a new axis.
        ///     The axis parameter specifies the index of the new axis in the dimensions of the result.
        ///     For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last dimension.
        /// </summary>
        /// <param name="arrays">Each array must have the same shape.</param>
        /// <param name="axis">The axis in the result array along which the input arrays are stacked.</param>
        /// <returns>The stacked array has one more dimension than the input arrays.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.stack.html</remarks>
        public static NDArray stack(NDArray[] arrays, int axis = 0)
        {
            if (arrays == null)
                throw new ArgumentNullException(nameof(arrays));

            if (arrays.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(arrays));

            arrays = np.atleast_1d(arrays); //handle scalars
            var first = arrays[0];
            arrays[0] = np.expand_dims(first, axis);
            for (int i = 1; i < arrays.Length; i++)
            {
                var curr = arrays[i];
                if (curr.Shape != first.Shape)
                    throw new InvalidOperationException("all input arrays must have the same shape");
                arrays[i] = np.expand_dims(curr, axis);
            }

            return np.concatenate(arrays, axis);
        }
    }
}
