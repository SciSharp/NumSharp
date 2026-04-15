using System;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Test whether all array elements along a given axis evaluate to True.
        /// </summary>
        /// <param name="a">Input array or object that can be converted to an array.</param>
        /// <returns>True if all elements evaluate to True (non-zero).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.all.html
        public static bool all(NDArray a)
        {
            return a.TensorEngine.All(a);
        }

        /// <summary>
        ///     Test whether all array elements along a given axis evaluate to True.
        /// </summary>
        /// <param name="nd">Input array or object that can be converted to an array.</param>
        /// <param name="axis">Axis or axes along which a logical AND reduction is performed. The default (axis = None) is to perform a logical OR over all the dimensions of the input array. axis may be negative, in which case it counts from the last to the first axis.</param>
        /// <param name="keepdims">If True, the reduced axes are left in the result as dimensions with size one.</param>
        /// <returns>A new boolean ndarray is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.all.html
        public static NDArray<bool> all(NDArray nd, int axis, bool keepdims = false)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd), "Can't operate with null array");

            if (nd.TensorEngine is DefaultEngine defaultEngine)
                return defaultEngine.All(nd, axis, keepdims);

            var result = nd.TensorEngine.All(nd, axis);
            if (keepdims && nd.ndim > 0)
            {
                axis = DefaultEngine.NormalizeAxis(axis, nd.ndim);
                var dims = (long[])nd.shape.Clone();
                dims[axis] = 1;
                result.Storage.Reshape(new Shape(dims));
            }

return result;
        }
    }
}
