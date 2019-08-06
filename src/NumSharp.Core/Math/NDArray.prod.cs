using System;
using System.Diagnostics.CodeAnalysis;

namespace NumSharp
{
    [SuppressMessage("ReSharper", "ParameterHidesMember")]
    public partial class NDArray
    {
        /// <summary>
        ///     Return the product of array elements over a given axis.
        /// </summary>
        /// <param name="axis">Axis or axes along which a product is performed. The default, axis=None, will calculate the product of all the elements in the input array. If axis is negative it counts from the last to the first axis.</param>
        /// <param name="dtype">The type of the returned array, as well as of the accumulator in which the elements are multiplied. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.</param>
        /// <returns>An array shaped as a but with the specified axis removed.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.prod.html</remarks>
        public NDArray prod(int? axis = null, Type dtype = null, bool keepdims = false)
            => np.prod(this, axis, dtype, keepdims);

    }
}
