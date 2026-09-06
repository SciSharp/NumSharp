using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Returns True if all elements evaluate to True. Refer to <see cref="np.all(NDArray, int?, NDArray, bool, NDArray)"/>
        ///     for full documentation.
        /// </summary>
        /// <param name="axis">Axis or axes along which a logical AND reduction is performed. The default (null) is to reduce over the flattened array.</param>
        /// <param name="out">Alternate output array in which to place the result. It must have the same shape as the expected output.</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one.</param>
        /// <param name="where">Elements to include in the reduction (null means include all).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.all.html</remarks>
        public NDArray all(int? axis = null, NDArray @out = null, bool keepdims = false, NDArray @where = null)
            => np.all(this, axis, @out, keepdims, @where);
    }
}
