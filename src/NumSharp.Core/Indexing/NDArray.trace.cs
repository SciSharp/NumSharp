using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return the sum along diagonals of the array. Refer to <see cref="np.trace(NDArray, int, int, int, Type, NDArray)"/>
        ///     for full documentation.
        /// </summary>
        /// <param name="offset">Offset of the diagonal from the main diagonal. Can be positive or negative. Defaults to 0.</param>
        /// <param name="axis1">First axis of the 2-D sub-arrays from which the diagonals are taken. Defaults to 0.</param>
        /// <param name="axis2">Second axis of the 2-D sub-arrays from which the diagonals are taken. Defaults to 1.</param>
        /// <param name="dtype">Output dtype. null (default) preserves the dtype, promoting integer dtypes narrower than int64 to int64.</param>
        /// <param name="out">Optional output array whose shape must equal the natural reduction output.</param>
        /// <returns>The sum along the diagonals. For a 2-D array this is a 0-d scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.trace.html</remarks>
        public NDArray trace(int offset = 0, int axis1 = 0, int axis2 = 1, Type dtype = null, NDArray @out = null)
            => np.trace(this, offset, axis1, axis2, dtype, @out);
    }
}
