using System;

namespace NumSharp
{
    public static partial class np
    {
        ///  <summary>
        ///     Return a new array with the same shape and type as a given array.
        ///  </summary>
        /// <param name="prototype">The shape and data-type of prototype define these same attributes of the returned array.</param>
        ///  <param name="dtype">Overrides the data type of the result.</param>
        ///  <returns>Array of uninitialized (arbitrary) data with the same shape and type as prototype.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.empty_like.html</remarks>
        public static NDArray empty_like(NDArray prototype, Type dtype = null)
        {
            return new NDArray(dtype ?? prototype.dtype, new Shape(prototype.shape), false);
        }
    }
}
