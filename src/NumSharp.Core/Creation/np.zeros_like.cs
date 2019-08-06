using System;

namespace NumSharp
{
    public static partial class np
    {
        ///  <summary>
        ///     Return an array of zeros with the same shape and type as a given array.
        ///  </summary>
        /// <param name="a">The shape and data-type of a define these same attributes of the returned array.</param>
        ///  <param name="dtype">Overrides the data type of the result.</param>
        ///  <returns>Array of zeros with the same shape and type as `nd`.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros_like.html</remarks>
        public static NDArray zeros_like(NDArray a, Type dtype = null)
        {
            return np.zeros(new Shape(a.shape), dtype ?? a.dtype);
        }
    }
}
