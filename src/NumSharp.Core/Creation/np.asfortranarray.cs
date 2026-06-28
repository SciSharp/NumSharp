using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return an array (ndim >= 1) laid out in Fortran order in memory.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="dtype">By default, the data-type is inferred from the input.</param>
        /// <returns>The input a in Fortran, or column-major, order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asfortranarray.html</remarks>
        public static NDArray asfortranarray(NDArray a, Type dtype = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return asarray(a, dtype, 'F');
        }
    }
}
