using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a contiguous array (ndim >= 1) in memory (C order).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="dtype">By default, the data-type is inferred from the input.</param>
        /// <returns>Contiguous array of same shape and content as a, with type dtype if specified.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html</remarks>
        public static NDArray ascontiguousarray(NDArray a, Type dtype = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return asarray(a, dtype, 'C');
        }
    }
}
