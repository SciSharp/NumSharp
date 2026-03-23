using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Numerical positive, element-wise.
        /// This is an identity operation - returns +x (a copy of the input).
        /// Equivalent to np.array(a, copy=True).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.positive.html</remarks>
        public NDArray positive()
        {
            // np.positive is the identity function: +x == x
            // It returns a copy of the array, preserving all values as-is
            return this.Clone();
        }
    }
}
