using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Stack arrays in sequence vertically (row wise).<br></br>
        ///     This is equivalent to concatenation along the first axis after 1-D arrays of shape(N,) have been reshaped to(1, N). Rebuilds arrays divided by vsplit.
        /// </summary>
        /// <param name="tup">The arrays must have the same shape along all but the first axis. 1-D arrays must have the same length.</param>
        /// <returns>The array formed by stacking the given arrays, will be at least 2-D.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html</remarks>
        public static NDArray vstack(params NDArray[] tup)
        {
            if (tup == null)
                throw new ArgumentNullException(nameof(tup));

            if (tup.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(tup));

            tup = np.atleast_2d(tup);
            return np.concatenate(tup, 0);
        }
    }
}
