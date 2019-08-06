using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Stack arrays in sequence horizontally (column wise).
        ///     This is equivalent to concatenation along the second axis, except for 1-D arrays where it concatenates along the first axis.Rebuilds arrays divided by hsplit.
        ///     This function makes most sense for arrays with up to 3 dimensions.For instance, for pixel-data with a height(first axis), width(second axis), 
        ///     and r/g/b channels(third axis). The functions concatenate, stack and block provide more general stacking and concatenation operations.
        /// </summary>
        /// <param name="tup">The arrays must have the same shape along all but the second axis, except 1-D arrays which can be any length.</param>
        /// <returns>The array formed by stacking the given arrays.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html</remarks>
        public static NDArray hstack(params NDArray[] tup)
        {
            if (tup == null)
                throw new ArgumentNullException(nameof(tup));

            if (tup.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(tup));

            tup = np.atleast_1d(tup); //handle scalars
            if (tup[0].shape.Length == 1)
                return np.concatenate(tup, 0);
            else
                return np.concatenate(tup, 1);
        }
    }
}
