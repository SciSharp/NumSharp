using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Stack arrays in sequence depth wise (along third axis).
        ///     This is equivalent to concatenation along the third axis after 2-D arrays of shape(M, N) have been reshaped to(M, N,1) and 1-D arrays of shape(N,) have been reshaped to(1, N,1).
        ///     Rebuilds arrays divided by dsplit. 
        ///     This function makes most sense for arrays with up to 3 dimensions.For instance, for pixel-data with a height(first axis), width(second axis), and r/g/b channels(third axis). The functions concatenate, stack and block provide more general stacking and concatenation operations.
        /// </summary>
        /// <param name="tup">The arrays must have the same shape along all but the third axis. 1-D or 2-D arrays must have the same shape.</param>
        /// <returns>The array formed by stacking the given arrays, will be at least 3-D.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html</remarks>
        public NDArray dstack(params NDArray[] tup)
        {
            if (tup == null)
                throw new ArgumentNullException(nameof(tup));

            NDArray[] list = new NDArray[1 + tup.Length];
            list[0] = this;
            tup.CopyTo(list, 1);
            return np.dstack(list);
        }
    }
}
