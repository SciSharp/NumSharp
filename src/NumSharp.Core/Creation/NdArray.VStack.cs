using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Stack arrays in sequence vertically (row wise).<br></br>
        ///     This is equivalent to concatenation along the first axis after 1-D arrays of shape(N,) have been reshaped to(1, N). Rebuilds arrays divided by vsplit.
        /// </summary>
        /// <typeparam name="T">The type dtype to return.</typeparam>
        /// <param name="tup">The arrays must have the same shape along all but the first axis. 1-D arrays must have the same length.</param>
        /// <returns>https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html</returns>
        public NDArray vstack<T>(params NDArray[] tup) where T : unmanaged
        {
            return np.vstack<T>(this.Yield().Concat(tup).ToArray());
        }
    }
}
