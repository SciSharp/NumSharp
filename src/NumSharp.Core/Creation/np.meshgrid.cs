using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core.Creation
{
    static partial class np
    {
        /// <summary>
        /// Return coordinate matrices from coordinate vectors.
        /// Make N-D coordinate arrays for vectorized evaluations of
        /// N-D scalar/vector fields over N-D grids, given
        /// one-dimensional coordinate arrays x1, x2,..., xn.
        /// .. versionchanged:: 1.9
        /// 1-D and 0-D cases are allowed.
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <returns></returns>
        public static NDArray meshgrid(NDArray x1, NDArray x2)
        {
            throw new NotFiniteNumberException();
        }
    }
}
