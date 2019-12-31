using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Diagnostics;
using NumSharp.Backends;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the indices of the elements that are non-zero.
        ///     Returns a tuple of arrays, one for each dimension of a, containing the indices of the non-zero elements in that dimension.The values in a are always tested and returned in row-major, C-style order.
        ///     To group the indices by element, rather than dimension, use argwhere, which returns a row for each non-zero element.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns>Indices of elements that are non-zero.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html</remarks>
        public static NDArray<int>[] nonzero(in NDArray a)
            => a.TensorEngine.NonZero(a);
    }
}
