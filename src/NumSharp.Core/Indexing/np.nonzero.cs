using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {

        /// <summary>
        /// Return the indices of the elements that are non-zero.
        /// 
        /// Returns a tuple of arrays, one for each dimension of a, containing the indices of the non-zero elements in that dimension.The values in a are always tested and returned in row-major, C-style order.
        ///
        /// To group the indices by element, rather than dimension, use argwhere, which returns a row for each non-zero element.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static NDArray nonzero(in NDArray x)
            => throw new NotImplementedException("Implement nonzero"); //x.TensorEngine.Nonzero(x);

    }
}
