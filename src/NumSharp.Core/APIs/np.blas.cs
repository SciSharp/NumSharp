using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Matrix or vector product between given NDArray and 2nd one.
        /// if both NDArrays are 1D, scalar product is returned independend of shape
        /// if both NDArrays are 2D matrix product is returned.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static NDArray dot(NDArray x, NDArray y)
            => BackendFactory.GetEngine().Dot(x, y);

        public static NDArray matmul(NDArray x, NDArray y)
            => BackendFactory.GetEngine().MatMul(x, y);
    }
}
