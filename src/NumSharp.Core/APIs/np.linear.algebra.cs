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
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static NDArray dot(NDArray a, NDArray b)
            => BackendFactory.GetEngine().Dot(a, b);
    }
}
