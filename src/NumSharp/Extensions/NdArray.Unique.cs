using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray_Legacy<T> Unique<T>(this NDArray_Legacy<T> np)
        {
            var list = np.Data.Distinct();

            return new NDArray_Legacy<T>().Array(list);
        }
    }
}
