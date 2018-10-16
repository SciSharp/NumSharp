using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<T> Unique<T>(this NDArray<T> np)
        {
            var list = np.Data.Distinct();

            return new NDArray<T>().Array(list);
        }
    }
}
