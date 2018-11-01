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
            var np2 = new NDArray<T>();
            np2.Data = np.Data.Distinct().ToArray();
            np2.Shape[0] = np2.Data.Length;

            return np2;
        }
    }
}
