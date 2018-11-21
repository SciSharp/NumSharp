using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArrayGeneric<T> Unique<T>(this NDArrayGeneric<T> np)
        {
            var np2 = new NDArrayGeneric<T>();
            np2.Data = np.Data.Distinct().ToArray();
            np2.Shape = new Shape(new int[] { np2.Data.Length });

            return np2;
        }
    }
}
