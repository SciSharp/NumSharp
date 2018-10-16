using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArrayRandom Random<TData>(this NDArray<TData> np)
        {
            var rand = new NDArrayRandom();
            return rand;
        }
    }
}
