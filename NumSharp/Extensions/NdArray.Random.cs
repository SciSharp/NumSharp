using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NdArrayExtensions
    {
        public static NdArrayRandom Random<TData>(this NdArray<TData> np)
        {
            var rand = new NdArrayRandom();
            return rand;
        }
    }
}
