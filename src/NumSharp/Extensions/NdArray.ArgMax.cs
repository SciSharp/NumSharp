using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NdArrayExtensions
    {
        public static int ArgMax(this NdArray<double> np)
        {
            var max = np.Data.Max();

            return np.Data.IndexOf(max);
        }
    }
}
