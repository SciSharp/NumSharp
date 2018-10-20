using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static int ArgMax(this NDArray<double> np )
        {
            var max = np.Data.Max();

            return np.Data.IndexOf(max);
        }
        public static int ArgMax(this NDArray<int> np )
        {
            var max = np.Data.Max();

            return np.Data.IndexOf(max);
        }
    }
}
