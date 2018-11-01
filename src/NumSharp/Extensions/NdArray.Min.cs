using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static int Min(this NDArray<double> np)
        {
            var val = np.Data.Min();

            return np.Data.ToList().IndexOf(val);
        }
    }
}
