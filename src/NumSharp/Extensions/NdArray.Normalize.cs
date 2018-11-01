using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static void Normalize(this NDArray<double> np)
        {
            var min = np.Min();
            var max = np.Max();

            np.Data = np.Data.Select(data => {
                    double der = max - min;
                    return (data - min) / der;
            }).ToArray();
        }
    }
}
