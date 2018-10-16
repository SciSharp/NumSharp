using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static void Normalize(this NDArray<NDArray<double>> np)
        {
            var min = np.Min();
            var max = np.Max();

            np.Data.ToList().ForEach(data => {
                for(int d = 0; d < np.NDim; d++)
                {
                    double der = max.Data[d] - min.Data[d];

                    data.Data[d] = (data.Data[d] - min.Data[d]) / der;
                }
            });
        }
    }
}
