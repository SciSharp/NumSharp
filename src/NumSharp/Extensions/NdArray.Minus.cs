using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<NDArray<double>> Minus(this NDArray<NDArray<double>> np, double minus)
        {
            return new NDArray<NDArray<double>>
            {
                Data = np.Data.Select(row => new NDArray<double>
                {
                    Data = row.Data.Select(col => col - minus).ToList()
                }).ToList()
            };
        }
    }
}
