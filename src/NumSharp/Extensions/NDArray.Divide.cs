using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<NDArray<double>> Divide(this NDArray<NDArray<double>> np, double divisor)
        {
            return new NDArray<NDArray<double>>
            {
                Data = np.Data.Select(row => new NDArray<double>
                {
                    Data = row.Data.Select(col => col / divisor).ToList()
                }).ToList()
            };
        }
    }
}
