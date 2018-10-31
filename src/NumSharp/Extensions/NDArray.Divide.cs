using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray_Legacy<NDArray_Legacy<double>> Divide(this NDArray_Legacy<NDArray_Legacy<double>> np, double divisor)
        {
            return new NDArray_Legacy<NDArray_Legacy<double>>
            {
                Data = np.Data.Select(row => new NDArray_Legacy<double>
                {
                    Data = row.Data.Select(col => col / divisor).ToList()
                }).ToList()
            };
        }
    }
}
