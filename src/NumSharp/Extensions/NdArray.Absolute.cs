using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<double> Absolute(this NDArray<double> np)
        {
            NDArray<double> res = new NDArray<double>
            {
                Shape = np.Shape,
                Data = new double[np.Size]
            };
            for (int i = 0; i < np.Size; i++)
            {
                res.Data[i] = Math.Abs(np.Data[i]);
            }           
            return res;
        }
    }
}
