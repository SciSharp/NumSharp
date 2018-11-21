using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArrayGeneric<double> Absolute(this NDArrayGeneric<double> np)
        {
            NDArrayGeneric<double> res = new NDArrayGeneric<double>
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
