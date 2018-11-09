using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<double> Sqrt(this NDArray<double> np)
        {
            for (int i = 0; i < np.Data.Length; i++)
            {
                np[i] = Math.Sqrt(np[i]);
            }

            return np;
        }
    }
}