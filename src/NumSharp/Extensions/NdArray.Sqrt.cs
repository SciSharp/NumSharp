using System;
using System.Collections.Generic;
using System.Numerics;
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

        public static NDArray<Complex> Sqrt(this NDArray<Complex> np)
        {
            for (int i = 0; i < np.Data.Length; i++)
            {
                np[i] = Complex.Sqrt(np[i]);
            }

            return np;
        }
    }
}