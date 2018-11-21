using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace NumSharp.Core.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArrayGeneric<double> Sqrt(this NDArrayGeneric<double> np)
        {
            for (int i = 0; i < np.Data.Length; i++)
            {
                np[i] = Math.Sqrt(np[i]);
            }

            return np;
        }

        public static NDArrayGeneric<Complex> Sqrt(this NDArrayGeneric<Complex> np)
        {
            for (int i = 0; i < np.Data.Length; i++)
            {
                np[i] = Complex.Sqrt(np[i]);
            }

            return np;
        }
    }
}