using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<double> Mean(this NDArray<double> np)
        {
            double mean = (np.Data.Sum() ) / np.Data.Count;

            var numSharpMean = new NDArray<double>();
            np.Zeros(1);
            numSharpMean.Data = new double[1] { mean };

            return numSharpMean;
        }
        public static NDArray<double> Mean(this NDArray<int> np)
        {
            double mean = (np.Data.Sum() ) / np.Data.Count;

            var numSharpMean = new NDArray<double>();
            np.Zeros(1);
            numSharpMean.Data = new double[1] { mean };

            return numSharpMean;
        }
    }
}
