using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NumPyExtensions
    {
        public static NDArray<double> min(this NDArray<double> np)
        {
            return np.AMin();
        }
    }
}
