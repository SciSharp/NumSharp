using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core.Extensions
{
    public static partial class NumPyExtensions
    {
        public static NDArrayGeneric<double> min(this NDArrayGeneric<double> np)
        {
            return np.AMin();
        }
    }
}
