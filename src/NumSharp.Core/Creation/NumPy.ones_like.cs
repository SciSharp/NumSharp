using NumSharp.Core.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public static partial class NumPyExtensions
    {
        public static NDArrayGeneric<T> ones_like<T>(this NumPyGeneric<T> np, NDArrayGeneric<T> nd, string order = "C")
        {
            return np.ones(new Shape(nd.Shape.Shapes));
        }
    }
}
