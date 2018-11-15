using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class NumPyExtensions
    {
        public static NDArray<T> ones_like<T>(this NumPy<T> np, NDArray<T> nd, string order = "C")
        {
            return np.ones(new Shape(nd.Shape.Shapes));
        }
    }
}
