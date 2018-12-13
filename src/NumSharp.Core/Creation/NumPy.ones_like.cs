using NumSharp.Core.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public static partial class NumPyExtensions
    {
        public static NDArray ones_like(this NumPy np, NDArray nd, string order = "C")
        {
            return np.ones(new Shape(nd.shape.Shapes));
        }
    }
}
