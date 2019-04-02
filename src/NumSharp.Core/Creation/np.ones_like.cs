using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray ones_like(NDArray nd, string order = "C")
        {
            return np.ones(new Shape(nd.shape));
        }
    }
}
