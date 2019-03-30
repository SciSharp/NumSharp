using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray mean(NDArray np, int axis = -1)
        {
            return np.mean(axis);
        }
    }
}
