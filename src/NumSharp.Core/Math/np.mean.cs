using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public partial class np
    {
        public static NDArray mean(NDArray np, int axis = -1)
        {
            return np.mean(axis);
        }
    }
}
