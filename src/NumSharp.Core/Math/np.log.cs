using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Extensions;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray log(NDArray nd)
        {
            return nd.log();
        }
    }
}
