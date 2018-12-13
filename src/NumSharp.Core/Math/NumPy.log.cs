using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public static partial class NumPy
    {
        public static NDArray log(NDArray nd)
        {
            return nd.log();
        }
    }
}
