using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public static partial class np
    {
        public static NDArray multiply(NDArray nd, float data2)
        {
            return nd * data2;
        }
    }
}
