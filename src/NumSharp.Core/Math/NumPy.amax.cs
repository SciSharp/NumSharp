using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public static partial class NumPy
    {
        public static NDArray amax(NDArray nd, int? axis = null)
        {
            return nd.amax(axis);
        }
    }
}
