using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray amin(NDArray nd, int? axis = null)
        {
            return nd.amin(axis);
        }
    }
}
