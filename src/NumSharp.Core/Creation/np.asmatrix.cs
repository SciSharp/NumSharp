using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static matrix asmatrix(NDArray nd)
        {
            return nd.AsMatrix();
        }
    }
}
