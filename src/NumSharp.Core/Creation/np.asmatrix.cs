using NumSharp.Core.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public static partial class np
    {
        public static matrix asmatrix(NDArray nd)
        {
            return nd.AsMatrix();
        }
    }
}
