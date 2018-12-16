using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public static partial class np
    {
        public static NDArray dot(NDArray a, NDArray b)
        {
            return a.dot(b);
        }
    }
}
