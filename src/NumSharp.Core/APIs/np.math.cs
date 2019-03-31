using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray add(NDArray x, NDArray y)
            => BackendFactory.GetEngine().Add(x, y);

        public static NDArray log(NDArray x)
            => BackendFactory.GetEngine().Log(x);
    }
}
