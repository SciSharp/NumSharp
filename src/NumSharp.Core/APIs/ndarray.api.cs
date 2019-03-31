using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray log()
             => BackendFactory.GetEngine().Log(this);

        public static NDArray operator +(NDArray x, NDArray y)
            => BackendFactory.GetEngine().Add(x, y);

        public static NDArray operator -(NDArray x, NDArray y)
            => BackendFactory.GetEngine().Sub(x, y);

        public static NDArray operator *(NDArray x, NDArray y)
            => BackendFactory.GetEngine().Multiply(x, y);

        public static NDArray operator /(NDArray x, NDArray y)
            => BackendFactory.GetEngine().Divide(x, y);
    }
}
