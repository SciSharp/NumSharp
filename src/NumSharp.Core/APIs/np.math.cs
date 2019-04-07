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

        public static NDArray divide(NDArray x, NDArray y)
            => BackendFactory.GetEngine().Add(x, y);

        public static NDArray multiply(NDArray x, NDArray y)
            => BackendFactory.GetEngine().Multiply(x, y);

        public static NDArray log(NDArray x)
            => BackendFactory.GetEngine().Log(x);

        public static NDArray subtract(NDArray x, NDArray y)
            => BackendFactory.GetEngine().Sub(x, y);

        public static NDArray sqrt(NDArray x) => x.sqrt();

        public static T amax<T>(NDArray nd)
            => nd.amax<T>();

        public static NDArray amax(NDArray nd, int axis)
            => nd.amax(axis);

        public static int argmax(NDArray nd)
            => nd.argmax();

        public static NDArray max(NDArray nd, int axis)
            => nd.max(axis);
    }
}
