using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static T amax<T>(NDArray nd)
            => nd.amax<T>();

        public static NDArray amin(NDArray nd, int? axis = null)
            => nd.amin(axis);

        public static NDArray amax(NDArray nd, int axis)
            => nd.amax(axis);

        public static NDArray max(NDArray nd, int axis)
            => nd.max(axis);

        public static NDArray mean(NDArray nd, int axis = -1)
            => nd.mean(axis);


    }
}
