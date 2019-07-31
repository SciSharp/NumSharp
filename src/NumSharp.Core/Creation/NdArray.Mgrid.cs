using System;
using System.Diagnostics.CodeAnalysis;
using NumSharp;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public partial class NDArray
    {
        public (NDArray, NDArray) mgrid(NDArray nd2)
        {
            return np.mgrid(this, nd2);
        }
    }
}
