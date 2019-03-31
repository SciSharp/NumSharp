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
    }
}
