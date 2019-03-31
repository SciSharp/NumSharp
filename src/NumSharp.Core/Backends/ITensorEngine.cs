using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Backends
{
    public interface ITensorEngine
    {
        NDArray Dot(NDArray x, NDArray y);
    }
}
