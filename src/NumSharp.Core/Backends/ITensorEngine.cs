using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public interface ITensorEngine
    {
        NDArray Add(NDArray x, NDArray y);
        NDArray Dot(NDArray x, NDArray y);
        NDArray Divide(NDArray x, NDArray y);
        NDArray Log(NDArray nd);
        NDArray MatMul(NDArray x, NDArray y);
        NDArray Multiply(NDArray x, NDArray y);
        NDArray Sub(NDArray x, NDArray y);
    }
}
