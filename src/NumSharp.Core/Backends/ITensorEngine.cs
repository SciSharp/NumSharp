using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Generic;

namespace NumSharp
{
    public interface ITensorEngine
    {
        #region Math
        NDArray Add(NDArray x, NDArray y);
        NDArray Dot(NDArray x, NDArray y);
        NDArray Divide(NDArray x, NDArray y);
        NDArray Log(NDArray nd);
        NDArray MatMul(NDArray x, NDArray y);
        NDArray Mean(NDArray x, int axis = -1);
        NDArray Multiply(NDArray x, NDArray y);
        NDArray Power(NDArray x, ValueType y);
        NDArray Sub(NDArray x, NDArray y);
        NDArray Sum(NDArray x, int axis = -1);
        #endregion

        #region Logic
        bool All(NDArray nd);
        NDArray<bool> All(NDArray nd, int axis);
        bool AllClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false);
        NDArray<bool> IsClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false);
        NDArray<bool> IsFinite(NDArray a);
        NDArray<bool> IsNan(NDArray a);
        #endregion

        #region Array Manipulation
        NDArray NDArray(Shape shape, Type dtype = null, Array buffer = null, string order = "F");
        NDArray Transpose(NDArray nd, int[] axes = null); 
        #endregion
    }
}
