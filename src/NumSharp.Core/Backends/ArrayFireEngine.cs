using ArrayFire;
using NumSharp.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;
using Array = ArrayFire.Array;

namespace NumSharp.Backends
{
    public class ArrayFireEngine : ITensorEngine
    {
        public NDArray Add(NDArray x, NDArray y)
        {
            throw new NotImplementedException();
        }

        public NDArray Dot(NDArray x, NDArray y)
        {
            return Vector.Dot(x, y);
        }
    }
}
