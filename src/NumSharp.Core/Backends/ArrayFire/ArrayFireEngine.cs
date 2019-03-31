using ArrayFire;
using System;
using System.Collections.Generic;
using System.Text;
using Array = ArrayFire.Array;

namespace NumSharp.Backends
{
    public class ArrayFireEngine : DefaultEngine
    {
        public override NDArray Add(NDArray x, NDArray y)
        {
            return base.Add(x, y);
        }

        public override NDArray Dot(NDArray x, NDArray y)
        {
            return base.Dot(x, y);
        }
    }
}
