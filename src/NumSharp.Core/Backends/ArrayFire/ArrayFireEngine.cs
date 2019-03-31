using ArrayFire;
using NumSharp.Interfaces;
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
    }
}
