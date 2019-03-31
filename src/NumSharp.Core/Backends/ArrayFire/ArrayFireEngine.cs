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
            if (x.ndim == 2 && y.ndim == 2)
            {
                var dx = Data.CreateArray(x.ToMuliDimArray<int>() as int[,]);
                var dy = Data.CreateArray(y.ToMuliDimArray<int>() as int[,]);
                var dot = Vector.Dot(dx, dy);

                throw new NotImplementedException("ArrayFireEngine.Dot");
            }
            else
            {
                return base.Dot(x, y);
            }
        }
    }
}
