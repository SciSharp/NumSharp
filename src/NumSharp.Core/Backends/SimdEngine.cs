using NumSharp.Interfaces;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace NumSharp.Backends
{
    public class SimdEngine : ITensorEngine
    {
        public NDArray Dot(NDArray x, NDArray y)
        {
            var dtype = x.dtype;

            switch (dtype.Name)
            {
                case "Int32":
                    var vx = new Vector<int>(x.Data<int>());
                    var vy = new Vector<int>(y.Data<int>());
                    Vector.Dot(vx, vy);
                    break;
            }

            throw new NotImplementedException("SimdEngine.dot");
        }
    }
}
