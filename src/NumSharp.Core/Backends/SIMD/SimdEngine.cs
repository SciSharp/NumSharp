using NumSharp.Interfaces;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace NumSharp.Backends
{
    public class SimdEngine : DefaultEngine
    {
        public override NDArray Add(NDArray x, NDArray y)
        {
            int[] lhs = x.Data<int>();
            int[] rhs = x.Data<int>();

            var simdLength = Vector<int>.Count;
            var result = new int[lhs.Length];
            var i = 0;
            for (i = 0; i <= lhs.Length - simdLength; i += simdLength)
            {
                var va = new Vector<int>(lhs, i);
                var vb = new Vector<int>(rhs, i);
                (va + vb).CopyTo(result, i);
            }

            for (; i < lhs.Length; ++i)
                result[i] = lhs[i] + rhs[i];

            return result;
        }

        public NDArray Dot(NDArray x, NDArray y)
        {
            var dtype = x.dtype;

            switch (dtype.Name)
            {
                case "Int32":
                    if(x.ndim == 2 && y.ndim == 2)
                    {
                        var vx = new Vector<int>(x.Data<int>());
                        var vy = new Vector<int>(y.Data<int>());
                        var vec = Vector.Dot(vx, vy);
                    }

                    break;
            }

            throw new NotImplementedException("SimdEngine.dot");
        }
    }
}
