using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public NDArray Transpose(NDArray x, int[] axes = null)
        {
            NDArray nd;

            if (x.ndim == 1)
            {
                nd = new NDArray(x.Array, x.shape);
            }
            else if (x.ndim == 2)
            {
                nd = new NDArray(x.Array, x.shape.Reverse().ToArray());

                nd = nd.reshape(nd.shape, order: x.order == 'C' ? 'F' : 'C');
            }
            else
            {
                throw new NotImplementedException();
            }

            return nd;
        }
    }
}
