using System;
using System.Linq;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Transpose(NDArray x, int[] axes = null)
        {
            NDArray nd;

            if (x.ndim == 1)
            {
                nd = new NDArray(x.Array, x.shape);
            }
            else if (x.ndim == 2)
            {
                nd = new NDArray(x.Array, x.shape.Reverse().ToArray());

                nd = nd.reshape(nd.shape); //TODO , order: x.order == 'C' ? 'F' : 'C'
            }
            else
            {
                throw new NotImplementedException();
            }

            return nd;
        }
    }
}
