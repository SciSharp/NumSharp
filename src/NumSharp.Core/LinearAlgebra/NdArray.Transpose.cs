using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray transpose()
        {
            var nd = new NDArray(dtype, new Shape(this.Storage.Shape.Dimensions.Reverse().ToArray()));

            if (ndim == 1)
            {
                nd.Storage = new NDStorage(dtype);
                nd.Storage.Allocate(dtype, new Shape(1, shape[0]));
            }
            else if (ndim == 2)
            {
                for (int idx = 0; idx < nd.shape[0]; idx++)
                    for (int jdx = 0; jdx < nd.shape[1]; jdx++)
                        nd[idx, jdx] = this[jdx, idx];
            }
            else
            {
                throw new NotImplementedException();
            }

            return nd;
        }
    }

}
