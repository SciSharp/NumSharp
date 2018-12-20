using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray transpose()
        {
            var nd = new NDArray(dtype, new Shape(this.Storage.Shape.Dimensions.Reverse().ToArray()));

            if (ndim == 1)
            {
                nd.Storage = new NDStorage(dtype);
                nd.Storage.Allocate(dtype, new Shape(1, shape.Dimensions[0]),1);
            }
            else if (ndim == 2)
            {
                for (int idx = 0; idx < nd.shape.Dimensions[0]; idx++)
                    for (int jdx = 0; jdx < nd.shape.Dimensions[1]; jdx++)
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
