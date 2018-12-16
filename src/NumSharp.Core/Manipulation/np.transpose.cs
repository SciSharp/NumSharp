using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public static partial class np
    {
        public static NDArray transpose(NDArray nd)
        {
            var np = new NDArray(nd.dtype);

            if (nd.ndim == 1)
            {
                np.Storage.Shape = new Shape(1, np.shape.Shapes[0]);
            }
            else 
            {
                np.Storage.Shape = new Shape(np.shape.Shapes.Reverse().ToArray());
                for (int idx = 0;idx < np.shape.Shapes[0];idx++)
                    for (int jdx = 0;jdx < np.shape.Shapes[1];jdx++)
                        np[idx,jdx] = nd[jdx,idx];
            }
            
            return np;
        }
    }
}
