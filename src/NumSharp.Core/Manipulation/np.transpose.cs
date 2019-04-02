using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Extensions;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray transpose(NDArray nd)
        {
            var np = new NDArray(nd.dtype);

            if (nd.ndim == 1)
            {
                np.Storage.Reshape(1, np.shape[0]);
            }
            else 
            {
                np.Storage.Reshape(np.shape.Reverse().ToArray());
                for (int idx = 0;idx < np.shape[0];idx++)
                    for (int jdx = 0;jdx < np.shape[1];jdx++)
                        np[idx,jdx] = nd[jdx,idx];
            }
            
            return np;
        }
    }
}
