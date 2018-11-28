using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        public NDArray transpose(NDArray nd)
        {
            var np = new NDArray(nd.dtype);

            if (nd.NDim == 1)
            {
                np.Storage.Shape = new Shape(1, np.Shape.Shapes[0]);
            }
            else 
            {
                np.Storage.Shape = new Shape(np.Shape.Shapes.Reverse().ToArray());
                for (int idx = 0;idx < np.Shape.Shapes[0];idx++)
                    for (int jdx = 0;jdx < np.Shape.Shapes[1];jdx++)
                        np[idx,jdx] = nd[jdx,idx];
            }
            
            return np;
        }
    }
}
