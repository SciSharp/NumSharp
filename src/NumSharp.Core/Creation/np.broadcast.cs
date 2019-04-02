using NumSharp.Casting;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {/// <summary>
     /// 
     /// </summary>
     /// <param name="nd1">  1-D arrays representing the coordinates of a grid </param>
     /// <param name="nd2">  1-D arrays representing the coordinates of a grid</param>
     /// <returns></returns>
        public static Broadcast broadcast(NDArray nd1, NDArray nd2)
        {
            Shape shape = null;
            int nd1maxAxis = nd1.shape[0] > nd1.shape[1] ? nd1.shape[0] : nd1.shape[1];
            int nd2maxAxis = nd2.shape[0] > nd2.shape[1] ? nd2.shape[0] : nd2.shape[1];
            if (nd1.shape[0] == 1 && nd2.shape[1] == 1)
            {
                shape = new Shape(nd2maxAxis, nd1maxAxis);
            } else if (nd1.shape[1] == 1 && nd2.shape[0] == 1)
            {
                shape = new Shape(nd1maxAxis, nd2maxAxis);
            }
            int nd = 2;
            int ndim = 2;
            int size = nd1maxAxis * nd2maxAxis;

            return new Broadcast(nd, ndim, shape, size);
        }
    }
}
