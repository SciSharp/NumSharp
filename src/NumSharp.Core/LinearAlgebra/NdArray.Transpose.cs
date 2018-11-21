using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public partial class NDArray<T> 
    {
        public NDArray<T> transpose()
        {
            var np = new NDArray<T>();
            np.Data = new T[this.Data.Length];

            if (NDim == 1)
            {
                np.Shape = new Shape(1,Shape.Shapes[0]);
            }
            else 
            {
                np.Shape = new Shape(this.Shape.Shapes.Reverse().ToArray());
                for (int idx = 0;idx < np.shape.Shapes[0];idx++)
                    for (int jdx = 0;jdx < np.shape.Shapes[1];jdx++)
                        np[idx,jdx] = this[jdx,idx];
            }
            
            return np;
        }
    }
}
