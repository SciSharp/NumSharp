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
            var np = new NDArray(dtype);

            if (NDim == 1)
            {
                np.Storage.Shape = new Shape(1, Shape.Shapes[0]);
            }
            else
            {
                np.Storage.Shape = new Shape(Shape.Shapes.Reverse().ToArray());
                for (int idx = 0; idx < np.Shape.Shapes[0]; idx++)
                    for (int jdx = 0; jdx < np.Shape.Shapes[1]; jdx++)
                        np[idx, jdx] = this[jdx, idx];
            }

            return np;
        }
    }

    public partial class NDArrayGeneric<T> 
    {
        public NDArrayGeneric<T> transpose()
        {
            var np = new NDArrayGeneric<T>();
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
