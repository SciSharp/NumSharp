using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Extensions;

namespace NumSharp
{
    public partial class NDArray<T> 
    {
        public NDArray<T> transpose()
        {
            var np = new NDArray<T>();
            np.Data = this.Data;

            if (NDim == 1)
            {
                np.Shape = new Shape(1,Shape.Shapes[0]);
            }
            else 
            {
                np.Shape = new Shape(this.Shape.Shapes.Reverse().ToArray());
            }
            
            return np;
        }
    }
}
