using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public NDArray NDArray(Shape shape, Type dtype = null, Array buffer = null, string order = "F")
        {
            if (dtype == null)
                dtype = np.float32;

            if (buffer == null)
            {
                switch (dtype.Name) //todo! support all types
                {
                    case "Int32":
                        buffer = new int[shape.Size];
                        break;
                    case "Single":
                        buffer = new float[shape.Size];
                        break;
                    case "Double":
                        buffer = new double[shape.Size];
                        break;
                    default:
                        break;
                }
            }
                

            return new NDArray(buffer, shape: shape, order: order);
        }
    }

}
