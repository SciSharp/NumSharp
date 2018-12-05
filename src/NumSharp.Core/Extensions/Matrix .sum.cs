using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class matrix
    {
        public new matrix transpose()
        {
            Storage.Shape = new Shape(this.Storage.Shape.Shapes.Reverse().ToArray());

            var nd = new NDArray(this.dtype, this.shape);
            switch (this.dtype.Name)
            {
                case "Double":
                    nd.Set(this.float64);
                    break;
                case "Int32":
                    nd.Set(this.int32);
                    break;
            }

            if (ndim == 1)
            {
                Storage = NDStorage.CreateByShapeAndType(dtype, new Shape(1, shape.Shapes[0]));
                Storage.SetData(nd.float64);
            }
            else if (ndim == 2)
            {
                for (int idx = 0; idx < shape.Shapes[0]; idx++)
                    for (int jdx = 0; jdx < shape.Shapes[1]; jdx++)
                        this[idx, jdx] = nd[jdx, idx];
            }
            else
            {
                throw new NotImplementedException();
            }

            return this;
        }
    }
}
