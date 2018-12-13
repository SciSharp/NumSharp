using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using np = NumSharp.Core.NumPy;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public void normalize()
        {
            var min = this.min(0);
            var max = this.max(0);

            if (ndim == 2)
            {
                for (int col = 0; col < shape.Shapes[1]; col++)
                {

                    double der = max.Data<double>(col) - min.Data<double>(col);
                    for (int row = 0; row < shape.Shapes[0]; row++)
                    {
                        this[row, col] = (Data<double>(row, col) - min.Data<double>(col)) / der;
                    }
                }   
            }
            else
            {
                throw new NotImplementedException();
            }
        }
    }
}
