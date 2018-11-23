using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public void Normalize()
        {
            var np = new NumPy();
            var min = this.min();
            var max = this.max();

            if (NDim == 2)
            {
                for (int col = 0; col < Shape.Shapes[1]; col++)
                {
                    double der = max.Data<double>(col) - min.Data<double>(col);
                    for (int row = 0; row < Shape.Shapes[0]; row++)
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
