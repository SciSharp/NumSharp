using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static void Normalize(this NDArray<double> np)
        {
            var min = np.min();
            var max = np.Max();

            if (np.NDim == 2)
            {
                for (int col = 0; col < np.Shape.Shapes[1]; col++)
                {
                    double der = max[col] - min[col];
                    for (int row = 0; row < np.Shape.Shapes[0]; row++)
                    {
                        np[row, col] = (np[row, col] - min[col]) / der;
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
