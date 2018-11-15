using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<double> Max(this NDArray<double> np)
        {
            if (np.NDim == 1)
            {
                var max = new NDArray<double>().Zeros(np.Size);
                max.Shape = new Shape(1);
                max.Data = new double[] { np.Data.Max() };

                return max;
            }
            else if (np.NDim == 2)
            {
                var max = new NDArray<double>().Zeros(np.Shape.Shapes[1]);

                for (int col = 0; col < np.Shape.Shapes[1]; col++)
                {
                    max[col] = np[0, col];
                    for (int row = 0; row < np.Shape.Shapes[0]; row++)
                    {
                        if (np[row, col] > max[col])
                        {
                            max[col] = np[row, col];
                        }
                    }
                }

                return max;
            }
            else
            {
                throw new NotImplementedException();
            }
        }
    }
}
