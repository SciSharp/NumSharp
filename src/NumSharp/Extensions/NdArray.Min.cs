using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<double> Min(this NDArray<double> np)
        {
            if(np.NDim == 2)
            {
                var min = new NDArray<double>().Zeros(np.Shape[1]);

                for (int col = 0; col < np.Shape[1]; col++)
                {
                    min[col] = np[0, col];
                    for (int row = 0; row < np.Shape[0]; row++)
                    {
                        if(np[row, col] < min[col])
                        {
                            min[col] = np[row, col];
                        }
                    }
                }

                return min;
            }
            else
            {
                throw new NotImplementedException();
            }
        }
    }
}
