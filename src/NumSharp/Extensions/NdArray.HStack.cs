using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<double> HStack(this NDArray<double> np, NDArray<double> np2 )
        {
            var puffer = np.Data.ToList();

            puffer.AddRange(np2.Data);

            var returnValue = new NDArray<double>().Zeros(np.Length + np2.Length);
            returnValue.Data = puffer;

            return returnValue;
        }

        public static NDArray<NDArray<double>> HStack(this NDArray<NDArray<double>> np, NDArray<int> np2)
        {
            var np3 = new NDArray<NDArray<double>>();

            for (int r = 0; r < np.Length; r++)
            {
                var row = new NDArray<double>();

                for (int c = 0; c < np[r].Length; c++)
                {
                    row.Data.Add(np[r][c]);
                }

                row.Data.Add(np2[r]);

                np3.Data.Add(row);
            }

            return np3;
        }
    }
}
