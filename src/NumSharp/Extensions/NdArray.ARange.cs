using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<int> ARange(this NDArray<int> np,int stop, int start = 0, int step = 1)
        {
            int index = 0;

            np.Data = Enumerable.Range(start,stop - start)
                                .Where(x => index++ % step == 0)
                                .ToArray();

            np.Shape = new Shape(new int[] { stop });

            return np;
        }

        public static NDArray<double> ARange(this NDArray<double> np, int stop, int start = 0, int step = 1)
        {
            var list = new double[(stop - start) / step];
            int index = 0;

            for (int i = start; i < stop; i += step)
            {
                if (i % step == 0)
                {
                    list[index] = i;
                }
                index++;
            }

            np.Data = list;
            np.Shape = new Shape(stop);

            return np;
        }
    }
}
