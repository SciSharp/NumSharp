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
            var list = new int[(int)Math.Ceiling((stop - start + 0.0) / step)];
            int index = 0;

            for (int i = start; i < stop; i += step)
                list[index++] = i;

            np.Data = list;
            np.Shape = new Shape(list.Length);

            return np;
        }

        public static NDArray<double> ARange(this NDArray<double> np, int stop, int start = 0, int step = 1)
        {
            var list = new double[(int)Math.Ceiling((stop - start + 0.0) / step)];
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
