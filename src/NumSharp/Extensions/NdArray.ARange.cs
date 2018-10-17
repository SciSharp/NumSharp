using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<int> ARange(this NDArray<int> np, int stop, int start = 0, int step = 1)
        {
            int index = 0;

            np.Data = Enumerable.Range(start, stop - start)
                .Where(x => index++ % step == 0)
                .ToList();

            return np;
        }
    }
}
