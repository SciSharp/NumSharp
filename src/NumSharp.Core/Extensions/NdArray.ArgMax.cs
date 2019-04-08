using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static int ArgMax<T>(this NDArray np )
        {
            var max = np.Data<T>().Max();

            return np.Data<T>().ToList().IndexOf(max);
        }
    }
}
