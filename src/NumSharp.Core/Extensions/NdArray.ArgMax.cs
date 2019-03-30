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
            var max = np.Storage.GetData<T>().Max();

            return np.Storage.GetData<T>().ToList().IndexOf(max);
        }
    }
}
