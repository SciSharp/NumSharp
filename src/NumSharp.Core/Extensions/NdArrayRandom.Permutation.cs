using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core.Extensions
{
    public static partial class NDArrayRandomExtensions
    {
        public static NDArray<int> Permutation(this NDArrayRandom rand, int max)
        {
            var random = new Random();
            int[] orders = new int[max];

            var np = new NDArray<int>().arange(max);

            for(int i = 0; i < max; i++)
            {
                var pos = random.Next(0, max);
                var zero = np.Data[0];
                np.Data[0] = np.Data[pos];
                np.Data[pos] = zero;
            }
            
            return np;
        }
    }
}
