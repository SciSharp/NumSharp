using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class NumPyRandom
    {
        public NDArray permutation(int max)
        {
            var random = new Random();
            int[] orders = new int[max];

            var np = new NumPy().arange(max);

            for (int i = 0; i < max; i++)
            {
                var pos = random.Next(0, max);
                var zero = np.Data<int>(0);
                np[0] = np.Data<int>(pos);
                np[pos] = zero;
            }

            return np;
        }
    }
}
