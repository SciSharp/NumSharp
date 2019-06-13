using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        public NDArray permutation(int max)
        {
            int[] orders = new int[max];

            var nd = np.arange(max);

            for (int i = 0; i < max; i++)
            {
                var pos = randomizer.Next(0, max);
                var zero = nd.Data<int>(0);
                nd[0] = nd.Data<int>(pos);
                nd[pos] = zero;
            }

            return nd;
        }
    }
}
