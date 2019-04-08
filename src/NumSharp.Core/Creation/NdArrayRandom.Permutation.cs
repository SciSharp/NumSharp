using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        public NDArray Permutation(int max)
        {
            var random = new Random();
            int[] orders = new int[max];

            var np = new NDArray(typeof(int)).arange(max);

            int[] npData = np.Data<int>();

            for(int i = 0; i < max; i++)
            {
                var pos = random.Next(0, max);
                var zero = npData[0];
                npData[0] = npData[pos];
                npData[pos] = zero;
            }
            
            return np;
        }
    }
}
