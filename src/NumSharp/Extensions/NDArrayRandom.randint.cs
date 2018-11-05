using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayRandomExtensions
    {
        public static NDArray<int> randint(this NDArrayRandom rand, int low, int? high = null, Shape size = null)
        {
            var rng = new Random();
            var data = new int[size.Size];
            for(int i = 0; i < data.Length; i++)
            {
                data[i] = rng.Next(low, high.HasValue ? high.Value : int.MaxValue);
            }

            var np = new NDArray<int>();
            np.Shape = size.Shapes;
            np.Data = data;

            return np;
        }
    }
}
