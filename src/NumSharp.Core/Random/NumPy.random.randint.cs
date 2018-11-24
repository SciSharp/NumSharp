using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace NumSharp.Core
{
    public partial class NumPyRandom
    {
        public NDArray randint(int low, int? high = null, Shape size = null)
        {
            var rng = new Random();
            var data = new int[size.Size];
            for(int i = 0; i < data.Length; i++)
            {
                data[i] = rng.Next(low, high.HasValue ? high.Value : int.MaxValue);
            }

            var np = new NDArray(typeof(int), size.Shapes.ToArray());
            np.Set(data);

            return np;
        }
    }
}
