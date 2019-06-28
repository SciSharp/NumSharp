using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        public NDArray randint(int low, int size = 1)
        {
            var data = new int[size];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = randomizer.Next(low, int.MaxValue);
            }

            var np = new NDArray(typeof(int), size);
            np.ReplaceData(data);

            return np;
        }

        public NDArray randint(int low, int? high = null, Shape shape = null)
        {
            if(high == null)
            {
                high = int.MaxValue;
            }
            if(shape == null)
            {
                shape = new Shape(high.Value - low);
            }
            var data = new int[shape.Size];
            for(int i = 0; i < data.Length; i++)
            {
                data[i] = randomizer.Next(low, high.Value);
            }

            var np = new NDArray(typeof(int), shape.Dimensions.ToArray());
            np.ReplaceData(data);

            return np;
        }
    }
}
