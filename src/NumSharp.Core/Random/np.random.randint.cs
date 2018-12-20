using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace NumSharp.Core
{
    public partial class NumPyRandom
    {
        public NDArray randint(int low, int size = 1)
        {
            var rng = new Random();
            var data = new int[size];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = rng.Next(low, int.MaxValue);
            }

            var np = new NDArray(typeof(int), size);
            np.Storage.SetData(data);

            return np;
        }

        public NDArray randint(int low, int? high = null, Shape shape = null)
        {
            var rng = new Random();
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
                data[i] = rng.Next(low, high.Value);
            }

            var np = new NDArray(typeof(int), shape.Dimensions.ToArray());
            np.Storage.SetData(data);

            return np;
        }
    }
}
