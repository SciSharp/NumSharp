using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace NumSharp.Core
{
    public partial class NumPyRandom
    {
        public void shuffle(NDArray list)
        {
            var rng = new Random();
            var count = list.size;
            while (count > 1)
            {
                count--;
                var k = rng.Next(count + 1);
                var value = list[k];
                list[k] = list[count];
                list[count] = value;
            }
        }
    }
}
