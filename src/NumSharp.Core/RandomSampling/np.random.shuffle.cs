using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Modify a sequence in-place by shuffling its contents.
        /// </summary>
        /// <param name="x">The array or list to be shuffled.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.shuffle.html <br></br>Does not copy <paramref name="x"/></remarks>
        public void shuffle(NDArray x)
        {
            var count = x.size;
            while (count-- > 1)
            {
                var swapAt = randomizer.Next(x);
                var tmp = x.GetAtIndex(0);
                x.SetAtIndex(x.GetAtIndex(swapAt), 0);
                x.SetAtIndex(tmp, swapAt);
            }
        }
    }
}
