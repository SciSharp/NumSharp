using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Randomly permute a sequence, or return a permuted range.
        /// </summary>
        /// <param name="x">If x is an integer, randomly permute np.arange(x).</param>
        /// <returns>Permuted sequence or array range.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html</remarks>
        public NDArray permutation(int x)
        {
            var nd = np.arange(x);

            for (int i = 0; i < x; i++)
            {
                var pos = randomizer.Next(x);
                var zero = nd.GetInt32(0);
                nd.SetAtIndex(nd.GetAtIndex(pos), 0);
                nd.SetAtIndex(zero, pos);
            }

            return nd;
        }

        /// <summary>
        ///     Randomly permute a sequence, or return a permuted range.
        /// </summary>
        /// <param name="x">If x is an integer, randomly permute np.arange(x).</param>
        /// <returns>Permuted sequence or array range.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html</remarks>
        public NDArray permutation(NDArray x)
        {
            var len = x.size;
            for (int i = 0; i < len; i++)
            {
                var pos = randomizer.Next(len);
                var zero = x.GetAtIndex(0); //TODO! this doesn't support ndim>1
                x.SetAtIndex(x.GetAtIndex(pos), 0);
                x.SetAtIndex(zero, pos);
            }

            return x;
        }
    }
}
