using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        /// //todo
        /// </summary>
        /// <param name="arr">If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were np.arange(a)</param>
        /// <param name="shape"></param>
        /// <param name="probabilities">The probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html</remarks>
        public NDArray choice(NDArray arr, Shape shape, double[] probabilities = null)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///  //todo
        /// </summary>
        /// <param name="a">If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were np.arange(a)</param>
        /// <param name="shape"></param>
        /// <param name="probabilities">The probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html</remarks>
        public NDArray choice(int a, Shape shape, double[] probabilities = null)
        {
            NDArray arr = np.arange(a);
            NDArray idx = null;
            //Debug.WriteLine($"arr: {arr}");
            if (probabilities is null)
            {
                idx = np.random.randint(0, arr.len, shape);
            }
            return idx;
        }
    }
}
