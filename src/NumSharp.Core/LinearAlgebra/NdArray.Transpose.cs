using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Returns a view of the array with axes transposed.
        /// For a 1-D array this has no effect, as a transposed vector is simply the same vector.To convert a 1-D array into a 2D column vector, an 
        /// additional dimension must be added. np.atleast2d(a).T achieves this, as does a[:, np.newaxis]. 
        /// For a 2-D array, this is a standard matrix transpose. For an n-D array, if axes are given, their 
        /// order indicates how the axes are permuted (see Examples). If axes are not provided 
        /// and a.shape = (i[0], i[1], ...i[n - 2], i[n - 1]), then a.transpose().shape = (i[n - 1], i[n - 2], ...i[1], i[0]).
        /// </summary>
        /// <param name="axes">None or no argument: reverses the order of the axes. 
        /// Tuple of ints: i in the j-th place in the tuple means a’s i-th axis becomes a.transpose()’s j-th axis.</param>
        /// <returns>View of a, with axes suitably permuted</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.transpose.html?highlight=transpose#numpy.ndarray.transpose</remarks>
        public NDArray transpose(int[] axes = null)
            => np.transpose(this);
    }
}
