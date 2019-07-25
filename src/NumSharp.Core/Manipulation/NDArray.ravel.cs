using System;
using System.Collections;
using System.Linq;
using System.Text;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return a contiguous flattened array. A 1-D array, containing the elements of the input, is returned
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html</remarks>
        /// <remarks><br></br>If this array's <see cref="Shape"/> is a slice, the a copy will be made.</remarks>
        public NDArray ravel()
        {
            return np.ravel(this);
        }
    }
}
