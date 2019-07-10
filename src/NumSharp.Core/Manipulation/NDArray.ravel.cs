using System;
using System.Collections;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Return a contiguous flattened array.
        /// 
        /// A 1-D array, containing the elements of the input, is returned.A copy is made only if needed.
        /// </summary>
        public NDArray ravel()
        {
            return reshape(Storage.Shape.Size);
        }
    }
}
