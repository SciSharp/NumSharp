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
        /// Return a contiguous flattened array.
        /// 
        /// A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.
        /// </summary>
        public NDArray ravel()
        {
            //TODO! if (Shape.IsSliced)
            //TODO!     return new NDArray(new UnmanagedStorage(Storage, Shape.Vector(Shape.size, Shape.ViewInfo)));

            return reshape(Shape.Vector(size));
        }
    }
}
