using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Copies values from one array to another, broadcasting as necessary.
        /// </summary>
        /// <param name="dst">The array into which values are copied.</param>
        /// <param name="src">The array from which values are copied.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.copyto.html</remarks>
        public static void copyto(NDArray dst, NDArray src) //todo! add where argument
        {
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));

            if (src == null)
                throw new ArgumentNullException(nameof(src));

            //try to perform memory copy
            if (dst.Shape.IsContiguous && src.Shape.IsContiguous && dst.dtype == src.dtype && src.size == dst.size)
            {
                unsafe
                {
                    src.CopyTo(dst.Address);
                    return;
                }
            }

            //perform manual copy with automatic casting
            MultiIterator.Assign(dst.Storage, src.Storage);
        }
    }
}
