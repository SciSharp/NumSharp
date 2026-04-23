using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.copyto.html</remarks>
        public static void copyto(NDArray dst, NDArray src) //todo! add where argument
        {
            if (dst is null)
                throw new ArgumentNullException(nameof(dst));

            if (src is null)
                throw new ArgumentNullException(nameof(src));

            NumSharpException.ThrowIfNotWriteable(dst.Shape);

            NpyIter.Copy(dst, src);
        }
    }
}
