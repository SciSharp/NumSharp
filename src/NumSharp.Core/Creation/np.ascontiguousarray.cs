using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a contiguous array (ndim >= 1) in memory (C order).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="dtype">By default, the data-type is inferred from the input.</param>
        /// <returns>Contiguous array of same shape and content as a, with type dtype if specified.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html</remarks>
        public static NDArray ascontiguousarray(NDArray a, Type dtype = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return asarray(a, dtype, 'C');
        }

        /// <summary>
        ///     Return a C-contiguous array from a <see cref="MemoryView"/> (obtained from
        ///     <see cref="NDArray.data"/>) — the consumer round-trip of <c>ndarray.data</c>. Matches NumPy's
        ///     <c>np.ascontiguousarray(a.data)</c>: a zero-copy VIEW when the source is already C-contiguous,
        ///     otherwise a C-order copy. Equivalent to <c>np.ascontiguousarray(buffer.obj, dtype)</c>.
        /// </summary>
        /// <param name="buffer">A <see cref="MemoryView"/> over an array.</param>
        /// <param name="dtype">By default, the data-type is inferred from the source.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html</remarks>
        public static NDArray ascontiguousarray(MemoryView buffer, Type dtype = null)
        {
            if (buffer is null)
                throw new ArgumentNullException(nameof(buffer));
            return ascontiguousarray(buffer.obj, dtype);
        }
    }
}
