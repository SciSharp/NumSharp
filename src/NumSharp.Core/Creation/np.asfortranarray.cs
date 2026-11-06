using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return an array (ndim >= 1) laid out in Fortran order in memory.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="dtype">By default, the data-type is inferred from the input.</param>
        /// <returns>The input a in Fortran, or column-major, order.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asfortranarray.html</remarks>
        public static NDArray asfortranarray(NDArray a, DType dtype = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return NDArray.asfortranarray(a, dtype);
        }

        /// <summary>
        ///     Return a Fortran-ordered array from a <see cref="MemoryView"/> (obtained from
        ///     <see cref="NDArray.data"/>) — the consumer round-trip of <c>ndarray.data</c>. Matches NumPy's
        ///     <c>np.asfortranarray(a.data)</c>: a zero-copy VIEW when the source is already F-contiguous,
        ///     otherwise an F-order copy. Equivalent to <c>np.asfortranarray(buffer.obj, dtype)</c>.
        /// </summary>
        /// <param name="buffer">A <see cref="MemoryView"/> over an array.</param>
        /// <param name="dtype">By default, the data-type is inferred from the source.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asfortranarray.html</remarks>
        public static NDArray asfortranarray(MemoryView buffer, DType dtype = null)
        {
            if (buffer is null)
                throw new ArgumentNullException(nameof(buffer));
            return asfortranarray(buffer.obj, dtype);
        }
    }
}
