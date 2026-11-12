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
        /// <returns>Contiguous array of same shape and content as a, with type dtype if specified.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html</remarks>
        public static NDArray ascontiguousarray(NDArray a, DType dtype = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return NDArray.ascontiguousarray(a, dtype);
        }

        /// <summary>
        ///     Return a C-contiguous array from a <see cref="MemoryView"/> (obtained from
        ///     <see cref="NDArray.data"/>) — the consumer round-trip of <c>ndarray.data</c>. Matches NumPy's
        ///     <c>np.ascontiguousarray(a.data)</c>: a zero-copy VIEW when the source is already C-contiguous,
        ///     otherwise a C-order copy. Equivalent to <c>np.ascontiguousarray(a.obj, dtype)</c>.
        /// </summary>
        /// <param name="a">A <see cref="MemoryView"/> over an array.</param>
        /// <param name="dtype">By default, the data-type is inferred from the source.</param>
        /// <returns>The C-contiguous array: a view sharing the source's memory when its layout and dtype already satisfy the
        ///     request, else a copy.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html. NumPy's parameter name <c>a</c> since API-parity batch 8 (it was <c>buffer</c>, a name NumPy's <c>ascontiguousarray</c> refuses as a keyword).</remarks>
        public static NDArray ascontiguousarray(MemoryView a, DType dtype = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return ascontiguousarray(a.obj, dtype);
        }
    }
}
