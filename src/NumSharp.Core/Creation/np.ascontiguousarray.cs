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
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Plain((a, dtype), static s => ascontiguousarray(s.a, s.dtype));
            // a numpy.char.chararray operand: NumPy converts with np.asarray here, so the result is a plain array (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.Plain((a, dtype), static s => ascontiguousarray(s.a, s.dtype));
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return NDArray.AsContiguousArray(a, dtype);
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
        public static NDArray ascontiguousarray(MemoryView buffer, DType dtype = null)
        {
            if (buffer is null)
                throw new ArgumentNullException(nameof(buffer));
            return ascontiguousarray(buffer.obj, dtype);
        }
    }
}
