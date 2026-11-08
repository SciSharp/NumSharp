using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Construct a byte array containing the raw data bytes of the array in the requested memory
        ///     <paramref name="order"/> (default C-order). Mirrors NumPy's <c>ndarray.tobytes(order='C')</c>:
        ///     the result is the <em>logical</em> array (strides, offset and broadcasting resolved), NOT the
        ///     raw underlying buffer. A view whose memory does not already lay its logical elements out
        ///     in the requested order (sliced/strided/transposed/broadcast, or C-order requested on an
        ///     F-contiguous view and vice-versa) is materialized into a fresh contiguous buffer first,
        ///     so the returned length is always <c>size * dtypesize</c>.
        /// </summary>
        /// <param name="order">
        ///     Controls the memory layout of the byte output:
        ///     <list type="bullet">
        ///         <item><c>'C'</c> - C-order (row-major). Default.</item>
        ///         <item><c>'F'</c> - F-order (column-major).</item>
        ///         <item><c>'A'</c> - "Any": 'F' if this array is F-contiguous (and not C-contiguous), else 'C'.</item>
        ///         <item><c>'K'</c> - accepted for NumPy parity; resolves to 'C' for the numeric dtypes
        ///             NumSharp supports (NumPy copies into a C-contiguous destination for <c>tobytes('K')</c>).</item>
        ///     </list>
        /// </param>
        /// <returns>A fresh, detached <see cref="byte"/> array of length <c>size * dtypesize</c>.</returns>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of C/F/A/K, either case (<c>order must be one of
        ///     'C', 'F', 'A', or 'K' (got 'Z')</c>): checked first, so an empty array refuses it too.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tobytes.html</remarks>
        public byte[] tobytes(char order = 'C') => Backends.UnmanagedStorage.ToBytes(this, order);
    }
}
