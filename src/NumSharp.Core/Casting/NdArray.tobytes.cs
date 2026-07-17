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
        /// <exception cref="ArgumentException">Thrown when <paramref name="order"/> is not one of C/F/A/K.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tobytes.html</remarks>
        public byte[] tobytes(char order = 'C')
        {
            if (size == 0)
                return System.Array.Empty<byte>();

            // Resolve the NumPy order char to a physical layout ('C' or 'F').
            //   C/F/A -> OrderResolver (A picks F only when this view is F-contiguous-and-not-C-contiguous,
            //            exactly NumPy's PyArray_ISFORTRAN test). Invalid chars raise here.
            //   K     -> 'C'. NumPy's tobytes routes the (non-reference) numeric dtypes through CopyInto
            //            into a C-contiguous destination, so tobytes('K') == tobytes('C') for every dtype
            //            NumSharp supports. This deliberately differs from OrderResolver's 'K' (which keeps
            //            an F-contiguous source as 'F') because tobytes never preserves F for 'K'.
            char physical = (order == 'K' || order == 'k')
                ? 'C'
                : OrderResolver.Resolve(order, this.Shape);

            // Fast path: read the logical bytes straight from this view's buffer when its memory already
            // lays elements out in the requested order AND its data starts at Storage.Address. The
            // offset==0 guard matters: simple contiguous slices are re-based (offset folded into Address),
            // but strided/negative-stride/F-sliced views keep their start in Shape.offset, so Address would
            // point at the wrong first element (e.g. a reversed [::-1] view, or an F-contiguous T[...,1:]).
            bool directable = physical == 'C'
                ? (Shape.IsContiguous && Shape.offset == 0)
                : (Shape.IsFContiguous && Shape.offset == 0);

            // Otherwise materialize a fresh contiguous copy in the target order. copy() drives the layout
            // resolution + fill through the NDIter copy primitive (handles scalar/(1,)/strided/broadcast/
            // transposed); copy('F') leaves a buffer whose linear bytes are exactly the column-major readout.
            NDArray src = directable ? this : this.copy(physical);

            unsafe
            {
                var addr = src.Storage.Address;
                long len = checked((long)src.size * src.dtypesize);

                // Allocate uninitialized: every byte is overwritten by the copy below, so the CLR's
                // default zero-fill would be pure waste (a redundant 2nd write over the whole buffer).
                // This mirrors NumPy's PyBytes_FromStringAndSize(NULL, n) — uninitialized then memcpy'd.
                // A byte[] is capped at int.MaxValue length anyway, so the checked (int) cast is the
                // real allocatable bound (a >2GB result throws OverflowException, as new byte[len] would).
                byte[] bytes = GC.AllocateUninitializedArray<byte>(checked((int)len));
                fixed (byte* @out = bytes)
                    Buffer.MemoryCopy(addr, @out, len, len);
                return bytes;
            }
        }
    }
}
