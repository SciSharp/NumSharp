using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Copy the array's elements, in C (row-major) order, into a newly allocated byte array.
        ///     Mirrors NumPy's <c>ndarray.tobytes()</c>: the result is the <em>logical</em> array
        ///     (honoring strides, offset and broadcasting), NOT the raw underlying buffer. A
        ///     non-contiguous or offset view (sliced/strided/transposed/broadcast) is materialized
        ///     into a fresh C-contiguous buffer first, so the returned length is always
        ///     <c>size * dtypesize</c>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tobytes.html</remarks>
        public byte[] ToByteArray()
        {
            if (size == 0)
                return System.Array.Empty<byte>();

            // A "sliced" shape is any view that does not own a pristine, full, C-contiguous,
            // offset-0 buffer (Shape.IsSliced covers non-zero offset, buffer != size, AND
            // non-contiguous strides — which includes broadcast and transposed views). Only a
            // pristine buffer lays its logical elements out contiguously from Address; everything
            // else must be materialized in C order so a single memcpy yields the logical bytes.
            NDArray src = Shape.IsSliced ? this.copy('C') : this;

            unsafe
            {
                var addr = src.Storage.Address;
                long len = checked((long)src.size * src.dtypesize);

                byte[] bytes = new byte[len];
                fixed (byte* @out = bytes)
                    Buffer.MemoryCopy(addr, @out, len, len);
                return bytes;
            }
        }
    }
}
