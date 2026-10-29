using System;
using System.ComponentModel;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>NumPy's text when <c>PyArray_CheckStrides</c> refuses the strides (and NumSharp's when its own bounds
        ///     check refuses a strided layout NumPy let through).</summary>
        private const string NdarrayStridesIncompatible = "strides is incompatible with shape of requested array and size of buffer";

        /// <summary>NumPy's text when a buffer holds fewer bytes than the array needs.</summary>
        private const string NdarrayBufferTooSmall = "buffer is too small for requested array";

        /// <summary>
        ///     Constructs an array — NumPy's low-level <c>numpy.ndarray(shape, dtype=float, buffer=None, offset=0,
        ///     strides=None, order=None)</c>. Without a buffer it allocates an UNINITIALIZED array, like
        ///     <see cref="empty(Shape, DType, string)"/>; with a buffer it lays the array over the buffer's BYTES without
        ///     copying them, so writes through the result change the buffer.
        /// </summary>
        /// <param name="shape">
        ///     The dimensions. Only <see cref="Shape.Dimensions"/> is read: the strides come from <paramref name="strides"/>
        ///     or <paramref name="order"/>. With a buffer, the one-dimensional shape <c>-1</c> takes every whole element
        ///     after <paramref name="offset"/>. <c>default(Shape)</c> is NumPy's <c>None</c> and is refused.
        /// </param>
        /// <param name="dtype">The element type; null is float64 (NumPy's default). With a buffer, the buffer's bytes are
        ///     read as this type, whatever the buffer's own element type.</param>
        /// <param name="buffer">
        ///     The memory to use, or null to allocate. A C- or F-contiguous <see cref="NDArray"/>, its
        ///     <see cref="NDArray.data"/> (<see cref="MemoryView"/>), an <see cref="Backends.Unmanaged.IArraySlice"/>, a C#
        ///     array of any rank whose elements hold no references (pinned while the result lives), an
        ///     <see cref="ArraySegment{T}"/> or <see cref="Memory{T}"/> of bytes, a <see cref="ReadOnlyMemory{T}"/> of bytes
        ///     (read-only, NumPy's <c>bytes</c>), or a <see cref="Half"/>, <see cref="char"/> or <see cref="decimal"/>
        ///     value (a read-only copy of its bytes, as a NumPy scalar exports). A larger buffer than needed is fine: the
        ///     array uses a part of it.
        /// </param>
        /// <param name="offset">The byte offset of the first element into <paramref name="buffer"/>. Ignored without a
        ///     buffer, as in NumPy. It need not be a multiple of the element size (the result is then not
        ///     <c>ALIGNED</c>).</param>
        /// <param name="strides">The BYTE stride of every axis (NumPy's unit, as <see cref="NDArray.strides"/> reports),
        ///     or null for the strides of <paramref name="order"/>. Checked against the buffer, or against the array's own
        ///     size without one. A stride of 0 repeats elements and keeps the array writeable.</param>
        /// <param name="order"><c>'C'</c> (the default, NumPy's <c>None</c>) or <c>'F'</c> for the row-major or
        ///     column-major strides; <c>'A'</c> and <c>'K'</c> mean <c>'C'</c> here, and lowercase is accepted. Ignored
        ///     when <paramref name="strides"/> is given.</param>
        /// <returns>
        ///     Without a buffer: a new array that owns its uninitialized memory (an empty one has all-zero strides, as
        ///     NumPy allocates it). With a buffer: a view of the buffer that does not own its data, whose <c>base</c> is the
        ///     buffer's owner (an array) or a <c>uint8</c> array over the whole buffer, writeable when the buffer is, and
        ///     <c>ALIGNED</c> as NumPy computes it from the data pointer and the strides.
        /// </returns>
        /// <exception cref="TypeError"><paramref name="shape"/> is <c>default(Shape)</c> (<c>Use () not None as shape
        ///     arguments</c>); <paramref name="buffer"/> is not a buffer (<c>a bytes-like object is required, not
        ///     'list'</c>); or the buffer is too small without <paramref name="strides"/> (<c>buffer is too small for
        ///     requested array</c>).</exception>
        /// <exception cref="ValueError"><paramref name="order"/> is invalid; <paramref name="strides"/> has the wrong length
        ///     or does not fit the buffer (<c>strides is incompatible with shape of requested array and size of
        ///     buffer</c>); a dimension is negative or the byte count overflows; an <see cref="NDArray"/> buffer is not
        ///     contiguous (<c>ndarray is not contiguous</c>); or a negative <paramref name="offset"/> puts elements before
        ///     the buffer.</exception>
        /// <exception cref="BufferError">A <see cref="MemoryView"/> buffer over a non-contiguous array.</exception>
        /// <exception cref="NotSupportedException">A stride that addresses elements is not a multiple of the element
        ///     size (NumSharp stores strides in whole elements), or <paramref name="dtype"/> has no storage yet
        ///     (datetime64).</exception>
        /// <remarks>
        ///     <para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html. The checks run in NumPy's order
        ///     (<c>array_new</c> in <c>numpy/_core/src/multiarray/arrayobject.c</c>): the buffer, the order, the strides'
        ///     length, the strides against the buffer, the shape <c>-1</c> or the too-small check, then the dimensions.
        ///     The shape and dtype are converted by C# before the call, so a bad dtype string fails first even when the
        ///     shape is also bad (NumPy reports the shape).
        ///     </para>
        ///     <para>
        ///     <b>Safer than NumPy.</b> NumPy's own checks let four inputs through that read outside the buffer: a negative
        ///     offset, the shape <c>-1</c> with a negative stride, an offset near 2^63 (its arithmetic wraps), and strides
        ///     over an empty buffer. After NumPy's checks pass, NumSharp checks the exact byte range and refuses those, with
        ///     the strides error when strides were given, else the offset error of <see cref="frombuffer(byte[], DType, long, long)"/>
        ///     for a negative offset, else the too-small error.
        ///     </para>
        ///     <para>
        ///     <b>Lifetime.</b> The result keeps the buffer's memory alive: a NumSharp array buffer may be disposed while
        ///     the result lives, and a C# array stays pinned until the last array over it is disposed or collected. Pinning
        ///     a large managed array for a long time fragments the managed heap, so dispose the result when done.
        ///     </para>
        ///     <para>
        ///     <b>Divergences</b> (pinned by <c>[Misaligned]</c> tests): a non-native dtype (<c>'&gt;i4'</c>) with a buffer
        ///     DECODES the bytes into a new native array instead of viewing them (NumSharp has no big-endian storage);
        ///     <c>complex64</c> does not exist; a multi-character order string cannot be expressed; a view of an unaligned
        ///     result reports <c>ALIGNED</c> again.
        ///     </para>
        /// </remarks>
        public static NDArray ndarray(Shape shape, DType dtype = null, object buffer = null, long offset = 0, long[] strides = null, char order = 'C')
        {
            // PyArray_IntpConverter refuses None before any other argument is converted.
            if (shape.dimensions is null)
                throw new TypeError("Use () not None as shape arguments");

            // NPY_DEFAULT_TYPE. A descriptor without storage fails here, before a buffer is pinned.
            dtype ??= DType.Double;
            NPTypeCode typeCode = dtype.GetTypeCode();

            // PyArray_BufferConverter: the buffer is resolved (and a managed buffer pinned) before the order is parsed.
            NDBuffer buf = NDBuffer.Acquire(buffer);
            try
            {
                // array_new asks only whether the order is NPY_FORTRANORDER; 'A' and 'K' are the C layout.
                bool fortran = OrderResolver.ResolveBufferOrder(order) == 'F';
                int itemSize = dtype.itemsize;
                long[] dims = shape.dimensions;

                if (strides != null)
                {
                    if (strides.Length != dims.Length)
                        throw new ValueError("strides, if given, must be the same length as shape");

                    // Without a buffer NumPy checks the strides against the array's own size at offset 0.
                    if (!Shape.CheckStrides(itemSize, dims, buf?.Length ?? 0, buf is null ? 0 : offset, strides))
                        throw new ValueError(NdarrayStridesIncompatible);
                }

                if (buf is null)
                    return NdarrayAllocate(dims, dtype, typeCode, strides, fortran);

                return NdarrayOverBuffer(buf, dims, dtype, typeCode, offset, strides, fortran);
            }
            catch
            {
                // A failed call must not keep a managed buffer pinned. After the hand-over this does nothing.
                buf?.Release();
                throw;
            }
        }

        /// <summary>
        ///     Catches the call shape of the old signature, <c>ndarray(shape, dtype, buffer, order)</c>: its fourth
        ///     positional argument was the order, and a C# <see cref="char"/> converts silently to the new
        ///     <see cref="long"/> <c>offset</c> (<c>'F'</c> would read as offset 70).
        /// </summary>
        /// <param name="shape">The dimensions.</param>
        /// <param name="dtype">The element type.</param>
        /// <param name="buffer">The buffer.</param>
        /// <param name="offset">The old order argument, passed by position.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="NotSupportedException">Always; the call does not compile anyway.</exception>
        /// <remarks>Name the argument instead: <c>np.ndarray(shape, dtype, buffer, order: 'F')</c>.</remarks>
        [Obsolete("np.ndarray's fourth parameter is `offset` (NumPy's order: shape, dtype, buffer, offset, strides, order), " +
                  "and a char converts to it silently. Name the order argument: np.ndarray(shape, dtype, buffer, order: 'F').", error: true)]
        [EditorBrowsable(EditorBrowsableState.Never)]
        public static NDArray ndarray(Shape shape, DType dtype, object buffer, char offset)
            => throw new NotSupportedException("Name the order argument: np.ndarray(shape, dtype, buffer, order: 'F').");

        /// <summary>
        ///     The no-buffer branch of <see cref="ndarray(Shape, DType, object, long, long[], char)"/>: NumPy's
        ///     <c>PyArray_NewFromDescr_int</c> with no data pointer.
        /// </summary>
        /// <param name="dims">The dimensions (not yet validated).</param>
        /// <param name="dtype">The element type.</param>
        /// <param name="typeCode">Its storage type.</param>
        /// <param name="strides">The checked byte strides, or null for the order's.</param>
        /// <param name="fortran">True for the column-major layout (ignored with strides).</param>
        /// <returns>A new array that owns its uninitialized memory.</returns>
        /// <exception cref="ValueError">A negative dimension, or a byte count that overflows.</exception>
        /// <exception cref="NotSupportedException">A stride that addresses elements splits an element.</exception>
        private static NDArray NdarrayAllocate(long[] dims, DType dtype, NPTypeCode typeCode, long[] strides, bool fortran)
        {
            int itemSize = dtype.itemsize;
            AllocationGuard.CheckDimensions(dims, itemSize);
            long size = ProductOfValidated(dims);

            if (strides is null)
            {
                // np.empty's allocation in C or F order. The shape's dimension array is immutable, so it is shared.
                var nd = new NDArray(dtype, Shape.Adopt(dims, fortran ? 'F' : 'C'), fillZeros: false);
                // NumPy zeroes every stride of an array it allocates empty (ctors.c: "Make sure all the strides are 0").
                if (size == 0 && dims.Length > 0)
                    nd.Storage.SetShapeUnsafe(new Shape(dims, new long[dims.Length], 0, 0));
                return nd;
            }

            long[] elementStrides;
            if (size == 0)
            {
                // The given strides are zeroed too: an empty allocation has no element to address.
                elementStrides = new long[dims.Length];
            }
            else if (!Shape.TryElementStrides(strides, dims, itemSize, out elementStrides, out int badAxis))
            {
                throw StrideSplitsElement(strides, badAxis, itemSize);
            }

            // CheckStrides kept every element inside prod(dims) elements from offset 0, so that is the whole buffer.
            var layout = new Shape(dims, elementStrides, 0, size);
            // NumPy leaves an array it creates with zero strides writeable; NumSharp's flags call a stride-0 axis a
            // broadcast and clear WRITEABLE, so it is set back (as setflags(write=True) on a broadcast does).
            if (!layout.IsWriteable)
                layout = layout.WithFlags(flagsToSet: ArrayFlags.WRITEABLE);

            var storage = new UnmanagedStorage(ArraySlice.Allocate(typeCode, size, false), layout);
            return new NDArray(storage);
        }

        /// <summary>
        ///     The buffer branch of <see cref="ndarray(Shape, DType, object, long, long[], char)"/>: NumPy's shape
        ///     <c>-1</c> and too-small checks, the dimensions, NumSharp's exact bounds check, then a view over the bytes.
        /// </summary>
        /// <param name="buf">The resolved buffer; its pin is handed to the result or released by the caller.</param>
        /// <param name="dims">The dimensions (not yet validated; <c>(-1,)</c> means "as many as fit").</param>
        /// <param name="dtype">The element type the bytes are read as.</param>
        /// <param name="typeCode">Its storage type.</param>
        /// <param name="offset">The byte offset of the first element.</param>
        /// <param name="strides">The checked byte strides, or null for the order's.</param>
        /// <param name="fortran">True for the column-major layout (ignored with strides).</param>
        /// <returns>A view of the buffer; for a non-native dtype, a decoded native copy.</returns>
        /// <exception cref="TypeError">The buffer is too small (without strides).</exception>
        /// <exception cref="ValueError">A negative dimension, an overflowing byte count, or a layout outside the buffer
        ///     (strides, or a negative offset).</exception>
        /// <exception cref="NotSupportedException">A stride that addresses elements splits an element.</exception>
        private static unsafe NDArray NdarrayOverBuffer(NDBuffer buf, long[] dims, DType dtype, NPTypeCode typeCode, long offset, long[] strides, bool fortran)
        {
            int itemSize = dtype.itemsize;
            long length = buf.Length;

            if (dims.Length == 1 && dims[0] == -1)
            {
                // array_new: the shape (-1,) is every whole element after the offset. C division truncates toward zero, so
                // an offset past the end gives a negative count, which the dimension check below reports.
                dims = new[] { unchecked(length - offset) / itemSize };
            }
            else if (strides is null && length < unchecked(offset + itemSize * Shape.MultiplyListWrapping(dims)))
            {
                // NumPy's arithmetic wraps here; the exact check below catches what the wrap lets through.
                throw new TypeError(NdarrayBufferTooSmall);
            }

            AllocationGuard.CheckDimensions(dims, itemSize);

            long[] byteStrides = strides ?? Shape.FillByteStrides(dims, itemSize, fortran);

            // NumSharp's exact bounds check (NumPy has none past its own): every byte an element occupies must lie in
            // [0, length). An empty array occupies nothing, so any offset NumPy accepted is kept for it.
            Shape.ExactByteExtent(itemSize, dims, byteStrides, out Int128 lower, out Int128 upper);
            if (upper != lower && ((Int128)offset + lower < 0 || (Int128)offset + upper > length))
            {
                if (strides != null)
                    throw new ValueError(NdarrayStridesIncompatible);
                if (offset < 0)
                    throw new ValueError($"offset must be non-negative and no greater than buffer length ({length})");
                throw new TypeError(NdarrayBufferTooSmall);
            }

            if (!Shape.TryElementStrides(byteStrides, dims, itemSize, out long[] elementStrides, out int badAxis))
                throw StrideSplitsElement(byteStrides, badAxis, itemSize);

            // Every check has passed: from here on the buffer belongs to the result.
            NDArray view = NdarrayViewBytes(buf, dims, dtype, typeCode, offset, elementStrides, (long)lower, (long)(upper - lower));
            if (dtype.isnative)
                return view;

            // NumSharp stores native values only, so a non-native dtype is decoded into a new array (np.frombuffer's
            // rule): copy the bytes, swap them, and drop the view (which releases the buffer if nothing else holds it).
            NDArray decoded = view.copy();
            view.Dispose();
            decoded.byteswap(inplace: true);
            return decoded;
        }

        /// <summary>
        ///     Lays the dimensions over the buffer's bytes without a copy, as ONE storage: over the buffer array's own slice
        ///     when the dtype and its element grid match (an ordinary view), else over a forwarding alias of the span the
        ///     array occupies. The flags are NumPy's for an array over a buffer.
        /// </summary>
        /// <param name="buf">The checked buffer (its pin is handed over here).</param>
        /// <param name="dims">The validated dimensions.</param>
        /// <param name="dtype">The element type.</param>
        /// <param name="typeCode">Its storage type (the native type for a non-native dtype).</param>
        /// <param name="offset">The byte offset of the first element.</param>
        /// <param name="elementStrides">The checked strides, in elements.</param>
        /// <param name="lowerBytes">The lowest byte the array occupies, relative to its first element (0 or negative).</param>
        /// <param name="spanBytes">The number of bytes from the lowest to one past the highest occupied byte (0 when
        ///     empty).</param>
        /// <returns>The view; it does not own its data, and its base is the buffer's root.</returns>
        private static unsafe NDArray NdarrayViewBytes(NDBuffer buf, long[] dims, DType dtype, NPTypeCode typeCode, long offset,
            long[] elementStrides, long lowerBytes, long spanBytes)
        {
            int itemSize = dtype.itemsize;
            // NumPy: offset + (char *)buffer.ptr. For an empty array this may point outside the buffer; nothing is read.
            byte* data = unchecked(buf.Start + offset);

            UnmanagedStorage root = buf.TakeRoot(typeCode, itemSize, out IArraySlice arcOwner);

            IArraySlice slice;
            long first;
            byte* ownerStart = (byte*)arcOwner.Address;
            if (spanBytes > 0 && arcOwner.TypeCode == typeCode && data >= ownerStart && (data - ownerStart) % itemSize == 0)
            {
                // The buffer array's own element type and grid: share its slice, as an ordinary view does (its reference
                // counting is the owner's already). The exact bounds check put every element inside it.
                slice = arcOwner;
                first = (data - ownerStart) / itemSize;
            }
            else
            {
                // Another dtype, an offset off the grid, or an empty array: an alias of exactly the occupied span, whose
                // reference counting forwards to the buffer's owner, so the buffer lives while any array over it does.
                slice = ArraySlice.WrapShared(typeCode, data + lowerBytes, spanBytes / itemSize, arcOwner);
                first = -lowerBytes / itemSize;
            }

            // NumPy's WRITEABLE for an array over a buffer: exactly when the buffer is writable (a stride of 0 does not
            // clear it, while NumSharp's flags call a stride-0 axis a read-only broadcast). ALIGNED needs no work here:
            // flags.aligned reads the memory's real alignment (UnmanagedStorage.IsDataAligned), which for whole-element
            // strides is NumPy's raw_array_is_aligned of the data pointer.
            var layout = new Shape(dims, elementStrides, first, slice.Count);
            layout = buf.Writable
                ? layout.WithFlags(flagsToSet: ArrayFlags.WRITEABLE)
                : layout.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);

            return new NDArray(UnmanagedStorage.OverBuffer(slice, layout, root));
        }

        /// <summary>The element count of dimensions that <see cref="AllocationGuard.CheckDimensions(long[], int)"/> has
        ///     accepted (non-negative, and the byte count fits a <see cref="long"/>, so the count does too).</summary>
        /// <param name="dims">The validated dimensions.</param>
        /// <returns>Their product; 1 for a 0-d shape.</returns>
        private static long ProductOfValidated(long[] dims)
        {
            long size = 1;
            for (int i = 0; i < dims.Length; i++)
                size *= dims[i];
            return size;
        }

        /// <summary>The refusal of a stride NumSharp cannot store: it addresses elements and is not a whole number of
        ///     them.</summary>
        /// <param name="byteStrides">The byte strides.</param>
        /// <param name="axis">The first axis whose stride splits an element.</param>
        /// <param name="itemSize">The element size.</param>
        /// <returns>The exception to throw.</returns>
        private static NotSupportedException StrideSplitsElement(long[] byteStrides, int axis, int itemSize)
            => new NotSupportedException(
                $"The byte stride {byteStrides[axis]} of axis {axis} is not a multiple of the element size {itemSize}. " +
                "NumSharp stores strides in whole elements, so it cannot lay out this array (NumPy accepts it as an unaligned array).");
    }
}
