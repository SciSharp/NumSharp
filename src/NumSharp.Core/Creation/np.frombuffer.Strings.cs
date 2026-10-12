using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {
        // -------------------------------------------------------------------------------------------------
        // The fixed-width string route of np.frombuffer (S / U dtypes).
        //
        // NumPy has ONE frombuffer: PyArray_FromBuffer checks the descriptor and the byte range, then lays a
        // 1-D array over the buffer's bytes (a view: OWNDATA false, writeable iff the buffer is). NumSharp's
        // overloads take the C# spellings of a buffer (byte[], MemoryView, Memory<byte>, ArraySegment<byte>,
        // a typed array, a pointer, a ReadOnlySpan<byte>). Their numeric paths size elements by NPTypeCode,
        // which a string lane cannot do: its width lives on the descriptor. Every string request therefore
        // comes here instead. The view itself is np.ndarray's (the port of array_new over a buffer), which
        // already pins a managed buffer, keeps a NumSharp buffer alive through its reference counting, roots
        // a foreign buffer as a non-owning base and carries the buffer's writeability; this file adds
        // frombuffer's own checks, which differ from np.ndarray's in order and text.
        // -------------------------------------------------------------------------------------------------

        /// <summary>NumPy's text when the descriptor has no width (<c>np.frombuffer(b, 'S')</c>, <c>np.str_</c>).</summary>
        private const string FromBufferUnsizedText = "itemsize cannot be zero in type";

        /// <summary>
        ///     NumPy's <c>PyArray_FromBuffer</c> checks for a string descriptor over a buffer of
        ///     <paramref name="bufferLength"/> bytes, in its order: the width, the offset, then the element count (every
        ///     whole element after the offset for a negative count, which must leave no partial element; else the count
        ///     must fit).
        /// </summary>
        /// <param name="bufferLength">The buffer's length in bytes.</param>
        /// <param name="dtype">A string descriptor (sized or not).</param>
        /// <param name="count">The requested element count; negative means "every element after the offset".</param>
        /// <param name="offset">The byte offset of the first element.</param>
        /// <returns>The number of elements the result has.</returns>
        /// <exception cref="ValueError">NumPy's texts: <c>itemsize cannot be zero in type</c>, <c>offset must be non-negative
        ///     and no greater than buffer length (N)</c>, <c>buffer size must be a multiple of element size</c>, or
        ///     <c>buffer is smaller than requested size</c>.</exception>
        private static long StringBufferCount(long bufferLength, DType dtype, long count, long offset)
        {
            int itemSize = dtype.itemsize;
            if (itemSize == 0)
                throw new ValueError(FromBufferUnsizedText);
            if (offset < 0 || offset > bufferLength)
                throw new ValueError($"offset must be non-negative and no greater than buffer length ({bufferLength})");

            long available = bufferLength - offset;
            if (count < 0)
            {
                if (available % itemSize != 0)
                    throw new ValueError("buffer size must be a multiple of element size");
                return available / itemSize;
            }

            // NumPy compares s < n * itemsize; dividing instead keeps a huge count from overflowing (for whole numbers
            // n * k > s exactly when n > floor(s / k)).
            if (count > available / itemSize)
                throw new ValueError("buffer is smaller than requested size");
            return count;
        }

        /// <summary>
        ///     The string route of every buffer that <c>np.ndarray(shape, buffer=...)</c> accepts (a <c>byte[]</c> or other
        ///     unmanaged-element C# array, a <see cref="MemoryView"/>, a <see cref="Memory{T}"/> or
        ///     <see cref="ArraySegment{T}"/> of bytes): frombuffer's own checks over the buffer's byte length, then
        ///     np.ndarray's 1-D view of the same bytes.
        /// </summary>
        /// <param name="buffer">The buffer object, as <c>np.ndarray</c>'s <c>buffer</c> argument.</param>
        /// <param name="bufferLength">Its length in bytes (the bound frombuffer checks against).</param>
        /// <param name="dtype">A string descriptor.</param>
        /// <param name="count">The requested element count; negative means all.</param>
        /// <param name="offset">The byte offset of the first element.</param>
        /// <returns>
        ///     A 1-D view of the buffer that does not own its data, writeable when the buffer is; for a non-native
        ///     <c>U</c> descriptor (<c>'&gt;U2'</c>, big-endian code points) a decoded native copy, because NumSharp stores
        ///     every array in host byte order (NumPy would keep the byte order in the descriptor).
        /// </returns>
        /// <exception cref="ValueError">See <see cref="StringBufferCount"/>.</exception>
        /// <remarks>
        ///     Once frombuffer's checks pass, np.ndarray's own checks cannot fail: the result's bytes lie inside the buffer.
        ///     A managed buffer is pinned until the last array over it is disposed or collected.
        /// </remarks>
        private static NDArray StringFromBufferObject(object buffer, long bufferLength, DType dtype, long count, long offset)
        {
            long elements = StringBufferCount(bufferLength, dtype, count, offset);
            return ndarray(new Shape(elements), dtype, buffer, offset);
        }

        /// <summary>
        ///     <c>np.frombuffer(bytearray, dtype='S3' / 'U2', count, offset)</c>: the bytes of a C# array read as fixed-width
        ///     string elements — a view over the pinned array.
        /// </summary>
        /// <param name="buffer">The bytes (a C# <c>byte[]</c> is mutable, NumPy's <c>bytearray</c>: the view is writeable).</param>
        /// <param name="dtype">A string descriptor.</param>
        /// <param name="count">Elements to read, or a negative number for every whole element after <paramref name="offset"/>.</param>
        /// <param name="offset">The byte offset of the first element.</param>
        /// <returns>The 1-D string array (see <see cref="StringFromBufferObject"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="buffer"/> is null.</exception>
        /// <exception cref="ValueError">See <see cref="StringBufferCount"/>.</exception>
        private static NDArray StringFromBuffer(byte[] buffer, DType dtype, long count, long offset)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            return StringFromBufferObject(buffer, buffer.LongLength, dtype, count, offset);
        }

        /// <summary>
        ///     <c>np.frombuffer(memoryview, dtype='S3', ...)</c>: the bytes of the array behind <paramref name="buffer"/> read
        ///     as string elements — a view that keeps that array's memory alive.
        /// </summary>
        /// <param name="buffer">A memoryview over a C-contiguous array (<see cref="NDArray.data"/>).</param>
        /// <param name="dtype">A string descriptor.</param>
        /// <param name="count">Elements to read, or a negative number for all.</param>
        /// <param name="offset">The byte offset of the first element.</param>
        /// <returns>The 1-D string array; read-only for good when the memoryview is.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="buffer"/> is null.</exception>
        /// <exception cref="ValueError">An unsized descriptor (checked first, as NumPy checks it before it asks for the
        ///     buffer), or the range checks of <see cref="StringBufferCount"/>.</exception>
        /// <exception cref="InvalidOperationException">The memoryview is not C-contiguous (NumPy's <c>BufferError</c>; the
        ///     numeric overload raises the same type and text).</exception>
        private static NDArray StringFromMemoryView(MemoryView buffer, DType dtype, long count, long offset)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            if (dtype.itemsize == 0)
                throw new ValueError(FromBufferUnsizedText);
            if (!buffer.c_contiguous)
                throw new InvalidOperationException("memoryview: underlying buffer is not C-contiguous");

            var result = StringFromBufferObject(buffer, buffer.nbytes, dtype, count, offset);
            // A read-only exporter refuses a writable buffer for good: setflags(write=True) on the result must fail, as the
            // numeric overload pins it. A decoded copy (a non-native descriptor) is the caller's own and stays writeable.
            if (buffer.@readonly && dtype.isnative)
            {
                result.Storage.WriteProtected = true;
                if (result.Shape.IsWriteable)
                    result.Storage.SetShapeUnsafe(result.Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));
            }

            return result;
        }

        /// <summary>
        ///     <c>np.frombuffer</c> over a <see cref="ReadOnlySpan{T}"/> with a string dtype: a span cannot be pinned past the
        ///     call, so the selected bytes are COPIED into a new array that owns them (the numeric span overload's rule).
        /// </summary>
        /// <param name="buffer">The bytes.</param>
        /// <param name="dtype">A string descriptor.</param>
        /// <param name="count">Elements to read, or a negative number for all.</param>
        /// <param name="offset">The byte offset of the first element.</param>
        /// <returns>A new 1-D string array (native byte order: a <c>'&gt;U2'</c> request is decoded).</returns>
        /// <exception cref="ValueError">See <see cref="StringBufferCount"/>.</exception>
        private static unsafe NDArray StringFromSpan(ReadOnlySpan<byte> buffer, DType dtype, long count, long offset)
        {
            long elements = StringBufferCount(buffer.Length, dtype, count, offset);
            var result = new NDArray(NDStringStorage.AllocationDescr(dtype), Shape.Vector(elements), false);
            if (elements == 0)
                return result;

            long bytes = elements * dtype.itemsize;
            // The count check put [offset, offset + bytes) inside the span, so offset indexes a real byte.
            fixed (byte* src = &buffer[(int)offset])
                Buffer.MemoryCopy(src, (void*)result.Storage.Address, bytes, bytes);
            if (!dtype.isnative)
                result.byteswap(inplace: true); // big-endian code points into host order, per unit
            return result;
        }

        /// <summary>
        ///     <c>np.frombuffer</c> over unmanaged memory with a string dtype: a 1-D view of the memory, or an array that
        ///     owns it through <paramref name="dispose"/> (the numeric pointer overload's ownership rule).
        /// </summary>
        /// <param name="address">The first byte of the memory.</param>
        /// <param name="byteLength">The memory's length in bytes.</param>
        /// <param name="dtype">A string descriptor.</param>
        /// <param name="count">Elements to read, or a negative number for all.</param>
        /// <param name="offset">The byte offset of the first element.</param>
        /// <param name="dispose">Frees the memory when the array is released, or null when the caller keeps it alive (the
        ///     result then does not own its data, as NumPy's array over a foreign buffer).</param>
        /// <returns>The 1-D string array; for a non-native <c>U</c> descriptor a decoded native copy (the memory is then
        ///     released at once when <paramref name="dispose"/> was given); for zero elements a fresh empty array (the
        ///     memory is released at once, as the numeric overload does).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="address"/> is null.</exception>
        /// <exception cref="ValueError">See <see cref="StringBufferCount"/>. Unlike the numeric pointer overload, which caps a
        ///     too-large count, a string request follows NumPy's rule and refuses it.</exception>
        /// <remarks>On a refused request <paramref name="dispose"/> is not called, as on the numeric overload: the caller
        ///     still owns the memory.</remarks>
        private static unsafe NDArray StringFromPointer(byte* address, long byteLength, DType dtype, long count, long offset, Action dispose)
        {
            if (address == null)
                throw new ArgumentNullException(nameof(address));

            long elements = StringBufferCount(byteLength, dtype, count, offset);
            var native = NDStringStorage.AllocationDescr(dtype);
            if (elements == 0)
            {
                // Nothing of the memory is used: the numeric overload frees it at once and returns a fresh empty array.
                dispose?.Invoke();
                return new NDArray(native, Shape.Vector(0), false);
            }

            long bytes = elements * dtype.itemsize;
            var block = dispose != null
                ? new UnmanagedMemoryBlock<byte>(address + offset, bytes, dispose)
                : new UnmanagedMemoryBlock<byte>(address + offset, bytes);
            var storage = new UnmanagedStorage(new StringArraySlice(new ArraySlice<byte>(block), native), Shape.Vector(elements));
            if (dispose == null)
            {
                // The memory is the caller's (NumPy's non-array base object): the array does not own its data.
                storage.ExternalBase = true;
                storage.OnReshaped();
            }

            var view = new NDArray(storage);
            if (dtype.isnative)
                return view;

            // Big-endian code points: a native decoded copy; the view (and, through dispose, the memory) goes at once.
            using (view)
                return view.byteswap();
        }

        /// <summary>
        ///     <c>np.frombuffer</c> over a C# array of any unmanaged element type with a string dtype: the array's bytes read
        ///     as string elements — a view over the pinned array (NumPy's frombuffer over any buffer-protocol object).
        /// </summary>
        /// <typeparam name="TSource">The array's element type.</typeparam>
        /// <param name="array">The array.</param>
        /// <param name="dtype">A string descriptor.</param>
        /// <param name="count">Elements to read, or a negative number for all.</param>
        /// <param name="offset">The byte offset of the first element.</param>
        /// <returns>The 1-D string array (see <see cref="StringFromBufferObject"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="array"/> is null.</exception>
        /// <exception cref="ValueError">See <see cref="StringBufferCount"/>.</exception>
        private static NDArray StringFromArray<TSource>(TSource[] array, DType dtype, long count, long offset) where TSource : unmanaged
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));
            return StringFromBufferObject(array, array.LongLength * Unsafe.SizeOf<TSource>(), dtype, count, offset);
        }

        /// <summary>
        ///     The string descriptor a dtype STRING names (<c>'S3'</c>, <c>'&lt;U2'</c>, <c>'c'</c>, <c>'a5'</c>,
        ///     <c>'str'</c>), or null when it names another dtype or nothing <see cref="np.dtype(string)"/> parses. The
        ///     string-spelled overloads ask this before their own parser, which knows only the fixed-size codes and once
        ///     read <c>'c'</c> and <c>'S1'</c> as NumSharp's 2-byte <c>Char</c> (NumPy: one-byte strings).
        /// </summary>
        /// <param name="dtype">The dtype string.</param>
        /// <returns>The string descriptor, or null.</returns>
        private static DType StringDescrOrNull(string dtype)
        {
            if (string.IsNullOrEmpty(dtype))
                return null;
            DType descr;
            try
            {
                descr = np.dtype(dtype);
            }
            catch (Exception e) when (e is NotSupportedException || e is ArgumentException || e is TypeError)
            {
                // Not a dtype np.dtype knows: the numeric parser reports it with frombuffer's own text.
                return null;
            }
            // 'T' (StringDType) is a reference dtype: PyArray_FromBuffer's first check refuses it.
            ThrowIfReferenceDType(descr);
            return NDStringStorage.IsString(descr) ? descr : null;
        }

        /// <summary>
        ///     NumPy's first check of <c>PyArray_FromBuffer</c>: a dtype holding references (<c>PyDataType_REFCHK</c> — the NEP 55
        ///     <c>StringDType</c>, whose cells are arena offsets and heap pointers) cannot be laid over a buffer's bytes.
        /// </summary>
        /// <param name="dtype">The requested descriptor (null is the default float64).</param>
        /// <exception cref="ValueError">NumPy's <c>cannot create an OBJECT array from memory buffer</c>.</exception>
        internal static void ThrowIfReferenceDType(DType dtype)
        {
            if (dtype?.Meta is VStringDTypeMeta)
                throw new ValueError("cannot create an OBJECT array from memory buffer");
        }
    }
}
