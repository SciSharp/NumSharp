using System;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        /// <summary>
        ///     A storage that lays <paramref name="layout"/> over <paramref name="slice"/>, memory it does not own, with
        ///     <paramref name="baseStorage"/> as its base — NumPy's <c>PyArray_NewFromDescr_int</c> with a data pointer and a
        ///     base object, the storage of <c>np.ndarray(shape, buffer=...)</c>'s result.
        /// </summary>
        /// <param name="slice">The elements: the buffer array's own slice (when the dtype and element grid match), or a
        ///     forwarding alias over the span the array occupies (<see cref="ArraySlice.WrapShared(NPTypeCode, void*, long, IArraySlice)"/>).
        ///     Its element type is the storage's dtype.</param>
        /// <param name="layout">The dimensions, element strides, offset and flags, as given. It is not validated: the
        ///     caller has proved that every element it addresses lies inside <paramref name="slice"/>.</param>
        /// <param name="baseStorage">The array's base (NumPy's collapsed <c>base</c>): the buffer array's owner, or the
        ///     storage that stands for a foreign buffer.</param>
        /// <returns>A storage that does not own its data (no <c>OWNDATA</c>), with the flags of
        ///     <paramref name="layout"/>.</returns>
        /// <remarks>
        ///     Unlike <see cref="Alias(Shape)"/>, this neither inherits writeability nor forces <c>ALIGNED</c>: the caller
        ///     states <c>WRITEABLE</c> in <paramref name="layout"/> (NumPy's rule for a buffer is "writeable exactly when the
        ///     buffer is", which a stride-0 axis does not change), and <c>flags.aligned</c> reads the memory's real
        ///     alignment (<see cref="IsDataAligned"/>). One storage is built, not a chain of reinterpreting aliases.
        /// </remarks>
        internal static UnmanagedStorage OverBuffer(IArraySlice slice, Shape layout, UnmanagedStorage baseStorage)
        {
            var r = new UnmanagedStorage();
            r._shape = layout;
            // The type code first: SetInternalArray dispatches on it to fill the typed-slice union.
            r._typecode = slice.TypeCode;
            r._dtype = r._typecode.AsType();
            r.SetInternalArray(slice);
            // Count is the slice's capacity, as SetShapeUnsafe sets it for a strided or broadcast layout.
            r.Count = layout.bufferSize;
            r._baseStorage = baseStorage;
            // a view of an array's buffer keeps that array's engine; over a C# buffer there is none, and the NDArray built
            // on this storage binds TensorEngine.Default
            r.Engine = baseStorage?.Engine;
            r.OnReshaped(); // a buffer view never owns its memory (NumPy: data passed in clears OWNDATA)
            return r;
        }

        // ---------------------------------------------------------------------------------------------
        // ctors.c's PyArray_FromBuffer, the body of np.frombuffer (the np-layer dtype plan's Appendix D home): np.frombuffer's
        // overloads convert their arguments (array_frombuffer's part) and call one of these, by the C# kind of buffer:
        //   - FromBufferObject, for every buffer np.ndarray takes (a byte[] or typed C# array, a MemoryView, a Memory<byte>,
        //     an ArraySegment<byte>);
        //   - FromBufferPointer, for an address and its byte length (np.ndarray takes no address);
        //   - FromBufferSpan, for a ReadOnlySpan<byte> (a span cannot stay pinned past the call: a copy).
        // Each runs NumPy's checks in its order:
        //   - the descriptor: a dtype holding references (PyDataType_REFCHK), then an unsized one (PyDataType_ISUNSIZED);
        //   - the buffer: a C# null is refused, a memoryview must be C-contiguous (PyBUF_SIMPLE's BufferError);
        //   - the range checks (FromBufferCount): the offset, then the item count - every whole item after the offset for a
        //     negative count, which must leave no partial item; else the count must fit, whatever bytes follow it.
        // Then the array: np.ndarray's 1-D view over the bytes (ArrayNew: it pins a managed buffer, borrows a NumSharp one
        // through its reference counting, does not own its data, and is writeable exactly when the buffer is). NumSharp
        // stores native values only, so a non-native descriptor ('>i4') is decoded into a new array (NumPy keeps the byte
        // order in the result's descriptor).
        // ---------------------------------------------------------------------------------------------

        /// <summary>NumPy's text when the descriptor has no width (<c>np.frombuffer(b, 'S')</c>, <c>np.str_</c>).</summary>
        private const string FromBufferUnsizedText = "itemsize cannot be zero in type";

        /// <summary>
        ///     NumPy's <c>PyArray_FromBuffer</c> (<c>ctors.c</c>) for every buffer <c>np.ndarray</c> takes (a C# array of
        ///     unmanaged elements, a <see cref="np.MemoryView"/>, a <see cref="Memory{T}"/> or <see cref="ArraySegment{T}"/> of
        ///     bytes): the descriptor checks, the buffer, the range checks, then np.ndarray's 1-D view of the bytes.
        /// </summary>
        /// <param name="buffer">The buffer object, as <c>np.ndarray</c>'s <c>buffer</c> argument; null is refused after the
        ///     descriptor checks.</param>
        /// <param name="bufferLength">Its length in bytes (the bound the range checks use; ignored for a null buffer).</param>
        /// <param name="dtype">The descriptor the bytes are read as; null is float64 (NumPy's <c>NPY_DEFAULT_TYPE</c>).</param>
        /// <param name="count">The requested item count; negative means every whole item after the offset.</param>
        /// <param name="offset">The byte offset of the first item.</param>
        /// <param name="bufferName">The C# parameter name a null buffer is reported under.</param>
        /// <returns>
        ///     A 1-D view of the buffer that does not own its data, writeable when the buffer is (a read-only memoryview's
        ///     result is read-only for good); for a non-native descriptor, a decoded native copy.
        /// </returns>
        /// <exception cref="ValueError">A descriptor holding references or without a width, or the range checks of
        ///     <see cref="FromBufferCount"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="buffer"/> is null.</exception>
        /// <exception cref="BufferError">A <see cref="np.MemoryView"/> that is not C-contiguous.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage.</exception>
        /// <remarks>
        ///     Once these checks pass, np.ndarray's own cannot fail: the result's bytes lie inside the buffer. A managed
        ///     buffer is pinned until the last array over it is disposed or collected.
        /// </remarks>
        internal static NDArray FromBufferObject(object buffer, long bufferLength, DType dtype, long count, long offset, string bufferName)
        {
            // array_frombuffer: no dtype given is NPY_DEFAULT_TYPE.
            dtype ??= NumSharp.DType.Double;
            // PyArray_FromBuffer's descriptor checks come before it asks for the buffer.
            ThrowIfReferenceDType(dtype);
            if (dtype.itemsize == 0)
                throw new ValueError(FromBufferUnsizedText);

            // PyObject_GetBuffer: None is no buffer (NumPy's TypeError; C# null is refused at the boundary as everywhere),
            // and a memoryview exports only a C-contiguous buffer for PyBUF_SIMPLE (np.ndarray's own converter also takes an
            // F-contiguous one, so the check is made here).
            if (buffer is null)
                throw new ArgumentNullException(bufferName);
            var view = buffer as np.MemoryView;
            if (view != null && !view.c_contiguous)
                throw new BufferError("memoryview: underlying buffer is not C-contiguous");

            long elements = FromBufferCount(bufferLength, dtype.itemsize, count, offset);
            // np.ndarray's body for a computed shape (its facade only refuses a None shape).
            var result = ArrayNew(new Shape(elements), dtype, buffer, offset, null, 'C');

            // A read-only exporter refuses a writable buffer for good: setflags(write=True) on the result must fail, as
            // NumPy's _IsWriteable refuses a non-writable buffer base. A decoded copy (a non-native descriptor) is the
            // caller's own and stays writeable.
            if (view != null && view.@readonly && dtype.isnative)
            {
                result.Storage.WriteProtected = true;
                if (result.Shape.IsWriteable)
                    result.Storage.SetShapeUnsafe(result.Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));
            }

            return result;
        }

        /// <summary>
        ///     NumPy's range checks of <c>PyArray_FromBuffer</c> over a buffer of <paramref name="bufferLength"/> bytes, in its
        ///     order: the offset, then the item count (every whole item after the offset for a negative count, which must
        ///     leave no partial item; else the count must fit, whatever bytes follow it).
        /// </summary>
        /// <param name="bufferLength">The buffer's length in bytes.</param>
        /// <param name="itemSize">The descriptor's item size in bytes (not 0: the caller refused an unsized descriptor).</param>
        /// <param name="count">The requested item count; negative means "every item after the offset".</param>
        /// <param name="offset">The byte offset of the first item.</param>
        /// <returns>The number of items the result has.</returns>
        /// <exception cref="ValueError">NumPy's texts: <c>offset must be non-negative and no greater than buffer length
        ///     (N)</c>, <c>buffer size must be a multiple of element size</c>, or <c>buffer is smaller than requested
        ///     size</c>.</exception>
        private static long FromBufferCount(long bufferLength, int itemSize, long count, long offset)
        {
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
        ///     <c>PyArray_FromBuffer</c> over unmanaged memory: the same checks as <see cref="FromBufferObject"/>, then a 1-D
        ///     array over the memory, or one that owns it through <paramref name="dispose"/>.
        /// </summary>
        /// <param name="address">The first byte of the memory.</param>
        /// <param name="byteLength">The memory's length in bytes.</param>
        /// <param name="dtype">The descriptor the bytes are read as; null is float64.</param>
        /// <param name="count">Items to read, or a negative number for all.</param>
        /// <param name="offset">The byte offset of the first item.</param>
        /// <param name="dispose">Frees the memory when the array is released, or null when the caller keeps it alive (the
        ///     result then does not own its data, as NumPy's array over a foreign buffer).</param>
        /// <returns>The 1-D array; for a non-native descriptor a native copy decoded by the class's
        ///     <see cref="DTypeMeta.copyswapn"/> (the memory is then released at once when <paramref name="dispose"/> was
        ///     given); for zero items with <paramref name="dispose"/> a fresh empty array (the memory is released at once),
        ///     without it an empty view that does not own its data.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="address"/> is null (after the descriptor checks).</exception>
        /// <exception cref="ValueError">A descriptor holding references or without a width, or the range checks of
        ///     <see cref="FromBufferCount"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage.</exception>
        /// <remarks>On a refused request <paramref name="dispose"/> is not called: the caller still owns the memory. The
        ///     items are typed slices over ONE byte block, which alone frees the memory (or owns nothing).</remarks>
        internal static unsafe NDArray FromBufferPointer(byte* address, long byteLength, DType dtype, long count, long offset, Action dispose)
        {
            dtype ??= NumSharp.DType.Double;
            ThrowIfReferenceDType(dtype);
            if (dtype.itemsize == 0)
                throw new ValueError(FromBufferUnsizedText);
            if (address == null)
                throw new ArgumentNullException(nameof(address));

            long elements = FromBufferCount(byteLength, dtype.itemsize, count, offset);
            // NumSharp stores native values only: the array is allocated or laid out with the native descriptor.
            DType native = dtype.isnative ? dtype : dtype.Meta.ensure_canonical(dtype);
            if (!dtype.isnative)
            {
                // A non-native descriptor: the items are decoded into a new array by the class's byte-order copy (NumPy's
                // copyswapn with swap), and nothing of the memory is kept, so it is released at once.
                var decoded = new NDArray(native, Shape.Vector(elements), false);
                dtype.Meta.copyswapn(dtype, (byte*)decoded.Storage.Address, dtype.itemsize, address + offset, dtype.itemsize,
                    elements, swap: true);
                dispose?.Invoke();
                return decoded;
            }
            if (elements == 0 && dispose != null)
            {
                // Nothing of the memory is used and the caller handed it over: it is released at once, and the result is a
                // fresh empty array. Without dispose the empty result is still a view of the caller's memory (NumPy's: it
                // does not own its data), built below over zero bytes.
                dispose();
                return new NDArray(native, Shape.Vector(0), false);
            }

            long bytes = elements * dtype.itemsize;
            var block = dispose != null
                ? new UnmanagedMemoryBlock<byte>(address + offset, bytes, dispose)
                : new UnmanagedMemoryBlock<byte>(address + offset, bytes);
            var byteSlice = new ArraySlice<byte>(block);
            // The items over the byte block: a string slice reads the bytes as the descriptor's width; any other lane is a
            // typed alias whose reference counting forwards to the byte block, so the block alone frees the memory.
            IArraySlice items = NDStringStorage.IsString(native)
                ? new StringArraySlice(byteSlice, native)
                : ArraySlice.WrapShared(native.GetTypeCode(), address + offset, elements, byteSlice);
            var storage = new UnmanagedStorage(items, Shape.Vector(elements));
            if (dispose == null)
            {
                // The memory is the caller's (NumPy's non-array base object): the array does not own its data.
                storage.ExternalBase = true;
                storage.OnReshaped();
            }

            return new NDArray(storage);
        }

        /// <summary>
        ///     <c>PyArray_FromBuffer</c> over a <see cref="ReadOnlySpan{T}"/>: a span cannot be pinned past the call, so the
        ///     selected bytes are COPIED into a new array that owns them, after the same checks as
        ///     <see cref="FromBufferObject"/>.
        /// </summary>
        /// <param name="buffer">The bytes.</param>
        /// <param name="dtype">The descriptor the bytes are read as; null is float64.</param>
        /// <param name="count">Items to read, or a negative number for all.</param>
        /// <param name="offset">The byte offset of the first item.</param>
        /// <returns>A new 1-D array (native byte order: a non-native request is decoded by the class's
        ///     <see cref="DTypeMeta.copyswapn"/>).</returns>
        /// <exception cref="ValueError">A descriptor holding references or without a width, or the range checks of
        ///     <see cref="FromBufferCount"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage.</exception>
        internal static unsafe NDArray FromBufferSpan(ReadOnlySpan<byte> buffer, DType dtype, long count, long offset)
        {
            dtype ??= NumSharp.DType.Double;
            ThrowIfReferenceDType(dtype);
            if (dtype.itemsize == 0)
                throw new ValueError(FromBufferUnsizedText);

            long elements = FromBufferCount(buffer.Length, dtype.itemsize, count, offset);
            var result = new NDArray(dtype.isnative ? dtype : dtype.Meta.ensure_canonical(dtype), Shape.Vector(elements), false);
            if (elements == 0)
                return result;

            // The count check put the items inside the span, so offset indexes a real byte. The class's byte-order copy
            // (NumPy's copyswapn) copies them and, for a non-native descriptor, reverses each item's units into host order.
            fixed (byte* src = &buffer[(int)offset])
                dtype.Meta.copyswapn(dtype, (byte*)result.Storage.Address, dtype.itemsize, src, dtype.itemsize, elements,
                    swap: !dtype.isnative);
            return result;
        }

        /// <summary>
        ///     The descriptor an <see cref="NPTypeCode"/> overload's dtype names (array_frombuffer's descriptor conversion).
        /// </summary>
        /// <param name="dtype">The storage lane.</param>
        /// <param name="whenEmpty">The descriptor <see cref="NPTypeCode.Empty"/> stands for: null (float64) for a byte buffer,
        ///     the element type for the generic overload.</param>
        /// <returns>The lane's descriptor, or <paramref name="whenEmpty"/>.</returns>
        /// <exception cref="ValueError"><see cref="NPTypeCode.String"/>: the string lane has no width, so it stands for NumPy's
        ///     unsized class (<c>np.str_</c>), which <c>PyArray_FromBuffer</c> refuses as <c>itemsize cannot be zero in
        ///     type</c> before it reads the buffer.</exception>
        internal static DType FromBufferDescr(NPTypeCode dtype, DType whenEmpty)
        {
            if (dtype == NPTypeCode.String)
                throw new ValueError(FromBufferUnsizedText);
            return dtype == NPTypeCode.Empty ? whenEmpty : NumSharp.DType.From(dtype);
        }

        /// <summary>
        ///     NumPy's first check of <c>PyArray_FromBuffer</c>: a dtype whose items hold references
        ///     (<c>PyDataType_REFCHK</c>, the class's <see cref="DTypeMeta.HasReferences"/> - NEP 55's <c>StringDType</c>, whose
        ///     cells are arena offsets and heap pointers) cannot be laid over a buffer's bytes.
        /// </summary>
        /// <param name="dtype">The requested descriptor (null is the default float64).</param>
        /// <exception cref="ValueError">NumPy's <c>cannot create an OBJECT array from memory buffer</c>.</exception>
        internal static void ThrowIfReferenceDType(DType dtype)
        {
            if (dtype is not null && dtype.Meta.HasReferences)
                throw new ValueError("cannot create an OBJECT array from memory buffer");
        }
    }
}
