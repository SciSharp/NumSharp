using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        // -------------------------------------------------------------------------------------------------
        // np.frombuffer: NumPy's array_frombuffer (multiarraymodule.c), which converts its arguments and calls
        // ctors.c's PyArray_FromBuffer. NumPy has ONE frombuffer over any object that exports a buffer; NumSharp's
        // overloads are the C# spellings of a buffer (a byte[] or any unmanaged-element C# array, a MemoryView, a
        // Memory<byte> or ArraySegment<byte>, an address with a byte length, a ReadOnlySpan<byte>) and of a dtype
        // (a descriptor, an NPTypeCode, a name read by np.dtype as PyArray_DescrConverter reads it). Every overload
        // converts its arguments and runs PyArray_FromBuffer's body, UnmanagedStorage's FromBufferObject /
        // FromBufferPointer / FromBufferSpan (Backends/Unmanaged/UnmanagedStorage.Buffer.cs): the descriptor checks, the
        // buffer, the range checks, then the array over the bytes.
        // -------------------------------------------------------------------------------------------------

        /// <summary>
        /// Interpret a buffer as a 1-dimensional array.
        /// </summary>
        /// <param name="buffer">The bytes (a C# <c>byte[]</c> is mutable, NumPy's <c>bytearray</c>: the result is writeable).</param>
        /// <param name="dtype">Data-type of the returned array; null is float64 (NumPy's default).</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>
        ///     A 1-D array that VIEWS the buffer (it pins the array and shares its memory: a write through it changes
        ///     <paramref name="buffer"/>); it does not own its data. For a non-native <paramref name="dtype"/> (<c>'&gt;i4'</c>)
        ///     a decoded native copy, because NumSharp stores every array in host byte order.
        /// </returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html
        ///
        /// The buffer stays pinned until the last array over it is disposed or collected.
        /// </remarks>
        /// <exception cref="ValueError">NumPy's texts, in its order: <c>cannot create an OBJECT array from memory buffer</c> (a
        ///     NEP 55 <c>StringDType</c>, whose cells hold references); <c>itemsize cannot be zero in type</c> (an unsized
        ///     string dtype); <c>offset must be non-negative and no greater than buffer length (N)</c>; <c>buffer size must be
        ///     a multiple of element size</c> (a negative <paramref name="count"/> that leaves a partial item);
        ///     <c>buffer is smaller than requested size</c>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="buffer"/> is null (checked after the dtype, as NumPy asks for
        ///     the buffer after its descriptor checks).</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage.</exception>
        public static NDArray frombuffer(byte[] buffer, DType dtype = null, long count = -1, long offset = 0)
            => UnmanagedStorage.FromBufferObject(buffer, buffer?.LongLength ?? 0, dtype, count, offset, nameof(buffer));

        /// <summary>
        /// Interpret a buffer as a 1-dimensional array, the dtype given as a storage lane.
        /// </summary>
        /// <param name="buffer">The bytes (writable: the result is writeable).</param>
        /// <param name="dtype">Data-type of the returned array: a storage lane; <see cref="NPTypeCode.Empty"/> is float64, and
        ///     <see cref="NPTypeCode.String"/> (a string lane without a width) is NumPy's unsized <c>np.str_</c>, refused.</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>A 1-D array that views the buffer (see <see cref="frombuffer(byte[], DType, long, long)"/>).</returns>
        /// <exception cref="ValueError">The texts of <see cref="frombuffer(byte[], DType, long, long)"/>;
        ///     <see cref="NPTypeCode.String"/> raises <c>itemsize cannot be zero in type</c> before the buffer is read.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="buffer"/> is null.</exception>
        public static NDArray frombuffer(byte[] buffer, NPTypeCode dtype, long count = -1, long offset = 0)
            => UnmanagedStorage.FromBufferObject(buffer, buffer?.LongLength ?? 0, UnmanagedStorage.FromBufferDescr(dtype, whenEmpty: null), count, offset, nameof(buffer));

        /// <summary>
        /// Interpret a <see cref="MemoryView"/> (obtained from <see cref="NDArray.data"/>) as a
        /// 1-dimensional array, sharing its memory (zero-copy) — the consumer side of <c>ndarray.data</c>.
        /// </summary>
        /// <param name="buffer">A <see cref="MemoryView"/> over a C-contiguous array.</param>
        /// <param name="dtype">Data-type of the returned array; null is float64. Any dtype reads the array's bytes, whatever
        ///     the array's own dtype.</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>
        ///     A 1-D <see cref="NDArray"/> that VIEWS the buffer's memory (writes through to the source); its base is the
        ///     source array's owner, which stays alive while the result does. Read-only for good when the memoryview is (a
        ///     broadcast or otherwise read-only source): <c>setflags(write: true)</c> refuses it. For a non-native
        ///     <paramref name="dtype"/>, a decoded native copy.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html</remarks>
        /// <exception cref="ValueError">The texts of <see cref="frombuffer(byte[], DType, long, long)"/>, in its order.</exception>
        /// <exception cref="BufferError">The memoryview is not C-contiguous (NumPy's <c>memoryview: underlying buffer is not
        ///     C-contiguous</c>, raised after the dtype checks and before the range checks).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="buffer"/> is null.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage.</exception>
        public static NDArray frombuffer(MemoryView buffer, DType dtype = null, long count = -1, long offset = 0)
            => UnmanagedStorage.FromBufferObject(buffer, buffer?.nbytes ?? 0, dtype, count, offset, nameof(buffer));

        /// <summary>
        /// Interpret a <see cref="MemoryView"/> as a 1-dimensional array, the dtype given as a storage lane.
        /// </summary>
        /// <param name="buffer">A <see cref="MemoryView"/> over a C-contiguous array.</param>
        /// <param name="dtype">Data-type of the returned array: a storage lane; <see cref="NPTypeCode.Empty"/> is float64, and
        ///     <see cref="NPTypeCode.String"/> is NumPy's unsized <c>np.str_</c>, refused.</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>A 1-D array that views the buffer (see <see cref="frombuffer(MemoryView, DType, long, long)"/>).</returns>
        /// <exception cref="ValueError">The texts of <see cref="frombuffer(byte[], DType, long, long)"/>.</exception>
        /// <exception cref="BufferError">The memoryview is not C-contiguous.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="buffer"/> is null.</exception>
        public static NDArray frombuffer(MemoryView buffer, NPTypeCode dtype, long count = -1, long offset = 0)
            => UnmanagedStorage.FromBufferObject(buffer, buffer?.nbytes ?? 0, UnmanagedStorage.FromBufferDescr(dtype, whenEmpty: null), count, offset, nameof(buffer));

        /// <summary>
        /// Interpret a <see cref="MemoryView"/> as a 1-dimensional array, given a dtype STRING
        /// (e.g. <c>"&lt;i4"</c>, <c>"&gt;u4"</c> for big-endian uint32, <c>"S3"</c>).
        /// </summary>
        /// <param name="buffer">A <see cref="MemoryView"/> over a C-contiguous array.</param>
        /// <param name="dtype">The dtype's name, read by <see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/>
        ///     (NumPy's <c>PyArray_DescrConverter</c>: a sized or one-letter code with an optional byte-order prefix, a
        ///     NumPy name, a fixed-width string); null is float64 (NumPy's <c>None</c>).</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>A 1-D array that views the buffer; a decoded native copy for a non-native name.</returns>
        /// <exception cref="ValueError"><paramref name="dtype"/> spells NEP 55 <c>StringDType</c> (<c>"T"</c>): NumPy's <c>cannot
        ///     create an OBJECT array from memory buffer</c>; or the texts of <see cref="frombuffer(byte[], DType, long, long)"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> names no dtype (np.dtype's <c>data type 'x' not
        ///     understood</c>, NumPy's TypeError, the empty name included) or one NumSharp does not have (complex64); raised
        ///     before the buffer is read.</exception>
        /// <exception cref="BufferError">The memoryview is not C-contiguous.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="buffer"/> is null.</exception>
        public static NDArray frombuffer(MemoryView buffer, string dtype, long count = -1, long offset = 0)
            => UnmanagedStorage.FromBufferObject(buffer, buffer?.nbytes ?? 0, FromBufferNameDescr(dtype), count, offset, nameof(buffer));

        /// <summary>
        /// Interpret a buffer as a 1-dimensional array, given a dtype STRING (e.g. <c>"&gt;u4"</c> for big-endian uint32).
        /// </summary>
        /// <param name="buffer">The bytes (writable: the result is writeable).</param>
        /// <param name="dtype">The dtype's name, read by <see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/>
        ///     (NumPy's <c>PyArray_DescrConverter</c>: a sized or one-letter code with an optional byte-order prefix, a
        ///     NumPy name, a fixed-width string); null is float64 (NumPy's <c>None</c>).</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>A 1-D array that views the buffer; a decoded native copy for a big-endian name (<c>"&gt;u4"</c>).</returns>
        /// <exception cref="ValueError"><paramref name="dtype"/> spells NEP 55 <c>StringDType</c> (<c>"T"</c>): NumPy's <c>cannot
        ///     create an OBJECT array from memory buffer</c>; or the texts of <see cref="frombuffer(byte[], DType, long, long)"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> names no dtype (np.dtype's <c>data type 'x' not
        ///     understood</c>, NumPy's TypeError, the empty name included) or one NumSharp does not have (complex64); raised
        ///     before the buffer is read.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="buffer"/> is null.</exception>
        public static NDArray frombuffer(byte[] buffer, string dtype, long count = -1, long offset = 0)
            => UnmanagedStorage.FromBufferObject(buffer, buffer?.LongLength ?? 0, FromBufferNameDescr(dtype), count, offset, nameof(buffer));

        /// <summary>
        /// Interpret a ReadOnlySpan as a 1-dimensional array.
        /// Note: ReadOnlySpan cannot be pinned, so this always creates a copy.
        /// </summary>
        /// <param name="buffer">The bytes.</param>
        /// <param name="dtype">Data-type of the returned array; null is float64.</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>A new 1-D array that owns a copy of the selected bytes (decoded to native order for a non-native dtype).</returns>
        /// <exception cref="ValueError">The texts of <see cref="frombuffer(byte[], DType, long, long)"/>, in its order.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage.</exception>
        public static NDArray frombuffer(ReadOnlySpan<byte> buffer, DType dtype = null, long count = -1, long offset = 0)
            => UnmanagedStorage.FromBufferSpan(buffer, dtype, count, offset);

        /// <summary>
        /// Interpret a ReadOnlySpan as a 1-dimensional array, the dtype given as a storage lane.
        /// Note: ReadOnlySpan cannot be pinned, so this always creates a copy.
        /// </summary>
        /// <param name="buffer">The bytes.</param>
        /// <param name="dtype">Data-type of the returned array: a storage lane; <see cref="NPTypeCode.Empty"/> is float64, and
        ///     <see cref="NPTypeCode.String"/> is NumPy's unsized <c>np.str_</c>, refused.</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>A new 1-D array that owns a copy of the selected bytes.</returns>
        /// <exception cref="ValueError">The texts of <see cref="frombuffer(byte[], DType, long, long)"/>.</exception>
        public static NDArray frombuffer(ReadOnlySpan<byte> buffer, NPTypeCode dtype, long count = -1, long offset = 0)
            => UnmanagedStorage.FromBufferSpan(buffer, UnmanagedStorage.FromBufferDescr(dtype, whenEmpty: null), count, offset);

        #region .NET-friendly overloads

        /// <summary>
        /// Interpret an ArraySegment as a 1-dimensional array: the segment IS the buffer (NumPy's frombuffer of a sliced
        /// memoryview), so its items end at the segment's end, not at the end of the array behind it.
        /// </summary>
        /// <param name="segment">The array segment to interpret. A default segment (no array) is an empty buffer, as
        ///     NumPy's <c>b''</c> is.</param>
        /// <param name="dtype">Data-type of the returned array; null is float64.</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item of the segment.</param>
        /// <returns>A 1-D array viewing the segment's bytes (it pins the array behind them; writeable).</returns>
        /// <exception cref="ValueError">The texts of <see cref="frombuffer(byte[], DType, long, long)"/>, in its order.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage.</exception>
        public static NDArray frombuffer(ArraySegment<byte> segment, DType dtype = null, long count = -1)
            => UnmanagedStorage.FromBufferObject(segment, segment.Count, dtype, count, 0, nameof(segment));

        /// <summary>
        /// Interpret an ArraySegment as a 1-dimensional array, the dtype given as a storage lane.
        /// </summary>
        /// <param name="segment">The array segment to interpret (a default segment is an empty buffer).</param>
        /// <param name="dtype">Data-type of the returned array: a storage lane; <see cref="NPTypeCode.Empty"/> is float64, and
        ///     <see cref="NPTypeCode.String"/> is NumPy's unsized <c>np.str_</c>, refused.</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item of the segment.</param>
        /// <returns>A 1-D array viewing the segment's bytes.</returns>
        /// <exception cref="ValueError">The texts of <see cref="frombuffer(byte[], DType, long, long)"/>.</exception>
        public static NDArray frombuffer(ArraySegment<byte> segment, NPTypeCode dtype, long count = -1)
            => UnmanagedStorage.FromBufferObject(segment, segment.Count, UnmanagedStorage.FromBufferDescr(dtype, whenEmpty: null), count, 0, nameof(segment));

        /// <summary>
        /// Interpret a Memory&lt;byte&gt; as a 1-dimensional array: a view of its bytes, whether an array or a memory
        /// manager backs it (the memory is pinned while an array over it lives).
        /// </summary>
        /// <param name="memory">The memory to interpret; it bounds the buffer (the bytes of an array behind it past its end
        ///     are never read).</param>
        /// <param name="dtype">Data-type of the returned array; null is float64.</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Byte offset within the memory. Default is 0.</param>
        /// <returns>A 1-D array viewing the memory (writeable). Footgun: a <see cref="Memory{T}"/> NumSharp hands out over an
        ///     array's buffer (<c>nd.Unsafe.Memory&lt;byte&gt;()</c>) does not keep that array alive, and neither does the
        ///     result.</returns>
        /// <exception cref="ValueError">The texts of <see cref="frombuffer(byte[], DType, long, long)"/>, in its order.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage.</exception>
        public static NDArray frombuffer(Memory<byte> memory, DType dtype = null, long count = -1, long offset = 0)
            => UnmanagedStorage.FromBufferObject(memory, memory.Length, dtype, count, offset, nameof(memory));

        /// <summary>
        /// Interpret a Memory&lt;byte&gt; as a 1-dimensional array, the dtype given as a storage lane.
        /// </summary>
        /// <param name="memory">The memory to interpret.</param>
        /// <param name="dtype">Data-type of the returned array: a storage lane; <see cref="NPTypeCode.Empty"/> is float64, and
        ///     <see cref="NPTypeCode.String"/> is NumPy's unsized <c>np.str_</c>, refused.</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Byte offset within the memory. Default is 0.</param>
        /// <returns>A 1-D array viewing the memory.</returns>
        /// <exception cref="ValueError">The texts of <see cref="frombuffer(byte[], DType, long, long)"/>.</exception>
        public static NDArray frombuffer(Memory<byte> memory, NPTypeCode dtype, long count = -1, long offset = 0)
            => UnmanagedStorage.FromBufferObject(memory, memory.Length, UnmanagedStorage.FromBufferDescr(dtype, whenEmpty: null), count, offset, nameof(memory));

        /// <summary>
        /// Interpret unmanaged memory at a pointer as a 1-dimensional array.
        /// </summary>
        /// <param name="address">Pointer to the start of the buffer.</param>
        /// <param name="byteLength">Total length of the buffer in bytes.</param>
        /// <param name="dtype">Data-type of the returned array; null is float64.</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Byte offset into the buffer. Default is 0.</param>
        /// <param name="dispose">
        /// Optional cleanup action called when NDArray is disposed.
        /// Use to transfer ownership: dispose: () => Marshal.FreeHGlobal(ptr)
        /// If null, caller is responsible for memory lifetime (view semantics).
        /// </param>
        /// <returns>
        ///     A 1-D array over the memory: without <paramref name="dispose"/> a view that does not own its data (NumPy's array
        ///     over a foreign buffer), with it an array that owns the memory and frees it through <paramref name="dispose"/>
        ///     when the last array over it is released. An empty result with <paramref name="dispose"/> is a fresh empty array,
        ///     and the memory is released at once through it. For a non-native <paramref name="dtype"/>, a decoded native
        ///     copy (the memory is then released at once too).
        /// </returns>
        /// <example>
        /// // View only (caller manages lifetime):
        /// var arr = np.frombuffer(ptr, length, typeof(float));
        ///
        /// // Take ownership (NumSharp frees on dispose):
        /// var ptr = Marshal.AllocHGlobal(1024);
        /// var arr = np.frombuffer(ptr, 1024, typeof(float), dispose: () => Marshal.FreeHGlobal(ptr));
        /// </example>
        /// <exception cref="ValueError">The texts of <see cref="frombuffer(byte[], DType, long, long)"/>, in its order: a count
        ///     past the memory is refused (<c>buffer is smaller than requested size</c>), never capped. On a refused request
        ///     <paramref name="dispose"/> is not called: the caller still owns the memory.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="address"/> is null.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage.</exception>
        public static unsafe NDArray frombuffer(IntPtr address, long byteLength, DType dtype = null, long count = -1, long offset = 0, Action dispose = null)
            => UnmanagedStorage.FromBufferPointer((byte*)address, byteLength, dtype, count, offset, dispose);

        /// <summary>
        /// Interpret unmanaged memory at a pointer as a 1-dimensional array, the dtype given as a storage lane.
        /// </summary>
        /// <param name="address">Pointer to the start of the buffer.</param>
        /// <param name="byteLength">Total length of the buffer in bytes.</param>
        /// <param name="dtype">Data-type of the returned array: a storage lane; <see cref="NPTypeCode.Empty"/> is float64, and
        ///     <see cref="NPTypeCode.String"/> is NumPy's unsized <c>np.str_</c>, refused.</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Byte offset into the buffer. Default is 0.</param>
        /// <param name="dispose">Optional cleanup action that transfers ownership of the memory to the array (see
        ///     <see cref="frombuffer(IntPtr, long, DType, long, long, Action)"/>).</param>
        /// <returns>A 1-D array over the memory (see <see cref="frombuffer(IntPtr, long, DType, long, long, Action)"/>).</returns>
        /// <exception cref="ValueError">The texts of <see cref="frombuffer(byte[], DType, long, long)"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="address"/> is null.</exception>
        public static unsafe NDArray frombuffer(IntPtr address, long byteLength, NPTypeCode dtype, long count = -1, long offset = 0, Action dispose = null)
            => UnmanagedStorage.FromBufferPointer((byte*)address, byteLength, UnmanagedStorage.FromBufferDescr(dtype, whenEmpty: null), count, offset, dispose);

        /// <summary>
        /// Interpret unmanaged memory at a pointer as a 1-dimensional array.
        /// </summary>
        /// <param name="address">Pointer to the start of the buffer.</param>
        /// <param name="byteLength">Total length of the buffer in bytes.</param>
        /// <param name="dtype">Data-type of the returned array. Default is float64.</param>
        /// <param name="count">Number of items to read. -1 means all data.</param>
        /// <param name="offset">Byte offset into the buffer. Default is 0.</param>
        /// <param name="dispose">Optional cleanup action called when NDArray is disposed.</param>
        /// <returns>1-dimensional NDArray viewing/owning the memory.</returns>
        /// <exception cref="ValueError">See <see cref="frombuffer(IntPtr, long, DType, long, long, Action)"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="address"/> is null.</exception>
        public static unsafe NDArray frombuffer(void* address, long byteLength, DType dtype = null, long count = -1, long offset = 0, Action dispose = null)
        {
            return frombuffer((IntPtr)address, byteLength, dtype, count, offset, dispose);
        }

        /// <summary>
        /// Interpret unmanaged memory at a pointer as a 1-dimensional array.
        /// </summary>
        /// <param name="address">Pointer to the start of the buffer.</param>
        /// <param name="byteLength">Total length of the buffer in bytes.</param>
        /// <param name="dtype">Data-type of the returned array, as a storage lane.</param>
        /// <param name="count">Number of items to read. -1 means all data.</param>
        /// <param name="offset">Byte offset into the buffer. Default is 0.</param>
        /// <param name="dispose">Optional cleanup action called when NDArray is disposed.</param>
        /// <returns>1-dimensional NDArray viewing/owning the memory.</returns>
        /// <exception cref="ValueError">See <see cref="frombuffer(IntPtr, long, NPTypeCode, long, long, Action)"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="address"/> is null.</exception>
        public static unsafe NDArray frombuffer(void* address, long byteLength, NPTypeCode dtype, long count = -1, long offset = 0, Action dispose = null)
        {
            return frombuffer((IntPtr)address, byteLength, dtype, count, offset, dispose);
        }

        /// <summary>
        /// Reinterpret a typed array as a different dtype.
        /// Like NumPy's view() but via frombuffer semantics.
        /// </summary>
        /// <typeparam name="TSource">Source element type.</typeparam>
        /// <param name="array">The source array to reinterpret: its bytes are the buffer (pinned while an array over it
        ///     lives).</param>
        /// <param name="dtype">Target data-type. Default (null) preserves the source type - a NumSharp extension: NumPy's
        ///     frombuffer of an ndarray reads float64.</param>
        /// <param name="count">Number of items of target dtype. A negative number reads every whole item.</param>
        /// <param name="offset">Byte offset. Default is 0.</param>
        /// <returns>1-dimensional NDArray viewing the array as the target dtype.</returns>
        /// <example>
        /// var ints = new int[] { 1, 2, 3, 4 };
        /// var asBytes = np.frombuffer(ints, typeof(byte));  // 16 bytes
        /// var asFloats = np.frombuffer(ints, typeof(float)); // 4 floats (same bits)
        /// </example>
        /// <exception cref="ValueError">The texts of <see cref="frombuffer(byte[], DType, long, long)"/>, in its order: a count
        ///     past the array is refused (<c>buffer is smaller than requested size</c>), never capped.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="array"/> is null.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage.</exception>
        public static NDArray frombuffer<TSource>(TSource[] array, DType dtype = null, long count = -1, long offset = 0)
            where TSource : unmanaged
            => UnmanagedStorage.FromBufferObject(array, (array?.LongLength ?? 0) * Unsafe.SizeOf<TSource>(),
                dtype ?? DType.From(InfoOf<TSource>.NPTypeCode), count, offset, nameof(array));

        /// <summary>
        /// Reinterpret a typed array as a different dtype, the dtype given as a storage lane.
        /// </summary>
        /// <typeparam name="TSource">Source element type.</typeparam>
        /// <param name="array">The source array to reinterpret (pinned while an array over it lives).</param>
        /// <param name="dtype">Target data-type: a storage lane; <see cref="NPTypeCode.Empty"/> preserves the source type, and
        ///     <see cref="NPTypeCode.String"/> is NumPy's unsized <c>np.str_</c>, refused.</param>
        /// <param name="count">Number of items of target dtype. A negative number reads every whole item.</param>
        /// <param name="offset">Byte offset. Default is 0.</param>
        /// <returns>1-dimensional NDArray viewing the array as the target dtype.</returns>
        /// <exception cref="ValueError">The texts of <see cref="frombuffer(byte[], DType, long, long)"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="array"/> is null.</exception>
        public static NDArray frombuffer<TSource>(TSource[] array, NPTypeCode dtype, long count = -1, long offset = 0)
            where TSource : unmanaged
            => UnmanagedStorage.FromBufferObject(array, (array?.LongLength ?? 0) * Unsafe.SizeOf<TSource>(),
                UnmanagedStorage.FromBufferDescr(dtype, whenEmpty: DType.From(InfoOf<TSource>.NPTypeCode)), count, offset, nameof(array));

        #endregion

        #region array_frombuffer's descriptor conversion

        /// <summary>
        ///     The descriptor a dtype NAME names (array_frombuffer's <c>PyArray_DescrConverter</c>): np.dtype's reading of
        ///     the name, and null for a null name (NumPy's <c>None</c>), which the body reads as the default float64.
        /// </summary>
        /// <param name="dtype">The dtype name, or null.</param>
        /// <returns>The descriptor, or null for the default.</returns>
        /// <exception cref="NotSupportedException">A name np.dtype does not understand (NumPy's TypeError text, the empty
        ///     name included) or a dtype NumSharp does not have.</exception>
        private static DType FromBufferNameDescr(string dtype) => dtype is null ? null : np.dtype(dtype);

        #endregion
    }
}
