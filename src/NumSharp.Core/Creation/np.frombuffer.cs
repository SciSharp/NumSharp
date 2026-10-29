using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;
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
        // (a descriptor, an NPTypeCode, a name). Every overload converts its arguments and runs the same body:
        //   - the descriptor checks, in NumPy's order: a dtype holding references (PyDataType_REFCHK), then an unsized
        //     one (PyDataType_ISUNSIZED);
        //   - the buffer: a C# null is refused here, a memoryview must be C-contiguous (PyBUF_SIMPLE's BufferError);
        //   - the range checks (FromBufferCount): the offset, then the element count - every whole element after the
        //     offset for a negative count, which must leave no partial element; else the count must fit, whatever the
        //     bytes after it;
        //   - the array: np.ndarray's 1-D view over the buffer's bytes (the port of array_new over a buffer), which pins a
        //     managed buffer, keeps a NumSharp buffer alive through its reference counting, does not own its data and is
        //     writeable exactly when the buffer is. NumSharp stores native values only, so a non-native descriptor
        //     ('>i4') is decoded into a new array (NumPy keeps the byte order in the result's descriptor).
        // The address overloads build their view directly (an address is no object np.ndarray takes), and the span
        // overloads copy (a span cannot stay pinned past the call); both run the same checks.
        // -------------------------------------------------------------------------------------------------

        /// <summary>NumPy's text when the descriptor has no width (<c>np.frombuffer(b, 'S')</c>, <c>np.str_</c>).</summary>
        private const string FromBufferUnsizedText = "itemsize cannot be zero in type";

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
            => FromBufferObject(buffer, buffer?.LongLength ?? 0, dtype, count, offset, nameof(buffer));

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
            => FromBufferObject(buffer, buffer?.LongLength ?? 0, FromBufferDescr(dtype, whenEmpty: null), count, offset, nameof(buffer));

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
            => FromBufferObject(buffer, buffer?.nbytes ?? 0, dtype, count, offset, nameof(buffer));

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
            => FromBufferObject(buffer, buffer?.nbytes ?? 0, FromBufferDescr(dtype, whenEmpty: null), count, offset, nameof(buffer));

        /// <summary>
        /// Interpret a <see cref="MemoryView"/> as a 1-dimensional array, given a dtype STRING
        /// (e.g. <c>"&lt;i4"</c>, <c>"&gt;u4"</c> for big-endian uint32, <c>"S3"</c>).
        /// </summary>
        /// <param name="buffer">A <see cref="MemoryView"/> over a C-contiguous array.</param>
        /// <param name="dtype">The dtype's name: a sized or one-letter code with an optional byte-order prefix, a NumPy name, a
        ///     fixed-width string; null or empty is float64.</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>A 1-D array that views the buffer; a decoded native copy for a non-native name.</returns>
        /// <exception cref="ValueError"><paramref name="dtype"/> spells NEP 55 <c>StringDType</c> (<c>"T"</c>): NumPy's <c>cannot
        ///     create an OBJECT array from memory buffer</c>; or the texts of <see cref="frombuffer(byte[], DType, long, long)"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> names no dtype NumSharp has.</exception>
        /// <exception cref="BufferError">The memoryview is not C-contiguous.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="buffer"/> is null.</exception>
        public static NDArray frombuffer(MemoryView buffer, string dtype, long count = -1, long offset = 0)
            => FromBufferObject(buffer, buffer?.nbytes ?? 0, FromBufferNameDescr(dtype), count, offset, nameof(buffer));

        /// <summary>
        /// Interpret a buffer as a 1-dimensional array, given a dtype STRING (e.g. <c>"&gt;u4"</c> for big-endian uint32).
        /// </summary>
        /// <param name="buffer">The bytes (writable: the result is writeable).</param>
        /// <param name="dtype">The dtype's name: a sized or one-letter code with an optional byte-order prefix, a NumPy name, a
        ///     fixed-width string; null or empty is float64.</param>
        /// <param name="count">Number of items to read. A negative number reads every whole item after <paramref name="offset"/>.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>A 1-D array that views the buffer; a decoded native copy for a big-endian name (<c>"&gt;u4"</c>).</returns>
        /// <exception cref="ValueError"><paramref name="dtype"/> spells NEP 55 <c>StringDType</c> (<c>"T"</c>): NumPy's <c>cannot
        ///     create an OBJECT array from memory buffer</c>; or the texts of <see cref="frombuffer(byte[], DType, long, long)"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> names no dtype NumSharp has.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="buffer"/> is null.</exception>
        public static NDArray frombuffer(byte[] buffer, string dtype, long count = -1, long offset = 0)
            => FromBufferObject(buffer, buffer?.LongLength ?? 0, FromBufferNameDescr(dtype), count, offset, nameof(buffer));

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
            => FromBufferSpan(buffer, dtype, count, offset);

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
            => FromBufferSpan(buffer, FromBufferDescr(dtype, whenEmpty: null), count, offset);

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
            => FromBufferObject(segment, segment.Count, dtype, count, 0, nameof(segment));

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
            => FromBufferObject(segment, segment.Count, FromBufferDescr(dtype, whenEmpty: null), count, 0, nameof(segment));

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
            => FromBufferObject(memory, memory.Length, dtype, count, offset, nameof(memory));

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
            => FromBufferObject(memory, memory.Length, FromBufferDescr(dtype, whenEmpty: null), count, offset, nameof(memory));

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
            => FromBufferPointer((byte*)address, byteLength, dtype, count, offset, dispose);

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
            => FromBufferPointer((byte*)address, byteLength, FromBufferDescr(dtype, whenEmpty: null), count, offset, dispose);

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
            => FromBufferObject(array, (array?.LongLength ?? 0) * Unsafe.SizeOf<TSource>(),
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
            => FromBufferObject(array, (array?.LongLength ?? 0) * Unsafe.SizeOf<TSource>(),
                FromBufferDescr(dtype, whenEmpty: DType.From(InfoOf<TSource>.NPTypeCode)), count, offset, nameof(array));

        #endregion

        #region PyArray_FromBuffer

        /// <summary>
        ///     NumPy's <c>PyArray_FromBuffer</c> (<c>ctors.c</c>) for every buffer <c>np.ndarray</c> takes (a C# array of
        ///     unmanaged elements, a <see cref="MemoryView"/>, a <see cref="Memory{T}"/> or <see cref="ArraySegment{T}"/> of
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
        /// <exception cref="BufferError">A <see cref="MemoryView"/> that is not C-contiguous.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage.</exception>
        /// <remarks>
        ///     Once these checks pass, np.ndarray's own cannot fail: the result's bytes lie inside the buffer. A managed
        ///     buffer is pinned until the last array over it is disposed or collected.
        /// </remarks>
        private static NDArray FromBufferObject(object buffer, long bufferLength, DType dtype, long count, long offset, string bufferName)
        {
            // array_frombuffer: no dtype given is NPY_DEFAULT_TYPE.
            dtype ??= DType.Double;
            // PyArray_FromBuffer's descriptor checks come before it asks for the buffer.
            ThrowIfReferenceDType(dtype);
            if (dtype.itemsize == 0)
                throw new ValueError(FromBufferUnsizedText);

            // PyObject_GetBuffer: None is no buffer (NumPy's TypeError; C# null is refused at the boundary as everywhere),
            // and a memoryview exports only a C-contiguous buffer for PyBUF_SIMPLE (np.ndarray's own converter also takes an
            // F-contiguous one, so the check is made here).
            if (buffer is null)
                throw new ArgumentNullException(bufferName);
            var view = buffer as MemoryView;
            if (view != null && !view.c_contiguous)
                throw new BufferError("memoryview: underlying buffer is not C-contiguous");

            long elements = FromBufferCount(bufferLength, dtype.itemsize, count, offset);
            var result = ndarray(new Shape(elements), dtype, buffer, offset);

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
        /// <returns>The 1-D array; for a non-native descriptor a decoded native copy (the memory is then released at once
        ///     when <paramref name="dispose"/> was given); for zero items with <paramref name="dispose"/> a fresh empty array
        ///     (the memory is released at once), without it an empty view that does not own its data.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="address"/> is null (after the descriptor checks).</exception>
        /// <exception cref="ValueError">A descriptor holding references or without a width, or the range checks of
        ///     <see cref="FromBufferCount"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage.</exception>
        /// <remarks>On a refused request <paramref name="dispose"/> is not called: the caller still owns the memory. The
        ///     items are typed slices over ONE byte block, which alone frees the memory (or owns nothing).</remarks>
        private static unsafe NDArray FromBufferPointer(byte* address, long byteLength, DType dtype, long count, long offset, Action dispose)
        {
            dtype ??= DType.Double;
            ThrowIfReferenceDType(dtype);
            if (dtype.itemsize == 0)
                throw new ValueError(FromBufferUnsizedText);
            if (address == null)
                throw new ArgumentNullException(nameof(address));

            long elements = FromBufferCount(byteLength, dtype.itemsize, count, offset);
            // NumSharp stores native values only: the array is allocated or laid out with the native descriptor, and a
            // non-native request is decoded below.
            DType native = dtype.isnative ? dtype : dtype.Meta.ensure_canonical(dtype);
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

            var view = new NDArray(storage);
            if (dtype.isnative)
                return view;

            // A non-native descriptor: a native decoded copy; the view (and, through dispose, the memory) goes at once.
            using (view)
                return view.byteswap();
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
        /// <returns>A new 1-D array (native byte order: a non-native request is decoded).</returns>
        /// <exception cref="ValueError">A descriptor holding references or without a width, or the range checks of
        ///     <see cref="FromBufferCount"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage.</exception>
        private static unsafe NDArray FromBufferSpan(ReadOnlySpan<byte> buffer, DType dtype, long count, long offset)
        {
            dtype ??= DType.Double;
            ThrowIfReferenceDType(dtype);
            if (dtype.itemsize == 0)
                throw new ValueError(FromBufferUnsizedText);

            long elements = FromBufferCount(buffer.Length, dtype.itemsize, count, offset);
            var result = new NDArray(dtype.isnative ? dtype : dtype.Meta.ensure_canonical(dtype), Shape.Vector(elements), false);
            if (elements == 0)
                return result;

            long bytes = elements * dtype.itemsize;
            // The count check put [offset, offset + bytes) inside the span, so offset indexes a real byte.
            fixed (byte* src = &buffer[(int)offset])
                Buffer.MemoryCopy(src, (void*)result.Storage.Address, bytes, bytes);
            if (!dtype.isnative)
                result.byteswap(inplace: true); // the non-native items into host order, per item
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
        private static DType FromBufferDescr(NPTypeCode dtype, DType whenEmpty)
        {
            if (dtype == NPTypeCode.String)
                throw new ValueError(FromBufferUnsizedText);
            return dtype == NPTypeCode.Empty ? whenEmpty : DType.From(dtype);
        }

        /// <summary>
        ///     The descriptor a dtype NAME names for the string-spelled overloads: a fixed-width string (<see cref="StringDescrOrNull"/>),
        ///     else the code table of <see cref="ParseDtypeString"/>, whose big-endian codes give a non-native descriptor.
        /// </summary>
        /// <param name="dtype">The dtype name; null or empty is float64.</param>
        /// <returns>The descriptor.</returns>
        /// <exception cref="ValueError"><c>"T"</c> (StringDType): <c>cannot create an OBJECT array from memory buffer</c>.</exception>
        /// <exception cref="NotSupportedException">A name the table does not know.</exception>
        private static DType FromBufferNameDescr(string dtype)
        {
            // A string dtype ('S3', '<U2', 'c', 'a5') reads fixed-width string elements; its width is the descriptor's, which
            // the code table cannot express.
            var stringDescr = StringDescrOrNull(dtype);
            if (stringDescr != null)
                return stringDescr;

            var (typeCode, needsByteSwap) = ParseDtypeString(dtype);
            var descr = DType.From(typeCode);
            // A big-endian code on a little-endian host: the bytes are read big-endian (a one-byte type keeps '|').
            return needsByteSwap ? descr.newbyteorder('>') : descr;
        }

        /// <summary>
        ///     The string descriptor a dtype STRING names (<c>'S3'</c>, <c>'&lt;U2'</c>, <c>'c'</c>, <c>'a5'</c>,
        ///     <c>'str'</c>), or null when it names another dtype or nothing <see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/> parses. The
        ///     string-spelled overloads ask this before their own parser, which knows only the fixed-size codes and once
        ///     read <c>'c'</c> and <c>'S1'</c> as NumSharp's 2-byte <c>Char</c> (NumPy: one-byte strings).
        /// </summary>
        /// <param name="dtype">The dtype string.</param>
        /// <returns>The string descriptor, or null.</returns>
        /// <exception cref="ValueError"><c>'T'</c> (StringDType), whose items hold references: NumPy's <c>cannot create an
        ///     OBJECT array from memory buffer</c>.</exception>
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

        #endregion

        #region Helpers

        /// <summary>
        ///     The code table of the string-spelled overloads: an optional byte-order prefix (<c>&gt;</c> / <c>!</c> big,
        ///     <c>&lt;</c> little, <c>=</c> / <c>|</c> native), then a sized code (<c>i4</c>), a one-letter code (<c>i</c>) or,
        ///     through <see cref="ParseDtypeName"/>, a name (<c>"float64"</c>).
        /// </summary>
        /// <param name="dtype">The dtype string; null or empty is float64.</param>
        /// <returns>The storage lane and whether the bytes are big-endian on this little-endian host.</returns>
        /// <exception cref="NotSupportedException">A complex64 code (NumSharp ships complex128 only) or a name nothing parses.</exception>
        private static (NPTypeCode typeCode, bool needsByteSwap) ParseDtypeString(string dtype)
        {
            if (string.IsNullOrEmpty(dtype))
                return (NPTypeCode.Double, false);

            // Check for byte order prefix
            bool isBigEndian = false;
            bool isLittleEndian = false;
            int startIndex = 0;

            if (dtype[0] == '>' || dtype[0] == '!')
            {
                isBigEndian = true;
                startIndex = 1;
            }
            else if (dtype[0] == '<')
            {
                isLittleEndian = true;
                startIndex = 1;
            }
            else if (dtype[0] == '=' || dtype[0] == '|')
            {
                // Native or not applicable
                startIndex = 1;
            }

            // System is little-endian on most platforms
            bool needsByteSwap = isBigEndian && BitConverter.IsLittleEndian;

            string typeStr = dtype.Substring(startIndex);

            // Parse type character and size. NumPy reference:
            //   https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing
            NPTypeCode typeCode = typeStr switch
            {
                "b1" or "?"        => NPTypeCode.Boolean,
                "u1" or "B"        => NPTypeCode.Byte,
                "i1" or "b"        => NPTypeCode.SByte,   // signed 8-bit integer (int8)
                "i2" or "h"        => NPTypeCode.Int16,
                "u2" or "H"        => NPTypeCode.UInt16,
                "i4" or "i" or "l" => NPTypeCode.Int32,
                "u4" or "I" or "L" => NPTypeCode.UInt32,
                "i8" or "q"        => NPTypeCode.Int64,
                "u8" or "Q"        => NPTypeCode.UInt64,
                "f2" or "e"        => NPTypeCode.Half,    // half-precision float (float16)
                "f4" or "f"        => NPTypeCode.Single,
                "f8" or "d"        => NPTypeCode.Double,
                // NumSharp only ships complex128. 'c8'/'F'/'complex64' (single-precision complex)
                // are REJECTED — consistent with np.dtype() — rather than silently widening to
                // complex128. Silent widening misreads genuine complex64 bytes (8B/elem) as
                // complex128 (16B/elem), halving the element count and corrupting the values.
                "c8"  or "F" or "complex64"
                                   => throw new NotSupportedException($"NumPy dtype '{typeStr}' is not supported by NumSharp"),
                "c16" or "D"       => NPTypeCode.Complex, // complex128
                // 'c' and 'S1' are NumPy's ONE-byte strings (np.dtype('c') is |S1); the string overloads route every string
                // dtype to the descriptor route before this parser, which used to read them as the 2-byte Char.
                _ => ParseDtypeName(typeStr, dtype)
            };

            return (typeCode, needsByteSwap);
        }

        /// <summary>
        ///     The NAME spellings — <c>"float64"</c>, <c>"int32"</c>, <c>"bool"</c>, <c>"complex128"</c>, <c>"float16"</c>
        ///     (NumPy's most common way to write <c>np.frombuffer(b, dtype="float64")</c>) — parse through the same
        ///     parser <see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/> uses, so they behave exactly like their sized codes
        ///     (<c>"&lt;f8"</c>). The sized/char codes above keep their own table because <see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/>
        ///     rejects <c>complex64</c> with its own message and the frombuffer parser must reject it identically;
        ///     anything neither table knows keeps the verbatim "dtype string … is not supported".
        /// </summary>
        /// <param name="typeStr">The name without its byte-order prefix.</param>
        /// <param name="dtype">The whole dtype string, for the message.</param>
        /// <returns>The name's storage lane.</returns>
        /// <exception cref="NotSupportedException">A name np.dtype does not parse.</exception>
        private static NPTypeCode ParseDtypeName(string typeStr, string dtype)
        {
            try
            {
                return np.dtype(typeStr).typecode;
            }
            catch (Exception e) when (e is NotSupportedException || e is ArgumentException)
            {
                throw new NotSupportedException($"dtype string '{dtype}' is not supported");
            }
        }

        #endregion
    }
}
