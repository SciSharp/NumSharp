using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Interpret a buffer as a 1-dimensional array.
        /// </summary>
        /// <param name="buffer">An object that exposes the buffer interface.</param>
        /// <param name="dtype">Data-type of the returned array. Default is float64.</param>
        /// <param name="count">Number of items to read. -1 means all data in the buffer.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>1-dimensional NDArray with data interpreted from the buffer.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html
        ///
        /// Like NumPy, this creates a VIEW of the buffer (pins the array, shares memory).
        /// Modifications to the NDArray will affect the original buffer.
        /// The buffer must stay alive while the NDArray is in use.
        /// </remarks>
        public static NDArray frombuffer(byte[] buffer, Type dtype = null, long count = -1, long offset = 0)
        {
            return frombuffer(buffer, (dtype ?? typeof(double)).GetTypeCode(), count, offset);
        }

        /// <summary>
        /// Interpret a buffer as a 1-dimensional array.
        /// </summary>
        /// <param name="buffer">An object that exposes the buffer interface.</param>
        /// <param name="dtype">Data-type of the returned array. Default is float64.</param>
        /// <param name="count">Number of items to read. -1 means all data in the buffer.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>1-dimensional NDArray with data interpreted from the buffer.</returns>
        public static NDArray frombuffer(byte[] buffer, NPTypeCode dtype, long count = -1, long offset = 0)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));

            if (dtype == NPTypeCode.Empty)
                dtype = NPTypeCode.Double;

            long bufferLength = buffer.Length;
            int itemSize = dtype.SizeOf();

            // Validate offset
            if (offset < 0 || offset > bufferLength)
                throw new ArgumentException(
                    $"offset must be non-negative and no greater than buffer length ({bufferLength})",
                    nameof(offset));

            long availableBytes = bufferLength - offset;

            // Validate alignment
            if (availableBytes % itemSize != 0)
                throw new ArgumentException(
                    "buffer size must be a multiple of element size",
                    nameof(buffer));

            long maxCount = availableBytes / itemSize;

            // Determine actual count
            long actualCount;
            if (count < 0)
            {
                actualCount = maxCount;
            }
            else
            {
                if (count > maxCount)
                    throw new ArgumentException(
                        "buffer is smaller than requested size",
                        nameof(count));
                actualCount = count;
            }

            // Handle empty result
            if (actualCount == 0)
                return new NDArray(dtype, Shape.Vector(0), false);

            // Create a VIEW of the buffer (pins the array, no copy)
            var slice = CreateArraySliceView(buffer, dtype, offset, actualCount);
            return new NDArray(new UnmanagedStorage(slice, Shape.Vector(actualCount)));
        }

        /// <summary>
        /// Create an ArraySlice that views into a byte buffer with offset support.
        /// </summary>
        private static IArraySlice CreateArraySliceView(byte[] buffer, NPTypeCode dtype, long byteOffset, long count)
        {
            switch (dtype)
            {
                case NPTypeCode.Boolean:
                    return new ArraySlice<bool>(UnmanagedMemoryBlock<bool>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.Byte:
                    return new ArraySlice<byte>(UnmanagedMemoryBlock<byte>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.SByte:
                    return new ArraySlice<sbyte>(UnmanagedMemoryBlock<sbyte>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.Int16:
                    return new ArraySlice<short>(UnmanagedMemoryBlock<short>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.UInt16:
                    return new ArraySlice<ushort>(UnmanagedMemoryBlock<ushort>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.Int32:
                    return new ArraySlice<int>(UnmanagedMemoryBlock<int>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.UInt32:
                    return new ArraySlice<uint>(UnmanagedMemoryBlock<uint>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.Int64:
                    return new ArraySlice<long>(UnmanagedMemoryBlock<long>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.UInt64:
                    return new ArraySlice<ulong>(UnmanagedMemoryBlock<ulong>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.Char:
                    return new ArraySlice<char>(UnmanagedMemoryBlock<char>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.Half:
                    return new ArraySlice<Half>(UnmanagedMemoryBlock<Half>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.Single:
                    return new ArraySlice<float>(UnmanagedMemoryBlock<float>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.Double:
                    return new ArraySlice<double>(UnmanagedMemoryBlock<double>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.Decimal:
                    return new ArraySlice<decimal>(UnmanagedMemoryBlock<decimal>.FromBuffer(buffer, byteOffset, count, copy: false));
                case NPTypeCode.Complex:
                    return new ArraySlice<System.Numerics.Complex>(UnmanagedMemoryBlock<System.Numerics.Complex>.FromBuffer(buffer, byteOffset, count, copy: false));
                default:
                    throw new NotSupportedException($"dtype {dtype} is not supported");
            }
        }

        /// <summary>
        /// Interpret a buffer as a 1-dimensional array.
        /// </summary>
        /// <param name="buffer">An object that exposes the buffer interface.</param>
        /// <param name="dtype">Data-type string (e.g., ">u4" for big-endian uint32).</param>
        /// <param name="count">Number of items to read. -1 means all data in the buffer.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>1-dimensional NDArray with data interpreted from the buffer.</returns>
        /// <remarks>
        /// Note: Big-endian dtype strings (">u4", ">i4", etc.) require a COPY to perform byte swapping.
        /// Little-endian and native endian create views without copying.
        /// </remarks>
        public static NDArray frombuffer(byte[] buffer, string dtype, long count = -1, long offset = 0)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));

            // Parse dtype string
            var (typeCode, needsByteSwap) = ParseDtypeString(dtype);
            int itemSize = typeCode.SizeOf();

            long bufferLength = buffer.Length;

            // Validate offset
            if (offset < 0 || offset > bufferLength)
                throw new ArgumentException(
                    $"offset must be non-negative and no greater than buffer length ({bufferLength})",
                    nameof(offset));

            long availableBytes = bufferLength - offset;

            // Validate alignment
            if (availableBytes % itemSize != 0)
                throw new ArgumentException(
                    "buffer size must be a multiple of element size",
                    nameof(buffer));

            long maxCount = availableBytes / itemSize;

            // Determine actual count
            long actualCount;
            if (count < 0)
            {
                actualCount = maxCount;
            }
            else
            {
                if (count > maxCount)
                    throw new ArgumentException(
                        "buffer is smaller than requested size",
                        nameof(count));
                actualCount = count;
            }

            // Handle empty result
            if (actualCount == 0)
                return new NDArray(typeCode, Shape.Vector(0), false);

            // If byte swap needed, we must copy
            if (needsByteSwap && itemSize > 1)
            {
                var slice = CreateArraySliceCopy(buffer, typeCode, offset, actualCount);
                var nd = new NDArray(new UnmanagedStorage(slice, Shape.Vector(actualCount)));
                ByteSwapInPlace(nd, typeCode, actualCount);
                return nd;
            }
            else
            {
                // Create a VIEW (no copy needed)
                var slice = CreateArraySliceView(buffer, typeCode, offset, actualCount);
                return new NDArray(new UnmanagedStorage(slice, Shape.Vector(actualCount)));
            }
        }

        /// <summary>
        /// Create an ArraySlice by copying from a byte buffer with offset support.
        /// </summary>
        private static IArraySlice CreateArraySliceCopy(byte[] buffer, NPTypeCode dtype, long byteOffset, long count)
        {
            switch (dtype)
            {
                case NPTypeCode.Boolean:
                    return new ArraySlice<bool>(UnmanagedMemoryBlock<bool>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.Byte:
                    return new ArraySlice<byte>(UnmanagedMemoryBlock<byte>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.SByte:
                    return new ArraySlice<sbyte>(UnmanagedMemoryBlock<sbyte>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.Int16:
                    return new ArraySlice<short>(UnmanagedMemoryBlock<short>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.UInt16:
                    return new ArraySlice<ushort>(UnmanagedMemoryBlock<ushort>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.Int32:
                    return new ArraySlice<int>(UnmanagedMemoryBlock<int>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.UInt32:
                    return new ArraySlice<uint>(UnmanagedMemoryBlock<uint>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.Int64:
                    return new ArraySlice<long>(UnmanagedMemoryBlock<long>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.UInt64:
                    return new ArraySlice<ulong>(UnmanagedMemoryBlock<ulong>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.Char:
                    return new ArraySlice<char>(UnmanagedMemoryBlock<char>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.Half:
                    return new ArraySlice<Half>(UnmanagedMemoryBlock<Half>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.Single:
                    return new ArraySlice<float>(UnmanagedMemoryBlock<float>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.Double:
                    return new ArraySlice<double>(UnmanagedMemoryBlock<double>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.Decimal:
                    return new ArraySlice<decimal>(UnmanagedMemoryBlock<decimal>.FromBuffer(buffer, byteOffset, count, copy: true));
                case NPTypeCode.Complex:
                    return new ArraySlice<System.Numerics.Complex>(UnmanagedMemoryBlock<System.Numerics.Complex>.FromBuffer(buffer, byteOffset, count, copy: true));
                default:
                    throw new NotSupportedException($"dtype {dtype} is not supported");
            }
        }

        /// <summary>
        /// Interpret a ReadOnlySpan as a 1-dimensional array.
        /// Note: ReadOnlySpan cannot be pinned, so this always creates a copy.
        /// </summary>
        public static NDArray frombuffer(ReadOnlySpan<byte> buffer, Type dtype = null, long count = -1, long offset = 0)
        {
            return frombuffer(buffer, (dtype ?? typeof(double)).GetTypeCode(), count, offset);
        }

        /// <summary>
        /// Interpret a ReadOnlySpan as a 1-dimensional array.
        /// Note: ReadOnlySpan cannot be pinned, so this always creates a copy.
        /// </summary>
        public static NDArray frombuffer(ReadOnlySpan<byte> buffer, NPTypeCode dtype, long count = -1, long offset = 0)
        {
            if (dtype == NPTypeCode.Empty)
                dtype = NPTypeCode.Double;

            long bufferLength = buffer.Length;
            int itemSize = dtype.SizeOf();

            // Validate offset
            if (offset < 0 || offset > bufferLength)
                throw new ArgumentException(
                    $"offset must be non-negative and no greater than buffer length ({bufferLength})",
                    nameof(offset));

            long availableBytes = bufferLength - offset;

            // Validate alignment
            if (availableBytes % itemSize != 0)
                throw new ArgumentException(
                    "buffer size must be a multiple of element size");

            long maxCount = availableBytes / itemSize;

            // Determine actual count
            long actualCount;
            if (count < 0)
            {
                actualCount = maxCount;
            }
            else
            {
                if (count > maxCount)
                    throw new ArgumentException(
                        "buffer is smaller than requested size",
                        nameof(count));
                actualCount = count;
            }

            // Handle empty result
            if (actualCount == 0)
                return new NDArray(dtype, Shape.Vector(0), false);

            // ReadOnlySpan cannot be pinned, must copy
            var nd = new NDArray(dtype, Shape.Vector(actualCount), false);
            long bytesToCopy = actualCount * itemSize;
            unsafe
            {
                fixed (byte* src = &buffer[(int)offset])
                {
                    Buffer.MemoryCopy(src, (void*)nd.Unsafe.Address, bytesToCopy, bytesToCopy);
                }
            }

            return nd;
        }

        #region .NET-friendly overloads

        /// <summary>
        /// Interpret an ArraySegment as a 1-dimensional array.
        /// Uses the segment's Offset and Count automatically.
        /// </summary>
        /// <param name="segment">The array segment to interpret.</param>
        /// <param name="dtype">Data-type of the returned array. Default is float64.</param>
        /// <param name="count">Number of items to read. -1 means all data in the segment.</param>
        /// <returns>1-dimensional NDArray viewing the segment's data.</returns>
        public static NDArray frombuffer(ArraySegment<byte> segment, Type dtype = null, long count = -1)
        {
            if (segment.Array == null)
                throw new ArgumentException("ArraySegment has no underlying array", nameof(segment));

            return frombuffer(segment.Array, dtype, count, segment.Offset);
        }

        /// <summary>
        /// Interpret an ArraySegment as a 1-dimensional array.
        /// </summary>
        public static NDArray frombuffer(ArraySegment<byte> segment, NPTypeCode dtype, long count = -1)
        {
            if (segment.Array == null)
                throw new ArgumentException("ArraySegment has no underlying array", nameof(segment));

            return frombuffer(segment.Array, dtype, count, segment.Offset);
        }

        /// <summary>
        /// Interpret a Memory&lt;byte&gt; as a 1-dimensional array.
        /// Creates a view if backed by an array, otherwise copies.
        /// </summary>
        /// <param name="memory">The memory to interpret.</param>
        /// <param name="dtype">Data-type of the returned array. Default is float64.</param>
        /// <param name="count">Number of items to read. -1 means all data.</param>
        /// <param name="offset">Byte offset within the memory. Default is 0.</param>
        /// <returns>1-dimensional NDArray.</returns>
        public static NDArray frombuffer(Memory<byte> memory, Type dtype = null, long count = -1, long offset = 0)
        {
            return frombuffer(memory, (dtype ?? typeof(double)).GetTypeCode(), count, offset);
        }

        /// <summary>
        /// Interpret a Memory&lt;byte&gt; as a 1-dimensional array.
        /// </summary>
        public static NDArray frombuffer(Memory<byte> memory, NPTypeCode dtype, long count = -1, long offset = 0)
        {
            // Try to get the underlying array for view semantics
            if (MemoryMarshal.TryGetArray<byte>(memory, out var segment))
            {
                return frombuffer(segment.Array!, dtype, count, segment.Offset + offset);
            }

            // Fallback to copy via span
            return frombuffer(memory.Span, dtype, count, offset);
        }

        /// <summary>
        /// Interpret unmanaged memory at a pointer as a 1-dimensional array.
        /// </summary>
        /// <param name="address">Pointer to the start of the buffer.</param>
        /// <param name="byteLength">Total length of the buffer in bytes.</param>
        /// <param name="dtype">Data-type of the returned array. Default is float64.</param>
        /// <param name="count">Number of items to read. -1 means all data.</param>
        /// <param name="offset">Byte offset into the buffer. Default is 0.</param>
        /// <param name="dispose">
        /// Optional cleanup action called when NDArray is disposed.
        /// Use to transfer ownership: dispose: () => Marshal.FreeHGlobal(ptr)
        /// If null, caller is responsible for memory lifetime (view semantics).
        /// </param>
        /// <returns>1-dimensional NDArray viewing/owning the memory.</returns>
        /// <example>
        /// // View only (caller manages lifetime):
        /// var arr = np.frombuffer(ptr, length, typeof(float));
        ///
        /// // Take ownership (NumSharp frees on dispose):
        /// var ptr = Marshal.AllocHGlobal(1024);
        /// var arr = np.frombuffer(ptr, 1024, typeof(float), dispose: () => Marshal.FreeHGlobal(ptr));
        /// </example>
        public static unsafe NDArray frombuffer(IntPtr address, long byteLength, Type dtype = null, long count = -1, long offset = 0, Action dispose = null)
        {
            return frombuffer(address, byteLength, (dtype ?? typeof(double)).GetTypeCode(), count, offset, dispose);
        }

        /// <summary>
        /// Interpret unmanaged memory at a pointer as a 1-dimensional array.
        /// </summary>
        public static unsafe NDArray frombuffer(IntPtr address, long byteLength, NPTypeCode dtype, long count = -1, long offset = 0, Action dispose = null)
        {
            if (address == IntPtr.Zero)
                throw new ArgumentNullException(nameof(address));

            if (dtype == NPTypeCode.Empty)
                dtype = NPTypeCode.Double;

            int itemSize = dtype.SizeOf();

            // Validate offset
            if (offset < 0 || offset > byteLength)
                throw new ArgumentException(
                    $"offset must be non-negative and no greater than buffer length ({byteLength})",
                    nameof(offset));

            long availableBytes = byteLength - offset;

            // Validate alignment
            if (availableBytes % itemSize != 0)
                throw new ArgumentException(
                    "buffer size must be a multiple of element size",
                    nameof(byteLength));

            long maxCount = availableBytes / itemSize;
            long actualCount = count < 0 ? maxCount : Math.Min(count, maxCount);

            if (actualCount == 0)
            {
                dispose?.Invoke(); // Clean up even for empty result
                return new NDArray(dtype, Shape.Vector(0), false);
            }

            // Create view/owned slice depending on dispose action
            IArraySlice slice;
            if (dispose != null)
            {
                slice = CreateArraySliceWithDispose((byte*)address + offset, dtype, actualCount, dispose);
            }
            else
            {
                slice = CreateArraySliceFromPointer((byte*)address + offset, dtype, actualCount);
            }
            return new NDArray(new UnmanagedStorage(slice, Shape.Vector(actualCount)));
        }

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
        public static unsafe NDArray frombuffer(void* address, long byteLength, Type dtype = null, long count = -1, long offset = 0, Action dispose = null)
        {
            return frombuffer((IntPtr)address, byteLength, dtype, count, offset, dispose);
        }

        /// <summary>
        /// Interpret unmanaged memory at a pointer as a 1-dimensional array.
        /// </summary>
        public static unsafe NDArray frombuffer(void* address, long byteLength, NPTypeCode dtype, long count = -1, long offset = 0, Action dispose = null)
        {
            return frombuffer((IntPtr)address, byteLength, dtype, count, offset, dispose);
        }

        /// <summary>
        /// Reinterpret a typed array as a different dtype.
        /// Like NumPy's view() but via frombuffer semantics.
        /// </summary>
        /// <typeparam name="TSource">Source element type.</typeparam>
        /// <param name="array">The source array to reinterpret.</param>
        /// <param name="dtype">Target data-type. Default preserves source type.</param>
        /// <param name="count">Number of items of target dtype. -1 for all.</param>
        /// <param name="offset">Byte offset. Default is 0.</param>
        /// <returns>1-dimensional NDArray viewing the array as the target dtype.</returns>
        /// <example>
        /// var ints = new int[] { 1, 2, 3, 4 };
        /// var asBytes = np.frombuffer(ints, typeof(byte));  // 16 bytes
        /// var asFloats = np.frombuffer(ints, typeof(float)); // 4 floats (same bits)
        /// </example>
        public static NDArray frombuffer<TSource>(TSource[] array, Type dtype = null, long count = -1, long offset = 0)
            where TSource : unmanaged
        {
            return frombuffer(array, (dtype ?? typeof(TSource)).GetTypeCode(), count, offset);
        }

        /// <summary>
        /// Reinterpret a typed array as a different dtype.
        /// </summary>
        public static unsafe NDArray frombuffer<TSource>(TSource[] array, NPTypeCode dtype, long count = -1, long offset = 0)
            where TSource : unmanaged
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            if (dtype == NPTypeCode.Empty)
                dtype = InfoOf<TSource>.NPTypeCode;

            int sourceItemSize = sizeof(TSource);
            int targetItemSize = dtype.SizeOf();
            long byteLength = array.Length * sourceItemSize;

            // Validate offset
            if (offset < 0 || offset > byteLength)
                throw new ArgumentException(
                    $"offset must be non-negative and no greater than buffer length ({byteLength})",
                    nameof(offset));

            long availableBytes = byteLength - offset;

            // Validate alignment
            if (availableBytes % targetItemSize != 0)
                throw new ArgumentException(
                    "buffer size must be a multiple of element size",
                    nameof(array));

            long maxCount = availableBytes / targetItemSize;
            long actualCount = count < 0 ? maxCount : Math.Min(count, maxCount);

            if (actualCount == 0)
                return new NDArray(dtype, Shape.Vector(0), false);

            // Pin the array and create a view
            var handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            var baseAddr = (byte*)handle.AddrOfPinnedObject();
            var slice = CreateArraySliceFromPinnedPointer(baseAddr + offset, dtype, actualCount, handle);
            return new NDArray(new UnmanagedStorage(slice, Shape.Vector(actualCount)));
        }

        /// <summary>
        /// Create ArraySlice from raw pointer with dispose action (takes ownership).
        /// </summary>
        private static unsafe IArraySlice CreateArraySliceWithDispose(byte* address, NPTypeCode dtype, long count, Action dispose)
        {
            switch (dtype)
            {
                case NPTypeCode.Boolean:
                    return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>((bool*)address, count, dispose));
                case NPTypeCode.Byte:
                    return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(address, count, dispose));
                case NPTypeCode.SByte:
                    return new ArraySlice<sbyte>(new UnmanagedMemoryBlock<sbyte>((sbyte*)address, count, dispose));
                case NPTypeCode.Int16:
                    return new ArraySlice<short>(new UnmanagedMemoryBlock<short>((short*)address, count, dispose));
                case NPTypeCode.UInt16:
                    return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>((ushort*)address, count, dispose));
                case NPTypeCode.Int32:
                    return new ArraySlice<int>(new UnmanagedMemoryBlock<int>((int*)address, count, dispose));
                case NPTypeCode.UInt32:
                    return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>((uint*)address, count, dispose));
                case NPTypeCode.Int64:
                    return new ArraySlice<long>(new UnmanagedMemoryBlock<long>((long*)address, count, dispose));
                case NPTypeCode.UInt64:
                    return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>((ulong*)address, count, dispose));
                case NPTypeCode.Char:
                    return new ArraySlice<char>(new UnmanagedMemoryBlock<char>((char*)address, count, dispose));
                case NPTypeCode.Half:
                    return new ArraySlice<Half>(new UnmanagedMemoryBlock<Half>((Half*)address, count, dispose));
                case NPTypeCode.Single:
                    return new ArraySlice<float>(new UnmanagedMemoryBlock<float>((float*)address, count, dispose));
                case NPTypeCode.Double:
                    return new ArraySlice<double>(new UnmanagedMemoryBlock<double>((double*)address, count, dispose));
                case NPTypeCode.Decimal:
                    return new ArraySlice<decimal>(new UnmanagedMemoryBlock<decimal>((decimal*)address, count, dispose));
                case NPTypeCode.Complex:
                    return new ArraySlice<System.Numerics.Complex>(new UnmanagedMemoryBlock<System.Numerics.Complex>((System.Numerics.Complex*)address, count, dispose));
                default:
                    throw new NotSupportedException($"dtype {dtype} is not supported");
            }
        }

        /// <summary>
        /// Create ArraySlice from raw pointer (no ownership - caller manages lifetime).
        /// </summary>
        private static unsafe IArraySlice CreateArraySliceFromPointer(byte* address, NPTypeCode dtype, long count)
        {
            switch (dtype)
            {
                case NPTypeCode.Boolean:
                    return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>((bool*)address, count));
                case NPTypeCode.Byte:
                    return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(address, count));
                case NPTypeCode.SByte:
                    return new ArraySlice<sbyte>(new UnmanagedMemoryBlock<sbyte>((sbyte*)address, count));
                case NPTypeCode.Int16:
                    return new ArraySlice<short>(new UnmanagedMemoryBlock<short>((short*)address, count));
                case NPTypeCode.UInt16:
                    return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>((ushort*)address, count));
                case NPTypeCode.Int32:
                    return new ArraySlice<int>(new UnmanagedMemoryBlock<int>((int*)address, count));
                case NPTypeCode.UInt32:
                    return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>((uint*)address, count));
                case NPTypeCode.Int64:
                    return new ArraySlice<long>(new UnmanagedMemoryBlock<long>((long*)address, count));
                case NPTypeCode.UInt64:
                    return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>((ulong*)address, count));
                case NPTypeCode.Char:
                    return new ArraySlice<char>(new UnmanagedMemoryBlock<char>((char*)address, count));
                case NPTypeCode.Half:
                    return new ArraySlice<Half>(new UnmanagedMemoryBlock<Half>((Half*)address, count));
                case NPTypeCode.Single:
                    return new ArraySlice<float>(new UnmanagedMemoryBlock<float>((float*)address, count));
                case NPTypeCode.Double:
                    return new ArraySlice<double>(new UnmanagedMemoryBlock<double>((double*)address, count));
                case NPTypeCode.Decimal:
                    return new ArraySlice<decimal>(new UnmanagedMemoryBlock<decimal>((decimal*)address, count));
                case NPTypeCode.Complex:
                    return new ArraySlice<System.Numerics.Complex>(new UnmanagedMemoryBlock<System.Numerics.Complex>((System.Numerics.Complex*)address, count));
                default:
                    throw new NotSupportedException($"dtype {dtype} is not supported");
            }
        }

        /// <summary>
        /// Create ArraySlice from pinned pointer with GCHandle disposal.
        /// </summary>
        private static unsafe IArraySlice CreateArraySliceFromPinnedPointer(byte* address, NPTypeCode dtype, long count, GCHandle handle)
        {
            Action dispose = () => handle.Free();
            switch (dtype)
            {
                case NPTypeCode.Boolean:
                    return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>((bool*)address, count, dispose));
                case NPTypeCode.Byte:
                    return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(address, count, dispose));
                case NPTypeCode.SByte:
                    return new ArraySlice<sbyte>(new UnmanagedMemoryBlock<sbyte>((sbyte*)address, count, dispose));
                case NPTypeCode.Int16:
                    return new ArraySlice<short>(new UnmanagedMemoryBlock<short>((short*)address, count, dispose));
                case NPTypeCode.UInt16:
                    return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>((ushort*)address, count, dispose));
                case NPTypeCode.Int32:
                    return new ArraySlice<int>(new UnmanagedMemoryBlock<int>((int*)address, count, dispose));
                case NPTypeCode.UInt32:
                    return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>((uint*)address, count, dispose));
                case NPTypeCode.Int64:
                    return new ArraySlice<long>(new UnmanagedMemoryBlock<long>((long*)address, count, dispose));
                case NPTypeCode.UInt64:
                    return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>((ulong*)address, count, dispose));
                case NPTypeCode.Char:
                    return new ArraySlice<char>(new UnmanagedMemoryBlock<char>((char*)address, count, dispose));
                case NPTypeCode.Half:
                    return new ArraySlice<Half>(new UnmanagedMemoryBlock<Half>((Half*)address, count, dispose));
                case NPTypeCode.Single:
                    return new ArraySlice<float>(new UnmanagedMemoryBlock<float>((float*)address, count, dispose));
                case NPTypeCode.Double:
                    return new ArraySlice<double>(new UnmanagedMemoryBlock<double>((double*)address, count, dispose));
                case NPTypeCode.Decimal:
                    return new ArraySlice<decimal>(new UnmanagedMemoryBlock<decimal>((decimal*)address, count, dispose));
                case NPTypeCode.Complex:
                    return new ArraySlice<System.Numerics.Complex>(new UnmanagedMemoryBlock<System.Numerics.Complex>((System.Numerics.Complex*)address, count, dispose));
                default:
                    throw new NotSupportedException($"dtype {dtype} is not supported");
            }
        }

        #endregion

        #region Helpers

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

            // Parse type character and size
            NPTypeCode typeCode = typeStr switch
            {
                "b1" or "?" => NPTypeCode.Boolean,
                "u1" or "B" => NPTypeCode.Byte,
                "i1" or "b" => NPTypeCode.Byte, // signed byte maps to byte
                "i2" or "h" => NPTypeCode.Int16,
                "u2" or "H" => NPTypeCode.UInt16,
                "i4" or "i" or "l" => NPTypeCode.Int32,
                "u4" or "I" or "L" => NPTypeCode.UInt32,
                "i8" or "q" => NPTypeCode.Int64,
                "u8" or "Q" => NPTypeCode.UInt64,
                "f4" or "f" => NPTypeCode.Single,
                "f8" or "d" => NPTypeCode.Double,
                "c" or "S1" => NPTypeCode.Char,
                _ => throw new NotSupportedException($"dtype string '{dtype}' is not supported")
            };

            return (typeCode, needsByteSwap);
        }

        private static unsafe void ByteSwapInPlace(NDArray nd, NPTypeCode typeCode, long count)
        {
            switch (typeCode)
            {
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                {
                    var ptr = (ushort*)nd.Unsafe.Address;
                    for (long i = 0; i < count; i++)
                        ptr[i] = BinaryPrimitives_ReverseEndianness(ptr[i]);
                    break;
                }
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Single:
                {
                    var ptr = (uint*)nd.Unsafe.Address;
                    for (long i = 0; i < count; i++)
                        ptr[i] = BinaryPrimitives_ReverseEndianness(ptr[i]);
                    break;
                }
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Double:
                {
                    var ptr = (ulong*)nd.Unsafe.Address;
                    for (long i = 0; i < count; i++)
                        ptr[i] = BinaryPrimitives_ReverseEndianness(ptr[i]);
                    break;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ushort BinaryPrimitives_ReverseEndianness(ushort value)
        {
            return (ushort)((value >> 8) | (value << 8));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint BinaryPrimitives_ReverseEndianness(uint value)
        {
            return (value >> 24) |
                   ((value >> 8) & 0x0000FF00) |
                   ((value << 8) & 0x00FF0000) |
                   (value << 24);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong BinaryPrimitives_ReverseEndianness(ulong value)
        {
            return ((ulong)BinaryPrimitives_ReverseEndianness((uint)value) << 32) |
                   BinaryPrimitives_ReverseEndianness((uint)(value >> 32));
        }

        #endregion
    }
}
