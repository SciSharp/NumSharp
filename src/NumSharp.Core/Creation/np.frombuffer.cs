using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends;
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
        /// Unlike NumPy, this creates a copy of the data (NumSharp uses unmanaged memory).
        /// NumPy creates a view that shares memory with the buffer.
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

            // Create output array
            var nd = new NDArray(dtype, Shape.Vector(actualCount), false);

            // Copy data efficiently using unsafe block copy
            long bytesToCopy = actualCount * itemSize;
            unsafe
            {
                fixed (byte* src = &buffer[offset])
                {
                    Buffer.MemoryCopy(src, (void*)nd.Unsafe.Address, bytesToCopy, bytesToCopy);
                }
            }

            return nd;
        }

        /// <summary>
        /// Interpret a buffer as a 1-dimensional array.
        /// </summary>
        /// <param name="buffer">An object that exposes the buffer interface.</param>
        /// <param name="dtype">Data-type string (e.g., ">u4" for big-endian uint32).</param>
        /// <param name="count">Number of items to read. -1 means all data in the buffer.</param>
        /// <param name="offset">Start reading the buffer from this offset (in bytes). Default is 0.</param>
        /// <returns>1-dimensional NDArray with data interpreted from the buffer.</returns>
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

            // Create output array
            var nd = new NDArray(typeCode, Shape.Vector(actualCount), false);

            // Copy data
            long bytesToCopy = actualCount * itemSize;
            unsafe
            {
                fixed (byte* src = &buffer[offset])
                {
                    Buffer.MemoryCopy(src, (void*)nd.Unsafe.Address, bytesToCopy, bytesToCopy);
                }
            }

            // Byte swap if needed (big-endian to little-endian)
            if (needsByteSwap && itemSize > 1)
            {
                ByteSwapInPlace(nd, typeCode, actualCount);
            }

            return nd;
        }

        /// <summary>
        /// Interpret a ReadOnlySpan as a 1-dimensional array.
        /// </summary>
        public static NDArray frombuffer(ReadOnlySpan<byte> buffer, Type dtype = null, long count = -1, long offset = 0)
        {
            return frombuffer(buffer, (dtype ?? typeof(double)).GetTypeCode(), count, offset);
        }

        /// <summary>
        /// Interpret a ReadOnlySpan as a 1-dimensional array.
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

            // Create output array
            var nd = new NDArray(dtype, Shape.Vector(actualCount), false);

            // Copy data
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
