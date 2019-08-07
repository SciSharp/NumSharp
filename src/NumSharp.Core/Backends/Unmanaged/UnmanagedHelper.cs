using System;
using NumSharp.Utilities;

namespace NumSharp.Backends.Unmanaged
{
    public static class UnmanagedHelper
    {
        /// <summary>
        ///     Copies the entire contents of this storage to given address (using <see cref="Count"/>).
        /// </summary>
        /// <param name="dst">The block to copy to.</param>
        public static unsafe void CopyTo(this IMemoryBlock src, IMemoryBlock dst)
        {
            if (dst.TypeCode != src.TypeCode)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (src.Count > dst.Count)
                throw new ArgumentOutOfRangeException(nameof(dst), $"Unable to copy from this storage to given array because this storage count is larger than the given array length.");

            var bytesCount = src.BytesLength;
            Buffer.MemoryCopy(src.Address, dst.Address, bytesCount, bytesCount);
        }

        /// <summary>
        ///     Copies the entire contents of this storage to given address (using <see cref="Count"/>).
        /// </summary>
        /// <param name="dst">The block to copy to.</param>
        public static unsafe void CopyTo(this IMemoryBlock src, IMemoryBlock dst, int countOffsetDesitinion)
        {
            if (dst.TypeCode != src.TypeCode)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (src.Count > dst.Count)
                throw new ArgumentOutOfRangeException(nameof(dst), $"Unable to copy from this storage to given array because this storage count is larger than the given array length.");

            var bytesCount = src.BytesLength;
            Buffer.MemoryCopy(src.Address, (byte*)dst.Address + countOffsetDesitinion * dst.ItemLength, bytesCount, bytesCount);
        }

        /// <summary>
        ///     Copies the entire contents of this storage to given address (using <see cref="Count"/>).
        /// </summary>
        /// <param name="dst">The block to copy to.</param>
        public static unsafe void CopyTo(this IMemoryBlock src, void* dstAddress, int countOffsetDesitinion)
        {
            if (dstAddress == null)
                throw new ArgumentNullException(nameof(dstAddress));

            var bytesCount = src.BytesLength;
            Buffer.MemoryCopy(src.Address, (byte*)dstAddress + countOffsetDesitinion * src.ItemLength, bytesCount, bytesCount);
        }

        /// <summary>
        ///     Copies the entire contents of this storage to given address.
        /// </summary>
        /// <param name="src">The source of the copying</param>
        /// <param name="dstAddress">The address to copy to.</param>
        public static unsafe void CopyTo(this IMemoryBlock src, void* dstAddress)
        {
            if (dstAddress == null)
                throw new ArgumentNullException(nameof(dstAddress));

            var bytesCount = src.BytesLength;
            Buffer.MemoryCopy(src.Address, dstAddress, bytesCount, bytesCount);
        }
    }
}
