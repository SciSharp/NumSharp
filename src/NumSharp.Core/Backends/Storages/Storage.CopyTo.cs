using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public abstract partial class Storage
    {
        /// <summary>
        ///     Copies the entire contents of this storage to given address.
        /// </summary>
        public unsafe bool CopyTo(IntPtr ptr)
            => CopyTo(ptr.ToPointer());

        public unsafe bool CopyTo<T>(T[] array) where T : unmanaged
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            if (typeof(T) != _dtype)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (Count > array.Length)
                throw new ArgumentOutOfRangeException(nameof(array), $"Unable to copy from this storage to given array because this storage count is larger than the given array length.");

            fixed (T* dst = array)
            {
                var bytesCount = Count * InfoOf<T>.Size;
                Buffer.MemoryCopy(Address, dst, bytesCount, bytesCount);
            }
            return true;
        }

        public unsafe bool CopyTo<T>(T* address) where T : unmanaged
        {
            if (address == (T*)0)
                throw new ArgumentNullException(nameof(address));

            if (typeof(T) != _dtype)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (!Shape.IsContiguous || Shape.ModifiedStrides)
            {
                var dst = ArraySlice.Wrap<T>(address, Count);
                MultiIterator.Assign(new UnmanagedStorage(dst, Shape.Clean()), this);
                return true;
            }

            var bytesCount = Count * InfoOf<T>.Size;
            Buffer.MemoryCopy(Address, address, bytesCount, bytesCount);
            return true;
        }

        public unsafe bool CopyTo(void* address)
            => _typecode switch
            {
                NPTypeCode.Boolean => CopyTo((bool*)address),
                NPTypeCode.Byte => CopyTo((byte*)address),
                NPTypeCode.Int32 => CopyTo((int*)address),
                NPTypeCode.Int64 => CopyTo((long*)address),
                NPTypeCode.Single => CopyTo((float*)address),
                NPTypeCode.Double => CopyTo((double*)address),
                _ => throw new NotSupportedException()
            };

        /// <summary>
        ///     Copies the entire contents of this storage to given address (using <see cref="Count"/>).
        /// </summary>
        /// <param name="block">The block to copy to.</param>
        public unsafe bool CopyTo(IMemoryBlock block)
        {
            if (block.TypeCode != _typecode)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (Count > block.Count)
                throw new ArgumentOutOfRangeException(nameof(block), $"Unable to copy from this storage to given memory block because this storage count is larger than the given memory block's length.");

            return CopyTo(block.Address);
        }

        /// <summary>
        ///     Copies the entire contents of this storage to given address (using <see cref="Count"/>).
        /// </summary>
        /// <param name="block">The block to copy to.</param>
        public unsafe bool CopyTo<T>(IMemoryBlock<T> block) where T : unmanaged
        {
            if (block.TypeCode != _typecode)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (Count > block.Count)
                throw new ArgumentOutOfRangeException(nameof(block), $"Unable to copy from this storage to given array because this storage count is larger than the given array length.");

            return CopyTo(block.Address);
        }
    }
}
