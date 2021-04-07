using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public partial class NDArray
    {

        /// <summary>
        ///     Copies the entire contents of this storage to given address (using <see cref="Count"/>).
        /// </summary>
        /// <param name="slice">The slice to copy to.</param>
        public void CopyTo(IMemoryBlock slice) => Storage.CopyTo(slice);

        /// <summary>
        ///     Copies the entire contents of this storage to given address (using <see cref="Count"/>).
        /// </summary>
        /// <param name="address">The address to copy to.</param>
        public unsafe void CopyTo(void* address) => Storage.CopyTo(address);

        /// <summary>
        ///     Copies the entire contents of this storage to given array.
        /// </summary>
        /// <param name="array">The array to copy to.</param>
        public void CopyTo<T>(T[] array) where T : unmanaged => Storage.CopyTo<T>(array);

        /// <summary>
        ///     Copies the entire contents of this storage to given address.
        /// </summary>
        public void CopyTo(IntPtr ptr) => Storage.CopyTo(ptr);

        /// <summary>
        ///     Copies the entire contents of this storage to given address (using <see cref="Count"/>).
        /// </summary>
        /// <param name="block">The slice to copy to.</param>
        public unsafe void CopyTo<T>(IMemoryBlock<T> block) where T : unmanaged => Storage.CopyTo<T>(block);

        /// <summary>
        ///     Copies the entire contents of this storage to given address.
        /// </summary>
        /// <param name="address">The address to copy to.</param>
        public unsafe void CopyTo<T>(T* address) where T : unmanaged => Storage.CopyTo<T>(address);
    }
}
