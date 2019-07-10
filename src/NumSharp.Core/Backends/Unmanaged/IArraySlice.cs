using System;
using System.Collections;

namespace NumSharp.Backends.Unmanaged
{
    public interface IArraySlice : ICloneable, IEnumerable
    {
        Type ArrayType { get; }

        /// <summary>
        ///     The size of a single item.
        /// </summary>
        int ItemSize { get; }

        unsafe void* Address { get; }
        int Count { get; }
        IUnmanagedArray MemoryBlock { get; }

        T GetIndex<T>(int index) where T : unmanaged;
        object GetIndex(int index);

        void SetIndex<T>(int index, T value) where T : unmanaged;
        void SetIndex(int index, object value);
    }
}
