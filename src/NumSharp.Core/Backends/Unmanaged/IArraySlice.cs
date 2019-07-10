using System;

namespace NumSharp.Backends.Unmanaged {
    public interface IArraySlice : ICloneable
    {
        Type ArrayType { get; }
        unsafe void* Address { get; }
        int Count { get; }
        IUnmanagedArray MemoryBlock { get; }

        T GetIndex<T>(int index) where T : unmanaged;
        void SetIndex<T>(int index, T value) where T : unmanaged;
    }
}