using System;
using System.Collections;

namespace NumSharp.Backends.Unmanaged {
    public interface IUnmanagedMemoryBlock : IEnumerable, IMemoryBlock, ICloneable {
        void Reallocate(int length, bool copyOldValues = false);
        void Free();
    }
}
