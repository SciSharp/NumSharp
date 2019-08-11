using System;
using System.Collections;

namespace NumSharp.Backends.Unmanaged {
    public interface IUnmanagedMemoryBlock : IEnumerable, IMemoryBlock, ICloneable {
        void Reallocate(long length, bool copyOldValues = false);
        void Free();
    }
}
