using System;
using System.Collections;

namespace NumSharp.Backends.Unmanaged {
    public interface IUnmanagedMemoryBlock : IEnumerable, IMemoryBlock, ICloneable {
        void Reallocate(long length, bool copyOldValues = false);
        void Free();

        /// <summary>
        ///     Atomically increment the reference count. Used when a new
        ///     logical owner (e.g. an <c>NDArray</c>) begins sharing this
        ///     block. Returns <c>false</c> if the block has already been
        ///     released and must not be referenced further.
        /// </summary>
        bool TryAddRef();

        /// <summary>
        ///     Atomically decrement the reference count. When the count
        ///     reaches zero the underlying buffer is released immediately,
        ///     synchronously on the calling thread.
        /// </summary>
        void Release();

        /// <summary>
        ///     Diagnostic: <c>true</c> once the buffer has been released.
        /// </summary>
        bool IsReleased { get; }
    }
}
