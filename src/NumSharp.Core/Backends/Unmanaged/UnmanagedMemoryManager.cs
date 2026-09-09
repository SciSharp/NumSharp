using System;
using System.Buffers;

namespace NumSharp.Backends.Unmanaged
{
    /// <summary>
    ///     Presents a block of UNMANAGED memory as a <see cref="System.Memory{T}"/> /
    ///     <see cref="System.ReadOnlyMemory{T}"/> WITHOUT taking ownership or a reference on the
    ///     memory holder — the backing of <see cref="NDArray._Unsafe.Memory{T}()"/> /
    ///     <see cref="NDArray._Unsafe.ReadOnlyMemory{T}()"/>.
    ///
    ///     <para>
    ///     It is deliberately <b>unsafe</b>: it stores only a raw pointer, so it does NOT root the
    ///     NDArray's ARC-counted buffer. A live <see cref="Memory{T}"/> keeps THIS manager object
    ///     alive (a managed reference the <c>Memory</c> struct holds), but nothing here keeps the
    ///     NDArray's unmanaged buffer alive. If the source <see cref="NDArray"/> is disposed or
    ///     garbage-collected while a <see cref="Memory{T}"/> handed out here is still in use, the
    ///     pointer dangles and every read/write corrupts memory. Keep the NDArray alive (side by
    ///     side) for the entire lifetime of the <see cref="Memory{T}"/>.
    ///     </para>
    /// </summary>
    /// <typeparam name="T">An unmanaged element type; must match the NDArray's element type.</typeparam>
    public sealed unsafe class UnmanagedMemoryManager<T> : MemoryManager<T> where T : unmanaged
    {
        private readonly T* _pointer;
        private readonly int _length;

        /// <param name="pointer">Pointer to the first element (the NDArray's logical element 0).</param>
        /// <param name="length">Element count (must fit <see cref="int"/> — a <see cref="Memory{T}"/> limit).</param>
        public UnmanagedMemoryManager(T* pointer, int length)
        {
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length));
            _pointer = pointer;
            _length = length;
        }

        /// <inheritdoc cref="UnmanagedMemoryManager{T}(T*,int)"/>
        public UnmanagedMemoryManager(void* pointer, int length) : this((T*)pointer, length) { }

        /// <summary>The span over the unmanaged block. Aliases — no copy.</summary>
        public override Span<T> GetSpan() => new Span<T>(_pointer, _length);

        /// <summary>
        ///     Returns a handle over the raw pointer. The memory is already unmanaged and fixed, so
        ///     no <see cref="System.Runtime.InteropServices.GCHandle"/> is taken and no GC pin is
        ///     needed — this is a plain pointer wrap.
        /// </summary>
        public override MemoryHandle Pin(int elementIndex = 0)
        {
            if ((uint)elementIndex > (uint)_length)
                throw new ArgumentOutOfRangeException(nameof(elementIndex));
            return new MemoryHandle(_pointer + elementIndex);
        }

        /// <summary>No-op — nothing was pinned (the memory is unmanaged and does not move).</summary>
        public override void Unpin() { }

        /// <summary>No-op — this manager owns nothing; the NDArray owns the buffer.</summary>
        protected override void Dispose(bool disposing) { }
    }
}
