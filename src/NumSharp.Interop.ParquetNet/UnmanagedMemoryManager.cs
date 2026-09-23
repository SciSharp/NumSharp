using System;
using System.Buffers;

namespace NumSharp.Interop.ParquetNet
{
    /// <summary>
    /// Presents a window of already-allocated UNMANAGED memory as a <see cref="Memory{T}"/>.
    ///
    /// <para>This is the bridge that lets Parquet.Net decode a column straight into an NDArray's own raw
    /// buffer. Parquet.Net's read API is buffer-based — <c>ParquetRowGroupReader.ReadAsync&lt;T&gt;(field,
    /// Memory&lt;T&gt;)</c> writes the decoded values into a caller-supplied <see cref="Memory{T}"/>. We create
    /// that <see cref="Memory{T}"/> here over an NDArray's <c>UnmanagedMemoryBlock&lt;T&gt;</c>, so the decoder
    /// lands its output directly in NDArray memory with no intermediate managed array and no extra copy.</para>
    ///
    /// <para>Ownership: this manager does NOT own the memory. It never allocates and never frees — the NDArray's
    /// <c>UnmanagedMemoryBlock&lt;T&gt;</c> owns the lifetime (reference-counted). The manager is only a typed
    /// window and is safe to use across an <c>await</c> because it is a heap object and the memory it points at
    /// (unmanaged) never moves.</para>
    /// </summary>
    /// <typeparam name="T">An unmanaged element type (matches the NDArray's dtype).</typeparam>
    internal sealed unsafe class UnmanagedMemoryManager<T> : MemoryManager<T> where T : unmanaged
    {
        private readonly T* _pointer;
        private readonly int _length;

        /// <param name="pointer">Address of the first element of the window.</param>
        /// <param name="length">Number of <typeparamref name="T"/> elements in the window.</param>
        public UnmanagedMemoryManager(T* pointer, int length)
        {
            _pointer = pointer;
            _length = length;
        }

        /// <summary>The span over the unmanaged window. Parquet.Net writes decoded values here.</summary>
        public override Span<T> GetSpan() => new Span<T>(_pointer, _length);

        /// <summary>Unmanaged memory never moves, so "pinning" simply returns the fixed address.</summary>
        public override MemoryHandle Pin(int elementIndex = 0) => new MemoryHandle(_pointer + elementIndex);

        /// <summary>No-op — the memory was never pinned by a GC handle.</summary>
        public override void Unpin() { }

        /// <summary>No-op — the underlying block is owned by the NDArray, not by this manager.</summary>
        protected override void Dispose(bool disposing) { }
    }
}
