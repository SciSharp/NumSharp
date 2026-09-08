using System;
using System.Buffers;
using System.Threading;

namespace NumSharp.Interop.OnnxRuntime
{
    /// <summary>
    ///     A <see cref="MemoryManager{T}"/> over NumSharp's unmanaged buffer — what makes a
    ///     <see cref="Memory{T}"/> (and therefore ORT's <c>DenseTensor&lt;T&gt;</c> /
    ///     <c>OrtValue.CreateTensorValueFromMemory</c>) possible without a managed <c>T[]</c> and without
    ///     copying. There is no <see cref="Memory{T}"/> over a raw pointer in the BCL; this is the shim.
    ///
    ///     <para>It owns NO memory and holds NO reference: the owning <see cref="OrtTensor{T}"/> handle
    ///     keeps the ARC pin on the NumSharp buffer and releases it. Disposing the manager only makes it
    ///     refuse further access, so a <see cref="Memory{T}"/> that outlives its handle fails loudly on
    ///     the next <see cref="GetSpan"/> instead of reading freed memory.</para>
    /// </summary>
    internal sealed unsafe class UnmanagedMemoryManager<T> : MemoryManager<T> where T : unmanaged
    {
        private readonly T* _pointer;
        private readonly int _length;
        private int _disposed;

        internal UnmanagedMemoryManager(T* pointer, int length)
        {
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length));
            _pointer = pointer;
            _length = length;
        }

        /// <summary>The element pointer this manager fronts (stable — unmanaged memory never moves).</summary>
        internal T* Pointer => _pointer;

        /// <inheritdoc />
        public override Span<T> GetSpan()
        {
            if (Volatile.Read(ref _disposed) != 0)
                throw new ObjectDisposedException(nameof(UnmanagedMemoryManager<T>), "the NumSharp buffer behind this Memory<T> has been released (its OrtTensor handle was disposed).");
            return new Span<T>(_pointer, _length);
        }

        /// <inheritdoc />
        /// <remarks>Unmanaged memory needs no GC pin, so the handle carries the address and this manager as its <see cref="IPinnable"/>.</remarks>
        public override MemoryHandle Pin(int elementIndex = 0)
        {
            if (Volatile.Read(ref _disposed) != 0)
                throw new ObjectDisposedException(nameof(UnmanagedMemoryManager<T>));
            if ((uint)elementIndex > (uint)_length)
                throw new ArgumentOutOfRangeException(nameof(elementIndex));
            return new MemoryHandle(_pointer + elementIndex, default, this);
        }

        /// <inheritdoc />
        public override void Unpin()
        {
        }

        /// <inheritdoc />
        protected override void Dispose(bool disposing)
        {
            Interlocked.Exchange(ref _disposed, 1);
        }
    }
}
