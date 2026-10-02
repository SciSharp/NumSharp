using System;
using System.Buffers;
using System.Threading;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.OnnxRuntime
{
    /// <summary>
    ///     A <see cref="MemoryManager{T}"/> over NumSharp's unmanaged buffer — what makes a
    ///     <see cref="Memory{T}"/> (and therefore ORT's <c>DenseTensor&lt;T&gt;</c> /
    ///     <c>OrtValue.CreateTensorValueFromMemory</c>) possible without a managed <c>T[]</c> and without
    ///     copying. There is no <see cref="Memory{T}"/> over a raw pointer in the BCL; this is the shim.
    ///
    ///     <para><b>Lifetime.</b> The owning <see cref="OrtTensor{T}"/> handle keeps one ARC reference on the
    ///     NumSharp buffer for its own lifetime. Every <see cref="Pin"/> takes ANOTHER reference, released by the
    ///     matching <see cref="Unpin"/>, so whoever pins the memory — ORT's <c>CreateTensorValueFromMemory</c>,
    ///     a NumSharp view over the <c>DenseTensor</c> (<c>AsNDArray</c>), any <see cref="Memory{T}.Pin"/> —
    ///     keeps the buffer alive for exactly as long as its pin, even past the handle's <c>Dispose</c> and the
    ///     source array's. Without that reference those consumers read freed memory once the handle and the
    ///     source were gone (a probe read another array's values through all three). Disposing the manager (the
    ///     handle does it) only makes it refuse NEW access: <see cref="GetSpan"/> and <see cref="Pin"/> throw,
    ///     pins taken before stay valid.</para>
    /// </summary>
    /// <typeparam name="T">The element type the memory is viewed as (the dtype's ORT type: Float16 for Half, ushort for Char).</typeparam>
    internal sealed unsafe class UnmanagedMemoryManager<T> : MemoryManager<T> where T : unmanaged
    {
        private readonly T* _pointer;
        private readonly int _length;
        private readonly IArraySlice _owner;
        private int _disposed;

        /// <summary>Fronts <paramref name="length"/> elements at <paramref name="pointer"/>, which lie in <paramref name="owner"/>'s buffer.</summary>
        /// <param name="pointer">The first element (the array's first logical element, not necessarily the buffer's start).</param>
        /// <param name="length">The element count (a <see cref="Memory{T}"/> is <c>int</c>-indexed).</param>
        /// <param name="owner">
        ///     The NumSharp buffer the elements belong to; each pin takes a reference on it. The caller (the export handle)
        ///     holds its own reference for as long as this manager hands out new spans or pins.
        /// </param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="owner"/> is null.</exception>
        internal UnmanagedMemoryManager(T* pointer, int length, IArraySlice owner)
        {
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length));
            _pointer = pointer;
            _length = length;
            _owner = owner ?? throw new ArgumentNullException(nameof(owner));
        }

        /// <summary>The element pointer this manager fronts (stable — unmanaged memory never moves).</summary>
        internal T* Pointer => _pointer;

        /// <inheritdoc />
        /// <exception cref="ObjectDisposedException">The owning handle has been disposed.</exception>
        public override Span<T> GetSpan()
        {
            if (Volatile.Read(ref _disposed) != 0)
                throw new ObjectDisposedException(nameof(UnmanagedMemoryManager<T>), "the NumSharp buffer behind this Memory<T> has been released (its OrtTensor handle was disposed).");
            return new Span<T>(_pointer, _length);
        }

        /// <summary>
        ///     Pins the memory: unmanaged memory needs no GC pin, so the handle carries the address and this manager as its
        ///     <see cref="IPinnable"/> — plus a reference on the NumSharp buffer that the pin's own <c>Dispose</c> releases
        ///     through <see cref="Unpin"/>, so the buffer stays alive for exactly as long as the pin does.
        /// </summary>
        /// <param name="elementIndex">The element the returned pointer addresses.</param>
        /// <returns>A handle whose <see cref="MemoryHandle.Dispose"/> releases the reference this pin took.</returns>
        /// <exception cref="ObjectDisposedException">The owning handle has been disposed, or the buffer is already released.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="elementIndex"/> is past the end.</exception>
        public override MemoryHandle Pin(int elementIndex = 0)
        {
            if (Volatile.Read(ref _disposed) != 0)
                throw new ObjectDisposedException(nameof(UnmanagedMemoryManager<T>), "the NumSharp buffer behind this Memory<T> has been released (its OrtTensor handle was disposed).");
            if ((uint)elementIndex > (uint)_length)
                throw new ArgumentOutOfRangeException(nameof(elementIndex));
            // The pin's own reference: it keeps the buffer out of the pool until this pin's MemoryHandle is disposed,
            // whatever happens to the handle and the source meanwhile. TryAddRef fails only once every reference is gone,
            // which the handle's own reference rules out while the manager is undisposed — but a pin racing the handle's
            // Dispose can land after it, and must then be refused rather than address a released buffer.
            if (!_owner.TryAddRef())
                throw new ObjectDisposedException(nameof(UnmanagedMemoryManager<T>), "the NumSharp buffer behind this Memory<T> has already been released.");
            return new MemoryHandle(_pointer + elementIndex, default, this);
        }

        /// <summary>
        ///     Releases the reference one <see cref="Pin"/> took. <see cref="MemoryHandle.Dispose"/> calls this once per
        ///     handle it disposes (and clears the handle), so pins and unpins balance.
        /// </summary>
        public override void Unpin() => _owner.Release();

        /// <summary>
        ///     Refuses new spans and pins from now on; outstanding pins keep their references (released by their own
        ///     <see cref="Unpin"/>). The manager holds no reference of its own, so there is nothing else to release.
        /// </summary>
        /// <param name="disposing">Unused: there is no managed or unmanaged state to free here.</param>
        protected override void Dispose(bool disposing)
        {
            Interlocked.Exchange(ref _disposed, 1);
        }
    }
}
