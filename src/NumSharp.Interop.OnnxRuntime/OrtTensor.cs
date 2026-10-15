using System;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.OnnxRuntime
{
    /// <summary>
    ///     A zero-copy ONNX Runtime tensor over a NumSharp buffer — the handle
    ///     <see cref="NDArrayOnnxInterop.AsOrtValue"/> returns. It owns BOTH the <see cref="OrtValue"/>
    ///     (which does not own the memory it points at) and an ARC reference on the NumSharp buffer, so the
    ///     memory stays valid for ORT even if the source <see cref="NDArray"/> is disposed or collected
    ///     while the tensor is in use; disposing the handle disposes the OrtValue and releases the pin.
    ///
    ///     <para><b>Reachable means valid.</b> The handle and its <see cref="Value"/> keep each other alive: holding
    ///     either one keeps the buffer referenced, so a caller that keeps only the <see cref="OrtValue"/> (in an input
    ///     list, say) never sees it disposed or its memory freed by a collection. When both become unreachable the
    ///     finalizer drops the reference — abandoning it, never freeing eagerly, so an alias of the buffer the caller
    ///     still holds stays valid — and ORT's own finalizer releases the native value.</para>
    ///
    ///     <para><b>Keep the handle (or its <see cref="Value"/>) reachable across <c>session.Run</c>.</b> ORT's own
    ///     <c>Run</c> copies the inputs' native handles and then no longer references the managed <see cref="OrtValue"/>s,
    ///     so a value referenced only by the argument list can be collected while the native run reads it — the rule ORT
    ///     sets for every <see cref="OrtValue"/>. A <c>using</c> around the run is the idiom. The same holds for every
    ///     native consumer that keeps the value past the call (<c>OrtIoBinding</c>, <c>SessionOptions.AddInitializer</c>,
    ///     a sequence built with <c>OrtValue.CreateSequence</c>): native references are invisible to the GC.</para>
    ///
    ///     <para>This is the deterministic analog of the pythonnet bridge's export keeper: ORT hands us a
    ///     real <c>Dispose</c>, so no <c>weakref.finalize</c> dance is needed.</para>
    /// </summary>
    public sealed class OrtTensor : IDisposable
    {
        /// <summary>
        ///     Every live <see cref="OrtValue"/> this package created over a NumSharp buffer, mapped to its handle. A
        ///     conditional weak table keeps the VALUE (the handle) alive exactly as long as the KEY (the OrtValue) is
        ///     reachable, without keeping the key alive itself — the only way to attach a reference to an object of a type
        ///     we do not own.
        /// </summary>
        /// <remarks>
        ///     Without it a caller holding just the <see cref="OrtValue"/> had nothing rooting the handle: the handle's
        ///     finalizer disposed the OrtValue under the caller (the next native call on it faults) and released the buffer
        ///     reference, so the native tensor pointed into freed memory. The handle → value → handle cycle is ephemeral:
        ///     the table does not count as a root for its own keys, so the pair is collected together once nothing else
        ///     reaches either.
        /// </remarks>
        private static readonly ConditionalWeakTable<OrtValue, OrtTensor> ValueOwners = new ConditionalWeakTable<OrtValue, OrtTensor>();

        private readonly OrtValue _value;
        private readonly NDArray _source;
        private readonly IArraySlice _slice;
        private readonly bool _ownsSource;
        private readonly TensorElementType _elementType;
        private readonly long[] _shape;
        private int _disposed;

        /// <summary>Takes ownership of <paramref name="value"/> and of the reference <paramref name="slice"/> carries.</summary>
        /// <param name="value">The OrtValue over the buffer (created by the caller; owned by the handle from here on).</param>
        /// <param name="source">The array ORT reads (the caller's, or a temporary the handle owns).</param>
        /// <param name="slice">The buffer, on which the caller already took the reference this handle now owns.</param>
        /// <param name="ownsSource"><c>true</c> when <paramref name="source"/> is a temporary the handle must dispose.</param>
        /// <param name="elementType">The ONNX element type the value was created with.</param>
        /// <param name="shape">The value's row-major shape.</param>
        /// <exception cref="OutOfMemoryException">The value could not be registered (the handle is then left unconstructed: its finalizer is suppressed, and the caller still owns the value and the reference).</exception>
        internal OrtTensor(OrtValue value, NDArray source, IArraySlice slice, bool ownsSource, TensorElementType elementType, long[] shape)
        {
            _value = value;
            _source = source;
            _slice = slice;
            _ownsSource = ownsSource;
            _elementType = elementType;
            _shape = shape;
            try
            {
                ValueOwners.AddOrUpdate(value, this);
            }
            catch
            {
                // An object with a finalizer is registered for finalization when it is ALLOCATED, so a constructor that
                // throws still leaves its finalizer to run — which would abandon the reference the caller's cleanup also
                // releases (one reference given back twice, the second one someone else's). Nothing of this handle was
                // handed out, so its finalizer must not run at all.
                GC.SuppressFinalize(this);
                throw;
            }
            NDArrayOnnxInterop.ExportOpened();
        }

        /// <summary>The ORT tensor, for <c>session.Run</c> / <c>OrtIoBinding</c>. Valid until the handle is disposed.</summary>
        public OrtValue Value
        {
            get
            {
                ThrowIfDisposed();
                return _value;
            }
        }

        /// <summary>
        ///     The NumSharp array ORT reads — the array passed to <see cref="NDArrayOnnxInterop.AsOrtValue"/>,
        ///     or, for a <see cref="NPTypeCode.Decimal"/> source, the float64 conversion temporary this handle
        ///     owns (mutations through ORT land in the temporary, not in the Decimal source).
        /// </summary>
        public NDArray Source => _source;

        /// <summary>The ONNX element type the tensor was created with.</summary>
        public TensorElementType ElementType => _elementType;

        /// <summary>The tensor's row-major shape (a copy).</summary>
        public long[] Shape => (long[])_shape.Clone();

        /// <summary><c>true</c> once <see cref="Dispose"/> ran; <see cref="Value"/> then throws.</summary>
        public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

        /// <summary>
        ///     Dispose the <see cref="OrtValue"/> and release this handle's reference on the NumSharp buffer
        ///     (freeing the memory if it was the last one), then the owned Decimal conversion temporary, if any.
        ///     Idempotent.
        /// </summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            GC.SuppressFinalize(this);
            try
            {
                _value.Dispose();
            }
            finally
            {
                // The disposed value no longer needs to keep the handle alive.
                ValueOwners.Remove(_value);
                _slice.Release();
                NDArrayOnnxInterop.ExportClosed();
                if (_ownsSource)
                    _source.Dispose();   // the Decimal→Double conversion temporary nobody else can dispose
                else
                    GC.KeepAlive(_source);
            }
        }

        /// <summary>
        ///     Safety net for a handle that was never disposed: gives the handle's reference on the NumSharp buffer back
        ///     so the buffer is not referenced for the life of the process — and does nothing else.
        /// </summary>
        /// <remarks>
        ///     <para><b>Abandon, not release.</b> The handle being unreachable proves nothing about OTHER reachable aliases
        ///     of the buffer that hold no counted reference (a bare <c>IArraySlice</c> or <c>UnmanagedStorage</c> taken from
        ///     the array): releasing the last reference here freed the buffer under such an alias, which then read whatever
        ///     the pool handed out next. <see cref="IArraySlice.Abandon"/> drops the count without freeing; the memory
        ///     block's own finalizer frees the buffer once nothing can reach it (the rule NumSharp's own <c>~NDArray</c>
        ///     follows).</para>
        ///     <para><b>Nothing else is touched.</b> The <see cref="OrtValue"/> is unreachable whenever this runs (it keeps the
        ///     handle reachable through <c>ValueOwners</c>), so ORT's own finalizer releases it; and the owned conversion
        ///     temporary, which a caller may still hold through <see cref="Source"/>, is released by its own finalizer if it
        ///     is unreachable too. Disposing either from here disposed objects a caller could still be using.</para>
        ///     <para><b>Any finalization order is safe:</b> this only decrements a count (an abandon after the block was freed
        ///     is a no-op on memory) and a counter.</para>
        /// </remarks>
        ~OrtTensor()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            _slice.Abandon();
            NDArrayOnnxInterop.ExportClosed();
        }

        private void ThrowIfDisposed()
        {
            if (Volatile.Read(ref _disposed) != 0)
                throw new ObjectDisposedException(nameof(OrtTensor), "the OrtTensor handle has been disposed; its OrtValue is gone and the NumSharp buffer is no longer pinned.");
        }
    }

    /// <summary>
    ///     A zero-copy <see cref="DenseTensor{T}"/> over a NumSharp buffer — the handle
    ///     <see cref="NDArrayOnnxInterop.AsDenseTensor{T}"/> returns. The tensor's <see cref="DenseTensor{T}.Buffer"/>
    ///     is a <see cref="Memory{T}"/> fronting NumSharp's unmanaged memory (through an internal
    ///     <see cref="System.Buffers.MemoryManager{T}"/>), and the handle holds the ARC reference that keeps
    ///     that memory alive; disposing it releases the reference and invalidates the Memory.
    ///
    ///     <para>Feed <see cref="Tensor"/> to the legacy <c>NamedOnnxValue.CreateFromTensor</c> /
    ///     <c>session.Run(IReadOnlyCollection&lt;NamedOnnxValue&gt;)</c> API, or wrap <see cref="Memory"/> with
    ///     <c>OrtValue.CreateTensorValueFromMemory</c>. Keep the handle — or the tensor — reachable across the run.</para>
    ///
    ///     <para><b>Reachable means valid.</b> The tensor's memory keeps this handle reachable, so as long as the caller
    ///     holds the <see cref="Tensor"/> or the <see cref="Memory"/> the handle keeps its reference on the buffer and a
    ///     collection never disables them; only <see cref="Dispose"/> does. When nothing reaches any of them the
    ///     finalizer drops the reference (abandoning it, never freeing eagerly, so an alias of the buffer the caller still
    ///     holds stays valid).</para>
    ///
    ///     <para><b>Pins outlive the handle safely.</b> Every pin of <see cref="Memory"/> — the one
    ///     <c>CreateTensorValueFromMemory</c> keeps inside the <c>OrtValue</c>, any <c>Memory.Pin()</c> — holds its own
    ///     reference on the NumSharp buffer, given back exactly once when the pin is disposed or, if it never is, when it
    ///     is collected; and a NumSharp view made with <c>Tensor.AsNDArray()</c> is an ordinary alias of the buffer, counted
    ///     on it like any view. So the buffer stays alive until the last of them is gone even if this handle and the source
    ///     array are disposed first. What the disposed handle refuses is NEW access: <c>Memory.Span</c>,
    ///     <c>Memory.Pin()</c>, <c>Tensor.AsNDArray()</c> and the properties below throw
    ///     <see cref="ObjectDisposedException"/>.</para>
    /// </summary>
    public sealed class OrtTensor<T> : IDisposable where T : unmanaged
    {
        private readonly DenseTensor<T> _tensor;
        private readonly UnmanagedMemoryManager<T> _manager;
        private readonly NDArray _source;
        private readonly IArraySlice _slice;
        private readonly bool _ownsSource;
        private int _disposed;

        /// <summary>Takes ownership of the reference <paramref name="slice"/> carries and ties the manager's lifetime to this handle.</summary>
        /// <param name="tensor">The tensor over <paramref name="manager"/>'s memory.</param>
        /// <param name="manager">The memory manager fronting the buffer; it keeps this handle reachable from now on.</param>
        /// <param name="source">The array the tensor reads (the caller's, or a temporary the handle owns).</param>
        /// <param name="slice">The buffer, on which the caller already took the reference this handle now owns.</param>
        /// <param name="ownsSource"><c>true</c> when <paramref name="source"/> is a temporary the handle must dispose.</param>
        internal OrtTensor(DenseTensor<T> tensor, UnmanagedMemoryManager<T> manager, NDArray source, IArraySlice slice, bool ownsSource)
        {
            _tensor = tensor;
            _manager = manager;
            _source = source;
            _slice = slice;
            _ownsSource = ownsSource;
            // The memory keeps the handle reachable: while the caller holds the tensor or its Memory, the handle's
            // reference stays counted and its finalizer cannot run (see AttachLifetimeOwner).
            manager.AttachLifetimeOwner(this);
            NDArrayOnnxInterop.ExportOpened();
        }

        /// <summary>The ORT tensor sharing the NumSharp buffer. Valid until the handle is disposed.</summary>
        public DenseTensor<T> Tensor
        {
            get
            {
                ThrowIfDisposed();
                return _tensor;
            }
        }

        /// <summary>
        ///     The tensor's backing <see cref="Memory{T}"/> — NumSharp's memory, for <c>OrtValue.CreateTensorValueFromMemory</c>.
        ///     A pin taken on it (which that factory takes and keeps in the <c>OrtValue</c>) holds its own reference on the
        ///     NumSharp buffer until the pin is released, so the <c>OrtValue</c> stays valid even after this handle is disposed.
        /// </summary>
        /// <exception cref="ObjectDisposedException">The handle has been disposed.</exception>
        public Memory<T> Memory
        {
            get
            {
                ThrowIfDisposed();
                return _tensor.Buffer;
            }
        }

        /// <summary>
        ///     The NumSharp array the tensor reads — the array passed to <see cref="NDArrayOnnxInterop.AsDenseTensor{T}"/>,
        ///     or the float64 conversion temporary this handle owns for a <see cref="NPTypeCode.Decimal"/> source.
        /// </summary>
        public NDArray Source => _source;

        /// <summary><c>true</c> once <see cref="Dispose"/> ran.</summary>
        public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

        /// <summary>
        ///     Invalidate the Memory (new spans and pins are refused from now on; pins taken before stay valid), release
        ///     this handle's reference on the NumSharp buffer, then the owned Decimal conversion temporary, if any.
        ///     Idempotent.
        /// </summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            GC.SuppressFinalize(this);
            ((IDisposable)_manager).Dispose();
            _slice.Release();
            NDArrayOnnxInterop.ExportClosed();
            if (_ownsSource)
                _source.Dispose();
            else
                GC.KeepAlive(_source);
        }

        /// <summary>
        ///     Safety net for a handle that was never disposed: gives the handle's reference on the NumSharp buffer back —
        ///     abandoning it, never freeing eagerly — and does nothing else.
        /// </summary>
        /// <remarks>
        ///     <para>This runs only once nothing reaches the tensor or its Memory either (the memory manager keeps the handle
        ///     reachable), so the manager needs no disposing — it used to be disposed here, which turned a tensor the caller
        ///     still held into <see cref="ObjectDisposedException"/> after a mere collection. The owned conversion temporary,
        ///     which a caller may still hold through <see cref="Source"/>, is released by its own finalizer once it is
        ///     unreachable too.</para>
        ///     <para><see cref="IArraySlice.Abandon"/> rather than <see cref="IArraySlice.Release"/>: an alias of the buffer
        ///     that holds no counted reference (a bare <c>IArraySlice</c> taken from the array) may still be reachable, and
        ///     releasing the last reference here freed the buffer under it. Any finalization order is safe: this only
        ///     decrements a count and a counter.</para>
        /// </remarks>
        ~OrtTensor()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            _slice.Abandon();
            NDArrayOnnxInterop.ExportClosed();
        }

        private void ThrowIfDisposed()
        {
            if (Volatile.Read(ref _disposed) != 0)
                throw new ObjectDisposedException(nameof(OrtTensor<T>), "the OrtTensor<T> handle has been disposed; the NumSharp buffer behind the DenseTensor is no longer pinned.");
        }
    }

    /// <summary>
    ///     One NumSharp-side lease on foreign memory (an ORT output buffer, or a pinned managed
    ///     <see cref="Memory{T}"/> behind a <see cref="DenseTensor{T}"/>), wired as the
    ///     <see cref="UnmanagedMemoryBlock{T}"/> dispose hook so it fires exactly once when the LAST NumSharp
    ///     view over the memory (derived slices included) drops its ARC reference — via <see cref="NDArray.Dispose"/>,
    ///     the NDArray finalizer, or the memory-block Disposer finalizer. Lock-free and safe from finalizer
    ///     threads: it only touches CLR state and ORT's (idempotent) <c>Dispose</c>.
    /// </summary>
    internal sealed class ImportLease
    {
        private Action _release;
        private readonly long _bytes;
        private int _released;

        internal ImportLease(Action release, long bytes)
        {
            _release = release;
            _bytes = bytes;
            if (_bytes > 0)
                GC.AddMemoryPressure(_bytes);
            NDArrayOnnxInterop.ImportOpened();
        }

        internal bool IsReleased => Volatile.Read(ref _released) != 0;

        internal void Release()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0)
                return;
            try
            {
                Action release = Interlocked.Exchange(ref _release, null);
                release?.Invoke();
            }
            finally
            {
                if (_bytes > 0)
                    GC.RemoveMemoryPressure(_bytes);
                NDArrayOnnxInterop.ImportClosed();
            }
        }
    }
}
