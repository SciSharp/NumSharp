using System;
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
    ///     <para><b>Keep the handle alive across <c>session.Run</c>.</b> Pass <see cref="Value"/> to ORT, and
    ///     dispose the handle (<c>using</c>) after the run returns — exactly like keeping a pinned buffer
    ///     alive across a native call. Letting the handle be collected mid-run would free the buffer under
    ///     ORT; the finalizer is a leak-prevention safety net, not a lifetime model.</para>
    ///
    ///     <para>This is the deterministic analog of the pythonnet bridge's export keeper: ORT hands us a
    ///     real <c>Dispose</c>, so no <c>weakref.finalize</c> dance is needed.</para>
    /// </summary>
    public sealed class OrtTensor : IDisposable
    {
        private readonly OrtValue _value;
        private readonly NDArray _source;
        private readonly IArraySlice _slice;
        private readonly bool _ownsSource;
        private readonly TensorElementType _elementType;
        private readonly long[] _shape;
        private int _disposed;

        internal OrtTensor(OrtValue value, NDArray source, IArraySlice slice, bool ownsSource, TensorElementType elementType, long[] shape)
        {
            _value = value;
            _source = source;
            _slice = slice;
            _ownsSource = ownsSource;
            _elementType = elementType;
            _shape = shape;
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
        ///     (freeing the memory if it was the last one). Idempotent.
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
                ReleaseCore();
            }
        }

        /// <summary>
        ///     Safety net for a handle that was never disposed: drop the ARC pin so the NumSharp buffer is
        ///     not leaked for the life of the process. <see cref="OrtValue"/> is disposed too (its own
        ///     <c>Dispose</c> is idempotent, so the order relative to its finalizer does not matter).
        /// </summary>
        ~OrtTensor()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            try
            {
                _value.Dispose();
            }
            catch
            {
                // a failing native release must never take down the finalizer thread
            }
            finally
            {
                ReleaseCore();
            }
        }

        private void ReleaseCore()
        {
            _slice.Release();
            NDArrayOnnxInterop.ExportClosed();
            if (_ownsSource)
                _source.Dispose();   // the Decimal→Double conversion temporary nobody else can dispose
            else
                GC.KeepAlive(_source);
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
    ///     <c>OrtValue.CreateTensorValueFromMemory</c>. Keep the handle alive across the run.</para>
    /// </summary>
    public sealed class OrtTensor<T> : IDisposable where T : unmanaged
    {
        private readonly DenseTensor<T> _tensor;
        private readonly UnmanagedMemoryManager<T> _manager;
        private readonly NDArray _source;
        private readonly IArraySlice _slice;
        private readonly bool _ownsSource;
        private int _disposed;

        internal OrtTensor(DenseTensor<T> tensor, UnmanagedMemoryManager<T> manager, NDArray source, IArraySlice slice, bool ownsSource)
        {
            _tensor = tensor;
            _manager = manager;
            _source = source;
            _slice = slice;
            _ownsSource = ownsSource;
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

        /// <summary>The tensor's backing <see cref="Memory{T}"/> — NumSharp's memory, for <c>OrtValue.CreateTensorValueFromMemory</c>.</summary>
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

        /// <summary>Release this handle's reference on the NumSharp buffer and invalidate the Memory. Idempotent.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            GC.SuppressFinalize(this);
            ReleaseCore();
        }

        /// <summary>Safety net for a handle that was never disposed (drops the ARC pin).</summary>
        ~OrtTensor()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            ReleaseCore();
        }

        private void ReleaseCore()
        {
            ((IDisposable)_manager).Dispose();
            _slice.Release();
            NDArrayOnnxInterop.ExportClosed();
            if (_ownsSource)
                _source.Dispose();
            else
                GC.KeepAlive(_source);
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
