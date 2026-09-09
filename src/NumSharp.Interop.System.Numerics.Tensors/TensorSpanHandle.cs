using System;
using System.Diagnostics.CodeAnalysis;
using System.Numerics.Tensors;
using System.Threading;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.Tensors
{
    /// <summary>
    ///     A zero-copy <see cref="TensorSpan{T}"/> / <see cref="ReadOnlyTensorSpan{T}"/> over a NumSharp buffer —
    ///     the handle <see cref="NDArrayTensorsInterop.AsTensorSpan{T}"/> returns.
    ///
    ///     <para><b>Why a handle and not the span directly.</b> <see cref="TensorSpan{T}"/> is a <c>ref struct</c>:
    ///     it cannot be stored in a field, captured, or returned to outlive the pin that keeps its memory valid.
    ///     So the verb returns this small heap object, which owns an ARC reference on the NumSharp buffer and
    ///     rebuilds the span on each <see cref="Span"/> / <see cref="ReadOnlySpan"/> access (unmanaged memory
    ///     never moves, so the pointer is stable). Use it inside a <c>using</c> and take the span from it:</para>
    ///     <code>
    ///     using var h = nd.AsTensorSpan&lt;float&gt;();
    ///     TensorSpan&lt;float&gt; span = h.Span;
    ///     Tensor.Add(span, 1f, span);           // writes straight into the NDArray
    ///     </code>
    ///
    ///     <para><b>Lifetime.</b> The handle takes its own atomic reference on the NumSharp buffer, so the memory
    ///     survives even if the source <see cref="NDArray"/> is disposed or collected while a span is in use;
    ///     disposing the handle releases the reference. The finalizer is a leak-prevention safety net, not a
    ///     lifetime model — dispose it deterministically.</para>
    ///
    ///     <para><b>Writeability.</b> <see cref="Span"/> (the mutable <see cref="TensorSpan{T}"/>) is available
    ///     only when the source array is writeable; a broadcast / read-only NumSharp view exposes
    ///     <see cref="ReadOnlySpan"/> only, and <see cref="Span"/> throws — writing through overlapping
    ///     stride-0 lanes would corrupt data, exactly as NumSharp forbids writing a broadcast view.</para>
    /// </summary>
    [Experimental("NUMSHARP_TENSORS")]
    public sealed unsafe class TensorSpanHandle<T> : IDisposable where T : unmanaged
    {
        private readonly T* _data;
        private readonly nint _dataLength;
        private readonly nint[] _lengths;
        private readonly nint[] _strides;
        private readonly bool _writeable;
        private readonly bool _empty;
        private readonly IArraySlice _slice;
        private readonly NDArray _source;
        private readonly bool _ownsSource;
        private int _disposed;

        internal TensorSpanHandle(T* data, nint dataLength, nint[] lengths, nint[] strides, bool writeable,
                                  bool empty, IArraySlice slice, NDArray source, bool ownsSource)
        {
            _data = data;
            _dataLength = dataLength;
            _lengths = lengths;
            _strides = strides;
            _writeable = writeable;
            _empty = empty;
            _slice = slice;
            _source = source;
            _ownsSource = ownsSource;
            NDArrayTensorsInterop.ExportOpened();
        }

        /// <summary>
        ///     The mutable tensor view over the NumSharp buffer. Writing through it mutates the array. Rebuilt on
        ///     each access (the span is a <c>ref struct</c> and cannot be cached).
        /// </summary>
        /// <exception cref="ObjectDisposedException">The handle has been disposed.</exception>
        /// <exception cref="InvalidOperationException">The source array is a broadcast / read-only view (use <see cref="ReadOnlySpan"/>).</exception>
        public TensorSpan<T> Span
        {
            get
            {
                ThrowIfDisposed();
                if (!_writeable)
                    throw new InvalidOperationException(
                        "the source is a broadcast / read-only NumSharp view (flags.writeable == False), so a mutable TensorSpan<T> " +
                        "would write through overlapping stride-0 lanes. Use ReadOnlySpan, or nd.copy() for a writeable array.");
                return _empty ? TensorSpan<T>.Empty : new TensorSpan<T>(_data, _dataLength, _lengths, _strides);
            }
        }

        /// <summary>The read-only tensor view over the NumSharp buffer. Available for every layout, broadcast included. Rebuilt on each access.</summary>
        /// <exception cref="ObjectDisposedException">The handle has been disposed.</exception>
        public ReadOnlyTensorSpan<T> ReadOnlySpan
        {
            get
            {
                ThrowIfDisposed();
                return _empty ? ReadOnlyTensorSpan<T>.Empty : new ReadOnlyTensorSpan<T>(_data, _dataLength, _lengths, _strides);
            }
        }

        /// <summary>The NumSharp array the span reads (the array passed to <see cref="NDArrayTensorsInterop.AsTensorSpan{T}"/>).</summary>
        public NDArray Source => _source;

        /// <summary>The tensor view's lengths (a copy).</summary>
        public nint[] Lengths => (nint[])_lengths.Clone();

        /// <summary>The tensor view's element strides (a copy).</summary>
        public nint[] Strides => (nint[])_strides.Clone();

        /// <summary><c>true</c> when <see cref="Span"/> is available (the source is writeable).</summary>
        public bool IsWriteable => _writeable;

        /// <summary><c>true</c> once <see cref="Dispose"/> ran; <see cref="Span"/> / <see cref="ReadOnlySpan"/> then throw.</summary>
        public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

        /// <summary>Release this handle's reference on the NumSharp buffer (freeing it if this was the last one). Idempotent.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            GC.SuppressFinalize(this);
            ReleaseCore();
        }

        /// <summary>Safety net for a handle that was never disposed (drops the ARC pin so the buffer is not leaked for the process life).</summary>
        ~TensorSpanHandle()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            ReleaseCore();
        }

        private void ReleaseCore()
        {
            _slice.Release();
            NDArrayTensorsInterop.ExportClosed();
            if (_ownsSource)
                _source.Dispose();
            else
                GC.KeepAlive(_source);
        }

        private void ThrowIfDisposed()
        {
            if (Volatile.Read(ref _disposed) != 0)
                throw new ObjectDisposedException(nameof(TensorSpanHandle<T>),
                    "the TensorSpanHandle<T> has been disposed; the NumSharp buffer behind the span is no longer pinned.");
        }
    }

    /// <summary>
    ///     One NumSharp-side lease on a pinned <see cref="System.Numerics.Tensors.Tensor{T}"/> managed backing
    ///     store, wired as the <see cref="UnmanagedMemoryBlock{T}"/> dispose hook so it fires exactly once when
    ///     the LAST NumSharp view over the memory (derived slices included) drops its ARC reference — via
    ///     <see cref="NDArray.Dispose"/>, the NDArray finalizer, or the memory-block Disposer finalizer.
    ///     Lock-free and safe from finalizer threads. The analog of the ONNX bridge's <c>ImportLease</c>.
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
            NDArrayTensorsInterop.ImportOpened();
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
                NDArrayTensorsInterop.ImportClosed();
            }
        }
    }
}
