using System;
using System.Diagnostics.CodeAnalysis;
using System.Numerics.Tensors;
using System.Threading;
using NumSharp.Backends.Unmanaged;
using NumSharp.Backends.Unmanaged.Pooling;

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
        /// <remarks>
        ///     For an empty array the span keeps the array's shape, with every stride 0, and fronts an empty managed
        ///     array instead of the NumSharp buffer: it has no element to address, so where it points is unobservable,
        ///     and it does not depend on what address NumSharp gives a zero-size buffer.
        /// </remarks>
        /// <value>A <see cref="TensorSpan{T}"/> with the array's lengths and element strides.</value>
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
                // Not TensorSpan<T>.Empty: that is rank 0 (the shape would be lost) and the BCL's own FlattenTo /
                // Tensor.Sum throw IndexOutOfRangeException on it. The all-zero strides make the shaped span legal.
                return _empty ? new TensorSpan<T>(Array.Empty<T>(), _lengths, _strides) : new TensorSpan<T>(_data, _dataLength, _lengths, _strides);
            }
        }

        /// <summary>The read-only tensor view over the NumSharp buffer. Available for every layout, broadcast included. Rebuilt on each access.</summary>
        /// <remarks>An empty array's span keeps its shape, with every stride 0 (see <see cref="Span"/>).</remarks>
        /// <value>A <see cref="ReadOnlyTensorSpan{T}"/> with the array's lengths and element strides.</value>
        /// <exception cref="ObjectDisposedException">The handle has been disposed.</exception>
        public ReadOnlyTensorSpan<T> ReadOnlySpan
        {
            get
            {
                ThrowIfDisposed();
                return _empty ? new ReadOnlyTensorSpan<T>(Array.Empty<T>(), _lengths, _strides) : new ReadOnlyTensorSpan<T>(_data, _dataLength, _lengths, _strides);
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
    ///     Lock-free and safe from finalizer threads.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why the pinned bytes are booked.</b> The backing array is managed memory the GC already counts, but the
    ///     pin is released only by the last view's Dispose or its finalizer. A FORGOTTEN view therefore keeps its array
    ///     alive past the collection that found the view dead; the GC reads that as large objects surviving and collects
    ///     less often, and 5,000 forgotten views of fresh 602 KB tensors held +2.6 GB with nothing reported (P8 Q3 of the
    ///     bridge QA, measured 2026-11-06). The lease books the bytes with NumSharp's memory accounting
    ///     (<see cref="SizeBucketedBufferPool.BookForeign"/>), which paces a collection only when the bookings GROW - the
    ///     same loop peaked at +96 MB - so code that disposes its views pays nothing.</para>
    ///     <para><b>Why not <see cref="GC.AddMemoryPressure"/>.</b> The GC counts every addition toward its next full
    ///     collection even after the matching removal: one call per lease ran 12-41 gen-2 collections per 5,000 views
    ///     disposed at once, and reported the array once per view over it. The owner chose the booking (2026-11-06).</para>
    ///     <para>Booking is accounting only: the array is never copied into, taken from or returned to NumSharp's pool. It
    ///     stays the tensor's, and the GC frees it once neither the tensor nor a pin holds it.</para>
    /// </remarks>
    internal sealed class ImportLease
    {
        /// <summary>What the release runs (unpins, keeps the tensor reachable until then); swapped to null by the one <see cref="Release"/> that runs it.</summary>
        private Action _release;

        /// <summary>The bytes of the pinned elements the views address, booked from construction to <see cref="Release"/>.</summary>
        private readonly long _bytes;

        /// <summary>1 once <see cref="Release"/> ran: its once-guard, safe across threads (the finalizer's included).</summary>
        private int _released;

        /// <summary>Opens a lease: books its bytes and counts it in <see cref="NDArrayTensorsInterop.LiveImports"/>.</summary>
        /// <param name="release">What to run when the last NumSharp view over the memory is gone: dispose the pin.</param>
        /// <param name="bytes">The bytes of the pinned elements the views address; nothing is booked for zero or less.</param>
        internal ImportLease(Action release, long bytes)
        {
            _release = release;
            _bytes = bytes;
            if (_bytes > 0)
                SizeBucketedBufferPool.BookForeign(_bytes);   // given back by Release, on whichever thread runs it
            NDArrayTensorsInterop.ImportOpened();
        }

        /// <summary><c>true</c> once <see cref="Release"/> ran: the pin is disposed (or being disposed) and the booking given back.</summary>
        internal bool IsReleased => Volatile.Read(ref _released) != 0;

        /// <summary>
        ///     Disposes the pin, gives the booking back and uncounts the lease - exactly once, whichever of the last view's
        ///     Dispose and a finalizer calls it first; every later call returns at once.
        /// </summary>
        /// <remarks>
        ///     The booking and the count are given back in <c>finally</c>, so a release action that throws still balances
        ///     NumSharp's books. Besides the action it touches counters only, so it is safe on the finalizer thread.
        /// </remarks>
        /// <exception cref="InvalidOperationException">The release action found its pin freed already (no caller does
        ///     that). The booking and the count are given back all the same.</exception>
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
                    SizeBucketedBufferPool.ReturnForeign(_bytes);
                NDArrayTensorsInterop.ImportClosed();
            }
        }
    }
}
