using System;
using System.Threading;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Interop.MLNet
{
    /// <summary>
    ///     One NumSharp-side lease on a pinned ML.NET buffer (a dense <see cref="Microsoft.ML.Data.VBuffer{T}"/>'s
    ///     backing <c>T[]</c>), wired as the <see cref="NumSharp.Backends.Unmanaged.UnmanagedMemoryBlock{T}"/> dispose
    ///     hook so it fires exactly once when the LAST NumSharp view over the memory (derived slices included) drops
    ///     its ARC reference — via <see cref="NDArray.Dispose"/>, the NDArray finalizer, or the memory-block Disposer
    ///     finalizer. Lock-free and safe from finalizer threads: it only frees a <see cref="System.Runtime.InteropServices.GCHandle"/>.
    ///     The same primitive the ONNX bridge leases ORT output buffers with.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why the pinned bytes are booked.</b> The array is managed memory the GC already counts, but the pin is
    ///     released only by the last view's Dispose or its finalizer. A FORGOTTEN view therefore keeps its array alive past
    ///     the collection that found the view dead; the GC reads that as large objects surviving and collects less often,
    ///     and 5,000 forgotten views of fresh 602 KB buffers held +2.3-2.7 GB with nothing reported (P8 Q3 of the bridge
    ///     QA, measured 2026-11-06). The lease books the bytes with NumSharp's memory accounting
    ///     (<see cref="SizeBucketedBufferPool.BookForeign"/>), which paces a collection only when the bookings GROW - the
    ///     same loop peaked at +96-299 MB - so code that disposes its views pays nothing.</para>
    ///     <para><b>Why not <see cref="GC.AddMemoryPressure"/>.</b> The GC counts every addition toward its next full
    ///     collection even after the matching removal: one call per lease ran 12-41 gen-2 collections per 5,000 views
    ///     disposed at once, and reported the array once per view over it. The owner chose the booking (2026-11-06).</para>
    ///     <para>Booking is accounting only: the array is never copied into, taken from or returned to NumSharp's pool. It
    ///     stays ML.NET's, and the GC frees it once neither the VBuffer nor a pin holds it.</para>
    /// </remarks>
    internal sealed class ImportLease
    {
        /// <summary>What the release runs (frees the pin); swapped to null by the one <see cref="Release"/> that runs it.</summary>
        private Action _release;

        /// <summary>The bytes of the pinned array the views address, booked from construction to <see cref="Release"/>.</summary>
        private readonly long _bytes;

        /// <summary>1 once <see cref="Release"/> ran: its once-guard, safe across threads (the finalizer's included).</summary>
        private int _released;

        /// <summary>Opens a lease: books its bytes and counts it in <see cref="NDArrayMLNetInterop.LiveImports"/>.</summary>
        /// <param name="release">What to run when the last NumSharp view over the memory is gone: free the pin.</param>
        /// <param name="bytes">The bytes of the pinned array the views address; nothing is booked for zero or less.</param>
        internal ImportLease(Action release, long bytes)
        {
            _release = release;
            _bytes = bytes;
            if (_bytes > 0)
                SizeBucketedBufferPool.BookForeign(_bytes);   // given back by Release, on whichever thread runs it
            NDArrayMLNetInterop.ImportOpened();
        }

        /// <summary><c>true</c> once <see cref="Release"/> ran: the pin is freed (or being freed) and the booking given back.</summary>
        internal bool IsReleased => Volatile.Read(ref _released) != 0;

        /// <summary>
        ///     Frees the pin, gives the booking back and uncounts the lease - exactly once, whichever of the last view's
        ///     Dispose and a finalizer calls it first; every later call returns at once.
        /// </summary>
        /// <remarks>
        ///     The booking and the count are given back in <c>finally</c>, so a release action that throws still balances
        ///     NumSharp's books. Besides the action it touches counters only, so it is safe on the finalizer thread.
        /// </remarks>
        /// <exception cref="InvalidOperationException">The release action found its pin freed already (a <c>GCHandle.Free</c>
        ///     of a freed handle; no caller does that). The booking and the count are given back all the same.</exception>
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
                NDArrayMLNetInterop.ImportClosed();
            }
        }
    }
}
