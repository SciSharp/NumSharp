using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;
using NumSharp.Backends.Unmanaged;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Interop.TorchSharp
{
    /// <summary>
    ///     One NumSharp buffer lent to libtorch: the ARC reference that keeps the buffer alive while a torch storage made by
    ///     <c>from_blob</c> points at it, released by libtorch's deleter when that storage dies.
    /// </summary>
    /// <remarks>
    ///     <para><b>When it ends.</b> libtorch frees a storage when its LAST tensor dies - the exported tensor and every torch
    ///     view of it (<c>t[0]</c>, <c>t.T</c>, an <c>expand</c>), whether disposed, collected by a <c>DisposeScope</c>, or
    ///     finalized. A torch view the caller forgets (an indexing temporary like <c>t[0, 0].item&lt;float&gt;()</c>) keeps the
    ///     NumSharp buffer until the GC finalizes it, exactly as it keeps a torch-allocated buffer.</para>
    ///     <para><b>Why Release, not Abandon.</b> NumSharp's own finalizers ABANDON a reference (they cannot prove no uncounted
    ///     alias still reads the buffer). This release is not a reachability guess: libtorch calls the deleter exactly when
    ///     nothing in torch can read the memory any more, so the deterministic <see cref="IArraySlice.Release"/> is right, on
    ///     whichever thread libtorch calls it.</para>
    ///     <para><b>Why a delegate per lease.</b> libtorch passes the deleter nothing but the data pointer, and two exports of
    ///     the same array share it. A closure per lease knows exactly which export ended, so a failed <c>THSTensor_new</c> can
    ///     tell whether libtorch already released the buffer (<see cref="IsReleased"/>). The delegate is rooted in
    ///     <see cref="Live"/> until it runs, because libtorch holds only its native thunk. It costs about 0.7 us per export
    ///     over a shared callback (measured: 3.95 vs 3.24 us for a 64-element export and dispose).</para>
    /// </remarks>
    internal sealed class ExportLease
    {
        /// <summary>Every lease libtorch has not released yet: what keeps each delegate (and its native thunk) alive.</summary>
        private static readonly ConcurrentDictionary<ExportLease, byte> Live = new ConcurrentDictionary<ExportLease, byte>();

        /// <summary>The ARC reference this lease owns; taken by the one <see cref="Release"/> that runs.</summary>
        private IArraySlice _slice;

        /// <summary>The delegate behind <see cref="FunctionPointer"/>: must outlive every call libtorch can make to the thunk.</summary>
        private readonly LibTorchSharpNative.Deleter _deleter;

        /// <summary>1 once <see cref="Release"/> ran: the once-guard between libtorch's deleter and a failed export's cleanup.</summary>
        private int _released;

        /// <summary>
        ///     Opens a lease over one ARC reference the caller already took (<see cref="IArraySlice.TryAddRef"/>) and hands to
        ///     this lease: from here on only <see cref="Release"/> gives it back.
        /// </summary>
        /// <param name="slice">The buffer's slice, with one reference owned by this lease.</param>
        internal ExportLease(IArraySlice slice)
        {
            _slice = slice;
            _deleter = OnStorageFreed;
            FunctionPointer = Marshal.GetFunctionPointerForDelegate(_deleter);
            Live[this] = 0;
            NDArrayTorchInterop.ExportOpened();
        }

        /// <summary>The native function pointer to pass to <c>THSTensor_new</c> as the deleter.</summary>
        internal IntPtr FunctionPointer { get; }

        /// <summary><c>true</c> once the reference was given back (by libtorch's deleter or a failed export's cleanup).</summary>
        internal bool IsReleased => Volatile.Read(ref _released) != 0;

        /// <summary>The number of leases libtorch has not released yet, across the process (a diagnostic for the tests).</summary>
        internal static int LiveCount => Live.Count;

        /// <summary>
        ///     libtorch's deleter. Runs on whatever thread drops the storage's last reference, so it must never throw: an
        ///     exception unwinding into libtorch's C++ frames ends the process.
        /// </summary>
        /// <param name="data">The pointer the storage was made over (unused: the lease knows its buffer).</param>
        private void OnStorageFreed(IntPtr data)
        {
            try
            {
                Release();
            }
            catch
            {
                // Swallowed on purpose: nothing above this frame can handle it (libtorch's destructor called us), and Release
                // already balanced the counters in its finally. A reference it failed to give back stays with the buffer,
                // which NumSharp's own Disposer finalizer still frees once nothing reaches it.
            }
        }

        /// <summary>
        ///     Gives the buffer reference back and closes the lease - exactly once, whichever of libtorch's deleter and a
        ///     failed export calls it first; every later call returns at once.
        /// </summary>
        /// <remarks>
        ///     The count and the <see cref="Live"/> entry are dropped in <c>finally</c>, so a release that throws still
        ///     balances. Removing the entry lets the GC collect the delegate: libtorch never calls a deleter twice, and a
        ///     failed export only gets here when the storage was never created or is already gone.
        /// </remarks>
        internal void Release()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0)
                return;
            try
            {
                Interlocked.Exchange(ref _slice, null)?.Release();
            }
            finally
            {
                Live.TryRemove(this, out _);
                NDArrayTorchInterop.ExportClosed();
            }
        }
    }

    /// <summary>
    ///     One torch storage lent to NumSharp: an <c>at::Tensor*</c> of the bridge's own (a <c>detach()</c> of the caller's
    ///     tensor, or a private intermediate such as the CPU copy of a CUDA tensor) that keeps the storage alive while a NumSharp
    ///     view reads it, wired as the view's memory-block dispose hook, so it is disposed exactly once when the LAST NumSharp
    ///     view over the memory (derived slices included) drops its reference - by <see cref="NDArray.Dispose"/>, the NDArray
    ///     finalizer, or the memory-block Disposer finalizer.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why a raw handle, not a <c>torch.Tensor</c>.</b> A managed tensor registers on the thread's current
    ///     <c>DisposeScope</c> when it is made, and the scope would dispose it - freeing the storage - while the NumSharp view
    ///     still reads it. A raw handle belongs to no scope, has no finalizer of its own, and is not counted in
    ///     <c>torch.Tensor.TotalCount</c>.</para>
    ///     <para><b>Why the storage bytes are booked.</b> The GC does not see libtorch's memory. A FORGOTTEN view keeps the whole
    ///     storage alive until a collection finalizes it, so the lease books the storage's byte size with NumSharp's memory
    ///     accounting (<see cref="SizeBucketedBufferPool.BookForeign"/>), which paces a collection when the bookings grow - the
    ///     rule every bridge lease follows. Booking is accounting only: the memory stays libtorch's.</para>
    /// </remarks>
    internal sealed class ImportLease
    {
        /// <summary>The owned <c>at::Tensor*</c>; swapped to zero by the one <see cref="Release"/> that disposes it.</summary>
        private IntPtr _handle;

        /// <summary>The bytes booked from construction to <see cref="Release"/>: the whole storage the handle keeps alive.</summary>
        private readonly long _bytes;

        /// <summary>1 once <see cref="Release"/> ran: its once-guard, safe across threads (the finalizer's included).</summary>
        private int _released;

        /// <summary>Opens a lease over an owned handle: books its storage and counts it in <see cref="NDArrayTorchInterop.LiveImports"/>.</summary>
        /// <param name="handle">An <c>at::Tensor*</c> this lease now owns and alone disposes.</param>
        /// <param name="bytes">The storage's byte size; nothing is booked for zero or less.</param>
        internal ImportLease(IntPtr handle, long bytes)
        {
            _handle = handle;
            _bytes = bytes;
            if (_bytes > 0)
                SizeBucketedBufferPool.BookForeign(_bytes);   // given back by Release, on whichever thread runs it
            NDArrayTorchInterop.ImportOpened();
        }

        /// <summary><c>true</c> once <see cref="Release"/> ran: the handle is disposed and the booking given back.</summary>
        internal bool IsReleased => Volatile.Read(ref _released) != 0;

        /// <summary>
        ///     Disposes the handle, gives the booking back and uncounts the lease - exactly once, whichever of the last view's
        ///     Dispose and a finalizer calls it first; every later call returns at once.
        /// </summary>
        /// <remarks>
        ///     <c>THSTensor_dispose</c> only drops a reference (atomic counts in libtorch), so it is safe on the finalizer thread.
        ///     The booking and the count are given back in <c>finally</c>, so they balance even if the native call throws.
        /// </remarks>
        internal void Release()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0)
                return;
            try
            {
                IntPtr handle = Interlocked.Exchange(ref _handle, IntPtr.Zero);
                if (handle != IntPtr.Zero)
                    LibTorchSharpNative.THSTensor_dispose(handle);
            }
            finally
            {
                if (_bytes > 0)
                    SizeBucketedBufferPool.ReturnForeign(_bytes);
                NDArrayTorchInterop.ImportClosed();
            }
        }
    }
}
