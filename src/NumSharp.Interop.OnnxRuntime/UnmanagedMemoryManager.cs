using System;
using System.Buffers;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.ML.OnnxRuntime.Tensors;
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
    ///     NumSharp buffer for its own lifetime, and this manager keeps the HANDLE reachable
    ///     (<see cref="AttachLifetimeOwner"/>): whoever can still reach the <see cref="Memory{T}"/> or the
    ///     <c>DenseTensor</c> over it keeps the handle — and so its reference — alive, so a collection alone never
    ///     disables a tensor the caller still holds and never lets the reference count under-report who uses the
    ///     buffer (which <c>resize(refcheck: true)</c> trusts). Every <see cref="Pin"/> takes ANOTHER reference, owned by
    ///     a per-pin <see cref="PinToken"/> and released exactly once — by the pin's <c>Dispose</c>, or by the token's
    ///     finalizer when the pin is dropped undisposed (ORT's own <c>OrtValue</c> finalizer skips the pin of a value
    ///     made with <c>CreateTensorValueFromMemory</c>, which used to leak that reference for the life of the
    ///     process). Disposing the manager (the handle's <c>Dispose</c> does it) only makes it refuse NEW access:
    ///     <see cref="GetSpan"/> and <see cref="Pin"/> throw, pins taken before stay valid.</para>
    /// </summary>
    /// <typeparam name="T">The element type the memory is viewed as (the dtype's ORT type: Float16 for Half, ushort for Char).</typeparam>
    internal sealed unsafe class UnmanagedMemoryManager<T> : MemoryManager<T> where T : unmanaged
    {
        private readonly T* _pointer;
        private readonly int _length;
        private readonly IArraySlice _owner;
        private readonly bool _writeable;
        private object _lifetimeOwner;
        private int _disposed;

        /// <summary>Fronts <paramref name="length"/> elements at <paramref name="pointer"/>, which lie in <paramref name="owner"/>'s buffer.</summary>
        /// <param name="pointer">The first element (the array's first logical element, not necessarily the buffer's start).</param>
        /// <param name="length">The element count (a <see cref="Memory{T}"/> is <c>int</c>-indexed).</param>
        /// <param name="owner">
        ///     The NumSharp buffer the elements belong to; each pin takes a reference on it. The caller (the export handle)
        ///     holds its own reference for as long as this manager hands out new spans or pins.
        /// </param>
        /// <param name="writeable">
        ///     Whether the NumSharp array the memory was exported from is writeable. It does not protect the
        ///     <see cref="Memory{T}"/> (a span is always writeable); a NumSharp view built over a tensor of this memory
        ///     reads it and stays read-only when the array is.
        /// </param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="owner"/> is null.</exception>
        internal UnmanagedMemoryManager(T* pointer, int length, IArraySlice owner, bool writeable)
        {
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length));
            _pointer = pointer;
            _length = length;
            _owner = owner ?? throw new ArgumentNullException(nameof(owner));
            _writeable = writeable;
        }

        /// <summary>
        ///     Whether the NumSharp array this memory was exported from is writeable; a NumSharp view of a tensor over this
        ///     memory inherits it (<see cref="NDArrayOnnxInterop.AsNDArray{T}(DenseTensor{T}, OnnxCodecMode)"/>).
        /// </summary>
        internal bool Writeable => _writeable;

        /// <summary>The element pointer this manager fronts (stable — unmanaged memory never moves).</summary>
        internal T* Pointer => _pointer;

        /// <summary>The number of elements this manager fronts.</summary>
        internal int Length => _length;

        /// <summary>
        ///     The NumSharp buffer the memory lies in — what a NumSharp view over a tensor of this memory forwards its
        ///     reference counting to (<see cref="NDArrayOnnxInterop.AsNDArray{T}(DenseTensor{T}, OnnxCodecMode)"/>), so the view is an
        ///     ordinary NumSharp alias of the buffer rather than a foreign lease on it.
        /// </summary>
        internal IArraySlice Owner => _owner;

        /// <summary><c>true</c> once the owning handle was disposed: new spans and pins are refused from then on.</summary>
        internal bool IsDisposed => Volatile.Read(ref _disposed) != 0;

        /// <summary>
        ///     Ties the lifetime of <paramref name="owner"/> (the export handle) to this manager: the manager holds a strong
        ///     reference to it, so as long as anything can still reach the memory — the <see cref="Memory{T}"/>, the
        ///     <c>DenseTensor</c> over it — the handle is reachable too, keeps its own reference on the buffer, and its
        ///     finalizer cannot run.
        /// </summary>
        /// <remarks>
        ///     Without it the handle was collectable while its tensor was still in use: its finalizer then disposed this
        ///     manager (a GC alone turned <c>tensor.Buffer.Span</c> into <see cref="ObjectDisposedException"/>) and dropped
        ///     its reference, after which <c>resize(refcheck: true)</c> on the source saw a uniquely referenced buffer and
        ///     reallocated it from under the tensor. The cycle this creates (handle → tensor → memory → manager → handle) is
        ///     collected as a whole once nothing outside it is reachable.
        /// </remarks>
        /// <param name="owner">The handle that owns this manager's reference on the buffer.</param>
        internal void AttachLifetimeOwner(object owner) => _lifetimeOwner = owner;

        /// <inheritdoc />
        /// <exception cref="ObjectDisposedException">The owning handle has been disposed.</exception>
        public override Span<T> GetSpan()
        {
            if (Volatile.Read(ref _disposed) != 0)
                throw new ObjectDisposedException(nameof(UnmanagedMemoryManager<T>), "the NumSharp buffer behind this Memory<T> has been released (its OrtTensor handle was disposed).");
            return new Span<T>(_pointer, _length);
        }

        /// <summary>
        ///     Pins the memory: unmanaged memory needs no GC pin, so the handle carries the address and a fresh
        ///     <see cref="PinToken"/> as its <see cref="IPinnable"/> — the token owns a reference on the NumSharp buffer that
        ///     it releases exactly once, so the buffer stays alive for exactly as long as the pin does.
        /// </summary>
        /// <remarks>
        ///     The token, not the manager, is the handle's pinnable. With the manager as the pinnable every
        ///     <see cref="MemoryHandle.Dispose"/> released a reference, and <see cref="MemoryHandle"/> is a struct: two
        ///     copies of one pin disposed released TWO references, the second one someone else's (the handle's or the
        ///     source array's), so the buffer could return to the pool under a live array; and a pin never disposed — ORT's
        ///     <c>OrtValue</c> finalizer does not dispose the pin of a value made over this memory — kept its reference
        ///     forever. The token releases once whatever the copies do, and its finalizer abandons the reference of a pin
        ///     that was dropped.
        /// </remarks>
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
            PinToken token;
            try
            {
                token = new PinToken(_owner);
            }
            catch
            {
                // No token owns the reference taken above (its constructor failed, and suppressed its own finalizer), so
                // give it back here — on the caller's thread, the deterministic path.
                _owner.Release();
                throw;
            }

            return new MemoryHandle(_pointer + elementIndex, default, token);
        }

        /// <summary>
        ///     Does nothing: no <see cref="MemoryHandle"/> carries this manager as its <see cref="IPinnable"/> (each carries
        ///     its own <see cref="PinToken"/>, which releases the pin's reference), so nothing pinned through the manager
        ///     is unpinned here.
        /// </summary>
        public override void Unpin()
        {
        }

        /// <summary>
        ///     Refuses new spans and pins from now on; outstanding pins keep their references (released by their own
        ///     tokens). The manager holds no reference of its own, so there is nothing else to release.
        /// </summary>
        /// <param name="disposing">Unused: there is no managed or unmanaged state to free here.</param>
        protected override void Dispose(bool disposing)
        {
            Interlocked.Exchange(ref _disposed, 1);
        }
    }

    /// <summary>
    ///     The <see cref="IPinnable"/> of one pin of NumSharp memory (<see cref="UnmanagedMemoryManager{T}.Pin"/>): it owns
    ///     the ARC reference the pin took and gives it back exactly once — deterministically when the
    ///     <see cref="MemoryHandle"/> is disposed (<see cref="Unpin"/>), or from its finalizer when the handle was dropped
    ///     without being disposed.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why a finalizer abandons rather than releases.</b> A finalizer proves only that THIS token became
    ///     unreachable. Other aliases of the buffer that hold no counted reference (a bare <c>IArraySlice</c> or
    ///     <c>UnmanagedStorage</c> obtained from the array) may still be reachable and in use, so the finalizer must not
    ///     claim the eager free that <see cref="IArraySlice.Release"/> performs at zero. <see cref="IArraySlice.Abandon"/>
    ///     drops the count without freeing; the memory block's own finalizer frees the buffer once nothing can reach it —
    ///     the rule NumSharp's own <c>~NDArray</c> follows.</para>
    ///     <para><b>Why <see cref="Unpin"/> can abandon too.</b> Disposing the pin is not always done by a live caller. ORT's
    ///     <c>OrtValue.Dispose()</c> disposes the pin the value holds, and an OrtValue can itself be disposed BY A FINALIZER:
    ///     the memory block of an owning import view (<c>value.AsNDArray(ownsValue: true)</c>) runs its lease when the block
    ///     is finalized, and the lease disposes the value (it must — that is what frees the GC pin of a value made over a
    ///     managed array, which ORT's own finalizer skips). That dispose comes from the collection of an unreachable graph,
    ///     so it must not claim the eager free either. The token tells the two cases apart by a short weak handle to ITSELF:
    ///     the collection that finds the token unreachable clears the handle before any finalizer runs, so a cleared handle
    ///     inside <see cref="Unpin"/> means the call comes from finalization (abandon), and a live one means a holder that
    ///     still reaches the token disposed it (release, freeing at zero — the deterministic path's rule). Before this, 10 of
    ///     the 120 finalization orders of that graph freed the buffer under a reachable alias.</para>
    ///     <para><b>Thread safety.</b> Release and finalization race through one interlocked flag, so the reference is
    ///     given back once even when copies of the <see cref="MemoryHandle"/> are disposed concurrently; whoever wins the
    ///     flag also frees the weak handle, so it is freed exactly once.</para>
    /// </remarks>
    internal sealed class PinToken : IPinnable
    {
        private readonly IArraySlice _owner;

        /// <summary>
        ///     A short weak handle to this token: cleared by the collection that finds the token unreachable, before any
        ///     finalizer runs — the test <see cref="Unpin"/> uses to know whether it is being called from finalization.
        /// </summary>
        private GCHandle _self;

        private int _released;

        /// <summary>Takes over a reference the caller already holds on <paramref name="owner"/>.</summary>
        /// <param name="owner">The buffer the pin keeps alive; the caller has already taken the reference this token now owns.</param>
        /// <exception cref="OutOfMemoryException">The weak handle could not be allocated (the token then owns nothing: its finalizer is suppressed, and the caller still owns the reference).</exception>
        internal PinToken(IArraySlice owner)
        {
            _owner = owner;
            try
            {
                _self = GCHandle.Alloc(this, GCHandleType.Weak);
            }
            catch
            {
                // An object with a finalizer is registered for finalization when it is ALLOCATED, so a constructor that
                // throws still leaves its finalizer to run — which would abandon the reference the caller gives back on
                // this failure (one reference given back twice, the second one someone else's).
                GC.SuppressFinalize(this);
                throw;
            }
        }

        /// <summary><c>true</c> once the pin's reference was given back (by <see cref="Unpin"/> or the finalizer).</summary>
        internal bool IsReleased => Volatile.Read(ref _released) != 0;

        /// <summary>
        ///     Not supported: a token IS one pin, created already pinned by <see cref="UnmanagedMemoryManager{T}.Pin"/>; the BCL
        ///     never re-pins a <see cref="MemoryHandle"/>'s pinnable (it only calls <see cref="Unpin"/>).
        /// </summary>
        /// <param name="elementIndex">Ignored.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="NotSupportedException">Always.</exception>
        public MemoryHandle Pin(int elementIndex) =>
            throw new NotSupportedException("a NumSharp pin token is already one pin; pin the Memory<T> again to take another.");

        /// <summary>
        ///     Gives the pin's reference back, once: the first call releases it — freeing the buffer if it was the last
        ///     reference, the deterministic path's rule — or, when the call comes from the finalization of an unreachable
        ///     holder, abandons it (see the remarks on the class); every later call — a second copy of the same
        ///     <see cref="MemoryHandle"/> disposed — does nothing.
        /// </summary>
        public void Unpin()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0)
                return;
            GC.SuppressFinalize(this);
            // A live caller reaches this token through the call itself, so the handle still sees it; a dispose run by a
            // finalizer finds the handle already cleared by the collection that queued that finalizer.
            bool reachable = _self.Target is not null;
            // The read above must happen while this frame still keeps the token reachable: without this, the JIT may treat
            // `this` as dead once its last field was read, and a collection landing just before the handle is read would
            // clear it under a LIVE caller — turning a release into an abandon (a deferred free, never an unsafe one, but
            // not the deterministic rule).
            GC.KeepAlive(this);
            _self.Free();
            if (reachable)
                _owner.Release();
            else
                _owner.Abandon();
        }

        /// <summary>
        ///     Safety net for a pin dropped without being disposed (ORT's <c>OrtValue</c> finalizer, for one, never disposes
        ///     the pin it holds): abandons the reference so the buffer's count does not stay raised forever, without the
        ///     eager free (see the remarks on the class).
        /// </summary>
        ~PinToken()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0)
                return;
            _self.Free();
            _owner.Abandon();
        }
    }
}
