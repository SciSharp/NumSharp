using System;
using System.Buffers;
using System.Linq.Expressions;
using System.Reflection;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Interop.OnnxRuntime
{
    /// <summary>
    ///     Makes an <see cref="OrtValue"/> that ONNX Runtime allocated visible to NumSharp's memory accounting, so a value
    ///     <see cref="NDArrayOnnxInterop.ToOrtValue"/> handed out and the caller forgot is no longer invisible to the GC.
    ///     ONNX Runtime keeps OWNING and FREEING the memory - which is what keeps every native holder of the value correct
    ///     (an <c>OrtIoBinding</c>, a run's feeds: each keeps its own native reference to the tensor) - while one booking
    ///     per value tells the GC about its bytes, ends the moment the caller disposes the value, and ends from a finalizer
    ///     when the value is collected undisposed.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why NumSharp books but never frees ORT's memory.</b> ORT's native references are invisible to any
    ///     managed counter. A counter that freed the memory at zero would free it under a binding that still reads it: a
    ///     NumSharp-owned copy tied to the managed value returned another array's values in all 150,528 outputs of a bound
    ///     run once the value was disposed or dropped (P8's R2, measured 2026-11-05). So NumSharp manages the accounting and
    ///     the lifetime signals, and ORT keeps the free.</para>
    ///     <para><b>How the caller's Dispose is seen.</b> ORT's managed <see cref="OrtValue"/> has exactly one hook into its
    ///     <c>Dispose</c>: the <see cref="MemoryHandle"/> a value made over user memory keeps in its private
    ///     <c>_memHandle</c> field. <c>Dispose(true)</c> disposes it; the finalizer (<c>Dispose(false)</c>) does not. A
    ///     value from ORT's allocator leaves that slot empty, so the booking's <see cref="IPinnable"/> token is stored there
    ///     through one compiled store, built once. The field exists at the floor and at the current release (1.16.0 and
    ///     1.31.1, both measured); when a future ONNX Runtime renames or retypes it, <see cref="IsAvailable"/> is false,
    ///     <see cref="Track"/> books nothing, and the value behaves exactly as it did before bookings existed. The current-ORT
    ///     test leg asserts <see cref="IsAvailable"/>, so a rename is caught rather than silently falling back.</para>
    ///     <para><b>Cost.</b> One small finalizable token and one compiled store per value: create + dispose of a 602 KB
    ///     float32 value measured 16.8-17.2 us against 16.5-16.6 without the booking, 2.2-2.3 us against 1.8-1.9 for 1 KB.
    ///     Code that disposes its values never triggers a collection; 5,000 forgotten values of 602 KB peaked at +68 MB
    ///     (1.31.1) / +72 MB (1.16.0) instead of +2,898 MB.</para>
    /// </remarks>
    internal static class OrtValueBooking
    {
        /// <summary>The private <c>OrtValue._memHandle</c> field; null when the loaded ONNX Runtime has no <c>MemoryHandle?</c> field of that name.</summary>
        private static readonly FieldInfo MemHandleField = FindMemHandleField();

        /// <summary>Stores a <see cref="MemoryHandle"/> into <c>OrtValue._memHandle</c>; null when the field is missing.</summary>
        private static readonly Action<OrtValue, MemoryHandle?> StoreHandle = CompileStore();

        /// <summary>Reads whether <c>OrtValue._memHandle</c> holds a handle; null when the field is missing.</summary>
        private static readonly Func<OrtValue, bool> ReadHasHandle = CompileHasHandle();

        /// <summary>Values booked and not yet given back; moved only by <see cref="Token"/> (interlocked, any thread).</summary>
        private static int _liveBookings;

        /// <summary>The bytes of those values; moved only by <see cref="Token"/> (interlocked, any thread).</summary>
        private static long _bookedBytes;

        /// <summary><c>true</c> when the loaded ONNX Runtime's <see cref="OrtValue"/> has the <c>_memHandle</c> slot the booking hooks into.</summary>
        internal static bool IsAvailable => StoreHandle is not null;

        /// <summary>The number of values booked and not yet disposed or collected. A diagnostic for the tests.</summary>
        internal static int LiveBookings => Volatile.Read(ref _liveBookings);

        /// <summary>The bytes of those values. A diagnostic for the tests.</summary>
        internal static long BookedBytes => Interlocked.Read(ref _bookedBytes);

        /// <summary>
        ///     Books <paramref name="bytes"/> for <paramref name="value"/> until the caller disposes the value or the GC
        ///     collects it. Call it once, on a value this package just created from ONNX Runtime's allocator.
        /// </summary>
        /// <remarks>
        ///     The value must not already carry a <see cref="MemoryHandle"/> (a value made by
        ///     <c>CreateTensorValueFromMemory</c> does): storing over it would drop that value's pin. Every caller passes a
        ///     value from <c>CreateAllocatedTensorValue</c>, whose slot is empty.
        /// </remarks>
        /// <param name="value">A value ONNX Runtime allocated; returned unchanged.</param>
        /// <param name="bytes">The tensor's byte size. Nothing is booked for zero or less.</param>
        /// <returns><paramref name="value"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null (checked before anything is booked).</exception>
        internal static unsafe OrtValue Track(OrtValue value, long bytes)
        {
            // Refuse null before the token books: a store into a null target would throw only after the booking was made.
            ArgumentNullException.ThrowIfNull(value);
            if (StoreHandle is null || bytes <= 0)
                return value;
            var token = new Token(bytes);   // books now
            try
            {
                // ORT's Dispose(true) disposes this handle, which unpins the token: the booking ends with the caller's Dispose.
                StoreHandle(value, new MemoryHandle(null, default, token));
            }
            catch
            {
                token.End();   // never stored: give the booking back now and keep the finalizer from doing it again
                throw;
            }
            return value;
        }

        /// <summary>
        ///     Does <paramref name="value"/> carry a <see cref="MemoryHandle"/>? Such a value is either booked already (a
        ///     <see cref="NDArrayOnnxInterop.ToOrtValue"/> result: the token is that handle) or made over memory ONNX Runtime
        ///     does not own (<c>CreateTensorValueFromMemory</c>: a managed array the GC counts, NumSharp memory the pool
        ///     counts, or the caller's own memory, which disposing the value does not free). An owning import view of it must
        ///     not book its bytes again: they would be counted twice, or counted as freed with the value when they are not.
        /// </summary>
        /// <param name="value">Any OrtValue, not disposed.</param>
        /// <returns><c>true</c> when the value holds a handle; <c>false</c> when it holds none or the loaded ONNX Runtime has no such field.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        internal static bool HasMemoryHandle(OrtValue value)
        {
            ArgumentNullException.ThrowIfNull(value);
            return ReadHasHandle is not null && ReadHasHandle(value);
        }

        /// <summary>
        ///     Finds the private <c>OrtValue._memHandle</c> field, or returns null when the loaded ONNX Runtime has no
        ///     <c>MemoryHandle?</c> field of that name.
        /// </summary>
        /// <remarks>
        ///     <c>typeof(OrtValue).GetField</c> with a constant name and flags is a pattern the trimmer resolves, so a trimmed
        ///     app keeps the field (the same reason <see cref="OrtDisposalProbe"/>'s generic reader needs an annotation and this
        ///     direct lookup does not).
        /// </remarks>
        /// <returns>The field, or null.</returns>
        private static FieldInfo FindMemHandleField()
        {
            FieldInfo field = typeof(OrtValue).GetField("_memHandle", BindingFlags.Instance | BindingFlags.NonPublic);
            return field is not null && field.FieldType == typeof(MemoryHandle?) ? field : null;
        }

        /// <summary>Compiles the store into <c>OrtValue._memHandle</c>, or returns null when the field is missing.</summary>
        /// <returns>The store, or null.</returns>
        private static Action<OrtValue, MemoryHandle?> CompileStore()
        {
            if (MemHandleField is null)
                return null;
            ParameterExpression target = Expression.Parameter(typeof(OrtValue), "target");
            ParameterExpression handle = Expression.Parameter(typeof(MemoryHandle?), "handle");
            return Expression.Lambda<Action<OrtValue, MemoryHandle?>>(
                Expression.Assign(Expression.Field(target, MemHandleField), handle), target, handle).Compile();
        }

        /// <summary>Compiles the read of <c>OrtValue._memHandle.HasValue</c>, or returns null when the field is missing.</summary>
        /// <returns>The read, or null.</returns>
        private static Func<OrtValue, bool> CompileHasHandle()
        {
            if (MemHandleField is null)
                return null;
            ParameterExpression target = Expression.Parameter(typeof(OrtValue), "target");
            return Expression.Lambda<Func<OrtValue, bool>>(
                Expression.Property(Expression.Field(target, MemHandleField), nameof(Nullable<MemoryHandle>.HasValue)), target).Compile();
        }

        /// <summary>
        ///     One value's booking: the <see cref="IPinnable"/> in its <c>_memHandle</c> slot. It ends exactly once - from
        ///     <see cref="Unpin"/> when the caller disposes the value, or from the finalizer when the value is collected
        ///     undisposed - and touches only counters, so it is safe on any thread, the finalizer thread included.
        /// </summary>
        private sealed class Token : IPinnable
        {
            /// <summary>The bytes booked by the constructor; exactly this count is given back.</summary>
            private readonly long _bytes;

            /// <summary>1 once the booking ended: the interlocked once-guard shared by <see cref="End"/> and the finalizer.</summary>
            private int _ended;

            /// <summary>Books <paramref name="bytes"/> with the pool.</summary>
            /// <param name="bytes">The value's byte size (positive).</param>
            internal Token(long bytes)
            {
                _bytes = bytes;
                SizeBucketedBufferPool.BookForeign(bytes);
                Interlocked.Increment(ref _liveBookings);
                Interlocked.Add(ref _bookedBytes, bytes);
            }

            /// <summary>Not a buffer: a booking is "pinned" once, by construction, and never pinned again.</summary>
            /// <param name="elementIndex">Ignored.</param>
            /// <returns>Never returns.</returns>
            /// <exception cref="NotSupportedException">Always.</exception>
            public MemoryHandle Pin(int elementIndex) =>
                throw new NotSupportedException("a NumSharp booking token is not a buffer; it cannot be pinned.");

            /// <summary>The caller disposed the value: <c>OrtValue.Dispose(true)</c> disposed the handle carrying this token.</summary>
            public void Unpin() => End();

            /// <summary>Ends the booking once, on the calling thread: whichever of <see cref="Unpin"/>, a failed store and the finalizer comes first wins.</summary>
            internal void End()
            {
                if (Interlocked.Exchange(ref _ended, 1) != 0)
                    return;
                GC.SuppressFinalize(this);
                GiveBack();
            }

            /// <summary>
            ///     The value was collected undisposed: ONNX Runtime's own finalizer frees its memory, and this ends the booking.
            ///     Both become unreachable together (the value holds the token), so the order the two finalizers run in does
            ///     not matter: neither touches the other.
            /// </summary>
            ~Token()
            {
                if (Interlocked.Exchange(ref _ended, 1) == 0)
                    GiveBack();
            }

            /// <summary>Returns the booking to the pool and the diagnostics.</summary>
            private void GiveBack()
            {
                SizeBucketedBufferPool.ReturnForeign(_bytes);
                Interlocked.Decrement(ref _liveBookings);
                Interlocked.Add(ref _bookedBytes, -_bytes);
            }
        }
    }
}
