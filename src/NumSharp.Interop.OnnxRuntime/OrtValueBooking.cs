using System;
using System.Buffers;
using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Interop.OnnxRuntime
{
    /// <summary>
    ///     The ARC of every <see cref="OrtValue"/> this package touches, and the memory accounting of the ones it allocates.
    ///     One <see cref="IPinnable"/> token in the value's private <c>_memHandle</c> slot - ONNX Runtime's only hook into its
    ///     own <c>Dispose</c> - does both. It books the bytes of a value <see cref="NDArrayOnnxInterop.ToOrtValue"/> handed
    ///     out, so a value the caller forgot is no longer invisible to the GC. And it makes the value's native release happen
    ///     EXACTLY ONCE, after the last NumSharp user of the value: a verb that calls ONNX Runtime on the value holds a CALL
    ///     lease for those calls, a NumSharp view of the value's memory holds a VIEW lease for its whole life, and
    ///     <c>Dispose</c> - from any thread, any number of times at once - waits for the calls in flight, then releases the
    ///     native value at once, or when the last view is gone.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why NumSharp books but never frees ORT's memory itself.</b> ORT's native references are invisible to any
    ///     managed counter. A counter that freed the memory at zero would free it under a binding that still reads it: a
    ///     NumSharp-owned copy tied to the managed value returned another array's values in all 150,528 outputs of a bound
    ///     run once the value was disposed or dropped (P8's R2, measured 2026-11-05). So NumSharp manages the accounting and
    ///     the lifetime signals, and ONNX Runtime keeps the free: a release the token takes over is still made by ONNX
    ///     Runtime's own <c>Dispose</c> code, on a shadow <see cref="OrtValue"/> that owns the native value from then on.</para>
    ///     <para><b>How the caller's Dispose is seen.</b> ORT's managed <see cref="OrtValue"/> has exactly one hook into its
    ///     <c>Dispose</c>: the <see cref="MemoryHandle"/> a value made over user memory keeps in its private
    ///     <c>_memHandle</c> field. <c>Dispose(true)</c> disposes it BEFORE it releases the native value; the finalizer
    ///     (<c>Dispose(false)</c>) does not touch it. A value from ORT's allocator leaves that slot empty, so the token is
    ///     stored there through one compiled store, built once. The field exists at the floor and at the current release
    ///     (1.16.0 and 1.31.1, both measured); when a future ONNX Runtime renames or retypes it, <see cref="IsAvailable"/> is
    ///     false, <see cref="Track"/> books nothing, and the value behaves exactly as it did before bookings existed. The
    ///     current-ORT test leg asserts <see cref="IsAvailable"/> and <see cref="CanTakeOver"/>, so a rename is caught
    ///     rather than silently falling back.</para>
    ///     <para><b>Why the release is taken over.</b> ONNX Runtime's <c>Dispose</c> takes no lock: it checks a plain
    ///     <c>_disposed</c> flag, releases the native value, then sets the flag. Measured 2026-11-07 (ORT 1.31.1, before the
    ///     takeover): two threads disposing one value both passed the check and both released it (heap corruption,
    ///     <c>0xC0000374</c>, 4 of 4 processes); a <c>Dispose</c> during <c>ToNDArray</c> or <c>AsNDArray</c> released the
    ///     value under ORT's calls (an access violation in <c>GetTensorTypeAndShape</c>, 4 of 4); and a view read after its
    ///     value's <c>Dispose</c> read whatever ORT's allocator put there next (400 of 400 reads returned another value's
    ///     data). The token's <see cref="Token.Unpin"/> runs inside <c>Dispose(true)</c> before ORT's own release: it waits
    ///     for the call leases in flight, hands the native value to the shadow, zeroes the disposed value's handle (ORT's own
    ///     release then releases nothing: <c>OrtReleaseValue(nullptr)</c> is a no-op, its argument is declared
    ///     <c>_Frees_ptr_opt_</c>) and marks it disposed. A second <c>Dispose</c> racing the first either returns at ORT's
    ///     flag or waits in its own <c>Unpin</c> until the handle is zeroed. The same races with the takeover: 0 crashes in
    ///     800 each, one native release per value, 0 wrong reads in 400.</para>
    ///     <para><b>Which values carry a token.</b> A <see cref="NDArrayOnnxInterop.ToOrtValue"/> result, the value of an
    ///     <see cref="NDArrayOnnxInterop.AsOrtValue"/> handle and a <see cref="NDArrayOnnxInterop.GetMapKeys"/> result carry
    ///     one from birth. A value ONNX Runtime allocated that a caller hands to a verb (a <c>session.Run</c> output,
    ///     <c>CreateAllocatedTensorValue</c>, an element read with <c>GetValue</c>) is ADOPTED by its first verb: its empty
    ///     slot receives a token then. Never adopted (they keep ONNX Runtime's own rules): a value over memory ORT does not
    ///     own (its slot already holds that memory's <see cref="MemoryHandle"/>), a value that does not own its native
    ///     value (<c>_owned == false</c>), and a value built from member values (<c>CreateSequence</c> / <c>CreateMap</c>:
    ///     ORT's <c>Dispose</c> disposes the members AFTER the hook, which no token could make exactly-once). The bridge's
    ///     own transient values (the <c>Run</c> tier's outputs, the elements a sequence or map reader takes) are not
    ///     adopted either: nothing outside the call can reach them.</para>
    ///     <para><b>What stays unprotected.</b> ONNX Runtime's own calls on a value take no lease (a <c>session.Run</c> that
    ///     reads it, <c>GetTensorDataAsSpan</c>, an <c>OrtIoBinding</c>): dispose a value only after those returned. And
    ///     adoption is a plain store into ORT's field, so a value another thread disposes DURING its first verb is not
    ///     covered: give a value that crosses threads its first verb on the thread that created it.</para>
    ///     <para><b>Cost.</b> One small finalizable token and one compiled store per value: create + dispose of a 602 KB
    ///     float32 value measured 16.8-17.2 us against 16.5-16.6 without the booking, 2.2-2.3 us against 1.8-1.9 for 1 KB.
    ///     Code that disposes its values never triggers a collection; 5,000 forgotten values of 602 KB peaked at +68 MB
    ///     (1.31.1) / +72 MB (1.16.0) instead of +2,898 MB. A lease is two interlocked operations and a table lookup:
    ///     <c>ToNDArray</c> of 16 KB measured +17..+58 ns on 1.83-1.85 us, <c>AsNDArray</c> -6..-13 ns on 1.62-1.64 us
    ///     (within noise; 2026-11-07, two pinned CPUs).</para>
    /// </remarks>
    internal static class OrtValueBooking
    {
        /// <summary>The private <c>OrtValue._memHandle</c> field; null when the loaded ONNX Runtime has no <c>MemoryHandle?</c> field of that name.</summary>
        private static readonly FieldInfo MemHandleField = FindMemHandleField();

        /// <summary>Stores a <see cref="MemoryHandle"/> into <c>OrtValue._memHandle</c>; null when the field is missing.</summary>
        private static readonly Action<OrtValue, MemoryHandle?> StoreHandle = CompileStore();

        /// <summary>Reads whether <c>OrtValue._memHandle</c> holds a handle; null when the field is missing.</summary>
        private static readonly Func<OrtValue, bool> ReadHasHandle = CompileHasHandle();

        /// <summary>Reads <c>OrtValue._handle</c>, the native value; null when the loaded ONNX Runtime has no <c>IntPtr</c> field of that name.</summary>
        private static readonly Func<OrtValue, IntPtr> ReadNative = CompileFieldRead<IntPtr>("_handle");

        /// <summary>Writes <c>OrtValue._handle</c>: the takeover zeroes it so ORT's own release releases nothing. Null when the field is missing.</summary>
        private static readonly Action<OrtValue, IntPtr> WriteNative = CompileFieldWrite<IntPtr>("_handle");

        /// <summary>Writes <c>OrtValue._disposed</c>: the takeover sets it so a later <c>Dispose</c> returns at ORT's own check. Null when the field is missing.</summary>
        private static readonly Action<OrtValue, bool> WriteDisposed = CompileFieldWrite<bool>("_disposed");

        /// <summary>
        ///     Reads <c>OrtValue._owned</c> (1.31.1); null on a version without it (1.16.0), where every value owns its native
        ///     value. A value that does not own its native value is never adopted: its release is not ORT's managed value's to make.
        /// </summary>
        private static readonly Func<OrtValue, bool> ReadOwned = CompileFieldRead<bool>("_owned");

        /// <summary>
        ///     Reads whether <c>OrtValue._compositeMembers</c> is set (a value built by <c>CreateSequence</c> /
        ///     <c>CreateMap</c>); null when the field is missing, and then only tensors are adopted (a tensor never has members).
        /// </summary>
        private static readonly Func<OrtValue, bool> ReadHasMembers = CompileHasMembers();

        /// <summary>
        ///     Makes a new managed <see cref="OrtValue"/> over a native value through ORT's internal
        ///     <c>(IntPtr, OnnxValueType)</c> constructor, the same at 1.16.0 and 1.31.1 and free of native calls: the shadow
        ///     that owns a native value once a takeover moved it out of the disposed value. Null when the constructor is missing.
        /// </summary>
        private static readonly Func<IntPtr, OnnxValueType, OrtValue> NewShadow = CompileShadowFactory();

        /// <summary>
        ///     The token of every value that carries one. <see cref="MemoryHandle"/> keeps its <see cref="IPinnable"/>
        ///     private, so this is how a verb finds a value's token; the table holds the token exactly as long as the value is
        ///     reachable and keeps neither alive (the token references the value, which the table does not count as a root).
        /// </summary>
        private static readonly ConditionalWeakTable<OrtValue, Token> Tokens = new ConditionalWeakTable<OrtValue, Token>();

        /// <summary>Values booked and not yet given back; moved only by <see cref="Token"/> (interlocked, any thread).</summary>
        private static int _liveBookings;

        /// <summary>The bytes of those values; moved only by <see cref="Token"/> (interlocked, any thread).</summary>
        private static long _bookedBytes;

        /// <summary>Native values released through a token's shadow; moved only by <see cref="Token"/> (interlocked, any thread).</summary>
        private static long _takenOverReleases;

        /// <summary><c>true</c> when the loaded ONNX Runtime's <see cref="OrtValue"/> has the <c>_memHandle</c> slot the booking hooks into.</summary>
        internal static bool IsAvailable => StoreHandle is not null;

        /// <summary>
        ///     <c>true</c> when the loaded ONNX Runtime has every member the takeover needs (<c>_memHandle</c>, <c>_handle</c>,
        ///     <c>_disposed</c> and the internal constructor). Without them a token only books, and every verb behaves as it
        ///     did before leases existed: <see cref="Enter"/> returns an empty lease after the disposal checks.
        /// </summary>
        internal static bool CanTakeOver =>
            StoreHandle is not null && ReadHasHandle is not null && ReadNative is not null && WriteNative is not null && WriteDisposed is not null && NewShadow is not null;

        /// <summary>The number of values booked and not yet given back. A diagnostic for the tests.</summary>
        internal static int LiveBookings => Volatile.Read(ref _liveBookings);

        /// <summary>The bytes of those values. A diagnostic for the tests.</summary>
        internal static long BookedBytes => Interlocked.Read(ref _bookedBytes);

        /// <summary>
        ///     How many native values a token's shadow has released since the process started: exactly one per disposed value
        ///     that carried a token. A diagnostic for the tests (a race that released a value twice would count it twice, or
        ///     corrupt the heap first).
        /// </summary>
        internal static long TakenOverReleases => Interlocked.Read(ref _takenOverReleases);

        /// <summary>Does <paramref name="value"/> carry a token (from birth, or adopted by a verb)? A diagnostic for the tests.</summary>
        /// <param name="value">Any OrtValue.</param>
        /// <returns><c>true</c> when the value has a token.</returns>
        internal static bool HasToken(OrtValue value) => value is not null && Tokens.TryGetValue(value, out _);

        /// <summary>
        ///     Gives <paramref name="value"/> its token: the ARC described on the class and, for <paramref name="bytes"/> &gt;
        ///     0, a booking until the native value is released (or the value is collected undisposed). Call it once, on a
        ///     value this package just created, before anything else can reach it.
        /// </summary>
        /// <remarks>
        ///     The value must not already carry a <see cref="MemoryHandle"/> (a value made by
        ///     <c>CreateTensorValueFromMemory</c> does): storing over it would drop that value's pin. Every caller passes a value
        ///     from <c>CreateAllocatedTensorValue</c>, <c>CreateTensorValueWithData</c>, <c>CreateTensorWithEmptyStrings</c>
        ///     or <c>GetValue</c>, whose slot is empty.
        /// </remarks>
        /// <param name="value">A value this package just created; returned unchanged.</param>
        /// <param name="bytes">The bytes to book (an ORT-allocated tensor's byte size); 0 or less books nothing (a value over NumSharp memory, a string tensor, an empty tensor).</param>
        /// <returns><paramref name="value"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null (checked before anything is booked).</exception>
        internal static unsafe OrtValue Track(OrtValue value, long bytes)
        {
            // Refuse null before the token books: a store into a null target would throw only after the booking was made.
            ArgumentNullException.ThrowIfNull(value);
            if (StoreHandle is null)
                return value;
            if (bytes <= 0 && !CanTakeOver)
                return value;   // nothing to book and no takeover to arm: the token would do nothing
            var token = new Token(value, bytes > 0 ? bytes : 0);   // books now
            try
            {
                // The table first, so a verb that finds the token in the slot's place can always find it here too.
                Tokens.AddOrUpdate(value, token);
                // ORT's Dispose(true) disposes this handle, which unpins the token: the takeover and the end of the booking.
                StoreHandle(value, new MemoryHandle(null, default, token));
            }
            catch
            {
                // Never stored (out of memory in the table or the store): give the booking back now, keep the finalizer from
                // doing it again, and leave the value as ORT made it.
                Tokens.Remove(value);
                token.End();
                throw;
            }
            return value;
        }

        /// <summary>
        ///     Opens a CALL lease on <paramref name="value"/> for a verb that is about to call ONNX Runtime on it: until the
        ///     lease is disposed, a <c>Dispose</c> on any thread waits before the native value is released. Every verb over an
        ///     <see cref="OrtValue"/> opens one before its first native call, in place of the disposal checks it made before.
        /// </summary>
        /// <remarks>
        ///     <para>The disposal checks come first, as before: the value of a disposed <see cref="OrtTensor"/> handle, or a
        ///     value ONNX Runtime has disposed, is refused with <see cref="ObjectDisposedException"/> before anything reaches
        ///     native code.</para>
        ///     <para>A value without a token is adopted when <paramref name="adopt"/> is <c>true</c> and the value qualifies
        ///     (see the class remarks); one that does not qualify gets an empty lease, which guards nothing beyond the checks -
        ///     exactly the behaviour before leases existed.</para>
        ///     <para>Never call <c>Dispose</c> on the value while holding its lease: the dispose waits for every call lease,
        ///     the caller's own included, so it would wait forever. The verbs that own their value dispose it after the
        ///     lease ends.</para>
        /// </remarks>
        /// <param name="value">The value the verb reads (not null).</param>
        /// <param name="verb">The public verb, named in a refusal.</param>
        /// <param name="adopt"><c>true</c> for a value a caller handed over (adopt it when it has no token yet); <c>false</c> for the bridge's own transient values, which nothing else can reach.</param>
        /// <returns>The lease; dispose it (a <c>using</c>) once the verb's last native call and its last read of the value's memory are done.</returns>
        /// <exception cref="ObjectDisposedException">The value belongs to a disposed <see cref="OrtTensor"/> handle, has been disposed, or is being disposed by another thread.</exception>
        internal static Lease Enter(OrtValue value, string verb, bool adopt)
        {
            NDArrayOnnxInterop.EnsureNotOfADisposedHandle(value, verb);
            if (!CanTakeOver)
                return default;
            if (!Tokens.TryGetValue(value, out Token token))
            {
                token = adopt ? Adopt(value, verb) : null;
                if (token is null)
                    return default;
            }
            if (!token.TryEnterCall())
                throw DisposedDuring(verb);
            return new Lease(token);
        }

        /// <summary>
        ///     Are <paramref name="value"/>'s bytes already counted by someone else, so an owning import view must not book them
        ///     again? Yes for a value whose token books them (a <see cref="NDArrayOnnxInterop.ToOrtValue"/> result), and for a
        ///     value made over memory ORT does not own (<c>CreateTensorValueFromMemory</c>: a managed array the GC counts,
        ///     NumSharp memory the pool counts, or the caller's own memory, which disposing the value does not free). No for a
        ///     value ORT allocated whose token only leases (an adopted <c>session.Run</c> output): its owning view books it.
        /// </summary>
        /// <param name="value">Any OrtValue, not disposed.</param>
        /// <returns><c>true</c> when the bytes are counted elsewhere.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        internal static bool IsAccountedElsewhere(OrtValue value)
        {
            ArgumentNullException.ThrowIfNull(value);
            if (Tokens.TryGetValue(value, out Token token))
                return token.Bytes > 0;
            return HasMemoryHandle(value);
        }

        /// <summary>
        ///     Does <paramref name="value"/> carry a <see cref="MemoryHandle"/>? Such a value carries a token already (the
        ///     token is that handle) or is made over memory ONNX Runtime does not own (<c>CreateTensorValueFromMemory</c>).
        /// </summary>
        /// <param name="value">Any OrtValue, not disposed.</param>
        /// <returns><c>true</c> when the value holds a handle; <c>false</c> when it holds none or the loaded ONNX Runtime has no such field.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        internal static bool HasMemoryHandle(OrtValue value)
        {
            ArgumentNullException.ThrowIfNull(value);
            return ReadHasHandle is not null && ReadHasHandle(value);
        }

        /// <summary>The refusal of a verb whose value another thread is disposing: its lease could not be opened.</summary>
        /// <param name="verb">The public verb.</param>
        /// <returns>The exception to throw.</returns>
        internal static ObjectDisposedException DisposedDuring(string verb) =>
            new ObjectDisposedException(nameof(OrtValue),
                $"{verb}: this OrtValue is being disposed by another thread. The calls and the views already reading it finish first, " +
                "and ONNX Runtime releases it once, after the last of them; a new call is refused. Read or view the value before it is disposed.");

        /// <summary>
        ///     Gives a value a caller handed to a verb its token, when it qualifies (see the class remarks). Two first verbs at
        ///     once publish ONE token: the table decides, and only the thread whose token won stores it into the slot.
        /// </summary>
        /// <param name="value">A value with no token yet.</param>
        /// <param name="verb">The verb that adopts it, named in a refusal.</param>
        /// <returns>The value's token, or null when the value does not qualify.</returns>
        /// <exception cref="ObjectDisposedException">Another thread disposed the value while it was being adopted (when it could still be told).</exception>
        private static unsafe Token Adopt(OrtValue value, string verb)
        {
            // Over memory ORT does not own: the slot holds that memory's pin, which a token must not replace.
            if (ReadHasHandle(value))
                return null;
            // Not the owner of its native value (1.31.1): ORT's own Dispose releases nothing for it, and neither may a token.
            if (ReadOwned is not null && !ReadOwned(value))
                return null;
            // Built from members: ORT's Dispose disposes them after the hook, which a racing second Dispose would repeat.
            if (ReadHasMembers is null ? value.OnnxType != OnnxValueType.ONNX_TYPE_TENSOR : ReadHasMembers(value))
                return null;

            var candidate = new Token(value, 0);
            Token token = Tokens.GetValue(value, _ => candidate);
            if (!ReferenceEquals(token, candidate))
            {
                // Another first verb won: its token is the value's. This one was never stored and booked nothing.
                candidate.End();
                return token;
            }

            StoreHandle(value, new MemoryHandle(null, default, token));
            // The store before the re-check: a Dispose that read the slot before it released the value as before adoption, and
            // when it has already marked the value disposed, the verb must not call ORT on it.
            Thread.MemoryBarrier();
            if (OrtDisposalProbe.IsDisposed(value))
                throw DisposedDuring(verb);
            return token;
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
        ///     Compiles the read of one private instance field of <see cref="OrtValue"/>, or returns null when the loaded ONNX
        ///     Runtime has no field of that name and type (the takeover is then unavailable, never wrong).
        /// </summary>
        /// <typeparam name="T">The field's type.</typeparam>
        /// <param name="name">The field's name.</param>
        /// <returns>The read, or null.</returns>
        private static Func<OrtValue, T> CompileFieldRead<T>(string name)
        {
            FieldInfo field = typeof(OrtValue).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
            if (field is null || field.FieldType != typeof(T))
                return null;
            ParameterExpression target = Expression.Parameter(typeof(OrtValue), "target");
            return Expression.Lambda<Func<OrtValue, T>>(Expression.Field(target, field), target).Compile();
        }

        /// <summary>Compiles the write of one private instance field of <see cref="OrtValue"/>, or returns null when it is missing (see <see cref="CompileFieldRead{T}"/>).</summary>
        /// <typeparam name="T">The field's type.</typeparam>
        /// <param name="name">The field's name.</param>
        /// <returns>The write, or null.</returns>
        private static Action<OrtValue, T> CompileFieldWrite<T>(string name)
        {
            FieldInfo field = typeof(OrtValue).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
            if (field is null || field.FieldType != typeof(T))
                return null;
            ParameterExpression target = Expression.Parameter(typeof(OrtValue), "target");
            ParameterExpression value = Expression.Parameter(typeof(T), "value");
            return Expression.Lambda<Action<OrtValue, T>>(Expression.Assign(Expression.Field(target, field), value), target, value).Compile();
        }

        /// <summary>Compiles <c>value._compositeMembers != null</c>, or returns null when the field is missing or not a reference.</summary>
        /// <returns>The read, or null.</returns>
        private static Func<OrtValue, bool> CompileHasMembers()
        {
            FieldInfo field = typeof(OrtValue).GetField("_compositeMembers", BindingFlags.Instance | BindingFlags.NonPublic);
            if (field is null || field.FieldType.IsValueType)
                return null;
            ParameterExpression target = Expression.Parameter(typeof(OrtValue), "target");
            return Expression.Lambda<Func<OrtValue, bool>>(
                Expression.NotEqual(Expression.Field(target, field), Expression.Constant(null, field.FieldType)), target).Compile();
        }

        /// <summary>Compiles ORT's internal <c>OrtValue(IntPtr, OnnxValueType)</c> constructor, or returns null when it is missing.</summary>
        /// <remarks>
        ///     That constructor only stores its arguments (no native call, so it cannot fail on a valid handle), and its
        ///     object's <c>Dispose</c> is ORT's own: the shadow releases the native value exactly as the disposed value would
        ///     have, with ORT's <c>_owned</c> default (true) at 1.31.1 - only owned values are adopted or tracked.
        /// </remarks>
        /// <returns>The factory, or null.</returns>
        private static Func<IntPtr, OnnxValueType, OrtValue> CompileShadowFactory()
        {
            ConstructorInfo constructor = typeof(OrtValue).GetConstructor(BindingFlags.Instance | BindingFlags.NonPublic, binder: null,
                new[] { typeof(IntPtr), typeof(OnnxValueType) }, modifiers: null);
            if (constructor is null)
                return null;
            ParameterExpression native = Expression.Parameter(typeof(IntPtr), "native");
            ParameterExpression kind = Expression.Parameter(typeof(OnnxValueType), "kind");
            return Expression.Lambda<Func<IntPtr, OnnxValueType, OrtValue>>(Expression.New(constructor, native, kind), native, kind).Compile();
        }

        /// <summary>
        ///     A verb's call lease (see <see cref="Enter"/>): a struct, so a <c>using</c> costs no allocation. The default lease
        ///     (a value without a token) guards nothing.
        /// </summary>
        internal readonly struct Lease : IDisposable
        {
            /// <summary>The value's token, or null for a value that carries none (ONNX Runtime's own rules apply to it).</summary>
            internal readonly Token Token;

            /// <summary>Wraps a call lease <paramref name="token"/> already counted.</summary>
            /// <param name="token">The token whose call count this lease holds.</param>
            internal Lease(Token token) => Token = token;

            /// <summary>Ends the call: a <c>Dispose</c> waiting for it may now take the release over.</summary>
            public void Dispose() => Token?.ExitCall();
        }

        /// <summary>
        ///     One value's ARC and booking: the <see cref="IPinnable"/> in its <c>_memHandle</c> slot. It counts the calls in
        ///     flight and the views alive, takes the native release over when the caller disposes the value, and makes that
        ///     release once, after the last view. The booking ends with the native release, or from the finalizer when the
        ///     value is collected undisposed. Every member is safe on any thread, the finalizer thread included.
        /// </summary>
        internal sealed class Token : IPinnable
        {
            /// <summary>The value this token belongs to: the takeover reads and writes its fields. The value holds the token (its slot), so the two live and die together.</summary>
            private readonly OrtValue _value;

            /// <summary>The bytes booked by the constructor; exactly this count is given back. 0 for a token that only leases.</summary>
            private readonly long _bytes;

            /// <summary>Verbs inside ONNX Runtime calls on the value right now; the takeover waits for 0.</summary>
            private int _calls;

            /// <summary>NumSharp views over the value's memory; the native release waits for 0.</summary>
            private int _views;

            /// <summary>1 once a <c>Dispose</c> began: no new lease of either kind is granted from then on.</summary>
            private int _closing;

            /// <summary>1 once the takeover finished: the handle is zeroed and the shadow (if any) owns the native value.</summary>
            private int _handedOver;

            /// <summary>1 once the native value was released: the once-guard of <see cref="FinalRelease"/>.</summary>
            private int _released;

            /// <summary>1 once the booking ended: the once-guard shared by <see cref="End"/> and the finalizer.</summary>
            private int _ended;

            /// <summary>The bytes a view reported for the value's memory: what a release deferred past the caller's Dispose keeps alive.</summary>
            private long _viewBytes;

            /// <summary>The bytes booked for views that outlived the value's Dispose (an adopted value books nothing of its own); given back at the release.</summary>
            private long _deferredBooking;

            /// <summary>The value's native value after the takeover, owned by ORT's own managed type; disposed by <see cref="FinalRelease"/>.</summary>
            private OrtValue _shadow;

            /// <summary>Creates the token of <paramref name="value"/> and books <paramref name="bytes"/> with the pool.</summary>
            /// <param name="value">The value the token is stored into.</param>
            /// <param name="bytes">The value's byte size to book; 0 books nothing.</param>
            internal Token(OrtValue value, long bytes)
            {
                _value = value;
                _bytes = bytes;
                if (bytes <= 0)
                    return;
                SizeBucketedBufferPool.BookForeign(bytes);
                Interlocked.Increment(ref _liveBookings);
                Interlocked.Add(ref _bookedBytes, bytes);
            }

            /// <summary>The bytes this token books (0 for a token that only leases).</summary>
            internal long Bytes => _bytes;

            /// <summary>
            ///     Opens a call: fails once a <c>Dispose</c> began. The increment comes before the read of the closing flag and
            ///     <see cref="Unpin"/> sets the flag before it reads the count, both with full fences, so either the call sees
            ///     the flag or the dispose sees the call - never neither.
            /// </summary>
            /// <returns><c>true</c> when the call may proceed.</returns>
            internal bool TryEnterCall()
            {
                Interlocked.Increment(ref _calls);
                if (Volatile.Read(ref _closing) == 0)
                    return true;
                Interlocked.Decrement(ref _calls);
                return false;
            }

            /// <summary>Ends a call opened by <see cref="TryEnterCall"/>.</summary>
            internal void ExitCall() => Interlocked.Decrement(ref _calls);

            /// <summary>
            ///     Opens a view lease, from inside a call lease (so a racing <c>Dispose</c> is still waiting for the call and
            ///     sees the view when it decides whether to release at once). Fails once a <c>Dispose</c> began.
            /// </summary>
            /// <param name="bytes">The byte size of the memory the view reads, booked if the release is deferred past the caller's Dispose.</param>
            /// <returns><c>true</c> when the view may be handed out; it must call <see cref="ReleaseView"/> exactly once when it is gone.</returns>
            internal bool TryAddView(long bytes)
            {
                Interlocked.CompareExchange(ref _viewBytes, bytes, 0);
                Interlocked.Increment(ref _views);
                if (Volatile.Read(ref _closing) == 0)
                    return true;
                ReleaseView();
                return false;
            }

            /// <summary>
            ///     Ends a view lease: the last one after the takeover releases the native value. The decrement is a full fence
            ///     and the takeover publishes <see cref="_handedOver"/> with one before it reads the view count, so the last
            ///     view and the dispose cannot both miss each other.
            /// </summary>
            internal void ReleaseView()
            {
                if (Interlocked.Decrement(ref _views) == 0 && Volatile.Read(ref _handedOver) != 0)
                    FinalRelease();
            }

            /// <summary>Not a buffer: a token is "pinned" once, by construction, and never pinned again.</summary>
            /// <param name="elementIndex">Ignored.</param>
            /// <returns>Never returns.</returns>
            /// <exception cref="NotSupportedException">Always.</exception>
            public MemoryHandle Pin(int elementIndex) =>
                throw new NotSupportedException("a NumSharp booking token is not a buffer; it cannot be pinned.");

            /// <summary>
            ///     The caller disposed the value: <c>OrtValue.Dispose(true)</c> disposed the handle carrying this token, BEFORE
            ///     its own native release. Takes that release over (see the class remarks): the first <c>Dispose</c> waits for
            ///     the calls in flight, moves the native value to a shadow, zeroes the value's handle and marks it disposed, then
            ///     releases at once when no view is alive; a racing second <c>Dispose</c> returns only once the handle is zeroed,
            ///     so its own ORT code releases nothing.
            /// </summary>
            /// <remarks>
            ///     Without the takeover members (an ONNX Runtime that renamed them) only the booking ends, as before leases
            ///     existed. If the shadow cannot be made (out of memory), the native release stays ORT's own, exactly as without
            ///     a token; the waiters are released either way.
            /// </remarks>
            public void Unpin()
            {
                if (!CanTakeOver)
                {
                    End();
                    return;
                }
                if (Interlocked.Exchange(ref _closing, 1) != 0)
                {
                    // A Dispose racing the first: ORT's code reads the handle right after this returns, so wait for the zero.
                    SpinWait.SpinUntil(() => Volatile.Read(ref _handedOver) != 0);
                    return;
                }

                // The calls already inside ORT finish first: the handle must not change under them (microseconds - metadata
                // calls and, for ToNDArray, one copy).
                SpinWait.SpinUntil(() => Volatile.Read(ref _calls) == 0);
                try
                {
                    IntPtr native = ReadNative(_value);
                    // The shadow exists before the handle is zeroed: a failure leaves the value exactly as ORT expects it.
                    if (native != IntPtr.Zero)
                        _shadow = NewShadow(native, _value.OnnxType);
                    WriteNative(_value, IntPtr.Zero);   // ORT's own release becomes OrtReleaseValue(nullptr): a no-op
                    WriteDisposed(_value, true);        // a later Dispose, and every disposal check, see a disposed value at once
                }
                catch (Exception)
                {
                    // No shadow (out of memory; or a value kind ORT's constructor refuses, which an owned value never has):
                    // the handle was not zeroed, so ORT's own code releases the value after this returns, as it did before
                    // the takeover existed. Nothing may escape: ORT's Dispose would throw before releasing anything, and a
                    // racing second Dispose would wait for the hand-over forever. A shadow made before a later step failed
                    // must not release a native value the disposed value still owns.
                    if (_shadow is not null && ReadNative(_value) != IntPtr.Zero)
                    {
                        GC.SuppressFinalize(_shadow);
                        _shadow = null;
                    }
                }

                // An adopted value books nothing of its own; memory its views keep alive past the caller's Dispose is booked
                // until the release, so a forgotten view of a disposed value is not invisible to the GC.
                if (_bytes == 0 && _shadow is not null && Volatile.Read(ref _views) > 0)
                    BookDeferred();
                // A full fence between publishing the hand-over and reading the view count (see ReleaseView).
                Interlocked.Exchange(ref _handedOver, 1);
                if (Volatile.Read(ref _views) == 0)
                    FinalRelease();
            }

            /// <summary>
            ///     Releases the native value once, through the shadow (ONNX Runtime's own <c>Dispose</c> code), then gives back
            ///     the deferred booking and the value's booking. Called by the takeover when no view is alive, or by the last
            ///     view's release - any thread, the finalizer's included.
            /// </summary>
            private void FinalRelease()
            {
                if (Interlocked.Exchange(ref _released, 1) != 0)
                    return;
                try
                {
                    OrtValue shadow = Interlocked.Exchange(ref _shadow, null);
                    if (shadow is not null)
                    {
                        shadow.Dispose();
                        Interlocked.Increment(ref _takenOverReleases);
                    }
                }
                finally
                {
                    GiveBackDeferred();
                    End();
                }
            }

            /// <summary>Books the bytes the views keep alive past the caller's Dispose (an adopted value's memory).</summary>
            private void BookDeferred()
            {
                long bytes = Interlocked.Read(ref _viewBytes);
                if (bytes <= 0 || Interlocked.CompareExchange(ref _deferredBooking, bytes, 0) != 0)
                    return;
                SizeBucketedBufferPool.BookForeign(bytes);
                NDArrayOnnxInterop.ImportPressureChanged(bytes);
            }

            /// <summary>Gives the deferred booking back, once: counters only, so safe on the finalizer thread.</summary>
            private void GiveBackDeferred()
            {
                long bytes = Interlocked.Exchange(ref _deferredBooking, 0);
                if (bytes <= 0)
                    return;
                SizeBucketedBufferPool.ReturnForeign(bytes);
                NDArrayOnnxInterop.ImportPressureChanged(-bytes);
            }

            /// <summary>Ends the booking once, on the calling thread: whichever of the release, a failed store and the finalizer comes first wins.</summary>
            internal void End()
            {
                if (Interlocked.Exchange(ref _ended, 1) != 0)
                    return;
                GC.SuppressFinalize(this);
                if (_bytes > 0)
                    GiveBack();
            }

            /// <summary>
            ///     The value was collected undisposed (ONNX Runtime's own finalizer frees its memory), or every holder of a
            ///     handed-over value is gone (the shadow's own finalizer frees it): this ends the bookings. The token, the value,
            ///     the shadow and the views become unreachable together, so the order their finalizers run in does not matter:
            ///     each step here and in ORT's <c>Dispose</c> is once-guarded, and no view can read the memory any more.
            /// </summary>
            ~Token()
            {
                if (Interlocked.Exchange(ref _ended, 1) == 0 && _bytes > 0)
                    GiveBack();
                GiveBackDeferred();
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
