using System;
using System.Collections.Concurrent;
using System.Linq.Expressions;
using System.Reflection;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Fuzz.Ownership
{
    /// <summary>
    ///     Reads the reference-counting (ARC) state of the memory block behind an <see cref="IArraySlice"/>: which kind of
    ///     memory it is, how many counted references it holds, and whether it was already freed.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The state lives in <c>UnmanagedMemoryBlock&lt;T&gt;.Disposer</c>, a private nested class, so it is reached through
    ///     reflection. A renamed field fails loudly (<see cref="InvalidOperationException"/>) instead of reading as "intact":
    ///     a probe that silently returned defaults would turn every ownership check green.
    ///     </para>
    ///     <para>
    ///     A forwarding alias (<c>AllocationType.Forward</c>: <c>np.real(z)</c>, <c>view(dtype)</c>, <c>getfield</c>) counts its
    ///     references on the block that owns the memory, so the probe follows the forward chain and reports the OWNER's state:
    ///     a caller's alias array is intact exactly when that owner block is.
    ///     </para>
    ///     <para>
    ///     <b>Cost.</b> Reaching the disposer from a slice boxes two structs through <see cref="FieldInfo.GetValue"/>, so a
    ///     snapshot reads it ONCE per array (<see cref="Read"/>); every later check reads the counter and the free flag of the
    ///     cached disposer through compiled field accessors (<see cref="ReadOwner"/>, a few nanoseconds, no allocation). The
    ///     corpus sweep checks every case's arrays and the process-wide singletons after every region, so the fast path is
    ///     what keeps the ownership gate from doubling the sweep's wall time.
    ///     </para>
    /// </remarks>
    internal static class ArcProbe
    {
        /// <summary>
        ///     The ARC state of one memory block, read at one moment.
        /// </summary>
        /// <param name="Owner">The block's <c>Disposer</c> object — the identity of the memory (two arrays share memory exactly
        ///     when their states name the same owner). Null for an array without memory.</param>
        /// <param name="Kind">The allocation kind: <c>Native</c> (pooled), <c>Virtual</c>, <c>GCHandle</c> (a pinned managed
        ///     array), <c>External</c> (a caller-supplied release action: a memory map, a pinned buffer), <c>Wrap</c> (non-owning,
        ///     never freed, never counted). Null for an array without memory.</param>
        /// <param name="RefCount">The live counted references (-1 once released). Meaningless for <c>Wrap</c>.</param>
        /// <param name="Freed">True once the memory was released (pooled, freed, unpinned or unmapped).</param>
        internal readonly record struct ArcState(object Owner, string Kind, long RefCount, bool Freed)
        {
            /// <summary>
            ///     False for the non-owning <c>Wrap</c> kind (whose count is never touched) and for an array without memory:
            ///     comparing their counts before and after a call says nothing.
            /// </summary>
            public bool CountsReferences => Owner is not null && Kind != "Wrap";
        }

        /// <summary>
        ///     Per concrete slice type, the fields that reach the block's <c>Disposer</c> object (the <c>MemoryBlock</c> field
        ///     of the slice struct, then the block struct's private <c>_disposer</c>).
        /// </summary>
        private static readonly ConcurrentDictionary<Type, (FieldInfo block, FieldInfo disposer)> SliceFields = new();

        /// <summary>Per <c>Disposer</c> type (one per element type), its compiled state accessors.</summary>
        private static readonly ConcurrentDictionary<Type, DisposerAccess> DisposerAccessors = new();

        /// <summary>The compiled accessors of one <c>Disposer</c> type.</summary>
        /// <param name="Kind">Reads the allocation-kind enum as text.</param>
        /// <param name="RefCount">Reads the live-reference counter.</param>
        /// <param name="Freed">Reads the single-shot free guard (non-zero once freed).</param>
        /// <param name="Forward">Reads the owner slice of a forwarding alias.</param>
        private sealed record DisposerAccess(Func<object, string> Kind, Func<object, long> RefCount, Func<object, int> Freed,
                                             Func<object, IArraySlice> Forward);

        /// <summary>Instance fields of any visibility.</summary>
        private const BindingFlags AnyInstance = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        /// <summary>
        ///     Reads the ARC state of the memory behind <paramref name="slice"/> (following forwarding aliases to the owner).
        /// </summary>
        /// <param name="slice">An array's <c>Storage.InternalArray</c>; null reads as the default state (no memory).</param>
        /// <returns>The owner block's state.</returns>
        /// <exception cref="InvalidOperationException">The slice or its block no longer has the fields the probe reads (a
        ///     refactoring of the ARC internals: update the probe rather than let it read defaults).</exception>
        public static ArcState Read(IArraySlice slice)
        {
            // A forwarding chain is short (an alias of an alias reaches the owning block in a step or two); the bound only
            // stops a corrupted chain from looping forever.
            for (int hop = 0; hop < 16 && slice is not null; hop++)
            {
                // A string slice is a typed view over a byte slice: the byte slice's block is the memory.
                if (slice is StringArraySlice strings)
                {
                    slice = strings.Bytes;
                    continue;
                }

                object disposer = DisposerOf(slice);
                var access = AccessorsOf(disposer);
                string kind = access.Kind(disposer);
                if (kind == "Forward")
                {
                    slice = access.Forward(disposer);
                    continue;
                }

                return new ArcState(disposer, kind, access.RefCount(disposer), access.Freed(disposer) != 0);
            }

            return default;
        }

        /// <summary>
        ///     Re-reads the counter and the free flag of an owner a snapshot already found (the allocation kind never changes).
        /// </summary>
        /// <param name="before">A state <see cref="Read"/> returned.</param>
        /// <returns>The same owner's state now; the default state when <paramref name="before"/> had no memory.</returns>
        public static ArcState ReadOwner(in ArcState before)
        {
            if (before.Owner is null)
                return default;
            var access = AccessorsOf(before.Owner);
            return new ArcState(before.Owner, before.Kind, access.RefCount(before.Owner), access.Freed(before.Owner) != 0);
        }

        /// <summary>Reaches the <c>Disposer</c> object of a slice's memory block.</summary>
        /// <param name="slice">A non-string slice (a boxed <c>ArraySlice&lt;T&gt;</c>).</param>
        /// <returns>The disposer.</returns>
        /// <exception cref="InvalidOperationException">The fields moved (see <see cref="Read"/>).</exception>
        private static object DisposerOf(IArraySlice slice)
        {
            var (block, disposer) = SliceFields.GetOrAdd(slice.GetType(), static t =>
            {
                var b = t.GetField("MemoryBlock", AnyInstance)
                        ?? throw new InvalidOperationException($"{t} has no MemoryBlock field: update ArcProbe");
                var d = b.FieldType.GetField("_disposer", AnyInstance)
                        ?? throw new InvalidOperationException($"{b.FieldType} has no _disposer field: update ArcProbe");
                return (b, d);
            });

            // The slice and the block are structs: GetValue boxes copies, but the Disposer they hold is a class, so the
            // object read here IS the live disposer.
            return disposer.GetValue(block.GetValue(slice))
                   ?? throw new InvalidOperationException($"{slice.GetType()} has a null disposer");
        }

        /// <summary>Returns (building once per type) the compiled accessors of a disposer's state fields.</summary>
        /// <param name="disposer">A <c>Disposer</c> instance.</param>
        /// <returns>Its type's accessors.</returns>
        /// <exception cref="InvalidOperationException">A field moved (see <see cref="Read"/>).</exception>
        private static DisposerAccess AccessorsOf(object disposer)
            => DisposerAccessors.GetOrAdd(disposer.GetType(), static t =>
            {
                FieldInfo Field(string name) => t.GetField(name, AnyInstance)
                                                ?? throw new InvalidOperationException($"{t} has no {name} field: update ArcProbe");

                // Compiled lambdas over the private fields: expression trees compile into dynamic methods that skip
                // visibility checks, so a check reads a counter as a field load instead of a reflective, boxing GetValue.
                Func<object, TValue> Getter<TValue>(FieldInfo f)
                {
                    var o = Expression.Parameter(typeof(object), "disposer");
                    Expression body = Expression.Field(Expression.Convert(o, t), f);
                    if (body.Type != typeof(TValue))
                        body = Expression.Convert(body, typeof(TValue));
                    return Expression.Lambda<Func<object, TValue>>(body, o).Compile();
                }

                var kindField = Field("_type");
                var kindGetter = Getter<object>(kindField);
                return new DisposerAccess(o => kindGetter(o).ToString(), Getter<long>(Field("_refCount")),
                                          Getter<int>(Field("_freed")), Getter<IArraySlice>(Field("_forward")));
            });
    }
}
