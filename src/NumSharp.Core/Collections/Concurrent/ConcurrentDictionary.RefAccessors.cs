// NumSharp-only extension of the vendored ConcurrentDictionary clone (ConcurrentDictionary.cs).
// The vendored file stays byte-faithful to dotnet/runtime apart from the single `partial` keyword; every
// NumSharp-specific member lives here so a future re-sync against the upstream source is a plain file replace.

#nullable enable

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace NumSharp.Collections.Concurrent
{
    public partial class ConcurrentDictionary<TKey, TValue>
    {
        /// <summary>
        ///     Looks up <paramref name="key" /> and returns a <b>direct managed reference to the value field of the
        ///     live hash-table node</b> (the <see cref="System.Runtime.InteropServices.CollectionsMarshal" /> naming
        ///     convention) — the seam that lets <see cref="ConcurrentOrderedDictionary{TKey,TValue}" />
        ///     read one field of a struct value without copying the whole struct, and mutate a single
        ///     atomically-writable field in place (zero allocation) instead of paying the node-replacing update the
        ///     rest of the public API performs for non-atomic value types.
        /// </summary>
        /// <param name="key">The key to locate; hashed and compared exactly like <see cref="TryGetValue" />.</param>
        /// <returns>
        ///     A reference into the node holding the key's value, or <see cref="Unsafe.NullRef{T}" /> when the key
        ///     is absent. The caller MUST test with <see cref="Unsafe.IsNullRef{T}(ref readonly T)" /> before touching it —
        ///     dereferencing the null ref crashes the process. (A null-ref sentinel instead of an <c>out bool</c>
        ///     on purpose: the sentinel test is a register compare, while an <c>out bool</c> forces a stack
        ///     round-trip per call on this very hot path — measured.)
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
        /// <remarks>
        ///     <para>
        ///         <b>This deliberately bypasses the tear-free discipline the rest of the public surface
        ///         guarantees</b>, so every caller — this member is public — inherits three obligations:
        ///     </para>
        ///     <para>
        ///         1. <b>Lifetime.</b> The reference points into a <c>Node</c>. Any mutation of this dictionary —
        ///         an update of this key (which replaces the node when <typeparamref name="TValue" /> is not
        ///         atomically writable), a remove, a <see cref="Clear" />, or a table growth triggered by any add —
        ///         can strand it on a dead node, silently discarding writes. Acquire it, use it, and drop it inside
        ///         one externally-serialized critical section; never cache it across operations.
        ///     </para>
        ///     <para>
        ///         2. <b>Tearing.</b> Lock-free readers (<see cref="TryGetValue" />, enumeration) copy the whole
        ///         value with plain loads concurrently with writes made through this reference. A write is only safe
        ///         when it targets a single field that the ECMA-335 §12.6.6 rules make atomic at its natural
        ///         alignment (an object reference, or a primitive no wider than the machine word) — then a
        ///         concurrent copy sees the old or the new field value, never a hybrid. Writing the whole struct, or
        ///         any field wider than the word size, through this reference can be observed torn.
        ///     </para>
        ///     <para>
        ///         3. <b>Write serialization.</b> Two threads writing through references to the same node race
        ///         exactly like two unsynchronized field writes. The caller must hold its own write lock (the
        ///         ordered dictionary holds its global write lock for every use).
        ///     </para>
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] // hot seam: inlining folds the bucket walk into the ordered dict's readers, matching the BCL's one-call depth
        public ref TValue GetValueRefOrNullRef(TKey key)
        {
            // typeof guard: a bare `key is null` on generic TKey is box+compare IL, and unoptimized (Debug)
            // codegen executes the box for value-type keys — 24 B per lookup. The short-circuit keeps the box
            // unreached for value types (boxing a reference type is a no-op), so the seam is allocation-free in
            // every configuration; Release codegen is identical (the JIT folds the guard).
            if (!typeof(TKey).IsValueType && key is null)
            {
                ThrowHelper.ThrowArgumentNullException(ExceptionArgument.key);
            }

            // Mirror of TryGetValue's lookup, returning `ref n._value` instead of copying it. The _tables local,
            // the value-type/comparer branch split and the hashcode-first comparison are kept identical to the
            // vendored method so the JIT specializes both the same way.
            Tables tables = _tables;

            IEqualityComparer<TKey>? comparer = tables._comparer;
            if (typeof(TKey).IsValueType && // comparer can only be null for value types; enable JIT to eliminate entire if block for ref types
                comparer is null)
            {
                int hashcode = key.GetHashCode();
                for (Node? n = GetBucket(tables, hashcode); n is not null; n = n._next)
                {
                    if (hashcode == n._hashcode && EqualityComparer<TKey>.Default.Equals(n._key, key))
                    {
                        return ref n._value;
                    }
                }
            }
            else
            {
                int hashcode = GetHashCode(comparer, key);
                for (Node? n = GetBucket(tables, hashcode); n is not null; n = n._next)
                {
                    if (hashcode == n._hashcode && comparer!.Equals(n._key, key))
                    {
                        return ref n._value;
                    }
                }
            }

            return ref Unsafe.NullRef<TValue>();
        }
    }
}
