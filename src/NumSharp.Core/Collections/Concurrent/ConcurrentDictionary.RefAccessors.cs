// NumSharp-only extension of the vendored ConcurrentDictionary clone (ConcurrentDictionary.cs).
// The vendored file stays byte-faithful to dotnet/runtime apart from the single `partial` keyword; every
// NumSharp-specific member lives here so a future re-sync against the upstream source is a plain file replace.
//
// Members: the public ref seam GetValueRefOrNullRef, and the internal node-handle trio (NodeHandle, FindNode,
// TryRemoveNode) that lets ConcurrentOrderedDictionary do every key-comparer call of a multi-key removal BEFORE its
// first mutation. They read the vendored internals Node (_key/_value/_next/_hashcode), Tables (_comparer/_locks/
// _countPerLock), _tables, GetBucket, GetBucketAndLock and GetHashCode(comparer, key) — re-check those after a re-sync.

#nullable enable

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace NumSharp.Collections.Concurrent
{
    public partial class ConcurrentDictionary<TKey, TValue>
    {
        /// <summary>
        ///     A handle to one live hash-table node, resolved by <see cref="FindNode" />. It lets an owner that
        ///     serializes its own writes split a multi-key mutation into a <b>lookup phase</b> — every call into the key
        ///     comparer, which is user code and may throw — and a <b>mutation phase</b> that never calls the comparer
        ///     (<see cref="Value" /> writes, <see cref="TryRemoveNode" /> unlinks), so a throwing comparer can abort the
        ///     operation only while nothing has changed yet.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///         <see cref="ConcurrentOrderedDictionary{TKey,TValue}" />'s removals are the consumer: they used to remove
        ///         a key from this map and only then re-index the shifted or moved keys through the comparer, so a comparer
        ///         throw in that window (including the ordered map's own <see cref="LockRecursionException" /> refusal of a
        ///         write-back) stranded the two access paths permanently inconsistent.
        ///     </para>
        ///     <para>
        ///         <b>Lifetime — the same rule as <see cref="GetValueRefOrNullRef" />.</b> A handle stays valid only until
        ///         the next mutation of this dictionary other than the owner's own writes through handles: an update of a
        ///         non-atomically-writable value swaps the node, a removal or <see cref="Clear" /> unlinks it, and a table
        ///         growth (any add can trigger one) re-creates every node. A stale handle is not detected on writes — they
        ///         land on the dead node and are silently discarded — and <see cref="TryRemoveNode" /> reports it as
        ///         absent. Resolve, use and drop handles inside one externally-serialized critical section.
        ///     </para>
        /// </remarks>
        internal readonly struct NodeHandle
        {
            /// <summary>
            ///     The node, typed <see cref="object" /> only because the vendored <c>Node</c> class is private and a less
            ///     accessible type may not appear in an internal member's signature; it is always a <c>Node</c> of the
            ///     dictionary that produced the handle, or <see langword="null" /> for "not found".
            /// </summary>
            internal readonly object? _node;

            /// <summary>Wraps a node found by <see cref="FindNode" /> — the only producer, which is what keeps the unchecked reinterpretation in <see cref="Value" /> sound.</summary>
            /// <param name="node">A live node of the dictionary that is producing the handle.</param>
            internal NodeHandle(object node)
            {
                Debug.Assert(node is Node, "a NodeHandle must wrap a Node of the dictionary that produced it");
                _node = node;
            }

            /// <summary>Gets whether this is the "not found" handle (<see langword="default" />).</summary>
            internal bool IsNull => _node is null;

            /// <summary>
            ///     Gets a reference to the node's value field — no hashing, no bucket walk, no comparer call. Every
            ///     obligation of <see cref="GetValueRefOrNullRef" /> applies to writes through it: single-field,
            ///     at-most-word-size atomic stores only, under the owner's write serialization.
            /// </summary>
            /// <exception cref="NullReferenceException">The handle <see cref="IsNull" />.</exception>
            internal ref TValue Value
            {
                // Unsafe.As, not a cast: re-index loops read this once per shifted entry, and the only producer
                // (FindNode) guarantees the type.
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => ref Unsafe.As<Node>(_node!)._value;
            }
        }

        /// <summary>
        ///     Looks up <paramref name="key" /> exactly like <see cref="GetValueRefOrNullRef" /> — the same hash and the
        ///     same comparer calls — and returns a handle to its live node, so the caller can finish every comparer call
        ///     it needs before it mutates anything.
        /// </summary>
        /// <param name="key">The key to locate.</param>
        /// <returns>A handle to the key's node, or the <see cref="NodeHandle.IsNull" /> handle when the key is absent.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
        /// <remarks>The handle's lifetime rules are on <see cref="NodeHandle" />. Any exception the comparer throws propagates unchanged; the dictionary is only read.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] // called once per shifted entry in the ordered map's re-index loops, like the ref seam
        internal NodeHandle FindNode(TKey key)
        {
            // Same typeof guard as the ref seam: a bare `key is null` boxes value-type keys under Debug codegen.
            if (!typeof(TKey).IsValueType && key is null)
            {
                ThrowHelper.ThrowArgumentNullException(ExceptionArgument.key);
            }

            // Mirror of GetValueRefOrNullRef's walk (the value-type/default-comparer split keeps the JIT's
            // devirtualized Equals), returning the node itself instead of a ref into it.
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
                        return new NodeHandle(n);
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
                        return new NodeHandle(n);
                    }
                }
            }

            return default;
        }

        /// <summary>
        ///     Unlinks the node <paramref name="handle" /> refers to BY IDENTITY — the bucket is located from the hash
        ///     code stored in the node, and the chain is walked comparing references — so no key comparer runs. The
        ///     bookkeeping is exactly <c>TryRemoveInternal</c>'s (stripe lock, bucket unlink, per-lock count).
        /// </summary>
        /// <param name="handle">A handle from <see cref="FindNode" /> on this dictionary.</param>
        /// <returns><see langword="true" /> if the node was linked and is now unlinked; <see langword="false" /> for the null handle or a stale one whose node is no longer in the table.</returns>
        /// <remarks>
        ///     Calls no user code, so the only way it can fail is the runtime itself failing (the stripe lock is the one
        ///     <c>TryRemove</c> takes). Safe against concurrent lock-free readers for the same reason the vendored
        ///     removal is: the unlink is a single volatile link store.
        /// </remarks>
        internal bool TryRemoveNode(NodeHandle handle)
        {
            if (handle.IsNull)
            {
                return false;
            }

            // A checked cast here (once per removal) — the per-entry hot path is NodeHandle.Value, not this.
            Node target = (Node)handle._node!;
            Tables tables = _tables;

            // The stored hash code selects the bucket, so the comparer is never consulted. It stays valid across
            // tables: every Tables generation of this clone carries the same comparer (the upstream string-comparer
            // upgrade that could change it is not vendored).
            int hashcode = target._hashcode;

            while (true)
            {
                object[] locks = tables._locks;
                ref Node? bucket = ref GetBucketAndLock(tables, hashcode, out uint lockNo);

                lock (locks[lockNo])
                {
                    // Resized while we waited for the stripe lock: retry on the live tables, exactly like
                    // TryRemoveInternal. A growth re-creates every node, so a pre-growth handle will then simply not be
                    // found — the correct answer for a handle that outlived its node.
                    if (tables != _tables)
                    {
                        tables = _tables;
                        continue;
                    }

                    Node? prev = null;
                    for (Node? curr = bucket; curr is not null; curr = curr._next)
                    {
                        Debug.Assert((prev is null && curr == bucket) || prev!._next == curr);

                        if (ReferenceEquals(curr, target))
                        {
                            if (prev is null)
                            {
                                Volatile.Write(ref bucket, curr._next);
                            }
                            else
                            {
                                prev._next = curr._next;
                            }

                            tables._countPerLock[lockNo]--;
                            return true;
                        }

                        prev = curr;
                    }
                }

                return false;
            }
        }

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
