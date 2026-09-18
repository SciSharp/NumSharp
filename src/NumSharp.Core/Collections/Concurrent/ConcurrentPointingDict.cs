#nullable enable

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using NumSharp.Collections.Concurrent;

namespace NumSharp.Collections;

/// <summary>
///     <b>EXPERIMENTAL — a third point in the ordered-concurrent-map design space, built to be measured, not yet
///     shipped.</b> A thread-safe, insertion-ordered, index-addressable map whose key path is the vendored
///     <see cref="Concurrent.ConcurrentDictionary{TKey,TValue}" /> clone used as
///     <c>ConcurrentDictionary&lt;TKey, int&gt;</c> — the hash node stores only an <b>int32 index</b> that "points"
///     into a contiguous <typeparamref name="TValue" /> list, so <b>each value is stored exactly once</b> (in the
///     list) rather than denormalized into the node. This is the user's "pointing dictionary" lead:
///     <c>hash → int32 index → List&lt;TValue&gt;</c>.
/// </summary>
/// <remarks>
///     <para>
///         <b>Where it sits between the two shipped siblings.</b>
///         <list type="bullet">
///             <item>
///                 <see cref="ConcurrentOrderedDict{TKey,TValue}" /> (COD) stores the value <b>twice</b> — inline in
///                 the hash node (<c>ValueIndex { value, index }</c>) AND in the contiguous array — buying a
///                 <b>one-load</b> key hit at the cost of the denormalized copy (~57 B/entry for <c>&lt;int,int&gt;</c>,
///                 growing with value width).
///             </item>
///             <item>
///                 <see cref="ConcurrentOrderedCompactDict{TKey,TValue}" /> (COCD) folds the whole index into a
///                 bespoke open-addressed table (no per-entry node) — value stored <b>once</b>, ~21–25 B/entry, key
///                 hit two loads (index word + value; the small-key tag avoids the keys array).
///             </item>
///             <item>
///                 <b>This type (CPD)</b> keeps the <b>proven</b> vendored CD for the hash part but moves the value
///                 out of the node into the list — value stored <b>once</b>, but the node (~40 B) stays. So its niche
///                 is <b>wide-value memory</b> (no inline value copy) reached with the <b>lowest novel-concurrency
///                 risk</b> (it reuses CD's battle-tested lock-free machinery rather than a hand-rolled table), at the
///                 cost of a key hit that is inherently the most load-heavy of the three (see below).
///             </item>
///         </list>
///     </para>
///     <para>
///         <b>The validated read — why the key hit costs three dependent loads.</b> COD reads the value straight out
///         of the node (one atomic load, always consistent). CPD's key→value path spans <b>two independently
///         published objects</b>: the node's <c>int</c> index and the <see cref="Store" />'s value array. A
///         concurrent order-preserving interior removal rewrites a moved key's node index (<c>c</c>) and then
///         publishes a fresh <see cref="Store" /> (<c>d</c>); in the window between them a lock-free reader can pair a
///         stale index with a fresh store (or vice versa) and read a <i>neighbour's</i> value. The fix is a
///         <b>validated read</b> that, after loading the value, verifies the slot actually holds the requested key —
///         <c>comparer.Equals(store._keys[idx], key)</c> — and retries otherwise. It is correct with <b>no
///         writer-side changes</b> because every published <see cref="Store" /> is immutable in its live slots (COD's
///         rule 2), so a consistent <c>(store, idx)</c> pair holding the key is stable for the whole read and an
///         inconsistent pair is always caught. The cost is the honest price of "pointing" on a node that cannot
///         co-locate the value: a hit reads the node index, then <c>keys[idx]</c> (validate), then <c>values[idx]</c>
///         — three dependent loads plus one comparer call, versus COD's one and COCD's two.
///     </para>
///     <para>
///         <b>One thing CPD makes simpler than COD.</b> The node holds only an <c>int</c>, which is always
///         atomically writable, so the node is <b>never</b> swapped for a wide (non-atomic) <typeparamref name="TValue" />
///         — a value replacement only ever touches the value array (in place when atomic, copy-on-write otherwise),
///         and a removal reindex is always an in-place atomic <c>int</c> store. COD needs a node swap for non-atomic
///         values; CPD does not.
///     </para>
///     <para>
///         <b>Memory model.</b> The list path is COD's exactly — a <see cref="Store" /> of parallel keys/values plus a
///         monotonic count and an append floor, published through one volatile field, with COD's three rules (count
///         only grows on an instance; a reader's visible slots are never structurally rewritten; value slots are
///         rewritten in place only for atomically-writable values). Reads (key lookup, membership,
///         <see cref="IndexOf" />, positional get, <see cref="Count" />, enumeration, <see cref="ToArray" />,
///         <see cref="Snapshot" />) are lock-free; writes are serialized by one lock, and reentrant writes from user
///         callbacks the collection runs under the lock are refused with <see cref="LockRecursionException" />.
///     </para>
///     <para>
///         <b>Status.</b> This is a scaffold for the head-to-head benchmark against COD/COCD; it carries the headline
///         surface (build, key get, enumerate, index get, replace, remove) and the concurrency-correct read, but not
///         yet the full COD surface (<c>AddRange</c>, <c>GetOrAdd</c>, <c>AddOrUpdate</c>, <c>TryUpdate</c>,
///         <c>Pairs</c>, <c>CopyTo</c>) nor the mirrored test/gun suites. Those follow only if measurement shows the
///         lead is worth shipping.
///     </para>
/// </remarks>
/// <typeparam name="TKey">The non-null key type; uniqueness and lookups use the configured comparer.</typeparam>
/// <typeparam name="TValue">The value type stored once in the list and yielded, in index order, by enumeration.</typeparam>
public sealed class ConcurrentPointingDict<TKey, TValue> : IReadOnlyList<TValue>
    where TKey : notnull
{
    /// <summary>The initial ordered-array capacity, and the size the arrays first grow to from empty (small: many ordered maps stay tiny).</summary>
    private const int DefaultCapacity = 4;

    /// <summary>
    ///     The list path: contiguous keys and values in insertion order, published through the single volatile
    ///     <see cref="_store" /> field; the live prefix is <c>[0, <see cref="_count" />)</c>. Identical in shape and
    ///     rules to <see cref="ConcurrentOrderedDict{TKey,TValue}" />'s store (this type reuses that proven model
    ///     verbatim — only the key map's payload differs).
    /// </summary>
    private sealed class Store
    {
        /// <summary>Keys in insertion order, occupying <c>[0, <see cref="_count" />)</c>. Parallel to <see cref="_values" />; the validated read reads it to confirm a slot holds the looked-up key.</summary>
        internal readonly TKey[] _keys;

        /// <summary>Values in insertion order, occupying <c>[0, <see cref="_count" />)</c> — the single copy of each value and the contiguous store the list path scans.</summary>
        internal readonly TValue[] _values;

        /// <summary>The first slot an in-place append may write; slots below it were exposed to readers by an earlier higher-count generation sharing these arrays (an O(1) tail removal), so an append below the floor must copy first. 0 for fresh arrays.</summary>
        internal readonly int _floor;

        /// <summary>The number of live entries — mutable and monotonically increasing on this instance (appends release-store it); every shrinking transition publishes a new <see cref="Store" />. Readers capture it once via <see cref="Volatile.Read(ref int)" />.</summary>
        internal int _count;

        /// <summary>Wraps parallel key/value arrays, their live count and the append floor into a publishable snapshot.</summary>
        /// <param name="keys">The ordered keys (only the first <paramref name="count" /> slots are meaningful).</param>
        /// <param name="values">The ordered values (only the first <paramref name="count" /> slots are meaningful).</param>
        /// <param name="count">The number of live entries at publication.</param>
        /// <param name="floor">The first slot an in-place append may write (0 for fresh arrays; the prior high-water mark when arrays are shared downward).</param>
        internal Store(TKey[] keys, TValue[] values, int count, int floor = 0)
        {
            _keys = keys;
            _values = values;
            _count = count;
            _floor = floor;
        }

        /// <summary>The shared empty snapshot used before the first insertion and after <see cref="Clear" />.</summary>
        internal static readonly Store Empty = new(Array.Empty<TKey>(), Array.Empty<TValue>(), 0);
    }

    /// <summary>
    ///     The key path: maps each key to its current insertion-order <b>int32 index</b> into <see cref="_store" />.
    ///     A vendored clone of dotnet's thread-safe dictionary (lock-free O(1) key reads, key uniqueness). The node
    ///     payload is a plain <c>int</c>, so every in-place node write (the reindex on removal) is atomic and no node
    ///     is ever swapped for a wide value type.
    /// </summary>
    /// <remarks>Constructed with <c>concurrencyLevel: 1</c>: every mutation happens under <see cref="_writeLock" />, so its striped write locks can never be contended and one stripe removes the lock-array footprint. Lock-free reads are unaffected by stripe count.</remarks>
    private readonly ConcurrentDictionary<TKey, int> _byKey;

    /// <summary>The list path: the current contiguous ordered snapshot. Volatile so structural publishes reach lock-free readers with acquire/release ordering.</summary>
    private volatile Store _store;

    /// <summary>Serializes every mutation so the global insertion order, the value list and the key map stay consistent; reads never acquire it. It also serializes all writes to <see cref="_byKey" />, which is what makes the in-place index writes through the clone's ref seam race-free.</summary>
    private readonly object _writeLock = new();

    /// <summary>The comparer cached from the key map, used by the validated read's per-hit key confirmation (hoisted so the hot path avoids a property call).</summary>
    private readonly IEqualityComparer<TKey> _comparer;

    /// <summary>Creates an empty collection using the default comparer for <typeparamref name="TKey" />.</summary>
    public ConcurrentPointingDict() : this((IEqualityComparer<TKey>?)null)
    {
    }

    /// <summary>Creates an empty collection using the supplied key comparer.</summary>
    /// <param name="comparer">The comparer used for key uniqueness and lookups, or <see langword="null" /> for the default.</param>
    public ConcurrentPointingDict(IEqualityComparer<TKey>? comparer)
    {
        // concurrencyLevel 1: all map writes are serialized by _writeLock, so extra stripes are pure overhead.
        _byKey = new ConcurrentDictionary<TKey, int>(1, 31, comparer);
        _comparer = _byKey.Comparer;
        _store = Store.Empty;
    }

    /// <summary>Creates an empty collection with room reserved for <paramref name="capacity" /> entries.</summary>
    /// <param name="capacity">The number of entries to pre-size both paths for; must be non-negative.</param>
    /// <param name="comparer">The comparer used for key uniqueness and lookups, or <see langword="null" /> for the default.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="capacity" /> is negative.</exception>
    public ConcurrentPointingDict(int capacity, IEqualityComparer<TKey>? comparer = null)
    {
        if (capacity < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity));
        }

        _byKey = new ConcurrentDictionary<TKey, int>(1, capacity, comparer);
        _comparer = _byKey.Comparer;
        _store = capacity == 0 ? Store.Empty : new Store(new TKey[capacity], new TValue[capacity], 0);
    }

    /// <summary>Gets the number of entries — a moment-in-time snapshot, consistent with lock-free enumeration.</summary>
    public int Count => Volatile.Read(ref _store._count);

    /// <summary>Gets whether the collection currently has no entries.</summary>
    public bool IsEmpty => Count == 0;

    /// <summary>Gets the comparer used to determine key equality and hashing.</summary>
    public IEqualityComparer<TKey> Comparer => _comparer;

    // ---------------------------------------------------------------------------------------------------------
    // Key path (dictionary semantics). For int keys, reach these through the named methods, not the indexer.
    // ---------------------------------------------------------------------------------------------------------

    /// <summary>Gets or sets the value for <paramref name="key" />. Get throws if absent; set updates in place if the key exists, otherwise appends at the end of insertion order.</summary>
    /// <param name="key">The key to read or write.</param>
    /// <value>The value associated with <paramref name="key" />.</value>
    /// <remarks>When <typeparamref name="TKey" /> is <c>int</c> this indexer is shadowed by <see cref="this[int]" />; use <see cref="GetByKey" />/<see cref="SetByKey" /> instead.</remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="KeyNotFoundException">The get accessor is used and <paramref name="key" /> is absent.</exception>
    public TValue this[TKey key]
    {
        get => GetByKey(key);
        set => SetByKey(key, value);
    }

    /// <summary>Reads the value for <paramref name="key" />, throwing if absent (the method form of the key getter — use this for <c>int</c> keys). Lock-free; a hit is three dependent loads plus a comparer call (the validated read).</summary>
    /// <param name="key">The key to look up.</param>
    /// <returns>The value associated with <paramref name="key" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="KeyNotFoundException"><paramref name="key" /> is absent.</exception>
    public TValue GetByKey(TKey key)
    {
        if (!TryGetValue(key, out TValue value))
        {
            throw new KeyNotFoundException(Concurrent.SR.Format(Concurrent.SR.Arg_KeyNotFoundWithKey, key.ToString()));
        }

        return value;
    }

    /// <summary>Upserts <paramref name="key" />: updates its value in place if present (keeping its position), otherwise appends (the method form of the key setter — use this for <c>int</c> keys). Serialized with other writes.</summary>
    /// <param name="key">The key to write.</param>
    /// <param name="value">The value to store.</param>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public void SetByKey(TKey key, TValue value)
    {
        NullCheck(key);
        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            // One bucket walk decides the branch. On a hit the node's int index tells us the value slot to rewrite;
            // the node itself is never touched by a value replace (its index is unchanged), unlike COD which stores
            // the value in the node.
            ref int idxRef = ref _byKey.GetValueRefOrNullRef(key);
            if (!Unsafe.IsNullRef(ref idxRef))
            {
                ReplaceValueUnderLock(idxRef, value);
            }
            else
            {
                bool added = TryAppendUnderLock(key, value);
                Debug.Assert(added, "key vanished between the ref lookup and the append while holding the write lock");
            }
        }
    }

    /// <summary>Determines whether <paramref name="key" /> is present. Lock-free, O(1) and allocation-free (no value read, so no validation needed).</summary>
    /// <param name="key">The key to test.</param>
    /// <returns><see langword="true" /> if the key exists; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool ContainsKey(TKey key) => !Unsafe.IsNullRef(ref _byKey.GetValueRefOrNullRef(key));

    /// <summary>
    ///     Attempts to read the value for <paramref name="key" /> without throwing. Lock-free. <b>The validated read:</b>
    ///     a hit loads the node's int index, captures the store, confirms the store's slot still holds this key, and
    ///     only then reads the value — retrying if a concurrent reindex left a stale index paired with a fresh store.
    /// </summary>
    /// <param name="key">The key to look up.</param>
    /// <param name="value">On success, the associated value; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if the key was found; otherwise <see langword="false" />.</returns>
    /// <remarks>
    ///     The <c>keys[idx]</c> confirmation is what makes "pointing" tear-free without any writer-side seqlock: a
    ///     published <see cref="Store" /> is immutable in its live slots, so a <c>(store, idx)</c> pair that holds the
    ///     key is stable for the whole read, and one from mismatched generations is always rejected. Under quiescence
    ///     (no concurrent structural writer) the confirmation passes on the first pass — the retry loop only spins
    ///     while an interior removal / swap-back is actively reindexing this key.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool TryGetValue(TKey key, out TValue value)
    {
        while (true)
        {
            // The node holds only the int index; a null ref means the key is absent.
            ref int idxRef = ref _byKey.GetValueRefOrNullRef(key);
            if (Unsafe.IsNullRef(ref idxRef))
            {
                value = default!;
                return false;
            }

            int idx = idxRef;   // the key's current index; may be from a newer generation than the store below
            Store s = _store;   // volatile acquire — everything published before it is visible

            // Confirm (s, idx) is consistent: idx is a live slot of s AND that slot still holds our key. If it does,
            // the value there IS our value (published stores are immutable in their live slots). If not, a concurrent
            // reindex skewed the pair — re-resolve the index against a re-read store on the next turn.
            if ((uint)idx < (uint)s._count && _comparer.Equals(s._keys[idx], key))
            {
                value = s._values[idx];
                return true;
            }

            // Fall through to retry. A key concurrently removed will resolve to a null ref next turn → false.
        }
    }

    /// <summary>Adds <paramref name="key" />/<paramref name="value" /> only if the key is absent, appending it at the end of order.</summary>
    /// <param name="key">The key to add.</param>
    /// <param name="value">The value to add.</param>
    /// <returns><see langword="true" /> if the entry was added; <see langword="false" /> if the key already existed.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public bool TryAdd(TKey key, TValue value)
    {
        NullCheck(key);
        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            return TryAppendUnderLock(key, value);
        }
    }

    /// <summary>Adds <paramref name="key" />/<paramref name="value" />, throwing if the key already exists.</summary>
    /// <param name="key">The key to add.</param>
    /// <param name="value">The value to add.</param>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="ArgumentException">An entry with the same key already exists.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public void Add(TKey key, TValue value)
    {
        if (!TryAdd(key, value))
        {
            throw new ArgumentException(Concurrent.SR.ConcurrentDictionary_KeyAlreadyExisted, nameof(key));
        }
    }

    /// <summary>Removes <paramref name="key" /> and its value. Removing the <b>last</b> entry is O(1); an interior entry shifts later entries down to keep indices contiguous — <b>O(n)</b>.</summary>
    /// <param name="key">The key to remove.</param>
    /// <param name="value">On success, the removed value; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if an entry was removed; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public bool TryRemove(TKey key, out TValue value)
    {
        NullCheck(key);

        // Lock-free fast negative: interior-removal work is O(n), so keep absent-key callers out of the lock.
        if (Unsafe.IsNullRef(ref _byKey.GetValueRefOrNullRef(key)))
        {
            value = default!;
            return false;
        }

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            ref int idxRef = ref _byKey.GetValueRefOrNullRef(key);
            if (Unsafe.IsNullRef(ref idxRef))
            {
                value = default!;
                return false;
            }

            int idx = idxRef;
            // Under the lock the (store, idx) pair is consistent (we are the only writer), so no validation is needed.
            value = _store._values[idx];
            RemoveCoreUnderLock(key, idx);
            return true;
        }
    }

    /// <summary>Removes <paramref name="key" /> if present. Convenience wrapper over <see cref="TryRemove" />.</summary>
    /// <param name="key">The key to remove.</param>
    /// <returns><see langword="true" /> if an entry was removed; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public bool Remove(TKey key) => TryRemove(key, out _);

    /// <summary>
    ///     Removes <paramref name="key" /> in O(1) by moving the current last entry into the removed slot — the
    ///     order-breaking escape hatch from the O(n) shifting removal. Only the moved entry's position changes.
    /// </summary>
    /// <param name="key">The key to remove.</param>
    /// <param name="value">On success, the removed value; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if an entry was removed; otherwise <see langword="false" />.</returns>
    /// <remarks>
    ///     Like the sibling types, this gives up insertion order for the relocated entry and (when both key and value
    ///     are atomically writable, where the swap is in place) snapshot purity for a concurrent list-path enumerator:
    ///     it can see the moved entry at both positions. The key path stays tear-free — the validated read confirms the
    ///     slot's key, so a reader mid-swap simply retries. Wide (non-atomic) key/value types degrade to an O(n) fresh
    ///     array copy that keeps the enumerator pure.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public bool TryRemoveSwapBack(TKey key, out TValue value)
    {
        NullCheck(key);

        if (Unsafe.IsNullRef(ref _byKey.GetValueRefOrNullRef(key)))
        {
            value = default!;
            return false;
        }

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            ref int idxRef = ref _byKey.GetValueRefOrNullRef(key);
            if (Unsafe.IsNullRef(ref idxRef))
            {
                value = default!;
                return false;
            }

            value = _store._values[idxRef];
            int idx = idxRef;
            Store s = _store;
            int n = s._count;

            if (idx == n - 1)
            {
                // Last entry: no move at all — identical to the tail fast path of TryRemove. Successor allocated
                // BEFORE the map removal (strand-proofing: once the key stops resolving, only throw-free ops remain).
                Store next = new(s._keys, s._values, n - 1, Math.Max(s._floor, n));
                _byKey.TryRemove(key, out _);
                _store = next;
                return true;
            }

            TKey movedKey = s._keys[n - 1];
            TValue movedValue = s._values[n - 1];

            if (ConcurrentDictionaryTypeProps<TKey>.IsWriteAtomic && ConcurrentDictionaryTypeProps<TValue>.IsWriteAtomic)
            {
                // In-place swap: single atomic stores per slot. The key path's validated read confirms keys[idx], so
                // a reader mid-swap retries rather than tearing; the list-path duplicate/missing anomaly is the
                // documented relaxation.
                Store next = new(s._keys, s._values, n - 1, Math.Max(s._floor, n)); // alloc before any mutation
                _byKey.TryRemove(key, out _);
                s._keys[idx] = movedKey;
                s._values[idx] = movedValue;
                RewriteIndexUnderLock(movedKey, idx);
                _store = next;
                return true;
            }
            else
            {
                // Wide key or value: copy fresh arrays (snapshot-pure, O(n) memcpy, still re-index-free). Allocations
                // before the map removal (strand-proofing).
                var keys = new TKey[s._keys.Length];
                var values = new TValue[s._values.Length];
                Array.Copy(s._keys, keys, n - 1);
                Array.Copy(s._values, values, n - 1);
                keys[idx] = movedKey;
                values[idx] = movedValue;
                Store next = new(keys, values, n - 1);
                _byKey.TryRemove(key, out _);
                RewriteIndexUnderLock(movedKey, idx);
                _store = next;
                return true;
            }
        }
    }

    /// <summary>Removes every entry matching <paramref name="match" /> in one compaction pass — O(n) total instead of O(n) per removed entry, with a single new snapshot.</summary>
    /// <param name="match">Returns <see langword="true" /> for entries to remove; receives each key and its current value in index order.</param>
    /// <returns>The number of entries removed.</returns>
    /// <remarks>The predicate runs inside the write lock and must not write back into the collection (refused with <see cref="LockRecursionException" />); a throwing predicate lands the collection in the consistent "removed everything matched so far" state.</remarks>
    /// <exception cref="ArgumentNullException"><paramref name="match" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public int RemoveWhere(Func<TKey, TValue, bool> match)
    {
        if (match is null)
        {
            throw new ArgumentNullException(nameof(match));
        }

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            Store s = _store;
            int n = s._count;

            // First pass: find the first match so the no-op case allocates nothing.
            int first = 0;
            while (first < n && !match(s._keys[first], s._values[first]))
            {
                first++;
            }

            if (first == n)
            {
                return 0;
            }

            var keys = new TKey[s._keys.Length];
            var values = new TValue[s._values.Length];
            Array.Copy(s._keys, keys, first);
            Array.Copy(s._values, values, first);

            int w = first;
            int removed = 0;
            int i = first;
            try
            {
                for (; i < n; i++)
                {
                    TKey k = s._keys[i];
                    TValue v = s._values[i];
                    if (match(k, v))
                    {
                        _byKey.TryRemove(k, out _);
                        removed++;
                    }
                    else
                    {
                        keys[w] = k;
                        values[w] = v;
                        if (w != i)
                        {
                            RewriteIndexUnderLock(k, w);
                        }

                        w++;
                    }
                }
            }
            finally
            {
                // A throwing predicate must not strand the pass half-applied: carry every not-yet-processed entry —
                // including the one whose predicate threw — over as a survivor and publish.
                for (; i < n; i++)
                {
                    TKey k = s._keys[i];
                    keys[w] = k;
                    values[w] = s._values[i];
                    if (w != i)
                    {
                        RewriteIndexUnderLock(k, w);
                    }

                    w++;
                }

                _store = new Store(keys, values, w);
            }

            return removed;
        }
    }

    // ---------------------------------------------------------------------------------------------------------
    // List path (index semantics). Enumeration and ToArray yield values in index order.
    // ---------------------------------------------------------------------------------------------------------

    /// <summary>Gets or sets the value at insertion-order position <paramref name="index" />. The setter replaces the value at that position (never inserts). Wins over <see cref="this[TKey]" /> when <typeparamref name="TKey" /> is <c>int</c>.</summary>
    /// <param name="index">The zero-based position in insertion order.</param>
    /// <value>The value at <paramref name="index" />.</value>
    /// <remarks>The getter pays one volatile snapshot read per call; for a hot scan take <see cref="Snapshot" /> once and index the view.</remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
    public TValue this[int index]
    {
        get
        {
            Store s = _store;
            if ((uint)index >= (uint)Volatile.Read(ref s._count))
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            return s._values[index];
        }
        set => SetAt(index, value);
    }

    /// <summary>Replaces the value at position <paramref name="index" /> without changing its key or position. Serialized with other writes.</summary>
    /// <param name="index">The zero-based position in insertion order.</param>
    /// <param name="value">The replacement value.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public void SetAt(int index, TValue value)
    {
        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            Store s = _store;
            if ((uint)index >= (uint)s._count)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            // The key at this position keeps its node index; only the value slot changes.
            ReplaceValueUnderLock(index, value);
        }
    }

    /// <summary>Attempts to read the value at position <paramref name="index" /> without throwing. Lock-free and O(1).</summary>
    /// <param name="index">The zero-based position in insertion order.</param>
    /// <param name="value">On success, the value at the position; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if <paramref name="index" /> is in range; otherwise <see langword="false" />.</returns>
    public bool TryGetAt(int index, out TValue value)
    {
        Store s = _store;
        if ((uint)index >= (uint)Volatile.Read(ref s._count))
        {
            value = default!;
            return false;
        }

        value = s._values[index];
        return true;
    }

    /// <summary>Returns the key at position <paramref name="index" /> in insertion order. Lock-free and O(1).</summary>
    /// <param name="index">The zero-based position in insertion order.</param>
    /// <returns>The key stored at that position.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
    public TKey GetKeyAt(int index)
    {
        Store s = _store;
        if ((uint)index >= (uint)Volatile.Read(ref s._count))
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        return s._keys[index];
    }

    /// <summary>Returns the insertion-order position of <paramref name="key" />, or -1 if absent — the "TKey to index" mapping. Lock-free and O(1), reading the node's int directly (one load, no validation).</summary>
    /// <param name="key">The key to locate.</param>
    /// <returns>The zero-based index of the key, or -1 if it is not present.</returns>
    /// <remarks>Under a concurrent structural writer the returned index is a snapshot and may be stale by the number of concurrent removals (the same named relaxation as the sibling types).</remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public int IndexOf(TKey key)
    {
        ref int idxRef = ref _byKey.GetValueRefOrNullRef(key);
        return Unsafe.IsNullRef(ref idxRef) ? -1 : idxRef;
    }

    /// <summary>Attempts to map <paramref name="key" /> to its insertion-order position without allocating a sentinel.</summary>
    /// <param name="key">The key to locate.</param>
    /// <param name="index">On success, the key's index; otherwise -1.</param>
    /// <returns><see langword="true" /> if the key was found; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool TryGetIndex(TKey key, out int index)
    {
        ref int idxRef = ref _byKey.GetValueRefOrNullRef(key);
        if (!Unsafe.IsNullRef(ref idxRef))
        {
            index = idxRef;
            return true;
        }

        index = -1;
        return false;
    }

    /// <summary>Removes the entry at position <paramref name="index" />, shifting later entries down. O(1) for the last position, <b>O(n)</b> otherwise.</summary>
    /// <param name="index">The zero-based position in insertion order to remove.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public void RemoveAt(int index)
    {
        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            Store s = _store;
            if ((uint)index >= (uint)s._count)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            RemoveCoreUnderLock(s._keys[index], index);
        }
    }

    /// <summary>Removes all entries and resets both paths to empty.</summary>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public void Clear()
    {
        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            _byKey.Clear();
            _store = Store.Empty;
        }
    }

    /// <summary>Copies the values, in index order, into a new array (a contiguous block copy — the single value copy makes this List-parity).</summary>
    /// <returns>A new array of the values in insertion order (a point-in-time snapshot).</returns>
    public TValue[] ToArray()
    {
        Store s = _store;
        int n = Volatile.Read(ref s._count);
        if (n == 0)
        {
            return Array.Empty<TValue>();
        }

        var array = new TValue[n];
        Array.Copy(s._values, array, n);
        return array;
    }

    /// <summary>
    ///     Captures the current contents as a <see cref="ValuesView" /> — a zero-allocation snapshot whose indexer and
    ///     spans read the contiguous value list with no per-access volatile read (the List-parity hot-loop path).
    /// </summary>
    /// <returns>A point-in-time view of the values (and keys) in insertion order.</returns>
    /// <remarks>Immutable in shape but not in value (an in-place atomic value replace is visible through it). Hold it only for the scan; it pins the captured arrays.</remarks>
    public ValuesView Snapshot()
    {
        Store s = _store;
        return new ValuesView(s._keys, s._values, Volatile.Read(ref s._count));
    }

    /// <summary>A point-in-time, index-addressable view over the values (and keys) captured by <see cref="Snapshot" /> — array-access cost, and <see cref="AsSpan" /> for span-based code.</summary>
    public readonly struct ValuesView
    {
        /// <summary>The captured keys array; only <c>[0, <see cref="Count" />)</c> is part of the view.</summary>
        private readonly TKey[] _keys;

        /// <summary>The captured values array; only <c>[0, <see cref="Count" />)</c> is part of the view.</summary>
        private readonly TValue[] _values;

        /// <summary>The captured live count — fixed for the lifetime of the view.</summary>
        private readonly int _count;

        /// <summary>Binds the view to a captured (keys, values, count) triple; called only by <see cref="Snapshot" />.</summary>
        /// <param name="keys">The captured ordered keys.</param>
        /// <param name="values">The captured ordered values.</param>
        /// <param name="count">The captured live count.</param>
        internal ValuesView(TKey[] keys, TValue[] values, int count)
        {
            _keys = keys;
            _values = values;
            _count = count;
        }

        /// <summary>Gets the number of entries captured in this view.</summary>
        public int Count => _count;

        /// <summary>Gets the value at position <paramref name="index" /> of the captured snapshot — a plain array read, no volatile.</summary>
        /// <param name="index">The zero-based position in insertion order at capture time.</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
        public TValue this[int index]
        {
            get
            {
                if ((uint)index >= (uint)_count)
                {
                    throw new ArgumentOutOfRangeException(nameof(index));
                }

                return _values[index];
            }
        }

        /// <summary>Gets the key at position <paramref name="index" /> of the captured snapshot.</summary>
        /// <param name="index">The zero-based position in insertion order at capture time.</param>
        /// <returns>The key captured at that position.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
        public TKey GetKeyAt(int index)
        {
            if ((uint)index >= (uint)_count)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            return _keys[index];
        }

        /// <summary>Returns the captured values as a read-only span — the fastest scan surface. An empty span for a default-constructed view.</summary>
        /// <returns>A span over the live prefix of the captured values array.</returns>
        public ReadOnlySpan<TValue> AsSpan() => _values is null ? default : new ReadOnlySpan<TValue>(_values, 0, _count);

        /// <summary>Returns the captured keys as a read-only span, parallel to <see cref="AsSpan" />.</summary>
        /// <returns>A span over the live prefix of the captured keys array.</returns>
        public ReadOnlySpan<TKey> KeysAsSpan() => _keys is null ? default : new ReadOnlySpan<TKey>(_keys, 0, _count);

        /// <summary>Returns a span enumerator over the captured values, so the view itself works in <c>foreach</c> at array speed.</summary>
        /// <returns>The value span's enumerator.</returns>
        public ReadOnlySpan<TValue>.Enumerator GetEnumerator() => AsSpan().GetEnumerator();
    }

    /// <summary>Returns an allocation-free enumerator over the values in insertion order (snapshotted at this call).</summary>
    /// <returns>A struct enumerator; each iteration yields the next value in index order.</returns>
    public Enumerator GetEnumerator() => new(this);

    /// <inheritdoc />
    IEnumerator<TValue> IEnumerable<TValue>.GetEnumerator() => GetEnumerator();

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>A forward, value-order enumerator bound to the contiguous value array and count captured when created; it scans that array like a <see cref="List{T}" /> enumerator and never throws for concurrent modification.</summary>
    public struct Enumerator : IEnumerator<TValue>
    {
        private readonly TValue[] _values;
        private readonly int _count;
        private int _i;
        private TValue _current;

        /// <summary>Binds the enumerator to a snapshot of the owner's current ordered store.</summary>
        /// <param name="owner">The collection to snapshot and enumerate.</param>
        internal Enumerator(ConcurrentPointingDict<TKey, TValue> owner)
        {
            Store store = owner._store;
            _values = store._values;
            // Acquire-read the count AFTER capturing the array: slots below it were release-published.
            _count = Volatile.Read(ref store._count);
            _i = -1;
            _current = default!;
        }

        /// <summary>Gets the value at the enumerator's current position.</summary>
        public TValue Current => _current;

        /// <inheritdoc />
        object? IEnumerator.Current => _current;

        /// <summary>Advances to the next value in index order.</summary>
        /// <returns><see langword="true" /> if a value is now current; <see langword="false" /> once the snapshot is exhausted.</returns>
        public bool MoveNext()
        {
            int next = _i + 1;
            if (next < _count)
            {
                _current = _values[next];
                _i = next;
                return true;
            }

            _current = default!;
            return false;
        }

        /// <summary>Resets the enumerator to before the first value of its snapshot.</summary>
        public void Reset()
        {
            _i = -1;
            _current = default!;
        }

        /// <summary>No-op; the enumerator holds no unmanaged resources.</summary>
        public void Dispose()
        {
        }
    }

    // ---------------------------------------------------------------------------------------------------------
    // Private write helpers. All callers hold _writeLock.
    // ---------------------------------------------------------------------------------------------------------

    /// <summary>
    ///     Appends <paramref name="key" />/<paramref name="value" /> at the end of insertion order if the key is
    ///     absent — the single-bucket-walk core every add funnels through. The map's <c>TryAdd</c> answers "already
    ///     exists?" itself, and the map value stored is the entry's <b>int index</b>.
    /// </summary>
    /// <param name="key">The key to append (already null-checked by the caller).</param>
    /// <param name="value">The value to store.</param>
    /// <returns><see langword="true" /> if appended; <see langword="false" /> if the key already existed.</returns>
    /// <remarks>The caller must hold <see cref="_writeLock" />. Every allocation (grown arrays, the node inside <c>TryAdd</c>) happens before the map insert succeeds, so only throw-free publishes remain afterwards — no key can resolve on the key path yet never appear on the list path.</remarks>
    private bool TryAppendUnderLock(TKey key, TValue value)
    {
        Store s = _store;
        int idx = s._count; // plain read: every count writer holds the lock we hold
        TKey[] keys = s._keys;
        TValue[] values = s._values;

        Store? grown = null;
        if (idx == values.Length || idx < s._floor)
        {
            // Double only when actually full; a copy forced only by the floor rule keeps its capacity (the
            // AddRange capacity-doubling bug's lesson — never compound the buffer on a floor-only copy).
            int newCap = values.Length == 0
                ? DefaultCapacity
                : idx == values.Length ? (int)Math.Min((long)values.Length * 2, Array.MaxLength) : values.Length;
            keys = CopyArray(s._keys, newCap, idx);
            values = CopyArray(s._values, newCap, idx);
            keys[idx] = key;
            values[idx] = value;
            grown = new Store(keys, values, idx + 1);
        }
        else
        {
            // Writing above the visible count: invisible to every reader until the count publish below.
            keys[idx] = key;
            values[idx] = value;
        }

        // One walk: the map insert IS the duplicate check. The stored value is the int index.
        if (!_byKey.TryAdd(key, idx))
        {
            return false;
        }

        if (grown is not null)
        {
            _store = grown; // volatile publish
        }
        else
        {
            Volatile.Write(ref s._count, idx + 1); // release store makes the written slots visible
        }

        return true;
    }

    /// <summary>
    ///     Replaces the value at position <paramref name="idx" /> (an existing entry's slot). The key's node index is
    ///     unchanged, so the key map is never touched — only the value list. In place for atomically-writable values
    ///     (O(1), zero allocation); otherwise a fresh value array is published so lock-free readers never tear (O(n)).
    /// </summary>
    /// <param name="idx">The insertion-order slot to overwrite (a live position; the caller resolved it under the lock).</param>
    /// <param name="value">The replacement value.</param>
    /// <remarks>The caller must hold <see cref="_writeLock" />. Simpler than the sibling COD, which must swap the hash node for a non-atomic value — here the node holds only the (unchanged) int index.</remarks>
    private void ReplaceValueUnderLock(int idx, TValue value)
    {
        Store s = _store;
        if ((uint)idx >= (uint)s._count)
        {
            return; // out of range under the lock should not happen; guard defensively
        }

        if (ConcurrentDictionaryTypeProps<TValue>.IsWriteAtomic)
        {
            s._values[idx] = value; // in-place atomic store; lock-free readers see old or new, never torn
            return;
        }

        // Non-atomic value: publish a fresh value array so no reader can tear a wide value. Keys are shared with the
        // old store, so its floor still protects any frozen tail slots.
        var values = (TValue[])s._values.Clone();
        values[idx] = value;
        _store = new Store(s._keys, values, s._count, s._floor);
    }

    /// <summary>
    ///     Removes the entry for <paramref name="key" /> at position <paramref name="idx" /> from both paths. The last
    ///     entry is removed in O(1) (share arrays, lower count via a new store, raise the floor); an interior entry
    ///     shifts the tail into fresh arrays and repairs the shifted entries' node indices — O(n) in the tail length.
    /// </summary>
    /// <param name="key">The key to remove (already resolved by the caller).</param>
    /// <param name="idx">The entry's current position (consistent under the held lock).</param>
    /// <remarks>The caller must hold <see cref="_writeLock" />. The map removal leads the store publish (the same skew direction as adds), and every allocation precedes the map removal (strand-proofing).</remarks>
    private void RemoveCoreUnderLock(TKey key, int idx)
    {
        Store s = _store;
        int n = s._count;

        if (idx == n - 1)
        {
            // Tail removal: nothing shifts, so share the arrays and only lower the count — via a NEW store (a
            // published count may only grow). The floor rises to the old high-water mark so a later append cannot
            // overwrite the vacated slot while an old enumerator can still read it.
            Store next = new(s._keys, s._values, n - 1, Math.Max(s._floor, n));
            _byKey.TryRemove(key, out _);
            _store = next;
            return;
        }

        // Interior removal: fresh arrays so concurrent readers of the old snapshot are undisturbed by the shift.
        var keys = new TKey[s._keys.Length];
        var values = new TValue[s._values.Length];
        Array.Copy(s._keys, keys, idx);                       // keep [0, idx)
        Array.Copy(s._keys, idx + 1, keys, idx, n - idx - 1); // shift (idx, n) down
        Array.Copy(s._values, values, idx);
        Array.Copy(s._values, idx + 1, values, idx, n - idx - 1);
        Store shifted = new(keys, values, n - 1);

        _byKey.TryRemove(key, out _);

        // Repair the node indices of the shifted entries — in-place atomic int stores through the ref seam.
        for (int i = idx; i < n - 1; i++)
        {
            RewriteIndexUnderLock(keys[i], i);
        }

        _store = shifted;
    }

    /// <summary>Rewrites the recorded insertion-order position of <paramref name="key" /> in place (an atomic <c>int</c> store into its hash node — always safe, allocation-free).</summary>
    /// <param name="key">The key whose position changed; expected present.</param>
    /// <param name="index">The key's new position.</param>
    /// <remarks>The caller must hold <see cref="_writeLock" />.</remarks>
    private void RewriteIndexUnderLock(TKey key, int index)
    {
        ref int idxRef = ref _byKey.GetValueRefOrNullRef(key);
        Debug.Assert(!Unsafe.IsNullRef(ref idxRef), "a key on the list path is missing from the key map under the write lock");
        if (!Unsafe.IsNullRef(ref idxRef))
        {
            idxRef = index;
        }
    }

    /// <summary>Allocates a <paramref name="capacity" />-sized array and copies the first <paramref name="count" /> elements of <paramref name="source" /> into it (the copy-on-write / growth primitive).</summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="source">The array to copy from.</param>
    /// <param name="capacity">The new array's length; must be at least <paramref name="count" />.</param>
    /// <param name="count">The number of live elements to carry over.</param>
    /// <returns>The fresh array.</returns>
    private static T[] CopyArray<T>(T[] source, int capacity, int count)
    {
        var fresh = new T[capacity];
        Array.Copy(source, fresh, count);
        return fresh;
    }

    /// <summary>Throws <see cref="ArgumentNullException" /> if <paramref name="key" /> is null (guards the reference-key case; the <c>typeof</c> guard keeps value-type keys allocation-free even in Debug).</summary>
    /// <param name="key">The key to validate.</param>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    private static void NullCheck(TKey key)
    {
        if (!typeof(TKey).IsValueType && key is null)
        {
            throw new ArgumentNullException(nameof(key));
        }
    }

    /// <summary>Refuses a write issued by the thread already inside this collection's write lock — user code the collection runs under the lock (a predicate, a comparer). <see cref="Monitor" /> is reentrant, so without this such a write would silently interleave two half-applied mutations; turning it into a deterministic exception is the point.</summary>
    /// <exception cref="LockRecursionException">The current thread already holds the write lock.</exception>
    private void ThrowIfReentrantWrite()
    {
        if (Monitor.IsEntered(_writeLock))
        {
            throw new LockRecursionException(
                "ConcurrentPointingDict does not support reentrant writes: a mutating operation was invoked from user " +
                "code running inside the collection's write lock (a RemoveWhere predicate or a key comparer). Perform " +
                "the mutation after the enclosing operation returns.");
        }
    }
}
