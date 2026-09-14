#nullable enable

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using NumSharp.Collections.Concurrent;

namespace NumSharp.Collections;

/// <summary>
///     A thread-safe collection with <b>two access paths over one set of entries</b>: a dictionary path keyed
///     by <typeparamref name="TKey" /> (uniqueness, dedup and O(1) lookup) and a list path addressed by
///     insertion-order <c>int</c> index. Enumeration, <see cref="ToArray" /> and <see cref="Values" /> yield the
///     <typeparamref name="TValue" />s in index order, so the type reads like a <see cref="List{T}" /> of values
///     that also happens to be keyed. It is the concurrent counterpart of an insertion-ordered dictionary.
/// </summary>
/// <remarks>
///     <para>
///         <b>Why this exists.</b> NumSharp needs a concurrent, insertion-ordered, index-addressable map:
///         a <see cref="System.Collections.Concurrent.ConcurrentDictionary{TKey,TValue}" /> gives thread-safe
///         key access but no order and no positional access, while an insertion-ordered dictionary gives
///         order but is not thread-safe. This type composes the two: the key path is a vendored clone of dotnet's
///         <see cref="Concurrent.ConcurrentDictionary{TKey,TValue}" /> (lock-free reads); the list
///         path is a <b>contiguous</b> value/key snapshot published through a single volatile field.
///     </para>
///     <para>
///         <b>How the two paths are wired.</b> The key path maps each key to a <see cref="ValueIndex" /> — the
///         value and the entry's current insertion-order position — stored <b>inline in the hash-table node</b>,
///         so a key lookup costs exactly one node visit, the same memory walk as the plain concurrent dictionary
///         (an earlier design interposed a separate heap <c>Entry</c> object and paid a second dependent cache
///         miss per lookup; see ConcurrentOrderedDict.TODO.md, Gap B1). The list path is a pair of contiguous
///         arrays (values and keys, in order); enumeration, <see cref="ToArray" />, <see cref="this[int]" /> and
///         <see cref="CopyTo" /> scan them directly, which is what keeps them close to <see cref="List{T}" />.
///         A write updates both representations, so both paths observe the same value.
///     </para>
///     <para>
///         <b>The <c>int</c>-key footgun.</b> Both <see cref="this[TKey]" /> and <see cref="this[int]" /> exist.
///         When <typeparamref name="TKey" /> is <c>int</c> they collide: the C# compiler binds <c>d[5]</c> to the
///         <i>index</i> indexer (the concrete <c>int</c> overload wins over the generic one), so the key path for
///         <c>int</c> keys must go through methods (<see cref="TryGetValue" />, <see cref="GetByKey" />,
///         <see cref="SetByKey" />). For every other key type the two brackets disambiguate by argument type.
///     </para>
///     <para>
///         <b>Performance.</b> Reads are lock-free: lookup by key, membership, <see cref="IndexOf" />, positional
///         get, <see cref="Count" />, enumeration and <see cref="ToArray" /> take no lock. Key lookups run the
///         plain concurrent dictionary's own bucket walk with the value inline, so they match it; enumeration,
///         <see cref="ToArray" /> and the <see cref="Snapshot" /> view scan a contiguous array and match
///         <see cref="List{T}" /> (the per-call <see cref="this[int]" /> keeps one volatile read of overhead — use
///         the view for hot scans). Insertion is amortized O(1) and allocates only the hash node (appends reuse
///         the published arrays and republish a count, not a fresh snapshot object). <b>Writes are serialized</b>
///         by a single lock because a global insertion order and the contiguous cache cannot be maintained
///         per-stripe; that is the deliberate cost of the ordering. The lock is held only for the structural
///         mutation and value swap itself — never while a user <c>valueFactory</c> or <c>updateValueFactory</c>
///         runs (those execute outside it, as in the framework dictionary). <b>Order-preserving removal of an
///         interior entry is O(n)</b> (like <see cref="List{T}.RemoveAt" />): it shifts the trailing entries down
///         into fresh arrays to keep indices contiguous. Removing the <b>last</b> entry is O(1) (no shift), and
///         <see cref="TryRemoveSwapBack" /> offers an explicitly order-breaking O(1) removal for any position.
///         <b>Replacing the value of an existing key is O(1) when <typeparamref name="TValue" /> is written
///         atomically</b> (reference types and machine-word primitives) and O(n) otherwise (a non-atomically
///         writable value type forces a fresh value array so lock-free readers never see a torn value).
///     </para>
///     <para>
///         <b>Memory.</b> Each value is held twice — once inline in the hash node (for the lock-free key path)
///         and once in the contiguous cache (for the List-like list path). This denormalization is what buys both
///         a one-hop key path and contiguous enumeration at once. Because removal never scrubs array slots (a
///         concurrent enumerator may still be reading them), a removed entry's key/value can stay reachable until
///         its slot is overwritten by a later append or the arrays are replaced — call <see cref="Clear" /> or
///         let interior removals compact if that retention matters for large reference values.
///     </para>
///     <para>
///         <b>Consistency.</b> A lock-free reader always sees valid, untorn keys and values and never corrupts
///         state. In the absence of concurrent <i>writers</i> the key and list paths are exactly consistent.
///         While a writer runs, a reader may briefly observe a just-added or just-removed element through the key
///         path a hair before the list path (writes touch the key map first — the skew is directional: an entry
///         visible on the list path is always already visible on the key path), an <see cref="IndexOf" /> off by
///         concurrent removals, or (for atomically-written values) a value update in one path before the other —
///         snapshot semantics, in the spirit of the plain concurrent dictionary. An enumerator captures the value
///         array and count when enumeration begins and structural changes never rewrite the slots it can see; the
///         one documented exception is <see cref="TryRemoveSwapBack" />, which trades that purity for O(1)
///         removal (a concurrent enumerator can see the moved element at both its old and new position).
///     </para>
///     <para>
///         <b>Enforced callback contract.</b> User code that this type itself runs <i>inside</i> the write lock —
///         an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key
///         comparer — must not write back into the collection; because <see cref="System.Threading.Monitor" /> is
///         reentrant such a write would interleave two half-applied mutations rather than deadlock, so it is
///         detected and refused with <see cref="System.Threading.LockRecursionException" /> instead of being
///         allowed to corrupt. <see cref="GetOrAdd(TKey, Func{TKey, TValue})" />/<see cref="AddOrUpdate" />
///         factories run <b>outside</b> the lock and may freely call back in. Lock-free reads are always legal
///         from anywhere. Mutations are also exception-tight: every fallible allocation happens before the
///         key map is touched, and batch operations publish in <c>finally</c>, so a throw (from a source
///         enumerator, a predicate, or out-of-memory) always lands the collection in a consistent
///         "applied everything up to the failure" state — never with a key that resolves but does not enumerate.
///     </para>
/// </remarks>
/// <typeparam name="TKey">The non-null key type; uniqueness and lookups use the configured comparer.</typeparam>
/// <typeparam name="TValue">The value type stored per key and yielded, in index order, by enumeration.</typeparam>
public sealed class ConcurrentOrderedDict<TKey, TValue> : IReadOnlyList<TValue>
    where TKey : notnull
{
    /// <summary>The initial ordered-array capacity, and the size the arrays first grow to from empty.</summary>
    /// <remarks>Deliberately small (unlike the key hashtable's prime capacity) because many ordered maps stay tiny.</remarks>
    private const int DefaultCapacity = 4;

    /// <summary>
    ///     The payload stored <b>inline in the key map's hash node</b>: the value plus the entry's current
    ///     insertion-order position. Inlining is the whole key-path speed story — a lookup dereferences exactly
    ///     one node, like the plain concurrent dictionary, instead of chasing a second heap object.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         <b>Tearing contract.</b> Lock-free readers copy this struct from the node with plain loads while a
    ///         (lock-serialized) writer may be storing <i>one</i> of its fields in place through the clone's
    ///         <c>GetValueRefOrNullRef</c> seam. Each in-place store targets a single naturally-aligned field that is
    ///         atomic per ECMA-335 §12.6.6 — <see cref="_index" /> always (an <c>int</c>), <see cref="_value" />
    ///         only when <see cref="ConcurrentDictionaryTypeProps{TValue}.IsWriteAtomic" /> says so — so a
    ///         concurrent copy sees each field old-or-new, never a hybrid of one field. A copy <i>can</i> pair the
    ///         old value with the new index (or vice versa) when two separate serialized writes straddle it;
    ///         that is safe because every lock-free consumer reads exactly one field (<see cref="TryGetValue" />
    ///         the value, <see cref="IndexOf" /> the index) — only lock-holding code reads both.
    ///     </para>
    ///     <para>
    ///         When <typeparamref name="TValue" /> is <b>not</b> atomically writable, the value field is never
    ///         stored in place: a replacement publishes a whole new node via the map indexer (the clone's own
    ///         tear-free discipline for wide values), and only the always-atomic index field is written in place.
    ///     </para>
    /// </remarks>
    private struct ValueIndex
    {
        /// <summary>The value on the key path. Written in place only when atomically writable; otherwise the node is swapped.</summary>
        internal TValue _value;

        /// <summary>The entry's current position in insertion order; rewritten in place (atomic int) when removals shift it.</summary>
        internal int _index;

        /// <summary>Binds a value to an insertion-order position.</summary>
        /// <param name="value">The value to store.</param>
        /// <param name="index">The entry's position in insertion order.</param>
        internal ValueIndex(TValue value, int index)
        {
            _value = value;
            _index = index;
        }
    }

    /// <summary>
    ///     The list path: contiguous keys and values in insertion order. Published through the single volatile
    ///     <see cref="_store" /> field; the live prefix is <c>[0, <see cref="_count" />)</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         <b>The memory model in three rules</b> (they are what make every read path lock-free and every
    ///         enumerator a true snapshot without allocating a fresh holder per append):
    ///     </para>
    ///     <para>
    ///         1. <b><see cref="_count" /> only ever grows on a given instance.</b> An append writes the new slots
    ///         first, then publishes them with a single release store of the count
    ///         (<see cref="Volatile.Write(ref int, int)" />); a reader that acquires the count therefore sees fully
    ///         written slots. Any transition that <i>loses</i> entries (remove, clear, swap-back) publishes a
    ///         <b>new</b> <see cref="Store" /> instead of ever decreasing a count a reader may have captured.
    ///     </para>
    ///     <para>
    ///         2. <b>Slots a reader can see are never structurally rewritten.</b> Appends only touch slots at
    ///         <c>index ≥ count</c>, invisible until the count publish. When arrays are carried into a
    ///         lower-count successor store (an O(1) tail removal), <see cref="_floor" /> records the old high-water
    ///         mark: a later append that would land below it copies the arrays first, so the slot an old
    ///         enumerator still reads is never overwritten. (<see cref="TryRemoveSwapBack" /> deliberately breaks
    ///         this one rule and documents it.)
    ///     </para>
    ///     <para>
    ///         3. <b>Value slots may be rewritten in place only for atomically-writable <typeparamref name="TValue" /></b>
    ///         (that is the documented "a reader may see a value update immediately" semantics); non-atomic value
    ///         replacement clones the value array so no reader can tear.
    ///     </para>
    /// </remarks>
    private sealed class Store
    {
        /// <summary>Keys in insertion order, occupying <c>[0, <see cref="_count" />)</c>. Parallel to <see cref="_values" />.</summary>
        internal readonly TKey[] _keys;

        /// <summary>Values in insertion order, occupying <c>[0, <see cref="_count" />)</c>. The contiguous store that makes the list path fast.</summary>
        internal readonly TValue[] _values;

        /// <summary>
        ///     The first slot an in-place append may write into these arrays. Slots below it were exposed to
        ///     readers by an earlier store generation with a higher count (an O(1) tail removal shares the arrays
        ///     downward), so writing them would corrupt a live enumerator's snapshot — an append below the floor
        ///     must copy the arrays first. 0 for freshly-allocated arrays.
        /// </summary>
        internal readonly int _floor;

        /// <summary>
        ///     The number of live entries. <b>Mutable and monotonically increasing on this instance</b>: appends
        ///     bump it with a release store after writing the slots; every shrinking transition publishes a new
        ///     <see cref="Store" /> instead. Readers must capture it once via <see cref="Volatile.Read(ref int)" />.
        /// </summary>
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

        /// <summary>The shared empty snapshot used before the first insertion and after <see cref="Clear" />. Its count is never mutated: an append into zero capacity always grows into a fresh store.</summary>
        internal static readonly Store Empty = new(Array.Empty<TKey>(), Array.Empty<TValue>(), 0);
    }

    /// <summary>
    ///     The key path: maps each key to its inline <see cref="ValueIndex" />. A vendored clone of dotnet's
    ///     thread-safe dictionary, giving lock-free O(1) key reads and enforcing key uniqueness/dedup.
    /// </summary>
    /// <remarks>
    ///     Constructed with <c>concurrencyLevel: 1</c> on purpose: every mutation of this map happens under
    ///     <see cref="_writeLock" />, so its striped write locks can never be contended — one stripe removes the
    ///     lock-array footprint and makes its whole-table operations (Clear, growth) cheaper. Reads are lock-free
    ///     regardless of stripe count, so the read path loses nothing.
    /// </remarks>
    private readonly ConcurrentDictionary<TKey, ValueIndex> _byKey;

    /// <summary>The list path: the current contiguous ordered snapshot. Volatile so structural publishes reach lock-free readers with acquire/release ordering.</summary>
    private volatile Store _store;

    /// <summary>
    ///     Serializes every mutation so the global insertion order, the contiguous cache and the key map stay
    ///     consistent. Reads never acquire it. It also serializes all writes to <see cref="_byKey" />, which is
    ///     what makes the in-place single-field node writes through the clone's ref seam race-free.
    /// </summary>
    private readonly object _writeLock = new();

    /// <summary>Creates an empty collection using the default comparer for <typeparamref name="TKey" />.</summary>
    public ConcurrentOrderedDict() : this((IEqualityComparer<TKey>?)null)
    {
    }

    /// <summary>Creates an empty collection using the supplied key comparer.</summary>
    /// <param name="comparer">The comparer used for key uniqueness and lookups, or <see langword="null" /> for the default.</param>
    public ConcurrentOrderedDict(IEqualityComparer<TKey>? comparer)
    {
        // concurrencyLevel 1: all map writes are serialized by _writeLock, so extra stripes are pure overhead.
        _byKey = new ConcurrentDictionary<TKey, ValueIndex>(1, 31, comparer);
        _store = Store.Empty;
    }

    /// <summary>Creates an empty collection with room reserved for <paramref name="capacity" /> entries.</summary>
    /// <param name="capacity">The number of entries to pre-size both paths for; must be non-negative.</param>
    /// <param name="comparer">The comparer used for key uniqueness and lookups, or <see langword="null" /> for the default.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="capacity" /> is negative.</exception>
    public ConcurrentOrderedDict(int capacity, IEqualityComparer<TKey>? comparer = null)
    {
        if (capacity < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity));
        }

        // Match the key hashtable's initial capacity to the request, and pre-size the ordered arrays so the first
        // several inserts don't reallocate. concurrencyLevel 1 for the same reason as the default constructor.
        _byKey = new ConcurrentDictionary<TKey, ValueIndex>(1, capacity, comparer);
        _store = capacity == 0 ? Store.Empty : new Store(new TKey[capacity], new TValue[capacity], 0);
    }

    /// <summary>
    ///     Creates a collection populated from <paramref name="source" /> in enumeration order, with Python
    ///     <c>dict(pairs)</c> duplicate semantics (last value wins, first-occurrence position) — matching
    ///     an insertion-ordered dictionary, not the throw-on-duplicate behavior of the framework dictionary.
    /// </summary>
    /// <param name="source">The key/value pairs to copy; repeated keys replace the value without moving the key.</param>
    /// <param name="comparer">The comparer used for key uniqueness and lookups, or <see langword="null" /> for the default.</param>
    /// <remarks>
    ///     When the source can report its length without enumerating (<see cref="Enumerable.TryGetNonEnumeratedCount{TSource}" /> —
    ///     arrays, lists, collections), BOTH representations are pre-sized to it. Pre-sizing the hash map matters as
    ///     much as the arrays for build memory churn: the map's growth re-creates every node it already holds
    ///     (measured: an unsized 100K-entry int build allocates ~2.5× the entry bytes in dead nodes and dead
    ///     doubled arrays; a fully pre-sized build allocates exactly the live bytes).
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="source" /> is <see langword="null" />.</exception>
    public ConcurrentOrderedDict(IEnumerable<KeyValuePair<TKey, TValue>> source, IEqualityComparer<TKey>? comparer = null)
        : this(PreSizeOf(source), comparer)
    {
        AddRange(source);
    }

    /// <summary>Reads a capacity hint from a countable pairs source without enumerating it (0 when unknown — the paths grow as usual).</summary>
    /// <param name="source">The constructor's source; a null slips through as 0 so <see cref="AddRange" /> can raise the contract's <see cref="ArgumentNullException" />.</param>
    /// <returns>The source's count when knowable without enumeration; otherwise 0.</returns>
    private static int PreSizeOf(IEnumerable<KeyValuePair<TKey, TValue>>? source)
        => source is not null && source.TryGetNonEnumeratedCount(out int count) ? count : 0;

    /// <summary>Gets the number of entries — a moment-in-time snapshot, consistent with lock-free enumeration.</summary>
    public int Count => Volatile.Read(ref _store._count);

    /// <summary>Gets whether the collection currently has no entries.</summary>
    public bool IsEmpty => Count == 0;

    /// <summary>Gets the comparer used to determine key equality and hashing.</summary>
    public IEqualityComparer<TKey> Comparer => _byKey.Comparer;

    // ---------------------------------------------------------------------------------------------------------
    // Key path (dictionary semantics). For int keys, reach these through the named methods, not the indexer.
    // ---------------------------------------------------------------------------------------------------------

    /// <summary>
    ///     Gets or sets the value for <paramref name="key" />. Get throws if the key is absent; set updates the
    ///     value in place when the key exists, otherwise appends a new entry at the end of insertion order.
    /// </summary>
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

    /// <summary>Reads the value for <paramref name="key" />, throwing if absent — the method form of the key getter (use this for <c>int</c> keys). Lock-free and O(1).</summary>
    /// <param name="key">The key to look up.</param>
    /// <returns>The value associated with <paramref name="key" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="KeyNotFoundException"><paramref name="key" /> is absent.</exception>
    public TValue GetByKey(TKey key)
    {
        if (!TryGetValue(key, out var value))
        {
            throw new KeyNotFoundException(Concurrent.SR.Format(Concurrent.SR.Arg_KeyNotFoundWithKey, key.ToString()));
        }

        return value;
    }

    /// <summary>
    ///     Upserts <paramref name="key" />: updates its value if present (keeping its position), otherwise appends
    ///     a new entry — the method form of the key setter (use this for <c>int</c> keys). Serialized with other writes.
    /// </summary>
    /// <param name="key">The key to write.</param>
    /// <param name="value">The value to store.</param>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public void SetByKey(TKey key, TValue value)
    {
        NullCheck(key);

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            // One bucket walk decides the branch AND hands back the in-place write target for the update case.
            ref ValueIndex vi = ref _byKey.GetValueRefOrNullRef(key);
            if (!Unsafe.IsNullRef(ref vi))
            {
                ReplaceExistingUnderLock(key, ref vi, value);
            }
            else
            {
                bool added = TryAppendUnderLock(key, value);
                Debug.Assert(added, "key vanished between the ref lookup and the append while holding the write lock");
            }
        }
    }

    /// <summary>Determines whether <paramref name="key" /> is present. Lock-free, O(1) and allocation-free.</summary>
    /// <param name="key">The key to test.</param>
    /// <returns><see langword="true" /> if the key exists; otherwise <see langword="false" />.</returns>
    /// <remarks>Routed through the ref seam rather than the vendored map's own <c>ContainsKey</c> so the membership test is allocation-free in Debug builds too (the vendored method's verbatim <c>key is null</c> boxes value-type keys under unoptimized codegen — see <see cref="NullCheck" />).</remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool ContainsKey(TKey key) => !Unsafe.IsNullRef(ref _byKey.GetValueRefOrNullRef(key));

    /// <summary>Attempts to read the value for <paramref name="key" /> without throwing. Lock-free and O(1) — one hash-node visit, like the plain concurrent dictionary.</summary>
    /// <param name="key">The key to look up.</param>
    /// <param name="value">On success, the associated value; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if the key was found; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] // measured: the un-inlined wrapper alone cost ~25% over the raw map lookup
    public bool TryGetValue(TKey key, out TValue value)
    {
        // Read through the node ref instead of copying the whole ValueIndex struct: one field load, and the
        // index half is never touched. Lock-free ref READS are safe where ref writes are not: an atomic TValue
        // is stored in place with a single atomic store (we read old-or-new), a non-atomic TValue is never
        // stored in place (node swap — the node we hold a ref into keeps its immutable old value, and the GC
        // keeps it alive for us).
        ref ValueIndex vi = ref _byKey.GetValueRefOrNullRef(key);
        if (!Unsafe.IsNullRef(ref vi))
        {
            value = vi._value;
            return true;
        }

        value = default!;
        return false;
    }

    /// <summary>Adds <paramref name="key" />/<paramref name="value" /> only if the key is absent, appending it at the end of order.</summary>
    /// <param name="key">The key to add.</param>
    /// <param name="value">The value to add.</param>
    /// <returns><see langword="true" /> if the entry was added; <see langword="false" /> if the key already existed.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public bool TryAdd(TKey key, TValue value)
    {
        NullCheck(key);

        // No lock-free pre-check: like the framework dictionary's TryAdd, the single map walk inside the append
        // answers "already exists" itself. A pre-check would cost a full extra bucket walk on the common
        // absent-key path (a build loop) to save only a brief uncontended lock acquisition on the duplicate path.
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
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public void Add(TKey key, TValue value)
    {
        if (!TryAdd(key, value))
        {
            throw new ArgumentException(Concurrent.SR.ConcurrentDictionary_KeyAlreadyExisted, nameof(key));
        }
    }

    /// <summary>
    ///     Upserts every pair from <paramref name="source" /> under <b>one</b> lock acquisition, with the same
    ///     duplicate semantics as the enumerable constructor (existing keys get the new value in their existing
    ///     position; new keys append in enumeration order). This is the fast way to bulk-load: it amortizes the
    ///     lock, the array growth and the snapshot publication over the whole batch, and newly appended entries
    ///     become visible to the list path <b>atomically as one batch</b> (a concurrent enumerator sees either
    ///     none or all of them).
    /// </summary>
    /// <param name="source">The key/value pairs to upsert; consumed once, in order.</param>
    /// <remarks>
    ///     <para>
    ///         The source is enumerated <b>inside</b> the write lock (that is what makes the batch atomic), so it
    ///         must not call back into this collection's write operations — such a callback is detected and
    ///         refused with <see cref="LockRecursionException" /> (a reentrant structural write would otherwise
    ///         corrupt the batch in progress). Feeding it a materialized array or list is the intended use.
    ///     </para>
    ///     <para>
    ///         Batch atomicity is a <b>list-path</b> guarantee (appended entries become enumerable together).
    ///         The key path leads per pair, as it does for every write: a concurrent
    ///         <see cref="TryGetValue" />/<see cref="ContainsKey" /> may see a pair — including a duplicate key's
    ///         replacement value — before the batch publishes to the list path.
    ///     </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="source" /> or a key inside it is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public void AddRange(IEnumerable<KeyValuePair<TKey, TValue>> source)
    {
        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            Store s = _store;
            int count = s._count; // plain read is fine: every count writer holds this lock
            TKey[] keys = s._keys;
            TValue[] values = s._values;

            // `fresh` = we are building private arrays invisible to readers, so plain writes to any slot are
            // legal and the batch publishes as one new Store at the end. While NOT fresh, appends write only
            // slots ≥ count (invisible until the final count publish) and must respect the floor rule.
            bool fresh = false;

            // Pre-size in one step when the source can tell us its length — this is the whole point of the
            // batch API for the "build once" pattern (single grow instead of log2(n) doublings).
            if (source.TryGetNonEnumeratedCount(out int incoming) && incoming > 0)
            {
                long needed = (long)count + incoming;
                if (needed > values.Length || count < s._floor)
                {
                    // Two DIFFERENT reasons to build fresh arrays, needing DIFFERENT capacities (the same split
                    // TryAppendUnderLock makes). Genuinely out of room (needed > capacity) → amortized-double so a
                    // build-once loop pays log2(n) grows, not n. But a copy forced ONLY by the floor rule
                    // (count < floor while needed <= capacity) already has room: growing there would DOUBLE the
                    // capacity on every batch that follows a tail removal/swap-back (which raises the floor), and
                    // since the floor is re-raised by the next removal the doublings compound without bound —
                    // the buffer balloons (4K→…→hundreds of millions) while the live count stays flat, ending in
                    // OutOfMemory. Keep the capacity when the copy is floor-driven.
                    int newCap = needed > values.Length
                        ? (int)Math.Min(Math.Max(needed, values.Length == 0 ? DefaultCapacity : (long)values.Length * 2), Array.MaxLength)
                        : values.Length;
                    keys = CopyArray(s._keys, newCap, count);
                    values = CopyArray(s._values, newCap, count);
                    fresh = true;
                }
            }

            try
            {
                foreach (KeyValuePair<TKey, TValue> kv in source)
                {
                    NullCheck(kv.Key);

                    // Copy/grow BEFORE the map insert (mirroring TryAppendUnderLock's strand-proofing): if the
                    // allocation throws, the map was not touched for this pair, so the finally-publish below
                    // leaves both paths exactly consistent. The copy is wasted only when this pair turns out to
                    // be a duplicate exactly at a capacity/floor boundary — rare and harmless (the arrays carry
                    // the same live content either way).
                    if (count == values.Length || (!fresh && count < s._floor))
                    {
                        // Same split as the pre-size step above: double only when actually full; a copy forced
                        // only by the floor rule (count < floor, but a slot is still free) keeps its capacity, so
                        // repeated post-removal batches cannot compound the buffer upward without bound.
                        int newCap = count == values.Length
                            ? (values.Length == 0 ? DefaultCapacity : (int)Math.Min((long)values.Length * 2, Array.MaxLength))
                            : values.Length;
                        keys = CopyArray(keys, newCap, count);
                        values = CopyArray(values, newCap, count);
                        fresh = true;
                    }

                    if (_byKey.TryAdd(kv.Key, new ValueIndex(kv.Value, count)))
                    {
                        keys[count] = kv.Key;
                        values[count] = kv.Value;
                        count++;
                    }
                    else
                    {
                        // Existing key: replace the value, keep the position (Python dict(pairs) semantics).
                        ref ValueIndex vi = ref _byKey.GetValueRefOrNullRef(kv.Key);
                        Debug.Assert(!Unsafe.IsNullRef(ref vi), "TryAdd said the key exists but the ref lookup missed it under the write lock");
                        int idx = vi._index;

                        if (ConcurrentDictionaryTypeProps<TValue>.IsWriteAtomic)
                        {
                            vi._value = kv.Value; // in-place atomic field store; readers see old or new, never torn
                        }
                        else
                        {
                            // Non-atomic value: never write a live array slot in place — if we are still on the
                            // published arrays, go copy-on-write for the rest of the batch, and do it BEFORE the
                            // node swap so an allocation failure cannot leave the key path holding a value the
                            // list path will never publish. Then swap the node (the clone's tear-free discipline
                            // for wide values).
                            if (!fresh)
                            {
                                keys = CopyArray(keys, keys.Length, count);
                                values = CopyArray(values, values.Length, count);
                                fresh = true;
                            }

                            _byKey[kv.Key] = new ValueIndex(kv.Value, idx);
                        }

                        if (idx < count)
                        {
                            values[idx] = kv.Value; // private array when fresh; documented in-place value update otherwise
                        }
                    }
                }
            }
            finally
            {
                // Publish once — including when the source enumerator or a null key threw mid-batch: the pairs
                // already inserted into the key map MUST reach the list path too, or the two paths stay
                // permanently inconsistent (a key that resolves but never enumerates). On the in-place path a
                // single release store of the count makes every appended slot visible together; on the fresh
                // path the volatile Store publish does the same — either way the batch is one atomic step.
                if (fresh)
                {
                    _store = new Store(keys, values, count);
                }
                else if (count != s._count)
                {
                    Volatile.Write(ref s._count, count);
                }
            }
        }
    }

    /// <summary>Returns the existing value for <paramref name="key" />, or appends and returns <paramref name="value" /> if absent.</summary>
    /// <param name="key">The key to read or add.</param>
    /// <param name="value">The value to append if the key is absent.</param>
    /// <returns>The existing value, or the newly added <paramref name="value" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public TValue GetOrAdd(TKey key, TValue value)
    {
        // Lock-free hit is the hot path for a cache-shaped workload (ref-read of the value field only — see
        // TryGetValue for the race-safety argument).
        ref ValueIndex hit = ref _byKey.GetValueRefOrNullRef(key);
        if (!Unsafe.IsNullRef(ref hit))
        {
            return hit._value;
        }

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            if (TryAppendUnderLock(key, value))
            {
                return value;
            }

            // Lost the race for this key between the lock-free miss and the lock: read the winner's value.
            bool got = _byKey.TryGetValue(key, out ValueIndex vi);
            Debug.Assert(got, "append failed for an existing key that then vanished while holding the write lock");
            return vi._value;
        }
    }

    /// <summary>Returns the existing value for <paramref name="key" />, or appends the result of <paramref name="valueFactory" /> if absent.</summary>
    /// <param name="key">The key to read or add.</param>
    /// <param name="valueFactory">Invoked with the key to produce the value when the key is absent. Runs <b>outside</b> the write lock (writers are never blocked on it); under contention it may be called on more than one thread, like the framework concurrent dictionary, with the losing result discarded.</param>
    /// <returns>The existing value, or the newly produced and added value.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> or <paramref name="valueFactory" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public TValue GetOrAdd(TKey key, Func<TKey, TValue> valueFactory)
    {
        NullCheck(key);
        if (valueFactory is null)
        {
            throw new ArgumentNullException(nameof(valueFactory));
        }

        ref ValueIndex hit = ref _byKey.GetValueRefOrNullRef(key);
        if (!Unsafe.IsNullRef(ref hit))
        {
            return hit._value;
        }

        // Run the factory OUTSIDE the lock so writers are never blocked on user code. As in the framework
        // dictionary, under contention the factory may run on several threads; only one result is kept.
        TValue value = valueFactory(key);

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            if (TryAppendUnderLock(key, value))
            {
                return value;
            }

            bool got = _byKey.TryGetValue(key, out ValueIndex vi);
            Debug.Assert(got, "append failed for an existing key that then vanished while holding the write lock");
            return vi._value; // lost the race for this key; discard the value we computed
        }
    }

    /// <summary>Appends <paramref name="addValue" /> if the key is absent, otherwise replaces the value via <paramref name="updateValueFactory" />.</summary>
    /// <param name="key">The key to add or update.</param>
    /// <param name="addValue">The value to append if the key is absent.</param>
    /// <param name="updateValueFactory">Invoked with the key and current value to produce the replacement. Runs <b>outside</b> the write lock (writers are never blocked on it); the result is applied with a compare-and-swap and, like the framework concurrent dictionary, the factory may run more than once under contention.</param>
    /// <returns>The value now stored for the key (the added or the updated value).</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> or <paramref name="updateValueFactory" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public TValue AddOrUpdate(TKey key, TValue addValue, Func<TKey, TValue, TValue> updateValueFactory)
    {
        NullCheck(key);
        if (updateValueFactory is null)
        {
            throw new ArgumentNullException(nameof(updateValueFactory));
        }

        // Match the framework dictionary: the factory runs OUTSIDE the lock (writers are never blocked on user
        // code), and the update is applied with a compare-and-swap, retried if the value moved under us. The
        // lock is entered only to append an absent key or (via TryUpdate) to perform the swap itself.
        while (true)
        {
            // Ref-read of the value field only (see TryGetValue for the race-safety argument); seam-routed so
            // the lock-free fast path is allocation-free in Debug too.
            ref ValueIndex vi = ref _byKey.GetValueRefOrNullRef(key);
            if (!Unsafe.IsNullRef(ref vi))
            {
                TValue old = vi._value;
                TValue updated = updateValueFactory(key, old);
                if (TryUpdate(key, updated, old))
                {
                    return updated;
                }

                continue; // the value changed, or the key was removed, before our swap; recompute and retry
            }

            ThrowIfReentrantWrite();

            lock (_writeLock)
            {
                if (TryAppendUnderLock(key, addValue))
                {
                    return addValue;
                }
            }

            // The key was added concurrently; loop back and take the update path.
        }
    }

    /// <summary>Replaces the value for <paramref name="key" /> only if its current value equals <paramref name="comparisonValue" />.</summary>
    /// <param name="key">The key whose value is conditionally replaced.</param>
    /// <param name="newValue">The replacement value.</param>
    /// <param name="comparisonValue">The value the current value must equal (by the default value comparer) for the update to occur.</param>
    /// <returns><see langword="true" /> if the value was replaced; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public bool TryUpdate(TKey key, TValue newValue, TValue comparisonValue)
    {
        NullCheck(key);

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            ref ValueIndex vi = ref _byKey.GetValueRefOrNullRef(key);
            if (!Unsafe.IsNullRef(ref vi) && EqualityComparer<TValue>.Default.Equals(vi._value, comparisonValue))
            {
                ReplaceExistingUnderLock(key, ref vi, newValue);
                return true;
            }

            return false;
        }
    }

    /// <summary>
    ///     Removes <paramref name="key" /> and its value. Removing the <b>last</b> entry is O(1); removing an
    ///     interior entry shifts later entries down to keep indices contiguous — <b>O(n)</b> in the number of
    ///     entries after the removed one (like <see cref="List{T}.RemoveAt" />). For bulk removal use
    ///     <see cref="RemoveWhere" />; for order-insensitive O(1) removal use <see cref="TryRemoveSwapBack" />.
    /// </summary>
    /// <param name="key">The key to remove.</param>
    /// <param name="value">On success, the removed value; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if an entry was removed; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public bool TryRemove(TKey key, out TValue value)
    {
        NullCheck(key);

        // Lock-free fast negative: the interior-removal work under the lock is O(n), so keeping absent-key
        // callers out of the lock entirely is worth the one extra bucket walk on the present path (noise there).
        // Seam-routed so the probe is allocation-free in Debug too (see NullCheck).
        if (Unsafe.IsNullRef(ref _byKey.GetValueRefOrNullRef(key)))
        {
            value = default!;
            return false;
        }

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            ref ValueIndex vi = ref _byKey.GetValueRefOrNullRef(key);
            if (Unsafe.IsNullRef(ref vi))
            {
                value = default!;
                return false;
            }

            // Reading both fields of the struct is safe here (unlike lock-free readers): writes are serialized
            // by the lock we hold, so the pair is consistent.
            value = vi._value;
            RemoveCoreUnderLock(key, vi._index);
            return true;
        }
    }

    /// <summary>Removes <paramref name="key" /> if present. Convenience wrapper over <see cref="TryRemove" />; same O(1)-tail / O(n)-interior cost.</summary>
    /// <param name="key">The key to remove.</param>
    /// <returns><see langword="true" /> if an entry was removed; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public bool Remove(TKey key) => TryRemove(key, out _);

    /// <summary>
    ///     Removes <paramref name="key" /> in O(1) by <b>moving the current last entry into the removed slot</b> —
    ///     the order-breaking escape hatch from the O(n) shifting removal. Only the moved entry's position
    ///     changes; every other entry keeps its index.
    /// </summary>
    /// <param name="key">The key to remove.</param>
    /// <param name="value">On success, the removed value; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if an entry was removed; otherwise <see langword="false" />.</returns>
    /// <remarks>
    ///     <para><b>This call gives up two guarantees the ordinary removal keeps</b> — use it only when neither matters:</para>
    ///     <para>
    ///         1. <b>Order:</b> the last entry is relocated to the removed entry's index, so insertion order is no
    ///         longer the enumeration order for that entry.
    ///     </para>
    ///     <para>
    ///         2. <b>Snapshot purity of concurrent enumerators</b> (only when both <typeparamref name="TKey" />
    ///         and <typeparamref name="TValue" /> are atomically writable, where the swap is done in place): an
    ///         enumerator already running over the same arrays can observe the moved entry at both its old and
    ///         its new position, and will not observe the removed entry at all. Keys and values are still never
    ///         torn (the in-place swap is taken only when both types are single-store atomic). Additionally,
    ///         because the swapped slot's key and value are two separate atomic stores, a reader that COMBINES
    ///         that one slot's key and value during the swap window (<see cref="GetKeyAt" /> +
    ///         <see cref="TryGetAt" />, or <see cref="Pairs" /> passing the slot) can transiently see the removed
    ///         entry's key paired with the moved entry's value or vice versa — each individual datum is real and
    ///         untorn, only the pairing at that single slot is momentarily mixed, and the key path
    ///         (<see cref="TryGetValue" />) is never affected. When either type is not atomically writable the
    ///         swap copies the arrays instead — pure snapshots and pure pairs, O(n) memcpy, but still no
    ///         per-entry re-indexing.
    ///     </para>
    ///     <para>
    ///         Code that asserts key↔value pairing invariants across the LIST path while racing removals must
    ///         therefore use the order-preserving <see cref="TryRemove" />/<see cref="RemoveWhere" /> (whose
    ///         fresh-array discipline keeps every published pair exact) — that is the contract split the two
    ///         removal families exist for.
    ///     </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public bool TryRemoveSwapBack(TKey key, out TValue value)
    {
        NullCheck(key);

        if (Unsafe.IsNullRef(ref _byKey.GetValueRefOrNullRef(key)))
        {
            value = default!;
            return false; // lock-free fast negative (seam-routed: allocation-free in Debug too)
        }

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            ref ValueIndex vi = ref _byKey.GetValueRefOrNullRef(key);
            if (Unsafe.IsNullRef(ref vi))
            {
                value = default!;
                return false;
            }

            value = vi._value;
            int idx = vi._index;
            Store s = _store;
            int n = s._count;

            if (idx == n - 1)
            {
                // Removing the last entry needs no move at all — identical to the tail fast path of TryRemove.
                // Successor allocated BEFORE the map removal (strand-proofing: after the key stops resolving,
                // only throw-free operations remain).
                Store next = new(s._keys, s._values, n - 1, Math.Max(s._floor, n));
                _byKey.TryRemove(key, out _); // key path first (same skew direction as every removal)
                _store = next;
                return true;
            }

            TKey movedKey = s._keys[n - 1];
            TValue movedValue = s._values[n - 1];

            if (ConcurrentDictionaryTypeProps<TKey>.IsWriteAtomic && ConcurrentDictionaryTypeProps<TValue>.IsWriteAtomic)
            {
                // In-place swap: single atomic stores per slot, so concurrent readers can never tear a key or a
                // value — they can only see the documented duplicate/missing anomaly for this one entry, plus
                // the transiently mixed key/value pairing at this one slot (two separate stores; see remarks).
                Store next = new(s._keys, s._values, n - 1, Math.Max(s._floor, n)); // alloc before any mutation
                _byKey.TryRemove(key, out _); // key path first (same skew direction as every removal)
                s._keys[idx] = movedKey;
                s._values[idx] = movedValue;

                // Publish the shrunken snapshot; the floor freezes the vacated last slot against in-place reuse
                // so an enumerator that captured count == n keeps reading valid (old) data there.
                RewriteIndexUnderLock(movedKey, idx);
                _store = next;
                return true;
            }
            else
            {
                // A wide key or value would tear under an in-place slot write, so degrade to copying arrays:
                // still order-breaking and re-index-free, but snapshot-pure and O(n) memcpy. Allocations happen
                // before the map removal (strand-proofing, as above).
                var keys = new TKey[s._keys.Length];
                var values = new TValue[s._values.Length];
                Array.Copy(s._keys, keys, n - 1);
                Array.Copy(s._values, values, n - 1);
                keys[idx] = movedKey;
                values[idx] = movedValue;
                Store next = new(keys, values, n - 1);
                _byKey.TryRemove(key, out _); // key path first (same skew direction as every removal)
                RewriteIndexUnderLock(movedKey, idx);
                _store = next;
                return true;
            }
        }
    }

    /// <summary>
    ///     Removes every entry matching <paramref name="match" /> in <b>one compaction pass</b> — the bulk form of
    ///     removal that costs O(n) total instead of O(n) per removed entry, and publishes a single new snapshot.
    /// </summary>
    /// <param name="match">Returns <see langword="true" /> for entries to remove; receives each key and its current value in index order.</param>
    /// <returns>The number of entries removed.</returns>
    /// <remarks>
    ///     The predicate runs <b>inside</b> the write lock (that is what makes the removal one atomic pass), so it
    ///     must be fast and must not call back into this collection's write operations — such a callback is
    ///     detected and refused with <see cref="LockRecursionException" /> (it would otherwise corrupt the pass in
    ///     progress). A predicate that throws is survivable: the pass lands in the consistent "removed everything
    ///     matched so far" state and the exception propagates.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="match" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
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

            // Compact the survivors into fresh arrays (readers keep the old snapshot), fixing each survivor's
            // recorded index in place as it lands on a new position.
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
                // A throwing predicate must not strand the pass half-applied: keys already removed from the map
                // would otherwise stay enumerable forever. Carry every not-yet-processed entry — INCLUDING the
                // one whose predicate threw (it was neither removed nor copied) — over as a survivor and publish,
                // so the collection lands in the consistent "removed everything matched so far" state whether
                // the pass completed (i == n: this loop is a no-op) or threw.
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
    // List path (index semantics). Enumeration and ToArray are here too, yielding values in index order.
    // ---------------------------------------------------------------------------------------------------------

    /// <summary>
    ///     Gets or sets the value at insertion-order position <paramref name="index" />. The setter takes a
    ///     <typeparamref name="TValue" /> and replaces the value at that position, leaving its key and position
    ///     unchanged (it never inserts). This indexer wins over <see cref="this[TKey]" /> when <typeparamref name="TKey" /> is <c>int</c>.
    /// </summary>
    /// <param name="index">The zero-based position in insertion order.</param>
    /// <value>The value at <paramref name="index" />.</value>
    /// <remarks>The getter pays one volatile snapshot read per call; for a hot scan take <see cref="Snapshot" /> once and index the view — that is the <see cref="List{T}" />-parity path.</remarks>
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
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
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

            ref ValueIndex vi = ref _byKey.GetValueRefOrNullRef(s._keys[index]);
            Debug.Assert(!Unsafe.IsNullRef(ref vi), "a live list slot's key is missing from the key map under the write lock");
            if (!Unsafe.IsNullRef(ref vi))
            {
                ReplaceExistingUnderLock(s._keys[index], ref vi, value);
            }
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

    /// <summary>
    ///     Returns the insertion-order position of <paramref name="key" />, or -1 if absent — the "TKey to index"
    ///     mapping. Lock-free and O(1).
    /// </summary>
    /// <param name="key">The key to locate.</param>
    /// <returns>The zero-based index of the key, or -1 if it is not present.</returns>
    /// <remarks>Under a concurrent structural writer the returned index is a snapshot and may be stale by the number of concurrent removals.</remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public int IndexOf(TKey key)
    {
        // Same ref-read pattern as TryGetValue: only the (always atomically written) index field is loaded.
        ref ValueIndex vi = ref _byKey.GetValueRefOrNullRef(key);
        return Unsafe.IsNullRef(ref vi) ? -1 : vi._index;
    }

    /// <summary>Attempts to map <paramref name="key" /> to its insertion-order position without allocating a sentinel.</summary>
    /// <param name="key">The key to locate.</param>
    /// <param name="index">On success, the key's index; otherwise -1.</param>
    /// <returns><see langword="true" /> if the key was found; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool TryGetIndex(TKey key, out int index)
    {
        // Ref-read of the single (atomic int) index field — see TryGetValue for why this is race-safe.
        ref ValueIndex vi = ref _byKey.GetValueRefOrNullRef(key);
        if (!Unsafe.IsNullRef(ref vi))
        {
            index = vi._index;
            return true;
        }

        index = -1;
        return false;
    }

    /// <summary>Removes the entry at position <paramref name="index" />, shifting later entries down. O(1) for the last position, <b>O(n)</b> otherwise.</summary>
    /// <param name="index">The zero-based position in insertion order to remove.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
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
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock — an <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public void Clear()
    {
        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            _byKey.Clear();
            _store = Store.Empty;
        }
    }

    /// <summary>
    ///     Copies the values, in index order, into a new array — the same content and order as enumeration, close
    ///     to <see cref="List{T}.ToArray" /> in cost (a contiguous block copy).
    /// </summary>
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

    /// <summary>Copies the values, in index order, into <paramref name="array" /> starting at <paramref name="arrayIndex" /> (a contiguous block copy).</summary>
    /// <param name="array">The destination array.</param>
    /// <param name="arrayIndex">The zero-based position in <paramref name="array" /> at which to start writing.</param>
    /// <exception cref="ArgumentNullException"><paramref name="array" /> is <see langword="null" />.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="arrayIndex" /> is negative.</exception>
    /// <exception cref="ArgumentException"><paramref name="array" /> has too little room from <paramref name="arrayIndex" />.</exception>
    public void CopyTo(TValue[] array, int arrayIndex)
    {
        if (array is null)
        {
            throw new ArgumentNullException(nameof(array));
        }

        if (arrayIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(arrayIndex));
        }

        Store s = _store;
        int n = Volatile.Read(ref s._count);
        if (array.Length - arrayIndex < n)
        {
            throw new ArgumentException(Concurrent.SR.ConcurrentDictionary_ArrayNotLargeEnough, nameof(array));
        }

        Array.Copy(s._values, 0, array, arrayIndex, n);
    }

    /// <summary>
    ///     Gets a point-in-time array of the keys in insertion order (a contiguous block copy). A fresh array each
    ///     call (snapshot semantics), parallel to <see cref="Values" />.
    /// </summary>
    public TKey[] Keys
    {
        get
        {
            Store s = _store;
            int n = Volatile.Read(ref s._count);
            if (n == 0)
            {
                return Array.Empty<TKey>();
            }

            var keys = new TKey[n];
            Array.Copy(s._keys, keys, n);
            return keys;
        }
    }

    /// <summary>Gets a point-in-time array of the values in insertion order (equivalent to <see cref="ToArray" />).</summary>
    public TValue[] Values => ToArray();

    /// <summary>
    ///     Enumerates the entries as key/value pairs in insertion order — the dictionary-shaped view, for when the
    ///     value-only default enumeration is not enough. A snapshot is taken when enumeration begins.
    /// </summary>
    public IEnumerable<KeyValuePair<TKey, TValue>> Pairs
    {
        get
        {
            Store s = _store;
            TKey[] keys = s._keys;
            TValue[] values = s._values;
            int n = Volatile.Read(ref s._count);
            for (int i = 0; i < n; i++)
            {
                yield return new KeyValuePair<TKey, TValue>(keys[i], values[i]);
            }
        }
    }

    /// <summary>
    ///     Captures the current contents as a <see cref="ValuesView" /> — a zero-allocation snapshot whose indexer
    ///     and spans read the contiguous arrays with <b>no per-access volatile read</b>, which is what closes the
    ///     last gap to <see cref="List{T}" /> for hot index loops (the live <see cref="this[int]" /> pays one
    ///     volatile snapshot read per call by design).
    /// </summary>
    /// <returns>A point-in-time view of the values (and keys) in insertion order.</returns>
    /// <remarks>
    ///     The view is immutable in shape but not in value: entries appended, removed or reordered after the
    ///     capture do not appear, while an in-place value replacement of an atomically-writable
    ///     <typeparamref name="TValue" /> <i>is</i> visible through it (the same live-value semantics enumeration
    ///     has always had). Hold it only as long as the scan runs; it pins the captured arrays against garbage
    ///     collection while referenced.
    /// </remarks>
    public ValuesView Snapshot()
    {
        Store s = _store;
        return new ValuesView(s._keys, s._values, Volatile.Read(ref s._count));
    }

    /// <summary>
    ///     A point-in-time, index-addressable view over the values (and keys) captured by <see cref="Snapshot" />.
    ///     Indexing it costs exactly an array access plus a bounds check — <see cref="List{T}" /> parity — and
    ///     <see cref="AsSpan" /> hands the values to span-based code (vectorized scans, <c>foreach</c>) directly.
    /// </summary>
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
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />. A default-constructed view has count 0, so any index throws rather than null-referencing.</exception>
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

        /// <summary>Returns the captured values as a read-only span — the fastest scan surface (a <c>foreach</c> over it matches an array loop). An empty span for a default-constructed view.</summary>
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

    /// <summary>
    ///     A forward, value-order enumerator bound to the contiguous value array and count captured when it was
    ///     created; it scans that array like a <see cref="List{T}" /> enumerator, observes a stable count, and
    ///     never throws for concurrent modification.
    /// </summary>
    public struct Enumerator : IEnumerator<TValue>
    {
        private readonly TValue[] _values;
        private readonly int _count;
        private int _i;
        private TValue _current;

        /// <summary>Binds the enumerator to a snapshot of the owner's current ordered store.</summary>
        /// <param name="owner">The collection to snapshot and enumerate.</param>
        internal Enumerator(ConcurrentOrderedDict<TKey, TValue> owner)
        {
            Store store = owner._store;
            _values = store._values;
            // Acquire-read the count AFTER capturing the array: slots below it were release-published, so every
            // element this enumerator can reach is fully written.
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
    ///     absent — the single-bucket-walk core every add-shaped operation funnels through. The map insert itself
    ///     answers the "does it already exist?" question, so the absent path costs exactly one hash walk (like the
    ///     framework dictionary's TryAdd) and, when the arrays have room, allocates nothing beyond the hash node.
    /// </summary>
    /// <param name="key">The key to append (already null-checked by the caller).</param>
    /// <param name="value">The value to store.</param>
    /// <returns><see langword="true" /> if appended; <see langword="false" /> if the key already existed (nothing changed).</returns>
    /// <remarks>
    ///     The caller must hold <see cref="_writeLock" />. Publication order: hash-map insert first (a reader may
    ///     find the key, with its correct inline value, a hair before the list path shows it), then either a
    ///     release store of the grown count (in-place append) or the volatile publish of a fresh
    ///     <see cref="Store" /> (growth, or an append blocked by the floor rule). Every allocation — grown arrays
    ///     AND the snapshot holder — happens BEFORE the map insert, so once <c>TryAdd</c> has succeeded only
    ///     throw-free publishes remain: an allocation failure can never strand a key that resolves on the key
    ///     path but never appears on the list path.
    /// </remarks>
    private bool TryAppendUnderLock(TKey key, TValue value)
    {
        Store s = _store;
        int idx = s._count; // plain read: every count writer holds the lock we hold
        TKey[] keys = s._keys;
        TValue[] values = s._values;

        // Copy the arrays when out of capacity, or when the next slot sits below the floor (it is still visible
        // to enumerators of an older, higher-count generation sharing these arrays — see Store's rules).
        Store? grown = null;
        if (idx == values.Length || idx < s._floor)
        {
            int newCap = values.Length == 0
                ? DefaultCapacity
                : idx == values.Length ? (int)Math.Min((long)values.Length * 2, Array.MaxLength) : values.Length;
            keys = CopyArray(keys, newCap, idx);
            values = CopyArray(values, newCap, idx);
            keys[idx] = key;
            values[idx] = value;
            grown = new Store(keys, values, idx + 1); // private until the publish below; floor resets — fresh arrays
        }
        else
        {
            // Writing above the visible count: invisible to every reader until the count publish. If this turns
            // out to be a duplicate, the stray slot content is inert — the next real append overwrites it.
            keys[idx] = key;
            values[idx] = value;
        }

        // One walk: the map insert IS the duplicate check. On a duplicate the (rare) freshly copied arrays are
        // simply discarded — the published store was never touched.
        if (!_byKey.TryAdd(key, new ValueIndex(value, idx)))
        {
            return false;
        }

        if (grown is not null)
        {
            _store = grown; // volatile publish
        }
        else
        {
            // This release store is what makes the slots written above visible.
            Volatile.Write(ref s._count, idx + 1);
        }

        return true;
    }

    /// <summary>
    ///     Replaces the value of an entry that is already present, on both paths. Writes in place for
    ///     atomically-writable values (O(1), zero allocation — one atomic field store into the hash node through
    ///     <paramref name="vi" /> and one into the value array); otherwise swaps the hash node and publishes a
    ///     fresh value array so lock-free readers never see a torn value (O(n)).
    /// </summary>
    /// <param name="key">The existing key being updated (used only on the non-atomic node-swap path).</param>
    /// <param name="vi">A live reference to the key's inline payload inside its hash node; stale after the non-atomic path swaps the node, so this method must be its last consumer.</param>
    /// <param name="value">The replacement value.</param>
    /// <remarks>The caller must hold <see cref="_writeLock" />.</remarks>
    private void ReplaceExistingUnderLock(TKey key, ref ValueIndex vi, TValue value)
    {
        Store s = _store;
        int idx = vi._index;

        if (ConcurrentDictionaryTypeProps<TValue>.IsWriteAtomic)
        {
            // In-place atomic single-field stores; lock-free readers see the old or new value, never a torn one.
            vi._value = value;
            if ((uint)idx < (uint)s._count)
            {
                s._values[idx] = value;
            }

            return;
        }

        // Non-atomic value: publish a fresh node (the clone's own tear-free discipline for wide values) and a
        // fresh value array. `vi` is dead the moment the indexer replaces the node — do not touch it again.
        _byKey[key] = new ValueIndex(value, idx);
        if ((uint)idx < (uint)s._count)
        {
            var values = (TValue[])s._values.Clone();
            values[idx] = value;
            // Keys are shared with the old store, so its floor still protects any frozen tail slots.
            _store = new Store(s._keys, values, s._count, s._floor);
        }
    }

    /// <summary>
    ///     Removes the entry for <paramref name="key" /> at position <paramref name="idx" /> from both paths.
    ///     The last entry is removed in O(1) by publishing a lower-count store over the same arrays (the floor
    ///     rule keeps the vacated slot safe for concurrent enumerators); an interior entry shifts the tail down
    ///     into fresh arrays and repairs the shifted entries' recorded positions — O(n) in the tail length.
    /// </summary>
    /// <param name="key">The key to remove (already resolved by the caller).</param>
    /// <param name="idx">The entry's current position (consistent under the held lock).</param>
    /// <remarks>The caller must hold <see cref="_writeLock" />.</remarks>
    private void RemoveCoreUnderLock(TKey key, int idx)
    {
        Store s = _store;
        int n = s._count;

        if (idx == n - 1)
        {
            // Tail removal: nothing shifts, so share the arrays and only lower the count — via a NEW store,
            // because a count on a published store may only grow. The floor rises to the old high-water mark so
            // a later append cannot overwrite the vacated slot while an old enumerator can still read it. The
            // slot itself is deliberately not cleared: an enumerator captured at count n must still find valid
            // data there (the removed value stays reachable until the slot is reused or the arrays replaced).
            // The successor is allocated BEFORE the map removal: once the key stops resolving, only throw-free
            // publishes remain, so an allocation failure can never strand a key that stops resolving on the key
            // path yet keeps enumerating forever.
            Store next = new(s._keys, s._values, n - 1, Math.Max(s._floor, n));
            _byKey.TryRemove(key, out _); // key path leads (the same skew direction as adds)
            _store = next;
            return;
        }

        // Interior removal: fresh arrays so concurrent readers of the old snapshot are undisturbed by the shift.
        // All allocations happen before the map removal, for the same strand-proofing reason as the tail path.
        var keys = new TKey[s._keys.Length];
        var values = new TValue[s._values.Length];
        Array.Copy(s._keys, keys, idx);                       // keep [0, idx)
        Array.Copy(s._keys, idx + 1, keys, idx, n - idx - 1); // shift (idx, n) down
        Array.Copy(s._values, values, idx);
        Array.Copy(s._values, idx + 1, values, idx, n - idx - 1);
        Store shifted = new(keys, values, n - 1);

        // Key path first: the key stops resolving before it stops enumerating (the same skew direction as adds,
        // where the key path leads).
        _byKey.TryRemove(key, out _);

        // Repair the recorded positions of the shifted entries so IndexOf stays correct. In-place atomic int
        // stores through the ref seam — no node allocation per shifted key.
        for (int i = idx; i < n - 1; i++)
        {
            RewriteIndexUnderLock(keys[i], i);
        }

        _store = shifted;
    }

    /// <summary>Rewrites the recorded insertion-order position of <paramref name="key" /> in place (an atomic <c>int</c> store into its hash node — safe for any <typeparamref name="TValue" />, allocation-free).</summary>
    /// <param name="key">The key whose position changed; expected present.</param>
    /// <param name="index">The key's new position.</param>
    /// <remarks>The caller must hold <see cref="_writeLock" />.</remarks>
    private void RewriteIndexUnderLock(TKey key, int index)
    {
        ref ValueIndex vi = ref _byKey.GetValueRefOrNullRef(key);
        Debug.Assert(!Unsafe.IsNullRef(ref vi), "a key on the list path is missing from the key map under the write lock");
        if (!Unsafe.IsNullRef(ref vi))
        {
            vi._index = index;
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

    /// <summary>Throws <see cref="ArgumentNullException" /> if <paramref name="key" /> is null (guards the reference-key case).</summary>
    /// <param name="key">The key to validate.</param>
    /// <remarks>
    ///     The <c>typeof</c> guard is load-bearing for allocation-freedom: a bare <c>key is null</c> on a generic
    ///     <typeparamref name="TKey" /> compiles to <c>box</c>+compare IL, and under unoptimized (Debug) codegen
    ///     that box EXECUTES for value-type keys — 24 B allocated per null-check on every hot path (Release JIT
    ///     elides it, which is why only Debug measurements exposed it). Short-circuiting on
    ///     <c>typeof(TKey).IsValueType</c> means the box IL is never reached for value types (and boxing a
    ///     reference type is a no-op by ECMA-335), so every configuration is allocation-free; Release codegen is
    ///     unchanged (the JIT folds the guard to a constant). A <c>Nullable&lt;T&gt;</c> key — already excluded by
    ///     the <c>notnull</c> constraint — would skip this check and fail later at hashing instead.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    private static void NullCheck(TKey key)
    {
        if (!typeof(TKey).IsValueType && key is null)
        {
            throw new ArgumentNullException(nameof(key));
        }
    }

    /// <summary>
    ///     Refuses a write operation issued by the thread that is ALREADY inside this collection's write lock —
    ///     which can only happen from user code the collection itself invoked under the lock: an
    ///     <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key
    ///     comparer / <c>GetHashCode</c> override running inside a locked lookup.
    /// </summary>
    /// <remarks>
    ///     <see cref="Monitor" /> is reentrant, so without this guard such a callback would NOT deadlock — it
    ///     would silently interleave a second structural mutation inside a half-applied one (both operations
    ///     working from stale local snapshots of the store) and corrupt the collection. Turning that into a
    ///     deterministic exception is the whole point: undefined corruption becomes a clear, immediate contract
    ///     violation at the exact call site. Factories handed to <see cref="GetOrAdd(TKey, Func{TKey, TValue})" />
    ///     and <see cref="AddOrUpdate" /> run OUTSIDE the lock and therefore may legitimately call back into any
    ///     operation — this guard passes for them. Lock-free reads are always legal from anywhere.
    /// </remarks>
    /// <exception cref="LockRecursionException">The current thread already holds the write lock (a reentrant write from user code running under it).</exception>
    private void ThrowIfReentrantWrite()
    {
        // Monitor.IsEntered answers for the CURRENT thread only; another thread holding the lock returns false
        // and we simply block on the lock as usual.
        if (Monitor.IsEntered(_writeLock))
        {
            throw new LockRecursionException(
                "ConcurrentOrderedDict does not support reentrant writes: a mutating operation was invoked from user code " +
                "running inside the collection's write lock (an AddRange source enumerator, a RemoveWhere predicate, or a " +
                "key comparer). Perform the mutation after the enclosing operation returns.");
        }
    }
}
