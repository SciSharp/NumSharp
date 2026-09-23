#nullable enable

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using NumSharp.Collections.Concurrent;

namespace NumSharp.Collections;

/// <summary>
///     A lock-free, insertion-ordered, index-addressable dictionary that stores each value <b>once</b>. Like
///     <see cref="ConcurrentOrderedDictionary{TKey,TValue}" /> it offers two access paths over one set of entries — a
///     dictionary path keyed by <typeparamref name="TKey" /> and a list path addressed by insertion-order
///     <c>int</c> index — but it is the <b>lean</b> counterpart: the value is held a single time (in one dense
///     array), so it uses ~1.5× less memory per entry and builds markedly faster, at the cost of a key lookup
///     that touches three arrays instead of one inline node.
/// </summary>
/// <remarks>
///     <para>
///         <b>Layout.</b> A key maps to a dense <c>int</c> insertion-order slot through a purpose-built
///         open-addressed index (an <c>int[]</c> of <c>slot+1</c> words, Fibonacci-hashed home + linear probe,
///         <c>0</c> = empty); the slot indexes two contiguous arrays, <c>keys</c> and <c>values</c>. A key lookup
///         is: acquire-read the index word → confirm <c>keys[slot]</c> → return <c>values[slot]</c>. Enumeration,
///         <see cref="ToArray" /> and <see cref="AsValuesSpan" /> scan the dense value array directly, so they read
///         like a <see cref="List{T}" /> of values. Because the value lives only in <c>values</c> (never inlined
///         beside the key), a value replace is a single store — there is no second copy to keep in sync.
///     </para>
///     <para>
///         <b>Concurrency (how it is lock-free without tearing).</b> All shape lives in one immutable-in-shape
///         <c>Tables</c> generation published through a single <see cref="Volatile" /> field, so a reader captures
///         a consistent snapshot in one acquiring read. <b>Every generation owns every array it references</b>: a
///         transition that replaces the generation (growth, removal, <see cref="Clear" />, and the replace of a
///         non-atomically-writable value) builds a fresh index, keys array AND values array, so no array is ever
///         shared by two generations. Within a generation:
///         <list type="bullet">
///             <item>
///                 <description>
///                     <b>Reads are lock-free and need no count-gate.</b> An in-place append writes
///                     <c>keys[n]</c>/<c>values[n]</c> <i>before</i> the index word, and publishes that word with
///                     <see cref="Volatile.Write(ref int, int)" /> (release); a reader acquire-reads it with
///                     <c>Volatile.Read</c>, so any word it observes already exposes its slot's
///                     key and value, and the <c>keys[slot]</c> equality confirm proves identity. The generation
///                     ownership above is what makes this sound: the only writes a generation's index ever receives
///                     are appends into that same generation, so a word a reader finds always points at a slot its own
///                     generation wrote.
///                 </description>
///             </item>
///             <item>
///                 <description>
///                     <b>An atomically-writable value replace is lock-free and never loses a completed write.</b> A
///                     lock-free <see cref="SetByKey" /> stores the value into the live generation, executes a full
///                     fence (<see cref="Interlocked.MemoryBarrier" />), and commits only if the resize flag is down and
///                     the generation is still the live one; otherwise it redoes the write under the lock on the live
///                     generation. Every generation-replacing resize raises the flag with a full-fence
///                     <see cref="Interlocked.Exchange(ref int, int)" /> before it reads the live value array, copies it,
///                     publishes, and lowers the flag in a <c>finally</c> (growth and the wide-value replace build their
///                     new keys and index before raising it, so their window holds only the copy and the publish; an
///                     order-preserving removal rebuilds inside it — measured faster single-threaded). The two halves are the
///                     store-buffering (Dekker) shape — "store mine, then load theirs" on both sides — which
///                     release/acquire ordering alone does NOT make safe (x86-TSO and ARM64 both let each side's load
///                     miss the other side's buffered store; measured before the fences: ~3% of replaces lost under copy
///                     churn). With a full fence on BOTH sides at least one of the two loads sees the other store: either
///                     the copy reads the new value, or the replacer sees the flag and retries. The generation check
///                     catches a resize that had already published before the replacer looked. The fence costs
///                     +2.6–3.2 ns per replace (measured on an i9-13900K P-core, pre-fix vs fixed source in one
///                     process: <c>&lt;int,int&gt;</c>/<c>&lt;int,long&gt;</c> 1.9 → 4.5–5.1 ns,
///                     <c>&lt;string,int&gt;</c> 5.2 → 6.7 ns) — still ~3× faster than the locked replace of the
///                     sibling maps (14.6 ns); lock-free reads pay nothing.
///                 </description>
///             </item>
///         </list>
///         <b>Writes are serialized by a single lock</b> because a global insertion order cannot be maintained
///         per-stripe — the deliberate cost of ordering — but the common value replace escapes that lock via the
///         path above. A non-atomically-writable value type (a wide struct) forces a fresh generation on replace
///         so a reader never sees a torn value, and its replace therefore takes the lock.
///     </para>
///     <para>
///         <b>Two named relaxations of the lock-free replace</b>, both confined to a <see cref="SetByKey" /> that is
///         still running, and both consequences of its store-then-verify design (the store is visible the moment it
///         lands; the verify decides afterwards whether it counted): (1) when a resize overlaps the replace and its
///         copy misses the store, the replace retries under the lock — meanwhile ANOTHER thread reading that key can
///         see the new value (in the older generation), then the old one (the resized generation's copy), then the new
///         one again (the retry); (2) a <see cref="TryRemove" /> of the same key that overlaps that retry can return
///         the new value as the removed one while the retry — an upsert — then re-adds the key at the end. Once
///         <see cref="SetByKey" /> returns, every later read of the key returns that value or a newer one. The
///         lock-based siblings (<see cref="ConcurrentOrderedDictionary{TKey,TValue}" />,
///         <see cref="ConcurrentOrderedCompactDictionary{TKey,TValue}" />) have neither relaxation — their replace
///         takes the write lock.
///     </para>
///     <para>
///         <b>The <c>int</c>-key footgun.</b> Both <see cref="this[TKey]" /> and <see cref="this[int]" /> exist;
///         when <typeparamref name="TKey" /> is <c>int</c> the compiler binds <c>d[5]</c> to the <i>index</i>
///         indexer, so use the methods (<see cref="TryGetValue" />, <see cref="GetByKey" />,
///         <see cref="SetByKey" />) for the key path with <c>int</c> keys.
///     </para>
///     <para>
///         <b>Consistency.</b> A lock-free reader always sees valid, untorn keys and values, and never a key paired
///         with a value that was not stored under it. While a writer runs a reader may miss a just-added entry (an
///         add races the read) — snapshot semantics, as in the framework concurrent dictionary. Enumeration reads the
///         generation field once and captures that generation's value array and count; structural changes never
///         rewrite the slots it can see (an atomically-writable value replaced meanwhile may be seen old or new).
///     </para>
///     <para>
///         <b>Keys and capacity.</b> A <see langword="null" /> key is refused with
///         <see cref="ArgumentNullException" /> on every key entry point, as the sibling ordered maps and the framework
///         dictionaries do. The index is a power-of-two <c>int[]</c>, and 2^30 words is the largest one .NET can
///         allocate, so the collection holds at most 751,619,276 entries (2^30 × 70 % load); a capacity or an add
///         beyond that is refused with <see cref="ArgumentOutOfRangeException" /> before anything is allocated.
///     </para>
///     <para>
///         <b>Enforced callback contract.</b> User code run <i>inside</i> the write lock (an <see cref="AddRange" />
///         source enumerator or the key comparer) must not write back into the collection; because
///         <see cref="Monitor" /> is reentrant such a write would interleave two half-applied mutations rather than
///         deadlock, so it is detected and refused with <see cref="LockRecursionException" />. A comparer that throws
///         during a resize (which re-hashes every surviving key) aborts it before anything is published, and the
///         resize flag is lowered in a <c>finally</c>, so the collection and its lock-free replace are both left
///         exactly as they were.
///     </para>
/// </remarks>
/// <typeparam name="TKey">The non-null key type; uniqueness and lookups use the configured comparer.</typeparam>
/// <typeparam name="TValue">The value type stored once per key and yielded, in index order, by enumeration.</typeparam>
public sealed class OrderedDictionary<TKey, TValue> : IReadOnlyList<TValue>
    where TKey : notnull
{
    /// <summary>The initial ordered-array capacity and the size the arrays first grow to from empty. Small on purpose — many ordered maps stay tiny.</summary>
    private const int DefaultCapacity = 4;

    /// <summary>Grow the index (and rehash) when the live count reaches this fraction of index length, as a percent — the open-addressing load factor.</summary>
    private const int LoadPercent = 70;

    /// <summary>
    ///     The largest index length: 2^30 is the largest power of two an <c>int[]</c> can have
    ///     (<see cref="Array.MaxLength" /> is just under 2^31), and the Fibonacci home requires a power-of-two length.
    ///     Doubling past it would overflow <c>int</c> — which is how an unguarded size loop once wrapped to
    ///     <see cref="int.MinValue" />, then to 0, and spun forever.
    /// </summary>
    private const int MaxIndexLength = 1 << 30;

    /// <summary>
    ///     The most entries the collection can hold: the largest count that stays under the load factor in a
    ///     <see cref="MaxIndexLength" />-word index (751,619,276). Both the constructor's capacity and growth refuse
    ///     anything above it with <see cref="ArgumentOutOfRangeException" />.
    /// </summary>
    private const int MaxCount = (int)((long)MaxIndexLength * LoadPercent / 100);

    /// <summary>
    ///     An immutable-in-shape generation of the whole structure: the open-addressed index, the dense keys and
    ///     values, the hash shift, and the live count. Shape (the arrays and shift) is fixed for the object's life;
    ///     only <see cref="_count" /> and the in-place slots advance. A grow/shrink publishes a whole new generation,
    ///     and a generation never shares an array with another one (see the type remarks for why that matters).
    /// </summary>
    private sealed class Tables
    {
        /// <summary>Open-addressed index: <c>_index[h] == slot + 1</c> (0 = empty). Power-of-two length; a reader acquire-reads a word, a writer release-writes it in place.</summary>
        internal readonly int[] _index;

        /// <summary>Dense keys in insertion order; <c>_keys[slot]</c> confirms a probe hit.</summary>
        internal readonly TKey[] _keys;

        /// <summary>Dense values in insertion order — the single value copy, and the contiguous enumeration surface.</summary>
        internal readonly TValue[] _values;

        /// <summary>Right-shift for the Fibonacci hash home: <c>32 - log2(index length)</c>.</summary>
        internal readonly int _shift;

        /// <summary>Live entry count (= next free dense slot). Advanced with <see cref="Volatile.Write(ref int, int)" /> after an in-place append; a shrink publishes a new generation.</summary>
        internal int _count;

        /// <summary>Builds a generation and derives its hash shift from the index length.</summary>
        /// <param name="index">The open-addressed index array (power-of-two length).</param>
        /// <param name="keys">The dense key array.</param>
        /// <param name="values">The dense value array.</param>
        /// <param name="count">The live entry count.</param>
        internal Tables(int[] index, TKey[] keys, TValue[] values, int count)
        {
            _index = index;
            _keys = keys;
            _values = values;
            _count = count;
            _shift = 32 - System.Numerics.BitOperations.TrailingZeroCount(index.Length);
        }
    }

    /// <summary>The current generation, published volatile so a lock-free reader acquires a consistent snapshot in one read.</summary>
    private volatile Tables _t;

    /// <summary>
    ///     1 while a generation-replacing resize copies the live value array and publishes (see
    ///     <see cref="BeginResize" />), 0 otherwise — the guard that makes the lock-free value replace rigorous: a
    ///     lock-free store that observes it set redoes under the lock.
    /// </summary>
    /// <remarks>
    ///     An <c>int</c> rather than a <c>volatile bool</c> so the resizer can raise it with
    ///     <see cref="Interlocked.Exchange(ref int, int)" />, a FULL fence: a release store would let the copy's
    ///     loads run before the flag is visible, which is exactly the store-buffering race that lost updates. Read
    ///     with <see cref="Volatile.Read(ref int)" /> (acquire), written only under the write lock.
    /// </remarks>
    private int _resizing;

    /// <summary>Serializes all structural mutation; lock-free reads and atomic value replaces take no lock.</summary>
    private readonly object _lock = new();

    /// <summary>The managed thread id currently holding <see cref="_lock" /> (0 = none) — the reentrancy guard that refuses a write-back from inside a lock-held callback.</summary>
    private int _ownerThreadId;

    /// <summary>Key comparer for hashing and the probe confirmation.</summary>
    private readonly IEqualityComparer<TKey> _comparer;

    /// <summary>Creates an empty dictionary with the default capacity and the default key comparer.</summary>
    public OrderedDictionary() : this(DefaultCapacity, null)
    {
    }

    /// <summary>Creates an empty dictionary sized to hold <paramref name="capacity" /> entries without a regrow, using the default key comparer.</summary>
    /// <param name="capacity">The number of entries to reserve; values &lt;= 0 use the default capacity.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="capacity" /> exceeds the 751,619,276 entries the index can address.</exception>
    public OrderedDictionary(int capacity) : this(capacity, null)
    {
    }

    /// <summary>Creates an empty dictionary with the default capacity and the given key comparer.</summary>
    /// <param name="comparer">The key comparer, or <see langword="null" /> for <see cref="EqualityComparer{T}.Default" />.</param>
    public OrderedDictionary(IEqualityComparer<TKey>? comparer) : this(DefaultCapacity, comparer)
    {
    }

    /// <summary>Creates an empty dictionary sized for <paramref name="capacity" /> entries with the given key comparer.</summary>
    /// <param name="capacity">The number of entries to reserve; values &lt;= 0 use the default capacity.</param>
    /// <param name="comparer">The key comparer, or <see langword="null" /> for <see cref="EqualityComparer{T}.Default" />.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="capacity" /> exceeds the 751,619,276 entries the index can address.</exception>
    public OrderedDictionary(int capacity, IEqualityComparer<TKey>? comparer)
    {
        _comparer = comparer ?? EqualityComparer<TKey>.Default;
        int cap = capacity <= 0 ? DefaultCapacity : capacity;
        // Size the index BEFORE any allocation so an unaddressable capacity is refused while nothing exists yet. The
        // old `while (len < cap * 100 / 70 + 1) len <<= 1;` on an int wrapped len to int.MinValue and then 0 for a
        // capacity of ~752 million or more, and `0 << 1 == 0` never terminated — the constructor spun forever.
        int len = IndexLengthFor(cap);
        _t = new Tables(new int[len], new TKey[cap], new TValue[cap], 0);
    }

    /// <summary>Gets the number of entries. Lock-free; a concurrent add may not yet be visible.</summary>
    public int Count => Volatile.Read(ref _t._count);

    /// <summary>Gets the key comparer used for hashing and equality.</summary>
    public IEqualityComparer<TKey> Comparer => _comparer;

    /// <summary>Fibonacci-hashed home slot for <paramref name="key" /> in generation <paramref name="t" />.</summary>
    /// <param name="t">The generation whose shift is used.</param>
    /// <param name="key">The key to hash.</param>
    /// <returns>The home index into the generation's index array.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int Home(Tables t, TKey key) => (int)((uint)_comparer.GetHashCode(key) * 2654435769u >> t._shift);

    /// <summary>
    ///     Looks up the value for <paramref name="key" />. Lock-free: acquire-read the index word (which
    ///     release-publishes its slot's key and value), confirm <c>keys[slot]</c>, return <c>values[slot]</c>.
    /// </summary>
    /// <param name="key">The key to find.</param>
    /// <param name="value">On success the stored value; otherwise <see langword="default" />.</param>
    /// <returns><see langword="true" /> if the key is present; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetValue(TKey key, out TValue value)
    {
        NullCheck(key);
        Tables t = _t;
        int[] index = t._index;
        TKey[] keys = t._keys;
        TValue[] values = t._values;
        int mask = index.Length - 1;
        int h = Home(t, key);
        while (true)
        {
            int w = Volatile.Read(ref index[h]);
            if (w == 0)
            {
                value = default!;
                return false;
            }

            int slot = w - 1;
            if (_comparer.Equals(keys[slot], key))
            {
                value = values[slot];
                return true;
            }

            h = (h + 1) & mask;
        }
    }

    /// <summary>Gets the value for <paramref name="key" /> or throws if absent.</summary>
    /// <param name="key">The key to find.</param>
    /// <returns>The stored value.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="KeyNotFoundException">No entry has the given key.</exception>
    public TValue GetByKey(TKey key)
    {
        if (TryGetValue(key, out TValue value)) return value;
        throw new KeyNotFoundException($"The key '{key}' was not present in the {nameof(OrderedDictionary<TKey, TValue>)}.");
    }

    /// <summary>Gets or sets the value for <paramref name="key" /> (the dictionary path). With <c>int</c> keys the compiler prefers <see cref="this[int]" /> — use <see cref="GetByKey" />/<see cref="SetByKey" /> instead.</summary>
    /// <param name="key">The key to read or write.</param>
    /// <returns>The stored value.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="KeyNotFoundException">On read, no entry has the given key.</exception>
    /// <exception cref="ArgumentOutOfRangeException">On write, the key is absent and adding it would exceed the 751,619,276 entries the index can address.</exception>
    /// <exception cref="LockRecursionException">On write, the locked path is reached from inside a callback this collection is running under its write lock.</exception>
    public TValue this[TKey key]
    {
        get => GetByKey(key);
        set => SetByKey(key, value);
    }

    /// <summary>Gets the value at insertion-order position <paramref name="index" /> (the list path). Lock-free.</summary>
    /// <param name="index">The zero-based insertion-order position.</param>
    /// <returns>The value at that position.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or &gt;= <see cref="Count" />.</exception>
    public TValue this[int index]
    {
        get
        {
            Tables t = _t;
            int count = Volatile.Read(ref t._count);
            if ((uint)index >= (uint)count)
                throw new ArgumentOutOfRangeException(nameof(index), index, $"Index must be in [0, {count}).");
            return t._values[index];
        }
    }

    /// <summary>Tests whether <paramref name="key" /> is present. Lock-free.</summary>
    /// <param name="key">The key to test.</param>
    /// <returns><see langword="true" /> if present; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool ContainsKey(TKey key)
    {
        NullCheck(key);
        return FindSlot(_t, key) >= 0;
    }

    /// <summary>Returns the insertion-order position of <paramref name="key" />, or -1 if absent. Lock-free.</summary>
    /// <param name="key">The key to locate.</param>
    /// <returns>The zero-based position, or -1.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public int IndexOf(TKey key)
    {
        NullCheck(key);
        return FindSlot(_t, key);
    }

    /// <summary>Adds the pair only if <paramref name="key" /> is absent.</summary>
    /// <param name="key">The key to add.</param>
    /// <param name="value">The value to associate.</param>
    /// <returns><see langword="true" /> if added; <see langword="false" /> if the key was already present.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The key is absent and adding it would exceed the 751,619,276 entries the index can address.</exception>
    /// <exception cref="LockRecursionException">Called from inside a callback this collection is running under its write lock.</exception>
    public bool TryAdd(TKey key, TValue value)
    {
        NullCheck(key);
        EnterWrite();
        try
        {
            Tables t = _t;
            int[] index = t._index;
            TKey[] keys = t._keys;
            int mask = index.Length - 1;
            int h = Home(t, key);
            int emptyH;
            // ONE fused probe: locate the first empty word (absent) or reject on a key match (present).
            while (true)
            {
                int w = index[h];
                if (w == 0)
                {
                    emptyH = h;
                    break;
                }

                if (_comparer.Equals(keys[w - 1], key)) return false;
                h = (h + 1) & mask;
            }

            int n = t._count;
            if (n >= (long)index.Length * LoadPercent / 100 || n == keys.Length)
            {
                // Grow republishes a fresh generation (emptyH is stale after the rehash, so it re-probes inside).
                AppendGrow(t, key, value);
            }
            else
            {
                // In-place: slots first, then RELEASE the index word (publishing them), then advance the count.
                t._keys[n] = key;
                t._values[n] = value;
                Volatile.Write(ref index[emptyH], n + 1);
                Volatile.Write(ref t._count, n + 1);
            }

            return true;
        }
        finally
        {
            ExitWrite();
        }
    }

    /// <summary>
    ///     Sets the value for <paramref name="key" />, adding it if absent. When the key exists and
    ///     <typeparamref name="TValue" /> is written atomically this is a <b>lock-free</b>, rigorous single store
    ///     plus one full fence (see the type remarks, including its two named relaxations); otherwise it takes the
    ///     write lock.
    /// </summary>
    /// <param name="key">The key to write.</param>
    /// <param name="value">The value to store.</param>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The key is absent and adding it would exceed the 751,619,276 entries the index can address.</exception>
    /// <exception cref="LockRecursionException">The locked path is reached from inside a callback this collection is running under its write lock.</exception>
    public void SetByKey(TKey key, TValue value)
    {
        NullCheck(key);

        // Lock-free fast path: present key + atomically-writable value → store, FULL fence, then verify no resize was
        // copying (flag) and none has published (generation). Either failing means the store may have hit a slot a
        // resize's copy missed, so redo under the lock on the live generation.
        if (ConcurrentDictionaryTypeProps<TValue>.IsWriteAtomic)
        {
            Tables t = _t;
            int slot = FindSlot(t, key);
            if (slot >= 0)
            {
                t._values[slot] = value;

                // The replacer's half of the store/flag handshake (the resizer's half is BeginResize). Without a full
                // fence the flag load below may be satisfied before this store leaves the store buffer, while the
                // resizer's copy reads the OLD value — each side misses the other's store and the write is lost
                // although this method returns normally (measured ~3% of replaces under copy churn). Release/acquire
                // cannot forbid that outcome on x86-TSO or ARM64; a full fence here AND in BeginResize can.
                Interlocked.MemoryBarrier();

                // Acquire-read the flag BEFORE the generation: a flag already lowered by a finished resize then
                // guarantees that resize's publish is visible, so the generation check below cannot miss it.
                if (Volatile.Read(ref _resizing) == 0 && ReferenceEquals(_t, t)) return;
            }
        }

        EnterWrite();
        try
        {
            Tables t = _t;
            int slot = FindSlot(t, key);
            if (slot >= 0)
            {
                if (ConcurrentDictionaryTypeProps<TValue>.IsWriteAtomic)
                {
                    // Under the lock no resize can run concurrently, so the plain atomic store is final.
                    t._values[slot] = value;
                }
                else
                {
                    ReplaceWideUnderLock(t, slot, value);
                }
            }
            else
            {
                AppendUnderLock(t, key, value);
            }
        }
        finally
        {
            ExitWrite();
        }
    }

    /// <summary>Returns the existing value for <paramref name="key" />, or adds and returns <paramref name="value" /> if absent.</summary>
    /// <param name="key">The key to read or add.</param>
    /// <param name="value">The value to add when the key is absent.</param>
    /// <returns>The existing value, or the newly added <paramref name="value" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" /> (raised by the lock-free lookup that runs first).</exception>
    /// <exception cref="ArgumentOutOfRangeException">The key is absent and adding it would exceed the 751,619,276 entries the index can address.</exception>
    /// <exception cref="LockRecursionException">Called from inside a callback this collection is running under its write lock.</exception>
    public TValue GetOrAdd(TKey key, TValue value)
    {
        // Lock-free fast read first: most GetOrAdd calls hit an existing key. It also performs the null-key check.
        if (TryGetValue(key, out TValue existing)) return existing;
        EnterWrite();
        try
        {
            Tables t = _t;
            int slot = FindSlot(t, key);
            if (slot >= 0) return t._values[slot]; // added by a racing writer between the read and the lock
            AppendUnderLock(t, key, value);
            return value;
        }
        finally
        {
            ExitWrite();
        }
    }

    /// <summary>Removes <paramref name="key" /> if present, preserving insertion order (trailing entries shift down into a fresh generation — O(n), like <see cref="List{T}.RemoveAt" />).</summary>
    /// <param name="key">The key to remove.</param>
    /// <param name="value">On success the removed value; otherwise <see langword="default" />.</param>
    /// <returns><see langword="true" /> if an entry was removed; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Called from inside a callback this collection is running under its write lock.</exception>
    public bool TryRemove(TKey key, out TValue value)
    {
        NullCheck(key);
        EnterWrite();
        try
        {
            Tables t = _t;
            int slot = FindSlot(t, key);
            if (slot < 0)
            {
                value = default!;
                return false;
            }

            int n = t._count;

            // Build a fresh, compacted generation (order-preserving) inside the fenced resize window, so a lock-free
            // replace racing the value copy either lands in it or sees the flag and redoes under the lock. The
            // removed value is read in there too: read before the flag, it could be the value a lock-free replace had
            // just overwritten — a replace that then committed (it saw the flag down) and whose write this removal
            // would silently report as never having happened. The flag is lowered in the finally: the index rebuild
            // runs the user's comparer, and a throw there used to leave the flag stuck, silently locking every later
            // replace. (Unlike growth, this path does NOT build the index before raising the flag: measured, doing so
            // made the single-threaded removal 16-21 % slower — the values must be copied right after the keys — so it
            // keeps the pre-fix order, where a replace overlapping the rebuild waits for the removal as it always did.)
            BeginResize();
            try
            {
                value = t._values[slot];
                var keys = new TKey[t._keys.Length];
                var values = new TValue[t._values.Length];
                Array.Copy(t._keys, keys, slot);
                Array.Copy(t._keys, slot + 1, keys, slot, n - slot - 1);
                Array.Copy(t._values, values, slot);
                Array.Copy(t._values, slot + 1, values, slot, n - slot - 1);
                var fresh = new Tables(new int[t._index.Length], keys, values, n - 1);
                for (int s = 0; s < n - 1; s++) InsertIndex(fresh, keys[s], s);
                _t = fresh;
            }
            finally
            {
                EndResize();
            }

            return true;
        }
        finally
        {
            ExitWrite();
        }
    }

    /// <summary>Removes all entries, publishing a fresh empty generation.</summary>
    /// <exception cref="LockRecursionException">Called from inside a callback this collection is running under its write lock.</exception>
    public void Clear()
    {
        EnterWrite();
        try
        {
            var fresh = new Tables(new int[8], new TKey[DefaultCapacity], new TValue[DefaultCapacity], 0);

            // No value copy happens here — the generation check alone would catch a store into the cleared generation —
            // but every generation replacement follows the same flag protocol, so no transition can be the one that
            // silently skips it.
            BeginResize();
            try
            {
                _t = fresh;
            }
            finally
            {
                EndResize();
            }
        }
        finally
        {
            ExitWrite();
        }
    }

    /// <summary>Adds many pairs under a single lock acquisition. A source that writes back into this collection is refused (see the callback contract).</summary>
    /// <param name="pairs">The pairs to add; existing keys are skipped (add-if-absent, like repeated <see cref="TryAdd" />).</param>
    /// <remarks>Each added pair is published on its own, so a throw mid-batch (a null key, a refused write-back, the source itself) leaves every earlier pair added and the collection consistent.</remarks>
    /// <exception cref="ArgumentNullException"><paramref name="pairs" /> is <see langword="null" />, or a key inside it is <see langword="null" />.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Adding the pairs would exceed the 751,619,276 entries the index can address.</exception>
    /// <exception cref="LockRecursionException">The source enumerator writes back into this collection.</exception>
    public void AddRange(IEnumerable<KeyValuePair<TKey, TValue>> pairs)
    {
        if (pairs is null) throw new ArgumentNullException(nameof(pairs));
        EnterWrite();
        try
        {
            foreach (var kv in pairs)
            {
                NullCheck(kv.Key);
                Tables t = _t;
                if (FindSlot(t, kv.Key) >= 0) continue;
                AppendUnderLock(t, kv.Key, kv.Value);
            }
        }
        finally
        {
            ExitWrite();
        }
    }

    /// <summary>Returns the values in insertion order as a <b>zero-copy</b> span over the live dense array — the fastest scan path. The span is valid until the next structural mutation.</summary>
    /// <returns>A read-only span over the current values, in insertion order.</returns>
    public ReadOnlySpan<TValue> AsValuesSpan()
    {
        Tables t = _t;
        return new ReadOnlySpan<TValue>(t._values, 0, Volatile.Read(ref t._count));
    }

    /// <summary>Copies the values, in insertion order, into a new array.</summary>
    /// <returns>A new array of the values in insertion order.</returns>
    public TValue[] ToArray()
    {
        Tables t = _t;
        int count = Volatile.Read(ref t._count);
        var result = new TValue[count];
        Array.Copy(t._values, result, count);
        return result;
    }

    /// <summary>Copies the keys, in insertion order, into a new array.</summary>
    /// <returns>A new array of the keys in insertion order.</returns>
    public TKey[] KeysToArray()
    {
        Tables t = _t;
        int count = Volatile.Read(ref t._count);
        var result = new TKey[count];
        Array.Copy(t._keys, result, count);
        return result;
    }

    /// <summary>Copies the values, in insertion order, into <paramref name="array" /> starting at <paramref name="arrayIndex" />.</summary>
    /// <param name="array">The destination array.</param>
    /// <param name="arrayIndex">The zero-based start position in <paramref name="array" />.</param>
    /// <exception cref="ArgumentNullException"><paramref name="array" /> is <see langword="null" />.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="arrayIndex" /> is negative.</exception>
    /// <exception cref="ArgumentException"><paramref name="array" /> is too small from <paramref name="arrayIndex" />.</exception>
    public void CopyTo(TValue[] array, int arrayIndex)
    {
        if (array is null) throw new ArgumentNullException(nameof(array));
        if (arrayIndex < 0) throw new ArgumentOutOfRangeException(nameof(arrayIndex), arrayIndex, "Non-negative index required.");
        Tables t = _t;
        int count = Volatile.Read(ref t._count);
        if (array.Length - arrayIndex < count) throw new ArgumentException("Destination array is too small.", nameof(array));
        Array.Copy(t._values, 0, array, arrayIndex, count);
    }

    /// <summary>Returns a struct enumerator over the values in insertion order (allocation-free, snapshot at the point of the call).</summary>
    /// <returns>An enumerator over one generation's values and count.</returns>
    public Enumerator GetEnumerator()
    {
        // Read the generation ONCE and take both fields from it. `new Enumerator(_t._values, Volatile.Read(ref
        // _t._count))` read the volatile field twice, so a growth published in between paired the OLD (shorter) value
        // array with the NEW (larger) count: the enumerator ran past the array (IndexOutOfRangeException) or yielded
        // slots its generation never wrote (phantom default values).
        Tables t = _t;
        return new Enumerator(t._values, Volatile.Read(ref t._count));
    }

    /// <inheritdoc />
    IEnumerator<TValue> IEnumerable<TValue>.GetEnumerator() => GetEnumerator();

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>Probes generation <paramref name="t" /> for <paramref name="key" />, returning its dense slot or -1.</summary>
    /// <param name="t">The generation to probe.</param>
    /// <param name="key">The key to find (already null-checked by the public entry point).</param>
    /// <returns>The dense slot, or -1 if absent.</returns>
    private int FindSlot(Tables t, TKey key)
    {
        int[] index = t._index;
        TKey[] keys = t._keys;
        int mask = index.Length - 1;
        int h = Home(t, key);
        while (true)
        {
            int w = Volatile.Read(ref index[h]);
            if (w == 0) return -1;
            int slot = w - 1;
            if (_comparer.Equals(keys[slot], key)) return slot;
            h = (h + 1) & mask;
        }
    }

    /// <summary>In-place append of a known-absent key under the write lock (release-publishes the index word); grows first if the load factor is reached.</summary>
    /// <param name="t">The current generation.</param>
    /// <param name="key">The key to append.</param>
    /// <param name="value">The value to append.</param>
    /// <exception cref="ArgumentOutOfRangeException">The append needs a growth past <see cref="MaxCount" /> entries.</exception>
    private void AppendUnderLock(Tables t, TKey key, TValue value)
    {
        int n = t._count;
        if (n >= (long)t._index.Length * LoadPercent / 100 || n == t._keys.Length)
        {
            AppendGrow(t, key, value);
            return;
        }

        int[] index = t._index;
        int mask = index.Length - 1;
        int h = Home(t, key);
        while (index[h] != 0) h = (h + 1) & mask;
        t._keys[n] = key;
        t._values[n] = value;
        Volatile.Write(ref index[h], n + 1);
        Volatile.Write(ref t._count, n + 1);
    }

    /// <summary>
    ///     Grows the arrays/index and publishes a fresh generation carrying the new pair. Everything but the copy of
    ///     the live value array is built first; that copy and the publish run inside the fenced resize window
    ///     (<see cref="BeginResize" />/<see cref="EndResize" />) so a lock-free replace cannot lose an update.
    /// </summary>
    /// <param name="t">The current generation.</param>
    /// <param name="key">The key to append.</param>
    /// <param name="value">The value to append.</param>
    /// <exception cref="ArgumentOutOfRangeException">The index is already <see cref="MaxIndexLength" /> words and would have to double: the collection holds <see cref="MaxCount" /> entries.</exception>
    private void AppendGrow(Tables t, TKey key, TValue value)
    {
        int n = t._count;
        int newCap = n == t._keys.Length ? (int)Math.Min((long)Math.Max(DefaultCapacity, t._keys.Length) * 2, Array.MaxLength) : t._keys.Length;
        int newLen = t._index.Length;
        if (n + 1 >= (long)newLen * LoadPercent / 100)
        {
            // Doubling a 2^30-word index overflows int (the unguarded `newLen <<= 1` wrapped to int.MinValue) and no
            // larger power-of-two int[] exists, so the collection is full: refuse before allocating or publishing.
            if (newLen >= MaxIndexLength) ThrowTooManyEntries((long)n + 1);
            newLen <<= 1;
        }

        // The new keys and the rebuilt index come first, outside the resize flag: the rebuild runs the user's comparer
        // (it may throw — nothing is published and the flag is still down), and a lock-free replace that lands during
        // it is still in the live value array the fenced copy below reads.
        Tables fresh = GrownWith(t, key, value, newCap, newLen);

        BeginResize(); // full fence BEFORE the copy: a lock-free replace either lands in the copy or sees the flag
        try
        {
            Array.Copy(t._values, fresh._values, n);
            _t = fresh; // publish
        }
        finally
        {
            EndResize(); // lowered AFTER the publish: from here the generation check alone catches stale stores
        }
    }

    /// <summary>
    ///     Builds, privately, the generation <see cref="AppendGrow" /> publishes: the keys of <paramref name="t" /> plus
    ///     <paramref name="key" />, a rebuilt index of <paramref name="newLen" /> words, and a value array holding only
    ///     the new <paramref name="value" /> — the live values are copied in later, inside the fenced resize window.
    /// </summary>
    /// <param name="t">The live generation (the caller holds the write lock).</param>
    /// <param name="key">The key being appended.</param>
    /// <param name="value">The value being appended.</param>
    /// <param name="newCap">The new keys/values capacity (at least <c>t._count + 1</c>).</param>
    /// <param name="newLen">The new power-of-two index length.</param>
    /// <returns>The unpublished generation, count <c>t._count + 1</c>.</returns>
    /// <remarks>
    ///     Runs the user's comparer once per entry (the index rebuild) and allocates three arrays; it runs before the
    ///     resize flag is raised, so a lock-free replace overlapping the rebuild commits without waiting for the growth
    ///     (the flag window holds only the value copy and the publish). Measured: no cost against the pre-fix growth
    ///     path (a 100,000-entry build from empty, 1.01×).
    /// </remarks>
    private Tables GrownWith(Tables t, TKey key, TValue value, int newCap, int newLen)
    {
        int n = t._count;
        var keys = new TKey[newCap];
        Array.Copy(t._keys, keys, n);
        keys[n] = key;
        var values = new TValue[newCap];
        values[n] = value;
        var fresh = new Tables(new int[newLen], keys, values, n + 1);
        for (int s = 0; s <= n; s++) InsertIndex(fresh, keys[s], s);
        return fresh;
    }

    /// <summary>
    ///     Replaces the value in <paramref name="slot" /> for a <typeparamref name="TValue" /> that is not atomically
    ///     writable, by publishing a <b>whole</b> new generation — fresh index, keys AND values — so a lock-free reader
    ///     never sees a torn value and no array is shared across generations. O(n), under the write lock.
    /// </summary>
    /// <param name="t">The current generation (holding the write lock).</param>
    /// <param name="slot">The dense slot of the existing key.</param>
    /// <param name="value">The replacement value.</param>
    /// <remarks>
    ///     A values-only clone (what this path used to do) SHARED the index and keys with the previous generation, so
    ///     the next in-place append wrote its index word and key into arrays an older-generation reader still probes;
    ///     that reader then resolved the new key to its OWN never-written value slot and returned
    ///     <see langword="default" /> for a present key (the rule ConcurrentOrderedDictionary.COMPACT.md §4.7 states:
    ///     a non-atomic replace copies the whole generation). The index is cloned word-for-word — same length, same
    ///     shift, same slots — so no key is re-hashed and no user code runs. No lock-free replace exists for such a
    ///     value type, so the resize flag guards nothing here; it is raised anyway so every generation replacement
    ///     follows one protocol. Cost of the correctness fix, measured against the values-only clone
    ///     (<c>&lt;int,decimal&gt;</c>): ~1.9× at 1,000 entries (0.44 → 0.86 µs), 2.6–3.2× at 100,000 (three
    ///     large-object-heap arrays — 1 MB index, 0.4 MB keys, 1.6 MB values — instead of the one value array).
    /// </remarks>
    private void ReplaceWideUnderLock(Tables t, int slot, TValue value)
    {
        int n = t._count;
        var index = (int[])t._index.Clone();
        var keys = new TKey[t._keys.Length];
        Array.Copy(t._keys, keys, n);
        var values = new TValue[t._values.Length];
        var fresh = new Tables(index, keys, values, n);

        BeginResize();
        try
        {
            Array.Copy(t._values, values, n);
            values[slot] = value;
            _t = fresh;
        }
        finally
        {
            EndResize();
        }
    }

    /// <summary>Inserts <c>slot</c> for <paramref name="key" /> into generation <paramref name="t" />'s index with a plain store (used only while building a fresh, not-yet-published generation).</summary>
    /// <param name="t">The generation whose index is written.</param>
    /// <param name="key">The key whose slot is recorded.</param>
    /// <param name="slot">The dense slot.</param>
    private void InsertIndex(Tables t, TKey key, int slot)
    {
        int[] index = t._index;
        int mask = index.Length - 1;
        int h = Home(t, key);
        while (index[h] != 0) h = (h + 1) & mask;
        index[h] = slot + 1;
    }

    /// <summary>
    ///     The resizer's half of the store/flag handshake: raises <see cref="_resizing" /> with a FULL fence. Call it
    ///     under the write lock after everything the new generation needs except the copy of the live value array has
    ///     been built, immediately before that copy, and pair it with <see cref="EndResize" /> in a <c>finally</c>.
    /// </summary>
    /// <remarks>
    ///     <see cref="Interlocked.Exchange(ref int, int)" /> is a full fence on every supported architecture, so no load
    ///     of the copy that follows can be satisfied before the flag is globally visible. Combined with the replacer's
    ///     own fence between its store and its flag load, at least one side sees the other's store: the copy reads the
    ///     replaced value, or the replacer reads the flag and retries under the lock. The shorter the window, the less
    ///     often a lock-free replace has to fall back (and wait for the resize); growth and the wide replace keep it to
    ///     the copy + publish.
    /// </remarks>
    private void BeginResize() => Interlocked.Exchange(ref _resizing, 1);

    /// <summary>
    ///     Lowers <see cref="_resizing" /> (release store) — after the publish on success, or after a failed resize
    ///     that published nothing, where lowering it is equally correct: the live generation is unchanged, so a
    ///     replacer that stored into it during the attempt stored into the generation that stays live.
    /// </summary>
    private void EndResize() => Volatile.Write(ref _resizing, 0);

    /// <summary>
    ///     The open-addressed index length that holds <paramref name="entries" /> entries under the load factor: the
    ///     smallest power of two (at least 8) above <c>entries / 0.7</c>, computed in <see cref="long" /> so neither the
    ///     scaling nor the doubling can overflow.
    /// </summary>
    /// <param name="entries">The entry count the index must accommodate without a regrow.</param>
    /// <returns>A power-of-two index length in [8, <see cref="MaxIndexLength" />].</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="entries" /> exceeds <see cref="MaxCount" />.</exception>
    private static int IndexLengthFor(int entries)
    {
        long wanted = Math.Max(8, (long)entries * 100 / LoadPercent + 1);
        if (wanted > MaxIndexLength) ThrowTooManyEntries(entries);
        return (int)BitOperations.RoundUpToPowerOf2((uint)wanted);
    }

    /// <summary>Throws the capacity refusal shared by the constructor and the growth path (mirrors the compact sibling, which refuses an unaddressable capacity the same way).</summary>
    /// <param name="requested">The entry count that could not be accommodated.</param>
    /// <exception cref="ArgumentOutOfRangeException">Always.</exception>
    [DoesNotReturn]
    private static void ThrowTooManyEntries(long requested)
        => throw new ArgumentOutOfRangeException("capacity", requested,
            $"{nameof(OrderedDictionary<TKey, TValue>)} cannot hold more than {MaxCount} entries: its open-addressed index would need more than 2^30 words, the largest power-of-two int[].");

    /// <summary>Throws <see cref="ArgumentNullException" /> if <paramref name="key" /> is null (guards the reference-key case), like the sibling ordered maps.</summary>
    /// <param name="key">The key to validate.</param>
    /// <remarks>
    ///     The <c>typeof</c> guard is load-bearing for allocation-freedom: a bare <c>key is null</c> on a generic
    ///     <typeparamref name="TKey" /> compiles to <c>box</c>+compare IL, and under unoptimized (Debug) codegen that
    ///     box EXECUTES for value-type keys. Short-circuiting on <c>typeof(TKey).IsValueType</c> keeps it unreached for
    ///     value types (boxing a reference type is a no-op); Release codegen folds the guard to a constant, so a
    ///     value-type key pays nothing and a reference key pays one null compare. The check matters because
    ///     <c>EqualityComparer&lt;string&gt;.Default.GetHashCode(null)</c> returns 0 instead of throwing, so without it
    ///     a null key was silently stored, found and enumerated.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NullCheck(TKey key)
    {
        if (!typeof(TKey).IsValueType && key is null)
        {
            throw new ArgumentNullException(nameof(key));
        }
    }

    /// <summary>Takes the write lock, refusing a reentrant write-back from inside a lock-held callback.</summary>
    /// <exception cref="LockRecursionException">The current thread already holds the write lock (a callback wrote back into the collection).</exception>
    private void EnterWrite()
    {
        // Read before Enter: if this thread already owns the lock, no other thread can be mutating the field, so the
        // check is race-free for the reentrancy it guards against. Monitor is reentrant, so without this a write-back
        // would silently interleave two half-applied mutations instead of failing.
        if (_ownerThreadId == Environment.CurrentManagedThreadId)
            throw new LockRecursionException($"A callback running inside {nameof(OrderedDictionary<TKey, TValue>)}'s write lock must not write back into the collection.");
        Monitor.Enter(_lock);
        _ownerThreadId = Environment.CurrentManagedThreadId;
    }

    /// <summary>Releases the write lock taken by <see cref="EnterWrite" />.</summary>
    private void ExitWrite()
    {
        _ownerThreadId = 0;
        Monitor.Exit(_lock);
    }

    /// <summary>An allocation-free struct enumerator over a captured value array and count, yielding values in insertion order.</summary>
    public struct Enumerator : IEnumerator<TValue>
    {
        /// <summary>The captured dense value array (snapshot of the generation live when enumeration began).</summary>
        private readonly TValue[] _values;

        /// <summary>The number of values to yield (the count captured at the start).</summary>
        private readonly int _count;

        /// <summary>The current position, -1 before the first <see cref="MoveNext" />.</summary>
        private int _pos;

        /// <summary>Captures the value array and count to enumerate.</summary>
        /// <param name="values">The dense value array.</param>
        /// <param name="count">The number of values to yield.</param>
        internal Enumerator(TValue[] values, int count)
        {
            _values = values;
            _count = count;
            _pos = -1;
        }

        /// <summary>Gets the value at the current position.</summary>
        public readonly TValue Current => _values[_pos];

        /// <inheritdoc />
        readonly object? IEnumerator.Current => Current;

        /// <summary>Advances to the next value.</summary>
        /// <returns><see langword="true" /> if there is another value; otherwise <see langword="false" />.</returns>
        public bool MoveNext() => ++_pos < _count;

        /// <summary>Resets to before the first value.</summary>
        public void Reset() => _pos = -1;

        /// <summary>No-op; the enumerator holds no unmanaged or disposable state.</summary>
        public readonly void Dispose()
        {
        }
    }
}
