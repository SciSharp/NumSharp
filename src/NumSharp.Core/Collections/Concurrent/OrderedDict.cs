#nullable enable

using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using NumSharp.Collections.Concurrent;

namespace NumSharp.Collections;

/// <summary>
///     A lock-free, insertion-ordered, index-addressable dictionary that stores each value <b>once</b>. Like
///     <see cref="ConcurrentOrderedDict{TKey,TValue}" /> it offers two access paths over one set of entries — a
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
///         a consistent snapshot in one acquiring read. Within a generation:
///         <list type="bullet">
///             <item>
///                 <description>
///                     <b>Reads are lock-free and need no count-gate.</b> An in-place append writes
///                     <c>keys[n]</c>/<c>values[n]</c> <i>before</i> the index word, and publishes that word with
///                     <see cref="Volatile.Write(ref int, int)" /> (release); a reader acquire-reads it with
///                     <see cref="Volatile.Read(ref int)" />, so any word it observes already exposes its slot's
///                     key and value, and the <c>keys[slot]</c> equality confirm proves identity.
///                 </description>
///             </item>
///             <item>
///                 <description>
///                     <b>An atomically-writable value replace is lock-free and rigorous.</b> A <see cref="_resizing" />
///                     flag is held for the <i>entire</i> duration of any generation-replacing resize (set before its
///                     array copy, cleared after its publish). A lock-free store commits only if, after the store,
///                     <c>!_resizing &amp;&amp; ReferenceEquals(_t, t)</c>: the flag catches a resize whose copy is in
///                     progress (which might miss the store), the generation check catches a resize that already
///                     published (leaving the store on an abandoned generation) — together they cover the whole resize
///                     lifecycle, so no update is ever lost. If either check fails the write falls back to the lock.
///                 </description>
///             </item>
///         </list>
///         <b>Writes are serialized by a single lock</b> because a global insertion order cannot be maintained
///         per-stripe — the deliberate cost of ordering — but the common value replace escapes that lock via the
///         path above. A non-atomically-writable value type (a wide struct) forces a fresh value array on replace
///         so a reader never sees a torn value, and its replace therefore takes the lock.
///     </para>
///     <para>
///         <b>The <c>int</c>-key footgun.</b> Both <see cref="this[TKey]" /> and <see cref="this[int]" /> exist;
///         when <typeparamref name="TKey" /> is <c>int</c> the compiler binds <c>d[5]</c> to the <i>index</i>
///         indexer, so use the methods (<see cref="TryGetValue" />, <see cref="GetByKey" />,
///         <see cref="SetByKey" />) for the key path with <c>int</c> keys.
///     </para>
///     <para>
///         <b>Consistency.</b> A lock-free reader always sees valid, untorn keys and values. While a writer runs a
///         reader may miss a just-added entry (an add races the read) or, for an atomically-written value, observe
///         the update a hair before/after another path — snapshot semantics, as in the framework concurrent
///         dictionary. Enumeration captures the value array and count at the start and structural changes never
///         rewrite the slots it can see.
///     </para>
///     <para>
///         <b>Enforced callback contract.</b> User code run <i>inside</i> the write lock (an <see cref="AddRange" />
///         source enumerator or the key comparer) must not write back into the collection; because
///         <see cref="Monitor" /> is reentrant such a write would interleave two half-applied mutations rather than
///         deadlock, so it is detected and refused with <see cref="LockRecursionException" />.
///     </para>
/// </remarks>
/// <typeparam name="TKey">The non-null key type; uniqueness and lookups use the configured comparer.</typeparam>
/// <typeparam name="TValue">The value type stored once per key and yielded, in index order, by enumeration.</typeparam>
public sealed class OrderedDict<TKey, TValue> : IReadOnlyList<TValue>
    where TKey : notnull
{
    /// <summary>The initial ordered-array capacity and the size the arrays first grow to from empty. Small on purpose — many ordered maps stay tiny.</summary>
    private const int DefaultCapacity = 4;

    /// <summary>Grow the index (and rehash) when the live count reaches this fraction of index length, as a percent — the open-addressing load factor.</summary>
    private const int LoadPercent = 70;

    /// <summary>
    ///     An immutable-in-shape generation of the whole structure: the open-addressed index, the dense keys and
    ///     values, the hash shift, and the live count. Shape (the arrays and shift) is fixed for the object's life;
    ///     only <see cref="_count" /> and the in-place slots advance. A grow/shrink publishes a whole new generation.
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
    ///     <see langword="true" /> for the entire duration of a generation-replacing resize (set before its array
    ///     copy, cleared after its publish). The guard that makes the lock-free value replace rigorous: a lock-free
    ///     store that observes this set redoes under the lock.
    /// </summary>
    private volatile bool _resizing;

    /// <summary>Serializes all structural mutation; lock-free reads and atomic value replaces take no lock.</summary>
    private readonly object _lock = new();

    /// <summary>The managed thread id currently holding <see cref="_lock" /> (0 = none) — the reentrancy guard that refuses a write-back from inside a lock-held callback.</summary>
    private int _ownerThreadId;

    /// <summary>Key comparer for hashing and the probe confirmation.</summary>
    private readonly IEqualityComparer<TKey> _comparer;

    /// <summary>Creates an empty dictionary with the default capacity and the default key comparer.</summary>
    public OrderedDict() : this(DefaultCapacity, null)
    {
    }

    /// <summary>Creates an empty dictionary sized to hold <paramref name="capacity" /> entries without a regrow, using the default key comparer.</summary>
    /// <param name="capacity">The number of entries to reserve; values &lt;= 0 use the default capacity.</param>
    public OrderedDict(int capacity) : this(capacity, null)
    {
    }

    /// <summary>Creates an empty dictionary with the default capacity and the given key comparer.</summary>
    /// <param name="comparer">The key comparer, or <see langword="null" /> for <see cref="EqualityComparer{T}.Default" />.</param>
    public OrderedDict(IEqualityComparer<TKey>? comparer) : this(DefaultCapacity, comparer)
    {
    }

    /// <summary>Creates an empty dictionary sized for <paramref name="capacity" /> entries with the given key comparer.</summary>
    /// <param name="capacity">The number of entries to reserve; values &lt;= 0 use the default capacity.</param>
    /// <param name="comparer">The key comparer, or <see langword="null" /> for <see cref="EqualityComparer{T}.Default" />.</param>
    public OrderedDict(int capacity, IEqualityComparer<TKey>? comparer)
    {
        _comparer = comparer ?? EqualityComparer<TKey>.Default;
        int cap = capacity <= 0 ? DefaultCapacity : capacity;
        int len = 8;
        // Size the index so `cap` entries fit under the load factor without a regrow (power-of-two length).
        while (len < (long)cap * 100 / LoadPercent + 1) len <<= 1;
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
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetValue(TKey key, out TValue value)
    {
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
    /// <exception cref="KeyNotFoundException">No entry has the given key.</exception>
    public TValue GetByKey(TKey key)
    {
        if (TryGetValue(key, out TValue value)) return value;
        throw new KeyNotFoundException($"The key '{key}' was not present in the {nameof(OrderedDict<TKey, TValue>)}.");
    }

    /// <summary>Gets or sets the value for <paramref name="key" /> (the dictionary path). With <c>int</c> keys the compiler prefers <see cref="this[int]" /> — use <see cref="GetByKey" />/<see cref="SetByKey" /> instead.</summary>
    /// <param name="key">The key to read or write.</param>
    /// <returns>The stored value.</returns>
    /// <exception cref="KeyNotFoundException">On read, no entry has the given key.</exception>
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
    public bool ContainsKey(TKey key) => FindSlot(_t, key) >= 0;

    /// <summary>Returns the insertion-order position of <paramref name="key" />, or -1 if absent. Lock-free.</summary>
    /// <param name="key">The key to locate.</param>
    /// <returns>The zero-based position, or -1.</returns>
    public int IndexOf(TKey key) => FindSlot(_t, key);

    /// <summary>Adds the pair only if <paramref name="key" /> is absent.</summary>
    /// <param name="key">The key to add.</param>
    /// <param name="value">The value to associate.</param>
    /// <returns><see langword="true" /> if added; <see langword="false" /> if the key was already present.</returns>
    /// <exception cref="LockRecursionException">Called from inside a callback this collection is running under its write lock.</exception>
    public bool TryAdd(TKey key, TValue value)
    {
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
    ///     (see the type remarks); otherwise it takes the write lock.
    /// </summary>
    /// <param name="key">The key to write.</param>
    /// <param name="value">The value to store.</param>
    /// <exception cref="LockRecursionException">The locked path is reached from inside a callback this collection is running under its write lock.</exception>
    public void SetByKey(TKey key, TValue value)
    {
        // Lock-free fast path: present key + atomically-writable value → store, then verify no resize was copying
        // (flag) and none has published (generation). Either failing means the store may have hit a slot a resize's
        // copy missed, so redo under the lock on the live generation.
        if (ConcurrentDictionaryTypeProps<TValue>.IsWriteAtomic)
        {
            Tables t = _t;
            int slot = FindSlot(t, key);
            if (slot >= 0)
            {
                t._values[slot] = value;
                if (!_resizing && ReferenceEquals(_t, t)) return;
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
                    t._values[slot] = value;
                }
                else
                {
                    // Wide value: publish a fresh value array so a lock-free reader never sees a torn value.
                    _resizing = true;
                    var values = (TValue[])t._values.Clone();
                    values[slot] = value;
                    _t = new Tables(t._index, t._keys, values, t._count);
                    _resizing = false;
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
    /// <exception cref="LockRecursionException">Called from inside a callback this collection is running under its write lock.</exception>
    public TValue GetOrAdd(TKey key, TValue value)
    {
        // Lock-free fast read first: most GetOrAdd calls hit an existing key.
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
    /// <exception cref="LockRecursionException">Called from inside a callback this collection is running under its write lock.</exception>
    public bool TryRemove(TKey key, out TValue value)
    {
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

            value = t._values[slot];
            int n = t._count;
            // Build a fresh, compacted generation (order-preserving) under the resize guard so a lock-free replace
            // racing the copy redoes under the lock rather than writing to the abandoned arrays.
            _resizing = true;
            var keys = new TKey[t._keys.Length];
            var values = new TValue[t._values.Length];
            Array.Copy(t._keys, keys, slot);
            Array.Copy(t._keys, slot + 1, keys, slot, n - slot - 1);
            Array.Copy(t._values, values, slot);
            Array.Copy(t._values, slot + 1, values, slot, n - slot - 1);
            var index = new int[t._index.Length];
            var fresh = new Tables(index, keys, values, n - 1);
            for (int s = 0; s < n - 1; s++) InsertIndex(fresh, keys[s], s);
            _t = fresh;
            _resizing = false;
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
            _resizing = true;
            _t = new Tables(new int[8], new TKey[DefaultCapacity], new TValue[DefaultCapacity], 0);
            _resizing = false;
        }
        finally
        {
            ExitWrite();
        }
    }

    /// <summary>Adds many pairs under a single lock acquisition. A source that writes back into this collection is refused (see the callback contract).</summary>
    /// <param name="pairs">The pairs to add; existing keys are skipped (add-if-absent, like repeated <see cref="TryAdd" />).</param>
    /// <exception cref="ArgumentNullException"><paramref name="pairs" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">The source enumerator writes back into this collection.</exception>
    public void AddRange(IEnumerable<KeyValuePair<TKey, TValue>> pairs)
    {
        if (pairs is null) throw new ArgumentNullException(nameof(pairs));
        EnterWrite();
        try
        {
            foreach (var kv in pairs)
            {
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
    /// <returns>An enumerator over the current values.</returns>
    public Enumerator GetEnumerator() => new Enumerator(_t._values, Volatile.Read(ref _t._count));

    /// <inheritdoc />
    IEnumerator<TValue> IEnumerable<TValue>.GetEnumerator() => GetEnumerator();

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>Probes generation <paramref name="t" /> for <paramref name="key" />, returning its dense slot or -1.</summary>
    /// <param name="t">The generation to probe.</param>
    /// <param name="key">The key to find.</param>
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

    /// <summary>Grows the arrays/index and publishes a fresh generation carrying the new pair, guarding the copy+publish with <see cref="_resizing" /> so a lock-free replace cannot lose an update.</summary>
    /// <param name="t">The current generation.</param>
    /// <param name="key">The key to append.</param>
    /// <param name="value">The value to append.</param>
    private void AppendGrow(Tables t, TKey key, TValue value)
    {
        int n = t._count;
        int newCap = n == t._keys.Length ? (int)Math.Min((long)Math.Max(DefaultCapacity, t._keys.Length) * 2, Array.MaxLength) : t._keys.Length;
        int newLen = t._index.Length;
        if (n + 1 >= (long)newLen * LoadPercent / 100) newLen <<= 1;
        _resizing = true; // set BEFORE the copy: a lock-free replace overlapping the copy sees this and redoes
        var keys = new TKey[newCap];
        var values = new TValue[newCap];
        Array.Copy(t._keys, keys, n);
        Array.Copy(t._values, values, n);
        keys[n] = key;
        values[n] = value;
        var index = new int[newLen];
        var fresh = new Tables(index, keys, values, n + 1);
        for (int s = 0; s <= n; s++) InsertIndex(fresh, keys[s], s);
        _t = fresh;        // publish
        _resizing = false; // cleared AFTER the publish: from here the generation check alone catches stale stores
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

    /// <summary>Takes the write lock, refusing a reentrant write-back from inside a lock-held callback.</summary>
    /// <exception cref="LockRecursionException">The current thread already holds the write lock (a callback wrote back into the collection).</exception>
    private void EnterWrite()
    {
        // Read before Enter: if this thread already owns the lock, no other thread can be mutating the field, so the
        // check is race-free for the reentrancy it guards against. Monitor is reentrant, so without this a write-back
        // would silently interleave two half-applied mutations instead of failing.
        if (_ownerThreadId == Environment.CurrentManagedThreadId)
            throw new LockRecursionException($"A callback running inside {nameof(OrderedDict<TKey, TValue>)}'s write lock must not write back into the collection.");
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
