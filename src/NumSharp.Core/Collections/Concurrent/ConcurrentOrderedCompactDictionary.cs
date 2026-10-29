#nullable enable

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using NumSharp.Collections.Concurrent;

namespace NumSharp.Collections;

/// <summary>
///     A thread-safe, insertion-ordered, index-addressable map with <b>one copy of every key and value</b>: the
///     same dense <c>keys</c>/<c>values</c> arrays serve the dictionary path (O(1) lookup by
///     <typeparamref name="TKey" /> through an open-addressed index) and the list path (O(1) access by insertion
///     position, contiguous enumeration, spans). It offers the exact public surface and thread-safety contract of
///     <see cref="ConcurrentOrderedDictionary{TKey,TValue}" /> at roughly a third of the memory (no hash node per entry)
///     and with builds and removals that allocate nothing per entry.
/// </summary>
/// <remarks>
///     <para>
///         <b>Why a second type.</b> <see cref="ConcurrentOrderedDictionary{TKey,TValue}" /> keeps a hash node per entry
///         (a vendored concurrent dictionary) beside its parallel key/value arrays; the node is ~40 bytes and the
///         arrays 8, so an <c>&lt;int,int&gt;</c> entry costs ~57 bytes. This type folds the hash index into a
///         compact table in the style of CPython's <c>dict</c> (a dense entries array plus an open-addressed index
///         of slot numbers): the index holds one 8-byte word per probe position, so the same entry costs ~16–25
///         bytes, appends allocate 0 bytes in capacity, growth copies arrays instead of re-creating nodes, and
///         removals repair the index with a streaming pass instead of a hash walk per shifted key. Measured against
///         the node design (the discovery ledger <c>ConcurrentOrderedDictionary.COMPACT.md</c>): key lookups at parity at
///         DRAM scale and 1.1–2.7× faster while the table fits cache, builds 3–4× faster, pop/swap-back drains ~3×
///         faster, front interior removals 5× faster; near-tail interior removals ~3× slower (the index is copied
///         along with the arrays).
///     </para>
///     <para>
///         <b>Layout.</b> A <i>generation</i> (<see cref="Tables" />) is an immutable-in-shape triple: the index
///         (<c>long[]</c>, power-of-two length), <c>keys</c> and <c>values</c> (insertion order, the live prefix is
///         <c>[0, count)</c>), plus the monotonic <c>count</c>, the append <c>floor</c> and the dummy count. An index
///         word is <c>(tag &lt;&lt; 32) | (slot + 1)</c> where the tag is the key's own bits for small value-type keys
///         under the default comparer (so a hit never touches the keys array) and the comparer's hash otherwise;
///         <c>0</c> is an empty word and a low half of <c>-1</c> a <i>dummy</i> — a deleted probe position that is
///         never reused within a generation. Probing is linear from a Fibonacci-hashed home position, load is kept
///         at or below two thirds, and a rebuild (a fresh generation) clears the dummies.
///     </para>
///     <para>
///         <b>Lock-free reads, one lock for writes.</b> Every read (key lookup, membership, <see cref="IndexOf" />,
///         positional get, <see cref="Count" />, enumeration, <see cref="ToArray" />, <see cref="Snapshot" />) takes
///         no lock. Publication is release/acquire: an append writes the slot fields, release-stores the index word
///         (the key path can now reach the slot), then release-stores the count (the list path can). Every shrinking
///         transition publishes a new generation object; a tail removal shares the arrays with the floor raised, so
///         an enumerator captured before it keeps reading valid data (the same three rules as the sibling type).
///         Writes are serialized by one lock, held only for the structural mutation — never while a user
///         <c>valueFactory</c>/<c>updateValueFactory</c> runs.
///     </para>
///     <para>
///         <b>The validated read.</b> A key-path hit reads the index word (acquire), the value (acquire), and then
///         re-reads the word: if it changed, the entry was removed or moved under the reader and the value may
///         belong to another key, so the probe restarts. Because dummied positions are never reused within a
///         generation there is no ABA. This is what lets <see cref="TryRemoveSwapBack" /> move an entry in place
///         while the key path stays tear-free — the only relaxation swap-back keeps is the list-path one
///         documented on the member.
///     </para>
///     <para>
///         <b>Costs to know.</b> Order-preserving removal of an interior entry is O(n) (fresh arrays, like
///         <see cref="List{T}.RemoveAt" /> plus the index); removing the last entry is O(1); replacing the value of
///         an existing key is O(1) for reference and machine-word value types and O(n) otherwise (a whole fresh
///         generation, so no reader can tear a wide value — use <see cref="AddRange" /> to batch such replacements).
///         A held enumerator or <see cref="ValuesView" /> pins the two captured arrays; a removed value stays
///         reachable until the next copy-on-write or <see cref="Clear" />.
///     </para>
///     <para>
///         <b>The <c>int</c>-key footgun</b>, the <b>callback contract</b> (user code the collection runs inside its
///         write lock — an <see cref="AddRange" /> source, a <see cref="RemoveWhere" /> predicate, a comparer — must
///         not write back; it is refused with <see cref="LockRecursionException" />) and the <b>consistency</b>
///         relaxations (the key path leads the list path by one operation; <see cref="IndexOf" /> may be stale by
///         concurrent removals) are exactly those of <see cref="ConcurrentOrderedDictionary{TKey,TValue}" />.
///     </para>
/// </remarks>
/// <typeparam name="TKey">The non-null key type; uniqueness and lookups use the configured comparer.</typeparam>
/// <typeparam name="TValue">The value type stored per key and yielded, in index order, by enumeration.</typeparam>
public sealed class ConcurrentOrderedCompactDictionary<TKey, TValue> : IReadOnlyList<TValue>
    where TKey : notnull
{
    /// <summary>The initial entry capacity, and the size the arrays first grow to from empty.</summary>
    private const int DefaultCapacity = 4;

    /// <summary>The low-half value of a dummy index word: a probe position whose entry was removed. Probes step past it; inserts never reuse it within a generation.</summary>
    private const int DummySlot = -1;

    /// <summary>
    ///     Whether an index word's tag can be the key's own bits instead of a hash — true for value-type keys of at
    ///     most four bytes that are primitives or enums (bit-equality is key-equality for them), never for
    ///     <see cref="float" /> (<c>-0.0 == 0.0</c> and <c>NaN.Equals(NaN)</c> disagree with bit-equality). With a
    ///     bit tag a hit never touches the keys array: one index line, one values line.
    /// </summary>
    private static readonly bool s_tagIsKey = ComputeTagIsKey();

    /// <summary>
    ///     Whether <see cref="TryRemoveSwapBack" /> may move an entry in place: both the key and the value must be
    ///     single-store atomic (ECMA-335 §12.6.6), otherwise a lock-free reader could tear the moved slot. Wide
    ///     types take the copy path. Also gates the validated read, which is only needed where slots can be
    ///     rewritten under a reader.
    /// </summary>
    private static readonly bool s_inPlaceSwapBack =
        ConcurrentDictionaryTypeProps<TKey>.IsWriteAtomic && ConcurrentDictionaryTypeProps<TValue>.IsWriteAtomic;

    /// <summary>
    ///     One generation of the table. Its arrays are shared by the holders that shrink it (tail removals,
    ///     swap-backs) and frozen once a copy-on-write generation supersedes it; readers hold whichever generation
    ///     they captured.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Rules, extending the sibling type's store rules to the index: (1) <see cref="_count" /> only grows on a
    ///         given instance (appends write slots above it, then release-publish it); every shrinking transition
    ///         publishes a new instance. (2) Slots a reader can see are never structurally rewritten, except the
    ///         documented in-place swap-back of atomic types; <see cref="_floor" /> freezes vacated tail slots
    ///         against in-place reuse. (3) Value slots are rewritten in place only for atomically-writable
    ///         <typeparamref name="TValue" />. (4) Index words are only ever written through
    ///         <see cref="Volatile" />/<see cref="Interlocked" /> (atomic on 32-bit runtimes too), a dummied position
    ///         is never reused, and generations never share an index or a keys array with a generation that has a
    ///         different values array.
    ///     </para>
    /// </remarks>
    private sealed class Tables
    {
        /// <summary>The open-addressed index: <c>(tag &lt;&lt; 32) | (slot + 1)</c> per position, <c>0</c> empty, low half <c>-1</c> dummy. Length is a power of two.</summary>
        internal readonly long[] _index;

        /// <summary>The Fibonacci-hash shift for this index length (<c>64 - log2(length)</c>).</summary>
        internal readonly int _shift;

        /// <summary>Keys in insertion order, occupying <c>[0, <see cref="_count" />)</c>.</summary>
        internal readonly TKey[] _keys;

        /// <summary>Values in insertion order, occupying <c>[0, <see cref="_count" />)</c> — the one copy both paths read.</summary>
        internal readonly TValue[] _values;

        /// <summary>The first slot an in-place append may write; slots below it were exposed to readers by an earlier, higher-count generation sharing these arrays.</summary>
        internal readonly int _floor;

        /// <summary>The number of dummied index positions this generation carries (writer bookkeeping for the rebuild threshold).</summary>
        internal readonly int _dummies;

        /// <summary>The number of live entries. Mutable and monotonically increasing on this instance; readers capture it once with <c>Volatile.Read</c>.</summary>
        internal int _count;

        /// <summary>Wraps the arrays of one generation.</summary>
        /// <param name="index">The index words (power-of-two length).</param>
        /// <param name="keys">The ordered keys.</param>
        /// <param name="values">The ordered values.</param>
        /// <param name="count">The live entry count at publication.</param>
        /// <param name="floor">The first slot an in-place append may write.</param>
        /// <param name="dummies">The dummied index positions carried.</param>
        internal Tables(long[] index, TKey[] keys, TValue[] values, int count, int floor, int dummies)
        {
            Debug.Assert(BitOperations.IsPow2(index.Length) && index.Length >= 2, "the index length must be a power of two of at least 2");
            _index = index;
            _shift = 64 - BitOperations.Log2((uint)index.Length);
            _keys = keys;
            _values = values;
            _count = count;
            _floor = floor;
            _dummies = dummies;
        }

        /// <summary>The entry capacity (the length of the key/value arrays).</summary>
        internal int Capacity => _keys.Length;

        /// <summary>The shared empty generation used before the first insertion and after <see cref="Clear" />. Never mutated: an append into zero capacity always grows into a fresh generation.</summary>
        internal static readonly Tables Empty = new(new long[2], Array.Empty<TKey>(), Array.Empty<TValue>(), 0, 0, 0);
    }

    /// <summary>The key comparer. <see langword="null" /> only for value-type keys under the default comparer (so the JIT devirtualizes the comparison); reference-type keys always store an instance.</summary>
    private readonly IEqualityComparer<TKey>? _comparer;

    /// <summary>Whether this instance tags index words with the key's bits (see <see cref="s_tagIsKey" />; also requires the default comparer).</summary>
    private readonly bool _tagIsKey;

    /// <summary>The current generation. Volatile so a fresh generation reaches lock-free readers with acquire/release ordering.</summary>
    private volatile Tables _tables;

    /// <summary>Serializes every mutation. Reads never acquire it.</summary>
    private readonly object _writeLock = new();

    /// <summary>Creates an empty collection using the default comparer for <typeparamref name="TKey" />.</summary>
    public ConcurrentOrderedCompactDictionary() : this((IEqualityComparer<TKey>?)null)
    {
    }

    /// <summary>Creates an empty collection using the supplied key comparer.</summary>
    /// <param name="comparer">The comparer used for key uniqueness and lookups, or <see langword="null" /> for the default.</param>
    public ConcurrentOrderedCompactDictionary(IEqualityComparer<TKey>? comparer) : this(0, comparer)
    {
    }

    /// <summary>Creates an empty collection with room reserved for <paramref name="capacity" /> entries.</summary>
    /// <param name="capacity">The number of entries to pre-size the arrays and the index for; must be non-negative.</param>
    /// <param name="comparer">The comparer used for key uniqueness and lookups, or <see langword="null" /> for the default.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="capacity" /> is negative, or larger than the ~715 million entries the index can address.</exception>
    public ConcurrentOrderedCompactDictionary(int capacity, IEqualityComparer<TKey>? comparer = null)
    {
        if (capacity < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity));
        }

        // The sibling concurrent dictionary's convention: reference-type keys always hold a comparer instance
        // (so the hot path never re-fetches EqualityComparer<TKey>.Default through shared generics), value-type
        // keys keep null for the default so the JIT devirtualizes the comparison entirely.
        if (!typeof(TKey).IsValueType)
        {
            _comparer = comparer ?? EqualityComparer<TKey>.Default;
        }
        else if (comparer is not null && !ReferenceEquals(comparer, EqualityComparer<TKey>.Default))
        {
            _comparer = comparer;
        }

        _tagIsKey = s_tagIsKey && _comparer is null;
        _tables = capacity == 0 ? Tables.Empty : new Tables(new long[IndexLengthFor(capacity)], new TKey[capacity], new TValue[capacity], 0, 0, 0);
    }

    /// <summary>
    ///     Creates a collection populated from <paramref name="source" /> in enumeration order, with Python
    ///     <c>dict(pairs)</c> duplicate semantics (last value wins, first-occurrence position).
    /// </summary>
    /// <param name="source">The key/value pairs to copy; repeated keys replace the value without moving the key.</param>
    /// <param name="comparer">The comparer used for key uniqueness and lookups, or <see langword="null" /> for the default.</param>
    /// <remarks>When the source can report its length without enumerating, the arrays and the index are pre-sized to it, so the build allocates exactly the live bytes.</remarks>
    /// <exception cref="ArgumentNullException"><paramref name="source" /> is <see langword="null" />.</exception>
    public ConcurrentOrderedCompactDictionary(IEnumerable<KeyValuePair<TKey, TValue>> source, IEqualityComparer<TKey>? comparer = null)
        : this(PreSizeOf(source), comparer)
    {
        AddRange(source);
    }

    /// <summary>Reads a capacity hint from a countable pairs source without enumerating it (0 when unknown).</summary>
    /// <param name="source">The constructor's source; a null slips through as 0 so <see cref="AddRange" /> can raise the contract's <see cref="ArgumentNullException" />.</param>
    /// <returns>The source's count when knowable without enumeration; otherwise 0.</returns>
    private static int PreSizeOf(IEnumerable<KeyValuePair<TKey, TValue>>? source)
        => source is not null && source.TryGetNonEnumeratedCount(out int count) ? count : 0;

    /// <summary>Gets the number of entries — a moment-in-time snapshot, consistent with lock-free enumeration.</summary>
    public int Count => Volatile.Read(ref _tables._count);

    /// <summary>Gets whether the collection currently has no entries.</summary>
    public bool IsEmpty => Count == 0;

    /// <summary>Gets the comparer used to determine key equality and hashing.</summary>
    public IEqualityComparer<TKey> Comparer => _comparer ?? EqualityComparer<TKey>.Default;

    // ---------------------------------------------------------------------------------------------------------
    // Tags, words, probing.
    // ---------------------------------------------------------------------------------------------------------

    /// <summary>Decides <see cref="s_tagIsKey" /> for the closed type (evaluated once; the JIT folds the readonly).</summary>
    /// <returns>Whether the key's bits can serve as the index tag.</returns>
    private static bool ComputeTagIsKey()
        => typeof(TKey).IsValueType
           && Unsafe.SizeOf<TKey>() <= sizeof(int)
           && (typeof(TKey).IsPrimitive || typeof(TKey).IsEnum)
           && typeof(TKey) != typeof(float);

    /// <summary>Reinterprets a small value-type key as its bits, zero-extended to an <see cref="int" /> (only called when <see cref="s_tagIsKey" />).</summary>
    /// <param name="key">The key.</param>
    /// <returns>The key's bits.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int KeyBits(TKey key)
    {
        if (Unsafe.SizeOf<TKey>() == 1)
        {
            return Unsafe.As<TKey, byte>(ref key);
        }

        if (Unsafe.SizeOf<TKey>() == 2)
        {
            return Unsafe.As<TKey, ushort>(ref key);
        }

        return Unsafe.As<TKey, int>(ref key);
    }

    /// <summary>Computes the tag an index word carries for <paramref name="key" />: its bits when <see cref="_tagIsKey" />, otherwise the comparer's hash code.</summary>
    /// <param name="key">The key.</param>
    /// <returns>The 32-bit tag.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int TagOf(TKey key)
    {
        if (_tagIsKey)
        {
            return KeyBits(key);
        }

        return typeof(TKey).IsValueType && _comparer is null
            ? EqualityComparer<TKey>.Default.GetHashCode(key)
            : _comparer!.GetHashCode(key);
    }

    /// <summary>Compares two keys with the configured comparer (devirtualized for value-type keys under the default).</summary>
    /// <param name="a">The stored key.</param>
    /// <param name="b">The key sought.</param>
    /// <returns>Whether the keys are equal.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool KeyEquals(TKey a, TKey b)
        => typeof(TKey).IsValueType && _comparer is null ? EqualityComparer<TKey>.Default.Equals(a, b) : _comparer!.Equals(a, b);

    /// <summary>Packs a tag and a slot field into an index word.</summary>
    /// <param name="tag">The key tag.</param>
    /// <param name="slotField">The 1-based slot, or <see cref="DummySlot" />.</param>
    /// <returns>The word.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long Word(int tag, int slotField) => ((long)tag << 32) | (uint)slotField;

    /// <summary>The home probe position of a tag in an index with the given shift (Fibonacci hashing spreads sequential and patterned keys evenly over the power-of-two table).</summary>
    /// <param name="tag">The key tag.</param>
    /// <param name="shift">The generation's <see cref="Tables._shift" />.</param>
    /// <returns>The starting position.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Home(int tag, int shift) => (uint)((ulong)(uint)tag * 0x9E3779B97F4A7C15UL >> shift);

    /// <summary>The index length for an entry capacity: the power of two above 1.5× the capacity, so a full table sits at or below two-thirds load.</summary>
    /// <param name="capacity">The entry capacity.</param>
    /// <returns>The index length.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The capacity exceeds what a power-of-two <see cref="long" /> index can address.</exception>
    private static int IndexLengthFor(int capacity)
    {
        long wanted = Math.Max(8, (long)capacity * 3 / 2 + 1);
        if (wanted > 1 << 30)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity), "the compact index cannot address more than ~715 million entries");
        }

        return (int)BitOperations.RoundUpToPowerOf2((uint)wanted);
    }

    /// <summary>Whether adding <paramref name="live" /> live entries to an index carrying <paramref name="dummies" /> would exceed two-thirds load (the rebuild trigger; also what bounds probe lengths).</summary>
    /// <param name="live">The live entries after the insert.</param>
    /// <param name="dummies">The dummied positions.</param>
    /// <param name="indexLength">The index length.</param>
    /// <returns>Whether a rebuild is required first.</returns>
    private static bool OverLoad(int live, int dummies, int indexLength) => ((long)live + dummies) * 3 > (long)indexLength * 2;

    /// <summary>
    ///     Reads a value slot with acquire semantics so the validating re-read of the index word that follows it
    ///     cannot be reordered ahead of it (a plain load may complete after a later acquire load on weakly-ordered
    ///     hardware, which would validate before reading). Compiles to a plain load on x64 and to an acquire load
    ///     on ARM64; atomic on 32-bit runtimes for 8-byte primitives.
    /// </summary>
    /// <param name="slot">The value slot.</param>
    /// <returns>The value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TValue ReadValueAcquire(ref TValue slot)
    {
        if (!typeof(TValue).IsValueType)
        {
            object? boxed = Volatile.Read(ref Unsafe.As<TValue, object?>(ref slot));
            return Unsafe.As<object?, TValue>(ref boxed);
        }

        if (!RuntimeHelpers.IsReferenceOrContainsReferences<TValue>())
        {
            if (Unsafe.SizeOf<TValue>() == 1)
            {
                byte b = Volatile.Read(ref Unsafe.As<TValue, byte>(ref slot));
                return Unsafe.As<byte, TValue>(ref b);
            }

            if (Unsafe.SizeOf<TValue>() == 2)
            {
                ushort s = Volatile.Read(ref Unsafe.As<TValue, ushort>(ref slot));
                return Unsafe.As<ushort, TValue>(ref s);
            }

            if (Unsafe.SizeOf<TValue>() == 4)
            {
                uint u = Volatile.Read(ref Unsafe.As<TValue, uint>(ref slot));
                return Unsafe.As<uint, TValue>(ref u);
            }

            if (Unsafe.SizeOf<TValue>() == 8)
            {
                ulong l = Volatile.Read(ref Unsafe.As<TValue, ulong>(ref slot));
                return Unsafe.As<ulong, TValue>(ref l);
            }
        }

        // Wide or reference-containing value type: never rewritten in place (those take the copy path), so the
        // copy cannot tear; the fence keeps the validating re-read after it.
        TValue v = slot;
        Interlocked.MemoryBarrier();
        return v;
    }

    /// <summary>
    ///     Under the write lock: walks <paramref name="key" />'s probe sequence and returns the position of its word,
    ///     or -1 with <paramref name="emptyPos" /> set to the first empty position (where an insert of this key
    ///     goes — dummies are never reused).
    /// </summary>
    /// <param name="t">The generation to probe.</param>
    /// <param name="key">The key.</param>
    /// <param name="tag">The key's tag.</param>
    /// <param name="emptyPos">On a miss, the first empty position of the probe sequence; otherwise -1.</param>
    /// <returns>The word position of the key, or -1.</returns>
    private int ProbeUnderLock(Tables t, TKey key, int tag, out int emptyPos)
    {
        long[] index = t._index;
        uint mask = (uint)index.Length - 1;
        uint p = Home(tag, t._shift);
        while (true)
        {
            long e = index[p];
            int slotField = (int)e;
            if (slotField == 0)
            {
                emptyPos = (int)p;
                return -1;
            }

            if (slotField > 0 && (int)(e >> 32) == tag && (_tagIsKey || KeyEquals(t._keys[slotField - 1], key)))
            {
                emptyPos = -1;
                return (int)p;
            }

            p = (p + 1) & mask;
        }
    }

    /// <summary>Under the write lock: the first empty position of <paramref name="tag" />'s probe sequence in a generation known not to contain the key.</summary>
    /// <param name="t">The generation.</param>
    /// <param name="tag">The key's tag.</param>
    /// <returns>The insert position.</returns>
    private static int FirstEmpty(Tables t, int tag)
    {
        long[] index = t._index;
        uint mask = (uint)index.Length - 1;
        uint p = Home(tag, t._shift);
        while ((int)index[p] != 0)
        {
            p = (p + 1) & mask;
        }

        return (int)p;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Generation construction (all copy-on-write; every allocation happens before any published state changes).
    // ---------------------------------------------------------------------------------------------------------

    /// <summary>Inserts every live word of <paramref name="oldIndex" /> into <paramref name="index" /> (a fresh, private index), dropping dummies — no re-hashing, the tags travel with the words.</summary>
    /// <param name="oldIndex">The source index.</param>
    /// <param name="index">The destination index (all zero).</param>
    /// <param name="shift">The destination's shift.</param>
    /// <param name="slotMap">Optional old-slot → new-slot map (-1 = dropped); <see langword="null" /> keeps slots.</param>
    private static void RebuildIndex(long[] oldIndex, long[] index, int shift, int[]? slotMap)
    {
        uint mask = (uint)index.Length - 1;
        for (int i = 0; i < oldIndex.Length; i++)
        {
            long e = oldIndex[i];
            int slotField = (int)e;
            if (slotField <= 0)
            {
                continue; // empty or dummy
            }

            if (slotMap is not null)
            {
                int mapped = slotMap[slotField - 1];
                if (mapped < 0)
                {
                    continue; // removed by the compaction that supplied the map
                }

                e = Word((int)(e >> 32), mapped + 1);
            }

            uint p = Home((int)(e >> 32), shift);
            while ((int)index[p] != 0)
            {
                p = (p + 1) & mask;
            }

            index[p] = e;
        }
    }

    /// <summary>
    ///     A fresh generation holding the same entries with capacity <paramref name="newCapacity" />: keys and values
    ///     copied, the index rebuilt from the old words (dummies cleared). Used for growth, for an append blocked by
    ///     the floor, for the load-triggered rebuild, and for every whole-generation copy-on-write.
    /// </summary>
    /// <param name="t">The current generation.</param>
    /// <param name="n">The live entry count to carry (the generation's count, or a batch's local count that includes pending in-place appends).</param>
    /// <param name="newCapacity">The new entry capacity (at least <paramref name="n" />).</param>
    /// <returns>The private, not-yet-published generation.</returns>
    private static Tables CopyGeneration(Tables t, int n, int newCapacity)
    {
        var keys = new TKey[newCapacity];
        var values = new TValue[newCapacity];
        Array.Copy(t._keys, keys, n);
        Array.Copy(t._values, values, n);
        var index = new long[IndexLengthFor(newCapacity)];
        var fresh = new Tables(index, keys, values, n, 0, 0);
        RebuildIndex(t._index, index, fresh._shift, null);
        return fresh;
    }

    /// <summary>
    ///     A fresh generation without the entry at <paramref name="removed" />: the tail shifted down one slot, and
    ///     the index either renumbered in one sequential pass (<c>slot &gt; removed → slot − 1</c>, the removed
    ///     word → dummy) or, when dummies would outnumber live entries, rebuilt clean.
    /// </summary>
    /// <param name="t">The current generation.</param>
    /// <param name="removed">The insertion-order position to drop.</param>
    /// <returns>The private, not-yet-published generation.</returns>
    private static Tables CopyGenerationWithout(Tables t, int removed)
    {
        int n = t._count;
        int cap = t.Capacity;
        var keys = new TKey[cap];
        var values = new TValue[cap];
        Array.Copy(t._keys, keys, removed);
        Array.Copy(t._keys, removed + 1, keys, removed, n - removed - 1);
        Array.Copy(t._values, values, removed);
        Array.Copy(t._values, removed + 1, values, removed, n - removed - 1);

        long[] old = t._index;
        var index = new long[old.Length];
        if (t._dummies + 1 > n - 1)
        {
            // Too many dead positions to carry: rebuild clean from the live words with a shifting map.
            var map = new int[n];
            for (int s = 0; s < n; s++)
            {
                map[s] = s < removed ? s : s == removed ? -1 : s - 1;
            }

            var fresh = new Tables(index, keys, values, n - 1, 0, 0);
            RebuildIndex(old, index, fresh._shift, map);
            return fresh;
        }

        int removedField = removed + 1;
        for (int i = 0; i < old.Length; i++)
        {
            long e = old[i];
            int slotField = (int)e;
            if (slotField == removedField)
            {
                index[i] = Word((int)(e >> 32), DummySlot);
            }
            else if (slotField > removedField)
            {
                index[i] = e - 1; // the low half is ≥ 2, so the decrement never borrows into the tag
            }
            else
            {
                index[i] = e;
            }
        }

        return new Tables(index, keys, values, n - 1, 0, t._dummies + 1);
    }

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
            throw new KeyNotFoundException(SR.Format(SR.Arg_KeyNotFoundWithKey, key.ToString()));
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
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public void SetByKey(TKey key, TValue value)
    {
        NullCheck(key);
        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            Tables t = _tables;
            int tag = TagOf(key);
            int pos = ProbeUnderLock(t, key, tag, out _);
            if (pos >= 0)
            {
                ReplaceExistingUnderLock(t, (int)t._index[pos] - 1, value);
            }
            else
            {
                bool added = TryAppendUnderLock(key, value, tag);
                Debug.Assert(added, "key vanished between the probe and the append while holding the write lock");
            }
        }
    }

    /// <summary>Determines whether <paramref name="key" /> is present. Lock-free, O(1) and allocation-free.</summary>
    /// <param name="key">The key to test.</param>
    /// <returns><see langword="true" /> if the key exists; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool ContainsKey(TKey key)
    {
        NullCheck(key);
        Tables t = _tables;
        int tag = TagOf(key);
        long[] index = t._index;
        uint mask = (uint)index.Length - 1;
        uint p = Home(tag, t._shift);
        while (true)
        {
            long e = Volatile.Read(ref index[p]);
            int slotField = (int)e;
            if (slotField == 0)
            {
                return false;
            }

            if (slotField > 0 && (int)(e >> 32) == tag && (_tagIsKey || KeyEquals(t._keys[slotField - 1], key)))
            {
                return true;
            }

            p = (p + 1) & mask;
        }
    }

    /// <summary>
    ///     Attempts to read the value for <paramref name="key" /> without throwing. Lock-free and O(1): one index
    ///     line and one values line for small value-type keys under the default comparer (the tag is the key), one
    ///     keys line more otherwise.
    /// </summary>
    /// <param name="key">The key to look up.</param>
    /// <param name="value">On success, the associated value; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if the key was found; otherwise <see langword="false" />.</returns>
    /// <remarks>The validated read: after loading the value the index word is re-read, and a changed word (the entry was removed or moved by a concurrent swap-back) restarts the probe instead of returning a value that may belong to another key.</remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetValue(TKey key, out TValue value)
    {
        NullCheck(key);
        Tables t = _tables;
        int tag = TagOf(key);
        long[] index = t._index;
        uint mask = (uint)index.Length - 1;
        uint p = Home(tag, t._shift);
        while (true)
        {
            long e = Volatile.Read(ref index[p]); // acquire: the slot fields were written before this word was published
            int slotField = (int)e;
            if (slotField == 0)
            {
                value = default!;
                return false;
            }

            if (slotField > 0 && (int)(e >> 32) == tag && (_tagIsKey || KeyEquals(t._keys[slotField - 1], key)))
            {
                if (!s_inPlaceSwapBack)
                {
                    // Slots of wide types are never rewritten in place, so the value cannot belong to another key.
                    value = t._values[slotField - 1];
                    return true;
                }

                TValue v = ReadValueAcquire(ref t._values[slotField - 1]);
                if (Volatile.Read(ref index[p]) == e)
                {
                    value = v;
                    return true;
                }

                // The word changed under us: the entry was dummied (removed) or re-pointed (swap-back moved the key),
                // so the value read may be another key's. Restart from home; the position is never reused, so a
                // second identical word can only mean nothing changed.
                p = Home(tag, t._shift);
                continue;
            }

            p = (p + 1) & mask;
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
            return TryAppendUnderLock(key, value, TagOf(key));
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
            throw new ArgumentException(SR.ConcurrentDictionary_KeyAlreadyExisted, nameof(key));
        }
    }

    /// <summary>
    ///     Upserts every pair from <paramref name="source" /> under <b>one</b> lock acquisition, with the same
    ///     duplicate semantics as the enumerable constructor (existing keys get the new value in their existing
    ///     position; new keys append in enumeration order). Newly appended entries become visible to the list
    ///     path <b>atomically as one batch</b>; wide-value replacements cost one copy-on-write for the whole batch.
    /// </summary>
    /// <param name="source">The key/value pairs to upsert; consumed once, in order.</param>
    /// <remarks>The source is enumerated inside the write lock, so it must not call back into this collection's write operations (refused with <see cref="LockRecursionException" />). The key path leads per pair, as it does for every write.</remarks>
    /// <exception cref="ArgumentNullException"><paramref name="source" /> or a key inside it is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public void AddRange(IEnumerable<KeyValuePair<TKey, TValue>> source)
    {
        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            Tables published = _tables;
            Tables target = published;
            int count = target._count; // plain read: every count writer holds this lock

            // `fresh` = target is a private generation invisible to readers: plain stores everywhere, one publish
            // at the end. While NOT fresh, appends write slots ≥ count and publish their word in place (key path
            // leads per pair); the count is published once, in the finally, so the batch is atomic on the list path.
            bool fresh = false;

            if (source.TryGetNonEnumeratedCount(out int incoming) && incoming > 0)
            {
                long needed = (long)count + incoming;
                if (needed > target.Capacity || count < target._floor || OverLoad((int)Math.Min(needed, int.MaxValue), target._dummies, target._index.Length))
                {
                    // Grow (amortized-double) ONLY when genuinely out of capacity. A rebuild forced merely by the
                    // floor rule or by accumulated dummies (OverLoad) — while `needed` still fits the current
                    // capacity — already has room, and the fresh generation drops the dummies on its own. Doubling
                    // there would compound the capacity on every batch that follows a removal (which raises the
                    // floor / leaves dummies), running the compact index up to its ~715M ceiling while the live
                    // count stays flat. Keep the capacity — the same split the new-key branch below already makes.
                    int newCap = needed > target.Capacity
                        ? (int)Math.Min(Math.Max(needed, target.Capacity == 0 ? DefaultCapacity : (long)target.Capacity * 2), Array.MaxLength)
                        : target.Capacity;
                    target = CopyGeneration(target, count, newCap);
                    fresh = true;
                }
            }

            try
            {
                foreach (KeyValuePair<TKey, TValue> kv in source)
                {
                    NullCheck(kv.Key);
                    int tag = TagOf(kv.Key);
                    int pos = ProbeUnderLock(target, kv.Key, tag, out int emptyPos);

                    if (pos >= 0)
                    {
                        // Existing key: replace the value, keep the position (Python dict(pairs) semantics).
                        int slot = (int)target._index[pos] - 1;
                        if (ConcurrentDictionaryTypeProps<TValue>.IsWriteAtomic)
                        {
                            target._values[slot] = kv.Value; // private array when fresh; documented in-place update otherwise
                        }
                        else
                        {
                            // A wide value is never stored into a live slot: go copy-on-write for the rest of the batch
                            // (the copy carries the batch's pending in-place appends: their words are already in the
                            // index being cloned and their slots lie below the local count).
                            if (!fresh)
                            {
                                target = CopyGeneration(target, count, target.Capacity);
                                fresh = true;
                            }

                            target._values[slot] = kv.Value;
                        }

                        continue;
                    }

                    // New key. Copy/grow BEFORE touching anything (strand-proofing: an allocation failure leaves
                    // both paths exactly as they were — the published generation's count is never touched here).
                    if (count == target.Capacity || (!fresh && count < target._floor) || OverLoad(count + 1, target._dummies, target._index.Length))
                    {
                        int newCap = count == target.Capacity
                            ? (target.Capacity == 0 ? DefaultCapacity : (int)Math.Min((long)target.Capacity * 2, Array.MaxLength))
                            : target.Capacity;
                        target = CopyGeneration(target, count, newCap);
                        fresh = true;
                        emptyPos = FirstEmpty(target, tag);
                    }

                    target._keys[count] = kv.Key;
                    target._values[count] = kv.Value;
                    long word = Word(tag, count + 1);
                    if (fresh)
                    {
                        target._index[emptyPos] = word;
                    }
                    else
                    {
                        Volatile.Write(ref target._index[emptyPos], word);
                    }

                    count++;
                }
            }
            finally
            {
                // Publish once — including when the source threw mid-batch: pairs whose words are already
                // published MUST reach the list path too, or a key would resolve yet never enumerate.
                if (fresh)
                {
                    target._count = count;
                    _tables = target;
                }
                else if (count != published._count)
                {
                    Volatile.Write(ref published._count, count);
                }
            }
        }
    }

    /// <summary>Returns the existing value for <paramref name="key" />, or appends and returns <paramref name="value" /> if absent.</summary>
    /// <param name="key">The key to read or add.</param>
    /// <param name="value">The value to append if the key is absent.</param>
    /// <returns>The existing value, or the newly added <paramref name="value" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public TValue GetOrAdd(TKey key, TValue value)
    {
        if (TryGetValue(key, out TValue existing))
        {
            return existing; // lock-free hit: the hot path for a cache-shaped workload
        }

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            int tag = TagOf(key);
            if (TryAppendUnderLock(key, value, tag))
            {
                return value;
            }

            // Lost the race for this key between the lock-free miss and the lock: read the winner's value.
            Tables t = _tables;
            int pos = ProbeUnderLock(t, key, tag, out _);
            Debug.Assert(pos >= 0, "append failed for an existing key that then vanished while holding the write lock");
            return t._values[(int)t._index[pos] - 1];
        }
    }

    /// <summary>Returns the existing value for <paramref name="key" />, or appends the result of <paramref name="valueFactory" /> if absent.</summary>
    /// <param name="key">The key to read or add.</param>
    /// <param name="valueFactory">Invoked with the key to produce the value when the key is absent. Runs <b>outside</b> the write lock; under contention it may be called on more than one thread, with the losing result discarded.</param>
    /// <returns>The existing value, or the newly produced and added value.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> or <paramref name="valueFactory" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public TValue GetOrAdd(TKey key, Func<TKey, TValue> valueFactory)
    {
        NullCheck(key);
        if (valueFactory is null)
        {
            throw new ArgumentNullException(nameof(valueFactory));
        }

        if (TryGetValue(key, out TValue existing))
        {
            return existing;
        }

        TValue value = valueFactory(key); // outside the lock: writers are never blocked on user code

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            int tag = TagOf(key);
            if (TryAppendUnderLock(key, value, tag))
            {
                return value;
            }

            Tables t = _tables;
            int pos = ProbeUnderLock(t, key, tag, out _);
            Debug.Assert(pos >= 0, "append failed for an existing key that then vanished while holding the write lock");
            return t._values[(int)t._index[pos] - 1]; // lost the race for this key; discard the value we computed
        }
    }

    /// <summary>Appends <paramref name="addValue" /> if the key is absent, otherwise replaces the value via <paramref name="updateValueFactory" />.</summary>
    /// <param name="key">The key to add or update.</param>
    /// <param name="addValue">The value to append if the key is absent.</param>
    /// <param name="updateValueFactory">Invoked with the key and current value to produce the replacement. Runs <b>outside</b> the write lock; applied with a compare-and-swap and retried if the value moved, so it may run more than once under contention.</param>
    /// <returns>The value now stored for the key.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> or <paramref name="updateValueFactory" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public TValue AddOrUpdate(TKey key, TValue addValue, Func<TKey, TValue, TValue> updateValueFactory)
    {
        NullCheck(key);
        if (updateValueFactory is null)
        {
            throw new ArgumentNullException(nameof(updateValueFactory));
        }

        while (true)
        {
            if (TryGetValue(key, out TValue old))
            {
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
                if (TryAppendUnderLock(key, addValue, TagOf(key)))
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
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public bool TryUpdate(TKey key, TValue newValue, TValue comparisonValue)
    {
        NullCheck(key);
        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            Tables t = _tables;
            int pos = ProbeUnderLock(t, key, TagOf(key), out _);
            if (pos < 0)
            {
                return false;
            }

            int slot = (int)t._index[pos] - 1;
            if (!EqualityComparer<TValue>.Default.Equals(t._values[slot], comparisonValue))
            {
                return false;
            }

            ReplaceExistingUnderLock(t, slot, newValue);
            return true;
        }
    }

    /// <summary>
    ///     Removes <paramref name="key" /> and its value. Removing the <b>last</b> entry is O(1); removing an
    ///     interior entry shifts later entries down to keep indices contiguous — <b>O(n)</b>. For bulk removal use
    ///     <see cref="RemoveWhere" />; for order-insensitive O(1) removal use <see cref="TryRemoveSwapBack" />.
    /// </summary>
    /// <param name="key">The key to remove.</param>
    /// <param name="value">On success, the removed value; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if an entry was removed; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public bool TryRemove(TKey key, out TValue value)
    {
        // Lock-free fast negative: the interior work under the lock is O(n), so absent-key callers skip it.
        if (!ContainsKey(key))
        {
            value = default!;
            return false;
        }

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            Tables t = _tables;
            int pos = ProbeUnderLock(t, key, TagOf(key), out _);
            if (pos < 0)
            {
                value = default!;
                return false;
            }

            int slot = (int)t._index[pos] - 1;
            value = t._values[slot];
            RemoveCoreUnderLock(t, pos, slot);
            return true;
        }
    }

    /// <summary>Removes <paramref name="key" /> if present. Convenience wrapper over <see cref="TryRemove" />; same O(1)-tail / O(n)-interior cost.</summary>
    /// <param name="key">The key to remove.</param>
    /// <returns><see langword="true" /> if an entry was removed; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public bool Remove(TKey key) => TryRemove(key, out _);

    /// <summary>
    ///     Removes <paramref name="key" /> in O(1) by <b>moving the current last entry into the removed slot</b> —
    ///     the order-breaking escape hatch from the O(n) shifting removal. Only the moved entry's position changes.
    /// </summary>
    /// <param name="key">The key to remove.</param>
    /// <param name="value">On success, the removed value; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if an entry was removed; otherwise <see langword="false" />.</returns>
    /// <remarks>
    ///     <para>
    ///         Gives up two guarantees the ordinary removal keeps: <b>order</b> (the last entry relocates) and, when
    ///         both <typeparamref name="TKey" /> and <typeparamref name="TValue" /> are single-store atomic so the
    ///         move is done in place, <b>snapshot purity of concurrent list-path readers</b>: an enumerator already
    ///         running over the same arrays can see the moved entry at both its old and new position and not the
    ///         removed one, and a reader combining that one slot's key and value during the swap window
    ///         (<see cref="GetKeyAt" /> + <see cref="TryGetAt" />, or <see cref="Pairs" /> passing the slot) can see
    ///         them transiently mixed. Wide types take a copy: pure snapshots, O(n) memcpy.
    ///     </para>
    ///     <para>
    ///         <b>The key path is never affected</b>: the removed key's index word is dummied with a full fence
    ///         before the slot is overwritten, the moved key is re-pointed by one atomic word store after it, and
    ///         <see cref="TryGetValue" /> re-validates the word after reading the value — a lookup of either key
    ///         returns that key's own value or reports it absent, never another key's value.
    ///     </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public bool TryRemoveSwapBack(TKey key, out TValue value)
    {
        if (!ContainsKey(key))
        {
            value = default!;
            return false; // lock-free fast negative
        }

        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            Tables t = _tables;
            int tag = TagOf(key);
            int pos = ProbeUnderLock(t, key, tag, out _);
            if (pos < 0)
            {
                value = default!;
                return false;
            }

            int slot = (int)t._index[pos] - 1;
            value = t._values[slot];
            int n = t._count;
            int last = n - 1;

            if (slot == last)
            {
                RemoveTailUnderLock(t, pos, tag);
                return true;
            }

            TKey movedKey = t._keys[last];
            int movedTag = TagOf(movedKey);
            int movedPos = ProbeUnderLock(t, movedKey, movedTag, out _);
            Debug.Assert(movedPos >= 0 && (int)t._index[movedPos] - 1 == last, "the last entry's word must resolve to the last slot under the write lock");

            if (s_inPlaceSwapBack)
            {
                // Allocate the successor BEFORE any mutation (strand-proofing), then: (1) dummy the removed word
                // with a FULL fence, so no later store can become visible before the key is gone; (2) overwrite the
                // slot — reachable only through the list path now (the documented relaxation); (3) re-point the moved
                // key with one release store; (4) publish the shrunken generation, the floor freezing the old tail.
                Tables next = new(t._index, t._keys, t._values, n - 1, Math.Max(t._floor, n), t._dummies + 1);
                Interlocked.Exchange(ref t._index[pos], Word(tag, DummySlot));
                t._keys[slot] = movedKey;
                t._values[slot] = t._values[last];
                Volatile.Write(ref t._index[movedPos], Word(movedTag, slot + 1));
                _tables = next;
                return true;
            }

            // Wide key or value: the move happens in a private copy (snapshot-pure), then publishes. Both words are
            // located in the rebuilt index BEFORE the slot is overwritten — the probe compares keys through the
            // slot, and the removed key's word would no longer match once the moved key sits there.
            Tables copy = CopyGeneration(t, n, t.Capacity);
            long[] index = copy._index;
            int copyPos = ProbeUnderLock(copy, key, tag, out _);
            int copyMovedPos = ProbeUnderLock(copy, movedKey, movedTag, out _);
            Debug.Assert(copyPos >= 0 && copyMovedPos >= 0, "both words must exist in the rebuilt private index");
            copy._keys[slot] = movedKey;
            copy._values[slot] = t._values[last];
            index[copyPos] = Word(tag, DummySlot);
            index[copyMovedPos] = Word(movedTag, slot + 1);
            _tables = new Tables(index, copy._keys, copy._values, n - 1, 0, 1);
            return true;
        }
    }

    /// <summary>
    ///     Removes every entry matching <paramref name="match" /> in <b>one compaction pass</b> — O(n) total instead
    ///     of O(n) per removed entry, publishing a single new generation whose index is repaired in one sequential
    ///     renumbering pass.
    /// </summary>
    /// <param name="match">Returns <see langword="true" /> for entries to remove; receives each key and its current value in index order.</param>
    /// <returns>The number of entries removed.</returns>
    /// <remarks>The predicate runs <b>inside</b> the write lock (that is what makes the pass atomic), so it must be fast and must not call back into this collection's write operations (refused with <see cref="LockRecursionException" />). A throwing predicate is survivable: the pass lands in the consistent "removed everything matched so far" state and the exception propagates.</remarks>
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
            Tables t = _tables;
            int n = t._count;

            // First pass: find the first match so the no-op case allocates nothing.
            int first = 0;
            while (first < n && !match(t._keys[first], t._values[first]))
            {
                first++;
            }

            if (first == n)
            {
                return 0;
            }

            // Compact the survivors into fresh arrays (readers keep the old generation); the slot map drives the
            // index repair afterwards.
            var keys = new TKey[t.Capacity];
            var values = new TValue[t.Capacity];
            var map = new int[n];
            Array.Copy(t._keys, keys, first);
            Array.Copy(t._values, values, first);
            for (int s = 0; s < first; s++)
            {
                map[s] = s;
            }

            int w = first;
            int removed = 0;
            int i = first;
            try
            {
                for (; i < n; i++)
                {
                    TKey k = t._keys[i];
                    TValue v = t._values[i];
                    if (match(k, v))
                    {
                        map[i] = -1;
                        removed++;
                    }
                    else
                    {
                        keys[w] = k;
                        values[w] = v;
                        map[i] = w;
                        w++;
                    }
                }
            }
            finally
            {
                // A throwing predicate must not strand the pass: carry every not-yet-processed entry — INCLUDING
                // the one whose predicate threw — over as a survivor and publish the consistent prefix state.
                for (; i < n; i++)
                {
                    keys[w] = t._keys[i];
                    values[w] = t._values[i];
                    map[i] = w;
                    w++;
                }

                var index = new long[t._index.Length];
                int dummies;
                if (removed + t._dummies > w)
                {
                    // More dead positions than live entries: rebuild clean instead of renumbering.
                    var fresh = new Tables(index, keys, values, w, 0, 0);
                    RebuildIndex(t._index, index, fresh._shift, map);
                    dummies = 0;
                }
                else
                {
                    long[] old = t._index;
                    for (int p = 0; p < old.Length; p++)
                    {
                        long e = old[p];
                        int slotField = (int)e;
                        if (slotField <= 0)
                        {
                            index[p] = e; // empty or dummy, unchanged
                        }
                        else
                        {
                            int mapped = map[slotField - 1];
                            index[p] = mapped < 0 ? Word((int)(e >> 32), DummySlot) : Word((int)(e >> 32), mapped + 1);
                        }
                    }

                    dummies = t._dummies + removed;
                }

                _tables = new Tables(index, keys, values, w, 0, dummies);
            }

            return removed;
        }
    }

    // ---------------------------------------------------------------------------------------------------------
    // List path (index semantics). Enumeration and ToArray are here too, yielding values in index order.
    // ---------------------------------------------------------------------------------------------------------

    /// <summary>
    ///     Gets or sets the value at insertion-order position <paramref name="index" />. The setter replaces the
    ///     value at that position, leaving its key and position unchanged (it never inserts). This indexer wins over
    ///     <see cref="this[TKey]" /> when <typeparamref name="TKey" /> is <c>int</c>.
    /// </summary>
    /// <param name="index">The zero-based position in insertion order.</param>
    /// <value>The value at <paramref name="index" />.</value>
    /// <remarks>The getter pays one volatile generation read per call; for a hot scan take <see cref="Snapshot" /> once and index the view.</remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
    public TValue this[int index]
    {
        get
        {
            Tables t = _tables;
            if ((uint)index >= (uint)Volatile.Read(ref t._count))
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            return t._values[index];
        }
        set => SetAt(index, value);
    }

    /// <summary>Replaces the value at position <paramref name="index" /> without changing its key or position — one store, no key lookup (both paths read the same slot). Serialized with other writes.</summary>
    /// <param name="index">The zero-based position in insertion order.</param>
    /// <param name="value">The replacement value.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public void SetAt(int index, TValue value)
    {
        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            Tables t = _tables;
            if ((uint)index >= (uint)t._count)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            ReplaceExistingUnderLock(t, index, value);
        }
    }

    /// <summary>Attempts to read the value at position <paramref name="index" /> without throwing. Lock-free and O(1).</summary>
    /// <param name="index">The zero-based position in insertion order.</param>
    /// <param name="value">On success, the value at the position; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if <paramref name="index" /> is in range; otherwise <see langword="false" />.</returns>
    public bool TryGetAt(int index, out TValue value)
    {
        Tables t = _tables;
        if ((uint)index >= (uint)Volatile.Read(ref t._count))
        {
            value = default!;
            return false;
        }

        value = t._values[index];
        return true;
    }

    /// <summary>Returns the key at position <paramref name="index" /> in insertion order. Lock-free and O(1).</summary>
    /// <param name="index">The zero-based position in insertion order.</param>
    /// <returns>The key stored at that position.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
    public TKey GetKeyAt(int index)
    {
        Tables t = _tables;
        if ((uint)index >= (uint)Volatile.Read(ref t._count))
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        return t._keys[index];
    }

    /// <summary>Returns the insertion-order position of <paramref name="key" />, or -1 if absent — the slot number its index word carries, no stored position field. Lock-free and O(1).</summary>
    /// <param name="key">The key to locate.</param>
    /// <returns>The zero-based index of the key, or -1 if it is not present.</returns>
    /// <remarks>Under a concurrent structural writer the returned index is a snapshot and may be stale by the number of concurrent removals.</remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public int IndexOf(TKey key)
    {
        NullCheck(key);
        Tables t = _tables;
        int tag = TagOf(key);
        long[] index = t._index;
        uint mask = (uint)index.Length - 1;
        uint p = Home(tag, t._shift);
        while (true)
        {
            long e = Volatile.Read(ref index[p]);
            int slotField = (int)e;
            if (slotField == 0)
            {
                return -1;
            }

            if (slotField > 0 && (int)(e >> 32) == tag && (_tagIsKey || KeyEquals(t._keys[slotField - 1], key)))
            {
                return slotField - 1;
            }

            p = (p + 1) & mask;
        }
    }

    /// <summary>Attempts to map <paramref name="key" /> to its insertion-order position without allocating a sentinel.</summary>
    /// <param name="key">The key to locate.</param>
    /// <param name="index">On success, the key's index; otherwise -1.</param>
    /// <returns><see langword="true" /> if the key was found; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool TryGetIndex(TKey key, out int index)
    {
        index = IndexOf(key);
        return index >= 0;
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
            Tables t = _tables;
            if ((uint)index >= (uint)t._count)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            TKey key = t._keys[index];
            int pos = ProbeUnderLock(t, key, TagOf(key), out _);
            Debug.Assert(pos >= 0 && (int)t._index[pos] - 1 == index, "a live slot's key must resolve to that slot under the write lock");
            RemoveCoreUnderLock(t, pos, index);
        }
    }

    /// <summary>Removes all entries and resets both paths to empty.</summary>
    /// <exception cref="LockRecursionException">Invoked reentrantly from user code running inside the collection's write lock (see <see cref="ThrowIfReentrantWrite" />).</exception>
    public void Clear()
    {
        ThrowIfReentrantWrite();

        lock (_writeLock)
        {
            _tables = Tables.Empty;
        }
    }

    /// <summary>Copies the values, in index order, into a new array — the same content and order as enumeration (a contiguous block copy).</summary>
    /// <returns>A new array of the values in insertion order (a point-in-time snapshot).</returns>
    public TValue[] ToArray()
    {
        Tables t = _tables;
        int n = Volatile.Read(ref t._count);
        if (n == 0)
        {
            return Array.Empty<TValue>();
        }

        var array = new TValue[n];
        Array.Copy(t._values, array, n);
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

        Tables t = _tables;
        int n = Volatile.Read(ref t._count);
        if (array.Length - arrayIndex < n)
        {
            throw new ArgumentException(SR.ConcurrentDictionary_ArrayNotLargeEnough, nameof(array));
        }

        Array.Copy(t._values, 0, array, arrayIndex, n);
    }

    /// <summary>Gets a point-in-time array of the keys in insertion order (a contiguous block copy), parallel to <see cref="Values" />.</summary>
    public TKey[] Keys
    {
        get
        {
            Tables t = _tables;
            int n = Volatile.Read(ref t._count);
            if (n == 0)
            {
                return Array.Empty<TKey>();
            }

            var keys = new TKey[n];
            Array.Copy(t._keys, keys, n);
            return keys;
        }
    }

    /// <summary>Gets a point-in-time array of the values in insertion order (equivalent to <see cref="ToArray" />).</summary>
    public TValue[] Values => ToArray();

    /// <summary>Enumerates the entries as key/value pairs in insertion order — the dictionary-shaped view. A snapshot is taken when enumeration begins.</summary>
    public IEnumerable<KeyValuePair<TKey, TValue>> Pairs
    {
        get
        {
            Tables t = _tables;
            TKey[] keys = t._keys;
            TValue[] values = t._values;
            int n = Volatile.Read(ref t._count);
            for (int i = 0; i < n; i++)
            {
                yield return new KeyValuePair<TKey, TValue>(keys[i], values[i]);
            }
        }
    }

    /// <summary>
    ///     Captures the current contents as a <see cref="ValuesView" /> — a zero-allocation snapshot whose indexer
    ///     and spans read the contiguous arrays with no per-access volatile read (the <see cref="List{T}" />-parity
    ///     path for hot index loops).
    /// </summary>
    /// <returns>A point-in-time view of the values (and keys) in insertion order.</returns>
    /// <remarks>The view is immutable in shape but not in value: entries appended, removed or reordered after the capture do not appear, while an in-place value replacement of an atomically-writable <typeparamref name="TValue" /> <i>is</i> visible through it. It pins the captured arrays while referenced.</remarks>
    public ValuesView Snapshot()
    {
        Tables t = _tables;
        return new ValuesView(t._keys, t._values, Volatile.Read(ref t._count));
    }

    /// <summary>A point-in-time, index-addressable view over the values (and keys) captured by <see cref="Snapshot" />: an array access plus a bounds check per read, and spans for vectorized scans.</summary>
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

        /// <summary>Gets the value at position <paramref name="index" /> of the captured snapshot — a plain array read.</summary>
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

    /// <summary>A forward, value-order enumerator bound to the contiguous value array and count captured when it was created; it observes a stable count and never throws for concurrent modification.</summary>
    public struct Enumerator : IEnumerator<TValue>
    {
        private readonly TValue[] _values;
        private readonly int _count;
        private int _i;
        private TValue _current;

        /// <summary>Binds the enumerator to a snapshot of the owner's current generation.</summary>
        /// <param name="owner">The collection to snapshot and enumerate.</param>
        internal Enumerator(ConcurrentOrderedCompactDictionary<TKey, TValue> owner)
        {
            Tables t = owner._tables;
            _values = t._values;
            // Acquire-read the count AFTER capturing the array: slots below it were release-published.
            _count = Volatile.Read(ref t._count);
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
    ///     absent — the core every add-shaped operation funnels through. One probe answers "already exists?" and
    ///     yields the insert position; in capacity the append allocates nothing.
    /// </summary>
    /// <param name="key">The key to append (already null-checked by the caller).</param>
    /// <param name="value">The value to store.</param>
    /// <param name="tag">The key's tag.</param>
    /// <returns><see langword="true" /> if appended; <see langword="false" /> if the key already existed (nothing changed).</returns>
    /// <remarks>Publication order: slot fields (invisible: no word points at the slot and it sits at or above the count), then the index word (release — the key path can reach it), then the count (release — the list path can). A fresh generation (growth, floor, load) is allocated and filled BEFORE anything published changes, so an allocation failure leaves both paths untouched.</remarks>
    private bool TryAppendUnderLock(TKey key, TValue value, int tag)
    {
        Tables t = _tables;
        if (ProbeUnderLock(t, key, tag, out int pos) >= 0)
        {
            return false;
        }

        int idx = t._count; // plain read: every count writer holds the lock we hold
        Tables target = t;
        bool fresh = false;
        if (idx == t.Capacity || idx < t._floor || OverLoad(idx + 1, t._dummies, t._index.Length))
        {
            int newCap = idx == t.Capacity
                ? (t.Capacity == 0 ? DefaultCapacity : (int)Math.Min((long)t.Capacity * 2, Array.MaxLength))
                : t.Capacity;
            target = CopyGeneration(t, idx, newCap);
            pos = FirstEmpty(target, tag);
            fresh = true;
        }

        target._keys[idx] = key;
        target._values[idx] = value;
        long word = Word(tag, idx + 1);
        if (fresh)
        {
            target._index[pos] = word;
            target._count = idx + 1;
            _tables = target; // volatile publish of the whole private generation
        }
        else
        {
            Volatile.Write(ref target._index[pos], word);
            Volatile.Write(ref target._count, idx + 1);
        }

        return true;
    }

    /// <summary>
    ///     Replaces the value of a live slot: one in-place store for atomically-writable values (readers see the old
    ///     or the new value, never a torn one), otherwise a whole fresh generation so no reader can tear a wide
    ///     value and no generation ever aliases another's arrays with a different values array.
    /// </summary>
    /// <param name="t">The current generation.</param>
    /// <param name="slot">The entry's position.</param>
    /// <param name="value">The replacement value.</param>
    private void ReplaceExistingUnderLock(Tables t, int slot, TValue value)
    {
        if (ConcurrentDictionaryTypeProps<TValue>.IsWriteAtomic)
        {
            t._values[slot] = value;
            return;
        }

        Tables copy = CopyGeneration(t, t._count, t.Capacity);
        copy._values[slot] = value;
        _tables = copy;
    }

    /// <summary>Removes the entry whose word sits at <paramref name="pos" /> and whose slot is <paramref name="slot" />: the O(1) tail path or the O(n) shifting copy.</summary>
    /// <param name="t">The current generation.</param>
    /// <param name="pos">The word position.</param>
    /// <param name="slot">The entry's position.</param>
    private void RemoveCoreUnderLock(Tables t, int pos, int slot)
    {
        if (slot == t._count - 1)
        {
            RemoveTailUnderLock(t, pos, (int)(t._index[pos] >> 32));
            return;
        }

        _tables = CopyGenerationWithout(t, slot);
    }

    /// <summary>
    ///     O(1) tail removal: dummies the entry's word (readers probe past it) and publishes a lower-count generation
    ///     over the same arrays with the floor raised, so an enumerator captured before the pop keeps reading valid
    ///     data in the vacated slot (deliberately not scrubbed) and a later append copies instead of reusing it.
    /// </summary>
    /// <param name="t">The current generation.</param>
    /// <param name="pos">The word position.</param>
    /// <param name="tag">The entry's tag (kept in the dummy word).</param>
    private void RemoveTailUnderLock(Tables t, int pos, int tag)
    {
        int n = t._count;
        // Successor allocated BEFORE the dummy store: once the key stops resolving, only throw-free publishes remain.
        Tables next = new(t._index, t._keys, t._values, n - 1, Math.Max(t._floor, n), t._dummies + 1);
        Volatile.Write(ref t._index[pos], Word(tag, DummySlot));
        _tables = next;
    }

    /// <summary>Throws <see cref="ArgumentNullException" /> if <paramref name="key" /> is null (guards the reference-key case).</summary>
    /// <param name="key">The key to validate.</param>
    /// <remarks>The <c>typeof</c> guard keeps the check allocation-free under Debug codegen: a bare <c>key is null</c> on a generic <typeparamref name="TKey" /> compiles to box+compare IL that unoptimized code executes for value types.</remarks>
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
    ///     <see cref="AddRange" /> source enumerator, a <see cref="RemoveWhere" /> predicate, or a key comparer
    ///     running inside a locked probe.
    /// </summary>
    /// <remarks><see cref="Monitor" /> is reentrant, so without this guard such a callback would not deadlock but interleave a second structural mutation inside a half-applied one. Factories handed to <see cref="GetOrAdd(TKey, Func{TKey, TValue})" /> and <see cref="AddOrUpdate" /> run outside the lock and may call back freely; lock-free reads are always legal.</remarks>
    /// <exception cref="LockRecursionException">The current thread already holds the write lock.</exception>
    private void ThrowIfReentrantWrite()
    {
        if (Monitor.IsEntered(_writeLock))
        {
            throw new LockRecursionException(
                "ConcurrentOrderedCompactDictionary does not support reentrant writes: a mutating operation was invoked from user code " +
                "running inside the collection's write lock (an AddRange source enumerator, a RemoveWhere predicate, or a " +
                "key comparer). Perform the mutation after the enclosing operation returns.");
        }
    }
}
