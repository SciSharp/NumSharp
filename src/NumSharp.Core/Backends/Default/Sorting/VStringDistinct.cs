using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged.Pooling;

// =============================================================================
// VStringDistinct.cs — the distinct values of NEP 55 StringDType text: np.unique's values-only route, and the text sets of
// np.isin
// =============================================================================
//
// WHAT NUMPY DOES (numpy/_core/src/multiarray/unique.cpp, unique_vstring; 2.4.2)
//   np.unique without a return flag (np.unique_values, the unique calls inside the set routines) hands the flattened
//   StringDType array to _unique_hash: every cell is loaded (NpyString_load, which gives an NA cell a NULL buffer WHATEVER
//   the descriptor's NA object), put into a std::unordered_set hashed with FNV-1a over the text and compared with memcmp,
//   the NULL-buffer cells hashing to one bucket and comparing equal to each other under equal_nan (to nothing but
//   themselves without it), never equal to a text. The set is packed into a fresh array of the input's descriptor (an NA
//   entry packed as an NA) and, with sorted=True, sorted in place with the default kind — so a None-like NA among two or
//   more distinct values raises the sort's ValueError, a NaN-like NA goes last, and an NA read as a default string sits
//   among the texts equal to it (in the hash set's order, which no other library reproduces). The set keeps one entry per
//   BYTE content while the sort orders by NpyString_cmp (strncmp: a NUL both texts share ends the byte comparison), so
//   byte-different texts equal up to a shared NUL ('a\0c', 'a\0b') are two entries that the sort leaves in the hash set's
//   order too.
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   The same values: an open-addressing table (power-of-two capacity grown at half load, linear probing, the hash stored
//   beside each slot) collects the FIRST occurrence of every distinct text — unless the texts are mostly distinct (after
//   an eighth of the array, at most ProbeTexts texts, the repeats among them estimate the whole array's distinct count;
//   later the table itself is watched), where the pass gives up and the route sorts every text (a stable argsort of the
//   array itself when it holds no NA) and drops the duplicates of each byte content within each run of NpyString_cmp-equal
//   texts. A text of at most 15 bytes in its cell hashes from its two cell words, masked to its size, in a few
//   instructions; every other text from its bytes — the same function, so equal texts hash alike wherever they live. The
//   NA cells are kept apart (one entry under equal_nan, each otherwise), and the entries are ordered as NumPy's sort orders
//   them: texts by NpyString_cmp, the NaN-like NAs last, NAs read as a default string right after the texts equal to that
//   string (NumPy's place among equal values — those, and byte-different texts equal up to a shared NUL — is its hash
//   set's order; NumSharp keeps first occurrences first: the same values either way).
// =============================================================================

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    ///     The distinct values of <c>StringDType</c> text (NumPy's <c>unique_vstring</c>), for <c>np.unique</c>'s values-only
    ///     route, and a set of texts for <c>np.isin</c>: see the file header.
    /// </summary>
    internal static unsafe class VStringDistinct
    {
        /// <summary>The fewest elements the hash pass is tried on: below it a sort of every text costs less than the table.</summary>
        internal const int MinElements = 64;

        /// <summary>
        ///     The most texts the hash pass reads before it judges the data (an eighth of the array, at least 256): from the
        ///     repeats among them it estimates how many distinct texts the whole array holds (<see cref="EstimateDistinct"/>),
        ///     and gives up for the sort route when that is more than half the array — the table would then hold most of the
        ///     texts, and sorting its entries would cost almost what sorting every text costs.
        /// </summary>
        internal const int ProbeTexts = 4096;

        /// <summary>
        ///     The estimated number of distinct texts in <paramref name="total"/> texts, from <paramref name="distinct"/>
        ///     distinct among the first <paramref name="sampled"/>: the texts read as draws from <c>D</c> equally likely values,
        ///     whose birthday estimate from the repeats is <c>D ≈ sampled² / (2 · repeats)</c>; the whole array then holds
        ///     <c>D · (1 − e^(−total / D))</c> of them. No repeat at all estimates every text distinct.
        /// </summary>
        /// <param name="sampled">The texts read.</param>
        /// <param name="distinct">The distinct texts among them.</param>
        /// <param name="total">The texts of the whole array.</param>
        /// <returns>The estimated distinct count of the whole array (at most <paramref name="total"/>).</returns>
        /// <remarks>A speed decision only: a wrong estimate costs time, never a wrong answer (both routes give the same
        ///     values). Data far from uniform (a sorted array, long runs) can mislead it; the later checks of the pass catch a
        ///     table that grows past half the array anyway.</remarks>
        internal static double EstimateDistinct(long sampled, long distinct, long total)
        {
            long repeats = sampled - distinct;
            if (repeats <= 0)
                return total;
            double d = (double)sampled * sampled / (2.0 * repeats);
            return Math.Min(total, d * (1.0 - Math.Exp(-total / d)));
        }

        /// <summary>Test hook: when set on a thread, the hash pass declines every call there (the sort route runs).</summary>
        [ThreadStatic] internal static bool Disabled;

        /// <summary>Test hook: the hash passes that completed since the process started.</summary>
        internal static long Runs;

        /// <summary>Test hook: the hash passes that gave up because too many texts were distinct.</summary>
        internal static long Bailouts;

        /// <summary>
        ///     The element indices of NumPy's <c>unique_vstring</c> result over a 1-D <c>StringDType</c> array, in NumPy's
        ///     sorted order (<paramref name="sorted"/>) — the caller gathers them (<c>np.take</c>) into the result.
        /// </summary>
        /// <param name="ar">The flattened array (1-D, only read): <c>np.unique</c>'s <c>ar.flatten()</c>.</param>
        /// <param name="equalNan">NumPy's <c>equal_nan</c>: the NA cells form one entry (the first one) when true, one entry
        ///     each when false.</param>
        /// <param name="sorted">NumPy's <c>sorted</c>: true runs the default-kind sort of the entries (its None-like NA
        ///     refusal included). False is NumPy's hash order, which NumSharp does not reproduce: the entries come back in the
        ///     sorted order, the NAs last, and nothing raises.</param>
        /// <returns>The indices of the distinct values in <paramref name="ar"/>, in their output order.</returns>
        /// <exception cref="ValueError">A None-like NA among two or more entries with <paramref name="sorted"/> (NumPy's
        ///     sort refusal <c>Cannot compare null that is not a nan-like value</c>).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt cell, or the scratch cannot be allocated.</exception>
        /// <remarks>Not scoped (it returns indices, not arrays): the one temporary array chain, inside
        ///     <see cref="OrderTexts"/>, is disposed there.</remarks>
        internal static long[] DistinctOrder(NDArray ar, bool equalNan, bool sorted)
        {
            long n = ar.size;
            if (n == 0)
                return Array.Empty<long>();
            var rule = NDVStringCells.RuleOf(ar.dtype);
            // One pass over the cells: the NA indices (input order) and the distinct texts' first occurrences (or every text
            // index when the table gives up).
            List<long> nas = new List<long>();
            long[] texts;
            bool hashed;
            var allocator = ar.Storage.VStringSlice.Allocator;
            allocator.acquire_allocator();
            try
            {
                const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
                byte* cells = (byte*)ar.Address + ar.Shape.offset * (long)cell;
                long stride = ar.Shape.strides[0] * (long)cell;
                texts = CollectTexts(cells, stride, n, allocator.ArenaBase, nas, out hashed);
            }
            finally
            {
                allocator.release_allocator();
            }

            // The NA entries: one (the first NA) under equal_nan, each otherwise.
            long[] naEntries = nas.Count == 0 ? Array.Empty<long>() : equalNan ? new[] { nas[0] } : nas.ToArray();

            // NumPy sorts the packed entries (sorted=True): a None-like NA among two or more entries is compared — and
            // refused — whatever the texts are. Any text makes one entry at least, so the NA entries plus one per text
            // present decide it without knowing how many texts are distinct.
            if (sorted && rule == VStringNARule.Null && naEntries.Length > 0 && (texts.Length > 0 ? 1 : 0) + naEntries.Length >= 2)
                throw VStringSort.NullCompareError();

            // The texts in byte order (the stable order of a subset: the gathered subset's stable argsort), then
            // deduplicated when the table gave up.
            long[] orderedTexts = OrderTexts(ar, texts, dedupe: !hashed);

            // Place the NA entries as NumPy's sort places them.
            if (naEntries.Length == 0)
                return orderedTexts;
            var result = new long[orderedTexts.Length + naEntries.Length];
            if (rule is VStringNARule.None or VStringNARule.String)
            {
                // Read as the default string: right after the texts not above it (equal values keep the texts first).
                long at = UpperBound(ar, orderedTexts, NDUtf8.Encode(ar.dtype.VString.default_string));
                Array.Copy(orderedTexts, 0, result, 0, at);
                Array.Copy(naEntries, 0, result, at, naEntries.Length);
                Array.Copy(orderedTexts, at, result, at + naEntries.Length, orderedTexts.Length - at);
                return result;
            }
            // A NaN-like NA is the greatest value; a None-like one reaches here only unsorted or alone: last either way.
            Array.Copy(orderedTexts, result, orderedTexts.Length);
            Array.Copy(naEntries, 0, result, orderedTexts.Length, naEntries.Length);
            return result;
        }

        /// <summary>
        ///     Loads every cell: the NA cells' indices go to <paramref name="nas"/>, and the texts are collected as the first
        ///     occurrence of each distinct text (hash table) — or, when the texts are mostly distinct (the estimate after the
        ///     first window, <see cref="EstimateDistinct"/>, or the table itself at a later check, above half the array) or the
        ///     array is small, every text index for the sort route.
        /// </summary>
        /// <param name="cells">Element 0's cell.</param>
        /// <param name="stride">The byte distance between cells.</param>
        /// <param name="n">The element count.</param>
        /// <param name="arena">The arena base (read under the allocator lock).</param>
        /// <param name="nas">Receives the NA indices in input order (cleared and refilled when the pass gives up).</param>
        /// <param name="hashed">Receives true when the returned texts are distinct (the table completed).</param>
        /// <returns>The text indices: distinct first occurrences (hashed) or every text, in input order.</returns>
        /// <exception cref="OutOfMemoryException">A corrupt cell, or the table cannot be allocated.</exception>
        /// <remarks>The pass that gives up reads the cells a second time for the sort route rather than recording every
        ///     index while it hashes: the record cost every hashed element a list append, the rescan costs one cell load per
        ///     element only when the table loses.</remarks>
        private static long[] CollectTexts(byte* cells, long stride, long n, byte* arena, List<long> nas, out bool hashed)
        {
            if (!Disabled && n >= MinElements && n <= int.MaxValue)
            {
                var table = new TextTable();
                try
                {
                    // The first check after an eighth of the array (256 to ProbeTexts texts) estimates the whole array's
                    // distinct count; the later ones, every window, give up only once the table itself passes half the array.
                    long window = Math.Clamp(n >> 3, 256, ProbeTexts);
                    long texts = 0, nextCheck = window;
                    bool first = true, gaveUp = false;
                    for (long i = 0; i < n; i++)
                    {
                        byte* c = cells + i * stride;
                        int r = NDVStringCells.Load(c, arena, out byte* buf, out long size);
                        if (r < 0)
                            throw VStringSort.CorruptCellError();
                        if (r == 1)
                        {
                            nas.Add(i);
                            continue;
                        }
                        // A text in its cell (buf is the cell itself) hashes from its two masked words; any other from its
                        // bytes — the same function either way (HashText).
                        table.AddIfAbsent(buf, size, buf == c ? HashCellText(c, size) : Hash(buf, size), i);
                        if (++texts == nextCheck)
                        {
                            // Mostly distinct texts: the table would hold most of the array, and sorting its entries would
                            // cost about what sorting every text costs — give the work to the sort route instead.
                            bool mostlyDistinct = first
                                ? 2 * EstimateDistinct(texts, table.Count, n) > n
                                : 2 * table.Count > n;
                            if (mostlyDistinct)
                            {
                                gaveUp = true;
                                break;
                            }
                            first = false;
                            nextCheck += window;
                        }
                    }
                    if (!gaveUp)
                    {
                        Runs++;
                        hashed = true;
                        return table.FirstOccurrences();
                    }
                }
                finally
                {
                    table.Dispose();
                }
                // The sort route below collects every index again: the NA list restarts with it.
                Bailouts++;
                nas.Clear();
            }

            // The sort route: every text's index (input order), the NA indices apart.
            var all = new long[n];
            long count = 0;
            for (long i = 0; i < n; i++)
            {
                int r = NDVStringCells.Load(cells + i * stride, arena, out _, out _);
                if (r < 0)
                    throw VStringSort.CorruptCellError();
                if (r == 1)
                    nas.Add(i);
                else
                    all[count++] = i;
            }
            if (count < n)
                Array.Resize(ref all, (int)count);
            hashed = false;
            return all;
        }

        /// <summary>
        ///     The given text indices of <paramref name="ar"/> in byte order (stable: equal texts keep their input order), with
        ///     adjacent duplicates dropped when <paramref name="dedupe"/> is set.
        /// </summary>
        /// <param name="ar">The 1-D array.</param>
        /// <param name="texts">Text element indices (no NA cell), in input order.</param>
        /// <param name="dedupe">Whether equal texts may repeat (the table gave up) and only the first of each is kept.</param>
        /// <returns>The ordered indices.</returns>
        private static long[] OrderTexts(NDArray ar, long[] texts, bool dedupe)
        {
            if (texts.Length <= 1)
                return texts;
            int m = texts.Length;
            // Every element a text (no NA, the common case of the sort route): the stable argsort of the array itself is the
            // order — no gathered copy of the strings, and its indices are already the array's. Otherwise the texts are
            // gathered (the NAs left out: a None-like NA must not be compared) and sorted.
            bool whole = m == ar.size;
            using var picks = whole ? null : np.array(texts);
            using var subset = whole ? null : UnmanagedStorage.TakeFrom(ar, picks, null, null, np.ClipModeRaise);
            NDArray sorted = subset ?? ar;
            using var order = AxisSort.ArgSortVString(sorted, 0, stable: true);
            // The argsort's result is a fresh C-contiguous int64 array: read it through its pointer, not per-element calls.
            long* ord = (long*)order.Address + order.Shape.offset;
            var ordered = new long[m];
            if (whole)
                new ReadOnlySpan<long>(ord, m).CopyTo(ordered);
            else
                for (int k = 0; k < m; k++)
                    ordered[k] = texts[ord[k]];
            if (!dedupe)
                return ordered;
            // Drop duplicates. NumPy's hash route keeps one entry per BYTE content (memcmp), while the sorted order groups
            // texts by NpyString_cmp, under which byte-different texts equal up to a NUL they share are equal: so the texts
            // of one byte content are adjacent only within a run of NpyString_cmp-equal texts, not always next to each
            // other. A text with no NUL is NpyString_cmp-equal to its own bytes only, so its copies ARE adjacent: the last
            // kept text decides (the common case, one compare). A text with a NUL may share its run with byte-different
            // texts, so each such run keeps its byte contents and drops a later copy of any of them (the stable order lists
            // first occurrences first).
            var kept = new List<long>(m);
            List<IntPtr> runBufs = null;
            List<long> runSizes = null;
            byte* lastBuf = null;
            long lastSize = -1;
            var allocator = sorted.Storage.VStringSlice.Allocator;
            allocator.acquire_allocator();
            try
            {
                const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
                byte* cells = (byte*)sorted.Address + sorted.Shape.offset * (long)cell;
                long stride = sorted.Shape.strides[0] * (long)cell;
                byte* arena = allocator.ArenaBase;
                for (int k = 0; k < m; k++)
                {
                    // ord[k] indexes the sorted operand (the array itself, or the subset whose cells hold the same texts as
                    // ar's at texts[ord[k]]).
                    byte* c = cells + ord[k] * stride;
                    NDVStringCells.Load(c, arena, out byte* buf, out long size);
                    // The last kept text again (copies of one byte content are adjacent within their run).
                    if (size == lastSize && SameBytes(buf, lastBuf, size))
                        continue;
                    bool hasNul = NDVStringCells.NulLengthInPlace(c, buf, size) < size;
                    if (!hasNul)
                    {
                        // A text without a NUL starts a run of its own bytes.
                        runBufs?.Clear();
                        runSizes?.Clear();
                    }
                    else if (runBufs is { Count: > 0 } && NDVStringCells.Order(buf, size, (byte*)runBufs[0], runSizes[0]) == 0)
                    {
                        // The current run of NUL texts: a byte content it already kept is a duplicate.
                        bool duplicate = false;
                        for (int r = 0; r < runBufs.Count && !duplicate; r++)
                            duplicate = runSizes[r] == size && SameBytes(buf, (byte*)runBufs[r], size);
                        if (duplicate)
                            continue;
                    }
                    else
                    {
                        // A NUL text that starts a new run.
                        (runBufs ??= new List<IntPtr>()).Clear();
                        (runSizes ??= new List<long>()).Clear();
                    }
                    kept.Add(ordered[k]);
                    lastBuf = buf;
                    lastSize = size;
                    if (hasNul)
                    {
                        runBufs.Add((IntPtr)buf);
                        runSizes.Add(size);
                    }
                }
            }
            finally
            {
                allocator.release_allocator();
            }
            return kept.ToArray();
        }

        /// <summary>Whether two texts of the same size hold the same bytes (a vectorized compare, any size).</summary>
        /// <param name="a">The first text.</param>
        /// <param name="b">The second text.</param>
        /// <param name="size">Their common size.</param>
        /// <returns>True when equal.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static bool SameBytes(byte* a, byte* b, long size)
        {
            // Span lengths are int: a text past 2 GiB (never built in practice) is compared in int-sized pieces.
            while (size > int.MaxValue)
            {
                if (!new ReadOnlySpan<byte>(a, int.MaxValue).SequenceEqual(new ReadOnlySpan<byte>(b, int.MaxValue)))
                    return false;
                a += int.MaxValue;
                b += int.MaxValue;
                size -= int.MaxValue;
            }
            return new ReadOnlySpan<byte>(a, (int)size).SequenceEqual(new ReadOnlySpan<byte>(b, (int)size));
        }

        /// <summary>
        ///     The number of leading entries of <paramref name="ordered"/> (texts of <paramref name="ar"/> in byte order) not
        ///     above <paramref name="text"/>: where a value equal to <paramref name="text"/> goes after its equals.
        /// </summary>
        /// <param name="ar">The 1-D array.</param>
        /// <param name="ordered">Text indices in byte order.</param>
        /// <param name="text">The UTF-8 text.</param>
        /// <returns>The insertion position.</returns>
        private static long UpperBound(NDArray ar, long[] ordered, byte[] text)
        {
            var allocator = ar.Storage.VStringSlice.Allocator;
            allocator.acquire_allocator();
            try
            {
                const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
                byte* cells = (byte*)ar.Address + ar.Shape.offset * (long)cell;
                long stride = ar.Shape.strides[0] * (long)cell;
                byte* arena = allocator.ArenaBase;
                fixed (byte* t = text)
                {
                    byte* tp = text.Length == 0 ? (byte*)1 : t; // a non-null pointer for an empty text (never read)
                    long lo = 0, hi = ordered.Length;
                    while (lo < hi)
                    {
                        long mid = lo + ((hi - lo) >> 1);
                        NDVStringCells.Load(cells + ordered[mid] * stride, arena, out byte* buf, out long size);
                        if (NDVStringCells.Order(buf, size, tp, text.Length) <= 0)
                            lo = mid + 1;
                        else
                            hi = mid;
                    }
                    return lo;
                }
            }
            finally
            {
                allocator.release_allocator();
            }
        }

        /// <summary>
        ///     A 64-bit hash of a text (multiply-xorshift over 8-byte words, a splitmix finalizer): equal texts hash alike; it
        ///     only has to spread the table's probes.
        /// </summary>
        /// <param name="buf">The text.</param>
        /// <param name="size">Its size.</param>
        /// <returns>The hash.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static ulong Hash(byte* buf, long size)
        {
            ulong h = 0x9E3779B97F4A7C15UL ^ (ulong)size;
            long i = 0;
            for (; i + 8 <= size; i += 8)
            {
                h ^= Unsafe.ReadUnaligned<ulong>(buf + i);
                h *= 0xBF58476D1CE4E5B9UL;
                h ^= h >> 31;
            }
            if (i < size)
            {
                ulong tail = 0;
                for (int b = 0; i < size; i++, b += 8)
                    tail |= (ulong)buf[i] << b;
                h ^= tail;
                h *= 0x94D049BB133111EBUL;
                h ^= h >> 29;
            }
            h ^= h >> 30;
            h *= 0xBF58476D1CE4E5B9UL;
            h ^= h >> 27;
            h *= 0x94D049BB133111EBUL;
            h ^= h >> 31;
            return h;
        }

        /// <summary>
        ///     The hash of a text's <c>NpyString_cmp</c> equality class: <see cref="Hash"/>'s mixing over the comparison prefix
        ///     (<see cref="NDVStringCells.NulLength"/>: the bytes up to and including the first NUL) seeded with the full size,
        ///     so texts equal under <c>NpyString_cmp</c> (the same size, the same prefix, any bytes after the NUL) hash alike.
        /// </summary>
        /// <param name="buf">The text.</param>
        /// <param name="size">Its size.</param>
        /// <returns>The hash.</returns>
        internal static ulong ClassHash(byte* buf, long size)
        {
            long len = NDVStringCells.NulLength(buf, size);
            ulong h = 0x9E3779B97F4A7C15UL ^ (ulong)size;
            long i = 0;
            for (; i + 8 <= len; i += 8)
            {
                h ^= Unsafe.ReadUnaligned<ulong>(buf + i);
                h *= 0xBF58476D1CE4E5B9UL;
                h ^= h >> 31;
            }
            if (i < len)
            {
                ulong tail = 0;
                for (int b = 0; i < len; i++, b += 8)
                    tail |= (ulong)buf[i] << b;
                h ^= tail;
                h *= 0x94D049BB133111EBUL;
                h ^= h >> 29;
            }
            h ^= h >> 30;
            h *= 0xBF58476D1CE4E5B9UL;
            h ^= h >> 27;
            h *= 0x94D049BB133111EBUL;
            h ^= h >> 31;
            return h;
        }

        /// <summary>
        ///     <see cref="Hash"/> of a text that lives in its own cell (at most 15 bytes, the cell's first bytes), computed from
        ///     the cell's two 8-byte words masked to the text's size instead of byte by byte: the same value as
        ///     <c>Hash(cell, size)</c>, so an equal text stored elsewhere (an arena allocation a shorter write reused, a heap
        ///     string) hashes alike.
        /// </summary>
        /// <param name="cell">The cell (16 bytes, all readable; byte 15 holds the size and flags).</param>
        /// <param name="size">The text's size (0..15).</param>
        /// <returns>The hash.</returns>
        /// <remarks>The masks drop the bytes past the text, which a cell may hold stale from an earlier, longer text, and
        ///     byte 15 (the size and flags): <see cref="Hash"/> reads only the text's bytes.</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static ulong HashCellText(byte* cell, long size)
        {
            ulong h = 0x9E3779B97F4A7C15UL ^ (ulong)size;
            if (size >= 8)
            {
                // Hash's whole-word loop runs once, on bytes 0..7; its tail (bytes 8..size-1, zero-padded) is the high
                // word masked to size - 8 bytes (none at size 8).
                h ^= Unsafe.ReadUnaligned<ulong>(cell);
                h *= 0xBF58476D1CE4E5B9UL;
                h ^= h >> 31;
                if (size > 8)
                {
                    h ^= Unsafe.ReadUnaligned<ulong>(cell + 8) & ((1UL << (int)((size - 8) * 8)) - 1);
                    h *= 0x94D049BB133111EBUL;
                    h ^= h >> 29;
                }
            }
            else if (size > 0)
            {
                // No whole word: Hash's tail is the low word masked to the text's bytes.
                h ^= Unsafe.ReadUnaligned<ulong>(cell) & ((1UL << (int)(size * 8)) - 1);
                h *= 0x94D049BB133111EBUL;
                h ^= h >> 29;
            }
            h ^= h >> 30;
            h *= 0xBF58476D1CE4E5B9UL;
            h ^= h >> 27;
            h *= 0x94D049BB133111EBUL;
            h ^= h >> 31;
            return h;
        }

        /// <summary>
        ///     The open-addressing table of <see cref="CollectTexts"/>: power-of-two capacity, grown (doubled and rehashed from
        ///     the stored hashes) when it would pass half load, linear probing, each slot holding an element index and its
        ///     text's pointer, size and hash. The texts stay where the cells point (the caller holds the allocator lock for the
        ///     table's life), so a slot never copies a text.
        /// </summary>
        private sealed class TextTable : IDisposable
        {
            /// <summary>The capacity a new table starts at (grows by doubling).</summary>
            private const long InitialCapacity = 64;

            /// <summary>The slots' element indices (-1 when empty).</summary>
            private long* _index;

            /// <summary>The slots' text pointers.</summary>
            private byte** _buf;

            /// <summary>The slots' text sizes.</summary>
            private long* _size;

            /// <summary>The slots' hashes.</summary>
            private ulong* _hash;

            /// <summary>The capacity (a power of two) and its mask.</summary>
            private long _capacity, _mask;

            /// <summary>The entries' element indices in insertion order (first occurrences).</summary>
            private readonly List<long> _order = new List<long>();

            /// <summary>The number of distinct texts recorded.</summary>
            internal long Count => _order.Count;

            /// <summary>Allocates an empty table of <see cref="InitialCapacity"/> slots.</summary>
            /// <exception cref="OutOfMemoryException">The slots cannot be allocated.</exception>
            internal TextTable()
            {
                Allocate(InitialCapacity);
            }

            /// <summary>Takes the four slot arrays of <paramref name="capacity"/> slots from the pool, every slot empty.</summary>
            /// <param name="capacity">The capacity (a power of two).</param>
            /// <exception cref="OutOfMemoryException">The pool cannot allocate (whatever was taken is returned).</exception>
            private void Allocate(long capacity)
            {
                _capacity = capacity;
                _mask = capacity - 1;
                try
                {
                    _index = (long*)SizeBucketedBufferPool.Take(capacity * sizeof(long));
                    _buf = (byte**)SizeBucketedBufferPool.Take(capacity * sizeof(long));
                    _size = (long*)SizeBucketedBufferPool.Take(capacity * sizeof(long));
                    _hash = (ulong*)SizeBucketedBufferPool.Take(capacity * sizeof(ulong));
                    new Span<long>(_index, (int)capacity).Fill(-1);
                }
                catch
                {
                    Dispose();
                    throw;
                }
            }

            /// <summary>
            ///     Records the text's first occurrence unless an equal text is already recorded (equal hash, size and bytes).
            /// </summary>
            /// <param name="buf">The text.</param>
            /// <param name="size">Its size.</param>
            /// <param name="h">Its hash (<see cref="Hash"/>, or <see cref="HashCellText"/> for a text in its cell: the same
            ///     value).</param>
            /// <param name="index">Its element index.</param>
            /// <exception cref="OutOfMemoryException">The table cannot grow.</exception>
            [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
            internal void AddIfAbsent(byte* buf, long size, ulong h, long index)
            {
                long slot = (long)(h & (ulong)_mask);
                while (_index[slot] >= 0)
                {
                    if (_hash[slot] == h && _size[slot] == size && SameBytes(_buf[slot], buf, size))
                        return;
                    slot = (slot + 1) & _mask;
                }
                if (2 * (_order.Count + 1) > _capacity)
                {
                    // Past half load the probes lengthen: double and place the entry in the new table.
                    Grow();
                    slot = (long)(h & (ulong)_mask);
                    while (_index[slot] >= 0)
                        slot = (slot + 1) & _mask;
                }
                _index[slot] = index;
                _buf[slot] = buf;
                _size[slot] = size;
                _hash[slot] = h;
                _order.Add(index);
            }

            /// <summary>Doubles the capacity, re-placing every entry by its stored hash (no text is hashed again).</summary>
            /// <exception cref="OutOfMemoryException">The larger slot arrays cannot be allocated (the table is left intact).</exception>
            private void Grow()
            {
                long* oldIndex = _index;
                byte** oldBuf = _buf;
                long* oldSize = _size;
                ulong* oldHash = _hash;
                long oldCapacity = _capacity;
                _index = null;
                _buf = null;
                _size = null;
                _hash = null;
                try
                {
                    Allocate(oldCapacity * 2);
                }
                catch
                {
                    // Allocate returned what it took: put the old arrays back so the table stays usable and disposable.
                    _index = oldIndex;
                    _buf = oldBuf;
                    _size = oldSize;
                    _hash = oldHash;
                    _capacity = oldCapacity;
                    _mask = oldCapacity - 1;
                    throw;
                }
                for (long s = 0; s < oldCapacity; s++)
                {
                    if (oldIndex[s] < 0)
                        continue;
                    long slot = (long)(oldHash[s] & (ulong)_mask);
                    while (_index[slot] >= 0)
                        slot = (slot + 1) & _mask;
                    _index[slot] = oldIndex[s];
                    _buf[slot] = oldBuf[s];
                    _size[slot] = oldSize[s];
                    _hash[slot] = oldHash[s];
                }
                long oldBytes = oldCapacity * sizeof(long);
                SizeBucketedBufferPool.Return((IntPtr)oldIndex, oldBytes);
                SizeBucketedBufferPool.Return((IntPtr)oldBuf, oldBytes);
                SizeBucketedBufferPool.Return((IntPtr)oldSize, oldBytes);
                SizeBucketedBufferPool.Return((IntPtr)oldHash, oldBytes);
            }

            /// <summary>The recorded first occurrences, in input order.</summary>
            /// <returns>The element indices.</returns>
            internal long[] FirstOccurrences() => _order.ToArray();

            /// <summary>Returns the slots (idempotent).</summary>
            public void Dispose()
            {
                long bytes = _capacity * sizeof(long);
                if (_index != null) { SizeBucketedBufferPool.Return((IntPtr)_index, bytes); _index = null; }
                if (_buf != null) { SizeBucketedBufferPool.Return((IntPtr)_buf, bytes); _buf = null; }
                if (_size != null) { SizeBucketedBufferPool.Return((IntPtr)_size, bytes); _size = null; }
                if (_hash != null) { SizeBucketedBufferPool.Return((IntPtr)_hash, bytes); _hash = null; }
            }
        }

        /// <summary>
        ///     A set of UTF-8 texts with owned copies (<c>np.isin</c>'s strings of the test set): built from any source, then
        ///     probed with texts read in place. Membership is NumPy's <c>==</c> of StringDType text, <c>NpyString_cmp</c>
        ///     equality: the same size and the same bytes up to and including the first NUL (<c>'a\0c'</c> finds
        ///     <c>'a\0b'</c>) — so the texts hash by that comparison prefix and their size (<see cref="ClassHash"/>), not by
        ///     all their bytes.
        /// </summary>
        internal sealed class ByteTextSet : IDisposable
        {
            /// <summary>The slots' offsets into <see cref="_bytes"/> (-1 when empty).</summary>
            private long[] _offset;

            /// <summary>The slots' sizes.</summary>
            private long[] _size;

            /// <summary>The slots' hashes.</summary>
            private ulong[] _hash;

            /// <summary>The texts' bytes, end to end (pooled; grown by doubling).</summary>
            private byte* _bytes;

            /// <summary>The used and allocated byte counts of <see cref="_bytes"/>.</summary>
            private long _used, _cap;

            /// <summary>The number of distinct texts.</summary>
            internal long Count { get; private set; }

            /// <summary>Creates an empty set.</summary>
            internal ByteTextSet()
            {
                _offset = new long[16];
                _size = new long[16];
                _hash = new ulong[16];
                Array.Fill(_offset, -1L);
            }

            /// <summary>Adds a text (copied) unless an equal one is present.</summary>
            /// <param name="buf">The text.</param>
            /// <param name="size">Its size.</param>
            /// <exception cref="OutOfMemoryException">The copy cannot be allocated.</exception>
            internal void Add(byte* buf, long size)
            {
                ulong h = ClassHash(buf, size);
                if (Find(buf, size, h) >= 0)
                    return;
                if (2 * (Count + 1) > _offset.Length)
                    Grow();
                if (_used + size > _cap)
                {
                    long cap = Math.Max(Math.Max(64, 2 * _cap), _used + size);
                    var bytes = (byte*)SizeBucketedBufferPool.Take(cap);
                    if (_bytes != null)
                    {
                        Buffer.MemoryCopy(_bytes, bytes, cap, _used);
                        SizeBucketedBufferPool.Return((IntPtr)_bytes, _cap);
                    }
                    _bytes = bytes;
                    _cap = cap;
                }
                if (size > 0)
                    Buffer.MemoryCopy(buf, _bytes + _used, size, size);
                long mask = _offset.Length - 1;
                long slot = (long)(h & (ulong)mask);
                while (_offset[slot] >= 0)
                    slot = (slot + 1) & mask;
                _offset[slot] = _used;
                _size[slot] = size;
                _hash[slot] = h;
                _used += size;
                Count++;
            }

            /// <summary>Whether an equal text is in the set.</summary>
            /// <param name="buf">The text.</param>
            /// <param name="size">Its size.</param>
            /// <returns>True when present.</returns>
            internal bool Contains(byte* buf, long size) => Find(buf, size, ClassHash(buf, size)) >= 0;

            /// <summary>The slot holding an equal text, or -1.</summary>
            /// <param name="buf">The text.</param>
            /// <param name="size">Its size.</param>
            /// <param name="h">Its hash.</param>
            /// <returns>The slot, or -1.</returns>
            private long Find(byte* buf, long size, ulong h)
            {
                long mask = _offset.Length - 1;
                long slot = (long)(h & (ulong)mask);
                while (_offset[slot] >= 0)
                {
                    if (_hash[slot] == h && _size[slot] == size
                        && (size == 0 || NDVStringCells.Order(_bytes + _offset[slot], size, buf, size) == 0))
                        return slot;
                    slot = (slot + 1) & mask;
                }
                return -1;
            }

            /// <summary>Doubles the slot arrays and re-inserts every entry.</summary>
            private void Grow()
            {
                var oldOffset = _offset;
                var oldSize = _size;
                var oldHash = _hash;
                int cap = oldOffset.Length * 2;
                _offset = new long[cap];
                _size = new long[cap];
                _hash = new ulong[cap];
                Array.Fill(_offset, -1L);
                long mask = cap - 1;
                for (int i = 0; i < oldOffset.Length; i++)
                {
                    if (oldOffset[i] < 0)
                        continue;
                    long slot = (long)(oldHash[i] & (ulong)mask);
                    while (_offset[slot] >= 0)
                        slot = (slot + 1) & mask;
                    _offset[slot] = oldOffset[i];
                    _size[slot] = oldSize[i];
                    _hash[slot] = oldHash[i];
                }
            }

            /// <summary>Returns the bytes (idempotent).</summary>
            public void Dispose()
            {
                if (_bytes != null)
                {
                    SizeBucketedBufferPool.Return((IntPtr)_bytes, _cap);
                    _bytes = null;
                }
            }
        }
    }
}
