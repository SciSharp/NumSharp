using System;
using System.Numerics;
using System.Threading;

// =============================================================================
// TextDistinct.cs — the distinct strings of fixed-width text (S / U) by hashing: np.unique's values-only route
// =============================================================================
//
// WHAT NUMPY DOES (numpy/_core/src/multiarray/unique.cpp, unique_string; 2.4.2)
//   np.unique without a return flag (and np.unique_values, and the unique calls inside the set routines) hands the
//   flattened text to _unique_hash: every element goes into a std::unordered_set keyed by FNV-1a over the whole item and
//   compared with memcmp, and the set — the distinct strings — is copied out and sorted (sorted=True). Hashing costs one
//   pass whatever the data; sorting then touches only the distinct strings, so text with few distinct values (categories,
//   codes, words) costs far less than a sort of every string.
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   The same two steps: an open-addressing table (power-of-two capacity, linear probing, the 32-bit hash stored beside each
//   slot's element index so a probe compares bytes only on a hash match) collects the FIRST occurrence of every distinct
//   string, the distinct strings are copied into a fresh array, and the caller sorts that array. Equal strings are equal
//   bytes at one width, so the sorted distinct strings are NumPy's values whatever hash or table either side uses.
//   Unlike NumPy, the pass gives up once more than max(256, n / 16) strings are distinct: the table would outgrow the
//   cache and the distinct strings would need a full sort anyway, so the caller sorts every string instead (the route that
//   is faster for mostly distinct text). Both routes return the same values.
// =============================================================================

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    ///     The distinct strings of fixed-width text by hashing (NumPy's <c>unique_string</c>), for <c>np.unique</c>'s
    ///     values-only route: see the file header.
    /// </summary>
    internal static unsafe class TextDistinct
    {
        /// <summary>
        ///     The fewest elements the hash pass is tried on: below it a sort of every string costs less than building the
        ///     table.
        /// </summary>
        internal const int MinElements = 64;

        /// <summary>The least number of distinct strings the pass accepts before it gives up (see <see cref="Limit"/>).</summary>
        internal const int MinLimit = 256;

        /// <summary>
        ///     Test hook: when set on a thread, <see cref="TryDistinct"/> declines every call there, so the caller's sort of
        ///     every string runs (the two routes are compared with it).
        /// </summary>
        [ThreadStatic] internal static bool Disabled;

        /// <summary>Test hook: the hash passes that completed (their distinct strings were returned) since the process started.</summary>
        internal static long Runs;

        /// <summary>Test hook: the hash passes that gave up because too many strings were distinct.</summary>
        internal static long Bailouts;

        /// <summary>
        ///     The most distinct strings the pass collects for <paramref name="n"/> elements: <c>max(256, n / 16)</c>. The
        ///     table then holds at most twice that many slots, so it stays in the cache while every element probes it.
        /// </summary>
        /// <param name="n">The element count.</param>
        /// <returns>The limit.</returns>
        internal static long Limit(long n) => Math.Max(MinLimit, n >> 4);

        /// <summary>
        ///     The distinct strings of a C-contiguous 1-D text array, in the order of their first occurrence, when at most
        ///     <see cref="Limit"/> of them are distinct; otherwise null, and the caller sorts every string.
        /// </summary>
        /// <param name="ar">A C-contiguous 1-D <c>S</c> or <c>U</c> array (only read): np.unique's flattened copy.</param>
        /// <returns>A fresh 1-D array of <paramref name="ar"/>'s dtype holding each distinct string once (unsorted), or null when
        ///     the array is too small (<see cref="MinElements"/>), its items have no width, it is not C-contiguous, the hook
        ///     <see cref="Disabled"/> is set, or more than <see cref="Limit"/> strings are distinct.</returns>
        internal static NDArray TryDistinct(NDArray ar)
        {
            long n = ar.size;
            int width = ar.dtype.itemsize;
            // The table stores element indices as int: a larger array takes the sort route (which handles any length).
            if (Disabled || n < MinElements || width == 0 || n > int.MaxValue || ar.ndim != 1 || !ar.Shape.IsContiguous)
                return null;

            int limit = (int)Math.Min(Limit(n), n);
            // Load at most one half: the probes of an absent string stay short.
            int capacity = (int)BitOperations.RoundUpToPowerOf2((uint)limit * 2);
            int mask = capacity - 1;
            // slots[s] = index + 1 of the slot's string (0: empty); hashes[s] = its hash, checked before the bytes.
            var slots = new int[capacity];
            var hashes = GC.AllocateUninitializedArray<uint>(capacity);
            var firsts = GC.AllocateUninitializedArray<int>(limit);
            int count = 0;
            // The array's logical start: a contiguous view may sit at an offset inside its buffer.
            byte* p = (byte*)ar.Address + ar.Shape.offset * (long)width;
            fixed (int* slot0 = slots)
            fixed (uint* hash0 = hashes)
            {
                for (long i = 0; i < n; i++)
                {
                    byte* e = p + i * width;
                    uint h = Hash(e, width);
                    int s = (int)h & mask;
                    while (true)
                    {
                        int occupant = slot0[s];
                        if (occupant == 0)
                        {
                            // A new distinct string: past the limit the table would grow beyond the cache and the distinct
                            // strings would need a full sort anyway, so the caller sorts every string instead.
                            if (count == limit)
                            {
                                Interlocked.Increment(ref Bailouts);
                                return null;
                            }
                            slot0[s] = (int)i + 1;
                            hash0[s] = h;
                            firsts[count++] = (int)i;
                            break;
                        }
                        if (hash0[s] == h && new ReadOnlySpan<byte>(e, width).SequenceEqual(
                                new ReadOnlySpan<byte>(p + (long)(occupant - 1) * width, width)))
                            break;
                        s = (s + 1) & mask;
                    }
                }
            }

            // The distinct strings, copied in first-occurrence order into a fresh array the caller sorts in place.
            NDArray distinct = new NDArray(ar.dtype, Shape.Vector(count), false);
            byte* dst = (byte*)distinct.Address;
            for (int k = 0; k < count; k++)
                Buffer.MemoryCopy(p + (long)firsts[k] * width, dst + (long)k * width, width, width);
            Interlocked.Increment(ref Runs);
            return distinct;
        }

        /// <summary>
        ///     A 32-bit hash of one item's bytes: 8 bytes a step mixed by a multiply and a shift (the last partial step
        ///     zero-padded), then a final avalanche (splitmix64's finalizer), so that strings differing in any byte — the
        ///     trailing NUL padding included — spread over the table's low bits.
        /// </summary>
        /// <param name="e">The item's first byte.</param>
        /// <param name="width">The item's width in bytes (positive).</param>
        /// <returns>The hash.</returns>
        private static uint Hash(byte* e, int width)
        {
            ulong h = 0x9E3779B97F4A7C15UL ^ (ulong)width;
            int i = 0;
            for (; i + 8 <= width; i += 8)
            {
                h = (h ^ *(ulong*)(e + i)) * 0xBF58476D1CE4E5B9UL;
                h ^= h >> 31;
            }
            if (i < width)
            {
                // The last 1 to 7 bytes (an S item's width need not be a multiple of 8; a U item leaves 4) read by at most
                // three loads of 4, 2 and 1 bytes: a variable-length copy here cost ~20 ns an element, more than the rest of
                // the pass.
                int rest = width - i;
                byte* t = e + i;
                ulong tail = 0;
                if ((rest & 4) != 0)
                {
                    tail = *(uint*)t;
                    t += 4;
                }
                if ((rest & 2) != 0)
                {
                    tail = (tail << 16) | *(ushort*)t;
                    t += 2;
                }
                if ((rest & 1) != 0)
                    tail = (tail << 8) | *t;
                h = (h ^ tail) * 0xBF58476D1CE4E5B9UL;
                h ^= h >> 31;
            }
            h ^= h >> 30;
            h *= 0xBF58476D1CE4E5B9UL;
            h ^= h >> 27;
            h *= 0x94D049BB133111EBUL;
            h ^= h >> 31;
            return (uint)h;
        }
    }
}
