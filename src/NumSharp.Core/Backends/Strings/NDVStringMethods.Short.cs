using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// NDVStringMethods.Short.cs — the short-cell forms of the read-only StringDType loops (strings phase 2, P2.5; strings plan
// D12)
// =============================================================================
//
// WHAT THIS IS
//   A StringDType text of at most 15 bytes lives IN its 16-byte cell (byte 15 = 0x60 | size). The loops that only READ
//   text — str_len, find / rfind / count / index / rindex, startswith / endswith — compute a pair of short cells from one
//   16-byte vector each: the masks of NUL bytes, of continuation bytes (0x80–0xBF) and of each needle byte's equality,
//   then a few bit operations. The general bodies (NDVStringMethods.cs) walk the bytes with branches on every lead byte and
//   call the long-safe span searches, which mispredict on every element of random text (14–32 ns per element, about
//   NumPy's own cost); these forms run without a branch on the data.
//
// WHAT THEY KEEP EXACTLY (the general bodies' rules, see NDVStringMethods.cs's header)
//   * Lengths and positions in code points without the trailing NULs: the length is one past the last nonzero byte, the
//     code points are its lead bytes (every byte that is not a continuation byte — T text is always well-formed UTF-8,
//     so the k-th code point starts at the k-th lead byte, which BMI2's pdep finds).
//   * The searches use the needle's FULL bytes, trailing NULs included, except find / rfind of a one-code-point ASCII
//     needle (one byte) and count of a one-byte needle; a match must lie inside the code-point window [start, end).
//   * tailmatch reads haystack bytes past the text as zero (the cell is masked to its size first; the bytes past a short
//     cell's size are not guaranteed zero).
//
// WHERE THEY RUN
//   Both cells short (never an NA: an NA cell carries the missing flag) on a host with SSSE3 / BMI2 (pshufb, pdep); any
//   other pair takes the general body. The unit tests compare the two forms over every size and every needle kind.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static unsafe partial class NDVStringMethods
    {
        /// <summary>Whether a cell holds a short string (its text in bytes 0–14, its size in byte 15's low nibble).</summary>
        /// <param name="cell">The 16-byte cell.</param>
        /// <returns>True for a short string (never an NA, an arena or a heap string).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsShort(byte* cell) => (cell[15] & VStringAllocator.NPY_STRING_FLAG_MASK) == ShortFlags;

        /// <summary>The size of a short cell's text (byte 15's low nibble).</summary>
        /// <param name="cell">A short cell.</param>
        /// <returns>0 to 15.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ShortSize(byte* cell) => cell[15] & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK;

        /// <summary>Whether the short-cell search forms can run on this machine: SSSE3 (<c>pshufb</c>) and BMI2 (<c>pdep</c>),
        ///     both JIT constants.</summary>
        private static bool ShortSearchSupported => Ssse3.IsSupported && Bmi2.IsSupported;

        /// <summary>The byte lanes below <paramref name="n"/> (0–16) of a 16-byte vector, all-ones.</summary>
        /// <param name="n">The lane count.</param>
        /// <returns>The mask vector.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<byte> ByteLanes(int n)
            => Vector128.GreaterThan(Vector128.Create((sbyte)n),
                                     Vector128.Create((sbyte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)).AsByte();

        /// <summary>The bits below <paramref name="n"/> (0–32) of a 32-bit mask.</summary>
        /// <param name="n">The bit count.</param>
        /// <returns>The mask.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint LowBits(int n) => n >= 32 ? uint.MaxValue : (1u << n) - 1;

        /// <summary>
        ///     The text of a short cell masked to its size, its length (one past the last nonzero byte) and its lead bytes
        ///     below that length — the three facts every short form starts from.
        /// </summary>
        /// <param name="cell">A short cell.</param>
        /// <param name="size">Its text size.</param>
        /// <param name="end">Receives one past the last nonzero byte (the text without its trailing NULs).</param>
        /// <param name="leads">Receives the lead (non-continuation) bytes below <paramref name="end"/>: one bit per code
        ///     point.</param>
        /// <returns>The 16 bytes, zero from <paramref name="size"/> on.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<byte> ShortText(byte* cell, int size, out int end, out uint leads)
        {
            var v = Vector128.Load(cell) & ByteLanes(size);
            uint nonzero = ~Vector128.Equals(v, Vector128<byte>.Zero).ExtractMostSignificantBits() & 0xFFFFu;
            end = 32 - BitOperations.LeadingZeroCount(nonzero);
            // A continuation byte is 0x80..0xBF: below -64 as a signed byte.
            uint cont = Vector128.LessThan(v.AsSByte(), Vector128.Create((sbyte)-64)).ExtractMostSignificantBits();
            leads = ~cont & LowBits(end);
            return v;
        }

        /// <summary>
        ///     NumPy's <c>num_codepoints</c> of a short cell: the lead bytes below one past the last nonzero byte, from one
        ///     vector (<see cref="CodePoints"/> walks the bytes).
        /// </summary>
        /// <param name="cell">A short cell.</param>
        /// <param name="size">Its text size.</param>
        /// <returns>The code points.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long ShortCodePoints(byte* cell, int size)
        {
            if (!Vector128.IsHardwareAccelerated)
                return CodePoints(cell, size);
            ShortText(cell, size, out _, out uint leads);
            return BitOperations.PopCount(leads);
        }

        /// <summary>
        ///     <see cref="FindCore"/> over two short cells: the needle's match mask in the window of code points
        ///     <c>[start, end)</c>, its lowest / highest bit or its greedy count, converted to code points.
        /// </summary>
        /// <param name="h">The haystack cell (short).</param>
        /// <param name="n">The needle cell (short).</param>
        /// <param name="start">The start (code points).</param>
        /// <param name="end">The end.</param>
        /// <param name="mode">0 find, 1 rfind, 2 count.</param>
        /// <returns>The code-point index, -1, or the count (<see cref="FindCore"/>'s answer).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long FindShort(byte* h, byte* n, long start, long end, int mode)
        {
            var hv = ShortText(h, ShortSize(h), out int hEnd, out uint hLeads);
            int ns = ShortSize(n);
            ShortText(n, ns, out _, out uint nLeads);
            long len1 = BitOperations.PopCount(hLeads);
            long len2 = BitOperations.PopCount(nLeads);
            NDTextElements.AdjustOffsets(ref start, ref end, len1);
            if (mode == 2)
            {
                if (end < start || end - start < len2)
                    return 0;
                if (len2 == 0)
                    return end - start + 1;
            }
            else
            {
                if (end - start < len2)
                    return -1;
                if (len2 == 0)
                    return mode == 0 ? start : end;
            }
            // The window's byte range (find_start_end_locs): the start-th and end-th code points' first bytes, or the
            // text's end (0 <= start <= end <= len1 here).
            int sb = start >= len1 ? hEnd : BitOperations.TrailingZeroCount(Bmi2.ParallelBitDeposit(1u << (int)start, hLeads));
            int eb = end >= len1 ? hEnd : BitOperations.TrailingZeroCount(Bmi2.ParallelBitDeposit(1u << (int)end, hLeads));
            uint cand;
            int k;
            if (mode == 2 ? ns == 1 : len2 == 1 && n[0] < 0x80)
            {
                // A one-byte search: count's one-byte needle, find's one-code-point ASCII needle (its trailing NULs ignored).
                cand = Vector128.Equals(hv, Vector128.Create(n[0])).ExtractMostSignificantBits();
                k = 1;
            }
            else
            {
                // The needle's full bytes, trailing NULs included.
                cand = Vector128.Equals(hv, Vector128.Create(n[0])).ExtractMostSignificantBits();
                for (int j = 1; j < ns && cand != 0; j++)
                    cand &= Vector128.Equals(hv, Vector128.Create(n[j])).ExtractMostSignificantBits() >> j;
                k = ns;
            }
            // A match at byte p must lie in the window: sb <= p and p + k <= eb.
            cand = eb - k < sb ? 0u : cand & LowBits(eb - k + 1) & ~LowBits(sb);
            if (mode == 2)
            {
                if (k == 1)
                    return BitOperations.PopCount(cand);
                long count = 0;
                while (cand != 0)
                {
                    int p = BitOperations.TrailingZeroCount(cand);
                    count++;
                    cand &= ~LowBits(p + k);
                }
                return count;
            }
            if (cand == 0)
                return -1;
            int at = mode == 0 ? BitOperations.TrailingZeroCount(cand) : 31 - BitOperations.LeadingZeroCount(cand);
            // utf8_character_index: the code points before the match.
            return BitOperations.PopCount(hLeads & LowBits(at));
        }

        /// <summary>
        ///     <see cref="TailMatchCore"/> over two short cells: the haystack shifted down to the tested code point's first
        ///     byte (bytes past its text zero) compared with the needle's full bytes.
        /// </summary>
        /// <param name="h">The haystack cell (short).</param>
        /// <param name="n">The needle cell (short).</param>
        /// <param name="start">The start (code points).</param>
        /// <param name="end">The end.</param>
        /// <param name="back">True for endswith.</param>
        /// <returns>1 for a match, 0 otherwise.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static byte TailMatchShort(byte* h, byte* n, long start, long end, bool back)
        {
            var hv = ShortText(h, ShortSize(h), out int hEnd, out uint hLeads);
            int ns = ShortSize(n);
            var nv = ShortText(n, ns, out _, out uint nLeads);
            long len1 = BitOperations.PopCount(hLeads);
            long len2 = BitOperations.PopCount(nLeads);
            NDTextElements.AdjustOffsets(ref start, ref end, len1);
            end -= len2;
            if (end < start)
                return 0;
            if (len2 == 0)
                return 1;
            // 0 <= pos <= end - len2 < len1: a code point of the text.
            long pos = back ? end : start;
            int pb = BitOperations.TrailingZeroCount(Bmi2.ParallelBitDeposit(1u << (int)pos, hLeads));
            // pshufb by pb + i with the 0x70 saturating bias: an index past byte 15 sets bit 7, which reads zero.
            var iota = Vector128.Create((byte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            var shifted = Ssse3.Shuffle(hv, Sse2.AddSaturate(iota + Vector128.Create((byte)pb), Vector128.Create((byte)0x70)));
            uint differ = ~Vector128.Equals(shifted, nv).ExtractMostSignificantBits() & LowBits(ns);
            return differ == 0 ? (byte)1 : (byte)0;
        }
    }
}
