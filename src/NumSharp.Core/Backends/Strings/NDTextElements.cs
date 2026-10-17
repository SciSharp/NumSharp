using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Utilities;

// =============================================================================
// NDTextElements.cs — the per-element bodies of NumPy's fixed-width string ufuncs (S / U), strings phase 2 P2.5
// =============================================================================
//
// WHAT THIS IS
//   One method per loop of numpy/_core/src/umath/string_ufuncs.cpp and the Buffer<ASCII> / Buffer<UTF32> routines of
//   string_buffer.h it calls: str_len, find / rfind / count (fastsearch), startswith / endswith (tailmatch), the six strip
//   loops, _slice, _center / _ljust / _rjust (string_pad), _zfill, _replace, _expandtabs(_length) and
//   _partition_index / _rpartition_index — plus the SIZE bodies of NumPy's Python wrappers (numpy/_core/strings.py), which
//   size the out= array of the computed-width functions with numeric ufuncs (strings plan D15: one fused pass here).
//   Each body reads one element (and the parameter elements beside it) and writes one result, exactly as NumPy's loop
//   body does for that element; the NDIter / trivial-loop drivers (NDTextLoop) and the IL shell of
//   ILKernelGenerator.Text.Methods.cs call them.
//
// THE UNIT TYPE
//   TUnit is byte for S (NumPy's ENCODING::ASCII: one unsigned byte per code unit) and uint for U (ENCODING::UTF32: one
//   npy_ucs4 per code unit). Every body is generic over it and branches on typeof(TUnit) / sizeof(TUnit), which the JIT
//   folds per instantiation; the IL shell calls the closed instantiation with the element widths as constants, so a
//   width test folds too (strings plan D11: emit-time widths without a method per width).
//
// NUMPY'S RULES THAT EVERY BODY SHARES
//   * A string's length is num_codepoints: the index of its last NONZERO unit plus one. Trailing NULs never count,
//     embedded ones do ('a\0' has length 1, 'a\0b' length 3); a U unit above U+10FFFF is just a nonzero unit.
//   * A parameter string (a needle, a strip set, a separator) has its own length by the same rule, so its trailing NULs
//     never take part ('b\0' searches for 'b').
//   * start / end follow CPython's adjust_offsets (negative values count from the end, end clamps to the length, start
//     does not), and positions and counts are in units.
//   * A result shorter than its output element is zero-filled to the element's end (buffer_fill_with_zeros_after_index).
// =============================================================================

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The per-element bodies of NumPy's fixed-width string ufuncs and of the sizing statements of NumPy's
    ///     <c>numpy.strings</c> wrappers, for <c>S</c> (the code unit <c>TUnit</c> = <see cref="byte"/>) and <c>U</c>
    ///     (<see cref="uint"/>) elements: a literal port of <c>string_buffer.h</c> per loop (see the file header).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Contract.</b> Every body reads exactly the element bytes <c>[0, width)</c> of each operand element it is given
    ///     and writes exactly the output element <c>[0, outWidth)</c>; it never reads past an element (a view can end at its
    ///     buffer's last byte, so an over-read could fault — the vector paths use overlapping in-element loads only). The
    ///     integer operands (start, end, width, count, tabsize, the slice indices) are the int64 values NumPy's promoters
    ///     cast them to.
    ///     </para>
    ///     <para>
    ///     <b>Inlining.</b> The bodies the IL shell calls per element carry <see cref="MethodImplOptionsConstants.OptimizeAndInline"/>,
    ///     so the dynamic method inlines them with its constant widths folded; the managed fallback kernels (hosts without
    ///     dynamic code) call the same bodies with run-time widths.
    ///     </para>
    /// </remarks>
    internal static unsafe partial class NDTextElements
    {
        /// <summary>The <c>STRIPTYPE</c> of a strip loop: strip leading units only (<c>lstrip</c>).</summary>
        internal const int LeftStrip = 0;

        /// <summary>The <c>STRIPTYPE</c> of a strip loop: strip trailing units only (<c>rstrip</c>).</summary>
        internal const int RightStrip = 1;

        /// <summary>The <c>STRIPTYPE</c> of a strip loop: strip both ends (<c>strip</c>).</summary>
        internal const int BothStrip = 2;

        /// <summary>The <c>JUSTPOSITION</c> of a pad loop: <c>center</c>.</summary>
        internal const int Center = 0;

        /// <summary>The <c>JUSTPOSITION</c> of a pad loop: <c>ljust</c> (the text first, the fill after).</summary>
        internal const int Left = 1;

        /// <summary>The <c>JUSTPOSITION</c> of a pad loop: <c>rjust</c> (the fill first, the text after).</summary>
        internal const int Right = 2;

        /// <summary>
        ///     <c>INT_MAX</c>: the longest string <c>_expandtabs_length</c> accepts before it raises
        ///     <c>OverflowError: new string is too long</c>.
        /// </summary>
        internal const long ExpandTabsMaxLength = int.MaxValue;

        // =============================================================================================================
        // Lengths
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>num_codepoints</c> of an <c>S</c> / <c>U</c> element: the number of units up to and including the
        ///     last NONZERO unit (trailing NULs never count, embedded ones do).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="p">The element's first byte.</param>
        /// <param name="width">The element width in bytes (a whole number of units).</param>
        /// <returns>The length in units, 0 for an element that is all NULs.</returns>
        /// <remarks>
        ///     Up to 32 bytes: scans from the element's END in vector chunks (32, then 16 bytes) and then 8 / 4 / 1-byte
        ///     steps, never reading before <paramref name="p"/> or past <c>p + width</c>; with a constant
        ///     <paramref name="width"/> the chunk sequence folds to straight-line code (one compare for a <c>&lt;U8</c>
        ///     element). Wider elements of at most 64 units take the lane masks (<see cref="UnitsByLanes{TUnit}"/>: every
        ///     chunk tested at once), because the scan's exit chunk depends on the text's length and mispredicts on
        ///     random lengths (<c>str_len</c> of a <c>&lt;U40</c> array: ~10 ns per element scanning, ~2.5 ns by the masks).
        /// </remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static int Units<TUnit>(byte* p, int width) where TUnit : unmanaged
            => width > 32 && MasksFit<TUnit>(width)
                ? UnitsByLanes<TUnit>(p, width)
                : LengthBytes(p, width, sizeof(TUnit)) / sizeof(TUnit);

        /// <summary>
        ///     The byte length of an element's text: one past its last nonzero byte, rounded up to the end of the unit that
        ///     holds it (<paramref name="unit"/> = 4 for <c>U</c>, whose units are little-endian <c>uint</c>s).
        /// </summary>
        /// <param name="p">The element's first byte.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <returns>The text's byte length, a whole number of units (0 for an all-NUL element).</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static int LengthBytes(byte* p, int width, int unit)
        {
            int end = width;
            if (Vector256.IsHardwareAccelerated)
            {
                while (end >= 32)
                {
                    uint nonzero = ~Vector256.Equals(Vector256.Load(p + end - 32), Vector256<byte>.Zero).ExtractMostSignificantBits();
                    if (nonzero != 0)
                        // The last nonzero byte is at end - 32 + (31 - lzcnt); one past it is end - lzcnt.
                        return RoundUpToUnit(end - BitOperations.LeadingZeroCount(nonzero), unit);
                    end -= 32;
                }
            }
            if (Vector128.IsHardwareAccelerated)
            {
                while (end >= 16)
                {
                    uint nonzero = ~Vector128.Equals(Vector128.Load(p + end - 16), Vector128<byte>.Zero).ExtractMostSignificantBits() & 0xFFFFu;
                    if (nonzero != 0)
                        // A 16-bit mask in a uint: lzcnt counts 16 high zero bits more than the chunk has.
                        return RoundUpToUnit(end - (BitOperations.LeadingZeroCount(nonzero) - 16), unit);
                    end -= 16;
                }
            }
            while (end >= 8)
            {
                ulong x = Unsafe.ReadUnaligned<ulong>(p + end - 8);
                if (x != 0)
                    // Little-endian: the highest nonzero byte of the chunk is byte 7 - lzcnt / 8.
                    return RoundUpToUnit(end - BitOperations.LeadingZeroCount(x) / 8, unit);
                end -= 8;
            }
            if (end >= 4)
            {
                uint x = Unsafe.ReadUnaligned<uint>(p + end - 4);
                if (x != 0)
                    return RoundUpToUnit(end - BitOperations.LeadingZeroCount(x) / 8, unit);
                end -= 4;
            }
            while (end > 0 && p[end - 1] == 0)
                end--;
            return RoundUpToUnit(end, unit);
        }

        /// <summary>Rounds a byte position up to the next unit boundary (identity for <c>S</c>).</summary>
        /// <param name="bytes">The byte position.</param>
        /// <param name="unit">1 or 4.</param>
        /// <returns>The position rounded up to a multiple of <paramref name="unit"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int RoundUpToUnit(int bytes, int unit) => unit == 1 ? bytes : (bytes + 3) & ~3;

        /// <summary>The unit at index <paramref name="i"/> of an element, as NumPy's <c>getchar</c> reads it (unsigned).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="p">The element's first byte.</param>
        /// <param name="i">The unit index.</param>
        /// <returns>The unit as a code value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint UnitAt<TUnit>(byte* p, long i) where TUnit : unmanaged
            => typeof(TUnit) == typeof(byte) ? p[i] : Unsafe.ReadUnaligned<uint>(p + i * 4);

        /// <summary>Writes the unit at index <paramref name="i"/> of an output element.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="p">The element's first byte.</param>
        /// <param name="i">The unit index.</param>
        /// <param name="value">The code value (truncated to a byte for <c>S</c>).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void SetUnit<TUnit>(byte* p, long i, uint value) where TUnit : unmanaged
        {
            if (typeof(TUnit) == typeof(byte))
                p[i] = (byte)value;
            else
                Unsafe.WriteUnaligned(p + i * 4, value);
        }

        /// <summary>
        ///     NumPy's whitespace test of one unit: <c>NumPyOS_ascii_isspace</c> for <c>S</c> (the six ASCII whitespace
        ///     bytes, as in the C locale) and <c>Py_UNICODE_ISSPACE</c> for <c>U</c> (which adds U+001C–U+001F, NEL, NBSP and
        ///     the Unicode spaces).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="c">The unit.</param>
        /// <returns>True for whitespace of the element's class.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsSpace<TUnit>(uint c) where TUnit : unmanaged
            => typeof(TUnit) == typeof(byte) ? c == 0x20 || c - 0x09 <= 4 : NDCharArray.IsPyWhitespace(c);

        // =============================================================================================================
        // find / rfind / count / startswith / endswith
        // =============================================================================================================

        /// <summary>
        ///     CPython's <c>adjust_offsets</c> (<c>bytes_methods.c</c>, which <c>string_buffer.h</c> copies): an end past the
        ///     length clamps to it, a negative end or start counts from the end and stops at 0; a start past the length is
        ///     left as it is (the callers then find the range empty).
        /// </summary>
        /// <param name="start">The start index (adjusted in place).</param>
        /// <param name="end">The end index (adjusted in place).</param>
        /// <param name="len">The string's length in units.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void AdjustOffsets(ref long start, ref long end, long len)
        {
            if (end > len)
                end = len;
            else if (end < 0)
            {
                end += len;
                if (end < 0)
                    end = 0;
            }
            if (start < 0)
            {
                start += len;
                if (start < 0)
                    start = 0;
            }
        }

        /// <summary>
        ///     NumPy's <c>string_find</c>: the lowest index in <c>hay[start, end)</c> where <c>sub</c> (its length, without
        ///     trailing NULs) begins, or -1. An empty <c>sub</c> is found at <c>start</c> when the range is not reversed.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The searched element.</param>
        /// <param name="hayWidth">Its width in bytes.</param>
        /// <param name="sub">The searched-for element.</param>
        /// <param name="subWidth">Its width in bytes.</param>
        /// <param name="start">The start index (CPython's slice notation, see <see cref="AdjustOffsets"/>).</param>
        /// <param name="end">The end index.</param>
        /// <returns>The index in units, or -1.</returns>
        /// <remarks>The results depend only on CPython's match semantics (fastsearch finds the first occurrence): the lane
        ///     masks' lowest match (<see cref="FindLanes{TUnit}"/>, elements of at most 64 units on AVX2 hosts) and
        ///     <see cref="MemoryExtensions.IndexOf{T}(ReadOnlySpan{T}, ReadOnlySpan{T})"/> (<see cref="FindScalar{TUnit}"/>)
        ///     share them.</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long Find<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end)
            where TUnit : unmanaged, IEquatable<TUnit>
            => MasksFit<TUnit>(hayWidth)
                ? FindLanes<TUnit>(hay, hayWidth, sub, subWidth, start, end)
                : FindScalar<TUnit>(hay, hayWidth, sub, subWidth, start, end);

        /// <summary><see cref="Find{TUnit}"/> by the lane masks: the lowest lane of the needle's match mask in the range.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The searched element.</param>
        /// <param name="hayWidth">Its width in bytes (<see cref="MasksFit{TUnit}"/> holds).</param>
        /// <param name="sub">The searched-for element.</param>
        /// <param name="subWidth">Its width in bytes.</param>
        /// <param name="start">The start index.</param>
        /// <param name="end">The end index.</param>
        /// <returns>The index in units, or -1.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long FindLanes<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end)
            where TUnit : unmanaged
        {
            int len1 = Units<TUnit>(hay, hayWidth);
            int len2 = Units<TUnit>(sub, subWidth);
            AdjustOffsets(ref start, ref end, len1);
            if (end - start < len2)
                return -1;
            if (len2 == 0)
                return start;
            // A match at p needs p in [start, end - len2]: 0 <= start and end <= len1 <= 64, so the range is lanes.
            ulong cand = MatchLanes<TUnit>(hay, hayWidth, sub, len2) & RangeLanes(start, end - len2);
            return cand == 0 ? -1 : BitOperations.TrailingZeroCount(cand);
        }

        /// <summary><see cref="Find{TUnit}"/> by span searches (any width; the hosts without AVX2).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The searched element.</param>
        /// <param name="hayWidth">Its width in bytes.</param>
        /// <param name="sub">The searched-for element.</param>
        /// <param name="subWidth">Its width in bytes.</param>
        /// <param name="start">The start index.</param>
        /// <param name="end">The end index.</param>
        /// <returns>The index in units, or -1.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long FindScalar<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            int len1 = Units<TUnit>(hay, hayWidth);
            int len2 = Units<TUnit>(sub, subWidth);
            AdjustOffsets(ref start, ref end, len1);
            if (end - start < len2)
                return -1;
            if (len2 == 0)
                return start;
            var h = new ReadOnlySpan<TUnit>(hay + start * sizeof(TUnit), (int)(end - start));
            int r = len2 == 1
                ? h.IndexOf(Unsafe.ReadUnaligned<TUnit>(sub))
                : h.IndexOf(new ReadOnlySpan<TUnit>(sub, len2));
            return r < 0 ? -1 : r + start;
        }

        /// <summary>
        ///     NumPy's <c>string_rfind</c>: the highest index in <c>hay[start, end)</c> where <c>sub</c> begins, or -1. An
        ///     empty <c>sub</c> is found at <c>end</c> (after <see cref="AdjustOffsets"/>).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The searched element.</param>
        /// <param name="hayWidth">Its width in bytes.</param>
        /// <param name="sub">The searched-for element.</param>
        /// <param name="subWidth">Its width in bytes.</param>
        /// <param name="start">The start index.</param>
        /// <param name="end">The end index.</param>
        /// <returns>The index in units, or -1.</returns>
        /// <remarks>The lane masks' highest match (<see cref="RFindLanes{TUnit}"/>) on AVX2 hosts for elements of at most 64
        ///     units, span searches (<see cref="RFindScalar{TUnit}"/>) otherwise.</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long RFind<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end)
            where TUnit : unmanaged, IEquatable<TUnit>
            => MasksFit<TUnit>(hayWidth)
                ? RFindLanes<TUnit>(hay, hayWidth, sub, subWidth, start, end)
                : RFindScalar<TUnit>(hay, hayWidth, sub, subWidth, start, end);

        /// <summary><see cref="RFind{TUnit}"/> by the lane masks: the highest lane of the needle's match mask in the range.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The searched element.</param>
        /// <param name="hayWidth">Its width in bytes (<see cref="MasksFit{TUnit}"/> holds).</param>
        /// <param name="sub">The searched-for element.</param>
        /// <param name="subWidth">Its width in bytes.</param>
        /// <param name="start">The start index.</param>
        /// <param name="end">The end index.</param>
        /// <returns>The index in units, or -1.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long RFindLanes<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end)
            where TUnit : unmanaged
        {
            int len1 = Units<TUnit>(hay, hayWidth);
            int len2 = Units<TUnit>(sub, subWidth);
            AdjustOffsets(ref start, ref end, len1);
            if (end - start < len2)
                return -1;
            if (len2 == 0)
                return end;
            ulong cand = MatchLanes<TUnit>(hay, hayWidth, sub, len2) & RangeLanes(start, end - len2);
            return cand == 0 ? -1 : 63 - BitOperations.LeadingZeroCount(cand);
        }

        /// <summary><see cref="RFind{TUnit}"/> by span searches (any width; the hosts without AVX2).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The searched element.</param>
        /// <param name="hayWidth">Its width in bytes.</param>
        /// <param name="sub">The searched-for element.</param>
        /// <param name="subWidth">Its width in bytes.</param>
        /// <param name="start">The start index.</param>
        /// <param name="end">The end index.</param>
        /// <returns>The index in units, or -1.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long RFindScalar<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            int len1 = Units<TUnit>(hay, hayWidth);
            int len2 = Units<TUnit>(sub, subWidth);
            AdjustOffsets(ref start, ref end, len1);
            if (end - start < len2)
                return -1;
            if (len2 == 0)
                return end;
            var h = new ReadOnlySpan<TUnit>(hay + start * sizeof(TUnit), (int)(end - start));
            int r = len2 == 1
                ? h.LastIndexOf(Unsafe.ReadUnaligned<TUnit>(sub))
                : h.LastIndexOf(new ReadOnlySpan<TUnit>(sub, len2));
            return r < 0 ? -1 : r + start;
        }

        /// <summary>
        ///     NumPy's <c>string_count</c>: the number of NON-OVERLAPPING occurrences of <c>sub</c> in <c>hay[start, end)</c>,
        ///     taken greedily from the left (fastsearch's FAST_COUNT); an empty <c>sub</c> occurs <c>end - start + 1</c>
        ///     times.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The searched element.</param>
        /// <param name="hayWidth">Its width in bytes.</param>
        /// <param name="sub">The counted element.</param>
        /// <param name="subWidth">Its width in bytes.</param>
        /// <param name="start">The start index.</param>
        /// <param name="end">The end index.</param>
        /// <returns>The count (0 for a reversed or too-short range).</returns>
        /// <remarks>The lane masks (<see cref="CountLanes{TUnit}"/>) on AVX2 hosts for elements of at most 64 units, span
        ///     searches (<see cref="CountScalar{TUnit}"/>) otherwise.</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long Count<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end)
            where TUnit : unmanaged, IEquatable<TUnit>
            => MasksFit<TUnit>(hayWidth)
                ? CountLanes<TUnit>(hay, hayWidth, sub, subWidth, start, end)
                : CountScalar<TUnit>(hay, hayWidth, sub, subWidth, start, end);

        /// <summary>
        ///     <see cref="Count{TUnit}"/> by the lane masks: the match mask's population for a one-unit needle, else the
        ///     greedy walk over its bits (each match clears the lanes it covers, as FAST_COUNT resumes after a match).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The searched element.</param>
        /// <param name="hayWidth">Its width in bytes (<see cref="MasksFit{TUnit}"/> holds).</param>
        /// <param name="sub">The counted element.</param>
        /// <param name="subWidth">Its width in bytes.</param>
        /// <param name="start">The start index.</param>
        /// <param name="end">The end index.</param>
        /// <returns>The count.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long CountLanes<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end)
            where TUnit : unmanaged
        {
            int len1 = Units<TUnit>(hay, hayWidth);
            int len2 = Units<TUnit>(sub, subWidth);
            AdjustOffsets(ref start, ref end, len1);
            if (end < start || end - start < len2)
                return 0;
            if (len2 == 0)
                return end - start + 1;
            ulong c = MatchLanes<TUnit>(hay, hayWidth, sub, len2) & RangeLanes(start, end - len2);
            if (len2 == 1)
                return BitOperations.PopCount(c);
            long count = 0;
            while (c != 0)
            {
                int p = BitOperations.TrailingZeroCount(c);
                count++;
                // The next match starts at or after p + len2.
                c &= ~LowLanes(p + len2);
            }
            return count;
        }

        /// <summary><see cref="Count{TUnit}"/> by span searches (any width; the hosts without AVX2).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The searched element.</param>
        /// <param name="hayWidth">Its width in bytes.</param>
        /// <param name="sub">The counted element.</param>
        /// <param name="subWidth">Its width in bytes.</param>
        /// <param name="start">The start index.</param>
        /// <param name="end">The end index.</param>
        /// <returns>The count.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long CountScalar<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            int len1 = Units<TUnit>(hay, hayWidth);
            int len2 = Units<TUnit>(sub, subWidth);
            AdjustOffsets(ref start, ref end, len1);
            if (end < start || end - start < len2)
                return 0;
            if (len2 == 0)
                return end - start + 1;
            var h = new ReadOnlySpan<TUnit>(hay + start * sizeof(TUnit), (int)(end - start));
            if (len2 == 1)
                return h.Count(Unsafe.ReadUnaligned<TUnit>(sub));
            var n = new ReadOnlySpan<TUnit>(sub, len2);
            long count = 0;
            int i;
            // Greedy from the left: each match consumes its units, the next search starts after it.
            while ((i = h.IndexOf(n)) >= 0)
            {
                count++;
                h = h.Slice(i + len2);
            }
            return count;
        }

        /// <summary>
        ///     NumPy's <c>tailmatch</c> (<c>startswith</c> / <c>endswith</c>): whether <c>sub</c> occurs at the start
        ///     (<paramref name="back"/> false) or at the end (true) of <c>hay[start, end)</c>. An empty <c>sub</c> matches
        ///     any range that is not reversed by more than its length.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The tested element.</param>
        /// <param name="hayWidth">Its width in bytes.</param>
        /// <param name="sub">The prefix / suffix element.</param>
        /// <param name="subWidth">Its width in bytes.</param>
        /// <param name="start">The start index.</param>
        /// <param name="end">The end index.</param>
        /// <param name="back">True for <c>endswith</c> (NumPy's <c>STARTPOSITION::BACK</c>).</param>
        /// <returns>1 for a match, 0 otherwise (the bool the loop stores).</returns>
        /// <remarks>A range from a non-negative <paramref name="start"/> to the element's end (NumPy's default
        ///     <c>end=None</c>) takes one compare at the one offset that can match (<see cref="TailMatchWholeRange{TUnit}"/>);
        ///     any other range takes the lane masks (<see cref="TailMatchLanes{TUnit}"/>: the match mask's bit at the tested
        ///     offset) on AVX2 hosts for elements of at most 64 units, a span compare (<see cref="TailMatchScalar{TUnit}"/>)
        ///     otherwise.</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static byte TailMatch<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end, bool back)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            // An end at or past the element's width can only clamp to the element's length, and a non-negative start is
            // not adjusted: the offset that can match is known without the general paths' mask of every offset.
            if (start >= 0 && end >= hayWidth / sizeof(TUnit))
                return TailMatchWholeRange<TUnit>(hay, hayWidth, sub, subWidth, start, back);
            return MasksFit<TUnit>(hayWidth)
                ? TailMatchLanes<TUnit>(hay, hayWidth, sub, subWidth, start, end, back)
                : TailMatchScalar<TUnit>(hay, hayWidth, sub, subWidth, start, end, back);
        }

        /// <summary>
        ///     <see cref="TailMatch{TUnit}"/> over a range that runs from a non-negative <paramref name="start"/> to the
        ///     element's end (an <c>end</c> at or past the element's width, as NumPy's default <c>None</c> is): one compare of
        ///     the needle's units at the one offset that can match — <paramref name="start"/> for <c>startswith</c>, the
        ///     element's length minus the needle's for <c>endswith</c>.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The tested element.</param>
        /// <param name="hayWidth">Its width in bytes.</param>
        /// <param name="sub">The prefix / suffix element.</param>
        /// <param name="subWidth">Its width in bytes.</param>
        /// <param name="start">The start index, 0 or more.</param>
        /// <param name="back">True for <c>endswith</c>.</param>
        /// <returns>1 for a match, 0 otherwise: the general paths' answer for the same range.</returns>
        /// <remarks>
        ///     <para>
        ///     <c>startswith</c> reads no length of the element. A non-empty needle's last unit is not NUL (its length runs to
        ///     its last nonzero unit), so units equal at <c>[start, start + len2)</c> put a nonzero unit at
        ///     <c>start + len2 - 1</c>, and the element's length is at least <c>start + len2</c>: the general paths' test
        ///     <c>min(end, len1) - len2 &gt;= start</c> holds. A start past the element's length meets only trailing NULs
        ///     there, so the compare fails as the general paths' empty range does. Only an empty needle with a start above 0
        ///     needs the length (it matches while <c>start &lt;= len1</c>).
        ///     </para>
        ///     <para>
        ///     <c>endswith</c> needs the length (the suffix ends at it) but not the match mask: the general lane path built the
        ///     needle's equality mask at every offset of the element and read one bit of it (one vector compare per needle
        ///     unit and element vector; <c>startswith('alpha be')</c> over <c>&lt;U24</c> text ran at NumPy's speed).
        ///     </para>
        /// </remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static byte TailMatchWholeRange<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, bool back)
            where TUnit : unmanaged
        {
            int len2 = Units<TUnit>(sub, subWidth);
            long offset;
            if (back)
            {
                // The range ends at the element's length: the suffix starts len2 units before it, at or after start.
                long len1 = Units<TUnit>(hay, hayWidth);
                offset = len1 - len2;
                if (offset < start)
                    return 0;
                if (len2 == 0)
                    return 1;
            }
            else
            {
                if (len2 == 0)
                    return start == 0 || start <= Units<TUnit>(hay, hayWidth) ? (byte)1 : (byte)0;
                offset = start;
                // A prefix that does not fit in the element's width cannot match (the element's length is at most it).
                if (offset > hayWidth / sizeof(TUnit) - len2)
                    return 0;
            }
            return BytesEqual(hay + offset * sizeof(TUnit), sub, len2 * sizeof(TUnit)) ? (byte)1 : (byte)0;
        }

        /// <summary>
        ///     Whether the <paramref name="bytes"/> bytes at <paramref name="a"/> equal those at <paramref name="b"/>, read
        ///     with overlapping loads that stay inside both ranges (no byte before or past either is touched).
        /// </summary>
        /// <param name="a">The first range.</param>
        /// <param name="b">The second range.</param>
        /// <param name="bytes">The length in bytes, 0 or more.</param>
        /// <returns>True when every byte is equal (true for 0 bytes).</returns>
        /// <remarks>Up to 32 bytes the compare is inline (two 16-byte loads per side, or two 8 / 4-byte reads); a longer
        ///     needle goes to the span compare, whose call costs less than its length there.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool BytesEqual(byte* a, byte* b, int bytes)
        {
            if (bytes >= 16)
            {
                if (bytes > 32)
                    return new ReadOnlySpan<byte>(a, bytes).SequenceEqual(new ReadOnlySpan<byte>(b, bytes));
                // The first and the last 16 bytes (they overlap below 32).
                var diff = (Vector128.Load(a) ^ Vector128.Load(b)) | (Vector128.Load(a + bytes - 16) ^ Vector128.Load(b + bytes - 16));
                return diff == Vector128<byte>.Zero;
            }
            if (bytes >= 8)
                return ((Unsafe.ReadUnaligned<ulong>(a) ^ Unsafe.ReadUnaligned<ulong>(b))
                        | (Unsafe.ReadUnaligned<ulong>(a + bytes - 8) ^ Unsafe.ReadUnaligned<ulong>(b + bytes - 8))) == 0;
            if (bytes >= 4)
                return ((Unsafe.ReadUnaligned<uint>(a) ^ Unsafe.ReadUnaligned<uint>(b))
                        | (Unsafe.ReadUnaligned<uint>(a + bytes - 4) ^ Unsafe.ReadUnaligned<uint>(b + bytes - 4))) == 0;
            // 0 to 3 bytes: only an S needle (a U needle is a whole number of 4-byte units).
            for (int i = 0; i < bytes; i++)
            {
                if (a[i] != b[i])
                    return false;
            }
            return true;
        }

        /// <summary><see cref="TailMatch{TUnit}"/> by the lane masks: the needle's match mask read at the tested offset.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The tested element.</param>
        /// <param name="hayWidth">Its width in bytes (<see cref="MasksFit{TUnit}"/> holds).</param>
        /// <param name="sub">The prefix / suffix element.</param>
        /// <param name="subWidth">Its width in bytes.</param>
        /// <param name="start">The start index.</param>
        /// <param name="end">The end index.</param>
        /// <param name="back">True for <c>endswith</c>.</param>
        /// <returns>1 for a match, 0 otherwise.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static byte TailMatchLanes<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end, bool back)
            where TUnit : unmanaged
        {
            int len1 = Units<TUnit>(hay, hayWidth);
            int len2 = Units<TUnit>(sub, subWidth);
            AdjustOffsets(ref start, ref end, len1);
            end -= len2;
            if (end < start)
                return 0;
            if (len2 == 0)
                return 1;
            // 0 <= start <= offset <= end = (adjusted end) - len2 <= 63: a lane of the mask.
            long offset = back ? end : start;
            return (byte)((MatchLanes<TUnit>(hay, hayWidth, sub, len2) >> (int)offset) & 1);
        }

        /// <summary><see cref="TailMatch{TUnit}"/> by a span compare (any width; the hosts without AVX2).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The tested element.</param>
        /// <param name="hayWidth">Its width in bytes.</param>
        /// <param name="sub">The prefix / suffix element.</param>
        /// <param name="subWidth">Its width in bytes.</param>
        /// <param name="start">The start index.</param>
        /// <param name="end">The end index.</param>
        /// <param name="back">True for <c>endswith</c>.</param>
        /// <returns>1 for a match, 0 otherwise.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static byte TailMatchScalar<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end, bool back)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            int len1 = Units<TUnit>(hay, hayWidth);
            int len2 = Units<TUnit>(sub, subWidth);
            AdjustOffsets(ref start, ref end, len1);
            end -= len2;
            if (end < start)
                return 0;
            if (len2 == 0)
                return 1;
            long offset = back ? end : start;
            return new ReadOnlySpan<TUnit>(hay + offset * sizeof(TUnit), len2).SequenceEqual(new ReadOnlySpan<TUnit>(sub, len2))
                ? (byte)1
                : (byte)0;
        }

        // =============================================================================================================
        // strip / slice (output element = the input's descriptor)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>string_lrstrip_whitespace</c>: the element without its leading (<see cref="LeftStrip"/>,
        ///     <see cref="BothStrip"/>) and trailing (<see cref="RightStrip"/>, <see cref="BothStrip"/>) whitespace, written to
        ///     an output element of the same width and zero-filled.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">The input (and output) width in bytes.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="type">The strip type (<see cref="LeftStrip"/>, <see cref="RightStrip"/>, <see cref="BothStrip"/>).</param>
        /// <remarks>
        ///     NumPy's trailing loop also strips an embedded NUL (<c>*traverse_buf != 0 &amp;&amp; !isspace</c> keeps only a
        ///     nonzero non-space unit), while the leading loop stops at one: <c>strip(' \0ab')</c> is <c>'\0ab'</c> and
        ///     <c>strip('ab\0 ')</c> is <c>'ab'</c> (probed 2.4.2). Whitespace is the element class's
        ///     (<see cref="IsSpace{TUnit}"/>). Elements of at most 64 bytes on AVX2 hosts take the lane masks and one gather
        ///     (<see cref="StripWhitespaceLanes{TUnit}"/>), the rest the unit loops (<see cref="StripWhitespaceScalar{TUnit}"/>):
        ///     a wider element's loops stop at the first kept unit from each end, where whole-element masks would test every
        ///     unit against NumPy's 29 whitespace code points (measured 1.5x slower on 1K <c>U27</c> elements).
        /// </remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void StripWhitespace<TUnit>(byte* src, int width, byte* dst, int type) where TUnit : unmanaged
        {
            if (SelectsFit(width))
                StripWhitespaceLanes<TUnit>(src, width, dst, type);
            else
                StripWhitespaceScalar<TUnit>(src, width, dst, type);
        }

        /// <summary>
        ///     <see cref="StripWhitespace{TUnit}"/> by the lane masks: the first kept unit is the lowest non-whitespace lane,
        ///     the last the highest lane that is neither whitespace nor NUL (NumPy's trailing loop), and one gather moves
        ///     the kept units to the front of the output.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">The input (and output) width in bytes (<see cref="SelectsFit"/> holds).</param>
        /// <param name="dst">The output element.</param>
        /// <param name="type">The strip type.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void StripWhitespaceLanes<TUnit>(byte* src, int width, byte* dst, int type) where TUnit : unmanaged
        {
            // One NUL mask serves the length (one past the last nonzero unit) and the trailing loop's NUL test.
            ulong zero = EqualLanes<TUnit>(src, width, 0);
            int len = 64 - BitOperations.LeadingZeroCount(~zero & LowLanes(width / sizeof(TUnit)));
            ulong inText = LowLanes(len);
            ulong space = SpaceLanes<TUnit>(src, width) & inText;
            int start = 0;
            if (type != RightStrip)
            {
                ulong kept = ~space & inText;
                start = kept == 0 ? len : BitOperations.TrailingZeroCount(kept);
            }
            int stop = len;
            if (type != LeftStrip)
            {
                // Only a nonzero non-space unit at or after start stops the trailing loop.
                ulong kept = ~(space | zero) & inText & ~LowLanes(start);
                stop = kept == 0 ? start : 64 - BitOperations.LeadingZeroCount(kept);
            }
            LoadElement(src, width, out var lo, out var hi);
            CopyLanes<TUnit>(lo, hi, width, start, stop - start, dst, width);
        }

        /// <summary><see cref="StripWhitespace{TUnit}"/> by NumPy's unit loops (any width; the hosts without AVX2).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">The input (and output) width in bytes.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="type">The strip type.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void StripWhitespaceScalar<TUnit>(byte* src, int width, byte* dst, int type) where TUnit : unmanaged
        {
            int len = Units<TUnit>(src, width);
            int start = 0;
            if (type != RightStrip)
                while (start < len && IsSpace<TUnit>(UnitAt<TUnit>(src, start)))
                    start++;
            int stop = len;
            if (type != LeftStrip)
            {
                while (stop > start)
                {
                    uint c = UnitAt<TUnit>(src, stop - 1);
                    if (c != 0 && !IsSpace<TUnit>(c))
                        break;
                    stop--;
                }
            }
            CopyUnitsAndClear<TUnit>(src + start * sizeof(TUnit), stop - start, dst, width);
        }

        /// <summary>
        ///     NumPy's <c>string_lrstrip_chars</c>: the element without its leading / trailing units that occur in
        ///     <paramref name="chars"/> (its length, without trailing NULs), written to an output element of the input's width.
        ///     An empty <paramref name="chars"/> strips nothing.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">The input (and output) width in bytes.</param>
        /// <param name="chars">The strip-set element.</param>
        /// <param name="charsWidth">Its width in bytes.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="type">The strip type.</param>
        /// <remarks>Elements of at most 64 bytes with a set of at most <see cref="StripCharsLaneSet"/> units take the lane
        ///     masks (<see cref="StripCharsLanes{TUnit}"/>: one equality mask per set unit, OR-ed); wider elements the masks
        ///     still describe (a <c>U</c> element of 17 to 64 units) scan 32-byte chunks from each end
        ///     (<see cref="StripCharsWide{TUnit}"/>); the rest the unit loops (<see cref="StripCharsScalar{TUnit}"/>).</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void StripChars<TUnit>(byte* src, int width, byte* chars, int charsWidth, byte* dst, int type)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            if (MasksFit<TUnit>(width))
            {
                int setUnits = Units<TUnit>(chars, charsWidth);
                if (setUnits <= StripCharsLaneSet)
                {
                    if (SelectsFit(width))
                        StripCharsLanes<TUnit>(src, width, chars, charsWidth, dst, type);
                    else
                        StripCharsWideSet<TUnit>(src, width, chars, setUnits, dst, type);
                    return;
                }
            }
            StripCharsScalar<TUnit>(src, width, chars, charsWidth, dst, type);
        }

        /// <summary>The largest strip set (units) <see cref="StripChars{TUnit}"/> tests lane by lane: each set unit costs one
        ///     equality mask, and a long set is cheaper searched per unit.</summary>
        internal const int StripCharsLaneSet = 16;

        /// <summary>
        ///     <see cref="StripChars{TUnit}"/> by the lane masks: the set's lanes are the OR of one equality mask per set
        ///     unit; the first kept unit is the lowest lane outside the set, the last the highest, and one gather moves them.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">The input (and output) width in bytes (<see cref="SelectsFit"/> holds).</param>
        /// <param name="chars">The strip-set element.</param>
        /// <param name="charsWidth">Its width in bytes.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="type">The strip type.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void StripCharsLanes<TUnit>(byte* src, int width, byte* chars, int charsWidth, byte* dst, int type)
            where TUnit : unmanaged
        {
            int len1 = Units<TUnit>(src, width);
            int len2 = len1 == 0 ? 0 : Units<TUnit>(chars, charsWidth);
            int start = 0, stop = len1;
            if (len2 != 0)
            {
                ulong inSet = 0;
                for (int j = 0; j < len2; j++)
                    inSet |= EqualLanes<TUnit>(src, width, UnitAt<TUnit>(chars, j));
                // The units past len1 are trailing NULs: never kept, never part of the text.
                ulong kept = ~inSet & LowLanes(len1);
                if (type != RightStrip)
                    start = kept == 0 ? len1 : BitOperations.TrailingZeroCount(kept);
                if (type != LeftStrip)
                {
                    ulong keptRight = kept & ~LowLanes(start);
                    stop = keptRight == 0 ? start : 64 - BitOperations.LeadingZeroCount(keptRight);
                }
            }
            LoadElement(src, width, out var lo, out var hi);
            CopyLanes<TUnit>(lo, hi, width, start, stop - start, dst, width);
        }

        /// <summary>
        ///     <see cref="StripChars{TUnit}"/> for an element wider than the select helpers hold (more than 64 bytes) that the
        ///     lane masks still describe (a <c>U</c> element of 17 to 64 units), with a set of at most
        ///     <see cref="StripCharsLaneSet"/> units: the length by the lane masks, then 32-byte chunks tested against every
        ///     set unit at once, from the front until a chunk holds a unit outside the set and from the back the same way,
        ///     then one copy and the zero fill.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">The input (and output) width in bytes (<see cref="MasksFit{TUnit}"/> holds, above 64).</param>
        /// <param name="chars">The strip-set element.</param>
        /// <param name="charsWidth">Its width in bytes.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="type">The strip type.</param>
        /// <remarks>
        ///     Stripped runs are short in text, so one chunk from each end usually decides: the cost is two chunks per set
        ///     unit, where whole-element masks cost every chunk per set unit and the scalar body a span search per tested
        ///     unit (1K <c>U27</c> elements with a three-unit set: 32.7 us scalar, 20.7 us whole masks; NumPy 39 us). A
        ///     chunk past the element's end is read as the element's last 32 bytes, so no byte outside
        ///     <c>[src, src + width)</c> is read. The units past the text are its trailing NULs, which a set holding an
        ///     embedded NUL would match: the masks are cut at the text's length.
        /// </remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void StripCharsWide<TUnit>(byte* src, int width, byte* chars, int charsWidth, byte* dst, int type)
            where TUnit : unmanaged
            => StripCharsWideSet<TUnit>(src, width, chars, Units<TUnit>(chars, charsWidth), dst, type);

        /// <summary>
        ///     <see cref="StripCharsWide{TUnit}"/> with the set's length already known (the dispatcher measured it to choose
        ///     this body, so it is read once per element).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">The input (and output) width in bytes (<see cref="MasksFit{TUnit}"/> holds, above 64).</param>
        /// <param name="chars">The strip-set element.</param>
        /// <param name="setUnits">The set's length in units (its trailing NULs excluded), at most
        ///     <see cref="StripCharsLaneSet"/>.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="type">The strip type.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void StripCharsWideSet<TUnit>(byte* src, int width, byte* chars, int setUnits, byte* dst, int type)
            where TUnit : unmanaged
        {
            int len1 = Units<TUnit>(src, width);
            int len2 = len1 == 0 ? 0 : setUnits;
            int start = 0, stop = len1;
            if (len2 != 0)
            {
                int lanes = 32 / sizeof(TUnit);
                int units = width / sizeof(TUnit);
                if (type != RightStrip)
                {
                    start = len1;
                    for (int at = 0; at < len1; at += lanes)
                    {
                        // The chunk [from, from + lanes) lies inside the element; past its end it is the last 32 bytes.
                        int from = Math.Min(at, units - lanes);
                        ulong inSet = (ulong)ChunkSetLanes<TUnit>(src + from * sizeof(TUnit), chars, len2) << from;
                        ulong kept = ~inSet & RangeLanes(at, Math.Min(at + lanes, len1) - 1);
                        if (kept != 0)
                        {
                            start = BitOperations.TrailingZeroCount(kept);
                            break;
                        }
                    }
                }
                if (type != LeftStrip)
                {
                    stop = start;
                    for (int end = len1; end > start; end -= lanes)
                    {
                        // The chunk ending at `end` (the element's first 32 bytes when fewer units are left).
                        int from = Math.Max(end - lanes, 0);
                        ulong inSet = (ulong)ChunkSetLanes<TUnit>(src + from * sizeof(TUnit), chars, len2) << from;
                        ulong kept = ~inSet & RangeLanes(Math.Max(from, start), end - 1);
                        if (kept != 0)
                        {
                            stop = 64 - BitOperations.LeadingZeroCount(kept);
                            break;
                        }
                    }
                }
            }
            ClearWide(dst, width);
            int bytes = (stop - start) * sizeof(TUnit);
            if (bytes > 0)
                Buffer.MemoryCopy(src + start * sizeof(TUnit), dst, bytes, bytes);
        }

        /// <summary>
        ///     Zero-fills an output element of 32 bytes or more with 32-byte stores: whole chunks, then one chunk ending at
        ///     the element's end (overlapping the last whole one). With a constant width (the IL shell's) the stores are
        ///     straight-line, where a span clear of a variable tail branches on its length.
        /// </summary>
        /// <param name="dst">The output element.</param>
        /// <param name="width">Its width in bytes (at least 32).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ClearWide(byte* dst, int width)
        {
            for (int at = 0; at + 32 <= width; at += 32)
                Vector256<byte>.Zero.Store(dst + at);
            if ((width & 31) != 0)
                Vector256<byte>.Zero.Store(dst + width - 32);
        }

        /// <summary>The units of one 32-byte chunk that occur in a strip set: the OR of one equality mask per set unit.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="chunk">The chunk's first byte (32 readable bytes).</param>
        /// <param name="chars">The strip-set element.</param>
        /// <param name="setUnits">The set's length in units (at least one).</param>
        /// <returns>One bit per unit of the chunk (8 for <c>U</c>, 32 for <c>S</c>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint ChunkSetLanes<TUnit>(byte* chunk, byte* chars, int setUnits) where TUnit : unmanaged
        {
            var v = Vector256.Load(chunk);
            uint m = 0;
            for (int j = 0; j < setUnits; j++)
                m |= Predicate<TUnit>(v, MaskEqual, UnitAt<TUnit>(chars, j));
            return m;
        }

        /// <summary><see cref="StripChars{TUnit}"/> by NumPy's unit loops (any width or set; the hosts without AVX2).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">The input (and output) width in bytes.</param>
        /// <param name="chars">The strip-set element.</param>
        /// <param name="charsWidth">Its width in bytes.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="type">The strip type.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void StripCharsScalar<TUnit>(byte* src, int width, byte* chars, int charsWidth, byte* dst, int type)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            int len1 = Units<TUnit>(src, width);
            int len2 = len1 == 0 ? 0 : Units<TUnit>(chars, charsWidth);
            int start = 0, stop = len1;
            if (len2 != 0)
            {
                var set = new ReadOnlySpan<TUnit>(chars, len2);
                if (type != RightStrip)
                    while (start < len1 && set.Contains(Unsafe.ReadUnaligned<TUnit>(src + start * sizeof(TUnit))))
                        start++;
                if (type != LeftStrip)
                    while (stop > start && set.Contains(Unsafe.ReadUnaligned<TUnit>(src + (stop - 1) * sizeof(TUnit))))
                        stop--;
            }
            CopyUnitsAndClear<TUnit>(src + start * sizeof(TUnit), stop - start, dst, width);
        }

        /// <summary>
        ///     NumPy's <c>string_slice_loop</c> body: the units <c>src[start : stop : step]</c> (CPython's
        ///     <c>PySlice_AdjustIndices</c> over the element's length) written to an output element of the input's width.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">The input (and output) width in bytes.</param>
        /// <param name="start">The slice start (the wrapper's default is the extreme of the step's direction).</param>
        /// <param name="stop">The slice stop.</param>
        /// <param name="step">The slice step (never 0: the wrapper raises first).</param>
        /// <param name="dst">The output element.</param>
        /// <remarks>Elements of at most 64 bytes on AVX2 hosts take one gather by <c>start + i·step</c>
        ///     (<see cref="SelectUnits{TUnit}"/>), the rest a copy (step 1) or a unit loop.</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void Slice<TUnit>(byte* src, int width, long start, long stop, long step, byte* dst) where TUnit : unmanaged
        {
            if (SelectsFit(width))
                SliceLanes<TUnit>(src, width, start, stop, step, dst);
            else
                SliceScalar<TUnit>(src, width, start, stop, step, dst);
        }

        /// <summary><see cref="Slice{TUnit}"/> by one gather: output unit i is input unit <c>start + i·step</c> for the
        ///     adjusted slice's units.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">The input (and output) width in bytes (<see cref="SelectsFit"/> holds).</param>
        /// <param name="start">The slice start.</param>
        /// <param name="stop">The slice stop.</param>
        /// <param name="step">The slice step (never 0).</param>
        /// <param name="dst">The output element.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void SliceLanes<TUnit>(byte* src, int width, long start, long stop, long step, byte* dst) where TUnit : unmanaged
        {
            int len = Units<TUnit>(src, width);
            long n = AdjustSliceIndices(len, ref start, ref stop, step);
            // n <= len <= 64, and every start + i*step below n is a unit index; a step past int's range selects at most
            // one unit, so its truncation reaches only lanes the count zeroes.
            LoadElement(src, width, out var lo, out var hi);
            SelectUnits<TUnit>(lo, hi, width, width, (int)start, unchecked((int)step), (int)n, out var rlo, out var rhi);
            StoreElement(dst, width, rlo, rhi);
        }

        /// <summary><see cref="Slice{TUnit}"/> by a copy (step 1) or a unit walk (any width; the hosts without AVX2).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">The input (and output) width in bytes.</param>
        /// <param name="start">The slice start.</param>
        /// <param name="stop">The slice stop.</param>
        /// <param name="step">The slice step (never 0).</param>
        /// <param name="dst">The output element.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void SliceScalar<TUnit>(byte* src, int width, long start, long stop, long step, byte* dst) where TUnit : unmanaged
        {
            int len = Units<TUnit>(src, width);
            long n = AdjustSliceIndices(len, ref start, ref stop, step);
            if (step == 1)
            {
                CopyUnitsAndClear<TUnit>(src + start * sizeof(TUnit), (int)n, dst, width);
                return;
            }
            // Only the n units the slice selects are read; the index of the last one is in range, so no step past it is
            // ever taken (NumPy's pointer walk steps once more, without reading).
            long index = start;
            for (long i = 0; i < n; i++, index += step)
                SetUnit<TUnit>(dst, i, UnitAt<TUnit>(src, index));
            ClearFrom(dst, (int)n * sizeof(TUnit), width);
        }

        /// <summary>
        ///     CPython's <c>PySlice_AdjustIndices</c>: clamps <paramref name="start"/> / <paramref name="stop"/> to the length
        ///     in the step's direction and returns the slice length.
        /// </summary>
        /// <param name="length">The sequence length.</param>
        /// <param name="start">The start (adjusted in place).</param>
        /// <param name="stop">The stop (adjusted in place).</param>
        /// <param name="step">The step (nonzero).</param>
        /// <returns>The number of selected items.</returns>
        /// <remarks>The arithmetic is CPython's on 64-bit integers, unchecked: a step of <see cref="long.MinValue"/>
        ///     (<c>-step</c> wraps to itself) selects one item, as CPython's release build computes.</remarks>
        internal static long AdjustSliceIndices(long length, ref long start, ref long stop, long step)
        {
            unchecked
            {
                if (start < 0)
                {
                    start += length;
                    if (start < 0)
                        start = step < 0 ? -1 : 0;
                }
                else if (start >= length)
                    start = step < 0 ? length - 1 : length;

                if (stop < 0)
                {
                    stop += length;
                    if (stop < 0)
                        stop = step < 0 ? -1 : 0;
                }
                else if (stop >= length)
                    stop = step < 0 ? length - 1 : length;

                if (step < 0)
                {
                    if (stop < start)
                        return (start - stop - 1) / -step + 1;
                }
                else if (start < stop)
                    return (stop - start - 1) / step + 1;
                return 0;
            }
        }

        // =============================================================================================================
        // center / ljust / rjust / zfill
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>string_pad</c> (<c>_center</c> / <c>_ljust</c> / <c>_rjust</c>): the element padded with
        ///     <paramref name="fill"/>'s first unit to <c>max(width, 0)</c> units (CPython's <c>str.center</c> split of the
        ///     padding for <see cref="Center"/>), or copied as it is when it is already that long; the output element is
        ///     zero-filled past the result.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="padWidth">The requested width in units (the wrapper passes <c>max(str_len(a), width)</c>; any int64
        ///     gives the same result).</param>
        /// <param name="fill">The fill element (its first unit is the fill character: the wrapper checked its length is 1
        ///     and cast it to the input's class).</param>
        /// <param name="dst">The output element.</param>
        /// <param name="outWidth">The output width in bytes (at least the result's).</param>
        /// <param name="pos">The justification (<see cref="Center"/>, <see cref="Left"/>, <see cref="Right"/>).</param>
        /// <remarks>Input and output elements of at most 64 bytes on AVX2 hosts take one gather with the fill blended in
        ///     (<see cref="PadLanes{TUnit}"/>), the rest fills and copies (<see cref="PadScalar{TUnit}"/>).</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void Pad<TUnit>(byte* src, int width, long padWidth, byte* fill, byte* dst, int outWidth, int pos)
            where TUnit : unmanaged
        {
            if (SelectsFit(width) && SelectsFit(outWidth))
                PadLanes<TUnit>(src, width, padWidth, UnitAt<TUnit>(fill, 0), dst, outWidth, pos);
            else
                PadScalar<TUnit>(src, width, padWidth, fill, dst, outWidth, pos);
        }

        /// <summary>
        ///     <see cref="Pad{TUnit}"/> (and <see cref="ZFill{TUnit}"/>'s padding) by one gather: output unit o is input unit
        ///     <c>o - left</c> inside the text, the fill elsewhere below the final width, and 0 past it.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes (<see cref="SelectsFit"/> holds).</param>
        /// <param name="padWidth">The requested width in units.</param>
        /// <param name="fillUnit">The fill character.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="outWidth">The output width in bytes (<see cref="SelectsFit"/> holds; at least the result's).</param>
        /// <param name="pos">The justification.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void PadLanes<TUnit>(byte* src, int width, long padWidth, uint fillUnit, byte* dst, int outWidth, int pos)
            where TUnit : unmanaged
        {
            int len = Units<TUnit>(src, width);
            long final = padWidth > 0 ? padWidth : 0;
            LoadElement(src, width, out var lo, out var hi);
            if (len >= final)
            {
                CopyLanes<TUnit>(lo, hi, width, 0, len, dst, outWidth);
                return;
            }
            long pad = final - len;
            long left = pos == Center ? pad / 2 + (pad & final & 1) : pos == Left ? 0 : pad;
            // final <= the output's units <= 64 (the wrapper sized the output to the largest result).
            PadUnits<TUnit>(lo, hi, width, outWidth, (int)left, len, (int)final, fillUnit, out var rlo, out var rhi);
            StoreElement(dst, outWidth, rlo, rhi);
        }

        /// <summary><see cref="Pad{TUnit}"/> by fills and copies (any width; the hosts without AVX2).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="padWidth">The requested width in units.</param>
        /// <param name="fill">The fill element (its first unit is the fill character).</param>
        /// <param name="dst">The output element.</param>
        /// <param name="outWidth">The output width in bytes.</param>
        /// <param name="pos">The justification.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void PadScalar<TUnit>(byte* src, int width, long padWidth, byte* fill, byte* dst, int outWidth, int pos)
            where TUnit : unmanaged
        {
            int len = Units<TUnit>(src, width);
            long final = padWidth > 0 ? padWidth : 0;
            if (len >= final)
            {
                CopyUnitsAndClear<TUnit>(src, len, dst, outWidth);
                return;
            }
            long pad = final - len;
            long left = pos == Center ? pad / 2 + (pad & final & 1) : pos == Left ? 0 : pad;
            uint f = UnitAt<TUnit>(fill, 0);
            FillUnits<TUnit>(dst, 0, left, f);
            Buffer.MemoryCopy(src, dst + left * sizeof(TUnit), len * sizeof(TUnit), len * sizeof(TUnit));
            FillUnits<TUnit>(dst, left + len, final, f);
            ClearFrom(dst, (int)(final * sizeof(TUnit)), outWidth);
        }

        /// <summary>
        ///     NumPy's <c>string_zfill</c>: the element right-justified with <c>'0'</c> to <c>max(width, 0)</c> units, and a
        ///     leading <c>'+'</c> / <c>'-'</c> of the original moved to the front (CPython's <c>str.zfill</c>).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="padWidth">The requested width in units.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="outWidth">The output width in bytes.</param>
        /// <remarks>
        ///     NumPy reads the sign at <c>final - len</c>, which assumes the width is at least the length (the wrapper's
        ///     <c>np.maximum(str_len(a), width)</c>); a smaller width is taken as the length here, which is that rule.
        ///     Input and output elements of at most 64 bytes on AVX2 hosts build the padded result by one gather
        ///     (<see cref="PadUnits{TUnit}"/>); the sign move is two unit writes either way.
        /// </remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void ZFill<TUnit>(byte* src, int width, long padWidth, byte* dst, int outWidth) where TUnit : unmanaged
            => ZFillCore<TUnit>(src, width, padWidth, dst, outWidth, SelectsFit(width) && SelectsFit(outWidth));

        /// <summary>
        ///     The body of <see cref="ZFill{TUnit}"/> with the padding's form chosen by the caller: one gather
        ///     (<paramref name="lanes"/>, which needs <see cref="SelectsFit"/> for both widths) or fills and a copy.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="padWidth">The requested width in units.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="outWidth">The output width in bytes.</param>
        /// <param name="lanes">True for the gather (a constant at the call, so the other form is dropped).</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void ZFillCore<TUnit>(byte* src, int width, long padWidth, byte* dst, int outWidth, bool lanes)
            where TUnit : unmanaged
        {
            int len = Units<TUnit>(src, width);
            long final = padWidth > len ? padWidth : len;
            long left = final - len;
            if (lanes)
            {
                LoadElement(src, width, out var lo, out var hi);
                // final <= the output's units <= 64.
                PadUnits<TUnit>(lo, hi, width, outWidth, (int)left, len, (int)final, '0', out var rlo, out var rhi);
                StoreElement(dst, outWidth, rlo, rhi);
            }
            else
            {
                FillUnits<TUnit>(dst, 0, left, '0');
                Buffer.MemoryCopy(src, dst + left * sizeof(TUnit), len * sizeof(TUnit), len * sizeof(TUnit));
                ClearFrom(dst, (int)(final * sizeof(TUnit)), outWidth);
            }
            if (len == 0)
                return;
            uint c = UnitAt<TUnit>(dst, left);
            if (c == '+' || c == '-')
            {
                SetUnit<TUnit>(dst, left, '0');
                SetUnit<TUnit>(dst, 0, c);
            }
        }

        // =============================================================================================================
        // replace / expandtabs / partition
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>string_replace</c>: <paramref name="count"/> leftmost non-overlapping occurrences of
        ///     <paramref name="old"/> replaced by <paramref name="new"/> (an empty <paramref name="old"/> inserts
        ///     <paramref name="new"/> before each unit and at the end, up to <paramref name="count"/> times), written to an
        ///     output element and zero-filled.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="old">The searched element.</param>
        /// <param name="oldWidth">Its width in bytes.</param>
        /// <param name="new">The replacement element.</param>
        /// <param name="newWidth">Its width in bytes.</param>
        /// <param name="count">The replacement limit the loop receives (the wrapper's <c>counts</c>: the occurrences,
        ///     capped by the caller's count; at most 0 replaces nothing).</param>
        /// <param name="dst">The output element.</param>
        /// <param name="outWidth">The output width in bytes (the wrapper's maximum result size).</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void Replace<TUnit>(byte* src, int width, byte* old, int oldWidth, byte* @new, int newWidth, long count,
                                            byte* dst, int outWidth)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            int len1 = Units<TUnit>(src, width);
            int len2 = Units<TUnit>(old, oldWidth);
            int len3 = Units<TUnit>(@new, newWidth);
            int u = sizeof(TUnit);
            // The cases that copy the input unchanged, in NumPy's order.
            if (count <= 0 || len1 < len2 || (len2 <= 0 && len3 <= 0)
                || (len2 == len3 && new ReadOnlySpan<byte>(old, len2 * u).SequenceEqual(new ReadOnlySpan<byte>(@new, len3 * u))))
            {
                CopyUnitsAndClear<TUnit>(src, len1, dst, outWidth);
                return;
            }

            int pos = 0, o = 0;
            if (len2 > 0)
            {
                var needle = new ReadOnlySpan<TUnit>(old, len2);
                for (long time = 0; time < count; time++)
                {
                    var rest = new ReadOnlySpan<TUnit>(src + pos * u, len1 - pos);
                    int p = len2 == 1 ? rest.IndexOf(needle[0]) : rest.IndexOf(needle);
                    if (p < 0)
                        break;
                    Buffer.MemoryCopy(src + pos * u, dst + o * u, p * u, p * u);
                    o += p;
                    pos += p;
                    Buffer.MemoryCopy(@new, dst + o * u, len3 * u, len3 * u);
                    o += len3;
                    pos += len2;
                }
            }
            else
            {
                // An empty old: new before every unit, then once at the end, count insertions in all.
                while (count > 0)
                {
                    Buffer.MemoryCopy(@new, dst + o * u, len3 * u, len3 * u);
                    o += len3;
                    if (--count <= 0)
                        break;
                    SetUnit<TUnit>(dst, o, UnitAt<TUnit>(src, pos));
                    o++;
                    pos++;
                }
            }
            CopyUnitsAndClear<TUnit>(src + pos * u, len1 - pos, dst + o * u, outWidth - o * u);
        }

        /// <summary>
        ///     NumPy's <c>string_expandtabs_length</c>: the length of the element with every tab expanded to the next
        ///     multiple of <paramref name="tabsize"/> (a column count reset by <c>'\n'</c> / <c>'\r'</c>; a tabsize of 0 or
        ///     less removes the tabs), or -1 when it passes <see cref="ExpandTabsMaxLength"/> (NumPy's
        ///     <c>OverflowError: new string is too long</c>, raised by the caller after the pass).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="tabsize">The tab size.</param>
        /// <returns>NumPy's expanded "length", or -1: a tab adds its column count, but every other character adds its
        ///     BYTES (<c>num_bytes_next_character</c>: 1 for <c>S</c>, 4 for <c>U</c>) — so a <c>U</c> output is sized four
        ///     times wider than its text needs (<c>expandtabs(['ab\tc'], 4)</c> is <c>'&lt;U16'</c> wide, probed 2.4.2), and
        ///     the overflow test sees that count.</returns>
        /// <remarks>The arithmetic is NumPy's npy_intp arithmetic, unchecked: a tab of a huge tabsize wraps the running
        ///     length negative, which the overflow test catches as NumPy's does.</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long ExpandTabsLength<TUnit>(byte* src, int width, long tabsize) where TUnit : unmanaged
            => MasksFit<TUnit>(width)
                ? ExpandTabsLengthLanes<TUnit>(src, width, tabsize)
                : ExpandTabsLengthScalar<TUnit>(src, width, tabsize);

        /// <summary>
        ///     <see cref="ExpandTabsLength{TUnit}"/> by the lane masks for an element without a line break: the non-tab units'
        ///     bytes at once, then one step per tab (an element holding <c>'\n'</c> / <c>'\r'</c> takes the unit loop, whose
        ///     column resets there).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width in bytes (<see cref="MasksFit{TUnit}"/> holds).</param>
        /// <param name="tabsize">The tab size.</param>
        /// <returns>NumPy's expanded "length", or -1 past <see cref="ExpandTabsMaxLength"/>.</returns>
        /// <remarks>The running length only grows (or wraps negative on a huge tab size), so testing it after each tab and
        ///     at the end finds every overflow the unit loop's test after each unit finds.</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long ExpandTabsLengthLanes<TUnit>(byte* src, int width, long tabsize) where TUnit : unmanaged
        {
            int len = Units<TUnit>(src, width);
            ulong inText = LowLanes(len);
            if (((EqualLanes<TUnit>(src, width, '\n') | EqualLanes<TUnit>(src, width, '\r')) & inText) != 0)
                return ExpandTabsLengthScalar<TUnit>(src, width, tabsize);
            ulong tabs = EqualLanes<TUnit>(src, width, '\t') & inText;
            unchecked
            {
                // Every other character adds its bytes (num_bytes_next_character); its column is its output position.
                long newLen = (long)(len - BitOperations.PopCount(tabs)) * sizeof(TUnit);
                long pos = 0;
                int prev = 0;
                while (tabs != 0)
                {
                    int i = BitOperations.TrailingZeroCount(tabs);
                    tabs &= tabs - 1;
                    pos += i - prev;
                    long incr = TabRun(pos, tabsize);
                    pos += incr;
                    newLen += incr;
                    if (newLen > ExpandTabsMaxLength || newLen < 0)
                        return -1;
                    prev = i + 1;
                }
                return newLen > ExpandTabsMaxLength || newLen < 0 ? -1 : newLen;
            }
        }

        /// <summary><see cref="ExpandTabsLength{TUnit}"/> by NumPy's unit loop (any element; the hosts without AVX2).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="tabsize">The tab size.</param>
        /// <returns>NumPy's expanded "length", or -1.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long ExpandTabsLengthScalar<TUnit>(byte* src, int width, long tabsize) where TUnit : unmanaged
        {
            int len = Units<TUnit>(src, width);
            unchecked
            {
                long newLen = 0, linePos = 0;
                for (int i = 0; i < len; i++)
                {
                    uint ch = UnitAt<TUnit>(src, i);
                    if (ch == '\t')
                    {
                        if (tabsize > 0)
                        {
                            long incr = tabsize - linePos % tabsize;
                            linePos += incr;
                            newLen += incr;
                        }
                    }
                    else
                    {
                        linePos++;
                        // NumPy adds tmp.num_bytes_next_character(): the unit's bytes, not 1.
                        newLen += sizeof(TUnit);
                        if (ch == '\n' || ch == '\r')
                            linePos = 0;
                    }
                    if (newLen > ExpandTabsMaxLength || newLen < 0)
                        return -1;
                }
                return newLen;
            }
        }

        /// <summary>
        ///     NumPy's <c>string_expandtabs</c>: the element with its tabs expanded (see <see cref="ExpandTabsLength{TUnit}"/>),
        ///     written to an output element and zero-filled.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="tabsize">The tab size.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="outWidth">The output width in bytes (the size pass's maximum, so the expansion fits).</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void ExpandTabs<TUnit>(byte* src, int width, long tabsize, byte* dst, int outWidth) where TUnit : unmanaged
        {
            if (SelectsFit(width))
            {
                // An element without a line break whose expansion fits 64 bytes: one gather with the runs blended in.
                int len = Units<TUnit>(src, width);
                ulong inText = LowLanes(len);
                if (((EqualLanes<TUnit>(src, width, '\n') | EqualLanes<TUnit>(src, width, '\r')) & inText) == 0)
                {
                    LoadElement(src, width, out var lo, out var hi);
                    if (ExpandTabsUnits<TUnit>(lo, hi, width, EqualLanes<TUnit>(src, width, '\t') & inText, len, tabsize,
                                               out var rlo, out var rhi))
                    {
                        StoreElementZeroTail(dst, outWidth, rlo, rhi);
                        return;
                    }
                }
            }
            ExpandTabsScalar<TUnit>(src, width, tabsize, dst, outWidth);
        }

        /// <summary><see cref="ExpandTabs{TUnit}"/> by NumPy's unit loop (any element; the hosts without AVX2).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="tabsize">The tab size.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="outWidth">The output width in bytes.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void ExpandTabsScalar<TUnit>(byte* src, int width, long tabsize, byte* dst, int outWidth) where TUnit : unmanaged
        {
            int len = Units<TUnit>(src, width);
            long o = 0, linePos = 0;
            unchecked
            {
                for (int i = 0; i < len; i++)
                {
                    uint ch = UnitAt<TUnit>(src, i);
                    if (ch == '\t')
                    {
                        if (tabsize > 0)
                        {
                            long incr = tabsize - linePos % tabsize;
                            linePos += incr;
                            FillUnits<TUnit>(dst, o, o + incr, ' ');
                            o += incr;
                        }
                    }
                    else
                    {
                        linePos++;
                        SetUnit<TUnit>(dst, o, ch);
                        o++;
                        if (ch == '\n' || ch == '\r')
                            linePos = 0;
                    }
                }
            }
            ClearFrom(dst, (int)(o * sizeof(TUnit)), outWidth);
        }

        /// <summary>
        ///     NumPy's <c>string_partition</c> (<c>_partition_index</c> / <c>_rpartition_index</c>): the element split at the
        ///     separator occurrence <paramref name="pos"/> into the text before it, the separator and the text after it —
        ///     the whole element in the first part (<see cref="Front"/>) or the third (back) when <paramref name="pos"/> is
        ///     negative. Each part is zero-filled to its output width.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="sep">The separator element (cast to the input's descriptor by the wrapper; never empty here — the
        ///     caller raised <c>ValueError: empty separator</c> first).</param>
        /// <param name="sepWidth">Its width in bytes.</param>
        /// <param name="back">True for <c>rpartition</c> (the last occurrence; the element in the third part when absent).</param>
        /// <param name="o1">The first part's output element.</param>
        /// <param name="w1">Its width in bytes.</param>
        /// <param name="o2">The separator part's output element.</param>
        /// <param name="w2">Its width in bytes.</param>
        /// <param name="o3">The third part's output element.</param>
        /// <param name="w3">Its width in bytes.</param>
        /// <remarks>The occurrence is found here (<see cref="Find{TUnit}"/> / <see cref="RFind{TUnit}"/> over the whole
        ///     element) instead of being read from the wrapper's <c>pos</c> array: the function is pure, so recomputing it
        ///     gives NumPy's position without a temporary.</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void Partition<TUnit>(byte* src, int width, byte* sep, int sepWidth, bool back,
                                              byte* o1, int w1, byte* o2, int w2, byte* o3, int w3)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            if (SelectsFit(width) && SelectsFit(sepWidth) && SelectsFit(w1) && SelectsFit(w2) && SelectsFit(w3))
                PartitionLanes<TUnit>(src, width, sep, sepWidth, back, o1, w1, o2, w2, o3, w3);
            else
                PartitionScalar<TUnit>(src, width, sep, sepWidth, back, o1, w1, o2, w2, o3, w3);
        }

        /// <summary>
        ///     <see cref="Partition{TUnit}"/> by the lane masks and gathers: the occurrence from <see cref="FindLanes{TUnit}"/>
        ///     / <see cref="RFindLanes{TUnit}"/>, each part one gather of the loaded element (or the separator) and one store.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes (<see cref="SelectsFit"/> holds for it and every part).</param>
        /// <param name="sep">The separator element (never empty here).</param>
        /// <param name="sepWidth">Its width in bytes.</param>
        /// <param name="back">True for <c>rpartition</c>.</param>
        /// <param name="o1">The first part's output element.</param>
        /// <param name="w1">Its width in bytes.</param>
        /// <param name="o2">The separator part's output element.</param>
        /// <param name="w2">Its width in bytes.</param>
        /// <param name="o3">The third part's output element.</param>
        /// <param name="w3">Its width in bytes.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void PartitionLanes<TUnit>(byte* src, int width, byte* sep, int sepWidth, bool back,
                                                   byte* o1, int w1, byte* o2, int w2, byte* o3, int w3)
            where TUnit : unmanaged
        {
            int len1 = Units<TUnit>(src, width);
            int len2 = Units<TUnit>(sep, sepWidth);
            long idx = back ? RFindLanes<TUnit>(src, width, sep, sepWidth, 0, long.MaxValue)
                            : FindLanes<TUnit>(src, width, sep, sepWidth, 0, long.MaxValue);
            LoadElement(src, width, out var lo, out var hi);
            if (idx < 0)
            {
                // The whole element in the first part (partition) or the third (rpartition); the others empty.
                CopyLanes<TUnit>(lo, hi, width, 0, back ? 0 : len1, o1, w1);
                StoreElement(o2, w2, Vector256<byte>.Zero, Vector256<byte>.Zero);
                CopyLanes<TUnit>(lo, hi, width, 0, back ? len1 : 0, o3, w3);
                return;
            }
            int at = (int)idx;
            CopyLanes<TUnit>(lo, hi, width, 0, at, o1, w1);
            LoadElement(sep, sepWidth, out var slo, out var shi);
            CopyLanes<TUnit>(slo, shi, sepWidth, 0, len2, o2, w2);
            CopyLanes<TUnit>(lo, hi, width, at + len2, len1 - at - len2, o3, w3);
        }

        /// <summary><see cref="Partition{TUnit}"/> by span searches and copies (any width; the hosts without AVX2).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="sep">The separator element (never empty here).</param>
        /// <param name="sepWidth">Its width in bytes.</param>
        /// <param name="back">True for <c>rpartition</c>.</param>
        /// <param name="o1">The first part's output element.</param>
        /// <param name="w1">Its width in bytes.</param>
        /// <param name="o2">The separator part's output element.</param>
        /// <param name="w2">Its width in bytes.</param>
        /// <param name="o3">The third part's output element.</param>
        /// <param name="w3">Its width in bytes.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void PartitionScalar<TUnit>(byte* src, int width, byte* sep, int sepWidth, bool back,
                                                    byte* o1, int w1, byte* o2, int w2, byte* o3, int w3)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            int u = sizeof(TUnit);
            int len1 = Units<TUnit>(src, width);
            int len2 = Units<TUnit>(sep, sepWidth);
            long idx = back ? RFindScalar<TUnit>(src, width, sep, sepWidth, 0, long.MaxValue)
                            : FindScalar<TUnit>(src, width, sep, sepWidth, 0, long.MaxValue);
            if (idx < 0)
            {
                CopyUnitsAndClear<TUnit>(back ? null : src, back ? 0 : len1, o1, w1);
                ClearFrom(o2, 0, w2);
                CopyUnitsAndClear<TUnit>(back ? src : null, back ? len1 : 0, o3, w3);
                return;
            }
            CopyUnitsAndClear<TUnit>(src, (int)idx, o1, w1);
            CopyUnitsAndClear<TUnit>(sep, len2, o2, w2);
            CopyUnitsAndClear<TUnit>(src + (idx + len2) * u, (int)(len1 - idx - len2), o3, w3);
        }

        /// <summary>The partition direction constant for <see cref="Partition{TUnit}"/>'s <c>back</c>: the first occurrence.</summary>
        internal const bool Front = false;

        // =============================================================================================================
        // The wrappers' size statements (strings plan D15: one fused pass computes the output width)
        // =============================================================================================================

        /// <summary>
        ///     The size of one element of <c>np.maximum(str_len(a), width)</c> (<c>center</c> / <c>ljust</c> / <c>rjust</c> /
        ///     <c>zfill</c>), folded into the running maximum at <c>acc[0]</c>.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="padWidth">The requested width (int64).</param>
        /// <param name="acc">The accumulator: <c>acc[0]</c> = the maximum so far.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void PadSize<TUnit>(byte* src, int width, long padWidth, long* acc) where TUnit : unmanaged
        {
            long size = Math.Max(Units<TUnit>(src, width), padWidth);
            if (size > acc[0])
                acc[0] = size;
        }

        /// <summary>
        ///     The size of one element of NumPy's <c>replace</c> sizing: <c>str_len(a) + counts * (str_len(new) -
        ///     str_len(old))</c> with <c>counts = count(a, old)</c> capped by a non-negative <paramref name="count"/>, folded
        ///     into the running maximum at <c>acc[0]</c>.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="old">The searched element.</param>
        /// <param name="oldWidth">Its width in bytes.</param>
        /// <param name="new">The replacement element.</param>
        /// <param name="newWidth">Its width in bytes.</param>
        /// <param name="count">The caller's count (int64; negative means every occurrence).</param>
        /// <param name="acc">The accumulator: <c>acc[0]</c> = the maximum so far.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void ReplaceSize<TUnit>(byte* src, int width, byte* old, int oldWidth, byte* @new, int newWidth, long count,
                                                long* acc)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            long found = Count<TUnit>(src, width, old, oldWidth, 0, long.MaxValue);
            long counts = count < 0 ? found : Math.Min(found, count);
            long size = Units<TUnit>(src, width) + counts * (Units<TUnit>(@new, newWidth) - Units<TUnit>(old, oldWidth));
            if (size > acc[0])
                acc[0] = size;
        }

        /// <summary>
        ///     The replacement limit NumPy's <c>replace</c> passes its loop for one element:
        ///     <c>np.where(count &lt; 0, counts, np.minimum(counts, count))</c> with <c>counts = count(a, old, 0, MAX)</c>.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="old">The searched element.</param>
        /// <param name="oldWidth">Its width in bytes.</param>
        /// <param name="count">The caller's count.</param>
        /// <returns>The limit.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long ReplaceLimit<TUnit>(byte* src, int width, byte* old, int oldWidth, long count)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            long found = Count<TUnit>(src, width, old, oldWidth, 0, long.MaxValue);
            return count < 0 ? found : Math.Min(found, count);
        }

        /// <summary>
        ///     <see cref="Replace{TUnit}"/> with NumPy's per-element limit (<see cref="ReplaceLimit{TUnit}"/>) computed from
        ///     the caller's <paramref name="count"/>: the fill pass of the fused <c>replace</c> (the loop recounts the
        ///     occurrences, as NumPy's sizing statement and loop both count them).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="old">The searched element.</param>
        /// <param name="oldWidth">Its width in bytes.</param>
        /// <param name="new">The replacement element.</param>
        /// <param name="newWidth">Its width in bytes.</param>
        /// <param name="count">The caller's count (negative: every occurrence).</param>
        /// <param name="dst">The output element.</param>
        /// <param name="outWidth">The output width in bytes.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void ReplaceCounted<TUnit>(byte* src, int width, byte* old, int oldWidth, byte* @new, int newWidth, long count,
                                                   byte* dst, int outWidth)
            where TUnit : unmanaged, IEquatable<TUnit>
            => Replace<TUnit>(src, width, old, oldWidth, @new, newWidth, ReplaceLimit<TUnit>(src, width, old, oldWidth, count), dst, outWidth);

        /// <summary>
        ///     The size of one element of <c>_expandtabs_length</c>, folded into the running maximum at <c>acc[0]</c>; an
        ///     element whose expansion passes <see cref="ExpandTabsMaxLength"/> sets <c>acc[1]</c> (NumPy's loop records its
        ///     OverflowError and finishes the pass; the caller raises it).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="tabsize">The tab size.</param>
        /// <param name="acc">The accumulator: <c>acc[0]</c> = the maximum, <c>acc[1]</c> = the overflow flag.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void ExpandTabsSize<TUnit>(byte* src, int width, long tabsize, long* acc) where TUnit : unmanaged
        {
            long size = ExpandTabsLength<TUnit>(src, width, tabsize);
            if (size < 0)
                acc[1] = 1;
            else if (size > acc[0])
                acc[0] = size;
        }

        /// <summary>
        ///     The sizes of one element of NumPy's <c>partition</c> / <c>rpartition</c> sizing: the before-part's length
        ///     (<c>acc[0]</c>), the separator's length (<c>acc[1]</c>), the after-part's length (<c>acc[2]</c>) as running
        ///     maxima; <c>acc[3]</c> stays 1 while no separator was found (NumPy's <c>np.all(not_found)</c>) and <c>acc[4]</c>
        ///     becomes 1 at an empty separator (the loop's <c>ValueError: empty separator</c>, raised by the caller).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="sep">The separator element (in the element's descriptor).</param>
        /// <param name="sepWidth">Its width in bytes.</param>
        /// <param name="back">True for <c>rpartition</c>.</param>
        /// <param name="acc">The five accumulators.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void PartitionSize<TUnit>(byte* src, int width, byte* sep, int sepWidth, bool back, long* acc)
            where TUnit : unmanaged, IEquatable<TUnit>
        {
            long len1 = Units<TUnit>(src, width);
            long len2 = Units<TUnit>(sep, sepWidth);
            long pos = back ? RFind<TUnit>(src, width, sep, sepWidth, 0, long.MaxValue)
                            : Find<TUnit>(src, width, sep, sepWidth, 0, long.MaxValue);
            bool notFound = pos < 0;
            long b1 = back ? (notFound ? 0 : pos) : (notFound ? len1 : pos);
            long b3 = back ? (notFound ? len1 : len1 - pos - len2) : (notFound ? 0 : len1 - pos - len2);
            if (b1 > acc[0])
                acc[0] = b1;
            if (len2 > acc[1])
                acc[1] = len2;
            if (b3 > acc[2])
                acc[2] = b3;
            if (!notFound)
                acc[3] = 0;
            if (len2 == 0)
                acc[4] = 1;
        }

        // =============================================================================================================
        // Copies
        // =============================================================================================================

        /// <summary>Copies <paramref name="units"/> units to an output element and zero-fills the rest of it.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="src">The first unit to copy (unused when <paramref name="units"/> is 0).</param>
        /// <param name="units">The number of units.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="dstBytes">The output element's width in bytes.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CopyUnitsAndClear<TUnit>(byte* src, int units, byte* dst, int dstBytes) where TUnit : unmanaged
        {
            int bytes = units * sizeof(TUnit);
            if (bytes > 0)
                Buffer.MemoryCopy(src, dst, bytes, bytes);
            ClearFrom(dst, bytes, dstBytes);
        }

        /// <summary>Zero-fills an output element from byte <paramref name="from"/> to its end.</summary>
        /// <param name="dst">The output element.</param>
        /// <param name="from">The first byte to clear.</param>
        /// <param name="dstBytes">The element's width in bytes.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ClearFrom(byte* dst, int from, int dstBytes)
        {
            if (dstBytes > from)
                new Span<byte>(dst + from, dstBytes - from).Clear();
        }

        /// <summary>Writes the unit <paramref name="value"/> at indices <c>[from, to)</c> of an output element.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="dst">The output element.</param>
        /// <param name="from">The first index.</param>
        /// <param name="to">One past the last index.</param>
        /// <param name="value">The unit.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void FillUnits<TUnit>(byte* dst, long from, long to, uint value) where TUnit : unmanaged
        {
            if (to <= from)
                return;
            if (typeof(TUnit) == typeof(byte))
                new Span<byte>(dst + from, (int)(to - from)).Fill((byte)value);
            else
                new Span<uint>(dst + from * 4, (int)(to - from)).Fill(value);
        }
    }
}
