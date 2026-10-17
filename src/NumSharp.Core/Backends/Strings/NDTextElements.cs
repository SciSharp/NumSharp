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
    internal static unsafe class NDTextElements
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
        ///     Scans from the element's END in vector chunks (32, then 16 bytes) and then 8 / 4 / 1-byte steps, never
        ///     reading before <paramref name="p"/> or past <c>p + width</c>; with a constant <paramref name="width"/> the
        ///     chunk sequence folds to straight-line code (one compare for a <c>&lt;U8</c> element).
        /// </remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static int Units<TUnit>(byte* p, int width) where TUnit : unmanaged
            => LengthBytes(p, width, sizeof(TUnit)) / sizeof(TUnit);

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
        /// <remarks>The results depend only on CPython's match semantics (fastsearch finds the first occurrence), which
        ///     <see cref="MemoryExtensions.IndexOf{T}(ReadOnlySpan{T}, ReadOnlySpan{T})"/> shares.</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long Find<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end)
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
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long RFind<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end)
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
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long Count<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end)
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
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static byte TailMatch<TUnit>(byte* hay, int hayWidth, byte* sub, int subWidth, long start, long end, bool back)
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
        ///     (<see cref="IsSpace{TUnit}"/>).
        /// </remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void StripWhitespace<TUnit>(byte* src, int width, byte* dst, int type) where TUnit : unmanaged
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
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void StripChars<TUnit>(byte* src, int width, byte* chars, int charsWidth, byte* dst, int type)
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
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void Slice<TUnit>(byte* src, int width, long start, long stop, long step, byte* dst) where TUnit : unmanaged
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
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void Pad<TUnit>(byte* src, int width, long padWidth, byte* fill, byte* dst, int outWidth, int pos)
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
        /// </remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void ZFill<TUnit>(byte* src, int width, long padWidth, byte* dst, int outWidth) where TUnit : unmanaged
        {
            int len = Units<TUnit>(src, width);
            long final = padWidth > len ? padWidth : len;
            long left = final - len;
            FillUnits<TUnit>(dst, 0, left, '0');
            Buffer.MemoryCopy(src, dst + left * sizeof(TUnit), len * sizeof(TUnit), len * sizeof(TUnit));
            ClearFrom(dst, (int)(final * sizeof(TUnit)), outWidth);
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
            int u = sizeof(TUnit);
            int len1 = Units<TUnit>(src, width);
            int len2 = Units<TUnit>(sep, sepWidth);
            long idx = back ? RFind<TUnit>(src, width, sep, sepWidth, 0, long.MaxValue)
                            : Find<TUnit>(src, width, sep, sepWidth, 0, long.MaxValue);
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
