using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// NDTextElements.Case.cs — NumPy's case maps upper / lower / swapcase / capitalize / title over S / U (strings phase 2,
// P2.6 milestone 2)
// =============================================================================
//
// NUMPY'S STRUCTURE (numpy/_core/strings.py, multiarraymodule.c _vec_string)
//   upper(a) is `a_arr = np.asarray(a); _vec_string(a_arr, a_arr.dtype, 'upper')` (lower / swapcase / capitalize / title
//   alike): a fresh C-ordered array of a's OWN dtype, filled in C order with `bytes.upper(np.bytes_ element)` for S and
//   `str.upper(np.str_ element)` for U, each result stored by setitem — cut to the element width, zero-padded. So a
//   mapping that grows ('ß'.upper() is 'SS', 'ﬃ' is 'FFI') is truncated: np.strings.upper(np.array(['ß'], 'U1')) is 'S'.
//   The element is the scalar NumPy makes: its trailing NULs dropped (an embedded NUL is a character).
//
// THE PYTHON METHODS (CPython 3.12)
//   bytes (bytes_methods.c, ASCII rules only, the length kept): upper / lower map 'a'-'z' / 'A'-'Z'; swapcase both;
//   capitalize: byte 0 upper, the rest lower; title: a letter after a letter is lowered, after anything else uppercased.
//   str (unicodeobject.c, case_operation with do_upper / do_lower / do_swapcase / do_capitalize / do_title): the FULL
//   mappings of the Unicode database (one code point can become up to three), lower_ucs4's final sigma (U+03A3 lowers to
//   U+03C2 when a cased character precedes it — skipping case-ignorable ones — and none follows), swapcase's upper →
//   lower / lower → upper / other kept, capitalize's titlecase first character (Python 3.8+) and lowered rest, title's
//   "previous character cased" state (_PyUnicode_IsCased of the previous INPUT character). A U unit past U+10FFFF makes
//   CPython raise SystemError "invalid maximum character passed to PyUnicode_New" — at the scalar for a one-unit text,
//   at the result for a longer one (case_operation's maxchar holds the unit, which maps to itself) — so any element
//   holding one raises; the error is the same for every element, so the order the loop visits them cannot show.
//
// HOW AN ELEMENT IS COMPUTED HERE (strings plan D14, D11)
//   * S: the bytes methods are length-preserving maps of (previous byte is a letter, byte), so a whole element is mapped
//     a 32-byte chunk at a time: compares for the letters, one shift for "previous", an XOR of 0x20. The trailing NULs map
//     to themselves, so the whole width is mapped as is.
//   * U, ASCII elements: the same chunk map over eight 32-bit units (ASCII str methods equal the bytes methods).
//   * U, the rest of the call after its first non-ASCII element (D14's sticky escalation): each chunk gathers the units'
//     BMP flags words (one vpgatherdd), then the mapping deltas by record (one or two more), when no unit needs more than
//     a delta: none past U+FFFF, none with a multi-code-point mapping (EXTENDED_CASE), no U+03A3 (the final sigma) for a
//     map that lowers. Any of those sends the element to CPython's algorithm, ported statement for statement
//     (CaseMapCodePoints), which also records the SystemError.
//   * Without AVX2: the scalar bodies (the bytes methods, CaseMapCodePoints), which the tests hold the vector forms to.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static unsafe partial class NDTextElements
    {
        /// <summary><c>upper</c>: every character's full uppercase mapping (<c>bytes</c>: ASCII letters only).</summary>
        internal const int UpperMap = 0;

        /// <summary><c>lower</c>: every character's full lowercase mapping, the final sigma included.</summary>
        internal const int LowerMap = 1;

        /// <summary><c>swapcase</c>: an uppercase character lowered, a lowercase one uppercased, any other kept.</summary>
        internal const int SwapcaseMap = 2;

        /// <summary><c>capitalize</c>: the first character in titlecase (<c>bytes</c>: uppercase), the rest lowered.</summary>
        internal const int CapitalizeMap = 3;

        /// <summary><c>title</c>: a character after a cased one lowered, any other titlecased.</summary>
        internal const int TitleMap = 4;

        /// <summary><see cref="CaseState.Status"/>: a <c>U</c> unit past U+10FFFF (CPython's SystemError).</summary>
        internal const long CaseStatusInvalidMaxChar = 1;

        /// <summary>CPython's SystemError text for a <c>str</c> holding a unit past U+10FFFF.</summary>
        internal const string InvalidMaxCharText = "invalid maximum character passed to PyUnicode_New";

        /// <summary>The Python method name of a case map (<c>upper</c> ...), as NumPy's <c>_vec_string</c> calls it.</summary>
        /// <param name="map">The map (<see cref="UpperMap"/> ... <see cref="TitleMap"/>).</param>
        /// <returns>The method name.</returns>
        /// <exception cref="ArgumentOutOfRangeException">An unknown map.</exception>
        internal static string CaseMapName(int map) => map switch
        {
            UpperMap => "upper",
            LowerMap => "lower",
            SwapcaseMap => "swapcase",
            CapitalizeMap => "capitalize",
            TitleMap => "title",
            _ => throw new ArgumentOutOfRangeException(nameof(map), map, "Not a case map."),
        };

        /// <summary>
        ///     The per-call state of a case-map kernel, passed as its auxdata: D14's escalation word, the error status and
        ///     the table pointers, read once per call (a <c>DynamicMethod</c> that read the statics itself would re-test the
        ///     class's initialization at every read).
        /// </summary>
        /// <remarks>One block for the whole call, so a later NDIter chunk starts where the earlier one left off.</remarks>
        [StructLayout(LayoutKind.Sequential)]
        internal struct CaseState
        {
            /// <summary>0 while every <c>U</c> element so far was ASCII (the compare path), 1 once one was not (the gather
            ///     path for the rest of the call).</summary>
            internal long Escalated;

            /// <summary>0, or <see cref="CaseStatusInvalidMaxChar"/> once an element held a unit past U+10FFFF.</summary>
            internal long Status;

            /// <summary><see cref="PyUnicodeData.BmpTable"/>.</summary>
            internal uint* Flags;

            /// <summary><see cref="PyUnicodeData.CaseDeltas"/>: the upper, lower and title mapping tables side by side.</summary>
            internal int* Deltas;

            /// <summary><see cref="PyUnicodeData.CaseDeltaStride"/>: the offset of the lower table in <see cref="Deltas"/> (the
            ///     title table starts at twice it).</summary>
            internal int Records;
        }

        /// <summary>A fresh call state: not escalated, no error, the table pointers read.</summary>
        /// <returns>The state.</returns>
        internal static CaseState NewCaseState() => new CaseState
        {
            Escalated = 0,
            Status = 0,
            Flags = PyUnicodeData.BmpTable,
            Deltas = PyUnicodeData.CaseDeltas,
            Records = PyUnicodeData.CaseDeltaStride,
        };

        // =============================================================================================================
        // The element body
        // =============================================================================================================

        /// <summary>
        ///     One element of NumPy's case map: <paramref name="dst"/> (an element of the same width) receives the mapped text
        ///     of <paramref name="src"/>, cut to the width and zero-padded (see the file header).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c> (the <c>bytes</c> methods), <see cref="uint"/> for <c>U</c>
        ///     (the <c>str</c> methods).</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">The element width in bytes (a constant in the IL shell).</param>
        /// <param name="dst">The output element (never overlapping <paramref name="src"/>: a fresh array).</param>
        /// <param name="map">The map (a constant in the IL shell).</param>
        /// <param name="state">The call's state (the escalation word, the error status, the tables).</param>
        /// <remarks>A <c>U</c> element with a unit past U+10FFFF sets <see cref="CaseState.Status"/> (the driver raises
        ///     CPython's SystemError after the loop) and leaves its output zero.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void CaseMap<TUnit>(byte* src, int width, byte* dst, int map, CaseState* state) where TUnit : unmanaged
        {
            if (typeof(TUnit) == typeof(byte))
            {
                if (Avx2.IsSupported)
                    CaseMapBytes(src, width, dst, map);
                else
                    CaseMapBytesScalar(src, width, dst, map);
                return;
            }
            if (Avx2.IsSupported)
            {
                if (state->Escalated == 0)
                {
                    if (CaseMapAsciiUnits(src, width, dst, map))
                        return;
                    // The first non-ASCII element of the call: the gather path for it and every later one.
                    state->Escalated = 1;
                }
                if (CaseMapGatherUnits(src, width, dst, map, state))
                    return;
            }
            CaseMapUnitsCold(src, width, dst, map, state);
        }

        // =============================================================================================================
        // S: the bytes methods, a 32-byte chunk at a time
        // =============================================================================================================

        /// <summary>
        ///     The <c>bytes</c> case map of a whole <c>S</c> element: one padded chunk up to 32 bytes, else whole chunks and
        ///     an overlapping last one (the map of a byte depends only on the byte before it, so the overlap recomputes the
        ///     same bytes).
        /// </summary>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width (bytes).</param>
        /// <param name="dst">The output element.</param>
        /// <param name="map">The map.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CaseMapBytes(byte* src, int width, byte* dst, int map)
        {
            var lane0 = Vector256.CreateScalar((byte)0xFF);
            if (width <= 32)
            {
                StorePadded32(dst, width, CaseBytes(LoadPadded32(src, width), map, Vector256<byte>.Zero, lane0));
                return;
            }
            // Chunk 0 holds byte 0 (capitalize's uppercased byte); later chunks take "previous is a letter" from the byte
            // before them.
            CaseBytes(Vector256.Load(src), map, Vector256<byte>.Zero, lane0).Store(dst);
            int o = 32;
            for (; o + 32 <= width; o += 32)
            {
                var v = Vector256.Load(src + o);
                CaseBytes(v, map, PreviousByteLetter(src + o, map), Vector256<byte>.Zero).Store(dst + o);
            }
            if (o < width)
            {
                int t = width - 32;
                CaseBytes(Vector256.Load(src + t), map, PreviousByteLetter(src + t, map), Vector256<byte>.Zero).Store(dst + t);
            }
        }

        /// <summary>
        ///     The title map's carry into a chunk: lane 0 all-ones when the byte before <paramref name="p"/> is an ASCII
        ///     letter (the bytes methods' <c>previous_is_cased</c>), else zero; zero for every other map.
        /// </summary>
        /// <param name="p">The chunk's first byte (not the element's first byte).</param>
        /// <param name="map">The map.</param>
        /// <returns>The carry vector.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> PreviousByteLetter(byte* p, int map)
        {
            if (map != TitleMap)
                return Vector256<byte>.Zero;
            uint b = (uint)(p[-1] | 0x20) - 'a';
            return Vector256.CreateScalar(b <= 25 ? (byte)0xFF : (byte)0);
        }

        /// <summary>
        ///     The <c>bytes</c> (and ASCII <c>str</c>) case map of 32 bytes: the letters by compares, the bits to flip by the
        ///     map, then an XOR of 0x20 (an ASCII letter's case bit).
        /// </summary>
        /// <param name="v">The bytes.</param>
        /// <param name="map">The map (a constant: every branch below folds).</param>
        /// <param name="carry">title: lane 0 all-ones when the byte before the chunk is a letter.</param>
        /// <param name="first">capitalize: lane 0 all-ones when the chunk starts the element.</param>
        /// <returns>The mapped bytes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> CaseBytes(Vector256<byte> v, int map, Vector256<byte> carry, Vector256<byte> first)
        {
            var lower = ByteInRange(v - Vector256.Create((byte)'a'));
            var upper = ByteInRange(v - Vector256.Create((byte)'A'));
            Vector256<byte> flip;
            if (map == UpperMap)
                flip = lower;
            else if (map == LowerMap)
                flip = upper;
            else if (map == SwapcaseMap)
                flip = lower | upper;
            else if (map == CapitalizeMap)
                flip = Vector256.ConditionalSelect(first, lower, upper);
            else
            {
                var letter = lower | upper;
                // Each byte's "previous byte is a letter": the letter mask moved up one byte across both 128-bit lanes
                // (vperm2i128 puts the low lane under the high one, vpalignr shifts by 15), the carry at byte 0.
                var prev = Avx2.AlignRight(letter, Avx2.Permute2x128(letter, letter, 0x08), 15) | carry;
                flip = Vector256.ConditionalSelect(prev, upper, lower);
            }
            return v ^ (flip & Vector256.Create((byte)0x20));
        }

        /// <summary>The lanes of <paramref name="d"/> at most 25 (a letter, after subtracting 'a' or 'A'): <c>vpminub</c> +
        ///     <c>vpcmpeqb</c>.</summary>
        /// <param name="d">The bytes minus the letter base.</param>
        /// <returns>All-ones lanes for the letters.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> ByteInRange(Vector256<byte> d) => Vector256.Equals(Vector256.Min(d, Vector256.Create((byte)25)), d);

        /// <summary>
        ///     The <c>bytes</c> case map of 16 bytes (a short <c>StringDType</c> cell, a 16-byte tail): <see cref="CaseBytes"/>
        ///     over one 128-bit vector.
        /// </summary>
        /// <param name="v">The bytes.</param>
        /// <param name="map">The map.</param>
        /// <param name="carry">title: lane 0 all-ones when the byte before is a letter.</param>
        /// <param name="first">capitalize: lane 0 all-ones when the vector starts the text.</param>
        /// <returns>The mapped bytes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector128<byte> CaseBytes128(Vector128<byte> v, int map, Vector128<byte> carry, Vector128<byte> first)
        {
            var max = Vector128.Create((byte)25);
            var dl = v - Vector128.Create((byte)'a');
            var du = v - Vector128.Create((byte)'A');
            var lower = Vector128.Equals(Vector128.Min(dl, max), dl);
            var upper = Vector128.Equals(Vector128.Min(du, max), du);
            Vector128<byte> flip;
            if (map == UpperMap)
                flip = lower;
            else if (map == LowerMap)
                flip = upper;
            else if (map == SwapcaseMap)
                flip = lower | upper;
            else if (map == CapitalizeMap)
                flip = Vector128.ConditionalSelect(first, lower, upper);
            else
            {
                var prev = Sse2.ShiftLeftLogical128BitLane(lower | upper, 1) | carry;
                flip = Vector128.ConditionalSelect(prev, upper, lower);
            }
            return v ^ (flip & Vector128.Create((byte)0x20));
        }

        /// <summary>
        ///     The <c>bytes</c> case map of any run of bytes (an ASCII <c>StringDType</c> text of runtime size): 16-byte
        ///     vectors, the last one overlapping, so no load or store leaves <c>[src, src + size)</c> / <c>[dst, dst +
        ///     size)</c>; a run under 16 bytes goes through a zeroed stack copy.
        /// </summary>
        /// <param name="src">The bytes.</param>
        /// <param name="size">Their number (any).</param>
        /// <param name="dst">The output (not overlapping <paramref name="src"/>).</param>
        /// <param name="map">The map.</param>
        /// <remarks>Needs SSE2 (every x64 host); the caller takes the scalar body elsewhere.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void CaseMapByteRun(byte* src, long size, byte* dst, int map)
        {
            var lane0 = Vector128.CreateScalar((byte)0xFF);
            if (size < 16)
            {
                byte* tmp = stackalloc byte[16];
                Vector128<byte>.Zero.Store(tmp);
                Buffer.MemoryCopy(src, tmp, 16, size);
                CaseBytes128(Vector128.Load(tmp), map, Vector128<byte>.Zero, lane0).Store(tmp);
                Buffer.MemoryCopy(tmp, dst, size, size);
                return;
            }
            CaseBytes128(Vector128.Load(src), map, Vector128<byte>.Zero, lane0).Store(dst);
            long o = 16;
            for (; o + 16 <= size; o += 16)
                CaseBytes128(Vector128.Load(src + o), map, PreviousByteLetter128(src + o, map), Vector128<byte>.Zero).Store(dst + o);
            if (o < size)
            {
                long t = size - 16;
                CaseBytes128(Vector128.Load(src + t), map, PreviousByteLetter128(src + t, map), Vector128<byte>.Zero).Store(dst + t);
            }
        }

        /// <summary>The 128-bit form of <see cref="PreviousByteLetter"/>.</summary>
        /// <param name="p">The vector's first byte (not the text's first byte).</param>
        /// <param name="map">The map.</param>
        /// <returns>The carry vector.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<byte> PreviousByteLetter128(byte* p, int map)
        {
            if (map != TitleMap)
                return Vector128<byte>.Zero;
            uint b = (uint)(p[-1] | 0x20) - 'a';
            return Vector128.CreateScalar(b <= 25 ? (byte)0xFF : (byte)0);
        }

        // =============================================================================================================
        // U: ASCII elements (the compare path)
        // =============================================================================================================

        /// <summary>
        ///     The case map of a <c>U</c> element whose units are all ASCII: the <c>bytes</c> rules over 32-bit lanes.
        /// </summary>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width in bytes (4 per unit).</param>
        /// <param name="dst">The output element.</param>
        /// <param name="map">The map.</param>
        /// <returns>False when a unit is not ASCII (the output may be partly written: the caller's path rewrites all of
        ///     it).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool CaseMapAsciiUnits(byte* src, int width, byte* dst, int map)
        {
            var lane0 = Vector256.CreateScalar(uint.MaxValue);
            if (width <= 32)
            {
                var u = LoadPadded32(src, width).AsUInt32();
                if (!IsAsciiUnits(u))
                    return false;
                StorePadded32(dst, width, CaseUnits(u, map, Vector256<uint>.Zero, lane0).AsByte());
                return true;
            }
            var first = Vector256.Load((uint*)src);
            if (!IsAsciiUnits(first))
                return false;
            CaseUnits(first, map, Vector256<uint>.Zero, lane0).Store((uint*)dst);
            int o = 32;
            for (; o + 32 <= width; o += 32)
            {
                var u = Vector256.Load((uint*)(src + o));
                if (!IsAsciiUnits(u))
                    return false;
                CaseUnits(u, map, PreviousUnitLetter(src + o, map), Vector256<uint>.Zero).Store((uint*)(dst + o));
            }
            if (o < width)
            {
                int t = width - 32;
                var u = Vector256.Load((uint*)(src + t));
                if (!IsAsciiUnits(u))
                    return false;
                CaseUnits(u, map, PreviousUnitLetter(src + t, map), Vector256<uint>.Zero).Store((uint*)(dst + t));
            }
            return true;
        }

        /// <summary>Whether eight units are all ASCII (one <c>vptest</c>).</summary>
        /// <param name="u">The units.</param>
        /// <returns>True when every unit is below 0x80.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsAsciiUnits(Vector256<uint> u) => Avx.TestZ(u, Vector256.Create(0xFFFF_FF80u));

        /// <summary>The title map's carry into a chunk of ASCII units: lane 0 all-ones when the unit before is a letter.</summary>
        /// <param name="p">The chunk's first byte (not the element's first byte).</param>
        /// <param name="map">The map.</param>
        /// <returns>The carry vector.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> PreviousUnitLetter(byte* p, int map)
        {
            if (map != TitleMap)
                return Vector256<uint>.Zero;
            uint c = ((uint*)p)[-1];
            return Vector256.CreateScalar(((c | 0x20) - 'a') <= 25 ? uint.MaxValue : 0u);
        }

        /// <summary>The <see cref="CaseBytes"/> map over eight ASCII units.</summary>
        /// <param name="u">The units (all ASCII).</param>
        /// <param name="map">The map.</param>
        /// <param name="carry">title: lane 0 all-ones when the unit before the chunk is a letter.</param>
        /// <param name="first">capitalize: lane 0 all-ones when the chunk starts the element.</param>
        /// <returns>The mapped units.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> CaseUnits(Vector256<uint> u, int map, Vector256<uint> carry, Vector256<uint> first)
        {
            var max = Vector256.Create(25u);
            var dl = u - Vector256.Create((uint)'a');
            var du = u - Vector256.Create((uint)'A');
            var lower = Vector256.Equals(Vector256.Min(dl, max), dl);
            var upper = Vector256.Equals(Vector256.Min(du, max), du);
            Vector256<uint> flip;
            if (map == UpperMap)
                flip = lower;
            else if (map == LowerMap)
                flip = upper;
            else if (map == SwapcaseMap)
                flip = lower | upper;
            else if (map == CapitalizeMap)
                flip = Vector256.ConditionalSelect(first, lower, upper);
            else
                flip = Vector256.ConditionalSelect(UnitsShiftedUp(lower | upper) | carry, upper, lower);
            return u ^ (flip & Vector256.Create(0x20u));
        }

        /// <summary>Eight 32-bit lanes moved up one lane (lane i receives lane i - 1, lane 0 receives zero), across the two
        ///     128-bit halves (<c>vperm2i128</c> + <c>vpalignr</c> by 12 bytes).</summary>
        /// <param name="x">The lanes.</param>
        /// <returns>The shifted lanes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> UnitsShiftedUp(Vector256<uint> x)
        {
            var b = x.AsByte();
            return Avx2.AlignRight(b, Avx2.Permute2x128(b, b, 0x08), 12).AsUInt32();
        }

        /// <summary>Eight 32-bit lanes moved up one lane with lane 7 of <paramref name="before"/> entering lane 0
        ///     (<c>vperm2i128</c> puts <paramref name="before"/>'s high half under <paramref name="x"/>'s low half, <c>vpalignr</c>
        ///     shifts by 12 bytes): <see cref="UnitsShiftedUp"/> continued from the previous chunk, at the same cost.</summary>
        /// <param name="x">The lanes.</param>
        /// <param name="before">The previous lanes (only lane 7 is read).</param>
        /// <returns>The shifted lanes: lane 0 is <paramref name="before"/>[7], lane i is <paramref name="x"/>[i - 1].</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> UnitsShiftedUpFrom(Vector256<uint> x, Vector256<uint> before)
        {
            var b = x.AsByte();
            return Avx2.AlignRight(b, Avx2.Permute2x128(b, before.AsByte(), 0x03), 12).AsUInt32();
        }

        // =============================================================================================================
        // U: the gather path (the rest of the call after its first non-ASCII element)
        // =============================================================================================================

        /// <summary>
        ///     The case map of a <c>U</c> element through the BMP table: per eight units one gather of the flags words and one
        ///     of the mapping deltas, when every unit's mapping is one code point (see the file header).
        /// </summary>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="map">The map.</param>
        /// <param name="state">The call's state (the tables).</param>
        /// <returns>False when a unit needs CPython's algorithm (past U+FFFF, a multi-code-point mapping, a final-sigma
        ///     candidate); the output may be partly written, the caller's path rewrites all of it.</returns>
        /// <remarks>
        ///     title's "previous unit cased" for a chunk's lane 0 is lane 7 of the previous chunk's cased mask, kept in a
        ///     register; it meets only the final select between the chunk's two mappings, whose gathers depend on the chunk's
        ///     own records, so no chunk waits for another's gathers. (Choosing the table per lane by the index instead, one
        ///     gather, put the carry in front of that gather and measured slower.) The overlapping last chunk does not follow
        ///     the previous one: it reads the flag of the unit before it from the table.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool CaseMapGatherUnits(byte* src, int width, byte* dst, int map, CaseState* state)
        {
            if (width <= 32)
            {
                if (!CaseUnitsGather(LoadPadded32(src, width).AsUInt32(), map, Vector256<uint>.Zero, true, state, out var r0, out _))
                    return false;
                StorePadded32(dst, width, r0.AsByte());
                return true;
            }
            if (!CaseUnitsGather(Vector256.Load((uint*)src), map, Vector256<uint>.Zero, true, state, out var r, out var cased))
                return false;
            r.Store((uint*)dst);
            int o = 32;
            for (; o + 32 <= width; o += 32)
            {
                // title: the previous chunk's cased mask is the "before" of this one (its lane 7 is the unit before).
                if (!CaseUnitsGather(Vector256.Load((uint*)(src + o)), map, cased, false, state, out r, out cased))
                    return false;
                r.Store((uint*)(dst + o));
            }
            if (o < width)
            {
                int t = width - 32;
                if (!CaseUnitsGather(Vector256.Load((uint*)(src + t)), map, PreviousUnitCased(src + t, map, state), false, state, out r, out _))
                    return false;
                r.Store((uint*)(dst + t));
            }
            return true;
        }

        /// <summary>
        ///     The title map's "before" vector for a chunk that does not follow the previous one (the overlapping last chunk):
        ///     every lane all-ones when the unit before the chunk is cased (<c>_PyUnicode_IsCased</c>; that unit passed an
        ///     earlier chunk's checks, so it is in the BMP), else zero — <see cref="CaseUnitsGather"/> reads lane 7 only.
        /// </summary>
        /// <param name="p">The chunk's first byte (not the element's first byte).</param>
        /// <param name="map">The map.</param>
        /// <param name="state">The call's state.</param>
        /// <returns>The vector (zero for every map but title).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> PreviousUnitCased(byte* p, int map, CaseState* state)
        {
            if (map != TitleMap)
                return Vector256<uint>.Zero;
            uint c = ((uint*)p)[-1];
            bool cased = c < 0x10000 && (state->Flags[c] & PyUnicodeData.CasedMask) != 0;
            return Vector256.Create(cased ? uint.MaxValue : 0u);
        }

        /// <summary>
        ///     Eight units through the tables: the flags words, then each unit plus its mapping's delta — one gather from the
        ///     side-by-side tables, each lane's table chosen by its index (upper / lower / title by the map; swapcase lowers an
        ///     uppercase unit, uppercases a lowercase one and keeps any other; capitalize titlecases unit 0 of the element and
        ///     lowers the rest) — except title, which gathers both of its mappings and selects per unit (a unit after a cased
        ///     one lowered, any other titlecased).
        /// </summary>
        /// <param name="u">The units.</param>
        /// <param name="map">The map (a constant).</param>
        /// <param name="before">title: lane 7 all-ones when the unit before the chunk is cased (the previous chunk's cased
        ///     mask, or <see cref="PreviousUnitCased"/>); the other lanes are not read.</param>
        /// <param name="first">capitalize: the chunk starts the element (a constant at each call site).</param>
        /// <param name="state">The call's state (the tables).</param>
        /// <param name="r">Receives the mapped units.</param>
        /// <param name="cased">title: receives the chunk's cased lanes (the next chunk's <paramref name="before"/>); zero for
        ///     every other map.</param>
        /// <returns>False when a unit needs CPython's algorithm.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool CaseUnitsGather(Vector256<uint> u, int map, Vector256<uint> before, bool first, CaseState* state,
                                            out Vector256<uint> r, out Vector256<uint> cased)
        {
            r = default;
            cased = default;
            // Past U+FFFF the BMP table has no entry (and past U+10FFFF CPython raises): the element's scalar loop.
            if (HasAstral(u))
                return false;
            var w = GatherFlags(u, state->Flags);
            // A multi-code-point mapping changes the element's length: the scalar loop.
            if (!Avx.TestZ(w, Vector256.Create((uint)PyUnicodeData.ExtendedCaseMask)))
                return false;
            // U+03A3 lowers by its context (lower_ucs4's final sigma): the scalar loop for every map that lowers.
            if (map != UpperMap && Vector256.EqualsAny(u, Vector256.Create(0x3A3u)))
                return false;
            var rec = Vector256.ShiftRightLogical(w, 16).AsInt32();
            // The index into [upper | lower | title]: the record plus the chosen table's offset.
            var lowerTable = Vector256.Create(state->Records);
            if (map == UpperMap)
            {
                r = u + GatherDelta(state->Deltas, rec);
            }
            else if (map == LowerMap)
            {
                r = u + GatherDelta(state->Deltas, rec + lowerTable);
            }
            else if (map == SwapcaseMap)
            {
                var upperFlag = Vector256.Create((uint)PyUnicodeData.UpperMask);
                var lowerFlag = Vector256.Create((uint)PyUnicodeData.LowerMask);
                var isUpper = Vector256.Equals(w & upperFlag, upperFlag).AsInt32();
                var isLower = Vector256.Equals(w & lowerFlag, lowerFlag).AsInt32();
                // Py_UNICODE_ISUPPER first (the lower table), then Py_UNICODE_ISLOWER (the upper table); any other unit
                // keeps itself (its delta masked to 0).
                var delta = GatherDelta(state->Deltas, rec + (isUpper & lowerTable));
                r = u + (delta & (isUpper | isLower).AsUInt32());
            }
            else if (map == CapitalizeMap)
            {
                var idx = rec + lowerTable;
                // Unit 0 of the element: the title table (lower's offset twice).
                if (first)
                    idx += Vector256.CreateScalar(state->Records);
                r = u + GatherDelta(state->Deltas, idx);
            }
            else
            {
                var casedFlag = Vector256.Create((uint)PyUnicodeData.CasedMask);
                cased = Vector256.Equals(w & casedFlag, casedFlag);
                // Each unit's "previous unit cased": the cased lanes moved up one, lane 7 of before entering lane 0.
                var prev = UnitsShiftedUpFrom(cased, before);
                // A unit after a cased one: the lower mapping; any other: the title mapping. Both gathers depend on the
                // chunk's records only, so the carry from the previous chunk meets them at the select.
                var lowered = GatherDelta(state->Deltas, rec + lowerTable);
                var titled = GatherDelta(state->Deltas, rec + lowerTable + lowerTable);
                r = u + Vector256.ConditionalSelect(prev, lowered, titled);
            }
            return true;
        }

        /// <summary>Eight mapping deltas gathered by index (one <c>vpgatherdd</c>).</summary>
        /// <param name="table">The side-by-side tables (<see cref="PyUnicodeData.CaseDeltas"/>).</param>
        /// <param name="index">Each lane's index (record plus its table's offset).</param>
        /// <returns>The deltas (as unsigned: added with wrap-around, CPython's <c>(Py_UCS4)(ch + delta)</c>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> GatherDelta(int* table, Vector256<int> index) => Avx2.GatherVector256(table, index, 4).AsUInt32();

        // =============================================================================================================
        // The scalar bodies: the bytes methods and CPython's str algorithm
        // =============================================================================================================

        /// <summary>
        ///     The <c>bytes</c> case map of an <c>S</c> element, byte by byte (bytes_methods.c: <c>_Py_bytes_upper</c>,
        ///     <c>_Py_bytes_lower</c>, <c>_Py_bytes_swapcase</c>, <c>_Py_bytes_capitalize</c>, <c>_Py_bytes_title</c>; the
        ///     trailing NULs map to themselves).
        /// </summary>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="map">The map.</param>
        internal static void CaseMapBytesScalar(byte* src, long width, byte* dst, int map)
        {
            bool previousIsCased = false;
            for (long i = 0; i < width; i++)
            {
                int c = src[i];
                bool isLower = (uint)(c - 'a') <= 25, isUpper = (uint)(c - 'A') <= 25;
                switch (map)
                {
                    case UpperMap:
                        if (isLower)
                            c -= 0x20;
                        break;
                    case LowerMap:
                        if (isUpper)
                            c += 0x20;
                        break;
                    case SwapcaseMap:
                        if (isLower)
                            c -= 0x20;
                        else if (isUpper)
                            c += 0x20;
                        break;
                    case CapitalizeMap:
                        if (i == 0 ? isLower : isUpper)
                            c ^= 0x20;
                        break;
                    default:
                        // _Py_bytes_title: a lowercase letter is uppercased unless a letter precedes it, an uppercase one
                        // lowered when one does; anything else resets the state.
                        if (isLower)
                        {
                            if (!previousIsCased)
                                c -= 0x20;
                            previousIsCased = true;
                        }
                        else if (isUpper)
                        {
                            if (previousIsCased)
                                c += 0x20;
                            previousIsCased = true;
                        }
                        else
                            previousIsCased = false;
                        break;
                }
                dst[i] = (byte)c;
            }
        }

        /// <summary>
        ///     The scalar case map of a <c>U</c> element (kept out of the kernel's hot loop): CPython's algorithm over the
        ///     element's code points, the result cut to the width and zero-padded; a unit past U+10FFFF records CPython's
        ///     SystemError and leaves the output zero.
        /// </summary>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="map">The map.</param>
        /// <param name="state">The call's state (its status).</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void CaseMapUnitsCold(byte* src, int width, byte* dst, int map, CaseState* state)
            => CaseMapUnitsScalar((uint*)src, width / 4, (uint*)dst, map, state);

        /// <summary>
        ///     CPython's <c>str</c> case map of a <c>U</c> element: its code points without the trailing NULs (the
        ///     <c>np.str_</c> scalar), mapped by <see cref="CaseMapCodePoints"/> straight into the output, cut at
        ///     <paramref name="units"/> and zero-padded.
        /// </summary>
        /// <param name="src">The element's units.</param>
        /// <param name="units">Its width in units.</param>
        /// <param name="dst">The output element's units (not overlapping <paramref name="src"/>).</param>
        /// <param name="map">The map.</param>
        /// <param name="state">The call's state: a unit past U+10FFFF sets <see cref="CaseState.Status"/>.</param>
        internal static void CaseMapUnitsScalar(uint* src, long units, uint* dst, int map, CaseState* state)
        {
            long n = units;
            while (n > 0 && src[n - 1] == 0)
                n--;
            for (long i = 0; i < n; i++)
            {
                if (src[i] > 0x10FFFF)
                {
                    // CPython raises before any result exists; the driver raises once the loop ends.
                    state->Status = CaseStatusInvalidMaxChar;
                    new Span<byte>(dst, checked((int)(units * 4))).Clear();
                    return;
                }
            }
            uint* mapped = stackalloc uint[PyUnicodeData.MaxMapping];
            long k = CaseMapCodePoints(src, n, map, dst, units, mapped);
            for (; k < units; k++)
                dst[k] = 0;
        }

        /// <summary>
        ///     CPython 3.12's <c>do_upper</c> / <c>do_lower</c> / <c>do_swapcase</c> / <c>do_capitalize</c> / <c>do_title</c>
        ///     over code points, writing at most <paramref name="cap"/> of the mapped code points (the rest is cut, as
        ///     NumPy's setitem cuts the result to the element width).
        /// </summary>
        /// <param name="s">The text's code points (every one at most U+10FFFF).</param>
        /// <param name="n">Their number.</param>
        /// <param name="map">The map.</param>
        /// <param name="dst">The output (not overlapping <paramref name="s"/>).</param>
        /// <param name="cap">The output's capacity in code points (<c>3 * n</c> holds every result).</param>
        /// <param name="mapped">Scratch for one character's mapping (<see cref="PyUnicodeData.MaxMapping"/> units).</param>
        /// <returns>The code points written.</returns>
        /// <remarks>
        ///     <c>capitalize</c> of an empty text is empty (CPython returns the string unchanged before reading character 0).
        ///     A BMP character without <c>EXTENDED_CASE</c> other than U+03A3 — nearly every character of real text — takes
        ///     the first branch: its flags word read once, its mapping one delta from the side-by-side tables, the choice
        ///     between them made by index arithmetic rather than a branch (swapcase's upper / lower / neither, title's
        ///     previous-cased state are data-dependent and mispredicted as branches), and written straight to the output
        ///     (no per-character copy loop). Any other character runs CPython's statements below; both give the same code
        ///     points (the delta of a record without <c>EXTENDED_CASE</c> is exactly what <c>_PyUnicode_To*Full</c> adds).
        /// </remarks>
        internal static long CaseMapCodePoints(uint* s, long n, int map, uint* dst, long cap, uint* mapped)
        {
            long k = 0;
            bool previousIsCased = false;
            uint* bmp = PyUnicodeData.BmpTable;
            int* deltas = PyUnicodeData.CaseDeltas;
            const int stride = PyUnicodeData.CaseDeltaStride;
            for (long i = 0; i < n && k < cap; i++)
            {
                uint c = s[i];
                if (c < 0x10000 && c != 0x3A3)
                {
                    uint w = bmp[c];
                    if ((w & PyUnicodeData.ExtendedCaseMask) == 0)
                    {
                        int index = (int)(w >> 16);
                        int keep = -1;
                        switch (map)
                        {
                            case UpperMap:
                                break;
                            case LowerMap:
                                index += stride;
                                break;
                            case SwapcaseMap:
                                // Py_UNICODE_ISUPPER: the lower table; else Py_UNICODE_ISLOWER: the upper table; else the
                                // character itself (its delta masked to 0).
                                int isUpper = (w & PyUnicodeData.UpperMask) != 0 ? 1 : 0;
                                int isLower = (w & PyUnicodeData.LowerMask) != 0 ? 1 : 0;
                                index += isUpper * stride;
                                keep = -(isUpper | isLower);
                                break;
                            case CapitalizeMap:
                                index += i == 0 ? 2 * stride : stride;
                                break;
                            default:
                                // A character after a cased one: the lower table; any other: the title table. do_title reads
                                // the state from the INPUT character.
                                index += previousIsCased ? stride : 2 * stride;
                                previousIsCased = (w & PyUnicodeData.CasedMask) != 0;
                                break;
                        }
                        dst[k++] = unchecked(c + (uint)(deltas[index] & keep));
                        continue;
                    }
                }
                int m;
                switch (map)
                {
                    case UpperMap:
                        m = PyUnicodeData.ToUpperFull(c, mapped);
                        break;
                    case LowerMap:
                        m = LowerUcs4(s, n, i, c, mapped);
                        break;
                    case SwapcaseMap:
                        int f = PyUnicodeData.Flags(c);
                        if ((f & PyUnicodeData.UpperMask) != 0)
                            m = LowerUcs4(s, n, i, c, mapped);
                        else if ((f & PyUnicodeData.LowerMask) != 0)
                            m = PyUnicodeData.ToUpperFull(c, mapped);
                        else
                        {
                            mapped[0] = c;
                            m = 1;
                        }
                        break;
                    case CapitalizeMap:
                        m = i == 0 ? PyUnicodeData.ToTitleFull(c, mapped) : LowerUcs4(s, n, i, c, mapped);
                        break;
                    default:
                        m = previousIsCased ? LowerUcs4(s, n, i, c, mapped) : PyUnicodeData.ToTitleFull(c, mapped);
                        // do_title reads the state from the INPUT character.
                        previousIsCased = PyUnicodeData.IsCased(c);
                        break;
                }
                for (int j = 0; j < m && k < cap; j++)
                    dst[k++] = mapped[j];
            }
            return k;
        }

        /// <summary>CPython's <c>lower_ucs4</c>: U+03A3 lowers to the final or the medial sigma by its context, any other
        ///     character to its full lowercase mapping.</summary>
        /// <param name="s">The text.</param>
        /// <param name="n">Its length.</param>
        /// <param name="i">The character's index.</param>
        /// <param name="c">The character.</param>
        /// <param name="mapped">Receives the mapping.</param>
        /// <returns>The mapping's length.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int LowerUcs4(uint* s, long n, long i, uint c, uint* mapped)
        {
            if (c == 0x3A3)
            {
                mapped[0] = HandleCapitalSigma(s, n, i);
                return 1;
            }
            return PyUnicodeData.ToLowerFull(c, mapped);
        }

        /// <summary>
        ///     CPython's <c>handle_capital_sigma</c>: U+03A3 is in the Final_Sigma context when a cased character precedes it
        ///     (case-ignorable characters skipped) and no cased character follows it (case-ignorable ones skipped).
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="n">Its length.</param>
        /// <param name="i">The sigma's index.</param>
        /// <returns>U+03C2 (final) or U+03C3.</returns>
        private static uint HandleCapitalSigma(uint* s, long n, long i)
        {
            long j;
            uint c = 0;
            for (j = i - 1; j >= 0; j--)
            {
                c = s[j];
                if (!PyUnicodeData.IsCaseIgnorable(c))
                    break;
            }
            bool finalSigma = j >= 0 && PyUnicodeData.IsCased(c);
            if (finalSigma)
            {
                for (j = i + 1; j < n; j++)
                {
                    c = s[j];
                    if (!PyUnicodeData.IsCaseIgnorable(c))
                        break;
                }
                finalSigma = j == n || !PyUnicodeData.IsCased(c);
            }
            return finalSigma ? 0x3C2u : 0x3C3u;
        }
    }
}
