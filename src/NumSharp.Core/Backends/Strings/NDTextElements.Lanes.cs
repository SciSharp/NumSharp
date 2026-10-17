using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// NDTextElements.Lanes.cs — the width-class helpers of the S / U loops (strings phase 2, P2.5; strings plan D11 / D13)
// =============================================================================
//
// WHAT THIS IS
//   The element-at-once SIMD primitives the S / U bodies of NDTextElements use when an element is small: LANE MASKS (bit i
//   of a ulong says something about unit i — equal to a code, whitespace, nonzero) for elements of up to 64 units, and
//   LOADS / SELECTS / STORES of whole elements of up to 64 bytes held in two Vector256 registers. A body built on them
//   computes its answer with a fixed sequence of vector operations and no branch on the data: the scalar bodies' unit
//   loops and BCL span searches mispredict on every element of random text (the find family cost 20-60 ns per element
//   that way, about NumPy's own cost).
//
// WIDTH CLASSES
//   The IL shell (ILKernelGenerator.Text.Methods.cs) calls each body with the element widths as constants, so every test
//   of a width below folds and one width compiles to the loads it needs: whole 32-byte chunks and one overlapping last
//   chunk; two overlapping 16-byte halves; two overlapping 8- or 4-byte reads moved into a vector; single bytes. No helper
//   reads or writes a byte outside [p, p + width): a view's last element can end at its buffer's last byte (the
//   FencedMemory tests pin this). The managed fallback kernels call the same helpers with run-time widths: the same
//   results, through the JIT's fallback for a non-constant shuffle immediate.
//
// PORTABILITY (strings plan D21)
//   Every fast path runs only where Avx2.IsSupported (a JIT constant); elsewhere — another architecture, or AVX2 /
//   hardware intrinsics switched off — the scalar bodies run. Both give the same results; the unit tests compare them on
//   every width class.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static unsafe partial class NDTextElements
    {
        /// <summary>The widest element (bytes) the load / select / store helpers hold: two <see cref="Vector256{T}"/>.</summary>
        internal const int LaneBytes = 64;

        /// <summary>The most units a lane mask describes (one bit each in a <see cref="ulong"/>).</summary>
        internal const int LaneUnits = 64;

        /// <summary>The lane-mask mode of <see cref="LaneMask{TUnit}"/>: unit == the code.</summary>
        private const int MaskEqual = 0;

        /// <summary>The lane-mask mode of <see cref="LaneMask{TUnit}"/>: unit is whitespace of the element's class
        ///     (<see cref="IsSpace{TUnit}"/>).</summary>
        private const int MaskSpace = 1;

        /// <summary>
        ///     Whether the lane masks can describe an element of <paramref name="width"/> bytes on this machine: AVX2, and at
        ///     most <see cref="LaneUnits"/> units of at least one unit.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="width">The element width in bytes (a JIT constant in the IL shell, so the test folds).</param>
        /// <returns>True when the mask helpers apply.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool MasksFit<TUnit>(int width) where TUnit : unmanaged
            => Avx2.IsSupported && width >= sizeof(TUnit) && width <= LaneUnits * sizeof(TUnit);

        /// <summary>
        ///     Whether the load / select / store helpers can hold an element of <paramref name="width"/> bytes: AVX2, and
        ///     1 to <see cref="LaneBytes"/> bytes.
        /// </summary>
        /// <param name="width">The element width in bytes.</param>
        /// <returns>True when the select helpers apply.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool SelectsFit(int width) => Avx2.IsSupported && width >= 1 && width <= LaneBytes;

        /// <summary>The lanes below <paramref name="n"/>: bits <c>[0, n)</c> (all 64 for n ≥ 64, none for n = 0).</summary>
        /// <param name="n">The lane count, 0 to 255 (every caller's count: a length, an index within one, or an index plus a
        ///     needle length — each at most 128).</param>
        /// <returns>The mask.</returns>
        /// <remarks>BMI2's <c>bzhi</c> where the host has it (every AVX2 host the lane paths run on): one instruction with
        ///     no branch — it keeps the low n bits and keeps all of them from n = 64 on.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ulong LowLanes(long n)
        {
            if (Bmi2.X64.IsSupported)
                return Bmi2.X64.ZeroHighBits(ulong.MaxValue, (ulong)n);
            return n >= 64 ? ulong.MaxValue : n <= 0 ? 0UL : (1UL << (int)n) - 1;
        }

        /// <summary>The lanes of the inclusive range <c>[lo, hi]</c> (both within <c>[0, 63]</c>, <c>lo ≤ hi + 1</c>).</summary>
        /// <param name="lo">The first lane.</param>
        /// <param name="hi">The last lane.</param>
        /// <returns>The mask (empty when <c>hi &lt; lo</c>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong RangeLanes(long lo, long hi) => LowLanes(hi + 1) & ~LowLanes(lo);

        // =============================================================================================================
        // Lane masks
        // =============================================================================================================

        /// <summary>The lanes whose unit equals <paramref name="code"/> (see <see cref="LaneMask{TUnit}"/>).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="p">The element's first byte.</param>
        /// <param name="width">The element width in bytes (<see cref="MasksFit{TUnit}"/> holds).</param>
        /// <param name="code">The code value (truncated to a byte for <c>S</c>).</param>
        /// <returns>Bit i set when unit i equals <paramref name="code"/>; no bit at or past the element's unit count.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ulong EqualLanes<TUnit>(byte* p, int width, uint code) where TUnit : unmanaged
            => LaneMask<TUnit>(p, width, MaskEqual, code);

        /// <summary>The lanes whose unit is whitespace of the element's class (<see cref="IsSpace{TUnit}"/>).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="p">The element's first byte.</param>
        /// <param name="width">The element width in bytes (<see cref="MasksFit{TUnit}"/> holds).</param>
        /// <returns>Bit i set when unit i is whitespace; no bit at or past the element's unit count.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ulong SpaceLanes<TUnit>(byte* p, int width) where TUnit : unmanaged
            => LaneMask<TUnit>(p, width, MaskSpace, 0);

        /// <summary>
        ///     NumPy's <c>num_codepoints</c> through the lane masks: one past the last nonzero unit, from all the element's
        ///     chunks at once (no scan from the end that stops on the data).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="p">The element's first byte.</param>
        /// <param name="width">The element width in bytes (<see cref="MasksFit{TUnit}"/> holds).</param>
        /// <returns>The length in units.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int UnitsByLanes<TUnit>(byte* p, int width) where TUnit : unmanaged
        {
            ulong nonzero = ~EqualLanes<TUnit>(p, width, 0) & LowLanes(width / sizeof(TUnit));
            return 64 - BitOperations.LeadingZeroCount(nonzero);
        }

        /// <summary>
        ///     One lane mask of an element: the predicate <paramref name="mode"/> (<see cref="MaskEqual"/> with
        ///     <paramref name="code"/>, or <see cref="MaskSpace"/>) evaluated on every unit, one bit per unit.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="p">The element's first byte.</param>
        /// <param name="width">The element width in bytes (<see cref="MasksFit{TUnit}"/> holds).</param>
        /// <param name="mode">The predicate (a constant at every call site, so the predicate's test folds).</param>
        /// <param name="code">The code for <see cref="MaskEqual"/>.</param>
        /// <returns>The mask; no bit at or past the element's unit count.</returns>
        /// <remarks>
        ///     Whole 32-byte chunks, then one more 32-byte chunk ending at the element's end (its lanes overlap the last
        ///     whole chunk's, which hold the same bits); below 32 bytes the small forms of <see cref="LaneMaskSmall{TUnit}"/>.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong LaneMask<TUnit>(byte* p, int width, int mode, uint code) where TUnit : unmanaged
        {
            if (width < 32)
                return LaneMaskSmall<TUnit>(p, width, mode, code);
            // The chunks are written out (at most 8: 256 bytes of U, 64 of S) rather than looped: with a constant width
            // every test below folds, leaving straight-line loads, compares and shifts by constants.
            int shiftPerByte = sizeof(TUnit) == 1 ? 0 : 2;
            ulong m = Predicate<TUnit>(Vector256.Load(p), mode, code);
            if (width >= 64)
                m |= (ulong)Predicate<TUnit>(Vector256.Load(p + 32), mode, code) << (32 >> shiftPerByte);
            if (width >= 96)
                m |= (ulong)Predicate<TUnit>(Vector256.Load(p + 64), mode, code) << (64 >> shiftPerByte);
            if (width >= 128)
                m |= (ulong)Predicate<TUnit>(Vector256.Load(p + 96), mode, code) << (96 >> shiftPerByte);
            if (width >= 160)
                m |= (ulong)Predicate<TUnit>(Vector256.Load(p + 128), mode, code) << (128 >> shiftPerByte);
            if (width >= 192)
                m |= (ulong)Predicate<TUnit>(Vector256.Load(p + 160), mode, code) << (160 >> shiftPerByte);
            if (width >= 224)
                m |= (ulong)Predicate<TUnit>(Vector256.Load(p + 192), mode, code) << (192 >> shiftPerByte);
            if (width >= 256)
                m |= (ulong)Predicate<TUnit>(Vector256.Load(p + 224), mode, code) << (224 >> shiftPerByte);
            if ((width & 31) != 0)
                m |= (ulong)Predicate<TUnit>(Vector256.Load(p + width - 32), mode, code) << ((width - 32) >> shiftPerByte);
            return m;
        }

        /// <summary>
        ///     <see cref="LaneMask{TUnit}"/> for an element below 32 bytes: two overlapping 16-byte halves (16 ≤ width), two
        ///     overlapping 8-byte reads (8 ≤ width), the element's bytes assembled into one 8-byte value (below 8), each moved
        ///     into a vector and tested, the lanes past the element discarded.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="p">The element's first byte.</param>
        /// <param name="width">The element width in bytes (1 to 31, a whole number of units).</param>
        /// <param name="mode">The predicate.</param>
        /// <param name="code">The code for <see cref="MaskEqual"/>.</param>
        /// <returns>The mask; no bit at or past the element's unit count.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong LaneMaskSmall<TUnit>(byte* p, int width, int mode, uint code) where TUnit : unmanaged
        {
            int lanesPer16 = 16 / sizeof(TUnit);
            int units = width / sizeof(TUnit);
            if (width >= 16)
            {
                ulong lo = Predicate<TUnit>(Vector128.Load(p), mode, code);
                ulong hi = Predicate<TUnit>(Vector128.Load(p + width - 16), mode, code);
                // The second half's lanes are units [units - lanesPer16, units).
                return lo | hi << (units - lanesPer16);
            }
            if (width >= 8)
            {
                int lanesPer8 = 8 / sizeof(TUnit);
                ulong lo = Predicate<TUnit>(Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<ulong>(p)).AsByte(), mode, code)
                           & LowLanes(lanesPer8);
                ulong hi = Predicate<TUnit>(Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<ulong>(p + width - 8)).AsByte(), mode, code)
                           & LowLanes(lanesPer8);
                return lo | hi << (units - lanesPer8);
            }
            // Below 8 bytes: the bytes in the low end of one 8-byte value (zero above), the lanes past the element dropped.
            ulong v = ReadSmall(p, width);
            return Predicate<TUnit>(Vector128.CreateScalarUnsafe(v).AsByte(), mode, code) & LowLanes(units);
        }

        /// <summary>
        ///     The bytes <c>[0, width)</c> of an element below 8 bytes in the low end of an 8-byte value, zero above, read as
        ///     at most two overlapping 4-byte loads (4 ≤ width) or single bytes — never past the element.
        /// </summary>
        /// <param name="p">The element's first byte.</param>
        /// <param name="width">1 to 7.</param>
        /// <returns>The bytes, little-endian.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong ReadSmall(byte* p, int width)
        {
            if (width >= 4)
            {
                ulong lo = Unsafe.ReadUnaligned<uint>(p);
                // The second read holds bytes [width - 4, width); shifted down it holds bytes [4, width) at their place.
                ulong hi = Unsafe.ReadUnaligned<uint>(p + width - 4) >> (8 * (8 - width));
                return lo | hi << 32;
            }
            ulong v = p[0];
            if (width > 1)
                v |= (ulong)p[1] << 8;
            if (width > 2)
                v |= (ulong)p[2] << 16;
            return v;
        }

        /// <summary>The predicate of a lane mask over one 32-byte chunk, as a bit per unit.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="v">The chunk.</param>
        /// <param name="mode">The predicate.</param>
        /// <param name="code">The code for <see cref="MaskEqual"/>.</param>
        /// <returns>32 bits (<c>S</c>) or 8 bits (<c>U</c>).</returns>
        /// <remarks>
        ///     <see cref="MaskSpace"/> is NumPy's whitespace test without a branch: <c>S</c> — <c>NumPyOS_ascii_isspace</c>
        ///     (space and <c>\t</c>–<c>\r</c>); <c>U</c> — <c>Py_UNICODE_ISSPACE</c>, the ASCII table plus U+001C–U+001F and
        ///     the nine non-ASCII spaces of <see cref="NDCharArray.IsPyWhitespace"/>, each a compare on every lane.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Predicate<TUnit>(Vector256<byte> v, int mode, uint code) where TUnit : unmanaged
        {
            if (typeof(TUnit) == typeof(byte))
            {
                if (mode == MaskEqual)
                    return Vector256.Equals(v, Vector256.Create((byte)code)).ExtractMostSignificantBits();
                var ws = Vector256.Equals(v, Vector256.Create((byte)0x20))
                         | Vector256.LessThanOrEqual(v - Vector256.Create((byte)0x09), Vector256.Create((byte)4));
                return ws.ExtractMostSignificantBits();
            }
            var u = v.AsUInt32();
            if (mode == MaskEqual)
                return Vector256.Equals(u, Vector256.Create(code)).ExtractMostSignificantBits();
            return SpaceU(u).ExtractMostSignificantBits();
        }

        /// <summary><see cref="Predicate{TUnit}(Vector256{byte}, int, uint)"/> over one 16-byte chunk.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="v">The chunk.</param>
        /// <param name="mode">The predicate.</param>
        /// <param name="code">The code for <see cref="MaskEqual"/>.</param>
        /// <returns>16 bits (<c>S</c>) or 4 bits (<c>U</c>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Predicate<TUnit>(Vector128<byte> v, int mode, uint code) where TUnit : unmanaged
        {
            if (typeof(TUnit) == typeof(byte))
            {
                if (mode == MaskEqual)
                    return Vector128.Equals(v, Vector128.Create((byte)code)).ExtractMostSignificantBits();
                var ws = Vector128.Equals(v, Vector128.Create((byte)0x20))
                         | Vector128.LessThanOrEqual(v - Vector128.Create((byte)0x09), Vector128.Create((byte)4));
                return ws.ExtractMostSignificantBits();
            }
            var u = v.AsUInt32();
            if (mode == MaskEqual)
                return Vector128.Equals(u, Vector128.Create(code)).ExtractMostSignificantBits();
            return SpaceU(u.ToVector256()).ExtractMostSignificantBits() & 0xFu;
        }

        /// <summary><c>Py_UNICODE_ISSPACE</c> on every lane of a <c>U</c> chunk (all-ones lanes for whitespace).</summary>
        /// <param name="u">The units.</param>
        /// <returns>The whitespace lanes.</returns>
        /// <remarks>The ASCII part is three tests; the nine non-ASCII spaces (all at U+0085 or above) are tested only when
        ///     some lane leaves ASCII — one branch per chunk that ASCII text always predicts.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> SpaceU(Vector256<uint> u)
        {
            var ws = Vector256.Equals(u, Vector256.Create(0x20u))
                     | Vector256.LessThanOrEqual(u - Vector256.Create(0x09u), Vector256.Create(4u))
                     | Vector256.LessThanOrEqual(u - Vector256.Create(0x1Cu), Vector256.Create(3u));
            if (!Vector256.EqualsAll(u & Vector256.Create(0xFFFFFF80u), Vector256<uint>.Zero))
                ws |= Vector256.Equals(u, Vector256.Create(0x85u))
                      | Vector256.Equals(u, Vector256.Create(0xA0u))
                      | Vector256.Equals(u, Vector256.Create(0x1680u))
                      | Vector256.LessThanOrEqual(u - Vector256.Create(0x2000u), Vector256.Create(0xAu))
                      | Vector256.LessThanOrEqual(u - Vector256.Create(0x2028u), Vector256.Create(1u))
                      | Vector256.Equals(u, Vector256.Create(0x202Fu))
                      | Vector256.Equals(u, Vector256.Create(0x205Fu))
                      | Vector256.Equals(u, Vector256.Create(0x3000u));
            return ws;
        }

        /// <summary>
        ///     The lanes where the first <paramref name="len2"/> units of <paramref name="sub"/> occur in the element: bit p
        ///     set when units <c>[p, p + len2)</c> equal them (the AND of each needle unit's equality mask shifted down by its
        ///     index — fastsearch's matches, all at once).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="hay">The searched element.</param>
        /// <param name="hayWidth">Its width in bytes (<see cref="MasksFit{TUnit}"/> holds).</param>
        /// <param name="sub">The needle element.</param>
        /// <param name="len2">The needle's length, 1 to the haystack's unit count.</param>
        /// <returns>The match lanes (a match never reaches past the element: its last unit is nonzero, and the units past
        ///     the element's length are NULs).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong MatchLanes<TUnit>(byte* hay, int hayWidth, byte* sub, int len2) where TUnit : unmanaged
        {
            ulong cand = EqualLanes<TUnit>(hay, hayWidth, UnitAt<TUnit>(sub, 0));
            // The loop ends early once no candidate is left (a short needle in random text usually ends after one unit).
            for (int j = 1; j < len2 && cand != 0; j++)
                cand &= EqualLanes<TUnit>(hay, hayWidth, UnitAt<TUnit>(sub, j)) >> j;
            return cand;
        }

        // =============================================================================================================
        // Loads, selects and stores of whole elements (up to 64 bytes in two Vector256)
        // =============================================================================================================

        /// <summary>
        ///     An element of up to 64 bytes in two vectors, zero past its width, read without touching a byte outside it.
        /// </summary>
        /// <param name="p">The element's first byte.</param>
        /// <param name="width">The width in bytes (<see cref="SelectsFit"/> holds).</param>
        /// <param name="lo">Bytes 0–31.</param>
        /// <param name="hi">Bytes 32–63.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void LoadElement(byte* p, int width, out Vector256<byte> lo, out Vector256<byte> hi)
        {
            if (width > 32)
            {
                lo = Vector256.Load(p);
                hi = LoadPadded32(p + 32, width - 32);
            }
            else
            {
                lo = LoadPadded32(p, width);
                hi = Vector256<byte>.Zero;
            }
        }

        /// <summary>The bytes <c>[0, width)</c> of up to 32 bytes in one vector, zero above, read within them.</summary>
        /// <param name="p">The first byte.</param>
        /// <param name="width">1 to 32.</param>
        /// <returns>The vector.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> LoadPadded32(byte* p, int width)
        {
            if (width == 32)
                return Vector256.Load(p);
            if (width > 16)
            {
                var lo = Vector128.Load(p);
                // The second half's read ends at the element's end; shifting it down by (32 - width) bytes puts bytes
                // [16, width) at their place and zeros above.
                var hi = Sse2.ShiftRightLogical128BitLane(Vector128.Load(p + width - 16), (byte)(32 - width));
                return Vector256.Create(lo, hi);
            }
            return LoadPadded16(p, width).ToVector256();
        }

        /// <summary>The bytes <c>[0, width)</c> of up to 16 bytes in one vector, zero above, read within them.</summary>
        /// <param name="p">The first byte.</param>
        /// <param name="width">1 to 16.</param>
        /// <returns>The vector.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<byte> LoadPadded16(byte* p, int width)
        {
            if (width == 16)
                return Vector128.Load(p);
            if (width > 8)
            {
                ulong lo = Unsafe.ReadUnaligned<ulong>(p);
                ulong hi = Unsafe.ReadUnaligned<ulong>(p + width - 8) >> (8 * (16 - width));
                return Vector128.Create(lo, hi).AsByte();
            }
            ulong v = width == 8 ? Unsafe.ReadUnaligned<ulong>(p) : ReadSmall(p, width);
            return Vector128.CreateScalar(v).AsByte();
        }

        /// <summary>Stores the bytes <c>[0, width)</c> of two vectors to an element, writing no byte past it.</summary>
        /// <param name="dst">The element's first byte.</param>
        /// <param name="width">Its width (<see cref="SelectsFit"/> holds).</param>
        /// <param name="lo">Bytes 0–31.</param>
        /// <param name="hi">Bytes 32–63.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void StoreElement(byte* dst, int width, Vector256<byte> lo, Vector256<byte> hi)
        {
            if (width > 32)
            {
                lo.Store(dst);
                StorePadded32(dst + 32, width - 32, hi);
            }
            else
                StorePadded32(dst, width, lo);
        }

        /// <summary>Stores the bytes <c>[0, width)</c> of one vector, writing no byte past them.</summary>
        /// <param name="dst">The first byte.</param>
        /// <param name="width">1 to 32.</param>
        /// <param name="v">The bytes.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void StorePadded32(byte* dst, int width, Vector256<byte> v)
        {
            if (width == 32)
            {
                v.Store(dst);
                return;
            }
            var lo = v.GetLower();
            if (width > 16)
            {
                lo.Store(dst);
                // Bytes [width - 16, width) of the 32: the lower half's tail followed by the upper half's head.
                Ssse3.AlignRight(v.GetUpper(), lo, (byte)(width - 16)).Store(dst + width - 16);
                return;
            }
            StorePadded16(dst, width, lo);
        }

        /// <summary>Stores the bytes <c>[0, width)</c> of one 16-byte vector, writing no byte past them.</summary>
        /// <param name="dst">The first byte.</param>
        /// <param name="width">1 to 16.</param>
        /// <param name="v">The bytes.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void StorePadded16(byte* dst, int width, Vector128<byte> v)
        {
            if (width == 16)
            {
                v.Store(dst);
                return;
            }
            ulong lo = v.AsUInt64().ToScalar();
            if (width > 8)
            {
                ulong hi = v.AsUInt64().GetElement(1);
                Unsafe.WriteUnaligned(dst, lo);
                // Bytes [width - 8, width): the low word's tail and the high word's head.
                Unsafe.WriteUnaligned(dst + width - 8, (lo >> (8 * (width - 8))) | (hi << (8 * (16 - width))));
                return;
            }
            if (width == 8)
            {
                Unsafe.WriteUnaligned(dst, lo);
                return;
            }
            if (width >= 4)
            {
                Unsafe.WriteUnaligned(dst, (uint)lo);
                Unsafe.WriteUnaligned(dst + width - 4, (uint)(lo >> (8 * (width - 4))));
                return;
            }
            dst[0] = (byte)lo;
            if (width > 1)
                dst[1] = (byte)(lo >> 8);
            if (width > 2)
                dst[2] = (byte)(lo >> 16);
        }

        /// <summary>
        ///     Gathers units of an element held in two vectors: output unit i is source unit <c>first + i·step</c> for
        ///     <c>i &lt; count</c>, and 0 for every later lane.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="lo">Source bytes 0–31.</param>
        /// <param name="hi">Source bytes 32–63.</param>
        /// <param name="srcWidth">The source width in bytes (a constant: the gathers from bytes past it are skipped).</param>
        /// <param name="outWidth">The output width in bytes (a constant: the second output vector is computed only past 32).</param>
        /// <param name="first">The source index of output unit 0.</param>
        /// <param name="step">The source step per output unit (any sign; 1 takes the shift without a multiply).</param>
        /// <param name="count">The number of output units, 0 to the output's units; every <c>first + i·step</c> below it is a
        ///     source unit index.</param>
        /// <param name="rlo">Output bytes 0–31.</param>
        /// <param name="rhi">Output bytes 32–63.</param>
        /// <remarks>
        ///     <c>U</c>: <c>vpermd</c> by the index lanes (and a blend between the two source vectors by bit 3 of the index).
        ///     <c>S</c>: <c>vpshufb</c> of each 16-byte quarter, broadcast to both halves, by the index minus the quarter's base
        ///     plus 0x70 with unsigned saturation — an index in the quarter keeps its low nibble with bit 7 clear, any other
        ///     sets bit 7, which makes <c>vpshufb</c> write 0, so the quarters' results OR together.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void SelectUnits<TUnit>(Vector256<byte> lo, Vector256<byte> hi, int srcWidth, int outWidth,
                                                int first, int step, int count, out Vector256<byte> rlo, out Vector256<byte> rhi)
            where TUnit : unmanaged
        {
            if (typeof(TUnit) == typeof(uint))
            {
                var iota = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7);
                var s0 = step == 1 ? Vector256.Create(first) + iota : Vector256.Create(first) + iota * step;
                rlo = (GatherDwords(lo, hi, srcWidth, s0) & Vector256.LessThan(iota, Vector256.Create(count))).AsByte();
                if (outWidth > 32)
                {
                    var s1 = s0 + Vector256.Create(8 * step);
                    rhi = (GatherDwords(lo, hi, srcWidth, s1) & Vector256.LessThan(iota + Vector256.Create(8), Vector256.Create(count)))
                        .AsByte();
                }
                else
                    rhi = Vector256<byte>.Zero;
                return;
            }
            var iotaB = Vector256.Create((byte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            var idx0 = step == 1 ? Vector256.Create((byte)first) + iotaB : SteppedBytes(first, step, 0);
            rlo = GatherBytes(lo, hi, srcWidth, idx0) & Vector256.GreaterThan(Vector256.Create((sbyte)count), iotaB.AsSByte()).AsByte();
            if (outWidth > 32)
            {
                var idx1 = step == 1 ? idx0 + Vector256.Create((byte)32) : SteppedBytes(first, step, 32);
                rhi = GatherBytes(lo, hi, srcWidth, idx1)
                      & Vector256.GreaterThan(Vector256.Create((sbyte)count), (iotaB + Vector256.Create((byte)32)).AsSByte()).AsByte();
            }
            else
                rhi = Vector256<byte>.Zero;
        }

        /// <summary>The byte indices <c>first + (base + i)·step</c> (mod 256) of 32 output lanes, through 16-bit products.</summary>
        /// <param name="first">The source index of output unit 0.</param>
        /// <param name="step">The step.</param>
        /// <param name="lane0">The output lane of the vector's first byte (0 or 32).</param>
        /// <returns>The indices; a lane whose true index lies in [0, 64) holds it exactly.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> SteppedBytes(int first, int step, int lane0)
        {
            var iota16 = Vector256.Create((short)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            var f = Vector256.Create((short)(first + lane0 * step));
            var s = Vector256.Create((short)step);
            var a = (f + iota16 * s) & Vector256.Create((short)0xFF);
            var b = (f + (iota16 + Vector256.Create((short)16)) * s) & Vector256.Create((short)0xFF);
            // packuswb interleaves the 128-bit halves; vpermq 0xD8 restores lane order.
            return Avx2.Permute4x64(Avx2.PackUnsignedSaturate(a, b).AsUInt64(), 0xD8).AsByte();
        }

        /// <summary>One gathered vector of <c>U</c> units: source unit <c>s[i]</c> (0–15) at lane i.</summary>
        /// <param name="lo">Source units 0–7.</param>
        /// <param name="hi">Source units 8–15.</param>
        /// <param name="srcWidth">The source width (past 32 bytes the second source vector takes part).</param>
        /// <param name="s">The source indices.</param>
        /// <returns>The gathered lanes (a lane whose index is outside 0–15 holds some unit: callers mask it).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<int> GatherDwords(Vector256<byte> lo, Vector256<byte> hi, int srcWidth, Vector256<int> s)
        {
            var a = Avx2.PermuteVar8x32(lo.AsInt32(), s);
            if (srcWidth <= 32)
                return a;
            var b = Avx2.PermuteVar8x32(hi.AsInt32(), s);
            // vpermd reads the index's low 3 bits; bit 3 picks the source vector (moved to the sign bit for the blend).
            return Avx.BlendVariable(a.AsSingle(), b.AsSingle(), Avx2.ShiftLeftLogical(s, 28).AsSingle()).AsInt32();
        }

        /// <summary>One gathered vector of <c>S</c> units: source byte <c>idx[i]</c> (0–63) at lane i, 0 for any other index.</summary>
        /// <param name="lo">Source bytes 0–31.</param>
        /// <param name="hi">Source bytes 32–63.</param>
        /// <param name="srcWidth">The source width (quarters past it are skipped: their bytes are zero).</param>
        /// <param name="idx">The source indices.</param>
        /// <returns>The gathered bytes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> GatherBytes(Vector256<byte> lo, Vector256<byte> hi, int srcWidth, Vector256<byte> idx)
        {
            var bias = Vector256.Create((byte)0x70);
            var r = Avx2.Shuffle(Vector256.Create(lo.GetLower(), lo.GetLower()), Avx2.AddSaturate(idx, bias));
            if (srcWidth > 16)
                r |= Avx2.Shuffle(Vector256.Create(lo.GetUpper(), lo.GetUpper()), Avx2.AddSaturate(idx - Vector256.Create((byte)16), bias));
            if (srcWidth > 32)
                r |= Avx2.Shuffle(Vector256.Create(hi.GetLower(), hi.GetLower()), Avx2.AddSaturate(idx - Vector256.Create((byte)32), bias));
            if (srcWidth > 48)
                r |= Avx2.Shuffle(Vector256.Create(hi.GetUpper(), hi.GetUpper()), Avx2.AddSaturate(idx - Vector256.Create((byte)48), bias));
            return r;
        }

        /// <summary>
        ///     A padded result in two vectors: output unit o is source unit <c>o - left</c> where that lies in
        ///     <c>[0, len)</c>, <paramref name="fillUnit"/> at the other units below <paramref name="final"/>, and 0 from
        ///     <paramref name="final"/> on (NumPy's <c>string_pad</c>: fill, text, fill, zeros).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="lo">Source bytes 0–31.</param>
        /// <param name="hi">Source bytes 32–63.</param>
        /// <param name="srcWidth">The source width in bytes.</param>
        /// <param name="outWidth">The output width in bytes (at most 64).</param>
        /// <param name="left">The fill units before the text (0 ≤ left ≤ 64).</param>
        /// <param name="len">The text's length in units.</param>
        /// <param name="final">The result's length in units (left + len ≤ final ≤ the output's units).</param>
        /// <param name="fillUnit">The fill character.</param>
        /// <param name="rlo">Output bytes 0–31.</param>
        /// <param name="rhi">Output bytes 32–63.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void PadUnits<TUnit>(Vector256<byte> lo, Vector256<byte> hi, int srcWidth, int outWidth, int left, int len,
                                             int final, uint fillUnit, out Vector256<byte> rlo, out Vector256<byte> rhi)
            where TUnit : unmanaged
        {
            if (typeof(TUnit) == typeof(uint))
            {
                var iota = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7);
                rlo = PadDwords(lo, hi, srcWidth, iota, left, len, final, fillUnit);
                rhi = outWidth > 32 ? PadDwords(lo, hi, srcWidth, iota + Vector256.Create(8), left, len, final, fillUnit)
                                    : Vector256<byte>.Zero;
                return;
            }
            var iotaB = Vector256.Create((sbyte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            rlo = PadBytes(lo, hi, srcWidth, iotaB, left, len, final, (byte)fillUnit);
            rhi = outWidth > 32 ? PadBytes(lo, hi, srcWidth, iotaB + Vector256.Create((sbyte)32), left, len, final, (byte)fillUnit)
                                : Vector256<byte>.Zero;
        }

        /// <summary>One output vector of <see cref="PadUnits{TUnit}"/> for <c>U</c>: the output lanes <paramref name="o"/>.</summary>
        /// <param name="lo">Source units 0–7.</param>
        /// <param name="hi">Source units 8–15.</param>
        /// <param name="srcWidth">The source width in bytes.</param>
        /// <param name="o">The output unit index of each lane.</param>
        /// <param name="left">The fill units before the text.</param>
        /// <param name="len">The text's length.</param>
        /// <param name="final">The result's length.</param>
        /// <param name="fillUnit">The fill character.</param>
        /// <returns>The lanes' units.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> PadDwords(Vector256<byte> lo, Vector256<byte> hi, int srcWidth, Vector256<int> o, int left,
                                                 int len, int final, uint fillUnit)
        {
            var s = o - Vector256.Create(left);
            var text = Vector256.GreaterThan(s, Vector256.Create(-1)) & Vector256.LessThan(s, Vector256.Create(len));
            var fill = Vector256.LessThan(o, Vector256.Create(final)) & ~text;
            return ((GatherDwords(lo, hi, srcWidth, s) & text) | (Vector256.Create((int)fillUnit) & fill)).AsByte();
        }

        /// <summary>One output vector of <see cref="PadUnits{TUnit}"/> for <c>S</c>: the output lanes <paramref name="o"/>
        ///     (0–63, held as signed bytes, as are the source indices <c>o - left</c> in [-64, 63]).</summary>
        /// <param name="lo">Source bytes 0–31.</param>
        /// <param name="hi">Source bytes 32–63.</param>
        /// <param name="srcWidth">The source width in bytes.</param>
        /// <param name="o">The output byte index of each lane.</param>
        /// <param name="left">The fill bytes before the text.</param>
        /// <param name="len">The text's length.</param>
        /// <param name="final">The result's length.</param>
        /// <param name="fillByte">The fill character.</param>
        /// <returns>The lanes' bytes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> PadBytes(Vector256<byte> lo, Vector256<byte> hi, int srcWidth, Vector256<sbyte> o, int left,
                                                int len, int final, byte fillByte)
        {
            var s = o - Vector256.Create((sbyte)left);
            var text = Vector256.GreaterThan(s, Vector256.Create((sbyte)-1)) & Vector256.GreaterThan(Vector256.Create((sbyte)len), s);
            var fill = Vector256.GreaterThan(Vector256.Create((sbyte)final), o) & ~text;
            // A negative index wraps to 192-255 as a byte, which GatherBytes turns into a zero lane.
            return (GatherBytes(lo, hi, srcWidth, s.AsByte()) & text.AsByte()) | (Vector256.Create(fillByte) & fill.AsByte());
        }

        /// <summary>
        ///     Stores up to 64 bytes of result held in two vectors to an element of any width, zero-filling the element past
        ///     the 64 bytes (an output sized wider than its text: NumPy's <c>expandtabs</c> sizes <c>U</c> four times wide).
        /// </summary>
        /// <param name="dst">The element's first byte.</param>
        /// <param name="width">Its width in bytes (at least 1).</param>
        /// <param name="lo">Result bytes 0–31.</param>
        /// <param name="hi">Result bytes 32–63 (zero past the result).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void StoreElementZeroTail(byte* dst, int width, Vector256<byte> lo, Vector256<byte> hi)
        {
            if (width <= LaneBytes)
            {
                StoreElement(dst, width, lo, hi);
                return;
            }
            lo.Store(dst);
            hi.Store(dst + 32);
            new Span<byte>(dst + 64, width - 64).Clear();
        }

        /// <summary>
        ///     NumPy's <c>string_expandtabs</c> of an element without line breaks, in two vectors: output unit o is a space
        ///     inside a tab's run, else input unit <c>o - delta(o)</c> (delta: the columns the earlier tabs added, less the tab
        ///     units themselves), and 0 from the result's length on.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="lo">Source bytes 0–31.</param>
        /// <param name="hi">Source bytes 32–63.</param>
        /// <param name="srcWidth">The source width in bytes.</param>
        /// <param name="tabs">The tab lanes below the element's length.</param>
        /// <param name="len">The element's length in units.</param>
        /// <param name="tabsize">The tab size (≤ 0 removes the tabs).</param>
        /// <param name="rlo">Result bytes 0–31.</param>
        /// <param name="rhi">Result bytes 32–63.</param>
        /// <returns>False when the result passes 64 bytes (the caller takes the unit loop; the outputs are then unset).</returns>
        /// <remarks>Without a line break the column of a tab is its output position, so each tab's run is
        ///     <c>tabsize - pos % tabsize</c> spaces at that position (a power-of-two tab size takes a mask, not a division).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool ExpandTabsUnits<TUnit>(Vector256<byte> lo, Vector256<byte> hi, int srcWidth, ulong tabs, int len,
                                                    long tabsize, out Vector256<byte> rlo, out Vector256<byte> rhi)
            where TUnit : unmanaged
        {
            int maxUnits = LaneBytes / sizeof(TUnit);
            rlo = rhi = default;
            if (typeof(TUnit) == typeof(uint))
            {
                var o0 = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7);
                var o1 = o0 + Vector256.Create(8);
                Vector256<int> delta0 = default, delta1 = default, space0 = default, space1 = default;
                long pos = 0;
                int prev = 0;
                while (tabs != 0)
                {
                    int i = BitOperations.TrailingZeroCount(tabs);
                    tabs &= tabs - 1;
                    pos += i - prev;
                    long incr = TabRun(pos, tabsize);
                    if (pos + incr > maxUnits)
                        return false;
                    var from = Vector256.Create((int)pos);
                    var to = Vector256.Create((int)(pos + incr));
                    space0 |= ~Vector256.LessThan(o0, from) & Vector256.LessThan(o0, to);
                    space1 |= ~Vector256.LessThan(o1, from) & Vector256.LessThan(o1, to);
                    // The units after the run come from (incr - 1) units earlier than their position (the tab itself
                    // takes one input unit).
                    delta0 += ~Vector256.LessThan(o0, to) & Vector256.Create((int)incr - 1);
                    delta1 += ~Vector256.LessThan(o1, to) & Vector256.Create((int)incr - 1);
                    pos += incr;
                    prev = i + 1;
                }
                long final = pos + (len - prev);
                if (final > maxUnits)
                    return false;
                var fin = Vector256.Create((int)final);
                var blank = Vector256.Create(0x20);
                rlo = ((GatherDwords(lo, hi, srcWidth, o0 - delta0) & (Vector256.LessThan(o0, fin) & ~space0)) | (blank & space0)).AsByte();
                rhi = ((GatherDwords(lo, hi, srcWidth, o1 - delta1) & (Vector256.LessThan(o1, fin) & ~space1)) | (blank & space1)).AsByte();
                return true;
            }
            var b0 = Vector256.Create((sbyte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            var b1 = b0 + Vector256.Create((sbyte)32);
            Vector256<sbyte> d0 = default, d1 = default, s0 = default, s1 = default;
            long bpos = 0;
            int bprev = 0;
            while (tabs != 0)
            {
                int i = BitOperations.TrailingZeroCount(tabs);
                tabs &= tabs - 1;
                bpos += i - bprev;
                long incr = TabRun(bpos, tabsize);
                if (bpos + incr > maxUnits)
                    return false;
                var from = Vector256.Create((sbyte)bpos);
                var to = Vector256.Create((sbyte)(bpos + incr));
                s0 |= ~Vector256.LessThan(b0, from) & Vector256.LessThan(b0, to);
                s1 |= ~Vector256.LessThan(b1, from) & Vector256.LessThan(b1, to);
                d0 += ~Vector256.LessThan(b0, to) & Vector256.Create((sbyte)(incr - 1));
                d1 += ~Vector256.LessThan(b1, to) & Vector256.Create((sbyte)(incr - 1));
                bpos += incr;
                bprev = i + 1;
            }
            long bfinal = bpos + (len - bprev);
            if (bfinal > maxUnits)
                return false;
            var bfin = Vector256.Create((sbyte)bfinal);
            var space = Vector256.Create((byte)0x20);
            rlo = (GatherBytes(lo, hi, srcWidth, (b0 - d0).AsByte()) & (Vector256.LessThan(b0, bfin) & ~s0).AsByte()) | (space & s0.AsByte());
            rhi = (GatherBytes(lo, hi, srcWidth, (b1 - d1).AsByte()) & (Vector256.LessThan(b1, bfin) & ~s1).AsByte()) | (space & s1.AsByte());
            return true;
        }

        /// <summary>The spaces a tab at output column <paramref name="col"/> expands to: <c>tabsize - col % tabsize</c>, or 0
        ///     for a tab size of 0 or less (NumPy removes the tab).</summary>
        /// <param name="col">The column (≥ 0).</param>
        /// <param name="tabsize">The tab size.</param>
        /// <returns>The run length.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long TabRun(long col, long tabsize)
        {
            if (tabsize <= 0)
                return 0;
            // A power-of-two tab size (the default 8) takes a mask instead of a 64-bit division.
            long rem = (tabsize & (tabsize - 1)) == 0 ? col & (tabsize - 1) : col % tabsize;
            return tabsize - rem;
        }

        /// <summary>
        ///     The lanes below <paramref name="count"/> units of an element held in two vectors, the rest zeroed (a prefix:
        ///     no gather needed).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="lo">Bytes 0–31 (masked in place).</param>
        /// <param name="hi">Bytes 32–63 (masked in place).</param>
        /// <param name="count">The units to keep, 0 to 64.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void KeepUnits<TUnit>(ref Vector256<byte> lo, ref Vector256<byte> hi, int count) where TUnit : unmanaged
        {
            var iotaB = Vector256.Create((byte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            // Keep byte b when b < count · unit (a byte count ≤ 256: compared as 16-bit values would be exact, but the
            // byte count of a kept prefix is at most 64, so a signed byte compare holds it).
            var n = Vector256.Create((sbyte)(count * sizeof(TUnit) > 64 ? 64 : count * sizeof(TUnit)));
            lo &= Vector256.GreaterThan(n, iotaB.AsSByte()).AsByte();
            hi &= Vector256.GreaterThan(n, (iotaB + Vector256.Create((byte)32)).AsSByte()).AsByte();
        }

        /// <summary>
        ///     Copies the units <c>[first, first + count)</c> of an element (up to 64 bytes) to an output element (up to 64
        ///     bytes), zero-filled past them — <see cref="CopyUnitsAndClear{TUnit}"/> without a call or a branch on the data.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="lo">The source element's bytes 0–31.</param>
        /// <param name="hi">Its bytes 32–63.</param>
        /// <param name="srcWidth">The source width in bytes.</param>
        /// <param name="first">The first unit copied (0 ≤ first; first + count ≤ the source's units).</param>
        /// <param name="count">The units copied (at most the output's units).</param>
        /// <param name="dst">The output element.</param>
        /// <param name="dstWidth">Its width in bytes (<see cref="SelectsFit"/> holds).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CopyLanes<TUnit>(Vector256<byte> lo, Vector256<byte> hi, int srcWidth, int first, int count,
                                             byte* dst, int dstWidth)
            where TUnit : unmanaged
        {
            SelectUnits<TUnit>(lo, hi, srcWidth, dstWidth, first, 1, count, out var rlo, out var rhi);
            StoreElement(dst, dstWidth, rlo, rhi);
        }
    }
}
