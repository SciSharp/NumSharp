using System;
using System.Numerics;
using System.Runtime.CompilerServices;

// =============================================================================
// FloatDigits.cs — the decimal digits of a binary float, computed with integer arithmetic only: the shortest digits that
// round-trip (Ryu) and the correctly rounded digits at a given precision (CPython's %f / %e / %g)
// =============================================================================
//
// WHY THIS EXISTS
//   .NET's own formatter is exact but slow for these shapes: on the benchmark host double.TryFormat costs 139 ns for "F2",
//   101 ns for "E3" and 152 ns for "R" (the shortest round-trip form), against 8.6 ns for a long's digits. NumPy's
//   np.strings.mod spends 580–1,040 ns per element, so a float conversion through .NET's formatter caps np.strings.mod at
//   4–7× NumPy, and the float → text casts pay the same "R" per element. The routines here produce the SAME digits with a
//   64×64→128-bit multiply and a shift (fixed / scientific) or Ryu's three multiplies (shortest): 10–25 ns.
//
// THE TWO CONTRACTS (both byte-identical to .NET's exact formatter, which the printing and text oracles already gate)
//   * Shortest (Ryu, Ulf Adams, PLDI 2018; a port of ulfjack/ryu d2s.c / f2s.c): the fewest decimal digits that read back
//     as the same float, and among those the digits closest to the exact binary value (a tie rounds to an even digit) —
//     .NET's "R", CPython's repr, NumPy's Dragon4 unique mode. One routine per precision, because the shortest form of a
//     float32 is NOT the shortest form of the same value widened to float64 (0.1f is "0.1", not "0.10000000149011612"):
//     Ryu's d2d for float64, its f2d for float32, and f2d with float16's field widths for float16 (its correctness over all
//     65,536 half values is checked by a unit test, as Ryu's own proofs cover float32 and float64).
//   * Rounded at a precision: round_half_even(|x| × 10^s) of the EXACT binary value x = m × 2^e — the rounding of CPython's
//     dtoa (and of .NET's "F" / "E" formats), which rounds the true value, never a shorter decimal string. With m below
//     2^53 and 10^s below 2^64 the product fits 128 bits, so the floor and the fraction's class (zero, below a half, a half,
//     above) come out of one multiply and one shift. Values whose digits do not fit a ulong report failure, and the caller
//     takes its exact general route.
//
// TABLES
//   Ryu's 5^q multipliers are GENERATED at type initialization from BigInteger (668 entries, under a millisecond), exact by
//   construction, rather than transcribed: POW5_INV_SPLIT[q] = floor(2^(bitlength(5^q) - 1 + 125) / 5^q) + 1 and
//   POW5_SPLIT[i] = 5^i shifted to 125 significant bits; the float routine uses their upper 64 bits (Ryu's default build).
// =============================================================================

namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     The decimal digits of a binary float by integer arithmetic (see the file header): Ryu's shortest round-trip digits
    ///     for float64 / float32 / float16, and the correctly rounded digits of CPython's <c>%f</c> / <c>%e</c> (and so
    ///     <c>%g</c>) at a precision. Every routine gives the digits .NET's exact formatter gives, at a fraction of its cost.
    /// </summary>
    /// <remarks>
    ///     Thread-safe: the tables are built once by the type initializer and only read afterwards. The routines take the
    ///     MAGNITUDE of a finite value (the caller writes the sign and handles NaN / infinity); a routine that cannot produce
    ///     its digits in 64 bits returns false instead of a wrong answer.
    /// </remarks>
    internal static unsafe class FloatDigits
    {
        /// <summary>Ryu's precision of the float64 inverse multipliers (<c>DOUBLE_POW5_INV_BITCOUNT</c>).</summary>
        private const int DoublePow5InvBitCount = 125;

        /// <summary>Ryu's precision of the float64 multipliers (<c>DOUBLE_POW5_BITCOUNT</c>).</summary>
        private const int DoublePow5BitCount = 125;

        /// <summary>The float32 routine's inverse-multiplier precision: the upper 64 bits of the float64 table.</summary>
        private const int FloatPow5InvBitCount = DoublePow5InvBitCount - 64;

        /// <summary>The float32 routine's multiplier precision: the upper 64 bits of the float64 table.</summary>
        private const int FloatPow5BitCount = DoublePow5BitCount - 64;

        /// <summary>Entries of the inverse table: q runs to floor(log10(2^1074)) - 1 + 2 for float64 (Ryu's 342).</summary>
        private const int Pow5InvTableSize = 342;

        /// <summary>Entries of the multiplier table: i runs to 1074 - floor(log10(5^1074)) + 1 for float64 (Ryu's 326).</summary>
        private const int Pow5TableSize = 326;

        /// <summary>
        ///     <c>DOUBLE_POW5_INV_SPLIT</c>: entry q is <c>floor(2^(bitlength(5^q) - 1 + 125) / 5^q) + 1</c> as two 64-bit
        ///     words, low word at <c>[2q]</c>, high word at <c>[2q + 1]</c>.
        /// </summary>
        private static readonly ulong[] Pow5InvSplit = BuildPow5InvSplit();

        /// <summary>
        ///     <c>DOUBLE_POW5_SPLIT</c>: entry i is <c>5^i</c> shifted to exactly 125 significant bits, low word at
        ///     <c>[2i]</c>, high word at <c>[2i + 1]</c>.
        /// </summary>
        private static readonly ulong[] Pow5Split = BuildPow5Split();

        /// <summary>The powers of ten that fit a ulong: <c>Pow10[k] = 10^k</c> for k from 0 to 19.</summary>
        internal static readonly ulong[] Pow10 =
        {
            1UL, 10UL, 100UL, 1000UL, 10000UL, 100000UL, 1000000UL, 10000000UL, 100000000UL, 1000000000UL,
            10000000000UL, 100000000000UL, 1000000000000UL, 10000000000000UL, 100000000000000UL, 1000000000000000UL,
            10000000000000000UL, 100000000000000000UL, 1000000000000000000UL, 10000000000000000000UL,
        };

        /// <summary>The largest precision <see cref="TryFixed"/> serves: <c>10^19</c> is the largest power of ten in a ulong.</summary>
        internal const int MaxFixedPrecision = 19;

        /// <summary>The largest precision <see cref="TryScientific"/> serves: its P + 1 digits must fit a ulong with one
        ///     carry digit to spare (<c>10^18</c>).</summary>
        internal const int MaxScientificPrecision = 17;

        /// <summary>"00" through "99": two ASCII digits per value, the digit writer's table.</summary>
        private static ReadOnlySpan<byte> TwoDigits =>
            "00010203040506070809101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899"u8;

        // =============================================================================================================
        // Tables
        // =============================================================================================================

        /// <summary>Builds <see cref="Pow5InvSplit"/> (see its summary) exactly with <see cref="BigInteger"/>.</summary>
        /// <returns>The table, two words per entry.</returns>
        private static ulong[] BuildPow5InvSplit()
        {
            var table = new ulong[Pow5InvTableSize * 2];
            var mask = (BigInteger.One << 64) - 1;
            for (int q = 0; q < Pow5InvTableSize; q++)
            {
                var pow = BigInteger.Pow(5, q);
                int j = (int)pow.GetBitLength() - 1 + DoublePow5InvBitCount;
                var inv = (BigInteger.One << j) / pow + 1;
                table[2 * q] = (ulong)(inv & mask);
                table[2 * q + 1] = (ulong)(inv >> 64);
            }
            return table;
        }

        /// <summary>Builds <see cref="Pow5Split"/> (see its summary) exactly with <see cref="BigInteger"/>.</summary>
        /// <returns>The table, two words per entry.</returns>
        private static ulong[] BuildPow5Split()
        {
            var table = new ulong[Pow5TableSize * 2];
            var mask = (BigInteger.One << 64) - 1;
            for (int i = 0; i < Pow5TableSize; i++)
            {
                var pow = BigInteger.Pow(5, i);
                int shift = (int)pow.GetBitLength() - DoublePow5BitCount;
                // Ryu's generator shifts right by bitlength - 125: a negative count (a small 5^i) is a shift LEFT.
                var split = shift >= 0 ? pow >> shift : pow << -shift;
                table[2 * i] = (ulong)(split & mask);
                table[2 * i + 1] = (ulong)(split >> 64);
            }
            return table;
        }

        // =============================================================================================================
        // Ryu's helpers
        // =============================================================================================================

        /// <summary><c>pow5bits</c>: <c>ceil(log2(5^e))</c> (1 for e = 0), for 0 ≤ e ≤ 3528.</summary>
        /// <param name="e">The power of five.</param>
        /// <returns>Its bit length.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Pow5Bits(int e) => (int)((((uint)e * 1217359u) >> 19) + 1);

        /// <summary><c>log10Pow2</c>: <c>floor(log10(2^e))</c>, for 0 ≤ e ≤ 1650.</summary>
        /// <param name="e">The power of two.</param>
        /// <returns>Its decimal exponent.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Log10Pow2(int e) => ((uint)e * 78913u) >> 18;

        /// <summary><c>log10Pow5</c>: <c>floor(log10(5^e))</c>, for 0 ≤ e ≤ 2620.</summary>
        /// <param name="e">The power of five.</param>
        /// <returns>Its decimal exponent.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Log10Pow5(int e) => ((uint)e * 732923u) >> 20;

        /// <summary>Whether <paramref name="value"/> is a multiple of <c>5^p</c> (Ryu's <c>pow5Factor(value) &gt;= p</c>).</summary>
        /// <param name="value">The value (not zero).</param>
        /// <param name="p">The power.</param>
        /// <returns>True when divisible.</returns>
        private static bool MultipleOfPowerOf5(ulong value, uint p)
        {
            uint count = 0;
            while (value % 5 == 0)
            {
                value /= 5;
                count++;
            }
            return count >= p;
        }

        /// <summary>Whether <paramref name="value"/> is a multiple of <c>2^p</c>, for p below 64.</summary>
        /// <param name="value">The value.</param>
        /// <param name="p">The power.</param>
        /// <returns>True when divisible.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool MultipleOfPowerOf2(ulong value, uint p) => (value & ((1UL << (int)p) - 1)) == 0;

        /// <summary>
        ///     Ryu's <c>mulShift64</c>: <c>(m × (hi:lo)) &gt;&gt; j</c> for a multiplier of the 125-bit tables, j between 65
        ///     and 127 (Ryu's analysis bounds the shift; <c>shiftright128</c> requires 0 &lt; j - 64 &lt; 64).
        /// </summary>
        /// <param name="m">The multiplicand (at most 55 bits).</param>
        /// <param name="lo">The multiplier's low word.</param>
        /// <param name="hi">The multiplier's high word.</param>
        /// <param name="j">The shift.</param>
        /// <returns>The shifted product's low 64 bits.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong MulShift64(ulong m, ulong lo, ulong hi, int j)
        {
            ulong b0Hi = Math.BigMul(m, lo, out _);
            ulong b2Hi = Math.BigMul(m, hi, out ulong b2Lo);
            ulong sumLo = b2Lo + b0Hi;
            ulong sumHi = b2Hi + (sumLo < b0Hi ? 1UL : 0UL);
            int dist = j - 64;
            return (sumHi << (64 - dist)) | (sumLo >> dist);
        }

        /// <summary>Ryu's <c>mulShift32</c>: <c>(m × factor) &gt;&gt; shift</c> for a 64-bit factor, shift above 32.</summary>
        /// <param name="m">The multiplicand.</param>
        /// <param name="factor">The multiplier.</param>
        /// <param name="shift">The shift (above 32).</param>
        /// <returns>The shifted product (fits 32 bits by Ryu's analysis).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint MulShift32(uint m, ulong factor, int shift)
        {
            ulong bits0 = (ulong)m * (uint)factor;
            ulong bits1 = (ulong)m * (uint)(factor >> 32);
            ulong sum = (bits0 >> 32) + bits1;
            return (uint)(sum >> (shift - 32));
        }

        /// <summary>The float routine's inverse multiplier: the float64 entry's upper word plus 1 (Ryu's default build: the
        ///     upper 64 bits are <c>floor(2^x / 5^q)</c>, and the stored +1 never carries into them).</summary>
        /// <param name="m">The multiplicand.</param>
        /// <param name="q">The power of five.</param>
        /// <param name="j">The shift.</param>
        /// <returns>The shifted product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint MulPow5InvDivPow2(uint m, uint q, int j) => MulShift32(m, Pow5InvSplit[2 * q + 1] + 1, j);

        /// <summary>The float routine's multiplier: the float64 entry's upper word.</summary>
        /// <param name="m">The multiplicand.</param>
        /// <param name="i">The power of five.</param>
        /// <param name="j">The shift.</param>
        /// <returns>The shifted product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint MulPow5DivPow2(uint m, uint i, int j) => MulShift32(m, Pow5Split[2 * i + 1], j);

        // =============================================================================================================
        // Shortest round-trip digits
        // =============================================================================================================

        /// <summary>
        ///     The shortest decimal digits that read back as <paramref name="value"/> (a float64; its sign is ignored), the
        ///     closest to the exact value among them: <c>|value| = digits × 10^exponent</c>, without trailing zeros. Zero is
        ///     <c>0 × 10^0</c>.
        /// </summary>
        /// <param name="value">A finite value.</param>
        /// <param name="exponent">Receives the decimal exponent of the last digit.</param>
        /// <returns>The digits as an integer (at most 17 digits).</returns>
        internal static ulong Shortest(double value, out int exponent)
        {
            ulong bits = BitConverter.DoubleToUInt64Bits(value);
            ulong ieeeMantissa = bits & ((1UL << 52) - 1);
            uint ieeeExponent = (uint)(bits >> 52) & 0x7FF;
            if (ieeeExponent == 0 && ieeeMantissa == 0)
            {
                exponent = 0;
                return 0;
            }

            // Step 1: the value as m2 × 2^e2, with two extra bits so the interval's bounds are integers too.
            int e2;
            ulong m2;
            if (ieeeExponent == 0)
            {
                e2 = 1 - 1023 - 52 - 2;
                m2 = ieeeMantissa;
            }
            else
            {
                e2 = (int)ieeeExponent - 1023 - 52 - 2;
                m2 = (1UL << 52) | ieeeMantissa;
            }
            // Round-half-even reading: an even mantissa owns both ends of its interval.
            bool acceptBounds = (m2 & 1) == 0;

            // Step 2: the interval of valid representations, [mm, mp] around mv (the lower gap is half as wide at a power of
            // two, except at the smallest normal exponent).
            ulong mv = 4 * m2;
            uint mmShift = ieeeMantissa != 0 || ieeeExponent <= 1 ? 1u : 0u;

            // Step 3: the three values in a decimal power base, with whether the dropped parts are zero.
            ulong vr, vp, vm;
            int e10;
            bool vmIsTrailingZeros = false, vrIsTrailingZeros = false;
            if (e2 >= 0)
            {
                uint q = Log10Pow2(e2) - (e2 > 3 ? 1u : 0u);
                e10 = (int)q;
                int k = DoublePow5InvBitCount + Pow5Bits((int)q) - 1;
                int i = -e2 + (int)q + k;
                ulong lo = Pow5InvSplit[2 * q], hi = Pow5InvSplit[2 * q + 1];
                vr = MulShift64(4 * m2, lo, hi, i);
                vp = MulShift64(4 * m2 + 2, lo, hi, i);
                vm = MulShift64(4 * m2 - 1 - mmShift, lo, hi, i);
                if (q <= 21)
                {
                    // Only one of mp, mv and mm can be a multiple of 5, if any.
                    if (mv % 5 == 0)
                        vrIsTrailingZeros = MultipleOfPowerOf5(mv, q);
                    else if (acceptBounds)
                        vmIsTrailingZeros = MultipleOfPowerOf5(mv - 1 - mmShift, q);
                    else
                        vp -= MultipleOfPowerOf5(mv + 2, q) ? 1UL : 0UL;
                }
            }
            else
            {
                uint q = Log10Pow5(-e2) - (-e2 > 1 ? 1u : 0u);
                e10 = (int)q + e2;
                int i = -e2 - (int)q;
                int k = Pow5Bits(i) - DoublePow5BitCount;
                int j = (int)q - k;
                ulong lo = Pow5Split[2 * i], hi = Pow5Split[2 * i + 1];
                vr = MulShift64(4 * m2, lo, hi, j);
                vp = MulShift64(4 * m2 + 2, lo, hi, j);
                vm = MulShift64(4 * m2 - 1 - mmShift, lo, hi, j);
                if (q <= 1)
                {
                    // mv = 4 * m2 has two trailing zero bits; mm = mv - 1 - mmShift has one exactly when mmShift is 1.
                    vrIsTrailingZeros = true;
                    if (acceptBounds)
                        vmIsTrailingZeros = mmShift == 1;
                    else
                        --vp;   // mp = mv + 2 always has a trailing zero bit
                }
                else if (q < 63)
                {
                    vrIsTrailingZeros = MultipleOfPowerOf2(mv, q);
                }
            }

            // Step 4: the shortest representation in the interval.
            int removed = 0;
            byte lastRemovedDigit = 0;
            ulong output;
            if (vmIsTrailingZeros || vrIsTrailingZeros)
            {
                // The general case (rare): the exact bounds matter.
                for (;;)
                {
                    ulong vpDiv10 = vp / 10, vmDiv10 = vm / 10;
                    if (vpDiv10 <= vmDiv10)
                        break;
                    uint vmMod10 = (uint)(vm - 10 * vmDiv10);
                    ulong vrDiv10 = vr / 10;
                    uint vrMod10 = (uint)(vr - 10 * vrDiv10);
                    vmIsTrailingZeros &= vmMod10 == 0;
                    vrIsTrailingZeros &= lastRemovedDigit == 0;
                    lastRemovedDigit = (byte)vrMod10;
                    vr = vrDiv10;
                    vp = vpDiv10;
                    vm = vmDiv10;
                    ++removed;
                }
                if (vmIsTrailingZeros)
                {
                    for (;;)
                    {
                        ulong vmDiv10 = vm / 10;
                        uint vmMod10 = (uint)(vm - 10 * vmDiv10);
                        if (vmMod10 != 0)
                            break;
                        ulong vpDiv10 = vp / 10;
                        ulong vrDiv10 = vr / 10;
                        uint vrMod10 = (uint)(vr - 10 * vrDiv10);
                        vrIsTrailingZeros &= lastRemovedDigit == 0;
                        lastRemovedDigit = (byte)vrMod10;
                        vr = vrDiv10;
                        vp = vpDiv10;
                        vm = vmDiv10;
                        ++removed;
                    }
                }
                // Round half to even when the exact number is ....50..0.
                if (vrIsTrailingZeros && lastRemovedDigit == 5 && vr % 2 == 0)
                    lastRemovedDigit = 4;
                // vr + 1 when vr is outside the bounds or the removed digits round up.
                output = vr + ((vr == vm && (!acceptBounds || !vmIsTrailingZeros)) || lastRemovedDigit >= 5 ? 1UL : 0UL);
            }
            else
            {
                // The common case: two digits at a time first, then one.
                bool roundUp = false;
                ulong vpDiv100 = vp / 100, vmDiv100 = vm / 100;
                if (vpDiv100 > vmDiv100)
                {
                    ulong vrDiv100 = vr / 100;
                    uint vrMod100 = (uint)(vr - 100 * vrDiv100);
                    roundUp = vrMod100 >= 50;
                    vr = vrDiv100;
                    vp = vpDiv100;
                    vm = vmDiv100;
                    removed += 2;
                }
                for (;;)
                {
                    ulong vpDiv10 = vp / 10, vmDiv10 = vm / 10;
                    if (vpDiv10 <= vmDiv10)
                        break;
                    ulong vrDiv10 = vr / 10;
                    uint vrMod10 = (uint)(vr - 10 * vrDiv10);
                    roundUp = vrMod10 >= 5;
                    vr = vrDiv10;
                    vp = vpDiv10;
                    vm = vmDiv10;
                    ++removed;
                }
                output = vr + (vr == vm || roundUp ? 1UL : 0UL);
            }
            exponent = e10 + removed;
            // The contract has no trailing zeros (the digit writers count on it); Ryu's search leaves none in practice,
            // and stripping here keeps the contract unconditional.
            while (output % 10 == 0)
            {
                output /= 10;
                exponent++;
            }
            return output;
        }

        /// <summary>
        ///     The shortest decimal digits of a float32 in its OWN precision (its sign ignored): <c>|value| = digits ×
        ///     10^exponent</c>, no trailing zeros; zero is <c>0 × 10^0</c>.
        /// </summary>
        /// <param name="value">A finite value.</param>
        /// <param name="exponent">Receives the decimal exponent of the last digit.</param>
        /// <returns>The digits (at most 9).</returns>
        internal static uint Shortest(float value, out int exponent)
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);
            uint mantissa = bits & 0x7FFFFFu;
            uint biased = (bits >> 23) & 0xFFu;
            if (biased == 0 && mantissa == 0)
            {
                exponent = 0;
                return 0;
            }
            return ShortestBinary32(mantissa, biased, 23, 127, out exponent);
        }

        /// <summary>
        ///     The shortest decimal digits of a float16 in its own precision (its sign ignored): Ryu's float32 routine with
        ///     float16's field widths (10 mantissa bits, bias 15).
        /// </summary>
        /// <param name="value">A finite value.</param>
        /// <param name="exponent">Receives the decimal exponent of the last digit.</param>
        /// <returns>The digits (at most 5).</returns>
        internal static uint Shortest(Half value, out int exponent)
        {
            ushort bits = BitConverter.HalfToUInt16Bits(value);
            uint mantissa = bits & 0x3FFu;
            uint biased = (uint)(bits >> 10) & 0x1Fu;
            if (biased == 0 && mantissa == 0)
            {
                exponent = 0;
                return 0;
            }
            return ShortestBinary32(mantissa, biased, 10, 15, out exponent);
        }

        /// <summary>
        ///     Ryu's <c>f2d</c> over a binary format of at most 32 bits: the float32 routine, parameterised by the field widths
        ///     so float16 runs it too.
        /// </summary>
        /// <param name="ieeeMantissa">The stored mantissa bits.</param>
        /// <param name="ieeeExponent">The stored (biased) exponent; not all ones (the caller handles NaN / infinity).</param>
        /// <param name="mantissaBits">The mantissa field width (23 for float32, 10 for float16).</param>
        /// <param name="bias">The exponent bias (127 / 15).</param>
        /// <param name="exponent">Receives the decimal exponent of the last digit.</param>
        /// <returns>The digits, no trailing zeros.</returns>
        private static uint ShortestBinary32(uint ieeeMantissa, uint ieeeExponent, int mantissaBits, int bias, out int exponent)
        {
            int e2;
            uint m2;
            if (ieeeExponent == 0)
            {
                e2 = 1 - bias - mantissaBits - 2;
                m2 = ieeeMantissa;
            }
            else
            {
                e2 = (int)ieeeExponent - bias - mantissaBits - 2;
                m2 = (1u << mantissaBits) | ieeeMantissa;
            }
            bool acceptBounds = (m2 & 1) == 0;

            uint mv = 4 * m2;
            uint mp = 4 * m2 + 2;
            uint mmShift = ieeeMantissa != 0 || ieeeExponent <= 1 ? 1u : 0u;
            uint mm = 4 * m2 - 1 - mmShift;

            uint vr, vp, vm;
            int e10;
            bool vmIsTrailingZeros = false, vrIsTrailingZeros = false;
            byte lastRemovedDigit = 0;
            if (e2 >= 0)
            {
                uint q = Log10Pow2(e2);
                e10 = (int)q;
                int k = FloatPow5InvBitCount + Pow5Bits((int)q) - 1;
                int i = -e2 + (int)q + k;
                vr = MulPow5InvDivPow2(mv, q, i);
                vp = MulPow5InvDivPow2(mp, q, i);
                vm = MulPow5InvDivPow2(mm, q, i);
                if (q != 0 && (vp - 1) / 10 <= vm / 10)
                {
                    // One removed digit is needed even when the loop below does not run (q - 1 keeps the arithmetic in 32
                    // bits, as Ryu does).
                    int l = FloatPow5InvBitCount + Pow5Bits((int)(q - 1)) - 1;
                    lastRemovedDigit = (byte)(MulPow5InvDivPow2(mv, q - 1, -e2 + (int)q - 1 + l) % 10);
                }
                if (q <= 9)
                {
                    if (mv % 5 == 0)
                        vrIsTrailingZeros = MultipleOfPowerOf5(mv, q);
                    else if (acceptBounds)
                        vmIsTrailingZeros = MultipleOfPowerOf5(mm, q);
                    else
                        vp -= MultipleOfPowerOf5(mp, q) ? 1u : 0u;
                }
            }
            else
            {
                uint q = Log10Pow5(-e2);
                e10 = (int)q + e2;
                int i = -e2 - (int)q;
                int k = Pow5Bits(i) - FloatPow5BitCount;
                int j = (int)q - k;
                vr = MulPow5DivPow2(mv, (uint)i, j);
                vp = MulPow5DivPow2(mp, (uint)i, j);
                vm = MulPow5DivPow2(mm, (uint)i, j);
                if (q != 0 && (vp - 1) / 10 <= vm / 10)
                {
                    j = (int)q - 1 - (Pow5Bits(i + 1) - FloatPow5BitCount);
                    lastRemovedDigit = (byte)(MulPow5DivPow2(mv, (uint)(i + 1), j) % 10);
                }
                if (q <= 1)
                {
                    vrIsTrailingZeros = true;
                    if (acceptBounds)
                        vmIsTrailingZeros = mmShift == 1;
                    else
                        --vp;
                }
                else if (q < 31)
                {
                    vrIsTrailingZeros = MultipleOfPowerOf2(mv, q - 1);
                }
            }

            int removed = 0;
            uint output;
            if (vmIsTrailingZeros || vrIsTrailingZeros)
            {
                while (vp / 10 > vm / 10)
                {
                    vmIsTrailingZeros &= vm % 10 == 0;
                    vrIsTrailingZeros &= lastRemovedDigit == 0;
                    lastRemovedDigit = (byte)(vr % 10);
                    vr /= 10;
                    vp /= 10;
                    vm /= 10;
                    ++removed;
                }
                if (vmIsTrailingZeros)
                {
                    while (vm % 10 == 0)
                    {
                        vrIsTrailingZeros &= lastRemovedDigit == 0;
                        lastRemovedDigit = (byte)(vr % 10);
                        vr /= 10;
                        vp /= 10;
                        vm /= 10;
                        ++removed;
                    }
                }
                if (vrIsTrailingZeros && lastRemovedDigit == 5 && vr % 2 == 0)
                    lastRemovedDigit = 4;
                output = vr + ((vr == vm && (!acceptBounds || !vmIsTrailingZeros)) || lastRemovedDigit >= 5 ? 1u : 0u);
            }
            else
            {
                while (vp / 10 > vm / 10)
                {
                    lastRemovedDigit = (byte)(vr % 10);
                    vr /= 10;
                    vp /= 10;
                    vm /= 10;
                    ++removed;
                }
                output = vr + (vr == vm || lastRemovedDigit >= 5 ? 1u : 0u);
            }
            exponent = e10 + removed;
            while (output % 10 == 0)
            {
                output /= 10;
                exponent++;
            }
            return output;
        }

        // =============================================================================================================
        // Digits at a precision (CPython's %f / %e, and %g through %e)
        // =============================================================================================================

        /// <summary>A finite non-negative double as <c>m × 2^e</c> with m below 2^53 (m = 0 for zero).</summary>
        /// <param name="abs">The magnitude.</param>
        /// <param name="m">Receives the integer significand.</param>
        /// <param name="e">Receives the binary exponent.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Decompose(double abs, out ulong m, out int e)
        {
            ulong bits = BitConverter.DoubleToUInt64Bits(abs) & 0x7FFF_FFFF_FFFF_FFFFUL;
            int biased = (int)(bits >> 52);
            m = bits & 0xF_FFFF_FFFF_FFFFUL;
            if (biased == 0)
            {
                e = -1074;
            }
            else
            {
                m |= 1UL << 52;
                e = biased - 1075;
            }
        }

        /// <summary>
        ///     <c>floor(m × 2^e × 10^s)</c> exactly, with the class of the dropped fraction, for 0 ≤ s ≤ 19: the product
        ///     <c>m × 10^s</c> has at most 117 bits, so one 128-bit multiply and one shift give both.
        /// </summary>
        /// <param name="m">The significand (below 2^53).</param>
        /// <param name="e">The binary exponent.</param>
        /// <param name="s">The decimal scale (0–19).</param>
        /// <param name="floor">Receives the floor.</param>
        /// <param name="fraction">Receives the dropped fraction's class: 0 none, 1 below a half, 2 exactly a half, 3 above.</param>
        /// <returns>False when the floor does not fit a ulong.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TryScale(ulong m, int e, int s, out ulong floor, out int fraction)
        {
            // N = m × 10^s as two words (hi below 2^53, as m < 2^53 and 10^s < 2^64). Word arithmetic instead of UInt128:
            // the common shifts stay within one word.
            ulong hi = Math.BigMul(m, Pow10[s], out ulong lo);
            floor = 0;
            fraction = 0;
            if (e >= 0)
            {
                // An integer: N << e must stay below 2^64.
                if (hi != 0 || e >= 64 || (e > 0 && (lo >> (64 - e)) != 0))
                    return false;
                floor = lo << e;
                return true;
            }
            int k = -e;
            if (k < 64)
            {
                // N >> k fits a ulong exactly when hi < 2^k; the remainder is lo's low k bits, a half is bit k - 1.
                if ((hi >> k) != 0)
                    return false;
                floor = (hi << (64 - k)) | (lo >> k);
                ulong rem = lo & ((1UL << k) - 1);
                if (rem != 0)
                {
                    ulong half = 1UL << (k - 1);
                    fraction = rem < half ? 1 : rem == half ? 2 : 3;
                }
                return true;
            }
            if (k == 64)
            {
                floor = hi;
                fraction = lo == 0 ? 0 : lo < 1UL << 63 ? 1 : lo == 1UL << 63 ? 2 : 3;
                return true;
            }
            if (k < 128)
            {
                // The remainder is (hi's low j bits):lo, a half is 2^(j-1):0.
                int j = k - 64;
                floor = hi >> j;
                ulong remHi = hi & ((1UL << j) - 1);
                ulong halfHi = 1UL << (j - 1);
                if (remHi == 0 && lo == 0)
                    fraction = 0;
                else if (remHi < halfHi)
                    fraction = 1;
                else if (remHi == halfHi)
                    fraction = lo == 0 ? 2 : 3;
                else
                    fraction = 3;
                return true;
            }
            // N < 2^117 < 2^(k-1): the value is below a half.
            fraction = (hi | lo) == 0 ? 0 : 1;
            return true;
        }

        /// <summary>
        ///     <c>round_half_even(m × 2^e × 10^s)</c> for 0 ≤ s ≤ 19. The common shift (1 to 63 bits: values from about
        ///     2^-11 up, depending on s) decides the rounding WITHOUT a branch — up when the dropped bits exceed a half, or
        ///     equal it and the kept value is odd — because that decision is a coin flip per element on real data, and a
        ///     mispredicted branch costs more than the whole computation; other shifts classify through <see cref="TryScale"/>.
        /// </summary>
        /// <param name="m">The significand (below 2^53).</param>
        /// <param name="e">The binary exponent.</param>
        /// <param name="s">The decimal scale (0–19).</param>
        /// <param name="rounded">Receives the rounded value.</param>
        /// <returns>False when the value does not fit a ulong.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TryScaleRounded(ulong m, int e, int s, out ulong rounded)
        {
            int k = -e;
            if ((uint)(k - 1) < 63u)
            {
                ulong hi = Math.BigMul(m, Pow10[s], out ulong lo);
                rounded = 0;
                if ((hi >> k) != 0)
                    return false;
                ulong q = (hi << (64 - k)) | (lo >> k);
                ulong rem = lo & ((1UL << k) - 1);
                ulong half = 1UL << (k - 1);
                ulong up = (rem > half ? 1UL : 0UL) | ((rem == half ? 1UL : 0UL) & q & 1UL);
                if (q == ulong.MaxValue && up != 0)
                    return false;   // q is 2^64 - 1 and rounds up: no ulong holds it (the first test is never true on real data)
                rounded = q + up;
                return true;
            }
            if (!TryScale(m, e, s, out ulong q2, out int fraction))
            {
                rounded = 0;
                return false;
            }
            if (fraction == 3 || (fraction == 2 && (q2 & 1) != 0))
            {
                if (q2 == ulong.MaxValue)
                {
                    rounded = 0;
                    return false;
                }
                q2++;
            }
            rounded = q2;
            return true;
        }

        /// <summary>
        ///     <c>round_half_even(|x| × 10^precision)</c> of the exact value: the digits of CPython's <c>%.{precision}f</c>
        ///     (the integer and fraction digits run together; the caller places the point <paramref name="precision"/>
        ///     digits from the right).
        /// </summary>
        /// <param name="abs">The magnitude (finite, not negative).</param>
        /// <param name="precision">The number of fraction digits (0 to <see cref="MaxFixedPrecision"/>).</param>
        /// <param name="digits">Receives the rounded scaled value.</param>
        /// <returns>False when the precision is out of range or the digits do not fit a ulong (the caller's general route
        ///     formats such a value).</returns>
        internal static bool TryFixed(double abs, int precision, out ulong digits)
        {
            digits = 0;
            if ((uint)precision > MaxFixedPrecision)
                return false;
            Decompose(abs, out ulong m, out int e);
            return TryScaleRounded(m, e, precision, out digits);
        }

        /// <summary>
        ///     The digits of CPython's <c>%.{precision}e</c>: <c>precision + 1</c> significant digits of the exact value,
        ///     rounded half to even, and the decimal exponent of the first one (a rounding that carries to the next power of
        ///     ten moves the exponent: 9.9996 at precision 3 is <c>1000</c>, exponent 1). Zero gives digits 0, exponent 0.
        /// </summary>
        /// <param name="abs">The magnitude (finite, not negative).</param>
        /// <param name="precision">The number of fraction digits of the mantissa (0 to <see cref="MaxScientificPrecision"/>).</param>
        /// <param name="digits">Receives the digits: <c>10^precision ≤ digits &lt; 10^(precision + 1)</c> (0 for zero).</param>
        /// <param name="exponent">Receives the decimal exponent.</param>
        /// <returns>False when the precision is out of range, or the value is too small or too large for the scale to fit
        ///     (about below 10^-(19 - precision) or at least 2^64).</returns>
        internal static bool TryScientific(double abs, int precision, out ulong digits, out int exponent)
        {
            digits = 0;
            exponent = 0;
            if ((uint)precision > MaxScientificPrecision)
                return false;
            Decompose(abs, out ulong m, out int e);
            if (m == 0)
                return true;

            // floor(log10(x)) is e0 or e0 + 1 for e0 = floor(log10(2^floor(log2 x))) (Ryu's log10Pow2, extended to negative
            // powers: floor(-n log10 2) = -(floor(n log10 2) + 1), as n log10 2 is never an integer).
            int lg2 = e + 63 - BitOperations.LeadingZeroCount(m);
            int e0 = lg2 >= 0 ? (int)Log10Pow2(lg2) : -((int)Log10Pow2(-lg2) + 1);
            ulong limit = Pow10[precision + 1];

            if (!TryRoundAt(m, e, precision - e0, out ulong q))
                return false;
            if (q < limit)
            {
                digits = q;
                exponent = e0;
                return true;
            }
            if (q == limit)
            {
                // A carry to the next power of ten, or a value exactly at it: both are 10^precision at e0 + 1.
                digits = Pow10[precision];
                exponent = e0 + 1;
                return true;
            }
            // floor(log10 x) is e0 + 1.
            if (!TryRoundAt(m, e, precision - e0 - 1, out q))
                return false;
            if (q == limit)
            {
                digits = Pow10[precision];
                exponent = e0 + 2;
                return true;
            }
            digits = q;
            exponent = e0 + 1;
            return true;
        }

        /// <summary>
        ///     <c>round_half_even(m × 2^e × 10^s)</c> for any decimal scale s from -19 to 19: a positive scale multiplies
        ///     (<see cref="TryScale"/>); a negative one takes the exact integer part and drops <c>-s</c> decimal digits from
        ///     it, the integer part's own fraction deciding a tie (anything left of a half's digits makes it larger).
        /// </summary>
        /// <param name="m">The significand.</param>
        /// <param name="e">The binary exponent.</param>
        /// <param name="s">The decimal scale.</param>
        /// <param name="rounded">Receives the rounded value.</param>
        /// <returns>False when out of range.</returns>
        private static bool TryRoundAt(ulong m, int e, int s, out ulong rounded)
        {
            rounded = 0;
            if (s >= 0)
                return s <= MaxFixedPrecision && TryScaleRounded(m, e, s, out rounded);
            int r = -s;
            if (r > MaxFixedPrecision || !TryScale(m, e, 0, out ulong ip, out int ipFraction))
                return false;
            ulong p = Pow10[r];
            ulong qd = ip / p;
            ulong dropped = ip - qd * p;
            ulong half = p / 2;
            // dropped + f (f the integer part's own fraction) against a half: above when dropped > half, or dropped == half
            // with f > 0; exactly a half only when f == 0 — then the even digit wins.
            if (dropped > half || (dropped == half && (ipFraction != 0 || (qd & 1) != 0)))
                qd++;
            rounded = qd;
            return true;
        }

        // =============================================================================================================
        // Digit writers
        // =============================================================================================================

        /// <summary>The number of decimal digits of <paramref name="v"/> (1 for 0).</summary>
        /// <param name="v">The value.</param>
        /// <returns>1–20.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int DecimalLength(ulong v)
        {
            if (v == 0)
                return 1;
            // t = floor(bitlength × log10 2) (1233 / 4096 ≈ log10 2): the digit count is t or t + 1, and one compare
            // against 10^t decides (Bit Twiddling Hacks' IntegerLog10). t stays at most 19, inside Pow10.
            int t = ((64 - BitOperations.LeadingZeroCount(v)) * 1233) >> 12;
            return v >= Pow10[t] ? t + 1 : t;
        }

        /// <summary>
        ///     Writes the decimal digits of <paramref name="v"/> as ASCII ending just before <paramref name="end"/> (right
        ///     to left, two digits per step); returns how many were written (1 for 0).
        /// </summary>
        /// <param name="v">The value.</param>
        /// <param name="end">One past the last digit's slot.</param>
        /// <returns>The number of digits.</returns>
        internal static int WriteDecimalBackwards(ulong v, byte* end)
        {
            byte* p = end;
            ref byte table = ref Unsafe.AsRef(in TwoDigits[0]);
            // Eight digits per 64-bit division, then 32-bit arithmetic within the group (a 64-bit division by a constant is
            // a 128-bit multiply-high; the 32-bit ones are cheaper).
            while (v >= 100000000)
            {
                ulong q = v / 100000000;
                uint r = (uint)(v - q * 100000000);
                uint upper = r / 10000;
                uint lower = r - upper * 10000;
                p -= 8;
                Put4(p + 4, lower, ref table);
                Put4(p, upper, ref table);
                v = q;
            }
            uint u = (uint)v;
            while (u >= 100)
            {
                uint q = u / 100;
                int r = (int)(u - q * 100) * 2;
                p -= 2;
                p[0] = Unsafe.Add(ref table, r);
                p[1] = Unsafe.Add(ref table, r + 1);
                u = q;
            }
            if (u >= 10)
            {
                int r = (int)u * 2;
                p -= 2;
                p[0] = Unsafe.Add(ref table, r);
                p[1] = Unsafe.Add(ref table, r + 1);
            }
            else
            {
                *--p = (byte)('0' + (int)u);
            }
            return (int)(end - p);
        }

        /// <summary>Writes exactly four ASCII digits of <paramref name="v"/> (below 10,000) at <paramref name="dst"/>.</summary>
        /// <param name="dst">The first digit's slot.</param>
        /// <param name="v">The value.</param>
        /// <param name="table">The two-digit table.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Put4(byte* dst, uint v, ref byte table)
        {
            uint hi = v / 100;
            int h = (int)hi * 2;
            int l = (int)(v - hi * 100) * 2;
            dst[0] = Unsafe.Add(ref table, h);
            dst[1] = Unsafe.Add(ref table, h + 1);
            dst[2] = Unsafe.Add(ref table, l);
            dst[3] = Unsafe.Add(ref table, l + 1);
        }

        /// <summary>
        ///     Writes exactly <paramref name="count"/> decimal digits of <paramref name="v"/> (zero-filled on the left) as
        ///     ASCII ending just before <paramref name="end"/>; <paramref name="v"/> must have at most that many digits.
        /// </summary>
        /// <param name="v">The value.</param>
        /// <param name="end">One past the last digit's slot.</param>
        /// <param name="count">The number of digits to write.</param>
        internal static void WriteDecimalBackwards(ulong v, byte* end, int count)
        {
            int n = WriteDecimalBackwards(v, end);
            for (byte* p = end - n; n < count; n++)
                *--p = (byte)'0';
        }
    }
}
