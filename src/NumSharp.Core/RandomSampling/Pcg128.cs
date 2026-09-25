using System;
using System.Globalization;
using System.Numerics;

namespace NumSharp
{
    /// <summary>
    ///     The 128-bit LCG arithmetic shared by <see cref="PCG64"/> and <see cref="PCG64DXSM"/> — NumPy's
    ///     <c>src/pcg64/pcg64.{h,c}</c> seeding, advance and jump constants, over .NET's <see cref="UInt128"/>.
    /// </summary>
    internal static class Pcg128
    {
        /// <summary>NumPy's <c>PCG_DEFAULT_MULTIPLIER_128</c> = (2549297995355413924 &lt;&lt; 64) + 4865540595714422341.</summary>
        internal static readonly UInt128 DefaultMultiplier = new UInt128(2549297995355413924UL, 4865540595714422341UL);

        /// <summary>NumPy's <c>PCG_CHEAP_MULTIPLIER_128</c>: the 64-bit multiplier PCG64DXSM steps with (widened to 128 bits).</summary>
        internal const ulong CheapMultiplier = 0xda942042e4dd58b5UL;

        /// <summary>2**128 - 1: NumPy's <c>wrap_int(val, 128)</c> mask.</summary>
        private static readonly BigInteger Mask128 = (BigInteger.One << 128) - 1;

        /// <summary>
        ///     The jump distance of <c>jumped()</c>: <c>0x9e3779b97f4a7c15f39cc0605cedc835</c>, i.e. (phi - 1) * 2**128
        ///     (NumPy's <c>jump_inplace</c> step), as a non-negative integer.
        /// </summary>
        internal static readonly BigInteger JumpStep =
            BigInteger.Parse("09e3779b97f4a7c15f39cc0605cedc835", NumberStyles.HexNumber, CultureInfo.InvariantCulture);

        /// <summary>
        ///     NumPy's <c>pcg_setseq_128_srandom_r</c>: odd increment from the sequence selector, then a DEFAULT-multiplier
        ///     step, the state add, and another step — the seeding both PCG64 and PCG64DXSM use (<c>pcg64_set_seed</c>).
        /// </summary>
        /// <param name="initState">The initial state word.</param>
        /// <param name="initSeq">The stream selector (the increment is <c>2·initSeq + 1</c>).</param>
        /// <param name="state">The seeded state.</param>
        /// <param name="inc">The increment.</param>
        internal static void Srandom(UInt128 initState, UInt128 initSeq, out UInt128 state, out UInt128 inc)
        {
            inc = (initSeq << 1) | UInt128.One;
            state = UInt128.Zero;
            state = state * DefaultMultiplier + inc;
            state += initState;
            state = state * DefaultMultiplier + inc;
        }

        /// <summary>
        ///     NumPy's <c>pcg_advance_lcg_128</c>: jump an LCG ahead by <paramref name="delta"/> steps in O(log delta)
        ///     (Brown's algorithm: compose the step's multiplier/increment by repeated squaring).
        /// </summary>
        /// <param name="state">The current state.</param>
        /// <param name="delta">The number of steps (mod 2**128).</param>
        /// <param name="curMult">The step multiplier.</param>
        /// <param name="curPlus">The step increment.</param>
        /// <returns>The advanced state.</returns>
        internal static UInt128 AdvanceLcg(UInt128 state, UInt128 delta, UInt128 curMult, UInt128 curPlus)
        {
            UInt128 accMult = UInt128.One;
            UInt128 accPlus = UInt128.Zero;
            while (delta > UInt128.Zero)
            {
                if ((delta & UInt128.One) != UInt128.Zero)
                {
                    accMult *= curMult;
                    accPlus = accPlus * curMult + curPlus;
                }
                curPlus = (curMult + UInt128.One) * curPlus;
                curMult *= curMult;
                delta >>= 1;
            }
            return accMult * state + accPlus;
        }

        /// <summary>NumPy's <c>wrap_int(delta, 128)</c>: the delta reduced into <c>[0, 2**128)</c> (a negative delta wraps, i.e. steps backwards).</summary>
        /// <param name="delta">The requested number of steps (any integer).</param>
        /// <returns>The wrapped delta.</returns>
        internal static UInt128 WrapDelta(BigInteger delta) => (UInt128)(delta & Mask128);

        /// <summary>NumPy's <c>pcg_output_xsl_rr_128_64</c> (PCG64's output): xor-fold the halves, rotate by the top 6 bits.</summary>
        /// <param name="state">The 128-bit state (the NEW state — PCG64 steps first).</param>
        /// <returns>The 64-bit output.</returns>
        internal static ulong OutputXslRr(UInt128 state)
        {
            ulong hi = (ulong)(state >> 64);
            ulong lo = (ulong)state;
            ulong x = hi ^ lo;
            int rot = (int)(hi >> 58);
            return (x >> rot) | (x << ((-rot) & 63));
        }

        /// <summary>
        ///     NumPy's <c>pcg_output_cm_128_64</c> (PCG64DXSM's "double xorshift multiply" output), taken from the
        ///     PRE-iterated state: <c>hi ^= hi &gt;&gt; 32; hi *= M; hi ^= hi &gt;&gt; 48; hi *= (lo | 1)</c>.
        /// </summary>
        /// <param name="state">The 128-bit state (the OLD state — DXSM outputs before stepping).</param>
        /// <returns>The 64-bit output.</returns>
        internal static ulong OutputDxsm(UInt128 state)
        {
            ulong hi = (ulong)(state >> 64);
            ulong lo = (ulong)state | 1UL;
            hi ^= hi >> 32;
            hi *= CheapMultiplier;
            hi ^= hi >> 48;
            hi *= lo;
            return hi;
        }
    }
}
