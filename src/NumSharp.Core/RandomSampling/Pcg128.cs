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

        /// <summary>The high limb of <see cref="DefaultMultiplier"/> (a JIT immediate in the hot loops).</summary>
        internal const ulong DefaultMulHi = 2549297995355413924UL;

        /// <summary>The low limb of <see cref="DefaultMultiplier"/>.</summary>
        internal const ulong DefaultMulLo = 4865540595714422341UL;

        /// <summary>
        ///     <c>M**2</c>, <c>M**3</c>, <c>M**4</c> (mod 2**128) for the default multiplier <c>M</c>: with them the states two,
        ///     three and four steps ahead come straight from the current one (<c>s_k = M**k * s + inc * (1 + M + … + M**(k-1))</c>),
        ///     so a bulk fill computes four states in parallel instead of one long multiply chain.
        /// </summary>
        private static readonly UInt128 DefaultMultiplier2 = DefaultMultiplier * DefaultMultiplier;

        /// <inheritdoc cref="DefaultMultiplier2"/>
        private static readonly UInt128 DefaultMultiplier3 = DefaultMultiplier2 * DefaultMultiplier;

        /// <inheritdoc cref="DefaultMultiplier2"/>
        private static readonly UInt128 DefaultMultiplier4 = DefaultMultiplier3 * DefaultMultiplier;

        /// <summary>
        ///     The increment factors of the multi-step jump: <c>1 + M</c>, <c>1 + M + M**2</c>, <c>1 + M + M**2 + M**3</c>
        ///     (mod 2**128) — times the generator's increment they give the additive term of a 2/3/4-step advance.
        /// </summary>
        private static readonly UInt128 DefaultIncFactor2 = UInt128.One + DefaultMultiplier;

        /// <inheritdoc cref="DefaultIncFactor2"/>
        private static readonly UInt128 DefaultIncFactor3 = DefaultIncFactor2 + DefaultMultiplier2;

        /// <inheritdoc cref="DefaultIncFactor2"/>
        private static readonly UInt128 DefaultIncFactor4 = DefaultIncFactor3 + DefaultMultiplier3;

        // The multi-step constants as limbs. `static readonly` primitives are folded to immediates by the tiered JIT.
        internal static readonly ulong Mul2Hi = (ulong)(DefaultMultiplier2 >> 64), Mul2Lo = (ulong)DefaultMultiplier2;
        internal static readonly ulong Mul3Hi = (ulong)(DefaultMultiplier3 >> 64), Mul3Lo = (ulong)DefaultMultiplier3;
        internal static readonly ulong Mul4Hi = (ulong)(DefaultMultiplier4 >> 64), Mul4Lo = (ulong)DefaultMultiplier4;
        internal static readonly ulong IncFactor2Hi = (ulong)(DefaultIncFactor2 >> 64), IncFactor2Lo = (ulong)DefaultIncFactor2;
        internal static readonly ulong IncFactor3Hi = (ulong)(DefaultIncFactor3 >> 64), IncFactor3Lo = (ulong)DefaultIncFactor3;
        internal static readonly ulong IncFactor4Hi = (ulong)(DefaultIncFactor4 >> 64), IncFactor4Lo = (ulong)DefaultIncFactor4;

        /// <summary>The low 128 bits of <c>a * b</c> over 64-bit limbs.</summary>
        /// <param name="ah">High limb of a.</param>
        /// <param name="al">Low limb of a.</param>
        /// <param name="bh">High limb of b.</param>
        /// <param name="bl">Low limb of b.</param>
        /// <param name="rh">High limb of the product.</param>
        /// <param name="rl">Low limb of the product.</param>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal static void Mul(ulong ah, ulong al, ulong bh, ulong bl, out ulong rh, out ulong rl)
        {
            rh = Math.BigMul(al, bl, out rl);
            rh += ah * bl + al * bh;
        }

        /// <summary>
        ///     One LCG step over limbs, <c>s * m + c</c> (mod 2**128): the full 64x64 low product, the two cross products
        ///     (their high halves fall off the 128-bit result), and a carried add.
        /// </summary>
        /// <param name="sh">High limb of the state.</param>
        /// <param name="sl">Low limb of the state.</param>
        /// <param name="mh">High limb of the multiplier.</param>
        /// <param name="ml">Low limb of the multiplier.</param>
        /// <param name="ch">High limb of the addend.</param>
        /// <param name="cl">Low limb of the addend.</param>
        /// <param name="rh">High limb of the result (may alias an input variable — inputs are read first).</param>
        /// <param name="rl">Low limb of the result.</param>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal static void MulAdd(ulong sh, ulong sl, ulong mh, ulong ml, ulong ch, ulong cl, out ulong rh, out ulong rl)
        {
            ulong ph = Math.BigMul(sl, ml, out ulong pl);
            ph += sh * ml + sl * mh;
            ulong lo = pl + cl;
            rh = ph + ch + (lo < pl ? 1UL : 0UL);
            rl = lo;
        }

        /// <summary><see cref="OutputXslRr"/> over limbs.</summary>
        /// <param name="hi">High limb of the (new) state.</param>
        /// <param name="lo">Low limb of the (new) state.</param>
        /// <returns>The 64-bit output.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal static ulong XslRr(ulong hi, ulong lo) => BitOperations.RotateRight(hi ^ lo, (int)(hi >> 58));

        /// <summary><see cref="OutputDxsm"/> over limbs.</summary>
        /// <param name="hi">High limb of the (pre-iterated) state.</param>
        /// <param name="lo">Low limb of the (pre-iterated) state.</param>
        /// <returns>The 64-bit output.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal static ulong Dxsm(ulong hi, ulong lo)
        {
            lo |= 1;
            hi ^= hi >> 32;
            hi *= CheapMultiplier;
            hi ^= hi >> 48;
            return hi * lo;
        }

        /// <summary>One PCG64DXSM step over limbs, <c>s * CheapMultiplier + inc</c>: the multiplier's zero high limb drops a product.</summary>
        /// <param name="sh">High limb of the state.</param>
        /// <param name="sl">Low limb of the state.</param>
        /// <param name="ih">High limb of the increment.</param>
        /// <param name="il">Low limb of the increment.</param>
        /// <param name="rh">High limb of the next state.</param>
        /// <param name="rl">Low limb of the next state.</param>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal static void StepCheap(ulong sh, ulong sl, ulong ih, ulong il, out ulong rh, out ulong rl)
        {
            ulong ph = Math.BigMul(sl, CheapMultiplier, out ulong pl);
            ph += sh * CheapMultiplier;
            ulong lo = pl + il;
            rh = ph + ih + (lo < pl ? 1UL : 0UL);
            rl = lo;
        }

        /// <summary>NumPy's <c>PCG_CHEAP_MULTIPLIER_128</c>: the 64-bit multiplier PCG64DXSM steps with (widened to 128 bits).</summary>
        internal const ulong CheapMultiplier = 0xda942042e4dd58b5UL;

        /// <summary><c>C**2</c> (mod 2**128) for the cheap multiplier <c>C</c>: PCG64DXSM's two-step jump multiplier.</summary>
        private static readonly UInt128 CheapMultiplier2 = (UInt128)CheapMultiplier * CheapMultiplier;

        /// <summary><c>1 + C</c>: times the increment it gives the additive term of PCG64DXSM's two-step jump.</summary>
        private static readonly UInt128 CheapIncFactor2 = UInt128.One + CheapMultiplier;

        // The two-step constants as limbs (folded to immediates by the tiered JIT).
        internal static readonly ulong CheapMul2Hi = (ulong)(CheapMultiplier2 >> 64), CheapMul2Lo = (ulong)CheapMultiplier2;
        internal static readonly ulong CheapIncFactor2Hi = (ulong)(CheapIncFactor2 >> 64), CheapIncFactor2Lo = (ulong)CheapIncFactor2;

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
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
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
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
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
