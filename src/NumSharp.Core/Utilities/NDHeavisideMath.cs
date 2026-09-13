using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Scalar kernels backing NumSharp's <c>np.heaviside</c> binary ufunc — the Heaviside step function
    /// <c>heaviside(x1, x2)</c>, a port of NumPy's <c>npy_heaviside</c>
    /// (<c>numpy/_core/src/npymath/npy_math_internal.h.src</c>):
    /// <code>
    ///   heaviside(x, h0) = NaN   if x is NaN
    ///                    = h0    if x == 0   (includes -0.0, since IEEE -0.0 == 0.0)
    ///                    = 0     if x &lt; 0   (includes -inf)
    ///                    = 1     if x &gt; 0   (includes +inf)
    /// </code>
    ///
    /// <para><b>Why a from-scratch helper and not a library call.</b> There is no single BCL function for
    /// this — it is a four-way branch on <paramref name="x"/> alone. The branch ORDER matters and is
    /// NumPy's exactly: the NaN test comes first, so an <c>x</c> of any NaN sign/payload yields a result
    /// NaN, and only then does <c>x == 0</c> return <c>h0</c>. <c>h0</c> is consulted for the <c>x == 0</c>
    /// case ALONE, so for every other <c>x</c> the value of <c>h0</c> (even a NaN <c>h0</c>) is irrelevant,
    /// matching NumPy.</para>
    ///
    /// <para><b>Two NaN behaviours that are easy to get wrong, both bit-verified against NumPy 2.4.2.</b>
    /// (1) When <c>x</c> is NaN the result is the POSITIVE canonical quiet NaN NumPy stores
    /// (<c>NPY_NAN</c> = <c>0x7ff8…</c> / <c>NPY_NANF</c> = <c>0x7fc00000</c> / <c>NPY_HALF_NAN</c> =
    /// <c>0x7e00</c>) — REGARDLESS of <c>x</c>'s own NaN sign or payload; note .NET's <see cref="double.NaN"/>
    /// is the NEGATIVE quiet NaN (<c>0xfff8…</c>), so returning it would flip the sign bit, hence the
    /// explicit constants here. (2) When <c>x == 0</c> the result is <c>h0</c>'s EXACT bits, so a NaN
    /// <c>h0</c> keeps its own sign (<c>heaviside(0, -NaN)</c> is a negative NaN) and a <c>-0.0</c>
    /// <c>h0</c> keeps its sign — the opposite of case (1).</para>
    ///
    /// <para><b>float16 computes directly (no float32 detour needed).</b> NumPy's <c>ee->e</c> loop carries
    /// <c>astype={'e':'f'}</c> — it widens to float32, runs <c>npy_heavisidef</c>, and narrows back — but
    /// every heaviside output is an EXACT value (0, 1, the passed-through <c>h0</c>, or the canonical NaN),
    /// and a <see cref="Half"/> <c>h0</c> round-trips through float32 losslessly, so evaluating in
    /// <see cref="Half"/> gives byte-identical bits (verified 0-diff over the adversarial half grid).</para>
    ///
    /// <para><b>Decimal</b> (no NumPy analog, NumSharp's <c>gg->g</c> stand-in) has no NaN/inf state, so it
    /// carries only the <c>== 0</c> / <c>&lt; 0</c> / else split.</para>
    ///
    /// <para>The whole function is comparison-and-select with no transcendental, which is why the SIMD
    /// driver (<see cref="NDHeavisideMath.Simd"/> in the sibling partial) can vectorize it branchlessly
    /// and stay bit-identical to these scalars lane-for-lane — unlike NumPy, whose <c>heaviside</c> is a
    /// scalar <c>BINARY_LOOP</c> with no SIMD on any platform.</para>
    /// </summary>
    public static partial class NDHeavisideMath
    {
        /// <summary>NumPy's positive quiet NaN <c>NPY_NAN</c> (<c>0x7ff8000000000000</c>) — NOT .NET's
        /// negative <see cref="double.NaN"/> (<c>0xfff8…</c>). Returned whenever the step argument is NaN.</summary>
        internal static readonly double NanD = BitConverter.Int64BitsToDouble(unchecked((long)0x7ff8000000000000UL));

        /// <summary>NumPy's positive quiet NaN <c>NPY_NANF</c> (<c>0x7fc00000</c>), the float32 counterpart
        /// of <see cref="NanD"/>.</summary>
        internal static readonly float NanF = BitConverter.Int32BitsToSingle(unchecked((int)0x7fc00000));

        /// <summary>NumPy's positive quiet half NaN <c>NPY_HALF_NAN</c> (<c>0x7e00</c>).</summary>
        internal static readonly Half NanH = BitConverter.UInt16BitsToHalf(0x7e00);

        /// <summary>
        /// The Heaviside step function for float64 (NumPy's <c>dd->d</c> loop / <c>npy_heaviside</c>).
        /// </summary>
        /// <param name="x">The step argument: its sign selects the branch (NaN → NaN, 0 → <paramref name="h0"/>,
        /// negative → 0, positive → 1).</param>
        /// <param name="h0">The value returned when <paramref name="x"/> is exactly 0 (its bits pass through
        /// verbatim); ignored for every other <paramref name="x"/>.</param>
        /// <returns>The step value; the positive canonical NaN when <paramref name="x"/> is NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Heaviside(double x, double h0)
        {
            if (double.IsNaN(x)) return NanD;   // x's NaN sign/payload is discarded → positive NPY_NAN
            if (x == 0.0) return h0;            // -0.0 == 0.0, so -0.0 also returns h0's exact bits
            if (x < 0.0) return 0.0;            // includes -inf
            return 1.0;                         // x > 0, includes +inf
        }

        /// <summary>
        /// The Heaviside step function for float32 (NumPy's <c>ff->f</c> loop / <c>npy_heavisidef</c>).
        /// See <see cref="Heaviside(double,double)"/>; identical logic at single precision.
        /// </summary>
        /// <param name="x">The step argument selecting the branch.</param>
        /// <param name="h0">The <c>x == 0</c> fill (exact bits pass through).</param>
        /// <returns>The step value; the positive canonical float NaN when <paramref name="x"/> is NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float HeavisideF(float x, float h0)
        {
            if (float.IsNaN(x)) return NanF;
            if (x == 0f) return h0;
            if (x < 0f) return 0f;
            return 1f;
        }

        /// <summary>
        /// The Heaviside step function for float16 (NumPy's <c>ee->e</c> loop, <c>astype e->f</c>). Computed
        /// directly in <see cref="Half"/> because every output is an exact value that a Half can hold and
        /// <paramref name="h0"/> round-trips losslessly — see the type remarks.
        /// </summary>
        /// <param name="x">The step argument selecting the branch.</param>
        /// <param name="h0">The <c>x == 0</c> fill (exact bits pass through, NaN sign included).</param>
        /// <returns>The step value; the positive canonical half NaN when <paramref name="x"/> is NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Half HeavisideHalf(Half x, Half h0)
        {
            if (Half.IsNaN(x)) return NanH;
            if (x == (Half)0) return h0;
            if (x < (Half)0) return (Half)0;
            return (Half)1;
        }

        /// <summary>
        /// The Heaviside step function for decimal (NumSharp's extension; no NumPy dtype). Decimal has no
        /// NaN/inf, so only the <c>== 0</c> / <c>&lt; 0</c> / else split applies.
        /// </summary>
        /// <param name="x">The step argument selecting the branch.</param>
        /// <param name="h0">The value returned when <paramref name="x"/> is exactly 0.</param>
        /// <returns>0 for negative, 1 for positive, and <paramref name="h0"/> for zero.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal HeavisideDecimal(decimal x, decimal h0)
        {
            if (x == 0m) return h0;
            if (x < 0m) return 0m;
            return 1m;
        }
    }
}
