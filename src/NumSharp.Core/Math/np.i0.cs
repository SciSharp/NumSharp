using System;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;   // NDFloatMath — NumPy's own float32 exp kernel (simd_exp_FLOAT)

namespace NumSharp
{
    public static partial class np
    {
        // =====================================================================
        // np.i0 — modified Bessel function of the first kind, order 0 (I_0).
        // Port of NumPy 2.4.2 numpy/lib/_function_base_impl.py::i0:
        //
        //     x = asanyarray(x)
        //     if x.dtype.kind == 'c': raise TypeError("i0 not supported for complex values")
        //     if x.dtype.kind != 'f': x = x.astype(float)          # bool/int → float64
        //     x = abs(x)
        //     return piecewise(x, [x <= 8.0], [_i0_1, _i0_2])      # cephes Chebyshev split
        //
        // TWO behaviours are load-bearing and easy to get wrong:
        //
        //  (1) DTYPE follows the INPUT float precision, NOT a blanket float64.
        //      NumPy's piecewise/_chbevl chain runs in the input dtype under NEP50
        //      weak-scalar promotion (the float64 coefficients adopt the array's
        //      dtype), so i0(float32) is computed *in* float32 (i0(0f)=0.99999994,
        //      NOT 1.0) and i0(float16) *in* float16. Only bool/int/char promote to
        //      float64; complex is refused. A float64-computed-then-cast result
        //      diverges from NumPy by up to 3–5 ULP on ~50 % of float16/float32
        //      inputs — so each float precision needs its OWN native-precision path.
        //
        //  (2) The float32 exp must be NumPy's OWN kernel, not MathF.Exp. NumPy's
        //      float32 i0 calls the float32 exp loop (simd_exp_FLOAT), which differs
        //      from the ~correctly-rounded MathF.Exp on ~39 % of inputs; NumSharp
        //      ports that exact kernel as NDFloatMath.Exp (the same one the fused
        //      Exp node emits). float16 exp/sqrt are the BCL Half.Exp/Half.Sqrt,
        //      which are byte-identical to NumPy's half loop; float64 exp/sqrt are
        //      Math.Exp/Math.Sqrt (== NumPy's ucrtbase, already bit-exact at f8).
        //      Every chbevl step is pure add/sub/mul, IEEE-exact per op at each
        //      width, so the recurrence matches NumPy bit-for-bit once the width is
        //      fixed. VERIFIED 0 bit-diffs vs NumPy 2.4.2 across 10 518 adversarial
        //      float64 AND float32 inputs (specials/±inf/NaN/subnormals/overflow)
        //      and ALL 65 536 float16 bit patterns.
        //
        // IMPLEMENTATION — one fused np.evaluate pass per precision:
        //   The cephes routine is a data-dependent Chebyshev recurrence (a Clenshaw
        //   loop over 30 / 25 coefficients with a per-element |x| <= 8 branch). It
        //   CANNOT be an unrolled NDExpr tree — Clenshaw reuses b0 as both b1 and b2,
        //   so a tree with no common-subexpression sharing explodes Fibonacci-like
        //   and stack-overflows the emitter. Instead the whole scalar routine rides
        //   ONE NDExpr.Call node (the np.kaiser pattern), so np.evaluate drives it
        //   through NDIter: one pass, every memory layout (C/F/strided/broadcast/
        //   negative-stride/sliced), no intermediate arrays (NumPy materializes ~30).
        //   The per-precision delegate is a static readonly field so the compiled
        //   kernel is cached once (a method group would JIT a fresh kernel per call).
        //
        // PERF (NPY/NS, Release, best-of-30 warm): the fused pass is COMPUTE-bound
        // (one scalar Clenshaw per element, zero intermediate arrays) where NumPy is
        // MEMORY-bound (~35 vectorized array passes: abs + the x<=8 mask + piecewise's
        // subset gather/scatter + exp + 30 chbevl passes). float64 — the dtype real
        // Bessel work uses — WINS at every size: 3.8× (1K, NumPy's Python overhead),
        // 1.9× (100K), 9.4× (10M, NumPy's traffic dominates). float32 wins at 1K
        // (~1.0×) and 10M (4.4×) but is ~0.84× at 100K, and float16 is ~0.4–0.5×,
        // because at cache-resident sizes NumPy's SIMD passes stay in L2/L3 and beat a
        // per-element scalar Clenshaw — and float16 additionally has NO BCL vector
        // arithmetic (Vector<Half> throws), the same ceiling Half hits library-wide.
        // Beating those two cells would need i0 as a first-class SIMD ILKernelGenerator
        // op (a 55-coefficient vector Clenshaw + width-specific NumPy exp), which is
        // disproportionate for a function NumPy itself documents as "not a proper
        // ufunc — use scipy". Correctness (bit-exact parity) is the gate and is met.
        //
        // Cephes coefficients (_i0A / _i0B) and the double scalar helper BesselI0 /
        // Chbevl live in np.windows.cs (np.kaiser is their other consumer); this file
        // adds the float32 / float16 / decimal precision variants.
        // =====================================================================

        /// <summary>
        ///     Modified Bessel function of the first kind, order 0 (I₀), evaluated element-wise —
        ///     the taper behind <see cref="kaiser(double,double)"/>, exposed as NumPy's <c>np.i0</c>.
        /// </summary>
        /// <param name="x">
        ///     Argument array. The float precision is PRESERVED and the whole cephes routine runs at
        ///     that precision (float16→float16, float32→float32, float64→float64), matching NumPy's
        ///     NEP50 weak-scalar promotion; bool / every integer width / Char promote to float64;
        ///     Decimal is preserved via the double bridge (a NumSharp extension — NumPy has no decimal);
        ///     Complex is refused (NumPy raises the same <c>TypeError</c>).
        /// </param>
        /// <returns>
        ///     A fresh array of <paramref name="x"/>'s shape holding I₀ at each element. The dtype is the
        ///     input float dtype (float64 for bool/int/Char inputs, Decimal for a Decimal input).
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.i0.html
        ///     <para>
        ///     Unlike a ufunc, <c>i0</c> has NO <c>out</c>/<c>where</c>/<c>dtype</c> parameters — the
        ///     signature is exactly <c>i0(x)</c>. |x| ≥ ~710 (float64) overflows <c>exp</c> to +∞ and
        ///     non-finite inputs propagate NaN/∞ exactly as NumPy's chain does.
        ///     </para>
        ///     <para>
        ///     Bit-identical to NumPy 2.4.2 for every float64/float32/float16 input (the shared cephes
        ///     recurrence is IEEE-exact per op at each width, and exp is NumPy's own kernel per width).
        ///     A Decimal input whose magnitude overflows the double→decimal bridge (I₀ exceeds
        ///     <see cref="decimal.MaxValue"/> at |x| ≳ 72) throws <see cref="OverflowException"/>, exactly
        ///     as the other decimal transcendentals do.
        ///     </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="IncorrectTypeException"><paramref name="x"/> is Complex — I₀ has no complex loop (NumPy's <c>TypeError("i0 not supported for complex values")</c>).</exception>
        /// <exception cref="OverflowException"><paramref name="x"/> is Decimal and I₀ overflows the decimal range (double→decimal bridge limit).</exception>
        public static NDArray i0(NDArray x)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));

            // Dispatch by PRECISION DOMAIN, not per-dtype: the cephes recurrence is bit-exact only
            // when run at the input's own float width (see the header). Each arm is a genuinely
            // different arithmetic — mirroring the kernel's own EmitUnaryHalfOperation / decimal /
            // complex split — not a mechanical per-NPTypeCode copy. bool/int/Char fall to the double
            // arm, where NDExpr.Call converts the integer operand to double at the call edge (NumPy's
            // astype(float) before the computation), so no explicit cast is needed.
            switch (x.typecode)
            {
                case NPTypeCode.Complex:
                    throw new IncorrectTypeException("i0 not supported for complex values");
                case NPTypeCode.Half:
                    return evaluate(NDExpr.Call(_i0Half, (NDExpr)x));
                case NPTypeCode.Single:
                    return evaluate(NDExpr.Call(_i0Single, (NDExpr)x));
                case NPTypeCode.Decimal:
                    return evaluate(NDExpr.Call(_i0Decimal, (NDExpr)x));
                default:
                    return evaluate(NDExpr.Call(_i0Double, (NDExpr)x));
            }
        }

        // Per-precision delegates held as static readonly fields: NDExpr.Call keys its compiled kernel
        // by delegate identity, so a stable field yields ONE cached kernel per precision (a method
        // group would allocate a fresh delegate — and JIT a fresh kernel — on every call).
        private static readonly Func<double, double> _i0Double = BesselI0;          // reuse the np.windows.cs helper
        private static readonly Func<float, float> _i0Single = BesselI0Single;
        private static readonly Func<Half, Half> _i0Half = BesselI0Half;
        private static readonly Func<decimal, decimal> _i0Decimal = BesselI0Decimal;

        // ------------------------------------------------------------------
        // float32 — exp via NumPy's own simd_exp_FLOAT port (NDFloatMath.Exp),
        // sqrt via MathF.Sqrt (IEEE correctly-rounded == NumPy), chbevl in float32.
        // ------------------------------------------------------------------

        /// <summary>float32 Clenshaw evaluation of a Chebyshev series (cephes <c>chbevl</c>), each op rounded to float32.</summary>
        /// <param name="x">The (recentred) evaluation point.</param>
        /// <param name="vals">The float64 Chebyshev coefficients; each is adopted to float32 per NumPy's NEP50 weak-scalar rule (<c>+ (float)vals[i]</c>).</param>
        /// <returns>The series value in float32.</returns>
        private static float ChbevlSingle(float x, double[] vals)
        {
            // The float64 coefficients ride into a float32 recurrence exactly as NumPy's ufunc chain
            // does: `float32_array * b1 - b2 + vals[i]` promotes the weak float64 scalar DOWN to
            // float32, so each `+ (float)vals[i]` matches, and every op is float32-rounded.
            float b0 = (float)vals[0], b1 = 0.0f, b2 = 0.0f;
            for (int i = 1; i < vals.Length; i++)
            {
                b2 = b1;
                b1 = b0;
                b0 = x * b1 - b2 + (float)vals[i];
            }
            return 0.5f * (b0 - b2);
        }

        /// <summary>
        ///     I₀ in float32 — bit-identical to NumPy's float32 <c>i0</c>. exp is NumPy's own float32
        ///     kernel (<see cref="NDFloatMath.Exp(float)"/> = simd_exp_FLOAT), NOT MathF.Exp.
        /// </summary>
        /// <param name="x">The argument (a single float32 element, supplied per element by the fused kernel).</param>
        /// <returns>I₀(<paramref name="x"/>) in float32.</returns>
        private static float BesselI0Single(float x)
        {
            x = MathF.Abs(x);
            if (x <= 8.0f)
                return NDFloatMath.Exp(x) * ChbevlSingle(x / 2.0f - 2.0f, _i0A);
            return NDFloatMath.Exp(x) * ChbevlSingle(32.0f / x - 2.0f, _i0B) / MathF.Sqrt(x);
        }

        // ------------------------------------------------------------------
        // float16 — every op in Half (each Half operator = round-to-half of the
        // float32 result, matching NumPy's npy_half loop); exp/sqrt via the BCL
        // Half.Exp/Half.Sqrt (byte-identical to NumPy's half loop, per the kernel).
        // ------------------------------------------------------------------

        /// <summary>float16 Clenshaw evaluation of a Chebyshev series (cephes <c>chbevl</c>), each op rounded to float16.</summary>
        /// <param name="x">The (recentred) evaluation point.</param>
        /// <param name="vals">The float64 Chebyshev coefficients; each is adopted to float16 (<c>+ (Half)vals[i]</c>), matching NumPy's weak-scalar promotion (many tiny coefficients round to 0 in float16 — deterministic, and NumPy's own behaviour).</param>
        /// <returns>The series value in float16.</returns>
        private static Half ChbevlHalf(Half x, double[] vals)
        {
            // .NET Half arithmetic operators compute (Half)((float)a OP (float)b) — exactly NumPy's
            // npy_half model — so the recurrence is bit-identical to NumPy's float16 chbevl.
            Half b0 = (Half)vals[0], b1 = (Half)0, b2 = (Half)0;
            for (int i = 1; i < vals.Length; i++)
            {
                b2 = b1;
                b1 = b0;
                b0 = x * b1 - b2 + (Half)vals[i];
            }
            return (Half)0.5 * (b0 - b2);
        }

        /// <summary>
        ///     I₀ in float16 — bit-identical to NumPy's float16 <c>i0</c> over all 65 536 inputs.
        ///     exp/sqrt are the BCL <see cref="Half.Exp(Half)"/>/<see cref="Half.Sqrt(Half)"/>, which
        ///     the kernel already proves byte-identical to NumPy's half loop.
        /// </summary>
        /// <param name="x">The argument (a single float16 element, supplied per element by the fused kernel).</param>
        /// <returns>I₀(<paramref name="x"/>) in float16.</returns>
        private static Half BesselI0Half(Half x)
        {
            x = Half.Abs(x);
            if (x <= (Half)8)
                return Half.Exp(x) * ChbevlHalf(x / (Half)2 - (Half)2, _i0A);
            return Half.Exp(x) * ChbevlHalf((Half)32 / x - (Half)2, _i0B) / Half.Sqrt(x);
        }

        // ------------------------------------------------------------------
        // Decimal — NumSharp extension (no NumPy analog). Computed through the
        // double bridge and returned as decimal, matching the sibling decimal
        // transcendentals (np.sinc etc.). Overflows the decimal range at |x| ≳ 72.
        // ------------------------------------------------------------------

        /// <summary>I₀ for a Decimal input via the double→decimal bridge (a NumSharp extension; NumPy has no decimal dtype).</summary>
        /// <param name="x">The argument (a single Decimal element, supplied per element by the fused kernel).</param>
        /// <returns>I₀(<paramref name="x"/>) as a Decimal.</returns>
        /// <exception cref="OverflowException">I₀(<paramref name="x"/>) exceeds <see cref="decimal.MaxValue"/> (|x| ≳ 72) — the shared decimal→double transcendental bridge limit.</exception>
        private static decimal BesselI0Decimal(decimal x) => (decimal)BesselI0((double)x);
    }
}
