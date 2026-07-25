using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Vector-width overloads of the <see cref="NDFloatMath"/> float32 kernels, so the IL-emitted
    /// SIMD unary loop can run the same NumPy algorithm a register at a time.
    ///
    /// <para>Every step of <c>simd_exp_FLOAT</c> is elementwise, so lane <c>i</c> of
    /// <c>Exp(vector)</c> is bit-identical to <c>Exp(vector[i])</c> — and to NumPy, at any width.
    /// That is not an accident of this port: NumPy generates its own kernel from one template for
    /// both <c>__m256</c> and <c>__m512</c> for exactly the same reason. The one place NumPy uses a
    /// whole-vector branch (the denormal half of <c>fma_scalef_ps</c>) is kept, because it is not
    /// only structural: taking it unconditionally costs a second <c>vdivps</c> on every vector, and
    /// exp's output is subnormal only below about -87.3, so the branch is essentially never taken.
    /// Measured, folding it away cost ~0.17 ns/element — a fifth of the whole kernel.</para>
    ///
    /// <para><b>One deliberate re-spelling of NumPy's integer steps</b>, exactly equal over the
    /// reachable domain: NumPy forms the divisor as <c>cvtepi32_ps(1 &lt;&lt; quadDiff)</c>
    /// (a variable per-lane shift, <c>vpsllvd</c>, which is AVX2-only), while this builds the same
    /// power of two straight into the exponent field as <c>(quadDiff + 127) &lt;&lt; 23</c> — equal
    /// for all <c>quadDiff ∈ [0, 25]</c>, the range NumPy's own bounds allow, and expressible with
    /// the cross-platform constant-shift API.</para>
    ///
    /// <para><b>Hardware gate.</b> Only the FMA is width-specific; <see cref="IsExpVectorAccelerated"/>
    /// reports whether this host has it, and the kernel emitters consult that before choosing the
    /// vector loop (falling back to the scalar entry point, which is the same bits). The software
    /// FMA fallback below is not a shipping path — it exists so the wider overloads stay testable on
    /// a host that cannot execute them natively.</para>
    /// </summary>
    public static partial class NDFloatMath
    {
        /// <summary>
        /// Whether <c>Exp(Vector{<paramref name="vectorBits"/>}&lt;float&gt;)</c> lowers to hardware
        /// FMA on this machine. NumPy gates its own vector exp the same way (its kernel is compiled
        /// only for AVX2+FMA3 / AVX-512F targets, and hosts without them run the scalar libm).
        /// </summary>
        public static bool IsExpVectorAccelerated(int vectorBits) => vectorBits switch
        {
            512 => Avx512F.IsSupported,
            256 => Fma.IsSupported,
            128 => Fma.IsSupported,
            _ => false
        };

        /// <summary>NumPy 2.4.2's float32 exponential, 4 lanes at a time. See <see cref="Exp(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector128<float> Exp(Vector128<float> x)
        {
            var zero = Vector128<float>.Zero;

            var nanMask = ~Vector128.Equals(x, x);
            x = Vector128.ConditionalSelect(nanMask, zero, x);

            var xmaxMask = Vector128.GreaterThanOrEqual(x, Vector128.Create(ExpXMax));
            var xminMask = Vector128.LessThanOrEqual(x, Vector128.Create(ExpXMin));
            x = Vector128.ConditionalSelect(nanMask | xminMask | xmaxMask, zero, x);

            var magic = Vector128.Create(RintCvtMagic);
            var quadrant = MulAdd(x, Vector128.Create(Log2E), magic) - magic;

            var y = MulAdd(quadrant, Vector128.Create(CodyWaiteLogE2High), x);
            y = MulAdd(quadrant, Vector128.Create(CodyWaiteLogE2Low), y);
            y = MulAdd(quadrant, zero, y);

            var num = MulAdd(Vector128.Create(ExpP5), y, Vector128.Create(ExpP4));
            num = MulAdd(num, y, Vector128.Create(ExpP3));
            num = MulAdd(num, y, Vector128.Create(ExpP2));
            num = MulAdd(num, y, Vector128.Create(ExpP1));
            num = MulAdd(num, y, Vector128.Create(ExpP0));
            var den = MulAdd(Vector128.Create(ExpQ2), y, Vector128.Create(ExpQ1));
            den = MulAdd(den, y, Vector128.Create(ExpQ0));
            var poly = num / den;

            var minQuad = Vector128.Create(-125f);
            var denormMask = Vector128.LessThanOrEqual(quadrant, minQuad);
            if (Vector128.ExtractMostSignificantBits(denormMask) != 0)
            {
                // At least one lane scales below the smallest normal, so take NumPy's split:
                // apply 2^-125 through the exponent field and let a division make the subnormal.
                var quadDiff = Vector128.ConditionalSelect(denormMask, zero - (quadrant - minQuad), zero);
                var scaled = (poly.AsInt32() + Vector128.ShiftLeft(
                    Vector128.ConvertToInt32(Vector128.Max(quadrant, minQuad)), 23)).AsSingle();
                var twoPow = Vector128.ShiftLeft(
                    Vector128.ConvertToInt32(quadDiff) + Vector128.Create(127), 23).AsSingle();
                poly = scaled / twoPow;   // non-denormal lanes divide by exactly 1.0 — an identity
            }
            else
            {
                poly = (poly.AsInt32() + Vector128.ShiftLeft(Vector128.ConvertToInt32(quadrant), 23)).AsSingle();
            }

            poly = Vector128.ConditionalSelect(nanMask, Vector128.Create(CanonicalNaN), poly);
            poly = Vector128.ConditionalSelect(xmaxMask, Vector128.Create(float.PositiveInfinity), poly);
            return Vector128.ConditionalSelect(xminMask, zero, poly);
        }

        /// <summary>NumPy 2.4.2's float32 exponential, 8 lanes at a time. See <see cref="Exp(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Exp(Vector256<float> x)
        {
            var zero = Vector256<float>.Zero;

            var nanMask = ~Vector256.Equals(x, x);
            x = Vector256.ConditionalSelect(nanMask, zero, x);

            var xmaxMask = Vector256.GreaterThanOrEqual(x, Vector256.Create(ExpXMax));
            var xminMask = Vector256.LessThanOrEqual(x, Vector256.Create(ExpXMin));
            x = Vector256.ConditionalSelect(nanMask | xminMask | xmaxMask, zero, x);

            var magic = Vector256.Create(RintCvtMagic);
            var quadrant = MulAdd(x, Vector256.Create(Log2E), magic) - magic;

            var y = MulAdd(quadrant, Vector256.Create(CodyWaiteLogE2High), x);
            y = MulAdd(quadrant, Vector256.Create(CodyWaiteLogE2Low), y);
            y = MulAdd(quadrant, zero, y);

            var num = MulAdd(Vector256.Create(ExpP5), y, Vector256.Create(ExpP4));
            num = MulAdd(num, y, Vector256.Create(ExpP3));
            num = MulAdd(num, y, Vector256.Create(ExpP2));
            num = MulAdd(num, y, Vector256.Create(ExpP1));
            num = MulAdd(num, y, Vector256.Create(ExpP0));
            var den = MulAdd(Vector256.Create(ExpQ2), y, Vector256.Create(ExpQ1));
            den = MulAdd(den, y, Vector256.Create(ExpQ0));
            var poly = num / den;

            var minQuad = Vector256.Create(-125f);
            var denormMask = Vector256.LessThanOrEqual(quadrant, minQuad);
            if (Vector256.ExtractMostSignificantBits(denormMask) != 0)
            {
                // At least one lane scales below the smallest normal, so take NumPy's split:
                // apply 2^-125 through the exponent field and let a division make the subnormal.
                var quadDiff = Vector256.ConditionalSelect(denormMask, zero - (quadrant - minQuad), zero);
                var scaled = (poly.AsInt32() + Vector256.ShiftLeft(
                    Vector256.ConvertToInt32(Vector256.Max(quadrant, minQuad)), 23)).AsSingle();
                var twoPow = Vector256.ShiftLeft(
                    Vector256.ConvertToInt32(quadDiff) + Vector256.Create(127), 23).AsSingle();
                poly = scaled / twoPow;   // non-denormal lanes divide by exactly 1.0 — an identity
            }
            else
            {
                poly = (poly.AsInt32() + Vector256.ShiftLeft(Vector256.ConvertToInt32(quadrant), 23)).AsSingle();
            }

            poly = Vector256.ConditionalSelect(nanMask, Vector256.Create(CanonicalNaN), poly);
            poly = Vector256.ConditionalSelect(xmaxMask, Vector256.Create(float.PositiveInfinity), poly);
            return Vector256.ConditionalSelect(xminMask, zero, poly);
        }

        /// <summary>NumPy 2.4.2's float32 exponential, 16 lanes at a time. See <see cref="Exp(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector512<float> Exp(Vector512<float> x)
        {
            var zero = Vector512<float>.Zero;

            var nanMask = ~Vector512.Equals(x, x);
            x = Vector512.ConditionalSelect(nanMask, zero, x);

            var xmaxMask = Vector512.GreaterThanOrEqual(x, Vector512.Create(ExpXMax));
            var xminMask = Vector512.LessThanOrEqual(x, Vector512.Create(ExpXMin));
            x = Vector512.ConditionalSelect(nanMask | xminMask | xmaxMask, zero, x);

            var magic = Vector512.Create(RintCvtMagic);
            var quadrant = MulAdd(x, Vector512.Create(Log2E), magic) - magic;

            var y = MulAdd(quadrant, Vector512.Create(CodyWaiteLogE2High), x);
            y = MulAdd(quadrant, Vector512.Create(CodyWaiteLogE2Low), y);
            y = MulAdd(quadrant, zero, y);

            var num = MulAdd(Vector512.Create(ExpP5), y, Vector512.Create(ExpP4));
            num = MulAdd(num, y, Vector512.Create(ExpP3));
            num = MulAdd(num, y, Vector512.Create(ExpP2));
            num = MulAdd(num, y, Vector512.Create(ExpP1));
            num = MulAdd(num, y, Vector512.Create(ExpP0));
            var den = MulAdd(Vector512.Create(ExpQ2), y, Vector512.Create(ExpQ1));
            den = MulAdd(den, y, Vector512.Create(ExpQ0));
            var poly = num / den;

            var minQuad = Vector512.Create(-125f);
            var denormMask = Vector512.LessThanOrEqual(quadrant, minQuad);
            if (Vector512.ExtractMostSignificantBits(denormMask) != 0)
            {
                // At least one lane scales below the smallest normal, so take NumPy's split:
                // apply 2^-125 through the exponent field and let a division make the subnormal.
                var quadDiff = Vector512.ConditionalSelect(denormMask, zero - (quadrant - minQuad), zero);
                var scaled = (poly.AsInt32() + Vector512.ShiftLeft(
                    Vector512.ConvertToInt32(Vector512.Max(quadrant, minQuad)), 23)).AsSingle();
                var twoPow = Vector512.ShiftLeft(
                    Vector512.ConvertToInt32(quadDiff) + Vector512.Create(127), 23).AsSingle();
                poly = scaled / twoPow;   // non-denormal lanes divide by exactly 1.0 — an identity
            }
            else
            {
                poly = (poly.AsInt32() + Vector512.ShiftLeft(Vector512.ConvertToInt32(quadrant), 23)).AsSingle();
            }

            poly = Vector512.ConditionalSelect(nanMask, Vector512.Create(CanonicalNaN), poly);
            poly = Vector512.ConditionalSelect(xmaxMask, Vector512.Create(float.PositiveInfinity), poly);
            return Vector512.ConditionalSelect(xminMask, zero, poly);
        }

        // ---- fused multiply-add per width -------------------------------------------------
        // a*b + c with a SINGLE rounding. Vector{N}.FusedMultiplyAdd is .NET 9+, and NumSharp also
        // targets net8.0, so the hardware form goes through the x86 intrinsics that exist on both.
        // The scalar fallbacks keep every overload callable (and therefore testable) on a host whose
        // width is not accelerated; IsExpVectorAccelerated is what keeps them off the hot path.

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> MulAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c)
            => Fma.IsSupported ? Fma.MultiplyAdd(a, b, c) : SoftwareMulAdd(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> MulAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c)
            => Fma.IsSupported ? Fma.MultiplyAdd(a, b, c) : SoftwareMulAdd(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<float> MulAdd(Vector512<float> a, Vector512<float> b, Vector512<float> c)
            => Avx512F.IsSupported ? Avx512F.FusedMultiplyAdd(a, b, c) : SoftwareMulAdd(a, b, c);

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector128<float> SoftwareMulAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c)
        {
            Span<float> r = stackalloc float[Vector128<float>.Count];
            for (int i = 0; i < r.Length; i++)
                r[i] = MathF.FusedMultiplyAdd(a[i], b[i], c[i]);
            return Vector128.Create<float>(r);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector256<float> SoftwareMulAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c)
        {
            Span<float> r = stackalloc float[Vector256<float>.Count];
            for (int i = 0; i < r.Length; i++)
                r[i] = MathF.FusedMultiplyAdd(a[i], b[i], c[i]);
            return Vector256.Create<float>(r);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector512<float> SoftwareMulAdd(Vector512<float> a, Vector512<float> b, Vector512<float> c)
        {
            Span<float> r = stackalloc float[Vector512<float>.Count];
            for (int i = 0; i < r.Length; i++)
                r[i] = MathF.FusedMultiplyAdd(a[i], b[i], c[i]);
            return Vector512.Create<float>(r);
        }
    }
}
