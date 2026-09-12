using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Vector-width driver for <see cref="NDHypotMath.Hypot(double,double)"/>. NumPy's <c>hypot</c> is a
    /// SCALAR <c>BINARY_LOOP</c> (no SIMD on any platform), so vectorizing Borges' correctly-rounded
    /// algorithm — which is nothing but FMA / sqrt / min-max / add — beats NumPy several-fold while staying
    /// bit-identical to the scalar kernel (every step is elementwise, so lane <c>i</c> equals
    /// <c>Hypot(x[i], y[i])</c>). Only the SAFE MIDDLE (|max operand| in [2^-500, 2^500], where the square
    /// stays a normal double) is vectorized; the rare lanes that need magnitude scaling or are non-finite /
    /// zero are masked out and completed by the scalar kernel, so the fast path never changes a result bit.
    /// </summary>
    public static partial class NDHypotMath
    {
        // Squaring stays in the normal range for |v| in [2^-500, 2^500]; outside it Borges needs its power
        // -of-2 rescale, which the scalar path does. These bounds also fence off 0/subnormal/inf/NaN (a NaN
        // fails both compares, so the "safe" AND is false and the lane is scalarised).
        private const double SafeLo = 3.05493636349960e-151;   // 2^-500
        private const double SafeHi = 3.27339060789614e+150;   // 2^+500

        /// <summary>Whether the Vector256 hypot fast path can run (needs AVX2 FMA + hardware acceleration).</summary>
        public static bool SimdAvailable => Fma.IsSupported && Vector256.IsHardwareAccelerated;

        // Borges' correctly-rounded core, four doubles at a time. Assumes each lane's max operand is in the
        // safe middle (the driver masks the rest). Bit-identical to NDHypotMath.Core lane-by-lane: the FMA
        // (Fma.MultiplySubtract == Math.FusedMultiplyAdd with a negated addend) and Vector256.Sqrt are the
        // same hardware ops the scalar path uses.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<double> Core256(Vector256<double> hi, Vector256<double> lo)
        {
            var xx = hi * hi;
            var e_xx = Fma.MultiplySubtract(hi, hi, xx);          // hi*hi - xx  (exact low bits)
            var yy = lo * lo;
            var e_yy = Fma.MultiplySubtract(lo, lo, yy);
            var s = xx + yy;
            var e_s = yy - (s - xx);                              // Fast2Sum error (xx>=yy)
            var sigma = e_xx + e_yy + e_s;
            var h = Vector256.Sqrt(s);
            var hh = h * h;
            var e_hh = Fma.MultiplySubtract(h, h, hh);
            var residual = (s - hh) + sigma - e_hh;              // (hi^2+lo^2) - h^2
            return h + residual / (h + h);                       // one Newton step -> correctly rounded
        }

        private static readonly Vector256<double> AbsMask256 =
            Vector256.Create(long.MaxValue).AsDouble();          // 0x7FFF... clears the sign bit

        /// <summary>
        /// hypot over four lanes: vectorized Core for safe-middle lanes, scalar completion for the rest.
        /// <paramref name="x"/>/<paramref name="y"/> are the raw operands; the result is written and the
        /// caller need not inspect it further.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<double> Hypot256(Vector256<double> x, Vector256<double> y,
            double* xs, double* ys, long i)
        {
            var a = Vector256.BitwiseAnd(x, AbsMask256);         // |x|
            var b = Vector256.BitwiseAnd(y, AbsMask256);         // |y|
            var hi = Vector256.Max(a, b);
            var lo = Vector256.Min(a, b);
            var res = Core256(hi, lo);

            // safe <=> hi in [SafeLo, SafeHi]; a NaN hi fails both compares -> unsafe (scalarised).
            var safe = Vector256.GreaterThanOrEqual(hi, Vector256.Create(SafeLo))
                     & Vector256.LessThanOrEqual(hi, Vector256.Create(SafeHi));
            uint mask = Vector256.ExtractMostSignificantBits(safe);
            if (mask != 0b1111u)
            {
                // Complete the unsafe lanes with the scalar kernel (rare: extremes / 0 / non-finite).
                for (int lane = 0; lane < 4; lane++)
                    if ((mask & (1u << lane)) == 0)
                        res = res.WithElement(lane, Hypot(xs[i + lane], ys[i + lane]));
            }
            return res;
        }

        // =====================================================================
        // float64 contiguous: r[i] = hypot(a[i], b[i])
        // =====================================================================
        public static unsafe void HypotContiguousF64(double* a, double* b, double* r, long n)
        {
            long i = 0;
            if (SimdAvailable)
                for (; i <= n - 4; i += 4)
                    Hypot256(Vector256.Load(a + i), Vector256.Load(b + i), a, b, i).Store(r + i);
            for (; i < n; i++)
                r[i] = Hypot(a[i], b[i]);
        }

        // float64 scalar-broadcast: r[i] = hypot(a[i], scalar)
        public static unsafe void HypotScalarF64(double* a, double scalar, double* r, long n)
        {
            long i = 0;
            if (SimdAvailable)
            {
                var yv = Vector256.Create(scalar);
                // Prefill a tiny scratch so the scalar-completion path can read the broadcast operand.
                double* ys = stackalloc double[4] { scalar, scalar, scalar, scalar };
                for (; i <= n - 4; i += 4)
                    Hypot256(Vector256.Load(a + i), yv, a + i, ys, 0).Store(r + i);
            }
            for (; i < n; i++)
                r[i] = Hypot(a[i], scalar);
        }

        // =====================================================================
        // float32 contiguous: widen a quad to double (VCVTPS2PD), Core, narrow (VCVTPD2PS). Byte-exact
        // with NumPy's hypotf — rounding the correctly-rounded double down to float coincides with it.
        // =====================================================================
        public static unsafe void HypotContiguousF32(float* a, float* b, float* r, long n)
        {
            long i = 0;
            if (SimdAvailable)
                for (; i <= n - 4; i += 4)
                {
                    var xv = Avx.ConvertToVector256Double(Vector128.Load(a + i));   // 4 f32 -> 4 f64
                    var yv = Avx.ConvertToVector256Double(Vector128.Load(b + i));
                    var res = Hypot256F32(xv, yv, a, b, i);
                    Avx.ConvertToVector128Single(res).Store(r + i);                 // 4 f64 -> 4 f32
                }
            for (; i < n; i++)
                r[i] = HypotF(a[i], b[i]);
        }

        public static unsafe void HypotScalarF32(float* a, float scalar, float* r, long n)
        {
            long i = 0;
            if (SimdAvailable)
            {
                var yv = Vector256.Create((double)scalar);
                for (; i <= n - 4; i += 4)
                {
                    var xv = Avx.ConvertToVector256Double(Vector128.Load(a + i));
                    var res = Hypot256F32Scalar(xv, yv, a + i, scalar);
                    Avx.ConvertToVector128Single(res).Store(r + i);
                }
            }
            for (; i < n; i++)
                r[i] = HypotF(a[i], scalar);
        }

        // float32 quad already widened to double: Core + safe mask, scalar completion reads the f32 sources.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<double> Hypot256F32(Vector256<double> x, Vector256<double> y,
            float* xs, float* ys, long i)
        {
            var a = Vector256.BitwiseAnd(x, AbsMask256);
            var b = Vector256.BitwiseAnd(y, AbsMask256);
            var hi = Vector256.Max(a, b);
            var lo = Vector256.Min(a, b);
            var res = Core256(hi, lo);
            var safe = Vector256.GreaterThanOrEqual(hi, Vector256.Create(SafeLo))
                     & Vector256.LessThanOrEqual(hi, Vector256.Create(SafeHi));
            uint mask = Vector256.ExtractMostSignificantBits(safe);
            if (mask != 0b1111u)
                for (int lane = 0; lane < 4; lane++)
                    if ((mask & (1u << lane)) == 0)
                        res = res.WithElement(lane, (double)HypotF(xs[i + lane], ys[i + lane]));
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<double> Hypot256F32Scalar(Vector256<double> x, Vector256<double> y,
            float* xs, float scalar)
        {
            var a = Vector256.BitwiseAnd(x, AbsMask256);
            var b = Vector256.BitwiseAnd(y, AbsMask256);
            var hi = Vector256.Max(a, b);
            var lo = Vector256.Min(a, b);
            var res = Core256(hi, lo);
            var safe = Vector256.GreaterThanOrEqual(hi, Vector256.Create(SafeLo))
                     & Vector256.LessThanOrEqual(hi, Vector256.Create(SafeHi));
            uint mask = Vector256.ExtractMostSignificantBits(safe);
            if (mask != 0b1111u)
                for (int lane = 0; lane < 4; lane++)
                    if ((mask & (1u << lane)) == 0)
                        res = res.WithElement(lane, (double)HypotF(xs[lane], scalar));
            return res;
        }
    }
}
