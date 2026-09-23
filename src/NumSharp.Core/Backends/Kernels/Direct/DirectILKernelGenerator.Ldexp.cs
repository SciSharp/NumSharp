using System;
using System.Runtime.CompilerServices;

// =============================================================================
// DirectILKernelGenerator.Ldexp — np.ldexp kernels (compose x * 2^exp)
// =============================================================================
//
// ldexp(x, exp) == x * 2^exp, the inverse of frexp. NumPy's loop signatures are ei->e / fi->f
// / di->d (int32 exponent) plus e{int64}->e / f{int64}->f / d{int64}->d (int64 exponent). The
// exponent is read as an integer and CLAMPED to the C-int range: a value outside [int.MinValue,
// int.MaxValue] behaves as ±MAX_INT, which then overflows/underflows exactly as NumPy's int64
// loop specifies (large +exp -> ±inf, large -exp -> ±0). NumSharp always widens the exponent to
// int64 upstream, so ONE clamp here covers both the int32 and int64 NumPy loops.
//
// Math.ScaleB IS ldexp (x * 2^n, correctly rounded, specials per IEEE) and is bit-identical to
// win-amd64 NumPy 2.4.2's npy_ldexp for every input (verified). There is no vector ScaleB
// intrinsic, so the loop is scalar — which is optimal here anyway: at scale ldexp is memory
// -bandwidth bound (three streams: x in, exp in, result out), and the scalar ScaleB loop already
// runs several-fold faster than NumPy's own scalar ldexp. Half is widen-compute-narrow (NumPy's
// HALF_ldexp) and Decimal is the double bridge (a NumSharp extension).
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        ///     Clamp an int64 exponent to the C-int range, reproducing NumPy's int64 ldexp loop: a value
        ///     beyond ±<see cref="int.MaxValue"/> is pinned to the range end (positive -&gt; MAX_INT,
        ///     negative -&gt; MIN_INT), which <see cref="Math.ScaleB(double,int)"/> then turns into ±inf / ±0.
        /// </summary>
        /// <param name="e">The (possibly out-of-int-range) exponent.</param>
        /// <returns>The exponent clamped to <c>[int.MinValue, int.MaxValue]</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int ClampExp(long e) => e > int.MaxValue ? int.MaxValue : (e < int.MinValue ? int.MinValue : (int)e);

        // ---- int32-exponent kernels (the common case: bool/int8..int32 exponents that ALWAYS fit the
        // C-int range, so no clamp is needed — read directly, half the exponent traffic of the int64
        // path). Used whenever the exponent's source dtype is int32 or narrower. ----

        /// <summary>Element-wise <c>ldexp</c> over contiguous double + int32-exponent buffers (no clamp needed).</summary>
        /// <param name="x">Mantissa/base values.</param>
        /// <param name="exp">Per-element exponents (int32, always in the C-int range).</param>
        /// <param name="result">Receives <c>x[i] * 2^exp[i]</c>.</param>
        /// <param name="n">Element count.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void LdexpHelper(double* x, int* exp, double* result, long n)
        {
            for (long i = 0; i < n; i++) result[i] = Math.ScaleB(x[i], exp[i]);
        }

        /// <summary>Element-wise <c>ldexp</c> over contiguous float + int32-exponent buffers (no clamp needed).</summary>
        /// <param name="x">Mantissa/base values.</param>
        /// <param name="exp">Per-element exponents (int32).</param>
        /// <param name="result">Receives <c>x[i] * 2^exp[i]</c>.</param>
        /// <param name="n">Element count.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void LdexpHelper(float* x, int* exp, float* result, long n)
        {
            for (long i = 0; i < n; i++) result[i] = MathF.ScaleB(x[i], exp[i]);
        }

        /// <summary>Element-wise <c>ldexp</c> for Half over an int32-exponent buffer (widen-compute-narrow).</summary>
        /// <param name="x">Half mantissa/base values.</param>
        /// <param name="exp">Per-element exponents (int32).</param>
        /// <param name="result">Receives <c>(Half)(x[i] * 2^exp[i])</c>.</param>
        /// <param name="n">Element count.</param>
        public static unsafe void LdexpHelper(Half* x, int* exp, Half* result, long n)
        {
            for (long i = 0; i < n; i++) result[i] = (Half)MathF.ScaleB((float)x[i], exp[i]);
        }

        /// <summary>Element-wise <c>ldexp</c> for Decimal over an int32-exponent buffer (double bridge).</summary>
        /// <param name="x">Decimal mantissa/base values.</param>
        /// <param name="exp">Per-element exponents (int32).</param>
        /// <param name="result">Receives <c>(decimal)(x[i] * 2^exp[i])</c>.</param>
        /// <param name="n">Element count.</param>
        /// <exception cref="OverflowException">A scaled value overflows the decimal range (double product ±inf).</exception>
        public static unsafe void LdexpHelper(decimal* x, int* exp, decimal* result, long n)
        {
            for (long i = 0; i < n; i++) result[i] = (decimal)Math.ScaleB((double)x[i], exp[i]);
        }

        /// <summary>Element-wise <c>ldexp</c> over contiguous double + int64-exponent buffers into a double result.</summary>
        /// <param name="x">Mantissa/base values.</param>
        /// <param name="exp">Per-element exponents (int64, clamped to int range before scaling).</param>
        /// <param name="result">Receives <c>x[i] * 2^exp[i]</c>.</param>
        /// <param name="n">Element count.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void LdexpHelper(double* x, long* exp, double* result, long n)
        {
            for (long i = 0; i < n; i++) result[i] = Math.ScaleB(x[i], ClampExp(exp[i]));
        }

        /// <summary>Element-wise <c>ldexp</c> over contiguous float + int64-exponent buffers into a float result.</summary>
        /// <param name="x">Mantissa/base values.</param>
        /// <param name="exp">Per-element exponents (int64, clamped to int range before scaling).</param>
        /// <param name="result">Receives <c>x[i] * 2^exp[i]</c>.</param>
        /// <param name="n">Element count.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void LdexpHelper(float* x, long* exp, float* result, long n)
        {
            for (long i = 0; i < n; i++) result[i] = MathF.ScaleB(x[i], ClampExp(exp[i]));
        }

        /// <summary>Element-wise <c>ldexp</c> for Half: widen to float, scale, narrow back (NumPy's HALF_ldexp).</summary>
        /// <param name="x">Half mantissa/base values.</param>
        /// <param name="exp">Per-element exponents (int64, clamped to int range before scaling).</param>
        /// <param name="result">Receives <c>(Half)(x[i] * 2^exp[i])</c>.</param>
        /// <param name="n">Element count.</param>
        public static unsafe void LdexpHelper(Half* x, long* exp, Half* result, long n)
        {
            for (long i = 0; i < n; i++) result[i] = (Half)MathF.ScaleB((float)x[i], ClampExp(exp[i]));
        }

        /// <summary>Element-wise <c>ldexp</c> for Decimal through the double bridge (a NumSharp extension).</summary>
        /// <param name="x">Decimal mantissa/base values.</param>
        /// <param name="exp">Per-element exponents (int64, clamped to int range before scaling).</param>
        /// <param name="result">Receives <c>(decimal)(x[i] * 2^exp[i])</c>.</param>
        /// <param name="n">Element count.</param>
        /// <exception cref="OverflowException">A scaled value overflows the decimal range (e.g. a huge exponent
        /// drives the double product to ±inf, which does not convert back to decimal) — the same bridge limit
        /// shared by every decimal transcendental in the engine.</exception>
        public static unsafe void LdexpHelper(decimal* x, long* exp, decimal* result, long n)
        {
            for (long i = 0; i < n; i++) result[i] = (decimal)Math.ScaleB((double)x[i], ClampExp(exp[i]));
        }

        /// <summary>Scalar-exponent fast path for double: scale a whole buffer by ONE (already-clamped) exponent
        /// without materializing a per-element exponent array — the common <c>np.ldexp(array, k)</c> shape.</summary>
        /// <param name="x">Mantissa/base values.</param>
        /// <param name="e">The single exponent, already clamped to int range.</param>
        /// <param name="result">Receives <c>x[i] * 2^e</c>.</param>
        /// <param name="n">Element count.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void LdexpScalarExp(double* x, int e, double* result, long n)
        {
            for (long i = 0; i < n; i++) result[i] = Math.ScaleB(x[i], e);
        }

        /// <summary>Scalar-exponent fast path for float (see <see cref="LdexpScalarExp(double*,int,double*,long)"/>).</summary>
        /// <param name="x">Mantissa/base values.</param>
        /// <param name="e">The single exponent, already clamped to int range.</param>
        /// <param name="result">Receives <c>x[i] * 2^e</c>.</param>
        /// <param name="n">Element count.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void LdexpScalarExp(float* x, int e, float* result, long n)
        {
            for (long i = 0; i < n; i++) result[i] = MathF.ScaleB(x[i], e);
        }

        /// <summary>Scalar-exponent fast path for Half (widen-compute-narrow).</summary>
        /// <param name="x">Half mantissa/base values.</param>
        /// <param name="e">The single exponent, already clamped to int range.</param>
        /// <param name="result">Receives <c>(Half)(x[i] * 2^e)</c>.</param>
        /// <param name="n">Element count.</param>
        public static unsafe void LdexpScalarExp(Half* x, int e, Half* result, long n)
        {
            for (long i = 0; i < n; i++) result[i] = (Half)MathF.ScaleB((float)x[i], e);
        }

        /// <summary>Scalar-exponent fast path for Decimal (double bridge).</summary>
        /// <param name="x">Decimal mantissa/base values.</param>
        /// <param name="e">The single exponent, already clamped to int range.</param>
        /// <param name="result">Receives <c>(decimal)(x[i] * 2^e)</c>.</param>
        /// <param name="n">Element count.</param>
        /// <exception cref="OverflowException">A scaled value overflows the decimal range (double product ±inf).</exception>
        public static unsafe void LdexpScalarExp(decimal* x, int e, decimal* result, long n)
        {
            for (long i = 0; i < n; i++) result[i] = (decimal)Math.ScaleB((double)x[i], e);
        }
    }
}
