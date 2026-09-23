using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// DirectILKernelGenerator.Frexp — np.frexp kernels (decompose x = mantissa * 2^exp)
// =============================================================================
//
// frexp(x) returns (mantissa, exponent) with mantissa in [0.5, 1) (or 0/inf/nan for the
// special inputs) and an INT32 exponent such that x == mantissa * 2^exponent. It is the
// inverse of ldexp. NumPy's loop signatures are e->ei / f->fi / d->di, i.e. the mantissa
// carries the input's float type and the exponent is ALWAYS int32 — NumSharp adds a Half
// path (widen-compute-narrow, NumPy's own HALF_frexp shape) and a Decimal extension.
//
// Algorithm (pure IEEE bit manipulation — no libm frexp exists in the BCL):
//   For a NORMAL value the biased exponent field already IS the answer: with x = 1.f·2^(E-bias)
//   and 1.f in [1,2), the mantissa in [0.5,1) is the same significand with its exponent field
//   forced to bias-1 (0.5's exponent), and the frexp exponent is (E - bias) + 1. So:
//       mantissa = (bits & sign|fraction) | (bias-1 << fracbits)
//       exponent = ((bits >> fracbits) & expmask) - (bias-1)
//   This is exact (multiplying/dividing by a power of two never rounds) and SIMD-friendly
//   (shift/and/or/subtract), so the common all-normal chunk is fully vectorized.
//
// Special / subnormal fixup (matches win-amd64 NumPy 2.4.2's SCALAR npy_frexp — the AVX512
// getmant/getexp path is not compiled into the shipped wheel on this ISA, so its exp==0 for
// specials does NOT apply):
//   ±0        -> (±0, 0)            (mantissa is the input, exponent 0)
//   ±inf      -> (±inf, -1)         (mantissa is the input, exponent -1 — the MSVC value)
//   NaN       -> (qNaN, -1)         (SIGNALLING NaNs are QUIETED — set the mantissa MSB —
//                                    exactly as the C runtime frexp does; verified bit-for-bit)
//   subnormal -> normalize by 2^p, run the normal-path formula, subtract p from the exponent.
// A vector chunk containing ANY special/subnormal lane falls to the scalar path (rare), so the
// hot all-normal path stays branch-free per element.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        #region Frexp scalar cores

        // Double IEEE-754 field layout: 1 sign + 11 exponent + 52 fraction, bias 1023.
        private const long FrexpDExpMask = 0x7FFL;                                   // 11-bit exponent field
        private const long FrexpDSignFrac = unchecked((long)0x800FFFFFFFFFFFFF);     // keep sign + 52 fraction bits
        private const long FrexpDHalfExp = 1022L << 52;                              // biased exponent of 0.5 (bias-1)
        private const long FrexpDFrac = 0xFFFFFFFFFFFFF;                             // 52 fraction bits
        private const long FrexpDQuiet = 0x0008000000000000L;                        // mantissa MSB (quiet-NaN bit)

        /// <summary>
        ///     Scalar <c>frexp</c> for a double: returns the mantissa in [0.5,1) (or the special input
        ///     itself for ±0/±inf/NaN) and writes the base-2 exponent. Bit-identical to win-amd64 NumPy
        ///     2.4.2's <c>npy_frexp</c>, signalling-NaN quieting included.
        /// </summary>
        /// <param name="v">The value to decompose.</param>
        /// <param name="exp">Receives the exponent: <c>0</c> for ±0, <c>-1</c> for ±inf/NaN, else the
        /// base-2 exponent such that <c>v == mantissa * 2^exp</c>.</param>
        /// <returns>The mantissa (in [0.5,1) for finite non-zero <paramref name="v"/>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static double FrexpStepD(double v, out int exp)
        {
            long bits = BitConverter.DoubleToInt64Bits(v);
            int eb = (int)((bits >> 52) & FrexpDExpMask);
            if (eb == 0x7FF)
            {
                // inf keeps its bits; a NaN is quieted (set the mantissa MSB) as the C runtime does.
                exp = -1;
                long outBits = (bits & FrexpDFrac) != 0 ? bits | FrexpDQuiet : bits;
                return BitConverter.Int64BitsToDouble(outBits);
            }
            if (eb == 0)
            {
                if ((bits & FrexpDFrac) == 0) { exp = 0; return v; } // ±0 passes through, exponent 0
                // Subnormal: scale up by 2^54 to a normal value, decompose, then remove the 54.
                double norm = v * 18014398509481984.0; // 2^54
                long nb = BitConverter.DoubleToInt64Bits(norm);
                exp = (int)((nb >> 52) & FrexpDExpMask) - 1022 - 54;
                return BitConverter.Int64BitsToDouble((nb & FrexpDSignFrac) | FrexpDHalfExp);
            }
            exp = eb - 1022;
            return BitConverter.Int64BitsToDouble((bits & FrexpDSignFrac) | FrexpDHalfExp);
        }

        // Single IEEE-754 field layout: 1 sign + 8 exponent + 23 fraction, bias 127.
        private const int FrexpFExpMask = 0xFF;
        private const int FrexpFSignFrac = unchecked((int)0x807FFFFF);
        private const int FrexpFHalfExp = 126 << 23;
        private const int FrexpFFrac = 0x7FFFFF;
        private const int FrexpFQuiet = 0x00400000;

        /// <summary>
        ///     Scalar <c>frexp</c> for a float — the <see cref="FrexpStepD"/> algorithm at single precision.
        ///     Also drives the Half path (widen the Half to float, decompose here, narrow the mantissa back),
        ///     matching NumPy's <c>HALF_frexp = npy_float_to_half(npy_frexpf((float)x, &amp;exp))</c>.
        /// </summary>
        /// <param name="v">The value to decompose.</param>
        /// <param name="exp">Receives the exponent (0 for ±0, -1 for ±inf/NaN, else the base-2 exponent).</param>
        /// <returns>The mantissa in [0.5,1) for finite non-zero <paramref name="v"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static float FrexpStepF(float v, out int exp)
        {
            int bits = BitConverter.SingleToInt32Bits(v);
            int eb = (bits >> 23) & FrexpFExpMask;
            if (eb == 0xFF)
            {
                exp = -1;
                int outBits = (bits & FrexpFFrac) != 0 ? bits | FrexpFQuiet : bits;
                return BitConverter.Int32BitsToSingle(outBits);
            }
            if (eb == 0)
            {
                if ((bits & FrexpFFrac) == 0) { exp = 0; return v; }
                float norm = v * 33554432.0f; // 2^25
                int nb = BitConverter.SingleToInt32Bits(norm);
                exp = ((nb >> 23) & FrexpFExpMask) - 126 - 25;
                return BitConverter.Int32BitsToSingle((nb & FrexpFSignFrac) | FrexpFHalfExp);
            }
            exp = eb - 126;
            return BitConverter.Int32BitsToSingle((bits & FrexpFSignFrac) | FrexpFHalfExp);
        }

        #endregion

        #region Frexp kernels (mantissa written in place; exponent to a parallel int32 buffer)

        /// <summary>
        ///     SIMD <c>frexp</c> over a contiguous double buffer: reads <paramref name="src"/>[i] and writes the
        ///     mantissa to <paramref name="mant"/>[i] and the int32 exponent to <paramref name="exp"/>[i]. Out of
        ///     place so the two outputs share one C-contiguous layout regardless of the source's layout. The
        ///     all-normal chunk is vectorized (integer field manipulation); any chunk with a zero/subnormal/inf/
        ///     NaN lane falls to the scalar core so the fast path stays branch-free.
        /// </summary>
        /// <param name="src">Source values (contiguous). May alias <paramref name="mant"/> only if identical.</param>
        /// <param name="mant">Receives the mantissas.</param>
        /// <param name="exp">Receives one int32 exponent per element.</param>
        /// <param name="n">Element count.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void FrexpHelper(double* src, double* mant, int* exp, long n)
        {
            if (n == 0) return;
            long i = 0;

            if (VectorBits >= 256 && n >= Vector256<long>.Count)
            {
                int W = Vector256<long>.Count; // 4 doubles/longs per vector
                long end = n - W;
                var expMask = Vector256.Create(FrexpDExpMask);
                var signFrac = Vector256.Create(FrexpDSignFrac);
                var halfExp = Vector256.Create(FrexpDHalfExp);
                var maxE = Vector256.Create(0x7FFL);
                var c1022 = Vector256.Create(1022L);
                var zero = Vector256<long>.Zero;
                for (; i <= end; i += W)
                {
                    var b = Vector256.Load(src + i).AsInt64();
                    var eb = Vector256.BitwiseAnd(Vector256.ShiftRightLogical(b, 52), expMask);
                    // A lane is "hard" (needs the scalar core) when its exponent field is 0 (zero/subnormal)
                    // or all-ones (inf/NaN). Everything else takes the exact branch-free normal-path formula.
                    var hard = Vector256.BitwiseOr(Vector256.Equals(eb, zero), Vector256.Equals(eb, maxE));
                    if (Vector256.EqualsAll(hard, zero))
                    {
                        var m = Vector256.BitwiseOr(Vector256.BitwiseAnd(b, signFrac), halfExp);
                        m.AsDouble().Store(mant + i);
                        var eOut = Vector256.Subtract(eb, c1022);
                        exp[i] = (int)eOut[0]; exp[i + 1] = (int)eOut[1];
                        exp[i + 2] = (int)eOut[2]; exp[i + 3] = (int)eOut[3];
                    }
                    else
                    {
                        for (long k = i; k < i + W; k++) mant[k] = FrexpStepD(src[k], out exp[k]);
                    }
                }
            }
            else if (VectorBits >= 128 && n >= Vector128<long>.Count)
            {
                int W = Vector128<long>.Count; // 2
                long end = n - W;
                var expMask = Vector128.Create(FrexpDExpMask);
                var signFrac = Vector128.Create(FrexpDSignFrac);
                var halfExp = Vector128.Create(FrexpDHalfExp);
                var maxE = Vector128.Create(0x7FFL);
                var c1022 = Vector128.Create(1022L);
                var zero = Vector128<long>.Zero;
                for (; i <= end; i += W)
                {
                    var b = Vector128.Load(src + i).AsInt64();
                    var eb = Vector128.BitwiseAnd(Vector128.ShiftRightLogical(b, 52), expMask);
                    var hard = Vector128.BitwiseOr(Vector128.Equals(eb, zero), Vector128.Equals(eb, maxE));
                    if (Vector128.EqualsAll(hard, zero))
                    {
                        var m = Vector128.BitwiseOr(Vector128.BitwiseAnd(b, signFrac), halfExp);
                        m.AsDouble().Store(mant + i);
                        var eOut = Vector128.Subtract(eb, c1022);
                        exp[i] = (int)eOut[0]; exp[i + 1] = (int)eOut[1];
                    }
                    else
                    {
                        for (long k = i; k < i + W; k++) mant[k] = FrexpStepD(src[k], out exp[k]);
                    }
                }
            }

            for (; i < n; i++) mant[i] = FrexpStepD(src[i], out exp[i]); // scalar tail (and whole loop when SIMD is off)
        }

        /// <summary>
        ///     SIMD <c>frexp</c> over a contiguous float buffer — the <see cref="FrexpHelper(double*,double*,int*,long)"/>
        ///     algorithm at single precision (exponents are computed as int32 lanes and stored directly).
        /// </summary>
        /// <param name="src">Source values (contiguous).</param>
        /// <param name="mant">Receives the mantissas.</param>
        /// <param name="exp">Receives one int32 exponent per element.</param>
        /// <param name="n">Element count.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void FrexpHelper(float* src, float* mant, int* exp, long n)
        {
            if (n == 0) return;
            long i = 0;

            if (VectorBits >= 256 && n >= Vector256<int>.Count)
            {
                int W = Vector256<int>.Count; // 8
                long end = n - W;
                var expMask = Vector256.Create(FrexpFExpMask);
                var signFrac = Vector256.Create(FrexpFSignFrac);
                var halfExp = Vector256.Create(FrexpFHalfExp);
                var maxE = Vector256.Create(0xFF);
                var c126 = Vector256.Create(126);
                var zero = Vector256<int>.Zero;
                for (; i <= end; i += W)
                {
                    var b = Vector256.Load(src + i).AsInt32();
                    var eb = Vector256.BitwiseAnd(Vector256.ShiftRightLogical(b, 23), expMask);
                    var hard = Vector256.BitwiseOr(Vector256.Equals(eb, zero), Vector256.Equals(eb, maxE));
                    if (Vector256.EqualsAll(hard, zero))
                    {
                        var m = Vector256.BitwiseOr(Vector256.BitwiseAnd(b, signFrac), halfExp);
                        m.AsSingle().Store(mant + i);
                        Vector256.Subtract(eb, c126).Store(exp + i); // exponent lanes are int32 — direct store
                    }
                    else
                    {
                        for (long k = i; k < i + W; k++) mant[k] = FrexpStepF(src[k], out exp[k]);
                    }
                }
            }
            else if (VectorBits >= 128 && n >= Vector128<int>.Count)
            {
                int W = Vector128<int>.Count; // 4
                long end = n - W;
                var expMask = Vector128.Create(FrexpFExpMask);
                var signFrac = Vector128.Create(FrexpFSignFrac);
                var halfExp = Vector128.Create(FrexpFHalfExp);
                var maxE = Vector128.Create(0xFF);
                var c126 = Vector128.Create(126);
                var zero = Vector128<int>.Zero;
                for (; i <= end; i += W)
                {
                    var b = Vector128.Load(src + i).AsInt32();
                    var eb = Vector128.BitwiseAnd(Vector128.ShiftRightLogical(b, 23), expMask);
                    var hard = Vector128.BitwiseOr(Vector128.Equals(eb, zero), Vector128.Equals(eb, maxE));
                    if (Vector128.EqualsAll(hard, zero))
                    {
                        var m = Vector128.BitwiseOr(Vector128.BitwiseAnd(b, signFrac), halfExp);
                        m.AsSingle().Store(mant + i);
                        Vector128.Subtract(eb, c126).Store(exp + i);
                    }
                    else
                    {
                        for (long k = i; k < i + W; k++) mant[k] = FrexpStepF(src[k], out exp[k]);
                    }
                }
            }

            for (; i < n; i++) mant[i] = FrexpStepF(src[i], out exp[i]);
        }

        /// <summary>
        ///     Scalar <c>frexp</c> over a contiguous Half buffer: widens each value to float, decomposes it
        ///     with <see cref="FrexpStepF"/> and narrows the mantissa back to Half — NumPy's exact
        ///     <c>HALF_frexp</c> shape. No vector path (the BCL has no Half arithmetic).
        /// </summary>
        /// <param name="src">Source Half values (contiguous).</param>
        /// <param name="mant">Receives the Half mantissas.</param>
        /// <param name="exp">Receives one int32 exponent per element.</param>
        /// <param name="n">Element count.</param>
        public static unsafe void FrexpHelper(Half* src, Half* mant, int* exp, long n)
        {
            for (long i = 0; i < n; i++)
                mant[i] = (Half)FrexpStepF((float)src[i], out exp[i]);
        }

        /// <summary>
        ///     Scalar <c>frexp</c> over a contiguous Decimal buffer (a NumSharp extension; NumPy has no decimal
        ///     dtype). Computed through the double bridge, so the mantissa carries double precision — consistent
        ///     with every other decimal transcendental in the engine.
        /// </summary>
        /// <param name="src">Source Decimal values (contiguous).</param>
        /// <param name="mant">Receives the Decimal mantissas.</param>
        /// <param name="exp">Receives one int32 exponent per element.</param>
        /// <param name="n">Element count.</param>
        public static unsafe void FrexpHelper(decimal* src, decimal* mant, int* exp, long n)
        {
            // A decimal is always finite, so the special (inf/NaN) branch of FrexpStepD is never taken here.
            for (long i = 0; i < n; i++)
                mant[i] = (decimal)FrexpStepD((double)src[i], out exp[i]);
        }

        #endregion
    }
}
