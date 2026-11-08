using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.IntToFloat.cs
        //   {int8, uint8, int16, uint16, char, int32, uint32} -> {float32, float64}, contiguous.
        //
        //   The generic cast emitter's int->float strategies measured at SCALAR speed on an L1-resident
        //   4096-element block (int32->float32 0.185 ns/element, int32->float64 0.211, int8/uint8->float
        //   0.22-0.37, uint32->float 0.22-0.37) — only int16/uint16->float32 actually vectorized
        //   (0.047). These kernels widen with the AVX2 sign/zero-extending loads (pmovsx/pmovzx straight
        //   from memory) and convert with cvtdq2pd / cvtdq2ps, 4-8 elements per instruction pair.
        //
        //   Bit-exact with the scalar conversions (EmitConvertTo / C casts, hence NumPy):
        //     * every int8/uint8/int16/uint16/char and every int32 is EXACT in float64, and every
        //       8/16-bit value is exact in float32 — the conversion cannot round;
        //     * int32 -> float32 rounds once, round-to-nearest-even, in cvtdq2ps exactly as in the scalar
        //       cvtsi2ss (both honour MXCSR's default RNE);
        //     * uint32 has no AVX2 unsigned convert (vcvtudq2pd is AVX-512): the value is biased into
        //       int32 range, u ^ 0x80000000 == u - 2^31 as a signed int, converted EXACTLY, and 2^31
        //       added back in float64 — also exact (the result is an integer below 2^32). uint32 ->
        //       float32 then narrows that exact float64 once (cvtpd2ps), the single rounding the scalar
        //       conv.r.un + conv.r4 and NumPy's C cast perform — never a biased float32 sum, which
        //       would round twice.
        //   int64/uint64 -> float64 has no AVX2 instruction either; the kernels build the double from two EXACT
        //   halves with the well-known exponent-bias splice (Mysticial's full-range conversion): the high part
        //   is written into the mantissa of a large power of two (2^84 for uint64 — its ulp is 2^32, so the
        //   high 32 bits land exactly; 3·2^67 for int64 — ulp 2^16, the sign-extended top 16 bits), the bias
        //   and 2^52 are subtracted exactly (Sterbenz), the low part is spliced into 2^52's mantissa (exact),
        //   and ONE final addition rounds — round-to-nearest-even, the same single rounding as the scalar
        //   cvtsi2sd (int64) and the Converts.ToDouble(ulong) path NumPy's cast is matched by. int64/uint64 ->
        //   float32 stays scalar: through float64 it would round twice.
        // =====================================================================

        /// <summary>
        ///     The contiguous AVX2 int→float cast kernel for (<paramref name="srcType"/>, <paramref name="dstType"/>),
        ///     or null when the pair is not one of this file's (see the header) or the host lacks AVX2.
        /// </summary>
        /// <param name="srcType">Source dtype.</param>
        /// <param name="dstType">Destination dtype.</param>
        /// <returns>The kernel, or null.</returns>
        internal static unsafe CastKernel TryGetIntToFloatKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (!Avx2.IsSupported)
                return null;
            if (dstType == NPTypeCode.Double)
            {
                switch (srcType)
                {
                    case NPTypeCode.SByte: return CastSByteToDoubleContig;
                    case NPTypeCode.Byte: return CastByteToDoubleContig;
                    case NPTypeCode.Int16: return CastInt16ToDoubleContig;
                    case NPTypeCode.UInt16:
                    case NPTypeCode.Char: return CastUInt16ToDoubleContig;
                    case NPTypeCode.Int32: return CastInt32ToDoubleContig;
                    case NPTypeCode.UInt32: return CastUInt32ToDoubleContig;
                    case NPTypeCode.Int64: return CastInt64ToDoubleContig;
                    case NPTypeCode.UInt64: return CastUInt64ToDoubleContig;
                }
            }
            else if (dstType == NPTypeCode.Single)
            {
                switch (srcType)
                {
                    case NPTypeCode.SByte: return CastSByteToSingleContig;
                    case NPTypeCode.Byte: return CastByteToSingleContig;
                    case NPTypeCode.Int16: return CastInt16ToSingleContig;
                    case NPTypeCode.UInt16:
                    case NPTypeCode.Char: return CastUInt16ToSingleContig;
                    case NPTypeCode.Int32: return CastInt32ToSingleContig;
                    case NPTypeCode.UInt32: return CastUInt32ToSingleContig;
                }
            }
            return null;
        }

        /// <summary>
        ///     Whether the contiguous house cast from <paramref name="srcType"/> to <paramref name="dstType"/> is a
        ///     VECTOR loop on this host — the question a caller asks before trading a fused scalar conversion for a
        ///     separate cast pass (the numpy.polynomial combine does). Covers this file's int→float kernels, the
        ///     float16 Giesen widens, and the generic emitter's float32→float64 strategy.
        /// </summary>
        /// <param name="srcType">Source dtype.</param>
        /// <param name="dstType">Destination dtype.</param>
        /// <returns>True for a vectorized contiguous cast.</returns>
        internal static bool IsVectorizedContiguousCast(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (!Avx2.IsSupported)
                return false;
            if (dstType == NPTypeCode.Double)
                return srcType is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char
                    or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64
                    or NPTypeCode.Half or NPTypeCode.Single;
            if (dstType == NPTypeCode.Single)
                return srcType is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char
                    or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Half;
            return false;
        }

        // ---- -> float64 (4 lanes per cvtdq2pd; two per iteration) ----------------------------------------------

        /// <summary>int8 → float64 (exact): pmovsxbd from memory, cvtdq2pd.</summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastSByteToDoubleContig(void* s, void* d, long n)
        {
            sbyte* src = (sbyte*)s; double* dst = (double*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
            {
                Avx.Store(dst + i, Avx.ConvertToVector256Double(Sse41.ConvertToVector128Int32(src + i)));
                Avx.Store(dst + i + 4, Avx.ConvertToVector256Double(Sse41.ConvertToVector128Int32(src + i + 4)));
            }
            for (; i < n; i++) dst[i] = src[i];
        }

        /// <summary>uint8 → float64 (exact): pmovzxbd from memory, cvtdq2pd.</summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastByteToDoubleContig(void* s, void* d, long n)
        {
            byte* src = (byte*)s; double* dst = (double*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
            {
                Avx.Store(dst + i, Avx.ConvertToVector256Double(Sse41.ConvertToVector128Int32(src + i)));
                Avx.Store(dst + i + 4, Avx.ConvertToVector256Double(Sse41.ConvertToVector128Int32(src + i + 4)));
            }
            for (; i < n; i++) dst[i] = src[i];
        }

        /// <summary>int16 → float64 (exact): pmovsxwd from memory, cvtdq2pd.</summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastInt16ToDoubleContig(void* s, void* d, long n)
        {
            short* src = (short*)s; double* dst = (double*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
            {
                Avx.Store(dst + i, Avx.ConvertToVector256Double(Sse41.ConvertToVector128Int32(src + i)));
                Avx.Store(dst + i + 4, Avx.ConvertToVector256Double(Sse41.ConvertToVector128Int32(src + i + 4)));
            }
            for (; i < n; i++) dst[i] = src[i];
        }

        /// <summary>uint16 / char → float64 (exact): pmovzxwd from memory, cvtdq2pd.</summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastUInt16ToDoubleContig(void* s, void* d, long n)
        {
            ushort* src = (ushort*)s; double* dst = (double*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
            {
                Avx.Store(dst + i, Avx.ConvertToVector256Double(Sse41.ConvertToVector128Int32(src + i)));
                Avx.Store(dst + i + 4, Avx.ConvertToVector256Double(Sse41.ConvertToVector128Int32(src + i + 4)));
            }
            for (; i < n; i++) dst[i] = src[i];
        }

        /// <summary>int32 → float64 (exact): cvtdq2pd of a 128-bit load.</summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastInt32ToDoubleContig(void* s, void* d, long n)
        {
            int* src = (int*)s; double* dst = (double*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
            {
                Avx.Store(dst + i, Avx.ConvertToVector256Double(Sse2.LoadVector128(src + i)));
                Avx.Store(dst + i + 4, Avx.ConvertToVector256Double(Sse2.LoadVector128(src + i + 4)));
            }
            for (; i < n; i++) dst[i] = src[i];
        }

        /// <summary>
        ///     Four uint32 values → float64, exactly: the sign-bias trick (see the file header) — no AVX2 instruction
        ///     converts unsigned integers.
        /// </summary>
        /// <param name="p">The four values.</param>
        /// <returns>The four exact float64 values.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<double> UInt32x4ToDouble(uint* p)
        {
            var biased = Sse2.Xor(Sse2.LoadVector128(p), Vector128.Create(0x80000000u)).AsInt32();
            return Avx.Add(Avx.ConvertToVector256Double(biased), Vector256.Create(2147483648.0));
        }

        /// <summary>uint32 → float64 (exact): the bias trick.</summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastUInt32ToDoubleContig(void* s, void* d, long n)
        {
            uint* src = (uint*)s; double* dst = (double*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
            {
                Avx.Store(dst + i, UInt32x4ToDouble(src + i));
                Avx.Store(dst + i + 4, UInt32x4ToDouble(src + i + 4));
            }
            for (; i < n; i++) dst[i] = src[i];
        }

        /// <summary>2^84 as a float64 (its bit pattern is the uint64 high-part carrier: ulp 2^32).</summary>
        private static readonly Vector256<double> _pow2_84 = Vector256.Create(19342813113834066795298816.0);
        /// <summary>2^84 + 2^52: the bias removed from the uint64 high part (exactly, by Sterbenz).</summary>
        private static readonly Vector256<double> _pow2_84_plus_52 = Vector256.Create(19342813118337666422669312.0);
        /// <summary>2^52 as a float64 (its bit pattern is the low-part carrier: ulp 1).</summary>
        private static readonly Vector256<double> _pow2_52 = Vector256.Create(4503599627370496.0);
        /// <summary>3·2^67 as a float64 (the int64 high-part carrier: ulp 2^16, room for a signed 16-bit part).</summary>
        private static readonly Vector256<double> _three_pow2_67 = Vector256.Create(442721857769029238784.0);
        /// <summary>3·2^67 + 2^52: the bias removed from the int64 high part.</summary>
        private static readonly Vector256<double> _three_pow2_67_plus_52 = Vector256.Create(442726361368656609280.0);

        /// <summary>
        ///     Four uint64 values → float64 with one round-to-nearest-even rounding: high 32 bits spliced into
        ///     2^84's mantissa, low 32 bits into 2^52's, the biases subtracted exactly, one final add.
        /// </summary>
        /// <param name="x">The four values.</param>
        /// <returns>The correctly rounded float64 values.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<double> UInt64x4ToDouble(Vector256<ulong> x)
        {
            var hi = Avx2.Or(Avx2.ShiftRightLogical(x, 32), _pow2_84.AsUInt64()).AsDouble();                  // 2^84 + hi·2^32
            var lo = Avx2.Blend(x.AsInt16(), _pow2_52.AsInt16(), 0b1100_1100).AsDouble();                     // 2^52 + lo
            return Avx.Add(Avx.Subtract(hi, _pow2_84_plus_52), lo);
        }

        /// <summary>
        ///     Four int64 values → float64 with one round-to-nearest-even rounding: the sign-extended top 16 bits
        ///     spliced into 3·2^67's mantissa, the low 48 bits into 2^52's, the biases subtracted exactly, one final add.
        /// </summary>
        /// <param name="x">The four values.</param>
        /// <returns>The correctly rounded float64 values.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<double> Int64x4ToDouble(Vector256<long> x)
        {
            // (x >> 48) as a 64-bit lane shifted up by 32: arithmetic-shift each 32-bit half by 16, then zero the low half.
            var top = Avx2.Blend(Avx2.ShiftRightArithmetic(x.AsInt32(), 16).AsInt16(), Vector256<short>.Zero, 0b0011_0011);
            var hi = Avx2.Add(top.AsInt64(), _three_pow2_67.AsInt64()).AsDouble();                            // 3·2^67 + (x>>48)·2^48
            var lo = Avx2.Blend(x.AsInt16(), _pow2_52.AsInt16(), 0b1000_1000).AsDouble();                     // 2^52 + (x & 2^48-1)
            return Avx.Add(Avx.Subtract(hi, _three_pow2_67_plus_52), lo);
        }

        /// <summary>int64 → float64 (one RNE rounding, like the scalar cvtsi2sd).</summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastInt64ToDoubleContig(void* s, void* d, long n)
        {
            long* src = (long*)s; double* dst = (double*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
            {
                Avx.Store(dst + i, Int64x4ToDouble(Avx.LoadVector256(src + i)));
                Avx.Store(dst + i + 4, Int64x4ToDouble(Avx.LoadVector256(src + i + 4)));
            }
            for (; i < n; i++) dst[i] = src[i];
        }

        /// <summary>uint64 → float64 (one RNE rounding, like <c>Converts.ToDouble(ulong)</c>).</summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastUInt64ToDoubleContig(void* s, void* d, long n)
        {
            ulong* src = (ulong*)s; double* dst = (double*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
            {
                Avx.Store(dst + i, UInt64x4ToDouble(Avx.LoadVector256(src + i)));
                Avx.Store(dst + i + 4, UInt64x4ToDouble(Avx.LoadVector256(src + i + 4)));
            }
            for (; i < n; i++) dst[i] = NumSharp.Utilities.Converts.ToDouble(src[i]);
        }

        // ---- -> float32 (8 lanes per cvtdq2ps) -----------------------------------------------------------------

        /// <summary>int8 → float32 (exact): pmovsxbd ymm from memory, cvtdq2ps.</summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastSByteToSingleContig(void* s, void* d, long n)
        {
            sbyte* src = (sbyte*)s; float* dst = (float*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
                Avx.Store(dst + i, Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(src + i)));
            for (; i < n; i++) dst[i] = src[i];
        }

        /// <summary>uint8 → float32 (exact): pmovzxbd ymm from memory, cvtdq2ps.</summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastByteToSingleContig(void* s, void* d, long n)
        {
            byte* src = (byte*)s; float* dst = (float*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
                Avx.Store(dst + i, Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(src + i)));
            for (; i < n; i++) dst[i] = src[i];
        }

        /// <summary>int16 → float32 (exact): pmovsxwd ymm from memory, cvtdq2ps.</summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastInt16ToSingleContig(void* s, void* d, long n)
        {
            short* src = (short*)s; float* dst = (float*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
                Avx.Store(dst + i, Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(src + i)));
            for (; i < n; i++) dst[i] = src[i];
        }

        /// <summary>uint16 / char → float32 (exact): pmovzxwd ymm from memory, cvtdq2ps.</summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastUInt16ToSingleContig(void* s, void* d, long n)
        {
            ushort* src = (ushort*)s; float* dst = (float*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
                Avx.Store(dst + i, Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(src + i)));
            for (; i < n; i++) dst[i] = src[i];
        }

        /// <summary>int32 → float32: cvtdq2ps, round-to-nearest-even like the scalar cvtsi2ss.</summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastInt32ToSingleContig(void* s, void* d, long n)
        {
            int* src = (int*)s; float* dst = (float*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
                Avx.Store(dst + i, Avx.ConvertToVector256Single(Avx.LoadVector256(src + i)));
            for (; i < n; i++) dst[i] = src[i];
        }

        /// <summary>
        ///     uint32 → float32 with ONE rounding: the exact bias-trick float64 value, narrowed by cvtpd2ps — the
        ///     scalar path's conv.r.un + conv.r4.
        /// </summary>
        /// <param name="s">Source elements.</param><param name="d">Destination elements.</param><param name="n">Element count.</param>
        private static unsafe void CastUInt32ToSingleContig(void* s, void* d, long n)
        {
            uint* src = (uint*)s; float* dst = (float*)d;
            long i = 0;
            for (; i + 8 <= n; i += 8)
            {
                Sse.Store(dst + i, Avx.ConvertToVector128Single(UInt32x4ToDouble(src + i)));
                Sse.Store(dst + i + 4, Avx.ConvertToVector128Single(UInt32x4ToDouble(src + i + 4)));
            }
            // (float)(double)u: the exact widen, then one rounding.
            for (; i < n; i++) dst[i] = (float)(double)src[i];
        }
    }
}
