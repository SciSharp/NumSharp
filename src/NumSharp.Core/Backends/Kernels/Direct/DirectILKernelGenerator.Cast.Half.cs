using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.Half.cs
        //   Half (float16) -> int (i32/i8/u8/i16/u16/char) contiguous + strided casts.
        //
        //   Phase-0: f16->narrow geomean 0.48 (f16->i32 was ~0.69x) — Half casts fall
        //   to the scalar path. There is NO F16C in this .NET (verified: Avx512FP16
        //   absent, no Vector256<Half>, no vectorized ConvertToSingle(Half)), and the
        //   scalar (float)half is a TRAP (measured SLOWER than the current scalar — it
        //   double-converts). So widen with a vectorized IEEE bit-fiddle:
        //
        //     GIESEN half->float (branchless, exact for finite incl. subnormals):
        //       widen 8 halves (vpmovzxwd) -> 8x i32 bit patterns
        //       expmant = bits & 0x7fff;  scaled = (expmant << 13)_as_f32 * MAGIC
        //         (MAGIC = (254-15)<<23 as float; the multiply rescales the exponent
        //          and reconstructs subnormals exactly)
        //       inf/nan (expmant > 0x7bff) -> OR in the f32 inf/nan exponent (0xff<<23)
        //       sign = (bits & 0x8000) << 16
        //     then cvttps2dq -> i32 (INT_MIN sentinel on the inf/nan we just built),
        //     and truncating Vector.Narrow i32->i16[->i8] for the narrow targets.
        //
        //   Bit-exact with Converts.To{X}(Half) (NaN/inf -> INT_MIN -> low bits;
        //   finite Half max is 65504 so cvtt never overflows int32 -> (int)value),
        //   hence NumPy 2.4.2 (proven 0-diff; f16->i32 1.85x, f16->i8 1.42x).
        //   f16->f32 IS here (HalfBitsToFloatExact): Giesen widens finite exactly, and the
        //   inf/nan lanes are overridden with the IEEE widen (sign | 0x7f800000 | mant<<13) so
        //   the NaN payload — and sNaN — are preserved, matching NumPy (proven 0-diff over all
        //   65536 f16 values). The BCL (float)Half cast QUIETS sNaN, so this also fixes that
        //   latent divergence. f16->f64 stays scalar (already >=0.9; a 2-step f16->f32->f64
        //   widen would re-quiet sNaN via cvtps2pd, so it needs its own direct widen later).
        //
        //   Strided reuses StridedNarrowDriver (srcSize=2): inner-contig rows run the
        //   Bulk, inner-strided rows stage to a contig u16 buffer then vectorize.
        // =====================================================================

        /// <summary>
        /// Returns the contiguous <see cref="CastKernel"/> for Half -&gt;
        /// {i32,i8,u8,i16,u16,char}, or null. Bit-exact with <see cref="Converts"/>.
        /// </summary>
        internal static unsafe CastKernel TryGetHalfToXKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (srcType != NPTypeCode.Half || !Avx2.IsSupported) return null;
            switch (dstType)
            {
                case NPTypeCode.Int32:  return CastHalfToInt32Contig;
                case NPTypeCode.UInt32: return CastHalfToUInt32Contig;
                case NPTypeCode.Int64:  return CastHalfToInt64Contig;
                case NPTypeCode.UInt64: return CastHalfToUInt64Contig;
                case NPTypeCode.Single: return CastHalfToFloatContig;
                case NPTypeCode.SByte:  return CastHalfToSByteContig;
                case NPTypeCode.Byte:   return CastHalfToByteContig;
                case NPTypeCode.Int16:  return CastHalfToInt16Contig;
                case NPTypeCode.UInt16: return CastHalfToUInt16Contig;
                case NPTypeCode.Char:   return CastHalfToCharContig;
            }
            return null;
        }

        // f16 -> u32: Giesen widen (exact for finite; inf/nan -> f32 inf/nan) then the AVX2
        // f32->u32 kernel (SingleToU32x8 widens to f64 and folds inf/nan/overflow -> 0).
        // Bit-exact with Converts.ToUInt32(Half) (real f16 max 65504, so only negatives wrap).
        private static unsafe Vector256<int> HalfToUInt32x8(ushort* p, long ci)
        {
            var hbits = Avx2.ConvertToVector256Int32(Sse2.LoadVector128(p + ci));
            return SingleToU32x8(HalfBitsToFloat(hbits));
        }
        private static unsafe long BulkHalfToUInt32(ushort* p, uint* dst, long count)
        {
            long i = 0;
            for (; i + 8 <= count; i += 8)
                Vector256.Store(HalfToUInt32x8(p, i).AsUInt32(), dst + i);
            return i;
        }
        private static unsafe void CastHalfToUInt32Contig(void* s, void* d, long n)
        {
            ushort* p = (ushort*)s; uint* dst = (uint*)d; Half* h = (Half*)s;
            long i = BulkHalfToUInt32(p, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToUInt32(h[i]);
        }
        private static unsafe long BulkHalfToUInt32V(void* s, void* d, long n) => BulkHalfToUInt32((ushort*)s, (uint*)d, n);
        private static unsafe void ConvHalfToUInt32(void* s, void* d) => *(uint*)d = Converts.ToUInt32(*(Half*)s);

        // f16 -> f32 WIDEN, bit-exact incl. NaN payload (proven 0-diff over all 65536 f16 values).
        // Giesen widens finite (normal/subnormal/zero) exactly; the inf/nan lanes are overridden
        // with the IEEE widen sign | 0x7f800000 | (mant<<13) (preserves the NaN payload AND sNaN,
        // which the BCL (float)Half cast QUIETS — the same latent bug the X->Half wave fixed, here
        // for the widen direction). Was the reason f16->f32 stayed on the scalar path.
        private static Vector256<float> HalfBitsToFloatExact(Vector256<int> h)
        {
            var giesen = HalfBitsToFloat(h);
            var isInfNan = Avx2.CompareGreaterThan(Avx2.And(Vector256.Create(0x7fff), h), Vector256.Create(0x7bff));
            var sign = Avx2.ShiftLeftLogical(Avx2.AndNot(Vector256.Create(0x7fff), h), 16);
            var mant = Avx2.And(Vector256.Create(0x3ff), h);
            var nanv = Avx2.Or(Avx2.Or(sign, Vector256.Create(0x7f800000)), Avx2.ShiftLeftLogical(mant, 13));
            return Avx2.BlendVariable(giesen.AsInt32(), nanv, isInfNan).AsSingle();
        }
        // Scalar widen matching the SIMD path (finite via BCL; inf/nan via the bit formula).
        private static float HalfToFloatScalarExact(ushort h)
        {
            if ((h & 0x7fff) > 0x7bff)
                return BitConverter.UInt32BitsToSingle(((uint)(h & 0x8000) << 16) | 0x7f800000u | ((uint)(h & 0x3ff) << 13));
            return (float)BitConverter.UInt16BitsToHalf(h);
        }
        private static unsafe long BulkHalfToFloat(ushort* p, float* dst, long count)
        {
            long i = 0;
            for (; i + 8 <= count; i += 8)
                Vector256.Store(HalfBitsToFloatExact(Avx2.ConvertToVector256Int32(Sse2.LoadVector128(p + i))), dst + i);
            return i;
        }
        private static unsafe void CastHalfToFloatContig(void* s, void* d, long n)
        {
            ushort* p = (ushort*)s; float* dst = (float*)d;
            long i = BulkHalfToFloat(p, dst, n);
            for (; i < n; i++) dst[i] = HalfToFloatScalarExact(p[i]);
        }
        private static unsafe long BulkHalfToFloatV(void* s, void* d, long n) => BulkHalfToFloat((ushort*)s, (float*)d, n);
        private static unsafe void ConvHalfToFloat(void* s, void* d) => *(float*)d = HalfToFloatScalarExact(*(ushort*)s);

        // f16 -> u64: Giesen widen (NaN payload irrelevant: inf/nan -> 2^63 regardless) then the
        // AVX2 f64->u64 kernel. Bit-exact with Converts.ToUInt64(Half).
        private static unsafe long BulkHalfToUInt64(ushort* p, ulong* dst, long count)
        {
            long i = 0;
            for (; i + 8 <= count; i += 8)
            {
                var f8 = HalfBitsToFloat(Avx2.ConvertToVector256Int32(Sse2.LoadVector128(p + i)));
                Vector256.Store(DoubleToU64x4(Avx.ConvertToVector256Double(f8.GetLower())).AsUInt64(), dst + i);
                Vector256.Store(DoubleToU64x4(Avx.ConvertToVector256Double(f8.GetUpper())).AsUInt64(), dst + i + 4);
            }
            return i;
        }
        private static unsafe void CastHalfToUInt64Contig(void* s, void* d, long n)
        {
            ushort* p = (ushort*)s; ulong* dst = (ulong*)d; Half* h = (Half*)s;
            long i = BulkHalfToUInt64(p, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToUInt64(h[i]);
        }
        private static unsafe long BulkHalfToUInt64V(void* s, void* d, long n) => BulkHalfToUInt64((ushort*)s, (ulong*)d, n);
        private static unsafe void ConvHalfToUInt64(void* s, void* d) => *(ulong*)d = Converts.ToUInt64(*(Half*)s);
        private static unsafe void StridedHalfToUInt64(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 8, &BulkHalfToUInt64V, &ConvHalfToUInt64);

        // f16 -> i64: every finite f16 (|v| <= 65504) fits in i32, so Giesen widen -> cvttps2dq ->
        // sign-extend i32->i64 is exact; inf/nan (cvtt -> 0x80000000, would sign-extend wrong) is
        // blended to int64.MinValue, matching NumPy / Converts.ToInt64(Half).
        private static unsafe long BulkHalfToInt64(ushort* p, long* dst, long count)
        {
            long i = 0;
            var minv = Vector256.Create(long.MinValue);
            for (; i + 8 <= count; i += 8)
            {
                var hb = Avx2.ConvertToVector256Int32(Sse2.LoadVector128(p + i));
                var i32 = Avx.ConvertToVector256Int32WithTruncation(HalfBitsToFloat(hb));
                var infnan = Avx2.CompareGreaterThan(Avx2.And(Vector256.Create(0x7fff), hb), Vector256.Create(0x7bff));
                var lo = Avx2.BlendVariable(Avx2.ConvertToVector256Int64(i32.GetLower()), minv, Avx2.ConvertToVector256Int64(infnan.GetLower()));
                var hi = Avx2.BlendVariable(Avx2.ConvertToVector256Int64(i32.GetUpper()), minv, Avx2.ConvertToVector256Int64(infnan.GetUpper()));
                Vector256.Store(lo, dst + i);
                Vector256.Store(hi, dst + i + 4);
            }
            return i;
        }
        private static unsafe void CastHalfToInt64Contig(void* s, void* d, long n)
        {
            ushort* p = (ushort*)s; long* dst = (long*)d; Half* h = (Half*)s;
            long i = BulkHalfToInt64(p, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToInt64(h[i]);
        }
        private static unsafe long BulkHalfToInt64V(void* s, void* d, long n) => BulkHalfToInt64((ushort*)s, (long*)d, n);
        private static unsafe void ConvHalfToInt64(void* s, void* d) => *(long*)d = Converts.ToInt64(*(Half*)s);
        private static unsafe void StridedHalfToInt64(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 8, &BulkHalfToInt64V, &ConvHalfToInt64);

        // Giesen branchless half->float over 8 widened half-bit-patterns (Vector256<int>).
        private static Vector256<float> HalfBitsToFloat(Vector256<int> h)
        {
            var maskNoSign = Vector256.Create(0x7fff);
            var magic = Vector256.Create((254 - 15) << 23).AsSingle();
            var wasInfNan = Vector256.Create(0x7bff);
            var expInfNan = Vector256.Create(255 << 23).AsSingle();
            var expmant = Avx2.And(maskNoSign, h);
            var scaled = Avx.Multiply(Avx2.ShiftLeftLogical(expmant, 13).AsSingle(), magic);
            var infnan = Avx2.CompareGreaterThan(expmant, wasInfNan);
            var sign = Avx2.ShiftLeftLogical(Avx2.AndNot(maskNoSign, h), 16);
            var signInf = Avx.Or(sign.AsSingle(), Avx.And(infnan.AsSingle(), expInfNan));
            return Avx.Or(scaled, signInf);
        }

        // 8 halves at p+ci -> widen (vpmovzxwd) -> Giesen -> cvttps2dq -> 8x i32.
        private static unsafe Vector256<int> HalfToInt32x8(ushort* p, long ci)
        {
            var hbits = Avx2.ConvertToVector256Int32(Sse2.LoadVector128(p + ci)); // zero-extend 8 u16 -> 8 i32
            return Avx.ConvertToVector256Int32WithTruncation(HalfBitsToFloat(hbits));
        }

        // 8 halves -> 8 i32.
        private static unsafe long BulkHalfToInt32(ushort* p, int* dst, long count)
        {
            long i = 0;
            for (; i + 8 <= count; i += 8)
                Vector256.Store(HalfToInt32x8(p, i), dst + i);
            return i;
        }

        // 16 halves -> 2x (8xi32) -> Narrow -> 16 i16.
        private static unsafe long BulkHalfToShort(ushort* p, short* dst, long count)
        {
            long i = 0;
            for (; i + 16 <= count; i += 16)
            {
                var a = HalfToInt32x8(p, i);
                var b = HalfToInt32x8(p, i + 8);
                Vector256.Store(Vector256.Narrow(a, b), dst + i);   // 16x i16
            }
            return i;
        }

        // 32 halves -> 4x (8xi32) -> 2x Narrow(i32->i16) -> 1x Narrow(i16->i8) -> 32 i8.
        private static unsafe long BulkHalfToByte(ushort* p, byte* dst, long count)
        {
            long i = 0;
            for (; i + 32 <= count; i += 32)
            {
                var a = HalfToInt32x8(p, i);
                var b = HalfToInt32x8(p, i + 8);
                var c = HalfToInt32x8(p, i + 16);
                var d = HalfToInt32x8(p, i + 24);
                var s0 = Vector256.Narrow(a, b);
                var s1 = Vector256.Narrow(c, d);
                Vector256.Store(Vector256.Narrow(s0, s1).AsByte(), dst + i);  // 32x i8
            }
            return i;
        }

        // Typed contiguous kernels: SIMD bulk + NumPy-faithful scalar tail (Converts.To{X}(Half)).
        private static unsafe void CastHalfToInt32Contig(void* s, void* d, long n)
        {
            ushort* p = (ushort*)s; int* dst = (int*)d; Half* h = (Half*)s;
            long i = BulkHalfToInt32(p, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToInt32(h[i]);
        }
        private static unsafe void CastHalfToSByteContig(void* s, void* d, long n)
        {
            ushort* p = (ushort*)s; sbyte* dst = (sbyte*)d; Half* h = (Half*)s;
            long i = BulkHalfToByte(p, (byte*)dst, n);
            for (; i < n; i++) dst[i] = Converts.ToSByte(h[i]);
        }
        private static unsafe void CastHalfToByteContig(void* s, void* d, long n)
        {
            ushort* p = (ushort*)s; byte* dst = (byte*)d; Half* h = (Half*)s;
            long i = BulkHalfToByte(p, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToByte(h[i]);
        }
        private static unsafe void CastHalfToInt16Contig(void* s, void* d, long n)
        {
            ushort* p = (ushort*)s; short* dst = (short*)d; Half* h = (Half*)s;
            long i = BulkHalfToShort(p, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToInt16(h[i]);
        }
        private static unsafe void CastHalfToUInt16Contig(void* s, void* d, long n)
        {
            ushort* p = (ushort*)s; ushort* dst = (ushort*)d; Half* h = (Half*)s;
            long i = BulkHalfToShort(p, (short*)dst, n);
            for (; i < n; i++) dst[i] = Converts.ToUInt16(h[i]);
        }
        private static unsafe void CastHalfToCharContig(void* s, void* d, long n)
        {
            ushort* p = (ushort*)s; char* dst = (char*)d; Half* h = (Half*)s;
            long i = BulkHalfToShort(p, (short*)dst, n);
            for (; i < n; i++) dst[i] = Converts.ToChar(h[i]);
        }

        // STRIDED: reuse StridedNarrowDriver (srcSize=2). Bulk void* trampolines + scalar convs.
        private static unsafe long BulkHalfToInt32V(void* s, void* d, long n) => BulkHalfToInt32((ushort*)s, (int*)d, n);
        private static unsafe long BulkHalfToShortV(void* s, void* d, long n) => BulkHalfToShort((ushort*)s, (short*)d, n);
        private static unsafe long BulkHalfToByteV(void* s, void* d, long n)  => BulkHalfToByte((ushort*)s, (byte*)d, n);

        private static unsafe void ConvHalfToInt32(void* s, void* d)  => *(int*)d    = Converts.ToInt32(*(Half*)s);
        private static unsafe void ConvHalfToSByte(void* s, void* d)  => *(sbyte*)d  = Converts.ToSByte(*(Half*)s);
        private static unsafe void ConvHalfToByte(void* s, void* d)   => *(byte*)d   = Converts.ToByte(*(Half*)s);
        private static unsafe void ConvHalfToInt16(void* s, void* d)  => *(short*)d  = Converts.ToInt16(*(Half*)s);
        private static unsafe void ConvHalfToUInt16(void* s, void* d) => *(ushort*)d = Converts.ToUInt16(*(Half*)s);
        private static unsafe void ConvHalfToChar(void* s, void* d)   => *(char*)d   = Converts.ToChar(*(Half*)s);

        internal static unsafe StridedCastKernel TryGetHalfToXStridedKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (srcType != NPTypeCode.Half || !Avx2.IsSupported) return null;
            switch (dstType)
            {
                case NPTypeCode.Int32:  return StridedHalfToInt32;
                case NPTypeCode.UInt32: return StridedHalfToUInt32;
                case NPTypeCode.Int64:  return StridedHalfToInt64;
                case NPTypeCode.UInt64: return StridedHalfToUInt64;
                case NPTypeCode.Single: return StridedHalfToFloat;
                case NPTypeCode.SByte:  return StridedHalfToSByte;
                case NPTypeCode.Byte:   return StridedHalfToByte;
                case NPTypeCode.Int16:  return StridedHalfToInt16;
                case NPTypeCode.UInt16: return StridedHalfToUInt16;
                case NPTypeCode.Char:   return StridedHalfToChar;
            }
            return null;
        }

        private static unsafe void StridedHalfToInt32(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 4, &BulkHalfToInt32V, &ConvHalfToInt32);
        private static unsafe void StridedHalfToUInt32(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 4, &BulkHalfToUInt32V, &ConvHalfToUInt32);
        private static unsafe void StridedHalfToFloat(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 4, &BulkHalfToFloatV, &ConvHalfToFloat);
        private static unsafe void StridedHalfToSByte(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 1, &BulkHalfToByteV,  &ConvHalfToSByte);
        private static unsafe void StridedHalfToByte(void* s, void* d, long* ss, long* ds, long* sh, int nd)   => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 1, &BulkHalfToByteV,  &ConvHalfToByte);
        private static unsafe void StridedHalfToInt16(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 2, &BulkHalfToShortV, &ConvHalfToInt16);
        private static unsafe void StridedHalfToUInt16(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 2, &BulkHalfToShortV, &ConvHalfToUInt16);
        private static unsafe void StridedHalfToChar(void* s, void* d, long* ss, long* ds, long* sh, int nd)   => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 2, &BulkHalfToShortV, &ConvHalfToChar);
    }
}
