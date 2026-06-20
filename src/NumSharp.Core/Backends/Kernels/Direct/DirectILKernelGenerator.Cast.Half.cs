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
        //   float16 (Half) -> {bool, i8, u8, i16, u16, char, i32} casts.
        //
        //   f16 was a lagging SOURCE in Phase 0 (geomean 0.69) — every f16->X fell
        //   to the IL scalar (Half is not IsFloatCast), and .NET 10 exposes no
        //   vectorized VCVTPH2PS. The widen below is the branchless Giesen
        //   "half_to_float_fast5" (shift + a single float multiply that rebiases
        //   normals AND subnormals uniformly, + one compare to keep Inf/NaN),
        //   EXHAUSTIVELY verified bit-exact vs (float)Half over all 65536 halves
        //   (modulo NaN *payload* — which is irrelevant here: every consumer is
        //   either ->int, where cvtt collapses NaN to INT_MIN, or ->bool, which
        //   tests the bit pattern directly and never widens).
        //
        //   f16->bool needs NO float: a half is truthy iff (bits & 0x7FFF) != 0
        //   (±0 -> false; Inf/NaN/subnormal -> true) — exactly Converts.ToBoolean.
        //   f16->int = widen -> cvtt (-> i32) [-> truncating Narrow for i8/i16].
        //   Bit-exact with Converts.To{X}(Half) for ALL halves (the scalar tail and
        //   the NaN/Inf -> INT_MIN sentinel both agree).
        // =====================================================================

        // 8 raw half-bit lanes -> 8 floats. Verified vs (float)Half over all 65536.
        private static Vector256<float> WidenHalf8(Vector128<ushort> hb)
        {
            var h = Avx2.ConvertToVector256Int32(hb).AsUInt32();             // zero-extend 8x u16 -> u32
            var o = (h & Vector256.Create(0x7fffu)) << 13;                  // exp/mantissa to f32 slot
            var of = o.AsSingle() * Vector256.Create(0x77800000u).AsSingle();   // * 2^112 (rebias norm+denorm)
            var inf = Vector256.GreaterThanOrEqual(of, Vector256.Create(0x47800000u).AsSingle()).AsUInt32()
                      & Vector256.Create(255u << 23);                       // force exp=255 where Inf/NaN
            var sign = (h & Vector256.Create(0x8000u)) << 16;
            return (of.AsUInt32() | inf | sign).AsSingle();
        }

        internal static unsafe CastKernel TryGetHalfToXKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (srcType != NPTypeCode.Half || !Avx2.IsSupported) return null;
            switch (dstType)
            {
                case NPTypeCode.Boolean: return CastHalfToBoolContig;
                case NPTypeCode.SByte:   return CastHalfToSByteContig;
                case NPTypeCode.Byte:    return CastHalfToByteContig;
                case NPTypeCode.Int16:   return CastHalfToInt16Contig;
                case NPTypeCode.UInt16:  return CastHalfToUInt16Contig;
                case NPTypeCode.Char:    return CastHalfToCharContig;
                case NPTypeCode.Int32:   return CastHalfToInt32Contig;
            }
            return null;
        }

        internal static unsafe StridedCastKernel TryGetHalfToXStridedKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (srcType != NPTypeCode.Half || !Avx2.IsSupported) return null;
            switch (dstType)
            {
                case NPTypeCode.Boolean: return StridedHalfToBool;
                case NPTypeCode.SByte:   return StridedHalfToSByte;
                case NPTypeCode.Byte:    return StridedHalfToByte;
                case NPTypeCode.Int16:   return StridedHalfToInt16;
                case NPTypeCode.UInt16:  return StridedHalfToUInt16;
                case NPTypeCode.Char:    return StridedHalfToChar;
                case NPTypeCode.Int32:   return StridedHalfToInt32;
            }
            return null;
        }

        // -------- SIMD bulks (src = raw half bits via ushort*) -----------------

        // 32 halves -> truthy byte (bits & 0x7FFF != 0). No float.
        private static unsafe long BulkHalfToBool(ushort* src, byte* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var mask = Vector256.Create((ushort)0x7fff);
                var one = Vector256.Create((ushort)1);
                for (; i + 32 <= count; i += 32)
                {
                    var a = (~Vector256.Equals(Vector256.Load(src + i) & mask, Vector256<ushort>.Zero)) & one;
                    var b = (~Vector256.Equals(Vector256.Load(src + i + 16) & mask, Vector256<ushort>.Zero)) & one;
                    Vector256.Store(Vector256.Narrow(a, b), dst + i);
                }
            }
            return i;
        }

        // 32 halves -> widen -> cvtt -> Narrow(i32->i16->i8) -> 32 bytes.
        private static unsafe long BulkHalfToByte(ushort* src, byte* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                for (; i + 32 <= count; i += 32)
                {
                    var a = Avx.ConvertToVector256Int32WithTruncation(WidenHalf8(Vector128.Load(src + i)));
                    var b = Avx.ConvertToVector256Int32WithTruncation(WidenHalf8(Vector128.Load(src + i + 8)));
                    var c = Avx.ConvertToVector256Int32WithTruncation(WidenHalf8(Vector128.Load(src + i + 16)));
                    var d = Avx.ConvertToVector256Int32WithTruncation(WidenHalf8(Vector128.Load(src + i + 24)));
                    Vector256.Store(Vector256.Narrow(Vector256.Narrow(a, b), Vector256.Narrow(c, d)).AsByte(), dst + i);
                }
            }
            return i;
        }

        // 16 halves -> widen -> cvtt -> Narrow(i32->i16) -> 16 shorts.
        private static unsafe long BulkHalfToShort(ushort* src, short* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                for (; i + 16 <= count; i += 16)
                {
                    var a = Avx.ConvertToVector256Int32WithTruncation(WidenHalf8(Vector128.Load(src + i)));
                    var b = Avx.ConvertToVector256Int32WithTruncation(WidenHalf8(Vector128.Load(src + i + 8)));
                    Vector256.Store(Vector256.Narrow(a, b), dst + i);
                }
            }
            return i;
        }

        // 8 halves -> widen -> cvtt -> 8 i32.
        private static unsafe long BulkHalfToInt32(ushort* src, int* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                for (; i + 8 <= count; i += 8)
                    Vector256.Store(Avx.ConvertToVector256Int32WithTruncation(WidenHalf8(Vector128.Load(src + i))), dst + i);
            }
            return i;
        }

        private static unsafe long BulkHalfToBoolV(void* s, void* d, long n)  => BulkHalfToBool((ushort*)s, (byte*)d, n);
        private static unsafe long BulkHalfToByteV(void* s, void* d, long n)  => BulkHalfToByte((ushort*)s, (byte*)d, n);
        private static unsafe long BulkHalfToShortV(void* s, void* d, long n) => BulkHalfToShort((ushort*)s, (short*)d, n);
        private static unsafe long BulkHalfToInt32V(void* s, void* d, long n) => BulkHalfToInt32((ushort*)s, (int*)d, n);

        private static unsafe void ConvHalfToBool(void* s, void* d)  => *(byte*)d  = Converts.ToBoolean(*(Half*)s) ? (byte)1 : (byte)0;
        private static unsafe void ConvHalfToSByte(void* s, void* d) => *(sbyte*)d = Converts.ToSByte(*(Half*)s);
        private static unsafe void ConvHalfToByte(void* s, void* d)  => *(byte*)d  = Converts.ToByte(*(Half*)s);
        private static unsafe void ConvHalfToInt16(void* s, void* d) => *(short*)d = Converts.ToInt16(*(Half*)s);
        private static unsafe void ConvHalfToUInt16(void* s, void* d)=> *(ushort*)d= Converts.ToUInt16(*(Half*)s);
        private static unsafe void ConvHalfToChar(void* s, void* d)  => *(char*)d  = Converts.ToChar(*(Half*)s);
        private static unsafe void ConvHalfToInt32(void* s, void* d) => *(int*)d   = Converts.ToInt32(*(Half*)s);

        // -------- Contiguous kernels: SIMD bulk + scalar tail ------------------
        private static unsafe void CastHalfToBoolContig(void* s, void* d, long n)   { ushort* src = (ushort*)s; byte* dst = (byte*)d;  long i = BulkHalfToBool(src, dst, n);  for (; i < n; i++) dst[i] = Converts.ToBoolean(((Half*)s)[i]) ? (byte)1 : (byte)0; }
        private static unsafe void CastHalfToSByteContig(void* s, void* d, long n)  { ushort* src = (ushort*)s; sbyte* dst = (sbyte*)d; long i = BulkHalfToByte(src, (byte*)dst, n); for (; i < n; i++) dst[i] = Converts.ToSByte(((Half*)s)[i]); }
        private static unsafe void CastHalfToByteContig(void* s, void* d, long n)   { ushort* src = (ushort*)s; byte* dst = (byte*)d;  long i = BulkHalfToByte(src, dst, n);  for (; i < n; i++) dst[i] = Converts.ToByte(((Half*)s)[i]); }
        private static unsafe void CastHalfToInt16Contig(void* s, void* d, long n)  { ushort* src = (ushort*)s; short* dst = (short*)d; long i = BulkHalfToShort(src, dst, n); for (; i < n; i++) dst[i] = Converts.ToInt16(((Half*)s)[i]); }
        private static unsafe void CastHalfToUInt16Contig(void* s, void* d, long n) { ushort* src = (ushort*)s; ushort* dst = (ushort*)d; long i = BulkHalfToShort(src, (short*)dst, n); for (; i < n; i++) dst[i] = Converts.ToUInt16(((Half*)s)[i]); }
        private static unsafe void CastHalfToCharContig(void* s, void* d, long n)   { ushort* src = (ushort*)s; char* dst = (char*)d;   long i = BulkHalfToShort(src, (short*)dst, n); for (; i < n; i++) dst[i] = Converts.ToChar(((Half*)s)[i]); }
        private static unsafe void CastHalfToInt32Contig(void* s, void* d, long n)  { ushort* src = (ushort*)s; int* dst = (int*)d;     long i = BulkHalfToInt32(src, dst, n); for (; i < n; i++) dst[i] = Converts.ToInt32(((Half*)s)[i]); }

        // -------- Strided kernels: reuse StridedNarrowDriver (srcSize=2) --------
        private static unsafe void StridedHalfToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd)   => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 1, &BulkHalfToBoolV,  &ConvHalfToBool);
        private static unsafe void StridedHalfToSByte(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 1, &BulkHalfToByteV,  &ConvHalfToSByte);
        private static unsafe void StridedHalfToByte(void* s, void* d, long* ss, long* ds, long* sh, int nd)   => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 1, &BulkHalfToByteV,  &ConvHalfToByte);
        private static unsafe void StridedHalfToInt16(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 2, &BulkHalfToShortV, &ConvHalfToInt16);
        private static unsafe void StridedHalfToUInt16(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 2, &BulkHalfToShortV, &ConvHalfToUInt16);
        private static unsafe void StridedHalfToChar(void* s, void* d, long* ss, long* ds, long* sh, int nd)   => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 2, &BulkHalfToShortV, &ConvHalfToChar);
        private static unsafe void StridedHalfToInt32(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 4, &BulkHalfToInt32V, &ConvHalfToInt32);
    }
}
