using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.ToBool.cs
        //   {int,float} -> Boolean contiguous + strided casts.
        //
        //   Phase-0: '*->bool' was the single worst dst column (geomean 0.61),
        //   111 lagging cells — because ResolveStrategy excludes Boolean from
        //   IsIntegerCast, so EVERY cast to bool fell to the per-element IL scalar
        //   while NumPy vectorizes the `!= 0` compare. Pure headroom.
        //
        //   MECHANISM (bit-exact with Converts.ToBoolean(x) == (x != 0), hence
        //   NumPy 2.4.2 — incl. float NaN->True (NaN != 0) and -0.0->False
        //   (-0.0 == 0.0)):
        //     mask = ~Vector.Equals(v, 0)        // all-ones lanes where v != 0
        //     ones = mask & 1                    // 0/1 per lane
        //     truncating Vector.Narrow down to bytes (0/1)   // reuses the narrow chain
        //   Floats compare in float/double (so -0.0 and NaN are handled by IEEE
        //   equality, NOT by an integer bit test). Output is 1 byte (0/1) per elem.
        //
        //   Reuses StridedNarrowDriver (dstSize=1) for every non-contiguous layout
        //   — same inner-contig Bulk / staged-strided / odometer machinery as the
        //   float->narrow kernels. Bit-exact: Bulk and the scalar tail/conv both
        //   reduce to `!= 0`.
        // =====================================================================

        internal static unsafe CastKernel TryGetToBoolKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (dstType != NPTypeCode.Boolean) return null;
            switch (srcType)
            {
                case NPTypeCode.Byte:   return CastByteToBoolContig;
                case NPTypeCode.SByte:  return CastSByteToBoolContig;
                case NPTypeCode.Int16:  return CastInt16ToBoolContig;
                case NPTypeCode.UInt16: return CastUInt16ToBoolContig;
                case NPTypeCode.Char:   return CastCharToBoolContig;
                case NPTypeCode.Int32:  return CastInt32ToBoolContig;
                case NPTypeCode.UInt32: return CastUInt32ToBoolContig;
                case NPTypeCode.Int64:  return CastInt64ToBoolContig;
                case NPTypeCode.UInt64: return CastUInt64ToBoolContig;
                case NPTypeCode.Single: return CastSingleToBoolContig;
                case NPTypeCode.Double: return CastDoubleToBoolContig;
            }
            return null;
        }

        internal static unsafe StridedCastKernel TryGetToBoolStridedKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (dstType != NPTypeCode.Boolean) return null;
            switch (srcType)
            {
                case NPTypeCode.Byte:   return StridedByteToBool;
                case NPTypeCode.SByte:  return StridedSByteToBool;
                case NPTypeCode.Int16:  return StridedInt16ToBool;
                case NPTypeCode.UInt16: return StridedUInt16ToBool;
                case NPTypeCode.Char:   return StridedCharToBool;
                case NPTypeCode.Int32:  return StridedInt32ToBool;
                case NPTypeCode.UInt32: return StridedUInt32ToBool;
                case NPTypeCode.Int64:  return StridedInt64ToBool;
                case NPTypeCode.UInt64: return StridedUInt64ToBool;
                case NPTypeCode.Single: return StridedSingleToBool;
                case NPTypeCode.Double: return StridedDoubleToBool;
            }
            return null;
        }

        // -------- SIMD bulk loops by source width (output 0/1 bytes) -----------

        // 1-byte src: 32-wide, no narrow.
        private static unsafe long BulkByteToBool(byte* src, byte* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var one = Vector256.Create((byte)1);
                for (; i + 32 <= count; i += 32)
                    Vector256.Store((~Vector256.Equals(Vector256.Load(src + i), Vector256<byte>.Zero)) & one, dst + i);
            }
            return i;
        }

        // 2-byte src: 32-wide -> 2x compare -> Narrow(short->byte).
        private static unsafe long BulkInt16ToBool(short* src, byte* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var one = Vector256.Create((short)1);
                for (; i + 32 <= count; i += 32)
                {
                    var a = (~Vector256.Equals(Vector256.Load(src + i), Vector256<short>.Zero)) & one;
                    var b = (~Vector256.Equals(Vector256.Load(src + i + 16), Vector256<short>.Zero)) & one;
                    Vector256.Store(Vector256.Narrow(a, b).AsByte(), dst + i);
                }
            }
            return i;
        }

        // 4-byte int src: 32-wide -> 4x compare -> Narrow(int->short->byte).
        private static unsafe long BulkInt32ToBool(int* src, byte* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var one = Vector256.Create(1);
                for (; i + 32 <= count; i += 32)
                {
                    var a = (~Vector256.Equals(Vector256.Load(src + i), Vector256<int>.Zero)) & one;
                    var b = (~Vector256.Equals(Vector256.Load(src + i + 8), Vector256<int>.Zero)) & one;
                    var c = (~Vector256.Equals(Vector256.Load(src + i + 16), Vector256<int>.Zero)) & one;
                    var d = (~Vector256.Equals(Vector256.Load(src + i + 24), Vector256<int>.Zero)) & one;
                    Vector256.Store(Vector256.Narrow(Vector256.Narrow(a, b), Vector256.Narrow(c, d)).AsByte(), dst + i);
                }
            }
            return i;
        }

        // 8-byte int src: 32-wide -> 8x compare -> Narrow(long->int->short->byte).
        private static unsafe long BulkInt64ToBool(long* src, byte* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var one = Vector256.Create(1L);
                for (; i + 32 <= count; i += 32)
                {
                    var a = (~Vector256.Equals(Vector256.Load(src + i),      Vector256<long>.Zero)) & one;
                    var b = (~Vector256.Equals(Vector256.Load(src + i + 4),  Vector256<long>.Zero)) & one;
                    var c = (~Vector256.Equals(Vector256.Load(src + i + 8),  Vector256<long>.Zero)) & one;
                    var d = (~Vector256.Equals(Vector256.Load(src + i + 12), Vector256<long>.Zero)) & one;
                    var e = (~Vector256.Equals(Vector256.Load(src + i + 16), Vector256<long>.Zero)) & one;
                    var f = (~Vector256.Equals(Vector256.Load(src + i + 20), Vector256<long>.Zero)) & one;
                    var g = (~Vector256.Equals(Vector256.Load(src + i + 24), Vector256<long>.Zero)) & one;
                    var h = (~Vector256.Equals(Vector256.Load(src + i + 28), Vector256<long>.Zero)) & one;
                    var s0 = Vector256.Narrow(Vector256.Narrow(a, b), Vector256.Narrow(c, d));   // 16 short
                    var s1 = Vector256.Narrow(Vector256.Narrow(e, f), Vector256.Narrow(g, h));   // 16 short
                    Vector256.Store(Vector256.Narrow(s0, s1).AsByte(), dst + i);
                }
            }
            return i;
        }

        // 4-byte float src: compare in float (handles -0.0/NaN), then narrow the int mask.
        private static unsafe long BulkSingleToBool(float* src, byte* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var one = Vector256.Create(1);
                for (; i + 32 <= count; i += 32)
                {
                    var a = (~Vector256.Equals(Vector256.Load(src + i),      Vector256<float>.Zero).AsInt32()) & one;
                    var b = (~Vector256.Equals(Vector256.Load(src + i + 8),  Vector256<float>.Zero).AsInt32()) & one;
                    var c = (~Vector256.Equals(Vector256.Load(src + i + 16), Vector256<float>.Zero).AsInt32()) & one;
                    var d = (~Vector256.Equals(Vector256.Load(src + i + 24), Vector256<float>.Zero).AsInt32()) & one;
                    Vector256.Store(Vector256.Narrow(Vector256.Narrow(a, b), Vector256.Narrow(c, d)).AsByte(), dst + i);
                }
            }
            return i;
        }

        // 8-byte double src: compare in double, then narrow the long mask.
        private static unsafe long BulkDoubleToBool(double* src, byte* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var one = Vector256.Create(1L);
                for (; i + 32 <= count; i += 32)
                {
                    var a = (~Vector256.Equals(Vector256.Load(src + i),      Vector256<double>.Zero).AsInt64()) & one;
                    var b = (~Vector256.Equals(Vector256.Load(src + i + 4),  Vector256<double>.Zero).AsInt64()) & one;
                    var c = (~Vector256.Equals(Vector256.Load(src + i + 8),  Vector256<double>.Zero).AsInt64()) & one;
                    var d = (~Vector256.Equals(Vector256.Load(src + i + 12), Vector256<double>.Zero).AsInt64()) & one;
                    var e = (~Vector256.Equals(Vector256.Load(src + i + 16), Vector256<double>.Zero).AsInt64()) & one;
                    var f = (~Vector256.Equals(Vector256.Load(src + i + 20), Vector256<double>.Zero).AsInt64()) & one;
                    var g = (~Vector256.Equals(Vector256.Load(src + i + 24), Vector256<double>.Zero).AsInt64()) & one;
                    var h = (~Vector256.Equals(Vector256.Load(src + i + 28), Vector256<double>.Zero).AsInt64()) & one;
                    var s0 = Vector256.Narrow(Vector256.Narrow(a, b), Vector256.Narrow(c, d));
                    var s1 = Vector256.Narrow(Vector256.Narrow(e, f), Vector256.Narrow(g, h));
                    Vector256.Store(Vector256.Narrow(s0, s1).AsByte(), dst + i);
                }
            }
            return i;
        }

        // void* bulk trampolines (one per source width; signedness is irrelevant to `!= 0`).
        private static unsafe long BulkByteToBoolV(void* s, void* d, long n)   => BulkByteToBool((byte*)s, (byte*)d, n);
        private static unsafe long BulkInt16ToBoolV(void* s, void* d, long n)  => BulkInt16ToBool((short*)s, (byte*)d, n);
        private static unsafe long BulkInt32ToBoolV(void* s, void* d, long n)  => BulkInt32ToBool((int*)s, (byte*)d, n);
        private static unsafe long BulkInt64ToBoolV(void* s, void* d, long n)  => BulkInt64ToBool((long*)s, (byte*)d, n);
        private static unsafe long BulkSingleToBoolV(void* s, void* d, long n) => BulkSingleToBool((float*)s, (byte*)d, n);
        private static unsafe long BulkDoubleToBoolV(void* s, void* d, long n) => BulkDoubleToBool((double*)s, (byte*)d, n);

        // Scalar element converts — bit-exact NumPy-faithful Converts.ToBoolean (== `!= 0`).
        private static unsafe void ConvByteToBool(void* s, void* d)   => *(byte*)d = Converts.ToBoolean(*(byte*)s)   ? (byte)1 : (byte)0;
        private static unsafe void ConvSByteToBool(void* s, void* d)  => *(byte*)d = Converts.ToBoolean(*(sbyte*)s)  ? (byte)1 : (byte)0;
        private static unsafe void ConvInt16ToBool(void* s, void* d)  => *(byte*)d = Converts.ToBoolean(*(short*)s)  ? (byte)1 : (byte)0;
        private static unsafe void ConvUInt16ToBool(void* s, void* d) => *(byte*)d = Converts.ToBoolean(*(ushort*)s) ? (byte)1 : (byte)0;
        private static unsafe void ConvCharToBool(void* s, void* d)   => *(byte*)d = Converts.ToBoolean(*(char*)s)   ? (byte)1 : (byte)0;
        private static unsafe void ConvInt32ToBool(void* s, void* d)  => *(byte*)d = Converts.ToBoolean(*(int*)s)    ? (byte)1 : (byte)0;
        private static unsafe void ConvUInt32ToBool(void* s, void* d) => *(byte*)d = Converts.ToBoolean(*(uint*)s)   ? (byte)1 : (byte)0;
        private static unsafe void ConvInt64ToBool(void* s, void* d)  => *(byte*)d = Converts.ToBoolean(*(long*)s)   ? (byte)1 : (byte)0;
        private static unsafe void ConvUInt64ToBool(void* s, void* d) => *(byte*)d = Converts.ToBoolean(*(ulong*)s)  ? (byte)1 : (byte)0;
        private static unsafe void ConvSingleToBool(void* s, void* d) => *(byte*)d = Converts.ToBoolean(*(float*)s)  ? (byte)1 : (byte)0;
        private static unsafe void ConvDoubleToBool(void* s, void* d) => *(byte*)d = Converts.ToBoolean(*(double*)s) ? (byte)1 : (byte)0;

        // -------- Contiguous kernels: SIMD bulk + scalar tail ------------------
        private static unsafe void CastByteToBoolContig(void* s, void* d, long n)   { byte* src = (byte*)s; byte* dst = (byte*)d;   long i = BulkByteToBool(src, dst, n);   for (; i < n; i++) dst[i] = Converts.ToBoolean(src[i]) ? (byte)1 : (byte)0; }
        private static unsafe void CastSByteToBoolContig(void* s, void* d, long n)  { sbyte* src = (sbyte*)s; byte* dst = (byte*)d; long i = BulkByteToBool((byte*)src, dst, n); for (; i < n; i++) dst[i] = Converts.ToBoolean(src[i]) ? (byte)1 : (byte)0; }
        private static unsafe void CastInt16ToBoolContig(void* s, void* d, long n)  { short* src = (short*)s; byte* dst = (byte*)d; long i = BulkInt16ToBool(src, dst, n);  for (; i < n; i++) dst[i] = Converts.ToBoolean(src[i]) ? (byte)1 : (byte)0; }
        private static unsafe void CastUInt16ToBoolContig(void* s, void* d, long n) { ushort* src = (ushort*)s; byte* dst = (byte*)d; long i = BulkInt16ToBool((short*)src, dst, n); for (; i < n; i++) dst[i] = Converts.ToBoolean(src[i]) ? (byte)1 : (byte)0; }
        private static unsafe void CastCharToBoolContig(void* s, void* d, long n)   { char* src = (char*)s; byte* dst = (byte*)d;   long i = BulkInt16ToBool((short*)src, dst, n); for (; i < n; i++) dst[i] = Converts.ToBoolean(src[i]) ? (byte)1 : (byte)0; }
        private static unsafe void CastInt32ToBoolContig(void* s, void* d, long n)  { int* src = (int*)s; byte* dst = (byte*)d;     long i = BulkInt32ToBool(src, dst, n);  for (; i < n; i++) dst[i] = Converts.ToBoolean(src[i]) ? (byte)1 : (byte)0; }
        private static unsafe void CastUInt32ToBoolContig(void* s, void* d, long n) { uint* src = (uint*)s; byte* dst = (byte*)d;   long i = BulkInt32ToBool((int*)src, dst, n); for (; i < n; i++) dst[i] = Converts.ToBoolean(src[i]) ? (byte)1 : (byte)0; }
        private static unsafe void CastInt64ToBoolContig(void* s, void* d, long n)  { long* src = (long*)s; byte* dst = (byte*)d;   long i = BulkInt64ToBool(src, dst, n);  for (; i < n; i++) dst[i] = Converts.ToBoolean(src[i]) ? (byte)1 : (byte)0; }
        private static unsafe void CastUInt64ToBoolContig(void* s, void* d, long n) { ulong* src = (ulong*)s; byte* dst = (byte*)d; long i = BulkInt64ToBool((long*)src, dst, n); for (; i < n; i++) dst[i] = Converts.ToBoolean(src[i]) ? (byte)1 : (byte)0; }
        private static unsafe void CastSingleToBoolContig(void* s, void* d, long n) { float* src = (float*)s; byte* dst = (byte*)d; long i = BulkSingleToBool(src, dst, n); for (; i < n; i++) dst[i] = Converts.ToBoolean(src[i]) ? (byte)1 : (byte)0; }
        private static unsafe void CastDoubleToBoolContig(void* s, void* d, long n) { double* src = (double*)s; byte* dst = (byte*)d; long i = BulkDoubleToBool(src, dst, n); for (; i < n; i++) dst[i] = Converts.ToBoolean(src[i]) ? (byte)1 : (byte)0; }

        // -------- Strided kernels: reuse StridedNarrowDriver (dstSize=1) --------
        private static unsafe void StridedByteToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd)   => StridedNarrowDriver(s, d, ss, ds, sh, nd, 1, 1, &BulkByteToBoolV,   &ConvByteToBool);
        private static unsafe void StridedSByteToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 1, 1, &BulkByteToBoolV,   &ConvSByteToBool);
        private static unsafe void StridedInt16ToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 1, &BulkInt16ToBoolV,  &ConvInt16ToBool);
        private static unsafe void StridedUInt16ToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 1, &BulkInt16ToBoolV,  &ConvUInt16ToBool);
        private static unsafe void StridedCharToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd)   => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 1, &BulkInt16ToBoolV,  &ConvCharToBool);
        private static unsafe void StridedInt32ToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 1, &BulkInt32ToBoolV,  &ConvInt32ToBool);
        private static unsafe void StridedUInt32ToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 1, &BulkInt32ToBoolV,  &ConvUInt32ToBool);
        private static unsafe void StridedInt64ToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 8, 1, &BulkInt64ToBoolV,  &ConvInt64ToBool);
        private static unsafe void StridedUInt64ToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 8, 1, &BulkInt64ToBoolV,  &ConvUInt64ToBool);
        private static unsafe void StridedSingleToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 1, &BulkSingleToBoolV, &ConvSingleToBool);
        private static unsafe void StridedDoubleToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 8, 1, &BulkDoubleToBoolV, &ConvDoubleToBool);
    }
}
