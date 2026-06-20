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
        //   {int,float,half,char} -> bool contiguous + strided casts.
        //
        //   Phase-0's worst dst COLUMN (*->bool ~0.61-0.94x): bool is byte-per-element
        //   0/1; NumPy stores (v != 0). The scalar path produced 0/1 per element; this
        //   vectorizes the compare.
        //
        //   MECHANISM (bit-exact with Converts.ToBoolean(T)):
        //     boolByte = AndNot(CompareEqual(v, 0), 1)   // 1 where v!=0 else 0
        //       int:   integer vpcmpeq* (== 0)
        //       float: OrderedEqualNonSignaling (so -0.0 == 0 -> False, NaN unordered -> True),
        //              bit-exact NumPy `!= 0`.
        //       half:  (bits & 0x7fff) == 0  ->  -0.0/+0.0 -> False, NaN/inf -> True.
        //     then truncating Vector.Narrow the 0/1 lanes down to bytes.
        //
        //   i8/u8 are 1-byte already (compare + &1, no narrow). Strided reuses
        //   StridedNarrowDriver (srcSize = element width, dstSize = 1).
        //   Complex/Decimal -> bool stay on the scalar path (Complex needs re|im OR;
        //   Decimal has no SIMD compare).
        // =====================================================================

        internal static unsafe CastKernel TryGetToBoolKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (dstType != NPTypeCode.Boolean || !Avx2.IsSupported) return null;
            switch (srcType)
            {
                case NPTypeCode.Single:  return CastSingleToBoolContig;
                case NPTypeCode.Double:  return CastDoubleToBoolContig;
                case NPTypeCode.Int32:   return CastInt32ToBoolContig;
                case NPTypeCode.UInt32:  return CastInt32ToBoolContig;
                case NPTypeCode.Int64:   return CastInt64ToBoolContig;
                case NPTypeCode.UInt64:  return CastInt64ToBoolContig;
                case NPTypeCode.Int16:   return CastInt16ToBoolContig;
                case NPTypeCode.UInt16:  return CastInt16ToBoolContig;
                case NPTypeCode.Char:    return CastInt16ToBoolContig;
                case NPTypeCode.SByte:   return CastByteToBoolContig;
                case NPTypeCode.Byte:    return CastByteToBoolContig;
                case NPTypeCode.Half:    return CastHalfToBoolContig;
            }
            return null;
        }

        // ---- bulk loops: write 0/1 bytes, return elements consumed ----

        private static unsafe long BulkSingleToBool(float* src, byte* dst, long n)
        {
            long i = 0; var z = Vector256<float>.Zero; var one = Vector256.Create(1);
            for (; i + 32 <= n; i += 32)
            {
                var m0 = Avx2.AndNot(Avx.Compare(Vector256.Load(src + i), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt32(), one);
                var m1 = Avx2.AndNot(Avx.Compare(Vector256.Load(src + i + 8), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt32(), one);
                var m2 = Avx2.AndNot(Avx.Compare(Vector256.Load(src + i + 16), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt32(), one);
                var m3 = Avx2.AndNot(Avx.Compare(Vector256.Load(src + i + 24), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt32(), one);
                var s0 = Vector256.Narrow(m0, m1); var s1 = Vector256.Narrow(m2, m3);
                Vector256.Store(Vector256.Narrow(s0, s1).AsByte(), dst + i);
            }
            return i;
        }

        private static unsafe long BulkDoubleToBool(double* src, byte* dst, long n)
        {
            long i = 0; var z = Vector256<double>.Zero; var one = Vector256.Create(1L);
            for (; i + 32 <= n; i += 32)
            {
                // 8 loads (4 doubles each) -> 0/1 i64; Narrow pairs preserve element order
                // (Narrow(lower,upper) = [lower | upper]); 3 narrow levels -> 32 bytes.
                var m0 = Avx2.AndNot(Avx.Compare(Vector256.Load(src + i + 0),  z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                var m1 = Avx2.AndNot(Avx.Compare(Vector256.Load(src + i + 8),  z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                var m2 = Avx2.AndNot(Avx.Compare(Vector256.Load(src + i + 16), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                var m3 = Avx2.AndNot(Avx.Compare(Vector256.Load(src + i + 24), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                var m4 = Avx2.AndNot(Avx.Compare(Vector256.Load(src + i + 4),  z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                var m5 = Avx2.AndNot(Avx.Compare(Vector256.Load(src + i + 12), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                var m6 = Avx2.AndNot(Avx.Compare(Vector256.Load(src + i + 20), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                var m7 = Avx2.AndNot(Avx.Compare(Vector256.Load(src + i + 28), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                var a0 = Vector256.Narrow(m0, m4); var a1 = Vector256.Narrow(m1, m5);
                var a2 = Vector256.Narrow(m2, m6); var a3 = Vector256.Narrow(m3, m7);
                var s0 = Vector256.Narrow(a0, a1); var s1 = Vector256.Narrow(a2, a3);
                Vector256.Store(Vector256.Narrow(s0, s1).AsByte(), dst + i);
            }
            return i;
        }

        private static unsafe long BulkInt32ToBool(int* src, byte* dst, long n)
        {
            long i = 0; var z = Vector256<int>.Zero; var one = Vector256.Create(1);
            for (; i + 32 <= n; i += 32)
            {
                var m0 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i), z), one);
                var m1 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i + 8), z), one);
                var m2 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i + 16), z), one);
                var m3 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i + 24), z), one);
                var s0 = Vector256.Narrow(m0, m1); var s1 = Vector256.Narrow(m2, m3);
                Vector256.Store(Vector256.Narrow(s0, s1).AsByte(), dst + i);
            }
            return i;
        }

        private static unsafe long BulkInt64ToBool(long* src, byte* dst, long n)
        {
            long i = 0; var z = Vector256<long>.Zero; var one = Vector256.Create(1L);
            for (; i + 32 <= n; i += 32)
            {
                var m0 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i), z), one);
                var m1 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i + 8), z), one);
                var m2 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i + 16), z), one);
                var m3 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i + 24), z), one);
                var m4 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i + 4), z), one);
                var m5 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i + 12), z), one);
                var m6 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i + 20), z), one);
                var m7 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i + 28), z), one);
                // interleave-safe pairing: a-lane holds 0,1,2,3 then 4,5,6,7 indices via Narrow order
                var a0 = Vector256.Narrow(m0, m4); var a1 = Vector256.Narrow(m1, m5);
                var a2 = Vector256.Narrow(m2, m6); var a3 = Vector256.Narrow(m3, m7);
                var s0 = Vector256.Narrow(a0, a1); var s1 = Vector256.Narrow(a2, a3);
                Vector256.Store(Vector256.Narrow(s0, s1).AsByte(), dst + i);
            }
            return i;
        }

        private static unsafe long BulkInt16ToBool(short* src, byte* dst, long n)
        {
            long i = 0; var z = Vector256<short>.Zero; var one = Vector256.Create((short)1);
            for (; i + 32 <= n; i += 32)
            {
                var m0 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i), z), one);
                var m1 = Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i + 16), z), one);
                Vector256.Store(Vector256.Narrow(m0, m1).AsByte(), dst + i);
            }
            return i;
        }

        private static unsafe long BulkByteToBool(byte* src, byte* dst, long n)
        {
            long i = 0; var z = Vector256<byte>.Zero; var one = Vector256.Create((byte)1);
            for (; i + 32 <= n; i += 32)
                Vector256.Store(Avx2.AndNot(Avx2.CompareEqual(Vector256.Load(src + i), z), one), dst + i);
            return i;
        }

        private static unsafe long BulkHalfToBool(ushort* src, byte* dst, long n)
        {
            long i = 0; var z = Vector256<short>.Zero; var one = Vector256.Create((short)1);
            var mag = Vector256.Create((short)0x7fff);
            for (; i + 32 <= n; i += 32)
            {
                var v0 = Avx2.And(Vector256.Load(src + i).AsInt16(), mag);
                var v1 = Avx2.And(Vector256.Load(src + i + 16).AsInt16(), mag);
                var m0 = Avx2.AndNot(Avx2.CompareEqual(v0, z), one);
                var m1 = Avx2.AndNot(Avx2.CompareEqual(v1, z), one);
                Vector256.Store(Vector256.Narrow(m0, m1).AsByte(), dst + i);
            }
            return i;
        }

        // ---- typed contig kernels: bulk + scalar tail ----
        private static unsafe void CastSingleToBoolContig(void* s, void* d, long n) { float* src = (float*)s; bool* dst = (bool*)d; long i = BulkSingleToBool(src, (byte*)dst, n); for (; i < n; i++) dst[i] = Converts.ToBoolean(src[i]); }
        private static unsafe void CastDoubleToBoolContig(void* s, void* d, long n) { double* src = (double*)s; bool* dst = (bool*)d; long i = BulkDoubleToBool(src, (byte*)dst, n); for (; i < n; i++) dst[i] = Converts.ToBoolean(src[i]); }
        private static unsafe void CastInt32ToBoolContig(void* s, void* d, long n) { int* src = (int*)s; bool* dst = (bool*)d; long i = BulkInt32ToBool(src, (byte*)dst, n); for (; i < n; i++) dst[i] = src[i] != 0; }
        private static unsafe void CastInt64ToBoolContig(void* s, void* d, long n) { long* src = (long*)s; bool* dst = (bool*)d; long i = BulkInt64ToBool(src, (byte*)dst, n); for (; i < n; i++) dst[i] = src[i] != 0; }
        private static unsafe void CastInt16ToBoolContig(void* s, void* d, long n) { short* src = (short*)s; bool* dst = (bool*)d; long i = BulkInt16ToBool(src, (byte*)dst, n); for (; i < n; i++) dst[i] = src[i] != 0; }
        private static unsafe void CastByteToBoolContig(void* s, void* d, long n) { byte* src = (byte*)s; bool* dst = (bool*)d; long i = BulkByteToBool(src, (byte*)dst, n); for (; i < n; i++) dst[i] = src[i] != 0; }
        private static unsafe void CastHalfToBoolContig(void* s, void* d, long n) { ushort* src = (ushort*)s; bool* dst = (bool*)d; Half* h = (Half*)s; long i = BulkHalfToBool(src, (byte*)dst, n); for (; i < n; i++) dst[i] = Converts.ToBoolean(h[i]); }

        // ---- strided via StridedNarrowDriver (dstSize = 1) ----
        private static unsafe long BulkSingleToBoolV(void* s, void* d, long n) => BulkSingleToBool((float*)s, (byte*)d, n);
        private static unsafe long BulkDoubleToBoolV(void* s, void* d, long n) => BulkDoubleToBool((double*)s, (byte*)d, n);
        private static unsafe long BulkInt32ToBoolV(void* s, void* d, long n) => BulkInt32ToBool((int*)s, (byte*)d, n);
        private static unsafe long BulkInt64ToBoolV(void* s, void* d, long n) => BulkInt64ToBool((long*)s, (byte*)d, n);
        private static unsafe long BulkInt16ToBoolV(void* s, void* d, long n) => BulkInt16ToBool((short*)s, (byte*)d, n);
        private static unsafe long BulkByteToBoolV(void* s, void* d, long n) => BulkByteToBool((byte*)s, (byte*)d, n);
        private static unsafe long BulkHalfToBoolV(void* s, void* d, long n) => BulkHalfToBool((ushort*)s, (byte*)d, n);

        private static unsafe void ConvSingleToBool(void* s, void* d) => *(bool*)d = Converts.ToBoolean(*(float*)s);
        private static unsafe void ConvDoubleToBool(void* s, void* d) => *(bool*)d = Converts.ToBoolean(*(double*)s);
        private static unsafe void ConvInt32ToBool(void* s, void* d) => *(bool*)d = *(int*)s != 0;
        private static unsafe void ConvInt64ToBool(void* s, void* d) => *(bool*)d = *(long*)s != 0;
        private static unsafe void ConvInt16ToBool(void* s, void* d) => *(bool*)d = *(short*)s != 0;
        private static unsafe void ConvByteToBool(void* s, void* d) => *(bool*)d = *(byte*)s != 0;
        private static unsafe void ConvHalfToBool(void* s, void* d) => *(bool*)d = Converts.ToBoolean(*(Half*)s);

        internal static unsafe StridedCastKernel TryGetToBoolStridedKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (dstType != NPTypeCode.Boolean || !Avx2.IsSupported) return null;
            switch (srcType)
            {
                case NPTypeCode.Single: return StridedSingleToBool;
                case NPTypeCode.Double: return StridedDoubleToBool;
                case NPTypeCode.Int32: case NPTypeCode.UInt32: return StridedInt32ToBool;
                case NPTypeCode.Int64: case NPTypeCode.UInt64: return StridedInt64ToBool;
                case NPTypeCode.Int16: case NPTypeCode.UInt16: case NPTypeCode.Char: return StridedInt16ToBool;
                case NPTypeCode.SByte: case NPTypeCode.Byte: return StridedByteToBool;
                case NPTypeCode.Half: return StridedHalfToBool;
            }
            return null;
        }

        private static unsafe void StridedSingleToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 1, &BulkSingleToBoolV, &ConvSingleToBool);
        private static unsafe void StridedDoubleToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 8, 1, &BulkDoubleToBoolV, &ConvDoubleToBool);
        private static unsafe void StridedInt32ToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 1, &BulkInt32ToBoolV, &ConvInt32ToBool);
        private static unsafe void StridedInt64ToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 8, 1, &BulkInt64ToBoolV, &ConvInt64ToBool);
        private static unsafe void StridedInt16ToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 1, &BulkInt16ToBoolV, &ConvInt16ToBool);
        private static unsafe void StridedByteToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 1, 1, &BulkByteToBoolV, &ConvByteToBool);
        private static unsafe void StridedHalfToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 1, &BulkHalfToBoolV, &ConvHalfToBool);
    }
}
