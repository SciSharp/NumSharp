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
                case NPTypeCode.Complex: return CastComplexToBoolContig;
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

        private static unsafe void ConvSingleToBool(void* s, void* d) => *(bool*)d = Converts.ToBoolean(*(float*)s);
        private static unsafe void ConvDoubleToBool(void* s, void* d) => *(bool*)d = Converts.ToBoolean(*(double*)s);
        private static unsafe void ConvInt32ToBool(void* s, void* d) => *(bool*)d = *(int*)s != 0;
        private static unsafe void ConvInt64ToBool(void* s, void* d) => *(bool*)d = *(long*)s != 0;
        private static unsafe void ConvInt16ToBool(void* s, void* d) => *(bool*)d = *(short*)s != 0;
        private static unsafe void ConvByteToBool(void* s, void* d) => *(bool*)d = *(byte*)s != 0;

        // =====================================================================
        // FUSED gather strided->bool — WHOLE ARRAY, idx hoisted (same model as the
        // float->narrow FusedGather* kernels: per-row function-pointer dispatch blocks
        // the JIT from pipelining gathers across rows). VPGATHERDD/VPGATHERQQ loads the
        // strided 4/8-byte lanes straight into the (v!=0) compare -> 0/1 -> Narrow pack
        // -> contig bool bytes. Reuses the exact BulkXToBool compare (OrderedEqual for
        // float: -0.0->False, NaN->True; vpcmpeq for int) so it's bit-exact with the
        // contig kernels and Converts.ToBoolean. Scalar drain mirrors the Conv* tail.
        // Chosen only for inner ss!=1 && inner ds==1 && AVX2 (see Strided*ToBool).
        // =====================================================================

        // f32 -> bool: 32-wide (4x VPGATHERDD, OrderedEqual !=0) + 8-wide mop-up + scalar tail.
        private static unsafe void FusedGatherSingleToBool(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            bool vec = ss >= int.MinValue / 8 && ss <= int.MaxValue / 8;
            int si = (int)ss; var idx = Vector256.Create(0, si, 2 * si, 3 * si, 4 * si, 5 * si, 6 * si, 7 * si);
            long g = 8L * ss; var z = Vector256<float>.Zero; var one = Vector256.Create(1);
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                int* p = (int*)(src + srcOff * 4); byte* dr = dst + dstOff; long i = 0;
                if (vec)
                {
                    for (; i + 32 <= innerN; i += 32)
                    {
                        var m0 = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p,         idx, 4).AsSingle(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt32(), one);
                        var m1 = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p + g,     idx, 4).AsSingle(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt32(), one);
                        var m2 = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p + 2 * g, idx, 4).AsSingle(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt32(), one);
                        var m3 = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p + 3 * g, idx, 4).AsSingle(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt32(), one);
                        var s0 = Vector256.Narrow(m0, m1); var s1 = Vector256.Narrow(m2, m3);
                        Vector256.Store(Vector256.Narrow(s0, s1).AsByte(), dr + i); p += 4 * g;
                    }
                    for (; i + 8 <= innerN; i += 8)
                    {
                        var m = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p, idx, 4).AsSingle(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt32(), one);
                        dr[i] = (byte)m.GetElement(0); dr[i + 1] = (byte)m.GetElement(1);
                        dr[i + 2] = (byte)m.GetElement(2); dr[i + 3] = (byte)m.GetElement(3);
                        dr[i + 4] = (byte)m.GetElement(4); dr[i + 5] = (byte)m.GetElement(5);
                        dr[i + 6] = (byte)m.GetElement(6); dr[i + 7] = (byte)m.GetElement(7);
                        p += g;
                    }
                }
                for (; i < innerN; i++) { dr[i] = Converts.ToBoolean(*(float*)p) ? (byte)1 : (byte)0; p += ss; }
                for (int ax = outer - 1; ax >= 0; ax--)
                {
                    coord[ax]++; srcOff += srcStrides[ax]; dstOff += dstStrides[ax];
                    if (coord[ax] < shape[ax]) break;
                    coord[ax] = 0; srcOff -= srcStrides[ax] * shape[ax]; dstOff -= dstStrides[ax] * shape[ax];
                }
            }
        }

        // f64 -> bool: 32-wide (8x VPGATHERQQ of 4, sequential Narrow pairing) + 4-wide mop-up + tail.
        private static unsafe void FusedGatherDoubleToBool(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            var idx = Vector256.Create(0L, ss, 2 * ss, 3 * ss);
            long g = 4L * ss; var z = Vector256<double>.Zero; var one = Vector256.Create(1L);
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                long* p = (long*)(src + srcOff * 8); byte* dr = dst + dstOff; long i = 0;
                for (; i + 32 <= innerN; i += 32)
                {
                    var g0 = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p,         idx, 8).AsDouble(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                    var g1 = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p + g,     idx, 8).AsDouble(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                    var g2 = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p + 2 * g, idx, 8).AsDouble(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                    var g3 = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p + 3 * g, idx, 8).AsDouble(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                    var g4 = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p + 4 * g, idx, 8).AsDouble(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                    var g5 = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p + 5 * g, idx, 8).AsDouble(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                    var g6 = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p + 6 * g, idx, 8).AsDouble(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                    var g7 = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p + 7 * g, idx, 8).AsDouble(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                    var a0 = Vector256.Narrow(g0, g1); var a1 = Vector256.Narrow(g2, g3);
                    var a2 = Vector256.Narrow(g4, g5); var a3 = Vector256.Narrow(g6, g7);
                    var s0 = Vector256.Narrow(a0, a1); var s1 = Vector256.Narrow(a2, a3);
                    Vector256.Store(Vector256.Narrow(s0, s1).AsByte(), dr + i); p += 8 * g;
                }
                for (; i + 4 <= innerN; i += 4)
                {
                    var m = Avx2.AndNot(Avx.Compare(Avx2.GatherVector256(p, idx, 8).AsDouble(), z, FloatComparisonMode.OrderedEqualNonSignaling).AsInt64(), one);
                    dr[i] = (byte)m.GetElement(0); dr[i + 1] = (byte)m.GetElement(1);
                    dr[i + 2] = (byte)m.GetElement(2); dr[i + 3] = (byte)m.GetElement(3);
                    p += g;
                }
                for (; i < innerN; i++) { dr[i] = Converts.ToBoolean(*(double*)p) ? (byte)1 : (byte)0; p += ss; }
                for (int ax = outer - 1; ax >= 0; ax--)
                {
                    coord[ax]++; srcOff += srcStrides[ax]; dstOff += dstStrides[ax];
                    if (coord[ax] < shape[ax]) break;
                    coord[ax] = 0; srcOff -= srcStrides[ax] * shape[ax]; dstOff -= dstStrides[ax] * shape[ax];
                }
            }
        }

        // i32/u32 -> bool: 32-wide (4x VPGATHERDD, vpcmpeqd !=0) + 8-wide mop-up + scalar tail.
        private static unsafe void FusedGatherInt32ToBool(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            bool vec = ss >= int.MinValue / 8 && ss <= int.MaxValue / 8;
            int si = (int)ss; var idx = Vector256.Create(0, si, 2 * si, 3 * si, 4 * si, 5 * si, 6 * si, 7 * si);
            long g = 8L * ss; var z = Vector256<int>.Zero; var one = Vector256.Create(1);
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                int* p = (int*)(src + srcOff * 4); byte* dr = dst + dstOff; long i = 0;
                if (vec)
                {
                    for (; i + 32 <= innerN; i += 32)
                    {
                        var m0 = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p,         idx, 4), z), one);
                        var m1 = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p + g,     idx, 4), z), one);
                        var m2 = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p + 2 * g, idx, 4), z), one);
                        var m3 = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p + 3 * g, idx, 4), z), one);
                        var s0 = Vector256.Narrow(m0, m1); var s1 = Vector256.Narrow(m2, m3);
                        Vector256.Store(Vector256.Narrow(s0, s1).AsByte(), dr + i); p += 4 * g;
                    }
                    for (; i + 8 <= innerN; i += 8)
                    {
                        var m = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p, idx, 4), z), one);
                        dr[i] = (byte)m.GetElement(0); dr[i + 1] = (byte)m.GetElement(1);
                        dr[i + 2] = (byte)m.GetElement(2); dr[i + 3] = (byte)m.GetElement(3);
                        dr[i + 4] = (byte)m.GetElement(4); dr[i + 5] = (byte)m.GetElement(5);
                        dr[i + 6] = (byte)m.GetElement(6); dr[i + 7] = (byte)m.GetElement(7);
                        p += g;
                    }
                }
                for (; i < innerN; i++) { dr[i] = *(int*)p != 0 ? (byte)1 : (byte)0; p += ss; }
                for (int ax = outer - 1; ax >= 0; ax--)
                {
                    coord[ax]++; srcOff += srcStrides[ax]; dstOff += dstStrides[ax];
                    if (coord[ax] < shape[ax]) break;
                    coord[ax] = 0; srcOff -= srcStrides[ax] * shape[ax]; dstOff -= dstStrides[ax] * shape[ax];
                }
            }
        }

        // i64/u64 -> bool: 32-wide (8x VPGATHERQQ of 4, sequential Narrow pairing) + 4-wide mop-up + tail.
        private static unsafe void FusedGatherInt64ToBool(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            var idx = Vector256.Create(0L, ss, 2 * ss, 3 * ss);
            long g = 4L * ss; var z = Vector256<long>.Zero; var one = Vector256.Create(1L);
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                long* p = (long*)(src + srcOff * 8); byte* dr = dst + dstOff; long i = 0;
                for (; i + 32 <= innerN; i += 32)
                {
                    var g0 = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p,         idx, 8), z), one);
                    var g1 = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p + g,     idx, 8), z), one);
                    var g2 = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p + 2 * g, idx, 8), z), one);
                    var g3 = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p + 3 * g, idx, 8), z), one);
                    var g4 = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p + 4 * g, idx, 8), z), one);
                    var g5 = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p + 5 * g, idx, 8), z), one);
                    var g6 = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p + 6 * g, idx, 8), z), one);
                    var g7 = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p + 7 * g, idx, 8), z), one);
                    var a0 = Vector256.Narrow(g0, g1); var a1 = Vector256.Narrow(g2, g3);
                    var a2 = Vector256.Narrow(g4, g5); var a3 = Vector256.Narrow(g6, g7);
                    var s0 = Vector256.Narrow(a0, a1); var s1 = Vector256.Narrow(a2, a3);
                    Vector256.Store(Vector256.Narrow(s0, s1).AsByte(), dr + i); p += 8 * g;
                }
                for (; i + 4 <= innerN; i += 4)
                {
                    var m = Avx2.AndNot(Avx2.CompareEqual(Avx2.GatherVector256(p, idx, 8), z), one);
                    dr[i] = (byte)m.GetElement(0); dr[i + 1] = (byte)m.GetElement(1);
                    dr[i + 2] = (byte)m.GetElement(2); dr[i + 3] = (byte)m.GetElement(3);
                    p += g;
                }
                for (; i < innerN; i++) { dr[i] = *(long*)p != 0 ? (byte)1 : (byte)0; p += ss; }
                for (int ax = outer - 1; ax >= 0; ax--)
                {
                    coord[ax]++; srcOff += srcStrides[ax]; dstOff += dstStrides[ax];
                    if (coord[ax] < shape[ax]) break;
                    coord[ax] = 0; srcOff -= srcStrides[ax] * shape[ax]; dstOff -= dstStrides[ax] * shape[ax];
                }
            }
        }

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
                case NPTypeCode.Complex: return StridedComplexToBool;
            }
            return null;
        }

        // Gatherable 4/8-byte sources (f32/f64/i32/u32/i64/u64): inner-strided + contig dst row + AVX2
        // -> whole-array fused gather (idx hoisted, rows pipelined). i16/u16/char/i8/u8/half (1/2-byte,
        // not gatherable) always take StridedNarrowDriver.
        private static unsafe void StridedSingleToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedGatherSingleToBool(s, d, ss, ds, sh, nd); else StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 1, &BulkSingleToBoolV, &ConvSingleToBool); }
        private static unsafe void StridedDoubleToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedGatherDoubleToBool(s, d, ss, ds, sh, nd); else StridedNarrowDriver(s, d, ss, ds, sh, nd, 8, 1, &BulkDoubleToBoolV, &ConvDoubleToBool); }
        private static unsafe void StridedInt32ToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedGatherInt32ToBool(s, d, ss, ds, sh, nd); else StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 1, &BulkInt32ToBoolV, &ConvInt32ToBool); }
        private static unsafe void StridedInt64ToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedGatherInt64ToBool(s, d, ss, ds, sh, nd); else StridedNarrowDriver(s, d, ss, ds, sh, nd, 8, 1, &BulkInt64ToBoolV, &ConvInt64ToBool); }
        private static unsafe void StridedInt16ToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 1, &BulkInt16ToBoolV, &ConvInt16ToBool);
        private static unsafe void StridedByteToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 1, 1, &BulkByteToBoolV, &ConvByteToBool);

        // f16 -> bool strided: f16 is 2-byte (not VPGATHER-eligible), so the generic
        // StridedNarrowDriver scalars the ss==2 (`[:, ::2]`) and ss==-1 (`[:, ::-1]`) inner rows
        // -> the f16|strided|bool 0.14x cliff. Apply the SubwordNarrow deinterleave (ss==2) /
        // reverse (ss==-1) shuffles, then NumPy's half-truthiness `(bits & 0x7fff) != 0` (so +-0.0
        // -> False, NaN/inf -> True). Inner ss==1 rows (sliced/negrow/bcast) run the contiguous
        // magnitude compare; arbitrary strides (F/T column-gather) and the tail keep the scalar
        // test. Inlined inner loops (a per-row helper costs ~6x — see the SubwordCopy PERF NOTE).
        private static unsafe void StridedHalfToBool(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            short* src = (short*)srcV;
            byte* dst = (byte*)dstV;
            if (ndim == 0) { dst[0] = (byte)((src[0] & 0x7fff) != 0 ? 1 : 0); return; }

            int outer = ndim - 1;
            long innerN = shape[outer], ss = srcStrides[outer], ds = dstStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;

            var one = _shortOne;
            var zero = Vector256<short>.Zero;
            var mag = Vector256.Create((short)0x7fff);
            var revw = _revWords;

            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                short* s = src + srcOff;
                byte* d = dst + dstOff;
                long i = 0;
                if (ds == 1)
                {
                    if (ss == 1)
                    {
                        for (; i + 32 <= innerN; i += 32)
                        {
                            var na = Avx2.AndNot(Avx2.CompareEqual(Avx2.And(Vector256.Load(s + i), mag), zero), one);
                            var nb = Avx2.AndNot(Avx2.CompareEqual(Avx2.And(Vector256.Load(s + i + 16), mag), zero), one);
                            Vector256.Store(Vector256.Narrow(na, nb).AsByte(), d + i);
                        }
                    }
                    else if (ss == 2)
                    {
                        for (; i + 16 <= innerN; i += 16)
                        {
                            var v0 = Vector256.Load((int*)(s + 2 * i));
                            var v1 = Vector256.Load((int*)(s + 2 * i + 16));
                            var evens = Avx2.Permute4x64(Avx2.PackUnsignedSaturate(Avx2.And(v0, _evenWordMask), Avx2.And(v1, _evenWordMask)).AsInt64(), 0xD8).AsInt16();
                            var nz = Avx2.AndNot(Avx2.CompareEqual(Avx2.And(evens, mag), zero), one);
                            Vector128.Store(Avx2.Permute4x64(Avx2.PackUnsignedSaturate(nz, nz).AsInt64(), 0xD8).GetLower().AsByte(), d + i);
                        }
                    }
                    else if (ss == -1)
                    {
                        for (; i + 16 <= innerN; i += 16)
                        {
                            var v = Vector256.Load(s - i - 15);
                            var rev = Avx2.Permute4x64(Avx2.Shuffle(v.AsByte(), revw).AsInt64(), 0x4E).AsInt16();
                            var nz = Avx2.AndNot(Avx2.CompareEqual(Avx2.And(rev, mag), zero), one);
                            Vector128.Store(Avx2.Permute4x64(Avx2.PackUnsignedSaturate(nz, nz).AsInt64(), 0xD8).GetLower().AsByte(), d + i);
                        }
                    }
                }
                for (; i < innerN; i++) d[i * ds] = (byte)((s[i * ss] & 0x7fff) != 0 ? 1 : 0);

                for (int ax = outer - 1; ax >= 0; ax--)
                {
                    coord[ax]++; srcOff += srcStrides[ax]; dstOff += dstStrides[ax];
                    if (coord[ax] < shape[ax]) break;
                    coord[ax] = 0; srcOff -= srcStrides[ax] * shape[ax]; dstOff -= dstStrides[ax] * shape[ax];
                }
            }
        }
    }
}
