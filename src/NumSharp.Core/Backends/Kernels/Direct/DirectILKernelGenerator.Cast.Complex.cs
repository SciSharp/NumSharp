using System;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.Complex.cs
        //   Complex128 -> int (i32/i8/u8/i16/u16/char) contiguous + strided casts.
        //
        //   Phase-0: c128->narrow geomean 0.40 (c128->i8 0.61x, c128->i32 0.90x) —
        //   complex casts fall to the scalar ConvertValue path. NumPy drops the
        //   imaginary part (ComplexWarning), takes the real (a double), then does
        //   float->int (cvtt). So the faithful SIMD chain is:
        //
        //       DEINTERLEAVE real parts:  Complex[N] is double[2N] = (re,im,re,im,..)
        //         a = [re0 im0 re1 im1], b = [re2 im2 re3 im3]
        //         vunpcklpd(a,b) = [re0 re2 re1 re3]; vpermq 0xD8 -> [re0 re1 re2 re3]
        //       cvttpd2dq -> 4x i32  (INT_MIN sentinel on NaN/overflow, matches NumPy)
        //       truncating Vector.Narrow i32->i16[->i8] for the narrow targets.
        //
        //   Bit-exact with Converts.To{X}(Complex) (which reads c.Real then the same
        //   ToInt32/narrow), hence with NumPy 2.4.2 (proven 0-diff, c128->i32 1.55x,
        //   c128->i8 1.38x; modest margin — 16-byte elements are memory-bound).
        //
        //   Plugged into TryGetCastKernel ahead of the ScalarOnly fallback. The
        //   scalar tail / non-contiguous-inner rows call Converts.*, so the result is
        //   identical with or without SIMD.
        // =====================================================================

        /// <summary>
        /// Returns the contiguous <see cref="CastKernel"/> for Complex -&gt;
        /// {i32,i8,u8,i16,u16,char}, or null. Bit-exact with <see cref="Converts"/>.
        /// </summary>
        internal static unsafe CastKernel TryGetComplexToIntKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (srcType != NPTypeCode.Complex) return null;
            switch (dstType)
            {
                case NPTypeCode.Int32:  return CastComplexToInt32Contig;
                case NPTypeCode.UInt32: return CastComplexToUInt32Contig;
                case NPTypeCode.UInt64: return CastComplexToUInt64Contig;
                case NPTypeCode.SByte:  return CastComplexToSByteContig;
                case NPTypeCode.Byte:   return CastComplexToByteContig;
                case NPTypeCode.Int16:  return CastComplexToInt16Contig;
                case NPTypeCode.UInt16: return CastComplexToUInt16Contig;
                case NPTypeCode.Char:   return CastComplexToCharContig;
            }
            return null;
        }

        // c128 -> u32: deinterleave reals (a double) then the AVX2 f64->u32 kernel. NumPy drops
        // the imaginary part (ComplexWarning); bit-exact with Converts.ToUInt32(Complex).
        private static unsafe long BulkComplexToUInt32(double* p, uint* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
                for (; i + 4 <= count; i += 4)
                    Vector128.Store(DoubleToU32x4(ComplexReals4(p, i)).AsUInt32(), dst + i);
            return i;
        }
        private static unsafe void CastComplexToUInt32Contig(void* s, void* d, long n)
        {
            double* p = (double*)s; uint* dst = (uint*)d; Complex* c = (Complex*)s;
            long i = BulkComplexToUInt32(p, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToUInt32(c[i]);
        }
        private static unsafe long BulkComplexToUInt32V(void* s, void* d, long n) => BulkComplexToUInt32((double*)s, (uint*)d, n);
        private static unsafe void ConvComplexToUInt32(void* s, void* d) => *(uint*)d = Converts.ToUInt32(*(Complex*)s);

        // c128 -> u64: deinterleave reals then the AVX2 f64->u64 kernel. Bit-exact with Converts.ToUInt64(Complex).
        private static unsafe long BulkComplexToUInt64(double* p, ulong* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
                for (; i + 4 <= count; i += 4)
                    Vector256.Store(DoubleToU64x4(ComplexReals4(p, i)).AsUInt64(), dst + i);
            return i;
        }
        private static unsafe void CastComplexToUInt64Contig(void* s, void* d, long n)
        {
            double* p = (double*)s; ulong* dst = (ulong*)d; Complex* c = (Complex*)s;
            long i = BulkComplexToUInt64(p, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToUInt64(c[i]);
        }
        private static unsafe long BulkComplexToUInt64V(void* s, void* d, long n) => BulkComplexToUInt64((double*)s, (ulong*)d, n);
        private static unsafe void ConvComplexToUInt64(void* s, void* d) => *(ulong*)d = Converts.ToUInt64(*(Complex*)s);

        // Deinterleave the real parts of 4 consecutive complex (at complex index ci)
        // into a Vector128<int> via cvttpd2dq. Requires Avx2 (vpermq).
        private static unsafe Vector128<int> ComplexRealsToInt32(double* p, long ci)
        {
            var a = Avx.LoadVector256(p + 2 * ci);       // re0 im0 re1 im1
            var b = Avx.LoadVector256(p + 2 * ci + 4);   // re2 im2 re3 im3
            var reals = Avx2.Permute4x64(Avx.UnpackLow(a, b), 0xD8); // re0 re1 re2 re3
            return Avx.ConvertToVector128Int32WithTruncation(reals); // 4x i32
        }

        // Deinterleave the real parts of 4 consecutive complex into a Vector256<double>.
        private static unsafe Vector256<double> ComplexReals4(double* p, long ci)
        {
            var a = Avx.LoadVector256(p + 2 * ci);       // re0 im0 re1 im1
            var b = Avx.LoadVector256(p + 2 * ci + 4);   // re2 im2 re3 im3
            return Avx2.Permute4x64(Avx.UnpackLow(a, b), 0xD8); // re0 re1 re2 re3
        }

        // 16 complex -> reals -> 16 f16 via the shared round-to-odd double->f16 (NaN -> scalar
        // fallback over the block). c128->f16 = realpart (NumPy ComplexWarning) then double->f16.
        private static unsafe long BulkComplexToHalf(double* p, ushort* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
                for (; i + 16 <= count; i += 16)
                {
                    var r0 = ComplexReals4(p, i); var r1 = ComplexReals4(p, i + 4);
                    var r2 = ComplexReals4(p, i + 8); var r3 = ComplexReals4(p, i + 12);
                    if (AnyNaN(r0) || AnyNaN(r1) || AnyNaN(r2) || AnyNaN(r3))
                    { for (int k = 0; k < 16; k++) dst[i + k] = DoubleToHalfBits(p[2 * (i + k)]); continue; }
                    var lo = FloatToHalfBits(Vector256.Create(RoundToOdd4(r0), RoundToOdd4(r1)));
                    var hi = FloatToHalfBits(Vector256.Create(RoundToOdd4(r2), RoundToOdd4(r3)));
                    Vector256.Store(Vector256.Narrow(lo, hi).AsUInt16(), dst + i);
                }
            return i;
        }
        private static unsafe long BulkComplexToHalfV(void* s, void* d, long n) => BulkComplexToHalf((double*)s, (ushort*)d, n);
        private static unsafe void ConvComplexToHalf(void* s, void* d) => *(ushort*)d = DoubleToHalfBits((*(Complex*)s).Real);

        private static unsafe void CastComplexToHalfContig(void* s, void* d, long n)
        {
            double* p = (double*)s; ushort* dst = (ushort*)d; Complex* c = (Complex*)s;
            long i = BulkComplexToHalf(p, dst, n);
            for (; i < n; i++) dst[i] = DoubleToHalfBits(c[i].Real);
        }
        private static unsafe void StridedComplexToHalf(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedComplexToHalf(s, d, ss, ds, sh, nd); else StridedComplexDriver(s, d, ss, ds, sh, nd, 2, &BulkComplexToHalfV, &ConvComplexToHalf); }

        // ---- Complex -> bool: nonzero(z) = (re != 0) | (im != 0). OrderedEqual against 0 treats
        // +-0 as zero and NaN as nonzero (NaN is unordered -> not-equal), matching NumPy. ----
        private static unsafe Vector256<long> ComplexNonzero4(double* p, long ci)
        {
            var a = Avx.LoadVector256(p + 2 * ci);       // re0 im0 re1 im1
            var b = Avx.LoadVector256(p + 2 * ci + 4);   // re2 im2 re3 im3
            var reals = Avx2.Permute4x64(Avx.UnpackLow(a, b), 0xD8);   // re0 re1 re2 re3
            var imags = Avx2.Permute4x64(Avx.UnpackHigh(a, b), 0xD8);  // im0 im1 im2 im3
            var er = Avx.Compare(reals, Vector256<double>.Zero, FloatComparisonMode.OrderedEqualNonSignaling);
            var ei = Avx.Compare(imags, Vector256<double>.Zero, FloatComparisonMode.OrderedEqualNonSignaling);
            return Avx2.AndNot(Avx.And(er, ei).AsInt64(), Vector256.Create(1L)); // ~(re==0 & im==0) & 1 -> 0/1
        }
        private static unsafe long BulkComplexToBool(double* p, byte* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
                for (; i + 4 <= count; i += 4)
                {
                    var nz = ComplexNonzero4(p, i);
                    dst[i] = (byte)nz.GetElement(0); dst[i + 1] = (byte)nz.GetElement(1);
                    dst[i + 2] = (byte)nz.GetElement(2); dst[i + 3] = (byte)nz.GetElement(3);
                }
            return i;
        }
        private static unsafe long BulkComplexToBoolV(void* s, void* d, long n) => BulkComplexToBool((double*)s, (byte*)d, n);
        private static unsafe void ConvComplexToBool(void* s, void* d)
        { var c = *(Complex*)s; *(byte*)d = (byte)((c.Real != 0.0 || c.Imaginary != 0.0) ? 1 : 0); }

        internal static unsafe void CastComplexToBoolContig(void* s, void* d, long n)
        {
            double* p = (double*)s; byte* dst = (byte*)d; Complex* c = (Complex*)s;
            long i = BulkComplexToBool(p, dst, n);
            for (; i < n; i++) { var z = c[i]; dst[i] = (byte)((z.Real != 0.0 || z.Imaginary != 0.0) ? 1 : 0); }
        }
        internal static unsafe void StridedComplexToBool(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedComplexToBool(s, d, ss, ds, sh, nd); else StridedComplexDriver(s, d, ss, ds, sh, nd, 1, &BulkComplexToBoolV, &ConvComplexToBool); }

        // Inner-strided: gather reals (idx) and imags (idx, base+1 double) per 4 logical complex.
        private static unsafe void FusedComplexToBool(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            long rs = 2L * ss; var idx = Vector256.Create(0L, rs, 2 * rs, 3 * rs);
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                double* p = (double*)(src + srcOff * 16); byte* dr = dst + dstOff; long i = 0;
                for (; i + 4 <= innerN; i += 4)
                {
                    var reals = Avx2.GatherVector256(p, idx, 8);
                    var imags = Avx2.GatherVector256(p + 1, idx, 8);
                    var er = Avx.Compare(reals, Vector256<double>.Zero, FloatComparisonMode.OrderedEqualNonSignaling);
                    var ei = Avx.Compare(imags, Vector256<double>.Zero, FloatComparisonMode.OrderedEqualNonSignaling);
                    var nz = Avx2.AndNot(Avx.And(er, ei).AsInt64(), Vector256.Create(1L));
                    dr[i] = (byte)nz.GetElement(0); dr[i + 1] = (byte)nz.GetElement(1);
                    dr[i + 2] = (byte)nz.GetElement(2); dr[i + 3] = (byte)nz.GetElement(3);
                    p += 4 * rs;
                }
                for (; i < innerN; i++) { var z = *(Complex*)(src + (srcOff + i * ss) * 16); dr[i] = (byte)((z.Real != 0.0 || z.Imaginary != 0.0) ? 1 : 0); }
                AdvanceOdometer(coord, srcStrides, dstStrides, shape, outer, ref srcOff, ref dstOff);
            }
        }

        // 8 strided complex -> 2x VPGATHERQQ-reals -> round-to-odd double->f16 -> 8 u16; scalar mop-up.
        // (NaN in a gather group -> the 8 elements take the scalar DoubleToHalfBits fallback.)
        private static unsafe void FusedComplexToHalf(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            long rs = 2L * ss; var idx = Vector256.Create(0L, rs, 2 * rs, 3 * rs);
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                double* p = (double*)(src + srcOff * 16); ushort* dr = (ushort*)(dst + dstOff * 2); long i = 0;
                for (; i + 8 <= innerN; i += 8)
                {
                    var r0 = Avx2.GatherVector256(p, idx, 8);
                    var r1 = Avx2.GatherVector256(p + 4 * rs, idx, 8);
                    if (AnyNaN(r0) || AnyNaN(r1))
                        for (int k = 0; k < 8; k++) dr[i + k] = DoubleToHalfBits(((Complex*)(src + (srcOff + (i + k) * ss) * 16))->Real);
                    else
                    {
                        var h = FloatToHalfBits(Vector256.Create(RoundToOdd4(r0), RoundToOdd4(r1)));
                        Vector128.Store(Vector256.Narrow(h, h).GetLower().AsUInt16(), dr + i);
                    }
                    p += 8 * rs;
                }
                for (; i < innerN; i++) dr[i] = DoubleToHalfBits(((Complex*)(src + (srcOff + i * ss) * 16))->Real);
                AdvanceOdometer(coord, srcStrides, dstStrides, shape, outer, ref srcOff, ref dstOff);
            }
        }

        // 4 complex -> 4 i32.
        private static unsafe long BulkComplexToInt32(double* p, int* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
                for (; i + 4 <= count; i += 4)
                    Vector128.Store(ComplexRealsToInt32(p, i), dst + i);
            return i;
        }

        // 8 complex -> 2x (4xi32) -> Narrow -> 8 i16.
        private static unsafe long BulkComplexToShort(double* p, short* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
                for (; i + 8 <= count; i += 8)
                {
                    var lo = ComplexRealsToInt32(p, i);
                    var hi = ComplexRealsToInt32(p, i + 4);
                    Vector128.Store(Vector128.Narrow(lo, hi), dst + i);   // 8x i16
                }
            return i;
        }

        // 16 complex -> 4x (4xi32) -> 2x Narrow(i32->i16) -> 1x Narrow(i16->i8) -> 16 i8.
        private static unsafe long BulkComplexToByte(double* p, byte* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
                for (; i + 16 <= count; i += 16)
                {
                    var i0 = ComplexRealsToInt32(p, i);
                    var i1 = ComplexRealsToInt32(p, i + 4);
                    var i2 = ComplexRealsToInt32(p, i + 8);
                    var i3 = ComplexRealsToInt32(p, i + 12);
                    var s0 = Vector128.Narrow(i0, i1);   // 8x i16
                    var s1 = Vector128.Narrow(i2, i3);   // 8x i16
                    Vector128.Store(Vector128.Narrow(s0, s1).AsByte(), dst + i);  // 16x i8
                }
            return i;
        }

        // Typed contiguous kernels: SIMD bulk + NumPy-faithful scalar tail (Converts.To{X}(Complex)).
        private static unsafe void CastComplexToInt32Contig(void* s, void* d, long n)
        {
            double* p = (double*)s; int* dst = (int*)d; Complex* c = (Complex*)s;
            long i = BulkComplexToInt32(p, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToInt32(c[i]);
        }
        private static unsafe void CastComplexToSByteContig(void* s, void* d, long n)
        {
            double* p = (double*)s; sbyte* dst = (sbyte*)d; Complex* c = (Complex*)s;
            long i = BulkComplexToByte(p, (byte*)dst, n);
            for (; i < n; i++) dst[i] = Converts.ToSByte(c[i]);
        }
        private static unsafe void CastComplexToByteContig(void* s, void* d, long n)
        {
            double* p = (double*)s; byte* dst = (byte*)d; Complex* c = (Complex*)s;
            long i = BulkComplexToByte(p, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToByte(c[i]);
        }
        private static unsafe void CastComplexToInt16Contig(void* s, void* d, long n)
        {
            double* p = (double*)s; short* dst = (short*)d; Complex* c = (Complex*)s;
            long i = BulkComplexToShort(p, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToInt16(c[i]);
        }
        private static unsafe void CastComplexToUInt16Contig(void* s, void* d, long n)
        {
            double* p = (double*)s; ushort* dst = (ushort*)d; Complex* c = (Complex*)s;
            long i = BulkComplexToShort(p, (short*)dst, n);
            for (; i < n; i++) dst[i] = Converts.ToUInt16(c[i]);
        }
        private static unsafe void CastComplexToCharContig(void* s, void* d, long n)
        {
            double* p = (double*)s; char* dst = (char*)d; Complex* c = (Complex*)s;
            long i = BulkComplexToShort(p, (short*)dst, n);
            for (; i < n; i++) dst[i] = Converts.ToChar(c[i]);
        }

        // =====================================================================
        // STRIDED complex -> int. Element is 16 bytes (2 doubles); the inner axis is
        // contiguous (ss==1: sliced / negrow) -> run the Bulk on the contiguous complex
        // row + scalar tail. Inner-strided / reversed-inner / F / T (ss!=1) -> scalar
        // conv per element (correct; complex strided is rare). dst is fresh-contig.
        // Bit-exact: Bulk and conv share Converts.*.
        // =====================================================================

        private static unsafe void ConvComplexToInt32(void* s, void* d)  => *(int*)d    = Converts.ToInt32(*(Complex*)s);
        private static unsafe void ConvComplexToSByte(void* s, void* d)  => *(sbyte*)d  = Converts.ToSByte(*(Complex*)s);
        private static unsafe void ConvComplexToByte(void* s, void* d)   => *(byte*)d   = Converts.ToByte(*(Complex*)s);
        private static unsafe void ConvComplexToInt16(void* s, void* d)  => *(short*)d  = Converts.ToInt16(*(Complex*)s);
        private static unsafe void ConvComplexToUInt16(void* s, void* d) => *(ushort*)d = Converts.ToUInt16(*(Complex*)s);
        private static unsafe void ConvComplexToChar(void* s, void* d)   => *(char*)d   = Converts.ToChar(*(Complex*)s);

        private static unsafe long BulkComplexToInt32V(void* s, void* d, long n) => BulkComplexToInt32((double*)s, (int*)d, n);
        private static unsafe long BulkComplexToShortV(void* s, void* d, long n) => BulkComplexToShort((double*)s, (short*)d, n);
        private static unsafe long BulkComplexToByteV(void* s, void* d, long n)  => BulkComplexToByte((double*)s, (byte*)d, n);

        internal static unsafe StridedCastKernel TryGetComplexToIntStridedKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (srcType != NPTypeCode.Complex) return null;
            switch (dstType)
            {
                case NPTypeCode.Int32:  return StridedComplexToInt32;
                case NPTypeCode.UInt32: return StridedComplexToUInt32;
                case NPTypeCode.UInt64: return StridedComplexToUInt64;
                case NPTypeCode.SByte:  return StridedComplexToSByte;
                case NPTypeCode.Byte:   return StridedComplexToByte;
                case NPTypeCode.Int16:  return StridedComplexToInt16;
                case NPTypeCode.UInt16: return StridedComplexToUInt16;
                case NPTypeCode.Char:   return StridedComplexToChar;
            }
            return null;
        }

        // Inner-strided ([:, ::2]) / reversed-inner ([:, ::-1]) complex rows: the real part of the
        // strided complex[i] lives at double-offset i*(2*ss), so VPGATHERQQ over the reals (double
        // stride 2*ss) feeds cvttpd2dq directly — killing the 0.17-0.24x scalar cliff (whole-array,
        // idx hoisted; proven c128->i8 strided 1.12x). ss==1 (contig inner) keeps the UnpackLow Bulk
        // via StridedComplexDriver; ds!=1 / no-AVX2 stay scalar there too.
        private static unsafe void StridedComplexToInt32(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedComplexToInt32(s, d, ss, ds, sh, nd); else StridedComplexDriver(s, d, ss, ds, sh, nd, 4, &BulkComplexToInt32V, &ConvComplexToInt32); }
        private static unsafe void StridedComplexToUInt32(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedComplexToUInt32(s, d, ss, ds, sh, nd); else StridedComplexDriver(s, d, ss, ds, sh, nd, 4, &BulkComplexToUInt32V, &ConvComplexToUInt32); }
        private static unsafe void StridedComplexToUInt64(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedComplexToUInt64(s, d, ss, ds, sh, nd); else StridedComplexDriver(s, d, ss, ds, sh, nd, 8, &BulkComplexToUInt64V, &ConvComplexToUInt64); }
        private static unsafe void StridedComplexToSByte(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedComplexToByte(s, d, ss, ds, sh, nd, false); else StridedComplexDriver(s, d, ss, ds, sh, nd, 1, &BulkComplexToByteV, &ConvComplexToSByte); }
        private static unsafe void StridedComplexToByte(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedComplexToByte(s, d, ss, ds, sh, nd, true); else StridedComplexDriver(s, d, ss, ds, sh, nd, 1, &BulkComplexToByteV, &ConvComplexToByte); }
        private static unsafe void StridedComplexToInt16(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedComplexToShort(s, d, ss, ds, sh, nd, 0); else StridedComplexDriver(s, d, ss, ds, sh, nd, 2, &BulkComplexToShortV, &ConvComplexToInt16); }
        private static unsafe void StridedComplexToUInt16(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedComplexToShort(s, d, ss, ds, sh, nd, 1); else StridedComplexDriver(s, d, ss, ds, sh, nd, 2, &BulkComplexToShortV, &ConvComplexToUInt16); }
        private static unsafe void StridedComplexToChar(void* s, void* d, long* ss, long* ds, long* sh, int nd)
        { if (UseFusedGather(ss, ds, nd)) FusedComplexToShort(s, d, ss, ds, sh, nd, 2); else StridedComplexDriver(s, d, ss, ds, sh, nd, 2, &BulkComplexToShortV, &ConvComplexToChar); }

        // Whole-array gather-real deinterleave for inner-strided complex->int. p points at re0 of the
        // row; idx steps by rs = 2*ss doubles so VPGATHERQQ pulls the reals of logical complex 0..3.
        private static unsafe void FusedComplexToInt32(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            long rs = 2L * ss; var idx = Vector256.Create(0L, rs, 2 * rs, 3 * rs);
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                double* p = (double*)(src + srcOff * 16); int* dr = (int*)(dst + dstOff * 4); long i = 0;
                for (; i + 4 <= innerN; i += 4)
                {
                    Vector128.Store(Avx.ConvertToVector128Int32WithTruncation(Avx2.GatherVector256(p, idx, 8)), dr + i);
                    p += 4 * rs;
                }
                for (; i < innerN; i++) dr[i] = Converts.ToInt32(*(System.Numerics.Complex*)(src + (srcOff + i * ss) * 16));
                AdvanceOdometer(coord, srcStrides, dstStrides, shape, outer, ref srcOff, ref dstOff);
            }
        }

        // Same as FusedComplexToInt32 but feeds the gathered reals through the AVX2 f64->u32
        // kernel (DoubleToU32x4) instead of plain cvttpd2dq, so the u32 modular-wrap is faithful.
        private static unsafe void FusedComplexToUInt32(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            long rs = 2L * ss; var idx = Vector256.Create(0L, rs, 2 * rs, 3 * rs);
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                double* p = (double*)(src + srcOff * 16); uint* dr = (uint*)(dst + dstOff * 4); long i = 0;
                for (; i + 4 <= innerN; i += 4)
                {
                    Vector128.Store(DoubleToU32x4(Avx2.GatherVector256(p, idx, 8)).AsUInt32(), dr + i);
                    p += 4 * rs;
                }
                for (; i < innerN; i++) dr[i] = Converts.ToUInt32(*(System.Numerics.Complex*)(src + (srcOff + i * ss) * 16));
                AdvanceOdometer(coord, srcStrides, dstStrides, shape, outer, ref srcOff, ref dstOff);
            }
        }

        // Gather reals then the AVX2 f64->u64 kernel (DoubleToU64x4) for inner-strided c128->u64.
        private static unsafe void FusedComplexToUInt64(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            long rs = 2L * ss; var idx = Vector256.Create(0L, rs, 2 * rs, 3 * rs);
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                double* p = (double*)(src + srcOff * 16); ulong* dr = (ulong*)(dst + dstOff * 8); long i = 0;
                if (ss == -1)
                {
                    // negcol ([:, ::-1]): the 8 doubles around p are contiguous-reversed
                    // (re3,im3,re2,im2,re1,im1,re0,im0). Two loads + UnpackLow + VPERMQ extract the
                    // 4 reals in logical order, beating a -2-stride real gather. Reads stay in the row.
                    for (; i + 4 <= innerN; i += 4)
                    {
                        var reals = Avx2.Permute4x64(Avx.UnpackLow(Vector256.Load(p - 2), Vector256.Load(p - 6)), 0x72);
                        Vector256.Store(DoubleToU64x4(reals).AsUInt64(), dr + i);
                        p -= 8;
                    }
                }
                else
                    for (; i + 4 <= innerN; i += 4)
                    {
                        Vector256.Store(DoubleToU64x4(Avx2.GatherVector256(p, idx, 8)).AsUInt64(), dr + i);
                        p += 4 * rs;
                    }
                for (; i < innerN; i++) dr[i] = Converts.ToUInt64(*(System.Numerics.Complex*)(src + (srcOff + i * ss) * 16));
                AdvanceOdometer(coord, srcStrides, dstStrides, shape, outer, ref srcOff, ref dstOff);
            }
        }

        // 16 strided complex -> 4x VPGATHERQQ-reals+cvttpd (4xi32) -> 2-level Narrow -> 16 i8; 4-wide mop-up.
        private static unsafe void FusedComplexToByte(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim, bool unsignedDst)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            long rs = 2L * ss; var idx = Vector256.Create(0L, rs, 2 * rs, 3 * rs);
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                double* p = (double*)(src + srcOff * 16); byte* dr = dst + dstOff; long i = 0;
                for (; i + 16 <= innerN; i += 16)
                {
                    var a = Avx.ConvertToVector128Int32WithTruncation(Avx2.GatherVector256(p, idx, 8));
                    var b = Avx.ConvertToVector128Int32WithTruncation(Avx2.GatherVector256(p + 4 * rs, idx, 8));
                    var c = Avx.ConvertToVector128Int32WithTruncation(Avx2.GatherVector256(p + 8 * rs, idx, 8));
                    var e = Avx.ConvertToVector128Int32WithTruncation(Avx2.GatherVector256(p + 12 * rs, idx, 8));
                    var s0 = Vector128.Narrow(a, b); var s1 = Vector128.Narrow(c, e);
                    Vector128.Store(Vector128.Narrow(s0, s1).AsByte(), dr + i);
                    p += 16 * rs;
                }
                for (; i + 4 <= innerN; i += 4)
                {
                    var v = Avx.ConvertToVector128Int32WithTruncation(Avx2.GatherVector256(p, idx, 8));
                    dr[i] = (byte)v.GetElement(0); dr[i + 1] = (byte)v.GetElement(1);
                    dr[i + 2] = (byte)v.GetElement(2); dr[i + 3] = (byte)v.GetElement(3);
                    p += 4 * rs;
                }
                for (; i < innerN; i++) { var z = *(System.Numerics.Complex*)(src + (srcOff + i * ss) * 16); dr[i] = unsignedDst ? Converts.ToByte(z) : (byte)Converts.ToSByte(z); }
                AdvanceOdometer(coord, srcStrides, dstStrides, shape, outer, ref srcOff, ref dstOff);
            }
        }

        // 8 strided complex -> 2x VPGATHERQQ-reals+cvttpd (4xi32) -> 1x Narrow -> 8 i16; 4-wide mop-up.
        private static unsafe void FusedComplexToShort(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim, int kind)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            long rs = 2L * ss; var idx = Vector256.Create(0L, rs, 2 * rs, 3 * rs);
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                double* p = (double*)(src + srcOff * 16); short* dr = (short*)(dst + dstOff * 2); long i = 0;
                for (; i + 8 <= innerN; i += 8)
                {
                    var a = Avx.ConvertToVector128Int32WithTruncation(Avx2.GatherVector256(p, idx, 8));
                    var b = Avx.ConvertToVector128Int32WithTruncation(Avx2.GatherVector256(p + 4 * rs, idx, 8));
                    Vector128.Store(Vector128.Narrow(a, b), dr + i);
                    p += 8 * rs;
                }
                for (; i + 4 <= innerN; i += 4)
                {
                    var v = Avx.ConvertToVector128Int32WithTruncation(Avx2.GatherVector256(p, idx, 8));
                    dr[i] = (short)v.GetElement(0); dr[i + 1] = (short)v.GetElement(1);
                    dr[i + 2] = (short)v.GetElement(2); dr[i + 3] = (short)v.GetElement(3);
                    p += 4 * rs;
                }
                for (; i < innerN; i++)
                {
                    var z = *(System.Numerics.Complex*)(src + (srcOff + i * ss) * 16);
                    dr[i] = kind == 0 ? Converts.ToInt16(z) : kind == 1 ? (short)Converts.ToUInt16(z) : (short)Converts.ToChar(z);
                }
                AdvanceOdometer(coord, srcStrides, dstStrides, shape, outer, ref srcOff, ref dstOff);
            }
        }

        private static unsafe void StridedComplexDriver(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim,
            int dstSize,
            delegate*<void*, void*, long, long> bulk,
            delegate*<void*, void*, void> conv)
        {
            const int CS = 16; // sizeof(Complex)
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            if (ndim == 0) { conv(src, dst); return; }

            int outer = ndim - 1;
            long innerN = shape[outer];
            long ss = srcStrides[outer], ds = dstStrides[outer];
            long outerCount = 1;
            for (int a = 0; a < outer; a++) outerCount *= shape[a];

            long* coord = stackalloc long[ndim];
            for (int a = 0; a < ndim; a++) coord[a] = 0;

            long srcOff = 0, dstOff = 0; // in elements
            for (long o = 0; o < outerCount; o++)
            {
                byte* sRow = src + srcOff * CS;
                byte* dRow = dst + dstOff * dstSize;

                if (ss == 1 && ds == 1)
                {
                    long i = bulk(sRow, dRow, innerN);
                    for (; i < innerN; i++) conv(sRow + i * CS, dRow + i * dstSize);
                }
                else
                {
                    for (long i = 0; i < innerN; i++)
                        conv(sRow + i * ss * CS, dRow + i * ds * dstSize);
                }

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
