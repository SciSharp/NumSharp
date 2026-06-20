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
                case NPTypeCode.SByte:  return CastComplexToSByteContig;
                case NPTypeCode.Byte:   return CastComplexToByteContig;
                case NPTypeCode.Int16:  return CastComplexToInt16Contig;
                case NPTypeCode.UInt16: return CastComplexToUInt16Contig;
                case NPTypeCode.Char:   return CastComplexToCharContig;
            }
            return null;
        }

        // Deinterleave the real parts of 4 consecutive complex (at complex index ci)
        // into a Vector128<int> via cvttpd2dq. Requires Avx2 (vpermq).
        private static unsafe Vector128<int> ComplexRealsToInt32(double* p, long ci)
        {
            var a = Avx.LoadVector256(p + 2 * ci);       // re0 im0 re1 im1
            var b = Avx.LoadVector256(p + 2 * ci + 4);   // re2 im2 re3 im3
            var reals = Avx2.Permute4x64(Avx.UnpackLow(a, b), 0xD8); // re0 re1 re2 re3
            return Avx.ConvertToVector128Int32WithTruncation(reals); // 4x i32
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
                case NPTypeCode.SByte:  return StridedComplexToSByte;
                case NPTypeCode.Byte:   return StridedComplexToByte;
                case NPTypeCode.Int16:  return StridedComplexToInt16;
                case NPTypeCode.UInt16: return StridedComplexToUInt16;
                case NPTypeCode.Char:   return StridedComplexToChar;
            }
            return null;
        }

        private static unsafe void StridedComplexToInt32(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedComplexDriver(s, d, ss, ds, sh, nd, 4, &BulkComplexToInt32V, &ConvComplexToInt32);
        private static unsafe void StridedComplexToSByte(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedComplexDriver(s, d, ss, ds, sh, nd, 1, &BulkComplexToByteV,  &ConvComplexToSByte);
        private static unsafe void StridedComplexToByte(void* s, void* d, long* ss, long* ds, long* sh, int nd)   => StridedComplexDriver(s, d, ss, ds, sh, nd, 1, &BulkComplexToByteV,  &ConvComplexToByte);
        private static unsafe void StridedComplexToInt16(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedComplexDriver(s, d, ss, ds, sh, nd, 2, &BulkComplexToShortV, &ConvComplexToInt16);
        private static unsafe void StridedComplexToUInt16(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedComplexDriver(s, d, ss, ds, sh, nd, 2, &BulkComplexToShortV, &ConvComplexToUInt16);
        private static unsafe void StridedComplexToChar(void* s, void* d, long* ss, long* ds, long* sh, int nd)   => StridedComplexDriver(s, d, ss, ds, sh, nd, 2, &BulkComplexToShortV, &ConvComplexToChar);

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
