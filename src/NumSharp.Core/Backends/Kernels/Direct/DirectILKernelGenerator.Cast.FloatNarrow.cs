using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.FloatNarrow.cs
        //   Float/double -> NARROW int (i8/u8/i16/u16/char) contiguous + strided casts.
        //
        //   These are the worst cells in the Phase-0 cast matrix: f32->i8 bottoms
        //   it at 0.09 (10.8x slower than NumPy), because ResolveStrategy gives
        //   every srcFloat->dstInt pair EXCEPT Single->Int32 the ScalarOnly
        //   strategy (a per-element EmitConvertTo loop). NumPy does NOT vectorize
        //   these either — its generic cast loop is scalar-speed — so a SIMD
        //   cvtt + truncating-narrow kernel beats NumPy outright (proven: f32->i8
        //   3.4-4x, f64->i16 1.7-2x, 0-diff vs NumPy 2.4.2).
        //
        //   MECHANISM (bit-exact with the Converts reference, hence with NumPy):
        //     Converts.ToSByte(double) == unchecked((sbyte)ToInt32(value)), and
        //     ToInt32(double) is the hardware truncating convert with the
        //     int.MinValue NaN/overflow sentinel. float overloads route through
        //     (double)value first. So the faithful chain is:
        //
        //         cvtt(float|double) -> i32   (VCVTTPS2DQ / VCVTTPD2DQ: trunc-to-
        //                                      zero, INT_MIN on NaN/overflow)
        //         truncating Vector.Narrow i32->i16[->i8]   (low-bits extract ==
        //                                      unchecked((short)/(sbyte)int))
        //
        //     For f32, cvttps2dq(f) == ToInt32((double)f) for ALL inputs: (double)f
        //     is exact, both truncate identically in-range, both give INT_MIN out of
        //     int32 range / on NaN. Vector.Narrow is TRUNCATING (proven: 0x80000000
        //     -> 0, 0x12340041 -> 0x0041), NOT the saturating vpackssdw — so it
        //     matches unchecked((narrow)int) exactly, including WRAP (NumPy oracle:
        //     f32->i8 128.5 -> -128, 256 -> 0, 300 -> 44, NaN -> 0).
        //
        //     Signedness of the dst does NOT change the body (truncating narrow
        //     produces identical low bits for i8/u8 and i16/u16/char); only the
        //     scalar-tail Converts call and the store width differ.
        //
        //   Plugged into TryGetCastKernel ahead of the ScalarOnly fallback. The
        //   scalar tail (and the whole body when AVX2 is absent) calls Converts.*,
        //   so the result is identical with or without SIMD — no new fallback.
        // =====================================================================

        /// <summary>
        /// Returns the NumPy-faithful contiguous <see cref="CastKernel"/> for
        /// float|double -&gt; {i8,u8,i16,u16,char}, or null if the pair isn't a
        /// float-&gt;narrow-int cast. Bit-exact with <see cref="Converts"/>.
        /// </summary>
        internal static unsafe CastKernel TryGetFloatToNarrowIntKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (srcType == NPTypeCode.Single)
            {
                switch (dstType)
                {
                    case NPTypeCode.SByte:  return CastSingleToSByteContig;
                    case NPTypeCode.Byte:   return CastSingleToByteContig;
                    case NPTypeCode.Int16:  return CastSingleToInt16Contig;
                    case NPTypeCode.UInt16: return CastSingleToUInt16Contig;
                    case NPTypeCode.Char:   return CastSingleToCharContig;
                }
            }
            else if (srcType == NPTypeCode.Double)
            {
                switch (dstType)
                {
                    case NPTypeCode.SByte:  return CastDoubleToSByteContig;
                    case NPTypeCode.Byte:   return CastDoubleToByteContig;
                    case NPTypeCode.Int16:  return CastDoubleToInt16Contig;
                    case NPTypeCode.UInt16: return CastDoubleToUInt16Contig;
                    case NPTypeCode.Char:   return CastDoubleToCharContig;
                }
            }
            return null;
        }

        // -----------------------------------------------------------------
        // SIMD bulk loops — return the number of elements consumed (a multiple
        // of the vector step). Each writes the truncating-narrow low bytes/words;
        // the typed wrappers below add the per-dtype scalar tail.
        // -----------------------------------------------------------------

        // 32 floats -> 4x VCVTTPS2DQ (8xi32) -> 2x Narrow(i32->i16) -> 1x Narrow(i16->i8) -> 32 bytes.
        private static unsafe long BulkSingleToByte(float* src, byte* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                for (; i + 32 <= count; i += 32)
                {
                    var a = Avx.ConvertToVector256Int32WithTruncation(Vector256.Load(src + i));
                    var b = Avx.ConvertToVector256Int32WithTruncation(Vector256.Load(src + i + 8));
                    var c = Avx.ConvertToVector256Int32WithTruncation(Vector256.Load(src + i + 16));
                    var d = Avx.ConvertToVector256Int32WithTruncation(Vector256.Load(src + i + 24));
                    var s0 = Vector256.Narrow(a, b);                 // 16x i16
                    var s1 = Vector256.Narrow(c, d);                 // 16x i16
                    Vector256.Store(Vector256.Narrow(s0, s1).AsByte(), dst + i);   // 32x i8
                }
            }
            else if (Sse2.IsSupported)
            {
                for (; i + 16 <= count; i += 16)
                {
                    var a = Sse2.ConvertToVector128Int32WithTruncation(Vector128.Load(src + i));
                    var b = Sse2.ConvertToVector128Int32WithTruncation(Vector128.Load(src + i + 4));
                    var c = Sse2.ConvertToVector128Int32WithTruncation(Vector128.Load(src + i + 8));
                    var d = Sse2.ConvertToVector128Int32WithTruncation(Vector128.Load(src + i + 12));
                    var s0 = Vector128.Narrow(a, b);                 // 8x i16
                    var s1 = Vector128.Narrow(c, d);                 // 8x i16
                    Vector128.Store(Vector128.Narrow(s0, s1).AsByte(), dst + i);   // 16x i8
                }
            }
            return i;
        }

        // 16 floats -> 2x VCVTTPS2DQ (8xi32) -> 1x Narrow(i32->i16) -> 16 shorts.
        private static unsafe long BulkSingleToShort(float* src, short* dst, long count)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                for (; i + 16 <= count; i += 16)
                {
                    var a = Avx.ConvertToVector256Int32WithTruncation(Vector256.Load(src + i));
                    var b = Avx.ConvertToVector256Int32WithTruncation(Vector256.Load(src + i + 8));
                    Vector256.Store(Vector256.Narrow(a, b), dst + i);              // 16x i16
                }
            }
            else if (Sse2.IsSupported)
            {
                for (; i + 8 <= count; i += 8)
                {
                    var a = Sse2.ConvertToVector128Int32WithTruncation(Vector128.Load(src + i));
                    var b = Sse2.ConvertToVector128Int32WithTruncation(Vector128.Load(src + i + 4));
                    Vector128.Store(Vector128.Narrow(a, b), dst + i);              // 8x i16
                }
            }
            return i;
        }

        // 16 doubles -> 4x VCVTTPD2DQ (ymm->xmm, 4xi32) -> 2x Narrow(i32->i16) -> 1x Narrow(i16->i8)
        // -> 16 bytes. Stays in Vector128 lanes — cvttpd2dq yields a 4-wide Vector128<int> directly,
        // so no Vector256.Create (vinsertf128) combine is needed (that stalled the memory pipeline).
        private static unsafe long BulkDoubleToByte(double* src, byte* dst, long count)
        {
            long i = 0;
            if (Avx.IsSupported)
            {
                for (; i + 16 <= count; i += 16)
                {
                    var a = Avx.ConvertToVector128Int32WithTruncation(Vector256.Load(src + i));
                    var b = Avx.ConvertToVector128Int32WithTruncation(Vector256.Load(src + i + 4));
                    var c = Avx.ConvertToVector128Int32WithTruncation(Vector256.Load(src + i + 8));
                    var d = Avx.ConvertToVector128Int32WithTruncation(Vector256.Load(src + i + 12));
                    var s0 = Vector128.Narrow(a, b);                 // 8x i16
                    var s1 = Vector128.Narrow(c, d);                 // 8x i16
                    Vector128.Store(Vector128.Narrow(s0, s1).AsByte(), dst + i);   // 16x i8
                }
            }
            return i;
        }

        // 8 doubles -> 2x VCVTTPD2DQ (4xi32) -> 1x Narrow(i32->i16) -> 8 shorts. Vector128 lanes only.
        private static unsafe long BulkDoubleToShort(double* src, short* dst, long count)
        {
            long i = 0;
            if (Avx.IsSupported)
            {
                for (; i + 8 <= count; i += 8)
                {
                    var a = Avx.ConvertToVector128Int32WithTruncation(Vector256.Load(src + i));
                    var b = Avx.ConvertToVector128Int32WithTruncation(Vector256.Load(src + i + 4));
                    Vector128.Store(Vector128.Narrow(a, b), dst + i);              // 8x i16
                }
            }
            return i;
        }

        // -----------------------------------------------------------------
        // Typed contiguous kernels: SIMD bulk + NumPy-faithful scalar tail.
        // i8/u8 share BulkSingleToByte/BulkDoubleToByte; i16/u16/char share the
        // short bulk — truncating narrow gives identical low bits, only the tail
        // Converts call (and the matched store width) differs.
        // -----------------------------------------------------------------

        private static unsafe void CastSingleToSByteContig(void* s, void* d, long n)
        {
            float* src = (float*)s; sbyte* dst = (sbyte*)d;
            long i = BulkSingleToByte(src, (byte*)dst, n);
            for (; i < n; i++) dst[i] = Converts.ToSByte(src[i]);
        }

        private static unsafe void CastSingleToByteContig(void* s, void* d, long n)
        {
            float* src = (float*)s; byte* dst = (byte*)d;
            long i = BulkSingleToByte(src, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToByte(src[i]);
        }

        private static unsafe void CastSingleToInt16Contig(void* s, void* d, long n)
        {
            float* src = (float*)s; short* dst = (short*)d;
            long i = BulkSingleToShort(src, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToInt16(src[i]);
        }

        private static unsafe void CastSingleToUInt16Contig(void* s, void* d, long n)
        {
            float* src = (float*)s; ushort* dst = (ushort*)d;
            long i = BulkSingleToShort(src, (short*)dst, n);
            for (; i < n; i++) dst[i] = Converts.ToUInt16(src[i]);
        }

        private static unsafe void CastSingleToCharContig(void* s, void* d, long n)
        {
            float* src = (float*)s; char* dst = (char*)d;
            long i = BulkSingleToShort(src, (short*)dst, n);
            for (; i < n; i++) dst[i] = Converts.ToChar(src[i]);
        }

        private static unsafe void CastDoubleToSByteContig(void* s, void* d, long n)
        {
            double* src = (double*)s; sbyte* dst = (sbyte*)d;
            long i = BulkDoubleToByte(src, (byte*)dst, n);
            for (; i < n; i++) dst[i] = Converts.ToSByte(src[i]);
        }

        private static unsafe void CastDoubleToByteContig(void* s, void* d, long n)
        {
            double* src = (double*)s; byte* dst = (byte*)d;
            long i = BulkDoubleToByte(src, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToByte(src[i]);
        }

        private static unsafe void CastDoubleToInt16Contig(void* s, void* d, long n)
        {
            double* src = (double*)s; short* dst = (short*)d;
            long i = BulkDoubleToShort(src, dst, n);
            for (; i < n; i++) dst[i] = Converts.ToInt16(src[i]);
        }

        private static unsafe void CastDoubleToUInt16Contig(void* s, void* d, long n)
        {
            double* src = (double*)s; ushort* dst = (ushort*)d;
            long i = BulkDoubleToShort(src, (short*)dst, n);
            for (; i < n; i++) dst[i] = Converts.ToUInt16(src[i]);
        }

        private static unsafe void CastDoubleToCharContig(void* s, void* d, long n)
        {
            double* src = (double*)s; char* dst = (char*)d;
            long i = BulkDoubleToShort(src, (short*)dst, n);
            for (; i < n; i++) dst[i] = Converts.ToChar(src[i]);
        }

        // =====================================================================
        // STRIDED float -> narrow-int. Same Phase-0 cliff in the sliced/negrow/
        // negcol/strided layouts (f32->i8 strided was 0.09x): the generic strided
        // emitter gives float->narrow the ScalarOnly inner loop (and excludes char
        // entirely), so every non-contiguous source fell to the IL scalar.
        //
        //   ss==1 (inner-contiguous: sliced, negrow): run the contig Bulk directly
        //          on each inner row + scalar tail.
        //   ss!=1 (inner-strided: [:, ::2]; reversed: [:, ::-1]): STAGE the strided
        //          source row into a contiguous chunk buffer with a tight typed
        //          strided load, then run the vectorized Bulk on the buffer. The
        //          per-element COST was the heavy conversion (float->double->ToInt32
        //          ->narrow), not the load — vectorizing the conversion is the win;
        //          the strided load is a single typed move per element.
        //   ds!=1 (cast output is fresh-contig, so this is rare): fully scalar.
        //
        // dst is always contiguous (a cast allocates a fresh C/F-contig array), so
        // ds==1 on the inner axis in every real astype path. Outer dims walk with the
        // proven incremental-coord odometer (element strides). Bit-exact: the Bulk and
        // the scalar tail/conv share Converts.* — identical to the contiguous kernels.
        // =====================================================================

        // Contig-bulk void* trampolines (only called when the inner row is contiguous).
        private static unsafe long BulkSingleToByteV(void* s, void* d, long n)  => BulkSingleToByte((float*)s, (byte*)d, n);
        private static unsafe long BulkSingleToShortV(void* s, void* d, long n) => BulkSingleToShort((float*)s, (short*)d, n);
        private static unsafe long BulkDoubleToByteV(void* s, void* d, long n)  => BulkDoubleToByte((double*)s, (byte*)d, n);
        private static unsafe long BulkDoubleToShortV(void* s, void* d, long n) => BulkDoubleToShort((double*)s, (short*)d, n);

        // Scalar element converts — bit-exact NumPy-faithful Converts.* (used for the
        // SIMD tail and the rare fully-strided dst).
        private static unsafe void ConvSingleToSByte(void* s, void* d)  => *(sbyte*)d  = Converts.ToSByte(*(float*)s);
        private static unsafe void ConvSingleToByte(void* s, void* d)   => *(byte*)d   = Converts.ToByte(*(float*)s);
        private static unsafe void ConvSingleToInt16(void* s, void* d)  => *(short*)d  = Converts.ToInt16(*(float*)s);
        private static unsafe void ConvSingleToUInt16(void* s, void* d) => *(ushort*)d = Converts.ToUInt16(*(float*)s);
        private static unsafe void ConvSingleToChar(void* s, void* d)   => *(char*)d   = Converts.ToChar(*(float*)s);
        private static unsafe void ConvDoubleToSByte(void* s, void* d)  => *(sbyte*)d  = Converts.ToSByte(*(double*)s);
        private static unsafe void ConvDoubleToByte(void* s, void* d)   => *(byte*)d   = Converts.ToByte(*(double*)s);
        private static unsafe void ConvDoubleToInt16(void* s, void* d)  => *(short*)d  = Converts.ToInt16(*(double*)s);
        private static unsafe void ConvDoubleToUInt16(void* s, void* d) => *(ushort*)d = Converts.ToUInt16(*(double*)s);
        private static unsafe void ConvDoubleToChar(void* s, void* d)   => *(char*)d   = Converts.ToChar(*(double*)s);

        /// <summary>
        /// Returns the strided <see cref="StridedCastKernel"/> for float|double -&gt;
        /// {i8,u8,i16,u16,char}, or null. Bit-exact with the contiguous kernels.
        /// </summary>
        internal static unsafe StridedCastKernel TryGetFloatToNarrowIntStridedKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (srcType == NPTypeCode.Single)
            {
                switch (dstType)
                {
                    case NPTypeCode.SByte:  return StridedSingleToSByte;
                    case NPTypeCode.Byte:   return StridedSingleToByte;
                    case NPTypeCode.Int16:  return StridedSingleToInt16;
                    case NPTypeCode.UInt16: return StridedSingleToUInt16;
                    case NPTypeCode.Char:   return StridedSingleToChar;
                }
            }
            else if (srcType == NPTypeCode.Double)
            {
                switch (dstType)
                {
                    case NPTypeCode.SByte:  return StridedDoubleToSByte;
                    case NPTypeCode.Byte:   return StridedDoubleToByte;
                    case NPTypeCode.Int16:  return StridedDoubleToInt16;
                    case NPTypeCode.UInt16: return StridedDoubleToUInt16;
                    case NPTypeCode.Char:   return StridedDoubleToChar;
                }
            }
            return null;
        }

        private static unsafe void StridedSingleToSByte(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 1, &BulkSingleToByteV,  &ConvSingleToSByte);
        private static unsafe void StridedSingleToByte(void* s, void* d, long* ss, long* ds, long* sh, int nd)   => StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 1, &BulkSingleToByteV,  &ConvSingleToByte);
        private static unsafe void StridedSingleToInt16(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 2, &BulkSingleToShortV, &ConvSingleToInt16);
        private static unsafe void StridedSingleToUInt16(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 2, &BulkSingleToShortV, &ConvSingleToUInt16);
        private static unsafe void StridedSingleToChar(void* s, void* d, long* ss, long* ds, long* sh, int nd)   => StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 2, &BulkSingleToShortV, &ConvSingleToChar);
        private static unsafe void StridedDoubleToSByte(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 8, 1, &BulkDoubleToByteV,  &ConvDoubleToSByte);
        private static unsafe void StridedDoubleToByte(void* s, void* d, long* ss, long* ds, long* sh, int nd)   => StridedNarrowDriver(s, d, ss, ds, sh, nd, 8, 1, &BulkDoubleToByteV,  &ConvDoubleToByte);
        private static unsafe void StridedDoubleToInt16(void* s, void* d, long* ss, long* ds, long* sh, int nd)  => StridedNarrowDriver(s, d, ss, ds, sh, nd, 8, 2, &BulkDoubleToShortV, &ConvDoubleToInt16);
        private static unsafe void StridedDoubleToUInt16(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 8, 2, &BulkDoubleToShortV, &ConvDoubleToUInt16);
        private static unsafe void StridedDoubleToChar(void* s, void* d, long* ss, long* ds, long* sh, int nd)   => StridedNarrowDriver(s, d, ss, ds, sh, nd, 8, 2, &BulkDoubleToShortV, &ConvDoubleToChar);

        private static unsafe void StridedNarrowDriver(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim,
            int srcSize, int dstSize,
            delegate*<void*, void*, long, long> bulk,
            delegate*<void*, void*, void> conv)
        {
            const int CHUNK = 4096;
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            if (ndim == 0) { conv(src, dst); return; }

            int outer = ndim - 1;
            long innerN = shape[outer];
            long ss = srcStrides[outer], ds = dstStrides[outer];
            long outerCount = 1;
            for (int a = 0; a < outer; a++) outerCount *= shape[a];

            byte* buf = stackalloc byte[CHUNK * srcSize];   // contiguous staging for strided rows
            long* coord = stackalloc long[ndim];
            for (int a = 0; a < ndim; a++) coord[a] = 0;

            long srcOff = 0, dstOff = 0;   // in elements
            for (long o = 0; o < outerCount; o++)
            {
                byte* sRow = src + srcOff * srcSize;
                byte* dRow = dst + dstOff * dstSize;

                if (ss == 1 && ds == 1)
                {
                    long i = bulk(sRow, dRow, innerN);
                    for (; i < innerN; i++) conv(sRow + i * srcSize, dRow + i * dstSize);
                }
                else if (ds == 1)
                {
                    // strided/reversed source -> stage to contig buf in chunks, vectorize the convert.
                    long j = 0;
                    while (j < innerN)
                    {
                        long c = Math.Min((long)CHUNK, innerN - j);
                        byte* sp = sRow + j * ss * srcSize;
                        // Tight typed strided load -> contiguous staging buffer. Must match srcSize
                        // exactly (1/2/4/8); a wider move would read past each element (the i16->bool
                        // strided bug: srcSize==2 must NOT fall into the 8-byte path).
                        switch (srcSize)
                        {
                            case 1: { byte* b = buf;            byte* s0 = sp;          for (long k = 0; k < c; k++) { b[k] = *s0; s0 += ss; } break; }
                            case 2: { short* b = (short*)buf;   short* s0 = (short*)sp; for (long k = 0; k < c; k++) { b[k] = *s0; s0 += ss; } break; }
                            case 4: { int* b = (int*)buf;       int* s0 = (int*)sp;     for (long k = 0; k < c; k++) { b[k] = *s0; s0 += ss; } break; }
                            default:{ long* b = (long*)buf;     long* s0 = (long*)sp;   for (long k = 0; k < c; k++) { b[k] = *s0; s0 += ss; } break; }
                        }
                        long done = bulk(buf, dRow + j * dstSize, c);
                        for (long k = done; k < c; k++) conv(buf + k * srcSize, dRow + (j + k) * dstSize);
                        j += c;
                    }
                }
                else
                {
                    for (long i = 0; i < innerN; i++)
                        conv(sRow + i * ss * srcSize, dRow + i * ds * dstSize);
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
