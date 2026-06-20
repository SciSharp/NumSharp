using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.ToHalf.cs
        //   X -> Half (float16) contiguous + strided casts for the SIMD-safe sources
        //   {bool,u8,i8,i16,u16,char,i32,f32}.
        //
        //   Phase-0/Wave-7 left f16 as the ONLY losing DST column (geomean ~0.78): every
        //   X->f16 fell to the scalar generic emitter. There is NO F16C in this .NET (no
        //   vectorized (Half) convert — verified in Cast.Half.cs), so narrow with the
        //   vectorized branchless IEEE bit-fiddle in the OTHER direction:
        //
        //     GIESEN float -> half (round-to-nearest-even, branchless): clear sign, split
        //     into normal / subnormal / inf-nan via blend, RTNE via the +0xfff+mantOdd
        //     carry, rebuild the f16 NaN payload as 0x7c00 | (mant>>13) | (payload==0).
        //     -> 8x f16 bit patterns in the low 16 bits of a Vector256<int>; two of those
        //     truncating-Narrow to 16 contiguous u16.
        //
        //   Int sources first widen to i32 (vpmovsx/zx) then cvtdq2ps to f32; THIS DOES NOT
        //   DOUBLE-ROUND for f16 because f16 overflows to +-inf at 65504 << 2^24, so any int
        //   large enough for the f32 step to round is already inf in f16 (proven: i32->f32->
        //   Giesen == NumPy i32->float16 over 2.14M cases incl. the +-65504 boundary, 0 diff).
        //
        //   Bit-exact with NumPy 2.4.2 (proven 0-diff over 5.38M f32 patterns incl. sNaN /
        //   qNaN / +-inf / f32+f16 subnormals / all 65536 round-trips / rounding boundaries).
        //   NOTE: this also FIXES a latent bug — the old scalar path went through the BCL
        //   (Half) cast which QUIETS signaling NaNs (0x7f..->0x7e..); NumPy does NOT, and
        //   Giesen matches NumPy (sNaN 0x7f800001 -> 0x7c01, not 0x7e00).
        //
        //   Deferred (kept on the correct-but-scalar path, no regression): u32 (i32-reinterpret
        //   flips sign for vals >= 2^31), i64/u64 (no AVX2 i64->f32 — needs AVX512), f64
        //   (f64->f32->f16 genuinely double-rounds), c128, f16->f16 (same-type copy family).
        // =====================================================================

        // Giesen branchless float -> half over 8 lanes; result low 16 bits = f16 pattern.
        private static Vector256<int> FloatToHalfBits(Vector256<float> fv)
        {
            var x0 = fv.AsInt32();
            var sign = Avx2.And(x0, Vector256.Create(unchecked((int)0x80000000)));
            var x = Avx2.Xor(x0, sign);                                   // |x| bits
            var f32inf = Vector256.Create(255 << 23);
            var f16max = Vector256.Create((127 + 16) << 23);             // 2^16; > any finite f16
            var denMagic = Vector256.Create(((127 - 15) + (23 - 10) + 1) << 23);
            // subnormal path: add the denormal magic in float space, subtract back as int.
            var sub = Avx2.Subtract(Avx2.Add(x.AsSingle(), denMagic.AsSingle()).AsInt32(), denMagic);
            // normal path: RTNE via +0xfff + (mantissa odd bit), then rebias the exponent.
            var mantOdd = Avx2.And(Avx2.ShiftRightLogical(x, 13), Vector256.Create(1));
            var xn = Avx2.Add(Avx2.Add(x, Vector256.Create(((15 - 127) << 23) + 0xfff)), mantOdd);
            var normal = Avx2.ShiftRightArithmetic(xn, 13);
            // NaN: preserve a non-zero payload (NumPy does NOT quiet sNaN).
            var payload = Avx2.ShiftRightLogical(Avx2.And(x, Vector256.Create(0x7fffff)), 13);
            var pz = Avx2.And(Avx2.CompareEqual(payload, Vector256<int>.Zero), Vector256.Create(1));
            var nanRes = Avx2.Or(Vector256.Create(0x7c00), Avx2.Or(payload, pz));
            var isNan = Avx2.CompareGreaterThan(x, f32inf);
            var infnan = Avx2.BlendVariable(Vector256.Create(0x7c00), nanRes, isNan);
            var isInfNan = Avx2.CompareGreaterThan(x, Avx2.Subtract(f16max, Vector256.Create(1)));
            var isSub = Avx2.CompareGreaterThan(Vector256.Create(113 << 23), x);
            var res = Avx2.BlendVariable(normal, sub, isSub);
            res = Avx2.BlendVariable(res, infnan, isInfNan);
            return Avx2.Or(res, Avx2.ShiftRightLogical(sign, 16));        // re-apply sign bit at f16 pos 15
        }

        // Scalar port of FloatToHalfBits — used for the < 16 tail and the strided scalar conv.
        // MUST match the vector path bit-for-bit (and therefore NumPy); NOT the BCL (Half) cast.
        private static ushort SingleToHalfBits(float fval)
        {
            int x0 = BitConverter.SingleToInt32Bits(fval);
            int sign = x0 & unchecked((int)0x80000000);
            int x = x0 ^ sign;
            int denMagic = ((127 - 15) + (23 - 10) + 1) << 23;
            int sub = BitConverter.SingleToInt32Bits(BitConverter.Int32BitsToSingle(x) + BitConverter.Int32BitsToSingle(denMagic)) - denMagic;
            int mantOdd = (int)((uint)x >> 13) & 1;
            int xn = x + (((15 - 127) << 23) + 0xfff) + mantOdd;
            int normal = xn >> 13;
            int payload = (int)((uint)(x & 0x7fffff) >> 13);
            int pz = payload == 0 ? 1 : 0;
            int nanRes = 0x7c00 | payload | pz;
            int f32inf = 255 << 23, f16max = (127 + 16) << 23;
            int infnan = x > f32inf ? nanRes : 0x7c00;
            int res = ((113 << 23) > x) ? sub : normal;
            if (x > f16max - 1) res = infnan;
            res |= (int)((uint)sign >> 16);
            return (ushort)res;
        }

        // ---- double -> f16 (Wave 11b): no F16C and cvtpd2ps double-ROUNDS, so narrow via
        // round-to-odd f64->f32 (Boldo/Melquiond: round-to-odd then RTNE == direct RTNE) then
        // the proven FloatToHalfBits. NaN payload is built directly from the double bits (the
        // (float) cast would quiet sNaN). Proven 0-diff vs NumPy over 1.88M cases incl. 63K
        // f16-midpoint double-rounding hazards. Shared by f64->f16 and c128->f16. ----
        private static readonly Vector256<int> RtoSel = Vector256.Create(0, 2, 4, 6, 0, 2, 4, 6);

        // round 4 doubles to f32 with round-to-odd (chunk is finite — NaN handled by the caller).
        internal static Vector128<float> RoundToOdd4(Vector256<double> d)
        {
            var f = Avx.ConvertToVector128Single(d);           // RTNE f64->f32
            var bf = Avx.ConvertToVector256Double(f);          // back to f64
            var eq = Avx.Compare(bf, d, FloatComparisonMode.OrderedEqualNonSignaling);
            var am = Vector256.Create(0x7fffffffffffffffL).AsDouble();
            var big = Avx.Compare(Avx.And(d, am), Avx.And(bf, am), FloatComparisonMode.OrderedLessThanSignaling); // |d| < |bf|
            var eqI = Avx2.PermuteVar8x32(eq.AsInt32(), RtoSel).GetLower();   // 4x64 mask -> 4x32
            var bigI = Avx2.PermuteVar8x32(big.AsInt32(), RtoSel).GetLower();
            var fi = f.AsInt32();
            var even = Avx2.CompareEqual(Avx2.And(fi, Vector128.Create(1)), Vector128<int>.Zero);
            var doOdd = Avx2.AndNot(eqI, even);                // ~eq & even  (inexact and even-significand)
            var adj = Avx2.BlendVariable(Vector128.Create(1), Vector128.Create(-1), bigI); // move 1 ULP toward d
            var fOdd = Avx2.Add(fi, adj);
            return Avx2.BlendVariable(fi, fOdd, doOdd).AsSingle();
        }

        internal static bool AnyNaN(Vector256<double> d)
            => Avx.MoveMask(Avx.Compare(d, d, FloatComparisonMode.UnorderedNonSignaling)) != 0;

        // Scalar double -> f16 bits, NumPy-faithful (round-to-odd + direct sNaN payload).
        internal static ushort DoubleToHalfBits(double value)
        {
            if (double.IsNaN(value))
            {
                long b = BitConverter.DoubleToInt64Bits(value);
                int sg = (int)(((ulong)b >> 48) & 0x8000);
                int pl = (int)((b >> 42) & 0x3FF);             // top 10 bits of the 52-bit mantissa
                if (pl == 0) pl = 1;
                return (ushort)(sg | 0x7C00 | pl);
            }
            if (double.IsInfinity(value)) return SingleToHalfBits((float)value);
            float f = (float)value; double bf = (double)f;
            if (bf != value)
            {
                uint u = BitConverter.SingleToUInt32Bits(f);
                if ((u & 1) == 0) { if (Math.Abs(bf) > Math.Abs(value)) u--; else u++; f = BitConverter.UInt32BitsToSingle(u); }
            }
            return SingleToHalfBits(f);
        }

        // 16 doubles -> 16 f16 (round-to-odd path; per-block scalar fallback when any lane is NaN).
        private static unsafe long BulkDoubleToHalf(void* s, void* d, long n)
        {
            double* src = (double*)s; ushort* dst = (ushort*)d; long i = 0;
            for (; i + 16 <= n; i += 16)
            {
                var a0 = Vector256.Load(src + i); var a1 = Vector256.Load(src + i + 4);
                var a2 = Vector256.Load(src + i + 8); var a3 = Vector256.Load(src + i + 12);
                if (AnyNaN(a0) || AnyNaN(a1) || AnyNaN(a2) || AnyNaN(a3))
                { for (int k = 0; k < 16; k++) dst[i + k] = DoubleToHalfBits(src[i + k]); continue; }
                var lo = FloatToHalfBits(Vector256.Create(RoundToOdd4(a0), RoundToOdd4(a1)));
                var hi = FloatToHalfBits(Vector256.Create(RoundToOdd4(a2), RoundToOdd4(a3)));
                Vector256.Store(Vector256.Narrow(lo, hi).AsUInt16(), dst + i);
            }
            return i;
        }

        // u32 -> f16: cvtdq2ps treats the i32-reinterpret as signed, so any u32 >= 2^31 (always
        // >= 65536 -> +inf in f16) gets its high-bit-set lanes forced to +inf; everything below
        // 2^31 converts faithfully. Proven 0-diff vs NumPy over 2.07M cases. Scalar uses the
        // correct C# (float)(uint).
        private static unsafe long BulkUInt32ToHalf(void* s, void* d, long n)
        {
            uint* src = (uint*)s; ushort* dst = (ushort*)d; long i = 0;
            for (; i + 16 <= n; i += 16)
            {
                var ra = Vector256.Load((int*)(src + i));
                var a = FloatToHalfBits(Avx.ConvertToVector256Single(ra));
                a = Avx2.BlendVariable(a, Vector256.Create(0x7c00), Avx2.CompareGreaterThan(Vector256<int>.Zero, ra));
                var rb = Vector256.Load((int*)(src + i + 8));
                var b = FloatToHalfBits(Avx.ConvertToVector256Single(rb));
                b = Avx2.BlendVariable(b, Vector256.Create(0x7c00), Avx2.CompareGreaterThan(Vector256<int>.Zero, rb));
                Vector256.Store(Vector256.Narrow(a, b).AsUInt16(), dst + i);
            }
            return i;
        }

        // ---- SIMD bulks: read N contiguous src elements, write N contiguous f16. ----
        // Each processes 16-at-a-time (two 8-lane Giesen -> truncating Narrow -> 16 u16) and
        // returns the count handled (multiple of 16); the caller scalar-converts the tail.

        private static unsafe long BulkSingleToHalf(void* s, void* d, long n)
        {
            float* src = (float*)s; ushort* dst = (ushort*)d; long i = 0;
            for (; i + 16 <= n; i += 16)
            {
                var a = FloatToHalfBits(Vector256.Load(src + i));
                var b = FloatToHalfBits(Vector256.Load(src + i + 8));
                Vector256.Store(Vector256.Narrow(a, b).AsUInt16(), dst + i);
            }
            return i;
        }
        private static unsafe long BulkInt32ToHalf(void* s, void* d, long n)
        {
            int* src = (int*)s; ushort* dst = (ushort*)d; long i = 0;
            for (; i + 16 <= n; i += 16)
            {
                var a = FloatToHalfBits(Avx.ConvertToVector256Single(Vector256.Load(src + i)));
                var b = FloatToHalfBits(Avx.ConvertToVector256Single(Vector256.Load(src + i + 8)));
                Vector256.Store(Vector256.Narrow(a, b).AsUInt16(), dst + i);
            }
            return i;
        }
        private static unsafe long BulkInt16ToHalf(void* s, void* d, long n)   // sign-extend i16
        {
            short* src = (short*)s; ushort* dst = (ushort*)d; long i = 0;
            for (; i + 16 <= n; i += 16)
            {
                var a = FloatToHalfBits(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.LoadVector128(src + i))));
                var b = FloatToHalfBits(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.LoadVector128(src + i + 8))));
                Vector256.Store(Vector256.Narrow(a, b).AsUInt16(), dst + i);
            }
            return i;
        }
        private static unsafe long BulkUInt16ToHalf(void* s, void* d, long n)  // zero-extend u16/char
        {
            ushort* src = (ushort*)s; ushort* dst = (ushort*)d; long i = 0;
            for (; i + 16 <= n; i += 16)
            {
                var a = FloatToHalfBits(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.LoadVector128(src + i))));
                var b = FloatToHalfBits(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.LoadVector128(src + i + 8))));
                Vector256.Store(Vector256.Narrow(a, b).AsUInt16(), dst + i);
            }
            return i;
        }
        private static unsafe long BulkByteToHalf(void* s, void* d, long n)    // zero-extend bool/u8
        {
            byte* src = (byte*)s; ushort* dst = (ushort*)d; long i = 0;
            for (; i + 16 <= n; i += 16)
            {
                var a = FloatToHalfBits(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.LoadScalarVector128((long*)(src + i)).AsByte())));
                var b = FloatToHalfBits(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.LoadScalarVector128((long*)(src + i + 8)).AsByte())));
                Vector256.Store(Vector256.Narrow(a, b).AsUInt16(), dst + i);
            }
            return i;
        }
        private static unsafe long BulkSByteToHalf(void* s, void* d, long n)   // sign-extend i8
        {
            sbyte* src = (sbyte*)s; ushort* dst = (ushort*)d; long i = 0;
            for (; i + 16 <= n; i += 16)
            {
                var a = FloatToHalfBits(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.LoadScalarVector128((long*)(src + i)).AsSByte())));
                var b = FloatToHalfBits(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.LoadScalarVector128((long*)(src + i + 8)).AsSByte())));
                Vector256.Store(Vector256.Narrow(a, b).AsUInt16(), dst + i);
            }
            return i;
        }

        // ---- scalar conv (void*,void*) for the strided driver; (float)widen then Giesen ----
        private static unsafe void ConvSingleToHalf(void* s, void* d) => *(ushort*)d = SingleToHalfBits(*(float*)s);
        private static unsafe void ConvInt32ToHalf(void* s, void* d) => *(ushort*)d = SingleToHalfBits((float)*(int*)s);
        private static unsafe void ConvInt16ToHalf(void* s, void* d) => *(ushort*)d = SingleToHalfBits((float)*(short*)s);
        private static unsafe void ConvUInt16ToHalf(void* s, void* d) => *(ushort*)d = SingleToHalfBits((float)*(ushort*)s);
        private static unsafe void ConvByteToHalf(void* s, void* d) => *(ushort*)d = SingleToHalfBits((float)*(byte*)s);
        private static unsafe void ConvSByteToHalf(void* s, void* d) => *(ushort*)d = SingleToHalfBits((float)*(sbyte*)s);
        private static unsafe void ConvUInt32ToHalf(void* s, void* d) => *(ushort*)d = SingleToHalfBits((float)*(uint*)s);
        private static unsafe void ConvDoubleToHalf(void* s, void* d) => *(ushort*)d = DoubleToHalfBits(*(double*)s);

        // ---- contiguous kernels: bulk + NumPy-faithful scalar tail ----
        private static unsafe void CastSingleToHalfContig(void* s, void* d, long n)
        { long i = BulkSingleToHalf(s, d, n); float* p = (float*)s; ushort* o = (ushort*)d; for (; i < n; i++) o[i] = SingleToHalfBits(p[i]); }
        private static unsafe void CastInt32ToHalfContig(void* s, void* d, long n)
        { long i = BulkInt32ToHalf(s, d, n); int* p = (int*)s; ushort* o = (ushort*)d; for (; i < n; i++) o[i] = SingleToHalfBits((float)p[i]); }
        private static unsafe void CastInt16ToHalfContig(void* s, void* d, long n)
        { long i = BulkInt16ToHalf(s, d, n); short* p = (short*)s; ushort* o = (ushort*)d; for (; i < n; i++) o[i] = SingleToHalfBits((float)p[i]); }
        private static unsafe void CastUInt16ToHalfContig(void* s, void* d, long n)
        { long i = BulkUInt16ToHalf(s, d, n); ushort* p = (ushort*)s; ushort* o = (ushort*)d; for (; i < n; i++) o[i] = SingleToHalfBits((float)p[i]); }
        private static unsafe void CastByteToHalfContig(void* s, void* d, long n)
        { long i = BulkByteToHalf(s, d, n); byte* p = (byte*)s; ushort* o = (ushort*)d; for (; i < n; i++) o[i] = SingleToHalfBits((float)p[i]); }
        private static unsafe void CastSByteToHalfContig(void* s, void* d, long n)
        { long i = BulkSByteToHalf(s, d, n); sbyte* p = (sbyte*)s; ushort* o = (ushort*)d; for (; i < n; i++) o[i] = SingleToHalfBits((float)p[i]); }
        private static unsafe void CastUInt32ToHalfContig(void* s, void* d, long n)
        { long i = BulkUInt32ToHalf(s, d, n); uint* p = (uint*)s; ushort* o = (ushort*)d; for (; i < n; i++) o[i] = SingleToHalfBits((float)p[i]); }
        private static unsafe void CastDoubleToHalfContig(void* s, void* d, long n)
        { long i = BulkDoubleToHalf(s, d, n); double* p = (double*)s; ushort* o = (ushort*)d; for (; i < n; i++) o[i] = DoubleToHalfBits(p[i]); }

        /// <summary>
        /// Contiguous <see cref="CastKernel"/> for {bool,u8,i8,i16,u16,char,i32,f32} -&gt; Half,
        /// or null. Bit-exact with NumPy 2.4.2 (Giesen RTNE, sNaN-preserving).
        /// </summary>
        internal static unsafe CastKernel TryGetXToHalfKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (dstType != NPTypeCode.Half || !Avx2.IsSupported) return null;
            switch (srcType)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte: return CastByteToHalfContig;
                case NPTypeCode.SByte: return CastSByteToHalfContig;
                case NPTypeCode.Int16: return CastInt16ToHalfContig;
                case NPTypeCode.UInt16:
                case NPTypeCode.Char: return CastUInt16ToHalfContig;
                case NPTypeCode.Int32: return CastInt32ToHalfContig;
                case NPTypeCode.UInt32: return CastUInt32ToHalfContig;
                case NPTypeCode.Single: return CastSingleToHalfContig;
                case NPTypeCode.Double: return CastDoubleToHalfContig;
                case NPTypeCode.Complex: return CastComplexToHalfContig;
            }
            return null;
        }

        // ---- strided via StridedNarrowDriver (dstSize = 2). Inner-contig rows run the bulk,
        // inner-strided rows stage to a contig buffer then vectorize, scalar conv elsewhere. ----
        private static unsafe void StridedSingleToHalf(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 2, &BulkSingleToHalf, &ConvSingleToHalf);
        private static unsafe void StridedInt32ToHalf(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 2, &BulkInt32ToHalf, &ConvInt32ToHalf);
        private static unsafe void StridedInt16ToHalf(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 2, &BulkInt16ToHalf, &ConvInt16ToHalf);
        private static unsafe void StridedUInt16ToHalf(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 2, 2, &BulkUInt16ToHalf, &ConvUInt16ToHalf);
        private static unsafe void StridedByteToHalf(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 1, 2, &BulkByteToHalf, &ConvByteToHalf);
        private static unsafe void StridedSByteToHalf(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 1, 2, &BulkSByteToHalf, &ConvSByteToHalf);
        private static unsafe void StridedUInt32ToHalf(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 4, 2, &BulkUInt32ToHalf, &ConvUInt32ToHalf);
        private static unsafe void StridedDoubleToHalf(void* s, void* d, long* ss, long* ds, long* sh, int nd) => StridedNarrowDriver(s, d, ss, ds, sh, nd, 8, 2, &BulkDoubleToHalf, &ConvDoubleToHalf);

        /// <summary>
        /// Strided <see cref="StridedCastKernel"/> for {bool,u8,i8,i16,u16,char,i32,f32} -&gt; Half, or null.
        /// </summary>
        internal static unsafe StridedCastKernel TryGetXToHalfStridedKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (dstType != NPTypeCode.Half || !Avx2.IsSupported) return null;
            switch (srcType)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte: return StridedByteToHalf;
                case NPTypeCode.SByte: return StridedSByteToHalf;
                case NPTypeCode.Int16: return StridedInt16ToHalf;
                case NPTypeCode.UInt16:
                case NPTypeCode.Char: return StridedUInt16ToHalf;
                case NPTypeCode.Int32: return StridedInt32ToHalf;
                case NPTypeCode.UInt32: return StridedUInt32ToHalf;
                case NPTypeCode.Single: return StridedSingleToHalf;
                case NPTypeCode.Double: return StridedDoubleToHalf;
                case NPTypeCode.Complex: return StridedComplexToHalf;
            }
            return null;
        }
    }
}
