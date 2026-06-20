using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.FloatToUInt.cs
        //   f64 -> u32 and f32 -> u32 — bit-exact AVX2 (no AVX512 needed).
        //
        //   THE PROBLEM (documented in Cast.FloatWideInt.cs as deferred):
        //   NumPy's float->u32 is `(npy_uint32)(npy_int64)trunc(x)` — a MODULAR wrap
        //   over the whole finite range, NOT a saturate:
        //       5e9 -> 705032704,  1e10 -> 1410065408,  -1 -> 4294967295,
        //       -3e9 -> 1294967296,  2^32 -> 0,  NaN/+-Inf/|x|>=2^63 -> 0.
        //   The naive AVX2 "subtract 2^31" range-shift only covers [0, 2^32) and
        //   saturates beyond it (5e9 -> 0), so it diverges from NumPy. And the AVX512
        //   VCVTTPS2UDQ instruction SATURATES too (5e9 -> 0xFFFFFFFF), so it ALSO does
        //   not match NumPy — even with AVX512 the faithful path would be
        //   VCVTTPS2QQ (float->i64) then a low-32 narrow.
        //
        //   THE FIX (AVX2, full finite range, bit-exact — proven 0 diffs / 800K incl.
        //   every edge): reduce mod 2^32 in double space, THEN range-shift:
        //       t = trunc(x)                                  // VROUNDPD toward zero
        //       r = t - 2^32 * floor(t * 2^-32)               // r = t mod 2^32 in [0,2^32)
        //                                                     // exact: t integer, all ops *2^k
        //       shifted = r - (r >= 2^31 ? 2^32 : 0)          // -> [-2^31, 2^31)
        //       i = cvttpd2dq(shifted)                        // signed int32 whose BITS are the u32
        //       result = (|x| < 2^63) ? i : 0                 // mask folds NaN/Inf/overflow -> 0
        //   The mod-2^32 step is what the old range-shift lacked; it makes the wrap
        //   (5e9 -> 705032704, -3e9 -> 1294967296) come out exactly. cvttpd2dq's signed
        //   wrap turns the [2^31,2^32) tail into the right unsigned bit-pattern, and the
        //   |x|<2^63 compare is false for NaN (unordered), Inf, and >=2^63 -> all map to 0.
        //
        //   f32 -> u32 widens each lane f32->f64 first (lossless): a direct-f32 mod-2^32
        //   reduction lands negative results like 4294967295 that aren't representable in
        //   float (round to 2^32), corrupting the cvtt. Widening sidesteps that for free.
        //
        //   Bit-exact with Converts.ToUInt32(double/float) — the NumPy-faithful scalar
        //   reference (NaN/Inf/|x|>=2^63 -> 0, else (uint)(long)trunc) — which the scalar
        //   tail also uses.
        // =====================================================================

        // 4 doubles -> 4 u32 (returned as int128; bits ARE the u32 values).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<int> DoubleToU32x4(Vector256<double> x)
        {
            var two32   = Vector256.Create(4294967296.0);
            var inv32   = Vector256.Create(2.3283064365386963e-10);          // 2^-32 (exact)
            var two31   = Vector256.Create(2147483648.0);
            var two63   = Vector256.Create(9223372036854775808.0);
            var absmask = Vector256.Create(-0.0);

            var t = Avx.RoundToZero(x);                                       // trunc toward zero
            var q = Avx.Floor(Avx.Multiply(t, inv32));                       // floor(t / 2^32)
            var r = Avx.Subtract(t, Avx.Multiply(q, two32));                // t mod 2^32 in [0,2^32)
            var lt = Avx.Compare(r, two31, FloatComparisonMode.OrderedLessThanSignaling);
            var shifted = Avx.Subtract(r, Avx.AndNot(lt, two32));           // [-2^31, 2^31)
            var valid = Avx.Compare(Avx.AndNot(absmask, x), two63, FloatComparisonMode.OrderedLessThanSignaling);
            return Avx.ConvertToVector128Int32WithTruncation(Avx.And(shifted, valid));
        }

        // 8 floats -> 8 u32 (widen each half to f64, convert, recombine).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<int> SingleToU32x8(Vector256<float> f)
        {
            var lo = DoubleToU32x4(Avx.ConvertToVector256Double(f.GetLower()));
            var hi = DoubleToU32x4(Avx.ConvertToVector256Double(f.GetUpper()));
            return Vector256.Create(lo, hi);
        }

        // ---- contig bulks (return count consumed by the SIMD body) ----
        private static unsafe long BulkDoubleToUInt32(void* s, void* d, long n)
        {
            double* src = (double*)s; uint* dst = (uint*)d; long i = 0;
            if (Avx2.IsSupported)
                for (; i + 4 <= n; i += 4)
                    Vector128.Store(DoubleToU32x4(Vector256.Load(src + i)).AsUInt32(), dst + i);
            return i;
        }

        private static unsafe long BulkSingleToUInt32(void* s, void* d, long n)
        {
            float* src = (float*)s; uint* dst = (uint*)d; long i = 0;
            if (Avx2.IsSupported)
                for (; i + 8 <= n; i += 8)
                    Vector256.Store(SingleToU32x8(Vector256.Load(src + i)).AsUInt32(), dst + i);
            return i;
        }

        // ---- contig kernels ----
        private static unsafe void CastDoubleToUInt32Contig(void* s, void* d, long n)
        {
            long i = BulkDoubleToUInt32(s, d, n);
            double* p = (double*)s; uint* o = (uint*)d;
            for (; i < n; i++) o[i] = Converts.ToUInt32(p[i]);
        }

        private static unsafe void CastSingleToUInt32Contig(void* s, void* d, long n)
        {
            long i = BulkSingleToUInt32(s, d, n);
            float* p = (float*)s; uint* o = (uint*)d;
            for (; i < n; i++) o[i] = Converts.ToUInt32(p[i]);
        }

        // ---- strided kernels (mirror CastSingleToInt32Strided: ss==1 contig bulk,
        //      ss!=1 VPGATHER + convert, ds!=1 / tail scalar Converts) ----
        private static unsafe void CastDoubleToUInt32Strided(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            double* src = (double*)srcV; uint* dst = (uint*)dstV;
            if (ndim == 0) { dst[0] = Converts.ToUInt32(src[0]); return; }

            int outer = ndim - 1;
            long innerN = shape[outer], ss = srcStrides[outer], ds = dstStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;

            bool gatherable = ds == 1 && ss != 1 && Avx2.IsSupported;
            var idx = Vector256.Create(0L, ss, 2 * ss, 3 * ss);
            long g = 4L * ss;

            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                double* sRow = src + srcOff; uint* dRow = dst + dstOff; long i = 0;
                if (ds == 1 && ss == 1)
                    i = BulkDoubleToUInt32(sRow, dRow, innerN);
                else if (gatherable)
                {
                    double* p = sRow;
                    for (; i + 4 <= innerN; i += 4)
                    {
                        Vector128.Store(DoubleToU32x4(Avx2.GatherVector256(p, idx, 8)).AsUInt32(), dRow + i);
                        p += g;
                    }
                }
                for (; i < innerN; i++) dRow[i * ds] = Converts.ToUInt32(sRow[i * ss]);

                for (int ax = outer - 1; ax >= 0; ax--)
                {
                    coord[ax]++; srcOff += srcStrides[ax]; dstOff += dstStrides[ax];
                    if (coord[ax] < shape[ax]) break;
                    coord[ax] = 0; srcOff -= srcStrides[ax] * shape[ax]; dstOff -= dstStrides[ax] * shape[ax];
                }
            }
        }

        private static unsafe void CastSingleToUInt32Strided(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            float* src = (float*)srcV; uint* dst = (uint*)dstV;
            if (ndim == 0) { dst[0] = Converts.ToUInt32(src[0]); return; }

            int outer = ndim - 1;
            long innerN = shape[outer], ss = srcStrides[outer], ds = dstStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;

            bool gatherable = ds == 1 && ss != 1 && ss >= int.MinValue / 8 && ss <= int.MaxValue / 8 && Avx2.IsSupported;
            int si = (int)ss;
            var idx = Vector256.Create(0, si, 2 * si, 3 * si, 4 * si, 5 * si, 6 * si, 7 * si);
            long g = 8L * ss;
            // Staging buffer (reused per row). f32's gather feeds an 8-wide widen->convert
            // chain; routing the gather straight into it stalls on gather latency (~0.69ms/500K).
            // Gathering into this contig buffer first lets the gathers pipeline, then the convert
            // runs at contig speed (~0.30ms/500K, 2.5x). f64 has no widen so it gathers inline.
            const int TILE = 4096;
            float* buf = stackalloc float[TILE];

            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                float* sRow = src + srcOff; uint* dRow = dst + dstOff; long i = 0;
                if (ds == 1 && ss == 1)
                    i = BulkSingleToUInt32(sRow, dRow, innerN);
                else if (gatherable)
                {
                    long off = 0;
                    while (off < innerN)
                    {
                        long m = innerN - off; if (m > TILE) m = TILE;
                        int* p = (int*)(sRow + off * ss); long j = 0;
                        for (; j + 8 <= m; j += 8) { Vector256.Store(Avx2.GatherVector256(p, idx, 4), (int*)buf + j); p += g; }
                        for (; j < m; j++) buf[j] = sRow[(off + j) * ss];
                        j = 0;
                        for (; j + 8 <= m; j += 8) Vector256.Store(SingleToU32x8(Vector256.Load(buf + j)).AsUInt32(), dRow + off + j);
                        for (; j < m; j++) dRow[off + j] = Converts.ToUInt32(buf[j]);
                        off += m;
                    }
                    i = innerN;
                }
                for (; i < innerN; i++) dRow[i * ds] = Converts.ToUInt32(sRow[i * ss]);

                for (int ax = outer - 1; ax >= 0; ax--)
                {
                    coord[ax]++; srcOff += srcStrides[ax]; dstOff += dstStrides[ax];
                    if (coord[ax] < shape[ax]) break;
                    coord[ax] = 0; srcOff -= srcStrides[ax] * shape[ax]; dstOff -= dstStrides[ax] * shape[ax];
                }
            }
        }

        /// <summary>NumPy-faithful AVX2 contig <see cref="CastKernel"/> for {f32,f64}-&gt;u32, or null.</summary>
        internal static unsafe CastKernel TryGetFloatToUInt32Kernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (dstType != NPTypeCode.UInt32 || !Avx2.IsSupported) return null;
            if (srcType == NPTypeCode.Double) return CastDoubleToUInt32Contig;
            if (srcType == NPTypeCode.Single) return CastSingleToUInt32Contig;
            return null;
        }

        /// <summary>NumPy-faithful AVX2 strided <see cref="StridedCastKernel"/> for {f32,f64}-&gt;u32, or null.</summary>
        internal static unsafe StridedCastKernel TryGetFloatToUInt32StridedKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (dstType != NPTypeCode.UInt32 || !Avx2.IsSupported) return null;
            if (srcType == NPTypeCode.Double) return CastDoubleToUInt32Strided;
            if (srcType == NPTypeCode.Single) return CastSingleToUInt32Strided;
            return null;
        }
    }
}
