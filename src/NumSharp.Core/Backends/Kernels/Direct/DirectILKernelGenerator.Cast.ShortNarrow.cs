using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.ShortNarrow.cs
        //   Char -> {i8,u8} contiguous narrow (32-wide truncating Vector.Narrow).
        //
        //   Measured at HEAD, char -> {i8,u8} lags on every layout (~0.8x) while
        //   i16/u16 -> {i8,u8} win: the generic IL emitter SIMD-narrows i16/u16 but
        //   treats Char as non-arithmetic and falls to a scalar per-element cast. This
        //   contig kernel SIMD-narrows char; i16/u16 keep the already-fast generic path.
        //
        //   The strided (2,1) case is intentionally NOT routed here: 2-byte sources are
        //   not VPGATHER-eligible, and staging strided rows to a contig buffer (the
        //   StridedNarrowDriver path the float->narrow kernels use) costs more than the
        //   one-cycle truncate it feeds — measured a REGRESSION vs the generic strided
        //   emitter (i16->i8 strided 0.82 -> 0.47, negcol 1.10 -> 0.39). Staging only
        //   pays off when the per-element convert is expensive (e.g. the Giesen f16
        //   narrow); a cheap low-byte truncate keeps the generic strided path.
        //
        //   Truncating Vector.Narrow == unchecked (byte) low-byte == NumPy int->int wrap
        //   (signedness-agnostic; the dst's signed/unsigned reading is the same byte).
        // =====================================================================

        // 32 contiguous u16 -> 32 contiguous bytes (truncate low byte), 32-wide.
        private static unsafe long BulkShortToByte(void* s, void* d, long n)
        {
            ushort* src = (ushort*)s; byte* dst = (byte*)d; long i = 0;
            for (; i + 32 <= n; i += 32)
            {
                var a = Vector256.Load(src + i);
                var b = Vector256.Load(src + i + 16);
                Vector256.Store(Vector256.Narrow(a, b), dst + i);   // 32x byte, low-byte truncation
            }
            return i;
        }

        private static unsafe void CastShortToByteContig(void* s, void* d, long n)
        {
            long i = BulkShortToByte(s, d, n);
            ushort* p = (ushort*)s; byte* o = (byte*)d;
            for (; i < n; i++) o[i] = (byte)p[i];
        }

        /// <summary>
        /// Contig <see cref="CastKernel"/> for Char -&gt; {i8,u8} (the generic emitter's
        /// char gap), or null. i16/u16 contig keep the already-fast generic path.
        /// </summary>
        internal static unsafe CastKernel TryGetCharToByteContigKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (!Avx2.IsSupported) return null;
            if (srcType == NPTypeCode.Char && (dstType == NPTypeCode.SByte || dstType == NPTypeCode.Byte))
                return CastShortToByteContig;
            return null;
        }
    }
}
