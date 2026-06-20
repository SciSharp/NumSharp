using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.FloatWideInt.cs
        //   f32 -> i32 strided cast (same-width 4->4, no Narrow step).
        //
        //   Phase-0 left f32->i32 strided at 0.24x: the contig f32->i32 kernel exists
        //   (TryGetFloatToInt32Kernel) but there was NO strided Single->i32 kernel (only
        //   Double->i32), so every strided f32->i32 fell to the IL scalar loop.
        //
        //   Mirrors the Wave-7 float->narrow FusedGather* model: WHOLE ARRAY in one call
        //   with the gather index hoisted OUTSIDE the outer odometer (a per-row call
        //   boundary blocks the JIT from pipelining gathers across rows — proven 1.68x vs
        //   0.70x). Inner ss==1 rows run a plain contiguous cvtt; inner ss!=1 rows (incl.
        //   reversed [:, ::-1], signed idx) run VPGATHERDD+cvtt; ds!=1 / no-AVX2 / the tail
        //   run the NumPy-faithful Converts scalar.
        //
        //   Bit-exact, proven vs NumPy 2.4.2: VCVTTPS2DQ == Converts.ToInt32 for ALL inputs
        //   — truncate-to-zero, INT_MIN (0x80000000) sentinel on NaN/Inf/overflow (incl.
        //   exactly 2^31 -> -2^31).
        //
        //   f32 -> u32 is intentionally NOT here: NumPy's u32 cast is a modular wrap over
        //   the full finite range (5e9 -> 705032704, -3e9 -> 1294967296) with only
        //   NaN/+/-Inf -> 0, which needs a float->i64 convert (VCVTTPS2QQ, AVX512) to
        //   vectorize faithfully. The AVX2 subtract-2^31 fixup only covers [0, 2^32), so
        //   u32 stays on the correct Converts.ToUInt32 scalar path until an AVX512 variant
        //   lands. (f32->u32 strided is a mild 0.49x, not a cliff.)
        // =====================================================================

        internal static unsafe StridedCastKernel TryGetFloatToWideIntStridedKernel(NPTypeCode srcType, NPTypeCode dstType)
            => (srcType == NPTypeCode.Single && dstType == NPTypeCode.Int32) ? CastSingleToInt32Strided : null;

        // f32 -> i32: ss==1 contig cvtt, ss!=1 VPGATHERDD+cvtt, scalar Converts tail.
        private static unsafe void CastSingleToInt32Strided(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            float* src = (float*)srcV; int* dst = (int*)dstV;
            if (ndim == 0) { dst[0] = Converts.ToInt32(src[0]); return; }

            int outer = ndim - 1;
            long innerN = shape[outer], ss = srcStrides[outer], ds = dstStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;

            bool gatherable = ds == 1 && ss != 1 && ss >= int.MinValue / 8 && ss <= int.MaxValue / 8 && Avx2.IsSupported;
            int si = (int)ss;
            var idx = Vector256.Create(0, si, 2 * si, 3 * si, 4 * si, 5 * si, 6 * si, 7 * si);
            long g = 8L * ss;

            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                float* sRow = src + srcOff; int* dRow = dst + dstOff; long i = 0;
                if (ds == 1 && ss == 1 && Avx.IsSupported)
                {
                    for (; i + 8 <= innerN; i += 8)
                        Vector256.Store(Avx.ConvertToVector256Int32WithTruncation(Vector256.Load(sRow + i)), dRow + i);
                }
                else if (gatherable)
                {
                    int* p = (int*)sRow;
                    for (; i + 8 <= innerN; i += 8)
                    {
                        Vector256.Store(Avx.ConvertToVector256Int32WithTruncation(Avx2.GatherVector256(p, idx, 4).AsSingle()), dRow + i);
                        p += g;
                    }
                }
                for (; i < innerN; i++) dRow[i * ds] = Converts.ToInt32(sRow[i * ss]);

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
