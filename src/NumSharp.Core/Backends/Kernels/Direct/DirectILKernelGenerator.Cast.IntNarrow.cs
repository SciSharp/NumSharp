using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.IntNarrow.cs
        //   {i32,u32,i64,u64} -> narrower int (i8/u8/i16/u16/char[/i32/u32]) STRIDED.
        //
        //   Phase-0 / Wave-7 left int->narrow strided at 0.81x (contig/sliced already
        //   win ~1.3x via the EmitNarrowInt auto-vectorized path; only the inner-strided
        //   ([:, ::2]) / reversed layout fell to the generic scalar inner loop).
        //
        //   Same whole-array fused-gather model as the Wave-7 float->narrow kernels, but
        //   INTEGER: there is NO cvtt — the gathered int IS the value; a truncating
        //   Vector.Narrow chain drops it to the dst width (low bits == unchecked (narrow)
        //   cast == NumPy int->int wrap; signedness-agnostic, so i32/u32 and the signed/
        //   unsigned dst variants share one kernel keyed by (srcSize, dstSize)).
        //
        //   Inner ss==1 rows run a plain contiguous Vector256.Load + Narrow; inner ss!=1
        //   rows (incl. reversed [:, ::-1], signed gather index) run VPGATHERDD/VPGATHERQQ
        //   + Narrow; ds!=1 / no-AVX2 / the tail run the scalar truncating cast. idx is
        //   hoisted outside the outer odometer (per-row call boundary blocks the JIT from
        //   pipelining gathers across rows — proven 1.68x vs 0.70x). Proven vs NumPy 2.4.2:
        //   i32->i8 1.51x, i32->i16 1.63x, i64->i8 1.01x strided, 0-diff.
        // =====================================================================

        internal static unsafe StridedCastKernel TryGetIntToNarrowStridedKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (!Avx2.IsSupported) return null;
            int ss = srcType switch { NPTypeCode.Int32 or NPTypeCode.UInt32 => 4, NPTypeCode.Int64 or NPTypeCode.UInt64 => 8, _ => 0 };
            if (ss == 0) return null;
            int ds = dstType switch
            {
                NPTypeCode.SByte or NPTypeCode.Byte => 1,
                NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char => 2,
                NPTypeCode.Int32 or NPTypeCode.UInt32 => 4,
                _ => 0
            };
            if (ds == 0 || ds >= ss) return null;   // must strictly narrow
            return (ss, ds) switch
            {
                (4, 1) => CastInt32ToByteStrided,
                (4, 2) => CastInt32ToShortStrided,
                (8, 1) => CastInt64ToByteStrided,
                (8, 2) => CastInt64ToShortStrided,
                (8, 4) => CastInt64ToIntStrided,
                _ => null
            };
        }

        // ---- (4,1) i32/u32 -> i8/u8 : 32-wide (4 loads/gathers -> 2-level Narrow) ----
        private static unsafe void CastInt32ToByteStrided(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            if (ndim == 0) { dst[0] = (byte)*(int*)src; return; }
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer], ds = dstStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            bool gather = ds == 1 && ss != 1 && ss >= int.MinValue / 8 && ss <= int.MaxValue / 8;
            int si = (int)ss; var idx = Vector256.Create(0, si, 2 * si, 3 * si, 4 * si, 5 * si, 6 * si, 7 * si);
            long g = 8L * ss;
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                int* p = (int*)(src + srcOff * 4); byte* dr = dst + dstOff; long i = 0;
                if (ds == 1 && ss == 1)
                {
                    for (; i + 32 <= innerN; i += 32)
                    {
                        var a = Vector256.Load(p + i); var b = Vector256.Load(p + i + 8);
                        var c = Vector256.Load(p + i + 16); var e = Vector256.Load(p + i + 24);
                        Vector256.Store(Vector256.Narrow(Vector256.Narrow(a, b), Vector256.Narrow(c, e)).AsByte(), dr + i);
                    }
                }
                else if (gather)
                {
                    int* q = p;
                    for (; i + 32 <= innerN; i += 32)
                    {
                        var a = Avx2.GatherVector256(q, idx, 4); var b = Avx2.GatherVector256(q + g, idx, 4);
                        var c = Avx2.GatherVector256(q + 2 * g, idx, 4); var e = Avx2.GatherVector256(q + 3 * g, idx, 4);
                        Vector256.Store(Vector256.Narrow(Vector256.Narrow(a, b), Vector256.Narrow(c, e)).AsByte(), dr + i);
                        q += 4 * g;
                    }
                    for (; i + 8 <= innerN; i += 8) { var v = Avx2.GatherVector256(q, idx, 4); for (int z = 0; z < 8; z++) dr[i + z] = (byte)v.GetElement(z); q += g; }
                }
                for (; i < innerN; i++) dr[i * ds] = (byte)*(int*)(src + (srcOff + i * ss) * 4);
                AdvanceOdometer(coord, srcStrides, dstStrides, shape, outer, ref srcOff, ref dstOff);
            }
        }

        // ---- (4,2) i32/u32 -> i16/u16/char : 16-wide (2 -> 1-level Narrow) ----
        private static unsafe void CastInt32ToShortStrided(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            if (ndim == 0) { *(short*)dst = (short)*(int*)src; return; }
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer], ds = dstStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            bool gather = ds == 1 && ss != 1 && ss >= int.MinValue / 8 && ss <= int.MaxValue / 8;
            int si = (int)ss; var idx = Vector256.Create(0, si, 2 * si, 3 * si, 4 * si, 5 * si, 6 * si, 7 * si);
            long g = 8L * ss;
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                int* p = (int*)(src + srcOff * 4); short* dr = (short*)(dst + dstOff * 2); long i = 0;
                if (ds == 1 && ss == 1)
                {
                    for (; i + 16 <= innerN; i += 16)
                        Vector256.Store(Vector256.Narrow(Vector256.Load(p + i), Vector256.Load(p + i + 8)), dr + i);
                }
                else if (gather)
                {
                    int* q = p;
                    for (; i + 16 <= innerN; i += 16)
                    {
                        Vector256.Store(Vector256.Narrow(Avx2.GatherVector256(q, idx, 4), Avx2.GatherVector256(q + g, idx, 4)), dr + i);
                        q += 2 * g;
                    }
                    for (; i + 8 <= innerN; i += 8) { var v = Avx2.GatherVector256(q, idx, 4); for (int z = 0; z < 8; z++) dr[i + z] = (short)v.GetElement(z); q += g; }
                }
                for (; i < innerN; i++) *(short*)(dst + (dstOff + i * ds) * 2) = (short)*(int*)(src + (srcOff + i * ss) * 4);
                AdvanceOdometer(coord, srcStrides, dstStrides, shape, outer, ref srcOff, ref dstOff);
            }
        }

        // ---- (8,1) i64/u64 -> i8/u8 : 32-wide (8x4 gather -> 3-level Narrow) ----
        private static unsafe void CastInt64ToByteStrided(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            if (ndim == 0) { dst[0] = (byte)*(long*)src; return; }
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer], ds = dstStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            bool gather = ds == 1 && ss != 1;
            var idx = Vector256.Create(0L, ss, 2 * ss, 3 * ss); long g = 4L * ss;
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                long* p = (long*)(src + srcOff * 8); byte* dr = dst + dstOff; long i = 0;
                if (ds == 1 && ss == 1)
                {
                    for (; i + 16 <= innerN; i += 16)
                    {
                        var a = Vector256.Load(p + i); var b = Vector256.Load(p + i + 4);
                        var c = Vector256.Load(p + i + 8); var e = Vector256.Load(p + i + 12);
                        var w = Vector256.Narrow(Vector256.Narrow(a, b), Vector256.Narrow(c, e));  // 16 i16
                        Vector128.Store(Vector256.Narrow(w, w).AsByte().GetLower(), dr + i);                 // 16 i8
                    }
                }
                else if (gather)
                {
                    long* q = p;
                    for (; i + 16 <= innerN; i += 16)
                    {
                        var a = Avx2.GatherVector256(q, idx, 8); var b = Avx2.GatherVector256(q + g, idx, 8);
                        var c = Avx2.GatherVector256(q + 2 * g, idx, 8); var e = Avx2.GatherVector256(q + 3 * g, idx, 8);
                        var w = Vector256.Narrow(Vector256.Narrow(a, b), Vector256.Narrow(c, e));
                        Vector128.Store(Vector256.Narrow(w, w).AsByte().GetLower(), dr + i);
                        q += 4 * g;
                    }
                    for (; i + 4 <= innerN; i += 4) { var v = Avx2.GatherVector256(q, idx, 8); for (int z = 0; z < 4; z++) dr[i + z] = (byte)v.GetElement(z); q += g; }
                }
                for (; i < innerN; i++) dr[i * ds] = (byte)*(long*)(src + (srcOff + i * ss) * 8);
                AdvanceOdometer(coord, srcStrides, dstStrides, shape, outer, ref srcOff, ref dstOff);
            }
        }

        // ---- (8,2) i64/u64 -> i16/u16/char : 16-wide (4x4 gather -> 2-level Narrow) ----
        private static unsafe void CastInt64ToShortStrided(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            if (ndim == 0) { *(short*)dst = (short)*(long*)src; return; }
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer], ds = dstStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            bool gather = ds == 1 && ss != 1;
            var idx = Vector256.Create(0L, ss, 2 * ss, 3 * ss); long g = 4L * ss;
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                long* p = (long*)(src + srcOff * 8); short* dr = (short*)(dst + dstOff * 2); long i = 0;
                if (ds == 1 && ss == 1)
                {
                    for (; i + 16 <= innerN; i += 16)
                    {
                        var a = Vector256.Load(p + i); var b = Vector256.Load(p + i + 4);
                        var c = Vector256.Load(p + i + 8); var e = Vector256.Load(p + i + 12);
                        Vector256.Store(Vector256.Narrow(Vector256.Narrow(a, b), Vector256.Narrow(c, e)), dr + i);
                    }
                }
                else if (gather)
                {
                    long* q = p;
                    for (; i + 16 <= innerN; i += 16)
                    {
                        var a = Avx2.GatherVector256(q, idx, 8); var b = Avx2.GatherVector256(q + g, idx, 8);
                        var c = Avx2.GatherVector256(q + 2 * g, idx, 8); var e = Avx2.GatherVector256(q + 3 * g, idx, 8);
                        Vector256.Store(Vector256.Narrow(Vector256.Narrow(a, b), Vector256.Narrow(c, e)), dr + i);
                        q += 4 * g;
                    }
                    for (; i + 4 <= innerN; i += 4) { var v = Avx2.GatherVector256(q, idx, 8); for (int z = 0; z < 4; z++) dr[i + z] = (short)v.GetElement(z); q += g; }
                }
                for (; i < innerN; i++) *(short*)(dst + (dstOff + i * ds) * 2) = (short)*(long*)(src + (srcOff + i * ss) * 8);
                AdvanceOdometer(coord, srcStrides, dstStrides, shape, outer, ref srcOff, ref dstOff);
            }
        }

        // ---- (8,4) i64/u64 -> i32/u32 : 8-wide (2x4 gather -> 1-level Narrow) ----
        private static unsafe void CastInt64ToIntStrided(void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV; byte* dst = (byte*)dstV;
            if (ndim == 0) { *(int*)dst = (int)*(long*)src; return; }
            int outer = ndim - 1; long innerN = shape[outer], ss = srcStrides[outer], ds = dstStrides[outer];
            long outerCount = 1; for (int a = 0; a < outer; a++) outerCount *= shape[a];
            long* coord = stackalloc long[ndim]; for (int a = 0; a < ndim; a++) coord[a] = 0;
            bool gather = ds == 1 && ss != 1;
            var idx = Vector256.Create(0L, ss, 2 * ss, 3 * ss); long g = 4L * ss;
            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                long* p = (long*)(src + srcOff * 8); int* dr = (int*)(dst + dstOff * 4); long i = 0;
                if (ds == 1 && ss == 1)
                {
                    for (; i + 8 <= innerN; i += 8)
                        Vector256.Store(Vector256.Narrow(Vector256.Load(p + i), Vector256.Load(p + i + 4)), dr + i);
                }
                else if (gather)
                {
                    long* q = p;
                    for (; i + 8 <= innerN; i += 8)
                    {
                        Vector256.Store(Vector256.Narrow(Avx2.GatherVector256(q, idx, 8), Avx2.GatherVector256(q + g, idx, 8)), dr + i);
                        q += 2 * g;
                    }
                    for (; i + 4 <= innerN; i += 4) { var v = Avx2.GatherVector256(q, idx, 8); for (int z = 0; z < 4; z++) dr[i + z] = (int)v.GetElement(z); q += g; }
                }
                for (; i < innerN; i++) *(int*)(dst + (dstOff + i * ds) * 4) = (int)*(long*)(src + (srcOff + i * ss) * 8);
                AdvanceOdometer(coord, srcStrides, dstStrides, shape, outer, ref srcOff, ref dstOff);
            }
        }

        // Shared outer-dim incremental-offset odometer (element strides).
        private static unsafe void AdvanceOdometer(long* coord, long* srcStrides, long* dstStrides, long* shape, int outer, ref long srcOff, ref long dstOff)
        {
            for (int ax = outer - 1; ax >= 0; ax--)
            {
                coord[ax]++; srcOff += srcStrides[ax]; dstOff += dstStrides[ax];
                if (coord[ax] < shape[ax]) break;
                coord[ax] = 0; srcOff -= srcStrides[ax] * shape[ax]; dstOff -= dstStrides[ax] * shape[ax];
            }
        }
    }
}
