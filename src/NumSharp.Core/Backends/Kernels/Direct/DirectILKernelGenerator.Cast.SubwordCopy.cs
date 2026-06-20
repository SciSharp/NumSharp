using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.SubwordCopy.cs
        //   Same-type 1-byte / 2-byte STRIDED copy via SIMD lane shuffles.
        //
        //   A same-type cast is pure byte movement (signed/unsigned/float-ness is
        //   irrelevant — the bits are copied verbatim), so ONE size-parameterised
        //   kernel covers every sub-word dtype: 1B {bool,u8,i8}, 2B {i16,u16,char,f16}.
        //
        //   For a strided view the generic MemoryCopy-strategy strided kernel detects
        //   a unit-stride inner axis (Buffer.MemoryCopy per row) but falls to a SCALAR
        //   per-element inner loop for any non-unit inner stride — the documented
        //   sub-word `strided`/`negcol` cliff (i8|strided 0.53x, char|strided 0.59x,
        //   i8|negcol 0.71x). 2-byte sources are not VPGATHER-eligible, so the float
        //   kernels' gather/stage tricks don't apply.
        //
        //   This kernel specialises the two strided layouts that dominate astype:
        //     * inner stride +2  ([:, ::2]) -> DEINTERLEAVE the even lanes:
        //         mask the odd half to 0, VPACKUSWB/VPACKUSDW down to the wanted width,
        //         VPERMQ to undo the pack's 128-bit-lane interleave. 32 (1B) / 16 (2B)
        //         elements per iter.
        //     * inner stride -1  ([:, ::-1]) -> REVERSE: load forward, VPSHUFB the bytes
        //         within each 128-bit lane, VPERMQ to swap the lanes. (Generalises the
        //         double->int32 ss==-1 precedent in InnerCastDoubleToInt32 to 1B/2B.)
        //     * inner stride +1 stays a per-row Buffer.MemoryCopy; everything else is the
        //       NumPy-faithful scalar tail (also the SIMD remainder).
        //
        //   PERF NOTE (measured): the inner SIMD loop MUST be inlined in the outer
        //   odometer body. Factoring it into a per-row helper method costs ~6x (0.020 ->
        //   0.125 ms for stride-2 1B); the JIT will not inline a method containing a loop
        //   even with AggressiveInlining, and the call boundary kills the loop's codegen.
        //   So the odometer is written once per element-size with the four inner cases
        //   inlined; the ss/ds branch is per-row but perfectly predictable (constant).
        //
        //   Measured standalone (1000x1000 src, view to 1000x500 / 1000x1000, warm dst,
        //   best-of-7) vs the scalar inner loop it replaces and NumPy:
        //     stride-2 1B  0.187 -> 0.020 ms (NumPy 0.093)
        //     stride-2 2B  0.206 -> 0.040 ms (NumPy 0.107)
        //     reverse  1B  0.370 -> 0.031 ms (NumPy 0.221)
        //   (benchmark/poc/subword_strided_poc.cs, subword_structure_poc.cs).
        // =====================================================================

        // Hoisted shuffle constants (loaded once before the odometer, not per row).
        private static readonly Vector256<short> _evenByteMask = Vector256.Create((short)0x00FF);
        private static readonly Vector256<int> _evenWordMask = Vector256.Create(0x0000FFFF);
        private static readonly Vector256<byte> _revBytes = Vector256.Create(
            (byte)15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                  15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        private static readonly Vector256<byte> _revWords = Vector256.Create(
            (byte)14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1,
                  14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);

        /// <summary>
        /// Strided <see cref="StridedCastKernel"/> for a SAME-TYPE 1-byte / 2-byte copy
        /// (the sub-word <c>strided</c>/<c>negcol</c> cliff), or null. Cross-type and
        /// 4/8/16-byte same-type copies return null and keep their existing fast paths
        /// (4/8-byte are VPGATHER-eligible and already win; 16-byte dec/c128 already win).
        /// </summary>
        internal static unsafe StridedCastKernel TryGetSubwordCopyStridedKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (!Avx2.IsSupported) return null;
            if (srcType != dstType) return null;
            switch (GetTypeSize(srcType))
            {
                case 1: return SubwordCopyStrided1B;
                case 2: return SubwordCopyStrided2B;
                default: return null;
            }
        }

        // Outer-dim odometer (innermost-first incremental offset, element strides) — same
        // shape as InnerCastDoubleToInt32's driver, but the inner SIMD loop is INLINED in
        // the body (see PERF NOTE above). ss/ds dispatch is per-row, predictable.
        private static unsafe void SubwordCopyStrided1B(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV;
            byte* dst = (byte*)dstV;
            if (ndim == 0) { dst[0] = src[0]; return; }

            int outer = ndim - 1;
            long innerN = shape[outer];
            long ss = srcStrides[outer];
            long ds = dstStrides[outer];

            long outerCount = 1;
            for (int a = 0; a < outer; a++) outerCount *= shape[a];

            long* coord = stackalloc long[ndim];
            for (int a = 0; a < ndim; a++) coord[a] = 0;

            var mask = _evenByteMask;
            var rev = _revBytes;

            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                byte* s = src + srcOff;
                byte* d = dst + dstOff;
                long i = 0;
                if (ds == 1)
                {
                    if (ss == 1) { Buffer.MemoryCopy(s, d, innerN, innerN); i = innerN; }
                    else if (ss == 2)
                    {
                        // d[i] = s[2i]: read contiguous shorts (oddByte<<8|evenByte), mask odd to 0,
                        // VPACKUSWB the low bytes, VPERMQ to undo the pack's lane interleave.
                        for (; i + 32 <= innerN; i += 32)
                        {
                            var v0 = Vector256.Load((short*)(s + 2 * i));
                            var v1 = Vector256.Load((short*)(s + 2 * i + 32));
                            var p = Avx2.PackUnsignedSaturate(Avx2.And(v0, mask), Avx2.And(v1, mask));
                            Vector256.Store(Avx2.Permute4x64(p.AsInt64(), 0xD8).AsByte(), d + i);
                        }
                    }
                    else if (ss == -1)
                    {
                        // s[i] lives at memory s[-i]; load forward [s-i-31 .. s-i], reverse 32 bytes.
                        for (; i + 32 <= innerN; i += 32)
                        {
                            var v = Vector256.Load(s - i - 31);
                            Vector256.Store(Avx2.Permute4x64(Avx2.Shuffle(v, rev).AsInt64(), 0x4E).AsByte(), d + i);
                        }
                    }
                }
                for (; i < innerN; i++) d[i * ds] = s[i * ss];

                for (int ax = outer - 1; ax >= 0; ax--)
                {
                    coord[ax]++; srcOff += srcStrides[ax]; dstOff += dstStrides[ax];
                    if (coord[ax] < shape[ax]) break;
                    coord[ax] = 0; srcOff -= srcStrides[ax] * shape[ax]; dstOff -= dstStrides[ax] * shape[ax];
                }
            }
        }

        private static unsafe void SubwordCopyStrided2B(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            short* src = (short*)srcV;
            short* dst = (short*)dstV;
            if (ndim == 0) { dst[0] = src[0]; return; }

            int outer = ndim - 1;
            long innerN = shape[outer];
            long ss = srcStrides[outer];
            long ds = dstStrides[outer];

            long outerCount = 1;
            for (int a = 0; a < outer; a++) outerCount *= shape[a];

            long* coord = stackalloc long[ndim];
            for (int a = 0; a < ndim; a++) coord[a] = 0;

            var mask = _evenWordMask;
            var rev = _revWords;

            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                short* s = src + srcOff;
                short* d = dst + dstOff;
                long i = 0;
                if (ds == 1)
                {
                    if (ss == 1) { Buffer.MemoryCopy(s, d, innerN * 2, innerN * 2); i = innerN; }
                    else if (ss == 2)
                    {
                        // d[i] = s[2i]: read contiguous ints (oddShort<<16|evenShort), mask odd to 0,
                        // VPACKUSDW the low shorts, VPERMQ to undo the pack's lane interleave.
                        for (; i + 16 <= innerN; i += 16)
                        {
                            var v0 = Vector256.Load((int*)(s + 2 * i));
                            var v1 = Vector256.Load((int*)(s + 2 * i + 16));
                            var p = Avx2.PackUnsignedSaturate(Avx2.And(v0, mask), Avx2.And(v1, mask));
                            Vector256.Store(Avx2.Permute4x64(p.AsInt64(), 0xD8).AsInt16(), d + i);
                        }
                    }
                    else if (ss == -1)
                    {
                        // s[i] lives at memory s[-i]; load forward [s-i-15 .. s-i], reverse 16 shorts.
                        for (; i + 16 <= innerN; i += 16)
                        {
                            var v = Vector256.Load(s - i - 15);
                            var sh = Avx2.Shuffle(v.AsByte(), rev).AsInt16();
                            Vector256.Store(Avx2.Permute4x64(sh.AsInt64(), 0x4E).AsInt16(), d + i);
                        }
                    }
                }
                for (; i < innerN; i++) d[i * ds] = s[i * ss];

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
