using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.SubwordWiden.cs
        //   1-byte integer -> 2-byte STRIDED cast: {bool,u8,i8} -> {i16,u16,char}.
        //   The inverse of SubwordNarrow — and the same SIMD shape.
        //
        //   1-byte sources are not VPGATHER-eligible, so the generic WidenInt IL kernel
        //   falls to a scalar inner loop on a non-unit inner stride (the i8|strided|i16
        //   0.90 / bool|strided|i16 0.55 / u8|strided|char 0.62 cliff).
        //
        //   Two-stage SIMD: deinterleave (ss==2) / reverse (ss==-1) the wanted 1-byte
        //   elements (the SubwordCopy shuffles), then widen to 2 bytes:
        //     signed src (i8)        -> Vector256.WidenLower/Upper on sbyte = SIGN-extend
        //     unsigned src (u8,bool) -> Vector256.WidenLower/Upper on byte  = ZERO-extend
        //   The dst signedness (i16/u16/char) is irrelevant — the 16-bit pattern is the
        //   same (i8 -56 -> 0xFFC8 reads as int16 -56 / uint16 65480, == NumPy). Inner
        //   ss==1 rows (sliced/negrow) run a contiguous Vector256.Widen; ds!=1 / tail run
        //   the scalar widening cast.
        //
        //   Only C-NON-contiguous layouts reach here; C-contig goes through TryGetCastKernel.
        //   Excluded: dst==Half (bool/u8/i8 -> f16 is int->float, a value cast on the Half
        //   kernels), and X->bool / same-size (handled by SubwordCopy / SubwordNarrow).
        //
        //   Measured standalone (1000x1000 i8 src, warm dst, best-of-7) vs scalar + NumPy:
        //     strided i8->i16  0.157 -> 0.024 ms (NumPy ~0.115)
        //     negcol  i8->i16  0.367 -> 0.055 ms (NumPy ~0.24)
        // =====================================================================

        /// <summary>
        /// Strided <see cref="StridedCastKernel"/> for a 1-byte-int -&gt; 2-byte-int widen
        /// ({bool,u8,i8} -&gt; {i16,u16,char}), or null. dst==Half is excluded (int-&gt;float
        /// value cast). The src signedness selects sign- vs zero-extend.
        /// </summary>
        internal static unsafe StridedCastKernel TryGetSubwordWidenStridedKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (!Avx2.IsSupported) return null;
            if (dstType != NPTypeCode.Int16 && dstType != NPTypeCode.UInt16 && dstType != NPTypeCode.Char)
                return null;
            switch (srcType)
            {
                case NPTypeCode.SByte: return SubwordWidenSigned1to2;
                case NPTypeCode.Byte:
                case NPTypeCode.Boolean: return SubwordWidenUnsigned1to2;
                default: return null;
            }
        }

        // i8 -> {i16,u16,char}: SIGN-extend. Odometer + inline inner (per the SubwordCopy PERF NOTE).
        private static unsafe void SubwordWidenSigned1to2(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            sbyte* src = (sbyte*)srcV;
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

            var mask = _evenByteMask;
            var rev = _revBytes;

            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                sbyte* s = src + srcOff;
                short* d = dst + dstOff;
                long i = 0;
                if (ds == 1)
                {
                    if (ss == 1)
                    {
                        for (; i + 32 <= innerN; i += 32)
                        {
                            var v = Vector256.Load(s + i);
                            Vector256.Store(Vector256.WidenLower(v), d + i);
                            Vector256.Store(Vector256.WidenUpper(v), d + i + 16);
                        }
                    }
                    else if (ss == 2)
                    {
                        for (; i + 32 <= innerN; i += 32)
                        {
                            var v0 = Vector256.Load((short*)(s + 2 * i));
                            var v1 = Vector256.Load((short*)(s + 2 * i + 32));
                            var ev = Avx2.Permute4x64(Avx2.PackUnsignedSaturate(Avx2.And(v0, mask), Avx2.And(v1, mask)).AsInt64(), 0xD8).AsSByte();
                            Vector256.Store(Vector256.WidenLower(ev), d + i);
                            Vector256.Store(Vector256.WidenUpper(ev), d + i + 16);
                        }
                    }
                    else if (ss == -1)
                    {
                        for (; i + 32 <= innerN; i += 32)
                        {
                            var v = Vector256.Load((byte*)(s - i - 31));
                            var rv = Avx2.Permute4x64(Avx2.Shuffle(v, rev).AsInt64(), 0x4E).AsSByte();
                            Vector256.Store(Vector256.WidenLower(rv), d + i);
                            Vector256.Store(Vector256.WidenUpper(rv), d + i + 16);
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

        // {bool,u8} -> {i16,u16,char}: ZERO-extend.
        private static unsafe void SubwordWidenUnsigned1to2(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            byte* src = (byte*)srcV;
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

            var mask = _evenByteMask;
            var rev = _revBytes;

            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                byte* s = src + srcOff;
                short* d = dst + dstOff;
                long i = 0;
                if (ds == 1)
                {
                    if (ss == 1)
                    {
                        for (; i + 32 <= innerN; i += 32)
                        {
                            var v = Vector256.Load(s + i);
                            Vector256.Store(Vector256.WidenLower(v).AsInt16(), d + i);
                            Vector256.Store(Vector256.WidenUpper(v).AsInt16(), d + i + 16);
                        }
                    }
                    else if (ss == 2)
                    {
                        for (; i + 32 <= innerN; i += 32)
                        {
                            var v0 = Vector256.Load((short*)(s + 2 * i));
                            var v1 = Vector256.Load((short*)(s + 2 * i + 32));
                            var ev = Avx2.Permute4x64(Avx2.PackUnsignedSaturate(Avx2.And(v0, mask), Avx2.And(v1, mask)).AsInt64(), 0xD8).AsByte();
                            Vector256.Store(Vector256.WidenLower(ev).AsInt16(), d + i);
                            Vector256.Store(Vector256.WidenUpper(ev).AsInt16(), d + i + 16);
                        }
                    }
                    else if (ss == -1)
                    {
                        for (; i + 32 <= innerN; i += 32)
                        {
                            var v = Vector256.Load(s - i - 31);
                            var rv = Avx2.Permute4x64(Avx2.Shuffle(v, rev).AsInt64(), 0x4E).AsByte();
                            Vector256.Store(Vector256.WidenLower(rv).AsInt16(), d + i);
                            Vector256.Store(Vector256.WidenUpper(rv).AsInt16(), d + i + 16);
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
