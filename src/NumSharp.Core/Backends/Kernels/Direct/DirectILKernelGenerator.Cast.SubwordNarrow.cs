using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.SubwordNarrow.cs
        //   2-byte integer -> 1-byte STRIDED cast: {i16,u16,char} -> {i8,u8} (low-byte
        //   truncate) and {i16,u16,char} -> bool (!=0). The cross-WIDTH continuation of
        //   SubwordCopy.
        //
        //   2-byte sources are not VPGATHER-eligible, so TryGetIntToNarrowStridedKernel
        //   (which gathers 4/8-byte sources) declines them and the generic NarrowInt IL
        //   kernel falls to a scalar inner loop on a non-unit inner stride — the
        //   i16|strided|i8 0.76 / i16|negcol|i8 0.89 / i16|strided|bool 0.65 cliff.
        //
        //   Two-stage SIMD: deinterleave (ss==2) / reverse (ss==-1) the wanted 2-byte
        //   elements (the SubwordCopy shuffles), then narrow to 1 byte:
        //     int  : mask low byte, VPACKUSWB -> low bytes (== unchecked (byte) == NumPy wrap)
        //     bool : (v != 0) ? 1 : 0  via AndNot(CompareEqual(v,0), one), then pack
        //   Inner ss==1 rows (sliced/negrow) run the contiguous Vector256.Narrow the
        //   generic kernel already wins with; ds!=1 / tail run the scalar cast.
        //
        //   Only the C-NON-contiguous layouts reach here (TryGetStridedCastKernel); the
        //   C-contiguous cast goes through TryGetCastKernel and is untouched.
        //
        //   Measured standalone (1000x1000 i16 src, warm dst, best-of-7) vs scalar + NumPy:
        //     strided i16->i8  0.156 -> 0.104 ms (NumPy 0.107; memory-bound, ~parity)
        //     negcol  i16->i8  0.369 -> 0.125 ms (NumPy ~0.24; ~1.9x)
        //     strided i16->bool 0.214 -> 0.105 ms (NumPy 0.143; ~1.4x)
        //   (benchmark/poc/subword_narrow_poc.cs).
        // =====================================================================

        private static readonly Vector256<short> _lowByteMask = Vector256.Create((short)0x00FF);
        private static readonly Vector256<short> _shortOne = Vector256.Create((short)1);

        /// <summary>
        /// Strided <see cref="StridedCastKernel"/> for a 2-byte-int -&gt; 1-byte cast NumPy
        /// performs as low-byte truncate (i16/u16/char -&gt; i8/u8) or a !=0 test
        /// (i16/u16/char -&gt; bool), or null. f16 source is excluded (float-&gt;int is a value
        /// truncate, not a low-byte narrow — kept on the Half kernels).
        /// </summary>
        internal static unsafe StridedCastKernel TryGetSubwordNarrowStridedKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (!Avx2.IsSupported) return null;
            // src must be a 2-byte INTEGER (i16/u16/char); Half is a value cast.
            if (srcType != NPTypeCode.Int16 && srcType != NPTypeCode.UInt16 && srcType != NPTypeCode.Char)
                return null;
            if (dstType == NPTypeCode.SByte || dstType == NPTypeCode.Byte) return SubwordNarrowInt2to1;
            if (dstType == NPTypeCode.Boolean) return SubwordNarrowBool2to1;
            return null;
        }

        // {i16,u16,char} -> {i8,u8}: low-byte truncate. Odometer with inline inner (see the
        // SubwordCopy PERF NOTE — a per-row helper costs ~6x).
        private static unsafe void SubwordNarrowInt2to1(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            short* src = (short*)srcV;
            byte* dst = (byte*)dstV;
            if (ndim == 0) { dst[0] = (byte)src[0]; return; }

            int outer = ndim - 1;
            long innerN = shape[outer];
            long ss = srcStrides[outer];
            long ds = dstStrides[outer];

            long outerCount = 1;
            for (int a = 0; a < outer; a++) outerCount *= shape[a];

            long* coord = stackalloc long[ndim];
            for (int a = 0; a < ndim; a++) coord[a] = 0;

            var wmask = _evenWordMask;
            var bmask = _lowByteMask;
            var revw = _revWords;

            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                short* s = src + srcOff;
                byte* d = dst + dstOff;
                long i = 0;
                if (ds == 1)
                {
                    if (ss == 1)
                    {
                        // contiguous low-byte narrow (sliced/negrow inner): 32/iter via truncating Narrow.
                        for (; i + 32 <= innerN; i += 32)
                            Vector256.Store(Vector256.Narrow(Vector256.Load(s + i), Vector256.Load(s + i + 16)).AsByte(), d + i);
                    }
                    else if (ss == 2)
                    {
                        // deinterleave even shorts (stage1), then low byte (stage2). 16/iter.
                        for (; i + 16 <= innerN; i += 16)
                        {
                            var v0 = Vector256.Load((int*)(s + 2 * i));
                            var v1 = Vector256.Load((int*)(s + 2 * i + 16));
                            var evens = Avx2.Permute4x64(Avx2.PackUnsignedSaturate(Avx2.And(v0, _evenWordMask), Avx2.And(v1, _evenWordMask)).AsInt64(), 0xD8).AsInt16();
                            var lo = Avx2.And(evens, bmask);
                            Vector128.Store(Avx2.Permute4x64(Avx2.PackUnsignedSaturate(lo, lo).AsInt64(), 0xD8).GetLower().AsByte(), d + i);
                        }
                    }
                    else if (ss == -1)
                    {
                        // reverse shorts, then low byte. 16/iter.
                        for (; i + 16 <= innerN; i += 16)
                        {
                            var v = Vector256.Load(s - i - 15);
                            var rev = Avx2.Permute4x64(Avx2.Shuffle(v.AsByte(), revw).AsInt64(), 0x4E).AsInt16();
                            var lo = Avx2.And(rev, bmask);
                            Vector128.Store(Avx2.Permute4x64(Avx2.PackUnsignedSaturate(lo, lo).AsInt64(), 0xD8).GetLower().AsByte(), d + i);
                        }
                    }
                }
                for (; i < innerN; i++) d[i * ds] = (byte)s[i * ss];

                for (int ax = outer - 1; ax >= 0; ax--)
                {
                    coord[ax]++; srcOff += srcStrides[ax]; dstOff += dstStrides[ax];
                    if (coord[ax] < shape[ax]) break;
                    coord[ax] = 0; srcOff -= srcStrides[ax] * shape[ax]; dstOff -= dstStrides[ax] * shape[ax];
                }
            }
        }

        // {i16,u16,char} -> bool: (v != 0) ? 1 : 0.
        private static unsafe void SubwordNarrowBool2to1(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            short* src = (short*)srcV;
            byte* dst = (byte*)dstV;
            if (ndim == 0) { dst[0] = (byte)(src[0] != 0 ? 1 : 0); return; }

            int outer = ndim - 1;
            long innerN = shape[outer];
            long ss = srcStrides[outer];
            long ds = dstStrides[outer];

            long outerCount = 1;
            for (int a = 0; a < outer; a++) outerCount *= shape[a];

            long* coord = stackalloc long[ndim];
            for (int a = 0; a < ndim; a++) coord[a] = 0;

            var one = _shortOne;
            var zero = Vector256<short>.Zero;
            var revw = _revWords;

            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                short* s = src + srcOff;
                byte* d = dst + dstOff;
                long i = 0;
                if (ds == 1)
                {
                    if (ss == 1)
                    {
                        for (; i + 32 <= innerN; i += 32)
                        {
                            var a0 = Vector256.Load(s + i); var b0 = Vector256.Load(s + i + 16);
                            var na = Avx2.AndNot(Avx2.CompareEqual(a0, zero), one);
                            var nb = Avx2.AndNot(Avx2.CompareEqual(b0, zero), one);
                            Vector256.Store(Vector256.Narrow(na, nb).AsByte(), d + i);
                        }
                    }
                    else if (ss == 2)
                    {
                        for (; i + 16 <= innerN; i += 16)
                        {
                            var v0 = Vector256.Load((int*)(s + 2 * i));
                            var v1 = Vector256.Load((int*)(s + 2 * i + 16));
                            var evens = Avx2.Permute4x64(Avx2.PackUnsignedSaturate(Avx2.And(v0, _evenWordMask), Avx2.And(v1, _evenWordMask)).AsInt64(), 0xD8).AsInt16();
                            var nz = Avx2.AndNot(Avx2.CompareEqual(evens, zero), one);
                            Vector128.Store(Avx2.Permute4x64(Avx2.PackUnsignedSaturate(nz, nz).AsInt64(), 0xD8).GetLower().AsByte(), d + i);
                        }
                    }
                    else if (ss == -1)
                    {
                        for (; i + 16 <= innerN; i += 16)
                        {
                            var v = Vector256.Load(s - i - 15);
                            var rev = Avx2.Permute4x64(Avx2.Shuffle(v.AsByte(), revw).AsInt64(), 0x4E).AsInt16();
                            var nz = Avx2.AndNot(Avx2.CompareEqual(rev, zero), one);
                            Vector128.Store(Avx2.Permute4x64(Avx2.PackUnsignedSaturate(nz, nz).AsInt64(), 0xD8).GetLower().AsByte(), d + i);
                        }
                    }
                }
                for (; i < innerN; i++) d[i * ds] = (byte)(s[i * ss] != 0 ? 1 : 0);

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
