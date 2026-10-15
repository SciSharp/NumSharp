using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // DirectILKernelGenerator.Cast.WordCopy.cs
        //   Same-type 4-byte / 8-byte STRIDED copy via SIMD lane shuffles and AVX2 gathers.
        //
        //   The 4/8-byte sibling of Cast.SubwordCopy.cs. A same-type (or same-size integer
        //   reinterpret) cast is pure bit movement, so one size-parameterised kernel serves every
        //   4-byte dtype {int32, uint32, float32} and every 8-byte one {int64, uint64, float64}. Before
        //   it, a strided 4/8-byte copy reached the generic MemoryCopy-strategy kernel, which copies a
        //   unit-stride inner axis per row but runs any other inner stride one element at a time —
        //   measured 29 µs for a 100K stride-2 float32 copy, slower than a plain scalar loop (20 µs).
        //
        //   Inner-axis cases (the outer axes are an incremental odometer, as in SubwordCopy):
        //     * ss == +1  -> Buffer.MemoryCopy per row;
        //     * ss == -1  -> REVERSE: load forward, VPERMD (4B) / VPERMQ (8B) the lanes back to front;
        //     * ss == +2  -> DEINTERLEAVE: two forward loads, VSHUFPS / VUNPCKLPD the even lanes
        //                    together, VPERMQ to undo the 128-bit-lane interleave. The loop stops one
        //                    vector early so the second load never touches the slot after the last
        //                    element (no over-read at a buffer's end);
        //     * any other |ss| within the gather budget -> VPGATHERDD / VPGATHERDQ with a hoisted
        //                    byte-offset index vector (a stride-0 broadcast gathers one element 8×);
        //     * a strided DESTINATION, or a stride past the gather budget -> the scalar loop.
        //   Every case moves the bits verbatim, so the copy is exact for every value, NaN payloads and
        //   signalling NaNs included.
        // =====================================================================

        /// <summary>Lane indices that reverse 8 int32 lanes (VPERMD).</summary>
        private static readonly Vector256<int> _revDwords = Vector256.Create(7, 6, 5, 4, 3, 2, 1, 0);

        /// <summary>
        ///     The strided copy kernel for a 4-byte / 8-byte cast that is a pure bit copy, or null: the same dtype,
        ///     or the same-size integer reinterprets (int32↔uint32, int64↔uint64 — NumPy's cast between them keeps
        ///     the bits). A float↔int pair of the same size is a VALUE cast and keeps its own kernels.
        /// </summary>
        /// <param name="srcType">Source dtype.</param>
        /// <param name="dstType">Destination dtype.</param>
        /// <returns>The kernel, or null (not a 4/8-byte bit copy, or no AVX2).</returns>
        internal static unsafe StridedCastKernel TryGetWordCopyStridedKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (!Avx2.IsSupported)
                return null;
            int sz = GetTypeSize(srcType);
            if ((sz != 4 && sz != 8) || GetTypeSize(dstType) != sz)
                return null;
            if (srcType != dstType)
            {
                // Only integer <-> integer reinterprets are bit copies; anything touching a float is a value cast.
                bool srcInt = srcType is NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64;
                bool dstInt = dstType is NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64;
                if (!srcInt || !dstInt)
                    return null;
            }
            // Only the storage dtypes of those widths reach here (Complex/Decimal are 16 bytes, Char 2).
            return sz == 4 ? WordCopyStrided4B : WordCopyStrided8B;
        }

        /// <summary>Strided copy of 4-byte elements (see the file header for the inner-axis cases).</summary>
        /// <param name="srcV">Source element 0.</param>
        /// <param name="dstV">Destination element 0.</param>
        /// <param name="srcStrides">Source strides in elements (length <paramref name="ndim"/>).</param>
        /// <param name="dstStrides">Destination strides in elements.</param>
        /// <param name="shape">Shape (length <paramref name="ndim"/>).</param>
        /// <param name="ndim">Dimension count (0 copies one element).</param>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        private static unsafe void WordCopyStrided4B(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            int* src = (int*)srcV;
            int* dst = (int*)dstV;
            if (ndim == 0) { dst[0] = src[0]; return; }

            int outer = ndim - 1;
            long innerN = shape[outer];
            long ss = srcStrides[outer];
            long ds = dstStrides[outer];

            long outerCount = 1;
            for (int a = 0; a < outer; a++) outerCount *= shape[a];

            long* coord = stackalloc long[ndim];
            for (int a = 0; a < ndim; a++) coord[a] = 0;

            var rev = _revDwords;
            // Gather indices are BYTE offsets in int32 lanes: the largest is 7·|ss|·4.
            bool gatherable = ss != 1 && ss != -1 && ss != 2 && ss >= int.MinValue / 32 && ss <= int.MaxValue / 32;
            int sb = (int)(gatherable ? ss * 4 : 0);
            var gidx = Vector256.Create(0, sb, 2 * sb, 3 * sb, 4 * sb, 5 * sb, 6 * sb, 7 * sb);

            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                int* s = src + srcOff;
                int* d = dst + dstOff;
                long i = 0;
                if (ds == 1)
                {
                    if (ss == 1) { Buffer.MemoryCopy(s, d, innerN * 4, innerN * 4); i = innerN; }
                    else if (ss == -1)
                    {
                        // s[i] lives at memory s[-i]: load the 8 elements ending there forward, reverse the lanes.
                        for (; i + 8 <= innerN; i += 8)
                            Vector256.Store(Avx2.PermuteVar8x32(Vector256.Load(s - i - 7), rev), d + i);
                    }
                    else if (ss == 2)
                    {
                        // d[i..i+7] = s[2i], s[2i+2], …: shufps picks the even lanes of the two 8-lane loads per
                        // 128-bit half (a0 a2 b0 b2 | a4 a6 b4 b6), VPERMQ 0xD8 restores the order. i + 9 <= innerN
                        // keeps the second load's last slot (2i+15) inside the view (its last element is 2·innerN-2).
                        for (; i + 9 <= innerN; i += 8)
                        {
                            var a = Vector256.Load((float*)(s + 2 * i));
                            var b = Vector256.Load((float*)(s + 2 * i + 8));
                            var even = Avx.Shuffle(a, b, 0b10_00_10_00);
                            Vector256.Store(Avx2.Permute4x64(even.AsDouble(), 0xD8).AsInt32(), d + i);
                        }
                    }
                    else if (gatherable)
                    {
                        for (; i + 8 <= innerN; i += 8)
                            Vector256.Store(Avx2.GatherVector256(s + i * ss, gidx, 1), d + i);
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

        /// <summary>Strided copy of 8-byte elements (see the file header for the inner-axis cases).</summary>
        /// <param name="srcV">Source element 0.</param>
        /// <param name="dstV">Destination element 0.</param>
        /// <param name="srcStrides">Source strides in elements (length <paramref name="ndim"/>).</param>
        /// <param name="dstStrides">Destination strides in elements.</param>
        /// <param name="shape">Shape (length <paramref name="ndim"/>).</param>
        /// <param name="ndim">Dimension count (0 copies one element).</param>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        private static unsafe void WordCopyStrided8B(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            long* src = (long*)srcV;
            long* dst = (long*)dstV;
            if (ndim == 0) { dst[0] = src[0]; return; }

            int outer = ndim - 1;
            long innerN = shape[outer];
            long ss = srcStrides[outer];
            long ds = dstStrides[outer];

            long outerCount = 1;
            for (int a = 0; a < outer; a++) outerCount *= shape[a];

            long* coord = stackalloc long[ndim];
            for (int a = 0; a < ndim; a++) coord[a] = 0;

            // Gather indices are BYTE offsets in int32 lanes: the largest is 3·|ss|·8.
            bool gatherable = ss != 1 && ss != -1 && ss != 2 && ss >= int.MinValue / 32 && ss <= int.MaxValue / 32;
            int sb = (int)(gatherable ? ss * 8 : 0);
            var gidx = Vector128.Create(0, sb, 2 * sb, 3 * sb);

            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                long* s = src + srcOff;
                long* d = dst + dstOff;
                long i = 0;
                if (ds == 1)
                {
                    if (ss == 1) { Buffer.MemoryCopy(s, d, innerN * 8, innerN * 8); i = innerN; }
                    else if (ss == -1)
                    {
                        // Load the 4 elements ending at s[-i] forward, VPERMQ 0x1B reverses them.
                        for (; i + 4 <= innerN; i += 4)
                            Vector256.Store(Avx2.Permute4x64(Vector256.Load(s - i - 3), 0x1B), d + i);
                    }
                    else if (ss == 2)
                    {
                        // unpcklpd pairs the even lanes per 128-bit half (a0 b0 | a2 b2), VPERMQ 0xD8 orders them
                        // a0 a2 b0 b2. i + 5 <= innerN keeps the second load's last slot (2i+7) inside the view.
                        for (; i + 5 <= innerN; i += 4)
                        {
                            var a = Vector256.Load((double*)(s + 2 * i));
                            var b = Vector256.Load((double*)(s + 2 * i + 4));
                            var even = Avx.UnpackLow(a, b);
                            Vector256.Store(Avx2.Permute4x64(even, 0xD8).AsInt64(), d + i);
                        }
                    }
                    else if (gatherable)
                    {
                        for (; i + 4 <= innerN; i += 4)
                            Vector256.Store(Avx2.GatherVector256(s + i * ss, gidx, 1), d + i);
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
