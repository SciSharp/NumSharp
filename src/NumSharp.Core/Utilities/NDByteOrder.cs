using System;
using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     The byte-order swap of runs of contiguous items: NumPy's <c>_strided_byte_swap</c> for the contiguous case, with
    ///     SIMD. A run is reversed in groups of one "unit" (a number's whole item, each half of a complex number, a
    ///     <c>U</c> element's 4-byte code units), so one kernel serves every dtype: the swap depends only on the unit's
    ///     byte width, never on what the bytes mean.
    /// </summary>
    /// <remarks>
    ///     The two users are the class slot <see cref="DTypeMeta.copyswapn"/> (NumPy's <c>PyArray_ArrFuncs.copyswapn</c>,
    ///     whose contiguous runs it swaps) and <see cref="NDArray.byteswap(bool)"/>. Strided elements are reversed one unit
    ///     at a time (<see cref="ReverseStrided"/>): only a contiguous run has whole vectors to shuffle.
    /// </remarks>
    internal static unsafe class NDByteOrder
    {
        // Per-128-bit-lane VPSHUFB masks: result[j] = src[mask[j]] reverses each `unit`-byte group. The 16-byte pattern is
        // repeated across both lanes (an AVX2 shuffle is lane-local and every unit <= 16 fits inside one lane), so the same
        // mask drives the SSSE3 128-bit path through GetLower().
        private static readonly Vector256<byte> Rev2 = Vector256.Create(
            (byte)1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
                   1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
        private static readonly Vector256<byte> Rev4 = Vector256.Create(
            (byte)3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                   3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
        private static readonly Vector256<byte> Rev8 = Vector256.Create(
            (byte)7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
                   7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
        private static readonly Vector256<byte> Rev16 = Vector256.Create(
            (byte)15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                   15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        /// <summary>
        ///     Above this size a fused copy-and-swap streams its writes non-temporally (below it the result stays cache-hot).
        ///     A fresh destination is not in cache, so an ordinary store pays a read-for-ownership a plain block copy avoids;
        ///     a streaming store avoids it too. Chosen to clear a typical L2, so only writes that leave the cache anyway
        ///     bypass it.
        /// </summary>
        private const long NonTemporalThresholdBytes = 1L << 21;   // 2 MiB

        /// <summary>
        ///     Reverses each <paramref name="unit"/>-byte group across <paramref name="totalBytes"/> contiguous bytes, in
        ///     place. VPSHUFB reverses lane-locally, so one per-lane mask drives AVX2 (32 bytes a step) and SSSE3 (16); a
        ///     scalar swap handles the tail and hosts without either.
        /// </summary>
        /// <param name="p">The first byte of the run.</param>
        /// <param name="totalBytes">The run's length, a multiple of <paramref name="unit"/> (a trailing partial unit is left
        ///     as it is).</param>
        /// <param name="unit">The group size in bytes: 2, 4, 8 or 16 (1 leaves the run as it is).</param>
        /// <remarks>Footgun: <paramref name="unit"/> must divide 16 for the vector steps to keep groups whole; every dtype's
        ///     unit does (2, 4, 8, 16).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void ReverseRun(byte* p, long totalBytes, int unit)
        {
            if (unit <= 1)
                return;
            long i = 0;
            if (Avx2.IsSupported)
            {
                Vector256<byte> m = MaskFor(unit);
                for (; i + 32 <= totalBytes; i += 32)
                    Vector256.Store(Avx2.Shuffle(Vector256.Load(p + i), m), p + i);
            }
            else if (Ssse3.IsSupported)
            {
                Vector128<byte> m = MaskFor(unit).GetLower();
                for (; i + 16 <= totalBytes; i += 16)
                    Sse2.Store(p + i, Ssse3.Shuffle(Vector128.Load(p + i), m));
            }
            for (; i + unit <= totalBytes; i += unit)
                ReverseUnit(p + i, unit);
        }

        /// <summary>
        ///     Fused copy and swap: reads <paramref name="totalBytes"/> from <paramref name="src"/>, reverses each
        ///     <paramref name="unit"/>-byte group and writes them to <paramref name="dst"/> - one pass instead of a copy pass
        ///     and a swap pass. A large, 32-byte-aligned destination is written non-temporally (no read-for-ownership).
        /// </summary>
        /// <param name="src">The first source byte.</param>
        /// <param name="dst">The first destination byte.</param>
        /// <param name="totalBytes">The run's length, a multiple of <paramref name="unit"/>.</param>
        /// <param name="unit">The group size in bytes: 2, 4, 8 or 16.</param>
        /// <remarks>Footgun: the two runs must not overlap (a vector is read before it is written, so only the same address
        ///     on both sides is safe); copy first and <see cref="ReverseRun"/> in place otherwise.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void CopyReverseRun(byte* src, byte* dst, long totalBytes, int unit)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                Vector256<byte> m = MaskFor(unit);
                if (totalBytes >= NonTemporalThresholdBytes && ((nuint)dst & 31) == 0)
                {
                    for (; i + 32 <= totalBytes; i += 32)
                        Avx.StoreAlignedNonTemporal(dst + i, Avx2.Shuffle(Vector256.Load(src + i), m));
                    Sse.StoreFence();   // publish the streaming stores before the buffer is read
                }
                else
                {
                    for (; i + 32 <= totalBytes; i += 32)
                        Vector256.Store(Avx2.Shuffle(Vector256.Load(src + i), m), dst + i);
                }
            }
            else if (Ssse3.IsSupported)
            {
                Vector128<byte> m = MaskFor(unit).GetLower();
                for (; i + 16 <= totalBytes; i += 16)
                    Sse2.Store(dst + i, Ssse3.Shuffle(Vector128.Load(src + i), m));
            }
            for (; i + unit <= totalBytes; i += unit)
                for (int lo = 0, hi = unit - 1; lo < hi; lo++, hi--)
                {
                    dst[i + lo] = src[i + hi];
                    dst[i + hi] = src[i + lo];
                }
        }

        /// <summary>Reverses the bytes of one <paramref name="unit"/>-byte group in place.</summary>
        /// <param name="p">The group's first byte.</param>
        /// <param name="unit">The group size in bytes.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ReverseUnit(byte* p, int unit)
        {
            for (int lo = 0, hi = unit - 1; lo < hi; lo++, hi--)
            {
                byte t = p[lo];
                p[lo] = p[hi];
                p[hi] = t;
            }
        }

        /// <summary>
        ///     Reverses each <paramref name="unit"/>-byte unit of <paramref name="n"/> elements of <paramref name="size"/>
        ///     bytes that lie <paramref name="stride"/> bytes apart, in place: the strided half of
        ///     <see cref="DTypeMeta.copyswapn"/>'s swap (NumPy's <c>_strided_byte_swap</c> over a run that is not
        ///     contiguous). The unit width is switched on once per call, and a 2-, 4- or 8-byte unit is one
        ///     <see cref="BinaryPrimitives"/> reversal (unaligned reads and writes: an element of a byte-offset view need
        ///     not be aligned); any other width is reversed byte by byte.
        /// </summary>
        /// <param name="dst">The first element's first byte.</param>
        /// <param name="stride">The byte distance between elements (any sign).</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="size">An element's size in bytes, a multiple of <paramref name="unit"/>.</param>
        /// <param name="unit">The unit width in bytes; 1 leaves the elements as they are.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void ReverseStrided(byte* dst, long stride, long n, int size, int unit)
        {
            switch (unit)
            {
                case 1:
                    return;
                case 2:
                    for (long i = 0; i < n; i++)
                        for (byte* u = dst + i * stride, end = u + size; u < end; u += 2)
                            Unsafe.WriteUnaligned(u, BinaryPrimitives.ReverseEndianness(Unsafe.ReadUnaligned<ushort>(u)));
                    return;
                case 4:
                    for (long i = 0; i < n; i++)
                        for (byte* u = dst + i * stride, end = u + size; u < end; u += 4)
                            Unsafe.WriteUnaligned(u, BinaryPrimitives.ReverseEndianness(Unsafe.ReadUnaligned<uint>(u)));
                    return;
                case 8:
                    for (long i = 0; i < n; i++)
                        for (byte* u = dst + i * stride, end = u + size; u < end; u += 8)
                            Unsafe.WriteUnaligned(u, BinaryPrimitives.ReverseEndianness(Unsafe.ReadUnaligned<ulong>(u)));
                    return;
                default:
                    for (long i = 0; i < n; i++)
                        for (byte* u = dst + i * stride, end = u + size; u < end; u += unit)
                            ReverseUnit(u, unit);
                    return;
            }
        }

        /// <summary>The VPSHUFB mask that reverses each <paramref name="unit"/>-byte group of a lane.</summary>
        /// <param name="unit">2, 4, 8 or 16 (anything else is read as 16: Decimal's whole item).</param>
        /// <returns>The mask, repeated in both 128-bit lanes.</returns>
        [MethodImpl(OptimizeAndInline)]
        private static Vector256<byte> MaskFor(int unit) => unit switch
        {
            2 => Rev2,
            4 => Rev4,
            8 => Rev8,
            _ => Rev16, // unit == 16 (Decimal)
        };
    }
}
