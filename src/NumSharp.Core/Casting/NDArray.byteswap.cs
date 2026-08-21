using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public partial class NDArray
    {
        // =====================================================================
        // NDArray.byteswap — port of NumPy 2.4.2's ndarray.byteswap / PyArray_Byteswap
        // (numpy/_core/src/multiarray/methods.c).
        //
        //   Reverse the bytes WITHIN each element, toggling the low/big-endian
        //   representation. The dtype is unchanged — only the raw bytes move, so the
        //   *reinterpreted* values change (int32 1 -> 0x01000000). It is a pure
        //   byte permutation: it depends ONLY on the element's byte width, never on
        //   the dtype's numeric meaning, so ONE width-dispatched SIMD kernel covers
        //   every dtype instead of a per-NPTypeCode path.
        //
        //   The "swap unit" is the element size, EXCEPT complex, whose real and
        //   imaginary halves are swapped independently (NumPy's copyswapn for a
        //   complex dtype swaps per component: '<c16' is two 8-byte doubles, not one
        //   16-byte blob). 1-byte dtypes (bool/i1/u1) are a value no-op.
        //
        //   NumSharp-only dtypes have no NumPy analog, so they take the consistent
        //   raw-itemsize rule: Char (2 bytes) swaps in 2s, Decimal (16 bytes) in 16s.
        //
        //   inplace=False (default): return a fresh COPY (never shares memory) in
        //     NumPy's PyArray_NewCopy(ANYORDER) layout — F iff the source is
        //     F-contiguous-and-not-C, else C — then swap it. Even a 1-byte dtype
        //     returns a copy.
        //   inplace=True: swap the viewed elements in place and return self. Requires
        //     a writeable array, else ValueError (a broadcast/read-only view raises).
        // =====================================================================

        // Per-128-bit-lane VPSHUFB masks: result[j] = src[mask[j]] reverses each
        // `unit`-byte group. The 16-byte pattern is repeated across both lanes (AVX2
        // shuffle is lane-local and every unit <= 16 fits inside one lane), so the
        // same mask drives the Ssse3 128-bit path via GetLower().
        private static readonly Vector256<byte> _bswapRev2 = Vector256.Create(
            (byte)1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
                   1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
        private static readonly Vector256<byte> _bswapRev4 = Vector256.Create(
            (byte)3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                   3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
        private static readonly Vector256<byte> _bswapRev8 = Vector256.Create(
            (byte)7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
                   7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
        private static readonly Vector256<byte> _bswapRev16 = Vector256.Create(
            (byte)15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                   15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        /// <summary>
        ///     Swap the bytes of the array elements — toggle between low-endian and big-endian data
        ///     representation. Mirrors NumPy's <c>ndarray.byteswap(inplace=False)</c>: the dtype is
        ///     unchanged and only the raw element bytes are reversed, so the reinterpreted values change.
        ///     A complex element has its real and imaginary parts swapped individually; 1-byte dtypes are
        ///     an in-place no-op (but <c>inplace=False</c> still returns a fresh copy).
        /// </summary>
        /// <param name="inplace">
        ///     When <c>true</c>, swap this array's data in place and return this same instance. When
        ///     <c>false</c> (default), return a byte-swapped copy and leave this array untouched.
        /// </param>
        /// <returns>The byte-swapped array (this instance when <paramref name="inplace"/>, else a copy).</returns>
        /// <exception cref="ValueError">
        ///     When <paramref name="inplace"/> is <c>true</c> and the array is not writeable (e.g. a broadcast view).
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.byteswap.html</remarks>
        public NDArray byteswap(bool inplace = false)
        {
            int itemsize = this.itemsize;
            // Complex swaps each component (real/imag) independently — half the itemsize per unit.
            int unit = typecode == NPTypeCode.Complex ? itemsize / 2 : itemsize;

            if (inplace)
            {
                // NumPy: PyArray_FailUnlessWriteable(self, "array to be byte-swapped").
                if (!Shape.IsWriteable)
                    throw new ValueError("array to be byte-swapped is read-only");

                if (unit > 1 && size > 0)
                    ByteswapInPlace(unit, itemsize);
                return this;
            }

            // NumPy: PyArray_NewCopy(self, NPY_ANYORDER) then byteswap the copy in place — i.e. a
            // copy pass THEN a swap pass. When the source is C-contiguous (the common case) both
            // fuse into ONE pass: allocate an uninitialized C-contiguous result and read→shuffle→
            // write straight across, halving the memory traffic and skipping the copy machinery.
            // A 1-byte dtype has no bytes to swap, so it is a straight contiguous copy.
            if (Shape.IsContiguous && size > 0)
            {
                var result = new NDArray(typecode, new Shape(shape), fillZeros: false);
                unsafe
                {
                    byte* src = (byte*)Storage.Address + Shape.offset * (long)itemsize;
                    byte* dst = (byte*)result.Storage.Address;   // fresh -> offset 0
                    long bytes = (long)size * itemsize;
                    if (unit > 1) SwapBlockCopy(src, dst, bytes, unit);
                    else Buffer.MemoryCopy(src, dst, bytes, bytes);
                }
                return result;
            }

            // F-contiguous-and-not-C / strided / broadcast / empty: copy in NumPy's ANYORDER layout
            // (F only for an F-contiguous-and-not-C source, via Shape.Order), then swap it.
            NDArray fallback = this.copy(Shape.Order);
            if (unit > 1 && fallback.size > 0)
            {
                unsafe
                {
                    byte* p = (byte*)fallback.Storage.Address + fallback.Shape.offset * (long)itemsize;
                    SwapBlock(p, (long)fallback.size * itemsize, unit);
                }
            }
            return fallback;
        }

        /// <summary>In-place byteswap over any layout. One-segment arrays swap their whole contiguous
        /// byte block; strided/transposed/reversed views drive a single-operand EXTERNAL_LOOP NDIter
        /// (memory order) whose kernel swaps each chunk — NumPy's IterAllButAxis inplace path.</summary>
        private unsafe void ByteswapInPlace(int unit, int itemsize)
        {
            // One segment (C- OR F-contiguous): the logical elements occupy a single contiguous byte
            // run of size*itemsize (column-major layout is still one block), so swap it wholesale.
            if (Shape.IsContiguous || Shape.IsFContiguous)
            {
                byte* p = (byte*)Storage.Address + Shape.offset * (long)itemsize;
                SwapBlock(p, (long)size * itemsize, unit);
                return;
            }

            // Non-contiguous: visit every element via a single READWRITE EXTERNAL_LOOP iterator in
            // memory order (K), so a strided array's contiguous inner runs coalesce into large chunks.
            var ctx = new SwapCtx { itemsize = itemsize, unit = unit };
            var iter = NDIterRef.AdvancedNew(1, new[] { this }, NDIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                new[] { NDIterPerOpFlags.READWRITE });
            try { iter.ForEach(ByteswapStridedKernel, &ctx); }
            finally { iter.Dispose(); }
        }

        private struct SwapCtx { public int itemsize; public int unit; }

        // NDInnerLoopFunc: swap one inner-loop chunk of `count` elements at byte stride strides[0].
        // A unit-stride chunk (the coalesced inner axis) is a contiguous block -> SIMD; any other
        // stride (non-unit / negative) reverses each element's bytes individually.
        private static unsafe void ByteswapStridedKernel(void** dataptrs, long* strides, long count, void* auxdata)
        {
            byte* data = (byte*)dataptrs[0];
            long s = strides[0];
            SwapCtx* c = (SwapCtx*)auxdata;
            int itemsize = c->itemsize, unit = c->unit;

            if (s == itemsize)
            {
                SwapBlock(data, count * itemsize, unit);
                return;
            }
            for (long i = 0; i < count; i++)
                ReverseElement(data + i * s, itemsize, unit);
        }

        /// <summary>Reverse each <paramref name="unit"/>-byte group across <paramref name="totalBytes"/>
        /// contiguous bytes (a multiple of <paramref name="unit"/>). VPSHUFB reverses lane-locally, so the
        /// same per-lane mask drives AVX2 (32 B/iter) and SSSE3 (16 B/iter); a scalar two-pointer swap
        /// handles the tail and the no-SIMD fallback.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SwapBlock(byte* p, long totalBytes, int unit)
        {
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
                ReverseBytes(p + i, unit);
        }

        // Above this size a fused not-inplace swap streams its writes non-temporally (below it the
        // result stays cache-hot). The fresh destination is not in cache, so an ordinary store pays a
        // read-for-ownership NumPy's Buffer.MemoryCopy avoids; a streaming store avoids it too. Chosen
        // to clear typical L2 so only genuinely out-of-cache writes bypass the cache.
        private const long NonTemporalThresholdBytes = 1L << 21;   // 2 MiB

        /// <summary>Fused copy+swap: read from <paramref name="src"/>, reverse each <paramref name="unit"/>-byte
        /// group, write to <paramref name="dst"/> — one pass instead of a copy pass then a swap pass. Same
        /// VPSHUFB mask/width dispatch as <see cref="SwapBlock"/>; the scalar tail reverses src→dst directly.
        /// A large, 32-byte-aligned destination streams non-temporally to skip the read-for-ownership.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SwapBlockCopy(byte* src, byte* dst, long totalBytes, int unit)
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> MaskFor(int unit) => unit switch
        {
            2 => _bswapRev2,
            4 => _bswapRev4,
            8 => _bswapRev8,
            _ => _bswapRev16, // unit == 16 (Decimal)
        };

        /// <summary>Reverse each of the <paramref name="unit"/>-byte components of one strided element
        /// (one group for every dtype but complex, whose two halves reverse independently).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ReverseElement(byte* e, int itemsize, int unit)
        {
            for (int u = 0; u < itemsize; u += unit)
                ReverseBytes(e + u, unit);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ReverseBytes(byte* p, int unit)
        {
            for (int lo = 0, hi = unit - 1; lo < hi; lo++, hi--)
            {
                byte t = p[lo];
                p[lo] = p[hi];
                p[hi] = t;
            }
        }
    }
}
