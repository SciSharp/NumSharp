using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Bit-packing kernels for <see cref="np.packbits(NDArray,int?,string)"/> and
    ///     <see cref="np.unpackbits(NDArray,int?,int?,string)"/> — a straight C# port of NumPy 2.4.2's
    ///     <c>pack_inner</c> / <c>unpack_bits</c> (numpy/_core/src/multiarray/compiled_base.c).
    ///
    ///     <para>
    ///     WHY a bespoke kernel class instead of the elementwise/reduction IL generators or a per-line
    ///     NDIter drive: bit-packing is neither elementwise (8 inputs → 1 output byte) nor a reduction,
    ///     it is <b>dtype-agnostic</b> (an element contributes a single bit by its raw-byte
    ///     nonzero-ness, so the ONLY dtype-dependent quantity is the element byte width), and NumPy itself
    ///     does NOT route it through its ufunc machinery — it hand-writes <c>pack_inner</c>. This mirrors
    ///     <see cref="FiniteScan"/> (the fused <c>asarray_chkfinite</c> SIMD reduction), which is the
    ///     established home in this codebase for a fused, layout-aware, dtype-agnostic byte kernel. A
    ///     per-line NDIter drive was measured at 0.05–0.66× NumPy on multi-row/strided/F inputs (per-line
    ///     delegate overhead + cache-hostile column walks), which is why the caller normalises to a
    ///     C-contiguous buffer and this kernel processes the whole array in one call.
    ///     </para>
    ///
    ///     <para>
    ///     The caller (<c>np.packbits</c>/<c>np.unpackbits</c>) hands a C-contiguous source and a
    ///     C-contiguous destination plus the collapsed all-but-axis geometry
    ///     <c>(n_outer, axisLen, nel)</c> where <c>n_outer</c> = product of the dims BEFORE the packed
    ///     axis, <c>axisLen</c> = the packed axis length, and <c>nel</c> = product of the dims AFTER it.
    ///     Two SIMD strategies, dispatched on <c>nel</c>:
    ///     <list type="bullet">
    ///       <item><b>nel == 1</b> (packed axis is innermost, lanes are contiguous): pack/unpack each
    ///       contiguous lane with a byte movemask (pack) / 8-byte lookup expansion (unpack). When the
    ///       axis length is a multiple of 8 (pack) or fully consumed (unpack) every lane is byte-aligned,
    ///       so the whole <c>(n_outer × axisLen)</c> block is processed as ONE flat lane — no per-row
    ///       overhead.</item>
    ///       <item><b>nel &gt; 1</b> (packed axis is outer, lanes are strided by <c>nel</c>): process 8
    ///       axis positions at a time and vectorise ACROSS the <c>nel</c> inner columns, so the reads are
    ///       cache-sequential (NumPy walks each strided lane instead, which is why it is 10–33× slower
    ///       here). The bit is placed with a per-position OR-mask (<c>1&lt;&lt;(7-i)</c> big /
    ///       <c>1&lt;&lt;i</c> little), avoiding any per-byte vector shift which AVX2 lacks.</item>
    ///     </list>
    ///     </para>
    /// </summary>
    internal static unsafe class PackBits
    {
        /// <summary>
        ///     Per-byte unpack lookup for 'big' bit order: <c>_lutBig[b]</c> is the 8 output bytes
        ///     <c>[bit7, bit6, …, bit0]</c> of byte <c>b</c> packed little-endian into a <see cref="ulong"/>,
        ///     so a single 8-byte store expands one input byte. Precomputed once (256 entries).
        /// </summary>
        private static readonly ulong[] _lutBig = BuildLut(true);

        /// <summary>
        ///     Per-byte unpack lookup for 'little' bit order: <c>_lutLittle[b]</c> is
        ///     <c>[bit0, bit1, …, bit7]</c> — the byte-reversed twin of <see cref="_lutBig"/>.
        /// </summary>
        private static readonly ulong[] _lutLittle = BuildLut(false);

        /// <summary>
        ///     Builds a 256-entry unpack lookup. Each entry expands one byte to eight 0/1 output bytes
        ///     packed into a ulong, in the requested bit order.
        /// </summary>
        /// <param name="big">True for MSB-first ('big') output ordering, false for LSB-first ('little').</param>
        /// <returns>The 256-entry table indexed by input byte value.</returns>
        private static ulong[] BuildLut(bool big)
        {
            var lut = new ulong[256];
            for (int j = 0; j < 256; j++)
            {
                ulong v = 0;
                for (int i = 0; i < 8; i++)
                {
                    // big: output position i takes bit (7-i); little: output position i takes bit i.
                    int bit = big ? ((j >> (7 - i)) & 1) : ((j >> i) & 1);
                    v |= (ulong)bit << (8 * i);
                }
                lut[j] = v;
            }
            return lut;
        }

        /// <summary>
        ///     <see cref="Avx2.Shuffle(Vector256{byte},Vector256{byte})"/> (vpshufb) indices that reverse
        ///     each contiguous 8-lane group WITHIN each 128-bit half. Applied before the movemask so that,
        ///     after <see cref="Vector256.ExtractMostSignificantBits(Vector256{byte})"/> packs lane <c>j</c>
        ///     into bit <c>j</c> (LSB-first), input element 0 of each group lands in the byte's MSB — i.e.
        ///     'big' bit order. Uses vpshufb's PER-128-lane indexing (not the cross-lane
        ///     <see cref="Vector256.Shuffle{T}(Vector256{T},Vector256{byte})"/>, which the JIT lowers to a
        ///     multi-op cross-lane sequence — measured ~2× slower on this within-lane reversal).
        /// </summary>
        private static readonly Vector256<byte> _rev8Avx = Vector256.Create(
            (byte)7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);

        // ======================================================================================
        //  PACK
        // ======================================================================================

        /// <summary>
        ///     Packs the binary-valued elements of <paramref name="src"/> into bits of
        ///     <paramref name="dst"/> along the collapsed axis, matching NumPy <c>pack_inner</c>.
        ///     An element contributes bit 1 iff any of its <paramref name="elementSize"/> bytes is nonzero.
        /// </summary>
        /// <param name="src">C-contiguous source buffer (any bool/integer dtype, <paramref name="elementSize"/> bytes/element).</param>
        /// <param name="dst">C-contiguous destination buffer (uint8), sized <c>n_outer · ceil(axisLen/8) · nel</c>.</param>
        /// <param name="n_outer">Product of the dims before the packed axis (number of outer lanes).</param>
        /// <param name="axisLen">Length of the packed axis (number of input elements per lane).</param>
        /// <param name="nel">Product of the dims after the packed axis (1 ⇒ contiguous lane).</param>
        /// <param name="elementSize">Byte width of one source element (1/2/4/8).</param>
        /// <param name="big">True for 'big' bit order (first element ⇒ MSB), false for 'little'.</param>
        /// <remarks>Assumes <paramref name="axisLen"/> &gt; 0 and the buffers are non-null; the caller guards empties.</remarks>
        internal static void Pack(byte* src, byte* dst, long n_outer, long axisLen, long nel, int elementSize, bool big)
        {
            long nOut = ((axisLen - 1) >> 3) + 1;   // ceil(axisLen / 8)

            if (nel == 1)
            {
                if (elementSize == 1)
                {
                    // Byte-aligned axis ⇒ every lane is a whole number of output bytes, so the whole
                    // n_outer×axisLen block is one flat lane (no per-row setup). Otherwise pack per lane.
                    if ((axisLen & 7) == 0)
                        PackLaneContig1(src, dst, n_outer * axisLen, big);
                    else
                        for (long o = 0; o < n_outer; o++)
                            PackLaneContig1(src + o * axisLen, dst + o * nOut, axisLen, big);
                }
                else
                {
                    // Multi-byte contiguous lane: the byte-aligned block also flattens across lanes.
                    if ((axisLen & 7) == 0)
                        PackLaneContigMulti(src, dst, n_outer * axisLen, elementSize, big);
                    else
                        for (long o = 0; o < n_outer; o++)
                            PackLaneContigMulti(src + o * axisLen * elementSize, dst + o * nOut, axisLen, elementSize, big);
                }
                return;
            }

            // nel > 1: packed axis is strided by nel; vectorise across the nel inner columns.
            PackColumns(src, dst, n_outer, axisLen, nOut, nel, elementSize, big);
        }

        /// <summary>
        ///     Packs one contiguous byte lane (<paramref name="n"/> single-byte elements) into
        ///     <c>ceil(n/8)</c> output bytes with a 32-wide byte movemask; the final partial byte and the
        ///     &lt;4-byte tail fall to a scalar build identical to NumPy's.
        /// </summary>
        /// <param name="ip">Lane source pointer.</param>
        /// <param name="op">Lane destination pointer.</param>
        /// <param name="n">Number of source bytes in the lane.</param>
        /// <param name="big">True for 'big' bit order, false for 'little'.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void PackLaneContig1(byte* ip, byte* op, long n, bool big)
        {
            long nOut = ((n - 1) >> 3) + 1;
            int remain = (int)(n & 7);
            if (remain == 0) remain = 8;                 // a full trailing byte
            long nFull = (remain == 8) ? nOut : nOut - 1; // count of complete 8-bit output bytes
            long k = 0;

            // AVX2 movemask fast path. 4× unroll (128 in → 16 out) breaks the (shuffle→)compare→movemask
            // dependency chain so the CPU pipelines four independent chains, matching NumPy pack_inner's
            // vstepx4 loop. 'big' reverses each 8-lane group first (vpshufb) so the LSB-first movemask
            // emits MSB-first bytes; 'little' skips the shuffle. Non-AVX2 hosts take the scalar loop below.
            if (Avx2.IsSupported)
            {
                var zero = Vector256<byte>.Zero;
                if (big)
                {
                    var rev = _rev8Avx;
                    for (; k + 16 <= nFull; k += 16, ip += 128)
                    {
                        var v0 = Avx2.Shuffle(Avx2.LoadVector256(ip), rev);
                        var v1 = Avx2.Shuffle(Avx2.LoadVector256(ip + 32), rev);
                        var v2 = Avx2.Shuffle(Avx2.LoadVector256(ip + 64), rev);
                        var v3 = Avx2.Shuffle(Avx2.LoadVector256(ip + 96), rev);
                        ulong b0 = Vector256.ExtractMostSignificantBits(~Vector256.Equals(v0, zero));
                        ulong b1 = Vector256.ExtractMostSignificantBits(~Vector256.Equals(v1, zero));
                        ulong b2 = Vector256.ExtractMostSignificantBits(~Vector256.Equals(v2, zero));
                        ulong b3 = Vector256.ExtractMostSignificantBits(~Vector256.Equals(v3, zero));
                        *(ulong*)(op + k) = b0 | (b1 << 32);
                        *(ulong*)(op + k + 8) = b2 | (b3 << 32);
                    }
                    for (; k + 4 <= nFull; k += 4, ip += 32)
                    {
                        var v = Avx2.Shuffle(Avx2.LoadVector256(ip), rev);
                        *(uint*)(op + k) = Vector256.ExtractMostSignificantBits(~Vector256.Equals(v, zero));
                    }
                }
                else
                {
                    for (; k + 16 <= nFull; k += 16, ip += 128)
                    {
                        ulong b0 = Vector256.ExtractMostSignificantBits(~Vector256.Equals(Avx2.LoadVector256(ip), zero));
                        ulong b1 = Vector256.ExtractMostSignificantBits(~Vector256.Equals(Avx2.LoadVector256(ip + 32), zero));
                        ulong b2 = Vector256.ExtractMostSignificantBits(~Vector256.Equals(Avx2.LoadVector256(ip + 64), zero));
                        ulong b3 = Vector256.ExtractMostSignificantBits(~Vector256.Equals(Avx2.LoadVector256(ip + 96), zero));
                        *(ulong*)(op + k) = b0 | (b1 << 32);
                        *(ulong*)(op + k + 8) = b2 | (b3 << 32);
                    }
                    for (; k + 4 <= nFull; k += 4, ip += 32)
                    {
                        *(uint*)(op + k) = Vector256.ExtractMostSignificantBits(~Vector256.Equals(Avx2.LoadVector256(ip), zero));
                    }
                }
            }

            // Scalar remainder of full bytes + the trailing partial byte (NumPy pack_inner scalar loop).
            if (big)
            {
                for (; k < nFull; k++, ip += 8)
                {
                    int b = 0;
                    for (int i = 0; i < 8; i++) { b <<= 1; b |= ip[i] != 0 ? 1 : 0; }
                    op[k] = (byte)b;
                }
                if (remain != 8)
                {
                    int b = 0;
                    for (int i = 0; i < remain; i++) { b <<= 1; b |= ip[i] != 0 ? 1 : 0; }
                    op[nOut - 1] = (byte)(b << (8 - remain));   // left-justify the partial byte
                }
            }
            else
            {
                for (; k < nFull; k++, ip += 8)
                {
                    int b = 0;
                    for (int i = 0; i < 8; i++) { b >>= 1; b |= ip[i] != 0 ? 128 : 0; }
                    op[k] = (byte)b;
                }
                if (remain != 8)
                {
                    int b = 0;
                    for (int i = 0; i < remain; i++) { b >>= 1; b |= ip[i] != 0 ? 128 : 0; }
                    op[nOut - 1] = (byte)(b >> (8 - remain));   // right-justify the partial byte
                }
            }
        }

        /// <summary>
        ///     Packs one contiguous multi-byte lane (element nonzero iff any of its
        ///     <paramref name="elementSize"/> bytes is nonzero) — the scalar path NumPy also takes for
        ///     element widths &gt; 1.
        /// </summary>
        /// <param name="ip">Lane source pointer.</param>
        /// <param name="op">Lane destination pointer.</param>
        /// <param name="n">Number of source elements in the lane.</param>
        /// <param name="elementSize">Byte width of one element.</param>
        /// <param name="big">True for 'big' bit order, false for 'little'.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void PackLaneContigMulti(byte* ip, byte* op, long n, int elementSize, bool big)
        {
            long nOut = ((n - 1) >> 3) + 1;
            int remain = (int)(n & 7);
            if (remain == 0) remain = 8;
            byte* p = ip;
            for (long index = 0; index < nOut; index++)
            {
                int build = 0;
                int maxi = (index == nOut - 1) ? remain : 8;
                if (big)
                {
                    for (int i = 0; i < maxi; i++) { build <<= 1; build |= AnyNonZero(p, elementSize); p += elementSize; }
                    if (index == nOut - 1) build <<= (8 - remain);
                }
                else
                {
                    for (int i = 0; i < maxi; i++) { build >>= 1; build |= AnyNonZero(p, elementSize) != 0 ? 128 : 0; p += elementSize; }
                    if (index == nOut - 1) build >>= (8 - remain);
                }
                op[index] = (byte)build;
            }
        }

        /// <summary>Returns 1 if any of <paramref name="size"/> bytes at <paramref name="p"/> is nonzero, else 0.</summary>
        /// <param name="p">Pointer to the element bytes.</param>
        /// <param name="size">Element byte width.</param>
        /// <returns>1 if the element is nonzero, else 0.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int AnyNonZero(byte* p, int size)
        {
            int nz = 0;
            for (int j = 0; j < size; j++) nz |= p[j];
            return nz != 0 ? 1 : 0;
        }

        /// <summary>
        ///     Packs when the axis is strided (<paramref name="nel"/> &gt; 1): for each 8-element group of
        ///     the axis, vectorise the nonzero test ACROSS the <paramref name="nel"/> inner columns and OR
        ///     the resulting bit into the output byte at its position (no per-byte vector shift). Reads are
        ///     cache-sequential over the columns, unlike NumPy's per-lane strided walk.
        /// </summary>
        /// <param name="src">C-contiguous source buffer.</param>
        /// <param name="dst">C-contiguous destination buffer.</param>
        /// <param name="n_outer">Number of outer lanes (product of dims before the axis).</param>
        /// <param name="axisLen">Packed axis length.</param>
        /// <param name="nOut">Output axis length = ceil(axisLen/8).</param>
        /// <param name="nel">Inner column count (product of dims after the axis).</param>
        /// <param name="elementSize">Byte width of one element.</param>
        /// <param name="big">True for 'big' bit order, false for 'little'.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void PackColumns(byte* src, byte* dst, long n_outer, long axisLen, long nOut, long nel, int elementSize, bool big)
        {
            bool simd = Vector256.IsHardwareAccelerated && elementSize == 1;
            var zero = Vector256<byte>.Zero;
            for (long o = 0; o < n_outer; o++)
            {
                byte* sO = src + o * axisLen * nel * elementSize;
                byte* dO = dst + o * nOut * nel;
                for (long g = 0; g < nOut; g++)
                {
                    int maxi = (int)Math.Min(8, axisLen - 8 * g);
                    byte* dRow = dO + g * nel;
                    long c = 0;
                    if (simd)
                    {
                        for (; c + 32 <= nel; c += 32)
                        {
                            var acc = zero;
                            for (int i = 0; i < maxi; i++)
                            {
                                var v = Vector256.Load(sO + (8 * g + i) * nel + c);
                                byte bit = (byte)(big ? (1 << (7 - i)) : (1 << i));
                                // bit where the column element is nonzero: bitVec & ~(v==0).
                                acc |= Vector256.Create(bit) & ~Vector256.Equals(v, zero);
                            }
                            Vector256.Store(acc, dRow + c);
                        }
                    }
                    for (; c < nel; c++)
                    {
                        int b = 0;
                        for (int i = 0; i < maxi; i++)
                        {
                            int nz = AnyNonZero(sO + ((8 * g + i) * nel + c) * elementSize, elementSize);
                            b |= nz << (big ? (7 - i) : i);
                        }
                        dRow[c] = (byte)b;
                    }
                }
            }
        }

        // ======================================================================================
        //  UNPACK
        // ======================================================================================

        /// <summary>
        ///     Unpacks the bits of each uint8 element of <paramref name="src"/> into 0/1 bytes of
        ///     <paramref name="dst"/> along the collapsed axis, matching NumPy <c>unpack_bits</c>.
        /// </summary>
        /// <param name="src">C-contiguous source buffer (uint8).</param>
        /// <param name="dst">C-contiguous destination buffer (uint8), sized <c>n_outer · outLen · nel</c>.</param>
        /// <param name="n_outer">Product of the dims before the unpacked axis.</param>
        /// <param name="nIn">Length of the input axis (bytes per lane).</param>
        /// <param name="nel">Product of the dims after the unpacked axis (1 ⇒ contiguous lane).</param>
        /// <param name="outLen">Output axis length (post-<c>count</c>: may truncate below or pad above <c>nIn·8</c>).</param>
        /// <param name="big">True for 'big' bit order (MSB first), false for 'little' (LSB first).</param>
        /// <remarks>Assumes <paramref name="outLen"/> &gt; 0 and the buffers are non-null; the caller guards empties.</remarks>
        internal static void Unpack(byte* src, byte* dst, long n_outer, long nIn, long nel, long outLen, bool big)
        {
            long full = nIn * 8;
            // How much of each lane is a full-byte expansion, a partial trailing byte, and zero padding.
            long in_n, in_tail, out_pad;
            if (outLen > full) { in_n = nIn; in_tail = 0; out_pad = outLen - full; }
            else { in_n = outLen / 8; in_tail = outLen % 8; out_pad = 0; }

            if (nel == 1)
            {
                var lut = big ? _lutBig : _lutLittle;
                // Whole axis consumed with no tail/pad ⇒ every lane is 8·nIn bytes, so flatten.
                if (in_n == nIn && in_tail == 0 && out_pad == 0)
                {
                    UnpackLaneFull(src, dst, n_outer * nIn, lut);
                }
                else
                {
                    for (long o = 0; o < n_outer; o++)
                        UnpackLane(src + o * nIn, dst + o * outLen, in_n, in_tail, out_pad, lut);
                }
                return;
            }

            UnpackColumns(src, dst, n_outer, nIn, nel, outLen, in_n, in_tail, out_pad, big);
        }

        /// <summary>
        ///     Expands <paramref name="nIn"/> contiguous input bytes to <c>8·nIn</c> output bytes via the
        ///     8-byte lookup — one store per input byte (NumPy's unity-stride fast path, but without the
        ///     per-byte <c>memcpy</c> call).
        /// </summary>
        /// <param name="ip">Source pointer.</param>
        /// <param name="op">Destination pointer.</param>
        /// <param name="nIn">Number of input bytes to expand.</param>
        /// <param name="lut">The bit-order lookup table.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void UnpackLaneFull(byte* ip, byte* op, long nIn, ulong[] lut)
        {
            fixed (ulong* pl = lut)
                for (long k = 0; k < nIn; k++)
                    *(ulong*)(op + 8 * k) = pl[ip[k]];
        }

        /// <summary>
        ///     Unpacks one contiguous lane with a partial trailing byte and/or zero padding
        ///     (the <c>count</c>-shortened / count-extended path).
        /// </summary>
        /// <param name="ip">Lane source pointer.</param>
        /// <param name="op">Lane destination pointer.</param>
        /// <param name="in_n">Number of full input bytes to expand (8 outputs each).</param>
        /// <param name="in_tail">Number of leading bits of the next input byte to emit (0–7).</param>
        /// <param name="out_pad">Number of trailing zero bytes to append.</param>
        /// <param name="lut">The bit-order lookup table.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void UnpackLane(byte* ip, byte* op, long in_n, long in_tail, long out_pad, ulong[] lut)
        {
            fixed (ulong* pl = lut)
            {
                for (long k = 0; k < in_n; k++)
                    *(ulong*)(op + 8 * k) = pl[ip[k]];
                byte* t = op + 8 * in_n;
                if (in_tail > 0)
                {
                    ulong v = pl[ip[in_n]];                 // the lookup already carries this lane's bit order
                    for (long i = 0; i < in_tail; i++) t[i] = (byte)(v >> (int)(8 * i));
                    t += in_tail;
                }
                for (long i = 0; i < out_pad; i++) t[i] = 0;
            }
        }

        /// <summary>
        ///     Unpacks when the axis is strided (<paramref name="nel"/> &gt; 1): for each input byte
        ///     position, emit the 8 output rows by testing one bit across all <paramref name="nel"/>
        ///     inner columns with a vector mask — writes are cache-sequential over the columns.
        /// </summary>
        /// <param name="src">C-contiguous source buffer.</param>
        /// <param name="dst">C-contiguous destination buffer.</param>
        /// <param name="n_outer">Number of outer lanes.</param>
        /// <param name="nIn">Input axis length.</param>
        /// <param name="nel">Inner column count.</param>
        /// <param name="outLen">Output axis length.</param>
        /// <param name="in_n">Full input bytes expanded per lane.</param>
        /// <param name="in_tail">Leading bits of the next input byte per lane.</param>
        /// <param name="out_pad">Trailing zero rows per lane.</param>
        /// <param name="big">True for 'big' bit order, false for 'little'.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void UnpackColumns(byte* src, byte* dst, long n_outer, long nIn, long nel, long outLen,
            long in_n, long in_tail, long out_pad, bool big)
        {
            bool simd = Vector256.IsHardwareAccelerated;
            var zero = Vector256<byte>.Zero;
            var one = Vector256.Create((byte)1);
            for (long o = 0; o < n_outer; o++)
            {
                byte* sO = src + o * nIn * nel;
                byte* dO = dst + o * outLen * nel;
                for (long k = 0; k < in_n; k++)
                {
                    byte* sRow = sO + k * nel;
                    for (int i = 0; i < 8; i++)
                        EmitBitRow(sRow, dO + (8 * k + i) * nel, nel, (byte)(big ? (1 << (7 - i)) : (1 << i)), simd, zero, one);
                }
                // partial trailing byte
                if (in_tail > 0)
                {
                    byte* sRow = sO + in_n * nel;
                    for (long i = 0; i < in_tail; i++)
                        EmitBitRow(sRow, dO + (8 * in_n + i) * nel, nel, (byte)(big ? (1 << (int)(7 - i)) : (1 << (int)i)), simd, zero, one);
                }
                // zero padding
                for (long i = 0; i < out_pad; i++)
                {
                    byte* dRow = dO + (8 * in_n + in_tail + i) * nel;
                    for (long c = 0; c < nel; c++) dRow[c] = 0;
                }
            }
        }

        /// <summary>
        ///     Writes one output row of <paramref name="nel"/> bytes, each the 0/1 test of
        ///     <paramref name="mask"/> against the corresponding input-row column.
        /// </summary>
        /// <param name="sRow">Input row (one byte per column).</param>
        /// <param name="dRow">Output row (one byte per column).</param>
        /// <param name="nel">Column count.</param>
        /// <param name="mask">Single-bit test mask (<c>1&lt;&lt;(7-i)</c> big / <c>1&lt;&lt;i</c> little).</param>
        /// <param name="simd">Whether the SIMD path is enabled.</param>
        /// <param name="zero">A cached zero vector.</param>
        /// <param name="one">A cached all-ones-byte (value 1) vector.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void EmitBitRow(byte* sRow, byte* dRow, long nel, byte mask, bool simd, Vector256<byte> zero, Vector256<byte> one)
        {
            long c = 0;
            if (simd)
            {
                var maskVec = Vector256.Create(mask);
                for (; c + 32 <= nel; c += 32)
                {
                    var v = Vector256.Load(sRow + c);
                    // 1 where (v & mask) != 0, else 0.
                    var bit = one & ~Vector256.Equals(v & maskVec, zero);
                    Vector256.Store(bit, dRow + c);
                }
            }
            for (; c < nel; c++)
                dRow[c] = (byte)((sRow[c] & mask) != 0 ? 1 : 0);
        }
    }
}
