using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.TransposeTile.cs — kernel: copy a tile of a matrix
// whose COLUMNS are contiguous in memory into rows that are contiguous
// =============================================================================
//
// RESPONSIBILITY
//   dst[b * dstRowStride + w * itemSize] = src[w * srcColumnStride + b * itemSize]
//   for b in [0, rows), w in [0, cols): `rows` consecutive elements of each of
//   `cols` source columns become `cols` consecutive elements of each of `rows`
//   destination rows. Element width 1, 2, 4, 8 or 16 bytes; both strides are in
//   BYTES and of either sign.
//
// WHY IT EXISTS
//   An F-ordered operand that meets a C-ordered one is read in C order: row by
//   row. One of its rows is a run with a stride of one COLUMN (rows * itemSize
//   bytes), so a row of N elements touches N cache lines, and N pages once a
//   column is 4 KB long. Gathering such rows one at a time cost 0.3 to 0.6 ns an
//   element for a (250, 400) matrix and 4 to 8 ns for a (3162, 3162) one, where a
//   contiguous comparison takes 0.03 to 0.14 ns.
//
//   The elements of one source COLUMN are adjacent. A tile of B rows therefore
//   reads B * itemSize contiguous bytes of each column once and deals them out to
//   B destination rows: B elements for each cache line and page touched, not one.
//
// HOW
//   Vector groups, AVX2 (measured ns an element for an L1-resident tile):
//     8 bytes   2 rows x  4 columns   0.13    2 loads + 2 loads folded into an insert, 2 unpacks, 2 stores
//     4 bytes   4 rows x  8 columns   0.08    in-lane 4 x 4 transposition of two lanes
//     2 bytes   8 rows x 16 columns   0.05    in-lane 8 x 8
//     1 byte   16 rows x 32 columns   0.03    in-lane 16 x 16
//   Each 256-bit register is built from two columns `far` apart: the first column's
//   piece in the low lane, the other one's in the high lane (vinserti128 with a
//   memory operand, which does not use the shuffle ports). The in-lane transposition
//   then yields rows that hold twice the columns, with no lane-crossing shuffle.
//   Integer unpacks are used for every width: current Intel cores run them on two
//   ports and the floating-point ones (vunpcklpd, vunpcklps) on one.
//
//   A tile that is not a whole number of groups ends with a group that OVERLAPS the
//   one before it (the same elements are copied twice), so there is no scalar tail.
//   That needs a tile of at least one group in each direction, and a destination
//   that is not the source.
//
//   A tile smaller than one group, a 16-byte element and a host without AVX2 use
//   moves: the strided move kernel along the tile's longer side, or a plain loop.
//
// WHAT IT READS AND WRITES
//   Exactly the rows x cols elements named above, on both sides. The vector groups
//   load whole column pieces of the tile and store whole row pieces of the tile;
//   no load or store reaches outside it (test: TransposeTileTests, with the source
//   and the destination against inaccessible pages).
//
// WHO USES IT
//   ComparisonRun.RunRows (Backends/Kernels/ComparisonRun.cs): an operand whose
//   successive ROWS are adjacent elements (an F-ordered matrix, a transposed view)
//   is staged tile by tile.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        ///     Rows of one vector group of <see cref="TransposeTile"/> for an element width, or 1 when that width has
        ///     no vector group on this host (a 16-byte element, an unknown width, no AVX2).
        /// </summary>
        /// <param name="itemSize">Element width in bytes.</param>
        /// <returns>16, 8, 4 or 2 for a width of 1, 2, 4 or 8 bytes with AVX2; otherwise 1.</returns>
        /// <remarks>
        ///     A caller that cuts a matrix into tiles keeps the tile height a multiple of this number: a tile of
        ///     another height is still copied correctly, but its last group repeats rows of the group before it.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int TransposeTileRowGroup(int itemSize)
        {
            if (!Avx2.IsSupported)
                return 1;
            return itemSize switch { 1 => 16, 2 => 8, 4 => 4, 8 => 2, _ => 1 };
        }

        /// <summary>
        ///     Columns of one vector group of <see cref="TransposeTile"/> for an element width, or 1 when that width has
        ///     no vector group on this host.
        /// </summary>
        /// <param name="itemSize">Element width in bytes.</param>
        /// <returns>32, 16, 8 or 4 for a width of 1, 2, 4 or 8 bytes with AVX2; otherwise 1.</returns>
        /// <remarks>See <see cref="TransposeTileRowGroup"/>: the same rule holds for a tile's width.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int TransposeTileColumnGroup(int itemSize)
        {
            if (!Avx2.IsSupported)
                return 1;
            return itemSize switch { 1 => 32, 2 => 16, 4 => 8, 8 => 4, _ => 1 };
        }

        /// <summary>
        ///     Copies a tile of <paramref name="rows"/> x <paramref name="cols"/> elements from a source whose columns
        ///     are contiguous to a destination whose rows are contiguous:
        ///     <c>dst[b * dstRowStride + w * itemSize] = src[w * srcColumnStride + b * itemSize]</c>.
        /// </summary>
        /// <param name="src">Element (row 0, column 0) of the tile in the source. The <paramref name="rows"/> elements
        ///     of one column follow each other from this address upwards.</param>
        /// <param name="srcColumnStride">Signed byte distance between two neighbouring columns of the source.</param>
        /// <param name="dst">Element (row 0, column 0) of the tile in the destination. The <paramref name="cols"/>
        ///     elements of one row follow each other from this address upwards.</param>
        /// <param name="dstRowStride">Signed byte distance between two neighbouring rows of the destination. A negative
        ///     value stores the rows downwards from <paramref name="dst"/>, which reverses their order in memory.</param>
        /// <param name="rows">Elements per source column, and rows written (0 or less copies nothing).</param>
        /// <param name="cols">Source columns, and elements per row written (0 or less copies nothing).</param>
        /// <param name="itemSize">Element width in bytes: 1, 2, 4, 8 or 16.</param>
        /// <exception cref="NotSupportedException"><paramref name="itemSize"/> is not 1, 2, 4, 8 or 16.</exception>
        /// <remarks>
        ///     <para><b>The destination must not overlap the source.</b> A tile that is not a whole number of vector
        ///     groups copies some elements twice (the last group overlaps the one before it), which is only correct
        ///     when the first copy did not change what the second one reads.</para>
        ///     <para>The method reads exactly the tile's elements of the source and writes exactly the tile's elements
        ///     of the destination: it can be used on a tile that ends at the last byte of its buffer.</para>
        ///     <para>Speed: see the file header. A tile with fewer rows than <see cref="TransposeTileRowGroup"/> or
        ///     fewer columns than <see cref="TransposeTileColumnGroup"/> is moved one element at a time (about 0.2 ns
        ///     an element), so a caller that can choose keeps its tiles at least one group large.</para>
        /// </remarks>
        internal static unsafe void TransposeTile(byte* src, long srcColumnStride, byte* dst, long dstRowStride, long rows, long cols,
            int itemSize)
        {
            if (rows <= 0 || cols <= 0)
                return;

            long cs = srcColumnStride, rs = dstRowStride;
            // A 16-byte element is a move whatever the instruction set: there is nothing to interleave.
            if (itemSize == 16 || !Avx2.IsSupported || rows < TransposeTileRowGroup(itemSize)
                || cols < TransposeTileColumnGroup(itemSize))
            {
                TransposeTileByMoves(src, cs, dst, rs, rows, cols, itemSize);
                return;
            }

            // One loop nest per width, so that the group body is inlined into a loop with nothing else in it.
            // In every nest the LAST group of each direction is moved back to end on the tile's edge.
            switch (itemSize)
            {
                case 8:
                    for (long w = 0; w < cols; w += 4)
                    {
                        if (w > cols - 4)
                            w = cols - 4;
                        byte* s = src + w * cs;
                        byte* d = dst + w * 8;
                        // Two row groups a step: the loop's own instructions are then a fifth of the work, not a third.
                        long b = 0;
                        for (; b + 4 <= rows; b += 4)
                        {
                            TransposeGroup8(s + b * 8, cs, d + b * rs, rs);
                            TransposeGroup8(s + b * 8 + 16, cs, d + (b + 2) * rs, rs);
                        }
                        if (b + 2 <= rows)
                        {
                            TransposeGroup8(s + b * 8, cs, d + b * rs, rs);
                            b += 2;
                        }
                        if (b < rows)       // an odd row count
                            TransposeGroup8(s + (rows - 2) * 8, cs, d + (rows - 2) * rs, rs);
                    }
                    break;

                case 4:
                    for (long w = 0; w < cols; w += 8)
                    {
                        if (w > cols - 8)
                            w = cols - 8;
                        byte* s = src + w * cs;
                        byte* d = dst + w * 4;
                        for (long b = 0; b < rows; b += 4)
                        {
                            if (b > rows - 4)
                                b = rows - 4;
                            TransposeGroup4(s + b * 4, cs, d + b * rs, rs);
                        }
                    }
                    break;

                case 2:
                    for (long w = 0; w < cols; w += 16)
                    {
                        if (w > cols - 16)
                            w = cols - 16;
                        byte* s = src + w * cs;
                        byte* d = dst + w * 2;
                        for (long b = 0; b < rows; b += 8)
                        {
                            if (b > rows - 8)
                                b = rows - 8;
                            TransposeGroup2(s + b * 2, cs, d + b * rs, rs);
                        }
                    }
                    break;

                case 1:
                    for (long w = 0; w < cols; w += 32)
                    {
                        if (w > cols - 32)
                            w = cols - 32;
                        byte* s = src + w * cs;
                        byte* d = dst + w;
                        for (long b = 0; b < rows; b += 16)
                        {
                            if (b > rows - 16)
                                b = rows - 16;
                            TransposeGroup1(s + b, cs, d + b * rs, rs);
                        }
                    }
                    break;

                default:
                    throw new NotSupportedException($"TransposeTile has no kernel for elements of {itemSize} bytes.");
            }
        }

        /// <summary>
        ///     The tile copy of <see cref="TransposeTile"/> without vector groups: one move an element, along the
        ///     tile's longer side.
        /// </summary>
        /// <param name="src">Element (row 0, column 0) of the tile in the source.</param>
        /// <param name="cs">Signed byte distance between two source columns.</param>
        /// <param name="dst">Element (row 0, column 0) of the tile in the destination.</param>
        /// <param name="rs">Signed byte distance between two destination rows.</param>
        /// <param name="rows">Rows of the tile (positive).</param>
        /// <param name="cols">Columns of the tile (positive).</param>
        /// <param name="itemSize">Element width: 1, 2, 4, 8 or 16 bytes.</param>
        /// <exception cref="NotSupportedException"><paramref name="itemSize"/> is not 1, 2, 4, 8 or 16.</exception>
        /// <remarks>
        ///     With the strided move kernel a tile of more rows than columns is copied a COLUMN at a time (a contiguous
        ///     read, a strided store), every other tile a ROW at a time (a strided read, a contiguous store): the
        ///     kernel's call is then paid for the shorter side only. Without runtime IL generation the same moves are
        ///     a plain loop (about three times slower for 16-byte elements; measured 0.6 against 0.2 ns an element).
        /// </remarks>
        private static unsafe void TransposeTileByMoves(byte* src, long cs, byte* dst, long rs, long rows, long cols, int itemSize)
        {
            var move = GetStridedMoveKernel(itemSize);
            if (move is not null)
            {
                if (rows > cols)
                {
                    for (long w = 0; w < cols; w++)
                        move(dst + w * itemSize, rs, src + w * cs, itemSize, rows);
                }
                else
                {
                    for (long b = 0; b < rows; b++)
                        move(dst + b * rs, itemSize, src + b * itemSize, cs, cols);
                }
                return;
            }

            for (long w = 0; w < cols; w++)
            {
                byte* s = src + w * cs;
                byte* d = dst + w * itemSize;
                switch (itemSize)
                {
                    case 16:
                        for (long b = 0; b < rows; b++)
                            Vector128.Store(Vector128.Load(s + b * 16), d + b * rs);
                        break;
                    case 8:
                        for (long b = 0; b < rows; b++)
                            *(ulong*)(d + b * rs) = *(ulong*)(s + b * 8);
                        break;
                    case 4:
                        for (long b = 0; b < rows; b++)
                            *(uint*)(d + b * rs) = *(uint*)(s + b * 4);
                        break;
                    case 2:
                        for (long b = 0; b < rows; b++)
                            *(ushort*)(d + b * rs) = *(ushort*)(s + b * 2);
                        break;
                    case 1:
                        for (long b = 0; b < rows; b++)
                            d[b * rs] = s[b];
                        break;
                    default:
                        throw new NotSupportedException($"TransposeTile has no kernel for elements of {itemSize} bytes.");
                }
            }
        }

        // -------------------------------------------------------------------------------------------------------------
        //  Vector groups (AVX2). In each of them register k holds a piece of column k in its low lane and the same
        //  rows of the column `far` further in its high lane.
        // -------------------------------------------------------------------------------------------------------------

        /// <summary>Two 16-byte pieces as one register: <paramref name="p"/> in the low lane,
        ///     <paramref name="p"/> + <paramref name="far"/> in the high lane (the high piece is a memory operand of
        ///     the insert).</summary>
        /// <param name="p">Address of the low piece.</param>
        /// <param name="far">Byte distance of the high piece from the low one.</param>
        /// <returns>The 32 bytes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<ulong> TransposePair8(byte* p, long far)
            => Avx2.InsertVector128(Sse2.LoadVector128((ulong*)p).ToVector256Unsafe(), Sse2.LoadVector128((ulong*)(p + far)), 1);

        /// <inheritdoc cref="TransposePair8"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<uint> TransposePair4(byte* p, long far)
            => Avx2.InsertVector128(Sse2.LoadVector128((uint*)p).ToVector256Unsafe(), Sse2.LoadVector128((uint*)(p + far)), 1);

        /// <inheritdoc cref="TransposePair8"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<ushort> TransposePair2(byte* p, long far)
            => Avx2.InsertVector128(Sse2.LoadVector128((ushort*)p).ToVector256Unsafe(), Sse2.LoadVector128((ushort*)(p + far)), 1);

        /// <inheritdoc cref="TransposePair8"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<byte> TransposePair1(byte* p, long far)
            => Avx2.InsertVector128(Sse2.LoadVector128(p).ToVector256Unsafe(), Sse2.LoadVector128(p + far), 1);

        /// <summary>Copies 2 rows of 4 columns of 8-byte elements.</summary>
        /// <param name="p">Element (row 0, column 0) of the group in the source.</param>
        /// <param name="cs">Byte distance between two source columns.</param>
        /// <param name="d">Element (row 0, column 0) of the group in the destination.</param>
        /// <param name="rs">Byte distance between two destination rows.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void TransposeGroup8(byte* p, long cs, byte* d, long rs)
        {
            var a = TransposePair8(p, 2 * cs);          // column 0 rows 0,1 | column 2 rows 0,1
            var b = TransposePair8(p + cs, 2 * cs);     // column 1 rows 0,1 | column 3 rows 0,1
            Avx.Store((ulong*)d, Avx2.UnpackLow(a, b));             // row 0: columns 0,1 | 2,3
            Avx.Store((ulong*)(d + rs), Avx2.UnpackHigh(a, b));     // row 1
        }

        /// <summary>Copies 4 rows of 8 columns of 4-byte elements.</summary>
        /// <param name="p">Element (row 0, column 0) of the group in the source.</param>
        /// <param name="cs">Byte distance between two source columns.</param>
        /// <param name="d">Element (row 0, column 0) of the group in the destination.</param>
        /// <param name="rs">Byte distance between two destination rows.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void TransposeGroup4(byte* p, long cs, byte* d, long rs)
        {
            long far = 4 * cs;
            var y0 = TransposePair4(p, far);            // column 0 rows 0..3 | column 4 rows 0..3
            var y1 = TransposePair4(p + cs, far);
            var y2 = TransposePair4(p + 2 * cs, far);
            var y3 = TransposePair4(p + 3 * cs, far);
            var t0 = Avx2.UnpackLow(y0, y1).AsUInt64();     // rows 0,1 of columns 0,1 | 4,5
            var t1 = Avx2.UnpackHigh(y0, y1).AsUInt64();    // rows 2,3
            var t2 = Avx2.UnpackLow(y2, y3).AsUInt64();     // rows 0,1 of columns 2,3 | 6,7
            var t3 = Avx2.UnpackHigh(y2, y3).AsUInt64();
            Avx.Store((ulong*)d, Avx2.UnpackLow(t0, t2));
            Avx.Store((ulong*)(d + rs), Avx2.UnpackHigh(t0, t2));
            Avx.Store((ulong*)(d + 2 * rs), Avx2.UnpackLow(t1, t3));
            Avx.Store((ulong*)(d + 3 * rs), Avx2.UnpackHigh(t1, t3));
        }

        /// <summary>Copies 8 rows of 16 columns of 2-byte elements.</summary>
        /// <param name="p">Element (row 0, column 0) of the group in the source.</param>
        /// <param name="cs">Byte distance between two source columns.</param>
        /// <param name="d">Element (row 0, column 0) of the group in the destination.</param>
        /// <param name="rs">Byte distance between two destination rows.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void TransposeGroup2(byte* p, long cs, byte* d, long rs)
        {
            long far = 8 * cs;
            var y0 = TransposePair2(p, far);            // column 0 rows 0..7 | column 8 rows 0..7
            var y1 = TransposePair2(p + cs, far);
            var y2 = TransposePair2(p + 2 * cs, far);
            var y3 = TransposePair2(p + 3 * cs, far);
            byte* q = p + 4 * cs;
            var y4 = TransposePair2(q, far);
            var y5 = TransposePair2(q + cs, far);
            var y6 = TransposePair2(q + 2 * cs, far);
            var y7 = TransposePair2(q + 3 * cs, far);
            // Words: rows 0..3 (low) and 4..7 (high) of two columns.
            var t0 = Avx2.UnpackLow(y0, y1).AsUInt32();
            var t1 = Avx2.UnpackHigh(y0, y1).AsUInt32();
            var t2 = Avx2.UnpackLow(y2, y3).AsUInt32();
            var t3 = Avx2.UnpackHigh(y2, y3).AsUInt32();
            var t4 = Avx2.UnpackLow(y4, y5).AsUInt32();
            var t5 = Avx2.UnpackHigh(y4, y5).AsUInt32();
            var t6 = Avx2.UnpackLow(y6, y7).AsUInt32();
            var t7 = Avx2.UnpackHigh(y6, y7).AsUInt32();
            // Double words: two rows of four columns.
            var u0 = Avx2.UnpackLow(t0, t2).AsUInt64();     // rows 0,1 of columns 0..3
            var u1 = Avx2.UnpackHigh(t0, t2).AsUInt64();    // rows 2,3
            var u2 = Avx2.UnpackLow(t1, t3).AsUInt64();     // rows 4,5
            var u3 = Avx2.UnpackHigh(t1, t3).AsUInt64();    // rows 6,7
            var u4 = Avx2.UnpackLow(t4, t6).AsUInt64();     // rows 0,1 of columns 4..7
            var u5 = Avx2.UnpackHigh(t4, t6).AsUInt64();
            var u6 = Avx2.UnpackLow(t5, t7).AsUInt64();
            var u7 = Avx2.UnpackHigh(t5, t7).AsUInt64();
            // Quad words: one row of eight columns in each lane.
            Avx.Store((ulong*)d, Avx2.UnpackLow(u0, u4));
            Avx.Store((ulong*)(d + rs), Avx2.UnpackHigh(u0, u4));
            Avx.Store((ulong*)(d + 2 * rs), Avx2.UnpackLow(u1, u5));
            Avx.Store((ulong*)(d + 3 * rs), Avx2.UnpackHigh(u1, u5));
            byte* d4 = d + 4 * rs;
            Avx.Store((ulong*)d4, Avx2.UnpackLow(u2, u6));
            Avx.Store((ulong*)(d4 + rs), Avx2.UnpackHigh(u2, u6));
            Avx.Store((ulong*)(d4 + 2 * rs), Avx2.UnpackLow(u3, u7));
            Avx.Store((ulong*)(d4 + 3 * rs), Avx2.UnpackHigh(u3, u7));
        }

        /// <summary>Copies 16 rows of 32 columns of 1-byte elements.</summary>
        /// <param name="p">Element (row 0, column 0) of the group in the source.</param>
        /// <param name="cs">Byte distance between two source columns.</param>
        /// <param name="d">Element (row 0, column 0) of the group in the destination.</param>
        /// <param name="rs">Byte distance between two destination rows.</param>
        /// <remarks>Not inlined: the group needs every vector register, and a call for 512 elements is not seen.</remarks>
        private static unsafe void TransposeGroup1(byte* p, long cs, byte* d, long rs)
        {
            long far = 16 * cs;
            // Bytes: a[2k] holds rows 0..7 of columns 2k and 2k+1 (in the high lane: of the columns 16 further),
            // a[2k+1] rows 8..15.
            var c0 = TransposePair1(p, far);
            var c1 = TransposePair1(p + cs, far);
            var a0 = Avx2.UnpackLow(c0, c1).AsUInt16();
            var a1 = Avx2.UnpackHigh(c0, c1).AsUInt16();
            c0 = TransposePair1(p + 2 * cs, far);
            c1 = TransposePair1(p + 3 * cs, far);
            var a2 = Avx2.UnpackLow(c0, c1).AsUInt16();
            var a3 = Avx2.UnpackHigh(c0, c1).AsUInt16();
            byte* q = p + 4 * cs;
            c0 = TransposePair1(q, far);
            c1 = TransposePair1(q + cs, far);
            var a4 = Avx2.UnpackLow(c0, c1).AsUInt16();
            var a5 = Avx2.UnpackHigh(c0, c1).AsUInt16();
            c0 = TransposePair1(q + 2 * cs, far);
            c1 = TransposePair1(q + 3 * cs, far);
            var a6 = Avx2.UnpackLow(c0, c1).AsUInt16();
            var a7 = Avx2.UnpackHigh(c0, c1).AsUInt16();
            q = p + 8 * cs;
            c0 = TransposePair1(q, far);
            c1 = TransposePair1(q + cs, far);
            var a8 = Avx2.UnpackLow(c0, c1).AsUInt16();
            var a9 = Avx2.UnpackHigh(c0, c1).AsUInt16();
            c0 = TransposePair1(q + 2 * cs, far);
            c1 = TransposePair1(q + 3 * cs, far);
            var a10 = Avx2.UnpackLow(c0, c1).AsUInt16();
            var a11 = Avx2.UnpackHigh(c0, c1).AsUInt16();
            q = p + 12 * cs;
            c0 = TransposePair1(q, far);
            c1 = TransposePair1(q + cs, far);
            var a12 = Avx2.UnpackLow(c0, c1).AsUInt16();
            var a13 = Avx2.UnpackHigh(c0, c1).AsUInt16();
            c0 = TransposePair1(q + 2 * cs, far);
            c1 = TransposePair1(q + 3 * cs, far);
            var a14 = Avx2.UnpackLow(c0, c1).AsUInt16();
            var a15 = Avx2.UnpackHigh(c0, c1).AsUInt16();

            // Words: b[4g + q] holds rows 4q..4q+3 of columns 4g..4g+3.
            var b0 = Avx2.UnpackLow(a0, a2).AsUInt32();
            var b1 = Avx2.UnpackHigh(a0, a2).AsUInt32();
            var b2 = Avx2.UnpackLow(a1, a3).AsUInt32();
            var b3 = Avx2.UnpackHigh(a1, a3).AsUInt32();
            var b4 = Avx2.UnpackLow(a4, a6).AsUInt32();
            var b5 = Avx2.UnpackHigh(a4, a6).AsUInt32();
            var b6 = Avx2.UnpackLow(a5, a7).AsUInt32();
            var b7 = Avx2.UnpackHigh(a5, a7).AsUInt32();
            var b8 = Avx2.UnpackLow(a8, a10).AsUInt32();
            var b9 = Avx2.UnpackHigh(a8, a10).AsUInt32();
            var b10 = Avx2.UnpackLow(a9, a11).AsUInt32();
            var b11 = Avx2.UnpackHigh(a9, a11).AsUInt32();
            var b12 = Avx2.UnpackLow(a12, a14).AsUInt32();
            var b13 = Avx2.UnpackHigh(a12, a14).AsUInt32();
            var b14 = Avx2.UnpackLow(a13, a15).AsUInt32();
            var b15 = Avx2.UnpackHigh(a13, a15).AsUInt32();

            // Double words: e[8h + m] holds rows 2m and 2m+1 of columns 8h..8h+7.
            var e0 = Avx2.UnpackLow(b0, b4).AsUInt64();
            var e1 = Avx2.UnpackHigh(b0, b4).AsUInt64();
            var e2 = Avx2.UnpackLow(b1, b5).AsUInt64();
            var e3 = Avx2.UnpackHigh(b1, b5).AsUInt64();
            var e4 = Avx2.UnpackLow(b2, b6).AsUInt64();
            var e5 = Avx2.UnpackHigh(b2, b6).AsUInt64();
            var e6 = Avx2.UnpackLow(b3, b7).AsUInt64();
            var e7 = Avx2.UnpackHigh(b3, b7).AsUInt64();
            var e8 = Avx2.UnpackLow(b8, b12).AsUInt64();
            var e9 = Avx2.UnpackHigh(b8, b12).AsUInt64();
            var e10 = Avx2.UnpackLow(b9, b13).AsUInt64();
            var e11 = Avx2.UnpackHigh(b9, b13).AsUInt64();
            var e12 = Avx2.UnpackLow(b10, b14).AsUInt64();
            var e13 = Avx2.UnpackHigh(b10, b14).AsUInt64();
            var e14 = Avx2.UnpackLow(b11, b15).AsUInt64();
            var e15 = Avx2.UnpackHigh(b11, b15).AsUInt64();

            // Quad words: one row of 16 columns in each lane, 32 columns a register.
            Avx.Store((ulong*)d, Avx2.UnpackLow(e0, e8));
            Avx.Store((ulong*)(d + rs), Avx2.UnpackHigh(e0, e8));
            Avx.Store((ulong*)(d + 2 * rs), Avx2.UnpackLow(e1, e9));
            Avx.Store((ulong*)(d + 3 * rs), Avx2.UnpackHigh(e1, e9));
            byte* dq = d + 4 * rs;
            Avx.Store((ulong*)dq, Avx2.UnpackLow(e2, e10));
            Avx.Store((ulong*)(dq + rs), Avx2.UnpackHigh(e2, e10));
            Avx.Store((ulong*)(dq + 2 * rs), Avx2.UnpackLow(e3, e11));
            Avx.Store((ulong*)(dq + 3 * rs), Avx2.UnpackHigh(e3, e11));
            dq = d + 8 * rs;
            Avx.Store((ulong*)dq, Avx2.UnpackLow(e4, e12));
            Avx.Store((ulong*)(dq + rs), Avx2.UnpackHigh(e4, e12));
            Avx.Store((ulong*)(dq + 2 * rs), Avx2.UnpackLow(e5, e13));
            Avx.Store((ulong*)(dq + 3 * rs), Avx2.UnpackHigh(e5, e13));
            dq = d + 12 * rs;
            Avx.Store((ulong*)dq, Avx2.UnpackLow(e6, e14));
            Avx.Store((ulong*)(dq + rs), Avx2.UnpackHigh(e6, e14));
            Avx.Store((ulong*)(dq + 2 * rs), Avx2.UnpackLow(e7, e15));
            Avx.Store((ulong*)(dq + 3 * rs), Avx2.UnpackHigh(e7, e15));
        }
    }
}
