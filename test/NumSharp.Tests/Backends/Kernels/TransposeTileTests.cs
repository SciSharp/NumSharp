using System;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using AwesomeAssertions;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     <see cref="DirectILKernelGenerator.TransposeTile"/>: a tile of a matrix whose columns are contiguous is
    ///     copied into rows that are contiguous, for every element width, tile shape, column stride and row direction.
    /// </summary>
    /// <remarks>
    ///     <para>The kernel copies vector groups (2 x 4 elements of 8 bytes up to 16 x 32 elements of one byte) and
    ///     ends a tile that is not a whole number of groups with a group that overlaps the one before it. The shapes of
    ///     these tests sit on both sides of every group size in both directions, so each tile is copied by whole
    ///     groups, by an overlapped last group, or (below one group) by the move fallback.</para>
    ///     <para><b>Two kinds of error are looked for.</b> A wrong element: every source byte is different from its
    ///     neighbours' (a hash of its address), so a transposed pair, a swapped lane or a repeated row shows. A touch
    ///     outside the tile: the destination's margins hold a sentinel, and in the fenced test the source and the
    ///     destination lie against inaccessible pages, so one byte read or written outside ends the test host.</para>
    /// </remarks>
    [TestClass]
    public unsafe class TransposeTileTests : TestClass
    {
        /// <summary>The element widths the kernel serves.</summary>
        private static readonly int[] Sizes = { 1, 2, 4, 8, 16 };

        /// <summary>Tile heights: around the row groups 2, 4, 8 and 16, and two groups plus a rest.</summary>
        private static readonly int[] RowCounts = { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40 };

        /// <summary>Tile widths: around the column groups 4, 8, 16 and 32, and two groups plus a rest.</summary>
        private static readonly int[] ColumnCounts = { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 70 };

        /// <summary>The byte the destination's margins hold before a call.</summary>
        private const byte Sentinel = 0xEE;

        /// <summary>A byte that depends on every bit of its position, so that no two neighbours are equal.</summary>
        /// <param name="i">The position.</param>
        /// <returns>The byte.</returns>
        private static byte Noise(long i)
        {
            ulong h = ((ulong)i + 1) * 0x9E3779B97F4A7C15UL;
            h ^= h >> 31;
            h *= 0xBF58476D1CE4E5B9UL;
            return (byte)(h >> 56);
        }

        /// <summary>
        ///     Asserts that the destination holds the transposed tile and that its margins are untouched.
        /// </summary>
        /// <param name="what">Label for the failure message.</param>
        /// <param name="src">Element (row 0, column 0) of the tile in the source.</param>
        /// <param name="cs">Source column stride in bytes.</param>
        /// <param name="area">Start of the destination area (margin, tile, margin).</param>
        /// <param name="margin">Bytes of each margin.</param>
        /// <param name="rows">Rows of the tile.</param>
        /// <param name="cols">Columns of the tile.</param>
        /// <param name="size">Element width.</param>
        /// <param name="rowsDownwards">The rows were stored from the tile's last row slot down.</param>
        private static void AssertTile(string what, byte* src, long cs, byte* area, int margin, int rows, int cols, int size,
            bool rowsDownwards)
        {
            long rowBytes = (long)cols * size;
            byte* tile = area + margin;
            for (long b = 0; b < rows; b++)
            {
                // Logical row b lies in slot b, or in slot rows-1-b when the rows were stored downwards.
                byte* row = tile + (rowsDownwards ? rows - 1 - b : b) * rowBytes;
                for (long w = 0; w < cols; w++)
                {
                    byte* want = src + w * cs + b * size;
                    for (int k = 0; k < size; k++)
                    {
                        if (row[w * size + k] != want[k])
                            Assert.Fail($"{what}: element [{b},{w}] byte {k} is {row[w * size + k]}, the source holds {want[k]}");
                    }
                }
            }
            long tileBytes = rows * rowBytes;
            var below = new ReadOnlySpan<byte>(area, margin);
            var above = new ReadOnlySpan<byte>(tile + tileBytes, margin);
            if (below.IndexOfAnyExcept(Sentinel) >= 0 || above.IndexOfAnyExcept(Sentinel) >= 0)
                Assert.Fail($"{what}: a byte outside the tile was written");
        }

        /// <summary>
        ///     Every width, tile shape and column stride (the columns directly after each other, columns with a gap,
        ///     columns in descending order), with the rows stored upwards and downwards: the destination holds
        ///     exactly the transposed tile.
        /// </summary>
        [TestMethod]
        public void EveryTileShape_IsCopiedExactly()
        {
            const int Margin = 64;
            const int SourceBytes = 1 << 18, AreaBytes = 1 << 16;
            byte* source = (byte*)NativeMemory.Alloc(SourceBytes);
            byte* area = (byte*)NativeMemory.Alloc(AreaBytes);
            try
            {
                for (int i = 0; i < SourceBytes; i++)
                    source[i] = Noise(i);

                foreach (int size in Sizes)
                foreach (int rows in RowCounts)
                foreach (int cols in ColumnCounts)
                {
                    long piece = (long)rows * size;
                    // Column strides: an F-ordered matrix of exactly `rows` rows, one of more rows (a gap after
                    // each column piece), and the same with the columns in descending order.
                    foreach (long cs in new[] { piece, piece + 3 * size, -(piece + size) })
                    foreach (bool downwards in new[] { false, true })
                    {
                        long span = (cols - 1) * System.Math.Abs(cs) + piece;
                        byte* src = cs > 0 ? source + 128 : source + 128 + (cols - 1) * -cs;
                        (128 + span).Should().BeLessThanOrEqualTo(SourceBytes);
                        long rowBytes = (long)cols * size;
                        (2 * Margin + rows * rowBytes).Should().BeLessThanOrEqualTo(AreaBytes);

                        new Span<byte>(area, (int)(2 * Margin + rows * rowBytes)).Fill(Sentinel);
                        byte* tile = area + Margin;
                        if (downwards)
                            DirectILKernelGenerator.TransposeTile(src, cs, tile + (rows - 1) * rowBytes, -rowBytes, rows, cols, size);
                        else
                            DirectILKernelGenerator.TransposeTile(src, cs, tile, rowBytes, rows, cols, size);
                        AssertTile($"size {size} tile {rows}x{cols} column stride {cs} downwards {downwards}", src, cs, area, Margin, rows,
                            cols, size, downwards);
                    }
                }
            }
            finally
            {
                NativeMemory.Free(source);
                NativeMemory.Free(area);
            }
        }

        /// <summary>
        ///     The tile's source and destination lie against inaccessible pages (the first byte directly above one, or
        ///     the last byte directly below one): the kernel reads and writes no byte outside the tile.
        /// </summary>
        /// <remarks>
        ///     A value test cannot see a read outside the tile, and margins see writes only. Windows only (the fences
        ///     are reserved pages); Inconclusive elsewhere.
        /// </remarks>
        [TestMethod]
        public void Tile_BetweenInaccessiblePages_TouchesNoByteOutsideIt()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("Fenced memory needs VirtualAlloc (Windows).");

            using var sourceFence = new FencedMemory(pages: 24);
            using var destinationFence = new FencedMemory(pages: 12);
            for (long i = 0; i < sourceFence.Bytes; i++)
                sourceFence.Start[i] = Noise(i);

            foreach (int size in Sizes)
            foreach (int rows in RowCounts)
            foreach (int cols in ColumnCounts)
            {
                long piece = (long)rows * size, rowBytes = (long)cols * size;
                foreach (long cs in new[] { piece, piece + size, -piece })
                foreach (bool atEnd in new[] { false, true })
                {
                    long span = (cols - 1) * System.Math.Abs(cs) + piece;
                    if (span > sourceFence.Bytes || rows * rowBytes > destinationFence.Bytes)
                        continue;
                    byte* low = sourceFence.Place(span, atEnd);
                    byte* src = cs > 0 ? low : low + (cols - 1) * -cs;
                    // The destination is placed against the OTHER edge of its fence, upwards and downwards in turn.
                    byte* tile = destinationFence.Place(rows * rowBytes, !atEnd);
                    bool downwards = ((rows + cols) & 1) == 1;
                    new Span<byte>(tile, (int)(rows * rowBytes)).Fill(Sentinel);
                    if (downwards)
                        DirectILKernelGenerator.TransposeTile(src, cs, tile + (rows - 1) * rowBytes, -rowBytes, rows, cols, size);
                    else
                        DirectILKernelGenerator.TransposeTile(src, cs, tile, rowBytes, rows, cols, size);

                    for (long b = 0; b < rows; b++)
                    {
                        byte* row = tile + (downwards ? rows - 1 - b : b) * rowBytes;
                        for (long w = 0; w < cols; w++)
                        {
                            if (!new ReadOnlySpan<byte>(row + w * size, size).SequenceEqual(new ReadOnlySpan<byte>(src + w * cs + b * size, size)))
                                Assert.Fail($"size {size} tile {rows}x{cols} column stride {cs} atEnd {atEnd}: element [{b},{w}] differs");
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     The group sizes the kernel reports are the ones of this host's instruction set, and a tile of exactly
        ///     one group, of one group less a row or a column, and of an empty extent is handled.
        /// </summary>
        [TestMethod]
        public void Groups_FollowTheInstructionSet_AndDegenerateTilesAreSafe()
        {
            foreach (int size in Sizes)
            {
                int rowGroup = DirectILKernelGenerator.TransposeTileRowGroup(size);
                int columnGroup = DirectILKernelGenerator.TransposeTileColumnGroup(size);
                if (Avx2.IsSupported && size <= 8)
                {
                    (rowGroup * size).Should().Be(16, $"a row group of {size}-byte elements is one 16-byte column piece");
                    (columnGroup * size).Should().Be(32, $"a column group of {size}-byte elements fills one 32-byte row piece");
                }
                else
                {
                    rowGroup.Should().Be(1);
                    columnGroup.Should().Be(1);
                }
            }
            DirectILKernelGenerator.TransposeTileRowGroup(3).Should().Be(1, "an unknown width has no vector group");

            // Nothing is read or written for an empty tile: the pointers are not valid addresses.
            DirectILKernelGenerator.TransposeTile((byte*)1, 8, (byte*)1, 8, 0, 5, 8);
            DirectILKernelGenerator.TransposeTile((byte*)1, 8, (byte*)1, 8, 5, 0, 8);
            DirectILKernelGenerator.TransposeTile((byte*)1, 8, (byte*)1, 8, -1, -1, 1);

            Action unknownWidth = () =>
            {
                byte* local = stackalloc byte[64];
                DirectILKernelGenerator.TransposeTile(local, 3, local + 32, 3, 1, 1, 3);
            };
            unknownWidth.Should().Throw<NotSupportedException>();
        }
    }
}
