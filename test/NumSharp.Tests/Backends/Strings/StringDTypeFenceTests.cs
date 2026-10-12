using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The <c>StringDType</c> routes read and write NO byte outside an operand's cells (strings phase 1, P1.9): every
    ///     source array of these tests lays its 16-byte cells over <see cref="FencedMemory"/>, with the first cell directly
    ///     above an inaccessible page or the last cell directly below one, and runs the copies, gathers, casts and
    ///     <c>tobytes()</c> of contiguous, stepped and reversed views.
    /// </summary>
    /// <remarks>
    ///     <para>A violation is not a failed assertion: the test host ends ("Test host process crashed") with the stack of the
    ///     route that left the cells. The values are checked too. Windows only (the fences are reserved pages);
    ///     Inconclusive elsewhere.</para>
    ///     <para>The cells' block does not own the allocator here (it wraps foreign memory), so the test frees the arena and
    ///     the heap strings itself with the allocator's clear loop.</para>
    /// </remarks>
    [TestClass]
    public unsafe class StringDTypeFenceTests
    {
        /// <summary>The texts written into the fenced cells: short (in the cell), medium (arena) and long (arena, 8-byte
        ///     prefix) strings, an empty one and multi-byte UTF-8.</summary>
        /// <param name="i">The element index.</param>
        /// <returns>The text of element <paramref name="i"/>.</returns>
        private static string Text(int i) => (i % 5) switch
        {
            0 => "s" + i,
            1 => new string('m', 20) + i,
            2 => new string('l', 300) + i,
            3 => "",
            _ => "é\U0001F600" + i,
        };

        /// <summary>
        ///     Builds a <c>StringDType</c> array of <paramref name="n"/> elements whose cells lie in fenced memory, runs
        ///     <paramref name="check"/> on it, then frees the strings.
        /// </summary>
        /// <param name="n">The element count.</param>
        /// <param name="atEnd">True: the last cell ends at the upper fence; false: the first cell starts at the lower one.</param>
        /// <param name="check">The routes to run on the fenced array.</param>
        private static void WithFencedArray(int n, bool atEnd, Action<NDArray> check)
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            using var fence = new FencedMemory(pages: (n * VStringAllocator.CellSize + FencedMemory.PageBytes - 1) / FencedMemory.PageBytes + 1);
            byte* cells = fence.Place(n * (long)VStringAllocator.CellSize, atEnd);
            new Span<byte>(cells, n * VStringAllocator.CellSize).Clear(); // all-zero cells are empty strings
            var allocator = new VStringAllocator();
            var descr = np.dtypes.StringDType();
            var slice = new VStringArraySlice(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(cells, n * (long)VStringAllocator.CellSize)), allocator, descr);
            try
            {
                for (int i = 0; i < n; i++)
                    slice.SetIndex(i, Text(i));
                var t = new NDArray(new UnmanagedStorage(slice, new Shape(n)));
                check(t);
            }
            finally
            {
                allocator.ClearAndFree(cells, n);
            }
        }

        /// <summary>
        ///     Copies, gathers, casts and <c>tobytes()</c> over fenced cells, for lengths around the 128-element blocks of the
        ///     casts, with the cells against either fence; every result is checked against the texts written.
        /// </summary>
        [TestMethod]
        public void Routes_StayInsideTheCells()
        {
            foreach (int n in new[] { 1, 2, 7, 127, 128, 129, 300 })
            foreach (bool atEnd in new[] { true, false })
                WithFencedArray(n, atEnd, t =>
                {
                    var expected = Enumerable.Range(0, n).Select(Text).ToArray();
                    using (var c = t.copy())
                        Enumerable.Range(0, n).Select(i => (string)c.GetValue(i)).Should().Equal(expected);
                    using (var r = t["::-1"].copy())
                        Enumerable.Range(0, n).Select(i => (string)r.GetValue(i)).Should().Equal(Enumerable.Reverse(expected));
                    using (var s = t["1::2"].copy())
                        Enumerable.Range(0, (int)s.size).Select(i => (string)s.GetValue(i)).Should().Equal(expected.Where((_, i) => i % 2 == 1));
                    using (var idx = np.array(Enumerable.Range(0, n).Reverse().Select(i => (long)i).ToArray()))
                    using (var g = np.take(t, idx))
                        Enumerable.Range(0, n).Select(i => (string)g.GetValue(i)).Should().Equal(Enumerable.Reverse(expected));
                    using (var u = t.astype(np.dtype("U400")))
                        NDArray.AsStringArray(u).Should().Equal(expected);
                    using (var mask = np.array(Enumerable.Range(0, n).Select(i => i % 3 != 0).ToArray()))
                    using (var m = t[mask])
                        Enumerable.Range(0, (int)m.size).Select(i => (string)m.GetValue(i)).Should().Equal(expected.Where((_, i) => i % 3 != 0));
                    t.tobytes().Length.Should().Be(n * VStringAllocator.CellSize);
                    t["::-1"].tobytes().Length.Should().Be(n * VStringAllocator.CellSize);
                    using (var back = np.array(new[] { "x" }, np.dtypes.StringDType()))
                    {
                        // a write INTO the fenced cells through the copy core (broadcast), then read back
                        t[":"] = back;
                        Enumerable.Range(0, n).Select(i => (string)t.GetValue(i)).Should().AllBe("x");
                    }
                });
        }
    }
}
