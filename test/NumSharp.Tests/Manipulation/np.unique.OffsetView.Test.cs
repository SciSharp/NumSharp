using System.Linq;
using NumSharp.Backends;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     <c>np.unique</c> over a C-contiguous VIEW whose shape carries a non-zero offset into its storage (the layout of
    ///     a view rebuilt over a buffer, as the oracle's replay and <c>np.ndarray(buffer=, offset=)</c> build them). The
    ///     hash path and the key extraction of the <c>return_index</c> / <c>return_inverse</c> / <c>return_counts</c> path
    ///     read from the storage's base address and dropped the offset, so they took the elements BEFORE the view and lost
    ///     its last ones. Expected values probed against NumPy 2.4.2: <c>buf = [5, 3, 9, 3, 1, 9, 7]; np.unique(buf[2:6],
    ///     return_index=True, return_inverse=True, return_counts=True)</c> is <c>([1, 3, 9], [2, 1, 0], [2, 1, 0, 2],
    ///     [1, 1, 2])</c>.
    /// </summary>
    [TestClass]
    public class NpUniqueOffsetViewTest
    {
        /// <summary>
        ///     The four elements <c>[9, 3, 1, 9]</c> of <c>[5, 3, 9, 3, 1, 9, 7]</c> in <paramref name="dtype"/>, as a
        ///     contiguous view that keeps the storage's base address and an offset of 2 (not a re-seated slice).
        /// </summary>
        private static NDArray OffsetView(NPTypeCode dtype)
        {
            var full = np.array(new long[] { 5, 3, 9, 3, 1, 9, 7 }).astype(dtype);
            var view = new NDArray(full.Storage, new Shape(new long[] { 4 }, new long[] { 1 }, 2, 7));
            view.Shape.IsContiguous.Should().BeTrue();
            view.Shape.offset.Should().Be(2);
            return view;
        }

        private static readonly NPTypeCode[] Numeric =
        {
            NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32,
            NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double,
            NPTypeCode.Decimal, NPTypeCode.Complex,
        };

        [TestMethod]
        public void Unique_OfAnOffsetView_ReadsTheViewsElements()
        {
            foreach (var dtype in Numeric)
            {
                var r = np.unique(OffsetView(dtype)).values;
                r.astype(NPTypeCode.Int64).ToArray<long>().Should().Equal(new long[] { 1, 3, 9 }, dtype.ToString());
            }
        }

        [TestMethod]
        public void Unique_WithEveryOutput_OfAnOffsetView()
        {
            foreach (var dtype in Numeric)
            {
                var r = np.unique(OffsetView(dtype), return_index: true, return_inverse: true, return_counts: true);
                r.values.astype(NPTypeCode.Int64).ToArray<long>().Should().Equal(new long[] { 1, 3, 9 }, dtype.ToString());
                r.indices.ToArray<long>().Should().Equal(new long[] { 2, 1, 0 }, dtype.ToString());
                r.inverse_indices.ToArray<long>().Should().Equal(new long[] { 2, 1, 0, 2 }, dtype.ToString());
                r.counts.ToArray<long>().Should().Equal(new long[] { 1, 1, 2 }, dtype.ToString());
            }
        }

        [TestMethod]
        public void Unique_OfABooleanOffsetView()
        {
            // buf = [T, F, F, T, F, T, T]; np.unique(buf[2:6], return_index=True, return_inverse=True, return_counts=True)
            // -> ([False, True], [0, 1], [0, 1, 0, 1], [2, 2]). The base-address read gave index [1, 0], inverse [1, 0, 0, 1].
            var full = np.array(new[] { true, false, false, true, false, true, true });
            var view = new NDArray(full.Storage, new Shape(new long[] { 4 }, new long[] { 1 }, 2, 7));
            var r = np.unique(view, return_index: true, return_inverse: true, return_counts: true);
            r.values.ToArray<bool>().Should().Equal(false, true);
            r.indices.ToArray<long>().Should().Equal(0L, 1L);
            r.inverse_indices.ToArray<long>().Should().Equal(0L, 1L, 0L, 1L);
            r.counts.ToArray<long>().Should().Equal(2L, 2L);
        }

        /// <summary>
        ///     The row path of an axis over an <c>np.split</c> child (C-contiguous, at an offset into its parent): the rows
        ///     compared and gathered are the child's own. The slab helpers read from the storage's base address, so
        ///     <c>np.unique(np.split(a, 3)[1], axis=0)</c> returned the parent's first two rows. Expected values probed against
        ///     NumPy 2.4.2: <c>np.unique(np.split(np.arange(24).reshape(6, 4) % 3, 3)[1], axis=0, return_index=True,
        ///     return_inverse=True, return_counts=True)</c> is <c>([[0, 1, 2, 0], [2, 0, 1, 2]], [1, 0], [1, 0], [1, 1])</c>.
        /// </summary>
        /// <exception cref="System.Exception">The assertion framework's failure exception, when a dtype's outputs are not the
        ///     child's.</exception>
        [TestMethod]
        public void UniqueAxis_OfASplitChild_ReadsTheChildsRows()
        {
            foreach (var dtype in Numeric.Append(NPTypeCode.Boolean))
            {
                var parts = np.split((np.arange(24).reshape(6, 4) % 3).astype(dtype), 3);
                parts[1].Shape.IsContiguous.Should().BeTrue();
                parts[1].Shape.offset.Should().Be(8, "an np.split child keeps the base address and an offset");
                var r = np.unique(parts[1], return_index: true, return_inverse: true, return_counts: true, axis: 0);
                if (dtype != NPTypeCode.Boolean)
                    r.values.astype(NPTypeCode.Int64).ToArray<long>().Should().Equal(new long[] { 0, 1, 2, 0, 2, 0, 1, 2 }, dtype.ToString());
                r.indices.ToArray<long>().Should().Equal(new long[] { 1, 0 }, dtype.ToString());
                r.inverse_indices.ToArray<long>().Should().Equal(new long[] { 1, 0 }, dtype.ToString());
                r.counts.ToArray<long>().Should().Equal(new long[] { 1, 1 }, dtype.ToString());

                // Every child and both axes: the view answers as its offset-free copy does.
                for (int k = 0; k < 3; k++)
                    for (int axis = 0; axis < 2; axis++)
                    {
                        var view = np.unique(parts[k], return_index: true, return_inverse: true, return_counts: true, axis: axis);
                        var copy = np.unique(parts[k].copy(), return_index: true, return_inverse: true, return_counts: true, axis: axis);
                        for (int o = 0; o < 4; o++)
                            view[o].array_equal(copy[o]).Should().BeTrue($"{dtype} part {k} axis {axis} output {o}");
                    }
            }
        }

        /// <summary>
        ///     The same for a 3-D child (the rows of axis 0 are 2 x 4 slabs), float64 and complex128. Every row differs from
        ///     the parent's first rows (a read from the base address would see other rows), and the second child holds a
        ///     repeated row (a count of 2).
        /// </summary>
        /// <exception cref="System.Exception">The assertion framework's failure exception, when an output is not the
        ///     offset-free copy's.</exception>
        [TestMethod]
        public void UniqueAxis_Of3DSplitChildren_MatchTheirCopies()
        {
            foreach (var dtype in new[] { NPTypeCode.Double, NPTypeCode.Complex })
            {
                var data = (np.arange(48).reshape(6, 2, 4) % 7).astype(dtype);
                data["3"] = data["2"].copy();   // the second child's two rows are equal
                var parts = np.split(data, 3);
                for (int k = 1; k < 3; k++)
                {
                    parts[k].Shape.offset.Should().BeGreaterThan(0);
                    var view = np.unique(parts[k], return_index: true, return_inverse: true, return_counts: true, axis: 0);
                    var copy = np.unique(parts[k].copy(), return_index: true, return_inverse: true, return_counts: true, axis: 0);
                    for (int o = 0; o < 4; o++)
                        view[o].array_equal(copy[o]).Should().BeTrue($"{dtype} part {k} output {o}");
                }
            }
        }

        /// <summary>
        ///     The long-indexed complex fallback (reached above <see cref="System.Array.MaxLength"/> elements, called here by
        ///     reflection) reads a C-contiguous view at an offset from its first element: it read from the base address.
        /// </summary>
        /// <exception cref="System.Exception">The assertion framework's failure exception, when an output is not the managed
        ///     path's over the offset-free copy.</exception>
        [TestMethod]
        public void UniqueLongComplexPath_OfAnOffsetView_ReadsTheViewsElements()
        {
            var full = np.array(new System.Numerics.Complex[] { new(9, 0), new(8, 0), new(7, 0), new(3, 1), new(1, 0), new(3, 1) });
            var view = np.split(full, 2)[1];   // [3+1j, 1+0j, 3+1j] at offset 3
            view.Shape.offset.Should().Be(3);
            var method = typeof(DefaultEngine)
                .GetMethods(System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                .First(m => m.Name == "uniqueFlatSortedLongComplex");
            // signature: (NDArray a, long n, bool equal_nan, bool return_index, bool return_inverse, bool return_counts)
            var r = (NDArray[])method.Invoke(BackendFactory.GetEngine<DefaultEngine>(), new object[] { view, view.size, true, true, true, true });
            var expected = np.unique(view.copy(), return_index: true, return_inverse: true, return_counts: true).ToArray();
            for (int o = 0; o < 4; o++)
                r[o].array_equal(expected[o]).Should().BeTrue($"output {o}");
        }
    }
}
