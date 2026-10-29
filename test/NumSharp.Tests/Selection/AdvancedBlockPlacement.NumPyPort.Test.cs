using System;
using AwesomeAssertions;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     Where an advanced index's block of dimensions goes, as NumPy's <c>mapiter_fill_info</c> decides: in place when the
    ///     advanced entries (index arrays, masks, 0-d bools and the integers beside them) are consecutive, in FRONT when a
    ///     slice, a new axis or the ellipsis stands between them — the ellipsis whatever it spans, none included. Expected
    ///     outcomes probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Found by a 6,000-key differential of the getter and setter against NumPy (P3 batch 8, np-layer dtype plan): an
    ///     ellipsis spanning no axis left no entry once expanded, so the routes kept the block in place; the setter's
    ///     single-index-array route ignored the integers, so <c>a[0, :, idx] = v</c> expected the block in place too.
    /// </remarks>
    [TestClass]
    public class advanced_block_placement_numpy_port_tests
    {
        /// <summary>A float64 array of <paramref name="dims"/> holding 0, 1, 2, ... in C order.</summary>
        /// <param name="dims">The shape.</param>
        /// <returns>The array.</returns>
        private static NDArray Ramp(params long[] dims)
        {
            long size = 1;
            foreach (var d in dims)
                size *= d;
            return np.arange((double)size).reshape(dims);
        }

        /// <summary>An ellipsis that spans no axis still separates two advanced entries: the block goes in front.</summary>
        [TestMethod]
        public void EllipsisSpanningNoAxis_SeparatesTheAdvancedEntries()
        {
            Ramp(3, 4)["1:3", np.array(new long[0]), Slice.Ellipsis, np.array(true)].shape.Should().Equal(0L, 2L);
            var column = Ramp(3, 4)[Slice.All, -1, Slice.Ellipsis, np.array(true)];
            column.shape.Should().Equal(1L, 3L);
            column.flatten().ToArray<double>().Should().Equal(3, 7, 11);
            Ramp(2, 3, 4)["1:3", np.array(new long[,] { { 0, 1 }, { 1, 0 } }), Slice.Ellipsis, -1].shape.Should().Equal(2L, 2L, 1L);
            Ramp(3, 4)[Slice.All, true, Slice.Ellipsis, np.array(new long[,] { { 0, 1 }, { 1, 0 } })].shape.Should().Equal(2L, 2L, 3L);
        }

        /// <summary>
        ///     A 0-d array indexed by two 0-d bools around an ellipsis: the block alone, (0,) for any False (it raised
        ///     IndexOutOfRangeException once the block was found in front), and a store checked against it.
        /// </summary>
        [TestMethod]
        public void ZeroDArray_TwoBoolsAroundAnEllipsis()
        {
            np.array(5.0)[false, Slice.Ellipsis, true].shape.Should().Equal(0L);
            np.array(5.0)[true, Slice.Ellipsis, true].shape.Should().Equal(1L);
            var x = np.array(5.0);
            x[true, Slice.Ellipsis, true] = np.array(7.0);
            x.GetDouble().Should().Be(7.0);
        }

        /// <summary>An integer, a slice and an index array: the block goes in front, for the setter as for the getter.</summary>
        [TestMethod]
        public void IntegerSliceIndexArray_StoresTheBlockInFront()
        {
            var a = Ramp(2, 3, 4, 5);
            a[0, Slice.All, np.array(new long[0])].shape.Should().Equal(0L, 3L, 5L);
            a[0, Slice.All, np.array(new long[0])] = np.zeros(new Shape(0, 3, 5));

            var b = Ramp(2, 3, 4, 5);
            var key = new object[] { 0, Slice.Ellipsis, "1:3", np.array(new long[] { 0, 1 }) };
            b[key].shape.Should().Equal(2L, 3L, 2L);
            b[key] = np.full(new Shape(2, 3, 2), -1.0);
            b[key].flatten().ToArray<double>().Should().OnlyContain(v => v == -1.0);
            b["1"].flatten().ToArray<double>().Should().Equal(Ramp(2, 3, 4, 5)["1"].flatten().ToArray<double>());
        }
    }
}
