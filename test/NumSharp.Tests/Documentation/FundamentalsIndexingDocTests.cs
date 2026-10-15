using System;
using NumSharp;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for every code example in
    ///     <c>docs/website-src/docs/fundamentals/indexing.md</c> (the "Indexing on NDArray"
    ///     fundamentals article). Each test mirrors one documented snippet and asserts the behaviour
    ///     the page claims. Values were captured by running the snippets against this branch.
    /// </summary>
    [TestClass]
    public class FundamentalsIndexingDocTests
    {
        // ── Basic indexing ─────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Basic_SingleElement_NegativeAndCoordinate()
        {
            var x = np.arange(10);
            ((long)x[2]).Should().Be(2L);
            ((long)x[-2]).Should().Be(8L);

            x = x.reshape(2, 5);
            ((long)x[1, 3]).Should().Be(8L);
            ((long)x[1, -1]).Should().Be(9L);
        }

        [TestMethod]
        public void Basic_PartialIndex_ReturnsSubArrayView()
        {
            var x = np.arange(10).reshape(2, 5);
            var row = x[0];
            row.shape.Should().Equal(new long[] { 5 });
            ((long)row[2]).Should().Be(2L);
            // x[0, 2] == x[0][2]
            ((long)x[0, 2]).Should().Be((long)x[0][2]);
        }

        [TestMethod]
        public void Basic_SliceAndStride()
        {
            var x = np.arange(10);
            x["1:7:2"].ToArray<long>().Should().Equal(1L, 3, 5);
            x["-2:10"].ToArray<long>().Should().Equal(8L, 9);
            x["-3:3:-1"].ToArray<long>().Should().Equal(7L, 6, 5, 4);
            x["5:"].ToArray<long>().Should().Equal(5L, 6, 7, 8, 9);
            ((long)x["::-1"][0]).Should().Be(9L, "reversed");
        }

        [TestMethod]
        public void Basic_EllipsisAndNewaxis()
        {
            var x = np.arange(6).reshape(2, 3, 1);
            x["..., 0"].shape.Should().Equal(new long[] { 2, 3 });

            var v = np.arange(5);
            v[np.newaxis].shape.Should().Equal(new long[] { 1, 5 });
            (v[np.newaxis].T + v[np.newaxis]).shape.Should().Equal(new long[] { 5, 5 }, "outer add via newaxis");
        }

        // ── Advanced indexing ──────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Advanced_IntegerArray_NegativeAllowed()
        {
            var x = np.arange(10, 1, -1);                        // [10 9 8 7 6 5 4 3 2]
            x[np.array(new[] { 3, 3, 1, 8 })].ToArray<long>().Should().Equal(7L, 7, 9, 2);
            x[np.array(new[] { 3, 3, -3, 8 })].ToArray<long>().Should().Equal(7L, 7, 4, 2);
        }

        [TestMethod]
        public void Advanced_PairedIndexArrays_IterateAsOne()
        {
            var y = np.arange(35).reshape(5, 7);
            y[np.array(new[] { 0, 2, 4 }), np.array(new[] { 0, 1, 2 })].ToArray<long>()
              .Should().Equal(0L, 15, 30);
        }

        [TestMethod]
        public void Advanced_IxUnderscore_MakesAGrid()
        {
            var x = np.arange(12).reshape(4, 3);
            var block = x[np.ix_(np.array(new[] { 0, 3 }), np.array(new[] { 0, 2 }))];
            block.shape.Should().Equal(new long[] { 2, 2 });
            block.ToArray<long>().Should().Equal(0L, 2, 9, 11);
        }

        [TestMethod]
        public void Advanced_RawIntArraySoleIndex_IsFancy_SelectsRows()
        {
            var a = np.arange(12).reshape(3, 4);
            var rows = a[new[] { 0, 2 }];                         // FANCY — selects rows
            rows.shape.Should().Equal(new long[] { 2, 4 });
            rows.ToArray<long>().Should().Equal(0L, 1, 2, 3, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Advanced_CoordinateAccess_UsesGetData()
        {
            var a = np.arange(12).reshape(3, 4);
            // GetData takes an int[]; a.GetData(new[]{0,2}) is the element at coordinate (0,2).
            ((long)a.GetData(new[] { 0, 2 })).Should().Be(2L);
        }

        [TestMethod]
        public void Advanced_BooleanMask()
        {
            var x = np.array(new[,] { { 1.0, 2.0 }, { double.NaN, 3.0 } });
            x[!np.isnan(x)].ToArray<double>().Should().Equal(1.0, 2.0, 3.0);

            var v = np.array(new[] { 1.0, -1.0, -2.0, 3.0 });
            v[v < 0] = 20;
            v.ToArray<double>().Should().Equal(1.0, 20.0, 20.0, 3.0);
        }

        [TestMethod]
        public void Advanced_BooleanMask_FewerDims_SelectsLeadingAxes()
        {
            var x = np.arange(35).reshape(5, 7);
            var b = x > 20;
            var rows = x[b["..., 5"]];                            // rows where column 5 exceeds 20 → rows 3,4
            rows.shape.Should().Equal(new long[] { 2, 7 });
            ((long)rows[0, 0]).Should().Be(21L);
        }

        // ── Flat iteration (write-through) ─────────────────────────────────────────────────────────

        [TestMethod]
        public void Flat_Flatiter_WritesThroughAnyLayout()
        {
            var t = np.arange(6).reshape(2, 3).T;                 // transposed (non-contiguous) view
            t.flatiter[5] = 99;                                   // logical C-order flat write
            ((long)t[2, 1]).Should().Be(99L);

            var m = np.arange(6).reshape(2, 3);
            m.flatiter["::2"] = 0;
            m.ToArray<long>().Should().Equal(0L, 1, 0, 3, 0, 5);
        }

        // ── Assignment ─────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Assignment_SliceFillAndBroadcast()
        {
            var x = np.arange(10);
            x["2:7"] = 1;
            x.ToArray<long>().Should().Equal(0L, 1, 1, 1, 1, 1, 1, 7, 8, 9);

            var x2 = np.arange(10);
            x2["2:7"] = np.arange(5);
            x2.ToArray<long>().Should().Equal(0L, 1, 0, 1, 2, 3, 4, 7, 8, 9);
        }

        [TestMethod]
        public void Assignment_FancyDuplicates_LastWriteWins()
        {
            var x = np.arange(5);
            x[np.array(new[] { 1, 1 })] = np.array(new[] { 10L, 20L });
            ((long)x[1]).Should().Be(20L, "last write wins on duplicate fancy indices");
        }
    }
}
