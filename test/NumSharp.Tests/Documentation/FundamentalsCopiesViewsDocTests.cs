using System;
using NumSharp;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for every code example in
    ///     <c>docs/website-src/docs/fundamentals/copies-and-views.md</c> (the "Copies and views"
    ///     fundamentals article). Each test mirrors one documented snippet and asserts the behaviour
    ///     the page claims. Values were captured by running the snippets against this branch.
    /// </summary>
    [TestClass]
    public class FundamentalsCopiesViewsDocTests
    {
        // ── View / Copy ────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void View_SharesBuffer_WriteThrough()
        {
            var x = np.arange(10);
            var y = x["1:3"];
            x["1:3"] = np.array(new[] { 10L, 11L });
            y.ToArray<long>().Should().Equal(new[] { 10L, 11L }, "the view sees the change");
            x.ToArray<long>().Should().Equal(0L, 10, 11, 3, 4, 5, 6, 7, 8, 9);
        }

        [TestMethod]
        public void Copy_IsIndependent()
        {
            var x = np.arange(10);
            var y = x["1:3"].copy();
            y[0] = 999;
            ((long)x[1]).Should().Be(1L, "the copy does not touch the parent");
        }

        // ── Indexing: basic → view, advanced → copy ────────────────────────────────────────────────

        [TestMethod]
        public void Indexing_BasicIsView_AdvancedIsCopy()
        {
            var x = np.arange(9);
            var v = x.reshape(3, 3);                              // basic → view
            (v.@base is not null).Should().BeTrue();

            var m = np.arange(9).reshape(3, 3);
            var c = m[np.array(new[] { 1, 2 })];                 // advanced → copy
            (c.@base is not null).Should().BeFalse();
        }

        // ── Other operations ───────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Reshape_Ravel_Flatten_ViewVsCopy()
        {
            // ravel of a contiguous array is a view (write-through)
            var contig = np.arange(4);
            contig.ravel()[0] = 77L;
            ((long)contig[0]).Should().Be(77L, "ravel of a contiguous array is a view");

            // flatten always copies
            var src = np.arange(4);
            src.flatten()[0] = 88L;
            ((long)src[0]).Should().Be(0L, "flatten always copies");

            // reshape(-1) is a view where possible
            np.arange(12).reshape(3, 4).reshape(-1).shape.Should().Equal(new long[] { 12 });

            // .T is a view
            var m = np.arange(6).reshape(2, 3);
            m.T[0, 0] = 100;
            ((long)m[0, 0]).Should().Be(100L, "transpose is a view");
        }

        // ── Broadcast views are read-only ──────────────────────────────────────────────────────────

        [TestMethod]
        public void Broadcast_IsReadOnly()
        {
            var small = np.array(new[] { 1, 2, 3 });
            var big = np.broadcast_to(small, (1_000_000, 3));
            big.Shape.IsBroadcasted.Should().BeTrue();
            big.Shape.IsWriteable.Should().BeFalse();

            Action write = () => big[0, 0] = 9;
            write.Should().Throw<ValueError>().WithMessage("*read-only*");
        }

        // ── How to tell a view from a copy ─────────────────────────────────────────────────────────

        [TestMethod]
        public void HowToTell_Base_And_SharesMemory()
        {
            var a = np.arange(10);                                // owns data
            var b = a["2:5"];                                    // view of a
            var d = a.copy();                                    // copy owns data

            (a.@base is not null).Should().BeFalse();
            (b.@base is not null).Should().BeTrue();
            (d.@base is not null).Should().BeFalse();

            np.shares_memory(a, b).Should().BeTrue("b shares a's buffer");

            var c = a[np.array(new[] { 0, 2 })];                 // copy
            np.may_share_memory(a, c).Should().BeFalse();
        }

        // ── In-place modification ──────────────────────────────────────────────────────────────────

        [TestMethod]
        public void InPlace_CompoundAssignIsNotInPlace_SliceAssignIs()
        {
            var x = np.array(new[] { 1, 2, 3 });
            var alias = x;
            x += 10;                                             // rebinds x to a NEW array
            x.ToArray<int>().Should().Equal(11, 12, 13);
            alias.ToArray<int>().Should().Equal(new[] { 1, 2, 3 }, "the alias still sees the original — unlike NumPy");

            var y = np.array(new[] { 1, 2, 3 });
            var yAlias = y;
            y[":"] = y + 1;                                      // in-place: rewrites y's buffer
            yAlias.ToArray<int>().Should().Equal(new[] { 2, 3, 4 }, "the alias sees the in-place change");
        }

        // ── ascontiguousarray ──────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void AsContiguousArray_MaterializesWhenNeeded()
        {
            var t = np.arange(6).reshape(2, 3).T;                // non-contiguous view
            var w = np.ascontiguousarray(t);
            w.Shape.IsContiguous.Should().BeTrue();
            w.ToArray<long>().Should().Equal(0L, 3, 1, 4, 2, 5);
        }
    }
}
