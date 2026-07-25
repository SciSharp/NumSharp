using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Indexing
{
    /// <summary>
    ///     Tests for <see cref="np.ndindex"/>. Every expected value comes from running
    ///     NumPy 2.4.2 (<c>list(np.ndindex(...))</c>) in the implementing session.
    /// </summary>
    [TestClass]
    public class np_ndindex_Test
    {
        private static string Fmt(long[] idx) => "(" + string.Join(",", idx) + ")";

        private static string Fmt(np.NDIndex it) => string.Join(" ", it.Select(Fmt));

        // ----------------------------------------------------------------- ordering

        [TestMethod]
        public void NdIndex_3D_LastAxisVariesFastest()
            // numpy: [(0,0,0),(0,1,0),(1,0,0),(1,1,0),(2,0,0),(2,1,0)]
            => Fmt(np.ndindex(3, 2, 1)).Should().Be("(0,0,0) (0,1,0) (1,0,0) (1,1,0) (2,0,0) (2,1,0)");

        [TestMethod]
        public void NdIndex_2D()
            // numpy: [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
            => Fmt(np.ndindex(3, 2)).Should().Be("(0,0) (0,1) (1,0) (1,1) (2,0) (2,1)");

        [TestMethod]
        public void NdIndex_1D_YieldsOneTuples()
            // numpy: [(0,),(1,),(2,),(3,),(4,)]
            => Fmt(np.ndindex(5)).Should().Be("(0) (1) (2) (3) (4)");

        [TestMethod]
        public void NdIndex_AllUnitDims_YieldsSingleZeroIndex()
            => Fmt(np.ndindex(1, 1, 1)).Should().Be("(0,0,0)");

        // ----------------------------------------------------------------- edge shapes

        [TestMethod]
        public void NdIndex_NoArguments_YieldsOneEmptyIndex()
        {
            // numpy: list(np.ndindex()) == [()] — product() with no iterables yields one empty tuple.
            var all = np.ndindex().ToArray();
            all.Should().HaveCount(1);
            all[0].Should().BeEmpty();
        }

        [TestMethod]
        public void NdIndex_EmptyShapeArray_YieldsOneEmptyIndex()
        {
            // numpy: list(np.ndindex(())) == [()]
            var all = np.ndindex(Array.Empty<long>()).ToArray();
            all.Should().HaveCount(1);
            all[0].Should().BeEmpty();
        }

        [TestMethod]
        public void NdIndex_ZeroLengthDimension_YieldsNothing()
            // numpy: list(np.ndindex(0, 3)) == []
            => np.ndindex(0, 3).Should().BeEmpty();

        [TestMethod]
        public void NdIndex_ZeroInLastDimension_YieldsNothing()
            => np.ndindex(3, 0).Should().BeEmpty();

        // ----------------------------------------------------------------- validation

        [TestMethod]
        public void NdIndex_NegativeDimension_Throws()
        {
            // numpy: ValueError("negative dimensions are not allowed"), raised in __init__ —
            // BEFORE any index is produced, so merely constructing must throw.
            new Action(() => np.ndindex(-1))
                .Should().Throw<ArgumentException>().WithMessage("negative dimensions are not allowed");
        }

        [TestMethod]
        public void NdIndex_NegativeInLaterDimension_Throws()
            => new Action(() => np.ndindex(2, -3))
                .Should().Throw<ArgumentException>().WithMessage("negative dimensions are not allowed");

        // ----------------------------------------------------------------- overloads

        [TestMethod]
        public void NdIndex_IntArrayOverload_MatchesParamsForm()
            => Fmt(np.ndindex(new[] {2, 2})).Should().Be(Fmt(np.ndindex(2L, 2L)));

        [TestMethod]
        public void NdIndex_LongArrayOverload_MatchesParamsForm()
            => Fmt(np.ndindex(new long[] {2, 2})).Should().Be("(0,0) (0,1) (1,0) (1,1)");

        [TestMethod]
        public void NdIndex_AcceptsArrayShapeDirectly()
        {
            var a = np.arange(6).reshape(2, 3);
            Fmt(np.ndindex(a.shape)).Should().Be("(0,0) (0,1) (0,2) (1,0) (1,1) (1,2)");
        }

        // ----------------------------------------------------------------- iterator identity

        [TestMethod]
        public void NdIndex_IsItsOwnIterator_SecondPassResumes()
        {
            // numpy: iter(i) is i — consuming two indices then iterating again yields the REST.
            var it = np.ndindex(2, 2);
            string first = string.Join(" ", it.Take(2).Select(Fmt));
            string rest = string.Join(" ", it.Select(Fmt));

            first.Should().Be("(0,0) (0,1)");
            rest.Should().Be("(1,0) (1,1)");
        }

        [TestMethod]
        public void NdIndex_YieldsFreshArrays_NotARecycledBuffer()
        {
            // list(np.ndindex(...)) must hold distinct indices, so materializing cannot alias
            // one mutating buffer.
            var all = np.ndindex(2, 2).ToArray();
            all.Select(Fmt).Should().Equal("(0,0)", "(0,1)", "(1,0)", "(1,1)");
        }

        [TestMethod]
        public void NdIndex_IndicesAreIntp()
            // NumPy's indices are intp == int64 on 64-bit; the house rule for index-valued output.
            => np.ndindex(2, 2).First().Should().BeOfType<long[]>();

        // ----------------------------------------------------------------- larger sweep

        [TestMethod]
        public void NdIndex_MatchesRowMajorEnumeration()
        {
            var expected = (from i in Enumerable.Range(0, 4)
                            from j in Enumerable.Range(0, 3)
                            from k in Enumerable.Range(0, 2)
                            select $"({i},{j},{k})").ToArray();

            np.ndindex(4, 3, 2).Select(Fmt).Should().Equal(expected);
        }
    }
}
