using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Indexing
{
    /// <summary>
    ///     NumPy's index tricks (<c>lib/_index_tricks_impl.py</c>) as NumPy 2.4.2 runs them, found when the np-layer dtype
    ///     plan's P3 batch 7 moved their statements into <c>NDArray.IndexTricks.cs</c> and replayed the snapshot (11,514 cases)
    ///     against NumPy on Windows and on Linux: <c>np.ix_</c>'s result object for one sequence and its empty-sequence rule.
    ///     Every expectation is NumPy 2.4.2's output.
    /// </summary>
    [TestClass]
    public class index_tricks_numpy_port_tests
    {
        /// <summary>A single sequence gets no axis, and NumPy's reshape to the same shape is still a new view.</summary>
        [TestMethod]
        public void Ix_SingleSequence_IsANewView()
        {
            // a = np.arange(3); r = np.ix_(a)[0] -> r is a: False, r.flags.owndata: False, np.shares_memory(r, a): True
            var a = np.arange(3L);
            var r = np.ix_(a)[0];
            ReferenceEquals(r, a).Should().BeFalse();
            r.flags.owndata.Should().BeFalse();
            np.shares_memory(r, a).Should().BeTrue();
            // r[0] = 9 writes through: a is [9, 1, 2]
            r[0] = 9L;
            a.ToArray<long>().Should().Equal(9L, 1L, 2L);

            // np.ix_([1, 2])[0].flags.owndata -> False: the array a list became is not handed out either
            var fromList = np.ix_(new object[] { new object[] { 1, 2 } })[0];
            fromList.flags.owndata.Should().BeFalse();
            fromList.ToArray<long>().Should().Equal(1L, 2L);

            // np.ix_(np.array([1, 2], np.int32))[0] -> array([1, 2], dtype=int32), owndata False
            var typed = np.ix_(new[] { 1, 2 })[0];
            typed.dtype.Should().Be(np.int32);
            typed.flags.owndata.Should().BeFalse();
        }

        /// <summary>Only a sequence that is not an ndarray is cast to intp when it comes out empty.</summary>
        [TestMethod]
        public void Ix_EmptySequence_OnlyAListBecomesIntp()
        {
            // np.ix_(np.array([], np.int32))[0].dtype -> int32; np.ix_(np.array([], np.float64))[0].dtype -> float64
            np.ix_(new int[0])[0].dtype.Should().Be(np.int32);
            np.ix_(new double[0])[0].dtype.Should().Be(np.float64);
            // np.ix_([])[0].dtype -> int64 (intp)
            np.ix_(new object[] { new object[0] })[0].dtype.Should().Be(np.int64);
        }
    }
}
