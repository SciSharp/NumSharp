using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Indexing
{
    /// <summary>
    /// Edge case tests for np.nonzero based on actual NumPy 2.4.2 behavior.
    /// </summary>
    public class np_nonzero_edge_cases : TestClass
    {
        [TestMethod]
        public void EmptyArray_ReturnsEmptyIndices()
        {
            // NumPy: np.nonzero(np.array([], dtype=np.int32)) returns (array([], dtype=int64),)
            var a = np.array(new int[0]);
            var r = np.nonzero(a);

            Assert.AreEqual(1, r.Length);
            Assert.AreEqual(0, r[0].size);
        }

        [TestMethod]
        public void NaN_IsNonZero()
        {
            // NumPy: nonzero([0, nan, 1]) = [1, 2] - NaN is considered nonzero
            var a = np.array(new[] { 0.0, double.NaN, 1.0 });
            var r = np.nonzero(a);

            r[0].Should().BeOfValues(1, 2);
        }

        [TestMethod]
        public void PositiveInfinity_IsNonZero()
        {
            // NumPy: nonzero([0, inf, 1]) = [1, 2]
            var a = np.array(new[] { 0.0, double.PositiveInfinity, 1.0 });
            var r = np.nonzero(a);

            r[0].Should().BeOfValues(1, 2);
        }

        [TestMethod]
        public void NegativeInfinity_IsNonZero()
        {
            // NumPy: nonzero([0, -inf, 1]) = [1, 2]
            var a = np.array(new[] { 0.0, double.NegativeInfinity, 1.0 });
            var r = np.nonzero(a);

            r[0].Should().BeOfValues(1, 2);
        }

        [TestMethod]
        public void NegativeZero_IsZero()
        {
            // NumPy: nonzero([0, -0.0, 1]) = [2] - negative zero is still zero
            var a = np.array(new[] { 0.0, -0.0, 1.0 });
            var r = np.nonzero(a);

            r[0].Should().BeOfValues(2);
        }

        [TestMethod]
        public void AllZeros_ReturnsEmptyIndices()
        {
            // NumPy: nonzero(zeros(5)) = []
            var a = np.zeros(5);
            var r = np.nonzero(a);

            Assert.AreEqual(0, r[0].size);
        }

        [TestMethod]
        public void AllNonZero_ReturnsAllIndices()
        {
            // NumPy: nonzero(ones(5)) = [0, 1, 2, 3, 4]
            var a = np.ones(5);
            var r = np.nonzero(a);

            r[0].Should().BeOfValues(0, 1, 2, 3, 4);
        }

        [TestMethod]
        public void NonContiguous_Slice_Works()
        {
            // NumPy: nonzero(arange(10)[::2]) = [1, 2, 3, 4] (values [0, 2, 4, 6, 8])
            var full = np.arange(10);
            var a = full["::2"];  // [0, 2, 4, 6, 8]

            // Verify we have the right slice
            a.Should().BeOfValues(0, 2, 4, 6, 8);
            Assert.IsFalse(a.Shape.IsContiguous);

            var r = np.nonzero(a);
            r[0].Should().BeOfValues(1, 2, 3, 4);  // indices of nonzero values
        }

        [TestMethod]
        public void TwoDimensional_Basic()
        {
            // NumPy: nonzero([[0,1,0],[2,0,3]]) = ([0,1,1], [1,0,2])
            var a = np.array(new[,] { { 0, 1, 0 }, { 2, 0, 3 } });
            var r = np.nonzero(a);

            Assert.AreEqual(2, r.Length);
            r[0].Should().BeOfValues(0, 1, 1);  // rows
            r[1].Should().BeOfValues(1, 0, 2);  // cols
        }
    }
}
