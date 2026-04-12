using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Tests for np.split and np.array_split functions.
    /// Based on NumPy 2.4.2 behavior.
    /// </summary>
    public class np_split_Tests : TestClass
    {
        #region np.split - Equal Division

        [Test]
        public void Split_1D_EqualParts()
        {
            // NumPy: np.split(np.arange(9), 3) -> [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]
            var a = np.arange(9);
            var result = np.split(a, 3);

            result.Length.Should().Be(3);
            result[0].ToArray<long>().Should().ContainInOrder(0, 1, 2);
            result[1].ToArray<long>().Should().ContainInOrder(3, 4, 5);
            result[2].ToArray<long>().Should().ContainInOrder(6, 7, 8);
        }

        [Test]
        public void Split_1D_UnequalDivision_ThrowsArgumentException()
        {
            // NumPy: np.split(np.arange(9), 4) raises ValueError
            var a = np.arange(9);

            Assert.ThrowsException<ArgumentException>(() => np.split(a, 4));
        }

        [Test]
        public void Split_2D_Axis0()
        {
            // NumPy: np.split(np.arange(12).reshape(3, 4), 3, axis=0)
            // -> [array([[0,1,2,3]]), array([[4,5,6,7]]), array([[8,9,10,11]])]
            var d = np.arange(12).reshape(3, 4);
            var result = np.split(d, 3, axis: 0);

            result.Length.Should().Be(3);
            result[0].Should().BeShaped(1, 4);
            result[1].Should().BeShaped(1, 4);
            result[2].Should().BeShaped(1, 4);
        }

        [Test]
        public void Split_2D_Axis1()
        {
            // NumPy: np.split(np.arange(12).reshape(3, 4), 2, axis=1)
            var d = np.arange(12).reshape(3, 4);
            var result = np.split(d, 2, axis: 1);

            result.Length.Should().Be(2);
            result[0].Should().BeShaped(3, 2);
            result[1].Should().BeShaped(3, 2);

            // First result: columns 0-1
            result[0]["0, :"].ToArray<long>().Should().ContainInOrder(0, 1);
            result[0]["1, :"].ToArray<long>().Should().ContainInOrder(4, 5);

            // Second result: columns 2-3
            result[1]["0, :"].ToArray<long>().Should().ContainInOrder(2, 3);
            result[1]["1, :"].ToArray<long>().Should().ContainInOrder(6, 7);
        }

        [Test]
        public void Split_NegativeAxis()
        {
            // NumPy: np.split(d, 3, axis=-2) == np.split(d, 3, axis=0)
            var d = np.arange(12).reshape(3, 4);
            var r1 = np.split(d, 3, axis: -2);
            var r2 = np.split(d, 3, axis: 0);

            r1.Length.Should().Be(r2.Length);
            for (int i = 0; i < r1.Length; i++)
            {
                ((bool)np.array_equal(r1[i], r2[i])).Should().BeTrue();
            }
        }

        #endregion

        #region np.split - With Indices

        [Test]
        public void Split_WithIndices()
        {
            // NumPy: np.split(a, [3, 5, 7]) -> a[:3], a[3:5], a[5:7], a[7:]
            var a = np.arange(9);
            var result = np.split(a, new[] { 3, 5, 7 });

            result.Length.Should().Be(4);
            result[0].ToArray<long>().Should().ContainInOrder(0, 1, 2);
            result[1].ToArray<long>().Should().ContainInOrder(3, 4);
            result[2].ToArray<long>().Should().ContainInOrder(5, 6);
            result[3].ToArray<long>().Should().ContainInOrder(7, 8);
        }

        [Test]
        public void Split_WithIndices_EmptyAtStart()
        {
            // NumPy: np.split(a, [0, 3, 6]) -> [], a[:3], a[3:6], a[6:]
            var a = np.arange(9);
            var result = np.split(a, new[] { 0, 3, 6 });

            result.Length.Should().Be(4);
            result[0].size.Should().Be(0);
            result[1].size.Should().Be(3);
            result[2].size.Should().Be(3);
            result[3].size.Should().Be(3);
        }

        [Test]
        public void Split_WithIndices_DuplicateIndices()
        {
            // NumPy: np.split(a, [3, 3, 6]) -> a[:3], [], a[3:6], a[6:]
            var a = np.arange(9);
            var result = np.split(a, new[] { 3, 3, 6 });

            result.Length.Should().Be(4);
            result[0].size.Should().Be(3);
            result[1].size.Should().Be(0);
            result[2].size.Should().Be(3);
            result[3].size.Should().Be(3);
        }

        [Test]
        public void Split_WithIndices_EmptyIndicesArray()
        {
            // NumPy: np.split(a, []) -> [a]
            var a = np.arange(9);
            var result = np.split(a, new int[] { });

            result.Length.Should().Be(1);
            result[0].size.Should().Be(9);
        }

        #endregion

        #region np.array_split - Unequal Division

        [Test]
        public void ArraySplit_UnequalDivision_9Into4()
        {
            // NumPy: np.array_split(np.arange(9), 4)
            // -> [array([0, 1, 2]), array([3, 4]), array([5, 6]), array([7, 8])]
            // 9 % 4 = 1 extra, so first 1 section gets size 3, rest get size 2
            var a = np.arange(9);
            var result = np.array_split(a, 4);

            result.Length.Should().Be(4);
            result[0].size.Should().Be(3); // 9//4 + 1 = 3
            result[1].size.Should().Be(2); // 9//4 = 2
            result[2].size.Should().Be(2);
            result[3].size.Should().Be(2);
        }

        [Test]
        public void ArraySplit_UnequalDivision_10Into3()
        {
            // NumPy: np.array_split(np.arange(10), 3)
            // -> [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
            // 10 % 3 = 1 extra, so first 1 section gets size 4, rest get size 3
            var b = np.arange(10);
            var result = np.array_split(b, 3);

            result.Length.Should().Be(3);
            result[0].size.Should().Be(4); // 10//3 + 1 = 4
            result[1].size.Should().Be(3); // 10//3 = 3
            result[2].size.Should().Be(3);
        }

        [Test]
        public void ArraySplit_UnequalDivision_8Into3()
        {
            // NumPy: np.array_split(np.arange(8), 3)
            // -> [array([0, 1, 2]), array([3, 4, 5]), array([6, 7])]
            // 8 % 3 = 2 extras, so first 2 sections get size 3, rest get size 2
            var c = np.arange(8);
            var result = np.array_split(c, 3);

            result.Length.Should().Be(3);
            result[0].size.Should().Be(3); // 8//3 + 1 = 3
            result[1].size.Should().Be(3); // 8//3 + 1 = 3
            result[2].size.Should().Be(2); // 8//3 = 2
        }

        [Test]
        public void ArraySplit_2D_UnequalAxis0()
        {
            // NumPy: np.array_split(np.arange(12).reshape(3, 4), 2, axis=0)
            // -> [array([[0,1,2,3],[4,5,6,7]]), array([[8,9,10,11]])]
            var d = np.arange(12).reshape(3, 4);
            var result = np.array_split(d, 2, axis: 0);

            result.Length.Should().Be(2);
            result[0].Should().BeShaped(2, 4);
            result[1].Should().BeShaped(1, 4);
        }

        [Test]
        public void ArraySplit_MoreSectionsThanElements()
        {
            // NumPy: np.array_split(np.array([1,2,3]), 5)
            // -> [array([1]), array([2]), array([3]), array([]), array([])]
            var small = np.array(new[] { 1, 2, 3 });
            var result = np.array_split(small, 5);

            result.Length.Should().Be(5);
            result[0].size.Should().Be(1);
            result[1].size.Should().Be(1);
            result[2].size.Should().Be(1);
            result[3].size.Should().Be(0);
            result[4].size.Should().Be(0);
        }

        [Test]
        public void ArraySplit_ZeroSections_ThrowsArgumentException()
        {
            // NumPy: np.array_split(a, 0) raises ValueError
            var a = np.arange(9);

            Assert.ThrowsException<ArgumentException>(() => np.array_split(a, 0));
        }

        [Test]
        public void ArraySplit_NegativeSections_ThrowsArgumentException()
        {
            // NumPy: np.array_split(a, -1) raises ValueError
            var a = np.arange(9);

            Assert.ThrowsException<ArgumentException>(() => np.array_split(a, -1));
        }

        #endregion

        #region Edge Cases

        [Test]
        public void Split_SingleElement()
        {
            var single = np.array(new[] { 42 });
            var result = np.split(single, 1);

            result.Length.Should().Be(1);
            result[0].ToArray<int>().Should().ContainInOrder(42);
        }

        [Test]
        public void Split_EmptyArray()
        {
            var empty = np.array(new double[0]);
            var result = np.split(empty, 1);

            result.Length.Should().Be(1);
            result[0].size.Should().Be(0);
        }

        [Test]
        public void ArraySplit_EmptyArray_MultipleSections()
        {
            var empty = np.array(new double[0]);
            var result = np.array_split(empty, 3);

            result.Length.Should().Be(3);
            foreach (var r in result)
            {
                r.size.Should().Be(0);
            }
        }

        [Test]
        public void Split_3D_Array()
        {
            // NumPy: np.split(np.arange(24).reshape(2, 3, 4), 2, axis=0)
            var e = np.arange(24).reshape(2, 3, 4);
            var result = np.split(e, 2, axis: 0);

            result.Length.Should().Be(2);
            result[0].Should().BeShaped(1, 3, 4);
            result[1].Should().BeShaped(1, 3, 4);
        }

        [Test]
        public void Split_ReturnsViews()
        {
            // NumPy: split returns views, not copies
            var a = np.arange(9);
            var result = np.split(a, 3);

            // Modifying a view should affect the original
            result[0].SetInt64(999, 0);
            a.GetInt64(0).Should().Be(999);
        }

        [Test]
        public void Split_DifferentDtypes()
        {
            // Test with float64
            var f = np.arange(6.0);
            var result = np.split(f, 3);

            result.Length.Should().Be(3);
            result[0].dtype.Should().Be(typeof(double));
            result[0].ToArray<double>().Should().ContainInOrder(0.0, 1.0);
        }

        #endregion
    }
}
