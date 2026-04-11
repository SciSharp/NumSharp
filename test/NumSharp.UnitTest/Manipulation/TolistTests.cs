using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using TUnit.Core;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    ///     Battle tests for NDArray.tolist() - NumPy parity verification.
    /// </summary>
    public class TolistTests
    {
        #region Basic Functionality

        [Test]
        public void Tolist_Scalar_ReturnsScalarValue()
        {
            // NumPy: np.array(42).tolist() -> 42
            var arr = np.array(42).reshape(Shape.Scalar);
            var result = arr.tolist();

            result.Should().Be(42L); // arange returns int64
        }

        [Test]
        public void Tolist_1D_ReturnsFlatList()
        {
            // NumPy: np.array([1, 2, 3, 4, 5]).tolist() -> [1, 2, 3, 4, 5]
            var arr = np.array(new int[] { 1, 2, 3, 4, 5 });
            var result = (List<object>)arr.tolist();

            result.Count.Should().Be(5);
            result[0].Should().Be(1);
            result[4].Should().Be(5);
        }

        [Test]
        public void Tolist_2D_ReturnsNestedList()
        {
            // NumPy: np.arange(6).reshape(2, 3).tolist() -> [[0, 1, 2], [3, 4, 5]]
            var arr = np.arange(6).reshape(2, 3);
            var result = (List<object>)arr.tolist();

            result.Count.Should().Be(2);

            var row0 = (List<object>)result[0];
            var row1 = (List<object>)result[1];

            row0.Count.Should().Be(3);
            row0[0].Should().Be(0L);
            row0[1].Should().Be(1L);
            row0[2].Should().Be(2L);

            row1[0].Should().Be(3L);
            row1[1].Should().Be(4L);
            row1[2].Should().Be(5L);
        }

        [Test]
        public void Tolist_3D_ReturnsTriplyNestedList()
        {
            // NumPy: np.arange(24).reshape(2, 3, 4).tolist()
            var arr = np.arange(24).reshape(2, 3, 4);
            var result = (List<object>)arr.tolist();

            result.Count.Should().Be(2);

            var block0 = (List<object>)result[0];
            block0.Count.Should().Be(3);

            var row00 = (List<object>)block0[0];
            row00.Count.Should().Be(4);
            row00[0].Should().Be(0L);
            row00[3].Should().Be(3L);
        }

        #endregion

        #region Dtype Preservation

        [Test]
        public void Tolist_Int32_PreservesType()
        {
            var arr = np.array(new int[] { 1, 2, 3 });
            var result = (List<object>)arr.tolist();

            result[0].Should().BeOfType<int>();
        }

        [Test]
        public void Tolist_Float64_PreservesType()
        {
            var arr = np.array(new double[] { 1.5, 2.5, 3.5 });
            var result = (List<object>)arr.tolist();

            result[0].Should().BeOfType<double>();
            result[0].Should().Be(1.5);
        }

        [Test]
        public void Tolist_Bool_PreservesType()
        {
            var arr = np.array(new bool[] { true, false, true });
            var result = (List<object>)arr.tolist();

            result[0].Should().BeOfType<bool>();
            result[0].Should().Be(true);
            result[1].Should().Be(false);
        }

        #endregion

        #region Edge Cases

        [Test]
        public void Tolist_SingleElement1D_ReturnsSingleElementList()
        {
            var arr = np.array(new int[] { 42 });
            var result = (List<object>)arr.tolist();

            result.Count.Should().Be(1);
            result[0].Should().Be(42);
        }

        [Test]
        public void Tolist_SlicedArray_WorksCorrectly()
        {
            // Test that sliced arrays produce correct nested lists
            var arr = np.arange(12).reshape(3, 4);
            var sliced = arr["1:3, 1:3"]; // [[5, 6], [9, 10]]
            var result = (List<object>)sliced.tolist();

            result.Count.Should().Be(2);

            var row0 = (List<object>)result[0];
            row0[0].Should().Be(5L);
            row0[1].Should().Be(6L);

            var row1 = (List<object>)result[1];
            row1[0].Should().Be(9L);
            row1[1].Should().Be(10L);
        }

        [Test]
        public void Tolist_TransposedArray_WorksCorrectly()
        {
            // NumPy: np.arange(6).reshape(2, 3).T.tolist()
            var arr = np.arange(6).reshape(2, 3).T;
            var result = (List<object>)arr.tolist();

            // Original: [[0, 1, 2], [3, 4, 5]]
            // Transposed: [[0, 3], [1, 4], [2, 5]]
            result.Count.Should().Be(3);

            var row0 = (List<object>)result[0];
            row0[0].Should().Be(0L);
            row0[1].Should().Be(3L);
        }

        #endregion
    }
}
