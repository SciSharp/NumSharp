using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NumSharp;

namespace NumSharp.UnitTest.Backends;

/// <summary>
/// Tests for container protocol implementation (__contains__, __hash__).
/// Verifies NumPy-compatible behavior for Contains and GetHashCode.
/// </summary>
public class ContainerProtocolTests
{
    #region Contains Tests

    [TestMethod]
    public async Task Contains_Int32_ValueExists_ReturnsTrue()
    {
        // NumPy: 3 in np.array([1, 2, 3, 4, 5]) = True
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr.Contains(3).Should().BeTrue();
        arr.__contains__(3).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_Int32_ValueNotExists_ReturnsFalse()
    {
        // NumPy: 10 in np.array([1, 2, 3, 4, 5]) = False
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        arr.Contains(10).Should().BeFalse();
        arr.__contains__(10).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_Double_ValueExists_ReturnsTrue()
    {
        // NumPy: 2.5 in np.array([1.0, 2.5, 3.0]) = True
        var arr = np.array(new[] { 1.0, 2.5, 3.0 });

        arr.Contains(2.5).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_Double_NaN_ReturnsFalse()
    {
        // NumPy: np.nan in np.array([1.0, np.nan, 3.0]) = False
        // NaN != NaN in IEEE 754, so Contains returns False
        var arr = np.array(new[] { 1.0, double.NaN, 3.0 });

        arr.Contains(double.NaN).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_EmptyArray_ReturnsFalse()
    {
        // NumPy: 1 in np.array([]) = False
        var arr = np.array(Array.Empty<int>());

        arr.Contains(1).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_2DArray_SearchesAllElements()
    {
        // NumPy: 5 in np.array([[1, 2], [3, 4], [5, 6]]) = True
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        arr.Contains(5).Should().BeTrue();
        arr.Contains(10).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_Boolean_True()
    {
        // NumPy: True in np.array([False, True, False]) = True
        var arr = np.array(new[] { false, true, false });

        arr.Contains(true).Should().BeTrue();
        arr.Contains(false).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_TypePromotion_IntInFloatArray()
    {
        // NumPy: 2 in np.array([1.0, 2.0, 3.0]) = True
        // Type promotion: int 2 is compared with float 2.0
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });

        arr.Contains(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_Null_ReturnsFalse()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        arr.Contains(null).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_FirstElement()
    {
        var arr = np.array(new[] { 42, 1, 2, 3 });

        arr.Contains(42).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_LastElement()
    {
        var arr = np.array(new[] { 1, 2, 3, 42 });

        arr.Contains(42).Should().BeTrue();
    }

    #endregion

    #region Hash Tests

    [TestMethod]
    public async Task GetHashCode_ThrowsNotSupportedException()
    {
        // NumPy: hash(np.array([1, 2, 3])) -> TypeError: unhashable type: 'numpy.ndarray'
        var arr = np.array(new[] { 1, 2, 3 });

        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<NotSupportedException>(() => arr.GetHashCode());
    }

    [TestMethod]
    public async Task __hash___ThrowsNotSupportedException()
    {
        // NumPy: hash(np.array([1, 2, 3])) -> TypeError: unhashable type: 'numpy.ndarray'
        var arr = np.array(new[] { 1, 2, 3 });

        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<NotSupportedException>(() => arr.__hash__());
    }

    [TestMethod]
    public async Task Dictionary_WithNDArrayKey_ThrowsOnAccess()
    {
        // Attempting to use NDArray as dictionary key should fail
        var arr = np.array(new[] { 1, 2, 3 });
        var dict = new Dictionary<object, string>();

        // Adding to dictionary calls GetHashCode, which throws
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<NotSupportedException>(() => dict.Add(arr, "value"));
    }

    [TestMethod]
    public async Task Dictionary_WithReferenceEqualityComparer_Works()
    {
        // Workaround: use ReferenceEqualityComparer
        var arr1 = np.array(new[] { 1, 2, 3 });
        var arr2 = np.array(new[] { 1, 2, 3 });
        var dict = new Dictionary<NDArray, string>(ReferenceEqualityComparer.Instance);

        dict[arr1] = "first";
        dict[arr2] = "second";

        dict[arr1].Should().Be("first");
        dict[arr2].Should().Be("second");
        dict.Count.Should().Be(2); // Two distinct references
    }

    #endregion

    #region Python Naming Convention Tests

    [TestMethod]
    public async Task __contains___IsSameAsContains()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        // Both methods should return the same result
        arr.__contains__(3).Should().Be(arr.Contains(3));
        arr.__contains__(10).Should().Be(arr.Contains(10));
    }

    #endregion

    #region Length Tests (__len__)

    [TestMethod]
    public async Task __len___1DArray_ReturnsLength()
    {
        // NumPy: len(np.array([1, 2, 3])) = 3
        var arr = np.array(new[] { 1, 2, 3 });

        arr.__len__().Should().Be(3);
    }

    [TestMethod]
    public async Task __len___2DArray_ReturnsFirstDimension()
    {
        // NumPy: len(np.array([[1, 2], [3, 4], [5, 6]])) = 3
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        arr.__len__().Should().Be(3);
    }

    [TestMethod]
    public async Task __len___ScalarArray_ThrowsTypeError()
    {
        // NumPy: len(np.array(5)) -> TypeError: len() of unsized object
        // Note: NDArray.Scalar creates a true 0-d array, while np.array(5) creates 1-d
        var scalar = NDArray.Scalar(5);

        scalar.ndim.Should().Be(0); // Verify it's a true scalar
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<TypeError>(() => scalar.__len__());
    }

    #endregion

    #region Iteration Tests (__iter__)

    [TestMethod]
    public async Task __iter___ReturnsEnumerator()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var iter = arr.__iter__();

        iter.Should().NotBeNull();
        iter.MoveNext().Should().BeTrue();
    }

    #endregion

    #region Indexing Tests (__getitem__, __setitem__)

    [TestMethod]
    public async Task __getitem___IntIndex_ReturnsElement()
    {
        var arr = np.array(new[] { 10, 20, 30 });

        var result = arr.__getitem__(1);
        ((int)result).Should().Be(20);
    }

    [TestMethod]
    public async Task __getitem___NegativeIndex_ReturnsFromEnd()
    {
        var arr = np.array(new[] { 10, 20, 30 });

        var result = arr.__getitem__(-1);
        ((int)result).Should().Be(30);
    }

    [TestMethod]
    public async Task __getitem___SliceString_ReturnsSlice()
    {
        var arr = np.array(new[] { 10, 20, 30, 40, 50 });

        var result = arr.__getitem__("1:4");
        result.size.Should().Be(3);
        ((int)result[0]).Should().Be(20);
    }

    [TestMethod]
    public async Task __setitem___IntIndex_SetsElement()
    {
        var arr = np.array(new[] { 10, 20, 30 });

        arr.__setitem__(1, 99);
        ((int)arr[1]).Should().Be(99);
    }

    [TestMethod]
    public async Task __setitem___SliceString_SetsSlice()
    {
        var arr = np.array(new[] { 10, 20, 30, 40, 50 });

        arr.__setitem__(":3", 0);
        ((int)arr[0]).Should().Be(0);
        ((int)arr[1]).Should().Be(0);
        ((int)arr[2]).Should().Be(0);
        ((int)arr[3]).Should().Be(40); // Unchanged
    }

    #endregion
}
