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

        await Assert.That(arr.Contains(3)).IsTrue();
        await Assert.That(arr.__contains__(3)).IsTrue();
    }

    [TestMethod]
    public async Task Contains_Int32_ValueNotExists_ReturnsFalse()
    {
        // NumPy: 10 in np.array([1, 2, 3, 4, 5]) = False
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        await Assert.That(arr.Contains(10)).IsFalse();
        await Assert.That(arr.__contains__(10)).IsFalse();
    }

    [TestMethod]
    public async Task Contains_Double_ValueExists_ReturnsTrue()
    {
        // NumPy: 2.5 in np.array([1.0, 2.5, 3.0]) = True
        var arr = np.array(new[] { 1.0, 2.5, 3.0 });

        await Assert.That(arr.Contains(2.5)).IsTrue();
    }

    [TestMethod]
    public async Task Contains_Double_NaN_ReturnsFalse()
    {
        // NumPy: np.nan in np.array([1.0, np.nan, 3.0]) = False
        // NaN != NaN in IEEE 754, so Contains returns False
        var arr = np.array(new[] { 1.0, double.NaN, 3.0 });

        await Assert.That(arr.Contains(double.NaN)).IsFalse();
    }

    [TestMethod]
    public async Task Contains_EmptyArray_ReturnsFalse()
    {
        // NumPy: 1 in np.array([]) = False
        var arr = np.array(Array.Empty<int>());

        await Assert.That(arr.Contains(1)).IsFalse();
    }

    [TestMethod]
    public async Task Contains_2DArray_SearchesAllElements()
    {
        // NumPy: 5 in np.array([[1, 2], [3, 4], [5, 6]]) = True
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        await Assert.That(arr.Contains(5)).IsTrue();
        await Assert.That(arr.Contains(10)).IsFalse();
    }

    [TestMethod]
    public async Task Contains_Boolean_True()
    {
        // NumPy: True in np.array([False, True, False]) = True
        var arr = np.array(new[] { false, true, false });

        await Assert.That(arr.Contains(true)).IsTrue();
        await Assert.That(arr.Contains(false)).IsTrue();
    }

    [TestMethod]
    public async Task Contains_TypePromotion_IntInFloatArray()
    {
        // NumPy: 2 in np.array([1.0, 2.0, 3.0]) = True
        // Type promotion: int 2 is compared with float 2.0
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });

        await Assert.That(arr.Contains(2)).IsTrue();
    }

    [TestMethod]
    public async Task Contains_Null_ReturnsFalse()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        await Assert.That(arr.Contains(null)).IsFalse();
    }

    [TestMethod]
    public async Task Contains_FirstElement()
    {
        var arr = np.array(new[] { 42, 1, 2, 3 });

        await Assert.That(arr.Contains(42)).IsTrue();
    }

    [TestMethod]
    public async Task Contains_LastElement()
    {
        var arr = np.array(new[] { 1, 2, 3, 42 });

        await Assert.That(arr.Contains(42)).IsTrue();
    }

    #endregion

    #region Hash Tests

    [TestMethod]
    public async Task GetHashCode_ThrowsNotSupportedException()
    {
        // NumPy: hash(np.array([1, 2, 3])) -> TypeError: unhashable type: 'numpy.ndarray'
        var arr = np.array(new[] { 1, 2, 3 });

        await Assert.That(() => arr.GetHashCode()).Throws<NotSupportedException>();
    }

    [TestMethod]
    public async Task __hash___ThrowsNotSupportedException()
    {
        // NumPy: hash(np.array([1, 2, 3])) -> TypeError: unhashable type: 'numpy.ndarray'
        var arr = np.array(new[] { 1, 2, 3 });

        await Assert.That(() => arr.__hash__()).Throws<NotSupportedException>();
    }

    [TestMethod]
    public async Task Dictionary_WithNDArrayKey_ThrowsOnAccess()
    {
        // Attempting to use NDArray as dictionary key should fail
        var arr = np.array(new[] { 1, 2, 3 });
        var dict = new Dictionary<object, string>();

        // Adding to dictionary calls GetHashCode, which throws
        await Assert.That(() => dict.Add(arr, "value")).Throws<NotSupportedException>();
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

        await Assert.That(dict[arr1]).IsEqualTo("first");
        await Assert.That(dict[arr2]).IsEqualTo("second");
        await Assert.That(dict.Count).IsEqualTo(2); // Two distinct references
    }

    #endregion

    #region Python Naming Convention Tests

    [TestMethod]
    public async Task __contains___IsSameAsContains()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        // Both methods should return the same result
        await Assert.That(arr.__contains__(3)).IsEqualTo(arr.Contains(3));
        await Assert.That(arr.__contains__(10)).IsEqualTo(arr.Contains(10));
    }

    #endregion

    #region Length Tests (__len__)

    [TestMethod]
    public async Task __len___1DArray_ReturnsLength()
    {
        // NumPy: len(np.array([1, 2, 3])) = 3
        var arr = np.array(new[] { 1, 2, 3 });

        await Assert.That(arr.__len__()).IsEqualTo(3);
    }

    [TestMethod]
    public async Task __len___2DArray_ReturnsFirstDimension()
    {
        // NumPy: len(np.array([[1, 2], [3, 4], [5, 6]])) = 3
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        await Assert.That(arr.__len__()).IsEqualTo(3);
    }

    [TestMethod]
    public async Task __len___ScalarArray_ThrowsTypeError()
    {
        // NumPy: len(np.array(5)) -> TypeError: len() of unsized object
        // Note: NDArray.Scalar creates a true 0-d array, while np.array(5) creates 1-d
        var scalar = NDArray.Scalar(5);

        await Assert.That(scalar.ndim).IsEqualTo(0); // Verify it's a true scalar
        await Assert.That(() => scalar.__len__()).Throws<TypeError>();
    }

    #endregion

    #region Iteration Tests (__iter__)

    [TestMethod]
    public async Task __iter___ReturnsEnumerator()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var iter = arr.__iter__();

        await Assert.That(iter).IsNotNull();
        await Assert.That(iter.MoveNext()).IsTrue();
    }

    #endregion

    #region Indexing Tests (__getitem__, __setitem__)

    [TestMethod]
    public async Task __getitem___IntIndex_ReturnsElement()
    {
        var arr = np.array(new[] { 10, 20, 30 });

        var result = arr.__getitem__(1);
        await Assert.That((int)result).IsEqualTo(20);
    }

    [TestMethod]
    public async Task __getitem___NegativeIndex_ReturnsFromEnd()
    {
        var arr = np.array(new[] { 10, 20, 30 });

        var result = arr.__getitem__(-1);
        await Assert.That((int)result).IsEqualTo(30);
    }

    [TestMethod]
    public async Task __getitem___SliceString_ReturnsSlice()
    {
        var arr = np.array(new[] { 10, 20, 30, 40, 50 });

        var result = arr.__getitem__("1:4");
        await Assert.That(result.size).IsEqualTo(3);
        await Assert.That((int)result[0]).IsEqualTo(20);
    }

    [TestMethod]
    public async Task __setitem___IntIndex_SetsElement()
    {
        var arr = np.array(new[] { 10, 20, 30 });

        arr.__setitem__(1, 99);
        await Assert.That((int)arr[1]).IsEqualTo(99);
    }

    [TestMethod]
    public async Task __setitem___SliceString_SetsSlice()
    {
        var arr = np.array(new[] { 10, 20, 30, 40, 50 });

        arr.__setitem__(":3", 0);
        await Assert.That((int)arr[0]).IsEqualTo(0);
        await Assert.That((int)arr[1]).IsEqualTo(0);
        await Assert.That((int)arr[2]).IsEqualTo(0);
        await Assert.That((int)arr[3]).IsEqualTo(40); // Unchanged
    }

    #endregion
}
