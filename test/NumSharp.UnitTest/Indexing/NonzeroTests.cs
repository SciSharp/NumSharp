using System;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Indexing;

/// <summary>
/// Tests for np.nonzero behavior.
/// NumPy: Returns a tuple of arrays, one for each dimension.
/// </summary>
[TestClass]
public class NonzeroTests
{
    #region 1D Arrays

    [TestMethod]
    public async Task Nonzero_1D_ReturnsIndices()
    {
        // NumPy: np.nonzero([0, 1, 0, 2]) = (array([1, 3]),)
        var a = np.array(new int[] { 0, 1, 0, 2 });
        var result = np.nonzero(a);

        result.Length.Should().Be(1);
        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(1);
        result[0].GetInt64(1).Should().Be(3);
    }

    [TestMethod]
    public async Task Nonzero_1D_Empty_ReturnsEmptyArray()
    {
        // NumPy: np.nonzero([0, 0, 0]) = (array([], dtype=int64),)
        var a = np.array(new int[] { 0, 0, 0 });
        var result = np.nonzero(a);

        result.Length.Should().Be(1);
        result[0].size.Should().Be(0);
    }

    [TestMethod]
    public async Task Nonzero_1D_AllNonzero_ReturnsAllIndices()
    {
        // NumPy: np.nonzero([1, 2, 3]) = (array([0, 1, 2]),)
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.nonzero(a);

        result.Length.Should().Be(1);
        result[0].size.Should().Be(3);
        result[0].GetInt64(0).Should().Be(0);
        result[0].GetInt64(1).Should().Be(1);
        result[0].GetInt64(2).Should().Be(2);
    }

    #endregion

    #region 2D Arrays

    [TestMethod]
    public async Task Nonzero_2D_ReturnsTupleOfIndices()
    {
        // NumPy: np.nonzero([[0, 1], [2, 0]]) = (array([0, 1]), array([1, 0]))
        // First array is row indices, second is column indices
        var a = np.array(new int[,] { { 0, 1 }, { 2, 0 } });
        var result = np.nonzero(a);

        result.Length.Should().Be(2);

        // Row indices
        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(0);
        result[0].GetInt64(1).Should().Be(1);

        // Column indices
        result[1].size.Should().Be(2);
        result[1].GetInt64(0).Should().Be(1);
        result[1].GetInt64(1).Should().Be(0);
    }

    [TestMethod]
    public async Task Nonzero_2D_ThreeElements()
    {
        // NumPy: np.nonzero([[1, 0], [0, 2], [3, 0]]) = (array([0, 1, 2]), array([0, 1, 0]))
        var a = np.array(new int[,] { { 1, 0 }, { 0, 2 }, { 3, 0 } });
        var result = np.nonzero(a);

        result.Length.Should().Be(2);

        // Row indices
        result[0].size.Should().Be(3);
        result[0].GetInt64(0).Should().Be(0);
        result[0].GetInt64(1).Should().Be(1);
        result[0].GetInt64(2).Should().Be(2);

        // Column indices
        result[1].size.Should().Be(3);
        result[1].GetInt64(0).Should().Be(0);
        result[1].GetInt64(1).Should().Be(1);
        result[1].GetInt64(2).Should().Be(0);
    }

    [TestMethod]
    public async Task Nonzero_2D_AllZero_ReturnsEmptyArrays()
    {
        // NumPy: np.nonzero([[0, 0], [0, 0]]) = (array([], dtype=int64), array([], dtype=int64))
        var a = np.array(new int[,] { { 0, 0 }, { 0, 0 } });
        var result = np.nonzero(a);

        result.Length.Should().Be(2);
        result[0].size.Should().Be(0);
        result[1].size.Should().Be(0);
    }

    #endregion

    #region 3D Arrays

    [TestMethod]
    public async Task Nonzero_3D_ReturnsTupleOfThreeIndices()
    {
        // 3D array: each nonzero element has (depth, row, col) index
        var a = np.zeros(new Shape(2, 2, 2), NPTypeCode.Int32);
        a.SetInt32(1, 0, 0, 0); // Position [0,0,0]
        a.SetInt32(1, 1, 1, 1); // Position [1,1,1]

        var result = np.nonzero(a);

        result.Length.Should().Be(3);

        // All index arrays should have 2 elements
        result[0].size.Should().Be(2);
        result[1].size.Should().Be(2);
        result[2].size.Should().Be(2);

        // First nonzero at [0,0,0]
        result[0].GetInt64(0).Should().Be(0);
        result[1].GetInt64(0).Should().Be(0);
        result[2].GetInt64(0).Should().Be(0);

        // Second nonzero at [1,1,1]
        result[0].GetInt64(1).Should().Be(1);
        result[1].GetInt64(1).Should().Be(1);
        result[2].GetInt64(1).Should().Be(1);
    }

    #endregion

    #region Boolean Arrays

    [TestMethod]
    public async Task Nonzero_Boolean_TrueIsTreatedAsNonzero()
    {
        // NumPy: np.nonzero([False, True, False, True]) = (array([1, 3]),)
        var a = np.array(new bool[] { false, true, false, true });
        var result = np.nonzero(a);

        result.Length.Should().Be(1);
        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(1);
        result[0].GetInt64(1).Should().Be(3);
    }

    #endregion

    #region Float Arrays

    [TestMethod]
    public async Task Nonzero_Float_ZeroIsExact()
    {
        // NumPy: np.nonzero([0.0, 1.0, 0.0, 2.0]) = (array([1, 3]),)
        var a = np.array(new double[] { 0.0, 1.0, 0.0, 2.0 });
        var result = np.nonzero(a);

        result.Length.Should().Be(1);
        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(1);
        result[0].GetInt64(1).Should().Be(3);
    }

    [TestMethod]
    public async Task Nonzero_Float_NaN_IsNonzero()
    {
        // NumPy: np.nonzero([0.0, nan]) = (array([1]),) - NaN is treated as nonzero
        var a = np.array(new double[] { 0.0, double.NaN });
        var result = np.nonzero(a);

        result.Length.Should().Be(1);
        result[0].size.Should().Be(1);
        result[0].GetInt64(0).Should().Be(1);
    }

    #endregion
}
