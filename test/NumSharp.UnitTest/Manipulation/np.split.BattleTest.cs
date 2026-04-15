using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation;

/// <summary>
/// Battle tests for np.hsplit, np.vsplit, np.dsplit to ensure NumPy alignment.
/// Based on NumPy 2.4.2 behavior.
/// </summary>
public class SplitBattleTests : TestClass
{
    #region hsplit tests

    [TestMethod]
    public void Hsplit_2D_SplitsColumns()
    {
        // NumPy: a = np.arange(16).reshape(4,4); np.hsplit(a, 2)
        // Returns two arrays of shape (4, 2)
        var a = np.arange(16).reshape(4, 4);
        var result = np.hsplit(a, 2);

        Assert.AreEqual(2, result.Length);
        result[0].Should().BeShaped(4, 2);
        result[1].Should().BeShaped(4, 2);

        // First half: columns 0-1
        Assert.AreEqual(0, (int)result[0][0, 0]);
        Assert.AreEqual(1, (int)result[0][0, 1]);
        Assert.AreEqual(4, (int)result[0][1, 0]);
        Assert.AreEqual(5, (int)result[0][1, 1]);

        // Second half: columns 2-3
        Assert.AreEqual(2, (int)result[1][0, 0]);
        Assert.AreEqual(3, (int)result[1][0, 1]);
        Assert.AreEqual(6, (int)result[1][1, 0]);
        Assert.AreEqual(7, (int)result[1][1, 1]);
    }

    [TestMethod]
    public void Hsplit_1D_SplitsAlongAxis0()
    {
        // NumPy: a = np.arange(6); np.hsplit(a, 3)
        // Returns three arrays of shape (2,)
        var a = np.arange(6);
        var result = np.hsplit(a, 3);

        Assert.AreEqual(3, result.Length);
        result[0].Should().BeShaped(2);
        result[1].Should().BeShaped(2);
        result[2].Should().BeShaped(2);

        Assert.AreEqual(0, (int)result[0][0]);
        Assert.AreEqual(1, (int)result[0][1]);
        Assert.AreEqual(2, (int)result[1][0]);
        Assert.AreEqual(3, (int)result[1][1]);
        Assert.AreEqual(4, (int)result[2][0]);
        Assert.AreEqual(5, (int)result[2][1]);
    }

    [TestMethod]
    public void Hsplit_3D_SplitsAlongAxis1()
    {
        // NumPy: a = np.arange(24).reshape(2, 4, 3); np.hsplit(a, 2)
        // Returns two arrays of shape (2, 2, 3)
        var a = np.arange(24).reshape(2, 4, 3);
        var result = np.hsplit(a, 2);

        Assert.AreEqual(2, result.Length);
        result[0].Should().BeShaped(2, 2, 3);
        result[1].Should().BeShaped(2, 2, 3);
    }

    [TestMethod]
    public void Hsplit_WithIndices()
    {
        // NumPy: a = np.arange(16).reshape(4,4); np.hsplit(a, [1, 3])
        // Returns three arrays of shapes (4, 1), (4, 2), (4, 1)
        var a = np.arange(16).reshape(4, 4);
        var result = np.hsplit(a, new[] { 1, 3 });

        Assert.AreEqual(3, result.Length);
        result[0].Should().BeShaped(4, 1);
        result[1].Should().BeShaped(4, 2);
        result[2].Should().BeShaped(4, 1);
    }

    [TestMethod]
    public void Hsplit_0D_ThrowsArgumentException()
    {
        // NumPy: np.hsplit(np.array(5), 1) raises ValueError
        var a = np.array(5);  // scalar -> 0D array
        Assert.ThrowsException<System.ArgumentException>(() => np.hsplit(a, 1));
    }

    #endregion

    #region vsplit tests

    [TestMethod]
    public void Vsplit_2D_SplitsRows()
    {
        // NumPy: a = np.arange(16).reshape(4,4); np.vsplit(a, 2)
        // Returns two arrays of shape (2, 4)
        var a = np.arange(16).reshape(4, 4);
        var result = np.vsplit(a, 2);

        Assert.AreEqual(2, result.Length);
        result[0].Should().BeShaped(2, 4);
        result[1].Should().BeShaped(2, 4);

        // First half: rows 0-1
        Assert.AreEqual(0, (int)result[0][0, 0]);
        Assert.AreEqual(1, (int)result[0][0, 1]);
        Assert.AreEqual(4, (int)result[0][1, 0]);
        Assert.AreEqual(7, (int)result[0][1, 3]);

        // Second half: rows 2-3
        Assert.AreEqual(8, (int)result[1][0, 0]);
        Assert.AreEqual(9, (int)result[1][0, 1]);
        Assert.AreEqual(12, (int)result[1][1, 0]);
        Assert.AreEqual(15, (int)result[1][1, 3]);
    }

    [TestMethod]
    public void Vsplit_3D_SplitsAlongAxis0()
    {
        // NumPy: a = np.arange(8).reshape(2, 2, 2); np.vsplit(a, 2)
        // Returns two arrays of shape (1, 2, 2)
        var a = np.arange(8).reshape(2, 2, 2);
        var result = np.vsplit(a, 2);

        Assert.AreEqual(2, result.Length);
        result[0].Should().BeShaped(1, 2, 2);
        result[1].Should().BeShaped(1, 2, 2);
    }

    [TestMethod]
    public void Vsplit_WithIndices()
    {
        // NumPy: a = np.arange(16).reshape(4,4); np.vsplit(a, [3, 6])
        // Returns three arrays
        var a = np.arange(16).reshape(4, 4);
        var result = np.vsplit(a, new[] { 3, 6 });

        Assert.AreEqual(3, result.Length);
        result[0].Should().BeShaped(3, 4);
        result[1].Should().BeShaped(1, 4);
        result[2].Should().BeShaped(0, 4);  // Empty array
    }

    [TestMethod]
    public void Vsplit_1D_ThrowsArgumentException()
    {
        // NumPy: np.vsplit(np.arange(6), 2) raises ValueError
        var a = np.arange(6);
        Assert.ThrowsException<System.ArgumentException>(() => np.vsplit(a, 2));
    }

    #endregion

    #region dsplit tests

    [TestMethod]
    public void Dsplit_3D_SplitsDepth()
    {
        // NumPy: a = np.arange(24).reshape(2, 3, 4); np.dsplit(a, 2)
        // Returns two arrays of shape (2, 3, 2)
        var a = np.arange(24).reshape(2, 3, 4);
        var result = np.dsplit(a, 2);

        Assert.AreEqual(2, result.Length);
        result[0].Should().BeShaped(2, 3, 2);
        result[1].Should().BeShaped(2, 3, 2);
    }

    [TestMethod]
    public void Dsplit_4D_SplitsAlongAxis2()
    {
        // NumPy: a = np.arange(48).reshape(2, 3, 4, 2); np.dsplit(a, 2)
        // Returns two arrays of shape (2, 3, 2, 2)
        var a = np.arange(48).reshape(2, 3, 4, 2);
        var result = np.dsplit(a, 2);

        Assert.AreEqual(2, result.Length);
        result[0].Should().BeShaped(2, 3, 2, 2);
        result[1].Should().BeShaped(2, 3, 2, 2);
    }

    [TestMethod]
    public void Dsplit_WithIndices()
    {
        // NumPy: a = np.arange(16).reshape(2, 2, 4); np.dsplit(a, [3, 6])
        // Returns three arrays
        var a = np.arange(16).reshape(2, 2, 4);
        var result = np.dsplit(a, new[] { 3, 6 });

        Assert.AreEqual(3, result.Length);
        result[0].Should().BeShaped(2, 2, 3);
        result[1].Should().BeShaped(2, 2, 1);
        result[2].Should().BeShaped(2, 2, 0);  // Empty array
    }

    [TestMethod]
    public void Dsplit_2D_ThrowsArgumentException()
    {
        // NumPy: np.dsplit(np.arange(16).reshape(4,4), 2) raises ValueError
        var a = np.arange(16).reshape(4, 4);
        Assert.ThrowsException<System.ArgumentException>(() => np.dsplit(a, 2));
    }

    [TestMethod]
    public void Dsplit_1D_ThrowsArgumentException()
    {
        // NumPy: np.dsplit(np.arange(6), 2) raises ValueError
        var a = np.arange(6);
        Assert.ThrowsException<System.ArgumentException>(() => np.dsplit(a, 2));
    }

    #endregion
}
