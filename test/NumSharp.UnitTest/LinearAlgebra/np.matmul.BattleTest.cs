using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra;

/// <summary>
/// Battle tests for np.matmul (@ operator) to ensure NumPy alignment.
/// </summary>
public class MatmulBattleTests : TestClass
{
    [TestMethod]
    public void Matmul_2D_2D()
    {
        // NumPy: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new[,] { { 5, 6 }, { 7, 8 } });
        var result = np.matmul(a, b);

        result.Should().BeShaped(2, 2);
        Assert.AreEqual(19, (int)result[0, 0]);
        Assert.AreEqual(22, (int)result[0, 1]);
        Assert.AreEqual(43, (int)result[1, 0]);
        Assert.AreEqual(50, (int)result[1, 1]);
    }

    [TestMethod]
    [OpenBugs]  // NumSharp requires 2D inputs, doesn't support 1D @ 1D
    public void Matmul_1D_1D_DotProduct()
    {
        // NumPy: [1,2,3] @ [4,5,6] = 32 (scalar)
        var a = np.array(new[] { 1, 2, 3 });
        var b = np.array(new[] { 4, 5, 6 });
        var result = np.matmul(a, b);

        Assert.AreEqual(32, (int)result);
    }

    [TestMethod]
    [OpenBugs]  // NumSharp returns shape (2,1) instead of (2,)
    public void Matmul_2D_1D()
    {
        // NumPy: [[1,2],[3,4]] @ [5,6] = [17, 39], shape=(2,)
        var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new[] { 5, 6 });
        var result = np.matmul(a, b);

        result.Should().BeShaped(2);
        result.Should().BeOfValues(17, 39);
    }

    [TestMethod]
    [OpenBugs]  // NumSharp throws dimension mismatch for 1D @ 2D
    public void Matmul_1D_2D()
    {
        // NumPy: [1,2] @ [[3,4],[5,6]] = [13, 16], shape=(2,)
        var a = np.array(new[] { 1, 2 });
        var b = np.array(new[,] { { 3, 4 }, { 5, 6 } });
        var result = np.matmul(a, b);

        result.Should().BeShaped(2);
        result.Should().BeOfValues(13, 16);
    }

    [TestMethod]
    [OpenBugs]  // NumSharp crashes on >2D broadcasting
    public void Matmul_Broadcasting_3D_2D()
    {
        // NumPy: (3,2,2) @ (2,2) = (3,2,2) via broadcasting
        var a = np.arange(12).reshape(3, 2, 2);
        var identity = np.array(new[,] { { 1, 0 }, { 0, 1 } });
        var result = np.matmul(a, identity);

        result.Should().BeShaped(3, 2, 2);
        // First block should equal itself (multiplied by identity)
        Assert.AreEqual(0, (int)result[0, 0, 0]);
        Assert.AreEqual(1, (int)result[0, 0, 1]);
        Assert.AreEqual(2, (int)result[0, 1, 0]);
        Assert.AreEqual(3, (int)result[0, 1, 1]);
    }

    [TestMethod]
    public void Matmul_LargeMatrices()
    {
        var a = np.arange(1000).reshape(100, 10).astype(np.float64);
        var b = np.arange(500).reshape(10, 50).astype(np.float64);
        var result = np.matmul(a, b);

        result.Should().BeShaped(100, 50);
    }

    [TestMethod]
    public void Matmul_Transposed()
    {
        // (2,3) @ (3,2) = (2,2)
        var a = np.arange(6).reshape(2, 3);
        var b = np.arange(6).reshape(2, 3).T;
        var result = np.matmul(a, b);

        result.Should().BeShaped(2, 2);
        Assert.AreEqual(5, (int)result[0, 0]);
        Assert.AreEqual(50, (int)result[1, 1]);
    }
}
