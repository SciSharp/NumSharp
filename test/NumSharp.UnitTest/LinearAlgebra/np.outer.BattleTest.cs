using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra;

/// <summary>
/// Battle tests for np.outer to ensure NumPy alignment.
/// </summary>
[TestClass]
public class OuterBattleTests : TestClass
{
    [TestMethod]
    public void Outer_1D_1D()
    {
        // NumPy: outer([1,2,3], [4,5]) = [[4,5],[8,10],[12,15]]
        var a = np.array(new[] { 1, 2, 3 });
        var b = np.array(new[] { 4, 5 });
        var result = np.outer(a, b);

        result.Should().BeShaped(3, 2);
        Assert.AreEqual(4, (int)result[0, 0]);
        Assert.AreEqual(5, (int)result[0, 1]);
        Assert.AreEqual(8, (int)result[1, 0]);
        Assert.AreEqual(10, (int)result[1, 1]);
        Assert.AreEqual(12, (int)result[2, 0]);
        Assert.AreEqual(15, (int)result[2, 1]);
    }

    [TestMethod]
    public void Outer_DifferentSizes()
    {
        // NumPy: outer(arange(5), arange(3)) has shape (5,3)
        var a = np.arange(5);
        var b = np.arange(3);
        var result = np.outer(a, b);

        result.Should().BeShaped(5, 3);
        // First row is all zeros (0 * anything)
        Assert.AreEqual(0, (int)result[0, 0]);
        Assert.AreEqual(0, (int)result[0, 1]);
        Assert.AreEqual(0, (int)result[0, 2]);
        // [4] * [0,1,2] = [0,4,8]
        Assert.AreEqual(0, (int)result[4, 0]);
        Assert.AreEqual(4, (int)result[4, 1]);
        Assert.AreEqual(8, (int)result[4, 2]);
    }

    [TestMethod]
    public void Outer_2D_Flattened()
    {
        // NumPy flattens 2D inputs: outer([[1,2],[3,4]], [[5,6],[7,8]]) has shape (4,4)
        var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new[,] { { 5, 6 }, { 7, 8 } });
        var result = np.outer(a, b);

        result.Should().BeShaped(4, 4);
        // First element: 1*5 = 5
        Assert.AreEqual(5, (int)result[0, 0]);
        // Last element: 4*8 = 32
        Assert.AreEqual(32, (int)result[3, 3]);
    }

    [TestMethod]
    public void Outer_WithFloats()
    {
        var a = np.array(new[] { 1.0, 2.0, 3.0 });
        var b = np.array(new[] { 0.5, 1.5 });
        var result = np.outer(a, b);

        result.Should().BeShaped(3, 2);
        Assert.AreEqual(0.5, (double)result[0, 0], 1e-10);
        Assert.AreEqual(1.5, (double)result[0, 1], 1e-10);
        Assert.AreEqual(4.5, (double)result[2, 1], 1e-10);  // 3.0 * 1.5
    }

    [TestMethod]
    public void Outer_SingleElement()
    {
        var a = np.array(new[] { 5 });
        var b = np.array(new[] { 3 });
        var result = np.outer(a, b);

        result.Should().BeShaped(1, 1);
        Assert.AreEqual(15, (int)result[0, 0]);
    }
}
