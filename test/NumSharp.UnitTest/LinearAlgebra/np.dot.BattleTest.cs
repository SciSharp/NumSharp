using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

using NumSharp.UnitTest;

namespace NumSharp.UnitTest.LinearAlgebra;

/// <summary>
/// Battle tests for np.dot to ensure NumPy alignment.
/// Based on running actual NumPy code and verifying NumSharp matches.
/// </summary>
public class DotBattleTests : TestClass
{
    [Test]
    public void Dot_1D_1D_InnerProduct()
    {
        // NumPy: np.dot([1,2,3], [4,5,6]) = 32, dtype=int64
        var a = np.array(new[] { 1, 2, 3 });
        var b = np.array(new[] { 4, 5, 6 });
        var result = np.dot(a, b);
        
        Assert.AreEqual(32, (int)result);
        Assert.AreEqual(0, result.ndim); // Scalar result
    }

    [Test]
    public void Dot_2D_1D()
    {
        // NumPy: [[1,2],[3,4]] dot [5,6] = [17, 39], shape=(2,)
        var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new[] { 5, 6 });
        var result = np.dot(a, b);
        
        result.Should().BeShaped(2);
        result.Should().BeOfValues(17, 39);
    }

    [Test]
    public void Dot_1D_2D()
    {
        // NumPy: [1,2] dot [[3,4],[5,6]] = [13, 16], shape=(2,)
        var a = np.array(new[] { 1, 2 });
        var b = np.array(new[,] { { 3, 4 }, { 5, 6 } });
        var result = np.dot(a, b);
        
        result.Should().BeShaped(2);
        result.Should().BeOfValues(13, 16);
    }

    [Test]
    public void Dot_2D_2D_MatrixMultiply()
    {
        // NumPy: [[1,2],[3,4]] dot [[5,6],[7,8]] = [[19,22],[43,50]]
        var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new[,] { { 5, 6 }, { 7, 8 } });
        var result = np.dot(a, b);
        
        result.Should().BeShaped(2, 2);
        Assert.AreEqual(19, (int)result[0, 0]);
        Assert.AreEqual(22, (int)result[0, 1]);
        Assert.AreEqual(43, (int)result[1, 0]);
        Assert.AreEqual(50, (int)result[1, 1]);
    }

    [Test]
    public void Dot_Scalars()
    {
        // NumPy: np.dot(3, 4) = 12
        var result = np.dot(np.array(3), np.array(4));
        Assert.AreEqual(12, (int)result);
    }

    [Test]
    public void Dot_MixedDtypes_Int32_Float64()
    {
        // NumPy: int32 dot float64 = float64
        var a = np.array(new[] { 1, 2, 3 }).astype(np.int32);
        var b = np.array(new[] { 1.5, 2.5, 3.5 });
        var result = np.dot(a, b);
        
        Assert.AreEqual(17.0, (double)result, 1e-10);
    }

    [Test]
    public void Dot_EmptyArrays()
    {
        // NumPy: dot of empty 1D = 0.0
        var a = np.array(Array.Empty<double>());
        var b = np.array(Array.Empty<double>());
        var result = np.dot(a, b);
        
        Assert.AreEqual(0.0, (double)result);
    }

    [Test]
    public void Dot_3D_2D_HigherDimensions()
    {
        // NumPy: shape(2,3,4) dot shape(4,5) = shape(2,3,5)
        var a = np.arange(24).reshape(2, 3, 4);
        var b = np.arange(20).reshape(4, 5);
        var result = np.dot(a, b);
        
        result.Should().BeShaped(2, 3, 5);
    }

    [Test]
    [OpenBugs]  // NumSharp returns (2,4) instead of (2,3)
    public void Dot_ND_1D()
    {
        // NumPy: shape(2,3,4) dot shape(4,) = shape(2,3)
        // Contracts last axis of a with only axis of b
        var a = np.arange(24).reshape(2, 3, 4);
        var b = np.array(new[] { 1, 2, 3, 4 });
        var result = np.dot(a, b);

        result.Should().BeShaped(2, 3);
        // Expected values: [[20,60,100],[140,180,220]]
        Assert.AreEqual(20, (int)result[0, 0]);
        Assert.AreEqual(220, (int)result[1, 2]);
    }

    [Test]
    public void Dot_StridedArrays_NonContiguous()
    {
        // NumPy: strided (non-contiguous) arrays
        // a = [0, 2, 4, 6, 8], b = [1, 3, 5, 7, 9]
        var full = np.arange(10);
        var a = full["::2"];  // [0, 2, 4, 6, 8]
        var b = full["1::2"]; // [1, 3, 5, 7, 9]
        var result = np.dot(a, b);
        
        // 0*1 + 2*3 + 4*5 + 6*7 + 8*9 = 0 + 6 + 20 + 42 + 72 = 140
        Assert.AreEqual(140, (int)result);
    }

    [Test]
    public void Dot_TransposedArrays()
    {
        // NumPy: (2,3) dot (3,2) = (2,2)
        var a = np.arange(6).reshape(2, 3);
        var b = np.arange(6).reshape(2, 3).T;  // Transposed to (3,2)
        var result = np.dot(a, b);
        
        result.Should().BeShaped(2, 2);
        // [[0,1,2],[3,4,5]] dot [[0,3],[1,4],[2,5]] = [[5,14],[14,50]]
        Assert.AreEqual(5, (int)result[0, 0]);
        Assert.AreEqual(14, (int)result[0, 1]);
        Assert.AreEqual(14, (int)result[1, 0]);
        Assert.AreEqual(50, (int)result[1, 1]);
    }

    [Test]
    public void Dot_LargeMatrices()
    {
        // Verify large matrix multiply works
        var a = np.arange(1000).reshape(100, 10).astype(np.float64);
        var b = np.arange(500).reshape(10, 50).astype(np.float64);
        var result = np.dot(a, b);
        
        result.Should().BeShaped(100, 50);
        // Verify at least one element
        var expected = 0.0;
        for (int k = 0; k < 10; k++)
            expected += (double)a[0, k] * (double)b[k, 0];
        Assert.AreEqual(expected, (double)result[0, 0], 1e-10);
    }

    [Test]
    public void Dot_ColumnVector_RowVector()
    {
        // (3,1) dot (1,3) = (3,3) outer product
        // [[1],[2],[3]] dot [[4,5,6]] = [[4,5,6],[8,10,12],[12,15,18]]
        var col = np.array(new[] { 1, 2, 3 }).reshape(3, 1);
        var row = np.array(new[] { 4, 5, 6 }).reshape(1, 3);
        var result = np.dot(col, row);

        result.Should().BeShaped(3, 3);
        Assert.AreEqual(4, (int)result[0, 0]);   // 1 * 4 = 4
        Assert.AreEqual(18, (int)result[2, 2]);  // 3 * 6 = 18
    }

    [Test]
    public void Dot_RowVector_ColumnVector()
    {
        // (1,3) dot (3,1) = (1,1) scalar in matrix form
        var row = np.array(new[] { 1, 2, 3 }).reshape(1, 3);
        var col = np.array(new[] { 4, 5, 6 }).reshape(3, 1);
        var result = np.dot(row, col);
        
        result.Should().BeShaped(1, 1);
        Assert.AreEqual(32, (int)result[0, 0]);
    }
}
