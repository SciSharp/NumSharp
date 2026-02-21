using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Tests for linear algebra operations.
/// All expected values are verified against NumPy 2.x output.
/// </summary>
public class LinearAlgebraTests
{
    #region Dot Product

    [Test]
    public void Dot_1D_1D()
    {
        // NumPy: np.dot([1,2,3], [4,5,6]) = 32
        var a = np.array(new[] { 1.0, 2.0, 3.0 });
        var b = np.array(new[] { 4.0, 5.0, 6.0 });

        var result = np.dot(a, b);

        Assert.AreEqual(32.0, (double)result, 1e-10);
    }

    [Test]
    public void Dot_2D_1D()
    {
        // NumPy: np.dot([[1,2],[3,4],[5,6]], [1,2]) = [5, 11, 17]
        var A = np.array(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } });
        var x = np.array(new[] { 1.0, 2.0 });

        var result = np.dot(A, x);

        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(3, result.size);
        Assert.AreEqual(5.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(11.0, result.GetDouble(1), 1e-10);
        Assert.AreEqual(17.0, result.GetDouble(2), 1e-10);
    }

    [Test]
    [OpenBugs]  // 1D @ 2D dot product fails
    public void Dot_1D_2D()
    {
        // NumPy: np.dot([1,2,3], [[1,2],[3,4],[5,6]]) = [22, 28]
        var y = np.array(new[] { 1.0, 2.0, 3.0 });
        var A = np.array(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } });

        var result = np.dot(y, A);

        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(22.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(28.0, result.GetDouble(1), 1e-10);
    }

    [Test]
    public void Dot_2D_2D()
    {
        // NumPy: [[1,2],[3,4],[5,6]] @ [[1,2,3],[4,5,6]] = [[9,12,15],[19,26,33],[29,40,51]]
        var A = np.array(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } });
        var B = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });

        var result = np.dot(A, B);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(3, result.shape[0]);
        Assert.AreEqual(3, result.shape[1]);
        Assert.AreEqual(9.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(26.0, result.GetDouble(1, 1), 1e-10);
        Assert.AreEqual(51.0, result.GetDouble(2, 2), 1e-10);
    }

    [Test]
    public void Dot_Empty()
    {
        // NumPy: np.dot([], []) = 0.0
        var a = np.array(new double[0]);
        var b = np.array(new double[0]);

        var result = np.dot(a, b);

        Assert.AreEqual(0.0, (double)result, 1e-10);
    }

    #endregion

    #region Matmul

    [Test]
    public void Matmul_2D_2D()
    {
        // NumPy: A @ B (same as dot for 2D)
        var A = np.array(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var B = np.array(new[,] { { 5.0, 6.0 }, { 7.0, 8.0 } });

        var result = np.matmul(A, B);

        Assert.AreEqual(19.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(22.0, result.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(43.0, result.GetDouble(1, 0), 1e-10);
        Assert.AreEqual(50.0, result.GetDouble(1, 1), 1e-10);
    }

    [Test]
    [OpenBugs]  // 3D matmul broadcasting fails
    public void Matmul_3D_2D_Broadcasting()
    {
        // NumPy: (2, 3, 4) @ (4, 5) = (2, 3, 5)
        var batch = np.ones(new[] { 2, 3, 4 });
        var mat = np.ones(new[] { 4, 5 }) * 2;

        var result = np.matmul(batch, mat);

        Assert.AreEqual(3, result.ndim);
        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(3, result.shape[1]);
        Assert.AreEqual(5, result.shape[2]);
        Assert.AreEqual(8.0, result.GetDouble(0, 0, 0), 1e-10);  // 4 * 2 = 8
    }

    #endregion

    #region Outer Product

    [Test]
    public void Outer_Simple()
    {
        // NumPy: np.outer([1,2,3], [10,20]) = [[10,20],[20,40],[30,60]]
        var a = np.array(new[] { 1.0, 2.0, 3.0 });
        var b = np.array(new[] { 10.0, 20.0 });

        var result = np.outer(a, b);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(3, result.shape[0]);
        Assert.AreEqual(2, result.shape[1]);
        Assert.AreEqual(10.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(40.0, result.GetDouble(1, 1), 1e-10);
        Assert.AreEqual(60.0, result.GetDouble(2, 1), 1e-10);
    }

    #endregion

    #region Identity and Eye

    [Test]
    public void Eye_Square()
    {
        // NumPy: np.eye(3) = [[1,0,0],[0,1,0],[0,0,1]]
        var result = np.eye(3);

        Assert.AreEqual(3, result.shape[0]);
        Assert.AreEqual(3, result.shape[1]);
        Assert.AreEqual(1.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(0.0, result.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(1.0, result.GetDouble(1, 1), 1e-10);
        Assert.AreEqual(1.0, result.GetDouble(2, 2), 1e-10);
    }

    [Test]
    public void Eye_Rectangular()
    {
        // NumPy: np.eye(3, 4)
        var result = np.eye(3, 4);

        Assert.AreEqual(3, result.shape[0]);
        Assert.AreEqual(4, result.shape[1]);
        Assert.AreEqual(1.0, result.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(0.0, result.GetDouble(0, 3), 1e-10);
    }

    #endregion

    #region Statistics-Based Linear Algebra

    [Test]
    public void Mean_2D_Axis0()
    {
        // NumPy: np.mean([[1,2,3],[4,5,6]], axis=0) = [2.5, 3.5, 4.5]
        var arr = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });

        var result = np.mean(arr, axis: 0);

        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(3, result.size);
        Assert.AreEqual(2.5, result.GetDouble(0), 1e-10);
        Assert.AreEqual(3.5, result.GetDouble(1), 1e-10);
        Assert.AreEqual(4.5, result.GetDouble(2), 1e-10);
    }

    [Test]
    public void Mean_2D_Axis1()
    {
        // NumPy: np.mean([[1,2,3],[4,5,6]], axis=1) = [2.0, 5.0]
        var arr = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });

        var result = np.mean(arr, axis: 1);

        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(2.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(5.0, result.GetDouble(1), 1e-10);
    }

    [Test]
    public void Std_Sample()
    {
        // NumPy: np.std([1, 2, 3, 4, 5]) = 1.4142... (population std)
        var arr = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        var result = np.std(arr);

        Assert.AreEqual(Math.Sqrt(2.0), result.GetDouble(0), 1e-10);
    }

    [Test]
    [OpenBugs]  // std with ddof parameter fails
    public void Std_WithDdof()
    {
        // NumPy: np.std([1, 2, 3, 4, 5], ddof=1) = 1.5811... (sample std)
        var arr = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        var result = np.std(arr, ddof: 1);

        // Sample std = sqrt(sum((x-mean)^2) / (n-1))
        Assert.AreEqual(Math.Sqrt(2.5), result.GetDouble(0), 1e-10);
    }

    #endregion

    #region Argmax/Argmin with Axis

    [Test]
    public void Argmax_NoAxis()
    {
        // NumPy: np.argmax([[1,5,3],[4,2,6]]) = 5 (flattened index)
        var arr = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });

        var result = np.argmax(arr);

        Assert.AreEqual(5, result);
    }

    [Test]
    public void Argmax_Axis0()
    {
        // NumPy: np.argmax([[1,5,3],[4,2,6]], axis=0) = [1, 0, 1]
        var arr = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });

        var result = np.argmax(arr, axis: 0);

        result.Should().BeOfValues(1, 0, 1);
    }

    [Test]
    public void Argmax_Axis1()
    {
        // NumPy: np.argmax([[1,5,3],[4,2,6]], axis=1) = [1, 2]
        var arr = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });

        var result = np.argmax(arr, axis: 1);

        result.Should().BeOfValues(1, 2);
    }

    [Test]
    public void Argmin_NoAxis()
    {
        // NumPy: np.argmin([[1,5,3],[4,2,6]]) = 0
        var arr = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });

        var result = np.argmin(arr);

        Assert.AreEqual(0, result);
    }

    [Test]
    public void Argmin_Axis0()
    {
        // NumPy: np.argmin([[1,5,3],[4,2,6]], axis=0) = [0, 1, 0]
        var arr = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });

        var result = np.argmin(arr, axis: 0);

        result.Should().BeOfValues(0, 1, 0);
    }

    #endregion

    #region Cumsum with Axis

    [Test]
    public void Cumsum_NoAxis_Flattens()
    {
        // NumPy: np.cumsum([[1,2,3],[4,5,6]]) = [1,3,6,10,15,21]
        var arr = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        var result = np.cumsum(arr);

        Assert.AreEqual(1, result.ndim);
        result.Should().BeOfValues(1, 3, 6, 10, 15, 21);
    }

    [Test]
    public void Cumsum_Axis0()
    {
        // NumPy: np.cumsum([[1,2,3],[4,5,6]], axis=0) = [[1,2,3],[5,7,9]]
        var arr = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        var result = np.cumsum(arr, axis: 0);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(1, result.GetInt32(0, 0));
        Assert.AreEqual(5, result.GetInt32(1, 0));
        Assert.AreEqual(9, result.GetInt32(1, 2));
    }

    [Test]
    public void Cumsum_Axis1()
    {
        // NumPy: np.cumsum([[1,2,3],[4,5,6]], axis=1) = [[1,3,6],[4,9,15]]
        var arr = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        var result = np.cumsum(arr, axis: 1);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(1, result.GetInt32(0, 0));
        Assert.AreEqual(6, result.GetInt32(0, 2));
        Assert.AreEqual(15, result.GetInt32(1, 2));
    }

    #endregion

    #region Searchsorted

    [Test]
    [OpenBugs]  // searchsorted returns NDArray not int
    public void Searchsorted_Simple()
    {
        // NumPy: np.searchsorted([1,2,3,4,5], 3) = 2
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        var result = np.searchsorted(arr, 3);

        Assert.AreEqual(2, result);
    }

    [Test]
    [OpenBugs]  // searchsorted returns NDArray not int
    public void Searchsorted_BeforeAll()
    {
        // NumPy: np.searchsorted([1,2,3,4,5], 0) = 0
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        var result = np.searchsorted(arr, 0);

        Assert.AreEqual(0, result);
    }

    [Test]
    [OpenBugs]  // searchsorted returns NDArray not int
    public void Searchsorted_AfterAll()
    {
        // NumPy: np.searchsorted([1,2,3,4,5], 10) = 5
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });

        var result = np.searchsorted(arr, 10);

        Assert.AreEqual(5, result);
    }

    #endregion
}
