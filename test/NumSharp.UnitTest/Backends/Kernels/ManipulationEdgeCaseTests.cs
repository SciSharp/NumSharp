using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Edge case tests for array manipulation operations.
/// All expected values are verified against NumPy 2.x output.
/// </summary>
public class ManipulationEdgeCaseTests
{
    #region Broadcasting Complex Scenarios

    [Test]
    public void Broadcast_3D_With_2D()
    {
        // NumPy: (2,3,4) + (3,4) = (2,3,4)
        var a = np.ones(new[] { 2, 3, 4 });
        var b = np.ones(new[] { 3, 4 }) * 2;

        var result = a + b;

        Assert.AreEqual(3, result.ndim);
        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(3, result.shape[1]);
        Assert.AreEqual(4, result.shape[2]);
        Assert.AreEqual(3.0, result.GetDouble(0, 0, 0));
    }

    [Test]
    public void Broadcast_3D_With_1D()
    {
        // NumPy: (2,3,4) + (4,) = (2,3,4)
        var a = np.ones(new[] { 2, 3, 4 });
        var b = np.array(new[] { 1.0, 2.0, 3.0, 4.0 });

        var result = a + b;

        Assert.AreEqual(3, result.ndim);
        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(3, result.shape[1]);
        Assert.AreEqual(4, result.shape[2]);
    }

    [Test]
    public void Broadcast_Row_Plus_Column()
    {
        // NumPy: (1,3) + (3,1) = (3,3)
        var row = np.array(new double[,] { { 1, 2, 3 } });  // (1, 3)
        var col = np.array(new double[,] { { 10 }, { 20 }, { 30 } });  // (3, 1)

        var result = row + col;

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(3, result.shape[0]);
        Assert.AreEqual(3, result.shape[1]);
        Assert.AreEqual(11.0, result.GetDouble(0, 0));
        Assert.AreEqual(23.0, result.GetDouble(1, 2));
        Assert.AreEqual(33.0, result.GetDouble(2, 2));
    }

    [Test]
    public void Broadcast_EmptyArray_WithShape()
    {
        // NumPy: np.zeros((0,3)) + np.array([1,2,3]) = shape (0, 3)
        var a = np.zeros(new[] { 0, 3 });
        var b = np.array(new[] { 1.0, 2.0, 3.0 });

        var result = a + b;

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(0, result.shape[0]);
        Assert.AreEqual(3, result.shape[1]);
        Assert.AreEqual(0, result.size);
    }

    #endregion

    #region Squeeze Edge Cases

    [Test]
    public void Squeeze_RemovesAllOnes()
    {
        // NumPy: np.squeeze([[[[1, 2, 3]]]]) = [1, 2, 3] with shape (3,)
        var arr = np.array(new double[,,,] { { { { 1, 2, 3 } } } });  // (1, 1, 1, 3)

        var result = np.squeeze(arr);

        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(3, result.shape[0]);
    }

    [Test]
    public void Squeeze_SpecificAxis()
    {
        // NumPy: np.squeeze(arr, axis=0) where arr.shape = (1, 1, 3)
        var arr = np.ones(new[] { 1, 1, 3 });

        var result = np.squeeze(arr, 0);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(1, result.shape[0]);
        Assert.AreEqual(3, result.shape[1]);
    }

    #endregion

    #region Expand Dims Edge Cases

    [Test]
    public void ExpandDims_NegativeAxis()
    {
        // NumPy: np.expand_dims([1,2,3], -1).shape = (3, 1)
        var arr = np.array(new[] { 1, 2, 3 });

        var result = np.expand_dims(arr, -1);

        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(3, result.shape[0]);
        Assert.AreEqual(1, result.shape[1]);
    }

    [Test]
    public void ExpandDims_Multiple()
    {
        // NumPy: np.expand_dims(np.expand_dims(arr, 0), 0).shape = (1, 1, 3)
        var arr = np.array(new[] { 1, 2, 3 });

        var result = np.expand_dims(np.expand_dims(arr, 0), 0);

        Assert.AreEqual(3, result.ndim);
        Assert.AreEqual(1, result.shape[0]);
        Assert.AreEqual(1, result.shape[1]);
        Assert.AreEqual(3, result.shape[2]);
    }

    #endregion

    #region Concatenate Edge Cases

    [Test]
    public void Concatenate_1D()
    {
        // NumPy: np.concatenate([[1,2,3], [4,5,6]]) = [1,2,3,4,5,6]
        var a = np.array(new[] { 1, 2, 3 });
        var b = np.array(new[] { 4, 5, 6 });

        var result = np.concatenate(new[] { a, b });

        result.Should().BeOfValues(1, 2, 3, 4, 5, 6);
    }

    [Test]
    public void Concatenate_2D_Axis0()
    {
        // NumPy: np.concatenate([[[1,2],[3,4]], [[5,6],[7,8]]], axis=0)
        var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new[,] { { 5, 6 }, { 7, 8 } });

        var result = np.concatenate(new[] { a, b }, axis: 0);

        Assert.AreEqual(4, result.shape[0]);
        Assert.AreEqual(2, result.shape[1]);
        Assert.AreEqual(5, result.GetInt32(2, 0));
    }

    [Test]
    public void Concatenate_2D_Axis1()
    {
        var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new[,] { { 5, 6 }, { 7, 8 } });

        var result = np.concatenate(new[] { a, b }, axis: 1);

        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(4, result.shape[1]);
        Assert.AreEqual(5, result.GetInt32(0, 2));
    }

    [Test]
    public void Concatenate_EmptyWithNonEmpty()
    {
        // NumPy: np.concatenate([[], [1,2]]) = [1., 2.]
        var empty = np.array(new double[0]);
        var b = np.array(new[] { 1.0, 2.0 });

        var result = np.concatenate(new[] { empty, b });

        Assert.AreEqual(2, result.size);
    }

    #endregion

    #region Stack Edge Cases

    [Test]
    public void Stack_DefaultAxis0()
    {
        // NumPy: np.stack([[1,2,3], [4,5,6]]) = [[1,2,3], [4,5,6]]
        var a = np.array(new[] { 1, 2, 3 });
        var b = np.array(new[] { 4, 5, 6 });

        var result = np.stack(new[] { a, b });

        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(3, result.shape[1]);
    }

    [Test]
    public void Stack_Axis1()
    {
        // NumPy: np.stack([[1,2,3], [4,5,6]], axis=1) = [[1,4], [2,5], [3,6]]
        var a = np.array(new[] { 1, 2, 3 });
        var b = np.array(new[] { 4, 5, 6 });

        var result = np.stack(new[] { a, b }, axis: 1);

        Assert.AreEqual(3, result.shape[0]);
        Assert.AreEqual(2, result.shape[1]);
        Assert.AreEqual(1, result.GetInt32(0, 0));
        Assert.AreEqual(4, result.GetInt32(0, 1));
    }

    #endregion

    #region VStack/HStack Edge Cases

    [Test]
    public void VStack_1D()
    {
        // NumPy: np.vstack([[1,2,3], [4,5,6]]) = [[1,2,3], [4,5,6]]
        var a = np.array(new[] { 1, 2, 3 });
        var b = np.array(new[] { 4, 5, 6 });

        var result = np.vstack(a, b);

        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(3, result.shape[1]);
    }

    [Test]
    public void HStack_1D()
    {
        // NumPy: np.hstack([[1,2,3], [4,5,6]]) = [1,2,3,4,5,6]
        var a = np.array(new[] { 1, 2, 3 });
        var b = np.array(new[] { 4, 5, 6 });

        var result = np.hstack(a, b);

        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(6, result.size);
    }

    [Test]
    public void HStack_2D()
    {
        var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new[,] { { 5, 6 }, { 7, 8 } });

        var result = np.hstack(a, b);

        Assert.AreEqual(2, result.shape[0]);
        Assert.AreEqual(4, result.shape[1]);
    }

    #endregion

    #region Repeat Edge Cases

    [Test]
    public void Repeat_Scalar()
    {
        // NumPy: np.repeat([1, 2, 3], 3) = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        var arr = np.array(new[] { 1, 2, 3 });

        var result = np.repeat(arr, 3);

        result.Should().BeOfValues(1, 1, 1, 2, 2, 2, 3, 3, 3);
    }

    [Test]
    [OpenBugs]  // Repeat with per-element counts fails
    public void Repeat_PerElement()
    {
        // NumPy: np.repeat([1, 2, 3], [1, 2, 3]) = [1, 2, 2, 3, 3, 3]
        var arr = np.array(new[] { 1, 2, 3 });
        var repeats = np.array(new[] { 1, 2, 3 });

        var result = np.repeat(arr, repeats);

        result.Should().BeOfValues(1, 2, 2, 3, 3, 3);
    }

    #endregion

    #region Flatten/Ravel Edge Cases

    [Test]
    public void Flatten_ReturnsContiguousCopy()
    {
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });

        var flat = arr.flatten();

        flat.Should().BeOfValues(1, 2, 3, 4);
        Assert.AreEqual(1, flat.ndim);
    }

    [Test]
    public void Ravel_Returns1D()
    {
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });

        var raveled = arr.ravel();

        raveled.Should().BeOfValues(1, 2, 3, 4);
        Assert.AreEqual(1, raveled.ndim);
    }

    [Test]
    public void Ravel_SlicedArray()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5, 6 });
        var sliced = arr["::2"];  // [1, 3, 5]

        var raveled = sliced.ravel();

        raveled.Should().BeOfValues(1, 3, 5);
    }

    #endregion

    #region Reshape Edge Cases

    [Test]
    public void Reshape_InferDimension()
    {
        // NumPy: np.reshape([1,2,3,4,5,6], (-1, 2)) = [[1,2],[3,4],[5,6]]
        var arr = np.array(new[] { 1, 2, 3, 4, 5, 6 });

        var result = arr.reshape(-1, 2);

        Assert.AreEqual(3, result.shape[0]);
        Assert.AreEqual(2, result.shape[1]);
    }

    [Test]
    public void Reshape_ToScalar()
    {
        // NumPy: np.array([5]).reshape(()) = 5 (0D scalar)
        var arr = np.array(new[] { 5 });

        var result = arr.reshape();  // Empty shape = scalar

        Assert.AreEqual(0, result.ndim);
        Assert.AreEqual(1, result.size);
    }

    [Test]
    public void Reshape_EmptyArray()
    {
        // NumPy: np.array([]).reshape((0, 5)).shape = (0, 5)
        var arr = np.array(new double[0]);

        var result = arr.reshape(0, 5);

        Assert.AreEqual(0, result.shape[0]);
        Assert.AreEqual(5, result.shape[1]);
        Assert.AreEqual(0, result.size);
    }

    #endregion

    #region Transpose Edge Cases

    [Test]
    public void Transpose_1D_NoOp()
    {
        // NumPy: arr1d.T.shape = (3,) (no change for 1D)
        var arr = np.array(new[] { 1, 2, 3 });

        var transposed = arr.T;

        Assert.AreEqual(1, transposed.ndim);
        Assert.AreEqual(3, transposed.shape[0]);
    }

    [Test]
    public void Transpose_2D()
    {
        var arr = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        var transposed = arr.T;

        Assert.AreEqual(3, transposed.shape[0]);
        Assert.AreEqual(2, transposed.shape[1]);
        Assert.AreEqual(1, transposed.GetInt32(0, 0));
        Assert.AreEqual(4, transposed.GetInt32(0, 1));
    }

    [Test]
    public void Transpose_3D()
    {
        // NumPy: arr3d.T reverses all axes
        var arr = np.arange(24).reshape(2, 3, 4);

        var transposed = arr.T;

        Assert.AreEqual(4, transposed.shape[0]);
        Assert.AreEqual(3, transposed.shape[1]);
        Assert.AreEqual(2, transposed.shape[2]);
    }

    #endregion

    #region Unique Edge Cases

    [Test]
    [OpenBugs]  // Bug: np.unique doesn't return sorted array
    public void Unique_ReturnsSorted()
    {
        // NumPy: np.unique([3, 1, 2, 1, 3, 2]) = [1, 2, 3] (sorted!)
        var arr = np.array(new[] { 3, 1, 2, 1, 3, 2 });

        var result = np.unique(arr);

        result.Should().BeOfValues(1, 2, 3);
    }

    [Test]
    [OpenBugs]  // Bug: np.unique with NaN values fails
    public void Unique_Float_WithNaN()
    {
        // NumPy: np.unique([1, nan, 2, nan, 1]) = [1, 2, nan]
        var arr = np.array(new[] { 1.0, double.NaN, 2.0, double.NaN, 1.0 });

        var result = np.unique(arr);

        Assert.AreEqual(3, result.size);
        Assert.AreEqual(1.0, result.GetDouble(0));
        Assert.AreEqual(2.0, result.GetDouble(1));
        Assert.IsTrue(double.IsNaN(result.GetDouble(2)));
    }

    #endregion
}
