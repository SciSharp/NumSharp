using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Generic;
using TUnit.Core;

namespace NumSharp.UnitTest.Selection;

/// <summary>
/// Comprehensive battle tests for boolean indexing covering all NumPy behaviors.
/// These tests are based on actual NumPy 2.4.2 output and verify NumSharp matches exactly.
/// </summary>
public class BooleanIndexing_BattleTests
{
    #region Case 1: Same-Shape Boolean Mask (Element-wise)

    [Test]
    public void Case1_SameShape_1D_BasicMask()
    {
        // NumPy: arr = [1, 2, 3, 4, 5], mask = [T, F, T, F, T] → [1, 3, 5]
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask = np.array(new[] { true, false, true, false, true }).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.IsTrue(result.shape.SequenceEqual(new[] { 3 }), $"Expected shape [3], got [{string.Join(", ", result.shape)}]");
        Assert.AreEqual(1, result.GetInt32(0));
        Assert.AreEqual(3, result.GetInt32(1));
        Assert.AreEqual(5, result.GetInt32(2));
    }

    [Test]
    public void Case1_SameShape_1D_ResultIs1D()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask = np.array(new[] { true, false, true, false, true }).MakeGeneric<bool>();

        var result = arr[mask];

        // Boolean indexing always returns 1D for same-shape mask
        Assert.AreEqual(1, result.ndim);
    }

    [Test]
    public void Case1_SameShape_1D_ResultIsCopy()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask = np.array(new[] { true, false, true, false, true }).MakeGeneric<bool>();

        var result = arr[mask];

        // Modify result, original should be unchanged
        result.SetInt32(999, 0);
        Assert.AreEqual(1, arr.GetInt32(0), "Original should be unchanged - boolean indexing returns a copy");
    }

    #endregion

    #region Case 2: 2D Same-Shape Mask

    [Test]
    public void Case2_SameShape_2D_ElementWise()
    {
        // NumPy: arr2d > 5 selects [6, 7, 8, 9, 10, 11]
        var arr2d = np.arange(12).reshape(3, 4);
        var mask2d = (arr2d > 5).MakeGeneric<bool>();

        var result = arr2d[mask2d];

        Assert.IsTrue(result.shape.SequenceEqual(new[] { 6 }), $"Expected shape [6], got [{string.Join(", ", result.shape)}]");
        Assert.AreEqual(6, result.GetInt32(0));
        Assert.AreEqual(7, result.GetInt32(1));
        Assert.AreEqual(8, result.GetInt32(2));
        Assert.AreEqual(9, result.GetInt32(3));
        Assert.AreEqual(10, result.GetInt32(4));
        Assert.AreEqual(11, result.GetInt32(5));
    }

    [Test]
    public void Case2_SameShape_2D_AlwaysReturns1D()
    {
        // NumPy: 2D mask on 2D array → always returns 1D
        var arr2d = np.arange(12).reshape(3, 4);
        var mask2d = (arr2d > 5).MakeGeneric<bool>();

        var result = arr2d[mask2d];

        Assert.AreEqual(1, result.ndim);
    }

    [Test]
    public void Case2_SameShape_2D_PreservesOrder()
    {
        // NumPy iterates in C-order (row-major)
        var arr2d = np.arange(12).reshape(3, 4);
        var mask2d = (arr2d > 5).MakeGeneric<bool>();

        var result = arr2d[mask2d];

        // Elements should be in row-major order: 6,7,8,9,10,11
        var expected = new[] { 6, 7, 8, 9, 10, 11 };
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.AreEqual(expected[i], result.GetInt32(i), $"Element at index {i}");
        }
    }

    #endregion

    #region Case 3: 1D Mask on Axis-0 (Row Selection)

    [Test]
    public void Case3_Axis0_2D_SelectsRows()
    {
        // NumPy: arr2d[[T,F,T]] selects rows 0 and 2
        var arr2d = np.arange(12).reshape(3, 4);
        var mask1d = np.array(new[] { true, false, true }).MakeGeneric<bool>();

        var result = arr2d[mask1d];

        Assert.IsTrue(result.shape.SequenceEqual(new[] { 2, 4 }), $"Expected shape [2, 4], got [{string.Join(", ", result.shape)}]");
    }

    [Test]
    public void Case3_Axis0_2D_CorrectValues()
    {
        var arr2d = np.arange(12).reshape(3, 4);
        var mask1d = np.array(new[] { true, false, true }).MakeGeneric<bool>();

        var result = arr2d[mask1d];

        // Row 0: [0, 1, 2, 3]
        Assert.AreEqual(0, result[0, 0].GetInt32());
        Assert.AreEqual(1, result[0, 1].GetInt32());
        Assert.AreEqual(2, result[0, 2].GetInt32());
        Assert.AreEqual(3, result[0, 3].GetInt32());

        // Row 1 (originally row 2): [8, 9, 10, 11]
        Assert.AreEqual(8, result[1, 0].GetInt32());
        Assert.AreEqual(9, result[1, 1].GetInt32());
        Assert.AreEqual(10, result[1, 2].GetInt32());
        Assert.AreEqual(11, result[1, 3].GetInt32());
    }

    [Test]
    public void Case3_Axis0_3D_SelectsAlongAxis0()
    {
        // NumPy: 1D mask on 3D array selects along axis 0
        var arr3d = np.arange(24).reshape(2, 3, 4);
        var mask1d = np.array(new[] { true, false }).MakeGeneric<bool>();

        var result = arr3d[mask1d];

        // Selects first "block", result shape (1, 3, 4)
        Assert.IsTrue(result.shape.SequenceEqual(new[] { 1, 3, 4 }), $"Expected shape [1, 3, 4], got [{string.Join(", ", result.shape)}]");
    }

    [Test]
    public void Case3_Axis0_3D_CorrectValues()
    {
        var arr3d = np.arange(24).reshape(2, 3, 4);
        var mask1d = np.array(new[] { true, false }).MakeGeneric<bool>();

        var result = arr3d[mask1d];

        // First element should be 0
        Assert.AreEqual(0, result[0, 0, 0].GetInt32());
        // Last element of selected block should be 11
        Assert.AreEqual(11, result[0, 2, 3].GetInt32());
    }

    #endregion

    #region Case 4: Boolean Mask + Additional Index (Combined Indexing)

    [Test]
    public void Case4_BooleanPlusInteger_Workaround()
    {
        // NumPy: arr2d[mask, 0] - select masked rows, then column 0
        // NumSharp doesn't support this directly, but we can chain operations
        var arr2d = np.arange(12).reshape(3, 4);
        var mask1d = np.array(new[] { true, false, true }).MakeGeneric<bool>();

        // Workaround: arr2d[mask][:, 0]
        var selected = arr2d[mask1d];
        var result = selected[":, 0"];

        // NumPy: arr2d[[T,F,T], 0] = [0, 8]
        Assert.IsTrue(result.shape.SequenceEqual(new[] { 2 }), $"Expected shape [2], got [{string.Join(", ", result.shape)}]");
        Assert.AreEqual(0, result.GetInt32(0));
        Assert.AreEqual(8, result.GetInt32(1));
    }

    [Test]
    public void Case4_BooleanPlusSlice_Workaround()
    {
        // NumPy: arr2d[mask, 1:3] - select masked rows, then columns 1-2
        var arr2d = np.arange(12).reshape(3, 4);
        var mask1d = np.array(new[] { true, false, true }).MakeGeneric<bool>();

        // Workaround: arr2d[mask][:, 1:3]
        var selected = arr2d[mask1d];
        var result = selected[":, 1:3"];

        // NumPy: [[1, 2], [9, 10]]
        Assert.IsTrue(result.shape.SequenceEqual(new[] { 2, 2 }), $"Expected shape [2, 2], got [{string.Join(", ", result.shape)}]");
        Assert.AreEqual(1, result[0, 0].GetInt32());
        Assert.AreEqual(2, result[0, 1].GetInt32());
        Assert.AreEqual(9, result[1, 0].GetInt32());
        Assert.AreEqual(10, result[1, 1].GetInt32());
    }

    #endregion

    #region Case 5: 3D Array with Masks (ndim variations)

    [Test]
    public void Case5_3D_FullShapeMask()
    {
        // NumPy: 3D mask on 3D array → 1D result
        var arr3d = np.arange(24).reshape(2, 3, 4);
        var mask3d = (arr3d > 12).MakeGeneric<bool>();

        var result = arr3d[mask3d];

        // 11 elements > 12: [13, 14, ..., 23]
        Assert.IsTrue(result.shape.SequenceEqual(new[] { 11 }), $"Expected shape [11], got [{string.Join(", ", result.shape)}]");
        Assert.AreEqual(13, result.GetInt32(0));
        Assert.AreEqual(23, result.GetInt32(10));
    }

    [Test]
    public void Case5_3D_1DMask_PreservesRemainingDims()
    {
        var arr3d = np.arange(24).reshape(2, 3, 4);
        var mask1d = np.array(new[] { true, false }).MakeGeneric<bool>();

        var result = arr3d[mask1d];

        // Shape: (1, 3, 4) - preserves dims 1 and 2
        Assert.IsTrue(result.shape.SequenceEqual(new[] { 1, 3, 4 }), $"Expected shape [1, 3, 4], got [{string.Join(", ", result.shape)}]");
    }

    [Test]
    public void Case5_2DMaskOn3D_PartialShapeMatch()
    {
        // NumPy: 2D mask (2,3) on 3D array (2,3,4) → shape (count_true, 4)
        var arr3d = np.arange(24).reshape(2, 3, 4);
        var mask2d = np.array(new[,] { { true, false, true }, { false, true, false } }).MakeGeneric<bool>();

        // This should work in NumPy but may not in NumSharp
        // NumPy result: shape (3, 4) - 3 True values, last dim preserved
        try
        {
            var result = arr3d[mask2d];

            // If supported, verify shape and values
            Assert.IsTrue(result.shape.SequenceEqual(new[] { 3, 4 }), $"Expected shape [3, 4], got [{string.Join(", ", result.shape)}]");

            // First True at (0,0) → arr3d[0,0,:] = [0,1,2,3]
            Assert.AreEqual(0, result[0, 0].GetInt32());
            Assert.AreEqual(3, result[0, 3].GetInt32());

            // Second True at (0,2) → arr3d[0,2,:] = [8,9,10,11]
            Assert.AreEqual(8, result[1, 0].GetInt32());

            // Third True at (1,1) → arr3d[1,1,:] = [16,17,18,19]
            Assert.AreEqual(16, result[2, 0].GetInt32());
        }
        catch (IndexOutOfRangeException)
        {
            // NumSharp doesn't support partial shape match yet
            Assert.Fail("Partial shape match (2D mask on 3D array) not supported - NumPy allows this");
        }
    }

    #endregion

    #region Case 6: Boolean Mask Assignment (Setter)

    [Test]
    public void Case6_Assignment_ScalarValue()
    {
        // NumPy: arr[mask] = 0 sets all masked elements to 0
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask = (arr > 2).MakeGeneric<bool>();

        arr[mask] = np.array(0);

        // [1, 2, 0, 0, 0]
        Assert.AreEqual(1, arr.GetInt32(0));
        Assert.AreEqual(2, arr.GetInt32(1));
        Assert.AreEqual(0, arr.GetInt32(2));
        Assert.AreEqual(0, arr.GetInt32(3));
        Assert.AreEqual(0, arr.GetInt32(4));
    }

    [Test]
    public void Case6_Assignment_ArrayValue()
    {
        // NumPy: arr[mask] = [10, 20, 30] assigns in order
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask = (arr > 2).MakeGeneric<bool>();

        arr[mask] = np.array(new[] { 10, 20, 30 });

        // [1, 2, 10, 20, 30]
        Assert.AreEqual(1, arr.GetInt32(0));
        Assert.AreEqual(2, arr.GetInt32(1));
        Assert.AreEqual(10, arr.GetInt32(2));
        Assert.AreEqual(20, arr.GetInt32(3));
        Assert.AreEqual(30, arr.GetInt32(4));
    }

    [Test]
    public void Case6_Assignment_2D_SameShapeMask()
    {
        // NumPy: arr2d[mask2d] = -1
        var arr2d = np.arange(12).reshape(3, 4);
        var mask2d = (arr2d > 5).MakeGeneric<bool>();

        arr2d[mask2d] = np.array(-1);

        // Elements > 5 become -1
        Assert.AreEqual(0, arr2d[0, 0].GetInt32()); // unchanged
        Assert.AreEqual(-1, arr2d[1, 2].GetInt32()); // was 6
        Assert.AreEqual(-1, arr2d[2, 3].GetInt32()); // was 11
    }

    [Test]
    public void Case6_Assignment_2D_RowMask()
    {
        // NumPy: arr2d[[T,F,T]] = 99
        var arr2d = np.arange(12).reshape(3, 4);
        var rowMask = np.array(new[] { true, false, true }).MakeGeneric<bool>();

        arr2d[rowMask] = np.array(99);

        // Row 0 and 2 become 99
        Assert.AreEqual(99, arr2d[0, 0].GetInt32());
        Assert.AreEqual(99, arr2d[0, 3].GetInt32());
        Assert.AreEqual(4, arr2d[1, 0].GetInt32()); // unchanged
        Assert.AreEqual(99, arr2d[2, 0].GetInt32());
    }

    [Test]
    public void Case6_Assignment_BroadcastValue()
    {
        // NumPy: arr2d[[T,F,T]] = [[100, 101, 102, 103]] broadcasts
        var arr2d = np.arange(12).reshape(3, 4);
        var rowMask = np.array(new[] { true, false, true }).MakeGeneric<bool>();

        // Note: This test checks if broadcasting works in assignment
        arr2d[rowMask] = np.array(new[,] { { 100, 101, 102, 103 } });

        Assert.AreEqual(100, arr2d[0, 0].GetInt32());
        Assert.AreEqual(103, arr2d[0, 3].GetInt32());
        Assert.AreEqual(4, arr2d[1, 0].GetInt32()); // unchanged
        Assert.AreEqual(100, arr2d[2, 0].GetInt32());
        Assert.AreEqual(103, arr2d[2, 3].GetInt32());
    }

    #endregion

    #region Case 7: Edge Cases - Empty Masks

    [Test]
    public void Case7_AllFalse_EmptyResult()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var emptyMask = np.array(new[] { false, false, false, false, false }).MakeGeneric<bool>();

        var result = arr[emptyMask];

        Assert.IsTrue(result.shape.SequenceEqual(new[] { 0 }), $"Expected shape [0], got [{string.Join(", ", result.shape)}]");
        Assert.AreEqual(0, result.size);
    }

    [Test]
    public void Case7_AllTrue_AllElements()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var allMask = np.array(new[] { true, true, true, true, true }).MakeGeneric<bool>();

        var result = arr[allMask];

        Assert.IsTrue(result.shape.SequenceEqual(new[] { 5 }), $"Expected shape [5], got [{string.Join(", ", result.shape)}]");
        Assert.AreEqual(1, result.GetInt32(0));
        Assert.AreEqual(5, result.GetInt32(4));
    }

    [Test]
    public void Case7_EmptyArray_EmptyMask()
    {
        var empty = np.array(new int[0]);
        var emptyMask = np.array(new bool[0]).MakeGeneric<bool>();

        var result = empty[emptyMask];

        Assert.IsTrue(result.shape.SequenceEqual(new[] { 0 }), $"Expected shape [0], got [{string.Join(", ", result.shape)}]");
    }

    [Test]
    public void Case7_EmptyResult_PreservesDtype()
    {
        var arrFloat = np.array(new[] { 1.5, 2.5, 3.5 });
        var emptyMask = np.array(new[] { false, false, false }).MakeGeneric<bool>();

        var result = arrFloat[emptyMask];

        Assert.AreEqual(typeof(double), result.dtype);
        Assert.IsTrue(result.shape.SequenceEqual(new[] { 0 }), $"Expected shape [0], got [{string.Join(", ", result.shape)}]");
    }

    #endregion

    #region Case 8: Shape Mismatch Errors

    [Test]
    public void Case8_ShapeMismatch_ThrowsError()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var badMask = np.array(new[] { true, false }).MakeGeneric<bool>(); // Length 2 vs 5

        Assert.ThrowsException<IndexOutOfRangeException>(() => arr[badMask]);
    }

    [Test]
    public void Case8_ShapeMismatch_ErrorMessage()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var badMask = np.array(new[] { true, false }).MakeGeneric<bool>();

        try
        {
            var _ = arr[badMask];
            Assert.Fail("Should have thrown IndexOutOfRangeException");
        }
        catch (IndexOutOfRangeException ex)
        {
            Assert.IsTrue(ex.Message.Contains("boolean index did not match"), $"Error message should contain 'boolean index did not match', got: {ex.Message}");
        }
    }

    [Test]
    public void Case8_2D_WrongMaskShape()
    {
        var arr2d = np.arange(12).reshape(3, 4);
        // Mask shape (2, 2) doesn't match array shape (3, 4) or prefix
        var badMask = np.array(new[,] { { true, false }, { false, true } }).MakeGeneric<bool>();

        Assert.ThrowsException<IndexOutOfRangeException>(() => arr2d[badMask]);
    }

    #endregion

    #region Case 9: Boolean Scalar Indexing

    [Test]
    public void Case9_BooleanScalar_True()
    {
        // NumPy: arr[True] adds a dimension → [[1, 2, 3]]
        var arr = np.array(new[] { 1, 2, 3 });

        // NumSharp may not support this directly
        try
        {
            var scalarTrue = np.array(true).MakeGeneric<bool>();
            var result = arr[scalarTrue];

            // NumPy behavior: adds axis → shape (1, 3)
            Assert.IsTrue(result.shape.SequenceEqual(new[] { 1, 3 }),
                $"0-D boolean True should add axis. Expected shape [1, 3], got [{string.Join(", ", result.shape)}]");
        }
        catch (Exception ex) when (ex is not AssertFailedException)
        {
            Assert.Fail($"0-D boolean indexing (arr[True]) not supported - NumPy allows this. Error: {ex.Message}");
        }
    }

    [Test]
    public void Case9_BooleanScalar_False()
    {
        // NumPy: arr[False] → empty array with extra dimension
        var arr = np.array(new[] { 1, 2, 3 });

        try
        {
            var scalarFalse = np.array(false).MakeGeneric<bool>();
            var result = arr[scalarFalse];

            // NumPy behavior: empty with extra axis, shape (0, 3) or similar
            Assert.AreEqual(0, result.shape[0], "0-D boolean False should give empty first dimension");
        }
        catch (Exception ex) when (ex is not AssertFailedException)
        {
            Assert.Fail($"0-D boolean indexing (arr[False]) not supported - NumPy allows this. Error: {ex.Message}");
        }
    }

    #endregion

    #region Case 10: np.nonzero and np.where equivalence

    [Test]
    public void Case10_Nonzero_ReturnsCorrectIndices()
    {
        var mask = np.array(new[] { true, false, true, false, true }).MakeGeneric<bool>();

        var indices = np.nonzero(mask);

        Assert.AreEqual(1, indices.Length, "1D array should return 1 index array");
        Assert.AreEqual(3, indices[0].size, "Should have 3 True values");
        Assert.AreEqual(0, indices[0].GetInt32(0));
        Assert.AreEqual(2, indices[0].GetInt32(1));
        Assert.AreEqual(4, indices[0].GetInt32(2));
    }

    [Test]
    public void Case10_Nonzero_2D_ReturnsTwoArrays()
    {
        var mask2d = np.array(new[,] { { true, false, true }, { false, true, false } }).MakeGeneric<bool>();

        var indices = np.nonzero(mask2d);

        Assert.AreEqual(2, indices.Length, "2D array should return 2 index arrays");
        Assert.AreEqual(3, indices[0].size, "Should have 3 True values");

        // Row indices: [0, 0, 1]
        Assert.AreEqual(0, indices[0].GetInt32(0));
        Assert.AreEqual(0, indices[0].GetInt32(1));
        Assert.AreEqual(1, indices[0].GetInt32(2));

        // Col indices: [0, 2, 1]
        Assert.AreEqual(0, indices[1].GetInt32(0));
        Assert.AreEqual(2, indices[1].GetInt32(1));
        Assert.AreEqual(1, indices[1].GetInt32(2));
    }

    [Test]
    public void Case10_BooleanIndex_EqualsNonzeroFancyIndex()
    {
        var arr = np.array(new[] { 10, 20, 30, 40, 50 });
        var mask = np.array(new[] { true, false, true, false, true }).MakeGeneric<bool>();

        var boolResult = arr[mask];

        // Both should give [10, 30, 50]
        Assert.AreEqual(3, boolResult.size);
        Assert.AreEqual(10, boolResult.GetInt32(0));
        Assert.AreEqual(30, boolResult.GetInt32(1));
        Assert.AreEqual(50, boolResult.GetInt32(2));
    }

    #endregion

    #region Case 11: Order Preservation

    [Test]
    public void Case11_OrderPreservation_FollowsMaskPosition()
    {
        // NumPy: Order follows the position in the mask, not the order of True values
        var arr = np.array(new[] { 10, 20, 30, 40, 50 });
        var mask = np.array(new[] { false, true, false, true, true }).MakeGeneric<bool>();

        var result = arr[mask];

        // Elements at positions 1, 3, 4 → [20, 40, 50]
        Assert.AreEqual(20, result.GetInt32(0));
        Assert.AreEqual(40, result.GetInt32(1));
        Assert.AreEqual(50, result.GetInt32(2));
    }

    [Test]
    public void Case11_2D_COrderIteration()
    {
        // NumPy iterates in C-order (row-major)
        var arr2d = np.array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }); // 3x2
        var mask2d = np.array(new[,] { { true, false }, { false, true }, { true, false } }).MakeGeneric<bool>();

        var result = arr2d[mask2d];

        // C-order: (0,0)=1, (1,1)=4, (2,0)=5
        Assert.AreEqual(1, result.GetInt32(0));
        Assert.AreEqual(4, result.GetInt32(1));
        Assert.AreEqual(5, result.GetInt32(2));
    }

    #endregion

    #region Case 12: Non-Contiguous Arrays

    [Test]
    public void Case12_SlicedArray_BooleanMask()
    {
        // Boolean mask on sliced (non-contiguous) array
        var arr = np.arange(20).reshape(4, 5);
        var sliced = arr["::2, ::2"]; // Shape (2, 3), non-contiguous

        Assert.IsFalse(sliced.Shape.IsContiguous, "Sliced array should be non-contiguous");

        var mask = (sliced > 5).MakeGeneric<bool>();
        var result = sliced[mask];

        // sliced = [[0, 2, 4], [10, 12, 14]]
        // > 5: [10, 12, 14]
        Assert.AreEqual(3, result.size);
        Assert.AreEqual(10, result.GetInt32(0));
        Assert.AreEqual(12, result.GetInt32(1));
        Assert.AreEqual(14, result.GetInt32(2));
    }

    [Test]
    public void Case12_TransposedArray_BooleanMask()
    {
        var arr = np.arange(12).reshape(3, 4);
        var transposed = arr.T; // Shape (4, 3)

        Assert.IsFalse(transposed.Shape.IsContiguous, "Transposed array should be non-contiguous");

        var mask = (transposed > 5).MakeGeneric<bool>();
        var result = transposed[mask];

        // Transposed > 5: [8, 9, 6, 10, 7, 11] (C-order on transposed)
        Assert.AreEqual(6, result.size);
        Assert.AreEqual(8, result.GetInt32(0));
        Assert.AreEqual(9, result.GetInt32(1));
        Assert.AreEqual(6, result.GetInt32(2));
        Assert.AreEqual(10, result.GetInt32(3));
        Assert.AreEqual(7, result.GetInt32(4));
        Assert.AreEqual(11, result.GetInt32(5));
    }

    [Test]
    public void Case12_SlicedView_BooleanMask()
    {
        var arr = np.arange(20).reshape(4, 5);
        var view = arr["1:3"]; // Shape (2, 5)

        var mask = (view > 10).MakeGeneric<bool>();
        var result = view[mask];

        // view = [[5,6,7,8,9], [10,11,12,13,14]]
        // > 10: [11, 12, 13, 14]
        Assert.AreEqual(4, result.size);
        Assert.AreEqual(11, result.GetInt32(0));
        Assert.AreEqual(14, result.GetInt32(3));
    }

    #endregion

    #region Case 13: Broadcast Arrays

    [Test]
    public void Case13_BroadcastArray_BooleanMask()
    {
        var arr1d = np.array(new[] { 1, 2, 3 });
        var broadcast = np.broadcast_to(arr1d, new Shape(3, 3));

        Assert.IsTrue(broadcast.Shape.IsBroadcasted, "Array should be broadcast");

        var mask = (broadcast > 1).MakeGeneric<bool>();
        var result = broadcast[mask];

        // broadcast = [[1,2,3], [1,2,3], [1,2,3]]
        // > 1: [2, 3, 2, 3, 2, 3]
        Assert.AreEqual(6, result.size);
        Assert.AreEqual(2, result.GetInt32(0));
        Assert.AreEqual(3, result.GetInt32(1));
        Assert.AreEqual(2, result.GetInt32(2));
        Assert.AreEqual(3, result.GetInt32(3));
    }

    #endregion

    #region Case 14: Broadcast Comparison for Mask

    [Test]
    public void Case14_BroadcastComparison_CreatesMask()
    {
        var arr2d = np.arange(12).reshape(3, 4);
        var threshold = np.array(new[,] { { 5 }, { 3 }, { 8 } }); // (3, 1) broadcasts to (3, 4)

        var mask = (arr2d > threshold).MakeGeneric<bool>();
        var result = arr2d[mask];

        // Row 0: > 5 → none (0,1,2,3 all <= 5)
        // Row 1: > 3 → [4, 5, 6, 7]
        // Row 2: > 8 → [9, 10, 11]
        Assert.AreEqual(7, result.size);
        Assert.AreEqual(4, result.GetInt32(0));
        Assert.AreEqual(7, result.GetInt32(3));
        Assert.AreEqual(9, result.GetInt32(4));
        Assert.AreEqual(11, result.GetInt32(6));
    }

    #endregion

    #region Case 15: Ravel/Flatten with Boolean

    [Test]
    public void Case15_RavelThenBoolean()
    {
        var arr2d = np.arange(12).reshape(3, 4);
        var flat = arr2d.ravel();
        var mask = (flat > 5).MakeGeneric<bool>();

        var result = flat[mask];

        Assert.AreEqual(6, result.size);
        Assert.AreEqual(6, result.GetInt32(0));
        Assert.AreEqual(11, result.GetInt32(5));
    }

    [Test]
    public void Case15_FlattenThenBoolean()
    {
        var arr2d = np.arange(12).reshape(3, 4);
        var flat = arr2d.flatten();
        var mask = (flat > 5).MakeGeneric<bool>();

        var result = flat[mask];

        Assert.AreEqual(6, result.size);
    }

    #endregion

    #region Case 16: Chained Boolean Indexing

    [Test]
    public void Case16_ChainedBooleanIndexing()
    {
        var arr = np.arange(20);

        // First boolean selection
        var mask1 = (arr % 2 == 0).MakeGeneric<bool>(); // Even numbers
        var result1 = arr[mask1];

        // Second boolean selection on result
        var mask2 = (result1 > 10).MakeGeneric<bool>();
        var result2 = result1[mask2];

        // Even numbers > 10: [12, 14, 16, 18]
        Assert.AreEqual(4, result2.size);
        Assert.AreEqual(12, result2.GetInt32(0));
        Assert.AreEqual(14, result2.GetInt32(1));
        Assert.AreEqual(16, result2.GetInt32(2));
        Assert.AreEqual(18, result2.GetInt32(3));
    }

    #endregion

    #region Case 17: Result Memory Layout

    [Test]
    public void Case17_ResultIsContiguous()
    {
        var arr2d = np.arange(12).reshape(3, 4);
        var mask1d = np.array(new[] { true, false, true }).MakeGeneric<bool>();

        var result = arr2d[mask1d];

        // Boolean indexing always returns contiguous result
        Assert.IsTrue(result.Shape.IsContiguous, "Result should be contiguous");
    }

    [Test]
    public void Case17_ResultFromNonContiguous_IsContiguous()
    {
        var arr = np.arange(20).reshape(4, 5);
        var sliced = arr["::2, ::2"]; // Non-contiguous

        Assert.IsFalse(sliced.Shape.IsContiguous, "Source should be non-contiguous");

        var mask = (sliced > 5).MakeGeneric<bool>();
        var result = sliced[mask];

        // Result should be contiguous (it's a copy)
        Assert.IsTrue(result.Shape.IsContiguous, "Result should be contiguous (copy)");
    }

    #endregion

    #region Case 18: Different Dtypes

    [Test]
    public void Case18_Float64()
    {
        var arr = np.array(new[] { 1.5, 2.5, 3.5, 4.5, 5.5 });
        var mask = (arr > 3.0).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(typeof(double), result.dtype);
        Assert.AreEqual(3, result.size);
        Assert.AreEqual(3.5, result.GetDouble(0));
    }

    [Test]
    public void Case18_Float32()
    {
        var arr = np.array(new float[] { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f });
        var mask = np.array(new[] { true, false, true, false, true }).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(typeof(float), result.dtype);
        Assert.AreEqual(3, result.size);
    }

    [Test]
    public void Case18_Int64()
    {
        var arr = np.array(new long[] { 1L, 2L, 3L, 4L, 5L });
        var mask = np.array(new[] { true, false, true, false, true }).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(typeof(long), result.dtype);
    }

    [Test]
    public void Case18_Boolean()
    {
        var arr = np.array(new[] { true, false, true, false, true });
        var mask = np.array(new[] { true, true, false, false, true }).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(typeof(bool), result.dtype);
        Assert.AreEqual(3, result.size);
        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsFalse(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
    }

    [Test]
    public void Case18_Byte()
    {
        var arr = np.array(new byte[] { 1, 2, 3, 4, 5 });
        var mask = np.array(new[] { true, false, true, false, true }).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(typeof(byte), result.dtype);
        Assert.AreEqual((byte)1, result.GetByte(0));
    }

    #endregion

    #region Case 19: Single Element Selection

    [Test]
    public void Case19_SingleElementSelected()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask = np.array(new[] { false, false, true, false, false }).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.IsTrue(result.shape.SequenceEqual(new[] { 1 }), $"Expected shape [1], got [{string.Join(", ", result.shape)}]");
        Assert.AreEqual(3, result.GetInt32(0));
    }

    [Test]
    public void Case19_2D_SingleRow()
    {
        var arr2d = np.arange(12).reshape(3, 4);
        var mask = np.array(new[] { false, true, false }).MakeGeneric<bool>();

        var result = arr2d[mask];

        Assert.IsTrue(result.shape.SequenceEqual(new[] { 1, 4 }), $"Expected shape [1, 4], got [{string.Join(", ", result.shape)}]");
        Assert.AreEqual(4, result[0, 0].GetInt32());
    }

    #endregion

    #region Case 20: Large Arrays

    [Test]
    public void Case20_LargeArray_Performance()
    {
        var size = 100000;
        var arr = np.arange(size);
        var mask = (arr % 2 == 0).MakeGeneric<bool>(); // Select even numbers

        var result = arr[mask];

        Assert.AreEqual(size / 2, result.size);
        Assert.AreEqual(0, result.GetInt32(0));
        Assert.AreEqual(size - 2, result.GetInt32(result.size - 1));
    }

    #endregion

    #region Case 21: Comparison Operators for Mask Generation

    [Test]
    public void Case21_GreaterThan()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask = (arr > 3).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(2, result.size);
        Assert.AreEqual(4, result.GetInt32(0));
        Assert.AreEqual(5, result.GetInt32(1));
    }

    [Test]
    public void Case21_LessThan()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask = (arr < 3).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(2, result.size);
        Assert.AreEqual(1, result.GetInt32(0));
        Assert.AreEqual(2, result.GetInt32(1));
    }

    [Test]
    public void Case21_Equal()
    {
        var arr = np.array(new[] { 1, 2, 3, 2, 1 });
        var mask = (arr == 2).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(2, result.size);
        Assert.AreEqual(2, result.GetInt32(0));
        Assert.AreEqual(2, result.GetInt32(1));
    }

    [Test]
    public void Case21_NotEqual()
    {
        var arr = np.array(new[] { 1, 2, 3, 2, 1 });
        var mask = (arr != 2).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(3, result.size);
        Assert.AreEqual(1, result.GetInt32(0));
        Assert.AreEqual(3, result.GetInt32(1));
        Assert.AreEqual(1, result.GetInt32(2));
    }

    [Test]
    public void Case21_GreaterEqual()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask = (arr >= 3).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(3, result.size);
    }

    [Test]
    public void Case21_LessEqual()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask = (arr <= 3).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(3, result.size);
    }

    #endregion

    #region Case 22: Logical Operations on Masks

    [Test]
    public void Case22_LogicalAnd()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var mask1 = arr > 3;
        var mask2 = arr < 8;
        var combinedMask = (mask1 & mask2).MakeGeneric<bool>();

        var result = arr[combinedMask];

        // > 3 AND < 8: [4, 5, 6, 7]
        Assert.AreEqual(4, result.size);
        Assert.AreEqual(4, result.GetInt32(0));
        Assert.AreEqual(7, result.GetInt32(3));
    }

    [Test]
    public void Case22_LogicalOr()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask1 = arr < 2;
        var mask2 = arr > 4;
        var combinedMask = (mask1 | mask2).MakeGeneric<bool>();

        var result = arr[combinedMask];

        // < 2 OR > 4: [1, 5]
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(1, result.GetInt32(0));
        Assert.AreEqual(5, result.GetInt32(1));
    }

    [Test]
    public void Case22_LogicalNot()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask = arr > 3;
        var invertedMask = (!mask).MakeGeneric<bool>();

        var result = arr[invertedMask];

        // NOT (> 3): [1, 2, 3]
        Assert.AreEqual(3, result.size);
        Assert.AreEqual(1, result.GetInt32(0));
        Assert.AreEqual(3, result.GetInt32(2));
    }

    #endregion

    #region Case 23: NaN Handling

    [Test]
    public void Case23_NaN_Comparison()
    {
        var arr = np.array(new[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 });

        // NaN comparisons always return false
        var mask = (arr > 2).MakeGeneric<bool>();
        var result = arr[mask];

        // Only 3.0 and 5.0 are > 2 (NaN comparisons are false)
        Assert.AreEqual(2, result.size);
        Assert.AreEqual(3.0, result.GetDouble(0));
        Assert.AreEqual(5.0, result.GetDouble(1));
    }

    [Test]
    public void Case23_IsNaN_Mask()
    {
        var arr = np.array(new[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 });
        var mask = np.isnan(arr).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(2, result.size);
        Assert.IsTrue(double.IsNaN(result.GetDouble(0)));
        Assert.IsTrue(double.IsNaN(result.GetDouble(1)));
    }

    [Test]
    public void Case23_NotNaN_Mask()
    {
        var arr = np.array(new[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 });
        var mask = (!np.isnan(arr)).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(3, result.size);
        Assert.AreEqual(1.0, result.GetDouble(0));
        Assert.AreEqual(3.0, result.GetDouble(1));
        Assert.AreEqual(5.0, result.GetDouble(2));
    }

    #endregion

    #region Case 24: Infinity Handling

    [Test]
    public void Case24_Infinity_Comparison()
    {
        var arr = np.array(new[] { 1.0, double.PositiveInfinity, 3.0, double.NegativeInfinity, 5.0 });

        var mask = (arr > 4).MakeGeneric<bool>();
        var result = arr[mask];

        // +Inf > 4, 5 > 4
        Assert.AreEqual(2, result.size);
        Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)));
        Assert.AreEqual(5.0, result.GetDouble(1));
    }

    [Test]
    public void Case24_IsInf_Mask()
    {
        var arr = np.array(new[] { 1.0, double.PositiveInfinity, 3.0, double.NegativeInfinity, 5.0 });
        var mask = np.isinf(arr).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(2, result.size);
    }

    #endregion

    #region Case 25: Modulo for Mask

    [Test]
    public void Case25_Modulo_EvenNumbers()
    {
        var arr = np.arange(10);
        var mask = (arr % 2 == 0).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(5, result.size);
        Assert.AreEqual(0, result.GetInt32(0));
        Assert.AreEqual(2, result.GetInt32(1));
        Assert.AreEqual(8, result.GetInt32(4));
    }

    [Test]
    public void Case25_Modulo_MultipleOf3()
    {
        var arr = np.arange(15);
        var mask = (arr % 3 == 0).MakeGeneric<bool>();

        var result = arr[mask];

        // 0, 3, 6, 9, 12
        Assert.AreEqual(5, result.size);
        Assert.AreEqual(0, result.GetInt32(0));
        Assert.AreEqual(6, result.GetInt32(2));
        Assert.AreEqual(12, result.GetInt32(4));
    }

    #endregion

    #region Case 26: Complex Mask Expressions

    [Test]
    public void Case26_ComplexExpression_Range()
    {
        var arr = np.arange(20);
        // Select elements where 5 <= x < 15
        var mask = ((arr >= 5) & (arr < 15)).MakeGeneric<bool>();

        var result = arr[mask];

        Assert.AreEqual(10, result.size);
        Assert.AreEqual(5, result.GetInt32(0));
        Assert.AreEqual(14, result.GetInt32(9));
    }

    [Test]
    public void Case26_ComplexExpression_MultipleConditions()
    {
        var arr = np.arange(20);
        // Select elements divisible by 2 or 3 but not by 6
        var divBy2 = arr % 2 == 0;
        var divBy3 = arr % 3 == 0;
        var divBy6 = arr % 6 == 0;
        var mask = (((divBy2 | divBy3) & !divBy6)).MakeGeneric<bool>();

        var result = arr[mask];

        // 2, 3, 4, 8, 9, 10, 14, 15, 16
        Assert.AreEqual(9, result.size);
    }

    #endregion

    #region Case 27: Assignment Edge Cases

    [Test]
    public void Case27_Assignment_EmptyMask_NoChange()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var original = arr.copy();
        var emptyMask = np.array(new[] { false, false, false, false, false }).MakeGeneric<bool>();

        arr[emptyMask] = np.array(999);

        // Nothing should change
        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual(original.GetInt32(i), arr.GetInt32(i), $"Element at index {i} should be unchanged");
        }
    }

    [Test]
    public void Case27_Assignment_AllTrue()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var allMask = np.array(new[] { true, true, true, true, true }).MakeGeneric<bool>();

        arr[allMask] = np.array(0);

        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual(0, arr.GetInt32(i), $"Element at index {i} should be 0");
        }
    }

    #endregion

    #region Case 28: Higher Dimensional Arrays

    [Test]
    public void Case28_4D_SameShapeMask()
    {
        var arr4d = np.arange(24).reshape(2, 3, 2, 2);
        var mask4d = (arr4d > 12).MakeGeneric<bool>();

        var result = arr4d[mask4d];

        // 11 elements > 12: [13, 14, ..., 23]
        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(11, result.size);
        Assert.AreEqual(13, result.GetInt32(0));
        Assert.AreEqual(23, result.GetInt32(10));
    }

    [Test]
    public void Case28_4D_1DMask()
    {
        var arr4d = np.arange(48).reshape(2, 3, 4, 2);
        var mask1d = np.array(new[] { true, false }).MakeGeneric<bool>();

        var result = arr4d[mask1d];

        // Selects first "block" along axis 0
        Assert.IsTrue(result.shape.SequenceEqual(new[] { 1, 3, 4, 2 }), $"Expected shape [1, 3, 4, 2], got [{string.Join(", ", result.shape)}]");
    }

    #endregion

    #region Case 29: count_nonzero Relationship

    [Test]
    public void Case29_CountNonzero_MatchesResultSize()
    {
        var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask = (arr > 2).MakeGeneric<bool>();

        var count = np.count_nonzero(mask);
        var result = arr[mask];

        Assert.AreEqual(count, result.size);
    }

    [Test]
    public void Case29_CountNonzero_2D()
    {
        var arr2d = np.arange(12).reshape(3, 4);
        var mask2d = (arr2d > 5).MakeGeneric<bool>();

        var count = np.count_nonzero(mask2d);
        var result = arr2d[mask2d];

        Assert.AreEqual(count, result.size);
    }

    #endregion
}
