using System;
using System.Threading.Tasks;
using NumSharp;

namespace NumSharp.UnitTest.Backends;

/// <summary>
/// Battle tests for Contains() NumPy alignment.
/// Tests that broadcasting errors propagate (Gap #1 fix).
///
/// NumPy behavior verified with Python 3.12, NumPy 2.x:
/// - Shape mismatch throws ValueError
/// - Type mismatch (string in int array) returns False
/// - Broadcastable shapes work correctly
/// </summary>
[TestClass]
public class ContainsNumPyAlignmentTests
{
    #region Gap #1: Broadcasting Errors Should Propagate

    [TestMethod]
    public async Task Contains_1DArrayIn1D_ShapeMismatch_Throws()
    {
        // NumPy: [1,2] in np.array([1,2,3]) throws ValueError
        var arr = np.array(new[] { 1, 2, 3 });

        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new[] { 1, 2 }));
    }

    [TestMethod]
    public async Task Contains_NDArrayIn1D_ShapeMismatch_Throws()
    {
        // NumPy: np.array([1,2]) in np.array([1,2,3]) throws ValueError
        var arr = np.array(new[] { 1, 2, 3 });
        var search = np.array(new[] { 1, 2 });

        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(search));
    }

    [TestMethod]
    public async Task Contains_LargerArrayIn1D_ShapeMismatch_Throws()
    {
        // NumPy: [1,2,3,4,5] in np.array([1,2,3]) throws ValueError
        var arr = np.array(new[] { 1, 2, 3 });

        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new[] { 1, 2, 3, 4, 5 }));
    }

    [TestMethod]
    public async Task Contains_2DArrayIn1D_ShapeMismatch_Throws()
    {
        // NumPy: [[1,2],[3,4]] in np.array([1,2,3]) throws ValueError
        var arr = np.array(new[] { 1, 2, 3 });
        var search = np.array(new[,] { { 1, 2 }, { 3, 4 } });

        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(search));
    }

    [TestMethod]
    public async Task Contains_EmptyArrayIn1D_ShapeMismatch_Throws()
    {
        // Empty array has shape (0,) which can't broadcast with (3,)
        var arr = np.array(new[] { 1, 2, 3 });
        var empty = np.array(new int[0]);

        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(empty));
    }

    #endregion

    #region Broadcastable Cases Should Work

    [TestMethod]
    public async Task Contains_ScalarIn1D_Works()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        arr.Contains(2).Should().BeTrue();
        arr.Contains(10).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_ScalarIn2D_SearchesAll()
    {
        // NumPy: 3 in np.array([[1,2],[3,4]]) returns True
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });

        arr.Contains(3).Should().BeTrue();
        arr.Contains(10).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_RowIn2D_Broadcasts()
    {
        // NumPy: [1,2] in np.array([[1,2],[3,4]]) returns True
        // Broadcasting: [1,2] compares element-wise, then any() checks if ANY match
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });

        arr.Contains(new[] { 1, 2 }).Should().BeTrue();  // All match first row
        arr.Contains(new[] { 3, 4 }).Should().BeTrue();  // All match second row
        arr.Contains(new[] { 1, 3 }).Should().BeTrue();  // 1 matches [0,0], so any()=True
        arr.Contains(new[] { 5, 6 }).Should().BeFalse(); // No element matches
    }

    [TestMethod]
    public async Task Contains_NDArrayRowIn2D_Broadcasts()
    {
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var row = np.array(new[] { 1, 2 });

        arr.Contains(row).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_ColumnIn2D_Broadcasts()
    {
        // [[1],[3]] broadcasts against [[1,2],[3,4]] -> checks column-wise
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var col = np.array(new[,] { { 1 }, { 3 } });

        arr.Contains(col).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_ScalarIn3D_SearchesAll()
    {
        var arr = np.arange(24).reshape(2, 3, 4);

        arr.Contains(0).Should().BeTrue();
        arr.Contains(23).Should().BeTrue();
        arr.Contains(100).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_0DScalar_Works()
    {
        var scalar = NDArray.Scalar(42);

        scalar.Contains(42).Should().BeTrue();
        scalar.Contains(0).Should().BeFalse();
    }

    #endregion

    #region Type Mismatch Returns False (Not Exception)

    [TestMethod]
    public async Task Contains_StringInIntArray_ReturnsFalse()
    {
        // NumPy: "hello" in np.array([1,2,3]) returns False
        var arr = np.array(new[] { 1, 2, 3 });

        arr.Contains("hello").Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_StringInFloatArray_ReturnsFalse()
    {
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });

        arr.Contains("test").Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_StringInBoolArray_ReturnsFalse()
    {
        var arr = np.array(new[] { true, false });

        arr.Contains("true").Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_NullInAnyArray_ReturnsFalse()
    {
        // NumPy: None in np.array([1,2,3]) returns False
        var arr = np.array(new[] { 1, 2, 3 });

        arr.Contains(null).Should().BeFalse();
    }

    #endregion

    #region Char Array Special Cases

    [TestMethod]
    public async Task Contains_CharInCharArray_Works()
    {
        var arr = np.array(new[] { 'a', 'b', 'c' });

        arr.Contains('a').Should().BeTrue();
        arr.Contains('d').Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_MatchingStringInCharArray_Broadcasts()
    {
        // "abc" creates char[3], broadcasts element-wise with char[3]
        var arr = np.array(new[] { 'a', 'b', 'c' });

        arr.Contains("abc").Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_MismatchedStringInCharArray_Throws()
    {
        // "hello" creates char[5], can't broadcast with char[3]
        var arr = np.array(new[] { 'a', 'b', 'c' });

        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains("hello"));
    }

    [TestMethod]
    public async Task Contains_SingleCharStringInCharArray_Broadcasts()
    {
        // "a" creates char[1], broadcasts against char[3]
        var arr = np.array(new[] { 'a', 'b', 'c' });

        // This broadcasts [a] against [a,b,c] -> [True, False, False].any() = True
        arr.Contains("a").Should().BeTrue();
    }

    #endregion

    #region All 12 Dtypes - Shape Mismatch Throws

    [TestMethod]
    public async Task Contains_ShapeMismatch_Boolean_Throws()
    {
        var arr = np.array(new[] { true, false, true });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new[] { true, false }));
    }

    [TestMethod]
    public async Task Contains_ShapeMismatch_Byte_Throws()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new byte[] { 1, 2 }));
    }

    [TestMethod]
    public async Task Contains_ShapeMismatch_Int16_Throws()
    {
        var arr = np.array(new short[] { 1, 2, 3 });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new short[] { 1, 2 }));
    }

    [TestMethod]
    public async Task Contains_ShapeMismatch_UInt16_Throws()
    {
        var arr = np.array(new ushort[] { 1, 2, 3 });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new ushort[] { 1, 2 }));
    }

    [TestMethod]
    public async Task Contains_ShapeMismatch_Int32_Throws()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new[] { 1, 2 }));
    }

    [TestMethod]
    public async Task Contains_ShapeMismatch_UInt32_Throws()
    {
        var arr = np.array(new uint[] { 1, 2, 3 });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new uint[] { 1, 2 }));
    }

    [TestMethod]
    public async Task Contains_ShapeMismatch_Int64_Throws()
    {
        var arr = np.array(new long[] { 1, 2, 3 });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new long[] { 1, 2 }));
    }

    [TestMethod]
    public async Task Contains_ShapeMismatch_UInt64_Throws()
    {
        var arr = np.array(new ulong[] { 1, 2, 3 });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new ulong[] { 1, 2 }));
    }

    [TestMethod]
    public async Task Contains_ShapeMismatch_Single_Throws()
    {
        var arr = np.array(new[] { 1f, 2f, 3f });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new[] { 1f, 2f }));
    }

    [TestMethod]
    public async Task Contains_ShapeMismatch_Double_Throws()
    {
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new[] { 1.0, 2.0 }));
    }

    [TestMethod]
    public async Task Contains_ShapeMismatch_Decimal_Throws()
    {
        var arr = np.array(new[] { 1m, 2m, 3m });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new[] { 1m, 2m }));
    }

    [TestMethod]
    public async Task Contains_ShapeMismatch_Char_Throws()
    {
        var arr = np.array(new[] { 'a', 'b', 'c' });
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(new[] { 'a', 'b' }));
    }

    #endregion

    #region Edge Cases

    [TestMethod]
    public async Task Contains_EmptyArray_ReturnsFalse()
    {
        var empty = np.array(new int[0]);

        empty.Contains(1).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_NaN_InFloatArray_ReturnsFalse()
    {
        // NaN == NaN is false in IEEE 754
        var arr = np.array(new[] { 1.0, double.NaN, 3.0 });

        arr.Contains(double.NaN).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_Infinity_Works()
    {
        var arr = np.array(new[] { 1.0, double.PositiveInfinity, 3.0 });

        arr.Contains(double.PositiveInfinity).Should().BeTrue();
        arr.Contains(double.NegativeInfinity).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_TypePromotion_IntInFloat_Works()
    {
        // 2 (int) should match 2.0 (double) after promotion
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });

        arr.Contains(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_TypePromotion_FloatInInt_Works()
    {
        // 2.0 should match 2 (int) after promotion
        var arr = np.array(new[] { 1, 2, 3 });

        arr.Contains(2.0).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_BoolIntInterop_Works()
    {
        // NumPy: 1 in np.array([True, False]) returns True (1 == True)
        var arr = np.array(new[] { true, false });

        arr.Contains(1).Should().BeTrue();
        arr.Contains(0).Should().BeTrue();
        arr.Contains(2).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_SlicedArray_Works()
    {
        var arr = np.arange(10);
        var sliced = arr["2:8:2"]; // [2, 4, 6]

        sliced.Contains(4).Should().BeTrue();
        sliced.Contains(3).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_SlicedArray_ShapeMismatch_Throws()
    {
        var arr = np.arange(10);
        var sliced = arr["2:5"]; // [2, 3, 4] - shape (3,)

        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => sliced.Contains(new[] { 1, 2 }));
    }

    [TestMethod]
    public async Task Contains_TransposedArray_Works()
    {
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var transposed = arr.T;

        transposed.Contains(3).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_BroadcastView_Works()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var broadcast = np.broadcast_to(arr, new Shape(4, 3));

        broadcast.Contains(2).Should().BeTrue();
        broadcast.Contains(10).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_ReversedArray_Works()
    {
        var arr = np.arange(5)["::-1"]; // [4, 3, 2, 1, 0]

        arr.Contains(3).Should().BeTrue();
        arr.Contains(10).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_LargeArray_Works()
    {
        var arr = np.arange(1_000_000);

        arr.Contains(500_000).Should().BeTrue();
        arr.Contains(2_000_000).Should().BeFalse();
    }

    #endregion

    #region N-Dimensional Broadcasting

    [TestMethod]
    public async Task Contains_3D_ScalarSearch_Works()
    {
        var arr = np.arange(24).reshape(2, 3, 4);

        arr.Contains(15).Should().BeTrue();
        arr.Contains(100).Should().BeFalse();
    }

    [TestMethod]
    public async Task Contains_3D_1DSearch_Broadcasts()
    {
        // Search for [0,1,2,3] in (2,3,4) array
        // Should broadcast and find matches in first row of each 2D slice
        var arr = np.arange(24).reshape(2, 3, 4);
        var search = np.array(new[] { 0, 1, 2, 3 });

        arr.Contains(search).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_3D_2DSearch_Broadcasts()
    {
        // Search for 2D slice in 3D array
        var arr = np.arange(24).reshape(2, 3, 4);
        var search = np.arange(12).reshape(3, 4); // First 2D slice

        arr.Contains(search).Should().BeTrue();
    }

    [TestMethod]
    public async Task Contains_3D_IncompatibleShape_Throws()
    {
        var arr = np.arange(24).reshape(2, 3, 4);
        var search = np.array(new[] { 1, 2 }); // Can't broadcast (2,) with (2,3,4)

        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.Contains(search));
    }

    #endregion

    #region __contains__ Method Alias

    [TestMethod]
    public async Task DunderContains_SameAsContains()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        arr.__contains__(2).Should().BeTrue();
        arr.__contains__(10).Should().BeFalse();
    }

    [TestMethod]
    public async Task DunderContains_ShapeMismatch_Throws()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<IncorrectShapeException>(() => arr.__contains__(new[] { 1, 2 }));
    }

    #endregion
}
