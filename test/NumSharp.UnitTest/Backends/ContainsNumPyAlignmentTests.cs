using System;
using System.Threading.Tasks;
using NumSharp;
using TUnit.Core;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;

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
public class ContainsNumPyAlignmentTests
{
    #region Gap #1: Broadcasting Errors Should Propagate

    [Test]
    public async Task Contains_1DArrayIn1D_ShapeMismatch_Throws()
    {
        // NumPy: [1,2] in np.array([1,2,3]) throws ValueError
        var arr = np.array(new[] { 1, 2, 3 });

        await Assert.That(() => arr.Contains(new[] { 1, 2 }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_NDArrayIn1D_ShapeMismatch_Throws()
    {
        // NumPy: np.array([1,2]) in np.array([1,2,3]) throws ValueError
        var arr = np.array(new[] { 1, 2, 3 });
        var search = np.array(new[] { 1, 2 });

        await Assert.That(() => arr.Contains(search))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_LargerArrayIn1D_ShapeMismatch_Throws()
    {
        // NumPy: [1,2,3,4,5] in np.array([1,2,3]) throws ValueError
        var arr = np.array(new[] { 1, 2, 3 });

        await Assert.That(() => arr.Contains(new[] { 1, 2, 3, 4, 5 }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_2DArrayIn1D_ShapeMismatch_Throws()
    {
        // NumPy: [[1,2],[3,4]] in np.array([1,2,3]) throws ValueError
        var arr = np.array(new[] { 1, 2, 3 });
        var search = np.array(new[,] { { 1, 2 }, { 3, 4 } });

        await Assert.That(() => arr.Contains(search))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_EmptyArrayIn1D_ShapeMismatch_Throws()
    {
        // Empty array has shape (0,) which can't broadcast with (3,)
        var arr = np.array(new[] { 1, 2, 3 });
        var empty = np.array(new int[0]);

        await Assert.That(() => arr.Contains(empty))
            .Throws<IncorrectShapeException>();
    }

    #endregion

    #region Broadcastable Cases Should Work

    [Test]
    public async Task Contains_ScalarIn1D_Works()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        await Assert.That(arr.Contains(2)).IsTrue();
        await Assert.That(arr.Contains(10)).IsFalse();
    }

    [Test]
    public async Task Contains_ScalarIn2D_SearchesAll()
    {
        // NumPy: 3 in np.array([[1,2],[3,4]]) returns True
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });

        await Assert.That(arr.Contains(3)).IsTrue();
        await Assert.That(arr.Contains(10)).IsFalse();
    }

    [Test]
    public async Task Contains_RowIn2D_Broadcasts()
    {
        // NumPy: [1,2] in np.array([[1,2],[3,4]]) returns True
        // Broadcasting: [1,2] compares element-wise, then any() checks if ANY match
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });

        await Assert.That(arr.Contains(new[] { 1, 2 })).IsTrue();  // All match first row
        await Assert.That(arr.Contains(new[] { 3, 4 })).IsTrue();  // All match second row
        await Assert.That(arr.Contains(new[] { 1, 3 })).IsTrue();  // 1 matches [0,0], so any()=True
        await Assert.That(arr.Contains(new[] { 5, 6 })).IsFalse(); // No element matches
    }

    [Test]
    public async Task Contains_NDArrayRowIn2D_Broadcasts()
    {
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var row = np.array(new[] { 1, 2 });

        await Assert.That(arr.Contains(row)).IsTrue();
    }

    [Test]
    public async Task Contains_ColumnIn2D_Broadcasts()
    {
        // [[1],[3]] broadcasts against [[1,2],[3,4]] -> checks column-wise
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var col = np.array(new[,] { { 1 }, { 3 } });

        await Assert.That(arr.Contains(col)).IsTrue();
    }

    [Test]
    public async Task Contains_ScalarIn3D_SearchesAll()
    {
        var arr = np.arange(24).reshape(2, 3, 4);

        await Assert.That(arr.Contains(0)).IsTrue();
        await Assert.That(arr.Contains(23)).IsTrue();
        await Assert.That(arr.Contains(100)).IsFalse();
    }

    [Test]
    public async Task Contains_0DScalar_Works()
    {
        var scalar = NDArray.Scalar(42);

        await Assert.That(scalar.Contains(42)).IsTrue();
        await Assert.That(scalar.Contains(0)).IsFalse();
    }

    #endregion

    #region Type Mismatch Returns False (Not Exception)

    [Test]
    public async Task Contains_StringInIntArray_ReturnsFalse()
    {
        // NumPy: "hello" in np.array([1,2,3]) returns False
        var arr = np.array(new[] { 1, 2, 3 });

        await Assert.That(arr.Contains("hello")).IsFalse();
    }

    [Test]
    public async Task Contains_StringInFloatArray_ReturnsFalse()
    {
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });

        await Assert.That(arr.Contains("test")).IsFalse();
    }

    [Test]
    public async Task Contains_StringInBoolArray_ReturnsFalse()
    {
        var arr = np.array(new[] { true, false });

        await Assert.That(arr.Contains("true")).IsFalse();
    }

    [Test]
    public async Task Contains_NullInAnyArray_ReturnsFalse()
    {
        // NumPy: None in np.array([1,2,3]) returns False
        var arr = np.array(new[] { 1, 2, 3 });

        await Assert.That(arr.Contains(null)).IsFalse();
    }

    #endregion

    #region Char Array Special Cases

    [Test]
    public async Task Contains_CharInCharArray_Works()
    {
        var arr = np.array(new[] { 'a', 'b', 'c' });

        await Assert.That(arr.Contains('a')).IsTrue();
        await Assert.That(arr.Contains('d')).IsFalse();
    }

    [Test]
    public async Task Contains_MatchingStringInCharArray_Broadcasts()
    {
        // "abc" creates char[3], broadcasts element-wise with char[3]
        var arr = np.array(new[] { 'a', 'b', 'c' });

        await Assert.That(arr.Contains("abc")).IsTrue();
    }

    [Test]
    public async Task Contains_MismatchedStringInCharArray_Throws()
    {
        // "hello" creates char[5], can't broadcast with char[3]
        var arr = np.array(new[] { 'a', 'b', 'c' });

        await Assert.That(() => arr.Contains("hello"))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_SingleCharStringInCharArray_Broadcasts()
    {
        // "a" creates char[1], broadcasts against char[3]
        var arr = np.array(new[] { 'a', 'b', 'c' });

        // This broadcasts [a] against [a,b,c] -> [True, False, False].any() = True
        await Assert.That(arr.Contains("a")).IsTrue();
    }

    #endregion

    #region All 12 Dtypes - Shape Mismatch Throws

    [Test]
    public async Task Contains_ShapeMismatch_Boolean_Throws()
    {
        var arr = np.array(new[] { true, false, true });
        await Assert.That(() => arr.Contains(new[] { true, false }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_ShapeMismatch_Byte_Throws()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        await Assert.That(() => arr.Contains(new byte[] { 1, 2 }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_ShapeMismatch_Int16_Throws()
    {
        var arr = np.array(new short[] { 1, 2, 3 });
        await Assert.That(() => arr.Contains(new short[] { 1, 2 }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_ShapeMismatch_UInt16_Throws()
    {
        var arr = np.array(new ushort[] { 1, 2, 3 });
        await Assert.That(() => arr.Contains(new ushort[] { 1, 2 }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_ShapeMismatch_Int32_Throws()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        await Assert.That(() => arr.Contains(new[] { 1, 2 }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_ShapeMismatch_UInt32_Throws()
    {
        var arr = np.array(new uint[] { 1, 2, 3 });
        await Assert.That(() => arr.Contains(new uint[] { 1, 2 }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_ShapeMismatch_Int64_Throws()
    {
        var arr = np.array(new long[] { 1, 2, 3 });
        await Assert.That(() => arr.Contains(new long[] { 1, 2 }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_ShapeMismatch_UInt64_Throws()
    {
        var arr = np.array(new ulong[] { 1, 2, 3 });
        await Assert.That(() => arr.Contains(new ulong[] { 1, 2 }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_ShapeMismatch_Single_Throws()
    {
        var arr = np.array(new[] { 1f, 2f, 3f });
        await Assert.That(() => arr.Contains(new[] { 1f, 2f }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_ShapeMismatch_Double_Throws()
    {
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });
        await Assert.That(() => arr.Contains(new[] { 1.0, 2.0 }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_ShapeMismatch_Decimal_Throws()
    {
        var arr = np.array(new[] { 1m, 2m, 3m });
        await Assert.That(() => arr.Contains(new[] { 1m, 2m }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_ShapeMismatch_Char_Throws()
    {
        var arr = np.array(new[] { 'a', 'b', 'c' });
        await Assert.That(() => arr.Contains(new[] { 'a', 'b' }))
            .Throws<IncorrectShapeException>();
    }

    #endregion

    #region Edge Cases

    [Test]
    public async Task Contains_EmptyArray_ReturnsFalse()
    {
        var empty = np.array(new int[0]);

        await Assert.That(empty.Contains(1)).IsFalse();
    }

    [Test]
    public async Task Contains_NaN_InFloatArray_ReturnsFalse()
    {
        // NaN == NaN is false in IEEE 754
        var arr = np.array(new[] { 1.0, double.NaN, 3.0 });

        await Assert.That(arr.Contains(double.NaN)).IsFalse();
    }

    [Test]
    public async Task Contains_Infinity_Works()
    {
        var arr = np.array(new[] { 1.0, double.PositiveInfinity, 3.0 });

        await Assert.That(arr.Contains(double.PositiveInfinity)).IsTrue();
        await Assert.That(arr.Contains(double.NegativeInfinity)).IsFalse();
    }

    [Test]
    public async Task Contains_TypePromotion_IntInFloat_Works()
    {
        // 2 (int) should match 2.0 (double) after promotion
        var arr = np.array(new[] { 1.0, 2.0, 3.0 });

        await Assert.That(arr.Contains(2)).IsTrue();
    }

    [Test]
    public async Task Contains_TypePromotion_FloatInInt_Works()
    {
        // 2.0 should match 2 (int) after promotion
        var arr = np.array(new[] { 1, 2, 3 });

        await Assert.That(arr.Contains(2.0)).IsTrue();
    }

    [Test]
    public async Task Contains_BoolIntInterop_Works()
    {
        // NumPy: 1 in np.array([True, False]) returns True (1 == True)
        var arr = np.array(new[] { true, false });

        await Assert.That(arr.Contains(1)).IsTrue();
        await Assert.That(arr.Contains(0)).IsTrue();
        await Assert.That(arr.Contains(2)).IsFalse();
    }

    [Test]
    public async Task Contains_SlicedArray_Works()
    {
        var arr = np.arange(10);
        var sliced = arr["2:8:2"]; // [2, 4, 6]

        await Assert.That(sliced.Contains(4)).IsTrue();
        await Assert.That(sliced.Contains(3)).IsFalse();
    }

    [Test]
    public async Task Contains_SlicedArray_ShapeMismatch_Throws()
    {
        var arr = np.arange(10);
        var sliced = arr["2:5"]; // [2, 3, 4] - shape (3,)

        await Assert.That(() => sliced.Contains(new[] { 1, 2 }))
            .Throws<IncorrectShapeException>();
    }

    [Test]
    public async Task Contains_TransposedArray_Works()
    {
        var arr = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var transposed = arr.T;

        await Assert.That(transposed.Contains(3)).IsTrue();
    }

    [Test]
    public async Task Contains_BroadcastView_Works()
    {
        var arr = np.array(new[] { 1, 2, 3 });
        var broadcast = np.broadcast_to(arr, new Shape(4, 3));

        await Assert.That(broadcast.Contains(2)).IsTrue();
        await Assert.That(broadcast.Contains(10)).IsFalse();
    }

    [Test]
    public async Task Contains_ReversedArray_Works()
    {
        var arr = np.arange(5)["::-1"]; // [4, 3, 2, 1, 0]

        await Assert.That(arr.Contains(3)).IsTrue();
        await Assert.That(arr.Contains(10)).IsFalse();
    }

    [Test]
    public async Task Contains_LargeArray_Works()
    {
        var arr = np.arange(1_000_000);

        await Assert.That(arr.Contains(500_000)).IsTrue();
        await Assert.That(arr.Contains(2_000_000)).IsFalse();
    }

    #endregion

    #region N-Dimensional Broadcasting

    [Test]
    public async Task Contains_3D_ScalarSearch_Works()
    {
        var arr = np.arange(24).reshape(2, 3, 4);

        await Assert.That(arr.Contains(15)).IsTrue();
        await Assert.That(arr.Contains(100)).IsFalse();
    }

    [Test]
    public async Task Contains_3D_1DSearch_Broadcasts()
    {
        // Search for [0,1,2,3] in (2,3,4) array
        // Should broadcast and find matches in first row of each 2D slice
        var arr = np.arange(24).reshape(2, 3, 4);
        var search = np.array(new[] { 0, 1, 2, 3 });

        await Assert.That(arr.Contains(search)).IsTrue();
    }

    [Test]
    public async Task Contains_3D_2DSearch_Broadcasts()
    {
        // Search for 2D slice in 3D array
        var arr = np.arange(24).reshape(2, 3, 4);
        var search = np.arange(12).reshape(3, 4); // First 2D slice

        await Assert.That(arr.Contains(search)).IsTrue();
    }

    [Test]
    public async Task Contains_3D_IncompatibleShape_Throws()
    {
        var arr = np.arange(24).reshape(2, 3, 4);
        var search = np.array(new[] { 1, 2 }); // Can't broadcast (2,) with (2,3,4)

        await Assert.That(() => arr.Contains(search))
            .Throws<IncorrectShapeException>();
    }

    #endregion

    #region __contains__ Method Alias

    [Test]
    public async Task DunderContains_SameAsContains()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        await Assert.That(arr.__contains__(2)).IsTrue();
        await Assert.That(arr.__contains__(10)).IsFalse();
    }

    [Test]
    public async Task DunderContains_ShapeMismatch_Throws()
    {
        var arr = np.array(new[] { 1, 2, 3 });

        await Assert.That(() => arr.__contains__(new[] { 1, 2 }))
            .Throws<IncorrectShapeException>();
    }

    #endregion
}
