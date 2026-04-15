using System;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.Generic;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.NpApiOverloads;

/// <summary>
/// Tests for overloads that had the 'in' parameter modifier removed.
/// Covers: Comparison (18), Logical (4), Axis Manipulation (8), Unique (1),
/// Counting/Indexing (3), Linear Algebra (3), Min with dtype (3).
///
/// These tests verify that the overloads compile and work correctly after
/// removing the 'in' modifier from NDArray parameters.
/// </summary>
[TestClass]
public class NpApiOverloadTests_LogicManipulation
{
    #region Comparison Operations - np.equal (3 overloads)

    [TestMethod]
    public async Task Equal_TwoArrays_Compiles()
    {
        // NumPy: np.equal([1, 2, 3], [1, 2, 4]) -> [True, True, False]
        var a = np.array(new int[] { 1, 2, 3 });
        var b = np.array(new int[] { 1, 2, 4 });
        var result = np.equal(a, b);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeFalse();
    }

    [TestMethod]
    public async Task Equal_ArrayAndScalar_Compiles()
    {
        // NumPy: np.equal([1, 2, 3], 2) -> [False, True, False]
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.equal(a, 2);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeFalse();
    }

    [TestMethod]
    public async Task Equal_ScalarAndArray_Compiles()
    {
        // NumPy: np.equal(2, [1, 2, 3]) -> [False, True, False]
        var b = np.array(new int[] { 1, 2, 3 });
        var result = np.equal(2, b);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeFalse();
    }

    #endregion

    #region Comparison Operations - np.not_equal (3 overloads)

    [TestMethod]
    public async Task NotEqual_TwoArrays_Compiles()
    {
        // NumPy: np.not_equal([1, 2, 3], [1, 2, 4]) -> [False, False, True]
        var a = np.array(new int[] { 1, 2, 3 });
        var b = np.array(new int[] { 1, 2, 4 });
        var result = np.not_equal(a, b);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task NotEqual_ArrayAndScalar_Compiles()
    {
        // NumPy: np.not_equal([1, 2, 3], 2) -> [True, False, True]
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.not_equal(a, 2);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task NotEqual_ScalarAndArray_Compiles()
    {
        // NumPy: np.not_equal(2, [1, 2, 3]) -> [True, False, True]
        var b = np.array(new int[] { 1, 2, 3 });
        var result = np.not_equal(2, b);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    #endregion

    #region Comparison Operations - np.less (3 overloads)

    [TestMethod]
    public async Task Less_TwoArrays_Compiles()
    {
        // NumPy: np.less([1, 2, 3], [2, 2, 2]) -> [True, False, False]
        var a = np.array(new int[] { 1, 2, 3 });
        var b = np.array(new int[] { 2, 2, 2 });
        var result = np.less(a, b);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeFalse();
    }

    [TestMethod]
    public async Task Less_ArrayAndScalar_Compiles()
    {
        // NumPy: np.less([1, 2, 3], 2) -> [True, False, False]
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.less(a, 2);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeFalse();
    }

    [TestMethod]
    public async Task Less_ScalarAndArray_Compiles()
    {
        // NumPy: np.less(2, [1, 2, 3]) -> [False, False, True]
        var b = np.array(new int[] { 1, 2, 3 });
        var result = np.less(2, b);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    #endregion

    #region Comparison Operations - np.greater (3 overloads)

    [TestMethod]
    public async Task Greater_TwoArrays_Compiles()
    {
        // NumPy: np.greater([1, 2, 3], [2, 2, 2]) -> [False, False, True]
        var a = np.array(new int[] { 1, 2, 3 });
        var b = np.array(new int[] { 2, 2, 2 });
        var result = np.greater(a, b);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task Greater_ArrayAndScalar_Compiles()
    {
        // NumPy: np.greater([1, 2, 3], 2) -> [False, False, True]
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.greater(a, 2);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task Greater_ScalarAndArray_Compiles()
    {
        // NumPy: np.greater(2, [1, 2, 3]) -> [True, False, False]
        var b = np.array(new int[] { 1, 2, 3 });
        var result = np.greater(2, b);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeFalse();
    }

    #endregion

    #region Comparison Operations - np.less_equal (3 overloads)

    [TestMethod]
    public async Task LessEqual_TwoArrays_Compiles()
    {
        // NumPy: np.less_equal([1, 2, 3], [2, 2, 2]) -> [True, True, False]
        var a = np.array(new int[] { 1, 2, 3 });
        var b = np.array(new int[] { 2, 2, 2 });
        var result = np.less_equal(a, b);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeFalse();
    }

    [TestMethod]
    public async Task LessEqual_ArrayAndScalar_Compiles()
    {
        // NumPy: np.less_equal([1, 2, 3], 2) -> [True, True, False]
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.less_equal(a, 2);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeFalse();
    }

    [TestMethod]
    public async Task LessEqual_ScalarAndArray_Compiles()
    {
        // NumPy: np.less_equal(2, [1, 2, 3]) -> [False, True, True]
        var b = np.array(new int[] { 1, 2, 3 });
        var result = np.less_equal(2, b);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeTrue();
    }

    #endregion

    #region Comparison Operations - np.greater_equal (3 overloads)

    [TestMethod]
    public async Task GreaterEqual_TwoArrays_Compiles()
    {
        // NumPy: np.greater_equal([1, 2, 3], [2, 2, 2]) -> [False, True, True]
        var a = np.array(new int[] { 1, 2, 3 });
        var b = np.array(new int[] { 2, 2, 2 });
        var result = np.greater_equal(a, b);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task GreaterEqual_ArrayAndScalar_Compiles()
    {
        // NumPy: np.greater_equal([1, 2, 3], 2) -> [False, True, True]
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.greater_equal(a, 2);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task GreaterEqual_ScalarAndArray_Compiles()
    {
        // NumPy: np.greater_equal(2, [1, 2, 3]) -> [True, True, False]
        var b = np.array(new int[] { 1, 2, 3 });
        var result = np.greater_equal(2, b);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeFalse();
    }

    #endregion

    #region Logical Operations (4 overloads)

    [TestMethod]
    public async Task LogicalAnd_TwoArrays_Compiles()
    {
        // NumPy: np.logical_and([True, True, False, False], [True, False, True, False])
        //        -> [True, False, False, False]
        var x = np.array(new bool[] { true, true, false, false });
        var y = np.array(new bool[] { true, false, true, false });
        var result = np.logical_and(x, y);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeFalse();
        result.GetBoolean(3).Should().BeFalse();
    }

    [TestMethod]
    public async Task LogicalOr_TwoArrays_Compiles()
    {
        // NumPy: np.logical_or([True, True, False, False], [True, False, True, False])
        //        -> [True, True, True, False]
        var x = np.array(new bool[] { true, true, false, false });
        var y = np.array(new bool[] { true, false, true, false });
        var result = np.logical_or(x, y);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeTrue();
        result.GetBoolean(3).Should().BeFalse();
    }

    [TestMethod]
    public async Task LogicalNot_Array_Compiles()
    {
        // NumPy: np.logical_not([True, True, False, False]) -> [False, False, True, True]
        var x = np.array(new bool[] { true, true, false, false });
        var result = np.logical_not(x);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
        result.GetBoolean(3).Should().BeTrue();
    }

    [TestMethod]
    public async Task LogicalXor_TwoArrays_Compiles()
    {
        // NumPy: np.logical_xor([True, True, False, False], [True, False, True, False])
        //        -> [False, True, True, False]
        var x = np.array(new bool[] { true, true, false, false });
        var y = np.array(new bool[] { true, false, true, false });
        var result = np.logical_xor(x, y);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeTrue();
        result.GetBoolean(3).Should().BeFalse();
    }

    #endregion

    #region Axis Manipulation - np.moveaxis (4 overloads)

    [TestMethod]
    public async Task MoveAxis_IntSource_IntDest_Compiles()
    {
        // NumPy: np.moveaxis(np.zeros((2, 3, 4)), 0, -1).shape -> (3, 4, 2)
        var a = np.zeros(new Shape(2, 3, 4));
        var result = np.moveaxis(a, 0, -1);

        result.Should().NotBeNull();
        result.shape[0].Should().Be(3);
        result.shape[1].Should().Be(4);
        result.shape[2].Should().Be(2);
    }

    [TestMethod]
    public async Task MoveAxis_ArraySource_IntDest_Compiles()
    {
        // NumPy: np.moveaxis(np.zeros((2, 3, 4)), [0, 1], 0) is not valid actually
        // Let's do a valid one: moveaxis with array source to single dest
        // Actually this overload moves multiple sources to a single dest which isn't standard
        // Testing that it compiles - using valid array to single position
        var a = np.zeros(new Shape(2, 3, 4));
        // This should move axis 0 to position 2 (equivalent to int, int)
        var result = np.moveaxis(a, new int[] { 0 }, 2);

        result.Should().NotBeNull();
        result.ndim.Should().Be(3);
    }

    [TestMethod]
    public async Task MoveAxis_IntSource_ArrayDest_Compiles()
    {
        // Similar - move single source to array of destinations
        var a = np.zeros(new Shape(2, 3, 4));
        var result = np.moveaxis(a, 0, new int[] { 2 });

        result.Should().NotBeNull();
        result.ndim.Should().Be(3);
    }

    [TestMethod]
    public async Task MoveAxis_ArraySource_ArrayDest_Compiles()
    {
        // NumPy: np.moveaxis(np.zeros((2, 3, 4)), [0, 1], [1, 0]).shape -> (3, 2, 4)
        var a = np.zeros(new Shape(2, 3, 4));
        var result = np.moveaxis(a, new int[] { 0, 1 }, new int[] { 1, 0 });

        result.Should().NotBeNull();
        result.shape[0].Should().Be(3);
        result.shape[1].Should().Be(2);
        result.shape[2].Should().Be(4);
    }

    #endregion

    #region Axis Manipulation - np.rollaxis (1 overload)

    [TestMethod]
    public async Task RollAxis_WithStart_Compiles()
    {
        // NumPy: np.rollaxis(np.zeros((2, 3, 4)), 2, 0).shape -> (4, 2, 3)
        var a = np.zeros(new Shape(2, 3, 4));
        var result = np.rollaxis(a, 2, 0);

        result.Should().NotBeNull();
        result.shape[0].Should().Be(4);
        result.shape[1].Should().Be(2);
        result.shape[2].Should().Be(3);
    }

    [TestMethod]
    public async Task RollAxis_DefaultStart_Compiles()
    {
        // NumPy: np.rollaxis(np.zeros((2, 3, 4)), 2).shape -> (4, 2, 3) (start defaults to 0)
        var a = np.zeros(new Shape(2, 3, 4));
        var result = np.rollaxis(a, 2);

        result.Should().NotBeNull();
        result.shape[0].Should().Be(4);
        result.shape[1].Should().Be(2);
        result.shape[2].Should().Be(3);
    }

    #endregion

    #region Axis Manipulation - np.swapaxes (1 overload)

    [TestMethod]
    public async Task SwapAxes_Compiles()
    {
        // NumPy: np.swapaxes(np.zeros((2, 3, 4)), 0, 2).shape -> (4, 3, 2)
        var a = np.zeros(new Shape(2, 3, 4));
        var result = np.swapaxes(a, 0, 2);

        result.Should().NotBeNull();
        result.shape[0].Should().Be(4);
        result.shape[1].Should().Be(3);
        result.shape[2].Should().Be(2);
    }

    #endregion

    #region Axis Manipulation - np.transpose (2 overloads)

    [TestMethod]
    public async Task Transpose_NoArgs_Compiles()
    {
        // NumPy: np.transpose(np.arange(6).reshape(2, 3)).shape -> (3, 2)
        var a = np.arange(6).reshape(2, 3);
        var result = np.transpose(a);

        result.Should().NotBeNull();
        result.shape[0].Should().Be(3);
        result.shape[1].Should().Be(2);
    }

    [TestMethod]
    public async Task Transpose_WithPermute_Compiles()
    {
        // NumPy: np.transpose(np.arange(6).reshape(2, 3), [1, 0]).shape -> (3, 2)
        var a = np.arange(6).reshape(2, 3);
        var result = np.transpose(a, new int[] { 1, 0 });

        result.Should().NotBeNull();
        result.shape[0].Should().Be(3);
        result.shape[1].Should().Be(2);
    }

    [TestMethod]
    public async Task Transpose_3D_Compiles()
    {
        // NumPy: np.transpose(np.arange(24).reshape(2, 3, 4), [2, 0, 1]).shape -> (4, 2, 3)
        var a = np.arange(24).reshape(2, 3, 4);
        var result = np.transpose(a, new int[] { 2, 0, 1 });

        result.Should().NotBeNull();
        result.shape[0].Should().Be(4);
        result.shape[1].Should().Be(2);
        result.shape[2].Should().Be(3);
    }

    #endregion

    #region np.unique (1 overload)

    [TestMethod]
    public async Task Unique_1D_Compiles()
    {
        // NumPy: np.unique([1, 2, 2, 3, 3, 3]) -> [1, 2, 3]
        var a = np.array(new int[] { 1, 2, 2, 3, 3, 3 });
        var result = np.unique(a);

        result.Should().NotBeNull();
        result.size.Should().Be(3);
        result.GetInt32(0).Should().Be(1);
        result.GetInt32(1).Should().Be(2);
        result.GetInt32(2).Should().Be(3);
    }

    [TestMethod]
    public async Task Unique_2D_Flattens_Compiles()
    {
        // NumPy: np.unique([[1, 1], [2, 3]]) -> [1, 2, 3]
        var a = np.array(new int[,] { { 1, 1 }, { 2, 3 } });
        var result = np.unique(a);

        result.Should().NotBeNull();
        result.ndim.Should().Be(1);
        result.size.Should().Be(3);
    }

    #endregion

    #region Counting/Indexing - np.count_nonzero (2 overloads)

    [TestMethod]
    public async Task CountNonzero_NoAxis_ReturnsInt_Compiles()
    {
        // NumPy: np.count_nonzero([0, 1, 0, 2, 0, 3]) -> 3
        var a = np.array(new int[] { 0, 1, 0, 2, 0, 3 });
        long result = np.count_nonzero(a);

        result.Should().Be(3);
    }

    [TestMethod]
    public async Task CountNonzero_WithAxis_ReturnsNDArray_Compiles()
    {
        // NumPy: np.count_nonzero([[0, 1, 2], [3, 0, 5]], axis=0) -> [1, 1, 2]
        var a = np.array(new int[,] { { 0, 1, 2 }, { 3, 0, 5 } });
        var result = np.count_nonzero(a, axis: 0);

        result.Should().NotBeNull();
        result.shape.Should().BeEquivalentTo(new long[] { 3 });
        result.GetInt64(0).Should().Be(1L);
        result.GetInt64(1).Should().Be(1L);
        result.GetInt64(2).Should().Be(2L);
    }

    [TestMethod]
    public async Task CountNonzero_WithAxisKeepdims_Compiles()
    {
        // NumPy: np.count_nonzero([[0, 1, 2], [3, 0, 5]], axis=1, keepdims=True) -> [[2], [2]]
        var a = np.array(new int[,] { { 0, 1, 2 }, { 3, 0, 5 } });
        var result = np.count_nonzero(a, axis: 1, keepdims: true);

        result.Should().NotBeNull();
        result.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
    }

    #endregion

    #region Counting/Indexing - np.nonzero (1 overload)

    [TestMethod]
    public async Task Nonzero_1D_ReturnsNDArrayIntArray_Compiles()
    {
        // NumPy: np.nonzero([0, 1, 0, 2]) -> (array([1, 3]),)
        var a = np.array(new int[] { 0, 1, 0, 2 });
        NDArray<long>[] result = np.nonzero(a);

        result.Should().NotBeNull();
        result.Length.Should().Be(1);
        result[0].size.Should().Be(2);
        result[0].GetInt64(0).Should().Be(1L);
        result[0].GetInt64(1).Should().Be(3L);
    }

    [TestMethod]
    public async Task Nonzero_2D_ReturnsMultipleArrays_Compiles()
    {
        // NumPy: np.nonzero([[0, 1], [2, 0]]) -> (array([0, 1]), array([1, 0]))
        var a = np.array(new int[,] { { 0, 1 }, { 2, 0 } });
        NDArray<long>[] result = np.nonzero(a);

        result.Should().NotBeNull();
        result.Length.Should().Be(2);
        // Row indices of nonzero elements
        result[0].size.Should().Be(2);
        // Column indices of nonzero elements
        result[1].size.Should().Be(2);
    }

    #endregion

    #region Linear Algebra - np.dot (1 overload)

    [TestMethod]
    public async Task Dot_2DMatrices_Compiles()
    {
        // NumPy: np.dot([[1, 2], [3, 4]], [[5, 6], [7, 8]]) -> [[19, 22], [43, 50]]
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new int[,] { { 5, 6 }, { 7, 8 } });
        var result = np.dot(a, b);

        result.Should().NotBeNull();
        result.shape.Should().BeEquivalentTo(new long[] { 2, 2 });
        result.GetInt32(0, 0).Should().Be(19);
        result.GetInt32(0, 1).Should().Be(22);
        result.GetInt32(1, 0).Should().Be(43);
        result.GetInt32(1, 1).Should().Be(50);
    }

    [TestMethod]
    public async Task Dot_1DVectors_Compiles()
    {
        // NumPy: np.dot([1, 2, 3], [4, 5, 6]) -> 32 (1*4 + 2*5 + 3*6)
        var a = np.array(new int[] { 1, 2, 3 });
        var b = np.array(new int[] { 4, 5, 6 });
        var result = np.dot(a, b);

        result.Should().NotBeNull();
        // For 1D vectors, dot returns a scalar (0D array)
        result.size.Should().Be(1);
    }

    #endregion

    #region Linear Algebra - np.matmul (1 overload)

    [TestMethod]
    public async Task Matmul_2DMatrices_Compiles()
    {
        // NumPy: np.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) -> [[19, 22], [43, 50]]
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new int[,] { { 5, 6 }, { 7, 8 } });
        var result = np.matmul(a, b);

        result.Should().NotBeNull();
        result.shape.Should().BeEquivalentTo(new long[] { 2, 2 });
        result.GetInt32(0, 0).Should().Be(19);
        result.GetInt32(0, 1).Should().Be(22);
        result.GetInt32(1, 0).Should().Be(43);
        result.GetInt32(1, 1).Should().Be(50);
    }

    #endregion

    #region Linear Algebra - np.outer (1 overload)

    [TestMethod]
    public async Task Outer_1DVectors_Compiles()
    {
        // NumPy: np.outer([1, 2, 3], [4, 5, 6]) ->
        //        [[ 4,  5,  6],
        //         [ 8, 10, 12],
        //         [12, 15, 18]]
        var a = np.array(new int[] { 1, 2, 3 });
        var b = np.array(new int[] { 4, 5, 6 });
        var result = np.outer(a, b);

        result.Should().NotBeNull();
        result.shape.Should().BeEquivalentTo(new long[] { 3, 3 });
        result.GetInt32(0, 0).Should().Be(4);
        result.GetInt32(0, 1).Should().Be(5);
        result.GetInt32(0, 2).Should().Be(6);
        result.GetInt32(1, 0).Should().Be(8);
        result.GetInt32(1, 1).Should().Be(10);
        result.GetInt32(1, 2).Should().Be(12);
        result.GetInt32(2, 0).Should().Be(12);
        result.GetInt32(2, 1).Should().Be(15);
        result.GetInt32(2, 2).Should().Be(18);
    }

    #endregion

    #region Min with dtype - np.amin (2 overloads)

    [TestMethod]
    public async Task AminGeneric_ReturnsT_Compiles()
    {
        // NumPy: np.amin([1, 2, 3, 4, 5]) -> 1
        var a = np.array(new int[] { 1, 2, 3, 4, 5 });
        int result = np.amin<int>(a);

        result.Should().Be(1);
    }

    [TestMethod]
    public async Task AminGeneric_Float_Compiles()
    {
        // NumPy: np.amin([1.5, 2.5, 0.5]) -> 0.5
        var a = np.array(new double[] { 1.5, 2.5, 0.5 });
        double result = np.amin<double>(a);

        result.Should().Be(0.5);
    }

    [TestMethod]
    public async Task Amin_WithAxis_Compiles()
    {
        // NumPy: np.amin([[0, 1, 2], [3, 0, 5]], axis=0) -> [0, 0, 2]
        var a = np.array(new int[,] { { 0, 1, 2 }, { 3, 0, 5 } });
        var result = np.amin(a, axis: 0);

        result.Should().NotBeNull();
        result.shape.Should().BeEquivalentTo(new long[] { 3 });
        result.GetInt32(0).Should().Be(0);
        result.GetInt32(1).Should().Be(0);
        result.GetInt32(2).Should().Be(2);
    }

    [TestMethod]
    public async Task Amin_WithAxisKeepdims_Compiles()
    {
        // NumPy: np.amin([[0, 1, 2], [3, 0, 5]], axis=1, keepdims=True) -> [[0], [0]]
        var a = np.array(new int[,] { { 0, 1, 2 }, { 3, 0, 5 } });
        var result = np.amin(a, axis: 1, keepdims: true);

        result.Should().NotBeNull();
        result.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
    }

    [TestMethod]
    public async Task Amin_WithDtype_Compiles()
    {
        // Test that dtype parameter compiles (behavior may vary)
        var a = np.array(new int[] { 1, 2, 3, 4, 5 });
        var result = np.amin(a, dtype: typeof(double));

        result.Should().NotBeNull();
    }

    #endregion

    #region Min with dtype - np.min (1 overload)

    [TestMethod]
    public async Task Min_WithAxis_Compiles()
    {
        // NumPy: np.min([[0, 1, 2], [3, 0, 5]], axis=0) -> [0, 0, 2]
        var a = np.array(new int[,] { { 0, 1, 2 }, { 3, 0, 5 } });
        var result = np.min(a, axis: 0);

        result.Should().NotBeNull();
        result.shape.Should().BeEquivalentTo(new long[] { 3 });
        result.GetInt32(0).Should().Be(0);
        result.GetInt32(1).Should().Be(0);
        result.GetInt32(2).Should().Be(2);
    }

    [TestMethod]
    public async Task Min_WithDtype_Compiles()
    {
        // Test that dtype parameter compiles (behavior may vary)
        var a = np.array(new int[] { 1, 2, 3, 4, 5 });
        var result = np.min(a, dtype: typeof(double));

        result.Should().NotBeNull();
    }

    [TestMethod]
    public async Task Min_WithAxisKeepdimsDtype_Compiles()
    {
        // Test full signature with all optional parameters
        var a = np.array(new int[,] { { 0, 1, 2 }, { 3, 0, 5 } });
        var result = np.min(a, axis: 1, keepdims: true, dtype: null);

        result.Should().NotBeNull();
        result.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
    }

    #endregion

    #region Additional Edge Cases

    [TestMethod]
    public async Task Comparison_WithFloats_Compiles()
    {
        // NumPy: np.equal([1.0, 2.0, 3.0], [1.0, 2.1, 3.0]) -> [True, False, True]
        var a = np.array(new double[] { 1.0, 2.0, 3.0 });
        var b = np.array(new double[] { 1.0, 2.1, 3.0 });
        var result = np.equal(a, b);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public async Task Logical_WithIntArrays_Compiles()
    {
        // NumPy: np.logical_and([1, 0, 1], [1, 1, 0]) -> [True, False, False]
        // Nonzero integers are truthy
        var x = np.array(new int[] { 1, 0, 1 });
        var y = np.array(new int[] { 1, 1, 0 });
        var result = np.logical_and(x, y);

        result.Should().NotBeNull();
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeFalse();
    }

    [TestMethod]
    public async Task Transpose_1D_ReturnsUnchanged_Compiles()
    {
        // NumPy: np.transpose([1, 2, 3]).shape -> (3,) - unchanged for 1D
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.transpose(a);

        result.Should().NotBeNull();
        result.shape.Should().BeEquivalentTo(new long[] { 3 });
    }

    [TestMethod]
    public async Task Unique_EmptyArray_Compiles()
    {
        // NumPy: np.unique([]) -> []
        var a = np.array(new int[0]);
        var result = np.unique(a);

        result.Should().NotBeNull();
        result.size.Should().Be(0);
    }

    [TestMethod]
    [OpenBugs]  // np.dot(matrix, vector) has existing bug - memory corruption
    public async Task Dot_MatrixVector_Compiles()
    {
        // NumPy: np.dot([[1, 2], [3, 4]], [1, 1]) -> [3, 7]
        // Note: This test verifies the signature compiles correctly.
        // The actual functionality has known issues tracked separately.
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new int[] { 1, 1 });
        var result = np.dot(a, b);

        result.Should().NotBeNull();
        result.shape.Should().BeEquivalentTo(new long[] { 2 });
        result.GetInt32(0).Should().Be(3);
        result.GetInt32(1).Should().Be(7);
    }

    #endregion
}
