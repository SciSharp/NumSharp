using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest;

/// <summary>
/// Battle tests for NDArray.ToString() comparing against NumPy output.
/// These tests validate that ToString correctly handles all combinations of:
/// - Basic arrays (1D through 4D)
/// - Sliced arrays (reversed, stepped, negative step, multi-axis)
/// - Broadcast arrays
/// - Broadcast + slice combinations
/// - Transposed arrays
/// - Edge cases (empty, scalar, single element)
/// - All supported dtypes
/// </summary>
[TestClass]
public class np_ToString_BattleTests
{
    private static string Normalize(string s) =>
        s.Replace("\r\n", " ").Replace("\n", " ").Replace("  ", " ").Trim();

    private static bool ValuesMatch(string expected, string actual)
    {
        var e = expected.Replace(" ", "").Replace(".0,", ",").Replace(".0]", "]");
        var a = Normalize(actual).Replace(" ", "").Replace(".0,", ",").Replace(".0]", "]");
        return e == a;
    }

    #region Basic Arrays

    [TestMethod]
    public void ToString_1D_Simple()
    {
        var arr = np.arange(5);
        ValuesMatch("[0, 1, 2, 3, 4]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_2D_Simple()
    {
        var arr = np.arange(6).reshape(2, 3);
        ValuesMatch("[[0, 1, 2], [3, 4, 5]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_3D_Simple()
    {
        var arr = np.arange(8).reshape(2, 2, 2);
        ValuesMatch("[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_4D_Simple()
    {
        var arr = np.arange(16).reshape(2, 2, 2, 2);
        ValuesMatch("[[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [10, 11]], [[12, 13], [14, 15]]]]",
            arr.ToString(false)).Should().BeTrue();
    }

    #endregion

    #region Sliced Arrays

    [TestMethod]
    public void ToString_1D_Reversed()
    {
        var arr = np.arange(5)["::-1"];
        ValuesMatch("[4, 3, 2, 1, 0]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_1D_Stepped()
    {
        var arr = np.arange(10)["::2"];
        ValuesMatch("[0, 2, 4, 6, 8]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_1D_NegativeStep()
    {
        var arr = np.arange(10)["8:2:-2"];
        ValuesMatch("[8, 6, 4]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_2D_RowSlice()
    {
        var arr = np.arange(12).reshape(3, 4)["1:3"];
        ValuesMatch("[[4, 5, 6, 7], [8, 9, 10, 11]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_2D_ColSlice()
    {
        var arr = np.arange(12).reshape(3, 4)[":, 1:3"];
        ValuesMatch("[[1, 2], [5, 6], [9, 10]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_2D_ReversedRows()
    {
        var arr = np.arange(12).reshape(3, 4)["::-1"];
        ValuesMatch("[[8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_2D_ReversedCols()
    {
        var arr = np.arange(12).reshape(3, 4)[":, ::-1"];
        ValuesMatch("[[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_2D_BothReversed()
    {
        var arr = np.arange(12).reshape(3, 4)["::-1, ::-1"];
        ValuesMatch("[[11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_2D_SteppedRows()
    {
        var arr = np.arange(12).reshape(4, 3)["::2"];
        ValuesMatch("[[0, 1, 2], [6, 7, 8]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_2D_SkipBoth()
    {
        var arr = np.arange(12).reshape(3, 4)["::2, ::2"];
        ValuesMatch("[[0, 2], [8, 10]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_2D_ReverseSkip()
    {
        var arr = np.arange(12).reshape(3, 4)["::-2, ::-2"];
        ValuesMatch("[[11, 9], [3, 1]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_3D_MiddleSlice()
    {
        var arr = np.arange(24).reshape(2, 3, 4)["1:2, 1:2"];
        ValuesMatch("[[[16, 17, 18, 19]]]", arr.ToString(false)).Should().BeTrue();
    }

    #endregion

    #region Broadcast Arrays

    [TestMethod]
    public void ToString_Broadcast_1DTo2D()
    {
        var arr = np.broadcast_to(np.arange(3), new Shape(2, 3));
        ValuesMatch("[[0, 1, 2], [0, 1, 2]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Broadcast_ColTo2D()
    {
        var arr = np.broadcast_to(np.arange(3).reshape(3, 1), new Shape(3, 4));
        ValuesMatch("[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Broadcast_ScalarTo2D()
    {
        var arr = np.broadcast_to(np.array(5), new Shape(2, 3));
        ValuesMatch("[[5, 5, 5], [5, 5, 5]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Broadcast_ScalarTo3D()
    {
        var arr = np.broadcast_to(np.array(7), new Shape(2, 2, 2));
        ValuesMatch("[[[7, 7], [7, 7]], [[7, 7], [7, 7]]]", arr.ToString(false)).Should().BeTrue();
    }

    #endregion

    #region Broadcast + Slice Combinations (Originally OpenBugs)

    [TestMethod]
    public void ToString_BroadcastSlice_Reversed()
    {
        // This was Bug_ToString_ReversedSliceBroadcast
        var arr = np.broadcast_to(np.arange(3)["::-1"], new Shape(2, 3));
        ValuesMatch("[[2, 1, 0], [2, 1, 0]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_BroadcastSlice_Stepped()
    {
        // This was Bug_ToString_StepSliceBroadcast
        var arr = np.broadcast_to(np.arange(6)["::2"], new Shape(2, 3));
        ValuesMatch("[[0, 2, 4], [0, 2, 4]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_BroadcastSlice_ColSlice()
    {
        // This was Bug_ToString_SlicedColumnBroadcast
        var arr = np.broadcast_to(np.arange(12).reshape(3, 4)[":, 1:2"], new Shape(3, 3));
        ValuesMatch("[[1, 1, 1], [5, 5, 5], [9, 9, 9]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_BroadcastSlice_DoubleSlice()
    {
        // This was Bug_ToString_DoubleSlicedBroadcast
        var x = np.arange(12).reshape(3, 4);
        var dslice = x["::2, :"];
        var dslice_col = dslice[":, 0:1"];
        var arr = np.broadcast_to(dslice_col, new Shape(2, 4));
        ValuesMatch("[[0, 0, 0, 0], [8, 8, 8, 8]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_BroadcastSlice_SliceOfBroadcast()
    {
        var b = np.broadcast_to(np.arange(3), new Shape(4, 3));
        var arr = b["1:3"];
        ValuesMatch("[[0, 1, 2], [0, 1, 2]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_BroadcastSlice_ReverseSliceOfBroadcast()
    {
        var b = np.broadcast_to(np.arange(3), new Shape(4, 3));
        var arr = b["3:1:-1"];
        ValuesMatch("[[0, 1, 2], [0, 1, 2]]", arr.ToString(false)).Should().BeTrue();
    }

    #endregion

    #region Transpose

    [TestMethod]
    public void ToString_Transpose_2D()
    {
        var arr = np.arange(6).reshape(2, 3).T;
        ValuesMatch("[[0, 3], [1, 4], [2, 5]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Transpose_3D()
    {
        var arr = np.arange(24).reshape(2, 3, 4).transpose(new int[] { 2, 0, 1 });
        ValuesMatch("[[[0, 4, 8], [12, 16, 20]], [[1, 5, 9], [13, 17, 21]], [[2, 6, 10], [14, 18, 22]], [[3, 7, 11], [15, 19, 23]]]",
            arr.ToString(false)).Should().BeTrue();
    }

    #endregion

    #region Edge Cases

    [TestMethod]
    public void ToString_Empty_1D()
    {
        var arr = np.array(new long[0]);
        ValuesMatch("[]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Empty_2D()
    {
        var arr = np.zeros(new Shape(0, 3));
        ValuesMatch("[]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Scalar()
    {
        var arr = np.array(42);
        ValuesMatch("42", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Single_1D()
    {
        var arr = np.array(new long[] { 7 });
        ValuesMatch("[7]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Single_2D()
    {
        var arr = np.array(new long[] { 7 }).reshape(1, 1);
        ValuesMatch("[[7]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_3D_AllReversed()
    {
        var arr = np.arange(8).reshape(2, 2, 2)["::-1, ::-1, ::-1"];
        ValuesMatch("[[[7, 6], [5, 4]], [[3, 2], [1, 0]]]", arr.ToString(false)).Should().BeTrue();
    }

    #endregion

    #region Chained Slices

    [TestMethod]
    public void ToString_ChainedSlices()
    {
        var e = np.arange(100).reshape(10, 10);
        var e1 = e["2:8"];
        var e2 = e1[":, 3:7"];
        var e3 = e2["1:4"];
        ValuesMatch("[[33, 34, 35, 36], [43, 44, 45, 46], [53, 54, 55, 56]]", e3.ToString(false)).Should().BeTrue();
    }

    #endregion

    #region Negative Indices

    [TestMethod]
    public void ToString_NegativeIndices()
    {
        var arr = np.arange(30).reshape(3, 10)["-3:-1, -4:"];
        ValuesMatch("[[6, 7, 8, 9], [16, 17, 18, 19]]", arr.ToString(false)).Should().BeTrue();
    }

    #endregion

    #region Dtypes

    [TestMethod]
    public void ToString_Dtype_Int16()
    {
        var arr = np.array(new short[] { 1, 2, 3 });
        ValuesMatch("[1, 2, 3]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Dtype_Int32()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        ValuesMatch("[1, 2, 3]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Dtype_Int64()
    {
        var arr = np.array(new long[] { 1, 2, 3 });
        ValuesMatch("[1, 2, 3]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Dtype_Float32()
    {
        var arr = np.array(new float[] { 1f, 2f, 3f });
        ValuesMatch("[1, 2, 3]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Dtype_Float64()
    {
        var arr = np.array(new double[] { 1.0, 2.0, 3.0 });
        ValuesMatch("[1, 2, 3]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Dtype_Bool()
    {
        var arr = np.array(new bool[] { true, false, true });
        ValuesMatch("[True, False, True]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Dtype_Byte()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        ValuesMatch("[1, 2, 3]", arr.ToString(false)).Should().BeTrue();
    }

    #endregion

    #region Extreme Edge Cases

    [TestMethod]
    public void ToString_5D_Array()
    {
        var arr = np.arange(16).reshape(1, 2, 2, 2, 2);
        ValuesMatch("[[[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [10, 11]], [[12, 13], [14, 15]]]]]",
            arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Long_1D()
    {
        var arr = np.arange(20);
        ValuesMatch("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]",
            arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_BroadcastThenSlice()
    {
        var a = np.arange(2).reshape(2, 1);
        var ba = np.broadcast_to(a, new Shape(2, 3));
        var arr = ba["0:1"];
        ValuesMatch("[[0, 0, 0]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_ReduceToSingleElement3D()
    {
        var arr = np.arange(24).reshape(2, 3, 4)["0:1, 1:2, 1:2"];
        ValuesMatch("[[[5]]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_AlternatingReverse()
    {
        var arr = np.arange(10)["::-2"];
        ValuesMatch("[9, 7, 5, 3, 1]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_BroadcastSlicedTransposed()
    {
        var b = np.arange(6).reshape(2, 3).T;  // Shape (3, 2)
        var sb = b["1:2, :"];  // Shape (1, 2) = [[1, 4]]
        var arr = np.broadcast_to(sb, new Shape(3, 2));
        ValuesMatch("[[1, 4], [1, 4], [1, 4]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_ScalarBroadcast3D()
    {
        var s = np.array(42);
        var arr = np.broadcast_to(s, new Shape(2, 2, 2));
        ValuesMatch("[[[42, 42], [42, 42]], [[42, 42], [42, 42]]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_SingleRowSlice()
    {
        var arr = np.arange(12).reshape(3, 4)["1:2"];
        ValuesMatch("[[4, 5, 6, 7]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_SingleColSlice()
    {
        var arr = np.arange(12).reshape(3, 4)[":, 1:2"];
        ValuesMatch("[[1], [5], [9]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_TransposeSliceBroadcast()
    {
        var c = np.arange(12).reshape(3, 4).T;  // (4, 3)
        var sc = c["1:3, ::2"];  // (2, 2) = [[1, 9], [2, 10]]
        var arr = np.broadcast_to(sc, new Shape(3, 2, 2));
        ValuesMatch("[[[1, 9], [2, 10]], [[1, 9], [2, 10]], [[1, 9], [2, 10]]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_TallNarrow()
    {
        var arr = np.arange(10).reshape(10, 1);
        ValuesMatch("[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_ShortWide()
    {
        var arr = np.arange(10).reshape(1, 10);
        ValuesMatch("[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]", arr.ToString(false)).Should().BeTrue();
    }

    #endregion

    #region Special Values

    [TestMethod]
    public void ToString_NaN()
    {
        var arr = np.array(new double[] { double.NaN });
        ValuesMatch("[NaN]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_NegativeInts()
    {
        var arr = np.arange(-5, 1);
        ValuesMatch("[-5, -4, -3, -2, -1, 0]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Negative2D()
    {
        var arr = np.arange(-3, 3).reshape(2, 3);
        ValuesMatch("[[-3, -2, -1], [0, 1, 2]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_LargeInts()
    {
        var arr = np.array(new long[] { 1000000, 2000000, 3000000 });
        ValuesMatch("[1000000, 2000000, 3000000]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Decimals()
    {
        var arr = np.array(new double[] { 0.5, 1.5, 2.5 });
        Normalize(arr.ToString(false)).Should().Contain("0.5").And.Contain("1.5").And.Contain("2.5");
    }

    [TestMethod]
    public void ToString_AllTrue()
    {
        var arr = np.array(new bool[] { true, true, true });
        ValuesMatch("[True, True, True]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_AllFalse()
    {
        var arr = np.array(new bool[] { false, false, false });
        ValuesMatch("[False, False, False]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Bool2D()
    {
        var arr = np.array(new bool[] { true, false, false, true }).reshape(2, 2);
        ValuesMatch("[[True, False], [False, True]]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Int32Extremes()
    {
        var arr = np.array(new int[] { int.MinValue, int.MaxValue });
        ValuesMatch("[-2147483648, 2147483647]", arr.ToString(false)).Should().BeTrue();
    }

    [TestMethod]
    public void ToString_Int64Extremes()
    {
        var arr = np.array(new long[] { long.MinValue, long.MaxValue });
        ValuesMatch("[-9223372036854775808, 9223372036854775807]", arr.ToString(false)).Should().BeTrue();
    }

    #endregion
}
