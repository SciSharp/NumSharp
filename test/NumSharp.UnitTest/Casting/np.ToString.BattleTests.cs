using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest;

/// <summary>
/// Battle tests for <see cref="NDArray.ToString()"/> asserting BYTE-EXACT parity with NumPy 2.4.2's
/// <c>str(ndarray)</c> (i.e. <c>np.array_str</c>). Every expected string was produced by running the
/// equivalent NumPy 2.4.2 code. These exercise the layout engine across views: basic dims, sliced
/// (reversed/stepped/negative-step/multi-axis), broadcast, broadcast+slice, transposed, and edge
/// cases (empty/scalar/single), plus all supported dtypes and special values.
/// </summary>
[TestClass]
public class np_ToString_BattleTests
{
    private static void Str(NDArray a, string expected) => a.ToString(false).Should().Be(expected);

    #region Basic Arrays

    [TestMethod] public void ToString_1D_Simple() => Str(np.arange(5), "[0 1 2 3 4]");
    [TestMethod] public void ToString_2D_Simple() => Str(np.arange(6).reshape(2, 3), "[[0 1 2]\n [3 4 5]]");
    [TestMethod] public void ToString_3D_Simple() => Str(np.arange(8).reshape(2, 2, 2), "[[[0 1]\n  [2 3]]\n\n [[4 5]\n  [6 7]]]");

    [TestMethod]
    public void ToString_4D_Simple() => Str(np.arange(16).reshape(2, 2, 2, 2),
        "[[[[ 0  1]\n   [ 2  3]]\n\n  [[ 4  5]\n   [ 6  7]]]\n\n\n [[[ 8  9]\n   [10 11]]\n\n  [[12 13]\n   [14 15]]]]");

    [TestMethod]
    public void ToString_5D_Array() => Str(np.arange(16).reshape(1, 2, 2, 2, 2),
        "[[[[[ 0  1]\n    [ 2  3]]\n\n   [[ 4  5]\n    [ 6  7]]]\n\n\n  [[[ 8  9]\n    [10 11]]\n\n   [[12 13]\n    [14 15]]]]]");

    #endregion

    #region Sliced Arrays

    [TestMethod] public void ToString_1D_Reversed() => Str(np.arange(5)["::-1"], "[4 3 2 1 0]");
    [TestMethod] public void ToString_1D_Stepped() => Str(np.arange(10)["::2"], "[0 2 4 6 8]");
    [TestMethod] public void ToString_1D_NegativeStep() => Str(np.arange(10)["8:2:-2"], "[8 6 4]");
    [TestMethod] public void ToString_1D_AlternatingReverse() => Str(np.arange(10)["::-2"], "[9 7 5 3 1]");
    [TestMethod] public void ToString_2D_RowSlice() => Str(np.arange(12).reshape(3, 4)["1:3"], "[[ 4  5  6  7]\n [ 8  9 10 11]]");
    [TestMethod] public void ToString_2D_ColSlice() => Str(np.arange(12).reshape(3, 4)[":, 1:3"], "[[ 1  2]\n [ 5  6]\n [ 9 10]]");
    [TestMethod] public void ToString_2D_ReversedRows() => Str(np.arange(12).reshape(3, 4)["::-1"], "[[ 8  9 10 11]\n [ 4  5  6  7]\n [ 0  1  2  3]]");
    [TestMethod] public void ToString_2D_ReversedCols() => Str(np.arange(12).reshape(3, 4)[":, ::-1"], "[[ 3  2  1  0]\n [ 7  6  5  4]\n [11 10  9  8]]");
    [TestMethod] public void ToString_2D_BothReversed() => Str(np.arange(12).reshape(3, 4)["::-1, ::-1"], "[[11 10  9  8]\n [ 7  6  5  4]\n [ 3  2  1  0]]");
    [TestMethod] public void ToString_2D_SteppedRows() => Str(np.arange(12).reshape(4, 3)["::2"], "[[0 1 2]\n [6 7 8]]");
    [TestMethod] public void ToString_2D_SkipBoth() => Str(np.arange(12).reshape(3, 4)["::2, ::2"], "[[ 0  2]\n [ 8 10]]");
    [TestMethod] public void ToString_2D_ReverseSkip() => Str(np.arange(12).reshape(3, 4)["::-2, ::-2"], "[[11  9]\n [ 3  1]]");
    [TestMethod] public void ToString_3D_MiddleSlice() => Str(np.arange(24).reshape(2, 3, 4)["1:2, 1:2"], "[[[16 17 18 19]]]");
    [TestMethod] public void ToString_SingleRowSlice() => Str(np.arange(12).reshape(3, 4)["1:2"], "[[4 5 6 7]]");
    [TestMethod] public void ToString_SingleColSlice() => Str(np.arange(12).reshape(3, 4)[":, 1:2"], "[[1]\n [5]\n [9]]");

    #endregion

    #region Broadcast Arrays

    [TestMethod] public void ToString_Broadcast_1DTo2D() => Str(np.broadcast_to(np.arange(3), new Shape(2, 3)), "[[0 1 2]\n [0 1 2]]");
    [TestMethod] public void ToString_Broadcast_ColTo2D() => Str(np.broadcast_to(np.arange(3).reshape(3, 1), new Shape(3, 4)), "[[0 0 0 0]\n [1 1 1 1]\n [2 2 2 2]]");
    [TestMethod] public void ToString_Broadcast_ScalarTo2D() => Str(np.broadcast_to(np.array(5), new Shape(2, 3)), "[[5 5 5]\n [5 5 5]]");
    [TestMethod] public void ToString_Broadcast_ScalarTo3D() => Str(np.broadcast_to(np.array(7), new Shape(2, 2, 2)), "[[[7 7]\n  [7 7]]\n\n [[7 7]\n  [7 7]]]");
    [TestMethod] public void ToString_ScalarBroadcast3D() => Str(np.broadcast_to(np.array(42), new Shape(2, 2, 2)), "[[[42 42]\n  [42 42]]\n\n [[42 42]\n  [42 42]]]");

    #endregion

    #region Broadcast + Slice Combinations

    [TestMethod] public void ToString_BroadcastSlice_Reversed() => Str(np.broadcast_to(np.arange(3)["::-1"], new Shape(2, 3)), "[[2 1 0]\n [2 1 0]]");
    [TestMethod] public void ToString_BroadcastSlice_Stepped() => Str(np.broadcast_to(np.arange(6)["::2"], new Shape(2, 3)), "[[0 2 4]\n [0 2 4]]");
    [TestMethod] public void ToString_BroadcastSlice_ColSlice() => Str(np.broadcast_to(np.arange(12).reshape(3, 4)[":, 1:2"], new Shape(3, 3)), "[[1 1 1]\n [5 5 5]\n [9 9 9]]");

    [TestMethod]
    public void ToString_BroadcastSlice_DoubleSlice()
    {
        var dslice_col = np.arange(12).reshape(3, 4)["::2, :"][":, 0:1"];
        Str(np.broadcast_to(dslice_col, new Shape(2, 4)), "[[0 0 0 0]\n [8 8 8 8]]");
    }

    [TestMethod] public void ToString_BroadcastSlice_SliceOfBroadcast() => Str(np.broadcast_to(np.arange(3), new Shape(4, 3))["1:3"], "[[0 1 2]\n [0 1 2]]");
    [TestMethod] public void ToString_BroadcastSlice_ReverseSliceOfBroadcast() => Str(np.broadcast_to(np.arange(3), new Shape(4, 3))["3:1:-1"], "[[0 1 2]\n [0 1 2]]");
    [TestMethod] public void ToString_BroadcastThenSlice() => Str(np.broadcast_to(np.arange(2).reshape(2, 1), new Shape(2, 3))["0:1"], "[[0 0 0]]");

    [TestMethod]
    public void ToString_BroadcastSlicedTransposed()
    {
        var sb = np.arange(6).reshape(2, 3).T["1:2, :"];
        Str(np.broadcast_to(sb, new Shape(3, 2)), "[[1 4]\n [1 4]\n [1 4]]");
    }

    [TestMethod]
    public void ToString_TransposeSliceBroadcast()
    {
        var sc = np.arange(12).reshape(3, 4).T["1:3, ::2"];
        Str(np.broadcast_to(sc, new Shape(3, 2, 2)), "[[[ 1  9]\n  [ 2 10]]\n\n [[ 1  9]\n  [ 2 10]]\n\n [[ 1  9]\n  [ 2 10]]]");
    }

    #endregion

    #region Transpose

    [TestMethod] public void ToString_Transpose_2D() => Str(np.arange(6).reshape(2, 3).T, "[[0 3]\n [1 4]\n [2 5]]");

    [TestMethod]
    public void ToString_Transpose_3D() => Str(np.arange(24).reshape(2, 3, 4).transpose(new int[] { 2, 0, 1 }),
        "[[[ 0  4  8]\n  [12 16 20]]\n\n [[ 1  5  9]\n  [13 17 21]]\n\n [[ 2  6 10]\n  [14 18 22]]\n\n [[ 3  7 11]\n  [15 19 23]]]");

    #endregion

    #region Edge Cases

    [TestMethod] public void ToString_Empty_1D() => Str(np.array(new long[0]), "[]");
    [TestMethod] public void ToString_Empty_2D() => Str(np.zeros(new Shape(0, 3)), "[]");
    [TestMethod] public void ToString_Scalar() => Str(np.array(42), "42");
    [TestMethod] public void ToString_Single_1D() => Str(np.array(new long[] { 7 }), "[7]");
    [TestMethod] public void ToString_Single_2D() => Str(np.array(new long[] { 7 }).reshape(1, 1), "[[7]]");
    [TestMethod] public void ToString_3D_AllReversed() => Str(np.arange(8).reshape(2, 2, 2)["::-1, ::-1, ::-1"], "[[[7 6]\n  [5 4]]\n\n [[3 2]\n  [1 0]]]");
    [TestMethod] public void ToString_ReduceToSingleElement3D() => Str(np.arange(24).reshape(2, 3, 4)["0:1, 1:2, 1:2"], "[[[5]]]");
    [TestMethod] public void ToString_TallNarrow() => Str(np.arange(10).reshape(10, 1), "[[0]\n [1]\n [2]\n [3]\n [4]\n [5]\n [6]\n [7]\n [8]\n [9]]");
    [TestMethod] public void ToString_ShortWide() => Str(np.arange(10).reshape(1, 10), "[[0 1 2 3 4 5 6 7 8 9]]");

    [TestMethod]
    public void ToString_ChainedSlices()
    {
        var e3 = np.arange(100).reshape(10, 10)["2:8"][":, 3:7"]["1:4"];
        Str(e3, "[[33 34 35 36]\n [43 44 45 46]\n [53 54 55 56]]");
    }

    [TestMethod] public void ToString_NegativeIndices() => Str(np.arange(30).reshape(3, 10)["-3:-1, -4:"], "[[ 6  7  8  9]\n [16 17 18 19]]");

    #endregion

    #region Dtypes

    [TestMethod] public void ToString_Dtype_Int16() => Str(np.array(new short[] { 1, 2, 3 }), "[1 2 3]");
    [TestMethod] public void ToString_Dtype_Int32() => Str(np.array(new int[] { 1, 2, 3 }), "[1 2 3]");
    [TestMethod] public void ToString_Dtype_Int64() => Str(np.array(new long[] { 1, 2, 3 }), "[1 2 3]");
    [TestMethod] public void ToString_Dtype_Byte() => Str(np.array(new byte[] { 1, 2, 3 }), "[1 2 3]");
    [TestMethod] public void ToString_Dtype_Float32() => Str(np.array(new float[] { 1f, 2f, 3f }), "[1. 2. 3.]");
    [TestMethod] public void ToString_Dtype_Float64() => Str(np.array(new double[] { 1.0, 2.0, 3.0 }), "[1. 2. 3.]");
    [TestMethod] public void ToString_Dtype_Bool() => Str(np.array(new bool[] { true, false, true }), "[ True False  True]");

    #endregion

    #region Special Values

    [TestMethod] public void ToString_NaN() => Str(np.array(new double[] { double.NaN }), "[nan]");
    [TestMethod] public void ToString_NegativeInts() => Str(np.arange(-5, 1), "[-5 -4 -3 -2 -1  0]");
    [TestMethod] public void ToString_Negative2D() => Str(np.arange(-3, 3).reshape(2, 3), "[[-3 -2 -1]\n [ 0  1  2]]");
    [TestMethod] public void ToString_LargeInts() => Str(np.array(new long[] { 1000000, 2000000, 3000000 }), "[1000000 2000000 3000000]");
    [TestMethod] public void ToString_Decimals() => Str(np.array(new double[] { 0.5, 1.5, 2.5 }), "[0.5 1.5 2.5]");
    [TestMethod] public void ToString_AllTrue() => Str(np.array(new bool[] { true, true, true }), "[ True  True  True]");
    [TestMethod] public void ToString_AllFalse() => Str(np.array(new bool[] { false, false, false }), "[False False False]");
    [TestMethod] public void ToString_Bool2D() => Str(np.array(new bool[] { true, false, false, true }).reshape(2, 2), "[[ True False]\n [False  True]]");
    [TestMethod] public void ToString_Int32Extremes() => Str(np.array(new int[] { int.MinValue, int.MaxValue }), "[-2147483648  2147483647]");
    [TestMethod] public void ToString_Int64Extremes() => Str(np.array(new long[] { long.MinValue, long.MaxValue }), "[-9223372036854775808  9223372036854775807]");
    [TestMethod] public void ToString_Long_1D() => Str(np.arange(20), "[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]");

    #endregion
}
