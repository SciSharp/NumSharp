using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

[TestClass]
public class NpyIterScanBattleTests
{
    [TestMethod]
    public void Cumsum_RowBroadcast_Axis0_MatchesNumPyAndMaterializesWritableOutput()
    {
        // NumPy 2.4.2:
        // >>> a = np.broadcast_to(np.array([1, 2, 3]), (3, 3))
        // >>> np.cumsum(a, axis=0)
        // array([[1, 2, 3],
        //        [2, 4, 6],
        //        [3, 6, 9]])
        var arr = np.broadcast_to(np.array(new int[] { 1, 2, 3 }), new Shape(3, 3));

        var result = np.cumsum(arr, axis: 0);

        result.Should().BeShaped(3, 3);
        result.Should().BeOfValues(1L, 2L, 3L, 2L, 4L, 6L, 3L, 6L, 9L);
        result.Shape.IsBroadcasted.Should().BeFalse();
        result.Shape.IsWriteable.Should().BeTrue();
    }

    [TestMethod]
    public void Cumsum_ColumnBroadcast_Axis0_MatchesNumPyAndMaterializesWritableOutput()
    {
        // NumPy 2.4.2:
        // >>> a = np.broadcast_to(np.array([[1], [2], [3]]), (3, 3))
        // >>> np.cumsum(a, axis=0)
        // array([[1, 1, 1],
        //        [3, 3, 3],
        //        [6, 6, 6]])
        var col = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
        var arr = np.broadcast_to(col, new Shape(3, 3));

        var result = np.cumsum(arr, axis: 0);

        result.Should().BeShaped(3, 3);
        result.Should().BeOfValues(1L, 1L, 1L, 3L, 3L, 3L, 6L, 6L, 6L);
        result.Shape.IsBroadcasted.Should().BeFalse();
        result.Shape.IsWriteable.Should().BeTrue();
    }

    [TestMethod]
    public void Cumsum_ColumnBroadcast_Axis1_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> a = np.broadcast_to(np.array([[1], [2], [3]]), (3, 3))
        // >>> np.cumsum(a, axis=1)
        // array([[1, 2, 3],
        //        [2, 4, 6],
        //        [3, 6, 9]])
        var col = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
        var arr = np.broadcast_to(col, new Shape(3, 3));

        var result = np.cumsum(arr, axis: 1);

        result.Should().BeShaped(3, 3);
        result.Should().BeOfValues(1L, 2L, 3L, 2L, 4L, 6L, 3L, 6L, 9L);
    }

    [TestMethod]
    public void Cumsum_TransposedView_NoAxis_FollowsViewIterationOrder()
    {
        // NumPy 2.4.2:
        // >>> np.cumsum(np.arange(1, 13).reshape(3, 4).T)
        // array([ 1,  6, 15, 17, 23, 33, 36, 43, 54, 58, 66, 78])
        var arr = np.array(new int[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 }
        }).T;

        var result = np.cumsum(arr);

        result.Should().BeShaped(12);
        result.Should().BeOfValues(1L, 6L, 15L, 17L, 23L, 33L, 36L, 43L, 54L, 58L, 66L, 78L);
    }

    [TestMethod]
    public void Cumsum_TransposedView_Axis1_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> np.cumsum(np.arange(1, 13).reshape(3, 4).T, axis=1)
        // array([[ 1,  6, 15],
        //        [ 2,  8, 18],
        //        [ 3, 10, 21],
        //        [ 4, 12, 24]])
        var arr = np.array(new int[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 }
        }).T;

        var result = np.cumsum(arr, axis: 1);

        result.Should().BeShaped(4, 3);
        result.Should().BeOfValues(1L, 6L, 15L, 2L, 8L, 18L, 3L, 10L, 21L, 4L, 12L, 24L);
    }

    [TestMethod]
    [TestCategory("OpenBugs")]
    public void Cumsum_ReversedColumns_Axis1_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> np.cumsum(np.arange(1, 13).reshape(3, 4)[:, ::-1], axis=1)
        // array([[ 4,  7,  9, 10],
        //        [ 8, 15, 21, 26],
        //        [12, 23, 33, 42]])
        var arr = np.array(new int[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 }
        })[":, ::-1"];

        var result = np.cumsum(arr, axis: 1);

        result.Should().BeShaped(3, 4);
        result.Should().BeOfValues(4L, 7L, 9L, 10L, 8L, 15L, 21L, 26L, 12L, 23L, 33L, 42L);
    }

    [TestMethod]
    public void Cumsum_RowBroadcast_AxisNegative1_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> a = np.broadcast_to(np.array([1, 2, 3, 4]), (3, 4))
        // >>> np.cumsum(a, axis=-1)
        // array([[ 1,  3,  6, 10],
        //        [ 1,  3,  6, 10],
        //        [ 1,  3,  6, 10]])
        var arr = np.broadcast_to(np.array(new int[] { 1, 2, 3, 4 }), new Shape(3, 4));

        var result = np.cumsum(arr, axis: -1);

        result.Should().BeShaped(3, 4);
        result.Should().BeOfValues(1L, 3L, 6L, 10L, 1L, 3L, 6L, 10L, 1L, 3L, 6L, 10L);
    }

    [TestMethod]
    public void Cumsum_ColumnBroadcast_Axis1_OnWiderBroadcast_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> a = np.broadcast_to(np.array([[1], [2], [3]]), (3, 4))
        // >>> np.cumsum(a, axis=1)
        // array([[ 1,  2,  3,  4],
        //        [ 2,  4,  6,  8],
        //        [ 3,  6,  9, 12]])
        var col = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
        var arr = np.broadcast_to(col, new Shape(3, 4));

        var result = np.cumsum(arr, axis: 1);

        result.Should().BeShaped(3, 4);
        result.Should().BeOfValues(1L, 2L, 3L, 4L, 2L, 4L, 6L, 8L, 3L, 6L, 9L, 12L);
    }

    [TestMethod]
    public void Cumprod_RowBroadcast_Axis0_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> a = np.broadcast_to(np.array([1, 2, 3]), (3, 3))
        // >>> np.cumprod(a, axis=0)
        // array([[ 1,  2,  3],
        //        [ 1,  4,  9],
        //        [ 1,  8, 27]])
        var arr = np.broadcast_to(np.array(new int[] { 1, 2, 3 }), new Shape(3, 3));

        var result = np.cumprod(arr, axis: 0);

        result.Should().BeShaped(3, 3);
        result.Should().BeOfValues(1L, 2L, 3L, 1L, 4L, 9L, 1L, 8L, 27L);
    }

    [TestMethod]
    public void Cumprod_ColumnBroadcast_Axis1_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> a = np.broadcast_to(np.array([[1], [2], [3]]), (3, 4))
        // >>> np.cumprod(a, axis=1)
        // array([[ 1,  1,  1,  1],
        //        [ 2,  4,  8, 16],
        //        [ 3,  9, 27, 81]])
        var col = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
        var arr = np.broadcast_to(col, new Shape(3, 4));

        var result = np.cumprod(arr, axis: 1);

        result.Should().BeShaped(3, 4);
        result.Should().BeOfValues(1L, 1L, 1L, 1L, 2L, 4L, 8L, 16L, 3L, 9L, 27L, 81L);
    }

    [TestMethod]
    public void Cumprod_TransposedView_Axis0_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> np.cumprod(np.arange(1, 13).reshape(3, 4).T, axis=0)
        // array([[    1,     5,     9],
        //        [    2,    30,    90],
        //        [    6,   210,   990],
        //        [   24,  1680, 11880]])
        var arr = np.array(new int[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 }
        }).T;

        var result = np.cumprod(arr, axis: 0);

        result.Should().BeShaped(4, 3);
        result.Should().BeOfValues(1L, 5L, 9L, 2L, 30L, 90L, 6L, 210L, 990L, 24L, 1680L, 11880L);
    }

    [TestMethod]
    public void Cumprod_ReversedColumns_Axis1_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> np.cumprod(np.arange(1, 13).reshape(3, 4)[:, ::-1], axis=1)
        // array([[    4,    12,    24,    24],
        //        [    8,    56,   336,  1680],
        //        [   12,   132,  1320, 11880]])
        var arr = np.array(new int[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 }
        })[":, ::-1"];

        var result = np.cumprod(arr, axis: 1);

        result.Should().BeShaped(3, 4);
        result.Should().BeOfValues(4L, 12L, 24L, 24L, 8L, 56L, 336L, 1680L, 12L, 132L, 1320L, 11880L);
    }
}
