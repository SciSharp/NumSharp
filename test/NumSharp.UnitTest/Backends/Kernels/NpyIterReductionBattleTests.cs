using AwesomeAssertions;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

public class NpyIterReductionBattleTests
{
    private const double Tolerance = 1e-10;

    [Test]
    public void Var_ColumnBroadcast_Axis0_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> a = np.broadcast_to(np.array([[1.0], [2.0], [3.0]]), (3, 3))
        // >>> np.var(a, axis=0)
        // array([0.66666667, 0.66666667, 0.66666667])
        var col = np.array(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        var arr = np.broadcast_to(col, new Shape(3, 3));

        var result = np.var(arr, axis: 0);

        result.Should().BeShaped(3);
        result.Should().BeOfValuesApproximately(Tolerance, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);
    }

    [Test]
    public void Var_ColumnBroadcast_Axis0_Keepdims_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> a = np.broadcast_to(np.array([[1.0], [2.0], [3.0]]), (3, 3))
        // >>> np.var(a, axis=0, keepdims=True)
        // array([[0.66666667, 0.66666667, 0.66666667]])
        var col = np.array(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        var arr = np.broadcast_to(col, new Shape(3, 3));

        var result = np.var(arr, axis: 0, keepdims: true);

        result.Should().BeShaped(1, 3);
        result.Should().BeOfValuesApproximately(Tolerance, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);
    }

    [Test]
    public void Std_ColumnBroadcast_Axis0_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> a = np.broadcast_to(np.array([[1.0], [2.0], [3.0]]), (3, 3))
        // >>> np.std(a, axis=0)
        // array([0.81649658, 0.81649658, 0.81649658])
        var col = np.array(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
        var arr = np.broadcast_to(col, new Shape(3, 3));

        var result = np.std(arr, axis: 0);

        result.Should().BeShaped(3);
        result.Should().BeOfValuesApproximately(Tolerance, 0.816496580927726, 0.816496580927726, 0.816496580927726);
    }

    [Test]
    public void Var_ChainedTransposedReversedView_Axis1_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> a = np.arange(1., 13.).reshape(3, 4).T[:, ::-1]
        // >>> np.var(a, axis=1)
        // array([10.66666667, 10.66666667, 10.66666667, 10.66666667])
        var arr = np.array(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0, 12.0 }
        }).T[":, ::-1"];

        var result = np.var(arr, axis: 1);

        result.Should().BeShaped(4);
        result.Should().BeOfValuesApproximately(
            Tolerance,
            10.666666666666666,
            10.666666666666666,
            10.666666666666666,
            10.666666666666666);
    }

    [Test]
    public void Var_ChainedTransposedReversedView_Axis1_Keepdims_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> a = np.arange(1., 13.).reshape(3, 4).T[:, ::-1]
        // >>> np.var(a, axis=1, keepdims=True)
        // array([[10.66666667],
        //        [10.66666667],
        //        [10.66666667],
        //        [10.66666667]])
        var arr = np.array(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0, 12.0 }
        }).T[":, ::-1"];

        var result = np.var(arr, axis: 1, keepdims: true);

        result.Should().BeShaped(4, 1);
        result.Should().BeOfValuesApproximately(
            Tolerance,
            10.666666666666666,
            10.666666666666666,
            10.666666666666666,
            10.666666666666666);
    }

    [Test]
    public void Std_ChainedTransposedReversedView_Axis0_Ddof1_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> a = np.arange(1., 13.).reshape(3, 4).T[:, ::-1]
        // >>> np.std(a, axis=0, ddof=1)
        // array([1.29099445, 1.29099445, 1.29099445])
        var arr = np.array(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0, 12.0 }
        }).T[":, ::-1"];

        var result = np.std(arr, axis: 0, ddof: 1);

        result.Should().BeShaped(3);
        result.Should().BeOfValuesApproximately(Tolerance, 1.2909944487358056, 1.2909944487358056, 1.2909944487358056);
    }

    [Test]
    public void Var_ReversedStrideView_Axis0_Keepdims_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> a = np.arange(1., 13.).reshape(3, 4)[:, ::-2]
        // >>> np.var(a, axis=0, keepdims=True)
        // array([[10.66666667, 10.66666667]])
        var arr = np.array(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0, 12.0 }
        })[":, ::-2"];

        var result = np.var(arr, axis: 0, keepdims: true);

        result.Should().BeShaped(1, 2);
        result.Should().BeOfValuesApproximately(Tolerance, 10.666666666666666, 10.666666666666666);
    }

    [Test]
    public void Std_ReversedStrideView_Axis1_MatchesNumPy()
    {
        // NumPy 2.4.2:
        // >>> a = np.arange(1., 13.).reshape(3, 4)[:, ::-2]
        // >>> np.std(a, axis=1)
        // array([1., 1., 1.])
        var arr = np.array(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0, 12.0 }
        })[":, ::-2"];

        var result = np.std(arr, axis: 1);

        result.Should().BeShaped(3);
        result.Should().BeOfValuesApproximately(Tolerance, 1.0, 1.0, 1.0);
    }
}
