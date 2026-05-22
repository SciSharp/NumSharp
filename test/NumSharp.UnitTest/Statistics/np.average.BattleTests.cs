using System;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Statistics;

/// <summary>
/// Battle tests for np.average, comparing against NumPy 2.4.2 reference values.
/// Covers unweighted, weighted (1D/N-D weights), tuple-axis, keepdims, returned=True,
/// dtype promotion (NEP50), shape-validation errors, ZeroDivisionError, and view layouts.
/// </summary>
[TestClass]
public class np_average_BattleTests
{
    private static double At(NDArray nd, int i) => Convert.ToDouble(nd.GetAtIndex(i));

    // ── unweighted basics (= np.mean) ──────────────────────────────────

    [TestMethod]
    public void Average_1D_Unweighted_MatchesMean()
    {
        // NumPy: np.average(np.arange(1, 5)) -> 2.5
        var a = np.arange(1, 5);
        var r = np.average(a);
        r.shape.Should().Equal(Array.Empty<long>());
        At(r, 0).Should().Be(2.5);
        r.typecode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Average_2D_AxisNone_Unweighted()
    {
        // NumPy: np.average(np.arange(6).reshape(3,2)) -> 2.5
        var a = np.arange(6).reshape(3, 2);
        var r = np.average(a);
        At(r, 0).Should().Be(2.5);
    }

    [TestMethod]
    public void Average_2D_Axis0_Unweighted()
    {
        // NumPy: np.average(np.arange(6).reshape(3,2), axis=0) -> [2., 3.]
        var a = np.arange(6).reshape(3, 2);
        var r = np.average(a, axis: 0);
        r.shape.Should().Equal(new long[] { 2 });
        At(r, 0).Should().Be(2.0);
        At(r, 1).Should().Be(3.0);
    }

    [TestMethod]
    public void Average_2D_Axis1_Unweighted()
    {
        // NumPy: np.average(np.arange(6).reshape(3,2), axis=1) -> [0.5, 2.5, 4.5]
        var a = np.arange(6).reshape(3, 2);
        var r = np.average(a, axis: 1);
        r.shape.Should().Equal(new long[] { 3 });
        At(r, 0).Should().Be(0.5);
        At(r, 1).Should().Be(2.5);
        At(r, 2).Should().Be(4.5);
    }

    [TestMethod]
    public void Average_Keepdims_PreservesShape()
    {
        // NumPy: np.average(np.arange(6).reshape(3,2), axis=1, keepdims=True).shape -> (3,1)
        var a = np.arange(6).reshape(3, 2);
        var r = np.average(a, axis: 1, keepdims: true);
        r.shape.Should().Equal(new long[] { 3, 1 });
        At(r, 0).Should().Be(0.5);
        At(r, 2).Should().Be(4.5);
    }

    [TestMethod]
    public void Average_AxisNone_KeepdimsTrue()
    {
        // NumPy: np.average(arr, keepdims=True).shape -> (1,1)
        var a = np.arange(6).reshape(3, 2);
        var r = np.average(a, axis: (int?)null, keepdims: true);
        r.shape.Should().Equal(new long[] { 1, 1 });
        At(r, 0).Should().Be(2.5);
    }

    [TestMethod]
    public void Average_NegativeAxis_MatchesPositive()
    {
        var a = np.arange(24).reshape(2, 3, 4).astype(NPTypeCode.Double);
        var r1 = np.average(a, axis: -1);
        var r2 = np.average(a, axis: 2);
        r1.shape.Should().Equal(r2.shape);
        for (int i = 0; i < (int)r1.size; i++) At(r1, i).Should().Be(At(r2, i));
    }

    // ── weighted basics ────────────────────────────────────────────────

    [TestMethod]
    public void Average_Weighted_1D_DocExample()
    {
        // NumPy: np.average(np.arange(1, 11), weights=np.arange(10, 0, -1)) -> 4.0
        var a = np.arange(1, 11);
        var w = np.arange(10, 0, -1);
        var r = np.average(a, weights: w);
        At(r, 0).Should().Be(4.0);
    }

    [TestMethod]
    public void Average_Weighted_2D_Axis1_DocExample()
    {
        // NumPy: np.average(np.arange(6).reshape(3,2), axis=1, weights=[0.25, 0.75])
        //         -> [0.75, 2.75, 4.75]
        var a = np.arange(6).reshape(3, 2);
        var r = np.average(a, axis: 1, weights: np.array(new[] { 0.25, 0.75 }));
        r.shape.Should().Equal(new long[] { 3 });
        At(r, 0).Should().BeApproximately(0.75, 1e-12);
        At(r, 1).Should().BeApproximately(2.75, 1e-12);
        At(r, 2).Should().BeApproximately(4.75, 1e-12);
    }

    [TestMethod]
    public void Average_Weighted_2D_Axis0()
    {
        // NumPy: np.average(np.arange(6).reshape(3,2), axis=0, weights=[1.,2.,3.])
        //         -> [8/6, 11/6] = [2.6666..., 3.6666...]
        var a = np.arange(6).reshape(3, 2);
        var r = np.average(a, axis: 0, weights: np.array(new[] { 1.0, 2.0, 3.0 }));
        r.shape.Should().Equal(new long[] { 2 });
        At(r, 0).Should().BeApproximately(8.0 / 3.0, 1e-12);
        At(r, 1).Should().BeApproximately(11.0 / 3.0, 1e-12);
    }

    [TestMethod]
    public void Average_Weighted_TupleAxis_ND_Weights()
    {
        // NumPy: data=np.arange(8).reshape(2,2,2); w=[[0.25,0.75],[1.,0.5]]
        //   np.average(data, axis=(0,1), weights=w) -> [3.4, 4.4]
        var data = np.arange(8).reshape(2, 2, 2);
        var w = np.array(new[,] { { 0.25, 0.75 }, { 1.0, 0.5 } });
        var r = np.average(data, axis: new[] { 0, 1 }, weights: w);
        r.shape.Should().Equal(new long[] { 2 });
        At(r, 0).Should().BeApproximately(3.4, 1e-12);
        At(r, 1).Should().BeApproximately(4.4, 1e-12);
    }

    [TestMethod]
    public void Average_Weighted_FullShapeMatch_NoAxis()
    {
        // NumPy: weights with full shape match -> reduces all axes (axis=None implied).
        // data shape (3,2), W shape (3,2), expected scalar = sum(data*W)/sum(W)
        var data = np.arange(6).reshape(3, 2).astype(NPTypeCode.Double);
        var W = np.array(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } });
        var r = np.average(data, weights: W);
        // Sum(data*W)=0+2+6+12+20+30=70 ; Sum(W)=21 ; avg=70/21=10/3
        At(r, 0).Should().BeApproximately(10.0 / 3.0, 1e-12);
    }

    // ── returned=True ──────────────────────────────────────────────────

    [TestMethod]
    public void Average_Returned_Weighted_AxisGiven()
    {
        // NumPy: returns (avg, scl); for axis=1 weights=[0.25,0.75], scl=[1,1,1]
        var data = np.arange(6).reshape(3, 2);
        var (avg, scl) = np.average_returned(data, axis: 1, weights: np.array(new[] { 0.25, 0.75 }));
        avg.shape.Should().Equal(new long[] { 3 });
        scl.shape.Should().Equal(new long[] { 3 });
        At(avg, 0).Should().BeApproximately(0.75, 1e-12);
        At(scl, 0).Should().BeApproximately(1.0, 1e-12);
        At(scl, 1).Should().BeApproximately(1.0, 1e-12);
        At(scl, 2).Should().BeApproximately(1.0, 1e-12);
    }

    [TestMethod]
    public void Average_Returned_Unweighted_AxisGiven_ScalarBroadcast()
    {
        // NumPy: scl broadcast to avg shape; for shape (3,2) axis=1, count=2 each
        var data = np.arange(6).reshape(3, 2);
        var (avg, scl) = np.average_returned(data, axis: 1);
        avg.shape.Should().Equal(new long[] { 3 });
        scl.shape.Should().Equal(new long[] { 3 });
        for (int i = 0; i < 3; i++) At(scl, i).Should().Be(2.0);
    }

    [TestMethod]
    public void Average_Returned_AxisNone_Unweighted()
    {
        // NumPy: np.average(data, returned=True) -> (2.5, 6.0)
        var data = np.arange(6).reshape(3, 2);
        var (avg, scl) = np.average_returned(data);
        At(avg, 0).Should().Be(2.5);
        At(scl, 0).Should().Be(6.0);
    }

    [TestMethod]
    public void Average_Returned_AxisNone_Weighted()
    {
        // NumPy: data=arange(6).reshape(3,2); W=[[1,2],[3,4],[5,6]] -> avg=10/3, scl=21
        var data = np.arange(6).reshape(3, 2).astype(NPTypeCode.Double);
        var W = np.array(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } });
        var (avg, scl) = np.average_returned(data, weights: W);
        At(avg, 0).Should().BeApproximately(10.0 / 3.0, 1e-12);
        At(scl, 0).Should().BeApproximately(21.0, 1e-12);
    }

    // ── dtype promotion (NEP50 alignment) ──────────────────────────────

    [TestMethod]
    public void Average_Int_Unweighted_PromotesToFloat64()
    {
        // NumPy: average(int32 array) -> float64
        var a = np.array(new[] { 1, 2, 3, 4 });
        var r = np.average(a);
        r.typecode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Average_IntInt_Weights_PromotesToFloat64()
    {
        // NumPy: average(int32, weights=int32) -> float64
        var r = np.average(np.array(new[] { 1, 2, 3, 4 }), weights: np.array(new[] { 1, 1, 1, 1 }));
        r.typecode.Should().Be(NPTypeCode.Double);
        At(r, 0).Should().Be(2.5);
    }

    [TestMethod]
    public void Average_Float32_Float32_Weights_StaysFloat32()
    {
        // NumPy: float32+float32 -> float32
        var r = np.average(np.array(new[] { 1f, 2f, 3f, 4f }), weights: np.array(new[] { 1f, 1f, 1f, 1f }));
        r.typecode.Should().Be(NPTypeCode.Single);
        At(r, 0).Should().BeApproximately(2.5, 1e-6);
    }

    [TestMethod]
    public void Average_Int32_Float32_Weights_PromotesToFloat64()
    {
        // NumPy: int+float -> float64 (cross-kind)
        var r = np.average(np.array(new[] { 1, 2, 3, 4 }), weights: np.array(new[] { 1f, 1f, 1f, 1f }));
        r.typecode.Should().Be(NPTypeCode.Double);
        At(r, 0).Should().Be(2.5);
    }

    [TestMethod]
    public void Average_Bool_Unweighted_PromotesToFloat64()
    {
        // NumPy: average(bool) -> float64 of 0.75
        var r = np.average(np.array(new[] { true, false, true, true }));
        r.typecode.Should().Be(NPTypeCode.Double);
        At(r, 0).Should().Be(0.75);
    }

    // ── error paths ────────────────────────────────────────────────────

    [TestMethod]
    public void Average_MismatchedWeights_AxisNone_ThrowsTypeError()
    {
        // NumPy: TypeError: Axis must be specified when shapes of a and weights differ.
        var data = np.arange(6).reshape(3, 2);
        Action act = () => np.average(data, weights: np.array(new[] { 0.25, 0.75 }));
        act.Should().Throw<ArgumentException>().WithMessage("*Axis must be specified*");
    }

    [TestMethod]
    public void Average_WrongLengthWeights_ThrowsValueError()
    {
        // NumPy: ValueError: Shape of weights must be consistent with shape of a along specified axis.
        var data = np.arange(6).reshape(3, 2);
        Action act = () => np.average(data, axis: 0, weights: np.array(new[] { 0.25, 0.75 }));
        act.Should().Throw<ArgumentException>().WithMessage("*Shape of weights*");
    }

    [TestMethod]
    public void Average_ZeroSumWeights_ThrowsZeroDivision()
    {
        // NumPy: ZeroDivisionError: Weights sum to zero, can't be normalized
        var a = np.array(new[] { 1.0, 2.0, 3.0 });
        var w = np.array(new[] { 1.0, -1.0, 0.0 });
        Action act = () => np.average(a, weights: w);
        act.Should().Throw<DivideByZeroException>();
    }

    [TestMethod]
    public void Average_NullArray_Throws()
    {
        Action act = () => np.average((NDArray)null);
        act.Should().Throw<ArgumentNullException>();
    }

    [TestMethod]
    public void Average_DuplicateAxis_Throws()
    {
        var a = np.arange(24).reshape(2, 3, 4).astype(NPTypeCode.Double);
        Action act = () => np.average(a, axis: new[] { 0, 0 });
        act.Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void Average_AxisOutOfBounds_Throws()
    {
        var a = np.arange(6).reshape(3, 2);
        Action act = () => np.average(a, axis: 5);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [TestMethod]
    public void Average_TupleAxis_MismatchedNDWeights_Throws()
    {
        // NumPy: ValueError when w is 2D but axis is single 1D
        var data = np.arange(8).reshape(2, 2, 2);
        var w = np.array(new[,] { { 0.25, 0.75 }, { 1.0, 0.5 } });
        Action act = () => np.average(data, axis: new[] { 0 }, weights: w);
        act.Should().Throw<ArgumentException>().WithMessage("*Shape of weights*");
    }

    // ── view-layout variations ──────────────────────────────────────────

    [TestMethod]
    public void Average_Transposed_MatchesAxisSwap()
    {
        // NumPy: np.average(a.T, axis=0) == np.average(a, axis=1) for 2D
        var a = np.arange(20).reshape(4, 5).astype(NPTypeCode.Double);
        var r1 = np.average(a.T, axis: 0);
        var r2 = np.average(a, axis: 1);
        r1.shape.Should().Equal(r2.shape);
        for (int i = 0; i < (int)r1.size; i++)
            At(r1, i).Should().BeApproximately(At(r2, i), 1e-12);
    }

    [TestMethod]
    public void Average_StridedView_Axis1()
    {
        // NumPy: np.average(np.arange(20).reshape(4,5)[:, ::2], axis=1) -> [2, 7, 12, 17]
        var a = np.arange(20).reshape(4, 5).astype(NPTypeCode.Double);
        var s = a[":, ::2"];
        var r = np.average(s, axis: 1);
        r.shape.Should().Equal(new long[] { 4 });
        At(r, 0).Should().Be(2.0);
        At(r, 1).Should().Be(7.0);
        At(r, 2).Should().Be(12.0);
        At(r, 3).Should().Be(17.0);
    }

    [TestMethod]
    public void Average_ZeroD_Scalar()
    {
        // NumPy: np.average(np.array(5.0)) -> 5.0
        var s = NDArray.Scalar(5.0);
        var r = np.average(s);
        At(r, 0).Should().Be(5.0);
    }

    [TestMethod]
    public void Average_TupleSingleElement_EqualsScalarAxis()
    {
        // NumPy: np.average(data, axis=(1,)) == np.average(data, axis=1)
        var data = np.arange(6).reshape(3, 2).astype(NPTypeCode.Double);
        var r1 = np.average(data, axis: new[] { 1 });
        var r2 = np.average(data, axis: 1);
        r1.shape.Should().Equal(r2.shape);
        for (int i = 0; i < 3; i++) At(r1, i).Should().Be(At(r2, i));
    }

    [TestMethod]
    public void Average_TupleAxis_Keepdims()
    {
        // NumPy: np.average(data3, axis=(0,1), keepdims=True).shape -> (1,1,2)
        var data3 = np.arange(8).reshape(2, 2, 2).astype(NPTypeCode.Double);
        var r = np.average(data3, axis: new[] { 0, 1 }, keepdims: true);
        r.shape.Should().Equal(new long[] { 1, 1, 2 });
    }

    [TestMethod]
    public void Average_HigherRank_5D()
    {
        // NumPy: np.average(np.arange(72).reshape(2,3,2,3,2)) -> 35.5
        var hi = np.arange(2 * 3 * 2 * 3 * 2).reshape(2, 3, 2, 3, 2);
        var r = np.average(hi);
        At(r, 0).Should().BeApproximately(35.5, 1e-12);
    }

    // ── post-audit additions ────────────────────────────────────────────

    [TestMethod]
    public void Average_Half_Weighted_PreservesDtype()
    {
        // NumPy: np.average(np.array([1,2,3,4], dtype=np.float16), weights=same) -> 2.5 float16
        var a = np.array(new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 });
        var w = np.array(new Half[] { (Half)1, (Half)1, (Half)1, (Half)1 });
        var r = np.average(a, weights: w);
        r.typecode.Should().Be(NPTypeCode.Half);
        ((float)r.GetAtIndex<Half>(0)).Should().BeApproximately(2.5f, 0.01f);
    }

    [TestMethod]
    public void Average_Complex_Weighted_PreservesDtype()
    {
        // NumPy: np.average(complex array, weights=complex) -> Complex preserving
        var a = np.array(new System.Numerics.Complex[] { new(1, 0), new(2, 1), new(3, -1) });
        var w = np.array(new System.Numerics.Complex[] { new(1, 0), new(1, 0), new(1, 0) });
        var r = np.average(a, weights: w);
        r.typecode.Should().Be(NPTypeCode.Complex);
        var v = r.GetAtIndex<System.Numerics.Complex>(0);
        v.Real.Should().Be(2.0);
        v.Imaginary.Should().Be(0.0);
    }

    [TestMethod]
    public void Average_Complex_ZeroSumWeights_Throws()
    {
        // NumPy: complex zero-sum weights -> ZeroDivisionError
        var a = np.array(new System.Numerics.Complex[] { new(1, 0), new(2, 0), new(3, 0) });
        var w = np.array(new System.Numerics.Complex[] { new(1, 0), new(-1, 0), new(0, 0) });
        Action act = () => np.average(a, weights: w);
        act.Should().Throw<DivideByZeroException>();
    }

    [TestMethod]
    public void Average_Empty1D_Unweighted_ReturnsNaN()
    {
        // NumPy: np.average(np.array([])) -> nan (mean returns 0-D scalar, scl=0/1=0)
        var a = np.array(new double[0]);
        var r = np.average(a);
        double.IsNaN(At(r, 0)).Should().BeTrue();
    }

    [TestMethod]
    public void Average_Empty2D_AxisNone_ReturnsNaN()
    {
        // NumPy: np.average(np.zeros((0,3))) -> nan
        var a = np.zeros(new Shape(0, 3));
        var r = np.average(a);
        double.IsNaN(At(r, 0)).Should().BeTrue();
    }

    [TestMethod]
    public void Average_Empty2D_Axis0_ReturnsNaNArray()
    {
        // NumPy: np.average(np.zeros((0,3)), axis=0) -> [nan, nan, nan]
        var a = np.zeros(new Shape(0, 3));
        var r = np.average(a, axis: 0);
        r.shape.Should().Equal(new long[] { 3 });
        for (int i = 0; i < 3; i++) double.IsNaN(At(r, i)).Should().BeTrue();
    }

    [TestMethod]
    public void Average_Empty2D_Axis1_RaisesZeroDivision()
    {
        // NumPy: np.average(np.zeros((0,3)), axis=1) raises ZeroDivisionError
        // because avg.size == 0 and scl = a.size / avg.size = 0/0
        var a = np.zeros(new Shape(0, 3));
        Action act = () => np.average(a, axis: 1);
        act.Should().Throw<DivideByZeroException>();
    }

    [TestMethod]
    public void Average_BroadcastedView_AxisReductions()
    {
        // NumPy: broadcast_to (3,1) -> (3,4) average reductions match expected
        var b = np.broadcast_to(np.arange(3).reshape(3, 1), new Shape(3, 4));
        At(np.average(b), 0).Should().Be(1.0);
        var ax0 = np.average(b, axis: 0);
        ax0.shape.Should().Equal(new long[] { 4 });
        for (int i = 0; i < 4; i++) At(ax0, i).Should().Be(1.0);
        var ax1 = np.average(b, axis: 1);
        ax1.shape.Should().Equal(new long[] { 3 });
        for (int i = 0; i < 3; i++) At(ax1, i).Should().Be(i);
    }

    [TestMethod]
    public void Average_NewAxis_PreservesSingletonDim()
    {
        // NumPy: np.average(a[None,:,:]) handles size-1 inserted axis
        var a = np.arange(6).reshape(3, 2)[np.newaxis];
        a.shape.Should().Equal(new long[] { 1, 3, 2 });
        At(np.average(a), 0).Should().Be(2.5);
        var ax1 = np.average(a, axis: 1);
        ax1.shape.Should().Equal(new long[] { 1, 2 });
        At(ax1, 0).Should().Be(2.0);
        At(ax1, 1).Should().Be(3.0);
    }

    [TestMethod]
    public void Average_SingletonDim_ReductionMatches()
    {
        // NumPy: shape (3,1) axis=0 -> [1.], axis=1 -> [0., 1., 2.]
        var a = np.arange(3).reshape(3, 1).astype(NPTypeCode.Double);
        var ax0 = np.average(a, axis: 0);
        ax0.shape.Should().Equal(new long[] { 1 });
        At(ax0, 0).Should().Be(1.0);

        var ax1 = np.average(a, axis: 1);
        ax1.shape.Should().Equal(new long[] { 3 });
        At(ax1, 0).Should().Be(0.0);
        At(ax1, 1).Should().Be(1.0);
        At(ax1, 2).Should().Be(2.0);
    }

    [TestMethod]
    public void Average_NegativeStrideView()
    {
        // NumPy: np.average(np.arange(20).reshape(4,5)[::-1]) -> 9.5 (same as forward)
        var a = np.arange(20).reshape(4, 5);
        var rev = a["::-1"];
        At(np.average(rev), 0).Should().Be(9.5);
        var ax0 = np.average(rev, axis: 0);
        ax0.shape.Should().Equal(new long[] { 5 });
        for (int i = 0; i < 5; i++) At(ax0, i).Should().BeApproximately(7.5 + i, 1e-12);
    }

    [TestMethod]
    public void Average_FContiguous_AxisReductions()
    {
        // F-contiguous view via transpose-then-transpose-back
        // F-layout (3,2) reduced along axis=0 should match (3,2) reduced along axis=0
        var c = np.arange(6).reshape(2, 3).astype(NPTypeCode.Double);  // C-contig (2,3)
        var f = c.T;  // shape (3,2), F-contiguous (stride[0]==1)
        f.shape.Should().Equal(new long[] { 3, 2 });
        var ax0 = np.average(f, axis: 0);
        ax0.shape.Should().Equal(new long[] { 2 });
        // f[:,0]=[0,1,2], f[:,1]=[3,4,5] -> means 1, 4
        At(ax0, 0).Should().Be(1.0);
        At(ax0, 1).Should().Be(4.0);
    }

    [TestMethod]
    public void Average_Returned_FullShape_Unweighted_AxisNone()
    {
        // NumPy: returned axis=None unweighted -> avg=scalar, scl=scalar count
        var data = np.arange(6).reshape(3, 2);
        var (avg, scl) = np.average_returned(data);
        avg.shape.Should().Equal(Array.Empty<long>());
        scl.shape.Should().Equal(Array.Empty<long>());
        At(avg, 0).Should().Be(2.5);
        At(scl, 0).Should().Be(6.0);
    }

    [TestMethod]
    public void Average_Returned_KeepdimsTrue_AxisNone()
    {
        // NumPy: returned, axis=None, keepdims=True -> shapes (1,1) both
        var data = np.arange(6).reshape(3, 2);
        var (avg, scl) = np.average_returned(data, axis: (int?)null, keepdims: true);
        avg.shape.Should().Equal(new long[] { 1, 1 });
        scl.shape.Should().Equal(new long[] { 1, 1 });
        At(avg, 0).Should().Be(2.5);
        At(scl, 0).Should().Be(6.0);
    }

    [TestMethod]
    public void Average_Float16Result_FromFloat16Inputs()
    {
        // NumPy: float16 + float16 weights -> float16 result
        var a = np.array(new[] { (Half)1, (Half)2, (Half)3, (Half)4 });
        var w = np.array(new[] { (Half)1, (Half)2, (Half)3, (Half)4 });
        var r = np.average(a, weights: w);
        r.typecode.Should().Be(NPTypeCode.Half);
        // sum(1*1+2*2+3*3+4*4)/sum(1+2+3+4) = 30/10 = 3.0
        ((float)r.GetAtIndex<Half>(0)).Should().BeApproximately(3.0f, 0.01f);
    }
}
