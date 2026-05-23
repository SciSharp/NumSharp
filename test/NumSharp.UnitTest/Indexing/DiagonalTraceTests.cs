using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Indexing;

/// <summary>
/// Tests for <c>np.diagonal</c> and <c>np.trace</c>. diagonal is a view-only
/// operation (stride trick: combine stride[axis1] + stride[axis2]); trace
/// composes diagonal + sum over the appended diagonal axis with NumPy's
/// integer-promotion-to-int64 rule for narrow integer sources.
/// </summary>
[TestClass]
public class DiagonalTraceTests
{
    // =================================================================
    // np.diagonal
    // =================================================================

    [TestMethod]
    public void Diagonal_2D_Square_Int32()
    {
        var a = np.arange(9, NPTypeCode.Int32).reshape(3, 3);
        var r = np.diagonal(a);
        r.Shape.Should().Be(new Shape(3));
        r.ToArray<int>().Should().Equal(0, 4, 8);
    }

    [TestMethod]
    public void Diagonal_2D_Square_OffsetPositive()
    {
        var a = np.arange(9, NPTypeCode.Int32).reshape(3, 3);
        var r = np.diagonal(a, offset: 1);
        r.Shape.Should().Be(new Shape(2));
        r.ToArray<int>().Should().Equal(1, 5);
    }

    [TestMethod]
    public void Diagonal_2D_Square_OffsetNegative()
    {
        var a = np.arange(9, NPTypeCode.Int32).reshape(3, 3);
        var r = np.diagonal(a, offset: -1);
        r.Shape.Should().Be(new Shape(2));
        r.ToArray<int>().Should().Equal(3, 7);
    }

    [TestMethod]
    public void Diagonal_2D_Square_OffsetEdgeBoundary()
    {
        var a = np.arange(9, NPTypeCode.Int32).reshape(3, 3);
        var r = np.diagonal(a, offset: 2);
        r.Shape.Should().Be(new Shape(1));
        r.GetInt32(0).Should().Be(2);
    }

    [TestMethod]
    public void Diagonal_2D_Square_OffsetOutOfRange_EmptyResult()
    {
        var a = np.arange(9, NPTypeCode.Int32).reshape(3, 3);
        var r = np.diagonal(a, offset: 3);
        r.size.Should().Be(0);
        r.Shape.Should().Be(new Shape(0));
    }

    [TestMethod]
    public void Diagonal_2D_Rectangular_3x4()
    {
        var a = np.arange(12, NPTypeCode.Int32).reshape(3, 4);
        var r = np.diagonal(a);
        r.ToArray<int>().Should().Equal(0, 5, 10);
    }

    [TestMethod]
    public void Diagonal_2D_Rectangular_OffsetPositive_CoversFullDim()
    {
        // 3x4 with offset=1 → diag along (0,1)(1,2)(2,3) → 3 elements.
        var a = np.arange(12, NPTypeCode.Int32).reshape(3, 4);
        var r = np.diagonal(a, offset: 1);
        r.ToArray<int>().Should().Equal(1, 6, 11);
    }

    [TestMethod]
    public void Diagonal_1D_Source_Raises()
    {
        var action = () => np.diagonal(np.arange(5));
        action.Should().Throw<ArgumentException>()
            .WithMessage("*at least two dimensions*");
    }

    [TestMethod]
    public void Diagonal_0D_Source_Raises()
    {
        var action = () => np.diagonal(NDArray.Scalar(5));
        action.Should().Throw<ArgumentException>()
            .WithMessage("*at least two dimensions*");
    }

    [TestMethod]
    public void Diagonal_3D_DefaultAxes_ShapeAndValues()
    {
        // shape (2,3,4) with axis1=0, axis2=1: diag along (i,i) over the
        // 2 minor dim → diag_size=min(2,3)=2; the remaining axis (size 4)
        // becomes the leading dim and the diagonal axis is appended →
        // result shape (4, 2).
        var c = np.arange(24, NPTypeCode.Int32).reshape(2, 3, 4);
        var r = np.diagonal(c);
        r.Shape.Should().Be(new Shape(4, 2));
        np.ravel(r).ToArray<int>().Should().Equal(0, 16, 1, 17, 2, 18, 3, 19);
    }

    [TestMethod]
    public void Diagonal_3D_Axis12()
    {
        // shape (2,3,4) with axis1=1, axis2=2: diag_size=min(3,4)=3 along the
        // (axis1, axis2) plane → result shape (2, 3) with values =
        // [[0,5,10], [12,17,22]].
        var c = np.arange(24, NPTypeCode.Int32).reshape(2, 3, 4);
        var r = np.diagonal(c, axis1: 1, axis2: 2);
        r.Shape.Should().Be(new Shape(2, 3));
        np.ravel(r).ToArray<int>().Should().Equal(0, 5, 10, 12, 17, 22);
    }

    [TestMethod]
    public void Diagonal_3D_NegativeAxes()
    {
        var c = np.arange(24, NPTypeCode.Int32).reshape(2, 3, 4);
        var r = np.diagonal(c, axis1: -2, axis2: -1);
        r.Shape.Should().Be(new Shape(2, 3));
        np.ravel(r).ToArray<int>().Should().Equal(0, 5, 10, 12, 17, 22);
    }

    [TestMethod]
    public void Diagonal_Axis1EqualsAxis2_Raises()
    {
        var c = np.arange(24, NPTypeCode.Int32).reshape(2, 3, 4);
        var action = () => np.diagonal(c, axis1: 0, axis2: 0);
        action.Should().Throw<ArgumentException>()
            .WithMessage("*axis1 and axis2 cannot be the same*");
    }

    [TestMethod]
    public void Diagonal_AxisOutOfRange_Raises()
    {
        var c = np.arange(24, NPTypeCode.Int32).reshape(2, 3, 4);
        var action = () => np.diagonal(c, axis1: 5);
        action.Should().Throw<ArgumentOutOfRangeException>();
    }

    [TestMethod]
    public void Diagonal_ReturnsView_SharesMemoryWithSource()
    {
        var a = np.arange(9, NPTypeCode.Int32).reshape(3, 3).copy();
        var d = np.diagonal(a);
        a.SetInt32(99, 0, 0);
        d.GetInt32(0).Should().Be(99);
    }

    [TestMethod]
    public void Diagonal_View_IsReadOnly()
    {
        var a = np.arange(9, NPTypeCode.Int32).reshape(3, 3);
        var d = np.diagonal(a);
        d.Shape.IsWriteable.Should().BeFalse();
    }

    [TestMethod]
    public void Diagonal_EmptyAxis_ZeroSizeResult()
    {
        var a = np.zeros(new Shape(0, 3), NPTypeCode.Double);
        var r = np.diagonal(a);
        r.Shape.Should().Be(new Shape(0));
    }

    [TestMethod]
    public void Diagonal_AllDtypes_Smoke()
    {
        // Every dtype walks the same stride-view path; one round-trip each.
        var a3x3 = (Func<NPTypeCode, NDArray>)(tc => np.arange(9, tc).reshape(3, 3));

        np.diagonal(a3x3(NPTypeCode.Boolean)).Shape.Should().Be(new Shape(3));
        np.diagonal(a3x3(NPTypeCode.Byte)).Shape.Should().Be(new Shape(3));
        np.diagonal(a3x3(NPTypeCode.SByte)).Shape.Should().Be(new Shape(3));
        np.diagonal(a3x3(NPTypeCode.Int16)).Shape.Should().Be(new Shape(3));
        np.diagonal(a3x3(NPTypeCode.UInt16)).Shape.Should().Be(new Shape(3));
        np.diagonal(a3x3(NPTypeCode.Int32)).Shape.Should().Be(new Shape(3));
        np.diagonal(a3x3(NPTypeCode.UInt32)).Shape.Should().Be(new Shape(3));
        np.diagonal(a3x3(NPTypeCode.Int64)).Shape.Should().Be(new Shape(3));
        np.diagonal(a3x3(NPTypeCode.UInt64)).Shape.Should().Be(new Shape(3));
        np.diagonal(a3x3(NPTypeCode.Char)).Shape.Should().Be(new Shape(3));
        np.diagonal(a3x3(NPTypeCode.Half)).Shape.Should().Be(new Shape(3));
        np.diagonal(a3x3(NPTypeCode.Single)).Shape.Should().Be(new Shape(3));
        np.diagonal(a3x3(NPTypeCode.Double)).Shape.Should().Be(new Shape(3));
        np.diagonal(a3x3(NPTypeCode.Decimal)).Shape.Should().Be(new Shape(3));

        // Complex constructed differently — no implicit int conversion.
        var cm = np.array(new Complex[,] {
            { new(1,2), new(3,4), new(5,6) },
            { new(7,8), new(9,10), new(11,12) },
            { new(13,14), new(15,16), new(17,18) } });
        np.diagonal(cm).Shape.Should().Be(new Shape(3));
    }

    [TestMethod]
    public void Diagonal_NullArg_Throws()
    {
        ((Action)(() => np.diagonal(null))).Should().Throw<ArgumentNullException>();
    }

    // =================================================================
    // np.trace
    // =================================================================

    [TestMethod]
    public void Trace_2D_Square_Int32_PromotesToInt64()
    {
        var a = np.arange(9, NPTypeCode.Int32).reshape(3, 3);
        var r = np.trace(a);
        r.dtype.Should().Be(typeof(long));
        r.GetInt64(0).Should().Be(12L); // 0+4+8
    }

    [TestMethod]
    public void Trace_2D_Square_Double_PreservesDtype()
    {
        var a = np.eye(3, dtype: typeof(double));
        var r = np.trace(a);
        r.dtype.Should().Be(typeof(double));
        r.GetDouble(0).Should().Be(3.0);
    }

    [TestMethod]
    public void Trace_Int8_PromotesToInt64()
    {
        var a = np.arange(9, NPTypeCode.SByte).reshape(3, 3);
        var r = np.trace(a);
        r.dtype.Should().Be(typeof(long));
        r.GetInt64(0).Should().Be(12L);
    }

    [TestMethod]
    public void Trace_OffsetPositive()
    {
        var a = np.arange(9, NPTypeCode.Int32).reshape(3, 3);
        np.trace(a, offset: 1).GetInt64(0).Should().Be(6L); // 1+5
    }

    [TestMethod]
    public void Trace_OffsetNegative()
    {
        var a = np.arange(9, NPTypeCode.Int32).reshape(3, 3);
        np.trace(a, offset: -1).GetInt64(0).Should().Be(10L); // 3+7
    }

    [TestMethod]
    public void Trace_3D_DefaultAxes()
    {
        // shape (2,3,4); diagonal of (axis=0,axis=1) → shape (4,2); sum axis=-1
        // → shape (4,) with values [0+16, 1+17, 2+18, 3+19] = [16, 18, 20, 22].
        var c = np.arange(24, NPTypeCode.Int32).reshape(2, 3, 4);
        var r = np.trace(c);
        r.Shape.Should().Be(new Shape(4));
        r.ToArray<long>().Should().Equal(16L, 18L, 20L, 22L);
    }

    [TestMethod]
    public void Trace_3D_Axis12()
    {
        // axis1=1,axis2=2; diag shape (2,3); sum axis=-1 → shape (2,)
        // values [0+5+10, 12+17+22] = [15, 51].
        var c = np.arange(24, NPTypeCode.Int32).reshape(2, 3, 4);
        var r = np.trace(c, axis1: 1, axis2: 2);
        r.Shape.Should().Be(new Shape(2));
        r.ToArray<long>().Should().Equal(15L, 51L);
    }

    [TestMethod]
    public void Trace_ExplicitDtype_Double()
    {
        var a = np.arange(9, NPTypeCode.Int32).reshape(3, 3);
        var r = np.trace(a, dtype: typeof(double));
        r.dtype.Should().Be(typeof(double));
        r.GetDouble(0).Should().Be(12.0);
    }

    [TestMethod]
    public void Trace_OutDispatch_ReturnsOut()
    {
        var c = np.arange(24, NPTypeCode.Int32).reshape(2, 3, 4);
        var outArr = np.zeros(new Shape(4), NPTypeCode.Int64);
        var r = np.trace(c, @out: outArr);
        ReferenceEquals(r, outArr).Should().BeTrue();
        r.ToArray<long>().Should().Equal(16L, 18L, 20L, 22L);
    }

    [TestMethod]
    public void Trace_OutDispatch_WrongShape_Raises()
    {
        var c = np.arange(24, NPTypeCode.Int32).reshape(2, 3, 4);
        var outArr = np.zeros(new Shape(5), NPTypeCode.Int64);
        var action = () => np.trace(c, @out: outArr);
        action.Should().Throw<ArgumentException>().WithMessage("*output array does not match*");
    }

    [TestMethod]
    public void Trace_1D_Source_Raises()
    {
        var action = () => np.trace(np.arange(5));
        action.Should().Throw<ArgumentException>()
            .WithMessage("*at least two dimensions*");
    }

    [TestMethod]
    public void Trace_EmptyAxis_ReturnsZeroOfDtype()
    {
        var a = np.zeros(new Shape(0, 3), NPTypeCode.Double);
        var r = np.trace(a);
        r.GetDouble(0).Should().Be(0.0);
    }

    [TestMethod]
    public void Trace_NaN_PropagatesIntoSum()
    {
        var a = np.array(new double[,] { { double.NaN, 1 }, { 2, 3.0 } });
        var r = np.trace(a);
        double.IsNaN(r.GetDouble(0)).Should().BeTrue();
    }

    [TestMethod]
    public void Trace_Complex_SumsDiagonal()
    {
        var a = np.array(new Complex[,] { { new(1, 2), new(3, 0) }, { new(4, 0), new(5, 6) } });
        var r = np.trace(a);
        // diag = [(1+2j), (5+6j)] sum = (6+8j)
        var v = (Complex)r.GetValue(0);
        v.Real.Should().Be(6);
        v.Imaginary.Should().Be(8);
    }

    [TestMethod]
    public void Trace_Boolean_PromotesToInt64()
    {
        // eye-like bool: diag=True,True,True → sum=3 as int64.
        var bools = new bool[3, 3];
        bools[0, 0] = true; bools[1, 1] = true; bools[2, 2] = true;
        var a = np.array(bools);
        var r = np.trace(a);
        r.dtype.Should().Be(typeof(long));
        r.GetInt64(0).Should().Be(3L);
    }

    [TestMethod]
    public void Trace_NullArg_Throws()
    {
        ((Action)(() => np.trace(null))).Should().Throw<ArgumentNullException>();
    }

    [TestMethod]
    public void Trace_TransposedSource_StillCorrect()
    {
        // T view of (3,4) is (4,3) non-contig; trace should work via the diagonal
        // view's stride trick which already accounts for non-unit strides.
        var src = np.arange(12, NPTypeCode.Int32).reshape(3, 4).T; // (4,3)
        var r = np.trace(src);
        r.dtype.Should().Be(typeof(long));
        // diag of (4,3) = (0,0), (1,1), (2,2) → src.T[0,0]=src[0,0]=0,
        // src.T[1,1]=src[1,1]=5, src.T[2,2]=src[2,2]=10
        r.GetInt64(0).Should().Be(15L); // 0+5+10
    }

    [TestMethod]
    public void Trace_NegativeStrideSource_StillCorrect()
    {
        // [::-1] on rows reverses the row order. trace walks the strided diag.
        var src = np.arange(9, NPTypeCode.Int32).reshape(3, 3)["::-1"];
        // src now = [[6,7,8],[3,4,5],[0,1,2]]; main diag = [6, 4, 2] → sum = 12.
        var r = np.trace(src);
        r.GetInt64(0).Should().Be(12L);
    }
}
