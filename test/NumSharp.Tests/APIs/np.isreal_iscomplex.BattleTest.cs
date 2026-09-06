using System;

namespace NumSharp.Tests.APIs;

/// <summary>
/// Battle tests for np.isreal, np.iscomplex, np.isrealobj, np.iscomplexobj.
/// </summary>
[TestClass]
public class NpIsRealIsComplexBattleTests
{
    #region isreal Tests

    [TestMethod]
    public void IsReal_IntArray_AllTrue()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.isreal(arr);
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeTrue();
        result.GetBoolean(2).Should().BeTrue();
    }

    [TestMethod]
    public void IsReal_FloatArray_AllTrue()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var result = np.isreal(arr);
        result.GetBoolean(0).Should().BeTrue();
    }

    [TestMethod]
    public void IsReal_DoubleArray_AllTrue()
    {
        var arr = np.array(new double[] { 1.0, 2.0, 3.0 });
        var result = np.isreal(arr);
        result.GetBoolean(0).Should().BeTrue();
    }

    [TestMethod]
    public void IsReal_ShapeMatches()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.isreal(arr);
        result.shape.Should().BeEquivalentTo(arr.shape);
    }

    [TestMethod]
    public void IsReal_Null_Throws()
    {
        new Action(() => np.isreal(null!)).Should().Throw<Exception>();
    }

    #endregion

    #region iscomplex Tests

    [TestMethod]
    public void IsComplex_IntArray_AllFalse()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.iscomplex(arr);
        result.GetBoolean(0).Should().BeFalse();
        result.GetBoolean(1).Should().BeFalse();
        result.GetBoolean(2).Should().BeFalse();
    }

    [TestMethod]
    public void IsComplex_FloatArray_AllFalse()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var result = np.iscomplex(arr);
        result.GetBoolean(0).Should().BeFalse();
    }

    [TestMethod]
    public void IsComplex_ShapeMatches()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.iscomplex(arr);
        result.shape.Should().BeEquivalentTo(arr.shape);
    }

    [TestMethod]
    public void IsComplex_Null_Throws()
    {
        new Action(() => np.iscomplex(null!)).Should().Throw<Exception>();
    }

    #endregion

    #region isrealobj Tests

    [TestMethod]
    public void IsRealObj_IntArray_True()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        np.isrealobj(arr).Should().BeTrue();
    }

    [TestMethod]
    public void IsRealObj_FloatArray_True()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f });
        np.isrealobj(arr).Should().BeTrue();
    }

    [TestMethod]
    public void IsRealObj_DoubleArray_True()
    {
        var arr = np.array(new double[] { 1.0, 2.0 });
        np.isrealobj(arr).Should().BeTrue();
    }

    [TestMethod]
    public void IsRealObj_AllTypes_True()
    {
        np.isrealobj(np.array(new bool[] { true })).Should().BeTrue();
        np.isrealobj(np.array(new byte[] { 1 })).Should().BeTrue();
        np.isrealobj(np.array(new short[] { 1 })).Should().BeTrue();
        np.isrealobj(np.array(new int[] { 1 })).Should().BeTrue();
        np.isrealobj(np.array(new long[] { 1 })).Should().BeTrue();
    }

    [TestMethod]
    public void IsRealObj_Null_Throws()
    {
        new Action(() => np.isrealobj(null!)).Should().Throw<Exception>();
    }

    #endregion

    #region iscomplexobj Tests

    [TestMethod]
    public void IsComplexObj_IntArray_False()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        np.iscomplexobj(arr).Should().BeFalse();
    }

    [TestMethod]
    public void IsComplexObj_FloatArray_False()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f });
        np.iscomplexobj(arr).Should().BeFalse();
    }

    [TestMethod]
    public void IsComplexObj_AllRealTypes_False()
    {
        np.iscomplexobj(np.array(new bool[] { true })).Should().BeFalse();
        np.iscomplexobj(np.array(new byte[] { 1 })).Should().BeFalse();
        np.iscomplexobj(np.array(new int[] { 1 })).Should().BeFalse();
        np.iscomplexobj(np.array(new double[] { 1.0 })).Should().BeFalse();
    }

    [TestMethod]
    public void IsComplexObj_Null_Throws()
    {
        new Action(() => np.iscomplexobj(null!)).Should().Throw<Exception>();
    }

    #endregion

    #region Various Array Shapes

    [TestMethod]
    public void IsReal_EmptyArray()
    {
        var arr = np.array(new int[0]);
        var result = np.isreal(arr);
        result.size.Should().Be(0);
    }

    [TestMethod]
    public void IsComplex_EmptyArray()
    {
        var arr = np.array(new int[0]);
        var result = np.iscomplex(arr);
        result.size.Should().Be(0);
    }

    [TestMethod]
    public void IsRealObj_EmptyArray_True()
    {
        var arr = np.array(new int[0]);
        np.isrealobj(arr).Should().BeTrue();
    }

    #endregion

    #region Complex input — VALUE-based imaginary inspection (NumPy 2.4.2)

    // isreal(x) = imag(x) == 0 ; iscomplex(x) = imag(x) != 0 (for a complex dtype).
    // np.iscomplex([1+2j, 3+0j, 0+1j, 5+0j]) -> [True, False, True, False]
    // np.isreal(...)                         -> [False, True, False, True]
    private static readonly System.Numerics.Complex[] Mixed =
        { new(1, 2), new(3, 0), new(0, 1), new(5, 0) };

    [TestMethod]
    public void IsComplex_Complex_InspectsImaginary()
    {
        var r = np.iscomplex(np.array(Mixed));
        r.GetBoolean(0).Should().BeTrue("1+2j has nonzero imaginary part");
        r.GetBoolean(1).Should().BeFalse("3+0j is real");
        r.GetBoolean(2).Should().BeTrue("0+1j has nonzero imaginary part");
        r.GetBoolean(3).Should().BeFalse("5+0j is real");
    }

    [TestMethod]
    public void IsReal_Complex_InspectsImaginary()
    {
        var r = np.isreal(np.array(Mixed));
        r.GetBoolean(0).Should().BeFalse("1+2j has nonzero imaginary part");
        r.GetBoolean(1).Should().BeTrue("3+0j is real");
        r.GetBoolean(2).Should().BeFalse("0+1j has nonzero imaginary part");
        r.GetBoolean(3).Should().BeTrue("5+0j is real");
    }

    [TestMethod]
    public void IsRealIsComplex_Complex_Specials_IeeeSemantics()
    {
        // -0.0 imag -> real (== 0 True); NaN/Inf imag -> NOT real (== 0 False, != 0 True).
        var c = np.array(new System.Numerics.Complex[]
        {
            new(1, 0.0), new(1, -0.0), new(2, double.NaN),
            new(3, double.PositiveInfinity), new(0, double.NegativeInfinity),
        });
        var ir = np.isreal(c);
        var ic = np.iscomplex(c);
        // isreal:    [ True, True, False, False, False ]
        ir.GetBoolean(0).Should().BeTrue();
        ir.GetBoolean(1).Should().BeTrue("-0.0 == 0");
        ir.GetBoolean(2).Should().BeFalse("NaN == 0 is False");
        ir.GetBoolean(3).Should().BeFalse("Inf == 0 is False");
        ir.GetBoolean(4).Should().BeFalse("-Inf == 0 is False");
        // iscomplex: [ False, False, True, True, True ]
        ic.GetBoolean(0).Should().BeFalse();
        ic.GetBoolean(1).Should().BeFalse("-0.0 != 0 is False");
        ic.GetBoolean(2).Should().BeTrue("NaN != 0 is True");
        ic.GetBoolean(3).Should().BeTrue();
        ic.GetBoolean(4).Should().BeTrue();
    }

    [TestMethod]
    public void IsRealIsComplex_Complex_0d_Scalar()
    {
        np.isreal(NDArray.Scalar(new System.Numerics.Complex(3, 0))).GetBoolean(0).Should().BeTrue();
        np.iscomplex(NDArray.Scalar(new System.Numerics.Complex(3, 0))).GetBoolean(0).Should().BeFalse();
        np.isreal(NDArray.Scalar(new System.Numerics.Complex(3, 1))).GetBoolean(0).Should().BeFalse();
        np.iscomplex(NDArray.Scalar(new System.Numerics.Complex(3, 1))).GetBoolean(0).Should().BeTrue();
    }

    [TestMethod]
    public void IsReal_StridedRealView_AllTrue_NoGarbage()
    {
        // Regression: np.ones/np.zeros were handed the view's Shape (strides+offset) and emitted
        // garbage bytes. A strided real view must be all-True (isreal) / all-False (iscomplex).
        var strided = np.arange(10)["::2"]; // stride-2 view, size 5, non-contiguous
        var ir = np.isreal(strided);
        var ic = np.iscomplex(strided);
        ir.size.Should().Be(5);
        for (int i = 0; i < 5; i++)
        {
            ir.GetBoolean(i).Should().BeTrue($"real element {i} is real");
            ic.GetBoolean(i).Should().BeFalse($"real element {i} is not complex");
        }
    }

    #endregion
}
