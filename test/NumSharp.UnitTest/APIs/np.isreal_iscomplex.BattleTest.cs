using System;

namespace NumSharp.UnitTest.APIs;

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
}
