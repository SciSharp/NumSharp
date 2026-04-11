using System;
using System.Threading.Tasks;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.isreal, np.iscomplex, np.isrealobj, np.iscomplexobj.
/// </summary>
public class NpIsRealIsComplexBattleTests
{
    #region isreal Tests

    [Test]
    public async Task IsReal_IntArray_AllTrue()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.isreal(arr);
        await Assert.That(result.GetBoolean(0)).IsTrue();
        await Assert.That(result.GetBoolean(1)).IsTrue();
        await Assert.That(result.GetBoolean(2)).IsTrue();
    }

    [Test]
    public async Task IsReal_FloatArray_AllTrue()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var result = np.isreal(arr);
        await Assert.That(result.GetBoolean(0)).IsTrue();
    }

    [Test]
    public async Task IsReal_DoubleArray_AllTrue()
    {
        var arr = np.array(new double[] { 1.0, 2.0, 3.0 });
        var result = np.isreal(arr);
        await Assert.That(result.GetBoolean(0)).IsTrue();
    }

    [Test]
    public async Task IsReal_ShapeMatches()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.isreal(arr);
        await Assert.That(result.shape).IsEquivalentTo(arr.shape);
    }

    [Test]
    public async Task IsReal_Null_Throws()
    {
        await Assert.That(() => np.isreal(null!)).ThrowsException();
    }

    #endregion

    #region iscomplex Tests

    [Test]
    public async Task IsComplex_IntArray_AllFalse()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.iscomplex(arr);
        await Assert.That(result.GetBoolean(0)).IsFalse();
        await Assert.That(result.GetBoolean(1)).IsFalse();
        await Assert.That(result.GetBoolean(2)).IsFalse();
    }

    [Test]
    public async Task IsComplex_FloatArray_AllFalse()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var result = np.iscomplex(arr);
        await Assert.That(result.GetBoolean(0)).IsFalse();
    }

    [Test]
    public async Task IsComplex_ShapeMatches()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.iscomplex(arr);
        await Assert.That(result.shape).IsEquivalentTo(arr.shape);
    }

    [Test]
    public async Task IsComplex_Null_Throws()
    {
        await Assert.That(() => np.iscomplex(null!)).ThrowsException();
    }

    #endregion

    #region isrealobj Tests

    [Test]
    public async Task IsRealObj_IntArray_True()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        await Assert.That(np.isrealobj(arr)).IsTrue();
    }

    [Test]
    public async Task IsRealObj_FloatArray_True()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f });
        await Assert.That(np.isrealobj(arr)).IsTrue();
    }

    [Test]
    public async Task IsRealObj_DoubleArray_True()
    {
        var arr = np.array(new double[] { 1.0, 2.0 });
        await Assert.That(np.isrealobj(arr)).IsTrue();
    }

    [Test]
    public async Task IsRealObj_AllTypes_True()
    {
        await Assert.That(np.isrealobj(np.array(new bool[] { true }))).IsTrue();
        await Assert.That(np.isrealobj(np.array(new byte[] { 1 }))).IsTrue();
        await Assert.That(np.isrealobj(np.array(new short[] { 1 }))).IsTrue();
        await Assert.That(np.isrealobj(np.array(new int[] { 1 }))).IsTrue();
        await Assert.That(np.isrealobj(np.array(new long[] { 1 }))).IsTrue();
    }

    [Test]
    public async Task IsRealObj_Null_Throws()
    {
        await Assert.That(() => np.isrealobj(null!)).ThrowsException();
    }

    #endregion

    #region iscomplexobj Tests

    [Test]
    public async Task IsComplexObj_IntArray_False()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        await Assert.That(np.iscomplexobj(arr)).IsFalse();
    }

    [Test]
    public async Task IsComplexObj_FloatArray_False()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f });
        await Assert.That(np.iscomplexobj(arr)).IsFalse();
    }

    [Test]
    public async Task IsComplexObj_AllRealTypes_False()
    {
        await Assert.That(np.iscomplexobj(np.array(new bool[] { true }))).IsFalse();
        await Assert.That(np.iscomplexobj(np.array(new byte[] { 1 }))).IsFalse();
        await Assert.That(np.iscomplexobj(np.array(new int[] { 1 }))).IsFalse();
        await Assert.That(np.iscomplexobj(np.array(new double[] { 1.0 }))).IsFalse();
    }

    [Test]
    public async Task IsComplexObj_Null_Throws()
    {
        await Assert.That(() => np.iscomplexobj(null!)).ThrowsException();
    }

    #endregion

    #region Various Array Shapes

    [Test]
    public async Task IsReal_EmptyArray()
    {
        var arr = np.array(new int[0]);
        var result = np.isreal(arr);
        await Assert.That(result.size).IsEqualTo(0);
    }

    [Test]
    public async Task IsComplex_EmptyArray()
    {
        var arr = np.array(new int[0]);
        var result = np.iscomplex(arr);
        await Assert.That(result.size).IsEqualTo(0);
    }

    [Test]
    public async Task IsRealObj_EmptyArray_True()
    {
        var arr = np.array(new int[0]);
        await Assert.That(np.isrealobj(arr)).IsTrue();
    }

    #endregion
}
