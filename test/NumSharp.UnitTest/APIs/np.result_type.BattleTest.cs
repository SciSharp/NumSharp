using System;
using System.Threading.Tasks;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.result_type - comprehensive coverage of type promotion.
/// </summary>
public class NpResultTypeBattleTests
{
    #region Single Type

    [TestMethod]
    public async Task ResultType_SingleType_ReturnsSame()
    {
        await Assert.That(np.result_type(NPTypeCode.Int32)).IsEqualTo(NPTypeCode.Int32);
        await Assert.That(np.result_type(NPTypeCode.Double)).IsEqualTo(NPTypeCode.Double);
    }

    #endregion

    #region Two Types - Integer Promotion

    [TestMethod]
    public async Task ResultType_Int32Int64_ReturnsInt64()
    {
        await Assert.That(np.result_type(NPTypeCode.Int32, NPTypeCode.Int64)).IsEqualTo(NPTypeCode.Int64);
    }

    [TestMethod]
    public async Task ResultType_Int16Int32_ReturnsInt32()
    {
        await Assert.That(np.result_type(NPTypeCode.Int16, NPTypeCode.Int32)).IsEqualTo(NPTypeCode.Int32);
    }

    [TestMethod]
    public async Task ResultType_SignedUnsigned_PromotesToContainBoth()
    {
        var result = np.result_type(NPTypeCode.UInt32, NPTypeCode.Int32);
        await Assert.That(result).IsEqualTo(NPTypeCode.Int64);
    }

    #endregion

    #region Two Types - Float Promotion

    [TestMethod]
    public async Task ResultType_Float32Float64_ReturnsFloat64()
    {
        await Assert.That(np.result_type(NPTypeCode.Single, NPTypeCode.Double)).IsEqualTo(NPTypeCode.Double);
    }

    #endregion

    #region Error Cases

    [TestMethod]
    public async Task ResultType_Empty_Throws()
    {
        await Assert.That(() => np.result_type(Array.Empty<NPTypeCode>())).ThrowsException();
    }

    [TestMethod]
    public async Task ResultType_NullArray_Throws()
    {
        await Assert.That(() => np.result_type((NPTypeCode[])null!)).ThrowsException();
    }

    #endregion

    #region Two-Arg Convenience Overloads

    [TestMethod]
    public async Task ResultType_TwoArg_NPTypeCode()
    {
        await Assert.That(np.result_type(NPTypeCode.Int32, NPTypeCode.Int64)).IsEqualTo(NPTypeCode.Int64);
    }

    [TestMethod]
    public async Task ResultType_TwoArg_Type()
    {
        await Assert.That(np.result_type(typeof(int), typeof(long))).IsEqualTo(NPTypeCode.Int64);
    }

    [TestMethod]
    public async Task ResultType_TwoArg_NDArray()
    {
        var a = np.array(new int[] { 1, 2 });
        var b = np.array(new long[] { 1, 2 });
        await Assert.That(np.result_type(a, b)).IsEqualTo(NPTypeCode.Int64);
    }

    [TestMethod]
    public async Task ResultType_TwoArg_NDArray_NullFirst_Throws()
    {
        var b = np.array(new int[] { 1, 2 });
        await Assert.That(() => np.result_type((NDArray)null!, b)).ThrowsException();
    }

    [TestMethod]
    public async Task ResultType_TwoArg_NDArray_NullSecond_Throws()
    {
        var a = np.array(new int[] { 1, 2 });
        await Assert.That(() => np.result_type(a, (NDArray)null!)).ThrowsException();
    }

    #endregion

    #region Symmetry Property

    [TestMethod]
    public async Task ResultType_Symmetric()
    {
        await Assert.That(np.result_type(NPTypeCode.Int32, NPTypeCode.Int64))
            .IsEqualTo(np.result_type(NPTypeCode.Int64, NPTypeCode.Int32));
    }

    #endregion
}
