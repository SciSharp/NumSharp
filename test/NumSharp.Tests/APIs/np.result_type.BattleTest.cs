using System;

namespace NumSharp.Tests.APIs;

/// <summary>
/// Battle tests for np.result_type - comprehensive coverage of type promotion.
/// </summary>
[TestClass]
public class NpResultTypeBattleTests
{
    #region Single Type

    [TestMethod]
    public void ResultType_SingleType_ReturnsSame()
    {
        np.result_type(NPTypeCode.Int32).Should().Be(NPTypeCode.Int32);
        np.result_type(NPTypeCode.Double).Should().Be(NPTypeCode.Double);
    }

    #endregion

    #region Two Types - Integer Promotion

    [TestMethod]
    public void ResultType_Int32Int64_ReturnsInt64()
    {
        np.result_type(NPTypeCode.Int32, NPTypeCode.Int64).Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void ResultType_Int16Int32_ReturnsInt32()
    {
        np.result_type(NPTypeCode.Int16, NPTypeCode.Int32).Should().Be(NPTypeCode.Int32);
    }

    [TestMethod]
    public void ResultType_SignedUnsigned_PromotesToContainBoth()
    {
        var result = np.result_type(NPTypeCode.UInt32, NPTypeCode.Int32);
        result.Should().Be(NPTypeCode.Int64);
    }

    #endregion

    #region Two Types - Float Promotion

    [TestMethod]
    public void ResultType_Float32Float64_ReturnsFloat64()
    {
        np.result_type(NPTypeCode.Single, NPTypeCode.Double).Should().Be(NPTypeCode.Double);
    }

    #endregion

    #region Error Cases

    [TestMethod]
    public void ResultType_Empty_Throws()
    {
        new Action(() => np.result_type(Array.Empty<NPTypeCode>())).Should().Throw<Exception>();
    }

    [TestMethod]
    public void ResultType_NullArray_Throws()
    {
        new Action(() => np.result_type((NPTypeCode[])null!)).Should().Throw<Exception>();
    }

    #endregion

    #region Two-Arg Convenience Overloads

    [TestMethod]
    public void ResultType_TwoArg_NPTypeCode()
    {
        np.result_type(NPTypeCode.Int32, NPTypeCode.Int64).Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void ResultType_TwoArg_Type()
    {
        np.result_type(typeof(int), typeof(long)).Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void ResultType_TwoArg_NDArray()
    {
        var a = np.array(new int[] { 1, 2 });
        var b = np.array(new long[] { 1, 2 });
        np.result_type(a, b).Should().Be(NPTypeCode.Int64);
    }

    /// <summary>
    ///     A null operand is NumPy's <c>None</c>, which <c>np.result_type</c> converts with <c>PyArray_DescrConverter</c>
    ///     into the strong float64 descriptor: <c>np.result_type(None, np.array([1, 2], np.int32))</c> is float64 in NumPy
    ///     2.4.2, in either order.
    /// </summary>
    [TestMethod]
    public void ResultType_TwoArg_NDArray_NullFirst_IsNumPysNone()
    {
        var b = np.array(new int[] { 1, 2 });
        np.result_type((NDArray)null!, b).Should().Be(NPTypeCode.Double);
    }

    /// <summary>The mirror of <see cref="ResultType_TwoArg_NDArray_NullFirst_IsNumPysNone"/>.</summary>
    [TestMethod]
    public void ResultType_TwoArg_NDArray_NullSecond_IsNumPysNone()
    {
        var a = np.array(new int[] { 1, 2 });
        np.result_type(a, (NDArray)null!).Should().Be(NPTypeCode.Double);
    }

    #endregion

    #region Symmetry Property

    [TestMethod]
    public void ResultType_Symmetric()
    {
        np.result_type(NPTypeCode.Int32, NPTypeCode.Int64).Should().Be(np.result_type(NPTypeCode.Int64, NPTypeCode.Int32));
    }

    #endregion
}
