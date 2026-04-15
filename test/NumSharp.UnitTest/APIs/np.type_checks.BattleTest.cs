using System;
using System.Threading.Tasks;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for type checking functions: issctype, isdtype, sctype2char, maximum_sctype.
/// </summary>
public class NpTypeChecksBattleTests
{
    #region issctype Tests

    [TestMethod]
    public async Task IsSctype_NPTypeCode_ValidTypes()
    {
        np.issctype(NPTypeCode.Int32).Should().BeTrue();
        np.issctype(NPTypeCode.Double).Should().BeTrue();
        np.issctype(NPTypeCode.Boolean).Should().BeTrue();
    }

    [TestMethod]
    public async Task IsSctype_NPTypeCode_Invalid()
    {
        np.issctype(NPTypeCode.Empty).Should().BeFalse();
        np.issctype(NPTypeCode.String).Should().BeFalse();
    }

    [TestMethod]
    public async Task IsSctype_Type_Valid()
    {
        np.issctype(typeof(int)).Should().BeTrue();
        np.issctype(typeof(double)).Should().BeTrue();
    }

    [TestMethod]
    public async Task IsSctype_Type_Invalid()
    {
        np.issctype(typeof(NDArray)).Should().BeFalse();
        np.issctype(typeof(string)).Should().BeFalse();
    }

    [TestMethod]
    public async Task IsSctype_Null_ReturnsFalse()
    {
        np.issctype(null).Should().BeFalse();
    }

    #endregion

    #region isdtype Tests - NPTypeCode

    [TestMethod]
    public async Task IsDtype_Bool()
    {
        np.isdtype(NPTypeCode.Boolean, "bool").Should().BeTrue();
        np.isdtype(NPTypeCode.Int32, "bool").Should().BeFalse();
    }

    [TestMethod]
    public async Task IsDtype_Integral()
    {
        np.isdtype(NPTypeCode.Int32, "integral").Should().BeTrue();
        np.isdtype(NPTypeCode.Byte, "integral").Should().BeTrue();
        np.isdtype(NPTypeCode.Double, "integral").Should().BeFalse();
    }

    [TestMethod]
    public async Task IsDtype_RealFloating()
    {
        np.isdtype(NPTypeCode.Double, "real floating").Should().BeTrue();
        np.isdtype(NPTypeCode.Int32, "real floating").Should().BeFalse();
    }

    [TestMethod]
    public async Task IsDtype_Numeric()
    {
        np.isdtype(NPTypeCode.Int32, "numeric").Should().BeTrue();
        np.isdtype(NPTypeCode.Double, "numeric").Should().BeTrue();
        np.isdtype(NPTypeCode.Boolean, "numeric").Should().BeFalse();
    }

    [TestMethod]
    public async Task IsDtype_MultipleKinds()
    {
        np.isdtype(NPTypeCode.Int32, new[] { "integral", "real floating" }).Should().BeTrue();
        np.isdtype(NPTypeCode.Double, new[] { "integral", "real floating" }).Should().BeTrue();
    }

    #endregion

    #region isdtype Tests - Type Overload

    [TestMethod]
    public async Task IsDtype_Type_Integral()
    {
        np.isdtype(typeof(int), "integral").Should().BeTrue();
        np.isdtype(typeof(double), "integral").Should().BeFalse();
    }

    #endregion

    #region isdtype Tests - NDArray Overload

    [TestMethod]
    public async Task IsDtype_NDArray_Integral()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        np.isdtype(arr, "integral").Should().BeTrue();
        np.isdtype(arr, "floating").Should().BeFalse();
    }

    [TestMethod]
    public async Task IsDtype_NDArray_Null_Throws()
    {
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<Exception>(() => np.isdtype((NDArray)null!, "integral"));
    }

    #endregion

    #region sctype2char Tests

    [TestMethod]
    public async Task Sctype2Char_Boolean()
    {
        np.sctype2char(NPTypeCode.Boolean).Should().Be('b');
    }

    [TestMethod]
    public async Task Sctype2Char_Byte()
    {
        np.sctype2char(NPTypeCode.Byte).Should().Be('B');
    }

    [TestMethod]
    public async Task Sctype2Char_Int32()
    {
        np.sctype2char(NPTypeCode.Int32).Should().Be('i');
    }

    [TestMethod]
    public async Task Sctype2Char_Int64()
    {
        np.sctype2char(NPTypeCode.Int64).Should().Be('q');
    }

    [TestMethod]
    public async Task Sctype2Char_Single()
    {
        np.sctype2char(NPTypeCode.Single).Should().Be('f');
    }

    [TestMethod]
    public async Task Sctype2Char_Double()
    {
        np.sctype2char(NPTypeCode.Double).Should().Be('d');
    }

    [TestMethod]
    public async Task Sctype2Char_Unknown()
    {
        np.sctype2char(NPTypeCode.Empty).Should().Be('?');
    }

    #endregion

    #region maximum_sctype Tests

    [TestMethod]
    public async Task MaximumSctype_Boolean_StaysBoolean()
    {
        np.maximum_sctype(NPTypeCode.Boolean).Should().Be(NPTypeCode.Boolean);
    }

    [TestMethod]
    public async Task MaximumSctype_SignedIntegers_ToInt64()
    {
        np.maximum_sctype(NPTypeCode.Int16).Should().Be(NPTypeCode.Int64);
        np.maximum_sctype(NPTypeCode.Int32).Should().Be(NPTypeCode.Int64);
        np.maximum_sctype(NPTypeCode.Int64).Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public async Task MaximumSctype_UnsignedIntegers_ToUInt64()
    {
        np.maximum_sctype(NPTypeCode.Byte).Should().Be(NPTypeCode.UInt64);
        np.maximum_sctype(NPTypeCode.UInt32).Should().Be(NPTypeCode.UInt64);
    }

    [TestMethod]
    public async Task MaximumSctype_Floats_ToDouble()
    {
        np.maximum_sctype(NPTypeCode.Single).Should().Be(NPTypeCode.Double);
        np.maximum_sctype(NPTypeCode.Double).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public async Task MaximumSctype_Decimal_StaysDecimal()
    {
        np.maximum_sctype(NPTypeCode.Decimal).Should().Be(NPTypeCode.Decimal);
    }

    #endregion
}
