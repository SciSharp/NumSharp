using System;
using System.Threading.Tasks;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for type checking functions: issctype, isdtype, sctype2char, maximum_sctype.
/// </summary>
public class NpTypeChecksBattleTests
{
    #region issctype Tests

    [Test]
    public async Task IsSctype_NPTypeCode_ValidTypes()
    {
        await Assert.That(np.issctype(NPTypeCode.Int32)).IsTrue();
        await Assert.That(np.issctype(NPTypeCode.Double)).IsTrue();
        await Assert.That(np.issctype(NPTypeCode.Boolean)).IsTrue();
    }

    [Test]
    public async Task IsSctype_NPTypeCode_Invalid()
    {
        await Assert.That(np.issctype(NPTypeCode.Empty)).IsFalse();
        await Assert.That(np.issctype(NPTypeCode.String)).IsFalse();
    }

    [Test]
    public async Task IsSctype_Type_Valid()
    {
        await Assert.That(np.issctype(typeof(int))).IsTrue();
        await Assert.That(np.issctype(typeof(double))).IsTrue();
    }

    [Test]
    public async Task IsSctype_Type_Invalid()
    {
        await Assert.That(np.issctype(typeof(NDArray))).IsFalse();
        await Assert.That(np.issctype(typeof(string))).IsFalse();
    }

    [Test]
    public async Task IsSctype_Null_ReturnsFalse()
    {
        await Assert.That(np.issctype(null)).IsFalse();
    }

    #endregion

    #region isdtype Tests - NPTypeCode

    [Test]
    public async Task IsDtype_Bool()
    {
        await Assert.That(np.isdtype(NPTypeCode.Boolean, "bool")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.Int32, "bool")).IsFalse();
    }

    [Test]
    public async Task IsDtype_Integral()
    {
        await Assert.That(np.isdtype(NPTypeCode.Int32, "integral")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.Byte, "integral")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.Double, "integral")).IsFalse();
    }

    [Test]
    public async Task IsDtype_RealFloating()
    {
        await Assert.That(np.isdtype(NPTypeCode.Double, "real floating")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.Int32, "real floating")).IsFalse();
    }

    [Test]
    public async Task IsDtype_Numeric()
    {
        await Assert.That(np.isdtype(NPTypeCode.Int32, "numeric")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.Double, "numeric")).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.Boolean, "numeric")).IsFalse();
    }

    [Test]
    public async Task IsDtype_MultipleKinds()
    {
        await Assert.That(np.isdtype(NPTypeCode.Int32, new[] { "integral", "real floating" })).IsTrue();
        await Assert.That(np.isdtype(NPTypeCode.Double, new[] { "integral", "real floating" })).IsTrue();
    }

    #endregion

    #region isdtype Tests - Type Overload

    [Test]
    public async Task IsDtype_Type_Integral()
    {
        await Assert.That(np.isdtype(typeof(int), "integral")).IsTrue();
        await Assert.That(np.isdtype(typeof(double), "integral")).IsFalse();
    }

    #endregion

    #region isdtype Tests - NDArray Overload

    [Test]
    public async Task IsDtype_NDArray_Integral()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        await Assert.That(np.isdtype(arr, "integral")).IsTrue();
        await Assert.That(np.isdtype(arr, "floating")).IsFalse();
    }

    [Test]
    public async Task IsDtype_NDArray_Null_Throws()
    {
        await Assert.That(() => np.isdtype((NDArray)null!, "integral")).ThrowsException();
    }

    #endregion

    #region sctype2char Tests

    [Test]
    public async Task Sctype2Char_Boolean()
    {
        await Assert.That(np.sctype2char(NPTypeCode.Boolean)).IsEqualTo('b');
    }

    [Test]
    public async Task Sctype2Char_Byte()
    {
        await Assert.That(np.sctype2char(NPTypeCode.Byte)).IsEqualTo('B');
    }

    [Test]
    public async Task Sctype2Char_Int32()
    {
        await Assert.That(np.sctype2char(NPTypeCode.Int32)).IsEqualTo('i');
    }

    [Test]
    public async Task Sctype2Char_Int64()
    {
        await Assert.That(np.sctype2char(NPTypeCode.Int64)).IsEqualTo('q');
    }

    [Test]
    public async Task Sctype2Char_Single()
    {
        await Assert.That(np.sctype2char(NPTypeCode.Single)).IsEqualTo('f');
    }

    [Test]
    public async Task Sctype2Char_Double()
    {
        await Assert.That(np.sctype2char(NPTypeCode.Double)).IsEqualTo('d');
    }

    [Test]
    public async Task Sctype2Char_Unknown()
    {
        await Assert.That(np.sctype2char(NPTypeCode.Empty)).IsEqualTo('?');
    }

    #endregion

    #region maximum_sctype Tests

    [Test]
    public async Task MaximumSctype_Boolean_StaysBoolean()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Boolean)).IsEqualTo(NPTypeCode.Boolean);
    }

    [Test]
    public async Task MaximumSctype_SignedIntegers_ToInt64()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Int16)).IsEqualTo(NPTypeCode.Int64);
        await Assert.That(np.maximum_sctype(NPTypeCode.Int32)).IsEqualTo(NPTypeCode.Int64);
        await Assert.That(np.maximum_sctype(NPTypeCode.Int64)).IsEqualTo(NPTypeCode.Int64);
    }

    [Test]
    public async Task MaximumSctype_UnsignedIntegers_ToUInt64()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Byte)).IsEqualTo(NPTypeCode.UInt64);
        await Assert.That(np.maximum_sctype(NPTypeCode.UInt32)).IsEqualTo(NPTypeCode.UInt64);
    }

    [Test]
    public async Task MaximumSctype_Floats_ToDouble()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Single)).IsEqualTo(NPTypeCode.Double);
        await Assert.That(np.maximum_sctype(NPTypeCode.Double)).IsEqualTo(NPTypeCode.Double);
    }

    [Test]
    public async Task MaximumSctype_Decimal_StaysDecimal()
    {
        await Assert.That(np.maximum_sctype(NPTypeCode.Decimal)).IsEqualTo(NPTypeCode.Decimal);
    }

    #endregion
}
