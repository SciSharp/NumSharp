using System;

namespace NumSharp.Tests.APIs;

/// <summary>
/// Battle tests for type checking functions: issctype, isdtype, sctype2char, maximum_sctype.
/// </summary>
[TestClass]
public class NpTypeChecksBattleTests
{
    #region issctype Tests

    [TestMethod]
    public void IsSctype_NPTypeCode_ValidTypes()
    {
        np.issctype(NPTypeCode.Int32).Should().BeTrue();
        np.issctype(NPTypeCode.Double).Should().BeTrue();
        np.issctype(NPTypeCode.Boolean).Should().BeTrue();
        np.issctype(NPTypeCode.String).Should().BeTrue("the lane of the string classes, each a scalar type in NumPy 1.x");
    }

    [TestMethod]
    public void IsSctype_NPTypeCode_Invalid()
    {
        np.issctype(NPTypeCode.Empty).Should().BeFalse();
    }

    [TestMethod]
    public void IsSctype_Type_Valid()
    {
        np.issctype(typeof(int)).Should().BeTrue();
        np.issctype(typeof(double)).Should().BeTrue();
        np.issctype(typeof(string)).Should().BeTrue("NumPy 1.26.4: np.issctype(str) is True");
    }

    [TestMethod]
    public void IsSctype_Type_Invalid()
    {
        np.issctype(typeof(NDArray)).Should().BeFalse();
    }

    [TestMethod]
    public void IsSctype_Null_ReturnsFalse()
    {
        np.issctype(null).Should().BeFalse();
    }

    /// <summary>
    ///     NumPy 1.x's <c>np.issctype(np.int32)</c> is True. Since Stage B <c>np.int32</c> is a <see cref="DType"/>
    ///     (it was <c>typeof(int)</c>), and <c>issctype</c>'s <c>object</c> parameter let the call compile while the
    ///     descriptor fell through to false; a descriptor now answers by its storage code like the other spellings.
    /// </summary>
    [TestMethod]
    public void IsSctype_DType_AnswersLikeItsStorageCode()
    {
        np.issctype(np.int32).Should().BeTrue();
        np.issctype(np.float64).Should().BeTrue();
        np.issctype(np.bool_).Should().BeTrue();
        np.issctype(np.complex128).Should().BeTrue();
        np.issctype(np.@decimal).Should().BeTrue();
        np.issctype(np.dtype("<i4")).Should().BeTrue();
        np.issctype(np.dtype(">f8")).Should().BeTrue("a non-native byte order still names float64's storage");
        np.issctype(np.dtype("M8[s]")).Should().BeFalse("datetime64 has no storage lane, as NPTypeCode.Empty");
        np.issctype(np.dtype("String")).Should().BeTrue("'<U0', a string class in the String lane");
    }

    /// <summary>
    ///     NumPy 1.x's issctype is False only for <c>object_</c> and for what is no type or dtype, so the string classes
    ///     are scalar types: probed on NumPy 1.26.4, <c>np.issctype</c> of <c>np.str_</c>, <c>np.bytes_</c>,
    ///     <c>np.dtype('U5')</c>, <c>np.dtype('S3')</c>, <c>np.dtype('U')</c>, <c>np.dtype('S')</c>, <c>np.dtype('c')</c>,
    ///     <c>str</c> and <c>bytes</c> is True. NumSharp answered False for all of them: its three arms excluded the String
    ///     lane, which holds every string class. StringDType is NumPy 2's, so 1.26.4 cannot be asked; its scalar type is
    ///     <c>str</c>, which answers True.
    /// </summary>
    [TestMethod]
    public void IsSctype_StringClasses_AreScalarTypes_LikeNumPy1()
    {
        np.issctype(np.str_).Should().BeTrue();
        np.issctype(np.bytes_).Should().BeTrue();
        np.issctype(np.dtype("U5")).Should().BeTrue();
        np.issctype(np.dtype("S3")).Should().BeTrue();
        np.issctype(np.dtype("U")).Should().BeTrue();
        np.issctype(np.dtype("S")).Should().BeTrue();
        np.issctype(np.dtype("c")).Should().BeTrue("'c' is S1");
        np.issctype(np.dtype("T")).Should().BeTrue("StringDType's scalar type is str");
        np.issctype(typeof(string)).Should().BeTrue();

        // The array's own descriptor answers like the dtype it reports.
        using var text = np.array(new[] { "ab", "cde" });
        np.issctype(text.dtype).Should().BeTrue();
    }

    #endregion

    #region isdtype Tests - NPTypeCode

    [TestMethod]
    public void IsDtype_Bool()
    {
        np.isdtype(NPTypeCode.Boolean, "bool").Should().BeTrue();
        np.isdtype(NPTypeCode.Int32, "bool").Should().BeFalse();
    }

    [TestMethod]
    public void IsDtype_Integral()
    {
        np.isdtype(NPTypeCode.Int32, "integral").Should().BeTrue();
        np.isdtype(NPTypeCode.Byte, "integral").Should().BeTrue();
        np.isdtype(NPTypeCode.Double, "integral").Should().BeFalse();
    }

    [TestMethod]
    public void IsDtype_RealFloating()
    {
        np.isdtype(NPTypeCode.Double, "real floating").Should().BeTrue();
        np.isdtype(NPTypeCode.Int32, "real floating").Should().BeFalse();
    }

    [TestMethod]
    public void IsDtype_Numeric()
    {
        np.isdtype(NPTypeCode.Int32, "numeric").Should().BeTrue();
        np.isdtype(NPTypeCode.Double, "numeric").Should().BeTrue();
        np.isdtype(NPTypeCode.Boolean, "numeric").Should().BeFalse();
    }

    [TestMethod]
    public void IsDtype_MultipleKinds()
    {
        np.isdtype(NPTypeCode.Int32, new[] { "integral", "real floating" }).Should().BeTrue();
        np.isdtype(NPTypeCode.Double, new[] { "integral", "real floating" }).Should().BeTrue();
    }

    #endregion

    #region isdtype Tests - Type Overload

    [TestMethod]
    public void IsDtype_Type_Integral()
    {
        np.isdtype(typeof(int), "integral").Should().BeTrue();
        np.isdtype(typeof(double), "integral").Should().BeFalse();
    }

    #endregion

    #region isdtype Tests - NDArray Overload

    [TestMethod]
    public void IsDtype_NDArray_Integral()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        np.isdtype(arr, "integral").Should().BeTrue();
        np.isdtype(arr, "real floating").Should().BeFalse();
        // "floating" is issubdtype's vocabulary, not isdtype's — NumPy 2.4.2 raises ValueError for it.
        Action looseKind = () => np.isdtype(arr, "floating");
        looseKind.Should().Throw<ValueError>().WithMessage("kind argument is a string, but 'floating' is not a known kind name.");
    }

    [TestMethod]
    public void IsDtype_NDArray_Null_Throws()
    {
        new Action(() => np.isdtype((NDArray)null!, "integral")).Should().Throw<Exception>();
    }

    #endregion

    #region sctype2char Tests

    [TestMethod]
    public void Sctype2Char_Boolean()
    {
        // NumPy 2.x: np.dtype(bool).char == '?'. 'b' is int8 (SByte).
        np.sctype2char(NPTypeCode.Boolean).Should().Be('?');
    }

    [TestMethod]
    public void Sctype2Char_SByte()
    {
        // NumPy: np.dtype(np.int8).char == 'b'.
        np.sctype2char(NPTypeCode.SByte).Should().Be('b');
    }

    [TestMethod]
    public void Sctype2Char_Byte()
    {
        np.sctype2char(NPTypeCode.Byte).Should().Be('B');
    }

    [TestMethod]
    public void Sctype2Char_Half()
    {
        // NumPy: np.dtype(np.float16).char == 'e'.
        np.sctype2char(NPTypeCode.Half).Should().Be('e');
    }

    [TestMethod]
    public void Sctype2Char_Int32()
    {
        np.sctype2char(NPTypeCode.Int32).Should().Be('i');
    }

    [TestMethod]
    public void Sctype2Char_Int64()
    {
        np.sctype2char(NPTypeCode.Int64).Should().Be('q');
    }

    [TestMethod]
    public void Sctype2Char_Single()
    {
        np.sctype2char(NPTypeCode.Single).Should().Be('f');
    }

    [TestMethod]
    public void Sctype2Char_Double()
    {
        np.sctype2char(NPTypeCode.Double).Should().Be('d');
    }

    [TestMethod]
    public void Sctype2Char_Unknown()
    {
        np.sctype2char(NPTypeCode.Empty).Should().Be('?');
    }

    #endregion

    #region maximum_sctype Tests

    [TestMethod]
    public void MaximumSctype_Boolean_StaysBoolean()
    {
        np.maximum_sctype(NPTypeCode.Boolean).Should().Be(NPTypeCode.Boolean);
    }

    [TestMethod]
    public void MaximumSctype_SignedIntegers_ToInt64()
    {
        np.maximum_sctype(NPTypeCode.Int16).Should().Be(NPTypeCode.Int64);
        np.maximum_sctype(NPTypeCode.Int32).Should().Be(NPTypeCode.Int64);
        np.maximum_sctype(NPTypeCode.Int64).Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void MaximumSctype_UnsignedIntegers_ToUInt64()
    {
        np.maximum_sctype(NPTypeCode.Byte).Should().Be(NPTypeCode.UInt64);
        np.maximum_sctype(NPTypeCode.UInt32).Should().Be(NPTypeCode.UInt64);
    }

    [TestMethod]
    public void MaximumSctype_Floats_ToDouble()
    {
        np.maximum_sctype(NPTypeCode.Single).Should().Be(NPTypeCode.Double);
        np.maximum_sctype(NPTypeCode.Double).Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void MaximumSctype_Decimal_StaysDecimal()
    {
        np.maximum_sctype(NPTypeCode.Decimal).Should().Be(NPTypeCode.Decimal);
    }

    #endregion
}
