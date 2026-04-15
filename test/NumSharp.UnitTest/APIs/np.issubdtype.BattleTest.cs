using System;
using System.Threading.Tasks;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.issubdtype - comprehensive coverage of type hierarchy.
/// </summary>
public class NpIsSubdtypeBattleTests
{
    private static readonly NPTypeCode[] AllTypes = new[]
    {
        NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16,
        NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
        NPTypeCode.Char, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal
    };

    #region Generic Category

    [TestMethod]
    public async Task IsSubdtype_AllTypes_AreGeneric()
    {
        foreach (var type in AllTypes)
        {
            np.issubdtype(type, "generic").Should().BeTrue();
        }
    }

    #endregion

    #region Number Category

    [TestMethod]
    public async Task IsSubdtype_IntegersAreNumbers()
    {
        np.issubdtype(NPTypeCode.Int32, "number").Should().BeTrue();
        np.issubdtype(NPTypeCode.Byte, "number").Should().BeTrue();
    }

    [TestMethod]
    public async Task IsSubdtype_FloatsAreNumbers()
    {
        np.issubdtype(NPTypeCode.Double, "number").Should().BeTrue();
    }

    [TestMethod]
    public async Task IsSubdtype_BoolNotNumber_NumPy2x()
    {
        np.issubdtype(NPTypeCode.Boolean, "number").Should().BeFalse();
    }

    #endregion

    #region Integer Category

    [TestMethod]
    public async Task IsSubdtype_IntegersAreInteger()
    {
        np.issubdtype(NPTypeCode.Int32, "integer").Should().BeTrue();
        np.issubdtype(NPTypeCode.Byte, "integer").Should().BeTrue();
    }

    [TestMethod]
    public async Task IsSubdtype_BoolNotInteger_NumPy2x()
    {
        np.issubdtype(NPTypeCode.Boolean, "integer").Should().BeFalse();
    }

    [TestMethod]
    public async Task IsSubdtype_FloatNotInteger()
    {
        np.issubdtype(NPTypeCode.Double, "integer").Should().BeFalse();
    }

    #endregion

    #region Signed/Unsigned Integer Categories

    [TestMethod]
    public async Task IsSubdtype_SignedIntegers_AreSignedInteger()
    {
        np.issubdtype(NPTypeCode.Int32, "signedinteger").Should().BeTrue();
        np.issubdtype(NPTypeCode.Int64, "signedinteger").Should().BeTrue();
    }

    [TestMethod]
    public async Task IsSubdtype_UnsignedIntegers_AreUnsignedInteger()
    {
        np.issubdtype(NPTypeCode.Byte, "unsignedinteger").Should().BeTrue();
        np.issubdtype(NPTypeCode.UInt32, "unsignedinteger").Should().BeTrue();
    }

    #endregion

    #region Floating Category

    [TestMethod]
    public async Task IsSubdtype_Floats_AreFloating()
    {
        np.issubdtype(NPTypeCode.Single, "floating").Should().BeTrue();
        np.issubdtype(NPTypeCode.Double, "floating").Should().BeTrue();
    }

    [TestMethod]
    public async Task IsSubdtype_Integers_NotFloating()
    {
        np.issubdtype(NPTypeCode.Int32, "floating").Should().BeFalse();
    }

    #endregion

    #region NDArray Overload

    [TestMethod]
    public async Task IsSubdtype_NDArray_String()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        np.issubdtype(arr, "integer").Should().BeTrue();
        np.issubdtype(arr, "floating").Should().BeFalse();
    }

    [TestMethod]
    public async Task IsSubdtype_NDArray_Null_Throws()
    {
        Microsoft.VisualStudio.TestTools.UnitTesting.Assert.ThrowsException<Exception>(() => np.issubdtype((NDArray)null!, "integer"));
    }

    #endregion

    #region Type Overloads

    [TestMethod]
    public async Task IsSubdtype_Type_String()
    {
        np.issubdtype(typeof(int), "integer").Should().BeTrue();
        np.issubdtype(typeof(double), "floating").Should().BeTrue();
    }

    [TestMethod]
    public async Task IsSubdtype_Type_Type()
    {
        // Same concrete types: returns True
        np.issubdtype(typeof(int), typeof(int)).Should().BeTrue();
        np.issubdtype(typeof(long), typeof(long)).Should().BeTrue();

        // Different concrete types: returns False (even if same kind)
        // This matches NumPy: np.issubdtype(np.int32, np.int64) == False
        np.issubdtype(typeof(int), typeof(long)).Should().BeFalse();
        np.issubdtype(typeof(float), typeof(double)).Should().BeFalse();
    }

    #endregion

    #region Invalid Category

    [TestMethod]
    public async Task IsSubdtype_InvalidCategory_ReturnsFalse()
    {
        np.issubdtype(NPTypeCode.Int32, "invalid_category").Should().BeFalse();
    }

    #endregion
}
