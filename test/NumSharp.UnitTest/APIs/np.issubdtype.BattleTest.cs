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
            await Assert.That(np.issubdtype(type, "generic")).IsTrue();
        }
    }

    #endregion

    #region Number Category

    [TestMethod]
    public async Task IsSubdtype_IntegersAreNumbers()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Int32, "number")).IsTrue();
        await Assert.That(np.issubdtype(NPTypeCode.Byte, "number")).IsTrue();
    }

    [TestMethod]
    public async Task IsSubdtype_FloatsAreNumbers()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Double, "number")).IsTrue();
    }

    [TestMethod]
    public async Task IsSubdtype_BoolNotNumber_NumPy2x()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Boolean, "number")).IsFalse();
    }

    #endregion

    #region Integer Category

    [TestMethod]
    public async Task IsSubdtype_IntegersAreInteger()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Int32, "integer")).IsTrue();
        await Assert.That(np.issubdtype(NPTypeCode.Byte, "integer")).IsTrue();
    }

    [TestMethod]
    public async Task IsSubdtype_BoolNotInteger_NumPy2x()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Boolean, "integer")).IsFalse();
    }

    [TestMethod]
    public async Task IsSubdtype_FloatNotInteger()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Double, "integer")).IsFalse();
    }

    #endregion

    #region Signed/Unsigned Integer Categories

    [TestMethod]
    public async Task IsSubdtype_SignedIntegers_AreSignedInteger()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Int32, "signedinteger")).IsTrue();
        await Assert.That(np.issubdtype(NPTypeCode.Int64, "signedinteger")).IsTrue();
    }

    [TestMethod]
    public async Task IsSubdtype_UnsignedIntegers_AreUnsignedInteger()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Byte, "unsignedinteger")).IsTrue();
        await Assert.That(np.issubdtype(NPTypeCode.UInt32, "unsignedinteger")).IsTrue();
    }

    #endregion

    #region Floating Category

    [TestMethod]
    public async Task IsSubdtype_Floats_AreFloating()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Single, "floating")).IsTrue();
        await Assert.That(np.issubdtype(NPTypeCode.Double, "floating")).IsTrue();
    }

    [TestMethod]
    public async Task IsSubdtype_Integers_NotFloating()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Int32, "floating")).IsFalse();
    }

    #endregion

    #region NDArray Overload

    [TestMethod]
    public async Task IsSubdtype_NDArray_String()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        await Assert.That(np.issubdtype(arr, "integer")).IsTrue();
        await Assert.That(np.issubdtype(arr, "floating")).IsFalse();
    }

    [TestMethod]
    public async Task IsSubdtype_NDArray_Null_Throws()
    {
        await Assert.That(() => np.issubdtype((NDArray)null!, "integer")).ThrowsException();
    }

    #endregion

    #region Type Overloads

    [TestMethod]
    public async Task IsSubdtype_Type_String()
    {
        await Assert.That(np.issubdtype(typeof(int), "integer")).IsTrue();
        await Assert.That(np.issubdtype(typeof(double), "floating")).IsTrue();
    }

    [TestMethod]
    public async Task IsSubdtype_Type_Type()
    {
        // Same concrete types: returns True
        await Assert.That(np.issubdtype(typeof(int), typeof(int))).IsTrue();
        await Assert.That(np.issubdtype(typeof(long), typeof(long))).IsTrue();

        // Different concrete types: returns False (even if same kind)
        // This matches NumPy: np.issubdtype(np.int32, np.int64) == False
        await Assert.That(np.issubdtype(typeof(int), typeof(long))).IsFalse();
        await Assert.That(np.issubdtype(typeof(float), typeof(double))).IsFalse();
    }

    #endregion

    #region Invalid Category

    [TestMethod]
    public async Task IsSubdtype_InvalidCategory_ReturnsFalse()
    {
        await Assert.That(np.issubdtype(NPTypeCode.Int32, "invalid_category")).IsFalse();
    }

    #endregion
}
