using System;
using System.Threading.Tasks;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.promote_types - comprehensive coverage of type promotion.
/// </summary>
[TestClass]
public class NpPromoteTypesBattleTests
{
    private static readonly NPTypeCode[] AllTypes = new[]
    {
        NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16,
        NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
        NPTypeCode.Char, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal
    };

    #region Same Type

    [TestMethod]
    public async Task PromoteTypes_SameType_ReturnsSame()
    {
        foreach (var type in AllTypes)
        {
            np.promote_types(type, type).Should().Be(type);
        }
    }

    #endregion

    #region Symmetric Property

    [TestMethod]
    public async Task PromoteTypes_Symmetric()
    {
        foreach (var t1 in AllTypes)
        {
            foreach (var t2 in AllTypes)
            {
                var result1 = np.promote_types(t1, t2);
                var result2 = np.promote_types(t2, t1);
                result1.Should().Be(result2);
            }
        }
    }

    #endregion

    #region Integer Promotion

    [TestMethod]
    public async Task PromoteTypes_Int16Int32()
    {
        np.promote_types(NPTypeCode.Int16, NPTypeCode.Int32).Should().Be(NPTypeCode.Int32);
    }

    [TestMethod]
    public async Task PromoteTypes_Int32Int64()
    {
        np.promote_types(NPTypeCode.Int32, NPTypeCode.Int64).Should().Be(NPTypeCode.Int64);
    }

    #endregion

    #region Float Promotion

    [TestMethod]
    public async Task PromoteTypes_Float32Float64()
    {
        np.promote_types(NPTypeCode.Single, NPTypeCode.Double).Should().Be(NPTypeCode.Double);
    }

    #endregion

    #region Generic Overload

    [TestMethod]
    public async Task PromoteTypes_Generic_IntLong()
    {
        np.promote_types<int, long>().Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public async Task PromoteTypes_Generic_FloatDouble()
    {
        np.promote_types<float, double>().Should().Be(NPTypeCode.Double);
    }

    #endregion

    #region Type Overload

    [TestMethod]
    public async Task PromoteTypes_Type_IntLong()
    {
        np.promote_types(typeof(int), typeof(long)).Should().Be(NPTypeCode.Int64);
    }

    #endregion
}
