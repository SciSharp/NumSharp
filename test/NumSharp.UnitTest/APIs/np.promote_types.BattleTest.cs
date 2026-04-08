using System;
using System.Threading.Tasks;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace NumSharp.UnitTest.APIs;

/// <summary>
/// Battle tests for np.promote_types - comprehensive coverage of type promotion.
/// </summary>
public class NpPromoteTypesBattleTests
{
    private static readonly NPTypeCode[] AllTypes = new[]
    {
        NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16,
        NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
        NPTypeCode.Char, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal
    };

    #region Same Type

    [Test]
    public async Task PromoteTypes_SameType_ReturnsSame()
    {
        foreach (var type in AllTypes)
        {
            await Assert.That(np.promote_types(type, type)).IsEqualTo(type);
        }
    }

    #endregion

    #region Symmetric Property

    [Test]
    public async Task PromoteTypes_Symmetric()
    {
        foreach (var t1 in AllTypes)
        {
            foreach (var t2 in AllTypes)
            {
                var result1 = np.promote_types(t1, t2);
                var result2 = np.promote_types(t2, t1);
                await Assert.That(result1).IsEqualTo(result2);
            }
        }
    }

    #endregion

    #region Integer Promotion

    [Test]
    public async Task PromoteTypes_Int16Int32()
    {
        await Assert.That(np.promote_types(NPTypeCode.Int16, NPTypeCode.Int32)).IsEqualTo(NPTypeCode.Int32);
    }

    [Test]
    public async Task PromoteTypes_Int32Int64()
    {
        await Assert.That(np.promote_types(NPTypeCode.Int32, NPTypeCode.Int64)).IsEqualTo(NPTypeCode.Int64);
    }

    #endregion

    #region Float Promotion

    [Test]
    public async Task PromoteTypes_Float32Float64()
    {
        await Assert.That(np.promote_types(NPTypeCode.Single, NPTypeCode.Double)).IsEqualTo(NPTypeCode.Double);
    }

    #endregion

    #region Generic Overload

    [Test]
    public async Task PromoteTypes_Generic_IntLong()
    {
        await Assert.That(np.promote_types<int, long>()).IsEqualTo(NPTypeCode.Int64);
    }

    [Test]
    public async Task PromoteTypes_Generic_FloatDouble()
    {
        await Assert.That(np.promote_types<float, double>()).IsEqualTo(NPTypeCode.Double);
    }

    #endregion

    #region Type Overload

    [Test]
    public async Task PromoteTypes_Type_IntLong()
    {
        await Assert.That(np.promote_types(typeof(int), typeof(long))).IsEqualTo(NPTypeCode.Int64);
    }

    #endregion
}
