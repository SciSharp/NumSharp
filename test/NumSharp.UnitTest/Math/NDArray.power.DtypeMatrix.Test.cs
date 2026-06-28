using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.MathOps;

/// <summary>
/// Sweep np.power across the full 15-dtype matrix to make sure no (lhs, rhs)
/// combination throws or crashes after the SByte/Half/Complex first-class
/// support pass. Values used are deliberately small (base=2, exp=3) so the
/// integer-overflow path is not exercised here — the existing Comprehensive
/// test covers that. The point of this file is "does every dtype pair survive
/// the dispatch pipeline?"
/// </summary>
[TestClass]
public class NDArray_Power_DtypeMatrix
{
    private static readonly NPTypeCode[] All15 = new[]
    {
        NPTypeCode.Boolean,
        NPTypeCode.Byte,
        NPTypeCode.SByte,
        NPTypeCode.Int16,
        NPTypeCode.UInt16,
        NPTypeCode.Int32,
        NPTypeCode.UInt32,
        NPTypeCode.Int64,
        NPTypeCode.UInt64,
        NPTypeCode.Char,
        NPTypeCode.Half,
        NPTypeCode.Single,
        NPTypeCode.Double,
        NPTypeCode.Decimal,
        NPTypeCode.Complex,
    };

    /// <summary>
    /// Materialise a (2,) scalar-like NDArray of the given dtype.
    /// </summary>
    private static NDArray MakeOnes(NPTypeCode tc)
    {
        // Use astype from a base of integer 2 (or 1 for Boolean which only has true/false).
        // For Boolean we use true (=1). All other dtypes can hold 2 exactly.
        if (tc == NPTypeCode.Boolean)
            return np.array(new[] { true, true });
        return np.array(new[] { 2, 2 }).astype(tc);
    }

    private static NDArray MakeTwos(NPTypeCode tc)
    {
        if (tc == NPTypeCode.Boolean)
            return np.array(new[] { true, true });
        return np.array(new[] { 2, 2 }).astype(tc);
    }

    [TestMethod]
    public void Power_15x15_DtypeMatrix_NoCrash()
    {
        foreach (var lhs in All15)
        {
            foreach (var rhs in All15)
            {
                NDArray a, b;
                try
                {
                    a = MakeTwos(lhs);
                    b = MakeOnes(rhs);
                }
                catch (Exception ex)
                {
                    Assert.Fail($"Setup failed for lhs={lhs}, rhs={rhs}: {ex.GetType().Name}: {ex.Message}");
                    return;
                }

                NDArray result;
                try
                {
                    result = np.power(a, b);
                }
                catch (Exception ex)
                {
                    Assert.Fail($"np.power({lhs}, {rhs}) threw: {ex.GetType().Name}: {ex.Message}");
                    return;
                }

                result.Should().NotBeNull(because: $"np.power({lhs}, {rhs}) must return a result");
                result.size.Should().BeGreaterThan(0, because: $"np.power({lhs}, {rhs}) must produce non-empty result");
            }
        }
    }

    /// <summary>
    /// SByte direct entry — was previously broken via List/IEnumerable input path (T1.49).
    /// </summary>
    [TestMethod]
    public void Power_SByteList_DoesNotThrow()
    {
        var bases = np.array(new sbyte[] { 2, 3 });
        var exps = new System.Collections.Generic.List<sbyte> { 2, 3 };
        Action act = () => np.power(bases, exps);
        act.Should().NotThrow();
    }

    [TestMethod]
    public void Power_HalfList_DoesNotThrow()
    {
        var bases = np.array(new Half[] { (Half)2.0f, (Half)3.0f });
        var exps = new System.Collections.Generic.List<Half> { (Half)2.0f, (Half)1.5f };
        Action act = () => np.power(bases, exps);
        act.Should().NotThrow();
    }

    [TestMethod]
    public void Power_ComplexList_DoesNotThrow()
    {
        var bases = np.array(new Complex[] { new Complex(2, 1), new Complex(3, 0) });
        var exps = new System.Collections.Generic.List<Complex> { new Complex(2, 0), new Complex(1, 0) };
        Action act = () => np.power(bases, exps);
        act.Should().NotThrow();
    }

    /// <summary>
    /// SByte ^ SByte: int8 ** int8 → int8 (NEP50). Squared-exp wrap.
    /// </summary>
    [TestMethod]
    public void Power_SByte_SByte_Wraps()
    {
        var a = np.array(new sbyte[] { 2 });
        var b = np.array(new sbyte[] { 3 });
        var r = np.power(a, b);
        r.typecode.Should().Be(NPTypeCode.SByte);
        r.GetSByte(0).Should().Be(8);
    }

    /// <summary>
    /// SByte ^ negative SByte: should raise ValueError-style exception (negative
    /// integer exponent on integer base is invalid in NumPy).
    /// </summary>
    [TestMethod]
    public void Power_SByte_NegativeSByte_Throws()
    {
        var a = np.array(new sbyte[] { 2 });
        var b = np.array(new sbyte[] { -3 });
        Action act = () => np.power(a, b);
        act.Should().Throw<ArgumentException>(because: "NumPy raises ValueError for int^negative-int");
    }

    /// <summary>
    /// Half ^ Half preserves Half dtype.
    /// </summary>
    [TestMethod]
    public void Power_Half_Half_ReturnsHalf()
    {
        var a = np.array(new Half[] { (Half)2.0f, (Half)3.0f });
        var b = np.array(new Half[] { (Half)2.0f, (Half)2.0f });
        var r = np.power(a, b);
        r.typecode.Should().Be(NPTypeCode.Half);
        ((float)r.GetHalf(0)).Should().BeApproximately(4.0f, 1e-2f);
        ((float)r.GetHalf(1)).Should().BeApproximately(9.0f, 1e-2f);
    }

    /// <summary>
    /// Complex ^ Complex returns Complex.
    /// </summary>
    [TestMethod]
    public void Power_Complex_Complex_ReturnsComplex()
    {
        var a = np.array(new Complex[] { new Complex(2, 0) });
        var b = np.array(new Complex[] { new Complex(3, 0) });
        var r = np.power(a, b);
        r.typecode.Should().Be(NPTypeCode.Complex);
        var v = r.GetComplex(0);
        v.Real.Should().BeApproximately(8.0, 1e-9);
        v.Imaginary.Should().BeApproximately(0.0, 1e-9);
    }

    /// <summary>
    /// Float base ^ Complex exp promotes to Complex.
    /// </summary>
    [TestMethod]
    public void Power_Float_Complex_PromotesToComplex()
    {
        var a = np.array(new float[] { 2.0f });
        var b = np.array(new Complex[] { new Complex(2, 0) });
        var r = np.power(a, b);
        r.typecode.Should().Be(NPTypeCode.Complex);
        r.GetComplex(0).Real.Should().BeApproximately(4.0, 1e-9);
    }

    /// <summary>
    /// Half base ^ float exp promotes to Single (float wider than Half).
    /// </summary>
    [TestMethod]
    public void Power_Half_Single_PromotesToSingle()
    {
        var a = np.array(new Half[] { (Half)2.0f });
        var b = np.array(new float[] { 2.0f });
        var r = np.power(a, b);
        r.typecode.Should().Be(NPTypeCode.Single);
        r.GetSingle(0).Should().BeApproximately(4.0f, 1e-3f);
    }
}
