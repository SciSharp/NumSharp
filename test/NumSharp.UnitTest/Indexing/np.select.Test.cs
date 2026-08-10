using System;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Indexing;

/// <summary>
/// Tests for <c>np.select(condlist, choicelist, default)</c> — port of NumPy 2.x
/// <c>numpy.select</c>. Every expected value/dtype/error was taken from running NumPy
/// 2.4.2. The bit-exact value/layout/precedence coverage lives in the differential-fuzz
/// corpus (groupa tier, op "select"); these tests pin what the corpus cannot encode:
/// NEP50 weak-scalar dtype resolution, the error contract, and the strong-vs-weak split.
/// </summary>
[TestClass]
public class SelectTests
{
    private static NDArray Ar(params long[] v) => np.array(v);

    // ---- basics / precedence -------------------------------------------------

    [TestMethod]
    public void Select_Basic_FirstMatchWins()
    {
        var x = np.arange(6).astype(NPTypeCode.Int64);
        // condlist [x<3, x>3], choices [-x, x**2], default 42 → [0,-1,-2,42,16,25]
        var r = np.select(new[] { x < 3, x > 3 }, new object[] { -x, np.power(x, 2) }, 42);
        r.GetTypeCode.Should().Be(NPTypeCode.Int64);
        r.ToArray<long>().Should().Equal(0, -1, -2, 42, 16, 25);
    }

    [TestMethod]
    public void Select_Precedence_FirstConditionTakesPrecedence()
    {
        var x = np.arange(6).astype(NPTypeCode.Int64);
        var r = np.select(new[] { x <= 4, x > 3 }, new object[] { x, np.power(x, 2) }, 55);
        r.ToArray<long>().Should().Equal(0, 1, 2, 3, 4, 25);
    }

    [TestMethod]
    public void Select_DefaultOmitted_IsZero()
    {
        var x = np.arange(6).astype(NPTypeCode.Int64);
        var r = np.select(new[] { x < 3 }, new object[] { x + 100 });
        r.ToArray<long>().Should().Equal(100, 101, 102, 0, 0, 0);
    }

    [TestMethod]
    public void Select_AllConditionsFalse_IsDefaultEverywhere()
    {
        var x = np.arange(6).astype(NPTypeCode.Int64);
        var r = np.select(new[] { x > 100 }, new object[] { x }, 5);
        r.ToArray<long>().Should().Equal(5, 5, 5, 5, 5, 5);
    }

    // ---- errors --------------------------------------------------------------

    [TestMethod]
    public void Select_EmptyCondlist_Throws()
    {
        Action act = () => np.select(new NDArray[0], new object[0]);
        act.Should().Throw<ValueError>()
            .WithMessage("select with an empty condition list is not possible");
    }

    [TestMethod]
    public void Select_LengthMismatch_Throws()
    {
        var x = np.arange(6).astype(NPTypeCode.Int64);
        Action act = () => np.select(new[] { x < 3, x > 3 }, new object[] { x });
        act.Should().Throw<ValueError>()
            .WithMessage("list of cases must be same length as list of conditions");
    }

    [TestMethod]
    public void Select_NonBooleanCondition_Throws()
    {
        var x = np.arange(6).astype(NPTypeCode.Int64);
        Action act = () => np.select(new[] { x }, new object[] { x });
        act.Should().Throw<TypeError>()
            .WithMessage("invalid entry 0 in condlist: should be boolean ndarray");
    }

    // ---- NEP50 weak-scalar dtype resolution ---------------------------------

    [TestMethod]
    public void Select_WeakIntChoices_ResolveToInt64()
    {
        // All-weak python ints → NEP50 default int64 (matches np.result_type(10, 99)).
        var x = np.arange(6).astype(NPTypeCode.Int64);
        var r = np.select(new[] { x < 3 }, new object[] { 10 }, 99);
        r.GetTypeCode.Should().Be(NPTypeCode.Int64);
        r.ToArray<long>().Should().Equal(10, 10, 10, 99, 99, 99);
    }

    [TestMethod]
    public void Select_WeakIntDefault_AdoptsStrongDtype_AndWraps()
    {
        // int8 strong choice + weak int default 1000 → int8; 1000 wraps to -24 (copyto unsafe).
        var x = np.arange(6).astype(NPTypeCode.Int64);
        var r = np.select(new[] { x < 3 }, new object[] { x.astype(NPTypeCode.SByte) }, 1000);
        r.GetTypeCode.Should().Be(NPTypeCode.SByte);
        r.ToArray<sbyte>().Should().Equal(0, 1, 2, -24, -24, -24);
    }

    [TestMethod]
    public void Select_WeakFloatDefault_WithIntChoice_PromotesToFloat64()
    {
        var x = np.arange(6).astype(NPTypeCode.Int64);
        var r = np.select(new[] { x < 3 }, new object[] { x }, 1.5);
        r.GetTypeCode.Should().Be(NPTypeCode.Double);
        r.ToArray<double>().Should().Equal(0d, 1d, 2d, 1.5, 1.5, 1.5);
    }

    [TestMethod]
    public void Select_WeakComplexDefault_PromotesToComplex()
    {
        var x = np.arange(6).astype(NPTypeCode.Int64);
        var r = np.select(new[] { x < 3 }, new object[] { x }, new Complex(0, 1));
        r.GetTypeCode.Should().Be(NPTypeCode.Complex);
        r.ToArray<Complex>().Should().Equal(
            new Complex(0, 0), new Complex(1, 0), new Complex(2, 0),
            new Complex(0, 1), new Complex(0, 1), new Complex(0, 1));
    }

    [TestMethod]
    public void Select_WeakFloatDefault_KeepsFloat16Width()
    {
        // float16 strong choice + weak float default → float16 (weak float adopts the float width).
        var x = np.arange(6).astype(NPTypeCode.Int64);
        var r = np.select(new[] { x < 3 }, new object[] { x.astype(NPTypeCode.Half) }, 1.5);
        r.GetTypeCode.Should().Be(NPTypeCode.Half);
    }

    [TestMethod]
    public void Select_PythonBoolChoice_IsStrong()
    {
        // NumPy asarray's a python bool (only int/float/complex are weak) → bool result here.
        var x = np.arange(6).astype(NPTypeCode.Int64);
        var r = np.select(new[] { x < 3 }, new object[] { true }, false);
        r.GetTypeCode.Should().Be(NPTypeCode.Boolean);
        r.ToArray<bool>().Should().Equal(true, true, true, false, false, false);
    }

    [TestMethod]
    public void Select_StrongScalarChoice_ViaNDArrayScalar()
    {
        // NDArray.Scalar((sbyte)5) is a STRONG int8 scalar (== numpy np.int8(5)); with weak int
        // choice 1000 the result is int8 and 1000 wraps to -24.
        var x = np.arange(6).astype(NPTypeCode.Int64);
        var r = np.select(new[] { x < 3, x >= 3 }, new object[] { NDArray.Scalar((sbyte)5), 1000 }, 0);
        r.GetTypeCode.Should().Be(NPTypeCode.SByte);
        r.ToArray<sbyte>().Should().Equal(5, 5, 5, -24, -24, -24);
    }

    // ---- broadcasting --------------------------------------------------------

    [TestMethod]
    public void Select_BroadcastConditionOverChoice()
    {
        // cond (3,) broadcast over choice (2,3).
        var choice = np.arange(6).astype(NPTypeCode.Int64).reshape(2, 3);
        var cond = np.array(new[] { true, false, true });
        var r = np.select(new[] { cond }, new object[] { choice }, -1);
        r.shape.Should().Equal(2, 3);
        r.ToArray<long>().Should().Equal(0, -1, 2, 3, -1, 5);
    }

    [TestMethod]
    public void Select_BroadcastChoiceOverCondition()
    {
        // cond (2,3), choice (3,) broadcast up.
        var cond = (np.arange(6).astype(NPTypeCode.Int64).reshape(2, 3)) < 3;
        var r = np.select(new[] { cond }, new object[] { Ar(7, 8, 9) }, -1);
        r.shape.Should().Equal(2, 3);
        r.ToArray<long>().Should().Equal(7, 8, 9, -1, -1, -1);
    }

    [TestMethod]
    public void Select_TransposedCondAndChoice()
    {
        // Non-contiguous (transposed) cond + choice must read through their strides.
        var m = np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4);
        var r = np.select(new[] { (m < 3).transpose() }, new object[] { m.transpose() }, -1);
        r.shape.Should().Equal(4, 3);
        r.ToArray<long>().Should().Equal(0, -1, -1, 1, -1, -1, 2, -1, -1, -1, -1, -1);
    }

    // ---- empty / 0-d ---------------------------------------------------------

    [TestMethod]
    public void Select_EmptyArrays_ReturnsEmpty()
    {
        var e = np.zeros(new Shape(0)).astype(NPTypeCode.Boolean);
        var ch = np.zeros(new Shape(0)).astype(NPTypeCode.Int32);
        var r = np.select(new[] { e }, new object[] { ch }, 0);
        r.size.Should().Be(0);
        r.GetTypeCode.Should().Be(NPTypeCode.Int32);
    }

    [TestMethod]
    public void Select_ZeroDim_Scalar()
    {
        var r = np.select(new[] { NDArray.Scalar(true) }, new object[] { NDArray.Scalar(5L) }, 0);
        r.ndim.Should().Be(0);
        r.GetValue(0).Should().Be(5L);
    }
}
