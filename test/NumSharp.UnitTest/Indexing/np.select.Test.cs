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

    // ---- fused SIMD fast path -----------------------------------------------
    // The contiguous, no-cast, full-size-array-choice case runs through
    // DirectILKernelGenerator's fused select kernel instead of the copyto
    // composition. The small tests above only reach that kernel's scalar tail;
    // these use a 1000-element array to exercise its SIMD body + 4x unroll +
    // 1-vector remainder + tail, across every element size (1/2/4/8 bytes), n,
    // and both default kinds (scalar-broadcast and full array). The oracle is a
    // naive first-match reference — identical semantics to NumPy's select.

    private static double[] NaiveSelect(bool[][] conds, double[][] choices, double[] def)
    {
        int size = def.Length;
        var r = new double[size];
        for (int i = 0; i < size; i++)
        {
            r[i] = def[i];
            for (int k = 0; k < conds.Length; k++)
                if (conds[k][i]) { r[i] = choices[k][i]; break; }
        }
        return r;
    }

    [TestMethod]
    public void Select_Fused_SimdPath_MatchesReference()
    {
        // Byte(1) / Int16(2) / Int32(4) / Int64(8) / Single(4) / Double(8) cover every
        // fused-kernel element size and both the integer and float SIMD lanes.
        NPTypeCode[] dtypes =
        {
            NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.Int32,
            NPTypeCode.Int64, NPTypeCode.Single, NPTypeCode.Double,
        };
        const int size = 1000; // > 4 * V256 lanes for every dtype -> hits unroll + remainder + tail

        foreach (var dt in dtypes)
        foreach (var n in new[] { 1, 2, 4 })
        foreach (var arrayDefault in new[] { false, true })
        {
            var conds = new bool[n][];
            var choices = new double[n][];
            var condArr = new NDArray[n];
            var choiceObj = new object[n];
            for (int k = 0; k < n; k++)
            {
                conds[k] = new bool[size];
                choices[k] = new double[size];
                for (int i = 0; i < size; i++)
                {
                    conds[k][i] = ((i + k) % (k + 2)) == 0;   // varied, overlapping masks
                    choices[k][i] = (i * (k + 1)) % 37;       // 0..36 — exact in every dtype incl. byte
                }
                condArr[k] = np.array(conds[k]);
                choiceObj[k] = np.array(choices[k]).astype(dt); // full-size contiguous choice
            }

            var def = new double[size];
            for (int i = 0; i < size; i++) def[i] = 5;
            // Scalar default: a size-1 strong array of dt (defScalar path, result stays dt).
            // Array default: a full-size dt array.
            object defObj = arrayDefault
                ? np.array(def).astype(dt)
                : np.array(new double[] { 5 }).astype(dt);

            var expected = NaiveSelect(conds, choices, def);
            var got = np.select(condArr, choiceObj, defObj);

            got.GetTypeCode.Should().Be(dt, $"result dtype for {dt} n={n} arrDef={arrayDefault}");
            var gd = got.astype(NPTypeCode.Double).ToArray<double>();
            for (int i = 0; i < size; i++)
                gd[i].Should().Be(expected[i], $"{dt} n={n} arrDef={arrayDefault} at [{i}]");
        }
    }

    [TestMethod]
    public void Select_Fused_ContiguousOffsetSlice_ReadsThroughOffset()
    {
        // A contiguous slice has its base pointer advanced and offset folded in; the fused
        // kernel must address logical element 0 (base + Shape.offset), not the buffer base.
        var big = np.arange(100).astype(NPTypeCode.Int64);
        var mask = np.array(new bool[100]);
        for (int i = 0; i < 100; i++) mask.SetValue(i % 2 == 0, i);

        var choice = big["10:30"];   // contiguous, non-zero offset
        var cond = mask["10:30"];
        var r = np.select(new[] { cond }, new object[] { choice }, -1L);

        r.GetTypeCode.Should().Be(NPTypeCode.Int64);
        var rd = r.ToArray<long>();
        for (int i = 0; i < 20; i++)
            rd[i].Should().Be((10 + i) % 2 == 0 ? 10 + i : -1L, $"at [{i}]");
    }
}
