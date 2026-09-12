using System;
using System.Numerics;

namespace NumSharp.Tests.Indexing;

/// <summary>
/// Tests for <c>np.piecewise(x, condlist, funclist[, args])</c> — port of NumPy 2.x
/// <c>numpy.piecewise</c>. Every expected value/dtype/error was taken from running NumPy 2.4.2.
/// The bit-exact SCALAR-funclist value/layout/precedence coverage lives in the differential-fuzz
/// corpus (groupa tier, op "piecewise"); these tests pin what the corpus cannot encode: the
/// CALLABLE funclist path (with <c>*args</c>), the weak-scalar assignment edges (float truncation,
/// out-of-range overflow, complex-into-real), the three overloads and their condlist promotion, the
/// error contract, and the F-/transposed-layout default-condition regression.
/// </summary>
[TestClass]
public class PiecewiseTests
{
    private static readonly Func<NDArray, NDArray> Neg = v => -v;
    private static readonly Func<NDArray, NDArray> Id = v => v;

    // ---- basics: scalar funcs & callables ------------------------------------

    [TestMethod]
    public void Piecewise_Signum_ScalarFuncs()
    {
        var x = np.linspace(-2.5, 2.5, 6);                       // [-2.5,-1.5,-0.5,0.5,1.5,2.5]
        var r = np.piecewise(x, new[] { x < 0, x >= 0 }, new object[] { -1, 1 });
        r.dtype.Should().Be(np.float64);                        // result dtype == x's dtype
        r.ToArray<double>().Should().Equal(-1, -1, -1, 1, 1, 1);
    }

    [TestMethod]
    public void Piecewise_Abs_Callables()
    {
        var x = np.linspace(-2.5, 2.5, 6);
        var r = np.piecewise(x, new[] { x < 0, x >= 0 }, new object[] { Neg, Id });
        r.ToArray<double>().Should().Equal(2.5, 1.5, 0.5, 0.5, 1.5, 2.5);
    }

    [TestMethod]
    public void Piecewise_DefaultFunc_WhenOneExtraFunction()
    {
        var x = np.linspace(-2.5, 2.5, 6);
        // 2 conditions, 3 funcs → the 3rd is the default (where no condition is true).
        var r = np.piecewise(x, new[] { x < -1, x > 1 }, new object[] { -1, 1, 0 });
        r.ToArray<double>().Should().Equal(-1, -1, 0, 0, 1, 1);
    }

    [TestMethod]
    public void Piecewise_OverlappingConditions_LastTrueWins()
    {
        var y = np.arange(5).astype(NPTypeCode.Int64);
        // Both conditions true everywhere; the LAST wins (forward overwrite — opposite of select).
        np.piecewise(y, new[] { y >= 0, y >= 0 }, new object[] { 10, 20 })
            .ToArray<long>().Should().Equal(20, 20, 20, 20, 20);
        np.piecewise(y, new[] { y >= 0, y >= 0 }, new object[] { 20, 10 })
            .ToArray<long>().Should().Equal(10, 10, 10, 10, 10);
        // Partial overlap: [y>=1,y>=3],[100,200] → [0,100,100,200,200].
        np.piecewise(y, new[] { y >= 1, y >= 3 }, new object[] { 100, 200 })
            .ToArray<long>().Should().Equal(0, 100, 100, 200, 200);
    }

    // ---- output dtype == x's dtype (zeros_like) ------------------------------

    [TestMethod]
    public void Piecewise_ResultDtypeIsXDtype_FloatFuncTruncatesIntoInt()
    {
        var xi = np.array(new[] { 1, 2, 3, 4 });                // int32
        // Float scalar funcs into an int32 x TRUNCATE toward zero (weak-scalar assignment).
        var r = np.piecewise(xi, new[] { xi < 3, xi >= 3 }, new object[] { 1.9, 2.9 });
        r.dtype.Should().Be(np.int32);
        r.ToArray<int>().Should().Equal(1, 1, 2, 2);
    }

    [TestMethod]
    public void Piecewise_ResultDtypeIsXDtype_LambdaProducingFloatTruncatesIntoInt()
    {
        var xi = np.array(new[] { 1, 2, 3, 4 });                // int32
        var r = np.piecewise(xi, new[] { xi < 3, xi >= 3 },
            new object[] { (Func<NDArray, NDArray>)(v => v * 1.5), (Func<NDArray, NDArray>)(v => v * 2.5) });
        r.dtype.Should().Be(np.int32);
        r.ToArray<int>().Should().Equal(1, 3, 7, 10);
    }

    [TestMethod]
    public void Piecewise_BoolX_StaysBool()
    {
        var xb = np.array(new[] { true, false, true });
        var r = np.piecewise(xb, new[] { xb, !xb }, new object[] { 1, 0 });
        r.dtype.Should().Be(np.@bool);
        r.ToArray<bool>().Should().Equal(true, false, true);
    }

    // ---- *args forwarded to Func<NDArray, object[], NDArray> ------------------

    [TestMethod]
    public void Piecewise_ArgsForwardedToCallables()
    {
        var x = np.linspace(-2.5, 2.5, 6);
        var r = np.piecewise(x, new[] { x < 0, x >= 0 }, new object[]
        {
            (Func<NDArray, object[], NDArray>)((v, a) => v * (int)a[0]),
            (Func<NDArray, object[], NDArray>)((v, a) => v + (int)a[0]),
        }, 100);
        r.ToArray<double>().Should().Equal(-250, -150, -50, 100.5, 101.5, 102.5);
    }

    [TestMethod]
    public void Piecewise_CallableNotCalledOnEmptySelection()
    {
        // The first condition selects nothing; the callable there must NOT run (NumPy's vals.size>0
        // guard). A callable that throws on empty input proves it is skipped.
        var x = np.linspace(-2.5, 2.5, 6);
        Func<NDArray, NDArray> boom = v =>
        {
            if (v.size == 0) throw new InvalidOperationException("called on empty");
            return v * 2;
        };
        var r = np.piecewise(x, new[] { x > 100, x >= 0 }, new object[] { boom, boom });
        r.ToArray<double>().Should().Equal(0, 0, 0, 1, 3, 5);   // only the second (x>=0) fires
    }

    // ---- callable returning a scalar / an NDArray func entry -----------------

    [TestMethod]
    public void Piecewise_CallableReturningScalar_Broadcasts()
    {
        var x = np.linspace(-2.5, 2.5, 6);
        var r = np.piecewise(x, new[] { x < 0, x >= 0 }, new object[]
        {
            (Func<NDArray, NDArray>)(v => NDArray.Scalar(42.0)),
            (Func<NDArray, NDArray>)(v => NDArray.Scalar(7.0)),
        });
        r.ToArray<double>().Should().Equal(42, 42, 42, 7, 7, 7);
    }

    [TestMethod]
    public void Piecewise_NDArrayFuncEntry_BroadcastAndExactLength()
    {
        var x = np.array(new[] { 1.0, 2, 3, 4 });
        // size-1 array broadcasts into the selected slots.
        np.piecewise(x, new[] { np.array(new[] { true, false, true, false }) },
                new object[] { np.array(new[] { 99.0 }) })
            .ToArray<double>().Should().Equal(99, 0, 99, 0);
        // exact-length array places into the selected slots in C-order.
        np.piecewise(x, new[] { np.array(new[] { true, true, false, false }) },
                new object[] { np.array(new[] { 7.0, 8.0 }) })
            .ToArray<double>().Should().Equal(7, 8, 0, 0);
    }

    // ---- weak-scalar assignment edges ---------------------------------------

    [TestMethod]
    public void Piecewise_OutOfRangeIntScalar_Overflows()
    {
        var x = np.array(new sbyte[] { 1, 2 });                 // int8
        Action act = () => np.piecewise(x, new[] { np.array(new[] { true, false }) }, new object[] { 300 });
        act.Should().Throw<OverflowException>()
            .WithMessage("Python integer 300 out of bounds for int8");
    }

    [TestMethod]
    public void Piecewise_ComplexScalarIntoRealX_Throws()
    {
        var x = np.array(new[] { 1.0, 2.0 });                   // float64
        Action act = () => np.piecewise(x, new[] { np.array(new[] { true, false }) },
            new object[] { new Complex(1, 2) });
        act.Should().Throw<TypeError>();
    }

    // ---- errors --------------------------------------------------------------

    [TestMethod]
    public void Piecewise_WrongFunctionCount_Throws()
    {
        var x = np.linspace(-2.5, 2.5, 6);
        Action act2 = () => np.piecewise(x, new[] { x < 0, x >= 0 }, new object[] { 1 });
        act2.Should().Throw<ValueError>()
            .WithMessage("with 2 condition(s), either 2 or 3 functions are expected");
        Action act1 = () => np.piecewise(x, new[] { x < 0 }, new object[] { 1, 2, 3 });
        act1.Should().Throw<ValueError>()
            .WithMessage("with 1 condition(s), either 1 or 2 functions are expected");
    }

    [TestMethod]
    public void Piecewise_EmptyCondlist_Throws()
    {
        var x = np.linspace(-2.5, 2.5, 6);
        Action act = () => np.piecewise(x, new NDArray[0], new object[] { 0 });
        act.Should().Throw<IndexError>().WithMessage("list index out of range");
    }

    [TestMethod]
    public void Piecewise_NullFuncEntry_Throws()
    {
        var x = np.linspace(-2.5, 2.5, 6);
        Action act = () => np.piecewise(x, new[] { x < 0 }, new object[] { null });
        act.Should().Throw<ArgumentNullException>();
    }

    // ---- overloads: bare-array & scalar-bool condlist promotion --------------

    [TestMethod]
    public void Piecewise_BareSingle1DCond_IsOneCondition()
    {
        var x = np.linspace(-2.5, 2.5, 6);
        // A bare 1-D condition over a non-scalar x is ONE whole-array condition.
        np.piecewise(x, x < 0, new object[] { 1 })
            .ToArray<double>().Should().Equal(1, 1, 1, 0, 0, 0);
        np.piecewise(x, x < 0, new object[] { 1, 2 })            // + default
            .ToArray<double>().Should().Equal(1, 1, 1, 2, 2, 2);
    }

    [TestMethod]
    public void Piecewise_BareStacked2DCond_IsUnstackedRows()
    {
        var x2 = np.arange(6).reshape(2, 3) - 2;
        var stacked = np.stack(new[] { x2 < 0, x2 >= 0 });      // (2,2,3): axis 0 = condition axis
        np.piecewise(x2, stacked, new object[] { -1, 1 })
            .ToArray<long>().Should().Equal(-1, -1, 1, 1, 1, 1);
    }

    [TestMethod]
    public void Piecewise_ScalarBoolCond_AllOrNothing()
    {
        var x = np.linspace(-2.5, 2.5, 6);
        np.piecewise(x, true, new object[] { 7 }).ToArray<double>().Should().Equal(7, 7, 7, 7, 7, 7);
        np.piecewise(x, false, new object[] { 7 }).ToArray<double>().Should().Equal(0, 0, 0, 0, 0, 0);
    }

    // ---- shapes: scalar / empty / 2-D / none-true ---------------------------

    [TestMethod]
    public void Piecewise_ScalarX_Returns0d()
    {
        var y = NDArray.Scalar(-2L);
        var r = np.piecewise(y, new[] { y < 0, y >= 0 }, new object[] { Neg, Id });
        r.ndim.Should().Be(0);
        r.GetAtIndex<long>(0).Should().Be(2);
    }

    [TestMethod]
    public void Piecewise_EmptyX_ReturnsEmpty()
    {
        var xe = np.array(new double[] { });
        var r = np.piecewise(xe, new[] { xe < 0, xe >= 0 }, new object[] { 1, 2 });
        r.size.Should().Be(0);
        r.shape.Should().Equal(0);
    }

    [TestMethod]
    public void Piecewise_2D_Callables()
    {
        var x2 = np.arange(6).reshape(2, 3) - 2;
        var r = np.piecewise(x2, new[] { x2 < 0, x2 >= 0 },
            new object[] { (Func<NDArray, NDArray>)(v => v * v), (Func<NDArray, NDArray>)(v => -v) });
        r.ToArray<long>().Should().Equal(4, 1, 0, -1, -2, -3);
    }

    [TestMethod]
    public void Piecewise_NoConditionTrue_NoDefault_StaysZero()
    {
        var y = np.arange(5).astype(NPTypeCode.Int64);
        np.piecewise(y, new[] { y > 100 }, new object[] { 999 })
            .ToArray<long>().Should().Equal(0, 0, 0, 0, 0);
    }

    [TestMethod]
    public void Piecewise_NonBoolConditionConvertedByNonzero()
    {
        var x = np.array(new[] { 10.0, 20, 30, 40, 50 });
        var intMask = np.array(new[] { 0, 1, 2, 0, 3 });        // nonzero → True
        np.piecewise(x, new[] { intMask }, new object[] { 9 })
            .ToArray<double>().Should().Equal(0, 9, 9, 0, 9);
    }

    // ---- copy semantics ------------------------------------------------------

    [TestMethod]
    public void Piecewise_ResultIsFreshCopy_DoesNotAliasX()
    {
        var x = np.array(new[] { 1.0, 2.0, 3.0 });
        var r = np.piecewise(x, new[] { x < 2 }, new object[] { 9 });
        r[0] = 123.0;
        x.ToArray<double>().Should().Equal(1, 2, 3);            // x untouched
    }

    // ---- layout: the default condition on non-C-contiguous x ----------------
    // Regression: condelse must be computed layout-correctly (via logical_not, not the ! operator)
    // so a default-function piecewise is right on F-contiguous and transposed x.

    [TestMethod]
    public void Piecewise_DefaultCondition_FContiguousX()
    {
        var baseA = (np.arange(12).astype(NPTypeCode.Double).reshape(3, 4)) - 5.0;
        var xf = np.asfortranarray(baseA);
        var r = np.piecewise(xf, new[] { xf < -1, xf > 1 }, new object[] { 10, 20, 5 });
        // C-order values: 10 where x<-1, 20 where x>1, 5 (default) where -1<=x<=1.
        r.ToArray<double>().Should().Equal(10, 10, 10, 10, 5, 5, 5, 20, 20, 20, 20, 20);
    }

    [TestMethod]
    public void Piecewise_DefaultCondition_TransposedX()
    {
        var baseA = (np.arange(12).astype(NPTypeCode.Double).reshape(3, 4)) - 5.0;
        var xt = baseA.T;                                        // (4,3), non-contiguous
        var r = np.piecewise(xt, new[] { xt < -1, xt > 1 }, new object[] { 10, 20, 5 });
        r.ToArray<double>().Should().Equal(10, 5, 20, 10, 5, 20, 10, 5, 20, 10, 20, 20);
    }
}
