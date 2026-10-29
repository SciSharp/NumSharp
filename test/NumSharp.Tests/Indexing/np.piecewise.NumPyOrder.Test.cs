using System;
using System.Numerics;

namespace NumSharp.Tests.Indexing;

/// <summary>
///     np.piecewise's constants as NumPy stores them, found when the np-layer dtype plan's P3 batch 7 moved its body to
///     <c>NDArray.Piecewise</c> and replayed the snapshot against NumPy 2.4.2: a constant "function" is
///     <c>y[cond] = func</c>, so the boolean index is checked before the value, and the value is converted by
///     <c>PyArray_FromAny(func, y.dtype, FORCECAST)</c> - Python's <c>int()</c> / <c>float()</c> texts for a complex,
///     a complex's truth into bool and its <c>str()</c> into text, None's setitem value, a NumPy scalar cast. Every
///     expectation is NumPy 2.4.2's output.
/// </summary>
[TestClass]
public class PiecewiseNumPyOrderTests
{
    private static NDArray Mask => np.array(new[] { true, false, true, false });
    private static NDArray Int8X => np.array(new sbyte[] { 1, 2, 3, 4 });

    /// <summary>A complex constant into an integer array is refused by Python's int(), into a float array by float().</summary>
    [TestMethod]
    public void ComplexConstant_IntegerAndFloatTargets_RaisePythonsTexts()
    {
        // np.piecewise(np.array([1, 2, 3, 4], np.int8), [c], [1+2j])
        Action intTarget = () => np.piecewise(Int8X, new[] { Mask }, new object[] { new Complex(1, 2) });
        intTarget.Should().ThrowExactly<TypeError>()
            .WithMessage("int() argument must be a string, a bytes-like object or a real number, not 'complex'");

        // np.piecewise(x.astype(np.float32), [c], [1+2j])
        Action floatTarget = () => np.piecewise(Int8X.astype(np.float32), new[] { Mask }, new object[] { new Complex(1, 2) });
        floatTarget.Should().ThrowExactly<TypeError>()
            .WithMessage("float() argument must be a string or a real number, not 'complex'");
    }

    /// <summary>A complex constant into bool stores its truth, into text its str() cut to the width.</summary>
    [TestMethod]
    public void ComplexConstant_BoolAndTextTargets_AreStored()
    {
        // np.piecewise(x.astype(bool), [c], [1+2j, 0j]) -> [True, False, True, False]
        var b = np.piecewise(Int8X.astype(np.@bool), new[] { Mask }, new object[] { new Complex(1, 2), Complex.Zero });
        b.ToArray<bool>().Should().Equal(true, false, true, false);

        // np.piecewise(x.astype('U5'), [c], [1+2j]) -> ['(1+2j', '', '(1+2j', '']
        var u = np.piecewise(Int8X.astype(np.dtype("U5")), new[] { Mask }, new object[] { new Complex(1, 2) });
        u.dtype.Should().Be(np.dtype("U5"));
        NDArray.AsStringArray(u).Should().Equal("(1+2j", "", "(1+2j", "");
    }

    /// <summary>
    ///     The boolean index is checked first: a condition that does not fit x raises NumPy's IndexError even when its
    ///     constant could not be stored either.
    /// </summary>
    [TestMethod]
    public void MaskThatDoesNotFit_IsReportedBeforeTheConstant()
    {
        // np.piecewise(np.zeros((2, 3), np.int8), np.ones((3, 3), bool), [1+2j, 0, 0])
        Action complexConstant = () => np.piecewise(np.zeros(new Shape(2, 3), np.int8), np.ones(new Shape(3, 3), np.@bool),
            new object[] { new Complex(1, 2), 0, 0 });
        complexConstant.Should().ThrowExactly<IndexError>()
            .WithMessage("boolean index did not match indexed array along axis 0; size of axis is 2 but size of " +
                         "corresponding boolean axis is 3");

        // np.piecewise(np.zeros((2, 3), np.uint8), [np.ones(3, bool)], [1000])
        Action bigConstant = () => np.piecewise(np.zeros(new Shape(2, 3), np.uint8), new[] { np.ones(new Shape(3), np.@bool) },
            new object[] { 1000 });
        bigConstant.Should().ThrowExactly<IndexError>()
            .WithMessage("boolean index did not match indexed array along axis 0; size of axis is 2 but size of " +
                         "corresponding boolean axis is 3");
    }

    /// <summary>A null constant is Python's None: NaN into a float array, int()'s TypeError into an integer one.</summary>
    [TestMethod]
    public void NullConstant_IsPythonsNone()
    {
        // np.piecewise(x.astype(np.float64), [c], [None]) -> [nan, 0, nan, 0]
        var f = np.piecewise(Int8X.astype(np.float64), new[] { Mask }, new object[] { null });
        var values = f.ToArray<double>();
        double.IsNaN(values[0]).Should().BeTrue();
        values[1].Should().Be(0);
        double.IsNaN(values[2]).Should().BeTrue();
        values[3].Should().Be(0);

        // np.piecewise(np.array([1, 2, 3, 4], np.int8), [c], [None])
        Action intTarget = () => np.piecewise(Int8X, new[] { Mask }, new object[] { null });
        intTarget.Should().ThrowExactly<TypeError>()
            .WithMessage("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
    }

    /// <summary>A Half is a NumPy scalar (np.float16), which the store CASTS: 300 into int8 is 44, not an OverflowError.</summary>
    [TestMethod]
    public void HalfConstant_IsCastLikeANumPyScalar()
    {
        // np.piecewise(np.array([1, 2, 3, 4], np.int8), [c], [np.float16(300)]) -> [44, 0, 44, 0]
        var r = np.piecewise(Int8X, new[] { Mask }, new object[] { (Half)300 });
        r.dtype.Should().Be(np.int8);
        r.ToArray<sbyte>().Should().Equal((sbyte)44, (sbyte)0, (sbyte)44, (sbyte)0);
    }

    /// <summary>
    ///     A C# array constant is a sequence with one value per selected slot (it took only its first element through the
    ///     fused path), and a length that is neither 1 nor the selected count is NumPy's ValueError.
    /// </summary>
    [TestMethod]
    public void ArrayConstant_FillsTheSelectedSlotsInOrder()
    {
        // np.piecewise(np.array([1, 2, 3, 4], np.int8), [c], [[7, 9]]) -> [7, 0, 9, 0]
        var r = np.piecewise(Int8X, new[] { Mask }, new object[] { new[] { 7, 9 } });
        r.ToArray<sbyte>().Should().Equal((sbyte)7, (sbyte)0, (sbyte)9, (sbyte)0);

        // np.piecewise(np.array([1, 2, 3, 4], np.int8), [c], [[7, 9, 11]])
        Action act = () => np.piecewise(Int8X, new[] { Mask }, new object[] { new[] { 7, 9, 11 } });
        act.Should().ThrowExactly<ValueError>()
            .WithMessage("NumPy boolean array indexing assignment cannot assign 3 input values to the 2 output values where " +
                         "the mask is true");
    }
}
