using System;
using System.Linq;
using NumSharp.Backends;

namespace NumSharp.Tests.Statistics;

/// <summary>
///     NumPy's <c>weights=</c> keyword of <c>np.quantile</c> / <c>np.percentile</c> / <c>np.nanquantile</c> /
///     <c>np.nanpercentile</c> (2.4.2, method <c>inverted_cdf</c> only): every expected value and text probed against
///     <c>numpy==2.4.2</c>. The oracle section <c>qfamily</c> gates the whole matrix; these pin the rules a reader of
///     <c>QuantileWeighted</c> needs first.
/// </summary>
[TestClass]
public class np_quantile_family_Weights_Tests
{
    /// <summary>The elements of an array as doubles in C order.</summary>
    /// <param name="a">The array.</param>
    /// <returns>The values.</returns>
    private static double[] D(NDArray a)
    {
        using var dense = a.copy();
        return Enumerable.Range(0, (int)dense.size)
            .Select(i => dense.Storage.GetAtIndex(i) is Half h ? (double)h : Convert.ToDouble(dense.Storage.GetAtIndex(i)))
            .ToArray();
    }

    /// <summary>
    ///     The cumulative weights pick the element: <c>np.quantile([3, 1, 2, 5, 4], [0, .3, .5, 1], weights=[1, 2, 1, 1, 1])</c>
    ///     is <c>[1, 1, 2, 5]</c> in the data's dtype, and a scalar q gives a 0-d ARRAY (NumPy's weighted branch does not
    ///     unpack it: writeable).
    /// </summary>
    [TestMethod]
    public void Weights_PickByCumulativeWeight_InTheDataDtype()
    {
        using var a = np.array(new[] { 3, 1, 2, 5, 4 });
        using var w = np.array(new[] { 1.0, 2, 1, 1, 1 });
        using var r = np.quantile(a, new[] { 0.0, 0.3, 0.5, 1.0 }, method: "inverted_cdf", weights: w);
        r.dtype.Should().Be(np.int32);
        D(r).Should().Equal(1, 1, 2, 5);
        using var s = np.percentile(a, 50.0, method: "inverted_cdf", weights: w);
        s.ndim.Should().Be(0);
        D(s).Should().Equal(2);
        s.Shape.IsWriteable.Should().BeTrue("NumPy's weighted result is a 0-d array, not a scalar");
    }

    /// <summary>
    ///     NumPy's checks after q's, in order: the method (<c>Only method 'inverted_cdf' supports weights. Got: linear.</c>,
    ///     the name not looked up), the axis (named: <c>axis: axis 3 ...</c>), the weights' shape, their sign; q's rank
    ///     comes later (<c>_quantile_ureduce_func</c>).
    /// </summary>
    [TestMethod]
    public void Weights_ChecksInNumPysOrder()
    {
        using var a = np.array(new[] { 3.0, 1, 2, 5, 4 });
        using var w = np.array(new[] { 1.0, 2, 1, 1, 1 });
        using var negative = -w;
        Action linear = () => np.quantile(a, 0.5, weights: w);
        linear.Should().Throw<ValueError>().WithMessage("Only method 'inverted_cdf' supports weights. Got: linear.");
        Action unknown = () => np.quantile(a, 0.5, method: "nope", weights: w);
        unknown.Should().Throw<ValueError>().WithMessage("Only method 'inverted_cdf' supports weights. Got: nope.");
        Action range = () => np.quantile(a, 1.5, method: "nope", weights: negative);
        range.Should().Throw<ValueError>().WithMessage("Quantiles must be in the range [0, 1]");
        Action axis = () => np.quantile(a, 0.5, axis: 3, method: "inverted_cdf", weights: w);
        axis.Should().Throw<AxisError>().WithMessage("axis: axis 3 is out of bounds for array of dimension 1*");
        using var two = np.array(new[] { 1.0, 2 });
        Action noAxis = () => np.quantile(a, 0.5, method: "inverted_cdf", weights: two);
        noAxis.Should().Throw<TypeError>().WithMessage("Axis must be specified when shapes of a and weights differ.");
        Action shape = () => np.quantile(a, 0.5, axis: 0, method: "inverted_cdf", weights: two);
        shape.Should().Throw<ValueError>().WithMessage("Shape of weights must be consistent with shape of a along specified axis.");
        Action sign = () => np.quantile(a, 0.5, method: "inverted_cdf", weights: negative);
        sign.Should().Throw<ValueError>().WithMessage("Weights must be non-negative.");
        using var q3 = np.zeros(new Shape(1, 1, 1));
        Action rank = () => np.quantile(a, q3, method: "inverted_cdf", weights: w);
        rank.Should().Throw<ValueError>().WithMessage("q must be a scalar or 1d");
        using var m = np.arange(6.0).reshape(2, 3);
        using var wm = np.ones(new Shape(2, 3));
        Action repeated = () => np.quantile(m, 0.5, new[] { 1, -1 }, method: "inverted_cdf", weights: wm);
        repeated.Should().Throw<ValueError>().WithMessage("repeated axis in `axis` argument");
    }

    /// <summary>
    ///     Weights that sum to NaN, inf or zero are refused after the sort (<c>Weights included NaN, inf or were all
    ///     zero.</c>); text weights raise the comparison's refusal (<c>weights &lt; 0</c> with a Python int); a leading zero
    ///     weight never answers q = 0 (<c>cdf[cdf == 0] = -1</c>).
    /// </summary>
    [TestMethod]
    public void Weights_InvalidSums_TextWeights_LeadingZeros()
    {
        using var a = np.array(new[] { 3.0, 1, 2, 5, 4 });
        foreach (var bad in new[] { new[] { 1, double.NaN, 1, 1, 1 }, new[] { 1, double.PositiveInfinity, 1, 1, 1 }, new double[5] })
        {
            using var w = np.array(bad);
            Action act = () => np.quantile(a, 0.5, method: "inverted_cdf", weights: w);
            act.Should().Throw<ValueError>().WithMessage("Weights included NaN, inf or were all zero.");
        }
        using var text = np.array(new[] { "a", "b", "c", "d", "e" });
        Action textWeights = () => np.quantile(a, 0.5, method: "inverted_cdf", weights: text);
        textWeights.Should().Throw<TypeError>().WithMessage(
            "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.StrDType'>, " +
            "<class 'numpy.dtypes._PyLongDType'>) -> None");
        using var lead = np.array(new[] { 1.0, 0, 1, 1, 1 });
        using var r = np.quantile(a, 0.0, method: "inverted_cdf", weights: lead);
        D(r).Should().Equal(2.0);
    }

    /// <summary>
    ///     A float q searches the cdf cast to ITS dtype: with ten equal weights the cdf steps are k / 10, and a float32 q of
    ///     0.3 is 0.30000001192092896 — past the float64 step 3 / 10 = 0.3, so a float64 search would pick the next element
    ///     (3.0); NumPy casts the cdf to float32 first, where the step is the same 0.30000001192092896, and picks 2.0, as the
    ///     float64 q does.
    /// </summary>
    [TestMethod]
    public void Weights_FloatQ_SearchesTheCdfInItsDtype()
    {
        using var a = np.arange(10.0);
        using var w = np.ones(new Shape(10));
        using var q64 = np.array(new[] { 0.3 });
        using var q32 = np.array(new[] { 0.3f });
        using var r64 = np.quantile(a, q64, method: "inverted_cdf", weights: w);
        using var r32 = np.quantile(a, q32, method: "inverted_cdf", weights: w);
        D(r64).Should().Equal(2.0);
        D(r32).Should().Equal(2.0);
        r32.dtype.Should().Be(np.float64, "the result keeps the data's dtype");
    }

    /// <summary>
    ///     Weights of the reduced axis only are transposed into place and broadcast; along a tuple axis weights of the
    ///     reduced axes' shape follow the axes' order (<c>axis=(1, 0)</c> takes <c>(3, 2)</c> weights of a <c>(2, 3)</c>
    ///     array).
    /// </summary>
    [TestMethod]
    public void Weights_OfTheReducedAxes_AreBroadcast()
    {
        using var m = np.array(new double[,] { { 3, 1, 2 }, { 6, 5, 4 } });
        using var w1 = np.array(new[] { 1.0, 2, 3 });
        using var r = np.quantile(m, new[] { 0.2, 0.7 }, 1, method: "inverted_cdf", weights: w1);
        r.shape.Should().Equal(2L, 2L);
        D(r).Should().Equal(1, 4, 2, 5);
        using var wt = np.arange(1.0, 7.0).reshape(3, 2);
        using var t = np.quantile(m, 0.5, new[] { 1, 0 }, method: "inverted_cdf", weights: wt);
        D(t).Should().Equal(4.0);
    }

    /// <summary>
    ///     NumPy 2.4.2's own bug, reproduced: the nan forms index the VALIDATED weights per kept position, and weights of
    ///     the reduced axis only have length 1 there, so a kept axis longer than 1 raises <c>index 1 is out of bounds for
    ///     axis 0 with size 1</c> (one row works). Weights of the data's shape work.
    /// </summary>
    [TestMethod]
    [Misaligned]
    public void NanForms_WeightsOfTheReducedAxis_RaiseNumPysIndexError()
    {
        using var m = np.array(new double[,] { { 3, 1, 2 }, { 6, 5, 4 } });
        using var w1 = np.array(new[] { 1.0, 2, 3 });
        Action act = () => np.nanquantile(m, 0.5, 1, method: "inverted_cdf", weights: w1);
        act.Should().Throw<IndexError>().WithMessage("index 1 is out of bounds for axis 0 with size 1");
        using var row = m["0:1"];
        using var one = np.nanquantile(row, new[] { 0.2, 0.7 }, 1, method: "inverted_cdf", weights: w1);
        D(one).Should().Equal(1, 2);
        using var full = np.ones(new Shape(2, 3));
        using var ok = np.nanquantile(m, new[] { 0.2, 0.7 }, 1, method: "inverted_cdf", weights: full);
        D(ok).Should().Equal(1, 4, 3, 6);
    }

    /// <summary>
    ///     The nan forms drop NaN elements AND their weights (<c>_remove_nan_1d</c> with a second array); the plain forms
    ///     answer a slice whose sorted last element is NaN with that NaN; an all-NaN slice of the nan forms is NaN.
    /// </summary>
    [TestMethod]
    public void NaNData_PlainAndNanForms()
    {
        using var a = np.array(new[] { 3.0, double.NaN, 2, 5, 4 });
        using var w = np.array(new[] { 1.0, 2, 1, 1, 1 });
        using var plain = np.quantile(a, new[] { 0.2, 0.5 }, method: "inverted_cdf", weights: w);
        D(plain).Should().AllSatisfy(v => double.IsNaN(v).Should().BeTrue());
        using var nan = np.nanquantile(a, new[] { 0.2, 0.5 }, method: "inverted_cdf", weights: w);
        D(nan).Should().Equal(2, 3);
        using var allNaN = np.full(new Shape(4), double.NaN);
        using var ones = np.ones(new Shape(4));
        using var r = np.nanquantile(allNaN, new[] { 0.2, 0.5 }, method: "inverted_cdf", weights: ones);
        D(r).Should().AllSatisfy(v => double.IsNaN(v).Should().BeTrue());
    }

    /// <summary>
    ///     out=: the plain forms check its shape (<c>Wrong shape of argument 'out', shape=(2,) is required; got
    ///     shape=(3,).</c>); the nan forms' per-position store does not (a <c>(3, 2)</c> out of a 2-row reduction is
    ///     written column by column).
    /// </summary>
    [TestMethod]
    public void Out_PlainChecksTheShape_NanFormsStorePerPosition()
    {
        using var m = np.array(new double[,] { { 3, 1, 2 }, { 6, 5, 4 } });
        using var w = np.ones(new Shape(2, 3));
        using var three = np.zeros(new Shape(3));
        Action wrong = () => np.quantile(m, 0.5, 1, three, method: "inverted_cdf", weights: w);
        wrong.Should().Throw<ValueError>().WithMessage("Wrong shape of argument 'out', shape=(2,) is required; got shape=(3,).");
        using var wide = np.zeros(new Shape(3, 2));
        var r = np.nanquantile(m, 0.5, 1, wide, method: "inverted_cdf", weights: w);
        ReferenceEquals(r, wide).Should().BeTrue();
        D(wide).Should().Equal(2, 5, 2, 5, 2, 5);
    }

    /// <summary>
    ///     Text data: the weighted branch sorts and takes, so <c>S</c> / <c>U</c> / <c>StringDType</c> compute
    ///     (<c>np.quantile(['b', 'a', 'c', 'd'], [.1, .9], weights=[1, 1, 5, 1])</c> is <c>['a', 'd']</c>); the nan forms
    ///     refuse fixed-width text at <c>isnan</c>; a chararray operand keeps its class.
    /// </summary>
    [TestMethod]
    public void Text_Computes_NanFormsRefuseFixedWidth_ChararrayKeepsItsClass()
    {
        using var t = np.array(new[] { "b", "a", "c", "d" });
        using var w = np.array(new long[] { 1, 1, 5, 1 });
        using var r = np.quantile(t, new[] { 0.1, 0.9 }, method: "inverted_cdf", weights: w);
        r.dtype.Should().Be(np.dtype("<U1"));
        NDArray.AsStringArray(r).Should().Equal("a", "d");
        using var vs = t.astype(np.dtype("T"));
        using var rv = np.nanquantile(vs, new[] { 0.1, 0.9 }, method: "inverted_cdf", weights: w);
        rv.ToString().Should().Be("['a' 'd']");
        Action fixedNan = () => np.nanquantile(t, 0.5, method: "inverted_cdf", weights: w);
        fixedNan.Should().Throw<TypeError>().WithMessage("ufunc 'isnan' not supported for the input types*");
        using var ch = np.@char.array(new[] { "b ", "a", "c" });
        using var ones = np.ones(new Shape(3));
        using var rc = np.quantile(ch, 0.5, method: "inverted_cdf", weights: ones);
        rc.Should().BeOfType<NDCharArray>();
    }

    /// <summary>
    ///     Empty data: the plain forms meet <c>cdf[-1, ...]</c> (<c>index -1 is out of bounds for axis 0 with size 0</c>);
    ///     the nan forms answer <c>np.nanmean</c> after the weights checks.
    /// </summary>
    [TestMethod]
    public void EmptyData()
    {
        using var e = np.array(new double[0]);
        using var we = np.array(new double[0]);
        Action plain = () => np.quantile(e, 0.5, method: "inverted_cdf", weights: we);
        plain.Should().Throw<IndexError>().WithMessage("index -1 is out of bounds for axis 0 with size 0");
        using var nan = np.nanquantile(e, 0.5, method: "inverted_cdf", weights: we);
        double.IsNaN(D(nan)[0]).Should().BeTrue();
        using var two = np.array(new[] { 1.0, 2 });
        Action checkFirst = () => np.nanquantile(e, 0.5, method: "inverted_cdf", weights: two);
        checkFirst.Should().Throw<TypeError>().WithMessage("Axis must be specified when shapes of a and weights differ.");
    }

    /// <summary>
    ///     One line searches the cumulative weights without normalizing them first when the search compares in float64 (a
    ///     float64, integer or bool q), and with NumPy's statements otherwise (a float16 / float32 q searches the cdf cast
    ///     to its dtype). Both give NumPy's indices at every boundary: data <c>[30, 10, 50, 20, 40]</c> with weights
    ///     <c>[1, 0, 2, 0, 1]</c> has the sorted cdf <c>[0, 0, .25, .5, 1]</c>, whose zeros count as -1 (q = 0 skips them)
    ///     and whose entries .25 and .5 are hit exactly; a tiny first weight beside huge ones divides to 0.0 and counts as
    ///     -1 too; a q equal to a cdf entry k / 55 picks that entry.
    /// </summary>
    [TestMethod]
    public void OneLine_SearchesLikeNumPy_AtEveryBoundary()
    {
        using var a = np.array(new[] { 30.0, 10, 50, 20, 40 });
        using var w = np.array(new[] { 1.0, 0, 2, 0, 1 });
        using var all = np.quantile(a, new[] { 0.0, 0.25, 0.5, 0.75, 1.0 }, method: "inverted_cdf", weights: w);
        D(all).Should().Equal(30, 30, 40, 50, 50);
        using var qInt = np.array(new long[] { 0, 1 });
        using var rInt = np.quantile(a, qInt, method: "inverted_cdf", weights: w);
        D(rInt).Should().Equal(30, 50);
        using var qBool = np.array(new[] { false, true });
        using var rBool = np.quantile(a, qBool, method: "inverted_cdf", weights: w);
        D(rBool).Should().Equal(30, 50);
        using var qU64 = np.array(new ulong[] { 0, 1 });
        using var rU64 = np.quantile(a, qU64, method: "inverted_cdf", weights: w);
        D(rU64).Should().Equal(30, 50);
        using var q0d = np.array(0L);
        using var r0d = np.quantile(a, q0d, method: "inverted_cdf", weights: w);
        r0d.ndim.Should().Be(0);
        D(r0d).Should().Equal(30);
        // The cast route: a float32 / float16 q searches the cdf rounded to its dtype, with NumPy's -1 rule after the cast.
        using var q32 = np.array(new[] { 0f, 0.25f, 0.5f });
        using var r32 = np.quantile(a, q32, method: "inverted_cdf", weights: w);
        D(r32).Should().Equal(30, 30, 40);
        using var q16 = np.array(new[] { (Half)0, (Half)0.25, (Half)0.5 });
        using var r16 = np.quantile(a, q16, method: "inverted_cdf", weights: w);
        D(r16).Should().Equal(30, 30, 40);
        // 5e-324 / 1e300 is 0.0: NumPy's `cdf[cdf == 0] = -1` takes it although the weight is not zero.
        using var b = np.array(new[] { 1.0, 2, 3 });
        using var wTiny = np.array(new[] { 5e-324, 1e300, 1e300 });
        using var rTiny = np.quantile(b, new[] { 0.0, 1e-310, 0.5, 1.0 }, method: "inverted_cdf", weights: wTiny);
        D(rTiny).Should().Equal(2, 2, 2, 3);
        using var c = np.arange(10.0);
        using var wSteps = np.arange(1.0, 11.0);
        using var rExact = np.quantile(c, new[] { 1 / 55.0, 3 / 55.0, 0.5, 54 / 55.0 }, method: "inverted_cdf", weights: wSteps);
        D(rExact).Should().Equal(0, 1, 6, 9);
    }

    /// <summary>
    ///     The chararray rules of the weighted call. Chararray weights refuse at NumPy's <c>weights &lt; 0</c>, which is the
    ///     chararray's own comparison (<c>comparison of non-string arrays</c>, whatever the text holds), after the shape
    ///     checks. A chararray of data in the nan forms gets the plain text's answer: <c>isnan</c> refuses it.
    /// </summary>
    [TestMethod]
    public void Chararray_WeightsRefuseAtTheComparison_DataInTheNanFormsRefusesAtIsnan()
    {
        using var a = np.array(new[] { 1.0, 2, 3 });
        using var cw = np.@char.array(new[] { "1", "2", "3" });
        Action plain = () => np.quantile(a, 0.5, method: "inverted_cdf", weights: cw);
        plain.Should().Throw<TypeError>().WithMessage("comparison of non-string arrays");
        Action nan = () => np.nanquantile(a, 0.5, method: "inverted_cdf", weights: cw);
        nan.Should().Throw<TypeError>().WithMessage("comparison of non-string arrays");
        using var two = np.@char.array(new[] { "1", "2" });
        Action shapeFirst = () => np.quantile(a, 0.5, method: "inverted_cdf", weights: two);
        shapeFirst.Should().Throw<TypeError>().WithMessage("Axis must be specified when shapes of a and weights differ.");
        using var ca = np.@char.array(new[] { "b", "a", "c" });
        using var ones = np.ones(new Shape(3));
        Action data = () => np.nanquantile(ca, 0.5, method: "inverted_cdf", weights: ones);
        data.Should().Throw<TypeError>().WithMessage("ufunc 'isnan' not supported for the input types*");
        Action both = () => np.quantile(ca, 0.5, method: "inverted_cdf", weights: cw);
        both.Should().Throw<TypeError>().WithMessage("comparison of non-string arrays");
    }
}
