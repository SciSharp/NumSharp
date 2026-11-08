using System;
using System.Linq;

namespace NumSharp.Tests.Statistics;

/// <summary>
///     The q forms of <c>np.quantile</c> / <c>np.percentile</c> / <c>np.nanquantile</c> / <c>np.nanpercentile</c>, NumPy
///     2.4.2's statements (every expected value probed against <c>numpy==2.4.2</c>):
///     <list type="bullet">
///       <item>a 2-d q keeps both axes in front of the result; rank 3 and above is refused by
///       <c>_quantile_ureduce_func</c> after the axis checks — except by the nan forms over data that is NaN everywhere,
///       whose every slice answers <c>np.full(q.shape, nan)</c> before reaching the check;</item>
///       <item><c>np.nanquantile</c>'s <c>q = np.asanyarray(q)</c>: an integer q makes 'linear''s virtual index an integer,
///       which NumPy takes instead of interpolating (the data's dtype), and an array q is a strong float64 weight;</item>
///       <item>the nan forms' all-NaN slices answer in the data's dtype, and <c>np.apply_along_axis</c> allocates from the
///       FIRST slice's answer;</item>
///       <item>the method name is looked up in <c>_quantile</c>, after the axis and rank checks.</item>
///     </list>
/// </summary>
[TestClass]
public class np_quantile_family_QKinds_Tests
{
    /// <summary>The elements of an array as doubles in C order (float16 read through its own conversion).</summary>
    /// <param name="a">The array.</param>
    /// <returns>The values.</returns>
    private static double[] D(NDArray a)
    {
        using var dense = a.copy();
        return Enumerable.Range(0, (int)dense.size)
            .Select(i => dense.Storage.GetAtIndex(i) is Half h ? (double)h : Convert.ToDouble(dense.Storage.GetAtIndex(i)))
            .ToArray();
    }

    /// <summary>A 2-d q of all four functions keeps both axes: <c>(2, 2)</c> of <c>[0.9, 4.5, 8.1, 9.0]</c>.</summary>
    [TestMethod]
    public void TwoDimensionalQ_KeepsBothAxes()
    {
        using var a = np.arange(10.0);
        using var q = np.array(new[,] { { 0.1, 0.5 }, { 0.9, 1.0 } });
        using var pct = q * 100;
        foreach (var r in new[] { np.quantile(a, q), np.nanquantile(a, q), np.percentile(a, pct), np.nanpercentile(a, pct) })
        {
            r.shape.Should().Equal(2L, 2L);
            D(r).Should().Equal(0.9, 4.5, 8.1, 9.0);
            r.Dispose();
        }
        using var @out = np.empty(new Shape(2, 2));
        np.quantile(a, q, @out: @out);
        D(@out).Should().Equal(0.9, 4.5, 8.1, 9.0);
    }

    /// <summary>Along an axis, q's two axes come first, then the kept axes; keepdims keeps the reduced one.</summary>
    [TestMethod]
    public void TwoDimensionalQ_AlongAnAxis_AndKeepdims()
    {
        using var m = np.arange(12.0).reshape(3, 4);
        using var q = np.array(new[,] { { 0.1, 0.5 }, { 0.9, 1.0 } });
        using var r = np.quantile(m, q, 1);
        r.shape.Should().Equal(2L, 2L, 3L);
        D(r).Should().Equal(0.30000000000000004, 4.3, 8.3, 1.5, 5.5, 9.5, 2.7, 6.7, 10.7, 3.0, 7.0, 11.0);
        using var kept = np.quantile(m, q, 1, keepdims: true);
        kept.shape.Should().Equal(2L, 2L, 3L, 1L);
        using var nan = np.nanquantile(np.array(new[,] { { 1.0, double.NaN, 3.0 }, { 4.0, 5.0, 6.0 } }), q, 1);
        nan.shape.Should().Equal(2L, 2L, 2L);
        D(nan).Should().Equal(1.2, 4.2, 2.0, 5.0, 2.8, 5.8, 3.0, 6.0);
    }

    /// <summary>
    ///     A q of rank 3 is refused after the axis checks (an out-of-range axis wins), but the nan forms over data that is
    ///     NaN everywhere answer NaN in the data's dtype, and over empty data <c>np.nanmean</c>'s NaN.
    /// </summary>
    [TestMethod]
    public void ThreeDimensionalQ_IsRefusedAfterTheAxes_ButNotByAllNaNSlices()
    {
        using var q3 = np.zeros(new Shape(2, 2, 2));
        Action plain = () => np.quantile(np.arange(10.0), q3);
        plain.Should().Throw<ValueError>().WithMessage("q must be a scalar or 1d");
        Action badAxis = () => np.quantile(np.arange(3.0), q3, 3);
        badAxis.Should().Throw<AxisError>();
        Action emptyData = () => np.quantile(np.zeros(new Shape(0)), q3);
        emptyData.Should().Throw<ValueError>().WithMessage("q must be a scalar or 1d");
        using var allNaN = np.nanquantile(np.full(new Shape(4), double.NaN), q3);
        allNaN.shape.Should().Equal(2L, 2L, 2L);
        allNaN.typecode.Should().Be(NPTypeCode.Double);
        D(allNaN).Should().OnlyContain(v => double.IsNaN(v));
        using var allNaN32 = np.nanquantile(np.full(new Shape(4), float.NaN), q3);
        allNaN32.typecode.Should().Be(NPTypeCode.Single);
        Action ints = () => np.nanquantile(np.arange(3), q3);
        ints.Should().Throw<ValueError>().WithMessage("q must be a scalar or 1d");
        using var empty = np.nanquantile(np.zeros(new Shape(0)), q3);
        empty.ndim.Should().Be(0);
        double.IsNaN(D(empty)[0]).Should().BeTrue();
    }

    /// <summary>
    ///     <c>np.nanquantile</c> with an integer q takes the element (NumPy's "supports_integers" path): the data's dtype,
    ///     and <c>inf</c> where interpolating at 1.0 makes <c>inf - inf</c>.
    /// </summary>
    [TestMethod]
    public void NanQuantile_IntegerQ_TakesTheElement()
    {
        using var ints = np.array(new long[] { 1, 2, 3 });
        using (var r = np.nanquantile(ints, 1))
        {
            r.typecode.Should().Be(NPTypeCode.Int64);
            D(r).Should().Equal(3.0);
        }
        using (var r = np.nanquantile(ints, np.array(new long[] { 0, 1 })))
        {
            r.typecode.Should().Be(NPTypeCode.Int64);
            D(r).Should().Equal(1.0, 3.0);
        }
        using (var r = np.nanquantile(ints, np.array(true)))
            D(r).Should().Equal(3.0);
        using var inf = np.array(new[] { 1.0, double.PositiveInfinity, double.NaN });
        using (var r = np.nanquantile(inf, 1))
            D(r).Should().Equal(double.PositiveInfinity);
        using (var r = np.nanquantile(inf, 1.0))
            double.IsNaN(D(r)[0]).Should().BeTrue();
        using var m = np.array(new[,] { { 1.0, double.NaN, 3.0 }, { 4.0, 5.0, double.NaN } });
        using (var r = np.nanquantile(m, 1, new[] { 0, 1 }))
            D(r).Should().Equal(5.0);
    }

    /// <summary>
    ///     A Python float q is a weak weight (float32 data stays float32), an array q a strong float64 one — for the nan
    ///     forms as for np.quantile.
    /// </summary>
    [TestMethod]
    public void NanForms_ArrayQ_IsAStrongWeight()
    {
        using var f32 = np.array(new float[] { 1, 2, float.NaN, 3 });
        using (var r = np.nanquantile(f32, 0.5))
            r.typecode.Should().Be(NPTypeCode.Single);
        using (var r = np.nanquantile(f32, np.array(0.5)))
            r.typecode.Should().Be(NPTypeCode.Double);
        using (var r = np.nanpercentile(f32, 50))
            r.typecode.Should().Be(NPTypeCode.Single);
        using (var r = np.nanpercentile(f32, np.array(50L)))
            r.typecode.Should().Be(NPTypeCode.Double);
    }

    /// <summary>
    ///     The nan forms' all-NaN slice answers in the data's dtype and <c>np.apply_along_axis</c> allocates from the FIRST
    ///     slice: float32 data with a strong q gives float32 when the first row is all NaN, float64 when another row is.
    /// </summary>
    [TestMethod]
    public void NanForms_AllNaNFirstSlice_GivesTheDataDtype()
    {
        using var m32 = np.array(new float[,] { { float.NaN, float.NaN }, { 1, 2 } });
        using var q = np.array(0.5);
        using (var r = np.nanquantile(m32, q, 1))
        {
            r.typecode.Should().Be(NPTypeCode.Single);
            double.IsNaN(D(r)[0]).Should().BeTrue();
            D(r)[1].Should().Be(1.5);
        }
        using var flipped = m32["::-1"];
        using (var r = np.nanquantile(flipped, q, 1))
            r.typecode.Should().Be(NPTypeCode.Double);
        using (var r = np.nanquantile(np.full(new Shape(3), float.NaN), q))
            r.typecode.Should().Be(NPTypeCode.Single);
    }

    /// <summary>
    ///     The method name is looked up in <c>_quantile</c>: an out-of-range axis and a rank-3 q are reported first, a bool
    ///     operand's subtract after it, and the nan forms never look it up when every slice is all NaN.
    /// </summary>
    [TestMethod]
    public void MethodName_IsCheckedWhereNumPyLooksItUp()
    {
        Action badAxis = () => np.quantile(np.arange(3.0), 0.5, 3, method: "bogus");
        badAxis.Should().Throw<AxisError>();
        Action rank3 = () => np.quantile(np.arange(3.0), np.zeros(new Shape(2, 2, 2)), method: "bogus");
        rank3.Should().Throw<ValueError>().WithMessage("q must be a scalar or 1d");
        Action boolData = () => np.quantile(np.array(new[] { true }), 0.5, method: "bogus");
        boolData.Should().Throw<ValueError>().WithMessage("'bogus' is not a valid method.*");
        Action range = () => np.quantile(np.arange(3.0), 2.0, method: "bogus");
        range.Should().Throw<ValueError>().WithMessage("Quantiles must be in the range [0, 1]");
        using var allNaN = np.nanquantile(np.full(new Shape(4), double.NaN), 0.5, method: "bogus");
        double.IsNaN(D(allNaN)[0]).Should().BeTrue();
    }

    /// <summary>The nan forms take an array q with a tuple axis (NumPy's <c>axis=(0, 1)</c>).</summary>
    [TestMethod]
    public void NanForms_ArrayQ_WithATupleAxis()
    {
        using var m = np.array(new[,] { { 1.0, double.NaN, 3.0 }, { 4.0, 5.0, double.NaN } });
        using (var r = np.nanquantile(m, np.array(new[] { 0.5, 1.0 }), new[] { 0, 1 }))
            D(r).Should().Equal(3.5, 5.0);
        using (var r = np.nanpercentile(m, np.array(new[] { 50.0, 100.0 }), new[] { 0, 1 }))
            D(r).Should().Equal(3.5, 5.0);
    }
}
