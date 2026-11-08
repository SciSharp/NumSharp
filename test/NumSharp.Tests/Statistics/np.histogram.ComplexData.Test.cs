using System;
using System.Numerics;

namespace NumSharp.Tests.Statistics;

/// <summary>
///     np.histogram / np.histogram_bin_edges over complex data, as NumPy 2.4.2 answers: NumPy's statements meet complex data
///     in complex128 (the outer edges are the data's lexicographic minimum and maximum, the equal-width edges are complex,
///     the bin index is the real part of a complex quotient, and an estimator whose width or range is complex stops at
///     <c>int(np.ceil(...))</c>). Every expected value is NumPy's output.
/// </summary>
[TestClass]
public class np_histogram_complex_data_tests
{
    /// <summary>The probes' data: <c>complex(d, -d if i % 3 == 0 else 0.5 * i)</c> over twelve integers.</summary>
    private static NDArray Data()
    {
        double[] plain = { 3, -1, 2, 0, 5, -4, 1, 7, -2, 6, 4, -3 };
        var c = new Complex[plain.Length];
        for (int i = 0; i < c.Length; i++) c[i] = new Complex(plain[i], i % 3 == 0 ? -plain[i] : 0.5 * i);
        return np.array(c);
    }

    private const string CeilRefusal =
        "ufunc 'ceil' not supported for the input types, and the inputs could not be safely coerced to any supported types " +
        "according to the casting rule ''safe''";

    private static Complex[] C(NDArray a) => a.ToArray<Complex>();

    /// <summary>
    ///     A bin count: the edges are complex128, a linspace from the lexicographic minimum <c>(-4+2.5j)</c> to the maximum
    ///     <c>(7+3.5j)</c>, and the counts follow the complex index arithmetic.
    /// </summary>
    [TestMethod]
    public void BinCount_ComplexEdgesFromTheDataMinAndMax()
    {
        var (h, e) = np.histogram(Data(), 3);
        h.typecode.Should().Be(NPTypeCode.Int64);
        h.ToArray<long>().Should().Equal(4L, 4L, 4L);
        e.typecode.Should().Be(NPTypeCode.Complex);
        C(e).Should().Equal(new Complex(-4, 2.5), new Complex(-0.3333333333333335, 2.8333333333333335),
            new Complex(3.333333333333333, 3.1666666666666665), new Complex(7, 3.5));

        np.histogram(Data(), 10).hist.ToArray<long>().Should().Equal(2L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 2L);
        C(np.histogram_bin_edges(np.array(new[] { new Complex(1, 2), new Complex(3, -1) }), 2))
            .Should().Equal(new Complex(1, 2), new Complex(2, 0.5), new Complex(3, -1));
    }

    /// <summary>
    ///     A range gives Python floats: the edges are a float64 linspace cast to complex128 (zero imaginary parts), and the
    ///     data are compared with them lexicographically.
    /// </summary>
    [TestMethod]
    public void Range_EdgesAreTheFloat64LinspaceAsComplex()
    {
        var (h, e) = np.histogram(Data(), 3, range: (-1.0, 10.0));
        h.ToArray<long>().Should().Equal(4L, 4L, 1L);
        C(e).Should().Equal(new Complex(-1, 0), new Complex(2.6666666666666665, 0), new Complex(6.333333333333333, 0),
            new Complex(10, 0));
        np.histogram(Data(), 10, range: (-1.0, 10.0)).hist.ToArray<long>().Should().Equal(2L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 0L, 0L);
    }

    /// <summary>
    ///     Complex outer edges run linspace's complex128 statements (the step is a Smith division by <c>(div + 0j)</c>, a
    ///     multiplication by its reciprocal), so even with zero imaginary parts an edge can differ from the float64 one:
    ///     edge 2 of 26 bins over <c>[-10.101787042252369, 30.31859454455258]</c> is <c>-6.992526920190449</c> for complex
    ///     data and <c>-6.99252692019045</c> for float64 data.
    /// </summary>
    [TestMethod]
    public void ComplexEdges_ZeroImaginaryParts_KeepComplexArithmetic()
    {
        const double lo = -10.101787042252369, hi = 30.31859454455258;
        var complexEdges = np.histogram_bin_edges(np.array(new[] { new Complex(lo, 0), new Complex(hi, 0) }), 26);
        complexEdges.GetAtIndex<Complex>(2).Real.Should().Be(-6.992526920190449);
        np.histogram_bin_edges(np.array(new[] { lo, hi }), 26).GetAtIndex<double>(2).Should().Be(-6.99252692019045);
    }

    /// <summary>
    ///     The estimators: <c>fd</c> / <c>auto</c> stop at <c>np.percentile</c>; the peak-to-peak value of complex data is
    ///     complex, so <c>sqrt</c> / <c>sturges</c> / <c>rice</c> / <c>doane</c> / <c>stone</c> stop at the ceil with or
    ///     without a range; <c>scott</c>'s width is real (the standard deviation), so it stops there only without a range,
    ///     where the range from the data is complex.
    /// </summary>
    [TestMethod]
    public void Estimators_StopWhereNumPysComplexValuesMeetRealOnlyCode()
    {
        foreach (var est in new[] { "fd", "auto" })
        {
            ((Action)(() => np.histogram(Data(), est))).Should().Throw<TypeError>().WithMessage("a must be an array of real numbers");
            ((Action)(() => np.histogram(Data(), est, range: (-1.0, 10.0)))).Should().Throw<TypeError>()
                .WithMessage("a must be an array of real numbers");
        }
        foreach (var est in new[] { "sqrt", "sturges", "rice", "doane", "stone" })
        {
            ((Action)(() => np.histogram(Data(), est))).Should().Throw<TypeError>().WithMessage(CeilRefusal);
            ((Action)(() => np.histogram_bin_edges(Data(), est, range: (-1.0, 10.0)))).Should().Throw<TypeError>()
                .WithMessage(CeilRefusal);
        }
        ((Action)(() => np.histogram(Data(), "scott"))).Should().Throw<TypeError>().WithMessage(CeilRefusal);

        var (h, e) = np.histogram(Data(), "scott", range: (-1.0, 10.0));
        h.ToArray<long>().Should().Equal(6L, 3L);
        C(e).Should().Equal(new Complex(-1, 0), new Complex(4.5, 0), new Complex(10, 0));

        // The range trims the data lexicographically before the estimator: [1-1j, 2+0j] are left, one bin.
        var trimmed = np.array(new[] { new Complex(0, 5), new Complex(1, -1), new Complex(2, 0), new Complex(3, 3) });
        var (ht, et) = np.histogram(trimmed, "scott", range: (0.5, 2.0));
        ht.ToArray<long>().Should().Equal(2L);
        C(et).Should().Equal(new Complex(0.5, 0), new Complex(2, 0));
    }

    /// <summary>
    ///     A zero width ends in one bin before the ceil: <c>doane</c> with two samples, <c>sqrt</c> / <c>scott</c> over equal
    ///     samples (a <c>0j</c> peak-to-peak value, a zero deviation), <c>stone</c> with one sample. The single bin spans the
    ///     equal samples widened by 0.5.
    /// </summary>
    [TestMethod]
    public void Estimators_ZeroWidth_OneBin()
    {
        var (hd, ed) = np.histogram(np.array(new[] { new Complex(1, 1), new Complex(2, 2) }), "doane");
        hd.ToArray<long>().Should().Equal(2L);
        C(ed).Should().Equal(new Complex(1, 1), new Complex(2, 2));

        var equal = np.array(new[] { new Complex(1, 1), new Complex(1, 1), new Complex(1, 1) });
        foreach (var est in new[] { "sqrt", "scott" })
        {
            var (h, e) = np.histogram(equal, est);
            h.ToArray<long>().Should().Equal(3L);
            C(e).Should().Equal(new Complex(0.5, 1), new Complex(1.5, 1));
        }
        var (hs, es) = np.histogram(np.array(new[] { new Complex(1, 1) }), "stone");
        hs.ToArray<long>().Should().Equal(1L);
        C(es).Should().Equal(new Complex(0.5, 1), new Complex(1.5, 1));
    }

    /// <summary>A non-finite part of the data's minimum or maximum: NumPy prints the complex scalars in its message.</summary>
    [TestMethod]
    public void NonFiniteRange_MessagePrintsComplexScalars()
    {
        ((Action)(() => np.histogram(np.array(new[] { new Complex(1, 1), new Complex(double.NaN, 0) }), 2)))
            .Should().Throw<ValueError>().WithMessage("autodetected range of [(nan+0j), (nan+0j)] is not finite");
        ((Action)(() => np.histogram(np.array(new[] { new Complex(1, 1), new Complex(0, double.PositiveInfinity) }), 2)))
            .Should().Throw<ValueError>().WithMessage("autodetected range of [infj, (1+1j)] is not finite");
        ((Action)(() => np.histogram(np.array(new[] { new Complex(double.NaN, double.NaN) }), 2)))
            .Should().Throw<ValueError>().WithMessage("autodetected range of [(nan+nanj), (nan+nanj)] is not finite");
    }

    /// <summary>Equal samples widen by 0.5 on both sides in complex arithmetic, and land in the last bin.</summary>
    [TestMethod]
    public void EqualSamples_WidenByHalf()
    {
        var (h, e) = np.histogram(np.array(new[] { new Complex(1, 2), new Complex(1, 2), new Complex(1, 2), new Complex(1, 2) }), 2);
        h.ToArray<long>().Should().Equal(0L, 4L);
        C(e).Should().Equal(new Complex(0.5, 2), new Complex(1, 2), new Complex(1.5, 2));
    }

    /// <summary>No data: the outer edges are the ints 0 and 1, so a float64 linspace as complex; a 0-d sample.</summary>
    [TestMethod]
    public void EmptyAndZeroDimensionalData()
    {
        var (h, e) = np.histogram(np.array(new Complex[0]), 3);
        h.typecode.Should().Be(NPTypeCode.Int64);
        h.ToArray<long>().Should().Equal(0L, 0L, 0L);
        C(e).Should().Equal(new Complex(0, 0), new Complex(0.3333333333333333, 0), new Complex(0.6666666666666666, 0),
            new Complex(1, 0));
        var (ha, ea) = np.histogram(np.array(new Complex[0]), "auto");
        ha.ToArray<long>().Should().Equal(0L);
        C(ea).Should().Equal(new Complex(0, 0), new Complex(1, 0));

        var (h0, e0) = np.histogram(np.array(new Complex(2, 3)), 3);
        h0.ToArray<long>().Should().Equal(0L, 1L, 0L);
        C(e0).Should().Equal(new Complex(1.5, 3), new Complex(1.8333333333333333, 3), new Complex(2.1666666666666665, 3),
            new Complex(2.5, 3));
    }

    /// <summary>
    ///     Where the imaginary parts pull the complex quotient past the edges, NumPy's own statements fail: a negative index
    ///     reaches <c>np.bincount</c> (its <see cref="ValueError"/>), or the
    ///     fancy index into the edges is out of bounds. <c>stone</c> meets the first in its own histograms.
    /// </summary>
    [TestMethod]
    public void ComplexQuotientOutsideTheEdges_NumPysErrors()
    {
        var negative = np.array(new[] { new Complex(0, 0), new Complex(1, 10), new Complex(0.1, -5) });
        ((Action)(() => np.histogram(negative, 3))).Should().Throw<ValueError>()
            .WithMessage("'list' argument must have no negative elements");
        ((Action)(() => np.histogram(negative, "stone"))).Should().Throw<ValueError>()
            .WithMessage("'list' argument must have no negative elements");

        var beyond = np.array(new[] { new Complex(0, 10), new Complex(1, 0), new Complex(0.9, -10) });
        ((Action)(() => np.histogram(beyond, 3))).Should().Throw<IndexError>()
            .WithMessage("index 5 is out of bounds for axis 0 with size 4");
    }

    /// <summary>Weights, density and explicit edges over complex data.</summary>
    [TestMethod]
    public void WeightsDensityAndExplicitEdges()
    {
        var hw = np.histogram(Data(), 3, weights: np.ones(new Shape(12))).hist;
        hw.typecode.Should().Be(NPTypeCode.Double);
        hw.ToArray<double>().Should().Equal(4.0, 4.0, 4.0);

        var cw = np.array(new Complex[12]);
        for (int i = 0; i < 12; i++) cw.SetAtIndex(new Complex(1, 0.5), i);
        C(np.histogram(Data(), 3, weights: cw).hist).Should().Equal(new Complex(4, 2), new Complex(4, 2), new Complex(4, 2));

        var hd = np.histogram(Data(), 3, density: true).hist;
        hd.typecode.Should().Be(NPTypeCode.Double);
        hd.ToArray<double>().Should().Equal(0.09090909090909093, 0.09090909090909093, 0.0909090909090909);

        var (hr, er) = np.histogram(Data(), np.array(new[] { -2.0, 0.0, 1.5, 5.0 }));
        hr.ToArray<long>().Should().Equal(2L, 2L, 3L);
        er.typecode.Should().Be(NPTypeCode.Double);
        var complexEdges = np.array(new[] { new Complex(-2, 0.25), new Complex(0, 0.25), new Complex(1.5, 0.25), new Complex(5, 0.25) });
        np.histogram(Data(), complexEdges).hist.ToArray<long>().Should().Equal(3L, 1L, 3L);
    }
}
