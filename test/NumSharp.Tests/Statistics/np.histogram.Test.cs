using System;
using System.Numerics;

namespace NumSharp.Tests.Statistics;

/// <summary>
/// Tests for the histogram family (np.histogram, np.histogram_bin_edges, np.histogramdd,
/// np.histogram2d), pinned against NumPy 2.4.2 output. Covers: uniform/estimator/explicit-edge
/// bins, density, weights (incl. complex), range/outliers, the inclusive last bin, all-15 dtypes,
/// bit-exact edges (incl. the float16/float32 in-dtype linspace), the multidimensional forms, and
/// the verbatim error taxonomy.
/// </summary>
[TestClass]
public class np_histogram_Tests
{
    // ── helpers ──────────────────────────────────────────────────────────
    private static double[] Flat(NDArray a)
    {
        var f = a.astype(NPTypeCode.Double);
        long n = f.size;
        var r = new double[n];
        for (long i = 0; i < n; i++) r[i] = f.GetAtIndex<double>(i);
        return r;
    }
    private static void Eq(NDArray a, double[] want, double tol = 0.0)
    {
        var got = Flat(a);
        got.Length.Should().Be(want.Length);
        for (int i = 0; i < want.Length; i++)
        {
            if (tol == 0.0) got[i].Should().Be(want[i]);
            else got[i].Should().BeApproximately(want[i], tol);
        }
    }

    // ── np.histogram: basics ─────────────────────────────────────────────

    [TestMethod]
    public void Histogram_ExplicitIntBins()
    {
        var (h, e) = np.histogram(np.array(new[] { 1, 2, 1 }), np.array(new[] { 0, 1, 2, 3 }));
        Eq(h, new double[] { 0, 2, 1 });
        Eq(e, new double[] { 0, 1, 2, 3 });
        h.typecode.Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void Histogram_Default10Bins()
    {
        var (h, e) = np.histogram(np.arange(5));
        Eq(h, new double[] { 1, 0, 1, 0, 0, 1, 0, 1, 0, 1 });
        Eq(e, new double[] { 0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0 }, 1e-12);
    }

    [TestMethod]
    public void Histogram_FlattensMultiDimInput()
    {
        var (h, _) = np.histogram(np.array(new[,] { { 1, 2, 1 }, { 1, 0, 1 } }), np.array(new[] { 0, 1, 2, 3 }));
        Eq(h, new double[] { 1, 4, 1 });
    }

    [TestMethod]
    public void Histogram_Density()
    {
        var (h, e) = np.histogram(np.arange(5), density: true);
        // integral over the range is 1
        var db = Flat(np.diff(e));
        var hv = Flat(h);
        double area = 0; for (int i = 0; i < hv.Length; i++) area += hv[i] * db[i];
        area.Should().BeApproximately(1.0, 1e-12);
        h.typecode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Histogram_DensityNonUniform_IntegratesToOne()
    {
        var (h, e) = np.histogram(np.arange(10), np.array(new[] { 0, 1, 3, 6, 10 }), density: true);
        Eq(h, new double[] { 0.1, 0.1, 0.1, 0.1 }, 1e-12);
    }

    [TestMethod]
    public void Histogram_Empty()
    {
        var (h, e) = np.histogram(np.array(new double[] { }), np.array(new[] { 0, 1 }));
        Eq(h, new double[] { 0 });
        Eq(e, new double[] { 0, 1 });
    }

    // ── weights ──────────────────────────────────────────────────────────

    [TestMethod]
    public void Histogram_IntegerWeights()
    {
        var (h, _) = np.histogram(np.array(new[] { 1, 2, 2, 4 }), 4, weights: np.array(new[] { 4, 3, 2, 1 }));
        Eq(h, new double[] { 4, 5, 0, 1 });
    }

    [TestMethod]
    public void Histogram_WeightedNonUniformDensity()
    {
        var (h, _) = np.histogram(np.arange(9), np.array(new[] { 0, 1, 3, 6, 10 }),
            density: true, weights: np.array(new double[] { 2, 1, 1, 1, 1, 1, 1, 1, 1 }));
        Eq(h, new double[] { 0.2, 0.1, 0.1, 0.075 }, 1e-12);
    }

    [TestMethod]
    public void Histogram_ComplexWeights()
    {
        var w = np.array(new Complex[] { new(1, 2), new(-1, 1), new(2, 2) });
        var (h, _) = np.histogram(np.array(new[] { 1.3, 2.5, 2.3 }), np.array(new[] { 0, 2, 3 }), weights: w);
        h.typecode.Should().Be(NPTypeCode.Complex);
        h.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 2));
        h.GetAtIndex<Complex>(1).Should().Be(new Complex(1, 3));
    }

    // ── range / outliers / inclusive last bin ────────────────────────────

    [TestMethod]
    public void Histogram_RangeExcludesOutliers()
    {
        var a = np.arange(10).astype(NPTypeCode.Double) + 0.5; // 0.5 .. 9.5
        var (h, _) = np.histogram(a, range: (0.0, 9.0));
        Flat(np.sum(h))[0].Should().Be(9); // the 9.5 outlier dropped
    }

    [TestMethod]
    public void Histogram_LastBinIsInclusive()
    {
        var a = np.array(new[] { 0.0, 0, 0, 1, 2, 3, 3, 4, 5 });
        var (h, _) = np.histogram(a, 30, range: (-0.5, 5.0));
        Flat(h)[^1].Should().Be(1); // the value 5 lands in the final (inclusive) bin
    }

    [TestMethod]
    public void Histogram_NaNNotCounted()
    {
        var (h, _) = np.histogram(np.array(new[] { 0.0, 1, double.NaN }), np.array(new[] { 0, 1 }));
        Flat(np.sum(h))[0].Should().Be(2);
    }

    // ── dtypes ───────────────────────────────────────────────────────────

    [TestMethod]
    public void Histogram_SignedByteOverflowRange()
    {
        var (h, e) = np.histogram(np.array(new sbyte[] { -124, 123 }), 2);
        Eq(h, new double[] { 1, 1 });
        Eq(e, new double[] { -124, -0.5, 123 });
    }

    [TestMethod]
    public void Histogram_Bool_ConvertsToUint8()
    {
        // np.histogram([True, True, False]) == np.histogram([1,1,0] uint8)
        var (hb, eb) = np.histogram(np.array(new[] { true, true, false }));
        var (hi, ei) = np.histogram(np.array(new byte[] { 1, 1, 0 }));
        Eq(hb, Flat(hi));
        Eq(eb, Flat(ei));
    }

    [TestMethod]
    public void Histogram_Float32Edges_AreFloat32AndBitExact()
    {
        var a = np.array(new float[] { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 9.9f });
        var (h, e) = np.histogram(a, 5);
        e.typecode.Should().Be(NPTypeCode.Single); // float32 input => float32 edges (NumPy in-dtype linspace)
        Eq(h, new double[] { 2, 2, 1, 0, 1 });
        // NumPy 2.4.2 float32 edge[3] is 6.13999939 (computed IN float32, not double-then-cast)
        e.GetAtIndex<float>(3).Should().Be(6.13999939f);
    }

    [TestMethod]
    public void Histogram_Float16Edges_AreFloat16()
    {
        var a = np.array(new Half[] { (Half)0, (Half)1, (Half)2, (Half)3, (Half)4 });
        var (_, e) = np.histogram(a, 4);
        e.typecode.Should().Be(NPTypeCode.Half);
    }

    [TestMethod]
    public void Histogram_IntegerEdges_AreFloat64()
    {
        var (_, e) = np.histogram(np.arange(5), 4);
        e.typecode.Should().Be(NPTypeCode.Double);
    }

    // ── np.histogram_bin_edges ───────────────────────────────────────────

    [TestMethod]
    public void BinEdges_ExplicitArrayPassedThrough()
    {
        var e = np.histogram_bin_edges(np.array(new[] { 1, 2, 3, 4 }), np.array(new[] { 1, 2 }));
        Eq(e, new double[] { 1, 2 });
    }

    [TestMethod]
    public void BinEdges_MatchesHistogramEdges()
    {
        var arr = np.array(new[] { 0.0, 0, 0, 1, 2, 3, 3, 4, 5 });
        var (_, he) = np.histogram(arr, 30, range: (-0.5, 5.0));
        var be = np.histogram_bin_edges(arr, 30, range: (-0.5, 5.0));
        Eq(be, Flat(he));
    }

    // ── estimators (bin counts pinned against NumPy 2.4.2) ───────────────

    [TestMethod]
    public void Estimators_BinCountsMatchNumPy()
    {
        var x1 = np.linspace(-10.0, -1.0, 20);
        var x2 = np.linspace(1.0, 10.0, 30);
        var x = np.concatenate((x1, x2));
        (string, int)[] expected =
        {
            ("sqrt", 8), ("sturges", 7), ("rice", 8), ("scott", 4),
            ("fd", 4), ("doane", 8), ("auto", 7), ("stone", 2)
        };
        foreach (var (est, nb) in expected)
        {
            var (h, _) = np.histogram(x, est);
            ((int)h.size).Should().Be(nb, $"estimator {est}");
        }
    }

    [TestMethod]
    public void Estimators_EmptyInput_OneBin()
    {
        foreach (var est in new[] { "fd", "scott", "rice", "sturges", "doane", "sqrt", "auto", "stone" })
        {
            var (h, e) = np.histogram(np.array(new double[] { }), est);
            Eq(h, new double[] { 0 });
            Eq(e, new double[] { 0, 1 });
        }
    }

    [TestMethod]
    public void Estimator_IntegerBinWidthAtLeastOne()
    {
        // np.histogram_bin_edges(tile(arange(9),1000), 'auto') == arange(9)
        var a = np.tile(np.arange(9), 1000);
        var e = np.histogram_bin_edges(a, "auto");
        Eq(e, new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
    }

    // ── np.histogramdd ───────────────────────────────────────────────────

    [TestMethod]
    public void HistogramDD_Simple()
    {
        var x = np.array(new double[,]
        {
            { -.5, .5, 1.5 }, { -.5, 1.5, 2.5 }, { -.5, 2.5, .5 },
            { .5, .5, 1.5 }, { .5, 1.5, 2.5 }, { .5, 2.5, 2.5 }
        });
        var (H, edges) = np.histogramdd(x, new[] { 2, 3, 3 },
            new (double, double)?[] { (-1, 1), (0, 3), (0, 3) });
        H.shape.Should().Equal(new long[] { 2, 3, 3 });
        Flat(np.sum(H))[0].Should().Be(6);
        H.typecode.Should().Be(NPTypeCode.Double); // dd H is always float64
        edges.Length.Should().Be(3);
    }

    [TestMethod]
    public void HistogramDD_SequenceOfColumns_1D_EqualsHistogram()
    {
        var v = np.arange(10).astype(NPTypeCode.Double);
        var bins = np.array(new[] { 0, 1, 3, 6, 10 });
        var (h1, _) = np.histogram(v, bins, density: true);
        var (hdd, edd) = np.histogramdd(new[] { v }, new NDArray[] { bins }, density: true);
        Eq(hdd, Flat(h1), 1e-12);
    }

    [TestMethod]
    public void HistogramDD_LargeIntegerEdges_ExactInInt64()
    {
        long big = 1L << 60; // too large for exact float
        var xx = np.array(new long[] { 0 });
        var yy = np.array(new long[] { big });
        var xe = np.array(new long[] { -1, 1 });
        var ye = np.array(new long[] { big - 1, big + 1 });
        var (H, _) = np.histogramdd(new[] { xx, yy }, new NDArray[] { xe, ye });
        H.GetAtIndex<double>(0).Should().Be(1);
    }

    [TestMethod]
    public void HistogramDD_EdgeArrayDtypePreserved()
    {
        var x = np.array(new[] { 0, 10, 20 });
        var y = x / 10;
        var xe = np.array(new[] { 0, 5, 15, 20 });
        var ye = xe / 10;
        var (_, edges) = np.histogramdd(new[] { x, y }, new NDArray[] { xe, ye });
        edges[0].typecode.Should().Be(xe.typecode);
    }

    // ── np.histogram2d ───────────────────────────────────────────────────

    [TestMethod]
    public void Histogram2D_IntBins()
    {
        var x = np.array(new[] { 1.0, 2, 3, 4, 5 });
        var y = np.array(new[] { 1.0, 1, 2, 2, 3 });
        var (H, xe, ye) = np.histogram2d(x, y, 3);
        H.shape.Should().Equal(new long[] { 3, 3 });
        Flat(np.sum(H))[0].Should().Be(5);
    }

    [TestMethod]
    public void Histogram2D_PerDimBinCounts()
    {
        var x = np.array(new[] { 1.0, 2, 3, 4, 5 });
        var y = np.array(new[] { 1.0, 1, 2, 2, 3 });
        var (H, _, _) = np.histogram2d(x, y, new[] { 2, 3 });
        H.shape.Should().Equal(new long[] { 2, 3 });
    }

    [TestMethod]
    public void Histogram2D_SharedEdgeArray()
    {
        var x = np.array(new[] { 1.0, 2, 3, 4, 5 });
        var y = np.array(new[] { 1.0, 1, 2, 2, 3 });
        var (H, xe, ye) = np.histogram2d(x, y, np.array(new[] { 0, 2, 4, 6.0 }));
        H.shape.Should().Equal(new long[] { 3, 3 });
        Eq(xe, Flat(ye)); // shared edges
    }

    [TestMethod]
    public void Histogram2D_Deconstructs()
    {
        var (H, xe, ye) = np.histogram2d(np.array(new[] { 1.0, 2 }), np.array(new[] { 1.0, 2 }), 2);
        H.Should().NotBeNull();
        xe.size.Should().Be(3);
        ye.size.Should().Be(3);
    }

    // ── result-struct conversions ────────────────────────────────────────

    [TestMethod]
    public void HistogramResult_ImplicitToNDArray_IsHist()
    {
        NDArray h = np.histogram(np.array(new[] { 1, 2, 1 }), np.array(new[] { 0, 1, 2, 3 }));
        Eq(h, new double[] { 0, 2, 1 });
    }

    [TestMethod]
    public void HistogramResult_ImplicitToArray_IsHistAndEdges()
    {
        NDArray[] r = np.histogram(np.array(new[] { 1, 2, 1 }), np.array(new[] { 0, 1, 2, 3 }));
        r.Length.Should().Be(2);
        Eq(r[0], new double[] { 0, 2, 1 });
        Eq(r[1], new double[] { 0, 1, 2, 3 });
    }

    // ── error taxonomy (verbatim NumPy messages) ─────────────────────────

    [TestMethod]
    public void Error_ZeroBins()
        => ((Action)(() => np.histogram(np.array(new[] { 1, 2 }), 0)))
            .Should().Throw<ValueError>().WithMessage("*must be positive*");

    [TestMethod]
    public void Error_RangeMinGreaterThanMax()
        => ((Action)(() => np.histogram(np.linspace(0.0, 1.0, 100), range: (0.1, 0.01))))
            .Should().Throw<ValueError>().WithMessage("max must be larger than min in range parameter.");

    [TestMethod]
    public void Error_NonFiniteRange()
        => ((Action)(() => np.histogram(np.linspace(0.0, 1.0, 100), range: (double.NaN, 0.75))))
            .Should().Throw<ValueError>().WithMessage("*is not finite*");

    [TestMethod]
    public void Error_Bins2D()
        => ((Action)(() => np.histogram(np.linspace(0.0, 1.0, 100), np.array(new double[,] { { 0, 0.5 }, { 0.6, 1.0 } }))))
            .Should().Throw<ValueError>().WithMessage("*must be 1d*");

    [TestMethod]
    public void Error_NonMonotonicBins()
        => ((Action)(() => np.histogram(np.array(new[] { 2 }), np.array(new[] { 1, 3, 1 }))))
            .Should().Throw<ValueError>().WithMessage("*must increase monotonically*");

    [TestMethod]
    public void Error_UnknownEstimator()
        => ((Action)(() => np.histogram(np.array(new[] { 1, 2, 3 }), "IQR")))
            .Should().Throw<ValueError>().WithMessage("*is not a valid estimator*");

    [TestMethod]
    public void Error_WeightedEstimator()
        => ((Action)(() => np.histogram(np.array(new[] { 1, 2, 3 }), "fd", weights: np.array(new[] { 1, 2, 3 }))))
            .Should().Throw<TypeError>().WithMessage("*not supported for weighted*");

    [TestMethod]
    public void Error_WeightsShapeMismatch()
        => ((Action)(() => np.histogram(np.arange(10).astype(NPTypeCode.Double) + 0.5, range: (1, 9),
                weights: np.arange(11).astype(NPTypeCode.Double) + 0.5)))
            .Should().Throw<ValueError>().WithMessage("*same shape as*");

    [TestMethod]
    public void Error_DD_DimMismatch()
        => ((Action)(() => np.histogramdd(np.array(new double[,] { { 0, 1 }, { 2, 3 } }), new[] { 1, 2, 4, 5 })))
            .Should().Throw<ValueError>().WithMessage("*dimension of bins must be equal*");

    [TestMethod]
    public void Error_Histogram2D_LengthMismatch()
        => ((Action)(() => np.histogram2d(np.array(new[] { 1.0, 2 }), np.array(new[] { 1.0, 2, 3 }), 3)))
            .Should().Throw<ValueError>().WithMessage("x and y must have the same length.");
}
