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

    /// <summary>
    ///     NumSharp's char is an unsignedinteger (its scalar hierarchy; the oracle's uint16), so NumPy's
    ///     <c>np.issubdtype(a.dtype, np.integer) and width &lt; 1</c> clamps its estimated width to 1 as for uint16: the
    ///     codes 97..100 in twelve samples get NumPy's uint16 answer for every estimator (sqrt got 4 bins, auto / sturges /
    ///     rice / doane 5 and stone 100 without the clamp; fd and scott estimate a width of at least 1 anyway).
    /// </summary>
    [TestMethod]
    public void Estimators_CharData_ClampWidthLikeUInt16()
    {
        var chars = np.broadcast_to(np.array(new[] { 'd', 'b', 'c', 'a' }).reshape(1, 4), new Shape(3, 4));
        foreach (var est in new[] { "auto", "fd", "sturges", "sqrt", "rice", "doane", "stone" })
        {
            var (h, e) = np.histogram(chars, est);
            Eq(h, new double[] { 3, 3, 6 });
            Eq(e, new double[] { 97, 98, 99, 100 });
        }
        var (hs, es) = np.histogram(chars, "scott");
        Eq(hs, new double[] { 6, 6 });
        Eq(es, new double[] { 97, 98.5, 100 });
    }

    /// <summary>
    ///     int32 data spanning the whole int32 range: NumSharp gives Linux NumPy's answer. NumPy's <c>_unsigned_subtract</c>
    ///     computes <c>max - min</c> in the unsigned type its table maps the scalar type to, and int32's scalar type is C int
    ///     on LP64 (in the table: uint32, 2147483655) but C long on Windows (not in the table: a wrapping int32 subtraction,
    ///     -2147483641), so Windows NumPy 2.4.2 raises "'list' argument must have no negative elements" for bins=10 and
    ///     "Number of samples, -2147483640, must be non-negative." for the estimators. NumSharp follows the LP64 model, as it
    ///     does for C long elsewhere. Expected values: NumPy 2.4.2 on Linux.
    /// </summary>
    [TestMethod]
    public void Histogram_Int32FullRange_IsLinuxNumPysAnswer()
    {
        var a = np.array(new[] { 3, int.MinValue, 0, int.MinValue, 2, -1, int.MinValue, 0, 7, -2, int.MinValue, -6 });
        var (h, e) = np.histogram(a);
        Eq(h, new double[] { 4, 0, 0, 0, 0, 0, 0, 0, 0, 8 });
        Eq(e, new[] { -2147483648.0, -1932735282.5, -1717986917.0, -1503238551.5, -1288490186.0, -1073741820.5,
                      -858993455.0, -644245089.5, -429496724.0, -214748358.5, 7.0 });

        var (ha, ea) = np.histogram(a, "auto");
        Eq(ha, new double[] { 4, 0, 0, 0, 8 });
        Eq(ea, new[] { -2147483648.0, -1717986917.0, -1288490186.0, -858993455.0, -429496724.0, 7.0 });

        var (hs, es) = np.histogram(a, "sqrt");
        Eq(hs, new double[] { 4, 0, 0, 8 });
        Eq(es, new[] { -2147483648.0, -1610612734.25, -1073741820.5, -536870906.75, 7.0 });

        Eq(np.histogram(a, "fd").hist, new double[] { 4, 8 });
        Eq(np.histogram(a, "scott").hist, new double[] { 4, 8 });
        Eq(np.histogram(a, "doane").hist, new double[] { 4, 0, 0, 0, 0, 8 });
    }

    /// <summary>
    ///     scott's width is a standard deviation in the data's dtype; when its squares overflow the width is infinite, the
    ///     range is not, and <c>int(np.ceil(delta / width))</c> is 0 bins. NumPy's index statements then give every sample
    ///     the index -1 and its np.bincount refuses them (NumSharp's np.bincount raises that ValueError text as an
    ///     <see cref="ArgumentException"/>). NumSharp's fused count kernel used to count the samples outside the zero-length
    ///     result: a heap corruption (STATUS_HEAP_CORRUPTION, found by a random differential against NumPy 2.4.2).
    /// </summary>
    [TestMethod]
    public void Estimator_ScottInfiniteWidth_ZeroBins_BincountRefusal()
    {
        const string text = "'list' argument must have no negative elements";
        var f16 = np.array(new[] { -169.5, 166.625, 82.1875, -114.5625, -269.25, -26.0, 66.0 }).astype(NPTypeCode.Half);
        var f32 = np.array(new[] { -3e19f, 1e19f, 2e19f, 0f });
        var f64 = np.array(new[] { -3e154, 1e154, 2e154, 0 });
        foreach (var a in new[] { f16, f32, f64 })
            for (int rep = 0; rep < 200; rep++)
                ((Action)(() => np.histogram(a, "scott"))).Should().Throw<ArgumentException>().WithMessage(text);

        // histogram_bin_edges does not count: the single edge of zero bins comes back.
        np.histogram_bin_edges(f64, "scott").ToArray<double>().Should().Equal(-3e154);
    }

    /// <summary>
    ///     Zero bins have a single edge, and NumPy's linspace makes it the START: with one point it takes
    ///     <c>step = nan; y = y * delta</c> (0 * delta), adds start, and sets the last point to stop only when there are two
    ///     or more. The float16 / float32 edges (computed in the data's own width) used to come back as the stop.
    /// </summary>
    [TestMethod]
    public void Estimator_ZeroBins_SingleEdgeIsTheStart()
    {
        var f16 = np.array(new[] { -169.5, 166.625, 82.1875, -114.5625, -269.25, -26.0, 66.0 }).astype(NPTypeCode.Half);
        var e16 = np.histogram_bin_edges(f16, "scott");
        e16.typecode.Should().Be(NPTypeCode.Half);
        e16.ToArray<Half>().Should().Equal((Half)(-269.25));

        var e32 = np.histogram_bin_edges(np.array(new[] { -3e19f, 1e19f, 2e19f, 0f }), "scott");
        e32.typecode.Should().Be(NPTypeCode.Single);
        e32.ToArray<float>().Should().Equal(-3e19f);
    }

    /// <summary>
    ///     NumPy's <c>n_equal_bins = int(np.ceil(delta / width))</c> and the linspace allocation it leads to, for counts too
    ///     large to allocate: integer data whose fd width is clamped to 1 count one bin per unit of their range (60 zeros,
    ///     40 ones and one outlier), float data with a tiny interquartile range and a far outlier. NumPy 2.4.2's errors per
    ///     regime of the edge count N (MemoryError, "array is too big", np.arange's IndexError at a float64 N of 2^63,
    ///     "Maximum allowed size exceeded"); .NET's saturating conversion gave misleading linspace errors.
    /// </summary>
    [TestMethod]
    public void Estimator_HugeBinCounts_NumPysAllocationErrors()
    {
        static NDArray Int64Span(long span)
        {
            var v = new long[101];
            for (int i = 60; i < 100; i++) v[i] = 1;
            v[100] = span;
            return np.array(v);
        }
        static NDArray UInt64Span(ulong span)
        {
            var v = new ulong[101];
            for (int i = 60; i < 100; i++) v[i] = 1;
            v[100] = span;
            return np.array(v);
        }

        ((Action)(() => np.histogram_bin_edges(Int64Span(1L << 33), "fd"))).Should().Throw<OutOfMemoryException>()
            .WithMessage("Unable to allocate 64.0 GiB for an array with shape (8589934593,) and data type float64");
        ((Action)(() => np.histogram(Int64Span((1L << 34) + 12345), "fd"))).Should().Throw<OutOfMemoryException>()
            .WithMessage("Unable to allocate 128. GiB for an array with shape (17179881530,) and data type float64");
        ((Action)(() => np.histogram_bin_edges(Int64Span(1L << 40), "fd"))).Should().Throw<OutOfMemoryException>()
            .WithMessage("Unable to allocate 8.00 TiB for an array with shape (1099511627777,) and data type float64");
        ((Action)(() => np.histogram_bin_edges(Int64Span((1L << 52) + 7), "fd"))).Should().Throw<OutOfMemoryException>()
            .WithMessage("Unable to allocate 32.0 PiB for an array with shape (4503599627370504,) and data type float64");
        ((Action)(() => np.histogram_bin_edges(Int64Span(1L << 61), "fd"))).Should().Throw<ValueError>()
            .WithMessage("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.");
        ((Action)(() => np.histogram_bin_edges(Int64Span(long.MaxValue - 511), "fd"))).Should().Throw<IndexError>()
            .WithMessage("index -1 is out of bounds for axis 0 with size 0");
        ((Action)(() => np.histogram_bin_edges(UInt64Span(1UL << 63), "fd"))).Should().Throw<IndexError>()
            .WithMessage("index -1 is out of bounds for axis 0 with size 0");
        ((Action)(() => np.histogram_bin_edges(UInt64Span((1UL << 63) + (1UL << 11)), "fd"))).Should().Throw<ValueError>()
            .WithMessage("Maximum allowed size exceeded");
        ((Action)(() => np.histogram_bin_edges(UInt64Span(ulong.MaxValue), "fd"))).Should().Throw<ValueError>()
            .WithMessage("Maximum allowed size exceeded");
        ((Action)(() => np.histogram_bin_edges(UInt64Span(1UL << 40), "fd", range: (0.0, 1099511627776.0))))
            .Should().Throw<OutOfMemoryException>()
            .WithMessage("Unable to allocate 8.00 TiB for an array with shape (1099511627777,) and data type float64");

        static NDArray TinyIqr(double spread)
        {
            var v = new double[101];
            for (int i = 0; i < 100; i++) v[i] = i < 60 ? 1.0 : 1.0 + spread;
            v[100] = 1e3;
            return np.array(v);
        }
        ((Action)(() => np.histogram_bin_edges(TinyIqr(1e-9), "fd"))).Should().Throw<OutOfMemoryException>()
            .WithMessage("Unable to allocate 16.9 TiB for an array with shape (2326176056681,) and data type float64");
        ((Action)(() => np.histogram_bin_edges(TinyIqr(1e-12), "fd"))).Should().Throw<OutOfMemoryException>()
            .WithMessage("Unable to allocate 16.5 PiB for an array with shape (2325969469107706,) and data type float64");
        ((Action)(() => np.histogram_bin_edges(TinyIqr(1e-15), "fd"))).Should().Throw<ValueError>()
            .WithMessage("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.");
    }

    /// <summary>
    ///     stone in NumPy's dtypes: its candidate widths <c>ptp_x / nbins</c> and the width it returns keep the peak-to-peak
    ///     value's float16 / float32 dtype, so <c>int(np.ceil(delta / width))</c> divides in that dtype, and jhat comes from
    ///     NumPy's <c>p_k = hist / n</c> and its dot product, the first minimum winning (Python's min). NumSharp computed it
    ///     in double from the sum of squared counts over n² and got other counts: 101 bins for the float16 samples below
    ///     (NumPy 2.4.2), 14 for the float32 ones, 17 for the float64 ones (it picked 3), and NumPy's "Too many bins" for
    ///     64 bins between two float16 samples 1/16 apart (it asked for 66).
    /// </summary>
    [TestMethod]
    public void Estimator_Stone_InNumPysDtypes()
    {
        var (h16, e16) = np.histogram(np.array(new[] { (Half)231.0, (Half)471.5, (Half)230.0 }), "stone");
        h16.size.Should().Be(101);
        h16.GetAtIndex<long>(0).Should().Be(2);
        h16.GetAtIndex<long>(100).Should().Be(1);
        e16.typecode.Should().Be(NPTypeCode.Half);
        e16.GetAtIndex<Half>(1).Should().Be((Half)232.375);
        e16.GetAtIndex<Half>(3).Should().Be((Half)237.125);
        e16.GetAtIndex<Half>(101).Should().Be((Half)471.5);

        var (h32, e32) = np.histogram(np.array(new[] { -2247.27197265625f, -2084.80615234375f, 154.9859161376953f }), "stone");
        Eq(h32, new double[] { 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 });
        e32.typecode.Should().Be(NPTypeCode.Single);
        e32.GetAtIndex<float>(1).Should().Be(-2075.68212890625f);
        e32.GetAtIndex<float>(3).Should().Be(-1732.50244140625f);

        var f64 = np.array(new[] { -293.8699097337545, -881.0695740733971, 1348.384654520067, 688.6263416669709,
                                   -232.47424791786034, -330.9475469634296 });
        var (h64, e64) = np.histogram(f64, "stone");
        Eq(h64, new double[] { 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 });
        e64.GetAtIndex<double>(1).Should().Be(-749.9252076855463);
        e64.GetAtIndex<double>(3).Should().Be(-487.63647490984465);

        ((Action)(() => np.histogram(np.array(new[] { (Half)(-1.515625), (Half)(-1.5791015625) }), "stone")))
            .Should().Throw<ValueError>().WithMessage("Too many bins for data range. Cannot create 64 finite-sized bins.");
    }

    /// <summary>
    ///     doane over float16 data with a range: the edges' linspace computes in float64 from the range's Python floats and
    ///     rounds to float16, and the width's dtypes are NumPy's: 5 bins for the three samples below (NumPy 2.4.2), where
    ///     NumSharp's double arithmetic gave 4.
    /// </summary>
    [TestMethod]
    public void Estimator_Doane_Float16Range_InNumPysDtypes()
    {
        var a = np.array(new[] { (Half)0.0027332305908203125, (Half)0.0026264190673828125, (Half)0.0033740997314453125 });
        var (h, e) = np.histogram(a, "doane", range: (0.0026000219202041673, 0.0034103449204022267));
        Eq(h, new double[] { 2, 0, 0, 0, 1 });
        e.typecode.Should().Be(NPTypeCode.Half);
        e.GetAtIndex<Half>(0).Should().Be((Half)0.0025997161865234375);
        e.GetAtIndex<Half>(1).Should().Be((Half)0.0027618408203125);
        e.GetAtIndex<Half>(5).Should().Be((Half)0.00341033935546875);
    }

    /// <summary>
    ///     Data with no spread: <c>_get_outer_edges</c> widens the edges by 0.5 in the data's own dtype, and scott's np.std
    ///     is the rounding error of the mean, a nonzero width. NumPy 2.4.2: three float16 3.3 get 212 bins (all three in
    ///     bin 106, the float16 width and quotient), six float16 0.7 ask for 1066 bins the float16 edges cannot hold, three
    ///     float64 0.1 for 29770864728363688 edges (NumPy's MemoryError). NumSharp's double width gave other counts.
    /// </summary>
    [TestMethod]
    public void Estimator_NoSpread_WidenedEdgesInTheDataDtype()
    {
        var (h, e) = np.histogram(np.full(Shape.Vector(3), (Half)3.3), "scott");
        h.size.Should().Be(212);
        h.GetAtIndex<long>(106).Should().Be(3);
        Flat(np.sum(h))[0].Should().Be(3);
        e.typecode.Should().Be(NPTypeCode.Half);
        e.GetAtIndex<Half>(0).Should().Be((Half)2.80078125);
        e.GetAtIndex<Half>(1).Should().Be((Half)2.8046875);
        e.GetAtIndex<Half>(212).Should().Be((Half)3.80078125);

        ((Action)(() => np.histogram(np.full(Shape.Vector(6), (Half)0.7), "scott")))
            .Should().Throw<ValueError>().WithMessage("Too many bins for data range. Cannot create 1066 finite-sized bins.");
        ((Action)(() => np.histogram(np.full(Shape.Vector(3), 0.1), "scott"))).Should().Throw<OutOfMemoryException>()
            .WithMessage("Unable to allocate 212. PiB for an array with shape (29770864728363688,) and data type float64");
    }

    /// <summary>
    ///     Data with no spread whose ±0.5 widening rounds back to the value itself (float16 1026, float32 1e9, uint64
    ///     2^64 - 1e9 - 1 as a float64): scott's nonzero width over a zero <c>delta</c> is 0 bins, and NumPy's index
    ///     <c>(tmp_a - first_edge) / norm_denom * 0</c> is 0 / 0, a NaN, which <c>astype(np.intp)</c> makes INT64_MIN and
    ///     <c>bin_edges[indices]</c> refuses (NumPy 2.4.2), before the bincount's refusal of a -1 index. A range of the
    ///     same value takes the same path (its Python floats widen to the same float64).
    /// </summary>
    [TestMethod]
    public void Estimator_NoSpread_ZeroWidth_IndexError()
    {
        const string text = "index -9223372036854775808 is out of bounds for axis 0 with size 1";
        ((Action)(() => np.histogram(np.full(Shape.Vector(64), (Half)1026), "scott")))
            .Should().Throw<IndexError>().WithMessage(text);
        ((Action)(() => np.histogram(np.full(Shape.Vector(10), 1e9f), "scott")))
            .Should().Throw<IndexError>().WithMessage(text);
        var u64 = np.full(Shape.Vector(5), 18446744072709551615UL);
        ((Action)(() => np.histogram(u64, "scott"))).Should().Throw<IndexError>().WithMessage(text);
        ((Action)(() => np.histogram(u64, "scott", range: (1.8446744072709552e+19, 1.8446744072709552e+19))))
            .Should().Throw<IndexError>().WithMessage(text);
    }

    /// <summary>
    ///     float16 / float32 data over a range: NEP 50 rounds the range's Python floats to the data's dtype where they meet
    ///     the data, but <c>norm_denom</c> is their float64 difference, so a kept sample's index is a float64 quotient that
    ///     can pass the bin count when the range is narrower than the dtype's spacing; the fancy index into the edges then
    ///     refuses it (NumPy 2.4.2). NumSharp computed the quotient from the rounded edges and counted the samples.
    /// </summary>
    [TestMethod]
    public void Histogram_NarrowFloatRange_IndexPassesTheEdges_IndexError()
    {
        var f16 = np.array(new[] { (Half)10080, (Half)10064 });
        (double, double) range = (10066.283933623668, 10076.727203988607);
        ((Action)(() => np.histogram(f16, 2, range: range)))
            .Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 0 with size 3");
        ((Action)(() => np.histogram(f16, "sturges", range: range)))
            .Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 0 with size 3");
        ((Action)(() => np.histogram(f16, 2, range: range, weights: np.array(new[] { 0.5, 2.0 }))))
            .Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 0 with size 3");

        var f32 = np.array(new[] { 1e8f, 1e8f + 16 });
        ((Action)(() => np.histogram(f32, 2, range: (1e8 + 3.9, 1e8 + 12.1))))
            .Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 0 with size 3");
    }

    /// <summary>
    ///     float16 / float32 data over a range with weights: the float64 quotient path sums each block's weights in float64
    ///     and folds them into the result in the weights' dtype, complex weights part by part (NumPy 2.4.2's answers).
    /// </summary>
    [TestMethod]
    public void Histogram_FloatRange_Weighted()
    {
        var (h, e) = np.histogram(np.array(new[] { 0.5f, 1.25f, 2.75f, 3.0f }), 3, range: (0.0, 3.0),
            weights: np.array(new[] { 1.5, 2.0, 0.25, 4.0 }));
        Eq(h, new double[] { 1.5, 2.0, 4.25 });
        h.typecode.Should().Be(NPTypeCode.Double);
        e.typecode.Should().Be(NPTypeCode.Single);

        var (hc, ec) = np.histogram(np.array(new[] { (Half)0.5, (Half)1.25, (Half)2.75, (Half)3.0 }), 3, range: (0.0, 3.0),
            weights: np.array(new[] { new Complex(1, 2), new Complex(2, 0), new Complex(0, 0.25), new Complex(4, 0) }));
        hc.typecode.Should().Be(NPTypeCode.Complex);
        hc.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 2));
        hc.GetAtIndex<Complex>(1).Should().Be(new Complex(2, 0));
        hc.GetAtIndex<Complex>(2).Should().Be(new Complex(4, 0.25));
        ec.typecode.Should().Be(NPTypeCode.Half);
    }

    /// <summary>
    ///     float16 / float32 data over a range with enough samples for the range count kernel's privatized accumulators
    ///     (4096 samples or more, at least four per bin): ten thousand values <c>(i² mod 9973) / 9973</c> in 7 bins over
    ///     (0.15, 0.85), NumPy 2.4.2's counts.
    /// </summary>
    [TestMethod]
    public void Histogram_FloatRange_ManySamples()
    {
        var values = new double[10000];
        for (int i = 0; i < values.Length; i++)
            values[i] = (double)((long)i * i % 9973) / 9973;
        var f64 = np.array(values);

        var (h32, e32) = np.histogram(f64.astype(NPTypeCode.Single), 7, range: (0.15, 0.85));
        Eq(h32, new double[] { 996, 1028, 970, 944, 970, 1028, 996 });
        e32.GetAtIndex<float>(3).Should().Be(0.44999998807907104f);

        var (h16, e16) = np.histogram(f64.astype(NPTypeCode.Half), 7, range: (0.15, 0.85));
        Eq(h16, new double[] { 994, 1030, 968, 940, 974, 1028, 1002 });
        e16.GetAtIndex<Half>(3).Should().Be((Half)0.449951171875);
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

    /// <summary>
    ///     histogramdd hands its weights to <c>np.bincount</c> as given (NumPy never ravels them), so weights that are not
    ///     1-D meet bincount's depth check, which comes before its cast: a (2, 2) or an (N, 1) array, deeper text and the
    ///     histogram2d form are <c>object too deep for desired array</c>, a 0-d array — complex too — beside one point
    ///     <c>object of too small depth for desired array</c> (NumPy 2.4.2's ValueErrors; the house dimension-error type).
    ///     The weights used to be raveled, so an (N, 1) array histogrammed and a (2, 2) one met the length check.
    /// </summary>
    [TestMethod]
    public void Error_DD_WeightsThatAreNot1D_MeetBincountsDepthCheck()
    {
        var s = np.array(new double[,] { { 0, 1 }, { 2, 3 } });
        var onePoint = np.array(new double[,] { { 0.5, 1.5 } });
        foreach (var act in new Action[]
                 {
                     () => np.histogramdd(s, 2, weights: np.ones(new Shape(2, 2))),
                     () => np.histogramdd(s, 2, weights: np.ones(new Shape(2, 1))),
                     () => np.histogramdd(s, 2, weights: np.array(new[] { "1", "2", "3", "4" }).reshape(2, 2)),
                     () => np.histogram2d(np.array(new[] { 0.0, 2.0 }), np.array(new[] { 1.0, 3.0 }), 2,
                         weights: np.ones(new Shape(2, 1))),
                 })
            act.Should().Throw<IncorrectShapeException>().WithMessage("object too deep for desired array");
        foreach (var act in new Action[]
                 {
                     () => np.histogramdd(onePoint, 2, weights: np.array(2.0)),
                     () => np.histogramdd(onePoint, 2, weights: np.array(new Complex(1, 2))),
                 })
            act.Should().Throw<IncorrectShapeException>().WithMessage("object of too small depth for desired array");
    }

    /// <summary>
    ///     A sequence sample is stacked first (<c>np.atleast_2d(sample).T</c>): entries of different shapes are refused by
    ///     array coercion with NumPy 2.4.2's texts — before any bin, range or weight is read — instead of being read with
    ///     the first entry's length (past the end of a shorter entry). The detected shape is what the shapes still agree on.
    /// </summary>
    [TestMethod]
    public void Error_DD_SequenceOfDifferentShapes_IsInhomogeneous()
    {
        var x3 = np.array(new[] { 0.5, 1.5, 2.5 });
        var y2 = np.array(new[] { 0.5, 1.5 });
        var z0 = np.array(1.5);
        var e0 = np.zeros(new Shape(0), np.float64);
        var m23 = np.arange(6.0).reshape(2, 3);
        var m33 = np.arange(9.0).reshape(3, 3);
        var m24 = np.arange(8.0).reshape(2, 4);
        var t3 = np.arange(24.0).reshape(2, 3, 4);
        const string head = "setting an array element with a sequence. The requested array has an inhomogeneous shape after ";
        foreach (var (sample, tail) in new (NDArray[], string)[]
                 {
                     (new[] { x3, y2 }, "1 dimensions. The detected shape was (2,) + inhomogeneous part."),
                     (new[] { x3, z0 }, "1 dimensions. The detected shape was (2,) + inhomogeneous part."),
                     (new[] { z0, x3 }, "1 dimensions. The detected shape was (2,) + inhomogeneous part."),
                     (new[] { e0, z0 }, "1 dimensions. The detected shape was (2,) + inhomogeneous part."),
                     (new[] { x3, null }, "1 dimensions. The detected shape was (2,) + inhomogeneous part."),
                     (new[] { m33, x3 }, "2 dimensions. The detected shape was (2, 3) + inhomogeneous part."),
                     (new[] { x3, m33 }, "2 dimensions. The detected shape was (2, 3) + inhomogeneous part."),
                     (new[] { m23, m24 }, "2 dimensions. The detected shape was (2, 2) + inhomogeneous part."),
                     (new[] { m23, t3 }, "3 dimensions. The detected shape was (2, 2, 3) + inhomogeneous part."),
                     (new[] { t3, m23 }, "3 dimensions. The detected shape was (2, 2, 3) + inhomogeneous part."),
                     (new[] { m23, m24, x3 }, "1 dimensions. The detected shape was (3,) + inhomogeneous part."),
                     (new[] { y2, x3, x3 }, "1 dimensions. The detected shape was (3,) + inhomogeneous part."),
                 })
            ((Action)(() => np.histogramdd(sample, 2))).Should().Throw<ValueError>().WithMessage(head + tail);
        // The stacking is the first statement: it wins over a wrong number of bins or ranges.
        ((Action)(() => np.histogramdd(new[] { x3, y2 }, new[] { 2 }))).Should().Throw<ValueError>().WithMessage(head + "*");
        ((Action)(() => np.histogramdd(new[] { x3, y2 }, 2, new (double, double)?[] { (0, 1) })))
            .Should().Throw<ValueError>().WithMessage(head + "*");
    }

    /// <summary>
    ///     After the stacking, <c>N, D = sample.shape</c> needs a 2-D stack: entries of two or more dimensions (one, or several
    ///     of one shape), an array sample of three or more dimensions, and histogram2d's two 2-D arrays of one shape leave
    ///     too many values (NumPy 2.4.2: <c>too many values to unpack (expected 2)</c>, before any bin is read).
    /// </summary>
    [TestMethod]
    public void Error_DD_SampleDeeperThan2D_IsTooManyValuesToUnpack()
    {
        var m23 = np.arange(6.0).reshape(2, 3);
        var t3 = np.arange(24.0).reshape(2, 3, 4);
        foreach (var act in new Action[]
                 {
                     () => np.histogramdd(new[] { m23 }, 2),
                     () => np.histogramdd(new[] { m23, m23 }, new[] { 2 }),
                     () => np.histogramdd(new[] { t3, t3 }, 2),
                     () => np.histogramdd(t3, 2),
                     () => np.histogramdd(t3, new[] { 2 }),
                     () => np.histogramdd(np.zeros(new Shape(1, 2, 3, 4), np.float64), 2),
                     () => np.histogram2d(m23, m23, 2),
                 })
            act.Should().Throw<ValueError>().WithMessage("too many values to unpack (expected 2)");
    }

    /// <summary>
    ///     0-d entries stack to <c>(n,)</c>, which <c>atleast_2d(..).T</c> makes <c>(n, 1)</c>: ONE dimension of n points,
    ///     in the entries' promoted dtype (int64 beside float64 is float64), weighted per point; an empty sequence is NumPy's
    ///     float64 <c>(0,)</c> stack, one empty dimension; a 0-d array sample is <c>(1, 1)</c>. NumPy 2.4.2's answers. The 0-d
    ///     entries used to be one dimension each, and the 0-d array sample was refused.
    /// </summary>
    [TestMethod]
    public void DD_ScalarEntriesAndZeroDimSample_AreOneColumn()
    {
        var z0 = np.array(1.5);
        var z0b = np.array(2.5);
        var (h, e) = np.histogramdd(new[] { z0, z0b }, 2);
        h.shape.Should().Equal(new long[] { 2 });
        Eq(h, new[] { 1.0, 1.0 });
        e.Length.Should().Be(1);
        Eq(e[0], new[] { 1.5, 2.0, 2.5 });

        (h, e) = np.histogramdd(new[] { z0, np.array(2L) }, 2);
        Eq(e[0], new[] { 1.5, 1.75, 2.0 });
        e[0].typecode.Should().Be(NPTypeCode.Double);

        (h, _) = np.histogramdd(new[] { z0, z0b }, 2, weights: np.array(new[] { 1.0, 3.0 }));
        Eq(h, new[] { 1.0, 3.0 });
        ((Action)(() => np.histogramdd(new[] { z0, z0b }, new[] { 2, 2 })))
            .Should().Throw<ValueError>().WithMessage("The dimension of bins must be equal to the dimension of the sample x.");

        (h, e) = np.histogramdd(new NDArray[0], 2);
        h.shape.Should().Equal(new long[] { 2 });
        Eq(h, new[] { 0.0, 0.0 });
        Eq(e[0], new[] { 0.0, 0.5, 1.0 });

        (h, e) = np.histogramdd(z0, 2);
        Eq(h, new[] { 0.0, 1.0 });
        Eq(e[0], new[] { 1.0, 1.5, 2.0 });
        ((Action)(() => np.histogramdd(z0, new[] { 2, 2 })))
            .Should().Throw<ValueError>().WithMessage("The dimension of bins must be equal to the dimension of the sample x.");
    }

    /// <summary>
    ///     0-d text entries stack into one text column of the promoted text dtype (a float64 beside a <c>&lt;U1</c> makes
    ///     <c>&lt;U32</c>, whose missing minimum loop NumPy names; with numeric edges the formatted text is searched, NumPy
    ///     2.4.2's <c>[1, 0]</c>). NumPy stacks None, or StringDType text beside a number, into an object array, which NumSharp
    ///     has no dtype for.
    /// </summary>
    [TestMethod]
    public void DD_TextScalarEntries_AreOneTextColumn_ObjectStacksAreRefused()
    {
        var u0 = np.squeeze(np.array(new[] { "3" }));
        var f0 = np.array(1.5);
        ((Action)(() => np.histogramdd(new[] { u0, f0 }, 2))).Should().Throw<TypeError>()
            .WithMessage("ufunc 'minimum' did not contain a loop with signature matching types (dtype('<U32'), dtype('<U32')) -> None");
        var (h, _) = np.histogramdd(new[] { u0, f0 }, new[] { np.array(new[] { 0.0, 2.0, 20.0 }) });
        Eq(h, new[] { 1.0, 0.0 });

        var t0 = np.squeeze(np.array(new[] { "5" }).astype(np.dtype("T")));
        foreach (var act in new Action[]
                 {
                     () => np.histogramdd(new NDArray[] { null }, 2),
                     () => np.histogramdd(new NDArray[] { null, null }, 2),
                     () => np.histogramdd(new[] { t0, f0 }, 2),
                 })
            act.Should().Throw<NotSupportedException>();
    }

    /// <summary>
    ///     histogram2d is <c>histogramdd([x, y])</c>: x and y of one length but different shapes are refused by the stacking
    ///     (NumPy 2.4.2: a <c>(2, 3)</c> x beside a <c>(2, 4)</c> or a <c>(2,)</c> y is inhomogeneous after 2 dimensions,
    ///     <c>(2, 2)</c>).
    /// </summary>
    [TestMethod]
    public void Error_Histogram2D_SameLengthDifferentShapes_IsInhomogeneous()
    {
        var m23 = np.arange(6.0).reshape(2, 3);
        foreach (var y in new[] { np.arange(8.0).reshape(2, 4), np.array(new[] { 0.5, 1.5 }) })
            ((Action)(() => np.histogram2d(m23, y, 2))).Should().Throw<ValueError>().WithMessage(
                "setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. " +
                "The detected shape was (2, 2) + inhomogeneous part.");
    }

    /// <summary>
    ///     Without points, bincount returns its zeros before it reads the weights, so weights of any shape or dtype pass and
    ///     H is all zeros (NumPy 2.4.2: <c>np.histogramdd(np.zeros((0, 2)), bins=2, weights=np.ones((3, 3)))</c>, and with a
    ///     0-d text weight).
    /// </summary>
    [TestMethod]
    public void DD_NoPoints_WeightsAreNeverRead()
    {
        var empty = np.zeros(new Shape(0, 2), np.float64);
        foreach (var w in new[] { np.ones(new Shape(3, 3)), np.squeeze(np.array(new[] { "a" })) })
        {
            var (H, edges) = np.histogramdd(empty, 2, weights: w);
            H.shape.Should().Equal(new long[] { 2, 2 });
            Flat(np.sum(H))[0].Should().Be(0);
            Eq(edges[1], new[] { 0.0, 0.5, 1.0 });
        }
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

    /// <summary>
    ///     histogram2d's first statement is NumPy's <c>if len(x) != len(y)</c>, and <c>len()</c> of a 0-d array raises
    ///     <c>TypeError: len() of unsized object</c> — for x or y, whatever the bins (NumPy 2.4.2:
    ///     <c>np.histogram2d(np.array(1.5), np.array([0.5, 1.5, 2.5]), bins=2)</c>, the swapped call, two 0-d arrays with
    ///     <c>bins=10</c> and with <c>bins=[2, 2]</c>). Two 0-d arrays used to histogram as two one-point columns.
    /// </summary>
    [TestMethod]
    public void Error_Histogram2D_ZeroDimOperand_IsLenOfUnsizedObject()
    {
        var z0 = np.array(1.5);
        var z0b = np.array(2.5);
        var x3 = np.array(new[] { 0.5, 1.5, 2.5 });
        foreach (var act in new Action[]
                 {
                     () => np.histogram2d(z0, x3, 2),
                     () => np.histogram2d(x3, z0, 2),
                     () => np.histogram2d(z0, z0b),
                     () => np.histogram2d(z0, z0b, new[] { 2, 2 }),
                 })
            act.Should().Throw<TypeError>().WithMessage("len() of unsized object");
    }

    /// <summary>
    ///     <c>len()</c> is the FIRST dimension, not the size: a <c>(2, 3)</c> x and a <c>(3, 2)</c> y, or a <c>(2, 3)</c> x
    ///     and a six-point y, have different lengths although their sizes agree (NumPy 2.4.2:
    ///     <c>x and y must have the same length.</c>), and the length check comes before the bins' own refusal
    ///     (<c>bins=[2]</c>, one count for two dimensions).
    /// </summary>
    [TestMethod]
    public void Error_Histogram2D_LengthIsTheFirstDimension()
    {
        var m23 = np.arange(6.0).reshape(2, 3);
        foreach (var act in new Action[]
                 {
                     () => np.histogram2d(m23, np.arange(6.0).reshape(3, 2), 2),
                     () => np.histogram2d(m23, np.arange(6.0), 2),
                     () => np.histogram2d(np.array(new[] { 0.5, 1.5, 2.5 }), np.array(new[] { 0.5, 1.5 }), new[] { 2 }),
                 })
            act.Should().Throw<ValueError>().WithMessage("x and y must have the same length.");
    }

    // ── mixed-dtype columns: NumPy stacks a sequence via atleast_2d(sample).T, promoting to a common dtype ──
    // (regression for the histogram2d/histogramdd column-promotion fix)

    [TestMethod]
    public void Histogram2D_MixedDtype_PromotesBothEdgesToFloat64()
    {
        // np.histogram2d(float32, float64, 4) => xedges AND yedges are float64 (the stacked sample promotes).
        var x = np.array(new float[] { 1, 2, 3, 4, 5 });
        var y = np.array(new[] { 1.0, 1, 2, 2, 3 });
        var (_, xe, ye) = np.histogram2d(x, y, 4);
        xe.typecode.Should().Be(NPTypeCode.Double);
        ye.typecode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Histogram2D_MixedDtype_HalfAndSingle_PromoteToSingle()
    {
        var x = np.array(new Half[] { (Half)1, (Half)2, (Half)3, (Half)4, (Half)5 });
        var y = np.array(new float[] { 1, 1, 2, 2, 3 });
        var (_, xe, ye) = np.histogram2d(x, y, 4);
        xe.typecode.Should().Be(NPTypeCode.Single);
        ye.typecode.Should().Be(NPTypeCode.Single);
    }

    [TestMethod]
    public void HistogramDD_MixedDtypeColumns_PromoteToFloat64()
    {
        var x = np.array(new float[] { 1, 2, 3, 4, 5 });
        var y = np.array(new[] { 1.0, 1, 2, 2, 3 });
        var (_, edges) = np.histogramdd(new[] { x, y }, 4);
        edges[0].typecode.Should().Be(NPTypeCode.Double);
        edges[1].typecode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void HistogramDD_SameDtypeColumns_StayFloat32()
    {
        // Regression guard: a same-dtype sequence must NOT be promoted (the common case).
        var x = np.array(new float[] { 1, 2, 3, 4, 5 });
        var (_, edges) = np.histogramdd(new[] { x, x.copy() }, 4);
        edges[0].typecode.Should().Be(NPTypeCode.Single);
        edges[1].typecode.Should().Be(NPTypeCode.Single);
    }

    // ── weak-endpoint edge dtype: histogramdd edges collapse to float64 for empty/range columns ──

    [TestMethod]
    public void HistogramDD_EmptyFloat32Column_EdgesFloat64()
    {
        // Empty column => endpoints are the int 0/1 defaults (weak) => float64 edges (NumPy: linspace has no
        // array to keep it strong, so result_type(int,int,num) = float64).
        var (_, edges) = np.histogramdd(new[] { np.array(new float[] { }) }, 4);
        edges[0].typecode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Histogram2D_RangeFloat32_EdgesFloat64()
    {
        // A supplied range makes the endpoints weak python floats => float64 edges even for float32 columns.
        var x = np.array(new float[] { 1, 2, 3, 4, 5 });
        var (_, xe, ye) = np.histogram2d(x, x.copy(), 4, range: new (double, double)?[] { (0, 6), (0, 6) });
        xe.typecode.Should().Be(NPTypeCode.Double);
        ye.typecode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Histogram_Float32Range_EdgesFloat32ComputedInFloat64()
    {
        // np.histogram keeps the RESULT dtype float32 (the array is in result_type), but with a range the edge
        // VALUES are computed in float64 then cast to float32 — bit-exact with NumPy 2.4.2.
        var a = np.array(new float[] { 0.5f, 1.5f, 2.5f, 3.5f, 0.9f });
        var e = np.histogram_bin_edges(a, 5, range: (0.0, 4.0));
        e.typecode.Should().Be(NPTypeCode.Single);
        // NumPy 2.4.2: [0, 0.800000011920929, 1.600000023841858, 2.4000000953674316, 3.200000047683716, 4]
        var got = new float[6];
        for (long i = 0; i < 6; i++) got[i] = e.GetAtIndex<float>(i);
        got.Should().Equal(0f, 0.8f, 1.6f, 2.4f, 3.2f, 4f);
    }

    // ── complex weights through the uniform (int-bins) path: the imaginary part must survive ──
    // Regression for the EnsureContiguousDtype offset fix: np.imag(size-1 weight) is a trivially-contiguous
    // OFFSET view, so reading .Address (ignoring the offset) grabbed the REAL part — hist was <re, re>.

    [TestMethod]
    public void Histogram_ComplexWeights_UniformPath_ImaginaryPartPreserved()
    {
        var w = np.array(new Complex[] { new(0.586, 0.150) });
        var (h, _) = np.histogram(np.array(new[] { 2.0 }), 16, weights: w); // INT bins => uniform path
        h.typecode.Should().Be(NPTypeCode.Complex);
        // The single weight lands whole in one bin; its imaginary part must be 0.150, not 0.586.
        h.GetAtIndex<Complex>(8).Should().Be(new Complex(0.586, 0.150));
    }

    // ── FormatEdge: the non-finite range message renders bounds like Python's str(float) ──

    [TestMethod]
    public void Error_AutodetectNonFiniteRange_WholeNumberBound_VerbatimMessage()
        => ((Action)(() => np.histogram(np.array(new[] { 1.0, double.PositiveInfinity }), 3)))
            .Should().Throw<ValueError>()
            .WithMessage("autodetected range of [1.0, inf] is not finite");

    [TestMethod]
    public void Error_SuppliedNonFiniteRange_WholeNumberBound_VerbatimMessage()
        => ((Action)(() => np.histogram(np.linspace(0.0, 1.0, 20), 5, range: (1.0, double.PositiveInfinity))))
            .Should().Throw<ValueError>()
            .WithMessage("supplied range of [1.0, inf] is not finite");
}
