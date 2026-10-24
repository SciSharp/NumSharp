using System;
using System.Numerics;

namespace NumSharp.Tests.Statistics;

/// <summary>
///     np.histogram where NumPy's cumulative statements decide the answer (<c>numpy/lib/_histograms_impl.py</c>: per block the
///     data sorted, every edge searched, the running total updated in place, the counts its <c>np.diff</c>), and a shortcut
///     that bins each sample would not give it. Every expected value is NumPy 2.4.2's output.
/// </summary>
[TestClass]
public class np_histogram_cumulative_loop_tests
{
    private static readonly double NaN = double.NaN;

    /// <summary>The pool the probes used: twelve integers, eight of them at least 0.</summary>
    private static readonly double[] Twelve = { 3, -1, 2, 0, 5, -4, 1, 7, -2, 6, 4, -3 };

    /// <summary>
    ///     An inner NaN edge passes the monotonic check (no comparison with it is True) and the search places it after every
    ///     number, so the next count is a negative difference: <c>np.histogram(x, bins=[0, nan, 2])</c> is <c>[8, -5]</c>.
    /// </summary>
    [TestMethod]
    public void InnerNaNEdge_CountsAreDifferencesOfSearchPositions()
    {
        var (h, e) = np.histogram(np.array(Twelve), np.array(new[] { 0.0, NaN, 2.0 }));
        h.typecode.Should().Be(NPTypeCode.Int64);
        h.ToArray<long>().Should().Equal(8L, -5L);
        e.ToArray<double>()[1].Should().Be(NaN);

        // A first NaN edge: the first count is negative, [-8, 3].
        np.histogram(np.array(Twelve), np.array(new[] { NaN, 0.0, 2.0 })).hist.ToArray<long>().Should().Equal(-8L, 3L);
    }

    /// <summary>The answer does not depend on the data's dtype: each is searched in its common type with the edges.</summary>
    [TestMethod]
    public void InnerNaNEdge_EveryNumericDataDtype()
    {
        np.histogram(np.array(Twelve).astype(NPTypeCode.Single), np.array(new[] { 0f, float.NaN, 2f })).hist
            .ToArray<long>().Should().Equal(8L, -5L);
        np.histogram(np.array(Twelve).astype(NPTypeCode.Half), np.array(new[] { (Half)0, Half.NaN, (Half)2 })).hist
            .ToArray<long>().Should().Equal(8L, -5L);
        np.histogram(np.array(Twelve).astype(NPTypeCode.SByte), np.array(new[] { 0.0, NaN, 2.0 })).hist
            .ToArray<long>().Should().Equal(8L, -5L);
        np.histogram(np.array(new byte[] { 3, 1, 2, 0, 5, 4, 1, 7, 2, 6, 4, 3 }), np.array(new[] { 0.0, NaN, 2.0 })).hist
            .ToArray<long>().Should().Equal(12L, -7L);

        // Complex data search as complex (lexicographic): [8, -6].
        var c = new Complex[Twelve.Length];
        for (int i = 0; i < c.Length; i++) c[i] = new Complex(Twelve[i], 0.5);
        np.histogram(np.array(c), np.array(new[] { 0.0, NaN, 2.0 })).hist.ToArray<long>().Should().Equal(8L, -6L);
    }

    /// <summary>
    ///     A NaN last edge is inclusive like every last edge, and NaN samples sort last, so they are counted:
    ///     <c>np.histogram(x + [nan, nan], bins=[0, 2, nan])</c> is <c>[2, 8]</c>, and <c>bins=[nan, nan]</c> counts only them.
    /// </summary>
    [TestMethod]
    public void NaNLastEdge_CountsTheNaNSamples()
    {
        var withNaN = np.array(new[] { 3, -1, 2, 0, 5, -4, 1, 7, -2, 6, 4, -3, NaN, NaN });
        np.histogram(withNaN, np.array(new[] { 0.0, 2.0, NaN })).hist.ToArray<long>().Should().Equal(2L, 8L);
        np.histogram(withNaN, np.array(new[] { NaN, NaN })).hist.ToArray<long>().Should().Equal(2L);

        // Without NaN samples the last bin holds every sample from its lower edge on.
        np.histogram(np.array(Twelve), np.array(new[] { 0.0, 2.0, NaN })).hist.ToArray<long>().Should().Equal(2L, 6L);
    }

    /// <summary>Infinite outer edges around a NaN: every sample is below the NaN, so <c>[12, 0]</c>.</summary>
    [TestMethod]
    public void NaNBetweenInfiniteEdges()
        => np.histogram(np.array(Twelve), np.array(new[] { double.NegativeInfinity, NaN, double.PositiveInfinity })).hist
            .ToArray<long>().Should().Equal(12L, 0L);

    /// <summary>The data's shape, an empty input and a 0-d input do not change the statements.</summary>
    [TestMethod]
    public void NaNEdge_DataShapes()
    {
        var edges = np.array(new[] { 0.0, NaN, 2.0 });
        np.histogram(np.arange(24.0).reshape(2, 3, 4), edges).hist.ToArray<long>().Should().Equal(24L, -21L);
        np.histogram(np.array(new double[0]), edges).hist.ToArray<long>().Should().Equal(0L, 0L);
        np.histogram(np.array(5.0), edges).hist.ToArray<long>().Should().Equal(1L, -1L);
    }

    /// <summary>
    ///     The loop runs per block of 65536 samples and sums the blocks' search positions: 70,000 samples give
    ///     <c>[70000, -69899]</c>.
    /// </summary>
    [TestMethod]
    public void NaNEdge_TwoBlocks()
        => np.histogram(np.arange(70000.0), np.array(new[] { 0.0, NaN, 100.0 })).hist
            .ToArray<long>().Should().Equal(70000L, -69899L);

    /// <summary>With density, the NaN bin widths make every density NaN (NumPy: <c>[nan, nan]</c>).</summary>
    [TestMethod]
    public void NaNEdge_Density_IsNaN()
    {
        var h = np.histogram(np.array(Twelve), np.array(new[] { 0.0, NaN, 2.0 }), density: true).hist;
        h.typecode.Should().Be(NPTypeCode.Double);
        double.IsNaN(h.GetDouble(0)).Should().BeTrue();
        double.IsNaN(h.GetDouble(1)).Should().BeTrue();
    }

    /// <summary>Edges given as a strided view are read through their strides: <c>[0, nan, 2, 9, nan][::2]</c> is
    ///     <c>[0, 2, nan]</c>, so <c>[2, 6]</c>.</summary>
    [TestMethod]
    public void NaNEdge_StridedEdges()
        => np.histogram(np.array(Twelve), np.array(new[] { 0.0, NaN, 2.0, 9.0, NaN })["::2"]).hist
            .ToArray<long>().Should().Equal(2L, 6L);

    // ── weights on explicit edges: NumPy's in-place `cum_n += cw[bin_index]` ─────────────────────────────────────────

    /// <summary>The explicit edges the weighted probes used: three bins, the weights summing to [2, 2, 4].</summary>
    private static NDArray Edges() => np.array(new[] { -2.0, 0.0, 1.5, 5.0 });

    /// <summary>
    ///     The summed weights keep the weights' dtype: NumPy's total is <c>np.zeros(..., weights.dtype)</c> and every block
    ///     is added into it in place, whatever the cumulative sum's own dtype (int64 for every integer and bool).
    /// </summary>
    [TestMethod]
    public void Weights_TotalKeepsTheWeightsDtype()
    {
        foreach (var (dtype, expected) in new[]
                 {
                     (NPTypeCode.SByte, NPTypeCode.SByte), (NPTypeCode.Byte, NPTypeCode.Byte),
                     (NPTypeCode.Int16, NPTypeCode.Int16), (NPTypeCode.UInt64, NPTypeCode.UInt64),
                     (NPTypeCode.Half, NPTypeCode.Half), (NPTypeCode.Single, NPTypeCode.Single),
                 })
        {
            var h = np.histogram(np.array(Twelve), Edges(), weights: np.ones(new Shape(12), dtype)).hist;
            h.typecode.Should().Be(expected, dtype.ToString());
            h.astype(NPTypeCode.Double).ToArray<double>().Should().Equal(new[] { 2.0, 2.0, 4.0 }, dtype.ToString());
        }
    }

    /// <summary>An int8 total wraps in int8: 300 weights of 1 in one bin are <c>[44, 0]</c> (300 - 256).</summary>
    [TestMethod]
    public void Weights_Int8TotalWraps()
    {
        var h = np.histogram(np.zeros(new Shape(300)), np.array(new[] { -1.0, 0.5, 2.0 }),
            weights: np.ones(new Shape(300), NPTypeCode.SByte)).hist;
        h.typecode.Should().Be(NPTypeCode.SByte);
        h.ToArray<sbyte>().Should().Equal((sbyte)44, (sbyte)0);
    }

    /// <summary>
    ///     Bool weights: the bool total cannot take the int64 cumulative sum under same_kind, so NumPy raises its
    ///     UFuncTypeError (NumSharp's house type for it is <see cref="ArgumentException"/>, the text is NumPy's) — even over
    ///     empty edges, where nothing is added.
    /// </summary>
    [TestMethod]
    public void Weights_Bool_RefusesTheInt64CumulativeSum()
    {
        const string text = "Cannot cast ufunc 'add' output from dtype('int64') to dtype('bool') with casting rule 'same_kind'";
        var w = np.ones(new Shape(12), NPTypeCode.Boolean);
        ((Action)(() => np.histogram(np.array(Twelve), Edges(), weights: w))).Should().Throw<ArgumentException>().WithMessage(text);
        ((Action)(() => np.histogram(np.array(Twelve), new[] { -2.0, 0.0, 1.5, 5.0 }, weights: w)))
            .Should().Throw<ArgumentException>().WithMessage(text);
        ((Action)(() => np.histogram(np.array(Twelve), np.array(new double[0]), weights: w)))
            .Should().Throw<ArgumentException>().WithMessage(text);
    }

    /// <summary>
    ///     Without data the loop never runs: the totals stay zeros of the weights' dtype and <c>np.diff</c> of a bool total
    ///     is its <c>not_equal</c> (<c>[False, False, False]</c>); an int8 one is <c>[0, 0, 0]</c> int8.
    /// </summary>
    [TestMethod]
    public void Weights_NoData_ZerosOfTheWeightsDtype()
    {
        var hb = np.histogram(np.array(new double[0]), Edges(), weights: np.ones(new Shape(0), NPTypeCode.Boolean)).hist;
        hb.typecode.Should().Be(NPTypeCode.Boolean);
        hb.ToArray<bool>().Should().Equal(false, false, false);

        var hi = np.histogram(np.array(new double[0]), Edges(), weights: np.ones(new Shape(0), NPTypeCode.SByte)).hist;
        hi.typecode.Should().Be(NPTypeCode.SByte);
        hi.ToArray<sbyte>().Should().Equal((sbyte)0, (sbyte)0, (sbyte)0);
    }

    /// <summary>
    ///     Equal-width bins take NumPy's bincount path, where the weights are cast to the total's dtype per block before the
    ///     add, so bool weights work there (a logical-or total: <c>[True, True, True]</c>) and int8 stays int8.
    /// </summary>
    [TestMethod]
    public void Weights_EqualWidthBins_CastPerBlock()
    {
        var hb = np.histogram(np.array(Twelve), 3, weights: np.ones(new Shape(12), NPTypeCode.Boolean)).hist;
        hb.typecode.Should().Be(NPTypeCode.Boolean);
        hb.ToArray<bool>().Should().Equal(true, true, true);

        var hi = np.histogram(np.array(Twelve), 3, weights: np.ones(new Shape(12), NPTypeCode.SByte)).hist;
        hi.typecode.Should().Be(NPTypeCode.SByte);
        hi.ToArray<sbyte>().Should().Equal((sbyte)4, (sbyte)4, (sbyte)4);
    }

    /// <summary>The density divides the int8 total by float64 widths: <c>[0.125, 1/6, 1/7]</c>.</summary>
    [TestMethod]
    public void Weights_Int8_Density()
    {
        var h = np.histogram(np.array(Twelve), Edges(), density: true, weights: np.ones(new Shape(12), NPTypeCode.SByte)).hist;
        h.typecode.Should().Be(NPTypeCode.Double);
        h.ToArray<double>().Should().Equal(0.125, 2.0 / 12.0, 4.0 / 28.0);
    }
}
